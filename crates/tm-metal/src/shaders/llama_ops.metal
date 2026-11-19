// End-to-end Llama / Qwen3 Metal compute kernels for Apple Silicon.
//
// Copyright (c) 2026 Pengfei Zhang and the tinymodel contributors (MIT OR Apache-2.0).
// Quantization block formats match ggml (Copyright (c) 2023-2026 The ggml authors, MIT).
// GEMV threadgroup geometry inspired by Strata `metal/kernels/gemv.metal` (MIT).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct GemvArgs {
    uint n_in;
    uint n_out;
    uint add_residual; // 0 = overwrite y, 1 = y += W * x
    uint row_bytes;
};

struct GemmArgs {
    uint n_in;
    uint n_out;
    uint n_tok;
    uint add_residual; // 0 = overwrite y, 1 = y += W * x
    uint row_bytes;
};

struct NormArgs {
    uint n;
    float eps;
};

struct RopeKvArgs {
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint rope_dim;
    uint pos;
    uint rope_type; // 0 = Norm (interleaved), 1 = Neox (split-half)
    uint has_freq_factors;
    float freq_base;
};

struct AttnArgs {
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint seq_len;
};

struct FlashAttnArgs {
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint seq_len; // causal length for token 0 in this batch (start_pos + 1)
    uint n_tok;   // number of query tokens in this batch
};

struct QkvGemvArgs {
    uint n_in;
    uint q_rows;
    uint k_rows;
    uint v_rows;
    uint row_bytes;
};

struct EmbedArgs {
    uint token_id;
    uint n_embd;
    uint row_bytes;
};

static inline float read_f16_u(device const uchar* p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

static inline float reduce_lpr8(float v) {
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

// ============================================================================
// 1. Token embedding lookup (Q4_0, Q8_0, Q6_K, F32)
// ============================================================================

kernel void embed_q4_0(
    device const uchar* embd [[buffer(0)]],
    device float* x [[buffer(1)]],
    constant EmbedArgs& a [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint nb = a.n_embd / 32;
    if (tid >= nb) return;
    device const uchar* b = embd + ulong(a.token_id) * ulong(a.row_bytes) + tid * 18;
    const float d = read_f16_u(b);
    device const uchar* qs = b + 2;
    const uint base = tid * 32;
    for (uint j = 0; j < 16; ++j) {
        uchar byte = qs[j];
        x[base + j]      = (float(byte & 0x0F) - 8.0f) * d;
        x[base + j + 16] = (float(byte >> 4) - 8.0f) * d;
    }
}

kernel void embed_q8_0(
    device const uchar* embd [[buffer(0)]],
    device float* x [[buffer(1)]],
    constant EmbedArgs& a [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint nb = a.n_embd / 32;
    if (tid >= nb) return;
    device const uchar* b = embd + ulong(a.token_id) * ulong(a.row_bytes) + tid * 34;
    const float d = read_f16_u(b);
    device const char* qs = (device const char*)(b + 2);
    const uint base = tid * 32;
    for (uint j = 0; j < 32; ++j) {
        x[base + j] = float(qs[j]) * d;
    }
}

kernel void embed_q6_k(
    device const uchar* embd [[buffer(0)]],
    device float* x [[buffer(1)]],
    constant EmbedArgs& a [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint nb = a.n_embd / 256;
    if (tid >= nb) return;
    device const uchar* b = embd + ulong(a.token_id) * ulong(a.row_bytes) + tid * 210;
    device const uchar* ql_all = b;
    device const uchar* qh_all = b + 128;
    device const char*  sc_all = (device const char*)(b + 192);
    const float d = read_f16_u(b + 208);

    const uint base = tid * 256;
    for (uint half_idx = 0; half_idx < 2; ++half_idx) {
        device const uchar* ql = ql_all + half_idx * 64;
        device const uchar* qh = qh_all + half_idx * 32;
        device const char*  sc = sc_all + half_idx * 8;
        device float* out = x + base + half_idx * 128;
        for (uint l = 0; l < 32; ++l) {
            uint is = l / 16;
            int q1 = int((ql[l]      & 0x0F) | ((qh[l] & 3) << 4)) - 32;
            int q2 = int((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = int((ql[l]      >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = int((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l]      = d * float(sc[is])     * float(q1);
            out[l + 32] = d * float(sc[is + 2]) * float(q2);
            out[l + 64] = d * float(sc[is + 4]) * float(q3);
            out[l + 96] = d * float(sc[is + 6]) * float(q4);
        }
    }
}

kernel void embed_f32(
    device const float* embd [[buffer(0)]],
    device float* x [[buffer(1)]],
    constant EmbedArgs& a [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= a.n_embd) return;
    x[tid] = embd[ulong(a.token_id) * ulong(a.n_embd) + tid];
}

// ============================================================================
// 2. RMSNorm (single threadgroup, 256 threads) & per-head QK-Norm
// ============================================================================

kernel void rms_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NormArgs& a [[buffer(3)]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NT = 256;
    threadgroup float partial[8];

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += NT) {
        float v = x[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) {
        partial[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (sg == 0) {
        float v = (lane < 8) ? partial[lane] : 0.0f;
        total = simd_sum(v);
        if (lane == 0) {
            partial[0] = rsqrt(total / float(a.n) + a.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = partial[0];
    for (uint i = tid; i < a.n; i += NT) {
        out[i] = x[i] * scale * w[i];
    }
}

kernel void qk_norm_f32(
    device float* vec [[buffer(0)]],
    device const float* w [[buffer(1)]],
    constant NormArgs& a [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    device float* h = vec + head * a.n;
    float ss = 0.0f;
    for (uint i = lane; i < a.n; i += 32) {
        float v = h[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(a.n) + a.eps);
    for (uint i = lane; i < a.n; i += 32) {
        h[i] = h[i] * scale * w[i];
    }
}

// ============================================================================
// 3. Fused RoPE (Q & K) + F16 KV cache write (K & V)
// ============================================================================

kernel void rope_and_kv_store_f16(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]],
    device half* v_cache [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant RopeKvArgs& a [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint half_rope = a.rope_dim / 2;
    const uint q_pairs = a.n_head * half_rope;
    const uint k_pairs = a.n_head_kv * half_rope;
    const uint kv_dim = a.n_head_kv * a.head_dim;

    if (tid < q_pairs + k_pairs) {
        const bool is_k = (tid >= q_pairs);
        const uint idx = is_k ? (tid - q_pairs) : tid;
        const uint h = idx / half_rope;
        const uint i = idx % half_rope;

        float freq = pow(a.freq_base, -2.0f * float(i) / float(a.rope_dim));
        if (a.has_freq_factors != 0) {
            freq /= freq_factors[i];
        }
        const float theta = float(a.pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);

        device float* head = (is_k ? k : q) + h * a.head_dim;
        uint i0 = (a.rope_type == 0) ? (2 * i) : i;
        uint i1 = (a.rope_type == 0) ? (2 * i + 1) : (i + half_rope);
        float x0 = head[i0];
        float x1 = head[i1];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;
        head[i0] = r0;
        head[i1] = r1;

        if (is_k) {
            const ulong kv_off = ulong(a.pos) * ulong(kv_dim) + h * a.head_dim;
            k_cache[kv_off + i0] = half(r0);
            k_cache[kv_off + i1] = half(r1);
        }
    }

    // Store V (and any unrotated tail of K if rope_dim < head_dim) into F16 KV cache.
    if (tid < kv_dim) {
        const ulong kv_off = ulong(a.pos) * ulong(kv_dim);
        v_cache[kv_off + tid] = half(v[tid]);
        if (a.rope_dim < a.head_dim) {
            uint d = tid % a.head_dim;
            if (d >= a.rope_dim) {
                k_cache[kv_off + tid] = half(k[tid]);
            }
        }
    }
}

// ============================================================================
// 4. Single-token GQA Attention over F16 KV cache (1 threadgroup per query head)
//    128 threads (4 simdgroups) cooperate over sequence length `seq_len`
//    using online softmax so zero sequence-length scratch buffer is needed!
// ============================================================================

kernel void gqa_attn_f16_kv(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttnArgs& a [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint MAX_HD = 128;

    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const float scale = rsqrt(float(a.head_dim));

    device const float* qh = q + head * a.head_dim;
    threadgroup float q_shared[MAX_HD];
    if (tid < a.head_dim) {
        q_shared[tid] = qh[tid] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup handles a strided subset of tokens `t = sg, sg + SG, ...`
    // and lanes within the simdgroup split `head_dim`.
    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // supports head_dim up to 128 (4 * 32)

    const uint n_vec = (a.head_dim + 31) / 32;

    for (uint t = sg; t < a.seq_len; t += SG) {
        const ulong kv_off = ulong(t) * ulong(kv_dim) + kv_h * a.head_dim;
        device const half* kh = k_cache + kv_off;
        device const half* vh = v_cache + kv_off;

        float dot = 0.0f;
        for (uint k = 0; k < n_vec; ++k) {
            uint d = k * 32 + lane;
            if (d < a.head_dim) {
                dot = fma(q_shared[d], float(kh[d]), dot);
            }
        }
        dot = simd_sum(dot);

        float m_new = max(m_local, dot);
        float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
        float p = exp(dot - m_new);
        l_local = l_local * alpha + p;
        m_local = m_new;

        for (uint k = 0; k < n_vec; ++k) {
            uint d = k * 32 + lane;
            if (d < a.head_dim) {
                o_local[k] = o_local[k] * alpha + p * float(vh[d]);
            }
        }
    }

    // Merge the 4 simdgroups in threadgroup memory
    threadgroup float tg_m[SG];
    threadgroup float tg_l[SG];
    threadgroup float tg_o[SG * MAX_HD];

    if (lane == 0) {
        tg_m[sg] = m_local;
        tg_l[sg] = l_local;
    }
    for (uint k = 0; k < n_vec; ++k) {
        uint d = k * 32 + lane;
        if (d < a.head_dim) {
            tg_o[sg * MAX_HD + d] = o_local[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < a.head_dim) {
        float m_all = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
        float denom = 0.0f;
        float acc = 0.0f;
        for (uint s = 0; s < SG; ++s) {
            if (tg_l[s] > 0.0f) {
                float w = exp(tg_m[s] - m_all);
                denom += tg_l[s] * w;
                acc += tg_o[s * MAX_HD + tid] * w;
            }
        }
        out[head * a.head_dim + tid] = acc / denom;
    }
}

// Fast 1-simdgroup (32 threads, zero threadgroup barriers) GQA attention for short contexts (seq_len <= 64).
kernel void gqa_attn_f16_kv_1sg(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttnArgs& a [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const float scale = rsqrt(float(a.head_dim));
    const uint n_vec = (a.head_dim + 31) / 32;

    device const float* qh = q + head * a.head_dim;
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < n_vec; ++k) {
        uint d = k * 32 + lane;
        if (d < a.head_dim) {
            q_reg[k] = qh[d] * scale;
        }
    }

    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint t = 0; t < a.seq_len; ++t) {
        const ulong kv_off = ulong(t) * ulong(kv_dim) + kv_h * a.head_dim;
        device const half* kh = k_cache + kv_off;
        device const half* vh = v_cache + kv_off;

        float dot = 0.0f;
        for (uint k = 0; k < n_vec; ++k) {
            uint d = k * 32 + lane;
            if (d < a.head_dim) {
                dot = fma(q_reg[k], float(kh[d]), dot);
            }
        }
        dot = simd_sum(dot);

        float m_new = max(m_local, dot);
        float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
        float p = exp(dot - m_new);
        l_local = l_local * alpha + p;
        m_local = m_new;

        for (uint k = 0; k < n_vec; ++k) {
            uint d = k * 32 + lane;
            if (d < a.head_dim) {
                o_local[k] = o_local[k] * alpha + p * float(vh[d]);
            }
        }
    }

    const float inv_l = 1.0f / l_local;
    device float* oh = out + head * a.head_dim;
    for (uint k = 0; k < n_vec; ++k) {
        uint d = k * 32 + lane;
        if (d < a.head_dim) {
            oh[d] = o_local[k] * inv_l;
        }
    }
}

// Fused single-dispatch Q/K/V Q4_0 GEMV (`q_rows` and `k_rows` are multiples of 32):
// Eliminates 2 GPU barriers per layer and eliminates tail-wave stalls on small K/V projections.
kernel void gemv_qkv_q4_0(
    device const uchar* Wq [[buffer(0)]],
    device const uchar* Wk [[buffer(1)]],
    device const uchar* Wv [[buffer(2)]],
    device const float* x  [[buffer(3)]],
    device float* q_out    [[buffer(4)]],
    device float* k_out    [[buffer(5)]],
    device float* v_out    [[buffer(6)]],
    constant QkvGemvArgs& a [[buffer(7)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint LPR = 8;
    constexpr uint NR = 2;
    constexpr uint RG = 4;
    constexpr uint SG = 4;

    const uint li = lane % LPR;
    const uint global_row0 = tg_id * (SG * RG * NR) + (sg * RG + lane / LPR) * NR;
    const uint total_rows = a.q_rows + a.k_rows + a.v_rows;
    if (global_row0 >= total_rows) return;

    device const uchar* W;
    device float* y;
    uint local_row0;
    uint mat_rows;
    if (global_row0 < a.q_rows) {
        W = Wq;
        y = q_out;
        local_row0 = global_row0;
        mat_rows = a.q_rows;
    } else if (global_row0 < a.q_rows + a.k_rows) {
        W = Wk;
        y = k_out;
        local_row0 = global_row0 - a.q_rows;
        mat_rows = a.k_rows;
    } else {
        W = Wv;
        y = v_out;
        local_row0 = global_row0 - (a.q_rows + a.k_rows);
        mat_rows = a.v_rows;
    }

    const uint nb = a.n_in / 32;
    const uint r0 = local_row0;
    const uint r1 = min(local_row0 + 1, mat_rows - 1);
    device const uchar* wp0 = W + ulong(r0) * ulong(a.row_bytes);
    device const uchar* wp1 = W + ulong(r1) * ulong(a.row_bytes);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint ib = li; ib < nb; ib += LPR) {
        device const float* xp = x + ib * 32;
        device const uchar* b0 = wp0 + ib * 18;
        device const uchar* b1 = wp1 + ib * 18;
        const float d0 = read_f16_u(b0);
        const float d1 = read_f16_u(b1);
        device const ushort* p0 = (device const ushort*)(b0 + 2);
        device const ushort* p1 = (device const ushort*)(b1 + 2);

        float dot0 = 0.0f, dot1 = 0.0f, sumx = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const float x0 = xp[2 * j];
            const float x1 = xp[2 * j + 1];
            const float x2 = xp[2 * j + 16];
            const float x3 = xp[2 * j + 17];
            sumx += (x0 + x1) + (x2 + x3);

            const ushort v0 = p0[j];
            dot0 = fma(float(v0 & 15), x0, dot0);
            dot0 = fma(float((v0 >> 4) & 15), x2, dot0);
            dot0 = fma(float((v0 >> 8) & 15), x1, dot0);
            dot0 = fma(float(v0 >> 12), x3, dot0);

            const ushort v1 = p1[j];
            dot1 = fma(float(v1 & 15), x0, dot1);
            dot1 = fma(float((v1 >> 4) & 15), x2, dot1);
            dot1 = fma(float((v1 >> 8) & 15), x1, dot1);
            dot1 = fma(float(v1 >> 12), x3, dot1);
        }
        const float corr = -8.0f * sumx;
        acc0 = fma(d0, dot0 + corr, acc0);
        acc1 = fma(d1, dot1 + corr, acc1);
    }

    const float out0 = reduce_lpr8(acc0);
    const float out1 = reduce_lpr8(acc1);
    if (li == 0) {
        y[r0] = out0;
        if (local_row0 + 1 < mat_rows) {
            y[local_row0 + 1] = out1;
        }
    }
}

// GPU-side greedy argmax over `vocab_size` logits (256 threads in 1 threadgroup).
// Ties are broken deterministically in favor of the smaller token index.
kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* out_id [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NT = 256;
    float best_v = -INFINITY;
    uint best_i = 0;
    for (uint i = tid; i < n; i += NT) {
        float v = logits[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    threadgroup float sh_v[NT];
    threadgroup uint  sh_i[NT];
    sh_v[tid] = best_v;
    sh_i[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = NT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float rv = sh_v[tid + stride];
            uint  ri = sh_i[tid + stride];
            float lv = sh_v[tid];
            uint  li = sh_i[tid];
            if (rv > lv || (rv == lv && ri < li)) {
                sh_v[tid] = rv;
                sh_i[tid] = ri;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out_id[0] = sh_i[0];
    }
}

// ============================================================================
// 5. Quantized GEMV kernels (Q4_0, Q4_1, Q8_0, Q6_K) + Fused Gate/Up SwiGLU
//    All support `add_residual` (0 = overwrite y, 1 = y += W * x).
// ============================================================================

kernel void gemv_q4_0_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint LPR = 8;
    constexpr uint NR = 2;
    constexpr uint RG = 4;
    constexpr uint SG = 4;

    const uint li = lane % LPR;
    const uint row0 = tg_id * (SG * RG * NR) + (sg * RG + lane / LPR) * NR;
    const uint nb = a.n_in / 32;
    if (row0 >= a.n_out) return;

    const uint r0 = row0;
    const uint r1 = min(row0 + 1, a.n_out - 1);
    device const uchar* wp0 = W + ulong(r0) * ulong(a.row_bytes);
    device const uchar* wp1 = W + ulong(r1) * ulong(a.row_bytes);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint ib = li; ib < nb; ib += LPR) {
        device const float* xp = x + ib * 32;
        device const uchar* b0 = wp0 + ib * 18;
        device const uchar* b1 = wp1 + ib * 18;
        const float d0 = read_f16_u(b0);
        const float d1 = read_f16_u(b1);
        device const ushort* p0 = (device const ushort*)(b0 + 2);
        device const ushort* p1 = (device const ushort*)(b1 + 2);

        float dot0 = 0.0f, dot1 = 0.0f, sumx = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const float x0 = xp[2 * j];
            const float x1 = xp[2 * j + 1];
            const float x2 = xp[2 * j + 16];
            const float x3 = xp[2 * j + 17];
            sumx += (x0 + x1) + (x2 + x3);

            const ushort v0 = p0[j];
            dot0 = fma(float(v0 & 15), x0, dot0);
            dot0 = fma(float((v0 >> 4) & 15), x2, dot0);
            dot0 = fma(float((v0 >> 8) & 15), x1, dot0);
            dot0 = fma(float(v0 >> 12), x3, dot0);

            const ushort v1 = p1[j];
            dot1 = fma(float(v1 & 15), x0, dot1);
            dot1 = fma(float((v1 >> 4) & 15), x2, dot1);
            dot1 = fma(float((v1 >> 8) & 15), x1, dot1);
            dot1 = fma(float(v1 >> 12), x3, dot1);
        }
        const float corr = -8.0f * sumx;
        acc0 = fma(d0, dot0 + corr, acc0);
        acc1 = fma(d1, dot1 + corr, acc1);
    }

    const float out0 = reduce_lpr8(acc0);
    const float out1 = reduce_lpr8(acc1);
    if (li == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + out0) : out0;
        if (row0 + 1 < a.n_out) {
            y[row0 + 1] = (a.add_residual != 0) ? (y[row0 + 1] + out1) : out1;
        }
    }
}

// Fused FFN Gate + Up + SwiGLU for Q4_0 weights:
// out[r] = silu(dot(W_gate[r], x)) * dot(W_up[r], x) in a SINGLE dispatch!
kernel void gemv_pair_silu_q4_0(
    device const uchar* Wg [[buffer(0)]],
    device const uchar* Wu [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant GemvArgs& a [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint LPR = 8;
    constexpr uint RG = 4;
    constexpr uint SG = 4;

    const uint li = lane % LPR;
    const uint row = tg_id * (SG * RG) + (sg * RG + lane / LPR);
    const uint nb = a.n_in / 32;
    if (row >= a.n_out) return;

    device const uchar* wg = Wg + ulong(row) * ulong(a.row_bytes);
    device const uchar* wu = Wu + ulong(row) * ulong(a.row_bytes);

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (uint ib = li; ib < nb; ib += LPR) {
        device const float* xp = x + ib * 32;
        device const uchar* bg = wg + ib * 18;
        device const uchar* bu = wu + ib * 18;
        const float dg = read_f16_u(bg);
        const float du = read_f16_u(bu);
        device const ushort* pg = (device const ushort*)(bg + 2);
        device const ushort* pu = (device const ushort*)(bu + 2);

        float dotg = 0.0f, dotu = 0.0f, sumx = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const float x0 = xp[2 * j];
            const float x1 = xp[2 * j + 1];
            const float x2 = xp[2 * j + 16];
            const float x3 = xp[2 * j + 17];
            sumx += (x0 + x1) + (x2 + x3);

            const ushort vg = pg[j];
            dotg = fma(float(vg & 15), x0, dotg);
            dotg = fma(float((vg >> 4) & 15), x2, dotg);
            dotg = fma(float((vg >> 8) & 15), x1, dotg);
            dotg = fma(float(vg >> 12), x3, dotg);

            const ushort vu = pu[j];
            dotu = fma(float(vu & 15), x0, dotu);
            dotu = fma(float((vu >> 4) & 15), x2, dotu);
            dotu = fma(float((vu >> 8) & 15), x1, dotu);
            dotu = fma(float(vu >> 12), x3, dotu);
        }
        const float corr = -8.0f * sumx;
        acc_g = fma(dg, dotg + corr, acc_g);
        acc_u = fma(du, dotu + corr, acc_u);
    }

    const float g = reduce_lpr8(acc_g);
    const float u = reduce_lpr8(acc_u);
    if (li == 0) {
        y[row] = (g / (1.0f + exp(-g))) * u;
    }
}

kernel void gemv_q4_1_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint LPR = 8;
    constexpr uint NR = 2;
    constexpr uint RG = 4;
    constexpr uint SG = 4;

    const uint li = lane % LPR;
    const uint row0 = tg_id * (SG * RG * NR) + (sg * RG + lane / LPR) * NR;
    const uint nb = a.n_in / 32;
    if (row0 >= a.n_out) return;

    const uint r0 = row0;
    const uint r1 = min(row0 + 1, a.n_out - 1);
    device const uchar* wp0 = W + ulong(r0) * ulong(a.row_bytes);
    device const uchar* wp1 = W + ulong(r1) * ulong(a.row_bytes);

    float acc0 = 0.0f, acc1 = 0.0f;

    for (uint ib = li; ib < nb; ib += LPR) {
        device const float* xp = x + ib * 32;
        device const uchar* b0 = wp0 + ib * 20;
        device const uchar* b1 = wp1 + ib * 20;
        const float d0 = read_f16_u(b0);
        const float m0 = read_f16_u(b0 + 2);
        const float d1 = read_f16_u(b1);
        const float m1 = read_f16_u(b1 + 2);
        device const ushort* p0 = (device const ushort*)(b0 + 4);
        device const ushort* p1 = (device const ushort*)(b1 + 4);

        float dot0 = 0.0f, dot1 = 0.0f, sumx = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const float x0 = xp[2 * j];
            const float x1 = xp[2 * j + 1];
            const float x2 = xp[2 * j + 16];
            const float x3 = xp[2 * j + 17];
            sumx += (x0 + x1) + (x2 + x3);

            const ushort v0 = p0[j];
            dot0 = fma(float(v0 & 15), x0, dot0);
            dot0 = fma(float((v0 >> 4) & 15), x2, dot0);
            dot0 = fma(float((v0 >> 8) & 15), x1, dot0);
            dot0 = fma(float(v0 >> 12), x3, dot0);

            const ushort v1 = p1[j];
            dot1 = fma(float(v1 & 15), x0, dot1);
            dot1 = fma(float((v1 >> 4) & 15), x2, dot1);
            dot1 = fma(float((v1 >> 8) & 15), x1, dot1);
            dot1 = fma(float(v1 >> 12), x3, dot1);
        }
        acc0 = fma(d0, dot0, fma(m0, sumx, acc0));
        acc1 = fma(d1, dot1, fma(m1, sumx, acc1));
    }

    const float out0 = reduce_lpr8(acc0);
    const float out1 = reduce_lpr8(acc1);
    if (li == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + out0) : out0;
        if (row0 + 1 < a.n_out) {
            y[row0 + 1] = (a.add_residual != 0) ? (y[row0 + 1] + out1) : out1;
        }
    }
}

// Coalesced Q8_0 GEMV: 32 lanes cooperate on each 32-element block (`lane` = element index),
// processing NR = 4 consecutive rows per simdgroup (SG = 4 -> 16 rows / threadgroup).
kernel void gemv_q8_0_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR = 4;
    constexpr uint SG = 4;

    const uint row0 = (tg_id * SG + sg) * NR;
    const uint nb = a.n_in / 32;
    if (row0 >= a.n_out) return;

    const uint r0 = row0;
    const uint r1 = min(row0 + 1, a.n_out - 1);
    const uint r2 = min(row0 + 2, a.n_out - 1);
    const uint r3 = min(row0 + 3, a.n_out - 1);
    device const uchar* wp0 = W + ulong(r0) * ulong(a.row_bytes);
    device const uchar* wp1 = W + ulong(r1) * ulong(a.row_bytes);
    device const uchar* wp2 = W + ulong(r2) * ulong(a.row_bytes);
    device const uchar* wp3 = W + ulong(r3) * ulong(a.row_bytes);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint ib = 0; ib < nb; ++ib) {
        const float xv = x[ib * 32 + lane];
        device const uchar* b0 = wp0 + ib * 34;
        device const uchar* b1 = wp1 + ib * 34;
        device const uchar* b2 = wp2 + ib * 34;
        device const uchar* b3 = wp3 + ib * 34;

        const float d0 = read_f16_u(b0);
        const float d1 = read_f16_u(b1);
        const float d2 = read_f16_u(b2);
        const float d3 = read_f16_u(b3);

        acc0 = fma(d0 * float(((device const char*)(b0 + 2))[lane]), xv, acc0);
        acc1 = fma(d1 * float(((device const char*)(b1 + 2))[lane]), xv, acc1);
        acc2 = fma(d2 * float(((device const char*)(b2 + 2))[lane]), xv, acc2);
        acc3 = fma(d3 * float(((device const char*)(b3 + 2))[lane]), xv, acc3);
    }

    const float out0 = simd_sum(acc0);
    const float out1 = simd_sum(acc1);
    const float out2 = simd_sum(acc2);
    const float out3 = simd_sum(acc3);
    if (lane == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + out0) : out0;
        if (row0 + 1 < a.n_out) y[r1] = (a.add_residual != 0) ? (y[r1] + out1) : out1;
        if (row0 + 2 < a.n_out) y[r2] = (a.add_residual != 0) ? (y[r2] + out2) : out2;
        if (row0 + 3 < a.n_out) y[r3] = (a.add_residual != 0) ? (y[r3] + out3) : out3;
    }
}

// Coalesced Q6_K GEMV: 32 lanes cooperate on each 256-element superblock (`lane` = `l` in 0..31),
// processing NR = 4 consecutive rows per simdgroup (SG = 4 -> 16 rows / threadgroup).
// Every ql/qh/x load is 100% contiguous across the 32-lane simdgroup.
kernel void gemv_q6_k_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR = 4;
    constexpr uint SG = 4;

    const uint row0 = (tg_id * SG + sg) * NR;
    const uint nb = a.n_in / 256;
    if (row0 >= a.n_out) return;

    const uint r0 = row0;
    const uint r1 = min(row0 + 1, a.n_out - 1);
    const uint r2 = min(row0 + 2, a.n_out - 1);
    const uint r3 = min(row0 + 3, a.n_out - 1);
    device const uchar* wp[NR] = {
        W + ulong(r0) * ulong(a.row_bytes),
        W + ulong(r1) * ulong(a.row_bytes),
        W + ulong(r2) * ulong(a.row_bytes),
        W + ulong(r3) * ulong(a.row_bytes)
    };

    float acc[NR] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint is = lane >> 4; // 0 for lanes 0..15, 1 for lanes 16..31

    for (uint ib = 0; ib < nb; ++ib) {
        device const float* xp = x + ib * 256;
        const ulong blk_off = ulong(ib) * 210;

        float d[NR];
        float sum[NR] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (uint k = 0; k < NR; ++k) {
            d[k] = read_f16_u(wp[k] + blk_off + 208);
        }

        #pragma unroll
        for (uint half_idx = 0; half_idx < 2; ++half_idx) {
            device const float* xh = xp + half_idx * 128;
            const float x_a = xh[lane];
            const float x_b = xh[lane + 32];
            const float x_c = xh[lane + 64];
            const float x_d = xh[lane + 96];

            #pragma unroll
            for (uint k = 0; k < NR; ++k) {
                device const uchar* b = wp[k] + blk_off;
                device const uchar* ql = b + half_idx * 64;
                device const uchar* qh = b + 128 + half_idx * 32;
                device const char*  sc = (device const char*)(b + 192 + half_idx * 8);

                const uchar ql_lo = ql[lane];
                const uchar ql_hi = ql[lane + 32];
                const uchar qh_v  = qh[lane];

                const int qa = int((ql_lo & 0x0F) | ((qh_v & 3) << 4)) - 32;
                const int qb = int((ql_hi & 0x0F) | (((qh_v >> 2) & 3) << 4)) - 32;
                const int qc = int((ql_lo >> 4)   | (((qh_v >> 4) & 3) << 4)) - 32;
                const int qd = int((ql_hi >> 4)   | (((qh_v >> 6) & 3) << 4)) - 32;

                sum[k] = fma(float(sc[is])     * float(qa), x_a, sum[k]);
                sum[k] = fma(float(sc[is + 2]) * float(qb), x_b, sum[k]);
                sum[k] = fma(float(sc[is + 4]) * float(qc), x_c, sum[k]);
                sum[k] = fma(float(sc[is + 6]) * float(qd), x_d, sum[k]);
            }
        }

        #pragma unroll
        for (uint k = 0; k < NR; ++k) {
            acc[k] = fma(d[k], sum[k], acc[k]);
        }
    }

    const float out0 = simd_sum(acc[0]);
    const float out1 = simd_sum(acc[1]);
    const float out2 = simd_sum(acc[2]);
    const float out3 = simd_sum(acc[3]);
    if (lane == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + out0) : out0;
        if (row0 + 1 < a.n_out) y[r1] = (a.add_residual != 0) ? (y[r1] + out1) : out1;
        if (row0 + 2 < a.n_out) y[r2] = (a.add_residual != 0) ? (y[r2] + out2) : out2;
        if (row0 + 3 < a.n_out) y[r3] = (a.add_residual != 0) ? (y[r3] + out3) : out3;
    }
}


// ============================================================================
// 6. Batched Prefill kernels (B tokens per dispatch, token-major [B, dim])
//    Every GEMM kernel uses the exact same per-token accumulation & reduction
//    order as its single-token GEMV counterpart -> bit-identical results.
// ============================================================================

kernel void embed_q4_0_batch(
    device const uchar* embd [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant EmbedArgs& a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint nb = a.n_embd / 32;
    const uint tid = gid.x;
    const uint t = gid.y;
    if (tid >= nb) return;
    const uint tok = token_ids[t];
    device const uchar* b = embd + ulong(tok) * ulong(a.row_bytes) + tid * 18;
    const float d = read_f16_u(b);
    device const uchar* qs = b + 2;
    device float* xt = x + ulong(t) * ulong(a.n_embd) + tid * 32;
    for (uint j = 0; j < 16; ++j) {
        uchar byte = qs[j];
        xt[j]      = (float(byte & 0x0F) - 8.0f) * d;
        xt[j + 16] = (float(byte >> 4) - 8.0f) * d;
    }
}

kernel void embed_q8_0_batch(
    device const uchar* embd [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant EmbedArgs& a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint nb = a.n_embd / 32;
    const uint tid = gid.x;
    const uint t = gid.y;
    if (tid >= nb) return;
    const uint tok = token_ids[t];
    device const uchar* b = embd + ulong(tok) * ulong(a.row_bytes) + tid * 34;
    const float d = read_f16_u(b);
    device const char* qs = (device const char*)(b + 2);
    device float* xt = x + ulong(t) * ulong(a.n_embd) + tid * 32;
    for (uint j = 0; j < 32; ++j) {
        xt[j] = float(qs[j]) * d;
    }
}

kernel void embed_q6_k_batch(
    device const uchar* embd [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant EmbedArgs& a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint nb = a.n_embd / 256;
    const uint tid = gid.x;
    const uint t = gid.y;
    if (tid >= nb) return;
    const uint tok = token_ids[t];
    device const uchar* b = embd + ulong(tok) * ulong(a.row_bytes) + tid * 210;
    const float d = read_f16_u(b + 208);
    device float* xt = x + ulong(t) * ulong(a.n_embd) + tid * 256;
    for (uint half_idx = 0; half_idx < 2; ++half_idx) {
        device const uchar* ql = b + half_idx * 64;
        device const uchar* qh = b + 128 + half_idx * 32;
        device const char*  sc = (device const char*)(b + 192 + half_idx * 8);
        device float* out = xt + half_idx * 128;
        for (uint l = 0; l < 32; ++l) {
            const uint is = l / 16;
            const int q1 = int((ql[l]      & 0x0F) | ((qh[l] & 3) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = int((ql[l]      >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l]      = d * float(sc[is])     * float(q1);
            out[l + 32] = d * float(sc[is + 2]) * float(q2);
            out[l + 64] = d * float(sc[is + 4]) * float(q3);
            out[l + 96] = d * float(sc[is + 6]) * float(q4);
        }
    }
}

kernel void embed_f32_batch(
    device const float* embd [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* x [[buffer(2)]],
    constant EmbedArgs& a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint tid = gid.x;
    const uint t = gid.y;
    if (tid >= a.n_embd) return;
    const uint tok = token_ids[t];
    x[ulong(t) * ulong(a.n_embd) + tid] = embd[ulong(tok) * ulong(a.n_embd) + tid];
}

kernel void rms_norm_f32_batch(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NormArgs& a [[buffer(3)]],
    uint t [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NT = 256;
    device const float* xt = x + ulong(t) * ulong(a.n);
    device float* ot = out + ulong(t) * ulong(a.n);

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += NT) {
        float v = xt[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    threadgroup float sg_sums[8];
    if (lane == 0) sg_sums[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (sg == 0) {
        total = (lane < 8) ? sg_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) sg_sums[0] = rsqrt(total / float(a.n) + a.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = sg_sums[0];
    for (uint i = tid; i < a.n; i += NT) {
        ot[i] = xt[i] * scale * w[i];
    }
}

kernel void qk_norm_f32_batch(
    device float* data [[buffer(0)]],
    device const float* w [[buffer(1)]],
    constant NormArgs& a [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 n_tgs [[threadgroups_per_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    device float* h = data + (ulong(tg.y) * ulong(n_tgs.x) + ulong(tg.x)) * ulong(a.n);
    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += 32) {
        float v = h[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(a.n) + a.eps);
    for (uint i = tid; i < a.n; i += 32) {
        h[i] = h[i] * scale * w[i];
    }
}

kernel void rope_and_kv_store_f16_batch(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]],
    device half* v_cache [[buffer(4)]],
    device const float* freq_factors [[buffer(5)]],
    constant RopeKvArgs& a [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint tid = gid.x;
    const uint t = gid.y;
    const uint pos = a.pos + t;
    const uint half_rope = a.rope_dim / 2;
    const uint q_dim = a.n_head * a.head_dim;
    const uint kv_dim = a.n_head_kv * a.head_dim;

    device float* qt = q + ulong(t) * ulong(q_dim);
    device float* kt = k + ulong(t) * ulong(kv_dim);
    device const float* vt = v + ulong(t) * ulong(kv_dim);

    const uint total_q_pairs = a.n_head * half_rope;
    const uint total_k_pairs = a.n_head_kv * half_rope;
    const uint total_pairs = total_q_pairs + total_k_pairs;

    if (tid < total_pairs) {
        const bool is_k = (tid >= total_q_pairs);
        const uint idx = is_k ? (tid - total_q_pairs) : tid;
        const uint h = idx / half_rope;
        const uint i = idx % half_rope;

        float freq = pow(a.freq_base, -2.0f * float(i) / float(a.rope_dim));
        if (a.has_freq_factors != 0) {
            freq /= freq_factors[i];
        }
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);

        device float* head = (is_k ? kt : qt) + h * a.head_dim;
        uint i0 = (a.rope_type == 0) ? (2 * i) : i;
        uint i1 = (a.rope_type == 0) ? (2 * i + 1) : (i + half_rope);
        float x0 = head[i0];
        float x1 = head[i1];
        float r0 = x0 * c - x1 * s;
        float r1 = x0 * s + x1 * c;
        head[i0] = r0;
        head[i1] = r1;

        if (is_k) {
            const ulong kv_off = ulong(pos) * ulong(kv_dim) + h * a.head_dim;
            k_cache[kv_off + i0] = half(r0);
            k_cache[kv_off + i1] = half(r1);
        }
    }

    if (tid < kv_dim) {
        const ulong kv_off = ulong(pos) * ulong(kv_dim);
        v_cache[kv_off + tid] = half(vt[tid]);
        if (a.rope_dim < a.head_dim) {
            uint d = tid % a.head_dim;
            if (d >= a.rope_dim) {
                k_cache[kv_off + tid] = half(kt[tid]);
            }
        }
    }
}

kernel void gqa_attn_f16_kv_batch(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttnArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint MAX_HD = 128;

    const uint head = tg.x;
    const uint tok = tg.y;
    const uint seq_len = a.seq_len + tok; // causal: start_pos + 1 + tok
    const uint q_dim = a.n_head * a.head_dim;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const float scale = rsqrt(float(a.head_dim));

    device const float* qh = q + ulong(tok) * ulong(q_dim) + head * a.head_dim;
    threadgroup float q_shared[MAX_HD];
    if (tid < a.head_dim) {
        q_shared[tid] = qh[tid] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n_vec = (a.head_dim + 31) / 32;

    for (uint t = sg; t < seq_len; t += SG) {
        const ulong kv_off = ulong(t) * ulong(kv_dim) + kv_h * a.head_dim;
        device const half* kh = k_cache + kv_off;
        device const half* vh = v_cache + kv_off;

        float dot = 0.0f;
        for (uint k = 0; k < n_vec; ++k) {
            uint d = k * 32 + lane;
            if (d < a.head_dim) {
                dot = fma(q_shared[d], float(kh[d]), dot);
            }
        }
        dot = simd_sum(dot);

        float m_new = max(m_local, dot);
        float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
        float p = exp(dot - m_new);
        l_local = l_local * alpha + p;
        m_local = m_new;

        for (uint k = 0; k < n_vec; ++k) {
            uint d = k * 32 + lane;
            if (d < a.head_dim) {
                o_local[k] = o_local[k] * alpha + p * float(vh[d]);
            }
        }
    }

    threadgroup float tg_m[SG];
    threadgroup float tg_l[SG];
    threadgroup float tg_o[SG * MAX_HD];

    if (lane == 0) {
        tg_m[sg] = m_local;
        tg_l[sg] = l_local;
    }
    for (uint k = 0; k < n_vec; ++k) {
        uint d = k * 32 + lane;
        if (d < a.head_dim) {
            tg_o[sg * MAX_HD + d] = o_local[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < a.head_dim) {
        float m_all = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
        float denom = 0.0f;
        float acc = 0.0f;
        for (uint s = 0; s < SG; ++s) {
            if (tg_l[s] > 0.0f) {
                float w = exp(tg_m[s] - m_all);
                denom += tg_l[s] * w;
                acc += tg_o[s * MAX_HD + tid] * w;
            }
        }
        out[ulong(tok) * ulong(q_dim) + head * a.head_dim + tid] = acc / denom;
    }
}

// Hardware-accelerated Tiled Causal FlashAttention (`simdgroup_mma`) for batched prefill:
// 1 threadgroup (128 threads = 4 simdgroups) computes BQ = 32 query tokens (8 queries/simdgroup)
// for one query head, streaming K/V in tiles of BK = 16 with FP32 online softmax.
// Cuts KV DRAM reads by 32x and runs Q*K^T, row-rescaling, and P*V on Apple GPU tensor cores.
kernel void fa_gqa_f16_kv_batch(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant FlashAttnArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint MAX_HD = 128;
    constexpr uint BQ = 32;
    constexpr uint BK = 16;

    const uint head = tg.x;
    const uint t0 = tg.y * BQ;
    if (t0 >= a.n_tok) return;

    const uint q_dim = a.n_head * a.head_dim;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const float scale = rsqrt(float(a.head_dim));
    const uint n_d8 = a.head_dim / 8;

    // 12 KB total threadgroup memory -> high occupancy on Apple M4
    threadgroup half  K_T_sh[MAX_HD][BK];
    threadgroup half  V_sh[BK][MAX_HD];
    threadgroup float S_sh[BQ][BK];
    threadgroup half  P_sh[BQ][BK];
    threadgroup float sg_stage[4][8][8];

    const uint r = lane >> 2;   // 0..7 (query row within this simdgroup)
    const uint sub = lane & 3;  // 0..3 (4 lanes cooperate on each query row)
    const uint q_tok = t0 + sg * 8 + r;
    const bool valid_q = (q_tok < a.n_tok);

    // Load Q (8 x head_dim per simdgroup) scaled by 1/sqrt(head_dim) into registers ONCE
    simdgroup_half8x8 q_mat[16];
    simdgroup_float8x8 o_acc[16];
    for (uint d8 = 0; d8 < n_d8; ++d8) {
        const uint d_col0 = d8 * 8 + sub * 2;
        if (valid_q) {
            device const float* qp = q + ulong(q_tok) * ulong(q_dim) + head * a.head_dim + d_col0;
            P_sh[sg * 8 + r][sub * 2 + 0] = half(qp[0] * scale);
            P_sh[sg * 8 + r][sub * 2 + 1] = half(qp[1] * scale);
        } else {
            P_sh[sg * 8 + r][sub * 2 + 0] = half(0.0f);
            P_sh[sg * 8 + r][sub * 2 + 1] = half(0.0f);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(q_mat[d8], &P_sh[sg * 8][0], BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        o_acc[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const uint last_tok_in_tg = min(t0 + BQ - 1, a.n_tok - 1);
    const uint max_kv = a.seq_len + last_tok_in_tg;
    const uint causal_limit = valid_q ? (a.seq_len + q_tok) : 0;

    float m_row = -INFINITY;
    float l_row = 0.0f;

    const uint k_loc = tid >> 3; // 0..15
    const uint d_sub = tid & 7;  // 0..7

    for (uint kt = 0; kt < max_kv; kt += BK) {
        // 1. Cooperatively load K_T[head_dim][16] and V[16][head_dim] into threadgroup memory
        if (kt + k_loc < max_kv) {
            const ulong kv_off = ulong(kt + k_loc) * ulong(kv_dim) + kv_h * a.head_dim;
            device const half* kh = k_cache + kv_off;
            device const half* vh = v_cache + kv_off;
            for (uint d = d_sub; d < a.head_dim; d += 8) {
                K_T_sh[d][k_loc] = kh[d];
                V_sh[k_loc][d]   = vh[d];
            }
        } else {
            for (uint d = d_sub; d < a.head_dim; d += 8) {
                K_T_sh[d][k_loc] = half(0.0f);
                V_sh[k_loc][d]   = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Hardware MMA: S (8 x 16) = Q_sg (8 x head_dim) * K_T (head_dim x 16)
        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d8 = 0; d8 < n_d8; ++d8) {
            simdgroup_half8x8 kt0, kt1;
            simdgroup_load(kt0, &K_T_sh[d8 * 8][0], BK);
            simdgroup_load(kt1, &K_T_sh[d8 * 8][8], BK);
            simdgroup_multiply_accumulate(s0, q_mat[d8], kt0, s0);
            simdgroup_multiply_accumulate(s1, q_mat[d8], kt1, s1);
        }
        simdgroup_store(s0, &S_sh[sg * 8][0], BK);
        simdgroup_store(s1, &S_sh[sg * 8][8], BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Causal mask + FP32 online softmax across 16 keys (4 lanes per query row)
        const uint c_base = sub * 4;
        float sc0 = (kt + c_base + 0 < causal_limit) ? S_sh[sg * 8 + r][c_base + 0] : -INFINITY;
        float sc1 = (kt + c_base + 1 < causal_limit) ? S_sh[sg * 8 + r][c_base + 1] : -INFINITY;
        float sc2 = (kt + c_base + 2 < causal_limit) ? S_sh[sg * 8 + r][c_base + 2] : -INFINITY;
        float sc3 = (kt + c_base + 3 < causal_limit) ? S_sh[sg * 8 + r][c_base + 3] : -INFINITY;

        float row_max = max(max(sc0, sc1), max(sc2, sc3));
        row_max = max(row_max, simd_shuffle_xor(row_max, 1));
        row_max = max(row_max, simd_shuffle_xor(row_max, 2));

        const float m_new = max(m_row, row_max);
        const float alpha = (m_row == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(m_row - m_new);

        const float p0_f = (sc0 == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(sc0 - m_new);
        const float p1_f = (sc1 == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(sc1 - m_new);
        const float p2_f = (sc2 == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(sc2 - m_new);
        const float p3_f = (sc3 == -INFINITY || m_new == -INFINITY) ? 0.0f : exp(sc3 - m_new);

        float row_sum = (p0_f + p1_f) + (p2_f + p3_f);
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 2);

        l_row = l_row * alpha + row_sum;
        m_row = m_new;

        P_sh[sg * 8 + r][c_base + 0] = half(p0_f);
        P_sh[sg * 8 + r][c_base + 1] = half(p1_f);
        P_sh[sg * 8 + r][c_base + 2] = half(p2_f);
        P_sh[sg * 8 + r][c_base + 3] = half(p3_f);

        // Build 8x8 diagonal matrix diag(alpha) in sg_stage[sg] for 1-instruction row rescaling
        sg_stage[sg][r][sub * 2 + 0] = 0.0f;
        sg_stage[sg][r][sub * 2 + 1] = 0.0f;
        if ((r >> 1) == sub) {
            sg_stage[sg][r][r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Hardware MMA: O_acc = diag(alpha) * O_acc + P (8 x 16) * V (16 x head_dim)
        simdgroup_float8x8 alpha_mat;
        simdgroup_load(alpha_mat, &sg_stage[sg][0][0], 8);
        simdgroup_half8x8 pm0, pm1;
        simdgroup_load(pm0, &P_sh[sg * 8][0], BK);
        simdgroup_load(pm1, &P_sh[sg * 8][8], BK);

        for (uint d8 = 0; d8 < n_d8; ++d8) {
            simdgroup_multiply(o_acc[d8], alpha_mat, o_acc[d8]);
            simdgroup_half8x8 vm0, vm1;
            simdgroup_load(vm0, &V_sh[0][d8 * 8], MAX_HD);
            simdgroup_load(vm1, &V_sh[8][d8 * 8], MAX_HD);
            simdgroup_multiply_accumulate(o_acc[d8], pm0, vm0, o_acc[d8]);
            simdgroup_multiply_accumulate(o_acc[d8], pm1, vm1, o_acc[d8]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Normalize by 1 / l_row via diag(inv_l) * O_acc and store to device memory
    const float inv_l = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
    sg_stage[sg][r][sub * 2 + 0] = 0.0f;
    sg_stage[sg][r][sub * 2 + 1] = 0.0f;
    if ((r >> 1) == sub) {
        sg_stage[sg][r][r] = inv_l;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 inv_mat;
    simdgroup_load(inv_mat, &sg_stage[sg][0][0], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d8 = 0; d8 < n_d8; ++d8) {
        simdgroup_multiply(o_acc[d8], inv_mat, o_acc[d8]);
        simdgroup_store(o_acc[d8], &sg_stage[sg][0][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (valid_q) {
            device float* op = out + ulong(q_tok) * ulong(q_dim) + head * a.head_dim + d8 * 8 + sub * 2;
            op[0] = sg_stage[sg][r][sub * 2 + 0];
            op[1] = sg_stage[sg][r][sub * 2 + 1];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void silu_mul_f32(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    const float g = gate[tid];
    gate[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

// Hardware-accelerated simdgroup_float8x8 tiled Q4_0 GEMM:
// 32 rows (n_out) x 32 tokens (n_tok) per threadgroup, BK = 32 (1 BlockQ4_0 per step).
// 128 threads = 4 simdgroups (2x2 over the 32x32 output tile, 16x16 per simdgroup).
kernel void gemm_q4_0_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint r0 = tg_id.x * 32;
    const uint t0 = tg_id.y * 32;
    if (r0 >= a.n_out || t0 >= a.n_tok) return;

    threadgroup half  X_tile[32][32];
    threadgroup half  W_tile[32][32];
    threadgroup float O_tile[32][32];

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m = (sg >> 1) * 16;
    const uint sg_n = (sg & 1) * 16;

    const uint loc_row = tid >> 2;      // 0..31
    const uint loc_grp = (tid & 3) * 8; // 0, 8, 16, 24
    const uint q_grp   = (tid & 3) * 4; // 0, 4, 8, 12

    const uint nb = a.n_in / 32;
    const bool valid_t = (t0 + loc_row) < a.n_tok;
    const bool valid_r = (r0 + loc_row) < a.n_out;
    device const float* x_row = valid_t ? (x + ulong(t0 + loc_row) * ulong(a.n_in)) : x;
    device const uchar* w_row = valid_r ? (W + ulong(r0 + loc_row) * ulong(a.row_bytes)) : W;

    for (uint ib = 0; ib < nb; ++ib) {
        // Cooperative load of X_tile[32][32] as half (8 elements / thread)
        if (valid_t) {
            device const float4* xp4 = (device const float4*)(x_row + ib * 32 + loc_grp);
            float4 v0 = xp4[0];
            float4 v1 = xp4[1];
            X_tile[loc_row][loc_grp + 0] = half(v0.x);
            X_tile[loc_row][loc_grp + 1] = half(v0.y);
            X_tile[loc_row][loc_grp + 2] = half(v0.z);
            X_tile[loc_row][loc_grp + 3] = half(v0.w);
            X_tile[loc_row][loc_grp + 4] = half(v1.x);
            X_tile[loc_row][loc_grp + 5] = half(v1.y);
            X_tile[loc_row][loc_grp + 6] = half(v1.z);
            X_tile[loc_row][loc_grp + 7] = half(v1.w);
        } else {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                X_tile[loc_row][loc_grp + j] = half(0.0f);
            }
        }

        // Cooperative dequant of W_tile[32][32] transposed as half: W_tile[k][r_loc]
        if (valid_r) {
            device const uchar* b = w_row + ib * 18;
            const ushort d_bits = ushort(b[0]) | (ushort(b[1]) << 8);
            const half d = as_type<half>(d_bits);
            device const uchar* qs = b + 2 + q_grp;
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                uchar byte = qs[j];
                W_tile[q_grp + j][loc_row]      = (half(byte & 0x0F) - half(8.0f)) * d;
                W_tile[16 + q_grp + j][loc_row] = (half(byte >> 4)   - half(8.0f)) * d;
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                W_tile[q_grp + j][loc_row]      = half(0.0f);
                W_tile[16 + q_grp + j][loc_row] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &X_tile[sg_m][kk], 32);
            simdgroup_load(a1, &X_tile[sg_m + 8][kk], 32);
            simdgroup_load(b0, &W_tile[kk][sg_n], 32);
            simdgroup_load(b1, &W_tile[kk][sg_n + 8], 32);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage FP32 output via O_tile[32][32] and write back with exact bounds check
    simdgroup_store(acc00, &O_tile[sg_m][sg_n], 32);
    simdgroup_store(acc01, &O_tile[sg_m][sg_n + 8], 32);
    simdgroup_store(acc10, &O_tile[sg_m + 8][sg_n], 32);
    simdgroup_store(acc11, &O_tile[sg_m + 8][sg_n + 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_t) {
        const ulong y_base = ulong(t0 + loc_row) * ulong(a.n_out) + r0 + loc_grp;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            if (r0 + loc_grp + j < a.n_out) {
                const float out = O_tile[loc_row][loc_grp + j];
                y[y_base + j] = (a.add_residual != 0) ? (y[y_base + j] + out) : out;
            }
        }
    }
}

// Hardware-accelerated simdgroup_float8x8 tiled Q4_1 GEMM (32x32x32).
kernel void gemm_q4_1_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint r0 = tg_id.x * 32;
    const uint t0 = tg_id.y * 32;
    if (r0 >= a.n_out || t0 >= a.n_tok) return;

    threadgroup float X_tile[32][32];
    threadgroup float W_tile[32][32];

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m = (sg >> 1) * 16;
    const uint sg_n = (sg & 1) * 16;

    const uint loc_row = tid >> 2;
    const uint loc_grp = (tid & 3) * 8;
    const uint q_grp   = (tid & 3) * 4;

    const uint nb = a.n_in / 32;
    const bool valid_t = (t0 + loc_row) < a.n_tok;
    const bool valid_r = (r0 + loc_row) < a.n_out;
    device const float* x_row = valid_t ? (x + ulong(t0 + loc_row) * ulong(a.n_in)) : x;
    device const uchar* w_row = valid_r ? (W + ulong(r0 + loc_row) * ulong(a.row_bytes)) : W;

    for (uint ib = 0; ib < nb; ++ib) {
        if (valid_t) {
            device const float4* xp4 = (device const float4*)(x_row + ib * 32 + loc_grp);
            float4 v0 = xp4[0];
            float4 v1 = xp4[1];
            X_tile[loc_row][loc_grp + 0] = v0.x;
            X_tile[loc_row][loc_grp + 1] = v0.y;
            X_tile[loc_row][loc_grp + 2] = v0.z;
            X_tile[loc_row][loc_grp + 3] = v0.w;
            X_tile[loc_row][loc_grp + 4] = v1.x;
            X_tile[loc_row][loc_grp + 5] = v1.y;
            X_tile[loc_row][loc_grp + 6] = v1.z;
            X_tile[loc_row][loc_grp + 7] = v1.w;
        } else {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                X_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }

        if (valid_r) {
            device const uchar* b = w_row + ib * 20;
            const float d = read_f16_u(b);
            const float m = read_f16_u(b + 2);
            device const uchar* qs = b + 4 + q_grp;
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                uchar byte = qs[j];
                W_tile[q_grp + j][loc_row]      = fma(float(byte & 0x0F), d, m);
                W_tile[16 + q_grp + j][loc_row] = fma(float(byte >> 4),   d, m);
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 4; ++j) {
                W_tile[q_grp + j][loc_row]      = 0.0f;
                W_tile[16 + q_grp + j][loc_row] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, &X_tile[sg_m][kk], 32);
            simdgroup_load(a1, &X_tile[sg_m + 8][kk], 32);
            simdgroup_load(b0, &W_tile[kk][sg_n], 32);
            simdgroup_load(b1, &W_tile[kk][sg_n + 8], 32);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, &X_tile[sg_m][sg_n], 32);
    simdgroup_store(acc01, &X_tile[sg_m][sg_n + 8], 32);
    simdgroup_store(acc10, &X_tile[sg_m + 8][sg_n], 32);
    simdgroup_store(acc11, &X_tile[sg_m + 8][sg_n + 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_t) {
        const ulong y_base = ulong(t0 + loc_row) * ulong(a.n_out) + r0 + loc_grp;
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            if (r0 + loc_grp + j < a.n_out) {
                const float out = X_tile[loc_row][loc_grp + j];
                y[y_base + j] = (a.add_residual != 0) ? (y[y_base + j] + out) : out;
            }
        }
    }
}

// Batched Coalesced Q8_0 GEMM: 4 rows/TG (SG=4, 32 lanes/row) x BT=8 tokens/TG.
kernel void gemm_q8_0_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint BT = 8;

    const uint row = tg_id.x * SG + sg;
    const uint t0 = tg_id.y * BT;
    if (row >= a.n_out || t0 >= a.n_tok) return;

    const uint nb = a.n_in / 32;
    device const uchar* wp = W + ulong(row) * ulong(a.row_bytes);

    ulong x_off[BT];
    #pragma unroll
    for (uint k = 0; k < BT; ++k) {
        x_off[k] = ulong(min(t0 + k, a.n_tok - 1)) * ulong(a.n_in);
    }

    float acc[BT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint ib = 0; ib < nb; ++ib) {
        device const uchar* b = wp + ib * 34;
        const float wv = read_f16_u(b) * float(((device const char*)(b + 2))[lane]);
        const uint xi = ib * 32 + lane;
        #pragma unroll
        for (uint k = 0; k < BT; ++k) {
            acc[k] = fma(wv, x[x_off[k] + xi], acc[k]);
        }
    }

    #pragma unroll
    for (uint k = 0; k < BT; ++k) {
        const float out = simd_sum(acc[k]);
        if (lane == 0 && (t0 + k) < a.n_tok) {
            const ulong yi = ulong(t0 + k) * ulong(a.n_out) + row;
            y[yi] = (a.add_residual != 0) ? (y[yi] + out) : out;
        }
    }
}

// Batched Coalesced Q6_K GEMM: 4 rows/TG (SG=4, 32 lanes/row) x BT=8 tokens/TG.
kernel void gemm_q6_k_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint BT = 8;

    const uint row = tg_id.x * SG + sg;
    const uint t0 = tg_id.y * BT;
    if (row >= a.n_out || t0 >= a.n_tok) return;

    const uint nb = a.n_in / 256;
    device const uchar* wp = W + ulong(row) * ulong(a.row_bytes);
    const uint is = lane >> 4;

    ulong x_off[BT];
    #pragma unroll
    for (uint k = 0; k < BT; ++k) {
        x_off[k] = ulong(min(t0 + k, a.n_tok - 1)) * ulong(a.n_in);
    }

    float acc[BT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint ib = 0; ib < nb; ++ib) {
        device const uchar* b = wp + ulong(ib) * 210;
        const float d = read_f16_u(b + 208);
        float sum[BT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (uint half_idx = 0; half_idx < 2; ++half_idx) {
            device const uchar* ql = b + half_idx * 64;
            device const uchar* qh = b + 128 + half_idx * 32;
            device const char*  sc = (device const char*)(b + 192 + half_idx * 8);

            const uchar ql_lo = ql[lane];
            const uchar ql_hi = ql[lane + 32];
            const uchar qh_v  = qh[lane];

            const float wa = float(sc[is])     * float(int((ql_lo & 0x0F) | ((qh_v & 3) << 4)) - 32);
            const float wb = float(sc[is + 2]) * float(int((ql_hi & 0x0F) | (((qh_v >> 2) & 3) << 4)) - 32);
            const float wc = float(sc[is + 4]) * float(int((ql_lo >> 4)   | (((qh_v >> 4) & 3) << 4)) - 32);
            const float wd = float(sc[is + 6]) * float(int((ql_hi >> 4)   | (((qh_v >> 6) & 3) << 4)) - 32);

            const uint xh_off = ib * 256 + half_idx * 128 + lane;
            #pragma unroll
            for (uint k = 0; k < BT; ++k) {
                device const float* xp = x + x_off[k] + xh_off;
                sum[k] = fma(wa, xp[0],  sum[k]);
                sum[k] = fma(wb, xp[32], sum[k]);
                sum[k] = fma(wc, xp[64], sum[k]);
                sum[k] = fma(wd, xp[96], sum[k]);
            }
        }
        #pragma unroll
        for (uint k = 0; k < BT; ++k) {
            acc[k] = fma(d, sum[k], acc[k]);
        }
    }

    #pragma unroll
    for (uint k = 0; k < BT; ++k) {
        const float out = simd_sum(acc[k]);
        if (lane == 0 && (t0 + k) < a.n_tok) {
            const ulong yi = ulong(t0 + k) * ulong(a.n_out) + row;
            y[yi] = (a.add_residual != 0) ? (y[yi] + out) : out;
        }
    }
}
