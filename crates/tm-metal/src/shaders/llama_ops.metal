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
    constexpr uint MAX_HD = 256;

    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const float scale = rsqrt(float(a.head_dim));

    device const float* qh = q + head * a.head_dim;
    threadgroup float q_shared[MAX_HD];
    if (tid < a.head_dim) {
        q_shared[tid] = qh[tid] * scale;
    }
    if (tid + 128 < a.head_dim) {
        q_shared[tid + 128] = qh[tid + 128] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup handles a strided subset of tokens `t = sg, sg + SG, ...`
    // and lanes within the simdgroup split `head_dim`.
    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // supports head_dim up to 256 (8 * 32)

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

    for (uint d = tid; d < a.head_dim; d += 128) {
        float m_all = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
        float denom = 0.0f;
        float acc = 0.0f;
        for (uint s = 0; s < SG; ++s) {
            if (tg_l[s] > 0.0f) {
                float w = exp(tg_m[s] - m_all);
                denom += tg_l[s] * w;
                acc += tg_o[s * MAX_HD + d] * w;
            }
        }
        out[head * a.head_dim + d] = acc / denom;
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
    constexpr uint MAX_HD = 256;

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
    if (tid + 128 < a.head_dim) {
        q_shared[tid + 128] = qh[tid + 128] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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

    for (uint d = tid; d < a.head_dim; d += 128) {
        float m_all = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
        float denom = 0.0f;
        float acc = 0.0f;
        for (uint s = 0; s < SG; ++s) {
            if (tg_l[s] > 0.0f) {
                float w = exp(tg_m[s] - m_all);
                denom += tg_l[s] * w;
                acc += tg_o[s * MAX_HD + d] * w;
            }
        }
        out[ulong(tok) * ulong(q_dim) + head * a.head_dim + d] = acc / denom;
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

// ============================================================================
// 7. Coalesced Q5_K GEMV & GEMM + HD=256 FlashAttention + Qwen3.5 GDN Kernels
// ============================================================================

static inline void unpack_q5k_sc_m(device const uchar* s, uint is, thread uchar& sc, thread uchar& m) {
    if (is < 4) {
        sc = s[is] & 63;
        m  = s[is + 4] & 63;
    } else {
        sc = (s[is + 4] & 0x0F) | ((s[is - 4] >> 6) << 4);
        m  = (s[is + 4] >> 4)   | ((s[is]     >> 6) << 4);
    }
}

// Coalesced Q5_K GEMV: 32 lanes cooperate on each 256-element superblock (`lane` in 0..31),
// processing NR = 2 consecutive rows per simdgroup (SG = 4 -> 8 rows / threadgroup)
// for high threadgroup occupancy on `n_out = 1024..2048` (`ssm_out.weight`).
kernel void gemv_q5_k_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NR = 2;
    constexpr uint SG = 4;

    const uint row0 = (tg_id * SG + sg) * NR;
    const uint nb = a.n_in / 256;
    if (row0 >= a.n_out) return;

    const uint r0 = row0;
    const uint r1 = min(row0 + 1, a.n_out - 1);
    device const uchar* wp[NR] = {
        W + ulong(r0) * ulong(a.row_bytes),
        W + ulong(r1) * ulong(a.row_bytes)
    };

    float acc[NR] = {0.0f, 0.0f};

    for (uint ib = 0; ib < nb; ++ib) {
        device const float* xp = x + ib * 256;
        const ulong blk_off = ulong(ib) * 176;

        float d[NR], dmin[NR];
        uchar qh_v[NR];
        float sum_q[NR] = {0.0f, 0.0f};
        float sum_m[NR] = {0.0f, 0.0f};

        #pragma unroll
        for (uint k = 0; k < NR; ++k) {
            device const uchar* b = wp[k] + blk_off;
            d[k]    = read_f16_u(b);
            dmin[k] = read_f16_u(b + 2);
            qh_v[k] = b[16 + lane];
        }

        #pragma unroll
        for (uint chunk = 0; chunk < 4; ++chunk) {
            const float x_lo = xp[chunk * 64 + lane];
            const float x_hi = xp[chunk * 64 + 32 + lane];
            const uint shift_lo = 2 * chunk;
            const uint shift_hi = 2 * chunk + 1;

            #pragma unroll
            for (uint k = 0; k < NR; ++k) {
                device const uchar* b = wp[k] + blk_off;
                uchar sc_lo, m_lo, sc_hi, m_hi;
                unpack_q5k_sc_m(b + 4, 2 * chunk,     sc_lo, m_lo);
                unpack_q5k_sc_m(b + 4, 2 * chunk + 1, sc_hi, m_hi);

                const uchar ql_b = b[48 + chunk * 32 + lane];
                const float q_lo = float((ql_b & 0x0F) | (((qh_v[k] >> shift_lo) & 1) << 4));
                const float q_hi = float((ql_b >> 4)   | (((qh_v[k] >> shift_hi) & 1) << 4));

                sum_q[k] = fma(float(sc_lo) * q_lo, x_lo, sum_q[k]);
                sum_q[k] = fma(float(sc_hi) * q_hi, x_hi, sum_q[k]);
                sum_m[k] = fma(float(m_lo), x_lo, sum_m[k]);
                sum_m[k] = fma(float(m_hi), x_hi, sum_m[k]);
            }
        }

        #pragma unroll
        for (uint k = 0; k < NR; ++k) {
            acc[k] = fma(d[k], sum_q[k], fma(-dmin[k], sum_m[k], acc[k]));
        }
    }

    const float out0 = simd_sum(acc[0]);
    const float out1 = simd_sum(acc[1]);
    if (lane == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + out0) : out0;
        if (row0 + 1 < a.n_out) y[r1] = (a.add_residual != 0) ? (y[r1] + out1) : out1;
    }
}

// Batched Coalesced Q5_K GEMM: 4 rows/TG (SG=4, 32 lanes/row) x BT=8 tokens/TG.
kernel void gemm_q5_k_res(
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

    ulong x_off[BT];
    #pragma unroll
    for (uint k = 0; k < BT; ++k) {
        x_off[k] = ulong(min(t0 + k, a.n_tok - 1)) * ulong(a.n_in);
    }

    float acc[BT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint ib = 0; ib < nb; ++ib) {
        device const uchar* b = wp + ulong(ib) * 176;
        const float d    = read_f16_u(b);
        const float dmin = read_f16_u(b + 2);
        const uchar qh_v = b[16 + lane];

        #pragma unroll
        for (uint chunk = 0; chunk < 4; ++chunk) {
            uchar sc_lo, m_lo, sc_hi, m_hi;
            unpack_q5k_sc_m(b + 4, 2 * chunk,     sc_lo, m_lo);
            unpack_q5k_sc_m(b + 4, 2 * chunk + 1, sc_hi, m_hi);

            const uchar ql_b = b[48 + chunk * 32 + lane];
            const float q_lo = float((ql_b & 0x0F) | (((qh_v >> (2 * chunk)) & 1) << 4));
            const float q_hi = float((ql_b >> 4)   | (((qh_v >> (2 * chunk + 1)) & 1) << 4));

            const float w_lo = fma(d * float(sc_lo), q_lo, -dmin * float(m_lo));
            const float w_hi = fma(d * float(sc_hi), q_hi, -dmin * float(m_hi));

            const uint x_base = ib * 256 + chunk * 64 + lane;
            #pragma unroll
            for (uint k = 0; k < BT; ++k) {
                device const float* xp = x + x_off[k] + x_base;
                acc[k] = fma(w_lo, xp[0],  acc[k]);
                acc[k] = fma(w_hi, xp[32], acc[k]);
            }
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

// Tiled Causal FlashAttention (`simdgroup_mma`) for `128 < head_dim <= 256` (e.g. Qwen3.5 HD=256):
// BQ = 32, BK = 16, 128 threads (4 simdgroups), 20 KB threadgroup memory.
kernel void fa_gqa_f16_kv_batch_hd256(
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
    constexpr uint MAX_HD = 256;
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

    threadgroup half  K_T_sh[MAX_HD][BK];
    threadgroup half  V_sh[BK][MAX_HD];
    threadgroup float S_sh[BQ][BK];
    threadgroup half  P_sh[BQ][BK];
    threadgroup float sg_stage[4][8][8];

    const uint r = lane >> 2;
    const uint sub = lane & 3;
    const uint q_tok = t0 + sg * 8 + r;
    const bool valid_q = (q_tok < a.n_tok);

    simdgroup_half8x8 q_mat[32];
    simdgroup_float8x8 o_acc[32];
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

    const uint k_loc = tid >> 3;
    const uint d_sub = tid & 7;

    for (uint kt = 0; kt < max_kv; kt += BK) {
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

        sg_stage[sg][r][sub * 2 + 0] = 0.0f;
        sg_stage[sg][r][sub * 2 + 1] = 0.0f;
        if ((r >> 1) == sub) {
            sg_stage[sg][r][r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

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

struct PairGemvArgs {
    uint n_in;
    uint a_rows;
    uint b_rows;
    uint row_bytes;
};

// Fused 2-matrix Q4_0 GEMV sharing input `x` (e.g. GDN `wqkv` + `wz` in a single dispatch!).
//Requires `a_rows` to be even (e.g. 6144).
kernel void gemv_pair_q4_0(
    device const uchar* Wa [[buffer(0)]],
    device const uchar* Wb [[buffer(1)]],
    device const float* x  [[buffer(2)]],
    device float* ya       [[buffer(3)]],
    device float* yb       [[buffer(4)]],
    constant PairGemvArgs& a [[buffer(5)]],
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
    const uint total_rows = a.a_rows + a.b_rows;
    if (row0 >= total_rows) return;

    device const uchar* mat_w;
    device float* y;
    uint local_row0;
    uint mat_rows;
    if (row0 < a.a_rows) {
        mat_w = Wa;
        y = ya;
        local_row0 = row0;
        mat_rows = a.a_rows;
    } else {
        mat_w = Wb;
        y = yb;
        local_row0 = row0 - a.a_rows;
        mat_rows = a.b_rows;
    }

    const uint nb = a.n_in / 32;
    const uint r0 = local_row0;
    const uint r1 = min(local_row0 + 1, mat_rows - 1);
    device const uchar* wp0 = mat_w + ulong(r0) * ulong(a.row_bytes);
    device const uchar* wp1 = mat_w + ulong(r1) * ulong(a.row_bytes);

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

// Fused 2-matrix Q8_0 GEMV sharing input `x` (e.g. GDN `ssm_alpha` + `ssm_beta` in 1 dispatch!).
// 1 row per simdgroup (32 coalesced lanes), SG = 4 rows / threadgroup.
kernel void gemv_pair_q8_0(
    device const uchar* Wa [[buffer(0)]],
    device const uchar* Wb [[buffer(1)]],
    device const float* x  [[buffer(2)]],
    device float* ya       [[buffer(3)]],
    device float* yb       [[buffer(4)]],
    constant PairGemvArgs& a [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    const uint row = tg_id * SG + sg;
    if (row >= a.a_rows + a.b_rows) return;

    device const uchar* mat_w = (row < a.a_rows) ? Wa : Wb;
    device float* y           = (row < a.a_rows) ? ya : yb;
    const uint local_row      = (row < a.a_rows) ? row : (row - a.a_rows);

    const uint nb = a.n_in / 32;
    device const uchar* wp = mat_w + ulong(local_row) * ulong(a.row_bytes);

    float acc = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        device const uchar* b = wp + ib * 34;
        const float d = read_f16_u(b);
        const float wv = d * float(((device const char*)(b + 2))[lane]);
        acc = fma(wv, x[ib * 32 + lane], acc);
    }
    const float out = simd_sum(acc);
    if (lane == 0) {
        y[local_row] = out;
    }
}

struct SplitQgArgs {
    uint n_head;
    uint head_dim;
    float eps;
};

// De-interleaves Qwen3.5 `qg_full` `[n_tok, n_head, 2 * head_dim]` into `q` (with per-head
// RMSNorm `q_norm` fused!) and `gate` (`[n_tok, n_head, head_dim]`). 1 simdgroup (32 lanes) per (head, tok).
kernel void qwen35_split_qg_qnorm_f32(
    device const float* qg_full [[buffer(0)]],
    device const float* q_norm  [[buffer(1)]],
    device float* q_out         [[buffer(2)]],
    device float* gate_out      [[buffer(3)]],
    constant SplitQgArgs& a     [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint head = tg.x;
    const uint tok  = tg.y;
    const uint hd   = a.head_dim;
    const ulong src_off = (ulong(tok) * ulong(a.n_head) + ulong(head)) * ulong(2 * hd);
    const ulong dst_off = (ulong(tok) * ulong(a.n_head) + ulong(head)) * ulong(hd);

    device const float* src_q = qg_full + src_off;
    device const float* src_g = qg_full + src_off + hd;
    device float* dst_q = q_out + dst_off;
    device float* dst_g = gate_out + dst_off;

    float ss = 0.0f;
    for (uint i = lane; i < hd; i += 32) {
        float v = src_q[i];
        ss = fma(v, v, ss);
        dst_g[i] = src_g[i];
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(hd) + a.eps);
    for (uint i = lane; i < hd; i += 32) {
        dst_q[i] = src_q[i] * scale * q_norm[i];
    }
}

kernel void sigmoid_mul_f32(
    device float* out [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    out[tid] *= 1.0f / (1.0f + exp(-gate[tid]));
}

struct GdnGatesArgs {
    uint n_v_heads;
    uint n_tok;
};

kernel void gdn_gates_f32(
    device const float* alpha   [[buffer(0)]],
    device const float* beta_in [[buffer(1)]],
    device const float* dt_bias [[buffer(2)]],
    device const float* ssm_a   [[buffer(3)]],
    device float* g_exp         [[buffer(4)]],
    device float* beta_out      [[buffer(5)]],
    constant GdnGatesArgs& a    [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint total = a.n_tok * a.n_v_heads;
    if (tid >= total) return;
    const uint vh = tid % a.n_v_heads;
    const float x = alpha[tid] + dt_bias[vh];
    const float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    g_exp[tid] = exp(sp * ssm_a[vh]);
    beta_out[tid] = 1.0f / (1.0f + exp(-beta_in[tid]));
}

struct GdnConvArgs {
    uint n_k_heads;
    uint n_v_heads;
    uint s_dim;
    uint n_tok;
    float eps;
};

// Fused Causal Conv1d (kernel=4) + SiLU (BEFORE split) + Split Q/K/V + per-head L2-Norm on Q & K ONLY!
// Grid: `2 * n_k_heads + n_v_heads` threadgroups (48 TGs for Qwen3.5), 128 threads (`s_dim == 128`) per TG.
kernel void gdn_conv_silu_l2_f32(
    device const float* qkv     [[buffer(0)]],
    device const float* conv1d  [[buffer(1)]],
    device float* conv_hist     [[buffer(2)]],
    device float* q_out         [[buffer(3)]],
    device float* k_out         [[buffer(4)]],
    device float* v_out         [[buffer(5)]],
    constant GdnConvArgs& a     [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint total_groups = 2 * a.n_k_heads + a.n_v_heads;
    if (group >= total_groups || tid >= a.s_dim) return;

    const uint conv_dim = total_groups * a.s_dim;
    const uint k_dim = a.n_k_heads * a.s_dim;
    const uint v_dim = a.n_v_heads * a.s_dim;
    const uint ch = group * a.s_dim + tid;

    float h0 = conv_hist[ch * 3 + 0];
    float h1 = conv_hist[ch * 3 + 1];
    float h2 = conv_hist[ch * 3 + 2];

    const float w0 = conv1d[ch * 4 + 0];
    const float w1 = conv1d[ch * 4 + 1];
    const float w2 = conv1d[ch * 4 + 2];
    const float w3 = conv1d[ch * 4 + 3];

    threadgroup float partial[4];
    const bool is_qk = (group < 2 * a.n_k_heads);

    for (uint t = 0; t < a.n_tok; ++t) {
        const float x3 = qkv[ulong(t) * ulong(conv_dim) + ch];
        const float raw = fma(h0, w0, fma(h1, w1, fma(h2, w2, x3 * w3)));
        h0 = h1;
        h1 = h2;
        h2 = x3;

        // AGENTS.md §9: SiLU BEFORE splitting into Q / K / V!
        float val = raw / (1.0f + exp(-raw));

        if (is_qk) {
            float ss = simd_sum(val * val);
            if (lane == 0) partial[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (sg == 0) {
                float v = (lane < 4) ? partial[lane] : 0.0f;
                float tot = simd_sum(v);
                if (lane == 0) partial[0] = rsqrt(tot + a.eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            val *= partial[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (group < a.n_k_heads) {
                q_out[ulong(t) * ulong(k_dim) + group * a.s_dim + tid] = val;
            } else {
                const uint kh = group - a.n_k_heads;
                k_out[ulong(t) * ulong(k_dim) + kh * a.s_dim + tid] = val;
            }
        } else {
            // AGENTS.md §9: NEVER L2-normalize V!
            const uint vh = group - 2 * a.n_k_heads;
            v_out[ulong(t) * ulong(v_dim) + vh * a.s_dim + tid] = val;
        }
    }

    conv_hist[ch * 3 + 0] = h0;
    conv_hist[ch * 3 + 1] = h1;
    conv_hist[ch * 3 + 2] = h2;
}

struct GdnRecurArgs {
    uint n_k_heads;
    uint n_v_heads;
    uint s_dim;
    uint n_tok;
};

// Register-resident Gated DeltaNet recurrence (`s_dim == 128`):
// 4 lanes cooperate on one transposed state row `j` (32 floats/lane in registers!).
// 1 threadgroup (128 threads = 4 simdgroups) owns 32 rows `j` of value head `vh`.
// Grid: `(s_dim / 32, n_v_heads)` = `(4, 16)` = 64 threadgroups on Qwen3.5!
// State is loaded from DRAM ONCE, updated across all `n_tok` steps in registers, and stored ONCE.
kernel void gdn_recurrence_f32(
    device const float* q     [[buffer(0)]],
    device const float* k     [[buffer(1)]],
    device const float* v     [[buffer(2)]],
    device const float* g_exp [[buffer(3)]],
    device const float* beta  [[buffer(4)]],
    device float* state       [[buffer(5)]],
    device float* out         [[buffer(6)]],
    constant GdnRecurArgs& a  [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint j = tg.x * 32 + sg * 8 + (lane >> 2);
    const uint vh = tg.y;
    if (vh >= a.n_v_heads || j >= a.s_dim) return;

    const uint sub = lane & 3;
    // AGENTS.md §9: Tiled head mapping `kh = vh % HK`!
    const uint kh = vh % a.n_k_heads;
    const uint S = a.s_dim; // 128
    const uint k_dim = a.n_k_heads * S;
    const uint v_dim = a.n_v_heads * S;
    const float scale = rsqrt(float(S));

    device float* s_row = state + (ulong(vh) * ulong(S) + ulong(j)) * ulong(S);
    float s_reg[32];
    #pragma unroll
    for (uint m = 0; m < 32; ++m) {
        s_reg[m] = s_row[m * 4 + sub];
    }

    for (uint t = 0; t < a.n_tok; ++t) {
        device const float* q_h = q + ulong(t) * ulong(k_dim) + kh * S;
        device const float* k_h = k + ulong(t) * ulong(k_dim) + kh * S;
        const float v_j = v[ulong(t) * ulong(v_dim) + vh * S + j];
        const float g   = g_exp[t * a.n_v_heads + vh];
        const float b   = beta[t * a.n_v_heads + vh];

        float partial_kv = 0.0f;
        #pragma unroll
        for (uint m = 0; m < 32; ++m) {
            const float sv = s_reg[m] * g;
            s_reg[m] = sv;
            partial_kv = fma(sv, k_h[m * 4 + sub], partial_kv);
        }
        float sum_kv = partial_kv + simd_shuffle_xor(partial_kv, 1);
        sum_kv += simd_shuffle_xor(sum_kv, 2);

        const float delta_j = (v_j - sum_kv) * b;

        float partial_oq = 0.0f;
        #pragma unroll
        for (uint m = 0; m < 32; ++m) {
            const float sv = fma(delta_j, k_h[m * 4 + sub], s_reg[m]);
            s_reg[m] = sv;
            partial_oq = fma(sv, q_h[m * 4 + sub], partial_oq);
        }
        float sum_oq = partial_oq + simd_shuffle_xor(partial_oq, 1);
        sum_oq += simd_shuffle_xor(sum_oq, 2);

        if (sub == 0) {
            out[ulong(t) * ulong(v_dim) + vh * S + j] = sum_oq * scale;
        }
    }

    #pragma unroll
    for (uint m = 0; m < 32; ++m) {
        s_row[m * 4 + sub] = s_reg[m];
    }
}

struct GdnNormArgs {
    uint n_v_heads;
    uint s_dim;
    float eps;
};

// Per-head gated RMSNorm with SiLU on `z`: `out_h = rms_norm(out_h, ssm_norm, eps) * silu(z_h)`.
// Grid: `(n_v_heads, n_tok)` threadgroups, 32 threads (1 simdgroup) per TG.
kernel void gdn_norm_gated_silu_f32(
    device float* out            [[buffer(0)]],
    device const float* ssm_norm [[buffer(1)]],
    device const float* z        [[buffer(2)]],
    constant GdnNormArgs& a      [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint vh = tg.x;
    const uint t  = tg.y;
    const uint S  = a.s_dim;
    const ulong off = (ulong(t) * ulong(a.n_v_heads) + ulong(vh)) * ulong(S);

    device float* o_h = out + off;
    device const float* z_h = z + off;

    float ss = 0.0f;
    for (uint i = lane; i < S; i += 32) {
        float v = o_h[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(S) + a.eps);

    for (uint i = lane; i < S; i += 32) {
        const float zv = z_h[i];
        const float silu_z = zv / (1.0f + exp(-zv));
        o_h[i] = (o_h[i] * scale * ssm_norm[i]) * silu_z;
    }
}

// ============================================================================
// 11. M2 Extensions: Gemma 4 (BF16, PLE, GeGLU, SWA/hd512, Softcap) & LFM2 (ShortConv)
// ============================================================================

inline float gelu_tanh_msl(float x) {
    return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

inline float bf16_to_f32_msl(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

struct ScaleArgs {
    uint n;
    float val;
};

kernel void scale_inplace_f32(
    device float* vec [[buffer(0)]],
    constant ScaleArgs& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < a.n) {
        vec[tid] *= a.val;
    }
}

kernel void softcap_inplace_f32(
    device float* vec [[buffer(0)]],
    constant ScaleArgs& a [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < a.n) {
        vec[tid] = precise::tanh(vec[tid] / a.val) * a.val;
    }
}

kernel void geglu_mul_f32(
    device float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        gate[tid] = gelu_tanh_msl(gate[tid]) * up[tid];
    }
}

// Fused FFN Gate + Up + GeGLU for Q4_0 weights (single-token decode):
// out[r] = gelu_tanh(dot(W_gate[r], x)) * dot(W_up[r], x) in a SINGLE dispatch!
kernel void gemv_pair_geglu_q4_0(
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
        y[row] = gelu_tanh_msl(g) * u;
    }
}

// Coalesced BF16 x F32 GEMV: 4 simdgroups x NR=4 rows/simdgroup = 16 rows per 128-thread TG.
kernel void gemv_bf16_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint NR = 4;

    const uint base_row = (tg_id * SG + sg) * NR;
    if (base_row >= a.n_out) return;

    const uint r0 = base_row;
    const uint r1 = min(base_row + 1, a.n_out - 1);
    const uint r2 = min(base_row + 2, a.n_out - 1);
    const uint r3 = min(base_row + 3, a.n_out - 1);
    device const ushort* w0 = (device const ushort*)(W + ulong(r0) * ulong(a.row_bytes));
    device const ushort* w1 = (device const ushort*)(W + ulong(r1) * ulong(a.row_bytes));
    device const ushort* w2 = (device const ushort*)(W + ulong(r2) * ulong(a.row_bytes));
    device const ushort* w3 = (device const ushort*)(W + ulong(r3) * ulong(a.row_bytes));

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const uint n4 = a.n_in / 4;
    device const float4* x4 = (device const float4*)x;
    device const ushort4* w0_4 = (device const ushort4*)w0;
    device const ushort4* w1_4 = (device const ushort4*)w1;
    device const ushort4* w2_4 = (device const ushort4*)w2;
    device const ushort4* w3_4 = (device const ushort4*)w3;

    for (uint i = lane; i < n4; i += 32) {
        const float4 xv = x4[i];
        const ushort4 b0 = w0_4[i];
        const ushort4 b1 = w1_4[i];
        const ushort4 b2 = w2_4[i];
        const ushort4 b3 = w3_4[i];
        acc0 = fma(bf16_to_f32_msl(b0.x), xv.x, acc0);
        acc0 = fma(bf16_to_f32_msl(b0.y), xv.y, acc0);
        acc0 = fma(bf16_to_f32_msl(b0.z), xv.z, acc0);
        acc0 = fma(bf16_to_f32_msl(b0.w), xv.w, acc0);

        acc1 = fma(bf16_to_f32_msl(b1.x), xv.x, acc1);
        acc1 = fma(bf16_to_f32_msl(b1.y), xv.y, acc1);
        acc1 = fma(bf16_to_f32_msl(b1.z), xv.z, acc1);
        acc1 = fma(bf16_to_f32_msl(b1.w), xv.w, acc1);

        acc2 = fma(bf16_to_f32_msl(b2.x), xv.x, acc2);
        acc2 = fma(bf16_to_f32_msl(b2.y), xv.y, acc2);
        acc2 = fma(bf16_to_f32_msl(b2.z), xv.z, acc2);
        acc2 = fma(bf16_to_f32_msl(b2.w), xv.w, acc2);

        acc3 = fma(bf16_to_f32_msl(b3.x), xv.x, acc3);
        acc3 = fma(bf16_to_f32_msl(b3.y), xv.y, acc3);
        acc3 = fma(bf16_to_f32_msl(b3.z), xv.z, acc3);
        acc3 = fma(bf16_to_f32_msl(b3.w), xv.w, acc3);
    }
    for (uint i = n4 * 4 + lane; i < a.n_in; i += 32) {
        const float xv = x[i];
        acc0 = fma(bf16_to_f32_msl(w0[i]), xv, acc0);
        acc1 = fma(bf16_to_f32_msl(w1[i]), xv, acc1);
        acc2 = fma(bf16_to_f32_msl(w2[i]), xv, acc2);
        acc3 = fma(bf16_to_f32_msl(w3[i]), xv, acc3);
    }

    const float s0 = simd_sum(acc0);
    const float s1 = simd_sum(acc1);
    const float s2 = simd_sum(acc2);
    const float s3 = simd_sum(acc3);
    if (lane == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + s0) : s0;
        if (base_row + 1 < a.n_out) y[r1] = (a.add_residual != 0) ? (y[r1] + s1) : s1;
        if (base_row + 2 < a.n_out) y[r2] = (a.add_residual != 0) ? (y[r2] + s2) : s2;
        if (base_row + 3 < a.n_out) y[r3] = (a.add_residual != 0) ? (y[r3] + s3) : s3;
    }
}

// Contiguous 4-block coalesced Q8_0 GEMV (`SG=4`, `NR=4` => 16 rows/TG):
// All 32 lanes read a contiguous 136-byte segment (4 consecutive 34-byte Q8_0 blocks) via `packed_char4` + `float4`!
kernel void gemv_q8_0_fast(
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

    const uint sub = lane >> 3; // 0..3
    const uint li  = lane & 7;  // 0..7

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    uint ib = 0;
    for (; ib + 4 <= nb; ib += 4) {
        const uint blk = ib + sub;
        const float4 xv = ((device const float4*)(x + blk * 32))[li];
        device const uchar* b0 = wp0 + blk * 34;
        device const uchar* b1 = wp1 + blk * 34;
        device const uchar* b2 = wp2 + blk * 34;
        device const uchar* b3 = wp3 + blk * 34;

        const float d0 = float(*(device const half*)b0);
        const float d1 = float(*(device const half*)b1);
        const float d2 = float(*(device const half*)b2);
        const float d3 = float(*(device const half*)b3);

        const char4 q0 = ((device const packed_char4*)(b0 + 2))[li];
        const char4 q1 = ((device const packed_char4*)(b1 + 2))[li];
        const char4 q2 = ((device const packed_char4*)(b2 + 2))[li];
        const char4 q3 = ((device const packed_char4*)(b3 + 2))[li];

        acc0 = fma(d0, dot(float4(q0), xv), acc0);
        acc1 = fma(d1, dot(float4(q1), xv), acc1);
        acc2 = fma(d2, dot(float4(q2), xv), acc2);
        acc3 = fma(d3, dot(float4(q3), xv), acc3);
    }
    for (; ib < nb; ++ib) {
        const float xv = x[ib * 32 + lane];
        device const uchar* b0 = wp0 + ib * 34;
        device const uchar* b1 = wp1 + ib * 34;
        device const uchar* b2 = wp2 + ib * 34;
        device const uchar* b3 = wp3 + ib * 34;
        acc0 = fma(float(*(device const half*)b0) * float(((device const char*)(b0 + 2))[lane]), xv, acc0);
        acc1 = fma(float(*(device const half*)b1) * float(((device const char*)(b1 + 2))[lane]), xv, acc1);
        acc2 = fma(float(*(device const half*)b2) * float(((device const char*)(b2 + 2))[lane]), xv, acc2);
        acc3 = fma(float(*(device const half*)b3) * float(((device const char*)(b3 + 2))[lane]), xv, acc3);
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

// Fused softcapping + greedy argmax in a SINGLE pass over `vocab_size` logits (saves 2.1 MB DRAM traffic/tok).
kernel void softcap_and_argmax_f32(
    device float* logits [[buffer(0)]],
    device uint* out_id [[buffer(1)]],
    constant ScaleArgs& a [[buffer(2)]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NT = 256;
    float best_v = -INFINITY;
    uint best_i = 0;
    const float inv_cap = 1.0f / a.val;
    for (uint i = tid; i < a.n; i += NT) {
        float v = precise::tanh(logits[i] * inv_cap) * a.val;
        logits[i] = v;
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

struct AttnExtArgs {
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint start_pos;
    uint n_tok;
    uint window;
    float attn_scale;
    uint _pad;
};

// Fast 1-simdgroup (32 threads, zero barriers, zero threadgroup memory) GQA decode attention
// supporting `head_dim <= 512`, custom `attn_scale`, and sliding window (`active_kv_len <= 128`).
kernel void gqa_attn_f16_kv_ext_1sg(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttnExtArgs& a [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    if (head >= a.n_head) return;
    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const uint seq_len = a.start_pos + 1;
    const uint start_kv = (a.window > 0 && seq_len > a.window) ? (seq_len - a.window) : 0;
    const uint n_vec = (a.head_dim + 31) / 32;

    device const float* qh = q + ulong(head) * ulong(a.head_dim);
    device float* oh = out + ulong(head) * ulong(a.head_dim);

    if (a.head_dim == 256) {
        float q_reg[8];
        float o_local[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            q_reg[k] = qh[k * 32 + lane] * a.attn_scale;
        }
        float m_local = -INFINITY;
        float l_local = 0.0f;
        for (uint t = start_kv; t < seq_len; ++t) {
            const ulong kv_off = ulong(t) * ulong(kv_dim) + ulong(kv_h) * 256ul;
            device const half* kh = k_cache + kv_off;
            device const half* vh = v_cache + kv_off;
            float dot = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                dot = fma(q_reg[k], float(kh[k * 32 + lane]), dot);
            }
            dot = simd_sum(dot);
            const float m_new = max(m_local, dot);
            const float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
            const float p = exp(dot - m_new);
            l_local = l_local * alpha + p;
            m_local = m_new;
            #pragma unroll
            for (uint k = 0; k < 8; ++k) {
                o_local[k] = o_local[k] * alpha + p * float(vh[k * 32 + lane]);
            }
        }
        const float inv_l = 1.0f / l_local;
        #pragma unroll
        for (uint k = 0; k < 8; ++k) {
            oh[k * 32 + lane] = o_local[k] * inv_l;
        }
        return;
    }

    if (a.head_dim == 512) {
        float q_reg[16];
        float o_local[16] = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            q_reg[k] = qh[k * 32 + lane] * a.attn_scale;
        }
        float m_local = -INFINITY;
        float l_local = 0.0f;
        for (uint t = start_kv; t < seq_len; ++t) {
            const ulong kv_off = ulong(t) * ulong(kv_dim) + ulong(kv_h) * 512ul;
            device const half* kh = k_cache + kv_off;
            device const half* vh = v_cache + kv_off;
            float dot = 0.0f;
            #pragma unroll
            for (uint k = 0; k < 16; ++k) {
                dot = fma(q_reg[k], float(kh[k * 32 + lane]), dot);
            }
            dot = simd_sum(dot);
            const float m_new = max(m_local, dot);
            const float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
            const float p = exp(dot - m_new);
            l_local = l_local * alpha + p;
            m_local = m_new;
            #pragma unroll
            for (uint k = 0; k < 16; ++k) {
                o_local[k] = o_local[k] * alpha + p * float(vh[k * 32 + lane]);
            }
        }
        const float inv_l = 1.0f / l_local;
        #pragma unroll
        for (uint k = 0; k < 16; ++k) {
            oh[k * 32 + lane] = o_local[k] * inv_l;
        }
        return;
    }

    float q_reg[16] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (uint k = 0; k < n_vec; ++k) {
        const uint d = k * 32 + lane;
        if (d < a.head_dim) {
            q_reg[k] = qh[d] * a.attn_scale;
        }
    }

    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[16] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    for (uint t = start_kv; t < seq_len; ++t) {
        const ulong kv_off = ulong(t) * ulong(kv_dim) + ulong(kv_h) * ulong(a.head_dim);
        device const half* kh = k_cache + kv_off;
        device const half* vh = v_cache + kv_off;

        float dot = 0.0f;
        for (uint k = 0; k < n_vec; ++k) {
            const uint d = k * 32 + lane;
            if (d < a.head_dim) {
                dot = fma(q_reg[k], float(kh[d]), dot);
            }
        }
        dot = simd_sum(dot);

        const float m_new = max(m_local, dot);
        const float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
        const float p = exp(dot - m_new);
        l_local = l_local * alpha + p;
        m_local = m_new;

        for (uint k = 0; k < n_vec; ++k) {
            const uint d = k * 32 + lane;
            if (d < a.head_dim) {
                o_local[k] = o_local[k] * alpha + p * float(vh[d]);
            }
        }
    }

    const float inv_l = 1.0f / l_local;
    for (uint k = 0; k < n_vec; ++k) {
        const uint d = k * 32 + lane;
        if (d < a.head_dim) {
            oh[d] = o_local[k] * inv_l;
        }
    }
}

// Hardware-accelerated simdgroup_float8x8 tiled BF16 GEMM (32x32x32) preserving full FP32 exponent range.
kernel void gemm_bf16_res(
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
    threadgroup float O_tile[32][32];

    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const uint sg_m = (sg >> 1) * 16;
    const uint sg_n = (sg & 1) * 16;
    const uint loc_row = tid >> 2;      // 0..31
    const uint loc_grp = (tid & 3) * 8; // 0, 8, 16, 24

    const uint nb = a.n_in / 32;
    const bool valid_t = (t0 + loc_row) < a.n_tok;
    const bool valid_r = (r0 + loc_row) < a.n_out;
    device const float* x_row = valid_t ? (x + ulong(t0 + loc_row) * ulong(a.n_in)) : x;
    device const ushort* w_row = valid_r ? (device const ushort*)(W + ulong(r0 + loc_row) * ulong(a.row_bytes)) : (device const ushort*)W;

    for (uint ib = 0; ib < nb; ++ib) {
        const uint k_base = ib * 32 + loc_grp;
        if (valid_t) {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                X_tile[loc_row][loc_grp + j] = x_row[k_base + j];
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                X_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }
        if (valid_r) {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_grp + j][loc_row] = bf16_to_f32_msl(w_row[k_base + j]);
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_grp + j][loc_row] = 0.0f;
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

    simdgroup_store(acc00, &O_tile[sg_m][sg_n], 32);
    simdgroup_store(acc01, &O_tile[sg_m][sg_n + 8], 32);
    simdgroup_store(acc10, &O_tile[sg_m + 8][sg_n], 32);
    simdgroup_store(acc11, &O_tile[sg_m + 8][sg_n + 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < 1024; idx += 128) {
        const uint tm = idx >> 5;
        const uint rn = idx & 31;
        if ((t0 + tm) < a.n_tok && (r0 + rn) < a.n_out) {
            const ulong yi = ulong(t0 + tm) * ulong(a.n_out) + ulong(r0 + rn);
            const float val = O_tile[tm][rn];
            y[yi] = (a.add_residual != 0) ? (y[yi] + val) : val;
        }
    }
}

// Coalesced F32 x F32 GEMV (8 rows / 128-thread TG) & GEMM (4 rows x 8 tokens / TG).
kernel void gemv_f32_res(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint NR = 2;
    const uint base_row = (tg_id * SG + sg) * NR;
    if (base_row >= a.n_out) return;

    const uint r0 = base_row;
    const uint r1 = min(base_row + 1, a.n_out - 1);
    device const float* w0 = (device const float*)(W + ulong(r0) * ulong(a.row_bytes));
    device const float* w1 = (device const float*)(W + ulong(r1) * ulong(a.row_bytes));

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint i = lane; i < a.n_in; i += 32) {
        const float xv = x[i];
        acc0 = fma(w0[i], xv, acc0);
        acc1 = fma(w1[i], xv, acc1);
    }
    const float s0 = simd_sum(acc0);
    const float s1 = simd_sum(acc1);
    if (lane == 0) {
        y[r0] = (a.add_residual != 0) ? (y[r0] + s0) : s0;
        if (base_row + 1 < a.n_out) {
            y[r1] = (a.add_residual != 0) ? (y[r1] + s1) : s1;
        }
    }
}

kernel void gemm_f32_res(
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

    device const float* wp = (device const float*)(W + ulong(row) * ulong(a.row_bytes));
    ulong x_off[BT];
    #pragma unroll
    for (uint k = 0; k < BT; ++k) {
        x_off[k] = ulong(min(t0 + k, a.n_tok - 1)) * ulong(a.n_in);
    }
    float acc[BT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint xi = lane; xi < a.n_in; xi += 32) {
        const float wv = wp[xi];
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

// Fused Post-Norm + Residual Add (`dst_res[t] += rms_norm(x[t], w, eps)`).
// Grid: `n_tok` threadgroups, 128 threads (4 simdgroups) per TG.
kernel void rms_norm_add_f32_batch(
    device float* dst_res [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* w [[buffer(2)]],
    constant NormArgs& a [[buffer(3)]],
    uint tok [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint NT = 128;
    const ulong off = ulong(tok) * ulong(a.n);
    device const float* xt = x + off;
    device float* dt = dst_res + off;

    float ss = 0.0f;
    for (uint i = tid; i < a.n; i += NT) {
        float v = xt[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);

    threadgroup float sg_sums[4];
    if (lane == 0) sg_sums[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (sg == 0) {
        total = (lane < 4) ? sg_sums[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) {
            sg_sums[0] = rsqrt(total / float(a.n) + a.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = sg_sums[0];
    for (uint i = tid; i < a.n; i += NT) {
        dt[i] += xt[i] * scale * w[i];
    }
}

// Per-(token, layer) PLE projection RMSNorm + accumulate into `ple_inputs`:
// Grid: `n_tok * n_layers` threadgroups, 32 threads (1 simdgroup) per TG.
kernel void ple_combine_proj_f32(
    device float* ple_inputs [[buffer(0)]],
    device const float* ple_proj [[buffer(1)]],
    device const float* proj_norm_scaled [[buffer(2)]],
    constant NormArgs& a [[buffer(3)]],
    uint slice_id [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const ulong off = ulong(slice_id) * ulong(a.n);
    device float* dst = ple_inputs + off;
    device const float* src = ple_proj + off;

    float ss = 0.0f;
    for (uint i = lane; i < a.n; i += 32) {
        float v = src[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(a.n) + a.eps);
    for (uint i = lane; i < a.n; i += 32) {
        dst[i] += src[i] * scale * proj_norm_scaled[i];
    }
}

struct PleGateArgs {
    uint ple_dim;
    uint ple_total;
    uint layer_idx;
    uint n_tok;
};

kernel void ple_gate_gelu_mul_f32(
    device float* gate [[buffer(0)]],
    device const float* ple_inputs [[buffer(1)]],
    constant PleGateArgs& a [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint total = a.n_tok * a.ple_dim;
    if (tid < total) {
        const uint t = tid / a.ple_dim;
        const uint d = tid % a.ple_dim;
        const ulong p_idx = ulong(t) * ulong(a.ple_total) + ulong(a.layer_idx) * ulong(a.ple_dim) + ulong(d);
        gate[tid] = gelu_tanh_msl(gate[tid]) * ple_inputs[p_idx];
    }
}

// Per-head unit-weight RMSNorm (`w = 1.0`, used for Gemma 4 V-norm).
// Grid: `n_tok * n_heads` threadgroups, 32 threads (1 simdgroup) per TG.
kernel void qk_norm_unit_f32_batch(
    device float* vec [[buffer(0)]],
    constant NormArgs& a [[buffer(1)]],
    uint head_idx [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    device float* h = vec + ulong(head_idx) * ulong(a.n);
    float ss = 0.0f;
    for (uint i = lane; i < a.n; i += 32) {
        float v = h[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    const float scale = rsqrt(ss / float(a.n) + a.eps);
    for (uint i = lane; i < a.n; i += 32) {
        h[i] *= scale;
    }
}

struct RopeQOnlyArgs {
    uint n_head;
    uint head_dim;
    uint rope_dim;
    uint start_pos;
    uint rope_type;
    uint has_freq_factors;
    uint n_tok;
    float freq_base;
};

// Q-only RoPE for KV-shared layers (`il >= n_layer_kv_from_start` in Gemma 4).
kernel void rope_q_only_f32_batch(
    device float* q [[buffer(0)]],
    device const float* freq_factors [[buffer(1)]],
    constant RopeQOnlyArgs& a [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint half_rope = a.rope_dim / 2;
    const uint per_tok = a.n_head * half_rope;
    const uint total = a.n_tok * per_tok;
    if (tid >= total) return;

    const uint tok = tid / per_tok;
    const uint rem = tid % per_tok;
    const uint h = rem / half_rope;
    const uint i = rem % half_rope;

    float freq = pow(a.freq_base, -2.0f * float(i) / float(a.rope_dim));
    if (a.has_freq_factors != 0) {
        freq /= freq_factors[i];
    }
    const float theta = float(a.start_pos + tok) * freq;
    const float c = cos(theta);
    const float s = sin(theta);

    const uint q_dim = a.n_head * a.head_dim;
    device float* head = q + ulong(tok) * ulong(q_dim) + ulong(h) * ulong(a.head_dim);
    const uint i0 = (a.rope_type == 0) ? (2 * i) : i;
    const uint i1 = (a.rope_type == 0) ? (2 * i + 1) : (i + half_rope);
    const float x0 = head[i0];
    const float x1 = head[i1];
    head[i0] = x0 * c - x1 * s;
    head[i1] = x0 * s + x1 * c;
}

// Extended GQA Attention over F16 KV cache supporting `head_dim <= 512`, custom `attn_scale`
// (e.g., `1.0f` on Gemma 4), and standard sliding window (`window == 0` => full causal).
// Grid: `(n_head, n_tok)` threadgroups, 128 threads (4 simdgroups) per TG.
kernel void gqa_attn_f16_kv_ext_batch(
    device const float* q [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant AttnExtArgs& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint SG = 4;
    constexpr uint MAX_HD = 512;

    const uint head = tg.x;
    const uint tok = tg.y;
    if (head >= a.n_head || tok >= a.n_tok) return;

    const uint group = a.n_head / a.n_head_kv;
    const uint kv_h = head / group;
    const uint q_dim = a.n_head * a.head_dim;
    const uint kv_dim = a.n_head_kv * a.head_dim;
    const uint seq_len = a.start_pos + tok + 1;
    const uint start_kv = (a.window > 0 && seq_len > a.window) ? (seq_len - a.window) : 0;

    device const float* qh = q + ulong(tok) * ulong(q_dim) + ulong(head) * ulong(a.head_dim);
    threadgroup float q_shared[MAX_HD];
    for (uint d = tid; d < a.head_dim; d += 128) {
        q_shared[d] = qh[d] * a.attn_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_local = -INFINITY;
    float l_local = 0.0f;
    float o_local[16] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    const uint n_vec = (a.head_dim + 31) / 32;

    for (uint t = start_kv + sg; t < seq_len; t += SG) {
        const ulong kv_off = ulong(t) * ulong(kv_dim) + ulong(kv_h) * ulong(a.head_dim);
        device const half* kh = k_cache + kv_off;
        device const half* vh = v_cache + kv_off;

        float dot = 0.0f;
        for (uint k = 0; k < n_vec; ++k) {
            const uint d = k * 32 + lane;
            if (d < a.head_dim) {
                dot = fma(q_shared[d], float(kh[d]), dot);
            }
        }
        dot = simd_sum(dot);

        const float m_new = max(m_local, dot);
        const float alpha = (m_local == -INFINITY) ? 0.0f : exp(m_local - m_new);
        const float p = exp(dot - m_new);
        l_local = l_local * alpha + p;
        m_local = m_new;

        for (uint k = 0; k < n_vec; ++k) {
            const uint d = k * 32 + lane;
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
        const uint d = k * 32 + lane;
        if (d < a.head_dim) {
            tg_o[sg * MAX_HD + d] = o_local[k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* oh = out + ulong(tok) * ulong(q_dim) + ulong(head) * ulong(a.head_dim);
    for (uint d = tid; d < a.head_dim; d += 128) {
        const float m_all = max(max(tg_m[0], tg_m[1]), max(tg_m[2], tg_m[3]));
        float denom = 0.0f;
        float acc = 0.0f;
        for (uint s = 0; s < SG; ++s) {
            if (tg_l[s] > 0.0f) {
                const float w = exp(tg_m[s] - m_all);
                denom += tg_l[s] * w;
                acc += tg_o[s * MAX_HD + d] * w;
            }
        }
        oh[d] = acc / denom;
    }
}

struct ShortConvArgs {
    uint n_embd;
    uint l_cache;
    uint n_tok;
    uint _pad;
};

// Causal gated depthwise ShortConv for LFM2 (`bx = b * x` -> `l_cache` conv -> `* c`).
// 1 thread per channel `d in 0..n_embd`, sequentially stepping `t = 0..n_tok` so multi-token
// prefill and single-token decode have identical per-token FP32 operation order (`AGENTS.md` §9).
kernel void lfm2_shortconv_f32(
    device const float* bcx [[buffer(0)]],
    device const float* conv_w [[buffer(1)]],
    device float* conv_hist [[buffer(2)]],
    device float* y_out [[buffer(3)]],
    constant ShortConvArgs& a [[buffer(4)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= a.n_embd) return;

    const uint l_cache = a.l_cache;
    const uint hist_len = (l_cache > 0) ? (l_cache - 1) : 0;
    device const float* w = conv_w + ulong(d) * ulong(l_cache);
    device float* h_ptr = conv_hist + ulong(d) * ulong(hist_len);

    float hist[16];
    for (uint k = 0; k < hist_len && k < 16; ++k) {
        hist[k] = h_ptr[k];
    }

    for (uint t = 0; t < a.n_tok; ++t) {
        device const float* row = bcx + ulong(t) * ulong(3 * a.n_embd);
        const float b = row[d];
        const float c = row[a.n_embd + d];
        const float xv = row[2 * a.n_embd + d];
        const float bx = b * xv;

        float acc = 0.0f;
        for (uint k = 0; k < hist_len && k < 16; ++k) {
            acc = fma(hist[k], w[k], acc);
        }
        if (l_cache > 0) {
            acc = fma(bx, w[hist_len], acc);
        }

        if (hist_len > 0) {
            for (uint k = 0; k + 1 < hist_len && k + 1 < 16; ++k) {
                hist[k] = hist[k + 1];
            }
            hist[min(hist_len - 1, 15u)] = bx;
        }

        y_out[ulong(t) * ulong(a.n_embd) + ulong(d)] = acc * c;
    }

    for (uint k = 0; k < hist_len && k < 16; ++k) {
        h_ptr[k] = hist[k];
    }
}

// ============================================================================
// 11. Phase T4: Hardware-Accelerated Transposed Quantized GEMM (`simdgroup_float8x8`)
//     Computes backward activation gradient:
//       dx[n_tok, n_in] = (add_residual ? dx : 0) + dy[n_tok, n_out] * W[n_out, n_in]
//     Grid: `(ceil(n_in / 32), ceil(n_tok / 32))`, 128 threads (4 simdgroups) per TG.
// ============================================================================

#define TM_METAL_TRANSPOSED_MMA_BODY(...)                                                          \
    const uint c0 = tg_id.x * 32;                                                                  \
    const uint t0 = tg_id.y * 32;                                                                  \
    if (c0 >= a.n_in || t0 >= a.n_tok) return;                                                     \
                                                                                                   \
    threadgroup float DY_tile[32][33];                                                             \
    threadgroup float W_tile[32][33];                                                              \
                                                                                                   \
    simdgroup_float8x8 acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                    \
    simdgroup_float8x8 acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                    \
    simdgroup_float8x8 acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                    \
    simdgroup_float8x8 acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                    \
                                                                                                   \
    const uint sg_m = (sg >> 1) * 16;                                                              \
    const uint sg_n = (sg & 1) * 16;                                                               \
    const uint loc_row = tid >> 2;      /* 0..31 */                                                \
    const uint loc_grp = (tid & 3) * 8; /* 0, 8, 16, 24 */                                         \
    const uint q_grp   = (tid & 3) * 4; /* 0, 4, 8, 12 */                                          \
    const bool valid_t = (t0 + loc_row) < a.n_tok;                                                 \
    device const float* dy_row = valid_t ? (dy + ulong(t0 + loc_row) * ulong(a.n_out)) : dy;       \
                                                                                                   \
    for (uint r0 = 0; r0 < a.n_out; r0 += 32) {                                                    \
        if (valid_t) {                                                                             \
            for (uint j = 0; j < 8; ++j) {                                                         \
                const uint r = r0 + loc_grp + j;                                                   \
                DY_tile[loc_row][loc_grp + j] = (r < a.n_out) ? dy_row[r] : 0.0f;                  \
            }                                                                                      \
        } else {                                                                                   \
            for (uint j = 0; j < 8; ++j) {                                                         \
                DY_tile[loc_row][loc_grp + j] = 0.0f;                                              \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        const bool valid_r = (r0 + loc_row) < a.n_out;                                             \
        __VA_ARGS__                                                                                \
                                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                           \
                                                                                                   \
        for (uint kk = 0; kk < 32; kk += 8) {                                                      \
            simdgroup_float8x8 a0, a1, b0, b1;                                                     \
            simdgroup_load(a0, &DY_tile[sg_m][kk], 33);                                            \
            simdgroup_load(a1, &DY_tile[sg_m + 8][kk], 33);                                        \
            simdgroup_load(b0, &W_tile[kk][sg_n], 33);                                             \
            simdgroup_load(b1, &W_tile[kk][sg_n + 8], 33);                                         \
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);                                   \
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);                                   \
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);                                   \
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);                                   \
        }                                                                                          \
                                                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                           \
    }                                                                                              \
                                                                                                   \
    simdgroup_store(acc00, &DY_tile[sg_m][sg_n], 33);                                              \
    simdgroup_store(acc01, &DY_tile[sg_m][sg_n + 8], 33);                                          \
    simdgroup_store(acc10, &DY_tile[sg_m + 8][sg_n], 33);                                          \
    simdgroup_store(acc11, &DY_tile[sg_m + 8][sg_n + 8], 33);                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                               \
                                                                                                   \
    if (valid_t) {                                                                                 \
        const ulong dx_base = ulong(t0 + loc_row) * ulong(a.n_in) + ulong(c0 + loc_grp);           \
        for (uint j = 0; j < 8; ++j) {                                                             \
            if (c0 + loc_grp + j < a.n_in) {                                                       \
                const float out = DY_tile[loc_row][loc_grp + j];                                   \
                dx[dx_base + j] = (a.add_residual != 0) ? (dx[dx_base + j] + out) : out;           \
            }                                                                                      \
        }                                                                                          \
    }

kernel void gemm_q4_0_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint ib = tg_id.x;
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        const uint qb = (tid & 3) * 4;
        if (valid_r) {
            device const uchar* b = W + ulong(r0 + loc_row) * ulong(a.row_bytes) + ulong(ib) * 18;
            const float d = read_f16_u(b);
            device const uchar* qs = b + 2 + qb;
            for (uint j = 0; j < 4; ++j) {
                const uchar byte = qs[j];
                W_tile[loc_row][qb + j]      = (float(byte & 0x0F) - 8.0f) * d;
                W_tile[loc_row][16 + qb + j] = (float(byte >> 4)   - 8.0f) * d;
            }
        } else {
            for (uint j = 0; j < 4; ++j) {
                W_tile[loc_row][qb + j]      = 0.0f;
                W_tile[loc_row][16 + qb + j] = 0.0f;
            }
        }
    })
}

kernel void gemm_q4_1_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint ib = tg_id.x;
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        const uint qb = (tid & 3) * 4;
        if (valid_r) {
            device const uchar* b = W + ulong(r0 + loc_row) * ulong(a.row_bytes) + ulong(ib) * 20;
            const float d = read_f16_u(b);
            const float m = read_f16_u(b + 2);
            device const uchar* qs = b + 4 + qb;
            for (uint j = 0; j < 4; ++j) {
                const uchar byte = qs[j];
                W_tile[loc_row][qb + j]      = fma(float(byte & 0x0F), d, m);
                W_tile[loc_row][16 + qb + j] = fma(float(byte >> 4),   d, m);
            }
        } else {
            for (uint j = 0; j < 4; ++j) {
                W_tile[loc_row][qb + j]      = 0.0f;
                W_tile[loc_row][16 + qb + j] = 0.0f;
            }
        }
    })
}

kernel void gemm_q8_0_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint ib = tg_id.x;
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        if (valid_r) {
            device const uchar* b = W + ulong(r0 + loc_row) * ulong(a.row_bytes) + ulong(ib) * 34;
            const float d = read_f16_u(b);
            device const char* qs = (device const char*)(b + 2) + loc_grp;
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_row][loc_grp + j] = float(qs[j]) * d;
            }
        } else {
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }
    })
}

kernel void gemm_q5_k_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint ib = tg_id.x >> 3;
    const uint sub32 = tg_id.x & 7;
    const uint chunk = sub32 >> 1;
    const uint hi = sub32 & 1;
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        if (valid_r) {
            device const uchar* b = W + ulong(r0 + loc_row) * ulong(a.row_bytes) + ulong(ib) * 176;
            const float d    = read_f16_u(b);
            const float dmin = read_f16_u(b + 2);
            uchar sc, m;
            unpack_q5k_sc_m(b + 4, sub32, sc, m);
            const float d_sc = d * float(sc);
            const float dm   = -dmin * float(m);
            for (uint j = 0; j < 8; ++j) {
                const uint l = loc_grp + j;
                const uchar ql_b = b[48 + chunk * 32 + l];
                const uchar qh_b = b[16 + l];
                const uint nib = (hi == 0) ? uint(ql_b & 0x0F) : uint(ql_b >> 4);
                const uint hb  = uint((qh_b >> sub32) & 1);
                W_tile[loc_row][l] = fma(d_sc, float(nib | (hb << 4)), dm);
            }
        } else {
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }
    })
}

kernel void gemm_q6_k_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    const uint ib = tg_id.x >> 3;
    const uint sub32 = tg_id.x & 7;
    const uint half_idx = sub32 >> 2;
    const uint qtr = sub32 & 3;
    const uint qh_shift = 2 * qtr;
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        if (valid_r) {
            device const uchar* b = W + ulong(r0 + loc_row) * ulong(a.row_bytes) + ulong(ib) * 210;
            const float d = read_f16_u(b + 208);
            device const uchar* ql = b + half_idx * 64 + (qtr & 1) * 32;
            device const uchar* qh = b + 128 + half_idx * 32;
            device const char*  sc = (device const char*)(b + 192 + half_idx * 8);
            for (uint j = 0; j < 8; ++j) {
                const uint l = loc_grp + j;
                const uint is = (l >> 4) + 2 * qtr;
                const uint nib = (qtr < 2) ? uint(ql[l] & 0x0F) : uint(ql[l] >> 4);
                const uint qhv = uint((qh[l] >> qh_shift) & 3);
                W_tile[loc_row][l] = d * float(sc[is]) * float(int(nib | (qhv << 4)) - 32);
            }
        } else {
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }
    })
}

kernel void gemm_bf16_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        if (valid_r) {
            device const ushort* wr = (device const ushort*)(W + ulong(r0 + loc_row) * ulong(a.row_bytes));
            for (uint j = 0; j < 8; ++j) {
                const uint c = c0 + loc_grp + j;
                W_tile[loc_row][loc_grp + j] = (c < a.n_in) ? bf16_to_f32_msl(wr[c]) : 0.0f;
            }
        } else {
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }
    })
}

kernel void gemm_f32_transposed(
    device const uchar* W [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    constant GemmArgs& a [[buffer(3)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    TM_METAL_TRANSPOSED_MMA_BODY({
        (void)q_grp;
        if (valid_r) {
            device const float* wr = (device const float*)(W + ulong(r0 + loc_row) * ulong(a.row_bytes));
            for (uint j = 0; j < 8; ++j) {
                const uint c = c0 + loc_grp + j;
                W_tile[loc_row][loc_grp + j] = (c < a.n_in) ? wr[c] : 0.0f;
            }
        } else {
            for (uint j = 0; j < 8; ++j) {
                W_tile[loc_row][loc_grp + j] = 0.0f;
            }
        }
    })
}

#undef TM_METAL_TRANSPOSED_MMA_BODY

// ============================================================================
// 12. Phase T4: Dynamic GPU LoRA Side-Branch Kernels (1-CB zero-sync)
// ============================================================================

struct LoraLinearArgs {
    uint d_in;
    uint d_out;
    uint rank;
    uint n_tok;
    float scale;
};

// `h[t, r] = scale * dot(A[r, 0..d_in], x[t, 0..d_in])`
// 1 simdgroup (32 lanes) per `(t, r)` pair; 4 simdgroups (128 threads) per TG.
// Identical per-token reduction order for `n_tok == 1` and `n_tok > 1` (`AGENTS.md` §9).
kernel void lora_down_f32(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* h [[buffer(2)]],
    constant LoraLinearArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint idx = tg_id * 4 + uint(sg);
    const uint total = a.n_tok * a.rank;
    if (idx >= total) return;

    const uint t = idx / a.rank;
    const uint r = idx % a.rank;
    device const float* ar = A + ulong(r) * ulong(a.d_in);
    device const float* xt = x + ulong(t) * ulong(a.d_in);

    float acc = 0.0f;
    for (uint i = lane; i < a.d_in; i += 32) {
        acc = fma(ar[i], xt[i], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        h[idx] = acc * a.scale;
    }
}

// `y[t, col] += sum_{r=0..rank-1} h[t, r] * B_T[r, col]`
// `B_T` is stored transposed `[rank, d_out]` so adjacent threads `col, col+1` coalesce 100%!
kernel void lora_up_axpy_f32(
    device const float* B_T [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant LoraLinearArgs& a [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint t = gid.y;
    if (col >= a.d_out || t >= a.n_tok) return;

    device const float* ht = h + ulong(t) * ulong(a.rank);
    float acc = 0.0f;
    for (uint r = 0; r < a.rank; ++r) {
        acc = fma(ht[r], B_T[ulong(r) * ulong(a.d_out) + col], acc);
    }
    y[ulong(t) * ulong(a.d_out) + col] += acc;
}

struct LoraEmbedArgs {
    uint d_out;
    uint rank;
    uint n_tok;
    uint token_id;
    uint use_tok_buf;
    float scale;
};

// `x[t, col] += sum_{r=0..rank-1} (scale * A[tok_t, r]) * B_T[r, col]`
kernel void lora_embed_f32(
    device const float* A [[buffer(0)]],
    device const float* B_T [[buffer(1)]],
    device const uint* token_ids [[buffer(2)]],
    device float* x [[buffer(3)]],
    constant LoraEmbedArgs& a [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint t = gid.y;
    if (col >= a.d_out || t >= a.n_tok) return;

    const uint tok = (a.use_tok_buf != 0) ? token_ids[t] : a.token_id;
    device const float* ar = A + ulong(tok) * ulong(a.rank);
    float acc = 0.0f;
    for (uint r = 0; r < a.rank; ++r) {
        acc = fma(a.scale * ar[r], B_T[ulong(r) * ulong(a.d_out) + col], acc);
    }
    x[ulong(t) * ulong(a.d_out) + col] += acc;
}

struct BiasArgs {
    uint d_out;
    uint n_tok;
};

kernel void add_bias_f32(
    device const float* bias [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant BiasArgs& a [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint t = gid.y;
    if (col >= a.d_out || t >= a.n_tok) return;
    y[ulong(t) * ulong(a.d_out) + col] += bias[col];
}
