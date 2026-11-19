// End-to-end Llama / Qwen3 Metal compute kernels for Apple Silicon.
//
// Copyright (c) 2026 Pengfei Zhang and the tinymodel contributors (MIT OR Apache-2.0).
// Quantization block formats match ggml (Copyright (c) 2023-2026 The ggml authors, MIT).
// GEMV threadgroup geometry inspired by Strata `metal/kernels/gemv.metal` (MIT).

#include <metal_stdlib>
using namespace metal;

struct GemvArgs {
    uint n_in;
    uint n_out;
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
