// Q4_0 x F32 quantized GEMV / multi-token shared-weight GEMV for Apple Silicon Metal.
//
// Design inspired by Strata `metal/kernels/gemv.metal` and `quant.metal`:
// - Copyright (c) 2026 Niko1221 and the Strata contributors (MIT License)
// - BlockQ4_0 layout (18 bytes = fp16 d + 16 bytes packed 4-bit nibbles) matches
//   ggml (Copyright (c) 2023-2026 The ggml authors, MIT License).
//
// Threadgroup geometry:
// - 128 threads = 4 simdgroups (SG = 4), each simdgroup has 32 lanes.
// - LPR = 8 lanes share one output row -> 4 row-groups per simdgroup.
// - Each lane processes NR = 2 consecutive rows -> 8 rows/simdgroup, 32 rows/threadgroup.
// - Each lane reads 2-byte-aligned ushort pairs from BlockQ4_0, accumulates
//   `dot(q, x) - 8 * sum(x)` (sharing `x` and `sum(x)` across NR rows),
//   then reduces across the 8 lanes via simd_shuffle_down.

#include <metal_stdlib>
using namespace metal;

struct GemvArgs {
    uint n_in;       // K (multiple of 32)
    uint n_out;      // M (number of weight rows)
    uint n_tok;      // N (number of tokens, 1..=8)
    uint row_bytes;  // bytes per row of W = (n_in / 32) * 18
};

static inline float read_f16_unaligned(device const uchar* p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

static inline float reduce_lpr8(float v) {
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

static inline float reduce_lpr32(float v) {
    return simd_sum(v);
}

// Single-token Q4_0 x F32 GEMV: y[r] = dot(W[r], x), LPR = 8, NR = 2, SG = 4 (32 rows / TG).
kernel void gemv_q4_0_f32_lpr8_nr2(
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
    constexpr uint RG = 32 / LPR; // 4
    constexpr uint SG = 4;

    const uint li = lane % LPR;
    const uint lrow0 = (sg * RG + lane / LPR) * NR;
    const uint row0 = tg_id * (SG * RG * NR) + lrow0;
    const uint nb = a.n_in / 32;

    if (row0 >= a.n_out) {
        return;
    }

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

        const float d0 = read_f16_unaligned(b0);
        const float d1 = read_f16_unaligned(b1);
        device const ushort* p0 = (device const ushort*)(b0 + 2);
        device const ushort* p1 = (device const ushort*)(b1 + 2);

        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float sumx = 0.0f;

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
        if (row0 + 1 < a.n_out) {
            y[row0 + 1] = out1;
        }
    }
}

// Multi-token (1..=8) shared-weight Q4_0 x F32 GEMV: y[t * n_out + r] = dot(W[r], x[t]).
// Unpacks each Q4_0 weight block ONCE and applies it across all n_tok activation vectors.
kernel void gemm_q4_0_f32_lpr8(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant GemvArgs& a [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    constexpr uint LPR = 8;
    constexpr uint RG = 32 / LPR; // 4
    constexpr uint SG = 4;

    const uint li = lane % LPR;
    const uint row = tg_id * (SG * RG) + (sg * RG + lane / LPR);
    const uint nb = a.n_in / 32;
    const uint n_tok = min(a.n_tok, 8u);

    if (row >= a.n_out) {
        return;
    }

    device const uchar* wp = W + ulong(row) * ulong(a.row_bytes);

    float acc[8];
    for (uint t = 0; t < n_tok; ++t) {
        acc[t] = 0.0f;
    }

    for (uint ib = li; ib < nb; ib += LPR) {
        device const uchar* b = wp + ib * 18;
        const float d = read_f16_unaligned(b);
        device const ushort* p = (device const ushort*)(b + 2);

        float dot[8];
        for (uint t = 0; t < n_tok; ++t) {
            dot[t] = 0.0f;
        }

        // Keep unpacked weights purely in registers (4 floats per ushort) to avoid local-memory spill.
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const ushort v = p[j];
            const float q0 = float(v & 15) - 8.0f;
            const float q2 = float((v >> 4) & 15) - 8.0f;
            const float q1 = float((v >> 8) & 15) - 8.0f;
            const float q3 = float(v >> 12) - 8.0f;

            const uint k0 = ib * 32 + 2 * j;
            for (uint t = 0; t < n_tok; ++t) {
                device const float* xp = x + ulong(t) * ulong(a.n_in) + k0;
                dot[t] = fma(q0, xp[0], dot[t]);
                dot[t] = fma(q1, xp[1], dot[t]);
                dot[t] = fma(q2, xp[16], dot[t]);
                dot[t] = fma(q3, xp[17], dot[t]);
            }
        }

        for (uint t = 0; t < n_tok; ++t) {
            acc[t] = fma(d, dot[t], acc[t]);
        }
    }

    for (uint t = 0; t < n_tok; ++t) {
        const float val = reduce_lpr8(acc[t]);
        if (li == 0) {
            y[ulong(t) * ulong(a.n_out) + row] = val;
        }
    }
}
