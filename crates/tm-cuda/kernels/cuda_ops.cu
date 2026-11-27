// CUDA C compute kernels for `tinymodel` (`tm-cuda`).
//
// Portions of the MMVQ Q8_1 `__dp4a` dot-product formulation and warp-sharded GDN state
// update are adapted from ggml / llama.cpp and `Strata/src/kernels/cuda/{native_mmvq.cu,native_gdn.cu}`:
//   MIT License — Copyright (c) 2023-2026 The ggml authors / Strata contributors.
//
// Target: `compute_75+` (NVIDIA Turing T4 `sm_75`, Ampere `sm_80/86`, Ada `sm_89`, Hopper `sm_90+`).

#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>
#include <stdint.h>

using namespace nvcuda;

// ============================================================================
// GGUF Quantization Block Layouts & Safe Unaligned Load Helpers
// ============================================================================

// Must match `tm_core::DType` integer tags passed from Rust:
#define DTYPE_F32    0
#define DTYPE_F16    1
#define DTYPE_Q4_0   2
#define DTYPE_Q4_1   3
#define DTYPE_Q5_0   6
#define DTYPE_Q8_0   8
#define DTYPE_Q3_K   11
#define DTYPE_Q4_K   12
#define DTYPE_Q5_K   13
#define DTYPE_Q6_K   14
#define DTYPE_IQ4_NL 20
#define DTYPE_IQ4_XS 23
#define DTYPE_BF16   30
#define DTYPE_NVFP4  40
#define DTYPE_Q2_0   42

__constant__ int8_t c_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

struct BlockQ5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
}; // 22 bytes (32 weights, 2-byte aligned)

struct BlockIQ4NL {
    half d;
    uint8_t qs[16];
}; // 18 bytes (32 weights, 2-byte aligned)

struct BlockQ3K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
}; // 110 bytes (256 weights, 2-byte aligned)

struct BlockIQ4XS {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
}; // 136 bytes (256 weights, 2-byte aligned)

struct BlockQ2_0 {
    half d;
    uint8_t qs[16];
}; // 18 bytes (64 weights, 2-byte aligned)

struct BlockNvFp4 {
    uint8_t d[4];
    uint8_t qs[32];
}; // 36 bytes (64 weights)

struct BlockQ4_0 {
    half d;
    uint8_t qs[16];
}; // 18 bytes (2-byte aligned)

struct BlockQ4_1 {
    half2 dm;       // x = d, y = min
    uint8_t qs[16]; // 4-byte aligned!
}; // 20 bytes (4-byte aligned)

struct BlockQ8_0 {
    half d;
    int8_t qs[32];
}; // 34 bytes (2-byte aligned)

struct BlockQ8_1 {
    half2 ds;       // x = d, y = d * sum(qs[0..31])
    int8_t qs[32];  // 4-byte aligned when array base is 4-byte aligned!
}; // 36 bytes

struct BlockQ4K {
    half2 dm;           // d, dmin
    uint8_t scales[12];
    uint8_t qs[128];
}; // 144 bytes (4-byte aligned)

struct BlockQ5K {
    half2 dm;           // d, dmin
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
}; // 176 bytes (4-byte aligned)

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
}; // 210 bytes (2-byte aligned)

static_assert(sizeof(BlockQ2_0) == 18 && alignof(BlockQ2_0) == 2, "BlockQ2_0 layout");
static_assert(sizeof(BlockQ4_0) == 18 && alignof(BlockQ4_0) == 2, "BlockQ4_0 layout");
static_assert(sizeof(BlockQ4_1) == 20 && alignof(BlockQ4_1) == 4, "BlockQ4_1 layout");
static_assert(sizeof(BlockQ8_0) == 34 && alignof(BlockQ8_0) == 2, "BlockQ8_0 layout");
static_assert(sizeof(BlockQ8_1) == 36 && alignof(BlockQ8_1) == 4, "BlockQ8_1 layout");
static_assert(sizeof(BlockQ4K) == 144 && alignof(BlockQ4K) == 4, "BlockQ4K layout");
static_assert(sizeof(BlockQ5K) == 176 && alignof(BlockQ5K) == 4, "BlockQ5K layout");
static_assert(sizeof(BlockQ6K) == 210 && alignof(BlockQ6K) == 2, "BlockQ6K layout");

__device__ __forceinline__ int dp4a_i8(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
    const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
    return c + int(pa[0]) * int(pb[0]) + int(pa[1]) * int(pb[1]) + int(pa[2]) * int(pb[2]) + int(pa[3]) * int(pb[3]);
#endif
}

// Safe load of 32-bit int from a 2-byte-aligned address (e.g. 18B BlockQ4_0 / 34B BlockQ8_0 / 210B BlockQ6K).
__device__ __forceinline__ int load_i32_2b(const void* p) {
    const uint16_t* u = static_cast<const uint16_t*>(p);
    return int(uint32_t(u[0]) | (uint32_t(u[1]) << 16));
}

__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
    return __uint_as_float(uint32_t(bits) << 16);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset, 32);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset, 32));
    }
    return v;
}

__device__ __forceinline__ void decode_q4k_scale_min(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Unpack one byte of `BlockQ2_0` (4 x 2-bit codes in 0..3) into signed int8x4 `(c - 1)` in {-1, 0, +1, +2} with zero borrow.
__device__ __forceinline__ int unpack_q2_0_byte_s8x4(uint8_t byte) {
    const uint32_t u = uint32_t(byte);
    const uint32_t p = (u & 3u) | (((u >> 2) & 3u) << 8) | (((u >> 4) & 3u) << 16) | (((u >> 6) & 3u) << 24);
    return int(((p | 0x80808080u) - 0x01010101u) ^ 0x80808080u);
}

// Dequantize one element `elem_in_row` (0..cols-1) from row `w_row` of any supported GGUF dtype.
__device__ __forceinline__ float dequant_elem(
    int dtype,
    const uint8_t* __restrict__ w_row,
    int col
) {
    switch (dtype) {
        case DTYPE_F32: {
            return reinterpret_cast<const float*>(w_row)[col];
        }
        case DTYPE_F16: {
            return __half2float(reinterpret_cast<const half*>(w_row)[col]);
        }
        case DTYPE_BF16: {
            return bf16_to_f32(reinterpret_cast<const uint16_t*>(w_row)[col]);
        }
        case DTYPE_Q4_0: {
            const int b = col >> 5;
            const int i = col & 31;
            const BlockQ4_0& blk = reinterpret_cast<const BlockQ4_0*>(w_row)[b];
            const float d = __half2float(blk.d);
            const uint8_t byte = blk.qs[i & 15];
            const int q = (i < 16) ? (byte & 0x0F) : (byte >> 4);
            return d * float(q - 8);
        }
        case DTYPE_Q4_1: {
            const int b = col >> 5;
            const int i = col & 31;
            const BlockQ4_1& blk = reinterpret_cast<const BlockQ4_1*>(w_row)[b];
            const float2 dm = __half22float2(blk.dm);
            const uint8_t byte = blk.qs[i & 15];
            const int q = (i < 16) ? (byte & 0x0F) : (byte >> 4);
            return dm.x * float(q) + dm.y;
        }
        case DTYPE_Q5_0: {
            const int b = col >> 5;
            const int i = col & 31;
            const BlockQ5_0& blk = reinterpret_cast<const BlockQ5_0*>(w_row)[b];
            const uint32_t qh = uint32_t(load_i32_2b(blk.qh));
            const int j = i & 15;
            const int nib = (i < 16) ? (blk.qs[j] & 0x0F) : (blk.qs[j] >> 4);
            const int hi = ((qh >> i) & 1u) << 4;
            return __half2float(blk.d) * float((nib | hi) - 16);
        }
        case DTYPE_IQ4_NL: {
            const int b = col >> 5;
            const int i = col & 31;
            const BlockIQ4NL& blk = reinterpret_cast<const BlockIQ4NL*>(w_row)[b];
            const int nib = (i < 16) ? (blk.qs[i] & 0x0F) : (blk.qs[i - 16] >> 4);
            return __half2float(blk.d) * float(c_kvalues_iq4nl[nib]);
        }
        case DTYPE_Q3_K: {
            const int sb = col >> 8;
            const int rem = col & 255;
            const BlockQ3K& blk = reinterpret_cast<const BlockQ3K*>(w_row)[sb];
            const uint32_t a0 = uint32_t(load_i32_2b(blk.scales + 0));
            const uint32_t a1 = uint32_t(load_i32_2b(blk.scales + 4));
            const uint32_t tmp = uint32_t(load_i32_2b(blk.scales + 8));
            const uint32_t km1 = 0x03030303u;
            const uint32_t km2 = 0x0f0f0f0fu;
            uint32_t aux[4];
            aux[0] = (a0 & km2) | ((tmp & km1) << 4);
            aux[1] = (a1 & km2) | (((tmp >> 2) & km1) << 4);
            aux[2] = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);
            aux[3] = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);
            const int is = rem >> 4; // 0..15
            const int sc = int(reinterpret_cast<const int8_t*>(aux)[is]) - 32;
            const int half128 = rem >> 7;
            const int j = (rem >> 5) & 3;
            const int l = rem & 31;
            const uint8_t m = uint8_t(1u << (4 * half128 + j));
            const int qv = int((blk.qs[half128 * 32 + l] >> (2 * j)) & 3u) - ((blk.hmask[l] & m) ? 0 : 4);
            return __half2float(blk.d) * float(sc) * float(qv);
        }
        case DTYPE_IQ4_XS: {
            const int sb = col >> 8;
            const int rem = col & 255;
            const BlockIQ4XS& blk = reinterpret_cast<const BlockIQ4XS*>(w_row)[sb];
            const int ib = rem >> 5;
            const int in32 = rem & 31;
            const int ls = int((blk.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F)
                         | int(((blk.scales_h >> (2 * ib)) & 3u) << 4);
            const uint8_t qb = blk.qs[ib * 16 + (in32 & 15)];
            const int nib = (in32 < 16) ? (qb & 0x0F) : (qb >> 4);
            return __half2float(blk.d) * float(ls - 32) * float(c_kvalues_iq4nl[nib]);
        }
        case DTYPE_Q8_0: {
            const int b = col >> 5;
            const int i = col & 31;
            const BlockQ8_0& blk = reinterpret_cast<const BlockQ8_0*>(w_row)[b];
            return __half2float(blk.d) * float(blk.qs[i]);
        }
        case DTYPE_Q2_0: {
            const int b = col >> 6;
            const int in64 = col & 63;
            const BlockQ2_0& blk = reinterpret_cast<const BlockQ2_0*>(w_row)[b];
            const uint8_t byte = blk.qs[in64 >> 2];
            const int code = (byte >> (2 * (in64 & 3))) & 3;
            return __half2float(blk.d) * float(code - 1);
        }
        case DTYPE_NVFP4: {
            static constexpr float kFp4Lut[16] = {
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f
            };
            const int b = col >> 6;
            const int in64 = col & 63;
            const int sub = in64 >> 4;
            const int idx = in64 & 15;
            const BlockNvFp4& blk = reinterpret_cast<const BlockNvFp4*>(w_row)[b];
            const uint8_t ue = blk.d[sub];
            float d = 0.0f;
            if ((ue & 0x7F) != 0 && (ue & 0x7F) != 0x7F) {
                const uint32_t exp = (ue >> 3) & 0x0F;
                const uint32_t mant = ue & 0x07;
                const float raw = (exp == 0)
                    ? (float(mant) * (1.0f / 1024.0f))
                    : __uint_as_float(((exp + 119u) << 23) | (mant << 20));
                d = (ue & 0x80) ? -raw : raw;
            }
            const uint8_t qbyte = blk.qs[sub * 8 + (idx & 7)];
            const int nib = (idx < 8) ? (qbyte & 0x0F) : (qbyte >> 4);
            return d * kFp4Lut[nib];
        }
        case DTYPE_Q4_K: {
            const int sb = col >> 8;
            const int rem = col & 255;
            const BlockQ4K& blk = reinterpret_cast<const BlockQ4K*>(w_row)[sb];
            const float2 dm = __half22float2(blk.dm);
            const int chunk64 = rem >> 6;          // 0..3
            const int in64 = rem & 63;
            const int half32 = in64 >> 5;          // 0..1
            const int l = in64 & 31;
            const int is = 2 * chunk64 + half32;
            uint8_t sc, m;
            decode_q4k_scale_min(is, blk.scales, sc, m);
            const uint8_t qbyte = blk.qs[chunk64 * 32 + l];
            const int q = (half32 == 0) ? (qbyte & 0x0F) : (qbyte >> 4);
            return dm.x * float(sc) * float(q) - dm.y * float(m);
        }
        case DTYPE_Q5_K: {
            const int sb = col >> 8;
            const int rem = col & 255;
            const BlockQ5K& blk = reinterpret_cast<const BlockQ5K*>(w_row)[sb];
            const float2 dm = __half22float2(blk.dm);
            const int chunk64 = rem >> 6;
            const int in64 = rem & 63;
            const int half32 = in64 >> 5;
            const int l = in64 & 31;
            const int is = 2 * chunk64 + half32;
            uint8_t sc, m;
            decode_q4k_scale_min(is, blk.scales, sc, m);
            const uint8_t qbyte = blk.qs[chunk64 * 32 + l];
            const int ql = (half32 == 0) ? (qbyte & 0x0F) : (qbyte >> 4);
            const int bit = chunk64 * 2 + half32;
            const int qh = ((blk.qh[l] >> bit) & 1) << 4;
            return dm.x * float(sc) * float(ql | qh) - dm.y * float(m);
        }
        case DTYPE_Q6_K: {
            const int sb = col >> 8;
            const int rem = col & 255;
            const BlockQ6K& blk = reinterpret_cast<const BlockQ6K*>(w_row)[sb];
            const float d = __half2float(blk.d);
            const int half128 = rem >> 7;       // 0..1
            const int in128 = rem & 127;
            const int quad = in128 >> 5;        // 0..3
            const int l = in128 & 31;
            const uint8_t* ql_ptr = blk.ql + half128 * 64;
            const uint8_t* qh_ptr = blk.qh + half128 * 32;
            const int8_t* sc_ptr = blk.scales + half128 * 8;
            int q_low = 0;
            if (quad == 0) q_low = ql_ptr[l] & 0x0F;
            else if (quad == 1) q_low = ql_ptr[l + 32] & 0x0F;
            else if (quad == 2) q_low = ql_ptr[l] >> 4;
            else q_low = ql_ptr[l + 32] >> 4;
            const int q_high = ((qh_ptr[l] >> (2 * quad)) & 3) << 4;
            const int q = (q_low | q_high) - 32;
            const int sc = int(sc_ptr[(l >> 4) + 2 * quad]);
            return d * float(sc) * float(q);
        }
        default:
            return 0.0f;
    }
}

extern "C" {

// ============================================================================
// 1. Dynamic Activation Quantization (`f32 -> Q8_1` & `f32 -> Q8_0`)
// ============================================================================

// Grid: (ceil(n_blocks / 4), n_tok, 1), Block: (32, 4, 1)
// Stores `ds = half2(d, sum_of_original_floats)` matching `ggml-cuda` & `Strata` (`q8_1_finite.hpp`).
__global__ void quantize_q8_1_f32(
    const float* __restrict__ x,
    BlockQ8_1* __restrict__ out,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int blk_idx = int(blockIdx.x) * int(blockDim.y) + int(threadIdx.y);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    if (blk_idx >= n_blocks || tok >= n_tok) return;

    const float val = x[size_t(tok) * cols + blk_idx * 32 + lane];
    const float amax = warp_reduce_max(fabsf(val));
    const float sum_x = warp_reduce_sum(val);
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int q = __float2int_rn(val * id);
    const int8_t qs = int8_t(max(-127, min(127, q)));

    BlockQ8_1& dst = out[size_t(tok) * n_blocks + blk_idx];
    dst.qs[lane] = qs;
    if (lane == 0) {
        dst.ds = __floats2half2_rn(d, sum_x);
    }
}

// Fused SwiGLU + Q8_1 quantization (`y_q8_1 = quantize_q8_1(silu(gate) * up)`), eliminating
// an entire global-memory roundtrip of `[n_tok, n_ff]` floats before `ffn_down`!
// Grid: (ceil(n_blocks / 4), n_tok, 1), Block: (32, 4, 1)
__global__ void swiglu_quantize_q8_1_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    BlockQ8_1* __restrict__ out,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int blk_idx = int(blockIdx.x) * int(blockDim.y) + int(threadIdx.y);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    if (blk_idx >= n_blocks || tok >= n_tok) return;

    const size_t idx = size_t(tok) * cols + blk_idx * 32 + lane;
    const float g = gate[idx];
    const float val = (g / (1.0f + expf(-g))) * up[idx];
    const float amax = warp_reduce_max(fabsf(val));
    const float sum_x = warp_reduce_sum(val);
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int q = __float2int_rn(val * id);
    const int8_t qs = int8_t(max(-127, min(127, q)));

    BlockQ8_1& dst = out[size_t(tok) * n_blocks + blk_idx];
    dst.qs[lane] = qs;
    if (lane == 0) {
        dst.ds = __floats2half2_rn(d, sum_x);
    }
}

// Fused `[gate, up]` (`[n_rows, 2 * n_ff]`) SwiGLU + `Q8_1` quantization (`[n_rows, n_ff / 32]`).
// Grid: (ceil(n_blocks / 4), n_rows, 1), Block: (32, 4, 1)
__global__ void swiglu_fused_quantize_q8_1_f32(
    const float* __restrict__ gate_up,
    BlockQ8_1* __restrict__ out,
    int n_ff,
    int n_rows
) {
    const int n_blocks = n_ff >> 5;
    const int blk_idx = int(blockIdx.x) * int(blockDim.y) + int(threadIdx.y);
    const int row = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    if (blk_idx >= n_blocks || row >= n_rows) return;

    const float* src = gate_up + size_t(row) * (2 * n_ff);
    const int col = blk_idx * 32 + lane;
    const float g = src[col];
    const float val = (g / (1.0f + expf(-g))) * src[n_ff + col];
    const float amax = warp_reduce_max(fabsf(val));
    const float sum_x = warp_reduce_sum(val);
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int q = __float2int_rn(val * id);
    const int8_t qs = int8_t(max(-127, min(127, q)));

    BlockQ8_1& dst = out[size_t(row) * n_blocks + blk_idx];
    dst.qs[lane] = qs;
    if (lane == 0) {
        dst.ds = __floats2half2_rn(d, sum_x);
    }
}

// ============================================================================
// 2. MMVQ Quantized Decode GEMV / Small-Batch Shared-Weight GEMM (`n_tok = 1..8`)
// ============================================================================

#define MMVQ_MAX_TOK 8

// `Qi = 4` (`VDR = 2`): 2 lanes cooperate on one `BlockQ4_0` (`iqs = 0` or `2`), reading
// coalesced 8-byte chunks (`288` contiguous bytes per 32-lane warp).
__device__ __forceinline__ float dot_q4_0_q8_1_half(const BlockQ4_0& wb, const BlockQ8_1& xb, int iqs) {
    int sumi = 0;
    const int* xqs = reinterpret_cast<const int*>(xb.qs);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_i32_2b(wb.qs + 4 * (iqs + i));
        const int vi0 = v & 0x0F0F0F0F;
        const int vi1 = (v >> 4) & 0x0F0F0F0F;
        sumi = dp4a_i8(vi0, xqs[iqs + i], sumi);
        sumi = dp4a_i8(vi1, xqs[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(xb.ds);
    return __half2float(wb.d) * (float(sumi) * ds.x - 4.0f * ds.y);
}

// Dual-mode `Q4_0 × Q8_1` MMVQ:
// - `n_tok == 1 && cols >= 2048`: `grid.x = rows`, 4 warps cooperate on 1 row (`64` blocks = 2048 cols/iter)
// - otherwise: `grid.x = ceil(rows / 4)`, 1 warp per row (`16` blocks = 512 cols/iter)
__global__ void __launch_bounds__(128, 1) mmvq_q4_0_q8_1(
    const BlockQ4_0* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);

    if (n_tok == 1 && cols >= 2048) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockQ4_0* w_row = w + size_t(row) * n_blocks;
        const int tid = warp_id * 32 + lane;
        const int iqs = 2 * (tid & 1);
        float acc = 0.0f;
        for (int kbx = tid >> 1; kbx < n_blocks; kbx += 64) {
            acc += dot_q4_0_q8_1_half(w_row[kbx], xq[kbx], iqs);
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) {
            partial[warp_id - 1][lane] = acc;
        }
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out[row] = acc;
            }
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const BlockQ4_0* w_row = w + size_t(row) * n_blocks;
    const int iqs = 2 * (lane & 1);
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int kbx = lane >> 1; kbx < n_blocks; kbx += 16) {
        const BlockQ4_0& wb = w_row[kbx];
        const float d_w = __half2float(wb.d);
        const int v0 = load_i32_2b(wb.qs + 4 * iqs);
        const int v1 = load_i32_2b(wb.qs + 4 * (iqs + 1));
        const int lo0 = v0 & 0x0F0F0F0F;
        const int hi0 = (v0 >> 4) & 0x0F0F0F0F;
        const int lo1 = v1 & 0x0F0F0F0F;
        const int hi1 = (v1 >> 4) & 0x0F0F0F0F;

        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_blocks + kbx];
            const float2 ds = __half22float2(xb.ds);
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
            sumi = dp4a_i8(lo0, xqs[iqs], sumi);
            sumi = dp4a_i8(hi0, xqs[iqs + 4], sumi);
            sumi = dp4a_i8(lo1, xqs[iqs + 1], sumi);
            sumi = dp4a_i8(hi1, xqs[iqs + 5], sumi);
            acc[t] += d_w * (float(sumi) * ds.x - 4.0f * ds.y);
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

// Dual-mode `Q4_1 × Q8_1` MMVQ (`20B` 4-byte aligned `BlockQ4_1`, used by `ffn_down` in GGUF `-Q4_0` models!):
// `sum_j (dm.x * q_j + dm.y) * x_j = dm.x * ds.x * sum(q_j * x_q_j) + dm.y * ds.y` (`0.5 * dm.y * ds.y` per half-block).
__global__ void __launch_bounds__(128, 1) mmvq_q4_1_q8_1(
    const BlockQ4_1* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);

    if (n_tok == 1 && cols >= 2048) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockQ4_1* w_row = w + size_t(row) * n_blocks;
        const int tid = warp_id * 32 + lane;
        const int iqs = 2 * (tid & 1);
        float acc = 0.0f;
        for (int kbx = tid >> 1; kbx < n_blocks; kbx += 64) {
            const BlockQ4_1& wb = w_row[kbx];
            const BlockQ8_1& xb = xq[kbx];
            const int* wqs = reinterpret_cast<const int*>(wb.qs);
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int v = wqs[iqs + i];
                sumi = dp4a_i8(v & 0x0F0F0F0F, xqs[iqs + i], sumi);
                sumi = dp4a_i8((v >> 4) & 0x0F0F0F0F, xqs[iqs + i + 4], sumi);
            }
            const float2 dm = __half22float2(wb.dm);
            const float2 ds = __half22float2(xb.ds);
            acc += dm.x * ds.x * float(sumi) + 0.5f * dm.y * ds.y;
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const BlockQ4_1* w_row = w + size_t(row) * n_blocks;
    const int iqs = 2 * (lane & 1);
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int kbx = lane >> 1; kbx < n_blocks; kbx += 16) {
        const BlockQ4_1& wb = w_row[kbx];
        const float2 dm = __half22float2(wb.dm);
        const int* wqs = reinterpret_cast<const int*>(wb.qs);
        const int v0 = wqs[iqs];
        const int v1 = wqs[iqs + 1];
        const int lo0 = v0 & 0x0F0F0F0F;
        const int hi0 = (v0 >> 4) & 0x0F0F0F0F;
        const int lo1 = v1 & 0x0F0F0F0F;
        const int hi1 = (v1 >> 4) & 0x0F0F0F0F;

        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_blocks + kbx];
            const float2 ds = __half22float2(xb.ds);
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
            sumi = dp4a_i8(lo0, xqs[iqs], sumi);
            sumi = dp4a_i8(hi0, xqs[iqs + 4], sumi);
            sumi = dp4a_i8(lo1, xqs[iqs + 1], sumi);
            sumi = dp4a_i8(hi1, xqs[iqs + 5], sumi);
            acc[t] += dm.x * ds.x * float(sumi) + 0.5f * dm.y * ds.y;
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

// `Qi = 8` (`VDR = 2`): 4 lanes cooperate on one `BlockQ8_0` (`iqs = 0, 2, 4, 6`),
// reading 100% coalesced 8-byte chunks!
__global__ void __launch_bounds__(128, 1) mmvq_q8_0_q8_1(
    const BlockQ8_0* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);

    if (n_tok == 1 && cols >= 1024) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockQ8_0* w_row = w + size_t(row) * n_blocks;
        const int tid = warp_id * 32 + lane;
        const int iqs = 2 * (tid & 3);
        float acc = 0.0f;
        for (int kbx = tid >> 2; kbx < n_blocks; kbx += 32) {
            const BlockQ8_0& wb = w_row[kbx];
            const BlockQ8_1& xb = xq[kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
            sumi = dp4a_i8(load_i32_2b(wb.qs + 4 * iqs), xqs[iqs], sumi);
            sumi = dp4a_i8(load_i32_2b(wb.qs + 4 * (iqs + 1)), xqs[iqs + 1], sumi);
            acc += __half2float(wb.d) * __low2float(xb.ds) * float(sumi);
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) {
            partial[warp_id - 1][lane] = acc;
        }
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out[row] = acc;
            }
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const BlockQ8_0* w_row = w + size_t(row) * n_blocks;
    const int iqs = 2 * (lane & 3);
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int kbx = lane >> 2; kbx < n_blocks; kbx += 8) {
        const BlockQ8_0& wb = w_row[kbx];
        const float d_w = __half2float(wb.d);
        const int w0 = load_i32_2b(wb.qs + 4 * iqs);
        const int w1 = load_i32_2b(wb.qs + 4 * (iqs + 1));
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_blocks + kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
            sumi = dp4a_i8(w0, xqs[iqs], sumi);
            sumi = dp4a_i8(w1, xqs[iqs + 1], sumi);
            acc[t] += d_w * __low2float(xb.ds) * float(sumi);
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

// `Q2_0 x Q8_1` MMVQ (`BlockQ2_0`: 64 weights in 18B = 2 x `BlockQ8_1`):
// 4 lanes cooperate on one `BlockQ2_0` (`sub = lane & 3`, reading 4 bytes = 16 weights = 4 `dp4a_i8` calls!).
__global__ void __launch_bounds__(128, 1) mmvq_q2_0_q8_1(
    const BlockQ2_0* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_b64 = cols >> 6;
    const int n_q8 = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;

    const BlockQ2_0* w_row = w + size_t(row) * n_b64;
    const int sub = lane & 3;         // 0..3 -> 16 weights each
    const int q8_half = sub >> 1;     // 0 or 1 (which of the two 32-elem BlockQ8_1s)
    const int q8_quad = (sub & 1) * 4;// 0 or 4 (int32 offset inside BlockQ8_1.qs)
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int b64 = lane >> 2; b64 < n_b64; b64 += 8) {
        const BlockQ2_0& wb = w_row[b64];
        const float d_w = __half2float(wb.d);
        const uint32_t raw4 = uint32_t(load_i32_2b(wb.qs + 4 * sub));
        const int s0 = unpack_q2_0_byte_s8x4(uint8_t(raw4 & 0xFFu));
        const int s1 = unpack_q2_0_byte_s8x4(uint8_t((raw4 >> 8) & 0xFFu));
        const int s2 = unpack_q2_0_byte_s8x4(uint8_t((raw4 >> 16) & 0xFFu));
        const int s3 = unpack_q2_0_byte_s8x4(uint8_t((raw4 >> 24) & 0xFFu));

        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_q8 + 2 * b64 + q8_half];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + q8_quad;
            int sumi = 0;
            sumi = dp4a_i8(s0, xqs[0], sumi);
            sumi = dp4a_i8(s1, xqs[1], sumi);
            sumi = dp4a_i8(s2, xqs[2], sumi);
            sumi = dp4a_i8(s3, xqs[3], sumi);
            acc[t] += d_w * __low2float(xb.ds) * float(sumi);
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}
// `Q5_0 x Q8_1` MMVQ (`BlockQ5_0`: 32 weights in 22B):
// `sub = lane & 3` (`j0 = sub * 4` in `0, 4, 8, 12`), 8 blocks per warp iteration!
__global__ void __launch_bounds__(128, 1) mmvq_q5_0_q8_1(
    const BlockQ5_0* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;

    const BlockQ5_0* w_row = w + size_t(row) * n_blocks;
    const int sub = lane & 3;
    const int j0 = sub * 4;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int kbx = lane >> 2; kbx < n_blocks; kbx += 8) {
        const BlockQ5_0& wb = w_row[kbx];
        const float d_w = __half2float(wb.d);
        const uint32_t qh = uint32_t(load_i32_2b(wb.qh));
        const uint32_t qs4 = uint32_t(load_i32_2b(wb.qs + j0));
        uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
            const int q0 = int((qb & 0x0Fu) | (((qh >> (j0 + k)) & 1u) << 4)) - 16;
            const int q1 = int((qb >> 4) | (((qh >> (16 + j0 + k)) & 1u) << 4)) - 16;
            w_lo |= uint32_t(uint8_t(int8_t(q0))) << (8 * k);
            w_hi |= uint32_t(uint8_t(int8_t(q1))) << (8 * k);
        }
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_blocks + kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[sub], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + sub], sumi);
            acc[t] += d_w * __low2float(xb.ds) * float(sumi);
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[size_t(t) * rows + row] = sum;
    }
}

// `IQ4_NL x Q8_1` MMVQ (`BlockIQ4NL`: 32 weights in 18B):
__global__ void __launch_bounds__(128, 1) mmvq_iq4_nl_q8_1(
    const BlockIQ4NL* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_blocks = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;

    const BlockIQ4NL* w_row = w + size_t(row) * n_blocks;
    const int sub = lane & 3;
    const int j0 = sub * 4;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int kbx = lane >> 2; kbx < n_blocks; kbx += 8) {
        const BlockIQ4NL& wb = w_row[kbx];
        const float d_w = __half2float(wb.d);
        const uint32_t qs4 = uint32_t(load_i32_2b(wb.qs + j0));
        uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
            w_lo |= uint32_t(uint8_t(c_kvalues_iq4nl[qb & 0x0Fu])) << (8 * k);
            w_hi |= uint32_t(uint8_t(c_kvalues_iq4nl[qb >> 4])) << (8 * k);
        }
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_blocks + kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[sub], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + sub], sumi);
            acc[t] += d_w * __low2float(xb.ds) * float(sumi);
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[size_t(t) * rows + row] = sum;
    }
}

// `Q3_K x Q8_1` MMVQ (`BlockQ3K`: 256 weights in 110B):
// `is = lane >> 1` (0..15 sub-blocks of 16), `part = lane & 1` (8 elements per lane -> 2 `dp4a_i8`!).
__global__ void __launch_bounds__(128, 1) mmvq_q3_k_q8_1(
    const BlockQ3K* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_sb = cols >> 8;
    const int n_q8 = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;

    const BlockQ3K* w_row = w + size_t(row) * n_sb;
    const int is = lane >> 1;            // 0..15
    const int part = lane & 1;           // 0..1
    const int half128 = is >> 3;         // 0..1
    const int j = (is >> 1) & 3;         // 0..3
    const int shift = 2 * j;
    const uint8_t m = uint8_t(1u << (4 * half128 + j));
    const int l0 = (is & 1) * 16 + part * 8; // 0, 8, 16, 24
    const int q8_sub = is >> 1;          // 0..7 (which 32-elem BlockQ8_1)
    const int q8_quad = (is & 1) * 4 + part * 2; // int32 index inside BlockQ8_1.qs (0, 2, 4, 6)
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int sb = 0; sb < n_sb; ++sb) {
        const BlockQ3K& blk = w_row[sb];
        const float d_all = __half2float(blk.d);
        const uint32_t a0 = uint32_t(load_i32_2b(blk.scales + 0));
        const uint32_t a1 = uint32_t(load_i32_2b(blk.scales + 4));
        const uint32_t tmp = uint32_t(load_i32_2b(blk.scales + 8));
        const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
        uint32_t aux[4];
        aux[0] = (a0 & km2) | ((tmp & km1) << 4);
        aux[1] = (a1 & km2) | (((tmp >> 2) & km1) << 4);
        aux[2] = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        const float dl = d_all * float(int(reinterpret_cast<const int8_t*>(aux)[is]) - 32);

        const uint8_t* qp = blk.qs + half128 * 32 + l0;
        const uint8_t* hp = blk.hmask + l0;
        uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int qv0 = int((qp[k] >> shift) & 3u) - ((hp[k] & m) ? 0 : 4);
            const int qv1 = int((qp[4 + k] >> shift) & 3u) - ((hp[4 + k] & m) ? 0 : 4);
            w0 |= uint32_t(uint8_t(int8_t(qv0))) << (8 * k);
            w1 |= uint32_t(uint8_t(int8_t(qv1))) << (8 * k);
        }
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_q8 + sb * 8 + q8_sub];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + q8_quad;
            int sumi = dp4a_i8(int(w0), xqs[0], 0);
            sumi = dp4a_i8(int(w1), xqs[1], sumi);
            acc[t] += dl * __low2float(xb.ds) * float(sumi);
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[size_t(t) * rows + row] = sum;
    }
}

// `IQ4_XS x Q8_1` MMVQ (`BlockIQ4XS`: 256 weights in 136B):
// `ib = lane >> 2` (0..7 sub-blocks of 32), `quad = lane & 3` (`j0 = quad * 4` in `0, 4, 8, 12`).
__global__ void __launch_bounds__(128, 4) mmvq_iq4_xs_q8_1(
    const BlockIQ4XS* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_sb = cols >> 8;
    const int n_q8 = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int ib = lane >> 2;
    const int quad = lane & 3;
    const int j0 = quad * 4;

    if (n_tok == 1 && cols >= 1024) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockIQ4XS* w_row = w + size_t(row) * n_sb;
        float acc = 0.0f;
        for (int sb = warp_id; sb < n_sb; sb += 4) {
            const BlockIQ4XS& blk = w_row[sb];
            const int ls = int((blk.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F)
                         | int(((blk.scales_h >> (2 * ib)) & 3u) << 4);
            const float dl = __half2float(blk.d) * float(ls - 32);
            const uint32_t qs4 = uint32_t(load_i32_2b(blk.qs + ib * 16 + j0));
            uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
                w_lo |= uint32_t(uint8_t(c_kvalues_iq4nl[qb & 0x0Fu])) << (8 * k);
                w_hi |= uint32_t(uint8_t(c_kvalues_iq4nl[qb >> 4])) << (8 * k);
            }
            const BlockQ8_1& xb = xq[sb * 8 + ib];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[quad], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + quad], sumi);
            acc += dl * __low2float(xb.ds) * float(sumi);
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;

    const BlockIQ4XS* w_row = w + size_t(row) * n_sb;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int sb = 0; sb < n_sb; ++sb) {
        const BlockIQ4XS& blk = w_row[sb];
        const int ls = int((blk.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F)
                     | int(((blk.scales_h >> (2 * ib)) & 3u) << 4);
        const float dl = __half2float(blk.d) * float(ls - 32);
        const uint32_t qs4 = uint32_t(load_i32_2b(blk.qs + ib * 16 + j0));
        uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
            w_lo |= uint32_t(uint8_t(c_kvalues_iq4nl[qb & 0x0Fu])) << (8 * k);
            w_hi |= uint32_t(uint8_t(c_kvalues_iq4nl[qb >> 4])) << (8 * k);
        }
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_q8 + sb * 8 + ib];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[quad], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + quad], sumi);
            acc[t] += dl * __low2float(xb.ds) * float(sumi);
        }
    }
    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[size_t(t) * rows + row] = sum;
    }
}

// Each superblock has 256 elements = 8 sub-blocks of 32 (`BlockQ8_1`).
// Across 32 lanes: `sub = lane >> 2` (0..7), `quad = lane & 3` (0..3, 4 bytes = 4 elements per quad!).
// Wait: 8 sub-blocks * 8 quads/sub-block = 64 quads per 256-elem superblock -> each lane handles 2 quads!

__global__ void __launch_bounds__(128, 4) mmvq_q4_k_q8_1(
    const BlockQ4K* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_sb = cols >> 8;
    const int n_q8 = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int chunk64 = lane >> 3;
    const int q_idx = lane & 7;

    if (n_tok == 1 && cols >= 1024) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockQ4K* w_row = w + size_t(row) * n_sb;
        float acc = 0.0f;
        for (int sb = warp_id; sb < n_sb; sb += 4) {
            const BlockQ4K& blk = w_row[sb];
            const float2 dm = __half22float2(blk.dm);
            uint8_t sc0, m0, sc1, m1;
            decode_q4k_scale_min(2 * chunk64 + 0, blk.scales, sc0, m0);
            decode_q4k_scale_min(2 * chunk64 + 1, blk.scales, sc1, m1);
            const int packed = reinterpret_cast<const int*>(blk.qs)[chunk64 * 8 + q_idx];
            const BlockQ8_1* x_sb = xq + sb * 8;
            const BlockQ8_1& xb0 = x_sb[2 * chunk64 + 0];
            const BlockQ8_1& xb1 = x_sb[2 * chunk64 + 1];
            const int dot0 = dp4a_i8(packed & 0x0F0F0F0F, reinterpret_cast<const int*>(xb0.qs)[q_idx], 0);
            const int dot1 = dp4a_i8((packed >> 4) & 0x0F0F0F0F, reinterpret_cast<const int*>(xb1.qs)[q_idx], 0);
            const float2 ds0 = __half22float2(xb0.ds);
            const float2 ds1 = __half22float2(xb1.ds);
            float term = dm.x * (float(sc0) * ds0.x * float(dot0) + float(sc1) * ds1.x * float(dot1));
            if (q_idx == 0) term -= dm.y * (float(m0) * ds0.y + float(m1) * ds1.y);
            acc += term;
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const BlockQ4K* w_row = w + size_t(row) * n_sb;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int sb = 0; sb < n_sb; ++sb) {
        const BlockQ4K& blk = w_row[sb];
        const float2 dm = __half22float2(blk.dm);
        uint8_t sc0, m0, sc1, m1;
        decode_q4k_scale_min(2 * chunk64 + 0, blk.scales, sc0, m0);
        decode_q4k_scale_min(2 * chunk64 + 1, blk.scales, sc1, m1);

        // `BlockQ4K` is 144B and `qs` is at byte offset 16 -> 4-byte aligned!
        const int packed = reinterpret_cast<const int*>(blk.qs)[chunk64 * 8 + q_idx];
        const int q_lo = packed & 0x0F0F0F0F;
        const int q_hi = (packed >> 4) & 0x0F0F0F0F;

        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1* x_sb = xq + size_t(t) * n_q8 + sb * 8;
            const BlockQ8_1& xb0 = x_sb[2 * chunk64 + 0];
            const BlockQ8_1& xb1 = x_sb[2 * chunk64 + 1];
            const int x0 = reinterpret_cast<const int*>(xb0.qs)[q_idx];
            const int x1 = reinterpret_cast<const int*>(xb1.qs)[q_idx];
            const int dot0 = dp4a_i8(q_lo, x0, 0);
            const int dot1 = dp4a_i8(q_hi, x1, 0);
            const float2 ds0 = __half22float2(xb0.ds);
            const float2 ds1 = __half22float2(xb1.ds);

            float term = dm.x * (float(sc0) * ds0.x * float(dot0) + float(sc1) * ds1.x * float(dot1));
            if (q_idx == 0) {
                term -= dm.y * (float(m0) * ds0.y + float(m1) * ds1.y);
            }
            acc[t] += term;
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

__global__ void __launch_bounds__(128, 1) mmvq_q5_k_q8_1(
    const BlockQ5K* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_sb = cols >> 8;
    const int n_q8 = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int chunk64 = lane >> 3;
    const int q_idx = lane & 7;

    if (n_tok == 1 && cols >= 1024) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockQ5K* w_row = w + size_t(row) * n_sb;
        float acc = 0.0f;
        for (int sb = warp_id; sb < n_sb; sb += 4) {
            const BlockQ5K& blk = w_row[sb];
            const float2 dm = __half22float2(blk.dm);
            uint8_t sc0, m0, sc1, m1;
            decode_q4k_scale_min(2 * chunk64 + 0, blk.scales, sc0, m0);
            decode_q4k_scale_min(2 * chunk64 + 1, blk.scales, sc1, m1);

            const int packed = reinterpret_cast<const int*>(blk.qs)[chunk64 * 8 + q_idx];
            const int qh4 = reinterpret_cast<const int*>(blk.qh)[q_idx];
            const int hi0 = ((qh4 >> (2 * chunk64)) & 0x01010101) << 4;
            const int hi1 = ((qh4 >> (2 * chunk64 + 1)) & 0x01010101) << 4;
            const int q0 = (packed & 0x0F0F0F0F) | hi0;
            const int q1 = ((packed >> 4) & 0x0F0F0F0F) | hi1;

            const BlockQ8_1* x_sb = xq + sb * 8;
            const BlockQ8_1& xb0 = x_sb[2 * chunk64 + 0];
            const BlockQ8_1& xb1 = x_sb[2 * chunk64 + 1];
            const int dot0 = dp4a_i8(q0, reinterpret_cast<const int*>(xb0.qs)[q_idx], 0);
            const int dot1 = dp4a_i8(q1, reinterpret_cast<const int*>(xb1.qs)[q_idx], 0);
            const float2 ds0 = __half22float2(xb0.ds);
            const float2 ds1 = __half22float2(xb1.ds);

            float term = dm.x * (float(sc0) * ds0.x * float(dot0) + float(sc1) * ds1.x * float(dot1));
            if (q_idx == 0) {
                term -= dm.y * (float(m0) * ds0.y + float(m1) * ds1.y);
            }
            acc += term;
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const BlockQ5K* w_row = w + size_t(row) * n_sb;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int sb = 0; sb < n_sb; ++sb) {
        const BlockQ5K& blk = w_row[sb];
        const float2 dm = __half22float2(blk.dm);
        uint8_t sc0, m0, sc1, m1;
        decode_q4k_scale_min(2 * chunk64 + 0, blk.scales, sc0, m0);
        decode_q4k_scale_min(2 * chunk64 + 1, blk.scales, sc1, m1);

        const int packed = reinterpret_cast<const int*>(blk.qs)[chunk64 * 8 + q_idx];
        const int qh4 = reinterpret_cast<const int*>(blk.qh)[q_idx];
        const int bit0 = 2 * chunk64;
        const int bit1 = 2 * chunk64 + 1;
        const int hi0 = ((qh4 >> bit0) & 0x01010101) << 4;
        const int hi1 = ((qh4 >> bit1) & 0x01010101) << 4;
        const int q0 = (packed & 0x0F0F0F0F) | hi0;
        const int q1 = ((packed >> 4) & 0x0F0F0F0F) | hi1;

        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1* x_sb = xq + size_t(t) * n_q8 + sb * 8;
            const BlockQ8_1& xb0 = x_sb[2 * chunk64 + 0];
            const BlockQ8_1& xb1 = x_sb[2 * chunk64 + 1];
            const int x0 = reinterpret_cast<const int*>(xb0.qs)[q_idx];
            const int x1 = reinterpret_cast<const int*>(xb1.qs)[q_idx];
            const int dot0 = dp4a_i8(q0, x0, 0);
            const int dot1 = dp4a_i8(q1, x1, 0);
            const float2 ds0 = __half22float2(xb0.ds);
            const float2 ds1 = __half22float2(xb1.ds);

            float term = dm.x * (float(sc0) * ds0.x * float(dot0) + float(sc1) * ds1.x * float(dot1));
            if (q_idx == 0) {
                term -= dm.y * (float(m0) * ds0.y + float(m1) * ds1.y);
            }
            acc[t] += term;
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

__global__ void __launch_bounds__(128, 1) mmvq_q6_k_q8_1(
    const BlockQ6K* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int n_sb = cols >> 8;
    const int n_q8 = cols >> 5;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);

    const int half128 = lane >> 4;
    const int sub_pair = (lane >> 3) & 1;
    const int q_idx = lane & 7;
    const int sc_half = q_idx >> 2;
    const int quad_a = sub_pair;
    const int quad_b = sub_pair + 2;

    if (n_tok == 1 && cols >= 2048) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const BlockQ6K* w_row = w + size_t(row) * n_sb;
        float acc = 0.0f;
        for (int sb = warp_id; sb < n_sb; sb += 4) {
            const BlockQ6K& blk = w_row[sb];
            const float d = __half2float(blk.d);
            const int ql4 = load_i32_2b(blk.ql + half128 * 64 + sub_pair * 32 + q_idx * 4);
            const int qh4 = load_i32_2b(blk.qh + half128 * 32 + q_idx * 4);
            const int8_t* sc_ptr = blk.scales + half128 * 8;

            const int qa_raw = (ql4 & 0x0F0F0F0F) | (((qh4 >> (2 * quad_a)) & 0x03030303) << 4);
            const int qb_raw = ((ql4 >> 4) & 0x0F0F0F0F) | (((qh4 >> (2 * quad_b)) & 0x03030303) << 4);
            const int qa = (qa_raw + 0x60606060) ^ 0x80808080;
            const int qb = (qb_raw + 0x60606060) ^ 0x80808080;

            const float sca = float(sc_ptr[sc_half + 2 * quad_a]);
            const float scb = float(sc_ptr[sc_half + 2 * quad_b]);
            const BlockQ8_1* x_sb = xq + sb * 8 + half128 * 4;
            const BlockQ8_1& xba = x_sb[quad_a];
            const BlockQ8_1& xbb = x_sb[quad_b];
            const int dota = dp4a_i8(qa, reinterpret_cast<const int*>(xba.qs)[q_idx], 0);
            const int dotb = dp4a_i8(qb, reinterpret_cast<const int*>(xbb.qs)[q_idx], 0);
            acc += d * (sca * __low2float(xba.ds) * float(dota) + scb * __low2float(xbb.ds) * float(dotb));
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const BlockQ6K* w_row = w + size_t(row) * n_sb;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int sb = 0; sb < n_sb; ++sb) {
        const BlockQ6K& blk = w_row[sb];
        const float d = __half2float(blk.d);
        const uint8_t* ql_ptr = blk.ql + half128 * 64 + sub_pair * 32 + q_idx * 4;
        const uint8_t* qh_ptr = blk.qh + half128 * 32 + q_idx * 4;
        const int8_t* sc_ptr = blk.scales + half128 * 8;

        const int ql4 = load_i32_2b(ql_ptr);
        const int qh4 = load_i32_2b(qh_ptr);
        const int qa_raw = (ql4 & 0x0F0F0F0F) | (((qh4 >> (2 * quad_a)) & 0x03030303) << 4);
        const int qb_raw = ((ql4 >> 4) & 0x0F0F0F0F) | (((qh4 >> (2 * quad_b)) & 0x03030303) << 4);
        const int qa = (qa_raw + 0x60606060) ^ 0x80808080;
        const int qb = (qb_raw + 0x60606060) ^ 0x80808080;

        const float sca = float(sc_ptr[sc_half + 2 * quad_a]);
        const float scb = float(sc_ptr[sc_half + 2 * quad_b]);

        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1* x_sb = xq + size_t(t) * n_q8 + sb * 8 + half128 * 4;
            const BlockQ8_1& xba = x_sb[quad_a];
            const BlockQ8_1& xbb = x_sb[quad_b];
            const int xa = reinterpret_cast<const int*>(xba.qs)[q_idx];
            const int xb = reinterpret_cast<const int*>(xbb.qs)[q_idx];
            const int dota = dp4a_i8(qa, xa, 0);
            const int dotb = dp4a_i8(qb, xb, 0);
            const float dxa = __low2float(xba.ds);
            const float dxb = __low2float(xbb.ds);
            acc[t] += d * (sca * dxa * float(dota) + scb * dxb * float(dotb));
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

// Unquantized F16 / BF16 / F32 GEMV & small-batch GEMM (`n_tok = 1..8`)
__global__ void __launch_bounds__(128, 4) mmvq_float_f32(
    const uint8_t* __restrict__ w_bytes,
    const float* __restrict__ x,
    float* __restrict__ out,
    int dtype,
    int rows,
    int cols,
    size_t row_bytes,
    int n_tok
) {
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);

    // 4-warp cooperative 128-bit vectorized fastpath for `n_tok == 1 && cols >= 1024` (`BF16`, `F16`, and `F32`!)
    if (n_tok == 1 && cols >= 1024 && (dtype == DTYPE_BF16 || dtype == DTYPE_F16 || dtype == DTYPE_F32) && (cols & 7) == 0 && ((uintptr_t(w_bytes) | row_bytes) & 15u) == 0u) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const int tid = warp_id * 32 + lane;
        float acc = 0.0f;
        if (dtype == DTYPE_F32) {
            const float4* w4 = reinterpret_cast<const float4*>(w_bytes + size_t(row) * row_bytes);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            const int n4 = cols >> 2;
            for (int i = tid; i < n4; i += 128) {
                const float4 wv = w4[i];
                const float4 xv = x4[i];
                acc = fmaf(wv.x, xv.x, acc);
                acc = fmaf(wv.y, xv.y, acc);
                acc = fmaf(wv.z, xv.z, acc);
                acc = fmaf(wv.w, xv.w, acc);
            }
        } else if (dtype == DTYPE_BF16) {
            const uint4* w8 = reinterpret_cast<const uint4*>(w_bytes + size_t(row) * row_bytes);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            const int n8 = cols >> 3;
            for (int i = tid; i < n8; i += 128) {
                const uint4 wv = w8[i];
                const float4 xa = x4[2 * i + 0];
                const float4 xb = x4[2 * i + 1];
                acc = fmaf(bf16_to_f32(uint16_t(wv.x & 0xFFFFu)), xa.x, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.x >> 16)), xa.y, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.y & 0xFFFFu)), xa.z, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.y >> 16)), xa.w, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.z & 0xFFFFu)), xb.x, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.z >> 16)), xb.y, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.w & 0xFFFFu)), xb.z, acc);
                acc = fmaf(bf16_to_f32(uint16_t(wv.w >> 16)), xb.w, acc);
            }
        } else {
            const uint4* w8 = reinterpret_cast<const uint4*>(w_bytes + size_t(row) * row_bytes);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            const int n8 = cols >> 3;
            for (int i = tid; i < n8; i += 128) {
                const uint4 wv = w8[i];
                const float4 xa = x4[2 * i + 0];
                const float4 xb = x4[2 * i + 1];
                const float2 h0 = __half22float2(*reinterpret_cast<const half2*>(&wv.x));
                const float2 h1 = __half22float2(*reinterpret_cast<const half2*>(&wv.y));
                const float2 h2 = __half22float2(*reinterpret_cast<const half2*>(&wv.z));
                const float2 h3 = __half22float2(*reinterpret_cast<const half2*>(&wv.w));
                acc = fmaf(h0.x, xa.x, acc);
                acc = fmaf(h0.y, xa.y, acc);
                acc = fmaf(h1.x, xa.z, acc);
                acc = fmaf(h1.y, xa.w, acc);
                acc = fmaf(h2.x, xb.x, acc);
                acc = fmaf(h2.y, xb.y, acc);
                acc = fmaf(h3.x, xb.z, acc);
                acc = fmaf(h3.y, xb.w, acc);
            }
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;

    const uint8_t* w_row = w_bytes + size_t(row) * row_bytes;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    if (dtype == DTYPE_F16) {
        const half2* wh2 = reinterpret_cast<const half2*>(w_row);
        const int n2 = cols >> 1;
        for (int i = lane; i < n2; i += 32) {
            const float2 wf = __half22float2(wh2[i]);
            for (int t = 0; t < n_tok; ++t) {
                const float* xt = x + size_t(t) * cols + 2 * i;
                acc[t] += wf.x * xt[0] + wf.y * xt[1];
            }
        }
    } else if (dtype == DTYPE_BF16) {
        const uint32_t* wb2 = reinterpret_cast<const uint32_t*>(w_row);
        const int n2 = cols >> 1;
        for (int i = lane; i < n2; i += 32) {
            const uint32_t p = wb2[i];
            const float w0 = bf16_to_f32(uint16_t(p & 0xFFFFu));
            const float w1 = bf16_to_f32(uint16_t(p >> 16));
            for (int t = 0; t < n_tok; ++t) {
                const float* xt = x + size_t(t) * cols + 2 * i;
                acc[t] += w0 * xt[0] + w1 * xt[1];
            }
        }
    } else {
        const float* wf = reinterpret_cast<const float*>(w_row);
        for (int c = lane; c < cols; c += 32) {
            const float wv = wf[c];
            for (int t = 0; t < n_tok; ++t) {
                acc[t] += wv * x[size_t(t) * cols + c];
            }
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            out[size_t(t) * rows + row] = sum;
        }
    }
}

// ============================================================================
// 3. Turing/Ampere+ `wmma` 16x16x16 FP16 Tensor Core Batched Prefill GEMM
// ============================================================================
// Computes `out[n_tok, rows] = x[n_tok, cols] @ W[rows, cols]^T` for ANY GGUF dtype!
// Tile: BM = 64 rows, BN = 32 tokens, BK = 32 cols.
// Block: (32, 4, 1) = 128 threads (4 warps). Each warp computes [16 rows x 32 tokens] = two 16x16 tiles!
// Grid: (ceil(rows / 64), ceil(n_tok / 32), 1)
__global__ void __launch_bounds__(128) wmma_gemm_mixed_f32(
    const uint8_t* __restrict__ w_bytes,
    const float* __restrict__ x,
    float* __restrict__ out,
    int dtype,
    int rows,
    int cols,
    size_t row_bytes,
    int n_tok
) {
    const int warp_id = int(threadIdx.y); // 0..3
    const int lane = int(threadIdx.x);    // 0..31
    const int tid = warp_id * 32 + lane;  // 0..127

    const int row_base = int(blockIdx.x) * 64;
    const int tok_base = int(blockIdx.y) * 32;

    __shared__ half smem_w[64 * 32]; // row-major [64, 32]
    __shared__ half smem_x[32 * 32]; // row-major [32, 32] (transposed into col_major for B!)
    __shared__ float smem_c[4 * 16 * 16]; // staging for boundary-safe output store

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);

    for (int k0 = 0; k0 < cols; k0 += 32) {
        // 1. Dequantize [64 rows x 32 cols] of W into `smem_w`
        if (dtype == DTYPE_Q4_0) {
            // Fast vectorized Q4_0 block unpack: 2 threads per row (`r_local = tid >> 1`, `half_id = tid & 1`)
            const int r_local = tid >> 1;
            const int half_id = tid & 1;
            const int r = row_base + r_local;
            half* dst = smem_w + r_local * 32;
            if (r < rows) {
                const BlockQ4_0& blk = reinterpret_cast<const BlockQ4_0*>(w_bytes + size_t(r) * row_bytes)[k0 >> 5];
                const float d = __half2float(blk.d);
                const uint8_t* qs = blk.qs + half_id * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = qs[j];
                    dst[half_id * 8 + j] = __float2half(d * float(int(byte & 0x0F) - 8));
                    dst[16 + half_id * 8 + j] = __float2half(d * float(int(byte >> 4) - 8));
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dst[half_id * 8 + j] = __float2half(0.0f);
                    dst[16 + half_id * 8 + j] = __float2half(0.0f);
                }
            }
        } else if (dtype == DTYPE_Q4_1) {
            const int r_local = tid >> 1;
            const int half_id = tid & 1;
            const int r = row_base + r_local;
            half* dst = smem_w + r_local * 32;
            if (r < rows) {
                const BlockQ4_1& blk = reinterpret_cast<const BlockQ4_1*>(w_bytes + size_t(r) * row_bytes)[k0 >> 5];
                const float2 dm = __half22float2(blk.dm);
                const uint8_t* qs = blk.qs + half_id * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = qs[j];
                    dst[half_id * 8 + j] = __float2half(dm.x * float(byte & 0x0F) + dm.y);
                    dst[16 + half_id * 8 + j] = __float2half(dm.x * float(byte >> 4) + dm.y);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dst[half_id * 8 + j] = __float2half(0.0f);
                    dst[16 + half_id * 8 + j] = __float2half(0.0f);
                }
            }
        } else {
#pragma unroll
            for (int idx = tid; idx < 64 * 32; idx += 128) {
                const int r_local = idx >> 5;
                const int c_local = idx & 31;
                const int r = row_base + r_local;
                float val = 0.0f;
                if (r < rows) {
                    val = dequant_elem(dtype, w_bytes + size_t(r) * row_bytes, k0 + c_local);
                }
                smem_w[idx] = __float2half(val);
            }
        }

        // 2. Load [32 tokens x 32 cols] of X into `smem_x` (1024 elems / 128 threads = 8 elems/thread)
#pragma unroll
        for (int idx = tid; idx < 32 * 32; idx += 128) {
            const int t_local = idx >> 5;
            const int c_local = idx & 31;
            const int t = tok_base + t_local;
            const float val = (t < n_tok) ? x[size_t(t) * cols + k0 + c_local] : 0.0f;
            smem_x[idx] = __float2half(val);
        }
        __syncthreads();

        // 3. Two k-steps of 16 (`kk = 0, 16`): A is `smem_w` [16x16, row_major, ld=32],
        //    B is `smem_x^T` [16x16, col_major, ld=32]!
        //    So C = A * B = W_sub[16, 32] * X_sub[32, 32]^T -> C[16 rows, 32 tokens]!
#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b0, frag_b1;
            wmma::load_matrix_sync(frag_a, smem_w + (warp_id * 16) * 32 + kk, 32);
            wmma::load_matrix_sync(frag_b0, smem_x + 0 * 32 + kk, 32);
            wmma::load_matrix_sync(frag_b1, smem_x + 16 * 32 + kk, 32);
            wmma::mma_sync(acc0, frag_a, frag_b0, acc0);
            wmma::mma_sync(acc1, frag_a, frag_b1, acc1);
        }
        __syncthreads();
    }

    // 4. Store C[16 rows, 32 tokens] transposed into `out[t * rows + r]`!
    float* warp_smem = smem_c + warp_id * 256;
    wmma::store_matrix_sync(warp_smem, acc0, 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 256; idx += 32) {
        const int r = row_base + warp_id * 16 + (idx >> 4);
        const int t = tok_base + (idx & 15);
        if (r < rows && t < n_tok) {
            out[size_t(t) * rows + r] = warp_smem[idx];
        }
    }
    __syncwarp();
    wmma::store_matrix_sync(warp_smem, acc1, 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = lane; idx < 256; idx += 32) {
        const int r = row_base + warp_id * 16 + (idx >> 4);
        const int t = tok_base + 16 + (idx & 15);
        if (r < rows && t < n_tok) {
            out[size_t(t) * rows + r] = warp_smem[idx];
        }
    }
}

// ============================================================================
// 4. Token Embedding, Norms, Activations & Elementwise Ops
// ============================================================================

__device__ __forceinline__ float accurate_tanh_f32(float x) {
    const float ax = fabsf(x);
    if (ax < 0.0625f) {
        const float x2 = x * x;
        return x * fmaf(x2, fmaf(x2, 2.0f / 15.0f, -1.0f / 3.0f), 1.0f);
    }
    if (ax > 10.0f) {
        return (x >= 0.0f) ? 1.0f : -1.0f;
    }
    const float e = expf(-2.0f * ax);
    const float t = (1.0f - e) / (1.0f + e);
    return (x >= 0.0f) ? t : -t;
}

__device__ __forceinline__ float block256_reduce_sum(float v, int tid, float warp_sums[8]) {
    v = warp_reduce_sum(v);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
    __syncthreads();
    if (tid < 32) {
        const float s = (tid < 8) ? warp_sums[tid] : 0.0f;
        const float tot = warp_reduce_sum(s);
        if (tid == 0) warp_sums[0] = tot;
    }
    __syncthreads();
    const float res = warp_sums[0];
    __syncthreads();
    return res;
}

// Grid: (ceil(n_embd / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void embed_tokens_f32(
    const uint8_t* __restrict__ emb_bytes,
    const uint32_t* __restrict__ token_ids,
    float* __restrict__ out,
    int dtype,
    int n_embd,
    size_t row_bytes,
    float emb_scale
) {
    const int col = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int tok_idx = int(blockIdx.y);
    if (col >= n_embd) return;
    const uint32_t tid = token_ids[tok_idx];
    const uint8_t* row = emb_bytes + size_t(tid) * row_bytes;
    out[size_t(tok_idx) * n_embd + col] = dequant_elem(dtype, row, col) * emb_scale;
}

// RMSNorm per row: Grid (n_tok, 1, 1), Block (256, 1, 1)
__global__ void rms_norm_rows_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int dim,
    float eps
) {
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const float* x_row = x + size_t(row) * dim;
    float* out_row = out + size_t(row) * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += int(blockDim.x)) {
        const float v = x_row[i];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();
    if (tid < 32) {
        const float s = (tid < (int(blockDim.x) >> 5)) ? warp_sums[tid] : 0.0f;
        const float total = warp_reduce_sum(s);
        if (tid == 0) warp_sums[0] = rsqrtf(total / float(dim) + eps);
    }
    __syncthreads();
    const float scale = warp_sums[0];
    for (int i = tid; i < dim; i += int(blockDim.x)) {
        out_row[i] = x_row[i] * scale * w[i];
    }
}

// Per-head (or full-vector) QK RMSNorm in-place: Grid (n_heads, n_tok, 1), Block (32, 1, 1)
__global__ void qk_rms_norm_inplace_f32(
    float* __restrict__ qk,
    const float* __restrict__ w,
    int n_heads,
    int norm_dim,
    float eps
) {
    const int h = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row_stride = n_heads * norm_dim;
    float* vec = qk + size_t(t) * row_stride + size_t(h) * norm_dim;

    float sq = 0.0f;
    for (int i = lane; i < norm_dim; i += 32) {
        const float v = vec[i];
        sq += v * v;
    }
    sq = warp_reduce_sum(sq);
    const float scale = rsqrtf(sq / float(norm_dim) + eps);
    for (int i = lane; i < norm_dim; i += 32) {
        vec[i] *= scale * w[i];
    }
}

__global__ void add_bias_rows_f32(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int dim,
    int n_tok
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(n_tok) * dim;
    if (idx >= total) return;
    x[idx] += bias[idx % dim];
}

__global__ void add_scaled_inplace_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    float scale,
    size_t n
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += scale * src[idx];
}

__global__ void scale_inplace_f32(float* __restrict__ dst, float scale, size_t n) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= scale;
}

// Strided 2D row copy (`dst[t * dst_stride + i] = src[t * src_stride + i]` for `i in 0..cols`, `t in 0..n_tok`):
// Grid: (ceil(cols / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void copy_2d_rows_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int cols,
    size_t dst_stride,
    size_t src_stride
) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int t = int(blockIdx.y);
    if (i < cols) dst[size_t(t) * dst_stride + i] = src[size_t(t) * src_stride + i];
}

// SwiGLU: `gate[i] = silu(gate[i]) * up[i]`
__global__ void swiglu_f32(
    float* __restrict__ gate,
    const float* __restrict__ up,
    size_t n
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = gate[idx];
    gate[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

// Split fused `[gate, up]` row-wise `[n_tok, 2 * n_ff]` -> `out[n_tok, n_ff] = silu(gate) * up`
__global__ void swiglu_fused_f32(
    const float* __restrict__ gate_up,
    float* __restrict__ out,
    int n_ff,
    int n_tok
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(n_tok) * n_ff;
    if (idx >= total) return;
    const int t = int(idx / n_ff);
    const int c = int(idx % n_ff);
    const float* row = gate_up + size_t(t) * (2 * n_ff);
    const float g = row[c];
    out[idx] = (g / (1.0f + expf(-g))) * row[n_ff + c];
}

// Sigmoid gate multiply: `dst[i] *= sigmoid(gate[i])`
__global__ void sigmoid_gate_mul_f32(
    float* __restrict__ dst,
    const float* __restrict__ gate,
    size_t n
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = gate[idx];
    dst[idx] *= 1.0f / (1.0f + expf(-g));
}

// MTP / DFly2 dual RMSNorm + concat: `out[t, 0..dim] = rms_norm(a[t], wa, eps)`, `out[t, dim..2*dim] = rms_norm(b[t], wb, eps)`
// Grid: (2, n_rows, 1), Block: (256, 1, 1)
__global__ void mtp_rms_norm_cat2_f32(
    const float* __restrict__ a,
    const float* __restrict__ wa,
    const float* __restrict__ b,
    const float* __restrict__ wb,
    float* __restrict__ out,
    int dim,
    float eps
) {
    const int part = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    __shared__ float warp_sums[8];

    const float* src = (part == 0) ? (a + size_t(row) * dim) : (b + size_t(row) * dim);
    const float* w = (part == 0) ? wa : wb;
    float* dst = out + size_t(row) * (2 * dim) + size_t(part) * dim;

    float ss = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        const float v = src[i];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(dim) + eps);
    for (int i = tid; i < dim; i += 256) {
        dst[i] = src[i] * inv * w[i];
    }
}

// Elementwise fused multiply-add: `dst[i] += a[i] * b[i]`
__global__ void elemwise_mul_add_f32(
    float* __restrict__ dst,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = fmaf(a[idx], b[idx], dst[idx]);
}

// Single-scalar bias + sigmoid store into `probs_out[slot]` (for DFly2 confidence head):
// Grid: (1, 1, 1), Block: (1, 1, 1)
__global__ void dfly_bias_sigmoid_slot_f32(
    const float* __restrict__ logit,
    const float* __restrict__ bias,
    float* __restrict__ probs_out,
    int slot
) {
    probs_out[slot] = 1.0f / (1.0f + expf(-(logit[0] + bias[0])));
}

// Per-row scalar sigmoid gate multiply (`dst[t * cols + c] *= sigmoid(gate[t])`) — stream-ordered, CUDA-Graph safe!
// Grid: (ceil(cols / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void scale_rows_by_sigmoid_scalar_f32(
    float* __restrict__ dst,
    const float* __restrict__ gate,
    int cols,
    int n_tok
) {
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int t = int(blockIdx.y);
    if (c >= cols || t >= n_tok) return;
    const float g = gate[t];
    const float sig = 1.0f / (1.0f + expf(-g));
    dst[size_t(t) * cols + c] *= sig;
}

// Split Qwen3.5 interleaved `[n_head * 2 * head_dim]` (`[q_h, gate_h]` per head!) into `q` and `gate`!
__global__ void split_qwen35_q_gate_f32(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    int n_head,
    int head_dim,
    int n_tok
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int q_dim = n_head * head_dim;
    const size_t total = size_t(n_tok) * q_dim;
    if (idx >= total) return;
    const int t = int(idx / q_dim);
    const int rem = int(idx % q_dim);
    const int h = rem / head_dim;
    const int d = rem % head_dim;
    const float* src = qg + size_t(t) * (2 * q_dim) + size_t(h) * (2 * head_dim);
    q[idx] = src[d];
    gate[idx] = src[head_dim + d];
}

// ============================================================================
// 5. RoPE (NeoX & Norm) + F16 / Q8_0 KV Cache Write
// ============================================================================

// Applies RoPE to Q and K for `[n_tok]` tokens starting at `start_pos`.
// `rope_type`: 0 = Norm (adjacent pairs `2*i, 2*i+1`), 2 = NeoX (split half `i, i + rope_dim/2`), -1 = NoPE!
// Grid: (ceil(max(q_dim, kv_dim) / 128), n_tok, 1), Block: (128, 1, 1)
__global__ void apply_rope_qk_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ inv_freqs,
    int n_head,
    int n_head_kv,
    int head_dim,
    int rope_dim,
    int rope_type,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    if (rope_type < 0 || rope_dim <= 0) return;
    const int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int tok = int(blockIdx.y);
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int half_rope = rope_dim >> 1;
    const int q_pairs = n_head * half_rope;
    const int k_pairs = n_head_kv * half_rope;

    if (idx < q_pairs) {
        const int h = idx / half_rope;
        const int pair = idx % half_rope;
        const float angle = float(pos) * inv_freqs[pair];
        float s, c;
        sincosf(angle, &s, &c);
        float* qh = q + (size_t(tok) * n_head + h) * head_dim;
        if (rope_type == 2) {
            const float x0 = qh[pair];
            const float x1 = qh[pair + half_rope];
            qh[pair] = x0 * c - x1 * s;
            qh[pair + half_rope] = x0 * s + x1 * c;
        } else {
            const float x0 = qh[2 * pair];
            const float x1 = qh[2 * pair + 1];
            qh[2 * pair] = x0 * c - x1 * s;
            qh[2 * pair + 1] = x0 * s + x1 * c;
        }
    }

    if (idx < k_pairs) {
        const int h = idx / half_rope;
        const int pair = idx % half_rope;
        const float angle = float(pos) * inv_freqs[pair];
        float s, c;
        sincosf(angle, &s, &c);
        float* kh = k + (size_t(tok) * n_head_kv + h) * head_dim;
        if (rope_type == 2) {
            const float x0 = kh[pair];
            const float x1 = kh[pair + half_rope];
            kh[pair] = x0 * c - x1 * s;
            kh[pair + half_rope] = x0 * s + x1 * c;
        } else {
            const float x0 = kh[2 * pair];
            const float x1 = kh[2 * pair + 1];
            kh[2 * pair] = x0 * c - x1 * s;
            kh[2 * pair + 1] = x0 * s + x1 * c;
        }
    }
}

// Writes K and V `[n_tok, kv_dim]` into `F16` KV cache at `start_pos .. start_pos + n_tok`
__global__ void write_kv_cache_f16(
    const float* __restrict__ k,
    const float* __restrict__ v,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int kv_dim,
    int start_pos,
    int n_tok,
    const int* __restrict__ pos_dev
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(n_tok) * kv_dim;
    if (idx >= total) return;
    const int tok = int(idx / kv_dim);
    const int col = int(idx % kv_dim);
    const int base_pos = pos_dev ? (*pos_dev) : start_pos;
    const size_t dst = size_t(base_pos + tok) * kv_dim + col;
    k_cache[dst] = __float2half(k[idx]);
    v_cache[dst] = __float2half(v[idx]);
}

// Writes K and V `[n_tok, kv_dim]` into `Q8_0` KV cache (`BlockQ8_0`)!
// Grid: (kv_dim / 32, n_tok, 1), Block: (32, 1, 1)
__global__ void write_kv_cache_q8_0(
    const float* __restrict__ k,
    const float* __restrict__ v,
    BlockQ8_0* __restrict__ k_cache,
    BlockQ8_0* __restrict__ v_cache,
    int kv_dim,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int blk = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int n_blk = kv_dim >> 5;

    const float kv_k = k[size_t(tok) * kv_dim + blk * 32 + lane];
    const float kv_v = v[size_t(tok) * kv_dim + blk * 32 + lane];

    const float amax_k = warp_reduce_max(fabsf(kv_k));
    const float dk = amax_k / 127.0f;
    const float idk = (dk != 0.0f) ? (1.0f / dk) : 0.0f;
    const int8_t qk = int8_t(max(-127, min(127, int(roundf(kv_k * idk)))));

    const float amax_v = warp_reduce_max(fabsf(kv_v));
    const float dv = amax_v / 127.0f;
    const float idv = (dv != 0.0f) ? (1.0f / dv) : 0.0f;
    const int8_t qv = int8_t(max(-127, min(127, int(roundf(kv_v * idv)))));

    const int base_pos = pos_dev ? (*pos_dev) : start_pos;
    const size_t dst_idx = size_t(base_pos + tok) * n_blk + blk;
    k_cache[dst_idx].qs[lane] = qk;
    v_cache[dst_idx].qs[lane] = qv;
    if (lane == 0) {
        k_cache[dst_idx].d = __float2half(dk);
        v_cache[dst_idx].d = __float2half(dv);
    }
}

// ============================================================================
// 6. Online-Softmax FlashDecoding & Causal Batched GQA Attention (`F16` & `Q8_0` KV)
// ============================================================================
// Grid: (n_head, n_tok, 1), Block: (32, 4, 1) = 4 warps cooperating across KV positions!
// Supports arbitrary `head_dim <= 256` (divisible by 2), GQA (`n_head / n_head_kv`),
// sliding-window attention (`swa_window`), custom `attn_scale`, and `softcap`.
__global__ void __launch_bounds__(128) flash_attn_gqa_f16_kv(
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int start_pos,
    int swa_window,
    float attn_scale,
    float softcap,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int warp_id = int(threadIdx.y); // 0..3
    const int lane = int(threadIdx.x);    // 0..31

    const int kv_h = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * head_dim;
    const int cur_pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int t_start = (swa_window > 0 && cur_pos >= swa_window) ? (cur_pos - swa_window + 1) : 0;

    const float* q_vec = q + (size_t(tok) * n_head + h) * head_dim;
    // Up to 256 head_dim -> max 8 elements per lane!
    float q_reg[8] = {0.0f};
    float o_reg[8] = {0.0f};
    const int elems_per_lane = (head_dim + 31) >> 5;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = e * 32 + lane;
        if (d < head_dim) q_reg[e] = q_vec[d];
    }

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    for (int p = t_start + warp_id; p <= cur_pos; p += 4) {
        const half* kp = k_cache + size_t(p) * kv_dim + size_t(kv_h) * head_dim;
        float dot = 0.0f;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = e * 32 + lane;
            if (d < head_dim) {
                dot += q_reg[e] * __half2float(kp[d]);
            }
        }
        dot = warp_reduce_sum(dot) * attn_scale;
        if (softcap > 0.0f) {
            dot = softcap * tanhf(dot / softcap);
        }

        const float m_new = fmaxf(m_i, dot);
        const float alpha = expf(m_i - m_new);
        const float weight = expf(dot - m_new);
        l_i = l_i * alpha + weight;
        m_i = m_new;

        const half* vp = v_cache + size_t(p) * kv_dim + size_t(kv_h) * head_dim;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = e * 32 + lane;
            if (d < head_dim) {
                o_reg[e] = o_reg[e] * alpha + weight * __half2float(vp[d]);
            }
        }
    }

    // Merge the 4 warps' `(m_i, l_i, o_reg)` online softmax states via shared memory
    __shared__ float smem_m[4];
    __shared__ float smem_l[4];
    __shared__ float smem_o[4 * 256];

    if (lane == 0) {
        smem_m[warp_id] = m_i;
        smem_l[warp_id] = l_i;
    }
    for (int e = 0; e < elems_per_lane; ++e) {
        const int d = e * 32 + lane;
        if (d < head_dim) {
            smem_o[warp_id * 256 + d] = o_reg[e];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        const float m_max = fmaxf(fmaxf(smem_m[0], smem_m[1]), fmaxf(smem_m[2], smem_m[3]));
        float l_sum = 0.0f;
        float w_scale[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            w_scale[w] = (smem_l[w] > 0.0f) ? expf(smem_m[w] - m_max) : 0.0f;
            l_sum += smem_l[w] * w_scale[w];
        }
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* out_vec = out + (size_t(tok) * n_head + h) * head_dim;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = e * 32 + lane;
            if (d < head_dim) {
                const float acc = smem_o[0 * 256 + d] * w_scale[0]
                                + smem_o[1 * 256 + d] * w_scale[1]
                                + smem_o[2 * 256 + d] * w_scale[2]
                                + smem_o[3 * 256 + d] * w_scale[3];
                out_vec[d] = acc * inv_l;
            }
        }
    }
}

__global__ void __launch_bounds__(128) flash_attn_gqa_q8_0_kv(
    const float* __restrict__ q,
    const BlockQ8_0* __restrict__ k_cache,
    const BlockQ8_0* __restrict__ v_cache,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int start_pos,
    int swa_window,
    float attn_scale,
    float softcap,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int warp_id = int(threadIdx.y);
    const int lane = int(threadIdx.x);

    const int kv_h = h / (n_head / n_head_kv);
    const int kv_blks_per_tok = (n_head_kv * head_dim) >> 5;
    const int head_blks = head_dim >> 5;
    const int cur_pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int t_start = (swa_window > 0 && cur_pos >= swa_window) ? (cur_pos - swa_window + 1) : 0;

    const float* q_vec = q + (size_t(tok) * n_head + h) * head_dim;
    float q_reg[8] = {0.0f};
    float o_reg[8] = {0.0f};
    for (int b = 0; b < head_blks; ++b) {
        q_reg[b] = q_vec[b * 32 + lane];
    }

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    for (int p = t_start + warp_id; p <= cur_pos; p += 4) {
        const BlockQ8_0* kp = k_cache + size_t(p) * kv_blks_per_tok + size_t(kv_h) * head_blks;
        float dot = 0.0f;
        for (int b = 0; b < head_blks; ++b) {
            dot += q_reg[b] * (__half2float(kp[b].d) * float(kp[b].qs[lane]));
        }
        dot = warp_reduce_sum(dot) * attn_scale;
        if (softcap > 0.0f) {
            dot = softcap * tanhf(dot / softcap);
        }

        const float m_new = fmaxf(m_i, dot);
        const float alpha = expf(m_i - m_new);
        const float weight = expf(dot - m_new);
        l_i = l_i * alpha + weight;
        m_i = m_new;

        const BlockQ8_0* vp = v_cache + size_t(p) * kv_blks_per_tok + size_t(kv_h) * head_blks;
        for (int b = 0; b < head_blks; ++b) {
            o_reg[b] = o_reg[b] * alpha + weight * (__half2float(vp[b].d) * float(vp[b].qs[lane]));
        }
    }

    __shared__ float smem_m[4];
    __shared__ float smem_l[4];
    __shared__ float smem_o[4 * 256];

    if (lane == 0) {
        smem_m[warp_id] = m_i;
        smem_l[warp_id] = l_i;
    }
    for (int b = 0; b < head_blks; ++b) {
        smem_o[warp_id * 256 + b * 32 + lane] = o_reg[b];
    }
    __syncthreads();

    if (warp_id == 0) {
        const float m_max = fmaxf(fmaxf(smem_m[0], smem_m[1]), fmaxf(smem_m[2], smem_m[3]));
        float l_sum = 0.0f;
        float w_scale[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            w_scale[w] = (smem_l[w] > 0.0f) ? expf(smem_m[w] - m_max) : 0.0f;
            l_sum += smem_l[w] * w_scale[w];
        }
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* out_vec = out + (size_t(tok) * n_head + h) * head_dim;
        for (int b = 0; b < head_blks; ++b) {
            const int d = b * 32 + lane;
            const float acc = smem_o[0 * 256 + d] * w_scale[0]
                            + smem_o[1 * 256 + d] * w_scale[1]
                            + smem_o[2 * 256 + d] * w_scale[2]
                            + smem_o[3 * 256 + d] * w_scale[3];
            out_vec[d] = acc * inv_l;
        }
    }
}

// ============================================================================
// 7. Qwen3.5 Gated DeltaNet (`GDN_CONTRACT` Compliant!)
// ============================================================================
// Strictly enforces all 5 `GDN_CONTRACT` invariants:
//   1. SiLU applied on `conv1d` output BEFORE splitting into q, k, v;
//   2. `q` L2-normalized and scaled by `1 / sqrt(S)`;
//   3. `k` L2-normalized; `v` NOT L2-normalized;
//   4. Key head index is `vh % h_k` (opposite of FLA!);
//   5. Output gate uses `rms_norm(out) * silu(z)`.

// Single-step causal Conv1d + SiLU + split into `(q_raw, k_raw, v)`:
// Grid: (ceil(conv_dim / 256), 1, 1), Block: (256, 1, 1)
__global__ void gdn_conv1d_silu_split_step_f32(
    const float* __restrict__ qkv_in,
    float* __restrict__ conv_hist,    // [hist_len, conv_dim], where hist_len = d_conv - 1
    const float* __restrict__ conv_w, // [conv_dim, d_conv]
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int conv_dim,
    int d_conv,
    int k_dim
) {
    const int ch = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (ch >= conv_dim) return;
    const int hist_len = d_conv - 1;
    const float x_cur = qkv_in[ch];
    const float* w_ch = conv_w + size_t(ch) * d_conv;

    float sum = 0.0f;
    for (int k = 0; k < hist_len; ++k) {
        sum += conv_hist[size_t(k) * conv_dim + ch] * w_ch[k];
    }
    sum += x_cur * w_ch[hist_len];

    // Shift history left and append `x_cur`
    for (int k = 0; k < hist_len - 1; ++k) {
        conv_hist[size_t(k) * conv_dim + ch] = conv_hist[size_t(k + 1) * conv_dim + ch];
    }
    if (hist_len > 0) {
        conv_hist[size_t(hist_len - 1) * conv_dim + ch] = x_cur;
    }

    // SiLU BEFORE splitting into q / k / v (`GDN_CONTRACT` #1)
    const float act = sum / (1.0f + expf(-sum));
    if (ch < k_dim) {
        q_out[ch] = act;
    } else if (ch < 2 * k_dim) {
        k_out[ch - k_dim] = act;
    } else {
        v_out[ch - 2 * k_dim] = act;
    }
}

// L2 normalize Q (`* 1/sqrt(S)`) and K per head (`S = state_size`, e.g. 128):
// Grid: (h_k, 1, 1), Block: (32, 1, 1)
__global__ void gdn_l2_norm_qk_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    int s_dim,
    float eps
) {
    const int hk = int(blockIdx.x);
    const int lane = int(threadIdx.x);
    float* qh = q + size_t(hk) * s_dim;
    float* kh = k + size_t(hk) * s_dim;

    float sq_q = 0.0f, sq_k = 0.0f;
    for (int i = lane; i < s_dim; i += 32) {
        sq_q += qh[i] * qh[i];
        sq_k += kh[i] * kh[i];
    }
    sq_q = warp_reduce_sum(sq_q);
    sq_k = warp_reduce_sum(sq_k);
    const float inv_q = rsqrtf(sq_q + eps) * rsqrtf(float(s_dim));
    const float inv_k = rsqrtf(sq_k + eps);
    for (int i = lane; i < s_dim; i += 32) {
        qh[i] *= inv_q;
        kh[i] *= inv_k;
    }
}

// Computes GDN `g[hv] = -exp(ssm_a[hv]) * softplus(alpha[hv] + dt_bias[hv])` and `beta[hv] = sigmoid(beta_raw[hv])`
__global__ void gdn_gates_f32(
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ g_out,
    float* __restrict__ beta_out,
    int h_v
) {
    const int hv = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (hv >= h_v) return;
    const float x = alpha[hv] + dt_bias[hv];
    const float sp = (x > 20.0f) ? x : log1pf(expf(x));
    // Note: in GGUF `ssm_a` is already stored as `-exp(A_log)` (negative)! Check `qwen35.rs`:
    // Wait: let's match `qwen35.rs` (`ssm_a[hv] * softplus(...)`)!
    g_out[hv] = ssm_a[hv] * sp;
    beta_out[hv] = 1.0f / (1.0f + expf(-beta_raw[hv]));
}

// Warp-parallel GDN recurrence step (`state` layout `[h_v, s_dim, dv_dim]`, matching `tm-models` & `tm-metal`!):
// Each warp (`threadIdx.x = 0..31`) owns ONE `(hv, col)` column over `i = 0..s_dim-1`!
// Grid: (h_v, ceil(dv_dim / 4), 1), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 2) gdn_recurrence_step_f32(
    float* __restrict__ state,      // [h_v, s_dim, dv_dim]
    const float* __restrict__ q,    // [h_k, s_dim] (already L2-normed * 1/sqrt(S))
    const float* __restrict__ k,    // [h_k, s_dim] (already L2-normed)
    const float* __restrict__ v,    // [h_v, dv_dim]
    const float* __restrict__ g,    // [h_v]
    const float* __restrict__ beta, // [h_v]
    float* __restrict__ out,        // [h_v, dv_dim]
    int h_k,
    int h_v,
    int s_dim,
    int dv_dim
) {
    const int hv = int(blockIdx.x);
    const int col = int(blockIdx.y) * 4 + int(threadIdx.y);
    const int lane = int(threadIdx.x);
    if (hv >= h_v || col >= dv_dim) return;

    const int kh = hv % h_k; // `GDN_CONTRACT` #4: `vh % HK`!
    const float* q_h = q + size_t(kh) * s_dim;
    const float* k_h = k + size_t(kh) * s_dim;
    float* st_h = state + size_t(hv) * s_dim * dv_dim + col;

    const float decay = expf(g[hv]);
    const float b_val = beta[hv];
    const float v_col = v[size_t(hv) * dv_dim + col];

    // Fast 100%-register-resident compile-time unrolled path for standard `S = 128` (zero local-memory spills!)
    if (s_dim == 128) {
        float s_reg[4], k_reg[4], q_reg[4];
        float kv_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r * 32 + lane;
            const float sv = st_h[size_t(i) * dv_dim] * decay;
            const float kv = k_h[i];
            s_reg[r] = sv;
            k_reg[r] = kv;
            q_reg[r] = q_h[i];
            kv_partial += sv * kv;
        }
        const float kv_mem = warp_reduce_sum(kv_partial);
        const float delta = (v_col - kv_mem) * b_val;
        float out_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = r * 32 + lane;
            const float s_new = s_reg[r] + k_reg[r] * delta;
            st_h[size_t(i) * dv_dim] = s_new;
            out_partial += s_new * q_reg[r];
        }
        const float y_col = warp_reduce_sum(out_partial);
        if (lane == 0) {
            out[size_t(hv) * dv_dim + col] = y_col;
        }
        return;
    }

    float kv_partial = 0.0f;
    for (int i = lane; i < s_dim; i += 32) {
        kv_partial += (st_h[size_t(i) * dv_dim] * decay) * k_h[i];
    }
    const float kv_mem = warp_reduce_sum(kv_partial);
    const float delta = (v_col - kv_mem) * b_val;

    float out_partial = 0.0f;
    for (int i = lane; i < s_dim; i += 32) {
        const float s_new = st_h[size_t(i) * dv_dim] * decay + k_h[i] * delta;
        st_h[size_t(i) * dv_dim] = s_new;
        out_partial += s_new * q_h[i];
    }
    const float y_col = warp_reduce_sum(out_partial);
    if (lane == 0) {
        out[size_t(hv) * dv_dim + col] = y_col;
    }
}

// Gated RMSNorm per value head: `out[hv, d] = rms_norm(out[hv, :], ssm_norm[:]) * silu(z[hv, d])`
// Grid: (h_v, 1, 1), Block: (32, 1, 1)
__global__ void gdn_rms_norm_gated_f32(
    float* __restrict__ out,
    const float* __restrict__ z,
    const float* __restrict__ norm_w,
    int dv_dim,
    float eps
) {
    const int hv = int(blockIdx.x);
    const int lane = int(threadIdx.x);
    float* y = out + size_t(hv) * dv_dim;
    const float* zh = z + size_t(hv) * dv_dim;

    float sq = 0.0f;
    for (int d = lane; d < dv_dim; d += 32) {
        sq += y[d] * y[d];
    }
    sq = warp_reduce_sum(sq);
    const float scale = rsqrtf(sq / float(dv_dim) + eps);
    for (int d = lane; d < dv_dim; d += 32) {
        const float zv = zh[d];
        const float silu_z = zv / (1.0f + expf(-zv));
        y[d] = y[d] * scale * norm_w[d] * silu_z;
    }
}

// ============================================================================
// 8. Fine-Grained Sparse MoE Router + Parallel `__dp4a` Indexed Expert Projection
// ============================================================================
// Grid: (n_tok, 1, 1), Block: (64, 1, 1) — supports up to 512 experts (`Qwen3.8` & `Hy4`),
// deterministic `<` index tie-breaking matching CPU, Metal, and Vulkan!
__global__ void moe_router_topk_f32(
    const float* __restrict__ logits,       // [n_tok, n_expert]
    const float* __restrict__ probs_bias,   // optional [n_expert] (null if absent)
    uint32_t* __restrict__ top_ids,         // [n_tok, k_used]
    float* __restrict__ top_weights,        // [n_tok, k_used]
    int n_expert,
    int k_used,
    int gating_func,     // 0/1 = Softmax, 2 = Sigmoid, 3 = SoftmaxWeight, 4 = SqrtSoftplus
    int norm_weights,
    float norm_eps,
    float weights_scale
) {
    const int tid = int(threadIdx.x); // 0..63
    const int lane = tid & 31;
    const int warp_id = tid >> 5;     // 0..1
    const int t = int(blockIdx.x);
    const float* row = logits + size_t(t) * n_expert;
    uint32_t* out_ids = top_ids + size_t(t) * k_used;
    float* out_w = top_weights + size_t(t) * k_used;

    __shared__ float sh_probs[512];
    __shared__ float sh_sel[512];
    __shared__ float sh_max[2];
    __shared__ float sh_sum[2];
    __shared__ float sh_w0[2];
    __shared__ int sh_i0[2];
    const int ne = min(n_expert, 512);

    if (gating_func == 2) {
        for (int e = tid; e < ne; e += 64) {
            sh_probs[e] = 1.0f / (1.0f + expf(-row[e]));
        }
    } else if (gating_func == 4) {
        for (int e = tid; e < ne; e += 64) {
            const float x = row[e];
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            sh_probs[e] = sqrtf(fmaxf(sp, 0.0f));
        }
    } else if (gating_func == 3) {
        for (int e = tid; e < ne; e += 64) {
            sh_probs[e] = row[e];
        }
    } else {
        float local_mx = -CUDART_INF_F;
        for (int e = tid; e < ne; e += 64) local_mx = fmaxf(local_mx, row[e]);
        local_mx = warp_reduce_max(local_mx);
        if (lane == 0) sh_max[warp_id] = local_mx;
        __syncthreads();
        const float mx = fmaxf(sh_max[0], sh_max[1]);

        float local_s = 0.0f;
        for (int e = tid; e < ne; e += 64) {
            const float p = expf(row[e] - mx);
            sh_probs[e] = p;
            local_s += p;
        }
        local_s = warp_reduce_sum(local_s);
        if (lane == 0) sh_sum[warp_id] = local_s;
        __syncthreads();
        const float inv = 1.0f / (sh_sum[0] + sh_sum[1]);
        for (int e = tid; e < ne; e += 64) sh_probs[e] *= inv;
    }

    for (int e = tid; e < ne; e += 64) {
        sh_sel[e] = sh_probs[e] + (probs_bias ? probs_bias[e] : 0.0f);
    }
    __syncthreads();

    for (int k = 0; k < k_used; ++k) {
        float best_v = -CUDART_INF_F;
        int best_i = 0x7fffffff;
        for (int e = tid; e < ne; e += 64) {
            const float v = sh_sel[e];
            if (v > best_v || (v == best_v && e < best_i)) {
                best_v = v;
                best_i = e;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset, 32);
            const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            sh_w0[warp_id] = best_v;
            sh_i0[warp_id] = best_i;
        }
        __syncthreads();
        if (tid == 0) {
            int win_i = sh_i0[0];
            float win_v = sh_w0[0];
            if (sh_w0[1] > win_v || (sh_w0[1] == win_v && sh_i0[1] < win_i)) {
                win_v = sh_w0[1];
                win_i = sh_i0[1];
            }
            out_ids[k] = uint32_t(win_i);
            out_w[k] = sh_probs[win_i];
            sh_sel[win_i] = -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (gating_func == 3) {
            float mx = -CUDART_INF_F;
            for (int k = 0; k < k_used; ++k) mx = fmaxf(mx, out_w[k]);
            float s = 0.0f;
            for (int k = 0; k < k_used; ++k) {
                out_w[k] = expf(out_w[k] - mx);
                s += out_w[k];
            }
            for (int k = 0; k < k_used; ++k) out_w[k] /= s;
        } else if (norm_weights) {
            float s = 0.0f;
            for (int k = 0; k < k_used; ++k) s += out_w[k];
            const float denom = (norm_eps > 0.0f) ? (s + norm_eps) : fmaxf(s, 6.1035156e-5f);
            for (int k = 0; k < k_used; ++k) out_w[k] /= denom;
        }

        if (weights_scale != 0.0f && fabsf(weights_scale - 1.0f) > 1e-12f) {
            for (int k = 0; k < k_used; ++k) {
                out_w[k] *= weights_scale;
            }
        }
    }
}

// Parallel multi-slot `__dp4a` MoE expert projection across ALL `(rows, slot_count, n_tok)` simultaneously!
// Grid: (ceil(rows / 4), slot_count, n_tok), Block: (32, 4, 1)
// - `x_mode == 0`: `xq` is `[n_tok, cols/32]`, `x_f32` is `[n_tok, cols]` (shared input per token, e.g. `gate`/`up`)
// - `x_mode == 1`: `xq` is `[n_tok * slot_count, cols/32]`, `x_f32` is `[n_tok * slot_count, cols]` (per-slot input, e.g. `down`)
// - `out_mode == 0`: writes `out[(t * slot_count + slot) * rows + row] = acc`
// - `out_mode == 1`: writes `out[(t * slot_count + slot) * rows + row] = top_weights[t * slot_count + slot] * acc`
// Shared `__dp4a` MoE expert row dot-product helper:
__device__ __forceinline__ float dot_moe_row_q8_1(
    const uint8_t* __restrict__ w_row,
    const BlockQ8_1* __restrict__ xr,
    const float* __restrict__ xt,
    int dtype,
    int cols,
    int lane
) {
    const int n_q8 = cols >> 5;
    float acc = 0.0f;

    if (dtype == DTYPE_Q8_0 && xr != nullptr && (cols & 31) == 0) {
        const BlockQ8_0* wr = reinterpret_cast<const BlockQ8_0*>(w_row);
        const int iqs = 2 * (lane & 3);
        for (int kbx = lane >> 2; kbx < n_q8; kbx += 8) {
            const BlockQ8_0& wb = wr[kbx];
            const BlockQ8_1& xb = xr[kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
            sumi = dp4a_i8(load_i32_2b(wb.qs + 4 * iqs), xqs[iqs], sumi);
            sumi = dp4a_i8(load_i32_2b(wb.qs + 4 * (iqs + 1)), xqs[iqs + 1], sumi);
            acc += __half2float(wb.d) * __low2float(xb.ds) * float(sumi);
        }
    } else if (dtype == DTYPE_Q4_0 && xr != nullptr && (cols & 31) == 0) {
        const BlockQ4_0* wr = reinterpret_cast<const BlockQ4_0*>(w_row);
        const int iqs = 2 * (lane & 1);
        for (int kbx = lane >> 1; kbx < n_q8; kbx += 16) {
            acc += dot_q4_0_q8_1_half(wr[kbx], xr[kbx], iqs);
        }
    } else if (dtype == DTYPE_Q4_1 && xr != nullptr && (cols & 31) == 0) {
        const BlockQ4_1* wr = reinterpret_cast<const BlockQ4_1*>(w_row);
        const int iqs = 2 * (lane & 1);
        for (int kbx = lane >> 1; kbx < n_q8; kbx += 16) {
            const BlockQ4_1& wb = wr[kbx];
            const BlockQ8_1& xb = xr[kbx];
            const int* wqs = reinterpret_cast<const int*>(wb.qs);
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = 0;
            const int v0 = wqs[iqs];
            const int v1 = wqs[iqs + 1];
            sumi = dp4a_i8(v0 & 0x0F0F0F0F, xqs[iqs], sumi);
            sumi = dp4a_i8((v0 >> 4) & 0x0F0F0F0F, xqs[iqs + 4], sumi);
            sumi = dp4a_i8(v1 & 0x0F0F0F0F, xqs[iqs + 1], sumi);
            sumi = dp4a_i8((v1 >> 4) & 0x0F0F0F0F, xqs[iqs + 5], sumi);
            const float2 dm = __half22float2(wb.dm);
            const float2 ds = __half22float2(xb.ds);
            acc += dm.x * ds.x * float(sumi) + 0.5f * dm.y * ds.y;
        }
    } else if (dtype == DTYPE_Q2_0 && xr != nullptr && (cols & 63) == 0) {
        const BlockQ2_0* wr = reinterpret_cast<const BlockQ2_0*>(w_row);
        const int n_b64 = cols >> 6;
        const int sub = lane & 3;
        const int q8_half = sub >> 1;
        const int q8_quad = (sub & 1) * 4;
        for (int b64 = lane >> 2; b64 < n_b64; b64 += 8) {
            const BlockQ2_0& wb = wr[b64];
            const BlockQ8_1& xb = xr[2 * b64 + q8_half];
            const uint32_t raw4 = uint32_t(load_i32_2b(wb.qs + 4 * sub));
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + q8_quad;
            int sumi = 0;
            sumi = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t(raw4 & 0xFFu)), xqs[0], sumi);
            sumi = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((raw4 >> 8) & 0xFFu)), xqs[1], sumi);
            sumi = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((raw4 >> 16) & 0xFFu)), xqs[2], sumi);
            sumi = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((raw4 >> 24) & 0xFFu)), xqs[3], sumi);
            acc += __half2float(wb.d) * __low2float(xb.ds) * float(sumi);
        }
    } else if (dtype == DTYPE_Q4_K && xr != nullptr && (cols & 255) == 0) {
        const BlockQ4K* wr = reinterpret_cast<const BlockQ4K*>(w_row);
        const int n_sb = cols >> 8;
        const int chunk64 = lane >> 3;
        const int q_idx = lane & 7;
        for (int sb = 0; sb < n_sb; ++sb) {
            const BlockQ4K& blk = wr[sb];
            const float2 dm = __half22float2(blk.dm);
            uint8_t sc0, m0, sc1, m1;
            decode_q4k_scale_min(2 * chunk64 + 0, blk.scales, sc0, m0);
            decode_q4k_scale_min(2 * chunk64 + 1, blk.scales, sc1, m1);
            const int packed = reinterpret_cast<const int*>(blk.qs)[chunk64 * 8 + q_idx];
            const BlockQ8_1& xb0 = xr[sb * 8 + 2 * chunk64 + 0];
            const BlockQ8_1& xb1 = xr[sb * 8 + 2 * chunk64 + 1];
            const int dot0 = dp4a_i8(packed & 0x0F0F0F0F, reinterpret_cast<const int*>(xb0.qs)[q_idx], 0);
            const int dot1 = dp4a_i8((packed >> 4) & 0x0F0F0F0F, reinterpret_cast<const int*>(xb1.qs)[q_idx], 0);
            const float2 ds0 = __half22float2(xb0.ds);
            const float2 ds1 = __half22float2(xb1.ds);
            float term = dm.x * (float(sc0) * ds0.x * float(dot0) + float(sc1) * ds1.x * float(dot1));
            if (q_idx == 0) term -= dm.y * (float(m0) * ds0.y + float(m1) * ds1.y);
            acc += term;
        }
    } else if (dtype == DTYPE_Q5_K && xr != nullptr && (cols & 255) == 0) {
        const BlockQ5K* wr = reinterpret_cast<const BlockQ5K*>(w_row);
        const int n_sb = cols >> 8;
        const int chunk64 = lane >> 3;
        const int q_idx = lane & 7;
        for (int sb = 0; sb < n_sb; ++sb) {
            const BlockQ5K& blk = wr[sb];
            const float2 dm = __half22float2(blk.dm);
            uint8_t sc0, m0, sc1, m1;
            decode_q4k_scale_min(2 * chunk64 + 0, blk.scales, sc0, m0);
            decode_q4k_scale_min(2 * chunk64 + 1, blk.scales, sc1, m1);
            const int packed = reinterpret_cast<const int*>(blk.qs)[chunk64 * 8 + q_idx];
            const int qh4 = reinterpret_cast<const int*>(blk.qh)[q_idx];
            const int q0 = (packed & 0x0F0F0F0F) | (((qh4 >> (2 * chunk64)) & 0x01010101) << 4);
            const int q1 = ((packed >> 4) & 0x0F0F0F0F) | (((qh4 >> (2 * chunk64 + 1)) & 0x01010101) << 4);
            const BlockQ8_1& xb0 = xr[sb * 8 + 2 * chunk64 + 0];
            const BlockQ8_1& xb1 = xr[sb * 8 + 2 * chunk64 + 1];
            const int dot0 = dp4a_i8(q0, reinterpret_cast<const int*>(xb0.qs)[q_idx], 0);
            const int dot1 = dp4a_i8(q1, reinterpret_cast<const int*>(xb1.qs)[q_idx], 0);
            const float2 ds0 = __half22float2(xb0.ds);
            const float2 ds1 = __half22float2(xb1.ds);
            float term = dm.x * (float(sc0) * ds0.x * float(dot0) + float(sc1) * ds1.x * float(dot1));
            if (q_idx == 0) term -= dm.y * (float(m0) * ds0.y + float(m1) * ds1.y);
            acc += term;
        }
    } else if (dtype == DTYPE_Q6_K && xr != nullptr && (cols & 255) == 0) {
        const BlockQ6K* wr = reinterpret_cast<const BlockQ6K*>(w_row);
        const int n_sb = cols >> 8;
        const int half128 = lane >> 4;
        const int sub_pair = (lane >> 3) & 1;
        const int q_idx = lane & 7;
        const int sc_half = q_idx >> 2;
        const int quad_a = sub_pair;
        const int quad_b = sub_pair + 2;
        for (int sb = 0; sb < n_sb; ++sb) {
            const BlockQ6K& blk = wr[sb];
            const float d = __half2float(blk.d);
            const int ql4 = load_i32_2b(blk.ql + half128 * 64 + sub_pair * 32 + q_idx * 4);
            const int qh4 = load_i32_2b(blk.qh + half128 * 32 + q_idx * 4);
            const int8_t* sc_ptr = blk.scales + half128 * 8;
            const int qa = (((ql4 & 0x0F0F0F0F) | (((qh4 >> (2 * quad_a)) & 0x03030303) << 4)) + 0x60606060) ^ 0x80808080;
            const int qb = ((((ql4 >> 4) & 0x0F0F0F0F) | (((qh4 >> (2 * quad_b)) & 0x03030303) << 4)) + 0x60606060) ^ 0x80808080;
            const BlockQ8_1& xba = xr[sb * 8 + half128 * 4 + quad_a];
            const BlockQ8_1& xbb = xr[sb * 8 + half128 * 4 + quad_b];
            const int dota = dp4a_i8(qa, reinterpret_cast<const int*>(xba.qs)[q_idx], 0);
            const int dotb = dp4a_i8(qb, reinterpret_cast<const int*>(xbb.qs)[q_idx], 0);
            acc += d * (float(sc_ptr[sc_half + 2 * quad_a]) * __low2float(xba.ds) * float(dota)
                      + float(sc_ptr[sc_half + 2 * quad_b]) * __low2float(xbb.ds) * float(dotb));
        }
    } else if (dtype == DTYPE_Q5_0 && xr != nullptr && (cols & 31) == 0) {
        const BlockQ5_0* wr = reinterpret_cast<const BlockQ5_0*>(w_row);
        const int sub = lane & 3;
        const int j0 = sub * 4;
        for (int kbx = lane >> 2; kbx < n_q8; kbx += 8) {
            const BlockQ5_0& wb = wr[kbx];
            const BlockQ8_1& xb = xr[kbx];
            const uint32_t qh = uint32_t(load_i32_2b(wb.qh));
            const uint32_t qs4 = uint32_t(load_i32_2b(wb.qs + j0));
            uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
                const int q0 = int((qb & 0x0Fu) | (((qh >> (j0 + k)) & 1u) << 4)) - 16;
                const int q1 = int((qb >> 4) | (((qh >> (16 + j0 + k)) & 1u) << 4)) - 16;
                w_lo |= uint32_t(uint8_t(int8_t(q0))) << (8 * k);
                w_hi |= uint32_t(uint8_t(int8_t(q1))) << (8 * k);
            }
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[sub], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + sub], sumi);
            acc += __half2float(wb.d) * __low2float(xb.ds) * float(sumi);
        }
    } else if (dtype == DTYPE_IQ4_NL && xr != nullptr && (cols & 31) == 0) {
        const BlockIQ4NL* wr = reinterpret_cast<const BlockIQ4NL*>(w_row);
        const int sub = lane & 3;
        for (int kbx = lane >> 2; kbx < n_q8; kbx += 8) {
            const BlockIQ4NL& wb = wr[kbx];
            const BlockQ8_1& xb = xr[kbx];
            const uint32_t qs4 = uint32_t(load_i32_2b(wb.qs + 4 * sub));
            uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
                w_lo |= uint32_t(uint8_t(c_kvalues_iq4nl[qb & 0x0Fu])) << (8 * k);
                w_hi |= uint32_t(uint8_t(c_kvalues_iq4nl[qb >> 4])) << (8 * k);
            }
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[sub], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + sub], sumi);
            acc += __half2float(wb.d) * __low2float(xb.ds) * float(sumi);
        }
    } else if (dtype == DTYPE_Q3_K && xr != nullptr && (cols & 255) == 0) {
        const BlockQ3K* wr = reinterpret_cast<const BlockQ3K*>(w_row);
        const int n_sb = cols >> 8;
        const int is = lane >> 1;
        const int part = lane & 1;
        const int half128 = is >> 3;
        const int j = (is >> 1) & 3;
        const int shift = 2 * j;
        const uint8_t m = uint8_t(1u << (4 * half128 + j));
        const int l0 = (is & 1) * 16 + part * 8;
        const int q8_sub = is >> 1;
        const int q8_quad = (is & 1) * 4 + part * 2;
        for (int sb = 0; sb < n_sb; ++sb) {
            const BlockQ3K& blk = wr[sb];
            const float d_all = __half2float(blk.d);
            const uint32_t a0 = uint32_t(load_i32_2b(blk.scales + 0));
            const uint32_t a1 = uint32_t(load_i32_2b(blk.scales + 4));
            const uint32_t tmp = uint32_t(load_i32_2b(blk.scales + 8));
            const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
            uint32_t aux[4];
            aux[0] = (a0 & km2) | ((tmp & km1) << 4);
            aux[1] = (a1 & km2) | (((tmp >> 2) & km1) << 4);
            aux[2] = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);
            aux[3] = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);
            const float dl = d_all * float(int(reinterpret_cast<const int8_t*>(aux)[is]) - 32);
            const uint8_t* qp = blk.qs + half128 * 32 + l0;
            const uint8_t* hp = blk.hmask + l0;
            uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int qv0 = int((qp[k] >> shift) & 3u) - ((hp[k] & m) ? 0 : 4);
                const int qv1 = int((qp[4 + k] >> shift) & 3u) - ((hp[4 + k] & m) ? 0 : 4);
                w0 |= uint32_t(uint8_t(int8_t(qv0))) << (8 * k);
                w1 |= uint32_t(uint8_t(int8_t(qv1))) << (8 * k);
            }
            const BlockQ8_1& xb = xr[sb * 8 + q8_sub];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + q8_quad;
            int sumi = dp4a_i8(int(w0), xqs[0], 0);
            sumi = dp4a_i8(int(w1), xqs[1], sumi);
            acc += dl * __low2float(xb.ds) * float(sumi);
        }
    } else if (dtype == DTYPE_IQ4_XS && xr != nullptr && (cols & 255) == 0) {
        const BlockIQ4XS* wr = reinterpret_cast<const BlockIQ4XS*>(w_row);
        const int n_sb = cols >> 8;
        const int ib = lane >> 2;
        const int quad = lane & 3;
        for (int sb = 0; sb < n_sb; ++sb) {
            const BlockIQ4XS& blk = wr[sb];
            const int ls = int((blk.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F)
                         | int(((blk.scales_h >> (2 * ib)) & 3u) << 4);
            const float dl = __half2float(blk.d) * float(ls - 32);
            const uint32_t qs4 = uint32_t(load_i32_2b(blk.qs + ib * 16 + quad * 4));
            uint32_t w_lo = 0u, w_hi = 0u;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qb = (qs4 >> (8 * k)) & 0xFFu;
                w_lo |= uint32_t(uint8_t(c_kvalues_iq4nl[qb & 0x0Fu])) << (8 * k);
                w_hi |= uint32_t(uint8_t(c_kvalues_iq4nl[qb >> 4])) << (8 * k);
            }
            const BlockQ8_1& xb = xr[sb * 8 + ib];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(int(w_lo), xqs[quad], 0);
            sumi = dp4a_i8(int(w_hi), xqs[4 + quad], sumi);
            acc += dl * __low2float(xb.ds) * float(sumi);
        }
    } else {
        for (int c = lane; c < cols; c += 32) {
            acc += dequant_elem(dtype, w_row, c) * xt[c];
        }
    }
    return acc;
}

__global__ void __launch_bounds__(128, 4) moe_indexed_gemv_q8_1(
    const uint8_t* __restrict__ w_exps,
    const BlockQ8_1* __restrict__ xq,
    const float* __restrict__ x_f32,
    const uint32_t* __restrict__ top_ids,
    const float* __restrict__ top_weights,
    const float* __restrict__ exp_scales,
    float* __restrict__ out,
    int dtype,
    int rows,
    int cols,
    size_t row_bytes,
    size_t expert_stride_bytes,
    int slot_count,
    int x_mode,
    int out_mode
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int slot = int(blockIdx.y);
    const int tok = int(blockIdx.z);
    const int lane = int(threadIdx.x);
    if (row >= rows) return;

    const size_t ts_idx = size_t(tok) * slot_count + slot;
    const uint32_t exp_id = top_ids[ts_idx];
    const uint8_t* w_row = w_exps + size_t(exp_id) * expert_stride_bytes + size_t(row) * row_bytes;
    const size_t x_row_idx = (x_mode == 0) ? size_t(tok) : ts_idx;
    const int n_q8 = cols >> 5;
    const BlockQ8_1* xr = (xq != nullptr) ? (xq + x_row_idx * n_q8) : nullptr;
    const float* xt = (x_f32 != nullptr) ? (x_f32 + x_row_idx * cols) : nullptr;

    float acc = dot_moe_row_q8_1(w_row, xr, xt, dtype, cols, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        if (exp_scales != nullptr) {
            acc *= exp_scales[exp_id];
        }
        const float w_slot = (out_mode == 1) ? top_weights[ts_idx] : 1.0f;
        out[ts_idx * rows + row] = w_slot * acc;
    }
}

// `P0-3`: Fused dual `Gate + Up + SwiGLU / SiTU` decode/small-batch MoE kernel!
// Reads `xq` ONCE per warp for `Q8_0`/`Q4_0`/`Q2_0`/`Q4_K` and eliminates intermediate `gate` & `up` global buffers!
// `act_mode`: `0` = SwiGLU (`silu(g) * u`), `1` = `Inkling-2` `SiTU` GLU (`situ_beta`, `situ_lb`).
// Grid: (ceil(n_ff_exp / 4), slot_count, n_tok), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) moe_indexed_dual_gu_act_f32(
    const uint8_t* __restrict__ w_gate_exps,
    const uint8_t* __restrict__ w_up_exps,
    const BlockQ8_1* __restrict__ xq,
    const float* __restrict__ x_f32,
    const uint32_t* __restrict__ top_ids,
    const float* __restrict__ gate_scales,
    const float* __restrict__ up_scales,
    float* __restrict__ out_act,
    int dtype,
    int n_ff_exp,
    int cols,
    size_t row_bytes,
    size_t gate_exp_stride,
    size_t up_exp_stride,
    size_t up_sub_offset_bytes,
    int slot_count,
    int act_mode,
    float situ_beta,
    float situ_lb
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int slot = int(blockIdx.y);
    const int tok = int(blockIdx.z);
    const int lane = int(threadIdx.x);
    if (row >= n_ff_exp) return;

    const size_t ts_idx = size_t(tok) * slot_count + slot;
    const uint32_t exp_id = top_ids[ts_idx];
    const uint8_t* wg_row = w_gate_exps + size_t(exp_id) * gate_exp_stride + size_t(row) * row_bytes;
    const uint8_t* wu_row = w_up_exps + size_t(exp_id) * up_exp_stride + up_sub_offset_bytes + size_t(row) * row_bytes;
    const int n_q8 = cols >> 5;
    const BlockQ8_1* xr = (xq != nullptr) ? (xq + size_t(tok) * n_q8) : nullptr;
    const float* xt = (x_f32 != nullptr) ? (x_f32 + size_t(tok) * cols) : nullptr;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    if (dtype == DTYPE_Q8_0 && xr != nullptr && (cols & 31) == 0) {
        const BlockQ8_0* wgr = reinterpret_cast<const BlockQ8_0*>(wg_row);
        const BlockQ8_0* wur = reinterpret_cast<const BlockQ8_0*>(wu_row);
        const int iqs = 2 * (lane & 3);
        for (int kbx = lane >> 2; kbx < n_q8; kbx += 8) {
            const BlockQ8_1& xb = xr[kbx];
            const float x_d = __low2float(xb.ds);
            const int x0 = reinterpret_cast<const int*>(xb.qs)[iqs];
            const int x1 = reinterpret_cast<const int*>(xb.qs)[iqs + 1];

            const BlockQ8_0& wgb = wgr[kbx];
            int sg = dp4a_i8(load_i32_2b(wgb.qs + 4 * iqs), x0, 0);
            sg = dp4a_i8(load_i32_2b(wgb.qs + 4 * (iqs + 1)), x1, sg);
            acc_g += __half2float(wgb.d) * x_d * float(sg);

            const BlockQ8_0& wub = wur[kbx];
            int su = dp4a_i8(load_i32_2b(wub.qs + 4 * iqs), x0, 0);
            su = dp4a_i8(load_i32_2b(wub.qs + 4 * (iqs + 1)), x1, su);
            acc_u += __half2float(wub.d) * x_d * float(su);
        }
    } else if (dtype == DTYPE_Q2_0 && xr != nullptr && (cols & 63) == 0) {
        const BlockQ2_0* wgr = reinterpret_cast<const BlockQ2_0*>(wg_row);
        const BlockQ2_0* wur = reinterpret_cast<const BlockQ2_0*>(wu_row);
        const int n_b64 = cols >> 6;
        const int sub = lane & 3;
        const int q8_half = sub >> 1;
        const int q8_quad = (sub & 1) * 4;
        for (int b64 = lane >> 2; b64 < n_b64; b64 += 8) {
            const BlockQ8_1& xb = xr[2 * b64 + q8_half];
            const float x_d = __low2float(xb.ds);
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + q8_quad;
            const int x0 = xqs[0], x1 = xqs[1], x2 = xqs[2], x3 = xqs[3];

            const BlockQ2_0& wgb = wgr[b64];
            const uint32_t rg = uint32_t(load_i32_2b(wgb.qs + 4 * sub));
            int sg = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t(rg & 0xFFu)), x0, 0);
            sg = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((rg >> 8) & 0xFFu)), x1, sg);
            sg = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((rg >> 16) & 0xFFu)), x2, sg);
            sg = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((rg >> 24) & 0xFFu)), x3, sg);
            acc_g += __half2float(wgb.d) * x_d * float(sg);

            const BlockQ2_0& wub = wur[b64];
            const uint32_t ru = uint32_t(load_i32_2b(wub.qs + 4 * sub));
            int su = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t(ru & 0xFFu)), x0, 0);
            su = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((ru >> 8) & 0xFFu)), x1, su);
            su = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((ru >> 16) & 0xFFu)), x2, su);
            su = dp4a_i8(unpack_q2_0_byte_s8x4(uint8_t((ru >> 24) & 0xFFu)), x3, su);
            acc_u += __half2float(wub.d) * x_d * float(su);
        }
    } else {
        acc_g = dot_moe_row_q8_1(wg_row, xr, xt, dtype, cols, lane);
        acc_u = dot_moe_row_q8_1(wu_row, xr, xt, dtype, cols, lane);
    }

    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if (lane == 0) {
        if (gate_scales != nullptr) acc_g *= gate_scales[exp_id];
        if (up_scales != nullptr) acc_u *= up_scales[exp_id];
        float act;
        if (act_mode == 1) {
            const float ag = situ_beta * accurate_tanh_f32(acc_g / situ_beta) / (1.0f + expf(-acc_g));
            const float au = (situ_lb > 0.0f) ? (situ_lb * accurate_tanh_f32(acc_u / situ_lb)) : acc_u;
            act = ag * au;
        } else {
            act = (acc_g / (1.0f + expf(-acc_g))) * acc_u;
        }
        out_act[ts_idx * n_ff_exp + row] = act;
    }
}

// `P0-2`: Deterministic on-GPU expert bucket builder (`n_expert <= 512`):
// Produces `exp_offs[0..=n_expert]`, `tile_offs[0..=n_expert]` (16-token tiles), and `sorted_ts[0..n_ent]`!
// Grid: (1, 1, 1), Block: (256, 1, 1)
__global__ void moe_build_expert_buckets_u32(
    const uint32_t* __restrict__ top_ids,
    uint32_t* __restrict__ exp_offs,
    uint32_t* __restrict__ tile_offs,
    uint32_t* __restrict__ sorted_ts,
    int n_ent,
    int n_expert
) {
    const int tid = int(threadIdx.x);
    __shared__ uint32_t s_counts[513];
    __shared__ uint32_t s_cursor[512];

    for (int e = tid; e <= n_expert; e += 256) {
        s_counts[e] = 0u;
    }
    __syncthreads();

    for (int i = tid; i < n_ent; i += 256) {
        const uint32_t e = top_ids[i];
        if (e < uint32_t(n_expert)) {
            atomicAdd(&s_counts[e], 1u);
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t cur_off = 0u;
        uint32_t cur_tile = 0u;
        for (int e = 0; e < n_expert; ++e) {
            const uint32_t cnt = s_counts[e];
            exp_offs[e] = cur_off;
            tile_offs[e] = cur_tile;
            s_cursor[e] = cur_off;
            cur_off += cnt;
            cur_tile += (cnt + 15u) >> 4;
        }
        exp_offs[n_expert] = cur_off;
        tile_offs[n_expert] = cur_tile;

        // Deterministic `0..n_ent` scatter (preserves exact `(tok, slot)` ordering inside every expert bucket)
        for (int i = 0; i < n_ent; ++i) {
            const uint32_t e = top_ids[i];
            if (e < uint32_t(n_expert)) {
                sorted_ts[s_cursor[e]++] = uint32_t(i);
            }
        }
    }
}

// `P0-2`: Grouped MoE `wmma` 16x16x16 Tensor Core GEMM (`BM = 64 rows, BN = 16 token-slots, BK = 32 cols`)!
// Grid: (ceil(rows / 64), max_tiles, 1), Block: (32, 4, 1) — excess `blockIdx.y >= tile_offs[n_expert]` exit in 1 instruction (zero host sync!).
__global__ void __launch_bounds__(128) wmma_moe_grouped_f32(
    const uint8_t* __restrict__ w_exps,
    const float* __restrict__ x_f32,
    const uint32_t* __restrict__ exp_offs,
    const uint32_t* __restrict__ tile_offs,
    const uint32_t* __restrict__ sorted_ts,
    const float* __restrict__ top_weights,
    const float* __restrict__ exp_scales,
    float* __restrict__ out,
    int dtype,
    int rows,
    int cols,
    size_t row_bytes,
    size_t expert_stride_bytes,
    size_t sub_offset_bytes,
    int n_expert,
    int slot_count,
    int x_mode,
    int out_mode
) {
    const int tile_id = int(blockIdx.y);
    if (uint32_t(tile_id) >= tile_offs[n_expert]) return;

    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int tid = warp_id * 32 + lane;
    const int row_base = int(blockIdx.x) * 64;

    __shared__ __align__(16) half smem_w[64 * 32];
    __shared__ __align__(16) half smem_x[16 * 32];
    __shared__ __align__(16) float smem_c[4 * 256];
    __shared__ uint32_t s_ts[16];
    __shared__ uint32_t s_xrow[16];
    __shared__ int s_exp_id;
    __shared__ int s_valid_n;

    if (tid == 0) {
        int lo = 0, hi = n_expert - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (int(tile_offs[mid]) <= tile_id) lo = mid;
            else hi = mid - 1;
        }
        s_exp_id = lo;
        const uint32_t b_start = exp_offs[lo];
        const uint32_t b_end = exp_offs[lo + 1];
        const uint32_t t_base = b_start + uint32_t(tile_id - int(tile_offs[lo])) * 16u;
        const int vn = min(16, int(b_end - t_base));
        s_valid_n = vn;
        for (int m = 0; m < vn; ++m) {
            const uint32_t ts = sorted_ts[t_base + m];
            s_ts[m] = ts;
            s_xrow[m] = (x_mode == 0) ? (ts / uint32_t(slot_count)) : ts;
        }
    }
    __syncthreads();

    const int exp_id = s_exp_id;
    const int valid_n = s_valid_n;
    const uint8_t* w_bytes = w_exps + size_t(exp_id) * expert_stride_bytes + sub_offset_bytes;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < cols; k0 += 32) {
        if (dtype == DTYPE_Q8_0) {
            const int r_local = tid >> 1;
            const int half_id = tid & 1;
            const int r = row_base + r_local;
            half* dst = smem_w + r_local * 32 + half_id * 16;
            if (r < rows) {
                const BlockQ8_0& blk = reinterpret_cast<const BlockQ8_0*>(w_bytes + size_t(r) * row_bytes)[k0 >> 5];
                const float d = __half2float(blk.d);
                const int8_t* qs = blk.qs + half_id * 16;
#pragma unroll
                for (int j = 0; j < 16; ++j) dst[j] = __float2half(d * float(qs[j]));
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j) dst[j] = __float2half(0.0f);
            }
        } else if (dtype == DTYPE_Q4_0) {
            const int r_local = tid >> 1;
            const int half_id = tid & 1;
            const int r = row_base + r_local;
            half* dst = smem_w + r_local * 32;
            if (r < rows) {
                const BlockQ4_0& blk = reinterpret_cast<const BlockQ4_0*>(w_bytes + size_t(r) * row_bytes)[k0 >> 5];
                const float d = __half2float(blk.d);
                const uint8_t* qs = blk.qs + half_id * 8;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint8_t byte = qs[j];
                    dst[half_id * 8 + j] = __float2half(d * float(int(byte & 0x0Fu) - 8));
                    dst[16 + half_id * 8 + j] = __float2half(d * float(int(byte >> 4) - 8));
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    dst[half_id * 8 + j] = __float2half(0.0f);
                    dst[16 + half_id * 8 + j] = __float2half(0.0f);
                }
            }
        } else {
#pragma unroll
            for (int idx = tid; idx < 64 * 32; idx += 128) {
                const int r_local = idx >> 5;
                const int c_local = idx & 31;
                const int r = row_base + r_local;
                const float val = (r < rows) ? dequant_elem(dtype, w_bytes + size_t(r) * row_bytes, k0 + c_local) : 0.0f;
                smem_w[idx] = __float2half(val);
            }
        }

        // Load `[16 token-slots x 32 cols]` of `X` (`512` elems / `128` threads = `4` elems/thread)
#pragma unroll
        for (int idx = tid; idx < 16 * 32; idx += 128) {
            const int m = idx >> 5;
            const int c_local = idx & 31;
            const float val = (m < valid_n) ? x_f32[size_t(s_xrow[m]) * cols + k0 + c_local] : 0.0f;
            smem_x[idx] = __float2half(val);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
            wmma::load_matrix_sync(frag_a, smem_w + (warp_id * 16) * 32 + kk, 32);
            wmma::load_matrix_sync(frag_b, smem_x + kk, 32);
            wmma::mma_sync(acc, frag_a, frag_b, acc);
        }
        __syncthreads();
    }

    float* warp_smem = smem_c + warp_id * 256;
    wmma::store_matrix_sync(warp_smem, acc, 16, wmma::mem_row_major);
    __syncwarp();
    const float e_sc = (exp_scales != nullptr) ? exp_scales[exp_id] : 1.0f;
    for (int idx = lane; idx < 256; idx += 32) {
        const int r = row_base + warp_id * 16 + (idx >> 4);
        const int m = idx & 15;
        if (r < rows && m < valid_n) {
            const uint32_t ts_idx = s_ts[m];
            const float w_slot = (out_mode == 1) ? top_weights[ts_idx] : 1.0f;
            out[size_t(ts_idx) * rows + r] = warp_smem[idx] * e_sc * w_slot;
        }
    }
}

// Deterministic ordered reduction of `[n_tok, slot_count, rows]` -> `[n_tok, rows]` (`0..slot_count` in order, zero atomicAdd!).
// Grid: (ceil(rows / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void moe_reduce_slots_f32(
    const float* __restrict__ slot_outs,
    float* __restrict__ out,
    int rows,
    int slot_count,
    int n_tok
) {
    const int r = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    const int t = int(blockIdx.y);
    if (r >= rows || t >= n_tok) return;
    const float* base = slot_outs + size_t(t) * slot_count * rows + r;
    float sum = 0.0f;
    for (int s = 0; s < slot_count; ++s) {
        sum += base[size_t(s) * rows];
    }
    out[size_t(t) * rows + r] = sum;
}

// Legacy single-slot scalar helper kept for direct unit tests:
// Grid: (ceil(rows / 4), 1, 1), Block: (32, 4, 1)
__global__ void moe_indexed_gemv_f32(
    const uint8_t* __restrict__ w_exps,
    const float* __restrict__ x,
    const uint32_t* __restrict__ top_ids,
    const float* __restrict__ top_weights,
    float* __restrict__ out,
    int dtype,
    int rows,
    int cols,
    size_t row_bytes,
    size_t expert_stride_bytes,
    int slot_idx,
    int accum_weighted
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int lane = int(threadIdx.x);
    if (row >= rows) return;

    const uint32_t exp_id = top_ids[slot_idx];
    const uint8_t* w_row = w_exps + size_t(exp_id) * expert_stride_bytes + size_t(row) * row_bytes;

    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        acc += dequant_elem(dtype, w_row, c) * x[c];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        if (accum_weighted) {
            out[row] += top_weights[slot_idx] * acc;
        } else {
            out[row] = acc;
        }
    }
}

// ============================================================================
// 9. Single-CTA Greedy Argmax (`vocab_size` floats -> 4-byte `uint32_t`)
// ============================================================================

// Grid: (1, n_rows, 1), Block: (256, 1, 1)
__global__ void argmax_f32(
    const float* __restrict__ logits,
    uint32_t* __restrict__ out_id,
    int vocab_size
) {
    const int tid = int(threadIdx.x);
    const int row = int(blockIdx.y);
    const float* row_logits = logits + size_t(row) * vocab_size;
    float best_v = -CUDART_INF_F;
    uint32_t best_i = 0;

    for (int i = tid; i < vocab_size; i += 256) {
        const float v = row_logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = uint32_t(i);
        }
    }

    // Warp reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset, 32);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, best_i, offset, 32);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float smem_v[8];
    __shared__ uint32_t smem_i[8];
    if ((tid & 31) == 0) {
        smem_v[tid >> 5] = best_v;
        smem_i[tid >> 5] = best_i;
    }
    __syncthreads();

    if (tid < 8) {
        float v = smem_v[tid];
        uint32_t idx = smem_i[tid];
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffu, v, offset, 8);
            const uint32_t oi = __shfl_xor_sync(0xffu, idx, offset, 8);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (tid == 0) {
            out_id[row] = idx;
        }
    }
}

// ============================================================================
// 10. Thinking Machines Lab `Inkling-2-Mini` (`inkling2_loop`) + `DSpark` (`dflash`)
// ============================================================================

// Unit RMSNorm (`no weight`): Grid (n_rows, 1, 1), Block (256, 1, 1)
__global__ void inkling2_rms_norm_unit_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int dim,
    float eps
) {
    const int row = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    __shared__ float warp_sums[8];
    const float* xr = x + size_t(row) * dim;
    float* yr = out + size_t(row) * dim;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        const float v = xr[i];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(dim) + eps);
    for (int i = tid; i < dim; i += 256) {
        yr[i] = xr[i] * inv;
    }
}

// Depth-residual attention `mix_depth_into`: Grid (n_queries, n_tok, 1), Block (256, 1, 1)
// `n_blocks <= 15` (`m = n_blocks + 1 <= 16`). When `replicate_count > 1` (SWA `n_queries == 1`),
// replicates output to `0..replicate_count` consecutive slots of `[n_embd]` for token `tok`.
__global__ void inkling2_mix_depth_f32(
    const float* __restrict__ blocks,
    const float* __restrict__ cur,
    const float* __restrict__ queries,
    float* __restrict__ out,
    int n_embd,
    int n_blocks,
    int blocks_tok_stride,
    int cur_tok_stride,
    int out_q_stride,
    int replicate_count
) {
    const int q_idx = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int m = n_blocks + 1;
    __shared__ float warp_sums[8];
    __shared__ float s_inv_rms[16];
    __shared__ float s_scores[16];

    const float* q_vec = queries + size_t(q_idx) * n_embd;
    const float* blk_base = blocks + size_t(tok) * blocks_tok_stride;
    const float* cur_base = cur + size_t(tok) * cur_tok_stride;
    const float scale = rsqrtf(float(n_embd));

    for (int r = 0; r < m; ++r) {
        const float* src = (r < n_blocks)
            ? (blk_base + size_t(n_blocks - 1 - r) * n_embd)
            : cur_base;
        float ss = 0.0f;
        for (int i = tid; i < n_embd; i += 256) {
            const float v = src[i];
            ss = fmaf(v, v, ss);
        }
        const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(n_embd) + 1.0e-6f);
        if (tid == 0) s_inv_rms[r] = inv;
        __syncthreads();

        float dq = 0.0f;
        for (int i = tid; i < n_embd; i += 256) {
            dq = fmaf(src[i] * inv, q_vec[i], dq);
        }
        const float sc = block256_reduce_sum(dq, tid, warp_sums) * scale;
        if (tid == 0) s_scores[r] = sc;
        __syncthreads();
    }

    if (tid == 0) {
        float mx = -CUDART_INF_F;
        for (int r = 0; r < m; ++r) mx = fmaxf(mx, s_scores[r]);
        float den = 0.0f;
        for (int r = 0; r < m; ++r) {
            const float e = expf(s_scores[r] - mx);
            s_scores[r] = e;
            den += e;
        }
        const float inv_d = 1.0f / den;
        for (int r = 0; r < m; ++r) s_scores[r] *= inv_d;
    }
    __syncthreads();

    float* dst_tok = out + size_t(tok) * out_q_stride * n_embd;
    for (int i = tid; i < n_embd; i += 256) {
        float acc = 0.0f;
        for (int r = 0; r < n_blocks; ++r) {
            const float* sb = blk_base + size_t(n_blocks - 1 - r) * n_embd;
            acc = fmaf(s_scores[r], sb[i] * s_inv_rms[r], acc);
        }
        acc = fmaf(s_scores[n_blocks], cur_base[i] * s_inv_rms[n_blocks], acc);
        if (replicate_count > 1 && q_idx == 0) {
            for (int rep = 0; rep < replicate_count; ++rep) {
                dst_tok[size_t(rep) * n_embd + i] = acc;
            }
        } else {
            dst_tok[size_t(q_idx) * n_embd + i] = acc;
        }
    }
}

// Prepares `[rms_unit(cur); rms_unit(msg)]` into `ctrl_in[n_tok, 2 * n_embd]`:
// Grid: (2, n_tok, 1), Block: (256, 1, 1)
__global__ void inkling2_control_prep_f32(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    float* __restrict__ ctrl_in,
    int n_embd,
    int cur_tok_stride,
    int msg_tok_stride
) {
    const int part = int(blockIdx.x); // 0 = cur, 1 = msg
    const int tok = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    __shared__ float warp_sums[8];
    const float* src = (part == 0)
        ? (cur + size_t(tok) * cur_tok_stride)
        : (msg + size_t(tok) * msg_tok_stride);
    float* dst = ctrl_in + size_t(tok) * (2 * n_embd) + size_t(part) * n_embd;

    float ss = 0.0f;
    for (int i = tid; i < n_embd; i += 256) {
        const float v = src[i];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(n_embd) + 1.0e-6f);
    for (int i = tid; i < n_embd; i += 256) {
        dst[i] = src[i] * inv;
    }
}

// Applies `(cur - tanh(forget)*cur) + tanh(gain)*msg`:
// Grid: (ceil(n_embd / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void inkling2_control_apply_f32(
    float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ gain_raw,
    const float* __restrict__ forget_raw,
    int n_embd,
    int cur_tok_stride,
    int msg_tok_stride
) {
    const int c = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int tok = int(blockIdx.y);
    if (c >= n_embd) return;
    float* cr = cur + size_t(tok) * cur_tok_stride;
    const float* mr = msg + size_t(tok) * msg_tok_stride;
    const size_t gi = size_t(tok) * n_embd + c;
    const float x = cr[c];
    const float g = accurate_tanh_f32(gain_raw[gi]);
    const float fg = accurate_tanh_f32(forget_raw[gi]);
    cr[c] = (x - fg * x) + g * mr[c];
}

// Multi-token causal ShortConv (`n_conv <= 9`) in registers with optional per-token post-step snapshot!
// Grid: (ceil(n_channels / 256), 1, 1), Block: (256, 1, 1)
__global__ void inkling2_shortconv_f32(
    float* __restrict__ cur,             // [n_tok, n_channels]
    const float* __restrict__ weight,    // [n_channels, n_conv]
    const float* __restrict__ dyn_taps,  // optional [n_tok, dyn_heads * n_conv]
    float* __restrict__ slot_hist,       // [total_slot_channels * hist_len]
    float* __restrict__ snap_out,        // optional [n_tok, snap_slot_floats] (post-step history for DSpark rollback)
    int n_channels,
    int n_conv,
    int chan_offset,
    int dyn_heads,
    int snap_base_floats,
    int snap_total_floats,
    int n_tok
) {
    const int c = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (c >= n_channels) return;
    const int hist_len = n_conv - 1;
    const size_t h_off = size_t(chan_offset + c) * hist_len;
    const size_t w_off = size_t(c) * n_conv;
    const int mult = (dyn_heads > 0 && dyn_taps != nullptr) ? 2 : 1;
    const float inv_scale = rsqrtf(float(mult * n_conv + 1));
    const int ch_per_head = (dyn_heads > 0) ? (n_channels / dyn_heads) : 1;
    const int h_idx = c / ch_per_head;
    const int dyn_tok_stride = dyn_heads * n_conv;

    float hist[8];
    float w_stat[9];
    for (int k = 0; k < hist_len; ++k) {
        hist[k] = slot_hist[h_off + k];
        w_stat[k] = weight[w_off + k];
    }
    w_stat[hist_len] = weight[w_off + hist_len];

    for (int tok = 0; tok < n_tok; ++tok) {
        const size_t idx = size_t(tok) * n_channels + c;
        const float cur_c = cur[idx];
        float acc = 0.0f;
        for (int k = 0; k < hist_len; ++k) {
            acc = fmaf(hist[k], w_stat[k], acc);
        }
        acc = fmaf(cur_c, w_stat[hist_len], acc);

        if (dyn_heads > 0 && dyn_taps != nullptr) {
            const float* tap = dyn_taps + size_t(tok) * dyn_tok_stride + size_t(h_idx) * n_conv;
            float dyn_acc = cur_c * tap[0];
            for (int lag = 1; lag < n_conv; ++lag) {
                dyn_acc = fmaf(hist[hist_len - lag], tap[lag], dyn_acc);
            }
            acc += dyn_acc;
        }

        for (int k = 0; k < hist_len - 1; ++k) {
            hist[k] = hist[k + 1];
        }
        if (hist_len > 0) {
            hist[hist_len - 1] = cur_c;
        }
        cur[idx] = (cur_c + acc) * inv_scale;

        if (snap_out != nullptr) {
            float* snap_dst = snap_out + size_t(tok) * snap_total_floats + snap_base_floats + h_off;
            for (int k = 0; k < hist_len; ++k) {
                snap_dst[k] = hist[k];
            }
        }
    }

    for (int k = 0; k < hist_len; ++k) {
        slot_hist[h_off + k] = hist[k];
    }
}

// Per-head L2 norm clip + optional scale + optional position-dependent `tau`:
// Grid: (n_heads, n_tok, 1), Block: (128, 1, 1)
__global__ void inkling2_clip_heads_f32(
    float* __restrict__ vec,
    const float* __restrict__ opt_scale,
    int n_heads,
    int dim,
    float thresh,
    int apply_tau,
    int log_floor,
    float log_alpha,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    float* head = vec + (size_t(tok) * n_heads + h) * dim;
    const float* sc = opt_scale ? (opt_scale + size_t(h) * dim) : nullptr;

    float ss = 0.0f;
    for (int i = tid; i < dim; i += 128) {
        float v = head[i];
        if (sc) {
            v *= sc[i];
            head[i] = v;
        }
        ss = fmaf(v, v, ss);
    }
    ss = warp_reduce_sum(ss);
    __shared__ float ws[4];
    if ((tid & 31) == 0) ws[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        const float v = (tid < 4) ? ws[tid] : 0.0f;
        const float tot = warp_reduce_sum(v);
        if (tid == 0) ws[0] = tot;
    }
    __syncthreads();

    float tau = 1.0f;
    if (apply_tau) {
        const float ratio = fmaxf(1.0f, float(pos + 1) / float(max(log_floor, 1)));
        tau = 1.0f + log_alpha * logf(ratio);
    }
    const float norm = sqrtf(ws[0]) + 1.0e-6f;
    const float factor = fminf(fmaxf(thresh / norm, 0.0f), 1.0f) * tau;
    for (int i = tid; i < dim; i += 128) {
        head[i] *= factor;
    }
}

// Fused `[gate, up]` (`[n_rows, 2 * n_ff]`) -> `SiTU` GLU (`[n_rows, n_ff]`)
__global__ void inkling2_situ_glu_fused_f32(
    const float* __restrict__ gate_up,
    float* __restrict__ out,
    int n_ff,
    int n_rows,
    float beta,
    float lb
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(n_rows) * n_ff;
    if (idx >= total) return;
    const int r = int(idx / n_ff);
    const int c = int(idx % n_ff);
    const float* src = gate_up + size_t(r) * (2 * n_ff);
    const float g = src[c];
    const float u = src[n_ff + c];
    const float act_g = beta * accurate_tanh_f32(g / beta) / (1.0f + expf(-g));
    const float act_u = (lb > 0.0f) ? (lb * accurate_tanh_f32(u / lb)) : u;
    out[idx] = act_g * act_u;
}

// Fused `[gate, up]` -> `SiTU` GLU + `BlockQ8_1` quantization (`[n_rows, n_ff / 32]`):
// Grid: (ceil((n_ff/32) / 4), n_rows, 1), Block: (32, 4, 1)
__global__ void inkling2_situ_glu_fused_quantize_q8_1_f32(
    const float* __restrict__ gate_up,
    BlockQ8_1* __restrict__ out,
    int n_ff,
    int n_rows,
    float beta,
    float lb
) {
    const int n_blocks = n_ff >> 5;
    const int blk_idx = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int row = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    if (blk_idx >= n_blocks || row >= n_rows) return;

    const float* src = gate_up + size_t(row) * (2 * n_ff);
    const int col = blk_idx * 32 + lane;
    const float g = src[col];
    const float u = src[n_ff + col];
    const float act_g = beta * accurate_tanh_f32(g / beta) / (1.0f + expf(-g));
    const float act_u = (lb > 0.0f) ? (lb * accurate_tanh_f32(u / lb)) : u;
    const float val = act_g * act_u;

    const float amax = warp_reduce_max(fabsf(val));
    const float sum_x = warp_reduce_sum(val);
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int q = __float2int_rn(val * id);
    const int8_t qs = int8_t(max(-127, min(127, q)));

    BlockQ8_1& dst = out[size_t(row) * n_blocks + blk_idx];
    dst.qs[lane] = qs;
    if (lane == 0) {
        dst.ds = __floats2half2_rn(d, sum_x);
    }
}

// Online-softmax GQA for `Inkling-2` with learned relative-position bias table + zero-logit sink!
// Grid: (n_head, n_tok, 1), Block: (32, 8, 1) (`head_dim <= 256`, `extent <= 1024` cached in smem!)
__global__ void __launch_bounds__(256, 2) inkling2_rel_attn_f16_kv(
    const float* __restrict__ q,
    const float* __restrict__ r_buf,
    const float* __restrict__ rel_bias,
    const float* __restrict__ sinks,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int n_rel,
    int extent,
    int n_swa,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int head = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int wid = int(threadIdx.y); // 0..7
    const int tid = wid * 32 + lane;
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int kv_start = (n_swa > 0 && (pos + 1) > n_swa) ? (pos + 1 - n_swa) : 0;
    const int kv_dim = n_head_kv * head_dim;
    const int kv_h = head / (n_head / n_head_kv);
    const float scale = rsqrtf(float(head_dim));
    const size_t head_idx = size_t(tok) * n_head + head;
    const float sink_h = sinks[head_idx];
    const float* r_h = r_buf + head_idx * n_rel;

    __shared__ float q_sh[256];
    __shared__ float rel_sh[1025];
    __shared__ float wg_m[8];
    __shared__ float wg_l[8];
    __shared__ float wg_o[8][256];

    const float* q_h = q + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += 256) {
        q_sh[d] = q_h[d] * scale;
    }
    const int cache_lim = min(extent, 1024);
    for (int dist = tid; dist < cache_lim; dist += 256) {
        const float* br = rel_bias + size_t(dist) * n_rel;
        float acc = 0.0f;
        for (int r = 0; r < n_rel; ++r) acc = fmaf(r_h[r], br[r], acc);
        rel_sh[dist] = acc - sink_h;
    }
    if (tid == 0) rel_sh[cache_lim] = -sink_h;
    __syncthreads();

    float m_loc = -CUDART_INF_F;
    float l_loc = 0.0f;
    float o_reg[8] = {0.0f};

    for (int t = kv_start + wid; t <= pos; t += 8) {
        const half* kh = k_cache + size_t(t) * kv_dim + size_t(kv_h) * head_dim;
        const half* vh = v_cache + size_t(t) * kv_dim + size_t(kv_h) * head_dim;
        float dot_v = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            dot_v = fmaf(q_sh[d], __half2float(kh[d]), dot_v);
        }
        dot_v = warp_reduce_sum(dot_v);

        const int dist = pos - t;
        float rel_b;
        if (extent <= 1024) {
            rel_b = rel_sh[min(dist, extent)];
        } else if (dist >= extent) {
            rel_b = -sink_h;
        } else {
            const float* br = rel_bias + size_t(dist) * n_rel;
            float acc = 0.0f;
            for (int r = 0; r < n_rel; ++r) acc = fmaf(r_h[r], br[r], acc);
            rel_b = acc - sink_h;
        }
        const float s = dot_v + rel_b;
        const float m_new = fmaxf(m_loc, s);
        const float alpha = (l_loc == 0.0f) ? 0.0f : expf(m_loc - m_new);
        const float p = expf(s - m_new);
        l_loc = l_loc * alpha + p;
        m_loc = m_new;
        for (int d = lane; d < head_dim; d += 32) {
            const int c = d >> 5;
            o_reg[c] = o_reg[c] * alpha + p * __half2float(vh[d]);
        }
    }

    if (lane == 0) {
        wg_m[wid] = m_loc;
        wg_l[wid] = l_loc;
    }
    for (int d = lane; d < head_dim; d += 32) {
        wg_o[wid][d] = o_reg[d >> 5];
    }
    __syncthreads();

    // Zero attention sink (`m_all >= 0.0`, `denom = exp(-m_all) + sum`)
    float m_all = 0.0f;
    for (int w = 0; w < 8; ++w) {
        if (wg_l[w] > 0.0f) m_all = fmaxf(m_all, wg_m[w]);
    }
    float* out_h = out + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += 256) {
        float denom = expf(-m_all);
        float acc = 0.0f;
        for (int w = 0; w < 8; ++w) {
            if (wg_l[w] > 0.0f) {
                const float wt = expf(wg_m[w] - m_all);
                denom += wg_l[w] * wt;
                acc += wg_o[w][d] * wt;
            }
        }
        out_h[d] = acc / denom;
    }
}

__global__ void softcap_inplace_f32(float* __restrict__ x, float cap, size_t n) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n || cap <= 0.0f) return;
    x[idx] = cap * accurate_tanh_f32(x[idx] / cap);
}

// Fused softcap + argmax + optional softmax top-1 probability (`out_probs` may be null; `vocab_map` may be null):
// Grid: (1, n_rows, 1), Block: (256, 1, 1)
__global__ void softcap_argmax_f32(
    const float* __restrict__ logits,     // [n_rows, row_stride]
    const uint32_t* __restrict__ vocab_map,
    uint32_t* __restrict__ out_ids,
    float* __restrict__ out_probs,
    int n_vocab,
    int row_stride,
    int out_slot_offset,
    float cap
) {
    const int row = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const float* src = logits + size_t(row) * row_stride;

    float best_v = -CUDART_INF_F;
    uint32_t best_i = 0;
    float loc_s = 0.0f;
    for (int i = tid; i < n_vocab; i += 256) {
        const float raw = src[i];
        const float v = (cap > 0.0f) ? (cap * accurate_tanh_f32(raw / cap)) : raw;
        if (loc_s == 0.0f) {
            best_v = v;
            best_i = uint32_t(i);
            loc_s = 1.0f;
        } else if (v > best_v) {
            loc_s = fmaf(loc_s, expf(best_v - v), 1.0f);
            best_v = v;
            best_i = uint32_t(i);
        } else {
            if (v == best_v && uint32_t(i) < best_i) best_i = uint32_t(i);
            loc_s += expf(v - best_v);
        }
    }

    __shared__ float sh_v[256];
    __shared__ uint32_t sh_i[256];
    __shared__ float warp_sums[8];
    sh_v[tid] = best_v;
    sh_i[tid] = best_i;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float rv = sh_v[tid + stride];
            const uint32_t ri = sh_i[tid + stride];
            const float lv = sh_v[tid];
            const uint32_t li = sh_i[tid];
            if (rv > lv || (rv == lv && ri < li)) {
                sh_v[tid] = rv;
                sh_i[tid] = ri;
            }
        }
        __syncthreads();
    }
    const uint32_t winner = sh_i[0];
    const float global_max = sh_v[0];
    if (out_probs != nullptr) {
        const float scaled_s = (loc_s > 0.0f) ? (loc_s * expf(best_v - global_max)) : 0.0f;
        const float total_s = block256_reduce_sum(scaled_s, tid, warp_sums);
        if (tid == 0) {
            out_ids[out_slot_offset + row] = vocab_map ? vocab_map[winner] : winner;
            out_probs[out_slot_offset + row] = 1.0f / fmaxf(total_s, 1.0f);
        }
    } else if (tid == 0) {
        out_ids[out_slot_offset + row] = vocab_map ? vocab_map[winner] : winner;
    }
}

// --- `DSpark` / `DFlash` auxiliary GPU kernels ---

// Fuses `n_taps` target visit vectors + weighted RMSNorm into linear or ring buffer:
// Grid: (1, count, 1), Block: (256, 1, 1)
__global__ void dflash_mix_rmsnorm_f32(
    const float* __restrict__ taps,
    const float* __restrict__ fuse_prob,
    const float* __restrict__ norm_w,
    float* __restrict__ out,
    int n_embd,
    int n_taps,
    float eps,
    int start_pos,
    int tok_skip,
    int ring_cap
) {
    const int tok = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    __shared__ float warp_sums[8];
    const int src_tok = tok_skip + tok;
    const int out_row = (ring_cap > 0) ? ((start_pos + src_tok) % ring_cap) : tok;
    float* dst = out + size_t(out_row) * n_embd;

    float ss = 0.0f;
    for (int i = tid; i < n_embd; i += 256) {
        float m = 0.0f;
        for (int t = 0; t < n_taps; ++t) {
            m = fmaf(taps[(size_t(src_tok) * n_taps + t) * n_embd + i], fuse_prob[t], m);
        }
        dst[i] = m;
        ss = fmaf(m, m, ss);
    }
    const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(n_embd) + eps);
    for (int i = tid; i < n_embd; i += 256) {
        dst[i] *= inv * norm_w[i];
    }
}

// Per-KV-head RMSNorm + NeoX RoPE + F16 KV cache write (`inject_batch`):
// Grid: (n_head_kv, count, 1), Block: (128, 1, 1)
__global__ void dflash_k_ctx_norm_rope_kv_f16(
    float* __restrict__ inj_k,
    const float* __restrict__ inj_v,
    const float* __restrict__ norm_w,
    const float* __restrict__ inv_freqs,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int n_head_kv,
    int head_dim,
    int rope_dim,
    int start_pos,
    float eps
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int kv_dim = n_head_kv * head_dim;
    const int pos = start_pos + tok;
    const size_t base = size_t(tok) * kv_dim + size_t(h) * head_dim;

    float ss = 0.0f;
    for (int d = tid; d < head_dim; d += 128) {
        const float v = inj_k[base + d];
        ss = fmaf(v, v, ss);
    }
    ss = warp_reduce_sum(ss);
    __shared__ float ws[4];
    if ((tid & 31) == 0) ws[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        const float s = (tid < 4) ? ws[tid] : 0.0f;
        const float tot = warp_reduce_sum(s);
        if (tid == 0) ws[0] = rsqrtf(tot / float(head_dim) + eps);
    }
    __syncthreads();
    const float inv = ws[0];
    for (int d = tid; d < head_dim; d += 128) {
        inj_k[base + d] *= inv * norm_w[d];
    }
    __syncthreads();

    const int half_rope = rope_dim >> 1;
    for (int i = tid; i < half_rope; i += 128) {
        const float theta = float(pos) * inv_freqs[i];
        float s, c;
        sincosf(theta, &s, &c);
        const float x0 = inj_k[base + i];
        const float x1 = inj_k[base + half_rope + i];
        inj_k[base + i] = x0 * c - x1 * s;
        inj_k[base + half_rope + i] = x0 * s + x1 * c;
    }
    __syncthreads();

    const size_t dst = size_t(pos) * kv_dim + size_t(h) * head_dim;
    for (int d = tid; d < head_dim; d += 128) {
        k_cache[dst + d] = __float2half_rn(inj_k[base + d]);
        v_cache[dst + d] = __float2half_rn(inj_v[base + d]);
    }
}

// Initializes `[bs, n_embd]` block input (`row 0 = anchor_normed`, `rows 1..bs = mask_embd`) + zeroes `rnn_state`:
// Grid: (ceil(bs * n_embd / 256), 1, 1), Block: (256, 1, 1)
__global__ void dflash_init_block_f32(
    float* __restrict__ inp_l,
    const float* __restrict__ anchor_normed,
    const float* __restrict__ mask_embd,
    float* __restrict__ rnn_state,
    int n_embd,
    int bs,
    int rank
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(bs) * n_embd;
    if (idx < total) {
        const int r = int(idx / n_embd);
        const int c = int(idx % n_embd);
        inp_l[idx] = (r == 0) ? anchor_normed[c] : mask_embd[c];
    }
    if (idx < size_t(rank)) {
        rnn_state[idx] = 0.0f;
    }
}

// Per-head Q/K RMSNorm + NeoX RoPE + exact F16 round-trip on block K & V:
// Grid: (n_head + n_head_kv, bs, 1), Block: (128, 1, 1)
__global__ void dflash_qk_norm_rope_f16round_f32(
    float* __restrict__ q_blk,
    float* __restrict__ k_blk,
    float* __restrict__ v_blk,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ inv_freqs,
    int n_head,
    int n_head_kv,
    int head_dim,
    int rope_dim,
    int pos0,
    float eps
) {
    const int h_all = int(blockIdx.x);
    const int r = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = pos0 + r;
    const int half_rope = rope_dim >> 1;
    __shared__ float ws[4];

    if (h_all < n_head) {
        float* vec = q_blk + (size_t(r) * n_head + h_all) * head_dim;
        float ss = 0.0f;
        for (int d = tid; d < head_dim; d += 128) ss = fmaf(vec[d], vec[d], ss);
        ss = warp_reduce_sum(ss);
        if ((tid & 31) == 0) ws[tid >> 5] = ss;
        __syncthreads();
        if (tid < 32) {
            const float tot = warp_reduce_sum((tid < 4) ? ws[tid] : 0.0f);
            if (tid == 0) ws[0] = rsqrtf(tot / float(head_dim) + eps);
        }
        __syncthreads();
        const float inv = ws[0];
        for (int d = tid; d < head_dim; d += 128) vec[d] *= inv * q_norm[d];
        __syncthreads();
        for (int i = tid; i < half_rope; i += 128) {
            float s, c;
            sincosf(float(pos) * inv_freqs[i], &s, &c);
            const float x0 = vec[i], x1 = vec[half_rope + i];
            vec[i] = x0 * c - x1 * s;
            vec[half_rope + i] = x0 * s + x1 * c;
        }
    } else {
        const int kvh = h_all - n_head;
        float* kv = k_blk + (size_t(r) * n_head_kv + kvh) * head_dim;
        float* vv = v_blk + (size_t(r) * n_head_kv + kvh) * head_dim;
        float ss = 0.0f;
        for (int d = tid; d < head_dim; d += 128) ss = fmaf(kv[d], kv[d], ss);
        ss = warp_reduce_sum(ss);
        if ((tid & 31) == 0) ws[tid >> 5] = ss;
        __syncthreads();
        if (tid < 32) {
            const float tot = warp_reduce_sum((tid < 4) ? ws[tid] : 0.0f);
            if (tid == 0) ws[0] = rsqrtf(tot / float(head_dim) + eps);
        }
        __syncthreads();
        const float inv = ws[0];
        for (int d = tid; d < head_dim; d += 128) kv[d] *= inv * k_norm[d];
        __syncthreads();
        for (int i = tid; i < half_rope; i += 128) {
            float s, c;
            sincosf(float(pos) * inv_freqs[i], &s, &c);
            const float x0 = kv[i], x1 = kv[half_rope + i];
            kv[i] = __half2float(__float2half_rn(x0 * c - x1 * s));
            kv[half_rope + i] = __half2float(__float2half_rn(x0 * s + x1 * c));
        }
        for (int d = rope_dim + tid; d < head_dim; d += 128) {
            kv[d] = __half2float(__float2half_rn(kv[d]));
        }
        for (int d = tid; d < head_dim; d += 128) {
            vv[d] = __half2float(__float2half_rn(vv[d]));
        }
    }
}

// Non-causal block GQA over committed F16 KV `[ctx_start..pos0)` + all `bs` F32 block rows:
// Grid: (n_head, bs, 1), Block: (32, 8, 1)
__global__ void __launch_bounds__(256, 2) dflash_noncausal_attn_f32(
    const float* __restrict__ q_blk,
    const float* __restrict__ k_blk,
    const float* __restrict__ v_blk,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    float* __restrict__ out_blk,
    int n_head,
    int n_head_kv,
    int head_dim,
    int bs,
    int ctx_start,
    int pos0
) {
    const int head = int(blockIdx.x);
    const int row = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int wid = int(threadIdx.y);
    const int tid = wid * 32 + lane;
    const int kv_dim = n_head_kv * head_dim;
    const int kv_h = head / (n_head / n_head_kv);
    const float scale = rsqrtf(float(head_dim));
    const size_t q_idx = size_t(row) * n_head + head;

    __shared__ float q_sh[256];
    __shared__ float wg_m[8];
    __shared__ float wg_l[8];
    __shared__ float wg_o[8][256];

    const float* q_h = q_blk + q_idx * head_dim;
    for (int d = tid; d < head_dim; d += 256) q_sh[d] = q_h[d] * scale;
    __syncthreads();

    float m_loc = -CUDART_INF_F;
    float l_loc = 0.0f;
    float o_reg[8] = {0.0f};
    const int ctx_len = max(0, pos0 - ctx_start);
    const int total_kv = ctx_len + bs;

    for (int idx = wid; idx < total_kv; idx += 8) {
        float dot_v = 0.0f;
        if (idx < ctx_len) {
            const int t = ctx_start + idx;
            const half* kh = k_cache + size_t(t) * kv_dim + size_t(kv_h) * head_dim;
            const half* vh = v_cache + size_t(t) * kv_dim + size_t(kv_h) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                dot_v = fmaf(q_sh[d], __half2float(kh[d]), dot_v);
            }
            dot_v = warp_reduce_sum(dot_v);
            const float m_new = fmaxf(m_loc, dot_v);
            const float alpha = (l_loc == 0.0f) ? 0.0f : expf(m_loc - m_new);
            const float p = expf(dot_v - m_new);
            l_loc = l_loc * alpha + p;
            m_loc = m_new;
            for (int d = lane; d < head_dim; d += 32) {
                const int c = d >> 5;
                o_reg[c] = o_reg[c] * alpha + p * __half2float(vh[d]);
            }
        } else {
            const int br = idx - ctx_len;
            const float* kh = k_blk + size_t(br) * kv_dim + size_t(kv_h) * head_dim;
            const float* vh = v_blk + size_t(br) * kv_dim + size_t(kv_h) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                dot_v = fmaf(q_sh[d], kh[d], dot_v);
            }
            dot_v = warp_reduce_sum(dot_v);
            const float m_new = fmaxf(m_loc, dot_v);
            const float alpha = (l_loc == 0.0f) ? 0.0f : expf(m_loc - m_new);
            const float p = expf(dot_v - m_new);
            l_loc = l_loc * alpha + p;
            m_loc = m_new;
            for (int d = lane; d < head_dim; d += 32) {
                const int c = d >> 5;
                o_reg[c] = o_reg[c] * alpha + p * vh[d];
            }
        }
    }

    if (lane == 0) {
        wg_m[wid] = m_loc;
        wg_l[wid] = l_loc;
    }
    for (int d = lane; d < head_dim; d += 32) wg_o[wid][d] = o_reg[d >> 5];
    __syncthreads();

    float m_all = -CUDART_INF_F;
    for (int w = 0; w < 8; ++w) {
        if (wg_l[w] > 0.0f) m_all = fmaxf(m_all, wg_m[w]);
    }
    float* out_h = out_blk + q_idx * head_dim;
    for (int d = tid; d < head_dim; d += 256) {
        float denom = 0.0f;
        float acc = 0.0f;
        for (int w = 0; w < 8; ++w) {
            if (wg_l[w] > 0.0f) {
                const float wt = expf(wg_m[w] - m_all);
                denom += wg_l[w] * wt;
                acc += wg_o[w][d] * wt;
            }
        }
        out_h[d] = acc / denom;
    }
}

__global__ void dflash_markov_prep_f32(
    float* __restrict__ joint_in,
    const float* __restrict__ rnn_state,
    const uint16_t* __restrict__ markov_w1_bf16,
    const float* __restrict__ hidden_blk,
    const uint32_t* __restrict__ draft_toks,
    const float* __restrict__ ctx_feat_ring,
    int rank,
    int n_embd,
    int step_idx,
    int ctx_ring_off
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(2 * rank + 2 * n_embd);
    if (idx >= total) return;
    const uint32_t prev_tok = draft_toks[step_idx];
    const size_t hid_off = size_t(step_idx + 1) * n_embd;
    if (idx < size_t(rank)) {
        joint_in[idx] = rnn_state[idx];
    } else if (idx < size_t(2 * rank)) {
        const size_t j = idx - rank;
        joint_in[idx] = bf16_to_f32(markov_w1_bf16[size_t(prev_tok) * rank + j]);
    } else if (idx < size_t(2 * rank + n_embd)) {
        joint_in[idx] = hidden_blk[hid_off + (idx - 2 * rank)];
    } else {
        joint_in[idx] = ctx_feat_ring[size_t(ctx_ring_off) + (idx - 2 * rank - n_embd)];
    }
}

__global__ void dflash_markov_gate_latent_f32(
    const float* __restrict__ joint_out,
    const float* __restrict__ joint_bias,
    float* __restrict__ rnn_state,
    float* __restrict__ latent,
    int rank
) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= rank) return;
    const float g_raw = joint_out[i] + joint_bias[i];
    const float c_raw = joint_out[rank + i] + joint_bias[rank + i];
    const float l_raw = joint_out[2 * rank + i] + joint_bias[2 * rank + i];
    const float gate = 1.0f / (1.0f + expf(-g_raw));
    const float cand = accurate_tanh_f32(c_raw);
    const float st = rnn_state[i];
    rnn_state[i] = cand + gate * (st - cand);
    latent[i] = accurate_tanh_f32(l_raw);
}

__global__ void dflash_markov_corr_f32(
    float* __restrict__ corr_hidden,
    const float* __restrict__ hidden_blk,
    const float* __restrict__ correction,
    const float* __restrict__ hidden_out_b,
    int n_embd,
    int step_idx
) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= n_embd) return;
    corr_hidden[i] = hidden_blk[size_t(step_idx + 1) * n_embd + i] + correction[i] + hidden_out_b[i];
}

// Pruned-vocabulary `Q8_0 x Q8_1` `__dp4a` LM-head GEMV (`vocab_map` -> `65536` rows):
// Grid: (ceil(n_pruned / 4), 1, 1), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 2) dflash_indexed_q8_0_q8_1_gemv(
    const BlockQ8_0* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    const uint32_t* __restrict__ vocab_map,
    float* __restrict__ out_logits,
    int n_pruned,
    int cols
) {
    const int idx = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int lane = int(threadIdx.x);
    if (idx >= n_pruned) return;
    const uint32_t row = vocab_map[idx];
    const int n_blocks = cols >> 5;
    const BlockQ8_0* w_row = w + size_t(row) * n_blocks;
    const int iqs = 2 * (lane & 3);
    float acc = 0.0f;
    for (int kbx = lane >> 2; kbx < n_blocks; kbx += 8) {
        const BlockQ8_0& wb = w_row[kbx];
        const BlockQ8_1& xb = xq[kbx];
        const int* xqs = reinterpret_cast<const int*>(xb.qs);
        int sumi = 0;
        sumi = dp4a_i8(load_i32_2b(wb.qs + 4 * iqs), xqs[iqs], sumi);
        sumi = dp4a_i8(load_i32_2b(wb.qs + 4 * (iqs + 1)), xqs[iqs + 1], sumi);
        acc += __half2float(wb.d) * __low2float(xb.ds) * float(sumi);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out_logits[idx] = acc;
}

// ============================================================================
// 11. `Qwen3.8-Flash-Next` (`qwen4exp`: HC + PLE + Sigmoid-GDN + QSA)
// ============================================================================

__global__ void qwen4exp_init_hc_f32(
    const float* __restrict__ x,
    float* __restrict__ res_hc,
    int n_embd,
    int hc,
    int n_tok
) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(n_tok) * hc * n_embd;
    if (idx >= total) return;
    const int d = int(idx % n_embd);
    const int t = int(idx / (size_t(hc) * n_embd));
    res_hc[idx] = x[size_t(t) * n_embd + d];
}

// Per-stream `(c, tok)` RMSNorm: `xn[t, c, d] = rms(res_hc[t, c, :], eps) * w_norm[c, d]`
// Grid: (hc, n_tok, 1), Block: (256, 1, 1)
__global__ void qwen4exp_hc_rms_norm_f32(
    const float* __restrict__ res_hc,
    const float* __restrict__ w_norm,
    float* __restrict__ xn,
    int n_embd,
    int hc,
    float eps
) {
    const int c = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    __shared__ float warp_sums[8];
    const size_t off = (size_t(t) * hc + c) * n_embd;
    const float* src = res_hc + off;
    const float* wn = w_norm + size_t(c) * n_embd;
    float* dst = xn + off;
    float ss = 0.0f;
    for (int d = tid; d < n_embd; d += 256) ss = fmaf(src[d], src[d], ss);
    const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(n_embd) + eps);
    for (int d = tid; d < n_embd; d += 256) dst[d] = src[d] * inv * wn[d];
}

__global__ void qwen4exp_scale_silu_f32(float* __restrict__ x, float scale, size_t n) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = x[idx] * scale;
    x[idx] = v / (1.0f + expf(-v));
}

// `mixed[t, d] = (1 / hc) * sum_{c=0..hc-1} xn[t, c, d] * sigmoid(gate[t, c, d])`
// Grid: (ceil(n_embd / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void qwen4exp_hc_mix_gate_f32(
    const float* __restrict__ xn,
    const float* __restrict__ gate,
    float* __restrict__ mixed,
    int n_embd,
    int hc
) {
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int t = int(blockIdx.y);
    if (d >= n_embd) return;
    const size_t base = size_t(t) * hc * n_embd;
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const size_t idx = base + size_t(c) * n_embd + d;
        const float g = gate[idx];
        acc += xn[idx] * (1.0f / (1.0f + expf(-g)));
    }
    mixed[size_t(t) * n_embd + d] = acc * (1.0f / float(hc));
}

// `res_hc[t, c, d] += sub_out[t, d] * (2 * sigmoid(inject[t, c] / hc))`
// Grid: (ceil((hc * n_embd) / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void qwen4exp_hc_combine_f32(
    float* __restrict__ res_hc,
    const float* __restrict__ sub_out,
    const float* __restrict__ inject,
    int n_embd,
    int hc
) {
    const int cd = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int t = int(blockIdx.y);
    if (cd >= hc * n_embd) return;
    const int c = cd / n_embd;
    const int d = cd % n_embd;
    const float inj = inject[size_t(t) * hc + c] * (1.0f / float(hc));
    const float w = 2.0f / (1.0f + expf(-inj));
    res_hc[size_t(t) * hc * n_embd + cd] += sub_out[size_t(t) * n_embd + d] * w;
}

// PLE signed-sqrt Sigmoid gate + pre-conv RMSNorm per `(c, tok)`:
// Grid: (hc, n_tok, 1), Block: (256, 1, 1)
__global__ void qwen4exp_ple_gate_norm_f32(
    const float* __restrict__ res_hc,
    const float* __restrict__ key,
    const float* __restrict__ val,
    const float* __restrict__ w_nk,
    const float* __restrict__ w_nq,
    const float* __restrict__ w_nc,
    float* __restrict__ gated_out,
    float* __restrict__ normed_out,
    int n_embd,
    int hc,
    float eps
) {
    const int c = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    __shared__ float warp_sums[8];
    const size_t off = (size_t(t) * hc + c) * n_embd;
    const float* qc_in = res_hc + off;
    const float* kc_in = key + off;
    const float* v_in = val + size_t(t) * n_embd;
    const float* nkc = w_nk + size_t(c) * n_embd;
    const float* nqc = w_nq + size_t(c) * n_embd;
    const float* ncc = w_nc + size_t(c) * n_embd;

    float sq_k = 0.0f, sq_q = 0.0f;
    for (int d = tid; d < n_embd; d += 256) {
        sq_k = fmaf(kc_in[d], kc_in[d], sq_k);
        sq_q = fmaf(qc_in[d], qc_in[d], sq_q);
    }
    const float inv_k = rsqrtf(block256_reduce_sum(sq_k, tid, warp_sums) / float(n_embd) + eps);
    const float inv_q = rsqrtf(block256_reduce_sum(sq_q, tid, warp_sums) / float(n_embd) + eps);

    float dot_v = 0.0f;
    for (int d = tid; d < n_embd; d += 256) {
        dot_v = fmaf((kc_in[d] * inv_k * nkc[d]), (qc_in[d] * inv_q * nqc[d]), dot_v);
    }
    const float s = block256_reduce_sum(dot_v, tid, warp_sums) * rsqrtf(float(n_embd));
    const float mag = sqrtf(fmaxf(fabsf(s), 1.0e-6f));
    const float signed_mag = (s > 0.0f) ? mag : ((s < 0.0f) ? -mag : 0.0f);
    const float gate = 1.0f / (1.0f + expf(-signed_mag));

    float sq_g = 0.0f;
    for (int d = tid; d < n_embd; d += 256) {
        const float gv = v_in[d] * gate;
        gated_out[off + d] = gv;
        sq_g = fmaf(gv, gv, sq_g);
    }
    const float inv_g = rsqrtf(block256_reduce_sum(sq_g, tid, warp_sums) / float(n_embd) + eps);
    for (int d = tid; d < n_embd; d += 256) {
        normed_out[off + d] = gated_out[off + d] * inv_g * ncc[d];
    }
}

// Dilated causal depthwise conv (`hist_len = (kern - 1) * stride_ngram`, e.g. `3 * 4 = 12`) + SiLU + residual add:
// Grid: (ceil(hc_dim / 256), 1, 1), Block: (256, 1, 1)
__global__ void qwen4exp_ple_dilated_conv_add_f32(
    float* __restrict__ res_hc,
    const float* __restrict__ gated,
    const float* __restrict__ normed,
    const float* __restrict__ conv_w, // [hc_dim, kern]
    float* __restrict__ ple_hist,     // [hist_len, hc_dim]
    int hc_dim,
    int kern,
    int stride_ngram,
    int n_tok
) {
    const int ch = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (ch >= hc_dim) return;
    const int hist_len = (kern - 1) * stride_ngram; // <= 16
    float hist[16];
    for (int h = 0; h < hist_len; ++h) {
        hist[h] = ple_hist[size_t(h) * hc_dim + ch];
    }
    const float* w_ch = conv_w + size_t(ch) * kern;

    for (int t = 0; t < n_tok; ++t) {
        const size_t idx = size_t(t) * hc_dim + ch;
        const float cur = normed[idx];
        float acc = cur * w_ch[kern - 1];
        for (int k = 0; k < kern - 1; ++k) {
            const int lag = (kern - 1 - k) * stride_ngram; // hist_len - lag in 0..hist_len-1
            acc = fmaf(hist[hist_len - lag], w_ch[k], acc);
        }
        for (int h = 0; h < hist_len - 1; ++h) hist[h] = hist[h + 1];
        if (hist_len > 0) hist[hist_len - 1] = cur;
        const float conv_act = acc / (1.0f + expf(-acc));
        res_hc[idx] += gated[idx] + conv_act;
    }

    for (int h = 0; h < hist_len; ++h) {
        ple_hist[size_t(h) * hc_dim + ch] = hist[h];
    }
}

// `Qwen4Exp` GDN output gate uses `sigmoid(z)` instead of `silu(z)` (`qwen4exp.cpp:489`!)
// Grid: (h_v, 1, 1), Block: (32, 1, 1)
__global__ void gdn_rms_norm_gated_sigmoid_f32(
    float* __restrict__ out,
    const float* __restrict__ z,
    const float* __restrict__ norm_w,
    int dv_dim,
    float eps
) {
    const int hv = int(blockIdx.x);
    const int lane = int(threadIdx.x);
    float* y = out + size_t(hv) * dv_dim;
    const float* zh = z + size_t(hv) * dv_dim;

    float sq = 0.0f;
    for (int d = lane; d < dv_dim; d += 32) sq += y[d] * y[d];
    sq = warp_reduce_sum(sq);
    const float scale = rsqrtf(sq / float(dv_dim) + eps);
    for (int d = lane; d < dv_dim; d += 32) {
        const float zv = zh[d];
        y[d] = y[d] * scale * norm_w[d] * (1.0f / (1.0f + expf(-zv)));
    }
}

// QSA indexer prep + block pooling (`idx_dim = 128`):
// `blockIdx.y = 0..n_idx_head-1`: RMSNorm + NeoX RoPE on `idx_q[t, h, :]`
// `blockIdx.y = n_idx_head`: store `idx_k_raw[t, :]` as F16 into `idx_k_cache[pos, :]`, and if `(pos + 1) % pool == 0`,
// mean-pool `pool` F16 raw keys, RMSNorm with `idx_k_norm`, NeoX-rotate to `pos - (pool - 1)`, store F16 into `idx_pooled[blk, :]`!
// Grid: (n_tok, n_idx_head + 1, 1), Block: (128, 1, 1)
__global__ void qwen4exp_qsa_idx_prep_pool_f32(
    float* __restrict__ idx_q,              // [n_tok, n_idx_head, idx_dim]
    const float* __restrict__ idx_k_raw,    // [n_tok, idx_dim]
    const float* __restrict__ idx_q_norm,   // [idx_dim]
    const float* __restrict__ idx_k_norm,   // [idx_dim]
    const float* __restrict__ inv_freqs,    // [rope_dim / 2]
    half* __restrict__ idx_k_cache,         // [max_ctx, idx_dim]
    half* __restrict__ idx_pooled,          // [max_ctx / pool, idx_dim]
    int n_idx_head,
    int idx_dim,
    int rope_dim,
    int pool,
    float eps,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int t = int(blockIdx.x);
    const int part = int(blockIdx.y);
    const int i = int(threadIdx.x);
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + t;
    const int half_rope = rope_dim >> 1;
    __shared__ float ws[4];
    __shared__ float buf[128];

    if (part < n_idx_head) {
        float* qh = idx_q + (size_t(t) * n_idx_head + part) * idx_dim;
        const float v = (i < idx_dim) ? qh[i] : 0.0f;
        float ss = warp_reduce_sum(v * v);
        if ((i & 31) == 0) ws[i >> 5] = ss;
        __syncthreads();
        if (i < 32) {
            const float tot = warp_reduce_sum((i < 4) ? ws[i] : 0.0f);
            if (i == 0) ws[0] = rsqrtf(tot / float(idx_dim) + eps);
        }
        __syncthreads();
        if (i < idx_dim) buf[i] = v * ws[0] * idx_q_norm[i];
        __syncthreads();
        if (i < idx_dim) {
            float out = buf[i];
            if (i < rope_dim) {
                const int k = i % half_rope;
                float s, c;
                sincosf(float(pos) * inv_freqs[k], &s, &c);
                const float x0 = buf[k], x1 = buf[k + half_rope];
                out = (i < half_rope) ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
            }
            qh[i] = out;
        }
    } else {
        if (i < idx_dim) {
            idx_k_cache[size_t(pos) * idx_dim + i] = __float2half_rn(idx_k_raw[size_t(t) * idx_dim + i]);
        }
        if ((pos + 1) % pool != 0) return;
        float v = 0.0f;
        if (i < idx_dim) {
            const int p0 = pos - (pool - 1);
            for (int m = 0; m < pool; ++m) {
                const int pm = p0 + m;
                const half hv = (pm >= (pos - t))
                    ? __float2half_rn(idx_k_raw[size_t(pm - (pos - t)) * idx_dim + i])
                    : idx_k_cache[size_t(pm) * idx_dim + i];
                v += __half2float(hv);
            }
            v *= 1.0f / float(pool);
        }
        float ss = warp_reduce_sum(v * v);
        if ((i & 31) == 0) ws[i >> 5] = ss;
        __syncthreads();
        if (i < 32) {
            const float tot = warp_reduce_sum((i < 4) ? ws[i] : 0.0f);
            if (i == 0) ws[0] = rsqrtf(tot / float(idx_dim) + eps);
        }
        __syncthreads();
        if (i < idx_dim) buf[i] = v * ws[0] * idx_k_norm[i];
        __syncthreads();
        if (i < idx_dim) {
            float out = buf[i];
            const int p_first = pos - (pool - 1);
            if (i < rope_dim) {
                const int k = i % half_rope;
                float s, c;
                sincosf(float(p_first) * inv_freqs[k], &s, &c);
                const float x0 = buf[k], x1 = buf[k + half_rope];
                out = (i < half_rope) ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
            }
            const int blk_j = (pos + 1) / pool - 1;
            idx_pooled[size_t(blk_j) * idx_dim + i] = __float2half_rn(out);
        }
    }
}

// QSA sparse block selection (`n_vis_blocks > top_blocks` -> marks top-`top_blocks` blocks in `allowed[t, 0..n_vis_blocks]`):
// Grid: (n_tok, 1, 1), Block: (128, 1, 1)
__global__ void qwen4exp_qsa_select_blocks_f32(
    const float* __restrict__ idx_q,        // [n_tok, n_idx_head, idx_dim]
    const half* __restrict__ idx_pooled,    // [max_blocks, idx_dim]
    float* __restrict__ scores_buf,         // [n_tok, max_blocks]
    uint8_t* __restrict__ allowed_buf,      // [n_tok, max_blocks]
    int n_idx_head,
    int idx_dim,
    int pool,
    int top_blocks,
    int max_blocks,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    if (pool <= 0) return;
    const int t = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const int lane = tid & 31;
    const int wid = tid >> 5; // 0..3
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + t;
    const int n_vis = (pos + 1) / pool;
    if (n_vis <= top_blocks) return;

    float* sc_row = scores_buf + size_t(t) * max_blocks;
    uint8_t* al_row = allowed_buf + size_t(t) * max_blocks;
    for (int b = tid; b < n_vis; b += 128) {
        al_row[b] = 0;
    }
    __syncthreads();

    const float idx_scale = rsqrtf(float(idx_dim));
    const float* qt = idx_q + size_t(t) * n_idx_head * idx_dim;
    for (int b = wid; b < n_vis; b += 4) {
        const half* pk = idx_pooled + size_t(b) * idx_dim;
        float sc = 0.0f;
        for (int h = 0; h < n_idx_head; ++h) {
            const float* qh = qt + size_t(h) * idx_dim;
            float dot = 0.0f;
            for (int j = lane; j < idx_dim; j += 32) {
                dot = fmaf(qh[j], __half2float(pk[j]), dot);
            }
            dot = warp_reduce_sum(dot);
            sc += fmaxf(dot, 0.0f) * idx_scale;
        }
        if (lane == 0) sc_row[b] = sc;
    }
    __syncthreads();

    __shared__ float sh_v[4];
    __shared__ int sh_i[4];
    for (int k = 0; k < top_blocks; ++k) {
        float best_v = -CUDART_INF_F;
        int best_i = 0x7fffffff;
        for (int b = tid; b < n_vis; b += 128) {
            if (al_row[b] == 0) {
                const float v = sc_row[b];
                if (v > best_v || (v == best_v && b < best_i)) {
                    best_v = v;
                    best_i = b;
                }
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset, 32);
            const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            sh_v[wid] = best_v;
            sh_i[wid] = best_i;
        }
        __syncthreads();
        if (tid == 0) {
            float wv = sh_v[0];
            int wi = sh_i[0];
            for (int w = 1; w < 4; ++w) {
                if (sh_v[w] > wv || (sh_v[w] == wv && sh_i[w] < wi)) {
                    wv = sh_v[w];
                    wi = sh_i[w];
                }
            }
            if (wi < n_vis) al_row[wi] = 1;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(128) qwen4exp_qsa_sparse_attn_f16_kv(
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const uint8_t* __restrict__ allowed_buf,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int pool,
    int top_blocks,
    int max_blocks,
    float attn_scale,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int warp_id = int(threadIdx.y);
    const int lane = int(threadIdx.x);

    const int kv_h = h / (n_head / n_head_kv);
    const int kv_dim = n_head_kv * head_dim;
    const int cur_pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int n_vis = (pool > 0) ? ((cur_pos + 1) / pool) : 0;
    const bool use_mask = (pool > 0) && (n_vis > top_blocks);
    const uint8_t* al_row = allowed_buf + size_t(tok) * max_blocks;

    const float* q_vec = q + (size_t(tok) * n_head + h) * head_dim;
    float q_reg[8] = {0.0f};
    float o_reg[8] = {0.0f};
    const int elems_per_lane = (head_dim + 31) >> 5;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = e * 32 + lane;
        if (d < head_dim) q_reg[e] = q_vec[d];
    }

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    for (int p = warp_id; p <= cur_pos; p += 4) {
        if (use_mask) {
            const int blk = p / pool;
            if (blk < n_vis && al_row[blk] == 0) continue;
        }
        const half* kp = k_cache + size_t(p) * kv_dim + size_t(kv_h) * head_dim;
        float dot = 0.0f;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = e * 32 + lane;
            if (d < head_dim) dot += q_reg[e] * __half2float(kp[d]);
        }
        dot = warp_reduce_sum(dot) * attn_scale;
        const float m_new = fmaxf(m_i, dot);
        const float alpha = expf(m_i - m_new);
        const float weight = expf(dot - m_new);
        l_i = l_i * alpha + weight;
        m_i = m_new;

        const half* vp = v_cache + size_t(p) * kv_dim + size_t(kv_h) * head_dim;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = e * 32 + lane;
            if (d < head_dim) o_reg[e] = o_reg[e] * alpha + weight * __half2float(vp[d]);
        }
    }

    __shared__ float smem_m[4];
    __shared__ float smem_l[4];
    __shared__ float smem_o[4 * 256];
    if (lane == 0) {
        smem_m[warp_id] = m_i;
        smem_l[warp_id] = l_i;
    }
    for (int e = 0; e < elems_per_lane; ++e) {
        const int d = e * 32 + lane;
        if (d < head_dim) smem_o[warp_id * 256 + d] = o_reg[e];
    }
    __syncthreads();

    if (warp_id == 0) {
        const float m_max = fmaxf(fmaxf(smem_m[0], smem_m[1]), fmaxf(smem_m[2], smem_m[3]));
        float l_sum = 0.0f;
        float w_scale[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            w_scale[w] = (smem_l[w] > 0.0f) ? expf(smem_m[w] - m_max) : 0.0f;
            l_sum += smem_l[w] * w_scale[w];
        }
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* out_vec = out + (size_t(tok) * n_head + h) * head_dim;
        for (int e = 0; e < elems_per_lane; ++e) {
            const int d = e * 32 + lane;
            if (d < head_dim) {
                const float acc = smem_o[0 * 256 + d] * w_scale[0]
                                + smem_o[1 * 256 + d] * w_scale[1]
                                + smem_o[2 * 256 + d] * w_scale[2]
                                + smem_o[3 * 256 + d] * w_scale[3];
                out_vec[d] = acc * inv_l;
            }
        }
    }
}

__global__ void __launch_bounds__(128) qwen4exp_qsa_sparse_attn_q8_0_kv(
    const float* __restrict__ q,
    const BlockQ8_0* __restrict__ k_cache,
    const BlockQ8_0* __restrict__ v_cache,
    const uint8_t* __restrict__ allowed_buf,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int pool,
    int top_blocks,
    int max_blocks,
    float attn_scale,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int warp_id = int(threadIdx.y);
    const int lane = int(threadIdx.x);

    const int kv_h = h / (n_head / n_head_kv);
    const int kv_blks_per_tok = (n_head_kv * head_dim) >> 5;
    const int head_blks = head_dim >> 5;
    const int cur_pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int n_vis = (pool > 0) ? ((cur_pos + 1) / pool) : 0;
    const bool use_mask = (pool > 0) && (n_vis > top_blocks);
    const uint8_t* al_row = allowed_buf + size_t(tok) * max_blocks;

    const float* q_vec = q + (size_t(tok) * n_head + h) * head_dim;
    float q_reg[8] = {0.0f};
    float o_reg[8] = {0.0f};
    for (int b = 0; b < head_blks; ++b) q_reg[b] = q_vec[b * 32 + lane];

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    for (int p = warp_id; p <= cur_pos; p += 4) {
        if (use_mask) {
            const int blk = p / pool;
            if (blk < n_vis && al_row[blk] == 0) continue;
        }
        const BlockQ8_0* kp = k_cache + size_t(p) * kv_blks_per_tok + size_t(kv_h) * head_blks;
        float dot = 0.0f;
        for (int b = 0; b < head_blks; ++b) {
            dot += q_reg[b] * (__half2float(kp[b].d) * float(kp[b].qs[lane]));
        }
        dot = warp_reduce_sum(dot) * attn_scale;
        const float m_new = fmaxf(m_i, dot);
        const float alpha = expf(m_i - m_new);
        const float weight = expf(dot - m_new);
        l_i = l_i * alpha + weight;
        m_i = m_new;

        const BlockQ8_0* vp = v_cache + size_t(p) * kv_blks_per_tok + size_t(kv_h) * head_blks;
        for (int b = 0; b < head_blks; ++b) {
            o_reg[b] = o_reg[b] * alpha + weight * (__half2float(vp[b].d) * float(vp[b].qs[lane]));
        }
    }

    __shared__ float smem_m[4];
    __shared__ float smem_l[4];
    __shared__ float smem_o[4 * 256];
    if (lane == 0) {
        smem_m[warp_id] = m_i;
        smem_l[warp_id] = l_i;
    }
    for (int b = 0; b < head_blks; ++b) smem_o[warp_id * 256 + b * 32 + lane] = o_reg[b];
    __syncthreads();

    if (warp_id == 0) {
        const float m_max = fmaxf(fmaxf(smem_m[0], smem_m[1]), fmaxf(smem_m[2], smem_m[3]));
        float l_sum = 0.0f;
        float w_scale[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            w_scale[w] = (smem_l[w] > 0.0f) ? expf(smem_m[w] - m_max) : 0.0f;
            l_sum += smem_l[w] * w_scale[w];
        }
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float* out_vec = out + (size_t(tok) * n_head + h) * head_dim;
        for (int b = 0; b < head_blks; ++b) {
            const int d = b * 32 + lane;
            const float acc = smem_o[0 * 256 + d] * w_scale[0]
                            + smem_o[1 * 256 + d] * w_scale[1]
                            + smem_o[2 * 256 + d] * w_scale[2]
                            + smem_o[3 * 256 + d] * w_scale[3];
            out_vec[d] = acc * inv_l;
        }
    }
}

// ============================================================================
// 12. `Tencent Hy4-Flash` (`.strata` v1.0 `S_K` + `Q8_ROW` + `MLA` + `LatentMoE`)
// ============================================================================

struct BlockQ8_1_G64 {
    int8_t qs[64];
    float d;
    float s; // d * sum(qs)
}; // 72 bytes
static_assert(sizeof(BlockQ8_1_G64) == 72, "BlockQ8_1_G64 size");

struct StrataExpertGpuEntry {
    uint16_t expert_id;
    uint8_t gu_code_id;
    uint8_t dn_code_id;
    uint16_t cluster_id;
    uint16_t flags;
    uint64_t body_offset;
    uint32_t body_bytes;
    uint32_t dn_offset;
    uint32_t aux0;
    uint32_t aux1;
}; // 32 bytes
static_assert(sizeof(StrataExpertGpuEntry) == 32, "StrataExpertGpuEntry size");

__constant__ int8_t c_s125_codebook[32][4] = {
    {-1, -1, -1, -1}, {-1, -1, -1,  1}, {-1, -1,  1, -1}, {-1, -1,  1,  1},
    {-1,  1, -1, -1}, {-1,  1, -1,  1}, {-1,  1,  1, -1}, {-1,  1,  1,  1},
    { 1, -1, -1, -1}, { 1, -1, -1,  1}, { 1, -1,  1, -1}, { 1, -1,  1,  1},
    { 1,  1, -1, -1}, { 1,  1, -1,  1}, { 1,  1,  1, -1}, { 1,  1,  1,  1},
    { 2,  0,  0,  0}, {-2,  0,  0,  0}, { 0,  2,  0,  0}, { 0, -2,  0,  0},
    { 0,  0,  2,  0}, { 0,  0, -2,  0}, { 0,  0,  0,  2}, { 0,  0,  0, -2},
    { 0,  0,  0,  0}, { 2,  2,  0,  0}, {-2, -2,  0,  0}, { 0,  0,  2,  2},
    { 0,  0, -2, -2}, { 2, -2,  0,  0}, { 0,  0,  2, -2}, { 2,  0,  2,  0},
};

__device__ __forceinline__ size_t sk_sb_bytes(int code) {
    switch (code) {
        case 0: return 48;  // S1.25
        case 1: return 58;  // S1.5
        case 2: return 70;  // S2_K
        case 3: return 102; // S3_K
        case 4: return 134; // S4_K
        case 5: return 166; // S5_K
        case 6: return 198; // S6_K
        case 8: return 262; // S8_K
        default: return 0;
    }
}

__device__ __forceinline__ float load_f16_unaligned(const uint8_t* p) {
    const uint16_t bits = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    return __half2float(__ushort_as_half(bits));
}

// Quantizes `x[n_rows, cols]` (`cols % 64 == 0`) into 64-element `BlockQ8_1_G64`:
// Grid: (ceil((cols / 64) / 2), n_rows, 1), Block: (32, 2, 1)
__global__ void quantize_q8_1_g64_f32(
    const float* __restrict__ x,
    BlockQ8_1_G64* __restrict__ out,
    int cols,
    int n_rows
) {
    const int n_b64 = cols >> 6;
    const int b64 = int(blockIdx.x) * 2 + int(threadIdx.y);
    const int row = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    if (b64 >= n_b64 || row >= n_rows) return;

    const float* src = x + size_t(row) * cols + b64 * 64;
    const float v0 = src[2 * lane];
    const float v1 = src[2 * lane + 1];
    const float amax = warp_reduce_max(fmaxf(fabsf(v0), fabsf(v1)));
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int q0 = max(-127, min(127, __float2int_rn(v0 * id)));
    const int q1 = max(-127, min(127, __float2int_rn(v1 * id)));
    const int q_sum = int(warp_reduce_sum(float(q0 + q1)));

    BlockQ8_1_G64& dst = out[size_t(row) * n_b64 + b64];
    dst.qs[2 * lane] = int8_t(q0);
    dst.qs[2 * lane + 1] = int8_t(q1);
    if (lane == 0) {
        dst.d = d;
        dst.s = d * float(q_sum);
    }
}

// Unpacks 8 weights (`g * 64 + sub * 8 + 0..7`, where `lane = g * 8 + sub`) of a 256-weight `S_K` superblock
// into two `int32` words (`q_lo4`, `q_hi4`) + `half_bias` (`0` for signed `S8_K`/`S1.5`/`S1.25`, `1.5..31.5` for `S2..S6`)!
__device__ __forceinline__ void unpack_sk_lane8(
    const uint8_t* __restrict__ sb_ptr,
    int sk_code,
    int lane,
    float& eff_scale,
    float& half_bias,
    int& q_lo4,
    int& q_hi4
) {
    const int g = lane >> 3;        // 0..3 (which 64-elem sub-group)
    const int sub = lane & 7;       // 0..7 (which 8-elem slice inside the 64-elem sub-group)
    const int elem0 = lane * 8;     // 0..248
    const float d_sb = load_f16_unaligned(sb_ptr);
    eff_scale = d_sb * (float(sb_ptr[2 + g]) * (1.0f / 255.0f));
    const uint8_t* pl = sb_ptr + 6;
    half_bias = 0.0f;

    int8_t w8[8];
    if (sk_code == 8) { // S8_K
#pragma unroll
        for (int j = 0; j < 8; ++j) w8[j] = int8_t(pl[elem0 + j]);
    } else if (sk_code == 4) { // S4_K
        half_bias = 7.5f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t b = pl[(elem0 >> 1) + j];
            w8[2 * j + 0] = int8_t(b & 0x0Fu);
            w8[2 * j + 1] = int8_t(b >> 4);
        }
    } else if (sk_code == 5) { // S5_K
        half_bias = 15.5f;
        const uint8_t h = pl[128 + lane];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t b = pl[(elem0 >> 1) + j];
            w8[2 * j + 0] = int8_t((b & 0x0Fu) | (((h >> (2 * j + 0)) & 1u) << 4));
            w8[2 * j + 1] = int8_t((b >> 4) | (((h >> (2 * j + 1)) & 1u) << 4));
        }
    } else if (sk_code == 6) { // S6_K
        half_bias = 31.5f;
        const uint8_t h0 = pl[128 + 2 * lane + 0];
        const uint8_t h1 = pl[128 + 2 * lane + 1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t b = pl[(elem0 >> 1) + j];
            const uint8_t hi_pair = (j < 2) ? (h0 >> (4 * j)) : (h1 >> (4 * (j - 2)));
            w8[2 * j + 0] = int8_t((b & 0x0Fu) | ((hi_pair & 0x03u) << 4));
            w8[2 * j + 1] = int8_t((b >> 4) | (((hi_pair >> 2) & 0x03u) << 4));
        }
    } else if (sk_code == 2) { // S2_K
        half_bias = 1.5f;
        const uint8_t b0 = pl[2 * lane + 0];
        const uint8_t b1 = pl[2 * lane + 1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w8[j] = int8_t((b0 >> (2 * j)) & 0x03u);
            w8[4 + j] = int8_t((b1 >> (2 * j)) & 0x03u);
        }
    } else if (sk_code == 3) { // S3_K
        half_bias = 3.5f;
        const uint8_t b0 = pl[2 * lane + 0];
        const uint8_t b1 = pl[2 * lane + 1];
        const uint8_t h = pl[64 + lane];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w8[j] = int8_t(((b0 >> (2 * j)) & 0x03u) | (((h >> j) & 1u) << 2));
            w8[4 + j] = int8_t(((b1 >> (2 * j)) & 0x03u) | (((h >> (4 + j)) & 1u) << 2));
        }
    } else if (sk_code == 1) { // S1.5 (5 trits per byte)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = elem0 + j;
            int x = int(pl[idx / 5]);
            const int rem = idx % 5;
            for (int r = 0; r < rem; ++r) x /= 3;
            w8[j] = int8_t((x % 3) - 1);
        }
    } else { // S1.25 (`sk_code == 0`)
        const uint8_t fb = pl[40 + (g >> 1)];
        const uint8_t nib = (g & 1) ? (fb >> 4) : (fb & 0x0Fu);
        eff_scale *= (0.5f + float(nib) * (1.0f / 15.0f));
        // Each 8 elements = two 4-element codebook vectors (`vec_idx = 2 * lane + v`, `g40 = vec_idx >> 3`, `j5 = vec_idx & 7`)
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int vec_idx = 2 * lane + v;
            const int g40 = vec_idx >> 3;
            const int j5 = vec_idx & 7;
            const uint8_t* p5 = pl + g40 * 5;
            const uint64_t bits40 = uint64_t(p5[0]) | (uint64_t(p5[1]) << 8) | (uint64_t(p5[2]) << 16)
                                  | (uint64_t(p5[3]) << 24) | (uint64_t(p5[4]) << 32);
            const uint32_t cb_idx = uint32_t((bits40 >> (5 * j5)) & 0x1Fu);
            w8[4 * v + 0] = c_s125_codebook[cb_idx][0];
            w8[4 * v + 1] = c_s125_codebook[cb_idx][1];
            w8[4 * v + 2] = c_s125_codebook[cb_idx][2];
            w8[4 * v + 3] = c_s125_codebook[cb_idx][3];
        }
    }

    q_lo4 = int(uint8_t(w8[0])) | (int(uint8_t(w8[1])) << 8) | (int(uint8_t(w8[2])) << 16) | (int(uint8_t(w8[3])) << 24);
    q_hi4 = int(uint8_t(w8[4])) | (int(uint8_t(w8[5])) << 8) | (int(uint8_t(w8[6])) << 16) | (int(uint8_t(w8[7])) << 24);
    (void)sub;
}

// Trunk `S_K x BlockQ8_1_G64` `__dp4a` MMVQ (`n_tok = 1..8`):
// Grid: (`rows` when `n_tok == 1 && cols >= 1024`, else `ceil(rows / 4)`, 1, 1), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) mmvq_sk_q8_1_g64(
    const uint8_t* __restrict__ w,
    const BlockQ8_1_G64* __restrict__ xq,
    float* __restrict__ out,
    int sk_code,
    int rows,
    int cols,
    int n_tok
) {
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);
    const int n_sb = cols >> 8;
    const int n_b64 = cols >> 6;
    const size_t sb_bytes = sk_sb_bytes(sk_code);
    const int g = lane >> 3;
    const int sub = lane & 7;

    if (n_tok == 1 && cols >= 1024) {
        const int row = int(blockIdx.x);
        if (row >= rows) return;
        const uint8_t* w_row = w + size_t(row) * n_sb * sb_bytes;
        float acc = 0.0f;
        for (int sb = warp_id; sb < n_sb; sb += 4) {
            float eff_sc, half_bias;
            int q0, q1;
            unpack_sk_lane8(w_row + size_t(sb) * sb_bytes, sk_code, lane, eff_sc, half_bias, q0, q1);
            const BlockQ8_1_G64& xb = xq[sb * 4 + g];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + 2 * sub;
            int dot = dp4a_i8(q0, xqs[0], 0);
            dot = dp4a_i8(q1, xqs[1], dot);
            float term = eff_sc * xb.d * float(dot);
            if (sub == 0 && half_bias != 0.0f) {
                term -= eff_sc * half_bias * xb.s;
            }
            acc += term;
        }
        __shared__ float partial[3][32];
        if (warp_id > 0) partial[warp_id - 1][lane] = acc;
        __syncthreads();
        if (warp_id == 0) {
            acc += partial[0][lane] + partial[1][lane] + partial[2][lane];
            acc = warp_reduce_sum(acc);
            if (lane == 0) out[row] = acc;
        }
        return;
    }

    const int row = int(blockIdx.x) * 4 + warp_id;
    if (row >= rows) return;
    const uint8_t* w_row = w + size_t(row) * n_sb * sb_bytes;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int sb = 0; sb < n_sb; ++sb) {
        float eff_sc, half_bias;
        int q0, q1;
        unpack_sk_lane8(w_row + size_t(sb) * sb_bytes, sk_code, lane, eff_sc, half_bias, q0, q1);
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1_G64& xb = xq[size_t(t) * n_b64 + sb * 4 + g];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + 2 * sub;
            int dot = dp4a_i8(q0, xqs[0], 0);
            dot = dp4a_i8(q1, xqs[1], dot);
            float term = eff_sc * xb.d * float(dot);
            if (sub == 0 && half_bias != 0.0f) {
                term -= eff_sc * half_bias * xb.s;
            }
            acc[t] += term;
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[size_t(t) * rows + row] = sum;
    }
}

// `Q8_ROW` (`float sc[rows]` at `W[0..4*rows]` + `int8 q[rows * cols]` at `W[4*rows..]`) `__dp4a` MMVQ (`n_tok = 1..8`):
// Grid: (ceil(rows / 4), 1, 1), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) mmvq_q8row_q8_1(
    const uint8_t* __restrict__ w_q8row,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int lane = int(threadIdx.x);
    if (row >= rows) return;

    const float row_sc = reinterpret_cast<const float*>(w_q8row)[row];
    const int8_t* q_row = reinterpret_cast<const int8_t*>(w_q8row + size_t(rows) * 4) + size_t(row) * cols;
    const int n_q8 = cols >> 5;
    const int iqs = 2 * (lane & 3);
    float acc[MMVQ_MAX_TOK] = {0.0f};

    for (int kbx = lane >> 2; kbx < n_q8; kbx += 8) {
        const int w0 = reinterpret_cast<const int*>(q_row + kbx * 32)[iqs];
        const int w1 = reinterpret_cast<const int*>(q_row + kbx * 32)[iqs + 1];
        for (int t = 0; t < n_tok; ++t) {
            const BlockQ8_1& xb = xq[size_t(t) * n_q8 + kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(w0, xqs[iqs], 0);
            sumi = dp4a_i8(w1, xqs[iqs + 1], sumi);
            acc[t] += __low2float(xb.ds) * float(sumi);
        }
    }

    for (int t = 0; t < n_tok; ++t) {
        const float sum = warp_reduce_sum(acc[t]) * row_sc;
        if (lane == 0) out[size_t(t) * rows + row] = sum;
    }
}

// Dequantizes `embed_tokens.weight` (`S_K` or `F32`/`Q8_ROW` `[vocab, H]`) for `token_ids[0..n_tok]` and replicates into `x_hc[n_tok, hc, H]`:
// Grid: (H / 256, n_tok, 1), Block: (32, 1, 1)
__global__ void hy4_embed_sk_f32(
    const uint8_t* __restrict__ emb_sk,
    const uint32_t* __restrict__ token_ids,
    float* __restrict__ x_hc,
    int sk_code,
    int hidden,
    int hc,
    int vocab_rows
) {
    const int sb = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x); // 0..31 (each lane unpacks 8 elements = 256 elements per block)
    const uint32_t id = token_ids[tok];
    const int base_col = sb * 256 + lane * 8;

    float vals[8];
    if (sk_code == 101) { // F32
        const float* row = reinterpret_cast<const float*>(emb_sk) + size_t(id) * hidden + base_col;
        for (int j = 0; j < 8; ++j) vals[j] = row[j];
    } else if (sk_code == 100) { // Q8_ROW
        const float sc = reinterpret_cast<const float*>(emb_sk)[id];
        const int8_t* row = reinterpret_cast<const int8_t*>(emb_sk + size_t(vocab_rows) * 4) + size_t(id) * hidden + base_col;
        for (int j = 0; j < 8; ++j) vals[j] = float(row[j]) * sc;
    } else {
        const int n_sb = hidden >> 8;
        const size_t sb_bytes = sk_sb_bytes(sk_code);
        const uint8_t* sb_ptr = emb_sk + (size_t(id) * n_sb + sb) * sb_bytes;
        float eff_sc, half_bias;
        int q0, q1;
        unpack_sk_lane8(sb_ptr, sk_code, lane, eff_sc, half_bias, q0, q1);
        const int8_t* p0 = reinterpret_cast<const int8_t*>(&q0);
        const int8_t* p1 = reinterpret_cast<const int8_t*>(&q1);
        for (int j = 0; j < 4; ++j) {
            const float raw0 = (half_bias != 0.0f) ? float(uint8_t(p0[j])) : float(p0[j]);
            const float raw1 = (half_bias != 0.0f) ? float(uint8_t(p1[j])) : float(p1[j]);
            vals[j] = (raw0 - half_bias) * eff_sc;
            vals[4 + j] = (raw1 - half_bias) * eff_sc;
        }
    }

    for (int h = 0; h < hc; ++h) {
        float* dst = x_hc + (size_t(tok) * hc + h) * hidden + base_col;
        for (int j = 0; j < 8; ++j) dst[j] = vals[j];
    }
}

// Applies `Hy4` `hc_pre` (`cur = sum_h x[h] * pre[h]`, `post_out[h] = hc_mag * sig(...) + hc_eps`) or `hc_head` (`post_out == nullptr`):
// Grid: (ceil(hidden / 256), n_tok, 1), Block: (256, 1, 1)
__global__ void hy4_hc_apply_pre_f32(
    const float* __restrict__ x_hc,
    const float* __restrict__ mix, // [n_tok, mix_stride] (`8` for layer `hc_pre`, `4` for `hc_head`)
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ cur_out,
    float* __restrict__ post_out,  // optional [n_tok, hc]
    int hidden,
    int hc,
    int mix_stride,
    float hc_eps,
    float hc_mag
) {
    const int d = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int t = int(blockIdx.y);
    const float* m = mix + size_t(t) * mix_stride;
    if (post_out != nullptr && blockIdx.x == 0 && int(threadIdx.x) < hc) {
        const int h = int(threadIdx.x);
        post_out[size_t(t) * hc + h] =
            hc_mag * (1.0f / (1.0f + expf(-(m[hc + h] * hc_scale[1] + hc_base[hc + h])))) + hc_eps;
    }
    if (d >= hidden) return;
    float acc = 0.0f;
    for (int h = 0; h < hc; ++h) {
        const float pre_h = (1.0f / (1.0f + expf(-(m[h] * hc_scale[0] + hc_base[h])))) + hc_eps;
        acc = fmaf(x_hc[(size_t(t) * hc + h) * hidden + d], pre_h, acc);
    }
    cur_out[size_t(t) * hidden + d] = acc;
}

// `x_hc[t, h, d] += sub[t, d] * post[t, h]`: Grid (ceil((hc * hidden) / 256), n_tok, 1), Block (256, 1, 1)
__global__ void hy4_hc_post_f32(
    float* __restrict__ x_hc,
    const float* __restrict__ sub,
    const float* __restrict__ post,
    int hidden,
    int hc
) {
    const int hd = int(blockIdx.x) * 256 + int(threadIdx.x);
    const int t = int(blockIdx.y);
    if (hd >= hc * hidden) return;
    const int h = hd / hidden;
    const int d = hd % hidden;
    x_hc[size_t(t) * hc * hidden + hd] += sub[size_t(t) * hidden + d] * post[size_t(t) * hc + h];
}

// `Hy4` Engram signed-sqrt Sigmoid gate + stream residual update:
// `kv_proj` layout: `[n_tok, 5, hidden]` (`0..3` = keys, `4` = value)!
// Grid: (hc, n_tok, 1), Block: (256, 1, 1)
__global__ void hy4_engram_gate_add_f32(
    float* __restrict__ x_hc,
    const float* __restrict__ kv_proj,       // [n_tok, 5, hidden]
    const float* __restrict__ norm_key,      // [hc, hidden]
    const float* __restrict__ norm_query,    // [hc, hidden]
    const float* __restrict__ hc_scales,     // [hc, hidden]
    const float* __restrict__ pre_gamma,     // [hidden]
    const float* __restrict__ any_valid,     // [n_tok]
    int hidden,
    int hc
) {
    const int j = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const float any = any_valid[t];
    if (any <= 0.0f) return;

    __shared__ float warp_sums[8];
    const float eps = 1.1920929e-7f;
    float* xs = x_hc + (size_t(t) * hc + j) * hidden;
    const float* key = kv_proj + (size_t(t) * (hc + 1) + j) * hidden;
    const float* val = kv_proj + (size_t(t) * (hc + 1) + hc) * hidden;
    const float* nk = norm_key + size_t(j) * hidden;
    const float* nq = norm_query + size_t(j) * hidden;
    const float* hs = hc_scales + size_t(j) * hidden;

    float sv = 0.0f, sx = 0.0f, sk = 0.0f;
    for (int d = tid; d < hidden; d += 256) {
        sv = fmaf(val[d], val[d], sv);
        sx = fmaf(xs[d], xs[d], sx);
        sk = fmaf(key[d], key[d], sk);
    }
    const float vinv = rsqrtf(block256_reduce_sum(sv, tid, warp_sums) / float(hidden) + eps);
    const float xinv = rsqrtf(block256_reduce_sum(sx, tid, warp_sums) / float(hidden) + eps);
    const float kinv = rsqrtf(block256_reduce_sum(sk, tid, warp_sums) / float(hidden) + eps);

    float dot_v = 0.0f;
    for (int d = tid; d < hidden; d += 256) {
        dot_v = fmaf((xs[d] * xinv) * (key[d] * kinv), nk[d] * nq[d], dot_v);
    }
    const float dotf = block256_reduce_sum(dot_v, tid, warp_sums) * rsqrtf(float(hidden));
    const float mag = sqrtf(fmaxf(fabsf(dotf), 1.0e-6f));
    const float gate = 1.0f / (1.0f + expf(-(dotf < 0.0f ? -mag : mag)));
    for (int d = tid; d < hidden; d += 256) {
        xs[d] += (val[d] * vinv * pre_gamma[d]) * (gate + hs[d]) * any;
    }
}

// Applies interleaved `(2i, 2i+1)` RoPE to `q[n_tok, n_head, qk_head]` (slice `qk_nope..qk_head`)
// AND `kva[n_tok, kv_lora + qk_rope]` (`k_pe` at `kv_lora..kv_lora+qk_rope`), copying `kva[0..kv_lora]` into `kv_cmpr_in`!
// Grid: (n_head + 1, n_tok, 1), Block: (128, 1, 1)
__global__ void hy4_mla_prep_rope_f32(
    float* __restrict__ q,
    const float* __restrict__ kva,
    float* __restrict__ kv_cmpr_in,
    float* __restrict__ k_pe_out,
    const float* __restrict__ inv_freqs,
    int n_head,
    int qk_nope,
    int qk_rope,
    int kv_lora,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + t;
    const int qk_head = qk_nope + qk_rope;
    const int half_rope = qk_rope >> 1;

    if (h < n_head) {
        float* qp = q + (size_t(t) * n_head + h) * qk_head + qk_nope;
        if (tid < half_rope) {
            float s, c;
            sincosf(float(pos) * inv_freqs[tid], &s, &c);
            const float x0 = qp[2 * tid], x1 = qp[2 * tid + 1];
            qp[2 * tid] = x0 * c - x1 * s;
            qp[2 * tid + 1] = x0 * s + x1 * c;
        }
    } else {
        const float* kva_t = kva + size_t(t) * (kv_lora + qk_rope);
        float* cmpr_t = kv_cmpr_in + size_t(t) * kv_lora;
        for (int i = tid; i < kv_lora; i += 128) cmpr_t[i] = kva_t[i];
        if (tid < half_rope) {
            const float* src_pe = kva_t + kv_lora;
            float* dst_pe = k_pe_out + size_t(t) * qk_rope;
            float s, c;
            sincosf(float(pos) * inv_freqs[tid], &s, &c);
            const float x0 = src_pe[2 * tid], x1 = src_pe[2 * tid + 1];
            dst_pe[2 * tid] = x0 * c - x1 * s;
            dst_pe[2 * tid + 1] = x0 * s + x1 * c;
        }
    }
}

// Packs expanded `kvx[n_tok, n_head, qk_nope + v_head]` + `k_pe[n_tok, qk_rope]` into F16 `k_cache` & `v_cache`
// (`ring_cap > 0` wraps SWA layers in a fixed `window`-token ring buffer!):
// Grid: (n_head, n_tok, 1), Block: (256, 1, 1)
__global__ void hy4_pack_mla_kv_f16(
    const float* __restrict__ kvx,
    const float* __restrict__ k_pe,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int n_head,
    int qk_nope,
    int qk_rope,
    int v_head,
    int ring_cap,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + t;
    const int slot = (ring_cap > 0) ? (pos % ring_cap) : pos;
    const int qk_head = qk_nope + qk_rope;
    const int ex = qk_nope + v_head;
    const float* src_ex = kvx + (size_t(t) * n_head + h) * ex;
    const float* src_pe = k_pe + size_t(t) * qk_rope;
    half* kc = k_cache + (size_t(slot) * n_head + h) * qk_head;
    half* vc = v_cache + (size_t(slot) * n_head + h) * v_head;

    if (tid < qk_nope) kc[tid] = __float2half_rn(src_ex[tid]);
    else if (tid < qk_head) kc[tid] = __float2half_rn(src_pe[tid - qk_nope]);
    if (tid < v_head) vc[tid] = __float2half_rn(src_ex[qk_nope + tid]);
}

// FlashDecoding MHA for `Hy4` with `learnable_sink_param[n_head]` (`denom = exp(sink[h] - m_all) + sum`)
// and separate `qk_head <= 320`, `v_head <= 256` + optional SWA `ring_cap`: Grid (n_head, n_tok, 1), Block (32, 8, 1)
__global__ void __launch_bounds__(256, 2) hy4_attn_f16_kv_with_sink(
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const float* __restrict__ sinks,
    float* __restrict__ out,
    int n_head,
    int qk_head, // <= 320 (e.g. 256 or 320)
    int v_head,  // <= 256 (e.g. 256)
    int n_swa,
    int ring_cap,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int wid = int(threadIdx.y);
    const int tid = wid * 32 + lane;
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int kv_start = (n_swa > 0 && (pos + 1) > n_swa) ? (pos + 1 - n_swa) : 0;
    const float scale = rsqrtf(float(qk_head));
    const size_t q_idx = size_t(tok) * n_head + h;

    __shared__ float q_sh[320];
    __shared__ float wg_m[8];
    __shared__ float wg_l[8];
    __shared__ float wg_o[8][256];

    const float* q_h = q + q_idx * qk_head;
    for (int d = tid; d < qk_head; d += 256) q_sh[d] = q_h[d] * scale;
    __syncthreads();

    float m_loc = -CUDART_INF_F;
    float l_loc = 0.0f;
    float o_reg[8] = {0.0f};

    for (int t = kv_start + wid; t <= pos; t += 8) {
        const int slot = (ring_cap > 0) ? (t % ring_cap) : t;
        const half* kh = k_cache + (size_t(slot) * n_head + h) * qk_head;
        const half* vh = v_cache + (size_t(slot) * n_head + h) * v_head;
        float dot_v = 0.0f;
        for (int d = lane; d < qk_head; d += 32) {
            dot_v = fmaf(q_sh[d], __half2float(kh[d]), dot_v);
        }
        dot_v = warp_reduce_sum(dot_v);
        const float m_new = fmaxf(m_loc, dot_v);
        const float alpha = (l_loc == 0.0f) ? 0.0f : expf(m_loc - m_new);
        const float p = expf(dot_v - m_new);
        l_loc = l_loc * alpha + p;
        m_loc = m_new;
        for (int d = lane; d < v_head; d += 32) {
            const int c = d >> 5;
            o_reg[c] = o_reg[c] * alpha + p * __half2float(vh[d]);
        }
    }

    if (lane == 0) {
        wg_m[wid] = m_loc;
        wg_l[wid] = l_loc;
    }
    for (int d = lane; d < v_head; d += 32) wg_o[wid][d] = o_reg[d >> 5];
    __syncthreads();

    float m_all = sinks[h];
    for (int w = 0; w < 8; ++w) {
        if (wg_l[w] > 0.0f) m_all = fmaxf(m_all, wg_m[w]);
    }
    float* out_h = out + q_idx * v_head;
    for (int d = tid; d < v_head; d += 256) {
        float denom = expf(sinks[h] - m_all);
        float acc = 0.0f;
        for (int w = 0; w < 8; ++w) {
            if (wg_l[w] > 0.0f) {
                const float wt = expf(wg_m[w] - m_all);
                denom += wg_l[w] * wt;
                acc += wg_o[w][d] * wt;
            }
        }
        out_h[d] = acc / denom;
    }
}

// Parallel multi-slot `__dp4a` `LatentMoE` `S_K` (or `F32` when `sk_code == 101`) expert projection:
// Grid: (ceil(rows / 4), slot_count, n_tok), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) hy4_latmoe_indexed_sk_q8_1_g64(
    const uint8_t* __restrict__ sec4_arena,
    const StrataExpertGpuEntry* __restrict__ layer_dir,
    const int32_t* __restrict__ rank_of_layer,
    const BlockQ8_1_G64* __restrict__ xq,
    const float* __restrict__ x_f32,
    const uint32_t* __restrict__ top_ids,
    const float* __restrict__ top_weights,
    float* __restrict__ out,
    int is_down, // 0 = gate_up (offset + 2*cols, shared x per tok), 1 = down (offset + dn_offset, per-slot x, weighted out)
    int rows,
    int cols,
    int slot_count
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int slot = int(blockIdx.y);
    const int tok = int(blockIdx.z);
    const int lane = int(threadIdx.x);
    if (row >= rows) return;

    const size_t ts_idx = size_t(tok) * slot_count + slot;
    const uint32_t exp_id = top_ids[ts_idx];
    const int rank = rank_of_layer[exp_id];
    const StrataExpertGpuEntry ent = layer_dir[rank];
    const int sk_code = is_down ? int(ent.dn_code_id) : int(ent.gu_code_id);
    const size_t sub_off = is_down ? size_t(ent.dn_offset) : (size_t(cols) * 2ull);

    float acc = 0.0f;
    if (sk_code == 101) {
        const float* w_row = reinterpret_cast<const float*>(sec4_arena + ent.body_offset + sub_off) + size_t(row) * cols;
        const float* xr = x_f32 + (is_down ? ts_idx : size_t(tok)) * cols;
        for (int c = lane; c < cols; c += 32) {
            acc = fmaf(w_row[c], xr[c], acc);
        }
    } else {
        const size_t sb_bytes = sk_sb_bytes(sk_code);
        const int n_sb = cols >> 8;
        const int n_b64 = cols >> 6;
        const uint8_t* w_row = sec4_arena + ent.body_offset + sub_off + size_t(row) * n_sb * sb_bytes;
        const BlockQ8_1_G64* xr = xq + (is_down ? ts_idx : size_t(tok)) * n_b64;

        const int g = lane >> 3;
        const int sub = lane & 7;
        for (int sb = 0; sb < n_sb; ++sb) {
            float eff_sc, half_bias;
            int q0, q1;
            unpack_sk_lane8(w_row + size_t(sb) * sb_bytes, sk_code, lane, eff_sc, half_bias, q0, q1);
            const BlockQ8_1_G64& xb = xr[sb * 4 + g];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + 2 * sub;
            int dot = dp4a_i8(q0, xqs[0], 0);
            dot = dp4a_i8(q1, xqs[1], dot);
            float term = eff_sc * xb.d * float(dot);
            if (sub == 0 && half_bias != 0.0f) term -= eff_sc * half_bias * xb.s;
            acc += term;
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const float w_slot = is_down ? top_weights[ts_idx] : 1.0f;
        out[ts_idx * rows + row] = w_slot * acc;
    }
}

// Clipped SwiGLU + SmoothQuant `sq16` divide + `BlockQ8_1_G64` quantization (plus optional `out_f32` for F32 parity) across `n_tok * slot_count` rows:
// `h[d] = (min(silu(gu[d]), limit) * clamp(gu[L + d], -limit, limit)) / __half2float(sq16[d])`
// Grid: (ceil((L / 64) / 2), n_tok * slot_count, 1), Block: (32, 2, 1)
__global__ void hy4_latmoe_swiglu_sq_quantize_g64(
    const float* __restrict__ gu_all, // [n_rows, 2 * L]
    const uint8_t* __restrict__ sec4_arena,
    const StrataExpertGpuEntry* __restrict__ layer_dir,
    const int32_t* __restrict__ rank_of_layer,
    const uint32_t* __restrict__ top_ids,
    BlockQ8_1_G64* __restrict__ out_xq,
    float* __restrict__ out_f32,
    int lat_dim,
    int n_rows,
    float limit
) {
    const int n_b64 = lat_dim >> 6;
    const int b64 = int(blockIdx.x) * 2 + int(threadIdx.y);
    const int row = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    if (b64 >= n_b64 || row >= n_rows) return;

    const uint32_t exp_id = top_ids[row];
    const StrataExpertGpuEntry ent = layer_dir[rank_of_layer[exp_id]];
    const half* sq16 = reinterpret_cast<const half*>(sec4_arena + ent.body_offset);
    const float* gu = gu_all + size_t(row) * (2 * lat_dim);

    float vals[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int d = b64 * 64 + 2 * lane + m;
        const float g = gu[d];
        const float u = gu[lat_dim + d];
        const float s = fminf(g / (1.0f + expf(-g)), limit);
        const float uc = fminf(fmaxf(u, -limit), limit);
        vals[m] = (s * uc) / __half2float(sq16[d]);
        if (out_f32 != nullptr) {
            out_f32[size_t(row) * lat_dim + d] = vals[m];
        }
    }

    const float amax = warp_reduce_max(fmaxf(fabsf(vals[0]), fabsf(vals[1])));
    const float d_sc = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int q0 = max(-127, min(127, __float2int_rn(vals[0] * id)));
    const int q1 = max(-127, min(127, __float2int_rn(vals[1] * id)));
    const int q_sum = int(warp_reduce_sum(float(q0 + q1)));

    BlockQ8_1_G64& dst = out_xq[size_t(row) * n_b64 + b64];
    dst.qs[2 * lane] = int8_t(q0);
    dst.qs[2 * lane + 1] = int8_t(q1);
    if (lane == 0) {
        dst.d = d_sc;
        dst.s = d_sc * float(q_sum);
    }
}

// `P0-3`: Fused dual `Gate + Up + clamped SwiGLU + SmoothQuant sq16` for `Hy4` `LatentMoE`!
// Reads `xq_g64` ONCE per superblock, dots `gate` (`row`) and `up` (`lat_dim + row`) simultaneously,
// applies clamped SwiGLU + `/ sq16[row]`, and writes `out_act_f32[ts_idx, row]` directly!
// Grid: (ceil(lat_dim / 4), slot_count, n_tok), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) hy4_latmoe_dual_gu_swiglu_sq_f32(
    const uint8_t* __restrict__ sec4_arena,
    const StrataExpertGpuEntry* __restrict__ layer_dir,
    const int32_t* __restrict__ rank_of_layer,
    const BlockQ8_1_G64* __restrict__ xq,
    const float* __restrict__ x_f32,
    const uint32_t* __restrict__ top_ids,
    float* __restrict__ out_act,
    int lat_dim,
    int cols,
    int slot_count,
    float limit
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int slot = int(blockIdx.y);
    const int tok = int(blockIdx.z);
    const int lane = int(threadIdx.x);
    if (row >= lat_dim) return;

    const size_t ts_idx = size_t(tok) * slot_count + slot;
    const uint32_t exp_id = top_ids[ts_idx];
    const StrataExpertGpuEntry ent = layer_dir[rank_of_layer[exp_id]];
    const int sk_code = int(ent.gu_code_id);
    const size_t sub_off = size_t(cols) * 2ull;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    if (sk_code == 101) {
        const float* wg_row = reinterpret_cast<const float*>(sec4_arena + ent.body_offset + sub_off) + size_t(row) * cols;
        const float* wu_row = wg_row + size_t(lat_dim) * cols;
        const float* xr = x_f32 + size_t(tok) * cols;
        for (int c = lane; c < cols; c += 32) {
            const float xv = xr[c];
            acc_g = fmaf(wg_row[c], xv, acc_g);
            acc_u = fmaf(wu_row[c], xv, acc_u);
        }
    } else {
        const size_t sb_bytes = sk_sb_bytes(sk_code);
        const int n_sb = cols >> 8;
        const int n_b64 = cols >> 6;
        const size_t row_bytes = size_t(n_sb) * sb_bytes;
        const uint8_t* wg_row = sec4_arena + ent.body_offset + sub_off + size_t(row) * row_bytes;
        const uint8_t* wu_row = wg_row + size_t(lat_dim) * row_bytes;
        const BlockQ8_1_G64* xr = xq + size_t(tok) * n_b64;

        const int g = lane >> 3;
        const int sub = lane & 7;
        for (int sb = 0; sb < n_sb; ++sb) {
            const BlockQ8_1_G64& xb = xr[sb * 4 + g];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + 2 * sub;
            const int x0 = xqs[0], x1 = xqs[1];

            float sc_g, hb_g;
            int g0, g1;
            unpack_sk_lane8(wg_row + size_t(sb) * sb_bytes, sk_code, lane, sc_g, hb_g, g0, g1);
            int dg = dp4a_i8(g0, x0, 0);
            dg = dp4a_i8(g1, x1, dg);
            float tg = sc_g * xb.d * float(dg);
            if (sub == 0 && hb_g != 0.0f) tg -= sc_g * hb_g * xb.s;
            acc_g += tg;

            float sc_u, hb_u;
            int u0, u1;
            unpack_sk_lane8(wu_row + size_t(sb) * sb_bytes, sk_code, lane, sc_u, hb_u, u0, u1);
            int du = dp4a_i8(u0, x0, 0);
            du = dp4a_i8(u1, x1, du);
            float tu = sc_u * xb.d * float(du);
            if (sub == 0 && hb_u != 0.0f) tu -= sc_u * hb_u * xb.s;
            acc_u += tu;
        }
    }

    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if (lane == 0) {
        const half* sq16 = reinterpret_cast<const half*>(sec4_arena + ent.body_offset);
        const float s = fminf(acc_g / (1.0f + expf(-acc_g)), limit);
        const float uc = fminf(fmaxf(acc_u, -limit), limit);
        out_act[ts_idx * lat_dim + row] = (s * uc) / __half2float(sq16[row]);
    }
}

// ============================================================================
// 13. `P1-1`: Matrix-Absorbed MLA (`latent_kv_cache` `[cap, kv_lora + qk_rope]` FP16 — `28x` smaller KV!)
// ============================================================================

// Applies RoPE to `q[..., qk_nope..qk_head]` AND RMSNorm + RoPE to `kva[0..kv_lora | kv_lora..kv_lora+qk_rope]`,
// writing compressed `[kv_cmpr(kv_lora) | k_pe(qk_rope)]` directly into FP16 `latent_kv_cache[slot]`!
// Grid: (n_head + 1, n_tok, 1), Block: (256, 1, 1)
__global__ void hy4_mla_prep_latent_kv_f16(
    float* __restrict__ q,
    const float* __restrict__ kva,
    const float* __restrict__ kv_ln_w,
    const float* __restrict__ inv_freqs,
    half* __restrict__ latent_kv_cache,
    int n_head,
    int qk_nope,
    int qk_rope,
    int kv_lora,
    float rms_eps,
    int ring_cap,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int t = int(blockIdx.y);
    const int tid = int(threadIdx.x);
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + t;
    const int qk_head = qk_nope + qk_rope;
    const int kv_lat = kv_lora + qk_rope;
    const int half_rope = qk_rope >> 1;

    if (h < n_head) {
        if (tid < half_rope) {
            float* qp = q + (size_t(t) * n_head + h) * qk_head + qk_nope;
            float s, c;
            sincosf(float(pos) * inv_freqs[tid], &s, &c);
            const float x0 = qp[2 * tid], x1 = qp[2 * tid + 1];
            qp[2 * tid] = x0 * c - x1 * s;
            qp[2 * tid + 1] = x0 * s + x1 * c;
        }
        return;
    }

    __shared__ float warp_sums[8];
    const int slot = (ring_cap > 0) ? (pos % ring_cap) : pos;
    const float* kva_t = kva + size_t(t) * kv_lat;
    half* dst_lat = latent_kv_cache + size_t(slot) * kv_lat;

    float ss = 0.0f;
    for (int i = tid; i < kv_lora; i += 256) {
        const float v = kva_t[i];
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block256_reduce_sum(ss, tid, warp_sums) / float(kv_lora) + rms_eps);
    for (int i = tid; i < kv_lora; i += 256) {
        dst_lat[i] = __float2half_rn(kva_t[i] * inv * kv_ln_w[i]);
    }
    if (tid < half_rope) {
        const float* src_pe = kva_t + kv_lora;
        half* dst_pe = dst_lat + kv_lora;
        float s, c;
        sincosf(float(pos) * inv_freqs[tid], &s, &c);
        const float x0 = src_pe[2 * tid], x1 = src_pe[2 * tid + 1];
        dst_pe[2 * tid] = __float2half_rn(x0 * c - x1 * s);
        dst_pe[2 * tid + 1] = __float2half_rn(x0 * s + x1 * c);
    }
}

// Absorbs `W_kb` (`rows h*(qk_nope+v_head) .. +qk_nope` of `kv_b_proj`) into `q_nope` across up to 4 tokens per tile:
// `qa[t, h, 0..kv_lora] = sum_{i=0..qk_nope-1} q_nope[t, h, i] * W_kb[h, i, 0..kv_lora]`
// `qa[t, h, kv_lora..kv_lat] = q_pe[t, h, 0..qk_rope]`
// Supports `S_K` (`0..=8`), `FMT_Q8ROW` (`100`), and `FMT_F32` (`101`)!
// Grid: (ceil(kv_lora / 256), n_head, ceil(n_tok / 4)), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) hy4_mla_absorb_q(
    const float* __restrict__ q,
    const uint8_t* __restrict__ w_kv_b,
    float* __restrict__ qa_out,
    int sk_code,
    int n_head,
    int qk_nope,
    int qk_rope,
    int v_head,
    int kv_lora,
    int n_tok
) {
    const int sb = int(blockIdx.x);
    const int h = int(blockIdx.y);
    const int tok_base = int(blockIdx.z) * 4;
    const int lane = int(threadIdx.x);
    const int warp_id = int(threadIdx.y);

    const int qk_head = qk_nope + qk_rope;
    const int kv_lat = kv_lora + qk_rope;
    const int ex = qk_nope + v_head;
    const int total_w_rows = n_head * ex;
    const int base_col = sb * 256 + lane * 8;
    const int valid_t = min(4, n_tok - tok_base);

    float acc[4][8];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;
    }

    const int n_sb = kv_lora >> 8;
    const size_t sb_bytes = sk_sb_bytes(sk_code);

    for (int i = warp_id; i < qk_nope; i += 4) {
        const int w_row_idx = h * ex + i;
        float w8[8];
        if (sk_code == 101) {
            const float* wr = reinterpret_cast<const float*>(w_kv_b) + size_t(w_row_idx) * kv_lora + base_col;
#pragma unroll
            for (int j = 0; j < 8; ++j) w8[j] = (base_col + j < kv_lora) ? wr[j] : 0.0f;
        } else if (sk_code == 100) {
            const float sc = reinterpret_cast<const float*>(w_kv_b)[w_row_idx];
            const int8_t* wr = reinterpret_cast<const int8_t*>(w_kv_b + size_t(total_w_rows) * 4)
                             + size_t(w_row_idx) * kv_lora + base_col;
#pragma unroll
            for (int j = 0; j < 8; ++j) w8[j] = (base_col + j < kv_lora) ? (float(wr[j]) * sc) : 0.0f;
        } else {
            const uint8_t* sb_ptr = w_kv_b + (size_t(w_row_idx) * n_sb + sb) * sb_bytes;
            float eff_sc, half_bias;
            int q0, q1;
            unpack_sk_lane8(sb_ptr, sk_code, lane, eff_sc, half_bias, q0, q1);
            const int8_t* p0 = reinterpret_cast<const int8_t*>(&q0);
            const int8_t* p1 = reinterpret_cast<const int8_t*>(&q1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float r0 = (half_bias != 0.0f) ? float(uint8_t(p0[j])) : float(p0[j]);
                const float r1 = (half_bias != 0.0f) ? float(uint8_t(p1[j])) : float(p1[j]);
                w8[j] = (r0 - half_bias) * eff_sc;
                w8[4 + j] = (r1 - half_bias) * eff_sc;
            }
        }

        for (int t = 0; t < valid_t; ++t) {
            const float qv = q[(size_t(tok_base + t) * n_head + h) * qk_head + i];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[t][j] = fmaf(qv, w8[j], acc[t][j]);
            }
        }
    }

    __shared__ float partial[3][256];
    for (int t = 0; t < valid_t; ++t) {
        if (warp_id > 0) {
#pragma unroll
            for (int j = 0; j < 8; ++j) partial[warp_id - 1][lane * 8 + j] = acc[t][j];
        }
        __syncthreads();
        if (warp_id == 0) {
            float* dst = qa_out + (size_t(tok_base + t) * n_head + h) * kv_lat;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = base_col + j;
                if (col < kv_lora) {
                    dst[col] = acc[t][j]
                             + partial[0][lane * 8 + j]
                             + partial[1][lane * 8 + j]
                             + partial[2][lane * 8 + j];
                }
            }
            if (sb == 0) {
                const float* q_pe = q + (size_t(tok_base + t) * n_head + h) * qk_head + qk_nope;
                for (int r = lane; r < qk_rope; r += 32) {
                    dst[kv_lora + r] = q_pe[r];
                }
            }
        }
        __syncthreads();
    }
}

// FlashDecoding MQA over shared `latent_kv_cache` (`[cap, kv_lora + qk_rope]` FP16) with `learnable_sink_param[n_head]`!
// `kv_lora <= 512`, `qk_rope <= 128` (`kv_lat <= 640`):
// Grid: (n_head, n_tok, 1), Block: (32, 8, 1)
__global__ void __launch_bounds__(256, 2) hy4_mla_latent_attn_f16_with_sink(
    const float* __restrict__ qa,
    const half* __restrict__ latent_kv_cache,
    const float* __restrict__ sinks,
    float* __restrict__ ctx_out,
    int n_head,
    int kv_lora,
    int qk_rope,
    int qk_head,
    int n_swa,
    int ring_cap,
    int start_pos,
    const int* __restrict__ pos_dev
) {
    const int h = int(blockIdx.x);
    const int tok = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int wid = int(threadIdx.y);
    const int tid = wid * 32 + lane;
    const int pos = (pos_dev ? (*pos_dev) : start_pos) + tok;
    const int kv_start = (n_swa > 0 && (pos + 1) > n_swa) ? (pos + 1 - n_swa) : 0;
    const int kv_lat = kv_lora + qk_rope;
    const float scale = rsqrtf(float(qk_head));
    const size_t q_idx = size_t(tok) * n_head + h;

    __shared__ float q_sh[640];
    __shared__ float wg_m[8];
    __shared__ float wg_l[8];
    __shared__ float wg_o[8][512];

    const float* qa_h = qa + q_idx * kv_lat;
    for (int d = tid; d < kv_lat; d += 256) q_sh[d] = qa_h[d] * scale;
    __syncthreads();

    float m_loc = -CUDART_INF_F;
    float l_loc = 0.0f;
    float o_reg[16] = {0.0f};

    for (int s = kv_start + wid; s <= pos; s += 8) {
        const int slot = (ring_cap > 0) ? (s % ring_cap) : s;
        const half* lat_s = latent_kv_cache + size_t(slot) * kv_lat;
        float dot_v = 0.0f;
        for (int d = lane; d < kv_lat; d += 32) {
            dot_v = fmaf(q_sh[d], __half2float(lat_s[d]), dot_v);
        }
        dot_v = warp_reduce_sum(dot_v);
        const float m_new = fmaxf(m_loc, dot_v);
        const float alpha = (l_loc == 0.0f) ? 0.0f : expf(m_loc - m_new);
        const float p = expf(dot_v - m_new);
        l_loc = l_loc * alpha + p;
        m_loc = m_new;
        for (int d = lane; d < kv_lora; d += 32) {
            const int c = d >> 5;
            o_reg[c] = o_reg[c] * alpha + p * __half2float(lat_s[d]);
        }
    }

    if (lane == 0) {
        wg_m[wid] = m_loc;
        wg_l[wid] = l_loc;
    }
    for (int d = lane; d < kv_lora; d += 32) wg_o[wid][d] = o_reg[d >> 5];
    __syncthreads();

    float m_all = sinks[h];
    for (int w = 0; w < 8; ++w) {
        if (wg_l[w] > 0.0f) m_all = fmaxf(m_all, wg_m[w]);
    }
    float* out_h = ctx_out + q_idx * kv_lora;
    for (int d = tid; d < kv_lora; d += 256) {
        float denom = expf(sinks[h] - m_all);
        float acc = 0.0f;
        for (int w = 0; w < 8; ++w) {
            if (wg_l[w] > 0.0f) {
                const float wt = expf(wg_m[w] - m_all);
                denom += wg_l[w] * wt;
                acc += wg_o[w][d] * wt;
            }
        }
        out_h[d] = acc / denom;
    }
}

// Projects latent context `ctx[n_tok, n_head, kv_lora]` through `W_vb[h]` (`rows h*(qk_nope+v_head)+qk_nope .. +v_head` of `kv_b_proj`)
// AND fuses `sigmoid(gate)` multiply in one pass!
// Supports `S_K` (`0..=8` via `BlockQ8_1_G64`), `FMT_Q8ROW` (`100` via `BlockQ8_1`), and `FMT_F32` (`101` via `ctx_f32`)!
// Grid: (ceil(v_head / 4), n_head, n_tok), Block: (32, 4, 1)
__global__ void __launch_bounds__(128, 4) hy4_mla_vup_gate(
    const uint8_t* __restrict__ w_kv_b,
    const BlockQ8_1_G64* __restrict__ ctx_g64,
    const BlockQ8_1* __restrict__ ctx_q8,
    const float* __restrict__ ctx_f32,
    const float* __restrict__ gate,
    float* __restrict__ attn_out,
    int sk_code,
    int n_head,
    int qk_nope,
    int v_head,
    int kv_lora
) {
    const int j = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int h = int(blockIdx.y);
    const int tok = int(blockIdx.z);
    const int lane = int(threadIdx.x);
    if (j >= v_head) return;

    const int ex = qk_nope + v_head;
    const int total_w_rows = n_head * ex;
    const int w_row_idx = h * ex + qk_nope + j;
    const size_t th_idx = size_t(tok) * n_head + h;

    float acc = 0.0f;
    if (sk_code == 101) {
        const float* wr = reinterpret_cast<const float*>(w_kv_b) + size_t(w_row_idx) * kv_lora;
        const float* xr = ctx_f32 + th_idx * kv_lora;
        for (int c = lane; c < kv_lora; c += 32) {
            acc = fmaf(wr[c], xr[c], acc);
        }
    } else if (sk_code == 100) {
        const float row_sc = reinterpret_cast<const float*>(w_kv_b)[w_row_idx];
        const int8_t* q_row = reinterpret_cast<const int8_t*>(w_kv_b + size_t(total_w_rows) * 4) + size_t(w_row_idx) * kv_lora;
        const int n_q8 = kv_lora >> 5;
        const BlockQ8_1* xr = ctx_q8 + th_idx * n_q8;
        const int iqs = 2 * (lane & 3);
        for (int kbx = lane >> 2; kbx < n_q8; kbx += 8) {
            const int w0 = reinterpret_cast<const int*>(q_row + kbx * 32)[iqs];
            const int w1 = reinterpret_cast<const int*>(q_row + kbx * 32)[iqs + 1];
            const BlockQ8_1& xb = xr[kbx];
            const int* xqs = reinterpret_cast<const int*>(xb.qs);
            int sumi = dp4a_i8(w0, xqs[iqs], 0);
            sumi = dp4a_i8(w1, xqs[iqs + 1], sumi);
            acc += __low2float(xb.ds) * float(sumi);
        }
        acc *= row_sc;
    } else {
        const int n_sb = kv_lora >> 8;
        const int n_b64 = kv_lora >> 6;
        const size_t sb_bytes = sk_sb_bytes(sk_code);
        const uint8_t* w_row = w_kv_b + size_t(w_row_idx) * n_sb * sb_bytes;
        const BlockQ8_1_G64* xr = ctx_g64 + th_idx * n_b64;
        const int g = lane >> 3;
        const int sub = lane & 7;
        for (int sb = 0; sb < n_sb; ++sb) {
            float eff_sc, half_bias;
            int q0, q1;
            unpack_sk_lane8(w_row + size_t(sb) * sb_bytes, sk_code, lane, eff_sc, half_bias, q0, q1);
            const BlockQ8_1_G64& xb = xr[sb * 4 + g];
            const int* xqs = reinterpret_cast<const int*>(xb.qs) + 2 * sub;
            int dot = dp4a_i8(q0, xqs[0], 0);
            dot = dp4a_i8(q1, xqs[1], dot);
            float term = eff_sc * xb.d * float(dot);
            if (sub == 0 && half_bias != 0.0f) term -= eff_sc * half_bias * xb.s;
            acc += term;
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const size_t out_idx = th_idx * v_head + j;
        const float g_val = gate[out_idx];
        attn_out[out_idx] = acc * (1.0f / (1.0f + expf(-g_val)));
    }
}

} // extern "C"
