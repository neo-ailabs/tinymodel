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
#define DTYPE_F32   0
#define DTYPE_F16   1
#define DTYPE_Q4_0  2
#define DTYPE_Q4_1  3
#define DTYPE_Q8_0  8
#define DTYPE_Q4_K  12
#define DTYPE_Q5_K  13
#define DTYPE_Q6_K  14
#define DTYPE_BF16  30
#define DTYPE_NVFP4 40

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
        case DTYPE_Q8_0: {
            const int b = col >> 5;
            const int i = col & 31;
            const BlockQ8_0& blk = reinterpret_cast<const BlockQ8_0*>(w_row)[b];
            return __half2float(blk.d) * float(blk.qs[i]);
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

// Fast cooperative warp-per-row K-quant MMVQ (`Q4_K`, `Q5_K`, `Q6_K`) using `__dp4a`!
// Each superblock has 256 elements = 8 sub-blocks of 32 (`BlockQ8_1`).
// Across 32 lanes: `sub = lane >> 2` (0..7), `quad = lane & 3` (0..3, 4 bytes = 4 elements per quad!).
// Wait: 8 sub-blocks * 8 quads/sub-block = 64 quads per 256-elem superblock -> each lane handles 2 quads!

__global__ void mmvq_q4_k_q8_1(
    const BlockQ4K* __restrict__ w,
    const BlockQ8_1* __restrict__ xq,
    float* __restrict__ out,
    int rows,
    int cols,
    int n_tok
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int lane = int(threadIdx.x);
    if (row >= rows) return;

    const int n_sb = cols >> 8;
    const int n_q8 = cols >> 5;
    const BlockQ4K* w_row = w + size_t(row) * n_sb;
    float acc[MMVQ_MAX_TOK] = {0.0f};

    // Lane mapping inside 256-elem superblock:
    // chunk64 = lane >> 3 (0..3), in_chunk_quad = lane & 7 (0..7 -> 32 bytes of `qs`!)
    const int chunk64 = lane >> 3;
    const int q_idx = lane & 7;

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
__global__ void mmvq_float_f32(
    const uint8_t* __restrict__ w_bytes,
    const float* __restrict__ x,
    float* __restrict__ out,
    int dtype,
    int rows,
    int cols,
    size_t row_bytes,
    int n_tok
) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.y);
    const int lane = int(threadIdx.x);
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
// 8. Fine-Grained Sparse MoE Router + Indexed Expert Projection (`ADR-018`)
// ============================================================================
// Grid: (n_tok, 1, 1), Block: (32, 1, 1) — deterministic top-k selection matching CPU & Metal!
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
    if (threadIdx.x != 0) return;
    const int t = int(blockIdx.x);
    const float* row = logits + size_t(t) * n_expert;
    uint32_t* out_ids = top_ids + size_t(t) * k_used;
    float* out_w = top_weights + size_t(t) * k_used;

    float probs[256];
    float sel[256];
    const int ne = min(n_expert, 256);

    if (gating_func == 2) {
        for (int e = 0; e < ne; ++e) {
            probs[e] = 1.0f / (1.0f + expf(-row[e]));
        }
    } else if (gating_func == 4) {
        for (int e = 0; e < ne; ++e) {
            const float x = row[e];
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            probs[e] = sqrtf(fmaxf(sp, 0.0f));
        }
    } else if (gating_func == 3) {
        for (int e = 0; e < ne; ++e) probs[e] = row[e];
    } else {
        float mx = -CUDART_INF_F;
        for (int e = 0; e < ne; ++e) mx = fmaxf(mx, row[e]);
        float s = 0.0f;
        for (int e = 0; e < ne; ++e) {
            probs[e] = expf(row[e] - mx);
            s += probs[e];
        }
        const float inv = 1.0f / s;
        for (int e = 0; e < ne; ++e) probs[e] *= inv;
    }

    for (int e = 0; e < ne; ++e) {
        sel[e] = probs[e] + (probs_bias ? probs_bias[e] : 0.0f);
    }

    for (int k = 0; k < k_used; ++k) {
        int best_i = 0;
        float best_v = -CUDART_INF_F;
        for (int e = 0; e < ne; ++e) {
            if (sel[e] > best_v) {
                best_v = sel[e];
                best_i = e;
            }
        }
        out_ids[k] = uint32_t(best_i);
        out_w[k] = probs[best_i];
        sel[best_i] = -CUDART_INF_F;
    }

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

// Indexed expert projection for single-token decode (`slot_idx in 0..k_used`):
// Reads `exp_id = top_ids[slot_idx]`, projects `w_exps[exp_id]` (`[rows, cols]`) against `x` (`[cols]`),
// and writes/accumulates scaled by `scale_slot ? top_weights[slot_idx] : 1.0f`!
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

// Grid: (1, 1, 1), Block: (256, 1, 1)
__global__ void argmax_f32(
    const float* __restrict__ logits,
    uint32_t* __restrict__ out_id,
    int vocab_size
) {
    const int tid = int(threadIdx.x);
    float best_v = -CUDART_INF_F;
    uint32_t best_i = 0;

    for (int i = tid; i < vocab_size; i += 256) {
        const float v = logits[i];
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
            *out_id = idx;
        }
    }
}

} // extern "C"
