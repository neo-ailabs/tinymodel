//! Scalar reference implementations (`generic` ISA tier).
//!
//! Serves as the ground-truth arithmetic reference for all SIMD/GPU parity tests.
//! Algorithm matches ggml's `ggml_vec_dot_q4_0_q8_0` scalar path:
//! Copyright (c) 2023-2026 The ggml authors (MIT License).
//! Upstream reference: llama.cpp `ggml/src/ggml-cpu/quants.c` (commit `3cf03257`).

use tm_core::{
    BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, QK_K, QK8_0, QK8_1,
    f16_to_f32, unpack_q5k_scales_mins,
};

/// Scalar dot product between one row of `BlockQ4_0` and one vector of `BlockQ8_0`.
#[inline]
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sumf = 0.0f32;

    for (bx, by) in x.iter().zip(y.iter()) {
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..(QK8_0 / 2) {
            let byte = bx.qs[j];
            let v0 = ((byte & 0x0F) as i32) - 8;
            let v1 = ((byte >> 4) as i32) - 8;

            sumi0 += v0 * (by.qs[j] as i32);
            sumi1 += v1 * (by.qs[j + QK8_0 / 2] as i32);
        }

        let sumi = sumi0 + sumi1;
        sumf += (sumi as f32) * (f16_to_f32(bx.d) * f16_to_f32(by.d));
    }

    sumf
}

/// Scalar GEMV: `out[r] = dot(w[r], x)` for `r in 0..rows`.
/// `w` has `rows * cols_blocks` blocks (row-major); `x` has `cols_blocks` blocks.
pub fn gemv_q4_0_q8_0(
    w: &[BlockQ4_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    for r in 0..rows {
        let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
        out[r] = vec_dot_q4_0_q8_0(row, x);
    }
}

/// Scalar multi-token GEMM: `out[t * rows + r] = dot(w[r], x[t])`.
/// `w`: `rows * cols_blocks`; `x`: `n_tokens * cols_blocks`; `out`: `n_tokens * rows`.
pub fn gemm_q4_0_q8_0(
    w: &[BlockQ4_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);

    for t in 0..n_tokens {
        let xt = &x[t * cols_blocks..(t + 1) * cols_blocks];
        let ot = &mut out[t * rows..(t + 1) * rows];
        gemv_q4_0_q8_0(w, xt, ot, rows, cols_blocks);
    }
}

/// Scalar dot product between `BlockQ4_1` and `BlockQ8_1`.
#[inline]
pub fn vec_dot_q4_1_q8_1(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sumf = 0.0f32;

    for (bx, by) in x.iter().zip(y.iter()) {
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..(QK8_1 / 2) {
            let byte = bx.qs[j];
            let v0 = (byte & 0x0F) as i32;
            let v1 = (byte >> 4) as i32;

            sumi0 += v0 * (by.qs[j] as i32);
            sumi1 += v1 * (by.qs[j + QK8_1 / 2] as i32);
        }

        let sumi = sumi0 + sumi1;
        sumf += (f16_to_f32(bx.d) * f16_to_f32(by.d)) * (sumi as f32)
            + f16_to_f32(bx.m) * f16_to_f32(by.s);
    }

    sumf
}

/// Scalar dot product between `BlockQ8_0` and `BlockQ8_0`.
#[inline]
pub fn vec_dot_q8_0_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sumf = 0.0f32;

    for (bx, by) in x.iter().zip(y.iter()) {
        let mut sumi: i32 = 0;
        for j in 0..QK8_0 {
            sumi += (bx.qs[j] as i32) * (by.qs[j] as i32);
        }
        sumf += (sumi as f32) * (f16_to_f32(bx.d) * f16_to_f32(by.d));
    }

    sumf
}

/// Scalar dot product between `BlockQ5K` and `BlockQ8K`.
/// Matches ggml `ggml_vec_dot_q5_K_q8_K` (Copyright (c) 2023-2026 The ggml authors, MIT).
#[inline]
pub fn vec_dot_q5_k_q8_k(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sumf = 0.0f32;

    for (bx, by) in x.iter().zip(y.iter()) {
        let (sc, mins) = unpack_q5k_scales_mins(&bx.scales);

        let mut sumi_mins: i32 = 0;
        for j in 0..(QK_K / 16) {
            sumi_mins += (by.bsums[j] as i32) * (mins[j / 2] as i32);
        }

        let mut sumi: i32 = 0;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for chunk in 0..4 {
            let ql = &bx.qs[chunk * 32..(chunk + 1) * 32];
            let q8_lo = &by.qs[chunk * 64..chunk * 64 + 32];
            let q8_hi = &by.qs[chunk * 64 + 32..(chunk + 1) * 64];

            let mut dot_lo: i32 = 0;
            let mut dot_hi: i32 = 0;
            for l in 0..32 {
                let q5_lo = ((ql[l] & 0x0F) + if (bx.qh[l] & u1) != 0 { 16 } else { 0 }) as i32;
                let q5_hi = ((ql[l] >> 4) + if (bx.qh[l] & u2) != 0 { 16 } else { 0 }) as i32;
                dot_lo += q5_lo * (q8_lo[l] as i32);
                dot_hi += q5_hi * (q8_hi[l] as i32);
            }
            sumi += (sc[2 * chunk] as i32) * dot_lo + (sc[2 * chunk + 1] as i32) * dot_hi;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }

        let d = f16_to_f32(bx.d) * by.d;
        let dmin = f16_to_f32(bx.dmin) * by.d;
        sumf += d * (sumi as f32) - dmin * (sumi_mins as f32);
    }

    sumf
}

/// Scalar dot product between `BlockQ6K` and `BlockQ8K`.
/// Matches ggml `ggml_vec_dot_q6_K_q8_K` (Copyright (c) 2023-2026 The ggml authors, MIT).
#[inline]
pub fn vec_dot_q6_k_q8_k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sumf = 0.0f32;

    for (bx, by) in x.iter().zip(y.iter()) {
        let mut aux8 = [0i8; QK_K];
        for half in 0..2 {
            let ql = &bx.ql[half * 64..(half + 1) * 64];
            let qh = &bx.qh[half * 32..(half + 1) * 32];
            let a = &mut aux8[half * 128..(half + 1) * 128];
            for l in 0..32 {
                a[l] = (((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i8).wrapping_sub(32);
                a[l + 32] =
                    (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8).wrapping_sub(32);
                a[l + 64] = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8).wrapping_sub(32);
                a[l + 96] =
                    (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8).wrapping_sub(32);
            }
        }

        let mut isum: i32 = 0;
        for j in 0..(QK_K / 16) {
            let scale = bx.scales[j] as i32;
            let mut sub: i32 = 0;
            let base = j * 16;
            for l in 0..16 {
                sub += (aux8[base + l] as i32) * (by.qs[base + l] as i32);
            }
            isum += scale * sub;
        }

        sumf += f16_to_f32(bx.d) * by.d * (isum as f32);
    }

    sumf
}

/// Scalar `f32` dot product.
#[inline]
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sum = 0.0f32;
    for (&a, &b) in x.iter().zip(y.iter()) {
        sum += a * b;
    }
    sum
}

/// Scalar GEMV for `Q8_0 x Q8_0`.
pub fn gemv_q8_0_q8_0(
    w: &[BlockQ8_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        out[r] = vec_dot_q8_0_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
    }
}

/// Scalar GEMV for `Q5_K x Q8_K`.
pub fn gemv_q5_k_q8_k(
    w: &[BlockQ5K],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        out[r] = vec_dot_q5_k_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
    }
}

/// Scalar GEMV for `Q6_K x Q8_K`.
pub fn gemv_q6_k_q8_k(
    w: &[BlockQ6K],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        out[r] = vec_dot_q6_k_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
    }
}

/// Scalar GEMV for `Q4_1 x Q8_1`.
pub fn gemv_q4_1_q8_1(
    w: &[BlockQ4_1],
    x: &[BlockQ8_1],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    for r in 0..rows {
        out[r] = vec_dot_q4_1_q8_1(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
    }
}

/// Scalar `f32 x f16` dot product (for F16 KV attention).
#[inline]
pub fn dot_f32_f16(q: &[f32], k_f16: &[u16]) -> f32 {
    debug_assert_eq!(q.len(), k_f16.len());
    let mut sum = 0.0f32;
    for (&qv, &kh) in q.iter().zip(k_f16.iter()) {
        sum += qv * f16_to_f32(kh);
    }
    sum
}

/// Scalar `out[d] += w * f16_to_f32(v_f16[d])` (for F16 V aggregation).
#[inline]
pub fn axpy_f32_f16(out: &mut [f32], w: f32, v_f16: &[u16]) {
    debug_assert_eq!(out.len(), v_f16.len());
    for (o, &vh) in out.iter_mut().zip(v_f16.iter()) {
        *o += w * f16_to_f32(vh);
    }
}

/// Scalar `f32` -> IEEE 754 `f16` (`u16`) slice conversion.
#[inline]
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = tm_core::f32_to_f16(s);
    }
}

/// Scalar `f32 -> Q8_0` activation quantization.
#[inline]
pub fn quantize_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    tm_core::quantize_row_q8_0(src, dst);
}

/// Scalar `f32 -> Q8_1` activation quantization.
#[inline]
pub fn quantize_q8_1(src: &[f32], dst: &mut [BlockQ8_1]) {
    tm_core::quantize_row_q8_1(src, dst);
}
