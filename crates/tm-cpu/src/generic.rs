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

#[inline]
fn sum_sq_8wide_f32(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks8 = n / 8;
    let mut acc0 = [0.0f32; 4];
    let mut acc1 = [0.0f32; 4];
    for c in 0..chunks8 {
        let base = c * 8;
        for lane in 0..4 {
            let v0 = x[base + lane];
            let v1 = x[base + 4 + lane];
            acc0[lane] = v0.mul_add(v0, acc0[lane]);
            acc1[lane] = v1.mul_add(v1, acc1[lane]);
        }
    }
    let s0 = acc0[0] + acc1[0];
    let s1 = acc0[1] + acc1[1];
    let s2 = acc0[2] + acc1[2];
    let s3 = acc0[3] + acc1[3];
    let mut sum = (s0 + s1) + (s2 + s3);
    for &v in &x[chunks8 * 8..] {
        sum = v.mul_add(v, sum);
    }
    sum
}

/// Scalar reference `RMSNorm` matching 8-wide SIMD pairwise FMA reduction bit-for-bit.
pub fn rms_norm_f32(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(out.len(), n);
    assert_eq!(weight.len(), n);
    if n == 0 {
        return;
    }
    let ss = sum_sq_8wide_f32(x);
    let scale = (1.0 / (((ss as f64) / (n as f64)) + (eps as f64)).sqrt()) as f32;
    for i in 0..n {
        out[i] = (x[i] * scale) * weight[i];
    }
}

/// Scalar reference in-place `L2` normalization matching 8-wide SIMD pairwise FMA bit-for-bit.
pub fn l2_norm_f32(x: &mut [f32], eps: f32) {
    let ss = sum_sq_8wide_f32(x);
    let scale = (1.0 / ((ss as f64) + (eps as f64)).sqrt()) as f32;
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Scalar reference single-head, single-token Gated DeltaNet recurrent step.
///
/// Uses 8-wide pairwise `f32::mul_add` accumulation + single state store per element so
/// SIMD (`vfmaq_f32` + `vpaddq_f32`) matches this reference **bit-for-bit** (`to_bits()`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence_head(
    q_h: &[f32],
    k_h: &[f32],
    v_h: &[f32],
    g: f32,
    beta: f32,
    scale: f32,
    s_vh: &mut [f32],
    o_h: &mut [f32],
    head_dim: usize,
) {
    assert_eq!(q_h.len(), head_dim);
    assert_eq!(k_h.len(), head_dim);
    assert_eq!(v_h.len(), head_dim);
    assert_eq!(s_vh.len(), head_dim * head_dim);
    assert_eq!(o_h.len(), head_dim);

    let chunks8 = head_dim / 8;
    for j in 0..head_dim {
        let s_row = &mut s_vh[j * head_dim..(j + 1) * head_dim];

        // Pass 1: read-only dot product `sum_kv = sum_i ((s_row[i] * g) * k_h[i])`
        let mut kv0 = [0.0f32; 4];
        let mut kv1 = [0.0f32; 4];
        for c in 0..chunks8 {
            let base = c * 8;
            for lane in 0..4 {
                let d0 = s_row[base + lane] * g;
                let d1 = s_row[base + 4 + lane] * g;
                kv0[lane] = d0.mul_add(k_h[base + lane], kv0[lane]);
                kv1[lane] = d1.mul_add(k_h[base + 4 + lane], kv1[lane]);
            }
        }
        let mut sum_kv =
            ((kv0[0] + kv1[0]) + (kv0[1] + kv1[1])) + ((kv0[2] + kv1[2]) + (kv0[3] + kv1[3]));
        for i in chunks8 * 8..head_dim {
            let d = s_row[i] * g;
            sum_kv = d.mul_add(k_h[i], sum_kv);
        }

        let delta_j = (v_h[j] - sum_kv) * beta;

        // Pass 2: single memory write `s_row[i] = delta_j * k_h[i] + (s_row[i] * g)` + query dot
        let mut oq0 = [0.0f32; 4];
        let mut oq1 = [0.0f32; 4];
        for c in 0..chunks8 {
            let base = c * 8;
            for lane in 0..4 {
                let u0 = delta_j.mul_add(k_h[base + lane], s_row[base + lane] * g);
                let u1 = delta_j.mul_add(k_h[base + 4 + lane], s_row[base + 4 + lane] * g);
                s_row[base + lane] = u0;
                s_row[base + 4 + lane] = u1;
                oq0[lane] = u0.mul_add(q_h[base + lane], oq0[lane]);
                oq1[lane] = u1.mul_add(q_h[base + 4 + lane], oq1[lane]);
            }
        }
        let mut sum_oq =
            ((oq0[0] + oq1[0]) + (oq0[1] + oq1[1])) + ((oq0[2] + oq1[2]) + (oq0[3] + oq1[3]));
        for i in chunks8 * 8..head_dim {
            let u = delta_j.mul_add(k_h[i], s_row[i] * g);
            s_row[i] = u;
            sum_oq = u.mul_add(q_h[i], sum_oq);
        }

        o_h[j] = sum_oq * scale;
    }
}
