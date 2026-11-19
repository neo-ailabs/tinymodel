//! ARMv8.6 `i8mm` (`smmla` via `asm!`) kernels for `Q4_0 x Q8_0`.
//!
//! Ported from ggml / Strata:
//! - Copyright (c) 2023-2026 The ggml authors (MIT License)
//! - Copyright (c) 2026 Niko1221 and the Strata contributors (MIT License)
//! - Upstream reference: `Strata/src/quant/arm/arm_dots.inc:104-175`
//!   (ggml `ggml/src/ggml-cpu/arch/arm/quants.c` commit `3cf03257`).

use core::arch::aarch64::*;
use tm_core::{BlockQ4_0, BlockQ8_0};

use super::asm::{f16_to_f32_hw, vmmlaq_s32};
use super::dotprod;

/// Computes a 2-row x 2-token `Q4_0 x Q8_0` tile using `smmla` (`i8mm`).
///
/// Returns `[dot(w0, x0), dot(w0, x1), dot(w1, x0), dot(w1, x1)]`.
///
/// # Safety
/// Caller must ensure CPU supports `i8mm` and all 4 slices have equal length.
#[target_feature(enable = "neon,dotprod,i8mm")]
#[inline]
pub unsafe fn vec_dot_2x2_q4_0_q8_0(
    w0: &[BlockQ4_0],
    w1: &[BlockQ4_0],
    x0: &[BlockQ8_0],
    x1: &[BlockQ8_0],
) -> [f32; 4] {
    let nb = w0.len();
    debug_assert_eq!(w1.len(), nb);
    debug_assert_eq!(x0.len(), nb);
    debug_assert_eq!(x1.len(), nb);

    // SAFETY: `i` stays strictly within `0..nb`; `i8mm` is enabled on this function.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);
        let mut sumv = vdupq_n_f32(0.0);

        for i in 0..nb {
            let bx0 = w0.get_unchecked(i);
            let bx1 = w1.get_unchecked(i);
            let by0 = x0.get_unchecked(i);
            let by1 = x1.get_unchecked(i);

            let v0_0 = vld1q_u8(bx0.qs.as_ptr());
            let v0_1 = vld1q_u8(bx1.qs.as_ptr());

            let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
            let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0_0)), s8b);
            let x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
            let x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0_1)), s8b);

            let y0_l = vld1q_s8(by0.qs.as_ptr());
            let y0_h = vld1q_s8(by0.qs.as_ptr().add(16));
            let y1_l = vld1q_s8(by1.qs.as_ptr());
            let y1_h = vld1q_s8(by1.qs.as_ptr().add(16));

            let dx0 = f16_to_f32_hw(bx0.d);
            let dx1 = f16_to_f32_hw(bx1.d);
            let dy0 = f16_to_f32_hw(by0.d);
            let dy1 = f16_to_f32_hw(by1.d);

            let scales = [dx0 * dy0, dx0 * dy1, dx1 * dy0, dx1 * dy1];
            let scale = vld1q_f32(scales.as_ptr());

            let l0 = vreinterpretq_s8_s64(vzip1q_s64(
                vreinterpretq_s64_s8(x0_l),
                vreinterpretq_s64_s8(x1_l),
            ));
            let l1 = vreinterpretq_s8_s64(vzip2q_s64(
                vreinterpretq_s64_s8(x0_l),
                vreinterpretq_s64_s8(x1_l),
            ));
            let l2 = vreinterpretq_s8_s64(vzip1q_s64(
                vreinterpretq_s64_s8(x0_h),
                vreinterpretq_s64_s8(x1_h),
            ));
            let l3 = vreinterpretq_s8_s64(vzip2q_s64(
                vreinterpretq_s64_s8(x0_h),
                vreinterpretq_s64_s8(x1_h),
            ));

            let r0 = vreinterpretq_s8_s64(vzip1q_s64(
                vreinterpretq_s64_s8(y0_l),
                vreinterpretq_s64_s8(y1_l),
            ));
            let r1 = vreinterpretq_s8_s64(vzip2q_s64(
                vreinterpretq_s64_s8(y0_l),
                vreinterpretq_s64_s8(y1_l),
            ));
            let r2 = vreinterpretq_s8_s64(vzip1q_s64(
                vreinterpretq_s64_s8(y0_h),
                vreinterpretq_s64_s8(y1_h),
            ));
            let r3 = vreinterpretq_s8_s64(vzip2q_s64(
                vreinterpretq_s64_s8(y0_h),
                vreinterpretq_s64_s8(y1_h),
            ));

            let mut acc = vmmlaq_s32(zero, l0, r0);
            acc = vmmlaq_s32(acc, l1, r1);
            acc = vmmlaq_s32(acc, l2, r2);
            acc = vmmlaq_s32(acc, l3, r3);

            sumv = vmlaq_f32(sumv, vcvtq_f32_s32(acc), scale);
        }

        let mut res = [0.0f32; 4];
        vst1q_f32(res.as_mut_ptr(), sumv);
        res
    }
}

/// Multi-token `Q4_0 x Q8_0` GEMM using ARMv8.6 `i8mm` `4-row x 8-token` `smmla` register blocking
/// (falls back cleanly to `2x2` `smmla` and 4-row `dotprod` for tail rows/tokens).
///
/// # Safety
/// Caller must ensure CPU supports `i8mm` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod,i8mm")]
pub unsafe fn gemm_q4_0_q8_0(
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

    if n_tokens < 8 {
        // SAFETY: `dotprod` is enabled and dimensions verified above.
        // For small batches (< 8, e.g. speculative verify), 4-token `dotprod` is faster
        // than 2x2 `smmla` setup AND preserves bit-exact batch-width invariance.
        unsafe {
            dotprod::gemm_q4_0_q8_0(w, x, out, rows, cols_blocks, n_tokens);
        }
        return;
    }

    // SAFETY: All row and token indices are strictly bounded by `rows`, `cols_blocks`, `n_tokens`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        // Fast path: 4 weight rows x 8 tokens per inner block loop (8 `2x2` smmla tiles = 32 smmla/block)
        let mut r = 0usize;
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut t = 0usize;
            while t + 8 <= n_tokens {
                let mut sum_a0 = vdupq_n_f32(0.0);
                let mut sum_a1 = vdupq_n_f32(0.0);
                let mut sum_a2 = vdupq_n_f32(0.0);
                let mut sum_a3 = vdupq_n_f32(0.0);
                let mut sum_b0 = vdupq_n_f32(0.0);
                let mut sum_b1 = vdupq_n_f32(0.0);
                let mut sum_b2 = vdupq_n_f32(0.0);
                let mut sum_b3 = vdupq_n_f32(0.0);

                for ib in 0..cols_blocks {
                    // Unpack & zip 4 weight rows ONCE for all 8 tokens
                    let bx0 = w0.get_unchecked(ib);
                    let bx1 = w1.get_unchecked(ib);
                    let bx2 = w2.get_unchecked(ib);
                    let bx3 = w3.get_unchecked(ib);

                    let q0 = vld1q_u8(bx0.qs.as_ptr());
                    let q1 = vld1q_u8(bx1.qs.as_ptr());
                    let q2 = vld1q_u8(bx2.qs.as_ptr());
                    let q3 = vld1q_u8(bx3.qs.as_ptr());

                    let x0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q0, m4b)), s8b);
                    let x0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0)), s8b);
                    let x1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q1, m4b)), s8b);
                    let x1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1)), s8b);
                    let x2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q2, m4b)), s8b);
                    let x2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q2)), s8b);
                    let x3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3, m4b)), s8b);
                    let x3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q3)), s8b);

                    let la0 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(x0l),
                        vreinterpretq_s64_s8(x1l),
                    ));
                    let la1 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(x0l),
                        vreinterpretq_s64_s8(x1l),
                    ));
                    let la2 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(x0h),
                        vreinterpretq_s64_s8(x1h),
                    ));
                    let la3 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(x0h),
                        vreinterpretq_s64_s8(x1h),
                    ));

                    let lb0 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(x2l),
                        vreinterpretq_s64_s8(x3l),
                    ));
                    let lb1 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(x2l),
                        vreinterpretq_s64_s8(x3l),
                    ));
                    let lb2 = vreinterpretq_s8_s64(vzip1q_s64(
                        vreinterpretq_s64_s8(x2h),
                        vreinterpretq_s64_s8(x3h),
                    ));
                    let lb3 = vreinterpretq_s8_s64(vzip2q_s64(
                        vreinterpretq_s64_s8(x2h),
                        vreinterpretq_s64_s8(x3h),
                    ));

                    let dx0 = f16_to_f32_hw(bx0.d);
                    let dx1 = f16_to_f32_hw(bx1.d);
                    let dx2 = f16_to_f32_hw(bx2.d);
                    let dx3 = f16_to_f32_hw(bx3.d);
                    let wd_a = vld1q_f32([dx0, dx0, dx1, dx1].as_ptr());
                    let wd_b = vld1q_f32([dx2, dx2, dx3, dx3].as_ptr());

                    macro_rules! tok_pair {
                        ($tp:expr, $sa:expr, $sb:expr) => {{
                            let by0 = x.get_unchecked(($tp) * cols_blocks + ib);
                            let by1 = x.get_unchecked(($tp + 1) * cols_blocks + ib);
                            let y0l = vld1q_s8(by0.qs.as_ptr());
                            let y0h = vld1q_s8(by0.qs.as_ptr().add(16));
                            let y1l = vld1q_s8(by1.qs.as_ptr());
                            let y1h = vld1q_s8(by1.qs.as_ptr().add(16));

                            let rr0 = vreinterpretq_s8_s64(vzip1q_s64(
                                vreinterpretq_s64_s8(y0l),
                                vreinterpretq_s64_s8(y1l),
                            ));
                            let rr1 = vreinterpretq_s8_s64(vzip2q_s64(
                                vreinterpretq_s64_s8(y0l),
                                vreinterpretq_s64_s8(y1l),
                            ));
                            let rr2 = vreinterpretq_s8_s64(vzip1q_s64(
                                vreinterpretq_s64_s8(y0h),
                                vreinterpretq_s64_s8(y1h),
                            ));
                            let rr3 = vreinterpretq_s8_s64(vzip2q_s64(
                                vreinterpretq_s64_s8(y0h),
                                vreinterpretq_s64_s8(y1h),
                            ));

                            let dy0 = f16_to_f32_hw(by0.d);
                            let dy1 = f16_to_f32_hw(by1.d);
                            let yd = vld1q_f32([dy0, dy1, dy0, dy1].as_ptr());

                            let mut acc_a = vmmlaq_s32(zero, la0, rr0);
                            acc_a = vmmlaq_s32(acc_a, la1, rr1);
                            acc_a = vmmlaq_s32(acc_a, la2, rr2);
                            acc_a = vmmlaq_s32(acc_a, la3, rr3);
                            $sa = vfmaq_f32($sa, vcvtq_f32_s32(acc_a), vmulq_f32(wd_a, yd));

                            let mut acc_b = vmmlaq_s32(zero, lb0, rr0);
                            acc_b = vmmlaq_s32(acc_b, lb1, rr1);
                            acc_b = vmmlaq_s32(acc_b, lb2, rr2);
                            acc_b = vmmlaq_s32(acc_b, lb3, rr3);
                            $sb = vfmaq_f32($sb, vcvtq_f32_s32(acc_b), vmulq_f32(wd_b, yd));
                        }};
                    }

                    tok_pair!(t, sum_a0, sum_b0);
                    tok_pair!(t + 2, sum_a1, sum_b1);
                    tok_pair!(t + 4, sum_a2, sum_b2);
                    tok_pair!(t + 6, sum_a3, sum_b3);
                }

                macro_rules! store_pair {
                    ($tp:expr, $sa:expr, $sb:expr) => {{
                        let mut ra = [0.0f32; 4];
                        let mut rb = [0.0f32; 4];
                        vst1q_f32(ra.as_mut_ptr(), $sa);
                        vst1q_f32(rb.as_mut_ptr(), $sb);
                        *out.get_unchecked_mut(($tp) * rows + r) = ra[0];
                        *out.get_unchecked_mut(($tp + 1) * rows + r) = ra[1];
                        *out.get_unchecked_mut(($tp) * rows + r + 1) = ra[2];
                        *out.get_unchecked_mut(($tp + 1) * rows + r + 1) = ra[3];
                        *out.get_unchecked_mut(($tp) * rows + r + 2) = rb[0];
                        *out.get_unchecked_mut(($tp + 1) * rows + r + 2) = rb[1];
                        *out.get_unchecked_mut(($tp) * rows + r + 3) = rb[2];
                        *out.get_unchecked_mut(($tp + 1) * rows + r + 3) = rb[3];
                    }};
                }

                store_pair!(t, sum_a0, sum_b0);
                store_pair!(t + 2, sum_a1, sum_b1);
                store_pair!(t + 4, sum_a2, sum_b2);
                store_pair!(t + 6, sum_a3, sum_b3);

                t += 8;
            }

            // Remaining tokens for rows `r..r+4`
            while t + 2 <= n_tokens {
                let x0 = &x[t * cols_blocks..(t + 1) * cols_blocks];
                let x1 = &x[(t + 1) * cols_blocks..(t + 2) * cols_blocks];
                let ta = vec_dot_2x2_q4_0_q8_0(w0, w1, x0, x1);
                let tb = vec_dot_2x2_q4_0_q8_0(w2, w3, x0, x1);
                *out.get_unchecked_mut(t * rows + r) = ta[0];
                *out.get_unchecked_mut((t + 1) * rows + r) = ta[1];
                *out.get_unchecked_mut(t * rows + r + 1) = ta[2];
                *out.get_unchecked_mut((t + 1) * rows + r + 1) = ta[3];
                *out.get_unchecked_mut(t * rows + r + 2) = tb[0];
                *out.get_unchecked_mut((t + 1) * rows + r + 2) = tb[1];
                *out.get_unchecked_mut(t * rows + r + 3) = tb[2];
                *out.get_unchecked_mut((t + 1) * rows + r + 3) = tb[3];
                t += 2;
            }
            if t < n_tokens {
                let xt = &x[t * cols_blocks..(t + 1) * cols_blocks];
                let mut o4 = [0.0f32; 4];
                dotprod::gemv_q4_0_q8_0(
                    &w[r * cols_blocks..(r + 4) * cols_blocks],
                    xt,
                    &mut o4,
                    4,
                    cols_blocks,
                );
                *out.get_unchecked_mut(t * rows + r) = o4[0];
                *out.get_unchecked_mut(t * rows + r + 1) = o4[1];
                *out.get_unchecked_mut(t * rows + r + 2) = o4[2];
                *out.get_unchecked_mut(t * rows + r + 3) = o4[3];
            }

            r += 4;
        }

        // Tail rows `< 4` across all `n_tokens`
        while r < rows {
            let wr = &w[r * cols_blocks..(r + 1) * cols_blocks];
            for t in 0..n_tokens {
                let xt = &x[t * cols_blocks..(t + 1) * cols_blocks];
                *out.get_unchecked_mut(t * rows + r) = dotprod::vec_dot_q4_0_q8_0(wr, xt);
            }
            r += 1;
        }
    }
}
