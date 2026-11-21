//! ARMv8.6 `i8mm` (`smmla` via `asm!`) kernels for `Q4_0 x Q8_0`.
//!
//! Ported from ggml / Strata:
//! - Copyright (c) 2023-2026 The ggml authors (MIT License)
//! - Copyright (c) 2026 Niko1221 and the Strata contributors (MIT License)
//! - Upstream reference: `Strata/src/quant/arm/arm_dots.inc:104-175`
//!   (ggml `ggml/src/ggml-cpu/arch/arm/quants.c` commit `3cf03257`).

use core::arch::aarch64::*;
use tm_core::{BlockQ4_0, BlockQ4_0x4, BlockQ8_0};

use super::asm::{f16_to_f32_hw, vmmlaq_s32};

/// Single-row `vec_dot_q4_0_q8_0` for `Isa::NeonI8mm` using exact per-block `i32` reduction
/// and `vfmaq_f32` so 1-token `gemv` and multi-token `smmla` `gemm` are 100% bit-for-bit identical
/// across ALL batch widths (`AGENTS.md` §9 Determinism).
///
/// # Safety
/// Caller must ensure CPU supports `i8mm` and `x.len() == y.len()`.
#[target_feature(enable = "neon,dotprod,i8mm")]
#[inline]
pub unsafe fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    // SAFETY: `ib` stays strictly within `0..nb`; `dotprod` + `i8mm` enabled.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);
        let mut sumv = vdupq_n_f32(0.0);

        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let q = vld1q_u8(bx.qs.as_ptr());
            let xl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4b)), s8b);
            let xh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), s8b);
            let yl = vld1q_s8(by.qs.as_ptr());
            let yh = vld1q_s8(by.qs.as_ptr().add(16));
            let p = vdotq_s32(vdotq_s32(zero, xl, yl), xh, yh);
            let sumi = vaddvq_s32(p);
            let scale = f16_to_f32_hw(bx.d) * f16_to_f32_hw(by.d);
            sumv = vfmaq_f32(sumv, vdupq_n_f32(sumi as f32), vdupq_n_f32(scale));
        }

        vgetq_lane_f32::<0>(sumv)
    }
}

/// 4-row unrolled `gemv_q4_0_q8_0` for `Isa::NeonI8mm`: uses `vdotq_s32` + `vpaddq_s32` to compute
/// the exact 32-element `i32` dot product per block for 4 rows simultaneously, then accumulates
/// with `vfmaq_f32` — **100% bit-for-bit identical** to `smmla` `gemm_q4_0_q8_0` across all batch widths!
///
/// # Safety
/// Caller must ensure CPU supports `i8mm` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod,i8mm")]
pub unsafe fn gemv_q4_0_q8_0(
    w: &[BlockQ4_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: All row and block indices are strictly bounded.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut sumv = vdupq_n_f32(0.0);
            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let yl = vld1q_s8(by.qs.as_ptr());
                let yh = vld1q_s8(by.qs.as_ptr().add(16));
                let dy = f16_to_f32_hw(by.d);

                let bx0 = w0.get_unchecked(ib);
                let bx1 = w1.get_unchecked(ib);
                let bx2 = w2.get_unchecked(ib);
                let bx3 = w3.get_unchecked(ib);

                let q0 = vld1q_u8(bx0.qs.as_ptr());
                let q1 = vld1q_u8(bx1.qs.as_ptr());
                let q2 = vld1q_u8(bx2.qs.as_ptr());
                let q3 = vld1q_u8(bx3.qs.as_ptr());

                let p0 = vdotq_s32(
                    vdotq_s32(
                        zero,
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q0, m4b)), s8b),
                        yl,
                    ),
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0)), s8b),
                    yh,
                );
                let p1 = vdotq_s32(
                    vdotq_s32(
                        zero,
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q1, m4b)), s8b),
                        yl,
                    ),
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1)), s8b),
                    yh,
                );
                let p2 = vdotq_s32(
                    vdotq_s32(
                        zero,
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q2, m4b)), s8b),
                        yl,
                    ),
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q2)), s8b),
                    yh,
                );
                let p3 = vdotq_s32(
                    vdotq_s32(
                        zero,
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3, m4b)), s8b),
                        yl,
                    ),
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q3)), s8b),
                    yh,
                );

                let p01 = vpaddq_s32(p0, p1);
                let p23 = vpaddq_s32(p2, p3);
                let acc4 = vpaddq_s32(p01, p23);

                let scales = [
                    f16_to_f32_hw(bx0.d) * dy,
                    f16_to_f32_hw(bx1.d) * dy,
                    f16_to_f32_hw(bx2.d) * dy,
                    f16_to_f32_hw(bx3.d) * dy,
                ];
                sumv = vfmaq_f32(sumv, vcvtq_f32_s32(acc4), vld1q_f32(scales.as_ptr()));
            }

            vst1q_f32(out.as_mut_ptr().add(r), sumv);
            r += 4;
        }

        while r < rows {
            let wr = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q4_0_q8_0(wr, x);
            r += 1;
        }
    }
}

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

            sumv = vfmaq_f32(sumv, vcvtq_f32_s32(acc), scale);
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

    if n_tokens == 1 {
        // SAFETY: `i8mm` is enabled and dimensions verified above.
        unsafe {
            gemv_q4_0_q8_0(w, x, out, rows, cols_blocks);
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

            // Remaining tokens `< 8` (`n_tokens = 2..7`, e.g. speculative verify!) for rows `r..r+4`:
            // Unpack 4 weight rows ONCE per K-block and run `smmla` across pairs of tokens!
            while t + 2 <= n_tokens {
                let mut sa = vdupq_n_f32(0.0);
                let mut sb = vdupq_n_f32(0.0);
                for ib in 0..cols_blocks {
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

                    let by0 = x.get_unchecked(t * cols_blocks + ib);
                    let by1 = x.get_unchecked((t + 1) * cols_blocks + ib);
                    let y0l = vreinterpretq_s64_s8(vld1q_s8(by0.qs.as_ptr()));
                    let y0h = vreinterpretq_s64_s8(vld1q_s8(by0.qs.as_ptr().add(16)));
                    let y1l = vreinterpretq_s64_s8(vld1q_s8(by1.qs.as_ptr()));
                    let y1h = vreinterpretq_s64_s8(vld1q_s8(by1.qs.as_ptr().add(16)));

                    let rr0 = vreinterpretq_s8_s64(vzip1q_s64(y0l, y1l));
                    let rr1 = vreinterpretq_s8_s64(vzip2q_s64(y0l, y1l));
                    let rr2 = vreinterpretq_s8_s64(vzip1q_s64(y0h, y1h));
                    let rr3 = vreinterpretq_s8_s64(vzip2q_s64(y0h, y1h));

                    let dx0 = f16_to_f32_hw(bx0.d);
                    let dx1 = f16_to_f32_hw(bx1.d);
                    let dx2 = f16_to_f32_hw(bx2.d);
                    let dx3 = f16_to_f32_hw(bx3.d);
                    let dy0 = f16_to_f32_hw(by0.d);
                    let dy1 = f16_to_f32_hw(by1.d);
                    let wd_a = vld1q_f32([dx0, dx0, dx1, dx1].as_ptr());
                    let wd_b = vld1q_f32([dx2, dx2, dx3, dx3].as_ptr());
                    let yd = vld1q_f32([dy0, dy1, dy0, dy1].as_ptr());

                    let acc_a = vmmlaq_s32(
                        vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(zero, la0, rr0), la1, rr1), la2, rr2),
                        la3,
                        rr3,
                    );
                    let acc_b = vmmlaq_s32(
                        vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(zero, lb0, rr0), lb1, rr1), lb2, rr2),
                        lb3,
                        rr3,
                    );
                    sa = vfmaq_f32(sa, vcvtq_f32_s32(acc_a), vmulq_f32(wd_a, yd));
                    sb = vfmaq_f32(sb, vcvtq_f32_s32(acc_b), vmulq_f32(wd_b, yd));
                }

                let mut ta = [0.0f32; 4];
                let mut tb = [0.0f32; 4];
                vst1q_f32(ta.as_mut_ptr(), sa);
                vst1q_f32(tb.as_mut_ptr(), sb);
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
                gemv_q4_0_q8_0(
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
                *out.get_unchecked_mut(t * rows + r) = vec_dot_q4_0_q8_0(wr, xt);
            }
            r += 1;
        }
    }
}

/// Single-token 4-row `gemv_q4_0x4_q8_0` for `Isa::NeonI8mm`.
///
/// Uses `vdotq_s32` + `vpaddq_s32` + `vfmaq_f32` on contiguous 72-byte `BlockQ4_0x4` super-blocks
/// (single hardware prefetcher stream instead of 4 strided streams!), matching `generic`, `dotprod`,
/// and multi-token `smmla` `gemm_q4_0x4_q8_0` **bit-for-bit** (`to_bits()`).
///
/// # Safety
/// Caller must ensure CPU supports `i8mm` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod,i8mm")]
pub unsafe fn gemv_q4_0x4_q8_0(
    w: &[BlockQ4_0x4],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    // SAFETY: `i8mm` implies `dotprod` + `neon` (verified in `isa.rs` and `#[target_feature]`).
    unsafe {
        super::dotprod::gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks);
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct ZippedQ8BlockPair {
    rr: [[i8; 16]; 4],
    yd: [f32; 4],
}

/// Zero-shuffle `8-row x 8-token` (`smmla`) GEMM microkernel for `BlockQ4_0x4 x BlockQ8_0`
/// (Apple M4 & Google Pixel 8 Tensor G3).
///
/// - Pre-zips token pairs `(t, t+1)` ONCE per outer token tile (up to `MAX_PACK_BLOCKS = 512`, i.e., `cols <= 16384`),
///   eliminating 100% of activation `vzip` instructions across all row groups!
/// - `BlockQ4_0x4` is already stored in `smmla`'s `[r0_0..8, r1_0..8]` 64-bit interleaved order,
///   eliminating 100% of weight `vzip` instructions AND reading a single contiguous memory stream!
/// - Computes **8 weight rows x 8 tokens** (`16` `float32x4_t` accumulators, `64` `smmla` instructions per K-block)
///   while preserving **100% bit-for-bit invariance** (`to_bits()`) across all batch widths `1..=32`.
///
/// # Safety
/// Caller must ensure CPU supports `i8mm` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod,i8mm")]
pub unsafe fn gemm_q4_0x4_q8_0(
    w: &[BlockQ4_0x4],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(rows % 4, 0);
    assert_eq!(w.len(), (rows / 4) * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);

    if n_tokens == 1 {
        // SAFETY: Verified dimensions above.
        unsafe {
            gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks);
        }
        return;
    }

    const K_TILE: usize = 32; // 32 * 4 * 80 = 10 KiB: L1D-resident & musl-stack safe!

    // SAFETY: All group, token, and block indices are strictly bounded by `rows / 4`, `n_tokens`, `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);
        let n_groups = rows / 4;

        let mut packed_pairs = [ZippedQ8BlockPair {
            rr: [[0; 16]; 4],
            yd: [0.0; 4],
        }; 4 * K_TILE];

        let mut t = 0usize;
        while t + 8 <= n_tokens {
            let mut rg = 0usize;
            // Process all row groups across K-tiles of 32 blocks; wait — pre-zipping per K-tile
            // outside the `rg` loop would require storing `n_groups` partial sums, OR pre-zipping
            // up to `K_TILE = 32` blocks when `cols_blocks <= 32`, and for `cols_blocks <= 128` (`cols <= 4096`,
            // covering ALL <=8B attention/FFN inputs except FFN down!): `128 * 4 * 80 = 40 KiB`.
            // Even better: allocate zero stack > 10 KiB by tiling `rg` in chunks of 16 super-rows (64 rows,
            // accumulators = `64 * 8 * 4 = 2 KiB`!), so each K-tile pre-zips 8 tokens ONCE for 64 weight rows!
            while rg < n_groups {
                let rg_end = (rg + 16).min(n_groups);
                let g_cnt = rg_end - rg;
                // 16 groups * 8 floats (4 pairs * 2 groups of 2 rows... wait: 4 rows * 8 tokens = 32 floats/group -> 16 * 32 = 512 f32s = 2 KiB!)
                let mut acc_buf = [[0.0f32; 32]; 16];

                let mut kb = 0usize;
                while kb < cols_blocks {
                    let kend = (kb + K_TILE).min(cols_blocks);
                    let klen = kend - kb;

                    // Pre-zip 4 token pairs `(t..t+8)` for `kb..kend` ONCE across up to 64 weight rows!
                    for p in 0..4 {
                        let tp = t + p * 2;
                        for ki in 0..klen {
                            let ib = kb + ki;
                            let by0 = x.get_unchecked(tp * cols_blocks + ib);
                            let by1 = x.get_unchecked((tp + 1) * cols_blocks + ib);
                            let y0l = vreinterpretq_s64_s8(vld1q_s8(by0.qs.as_ptr()));
                            let y0h = vreinterpretq_s64_s8(vld1q_s8(by0.qs.as_ptr().add(16)));
                            let y1l = vreinterpretq_s64_s8(vld1q_s8(by1.qs.as_ptr()));
                            let y1h = vreinterpretq_s64_s8(vld1q_s8(by1.qs.as_ptr().add(16)));
                            let dst = packed_pairs.get_unchecked_mut(ki * 4 + p);
                            vst1q_s8(
                                dst.rr[0].as_mut_ptr(),
                                vreinterpretq_s8_s64(vzip1q_s64(y0l, y1l)),
                            );
                            vst1q_s8(
                                dst.rr[1].as_mut_ptr(),
                                vreinterpretq_s8_s64(vzip2q_s64(y0l, y1l)),
                            );
                            vst1q_s8(
                                dst.rr[2].as_mut_ptr(),
                                vreinterpretq_s8_s64(vzip1q_s64(y0h, y1h)),
                            );
                            vst1q_s8(
                                dst.rr[3].as_mut_ptr(),
                                vreinterpretq_s8_s64(vzip2q_s64(y0h, y1h)),
                            );
                            let dy0 = f16_to_f32_hw(by0.d);
                            let dy1 = f16_to_f32_hw(by1.d);
                            dst.yd = [dy0, dy1, dy0, dy1];
                        }
                    }

                    let mut lg = 0usize;
                    while lg + 2 <= g_cnt {
                        let wg0 = &w[(rg + lg) * cols_blocks + kb..(rg + lg) * cols_blocks + kend];
                        let wg1 = &w
                            [(rg + lg + 1) * cols_blocks + kb..(rg + lg + 1) * cols_blocks + kend];

                        let ap0 = acc_buf[lg].as_mut_ptr();
                        let ap1 = acc_buf[lg + 1].as_mut_ptr();
                        let mut g0_a0 = vld1q_f32(ap0);
                        let mut g0_b0 = vld1q_f32(ap0.add(4));
                        let mut g0_a1 = vld1q_f32(ap0.add(8));
                        let mut g0_b1 = vld1q_f32(ap0.add(12));
                        let mut g0_a2 = vld1q_f32(ap0.add(16));
                        let mut g0_b2 = vld1q_f32(ap0.add(20));
                        let mut g0_a3 = vld1q_f32(ap0.add(24));
                        let mut g0_b3 = vld1q_f32(ap0.add(28));

                        let mut g1_a0 = vld1q_f32(ap1);
                        let mut g1_b0 = vld1q_f32(ap1.add(4));
                        let mut g1_a1 = vld1q_f32(ap1.add(8));
                        let mut g1_b1 = vld1q_f32(ap1.add(12));
                        let mut g1_a2 = vld1q_f32(ap1.add(16));
                        let mut g1_b2 = vld1q_f32(ap1.add(20));
                        let mut g1_a3 = vld1q_f32(ap1.add(24));
                        let mut g1_b3 = vld1q_f32(ap1.add(28));

                        for ki in 0..klen {
                            let sb0 = wg0.get_unchecked(ki);
                            let sb1 = wg1.get_unchecked(ki);

                            let u0 = vld1q_u8(sb0.qs.as_ptr());
                            let u1 = vld1q_u8(sb0.qs.as_ptr().add(16));
                            let u2 = vld1q_u8(sb0.qs.as_ptr().add(32));
                            let u3 = vld1q_u8(sb0.qs.as_ptr().add(48));
                            let la0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u0, m4b)), s8b);
                            let la2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u0)), s8b);
                            let la1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u1, m4b)), s8b);
                            let la3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u1)), s8b);
                            let lb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u2, m4b)), s8b);
                            let lb2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u2)), s8b);
                            let lb1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u3, m4b)), s8b);
                            let lb3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u3)), s8b);

                            let v0 = vld1q_u8(sb1.qs.as_ptr());
                            let v1 = vld1q_u8(sb1.qs.as_ptr().add(16));
                            let v2 = vld1q_u8(sb1.qs.as_ptr().add(32));
                            let v3 = vld1q_u8(sb1.qs.as_ptr().add(48));
                            let lc0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
                            let lc2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), s8b);
                            let lc1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4b)), s8b);
                            let lc3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), s8b);
                            let ld0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v2, m4b)), s8b);
                            let ld2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v2)), s8b);
                            let ld1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v3, m4b)), s8b);
                            let ld3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v3)), s8b);

                            let d00 = f16_to_f32_hw(sb0.d[0]);
                            let d01 = f16_to_f32_hw(sb0.d[1]);
                            let d02 = f16_to_f32_hw(sb0.d[2]);
                            let d03 = f16_to_f32_hw(sb0.d[3]);
                            let wd0_a = vld1q_f32([d00, d00, d01, d01].as_ptr());
                            let wd0_b = vld1q_f32([d02, d02, d03, d03].as_ptr());

                            let d10 = f16_to_f32_hw(sb1.d[0]);
                            let d11 = f16_to_f32_hw(sb1.d[1]);
                            let d12 = f16_to_f32_hw(sb1.d[2]);
                            let d13 = f16_to_f32_hw(sb1.d[3]);
                            let wd1_a = vld1q_f32([d10, d10, d11, d11].as_ptr());
                            let wd1_b = vld1q_f32([d12, d12, d13, d13].as_ptr());

                            macro_rules! mma8 {
                                ($p:expr, $s0a:expr, $s0b:expr, $s1a:expr, $s1b:expr) => {{
                                    let pk = packed_pairs.get_unchecked(ki * 4 + $p);
                                    let rr0 = vld1q_s8(pk.rr[0].as_ptr());
                                    let rr1 = vld1q_s8(pk.rr[1].as_ptr());
                                    let rr2 = vld1q_s8(pk.rr[2].as_ptr());
                                    let rr3 = vld1q_s8(pk.rr[3].as_ptr());
                                    let yd = vld1q_f32(pk.yd.as_ptr());

                                    let a0 = vmmlaq_s32(
                                        vmmlaq_s32(
                                            vmmlaq_s32(vmmlaq_s32(zero, la0, rr0), la1, rr1),
                                            la2,
                                            rr2,
                                        ),
                                        la3,
                                        rr3,
                                    );
                                    $s0a = vfmaq_f32($s0a, vcvtq_f32_s32(a0), vmulq_f32(wd0_a, yd));

                                    let b0 = vmmlaq_s32(
                                        vmmlaq_s32(
                                            vmmlaq_s32(vmmlaq_s32(zero, lb0, rr0), lb1, rr1),
                                            lb2,
                                            rr2,
                                        ),
                                        lb3,
                                        rr3,
                                    );
                                    $s0b = vfmaq_f32($s0b, vcvtq_f32_s32(b0), vmulq_f32(wd0_b, yd));

                                    let c0 = vmmlaq_s32(
                                        vmmlaq_s32(
                                            vmmlaq_s32(vmmlaq_s32(zero, lc0, rr0), lc1, rr1),
                                            lc2,
                                            rr2,
                                        ),
                                        lc3,
                                        rr3,
                                    );
                                    $s1a = vfmaq_f32($s1a, vcvtq_f32_s32(c0), vmulq_f32(wd1_a, yd));

                                    let d0 = vmmlaq_s32(
                                        vmmlaq_s32(
                                            vmmlaq_s32(vmmlaq_s32(zero, ld0, rr0), ld1, rr1),
                                            ld2,
                                            rr2,
                                        ),
                                        ld3,
                                        rr3,
                                    );
                                    $s1b = vfmaq_f32($s1b, vcvtq_f32_s32(d0), vmulq_f32(wd1_b, yd));
                                }};
                            }

                            mma8!(0, g0_a0, g0_b0, g1_a0, g1_b0);
                            mma8!(1, g0_a1, g0_b1, g1_a1, g1_b1);
                            mma8!(2, g0_a2, g0_b2, g1_a2, g1_b2);
                            mma8!(3, g0_a3, g0_b3, g1_a3, g1_b3);
                        }

                        vst1q_f32(ap0, g0_a0);
                        vst1q_f32(ap0.add(4), g0_b0);
                        vst1q_f32(ap0.add(8), g0_a1);
                        vst1q_f32(ap0.add(12), g0_b1);
                        vst1q_f32(ap0.add(16), g0_a2);
                        vst1q_f32(ap0.add(20), g0_b2);
                        vst1q_f32(ap0.add(24), g0_a3);
                        vst1q_f32(ap0.add(28), g0_b3);

                        vst1q_f32(ap1, g1_a0);
                        vst1q_f32(ap1.add(4), g1_b0);
                        vst1q_f32(ap1.add(8), g1_a1);
                        vst1q_f32(ap1.add(12), g1_b1);
                        vst1q_f32(ap1.add(16), g1_a2);
                        vst1q_f32(ap1.add(20), g1_b2);
                        vst1q_f32(ap1.add(24), g1_a3);
                        vst1q_f32(ap1.add(28), g1_b3);
                        lg += 2;
                    }

                    if lg < g_cnt {
                        let wg0 = &w[(rg + lg) * cols_blocks + kb..(rg + lg) * cols_blocks + kend];
                        let ap0 = acc_buf[lg].as_mut_ptr();
                        let mut g0_a0 = vld1q_f32(ap0);
                        let mut g0_b0 = vld1q_f32(ap0.add(4));
                        let mut g0_a1 = vld1q_f32(ap0.add(8));
                        let mut g0_b1 = vld1q_f32(ap0.add(12));
                        let mut g0_a2 = vld1q_f32(ap0.add(16));
                        let mut g0_b2 = vld1q_f32(ap0.add(20));
                        let mut g0_a3 = vld1q_f32(ap0.add(24));
                        let mut g0_b3 = vld1q_f32(ap0.add(28));

                        for ki in 0..klen {
                            let sb0 = wg0.get_unchecked(ki);
                            let u0 = vld1q_u8(sb0.qs.as_ptr());
                            let u1 = vld1q_u8(sb0.qs.as_ptr().add(16));
                            let u2 = vld1q_u8(sb0.qs.as_ptr().add(32));
                            let u3 = vld1q_u8(sb0.qs.as_ptr().add(48));
                            let la0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u0, m4b)), s8b);
                            let la2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u0)), s8b);
                            let la1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u1, m4b)), s8b);
                            let la3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u1)), s8b);
                            let lb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u2, m4b)), s8b);
                            let lb2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u2)), s8b);
                            let lb1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u3, m4b)), s8b);
                            let lb3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u3)), s8b);

                            let d00 = f16_to_f32_hw(sb0.d[0]);
                            let d01 = f16_to_f32_hw(sb0.d[1]);
                            let d02 = f16_to_f32_hw(sb0.d[2]);
                            let d03 = f16_to_f32_hw(sb0.d[3]);
                            let wd0_a = vld1q_f32([d00, d00, d01, d01].as_ptr());
                            let wd0_b = vld1q_f32([d02, d02, d03, d03].as_ptr());

                            macro_rules! mma4 {
                                ($p:expr, $sa:expr, $sb:expr) => {{
                                    let pk = packed_pairs.get_unchecked(ki * 4 + $p);
                                    let rr0 = vld1q_s8(pk.rr[0].as_ptr());
                                    let rr1 = vld1q_s8(pk.rr[1].as_ptr());
                                    let rr2 = vld1q_s8(pk.rr[2].as_ptr());
                                    let rr3 = vld1q_s8(pk.rr[3].as_ptr());
                                    let yd = vld1q_f32(pk.yd.as_ptr());
                                    let a0 = vmmlaq_s32(
                                        vmmlaq_s32(
                                            vmmlaq_s32(vmmlaq_s32(zero, la0, rr0), la1, rr1),
                                            la2,
                                            rr2,
                                        ),
                                        la3,
                                        rr3,
                                    );
                                    let b0 = vmmlaq_s32(
                                        vmmlaq_s32(
                                            vmmlaq_s32(vmmlaq_s32(zero, lb0, rr0), lb1, rr1),
                                            lb2,
                                            rr2,
                                        ),
                                        lb3,
                                        rr3,
                                    );
                                    $sa = vfmaq_f32($sa, vcvtq_f32_s32(a0), vmulq_f32(wd0_a, yd));
                                    $sb = vfmaq_f32($sb, vcvtq_f32_s32(b0), vmulq_f32(wd0_b, yd));
                                }};
                            }
                            mma4!(0, g0_a0, g0_b0);
                            mma4!(1, g0_a1, g0_b1);
                            mma4!(2, g0_a2, g0_b2);
                            mma4!(3, g0_a3, g0_b3);
                        }

                        vst1q_f32(ap0, g0_a0);
                        vst1q_f32(ap0.add(4), g0_b0);
                        vst1q_f32(ap0.add(8), g0_a1);
                        vst1q_f32(ap0.add(12), g0_b1);
                        vst1q_f32(ap0.add(16), g0_a2);
                        vst1q_f32(ap0.add(20), g0_b2);
                        vst1q_f32(ap0.add(24), g0_a3);
                        vst1q_f32(ap0.add(28), g0_b3);
                    }

                    kb = kend;
                }

                // Scatter `acc_buf[0..g_cnt]` to `out`
                for (lg, ab) in acc_buf.iter().enumerate().take(g_cnt) {
                    let r = (rg + lg) * 4;
                    for p in 0..4 {
                        let tp = t + p * 2;
                        let base = p * 8;
                        *out.get_unchecked_mut(tp * rows + r) = ab[base];
                        *out.get_unchecked_mut((tp + 1) * rows + r) = ab[base + 1];
                        *out.get_unchecked_mut(tp * rows + r + 1) = ab[base + 2];
                        *out.get_unchecked_mut((tp + 1) * rows + r + 1) = ab[base + 3];
                        *out.get_unchecked_mut(tp * rows + r + 2) = ab[base + 4];
                        *out.get_unchecked_mut((tp + 1) * rows + r + 2) = ab[base + 5];
                        *out.get_unchecked_mut(tp * rows + r + 3) = ab[base + 6];
                        *out.get_unchecked_mut((tp + 1) * rows + r + 3) = ab[base + 7];
                    }
                }
                rg = rg_end;
            }
            t += 8;
        }

        // Remaining `< 8` tokens (speculative verification windows `n_tokens = 2..7`)
        while t + 2 <= n_tokens {
            let mut rg = 0usize;
            while rg < n_groups {
                let rg_end = (rg + 16).min(n_groups);
                let g_cnt = rg_end - rg;
                let mut acc_buf = [[0.0f32; 8]; 16];

                let mut kb = 0usize;
                while kb < cols_blocks {
                    let kend = (kb + K_TILE).min(cols_blocks);
                    let klen = kend - kb;
                    for ki in 0..klen {
                        let ib = kb + ki;
                        let by0 = x.get_unchecked(t * cols_blocks + ib);
                        let by1 = x.get_unchecked((t + 1) * cols_blocks + ib);
                        let y0l = vreinterpretq_s64_s8(vld1q_s8(by0.qs.as_ptr()));
                        let y0h = vreinterpretq_s64_s8(vld1q_s8(by0.qs.as_ptr().add(16)));
                        let y1l = vreinterpretq_s64_s8(vld1q_s8(by1.qs.as_ptr()));
                        let y1h = vreinterpretq_s64_s8(vld1q_s8(by1.qs.as_ptr().add(16)));
                        let dst = packed_pairs.get_unchecked_mut(ki);
                        vst1q_s8(
                            dst.rr[0].as_mut_ptr(),
                            vreinterpretq_s8_s64(vzip1q_s64(y0l, y1l)),
                        );
                        vst1q_s8(
                            dst.rr[1].as_mut_ptr(),
                            vreinterpretq_s8_s64(vzip2q_s64(y0l, y1l)),
                        );
                        vst1q_s8(
                            dst.rr[2].as_mut_ptr(),
                            vreinterpretq_s8_s64(vzip1q_s64(y0h, y1h)),
                        );
                        vst1q_s8(
                            dst.rr[3].as_mut_ptr(),
                            vreinterpretq_s8_s64(vzip2q_s64(y0h, y1h)),
                        );
                        dst.yd = [
                            f16_to_f32_hw(by0.d),
                            f16_to_f32_hw(by1.d),
                            f16_to_f32_hw(by0.d),
                            f16_to_f32_hw(by1.d),
                        ];
                    }

                    for (lg, ab) in acc_buf.iter_mut().enumerate().take(g_cnt) {
                        let wg0 = &w[(rg + lg) * cols_blocks + kb..(rg + lg) * cols_blocks + kend];
                        let mut sa = vld1q_f32(ab.as_ptr());
                        let mut sb = vld1q_f32(ab.as_ptr().add(4));
                        for ki in 0..klen {
                            let sb0 = wg0.get_unchecked(ki);
                            let pk = packed_pairs.get_unchecked(ki);
                            let u0 = vld1q_u8(sb0.qs.as_ptr());
                            let u1 = vld1q_u8(sb0.qs.as_ptr().add(16));
                            let u2 = vld1q_u8(sb0.qs.as_ptr().add(32));
                            let u3 = vld1q_u8(sb0.qs.as_ptr().add(48));
                            let la0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u0, m4b)), s8b);
                            let la2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u0)), s8b);
                            let la1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u1, m4b)), s8b);
                            let la3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u1)), s8b);
                            let lb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u2, m4b)), s8b);
                            let lb2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u2)), s8b);
                            let lb1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(u3, m4b)), s8b);
                            let lb3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(u3)), s8b);

                            let rr0 = vld1q_s8(pk.rr[0].as_ptr());
                            let rr1 = vld1q_s8(pk.rr[1].as_ptr());
                            let rr2 = vld1q_s8(pk.rr[2].as_ptr());
                            let rr3 = vld1q_s8(pk.rr[3].as_ptr());
                            let yd = vld1q_f32(pk.yd.as_ptr());
                            let d00 = f16_to_f32_hw(sb0.d[0]);
                            let d01 = f16_to_f32_hw(sb0.d[1]);
                            let d02 = f16_to_f32_hw(sb0.d[2]);
                            let d03 = f16_to_f32_hw(sb0.d[3]);
                            let wd_a = vld1q_f32([d00, d00, d01, d01].as_ptr());
                            let wd_b = vld1q_f32([d02, d02, d03, d03].as_ptr());

                            let a0 = vmmlaq_s32(
                                vmmlaq_s32(
                                    vmmlaq_s32(vmmlaq_s32(zero, la0, rr0), la1, rr1),
                                    la2,
                                    rr2,
                                ),
                                la3,
                                rr3,
                            );
                            let b0 = vmmlaq_s32(
                                vmmlaq_s32(
                                    vmmlaq_s32(vmmlaq_s32(zero, lb0, rr0), lb1, rr1),
                                    lb2,
                                    rr2,
                                ),
                                lb3,
                                rr3,
                            );
                            sa = vfmaq_f32(sa, vcvtq_f32_s32(a0), vmulq_f32(wd_a, yd));
                            sb = vfmaq_f32(sb, vcvtq_f32_s32(b0), vmulq_f32(wd_b, yd));
                        }
                        vst1q_f32(ab.as_mut_ptr(), sa);
                        vst1q_f32(ab.as_mut_ptr().add(4), sb);
                    }
                    kb = kend;
                }

                for (lg, ab) in acc_buf.iter().enumerate().take(g_cnt) {
                    let r = (rg + lg) * 4;
                    *out.get_unchecked_mut(t * rows + r) = ab[0];
                    *out.get_unchecked_mut((t + 1) * rows + r) = ab[1];
                    *out.get_unchecked_mut(t * rows + r + 1) = ab[2];
                    *out.get_unchecked_mut((t + 1) * rows + r + 1) = ab[3];
                    *out.get_unchecked_mut(t * rows + r + 2) = ab[4];
                    *out.get_unchecked_mut((t + 1) * rows + r + 2) = ab[5];
                    *out.get_unchecked_mut(t * rows + r + 3) = ab[6];
                    *out.get_unchecked_mut((t + 1) * rows + r + 3) = ab[7];
                }
                rg = rg_end;
            }
            t += 2;
        }

        if t < n_tokens {
            gemv_q4_0x4_q8_0(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}
