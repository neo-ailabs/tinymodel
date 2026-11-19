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

/// Multi-token `Q4_0 x Q8_0` GEMM using ARMv8.6 `i8mm` 2x2 `smmla` tiles.
///
/// For `n_tokens == 1`, delegates to `dotprod::gemv_q4_0_q8_0` (4-row unrolled).
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
        // SAFETY: `dotprod` is enabled and dimensions verified above.
        unsafe {
            dotprod::gemv_q4_0_q8_0(w, x, out, rows, cols_blocks);
        }
        return;
    }

    // SAFETY: All row and token indices are strictly bounded.
    unsafe {
        let mut t = 0usize;
        while t + 2 <= n_tokens {
            let x0 = &x[t * cols_blocks..(t + 1) * cols_blocks];
            let x1 = &x[(t + 1) * cols_blocks..(t + 2) * cols_blocks];

            let mut r = 0usize;
            while r + 2 <= rows {
                let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
                let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
                let tile = vec_dot_2x2_q4_0_q8_0(w0, w1, x0, x1);
                *out.get_unchecked_mut(t * rows + r) = tile[0];
                *out.get_unchecked_mut((t + 1) * rows + r) = tile[1];
                *out.get_unchecked_mut(t * rows + (r + 1)) = tile[2];
                *out.get_unchecked_mut((t + 1) * rows + (r + 1)) = tile[3];
                r += 2;
            }

            if r < rows {
                let wr = &w[r * cols_blocks..(r + 1) * cols_blocks];
                *out.get_unchecked_mut(t * rows + r) = dotprod::vec_dot_q4_0_q8_0(wr, x0);
                *out.get_unchecked_mut((t + 1) * rows + r) = dotprod::vec_dot_q4_0_q8_0(wr, x1);
            }

            t += 2;
        }

        if t < n_tokens {
            let xt = &x[t * cols_blocks..(t + 1) * cols_blocks];
            let ot = &mut out[t * rows..(t + 1) * rows];
            dotprod::gemv_q4_0_q8_0(w, xt, ot, rows, cols_blocks);
        }
    }
}
