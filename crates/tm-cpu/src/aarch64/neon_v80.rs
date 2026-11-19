//! Pure ARMv8.0 NEON kernels for `Q4_0 x Q8_0` (no `dotprod`, no `i8mm`, no FP16 arithmetic).
//!
//! Targets Raspberry Pi Zero 2W (Cortex-A53) and Raspberry Pi 4 (Cortex-A72).
//! Uses widening `vmull_s8` + `vmlal_s8` + `vpaddlq_s16` to accumulate 32 int8
//! products per block into `int32x4_t` without requiring `sdot`.

use core::arch::aarch64::*;
use tm_core::{BlockQ4_0, BlockQ8_0};

use super::asm::f16_to_f32_hw;

/// Computes the 4-lane `int32x4_t` partial dot product of one `BlockQ4_0` and one `BlockQ8_0`
/// using only base ARMv8.0 NEON instructions.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn block_dot_v80(
    bx: &BlockQ4_0,
    by: &BlockQ8_0,
    m4b: uint8x16_t,
    s8b: int8x16_t,
) -> int32x4_t {
    // SAFETY: `bx.qs` has 16 bytes, `by.qs` has 32 bytes; all loads are in-bounds
    // and NEON is enabled on this function.
    unsafe {
        let q = vld1q_u8(bx.qs.as_ptr());
        let xl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4b)), s8b);
        let xh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), s8b);

        let yl = vld1q_s8(by.qs.as_ptr());
        let yh = vld1q_s8(by.qs.as_ptr().add(16));

        // Each 16-byte dot product splits into low 8 and high 8 signed bytes.
        // Max magnitude per i16 lane: 2 * 8 * 127 = 2032 (well within i16 range [-32768, 32767]).
        let p0 = vmlal_s8(
            vmull_s8(vget_low_s8(xl), vget_low_s8(yl)),
            vget_low_s8(xh),
            vget_low_s8(yh),
        );
        let p1 = vmlal_s8(
            vmull_s8(vget_high_s8(xl), vget_high_s8(yl)),
            vget_high_s8(xh),
            vget_high_s8(yh),
        );

        vpadalq_s16(vpaddlq_s16(p0), p1)
    }
}

/// Pure ARMv8.0 NEON `vec_dot_q4_0_q8_0`.
///
/// # Safety
/// Caller must ensure `x.len() == y.len()` and NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut ib = 0usize;

    // SAFETY: All pointer arithmetic stays strictly within `0..nb`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);

        while ib + 1 < nb {
            let x0 = x.get_unchecked(ib);
            let x1 = x.get_unchecked(ib + 1);
            let y0 = y.get_unchecked(ib);
            let y1 = y.get_unchecked(ib + 1);

            let p0 = block_dot_v80(x0, y0, m4b, s8b);
            let p1 = block_dot_v80(x1, y1, m4b, s8b);

            let d0 = f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d);
            let d1 = f16_to_f32_hw(x1.d) * f16_to_f32_hw(y1.d);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), d0);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), d1);

            ib += 2;
        }

        let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

        if ib < nb {
            let x0 = x.get_unchecked(ib);
            let y0 = y.get_unchecked(ib);
            let p0 = block_dot_v80(x0, y0, m4b, s8b);
            let d0 = f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d);
            sumf += (vaddvq_s32(p0) as f32) * d0;
        }

        sumf
    }
}

/// 4-row unrolled ARMv8.0 NEON GEMV: loads each `x` block once across 4 weight rows.
///
/// # Safety
/// Caller must ensure `w.len() == rows * cols_blocks`, `x.len() == cols_blocks`,
/// `out.len() == rows`.
#[target_feature(enable = "neon")]
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
    // SAFETY: Row and block indices are bounded by `rows` and `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let dy = f16_to_f32_hw(by.d);

                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);

                let p0 = block_dot_v80(b0, by, m4b, s8b);
                let p1 = block_dot_v80(b1, by, m4b, s8b);
                let p2 = block_dot_v80(b2, by, m4b, s8b);
                let p3 = block_dot_v80(b3, by, m4b, s8b);

                acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), f16_to_f32_hw(b0.d) * dy);
                acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), f16_to_f32_hw(b1.d) * dy);
                acc2 = vmlaq_n_f32(acc2, vcvtq_f32_s32(p2), f16_to_f32_hw(b2.d) * dy);
                acc3 = vmlaq_n_f32(acc3, vcvtq_f32_s32(p3), f16_to_f32_hw(b3.d) * dy);
            }

            *out.get_unchecked_mut(r) = vaddvq_f32(acc0);
            *out.get_unchecked_mut(r + 1) = vaddvq_f32(acc1);
            *out.get_unchecked_mut(r + 2) = vaddvq_f32(acc2);
            *out.get_unchecked_mut(r + 3) = vaddvq_f32(acc3);
            r += 4;
        }

        while r < rows {
            let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q4_0_q8_0(row, x);
            r += 1;
        }
    }
}

/// Multi-token GEMM for ARMv8.0 NEON.
///
/// # Safety
/// Caller must ensure buffer lengths match dimensions.
#[target_feature(enable = "neon")]
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

    // SAFETY: Slices are verified above; calling `gemv_q4_0_q8_0` guarantees
    // exact batch-width invariance (1 token vs N tokens).
    unsafe {
        for t in 0..n_tokens {
            let xt = &x[t * cols_blocks..(t + 1) * cols_blocks];
            let ot = &mut out[t * rows..(t + 1) * rows];
            gemv_q4_0_q8_0(w, xt, ot, rows, cols_blocks);
        }
    }
}
