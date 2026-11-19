//! Pure ARMv8.0 NEON kernels for `Q4_0 x Q8_0` (no `dotprod`, no `i8mm`, no FP16 arithmetic).
//!
//! Targets Raspberry Pi Zero 2W (Cortex-A53) and Raspberry Pi 4 (Cortex-A72).
//! Uses widening `vmull_s8` + `vmlal_s8` + `vpaddlq_s16` to accumulate 32 int8
//! products per block into `int32x4_t` without requiring `sdot`.

use core::arch::aarch64::*;
use tm_core::{BlockQ4_0, BlockQ8_0, BlockQ8_1, QK8_0, QK8_1};

use super::asm::{cvt_f16x8_to_f32x4x2, cvt_f32x4x2_to_f16x8, f16_to_f32_hw, f32_to_f16_hw};

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

/// Base ARMv8.0 NEON `f32 x f16` dot product using hardware `fcvt`/`fcvt2` + `fmla`.
///
/// # Safety
/// Caller must ensure NEON is available and `q.len() == k_f16.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_f16(q: &[f32], k_f16: &[u16]) -> f32 {
    debug_assert_eq!(q.len(), k_f16.len());
    let n = q.len();
    let mut i = 0usize;

    // SAFETY: All pointer offsets are strictly bounded by `n`.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let qp = q.as_ptr();
        let kp = k_f16.as_ptr();

        while i + 8 <= n {
            let kh = vld1q_u16(kp.add(i));
            let (kl, khi) = cvt_f16x8_to_f32x4x2(kh);
            let ql = vld1q_f32(qp.add(i));
            let qhi = vld1q_f32(qp.add(i + 4));
            acc0 = vfmaq_f32(acc0, ql, kl);
            acc1 = vfmaq_f32(acc1, qhi, khi);
            i += 8;
        }

        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < n {
            sum += *qp.add(i) * f16_to_f32_hw(*kp.add(i));
            i += 1;
        }
        sum
    }
}

/// Base ARMv8.0 NEON `out[d] += w * f16_to_f32(v_f16[d])` using hardware `fcvt`/`fcvt2` + `fmla`.
///
/// # Safety
/// Caller must ensure NEON is available and `out.len() == v_f16.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn axpy_f32_f16(out: &mut [f32], w: f32, v_f16: &[u16]) {
    debug_assert_eq!(out.len(), v_f16.len());
    let n = out.len();
    let mut i = 0usize;

    // SAFETY: All pointer offsets are strictly bounded by `n`.
    unsafe {
        let vw = vdupq_n_f32(w);
        let op = out.as_mut_ptr();
        let vp = v_f16.as_ptr();

        while i + 8 <= n {
            let vh = vld1q_u16(vp.add(i));
            let (vl, vhi) = cvt_f16x8_to_f32x4x2(vh);
            let ol = vld1q_f32(op.add(i));
            let ohi = vld1q_f32(op.add(i + 4));
            vst1q_f32(op.add(i), vfmaq_f32(ol, vw, vl));
            vst1q_f32(op.add(i + 4), vfmaq_f32(ohi, vw, vhi));
            i += 8;
        }

        while i < n {
            *op.add(i) += w * f16_to_f32_hw(*vp.add(i));
            i += 1;
        }
    }
}

/// Base ARMv8.0 NEON `f32` -> `f16` (`u16`) slice conversion via `fcvtn` / `fcvtn2`.
///
/// # Safety
/// Caller must ensure NEON is available and `src.len() == dst.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let mut i = 0usize;

    // SAFETY: All pointer offsets are strictly bounded by `n`.
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        while i + 8 <= n {
            let lo = vld1q_f32(sp.add(i));
            let hi = vld1q_f32(sp.add(i + 4));
            vst1q_u16(dp.add(i), cvt_f32x4x2_to_f16x8(lo, hi));
            i += 8;
        }
        while i < n {
            *dp.add(i) = f32_to_f16_hw(*sp.add(i));
            i += 1;
        }
    }
}

/// Base ARMv8.0 NEON `f32 -> BlockQ8_0` activation quantization (bit-exact with `tm_core`).
///
/// # Safety
/// Caller must ensure NEON is available, `src.len() == dst.len() * 32`.
#[target_feature(enable = "neon")]
pub unsafe fn quantize_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    assert_eq!(src.len(), dst.len() * QK8_0);
    let nb = dst.len();

    // SAFETY: Each block reads 32 `f32`s and writes 32 `i8`s strictly in bounds.
    unsafe {
        let min_q = vdupq_n_s8(-127);
        let sp_base = src.as_ptr();
        for (ib, block) in dst.iter_mut().enumerate() {
            let sp = sp_base.add(ib * 32);
            let v0 = vld1q_f32(sp);
            let v1 = vld1q_f32(sp.add(4));
            let v2 = vld1q_f32(sp.add(8));
            let v3 = vld1q_f32(sp.add(12));
            let v4 = vld1q_f32(sp.add(16));
            let v5 = vld1q_f32(sp.add(20));
            let v6 = vld1q_f32(sp.add(24));
            let v7 = vld1q_f32(sp.add(28));

            let m01 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
            let m23 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
            let m45 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
            let m67 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
            let amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m01, m23), vmaxq_f32(m45, m67)));

            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            block.d = f32_to_f16_hw(d);
            let vid = vdupq_n_f32(id);

            let i0 = vcvtaq_s32_f32(vmulq_f32(v0, vid));
            let i1 = vcvtaq_s32_f32(vmulq_f32(v1, vid));
            let i2 = vcvtaq_s32_f32(vmulq_f32(v2, vid));
            let i3 = vcvtaq_s32_f32(vmulq_f32(v3, vid));
            let i4 = vcvtaq_s32_f32(vmulq_f32(v4, vid));
            let i5 = vcvtaq_s32_f32(vmulq_f32(v5, vid));
            let i6 = vcvtaq_s32_f32(vmulq_f32(v6, vid));
            let i7 = vcvtaq_s32_f32(vmulq_f32(v7, vid));

            let s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            let s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            let s45 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
            let s67 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));

            let q_lo = vmaxq_s8(vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)), min_q);
            let q_hi = vmaxq_s8(vcombine_s8(vqmovn_s16(s45), vqmovn_s16(s67)), min_q);
            vst1q_s8(block.qs.as_mut_ptr(), q_lo);
            vst1q_s8(block.qs.as_mut_ptr().add(16), q_hi);
        }
        let _ = nb;
    }
}

/// Base ARMv8.0 NEON `f32 -> BlockQ8_1` activation quantization (bit-exact with `tm_core`).
///
/// # Safety
/// Caller must ensure NEON is available, `src.len() == dst.len() * 32`.
#[target_feature(enable = "neon")]
pub unsafe fn quantize_q8_1(src: &[f32], dst: &mut [BlockQ8_1]) {
    assert_eq!(src.len(), dst.len() * QK8_1);

    // SAFETY: Each block reads 32 `f32`s and writes 32 `i8`s strictly in bounds.
    unsafe {
        let min_q = vdupq_n_s8(-127);
        let sp_base = src.as_ptr();
        for (ib, block) in dst.iter_mut().enumerate() {
            let sp = sp_base.add(ib * 32);
            let v0 = vld1q_f32(sp);
            let v1 = vld1q_f32(sp.add(4));
            let v2 = vld1q_f32(sp.add(8));
            let v3 = vld1q_f32(sp.add(12));
            let v4 = vld1q_f32(sp.add(16));
            let v5 = vld1q_f32(sp.add(20));
            let v6 = vld1q_f32(sp.add(24));
            let v7 = vld1q_f32(sp.add(28));

            let m01 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
            let m23 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
            let m45 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
            let m67 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
            let amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m01, m23), vmaxq_f32(m45, m67)));

            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let vid = vdupq_n_f32(id);

            let i0 = vcvtaq_s32_f32(vmulq_f32(v0, vid));
            let i1 = vcvtaq_s32_f32(vmulq_f32(v1, vid));
            let i2 = vcvtaq_s32_f32(vmulq_f32(v2, vid));
            let i3 = vcvtaq_s32_f32(vmulq_f32(v3, vid));
            let i4 = vcvtaq_s32_f32(vmulq_f32(v4, vid));
            let i5 = vcvtaq_s32_f32(vmulq_f32(v5, vid));
            let i6 = vcvtaq_s32_f32(vmulq_f32(v6, vid));
            let i7 = vcvtaq_s32_f32(vmulq_f32(v7, vid));

            let s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            let s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            let s45 = vcombine_s16(vqmovn_s32(i4), vqmovn_s32(i5));
            let s67 = vcombine_s16(vqmovn_s32(i6), vqmovn_s32(i7));

            let q_lo = vmaxq_s8(vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)), min_q);
            let q_hi = vmaxq_s8(vcombine_s8(vqmovn_s16(s45), vqmovn_s16(s67)), min_q);
            vst1q_s8(block.qs.as_mut_ptr(), q_lo);
            vst1q_s8(block.qs.as_mut_ptr().add(16), q_hi);

            let sum16 = vpaddlq_s8(q_lo);
            let sum16 = vpadalq_s8(sum16, q_hi);
            let sum = vaddlvq_s16(sum16) as i32;

            block.d = f32_to_f16_hw(d);
            block.s = f32_to_f16_hw((sum as f32) * d);
        }
    }
}
