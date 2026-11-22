//! Pure ARMv8.0 NEON kernels for `Q4_0 x Q8_0` (no `dotprod`, no `i8mm`, no FP16 arithmetic).
//!
//! Targets Raspberry Pi Zero 2W (Cortex-A53) and Raspberry Pi 4 (Cortex-A72).
//! Uses widening `vmull_s8` + `vmlal_s8` + `vpaddlq_s16` to accumulate 32 int8
//! products per block into `int32x4_t` without requiring `sdot`.

use core::arch::aarch64::*;
use tm_core::{
    BlockQ4_0, BlockQ4_0x4, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, QK_K,
    QK4_0, QK8_0, QK8_1, unpack_q5k_scales_mins,
};

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

    // SAFETY: All pointer arithmetic stays strictly within `0..nb`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let mut sumf = 0.0f32;

        for ib in 0..nb {
            let x0 = x.get_unchecked(ib);
            let y0 = y.get_unchecked(ib);
            let p0 = block_dot_v80(x0, y0, m4b, s8b);
            let d0 = f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d);
            sumf = (vaddvq_s32(p0) as f32).mul_add(d0, sumf);
        }

        sumf
    }
}

/// 4-row unrolled ARMv8.0 NEON GEMV: loads each `x` block once across 4 weight rows
/// and reduces each block's 32 int8 products in exact integer arithmetic before `vfmaq_f32`,
/// matching `generic`, `dotprod`, `i8mm`, and `gemv_q4_0x4_q8_0` **bit-for-bit** (`to_bits()`).
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

            let mut acc4 = vdupq_n_f32(0.0);

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

                let sumi = [
                    vaddvq_s32(p0),
                    vaddvq_s32(p1),
                    vaddvq_s32(p2),
                    vaddvq_s32(p3),
                ];
                let scales = [
                    f16_to_f32_hw(b0.d) * dy,
                    f16_to_f32_hw(b1.d) * dy,
                    f16_to_f32_hw(b2.d) * dy,
                    f16_to_f32_hw(b3.d) * dy,
                ];
                acc4 = vfmaq_f32(
                    acc4,
                    vcvtq_f32_s32(vld1q_s32(sumi.as_ptr())),
                    vld1q_f32(scales.as_ptr()),
                );
            }

            vst1q_f32(out.as_mut_ptr().add(r), acc4);
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

/// 16-element `i8 x i8 -> int32x4_t` dot product on base ARMv8.0 NEON where `|x_i| <= 64`
/// (holds for `Q4_1` `[0,15]`, `Q5_K` `[0,31]`, and `Q6_K` `[-32,31]`, so `2 * 64 * 128 = 16384 <= i16::MAX`).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot16_s8_bounded_v80(x: int8x16_t, y: int8x16_t) -> int32x4_t {
    let p = vmlal_s8(
        vmull_s8(vget_low_s8(x), vget_low_s8(y)),
        vget_high_s8(x),
        vget_high_s8(y),
    );
    vpaddlq_s16(p)
}

/// 32-element `i8 x i8 -> int32x4_t` dot product on base ARMv8.0 NEON where `|x_i| <= 64`
/// (`4 * 64 * 128 = 32768` could hit `+32768` only if `|x|==64 && y==-128`; for `Q4_1` `<=15`
/// and `Q5_K` `<=31`, max is `4 * 31 * 128 = 15872 <= 32767`, completely overflow-free).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot32_s8_bounded_v80(
    xl: int8x16_t,
    xh: int8x16_t,
    yl: int8x16_t,
    yh: int8x16_t,
) -> int32x4_t {
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

/// 32-element full-range `i8 x i8 -> int32x4_t` dot product on base ARMv8.0 NEON (`Q8_0 x Q8_0`).
/// Widens each 8-lane product (`max (-128)*(-128) = +16384 <= 32767`) directly to `i32`
/// via `vpaddlq_s16` / `vpadalq_s16`, guaranteeing zero `i16` overflow for all `i8` inputs.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot32_s8_full_v80(
    xl: int8x16_t,
    xh: int8x16_t,
    yl: int8x16_t,
    yh: int8x16_t,
) -> int32x4_t {
    let p0 = vmull_s8(vget_low_s8(xl), vget_low_s8(yl));
    let p1 = vmull_s8(vget_high_s8(xl), vget_high_s8(yl));
    let p2 = vmull_s8(vget_low_s8(xh), vget_low_s8(yh));
    let p3 = vmull_s8(vget_high_s8(xh), vget_high_s8(yh));
    vaddq_s32(
        vpadalq_s16(vpaddlq_s16(p0), p1),
        vpadalq_s16(vpaddlq_s16(p2), p3),
    )
}

/// Pure ARMv8.0 NEON `vec_dot_q4_1_q8_1` (Raspberry Pi Zero 2W / Pi 4B).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q4_1_q8_1(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut ib = 0usize;

    // SAFETY: `ib` stays strictly within `0..nb`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);
        let mut summs0 = 0.0f32;
        let mut summs1 = 0.0f32;

        while ib + 1 < nb {
            let x0 = x.get_unchecked(ib);
            let x1 = x.get_unchecked(ib + 1);
            let y0 = y.get_unchecked(ib);
            let y1 = y.get_unchecked(ib + 1);

            let v0 = vld1q_u8(x0.qs.as_ptr());
            let v1 = vld1q_u8(x1.qs.as_ptr());
            let x0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
            let x0h = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0));
            let x1l = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
            let x1h = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1));

            let y0l = vld1q_s8(y0.qs.as_ptr());
            let y0h = vld1q_s8(y0.qs.as_ptr().add(16));
            let y1l = vld1q_s8(y1.qs.as_ptr());
            let y1h = vld1q_s8(y1.qs.as_ptr().add(16));

            let p0 = dot32_s8_bounded_v80(x0l, x0h, y0l, y0h);
            let p1 = dot32_s8_bounded_v80(x1l, x1h, y1l, y1h);

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(p0),
                f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d),
            );
            sumv1 = vmlaq_n_f32(
                sumv1,
                vcvtq_f32_s32(p1),
                f16_to_f32_hw(x1.d) * f16_to_f32_hw(y1.d),
            );
            summs0 += f16_to_f32_hw(x0.m) * f16_to_f32_hw(y0.s);
            summs1 += f16_to_f32_hw(x1.m) * f16_to_f32_hw(y1.s);

            ib += 2;
        }

        let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;
        if ib < nb {
            let x0 = x.get_unchecked(ib);
            let y0 = y.get_unchecked(ib);
            let v0 = vld1q_u8(x0.qs.as_ptr());
            let x0l = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
            let x0h = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0));
            let y0l = vld1q_s8(y0.qs.as_ptr());
            let y0h = vld1q_s8(y0.qs.as_ptr().add(16));
            let p0 = dot32_s8_bounded_v80(x0l, x0h, y0l, y0h);
            sumf += (vaddvq_s32(p0) as f32) * (f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d))
                + f16_to_f32_hw(x0.m) * f16_to_f32_hw(y0.s);
        }
        sumf
    }
}

/// 4-row unrolled ARMv8.0 NEON `gemv_q4_1_q8_1`.
///
/// # Safety
/// Caller must ensure NEON is available and buffer lengths match dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q4_1_q8_1(
    w: &[BlockQ4_1],
    x: &[BlockQ8_1],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: All row and block indices are bounded by `rows` and `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        while r + 4 <= rows {
            let r0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let r1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let r2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let r3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut ms0 = 0.0f32;
            let mut ms1 = 0.0f32;
            let mut ms2 = 0.0f32;
            let mut ms3 = 0.0f32;

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let yl = vld1q_s8(by.qs.as_ptr());
                let yh = vld1q_s8(by.qs.as_ptr().add(16));
                let dy = f16_to_f32_hw(by.d);
                let sy = f16_to_f32_hw(by.s);

                macro_rules! step_q41 {
                    ($row:expr, $acc:expr, $ms:expr) => {{
                        let bx = $row.get_unchecked(ib);
                        let v = vld1q_u8(bx.qs.as_ptr());
                        let xl = vreinterpretq_s8_u8(vandq_u8(v, m4b));
                        let xh = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v));
                        let p = dot32_s8_bounded_v80(xl, xh, yl, yh);
                        $acc = vmlaq_n_f32($acc, vcvtq_f32_s32(p), f16_to_f32_hw(bx.d) * dy);
                        $ms += f16_to_f32_hw(bx.m) * sy;
                    }};
                }

                step_q41!(r0, acc0, ms0);
                step_q41!(r1, acc1, ms1);
                step_q41!(r2, acc2, ms2);
                step_q41!(r3, acc3, ms3);
            }

            *out.get_unchecked_mut(r) = vaddvq_f32(acc0) + ms0;
            *out.get_unchecked_mut(r + 1) = vaddvq_f32(acc1) + ms1;
            *out.get_unchecked_mut(r + 2) = vaddvq_f32(acc2) + ms2;
            *out.get_unchecked_mut(r + 3) = vaddvq_f32(acc3) + ms3;
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q4_1_q8_1(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Pure ARMv8.0 NEON `vec_dot_q8_0_q8_0` (Raspberry Pi Zero 2W / Pi 4B).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q8_0_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut ib = 0usize;

    // SAFETY: `ib` stays strictly within `0..nb`.
    unsafe {
        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);

        while ib + 1 < nb {
            let x0 = x.get_unchecked(ib);
            let x1 = x.get_unchecked(ib + 1);
            let y0 = y.get_unchecked(ib);
            let y1 = y.get_unchecked(ib + 1);

            let p0 = dot32_s8_full_v80(
                vld1q_s8(x0.qs.as_ptr()),
                vld1q_s8(x0.qs.as_ptr().add(16)),
                vld1q_s8(y0.qs.as_ptr()),
                vld1q_s8(y0.qs.as_ptr().add(16)),
            );
            let p1 = dot32_s8_full_v80(
                vld1q_s8(x1.qs.as_ptr()),
                vld1q_s8(x1.qs.as_ptr().add(16)),
                vld1q_s8(y1.qs.as_ptr()),
                vld1q_s8(y1.qs.as_ptr().add(16)),
            );

            sumv0 = vmlaq_n_f32(
                sumv0,
                vcvtq_f32_s32(p0),
                f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d),
            );
            sumv1 = vmlaq_n_f32(
                sumv1,
                vcvtq_f32_s32(p1),
                f16_to_f32_hw(x1.d) * f16_to_f32_hw(y1.d),
            );
            ib += 2;
        }

        let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        if ib < nb {
            let x0 = x.get_unchecked(ib);
            let y0 = y.get_unchecked(ib);
            let p0 = dot32_s8_full_v80(
                vld1q_s8(x0.qs.as_ptr()),
                vld1q_s8(x0.qs.as_ptr().add(16)),
                vld1q_s8(y0.qs.as_ptr()),
                vld1q_s8(y0.qs.as_ptr().add(16)),
            );
            sumf += (vaddvq_s32(p0) as f32) * (f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d));
        }
        sumf
    }
}

/// 4-row unrolled ARMv8.0 NEON `gemv_q8_0_q8_0`: shares `x` loads across 4 weight rows.
/// Accelerates `SmolLM2` `output.weight` (`lm_head`) by ~3x on Cortex-A53 / Cortex-A72!
///
/// # Safety
/// Caller must ensure NEON is available and buffer lengths match dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q8_0_q8_0(
    w: &[BlockQ8_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: All row and block indices are bounded by `rows` and `cols_blocks`.
    unsafe {
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
                let yl = vld1q_s8(by.qs.as_ptr());
                let yh = vld1q_s8(by.qs.as_ptr().add(16));
                let dy = f16_to_f32_hw(by.d);

                macro_rules! step_q80 {
                    ($row:expr, $acc:expr) => {{
                        let bx = $row.get_unchecked(ib);
                        let p = dot32_s8_full_v80(
                            vld1q_s8(bx.qs.as_ptr()),
                            vld1q_s8(bx.qs.as_ptr().add(16)),
                            yl,
                            yh,
                        );
                        $acc = vmlaq_n_f32($acc, vcvtq_f32_s32(p), f16_to_f32_hw(bx.d) * dy);
                    }};
                }

                step_q80!(w0, acc0);
                step_q80!(w1, acc1);
                step_q80!(w2, acc2);
                step_q80!(w3, acc3);
            }

            *out.get_unchecked_mut(r) = vaddvq_f32(acc0);
            *out.get_unchecked_mut(r + 1) = vaddvq_f32(acc1);
            *out.get_unchecked_mut(r + 2) = vaddvq_f32(acc2);
            *out.get_unchecked_mut(r + 3) = vaddvq_f32(acc3);
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q8_0_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Pure ARMv8.0 NEON `vec_dot_q5_k_q8_k` (Raspberry Pi Zero 2W / Pi 4B).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q5_k_q8_k(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut sumf = 0.0f32;

    // SAFETY: All indices are bounded by `nb` and `QK_K = 256`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);
        let zero = vdupq_n_s32(0);

        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);

            let (sc, mins) = unpack_q5k_scales_mins(&bx.scales);
            let q8sums = vpaddq_s16(
                vld1q_s16(by.bsums.as_ptr()),
                vld1q_s16(by.bsums.as_ptr().add(8)),
            );
            let mins_v = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins.as_ptr())));
            let sumi_mins = vaddvq_s32(vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins_v)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins_v)),
            ));

            let mut qh0 = vld1q_u8(bx.qh.as_ptr());
            let mut qh1 = vld1q_u8(bx.qh.as_ptr().add(16));
            let mut vsum = zero;

            for chunk in 0..4 {
                let q5_ptr = bx.qs.as_ptr().add(chunk * 32);
                let q8_ptr = by.qs.as_ptr().add(chunk * 64);

                let ql0 = vld1q_u8(q5_ptr);
                let ql1 = vld1q_u8(q5_ptr.add(16));

                let h_lo0 = vshlq_n_u8::<4>(vandq_u8(qh0, mone));
                let h_lo1 = vshlq_n_u8::<4>(vandq_u8(qh1, mone));
                let h_hi0 = vshlq_n_u8::<3>(vandq_u8(qh0, mtwo));
                let h_hi1 = vshlq_n_u8::<3>(vandq_u8(qh1, mtwo));
                qh0 = vshrq_n_u8::<2>(qh0);
                qh1 = vshrq_n_u8::<2>(qh1);

                let q0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), h_lo0));
                let q1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), h_lo1));
                let q2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql0), h_hi0));
                let q3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql1), h_hi1));

                let d_lo = dot32_s8_bounded_v80(q0, q1, vld1q_s8(q8_ptr), vld1q_s8(q8_ptr.add(16)));
                let d_hi = dot32_s8_bounded_v80(
                    q2,
                    q3,
                    vld1q_s8(q8_ptr.add(32)),
                    vld1q_s8(q8_ptr.add(48)),
                );

                vsum = vmlaq_n_s32(vsum, d_lo, *sc.get_unchecked(2 * chunk) as i32);
                vsum = vmlaq_n_s32(vsum, d_hi, *sc.get_unchecked(2 * chunk + 1) as i32);
            }

            let sumi = vaddvq_s32(vsum);
            let d = f16_to_f32_hw(bx.d) * by.d;
            let dmin = f16_to_f32_hw(bx.dmin) * by.d;
            sumf += d * (sumi as f32) - dmin * (sumi_mins as f32);
        }
    }
    sumf
}

/// 2-row unrolled ARMv8.0 NEON `gemv_q5_k_q8_k`.
///
/// # Safety
/// Caller must ensure NEON is available and buffer lengths match dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q5_k_q8_k(
    w: &[BlockQ5K],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: All row and block indices are bounded by `rows` and `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);
        let zero = vdupq_n_s32(0);

        while r + 2 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let bx0 = w0.get_unchecked(ib);
                let bx1 = w1.get_unchecked(ib);

                let q8sums = vpaddq_s16(
                    vld1q_s16(by.bsums.as_ptr()),
                    vld1q_s16(by.bsums.as_ptr().add(8)),
                );
                let (sc0, mins0) = unpack_q5k_scales_mins(&bx0.scales);
                let (sc1, mins1) = unpack_q5k_scales_mins(&bx1.scales);

                let m0_v = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins0.as_ptr())));
                let m1_v = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mins1.as_ptr())));
                let sumi_mins0 = vaddvq_s32(vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums), vget_low_s16(m0_v)),
                    vmull_s16(vget_high_s16(q8sums), vget_high_s16(m0_v)),
                ));
                let sumi_mins1 = vaddvq_s32(vaddq_s32(
                    vmull_s16(vget_low_s16(q8sums), vget_low_s16(m1_v)),
                    vmull_s16(vget_high_s16(q8sums), vget_high_s16(m1_v)),
                ));

                let mut qh0_a = vld1q_u8(bx0.qh.as_ptr());
                let mut qh0_b = vld1q_u8(bx0.qh.as_ptr().add(16));
                let mut qh1_a = vld1q_u8(bx1.qh.as_ptr());
                let mut qh1_b = vld1q_u8(bx1.qh.as_ptr().add(16));

                let mut vsum0 = zero;
                let mut vsum1 = zero;

                for chunk in 0..4 {
                    let q8_ptr = by.qs.as_ptr().add(chunk * 64);
                    let y0 = vld1q_s8(q8_ptr);
                    let y1 = vld1q_s8(q8_ptr.add(16));
                    let y2 = vld1q_s8(q8_ptr.add(32));
                    let y3 = vld1q_s8(q8_ptr.add(48));

                    macro_rules! step_q5k_v80 {
                        ($bx:expr, $sc:expr, $qha:expr, $qhb:expr, $vsum:expr) => {{
                            let q5_ptr = $bx.qs.as_ptr().add(chunk * 32);
                            let ql0 = vld1q_u8(q5_ptr);
                            let ql1 = vld1q_u8(q5_ptr.add(16));

                            let h_lo0 = vshlq_n_u8::<4>(vandq_u8($qha, mone));
                            let h_lo1 = vshlq_n_u8::<4>(vandq_u8($qhb, mone));
                            let h_hi0 = vshlq_n_u8::<3>(vandq_u8($qha, mtwo));
                            let h_hi1 = vshlq_n_u8::<3>(vandq_u8($qhb, mtwo));
                            $qha = vshrq_n_u8::<2>($qha);
                            $qhb = vshrq_n_u8::<2>($qhb);

                            let q0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), h_lo0));
                            let q1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), h_lo1));
                            let q2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql0), h_hi0));
                            let q3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql1), h_hi1));

                            let d_lo = dot32_s8_bounded_v80(q0, q1, y0, y1);
                            let d_hi = dot32_s8_bounded_v80(q2, q3, y2, y3);
                            $vsum = vmlaq_n_s32($vsum, d_lo, *$sc.get_unchecked(2 * chunk) as i32);
                            $vsum =
                                vmlaq_n_s32($vsum, d_hi, *$sc.get_unchecked(2 * chunk + 1) as i32);
                        }};
                    }

                    step_q5k_v80!(bx0, sc0, qh0_a, qh0_b, vsum0);
                    step_q5k_v80!(bx1, sc1, qh1_a, qh1_b, vsum1);
                }

                let d0 = f16_to_f32_hw(bx0.d) * by.d;
                let dmin0 = f16_to_f32_hw(bx0.dmin) * by.d;
                sum0 += d0 * (vaddvq_s32(vsum0) as f32) - dmin0 * (sumi_mins0 as f32);

                let d1 = f16_to_f32_hw(bx1.d) * by.d;
                let dmin1 = f16_to_f32_hw(bx1.dmin) * by.d;
                sum1 += d1 * (vaddvq_s32(vsum1) as f32) - dmin1 * (sumi_mins1 as f32);
            }

            *out.get_unchecked_mut(r) = sum0;
            *out.get_unchecked_mut(r + 1) = sum1;
            r += 2;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q5_k_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Pure ARMv8.0 NEON `vec_dot_q6_k_q8_k` (Raspberry Pi Zero 2W / Pi 4B).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q6_k_q8_k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut sumf = 0.0f32;

    // SAFETY: All indices are bounded by `nb` and `QK_K = 256`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let m2b = vdupq_n_u8(0x03);
        let s32 = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);

        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let mut vsum = zero;

            for half in 0..2 {
                let ql_ptr = bx.ql.as_ptr().add(half * 64);
                let qh_ptr = bx.qh.as_ptr().add(half * 32);
                let q8_ptr = by.qs.as_ptr().add(half * 128);
                let sc_base = half * 8;

                for chunk in 0..2 {
                    let off = chunk * 16;
                    let ql0 = vld1q_u8(ql_ptr.add(off));
                    let ql1 = vld1q_u8(ql_ptr.add(32 + off));
                    let qh = vld1q_u8(qh_ptr.add(off));

                    let q0 = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(ql0, m4b),
                            vshlq_n_u8::<4>(vandq_u8(qh, m2b)),
                        )),
                        s32,
                    );
                    let q1 = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(ql1, m4b),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh), m2b)),
                        )),
                        s32,
                    );
                    let q2 = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8::<4>(ql0),
                            vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh), m2b)),
                        )),
                        s32,
                    );
                    let q3 = vsubq_s8(
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8::<4>(ql1),
                            vshlq_n_u8::<4>(vshrq_n_u8::<6>(qh)),
                        )),
                        s32,
                    );

                    let y0 = vld1q_s8(q8_ptr.add(off));
                    let y1 = vld1q_s8(q8_ptr.add(32 + off));
                    let y2 = vld1q_s8(q8_ptr.add(64 + off));
                    let y3 = vld1q_s8(q8_ptr.add(96 + off));

                    vsum = vmlaq_n_s32(
                        vsum,
                        dot16_s8_bounded_v80(q0, y0),
                        *bx.scales.get_unchecked(sc_base + chunk) as i32,
                    );
                    vsum = vmlaq_n_s32(
                        vsum,
                        dot16_s8_bounded_v80(q1, y1),
                        *bx.scales.get_unchecked(sc_base + 2 + chunk) as i32,
                    );
                    vsum = vmlaq_n_s32(
                        vsum,
                        dot16_s8_bounded_v80(q2, y2),
                        *bx.scales.get_unchecked(sc_base + 4 + chunk) as i32,
                    );
                    vsum = vmlaq_n_s32(
                        vsum,
                        dot16_s8_bounded_v80(q3, y3),
                        *bx.scales.get_unchecked(sc_base + 6 + chunk) as i32,
                    );
                }
            }

            sumf += f16_to_f32_hw(bx.d) * by.d * (vaddvq_s32(vsum) as f32);
        }
    }
    sumf
}

/// 2-row unrolled ARMv8.0 NEON `gemv_q6_k_q8_k`: shares `y` loads across 2 weight rows.
///
/// # Safety
/// Caller must ensure NEON is available and buffer lengths match dimensions.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q6_k_q8_k(
    w: &[BlockQ6K],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: All row and block indices are bounded by `rows` and `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let m2b = vdupq_n_u8(0x03);
        let s32 = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);

        while r + 2 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let bx0 = w0.get_unchecked(ib);
                let bx1 = w1.get_unchecked(ib);
                let mut vsum0 = zero;
                let mut vsum1 = zero;

                for half in 0..2 {
                    let q8_ptr = by.qs.as_ptr().add(half * 128);
                    let sc_base = half * 8;

                    for chunk in 0..2 {
                        let off = chunk * 16;
                        let y0 = vld1q_s8(q8_ptr.add(off));
                        let y1 = vld1q_s8(q8_ptr.add(32 + off));
                        let y2 = vld1q_s8(q8_ptr.add(64 + off));
                        let y3 = vld1q_s8(q8_ptr.add(96 + off));

                        macro_rules! step_q6k_v80 {
                            ($bx:expr, $vsum:expr) => {{
                                let ql_ptr = $bx.ql.as_ptr().add(half * 64 + off);
                                let qh_ptr = $bx.qh.as_ptr().add(half * 32 + off);
                                let ql0 = vld1q_u8(ql_ptr);
                                let ql1 = vld1q_u8(ql_ptr.add(32));
                                let qh = vld1q_u8(qh_ptr);

                                let q0 = vsubq_s8(
                                    vreinterpretq_s8_u8(vorrq_u8(
                                        vandq_u8(ql0, m4b),
                                        vshlq_n_u8::<4>(vandq_u8(qh, m2b)),
                                    )),
                                    s32,
                                );
                                let q1 = vsubq_s8(
                                    vreinterpretq_s8_u8(vorrq_u8(
                                        vandq_u8(ql1, m4b),
                                        vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh), m2b)),
                                    )),
                                    s32,
                                );
                                let q2 = vsubq_s8(
                                    vreinterpretq_s8_u8(vorrq_u8(
                                        vshrq_n_u8::<4>(ql0),
                                        vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh), m2b)),
                                    )),
                                    s32,
                                );
                                let q3 = vsubq_s8(
                                    vreinterpretq_s8_u8(vorrq_u8(
                                        vshrq_n_u8::<4>(ql1),
                                        vshlq_n_u8::<4>(vshrq_n_u8::<6>(qh)),
                                    )),
                                    s32,
                                );

                                $vsum = vmlaq_n_s32(
                                    $vsum,
                                    dot16_s8_bounded_v80(q0, y0),
                                    *$bx.scales.get_unchecked(sc_base + chunk) as i32,
                                );
                                $vsum = vmlaq_n_s32(
                                    $vsum,
                                    dot16_s8_bounded_v80(q1, y1),
                                    *$bx.scales.get_unchecked(sc_base + 2 + chunk) as i32,
                                );
                                $vsum = vmlaq_n_s32(
                                    $vsum,
                                    dot16_s8_bounded_v80(q2, y2),
                                    *$bx.scales.get_unchecked(sc_base + 4 + chunk) as i32,
                                );
                                $vsum = vmlaq_n_s32(
                                    $vsum,
                                    dot16_s8_bounded_v80(q3, y3),
                                    *$bx.scales.get_unchecked(sc_base + 6 + chunk) as i32,
                                );
                            }};
                        }

                        step_q6k_v80!(bx0, vsum0);
                        step_q6k_v80!(bx1, vsum1);
                    }
                }

                sum0 += f16_to_f32_hw(bx0.d) * by.d * (vaddvq_s32(vsum0) as f32);
                sum1 += f16_to_f32_hw(bx1.d) * by.d * (vaddvq_s32(vsum1) as f32);
            }

            *out.get_unchecked_mut(r) = sum0;
            *out.get_unchecked_mut(r + 1) = sum1;
            r += 2;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q6_k_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_pairwise_8wide_f32(acc0: float32x4_t, acc1: float32x4_t) -> f32 {
    let s4 = vaddq_f32(acc0, acc1);
    let p2 = vpaddq_f32(s4, s4);
    let p1 = vpaddq_f32(p2, p2);
    vgetq_lane_f32::<0>(p1)
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn sum_sq_8wide_neon(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks8 = n / 8;
    // SAFETY: Loop bounds `c * 8 + 8 <= n` are guaranteed by `chunks8 = n / 8`.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let xp = x.as_ptr();
        for c in 0..chunks8 {
            let off = c * 8;
            let v0 = vld1q_f32(xp.add(off));
            let v1 = vld1q_f32(xp.add(off + 4));
            acc0 = vfmaq_f32(acc0, v0, v0);
            acc1 = vfmaq_f32(acc1, v1, v1);
        }
        let mut sum = hsum_pairwise_8wide_f32(acc0, acc1);
        for &v in &x[chunks8 * 8..] {
            sum = v.mul_add(v, sum);
        }
        sum
    }
}

/// ARMv8.0 NEON `f32 x bf16` dot product matching `generic::dot_f32_bf16` bit-for-bit.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_bf16(x: &[f32], w_bf16: &[u16]) -> f32 {
    let n = x.len();
    assert_eq!(w_bf16.len(), n);
    let chunks8 = n / 8;
    // SAFETY: `x` and `w_bf16` both have length `n`; `c * 8 + 8 <= n`.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let xp = x.as_ptr();
        let wp = w_bf16.as_ptr();
        for c in 0..chunks8 {
            let off = c * 8;
            let raw8 = vld1q_u16(wp.add(off));
            let w0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(raw8)));
            let w1 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(raw8)));
            let x0 = vld1q_f32(xp.add(off));
            let x1 = vld1q_f32(xp.add(off + 4));
            acc0 = vfmaq_f32(acc0, w0, x0);
            acc1 = vfmaq_f32(acc1, w1, x1);
        }
        let mut sum = hsum_pairwise_8wide_f32(acc0, acc1);
        for i in chunks8 * 8..n {
            let w = f32::from_bits((*wp.add(i) as u32) << 16);
            sum = w.mul_add(*xp.add(i), sum);
        }
        sum
    }
}

/// ARMv8.0 NEON `rms_norm_f32` matching `generic::rms_norm_f32` bit-for-bit.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn rms_norm_f32(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(out.len(), n);
    assert_eq!(weight.len(), n);
    if n == 0 {
        return;
    }
    // SAFETY: `out`, `x`, and `weight` all have verified length `n`.
    unsafe {
        let ss = sum_sq_8wide_neon(x);
        let scale = (1.0 / (((ss as f64) / (n as f64)) + (eps as f64)).sqrt()) as f32;
        let vscale = vdupq_n_f32(scale);
        let chunks4 = n / 4;
        let xp = x.as_ptr();
        let wp = weight.as_ptr();
        let op = out.as_mut_ptr();
        for c in 0..chunks4 {
            let off = c * 4;
            let vx = vld1q_f32(xp.add(off));
            let vw = vld1q_f32(wp.add(off));
            vst1q_f32(op.add(off), vmulq_f32(vmulq_f32(vx, vscale), vw));
        }
        for i in chunks4 * 4..n {
            *op.add(i) = (*xp.add(i) * scale) * *wp.add(i);
        }
    }
}

/// ARMv8.0 NEON in-place `l2_norm_f32` matching `generic::l2_norm_f32` bit-for-bit.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn l2_norm_f32(x: &mut [f32], eps: f32) {
    let n = x.len();
    // SAFETY: `x` has length `n`; 4-wide chunks stay within `[0..n)`.
    unsafe {
        let ss = sum_sq_8wide_neon(x);
        let scale = (1.0 / ((ss as f64) + (eps as f64)).sqrt()) as f32;
        let vscale = vdupq_n_f32(scale);
        let chunks4 = n / 4;
        let xp = x.as_mut_ptr();
        for c in 0..chunks4 {
            let off = c * 4;
            let vx = vld1q_f32(xp.add(off));
            vst1q_f32(xp.add(off), vmulq_f32(vx, vscale));
        }
        for v in &mut x[chunks4 * 4..] {
            *v *= scale;
        }
    }
}

/// ARMv8.0 NEON single-head, single-token Gated DeltaNet recurrent step.
///
/// Uses 8-wide dual-accumulator `vfmaq_f32` + pairwise `vpaddq_f32` + single state memory store,
/// matching `generic::gdn_recurrence_head` **bit-for-bit** (`to_bits()`).
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_recurrence_head(
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
    // SAFETY: All slice bounds are asserted above; `c * 8 + 8 <= head_dim` and `j < head_dim`.
    unsafe {
        let vg = vdupq_n_f32(g);
        let qp = q_h.as_ptr();
        let kp = k_h.as_ptr();
        let vp = v_h.as_ptr();
        let sp_base = s_vh.as_mut_ptr();
        let op = o_h.as_mut_ptr();

        for j in 0..head_dim {
            let sp = sp_base.add(j * head_dim);

            let mut kv0 = vdupq_n_f32(0.0);
            let mut kv1 = vdupq_n_f32(0.0);
            for c in 0..chunks8 {
                let off = c * 8;
                let d0 = vmulq_f32(vld1q_f32(sp.add(off)), vg);
                let d1 = vmulq_f32(vld1q_f32(sp.add(off + 4)), vg);
                let k0 = vld1q_f32(kp.add(off));
                let k1 = vld1q_f32(kp.add(off + 4));
                kv0 = vfmaq_f32(kv0, d0, k0);
                kv1 = vfmaq_f32(kv1, d1, k1);
            }
            let mut sum_kv = hsum_pairwise_8wide_f32(kv0, kv1);
            for i in chunks8 * 8..head_dim {
                let d = *sp.add(i) * g;
                sum_kv = d.mul_add(*kp.add(i), sum_kv);
            }

            let delta_j = (*vp.add(j) - sum_kv) * beta;
            let vdelta = vdupq_n_f32(delta_j);

            let mut oq0 = vdupq_n_f32(0.0);
            let mut oq1 = vdupq_n_f32(0.0);
            for c in 0..chunks8 {
                let off = c * 8;
                let d0 = vmulq_f32(vld1q_f32(sp.add(off)), vg);
                let d1 = vmulq_f32(vld1q_f32(sp.add(off + 4)), vg);
                let k0 = vld1q_f32(kp.add(off));
                let k1 = vld1q_f32(kp.add(off + 4));
                let u0 = vfmaq_f32(d0, vdelta, k0);
                let u1 = vfmaq_f32(d1, vdelta, k1);
                vst1q_f32(sp.add(off), u0);
                vst1q_f32(sp.add(off + 4), u1);
                let q0 = vld1q_f32(qp.add(off));
                let q1 = vld1q_f32(qp.add(off + 4));
                oq0 = vfmaq_f32(oq0, u0, q0);
                oq1 = vfmaq_f32(oq1, u1, q1);
            }
            let mut sum_oq = hsum_pairwise_8wide_f32(oq0, oq1);
            for i in chunks8 * 8..head_dim {
                let u = delta_j.mul_add(*kp.add(i), *sp.add(i) * g);
                *sp.add(i) = u;
                sum_oq = u.mul_add(*qp.add(i), sum_oq);
            }

            *op.add(j) = sum_oq * scale;
        }
    }
}

/// Base ARMv8.0 NEON `gemv_q4_0x4_q8_0` (Pi Zero 2W / Pi 4B).
///
/// Matches `generic::gemv_q4_0x4_q8_0` **bit-for-bit** (`to_bits()`) across all scales.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q4_0x4_q8_0(
    w: &[BlockQ4_0x4],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(rows % 4, 0);
    assert_eq!(w.len(), (rows / 4) * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    // SAFETY: All group and block indices stay strictly within verified slice bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);

        for rg in 0..(rows / 4) {
            let w_group = &w[rg * cols_blocks..(rg + 1) * cols_blocks];
            let mut acc4 = vdupq_n_f32(0.0);

            for ib in 0..cols_blocks {
                let sb = w_group.get_unchecked(ib);
                let by = x.get_unchecked(ib);
                let yl = vld1q_s8(by.qs.as_ptr());
                let yh = vld1q_s8(by.qs.as_ptr().add(16));
                let y0 = vget_low_s8(yl);
                let y1 = vget_high_s8(yl);
                let y2 = vget_low_s8(yh);
                let y3 = vget_high_s8(yh);

                let v0 = vld1q_u8(sb.qs.as_ptr());
                let v1 = vld1q_u8(sb.qs.as_ptr().add(16));
                let v2 = vld1q_u8(sb.qs.as_ptr().add(32));
                let v3 = vld1q_u8(sb.qs.as_ptr().add(48));

                let la0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
                let la2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), s8b);
                let la1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4b)), s8b);
                let la3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), s8b);

                let lb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v2, m4b)), s8b);
                let lb2 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v2)), s8b);
                let lb1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v3, m4b)), s8b);
                let lb3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v3)), s8b);

                // Max per-lane i16 magnitude: 4 * 8 * 128 = 4096 <= 32767 (zero overflow).
                let p0 = vmlal_s8(
                    vmlal_s8(
                        vmlal_s8(vmull_s8(vget_low_s8(la0), y0), vget_low_s8(la1), y1),
                        vget_low_s8(la2),
                        y2,
                    ),
                    vget_low_s8(la3),
                    y3,
                );
                let p1 = vmlal_s8(
                    vmlal_s8(
                        vmlal_s8(vmull_s8(vget_high_s8(la0), y0), vget_high_s8(la1), y1),
                        vget_high_s8(la2),
                        y2,
                    ),
                    vget_high_s8(la3),
                    y3,
                );
                let p2 = vmlal_s8(
                    vmlal_s8(
                        vmlal_s8(vmull_s8(vget_low_s8(lb0), y0), vget_low_s8(lb1), y1),
                        vget_low_s8(lb2),
                        y2,
                    ),
                    vget_low_s8(lb3),
                    y3,
                );
                let p3 = vmlal_s8(
                    vmlal_s8(
                        vmlal_s8(vmull_s8(vget_high_s8(lb0), y0), vget_high_s8(lb1), y1),
                        vget_high_s8(lb2),
                        y2,
                    ),
                    vget_high_s8(lb3),
                    y3,
                );

                let sumi = [
                    vaddlvq_s16(p0),
                    vaddlvq_s16(p1),
                    vaddlvq_s16(p2),
                    vaddlvq_s16(p3),
                ];
                let dy = f16_to_f32_hw(by.d);
                let scales = [
                    f16_to_f32_hw(sb.d[0]) * dy,
                    f16_to_f32_hw(sb.d[1]) * dy,
                    f16_to_f32_hw(sb.d[2]) * dy,
                    f16_to_f32_hw(sb.d[3]) * dy,
                ];
                acc4 = vfmaq_f32(
                    acc4,
                    vcvtq_f32_s32(vld1q_s32(sumi.as_ptr())),
                    vld1q_f32(scales.as_ptr()),
                );
            }

            vst1q_f32(out.as_mut_ptr().add(rg * 4), acc4);
        }
    }
}

/// Base ARMv8.0 NEON multi-token GEMM for `BlockQ4_0x4 x BlockQ8_0`.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
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

    // SAFETY: Verified slice lengths above.
    unsafe {
        for t in 0..n_tokens {
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

/// Base ARMv8.0 NEON multi-token GEMM for `Q8_0 x Q8_0`.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_q8_0_q8_0(
    w: &[BlockQ8_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified slice lengths above.
    unsafe {
        for t in 0..n_tokens {
            gemv_q8_0_q8_0(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// Base ARMv8.0 NEON multi-token GEMM for `Q6_K x Q8_K`.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_q6_k_q8_k(
    w: &[BlockQ6K],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified slice lengths above.
    unsafe {
        for t in 0..n_tokens {
            gemv_q6_k_q8_k(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// Base ARMv8.0 NEON `f32 -> BlockQ8K` quantization, bit-for-bit identical to `tm_core::quantize_row_q8_k`.
///
/// # Safety
/// Caller must ensure NEON is available and `src.len() == dst.len() * 256`.
#[target_feature(enable = "neon")]
pub unsafe fn quantize_q8_k(src: &[f32], dst: &mut [BlockQ8K]) {
    assert_eq!(src.len(), dst.len() * QK_K);
    // SAFETY: Each iteration reads 256 `f32`s and writes 256 `i8`s + 16 `i16`s strictly in bounds.
    unsafe {
        let sp_base = src.as_ptr();
        for (ib, block) in dst.iter_mut().enumerate() {
            let sp = sp_base.add(ib * QK_K);

            // Fast vectorized max-abs reduction across 256 elements
            let mut max_vec = vdupq_n_f32(0.0);
            for c in 0..(QK_K / 16) {
                let p = sp.add(c * 16);
                let a01 = vmaxq_f32(vabsq_f32(vld1q_f32(p)), vabsq_f32(vld1q_f32(p.add(4))));
                let a23 = vmaxq_f32(
                    vabsq_f32(vld1q_f32(p.add(8))),
                    vabsq_f32(vld1q_f32(p.add(12))),
                );
                max_vec = vmaxq_f32(max_vec, vmaxq_f32(a01, a23));
            }
            let amax = vmaxvq_f32(max_vec);
            if amax == 0.0 {
                block.d = 0.0;
                block.qs = [0; QK_K];
                block.bsums = [0; QK_K / 16];
                continue;
            }

            // Locate first signed value attaining `amax` (exact sign match with `tm_core::quantize_row_q8_k`)
            let mut max_v = 0.0f32;
            let slice = core::slice::from_raw_parts(sp, QK_K);
            for &v in slice {
                if v.abs() == amax {
                    max_v = v;
                    break;
                }
            }

            let iscale = -127.0 / max_v;
            let viscale = vdupq_n_f32(iscale);
            let qp = block.qs.as_mut_ptr();

            for j in 0..(QK_K / 16) {
                let p = sp.add(j * 16);
                let i0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p), viscale));
                let i1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(4)), viscale));
                let i2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(8)), viscale));
                let i3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(12)), viscale));

                let s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                let s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                let q16 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                vst1q_s8(qp.add(j * 16), q16);
                block.bsums[j] = vaddlvq_s16(vpaddlq_s8(q16)) as i16;
            }
            block.d = 1.0 / iscale;
        }
    }
}

/// Base ARMv8.0 NEON `f32 x BlockQ8_0` dot product (for `Q8_0` KV cache attention),
/// matching `generic::dot_f32_q8_0` **bit-for-bit** (`to_bits()`).
///
/// # Safety
/// Caller must ensure NEON is available and `q.len() == k_q8.len() * 32`.
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_q8_0(q: &[f32], k_q8: &[BlockQ8_0]) -> f32 {
    assert_eq!(q.len(), k_q8.len() * QK8_0);
    // SAFETY: `q` has `k_q8.len() * 32` elements; all 32-element block loads are in bounds.
    unsafe {
        let qp_base = q.as_ptr();
        let mut total = 0.0f32;

        for (ib, blk) in k_q8.iter().enumerate() {
            let qp = qp_base.add(ib * 32);
            let kp = blk.qs.as_ptr();
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);

            for c in 0..4 {
                let off = c * 8;
                let k8 = vld1_s8(kp.add(off));
                let k16 = vmovl_s8(k8);
                let kf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k16)));
                let kf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(k16)));
                let q0 = vld1q_f32(qp.add(off));
                let q1 = vld1q_f32(qp.add(off + 4));
                acc0 = vfmaq_f32(acc0, kf0, q0);
                acc1 = vfmaq_f32(acc1, kf1, q1);
            }

            let bsum = hsum_pairwise_8wide_f32(acc0, acc1);
            total = bsum.mul_add(f16_to_f32_hw(blk.d), total);
        }
        total
    }
}

/// Base ARMv8.0 NEON `out[d] += w * dequant(v_q8[d])` (for `Q8_0` V cache aggregation),
/// matching `generic::axpy_f32_q8_0` **bit-for-bit** (`to_bits()`).
///
/// # Safety
/// Caller must ensure NEON is available and `out.len() == v_q8.len() * 32`.
#[target_feature(enable = "neon")]
pub unsafe fn axpy_f32_q8_0(out: &mut [f32], w: f32, v_q8: &[BlockQ8_0]) {
    assert_eq!(out.len(), v_q8.len() * QK8_0);
    // SAFETY: `out` has `v_q8.len() * 32` elements; all 32-element block loads/stores are in bounds.
    unsafe {
        let op_base = out.as_mut_ptr();
        for (ib, blk) in v_q8.iter().enumerate() {
            let vscale = vdupq_n_f32(w * f16_to_f32_hw(blk.d));
            let op = op_base.add(ib * 32);
            let vp = blk.qs.as_ptr();
            for c in 0..4 {
                let off = c * 8;
                let v8 = vld1_s8(vp.add(off));
                let v16 = vmovl_s8(v8);
                let vf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16)));
                let vf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16)));
                let o0 = vld1q_f32(op.add(off));
                let o1 = vld1q_f32(op.add(off + 4));
                vst1q_f32(op.add(off), vfmaq_f32(o0, vf0, vscale));
                vst1q_f32(op.add(off + 4), vfmaq_f32(o1, vf1, vscale));
            }
        }
    }
}

/// Base ARMv8.0 NEON `out[i] += w * x[i]`, matching `generic::axpy_f32` **bit-for-bit**.
///
/// # Safety
/// Caller must ensure NEON is available and `out.len() == x.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn axpy_f32(out: &mut [f32], w: f32, x: &[f32]) {
    assert_eq!(out.len(), x.len());
    let n = out.len();
    // SAFETY: `out` and `x` both have length `n`; 4-element vector accesses stay within `0..n4`.
    unsafe {
        let vw = vdupq_n_f32(w);
        let op = out.as_mut_ptr();
        let xp = x.as_ptr();
        let mut i = 0usize;
        while i + 8 <= n {
            let o0 = vld1q_f32(op.add(i));
            let o1 = vld1q_f32(op.add(i + 4));
            let x0 = vld1q_f32(xp.add(i));
            let x1 = vld1q_f32(xp.add(i + 4));
            vst1q_f32(op.add(i), vfmaq_f32(o0, x0, vw));
            vst1q_f32(op.add(i + 4), vfmaq_f32(o1, x1, vw));
            i += 8;
        }
        while i + 4 <= n {
            let o0 = vld1q_f32(op.add(i));
            let x0 = vld1q_f32(xp.add(i));
            vst1q_f32(op.add(i), vfmaq_f32(o0, x0, vw));
            i += 4;
        }
        while i < n {
            let oi = *op.add(i);
            let xi = *xp.add(i);
            *op.add(i) = xi.mul_add(w, oi);
            i += 1;
        }
    }
}

/// Base ARMv8.0 NEON register-blocked transposed `Q4_0` column-block accumulation:
/// `dx_tile[t, 0..32] += sum_r dy[t, r] * dequant(W[r, cb])`, matching
/// `generic::gemm_q4_0_transposed_tile` **bit-for-bit** (`0 ULP`).
///
/// Processes 2 tokens at a time in 16 accumulator registers + 8 unpacked weight registers (24 NEON
/// registers total, zero spills, zero memory round-trips inside the `r = 0..rows` inner loop).
///
/// # Safety
/// Caller must ensure NEON is available and slice bounds match `rows`, `cols_blocks`, `n_tok`.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_q4_0_transposed_tile(
    w: &[BlockQ4_0],
    cols_blocks: usize,
    cb: usize,
    dy: &[f32],
    dx_tile: &mut [f32],
    rows: usize,
    n_tok: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert!(cb < cols_blocks);
    assert_eq!(dy.len(), n_tok * rows);
    assert_eq!(dx_tile.len(), n_tok * QK4_0);
    if rows == 0 || n_tok == 0 {
        return;
    }

    // SAFETY: All bounds verified above; `dx_tile` has `n_tok * 32` floats and `w` has `rows * cols_blocks` blocks.
    unsafe {
        let mask_low = vdupq_n_u8(0x0F);
        let bias8 = vdupq_n_s16(8);
        let wp_base = w.as_ptr().add(cb);
        let dyp_base = dy.as_ptr();
        let dxp_base = dx_tile.as_mut_ptr();

        let mut t = 0usize;
        while t + 2 <= n_tok {
            let dx0 = dxp_base.add(t * 32);
            let dx1 = dxp_base.add((t + 1) * 32);
            let mut a0_0 = vld1q_f32(dx0);
            let mut a0_1 = vld1q_f32(dx0.add(4));
            let mut a0_2 = vld1q_f32(dx0.add(8));
            let mut a0_3 = vld1q_f32(dx0.add(12));
            let mut a0_4 = vld1q_f32(dx0.add(16));
            let mut a0_5 = vld1q_f32(dx0.add(20));
            let mut a0_6 = vld1q_f32(dx0.add(24));
            let mut a0_7 = vld1q_f32(dx0.add(28));

            let mut a1_0 = vld1q_f32(dx1);
            let mut a1_1 = vld1q_f32(dx1.add(4));
            let mut a1_2 = vld1q_f32(dx1.add(8));
            let mut a1_3 = vld1q_f32(dx1.add(12));
            let mut a1_4 = vld1q_f32(dx1.add(16));
            let mut a1_5 = vld1q_f32(dx1.add(20));
            let mut a1_6 = vld1q_f32(dx1.add(24));
            let mut a1_7 = vld1q_f32(dx1.add(28));

            let dy0 = dyp_base.add(t * rows);
            let dy1 = dyp_base.add((t + 1) * rows);

            for r in 0..rows {
                let blk = &*wp_base.add(r * cols_blocks);
                let vd = vdupq_n_f32(f16_to_f32_hw(blk.d));
                let s0 = vdupq_n_f32(*dy0.add(r));
                let s1 = vdupq_n_f32(*dy1.add(r));

                let raw = vld1q_u8(blk.qs.as_ptr());
                let lo_u8 = vandq_u8(raw, mask_low);
                let hi_u8 = vshrq_n_u8::<4>(raw);

                let lo_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo_u8))), bias8);
                let lo_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo_u8))), bias8);
                let hi_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi_u8))), bias8);
                let hi_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi_u8))), bias8);

                let vf0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_0))), vd);
                let vf1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_0))), vd);
                let vf2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_1))), vd);
                let vf3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_1))), vd);
                let vf4 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_0))), vd);
                let vf5 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_0))), vd);
                let vf6 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_1))), vd);
                let vf7 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_1))), vd);

                a0_0 = vfmaq_f32(a0_0, vf0, s0);
                a0_1 = vfmaq_f32(a0_1, vf1, s0);
                a0_2 = vfmaq_f32(a0_2, vf2, s0);
                a0_3 = vfmaq_f32(a0_3, vf3, s0);
                a0_4 = vfmaq_f32(a0_4, vf4, s0);
                a0_5 = vfmaq_f32(a0_5, vf5, s0);
                a0_6 = vfmaq_f32(a0_6, vf6, s0);
                a0_7 = vfmaq_f32(a0_7, vf7, s0);

                a1_0 = vfmaq_f32(a1_0, vf0, s1);
                a1_1 = vfmaq_f32(a1_1, vf1, s1);
                a1_2 = vfmaq_f32(a1_2, vf2, s1);
                a1_3 = vfmaq_f32(a1_3, vf3, s1);
                a1_4 = vfmaq_f32(a1_4, vf4, s1);
                a1_5 = vfmaq_f32(a1_5, vf5, s1);
                a1_6 = vfmaq_f32(a1_6, vf6, s1);
                a1_7 = vfmaq_f32(a1_7, vf7, s1);
            }

            vst1q_f32(dx0, a0_0);
            vst1q_f32(dx0.add(4), a0_1);
            vst1q_f32(dx0.add(8), a0_2);
            vst1q_f32(dx0.add(12), a0_3);
            vst1q_f32(dx0.add(16), a0_4);
            vst1q_f32(dx0.add(20), a0_5);
            vst1q_f32(dx0.add(24), a0_6);
            vst1q_f32(dx0.add(28), a0_7);

            vst1q_f32(dx1, a1_0);
            vst1q_f32(dx1.add(4), a1_1);
            vst1q_f32(dx1.add(8), a1_2);
            vst1q_f32(dx1.add(12), a1_3);
            vst1q_f32(dx1.add(16), a1_4);
            vst1q_f32(dx1.add(20), a1_5);
            vst1q_f32(dx1.add(24), a1_6);
            vst1q_f32(dx1.add(28), a1_7);

            t += 2;
        }

        if t < n_tok {
            let dx0 = dxp_base.add(t * 32);
            let mut a0_0 = vld1q_f32(dx0);
            let mut a0_1 = vld1q_f32(dx0.add(4));
            let mut a0_2 = vld1q_f32(dx0.add(8));
            let mut a0_3 = vld1q_f32(dx0.add(12));
            let mut a0_4 = vld1q_f32(dx0.add(16));
            let mut a0_5 = vld1q_f32(dx0.add(20));
            let mut a0_6 = vld1q_f32(dx0.add(24));
            let mut a0_7 = vld1q_f32(dx0.add(28));
            let dy0 = dyp_base.add(t * rows);

            for r in 0..rows {
                let blk = &*wp_base.add(r * cols_blocks);
                let vd = vdupq_n_f32(f16_to_f32_hw(blk.d));
                let s0 = vdupq_n_f32(*dy0.add(r));

                let raw = vld1q_u8(blk.qs.as_ptr());
                let lo_u8 = vandq_u8(raw, mask_low);
                let hi_u8 = vshrq_n_u8::<4>(raw);

                let lo_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(lo_u8))), bias8);
                let lo_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(lo_u8))), bias8);
                let hi_0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(hi_u8))), bias8);
                let hi_1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(hi_u8))), bias8);

                a0_0 = vfmaq_f32(
                    a0_0,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_0))), vd),
                    s0,
                );
                a0_1 = vfmaq_f32(
                    a0_1,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_0))), vd),
                    s0,
                );
                a0_2 = vfmaq_f32(
                    a0_2,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_1))), vd),
                    s0,
                );
                a0_3 = vfmaq_f32(
                    a0_3,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_1))), vd),
                    s0,
                );
                a0_4 = vfmaq_f32(
                    a0_4,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_0))), vd),
                    s0,
                );
                a0_5 = vfmaq_f32(
                    a0_5,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_0))), vd),
                    s0,
                );
                a0_6 = vfmaq_f32(
                    a0_6,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_1))), vd),
                    s0,
                );
                a0_7 = vfmaq_f32(
                    a0_7,
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_1))), vd),
                    s0,
                );
            }

            vst1q_f32(dx0, a0_0);
            vst1q_f32(dx0.add(4), a0_1);
            vst1q_f32(dx0.add(8), a0_2);
            vst1q_f32(dx0.add(12), a0_3);
            vst1q_f32(dx0.add(16), a0_4);
            vst1q_f32(dx0.add(20), a0_5);
            vst1q_f32(dx0.add(24), a0_6);
            vst1q_f32(dx0.add(28), a0_7);
        }
    }
}
