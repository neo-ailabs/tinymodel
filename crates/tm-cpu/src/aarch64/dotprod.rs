//! ARMv8.2 NEON `dotprod` (`vdotq_s32`) kernels for `Q4_0 x Q8_0`.
//!
//! Ported from ggml / Strata:
//! - Copyright (c) 2023-2026 The ggml authors (MIT License)
//! - Copyright (c) 2026 Niko1221 and the Strata contributors (MIT License)
//! - Upstream reference: `Strata/src/quant/arm/arm_dots.inc:317-377`
//!   (ggml `ggml/src/ggml-cpu/arch/arm/quants.c` commit `3cf03257`).

use core::arch::aarch64::*;
use tm_core::{
    BlockQ4_0, BlockQ4_0x4, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, QK_K,
    unpack_q5k_scales_mins,
};

use super::asm::f16_to_f32_hw;

/// Single-row `vec_dot_q4_0_q8_0` using ARMv8.2 `vdotq_s32` (2-block unrolled).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and `x.len() == y.len()`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut ib = 0usize;

    // SAFETY: `ib` stays strictly within `0..nb`; `qs` slices have 16 and 32 bytes.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        let mut sumv0 = vdupq_n_f32(0.0);
        let mut sumv1 = vdupq_n_f32(0.0);

        while ib + 1 < nb {
            let x0 = x.get_unchecked(ib);
            let x1 = x.get_unchecked(ib + 1);
            let y0 = y.get_unchecked(ib);
            let y1 = y.get_unchecked(ib + 1);

            let v0_0 = vld1q_u8(x0.qs.as_ptr());
            let v0_1 = vld1q_u8(x1.qs.as_ptr());

            let v0_0ls = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);
            let v0_0hs = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0_0)), s8b);
            let v0_1ls = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_1, m4b)), s8b);
            let v0_1hs = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0_1)), s8b);

            let v1_0l = vld1q_s8(y0.qs.as_ptr());
            let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
            let v1_1l = vld1q_s8(y1.qs.as_ptr());
            let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

            let p0 = vdotq_s32(vdotq_s32(zero, v0_0ls, v1_0l), v0_0hs, v1_0h);
            let p1 = vdotq_s32(vdotq_s32(zero, v0_1ls, v1_1l), v0_1hs, v1_1h);

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
            let v0 = vld1q_u8(x0.qs.as_ptr());
            let v0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
            let v0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), s8b);
            let v1l = vld1q_s8(y0.qs.as_ptr());
            let v1h = vld1q_s8(y0.qs.as_ptr().add(16));
            let p0 = vdotq_s32(vdotq_s32(zero, v0l, v1l), v0h, v1h);
            sumf += (vaddvq_s32(p0) as f32) * (f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d));
        }

        sumf
    }
}

/// 4-row unrolled `dotprod` GEMV: loads activation vector `x` once per 4 weight rows.
///
/// Uses the exact same 2-block accumulator pair (`sumv0`, `sumv1`) per row as
/// `vec_dot_q4_0_q8_0` so single-row and 4-row outputs are bit-for-bit identical.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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
    // SAFETY: All row and block indices are strictly within bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut s0a = vdupq_n_f32(0.0);
            let mut s0b = vdupq_n_f32(0.0);
            let mut s1a = vdupq_n_f32(0.0);
            let mut s1b = vdupq_n_f32(0.0);
            let mut s2a = vdupq_n_f32(0.0);
            let mut s2b = vdupq_n_f32(0.0);
            let mut s3a = vdupq_n_f32(0.0);
            let mut s3b = vdupq_n_f32(0.0);

            let mut ib = 0usize;
            while ib + 1 < cols_blocks {
                let y0 = x.get_unchecked(ib);
                let y1 = x.get_unchecked(ib + 1);
                let y0l = vld1q_s8(y0.qs.as_ptr());
                let y0h = vld1q_s8(y0.qs.as_ptr().add(16));
                let y1l = vld1q_s8(y1.qs.as_ptr());
                let y1h = vld1q_s8(y1.qs.as_ptr().add(16));
                let dy0 = f16_to_f32_hw(y0.d);
                let dy1 = f16_to_f32_hw(y1.d);

                macro_rules! step_row {
                    ($row:expr, $sa:expr, $sb:expr) => {{
                        let xa = $row.get_unchecked(ib);
                        let xb = $row.get_unchecked(ib + 1);
                        let va = vld1q_u8(xa.qs.as_ptr());
                        let vb = vld1q_u8(xb.qs.as_ptr());
                        let val = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(va, m4b)), s8b);
                        let vah = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(va)), s8b);
                        let vbl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vb, m4b)), s8b);
                        let vbh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(vb)), s8b);
                        let pa = vdotq_s32(vdotq_s32(zero, val, y0l), vah, y0h);
                        let pb = vdotq_s32(vdotq_s32(zero, vbl, y1l), vbh, y1h);
                        $sa = vmlaq_n_f32($sa, vcvtq_f32_s32(pa), f16_to_f32_hw(xa.d) * dy0);
                        $sb = vmlaq_n_f32($sb, vcvtq_f32_s32(pb), f16_to_f32_hw(xb.d) * dy1);
                    }};
                }

                step_row!(w0, s0a, s0b);
                step_row!(w1, s1a, s1b);
                step_row!(w2, s2a, s2b);
                step_row!(w3, s3a, s3b);

                ib += 2;
            }

            let mut out0 = vaddvq_f32(s0a) + vaddvq_f32(s0b);
            let mut out1 = vaddvq_f32(s1a) + vaddvq_f32(s1b);
            let mut out2 = vaddvq_f32(s2a) + vaddvq_f32(s2b);
            let mut out3 = vaddvq_f32(s3a) + vaddvq_f32(s3b);

            if ib < cols_blocks {
                let y0 = x.get_unchecked(ib);
                let y0l = vld1q_s8(y0.qs.as_ptr());
                let y0h = vld1q_s8(y0.qs.as_ptr().add(16));
                let dy0 = f16_to_f32_hw(y0.d);

                macro_rules! tail_row {
                    ($row:expr, $acc:expr) => {{
                        let xa = $row.get_unchecked(ib);
                        let va = vld1q_u8(xa.qs.as_ptr());
                        let val = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(va, m4b)), s8b);
                        let vah = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(va)), s8b);
                        let pa = vdotq_s32(vdotq_s32(zero, val, y0l), vah, y0h);
                        $acc += (vaddvq_s32(pa) as f32) * (f16_to_f32_hw(xa.d) * dy0);
                    }};
                }

                tail_row!(w0, out0);
                tail_row!(w1, out1);
                tail_row!(w2, out2);
                tail_row!(w3, out3);
            }

            *out.get_unchecked_mut(r) = out0;
            *out.get_unchecked_mut(r + 1) = out1;
            *out.get_unchecked_mut(r + 2) = out2;
            *out.get_unchecked_mut(r + 3) = out3;
            r += 4;
        }

        while r < rows {
            let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q4_0_q8_0(row, x);
            r += 1;
        }
    }
}

/// Multi-token shared-weight `dotprod` GEMM (for speculative verify / small-batch decode).
///
/// Unpacks each weight block ONCE and dots it against up to 4 tokens per inner loop,
/// maintaining the exact per-token accumulator order of `vec_dot_q4_0_q8_0` so
/// batch-width invariance (1 token vs 1..8 tokens) holds bit-for-bit.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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
        // SAFETY: dimensions checked above.
        unsafe {
            gemv_q4_0_q8_0(w, x, out, rows, cols_blocks);
        }
        return;
    }

    // SAFETY: All indices are strictly bounded by `rows`, `cols_blocks`, `n_tokens`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        let mut t0 = 0usize;
        while t0 + 4 <= n_tokens {
            let xt0 = &x[t0 * cols_blocks..(t0 + 1) * cols_blocks];
            let xt1 = &x[(t0 + 1) * cols_blocks..(t0 + 2) * cols_blocks];
            let xt2 = &x[(t0 + 2) * cols_blocks..(t0 + 3) * cols_blocks];
            let xt3 = &x[(t0 + 3) * cols_blocks..(t0 + 4) * cols_blocks];

            for r in 0..rows {
                let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
                let mut s0a = vdupq_n_f32(0.0);
                let mut s0b = vdupq_n_f32(0.0);
                let mut s1a = vdupq_n_f32(0.0);
                let mut s1b = vdupq_n_f32(0.0);
                let mut s2a = vdupq_n_f32(0.0);
                let mut s2b = vdupq_n_f32(0.0);
                let mut s3a = vdupq_n_f32(0.0);
                let mut s3b = vdupq_n_f32(0.0);

                let mut ib = 0usize;
                while ib + 1 < cols_blocks {
                    let xa = row.get_unchecked(ib);
                    let xb = row.get_unchecked(ib + 1);
                    let va = vld1q_u8(xa.qs.as_ptr());
                    let vb = vld1q_u8(xb.qs.as_ptr());
                    let val = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(va, m4b)), s8b);
                    let vah = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(va)), s8b);
                    let vbl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vb, m4b)), s8b);
                    let vbh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(vb)), s8b);
                    let dxa = f16_to_f32_hw(xa.d);
                    let dxb = f16_to_f32_hw(xb.d);

                    macro_rules! step_tok {
                        ($xt:expr, $sa:expr, $sb:expr) => {{
                            let ya = $xt.get_unchecked(ib);
                            let yb = $xt.get_unchecked(ib + 1);
                            let yal = vld1q_s8(ya.qs.as_ptr());
                            let yah = vld1q_s8(ya.qs.as_ptr().add(16));
                            let ybl = vld1q_s8(yb.qs.as_ptr());
                            let ybh = vld1q_s8(yb.qs.as_ptr().add(16));
                            let pa = vdotq_s32(vdotq_s32(zero, val, yal), vah, yah);
                            let pb = vdotq_s32(vdotq_s32(zero, vbl, ybl), vbh, ybh);
                            $sa = vmlaq_n_f32($sa, vcvtq_f32_s32(pa), dxa * f16_to_f32_hw(ya.d));
                            $sb = vmlaq_n_f32($sb, vcvtq_f32_s32(pb), dxb * f16_to_f32_hw(yb.d));
                        }};
                    }

                    step_tok!(xt0, s0a, s0b);
                    step_tok!(xt1, s1a, s1b);
                    step_tok!(xt2, s2a, s2b);
                    step_tok!(xt3, s3a, s3b);

                    ib += 2;
                }

                let mut o0 = vaddvq_f32(s0a) + vaddvq_f32(s0b);
                let mut o1 = vaddvq_f32(s1a) + vaddvq_f32(s1b);
                let mut o2 = vaddvq_f32(s2a) + vaddvq_f32(s2b);
                let mut o3 = vaddvq_f32(s3a) + vaddvq_f32(s3b);

                if ib < cols_blocks {
                    let xa = row.get_unchecked(ib);
                    let va = vld1q_u8(xa.qs.as_ptr());
                    let val = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(va, m4b)), s8b);
                    let vah = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(va)), s8b);
                    let dxa = f16_to_f32_hw(xa.d);

                    macro_rules! tail_tok {
                        ($xt:expr, $acc:expr) => {{
                            let ya = $xt.get_unchecked(ib);
                            let yal = vld1q_s8(ya.qs.as_ptr());
                            let yah = vld1q_s8(ya.qs.as_ptr().add(16));
                            let pa = vdotq_s32(vdotq_s32(zero, val, yal), vah, yah);
                            $acc += (vaddvq_s32(pa) as f32) * (dxa * f16_to_f32_hw(ya.d));
                        }};
                    }

                    tail_tok!(xt0, o0);
                    tail_tok!(xt1, o1);
                    tail_tok!(xt2, o2);
                    tail_tok!(xt3, o3);
                }

                *out.get_unchecked_mut(t0 * rows + r) = o0;
                *out.get_unchecked_mut((t0 + 1) * rows + r) = o1;
                *out.get_unchecked_mut((t0 + 2) * rows + r) = o2;
                *out.get_unchecked_mut((t0 + 3) * rows + r) = o3;
            }
            t0 += 4;
        }

        while t0 < n_tokens {
            let xt = &x[t0 * cols_blocks..(t0 + 1) * cols_blocks];
            let ot = &mut out[t0 * rows..(t0 + 1) * rows];
            gemv_q4_0_q8_0(w, xt, ot, rows, cols_blocks);
            t0 += 1;
        }
    }
}

/// ARMv8.2 `dotprod` `vec_dot_q4_1_q8_1`.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and `x.len() == y.len()`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_1_q8_1(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut ib = 0usize;

    // SAFETY: `ib` stays strictly within `0..nb`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let zero = vdupq_n_s32(0);
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

            let p0 = vdotq_s32(vdotq_s32(zero, x0l, y0l), x0h, y0h);
            let p1 = vdotq_s32(vdotq_s32(zero, x1l, y1l), x1h, y1h);

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
            let p0 = vdotq_s32(vdotq_s32(zero, x0l, y0l), x0h, y0h);
            sumf += (vaddvq_s32(p0) as f32) * (f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d))
                + f16_to_f32_hw(x0.m) * f16_to_f32_hw(y0.s);
        }

        sumf
    }
}

/// 4-row unrolled ARMv8.2 `dotprod` `gemv_q4_1_q8_1` (shares `y` loads across 4 weight rows).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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

    // SAFETY: All row/block indices are strictly bounded by `rows` and `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let zero = vdupq_n_s32(0);

        let mut r = 0usize;
        while r + 4 <= rows {
            let r0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let r1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let r2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let r3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut sv0a = vdupq_n_f32(0.0);
            let mut sv0b = vdupq_n_f32(0.0);
            let mut sv1a = vdupq_n_f32(0.0);
            let mut sv1b = vdupq_n_f32(0.0);
            let mut sv2a = vdupq_n_f32(0.0);
            let mut sv2b = vdupq_n_f32(0.0);
            let mut sv3a = vdupq_n_f32(0.0);
            let mut sv3b = vdupq_n_f32(0.0);

            let mut sms0a = 0.0f32;
            let mut sms0b = 0.0f32;
            let mut sms1a = 0.0f32;
            let mut sms1b = 0.0f32;
            let mut sms2a = 0.0f32;
            let mut sms2b = 0.0f32;
            let mut sms3a = 0.0f32;
            let mut sms3b = 0.0f32;

            let mut ib = 0usize;
            while ib + 1 < cols_blocks {
                let ya = x.get_unchecked(ib);
                let yb = x.get_unchecked(ib + 1);
                let yal = vld1q_s8(ya.qs.as_ptr());
                let yah = vld1q_s8(ya.qs.as_ptr().add(16));
                let ybl = vld1q_s8(yb.qs.as_ptr());
                let ybh = vld1q_s8(yb.qs.as_ptr().add(16));
                let dya = f16_to_f32_hw(ya.d);
                let dyb = f16_to_f32_hw(yb.d);
                let sya = f16_to_f32_hw(ya.s);
                let syb = f16_to_f32_hw(yb.s);

                macro_rules! step_row {
                    ($row:expr, $sva:expr, $svb:expr, $smsa:expr, $smsb:expr) => {{
                        let ba = $row.get_unchecked(ib);
                        let bb = $row.get_unchecked(ib + 1);
                        let va = vld1q_u8(ba.qs.as_ptr());
                        let vb = vld1q_u8(bb.qs.as_ptr());
                        let xal = vreinterpretq_s8_u8(vandq_u8(va, m4b));
                        let xah = vreinterpretq_s8_u8(vshrq_n_u8::<4>(va));
                        let xbl = vreinterpretq_s8_u8(vandq_u8(vb, m4b));
                        let xbh = vreinterpretq_s8_u8(vshrq_n_u8::<4>(vb));
                        let pa = vdotq_s32(vdotq_s32(zero, xal, yal), xah, yah);
                        let pb = vdotq_s32(vdotq_s32(zero, xbl, ybl), xbh, ybh);
                        $sva = vmlaq_n_f32($sva, vcvtq_f32_s32(pa), f16_to_f32_hw(ba.d) * dya);
                        $svb = vmlaq_n_f32($svb, vcvtq_f32_s32(pb), f16_to_f32_hw(bb.d) * dyb);
                        $smsa += f16_to_f32_hw(ba.m) * sya;
                        $smsb += f16_to_f32_hw(bb.m) * syb;
                    }};
                }

                step_row!(r0, sv0a, sv0b, sms0a, sms0b);
                step_row!(r1, sv1a, sv1b, sms1a, sms1b);
                step_row!(r2, sv2a, sv2b, sms2a, sms2b);
                step_row!(r3, sv3a, sv3b, sms3a, sms3b);

                ib += 2;
            }

            let mut o0 = vaddvq_f32(sv0a) + vaddvq_f32(sv0b) + sms0a + sms0b;
            let mut o1 = vaddvq_f32(sv1a) + vaddvq_f32(sv1b) + sms1a + sms1b;
            let mut o2 = vaddvq_f32(sv2a) + vaddvq_f32(sv2b) + sms2a + sms2b;
            let mut o3 = vaddvq_f32(sv3a) + vaddvq_f32(sv3b) + sms3a + sms3b;

            if ib < cols_blocks {
                let ya = x.get_unchecked(ib);
                let yal = vld1q_s8(ya.qs.as_ptr());
                let yah = vld1q_s8(ya.qs.as_ptr().add(16));
                let dya = f16_to_f32_hw(ya.d);
                let sya = f16_to_f32_hw(ya.s);

                macro_rules! tail_row {
                    ($row:expr, $acc:expr) => {{
                        let ba = $row.get_unchecked(ib);
                        let va = vld1q_u8(ba.qs.as_ptr());
                        let xal = vreinterpretq_s8_u8(vandq_u8(va, m4b));
                        let xah = vreinterpretq_s8_u8(vshrq_n_u8::<4>(va));
                        let pa = vdotq_s32(vdotq_s32(zero, xal, yal), xah, yah);
                        $acc += (vaddvq_s32(pa) as f32) * (f16_to_f32_hw(ba.d) * dya)
                            + f16_to_f32_hw(ba.m) * sya;
                    }};
                }

                tail_row!(r0, o0);
                tail_row!(r1, o1);
                tail_row!(r2, o2);
                tail_row!(r3, o3);
            }

            *out.get_unchecked_mut(r) = o0;
            *out.get_unchecked_mut(r + 1) = o1;
            *out.get_unchecked_mut(r + 2) = o2;
            *out.get_unchecked_mut(r + 3) = o3;
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q4_1_q8_1(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// ARMv8.2 `dotprod` `vec_dot_q8_0_q8_0` (`0-ULP` bit-exact with `generic::vec_dot_q8_0_q8_0`).
///
/// Reduces all 32 `i8 * i8` products per block in exact `i32` arithmetic (`vpaddq_s32` / `vaddvq_s32`)
/// before scaling by `bx.d * by.d`, matching `generic` and `avx2` bit-for-bit across all scales.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and `x.len() == y.len()`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q8_0_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut ib = 0usize;
    let mut sumf = 0.0f32;

    // SAFETY: `ib` stays strictly within `0..nb`.
    unsafe {
        let zero = vdupq_n_s32(0);

        while ib + 4 <= nb {
            let x0 = x.get_unchecked(ib);
            let x1 = x.get_unchecked(ib + 1);
            let x2 = x.get_unchecked(ib + 2);
            let x3 = x.get_unchecked(ib + 3);
            let y0 = y.get_unchecked(ib);
            let y1 = y.get_unchecked(ib + 1);
            let y2 = y.get_unchecked(ib + 2);
            let y3 = y.get_unchecked(ib + 3);

            let p0 = vdotq_s32(
                vdotq_s32(zero, vld1q_s8(x0.qs.as_ptr()), vld1q_s8(y0.qs.as_ptr())),
                vld1q_s8(x0.qs.as_ptr().add(16)),
                vld1q_s8(y0.qs.as_ptr().add(16)),
            );
            let p1 = vdotq_s32(
                vdotq_s32(zero, vld1q_s8(x1.qs.as_ptr()), vld1q_s8(y1.qs.as_ptr())),
                vld1q_s8(x1.qs.as_ptr().add(16)),
                vld1q_s8(y1.qs.as_ptr().add(16)),
            );
            let p2 = vdotq_s32(
                vdotq_s32(zero, vld1q_s8(x2.qs.as_ptr()), vld1q_s8(y2.qs.as_ptr())),
                vld1q_s8(x2.qs.as_ptr().add(16)),
                vld1q_s8(y2.qs.as_ptr().add(16)),
            );
            let p3 = vdotq_s32(
                vdotq_s32(zero, vld1q_s8(x3.qs.as_ptr()), vld1q_s8(y3.qs.as_ptr())),
                vld1q_s8(x3.qs.as_ptr().add(16)),
                vld1q_s8(y3.qs.as_ptr().add(16)),
            );

            let sum4 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
            let scales = [
                f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d),
                f16_to_f32_hw(x1.d) * f16_to_f32_hw(y1.d),
                f16_to_f32_hw(x2.d) * f16_to_f32_hw(y2.d),
                f16_to_f32_hw(x3.d) * f16_to_f32_hw(y3.d),
            ];
            let prod4 = vmulq_f32(vcvtq_f32_s32(sum4), vld1q_f32(scales.as_ptr()));
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), prod4);
            sumf = (((sumf + lanes[0]) + lanes[1]) + lanes[2]) + lanes[3];
            ib += 4;
        }

        while ib < nb {
            let x0 = x.get_unchecked(ib);
            let y0 = y.get_unchecked(ib);
            let x0l = vld1q_s8(x0.qs.as_ptr());
            let x0h = vld1q_s8(x0.qs.as_ptr().add(16));
            let y0l = vld1q_s8(y0.qs.as_ptr());
            let y0h = vld1q_s8(y0.qs.as_ptr().add(16));
            let p0 = vdotq_s32(vdotq_s32(zero, x0l, y0l), x0h, y0h);
            sumf += (vaddvq_s32(p0) as f32) * (f16_to_f32_hw(x0.d) * f16_to_f32_hw(y0.d));
            ib += 1;
        }

        sumf
    }
}

/// ARMv8.2 `dotprod` `vec_dot_q6_k_q8_k`.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and `x.len() == y.len()`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q6_k_q8_k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut sumf = 0.0f32;

    // SAFETY: All indices are strictly bounded by `nb` and `QK_K = 256`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let m2b = vdupq_n_u8(0x03);
        let s32 = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);

        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let mut isum: i32 = 0;

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

                    let d0 = vaddvq_s32(vdotq_s32(zero, q0, y0));
                    let d1 = vaddvq_s32(vdotq_s32(zero, q1, y1));
                    let d2 = vaddvq_s32(vdotq_s32(zero, q2, y2));
                    let d3 = vaddvq_s32(vdotq_s32(zero, q3, y3));

                    let s0 = *bx.scales.get_unchecked(sc_base + chunk) as i32;
                    let s1 = *bx.scales.get_unchecked(sc_base + 2 + chunk) as i32;
                    let s2 = *bx.scales.get_unchecked(sc_base + 4 + chunk) as i32;
                    let s3 = *bx.scales.get_unchecked(sc_base + 6 + chunk) as i32;

                    isum += s0 * d0 + s1 * d1 + s2 * d2 + s3 * d3;
                }
            }

            let _ = QK_K;
            sumf += f16_to_f32_hw(bx.d) * by.d * (isum as f32);
        }
    }

    sumf
}

/// 4-row unrolled `dotprod` GEMV for `Q8_0 x Q8_0`: loads `x` once across 4 weight rows and
/// reduces each 32-element block in exact `i32` (`vpaddq_s32`) before scaling (`0-ULP` bit-exact
/// with `generic::gemv_q8_0_q8_0` and `avx2::gemv_q8_0_q8_0`).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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
    // SAFETY: All row and block indices are strictly within bounds.
    unsafe {
        let zero = vdupq_n_s32(0);

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut acc4 = vdupq_n_f32(0.0);

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let yl = vld1q_s8(by.qs.as_ptr());
                let yh = vld1q_s8(by.qs.as_ptr().add(16));
                let dy = vdupq_n_f32(f16_to_f32_hw(by.d));

                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);

                let p0 = vdotq_s32(
                    vdotq_s32(zero, vld1q_s8(b0.qs.as_ptr()), yl),
                    vld1q_s8(b0.qs.as_ptr().add(16)),
                    yh,
                );
                let p1 = vdotq_s32(
                    vdotq_s32(zero, vld1q_s8(b1.qs.as_ptr()), yl),
                    vld1q_s8(b1.qs.as_ptr().add(16)),
                    yh,
                );
                let p2 = vdotq_s32(
                    vdotq_s32(zero, vld1q_s8(b2.qs.as_ptr()), yl),
                    vld1q_s8(b2.qs.as_ptr().add(16)),
                    yh,
                );
                let p3 = vdotq_s32(
                    vdotq_s32(zero, vld1q_s8(b3.qs.as_ptr()), yl),
                    vld1q_s8(b3.qs.as_ptr().add(16)),
                    yh,
                );

                let sum4 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
                let dx4 = vld1q_f32(
                    [
                        f16_to_f32_hw(b0.d),
                        f16_to_f32_hw(b1.d),
                        f16_to_f32_hw(b2.d),
                        f16_to_f32_hw(b3.d),
                    ]
                    .as_ptr(),
                );
                let scale4 = vmulq_f32(dx4, dy);
                acc4 = vaddq_f32(acc4, vmulq_f32(vcvtq_f32_s32(sum4), scale4));
            }

            vst1q_f32(out.as_mut_ptr().add(r), acc4);
            r += 4;
        }

        while r < rows {
            let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q8_0_q8_0(row, x);
            r += 1;
        }
    }
}

/// 4-row unrolled `dotprod` GEMV for `Q6_K x Q8_K`: loads `y` once across 4 weight rows
/// and accumulates scaled `int32x4_t` vectors before a single horizontal `vaddvq_s32` per super-block.
/// Accelerates 128k–262k vocab `lm_head` on Decode!
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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
    // SAFETY: All row and block indices are strictly within bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let m2b = vdupq_n_u8(0x03);
        let s32 = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let bx0 = w0.get_unchecked(ib);
                let bx1 = w1.get_unchecked(ib);
                let bx2 = w2.get_unchecked(ib);
                let bx3 = w3.get_unchecked(ib);

                let mut vsum0 = zero;
                let mut vsum1 = zero;
                let mut vsum2 = zero;
                let mut vsum3 = zero;

                for half in 0..2 {
                    let q8_ptr = by.qs.as_ptr().add(half * 128);
                    let sc_base = half * 8;

                    for chunk in 0..2 {
                        let off = chunk * 16;
                        let y0 = vld1q_s8(q8_ptr.add(off));
                        let y1 = vld1q_s8(q8_ptr.add(32 + off));
                        let y2 = vld1q_s8(q8_ptr.add(64 + off));
                        let y3 = vld1q_s8(q8_ptr.add(96 + off));

                        macro_rules! step_q6k {
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

                                let s0 = *$bx.scales.get_unchecked(sc_base + chunk) as i32;
                                let s1 = *$bx.scales.get_unchecked(sc_base + 2 + chunk) as i32;
                                let s2 = *$bx.scales.get_unchecked(sc_base + 4 + chunk) as i32;
                                let s3 = *$bx.scales.get_unchecked(sc_base + 6 + chunk) as i32;

                                $vsum = vmlaq_n_s32($vsum, vdotq_s32(zero, q0, y0), s0);
                                $vsum = vmlaq_n_s32($vsum, vdotq_s32(zero, q1, y1), s1);
                                $vsum = vmlaq_n_s32($vsum, vdotq_s32(zero, q2, y2), s2);
                                $vsum = vmlaq_n_s32($vsum, vdotq_s32(zero, q3, y3), s3);
                            }};
                        }

                        step_q6k!(bx0, vsum0);
                        step_q6k!(bx1, vsum1);
                        step_q6k!(bx2, vsum2);
                        step_q6k!(bx3, vsum3);
                    }
                }

                sum0 += f16_to_f32_hw(bx0.d) * by.d * (vaddvq_s32(vsum0) as f32);
                sum1 += f16_to_f32_hw(bx1.d) * by.d * (vaddvq_s32(vsum1) as f32);
                sum2 += f16_to_f32_hw(bx2.d) * by.d * (vaddvq_s32(vsum2) as f32);
                sum3 += f16_to_f32_hw(bx3.d) * by.d * (vaddvq_s32(vsum3) as f32);
            }

            *out.get_unchecked_mut(r) = sum0;
            *out.get_unchecked_mut(r + 1) = sum1;
            *out.get_unchecked_mut(r + 2) = sum2;
            *out.get_unchecked_mut(r + 3) = sum3;
            r += 4;
        }

        while r < rows {
            let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q6_k_q8_k(row, x);
            r += 1;
        }
    }
}

/// ARMv8.2 `dotprod` `vec_dot_q5_k_q8_k`.
/// Matches `ggml_vec_dot_q5_K_q8_K` (`ggml/src/ggml-cpu/arch/arm/quants.c`, MIT License).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and `x.len() == y.len()`.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_k_q8_k(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut sumf = 0.0f32;

    // SAFETY: All indices are strictly bounded by `nb` and `QK_K = 256`.
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
            let prod_m = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins_v)),
                vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins_v)),
            );
            let sumi_mins = vaddvq_s32(prod_m);

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

                let y0 = vld1q_s8(q8_ptr);
                let y1 = vld1q_s8(q8_ptr.add(16));
                let y2 = vld1q_s8(q8_ptr.add(32));
                let y3 = vld1q_s8(q8_ptr.add(48));

                let d_lo = vdotq_s32(vdotq_s32(zero, q0, y0), q1, y1);
                let d_hi = vdotq_s32(vdotq_s32(zero, q2, y2), q3, y3);

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

/// 2-row unrolled `dotprod` GEMV for `Q5_K x Q8_K`: loads `y` and `q8sums` once across 2 rows.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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
    // SAFETY: All row and block indices are strictly within bounds.
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

                    macro_rules! step_q5k {
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

                            let d_lo = vdotq_s32(vdotq_s32(zero, q0, y0), q1, y1);
                            let d_hi = vdotq_s32(vdotq_s32(zero, q2, y2), q3, y3);

                            $vsum = vmlaq_n_s32($vsum, d_lo, *$sc.get_unchecked(2 * chunk) as i32);
                            $vsum =
                                vmlaq_n_s32($vsum, d_hi, *$sc.get_unchecked(2 * chunk + 1) as i32);
                        }};
                    }

                    step_q5k!(bx0, sc0, qh0_a, qh0_b, vsum0);
                    step_q5k!(bx1, sc1, qh1_a, qh1_b, vsum1);
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
            let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q5_k_q8_k(row, x);
            r += 1;
        }
    }
}

/// ARMv8.2 `dotprod` 4-row GEMV for `BlockQ4_0x4 x BlockQ8_0` (Pi 5 / RK3588 Cortex-A76).
///
/// Reads a single contiguous 72-byte `BlockQ4_0x4` super-block per 4 output rows and matches
/// `generic::gemv_q4_0x4_q8_0` **bit-for-bit** (`to_bits()`).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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

    // SAFETY: All group and block indices are strictly bounded by `rows / 4` and `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        for rg in 0..(rows / 4) {
            let w_group = &w[rg * cols_blocks..(rg + 1) * cols_blocks];
            let mut acc4 = vdupq_n_f32(0.0);

            for ib in 0..cols_blocks {
                let sb = w_group.get_unchecked(ib);
                let by = x.get_unchecked(ib);
                let yl = vreinterpretq_s64_s8(vld1q_s8(by.qs.as_ptr()));
                let yh = vreinterpretq_s64_s8(vld1q_s8(by.qs.as_ptr().add(16)));
                let y00 = vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(yl));
                let y01 = vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(yl));
                let y10 = vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(yh));
                let y11 = vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(yh));

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

                let p01 = vdotq_s32(
                    vdotq_s32(vdotq_s32(vdotq_s32(zero, la0, y00), la1, y01), la2, y10),
                    la3,
                    y11,
                );
                let p23 = vdotq_s32(
                    vdotq_s32(vdotq_s32(vdotq_s32(zero, lb0, y00), lb1, y01), lb2, y10),
                    lb3,
                    y11,
                );
                let sumi4 = vpaddq_s32(p01, p23);

                let dy = f16_to_f32_hw(by.d);
                let scales = [
                    f16_to_f32_hw(sb.d[0]) * dy,
                    f16_to_f32_hw(sb.d[1]) * dy,
                    f16_to_f32_hw(sb.d[2]) * dy,
                    f16_to_f32_hw(sb.d[3]) * dy,
                ];
                acc4 = vfmaq_f32(acc4, vcvtq_f32_s32(sumi4), vld1q_f32(scales.as_ptr()));
            }

            vst1q_f32(out.as_mut_ptr().add(rg * 4), acc4);
        }
    }
}

/// ARMv8.2 `dotprod` `4-row x 4-token` register-blocked GEMM microkernel for `BlockQ4_0x4 x BlockQ8_0`
/// (Pi 5 / RK3588 Cortex-A76). Unpacks each 72-byte `BlockQ4_0x4` super-block ONCE across 4 tokens,
/// maintaining **100% bit-for-bit invariance** across all batch widths `1..=MAX_CPU_BATCH`.
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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

    // SAFETY: All group, token, and block indices are bounded by `rows / 4`, `n_tokens`, `cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(8);
        let zero = vdupq_n_s32(0);

        for rg in 0..(rows / 4) {
            let w_group = &w[rg * cols_blocks..(rg + 1) * cols_blocks];
            let r = rg * 4;
            let mut t = 0usize;

            while t + 4 <= n_tokens {
                let mut acc_t0 = vdupq_n_f32(0.0);
                let mut acc_t1 = vdupq_n_f32(0.0);
                let mut acc_t2 = vdupq_n_f32(0.0);
                let mut acc_t3 = vdupq_n_f32(0.0);

                for ib in 0..cols_blocks {
                    let sb = w_group.get_unchecked(ib);
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

                    let wd = vld1q_f32(
                        [
                            f16_to_f32_hw(sb.d[0]),
                            f16_to_f32_hw(sb.d[1]),
                            f16_to_f32_hw(sb.d[2]),
                            f16_to_f32_hw(sb.d[3]),
                        ]
                        .as_ptr(),
                    );

                    macro_rules! step_tok4x4 {
                        ($ti:expr, $acc:expr) => {{
                            let by = x.get_unchecked(($ti) * cols_blocks + ib);
                            let yl = vreinterpretq_s64_s8(vld1q_s8(by.qs.as_ptr()));
                            let yh = vreinterpretq_s64_s8(vld1q_s8(by.qs.as_ptr().add(16)));
                            let y00 = vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(yl));
                            let y01 = vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(yl));
                            let y10 = vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(yh));
                            let y11 = vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(yh));
                            let p01 = vdotq_s32(
                                vdotq_s32(vdotq_s32(vdotq_s32(zero, la0, y00), la1, y01), la2, y10),
                                la3,
                                y11,
                            );
                            let p23 = vdotq_s32(
                                vdotq_s32(vdotq_s32(vdotq_s32(zero, lb0, y00), lb1, y01), lb2, y10),
                                lb3,
                                y11,
                            );
                            let sumi4 = vpaddq_s32(p01, p23);
                            let dy = vdupq_n_f32(f16_to_f32_hw(by.d));
                            $acc = vfmaq_f32($acc, vcvtq_f32_s32(sumi4), vmulq_f32(wd, dy));
                        }};
                    }

                    step_tok4x4!(t, acc_t0);
                    step_tok4x4!(t + 1, acc_t1);
                    step_tok4x4!(t + 2, acc_t2);
                    step_tok4x4!(t + 3, acc_t3);
                }

                vst1q_f32(out.as_mut_ptr().add(t * rows + r), acc_t0);
                vst1q_f32(out.as_mut_ptr().add((t + 1) * rows + r), acc_t1);
                vst1q_f32(out.as_mut_ptr().add((t + 2) * rows + r), acc_t2);
                vst1q_f32(out.as_mut_ptr().add((t + 3) * rows + r), acc_t3);
                t += 4;
            }

            while t < n_tokens {
                let mut o4 = [0.0f32; 4];
                gemv_q4_0x4_q8_0(
                    w_group,
                    &x[t * cols_blocks..(t + 1) * cols_blocks],
                    &mut o4,
                    4,
                    cols_blocks,
                );
                out[t * rows + r..t * rows + r + 4].copy_from_slice(&o4);
                t += 1;
            }
        }
    }
}

/// Multi-token shared-weight `dotprod` GEMM for `Q8_0 x Q8_0`: loads each weight block once
/// across up to 4 tokens per inner loop (`0-ULP` bit-exact with `generic::gemm_q8_0_q8_0` and
/// `vec_dot_q8_0_q8_0` across all batch widths).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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

    if n_tokens == 1 {
        // SAFETY: Verified dimensions above.
        unsafe {
            gemv_q8_0_q8_0(w, x, out, rows, cols_blocks);
        }
        return;
    }

    // SAFETY: All indices are strictly bounded by `rows`, `cols_blocks`, `n_tokens`.
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut t0 = 0usize;
        while t0 + 4 <= n_tokens {
            let xt0 = &x[t0 * cols_blocks..(t0 + 1) * cols_blocks];
            let xt1 = &x[(t0 + 1) * cols_blocks..(t0 + 2) * cols_blocks];
            let xt2 = &x[(t0 + 2) * cols_blocks..(t0 + 3) * cols_blocks];
            let xt3 = &x[(t0 + 3) * cols_blocks..(t0 + 4) * cols_blocks];

            for r in 0..rows {
                let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
                let mut acc4 = vdupq_n_f32(0.0);

                for ib in 0..cols_blocks {
                    let xa = row.get_unchecked(ib);
                    let xal = vld1q_s8(xa.qs.as_ptr());
                    let xah = vld1q_s8(xa.qs.as_ptr().add(16));
                    let dx = vdupq_n_f32(f16_to_f32_hw(xa.d));

                    let y0 = xt0.get_unchecked(ib);
                    let y1 = xt1.get_unchecked(ib);
                    let y2 = xt2.get_unchecked(ib);
                    let y3 = xt3.get_unchecked(ib);

                    let p0 = vdotq_s32(
                        vdotq_s32(zero, xal, vld1q_s8(y0.qs.as_ptr())),
                        xah,
                        vld1q_s8(y0.qs.as_ptr().add(16)),
                    );
                    let p1 = vdotq_s32(
                        vdotq_s32(zero, xal, vld1q_s8(y1.qs.as_ptr())),
                        xah,
                        vld1q_s8(y1.qs.as_ptr().add(16)),
                    );
                    let p2 = vdotq_s32(
                        vdotq_s32(zero, xal, vld1q_s8(y2.qs.as_ptr())),
                        xah,
                        vld1q_s8(y2.qs.as_ptr().add(16)),
                    );
                    let p3 = vdotq_s32(
                        vdotq_s32(zero, xal, vld1q_s8(y3.qs.as_ptr())),
                        xah,
                        vld1q_s8(y3.qs.as_ptr().add(16)),
                    );

                    let sum4 = vpaddq_s32(vpaddq_s32(p0, p1), vpaddq_s32(p2, p3));
                    let dy4 = vld1q_f32(
                        [
                            f16_to_f32_hw(y0.d),
                            f16_to_f32_hw(y1.d),
                            f16_to_f32_hw(y2.d),
                            f16_to_f32_hw(y3.d),
                        ]
                        .as_ptr(),
                    );
                    let scale4 = vmulq_f32(dx, dy4);
                    acc4 = vaddq_f32(acc4, vmulq_f32(vcvtq_f32_s32(sum4), scale4));
                }

                let mut res = [0.0f32; 4];
                vst1q_f32(res.as_mut_ptr(), acc4);
                *out.get_unchecked_mut(t0 * rows + r) = res[0];
                *out.get_unchecked_mut((t0 + 1) * rows + r) = res[1];
                *out.get_unchecked_mut((t0 + 2) * rows + r) = res[2];
                *out.get_unchecked_mut((t0 + 3) * rows + r) = res[3];
            }
            t0 += 4;
        }

        while t0 < n_tokens {
            gemv_q8_0_q8_0(
                w,
                &x[t0 * cols_blocks..(t0 + 1) * cols_blocks],
                &mut out[t0 * rows..(t0 + 1) * rows],
                rows,
                cols_blocks,
            );
            t0 += 1;
        }
    }
}

/// Multi-token shared-weight `dotprod` GEMM for `Q6_K x Q8_K` (`ARCHITECTURE.md` §5.3 & §5.6):
/// unpacks each `BlockQ6K` super-block ONCE in registers and dots against up to 4 tokens per
/// inner loop (crucial for speculative verification on 128k–262k vocab `lm_head`!), maintaining
/// **100% bit-for-bit invariance** with `vec_dot_q6_k_q8_k` (`0` ULP).
///
/// # Safety
/// Caller must ensure CPU supports `dotprod` and buffer lengths match dimensions.
#[target_feature(enable = "neon,dotprod")]
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

    if n_tokens == 1 {
        // SAFETY: Verified dimensions above.
        unsafe {
            gemv_q6_k_q8_k(w, x, out, rows, cols_blocks);
        }
        return;
    }

    // SAFETY: All indices are strictly bounded by `rows`, `cols_blocks`, `n_tokens`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let m2b = vdupq_n_u8(0x03);
        let s32 = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);

        for r in 0..rows {
            let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let mut t0 = 0usize;

            while t0 + 4 <= n_tokens {
                let xt0 = &x[t0 * cols_blocks..(t0 + 1) * cols_blocks];
                let xt1 = &x[(t0 + 1) * cols_blocks..(t0 + 2) * cols_blocks];
                let xt2 = &x[(t0 + 2) * cols_blocks..(t0 + 3) * cols_blocks];
                let xt3 = &x[(t0 + 3) * cols_blocks..(t0 + 4) * cols_blocks];

                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut sum3 = 0.0f32;

                for ib in 0..cols_blocks {
                    let bx = row.get_unchecked(ib);
                    let dx = f16_to_f32_hw(bx.d);
                    let mut vsum0 = zero;
                    let mut vsum1 = zero;
                    let mut vsum2 = zero;
                    let mut vsum3 = zero;

                    for half in 0..2 {
                        let ql_ptr = bx.ql.as_ptr().add(half * 64);
                        let qh_ptr = bx.qh.as_ptr().add(half * 32);
                        let sc_base = half * 8;

                        for chunk in 0..2 {
                            let off = chunk * 16;
                            let ql0 = vld1q_u8(ql_ptr.add(off));
                            let ql1 = vld1q_u8(ql_ptr.add(32 + off));
                            let qh = vld1q_u8(qh_ptr.add(off));

                            // Unpack 64 6-bit weights ONCE for all 4 tokens!
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

                            let s0 = *bx.scales.get_unchecked(sc_base + chunk) as i32;
                            let s1 = *bx.scales.get_unchecked(sc_base + 2 + chunk) as i32;
                            let s2 = *bx.scales.get_unchecked(sc_base + 4 + chunk) as i32;
                            let s3 = *bx.scales.get_unchecked(sc_base + 6 + chunk) as i32;

                            macro_rules! dot_tok6k {
                                ($xt:expr, $vsum:expr) => {{
                                    let q8_ptr =
                                        $xt.get_unchecked(ib).qs.as_ptr().add(half * 128 + off);
                                    $vsum = vmlaq_n_s32(
                                        $vsum,
                                        vdotq_s32(zero, q0, vld1q_s8(q8_ptr)),
                                        s0,
                                    );
                                    $vsum = vmlaq_n_s32(
                                        $vsum,
                                        vdotq_s32(zero, q1, vld1q_s8(q8_ptr.add(32))),
                                        s1,
                                    );
                                    $vsum = vmlaq_n_s32(
                                        $vsum,
                                        vdotq_s32(zero, q2, vld1q_s8(q8_ptr.add(64))),
                                        s2,
                                    );
                                    $vsum = vmlaq_n_s32(
                                        $vsum,
                                        vdotq_s32(zero, q3, vld1q_s8(q8_ptr.add(96))),
                                        s3,
                                    );
                                }};
                            }

                            dot_tok6k!(xt0, vsum0);
                            dot_tok6k!(xt1, vsum1);
                            dot_tok6k!(xt2, vsum2);
                            dot_tok6k!(xt3, vsum3);
                        }
                    }

                    sum0 += dx * xt0.get_unchecked(ib).d * (vaddvq_s32(vsum0) as f32);
                    sum1 += dx * xt1.get_unchecked(ib).d * (vaddvq_s32(vsum1) as f32);
                    sum2 += dx * xt2.get_unchecked(ib).d * (vaddvq_s32(vsum2) as f32);
                    sum3 += dx * xt3.get_unchecked(ib).d * (vaddvq_s32(vsum3) as f32);
                }

                *out.get_unchecked_mut(t0 * rows + r) = sum0;
                *out.get_unchecked_mut((t0 + 1) * rows + r) = sum1;
                *out.get_unchecked_mut((t0 + 2) * rows + r) = sum2;
                *out.get_unchecked_mut((t0 + 3) * rows + r) = sum3;
                t0 += 4;
            }

            while t0 < n_tokens {
                *out.get_unchecked_mut(t0 * rows + r) =
                    vec_dot_q6_k_q8_k(row, &x[t0 * cols_blocks..(t0 + 1) * cols_blocks]);
                t0 += 1;
            }
        }
    }
}
