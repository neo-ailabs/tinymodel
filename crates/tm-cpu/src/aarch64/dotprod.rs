//! ARMv8.2 NEON `dotprod` (`vdotq_s32`) kernels for `Q4_0 x Q8_0`.
//!
//! Ported from ggml / Strata:
//! - Copyright (c) 2023-2026 The ggml authors (MIT License)
//! - Copyright (c) 2026 Niko1221 and the Strata contributors (MIT License)
//! - Upstream reference: `Strata/src/quant/arm/arm_dots.inc:317-377`
//!   (ggml `ggml/src/ggml-cpu/arch/arm/quants.c` commit `3cf03257`).

use core::arch::aarch64::*;
use tm_core::{BlockQ4_0, BlockQ8_0};

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
