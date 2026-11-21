// Portions of codebook & block math adapted from llama.cpp `ggml/src/ggml-cpu/quants.c`
// and `Strata/src/quant/dot_generic.cpp` (commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`),
// Copyright (c) 2023-2026 The ggml authors & Niko1221 / Strata contributors, MIT License.
// 16-row T-MAC `vqtbl1q_u8`/`vqtbl1q_s8` activation-LUT algorithm written from scratch for `tinymodel`.

//! ARMv8.0 NEON `TBL` (`vqtbl1q_s8` / `vqtbl1q_u8`) lookup-table kernels for `micro` & `edge` tiers:
//! - **T-MAC 16-row Activation-LUT GEMV & GEMM** for `Q2_0` (2.25 bpw) and BitNet `TQ2_0` (2.06 bpw):
//!   precomputes 16-entry signed `i16` partial-dot tables `(lo: u8, hi: i8)` from quantized activations
//!   ONCE per K-tile on stack (zero heap allocations, L1D-resident on Cortex-A53's 32 KiB cache),
//!   evaluating 64 multiply-accumulates across 16 output rows with **zero multiplications** via `vqtbl1q`!
//! - **NEON Codebook + Parity-Sign XOR** kernels for `IQ2_XXS` (2.06 bpw), `IQ3_XXS` (3.06 bpw),
//!   and `IQ4_NL` (`vqtbl1q_s8` 1-cycle non-linear 4-bit codebook lookup), plus `dotprod` `sdot` paths.
//! - Every kernel is **0-ULP bit-for-bit identical** to `crate::generic` across all scales and batch widths!

use core::arch::aarch64::*;

use tm_core::{
    BlockIQ2XXS, BlockIQ3XXS, BlockIQ4NL, BlockQ2_0, BlockQ8_0, BlockQ8K, BlockTQ2_0, IQ2XXS_GRID,
    IQ3XXS_GRID, KSIGNS64, KVALUES_IQ4NL,
};

use crate::aarch64::asm::f16_to_f32_hw;

/// Precomputed 16-entry signed `i16` table `val[nib] = ((nib & 3) - 1) * y0 + (((nib >> 2) & 3) - 1) * y1`
/// split into unsigned low bytes `lo[nib] = (val as u16 & 0xFF) as u8` and signed high bytes `hi[nib] = (val >> 8) as i8`
/// so `(hi as i16) * 256 + (lo as i16) == val` holds identically for all two's complement `i16`!
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct LutPair16 {
    lo: [u8; 16],
    hi: [i8; 16],
}

impl Default for LutPair16 {
    #[inline]
    fn default() -> Self {
        Self {
            lo: [0u8; 16],
            hi: [0i8; 16],
        }
    }
}

#[inline]
fn build_lut_pair_2bit(y0: i8, y1: i8) -> LutPair16 {
    let a = y0 as i32;
    let b = y1 as i32;
    let mut lo = [0u8; 16];
    let mut hi = [0i8; 16];
    let mut nib = 0usize;
    while nib < 16 {
        let q0 = ((nib & 3) as i32) - 1;
        let q1 = (((nib >> 2) & 3) as i32) - 1;
        let val = (q0 * a + q1 * b) as i16;
        lo[nib] = (val as u16 & 0x00FF) as u8;
        hi[nib] = (val >> 8) as i8;
        nib += 1;
    }
    LutPair16 { lo, hi }
}

/// In-register 16×16 byte transpose (`rows[m]` = 16 contiguous bytes of row `m` -> `cols[b]` = byte `b` across all 16 rows).
/// Eliminates 16× strided scalar `ldrb` loads and L1D 4-way set conflicts on Cortex-A53 / Cortex-A510!
#[target_feature(enable = "neon")]
#[inline]
unsafe fn transpose_16x16_u8(r: &[uint8x16_t; 16]) -> [uint8x16_t; 16] {
    let t0 = vtrn1q_u8(r[0], r[1]);
    let t1 = vtrn2q_u8(r[0], r[1]);
    let t2 = vtrn1q_u8(r[2], r[3]);
    let t3 = vtrn2q_u8(r[2], r[3]);
    let t4 = vtrn1q_u8(r[4], r[5]);
    let t5 = vtrn2q_u8(r[4], r[5]);
    let t6 = vtrn1q_u8(r[6], r[7]);
    let t7 = vtrn2q_u8(r[6], r[7]);
    let t8 = vtrn1q_u8(r[8], r[9]);
    let t9 = vtrn2q_u8(r[8], r[9]);
    let t10 = vtrn1q_u8(r[10], r[11]);
    let t11 = vtrn2q_u8(r[10], r[11]);
    let t12 = vtrn1q_u8(r[12], r[13]);
    let t13 = vtrn2q_u8(r[12], r[13]);
    let t14 = vtrn1q_u8(r[14], r[15]);
    let t15 = vtrn2q_u8(r[14], r[15]);

    let u0 = vtrn1q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2));
    let u2 = vtrn2q_u16(vreinterpretq_u16_u8(t0), vreinterpretq_u16_u8(t2));
    let u1 = vtrn1q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3));
    let u3 = vtrn2q_u16(vreinterpretq_u16_u8(t1), vreinterpretq_u16_u8(t3));
    let u4 = vtrn1q_u16(vreinterpretq_u16_u8(t4), vreinterpretq_u16_u8(t6));
    let u6 = vtrn2q_u16(vreinterpretq_u16_u8(t4), vreinterpretq_u16_u8(t6));
    let u5 = vtrn1q_u16(vreinterpretq_u16_u8(t5), vreinterpretq_u16_u8(t7));
    let u7 = vtrn2q_u16(vreinterpretq_u16_u8(t5), vreinterpretq_u16_u8(t7));
    let u8 = vtrn1q_u16(vreinterpretq_u16_u8(t8), vreinterpretq_u16_u8(t10));
    let u10 = vtrn2q_u16(vreinterpretq_u16_u8(t8), vreinterpretq_u16_u8(t10));
    let u9 = vtrn1q_u16(vreinterpretq_u16_u8(t9), vreinterpretq_u16_u8(t11));
    let u11 = vtrn2q_u16(vreinterpretq_u16_u8(t9), vreinterpretq_u16_u8(t11));
    let u12 = vtrn1q_u16(vreinterpretq_u16_u8(t12), vreinterpretq_u16_u8(t14));
    let u14 = vtrn2q_u16(vreinterpretq_u16_u8(t12), vreinterpretq_u16_u8(t14));
    let u13 = vtrn1q_u16(vreinterpretq_u16_u8(t13), vreinterpretq_u16_u8(t15));
    let u15 = vtrn2q_u16(vreinterpretq_u16_u8(t13), vreinterpretq_u16_u8(t15));

    let v0 = vtrn1q_u32(vreinterpretq_u32_u16(u0), vreinterpretq_u32_u16(u4));
    let v4 = vtrn2q_u32(vreinterpretq_u32_u16(u0), vreinterpretq_u32_u16(u4));
    let v1 = vtrn1q_u32(vreinterpretq_u32_u16(u1), vreinterpretq_u32_u16(u5));
    let v5 = vtrn2q_u32(vreinterpretq_u32_u16(u1), vreinterpretq_u32_u16(u5));
    let v2 = vtrn1q_u32(vreinterpretq_u32_u16(u2), vreinterpretq_u32_u16(u6));
    let v6 = vtrn2q_u32(vreinterpretq_u32_u16(u2), vreinterpretq_u32_u16(u6));
    let v3 = vtrn1q_u32(vreinterpretq_u32_u16(u3), vreinterpretq_u32_u16(u7));
    let v7 = vtrn2q_u32(vreinterpretq_u32_u16(u3), vreinterpretq_u32_u16(u7));
    let v8 = vtrn1q_u32(vreinterpretq_u32_u16(u8), vreinterpretq_u32_u16(u12));
    let v12 = vtrn2q_u32(vreinterpretq_u32_u16(u8), vreinterpretq_u32_u16(u12));
    let v9 = vtrn1q_u32(vreinterpretq_u32_u16(u9), vreinterpretq_u32_u16(u13));
    let v13 = vtrn2q_u32(vreinterpretq_u32_u16(u9), vreinterpretq_u32_u16(u13));
    let v10 = vtrn1q_u32(vreinterpretq_u32_u16(u10), vreinterpretq_u32_u16(u14));
    let v14 = vtrn2q_u32(vreinterpretq_u32_u16(u10), vreinterpretq_u32_u16(u14));
    let v11 = vtrn1q_u32(vreinterpretq_u32_u16(u11), vreinterpretq_u32_u16(u15));
    let v15 = vtrn2q_u32(vreinterpretq_u32_u16(u11), vreinterpretq_u32_u16(u15));

    [
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v0),
            vreinterpretq_u64_u32(v8),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v1),
            vreinterpretq_u64_u32(v9),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v2),
            vreinterpretq_u64_u32(v10),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v3),
            vreinterpretq_u64_u32(v11),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v4),
            vreinterpretq_u64_u32(v12),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v5),
            vreinterpretq_u64_u32(v13),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v6),
            vreinterpretq_u64_u32(v14),
        )),
        vreinterpretq_u8_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(v7),
            vreinterpretq_u64_u32(v15),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v0),
            vreinterpretq_u64_u32(v8),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v1),
            vreinterpretq_u64_u32(v9),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v2),
            vreinterpretq_u64_u32(v10),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v3),
            vreinterpretq_u64_u32(v11),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v4),
            vreinterpretq_u64_u32(v12),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v5),
            vreinterpretq_u64_u32(v13),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v6),
            vreinterpretq_u64_u32(v14),
        )),
        vreinterpretq_u8_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(v7),
            vreinterpretq_u64_u32(v15),
        )),
    ]
}

/// Widens and sums `sum_lo: uint16x8x2` + `(sum_hi: int16x8x2) << 8` into four `int32x4_t` vectors `[r0..3, r4..7, r8..11, r12..15]`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn finalize_lut16_i32(
    lo_07: uint16x8_t,
    lo_8f: uint16x8_t,
    hi_07: int16x8_t,
    hi_8f: int16x8_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let u0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo_07)));
    let u1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo_07)));
    let u2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo_8f)));
    let u3 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo_8f)));

    let h0 = vshlq_n_s32::<8>(vmovl_s16(vget_low_s16(hi_07)));
    let h1 = vshlq_n_s32::<8>(vmovl_s16(vget_high_s16(hi_07)));
    let h2 = vshlq_n_s32::<8>(vmovl_s16(vget_low_s16(hi_8f)));
    let h3 = vshlq_n_s32::<8>(vmovl_s16(vget_high_s16(hi_8f)));

    (
        vaddq_s32(h0, u0),
        vaddq_s32(h1, u1),
        vaddq_s32(h2, u2),
        vaddq_s32(h3, u3),
    )
}

/// Base ARMv8.0 NEON single-row `BlockQ2_0 x BlockQ8_0` dot product using `vqtbl1q_s8` 2-bit decode.
///
/// # Safety
/// Caller must ensure NEON is available and `y.len() == x.len() * 2`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q2_0_q8_0(x: &[BlockQ2_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(y.len(), x.len() * 2);
    // SAFETY: All slice accesses are bounded by `x.len()` and fixed block widths (`16` and `32` bytes).
    unsafe {
        let code_lut = vld1q_s8([-1i8, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr());
        let m2b = vdupq_n_u8(0x03);
        let mut sumf = 0.0f32;

        for (ib, bx) in x.iter().enumerate() {
            let d0 = f16_to_f32_hw(bx.d);
            let yb0 = y.get_unchecked(ib * 2);
            let yb1 = y.get_unchecked(ib * 2 + 1);

            let raw16 = vld1q_u8(bx.qs.as_ptr());
            let c0 = vqtbl1q_s8(code_lut, vandq_u8(raw16, m2b));
            let c1 = vqtbl1q_s8(code_lut, vandq_u8(vshrq_n_u8::<2>(raw16), m2b));
            let c2 = vqtbl1q_s8(code_lut, vandq_u8(vshrq_n_u8::<4>(raw16), m2b));
            let c3 = vqtbl1q_s8(code_lut, vshrq_n_u8::<6>(raw16));

            // Transpose 4x16 (`c0, c1, c2, c3` at stride 4) to match contiguous `yb0.qs[0..32]` and `yb1.qs[0..32]`:
            // Or equivalently deinterleave `yb` by 4 via `vld4_s8` (`4` contiguous 8-byte lanes for `b=0..8`!)!
            let y0_4 = vld4_s8(yb0.qs.as_ptr());
            let mut p0 = vmull_s8(vget_low_s8(c0), y0_4.0);
            p0 = vmlal_s8(p0, vget_low_s8(c1), y0_4.1);
            p0 = vmlal_s8(p0, vget_low_s8(c2), y0_4.2);
            p0 = vmlal_s8(p0, vget_low_s8(c3), y0_4.3);
            let s0 = vaddlvq_s16(p0);

            let y1_4 = vld4_s8(yb1.qs.as_ptr());
            let mut p1 = vmull_s8(vget_high_s8(c0), y1_4.0);
            p1 = vmlal_s8(p1, vget_high_s8(c1), y1_4.1);
            p1 = vmlal_s8(p1, vget_high_s8(c2), y1_4.2);
            p1 = vmlal_s8(p1, vget_high_s8(c3), y1_4.3);
            let s1 = vaddlvq_s16(p1);

            let sub = (s0 as f32) * f16_to_f32_hw(yb0.d) + (s1 as f32) * f16_to_f32_hw(yb1.d);
            sumf = d0.mul_add(sub, sumf);
        }
        sumf
    }
}

/// ARMv8.0 NEON **16-row T-MAC Activation-LUT** `Q2_0 x Q8_0` GEMV (`vqtbl1q_u8` / `vqtbl1q_s8`,
/// zero multiplications in the inner 16-row weight loop, 0-ULP bit-exact with `generic::gemv_q2_0_q8_0`!).
///
/// # Safety
/// Caller must ensure NEON is available and slice dimensions match `rows` and `cols_blocks`.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_q2_0_q8_0(
    w: &[BlockQ2_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks * 2);
    assert_eq!(out.len(), rows);

    const K_TILE: usize = 8; // 8 `BlockQ2_0` = 512 cols -> `8 * 32 = 256` `LutPair16` = 8 KiB in L1D!

    // SAFETY: Bounds checked above; `r + 16 <= rows` and `ib < cols_blocks` strictly respected.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let mut lut_buf = [LutPair16::default(); K_TILE * 32];

        let mut kb = 0usize;
        // Initialize output to 0 before K-tiled accumulation over 16-row groups
        let full16_end = (rows / 16) * 16;
        out[..full16_end].fill(0.0);

        while kb < cols_blocks && full16_end > 0 {
            let kend = (kb + K_TILE).min(cols_blocks);
            let klen = kend - kb;

            // Precompute activation LUTs ONCE for all `rows` in this K-tile (`0` heap allocs!)
            for ki in 0..klen {
                let ib = kb + ki;
                for sub in 0..2 {
                    let yb = x.get_unchecked(ib * 2 + sub);
                    let base_lut = ki * 32 + sub * 16;
                    for b in 0..8 {
                        lut_buf[base_lut + b * 2] =
                            build_lut_pair_2bit(yb.qs[b * 4], yb.qs[b * 4 + 1]);
                        lut_buf[base_lut + b * 2 + 1] =
                            build_lut_pair_2bit(yb.qs[b * 4 + 2], yb.qs[b * 4 + 3]);
                    }
                }
            }

            let mut r = 0usize;
            while r + 16 <= rows {
                let op = out.as_mut_ptr().add(r);
                let mut sum0 = vld1q_f32(op);
                let mut sum1 = vld1q_f32(op.add(4));
                let mut sum2 = vld1q_f32(op.add(8));
                let mut sum3 = vld1q_f32(op.add(12));

                for ki in 0..klen {
                    let ib = kb + ki;
                    let dy0 = vdupq_n_f32(f16_to_f32_hw(x.get_unchecked(ib * 2).d));
                    let dy1 = vdupq_n_f32(f16_to_f32_hw(x.get_unchecked(ib * 2 + 1).d));

                    // Load all 16 contiguous `qs` bytes per row in one `vld1q_u8` and transpose 16x16 in NEON!
                    let mut d_arr = [0.0f32; 16];
                    let mut raw_rows = [vdupq_n_u8(0); 16];
                    for m in 0..16 {
                        let bx = w.get_unchecked((r + m) * cols_blocks + ib);
                        d_arr[m] = f16_to_f32_hw(bx.d);
                        raw_rows[m] = vld1q_u8(bx.qs.as_ptr());
                    }
                    let cols16 = transpose_16x16_u8(&raw_rows);

                    macro_rules! eval_sub8 {
                        ($sub_idx:expr) => {{
                            let mut lo_07 = vdupq_n_u16(0);
                            let mut lo_8f = vdupq_n_u16(0);
                            let mut hi_07 = vdupq_n_s16(0);
                            let mut hi_8f = vdupq_n_s16(0);
                            let byte_base = ($sub_idx) * 8;
                            let lut_base = ki * 32 + ($sub_idx) * 16;

                            for b in 0..8 {
                                let vbytes = cols16[byte_base + b];
                                let nib_lo = vandq_u8(vbytes, m4b);
                                let nib_hi = vshrq_n_u8::<4>(vbytes);

                                let lp0 = lut_buf.get_unchecked(lut_base + b * 2);
                                let lp1 = lut_buf.get_unchecked(lut_base + b * 2 + 1);

                                let a_lo = vqtbl1q_u8(vld1q_u8(lp0.lo.as_ptr()), nib_lo);
                                let a_hi = vqtbl1q_s8(vld1q_s8(lp0.hi.as_ptr()), nib_lo);
                                let b_lo = vqtbl1q_u8(vld1q_u8(lp1.lo.as_ptr()), nib_hi);
                                let b_hi = vqtbl1q_s8(vld1q_s8(lp1.hi.as_ptr()), nib_hi);

                                let sum_u16_07 = vaddl_u8(vget_low_u8(a_lo), vget_low_u8(b_lo));
                                let sum_u16_8f = vaddl_u8(vget_high_u8(a_lo), vget_high_u8(b_lo));
                                lo_07 = vaddq_u16(lo_07, sum_u16_07);
                                lo_8f = vaddq_u16(lo_8f, sum_u16_8f);

                                let sum_s16_07 = vaddl_s8(vget_low_s8(a_hi), vget_low_s8(b_hi));
                                let sum_s16_8f = vaddl_s8(vget_high_s8(a_hi), vget_high_s8(b_hi));
                                hi_07 = vaddq_s16(hi_07, sum_s16_07);
                                hi_8f = vaddq_s16(hi_8f, sum_s16_8f);
                            }
                            finalize_lut16_i32(lo_07, lo_8f, hi_07, hi_8f)
                        }};
                    }

                    let (s0_0, s0_1, s0_2, s0_3) = eval_sub8!(0);
                    let (s1_0, s1_1, s1_2, s1_3) = eval_sub8!(1);

                    let sub0 = vaddq_f32(
                        vmulq_f32(vcvtq_f32_s32(s0_0), dy0),
                        vmulq_f32(vcvtq_f32_s32(s1_0), dy1),
                    );
                    let sub1 = vaddq_f32(
                        vmulq_f32(vcvtq_f32_s32(s0_1), dy0),
                        vmulq_f32(vcvtq_f32_s32(s1_1), dy1),
                    );
                    let sub2 = vaddq_f32(
                        vmulq_f32(vcvtq_f32_s32(s0_2), dy0),
                        vmulq_f32(vcvtq_f32_s32(s1_2), dy1),
                    );
                    let sub3 = vaddq_f32(
                        vmulq_f32(vcvtq_f32_s32(s0_3), dy0),
                        vmulq_f32(vcvtq_f32_s32(s1_3), dy1),
                    );

                    sum0 = vfmaq_f32(sum0, vld1q_f32(d_arr.as_ptr()), sub0);
                    sum1 = vfmaq_f32(sum1, vld1q_f32(d_arr.as_ptr().add(4)), sub1);
                    sum2 = vfmaq_f32(sum2, vld1q_f32(d_arr.as_ptr().add(8)), sub2);
                    sum3 = vfmaq_f32(sum3, vld1q_f32(d_arr.as_ptr().add(12)), sub3);
                }

                vst1q_f32(op, sum0);
                vst1q_f32(op.add(4), sum1);
                vst1q_f32(op.add(8), sum2);
                vst1q_f32(op.add(12), sum3);
                r += 16;
            }

            kb = kend;
        }

        for r in full16_end..rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q2_0_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
        }
    }
}

/// Multi-token `Q2_0 x Q8_0` GEMM (0-ULP bit-exact with `generic::gemm_q2_0_q8_0`).
///
/// # Safety
/// Caller must ensure NEON is available and slice dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_q2_0_q8_0(
    w: &[BlockQ2_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    let q8_blocks = cols_blocks * 2;
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * q8_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Each token `t` runs `gemv_q2_0_q8_0` on its disjoint activation/output slice.
    unsafe {
        for t in 0..n_tokens {
            gemv_q2_0_q8_0(
                w,
                &x[t * q8_blocks..(t + 1) * q8_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// Base ARMv8.0 NEON single-row BitNet b1.58 ternary `BlockTQ2_0 x BlockQ8K` dot product
/// using `vqtbl1q_s8` trit decoding (`{-1, 0, +1}`).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_tq2_0_q8_k(x: &[BlockTQ2_0], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: Fixed 256-element block indices (`64` weight bytes and `256` activation bytes).
    unsafe {
        let trit_lut = vld1q_s8([-1i8, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr());
        let m2b = vdupq_n_u8(0x03);
        let mut sumf = 0.0f32;

        for (bx, by) in x.iter().zip(y.iter()) {
            let mut sumi_vec = vdupq_n_s32(0);

            for half in 0..2 {
                let qp = bx.qs.as_ptr().add(half * 32);
                let yp = by.qs.as_ptr().add(half * 128);

                for seg in 0..2 {
                    let raw = vld1q_u8(qp.add(seg * 16));
                    let t0 = vqtbl1q_s8(trit_lut, vandq_u8(raw, m2b));
                    let t1 = vqtbl1q_s8(trit_lut, vandq_u8(vshrq_n_u8::<2>(raw), m2b));
                    let t2 = vqtbl1q_s8(trit_lut, vandq_u8(vshrq_n_u8::<4>(raw), m2b));
                    let t3 = vqtbl1q_s8(trit_lut, vshrq_n_u8::<6>(raw));

                    let y0 = vld1q_s8(yp.add(seg * 16));
                    let y1 = vld1q_s8(yp.add(32 + seg * 16));
                    let y2 = vld1q_s8(yp.add(64 + seg * 16));
                    let y3 = vld1q_s8(yp.add(96 + seg * 16));

                    let mut p = vmull_s8(vget_low_s8(t0), vget_low_s8(y0));
                    p = vmlal_s8(p, vget_high_s8(t0), vget_high_s8(y0));
                    p = vmlal_s8(p, vget_low_s8(t1), vget_low_s8(y1));
                    p = vmlal_s8(p, vget_high_s8(t1), vget_high_s8(y1));
                    sumi_vec = vpadalq_s16(sumi_vec, p);

                    let mut q = vmull_s8(vget_low_s8(t2), vget_low_s8(y2));
                    q = vmlal_s8(q, vget_high_s8(t2), vget_high_s8(y2));
                    q = vmlal_s8(q, vget_low_s8(t3), vget_low_s8(y3));
                    q = vmlal_s8(q, vget_high_s8(t3), vget_high_s8(y3));
                    sumi_vec = vpadalq_s16(sumi_vec, q);
                }
            }

            let sumi = vaddvq_s32(sumi_vec);
            let d = f16_to_f32_hw(bx.d) * by.d;
            sumf = (sumi as f32).mul_add(d, sumf);
        }
        sumf
    }
}

/// ARMv8.0 NEON **16-row T-MAC Ternary Activation-LUT** `TQ2_0 x Q8_K` GEMV:
/// evaluates 64 ternary multiply-accumulates per 16-byte load using ONLY `vqtbl1q_u8`/`vqtbl1q_s8`
/// and byte-widening adds (ZERO multiplications in the inner weight loop, 0-ULP bit-exact!).
///
/// # Safety
/// Caller must ensure NEON is available and slice dimensions match `rows` and `cols_blocks`.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_tq2_0_q8_k(
    w: &[BlockTQ2_0],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    const K_TILE: usize = 4; // 4 `BlockTQ2_0` = 1024 cols -> `4 * 128 = 512` `LutPair16` = 16 KiB in L1D!

    // SAFETY: Bounds checked above; `r + 16 <= rows` and `ib < cols_blocks`.
    unsafe {
        let m4b = vdupq_n_u8(0x0F);
        let mut lut_buf = [LutPair16::default(); K_TILE * 128];

        let full16_end = (rows / 16) * 16;
        out[..full16_end].fill(0.0);

        let mut kb = 0usize;
        while kb < cols_blocks && full16_end > 0 {
            let kend = (kb + K_TILE).min(cols_blocks);
            let klen = kend - kb;

            // Precompute 128 `LutPair16` tables per `BlockQ8K` ONCE on stack
            for ki in 0..klen {
                let by = x.get_unchecked(kb + ki);
                for half in 0..2 {
                    let qy = &by.qs[half * 128..(half + 1) * 128];
                    let base_lut = ki * 128 + half * 64;
                    for k in 0..32 {
                        lut_buf[base_lut + k * 2] = build_lut_pair_2bit(qy[k], qy[32 + k]);
                        lut_buf[base_lut + k * 2 + 1] = build_lut_pair_2bit(qy[64 + k], qy[96 + k]);
                    }
                }
            }

            let mut r = 0usize;
            while r + 16 <= rows {
                let op = out.as_mut_ptr().add(r);
                let mut sum0 = vld1q_f32(op);
                let mut sum1 = vld1q_f32(op.add(4));
                let mut sum2 = vld1q_f32(op.add(8));
                let mut sum3 = vld1q_f32(op.add(12));

                for ki in 0..klen {
                    let ib = kb + ki;
                    let dy = vdupq_n_f32(x.get_unchecked(ib).d);
                    let mut d_arr = [0.0f32; 16];
                    for (m, d_slot) in d_arr.iter_mut().enumerate() {
                        *d_slot = f16_to_f32_hw(w.get_unchecked((r + m) * cols_blocks + ib).d);
                    }

                    // Over `64` bytes (`128` lookups per row):
                    // `128 * 255 = 32,640 <= 65,535` fits in a SINGLE `uint16x8_t` WITHOUT overflow!
                    // `128 * 2   = 256   <= 32,767` fits in a SINGLE `int16x8_t` WITHOUT overflow!
                    let mut lo_07 = vdupq_n_u16(0);
                    let mut lo_8f = vdupq_n_u16(0);
                    let mut hi_07 = vdupq_n_s16(0);
                    let mut hi_8f = vdupq_n_s16(0);
                    let lut_block_base = ki * 128;

                    for seg in 0..4 {
                        let seg_off = seg * 16;
                        let mut raw_rows = [vdupq_n_u8(0); 16];
                        for (m, row_slot) in raw_rows.iter_mut().enumerate() {
                            *row_slot = vld1q_u8(
                                w.get_unchecked((r + m) * cols_blocks + ib)
                                    .qs
                                    .as_ptr()
                                    .add(seg_off),
                            );
                        }
                        let cols16 = transpose_16x16_u8(&raw_rows);

                        for (b, &vbytes) in cols16.iter().enumerate() {
                            let byte_idx = seg_off + b;
                            let nib_lo = vandq_u8(vbytes, m4b);
                            let nib_hi = vshrq_n_u8::<4>(vbytes);

                            let lp0 = lut_buf.get_unchecked(lut_block_base + byte_idx * 2);
                            let lp1 = lut_buf.get_unchecked(lut_block_base + byte_idx * 2 + 1);

                            let a_lo = vqtbl1q_u8(vld1q_u8(lp0.lo.as_ptr()), nib_lo);
                            let a_hi = vqtbl1q_s8(vld1q_s8(lp0.hi.as_ptr()), nib_lo);
                            let b_lo = vqtbl1q_u8(vld1q_u8(lp1.lo.as_ptr()), nib_hi);
                            let b_hi = vqtbl1q_s8(vld1q_s8(lp1.hi.as_ptr()), nib_hi);

                            lo_07 =
                                vaddq_u16(lo_07, vaddl_u8(vget_low_u8(a_lo), vget_low_u8(b_lo)));
                            lo_8f =
                                vaddq_u16(lo_8f, vaddl_u8(vget_high_u8(a_lo), vget_high_u8(b_lo)));
                            hi_07 =
                                vaddq_s16(hi_07, vaddl_s8(vget_low_s8(a_hi), vget_low_s8(b_hi)));
                            hi_8f =
                                vaddq_s16(hi_8f, vaddl_s8(vget_high_s8(a_hi), vget_high_s8(b_hi)));
                        }
                    }

                    let (i0, i1, i2, i3) = finalize_lut16_i32(lo_07, lo_8f, hi_07, hi_8f);
                    let sc0 = vmulq_f32(vld1q_f32(d_arr.as_ptr()), dy);
                    let sc1 = vmulq_f32(vld1q_f32(d_arr.as_ptr().add(4)), dy);
                    let sc2 = vmulq_f32(vld1q_f32(d_arr.as_ptr().add(8)), dy);
                    let sc3 = vmulq_f32(vld1q_f32(d_arr.as_ptr().add(12)), dy);

                    sum0 = vfmaq_f32(sum0, vcvtq_f32_s32(i0), sc0);
                    sum1 = vfmaq_f32(sum1, vcvtq_f32_s32(i1), sc1);
                    sum2 = vfmaq_f32(sum2, vcvtq_f32_s32(i2), sc2);
                    sum3 = vfmaq_f32(sum3, vcvtq_f32_s32(i3), sc3);
                }

                vst1q_f32(op, sum0);
                vst1q_f32(op.add(4), sum1);
                vst1q_f32(op.add(8), sum2);
                vst1q_f32(op.add(12), sum3);
                r += 16;
            }

            kb = kend;
        }

        for r in full16_end..rows {
            *out.get_unchecked_mut(r) =
                vec_dot_tq2_0_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
        }
    }
}

/// Multi-token `TQ2_0 x Q8_K` GEMM (0-ULP bit-exact with `generic::gemm_tq2_0_q8_k`).
///
/// # Safety
/// Caller must ensure NEON is available and slice dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_tq2_0_q8_k(
    w: &[BlockTQ2_0],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified dimensions above.
    unsafe {
        for t in 0..n_tokens {
            gemv_tq2_0_q8_k(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// Base ARMv8.0 NEON `BlockIQ2XXS x BlockQ8K` dot product using `KSIGNS64` two's-complement
/// branchless sign application `(g ^ sm) - sm` + `vmull_s8`/`vpadalq_s16` (0-ULP bit-exact!).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_iq2_xxs_q8_k(x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: Fixed 256-element block geometry; all `IQ2XXS_GRID[u8]` and `KSIGNS64[0..128]` indices are bounded.
    unsafe {
        let mut sumf = 0.0f32;
        for (bx, by) in x.iter().zip(y.iter()) {
            let mut bsum: i32 = 0;
            let yp_base = by.qs.as_ptr();

            for ib32 in 0..8 {
                let aux0 = (bx.qs[ib32 * 4] as u32) | ((bx.qs[ib32 * 4 + 1] as u32) << 16);
                let aux1 = (bx.qs[ib32 * 4 + 2] as u32) | ((bx.qs[ib32 * 4 + 3] as u32) << 16);
                let g_idx = aux0.to_le_bytes();
                let ls = (2 * (aux1 >> 28) + 1) as i32;

                // Pack 32 magnitudes (`4 x u64`) and 32 sign masks (`4 x u64`) into two 128-bit NEON pairs!
                let g01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[0] as usize])),
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[1] as usize])),
                );
                let g23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[2] as usize])),
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[3] as usize])),
                );

                let s01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[(aux1 & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux1 >> 7) & 127) as usize])),
                );
                let s23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux1 >> 14) & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux1 >> 21) & 127) as usize])),
                );

                let w01 = vsubq_s8(veorq_s8(g01, s01), s01);
                let w23 = vsubq_s8(veorq_s8(g23, s23), s23);

                let y01 = vld1q_s8(yp_base.add(ib32 * 32));
                let y23 = vld1q_s8(yp_base.add(ib32 * 32 + 16));

                let mut p = vmull_s8(vget_low_s8(w01), vget_low_s8(y01));
                p = vmlal_s8(p, vget_high_s8(w01), vget_high_s8(y01));
                p = vmlal_s8(p, vget_low_s8(w23), vget_low_s8(y23));
                p = vmlal_s8(p, vget_high_s8(w23), vget_high_s8(y23));
                let sumi = vaddlvq_s16(p);
                bsum += sumi * ls;
            }

            let d = f16_to_f32_hw(bx.d) * by.d;
            sumf = (bsum as f32).mul_add(d, sumf);
        }
        0.125 * sumf
    }
}

/// Base ARMv8.0 NEON `IQ2_XXS x Q8_K` GEMV (2-row unrolled).
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_iq2_xxs_q8_k(
    w: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for (r, o) in out.iter_mut().enumerate() {
            *o = vec_dot_iq2_xxs_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
        }
    }
}

/// Base ARMv8.0 NEON `IQ2_XXS x Q8_K` GEMM.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_iq2_xxs_q8_k(
    w: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for t in 0..n_tokens {
            gemv_iq2_xxs_q8_k(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// Base ARMv8.0 NEON `BlockIQ3XXS x BlockQ8K` dot product (0-ULP bit-exact!).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_iq3_xxs_q8_k(x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: Fixed 256-element block geometry; all codebook & sign indices are bounded.
    unsafe {
        let mut sumf = 0.0f32;
        for (bx, by) in x.iter().zip(y.iter()) {
            let q3 = &bx.qs[0..64];
            let gas = &bx.qs[64..96];
            let yp_base = by.qs.as_ptr();
            let mut bsum: i32 = 0;

            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    gas[ib32 * 4],
                    gas[ib32 * 4 + 1],
                    gas[ib32 * 4 + 2],
                    gas[ib32 * 4 + 3],
                ]);
                let ls = (2 * (aux32 >> 28) + 1) as i32;
                let q3s = &q3[ib32 * 8..(ib32 + 1) * 8];

                let g_u64_0 = (IQ3XXS_GRID[q3s[0] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[1] as usize] as u64) << 32);
                let g_u64_1 = (IQ3XXS_GRID[q3s[2] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[3] as usize] as u64) << 32);
                let g_u64_2 = (IQ3XXS_GRID[q3s[4] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[5] as usize] as u64) << 32);
                let g_u64_3 = (IQ3XXS_GRID[q3s[6] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[7] as usize] as u64) << 32);

                let g01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(g_u64_0)),
                    vreinterpret_s8_u64(vcreate_u64(g_u64_1)),
                );
                let g23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(g_u64_2)),
                    vreinterpret_s8_u64(vcreate_u64(g_u64_3)),
                );

                let s01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[(aux32 & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux32 >> 7) & 127) as usize])),
                );
                let s23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux32 >> 14) & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux32 >> 21) & 127) as usize])),
                );

                let w01 = vsubq_s8(veorq_s8(g01, s01), s01);
                let w23 = vsubq_s8(veorq_s8(g23, s23), s23);

                let y01 = vld1q_s8(yp_base.add(ib32 * 32));
                let y23 = vld1q_s8(yp_base.add(ib32 * 32 + 16));

                let mut p = vmull_s8(vget_low_s8(w01), vget_low_s8(y01));
                p = vmlal_s8(p, vget_high_s8(w01), vget_high_s8(y01));
                p = vmlal_s8(p, vget_low_s8(w23), vget_low_s8(y23));
                p = vmlal_s8(p, vget_high_s8(w23), vget_high_s8(y23));
                let sumi = vaddlvq_s16(p);
                bsum += sumi * ls;
            }

            let d = f16_to_f32_hw(bx.d) * by.d;
            sumf = (bsum as f32).mul_add(d, sumf);
        }
        0.25 * sumf
    }
}

/// Base ARMv8.0 NEON `IQ3_XXS x Q8_K` GEMV.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_iq3_xxs_q8_k(
    w: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for (r, o) in out.iter_mut().enumerate() {
            *o = vec_dot_iq3_xxs_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
        }
    }
}

/// Base ARMv8.0 NEON `IQ3_XXS x Q8_K` GEMM.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_iq3_xxs_q8_k(
    w: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for t in 0..n_tokens {
            gemv_iq3_xxs_q8_k(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// Base ARMv8.0 NEON `BlockIQ4NL x BlockQ8_0` dot product using single-instruction `vqtbl1q_s8`
/// non-linear 16-entry codebook lookup (0-ULP bit-exact!).
///
/// # Safety
/// Caller must ensure NEON is available and `x.len() == y.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_iq4_nl_q8_0(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: `KVALUES_IQ4NL` is 16 bytes; `bx.qs` is 16 bytes; `by.qs` is 32 bytes.
    unsafe {
        let kv_lut = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let m4b = vdupq_n_u8(0x0F);
        let mut sumf = 0.0f32;

        for (bx, by) in x.iter().zip(y.iter()) {
            let raw = vld1q_u8(bx.qs.as_ptr());
            let w_lo = vqtbl1q_s8(kv_lut, vandq_u8(raw, m4b));
            let w_hi = vqtbl1q_s8(kv_lut, vshrq_n_u8::<4>(raw));

            let y_lo = vld1q_s8(by.qs.as_ptr());
            let y_hi = vld1q_s8(by.qs.as_ptr().add(16));

            let p_lo = vmlal_s8(
                vmull_s8(vget_low_s8(w_lo), vget_low_s8(y_lo)),
                vget_high_s8(w_lo),
                vget_high_s8(y_lo),
            );
            let p_hi = vmlal_s8(
                vmull_s8(vget_low_s8(w_hi), vget_low_s8(y_hi)),
                vget_high_s8(w_hi),
                vget_high_s8(y_hi),
            );
            let sumi = vaddlvq_s16(p_lo) + vaddlvq_s16(p_hi);

            let d = f16_to_f32_hw(bx.d) * f16_to_f32_hw(by.d);
            sumf = (sumi as f32).mul_add(d, sumf);
        }
        sumf
    }
}

/// Base ARMv8.0 NEON 4-row unrolled `IQ4_NL x Q8_0` GEMV using `vqtbl1q_s8`.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemv_iq4_nl_q8_0(
    w: &[BlockIQ4NL],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    // SAFETY: Verified bounds above.
    unsafe {
        let kv_lut = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let m4b = vdupq_n_u8(0x0F);

        let mut r = 0usize;
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut sumv = vdupq_n_f32(0.0);

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let y_lo = vld1q_s8(by.qs.as_ptr());
                let y_hi = vld1q_s8(by.qs.as_ptr().add(16));
                let dy = f16_to_f32_hw(by.d);

                macro_rules! dot_row {
                    ($wr:expr) => {{
                        let bx = $wr.get_unchecked(ib);
                        let raw = vld1q_u8(bx.qs.as_ptr());
                        let wl = vqtbl1q_s8(kv_lut, vandq_u8(raw, m4b));
                        let wh = vqtbl1q_s8(kv_lut, vshrq_n_u8::<4>(raw));
                        let p_lo = vmlal_s8(
                            vmull_s8(vget_low_s8(wl), vget_low_s8(y_lo)),
                            vget_high_s8(wl),
                            vget_high_s8(y_lo),
                        );
                        let p_hi = vmlal_s8(
                            vmull_s8(vget_low_s8(wh), vget_low_s8(y_hi)),
                            vget_high_s8(wh),
                            vget_high_s8(y_hi),
                        );
                        (
                            vaddlvq_s16(p_lo) + vaddlvq_s16(p_hi),
                            f16_to_f32_hw(bx.d) * dy,
                        )
                    }};
                }

                let (s0, d0) = dot_row!(w0);
                let (s1, d1) = dot_row!(w1);
                let (s2, d2) = dot_row!(w2);
                let (s3, d3) = dot_row!(w3);

                let sv = vcvtq_f32_s32(vld1q_s32([s0, s1, s2, s3].as_ptr()));
                let dv = vld1q_f32([d0, d1, d2, d3].as_ptr());
                sumv = vfmaq_f32(sumv, sv, dv);
            }

            vst1q_f32(out.as_mut_ptr().add(r), sumv);
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_iq4_nl_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Base ARMv8.0 NEON `IQ4_NL x Q8_0` GEMM.
///
/// # Safety
/// Caller must ensure NEON is available and dimensions match.
#[target_feature(enable = "neon")]
pub unsafe fn gemm_iq4_nl_q8_0(
    w: &[BlockIQ4NL],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for t in 0..n_tokens {
            gemv_iq4_nl_q8_0(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

// ============================================================================
// ARMv8.2 `dotprod` (`vqtbl1q_s8` + `vdotq_s32`) fast paths for `edge`/`host`
// ============================================================================

/// ARMv8.2 `dotprod` `BlockIQ2XXS x BlockQ8K` dot product.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_vec_dot_iq2_xxs_q8_k(x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: `dotprod` enabled by caller; fixed 256-element block indices.
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut sumf = 0.0f32;
        for (bx, by) in x.iter().zip(y.iter()) {
            let mut bsum: i32 = 0;
            let yp_base = by.qs.as_ptr();

            for ib32 in 0..8 {
                let aux0 = (bx.qs[ib32 * 4] as u32) | ((bx.qs[ib32 * 4 + 1] as u32) << 16);
                let aux1 = (bx.qs[ib32 * 4 + 2] as u32) | ((bx.qs[ib32 * 4 + 3] as u32) << 16);
                let g_idx = aux0.to_le_bytes();
                let ls = (2 * (aux1 >> 28) + 1) as i32;

                let g01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[0] as usize])),
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[1] as usize])),
                );
                let g23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[2] as usize])),
                    vreinterpret_s8_u64(vcreate_u64(IQ2XXS_GRID[g_idx[3] as usize])),
                );

                let s01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[(aux1 & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux1 >> 7) & 127) as usize])),
                );
                let s23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux1 >> 14) & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux1 >> 21) & 127) as usize])),
                );

                let w01 = vsubq_s8(veorq_s8(g01, s01), s01);
                let w23 = vsubq_s8(veorq_s8(g23, s23), s23);

                let y01 = vld1q_s8(yp_base.add(ib32 * 32));
                let y23 = vld1q_s8(yp_base.add(ib32 * 32 + 16));

                let acc = vdotq_s32(vdotq_s32(zero, w01, y01), w23, y23);
                bsum += vaddvq_s32(acc) * ls;
            }
            let d = f16_to_f32_hw(bx.d) * by.d;
            sumf = (bsum as f32).mul_add(d, sumf);
        }
        0.125 * sumf
    }
}

/// ARMv8.2 `dotprod` `IQ2_XXS x Q8_K` GEMV.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_gemv_iq2_xxs_q8_k(
    w: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dotprod_vec_dot_iq2_xxs_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
        }
    }
}

/// ARMv8.2 `dotprod` `IQ2_XXS x Q8_K` GEMM.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_gemm_iq2_xxs_q8_k(
    w: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for t in 0..n_tokens {
            dotprod_gemv_iq2_xxs_q8_k(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// ARMv8.2 `dotprod` `BlockIQ3XXS x BlockQ8K` dot product.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_vec_dot_iq3_xxs_q8_k(x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: `dotprod` enabled by caller; fixed 256-element block indices.
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut sumf = 0.0f32;
        for (bx, by) in x.iter().zip(y.iter()) {
            let q3 = &bx.qs[0..64];
            let gas = &bx.qs[64..96];
            let yp_base = by.qs.as_ptr();
            let mut bsum: i32 = 0;

            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    gas[ib32 * 4],
                    gas[ib32 * 4 + 1],
                    gas[ib32 * 4 + 2],
                    gas[ib32 * 4 + 3],
                ]);
                let ls = (2 * (aux32 >> 28) + 1) as i32;
                let q3s = &q3[ib32 * 8..(ib32 + 1) * 8];

                let g_u64_0 = (IQ3XXS_GRID[q3s[0] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[1] as usize] as u64) << 32);
                let g_u64_1 = (IQ3XXS_GRID[q3s[2] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[3] as usize] as u64) << 32);
                let g_u64_2 = (IQ3XXS_GRID[q3s[4] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[5] as usize] as u64) << 32);
                let g_u64_3 = (IQ3XXS_GRID[q3s[6] as usize] as u64)
                    | ((IQ3XXS_GRID[q3s[7] as usize] as u64) << 32);

                let g01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(g_u64_0)),
                    vreinterpret_s8_u64(vcreate_u64(g_u64_1)),
                );
                let g23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(g_u64_2)),
                    vreinterpret_s8_u64(vcreate_u64(g_u64_3)),
                );

                let s01 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[(aux32 & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux32 >> 7) & 127) as usize])),
                );
                let s23 = vcombine_s8(
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux32 >> 14) & 127) as usize])),
                    vreinterpret_s8_u64(vcreate_u64(KSIGNS64[((aux32 >> 21) & 127) as usize])),
                );

                let w01 = vsubq_s8(veorq_s8(g01, s01), s01);
                let w23 = vsubq_s8(veorq_s8(g23, s23), s23);

                let y01 = vld1q_s8(yp_base.add(ib32 * 32));
                let y23 = vld1q_s8(yp_base.add(ib32 * 32 + 16));

                let acc = vdotq_s32(vdotq_s32(zero, w01, y01), w23, y23);
                bsum += vaddvq_s32(acc) * ls;
            }

            let d = f16_to_f32_hw(bx.d) * by.d;
            sumf = (bsum as f32).mul_add(d, sumf);
        }
        0.25 * sumf
    }
}

/// ARMv8.2 `dotprod` `IQ3_XXS x Q8_K` GEMV.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_gemv_iq3_xxs_q8_k(
    w: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dotprod_vec_dot_iq3_xxs_q8_k(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
        }
    }
}

/// ARMv8.2 `dotprod` `IQ3_XXS x Q8_K` GEMM.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_gemm_iq3_xxs_q8_k(
    w: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for t in 0..n_tokens {
            dotprod_gemv_iq3_xxs_q8_k(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}

/// ARMv8.2 `dotprod` `BlockIQ4NL x BlockQ8_0` dot product.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_vec_dot_iq4_nl_q8_0(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    // SAFETY: `dotprod` enabled by caller; `KVALUES_IQ4NL` is 16 bytes.
    unsafe {
        let kv_lut = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let m4b = vdupq_n_u8(0x0F);
        let zero = vdupq_n_s32(0);
        let mut sumf = 0.0f32;

        for (bx, by) in x.iter().zip(y.iter()) {
            let raw = vld1q_u8(bx.qs.as_ptr());
            let w_lo = vqtbl1q_s8(kv_lut, vandq_u8(raw, m4b));
            let w_hi = vqtbl1q_s8(kv_lut, vshrq_n_u8::<4>(raw));
            let y_lo = vld1q_s8(by.qs.as_ptr());
            let y_hi = vld1q_s8(by.qs.as_ptr().add(16));
            let acc = vdotq_s32(vdotq_s32(zero, w_lo, y_lo), w_hi, y_hi);
            let d = f16_to_f32_hw(bx.d) * f16_to_f32_hw(by.d);
            sumf = (vaddvq_s32(acc) as f32).mul_add(d, sumf);
        }
        sumf
    }
}

/// ARMv8.2 `dotprod` `IQ4_NL x Q8_0` GEMV.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_gemv_iq4_nl_q8_0(
    w: &[BlockIQ4NL],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);
    // SAFETY: Verified bounds above.
    unsafe {
        let kv_lut = vld1q_s8(KVALUES_IQ4NL.as_ptr());
        let m4b = vdupq_n_u8(0x0F);
        let zero = vdupq_n_s32(0);

        let mut r = 0usize;
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut sumv = vdupq_n_f32(0.0);

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let y_lo = vld1q_s8(by.qs.as_ptr());
                let y_hi = vld1q_s8(by.qs.as_ptr().add(16));
                let dy = f16_to_f32_hw(by.d);

                macro_rules! row_sdot {
                    ($wr:expr) => {{
                        let bx = $wr.get_unchecked(ib);
                        let raw = vld1q_u8(bx.qs.as_ptr());
                        let wl = vqtbl1q_s8(kv_lut, vandq_u8(raw, m4b));
                        let wh = vqtbl1q_s8(kv_lut, vshrq_n_u8::<4>(raw));
                        let acc = vdotq_s32(vdotq_s32(zero, wl, y_lo), wh, y_hi);
                        (vaddvq_s32(acc), f16_to_f32_hw(bx.d) * dy)
                    }};
                }

                let (s0, d0) = row_sdot!(w0);
                let (s1, d1) = row_sdot!(w1);
                let (s2, d2) = row_sdot!(w2);
                let (s3, d3) = row_sdot!(w3);
                let sv = vcvtq_f32_s32(vld1q_s32([s0, s1, s2, s3].as_ptr()));
                let dv = vld1q_f32([d0, d1, d2, d3].as_ptr());
                sumv = vfmaq_f32(sumv, sv, dv);
            }
            vst1q_f32(out.as_mut_ptr().add(r), sumv);
            r += 4;
        }
        while r < rows {
            *out.get_unchecked_mut(r) =
                dotprod_vec_dot_iq4_nl_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// ARMv8.2 `dotprod` `IQ4_NL x Q8_0` GEMM.
///
/// # Safety
/// Caller must ensure `neon` and `dotprod` CPU features are available.
#[cfg(feature = "dotprod")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dotprod_gemm_iq4_nl_q8_0(
    w: &[BlockIQ4NL],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols_blocks);
    assert_eq!(out.len(), n_tokens * rows);
    // SAFETY: Verified bounds above.
    unsafe {
        for t in 0..n_tokens {
            dotprod_gemv_iq4_nl_q8_0(
                w,
                &x[t * cols_blocks..(t + 1) * cols_blocks],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }
}
