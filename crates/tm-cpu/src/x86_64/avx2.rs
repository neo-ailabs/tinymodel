//! `x86_64` `AVX2 + FMA` compute kernels (`100%` Rust stable, `0-ULP` bit-exact with `generic`).
//!
//! Algorithm design inspired by ggml `ggml/src/ggml-cpu/arch/x86/quants.c`
//! (Copyright (c) 2023-2026 The ggml authors, MIT License) and Strata `src/quant/x86/`
//! (Copyright (c) 2026 Niko1221 and the Strata contributors, MIT License).
//!
//! Key arithmetic invariants (`AGENTS.md` Rule #3 & §9):
//! - Every integer block dot product (`32` elements for `Q8_0` / `Q4_0` / `Q4_0x4`) is widened
//!   through `_mm256_cvtepi8_epi16` + `_mm256_madd_epi16` directly into `i32` lanes, completely
//!   eliminating `i16` overflow even on adversarial `[-128; 32] x [-128; 32]` inputs (`+524,288`).
//! - Floating-point block accumulation strictly mirrors `generic`:
//!   - `Q8_0`: `_mm_add_ps(acc, _mm_mul_ps(sumi, scale))` (matches `generic::vec_dot_q8_0_q8_0`);
//!   - `Q4_0` / `Q4_0x4`: `_mm_fmadd_ps(sumi, scale, acc)` (matches `generic::vec_dot_q4_0_q8_0`
//!     and `generic::gemv_q4_0x4_q8_0`);
//!   - `rms_norm_f32` / `l2_norm_f32` / `dot_f32_bf16` / `dot_f32_q8_0`: 8-wide `_mm256_fmadd_ps`
//!     with exact `(lo128 + hi128)` pairwise reduction matching `generic` bit-for-bit.

use core::arch::x86_64::*;
use tm_core::{BlockQ4_0, BlockQ4_0x4, BlockQ8_0, QK8_0, f16_to_f32};

use crate::generic;

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let s4 = _mm_add_epi32(lo, hi);
    let s2 = _mm_add_epi32(s4, _mm_srli_si128::<8>(s4));
    let s1 = _mm_add_epi32(s2, _mm_srli_si128::<4>(s2));
    _mm_cvtsi128_si32(s1)
}

/// Reduces four 8-lane `__m256i` integer accumulators `[p0, p1, p2, p3]` into a single
/// 4-lane `__m128i` `[sum(p0), sum(p1), sum(p2), sum(p3)]`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum4_epi32_avx2(p0: __m256i, p1: __m256i, p2: __m256i, p3: __m256i) -> __m128i {
    let h01 = _mm256_hadd_epi32(p0, p1);
    let h23 = _mm256_hadd_epi32(p2, p3);
    let h0123 = _mm256_hadd_epi32(h01, h23);
    _mm_add_epi32(
        _mm256_castsi256_si128(h0123),
        _mm256_extracti128_si256::<1>(h0123),
    )
}

/// Unpacks 16 packed `Q4_0` bytes (`low` nibbles -> `[0..16]`, `high` nibbles -> `[16..32]`, minus 8)
/// into two 16-lane `i16` AVX2 vectors `(xl16, xh16)`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_q4_0_i16_avx2(
    q128: __m128i,
    mask_lo: __m128i,
    eight8: __m128i,
) -> (__m256i, __m256i) {
    let lo8 = _mm_sub_epi8(_mm_and_si128(q128, mask_lo), eight8);
    let hi8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<4>(q128), mask_lo), eight8);
    (_mm256_cvtepi8_epi16(lo8), _mm256_cvtepi8_epi16(hi8))
}

/// Single-row `AVX2 + FMA` `vec_dot_q4_0_q8_0` (`0-ULP` bit-exact with `generic::vec_dot_q4_0_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mask_lo = _mm_set1_epi8(0x0F);
    let eight8 = _mm_set1_epi8(8);
    let mut sumf = 0.0f32;

    // SAFETY: Loop index `ib` is strictly `< nb == x.len() == y.len()`, and `qs` arrays have fixed lengths 16 and 32.
    unsafe {
        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let q = _mm_loadu_si128(bx.qs.as_ptr() as *const __m128i);
            let (xl16, xh16) = unpack_q4_0_i16_avx2(q, mask_lo, eight8);
            let yl16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
            let yh16 =
                _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr().add(16) as *const __m128i));
            let p = _mm256_add_epi32(_mm256_madd_epi16(xl16, yl16), _mm256_madd_epi16(xh16, yh16));
            let sumi = hsum_epi32_avx2(p);
            let scale = f16_to_f32(bx.d) * f16_to_f32(by.d);
            sumf = (sumi as f32).mul_add(scale, sumf);
        }
    }

    sumf
}

/// 4-row shared-activation `AVX2 + FMA` `gemv_q4_0_q8_0`.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
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

    let mask_lo = _mm_set1_epi8(0x0F);
    let eight8 = _mm_set1_epi8(8);
    let mut r = 0usize;

    // SAFETY: Row and block bounds are verified by assertions above.
    unsafe {
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut acc4 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let yl16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
                let yh16 =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr().add(16) as *const __m128i));
                let dy = _mm_set1_ps(f16_to_f32(by.d));

                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);

                let (x0l, x0h) = unpack_q4_0_i16_avx2(
                    _mm_loadu_si128(b0.qs.as_ptr() as *const __m128i),
                    mask_lo,
                    eight8,
                );
                let (x1l, x1h) = unpack_q4_0_i16_avx2(
                    _mm_loadu_si128(b1.qs.as_ptr() as *const __m128i),
                    mask_lo,
                    eight8,
                );
                let (x2l, x2h) = unpack_q4_0_i16_avx2(
                    _mm_loadu_si128(b2.qs.as_ptr() as *const __m128i),
                    mask_lo,
                    eight8,
                );
                let (x3l, x3h) = unpack_q4_0_i16_avx2(
                    _mm_loadu_si128(b3.qs.as_ptr() as *const __m128i),
                    mask_lo,
                    eight8,
                );

                let p0 =
                    _mm256_add_epi32(_mm256_madd_epi16(x0l, yl16), _mm256_madd_epi16(x0h, yh16));
                let p1 =
                    _mm256_add_epi32(_mm256_madd_epi16(x1l, yl16), _mm256_madd_epi16(x1h, yh16));
                let p2 =
                    _mm256_add_epi32(_mm256_madd_epi16(x2l, yl16), _mm256_madd_epi16(x2h, yh16));
                let p3 =
                    _mm256_add_epi32(_mm256_madd_epi16(x3l, yl16), _mm256_madd_epi16(x3h, yh16));

                let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx2(p0, p1, p2, p3));
                let dx4 = _mm_set_ps(
                    f16_to_f32(b3.d),
                    f16_to_f32(b2.d),
                    f16_to_f32(b1.d),
                    f16_to_f32(b0.d),
                );
                let scale4 = _mm_mul_ps(dx4, dy);
                acc4 = _mm_fmadd_ps(sum4, scale4, acc4);
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), acc4);
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q4_0_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Multi-token shared-weight `AVX2 + FMA` `gemm_q4_0_q8_0`: unpacks each weight block ONCE
/// across up to `4` tokens in registers (`0-ULP` batch-width invariant!).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
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

    if n_tokens == 0 {
        return;
    }
    if n_tokens == 1 {
        // SAFETY: Same target features guaranteed by caller.
        unsafe { gemv_q4_0_q8_0(w, x, out, rows, cols_blocks) };
        return;
    }

    let mask_lo = _mm_set1_epi8(0x0F);
    let eight8 = _mm_set1_epi8(8);

    // SAFETY: All slice accesses are bounded by assertions above.
    unsafe {
        let mut t0 = 0usize;
        while t0 < n_tokens {
            let bt = (n_tokens - t0).min(4);
            let mut r = 0usize;
            while r + 4 <= rows {
                let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
                let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
                let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
                let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
                let mut acc = [_mm_setzero_ps(); 4];

                for ib in 0..cols_blocks {
                    let b0 = w0.get_unchecked(ib);
                    let b1 = w1.get_unchecked(ib);
                    let b2 = w2.get_unchecked(ib);
                    let b3 = w3.get_unchecked(ib);

                    let (x0l, x0h) = unpack_q4_0_i16_avx2(
                        _mm_loadu_si128(b0.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let (x1l, x1h) = unpack_q4_0_i16_avx2(
                        _mm_loadu_si128(b1.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let (x2l, x2h) = unpack_q4_0_i16_avx2(
                        _mm_loadu_si128(b2.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let (x3l, x3h) = unpack_q4_0_i16_avx2(
                        _mm_loadu_si128(b3.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let dx4 = _mm_set_ps(
                        f16_to_f32(b3.d),
                        f16_to_f32(b2.d),
                        f16_to_f32(b1.d),
                        f16_to_f32(b0.d),
                    );

                    for (ti, acc_ti) in acc.iter_mut().enumerate().take(bt) {
                        let by = x.get_unchecked((t0 + ti) * cols_blocks + ib);
                        let yl16 =
                            _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
                        let yh16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            by.qs.as_ptr().add(16) as *const __m128i
                        ));
                        let p0 = _mm256_add_epi32(
                            _mm256_madd_epi16(x0l, yl16),
                            _mm256_madd_epi16(x0h, yh16),
                        );
                        let p1 = _mm256_add_epi32(
                            _mm256_madd_epi16(x1l, yl16),
                            _mm256_madd_epi16(x1h, yh16),
                        );
                        let p2 = _mm256_add_epi32(
                            _mm256_madd_epi16(x2l, yl16),
                            _mm256_madd_epi16(x2h, yh16),
                        );
                        let p3 = _mm256_add_epi32(
                            _mm256_madd_epi16(x3l, yl16),
                            _mm256_madd_epi16(x3h, yh16),
                        );
                        let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx2(p0, p1, p2, p3));
                        let scale4 = _mm_mul_ps(dx4, _mm_set1_ps(f16_to_f32(by.d)));
                        *acc_ti = _mm_fmadd_ps(sum4, scale4, *acc_ti);
                    }
                }

                for (ti, &acc_ti) in acc.iter().enumerate().take(bt) {
                    _mm_storeu_ps(out.as_mut_ptr().add((t0 + ti) * rows + r), acc_ti);
                }
                r += 4;
            }

            while r < rows {
                let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
                for ti in 0..bt {
                    let xt = &x[(t0 + ti) * cols_blocks..(t0 + ti + 1) * cols_blocks];
                    *out.get_unchecked_mut((t0 + ti) * rows + r) = vec_dot_q4_0_q8_0(w_row, xt);
                }
                r += 1;
            }
            t0 += bt;
        }
    }
}

/// 4-row `AVX2 + FMA` `gemv_q4_0x4_q8_0` (`0-ULP` bit-exact with `generic::gemv_q4_0x4_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
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

    let mask_lo = _mm_set1_epi8(0x0F);
    let eight8 = _mm_set1_epi8(8);

    // SAFETY: `rg` and `ib` stay within verified slice bounds.
    unsafe {
        for rg in 0..(rows / 4) {
            let w_group = &w[rg * cols_blocks..(rg + 1) * cols_blocks];
            let mut acc4 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let sb = w_group.get_unchecked(ib);
                let by = x.get_unchecked(ib);
                let yl16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
                let yh16 =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr().add(16) as *const __m128i));
                let dy = _mm_set1_ps(f16_to_f32(by.d));

                let qs_ptr = sb.qs.as_ptr();
                let r0_q = _mm_set_epi64x(
                    core::ptr::read_unaligned(qs_ptr.add(16) as *const i64),
                    core::ptr::read_unaligned(qs_ptr as *const i64),
                );
                let r1_q = _mm_set_epi64x(
                    core::ptr::read_unaligned(qs_ptr.add(24) as *const i64),
                    core::ptr::read_unaligned(qs_ptr.add(8) as *const i64),
                );
                let r2_q = _mm_set_epi64x(
                    core::ptr::read_unaligned(qs_ptr.add(48) as *const i64),
                    core::ptr::read_unaligned(qs_ptr.add(32) as *const i64),
                );
                let r3_q = _mm_set_epi64x(
                    core::ptr::read_unaligned(qs_ptr.add(56) as *const i64),
                    core::ptr::read_unaligned(qs_ptr.add(40) as *const i64),
                );

                let (x0l, x0h) = unpack_q4_0_i16_avx2(r0_q, mask_lo, eight8);
                let (x1l, x1h) = unpack_q4_0_i16_avx2(r1_q, mask_lo, eight8);
                let (x2l, x2h) = unpack_q4_0_i16_avx2(r2_q, mask_lo, eight8);
                let (x3l, x3h) = unpack_q4_0_i16_avx2(r3_q, mask_lo, eight8);

                let p0 =
                    _mm256_add_epi32(_mm256_madd_epi16(x0l, yl16), _mm256_madd_epi16(x0h, yh16));
                let p1 =
                    _mm256_add_epi32(_mm256_madd_epi16(x1l, yl16), _mm256_madd_epi16(x1h, yh16));
                let p2 =
                    _mm256_add_epi32(_mm256_madd_epi16(x2l, yl16), _mm256_madd_epi16(x2h, yh16));
                let p3 =
                    _mm256_add_epi32(_mm256_madd_epi16(x3l, yl16), _mm256_madd_epi16(x3h, yh16));

                let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx2(p0, p1, p2, p3));
                let dx4 = _mm_set_ps(
                    f16_to_f32(sb.d[3]),
                    f16_to_f32(sb.d[2]),
                    f16_to_f32(sb.d[1]),
                    f16_to_f32(sb.d[0]),
                );
                acc4 = _mm_fmadd_ps(sum4, _mm_mul_ps(dx4, dy), acc4);
            }

            _mm_storeu_ps(out.as_mut_ptr().add(rg * 4), acc4);
        }
    }
}

/// Multi-token shared-weight `AVX2 + FMA` `gemm_q4_0x4_q8_0`.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
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

    if n_tokens == 0 {
        return;
    }
    if n_tokens == 1 {
        // SAFETY: Same target features guaranteed by caller.
        unsafe { gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks) };
        return;
    }

    let mask_lo = _mm_set1_epi8(0x0F);
    let eight8 = _mm_set1_epi8(8);

    // SAFETY: All slice accesses stay within asserted bounds.
    unsafe {
        let mut t0 = 0usize;
        while t0 < n_tokens {
            let bt = (n_tokens - t0).min(4);
            for rg in 0..(rows / 4) {
                let w_group = &w[rg * cols_blocks..(rg + 1) * cols_blocks];
                let mut acc = [_mm_setzero_ps(); 4];

                for ib in 0..cols_blocks {
                    let sb = w_group.get_unchecked(ib);
                    let qs_ptr = sb.qs.as_ptr();
                    let r0_q = _mm_set_epi64x(
                        core::ptr::read_unaligned(qs_ptr.add(16) as *const i64),
                        core::ptr::read_unaligned(qs_ptr as *const i64),
                    );
                    let r1_q = _mm_set_epi64x(
                        core::ptr::read_unaligned(qs_ptr.add(24) as *const i64),
                        core::ptr::read_unaligned(qs_ptr.add(8) as *const i64),
                    );
                    let r2_q = _mm_set_epi64x(
                        core::ptr::read_unaligned(qs_ptr.add(48) as *const i64),
                        core::ptr::read_unaligned(qs_ptr.add(32) as *const i64),
                    );
                    let r3_q = _mm_set_epi64x(
                        core::ptr::read_unaligned(qs_ptr.add(56) as *const i64),
                        core::ptr::read_unaligned(qs_ptr.add(40) as *const i64),
                    );

                    let (x0l, x0h) = unpack_q4_0_i16_avx2(r0_q, mask_lo, eight8);
                    let (x1l, x1h) = unpack_q4_0_i16_avx2(r1_q, mask_lo, eight8);
                    let (x2l, x2h) = unpack_q4_0_i16_avx2(r2_q, mask_lo, eight8);
                    let (x3l, x3h) = unpack_q4_0_i16_avx2(r3_q, mask_lo, eight8);
                    let dx4 = _mm_set_ps(
                        f16_to_f32(sb.d[3]),
                        f16_to_f32(sb.d[2]),
                        f16_to_f32(sb.d[1]),
                        f16_to_f32(sb.d[0]),
                    );

                    for (ti, acc_ti) in acc.iter_mut().enumerate().take(bt) {
                        let by = x.get_unchecked((t0 + ti) * cols_blocks + ib);
                        let yl16 =
                            _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
                        let yh16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            by.qs.as_ptr().add(16) as *const __m128i
                        ));
                        let p0 = _mm256_add_epi32(
                            _mm256_madd_epi16(x0l, yl16),
                            _mm256_madd_epi16(x0h, yh16),
                        );
                        let p1 = _mm256_add_epi32(
                            _mm256_madd_epi16(x1l, yl16),
                            _mm256_madd_epi16(x1h, yh16),
                        );
                        let p2 = _mm256_add_epi32(
                            _mm256_madd_epi16(x2l, yl16),
                            _mm256_madd_epi16(x2h, yh16),
                        );
                        let p3 = _mm256_add_epi32(
                            _mm256_madd_epi16(x3l, yl16),
                            _mm256_madd_epi16(x3h, yh16),
                        );
                        let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx2(p0, p1, p2, p3));
                        let scale4 = _mm_mul_ps(dx4, _mm_set1_ps(f16_to_f32(by.d)));
                        *acc_ti = _mm_fmadd_ps(sum4, scale4, *acc_ti);
                    }
                }

                for (ti, &acc_ti) in acc.iter().enumerate().take(bt) {
                    _mm_storeu_ps(out.as_mut_ptr().add((t0 + ti) * rows + rg * 4), acc_ti);
                }
            }
            t0 += bt;
        }
    }
}

/// Single-row `AVX2` `vec_dot_q8_0_q8_0` (`0-ULP` bit-exact with `generic::vec_dot_q8_0_q8_0`,
/// immune to `-128 * -128` `i16` overflow via `_mm256_madd_epi16` `i32` widening).
///
/// # Safety
/// Caller must ensure `avx2` CPU feature is supported.
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q8_0_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut sumf = 0.0f32;

    // SAFETY: `ib` stays strictly `< nb == x.len() == y.len()`.
    unsafe {
        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let xl16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bx.qs.as_ptr() as *const __m128i));
            let xh16 =
                _mm256_cvtepi8_epi16(_mm_loadu_si128(bx.qs.as_ptr().add(16) as *const __m128i));
            let yl16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
            let yh16 =
                _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr().add(16) as *const __m128i));
            let p = _mm256_add_epi32(_mm256_madd_epi16(xl16, yl16), _mm256_madd_epi16(xh16, yh16));
            let sumi = hsum_epi32_avx2(p);
            sumf += (sumi as f32) * (f16_to_f32(bx.d) * f16_to_f32(by.d));
        }
    }

    sumf
}

/// 4-row shared-activation `AVX2` `gemv_q8_0_q8_0` (`0-ULP` bit-exact with `generic::gemv_q8_0_q8_0`).
///
/// Primary workhorse for `Inkling-2-Mini-Q8_0` 128-expert MoE & dense `Q8_0` projections on x86_64!
///
/// # Safety
/// Caller must ensure `avx2` CPU feature is supported.
#[target_feature(enable = "avx2")]
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
    // SAFETY: Row and block bounds are asserted above.
    unsafe {
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut acc4 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let yl16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
                let yh16 =
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr().add(16) as *const __m128i));
                let dy = _mm_set1_ps(f16_to_f32(by.d));

                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);

                let p0 = _mm256_add_epi32(
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b0.qs.as_ptr() as *const __m128i)),
                        yl16,
                    ),
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            b0.qs.as_ptr().add(16) as *const __m128i
                        )),
                        yh16,
                    ),
                );
                let p1 = _mm256_add_epi32(
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b1.qs.as_ptr() as *const __m128i)),
                        yl16,
                    ),
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            b1.qs.as_ptr().add(16) as *const __m128i
                        )),
                        yh16,
                    ),
                );
                let p2 = _mm256_add_epi32(
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b2.qs.as_ptr() as *const __m128i)),
                        yl16,
                    ),
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            b2.qs.as_ptr().add(16) as *const __m128i
                        )),
                        yh16,
                    ),
                );
                let p3 = _mm256_add_epi32(
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b3.qs.as_ptr() as *const __m128i)),
                        yl16,
                    ),
                    _mm256_madd_epi16(
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            b3.qs.as_ptr().add(16) as *const __m128i
                        )),
                        yh16,
                    ),
                );

                let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx2(p0, p1, p2, p3));
                let dx4 = _mm_set_ps(
                    f16_to_f32(b3.d),
                    f16_to_f32(b2.d),
                    f16_to_f32(b1.d),
                    f16_to_f32(b0.d),
                );
                let scale4 = _mm_mul_ps(dx4, dy);
                acc4 = _mm_add_ps(acc4, _mm_mul_ps(sum4, scale4));
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), acc4);
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q8_0_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Multi-token shared-weight `AVX2` `gemm_q8_0_q8_0`: loads each `BlockQ8_0` weight block ONCE
/// across up to `4` tokens (`0-ULP` batch-width invariant!).
///
/// # Safety
/// Caller must ensure `avx2` CPU feature is supported.
#[target_feature(enable = "avx2")]
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

    if n_tokens == 0 {
        return;
    }
    if n_tokens == 1 {
        // SAFETY: `avx2` guaranteed by caller.
        unsafe { gemv_q8_0_q8_0(w, x, out, rows, cols_blocks) };
        return;
    }

    // SAFETY: All slice bounds are asserted above.
    unsafe {
        let mut t0 = 0usize;
        while t0 < n_tokens {
            let bt = (n_tokens - t0).min(4);
            let mut r = 0usize;
            while r + 4 <= rows {
                let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
                let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
                let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
                let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
                let mut acc = [_mm_setzero_ps(); 4];

                for ib in 0..cols_blocks {
                    let b0 = w0.get_unchecked(ib);
                    let b1 = w1.get_unchecked(ib);
                    let b2 = w2.get_unchecked(ib);
                    let b3 = w3.get_unchecked(ib);

                    let x0l =
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b0.qs.as_ptr() as *const __m128i));
                    let x0h = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                        b0.qs.as_ptr().add(16) as *const __m128i
                    ));
                    let x1l =
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b1.qs.as_ptr() as *const __m128i));
                    let x1h = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                        b1.qs.as_ptr().add(16) as *const __m128i
                    ));
                    let x2l =
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b2.qs.as_ptr() as *const __m128i));
                    let x2h = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                        b2.qs.as_ptr().add(16) as *const __m128i
                    ));
                    let x3l =
                        _mm256_cvtepi8_epi16(_mm_loadu_si128(b3.qs.as_ptr() as *const __m128i));
                    let x3h = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                        b3.qs.as_ptr().add(16) as *const __m128i
                    ));
                    let dx4 = _mm_set_ps(
                        f16_to_f32(b3.d),
                        f16_to_f32(b2.d),
                        f16_to_f32(b1.d),
                        f16_to_f32(b0.d),
                    );

                    for (ti, acc_ti) in acc.iter_mut().enumerate().take(bt) {
                        let by = x.get_unchecked((t0 + ti) * cols_blocks + ib);
                        let yl16 =
                            _mm256_cvtepi8_epi16(_mm_loadu_si128(by.qs.as_ptr() as *const __m128i));
                        let yh16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                            by.qs.as_ptr().add(16) as *const __m128i
                        ));
                        let p0 = _mm256_add_epi32(
                            _mm256_madd_epi16(x0l, yl16),
                            _mm256_madd_epi16(x0h, yh16),
                        );
                        let p1 = _mm256_add_epi32(
                            _mm256_madd_epi16(x1l, yl16),
                            _mm256_madd_epi16(x1h, yh16),
                        );
                        let p2 = _mm256_add_epi32(
                            _mm256_madd_epi16(x2l, yl16),
                            _mm256_madd_epi16(x2h, yh16),
                        );
                        let p3 = _mm256_add_epi32(
                            _mm256_madd_epi16(x3l, yl16),
                            _mm256_madd_epi16(x3h, yh16),
                        );
                        let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx2(p0, p1, p2, p3));
                        let scale4 = _mm_mul_ps(dx4, _mm_set1_ps(f16_to_f32(by.d)));
                        *acc_ti = _mm_add_ps(*acc_ti, _mm_mul_ps(sum4, scale4));
                    }
                }

                for (ti, &acc_ti) in acc.iter().enumerate().take(bt) {
                    _mm_storeu_ps(out.as_mut_ptr().add((t0 + ti) * rows + r), acc_ti);
                }
                r += 4;
            }

            while r < rows {
                let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
                for ti in 0..bt {
                    let xt = &x[(t0 + ti) * cols_blocks..(t0 + ti + 1) * cols_blocks];
                    *out.get_unchecked_mut((t0 + ti) * rows + r) = vec_dot_q8_0_q8_0(w_row, xt);
                }
                r += 1;
            }
            t0 += bt;
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_8wide_pairwise_ps(acc256: __m256) -> f32 {
    // Exact IEEE-754 order: `((s0 + s1) + (s2 + s3))` where `s = lo128 + hi128`,
    // computed with 1-uop shuffles instead of 3-uop Port-5 `_mm_hadd_ps`.
    let s4 = _mm_add_ps(
        _mm256_castps256_ps128(acc256),
        _mm256_extractf128_ps::<1>(acc256),
    ); // [s0, s1, s2, s3]
    let odd = _mm_movehdup_ps(s4); // [s1, s1, s3, s3]
    let p2 = _mm_add_ps(s4, odd); // [s0 + s1, _, s2 + s3, _]
    let hi = _mm_movehl_ps(p2, p2); // [s2 + s3, _, _, _]
    _mm_cvtss_f32(_mm_add_ss(p2, hi))
}

/// `AVX2 + FMA` `dot_f32_bf16` (`0-ULP` bit-exact with `generic::dot_f32_bf16`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_bf16(x: &[f32], w_bf16: &[u16]) -> f32 {
    let n = x.len();
    assert_eq!(w_bf16.len(), n);
    let chunks8 = n / 8;

    // SAFETY: `c * 8 + 8 <= n` in the SIMD loop.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for c in 0..chunks8 {
            let base = c * 8;
            let raw16 = _mm_loadu_si128(w_bf16.as_ptr().add(base) as *const __m128i);
            let w32 = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(raw16));
            let w_ps = _mm256_castsi256_ps(w32);
            let x_ps = _mm256_loadu_ps(x.as_ptr().add(base));
            acc = _mm256_fmadd_ps(w_ps, x_ps, acc);
        }
        let mut sum = hsum_8wide_pairwise_ps(acc);
        for i in chunks8 * 8..n {
            let w = f32::from_bits((*w_bf16.get_unchecked(i) as u32) << 16);
            sum = w.mul_add(*x.get_unchecked(i), sum);
        }
        sum
    }
}

/// `AVX2 + FMA` `dot_f32_q8_0` (`0-ULP` bit-exact with `generic::dot_f32_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_q8_0(q: &[f32], k_q8: &[BlockQ8_0]) -> f32 {
    assert_eq!(q.len(), k_q8.len() * QK8_0);
    let mut total = 0.0f32;

    // SAFETY: Each block has `QK8_0 == 32` elements (`4 * 8` lanes).
    unsafe {
        for (ib, blk) in k_q8.iter().enumerate() {
            let q_ptr = q.as_ptr().add(ib * QK8_0);
            let k_ptr = blk.qs.as_ptr();
            let mut acc = _mm256_setzero_ps();
            for c in 0..4 {
                let off = c * 8;
                let k8 = _mm_loadl_epi64(k_ptr.add(off) as *const __m128i);
                let k_ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k8));
                let q_ps = _mm256_loadu_ps(q_ptr.add(off));
                acc = _mm256_fmadd_ps(k_ps, q_ps, acc);
            }
            let bsum = hsum_8wide_pairwise_ps(acc);
            total = bsum.mul_add(f16_to_f32(blk.d), total);
        }
    }
    total
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum4_8wide_pairwise_ps(a0: __m256, a1: __m256, a2: __m256, a3: __m256) -> __m128 {
    // 4x4 transpose + vertical `_mm_add_ps` preserves exact `((s0 + s1) + (s2 + s3))` order
    // while avoiding 3-uop Port-5 `_mm_hadd_ps` bottlenecks on Skylake-X / Zen.
    let s4_0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps::<1>(a0));
    let s4_1 = _mm_add_ps(_mm256_castps256_ps128(a1), _mm256_extractf128_ps::<1>(a1));
    let s4_2 = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps::<1>(a2));
    let s4_3 = _mm_add_ps(_mm256_castps256_ps128(a3), _mm256_extractf128_ps::<1>(a3));
    let t0 = _mm_unpacklo_ps(s4_0, s4_1);
    let t1 = _mm_unpackhi_ps(s4_0, s4_1);
    let t2 = _mm_unpacklo_ps(s4_2, s4_3);
    let t3 = _mm_unpackhi_ps(s4_2, s4_3);
    let c0 = _mm_movelh_ps(t0, t2);
    let c1 = _mm_movehl_ps(t2, t0);
    let c2 = _mm_movelh_ps(t1, t3);
    let c3 = _mm_movehl_ps(t3, t1);
    _mm_add_ps(_mm_add_ps(c0, c1), _mm_add_ps(c2, c3))
}

/// 8-row shared-activation `AVX2 + FMA + F16C` `gemv_f32_q8_0` (`0-ULP` bit-exact with `generic::gemv_f32_q8_0`).
///
/// Uses 8 YMM accumulators to share `x` loads across 8 rows simultaneously and converts all 8
/// `f16` block scales in a single hardware `VCVTPH2PS` instruction (`_mm256_cvtph_ps`).
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are supported.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemv_f32_q8_0(
    w: &[BlockQ8_0],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks * QK8_0);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: Row and block bounds are asserted above.
    unsafe {
        while r + 8 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let w4 = &w[(r + 4) * cols_blocks..(r + 5) * cols_blocks];
            let w5 = &w[(r + 5) * cols_blocks..(r + 6) * cols_blocks];
            let w6 = &w[(r + 6) * cols_blocks..(r + 7) * cols_blocks];
            let w7 = &w[(r + 7) * cols_blocks..(r + 8) * cols_blocks];
            let mut tot0 = _mm_setzero_ps();
            let mut tot1 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);
                let b4 = w4.get_unchecked(ib);
                let b5 = w5.get_unchecked(ib);
                let b6 = w6.get_unchecked(ib);
                let b7 = w7.get_unchecked(ib);
                let q_ptr = x.as_ptr().add(ib * QK8_0);
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut acc4 = _mm256_setzero_ps();
                let mut acc5 = _mm256_setzero_ps();
                let mut acc6 = _mm256_setzero_ps();
                let mut acc7 = _mm256_setzero_ps();

                for c in 0..4 {
                    let off = c * 8;
                    let q_ps = _mm256_loadu_ps(q_ptr.add(off));
                    acc0 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b0.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc0,
                    );
                    acc1 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b1.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc1,
                    );
                    acc2 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b2.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc2,
                    );
                    acc3 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b3.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc3,
                    );
                    acc4 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b4.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc4,
                    );
                    acc5 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b5.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc5,
                    );
                    acc6 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b6.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc6,
                    );
                    acc7 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b7.qs.as_ptr().add(off) as *const __m128i,
                        ))),
                        q_ps,
                        acc7,
                    );
                }

                let d8 = _mm256_cvtph_ps(_mm_set_epi16(
                    b7.d as i16,
                    b6.d as i16,
                    b5.d as i16,
                    b4.d as i16,
                    b3.d as i16,
                    b2.d as i16,
                    b1.d as i16,
                    b0.d as i16,
                ));
                let bsum0 = hsum4_8wide_pairwise_ps(acc0, acc1, acc2, acc3);
                let bsum1 = hsum4_8wide_pairwise_ps(acc4, acc5, acc6, acc7);
                tot0 = _mm_fmadd_ps(bsum0, _mm256_castps256_ps128(d8), tot0);
                tot1 = _mm_fmadd_ps(bsum1, _mm256_extractf128_ps::<1>(d8), tot1);
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), tot0);
            _mm_storeu_ps(out.as_mut_ptr().add(r + 4), tot1);
            r += 8;
        }

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut tot4 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);
                let q_ptr = x.as_ptr().add(ib * QK8_0);
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                for c in 0..4 {
                    let off = c * 8;
                    let q_ps = _mm256_loadu_ps(q_ptr.add(off));
                    let k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        b0.qs.as_ptr().add(off) as *const __m128i,
                    )));
                    let k1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        b1.qs.as_ptr().add(off) as *const __m128i,
                    )));
                    let k2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        b2.qs.as_ptr().add(off) as *const __m128i,
                    )));
                    let k3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        b3.qs.as_ptr().add(off) as *const __m128i,
                    )));
                    acc0 = _mm256_fmadd_ps(k0, q_ps, acc0);
                    acc1 = _mm256_fmadd_ps(k1, q_ps, acc1);
                    acc2 = _mm256_fmadd_ps(k2, q_ps, acc2);
                    acc3 = _mm256_fmadd_ps(k3, q_ps, acc3);
                }

                let bsum4 = hsum4_8wide_pairwise_ps(acc0, acc1, acc2, acc3);
                let d4 = _mm_cvtph_ps(_mm_set_epi16(
                    0,
                    0,
                    0,
                    0,
                    b3.d as i16,
                    b2.d as i16,
                    b1.d as i16,
                    b0.d as i16,
                ));
                tot4 = _mm_fmadd_ps(bsum4, d4, tot4);
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), tot4);
            r += 4;
        }

        while r < rows {
            let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = dot_f32_q8_0(x, w_row);
            r += 1;
        }
    }
}

/// Multi-token L1-pinned `4 rows x 3 tokens` `AVX2 + FMA + F16C` `gemm_f32_q8_0` (`0-ULP` bit-exact).
///
/// Steps `r` by 4 in the outer loop so 4 weight rows (`< 10 KiB`) remain L1D-resident across all
/// `n_tokens`, using 12 YMM accumulators (`4 rows x 3 tokens`) to cut `int8 -> f32` weight unpacks
/// by 33% over `4x2` with zero YMM register spills.
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are supported.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemm_f32_q8_0(
    w: &[BlockQ8_0],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tokens: usize,
) {
    let cols = cols_blocks * QK8_0;
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tokens * cols);
    assert_eq!(out.len(), n_tokens * rows);

    if n_tokens == 0 {
        return;
    }
    if n_tokens == 1 {
        // SAFETY: Same target features guaranteed by caller.
        unsafe { gemv_f32_q8_0(w, x, out, rows, cols_blocks) };
        return;
    }

    let mut r = 0usize;
    // SAFETY: Row, block, and token bounds are asserted above.
    unsafe {
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];

            let mut t0 = 0usize;
            while t0 + 3 <= n_tokens {
                let x0 = x.as_ptr().add(t0 * cols);
                let x1 = x.as_ptr().add((t0 + 1) * cols);
                let x2 = x.as_ptr().add((t0 + 2) * cols);
                let mut tot_t0 = _mm_setzero_ps();
                let mut tot_t1 = _mm_setzero_ps();
                let mut tot_t2 = _mm_setzero_ps();

                for ib in 0..cols_blocks {
                    let b0 = w0.get_unchecked(ib);
                    let b1 = w1.get_unchecked(ib);
                    let b2 = w2.get_unchecked(ib);
                    let b3 = w3.get_unchecked(ib);
                    let base = ib * QK8_0;
                    let mut a0_t0 = _mm256_setzero_ps();
                    let mut a1_t0 = _mm256_setzero_ps();
                    let mut a2_t0 = _mm256_setzero_ps();
                    let mut a3_t0 = _mm256_setzero_ps();
                    let mut a0_t1 = _mm256_setzero_ps();
                    let mut a1_t1 = _mm256_setzero_ps();
                    let mut a2_t1 = _mm256_setzero_ps();
                    let mut a3_t1 = _mm256_setzero_ps();
                    let mut a0_t2 = _mm256_setzero_ps();
                    let mut a1_t2 = _mm256_setzero_ps();
                    let mut a2_t2 = _mm256_setzero_ps();
                    let mut a3_t2 = _mm256_setzero_ps();

                    for c in 0..4 {
                        let off = c * 8;
                        let q_t0 = _mm256_loadu_ps(x0.add(base + off));
                        let q_t1 = _mm256_loadu_ps(x1.add(base + off));
                        let q_t2 = _mm256_loadu_ps(x2.add(base + off));

                        let k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b0.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        a0_t0 = _mm256_fmadd_ps(k0, q_t0, a0_t0);
                        a0_t1 = _mm256_fmadd_ps(k0, q_t1, a0_t1);
                        a0_t2 = _mm256_fmadd_ps(k0, q_t2, a0_t2);

                        let k1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b1.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        a1_t0 = _mm256_fmadd_ps(k1, q_t0, a1_t0);
                        a1_t1 = _mm256_fmadd_ps(k1, q_t1, a1_t1);
                        a1_t2 = _mm256_fmadd_ps(k1, q_t2, a1_t2);

                        let k2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b2.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        a2_t0 = _mm256_fmadd_ps(k2, q_t0, a2_t0);
                        a2_t1 = _mm256_fmadd_ps(k2, q_t1, a2_t1);
                        a2_t2 = _mm256_fmadd_ps(k2, q_t2, a2_t2);

                        let k3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b3.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        a3_t0 = _mm256_fmadd_ps(k3, q_t0, a3_t0);
                        a3_t1 = _mm256_fmadd_ps(k3, q_t1, a3_t1);
                        a3_t2 = _mm256_fmadd_ps(k3, q_t2, a3_t2);
                    }

                    let d4 = _mm_cvtph_ps(_mm_set_epi16(
                        0,
                        0,
                        0,
                        0,
                        b3.d as i16,
                        b2.d as i16,
                        b1.d as i16,
                        b0.d as i16,
                    ));
                    let s_t0 = hsum4_8wide_pairwise_ps(a0_t0, a1_t0, a2_t0, a3_t0);
                    let s_t1 = hsum4_8wide_pairwise_ps(a0_t1, a1_t1, a2_t1, a3_t1);
                    let s_t2 = hsum4_8wide_pairwise_ps(a0_t2, a1_t2, a2_t2, a3_t2);
                    tot_t0 = _mm_fmadd_ps(s_t0, d4, tot_t0);
                    tot_t1 = _mm_fmadd_ps(s_t1, d4, tot_t1);
                    tot_t2 = _mm_fmadd_ps(s_t2, d4, tot_t2);
                }

                _mm_storeu_ps(out.as_mut_ptr().add(t0 * rows + r), tot_t0);
                _mm_storeu_ps(out.as_mut_ptr().add((t0 + 1) * rows + r), tot_t1);
                _mm_storeu_ps(out.as_mut_ptr().add((t0 + 2) * rows + r), tot_t2);
                t0 += 3;
            }

            while t0 + 2 <= n_tokens {
                let x0 = x.as_ptr().add(t0 * cols);
                let x1 = x.as_ptr().add((t0 + 1) * cols);
                let mut tot_t0 = _mm_setzero_ps();
                let mut tot_t1 = _mm_setzero_ps();

                for ib in 0..cols_blocks {
                    let b0 = w0.get_unchecked(ib);
                    let b1 = w1.get_unchecked(ib);
                    let b2 = w2.get_unchecked(ib);
                    let b3 = w3.get_unchecked(ib);
                    let base = ib * QK8_0;
                    let mut a0_t0 = _mm256_setzero_ps();
                    let mut a1_t0 = _mm256_setzero_ps();
                    let mut a2_t0 = _mm256_setzero_ps();
                    let mut a3_t0 = _mm256_setzero_ps();
                    let mut a0_t1 = _mm256_setzero_ps();
                    let mut a1_t1 = _mm256_setzero_ps();
                    let mut a2_t1 = _mm256_setzero_ps();
                    let mut a3_t1 = _mm256_setzero_ps();

                    for c in 0..4 {
                        let off = c * 8;
                        let q_t0 = _mm256_loadu_ps(x0.add(base + off));
                        let q_t1 = _mm256_loadu_ps(x1.add(base + off));
                        let k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b0.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        let k1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b1.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        let k2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b2.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        let k3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b3.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        a0_t0 = _mm256_fmadd_ps(k0, q_t0, a0_t0);
                        a1_t0 = _mm256_fmadd_ps(k1, q_t0, a1_t0);
                        a2_t0 = _mm256_fmadd_ps(k2, q_t0, a2_t0);
                        a3_t0 = _mm256_fmadd_ps(k3, q_t0, a3_t0);
                        a0_t1 = _mm256_fmadd_ps(k0, q_t1, a0_t1);
                        a1_t1 = _mm256_fmadd_ps(k1, q_t1, a1_t1);
                        a2_t1 = _mm256_fmadd_ps(k2, q_t1, a2_t1);
                        a3_t1 = _mm256_fmadd_ps(k3, q_t1, a3_t1);
                    }

                    let d4 = _mm_cvtph_ps(_mm_set_epi16(
                        0,
                        0,
                        0,
                        0,
                        b3.d as i16,
                        b2.d as i16,
                        b1.d as i16,
                        b0.d as i16,
                    ));
                    let s_t0 = hsum4_8wide_pairwise_ps(a0_t0, a1_t0, a2_t0, a3_t0);
                    let s_t1 = hsum4_8wide_pairwise_ps(a0_t1, a1_t1, a2_t1, a3_t1);
                    tot_t0 = _mm_fmadd_ps(s_t0, d4, tot_t0);
                    tot_t1 = _mm_fmadd_ps(s_t1, d4, tot_t1);
                }

                _mm_storeu_ps(out.as_mut_ptr().add(t0 * rows + r), tot_t0);
                _mm_storeu_ps(out.as_mut_ptr().add((t0 + 1) * rows + r), tot_t1);
                t0 += 2;
            }

            if t0 < n_tokens {
                let x_last = &x[t0 * cols..(t0 + 1) * cols];
                let out_last = &mut out[t0 * rows + r..t0 * rows + r + 4];
                gemv_f32_q8_0(
                    &w[r * cols_blocks..(r + 4) * cols_blocks],
                    x_last,
                    out_last,
                    4,
                    cols_blocks,
                );
            }

            r += 4;
        }

        while r < rows {
            let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            for t in 0..n_tokens {
                let xt = &x[t * cols..(t + 1) * cols];
                *out.get_unchecked_mut(t * rows + r) = dot_f32_q8_0(xt, w_row);
            }
            r += 1;
        }
    }
}

/// `AVX2 + FMA` `axpy_f32_q8_0` (`0-ULP` bit-exact with `generic::axpy_f32_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_f32_q8_0(out: &mut [f32], w: f32, v_q8: &[BlockQ8_0]) {
    assert_eq!(out.len(), v_q8.len() * QK8_0);

    // SAFETY: Each block spans 32 contiguous `f32` outputs and 32 `i8` values.
    unsafe {
        for (ib, blk) in v_q8.iter().enumerate() {
            let scale = _mm256_set1_ps(w * f16_to_f32(blk.d));
            let o_ptr = out.as_mut_ptr().add(ib * QK8_0);
            let v_ptr = blk.qs.as_ptr();
            for c in 0..4 {
                let off = c * 8;
                let v8 = _mm_loadl_epi64(v_ptr.add(off) as *const __m128i);
                let v_ps = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8));
                let cur = _mm256_loadu_ps(o_ptr.add(off));
                _mm256_storeu_ps(o_ptr.add(off), _mm256_fmadd_ps(v_ps, scale, cur));
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sum_sq_8wide_avx2(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks8 = n / 8;
    // SAFETY: `c * 8 + 8 <= n`.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for c in 0..chunks8 {
            let v = _mm256_loadu_ps(x.as_ptr().add(c * 8));
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        let mut sum = hsum_8wide_pairwise_ps(acc);
        for &v in &x[chunks8 * 8..] {
            sum = v.mul_add(v, sum);
        }
        sum
    }
}

/// `AVX2 + FMA` `rms_norm_f32` (`0-ULP` bit-exact with `generic::rms_norm_f32`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_f32(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(out.len(), n);
    assert_eq!(weight.len(), n);
    if n == 0 {
        return;
    }
    // SAFETY: `avx2,fma` enabled; slices have equal length `n`.
    unsafe {
        let ss = sum_sq_8wide_avx2(x);
        let scale = (1.0 / (((ss as f64) / (n as f64)) + (eps as f64)).sqrt()) as f32;
        let sv = _mm256_set1_ps(scale);
        let chunks8 = n / 8;
        for c in 0..chunks8 {
            let off = c * 8;
            let xv = _mm256_loadu_ps(x.as_ptr().add(off));
            let wv = _mm256_loadu_ps(weight.as_ptr().add(off));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(off),
                _mm256_mul_ps(_mm256_mul_ps(xv, sv), wv),
            );
        }
        for i in chunks8 * 8..n {
            *out.get_unchecked_mut(i) = (*x.get_unchecked(i) * scale) * *weight.get_unchecked(i);
        }
    }
}

/// `AVX2 + FMA` `l2_norm_f32` (`0-ULP` bit-exact with `generic::l2_norm_f32`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_norm_f32(x: &mut [f32], eps: f32) {
    let n = x.len();
    // SAFETY: `avx2,fma` enabled; `c * 8 + 8 <= n`.
    unsafe {
        let ss = sum_sq_8wide_avx2(x);
        let scale = (1.0 / ((ss as f64) + (eps as f64)).sqrt()) as f32;
        let sv = _mm256_set1_ps(scale);
        let chunks8 = n / 8;
        for c in 0..chunks8 {
            let off = c * 8;
            let xv = _mm256_loadu_ps(x.as_ptr().add(off));
            _mm256_storeu_ps(x.as_mut_ptr().add(off), _mm256_mul_ps(xv, sv));
        }
        for v in &mut x[chunks8 * 8..] {
            *v *= scale;
        }
    }
}

/// `AVX2 + FMA` single-head Gated DeltaNet recurrent step (`0-ULP` bit-exact with `generic::gdn_recurrence_head`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
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

    if !head_dim.is_multiple_of(8) {
        generic::gdn_recurrence_head(q_h, k_h, v_h, g, beta, scale, s_vh, o_h, head_dim);
        return;
    }

    let chunks8 = head_dim / 8;
    // SAFETY: `head_dim` is a multiple of 8 and all slice lengths are verified above.
    unsafe {
        let gv = _mm256_set1_ps(g);
        for j in 0..head_dim {
            let s_ptr = s_vh.as_mut_ptr().add(j * head_dim);
            let mut kv_acc = _mm256_setzero_ps();
            for c in 0..chunks8 {
                let off = c * 8;
                let d = _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(off)), gv);
                let kv = _mm256_loadu_ps(k_h.as_ptr().add(off));
                kv_acc = _mm256_fmadd_ps(d, kv, kv_acc);
            }
            let sum_kv = hsum_8wide_pairwise_ps(kv_acc);
            let delta_j = (*v_h.get_unchecked(j) - sum_kv) * beta;
            let dv = _mm256_set1_ps(delta_j);

            let mut oq_acc = _mm256_setzero_ps();
            for c in 0..chunks8 {
                let off = c * 8;
                let s_dec = _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(off)), gv);
                let kv = _mm256_loadu_ps(k_h.as_ptr().add(off));
                let u = _mm256_fmadd_ps(dv, kv, s_dec);
                _mm256_storeu_ps(s_ptr.add(off), u);
                let qv = _mm256_loadu_ps(q_h.as_ptr().add(off));
                oq_acc = _mm256_fmadd_ps(u, qv, oq_acc);
            }
            *o_h.get_unchecked_mut(j) = hsum_8wide_pairwise_ps(oq_acc) * scale;
        }
    }
}

/// AVX2+FMA single-head Mamba-2 selective state scan (`ADR-017`).
///
/// Matches `generic::mamba2_scan_head` **bit-for-bit (`0 ULP`)**.
///
/// # Safety
/// Caller must ensure `avx2` and `fma` are available on the current CPU.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mamba2_scan_head(
    x_h: &[f32],
    b_g: &[f32],
    c_g: &[f32],
    da: f32,
    dt_sp: f32,
    d_skip: f32,
    s_h: &mut [f32],
    y_h: &mut [f32],
    head_dim: usize,
    d_state: usize,
) {
    assert_eq!(x_h.len(), head_dim);
    assert_eq!(b_g.len(), d_state);
    assert_eq!(c_g.len(), d_state);
    assert_eq!(s_h.len(), head_dim * d_state);
    assert_eq!(y_h.len(), head_dim);

    if !d_state.is_multiple_of(8) {
        generic::mamba2_scan_head(
            x_h, b_g, c_g, da, dt_sp, d_skip, s_h, y_h, head_dim, d_state,
        );
        return;
    }

    let chunks8 = d_state / 8;
    // SAFETY: `d_state` is a multiple of 8 and all slice lengths are verified above.
    unsafe {
        let vda = _mm256_set1_ps(da);
        for j in 0..head_dim {
            let s_ptr = s_h.as_mut_ptr().add(j * d_state);
            let x_val = *x_h.get_unchecked(j);
            let x_dt = x_val * dt_sp;
            let vxdt = _mm256_set1_ps(x_dt);
            let mut acc = _mm256_setzero_ps();
            for c in 0..chunks8 {
                let off = c * 8;
                let s_dec = _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(off)), vda);
                let bv = _mm256_loadu_ps(b_g.as_ptr().add(off));
                let u = _mm256_fmadd_ps(bv, vxdt, s_dec);
                _mm256_storeu_ps(s_ptr.add(off), u);
                let cv = _mm256_loadu_ps(c_g.as_ptr().add(off));
                acc = _mm256_fmadd_ps(u, cv, acc);
            }
            let sum_y = hsum_8wide_pairwise_ps(acc);
            *y_h.get_unchecked_mut(j) = x_val.mul_add(d_skip, sum_y);
        }
    }
}

/// `AVX2 + FMA + F16C` `dot_f32_f16` (`0-ULP` bit-exact on exact powers of two).
///
/// # Safety
/// Caller must ensure `avx2`, `fma`, and `f16c` CPU features are supported.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_f32_f16(q: &[f32], k_f16: &[u16]) -> f32 {
    debug_assert_eq!(q.len(), k_f16.len());
    let n = q.len();
    let chunks8 = n / 8;
    // SAFETY: `chunks8 * 8 <= n == q.len() == k_f16.len()`.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for c in 0..chunks8 {
            let off = c * 8;
            let vq = _mm256_loadu_ps(q.as_ptr().add(off));
            let vk16 = _mm_loadu_si128(k_f16.as_ptr().add(off) as *const __m128i);
            acc = _mm256_fmadd_ps(vq, _mm256_cvtph_ps(vk16), acc);
        }
        let mut sum = hsum_8wide_pairwise_ps(acc);
        for i in (chunks8 * 8)..n {
            sum += *q.get_unchecked(i) * f16_to_f32(*k_f16.get_unchecked(i));
        }
        sum
    }
}

/// `AVX2 + F16C` `axpy_f32_f16` (`0-ULP` bit-exact with `generic::axpy_f32_f16`).
///
/// # Safety
/// Caller must ensure `avx2` and `f16c` CPU features are supported.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn axpy_f32_f16(out: &mut [f32], w: f32, v_f16: &[u16]) {
    debug_assert_eq!(out.len(), v_f16.len());
    let n = out.len();
    let chunks8 = n / 8;
    // SAFETY: `chunks8 * 8 <= n == out.len() == v_f16.len()`.
    unsafe {
        let vw = _mm256_set1_ps(w);
        for c in 0..chunks8 {
            let off = c * 8;
            let vo = _mm256_loadu_ps(out.as_ptr().add(off));
            let vv16 = _mm_loadu_si128(v_f16.as_ptr().add(off) as *const __m128i);
            let vv = _mm256_cvtph_ps(vv16);
            _mm256_storeu_ps(
                out.as_mut_ptr().add(off),
                _mm256_add_ps(vo, _mm256_mul_ps(vw, vv)),
            );
        }
        for i in (chunks8 * 8)..n {
            *out.get_unchecked_mut(i) += w * f16_to_f32(*v_f16.get_unchecked(i));
        }
    }
}

/// `AVX2 + F16C` `f32_to_f16_slice` (`0-ULP` IEEE-754 round-to-nearest-even).
///
/// # Safety
/// Caller must ensure `avx2` and `f16c` CPU features are supported.
#[target_feature(enable = "avx2,f16c")]
pub unsafe fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let chunks8 = n / 8;
    // SAFETY: `chunks8 * 8 <= n == src.len() == dst.len()`.
    unsafe {
        for c in 0..chunks8 {
            let off = c * 8;
            let vs = _mm256_loadu_ps(src.as_ptr().add(off));
            let vh = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(vs);
            _mm_storeu_si128(dst.as_mut_ptr().add(off) as *mut __m128i, vh);
        }
        for i in (chunks8 * 8)..n {
            *dst.get_unchecked_mut(i) = tm_core::f32_to_f16(*src.get_unchecked(i));
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_q2_0_half32_u2(half8: __m128i) -> __m256i {
    // Expands 8 bytes (`32` 2-bit codes in `[4*j + k]` layout) into 32 unsigned bytes `{0, 1, 2, 3}`!
    let dw = _mm256_cvtepu8_epi32(half8);
    let m0 = _mm256_set1_epi32(0x0000_0003);
    let m1 = _mm256_set1_epi32(0x0000_0300);
    let m2 = _mm256_set1_epi32(0x0003_0000);
    let m3 = _mm256_set1_epi32(0x0300_0000);
    let c0 = _mm256_and_si256(dw, m0);
    let c1 = _mm256_and_si256(_mm256_slli_epi32::<6>(dw), m1);
    let c2 = _mm256_and_si256(_mm256_slli_epi32::<12>(dw), m2);
    let c3 = _mm256_and_si256(_mm256_slli_epi32::<18>(dw), m3);
    _mm256_or_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3))
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_q2_half_q8_i32(u2: __m256i, y_s8: __m256i) -> i32 {
    let one = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let prod = _mm256_sub_epi16(
        _mm256_maddubs_epi16(u2, y_s8),
        _mm256_maddubs_epi16(one, y_s8),
    );
    // SAFETY: Caller guarantees `avx2` is available.
    unsafe { hsum_epi32_avx2(_mm256_madd_epi16(prod, ones16)) }
}

/// `AVX2 + FMA` `vec_dot_q2_0_q8_0` (`0-ULP` bit-exact with `generic::vec_dot_q2_0_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_q2_0_q8_0(x: &[tm_core::BlockQ2_0], y: &[BlockQ8_0]) -> f32 {
    assert_eq!(x.len() * 2, y.len());
    let mut sumf = 0.0f32;
    // SAFETY: `y.len() == 2 * x.len()`, each `BlockQ2_0.qs` is 16 bytes and `BlockQ8_0.qs` is 32 bytes.
    unsafe {
        for (i, xb) in x.iter().enumerate() {
            let yb0 = y.get_unchecked(2 * i);
            let yb1 = y.get_unchecked(2 * i + 1);
            let raw = _mm_loadu_si128(xb.qs.as_ptr().cast::<__m128i>());
            let u2_0 = unpack_q2_0_half32_u2(raw);
            let u2_1 = unpack_q2_0_half32_u2(_mm_srli_si128::<8>(raw));
            let y0 = _mm256_loadu_si256(yb0.qs.as_ptr().cast::<__m256i>());
            let y1 = _mm256_loadu_si256(yb1.qs.as_ptr().cast::<__m256i>());
            let s0 = dot_q2_half_q8_i32(u2_0, y0);
            let s1 = dot_q2_half_q8_i32(u2_1, y1);
            let sub = (s0 as f32) * f16_to_f32(yb0.d) + (s1 as f32) * f16_to_f32(yb1.d);
            sumf = f16_to_f32(xb.d).mul_add(sub, sumf);
        }
    }
    sumf
}

/// 4-row shared-activation `AVX2 + FMA` `gemv_q2_0_q8_0` (`0-ULP` bit-exact with `generic::gemv_q2_0_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q2_0_q8_0(
    w: &[tm_core::BlockQ2_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks * 2);
    assert_eq!(out.len(), rows);

    let mut r = 0usize;
    // SAFETY: Bounds checked by assertions above.
    unsafe {
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;

            for i in 0..cols_blocks {
                let yb0 = x.get_unchecked(2 * i);
                let yb1 = x.get_unchecked(2 * i + 1);
                let y0 = _mm256_loadu_si256(yb0.qs.as_ptr().cast::<__m256i>());
                let y1 = _mm256_loadu_si256(yb1.qs.as_ptr().cast::<__m256i>());
                let yd0 = f16_to_f32(yb0.d);
                let yd1 = f16_to_f32(yb1.d);

                let b0 = w0.get_unchecked(i);
                let r0 = _mm_loadu_si128(b0.qs.as_ptr().cast::<__m128i>());
                let sub0 = (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(r0), y0) as f32) * yd0
                    + (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(_mm_srli_si128::<8>(r0)), y1)
                        as f32)
                        * yd1;
                sum0 = f16_to_f32(b0.d).mul_add(sub0, sum0);

                let b1 = w1.get_unchecked(i);
                let r1 = _mm_loadu_si128(b1.qs.as_ptr().cast::<__m128i>());
                let sub1 = (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(r1), y0) as f32) * yd0
                    + (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(_mm_srli_si128::<8>(r1)), y1)
                        as f32)
                        * yd1;
                sum1 = f16_to_f32(b1.d).mul_add(sub1, sum1);

                let b2 = w2.get_unchecked(i);
                let r2 = _mm_loadu_si128(b2.qs.as_ptr().cast::<__m128i>());
                let sub2 = (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(r2), y0) as f32) * yd0
                    + (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(_mm_srli_si128::<8>(r2)), y1)
                        as f32)
                        * yd1;
                sum2 = f16_to_f32(b2.d).mul_add(sub2, sum2);

                let b3 = w3.get_unchecked(i);
                let r3 = _mm_loadu_si128(b3.qs.as_ptr().cast::<__m128i>());
                let sub3 = (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(r3), y0) as f32) * yd0
                    + (dot_q2_half_q8_i32(unpack_q2_0_half32_u2(_mm_srli_si128::<8>(r3)), y1)
                        as f32)
                        * yd1;
                sum3 = f16_to_f32(b3.d).mul_add(sub3, sum3);
            }

            *out.get_unchecked_mut(r) = sum0;
            *out.get_unchecked_mut(r + 1) = sum1;
            *out.get_unchecked_mut(r + 2) = sum2;
            *out.get_unchecked_mut(r + 3) = sum3;
            r += 4;
        }

        while r < rows {
            *out.get_unchecked_mut(r) =
                vec_dot_q2_0_q8_0(&w[r * cols_blocks..(r + 1) * cols_blocks], x);
            r += 1;
        }
    }
}

/// Multi-token `AVX2 + FMA` `gemm_q2_0_q8_0` (`0-ULP` bit-exact with `generic::gemm_q2_0_q8_0`).
///
/// # Safety
/// Caller must ensure `avx2` and `fma` CPU features are supported.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_q2_0_q8_0(
    w: &[tm_core::BlockQ2_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
    n_tok: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), n_tok * cols_blocks * 2);
    assert_eq!(out.len(), n_tok * rows);

    // SAFETY: `avx2,fma` guaranteed by caller; unpacks each `BlockQ2_0` once across up to 4 tokens!
    unsafe {
        let mut t0 = 0usize;
        while t0 < n_tok {
            let bt = (n_tok - t0).min(4);
            for r in 0..rows {
                let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
                let mut sums = [0.0f32; 4];
                for (i, xb) in w_row.iter().enumerate() {
                    let raw = _mm_loadu_si128(xb.qs.as_ptr().cast::<__m128i>());
                    let u2_0 = unpack_q2_0_half32_u2(raw);
                    let u2_1 = unpack_q2_0_half32_u2(_mm_srli_si128::<8>(raw));
                    let wd = f16_to_f32(xb.d);
                    for (ti, st) in sums.iter_mut().enumerate().take(bt) {
                        let yb0 = x.get_unchecked((t0 + ti) * cols_blocks * 2 + 2 * i);
                        let yb1 = x.get_unchecked((t0 + ti) * cols_blocks * 2 + 2 * i + 1);
                        let y0 = _mm256_loadu_si256(yb0.qs.as_ptr().cast::<__m256i>());
                        let y1 = _mm256_loadu_si256(yb1.qs.as_ptr().cast::<__m256i>());
                        let sub = (dot_q2_half_q8_i32(u2_0, y0) as f32) * f16_to_f32(yb0.d)
                            + (dot_q2_half_q8_i32(u2_1, y1) as f32) * f16_to_f32(yb1.d);
                        *st = wd.mul_add(sub, *st);
                    }
                }
                for (ti, &sv) in sums.iter().enumerate().take(bt) {
                    *out.get_unchecked_mut((t0 + ti) * rows + r) = sv;
                }
            }
            t0 += bt;
        }
    }
}
