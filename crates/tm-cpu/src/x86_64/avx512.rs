//! `x86_64` `AVX-512 (F + BW + VL) + FMA` compute kernels (`100%` Rust stable, `0-ULP` bit-exact
//! with `generic`).
//!
//! Processes an entire 32-element `Q8_0` or `Q4_0` block in a single 512-bit ZMM register via
//! `_mm512_cvtepi8_epi16` + `_mm512_madd_epi16` (16 `i32` lanes, immune to `i16` overflow),
//! without requiring `avx512vnni` so Skylake-X / Cascade Lake Xeons (e.g. Google Colab) as well
//! as Ice Lake / Sapphire Rapids / Zen 4 / Zen 5 all get full 512-bit throughput.

use core::arch::x86_64::*;
use tm_core::{BlockQ4_0, BlockQ8_0, f16_to_f32};

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn hsum4_epi32_avx512(p0: __m512i, p1: __m512i, p2: __m512i, p3: __m512i) -> __m128i {
    // Fold each 512-bit (16 x i32) register to 256-bit (8 x i32), then horizontal-add 4 rows -> 128-bit.
    let q0 = _mm256_add_epi32(
        _mm512_castsi512_si256(p0),
        _mm512_extracti64x4_epi64::<1>(p0),
    );
    let q1 = _mm256_add_epi32(
        _mm512_castsi512_si256(p1),
        _mm512_extracti64x4_epi64::<1>(p1),
    );
    let q2 = _mm256_add_epi32(
        _mm512_castsi512_si256(p2),
        _mm512_extracti64x4_epi64::<1>(p2),
    );
    let q3 = _mm256_add_epi32(
        _mm512_castsi512_si256(p3),
        _mm512_extracti64x4_epi64::<1>(p3),
    );
    let h01 = _mm256_hadd_epi32(q0, q1);
    let h23 = _mm256_hadd_epi32(q2, q3);
    let h0123 = _mm256_hadd_epi32(h01, h23);
    _mm_add_epi32(
        _mm256_castsi256_si128(h0123),
        _mm256_extracti128_si256::<1>(h0123),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn unpack_q4_0_i16_avx512(q128: __m128i, mask_lo: __m128i, eight8: __m128i) -> __m512i {
    let lo8 = _mm_sub_epi8(_mm_and_si128(q128, mask_lo), eight8);
    let hi8 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16::<4>(q128), mask_lo), eight8);
    let q32 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo8), hi8);
    _mm512_cvtepi8_epi16(q32)
}

/// Single-row `AVX-512` `vec_dot_q8_0_q8_0` (`0-ULP` bit-exact with `generic::vec_dot_q8_0_q8_0`).
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, and `avx512vl` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn vec_dot_q8_0_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mut sumf = 0.0f32;

    // SAFETY: `ib` stays strictly `< nb == x.len() == y.len()`.
    unsafe {
        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let x32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(bx.qs.as_ptr() as *const __m256i));
            let y32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(by.qs.as_ptr() as *const __m256i));
            let sumi = _mm512_reduce_add_epi32(_mm512_madd_epi16(x32, y32));
            sumf += (sumi as f32) * (f16_to_f32(bx.d) * f16_to_f32(by.d));
        }
    }

    sumf
}

/// 8-row shared-activation `AVX-512` `gemv_q8_0_q8_0` (`0-ULP` bit-exact with `generic`).
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, and `avx512vl` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
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
        while r + 8 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let w4 = &w[(r + 4) * cols_blocks..(r + 5) * cols_blocks];
            let w5 = &w[(r + 5) * cols_blocks..(r + 6) * cols_blocks];
            let w6 = &w[(r + 6) * cols_blocks..(r + 7) * cols_blocks];
            let w7 = &w[(r + 7) * cols_blocks..(r + 8) * cols_blocks];
            let mut acc0 = _mm_setzero_ps();
            let mut acc1 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let y32 =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(by.qs.as_ptr() as *const __m256i));
                let dy = _mm_set1_ps(f16_to_f32(by.d));

                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);
                let p0 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b0.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p1 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b1.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p2 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b2.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p3 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b3.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let sum0 = _mm_cvtepi32_ps(hsum4_epi32_avx512(p0, p1, p2, p3));
                let dx0 = _mm_set_ps(
                    f16_to_f32(b3.d),
                    f16_to_f32(b2.d),
                    f16_to_f32(b1.d),
                    f16_to_f32(b0.d),
                );
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(sum0, _mm_mul_ps(dx0, dy)));

                let b4 = w4.get_unchecked(ib);
                let b5 = w5.get_unchecked(ib);
                let b6 = w6.get_unchecked(ib);
                let b7 = w7.get_unchecked(ib);
                let p4 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b4.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p5 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b5.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p6 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b6.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p7 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b7.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let sum1 = _mm_cvtepi32_ps(hsum4_epi32_avx512(p4, p5, p6, p7));
                let dx1 = _mm_set_ps(
                    f16_to_f32(b7.d),
                    f16_to_f32(b6.d),
                    f16_to_f32(b5.d),
                    f16_to_f32(b4.d),
                );
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(sum1, _mm_mul_ps(dx1, dy)));
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), acc0);
            _mm_storeu_ps(out.as_mut_ptr().add(r + 4), acc1);
            r += 8;
        }

        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut acc = _mm_setzero_ps();
            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let y32 =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(by.qs.as_ptr() as *const __m256i));
                let dy = _mm_set1_ps(f16_to_f32(by.d));
                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);
                let p0 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b0.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p1 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b1.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p2 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b2.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let p3 = _mm512_madd_epi16(
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(b3.qs.as_ptr() as *const __m256i)),
                    y32,
                );
                let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx512(p0, p1, p2, p3));
                let dx4 = _mm_set_ps(
                    f16_to_f32(b3.d),
                    f16_to_f32(b2.d),
                    f16_to_f32(b1.d),
                    f16_to_f32(b0.d),
                );
                acc = _mm_add_ps(acc, _mm_mul_ps(sum4, _mm_mul_ps(dx4, dy)));
            }
            _mm_storeu_ps(out.as_mut_ptr().add(r), acc);
            r += 4;
        }

        while r < rows {
            let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q8_0_q8_0(w_row, x);
            r += 1;
        }
    }
}

/// Multi-token shared-weight `AVX-512` `gemm_q8_0_q8_0` (`4 rows x 4 tokens`, `0-ULP` bit-exact).
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, and `avx512vl` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
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
        // SAFETY: Same target features guaranteed by caller.
        unsafe { gemv_q8_0_q8_0(w, x, out, rows, cols_blocks) };
        return;
    }

    // SAFETY: Bounds verified by assertions above.
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
                    let x0 =
                        _mm512_cvtepi8_epi16(_mm256_loadu_si256(b0.qs.as_ptr() as *const __m256i));
                    let x1 =
                        _mm512_cvtepi8_epi16(_mm256_loadu_si256(b1.qs.as_ptr() as *const __m256i));
                    let x2 =
                        _mm512_cvtepi8_epi16(_mm256_loadu_si256(b2.qs.as_ptr() as *const __m256i));
                    let x3 =
                        _mm512_cvtepi8_epi16(_mm256_loadu_si256(b3.qs.as_ptr() as *const __m256i));
                    let dx4 = _mm_set_ps(
                        f16_to_f32(b3.d),
                        f16_to_f32(b2.d),
                        f16_to_f32(b1.d),
                        f16_to_f32(b0.d),
                    );

                    for (ti, acc_ti) in acc.iter_mut().enumerate().take(bt) {
                        let by = x.get_unchecked((t0 + ti) * cols_blocks + ib);
                        let y32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            by.qs.as_ptr() as *const __m256i
                        ));
                        let p0 = _mm512_madd_epi16(x0, y32);
                        let p1 = _mm512_madd_epi16(x1, y32);
                        let p2 = _mm512_madd_epi16(x2, y32);
                        let p3 = _mm512_madd_epi16(x3, y32);
                        let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx512(p0, p1, p2, p3));
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

/// Single-row `AVX-512 + FMA` `vec_dot_q4_0_q8_0` (`0-ULP` bit-exact with `generic`).
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, `avx512vl`, and `fma` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
pub unsafe fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let nb = x.len();
    let mask_lo = _mm_set1_epi8(0x0F);
    let eight8 = _mm_set1_epi8(8);
    let mut sumf = 0.0f32;

    // SAFETY: `ib` stays strictly `< nb == x.len() == y.len()`.
    unsafe {
        for ib in 0..nb {
            let bx = x.get_unchecked(ib);
            let by = y.get_unchecked(ib);
            let x32 = unpack_q4_0_i16_avx512(
                _mm_loadu_si128(bx.qs.as_ptr() as *const __m128i),
                mask_lo,
                eight8,
            );
            let y32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(by.qs.as_ptr() as *const __m256i));
            let sumi = _mm512_reduce_add_epi32(_mm512_madd_epi16(x32, y32));
            let scale = f16_to_f32(bx.d) * f16_to_f32(by.d);
            sumf = (sumi as f32).mul_add(scale, sumf);
        }
    }

    sumf
}

/// 4-row shared-activation `AVX-512 + FMA` `gemv_q4_0_q8_0` (`0-ULP` bit-exact with `generic`).
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, `avx512vl`, and `fma` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
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

    // SAFETY: Bounds verified by assertions above.
    unsafe {
        while r + 4 <= rows {
            let w0 = &w[r * cols_blocks..(r + 1) * cols_blocks];
            let w1 = &w[(r + 1) * cols_blocks..(r + 2) * cols_blocks];
            let w2 = &w[(r + 2) * cols_blocks..(r + 3) * cols_blocks];
            let w3 = &w[(r + 3) * cols_blocks..(r + 4) * cols_blocks];
            let mut acc4 = _mm_setzero_ps();

            for ib in 0..cols_blocks {
                let by = x.get_unchecked(ib);
                let y32 =
                    _mm512_cvtepi8_epi16(_mm256_loadu_si256(by.qs.as_ptr() as *const __m256i));
                let dy = _mm_set1_ps(f16_to_f32(by.d));

                let b0 = w0.get_unchecked(ib);
                let b1 = w1.get_unchecked(ib);
                let b2 = w2.get_unchecked(ib);
                let b3 = w3.get_unchecked(ib);
                let p0 = _mm512_madd_epi16(
                    unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b0.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    ),
                    y32,
                );
                let p1 = _mm512_madd_epi16(
                    unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b1.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    ),
                    y32,
                );
                let p2 = _mm512_madd_epi16(
                    unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b2.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    ),
                    y32,
                );
                let p3 = _mm512_madd_epi16(
                    unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b3.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    ),
                    y32,
                );
                let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx512(p0, p1, p2, p3));
                let dx4 = _mm_set_ps(
                    f16_to_f32(b3.d),
                    f16_to_f32(b2.d),
                    f16_to_f32(b1.d),
                    f16_to_f32(b0.d),
                );
                acc4 = _mm_fmadd_ps(sum4, _mm_mul_ps(dx4, dy), acc4);
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), acc4);
            r += 4;
        }

        while r < rows {
            let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            *out.get_unchecked_mut(r) = vec_dot_q4_0_q8_0(w_row, x);
            r += 1;
        }
    }
}

/// Multi-token shared-weight `AVX-512 + FMA` `gemm_q4_0_q8_0` (`4 rows x 4 tokens`, `0-ULP` bit-exact).
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, `avx512vl`, and `fma` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
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

    // SAFETY: Bounds verified by assertions above.
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
                    let x0 = unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b0.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let x1 = unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b1.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let x2 = unpack_q4_0_i16_avx512(
                        _mm_loadu_si128(b2.qs.as_ptr() as *const __m128i),
                        mask_lo,
                        eight8,
                    );
                    let x3 = unpack_q4_0_i16_avx512(
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
                        let y32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                            by.qs.as_ptr() as *const __m256i
                        ));
                        let p0 = _mm512_madd_epi16(x0, y32);
                        let p1 = _mm512_madd_epi16(x1, y32);
                        let p2 = _mm512_madd_epi16(x2, y32);
                        let p3 = _mm512_madd_epi16(x3, y32);
                        let sum4 = _mm_cvtepi32_ps(hsum4_epi32_avx512(p0, p1, p2, p3));
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
