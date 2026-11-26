//! `x86_64` `AVX-512 (F + BW + VL) + FMA` compute kernels (`100%` Rust stable, `0-ULP` bit-exact
//! with `generic`).
//!
//! Processes an entire 32-element `Q8_0` or `Q4_0` block in a single 512-bit ZMM register via
//! `_mm512_cvtepi8_epi16` + `_mm512_madd_epi16` (16 `i32` lanes, immune to `i16` overflow),
//! without requiring `avx512vnni` so Skylake-X / Cascade Lake Xeons (e.g. Google Colab) as well
//! as Ice Lake / Sapphire Rapids / Zen 4 / Zen 5 all get full 512-bit throughput.

use core::arch::x86_64::*;
use tm_core::{BlockQ4_0, BlockQ8_0, QK8_0, f16_to_f32};

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

#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
unsafe fn hsum2x2_8wide_pairwise_ps(p01: __m512, p23: __m512) -> __m128 {
    let a0 = _mm512_castps512_ps256(p01);
    let a1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64::<1>(_mm512_castps_si512(p01)));
    let a2 = _mm512_castps512_ps256(p23);
    let a3 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64::<1>(_mm512_castps_si512(p23)));
    let s4_0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps::<1>(a0));
    let s4_1 = _mm_add_ps(_mm256_castps256_ps128(a1), _mm256_extractf128_ps::<1>(a1));
    let s4_2 = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps::<1>(a2));
    let s4_3 = _mm_add_ps(_mm256_castps256_ps128(a3), _mm256_extractf128_ps::<1>(a3));
    let h01 = _mm_hadd_ps(s4_0, s4_1);
    let h23 = _mm_hadd_ps(s4_2, s4_3);
    _mm_hadd_ps(h01, h23)
}

/// 8-row shared-activation `AVX-512` `gemv_f32_q8_0` (`0-ULP` bit-exact with `generic::gemv_f32_q8_0`).
///
/// Pairs two rows' 8-element slices into each 512-bit ZMM FMA (`4` ZMM accumulators = `8` rows),
/// preserving the exact 8-wide `c = 0..4` FMA chain and pairwise reduction of `generic::dot_f32_q8_0`.
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, `avx512vl`, and `fma` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
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

                let mut acc01 = _mm512_setzero_ps();
                let mut acc23 = _mm512_setzero_ps();
                let mut acc45 = _mm512_setzero_ps();
                let mut acc67 = _mm512_setzero_ps();

                for c in 0..4 {
                    let off = c * 8;
                    let q256 = _mm256_loadu_ps(q_ptr.add(off));
                    let q512 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(q256), q256);

                    let k01 = _mm_unpacklo_epi64(
                        _mm_loadl_epi64(b0.qs.as_ptr().add(off) as *const __m128i),
                        _mm_loadl_epi64(b1.qs.as_ptr().add(off) as *const __m128i),
                    );
                    let k23 = _mm_unpacklo_epi64(
                        _mm_loadl_epi64(b2.qs.as_ptr().add(off) as *const __m128i),
                        _mm_loadl_epi64(b3.qs.as_ptr().add(off) as *const __m128i),
                    );
                    let k45 = _mm_unpacklo_epi64(
                        _mm_loadl_epi64(b4.qs.as_ptr().add(off) as *const __m128i),
                        _mm_loadl_epi64(b5.qs.as_ptr().add(off) as *const __m128i),
                    );
                    let k67 = _mm_unpacklo_epi64(
                        _mm_loadl_epi64(b6.qs.as_ptr().add(off) as *const __m128i),
                        _mm_loadl_epi64(b7.qs.as_ptr().add(off) as *const __m128i),
                    );

                    acc01 =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(k01)), q512, acc01);
                    acc23 =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(k23)), q512, acc23);
                    acc45 =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(k45)), q512, acc45);
                    acc67 =
                        _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(k67)), q512, acc67);
                }

                let bsum0 = hsum2x2_8wide_pairwise_ps(acc01, acc23);
                let bsum1 = hsum2x2_8wide_pairwise_ps(acc45, acc67);
                let d0 = _mm_set_ps(
                    f16_to_f32(b3.d),
                    f16_to_f32(b2.d),
                    f16_to_f32(b1.d),
                    f16_to_f32(b0.d),
                );
                let d1 = _mm_set_ps(
                    f16_to_f32(b7.d),
                    f16_to_f32(b6.d),
                    f16_to_f32(b5.d),
                    f16_to_f32(b4.d),
                );
                tot0 = _mm_fmadd_ps(bsum0, d0, tot0);
                tot1 = _mm_fmadd_ps(bsum1, d1, tot1);
            }

            _mm_storeu_ps(out.as_mut_ptr().add(r), tot0);
            _mm_storeu_ps(out.as_mut_ptr().add(r + 4), tot1);
            r += 8;
        }

        if r < rows {
            super::avx2::gemv_f32_q8_0(
                &w[r * cols_blocks..],
                x,
                &mut out[r..],
                rows - r,
                cols_blocks,
            );
        }
    }
}

/// Multi-token L1-pinned `4 rows x 4 tokens` `AVX-512` `gemm_f32_q8_0` (`0-ULP` bit-exact).
///
/// Steps `r` by 4 in the outer loop (`< 10 KiB` of weights pinned in L1D cache across all tokens)
/// and processes `4` tokens per inner pass using `8` 512-bit ZMM accumulators (`4 rows x 2 token-pairs`),
/// converting each `BlockQ8_0` from `i8` to `f32` **once** across all 4 tokens!
///
/// # Safety
/// Caller must ensure `avx512f`, `avx512bw`, `avx512vl`, and `fma` CPU features are supported.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
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
            while t0 + 4 <= n_tokens {
                let x0 = x.as_ptr().add(t0 * cols);
                let x1 = x.as_ptr().add((t0 + 1) * cols);
                let x2 = x.as_ptr().add((t0 + 2) * cols);
                let x3 = x.as_ptr().add((t0 + 3) * cols);

                // `tot_r[i]` holds `[sum(r_i, t0), sum(r_i, t1), sum(r_i, t2), sum(r_i, t3)]`
                let mut tot0 = _mm_setzero_ps();
                let mut tot1 = _mm_setzero_ps();
                let mut tot2 = _mm_setzero_ps();
                let mut tot3 = _mm_setzero_ps();

                for ib in 0..cols_blocks {
                    let b0 = w0.get_unchecked(ib);
                    let b1 = w1.get_unchecked(ib);
                    let b2 = w2.get_unchecked(ib);
                    let b3 = w3.get_unchecked(ib);
                    let base = ib * QK8_0;

                    let mut a0_01 = _mm512_setzero_ps();
                    let mut a0_23 = _mm512_setzero_ps();
                    let mut a1_01 = _mm512_setzero_ps();
                    let mut a1_23 = _mm512_setzero_ps();
                    let mut a2_01 = _mm512_setzero_ps();
                    let mut a2_23 = _mm512_setzero_ps();
                    let mut a3_01 = _mm512_setzero_ps();
                    let mut a3_23 = _mm512_setzero_ps();

                    for c in 0..4 {
                        let off = c * 8;
                        let q0 = _mm256_loadu_ps(x0.add(base + off));
                        let q1 = _mm256_loadu_ps(x1.add(base + off));
                        let q2 = _mm256_loadu_ps(x2.add(base + off));
                        let q3 = _mm256_loadu_ps(x3.add(base + off));
                        let q01 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(q0), q1);
                        let q23 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(q2), q3);

                        let k0_256 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b0.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        let k1_256 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b1.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        let k2_256 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b2.qs.as_ptr().add(off) as *const __m128i,
                        )));
                        let k3_256 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                            b3.qs.as_ptr().add(off) as *const __m128i,
                        )));

                        let k0 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(k0_256), k0_256);
                        let k1 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(k1_256), k1_256);
                        let k2 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(k2_256), k2_256);
                        let k3 = _mm512_insertf32x8::<1>(_mm512_castps256_ps512(k3_256), k3_256);

                        a0_01 = _mm512_fmadd_ps(k0, q01, a0_01);
                        a0_23 = _mm512_fmadd_ps(k0, q23, a0_23);
                        a1_01 = _mm512_fmadd_ps(k1, q01, a1_01);
                        a1_23 = _mm512_fmadd_ps(k1, q23, a1_23);
                        a2_01 = _mm512_fmadd_ps(k2, q01, a2_01);
                        a2_23 = _mm512_fmadd_ps(k2, q23, a2_23);
                        a3_01 = _mm512_fmadd_ps(k3, q01, a3_01);
                        a3_23 = _mm512_fmadd_ps(k3, q23, a3_23);
                    }

                    tot0 = _mm_fmadd_ps(
                        hsum2x2_8wide_pairwise_ps(a0_01, a0_23),
                        _mm_set1_ps(f16_to_f32(b0.d)),
                        tot0,
                    );
                    tot1 = _mm_fmadd_ps(
                        hsum2x2_8wide_pairwise_ps(a1_01, a1_23),
                        _mm_set1_ps(f16_to_f32(b1.d)),
                        tot1,
                    );
                    tot2 = _mm_fmadd_ps(
                        hsum2x2_8wide_pairwise_ps(a2_01, a2_23),
                        _mm_set1_ps(f16_to_f32(b2.d)),
                        tot2,
                    );
                    tot3 = _mm_fmadd_ps(
                        hsum2x2_8wide_pairwise_ps(a3_01, a3_23),
                        _mm_set1_ps(f16_to_f32(b3.d)),
                        tot3,
                    );
                }

                // 4x4 transpose from `(row, token)` to contiguous `(token, row)` stores
                let tmp0 = _mm_unpacklo_ps(tot0, tot1);
                let tmp1 = _mm_unpackhi_ps(tot0, tot1);
                let tmp2 = _mm_unpacklo_ps(tot2, tot3);
                let tmp3 = _mm_unpackhi_ps(tot2, tot3);
                _mm_storeu_ps(
                    out.as_mut_ptr().add(t0 * rows + r),
                    _mm_movelh_ps(tmp0, tmp2),
                );
                _mm_storeu_ps(
                    out.as_mut_ptr().add((t0 + 1) * rows + r),
                    _mm_movehl_ps(tmp2, tmp0),
                );
                _mm_storeu_ps(
                    out.as_mut_ptr().add((t0 + 2) * rows + r),
                    _mm_movelh_ps(tmp1, tmp3),
                );
                _mm_storeu_ps(
                    out.as_mut_ptr().add((t0 + 3) * rows + r),
                    _mm_movehl_ps(tmp3, tmp1),
                );
                t0 += 4;
            }

            while t0 < n_tokens {
                let xt = &x[t0 * cols..(t0 + 1) * cols];
                let dst = &mut out[t0 * rows + r..t0 * rows + r + 4];
                super::avx2::gemv_f32_q8_0(
                    &w[r * cols_blocks..(r + 4) * cols_blocks],
                    xt,
                    dst,
                    4,
                    cols_blocks,
                );
                t0 += 1;
            }

            r += 4;
        }

        while r < rows {
            let w_row = &w[r * cols_blocks..(r + 1) * cols_blocks];
            for t in 0..n_tokens {
                let xt = &x[t * cols..(t + 1) * cols];
                *out.get_unchecked_mut(t * rows + r) = super::avx2::dot_f32_q8_0(xt, w_row);
            }
            r += 1;
        }
    }
}
