//! Scalar reference implementations (`generic` ISA tier).
//!
//! Serves as the ground-truth arithmetic reference for all SIMD/GPU parity tests.
//! Algorithm matches ggml's `ggml_vec_dot_q4_0_q8_0` scalar path:
//! Copyright (c) 2023-2026 The ggml authors (MIT License).
//! Upstream reference: llama.cpp `ggml/src/ggml-cpu/quants.c` (commit `3cf03257`).

use tm_core::{BlockQ4_0, BlockQ8_0, QK8_0, f16_to_f32};

/// Scalar dot product between one row of `BlockQ4_0` and one vector of `BlockQ8_0`.
#[inline]
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let mut sumf = 0.0f32;

    for (bx, by) in x.iter().zip(y.iter()) {
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..(QK8_0 / 2) {
            let byte = bx.qs[j];
            let v0 = ((byte & 0x0F) as i32) - 8;
            let v1 = ((byte >> 4) as i32) - 8;

            sumi0 += v0 * (by.qs[j] as i32);
            sumi1 += v1 * (by.qs[j + QK8_0 / 2] as i32);
        }

        let sumi = sumi0 + sumi1;
        sumf += (sumi as f32) * (f16_to_f32(bx.d) * f16_to_f32(by.d));
    }

    sumf
}

/// Scalar GEMV: `out[r] = dot(w[r], x)` for `r in 0..rows`.
/// `w` has `rows * cols_blocks` blocks (row-major); `x` has `cols_blocks` blocks.
pub fn gemv_q4_0_q8_0(
    w: &[BlockQ4_0],
    x: &[BlockQ8_0],
    out: &mut [f32],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(w.len(), rows * cols_blocks);
    assert_eq!(x.len(), cols_blocks);
    assert_eq!(out.len(), rows);

    for r in 0..rows {
        let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
        out[r] = vec_dot_q4_0_q8_0(row, x);
    }
}

/// Scalar multi-token GEMM: `out[t * rows + r] = dot(w[r], x[t])`.
/// `w`: `rows * cols_blocks`; `x`: `n_tokens * cols_blocks`; `out`: `n_tokens * rows`.
pub fn gemm_q4_0_q8_0(
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

    for t in 0..n_tokens {
        let xt = &x[t * cols_blocks..(t + 1) * cols_blocks];
        let ot = &mut out[t * rows..(t + 1) * rows];
        gemv_q4_0_q8_0(w, xt, ot, rows, cols_blocks);
    }
}
