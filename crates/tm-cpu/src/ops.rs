//! Core CPU Transformer operators (RMSNorm, RoPE, SwiGLU, GQA attention,
//! and dtype-dispatched quantized matrix-vector multiplication).
//!
//! All operators are zero-allocation on the token generation path: callers pass
//! pre-allocated scratch buffers.

use tm_core::{
    BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, DType, QK_K, QK4_0,
    QK4_1, QK8_0, f16_to_f32, quantize_row_q8_k,
};

use crate::{KernelTable, ThreadPool, generic};

/// Maximum batch size per CPU prefill chunk (kept compact so L2 tiles + scratch stay < 300 KB).
pub const MAX_CPU_BATCH: usize = 32;

/// Rotary Position Embedding pairing convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Interleaved adjacent pairs `(2i, 2i + 1)` — used by GGUF `llama` models.
    Norm,
    /// Split-half pairs `(i, i + head_dim / 2)` — used by GGUF `qwen2` / `qwen3`.
    Neox,
}

/// Pre-allocated activation quantization scratch buffers (reused across every matvec / matmul).
pub struct MatvecScratch {
    pub q8_0: Vec<BlockQ8_0>,
    pub q8_1: Vec<BlockQ8_1>,
    pub q8_k: Vec<BlockQ8K>,
}

impl MatvecScratch {
    /// Allocates scratch space large enough for vectors up to `max_cols` elements
    /// and batches up to `MAX_CPU_BATCH` tokens.
    pub fn new(max_cols: usize) -> Self {
        let nb32 = max_cols.div_ceil(QK8_0) * MAX_CPU_BATCH;
        let nb256 = max_cols.div_ceil(QK_K) * MAX_CPU_BATCH;
        Self {
            q8_0: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; nb32],
            q8_1: vec![
                BlockQ8_1 {
                    d: 0,
                    s: 0,
                    qs: [0; 32]
                };
                nb32
            ],
            q8_k: vec![
                BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16]
                };
                nb256
            ],
        }
    }
}

/// Dispatches quantized `y = W * x` across `pool` strictly by `dtype` of tensor `W`.
///
/// Zero heap allocations: activation quantization uses `scratch`.
#[allow(clippy::too_many_arguments)]
pub fn matvec_f32(
    pool: &ThreadPool,
    kt: &KernelTable,
    dtype: DType,
    w_bytes: &[u8],
    x: &[f32],
    scratch: &mut MatvecScratch,
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert_eq!(w_bytes.len(), rows * dtype.row_bytes(cols));

    // Align chunk to 4 rows for 4-row unrolled GEMV and 1 chunk per thread on homogeneous cores.
    let n_thr = pool.num_threads();
    let chunk = rows.div_ceil(n_thr).next_multiple_of(4).max(4);
    let out_ptr = out.as_mut_ptr() as usize;

    match dtype {
        DType::Q4_0 => {
            let cols_blocks = cols / QK4_0;
            let xq = &mut scratch.q8_0[..cols_blocks];
            (kt.quantize_q8_0)(x, xq);
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>()),
                "Q4_0 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment of `w_bytes` are verified above; `BlockQ4_0` is `#[repr(C)]` POD.
            let w: &[BlockQ4_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_0,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0_q8_0)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Q4_1 => {
            let cols_blocks = cols / QK4_1;
            let xq = &mut scratch.q8_1[..cols_blocks];
            (kt.quantize_q8_1)(x, xq);
            let xq_ref: &[BlockQ8_1] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_1>()),
                "Q4_1 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment of `w_bytes` are verified above; `BlockQ4_1` is `#[repr(C)]` POD.
            let w: &[BlockQ4_1] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_1,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_1_q8_1)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Q8_0 => {
            let cols_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..cols_blocks];
            (kt.quantize_q8_0)(x, xq);
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ8_0>()),
                "Q8_0 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment of `w_bytes` are verified above; `BlockQ8_0` is `#[repr(C)]` POD.
            let w: &[BlockQ8_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ8_0,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q8_0_q8_0)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Q5K => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..cols_blocks];
            quantize_row_q8_k(x, xq);
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ5K>()),
                "Q5K tensor pointer misaligned"
            );
            // SAFETY: Length and alignment of `w_bytes` are verified above; `BlockQ5K` is `#[repr(C)]` POD.
            let w: &[BlockQ5K] = unsafe {
                core::slice::from_raw_parts(w_bytes.as_ptr() as *const BlockQ5K, rows * cols_blocks)
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q5_k_q8_k)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Q6K => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..cols_blocks];
            quantize_row_q8_k(x, xq);
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ6K>()),
                "Q6K tensor pointer misaligned"
            );
            // SAFETY: Length and alignment of `w_bytes` are verified above; `BlockQ6K` is `#[repr(C)]` POD.
            let w: &[BlockQ6K] = unsafe {
                core::slice::from_raw_parts(w_bytes.as_ptr() as *const BlockQ6K, rows * cols_blocks)
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q6_k_q8_k)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::F32 => {
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()),
                "F32 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above.
            let w: &[f32] =
                unsafe { core::slice::from_raw_parts(w_bytes.as_ptr() as *const f32, rows * cols) };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                for i in 0..len {
                    o[i] = generic::vec_dot_f32(&w[(r0 + i) * cols..(r0 + i + 1) * cols], x);
                }
            });
        }
        DType::F16 => {
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<u16>()),
                "F16 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above.
            let w: &[u16] =
                unsafe { core::slice::from_raw_parts(w_bytes.as_ptr() as *const u16, rows * cols) };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                for i in 0..len {
                    let row = &w[(r0 + i) * cols..(r0 + i + 1) * cols];
                    let mut sum = 0.0f32;
                    for (&wh, &xv) in row.iter().zip(x.iter()) {
                        sum += f16_to_f32(wh) * xv;
                    }
                    o[i] = sum;
                }
            });
        }
        other => panic!("matvec_f32: unsupported weight dtype {other:?}"),
    }
}

/// RMS normalization: `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(weight.len(), n);

    let mut ss = 0.0f64;
    for &v in x {
        let vf = v as f64;
        ss += vf * vf;
    }
    let scale = (1.0 / ((ss / n as f64) + eps as f64).sqrt()) as f32;
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place Rotary Position Embedding across `n_heads` heads of dimension `head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn rope_inplace(
    vec: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    freq_base: f32,
    freq_factors: Option<&[f32]>,
    rope_type: RopeType,
) {
    debug_assert_eq!(vec.len(), n_heads * head_dim);
    let half = rope_dim / 2;
    let pos_f = pos as f64;
    let base_f = freq_base as f64;
    let dim_f = rope_dim as f64;

    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        match rope_type {
            RopeType::Norm => {
                for i in 0..half {
                    let mut freq = base_f.powf(-2.0 * (i as f64) / dim_f);
                    if let Some(factors) = freq_factors {
                        freq /= factors[i] as f64;
                    }
                    let theta = pos_f * freq;
                    let (sin_t, cos_t) = theta.sin_cos();
                    let (sin_t, cos_t) = (sin_t as f32, cos_t as f32);
                    let x0 = head[2 * i];
                    let x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cos_t - x1 * sin_t;
                    head[2 * i + 1] = x0 * sin_t + x1 * cos_t;
                }
            }
            RopeType::Neox => {
                for i in 0..half {
                    let mut freq = base_f.powf(-2.0 * (i as f64) / dim_f);
                    if let Some(factors) = freq_factors {
                        freq /= factors[i] as f64;
                    }
                    let theta = pos_f * freq;
                    let (sin_t, cos_t) = theta.sin_cos();
                    let (sin_t, cos_t) = (sin_t as f32, cos_t as f32);
                    let x0 = head[i];
                    let x1 = head[i + half];
                    head[i] = x0 * cos_t - x1 * sin_t;
                    head[i + half] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    }
}

/// In-place SwiGLU activation: `gate[i] = silu(gate[i]) * up[i]`.
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        let x = *g;
        let silu = x / (1.0 + (-x).exp());
        *g = silu * u;
    }
}

/// In-place numerically stable softmax.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Single-token GQA attention against F16 KV cache (layout: `[seq, n_kv_heads * head_dim]`).
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv(
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scores: &mut [f32],
) {
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sc = &mut scores[..seq_len];

    for h in 0..n_heads {
        let kv_h = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];

        for (t, s_slot) in sc.iter_mut().enumerate() {
            let k_off = t * kv_dim + kv_h * head_dim;
            let kh = &k_cache[k_off..k_off + head_dim];
            *s_slot = (kt.dot_f32_f16)(qh, kh) * scale;
        }

        softmax_inplace(sc);

        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        for (t, &w) in sc.iter().enumerate() {
            let v_off = t * kv_dim + kv_h * head_dim;
            let vh = &v_cache[v_off..v_off + head_dim];
            (kt.axpy_f32_f16)(oh, w, vh);
        }
    }
}

/// Precomputes `cos(pos * inv_freq[i])` and `sin(pos * inv_freq[i])` once per token position.
pub fn rope_compute_cos_sin(
    inv_freqs: &[f64],
    pos: usize,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) {
    let pos_f = pos as f64;
    for (i, &freq) in inv_freqs.iter().enumerate() {
        let (s, c) = (pos_f * freq).sin_cos();
        cos_out[i] = c as f32;
        sin_out[i] = s as f32;
    }
}

/// Applies RoPE in-place using precomputed `cos` / `sin` tables (zero `powf` / `sin_cos` per layer).
pub fn rope_inplace_tables(
    vec: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    cos: &[f32],
    sin: &[f32],
    rope_type: RopeType,
) {
    debug_assert_eq!(vec.len(), n_heads * head_dim);
    let half = rope_dim / 2;
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);

    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        match rope_type {
            RopeType::Norm => {
                for i in 0..half {
                    let c = cos[i];
                    let s = sin[i];
                    let x0 = head[2 * i];
                    let x1 = head[2 * i + 1];
                    head[2 * i] = x0 * c - x1 * s;
                    head[2 * i + 1] = x0 * s + x1 * c;
                }
            }
            RopeType::Neox => {
                for i in 0..half {
                    let c = cos[i];
                    let s = sin[i];
                    let x0 = head[i];
                    let x1 = head[i + half];
                    head[i] = x0 * c - x1 * s;
                    head[i + half] = x0 * s + x1 * c;
                }
            }
        }
    }
}

/// Parallelized single-token GQA attention across heads (F16 KV cache).
/// `scores` must have length `>= n_heads * max_ctx` so each head has a disjoint scratch slice.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    max_ctx: usize,
    scores: &mut [f32],
) {
    // Below 24 tokens run inline to avoid waking threads; above 24 parallelize across all cores.
    if pool.num_threads() == 1 || seq_len < 24 {
        gqa_attention_f16_kv(
            kt, out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, scores,
        );
        return;
    }

    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out_ptr = out.as_mut_ptr() as usize;
    let scores_ptr = scores.as_mut_ptr() as usize;
    let chunk = n_heads.div_ceil(pool.num_threads()).max(1);

    pool.parallel_for(n_heads, chunk, |h0, h1| {
        for h in h0..h1 {
            let kv_h = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            // SAFETY: Disjoint `[h * max_ctx .. h * max_ctx + seq_len)` per head `h`.
            let sc = unsafe {
                core::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(h * max_ctx), seq_len)
            };

            for (t, s_slot) in sc.iter_mut().enumerate() {
                let k_off = t * kv_dim + kv_h * head_dim;
                let kh = &k_cache[k_off..k_off + head_dim];
                *s_slot = (kt.dot_f32_f16)(qh, kh) * scale;
            }

            softmax_inplace(sc);

            // SAFETY: Disjoint `[h * head_dim .. (h + 1) * head_dim)` per head `h`.
            let oh = unsafe {
                core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(h * head_dim), head_dim)
            };
            oh.fill(0.0);
            for (t, &w) in sc.iter().enumerate() {
                let v_off = t * kv_dim + kv_h * head_dim;
                let vh = &v_cache[v_off..v_off + head_dim];
                (kt.axpy_f32_f16)(oh, w, vh);
            }
        }
    });
}

/// Fused 2-matrix projection (`w_a`, `w_b`) sharing the exact same input vector `x`
/// (e.g. `ffn_gate` + `ffn_up`). Quantizes `x` ONCE and executes both projections
/// inside a single `ThreadPool` barrier when both weights are `Q4_0`.
#[allow(clippy::too_many_arguments)]
pub fn matvec_pair_f32(
    pool: &ThreadPool,
    kt: &KernelTable,
    dtype_a: DType,
    wa_bytes: &[u8],
    rows_a: usize,
    out_a: &mut [f32],
    dtype_b: DType,
    wb_bytes: &[u8],
    rows_b: usize,
    out_b: &mut [f32],
    x: &[f32],
    scratch: &mut MatvecScratch,
    cols: usize,
) {
    if dtype_a == DType::Q4_0 && dtype_b == DType::Q4_0 {
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..cols_blocks];
        (kt.quantize_q8_0)(x, xq);
        let xq_ref: &[BlockQ8_0] = xq;

        assert!(
            (wa_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
                && (wb_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
        );
        // SAFETY: Lengths and alignment verified; `BlockQ4_0` is `#[repr(C)]` POD.
        let wa: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wa_bytes.as_ptr() as *const BlockQ4_0, rows_a * cols_blocks)
        };
        let wb: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0, rows_b * cols_blocks)
        };

        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let total = rows_a + rows_b;
        let chunk = total
            .div_ceil(pool.num_threads())
            .next_multiple_of(4)
            .max(4);

        pool.parallel_for(total, chunk, |s, e| {
            if s < rows_a {
                let a1 = e.min(rows_a);
                let len = a1 - s;
                // SAFETY: Disjoint `[s..a1)` range in `out_a`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((oa_ptr as *mut f32).add(s), len) };
                (kt.gemv_q4_0_q8_0)(
                    &wa[s * cols_blocks..a1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
            if e > rows_a {
                let b0 = s.saturating_sub(rows_a);
                let b1 = e - rows_a;
                let len = b1 - b0;
                // SAFETY: Disjoint `[b0..b1)` range in `out_b`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((ob_ptr as *mut f32).add(b0), len) };
                (kt.gemv_q4_0_q8_0)(
                    &wb[b0 * cols_blocks..b1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
        });
        return;
    }

    matvec_f32(pool, kt, dtype_a, wa_bytes, x, scratch, out_a, rows_a, cols);
    matvec_f32(pool, kt, dtype_b, wb_bytes, x, scratch, out_b, rows_b, cols);
}

/// Fused 3-matrix projection (`wq`, `wk`, `wv`) sharing the exact same input vector `x`.
/// Quantizes `x` ONCE and executes Q, K, V inside a single `ThreadPool` barrier when all are `Q4_0`.
#[allow(clippy::too_many_arguments)]
pub fn matvec_triple_f32(
    pool: &ThreadPool,
    kt: &KernelTable,
    dtype_a: DType,
    wa_bytes: &[u8],
    rows_a: usize,
    out_a: &mut [f32],
    dtype_b: DType,
    wb_bytes: &[u8],
    rows_b: usize,
    out_b: &mut [f32],
    dtype_c: DType,
    wc_bytes: &[u8],
    rows_c: usize,
    out_c: &mut [f32],
    x: &[f32],
    scratch: &mut MatvecScratch,
    cols: usize,
) {
    if dtype_a == DType::Q4_0 && dtype_b == DType::Q4_0 && dtype_c == DType::Q4_0 {
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..cols_blocks];
        (kt.quantize_q8_0)(x, xq);
        let xq_ref: &[BlockQ8_0] = xq;

        assert!(
            (wa_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
                && (wb_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
                && (wc_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
        );
        // SAFETY: Lengths and alignment verified; `BlockQ4_0` is `#[repr(C)]` POD.
        let wa: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wa_bytes.as_ptr() as *const BlockQ4_0, rows_a * cols_blocks)
        };
        let wb: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0, rows_b * cols_blocks)
        };
        let wc: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wc_bytes.as_ptr() as *const BlockQ4_0, rows_c * cols_blocks)
        };

        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let oc_ptr = out_c.as_mut_ptr() as usize;
        let ab = rows_a + rows_b;
        let total = ab + rows_c;
        let chunk = total
            .div_ceil(pool.num_threads())
            .next_multiple_of(4)
            .max(4);

        pool.parallel_for(total, chunk, |s, e| {
            if s < rows_a {
                let a1 = e.min(rows_a);
                let len = a1 - s;
                // SAFETY: Disjoint `[s..a1)` range in `out_a`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((oa_ptr as *mut f32).add(s), len) };
                (kt.gemv_q4_0_q8_0)(
                    &wa[s * cols_blocks..a1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
            if s < ab && e > rows_a {
                let b0 = s.saturating_sub(rows_a);
                let b1 = e.min(ab) - rows_a;
                let len = b1 - b0;
                // SAFETY: Disjoint `[b0..b1)` range in `out_b`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((ob_ptr as *mut f32).add(b0), len) };
                (kt.gemv_q4_0_q8_0)(
                    &wb[b0 * cols_blocks..b1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
            if e > ab {
                let c0 = s.saturating_sub(ab);
                let c1 = e - ab;
                let len = c1 - c0;
                // SAFETY: Disjoint `[c0..c1)` range in `out_c`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((oc_ptr as *mut f32).add(c0), len) };
                (kt.gemv_q4_0_q8_0)(
                    &wc[c0 * cols_blocks..c1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
        });
        return;
    }

    matvec_f32(pool, kt, dtype_a, wa_bytes, x, scratch, out_a, rows_a, cols);
    matvec_f32(pool, kt, dtype_b, wb_bytes, x, scratch, out_b, rows_b, cols);
    matvec_f32(pool, kt, dtype_c, wc_bytes, x, scratch, out_c, rows_c, cols);
}

/// Batched quantized matrix multiplication `out[n_tok, rows] = x[n_tok, cols] * W^T[rows, cols]`
/// across `pool`, using L2-cached weight tiles and ISA-dispatched `gemm_q4_0_q8_0` (`smmla`/`sdot`).
///
/// Zero heap allocations: uses pre-allocated `scratch` (`n_tok <= MAX_CPU_BATCH`).
#[allow(clippy::too_many_arguments)]
pub fn matmul_f32(
    pool: &ThreadPool,
    kt: &KernelTable,
    dtype: DType,
    w_bytes: &[u8],
    x: &[f32],
    scratch: &mut MatvecScratch,
    out: &mut [f32],
    rows: usize,
    cols: usize,
    n_tok: usize,
) {
    assert!((1..=MAX_CPU_BATCH).contains(&n_tok));
    assert_eq!(x.len(), n_tok * cols);
    assert_eq!(out.len(), n_tok * rows);
    if n_tok == 1 {
        matvec_f32(pool, kt, dtype, w_bytes, x, scratch, out, rows, cols);
        return;
    }

    let n_thr = pool.num_threads();
    let chunk = rows.div_ceil(n_thr).next_multiple_of(4).max(4);
    let out_ptr = out.as_mut_ptr() as usize;

    match dtype {
        DType::Q4_0 => {
            let cols_blocks = cols / QK4_0;
            let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_0)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8_0] = xq;
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>()));
            // SAFETY: Alignment and length verified; `BlockQ4_0` is `#[repr(C)]` POD.
            let w: &[BlockQ4_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_0,
                    rows * cols_blocks,
                )
            };
            // Tile by 4 rows (<= 9 KB, stays in L1D cache) using ISA-dispatched `gemm_q4_0_q8_0`
            // (`smmla` 2x2 on `i8mm`, 4-token `sdot` on `dotprod`, NEON on `micro`).
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 4 * MAX_CPU_BATCH];
                while r + 4 <= r1 {
                    let w4 = &w[r * cols_blocks..(r + 4) * cols_blocks];
                    (kt.gemm_q4_0_q8_0)(w4, xq_ref, &mut tile[..4 * n_tok], 4, cols_blocks, n_tok);
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + 4)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                4,
                            )
                        };
                        dst.copy_from_slice(&tile[t * 4..(t + 1) * 4]);
                    }
                    r += 4;
                }
                if r < r1 {
                    let rem = r1 - r;
                    let wr = &w[r * cols_blocks..r1 * cols_blocks];
                    (kt.gemm_q4_0_q8_0)(
                        wr,
                        xq_ref,
                        &mut tile[..rem * n_tok],
                        rem,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r1)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                rem,
                            )
                        };
                        dst.copy_from_slice(&tile[t * rem..(t + 1) * rem]);
                    }
                }
            });
        }
        DType::Q4_1 => {
            let cols_blocks = cols / QK4_1;
            let xq = &mut scratch.q8_1[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_1)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8_1] = xq;
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_1>()));
            // SAFETY: Alignment and length verified; `BlockQ4_1` is `#[repr(C)]` POD.
            let w: &[BlockQ4_1] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_1,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                while r + 4 <= r1 {
                    let w4 = &w[r * cols_blocks..(r + 4) * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + 4)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                4,
                            )
                        };
                        (kt.gemv_q4_1_q8_1)(
                            w4,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            4,
                            cols_blocks,
                        );
                    }
                    r += 4;
                }
                if r < r1 {
                    let rem = r1 - r;
                    let wr = &w[r * cols_blocks..r1 * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r1)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                rem,
                            )
                        };
                        (kt.gemv_q4_1_q8_1)(
                            wr,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            rem,
                            cols_blocks,
                        );
                    }
                }
            });
        }
        DType::Q8_0 => {
            let cols_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_0)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8_0] = xq;
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ8_0>()));
            // SAFETY: Alignment and length verified; `BlockQ8_0` is `#[repr(C)]` POD.
            let w: &[BlockQ8_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ8_0,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                while r + 4 <= r1 {
                    let w4 = &w[r * cols_blocks..(r + 4) * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + 4)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                4,
                            )
                        };
                        (kt.gemv_q8_0_q8_0)(
                            w4,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            4,
                            cols_blocks,
                        );
                    }
                    r += 4;
                }
                if r < r1 {
                    let rem = r1 - r;
                    let wr = &w[r * cols_blocks..r1 * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r1)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                rem,
                            )
                        };
                        (kt.gemv_q8_0_q8_0)(
                            wr,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            rem,
                            cols_blocks,
                        );
                    }
                }
            });
        }
        DType::Q5K => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..n_tok * cols_blocks];
            for t in 0..n_tok {
                quantize_row_q8_k(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8K] = xq;
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ5K>()));
            // SAFETY: Alignment and length verified; `BlockQ5K` is `#[repr(C)]` POD.
            let w: &[BlockQ5K] = unsafe {
                core::slice::from_raw_parts(w_bytes.as_ptr() as *const BlockQ5K, rows * cols_blocks)
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                while r + 2 <= r1 {
                    let w2 = &w[r * cols_blocks..(r + 2) * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + 2)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                2,
                            )
                        };
                        (kt.gemv_q5_k_q8_k)(
                            w2,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            2,
                            cols_blocks,
                        );
                    }
                    r += 2;
                }
                if r < r1 {
                    let wr = &w[r * cols_blocks..r1 * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r]` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                1,
                            )
                        };
                        (kt.gemv_q5_k_q8_k)(
                            wr,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            1,
                            cols_blocks,
                        );
                    }
                }
            });
        }
        DType::Q6K => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..n_tok * cols_blocks];
            for t in 0..n_tok {
                quantize_row_q8_k(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8K] = xq;
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ6K>()));
            // SAFETY: Alignment and length verified; `BlockQ6K` is `#[repr(C)]` POD.
            let w: &[BlockQ6K] = unsafe {
                core::slice::from_raw_parts(w_bytes.as_ptr() as *const BlockQ6K, rows * cols_blocks)
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                while r + 2 <= r1 {
                    let w2 = &w[r * cols_blocks..(r + 2) * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + 2)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                2,
                            )
                        };
                        (kt.gemv_q6_k_q8_k)(
                            w2,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            2,
                            cols_blocks,
                        );
                    }
                    r += 2;
                }
                if r < r1 {
                    let wr = &w[r * cols_blocks..r1 * cols_blocks];
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r]` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                1,
                            )
                        };
                        (kt.gemv_q6_k_q8_k)(
                            wr,
                            &xq_ref[t * cols_blocks..(t + 1) * cols_blocks],
                            dst,
                            1,
                            cols_blocks,
                        );
                    }
                }
            });
        }
        _ => {
            for t in 0..n_tok {
                matvec_f32(
                    pool,
                    kt,
                    dtype,
                    w_bytes,
                    &x[t * cols..(t + 1) * cols],
                    scratch,
                    &mut out[t * rows..(t + 1) * rows],
                    rows,
                    cols,
                );
            }
        }
    }
}

/// In-place L2 normalization per head: `x[i] /= sqrt(sum(x^2) + eps)`.
/// Matches `build_gdn_l2_norm` in llama.cpp (`models.h`).
#[inline]
pub fn l2_norm_inplace(x: &mut [f32], eps: f32) {
    let mut ss = 0.0f64;
    for &v in x.iter() {
        let vf = v as f64;
        ss += vf * vf;
    }
    let scale = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Multi-head Gated DeltaNet recurrent step across `n_tok` tokens (`n_tok >= 1`),
/// parallelized across `n_v_heads` value heads on `pool`.
///
/// Follows `Strata/docs/GDN_CONTRACT.md` and `AGENTS.md` §9:
/// - Key head mapping: `kh = vh % n_k_heads` (tiled, NOT contiguous repeat).
/// - `state` is stored transposed per value head `[n_v_heads, head_dim, head_dim]`:
///   row `j` (`state_vh[j * head_dim .. (j + 1) * head_dim]`) holds column `j` of `S`.
/// - `out` is scaled by `1 / sqrt(head_dim)`.
/// - Multi-token (`n_tok > 1`) executes the exact same per-token operation order as `n_tok == 1`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence_parallel(
    pool: &ThreadPool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
    n_tok: usize,
) {
    let k_dim = n_k_heads * head_dim;
    let v_dim = n_v_heads * head_dim;
    let state_per_head = head_dim * head_dim;
    assert_eq!(q.len(), n_tok * k_dim);
    assert_eq!(k.len(), n_tok * k_dim);
    assert_eq!(v.len(), n_tok * v_dim);
    assert_eq!(g_exp.len(), n_tok * n_v_heads);
    assert_eq!(beta.len(), n_tok * n_v_heads);
    assert_eq!(state.len(), n_v_heads * state_per_head);
    assert_eq!(out.len(), n_tok * v_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let state_ptr = state.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = n_v_heads.div_ceil(pool.num_threads()).max(1);

    pool.parallel_for(n_v_heads, chunk, |vh0, vh1| {
        for vh in vh0..vh1 {
            let kh = vh % n_k_heads;
            // SAFETY: Each value head `vh` owns disjoint `[vh * state_per_head .. (vh + 1) * state_per_head]`.
            let s_vh = unsafe {
                core::slice::from_raw_parts_mut(
                    (state_ptr as *mut f32).add(vh * state_per_head),
                    state_per_head,
                )
            };

            for t in 0..n_tok {
                let q_h = &q[t * k_dim + kh * head_dim..t * k_dim + (kh + 1) * head_dim];
                let k_h = &k[t * k_dim + kh * head_dim..t * k_dim + (kh + 1) * head_dim];
                let v_h = &v[t * v_dim + vh * head_dim..t * v_dim + (vh + 1) * head_dim];
                let g = g_exp[t * n_v_heads + vh];
                let b = beta[t * n_v_heads + vh];

                // SAFETY: Disjoint `[t * v_dim + vh * head_dim .. + head_dim]` per `vh`.
                let o_h = unsafe {
                    core::slice::from_raw_parts_mut(
                        (out_ptr as *mut f32).add(t * v_dim + vh * head_dim),
                        head_dim,
                    )
                };

                for j in 0..head_dim {
                    let s_row = &mut s_vh[j * head_dim..(j + 1) * head_dim];
                    let mut sum_kv = 0.0f32;
                    for (s_val, &kv) in s_row.iter_mut().zip(k_h.iter()) {
                        let decayed = *s_val * g;
                        *s_val = decayed;
                        sum_kv += decayed * kv;
                    }

                    let delta_j = (v_h[j] - sum_kv) * b;

                    let mut sum_oq = 0.0f32;
                    for ((s_val, &kv), &qv) in s_row.iter_mut().zip(k_h.iter()).zip(q_h.iter()) {
                        let updated = *s_val + delta_j * kv;
                        *s_val = updated;
                        sum_oq += updated * qv;
                    }

                    o_h[j] = sum_oq * scale;
                }
            }
        }
    });
}
