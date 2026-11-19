//! Core CPU Transformer operators (RMSNorm, RoPE, SwiGLU, GQA attention,
//! and dtype-dispatched quantized matrix-vector multiplication).
//!
//! All operators are zero-allocation on the token generation path: callers pass
//! pre-allocated scratch buffers.

use tm_core::{
    BlockQ4_0, BlockQ4_1, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, DType, QK_K, QK4_0, QK4_1,
    QK8_0, f16_to_f32, quantize_row_q8_0, quantize_row_q8_1, quantize_row_q8_k,
};

use crate::{KernelTable, ThreadPool, generic};

/// Rotary Position Embedding pairing convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Interleaved adjacent pairs `(2i, 2i + 1)` — used by GGUF `llama` models.
    Norm,
    /// Split-half pairs `(i, i + head_dim / 2)` — used by GGUF `qwen2` / `qwen3`.
    Neox,
}

/// Pre-allocated activation quantization scratch buffers (reused across every matvec).
pub struct MatvecScratch {
    pub q8_0: Vec<BlockQ8_0>,
    pub q8_1: Vec<BlockQ8_1>,
    pub q8_k: Vec<BlockQ8K>,
}

impl MatvecScratch {
    /// Allocates scratch space large enough for vectors up to `max_cols` elements.
    pub fn new(max_cols: usize) -> Self {
        let nb32 = max_cols.div_ceil(QK8_0);
        let nb256 = max_cols.div_ceil(QK_K);
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

    let chunk = (rows / (pool.num_threads() * 4)).max(16);
    let out_ptr = out.as_mut_ptr() as usize;

    match dtype {
        DType::Q4_0 => {
            let cols_blocks = cols / QK4_0;
            let xq = &mut scratch.q8_0[..cols_blocks];
            quantize_row_q8_0(x, xq);
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
            quantize_row_q8_1(x, xq);
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
                for i in 0..len {
                    let row = &w[(r0 + i) * cols_blocks..(r0 + i + 1) * cols_blocks];
                    o[i] = (kt.vec_dot_q4_1_q8_1)(row, xq_ref);
                }
            });
        }
        DType::Q8_0 => {
            let cols_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..cols_blocks];
            quantize_row_q8_0(x, xq);
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
                for i in 0..len {
                    let row = &w[(r0 + i) * cols_blocks..(r0 + i + 1) * cols_blocks];
                    o[i] = (kt.vec_dot_q8_0_q8_0)(row, xq_ref);
                }
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
                for i in 0..len {
                    let row = &w[(r0 + i) * cols_blocks..(r0 + i + 1) * cols_blocks];
                    o[i] = (kt.vec_dot_q6_k_q8_k)(row, xq_ref);
                }
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
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * f16_to_f32(kh[d]);
            }
            *s_slot = dot * scale;
        }

        softmax_inplace(sc);

        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        for (t, &w) in sc.iter().enumerate() {
            let v_off = t * kv_dim + kv_h * head_dim;
            let vh = &v_cache[v_off..v_off + head_dim];
            for d in 0..head_dim {
                oh[d] += w * f16_to_f32(vh[d]);
            }
        }
    }
}
