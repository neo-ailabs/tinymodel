//! Core CPU Transformer operators (RMSNorm, RoPE, SwiGLU, GQA attention,
//! and dtype-dispatched quantized matrix-vector multiplication).
//!
//! All operators are zero-allocation on the token generation path: callers pass
//! pre-allocated scratch buffers.

use tm_core::{
    BlockIQ2XXS, BlockIQ3XXS, BlockIQ4NL, BlockNvFp4, BlockQ2_0, BlockQ4_0, BlockQ4_0x4, BlockQ4_1,
    BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, BlockTQ2_0, DType, QK_K, QK_NVFP4, QK2_0,
    QK4_0, QK4_1, QK4_NL, QK8_0, dequantize_q4_0x4_single_row, dequantize_row_bytes, f16_to_f32,
};

use crate::{KernelTable, ThreadPool, generic};

/// Maximum batch size per CPU prefill chunk (`64` tokens keeps L1/L2 tile buffers `<= 16 KiB` on stack
/// while halving DRAM weight reloads on `pp512`).
pub const MAX_CPU_BATCH: usize = 64;

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

/// Computes optimal row chunk size:
/// - For `n_thr <= 4` (homogeneous 1–4 core pools: M4 P-cores, Pi 5, Pi 4B, RK3588 4×A76, Pixel 8 4×A715),
///   uses 1 contiguous hardware-prefetched slice per thread.
/// - For `n_thr > 4` (heterogeneous big.LITTLE pools such as Pixel 8 `1×X3 + 4×A715` or 9-core unpinned),
///   uses `4×` finer chunks so faster cores dynamically steal work via lock-free `fetch_add`.
#[inline]
fn dynamic_row_chunk(total_rows: usize, n_thr: usize) -> usize {
    if n_thr <= 4 {
        total_rows.div_ceil(n_thr.max(1)).next_multiple_of(4).max(4)
    } else {
        total_rows.div_ceil(n_thr * 4).next_multiple_of(8).max(32)
    }
}

/// Computes `y = W * x` without quantizing `x` when `dtype == DType::Q8_0` (evaluates exact
/// `Q8_0 × F32` dequantized dot products via `kt.dot_f32_q8_0`, matching GPU float-activation
/// shaders across hybrid VRAM + Host RAM + NVMe MoE execution). Falls back to `matvec_f32`
/// for other `DType`s.
#[allow(clippy::too_many_arguments)]
pub fn matvec_f32_unquant_act(
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
    if dtype == DType::Q8_0 {
        assert_eq!(x.len(), cols);
        assert_eq!(out.len(), rows);
        assert_eq!(w_bytes.len(), rows * dtype.row_bytes(cols));
        let cols_blocks = cols / QK8_0;
        assert!(
            (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ8_0>()),
            "Q8_0 tensor pointer misaligned"
        );
        // SAFETY: Length and alignment of `w_bytes` verified above; `BlockQ8_0` is `#[repr(C)]` POD.
        let w: &[BlockQ8_0] = unsafe {
            core::slice::from_raw_parts(w_bytes.as_ptr() as *const BlockQ8_0, rows * cols_blocks)
        };
        let n_thr = pool.num_threads();
        if n_thr <= 1 {
            (kt.gemv_f32_q8_0)(w, x, out, rows, cols_blocks);
        } else {
            let chunk = dynamic_row_chunk(rows, n_thr);
            let out_ptr = out.as_mut_ptr() as usize;
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_f32_q8_0)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    x,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        return;
    }
    matvec_f32(pool, kt, dtype, w_bytes, x, scratch, out, rows, cols);
}

/// Multi-vector exact float-activation matrix-vector product `out_batch[m, r] = dot(W[r], x_batch[m])`
/// across `n_vec` vectors (`x_batch: [n_vec, cols]`, `out_batch: [n_vec, rows]`), streaming each
/// row of `W` **once** into L1D cache across all `n_vec` vectors while preserving `0-ULP`
/// bit-for-bit equivalence with `n_vec` calls to [`matvec_f32_unquant_act`].
#[allow(clippy::too_many_arguments)]
pub fn matvec_batch_f32_unquant_act(
    pool: &ThreadPool,
    kt: &KernelTable,
    dtype: DType,
    w_bytes: &[u8],
    x_batch: &[f32],
    n_vec: usize,
    scratch: &mut MatvecScratch,
    out_batch: &mut [f32],
    rows: usize,
    cols: usize,
) {
    if n_vec == 0 {
        return;
    }
    if n_vec == 1 {
        matvec_f32_unquant_act(
            pool,
            kt,
            dtype,
            w_bytes,
            &x_batch[..cols],
            scratch,
            &mut out_batch[..rows],
            rows,
            cols,
        );
        return;
    }
    if dtype == DType::Q8_0 {
        assert_eq!(x_batch.len(), n_vec * cols);
        assert_eq!(out_batch.len(), n_vec * rows);
        assert_eq!(w_bytes.len(), rows * dtype.row_bytes(cols));
        let cols_blocks = cols / QK8_0;
        assert!(
            (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ8_0>()),
            "Q8_0 tensor pointer misaligned"
        );
        // SAFETY: Length and alignment of `w_bytes` verified above; `BlockQ8_0` is `#[repr(C)]` POD.
        let w: &[BlockQ8_0] = unsafe {
            core::slice::from_raw_parts(w_bytes.as_ptr() as *const BlockQ8_0, rows * cols_blocks)
        };
        let n_thr = pool.num_threads();
        if n_thr <= 1 {
            (kt.gemm_f32_q8_0)(w, x_batch, out_batch, rows, cols_blocks, n_vec);
        } else {
            const TILE_TOK: usize = 64;
            let chunk = dynamic_row_chunk(rows, n_thr);
            let out_ptr = out_batch.as_mut_ptr() as usize;
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 8 * TILE_TOK];
                while r < r1 {
                    let b_rows = (r1 - r).min(8);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    let mut m0 = 0usize;
                    while m0 < n_vec {
                        let bm = (n_vec - m0).min(TILE_TOK);
                        (kt.gemm_f32_q8_0)(
                            wr,
                            &x_batch[m0 * cols..(m0 + bm) * cols],
                            &mut tile[..b_rows * bm],
                            b_rows,
                            cols_blocks,
                            bm,
                        );
                        for t in 0..bm {
                            // SAFETY: Disjoint `[(m0 + t) * rows + r .. + b_rows)` per worker.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (out_ptr as *mut f32).add((m0 + t) * rows + r),
                                    b_rows,
                                )
                            };
                            dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                        }
                        m0 += bm;
                    }
                    r += b_rows;
                }
            });
        }
        return;
    }
    if matches!(dtype, DType::BF16 | DType::NvFp4) {
        matmul_f32(
            pool, kt, dtype, w_bytes, x_batch, scratch, out_batch, rows, cols, n_vec,
        );
        return;
    }
    for m in 0..n_vec {
        matvec_f32_unquant_act(
            pool,
            kt,
            dtype,
            w_bytes,
            &x_batch[m * cols..(m + 1) * cols],
            scratch,
            &mut out_batch[m * rows..(m + 1) * rows],
            rows,
            cols,
        );
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

    let n_thr = pool.num_threads();
    let chunk = dynamic_row_chunk(rows, n_thr);
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
        DType::Q4_0x4 => {
            assert_eq!(rows % 4, 0, "Q4_0x4 requires rows divisible by 4");
            let cols_blocks = cols / QK4_0;
            let xq = &mut scratch.q8_0[..cols_blocks];
            (kt.quantize_q8_0)(x, xq);
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0x4>()),
                "Q4_0x4 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockQ4_0x4` is `#[repr(C)]` POD.
            let w: &[BlockQ4_0x4] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_0x4,
                    (rows / 4) * cols_blocks,
                )
            };
            let n_groups = rows / 4;
            let group_chunk = dynamic_row_chunk(rows, n_thr) / 4;
            pool.parallel_for(n_groups, group_chunk.max(1), |g0, g1| {
                let r0 = g0 * 4;
                let len = (g1 - g0) * 4;
                // SAFETY: Disjoint `[r0..r0 + len)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &w[g0 * cols_blocks..g1 * cols_blocks],
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
            (kt.quantize_q8_k)(x, xq);
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
            (kt.quantize_q8_k)(x, xq);
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
        DType::Q2_0 => {
            let cols_blocks = cols / QK2_0;
            let q8_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..q8_blocks];
            (kt.quantize_q8_0)(x, xq);
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ2_0>()),
                "Q2_0 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockQ2_0` is `#[repr(C)]` POD.
            let w: &[BlockQ2_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ2_0,
                    rows * cols_blocks,
                )
            };
            let chunk16 = chunk.next_multiple_of(16).max(16);
            pool.parallel_for(rows, chunk16, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q2_0_q8_0)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::TQ2_0 => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..cols_blocks];
            (kt.quantize_q8_k)(x, xq);
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockTQ2_0>()),
                "TQ2_0 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockTQ2_0` is `#[repr(C)]` POD.
            let w: &[BlockTQ2_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockTQ2_0,
                    rows * cols_blocks,
                )
            };
            let chunk16 = chunk.next_multiple_of(16).max(16);
            pool.parallel_for(rows, chunk16, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_tq2_0_q8_k)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Iq2Xxs => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..cols_blocks];
            (kt.quantize_q8_k)(x, xq);
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockIQ2XXS>()),
                "IQ2_XXS tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockIQ2XXS` is `#[repr(C)]` POD.
            let w: &[BlockIQ2XXS] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockIQ2XXS,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_iq2_xxs_q8_k)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Iq3Xxs => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..cols_blocks];
            (kt.quantize_q8_k)(x, xq);
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockIQ3XXS>()),
                "IQ3_XXS tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockIQ3XXS` is `#[repr(C)]` POD.
            let w: &[BlockIQ3XXS] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockIQ3XXS,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_iq3_xxs_q8_k)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::Iq4Nl => {
            let cols_blocks = cols / QK4_NL;
            let xq = &mut scratch.q8_0[..cols_blocks];
            (kt.quantize_q8_0)(x, xq);
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockIQ4NL>()),
                "IQ4_NL tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockIQ4NL` is `#[repr(C)]` POD.
            let w: &[BlockIQ4NL] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockIQ4NL,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_iq4_nl_q8_0)(
                    &w[r0 * cols_blocks..r1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            });
        }
        DType::NvFp4 => {
            let cols_blocks = cols / QK_NVFP4;
            let q8_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..q8_blocks];
            (kt.quantize_q8_0)(x, xq);
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockNvFp4>()),
                "NVFP4 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above; `BlockNvFp4` is `#[repr(C)]` POD (`align = 1`).
            let w: &[BlockNvFp4] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockNvFp4,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_nvfp4_q8_0)(
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
                    o[i] = (kt.dot_f32_f16)(x, row);
                }
            });
        }
        DType::BF16 => {
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<u16>()),
                "BF16 tensor pointer misaligned"
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
                    o[i] = (kt.dot_f32_bf16)(x, row);
                }
            });
        }
        DType::Q5_0 | DType::Q3K | DType::Q4K | DType::Iq4Xs => {
            let blk_elems = dtype.block_elements();
            let blk_bytes = dtype.block_bytes();
            let row_bytes = dtype.row_bytes(cols);
            let n_blks = cols / blk_elems;
            pool.parallel_for(rows, chunk, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` output ranges per parallel chunk.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                let mut deq_blk = [0.0f32; QK_K];
                for i in 0..len {
                    let rb = &w_bytes[(r0 + i) * row_bytes..(r0 + i + 1) * row_bytes];
                    let mut acc = 0.0f32;
                    for b in 0..n_blks {
                        dequantize_row_bytes(
                            dtype,
                            &rb[b * blk_bytes..(b + 1) * blk_bytes],
                            &mut deq_blk[..blk_elems],
                        );
                        acc += generic::vec_dot_f32(
                            &deq_blk[..blk_elems],
                            &x[b * blk_elems..(b + 1) * blk_elems],
                        );
                    }
                    o[i] = acc;
                }
            });
        }
        other => panic!("matvec_f32: unsupported weight dtype {other:?}"),
    }
}

/// RMS normalization: `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`.
#[inline]
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    (crate::registry::active_kernels().rms_norm_f32)(out, x, weight, eps);
}

/// Parallel multi-token RMS normalization across `n_tok` contiguous rows of width `dim`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_batch_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    eps: f32,
    dim: usize,
    n_tok: usize,
) {
    assert_eq!(out.len(), n_tok * dim);
    assert_eq!(x.len(), n_tok * dim);
    assert_eq!(weight.len(), dim);
    if pool.num_threads() == 1 || n_tok <= 2 {
        for t in 0..n_tok {
            (kt.rms_norm_f32)(
                &mut out[t * dim..(t + 1) * dim],
                &x[t * dim..(t + 1) * dim],
                weight,
                eps,
            );
        }
        return;
    }
    let out_ptr = out.as_mut_ptr() as usize;
    let chunk = dynamic_row_chunk(n_tok, pool.num_threads());
    pool.parallel_for(n_tok, chunk, |t0, t1| {
        for t in t0..t1 {
            // SAFETY: Each token `t` writes to disjoint `[t * dim .. (t + 1) * dim)`.
            let out_t =
                unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(t * dim), dim) };
            (kt.rms_norm_f32)(out_t, &x[t * dim..(t + 1) * dim], weight, eps);
        }
    });
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

/// Multi-threaded in-place SwiGLU activation (`gate[i] = silu(gate[i]) * up[i]`).
/// Purely elementwise so parallel execution is bit-for-bit identical to `swiglu_inplace`.
pub fn swiglu_inplace_parallel(pool: &ThreadPool, gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    assert_eq!(n, up.len());
    if pool.num_threads() == 1 || n < 2048 {
        swiglu_inplace(gate, up);
        return;
    }
    let gate_ptr = gate.as_mut_ptr() as usize;
    let blocks64 = n.div_ceil(64);
    let chunk_blocks = dynamic_row_chunk(blocks64, pool.num_threads());
    pool.parallel_for(blocks64, chunk_blocks, |b0, b1| {
        let s = b0 * 64;
        let e = (b1 * 64).min(n);
        if s < e {
            // SAFETY: Disjoint `[s..e)` 256-byte cache-line-aligned slices of `gate`.
            let g_slice =
                unsafe { core::slice::from_raw_parts_mut((gate_ptr as *mut f32).add(s), e - s) };
            swiglu_inplace(g_slice, &up[s..e]);
        }
    });
}

/// Exact ggml tanh-approximation GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))`.
#[inline]
pub fn gelu_tanh_f32(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEF_A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// In-place GeGLU activation (`gate[i] = gelu_tanh(gate[i]) * up[i]`), matching `LLM_FFN_GELU` in ggml.
pub fn geglu_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g = gelu_tanh_f32(*g) * u;
    }
}

/// Multi-threaded in-place GeGLU activation (`gate[i] = gelu_tanh(gate[i]) * up[i]`).
pub fn geglu_inplace_parallel(pool: &ThreadPool, gate: &mut [f32], up: &[f32]) {
    let n = gate.len();
    assert_eq!(n, up.len());
    if pool.num_threads() == 1 || n < 2048 {
        geglu_inplace(gate, up);
        return;
    }
    let gate_ptr = gate.as_mut_ptr() as usize;
    let blocks64 = n.div_ceil(64);
    let chunk_blocks = dynamic_row_chunk(blocks64, pool.num_threads());
    pool.parallel_for(blocks64, chunk_blocks, |b0, b1| {
        let s = b0 * 64;
        let e = (b1 * 64).min(n);
        if s < e {
            // SAFETY: Disjoint `[s..e)` slices of `gate`.
            let g_slice =
                unsafe { core::slice::from_raw_parts_mut((gate_ptr as *mut f32).add(s), e - s) };
            geglu_inplace(g_slice, &up[s..e]);
        }
    });
}

/// In-place unit-weight RMS normalization (`x[i] /= sqrt(mean(x^2) + eps)`), matching `ggml_rms_norm`.
#[inline]
pub fn rms_norm_unit_inplace(x: &mut [f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let n = x.len() as f32;
    (crate::registry::active_kernels().l2_norm_f32)(x, eps * n);
    let sqrt_n = n.sqrt();
    for v in x.iter_mut() {
        *v *= sqrt_n;
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
    let chunk = if pool.num_threads() <= 4 {
        n_heads.div_ceil(pool.num_threads()).max(1)
    } else {
        1
    };

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
    if dtype_a == DType::Q4_0x4 && dtype_b == DType::Q4_0x4 {
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..cols_blocks];
        (kt.quantize_q8_0)(x, xq);
        let xq_ref: &[BlockQ8_0] = xq;

        assert!(
            (wa_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
                && (wb_bytes.as_ptr() as usize)
                    .is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
        );
        let ga = rows_a / 4;
        let gb = rows_b / 4;
        // SAFETY: Lengths and alignment verified; `BlockQ4_0x4` is `#[repr(C)]` POD.
        let wa: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wa_bytes.as_ptr() as *const BlockQ4_0x4, ga * cols_blocks)
        };
        let wb: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0x4, gb * cols_blocks)
        };

        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let total_g = ga + gb;
        let g_chunk = (dynamic_row_chunk(rows_a + rows_b, pool.num_threads()) / 4).max(1);

        pool.parallel_for(total_g, g_chunk, |gs, ge| {
            if gs < ga {
                let ga1 = ge.min(ga);
                let r0 = gs * 4;
                let len = (ga1 - gs) * 4;
                // SAFETY: Disjoint `[r0..r0 + len)` range in `out_a`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((oa_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &wa[gs * cols_blocks..ga1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
            if ge > ga {
                let gb0 = gs.saturating_sub(ga);
                let gb1 = ge - ga;
                let r0 = gb0 * 4;
                let len = (gb1 - gb0) * 4;
                // SAFETY: Disjoint `[r0..r0 + len)` range in `out_b`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((ob_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &wb[gb0 * cols_blocks..gb1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
        });
        return;
    }

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
        let chunk = dynamic_row_chunk(total, pool.num_threads());

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
    if dtype_a == DType::Q4_0x4 && dtype_b == DType::Q4_0x4 && dtype_c == DType::Q4_0x4 {
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..cols_blocks];
        (kt.quantize_q8_0)(x, xq);
        let xq_ref: &[BlockQ8_0] = xq;

        assert!(
            (wa_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
                && (wb_bytes.as_ptr() as usize)
                    .is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
                && (wc_bytes.as_ptr() as usize)
                    .is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
        );
        let ga = rows_a / 4;
        let gb = rows_b / 4;
        let gc = rows_c / 4;
        // SAFETY: Lengths and alignment verified; `BlockQ4_0x4` is `#[repr(C)]` POD.
        let wa: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wa_bytes.as_ptr() as *const BlockQ4_0x4, ga * cols_blocks)
        };
        // SAFETY: Lengths and alignment verified; `BlockQ4_0x4` is `#[repr(C)]` POD.
        let wb: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0x4, gb * cols_blocks)
        };
        // SAFETY: Lengths and alignment verified; `BlockQ4_0x4` is `#[repr(C)]` POD.
        let wc: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wc_bytes.as_ptr() as *const BlockQ4_0x4, gc * cols_blocks)
        };

        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let oc_ptr = out_c.as_mut_ptr() as usize;
        let gab = ga + gb;
        let total_g = gab + gc;
        let g_chunk = (dynamic_row_chunk(rows_a + rows_b + rows_c, pool.num_threads()) / 4).max(1);

        pool.parallel_for(total_g, g_chunk, |gs, ge| {
            if gs < ga {
                let ga1 = ge.min(ga);
                let r0 = gs * 4;
                let len = (ga1 - gs) * 4;
                // SAFETY: Disjoint `[r0..r0 + len)` range in `out_a`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((oa_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &wa[gs * cols_blocks..ga1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
            if gs < gab && ge > ga {
                let gb0 = gs.saturating_sub(ga);
                let gb1 = ge.min(gab) - ga;
                let r0 = gb0 * 4;
                let len = (gb1 - gb0) * 4;
                // SAFETY: Disjoint `[r0..r0 + len)` range in `out_b`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((ob_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &wb[gb0 * cols_blocks..gb1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
            if ge > gab {
                let gc0 = gs.saturating_sub(gab);
                let gc1 = ge - gab;
                let r0 = gc0 * 4;
                let len = (gc1 - gc0) * 4;
                // SAFETY: Disjoint `[r0..r0 + len)` range in `out_c`.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((oc_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &wc[gc0 * cols_blocks..gc1 * cols_blocks],
                    xq_ref,
                    o,
                    len,
                    cols_blocks,
                );
            }
        });
        return;
    }

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
        // SAFETY: Lengths and alignment verified; `BlockQ4_0` is `#[repr(C)]` POD.
        let wb: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0, rows_b * cols_blocks)
        };
        // SAFETY: Lengths and alignment verified; `BlockQ4_0` is `#[repr(C)]` POD.
        let wc: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wc_bytes.as_ptr() as *const BlockQ4_0, rows_c * cols_blocks)
        };

        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let oc_ptr = out_c.as_mut_ptr() as usize;
        let ab = rows_a + rows_b;
        let total = ab + rows_c;
        let chunk = dynamic_row_chunk(total, pool.num_threads());

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
    let chunk = dynamic_row_chunk(rows, n_thr);
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
            // Tile by 64 rows (`64 * MAX_CPU_BATCH * 4 = 8 KiB` in L1D) so `i8mm` 8x8 and `dotprod` 4x4
            // reuse pre-packed activation tiles across 64 rows at a time!
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 64 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(64);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_q4_0_q8_0)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::Q4_0x4 => {
            assert_eq!(rows % 4, 0, "Q4_0x4 requires rows divisible by 4");
            let cols_blocks = cols / QK4_0;
            let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_0)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
            );
            let n_groups = rows / 4;
            // SAFETY: Alignment and length verified; `BlockQ4_0x4` is `#[repr(C)]` POD.
            let w: &[BlockQ4_0x4] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_0x4,
                    n_groups * cols_blocks,
                )
            };
            let g_chunk = (chunk / 4).max(1);
            pool.parallel_for(n_groups, g_chunk, |g0, g1| {
                let mut g = g0;
                let mut tile = [0.0f32; 64 * MAX_CPU_BATCH];
                while g < g1 {
                    let bg = (g1 - g).min(16); // up to 16 super-rows = 64 rows!
                    let b_rows = bg * 4;
                    let r = g * 4;
                    let wg = &w[g * cols_blocks..(g + bg) * cols_blocks];
                    (kt.gemm_q4_0x4_q8_0)(
                        wg,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    g += bg;
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
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_q8_0_q8_0)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::Q5K => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_k)(
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
                (kt.quantize_q8_k)(
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
                let mut tile = [0.0f32; 8 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(8);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_q6_k_q8_k)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. t * rows + r + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::Q2_0 => {
            let cols_blocks = cols / QK2_0;
            let q8_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..n_tok * q8_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_0)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * q8_blocks..(t + 1) * q8_blocks],
                );
            }
            let xq_ref: &[BlockQ8_0] = xq;
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ2_0>()));
            // SAFETY: Alignment and length verified; `BlockQ2_0` is `#[repr(C)]` POD.
            let w: &[BlockQ2_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ2_0,
                    rows * cols_blocks,
                )
            };
            let chunk16 = chunk.next_multiple_of(16).max(16);
            pool.parallel_for(rows, chunk16, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_q2_0_q8_0)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::TQ2_0 => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_k)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockTQ2_0>())
            );
            // SAFETY: Alignment and length verified; `BlockTQ2_0` is `#[repr(C)]` POD.
            let w: &[BlockTQ2_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockTQ2_0,
                    rows * cols_blocks,
                )
            };
            let chunk16 = chunk.next_multiple_of(16).max(16);
            pool.parallel_for(rows, chunk16, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_tq2_0_q8_k)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::Iq2Xxs => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_k)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockIQ2XXS>())
            );
            // SAFETY: Alignment and length verified; `BlockIQ2XXS` is `#[repr(C)]` POD.
            let w: &[BlockIQ2XXS] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockIQ2XXS,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_iq2_xxs_q8_k)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::Iq3Xxs => {
            let cols_blocks = cols / QK_K;
            let xq = &mut scratch.q8_k[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_k)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8K] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockIQ3XXS>())
            );
            // SAFETY: Alignment and length verified; `BlockIQ3XXS` is `#[repr(C)]` POD.
            let w: &[BlockIQ3XXS] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockIQ3XXS,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_iq3_xxs_q8_k)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::Iq4Nl => {
            let cols_blocks = cols / QK4_NL;
            let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_0)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
                );
            }
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockIQ4NL>())
            );
            // SAFETY: Alignment and length verified; `BlockIQ4NL` is `#[repr(C)]` POD.
            let w: &[BlockIQ4NL] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockIQ4NL,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_iq4_nl_q8_0)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::NvFp4 => {
            let cols_blocks = cols / QK_NVFP4;
            let q8_blocks = cols / QK8_0;
            let xq = &mut scratch.q8_0[..n_tok * q8_blocks];
            for t in 0..n_tok {
                (kt.quantize_q8_0)(
                    &x[t * cols..(t + 1) * cols],
                    &mut xq[t * q8_blocks..(t + 1) * q8_blocks],
                );
            }
            let xq_ref: &[BlockQ8_0] = xq;
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockNvFp4>())
            );
            // SAFETY: Alignment and length verified; `BlockNvFp4` is `#[repr(C)]` POD.
            let w: &[BlockNvFp4] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockNvFp4,
                    rows * cols_blocks,
                )
            };
            pool.parallel_for(rows, chunk, |r0, r1| {
                let mut r = r0;
                let mut tile = [0.0f32; 16 * MAX_CPU_BATCH];
                while r < r1 {
                    let b_rows = (r1 - r).min(16);
                    let wr = &w[r * cols_blocks..(r + b_rows) * cols_blocks];
                    (kt.gemm_nvfp4_q8_0)(
                        wr,
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows + r .. + b_rows)` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (out_ptr as *mut f32).add(t * rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            });
        }
        DType::BF16 => {
            assert!(
                (w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<u16>()),
                "BF16 tensor pointer misaligned"
            );
            // SAFETY: Length and alignment verified above.
            let w: &[u16] =
                unsafe { core::slice::from_raw_parts(w_bytes.as_ptr() as *const u16, rows * cols) };
            let out_ptr = out.as_mut_ptr() as usize;
            let chunk = dynamic_row_chunk(rows, pool.num_threads());
            pool.parallel_for(rows, chunk, |r0, r1| {
                for r in r0..r1 {
                    let wr = &w[r * cols..(r + 1) * cols];
                    for t in 0..n_tok {
                        let xt = &x[t * cols..(t + 1) * cols];
                        // SAFETY: Unique `(t, r)` element per worker `r`.
                        unsafe {
                            *(out_ptr as *mut f32).add(t * rows + r) = (kt.dot_f32_bf16)(xt, wr);
                        }
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

/// Batched multi-token causal GQA attention against F16 KV cache (`start_pos .. start_pos + n_tok`).
///
/// Assumes `k_cache` and `v_cache` are already populated up through `start_pos + n_tok - 1`.
/// Dispatches all `n_tok * n_heads` independent `(token, head)` pairs in ONE `ThreadPool` barrier,
/// preserving the exact per-token, per-head operation order of single-token `gqa_attention_f16_kv`
/// (`0` ULP difference, zero heap allocations).
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_batch_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    max_ctx: usize,
    scores: &mut [f32],
) {
    let q_dim = n_heads * head_dim;
    assert_eq!(out.len(), n_tok * q_dim);
    assert_eq!(q.len(), n_tok * q_dim);

    const STACK_SCORES_CAP: usize = 2048;
    if pool.num_threads() == 1 || n_tok == 0 || start_pos + n_tok > STACK_SCORES_CAP {
        for t in 0..n_tok {
            gqa_attention_f16_kv_parallel(
                pool,
                kt,
                &mut out[t * q_dim..(t + 1) * q_dim],
                &q[t * q_dim..(t + 1) * q_dim],
                k_cache,
                v_cache,
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos + t + 1,
                max_ctx,
                scores,
            );
        }
        return;
    }

    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out_ptr = out.as_mut_ptr() as usize;
    let total_tasks = n_tok * n_heads;
    let chunk = dynamic_row_chunk(total_tasks, pool.num_threads());

    pool.parallel_for(total_tasks, chunk, |idx0, idx1| {
        let mut sc_buf = [0.0f32; STACK_SCORES_CAP];
        for idx in idx0..idx1 {
            let t = idx / n_heads;
            let h = idx % n_heads;
            let seq_len = start_pos + t + 1;
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let sc = &mut sc_buf[..seq_len];

            for (step, s_slot) in sc.iter_mut().enumerate() {
                let k_off = step * kv_dim + kv_h * head_dim;
                let kh = &k_cache[k_off..k_off + head_dim];
                *s_slot = (kt.dot_f32_f16)(qh, kh) * scale;
            }

            softmax_inplace(sc);

            // SAFETY: Each `idx = t * n_heads + h` owns a disjoint `[head_dim]` slice of `out`.
            let oh = unsafe {
                core::slice::from_raw_parts_mut(
                    (out_ptr as *mut f32).add(t * q_dim + h * head_dim),
                    head_dim,
                )
            };
            oh.fill(0.0);
            for (step, &w) in sc.iter().enumerate() {
                let v_off = step * kv_dim + kv_h * head_dim;
                let vh = &v_cache[v_off..v_off + head_dim];
                (kt.axpy_f32_f16)(oh, w, vh);
            }
        }
    });
}

/// Extended single-token GQA attention supporting custom pre-softmax `attn_scale` (e.g. `1.0` on Gemma 4)
/// and optional standard sliding-window `window` (`pos - step < window`, i.e. `[pos + 1 - window ..= pos]`).
#[inline]
fn apply_attn_softcap(scores: &mut [f32], softcap: Option<f32>) {
    if let Some(cap) = softcap
        && cap > 0.0
    {
        let inv = 1.0 / cap;
        for s in scores.iter_mut() {
            *s = (*s * inv).tanh() * cap;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_ext_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    attn_scale: f32,
    window: Option<usize>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    gqa_attention_f16_kv_softcap_parallel(
        pool, kt, out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, pos, attn_scale, window,
        None, max_ctx, scores,
    );
}

/// Extended single-token GQA attention with custom `attn_scale`, `window`, and `attn_softcap`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_softcap_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    attn_scale: f32,
    window: Option<usize>,
    attn_softcap: Option<f32>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    let end_pos = pos + 1;
    let start_kv = match window {
        Some(w) if w > 0 => end_pos.saturating_sub(w),
        _ => 0,
    };
    let win_len = end_pos - start_kv;
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;

    if pool.num_threads() == 1 || win_len < 24 {
        let sc = &mut scores[..win_len];
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let k_off = t * kv_dim + kv_h * head_dim;
                sc[idx] = (kt.dot_f32_f16)(qh, &k_cache[k_off..k_off + head_dim]) * attn_scale;
            }
            apply_attn_softcap(sc, attn_softcap);
            softmax_inplace(sc);
            let oh = &mut out[h * head_dim..(h + 1) * head_dim];
            oh.fill(0.0);
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let v_off = t * kv_dim + kv_h * head_dim;
                (kt.axpy_f32_f16)(oh, sc[idx], &v_cache[v_off..v_off + head_dim]);
            }
        }
        return;
    }

    let out_ptr = out.as_mut_ptr() as usize;
    let scores_ptr = scores.as_mut_ptr() as usize;
    let chunk = if pool.num_threads() <= 4 {
        n_heads.div_ceil(pool.num_threads()).max(1)
    } else {
        1
    };

    pool.parallel_for(n_heads, chunk, |h0, h1| {
        for h in h0..h1 {
            let kv_h = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            // SAFETY: Disjoint `[h * max_ctx .. h * max_ctx + win_len)` per head `h`.
            let sc = unsafe {
                core::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(h * max_ctx), win_len)
            };
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let k_off = t * kv_dim + kv_h * head_dim;
                sc[idx] = (kt.dot_f32_f16)(qh, &k_cache[k_off..k_off + head_dim]) * attn_scale;
            }
            apply_attn_softcap(sc, attn_softcap);
            softmax_inplace(sc);
            // SAFETY: Disjoint `[h * head_dim .. (h + 1) * head_dim)` per head `h`.
            let oh = unsafe {
                core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(h * head_dim), head_dim)
            };
            oh.fill(0.0);
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let v_off = t * kv_dim + kv_h * head_dim;
                (kt.axpy_f32_f16)(oh, sc[idx], &v_cache[v_off..v_off + head_dim]);
            }
        }
    });
}

/// Extended batched multi-token GQA attention supporting custom `attn_scale` and optional `window`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_ext_batch_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    gqa_attention_f16_kv_softcap_batch_parallel(
        pool, kt, out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, start_pos, n_tok,
        attn_scale, window, None, max_ctx, scores,
    );
}

/// Extended batched multi-token GQA attention with custom `attn_scale`, `window`, and `attn_softcap`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_softcap_batch_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
    attn_softcap: Option<f32>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    let q_dim = n_heads * head_dim;
    assert_eq!(out.len(), n_tok * q_dim);
    assert_eq!(q.len(), n_tok * q_dim);

    const STACK_SCORES_CAP: usize = 2048;
    let max_win_needed = match window {
        Some(w) if w > 0 => w.min(start_pos + n_tok),
        _ => start_pos + n_tok,
    };
    if pool.num_threads() == 1 || n_tok == 0 || max_win_needed > STACK_SCORES_CAP {
        for t in 0..n_tok {
            gqa_attention_f16_kv_softcap_parallel(
                pool,
                kt,
                &mut out[t * q_dim..(t + 1) * q_dim],
                &q[t * q_dim..(t + 1) * q_dim],
                k_cache,
                v_cache,
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos + t,
                attn_scale,
                window,
                attn_softcap,
                max_ctx,
                scores,
            );
        }
        return;
    }

    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let out_ptr = out.as_mut_ptr() as usize;
    let total_tasks = n_tok * n_heads;
    let chunk = dynamic_row_chunk(total_tasks, pool.num_threads());

    pool.parallel_for(total_tasks, chunk, |idx0, idx1| {
        let mut sc_buf = [0.0f32; STACK_SCORES_CAP];
        for idx in idx0..idx1 {
            let t = idx / n_heads;
            let h = idx % n_heads;
            let end_pos = start_pos + t + 1;
            let start_kv = match window {
                Some(w) if w > 0 => end_pos.saturating_sub(w),
                _ => 0,
            };
            let win_len = end_pos - start_kv;
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let sc = &mut sc_buf[..win_len];

            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let k_off = step * kv_dim + kv_h * head_dim;
                sc[w_idx] = (kt.dot_f32_f16)(qh, &k_cache[k_off..k_off + head_dim]) * attn_scale;
            }
            apply_attn_softcap(sc, attn_softcap);
            softmax_inplace(sc);

            // SAFETY: Disjoint `[head_dim]` slice of `out` per `(t, h)`.
            let oh = unsafe {
                core::slice::from_raw_parts_mut(
                    (out_ptr as *mut f32).add(t * q_dim + h * head_dim),
                    head_dim,
                )
            };
            oh.fill(0.0);
            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let v_off = step * kv_dim + kv_h * head_dim;
                (kt.axpy_f32_f16)(oh, sc[w_idx], &v_cache[v_off..v_off + head_dim]);
            }
        }
    });
}

/// In-place L2 normalization per head: `x[i] /= sqrt(sum(x^2) + eps)`.
/// Matches `build_gdn_l2_norm` in llama.cpp (`models.h`).
#[inline]
pub fn l2_norm_inplace(x: &mut [f32], eps: f32) {
    (crate::registry::active_kernels().l2_norm_f32)(x, eps);
}

/// Parallelized Qwen3.5 GDN depthwise causal `conv1d + SiLU` + `q/k/v` split + `q/k` per-head `L2` norm.
///
/// Parallelizes across all `conv_dim = 2 * k_gdn_dim + v_gdn_dim` independent channels (`6144` on
/// Qwen3.5-0.8B/2B) and then across all `n_tok * 2 * hk` independent `q`/`k` head slices,
/// preserving the exact per-channel state update and per-token arithmetic order of `n_tok == 1`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv1d_silu_split_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    qkv_buf: &[f32],
    conv_hist: &mut [f32],
    conv1d: &[f32],
    q_buf: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    d_conv: usize,
    hk: usize,
    s_dim: usize,
    v_gdn_dim: usize,
    rms_eps: f32,
    n_tok: usize,
) {
    let k_gdn_dim = hk * s_dim;
    let conv_dim = 2 * k_gdn_dim + v_gdn_dim;
    let hist_len = d_conv - 1;
    assert_eq!(qkv_buf.len(), n_tok * conv_dim);
    assert_eq!(conv_hist.len(), conv_dim * hist_len);
    assert_eq!(conv1d.len(), conv_dim * d_conv);
    assert_eq!(q_buf.len(), n_tok * k_gdn_dim);
    assert_eq!(k_buf.len(), n_tok * k_gdn_dim);
    assert_eq!(v_buf.len(), n_tok * v_gdn_dim);

    // 1. Parallelize depthwise causal conv1d + SiLU across independent channels `[0..conv_dim)`
    let hist_ptr = conv_hist.as_mut_ptr() as usize;
    let q_ptr = q_buf.as_mut_ptr() as usize;
    let k_ptr = k_buf.as_mut_ptr() as usize;
    let v_ptr = v_buf.as_mut_ptr() as usize;
    let ch_blocks = conv_dim.div_ceil(32);
    let ch_chunk = dynamic_row_chunk(ch_blocks, pool.num_threads());

    pool.parallel_for(ch_blocks, ch_chunk, |b0, b1| {
        let ch_start = b0 * 32;
        let ch_end = (b1 * 32).min(conv_dim);
        for ch in ch_start..ch_end {
            // SAFETY: Each channel `ch` owns disjoint `[ch * hist_len .. (ch + 1) * hist_len]`.
            let h_slice = unsafe {
                core::slice::from_raw_parts_mut((hist_ptr as *mut f32).add(ch * hist_len), hist_len)
            };
            let w_slice = &conv1d[ch * d_conv..(ch + 1) * d_conv];

            for t in 0..n_tok {
                let cur_val = qkv_buf[t * conv_dim + ch];
                let mut sum = 0.0f32;
                for tap in 0..hist_len {
                    sum += h_slice[tap] * w_slice[tap];
                }
                sum += cur_val * w_slice[hist_len];
                for tap in 0..hist_len.saturating_sub(1) {
                    h_slice[tap] = h_slice[tap + 1];
                }
                if hist_len > 0 {
                    h_slice[hist_len - 1] = cur_val;
                }
                let activated = sum / (1.0 + (-sum).exp());
                // SAFETY: Each `(t, ch)` maps to a unique index in `q_buf`, `k_buf`, or `v_buf`.
                unsafe {
                    if ch < k_gdn_dim {
                        *(q_ptr as *mut f32).add(t * k_gdn_dim + ch) = activated;
                    } else if ch < 2 * k_gdn_dim {
                        *(k_ptr as *mut f32).add(t * k_gdn_dim + (ch - k_gdn_dim)) = activated;
                    } else {
                        *(v_ptr as *mut f32).add(t * v_gdn_dim + (ch - 2 * k_gdn_dim)) = activated;
                    }
                }
            }
        }
    });

    // 2. Per-head L2 norm on `q_buf` and `k_buf` (`2 * n_tok * hk` independent `[s_dim]` slices)
    let total_qk_heads = 2 * n_tok * hk;
    if pool.num_threads() == 1 || n_tok == 1 {
        for t in 0..n_tok {
            for kh in 0..hk {
                let off = t * k_gdn_dim + kh * s_dim;
                (kt.l2_norm_f32)(&mut q_buf[off..off + s_dim], rms_eps);
                (kt.l2_norm_f32)(&mut k_buf[off..off + s_dim], rms_eps);
            }
        }
    } else {
        let qk_chunk = dynamic_row_chunk(total_qk_heads, pool.num_threads());
        pool.parallel_for(total_qk_heads, qk_chunk, |idx0, idx1| {
            let n_q_heads = n_tok * hk;
            for idx in idx0..idx1 {
                if idx < n_q_heads {
                    // SAFETY: Disjoint `[idx * s_dim .. (idx + 1) * s_dim)` slice in `q_buf`.
                    let slice = unsafe {
                        core::slice::from_raw_parts_mut((q_ptr as *mut f32).add(idx * s_dim), s_dim)
                    };
                    (kt.l2_norm_f32)(slice, rms_eps);
                } else {
                    let k_idx = idx - n_q_heads;
                    // SAFETY: Disjoint `[k_idx * s_dim .. (k_idx + 1) * s_dim)` slice in `k_buf`.
                    let slice = unsafe {
                        core::slice::from_raw_parts_mut(
                            (k_ptr as *mut f32).add(k_idx * s_dim),
                            s_dim,
                        )
                    };
                    (kt.l2_norm_f32)(slice, rms_eps);
                }
            }
        });
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
    let kt = crate::registry::active_kernels();
    gdn_recurrence_with_gate_inner(
        pool, kt, q, k, v, g_exp, beta, state, out, None, n_k_heads, n_v_heads, head_dim, n_tok,
    );
}

/// Multi-head Gated DeltaNet recurrent step + fused L1-resident per-head gated `RMSNorm * SiLU(z)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence_gated_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    gate_z: &[f32],
    ssm_norm: &[f32],
    rms_eps: f32,
    n_k_heads: usize,
    n_v_heads: usize,
    head_dim: usize,
    n_tok: usize,
) {
    assert_eq!(gate_z.len(), n_tok * n_v_heads * head_dim);
    assert_eq!(ssm_norm.len(), head_dim);
    gdn_recurrence_with_gate_inner(
        pool,
        kt,
        q,
        k,
        v,
        g_exp,
        beta,
        state,
        out,
        Some((gate_z, ssm_norm, rms_eps)),
        n_k_heads,
        n_v_heads,
        head_dim,
        n_tok,
    );
}

#[allow(clippy::too_many_arguments)]
fn gdn_recurrence_with_gate_inner(
    pool: &ThreadPool,
    kt: &KernelTable,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g_exp: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    gate_norm: Option<(&[f32], &[f32], f32)>,
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
    let chunk = if pool.num_threads() <= 4 {
        n_v_heads.div_ceil(pool.num_threads()).max(1)
    } else {
        1
    };

    pool.parallel_for(n_v_heads, chunk, |vh0, vh1| {
        let mut norm_tmp = [0.0f32; 256];
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

                (kt.gdn_recurrence_head)(q_h, k_h, v_h, g, b, scale, s_vh, o_h, head_dim);

                if let Some((gate_z, ssm_norm, rms_eps)) = gate_norm {
                    let z_off = t * v_dim + vh * head_dim;
                    let z_h = &gate_z[z_off..z_off + head_dim];
                    if head_dim <= norm_tmp.len() {
                        (kt.rms_norm_f32)(&mut norm_tmp[..head_dim], o_h, ssm_norm, rms_eps);
                        for i in 0..head_dim {
                            let z = z_h[i];
                            o_h[i] = norm_tmp[i] * (z / (1.0 + (-z).exp()));
                        }
                    } else {
                        let mut ss = 0.0f64;
                        for &ov in o_h.iter() {
                            let vf = ov as f64;
                            ss += vf * vf;
                        }
                        let n_scale =
                            (1.0 / ((ss / head_dim as f64) + rms_eps as f64).sqrt()) as f32;
                        for i in 0..head_dim {
                            let z = z_h[i];
                            o_h[i] = ((o_h[i] * n_scale) * ssm_norm[i]) * (z / (1.0 + (-z).exp()));
                        }
                    }
                }
            }
        }
    });
}

/// Fused 2-matrix multi-token GEMM (`out_a`, `out_b`) sharing the exact same batch input `x[n_tok, cols]`.
/// Quantizes `x` ONCE and executes both projections inside a single `ThreadPool` barrier when both are `Q4_0x4` or `Q4_0`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_pair_f32(
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
    n_tok: usize,
) {
    if n_tok == 1 {
        matvec_pair_f32(
            pool, kt, dtype_a, wa_bytes, rows_a, out_a, dtype_b, wb_bytes, rows_b, out_b, x,
            scratch, cols,
        );
        return;
    }
    if dtype_a == DType::Q4_0x4 && dtype_b == DType::Q4_0x4 {
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
        for t in 0..n_tok {
            (kt.quantize_q8_0)(
                &x[t * cols..(t + 1) * cols],
                &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
            );
        }
        let xq_ref: &[BlockQ8_0] = xq;
        let ga = rows_a / 4;
        let gb = rows_b / 4;
        assert!(
            (wa_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
                && (wb_bytes.as_ptr() as usize)
                    .is_multiple_of(core::mem::align_of::<BlockQ4_0x4>())
        );
        // SAFETY: Alignment & bounds checked.
        let wa: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wa_bytes.as_ptr() as *const BlockQ4_0x4, ga * cols_blocks)
        };
        let wb: &[BlockQ4_0x4] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0x4, gb * cols_blocks)
        };
        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let total_g = ga + gb;
        let g_chunk = (dynamic_row_chunk(rows_a + rows_b, pool.num_threads()) / 4).max(1);

        pool.parallel_for(total_g, g_chunk, |gs, ge| {
            let mut tile = [0.0f32; 64 * MAX_CPU_BATCH];
            if gs < ga {
                let mut g = gs;
                let ga1 = ge.min(ga);
                while g < ga1 {
                    let bg = (ga1 - g).min(16);
                    let b_rows = bg * 4;
                    let r = g * 4;
                    (kt.gemm_q4_0x4_q8_0)(
                        &wa[g * cols_blocks..(g + bg) * cols_blocks],
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows_a + r .. + b_rows]` in `out_a`.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (oa_ptr as *mut f32).add(t * rows_a + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    g += bg;
                }
            }
            if ge > ga {
                let mut g = gs.saturating_sub(ga);
                let gb1 = ge - ga;
                while g < gb1 {
                    let bg = (gb1 - g).min(16);
                    let b_rows = bg * 4;
                    let r = g * 4;
                    (kt.gemm_q4_0x4_q8_0)(
                        &wb[g * cols_blocks..(g + bg) * cols_blocks],
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows_b + r .. + b_rows]` in `out_b`.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (ob_ptr as *mut f32).add(t * rows_b + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    g += bg;
                }
            }
        });
        return;
    }

    if dtype_a == DType::Q4_0 && dtype_b == DType::Q4_0 {
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
        for t in 0..n_tok {
            (kt.quantize_q8_0)(
                &x[t * cols..(t + 1) * cols],
                &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
            );
        }
        let xq_ref: &[BlockQ8_0] = xq;
        assert!(
            (wa_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
                && (wb_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>())
        );
        // SAFETY: Alignment & bounds checked.
        let wa: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wa_bytes.as_ptr() as *const BlockQ4_0, rows_a * cols_blocks)
        };
        let wb: &[BlockQ4_0] = unsafe {
            core::slice::from_raw_parts(wb_bytes.as_ptr() as *const BlockQ4_0, rows_b * cols_blocks)
        };
        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let total = rows_a + rows_b;
        let chunk = dynamic_row_chunk(total, pool.num_threads());

        pool.parallel_for(total, chunk, |s, e| {
            let mut tile = [0.0f32; 64 * MAX_CPU_BATCH];
            if s < rows_a {
                let mut r = s;
                let a1 = e.min(rows_a);
                while r < a1 {
                    let b_rows = (a1 - r).min(64);
                    (kt.gemm_q4_0_q8_0)(
                        &wa[r * cols_blocks..(r + b_rows) * cols_blocks],
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows_a + r .. + b_rows]` in `out_a`.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (oa_ptr as *mut f32).add(t * rows_a + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            }
            if e > rows_a {
                let mut r = s.saturating_sub(rows_a);
                let b1 = e - rows_a;
                while r < b1 {
                    let b_rows = (b1 - r).min(64);
                    (kt.gemm_q4_0_q8_0)(
                        &wb[r * cols_blocks..(r + b_rows) * cols_blocks],
                        xq_ref,
                        &mut tile[..b_rows * n_tok],
                        b_rows,
                        cols_blocks,
                        n_tok,
                    );
                    for t in 0..n_tok {
                        // SAFETY: Disjoint `[t * rows_b + r .. + b_rows]` in `out_b`.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (ob_ptr as *mut f32).add(t * rows_b + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    r += b_rows;
                }
            }
        });
        return;
    }

    matmul_f32(
        pool, kt, dtype_a, wa_bytes, x, scratch, out_a, rows_a, cols, n_tok,
    );
    matmul_f32(
        pool, kt, dtype_b, wb_bytes, x, scratch, out_b, rows_b, cols, n_tok,
    );
}

/// Fused 3-matrix multi-token GEMM (`out_a`, `out_b`, `out_c`) sharing the exact same batch input `x[n_tok, cols]`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_triple_f32(
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
    n_tok: usize,
) {
    if n_tok == 1 {
        matvec_triple_f32(
            pool, kt, dtype_a, wa_bytes, rows_a, out_a, dtype_b, wb_bytes, rows_b, out_b, dtype_c,
            wc_bytes, rows_c, out_c, x, scratch, cols,
        );
        return;
    }
    if (dtype_a == DType::Q4_0x4 && dtype_b == DType::Q4_0x4 && dtype_c == DType::Q4_0x4)
        || (dtype_a == DType::Q4_0 && dtype_b == DType::Q4_0 && dtype_c == DType::Q4_0)
    {
        // Fuse `(wa, wb)` in 1 barrier + `wc` or all 3 in 1 barrier:
        let cols_blocks = cols / QK4_0;
        let xq = &mut scratch.q8_0[..n_tok * cols_blocks];
        for t in 0..n_tok {
            (kt.quantize_q8_0)(
                &x[t * cols..(t + 1) * cols],
                &mut xq[t * cols_blocks..(t + 1) * cols_blocks],
            );
        }
        let xq_ref: &[BlockQ8_0] = xq;
        let oa_ptr = out_a.as_mut_ptr() as usize;
        let ob_ptr = out_b.as_mut_ptr() as usize;
        let oc_ptr = out_c.as_mut_ptr() as usize;

        if dtype_a == DType::Q4_0x4 {
            let ga = rows_a / 4;
            let gb = rows_b / 4;
            let gc = rows_c / 4;
            // SAFETY: Alignment & bounds checked.
            let wa: &[BlockQ4_0x4] = unsafe {
                core::slice::from_raw_parts(
                    wa_bytes.as_ptr() as *const BlockQ4_0x4,
                    ga * cols_blocks,
                )
            };
            // SAFETY: Alignment & bounds checked.
            let wb: &[BlockQ4_0x4] = unsafe {
                core::slice::from_raw_parts(
                    wb_bytes.as_ptr() as *const BlockQ4_0x4,
                    gb * cols_blocks,
                )
            };
            // SAFETY: Alignment & bounds checked.
            let wc: &[BlockQ4_0x4] = unsafe {
                core::slice::from_raw_parts(
                    wc_bytes.as_ptr() as *const BlockQ4_0x4,
                    gc * cols_blocks,
                )
            };
            let gab = ga + gb;
            let total_g = gab + gc;
            let g_chunk =
                (dynamic_row_chunk(rows_a + rows_b + rows_c, pool.num_threads()) / 4).max(1);
            pool.parallel_for(total_g, g_chunk, |gs, ge| {
                let mut tile = [0.0f32; 64 * MAX_CPU_BATCH];
                let run_seg = |w_mat: &[BlockQ4_0x4],
                               g0: usize,
                               g1: usize,
                               mat_rows: usize,
                               out_p: usize,
                               t_buf: &mut [f32]| {
                    let mut g = g0;
                    while g < g1 {
                        let bg = (g1 - g).min(16);
                        let b_rows = bg * 4;
                        let r = g * 4;
                        (kt.gemm_q4_0x4_q8_0)(
                            &w_mat[g * cols_blocks..(g + bg) * cols_blocks],
                            xq_ref,
                            &mut t_buf[..b_rows * n_tok],
                            b_rows,
                            cols_blocks,
                            n_tok,
                        );
                        for t in 0..n_tok {
                            // SAFETY: Disjoint `[t * mat_rows + r .. + b_rows]` per worker.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (out_p as *mut f32).add(t * mat_rows + r),
                                    b_rows,
                                )
                            };
                            dst.copy_from_slice(&t_buf[t * b_rows..(t + 1) * b_rows]);
                        }
                        g += bg;
                    }
                };
                if gs < ga {
                    run_seg(wa, gs, ge.min(ga), rows_a, oa_ptr, &mut tile);
                }
                if gs < gab && ge > ga {
                    run_seg(
                        wb,
                        gs.saturating_sub(ga),
                        ge.min(gab) - ga,
                        rows_b,
                        ob_ptr,
                        &mut tile,
                    );
                }
                if ge > gab {
                    run_seg(
                        wc,
                        gs.saturating_sub(gab),
                        ge - gab,
                        rows_c,
                        oc_ptr,
                        &mut tile,
                    );
                }
            });
            return;
        } else {
            // Standard `Q4_0` fused 3-matrix GEMM
            // SAFETY: Alignment & bounds checked.
            let wa: &[BlockQ4_0] = unsafe {
                core::slice::from_raw_parts(
                    wa_bytes.as_ptr() as *const BlockQ4_0,
                    rows_a * cols_blocks,
                )
            };
            // SAFETY: Alignment & bounds checked.
            let wb: &[BlockQ4_0] = unsafe {
                core::slice::from_raw_parts(
                    wb_bytes.as_ptr() as *const BlockQ4_0,
                    rows_b * cols_blocks,
                )
            };
            // SAFETY: Alignment & bounds checked.
            let wc: &[BlockQ4_0] = unsafe {
                core::slice::from_raw_parts(
                    wc_bytes.as_ptr() as *const BlockQ4_0,
                    rows_c * cols_blocks,
                )
            };
            let ab = rows_a + rows_b;
            let total = ab + rows_c;
            let chunk = dynamic_row_chunk(total, pool.num_threads());
            pool.parallel_for(total, chunk, |s, e| {
                let mut tile = [0.0f32; 64 * MAX_CPU_BATCH];
                let run_seg = |w_mat: &[BlockQ4_0],
                               r0: usize,
                               r1: usize,
                               mat_rows: usize,
                               out_p: usize,
                               t_buf: &mut [f32]| {
                    let mut r = r0;
                    while r < r1 {
                        let b_rows = (r1 - r).min(64);
                        (kt.gemm_q4_0_q8_0)(
                            &w_mat[r * cols_blocks..(r + b_rows) * cols_blocks],
                            xq_ref,
                            &mut t_buf[..b_rows * n_tok],
                            b_rows,
                            cols_blocks,
                            n_tok,
                        );
                        for t in 0..n_tok {
                            // SAFETY: Disjoint `[t * mat_rows + r .. + b_rows]` per worker.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (out_p as *mut f32).add(t * mat_rows + r),
                                    b_rows,
                                )
                            };
                            dst.copy_from_slice(&t_buf[t * b_rows..(t + 1) * b_rows]);
                        }
                        r += b_rows;
                    }
                };
                if s < rows_a {
                    run_seg(wa, s, e.min(rows_a), rows_a, oa_ptr, &mut tile);
                }
                if s < ab && e > rows_a {
                    run_seg(
                        wb,
                        s.saturating_sub(rows_a),
                        e.min(ab) - rows_a,
                        rows_b,
                        ob_ptr,
                        &mut tile,
                    );
                }
                if e > ab {
                    run_seg(wc, s.saturating_sub(ab), e - ab, rows_c, oc_ptr, &mut tile);
                }
            });
            return;
        }
    }

    matmul_f32(
        pool, kt, dtype_a, wa_bytes, x, scratch, out_a, rows_a, cols, n_tok,
    );
    matmul_f32(
        pool, kt, dtype_b, wb_bytes, x, scratch, out_b, rows_b, cols, n_tok,
    );
    matmul_f32(
        pool, kt, dtype_c, wc_bytes, x, scratch, out_c, rows_c, cols, n_tok,
    );
}

/// Extended single-token GQA attention against `BlockQ8_0` quantized KV cache (`F-14`).
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q8_0_kv_ext_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[BlockQ8_0],
    v_cache: &[BlockQ8_0],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    attn_scale: f32,
    window: Option<usize>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    gqa_attention_q8_0_kv_softcap_parallel(
        pool, kt, out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, pos, attn_scale, window,
        None, max_ctx, scores,
    );
}

/// Extended single-token GQA attention against `BlockQ8_0` KV cache with optional `attn_softcap`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q8_0_kv_softcap_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[BlockQ8_0],
    v_cache: &[BlockQ8_0],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    attn_scale: f32,
    window: Option<usize>,
    attn_softcap: Option<f32>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    let end_pos = pos + 1;
    let start_kv = match window {
        Some(w) if w > 0 => end_pos.saturating_sub(w),
        _ => 0,
    };
    let win_len = end_pos - start_kv;
    let hb = head_dim / QK8_0;
    let kv_blocks = n_kv_heads * hb;
    let group = n_heads / n_kv_heads;

    if pool.num_threads() == 1 || win_len < 24 {
        let sc = &mut scores[..win_len];
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let k_off = t * kv_blocks + kv_h * hb;
                sc[idx] = (kt.dot_f32_q8_0)(qh, &k_cache[k_off..k_off + hb]) * attn_scale;
            }
            apply_attn_softcap(sc, attn_softcap);
            softmax_inplace(sc);
            let oh = &mut out[h * head_dim..(h + 1) * head_dim];
            oh.fill(0.0);
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let v_off = t * kv_blocks + kv_h * hb;
                (kt.axpy_f32_q8_0)(oh, sc[idx], &v_cache[v_off..v_off + hb]);
            }
        }
        return;
    }

    let out_ptr = out.as_mut_ptr() as usize;
    let scores_ptr = scores.as_mut_ptr() as usize;
    let chunk = if pool.num_threads() <= 4 {
        n_heads.div_ceil(pool.num_threads()).max(1)
    } else {
        1
    };

    pool.parallel_for(n_heads, chunk, |h0, h1| {
        for h in h0..h1 {
            let kv_h = h / group;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            // SAFETY: Disjoint `[h * max_ctx .. h * max_ctx + win_len)` per head `h`.
            let sc = unsafe {
                core::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(h * max_ctx), win_len)
            };
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let k_off = t * kv_blocks + kv_h * hb;
                sc[idx] = (kt.dot_f32_q8_0)(qh, &k_cache[k_off..k_off + hb]) * attn_scale;
            }
            apply_attn_softcap(sc, attn_softcap);
            softmax_inplace(sc);
            // SAFETY: Disjoint `[h * head_dim .. (h + 1) * head_dim)` per head `h`.
            let oh = unsafe {
                core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(h * head_dim), head_dim)
            };
            oh.fill(0.0);
            for (idx, t) in (start_kv..end_pos).enumerate() {
                let v_off = t * kv_blocks + kv_h * hb;
                (kt.axpy_f32_q8_0)(oh, sc[idx], &v_cache[v_off..v_off + hb]);
            }
        }
    });
}

/// Extended batched multi-token GQA attention against `BlockQ8_0` quantized KV cache (`F-14`).
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q8_0_kv_ext_batch_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[BlockQ8_0],
    v_cache: &[BlockQ8_0],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    gqa_attention_q8_0_kv_softcap_batch_parallel(
        pool, kt, out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, start_pos, n_tok,
        attn_scale, window, None, max_ctx, scores,
    );
}

/// Extended batched multi-token GQA attention against `BlockQ8_0` KV cache with optional `attn_softcap`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_q8_0_kv_softcap_batch_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    out: &mut [f32],
    q: &[f32],
    k_cache: &[BlockQ8_0],
    v_cache: &[BlockQ8_0],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
    attn_softcap: Option<f32>,
    max_ctx: usize,
    scores: &mut [f32],
) {
    let q_dim = n_heads * head_dim;
    assert_eq!(out.len(), n_tok * q_dim);
    assert_eq!(q.len(), n_tok * q_dim);

    const STACK_SCORES_CAP: usize = 2048;
    let max_win_needed = match window {
        Some(w) if w > 0 => w.min(start_pos + n_tok),
        _ => start_pos + n_tok,
    };
    if pool.num_threads() == 1 || n_tok == 0 || max_win_needed > STACK_SCORES_CAP {
        for t in 0..n_tok {
            gqa_attention_q8_0_kv_softcap_parallel(
                pool,
                kt,
                &mut out[t * q_dim..(t + 1) * q_dim],
                &q[t * q_dim..(t + 1) * q_dim],
                k_cache,
                v_cache,
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos + t,
                attn_scale,
                window,
                attn_softcap,
                max_ctx,
                scores,
            );
        }
        return;
    }

    let hb = head_dim / QK8_0;
    let kv_blocks = n_kv_heads * hb;
    let group = n_heads / n_kv_heads;
    let out_ptr = out.as_mut_ptr() as usize;
    let total_tasks = n_tok * n_heads;
    let chunk = dynamic_row_chunk(total_tasks, pool.num_threads());

    pool.parallel_for(total_tasks, chunk, |idx0, idx1| {
        let mut sc_buf = [0.0f32; STACK_SCORES_CAP];
        for idx in idx0..idx1 {
            let t = idx / n_heads;
            let h = idx % n_heads;
            let end_pos = start_pos + t + 1;
            let start_kv = match window {
                Some(w) if w > 0 => end_pos.saturating_sub(w),
                _ => 0,
            };
            let win_len = end_pos - start_kv;
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let sc = &mut sc_buf[..win_len];

            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let k_off = step * kv_blocks + kv_h * hb;
                sc[w_idx] = (kt.dot_f32_q8_0)(qh, &k_cache[k_off..k_off + hb]) * attn_scale;
            }
            apply_attn_softcap(sc, attn_softcap);
            softmax_inplace(sc);

            // SAFETY: Disjoint `[head_dim]` slice of `out` per `(t, h)`.
            let oh = unsafe {
                core::slice::from_raw_parts_mut(
                    (out_ptr as *mut f32).add(t * q_dim + h * head_dim),
                    head_dim,
                )
            };
            oh.fill(0.0);
            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let v_off = step * kv_blocks + kv_h * hb;
                (kt.axpy_f32_q8_0)(oh, sc[w_idx], &v_cache[v_off..v_off + hb]);
            }
        }
    });
}

/// Parallelized in-place tanh logit soft-capping (`x = tanh(x / cap) * cap`) across `pool`.
/// Avoids single-threaded 262k-element `tanhf` bottleneck on large-vocab models (e.g. Gemma 4).
pub fn softcap_inplace_parallel(pool: &ThreadPool, vec: &mut [f32], cap: f32) {
    let n = vec.len();
    if n == 0 {
        return;
    }
    let inv_cap = 1.0 / cap;
    if pool.num_threads() <= 1 || n < 4096 {
        for l in vec {
            *l = (*l * inv_cap).tanh() * cap;
        }
        return;
    }
    let ptr = vec.as_mut_ptr() as usize;
    let chunk = n.div_ceil(pool.num_threads()).max(1024);
    pool.parallel_for(n, chunk, |i0, i1| {
        // SAFETY: Disjoint `[i0..i1)` slices across worker threads.
        let slice = unsafe { core::slice::from_raw_parts_mut((ptr as *mut f32).add(i0), i1 - i0) };
        for l in slice {
            *l = (*l * inv_cap).tanh() * cap;
        }
    });
}

/// Zero-copy transposed quantized matrix multiplication for backward passes (`ADR-015`):
/// computes `dx[n_tok, cols] = (accum ? dx : 0) + dy[n_tok, rows] * W[rows, cols]`
/// directly from read-only `mmap` quantized `w_bytes` without materializing or transposing `W`.
///
/// Parallelizes across disjoint column blocks (`cb`) so worker threads write to disjoint slices
/// of `dx` and accumulate `r = 0..rows` in strict sequential order — guaranteeing **0-ULP bit-exact
/// parity** across `generic` / SIMD ISAs, thread counts (`1..N`), and batch widths (`1..T`).
#[allow(clippy::too_many_arguments)]
pub fn matmul_q_transposed_f32(
    pool: &ThreadPool,
    kt: &KernelTable,
    w_dtype: DType,
    w_bytes: &[u8],
    dy: &[f32],
    dx: &mut [f32],
    rows: usize,
    cols: usize,
    n_tok: usize,
    accum: bool,
) {
    assert_eq!(dy.len(), n_tok * rows);
    assert_eq!(dx.len(), n_tok * cols);
    if n_tok == 0 || cols == 0 {
        return;
    }
    if !accum {
        dx.fill(0.0);
    }
    if rows == 0 {
        return;
    }

    let dx_ptr = dx.as_mut_ptr() as usize;

    match w_dtype {
        DType::Q4_0 => {
            assert!(cols.is_multiple_of(QK4_0));
            let cols_blocks = cols / QK4_0;
            assert_eq!(
                w_bytes.len(),
                rows * cols_blocks * core::mem::size_of::<BlockQ4_0>()
            );
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ4_0>()));
            // SAFETY: Alignment and byte length verified; `BlockQ4_0` is `#[repr(C)]` POD.
            let w: &[BlockQ4_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ4_0,
                    rows * cols_blocks,
                )
            };
            let chunk = dynamic_row_chunk(cols_blocks, pool.num_threads());
            pool.parallel_for(cols_blocks, chunk, |cb0, cb1| {
                const TILE_TOK: usize = 8;
                let mut tile = [0.0f32; TILE_TOK * QK4_0];
                for cb in cb0..cb1 {
                    let c_off = cb * QK4_0;
                    let mut t0 = 0usize;
                    while t0 < n_tok {
                        let bt = (n_tok - t0).min(TILE_TOK);
                        let sub = &mut tile[..bt * QK4_0];
                        for t in 0..bt {
                            // SAFETY: Disjoint column slice `[c_off .. c_off + 32)` per worker `cb`.
                            let src = unsafe {
                                core::slice::from_raw_parts(
                                    (dx_ptr as *const f32).add((t0 + t) * cols + c_off),
                                    QK4_0,
                                )
                            };
                            sub[t * QK4_0..(t + 1) * QK4_0].copy_from_slice(src);
                        }
                        (kt.gemm_q4_0_transposed_tile)(
                            w,
                            cols_blocks,
                            cb,
                            &dy[t0 * rows..(t0 + bt) * rows],
                            sub,
                            rows,
                            bt,
                        );
                        for t in 0..bt {
                            // SAFETY: Disjoint column slice `[c_off .. c_off + 32)` per worker `cb`.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (dx_ptr as *mut f32).add((t0 + t) * cols + c_off),
                                    QK4_0,
                                )
                            };
                            dst.copy_from_slice(&sub[t * QK4_0..(t + 1) * QK4_0]);
                        }
                        t0 += bt;
                    }
                }
            });
        }
        DType::Q8_0 => {
            assert!(cols.is_multiple_of(QK8_0));
            let cols_blocks = cols / QK8_0;
            assert_eq!(
                w_bytes.len(),
                rows * cols_blocks * core::mem::size_of::<BlockQ8_0>()
            );
            assert!((w_bytes.as_ptr() as usize).is_multiple_of(core::mem::align_of::<BlockQ8_0>()));
            // SAFETY: Alignment and byte length verified; `BlockQ8_0` is `#[repr(C)]` POD.
            let w: &[BlockQ8_0] = unsafe {
                core::slice::from_raw_parts(
                    w_bytes.as_ptr() as *const BlockQ8_0,
                    rows * cols_blocks,
                )
            };
            let chunk = dynamic_row_chunk(cols_blocks, pool.num_threads());
            pool.parallel_for(cols_blocks, chunk, |cb0, cb1| {
                for cb in cb0..cb1 {
                    let c_off = cb * QK8_0;
                    for r in 0..rows {
                        let blk = &w[r * cols_blocks + cb..r * cols_blocks + cb + 1];
                        for t in 0..n_tok {
                            let scale = dy[t * rows + r];
                            // SAFETY: Disjoint `[c_off .. c_off + 32)` per worker `cb`.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (dx_ptr as *mut f32).add(t * cols + c_off),
                                    QK8_0,
                                )
                            };
                            (kt.axpy_f32_q8_0)(dst, scale, blk);
                        }
                    }
                }
            });
        }
        DType::Q4_0x4 => {
            assert!(cols.is_multiple_of(QK4_0) && rows.is_multiple_of(4));
            let cols_blocks = cols / QK4_0;
            let group_bytes = cols_blocks * 72;
            assert_eq!(w_bytes.len(), (rows / 4) * group_bytes);
            let chunk = dynamic_row_chunk(cols_blocks, pool.num_threads());
            pool.parallel_for(cols_blocks, chunk, |cb0, cb1| {
                let mut w4 = [[0.0f32; QK4_0]; 4];
                for cb in cb0..cb1 {
                    let c_off = cb * QK4_0;
                    for rg in 0..(rows / 4) {
                        let sub_bytes =
                            &w_bytes[rg * group_bytes + cb * 72..rg * group_bytes + (cb + 1) * 72];
                        for (m, w_row) in w4.iter_mut().enumerate() {
                            dequantize_q4_0x4_single_row(sub_bytes, QK4_0, m, w_row);
                        }
                        for (m, w_row) in w4.iter().enumerate() {
                            let r = rg * 4 + m;
                            for t in 0..n_tok {
                                let scale = dy[t * rows + r];
                                // SAFETY: Disjoint `[c_off .. c_off + 32)` per worker `cb`.
                                let dst = unsafe {
                                    core::slice::from_raw_parts_mut(
                                        (dx_ptr as *mut f32).add(t * cols + c_off),
                                        QK4_0,
                                    )
                                };
                                (kt.axpy_f32)(dst, scale, w_row);
                            }
                        }
                    }
                }
            });
        }
        DType::F32 => {
            assert_eq!(w_bytes.len(), rows * cols * 4);
            const COL_TILE: usize = 64;
            let n_tiles = cols.div_ceil(COL_TILE);
            let chunk = dynamic_row_chunk(n_tiles, pool.num_threads());
            pool.parallel_for(n_tiles, chunk, |ct0, ct1| {
                let mut w_buf = [0.0f32; COL_TILE];
                for ct in ct0..ct1 {
                    let c0 = ct * COL_TILE;
                    let c_len = (cols - c0).min(COL_TILE);
                    for r in 0..rows {
                        let b_off = (r * cols + c0) * 4;
                        for (j, slot) in w_buf[..c_len].iter_mut().enumerate() {
                            let o = b_off + j * 4;
                            *slot = f32::from_le_bytes([
                                w_bytes[o],
                                w_bytes[o + 1],
                                w_bytes[o + 2],
                                w_bytes[o + 3],
                            ]);
                        }
                        let w_slice = &w_buf[..c_len];
                        for t in 0..n_tok {
                            let scale = dy[t * rows + r];
                            // SAFETY: Disjoint `[c0 .. c0 + c_len)` per worker `ct`.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (dx_ptr as *mut f32).add(t * cols + c0),
                                    c_len,
                                )
                            };
                            (kt.axpy_f32)(dst, scale, w_slice);
                        }
                    }
                }
            });
        }
        other => {
            // General block-parallel path covering `F16`, `BF16`, `Q4_1`, `Q5_K`, `Q6_K`,
            // `Q2_0`, `TQ2_0`, `Iq2Xxs`, `Iq3Xxs`, `Iq4Nl` (`block_elements <= 256`).
            let blk_elems = if matches!(other, DType::F16 | DType::BF16) {
                cols.min(64)
            } else {
                other.block_elements()
            };
            let blk_bytes = if matches!(other, DType::F16 | DType::BF16) {
                blk_elems * 2
            } else {
                other.block_bytes()
            };
            let row_bytes = other.row_bytes(cols);
            assert_eq!(w_bytes.len(), rows * row_bytes);
            let n_tiles = cols.div_ceil(blk_elems);
            let chunk = dynamic_row_chunk(n_tiles, pool.num_threads());
            pool.parallel_for(n_tiles, chunk, |ct0, ct1| {
                let mut w_buf = [0.0f32; QK_K];
                for ct in ct0..ct1 {
                    let c0 = ct * blk_elems;
                    let c_len = (cols - c0).min(blk_elems);
                    let byte_c0 = if matches!(other, DType::F16 | DType::BF16) {
                        c0 * 2
                    } else {
                        ct * blk_bytes
                    };
                    let byte_len = if matches!(other, DType::F16 | DType::BF16) {
                        c_len * 2
                    } else {
                        blk_bytes
                    };
                    for r in 0..rows {
                        let src_b =
                            &w_bytes[r * row_bytes + byte_c0..r * row_bytes + byte_c0 + byte_len];
                        dequantize_row_bytes(other, src_b, &mut w_buf[..c_len]);
                        let w_slice = &w_buf[..c_len];
                        for t in 0..n_tok {
                            let scale = dy[t * rows + r];
                            // SAFETY: Disjoint `[c0 .. c0 + c_len)` per worker `ct`.
                            let dst = unsafe {
                                core::slice::from_raw_parts_mut(
                                    (dx_ptr as *mut f32).add(t * cols + c0),
                                    c_len,
                                )
                            };
                            (kt.axpy_f32)(dst, scale, w_slice);
                        }
                    }
                }
            });
        }
    }
}

/// Exact analytical backward pass for `rms_norm`:
/// `y_i = x_i * rsqrt(mean(x^2) + eps) * weight_i`.
/// Accumulates into `dweight` when `Some`.
pub fn rms_norm_bwd_f32(
    dx: &mut [f32],
    mut dweight: Option<&mut [f32]>,
    x: &[f32],
    weight: &[f32],
    dy: &[f32],
    eps: f32,
) {
    let n = x.len();
    assert_eq!(dx.len(), n);
    assert_eq!(weight.len(), n);
    assert_eq!(dy.len(), n);
    if n == 0 {
        return;
    }

    let mut ss = 0.0f64;
    for &xi in x {
        let xf = xi as f64;
        ss += xf * xf;
    }
    let inv_rms = 1.0f64 / (ss / (n as f64) + (eps as f64)).sqrt();

    if let Some(ref mut dw) = dweight {
        assert_eq!(dw.len(), n);
        for i in 0..n {
            dw[i] += ((dy[i] as f64) * (x[i] as f64) * inv_rms) as f32;
        }
    }

    let mut dot_gx = 0.0f64;
    for i in 0..n {
        let gi = (dy[i] as f64) * (weight[i] as f64);
        dot_gx += gi * (x[i] as f64);
    }
    let coeff = (inv_rms * inv_rms) * dot_gx / (n as f64);
    for i in 0..n {
        let gi = (dy[i] as f64) * (weight[i] as f64);
        dx[i] = (inv_rms * (gi - (x[i] as f64) * coeff)) as f32;
    }
}

/// Exact analytical backward pass (orthogonal transpose `R(-theta)`) for `rope_inplace_tables`.
pub fn rope_bwd_inplace_tables(
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
                    let dy0 = head[2 * i];
                    let dy1 = head[2 * i + 1];
                    head[2 * i] = dy0 * c + dy1 * s;
                    head[2 * i + 1] = -dy0 * s + dy1 * c;
                }
            }
            RopeType::Neox => {
                for i in 0..half {
                    let c = cos[i];
                    let s = sin[i];
                    let dy0 = head[i];
                    let dy1 = head[i + half];
                    head[i] = dy0 * c + dy1 * s;
                    head[i + half] = -dy0 * s + dy1 * c;
                }
            }
        }
    }
}

/// Exact analytical derivative `d(gelu_tanh(x)) / dx` evaluated in `f64` (`ops.rs:627`).
#[inline]
pub fn gelu_tanh_bwd_f32(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f64 = 0.797_884_6_f32 as f64;
    const GELU_COEF_A: f64 = 0.044_715_f32 as f64;
    let xf = x as f64;
    let x2 = xf * xf;
    let u = SQRT_2_OVER_PI * xf * (1.0 + GELU_COEF_A * x2);
    let t = u.tanh();
    let du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEF_A * x2);
    (0.5 * (1.0 + t) + 0.5 * xf * (1.0 - t * t) * du) as f32
}

/// Exact analytical backward pass for `geglu` (`out = gelu_tanh(gate) * up`).
pub fn geglu_bwd_f32(
    d_gate: &mut [f32],
    d_up: &mut [f32],
    gate: &[f32],
    up: &[f32],
    d_out: &[f32],
) {
    let n = d_out.len();
    assert_eq!(d_gate.len(), n);
    assert_eq!(d_up.len(), n);
    assert_eq!(gate.len(), n);
    assert_eq!(up.len(), n);

    for i in 0..n {
        let g = gate[i];
        let dy = d_out[i] as f64;
        let gelu_g = gelu_tanh_f32(g) as f64;
        let dgelu_g = gelu_tanh_bwd_f32(g) as f64;
        d_up[i] = (dy * gelu_g) as f32;
        d_gate[i] = (dy * (up[i] as f64) * dgelu_g) as f32;
    }
}

/// Exact analytical backward pass for per-head `L2` normalization (`y = x / sqrt(sum(x^2) + eps)`).
pub fn l2_norm_bwd_f32(dx: &mut [f32], x: &[f32], dy: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(dx.len(), n);
    assert_eq!(dy.len(), n);
    if n == 0 {
        return;
    }
    let mut ss = 0.0f64;
    let mut dot = 0.0f64;
    for i in 0..n {
        let xi = x[i] as f64;
        ss += xi * xi;
        dot += (dy[i] as f64) * xi;
    }
    let inv_norm2 = 1.0 / (ss + (eps as f64));
    let inv_norm = inv_norm2.sqrt();
    let coeff = inv_norm2 * dot;
    for i in 0..n {
        dx[i] = (inv_norm * ((dy[i] as f64) - (x[i] as f64) * coeff)) as f32;
    }
}

/// Exact analytical backward pass for unit-weight RMS normalization (`y = x / sqrt(mean(x^2) + eps)`).
pub fn rms_norm_unit_bwd_f32(dx: &mut [f32], x: &[f32], dy: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(dx.len(), n);
    assert_eq!(dy.len(), n);
    if n == 0 {
        return;
    }
    let mut ss = 0.0f64;
    let mut dot = 0.0f64;
    for i in 0..n {
        let xi = x[i] as f64;
        ss += xi * xi;
        dot += (dy[i] as f64) * xi;
    }
    let inv_rms2 = 1.0 / (ss / (n as f64) + (eps as f64));
    let inv_rms = inv_rms2.sqrt();
    let coeff = inv_rms2 * dot / (n as f64);
    for i in 0..n {
        dx[i] = (inv_rms * ((dy[i] as f64) - (x[i] as f64) * coeff)) as f32;
    }
}

/// Exact analytical backward pass for `swiglu` (`out = silu(gate) * up`).
pub fn swiglu_bwd_f32(
    d_gate: &mut [f32],
    d_up: &mut [f32],
    gate: &[f32],
    up: &[f32],
    d_out: &[f32],
) {
    let n = d_out.len();
    assert_eq!(d_gate.len(), n);
    assert_eq!(d_up.len(), n);
    assert_eq!(gate.len(), n);
    assert_eq!(up.len(), n);

    for i in 0..n {
        let g = gate[i] as f64;
        let u = up[i] as f64;
        let dy = d_out[i] as f64;
        let sig = 1.0 / (1.0 + (-g).exp());
        let silu = g * sig;
        let dsilu = sig * (1.0 + g * (1.0 - sig));
        d_up[i] = (dy * silu) as f32;
        d_gate[i] = (dy * u * dsilu) as f32;
    }
}

/// Causal GQA attention backward pass over an `F16` KV sequence (`k_f16`, `v_f16` of length `n_tok * kv_dim`),
/// matching `gqa_attention_f16_kv_batch_parallel` forward with straight-through estimator (STE) on `F16` K/V.
///
/// Deterministic accumulation order (`t = 0..n_tok, h = 0..n_heads`) guarantees reproducible `dk`/`dv`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_bwd_f32(
    kt: &KernelTable,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    d_out: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_tok: usize,
    window: Option<usize>,
) {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    gqa_attention_f16_kv_bwd_ext_f32(
        kt, dq, dk, dv, q, k_f16, v_f16, d_out, n_heads, n_kv_heads, head_dim, n_tok, scale, window,
    );
}

/// Extended causal GQA attention backward pass supporting custom pre-softmax `attn_scale`
/// (e.g. `1.0` on Gemma 4) and optional sliding window `window`, with zero heap allocations
/// for `n_tok <= 2048`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_bwd_ext_f32(
    kt: &KernelTable,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    d_out: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
) {
    gqa_attention_f16_kv_bwd_softcap_f32(
        kt, dq, dk, dv, q, k_f16, v_f16, d_out, n_heads, n_kv_heads, head_dim, n_tok, attn_scale,
        window, None,
    );
}

/// Extended causal GQA attention backward pass supporting custom `attn_scale`, `window`, and `attn_softcap`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f16_kv_bwd_softcap_f32(
    kt: &KernelTable,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    q: &[f32],
    k_f16: &[u16],
    v_f16: &[u16],
    d_out: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
    attn_softcap: Option<f32>,
) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    assert_eq!(dq.len(), n_tok * q_dim);
    assert_eq!(dk.len(), n_tok * kv_dim);
    assert_eq!(dv.len(), n_tok * kv_dim);
    assert_eq!(q.len(), n_tok * q_dim);
    assert_eq!(k_f16.len(), n_tok * kv_dim);
    assert_eq!(v_f16.len(), n_tok * kv_dim);
    assert_eq!(d_out.len(), n_tok * q_dim);

    dq.fill(0.0);
    dk.fill(0.0);
    dv.fill(0.0);
    if n_tok == 0 {
        return;
    }

    const STACK_CAP: usize = 2048;
    let mut stack_probs = [0.0f32; STACK_CAP];
    let mut stack_dp = [0.0f32; STACK_CAP];
    let mut stack_sech2 = [1.0f32; STACK_CAP];
    let mut heap_probs: Vec<f32>;
    let mut heap_dp: Vec<f32>;
    let mut heap_sech2: Vec<f32>;
    let (probs_all, dp_all, sech2_all): (&mut [f32], &mut [f32], &mut [f32]) = if n_tok <= STACK_CAP
    {
        (
            &mut stack_probs[..n_tok],
            &mut stack_dp[..n_tok],
            &mut stack_sech2[..n_tok],
        )
    } else {
        heap_probs = vec![0.0f32; n_tok];
        heap_dp = vec![0.0f32; n_tok];
        heap_sech2 = vec![1.0f32; n_tok];
        (&mut heap_probs[..], &mut heap_dp[..], &mut heap_sech2[..])
    };

    let group = n_heads / n_kv_heads;
    let scale = attn_scale;

    for t in 0..n_tok {
        let end_pos = t + 1;
        let start_kv = match window {
            Some(w) if w > 0 => end_pos.saturating_sub(w),
            _ => 0,
        };
        let win_len = end_pos - start_kv;
        let p = &mut probs_all[..win_len];
        let d_prob = &mut dp_all[..win_len];
        let s2 = &mut sech2_all[..win_len];

        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let doh = &d_out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];

            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let k_off = step * kv_dim + kv_h * head_dim;
                let raw = (kt.dot_f32_f16)(qh, &k_f16[k_off..k_off + head_dim]) * scale;
                if let Some(cap) = attn_softcap
                    && cap > 0.0
                {
                    let th = (raw / cap).tanh();
                    p[w_idx] = th * cap;
                    s2[w_idx] = 1.0 - th * th;
                } else {
                    p[w_idx] = raw;
                    s2[w_idx] = 1.0;
                }
            }
            softmax_inplace(p);

            let mut sum_p_dp = 0.0f64;
            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let pw = p[w_idx];
                let v_off = step * kv_dim + kv_h * head_dim;
                let dvh = &mut dv[v_off..v_off + head_dim];
                let vh_f16 = &v_f16[v_off..v_off + head_dim];

                let mut dot_dov = 0.0f64;
                for d in 0..head_dim {
                    dvh[d] = doh[d].mul_add(pw, dvh[d]);
                    dot_dov += (doh[d] as f64) * (f16_to_f32(vh_f16[d]) as f64);
                }
                d_prob[w_idx] = dot_dov as f32;
                sum_p_dp += (pw as f64) * dot_dov;
            }

            let dqh = &mut dq[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            for (w_idx, step) in (start_kv..end_pos).enumerate() {
                let ds = ((p[w_idx] as f64)
                    * ((d_prob[w_idx] as f64) - sum_p_dp)
                    * (s2[w_idx] as f64)
                    * (scale as f64)) as f32;
                if ds == 0.0 {
                    continue;
                }
                let k_off = step * kv_dim + kv_h * head_dim;
                let kh_f16 = &k_f16[k_off..k_off + head_dim];
                (kt.axpy_f32_f16)(dqh, ds, kh_f16);
                let dkh = &mut dk[k_off..k_off + head_dim];
                (kt.axpy_f32)(dkh, ds, qh);
            }
        }
    }
}

/// Softplus matching `ggml_compute_softplus_f32`: `if x > 20.0 { x } else { (1.0 + x.exp()).ln() }`.
#[inline]
pub fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Squared-ReLU activation (`y = max(x, 0)^2`, used by `Nemotron-H` FFN layers).
pub fn relu2_inplace_parallel(pool: &ThreadPool, x: &mut [f32]) {
    if pool.num_threads() == 1 || x.len() < 1024 {
        for v in x.iter_mut() {
            let r = v.max(0.0);
            *v = r * r;
        }
        return;
    }
    let ptr = x.as_mut_ptr() as usize;
    let chunk = dynamic_row_chunk(x.len(), pool.num_threads()).max(256);
    pool.parallel_for(x.len(), chunk, |s, e| {
        // SAFETY: Disjoint `[s..e)` range in `x`.
        let sub = unsafe { core::slice::from_raw_parts_mut((ptr as *mut f32).add(s), e - s) };
        for v in sub {
            let r = v.max(0.0);
            *v = r * r;
        }
    });
}

/// Analytical backward for Squared-ReLU (`dx = dy * 2 * max(x_pre, 0)`).
pub fn relu2_bwd_f32(dx: &mut [f32], x_pre: &[f32], dy: &[f32]) {
    assert_eq!(dx.len(), x_pre.len());
    assert_eq!(dx.len(), dy.len());
    for ((dst, &x), &g) in dx.iter_mut().zip(x_pre.iter()).zip(dy.iter()) {
        *dst = if x > 0.0 { 2.0 * x * g } else { 0.0 };
    }
}

/// Multi-head parallel Mamba-2 selective state scan across `n_tok` sequential steps (`ADR-017`).
///
/// Preserves **0-ULP thread-count and batch-width invariance** (`AGENTS.md` §9): each head `h`
/// owns an independent `[head_dim, d_state]` slice of `ssm_state` and processes `t = 0..n_tok`
/// sequentially via `(kt.mamba2_scan_head)`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_scan_parallel(
    pool: &ThreadPool,
    kt: &KernelTable,
    x_all: &[f32],
    b_all: &[f32],
    c_all: &[f32],
    dt_raw_all: &[f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    ssm_d: &[f32],
    ssm_state: &mut [f32],
    y_all: &mut [f32],
    n_heads: usize,
    n_groups: usize,
    head_dim: usize,
    d_state: usize,
    n_tok: usize,
) {
    let d_inner = n_heads * head_dim;
    let bc_dim = n_groups * d_state;
    let group_ratio = n_heads / n_groups;
    assert_eq!(x_all.len(), n_tok * d_inner);
    assert_eq!(b_all.len(), n_tok * bc_dim);
    assert_eq!(c_all.len(), n_tok * bc_dim);
    assert_eq!(dt_raw_all.len(), n_tok * n_heads);
    assert_eq!(dt_bias.len(), n_heads);
    assert_eq!(ssm_a.len(), n_heads);
    assert_eq!(ssm_d.len(), n_heads);
    assert_eq!(ssm_state.len(), d_inner * d_state);
    assert_eq!(y_all.len(), n_tok * d_inner);

    let state_per_head = head_dim * d_state;
    if pool.num_threads() == 1 || n_heads < 4 {
        for h in 0..n_heads {
            let g = h / group_ratio;
            let s_h = &mut ssm_state[h * state_per_head..(h + 1) * state_per_head];
            let a_h = ssm_a[h];
            let d_h = ssm_d[h];
            let dt_b = dt_bias[h];
            for t in 0..n_tok {
                let dt_sp = softplus_f32(dt_raw_all[t * n_heads + h] + dt_b);
                let da = (dt_sp * a_h).exp();
                let x_off = t * d_inner + h * head_dim;
                let bc_off = t * bc_dim + g * d_state;
                (kt.mamba2_scan_head)(
                    &x_all[x_off..x_off + head_dim],
                    &b_all[bc_off..bc_off + d_state],
                    &c_all[bc_off..bc_off + d_state],
                    da,
                    dt_sp,
                    d_h,
                    s_h,
                    &mut y_all[x_off..x_off + head_dim],
                    head_dim,
                    d_state,
                );
            }
        }
        return;
    }

    let s_ptr = ssm_state.as_mut_ptr() as usize;
    let y_ptr = y_all.as_mut_ptr() as usize;
    let chunk = n_heads.div_ceil(pool.num_threads()).max(1);
    pool.parallel_for(n_heads, chunk, |h0, h1| {
        for h in h0..h1 {
            let g = h / group_ratio;
            // SAFETY: Disjoint `[h * state_per_head .. (h + 1) * state_per_head)` per head `h`.
            let s_h = unsafe {
                core::slice::from_raw_parts_mut(
                    (s_ptr as *mut f32).add(h * state_per_head),
                    state_per_head,
                )
            };
            let a_h = ssm_a[h];
            let d_h = ssm_d[h];
            let dt_b = dt_bias[h];
            for t in 0..n_tok {
                let dt_sp = softplus_f32(dt_raw_all[t * n_heads + h] + dt_b);
                let da = (dt_sp * a_h).exp();
                let x_off = t * d_inner + h * head_dim;
                let bc_off = t * bc_dim + g * d_state;
                // SAFETY: Disjoint `[t * d_inner + h * head_dim .. + head_dim)` per `(t, h)`.
                let y_h = unsafe {
                    core::slice::from_raw_parts_mut((y_ptr as *mut f32).add(x_off), head_dim)
                };
                (kt.mamba2_scan_head)(
                    &x_all[x_off..x_off + head_dim],
                    &b_all[bc_off..bc_off + d_state],
                    &c_all[bc_off..bc_off + d_state],
                    da,
                    dt_sp,
                    d_h,
                    s_h,
                    y_h,
                    head_dim,
                    d_state,
                );
            }
        }
    });
}
