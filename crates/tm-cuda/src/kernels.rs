//! Typed kernel table & launch wrappers for `cuda_ops.ptx`.

use std::ffi::c_void;
use std::sync::Arc;

use tm_core::DType;

use crate::CudaError;
use crate::driver::{CUdeviceptr, CUfunction, CudaBuffer, CudaContext};

/// Maximum tokens handled by the register-cached MMVQ decode/small-batch path;
/// batches above `MMVQ_MAX_TOK` dispatch to the `wmma` 16×16×16 Tensor Core GEMM!
pub const MMVQ_MAX_TOK: usize = 8;

/// Size of one `BlockQ8_1` (`half2 ds` + `int8_t qs[32]`) in bytes.
pub const BLOCK_Q8_1_BYTES: usize = 36;

/// Maps `tm_core::DType` to the integer tag expected by `cuda_ops.cu`.
pub fn dtype_cuda_tag(dt: DType) -> Result<i32, CudaError> {
    match dt {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::Q4_0 => Ok(2),
        DType::Q4_1 => Ok(3),
        DType::Q5_0 => Ok(6),
        DType::Q8_0 => Ok(8),
        DType::Q3K => Ok(11),
        DType::Q4K => Ok(12),
        DType::Q5K => Ok(13),
        DType::Q6K => Ok(14),
        DType::Iq4Nl => Ok(20),
        DType::Iq4Xs => Ok(23),
        DType::BF16 => Ok(30),
        DType::NvFp4 => Ok(40),
        DType::Q2_0 => Ok(42),
        other => Err(CudaError::UnsupportedDType(other)),
    }
}

/// Returns `true` when `dt` uses the `Q8_1` (`__dp4a`) MMVQ path in `launch_linear`.
#[inline]
pub fn is_q8_1_mmvq_dtype(dt: DType) -> bool {
    matches!(
        dt,
        DType::Q2_0
            | DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q8_0
            | DType::Q3K
            | DType::Q4K
            | DType::Q5K
            | DType::Q6K
            | DType::Iq4Nl
            | DType::Iq4Xs
    )
}

/// Device weight matrix handle carrying its own `DType` (`AGENTS.md` Rule #5).
#[derive(Clone)]
pub struct CudaMat {
    pub buf: Arc<CudaBuffer>,
    pub offset: usize,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
}

impl CudaMat {
    #[inline]
    pub fn ptr(&self) -> CUdeviceptr {
        self.buf.device_ptr_at(self.offset)
    }
}

/// Resolved `CUfunction` handles for every kernel in `cuda_ops.ptx`.
pub struct CudaKernels {
    pub quantize_q8_1_f32: CUfunction,
    pub swiglu_quantize_q8_1_f32: CUfunction,
    pub swiglu_fused_quantize_q8_1_f32: CUfunction,
    pub mmvq_q2_0_q8_1: CUfunction,
    pub mmvq_q4_0_q8_1: CUfunction,
    pub mmvq_q4_1_q8_1: Option<CUfunction>,
    pub mmvq_q5_0_q8_1: CUfunction,
    pub mmvq_q8_0_q8_1: CUfunction,
    pub mmvq_q3_k_q8_1: CUfunction,
    pub mmvq_q4_k_q8_1: CUfunction,
    pub mmvq_q5_k_q8_1: CUfunction,
    pub mmvq_q6_k_q8_1: CUfunction,
    pub mmvq_iq4_nl_q8_1: CUfunction,
    pub mmvq_iq4_xs_q8_1: CUfunction,
    pub mmvq_float_f32: CUfunction,
    pub wmma_gemm_mixed_f32: CUfunction,
    pub embed_tokens_f32: CUfunction,
    pub rms_norm_rows_f32: CUfunction,
    pub qk_rms_norm_inplace_f32: CUfunction,
    pub add_bias_rows_f32: CUfunction,
    pub add_scaled_inplace_f32: CUfunction,
    pub scale_inplace_f32: CUfunction,
    pub copy_2d_rows_f32: CUfunction,
    pub swiglu_f32: CUfunction,
    pub swiglu_fused_f32: CUfunction,
    pub sigmoid_gate_mul_f32: CUfunction,
    pub scale_rows_by_sigmoid_scalar_f32: CUfunction,
    pub split_qwen35_q_gate_f32: CUfunction,
    pub apply_rope_qk_f32: CUfunction,
    pub write_kv_cache_f16: CUfunction,
    pub write_kv_cache_q8_0: CUfunction,
    pub flash_attn_gqa_f16_kv: CUfunction,
    pub flash_attn_gqa_q8_0_kv: CUfunction,
    pub gdn_conv1d_silu_split_step_f32: CUfunction,
    pub gdn_l2_norm_qk_f32: CUfunction,
    pub gdn_gates_f32: CUfunction,
    pub gdn_recurrence_step_f32: CUfunction,
    pub gdn_rms_norm_gated_f32: CUfunction,
    pub moe_router_topk_f32: CUfunction,
    pub moe_indexed_gemv_q8_1: CUfunction,
    pub moe_indexed_dual_gu_act_f32: CUfunction,
    pub moe_build_expert_buckets_u32: CUfunction,
    pub wmma_moe_grouped_f32: CUfunction,
    pub moe_reduce_slots_f32: CUfunction,
    pub moe_indexed_gemv_f32: CUfunction,
    pub mtp_rms_norm_cat2_f32: CUfunction,
    pub elemwise_mul_add_f32: CUfunction,
    pub dfly_bias_sigmoid_slot_f32: CUfunction,
    pub argmax_f32: CUfunction,
}

#[allow(clippy::too_many_arguments)]
impl CudaKernels {
    pub fn load(ctx: &CudaContext) -> Result<Self, CudaError> {
        Ok(Self {
            quantize_q8_1_f32: ctx.get_function("quantize_q8_1_f32")?,
            swiglu_quantize_q8_1_f32: ctx.get_function("swiglu_quantize_q8_1_f32")?,
            swiglu_fused_quantize_q8_1_f32: ctx.get_function("swiglu_fused_quantize_q8_1_f32")?,
            mmvq_q2_0_q8_1: ctx.get_function("mmvq_q2_0_q8_1")?,
            mmvq_q4_0_q8_1: ctx.get_function("mmvq_q4_0_q8_1")?,
            mmvq_q4_1_q8_1: ctx.get_function("mmvq_q4_1_q8_1").ok(),
            mmvq_q5_0_q8_1: ctx.get_function("mmvq_q5_0_q8_1")?,
            mmvq_q8_0_q8_1: ctx.get_function("mmvq_q8_0_q8_1")?,
            mmvq_q3_k_q8_1: ctx.get_function("mmvq_q3_k_q8_1")?,
            mmvq_q4_k_q8_1: ctx.get_function("mmvq_q4_k_q8_1")?,
            mmvq_q5_k_q8_1: ctx.get_function("mmvq_q5_k_q8_1")?,
            mmvq_q6_k_q8_1: ctx.get_function("mmvq_q6_k_q8_1")?,
            mmvq_iq4_nl_q8_1: ctx.get_function("mmvq_iq4_nl_q8_1")?,
            mmvq_iq4_xs_q8_1: ctx.get_function("mmvq_iq4_xs_q8_1")?,
            mmvq_float_f32: ctx.get_function("mmvq_float_f32")?,
            wmma_gemm_mixed_f32: ctx.get_function("wmma_gemm_mixed_f32")?,
            embed_tokens_f32: ctx.get_function("embed_tokens_f32")?,
            rms_norm_rows_f32: ctx.get_function("rms_norm_rows_f32")?,
            qk_rms_norm_inplace_f32: ctx.get_function("qk_rms_norm_inplace_f32")?,
            add_bias_rows_f32: ctx.get_function("add_bias_rows_f32")?,
            add_scaled_inplace_f32: ctx.get_function("add_scaled_inplace_f32")?,
            scale_inplace_f32: ctx.get_function("scale_inplace_f32")?,
            copy_2d_rows_f32: ctx.get_function("copy_2d_rows_f32")?,
            swiglu_f32: ctx.get_function("swiglu_f32")?,
            swiglu_fused_f32: ctx.get_function("swiglu_fused_f32")?,
            sigmoid_gate_mul_f32: ctx.get_function("sigmoid_gate_mul_f32")?,
            scale_rows_by_sigmoid_scalar_f32: ctx
                .get_function("scale_rows_by_sigmoid_scalar_f32")?,
            split_qwen35_q_gate_f32: ctx.get_function("split_qwen35_q_gate_f32")?,
            apply_rope_qk_f32: ctx.get_function("apply_rope_qk_f32")?,
            write_kv_cache_f16: ctx.get_function("write_kv_cache_f16")?,
            write_kv_cache_q8_0: ctx.get_function("write_kv_cache_q8_0")?,
            flash_attn_gqa_f16_kv: ctx.get_function("flash_attn_gqa_f16_kv")?,
            flash_attn_gqa_q8_0_kv: ctx.get_function("flash_attn_gqa_q8_0_kv")?,
            gdn_conv1d_silu_split_step_f32: ctx.get_function("gdn_conv1d_silu_split_step_f32")?,
            gdn_l2_norm_qk_f32: ctx.get_function("gdn_l2_norm_qk_f32")?,
            gdn_gates_f32: ctx.get_function("gdn_gates_f32")?,
            gdn_recurrence_step_f32: ctx.get_function("gdn_recurrence_step_f32")?,
            gdn_rms_norm_gated_f32: ctx.get_function("gdn_rms_norm_gated_f32")?,
            moe_router_topk_f32: ctx.get_function("moe_router_topk_f32")?,
            moe_indexed_gemv_q8_1: ctx.get_function("moe_indexed_gemv_q8_1")?,
            moe_indexed_dual_gu_act_f32: ctx.get_function("moe_indexed_dual_gu_act_f32")?,
            moe_build_expert_buckets_u32: ctx.get_function("moe_build_expert_buckets_u32")?,
            wmma_moe_grouped_f32: ctx.get_function("wmma_moe_grouped_f32")?,
            moe_reduce_slots_f32: ctx.get_function("moe_reduce_slots_f32")?,
            moe_indexed_gemv_f32: ctx.get_function("moe_indexed_gemv_f32")?,
            mtp_rms_norm_cat2_f32: ctx.get_function("mtp_rms_norm_cat2_f32")?,
            elemwise_mul_add_f32: ctx.get_function("elemwise_mul_add_f32")?,
            dfly_bias_sigmoid_slot_f32: ctx.get_function("dfly_bias_sigmoid_slot_f32")?,
            argmax_f32: ctx.get_function("argmax_f32")?,
        })
    }

    /// Quantizes `x` (`[n_tok, cols]` f32) into `xq_buf` (`[n_tok, cols / 32]` `BlockQ8_1`).
    pub fn launch_quantize_q8_1(
        &self,
        ctx: &CudaContext,
        x_ptr: CUdeviceptr,
        xq_ptr: CUdeviceptr,
        cols: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut p_x = x_ptr;
        let mut p_xq = xq_ptr;
        let mut c_cols = cols as i32;
        let mut c_ntok = n_tok as i32;
        let n_blocks = (cols / 32) as u32;
        let mut args: [*mut c_void; 4] = [
            (&mut p_x as *mut CUdeviceptr).cast(),
            (&mut p_xq as *mut CUdeviceptr).cast(),
            (&mut c_cols as *mut i32).cast(),
            (&mut c_ntok as *mut i32).cast(),
        ];
        // SAFETY: Argument types match `quantize_q8_1_f32(const float*, BlockQ8_1*, int, int)`.
        unsafe {
            ctx.launch(
                "quantize_q8_1_f32",
                self.quantize_q8_1_f32,
                (n_blocks.div_ceil(4), n_tok as u32, 1),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    /// Fuses `silu(gate) * up` with `Q8_1` activation quantization in a single pass.
    pub fn launch_swiglu_quantize_q8_1(
        &self,
        ctx: &CudaContext,
        gate_ptr: CUdeviceptr,
        up_ptr: CUdeviceptr,
        xq_ptr: CUdeviceptr,
        cols: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut p_g = gate_ptr;
        let mut p_u = up_ptr;
        let mut p_xq = xq_ptr;
        let mut c_cols = cols as i32;
        let mut c_ntok = n_tok as i32;
        let n_blocks = (cols / 32) as u32;
        let mut args: [*mut c_void; 5] = [
            (&mut p_g as *mut CUdeviceptr).cast(),
            (&mut p_u as *mut CUdeviceptr).cast(),
            (&mut p_xq as *mut CUdeviceptr).cast(),
            (&mut c_cols as *mut i32).cast(),
            (&mut c_ntok as *mut i32).cast(),
        ];
        // SAFETY: Matches `swiglu_quantize_q8_1_f32`.
        unsafe {
            ctx.launch(
                "swiglu_quantize_q8_1_f32",
                self.swiglu_quantize_q8_1_f32,
                (n_blocks.div_ceil(4), n_tok as u32, 1),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    /// Fuses `[gate, up]` (`[n_rows, 2 * n_ff]`) SwiGLU + `Q8_1` quantization (`[n_rows, n_ff / 32]`).
    pub fn launch_swiglu_fused_quantize_q8_1(
        &self,
        ctx: &CudaContext,
        gate_up_ptr: CUdeviceptr,
        xq_ptr: CUdeviceptr,
        n_ff: usize,
        n_rows: usize,
    ) -> Result<(), CudaError> {
        let mut p_gu = gate_up_ptr;
        let mut p_xq = xq_ptr;
        let mut c_nff = n_ff as i32;
        let mut c_rows = n_rows as i32;
        let n_blocks = (n_ff / 32) as u32;
        let mut args: [*mut c_void; 4] = [
            (&mut p_gu as *mut CUdeviceptr).cast(),
            (&mut p_xq as *mut CUdeviceptr).cast(),
            (&mut c_nff as *mut i32).cast(),
            (&mut c_rows as *mut i32).cast(),
        ];
        // SAFETY: Matches `swiglu_fused_quantize_q8_1_f32`.
        unsafe {
            ctx.launch(
                "swiglu_fused_quantize_q8_1_f32",
                self.swiglu_fused_quantize_q8_1_f32,
                (n_blocks.div_ceil(4), n_rows as u32, 1),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    /// Dispatches `out[n_tok, w.rows] = x[n_tok, w.cols] @ w^T`:
    /// - For `n_tok <= MMVQ_MAX_TOK`: uses `__dp4a` MMVQ (`xq_ptr` pre-quantized when `xq_already_quantized` is true);
    /// - For `n_tok > MMVQ_MAX_TOK` and `cc >= 7.0`: uses `wmma` `16×16×16` FP16 Tensor Core GEMM!
    #[allow(clippy::too_many_arguments)]
    pub fn launch_linear(
        &self,
        ctx: &CudaContext,
        w: &CudaMat,
        x_ptr: CUdeviceptr,
        xq_ptr: CUdeviceptr,
        xq_already_quantized: bool,
        out_ptr: CUdeviceptr,
        n_tok: usize,
        force_mmvq: bool,
    ) -> Result<(), CudaError> {
        if n_tok == 0 || w.rows == 0 {
            return Ok(());
        }
        let use_wmma = !force_mmvq
            && n_tok > MMVQ_MAX_TOK
            && ctx.info().has_fp16_wmma()
            && w.cols.is_multiple_of(32);

        if use_wmma {
            let mut p_w = w.ptr();
            let mut p_x = x_ptr;
            let mut p_out = out_ptr;
            let mut tag = dtype_cuda_tag(w.dtype)?;
            let mut rows = w.rows as i32;
            let mut cols = w.cols as i32;
            let mut rb = w.row_bytes;
            let mut nt = n_tok as i32;
            let mut args: [*mut c_void; 8] = [
                (&mut p_w as *mut CUdeviceptr).cast(),
                (&mut p_x as *mut CUdeviceptr).cast(),
                (&mut p_out as *mut CUdeviceptr).cast(),
                (&mut tag as *mut i32).cast(),
                (&mut rows as *mut i32).cast(),
                (&mut cols as *mut i32).cast(),
                (&mut rb as *mut usize).cast(),
                (&mut nt as *mut i32).cast(),
            ];
            // SAFETY: Matches `wmma_gemm_mixed_f32`.
            return unsafe {
                ctx.launch(
                    "wmma_gemm_mixed_f32",
                    self.wmma_gemm_mixed_f32,
                    ((w.rows as u32).div_ceil(64), (n_tok as u32).div_ceil(32), 1),
                    (32, 4, 1),
                    0,
                    &mut args,
                )
            };
        }

        // Chunk `n_tok` in windows of `MMVQ_MAX_TOK` (8)
        let mut off = 0usize;
        let mut q_ready = xq_already_quantized;
        while off < n_tok {
            let b = (n_tok - off).min(MMVQ_MAX_TOK);
            let x_sub = x_ptr + (off * w.cols * 4) as u64;
            let out_sub = out_ptr + (off * w.rows * 4) as u64;

            match w.dtype {
                DType::Q2_0
                | DType::Q4_0
                | DType::Q5_0
                | DType::Q8_0
                | DType::Q3K
                | DType::Q4K
                | DType::Q5K
                | DType::Q6K
                | DType::Iq4Nl
                | DType::Iq4Xs => {
                    let xq_sub = if q_ready && off == 0 && n_tok <= MMVQ_MAX_TOK {
                        xq_ptr
                    } else {
                        self.launch_quantize_q8_1(ctx, x_sub, xq_ptr, w.cols, b)?;
                        q_ready = false;
                        xq_ptr
                    };
                    let (func, kname, coop_4w): (CUfunction, &'static str, bool) = match w.dtype {
                        DType::Q2_0 => (self.mmvq_q2_0_q8_1, "mmvq_q2_0_q8_1", false),
                        DType::Q4_0 => (
                            self.mmvq_q4_0_q8_1,
                            "mmvq_q4_0_q8_1",
                            b == 1 && w.cols >= 2048,
                        ),
                        DType::Q5_0 => (self.mmvq_q5_0_q8_1, "mmvq_q5_0_q8_1", false),
                        DType::Q8_0 => (
                            self.mmvq_q8_0_q8_1,
                            "mmvq_q8_0_q8_1",
                            b == 1 && w.cols >= 1024,
                        ),
                        DType::Q3K => (self.mmvq_q3_k_q8_1, "mmvq_q3_k_q8_1", false),
                        DType::Q4K => (
                            self.mmvq_q4_k_q8_1,
                            "mmvq_q4_k_q8_1",
                            b == 1 && w.cols >= 1024,
                        ),
                        DType::Q5K => (
                            self.mmvq_q5_k_q8_1,
                            "mmvq_q5_k_q8_1",
                            b == 1 && w.cols >= 1024,
                        ),
                        DType::Q6K => (
                            self.mmvq_q6_k_q8_1,
                            "mmvq_q6_k_q8_1",
                            b == 1 && w.cols >= 2048,
                        ),
                        DType::Iq4Nl => (self.mmvq_iq4_nl_q8_1, "mmvq_iq4_nl_q8_1", false),
                        DType::Iq4Xs => (
                            self.mmvq_iq4_xs_q8_1,
                            "mmvq_iq4_xs_q8_1",
                            b == 1 && w.cols >= 1024,
                        ),
                        _ => unreachable!(),
                    };
                    let grid_x = if coop_4w {
                        w.rows as u32
                    } else {
                        (w.rows as u32).div_ceil(4)
                    };
                    let mut p_w = w.ptr();
                    let mut p_xq = xq_sub;
                    let mut p_out = out_sub;
                    let mut rows = w.rows as i32;
                    let mut cols = w.cols as i32;
                    let mut nt = b as i32;
                    let mut args: [*mut c_void; 6] = [
                        (&mut p_w as *mut CUdeviceptr).cast(),
                        (&mut p_xq as *mut CUdeviceptr).cast(),
                        (&mut p_out as *mut CUdeviceptr).cast(),
                        (&mut rows as *mut i32).cast(),
                        (&mut cols as *mut i32).cast(),
                        (&mut nt as *mut i32).cast(),
                    ];
                    // SAFETY: Matches `mmvq_*_q8_1` signature.
                    unsafe {
                        ctx.launch(kname, func, (grid_x, 1, 1), (32, 4, 1), 0, &mut args)?;
                    }
                }
                DType::Q4_1 if self.mmvq_q4_1_q8_1.is_some() => {
                    let func = self.mmvq_q4_1_q8_1.expect("checked Some");
                    let xq_sub = if q_ready && off == 0 && n_tok <= MMVQ_MAX_TOK {
                        xq_ptr
                    } else {
                        self.launch_quantize_q8_1(ctx, x_sub, xq_ptr, w.cols, b)?;
                        q_ready = false;
                        xq_ptr
                    };
                    let coop_4w = b == 1 && w.cols >= 2048;
                    let grid_x = if coop_4w {
                        w.rows as u32
                    } else {
                        (w.rows as u32).div_ceil(4)
                    };
                    let mut p_w = w.ptr();
                    let mut p_xq = xq_sub;
                    let mut p_out = out_sub;
                    let mut rows = w.rows as i32;
                    let mut cols = w.cols as i32;
                    let mut nt = b as i32;
                    let mut args: [*mut c_void; 6] = [
                        (&mut p_w as *mut CUdeviceptr).cast(),
                        (&mut p_xq as *mut CUdeviceptr).cast(),
                        (&mut p_out as *mut CUdeviceptr).cast(),
                        (&mut rows as *mut i32).cast(),
                        (&mut cols as *mut i32).cast(),
                        (&mut nt as *mut i32).cast(),
                    ];
                    // SAFETY: Matches `mmvq_q4_1_q8_1` signature.
                    unsafe {
                        ctx.launch(
                            "mmvq_q4_1_q8_1",
                            func,
                            (grid_x, 1, 1),
                            (32, 4, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }
                DType::F16 | DType::BF16 | DType::F32 | DType::Q4_1 | DType::NvFp4 => {
                    let coop_4w = b == 1
                        && w.cols >= 1024
                        && matches!(w.dtype, DType::BF16 | DType::F16 | DType::F32)
                        && w.cols.is_multiple_of(8)
                        && ((w.ptr() as usize) | w.row_bytes) & 15 == 0;
                    let grid_x = if coop_4w {
                        w.rows as u32
                    } else {
                        (w.rows as u32).div_ceil(4)
                    };
                    let mut p_w = w.ptr();
                    let mut p_x = x_sub;
                    let mut p_out = out_sub;
                    let mut tag = dtype_cuda_tag(w.dtype)?;
                    let mut rows = w.rows as i32;
                    let mut cols = w.cols as i32;
                    let mut rb = w.row_bytes;
                    let mut nt = b as i32;
                    let mut args: [*mut c_void; 8] = [
                        (&mut p_w as *mut CUdeviceptr).cast(),
                        (&mut p_x as *mut CUdeviceptr).cast(),
                        (&mut p_out as *mut CUdeviceptr).cast(),
                        (&mut tag as *mut i32).cast(),
                        (&mut rows as *mut i32).cast(),
                        (&mut cols as *mut i32).cast(),
                        (&mut rb as *mut usize).cast(),
                        (&mut nt as *mut i32).cast(),
                    ];
                    // SAFETY: Matches `mmvq_float_f32` signature.
                    unsafe {
                        ctx.launch(
                            "mmvq_float_f32",
                            self.mmvq_float_f32,
                            (grid_x, 1, 1),
                            (32, 4, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }
                other => return Err(CudaError::UnsupportedDType(other)),
            }
            off += b;
        }
        Ok(())
    }

    pub fn launch_embed(
        &self,
        ctx: &CudaContext,
        emb: &CudaMat,
        token_ids_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        n_tok: usize,
        emb_scale: f32,
    ) -> Result<(), CudaError> {
        let mut p_emb = emb.ptr();
        let mut p_ids = token_ids_ptr;
        let mut p_out = out_ptr;
        let mut tag = dtype_cuda_tag(emb.dtype)?;
        let mut n_embd = emb.cols as i32;
        let mut rb = emb.row_bytes;
        let mut scale = emb_scale;
        let mut args: [*mut c_void; 7] = [
            (&mut p_emb as *mut CUdeviceptr).cast(),
            (&mut p_ids as *mut CUdeviceptr).cast(),
            (&mut p_out as *mut CUdeviceptr).cast(),
            (&mut tag as *mut i32).cast(),
            (&mut n_embd as *mut i32).cast(),
            (&mut rb as *mut usize).cast(),
            (&mut scale as *mut f32).cast(),
        ];
        // SAFETY: Matches `embed_tokens_f32`.
        unsafe {
            ctx.launch(
                "embed_tokens_f32",
                self.embed_tokens_f32,
                ((emb.cols as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_rms_norm(
        &self,
        ctx: &CudaContext,
        x_ptr: CUdeviceptr,
        w_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dim: usize,
        n_tok: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let mut p_x = x_ptr;
        let mut p_w = w_ptr;
        let mut p_out = out_ptr;
        let mut c_dim = dim as i32;
        let mut c_eps = eps;
        let mut args: [*mut c_void; 5] = [
            (&mut p_x as *mut CUdeviceptr).cast(),
            (&mut p_w as *mut CUdeviceptr).cast(),
            (&mut p_out as *mut CUdeviceptr).cast(),
            (&mut c_dim as *mut i32).cast(),
            (&mut c_eps as *mut f32).cast(),
        ];
        // SAFETY: Matches `rms_norm_rows_f32`.
        unsafe {
            ctx.launch(
                "rms_norm_rows_f32",
                self.rms_norm_rows_f32,
                (n_tok as u32, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_qk_rms_norm(
        &self,
        ctx: &CudaContext,
        qk_ptr: CUdeviceptr,
        w_ptr: CUdeviceptr,
        n_heads: usize,
        norm_dim: usize,
        n_tok: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let mut p_qk = qk_ptr;
        let mut p_w = w_ptr;
        let mut nh = n_heads as i32;
        let mut nd = norm_dim as i32;
        let mut c_eps = eps;
        let mut args: [*mut c_void; 5] = [
            (&mut p_qk as *mut CUdeviceptr).cast(),
            (&mut p_w as *mut CUdeviceptr).cast(),
            (&mut nh as *mut i32).cast(),
            (&mut nd as *mut i32).cast(),
            (&mut c_eps as *mut f32).cast(),
        ];
        // SAFETY: Matches `qk_rms_norm_inplace_f32`.
        unsafe {
            ctx.launch(
                "qk_rms_norm_inplace_f32",
                self.qk_rms_norm_inplace_f32,
                (n_heads as u32, n_tok as u32, 1),
                (32, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_add_bias(
        &self,
        ctx: &CudaContext,
        x_ptr: CUdeviceptr,
        bias_ptr: CUdeviceptr,
        dim: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let total = (n_tok * dim) as u32;
        let mut p_x = x_ptr;
        let mut p_b = bias_ptr;
        let mut c_dim = dim as i32;
        let mut nt = n_tok as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut p_x as *mut CUdeviceptr).cast(),
            (&mut p_b as *mut CUdeviceptr).cast(),
            (&mut c_dim as *mut i32).cast(),
            (&mut nt as *mut i32).cast(),
        ];
        // SAFETY: Matches `add_bias_rows_f32`.
        unsafe {
            ctx.launch(
                "add_bias_rows_f32",
                self.add_bias_rows_f32,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_add_scaled(
        &self,
        ctx: &CudaContext,
        dst_ptr: CUdeviceptr,
        src_ptr: CUdeviceptr,
        scale: f32,
        n: usize,
    ) -> Result<(), CudaError> {
        let mut p_dst = dst_ptr;
        let mut p_src = src_ptr;
        let mut sc = scale;
        let mut len = n;
        let mut args: [*mut c_void; 4] = [
            (&mut p_dst as *mut CUdeviceptr).cast(),
            (&mut p_src as *mut CUdeviceptr).cast(),
            (&mut sc as *mut f32).cast(),
            (&mut len as *mut usize).cast(),
        ];
        // SAFETY: Matches `add_scaled_inplace_f32`.
        unsafe {
            ctx.launch(
                "add_scaled_inplace_f32",
                self.add_scaled_inplace_f32,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_scale_inplace(
        &self,
        ctx: &CudaContext,
        dst_ptr: CUdeviceptr,
        scale: f32,
        n: usize,
    ) -> Result<(), CudaError> {
        if (scale - 1.0).abs() < 1e-12 {
            return Ok(());
        }
        let mut p_dst = dst_ptr;
        let mut sc = scale;
        let mut len = n;
        let mut args: [*mut c_void; 3] = [
            (&mut p_dst as *mut CUdeviceptr).cast(),
            (&mut sc as *mut f32).cast(),
            (&mut len as *mut usize).cast(),
        ];
        // SAFETY: Matches `scale_inplace_f32`.
        unsafe {
            ctx.launch(
                "scale_inplace_f32",
                self.scale_inplace_f32,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_copy_2d_rows(
        &self,
        ctx: &CudaContext,
        dst_ptr: CUdeviceptr,
        src_ptr: CUdeviceptr,
        cols: usize,
        dst_stride: usize,
        src_stride: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        if cols == 0 || n_tok == 0 {
            return Ok(());
        }
        let mut pd = dst_ptr;
        let mut ps = src_ptr;
        let mut c_cols = cols as i32;
        let mut ds = dst_stride;
        let mut ss = src_stride;
        let mut args: [*mut c_void; 5] = [
            (&mut pd as *mut CUdeviceptr).cast(),
            (&mut ps as *mut CUdeviceptr).cast(),
            (&mut c_cols as *mut i32).cast(),
            (&mut ds as *mut usize).cast(),
            (&mut ss as *mut usize).cast(),
        ];
        // SAFETY: Matches `copy_2d_rows_f32`.
        unsafe {
            ctx.launch(
                "copy_2d_rows_f32",
                self.copy_2d_rows_f32,
                ((cols as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_swiglu(
        &self,
        ctx: &CudaContext,
        gate_ptr: CUdeviceptr,
        up_ptr: CUdeviceptr,
        n: usize,
    ) -> Result<(), CudaError> {
        let mut p_g = gate_ptr;
        let mut p_u = up_ptr;
        let mut len = n;
        let mut args: [*mut c_void; 3] = [
            (&mut p_g as *mut CUdeviceptr).cast(),
            (&mut p_u as *mut CUdeviceptr).cast(),
            (&mut len as *mut usize).cast(),
        ];
        // SAFETY: Matches `swiglu_f32`.
        unsafe {
            ctx.launch(
                "swiglu_f32",
                self.swiglu_f32,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_swiglu_fused(
        &self,
        ctx: &CudaContext,
        gate_up_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        n_ff: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let total = (n_tok * n_ff) as u32;
        let mut p_gu = gate_up_ptr;
        let mut p_out = out_ptr;
        let mut c_nff = n_ff as i32;
        let mut nt = n_tok as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut p_gu as *mut CUdeviceptr).cast(),
            (&mut p_out as *mut CUdeviceptr).cast(),
            (&mut c_nff as *mut i32).cast(),
            (&mut nt as *mut i32).cast(),
        ];
        // SAFETY: Matches `swiglu_fused_f32`.
        unsafe {
            ctx.launch(
                "swiglu_fused_f32",
                self.swiglu_fused_f32,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_sigmoid_gate_mul(
        &self,
        ctx: &CudaContext,
        dst_ptr: CUdeviceptr,
        gate_ptr: CUdeviceptr,
        n: usize,
    ) -> Result<(), CudaError> {
        let mut p_d = dst_ptr;
        let mut p_g = gate_ptr;
        let mut len = n;
        let mut args: [*mut c_void; 3] = [
            (&mut p_d as *mut CUdeviceptr).cast(),
            (&mut p_g as *mut CUdeviceptr).cast(),
            (&mut len as *mut usize).cast(),
        ];
        // SAFETY: Matches `sigmoid_gate_mul_f32`.
        unsafe {
            ctx.launch(
                "sigmoid_gate_mul_f32",
                self.sigmoid_gate_mul_f32,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_rope_qk(
        &self,
        ctx: &CudaContext,
        q_ptr: CUdeviceptr,
        k_ptr: CUdeviceptr,
        inv_freqs_ptr: CUdeviceptr,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        rope_dim: usize,
        rope_type: i32,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        if rope_type < 0 || rope_dim == 0 {
            return Ok(());
        }
        let max_pairs = (n_head.max(n_head_kv) * (rope_dim / 2)) as u32;
        let mut p_q = q_ptr;
        let mut p_k = k_ptr;
        let mut p_f = inv_freqs_ptr;
        let mut nh = n_head as i32;
        let mut nhkv = n_head_kv as i32;
        let mut hd = head_dim as i32;
        let mut rd = rope_dim as i32;
        let mut rt = rope_type;
        let mut sp = start_pos as i32;
        let mut p_pos = pos_dev_ptr;
        let mut args: [*mut c_void; 10] = [
            (&mut p_q as *mut CUdeviceptr).cast(),
            (&mut p_k as *mut CUdeviceptr).cast(),
            (&mut p_f as *mut CUdeviceptr).cast(),
            (&mut nh as *mut i32).cast(),
            (&mut nhkv as *mut i32).cast(),
            (&mut hd as *mut i32).cast(),
            (&mut rd as *mut i32).cast(),
            (&mut rt as *mut i32).cast(),
            (&mut sp as *mut i32).cast(),
            (&mut p_pos as *mut CUdeviceptr).cast(),
        ];
        // SAFETY: Matches `apply_rope_qk_f32`.
        unsafe {
            ctx.launch(
                "apply_rope_qk_f32",
                self.apply_rope_qk_f32,
                (max_pairs.div_ceil(128), n_tok as u32, 1),
                (128, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_kv_f16(
        &self,
        ctx: &CudaContext,
        k_ptr: CUdeviceptr,
        v_ptr: CUdeviceptr,
        kc_ptr: CUdeviceptr,
        vc_ptr: CUdeviceptr,
        kv_dim: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let total = (n_tok * kv_dim) as u32;
        let mut pk = k_ptr;
        let mut pv = v_ptr;
        let mut pkc = kc_ptr;
        let mut pvc = vc_ptr;
        let mut kvd = kv_dim as i32;
        let mut sp = start_pos as i32;
        let mut nt = n_tok as i32;
        let mut p_pos = pos_dev_ptr;
        let mut args: [*mut c_void; 8] = [
            (&mut pk as *mut CUdeviceptr).cast(),
            (&mut pv as *mut CUdeviceptr).cast(),
            (&mut pkc as *mut CUdeviceptr).cast(),
            (&mut pvc as *mut CUdeviceptr).cast(),
            (&mut kvd as *mut i32).cast(),
            (&mut sp as *mut i32).cast(),
            (&mut nt as *mut i32).cast(),
            (&mut p_pos as *mut CUdeviceptr).cast(),
        ];
        // SAFETY: Matches `write_kv_cache_f16`.
        unsafe {
            ctx.launch(
                "write_kv_cache_f16",
                self.write_kv_cache_f16,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_write_kv_q8_0(
        &self,
        ctx: &CudaContext,
        k_ptr: CUdeviceptr,
        v_ptr: CUdeviceptr,
        kc_ptr: CUdeviceptr,
        vc_ptr: CUdeviceptr,
        kv_dim: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut pk = k_ptr;
        let mut pv = v_ptr;
        let mut pkc = kc_ptr;
        let mut pvc = vc_ptr;
        let mut kvd = kv_dim as i32;
        let mut sp = start_pos as i32;
        let mut p_pos = pos_dev_ptr;
        let mut args: [*mut c_void; 7] = [
            (&mut pk as *mut CUdeviceptr).cast(),
            (&mut pv as *mut CUdeviceptr).cast(),
            (&mut pkc as *mut CUdeviceptr).cast(),
            (&mut pvc as *mut CUdeviceptr).cast(),
            (&mut kvd as *mut i32).cast(),
            (&mut sp as *mut i32).cast(),
            (&mut p_pos as *mut CUdeviceptr).cast(),
        ];
        // SAFETY: Matches `write_kv_cache_q8_0`.
        unsafe {
            ctx.launch(
                "write_kv_cache_q8_0",
                self.write_kv_cache_q8_0,
                ((kv_dim / 32) as u32, n_tok as u32, 1),
                (32, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_flash_attn_gqa(
        &self,
        ctx: &CudaContext,
        q_ptr: CUdeviceptr,
        kc_ptr: CUdeviceptr,
        vc_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        is_q8_kv: bool,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        n_tok: usize,
        swa_window: usize,
        attn_scale: f32,
        softcap: f32,
    ) -> Result<(), CudaError> {
        let mut pq = q_ptr;
        let mut pkc = kc_ptr;
        let mut pvc = vc_ptr;
        let mut pout = out_ptr;
        let mut nh = n_head as i32;
        let mut nhkv = n_head_kv as i32;
        let mut hd = head_dim as i32;
        let mut sp = start_pos as i32;
        let mut swa = swa_window as i32;
        let mut sc = attn_scale;
        let mut cap = softcap;
        let mut p_pos = pos_dev_ptr;
        let mut args: [*mut c_void; 12] = [
            (&mut pq as *mut CUdeviceptr).cast(),
            (&mut pkc as *mut CUdeviceptr).cast(),
            (&mut pvc as *mut CUdeviceptr).cast(),
            (&mut pout as *mut CUdeviceptr).cast(),
            (&mut nh as *mut i32).cast(),
            (&mut nhkv as *mut i32).cast(),
            (&mut hd as *mut i32).cast(),
            (&mut sp as *mut i32).cast(),
            (&mut swa as *mut i32).cast(),
            (&mut sc as *mut f32).cast(),
            (&mut cap as *mut f32).cast(),
            (&mut p_pos as *mut CUdeviceptr).cast(),
        ];
        let (func, kname) = if is_q8_kv {
            (self.flash_attn_gqa_q8_0_kv, "flash_attn_gqa_q8_0_kv")
        } else {
            (self.flash_attn_gqa_f16_kv, "flash_attn_gqa_f16_kv")
        };
        // SAFETY: Matches `flash_attn_gqa_{f16,q8_0}_kv`.
        unsafe {
            ctx.launch(
                kname,
                func,
                (n_head as u32, n_tok as u32, 1),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_argmax(
        &self,
        ctx: &CudaContext,
        logits_ptr: CUdeviceptr,
        out_id_ptr: CUdeviceptr,
        vocab_size: usize,
    ) -> Result<(), CudaError> {
        self.launch_argmax_rows(ctx, logits_ptr, out_id_ptr, vocab_size, 1)
    }

    pub fn launch_argmax_rows(
        &self,
        ctx: &CudaContext,
        logits_ptr: CUdeviceptr,
        out_id_ptr: CUdeviceptr,
        vocab_size: usize,
        n_rows: usize,
    ) -> Result<(), CudaError> {
        if n_rows == 0 {
            return Ok(());
        }
        let mut pl = logits_ptr;
        let mut pid = out_id_ptr;
        let mut vs = vocab_size as i32;
        let mut args: [*mut c_void; 3] = [
            (&mut pl as *mut CUdeviceptr).cast(),
            (&mut pid as *mut CUdeviceptr).cast(),
            (&mut vs as *mut i32).cast(),
        ];
        // SAFETY: Matches `argmax_f32` (`Grid: (1, n_rows, 1)`).
        unsafe {
            ctx.launch(
                "argmax_f32",
                self.argmax_f32,
                (1, n_rows as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_mtp_rms_norm_cat2(
        &self,
        ctx: &CudaContext,
        a_ptr: CUdeviceptr,
        wa_ptr: CUdeviceptr,
        b_ptr: CUdeviceptr,
        wb_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dim: usize,
        eps: f32,
        n_rows: usize,
    ) -> Result<(), CudaError> {
        if n_rows == 0 {
            return Ok(());
        }
        let mut pa = a_ptr;
        let mut pwa = wa_ptr;
        let mut pb = b_ptr;
        let mut pwb = wb_ptr;
        let mut pout = out_ptr;
        let mut d = dim as i32;
        let mut e = eps;
        let mut args: [*mut c_void; 7] = [
            (&mut pa as *mut CUdeviceptr).cast(),
            (&mut pwa as *mut CUdeviceptr).cast(),
            (&mut pb as *mut CUdeviceptr).cast(),
            (&mut pwb as *mut CUdeviceptr).cast(),
            (&mut pout as *mut CUdeviceptr).cast(),
            (&mut d as *mut i32).cast(),
            (&mut e as *mut f32).cast(),
        ];
        // SAFETY: Matches `mtp_rms_norm_cat2_f32`.
        unsafe {
            ctx.launch(
                "mtp_rms_norm_cat2_f32",
                self.mtp_rms_norm_cat2_f32,
                (2, n_rows as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_elemwise_mul_add(
        &self,
        ctx: &CudaContext,
        dst_ptr: CUdeviceptr,
        a_ptr: CUdeviceptr,
        b_ptr: CUdeviceptr,
        n: usize,
    ) -> Result<(), CudaError> {
        if n == 0 {
            return Ok(());
        }
        let mut pd = dst_ptr;
        let mut pa = a_ptr;
        let mut pb = b_ptr;
        let mut cn = n;
        let mut args: [*mut c_void; 4] = [
            (&mut pd as *mut CUdeviceptr).cast(),
            (&mut pa as *mut CUdeviceptr).cast(),
            (&mut pb as *mut CUdeviceptr).cast(),
            (&mut cn as *mut usize).cast(),
        ];
        // SAFETY: Matches `elemwise_mul_add_f32`.
        unsafe {
            ctx.launch(
                "elemwise_mul_add_f32",
                self.elemwise_mul_add_f32,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_dfly_bias_sigmoid_slot(
        &self,
        ctx: &CudaContext,
        logit_ptr: CUdeviceptr,
        bias_ptr: CUdeviceptr,
        probs_out_ptr: CUdeviceptr,
        slot: usize,
    ) -> Result<(), CudaError> {
        let mut pl = logit_ptr;
        let mut pb = bias_ptr;
        let mut pp = probs_out_ptr;
        let mut cs = slot as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut pl as *mut CUdeviceptr).cast(),
            (&mut pb as *mut CUdeviceptr).cast(),
            (&mut pp as *mut CUdeviceptr).cast(),
            (&mut cs as *mut i32).cast(),
        ];
        // SAFETY: Matches `dfly_bias_sigmoid_slot_f32`.
        unsafe {
            ctx.launch(
                "dfly_bias_sigmoid_slot_f32",
                self.dfly_bias_sigmoid_slot_f32,
                (1, 1, 1),
                (1, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_scale_rows_by_sigmoid_scalar(
        &self,
        ctx: &CudaContext,
        dst_ptr: CUdeviceptr,
        gate_ptr: CUdeviceptr,
        cols: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut pd = dst_ptr;
        let mut pg = gate_ptr;
        let mut c_cols = cols as i32;
        let mut c_nt = n_tok as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut pd as *mut CUdeviceptr).cast(),
            (&mut pg as *mut CUdeviceptr).cast(),
            (&mut c_cols as *mut i32).cast(),
            (&mut c_nt as *mut i32).cast(),
        ];
        // SAFETY: Matches `scale_rows_by_sigmoid_scalar_f32`.
        unsafe {
            ctx.launch(
                "scale_rows_by_sigmoid_scalar_f32",
                self.scale_rows_by_sigmoid_scalar_f32,
                ((cols as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_moe_router_topk(
        &self,
        ctx: &CudaContext,
        logits_ptr: CUdeviceptr,
        probs_bias_ptr: CUdeviceptr,
        top_ids_ptr: CUdeviceptr,
        top_weights_ptr: CUdeviceptr,
        n_expert: usize,
        k_used: usize,
        gating_code: i32,
        norm_weights: bool,
        norm_eps: f32,
        weights_scale: f32,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut pl = logits_ptr;
        let mut pb = probs_bias_ptr;
        let mut pids = top_ids_ptr;
        let mut pw = top_weights_ptr;
        let mut c_ne = n_expert as i32;
        let mut c_ku = k_used as i32;
        let mut c_gf = gating_code;
        let mut c_nw = i32::from(norm_weights);
        let mut c_eps = norm_eps;
        let mut c_sc = weights_scale;
        let mut args: [*mut c_void; 10] = [
            (&mut pl as *mut CUdeviceptr).cast(),
            (&mut pb as *mut CUdeviceptr).cast(),
            (&mut pids as *mut CUdeviceptr).cast(),
            (&mut pw as *mut CUdeviceptr).cast(),
            (&mut c_ne as *mut i32).cast(),
            (&mut c_ku as *mut i32).cast(),
            (&mut c_gf as *mut i32).cast(),
            (&mut c_nw as *mut i32).cast(),
            (&mut c_eps as *mut f32).cast(),
            (&mut c_sc as *mut f32).cast(),
        ];
        // SAFETY: Matches `moe_router_topk_f32` (`Block: (64, 1, 1)`, up to 512 experts).
        unsafe {
            ctx.launch(
                "moe_router_topk_f32",
                self.moe_router_topk_f32,
                (n_tok as u32, 1, 1),
                (64, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_moe_indexed_gemv_q8_1(
        &self,
        ctx: &CudaContext,
        w_exps_ptr: CUdeviceptr,
        xq_ptr: CUdeviceptr,
        x_f32_ptr: CUdeviceptr,
        top_ids_ptr: CUdeviceptr,
        top_weights_ptr: CUdeviceptr,
        exp_scales_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dtype: i32,
        rows: usize,
        cols: usize,
        row_bytes: usize,
        exp_stride_bytes: usize,
        slot_count: usize,
        n_tok: usize,
        x_mode: i32,
        out_mode: i32,
    ) -> Result<(), CudaError> {
        let mut pw = w_exps_ptr;
        let mut pxq = xq_ptr;
        let mut pxf = x_f32_ptr;
        let mut pids = top_ids_ptr;
        let mut pws = top_weights_ptr;
        let mut psc = exp_scales_ptr;
        let mut pout = out_ptr;
        let mut dt = dtype;
        let mut r = rows as i32;
        let mut c = cols as i32;
        let mut rb = row_bytes;
        let mut es = exp_stride_bytes;
        let mut sc = slot_count as i32;
        let mut xm = x_mode;
        let mut om = out_mode;
        let mut args: [*mut c_void; 15] = [
            (&mut pw as *mut CUdeviceptr).cast(),
            (&mut pxq as *mut CUdeviceptr).cast(),
            (&mut pxf as *mut CUdeviceptr).cast(),
            (&mut pids as *mut CUdeviceptr).cast(),
            (&mut pws as *mut CUdeviceptr).cast(),
            (&mut psc as *mut CUdeviceptr).cast(),
            (&mut pout as *mut CUdeviceptr).cast(),
            (&mut dt as *mut i32).cast(),
            (&mut r as *mut i32).cast(),
            (&mut c as *mut i32).cast(),
            (&mut rb as *mut usize).cast(),
            (&mut es as *mut usize).cast(),
            (&mut sc as *mut i32).cast(),
            (&mut xm as *mut i32).cast(),
            (&mut om as *mut i32).cast(),
        ];
        // SAFETY: Matches `moe_indexed_gemv_q8_1`.
        unsafe {
            ctx.launch(
                "moe_indexed_gemv_q8_1",
                self.moe_indexed_gemv_q8_1,
                ((rows as u32).div_ceil(4), slot_count as u32, n_tok as u32),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_moe_reduce_slots(
        &self,
        ctx: &CudaContext,
        slot_outs_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        rows: usize,
        slot_count: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut ps = slot_outs_ptr;
        let mut po = out_ptr;
        let mut r = rows as i32;
        let mut sc = slot_count as i32;
        let mut nt = n_tok as i32;
        let mut args: [*mut c_void; 5] = [
            (&mut ps as *mut CUdeviceptr).cast(),
            (&mut po as *mut CUdeviceptr).cast(),
            (&mut r as *mut i32).cast(),
            (&mut sc as *mut i32).cast(),
            (&mut nt as *mut i32).cast(),
        ];
        // SAFETY: Matches `moe_reduce_slots_f32`.
        unsafe {
            ctx.launch(
                "moe_reduce_slots_f32",
                self.moe_reduce_slots_f32,
                ((rows as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_moe_indexed_dual_gu_act(
        &self,
        ctx: &CudaContext,
        w_gate_ptr: CUdeviceptr,
        w_up_ptr: CUdeviceptr,
        xq_ptr: CUdeviceptr,
        x_f32_ptr: CUdeviceptr,
        top_ids_ptr: CUdeviceptr,
        gate_scales_ptr: CUdeviceptr,
        up_scales_ptr: CUdeviceptr,
        out_act_ptr: CUdeviceptr,
        dtype: i32,
        n_ff_exp: usize,
        cols: usize,
        row_bytes: usize,
        gate_exp_stride: usize,
        up_exp_stride: usize,
        up_sub_offset_bytes: usize,
        slot_count: usize,
        n_tok: usize,
        act_mode: i32,
        situ_beta: f32,
        situ_lb: f32,
    ) -> Result<(), CudaError> {
        let mut pwg = w_gate_ptr;
        let mut pwu = w_up_ptr;
        let mut pxq = xq_ptr;
        let mut pxf = x_f32_ptr;
        let mut pids = top_ids_ptr;
        let mut pgs = gate_scales_ptr;
        let mut pus = up_scales_ptr;
        let mut pout = out_act_ptr;
        let mut dt = dtype;
        let mut r = n_ff_exp as i32;
        let mut c = cols as i32;
        let mut rb = row_bytes;
        let mut ges = gate_exp_stride;
        let mut ues = up_exp_stride;
        let mut uso = up_sub_offset_bytes;
        let mut sc = slot_count as i32;
        let mut am = act_mode;
        let mut sb = situ_beta;
        let mut sl = situ_lb;
        let mut args: [*mut c_void; 19] = [
            (&mut pwg as *mut CUdeviceptr).cast(),
            (&mut pwu as *mut CUdeviceptr).cast(),
            (&mut pxq as *mut CUdeviceptr).cast(),
            (&mut pxf as *mut CUdeviceptr).cast(),
            (&mut pids as *mut CUdeviceptr).cast(),
            (&mut pgs as *mut CUdeviceptr).cast(),
            (&mut pus as *mut CUdeviceptr).cast(),
            (&mut pout as *mut CUdeviceptr).cast(),
            (&mut dt as *mut i32).cast(),
            (&mut r as *mut i32).cast(),
            (&mut c as *mut i32).cast(),
            (&mut rb as *mut usize).cast(),
            (&mut ges as *mut usize).cast(),
            (&mut ues as *mut usize).cast(),
            (&mut uso as *mut usize).cast(),
            (&mut sc as *mut i32).cast(),
            (&mut am as *mut i32).cast(),
            (&mut sb as *mut f32).cast(),
            (&mut sl as *mut f32).cast(),
        ];
        // SAFETY: Matches `moe_indexed_dual_gu_act_f32`.
        unsafe {
            ctx.launch(
                "moe_indexed_dual_gu_act_f32",
                self.moe_indexed_dual_gu_act_f32,
                (
                    (n_ff_exp as u32).div_ceil(4),
                    slot_count as u32,
                    n_tok as u32,
                ),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn launch_moe_build_expert_buckets(
        &self,
        ctx: &CudaContext,
        top_ids_ptr: CUdeviceptr,
        exp_offs_ptr: CUdeviceptr,
        tile_offs_ptr: CUdeviceptr,
        sorted_ts_ptr: CUdeviceptr,
        n_ent: usize,
        n_expert: usize,
    ) -> Result<(), CudaError> {
        let mut pids = top_ids_ptr;
        let mut peo = exp_offs_ptr;
        let mut pto = tile_offs_ptr;
        let mut pst = sorted_ts_ptr;
        let mut c_ne = n_ent as i32;
        let mut c_nx = n_expert as i32;
        let mut args: [*mut c_void; 6] = [
            (&mut pids as *mut CUdeviceptr).cast(),
            (&mut peo as *mut CUdeviceptr).cast(),
            (&mut pto as *mut CUdeviceptr).cast(),
            (&mut pst as *mut CUdeviceptr).cast(),
            (&mut c_ne as *mut i32).cast(),
            (&mut c_nx as *mut i32).cast(),
        ];
        // SAFETY: Matches `moe_build_expert_buckets_u32`.
        unsafe {
            ctx.launch(
                "moe_build_expert_buckets_u32",
                self.moe_build_expert_buckets_u32,
                (1, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_wmma_moe_grouped(
        &self,
        ctx: &CudaContext,
        w_exps_ptr: CUdeviceptr,
        x_f32_ptr: CUdeviceptr,
        exp_offs_ptr: CUdeviceptr,
        tile_offs_ptr: CUdeviceptr,
        sorted_ts_ptr: CUdeviceptr,
        top_weights_ptr: CUdeviceptr,
        exp_scales_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dtype: i32,
        rows: usize,
        cols: usize,
        row_bytes: usize,
        exp_stride_bytes: usize,
        sub_offset_bytes: usize,
        n_expert: usize,
        slot_count: usize,
        n_tok: usize,
        x_mode: i32,
        out_mode: i32,
    ) -> Result<(), CudaError> {
        let n_ent = n_tok * slot_count;
        let max_tiles = (n_ent.div_ceil(16) + n_ent.min(n_expert)).max(1) as u32;
        let mut pw = w_exps_ptr;
        let mut px = x_f32_ptr;
        let mut peo = exp_offs_ptr;
        let mut pto = tile_offs_ptr;
        let mut pst = sorted_ts_ptr;
        let mut pws = top_weights_ptr;
        let mut psc = exp_scales_ptr;
        let mut pout = out_ptr;
        let mut dt = dtype;
        let mut r = rows as i32;
        let mut c = cols as i32;
        let mut rb = row_bytes;
        let mut es = exp_stride_bytes;
        let mut so = sub_offset_bytes;
        let mut nx = n_expert as i32;
        let mut sc = slot_count as i32;
        let mut xm = x_mode;
        let mut om = out_mode;
        let mut args: [*mut c_void; 18] = [
            (&mut pw as *mut CUdeviceptr).cast(),
            (&mut px as *mut CUdeviceptr).cast(),
            (&mut peo as *mut CUdeviceptr).cast(),
            (&mut pto as *mut CUdeviceptr).cast(),
            (&mut pst as *mut CUdeviceptr).cast(),
            (&mut pws as *mut CUdeviceptr).cast(),
            (&mut psc as *mut CUdeviceptr).cast(),
            (&mut pout as *mut CUdeviceptr).cast(),
            (&mut dt as *mut i32).cast(),
            (&mut r as *mut i32).cast(),
            (&mut c as *mut i32).cast(),
            (&mut rb as *mut usize).cast(),
            (&mut es as *mut usize).cast(),
            (&mut so as *mut usize).cast(),
            (&mut nx as *mut i32).cast(),
            (&mut sc as *mut i32).cast(),
            (&mut xm as *mut i32).cast(),
            (&mut om as *mut i32).cast(),
        ];
        // SAFETY: Matches `wmma_moe_grouped_f32`.
        unsafe {
            ctx.launch(
                "wmma_moe_grouped_f32",
                self.wmma_moe_grouped_f32,
                ((rows as u32).div_ceil(64), max_tiles, 1),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }
}
