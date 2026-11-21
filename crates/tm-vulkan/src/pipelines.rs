//! Pre-compiled Vulkan compute pipelines and `#[repr(C)]` push-constant payloads.

use std::sync::Arc;

use tm_core::DType;

use crate::context::{ComputeKernel, VulkanContext, VulkanError};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemvPush {
    pub n_in: u32,
    pub n_out: u32,
    pub row_bytes: u32,
    pub add_residual: u32,
    pub out_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemmPush {
    pub n_in: u32,
    pub n_out: u32,
    pub n_tok: u32,
    pub row_bytes: u32,
    pub add_residual: u32,
    pub out_stride: u32,
    pub out_col_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedPush {
    pub n_embd: u32,
    pub row_bytes: u32,
    pub dtype_code: u32,
    pub n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NormPush {
    pub n: u32,
    pub in_offset: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QkNormPush {
    pub n_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopeKvPush {
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub start_pos: u32,
    pub n_tok: u32,
    pub rope_type: u32,
    pub has_freq_factors: u32,
    pub freq_base: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnPush {
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub start_pos: u32,
    pub n_tok: u32,
    pub scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ElemPush {
    pub total_elems: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArgmaxPush {
    pub vocab_size: u32,
    pub logits_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SplitQgPush {
    pub n_head: u32,
    pub head_dim: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GdnGatesPush {
    pub n_v_heads: u32,
    pub n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GdnConvPush {
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    pub s_dim: u32,
    pub d_conv: u32,
    pub n_tok: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GdnRecurPush {
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    pub s_dim: u32,
    pub n_tok: u32,
    pub eps: f32,
}

/// Complete registry of Vulkan compute pipelines for LLM decode & prefill.
pub struct VulkanPipelines {
    pub gemv_q4_0: ComputeKernel,
    pub gemv_q4_0x4: ComputeKernel,
    pub gemv_q4_1: ComputeKernel,
    pub gemv_q8_0: ComputeKernel,
    pub gemv_q5_k: ComputeKernel,
    pub gemv_q6_k: ComputeKernel,
    pub gemv_f16: ComputeKernel,
    pub gemv_bf16: ComputeKernel,
    pub gemv_f32: ComputeKernel,

    pub gemm_q4_0: ComputeKernel,
    pub gemm_q4_0x4: ComputeKernel,
    pub gemm_q4_1: ComputeKernel,
    pub gemm_q8_0: ComputeKernel,
    pub gemm_q5_k: ComputeKernel,
    pub gemm_q6_k: ComputeKernel,
    pub gemm_f16: ComputeKernel,
    pub gemm_bf16: ComputeKernel,
    pub gemm_f32: ComputeKernel,

    pub embed: ComputeKernel,
    pub rms_norm: ComputeKernel,
    pub qk_norm: ComputeKernel,
    pub rope_kv: ComputeKernel,
    pub gqa_attn: ComputeKernel,
    pub swiglu: ComputeKernel,
    pub add_inplace: ComputeKernel,
    pub argmax: ComputeKernel,
    pub split_qg_norm: ComputeKernel,
    pub sigmoid_gate: ComputeKernel,
    pub gdn_gates: ComputeKernel,
    pub gdn_conv_silu_l2: ComputeKernel,
    pub gdn_recur_gated: ComputeKernel,
}

impl VulkanPipelines {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self, VulkanError> {
        let gv_pc = core::mem::size_of::<GemvPush>() as u32;
        let gm_pc = core::mem::size_of::<GemmPush>() as u32;

        Ok(Self {
            gemv_q4_0: ComputeKernel::from_spirv(ctx, shaders::GEMV_Q4_0, 3, gv_pc, [32, 4, 1])?,
            gemv_q4_0x4: ComputeKernel::from_spirv(
                ctx,
                shaders::GEMV_Q4_0X4,
                3,
                gv_pc,
                [32, 4, 1],
            )?,
            gemv_q4_1: ComputeKernel::from_spirv(ctx, shaders::GEMV_Q4_1, 3, gv_pc, [32, 4, 1])?,
            gemv_q8_0: ComputeKernel::from_spirv(ctx, shaders::GEMV_Q8_0, 3, gv_pc, [32, 4, 1])?,
            gemv_q5_k: ComputeKernel::from_spirv(ctx, shaders::GEMV_Q5_K, 3, gv_pc, [32, 4, 1])?,
            gemv_q6_k: ComputeKernel::from_spirv(ctx, shaders::GEMV_Q6_K, 3, gv_pc, [32, 4, 1])?,
            gemv_f16: ComputeKernel::from_spirv(ctx, shaders::GEMV_F16, 3, gv_pc, [32, 4, 1])?,
            gemv_bf16: ComputeKernel::from_spirv(ctx, shaders::GEMV_BF16, 3, gv_pc, [32, 4, 1])?,
            gemv_f32: ComputeKernel::from_spirv(ctx, shaders::GEMV_F32, 3, gv_pc, [32, 4, 1])?,

            gemm_q4_0: ComputeKernel::from_spirv(ctx, shaders::GEMM_Q4_0, 3, gm_pc, [32, 4, 1])?,
            gemm_q4_0x4: ComputeKernel::from_spirv(
                ctx,
                shaders::GEMM_Q4_0X4,
                3,
                gm_pc,
                [32, 4, 1],
            )?,
            gemm_q4_1: ComputeKernel::from_spirv(ctx, shaders::GEMM_Q4_1, 3, gm_pc, [32, 4, 1])?,
            gemm_q8_0: ComputeKernel::from_spirv(ctx, shaders::GEMM_Q8_0, 3, gm_pc, [32, 4, 1])?,
            gemm_q5_k: ComputeKernel::from_spirv(ctx, shaders::GEMM_Q5_K, 3, gm_pc, [32, 4, 1])?,
            gemm_q6_k: ComputeKernel::from_spirv(ctx, shaders::GEMM_Q6_K, 3, gm_pc, [32, 4, 1])?,
            gemm_f16: ComputeKernel::from_spirv(ctx, shaders::GEMM_F16, 3, gm_pc, [32, 4, 1])?,
            gemm_bf16: ComputeKernel::from_spirv(ctx, shaders::GEMM_BF16, 3, gm_pc, [32, 4, 1])?,
            gemm_f32: ComputeKernel::from_spirv(ctx, shaders::GEMM_F32, 3, gm_pc, [32, 4, 1])?,

            embed: ComputeKernel::from_spirv(
                ctx,
                shaders::EMBED,
                3,
                core::mem::size_of::<EmbedPush>() as u32,
                [64, 1, 1],
            )?,
            rms_norm: ComputeKernel::from_spirv(
                ctx,
                shaders::RMS_NORM,
                3,
                core::mem::size_of::<NormPush>() as u32,
                [256, 1, 1],
            )?,
            qk_norm: ComputeKernel::from_spirv(
                ctx,
                shaders::QK_NORM,
                2,
                core::mem::size_of::<QkNormPush>() as u32,
                [64, 1, 1],
            )?,
            rope_kv: ComputeKernel::from_spirv(
                ctx,
                shaders::ROPE_KV,
                6,
                core::mem::size_of::<RopeKvPush>() as u32,
                [64, 1, 1],
            )?,
            gqa_attn: ComputeKernel::from_spirv(
                ctx,
                shaders::GQA_ATTN,
                4,
                core::mem::size_of::<AttnPush>() as u32,
                [64, 1, 1],
            )?,
            swiglu: ComputeKernel::from_spirv(
                ctx,
                shaders::SWIGLU,
                2,
                core::mem::size_of::<ElemPush>() as u32,
                [256, 1, 1],
            )?,
            add_inplace: ComputeKernel::from_spirv(
                ctx,
                shaders::ADD_INPLACE,
                2,
                core::mem::size_of::<ElemPush>() as u32,
                [256, 1, 1],
            )?,
            argmax: ComputeKernel::from_spirv(
                ctx,
                shaders::ARGMAX,
                2,
                core::mem::size_of::<ArgmaxPush>() as u32,
                [256, 1, 1],
            )?,
            split_qg_norm: ComputeKernel::from_spirv(
                ctx,
                shaders::SPLIT_QG_NORM,
                4,
                core::mem::size_of::<SplitQgPush>() as u32,
                [64, 1, 1],
            )?,
            sigmoid_gate: ComputeKernel::from_spirv(
                ctx,
                shaders::SIGMOID_GATE,
                2,
                core::mem::size_of::<ElemPush>() as u32,
                [256, 1, 1],
            )?,
            gdn_gates: ComputeKernel::from_spirv(
                ctx,
                shaders::GDN_GATES,
                5,
                core::mem::size_of::<GdnGatesPush>() as u32,
                [64, 1, 1],
            )?,
            gdn_conv_silu_l2: ComputeKernel::from_spirv(
                ctx,
                shaders::GDN_CONV_SILU_L2,
                6,
                core::mem::size_of::<GdnConvPush>() as u32,
                [128, 1, 1],
            )?,
            gdn_recur_gated: ComputeKernel::from_spirv(
                ctx,
                shaders::GDN_RECUR_GATED,
                9,
                core::mem::size_of::<GdnRecurPush>() as u32,
                [128, 1, 1],
            )?,
        })
    }

    pub fn gemv_for_dtype<'a>(
        &'a self,
        dtype: DType,
        tensor: &str,
    ) -> Result<&'a ComputeKernel, VulkanError> {
        match dtype {
            DType::Q4_0 => Ok(&self.gemv_q4_0),
            DType::Q4_0x4 => Ok(&self.gemv_q4_0x4),
            DType::Q4_1 => Ok(&self.gemv_q4_1),
            DType::Q8_0 => Ok(&self.gemv_q8_0),
            DType::Q5K => Ok(&self.gemv_q5_k),
            DType::Q6K => Ok(&self.gemv_q6_k),
            DType::F16 => Ok(&self.gemv_f16),
            DType::BF16 => Ok(&self.gemv_bf16),
            DType::F32 => Ok(&self.gemv_f32),
            other => Err(VulkanError::UnsupportedDType {
                tensor: tensor.to_string(),
                dtype: other,
            }),
        }
    }

    pub fn gemm_for_dtype<'a>(
        &'a self,
        dtype: DType,
        tensor: &str,
    ) -> Result<&'a ComputeKernel, VulkanError> {
        match dtype {
            DType::Q4_0 => Ok(&self.gemm_q4_0),
            DType::Q4_0x4 => Ok(&self.gemm_q4_0x4),
            DType::Q4_1 => Ok(&self.gemm_q4_1),
            DType::Q8_0 => Ok(&self.gemm_q8_0),
            DType::Q5K => Ok(&self.gemm_q5_k),
            DType::Q6K => Ok(&self.gemm_q6_k),
            DType::F16 => Ok(&self.gemm_f16),
            DType::BF16 => Ok(&self.gemm_bf16),
            DType::F32 => Ok(&self.gemm_f32),
            other => Err(VulkanError::UnsupportedDType {
                tensor: tensor.to_string(),
                dtype: other,
            }),
        }
    }
}
