//! Cross-platform Vulkan 1.2/1.3 compute backend (`docs/ARCHITECTURE.md` §5.4 & `ADR-008`).
//!
//! Dynamically loads `libvulkan` at runtime (`ash`), compiles all GLSL 450 compute shaders
//! to SPIR-V at build time (`naga`), detects `SubgroupProperties`, `VK_KHR_cooperative_matrix`,
//! and unified-memory (`UMA`) heaps, and executes full `llama` / `qwen3` / `qwen35` forward passes
//! inside a single `VkCommandBuffer` per decode step.

pub mod context;
pub mod llama_runner;
pub mod pipelines;
pub mod qwen35_runner;

use tm_gguf::GgufFile;

pub use context::{
    ComputeKernel, DescriptorArena, GpuBuffer, VulkanContext, VulkanDeviceCaps, VulkanError,
};
pub use llama_runner::{LlamaVulkanConfig, LlamaVulkanRunner};
pub use pipelines::VulkanPipelines;
pub use qwen35_runner::{Qwen35VulkanConfig, Qwen35VulkanRunner};

/// Probes the runtime Vulkan device and compiles all 31 compute pipelines as a smoke check.
pub fn probe_vulkan_device() -> Result<VulkanDeviceCaps, VulkanError> {
    let ctx = VulkanContext::new()?;
    let _pipes = VulkanPipelines::new(&ctx)?;
    Ok(ctx.caps.clone())
}

/// Unified Vulkan GPU runner dispatching across `llama`, `qwen3`, and `qwen35`.
pub enum VulkanRunner {
    Llama(Box<LlamaVulkanRunner>),
    Qwen35(Box<Qwen35VulkanRunner>),
}

impl VulkanRunner {
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, VulkanError> {
        let arch = gguf.req_str("general.architecture")?;
        match arch {
            "llama" | "qwen3" => Ok(VulkanRunner::Llama(Box::new(LlamaVulkanRunner::from_gguf(
                gguf, max_ctx,
            )?))),
            "qwen35" => Ok(VulkanRunner::Qwen35(Box::new(
                Qwen35VulkanRunner::from_gguf(gguf, max_ctx)?,
            ))),
            other => Err(VulkanError::UnsupportedArch(other.to_string())),
        }
    }

    pub fn arch(&self) -> &str {
        match self {
            VulkanRunner::Llama(r) => &r.cfg.arch,
            VulkanRunner::Qwen35(r) => &r.cfg.arch,
        }
    }

    pub fn device_name(&self) -> &str {
        match self {
            VulkanRunner::Llama(r) => &r.ctx.caps.device_name,
            VulkanRunner::Qwen35(r) => &r.ctx.caps.device_name,
        }
    }

    pub fn caps(&self) -> &VulkanDeviceCaps {
        match self {
            VulkanRunner::Llama(r) => &r.ctx.caps,
            VulkanRunner::Qwen35(r) => &r.ctx.caps,
        }
    }

    pub fn is_uma(&self) -> bool {
        self.caps().is_uma
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            VulkanRunner::Llama(r) => r.cfg.vocab_size,
            VulkanRunner::Qwen35(r) => r.cfg.vocab_size,
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            VulkanRunner::Llama(r) => r.cfg.n_layers,
            VulkanRunner::Qwen35(r) => r.cfg.n_layers,
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            VulkanRunner::Llama(r) => r.reset_state(),
            VulkanRunner::Qwen35(r) => r.reset_state(),
        }
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        match self {
            VulkanRunner::Llama(r) => r.forward_token(token_id, pos, compute_logits),
            VulkanRunner::Qwen35(r) => r.forward_token(token_id, pos, compute_logits),
        }
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, VulkanError> {
        match self {
            VulkanRunner::Llama(r) => r.forward_token_greedy(token_id, pos),
            VulkanRunner::Qwen35(r) => r.forward_token_greedy(token_id, pos),
        }
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        match self {
            VulkanRunner::Llama(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            VulkanRunner::Qwen35(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
        }
    }
}

/// Embedded build-time `naga`-compiled SPIR-V shader modules.
pub mod shaders {
    pub const GEMV_Q4_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_0.spv"));
    pub const GEMV_Q4_0X4: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_0x4.spv"));
    pub const GEMV_Q4_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_1.spv"));
    pub const GEMV_Q8_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q8_0.spv"));
    pub const GEMV_Q5_K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q5_k.spv"));
    pub const GEMV_Q6_K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q6_k.spv"));
    pub const GEMV_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_f16.spv"));
    pub const GEMV_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_bf16.spv"));
    pub const GEMV_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_f32.spv"));

    pub const GEMM_Q4_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q4_0.spv"));
    pub const GEMM_Q4_0X4: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q4_0x4.spv"));
    pub const GEMM_Q4_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q4_1.spv"));
    pub const GEMM_Q8_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q8_0.spv"));
    pub const GEMM_Q5_K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q5_k.spv"));
    pub const GEMM_Q6_K: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q6_k.spv"));
    pub const GEMM_F16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f16.spv"));
    pub const GEMM_BF16: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_bf16.spv"));
    pub const GEMM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f32.spv"));

    pub const EMBED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv"));
    pub const RMS_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv"));
    pub const QK_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm.spv"));
    pub const ROPE_KV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_kv.spv"));
    pub const GQA_ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gqa_attn.spv"));
    pub const GQA_ATTN_BATCH: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/gqa_attn_batch.spv"));
    pub const SWIGLU: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv"));
    pub const ADD_INPLACE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_inplace.spv"));
    pub const ARGMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
    pub const SPLIT_QG_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/split_qg_norm.spv"));
    pub const SIGMOID_GATE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sigmoid_gate.spv"));
    pub const GDN_GATES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_gates.spv"));
    pub const GDN_CONV_SILU_L2: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/gdn_conv_silu_l2.spv"));
    pub const GDN_RECUR_GATED: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/gdn_recur_gated.spv"));
}
