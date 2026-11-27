//! NVIDIA CUDA GPU backend for `tinymodel` (`libcuda` Driver API + MMVQ `__dp4a` / `wmma` Tensor Core kernels).
//!
//! All `unsafe` CUDA Driver FFI calls live in `driver.rs` and `kernels.rs` with explicit `// SAFETY:`
//! comments (`AGENTS.md` Rule #2). Every kernel launch validates block dimensions and dynamic shared
//! memory against the device's hardware limits before calling `cuLaunchKernel`.

pub mod driver;
pub mod hy4_runner;
pub mod inkling2_runner;
pub mod kernels;
pub mod llama_runner;
pub mod qwen35_runner;
pub mod qwen4exp_runner;

pub use driver::{CUdeviceptr, CudaBuffer, CudaContext, CudaDeviceInfo, CudaEvent};
pub use hy4_runner::Hy4CudaRunner;
pub use inkling2_runner::{DFlashCudaRunner, Inkling2CudaRunner, MAX_INKLING2_CUDA_SPEC_WINDOW};
pub use kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat, MMVQ_MAX_TOK, dtype_cuda_tag};
pub use llama_runner::{CudaProfileReport, CudaStageStat, LlamaCudaRunner};
pub use qwen4exp_runner::Qwen4ExpCudaRunner;
pub use qwen35_runner::Qwen35CudaRunner;

use thiserror::Error;
use tm_core::KvType;

/// Errors raised by the CUDA GPU backend.
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("failed to load NVIDIA CUDA driver (libcuda): {0}")]
    DriverLoad(String),
    #[error("CUDA call '{call}' failed: {detail}")]
    DriverCall { call: &'static str, detail: String },
    #[error("no NVIDIA CUDA GPU device found")]
    NoDevice,
    #[error("CUDA device '{0}' not found")]
    DeviceNotFound(String),
    #[error("CUDA thread block size {requested} exceeds device limit ({limit}) for '{kernel}'")]
    BlockTooLarge {
        kernel: &'static str,
        requested: u32,
        limit: u32,
    },
    #[error("CUDA shared memory {requested} B exceeds device limit ({limit} B) for '{kernel}'")]
    SharedMemTooLarge {
        kernel: &'static str,
        requested: u32,
        limit: u32,
    },
    #[error(
        "model requires {needed_mib:.1} MiB of CUDA VRAM (max_ctx={max_ctx}), exceeding free VRAM {free_mib:.1} MiB on '{device}'"
    )]
    OutOfDeviceMemory {
        needed_mib: f64,
        free_mib: f64,
        max_ctx: usize,
        device: String,
    },
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("GGUF error: {0}")]
    Gguf(#[from] tm_gguf::GgufError),
    #[error("model error: {0}")]
    Model(#[from] tm_models::ModelError),
    #[error("LoRA error: {0}")]
    Lora(#[from] tm_lora::LoraError),
    #[error("unsupported CUDA weight dtype {0:?}")]
    UnsupportedDType(tm_core::DType),
    #[error("unsupported CUDA model architecture '{0}'")]
    UnsupportedArch(String),
}

/// Unified NVIDIA CUDA GPU runner dispatching across supported model families.
#[allow(clippy::large_enum_variant)]
pub enum CudaRunner {
    Llama(LlamaCudaRunner),
    Qwen35(Qwen35CudaRunner),
    Qwen4Exp(Qwen4ExpCudaRunner),
    Inkling2(Inkling2CudaRunner),
    Hy4(Box<Hy4CudaRunner>),
}

impl CudaRunner {
    pub fn from_gguf(gguf: &tm_gguf::GgufFile, max_ctx: usize) -> Result<Self, CudaError> {
        Self::from_gguf_with_kv_type(gguf, max_ctx, KvType::F16)
    }

    pub fn from_strata(
        model_path: &std::path::Path,
        engram_path: Option<&std::path::Path>,
        max_ctx: usize,
    ) -> Result<(Self, String), CudaError> {
        let (r, tok_json) = Hy4CudaRunner::from_strata(model_path, engram_path, max_ctx)?;
        Ok((CudaRunner::Hy4(Box::new(r)), tok_json))
    }

    pub fn from_gguf_with_kv_type(
        gguf: &tm_gguf::GgufFile,
        max_ctx: usize,
        kv_type: KvType,
    ) -> Result<Self, CudaError> {
        let arch = gguf.req_str("general.architecture")?;
        match arch {
            "llama" | "qwen2" | "qwen3" | "smollm3" | "phi3" | "mistral" | "mistral3" | "olmo2"
            | "olmo3" | "granite" | "exaone" | "olmoe" | "granitemoe" | "qwen2moe" | "qwen3moe"
            | "bailingmoe2" => Ok(CudaRunner::Llama(LlamaCudaRunner::from_gguf_with_kv_type(
                gguf, max_ctx, kv_type,
            )?)),
            "qwen35" | "qwen35moe" => Ok(CudaRunner::Qwen35(
                Qwen35CudaRunner::from_gguf_with_kv_type(gguf, max_ctx, kv_type)?,
            )),
            "qwen4exp" => Ok(CudaRunner::Qwen4Exp(
                Qwen4ExpCudaRunner::from_gguf_with_kv_type(gguf, max_ctx, kv_type)?,
            )),
            "inkling2" | "inkling2_loop" => Ok(CudaRunner::Inkling2(
                Inkling2CudaRunner::from_gguf_with_kv_type(gguf, max_ctx, kv_type)?,
            )),
            other => Err(CudaError::UnsupportedArch(other.to_string())),
        }
    }

    pub fn arch(&self) -> &str {
        match self {
            CudaRunner::Llama(r) => &r.cfg.arch,
            CudaRunner::Qwen35(r) => &r.cfg.arch,
            CudaRunner::Qwen4Exp(r) => &r.cfg.arch,
            CudaRunner::Inkling2(r) => &r.cfg.arch,
            CudaRunner::Hy4(r) => &r.cfg.arch,
        }
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        match self {
            CudaRunner::Llama(r) => r.ctx.info(),
            CudaRunner::Qwen35(r) => r.ctx.info(),
            CudaRunner::Qwen4Exp(r) => r.ctx.info(),
            CudaRunner::Inkling2(r) => r.ctx.info(),
            CudaRunner::Hy4(r) => r.ctx.info(),
        }
    }

    pub fn device_name(&self) -> &str {
        &self.device_info().name
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            CudaRunner::Llama(r) => r.cfg.vocab_size,
            CudaRunner::Qwen35(r) => r.cfg.vocab_size,
            CudaRunner::Qwen4Exp(r) => r.cfg.vocab_size,
            CudaRunner::Inkling2(r) => r.cfg.vocab_size,
            CudaRunner::Hy4(r) => r.cfg.vocab_size,
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            CudaRunner::Llama(r) => r.cfg.n_layers,
            CudaRunner::Qwen35(r) => r.cfg.n_layers,
            CudaRunner::Qwen4Exp(r) => r.cfg.n_layers,
            CudaRunner::Inkling2(r) => r.cfg.n_layers,
            CudaRunner::Hy4(r) => r.cfg.n_layers,
        }
    }

    pub fn vram_allocated_bytes(&self) -> usize {
        match self {
            CudaRunner::Llama(r) => r.vram_bytes,
            CudaRunner::Qwen35(r) => r.vram_bytes,
            CudaRunner::Qwen4Exp(r) => r.vram_bytes,
            CudaRunner::Inkling2(r) => r.vram_bytes,
            CudaRunner::Hy4(r) => r.vram_bytes,
        }
    }

    pub fn active_decode_bytes_override(&self) -> Option<u64> {
        match self {
            CudaRunner::Hy4(r) => Some(r.active_decode_bytes as u64),
            _ => None,
        }
    }

    pub fn bos_eos_ids(&self) -> Option<(u32, u32)> {
        match self {
            CudaRunner::Hy4(r) => Some((r.cfg.bos_id, r.cfg.eos_id)),
            _ => None,
        }
    }

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        match self {
            CudaRunner::Llama(r) => r.reset_state(),
            CudaRunner::Qwen35(r) => r.reset_state(),
            CudaRunner::Qwen4Exp(r) => r.reset_state(),
            CudaRunner::Inkling2(r) => r.reset_state(),
            CudaRunner::Hy4(r) => r.reset_state(),
        }
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        match self {
            CudaRunner::Llama(r) => r.forward_token(token_id, pos, compute_logits),
            CudaRunner::Qwen35(r) => r.forward_token(token_id, pos, compute_logits),
            CudaRunner::Qwen4Exp(r) => r.forward_token(token_id, pos, compute_logits),
            CudaRunner::Inkling2(r) => r.forward_token(token_id, pos, compute_logits),
            CudaRunner::Hy4(r) => r.forward_token(token_id, pos, compute_logits),
        }
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, CudaError> {
        match self {
            CudaRunner::Llama(r) => r.forward_token_greedy(token_id, pos),
            CudaRunner::Qwen35(r) => r.forward_token_greedy(token_id, pos),
            CudaRunner::Qwen4Exp(r) => r.forward_token_greedy(token_id, pos),
            CudaRunner::Inkling2(r) => r.forward_token_greedy(token_id, pos),
            CudaRunner::Hy4(r) => r.forward_token_greedy(token_id, pos),
        }
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], CudaError> {
        match self {
            CudaRunner::Llama(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            CudaRunner::Qwen35(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            CudaRunner::Qwen4Exp(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            CudaRunner::Inkling2(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            CudaRunner::Hy4(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
        }
    }

    pub fn forward_embed(
        &mut self,
        embed: &[f32],
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        match self {
            CudaRunner::Llama(r) => r.forward_embed(embed, pos, compute_logits),
            CudaRunner::Qwen35(r) => r.forward_embed(embed, pos, compute_logits),
            CudaRunner::Qwen4Exp(r) => r.forward_embed(embed, pos, compute_logits),
            CudaRunner::Inkling2(r) => r.forward_embed(embed, pos, compute_logits),
            CudaRunner::Hy4(r) => r.forward_embed(embed, pos, compute_logits),
        }
    }

    pub fn profile(
        &mut self,
        n_runs: usize,
        ctx_pos: usize,
    ) -> Result<CudaProfileReport, CudaError> {
        match self {
            CudaRunner::Llama(r) => r.profile(n_runs, ctx_pos),
            CudaRunner::Qwen35(r) => r.profile(n_runs, ctx_pos),
            CudaRunner::Qwen4Exp(r) => r.profile(n_runs, ctx_pos),
            CudaRunner::Inkling2(r) => r.profile(n_runs, ctx_pos),
            CudaRunner::Hy4(r) => r.profile(n_runs, ctx_pos),
        }
    }
}
