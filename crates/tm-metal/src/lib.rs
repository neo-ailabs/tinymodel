//! Apple Metal GPU backend for `tinymodel` (`objc2-metal` + MSL compute kernels).
//!
//! All `unsafe` FFI calls to Metal are contained in this crate with explicit
//! `// SAFETY:` comments, and every compute dispatch checks
//! `maxTotalThreadsPerThreadgroup` to prevent silent Metal no-op execution.

use thiserror::Error;

/// Errors returned by the Metal backend.
#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal backend is only available on Apple platforms")]
    UnsupportedPlatform,
    #[error("no Metal GPU device found")]
    NoDevice,
    #[error("failed to create Metal command queue")]
    CommandQueueFailed,
    #[error("failed to compile Metal shader library: {0}")]
    ShaderCompile(String),
    #[error("Metal function '{0}' not found in compiled library")]
    FunctionNotFound(String),
    #[error("failed to create Metal compute pipeline for '{name}': {reason}")]
    PipelineCreation { name: String, reason: String },
    #[error("failed to allocate Metal buffer of {bytes} bytes")]
    BufferAlloc { bytes: usize },
    #[error("failed to create Metal command buffer or compute encoder")]
    CommandEncodeFailed,
    #[error(
        "threadgroup size {requested} exceeds pipeline maxTotalThreadsPerThreadgroup ({limit}) for '{kernel}'"
    )]
    ThreadgroupTooLarge {
        kernel: &'static str,
        requested: usize,
        limit: usize,
    },
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("GGUF error: {0}")]
    Gguf(#[from] tm_gguf::GgufError),
    #[error("unsupported Metal weight dtype {0:?}")]
    UnsupportedDType(tm_core::DType),
    #[error("unsupported model architecture '{0}'")]
    UnsupportedArch(String),
}

#[cfg(target_vendor = "apple")]
mod apple;
#[cfg(target_vendor = "apple")]
pub mod gemma4_runner;
#[cfg(target_vendor = "apple")]
pub mod lfm2_runner;
#[cfg(target_vendor = "apple")]
pub mod llama_runner;
#[cfg(target_vendor = "apple")]
pub mod qwen35_runner;

#[cfg(target_vendor = "apple")]
pub use apple::{MetalBuffer, MetalContext};
#[cfg(target_vendor = "apple")]
pub use gemma4_runner::{Gemma4MetalConfig, Gemma4MetalRunner};
#[cfg(target_vendor = "apple")]
pub use lfm2_runner::{Lfm2MetalConfig, Lfm2MetalRunner};
#[cfg(target_vendor = "apple")]
pub use llama_runner::{LlamaMetalConfig, LlamaMetalRunner, MetalProfileReport, MetalStageStat};
#[cfg(target_vendor = "apple")]
pub use qwen35_runner::{Qwen35MetalConfig, Qwen35MetalRunner};

/// Unified Apple Metal GPU runner dispatching across all supported M2 architectures
/// (`llama`, `qwen3`, `qwen35`, `gemma4`, `lfm2`).
#[cfg(target_vendor = "apple")]
pub enum MetalRunner {
    Llama(LlamaMetalRunner),
    Qwen35(Qwen35MetalRunner),
    Gemma4(Gemma4MetalRunner),
    Lfm2(Lfm2MetalRunner),
}

#[cfg(target_vendor = "apple")]
impl MetalRunner {
    pub fn from_gguf(
        gguf: std::sync::Arc<tm_gguf::GgufFile>,
        max_ctx: usize,
    ) -> Result<Self, MetalError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        match arch.as_str() {
            "llama" | "qwen3" => Ok(MetalRunner::Llama(LlamaMetalRunner::from_gguf(
                gguf, max_ctx,
            )?)),
            "qwen35" => Ok(MetalRunner::Qwen35(Qwen35MetalRunner::from_gguf(
                gguf, max_ctx,
            )?)),
            "gemma4" => Ok(MetalRunner::Gemma4(Gemma4MetalRunner::from_gguf(
                gguf, max_ctx,
            )?)),
            "lfm2" => Ok(MetalRunner::Lfm2(Lfm2MetalRunner::from_gguf(
                gguf, max_ctx,
            )?)),
            _ => Err(MetalError::UnsupportedArch(arch)),
        }
    }

    pub fn arch(&self) -> &str {
        match self {
            MetalRunner::Llama(r) => &r.cfg.arch,
            MetalRunner::Qwen35(r) => &r.cfg.arch,
            MetalRunner::Gemma4(r) => &r.cfg.arch,
            MetalRunner::Lfm2(r) => &r.cfg.arch,
        }
    }

    pub fn device_name(&self) -> String {
        match self {
            MetalRunner::Llama(r) => r.device_name(),
            MetalRunner::Qwen35(r) => r.device_name(),
            MetalRunner::Gemma4(r) => r.device_name(),
            MetalRunner::Lfm2(r) => r.device_name(),
        }
    }

    pub fn is_zero_copy_weights(&self) -> bool {
        match self {
            MetalRunner::Llama(r) => r.is_zero_copy(),
            MetalRunner::Qwen35(r) => r.is_zero_copy_weights(),
            MetalRunner::Gemma4(r) => r.is_zero_copy_weights(),
            MetalRunner::Lfm2(r) => r.is_zero_copy_weights(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            MetalRunner::Llama(r) => r.cfg.vocab_size,
            MetalRunner::Qwen35(r) => r.cfg.vocab_size,
            MetalRunner::Gemma4(r) => r.cfg.vocab_size,
            MetalRunner::Lfm2(r) => r.cfg.vocab_size,
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            MetalRunner::Llama(r) => r.cfg.n_layers,
            MetalRunner::Qwen35(r) => r.cfg.n_layers,
            MetalRunner::Gemma4(r) => r.cfg.n_layers,
            MetalRunner::Lfm2(r) => r.cfg.n_layers,
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            MetalRunner::Llama(_) => {}
            MetalRunner::Qwen35(r) => r.reset_state(),
            MetalRunner::Gemma4(r) => r.reset_state(),
            MetalRunner::Lfm2(r) => r.reset_state(),
        }
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], MetalError> {
        match self {
            MetalRunner::Llama(r) => r.forward_token(token_id, pos, compute_logits),
            MetalRunner::Qwen35(r) => {
                r.forward_token(token_id, pos, compute_logits)?;
                Ok(r.logits())
            }
            MetalRunner::Gemma4(r) => r.forward_token(token_id, pos, compute_logits),
            MetalRunner::Lfm2(r) => r.forward_token(token_id, pos, compute_logits),
        }
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, MetalError> {
        match self {
            MetalRunner::Llama(r) => r.forward_token_greedy(token_id, pos),
            MetalRunner::Qwen35(r) => r.forward_token_greedy(token_id, pos),
            MetalRunner::Gemma4(r) => r.forward_token_greedy(token_id, pos),
            MetalRunner::Lfm2(r) => r.forward_token_greedy(token_id, pos),
        }
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], MetalError> {
        match self {
            MetalRunner::Llama(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            MetalRunner::Qwen35(r) => {
                r.forward_batch(token_ids, start_pos, compute_last_logits)?;
                Ok(r.logits())
            }
            MetalRunner::Gemma4(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
            MetalRunner::Lfm2(r) => r.forward_batch(token_ids, start_pos, compute_last_logits),
        }
    }

    pub fn profile(
        &mut self,
        n_runs: usize,
        ctx_pos: usize,
    ) -> Result<MetalProfileReport, MetalError> {
        match self {
            MetalRunner::Llama(r) => r.profile(n_runs, ctx_pos),
            MetalRunner::Qwen35(r) => r.profile(n_runs, ctx_pos),
            MetalRunner::Gemma4(r) => r.profile(n_runs, ctx_pos),
            MetalRunner::Lfm2(r) => r.profile(n_runs, ctx_pos),
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
pub struct MetalContext;

#[cfg(not(target_vendor = "apple"))]
impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::UnsupportedPlatform)
    }
}
