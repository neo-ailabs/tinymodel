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
pub mod llama_runner;
#[cfg(target_vendor = "apple")]
pub mod qwen35_runner;

#[cfg(target_vendor = "apple")]
pub use apple::{MetalBuffer, MetalContext};
#[cfg(target_vendor = "apple")]
pub use llama_runner::{LlamaMetalConfig, LlamaMetalRunner, MetalProfileReport, MetalStageStat};
#[cfg(target_vendor = "apple")]
pub use qwen35_runner::{Qwen35MetalConfig, Qwen35MetalRunner};

#[cfg(not(target_vendor = "apple"))]
pub struct MetalContext;

#[cfg(not(target_vendor = "apple"))]
impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError::UnsupportedPlatform)
    }
}
