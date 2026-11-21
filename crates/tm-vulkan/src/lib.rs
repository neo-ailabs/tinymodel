//! Vulkan GPU backend for `tinymodel` (`ash` + GLSL compute shaders compiled to SPIR-V).
//!
//! One implementation targets AMD (RADV / AMDVLK), Intel (ANV), NVIDIA, Arm Mali (Pixel 8,
//! RK3588), Broadcom V3D (Pi 5) and lavapipe (CPU, used for CI parity tests).
//!
//! All `unsafe` Vulkan FFI calls live in this crate with `// SAFETY:` comments. Every
//! dispatch validates workgroup dimensions against device limits and returns an error
//! instead of silently dropping work.

mod context;
mod kernels;
pub mod llama_runner;
pub mod qwen35_runner;

pub use context::{DeviceInfo, VkBuffer, VkPrefillMode, VulkanContext};
pub use kernels::{Q4Layout, VkMatQ4_0};
pub use llama_runner::{LlamaVkConfig, LlamaVkRunner, VkProfileReport, VkStageStat};
pub use qwen35_runner::{Qwen35VkConfig, Qwen35VkRunner};

use thiserror::Error;

/// Errors returned by the Vulkan backend.
#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("failed to load the Vulkan loader (libvulkan): {0}")]
    Loader(String),
    #[error("Vulkan call {call} failed: {code:?}")]
    Vk {
        call: &'static str,
        code: ash::vk::Result,
    },
    #[error("no Vulkan device with a compute queue found")]
    NoDevice,
    #[error("Vulkan device '{0}' (TM_VK_DEVICE) not found")]
    DeviceNotFound(String),
    #[error("Vulkan device '{device}' lacks required feature: {feature}")]
    MissingFeature {
        device: String,
        feature: &'static str,
    },
    #[error("Vulkan device API version {major}.{minor} is too old (need 1.2)")]
    ApiTooOld { major: u32, minor: u32 },
    #[error("no suitable Vulkan memory type for {0}")]
    NoMemoryType(&'static str),
    #[error("buffer of {bytes} bytes exceeds device limit of {limit} bytes")]
    BufferTooLarge { bytes: u64, limit: u64 },
    #[error(
        "model requires {needed_mib:.1} MiB of Vulkan device memory (max_ctx={max_ctx}), exceeding budget {budget_mib:.1} MiB on '{device}'"
    )]
    OutOfDeviceMemory {
        needed_mib: f64,
        budget_mib: f64,
        max_ctx: usize,
        device: String,
    },
    #[error("workgroup size {requested} exceeds device limit ({limit}) for '{kernel}'")]
    WorkgroupTooLarge {
        kernel: &'static str,
        requested: u32,
        limit: u32,
    },
    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),
    #[error("GGUF error: {0}")]
    Gguf(#[from] tm_gguf::GgufError),
    #[error("unsupported Vulkan weight dtype {0:?}")]
    UnsupportedDType(tm_core::DType),
    #[error("unsupported model architecture '{0}'")]
    UnsupportedArch(String),
}

pub(crate) trait VkResultExt<T> {
    fn ctx(self, call: &'static str) -> Result<T, VulkanError>;
}

impl<T> VkResultExt<T> for Result<T, ash::vk::Result> {
    fn ctx(self, call: &'static str) -> Result<T, VulkanError> {
        self.map_err(|code| VulkanError::Vk { call, code })
    }
}
