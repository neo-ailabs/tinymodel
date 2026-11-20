#![forbid(unsafe_code)]

//! Hand-written per-architecture LLM forward passes with zero-copy GGUF weight views
//! and zero heap allocations on the token generation path.

pub mod llama;
pub mod qwen35;
pub mod sampler;

pub use llama::{CpuDecodeProfile, LlamaConfig, LlamaModel, LlamaState, ModelError};
pub use qwen35::{LayerState, Qwen35Config, Qwen35DecodeProfile, Qwen35Model, Qwen35State};
pub use sampler::{Sampler, SamplerConfig, argmax};
