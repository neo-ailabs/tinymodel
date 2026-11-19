#![forbid(unsafe_code)]

//! Hand-written per-architecture LLM forward passes with zero-copy GGUF weight views
//! and zero heap allocations on the token generation path.

pub mod llama;
pub mod sampler;

pub use llama::{CpuDecodeProfile, LlamaConfig, LlamaModel, LlamaState, ModelError};
pub use sampler::{Sampler, SamplerConfig, argmax};
