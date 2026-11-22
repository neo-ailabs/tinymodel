#![forbid(unsafe_code)]

//! Hand-written per-architecture LLM forward passes with zero-copy GGUF weight views
//! and zero heap allocations on the token generation path.

pub mod gemma4;
pub mod lfm2;
pub mod llama;
pub mod qwen35;
pub mod sampler;
pub mod state;

pub use gemma4::{Gemma4Config, Gemma4Model, Gemma4State};
pub use lfm2::{Lfm2Config, Lfm2Model, Lfm2State};
pub use llama::{
    CpuDecodeProfile, LlamaConfig, LlamaLayerWeights, LlamaModel, LlamaState, ModelError,
    RawMatView,
};
pub use qwen35::{Qwen35Config, Qwen35DecodeProfile, Qwen35Model, Qwen35State};
pub use sampler::{Sampler, SamplerConfig, argmax};
pub use state::{LayerState, MAX_SPEC_WINDOW, SpecVerifyBuf, decode_embedding_row};
