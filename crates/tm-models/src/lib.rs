#![forbid(unsafe_code)]

//! Hand-written per-architecture LLM forward passes with zero-copy GGUF weight views
//! and zero heap allocations on the token generation path.

pub mod dflash;
pub mod gemma4;
pub mod inkling2;
pub mod lfm2;
pub mod llama;
pub mod qwen35;
pub mod sampler;
pub mod state;

pub use dflash::{DFlashConfig, DFlashModel, DFlashState, MAX_DFLASH_BLOCK_SIZE};
pub use gemma4::{Gemma4Config, Gemma4Model, Gemma4State};
pub use inkling2::{
    BytesBacking, ExpertLayerStore, ExpertWorkSlot, Inkling2Config, Inkling2LayerWeights,
    Inkling2LoadOptions, Inkling2LoopWeights, Inkling2Model, Inkling2PlacementStats, Inkling2State,
    LOOP_CTRL_COUNT, LOOP_CTRL_FFN, LOOP_CTRL_K, LOOP_CTRL_Q, LOOP_CTRL_V, apply_situ_glu,
    clip_heads_inplace, control_inplace, convolve_inplace, mix_depth_into,
};
pub use lfm2::{Lfm2Config, Lfm2Model, Lfm2State};
pub use llama::{CpuDecodeProfile, LlamaConfig, LlamaModel, LlamaState, ModelError};
pub use qwen35::{Qwen35Config, Qwen35DecodeProfile, Qwen35Model, Qwen35State};
pub use sampler::{Sampler, SamplerConfig, argmax};
pub use state::{LayerState, MAX_SPEC_WINDOW, SpecVerifyBuf, decode_embedding_row};
