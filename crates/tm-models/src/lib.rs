#![forbid(unsafe_code)]

//! Hand-written per-architecture LLM forward passes with zero-copy GGUF weight views
//! and zero heap allocations on the token generation path.

pub mod dflash;
pub mod gemma4;
pub mod hy4;
pub mod inkling2;
pub mod lfm2;
pub mod llama;
pub mod mamba2;
pub mod moe;
pub mod qwen35;
pub mod qwen4exp;
pub mod sampler;
pub mod state;
pub mod vision;

pub use dflash::{DFlashConfig, DFlashModel, DFlashState, MAX_DFLASH_BLOCK_SIZE};
pub use gemma4::{
    Gemma4Config, Gemma4LayerWeights, Gemma4Model, Gemma4State, KvLayerWeights, PleGlobalWeights,
    PleLayerWeights,
};
pub use hy4::{
    HY4_MAGIC, Hy4Config, Hy4EngramCfg, Hy4EngramReader, Hy4Manifest, Hy4Model, Hy4State,
    Hy4TensorInfo, StrataExpertEntry, StrataHeader, StrataProfile, StrataSectionEntry,
    dequant_hy4_row, discover_sibling_engram_strata, is_strata_file,
};
pub use inkling2::{
    BytesBacking, ExpertLayerStore, ExpertWorkSlot, Inkling2Config, Inkling2DecodeProfile,
    Inkling2LayerWeights, Inkling2LoadOptions, Inkling2LoopWeights, Inkling2Model,
    Inkling2PlacementStats, Inkling2State, LOOP_CTRL_COUNT, LOOP_CTRL_FFN, LOOP_CTRL_K,
    LOOP_CTRL_Q, LOOP_CTRL_V, apply_situ_glu, clip_heads_inplace, control_inplace,
    convolve_inplace, mix_depth_into,
};
pub use lfm2::{Lfm2Config, Lfm2LayerWeights, Lfm2Model, Lfm2OpWeights, Lfm2State};
pub use llama::{
    CpuDecodeProfile, LlamaConfig, LlamaLayerWeights, LlamaModel, LlamaState, ModelError,
    RawMatView,
};
pub use mamba2::{
    Mamba2AttnWeights, Mamba2Block, Mamba2Config, Mamba2FfnWeights, Mamba2LayerWeights,
    Mamba2Model, Mamba2SsmWeights, Mamba2State, mamba2_conv1d_silu_split,
    mamba2_gate_and_grouped_norm,
};
pub use moe::{
    ExpertGatingFunc, MOE_NORM_CLAMP_F16_MIN, MoeArchParams, MoeScratch, PackedExperts,
    SharedExpertWeights, SparseMoeConfig, SparseMoeFfn, compute_moe_routing_row,
    eval_sparse_moe_ffn, load_layer_moe_ffn, parse_moe_arch_params,
};
pub use qwen4exp::{
    Qwen4ExpAttnWeights, Qwen4ExpConfig, Qwen4ExpHcWeights, Qwen4ExpLayerWeights, Qwen4ExpModel,
    Qwen4ExpPleWeights, Qwen4ExpState,
};
pub use qwen35::{
    Qwen35AttnWeights, Qwen35Config, Qwen35DecodeProfile, Qwen35FfnWeights, Qwen35LayerWeights,
    Qwen35Model, Qwen35MtpWeights, Qwen35State,
};
pub use sampler::{Sampler, SamplerConfig, argmax};
pub use state::{LayerState, MAX_SPEC_WINDOW, SpecVerifyBuf, decode_embedding_row};
pub use vision::{
    PreprocessedImage, RgbImage, VisionAct, VisionAttnProj, VisionConfig, VisionEmbeddings,
    VisionFfn, VisionLayerWeights, VisionModel, VisionProjector, VisionProjectorType,
    smart_resize_dims,
};
