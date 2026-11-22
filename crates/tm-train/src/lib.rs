#![forbid(unsafe_code)]

//! On-device parameter-efficient fine-tuning (`ADR-015`):
//! - **Phase `T2a` (`mtp_trainer`)**: Trains a `LinearMtpHead` speculative draft cell
//!   (`general.type = "mtp"`) on top of any frozen supported `CpuModel` (`Llama`, `Qwen3`,
//!   `Qwen3.5`, `Gemma 4`, `LFM2`).
//! - **Phase `T2b` (`qlora`)**: Trains a `llama.cpp`-compatible GGUF LoRA adapter (`general.type = "adapter"`)
//!   directly on frozen quantized `Llama / Qwen3 / SmolLM2` GGUF base models using `FP16` per-layer
//!   activation checkpointing, masked chunked cross-entropy (`label == -100` skip), zero-copy
//!   transposed quantized matmul (`matmul_q_transposed_f32`), AdamW / LoRA+, and crash-safe checkpoints.

pub mod dataset;
pub mod mtp_trainer;
pub mod optim;
pub mod qlora;

use thiserror::Error;
use tm_gguf::GgufError;
use tm_lora::LoraError;
use tm_models::ModelError;
use tm_tokenizer::TokenizerError;

pub use dataset::{TrainDataset, TrainSample, write_tmsft_file};
pub use mtp_trainer::{MtpTrainConfig, MtpTrainReport, train_mtp_head};
pub use optim::{
    AdamW, AdamWConfig, LoraGradStore, LoraTargets, SplitMix64, init_lora_adapter,
    load_train_checkpoint, lr_at_step, save_train_checkpoint,
};
pub use qlora::{
    CheckpointMode, LlamaTrainer, QloraConfig, QloraStepMetrics, QloraTrainReport,
    run_qlora_training,
};

#[derive(Debug, Error)]
pub enum TrainError {
    #[error(transparent)]
    Gguf(Box<GgufError>),
    #[error(transparent)]
    Lora(Box<LoraError>),
    #[error(transparent)]
    Model(Box<ModelError>),
    #[error(transparent)]
    Tokenizer(Box<TokenizerError>),
    #[error("I/O error on '{path}': {message}")]
    Io { path: String, message: String },
    #[error("dataset '{path}' error: {detail}")]
    Dataset { path: String, detail: String },
    #[error("invalid training configuration: {0}")]
    InvalidConfig(String),
    #[error(
        "insufficient RAM for training: need {required_mib:.1} MiB dirty RAM, only {available_mib:.1} MiB safely available{note}"
    )]
    OutOfMemoryBudget {
        required_mib: f64,
        available_mib: f64,
        note: String,
    },
    #[error("checkpoint '{path}' error: {detail}")]
    Checkpoint { path: String, detail: String },
}

impl From<GgufError> for TrainError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(Box::new(e))
    }
}

impl From<LoraError> for TrainError {
    fn from(e: LoraError) -> Self {
        Self::Lora(Box::new(e))
    }
}

impl From<ModelError> for TrainError {
    fn from(e: ModelError) -> Self {
        Self::Model(Box::new(e))
    }
}

impl From<TokenizerError> for TrainError {
    fn from(e: TokenizerError) -> Self {
        Self::Tokenizer(Box::new(e))
    }
}
