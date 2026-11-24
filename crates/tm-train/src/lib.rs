#![forbid(unsafe_code)]

//! On-device parameter-efficient fine-tuning (`ADR-015` Phases `T2a`, `T2b`, `T3`, `T4`):
//! - **Phase `T2a` (`mtp_trainer`)**: Trains a `LinearMtpHead` speculative draft cell
//!   (`general.type = "mtp"`) on top of any frozen supported `CpuModel` (`Llama`, `Qwen3`,
//!   `Qwen3.5`, `Gemma 4`, `LFM2`).
//! - **Phases `T2b + T3 + T4` (`qlora`, `qwen35_qlora`, `gemma4_qlora`, `lfm2_qlora`, `accel`)**:
//!   Trains `llama.cpp`-compatible GGUF LoRA adapters (`general.type = "adapter"`) across ALL 5
//!   model architectures (`Llama`, `Qwen3`, `Qwen3.5`, `Gemma 4`, `LFM2`) directly on frozen
//!   quantized GGUF weights, with optional Apple Metal & Vulkan GPU linear acceleration (`LinearAccel`).

pub mod accel;
pub mod dataset;
pub mod gemma4_qlora;
pub mod lfm2_qlora;
pub mod mamba2_qlora;
pub mod mtp_trainer;
pub mod optim;
pub mod qlora;
pub mod qwen35_qlora;

use std::time::Instant;

use thiserror::Error;
use tm_cpu::ThreadPool;
use tm_gguf::GgufError;
use tm_lora::{LoraAdapter, LoraError};
use tm_models::ModelError;
use tm_runtime::CpuModel;
use tm_tokenizer::TokenizerError;

pub use accel::LinearAccel;
pub use dataset::{TrainDataset, TrainSample, write_tmsft_file};
pub use gemma4_qlora::{Gemma4Trainer, estimate_gemma4_qlora_dirty_bytes};
pub use lfm2_qlora::{Lfm2Trainer, estimate_lfm2_qlora_dirty_bytes};
pub use mamba2_qlora::{Mamba2Trainer, estimate_mamba2_qlora_dirty_bytes};
pub use mtp_trainer::{MtpTrainConfig, MtpTrainReport, train_mtp_head};
pub use optim::{
    AdamW, AdamWConfig, LoraGradStore, LoraTargets, SplitMix64, init_lora_adapter,
    init_lora_adapter_for_cpu_model, init_lora_adapter_gemma4, init_lora_adapter_lfm2,
    init_lora_adapter_mamba2, init_lora_adapter_qwen35, load_train_checkpoint, lr_at_step,
    save_train_checkpoint,
};
pub use qlora::{
    CheckpointMode, LlamaTrainer, QloraConfig, QloraStepMetrics, QloraTrainReport,
    check_training_memory_budget, estimate_qlora_dirty_bytes, run_qlora_training,
};
pub use qwen35_qlora::{Qwen35Trainer, estimate_qwen35_qlora_dirty_bytes};

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
    #[error("GPU training accelerator error: {0}")]
    Accelerator(String),
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

/// Estimates anonymous dirty-page RAM required to QLoRA fine-tune any supported [`CpuModel`].
pub fn estimate_cpu_qlora_dirty_bytes(
    model: &CpuModel<'_>,
    max_seq: usize,
    rank: usize,
    targets: &LoraTargets,
    ckpt_mode: CheckpointMode,
) -> u64 {
    match model {
        CpuModel::Llama(m) => estimate_qlora_dirty_bytes(m, max_seq, rank, targets, ckpt_mode),
        CpuModel::Qwen35(m) => {
            estimate_qwen35_qlora_dirty_bytes(m, max_seq, rank, targets, ckpt_mode)
        }
        CpuModel::Gemma4(m) => {
            estimate_gemma4_qlora_dirty_bytes(m, max_seq, rank, targets, ckpt_mode)
        }
        CpuModel::Lfm2(m) => estimate_lfm2_qlora_dirty_bytes(m, max_seq, rank, targets, ckpt_mode),
        CpuModel::Mamba2(m) => {
            estimate_mamba2_qlora_dirty_bytes(m, max_seq, rank, targets, ckpt_mode)
        }
        CpuModel::Inkling2(m) => {
            let c = &m.cfg;
            ((max_seq * c.n_embd * 16 + c.vocab_padded * 2) * 4) as u64
        }
    }
}

enum PolyTrainer<'a> {
    Llama(Box<LlamaTrainer<'a>>),
    Qwen35(Box<Qwen35Trainer<'a>>),
    Gemma4(Box<Gemma4Trainer<'a>>),
    Lfm2(Box<Lfm2Trainer<'a>>),
    Mamba2(Box<Mamba2Trainer<'a>>),
}

impl<'a> PolyTrainer<'a> {
    fn new(
        model: &'a CpuModel<'a>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Result<Self, TrainError> {
        match model {
            CpuModel::Llama(m) => Ok(Self::Llama(Box::new(LlamaTrainer::new(
                m, max_seq, max_rank, ckpt_mode,
            )))),
            CpuModel::Qwen35(m) => Ok(Self::Qwen35(Box::new(Qwen35Trainer::new(
                m, max_seq, max_rank, ckpt_mode,
            )))),
            CpuModel::Gemma4(m) => Ok(Self::Gemma4(Box::new(Gemma4Trainer::new(
                m, max_seq, max_rank, ckpt_mode,
            )))),
            CpuModel::Lfm2(m) => Ok(Self::Lfm2(Box::new(Lfm2Trainer::new(
                m, max_seq, max_rank, ckpt_mode,
            )))),
            CpuModel::Mamba2(m) => Ok(Self::Mamba2(Box::new(Mamba2Trainer::new(
                m, max_seq, max_rank, ckpt_mode,
            )))),
            CpuModel::Inkling2(m) => Err(TrainError::InvalidConfig(format!(
                "QLoRA fine-tuning is not yet supported on architecture '{}'",
                m.cfg.arch
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_microbatch(
        &mut self,
        accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        adapter: &LoraAdapter,
        grads: &mut LoraGradStore,
        tokens: &[u32],
        loss_mask: &[u8],
        grad_scale: f32,
    ) -> (f32, usize) {
        match self {
            Self::Llama(t) => t.forward_backward_microbatch_with_accel(
                accel, pool, adapter, grads, tokens, loss_mask, grad_scale,
            ),
            Self::Qwen35(t) => t.forward_backward_microbatch(
                accel, pool, adapter, grads, tokens, loss_mask, grad_scale,
            ),
            Self::Gemma4(t) => t.forward_backward_microbatch(
                accel, pool, adapter, grads, tokens, loss_mask, grad_scale,
            ),
            Self::Lfm2(t) => t.forward_backward_microbatch(
                accel, pool, adapter, grads, tokens, loss_mask, grad_scale,
            ),
            Self::Mamba2(t) => t.forward_backward_microbatch(
                accel, pool, adapter, grads, tokens, loss_mask, grad_scale,
            ),
        }
    }
}

/// Polymorphic QLoRA fine-tuning loop across ALL 5 architectures (`llama`, `qwen3`, `qwen35`,
/// `gemma4`, `lfm2`) with optional Metal / Vulkan GPU linear acceleration (`accel`).
pub fn run_cpu_qlora_training<F>(
    model: &CpuModel<'_>,
    pool: &ThreadPool,
    dataset: &TrainDataset,
    cfg: &QloraConfig,
    mut accel: Option<&mut (dyn LinearAccel + '_)>,
    mut on_step: F,
) -> Result<(LoraAdapter, QloraTrainReport), TrainError>
where
    F: FnMut(&QloraStepMetrics) -> Result<(), TrainError>,
{
    if cfg.steps == 0 {
        return Err(TrainError::InvalidConfig("steps must be >= 1".to_string()));
    }
    let grad_accum = cfg.grad_accum.max(1);
    let max_seq = cfg.seq_len.max(2);

    let est_bytes =
        estimate_cpu_qlora_dirty_bytes(model, max_seq, cfg.rank, &cfg.targets, cfg.checkpoint_mode);
    check_training_memory_budget(est_bytes)?;

    let (mut adapter, mut opt) = if let Some(ref resume_p) = cfg.resume_ckpt {
        load_train_checkpoint(resume_p, cfg.optim)?
    } else {
        let ad =
            init_lora_adapter_for_cpu_model(model, cfg.rank, cfg.alpha, &cfg.targets, cfg.seed)?;
        let op = AdamW::new(&ad, cfg.optim);
        (ad, op)
    };

    let max_rank = adapter
        .pairs
        .values()
        .map(|p| p.rank)
        .max()
        .unwrap_or(cfg.rank);
    let mut grads = LoraGradStore::from_adapter(&adapter);
    let trainable_params = grads.total_trainable_params();
    let mut trainer = PolyTrainer::new(model, max_seq, max_rank, cfg.checkpoint_mode)?;

    let t_total = Instant::now();
    let start_step = opt.step as usize;
    let mut sample_cursor = start_step * grad_accum;

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let mut min_loss = f32::INFINITY;
    let mut total_toks = 0usize;
    let mut sup_toks = 0usize;

    let end_step = cfg.stop_at_step.unwrap_or(cfg.steps).min(cfg.steps);
    for step in start_step..end_step {
        let t_step = Instant::now();
        grads.zero_grad();
        let micro_scale = 1.0f32 / (grad_accum as f32);

        let mut step_loss_sum = 0.0f64;
        let mut step_sup = 0usize;
        let mut step_toks = 0usize;

        for _ in 0..grad_accum {
            let sample = &dataset.samples[sample_cursor % dataset.samples.len()];
            sample_cursor += 1;
            let keep = sample.tokens.len().min(max_seq + 1);
            let (l, n_sup) = trainer.step_microbatch(
                accel.as_deref_mut(),
                pool,
                &adapter,
                &mut grads,
                &sample.tokens[..keep],
                &sample.loss_mask[..keep],
                micro_scale,
            );
            step_loss_sum += (l as f64) * (micro_scale as f64);
            step_sup += n_sup;
            step_toks += keep - 1;
        }

        let g_norm = grads.clip_grad_norm(cfg.optim.max_grad_norm);
        let cur_lr = lr_at_step(step, cfg.steps, cfg.warmup_steps, cfg.optim.lr, 0.1);
        opt.step_update(&mut adapter, &grads, cur_lr);

        let step_loss = step_loss_sum as f32;
        if step == start_step {
            initial_loss = step_loss;
        }
        final_loss = step_loss;
        if step_loss < min_loss {
            min_loss = step_loss;
        }
        total_toks += step_toks;
        sup_toks += step_sup;

        if cfg.save_every > 0
            && (step + 1).is_multiple_of(cfg.save_every)
            && let Some(ref ckpt_p) = cfg.ckpt_path
        {
            save_train_checkpoint(ckpt_p, &adapter, &opt)?;
        }

        let metrics = QloraStepMetrics {
            step: step + 1,
            loss: step_loss,
            lr: cur_lr,
            grad_norm: g_norm,
            supervised_tokens: step_sup,
            total_tokens: step_toks,
            step_ms: t_step.elapsed().as_secs_f64() * 1e3,
        };
        on_step(&metrics)?;
    }

    if let Some(ref ckpt_p) = cfg.ckpt_path {
        save_train_checkpoint(ckpt_p, &adapter, &opt)?;
    }

    let report = QloraTrainReport {
        initial_loss,
        final_loss,
        min_loss,
        total_steps: end_step.saturating_sub(start_step),
        total_tokens: total_toks,
        supervised_tokens: sup_toks,
        trainable_params,
        estimated_dirty_bytes: est_bytes,
        elapsed_sec: t_total.elapsed().as_secs_f64(),
    };
    Ok((adapter, report))
}
