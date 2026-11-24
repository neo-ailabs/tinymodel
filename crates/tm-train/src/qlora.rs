//! Phase `T2b`: CPU QLoRA trainer for `Llama / Qwen3 / SmolLM2` (`ADR-015`).
//!
//! Key guarantees:
//! 1. **0-ULP Forward Parity Gate**: Independent training forward uses the exact same `tm-cpu`
//!    kernels and `MAX_CPU_BATCH` chunking as `LlamaModel::forward_batch` (and `forward_token`
//!    for `T == 1`); when LoRA is zero, logits match `LlamaModel` to **0 ULP (`to_bits()`)**.
//! 2. **Bounded Arena Memory (`AGENTS.md` Rule #4)**: Per-layer `FP16` (or `FP32`) residual
//!    checkpoints + reusable single-layer `TrainArena` — zero heap allocations inside `forward_backward_microbatch`.
//! 3. **Masked Chunked Cross-Entropy (`CHUNK = 16`)**: Unsupervised prompt tokens (`loss_mask == 0`)
//!    skip `lm_head` forward AND backward completely, bounding peak vocab buffer to `16 * vocab * 4` bytes.
//! 4. **Deterministic Gradients (`AGENTS.md` §9)**: Token order `t = 0..T` and column-block parallel
//!    `matmul_q_transposed_f32` guarantee identical parameter gradients regardless of thread count.

use std::path::PathBuf;
use std::time::Instant;

use tm_core::{f16_to_f32, probe_system_memory};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, ThreadPool, active_kernels,
    gqa_attention_f16_kv_batch_parallel, gqa_attention_f16_kv_bwd_ext_f32,
    gqa_attention_f16_kv_ext_batch_parallel, gqa_attention_f16_kv_ext_parallel,
    gqa_attention_f16_kv_parallel, matmul_f32, matmul_pair_f32, matmul_triple_f32, matvec_f32,
    matvec_pair_f32, matvec_triple_f32, rms_norm, rms_norm_batch_parallel, rms_norm_bwd_f32,
    rope_bwd_inplace_tables, rope_compute_cos_sin, rope_inplace_tables, swiglu_bwd_f32,
    swiglu_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch, LoraSet};
use tm_models::{LlamaModel, decode_embedding_row};

use crate::TrainError;
use crate::accel::{LinearAccel, linear_bwd_with_accel, linear_fwd_with_accel};
use crate::dataset::TrainDataset;
use crate::optim::{
    AdamW, AdamWConfig, LoraGradStore, LoraTargets, init_lora_adapter, load_train_checkpoint,
    lr_at_step, save_train_checkpoint,
};

pub const CE_CHUNK_TOKENS: usize = 16;

/// Precision of saved per-layer input residual checkpoints `X^{(0..L)}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointMode {
    /// Halves checkpoint RAM (`2 * L * T * n_embd` bytes); default for edge & host QLoRA.
    #[default]
    Fp16,
    /// Exact `f32` checkpoints (used by `< 1e-5` `f64` finite-difference Jacobian verification).
    Fp32,
}

/// High-level QLoRA training job configuration (`tinymodel finetune`).
#[derive(Debug, Clone)]
pub struct QloraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub targets: LoraTargets,
    pub steps: usize,
    pub seq_len: usize,
    pub grad_accum: usize,
    pub warmup_steps: usize,
    pub optim: AdamWConfig,
    pub seed: u64,
    pub checkpoint_mode: CheckpointMode,
    pub save_every: usize,
    pub save_f16_adapter: bool,
    pub ckpt_path: Option<PathBuf>,
    pub resume_ckpt: Option<PathBuf>,
    pub stop_at_step: Option<usize>,
}

impl Default for QloraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            targets: LoraTargets::default(),
            steps: 50,
            seq_len: 128,
            grad_accum: 1,
            warmup_steps: 5,
            optim: AdamWConfig::default(),
            seed: 42,
            checkpoint_mode: CheckpointMode::Fp16,
            save_every: 0,
            save_f16_adapter: false,
            ckpt_path: None,
            resume_ckpt: None,
            stop_at_step: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QloraStepMetrics {
    pub step: usize,
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub supervised_tokens: usize,
    pub total_tokens: usize,
    pub step_ms: f64,
}

#[derive(Debug, Clone)]
pub struct QloraTrainReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub min_loss: f32,
    pub total_steps: usize,
    pub total_tokens: usize,
    pub supervised_tokens: usize,
    pub trainable_params: usize,
    pub estimated_dirty_bytes: u64,
    pub elapsed_sec: f64,
}

/// Reusable pre-allocated buffers for single-layer recomputation + backward (`AGENTS.md` Rule #4).
struct TrainArena {
    max_seq: usize,
    ckpt_mode: CheckpointMode,
    ckpt_f16: Vec<u16>,
    ckpt_f32: Vec<f32>,

    x_cur: Vec<f32>,
    x_layer_in: Vec<f32>,
    dx: Vec<f32>,

    // Recomputed single-layer forward activations (`[max_seq, dim]`)
    x_attn_norm: Vec<f32>,
    q_pre_norm: Vec<f32>,
    k_pre_norm: Vec<f32>,
    q_rope: Vec<f32>,
    v_full_f32: Vec<f32>,
    k_rope_f32: Vec<f32>,
    k_f16: Vec<u16>,
    v_f16: Vec<u16>,
    attn_out: Vec<f32>,
    attn_proj_pre_norm: Vec<f32>,
    x_mid: Vec<f32>,
    x_ffn_norm: Vec<f32>,
    gate_pre: Vec<f32>,
    up_pre: Vec<f32>,
    swiglu_out: Vec<f32>,
    ffn_proj_pre_norm: Vec<f32>,

    // Single-layer backward gradient workspaces
    xb2_chunk: Vec<f32>,
    d_branch: Vec<f32>,
    d_swiglu: Vec<f32>,
    d_gate: Vec<f32>,
    d_up: Vec<f32>,
    d_x_norm: Vec<f32>,
    d_attn_out: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    tmp_embd: Vec<f32>,
    tmp_qk: Vec<f32>,

    // Masked chunked CE buffers (`CE_CHUNK_TOKENS = 16`)
    ce_pos: Vec<(usize, u32)>,
    ce_x_sub: Vec<f32>,
    ce_x_norm: Vec<f32>,
    ce_logits: Vec<f32>,
    ce_dx_norm: Vec<f32>,

    // Attention & GEMM / LoRA scratch
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    attn_scores: Vec<f32>,
    bwd_probs: Vec<f64>,
    bwd_dp: Vec<f64>,
    mv_scratch: MatvecScratch,
    lora_scratch: LoraScratch,
    lora_h: Vec<f32>,
    lora_dh: Vec<f32>,
    lora_bt: Vec<f32>,
}

impl TrainArena {
    fn new(
        model: &LlamaModel<'_>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let c = &model.cfg;
        let d = c.n_embd;
        let q_dim = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        let ff = c.n_ff;
        let has_q_norm = model.layers.iter().any(|l| l.q_norm.is_some());
        let has_k_norm = model.layers.iter().any(|l| l.k_norm.is_some());
        let has_post_attn_norm = model.layers.iter().any(|l| l.post_attn_norm.is_some());
        let has_post_ffn_norm = model.layers.iter().any(|l| l.post_ffn_norm.is_some());
        let max_cols = d.max(ff).max(q_dim).max(c.vocab_size);
        let r_cap = max_rank.max(1);
        let max_d_out = d.max(2 * ff).max(q_dim + 2 * kv_dim).max(c.vocab_size);

        let (ckpt_f16, ckpt_f32) = match ckpt_mode {
            CheckpointMode::Fp16 => (vec![0u16; c.n_layers * max_seq * d], Vec::new()),
            CheckpointMode::Fp32 => (Vec::new(), vec![0.0f32; c.n_layers * max_seq * d]),
        };

        Self {
            max_seq,
            ckpt_mode,
            ckpt_f16,
            ckpt_f32,
            x_cur: vec![0.0; max_seq * d],
            x_layer_in: vec![0.0; max_seq * d],
            dx: vec![0.0; max_seq * d],
            x_attn_norm: vec![0.0; max_seq * d],
            q_pre_norm: if has_q_norm {
                vec![0.0; max_seq * q_dim]
            } else {
                Vec::new()
            },
            k_pre_norm: if has_k_norm {
                vec![0.0; max_seq * kv_dim]
            } else {
                Vec::new()
            },
            q_rope: vec![0.0; max_seq * q_dim],
            v_full_f32: vec![0.0; max_seq * kv_dim],
            k_rope_f32: vec![0.0; max_seq * kv_dim],
            k_f16: vec![0u16; max_seq * kv_dim],
            v_f16: vec![0u16; max_seq * kv_dim],
            attn_out: vec![0.0; max_seq * q_dim],
            attn_proj_pre_norm: if has_post_attn_norm {
                vec![0.0; max_seq * d]
            } else {
                Vec::new()
            },
            x_mid: vec![0.0; max_seq * d],
            x_ffn_norm: vec![0.0; max_seq * d],
            gate_pre: vec![0.0; max_seq * ff],
            up_pre: vec![0.0; max_seq * ff],
            swiglu_out: vec![0.0; max_seq * ff],
            ffn_proj_pre_norm: if has_post_ffn_norm {
                vec![0.0; max_seq * d]
            } else {
                Vec::new()
            },
            xb2_chunk: vec![0.0; MAX_CPU_BATCH * d.max(q_dim)],
            d_branch: vec![0.0; max_seq * d],
            d_swiglu: vec![0.0; max_seq * ff],
            d_gate: vec![0.0; max_seq * ff],
            d_up: vec![0.0; max_seq * ff],
            d_x_norm: vec![0.0; max_seq * d],
            d_attn_out: vec![0.0; max_seq * q_dim],
            dq: vec![0.0; max_seq * q_dim],
            dk: vec![0.0; max_seq * kv_dim],
            dv: vec![0.0; max_seq * kv_dim],
            tmp_embd: vec![0.0; d],
            tmp_qk: vec![0.0; q_dim.max(kv_dim).max(c.head_dim)],
            ce_pos: Vec::with_capacity(max_seq),
            ce_x_sub: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_x_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_logits: vec![0.0; CE_CHUNK_TOKENS * c.vocab_size],
            ce_dx_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            rope_cos: vec![0.0; c.rope_dim / 2],
            rope_sin: vec![0.0; c.rope_dim / 2],
            attn_scores: vec![0.0; c.n_head * max_seq],
            bwd_probs: vec![0.0; max_seq],
            bwd_dp: vec![0.0; max_seq],
            mv_scratch: MatvecScratch::new(max_cols),
            lora_scratch: LoraScratch::new(r_cap),
            lora_h: vec![0.0; max_seq * r_cap],
            lora_dh: vec![0.0; max_seq * r_cap],
            lora_bt: vec![0.0; r_cap * max_d_out],
        }
    }

    fn save_layer_ckpt(&mut self, kt: &KernelTable, layer: usize, seq_len: usize, d: usize) {
        let len = seq_len * d;
        let off = layer * self.max_seq * d;
        match self.ckpt_mode {
            CheckpointMode::Fp16 => {
                (kt.f32_to_f16_slice)(&self.x_cur[..len], &mut self.ckpt_f16[off..off + len]);
            }
            CheckpointMode::Fp32 => {
                self.ckpt_f32[off..off + len].copy_from_slice(&self.x_cur[..len]);
            }
        }
    }

    fn load_layer_ckpt(&mut self, layer: usize, seq_len: usize, d: usize) {
        let len = seq_len * d;
        let off = layer * self.max_seq * d;
        match self.ckpt_mode {
            CheckpointMode::Fp16 => {
                for (dst, &bits) in self.x_layer_in[..len]
                    .iter_mut()
                    .zip(self.ckpt_f16[off..off + len].iter())
                {
                    *dst = f16_to_f32(bits);
                }
            }
            CheckpointMode::Fp32 => {
                self.x_layer_in[..len].copy_from_slice(&self.ckpt_f32[off..off + len]);
            }
        }
    }
}

/// Estimates anonymous dirty-page RAM required by [`LlamaTrainer`] + `LoraAdapter` + `AdamW`
/// BEFORE allocating (`AGENTS.md` §9 & `F-04`).
pub fn estimate_qlora_dirty_bytes(
    model: &LlamaModel<'_>,
    max_seq: usize,
    rank: usize,
    targets: &LoraTargets,
    ckpt_mode: CheckpointMode,
) -> u64 {
    let c = &model.cfg;
    let d = c.n_embd;
    let q_dim = c.n_head * c.head_dim;
    let kv_dim = c.n_head_kv * c.head_dim;
    let ff = c.n_ff;

    let ckpt_elem_bytes = match ckpt_mode {
        CheckpointMode::Fp16 => 2usize,
        CheckpointMode::Fp32 => 4usize,
    };
    let ckpt_bytes = c.n_layers * max_seq * d * ckpt_elem_bytes;

    // Single-layer activations + grad buffers + chunked CE (`16 * vocab * 4`)
    let per_tok_floats = 6 * d + 4 * q_dim + 4 * kv_dim + 6 * ff;
    let arena_bytes = max_seq * per_tok_floats * 4
        + 2 * max_seq * kv_dim * 2
        + CE_CHUNK_TOKENS * (c.vocab_size + 3 * d) * 4;

    let mut lora_params = 0usize;
    let fused_qkv = model.layers.first().is_some_and(|l| l.wq.is_sliced);
    let fused_up = model.layers.first().is_some_and(|l| l.w_gate.is_sliced);
    if fused_qkv {
        if targets.q || targets.k || targets.v {
            lora_params += c.n_layers * rank * (d + q_dim + 2 * kv_dim);
        }
    } else {
        for (en, din, dout) in [
            (targets.q, d, q_dim),
            (targets.k, d, kv_dim),
            (targets.v, d, kv_dim),
        ] {
            if en {
                lora_params += c.n_layers * rank * (din + dout);
            }
        }
    }
    if targets.o {
        lora_params += c.n_layers * rank * (q_dim + d);
    }
    if fused_up {
        if targets.gate || targets.up {
            lora_params += c.n_layers * rank * (d + 2 * ff);
        }
    } else {
        for (en, din, dout) in [(targets.gate, d, ff), (targets.up, d, ff)] {
            if en {
                lora_params += c.n_layers * rank * (din + dout);
            }
        }
    }
    if targets.down {
        lora_params += c.n_layers * rank * (ff + d);
    }
    if targets.embed {
        lora_params += rank * (c.vocab_size + d);
    }
    if targets.lm_head {
        lora_params += rank * (d + c.vocab_size);
    }
    // Master weights (4B) + grads (4B) + bound copy (8B) + AdamW m & v (8B) = 24 bytes/param
    let param_state_bytes = lora_params * 24;

    (ckpt_bytes + arena_bytes + param_state_bytes) as u64
}

/// Checks Linux / Raspberry Pi / Android / cgroup v2 `available_bytes` budget up front.
pub fn check_training_memory_budget(required_bytes: u64) -> Result<(), TrainError> {
    const HEADROOM_BYTES: u64 = 48 * 1024 * 1024;
    if let Ok(budget) = probe_system_memory() {
        let safe_avail = budget.available_bytes.saturating_sub(HEADROOM_BYTES);
        if required_bytes > safe_avail {
            let note = budget
                .cgroup_max_bytes
                .map(|m| format!(" (cgroup memory.max={:.1} MiB)", m as f64 / 1048576.0))
                .unwrap_or_default();
            return Err(TrainError::OutOfMemoryBudget {
                required_mib: required_bytes as f64 / 1048576.0,
                available_mib: safe_avail as f64 / 1048576.0,
                note,
            });
        }
    }
    Ok(())
}

/// Independent training engine for `Llama / Qwen3 / SmolLM2` models (`#![forbid(unsafe_code)]`).
pub struct LlamaTrainer<'a> {
    pub model: &'a LlamaModel<'a>,
    arena: TrainArena,
    bound_lora: BoundLora,
    /// When `true`, skips `f32 -> f16` KV rounding during forward/backward so `f64` finite
    /// differences (`eps ~ 1e-3`) do not step across discontinuous `f16` quantization bins.
    /// Defaults to `false` (**0-ULP bit-exact** with `LlamaModel` inference).
    pub exact_f32_kv_gradcheck: bool,
}

impl<'a> LlamaTrainer<'a> {
    pub fn new(
        model: &'a LlamaModel<'a>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let seq_cap = max_seq.max(2);
        Self {
            model,
            arena: TrainArena::new(model, seq_cap, max_rank, ckpt_mode),
            bound_lora: BoundLora::default(),
            exact_f32_kv_gradcheck: false,
        }
    }

    fn sync_bound_lora(&mut self, adapter: &LoraAdapter) {
        let mut set = LoraSet::new();
        set.push(adapter.clone(), 1.0);
        self.bound_lora = BoundLora::from_lora_set(&set);
        self.arena
            .lora_scratch
            .ensure_rank(self.bound_lora.max_rank);
    }

    /// Runs Layer `l` forward over `seq_len` tokens in `MAX_CPU_BATCH` chunks, using the **exact**
    /// same `tm-cpu` kernels as `LlamaModel::forward_batch` (and `forward_token` when `seq_len == 1`).
    ///
    /// - Reads input `[seq_len, n_embd]` from `x_in` (`x_cur` during forward, `x_layer_in` during recompute).
    /// - Writes output `[seq_len, n_embd]` to `x_out` (`x_cur` during forward, `x_mid` + FFN during recompute).
    /// - When `save_recompute == true`, populates `arena.{x_attn_norm, q_rope, k_f16, v_f16, attn_out, x_mid, x_ffn_norm, gate_pre, up_pre, swiglu_out}`.
    #[allow(clippy::too_many_arguments)]
    fn run_layer_forward(
        model: &LlamaModel<'_>,
        bound: &BoundLora,
        arena: &mut TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        layer_idx: usize,
        seq_len: usize,
        from_ckpt_input: bool,
        save_recompute: bool,
        exact_f32_kv: bool,
    ) {
        let c = &model.cfg;
        let lw = &model.layers[layer_idx];
        let d = c.n_embd;
        let q_dim = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        let ff = c.n_ff;

        let slot_wq = bound.slot_for(lw.wq.name);
        let slot_wk = bound.slot_for(lw.wk.name);
        let slot_wv = bound.slot_for(lw.wv.name);
        let slot_wo = bound.slot_for(lw.wo.name);
        let slot_wg = bound.slot_for(lw.w_gate.name);
        let slot_wu = bound.slot_for(lw.w_up.name);
        let slot_wd = bound.slot_for(lw.w_down.name);

        let single_tok_mode = seq_len == 1;
        let mut offset = 0usize;
        while offset < seq_len {
            let b = (seq_len - offset).min(MAX_CPU_BATCH);
            let x_src: &[f32] = if from_ckpt_input {
                &arena.x_layer_in[offset * d..(offset + b) * d]
            } else {
                &arena.x_cur[offset * d..(offset + b) * d]
            };

            // 1. Attention RMSNorm (optional on OLMo 2/3)
            if let Some(ref an) = lw.attn_norm {
                if single_tok_mode {
                    rms_norm(
                        &mut arena.x_attn_norm[offset * d..(offset + 1) * d],
                        x_src,
                        an,
                        c.rms_eps,
                    );
                } else {
                    rms_norm_batch_parallel(
                        pool,
                        kt,
                        &mut arena.x_attn_norm[offset * d..(offset + b) * d],
                        x_src,
                        an,
                        c.rms_eps,
                        d,
                        b,
                    );
                }
            } else {
                arena.x_attn_norm[offset * d..(offset + b) * d].copy_from_slice(x_src);
            }

            // 2. QKV projections + LoRA
            let xn_chunk = &arena.x_attn_norm[offset * d..(offset + b) * d];
            if accel.is_some() {
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &lw.wq,
                    bound,
                    slot_wq,
                    xn_chunk,
                    &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &lw.wk,
                    bound,
                    slot_wk,
                    xn_chunk,
                    &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &lw.wv,
                    bound,
                    slot_wv,
                    xn_chunk,
                    &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
            } else {
                if single_tok_mode {
                    matvec_triple_f32(
                        pool,
                        kt,
                        lw.wq.dtype,
                        lw.wq.bytes,
                        q_dim,
                        &mut arena.q_rope[..q_dim],
                        lw.wk.dtype,
                        lw.wk.bytes,
                        kv_dim,
                        &mut arena.k_rope_f32[..kv_dim],
                        lw.wv.dtype,
                        lw.wv.bytes,
                        kv_dim,
                        &mut arena.v_full_f32[..kv_dim],
                        xn_chunk,
                        &mut arena.mv_scratch,
                        d,
                    );
                } else {
                    matmul_triple_f32(
                        pool,
                        kt,
                        lw.wq.dtype,
                        lw.wq.bytes,
                        q_dim,
                        &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                        lw.wk.dtype,
                        lw.wk.bytes,
                        kv_dim,
                        &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                        lw.wv.dtype,
                        lw.wv.bytes,
                        kv_dim,
                        &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                        xn_chunk,
                        &mut arena.mv_scratch,
                        d,
                        b,
                    );
                }
                if !bound.is_empty() {
                    bound.apply_linear_slice(
                        slot_wq,
                        pool,
                        kt,
                        xn_chunk,
                        &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                        b,
                        lw.wq.row_offset,
                        q_dim,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear_slice(
                        slot_wk,
                        pool,
                        kt,
                        xn_chunk,
                        &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                        b,
                        lw.wk.row_offset,
                        kv_dim,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear_slice(
                        slot_wv,
                        pool,
                        kt,
                        xn_chunk,
                        &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                        b,
                        lw.wv.row_offset,
                        kv_dim,
                        &mut arena.lora_scratch,
                    );
                }
            }

            if let Some(ref bq) = lw.bq {
                for t in 0..b {
                    let dst = &mut arena.q_rope[(offset + t) * q_dim..(offset + t + 1) * q_dim];
                    for (y, &bi) in dst.iter_mut().zip(bq.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref bk) = lw.bk {
                for t in 0..b {
                    let dst =
                        &mut arena.k_rope_f32[(offset + t) * kv_dim..(offset + t + 1) * kv_dim];
                    for (y, &bi) in dst.iter_mut().zip(bk.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref bv) = lw.bv {
                for t in 0..b {
                    let dst =
                        &mut arena.v_full_f32[(offset + t) * kv_dim..(offset + t + 1) * kv_dim];
                    for (y, &bi) in dst.iter_mut().zip(bv.iter()) {
                        *y += bi;
                    }
                }
            }

            if save_recompute {
                if lw.q_norm.is_some() {
                    arena.q_pre_norm[offset * q_dim..(offset + b) * q_dim]
                        .copy_from_slice(&arena.q_rope[offset * q_dim..(offset + b) * q_dim]);
                }
                if lw.k_norm.is_some() {
                    arena.k_pre_norm[offset * kv_dim..(offset + b) * kv_dim]
                        .copy_from_slice(&arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim]);
                }
            }

            // 3. Per-token QK-Norm (per-head or full-vector), RoPE, and F16 KV store
            let rope_freqs_layer = model.rope_inv_freqs_for_layer(layer_idx, arena.max_seq);
            for t in 0..b {
                let pos = offset + t;
                let qt = &mut arena.q_rope[pos * q_dim..(pos + 1) * q_dim];
                let kt_vec = &mut arena.k_rope_f32[pos * kv_dim..(pos + 1) * kv_dim];
                let vt_vec = &arena.v_full_f32[pos * kv_dim..(pos + 1) * kv_dim];

                if let Some(ref qn) = lw.q_norm {
                    if qn.len() == q_dim {
                        rms_norm(&mut arena.tmp_qk[..q_dim], qt, qn, c.rms_eps);
                        qt.copy_from_slice(&arena.tmp_qk[..q_dim]);
                    } else {
                        for h in 0..c.n_head {
                            let s = h * c.head_dim;
                            let e = s + c.head_dim;
                            rms_norm(&mut arena.tmp_qk[..c.head_dim], &qt[s..e], qn, c.rms_eps);
                            qt[s..e].copy_from_slice(&arena.tmp_qk[..c.head_dim]);
                        }
                    }
                }
                if let Some(ref kn) = lw.k_norm {
                    if kn.len() == kv_dim {
                        rms_norm(&mut arena.tmp_qk[..kv_dim], kt_vec, kn, c.rms_eps);
                        kt_vec.copy_from_slice(&arena.tmp_qk[..kv_dim]);
                    } else {
                        for h in 0..c.n_head_kv {
                            let s = h * c.head_dim;
                            let e = s + c.head_dim;
                            rms_norm(
                                &mut arena.tmp_qk[..c.head_dim],
                                &kt_vec[s..e],
                                kn,
                                c.rms_eps,
                            );
                            kt_vec[s..e].copy_from_slice(&arena.tmp_qk[..c.head_dim]);
                        }
                    }
                }

                if c.layer_uses_rope(layer_idx) {
                    rope_compute_cos_sin(
                        rope_freqs_layer,
                        pos,
                        &mut arena.rope_cos,
                        &mut arena.rope_sin,
                    );
                    rope_inplace_tables(
                        qt,
                        c.n_head,
                        c.head_dim,
                        c.rope_dim,
                        &arena.rope_cos,
                        &arena.rope_sin,
                        c.rope_type,
                    );
                    rope_inplace_tables(
                        kt_vec,
                        c.n_head_kv,
                        c.head_dim,
                        c.rope_dim,
                        &arena.rope_cos,
                        &arena.rope_sin,
                        c.rope_type,
                    );
                }

                if !exact_f32_kv {
                    let kv_off = pos * kv_dim;
                    (kt.f32_to_f16_slice)(kt_vec, &mut arena.k_f16[kv_off..kv_off + kv_dim]);
                    (kt.f32_to_f16_slice)(vt_vec, &mut arena.v_f16[kv_off..kv_off + kv_dim]);
                }
            }

            // 4. Causal GQA attention
            let win = c.layer_window(layer_idx);
            let needs_ext = c.layer_needs_ext_attn(layer_idx);
            if exact_f32_kv {
                gqa_attention_f32_kv_exact(
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.k_rope_f32[..(offset + b) * kv_dim],
                    &arena.v_full_f32[..(offset + b) * kv_dim],
                    c.n_head,
                    c.n_head_kv,
                    c.head_dim,
                    offset,
                    b,
                    c.attn_scale as f64,
                    win,
                );
            } else if single_tok_mode {
                if needs_ext {
                    gqa_attention_f16_kv_ext_parallel(
                        pool,
                        kt,
                        &mut arena.attn_out[..q_dim],
                        &arena.q_rope[..q_dim],
                        &arena.k_f16,
                        &arena.v_f16,
                        c.n_head,
                        c.n_head_kv,
                        c.head_dim,
                        0,
                        c.attn_scale,
                        win,
                        arena.max_seq,
                        &mut arena.attn_scores,
                    );
                } else {
                    gqa_attention_f16_kv_parallel(
                        pool,
                        kt,
                        &mut arena.attn_out[..q_dim],
                        &arena.q_rope[..q_dim],
                        &arena.k_f16,
                        &arena.v_f16,
                        c.n_head,
                        c.n_head_kv,
                        c.head_dim,
                        1,
                        arena.max_seq,
                        &mut arena.attn_scores,
                    );
                }
            } else if needs_ext {
                gqa_attention_f16_kv_ext_batch_parallel(
                    pool,
                    kt,
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.k_f16,
                    &arena.v_f16,
                    c.n_head,
                    c.n_head_kv,
                    c.head_dim,
                    offset,
                    b,
                    c.attn_scale,
                    win,
                    arena.max_seq,
                    &mut arena.attn_scores,
                );
            } else {
                gqa_attention_f16_kv_batch_parallel(
                    pool,
                    kt,
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.k_f16,
                    &arena.v_f16,
                    c.n_head,
                    c.n_head_kv,
                    c.head_dim,
                    offset,
                    b,
                    arena.max_seq,
                    &mut arena.attn_scores,
                );
            }

            // 5. Output projection (`wo`) + optional bias + optional post-attn norm + residual -> `x_mid`
            let ao_chunk = &arena.attn_out[offset * q_dim..(offset + b) * q_dim];
            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wo,
                bound,
                slot_wo,
                ao_chunk,
                &mut arena.xb2_chunk[..b * d],
                b,
                single_tok_mode,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );
            if let Some(ref bo) = lw.bo {
                for t in 0..b {
                    let dst = &mut arena.xb2_chunk[t * d..(t + 1) * d];
                    for (y, &bi) in dst.iter_mut().zip(bo.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref pan) = lw.post_attn_norm {
                if save_recompute {
                    arena.attn_proj_pre_norm[offset * d..(offset + b) * d]
                        .copy_from_slice(&arena.xb2_chunk[..b * d]);
                }
                if single_tok_mode {
                    rms_norm(&mut arena.tmp_embd, &arena.xb2_chunk[..d], pan, c.rms_eps);
                    arena.xb2_chunk[..d].copy_from_slice(&arena.tmp_embd);
                } else {
                    rms_norm_batch_parallel(
                        pool,
                        kt,
                        &mut arena.d_branch[..b * d],
                        &arena.xb2_chunk[..b * d],
                        pan,
                        c.rms_eps,
                        d,
                        b,
                    );
                    arena.xb2_chunk[..b * d].copy_from_slice(&arena.d_branch[..b * d]);
                }
            }
            let res_scale = c.residual_scale;
            for i in 0..(b * d) {
                let xin_i = if from_ckpt_input {
                    arena.x_layer_in[offset * d + i]
                } else {
                    arena.x_cur[offset * d + i]
                };
                arena.x_mid[offset * d + i] = xin_i + res_scale * arena.xb2_chunk[i];
            }

            // 6. FFN RMSNorm (optional on OLMo 2/3) + SwiGLU FFN + residual
            let xm_chunk = &arena.x_mid[offset * d..(offset + b) * d];
            if let Some(ref fn_w) = lw.ffn_norm {
                if single_tok_mode {
                    rms_norm(
                        &mut arena.x_ffn_norm[offset * d..(offset + 1) * d],
                        xm_chunk,
                        fn_w,
                        c.rms_eps,
                    );
                } else {
                    rms_norm_batch_parallel(
                        pool,
                        kt,
                        &mut arena.x_ffn_norm[offset * d..(offset + b) * d],
                        xm_chunk,
                        fn_w,
                        c.rms_eps,
                        d,
                        b,
                    );
                }
            } else {
                arena.x_ffn_norm[offset * d..(offset + b) * d].copy_from_slice(xm_chunk);
            }

            let xfn_chunk = &arena.x_ffn_norm[offset * d..(offset + b) * d];
            if accel.is_some() {
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &lw.w_gate,
                    bound,
                    slot_wg,
                    xfn_chunk,
                    &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &lw.w_up,
                    bound,
                    slot_wu,
                    xfn_chunk,
                    &mut arena.up_pre[offset * ff..(offset + b) * ff],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
            } else {
                if single_tok_mode {
                    matvec_pair_f32(
                        pool,
                        kt,
                        lw.w_gate.dtype,
                        lw.w_gate.bytes,
                        ff,
                        &mut arena.gate_pre[..ff],
                        lw.w_up.dtype,
                        lw.w_up.bytes,
                        ff,
                        &mut arena.up_pre[..ff],
                        xfn_chunk,
                        &mut arena.mv_scratch,
                        d,
                    );
                } else {
                    matmul_pair_f32(
                        pool,
                        kt,
                        lw.w_gate.dtype,
                        lw.w_gate.bytes,
                        ff,
                        &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                        lw.w_up.dtype,
                        lw.w_up.bytes,
                        ff,
                        &mut arena.up_pre[offset * ff..(offset + b) * ff],
                        xfn_chunk,
                        &mut arena.mv_scratch,
                        d,
                        b,
                    );
                }
                if !bound.is_empty() {
                    bound.apply_linear_slice(
                        slot_wg,
                        pool,
                        kt,
                        xfn_chunk,
                        &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                        b,
                        lw.w_gate.row_offset,
                        ff,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear_slice(
                        slot_wu,
                        pool,
                        kt,
                        xfn_chunk,
                        &mut arena.up_pre[offset * ff..(offset + b) * ff],
                        b,
                        lw.w_up.row_offset,
                        ff,
                        &mut arena.lora_scratch,
                    );
                }
            }
            if let Some(ref bg) = lw.b_gate {
                for t in 0..b {
                    let dst = &mut arena.gate_pre[(offset + t) * ff..(offset + t + 1) * ff];
                    for (y, &bi) in dst.iter_mut().zip(bg.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref bu) = lw.b_up {
                for t in 0..b {
                    let dst = &mut arena.up_pre[(offset + t) * ff..(offset + t + 1) * ff];
                    for (y, &bi) in dst.iter_mut().zip(bu.iter()) {
                        *y += bi;
                    }
                }
            }

            arena.swiglu_out[offset * ff..(offset + b) * ff]
                .copy_from_slice(&arena.gate_pre[offset * ff..(offset + b) * ff]);
            swiglu_inplace_parallel(
                pool,
                &mut arena.swiglu_out[offset * ff..(offset + b) * ff],
                &arena.up_pre[offset * ff..(offset + b) * ff],
            );

            let sw_chunk = &arena.swiglu_out[offset * ff..(offset + b) * ff];
            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_down,
                bound,
                slot_wd,
                sw_chunk,
                &mut arena.xb2_chunk[..b * d],
                b,
                single_tok_mode,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );
            if let Some(ref bd) = lw.b_down {
                for t in 0..b {
                    let dst = &mut arena.xb2_chunk[t * d..(t + 1) * d];
                    for (y, &bi) in dst.iter_mut().zip(bd.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref pfn) = lw.post_ffn_norm {
                if save_recompute {
                    arena.ffn_proj_pre_norm[offset * d..(offset + b) * d]
                        .copy_from_slice(&arena.xb2_chunk[..b * d]);
                }
                if !save_recompute {
                    if single_tok_mode {
                        rms_norm(&mut arena.tmp_embd, &arena.xb2_chunk[..d], pfn, c.rms_eps);
                        arena.xb2_chunk[..d].copy_from_slice(&arena.tmp_embd);
                    } else {
                        rms_norm_batch_parallel(
                            pool,
                            kt,
                            &mut arena.d_branch[..b * d],
                            &arena.xb2_chunk[..b * d],
                            pfn,
                            c.rms_eps,
                            d,
                            b,
                        );
                        arena.xb2_chunk[..b * d].copy_from_slice(&arena.d_branch[..b * d]);
                    }
                }
            }

            if !save_recompute {
                for (i, (&xm_i, &xb2_i)) in xm_chunk
                    .iter()
                    .zip(arena.xb2_chunk[..b * d].iter())
                    .enumerate()
                {
                    arena.x_cur[offset * d + i] = xm_i + res_scale * xb2_i;
                }
            }

            offset += b;
        }
    }

    /// Evaluates the training forward pass on `token_ids` and writes the final-token logits
    /// (`[vocab_size]`) to `out_last_logits`.
    ///
    /// Used by the **0-ULP Bit-Exact Training Forward Gate** (`ADR-015` §3) against
    /// `LlamaModel::forward_batch` / `LlamaModel::forward_token`.
    pub fn forward_last_logits(
        &mut self,
        pool: &ThreadPool,
        adapter: Option<&LoraAdapter>,
        token_ids: &[u32],
        out_last_logits: &mut [f32],
    ) {
        assert!(!token_ids.is_empty() && token_ids.len() <= self.arena.max_seq);
        assert_eq!(out_last_logits.len(), self.model.cfg.vocab_size);
        if let Some(ad) = adapter {
            self.sync_bound_lora(ad);
        } else {
            self.bound_lora = BoundLora::default();
        }
        let kt = active_kernels();
        let d = self.model.cfg.n_embd;
        let n_tok = token_ids.len();

        let slot_emb = self.bound_lora.slot_for(self.model.token_embd.name);
        for (t, &tok) in token_ids.iter().enumerate() {
            let dst = &mut self.arena.x_cur[t * d..(t + 1) * d];
            decode_embedding_row(
                self.model.token_embd.dtype,
                self.model.token_embd.bytes,
                d,
                tok as usize,
                dst,
            );
            self.bound_lora
                .apply_embed_row(slot_emb, kt, tok as usize, dst);
            if (self.model.cfg.embedding_scale - 1.0).abs() > 1e-12 {
                let s = self.model.cfg.embedding_scale;
                for v in dst.iter_mut() {
                    *v *= s;
                }
            }
        }

        for l in 0..self.model.cfg.n_layers {
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                None,
                pool,
                kt,
                l,
                n_tok,
                false,
                false,
                self.exact_f32_kv_gradcheck,
            );
        }

        let last_x = &self.arena.x_cur[(n_tok - 1) * d..n_tok * d];
        rms_norm(
            &mut self.arena.tmp_embd,
            last_x,
            &self.model.output_norm,
            self.model.cfg.rms_eps,
        );
        matvec_f32(
            pool,
            kt,
            self.model.output.dtype,
            self.model.output.bytes,
            &self.arena.tmp_embd,
            &mut self.arena.mv_scratch,
            out_last_logits,
            self.model.output.rows,
            self.model.output.cols,
        );
        let slot_out = self.bound_lora.slot_for(self.model.output.name);
        self.bound_lora.apply_linear(
            slot_out,
            pool,
            kt,
            &self.arena.tmp_embd,
            out_last_logits,
            1,
            &mut self.arena.lora_scratch,
        );
        if let Some(ref ob) = self.model.output_bias {
            for (y, &bi) in out_last_logits.iter_mut().zip(ob.iter()) {
                *y += bi;
            }
        }
        if (self.model.cfg.inv_logit_scale - 1.0).abs() > 1e-12 {
            let s = self.model.cfg.inv_logit_scale;
            for v in out_last_logits.iter_mut() {
                *v *= s;
            }
        }
    }

    /// Executes 1 micro-batch forward + masked chunked cross-entropy + per-layer recomputed backward,
    /// accumulating `grad_scale`-weighted gradients into `grads` (zero heap allocations!).
    ///
    /// Returns `(mean_ce_loss, supervised_token_count)`.
    pub fn forward_backward_microbatch(
        &mut self,
        pool: &ThreadPool,
        adapter: &LoraAdapter,
        grads: &mut LoraGradStore,
        tokens: &[u32],
        loss_mask: &[u8],
        grad_scale: f32,
    ) -> (f32, usize) {
        self.forward_backward_microbatch_with_accel(
            None, pool, adapter, grads, tokens, loss_mask, grad_scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward_microbatch_with_accel(
        &mut self,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        adapter: &LoraAdapter,
        grads: &mut LoraGradStore,
        tokens: &[u32],
        loss_mask: &[u8],
        grad_scale: f32,
    ) -> (f32, usize) {
        assert_eq!(tokens.len(), loss_mask.len());
        assert!(tokens.len() >= 2);
        let seq_len = (tokens.len() - 1).min(self.arena.max_seq);
        let input_toks = &tokens[..seq_len];

        self.arena.ce_pos.clear();
        for t in 0..seq_len {
            if loss_mask[t + 1] != 0 {
                self.arena.ce_pos.push((t, tokens[t + 1]));
            }
        }
        let n_valid = self.arena.ce_pos.len();
        if n_valid == 0 {
            return (0.0, 0);
        }

        self.sync_bound_lora(adapter);
        let kt = active_kernels();
        let c = &self.model.cfg;
        let d = c.n_embd;
        let q_dim = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        let ff = c.n_ff;
        let vocab = c.vocab_size;

        // 1. Forward pass with per-layer input residual checkpointing (`X^{(0..L)}`)
        let slot_emb = self.bound_lora.slot_for(self.model.token_embd.name);
        for (t, &tok) in input_toks.iter().enumerate() {
            let dst = &mut self.arena.x_cur[t * d..(t + 1) * d];
            decode_embedding_row(
                self.model.token_embd.dtype,
                self.model.token_embd.bytes,
                d,
                tok as usize,
                dst,
            );
            self.bound_lora
                .apply_embed_row(slot_emb, kt, tok as usize, dst);
            if (c.embedding_scale - 1.0).abs() > 1e-12 {
                let s = c.embedding_scale;
                for v in dst.iter_mut() {
                    *v *= s;
                }
            }
        }

        for l in 0..c.n_layers {
            self.arena.save_layer_ckpt(kt, l, seq_len, d);
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                accel.as_deref_mut(),
                pool,
                kt,
                l,
                seq_len,
                false,
                false,
                self.exact_f32_kv_gradcheck,
            );
        }

        // 2. Masked Chunked Cross-Entropy (`CE_CHUNK_TOKENS = 16`) + `output_norm` / `output` backward
        self.arena.dx[..seq_len * d].fill(0.0);
        let mut loss_sum = 0.0f64;
        let slot_out = self.bound_lora.slot_for(self.model.output.name);
        let inv_logit = c.inv_logit_scale;
        let inv_valid = (grad_scale as f64) * (inv_logit as f64) / (n_valid as f64);

        let mut c0 = 0usize;
        while c0 < n_valid {
            let b = (n_valid - c0).min(CE_CHUNK_TOKENS);
            for k in 0..b {
                let t = self.arena.ce_pos[c0 + k].0;
                let x_t = &self.arena.x_cur[t * d..(t + 1) * d];
                self.arena.ce_x_sub[k * d..(k + 1) * d].copy_from_slice(x_t);
                rms_norm(
                    &mut self.arena.ce_x_norm[k * d..(k + 1) * d],
                    x_t,
                    &self.model.output_norm,
                    c.rms_eps,
                );
            }

            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &self.model.output,
                &self.bound_lora,
                slot_out,
                &self.arena.ce_x_norm[..b * d],
                &mut self.arena.ce_logits[..b * vocab],
                b,
                false,
                &mut self.arena.mv_scratch,
                &mut self.arena.lora_scratch,
            );
            if let Some(ref ob) = self.model.output_bias {
                for k in 0..b {
                    let row = &mut self.arena.ce_logits[k * vocab..(k + 1) * vocab];
                    for (y, &bi) in row.iter_mut().zip(ob.iter()) {
                        *y += bi;
                    }
                }
            }
            if (inv_logit - 1.0).abs() > 1e-12 {
                for lv in &mut self.arena.ce_logits[..b * vocab] {
                    *lv *= inv_logit;
                }
            }

            for k in 0..b {
                let target = self.arena.ce_pos[c0 + k].1 as usize;
                let row = &mut self.arena.ce_logits[k * vocab..(k + 1) * vocab];
                let mut max_l = f32::NEG_INFINITY;
                for &lv in row.iter() {
                    if lv > max_l {
                        max_l = lv;
                    }
                }
                let mut sum_exp = 0.0f64;
                for &lv in row.iter() {
                    sum_exp += ((lv - max_l) as f64).exp();
                }
                let log_z = (max_l as f64) + sum_exp.ln();
                loss_sum += log_z - (row[target] as f64);

                for (v_idx, lv) in row.iter_mut().enumerate() {
                    let p = ((*lv as f64) - log_z).exp();
                    let dl = (p - if v_idx == target { 1.0 } else { 0.0 }) * inv_valid;
                    *lv = dl as f32;
                }
            }

            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &self.model.output,
                adapter,
                grads,
                &self.arena.ce_x_norm[..b * d],
                &self.arena.ce_logits[..b * vocab],
                &mut self.arena.ce_dx_norm[..b * d],
                b,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );

            for k in 0..b {
                let t = self.arena.ce_pos[c0 + k].0;
                rms_norm_bwd_f32(
                    &mut self.arena.dx[t * d..(t + 1) * d],
                    None,
                    &self.arena.ce_x_sub[k * d..(k + 1) * d],
                    &self.model.output_norm,
                    &self.arena.ce_dx_norm[k * d..(k + 1) * d],
                    c.rms_eps,
                );
            }

            c0 += b;
        }

        // 3. Reverse layer loop (`l = n_layers - 1 ..= 0`): recompute from `ckpt_x[l]` + `backward_layer`
        for l in (0..c.n_layers).rev() {
            self.arena.load_layer_ckpt(l, seq_len, d);
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                accel.as_deref_mut(),
                pool,
                kt,
                l,
                seq_len,
                true,
                true,
                self.exact_f32_kv_gradcheck,
            );

            let lw = &self.model.layers[l];
            let res_scale = c.residual_scale;

            // 3a. FFN backward (`post_ffn_norm` -> `w_down` -> `swiglu` -> `w_gate`/`w_up` -> `ffn_norm` + residual)
            if let Some(ref pfn) = lw.post_ffn_norm {
                for t in 0..seq_len {
                    for i in 0..d {
                        self.arena.tmp_embd[i] = res_scale * self.arena.dx[t * d + i];
                    }
                    rms_norm_bwd_f32(
                        &mut self.arena.d_branch[t * d..(t + 1) * d],
                        None,
                        &self.arena.ffn_proj_pre_norm[t * d..(t + 1) * d],
                        pfn,
                        &self.arena.tmp_embd,
                        c.rms_eps,
                    );
                }
            } else {
                for i in 0..(seq_len * d) {
                    self.arena.d_branch[i] = res_scale * self.arena.dx[i];
                }
            }

            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_down,
                adapter,
                grads,
                &self.arena.swiglu_out[..seq_len * ff],
                &self.arena.d_branch[..seq_len * d],
                &mut self.arena.d_swiglu[..seq_len * ff],
                seq_len,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
            swiglu_bwd_f32(
                &mut self.arena.d_gate[..seq_len * ff],
                &mut self.arena.d_up[..seq_len * ff],
                &self.arena.gate_pre[..seq_len * ff],
                &self.arena.up_pre[..seq_len * ff],
                &self.arena.d_swiglu[..seq_len * ff],
            );
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_gate,
                adapter,
                grads,
                &self.arena.x_ffn_norm[..seq_len * d],
                &self.arena.d_gate[..seq_len * ff],
                &mut self.arena.d_x_norm[..seq_len * d],
                seq_len,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_up,
                adapter,
                grads,
                &self.arena.x_ffn_norm[..seq_len * d],
                &self.arena.d_up[..seq_len * ff],
                &mut self.arena.d_x_norm[..seq_len * d],
                seq_len,
                true,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
            if let Some(ref fn_w) = lw.ffn_norm {
                for t in 0..seq_len {
                    rms_norm_bwd_f32(
                        &mut self.arena.tmp_embd,
                        None,
                        &self.arena.x_mid[t * d..(t + 1) * d],
                        fn_w,
                        &self.arena.d_x_norm[t * d..(t + 1) * d],
                        c.rms_eps,
                    );
                    (kt.axpy_f32)(
                        &mut self.arena.dx[t * d..(t + 1) * d],
                        1.0,
                        &self.arena.tmp_embd,
                    );
                }
            } else {
                (kt.axpy_f32)(
                    &mut self.arena.dx[..seq_len * d],
                    1.0,
                    &self.arena.d_x_norm[..seq_len * d],
                );
            }

            // 3b. GQA Attention backward (`post_attn_norm` -> `wo` -> `gqa_attn` -> `RoPE` -> `QK-Norm` -> `wq`/`wk`/`wv` -> `attn_norm` + residual)
            if let Some(ref pan) = lw.post_attn_norm {
                for t in 0..seq_len {
                    for i in 0..d {
                        self.arena.tmp_embd[i] = res_scale * self.arena.dx[t * d + i];
                    }
                    rms_norm_bwd_f32(
                        &mut self.arena.d_branch[t * d..(t + 1) * d],
                        None,
                        &self.arena.attn_proj_pre_norm[t * d..(t + 1) * d],
                        pan,
                        &self.arena.tmp_embd,
                        c.rms_eps,
                    );
                }
            } else {
                for i in 0..(seq_len * d) {
                    self.arena.d_branch[i] = res_scale * self.arena.dx[i];
                }
            }

            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wo,
                adapter,
                grads,
                &self.arena.attn_out[..seq_len * q_dim],
                &self.arena.d_branch[..seq_len * d],
                &mut self.arena.d_attn_out[..seq_len * q_dim],
                seq_len,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );

            let win = c.layer_window(l);
            if self.exact_f32_kv_gradcheck {
                gqa_attention_f32_kv_bwd_exact(
                    kt,
                    &mut self.arena.dq[..seq_len * q_dim],
                    &mut self.arena.dk[..seq_len * kv_dim],
                    &mut self.arena.dv[..seq_len * kv_dim],
                    &self.arena.q_rope[..seq_len * q_dim],
                    &self.arena.k_rope_f32[..seq_len * kv_dim],
                    &self.arena.v_full_f32[..seq_len * kv_dim],
                    &self.arena.d_attn_out[..seq_len * q_dim],
                    c.n_head,
                    c.n_head_kv,
                    c.head_dim,
                    seq_len,
                    c.attn_scale as f64,
                    win,
                    &mut self.arena.bwd_probs,
                    &mut self.arena.bwd_dp,
                );
            } else {
                gqa_attention_f16_kv_bwd_ext_f32(
                    kt,
                    &mut self.arena.dq[..seq_len * q_dim],
                    &mut self.arena.dk[..seq_len * kv_dim],
                    &mut self.arena.dv[..seq_len * kv_dim],
                    &self.arena.q_rope[..seq_len * q_dim],
                    &self.arena.k_f16[..seq_len * kv_dim],
                    &self.arena.v_f16[..seq_len * kv_dim],
                    &self.arena.d_attn_out[..seq_len * q_dim],
                    c.n_head,
                    c.n_head_kv,
                    c.head_dim,
                    seq_len,
                    c.attn_scale,
                    win,
                );
            }

            let rope_freqs_layer = self.model.rope_inv_freqs_for_layer(l, self.arena.max_seq);
            for t in 0..seq_len {
                let dqt = &mut self.arena.dq[t * q_dim..(t + 1) * q_dim];
                let dkt = &mut self.arena.dk[t * kv_dim..(t + 1) * kv_dim];
                if c.layer_uses_rope(l) {
                    rope_compute_cos_sin(
                        rope_freqs_layer,
                        t,
                        &mut self.arena.rope_cos,
                        &mut self.arena.rope_sin,
                    );
                    rope_bwd_inplace_tables(
                        dqt,
                        c.n_head,
                        c.head_dim,
                        c.rope_dim,
                        &self.arena.rope_cos,
                        &self.arena.rope_sin,
                        c.rope_type,
                    );
                    rope_bwd_inplace_tables(
                        dkt,
                        c.n_head_kv,
                        c.head_dim,
                        c.rope_dim,
                        &self.arena.rope_cos,
                        &self.arena.rope_sin,
                        c.rope_type,
                    );
                }

                if let Some(ref qn) = lw.q_norm {
                    let q_pre_t = &self.arena.q_pre_norm[t * q_dim..(t + 1) * q_dim];
                    if qn.len() == q_dim {
                        rms_norm_bwd_f32(
                            &mut self.arena.tmp_qk[..q_dim],
                            None,
                            q_pre_t,
                            qn,
                            dqt,
                            c.rms_eps,
                        );
                        dqt.copy_from_slice(&self.arena.tmp_qk[..q_dim]);
                    } else {
                        for h in 0..c.n_head {
                            let s = h * c.head_dim;
                            let e = s + c.head_dim;
                            rms_norm_bwd_f32(
                                &mut self.arena.tmp_qk[..c.head_dim],
                                None,
                                &q_pre_t[s..e],
                                qn,
                                &dqt[s..e],
                                c.rms_eps,
                            );
                            dqt[s..e].copy_from_slice(&self.arena.tmp_qk[..c.head_dim]);
                        }
                    }
                }
                if let Some(ref kn) = lw.k_norm {
                    let k_pre_t = &self.arena.k_pre_norm[t * kv_dim..(t + 1) * kv_dim];
                    if kn.len() == kv_dim {
                        rms_norm_bwd_f32(
                            &mut self.arena.tmp_qk[..kv_dim],
                            None,
                            k_pre_t,
                            kn,
                            dkt,
                            c.rms_eps,
                        );
                        dkt.copy_from_slice(&self.arena.tmp_qk[..kv_dim]);
                    } else {
                        for h in 0..c.n_head_kv {
                            let s = h * c.head_dim;
                            let e = s + c.head_dim;
                            rms_norm_bwd_f32(
                                &mut self.arena.tmp_qk[..c.head_dim],
                                None,
                                &k_pre_t[s..e],
                                kn,
                                &dkt[s..e],
                                c.rms_eps,
                            );
                            dkt[s..e].copy_from_slice(&self.arena.tmp_qk[..c.head_dim]);
                        }
                    }
                }
            }

            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wq,
                adapter,
                grads,
                &self.arena.x_attn_norm[..seq_len * d],
                &self.arena.dq[..seq_len * q_dim],
                &mut self.arena.d_x_norm[..seq_len * d],
                seq_len,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wk,
                adapter,
                grads,
                &self.arena.x_attn_norm[..seq_len * d],
                &self.arena.dk[..seq_len * kv_dim],
                &mut self.arena.d_x_norm[..seq_len * d],
                seq_len,
                true,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wv,
                adapter,
                grads,
                &self.arena.x_attn_norm[..seq_len * d],
                &self.arena.dv[..seq_len * kv_dim],
                &mut self.arena.d_x_norm[..seq_len * d],
                seq_len,
                true,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );

            if let Some(ref an) = lw.attn_norm {
                for t in 0..seq_len {
                    rms_norm_bwd_f32(
                        &mut self.arena.tmp_embd,
                        None,
                        &self.arena.x_layer_in[t * d..(t + 1) * d],
                        an,
                        &self.arena.d_x_norm[t * d..(t + 1) * d],
                        c.rms_eps,
                    );
                    (kt.axpy_f32)(
                        &mut self.arena.dx[t * d..(t + 1) * d],
                        1.0,
                        &self.arena.tmp_embd,
                    );
                }
            } else {
                (kt.axpy_f32)(
                    &mut self.arena.dx[..seq_len * d],
                    1.0,
                    &self.arena.d_x_norm[..seq_len * d],
                );
            }
        }

        // 4. Optional `token_embd.weight` LoRA backward (`A: [vocab, r]`, `B: [d, r]`)
        if let Some(pair) = adapter.pairs.get(self.model.token_embd.name)
            && pair.is_token_embd
        {
            let scale = pair.effective_scale(adapter.alpha, 1.0) * c.embedding_scale;
            if scale != 0.0 {
                let r = pair.rank;
                let g = grads
                    .grads
                    .get_mut(self.model.token_embd.name)
                    .expect("embed grad slot");
                for (t, &tok) in input_toks.iter().enumerate() {
                    let tid = tok as usize;
                    let dxt = &self.arena.dx[t * d..(t + 1) * d];
                    let a_row = &pair.a[tid * r..(tid + 1) * r];
                    for (r_idx, &ar_val) in a_row.iter().enumerate() {
                        let mut dot_dxb = 0.0f64;
                        for (o, &dx_o) in dxt.iter().enumerate() {
                            dot_dxb += (dx_o as f64) * (pair.b[o * r + r_idx] as f64);
                            g.db[o * r + r_idx] =
                                (dx_o * scale).mul_add(ar_val, g.db[o * r + r_idx]);
                        }
                        g.da[tid * r + r_idx] += (dot_dxb * (scale as f64)) as f32;
                    }
                }
            }
        }

        let mean_loss = (loss_sum / (n_valid as f64)) as f32;
        (mean_loss, n_valid)
    }

    /// Evaluates the exact `f64` masked cross-entropy loss (forward only, no backward),
    /// avoiding `f32` scalar loss rounding when running `< 1e-5` central finite-difference checks.
    pub fn forward_loss_f64(
        &mut self,
        pool: &ThreadPool,
        adapter: &LoraAdapter,
        tokens: &[u32],
        loss_mask: &[u8],
    ) -> f64 {
        assert_eq!(tokens.len(), loss_mask.len());
        let seq_len = (tokens.len() - 1).min(self.arena.max_seq);
        let input_toks = &tokens[..seq_len];
        self.arena.ce_pos.clear();
        for t in 0..seq_len {
            if loss_mask[t + 1] != 0 {
                self.arena.ce_pos.push((t, tokens[t + 1]));
            }
        }
        let n_valid = self.arena.ce_pos.len();
        if n_valid == 0 {
            return 0.0;
        }
        self.sync_bound_lora(adapter);
        let kt = active_kernels();
        let c = &self.model.cfg;
        let d = c.n_embd;
        let vocab = c.vocab_size;

        let slot_emb = self.bound_lora.slot_for(self.model.token_embd.name);
        for (t, &tok) in input_toks.iter().enumerate() {
            let dst = &mut self.arena.x_cur[t * d..(t + 1) * d];
            decode_embedding_row(
                self.model.token_embd.dtype,
                self.model.token_embd.bytes,
                d,
                tok as usize,
                dst,
            );
            self.bound_lora
                .apply_embed_row(slot_emb, kt, tok as usize, dst);
            if (c.embedding_scale - 1.0).abs() > 1e-12 {
                let s = c.embedding_scale;
                for v in dst.iter_mut() {
                    *v *= s;
                }
            }
        }
        for l in 0..c.n_layers {
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                None,
                pool,
                kt,
                l,
                seq_len,
                false,
                false,
                self.exact_f32_kv_gradcheck,
            );
        }

        let mut loss_sum = 0.0f64;
        let slot_out = self.bound_lora.slot_for(self.model.output.name);
        let inv_logit = c.inv_logit_scale;
        let mut c0 = 0usize;
        while c0 < n_valid {
            let b = (n_valid - c0).min(CE_CHUNK_TOKENS);
            for k in 0..b {
                let t = self.arena.ce_pos[c0 + k].0;
                rms_norm(
                    &mut self.arena.ce_x_norm[k * d..(k + 1) * d],
                    &self.arena.x_cur[t * d..(t + 1) * d],
                    &self.model.output_norm,
                    c.rms_eps,
                );
            }
            matmul_f32(
                pool,
                kt,
                self.model.output.dtype,
                self.model.output.bytes,
                &self.arena.ce_x_norm[..b * d],
                &mut self.arena.mv_scratch,
                &mut self.arena.ce_logits[..b * vocab],
                vocab,
                d,
                b,
            );
            self.bound_lora.apply_linear(
                slot_out,
                pool,
                kt,
                &self.arena.ce_x_norm[..b * d],
                &mut self.arena.ce_logits[..b * vocab],
                b,
                &mut self.arena.lora_scratch,
            );
            if let Some(ref ob) = self.model.output_bias {
                for k in 0..b {
                    let row = &mut self.arena.ce_logits[k * vocab..(k + 1) * vocab];
                    for (y, &bi) in row.iter_mut().zip(ob.iter()) {
                        *y += bi;
                    }
                }
            }
            if (inv_logit - 1.0).abs() > 1e-12 {
                for lv in &mut self.arena.ce_logits[..b * vocab] {
                    *lv *= inv_logit;
                }
            }
            for k in 0..b {
                let target = self.arena.ce_pos[c0 + k].1 as usize;
                let row = &self.arena.ce_logits[k * vocab..(k + 1) * vocab];
                let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f64 = row.iter().map(|&lv| ((lv - max_l) as f64).exp()).sum();
                let log_z = (max_l as f64) + sum_exp.ln();
                loss_sum += log_z - (row[target] as f64);
            }
            c0 += b;
        }
        loss_sum / (n_valid as f64)
    }
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_f32_kv_exact(
    out: &mut [f32],
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    scale: f64,
    window: Option<usize>,
) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;

    for t in 0..n_tok {
        let end_pos = start_pos + t + 1;
        let start_kv = match window {
            Some(w) if w > 0 && end_pos > w => end_pos - w,
            _ => 0,
        };
        let win_len = end_pos - start_kv;
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let oh = &mut out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let mut scores = vec![0.0f64; win_len];
            let mut max_s = f64::NEG_INFINITY;
            for (idx, sc) in scores.iter_mut().enumerate() {
                let step = start_kv + idx;
                let kh =
                    &k_all[step * kv_dim + kv_h * head_dim..step * kv_dim + (kv_h + 1) * head_dim];
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (qh[d] as f64) * (kh[d] as f64);
                }
                *sc = dot * scale;
                if *sc > max_s {
                    max_s = *sc;
                }
            }
            let mut sum = 0.0f64;
            for sc in &mut scores {
                *sc = (*sc - max_s).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for (idx, &e) in scores.iter().enumerate() {
                    let step = start_kv + idx;
                    acc += (e * inv) * (v_all[step * kv_dim + kv_h * head_dim + d] as f64);
                }
                oh[d] = acc as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_f32_kv_bwd_exact(
    kt: &KernelTable,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    d_out: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_tok: usize,
    scale: f64,
    window: Option<usize>,
    probs: &mut [f64],
    dp: &mut [f64],
) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    dq.fill(0.0);
    dk.fill(0.0);
    dv.fill(0.0);
    let group = n_heads / n_kv_heads;

    for t in 0..n_tok {
        let end_pos = t + 1;
        let start_kv = match window {
            Some(w) if w > 0 && end_pos > w => end_pos - w,
            _ => 0,
        };
        let win_len = end_pos - start_kv;
        let p = &mut probs[..win_len];
        let d_prob = &mut dp[..win_len];
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let doh = &d_out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let mut max_s = f64::NEG_INFINITY;
            for (idx, ps) in p.iter_mut().enumerate() {
                let step = start_kv + idx;
                let kh =
                    &k_all[step * kv_dim + kv_h * head_dim..step * kv_dim + (kv_h + 1) * head_dim];
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (qh[d] as f64) * (kh[d] as f64);
                }
                *ps = dot * scale;
                if *ps > max_s {
                    max_s = *ps;
                }
            }
            let mut sum = 0.0f64;
            for ps in p.iter_mut() {
                *ps = (*ps - max_s).exp();
                sum += *ps;
            }
            let inv = 1.0 / sum;
            for ps in p.iter_mut() {
                *ps *= inv;
            }

            let mut sum_p_dp = 0.0f64;
            for idx in 0..win_len {
                let step = start_kv + idx;
                let pw = p[idx];
                let v_off = step * kv_dim + kv_h * head_dim;
                let vh = &v_all[v_off..v_off + head_dim];
                let dvh = &mut dv[v_off..v_off + head_dim];
                let mut dot_dov = 0.0f64;
                for d in 0..head_dim {
                    dvh[d] += (doh[d] as f64 * pw) as f32;
                    dot_dov += (doh[d] as f64) * (vh[d] as f64);
                }
                d_prob[idx] = dot_dov;
                sum_p_dp += pw * dot_dov;
            }

            let dqh = &mut dq[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            for idx in 0..win_len {
                let step = start_kv + idx;
                let ds = (p[idx] * (d_prob[idx] - sum_p_dp) * scale) as f32;
                if ds == 0.0 {
                    continue;
                }
                let k_off = step * kv_dim + kv_h * head_dim;
                (kt.axpy_f32)(dqh, ds, &k_all[k_off..k_off + head_dim]);
                (kt.axpy_f32)(&mut dk[k_off..k_off + head_dim], ds, qh);
            }
        }
    }
}

/// Runs a complete QLoRA fine-tuning loop with gradient accumulation, warmup+cosine LR,
/// global L2 grad clipping, optional periodic atomic checkpointing, and optional thermal/step callback.
pub fn run_qlora_training<F>(
    model: &LlamaModel<'_>,
    pool: &ThreadPool,
    dataset: &TrainDataset,
    cfg: &QloraConfig,
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
        estimate_qlora_dirty_bytes(model, max_seq, cfg.rank, &cfg.targets, cfg.checkpoint_mode);
    check_training_memory_budget(est_bytes)?;

    let (mut adapter, mut opt) = if let Some(ref resume_p) = cfg.resume_ckpt {
        load_train_checkpoint(resume_p, cfg.optim)?
    } else {
        let ad = init_lora_adapter(model, cfg.rank, cfg.alpha, &cfg.targets, cfg.seed)?;
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
    let mut trainer = LlamaTrainer::new(model, max_seq, max_rank, cfg.checkpoint_mode);

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
            let (l, n_sup) = trainer.forward_backward_microbatch(
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
