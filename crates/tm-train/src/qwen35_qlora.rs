//! Phase `T3`: Hybrid `Qwen3.5` (`Gated DeltaNet` + `Gated Full-Attention` + SwiGLU) QLoRA trainer (`ADR-015`).
//!
//! Strictly enforces every `GDN_CONTRACT.md` & `AGENTS.md` §9 invariant in both forward and backward:
//! 1. Causal `conv1d` over `qkv` followed by **SiLU BEFORE splitting** `q`, `k`, `v`.
//! 2. Per-head L2-normalization on `q` and `k` ONLY (**never** `v`).
//! 3. Tiled head grouping `kh = vh % n_k_heads` (gradients accumulate across `vh` sharing `kh`).
//! 4. Output query scaled by `1 / sqrt(head_v_dim)` on the updated recurrent state `S_t`.
//! 5. Per-head gated RMSNorm with **SiLU** on `z` (`FullAttn` layers use `sigmoid` on `gate`).
//! 6. Reverse-time GDN scan uses `G = 8`-token state checkpoints (`S_{0, 8, 16, ...}`) so backward
//!    never divides by small decay gates `g_t` and keeps memory strictly bounded.

use tm_core::f16_to_f32;
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool, active_kernels,
    gdn_conv1d_silu_split_parallel, gdn_recurrence_gated_parallel,
    gqa_attention_f16_kv_bwd_ext_f32, gqa_attention_f16_kv_ext_batch_parallel,
    gqa_attention_f16_kv_ext_parallel, l2_norm_bwd_f32, matmul_pair_f32, matmul_triple_f32,
    matvec_pair_f32, matvec_triple_f32, rms_norm, rms_norm_batch_parallel, rms_norm_bwd_f32,
    rope_bwd_inplace_tables, rope_compute_cos_sin, rope_inplace_tables, swiglu_bwd_f32,
    swiglu_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch, LoraSet};
use tm_models::{Qwen35AttnWeights, Qwen35Model, decode_embedding_row};

use crate::accel::{
    LinearAccel, embed_lora_backward, linear_bwd_with_accel, linear_fwd_with_accel,
    masked_chunked_ce_fwd_bwd, masked_chunked_ce_loss_f64,
};
use crate::lfm2_qlora::{gqa_attn_f32_bwd_ext, gqa_attn_f32_ext};
use crate::optim::{LoraGradStore, LoraTargets};
use crate::qlora::{CE_CHUNK_TOKENS, CheckpointMode};

const GDN_STATE_CKPT_STRIDE: usize = 8;

#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct Qwen35TrainArena {
    max_seq: usize,
    ckpt_mode: CheckpointMode,
    ckpt_f16: Vec<u16>,
    ckpt_f32: Vec<f32>,

    x_cur: Vec<f32>,
    x_layer_in: Vec<f32>,
    dx: Vec<f32>,

    // Shared / FullAttn / GDN forward recompute buffers
    x_attn_norm: Vec<f32>,
    qkv_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    alpha_raw: Vec<f32>,
    beta_raw: Vec<f32>,
    g_exp_buf: Vec<f32>,
    beta_sig_buf: Vec<f32>,
    conv_sum: Vec<f32>,
    q_pre_norm: Vec<f32>,
    k_pre_norm: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    k_f16: Vec<u16>,
    v_f16: Vec<u16>,
    attn_raw: Vec<f32>,
    attn_out: Vec<f32>,
    x_mid: Vec<f32>,
    x_ffn_norm: Vec<f32>,
    gate_pre: Vec<f32>,
    up_pre: Vec<f32>,
    swiglu_out: Vec<f32>,

    // GDN state scratch + `G = 8`-token block checkpoints
    ssm_state_work: Vec<f32>,
    conv_hist_work: Vec<f32>,
    ssm_ckpts: Vec<f32>,
    ssm_local_steps: Vec<f32>,
    o_raw_local: Vec<f32>,
    d_ssm: Vec<f32>,

    // Backward gradient workspaces
    xb2_chunk: Vec<f32>,
    d_swiglu: Vec<f32>,
    d_gate: Vec<f32>,
    d_up: Vec<f32>,
    d_x_norm: Vec<f32>,
    d_attn_out: Vec<f32>,
    d_attn_raw: Vec<f32>,
    d_gate_z: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dq_act: Vec<f32>,
    dk_act: Vec<f32>,
    d_alpha: Vec<f32>,
    d_beta_raw: Vec<f32>,
    d_qkv: Vec<f32>,
    tmp_embd: Vec<f32>,
    tmp_head: Vec<f32>,
    tmp_s_norm: Vec<f32>,
    tmp_do_raw: Vec<f32>,

    // Masked chunked CE
    ce_pos: Vec<(usize, u32)>,
    ce_x_sub: Vec<f32>,
    ce_x_norm: Vec<f32>,
    ce_logits: Vec<f32>,
    ce_dx_norm: Vec<f32>,

    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    attn_scores: Vec<f32>,
    mv_scratch: MatvecScratch,
    lora_scratch: LoraScratch,
    lora_h: Vec<f32>,
    lora_dh: Vec<f32>,
    lora_bt: Vec<f32>,
}

impl Qwen35TrainArena {
    fn new(
        model: &Qwen35Model<'_>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let c = &model.cfg;
        let d = c.n_embd;
        let ff = c.n_ff;
        let q_dim = c.n_head * c.head_dim;
        let qg_dim = q_dim * 2;
        let kv_dim = c.n_head_kv * c.head_dim;
        let conv_dim = c.conv_dim();
        let k_gdn_dim = c.ssm_n_group * c.ssm_state_size;
        let v_gdn_dim = c.ssm_d_inner;
        let hv = c.ssm_dt_rank;
        let s_dim = c.ssm_state_size;
        let hist_len = c.ssm_conv_kernel.saturating_sub(1);
        let state_elems = hv * s_dim * c.ssm_head_v_dim();
        let n_ssm_ckpts = max_seq.div_ceil(GDN_STATE_CKPT_STRIDE) + 1;

        let max_qkv = qg_dim.max(conv_dim);
        let max_q = q_dim.max(k_gdn_dim);
        let max_k = kv_dim.max(k_gdn_dim);
        let max_v = kv_dim.max(v_gdn_dim);
        let max_gate = q_dim.max(v_gdn_dim);
        let max_cols = d
            .max(ff)
            .max(q_dim)
            .max(v_gdn_dim)
            .max(max_qkv)
            .max(c.vocab_size);
        let r_cap = max_rank.max(1);
        let max_d_out = max_qkv.max(ff).max(d).max(c.vocab_size);

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
            qkv_buf: vec![0.0; max_seq * max_qkv],
            gate_buf: vec![0.0; max_seq * max_gate],
            alpha_raw: vec![0.0; max_seq * hv.max(1)],
            beta_raw: vec![0.0; max_seq * hv.max(1)],
            g_exp_buf: vec![0.0; max_seq * hv.max(1)],
            beta_sig_buf: vec![0.0; max_seq * hv.max(1)],
            conv_sum: vec![0.0; max_seq * conv_dim.max(1)],
            q_pre_norm: vec![0.0; max_seq * max_q.max(1)],
            k_pre_norm: vec![0.0; max_seq * max_k.max(1)],
            q_buf: vec![0.0; max_seq * max_q.max(1)],
            k_buf: vec![0.0; max_seq * max_k.max(1)],
            v_buf: vec![0.0; max_seq * max_v.max(1)],
            k_f16: vec![0u16; max_seq * kv_dim.max(1)],
            v_f16: vec![0u16; max_seq * kv_dim.max(1)],
            attn_raw: vec![0.0; max_seq * q_dim.max(1)],
            attn_out: vec![0.0; max_seq * max_gate.max(1)],
            x_mid: vec![0.0; max_seq * d],
            x_ffn_norm: vec![0.0; max_seq * d],
            gate_pre: vec![0.0; max_seq * ff],
            up_pre: vec![0.0; max_seq * ff],
            swiglu_out: vec![0.0; max_seq * ff],
            ssm_state_work: vec![0.0; state_elems.max(1)],
            conv_hist_work: vec![0.0; (conv_dim * hist_len).max(1)],
            ssm_ckpts: vec![0.0; n_ssm_ckpts * state_elems.max(1)],
            ssm_local_steps: vec![0.0; (GDN_STATE_CKPT_STRIDE + 1) * state_elems.max(1)],
            o_raw_local: vec![0.0; GDN_STATE_CKPT_STRIDE * v_gdn_dim.max(1)],
            d_ssm: vec![0.0; state_elems.max(1)],
            xb2_chunk: vec![0.0; MAX_CPU_BATCH * d.max(max_gate)],
            d_swiglu: vec![0.0; max_seq * ff],
            d_gate: vec![0.0; max_seq * ff],
            d_up: vec![0.0; max_seq * ff],
            d_x_norm: vec![0.0; max_seq * d],
            d_attn_out: vec![0.0; max_seq * max_gate.max(1)],
            d_attn_raw: vec![0.0; max_seq * q_dim.max(1)],
            d_gate_z: vec![0.0; max_seq * max_gate.max(1)],
            dq: vec![0.0; max_seq * max_q.max(1)],
            dk: vec![0.0; max_seq * max_k.max(1)],
            dv: vec![0.0; max_seq * max_v.max(1)],
            dq_act: vec![0.0; max_seq * k_gdn_dim.max(1)],
            dk_act: vec![0.0; max_seq * k_gdn_dim.max(1)],
            d_alpha: vec![0.0; max_seq * hv.max(1)],
            d_beta_raw: vec![0.0; max_seq * hv.max(1)],
            d_qkv: vec![0.0; max_seq * max_qkv.max(1)],
            tmp_embd: vec![0.0; d],
            tmp_head: vec![0.0; c.head_dim.max(s_dim).max(1)],
            tmp_s_norm: vec![0.0; s_dim.max(1)],
            tmp_do_raw: vec![0.0; s_dim.max(1)],
            ce_pos: Vec::with_capacity(max_seq),
            ce_x_sub: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_x_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_logits: vec![0.0; CE_CHUNK_TOKENS * c.vocab_size],
            ce_dx_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            rope_cos: vec![0.0; (c.rope_dim / 2).max(1)],
            rope_sin: vec![0.0; (c.rope_dim / 2).max(1)],
            attn_scores: vec![0.0; c.n_head.max(1) * max_seq],
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

pub fn estimate_qwen35_qlora_dirty_bytes(
    model: &Qwen35Model<'_>,
    max_seq: usize,
    rank: usize,
    targets: &LoraTargets,
    ckpt_mode: CheckpointMode,
) -> u64 {
    let c = &model.cfg;
    let d = c.n_embd;
    let ff = c.n_ff;
    let q_dim = c.n_head * c.head_dim;
    let conv_dim = c.conv_dim();
    let v_gdn_dim = c.ssm_d_inner;
    let state_elems = c.ssm_dt_rank * c.ssm_state_size * c.ssm_head_v_dim();
    let n_ssm_ckpts = max_seq.div_ceil(GDN_STATE_CKPT_STRIDE) + GDN_STATE_CKPT_STRIDE + 3;

    let ckpt_elem = match ckpt_mode {
        CheckpointMode::Fp16 => 2usize,
        CheckpointMode::Fp32 => 4usize,
    };
    let ckpt_bytes = c.n_layers * max_seq * d * ckpt_elem;
    let per_tok_floats = 6 * d + 3 * conv_dim.max(2 * q_dim) + 4 * v_gdn_dim.max(q_dim) + 6 * ff;
    let arena_bytes = max_seq * per_tok_floats * 4
        + n_ssm_ckpts * state_elems * 4
        + CE_CHUNK_TOKENS * (c.vocab_size + 3 * d) * 4;

    let mut lora_params = 0usize;
    for lw in &model.layers {
        match &lw.attn {
            Qwen35AttnWeights::Gdn {
                wqkv,
                wz,
                w_alpha,
                w_out,
                ..
            } => {
                if targets.q || targets.k || targets.v {
                    lora_params += rank * (wqkv.cols + wqkv.rows);
                }
                if targets.ssm_gates {
                    lora_params += rank * (wz.cols + wz.rows);
                    let r_s = rank.min(w_alpha.rows).max(1);
                    lora_params += r_s * (w_alpha.cols + w_alpha.rows) * 2;
                }
                if targets.o {
                    lora_params += rank * (w_out.cols + w_out.rows);
                }
            }
            Qwen35AttnWeights::FullAttn {
                wqg, wk, wv, wo, ..
            } => {
                if targets.q {
                    lora_params += rank * (wqg.cols + wqg.rows);
                }
                if targets.k {
                    lora_params += rank * (wk.cols + wk.rows);
                }
                if targets.v {
                    lora_params += rank * (wv.cols + wv.rows);
                }
                if targets.o {
                    lora_params += rank * (wo.cols + wo.rows);
                }
            }
        }
        if targets.gate {
            lora_params += rank * (d + ff);
        }
        if targets.up {
            lora_params += rank * (d + ff);
        }
        if targets.down {
            lora_params += rank * (ff + d);
        }
    }
    if targets.embed || targets.lm_head {
        lora_params += rank * (c.vocab_size + d);
    }
    (ckpt_bytes + arena_bytes + lora_params * 24) as u64
}

/// Independent QLoRA training engine for hybrid `Qwen35Model` (`#![forbid(unsafe_code)]`).
pub struct Qwen35Trainer<'a> {
    pub model: &'a Qwen35Model<'a>,
    arena: Qwen35TrainArena,
    bound_lora: BoundLora,
    pub exact_f32_kv_gradcheck: bool,
}

impl<'a> Qwen35Trainer<'a> {
    pub fn new(
        model: &'a Qwen35Model<'a>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let seq_cap = max_seq.max(2);
        Self {
            model,
            arena: Qwen35TrainArena::new(model, seq_cap, max_rank, ckpt_mode),
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

    #[allow(clippy::too_many_arguments)]
    fn run_layer_forward(
        model: &Qwen35Model<'_>,
        bound: &BoundLora,
        arena: &mut Qwen35TrainArena,
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
        let ff = c.n_ff;
        let hd = c.head_dim;
        let q_dim = c.n_head * hd;
        let qg_dim = q_dim * 2;
        let kv_dim = c.n_head_kv * hd;
        let conv_dim = c.conv_dim();
        let hk = c.ssm_n_group;
        let hv = c.ssm_dt_rank;
        let s_dim = c.ssm_state_size;
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = c.ssm_d_inner;
        let d_conv = c.ssm_conv_kernel;
        let hist_len = d_conv.saturating_sub(1);
        let state_elems = hv * s_dim * c.ssm_head_v_dim();

        let (w_gate, w_up, w_down) = match &lw.ffn {
            tm_models::Qwen35FfnWeights::Dense {
                w_gate,
                w_up,
                w_down,
            } => (w_gate, w_up, w_down),
            tm_models::Qwen35FfnWeights::Moe(_) => {
                unreachable!("Qwen35Trainer only supports dense Qwen3.5 models")
            }
        };
        let slot_wg = bound.slot_for(w_gate.name);
        let slot_wu = bound.slot_for(w_up.name);
        let slot_wd = bound.slot_for(w_down.name);

        if matches!(lw.attn, Qwen35AttnWeights::Gdn { .. }) {
            arena.ssm_state_work[..state_elems].fill(0.0);
            if hist_len > 0 {
                arena.conv_hist_work[..conv_dim * hist_len].fill(0.0);
            }
        }

        let single_tok_mode = seq_len == 1;
        let mut offset = 0usize;
        while offset < seq_len {
            let b = (seq_len - offset).min(MAX_CPU_BATCH);
            let x_src: &[f32] = if from_ckpt_input {
                &arena.x_layer_in[offset * d..(offset + b) * d]
            } else {
                &arena.x_cur[offset * d..(offset + b) * d]
            };

            if single_tok_mode {
                rms_norm(
                    &mut arena.x_attn_norm[offset * d..(offset + 1) * d],
                    x_src,
                    &lw.attn_norm,
                    c.rms_eps,
                );
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_attn_norm[offset * d..(offset + b) * d],
                    x_src,
                    &lw.attn_norm,
                    c.rms_eps,
                    d,
                    b,
                );
            }

            let xn_chunk = &arena.x_attn_norm[offset * d..(offset + b) * d];
            match &lw.attn {
                Qwen35AttnWeights::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                } => {
                    let slot_qkv = bound.slot_for(wqkv.name);
                    let slot_wz = bound.slot_for(wz.name);
                    let slot_wa = bound.slot_for(w_alpha.name);
                    let slot_wb = bound.slot_for(w_beta.name);
                    let slot_wo = bound.slot_for(w_out.name);

                    if accel.is_some() {
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            wqkv,
                            bound,
                            slot_qkv,
                            xn_chunk,
                            &mut arena.qkv_buf[offset * conv_dim..(offset + b) * conv_dim],
                            b,
                            single_tok_mode,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            wz,
                            bound,
                            slot_wz,
                            xn_chunk,
                            &mut arena.gate_buf[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                            b,
                            single_tok_mode,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            w_alpha,
                            bound,
                            slot_wa,
                            xn_chunk,
                            &mut arena.alpha_raw[offset * hv..(offset + b) * hv],
                            b,
                            single_tok_mode,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            w_beta,
                            bound,
                            slot_wb,
                            xn_chunk,
                            &mut arena.beta_raw[offset * hv..(offset + b) * hv],
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
                                wqkv.dtype,
                                wqkv.bytes,
                                wqkv.rows,
                                &mut arena.qkv_buf[..conv_dim],
                                wz.dtype,
                                wz.bytes,
                                wz.rows,
                                &mut arena.gate_buf[..v_gdn_dim],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                            );
                            matvec_pair_f32(
                                pool,
                                kt,
                                w_alpha.dtype,
                                w_alpha.bytes,
                                w_alpha.rows,
                                &mut arena.alpha_raw[..hv],
                                w_beta.dtype,
                                w_beta.bytes,
                                w_beta.rows,
                                &mut arena.beta_raw[..hv],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                            );
                        } else {
                            matmul_pair_f32(
                                pool,
                                kt,
                                wqkv.dtype,
                                wqkv.bytes,
                                wqkv.rows,
                                &mut arena.qkv_buf[offset * conv_dim..(offset + b) * conv_dim],
                                wz.dtype,
                                wz.bytes,
                                wz.rows,
                                &mut arena.gate_buf[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                                b,
                            );
                            matmul_pair_f32(
                                pool,
                                kt,
                                w_alpha.dtype,
                                w_alpha.bytes,
                                w_alpha.rows,
                                &mut arena.alpha_raw[offset * hv..(offset + b) * hv],
                                w_beta.dtype,
                                w_beta.bytes,
                                w_beta.rows,
                                &mut arena.beta_raw[offset * hv..(offset + b) * hv],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                                b,
                            );
                        }
                        if !bound.is_empty() {
                            bound.apply_linear(
                                slot_qkv,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.qkv_buf[offset * conv_dim..(offset + b) * conv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wz,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.gate_buf[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wa,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.alpha_raw[offset * hv..(offset + b) * hv],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wb,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.beta_raw[offset * hv..(offset + b) * hv],
                                b,
                                &mut arena.lora_scratch,
                            );
                        }
                    }

                    for t in 0..b {
                        let pos = offset + t;
                        for vh in 0..hv {
                            let idx = pos * hv + vh;
                            let sp = softplus_f32(arena.alpha_raw[idx] + ssm_dt_bias[vh]);
                            arena.g_exp_buf[idx] = (sp * ssm_a[vh]).exp();
                            arena.beta_sig_buf[idx] = sigmoid_f32(arena.beta_raw[idx]);
                        }
                    }

                    gdn_conv1d_silu_split_parallel(
                        pool,
                        kt,
                        &arena.qkv_buf[offset * conv_dim..(offset + b) * conv_dim],
                        &mut arena.conv_hist_work[..conv_dim * hist_len],
                        conv1d,
                        &mut arena.q_buf[offset * k_gdn_dim..(offset + b) * k_gdn_dim],
                        &mut arena.k_buf[offset * k_gdn_dim..(offset + b) * k_gdn_dim],
                        &mut arena.v_buf[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                        d_conv,
                        hk,
                        s_dim,
                        v_gdn_dim,
                        c.rms_eps,
                        b,
                    );

                    gdn_recurrence_gated_parallel(
                        pool,
                        kt,
                        &arena.q_buf[offset * k_gdn_dim..(offset + b) * k_gdn_dim],
                        &arena.k_buf[offset * k_gdn_dim..(offset + b) * k_gdn_dim],
                        &arena.v_buf[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                        &arena.g_exp_buf[offset * hv..(offset + b) * hv],
                        &arena.beta_sig_buf[offset * hv..(offset + b) * hv],
                        &mut arena.ssm_state_work[..state_elems],
                        &mut arena.attn_out[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                        &arena.gate_buf[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                        ssm_norm,
                        c.rms_eps,
                        hk,
                        hv,
                        s_dim,
                        b,
                    );

                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        w_out,
                        bound,
                        slot_wo,
                        &arena.attn_out[offset * v_gdn_dim..(offset + b) * v_gdn_dim],
                        &mut arena.xb2_chunk[..b * d],
                        b,
                        single_tok_mode,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );
                }
                Qwen35AttnWeights::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                } => {
                    let slot_wqg = bound.slot_for(wqg.name);
                    let slot_wk = bound.slot_for(wk.name);
                    let slot_wv = bound.slot_for(wv.name);
                    let slot_wo = bound.slot_for(wo.name);

                    if accel.is_some() {
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            wqg,
                            bound,
                            slot_wqg,
                            xn_chunk,
                            &mut arena.qkv_buf[offset * qg_dim..(offset + b) * qg_dim],
                            b,
                            single_tok_mode,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            wk,
                            bound,
                            slot_wk,
                            xn_chunk,
                            &mut arena.k_buf[offset * kv_dim..(offset + b) * kv_dim],
                            b,
                            single_tok_mode,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            wv,
                            bound,
                            slot_wv,
                            xn_chunk,
                            &mut arena.v_buf[offset * kv_dim..(offset + b) * kv_dim],
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
                                wqg.dtype,
                                wqg.bytes,
                                wqg.rows,
                                &mut arena.qkv_buf[..qg_dim],
                                wk.dtype,
                                wk.bytes,
                                wk.rows,
                                &mut arena.k_buf[..kv_dim],
                                wv.dtype,
                                wv.bytes,
                                wv.rows,
                                &mut arena.v_buf[..kv_dim],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                            );
                        } else {
                            matmul_triple_f32(
                                pool,
                                kt,
                                wqg.dtype,
                                wqg.bytes,
                                wqg.rows,
                                &mut arena.qkv_buf[offset * qg_dim..(offset + b) * qg_dim],
                                wk.dtype,
                                wk.bytes,
                                wk.rows,
                                &mut arena.k_buf[offset * kv_dim..(offset + b) * kv_dim],
                                wv.dtype,
                                wv.bytes,
                                wv.rows,
                                &mut arena.v_buf[offset * kv_dim..(offset + b) * kv_dim],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                                b,
                            );
                        }
                        if !bound.is_empty() {
                            bound.apply_linear(
                                slot_wqg,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.qkv_buf[offset * qg_dim..(offset + b) * qg_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wk,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.k_buf[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wv,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.v_buf[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                        }
                    }

                    if save_recompute {
                        arena.k_pre_norm[offset * kv_dim..(offset + b) * kv_dim]
                            .copy_from_slice(&arena.k_buf[offset * kv_dim..(offset + b) * kv_dim]);
                    }

                    for t in 0..b {
                        let pos = offset + t;
                        let qg_t = &arena.qkv_buf[pos * qg_dim..(pos + 1) * qg_dim];
                        let q_t = &mut arena.q_buf[pos * q_dim..(pos + 1) * q_dim];
                        let g_t = &mut arena.gate_buf[pos * q_dim..(pos + 1) * q_dim];
                        let k_t = &mut arena.k_buf[pos * kv_dim..(pos + 1) * kv_dim];
                        let v_t = &arena.v_buf[pos * kv_dim..(pos + 1) * kv_dim];

                        for h in 0..c.n_head {
                            let src_q = &qg_t[h * 2 * hd..h * 2 * hd + hd];
                            if save_recompute {
                                arena.q_pre_norm[pos * q_dim + h * hd..pos * q_dim + (h + 1) * hd]
                                    .copy_from_slice(src_q);
                            }
                            rms_norm(&mut q_t[h * hd..(h + 1) * hd], src_q, q_norm, c.rms_eps);
                            g_t[h * hd..(h + 1) * hd]
                                .copy_from_slice(&qg_t[h * 2 * hd + hd..(h + 1) * 2 * hd]);
                        }
                        for h in 0..c.n_head_kv {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(&mut arena.tmp_head[..hd], &k_t[s..e], k_norm, c.rms_eps);
                            k_t[s..e].copy_from_slice(&arena.tmp_head[..hd]);
                        }

                        rope_compute_cos_sin(
                            &model.rope_inv_freqs,
                            pos,
                            &mut arena.rope_cos,
                            &mut arena.rope_sin,
                        );
                        rope_inplace_tables(
                            q_t,
                            c.n_head,
                            hd,
                            c.rope_dim,
                            &arena.rope_cos,
                            &arena.rope_sin,
                            RopeType::Neox,
                        );
                        rope_inplace_tables(
                            k_t,
                            c.n_head_kv,
                            hd,
                            c.rope_dim,
                            &arena.rope_cos,
                            &arena.rope_sin,
                            RopeType::Neox,
                        );

                        if !exact_f32_kv {
                            let kv_off = pos * kv_dim;
                            (kt.f32_to_f16_slice)(k_t, &mut arena.k_f16[kv_off..kv_off + kv_dim]);
                            (kt.f32_to_f16_slice)(v_t, &mut arena.v_f16[kv_off..kv_off + kv_dim]);
                        }
                    }

                    let attn_scale = 1.0f32 / (hd as f32).sqrt();
                    if exact_f32_kv {
                        gqa_attn_f32_ext(
                            &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                            &arena.q_buf[offset * q_dim..(offset + b) * q_dim],
                            &arena.k_buf[..(offset + b) * kv_dim],
                            &arena.v_buf[..(offset + b) * kv_dim],
                            c.n_head,
                            c.n_head_kv,
                            hd,
                            offset,
                            b,
                            attn_scale,
                            None,
                        );
                    } else if single_tok_mode {
                        gqa_attention_f16_kv_ext_parallel(
                            pool,
                            kt,
                            &mut arena.attn_out[..q_dim],
                            &arena.q_buf[..q_dim],
                            &arena.k_f16,
                            &arena.v_f16,
                            c.n_head,
                            c.n_head_kv,
                            hd,
                            0,
                            attn_scale,
                            None,
                            arena.max_seq,
                            &mut arena.attn_scores,
                        );
                    } else {
                        gqa_attention_f16_kv_ext_batch_parallel(
                            pool,
                            kt,
                            &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                            &arena.q_buf[offset * q_dim..(offset + b) * q_dim],
                            &arena.k_f16,
                            &arena.v_f16,
                            c.n_head,
                            c.n_head_kv,
                            hd,
                            offset,
                            b,
                            attn_scale,
                            None,
                            arena.max_seq,
                            &mut arena.attn_scores,
                        );
                    }

                    if save_recompute {
                        arena.attn_raw[offset * q_dim..(offset + b) * q_dim]
                            .copy_from_slice(&arena.attn_out[offset * q_dim..(offset + b) * q_dim]);
                    }
                    for (o, &g) in arena.attn_out[offset * q_dim..(offset + b) * q_dim]
                        .iter_mut()
                        .zip(arena.gate_buf[offset * q_dim..(offset + b) * q_dim].iter())
                    {
                        *o *= sigmoid_f32(g);
                    }

                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        wo,
                        bound,
                        slot_wo,
                        &arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                        &mut arena.xb2_chunk[..b * d],
                        b,
                        single_tok_mode,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );
                }
            }

            for i in 0..(b * d) {
                let xin_i = if from_ckpt_input {
                    arena.x_layer_in[offset * d + i]
                } else {
                    arena.x_cur[offset * d + i]
                };
                arena.x_mid[offset * d + i] = xin_i + arena.xb2_chunk[i];
            }

            // Post-attention RMSNorm + SwiGLU FFN
            let xm_chunk = &arena.x_mid[offset * d..(offset + b) * d];
            if single_tok_mode {
                rms_norm(
                    &mut arena.x_ffn_norm[offset * d..(offset + 1) * d],
                    xm_chunk,
                    &lw.post_attn_norm,
                    c.rms_eps,
                );
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_ffn_norm[offset * d..(offset + b) * d],
                    xm_chunk,
                    &lw.post_attn_norm,
                    c.rms_eps,
                    d,
                    b,
                );
            }

            let xfn_chunk = &arena.x_ffn_norm[offset * d..(offset + b) * d];
            if accel.is_some() {
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    w_gate,
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
                    w_up,
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
                        w_gate.dtype,
                        w_gate.bytes,
                        w_gate.rows,
                        &mut arena.gate_pre[..ff],
                        w_up.dtype,
                        w_up.bytes,
                        w_up.rows,
                        &mut arena.up_pre[..ff],
                        xfn_chunk,
                        &mut arena.mv_scratch,
                        d,
                    );
                } else {
                    matmul_pair_f32(
                        pool,
                        kt,
                        w_gate.dtype,
                        w_gate.bytes,
                        w_gate.rows,
                        &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                        w_up.dtype,
                        w_up.bytes,
                        w_up.rows,
                        &mut arena.up_pre[offset * ff..(offset + b) * ff],
                        xfn_chunk,
                        &mut arena.mv_scratch,
                        d,
                        b,
                    );
                }
                if !bound.is_empty() {
                    bound.apply_linear(
                        slot_wg,
                        pool,
                        kt,
                        xfn_chunk,
                        &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                        b,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear(
                        slot_wu,
                        pool,
                        kt,
                        xfn_chunk,
                        &mut arena.up_pre[offset * ff..(offset + b) * ff],
                        b,
                        &mut arena.lora_scratch,
                    );
                }
            }

            arena.swiglu_out[offset * ff..(offset + b) * ff]
                .copy_from_slice(&arena.gate_pre[offset * ff..(offset + b) * ff]);
            swiglu_inplace_parallel(
                pool,
                &mut arena.swiglu_out[offset * ff..(offset + b) * ff],
                &arena.up_pre[offset * ff..(offset + b) * ff],
            );

            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                w_down,
                bound,
                slot_wd,
                &arena.swiglu_out[offset * ff..(offset + b) * ff],
                &mut arena.xb2_chunk[..b * d],
                b,
                single_tok_mode,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );

            if !save_recompute {
                for (i, (&xm_i, &xb2_i)) in xm_chunk
                    .iter()
                    .zip(arena.xb2_chunk[..b * d].iter())
                    .enumerate()
                {
                    arena.x_cur[offset * d + i] = xm_i + xb2_i;
                }
            }

            offset += b;
        }

        // Populate pre-L2 `q_pre_norm`, `k_pre_norm`, `conv_sum`, and `G = 8` GDN state checkpoints
        // when recomputing a GDN layer for backward (`AGENTS.md` §9).
        if save_recompute && let Qwen35AttnWeights::Gdn { conv1d, .. } = &lw.attn {
            for t in 0..seq_len {
                for ch in 0..conv_dim {
                    let w_slice = &conv1d[ch * d_conv..(ch + 1) * d_conv];
                    let mut sum = 0.0f32;
                    for (tap, &wt) in w_slice.iter().enumerate() {
                        if t + tap >= hist_len {
                            let s = t + tap - hist_len;
                            sum += arena.qkv_buf[s * conv_dim + ch] * wt;
                        }
                    }
                    arena.conv_sum[t * conv_dim + ch] = sum;
                    let act = sum / (1.0 + (-sum).exp());
                    if ch < k_gdn_dim {
                        arena.q_pre_norm[t * k_gdn_dim + ch] = act;
                    } else if ch < 2 * k_gdn_dim {
                        arena.k_pre_norm[t * k_gdn_dim + (ch - k_gdn_dim)] = act;
                    }
                }
            }

            let scale = 1.0f32 / (s_dim as f32).sqrt();
            arena.ssm_state_work[..state_elems].fill(0.0);
            let n_blocks = seq_len.div_ceil(GDN_STATE_CKPT_STRIDE);
            for blk in 0..n_blocks {
                arena.ssm_ckpts[blk * state_elems..(blk + 1) * state_elems]
                    .copy_from_slice(&arena.ssm_state_work[..state_elems]);
                let b_start = blk * GDN_STATE_CKPT_STRIDE;
                let b_end = (b_start + GDN_STATE_CKPT_STRIDE).min(seq_len);
                for t in b_start..b_end {
                    for vh in 0..hv {
                        let kh = vh % hk;
                        let q_h = &arena.q_buf
                            [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                        let k_h = &arena.k_buf
                            [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                        let v_h = &arena.v_buf
                            [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];
                        let g = arena.g_exp_buf[t * hv + vh];
                        let beta = arena.beta_sig_buf[t * hv + vh];
                        let s_vh =
                            &mut arena.ssm_state_work[vh * s_dim * s_dim..(vh + 1) * s_dim * s_dim];
                        (kt.gdn_recurrence_head)(
                            q_h,
                            k_h,
                            v_h,
                            g,
                            beta,
                            scale,
                            s_vh,
                            &mut arena.tmp_s_norm[..s_dim],
                            s_dim,
                        );
                    }
                }
            }
        }
    }

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
        let slot_out = self.bound_lora.slot_for(self.model.output.name);
        linear_fwd_with_accel(
            None,
            pool,
            kt,
            &self.model.output,
            &self.bound_lora,
            slot_out,
            &self.arena.tmp_embd,
            out_last_logits,
            1,
            true,
            &mut self.arena.mv_scratch,
            &mut self.arena.lora_scratch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_backward_microbatch(
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
        let ff = c.n_ff;
        let hd = c.head_dim;
        let q_dim = c.n_head * hd;
        let qg_dim = q_dim * 2;
        let kv_dim = c.n_head_kv * hd;
        let conv_dim = c.conv_dim();
        let hk = c.ssm_n_group;
        let hv = c.ssm_dt_rank;
        let s_dim = c.ssm_state_size;
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = c.ssm_d_inner;
        let d_conv = c.ssm_conv_kernel;
        let hist_len = d_conv.saturating_sub(1);
        let state_per_head = s_dim * s_dim;
        let state_elems = hv * state_per_head;

        // 1. Forward pass with per-layer checkpointing
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

        // 2. Masked chunked CE
        let mean_loss = masked_chunked_ce_fwd_bwd(
            accel.as_deref_mut(),
            pool,
            kt,
            &self.model.output_norm,
            c.rms_eps,
            &self.model.output,
            None,
            &self.bound_lora,
            adapter,
            grads,
            &self.arena.x_cur[..seq_len * d],
            &mut self.arena.dx[..seq_len * d],
            &self.arena.ce_pos,
            grad_scale,
            d,
            c.vocab_size,
            &mut self.arena.ce_x_sub,
            &mut self.arena.ce_x_norm,
            &mut self.arena.ce_logits,
            &mut self.arena.ce_dx_norm,
            &mut self.arena.mv_scratch,
            &mut self.arena.lora_scratch,
            &mut self.arena.lora_h,
            &mut self.arena.lora_dh,
            &mut self.arena.lora_bt,
        );

        // 3. Reverse layer loop
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
            let (w_gate, w_up, w_down) = match &lw.ffn {
                tm_models::Qwen35FfnWeights::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => (w_gate, w_up, w_down),
                tm_models::Qwen35FfnWeights::Moe(_) => {
                    unreachable!("Qwen35Trainer only supports dense Qwen3.5 models")
                }
            };

            // 3a. SwiGLU FFN backward (`post_attn_norm`)
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                w_down,
                adapter,
                grads,
                &self.arena.swiglu_out[..seq_len * ff],
                &self.arena.dx[..seq_len * d],
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
                w_gate,
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
                w_up,
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
            for t in 0..seq_len {
                rms_norm_bwd_f32(
                    &mut self.arena.tmp_embd,
                    None,
                    &self.arena.x_mid[t * d..(t + 1) * d],
                    &lw.post_attn_norm,
                    &self.arena.d_x_norm[t * d..(t + 1) * d],
                    c.rms_eps,
                );
                (kt.axpy_f32)(
                    &mut self.arena.dx[t * d..(t + 1) * d],
                    1.0,
                    &self.arena.tmp_embd,
                );
            }

            // 3b. Hybrid GDN or FullAttn backward
            match &lw.attn {
                Qwen35AttnWeights::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                } => {
                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        w_out,
                        adapter,
                        grads,
                        &self.arena.attn_out[..seq_len * v_gdn_dim],
                        &self.arena.dx[..seq_len * d],
                        &mut self.arena.d_attn_out[..seq_len * v_gdn_dim],
                        seq_len,
                        false,
                        &mut self.arena.lora_h,
                        &mut self.arena.lora_dh,
                        &mut self.arena.lora_bt,
                    );

                    self.arena.dq[..seq_len * k_gdn_dim].fill(0.0);
                    self.arena.dk[..seq_len * k_gdn_dim].fill(0.0);
                    self.arena.dv[..seq_len * v_gdn_dim].fill(0.0);
                    self.arena.d_ssm[..state_elems].fill(0.0);

                    let scale = 1.0f32 / (s_dim as f32).sqrt();
                    let n_blocks = seq_len.div_ceil(GDN_STATE_CKPT_STRIDE);

                    for blk in (0..n_blocks).rev() {
                        let b_start = blk * GDN_STATE_CKPT_STRIDE;
                        let b_end = (b_start + GDN_STATE_CKPT_STRIDE).min(seq_len);
                        let b_len = b_end - b_start;

                        // Re-step forward from `ssm_ckpts[blk]` to materialize exact `S_{b_start..=b_end}` and `o_raw`
                        self.arena.ssm_local_steps[..state_elems].copy_from_slice(
                            &self.arena.ssm_ckpts[blk * state_elems..(blk + 1) * state_elems],
                        );
                        for lt in 0..b_len {
                            let t = b_start + lt;
                            let (prev_slice, next_rest) = self
                                .arena
                                .ssm_local_steps
                                .split_at_mut((lt + 1) * state_elems);
                            let s_cur_src = &prev_slice[lt * state_elems..(lt + 1) * state_elems];
                            let s_next_dst = &mut next_rest[..state_elems];
                            s_next_dst.copy_from_slice(s_cur_src);

                            for vh in 0..hv {
                                let kh = vh % hk;
                                let q_h = &self.arena.q_buf
                                    [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                                let k_h = &self.arena.k_buf
                                    [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                                let v_h = &self.arena.v_buf
                                    [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];
                                let g = self.arena.g_exp_buf[t * hv + vh];
                                let beta = self.arena.beta_sig_buf[t * hv + vh];
                                let s_vh =
                                    &mut s_next_dst[vh * state_per_head..(vh + 1) * state_per_head];
                                let o_dst = &mut self.arena.o_raw_local[lt * v_gdn_dim + vh * s_dim
                                    ..lt * v_gdn_dim + (vh + 1) * s_dim];
                                (kt.gdn_recurrence_head)(
                                    q_h, k_h, v_h, g, beta, scale, s_vh, o_dst, s_dim,
                                );
                            }
                        }

                        // Reverse-time scan across `lt = (b_len - 1) ..= 0`
                        for lt in (0..b_len).rev() {
                            let t = b_start + lt;
                            for vh in 0..hv {
                                let kh = vh % hk;
                                let o_raw = &self.arena.o_raw_local[lt * v_gdn_dim + vh * s_dim
                                    ..lt * v_gdn_dim + (vh + 1) * s_dim];
                                let z_h = &self.arena.gate_buf
                                    [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];
                                let dout_h = &self.arena.d_attn_out
                                    [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];

                                // 1. Gated RMSNorm * SiLU(z) backward (`AGENTS.md` §9: Qwen3.5 uses SiLU on `z`)
                                rms_norm(
                                    &mut self.arena.tmp_s_norm[..s_dim],
                                    o_raw,
                                    ssm_norm,
                                    c.rms_eps,
                                );
                                let dz_h = &mut self.arena.d_gate_z
                                    [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];
                                for j in 0..s_dim {
                                    let z = z_h[j] as f64;
                                    let dy = dout_h[j] as f64;
                                    let sig = 1.0 / (1.0 + (-z).exp());
                                    let silu_z = z * sig;
                                    let dsilu_z = sig * (1.0 + z * (1.0 - sig));
                                    dz_h[j] =
                                        (dy * (self.arena.tmp_s_norm[j] as f64) * dsilu_z) as f32;
                                    // Reuse `tmp_head[j]` as `d_norm_o[j]`
                                    self.arena.tmp_head[j] = (dy * silu_z) as f32;
                                }
                                rms_norm_bwd_f32(
                                    &mut self.arena.tmp_do_raw[..s_dim],
                                    None,
                                    o_raw,
                                    ssm_norm,
                                    &self.arena.tmp_head[..s_dim],
                                    c.rms_eps,
                                );

                                // 2. Single-head GDN recurrence reverse step
                                let q_h = &self.arena.q_buf
                                    [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                                let k_h = &self.arena.k_buf
                                    [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                                let v_h = &self.arena.v_buf
                                    [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];
                                let g = self.arena.g_exp_buf[t * hv + vh] as f64;
                                let beta = self.arena.beta_sig_buf[t * hv + vh] as f64;

                                let s_prev_all = &self.arena.ssm_local_steps[lt * state_elems
                                    + vh * state_per_head
                                    ..lt * state_elems + (vh + 1) * state_per_head];
                                let s_new_all = &self.arena.ssm_local_steps[(lt + 1) * state_elems
                                    + vh * state_per_head
                                    ..(lt + 1) * state_elems + (vh + 1) * state_per_head];
                                let ds_vh = &mut self.arena.d_ssm
                                    [vh * state_per_head..(vh + 1) * state_per_head];

                                let dq_kh = &mut self.arena.dq
                                    [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                                let dk_kh = &mut self.arena.dk
                                    [t * k_gdn_dim + kh * s_dim..t * k_gdn_dim + (kh + 1) * s_dim];
                                let dv_vh = &mut self.arena.dv
                                    [t * v_gdn_dim + vh * s_dim..t * v_gdn_dim + (vh + 1) * s_dim];

                                let mut dg_acc = 0.0f64;
                                let mut dbeta_acc = 0.0f64;

                                for j in 0..s_dim {
                                    let s_prev_row = &s_prev_all[j * s_dim..(j + 1) * s_dim];
                                    let s_new_row = &s_new_all[j * s_dim..(j + 1) * s_dim];
                                    let ds_row = &mut ds_vh[j * s_dim..(j + 1) * s_dim];
                                    let do_s = (self.arena.tmp_do_raw[j] as f64) * (scale as f64);

                                    let mut sum_kv = 0.0f64;
                                    for i in 0..s_dim {
                                        sum_kv += ((s_prev_row[i] as f64) * g) * (k_h[i] as f64);
                                    }
                                    let diff = (v_h[j] as f64) - sum_kv;
                                    let delta_j = diff * beta;

                                    let mut d_delta = 0.0f64;
                                    for i in 0..s_dim {
                                        dq_kh[i] += (do_s * (s_new_row[i] as f64)) as f32;
                                        let g_tot = (ds_row[i] as f64) + do_s * (q_h[i] as f64);
                                        ds_row[i] = g_tot as f32;
                                        d_delta += g_tot * (k_h[i] as f64);
                                    }

                                    dv_vh[j] = (d_delta * beta) as f32;
                                    dbeta_acc += d_delta * diff;
                                    let d_sum_kv = -d_delta * beta;

                                    for i in 0..s_dim {
                                        let g_tot = ds_row[i] as f64;
                                        let ki = k_h[i] as f64;
                                        let sp_i = s_prev_row[i] as f64;
                                        dk_kh[i] +=
                                            (g_tot * delta_j + d_sum_kv * (sp_i * g)) as f32;
                                        let d_dold = g_tot + d_sum_kv * ki;
                                        dg_acc += d_dold * sp_i;
                                        ds_row[i] = (d_dold * g) as f32;
                                    }
                                }

                                // 3. Scalar gate backward (`alpha_raw`, `beta_raw`)
                                let idx = t * hv + vh;
                                let u_sp = self.arena.alpha_raw[idx] + ssm_dt_bias[vh];
                                let dsp = if u_sp > 20.0 {
                                    1.0f64
                                } else {
                                    1.0 / (1.0 + (-u_sp as f64).exp())
                                };
                                self.arena.d_alpha[idx] =
                                    (dg_acc * g * (ssm_a[vh] as f64) * dsp) as f32;
                                self.arena.d_beta_raw[idx] =
                                    (dbeta_acc * beta * (1.0 - beta)) as f32;
                            }
                        }
                    }

                    // 4. Per-head L2-norm backward on `q` and `k` ONLY (`v` is NOT L2-normalized!)
                    for t in 0..seq_len {
                        for kh in 0..hk {
                            let off = t * k_gdn_dim + kh * s_dim;
                            l2_norm_bwd_f32(
                                &mut self.arena.dq_act[off..off + s_dim],
                                &self.arena.q_pre_norm[off..off + s_dim],
                                &self.arena.dq[off..off + s_dim],
                                c.rms_eps,
                            );
                            l2_norm_bwd_f32(
                                &mut self.arena.dk_act[off..off + s_dim],
                                &self.arena.k_pre_norm[off..off + s_dim],
                                &self.arena.dk[off..off + s_dim],
                                c.rms_eps,
                            );
                        }
                    }

                    // 5. SiLU BEFORE q/k/v split + causal depthwise `conv1d` backward
                    // Reuse `conv_sum` in-place to hold `d_conv_sum` after applying `dsilu`:
                    for t in 0..seq_len {
                        for ch in 0..conv_dim {
                            let d_act = if ch < k_gdn_dim {
                                self.arena.dq_act[t * k_gdn_dim + ch]
                            } else if ch < 2 * k_gdn_dim {
                                self.arena.dk_act[t * k_gdn_dim + (ch - k_gdn_dim)]
                            } else {
                                self.arena.dv[t * v_gdn_dim + (ch - 2 * k_gdn_dim)]
                            } as f64;
                            let cs = self.arena.conv_sum[t * conv_dim + ch] as f64;
                            let sig = 1.0 / (1.0 + (-cs).exp());
                            let dsilu = sig * (1.0 + cs * (1.0 - sig));
                            self.arena.conv_sum[t * conv_dim + ch] = (d_act * dsilu) as f32;
                        }
                    }

                    for s in 0..seq_len {
                        let dqkv_s = &mut self.arena.d_qkv[s * conv_dim..(s + 1) * conv_dim];
                        for ch in 0..conv_dim {
                            let w_slice = &conv1d[ch * d_conv..(ch + 1) * d_conv];
                            let mut acc = 0.0f64;
                            for (tap, &wt) in w_slice.iter().enumerate() {
                                if s + hist_len >= tap {
                                    let t = s + hist_len - tap;
                                    if t < seq_len {
                                        acc += (self.arena.conv_sum[t * conv_dim + ch] as f64)
                                            * (wt as f64);
                                    }
                                }
                            }
                            dqkv_s[ch] = acc as f32;
                        }
                    }

                    // 6. Linear projections backward (`wqkv`, `wz`, `w_alpha`, `w_beta`)
                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        wqkv,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.d_qkv[..seq_len * conv_dim],
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
                        wz,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.d_gate_z[..seq_len * v_gdn_dim],
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
                        w_alpha,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.d_alpha[..seq_len * hv],
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
                        w_beta,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.d_beta_raw[..seq_len * hv],
                        &mut self.arena.d_x_norm[..seq_len * d],
                        seq_len,
                        true,
                        &mut self.arena.lora_h,
                        &mut self.arena.lora_dh,
                        &mut self.arena.lora_bt,
                    );
                }
                Qwen35AttnWeights::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                } => {
                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        wo,
                        adapter,
                        grads,
                        &self.arena.attn_out[..seq_len * q_dim],
                        &self.arena.dx[..seq_len * d],
                        &mut self.arena.d_attn_out[..seq_len * q_dim],
                        seq_len,
                        false,
                        &mut self.arena.lora_h,
                        &mut self.arena.lora_dh,
                        &mut self.arena.lora_bt,
                    );

                    // FullAttn output gate uses `sigmoid` (`attn_out = attn_raw * sigmoid(gate)`)
                    for i in 0..(seq_len * q_dim) {
                        let dy = self.arena.d_attn_out[i] as f64;
                        let g = self.arena.gate_buf[i] as f64;
                        let raw = self.arena.attn_raw[i] as f64;
                        let sig = 1.0 / (1.0 + (-g).exp());
                        self.arena.d_attn_raw[i] = (dy * sig) as f32;
                        self.arena.d_gate_z[i] = (dy * raw * sig * (1.0 - sig)) as f32;
                    }

                    let attn_scale = 1.0f32 / (hd as f32).sqrt();
                    if self.exact_f32_kv_gradcheck {
                        gqa_attn_f32_bwd_ext(
                            kt,
                            &mut self.arena.dq[..seq_len * q_dim],
                            &mut self.arena.dk[..seq_len * kv_dim],
                            &mut self.arena.dv[..seq_len * kv_dim],
                            &self.arena.q_buf[..seq_len * q_dim],
                            &self.arena.k_buf[..seq_len * kv_dim],
                            &self.arena.v_buf[..seq_len * kv_dim],
                            &self.arena.d_attn_raw[..seq_len * q_dim],
                            c.n_head,
                            c.n_head_kv,
                            hd,
                            seq_len,
                            attn_scale,
                            None,
                        );
                    } else {
                        gqa_attention_f16_kv_bwd_ext_f32(
                            kt,
                            &mut self.arena.dq[..seq_len * q_dim],
                            &mut self.arena.dk[..seq_len * kv_dim],
                            &mut self.arena.dv[..seq_len * kv_dim],
                            &self.arena.q_buf[..seq_len * q_dim],
                            &self.arena.k_f16[..seq_len * kv_dim],
                            &self.arena.v_f16[..seq_len * kv_dim],
                            &self.arena.d_attn_raw[..seq_len * q_dim],
                            c.n_head,
                            c.n_head_kv,
                            hd,
                            seq_len,
                            attn_scale,
                            None,
                        );
                    }

                    for t in 0..seq_len {
                        rope_compute_cos_sin(
                            &self.model.rope_inv_freqs,
                            t,
                            &mut self.arena.rope_cos,
                            &mut self.arena.rope_sin,
                        );
                        let dqt = &mut self.arena.dq[t * q_dim..(t + 1) * q_dim];
                        let dkt = &mut self.arena.dk[t * kv_dim..(t + 1) * kv_dim];
                        rope_bwd_inplace_tables(
                            dqt,
                            c.n_head,
                            hd,
                            c.rope_dim,
                            &self.arena.rope_cos,
                            &self.arena.rope_sin,
                            RopeType::Neox,
                        );
                        rope_bwd_inplace_tables(
                            dkt,
                            c.n_head_kv,
                            hd,
                            c.rope_dim,
                            &self.arena.rope_cos,
                            &self.arena.rope_sin,
                            RopeType::Neox,
                        );

                        let q_pre_t = &self.arena.q_pre_norm[t * q_dim..(t + 1) * q_dim];
                        let dg_t = &self.arena.d_gate_z[t * q_dim..(t + 1) * q_dim];
                        let dqg_t = &mut self.arena.d_qkv[t * qg_dim..(t + 1) * qg_dim];
                        for h in 0..c.n_head {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm_bwd_f32(
                                &mut dqg_t[h * 2 * hd..h * 2 * hd + hd],
                                None,
                                &q_pre_t[s..e],
                                q_norm,
                                &dqt[s..e],
                                c.rms_eps,
                            );
                            dqg_t[h * 2 * hd + hd..(h + 1) * 2 * hd].copy_from_slice(&dg_t[s..e]);
                        }

                        let k_pre_t = &self.arena.k_pre_norm[t * kv_dim..(t + 1) * kv_dim];
                        for h in 0..c.n_head_kv {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm_bwd_f32(
                                &mut self.arena.tmp_head[..hd],
                                None,
                                &k_pre_t[s..e],
                                k_norm,
                                &dkt[s..e],
                                c.rms_eps,
                            );
                            dkt[s..e].copy_from_slice(&self.arena.tmp_head[..hd]);
                        }
                    }

                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        wqg,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.d_qkv[..seq_len * qg_dim],
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
                        wk,
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
                        wv,
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
                }
            }

            for t in 0..seq_len {
                rms_norm_bwd_f32(
                    &mut self.arena.tmp_embd,
                    None,
                    &self.arena.x_layer_in[t * d..(t + 1) * d],
                    &lw.attn_norm,
                    &self.arena.d_x_norm[t * d..(t + 1) * d],
                    c.rms_eps,
                );
                (kt.axpy_f32)(
                    &mut self.arena.dx[t * d..(t + 1) * d],
                    1.0,
                    &self.arena.tmp_embd,
                );
            }
        }

        embed_lora_backward(
            self.model.token_embd.name,
            adapter,
            grads,
            input_toks,
            &self.arena.dx[..seq_len * d],
            d,
            1.0,
        );

        (mean_loss, n_valid)
    }

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
        if self.arena.ce_pos.is_empty() {
            return 0.0;
        }
        self.sync_bound_lora(adapter);
        let kt = active_kernels();
        let c = &self.model.cfg;
        let d = c.n_embd;
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
        masked_chunked_ce_loss_f64(
            None,
            pool,
            kt,
            &self.model.output_norm,
            c.rms_eps,
            &self.model.output,
            None,
            &self.bound_lora,
            &self.arena.x_cur[..seq_len * d],
            &self.arena.ce_pos,
            d,
            c.vocab_size,
            &mut self.arena.ce_x_norm,
            &mut self.arena.ce_logits,
            &mut self.arena.mv_scratch,
            &mut self.arena.lora_scratch,
        )
    }
}
