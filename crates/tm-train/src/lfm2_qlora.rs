//! Phase `T3`: Hybrid `LFM2` (`ShortConv` + `GQA` + SwiGLU) QLoRA trainer (`ADR-015`).
//!
//! Guarantees:
//! - **0-ULP Forward Parity** with `Lfm2Model::forward_batch` & `forward_token` when `accel` is `None`.
//! - **Exact `ShortConv` & `GQA` Analytical Backward**: Backpropagates through gated causal
//!   depthwise `ShortConv` (`B * X -> conv1d(l_cache) -> C * conv_sum -> out_proj`) and
//!   QK-Norm NeoX RoPE GQA layers with `FP16`/`FP32` per-layer input residual checkpointing.
//! - Zero heap allocations inside `forward_backward_microbatch`.

use tm_core::f16_to_f32;
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool, active_kernels,
    gqa_attention_f16_kv_bwd_ext_f32, gqa_attention_f16_kv_ext_batch_parallel,
    gqa_attention_f16_kv_ext_parallel, matmul_pair_f32, matmul_triple_f32, matvec_pair_f32,
    matvec_triple_f32, rms_norm, rms_norm_batch_parallel, rms_norm_bwd_f32,
    rope_bwd_inplace_tables, rope_compute_cos_sin, rope_inplace_tables, swiglu_bwd_f32,
    swiglu_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch, LoraSet};
use tm_models::{Lfm2Model, Lfm2OpWeights, decode_embedding_row};

use crate::accel::{
    LinearAccel, embed_lora_backward, linear_bwd_with_accel, linear_fwd_with_accel,
    masked_chunked_ce_fwd_bwd, masked_chunked_ce_loss_f64,
};
use crate::optim::{LoraGradStore, LoraTargets};
use crate::qlora::{CE_CHUNK_TOKENS, CheckpointMode};

struct Lfm2TrainArena {
    max_seq: usize,
    ckpt_mode: CheckpointMode,
    ckpt_f16: Vec<u16>,
    ckpt_f32: Vec<f32>,

    x_cur: Vec<f32>,
    x_layer_in: Vec<f32>,
    dx: Vec<f32>,

    // Recomputed single-layer activations
    x_attn_norm: Vec<f32>,
    bcx_buf: Vec<f32>,
    conv_sum: Vec<f32>,
    conv_y: Vec<f32>,
    conv_hist_scratch: Vec<f32>,
    q_pre_norm: Vec<f32>,
    k_pre_norm: Vec<f32>,
    q_rope: Vec<f32>,
    k_rope_f32: Vec<f32>,
    v_full_f32: Vec<f32>,
    k_f16: Vec<u16>,
    v_f16: Vec<u16>,
    attn_out: Vec<f32>,
    x_mid: Vec<f32>,
    x_ffn_norm: Vec<f32>,
    gate_pre: Vec<f32>,
    up_pre: Vec<f32>,
    swiglu_out: Vec<f32>,

    // Backward gradient workspaces
    xb2_chunk: Vec<f32>,
    d_swiglu: Vec<f32>,
    d_gate: Vec<f32>,
    d_up: Vec<f32>,
    d_x_norm: Vec<f32>,
    d_conv_y: Vec<f32>,
    d_conv_sum: Vec<f32>,
    d_bcx: Vec<f32>,
    d_attn_out: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    tmp_embd: Vec<f32>,
    tmp_head: Vec<f32>,

    // Chunked CE
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

impl Lfm2TrainArena {
    fn new(
        model: &Lfm2Model<'_>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let c = &model.cfg;
        let d = c.n_embd;
        let ff = c.n_ff;
        let q_dim = c.n_head * c.head_dim;
        let max_kv_dim = (c.max_n_head_kv() * c.head_dim).max(1);
        let hist_len = c.shortconv_l_cache.saturating_sub(1);
        let max_cols = d.max(ff).max(q_dim).max(3 * d).max(c.vocab_size);
        let r_cap = max_rank.max(1);
        let max_d_out = (3 * d).max(ff).max(q_dim).max(max_kv_dim).max(c.vocab_size);

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
            bcx_buf: vec![0.0; max_seq * 3 * d],
            conv_sum: vec![0.0; max_seq * d],
            conv_y: vec![0.0; max_seq * d],
            conv_hist_scratch: vec![0.0; d * hist_len.max(1)],
            q_pre_norm: vec![0.0; max_seq * q_dim.max(1)],
            k_pre_norm: vec![0.0; max_seq * max_kv_dim],
            q_rope: vec![0.0; max_seq * q_dim.max(1)],
            k_rope_f32: vec![0.0; max_seq * max_kv_dim],
            v_full_f32: vec![0.0; max_seq * max_kv_dim],
            k_f16: vec![0u16; max_seq * max_kv_dim],
            v_f16: vec![0u16; max_seq * max_kv_dim],
            attn_out: vec![0.0; max_seq * q_dim.max(1)],
            x_mid: vec![0.0; max_seq * d],
            x_ffn_norm: vec![0.0; max_seq * d],
            gate_pre: vec![0.0; max_seq * ff],
            up_pre: vec![0.0; max_seq * ff],
            swiglu_out: vec![0.0; max_seq * ff],
            xb2_chunk: vec![0.0; MAX_CPU_BATCH * d.max(q_dim)],
            d_swiglu: vec![0.0; max_seq * ff],
            d_gate: vec![0.0; max_seq * ff],
            d_up: vec![0.0; max_seq * ff],
            d_x_norm: vec![0.0; max_seq * d],
            d_conv_y: vec![0.0; max_seq * d],
            d_conv_sum: vec![0.0; max_seq * d],
            d_bcx: vec![0.0; max_seq * 3 * d],
            d_attn_out: vec![0.0; max_seq * q_dim.max(1)],
            dq: vec![0.0; max_seq * q_dim.max(1)],
            dk: vec![0.0; max_seq * max_kv_dim],
            dv: vec![0.0; max_seq * max_kv_dim],
            tmp_embd: vec![0.0; d],
            tmp_head: vec![0.0; c.head_dim.max(1)],
            ce_pos: Vec::with_capacity(max_seq),
            ce_x_sub: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_x_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_logits: vec![0.0; CE_CHUNK_TOKENS * c.vocab_size],
            ce_dx_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            rope_cos: vec![0.0; c.rope_dim / 2],
            rope_sin: vec![0.0; c.rope_dim / 2],
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

pub fn estimate_lfm2_qlora_dirty_bytes(
    model: &Lfm2Model<'_>,
    max_seq: usize,
    rank: usize,
    targets: &LoraTargets,
    ckpt_mode: CheckpointMode,
) -> u64 {
    let c = &model.cfg;
    let d = c.n_embd;
    let ff = c.n_ff;
    let q_dim = c.n_head * c.head_dim;
    let max_kv_dim = c.max_n_head_kv() * c.head_dim;
    let ckpt_elem = match ckpt_mode {
        CheckpointMode::Fp16 => 2usize,
        CheckpointMode::Fp32 => 4usize,
    };
    let ckpt_bytes = c.n_layers * max_seq * d * ckpt_elem;
    let per_tok_floats = 16 * d + 4 * q_dim + 4 * max_kv_dim + 6 * ff;
    let arena_bytes = max_seq * per_tok_floats * 4
        + 2 * max_seq * max_kv_dim * 2
        + CE_CHUNK_TOKENS * (c.vocab_size + 3 * d) * 4;

    let mut lora_params = 0usize;
    for lw in &model.layers {
        match &lw.op {
            Lfm2OpWeights::ShortConv {
                in_proj, out_proj, ..
            } => {
                if targets.q || targets.k || targets.v {
                    lora_params += rank * (in_proj.cols + in_proj.rows);
                }
                if targets.o {
                    lora_params += rank * (out_proj.cols + out_proj.rows);
                }
            }
            Lfm2OpWeights::Gqa { wq, wk, wv, wo, .. } => {
                if targets.q {
                    lora_params += rank * (wq.cols + wq.rows);
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

/// Independent QLoRA training engine for `Lfm2Model` (`#![forbid(unsafe_code)]`).
pub struct Lfm2Trainer<'a> {
    pub model: &'a Lfm2Model<'a>,
    arena: Lfm2TrainArena,
    bound_lora: BoundLora,
    pub exact_f32_kv_gradcheck: bool,
}

impl<'a> Lfm2Trainer<'a> {
    pub fn new(
        model: &'a Lfm2Model<'a>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let seq_cap = max_seq.max(2);
        Self {
            model,
            arena: Lfm2TrainArena::new(model, seq_cap, max_rank, ckpt_mode),
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
        model: &Lfm2Model<'_>,
        bound: &BoundLora,
        arena: &mut Lfm2TrainArena,
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
        let l_cache = c.shortconv_l_cache;
        let hist_len = l_cache.saturating_sub(1);
        let attn_scale = 1.0f32 / (hd as f32).sqrt();

        let slot_wg = bound.slot_for(lw.w_gate.name);
        let slot_wu = bound.slot_for(lw.w_up.name);
        let slot_wd = bound.slot_for(lw.w_down.name);

        if hist_len > 0 {
            arena.conv_hist_scratch[..d * hist_len].fill(0.0);
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
            match &lw.op {
                Lfm2OpWeights::ShortConv {
                    in_proj,
                    conv,
                    out_proj,
                } => {
                    let slot_in = bound.slot_for(in_proj.name);
                    let slot_out = bound.slot_for(out_proj.name);
                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        in_proj,
                        bound,
                        slot_in,
                        xn_chunk,
                        &mut arena.bcx_buf[offset * 3 * d..(offset + b) * 3 * d],
                        b,
                        single_tok_mode,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );

                    for t in 0..b {
                        let pos = offset + t;
                        let bcx_t = &arena.bcx_buf[pos * 3 * d..(pos + 1) * 3 * d];
                        let y_t = &mut arena.conv_y[pos * d..(pos + 1) * d];
                        let cs_t = &mut arena.conv_sum[pos * d..(pos + 1) * d];
                        for ch in 0..d {
                            let bx_val = bcx_t[ch] * bcx_t[2 * d + ch];
                            let c_val = bcx_t[d + ch];
                            let mut sum = 0.0f32;
                            if hist_len > 0 {
                                let h_slice = &mut arena.conv_hist_scratch
                                    [ch * hist_len..(ch + 1) * hist_len];
                                let w_slice = &conv[ch * l_cache..(ch + 1) * l_cache];
                                for tap in 0..hist_len {
                                    sum += h_slice[tap] * w_slice[tap];
                                }
                                sum += bx_val * w_slice[hist_len];
                                for tap in 0..hist_len.saturating_sub(1) {
                                    h_slice[tap] = h_slice[tap + 1];
                                }
                                h_slice[hist_len - 1] = bx_val;
                            } else {
                                sum += bx_val * conv[ch * l_cache];
                            }
                            if save_recompute {
                                cs_t[ch] = sum;
                            }
                            y_t[ch] = c_val * sum;
                        }
                    }

                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        out_proj,
                        bound,
                        slot_out,
                        &arena.conv_y[offset * d..(offset + b) * d],
                        &mut arena.xb2_chunk[..b * d],
                        b,
                        single_tok_mode,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );
                }
                Lfm2OpWeights::Gqa {
                    n_head_kv,
                    q_norm,
                    k_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                } => {
                    let kv_dim = *n_head_kv * hd;
                    let slot_wq = bound.slot_for(wq.name);
                    let slot_wk = bound.slot_for(wk.name);
                    let slot_wv = bound.slot_for(wv.name);
                    let slot_wo = bound.slot_for(wo.name);

                    if accel.is_some() {
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            wq,
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
                            wk,
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
                            wv,
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
                                wq.dtype,
                                wq.bytes,
                                q_dim,
                                &mut arena.q_rope[..q_dim],
                                wk.dtype,
                                wk.bytes,
                                kv_dim,
                                &mut arena.k_rope_f32[..kv_dim],
                                wv.dtype,
                                wv.bytes,
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
                                wq.dtype,
                                wq.bytes,
                                q_dim,
                                &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                                wk.dtype,
                                wk.bytes,
                                kv_dim,
                                &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                                wv.dtype,
                                wv.bytes,
                                kv_dim,
                                &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                                xn_chunk,
                                &mut arena.mv_scratch,
                                d,
                                b,
                            );
                        }
                        if !bound.is_empty() {
                            bound.apply_linear(
                                slot_wq,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wk,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wv,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                        }
                    }

                    if save_recompute {
                        arena.q_pre_norm[offset * q_dim..(offset + b) * q_dim]
                            .copy_from_slice(&arena.q_rope[offset * q_dim..(offset + b) * q_dim]);
                        arena.k_pre_norm[offset * kv_dim..(offset + b) * kv_dim].copy_from_slice(
                            &arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                        );
                    }

                    for t in 0..b {
                        let pos = offset + t;
                        let qt = &mut arena.q_rope[pos * q_dim..(pos + 1) * q_dim];
                        let kt_vec = &mut arena.k_rope_f32[pos * kv_dim..(pos + 1) * kv_dim];
                        let vt_vec = &arena.v_full_f32[pos * kv_dim..(pos + 1) * kv_dim];

                        for h in 0..c.n_head {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(&mut arena.tmp_head[..hd], &qt[s..e], q_norm, c.rms_eps);
                            qt[s..e].copy_from_slice(&arena.tmp_head[..hd]);
                        }
                        for h in 0..*n_head_kv {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(&mut arena.tmp_head[..hd], &kt_vec[s..e], k_norm, c.rms_eps);
                            kt_vec[s..e].copy_from_slice(&arena.tmp_head[..hd]);
                        }

                        rope_compute_cos_sin(
                            &model.rope_inv_freqs,
                            pos,
                            &mut arena.rope_cos,
                            &mut arena.rope_sin,
                        );
                        rope_inplace_tables(
                            qt,
                            c.n_head,
                            hd,
                            c.rope_dim,
                            &arena.rope_cos,
                            &arena.rope_sin,
                            RopeType::Neox,
                        );
                        rope_inplace_tables(
                            kt_vec,
                            *n_head_kv,
                            hd,
                            c.rope_dim,
                            &arena.rope_cos,
                            &arena.rope_sin,
                            RopeType::Neox,
                        );

                        if !exact_f32_kv {
                            let kv_off = pos * kv_dim;
                            (kt.f32_to_f16_slice)(
                                kt_vec,
                                &mut arena.k_f16[kv_off..kv_off + kv_dim],
                            );
                            (kt.f32_to_f16_slice)(
                                vt_vec,
                                &mut arena.v_f16[kv_off..kv_off + kv_dim],
                            );
                        }
                    }

                    if exact_f32_kv {
                        gqa_attn_f32_ext(
                            &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                            &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                            &arena.k_rope_f32[..(offset + b) * kv_dim],
                            &arena.v_full_f32[..(offset + b) * kv_dim],
                            c.n_head,
                            *n_head_kv,
                            hd,
                            offset,
                            b,
                            attn_scale,
                            c.n_swa,
                        );
                    } else if single_tok_mode {
                        gqa_attention_f16_kv_ext_parallel(
                            pool,
                            kt,
                            &mut arena.attn_out[..q_dim],
                            &arena.q_rope[..q_dim],
                            &arena.k_f16,
                            &arena.v_f16,
                            c.n_head,
                            *n_head_kv,
                            hd,
                            0,
                            attn_scale,
                            c.n_swa,
                            arena.max_seq,
                            &mut arena.attn_scores,
                        );
                    } else {
                        gqa_attention_f16_kv_ext_batch_parallel(
                            pool,
                            kt,
                            &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                            &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                            &arena.k_f16,
                            &arena.v_f16,
                            c.n_head,
                            *n_head_kv,
                            hd,
                            offset,
                            b,
                            attn_scale,
                            c.n_swa,
                            arena.max_seq,
                            &mut arena.attn_scores,
                        );
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

            // SwiGLU FFN
            let xm_chunk = &arena.x_mid[offset * d..(offset + b) * d];
            if single_tok_mode {
                rms_norm(
                    &mut arena.x_ffn_norm[offset * d..(offset + 1) * d],
                    xm_chunk,
                    &lw.ffn_norm,
                    c.rms_eps,
                );
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_ffn_norm[offset * d..(offset + b) * d],
                    xm_chunk,
                    &lw.ffn_norm,
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
                &lw.w_down,
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
        let l_cache = c.shortconv_l_cache;
        let hist_len = l_cache.saturating_sub(1);
        let attn_scale = 1.0f32 / (hd as f32).sqrt();

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

            // 3a. SwiGLU FFN backward
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_down,
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
            for t in 0..seq_len {
                rms_norm_bwd_f32(
                    &mut self.arena.tmp_embd,
                    None,
                    &self.arena.x_mid[t * d..(t + 1) * d],
                    &lw.ffn_norm,
                    &self.arena.d_x_norm[t * d..(t + 1) * d],
                    c.rms_eps,
                );
                (kt.axpy_f32)(
                    &mut self.arena.dx[t * d..(t + 1) * d],
                    1.0,
                    &self.arena.tmp_embd,
                );
            }

            // 3b. Operator backward (`ShortConv` or `GQA`)
            match &lw.op {
                Lfm2OpWeights::ShortConv {
                    in_proj,
                    conv,
                    out_proj,
                } => {
                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        out_proj,
                        adapter,
                        grads,
                        &self.arena.conv_y[..seq_len * d],
                        &self.arena.dx[..seq_len * d],
                        &mut self.arena.d_conv_y[..seq_len * d],
                        seq_len,
                        false,
                        &mut self.arena.lora_h,
                        &mut self.arena.lora_dh,
                        &mut self.arena.lora_bt,
                    );

                    // `conv_y[t, ch] = C[t, ch] * conv_sum[t, ch]`
                    for t in 0..seq_len {
                        let bcx_t = &self.arena.bcx_buf[t * 3 * d..(t + 1) * 3 * d];
                        let cs_t = &self.arena.conv_sum[t * d..(t + 1) * d];
                        let dy_t = &self.arena.d_conv_y[t * d..(t + 1) * d];
                        let dcs_t = &mut self.arena.d_conv_sum[t * d..(t + 1) * d];
                        let dbcx_t = &mut self.arena.d_bcx[t * 3 * d..(t + 1) * 3 * d];
                        for ch in 0..d {
                            let c_val = bcx_t[d + ch] as f64;
                            let dy_v = dy_t[ch] as f64;
                            dbcx_t[d + ch] = (dy_v * (cs_t[ch] as f64)) as f32;
                            dcs_t[ch] = (dy_v * c_val) as f32;
                        }
                    }

                    // Causal depthwise conv backward: at time `t`, `tap` (`0..l_cache`) multiplies
                    // `bx[s, ch]` where `s = t + tap - hist_len`, i.e., `t = s + hist_len - tap`!
                    for s in 0..seq_len {
                        let bcx_s = &self.arena.bcx_buf[s * 3 * d..(s + 1) * 3 * d];
                        let dbcx_s = &mut self.arena.d_bcx[s * 3 * d..(s + 1) * 3 * d];
                        for ch in 0..d {
                            let w_slice = &conv[ch * l_cache..(ch + 1) * l_cache];
                            let mut dbx = 0.0f64;
                            for (tap, &w_tap) in w_slice.iter().enumerate() {
                                if s + hist_len >= tap {
                                    let t = s + hist_len - tap;
                                    if t < seq_len {
                                        dbx += (self.arena.d_conv_sum[t * d + ch] as f64)
                                            * (w_tap as f64);
                                    }
                                }
                            }
                            let b_val = bcx_s[ch] as f64;
                            let x_val = bcx_s[2 * d + ch] as f64;
                            dbcx_s[ch] = (dbx * x_val) as f32;
                            dbcx_s[2 * d + ch] = (dbx * b_val) as f32;
                        }
                    }

                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        in_proj,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.d_bcx[..seq_len * 3 * d],
                        &mut self.arena.d_x_norm[..seq_len * d],
                        seq_len,
                        false,
                        &mut self.arena.lora_h,
                        &mut self.arena.lora_dh,
                        &mut self.arena.lora_bt,
                    );
                }
                Lfm2OpWeights::Gqa {
                    n_head_kv,
                    q_norm,
                    k_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                } => {
                    let kv_dim = *n_head_kv * hd;
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

                    if self.exact_f32_kv_gradcheck {
                        gqa_attn_f32_bwd_ext(
                            kt,
                            &mut self.arena.dq[..seq_len * q_dim],
                            &mut self.arena.dk[..seq_len * kv_dim],
                            &mut self.arena.dv[..seq_len * kv_dim],
                            &self.arena.q_rope[..seq_len * q_dim],
                            &self.arena.k_rope_f32[..seq_len * kv_dim],
                            &self.arena.v_full_f32[..seq_len * kv_dim],
                            &self.arena.d_attn_out[..seq_len * q_dim],
                            c.n_head,
                            *n_head_kv,
                            hd,
                            seq_len,
                            attn_scale,
                            c.n_swa,
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
                            *n_head_kv,
                            hd,
                            seq_len,
                            attn_scale,
                            c.n_swa,
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
                            *n_head_kv,
                            hd,
                            c.rope_dim,
                            &self.arena.rope_cos,
                            &self.arena.rope_sin,
                            RopeType::Neox,
                        );

                        let q_pre_t = &self.arena.q_pre_norm[t * q_dim..(t + 1) * q_dim];
                        for h in 0..c.n_head {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm_bwd_f32(
                                &mut self.arena.tmp_head[..hd],
                                None,
                                &q_pre_t[s..e],
                                q_norm,
                                &dqt[s..e],
                                c.rms_eps,
                            );
                            dqt[s..e].copy_from_slice(&self.arena.tmp_head[..hd]);
                        }
                        let k_pre_t = &self.arena.k_pre_norm[t * kv_dim..(t + 1) * kv_dim];
                        for h in 0..*n_head_kv {
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
                        wq,
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attn_f32_ext(
    out: &mut [f32],
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    n_tok: usize,
    attn_scale: f32,
    window: Option<usize>,
) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    let scale = attn_scale as f64;
    for t in 0..n_tok {
        let end_pos = start_pos + t + 1;
        let start_kv = match window {
            Some(w) if w > 0 => end_pos.saturating_sub(w),
            _ => 0,
        };
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let oh = &mut out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let mut scores = Vec::with_capacity(end_pos - start_kv);
            let mut max_s = f64::NEG_INFINITY;
            for step in start_kv..end_pos {
                let kh =
                    &k_all[step * kv_dim + kv_h * head_dim..step * kv_dim + (kv_h + 1) * head_dim];
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (qh[d] as f64) * (kh[d] as f64);
                }
                let sc = dot * scale;
                if sc > max_s {
                    max_s = sc;
                }
                scores.push(sc);
            }
            let mut sum = 0.0f64;
            for sc in &mut scores {
                *sc = (*sc - max_s).exp();
                sum += *sc;
            }
            let inv = 1.0 / sum;
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for (wi, step) in (start_kv..end_pos).enumerate() {
                    acc += (scores[wi] * inv) * (v_all[step * kv_dim + kv_h * head_dim + d] as f64);
                }
                oh[d] = acc as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_attn_f32_bwd_ext(
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
    attn_scale: f32,
    window: Option<usize>,
) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    dq.fill(0.0);
    dk.fill(0.0);
    dv.fill(0.0);
    let group = n_heads / n_kv_heads;
    let scale = attn_scale as f64;
    for t in 0..n_tok {
        let end_pos = t + 1;
        let start_kv = match window {
            Some(w) if w > 0 => end_pos.saturating_sub(w),
            _ => 0,
        };
        let win_len = end_pos - start_kv;
        let mut p = vec![0.0f64; win_len];
        let mut d_prob = vec![0.0f64; win_len];
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let doh = &d_out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let mut max_s = f64::NEG_INFINITY;
            for (wi, step) in (start_kv..end_pos).enumerate() {
                let kh =
                    &k_all[step * kv_dim + kv_h * head_dim..step * kv_dim + (kv_h + 1) * head_dim];
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += (qh[d] as f64) * (kh[d] as f64);
                }
                p[wi] = dot * scale;
                if p[wi] > max_s {
                    max_s = p[wi];
                }
            }
            let mut sum = 0.0f64;
            for ps in &mut p {
                *ps = (*ps - max_s).exp();
                sum += *ps;
            }
            let inv = 1.0 / sum;
            for ps in &mut p {
                *ps *= inv;
            }

            let mut sum_p_dp = 0.0f64;
            for (wi, step) in (start_kv..end_pos).enumerate() {
                let pw = p[wi];
                let v_off = step * kv_dim + kv_h * head_dim;
                let vh = &v_all[v_off..v_off + head_dim];
                let dvh = &mut dv[v_off..v_off + head_dim];
                let mut dot_dov = 0.0f64;
                for d in 0..head_dim {
                    dvh[d] += (doh[d] as f64 * pw) as f32;
                    dot_dov += (doh[d] as f64) * (vh[d] as f64);
                }
                d_prob[wi] = dot_dov;
                sum_p_dp += pw * dot_dov;
            }
            let dqh = &mut dq[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            for (wi, step) in (start_kv..end_pos).enumerate() {
                let ds = (p[wi] * (d_prob[wi] - sum_p_dp) * scale) as f32;
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
