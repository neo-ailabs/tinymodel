//! Phase `T3`: `Gemma 4` (`PLE` + `SWA` + `Shared KV` + `GeGLU` + `Softcap`) QLoRA trainer (`ADR-015`).
//!
//! Handles every Gemma 4 architectural invariant accurately in both forward and backward:
//! 1. Scaled input embeddings (`sqrt(n_embd)`) and on-demand global Per-Layer Embeddings (`ple_global`).
//! 2. Cross-layer KV sharing (`kv_source[il]`): upper layers `il >= n_layer_kv_from_start` attend
//!    to `own_k/v[kv_source[il]]` and fan their `(dk, dv)` gradients back into `dk_shared[kv_src]`
//!    & `dv_shared[kv_src]` before layer `kv_src` backpropagates through `wk`/`wv` (including `wv == None` KV tying).
//! 3. Custom pre-softmax `attn_scale = 1.0`, interleaved SWA (`window = Some(n_swa)`) + proportional
//!    full-attention RoPE, and unit-weight `V` RMSNorm (`rms_norm_unit_bwd_f32`).
//! 4. Post-attention / post-FFN / post-PLE RMSNorms, variable-width GeGLU FFN, optional `out_scale`,
//!    and `final_logit_softcapping` (`tanh(z / cap) * cap`).

use tm_core::f16_to_f32;
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool, active_kernels, geglu_bwd_f32,
    geglu_inplace_parallel, gelu_tanh_bwd_f32, gelu_tanh_f32, gqa_attention_f16_kv_bwd_softcap_f32,
    gqa_attention_f16_kv_softcap_batch_parallel, gqa_attention_f16_kv_softcap_parallel,
    matmul_pair_f32, matmul_triple_f32, matvec_pair_f32, matvec_triple_f32, rms_norm,
    rms_norm_batch_parallel, rms_norm_bwd_f32, rms_norm_unit_bwd_f32, rms_norm_unit_inplace,
    rope_bwd_inplace_tables, rope_compute_cos_sin, rope_inplace_tables, softcap_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch, LoraSet};
use tm_models::{Gemma4Model, decode_embedding_row};

use crate::accel::{
    LinearAccel, embed_lora_backward, linear_bwd_with_accel, linear_fwd_with_accel,
    masked_chunked_ce_fwd_bwd, masked_chunked_ce_loss_f64,
};
use crate::lfm2_qlora::{gqa_attn_f32_bwd_ext_softcap, gqa_attn_f32_ext_softcap};
use crate::optim::{LoraGradStore, LoraTargets};
use crate::qlora::{CE_CHUNK_TOKENS, CheckpointMode};

struct Gemma4TrainArena {
    max_seq: usize,
    ckpt_mode: CheckpointMode,
    ckpt_f16: Vec<u16>,
    ckpt_f32: Vec<f32>,

    x_embd_scaled: Vec<f32>,
    x_cur: Vec<f32>,
    x_layer_in: Vec<f32>,
    dx: Vec<f32>,

    // Global PLE activations & gradients (`[max_seq, n_layers * ple_dim]`)
    ple_inputs: Vec<f32>,
    ple_proj_raw: Vec<f32>,
    d_ple_inputs: Vec<f32>,
    d_ple_proj_raw: Vec<f32>,

    // Own-KV caches across `0..n_layer_kv_from_start` + cross-layer shared gradient accumulators
    own_k_f16: Vec<Vec<u16>>,
    own_v_f16: Vec<Vec<u16>>,
    own_k_f32: Vec<Vec<f32>>,
    own_v_f32: Vec<Vec<f32>>,
    dk_shared: Vec<Vec<f32>>,
    dv_shared: Vec<Vec<f32>>,

    // Single-layer recomputed forward activations
    x_attn_norm: Vec<f32>,
    q_pre_norm: Vec<f32>,
    k_pre_norm: Vec<f32>,
    v_pre_norm: Vec<f32>,
    q_rope: Vec<f32>,
    k_work: Vec<f32>,
    v_work: Vec<f32>,
    attn_out: Vec<f32>,
    wo_raw: Vec<f32>,
    attn_res: Vec<f32>,
    x_ffn_norm: Vec<f32>,
    gate_pre: Vec<f32>,
    up_pre: Vec<f32>,
    geglu_out: Vec<f32>,
    ffn_down_raw: Vec<f32>,
    x_after_ffn: Vec<f32>,
    ple_gate_pre: Vec<f32>,
    ple_gated: Vec<f32>,
    ple_proj_l_raw: Vec<f32>,

    // Single-layer backward gradient workspaces
    xb_chunk: Vec<f32>,
    d_post: Vec<f32>,
    d_ple_gated: Vec<f32>,
    d_ple_gate_pre: Vec<f32>,
    d_geglu: Vec<f32>,
    d_gate: Vec<f32>,
    d_up: Vec<f32>,
    d_x_norm: Vec<f32>,
    d_attn_out: Vec<f32>,
    dq: Vec<f32>,
    dk_cur: Vec<f32>,
    dv_cur: Vec<f32>,
    tmp_embd: Vec<f32>,
    tmp_head: Vec<f32>,
    tmp_ple: Vec<f32>,

    // Chunked CE
    ce_pos: Vec<(usize, u32)>,
    ce_x_sub: Vec<f32>,
    ce_x_norm: Vec<f32>,
    ce_logits: Vec<f32>,
    ce_dx_norm: Vec<f32>,

    rope_cos_full: Vec<f32>,
    rope_sin_full: Vec<f32>,
    rope_cos_swa: Vec<f32>,
    rope_sin_swa: Vec<f32>,
    attn_scores: Vec<f32>,
    mv_scratch: MatvecScratch,
    lora_scratch: LoraScratch,
    lora_h: Vec<f32>,
    lora_dh: Vec<f32>,
    lora_bt: Vec<f32>,
}

impl Gemma4TrainArena {
    fn new(
        model: &Gemma4Model<'_>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
        exact_f32_kv: bool,
    ) -> Self {
        let c = &model.cfg;
        let d = c.n_embd;
        let max_hd = c.max_head_dim();
        let max_q_dim = c.n_head * max_hd;
        let max_kv_dim = c.n_head_kv * max_hd;
        let max_ff = c.max_n_ff();
        let ple_dim = c.n_embd_per_layer;
        let ple_total = c.n_layers * ple_dim;
        let max_cols = d
            .max(max_ff)
            .max(max_q_dim)
            .max(ple_total)
            .max(c.vocab_size);
        let r_cap = max_rank.max(1);
        let max_d_out = d
            .max(max_ff)
            .max(max_q_dim)
            .max(ple_total)
            .max(c.vocab_size);

        let (ckpt_f16, ckpt_f32) = match ckpt_mode {
            CheckpointMode::Fp16 => (vec![0u16; c.n_layers * max_seq * d], Vec::new()),
            CheckpointMode::Fp32 => (Vec::new(), vec![0.0f32; c.n_layers * max_seq * d]),
        };

        let mut own_k_f16 = Vec::with_capacity(c.n_layer_kv_from_start);
        let mut own_v_f16 = Vec::with_capacity(c.n_layer_kv_from_start);
        let mut own_k_f32 = Vec::with_capacity(c.n_layer_kv_from_start);
        let mut own_v_f32 = Vec::with_capacity(c.n_layer_kv_from_start);
        let mut dk_shared = Vec::with_capacity(c.n_layer_kv_from_start);
        let mut dv_shared = Vec::with_capacity(c.n_layer_kv_from_start);

        for il in 0..c.n_layer_kv_from_start {
            let kv_dim = c.n_head_kv * c.head_dim(il);
            own_k_f16.push(vec![0u16; max_seq * kv_dim]);
            own_v_f16.push(vec![0u16; max_seq * kv_dim]);
            if exact_f32_kv {
                own_k_f32.push(vec![0.0f32; max_seq * kv_dim]);
                own_v_f32.push(vec![0.0f32; max_seq * kv_dim]);
            }
            dk_shared.push(vec![0.0f32; max_seq * kv_dim]);
            dv_shared.push(vec![0.0f32; max_seq * kv_dim]);
        }

        Self {
            max_seq,
            ckpt_mode,
            ckpt_f16,
            ckpt_f32,
            x_embd_scaled: vec![0.0; max_seq * d],
            x_cur: vec![0.0; max_seq * d],
            x_layer_in: vec![0.0; max_seq * d],
            dx: vec![0.0; max_seq * d],
            ple_inputs: vec![0.0; max_seq * ple_total.max(1)],
            ple_proj_raw: vec![0.0; max_seq * ple_total.max(1)],
            d_ple_inputs: vec![0.0; max_seq * ple_total.max(1)],
            d_ple_proj_raw: vec![0.0; max_seq * ple_total.max(1)],
            own_k_f16,
            own_v_f16,
            own_k_f32,
            own_v_f32,
            dk_shared,
            dv_shared,
            x_attn_norm: vec![0.0; max_seq * d],
            q_pre_norm: vec![0.0; max_seq * max_q_dim],
            k_pre_norm: vec![0.0; max_seq * max_kv_dim],
            v_pre_norm: vec![0.0; max_seq * max_kv_dim],
            q_rope: vec![0.0; max_seq * max_q_dim],
            k_work: vec![0.0; max_seq * max_kv_dim],
            v_work: vec![0.0; max_seq * max_kv_dim],
            attn_out: vec![0.0; max_seq * max_q_dim],
            wo_raw: vec![0.0; max_seq * d],
            attn_res: vec![0.0; max_seq * d],
            x_ffn_norm: vec![0.0; max_seq * d],
            gate_pre: vec![0.0; max_seq * max_ff],
            up_pre: vec![0.0; max_seq * max_ff],
            geglu_out: vec![0.0; max_seq * max_ff],
            ffn_down_raw: vec![0.0; max_seq * d],
            x_after_ffn: vec![0.0; max_seq * d],
            ple_gate_pre: vec![0.0; max_seq * ple_dim.max(1)],
            ple_gated: vec![0.0; max_seq * ple_dim.max(1)],
            ple_proj_l_raw: vec![0.0; max_seq * d],
            xb_chunk: vec![0.0; MAX_CPU_BATCH * d.max(max_q_dim)],
            d_post: vec![0.0; max_seq * d],
            d_ple_gated: vec![0.0; max_seq * ple_dim.max(1)],
            d_ple_gate_pre: vec![0.0; max_seq * ple_dim.max(1)],
            d_geglu: vec![0.0; max_seq * max_ff],
            d_gate: vec![0.0; max_seq * max_ff],
            d_up: vec![0.0; max_seq * max_ff],
            d_x_norm: vec![0.0; max_seq * d],
            d_attn_out: vec![0.0; max_seq * max_q_dim],
            dq: vec![0.0; max_seq * max_q_dim],
            dk_cur: vec![0.0; max_seq * max_kv_dim],
            dv_cur: vec![0.0; max_seq * max_kv_dim],
            tmp_embd: vec![0.0; d],
            tmp_head: vec![0.0; max_hd.max(1)],
            tmp_ple: vec![0.0; ple_dim.max(1)],
            ce_pos: Vec::with_capacity(max_seq),
            ce_x_sub: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_x_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_logits: vec![0.0; CE_CHUNK_TOKENS * c.vocab_size],
            ce_dx_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            rope_cos_full: vec![0.0; (c.rope_dim_full / 2).max(1)],
            rope_sin_full: vec![0.0; (c.rope_dim_full / 2).max(1)],
            rope_cos_swa: vec![0.0; (c.rope_dim_swa / 2).max(1)],
            rope_sin_swa: vec![0.0; (c.rope_dim_swa / 2).max(1)],
            attn_scores: vec![0.0; c.n_head.max(1) * max_seq],
            mv_scratch: MatvecScratch::new(max_cols),
            lora_scratch: LoraScratch::new(r_cap),
            lora_h: vec![0.0; max_seq * r_cap],
            lora_dh: vec![0.0; max_seq * r_cap],
            lora_bt: vec![0.0; r_cap * max_d_out],
        }
    }

    fn ensure_f32_kv(&mut self, c: &tm_models::Gemma4Config) {
        if self.own_k_f32.len() < c.n_layer_kv_from_start {
            self.own_k_f32.clear();
            self.own_v_f32.clear();
            for il in 0..c.n_layer_kv_from_start {
                let kv_dim = c.n_head_kv * c.head_dim(il);
                self.own_k_f32.push(vec![0.0f32; self.max_seq * kv_dim]);
                self.own_v_f32.push(vec![0.0f32; self.max_seq * kv_dim]);
            }
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

pub fn estimate_gemma4_qlora_dirty_bytes(
    model: &Gemma4Model<'_>,
    max_seq: usize,
    rank: usize,
    targets: &LoraTargets,
    ckpt_mode: CheckpointMode,
) -> u64 {
    let c = &model.cfg;
    let d = c.n_embd;
    let max_hd = c.max_head_dim();
    let max_q_dim = c.n_head * max_hd;
    let max_kv_dim = c.n_head_kv * max_hd;
    let max_ff = c.max_n_ff();
    let ple_total = c.n_layers * c.n_embd_per_layer;

    let ckpt_elem = match ckpt_mode {
        CheckpointMode::Fp16 => 2usize,
        CheckpointMode::Fp32 => 4usize,
    };
    let ckpt_bytes = c.n_layers * max_seq * d * ckpt_elem;

    let mut kv_bytes = 0usize;
    for il in 0..c.n_layer_kv_from_start {
        let kv_dim = c.n_head_kv * c.head_dim(il);
        // F16 K/V (4B) + shared FP32 dk/dv accumulators (8B)
        kv_bytes += max_seq * kv_dim * 12;
    }

    let per_tok_floats = 12 * d + 4 * max_q_dim + 4 * max_kv_dim + 6 * max_ff + 4 * ple_total;
    let arena_bytes =
        max_seq * per_tok_floats * 4 + kv_bytes + CE_CHUNK_TOKENS * (c.vocab_size + 3 * d) * 4;

    let mut lora_params = 0usize;
    if targets.ple
        && let Some(ref pg) = model.ple_global
    {
        lora_params += rank * (pg.model_proj.cols + pg.model_proj.rows);
    }
    for lw in &model.layers {
        if targets.q {
            lora_params += rank * (lw.wq.cols + lw.wq.rows);
        }
        if let Some(ref kv) = lw.kv {
            if targets.k {
                lora_params += rank * (kv.wk.cols + kv.wk.rows);
            }
            if targets.v
                && let Some(ref wv) = kv.wv
            {
                lora_params += rank * (wv.cols + wv.rows);
            }
        }
        if targets.o {
            lora_params += rank * (lw.wo.cols + lw.wo.rows);
        }
        if targets.gate {
            lora_params += rank * (lw.w_gate.cols + lw.w_gate.rows);
        }
        if targets.up {
            lora_params += rank * (lw.w_up.cols + lw.w_up.rows);
        }
        if targets.down {
            lora_params += rank * (lw.w_down.cols + lw.w_down.rows);
        }
        if targets.ple
            && let Some(ref ple) = lw.ple
        {
            let rp = rank.min(ple.inp_gate.rows).max(1);
            lora_params += rp * (ple.inp_gate.cols + ple.inp_gate.rows);
            lora_params += rp * (ple.proj.cols + ple.proj.rows);
        }
    }
    if targets.embed || targets.lm_head {
        lora_params += rank * (c.vocab_size + d);
    }
    (ckpt_bytes + arena_bytes + lora_params * 24) as u64
}

/// Independent QLoRA training engine for `Gemma4Model` (`#![forbid(unsafe_code)]`).
pub struct Gemma4Trainer<'a> {
    pub model: &'a Gemma4Model<'a>,
    arena: Gemma4TrainArena,
    bound_lora: BoundLora,
    pub exact_f32_kv_gradcheck: bool,
}

impl<'a> Gemma4Trainer<'a> {
    pub fn new(
        model: &'a Gemma4Model<'a>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let seq_cap = max_seq.max(2);
        Self {
            model,
            arena: Gemma4TrainArena::new(model, seq_cap, max_rank, ckpt_mode, false),
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

    fn embed_and_ple_forward(
        model: &Gemma4Model<'_>,
        bound: &BoundLora,
        arena: &mut Gemma4TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        input_toks: &[u32],
    ) {
        let c = &model.cfg;
        let d = c.n_embd;
        let n_tok = input_toks.len();
        let emb_scale = (d as f32).sqrt();
        let slot_emb = bound.slot_for(model.token_embd.name);

        for (t, &tok) in input_toks.iter().enumerate() {
            let dst = &mut arena.x_embd_scaled[t * d..(t + 1) * d];
            decode_embedding_row(
                model.token_embd.dtype,
                model.token_embd.bytes,
                d,
                tok as usize,
                dst,
            );
            bound.apply_embed_row(slot_emb, kt, tok as usize, dst);
            for v in dst.iter_mut() {
                *v *= emb_scale;
            }
        }
        arena.x_cur[..n_tok * d].copy_from_slice(&arena.x_embd_scaled[..n_tok * d]);

        if let Some(ref ple_g) = model.ple_global {
            let ple_dim = c.n_embd_per_layer;
            let ple_total = c.n_layers * ple_dim;
            let tok_ple_scale = (ple_dim as f32).sqrt() / 2.0f32.sqrt();
            let slot_mp = bound.slot_for(ple_g.model_proj.name);
            let single_tok_mode = n_tok == 1;
            let proj_eps = c.rms_eps * (d as f32);

            let mut offset = 0usize;
            while offset < n_tok {
                let b = (n_tok - offset).min(MAX_CPU_BATCH);
                for t in 0..b {
                    let pos = offset + t;
                    let dst = &mut arena.ple_inputs[pos * ple_total..(pos + 1) * ple_total];
                    decode_embedding_row(
                        ple_g.tok_embd.dtype,
                        ple_g.tok_embd.bytes,
                        ple_total,
                        input_toks[pos] as usize,
                        dst,
                    );
                    for v in dst.iter_mut() {
                        *v *= tok_ple_scale;
                    }
                }

                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &ple_g.model_proj,
                    bound,
                    slot_mp,
                    &arena.x_embd_scaled[offset * d..(offset + b) * d],
                    &mut arena.ple_proj_raw[offset * ple_total..(offset + b) * ple_total],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );

                for t in 0..b {
                    let pos = offset + t;
                    for il in 0..c.n_layers {
                        let s = pos * ple_total + il * ple_dim;
                        let e = s + ple_dim;
                        rms_norm(
                            &mut arena.tmp_ple[..ple_dim],
                            &arena.ple_proj_raw[s..e],
                            &ple_g.proj_norm_scaled,
                            proj_eps,
                        );
                        for j in 0..ple_dim {
                            arena.ple_inputs[s + j] += arena.tmp_ple[j];
                        }
                    }
                }

                offset += b;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_forward(
        model: &Gemma4Model<'_>,
        bound: &BoundLora,
        arena: &mut Gemma4TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        il: usize,
        seq_len: usize,
        from_ckpt_input: bool,
        save_recompute: bool,
        exact_f32_kv: bool,
    ) {
        let c = &model.cfg;
        let lw = &model.layers[il];
        let d = c.n_embd;
        let hd = c.head_dim(il);
        let rope_dim = c.rope_dim(il);
        let q_dim = c.n_head * hd;
        let kv_dim = c.n_head_kv * hd;
        let is_swa = c.is_swa[il];
        let window = if is_swa { Some(c.n_swa) } else { None };
        let n_ff = c.n_ff_per_layer[il];
        let ple_dim = c.n_embd_per_layer;
        let ple_total = c.n_layers * ple_dim;
        let kv_src = c.kv_source[il];

        let slot_wq = bound.slot_for(lw.wq.name);
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

            // 1. Attention pre-norm
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

            // 2. Q / optional own-KV projections + LoRA
            if let Some(ref kv_w) = lw.kv {
                let slot_wk = bound.slot_for(kv_w.wk.name);
                if let Some(ref wv) = kv_w.wv {
                    let slot_wv = bound.slot_for(wv.name);
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
                            &kv_w.wk,
                            bound,
                            slot_wk,
                            xn_chunk,
                            &mut arena.k_work[offset * kv_dim..(offset + b) * kv_dim],
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
                            &mut arena.v_work[offset * kv_dim..(offset + b) * kv_dim],
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
                                kv_w.wk.dtype,
                                kv_w.wk.bytes,
                                kv_dim,
                                &mut arena.k_work[..kv_dim],
                                wv.dtype,
                                wv.bytes,
                                kv_dim,
                                &mut arena.v_work[..kv_dim],
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
                                kv_w.wk.dtype,
                                kv_w.wk.bytes,
                                kv_dim,
                                &mut arena.k_work[offset * kv_dim..(offset + b) * kv_dim],
                                wv.dtype,
                                wv.bytes,
                                kv_dim,
                                &mut arena.v_work[offset * kv_dim..(offset + b) * kv_dim],
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
                                &mut arena.k_work[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wv,
                                pool,
                                kt,
                                xn_chunk,
                                &mut arena.v_work[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                        }
                    }
                } else {
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
                            &kv_w.wk,
                            bound,
                            slot_wk,
                            xn_chunk,
                            &mut arena.k_work[offset * kv_dim..(offset + b) * kv_dim],
                            b,
                            single_tok_mode,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                    } else if single_tok_mode {
                        linear_fwd_with_accel(
                            None,
                            pool,
                            kt,
                            &lw.wq,
                            bound,
                            slot_wq,
                            xn_chunk,
                            &mut arena.q_rope[..q_dim],
                            1,
                            true,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                        linear_fwd_with_accel(
                            None,
                            pool,
                            kt,
                            &kv_w.wk,
                            bound,
                            slot_wk,
                            xn_chunk,
                            &mut arena.k_work[..kv_dim],
                            1,
                            true,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                    } else {
                        matmul_pair_f32(
                            pool,
                            kt,
                            lw.wq.dtype,
                            lw.wq.bytes,
                            q_dim,
                            &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                            kv_w.wk.dtype,
                            kv_w.wk.bytes,
                            kv_dim,
                            &mut arena.k_work[offset * kv_dim..(offset + b) * kv_dim],
                            xn_chunk,
                            &mut arena.mv_scratch,
                            d,
                            b,
                        );
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
                                &mut arena.k_work[offset * kv_dim..(offset + b) * kv_dim],
                                b,
                                &mut arena.lora_scratch,
                            );
                        }
                    }
                    arena.v_work[offset * kv_dim..(offset + b) * kv_dim]
                        .copy_from_slice(&arena.k_work[offset * kv_dim..(offset + b) * kv_dim]);
                }
            } else {
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
            }

            if save_recompute {
                arena.q_pre_norm[offset * q_dim..(offset + b) * q_dim]
                    .copy_from_slice(&arena.q_rope[offset * q_dim..(offset + b) * q_dim]);
                if lw.kv.is_some() {
                    arena.k_pre_norm[offset * kv_dim..(offset + b) * kv_dim]
                        .copy_from_slice(&arena.k_work[offset * kv_dim..(offset + b) * kv_dim]);
                    arena.v_pre_norm[offset * kv_dim..(offset + b) * kv_dim]
                        .copy_from_slice(&arena.v_work[offset * kv_dim..(offset + b) * kv_dim]);
                }
            }

            // 3. Per-token Q/K/V norms + RoPE + own-KV store
            for t in 0..b {
                let pos = offset + t;
                let qt = &mut arena.q_rope[pos * q_dim..(pos + 1) * q_dim];
                if let Some(ref qn) = lw.q_norm {
                    for h in 0..c.n_head {
                        let s = h * hd;
                        let e = s + hd;
                        rms_norm(&mut arena.tmp_head[..hd], &qt[s..e], qn, c.rms_eps);
                        qt[s..e].copy_from_slice(&arena.tmp_head[..hd]);
                    }
                }

                if is_swa {
                    rope_compute_cos_sin(
                        &model.rope_inv_freqs_swa,
                        pos,
                        &mut arena.rope_cos_swa,
                        &mut arena.rope_sin_swa,
                    );
                    rope_inplace_tables(
                        qt,
                        c.n_head,
                        hd,
                        rope_dim,
                        &arena.rope_cos_swa,
                        &arena.rope_sin_swa,
                        RopeType::Neox,
                    );
                } else {
                    rope_compute_cos_sin(
                        &model.rope_inv_freqs_full,
                        pos,
                        &mut arena.rope_cos_full,
                        &mut arena.rope_sin_full,
                    );
                    rope_inplace_tables(
                        qt,
                        c.n_head,
                        hd,
                        rope_dim,
                        &arena.rope_cos_full,
                        &arena.rope_sin_full,
                        RopeType::Neox,
                    );
                }

                if let Some(ref kv_w) = lw.kv {
                    let kt_vec = &mut arena.k_work[pos * kv_dim..(pos + 1) * kv_dim];
                    let vt_vec = &mut arena.v_work[pos * kv_dim..(pos + 1) * kv_dim];
                    for h in 0..c.n_head_kv {
                        let s = h * hd;
                        let e = s + hd;
                        if let Some(ref kn) = kv_w.k_norm {
                            rms_norm(&mut arena.tmp_head[..hd], &kt_vec[s..e], kn, c.rms_eps);
                            kt_vec[s..e].copy_from_slice(&arena.tmp_head[..hd]);
                        }
                        if c.norm_v {
                            rms_norm_unit_inplace(&mut vt_vec[s..e], c.rms_eps);
                        }
                    }
                    let (cos_s, sin_s) = if is_swa {
                        (&arena.rope_cos_swa[..], &arena.rope_sin_swa[..])
                    } else {
                        (&arena.rope_cos_full[..], &arena.rope_sin_full[..])
                    };
                    rope_inplace_tables(
                        kt_vec,
                        c.n_head_kv,
                        hd,
                        rope_dim,
                        cos_s,
                        sin_s,
                        RopeType::Neox,
                    );

                    if !save_recompute {
                        let kv_off = pos * kv_dim;
                        if exact_f32_kv {
                            arena.own_k_f32[il][kv_off..kv_off + kv_dim].copy_from_slice(kt_vec);
                            arena.own_v_f32[il][kv_off..kv_off + kv_dim].copy_from_slice(vt_vec);
                        } else {
                            (kt.f32_to_f16_slice)(
                                kt_vec,
                                &mut arena.own_k_f16[il][kv_off..kv_off + kv_dim],
                            );
                            (kt.f32_to_f16_slice)(
                                vt_vec,
                                &mut arena.own_v_f16[il][kv_off..kv_off + kv_dim],
                            );
                        }
                    }
                }
            }

            // 4. Attention (`c.attn_scale`, `c.attn_logit_softcapping`, `kv_source[il]`)
            if exact_f32_kv {
                gqa_attn_f32_ext_softcap(
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.own_k_f32[kv_src][..(offset + b) * kv_dim],
                    &arena.own_v_f32[kv_src][..(offset + b) * kv_dim],
                    c.n_head,
                    c.n_head_kv,
                    hd,
                    offset,
                    b,
                    c.attn_scale,
                    window,
                    c.attn_logit_softcapping,
                );
            } else if single_tok_mode {
                gqa_attention_f16_kv_softcap_parallel(
                    pool,
                    kt,
                    &mut arena.attn_out[..q_dim],
                    &arena.q_rope[..q_dim],
                    &arena.own_k_f16[kv_src],
                    &arena.own_v_f16[kv_src],
                    c.n_head,
                    c.n_head_kv,
                    hd,
                    0,
                    c.attn_scale,
                    window,
                    c.attn_logit_softcapping,
                    arena.max_seq,
                    &mut arena.attn_scores,
                );
            } else {
                gqa_attention_f16_kv_softcap_batch_parallel(
                    pool,
                    kt,
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.own_k_f16[kv_src],
                    &arena.own_v_f16[kv_src],
                    c.n_head,
                    c.n_head_kv,
                    hd,
                    offset,
                    b,
                    c.attn_scale,
                    window,
                    c.attn_logit_softcapping,
                    arena.max_seq,
                    &mut arena.attn_scores,
                );
            }

            // 5. `wo` + `post_attn_norm` + residual -> `attn_res`
            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wo,
                bound,
                slot_wo,
                &arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                &mut arena.wo_raw[offset * d..(offset + b) * d],
                b,
                single_tok_mode,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );
            if single_tok_mode {
                rms_norm(
                    &mut arena.xb_chunk[..d],
                    &arena.wo_raw[offset * d..(offset + 1) * d],
                    &lw.post_attn_norm,
                    c.rms_eps,
                );
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.xb_chunk[..b * d],
                    &arena.wo_raw[offset * d..(offset + b) * d],
                    &lw.post_attn_norm,
                    c.rms_eps,
                    d,
                    b,
                );
            }
            for i in 0..(b * d) {
                let xin_i = if from_ckpt_input {
                    arena.x_layer_in[offset * d + i]
                } else {
                    arena.x_cur[offset * d + i]
                };
                arena.attn_res[offset * d + i] = xin_i + arena.xb_chunk[i];
            }

            // 6. GeGLU FFN + `post_ffw_norm` + residual -> `x_after_ffn`
            let ar_chunk = &arena.attn_res[offset * d..(offset + b) * d];
            if single_tok_mode {
                rms_norm(
                    &mut arena.x_ffn_norm[offset * d..(offset + 1) * d],
                    ar_chunk,
                    &lw.ffn_norm,
                    c.rms_eps,
                );
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_ffn_norm[offset * d..(offset + b) * d],
                    ar_chunk,
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
                    &mut arena.gate_pre[offset * n_ff..(offset + b) * n_ff],
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
                    &mut arena.up_pre[offset * n_ff..(offset + b) * n_ff],
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
                        n_ff,
                        &mut arena.gate_pre[..n_ff],
                        lw.w_up.dtype,
                        lw.w_up.bytes,
                        n_ff,
                        &mut arena.up_pre[..n_ff],
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
                        n_ff,
                        &mut arena.gate_pre[offset * n_ff..(offset + b) * n_ff],
                        lw.w_up.dtype,
                        lw.w_up.bytes,
                        n_ff,
                        &mut arena.up_pre[offset * n_ff..(offset + b) * n_ff],
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
                        &mut arena.gate_pre[offset * n_ff..(offset + b) * n_ff],
                        b,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear(
                        slot_wu,
                        pool,
                        kt,
                        xfn_chunk,
                        &mut arena.up_pre[offset * n_ff..(offset + b) * n_ff],
                        b,
                        &mut arena.lora_scratch,
                    );
                }
            }

            arena.geglu_out[offset * n_ff..(offset + b) * n_ff]
                .copy_from_slice(&arena.gate_pre[offset * n_ff..(offset + b) * n_ff]);
            geglu_inplace_parallel(
                pool,
                &mut arena.geglu_out[offset * n_ff..(offset + b) * n_ff],
                &arena.up_pre[offset * n_ff..(offset + b) * n_ff],
            );

            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_down,
                bound,
                slot_wd,
                &arena.geglu_out[offset * n_ff..(offset + b) * n_ff],
                &mut arena.ffn_down_raw[offset * d..(offset + b) * d],
                b,
                single_tok_mode,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );
            if single_tok_mode {
                rms_norm(
                    &mut arena.xb_chunk[..d],
                    &arena.ffn_down_raw[offset * d..(offset + 1) * d],
                    &lw.post_ffw_norm,
                    c.rms_eps,
                );
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.xb_chunk[..b * d],
                    &arena.ffn_down_raw[offset * d..(offset + b) * d],
                    &lw.post_ffw_norm,
                    c.rms_eps,
                    d,
                    b,
                );
            }
            for (i, (&ar_i, &xb_i)) in ar_chunk
                .iter()
                .zip(arena.xb_chunk[..b * d].iter())
                .enumerate()
            {
                arena.x_after_ffn[offset * d + i] = ar_i + xb_i;
            }

            // 7. Optional per-layer PLE gated residual + `out_scale`
            if let Some(ref ple_l) = lw.ple {
                let slot_ig = bound.slot_for(ple_l.inp_gate.name);
                let slot_pr = bound.slot_for(ple_l.proj.name);
                let xaf_chunk = &arena.x_after_ffn[offset * d..(offset + b) * d];
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &ple_l.inp_gate,
                    bound,
                    slot_ig,
                    xaf_chunk,
                    &mut arena.ple_gate_pre[offset * ple_dim..(offset + b) * ple_dim],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                for t in 0..b {
                    let pos = offset + t;
                    let ple_in_t = &arena.ple_inputs
                        [pos * ple_total + il * ple_dim..pos * ple_total + (il + 1) * ple_dim];
                    let gp_t = &arena.ple_gate_pre[pos * ple_dim..(pos + 1) * ple_dim];
                    let gd_t = &mut arena.ple_gated[pos * ple_dim..(pos + 1) * ple_dim];
                    for j in 0..ple_dim {
                        gd_t[j] = gelu_tanh_f32(gp_t[j]) * ple_in_t[j];
                    }
                }
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &ple_l.proj,
                    bound,
                    slot_pr,
                    &arena.ple_gated[offset * ple_dim..(offset + b) * ple_dim],
                    &mut arena.ple_proj_l_raw[offset * d..(offset + b) * d],
                    b,
                    single_tok_mode,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                if single_tok_mode {
                    rms_norm(
                        &mut arena.xb_chunk[..d],
                        &arena.ple_proj_l_raw[offset * d..(offset + 1) * d],
                        &ple_l.post_norm,
                        c.rms_eps,
                    );
                } else {
                    rms_norm_batch_parallel(
                        pool,
                        kt,
                        &mut arena.xb_chunk[..b * d],
                        &arena.ple_proj_l_raw[offset * d..(offset + b) * d],
                        &ple_l.post_norm,
                        c.rms_eps,
                        d,
                        b,
                    );
                }
                if !save_recompute {
                    for (i, (&xaf_i, &xb_i)) in xaf_chunk
                        .iter()
                        .zip(arena.xb_chunk[..b * d].iter())
                        .enumerate()
                    {
                        arena.x_cur[offset * d + i] = xaf_i + xb_i;
                    }
                }
            } else if !save_recompute {
                arena.x_cur[offset * d..(offset + b) * d]
                    .copy_from_slice(&arena.x_after_ffn[offset * d..(offset + b) * d]);
            }

            if !save_recompute && let Some(scale) = lw.out_scale {
                for v in &mut arena.x_cur[offset * d..(offset + b) * d] {
                    *v *= scale;
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
        if self.exact_f32_kv_gradcheck {
            self.arena.ensure_f32_kv(&self.model.cfg);
        }
        if let Some(ad) = adapter {
            self.sync_bound_lora(ad);
        } else {
            self.bound_lora = BoundLora::default();
        }
        let kt = active_kernels();
        let d = self.model.cfg.n_embd;
        let n_tok = token_ids.len();

        Self::embed_and_ple_forward(
            self.model,
            &self.bound_lora,
            &mut self.arena,
            None,
            pool,
            kt,
            token_ids,
        );
        for il in 0..self.model.cfg.n_layers {
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                None,
                pool,
                kt,
                il,
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
        if let Some(cap) = self.model.cfg.final_logit_softcapping {
            softcap_inplace_parallel(pool, out_last_logits, cap);
        }
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

        if self.exact_f32_kv_gradcheck {
            self.arena.ensure_f32_kv(&self.model.cfg);
        }
        self.sync_bound_lora(adapter);
        let kt = active_kernels();
        let c = &self.model.cfg;
        let d = c.n_embd;
        let ple_dim = c.n_embd_per_layer;
        let ple_total = c.n_layers * ple_dim;

        // 1. Forward pass with per-layer checkpointing and own-KV caching
        Self::embed_and_ple_forward(
            self.model,
            &self.bound_lora,
            &mut self.arena,
            accel.as_deref_mut(),
            pool,
            kt,
            input_toks,
        );
        for il in 0..c.n_layers {
            self.arena.save_layer_ckpt(kt, il, seq_len, d);
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                accel.as_deref_mut(),
                pool,
                kt,
                il,
                seq_len,
                false,
                false,
                self.exact_f32_kv_gradcheck,
            );
        }

        // 2. Masked chunked CE (with optional `final_logit_softcapping`)
        let mean_loss = masked_chunked_ce_fwd_bwd(
            accel.as_deref_mut(),
            pool,
            kt,
            &self.model.output_norm,
            c.rms_eps,
            &self.model.output,
            c.final_logit_softcapping,
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

        let track_ple_global_grad = self.model.ple_global.as_ref().is_some_and(|pg| {
            adapter.pairs.contains_key(pg.model_proj.name)
                || adapter.pairs.contains_key(self.model.token_embd.name)
        });
        if track_ple_global_grad && ple_total > 0 {
            self.arena.d_ple_inputs[..seq_len * ple_total].fill(0.0);
        }
        for il_kv in 0..c.n_layer_kv_from_start {
            let kv_dim = c.n_head_kv * c.head_dim(il_kv);
            self.arena.dk_shared[il_kv][..seq_len * kv_dim].fill(0.0);
            self.arena.dv_shared[il_kv][..seq_len * kv_dim].fill(0.0);
        }

        // 3. Reverse layer loop (`il = n_layers - 1 ..= 0`)
        for il in (0..c.n_layers).rev() {
            self.arena.load_layer_ckpt(il, seq_len, d);
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                accel.as_deref_mut(),
                pool,
                kt,
                il,
                seq_len,
                true,
                true,
                self.exact_f32_kv_gradcheck,
            );

            let lw = &self.model.layers[il];
            let hd = c.head_dim(il);
            let rope_dim = c.rope_dim(il);
            let q_dim = c.n_head * hd;
            let kv_dim = c.n_head_kv * hd;
            let is_swa = c.is_swa[il];
            let window = if is_swa { Some(c.n_swa) } else { None };
            let n_ff = c.n_ff_per_layer[il];
            let kv_src = c.kv_source[il];

            if let Some(scale) = lw.out_scale {
                for v in &mut self.arena.dx[..seq_len * d] {
                    *v *= scale;
                }
            }

            // 3a. Optional per-layer PLE backward
            if let Some(ref ple_l) = lw.ple {
                for t in 0..seq_len {
                    rms_norm_bwd_f32(
                        &mut self.arena.d_post[t * d..(t + 1) * d],
                        None,
                        &self.arena.ple_proj_l_raw[t * d..(t + 1) * d],
                        &ple_l.post_norm,
                        &self.arena.dx[t * d..(t + 1) * d],
                        c.rms_eps,
                    );
                }
                linear_bwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &ple_l.proj,
                    adapter,
                    grads,
                    &self.arena.ple_gated[..seq_len * ple_dim],
                    &self.arena.d_post[..seq_len * d],
                    &mut self.arena.d_ple_gated[..seq_len * ple_dim],
                    seq_len,
                    false,
                    &mut self.arena.lora_h,
                    &mut self.arena.lora_dh,
                    &mut self.arena.lora_bt,
                );

                for t in 0..seq_len {
                    let ple_off = t * ple_total + il * ple_dim;
                    let ple_in_t = &self.arena.ple_inputs[ple_off..ple_off + ple_dim];
                    let gp_t = &self.arena.ple_gate_pre[t * ple_dim..(t + 1) * ple_dim];
                    let dgd_t = &self.arena.d_ple_gated[t * ple_dim..(t + 1) * ple_dim];
                    let dgp_t = &mut self.arena.d_ple_gate_pre[t * ple_dim..(t + 1) * ple_dim];
                    for j in 0..ple_dim {
                        let dy = dgd_t[j];
                        let gp = gp_t[j];
                        dgp_t[j] = dy * ple_in_t[j] * gelu_tanh_bwd_f32(gp);
                        if track_ple_global_grad {
                            self.arena.d_ple_inputs[ple_off + j] += dy * gelu_tanh_f32(gp);
                        }
                    }
                }

                linear_bwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &ple_l.inp_gate,
                    adapter,
                    grads,
                    &self.arena.x_after_ffn[..seq_len * d],
                    &self.arena.d_ple_gate_pre[..seq_len * ple_dim],
                    &mut self.arena.dx[..seq_len * d],
                    seq_len,
                    true,
                    &mut self.arena.lora_h,
                    &mut self.arena.lora_dh,
                    &mut self.arena.lora_bt,
                );
            }

            // 3b. GeGLU FFN + `post_ffw_norm` + `ffn_norm` backward
            for t in 0..seq_len {
                rms_norm_bwd_f32(
                    &mut self.arena.d_post[t * d..(t + 1) * d],
                    None,
                    &self.arena.ffn_down_raw[t * d..(t + 1) * d],
                    &lw.post_ffw_norm,
                    &self.arena.dx[t * d..(t + 1) * d],
                    c.rms_eps,
                );
            }
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_down,
                adapter,
                grads,
                &self.arena.geglu_out[..seq_len * n_ff],
                &self.arena.d_post[..seq_len * d],
                &mut self.arena.d_geglu[..seq_len * n_ff],
                seq_len,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
            geglu_bwd_f32(
                &mut self.arena.d_gate[..seq_len * n_ff],
                &mut self.arena.d_up[..seq_len * n_ff],
                &self.arena.gate_pre[..seq_len * n_ff],
                &self.arena.up_pre[..seq_len * n_ff],
                &self.arena.d_geglu[..seq_len * n_ff],
            );
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.w_gate,
                adapter,
                grads,
                &self.arena.x_ffn_norm[..seq_len * d],
                &self.arena.d_gate[..seq_len * n_ff],
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
                &self.arena.d_up[..seq_len * n_ff],
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
                    &self.arena.attn_res[t * d..(t + 1) * d],
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

            // 3c. Attention (`post_attn_norm` -> `wo` -> SWA/Full GQA -> shared KV fan-in)
            for t in 0..seq_len {
                rms_norm_bwd_f32(
                    &mut self.arena.d_post[t * d..(t + 1) * d],
                    None,
                    &self.arena.wo_raw[t * d..(t + 1) * d],
                    &lw.post_attn_norm,
                    &self.arena.dx[t * d..(t + 1) * d],
                    c.rms_eps,
                );
            }
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &lw.wo,
                adapter,
                grads,
                &self.arena.attn_out[..seq_len * q_dim],
                &self.arena.d_post[..seq_len * d],
                &mut self.arena.d_attn_out[..seq_len * q_dim],
                seq_len,
                false,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );

            if self.exact_f32_kv_gradcheck {
                gqa_attn_f32_bwd_ext_softcap(
                    kt,
                    &mut self.arena.dq[..seq_len * q_dim],
                    &mut self.arena.dk_cur[..seq_len * kv_dim],
                    &mut self.arena.dv_cur[..seq_len * kv_dim],
                    &self.arena.q_rope[..seq_len * q_dim],
                    &self.arena.own_k_f32[kv_src][..seq_len * kv_dim],
                    &self.arena.own_v_f32[kv_src][..seq_len * kv_dim],
                    &self.arena.d_attn_out[..seq_len * q_dim],
                    c.n_head,
                    c.n_head_kv,
                    hd,
                    seq_len,
                    c.attn_scale,
                    window,
                    c.attn_logit_softcapping,
                );
            } else {
                gqa_attention_f16_kv_bwd_softcap_f32(
                    kt,
                    &mut self.arena.dq[..seq_len * q_dim],
                    &mut self.arena.dk_cur[..seq_len * kv_dim],
                    &mut self.arena.dv_cur[..seq_len * kv_dim],
                    &self.arena.q_rope[..seq_len * q_dim],
                    &self.arena.own_k_f16[kv_src][..seq_len * kv_dim],
                    &self.arena.own_v_f16[kv_src][..seq_len * kv_dim],
                    &self.arena.d_attn_out[..seq_len * q_dim],
                    c.n_head,
                    c.n_head_kv,
                    hd,
                    seq_len,
                    c.attn_scale,
                    window,
                    c.attn_logit_softcapping,
                );
            }

            // Fan-in `(dk_cur, dv_cur)` into `kv_src`'s shared KV gradient accumulators
            (kt.axpy_f32)(
                &mut self.arena.dk_shared[kv_src][..seq_len * kv_dim],
                1.0,
                &self.arena.dk_cur[..seq_len * kv_dim],
            );
            (kt.axpy_f32)(
                &mut self.arena.dv_shared[kv_src][..seq_len * kv_dim],
                1.0,
                &self.arena.dv_cur[..seq_len * kv_dim],
            );

            for t in 0..seq_len {
                let dqt = &mut self.arena.dq[t * q_dim..(t + 1) * q_dim];
                if is_swa {
                    rope_compute_cos_sin(
                        &self.model.rope_inv_freqs_swa,
                        t,
                        &mut self.arena.rope_cos_swa,
                        &mut self.arena.rope_sin_swa,
                    );
                    rope_bwd_inplace_tables(
                        dqt,
                        c.n_head,
                        hd,
                        rope_dim,
                        &self.arena.rope_cos_swa,
                        &self.arena.rope_sin_swa,
                        RopeType::Neox,
                    );
                } else {
                    rope_compute_cos_sin(
                        &self.model.rope_inv_freqs_full,
                        t,
                        &mut self.arena.rope_cos_full,
                        &mut self.arena.rope_sin_full,
                    );
                    rope_bwd_inplace_tables(
                        dqt,
                        c.n_head,
                        hd,
                        rope_dim,
                        &self.arena.rope_cos_full,
                        &self.arena.rope_sin_full,
                        RopeType::Neox,
                    );
                }

                if let Some(ref qn) = lw.q_norm {
                    let q_pre_t = &self.arena.q_pre_norm[t * q_dim..(t + 1) * q_dim];
                    for h in 0..c.n_head {
                        let s = h * hd;
                        let e = s + hd;
                        rms_norm_bwd_f32(
                            &mut self.arena.tmp_head[..hd],
                            None,
                            &q_pre_t[s..e],
                            qn,
                            &dqt[s..e],
                            c.rms_eps,
                        );
                        dqt[s..e].copy_from_slice(&self.arena.tmp_head[..hd]);
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

            // Own-KV layer (`il < n_layer_kv_from_start`): all upper layers sharing `il` have now
            // accumulated their gradients into `dk_shared[il]` and `dv_shared[il]`.
            if let Some(ref kv_w) = lw.kv {
                self.arena.dk_cur[..seq_len * kv_dim]
                    .copy_from_slice(&self.arena.dk_shared[il][..seq_len * kv_dim]);
                self.arena.dv_cur[..seq_len * kv_dim]
                    .copy_from_slice(&self.arena.dv_shared[il][..seq_len * kv_dim]);

                for t in 0..seq_len {
                    let dkt = &mut self.arena.dk_cur[t * kv_dim..(t + 1) * kv_dim];
                    let dvt = &mut self.arena.dv_cur[t * kv_dim..(t + 1) * kv_dim];
                    if is_swa {
                        rope_compute_cos_sin(
                            &self.model.rope_inv_freqs_swa,
                            t,
                            &mut self.arena.rope_cos_swa,
                            &mut self.arena.rope_sin_swa,
                        );
                        rope_bwd_inplace_tables(
                            dkt,
                            c.n_head_kv,
                            hd,
                            rope_dim,
                            &self.arena.rope_cos_swa,
                            &self.arena.rope_sin_swa,
                            RopeType::Neox,
                        );
                    } else {
                        rope_compute_cos_sin(
                            &self.model.rope_inv_freqs_full,
                            t,
                            &mut self.arena.rope_cos_full,
                            &mut self.arena.rope_sin_full,
                        );
                        rope_bwd_inplace_tables(
                            dkt,
                            c.n_head_kv,
                            hd,
                            rope_dim,
                            &self.arena.rope_cos_full,
                            &self.arena.rope_sin_full,
                            RopeType::Neox,
                        );
                    }

                    let k_pre_t = &self.arena.k_pre_norm[t * kv_dim..(t + 1) * kv_dim];
                    let v_pre_t = &self.arena.v_pre_norm[t * kv_dim..(t + 1) * kv_dim];
                    for h in 0..c.n_head_kv {
                        let s = h * hd;
                        let e = s + hd;
                        if let Some(ref kn) = kv_w.k_norm {
                            rms_norm_bwd_f32(
                                &mut self.arena.tmp_head[..hd],
                                None,
                                &k_pre_t[s..e],
                                kn,
                                &dkt[s..e],
                                c.rms_eps,
                            );
                            dkt[s..e].copy_from_slice(&self.arena.tmp_head[..hd]);
                        }

                        if c.norm_v {
                            rms_norm_unit_bwd_f32(
                                &mut self.arena.tmp_head[..hd],
                                &v_pre_t[s..e],
                                &dvt[s..e],
                                c.rms_eps,
                            );
                            dvt[s..e].copy_from_slice(&self.arena.tmp_head[..hd]);
                        }
                    }
                }

                if let Some(ref wv) = kv_w.wv {
                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        &kv_w.wk,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.dk_cur[..seq_len * kv_dim],
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
                        &self.arena.dv_cur[..seq_len * kv_dim],
                        &mut self.arena.d_x_norm[..seq_len * d],
                        seq_len,
                        true,
                        &mut self.arena.lora_h,
                        &mut self.arena.lora_dh,
                        &mut self.arena.lora_bt,
                    );
                } else {
                    (kt.axpy_f32)(
                        &mut self.arena.dk_cur[..seq_len * kv_dim],
                        1.0,
                        &self.arena.dv_cur[..seq_len * kv_dim],
                    );
                    linear_bwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        &kv_w.wk,
                        adapter,
                        grads,
                        &self.arena.x_attn_norm[..seq_len * d],
                        &self.arena.dk_cur[..seq_len * kv_dim],
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

        // 4. Global PLE backward (`per_layer_model_proj` -> `x_embd_scaled`) when active
        if track_ple_global_grad && let Some(ref ple_g) = self.model.ple_global {
            let proj_eps = c.rms_eps * (d as f32);
            for t in 0..seq_len {
                for il in 0..c.n_layers {
                    let s = t * ple_total + il * ple_dim;
                    let e = s + ple_dim;
                    rms_norm_bwd_f32(
                        &mut self.arena.d_ple_proj_raw[s..e],
                        None,
                        &self.arena.ple_proj_raw[s..e],
                        &ple_g.proj_norm_scaled,
                        &self.arena.d_ple_inputs[s..e],
                        proj_eps,
                    );
                }
            }
            linear_bwd_with_accel(
                accel,
                pool,
                kt,
                &ple_g.model_proj,
                adapter,
                grads,
                &self.arena.x_embd_scaled[..seq_len * d],
                &self.arena.d_ple_proj_raw[..seq_len * ple_total],
                &mut self.arena.dx[..seq_len * d],
                seq_len,
                true,
                &mut self.arena.lora_h,
                &mut self.arena.lora_dh,
                &mut self.arena.lora_bt,
            );
        }

        embed_lora_backward(
            self.model.token_embd.name,
            adapter,
            grads,
            input_toks,
            &self.arena.dx[..seq_len * d],
            d,
            (d as f32).sqrt(),
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
        if self.exact_f32_kv_gradcheck {
            self.arena.ensure_f32_kv(&self.model.cfg);
        }
        self.sync_bound_lora(adapter);
        let kt = active_kernels();
        let c = &self.model.cfg;
        let d = c.n_embd;

        Self::embed_and_ple_forward(
            self.model,
            &self.bound_lora,
            &mut self.arena,
            None,
            pool,
            kt,
            input_toks,
        );
        for il in 0..c.n_layers {
            Self::run_layer_forward(
                self.model,
                &self.bound_lora,
                &mut self.arena,
                None,
                pool,
                kt,
                il,
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
            c.final_logit_softcapping,
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
