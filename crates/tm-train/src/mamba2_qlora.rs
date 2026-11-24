//! CPU QLoRA trainer for unified Mamba-2 & Hybrid architectures (`mamba2`, `nemotron_h`, `falcon-h1`, `granitehybrid`, `ADR-017`).
//!
//! Key guarantees:
//! 1. **0-ULP Forward Parity Gate**: Independent training forward uses the exact same `tm-cpu`
//!    kernels and `MAX_CPU_BATCH` chunking as `Mamba2Model::forward_batch` (`forward_token` for `T == 1`),
//!    matching `Mamba2Model` to **0 ULP (`to_bits()`)**.
//! 2. **Boundary SSM State Checkpointing (`SSM_CKPT_STRIDE = 8`)**: Saves selective state `S`
//!    every 8 tokens during single-layer recomputation, bounding recurrent activation RAM while
//!    executing exact reverse-time BPTT through `mamba2_scan` + `ssm_conv1d` + grouped `ssm_norm`.
//! 3. **`#![forbid(unsafe_code)]`** (inherited from `tm-train` crate root) + zero heap allocations inside `forward_backward_microbatch`.

use tm_core::f16_to_f32;
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, ThreadPool, active_kernels,
    gqa_attention_f16_kv_batch_parallel, gqa_attention_f16_kv_bwd_ext_f32,
    gqa_attention_f16_kv_ext_batch_parallel, gqa_attention_f16_kv_ext_parallel,
    gqa_attention_f16_kv_parallel, mamba2_scan_parallel, matmul_f32, matmul_pair_f32,
    matmul_triple_f32, matvec_f32, matvec_pair_f32, matvec_triple_f32, relu2_bwd_f32,
    relu2_inplace_parallel, rms_norm, rms_norm_batch_parallel, rms_norm_bwd_f32,
    rope_bwd_inplace_tables, rope_compute_cos_sin, rope_inplace_tables, softplus_f32,
    swiglu_bwd_f32, swiglu_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch, LoraSet};
use tm_models::{
    Mamba2AttnWeights, Mamba2Block, Mamba2FfnWeights, Mamba2Model, Mamba2SsmWeights,
    decode_embedding_row, mamba2_conv1d_silu_split, mamba2_gate_and_grouped_norm,
};

use crate::accel::{LinearAccel, linear_bwd_with_accel, linear_fwd_with_accel};
use crate::optim::{LoraGradStore, LoraTargets};
use crate::qlora::{CE_CHUNK_TOKENS, CheckpointMode};

const SSM_CKPT_STRIDE: usize = 8;

struct Mamba2TrainArena {
    max_seq: usize,
    ckpt_mode: CheckpointMode,
    ckpt_f16: Vec<u16>,
    ckpt_f32: Vec<f32>,

    x_cur: Vec<f32>,
    x_layer_in: Vec<f32>,
    dx: Vec<f32>,

    x_attn_norm: Vec<f32>,
    x_mid: Vec<f32>,
    x_ffn_norm: Vec<f32>,
    xb2_chunk: Vec<f32>,
    d_branch: Vec<f32>,
    d_x_norm: Vec<f32>,
    d_x_norm_par: Vec<f32>,

    // SSM forward & recompute buffers
    in_proj: Vec<f32>,
    ssm_z: Vec<f32>,
    ssm_xbc: Vec<f32>,
    ssm_dt: Vec<f32>,
    ssm_conv_pre_silu: Vec<f32>,
    ssm_x: Vec<f32>,
    ssm_b: Vec<f32>,
    ssm_c: Vec<f32>,
    ssm_y_scan: Vec<f32>,
    ssm_y_gated: Vec<f32>,
    ssm_y_out: Vec<f32>,
    ssm_state_cur: Vec<f32>,
    ssm_conv_hist: Vec<f32>,
    /// Boundary state checkpoints `[n_ckpts, d_inner * d_state]`, `n_ckpts = max_seq / 8 + 2`.
    ssm_state_ckpts: Vec<f32>,
    /// 8-step local window states `[SSM_CKPT_STRIDE + 1, d_inner * d_state]` for reverse scan.
    ssm_win_states: Vec<f32>,

    // SSM backward buffers
    d_ssm_y_out: Vec<f32>,
    d_ssm_u: Vec<f32>,
    d_ssm_z: Vec<f32>,
    d_ssm_y_scan: Vec<f32>,
    d_ssm_x: Vec<f32>,
    d_ssm_b_f64: Vec<f64>,
    d_ssm_c_f64: Vec<f64>,
    d_ssm_dt: Vec<f32>,
    d_ssm_xbc: Vec<f32>,
    d_in_proj: Vec<f32>,
    d_ssm_state_f64: Vec<f64>,

    // Optional GQA attention buffers
    q_rope: Vec<f32>,
    k_rope_f32: Vec<f32>,
    v_full_f32: Vec<f32>,
    k_f16: Vec<u16>,
    v_f16: Vec<u16>,
    attn_out: Vec<f32>,
    d_attn_out: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,

    // Optional FFN buffers
    gate_pre: Vec<f32>,
    up_pre: Vec<f32>,
    ffn_act: Vec<f32>,
    d_ffn_act: Vec<f32>,
    d_gate: Vec<f32>,
    d_up: Vec<f32>,

    // Scratch & CE
    tmp_embd: Vec<f32>,
    tmp_group: Vec<f32>,
    ce_pos: Vec<(usize, u32)>,
    ce_x_sub: Vec<f32>,
    ce_x_norm: Vec<f32>,
    ce_logits: Vec<f32>,
    ce_dx_norm: Vec<f32>,
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

impl Mamba2TrainArena {
    fn new(
        model: &Mamba2Model<'_>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        let c = &model.cfg;
        let d = c.n_embd;
        let d_inner = c.ssm_inner_size;
        let bc_dim = c.bc_dim();
        let conv_dim = c.conv_dim();
        let d_in_proj = c.d_in_proj();
        let n_heads = c.ssm_n_head;
        let state_elems = d_inner * c.ssm_state_size;
        let hist_elems = conv_dim * c.ssm_conv_kernel.saturating_sub(1);
        let n_ckpts = max_seq.div_ceil(SSM_CKPT_STRIDE) + 2;

        let max_q = c.max_q_dim();
        let max_kv = c.max_kv_dim();
        let max_ff = c.max_n_ff();
        let max_n_head = c.n_head_per_layer.iter().copied().max().unwrap_or(0);
        let max_cols = d
            .max(d_inner)
            .max(d_in_proj)
            .max(max_q)
            .max(max_ff)
            .max(c.vocab_size);
        let r_cap = max_rank.max(1);
        let max_d_out = d
            .max(d_in_proj)
            .max(max_q + 2 * max_kv)
            .max(2 * max_ff)
            .max(c.vocab_size);

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
            x_mid: vec![0.0; max_seq * d],
            x_ffn_norm: vec![0.0; max_seq * d],
            xb2_chunk: vec![0.0; max_seq * d],
            d_branch: vec![0.0; max_seq * d],
            d_x_norm: vec![0.0; max_seq * d],
            d_x_norm_par: if c.arch == "falcon-h1" {
                vec![0.0; max_seq * d]
            } else {
                Vec::new()
            },
            in_proj: vec![0.0; max_seq * d_in_proj],
            ssm_z: vec![0.0; max_seq * d_inner],
            ssm_xbc: vec![0.0; max_seq * conv_dim],
            ssm_dt: vec![0.0; max_seq * n_heads],
            ssm_conv_pre_silu: vec![0.0; max_seq * conv_dim],
            ssm_x: vec![0.0; max_seq * d_inner],
            ssm_b: vec![0.0; max_seq * bc_dim],
            ssm_c: vec![0.0; max_seq * bc_dim],
            ssm_y_scan: vec![0.0; max_seq * d_inner],
            ssm_y_gated: vec![0.0; max_seq * d_inner],
            ssm_y_out: vec![0.0; max_seq * d_inner],
            ssm_state_cur: vec![0.0; state_elems],
            ssm_conv_hist: vec![0.0; hist_elems],
            ssm_state_ckpts: vec![0.0; n_ckpts * state_elems],
            ssm_win_states: vec![0.0; (SSM_CKPT_STRIDE + 1) * state_elems],
            d_ssm_y_out: vec![0.0; max_seq * d_inner],
            d_ssm_u: vec![0.0; max_seq * d_inner],
            d_ssm_z: vec![0.0; max_seq * d_inner],
            d_ssm_y_scan: vec![0.0; max_seq * d_inner],
            d_ssm_x: vec![0.0; max_seq * d_inner],
            d_ssm_b_f64: vec![0.0; max_seq * bc_dim],
            d_ssm_c_f64: vec![0.0; max_seq * bc_dim],
            d_ssm_dt: vec![0.0; max_seq * n_heads],
            d_ssm_xbc: vec![0.0; max_seq * conv_dim],
            d_in_proj: vec![0.0; max_seq * d_in_proj],
            d_ssm_state_f64: vec![0.0; state_elems],
            q_rope: vec![0.0; max_seq * max_q],
            k_rope_f32: vec![0.0; max_seq * max_kv],
            v_full_f32: vec![0.0; max_seq * max_kv],
            k_f16: vec![0u16; max_seq * max_kv],
            v_f16: vec![0u16; max_seq * max_kv],
            attn_out: vec![0.0; max_seq * max_q],
            d_attn_out: vec![0.0; max_seq * max_q],
            dq: vec![0.0; max_seq * max_q],
            dk: vec![0.0; max_seq * max_kv],
            dv: vec![0.0; max_seq * max_kv],
            gate_pre: vec![0.0; max_seq * max_ff],
            up_pre: vec![0.0; max_seq * max_ff],
            ffn_act: vec![0.0; max_seq * max_ff],
            d_ffn_act: vec![0.0; max_seq * max_ff],
            d_gate: vec![0.0; max_seq * max_ff],
            d_up: vec![0.0; max_seq * max_ff],
            tmp_embd: vec![0.0; d],
            tmp_group: vec![0.0; c.ssm_group_dim().max(1)],
            ce_pos: Vec::with_capacity(max_seq),
            ce_x_sub: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_x_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            ce_logits: vec![0.0; CE_CHUNK_TOKENS * c.vocab_size],
            ce_dx_norm: vec![0.0; CE_CHUNK_TOKENS * d],
            rope_cos: vec![0.0; c.rope_dim / 2],
            rope_sin: vec![0.0; c.rope_dim / 2],
            attn_scores: vec![0.0; max_n_head * max_seq],
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

/// Estimates anonymous dirty-page RAM required by [`Mamba2Trainer`] BEFORE allocating.
pub fn estimate_mamba2_qlora_dirty_bytes(
    model: &Mamba2Model<'_>,
    max_seq: usize,
    rank: usize,
    targets: &LoraTargets,
    ckpt_mode: CheckpointMode,
) -> u64 {
    let c = &model.cfg;
    let d = c.n_embd;
    let d_inner = c.ssm_inner_size;
    let conv_dim = c.conv_dim();
    let d_in_proj = c.d_in_proj();
    let state_elems = d_inner * c.ssm_state_size;
    let n_ckpts = max_seq.div_ceil(SSM_CKPT_STRIDE) + 2;

    let ckpt_elem = match ckpt_mode {
        CheckpointMode::Fp16 => 2usize,
        CheckpointMode::Fp32 => 4usize,
    };
    let ckpt_bytes = c.n_layers * max_seq * d * ckpt_elem;
    let ssm_arena = (n_ckpts + SSM_CKPT_STRIDE + 3) * state_elems * 4
        + max_seq * (3 * d_in_proj + 6 * d_inner + 3 * conv_dim + 6 * d + 6 * c.max_n_ff()) * 4
        + CE_CHUNK_TOKENS * (c.vocab_size + 3 * d) * 4;
    let est_lora = c.n_layers
        * rank
        * (if targets.q || targets.ssm_gates {
            d + d_in_proj
        } else {
            0
        } + if targets.o { d_inner + d } else { 0 }
            + if targets.up {
                2 * (d + c.max_n_ff())
            } else {
                0
            })
        * 24;
    (ckpt_bytes + ssm_arena + est_lora) as u64
}

/// Independent QLoRA trainer for `Mamba2Model` (`mamba2`, `nemotron_h`, `falcon-h1`, `granitehybrid`).
pub struct Mamba2Trainer<'a> {
    pub model: &'a Mamba2Model<'a>,
    arena: Mamba2TrainArena,
    bound_lora: BoundLora,
    pub exact_f32_kv_gradcheck: bool,
}

impl<'a> Mamba2Trainer<'a> {
    pub fn new(
        model: &'a Mamba2Model<'a>,
        max_seq: usize,
        max_rank: usize,
        ckpt_mode: CheckpointMode,
    ) -> Self {
        Self {
            model,
            arena: Mamba2TrainArena::new(model, max_seq.max(2), max_rank, ckpt_mode),
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
    fn run_ssm_forward(
        model: &Mamba2Model<'_>,
        ssm: &Mamba2SsmWeights<'_>,
        bound: &BoundLora,
        arena: &mut Mamba2TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        seq_len: usize,
        save_recompute: bool,
    ) {
        let c = &model.cfg;
        let d = c.n_embd;
        let d_inner = c.ssm_inner_size;
        let bc_dim = c.bc_dim();
        let conv_dim = c.conv_dim();
        let d_in_proj = c.d_in_proj();
        let n_heads = c.ssm_n_head;
        let d_conv = c.ssm_conv_kernel;
        let state_elems = d_inner * c.ssm_state_size;

        let slot_in = bound.slot_for(ssm.ssm_in.name);
        let slot_out = bound.slot_for(ssm.ssm_out.name);

        arena.ssm_state_cur.fill(0.0);
        arena.ssm_conv_hist.fill(0.0);

        let single_tok_mode = seq_len == 1;
        let mut offset = 0usize;
        while offset < seq_len {
            let b = (seq_len - offset).min(MAX_CPU_BATCH);
            let single_chunk = b == 1;
            let xn = &arena.x_attn_norm[offset * d..(offset + b) * d];

            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &ssm.ssm_in,
                bound,
                slot_in,
                xn,
                &mut arena.in_proj[offset * d_in_proj..(offset + b) * d_in_proj],
                b,
                single_tok_mode || single_chunk,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );

            for t in 0..b {
                let g_t = offset + t;
                let row = &arena.in_proj[g_t * d_in_proj..(g_t + 1) * d_in_proj];
                arena.ssm_z[g_t * d_inner..(g_t + 1) * d_inner].copy_from_slice(&row[..d_inner]);
                arena.ssm_xbc[g_t * conv_dim..(g_t + 1) * conv_dim]
                    .copy_from_slice(&row[d_inner..d_inner + conv_dim]);
                arena.ssm_dt[g_t * n_heads..(g_t + 1) * n_heads]
                    .copy_from_slice(&row[d_inner + conv_dim..d_in_proj]);
            }

            if save_recompute {
                // Record exact pre-SiLU conv1d values for backward
                let hist_len = d_conv.saturating_sub(1);
                for t in 0..b {
                    let g_t = offset + t;
                    for ch in 0..conv_dim {
                        let w_ch = &ssm.conv_w[ch * d_conv..(ch + 1) * d_conv];
                        let mut acc = ssm.conv_b.as_ref().map_or(0.0f32, |bv| bv[ch]);
                        for (k, &wk) in w_ch[..hist_len].iter().enumerate() {
                            let v = if g_t + k >= hist_len {
                                arena.ssm_xbc[(g_t + k - hist_len) * conv_dim + ch]
                            } else {
                                0.0
                            };
                            acc = v.mul_add(wk, acc);
                        }
                        if d_conv > 0 {
                            acc = arena.ssm_xbc[g_t * conv_dim + ch].mul_add(w_ch[hist_len], acc);
                        }
                        arena.ssm_conv_pre_silu[g_t * conv_dim + ch] = acc;
                    }
                }
            }

            mamba2_conv1d_silu_split(
                &arena.ssm_xbc[offset * conv_dim..(offset + b) * conv_dim],
                &mut arena.ssm_conv_hist,
                &ssm.conv_w,
                ssm.conv_b.as_deref(),
                &mut arena.ssm_x[offset * d_inner..(offset + b) * d_inner],
                &mut arena.ssm_b[offset * bc_dim..(offset + b) * bc_dim],
                &mut arena.ssm_c[offset * bc_dim..(offset + b) * bc_dim],
                d_inner,
                bc_dim,
                d_conv,
                b,
            );

            if save_recompute {
                // Step in `SSM_CKPT_STRIDE` sub-windows so boundary states are saved every 8 steps
                let mut sub = 0usize;
                while sub < b {
                    let g_t = offset + sub;
                    if g_t.is_multiple_of(SSM_CKPT_STRIDE) {
                        let ck_idx = g_t / SSM_CKPT_STRIDE;
                        arena.ssm_state_ckpts[ck_idx * state_elems..(ck_idx + 1) * state_elems]
                            .copy_from_slice(&arena.ssm_state_cur);
                    }
                    let next_ck = ((g_t / SSM_CKPT_STRIDE) + 1) * SSM_CKPT_STRIDE;
                    let step_len = (next_ck - g_t).min(b - sub);
                    mamba2_scan_parallel(
                        pool,
                        kt,
                        &arena.ssm_x[g_t * d_inner..(g_t + step_len) * d_inner],
                        &arena.ssm_b[g_t * bc_dim..(g_t + step_len) * bc_dim],
                        &arena.ssm_c[g_t * bc_dim..(g_t + step_len) * bc_dim],
                        &arena.ssm_dt[g_t * n_heads..(g_t + step_len) * n_heads],
                        &ssm.dt_bias,
                        &ssm.ssm_a,
                        &ssm.ssm_d,
                        &mut arena.ssm_state_cur,
                        &mut arena.ssm_y_scan[g_t * d_inner..(g_t + step_len) * d_inner],
                        n_heads,
                        c.ssm_n_group,
                        c.ssm_head_dim(),
                        c.ssm_state_size,
                        step_len,
                    );
                    sub += step_len;
                }
            } else {
                mamba2_scan_parallel(
                    pool,
                    kt,
                    &arena.ssm_x[offset * d_inner..(offset + b) * d_inner],
                    &arena.ssm_b[offset * bc_dim..(offset + b) * bc_dim],
                    &arena.ssm_c[offset * bc_dim..(offset + b) * bc_dim],
                    &arena.ssm_dt[offset * n_heads..(offset + b) * n_heads],
                    &ssm.dt_bias,
                    &ssm.ssm_a,
                    &ssm.ssm_d,
                    &mut arena.ssm_state_cur,
                    &mut arena.ssm_y_scan[offset * d_inner..(offset + b) * d_inner],
                    n_heads,
                    c.ssm_n_group,
                    c.ssm_head_dim(),
                    c.ssm_state_size,
                    b,
                );
            }

            arena.ssm_y_out[offset * d_inner..(offset + b) * d_inner]
                .copy_from_slice(&arena.ssm_y_scan[offset * d_inner..(offset + b) * d_inner]);
            if save_recompute && ssm.ssm_norm.is_some() {
                for i in (offset * d_inner)..((offset + b) * d_inner) {
                    let zv = arena.ssm_z[i];
                    arena.ssm_y_gated[i] = arena.ssm_y_scan[i] * (zv / (1.0 + (-zv).exp()));
                }
            }
            mamba2_gate_and_grouped_norm(
                kt,
                &mut arena.ssm_y_out[offset * d_inner..(offset + b) * d_inner],
                &arena.ssm_z[offset * d_inner..(offset + b) * d_inner],
                ssm.ssm_norm.as_deref(),
                &mut arena.tmp_group,
                d_inner,
                c.ssm_n_group,
                c.ssm_norm_eps,
                b,
            );

            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &ssm.ssm_out,
                bound,
                slot_out,
                &arena.ssm_y_out[offset * d_inner..(offset + b) * d_inner],
                &mut arena.xb2_chunk[offset * d..(offset + b) * d],
                b,
                single_tok_mode || single_chunk,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );

            offset += b;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_ssm(
        model: &Mamba2Model<'_>,
        ssm: &Mamba2SsmWeights<'_>,
        adapter: &LoraAdapter,
        grads: &mut LoraGradStore,
        arena: &mut Mamba2TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        seq_len: usize,
        d_branch_in: &[f32],
        d_x_norm_out: &mut [f32],
        accumulate_dx: bool,
    ) {
        let c = &model.cfg;
        let d = c.n_embd;
        let d_inner = c.ssm_inner_size;
        let bc_dim = c.bc_dim();
        let conv_dim = c.conv_dim();
        let d_in_proj = c.d_in_proj();
        let n_heads = c.ssm_n_head;
        let n_groups = c.ssm_n_group;
        let group_ratio = n_heads / n_groups;
        let head_dim = c.ssm_head_dim();
        let d_state = c.ssm_state_size;
        let group_dim = c.ssm_group_dim();
        let state_elems = d_inner * d_state;
        let d_conv = c.ssm_conv_kernel;

        // 1. `ssm_out` backward -> `d_ssm_y_out`
        linear_bwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            &ssm.ssm_out,
            adapter,
            grads,
            &arena.ssm_y_out[..seq_len * d_inner],
            d_branch_in,
            &mut arena.d_ssm_y_out[..seq_len * d_inner],
            seq_len,
            false,
            &mut arena.lora_h,
            &mut arena.lora_dh,
            &mut arena.lora_bt,
        );

        // 2. Grouped `ssm_norm` backward + `silu(z) * y_scan` backward
        if let Some(ref nw) = ssm.ssm_norm {
            for t in 0..seq_len {
                for g in 0..n_groups {
                    let s = t * d_inner + g * group_dim;
                    let e = s + group_dim;
                    let ws = g * group_dim;
                    rms_norm_bwd_f32(
                        &mut arena.d_ssm_u[s..e],
                        None,
                        &arena.ssm_y_gated[s..e],
                        &nw[ws..ws + group_dim],
                        &arena.d_ssm_y_out[s..e],
                        c.ssm_norm_eps,
                    );
                }
            }
        } else {
            arena.d_ssm_u[..seq_len * d_inner]
                .copy_from_slice(&arena.d_ssm_y_out[..seq_len * d_inner]);
        }

        swiglu_bwd_f32(
            &mut arena.d_ssm_z[..seq_len * d_inner],
            &mut arena.d_ssm_y_scan[..seq_len * d_inner],
            &arena.ssm_z[..seq_len * d_inner],
            &arena.ssm_y_scan[..seq_len * d_inner],
            &arena.d_ssm_u[..seq_len * d_inner],
        );

        // 3. Reverse-time selective state scan backward using 8-token boundary checkpoints
        arena.d_ssm_state_f64.fill(0.0);
        arena.d_ssm_b_f64[..seq_len * bc_dim].fill(0.0);
        arena.d_ssm_c_f64[..seq_len * bc_dim].fill(0.0);

        let n_ck_chunks = seq_len.div_ceil(SSM_CKPT_STRIDE);
        for ck_idx in (0..n_ck_chunks).rev() {
            let t0 = ck_idx * SSM_CKPT_STRIDE;
            let t1 = (t0 + SSM_CKPT_STRIDE).min(seq_len);
            let win_len = t1 - t0;

            // Replay `win_len` states forward from checkpoint `ck_idx` into `ssm_win_states[0..=win_len]`
            arena.ssm_win_states[..state_elems].copy_from_slice(
                &arena.ssm_state_ckpts[ck_idx * state_elems..(ck_idx + 1) * state_elems],
            );
            for lt in 0..win_len {
                let g_t = t0 + lt;
                let (prev_part, next_part) =
                    arena.ssm_win_states.split_at_mut((lt + 1) * state_elems);
                next_part[..state_elems]
                    .copy_from_slice(&prev_part[lt * state_elems..(lt + 1) * state_elems]);
                let mut dummy_y = [0.0f32; 1];
                let _ = &mut dummy_y;
                mamba2_scan_parallel(
                    pool,
                    kt,
                    &arena.ssm_x[g_t * d_inner..(g_t + 1) * d_inner],
                    &arena.ssm_b[g_t * bc_dim..(g_t + 1) * bc_dim],
                    &arena.ssm_c[g_t * bc_dim..(g_t + 1) * bc_dim],
                    &arena.ssm_dt[g_t * n_heads..(g_t + 1) * n_heads],
                    &ssm.dt_bias,
                    &ssm.ssm_a,
                    &ssm.ssm_d,
                    &mut next_part[..state_elems],
                    &mut arena.ssm_y_out[..d_inner],
                    n_heads,
                    n_groups,
                    head_dim,
                    d_state,
                    1,
                );
            }

            // Reverse scan `lt = (0..win_len).rev()`
            for lt in (0..win_len).rev() {
                let g_t = t0 + lt;
                let s_prev_all = &arena.ssm_win_states[lt * state_elems..(lt + 1) * state_elems];
                let s_cur_all =
                    &arena.ssm_win_states[(lt + 1) * state_elems..(lt + 2) * state_elems];

                for h in 0..n_heads {
                    let g = h / group_ratio;
                    let u_dt = arena.ssm_dt[g_t * n_heads + h] + ssm.dt_bias[h];
                    let dt_sp = softplus_f32(u_dt);
                    let a_h = ssm.ssm_a[h];
                    let d_h = ssm.ssm_d[h] as f64;
                    let da = (dt_sp * a_h).exp();
                    let da_f64 = da as f64;
                    let dt_f64 = dt_sp as f64;

                    let bc_off = g_t * bc_dim + g * d_state;
                    let b_g = &arena.ssm_b[bc_off..bc_off + d_state];
                    let c_g = &arena.ssm_c[bc_off..bc_off + d_state];
                    let db_g = &mut arena.d_ssm_b_f64[bc_off..bc_off + d_state];
                    let dc_g = &mut arena.d_ssm_c_f64[bc_off..bc_off + d_state];

                    let mut d_da_sum = 0.0f64;
                    let mut d_dt_sp_sum = 0.0f64;

                    for j in 0..head_dim {
                        let row = h * head_dim + j;
                        let s_off = row * d_state;
                        let x_j = arena.ssm_x[g_t * d_inner + row] as f64;
                        let dy_j = arena.d_ssm_y_scan[g_t * d_inner + row] as f64;
                        let s_prev = &s_prev_all[s_off..s_off + d_state];
                        let s_cur = &s_cur_all[s_off..s_off + d_state];
                        let ds_row = &mut arena.d_ssm_state_f64[s_off..s_off + d_state];

                        let mut dx_j = dy_j * d_h;
                        let x_dt = x_j * dt_f64;
                        for k in 0..d_state {
                            let ck = c_g[k] as f64;
                            let bk = b_g[k] as f64;
                            dc_g[k] += dy_j * (s_cur[k] as f64);
                            let ds_jk = ds_row[k] + dy_j * ck;
                            d_da_sum += ds_jk * (s_prev[k] as f64);
                            d_dt_sp_sum += ds_jk * bk * x_j;
                            dx_j += ds_jk * bk * dt_f64;
                            db_g[k] += ds_jk * x_dt;
                            ds_row[k] = ds_jk * da_f64;
                        }
                        arena.d_ssm_x[g_t * d_inner + row] = dx_j as f32;
                    }

                    let d_dt_sp_tot = d_dt_sp_sum + d_da_sum * da_f64 * (a_h as f64);
                    let sig = if u_dt > 20.0 {
                        1.0
                    } else {
                        1.0 / (1.0 + (-(u_dt as f64)).exp())
                    };
                    arena.d_ssm_dt[g_t * n_heads + h] = (d_dt_sp_tot * sig) as f32;
                }
            }
        }

        // 4. Causal Conv1d + SiLU backward -> `d_ssm_xbc`
        arena.d_ssm_xbc[..seq_len * conv_dim].fill(0.0);
        let hist_len = d_conv.saturating_sub(1);
        for t in 0..seq_len {
            for ch in 0..conv_dim {
                let d_post = if ch < d_inner {
                    arena.d_ssm_x[t * d_inner + ch] as f64
                } else if ch < d_inner + bc_dim {
                    arena.d_ssm_b_f64[t * bc_dim + (ch - d_inner)]
                } else {
                    arena.d_ssm_c_f64[t * bc_dim + (ch - d_inner - bc_dim)]
                };
                let raw = arena.ssm_conv_pre_silu[t * conv_dim + ch] as f64;
                let s = 1.0 / (1.0 + (-raw).exp());
                let d_conv_val = (d_post * s * (1.0 + raw * (1.0 - s))) as f32;

                let w_ch = &ssm.conv_w[ch * d_conv..(ch + 1) * d_conv];
                for (k, &wk) in w_ch[..hist_len].iter().enumerate() {
                    if t + k >= hist_len {
                        let src_t = t + k - hist_len;
                        arena.d_ssm_xbc[src_t * conv_dim + ch] += d_conv_val * wk;
                    }
                }
                if d_conv > 0 {
                    arena.d_ssm_xbc[t * conv_dim + ch] += d_conv_val * w_ch[hist_len];
                }
            }
        }

        // 5. Pack `(d_z, d_xBC, d_dt)` -> `d_in_proj` -> `ssm_in` backward
        for t in 0..seq_len {
            let dst = &mut arena.d_in_proj[t * d_in_proj..(t + 1) * d_in_proj];
            dst[..d_inner].copy_from_slice(&arena.d_ssm_z[t * d_inner..(t + 1) * d_inner]);
            dst[d_inner..d_inner + conv_dim]
                .copy_from_slice(&arena.d_ssm_xbc[t * conv_dim..(t + 1) * conv_dim]);
            dst[d_inner + conv_dim..d_in_proj]
                .copy_from_slice(&arena.d_ssm_dt[t * n_heads..(t + 1) * n_heads]);
        }

        linear_bwd_with_accel(
            accel,
            pool,
            kt,
            &ssm.ssm_in,
            adapter,
            grads,
            &arena.x_attn_norm[..seq_len * d],
            &arena.d_in_proj[..seq_len * d_in_proj],
            d_x_norm_out,
            seq_len,
            accumulate_dx,
            &mut arena.lora_h,
            &mut arena.lora_dh,
            &mut arena.lora_bt,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attn_forward(
        model: &Mamba2Model<'_>,
        attn: &Mamba2AttnWeights<'_>,
        bound: &BoundLora,
        arena: &mut Mamba2TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        seq_len: usize,
        exact_f32_kv: bool,
    ) {
        let c = &model.cfg;
        let d = c.n_embd;
        let hd = c.head_dim;
        let q_dim = attn.n_head * hd;
        let kv_dim = attn.n_head_kv * hd;
        let slot_wq = bound.slot_for(attn.wq.name);
        let slot_wk = bound.slot_for(attn.wk.name);
        let slot_wv = bound.slot_for(attn.wv.name);
        let slot_wo = bound.slot_for(attn.wo.name);
        let rope_freqs = model.rope_inv_freqs_for_ctx(arena.max_seq);

        let mut offset = 0usize;
        while offset < seq_len {
            let b = (seq_len - offset).min(MAX_CPU_BATCH);
            let single = b == 1;
            let xn = &arena.x_attn_norm[offset * d..(offset + b) * d];

            if accel.is_some() {
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &attn.wq,
                    bound,
                    slot_wq,
                    xn,
                    &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    b,
                    single,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &attn.wk,
                    bound,
                    slot_wk,
                    xn,
                    &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                    b,
                    single,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
                linear_fwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    &attn.wv,
                    bound,
                    slot_wv,
                    xn,
                    &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                    b,
                    single,
                    &mut arena.mv_scratch,
                    &mut arena.lora_scratch,
                );
            } else {
                if single {
                    matvec_triple_f32(
                        pool,
                        kt,
                        attn.wq.dtype,
                        attn.wq.bytes,
                        q_dim,
                        &mut arena.q_rope[offset * q_dim..(offset + 1) * q_dim],
                        attn.wk.dtype,
                        attn.wk.bytes,
                        kv_dim,
                        &mut arena.k_rope_f32[offset * kv_dim..(offset + 1) * kv_dim],
                        attn.wv.dtype,
                        attn.wv.bytes,
                        kv_dim,
                        &mut arena.v_full_f32[offset * kv_dim..(offset + 1) * kv_dim],
                        xn,
                        &mut arena.mv_scratch,
                        d,
                    );
                } else {
                    matmul_triple_f32(
                        pool,
                        kt,
                        attn.wq.dtype,
                        attn.wq.bytes,
                        q_dim,
                        &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                        attn.wk.dtype,
                        attn.wk.bytes,
                        kv_dim,
                        &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                        attn.wv.dtype,
                        attn.wv.bytes,
                        kv_dim,
                        &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                        xn,
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
                        xn,
                        &mut arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                        b,
                        attn.wq.row_offset,
                        q_dim,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear_slice(
                        slot_wk,
                        pool,
                        kt,
                        xn,
                        &mut arena.k_rope_f32[offset * kv_dim..(offset + b) * kv_dim],
                        b,
                        attn.wk.row_offset,
                        kv_dim,
                        &mut arena.lora_scratch,
                    );
                    bound.apply_linear_slice(
                        slot_wv,
                        pool,
                        kt,
                        xn,
                        &mut arena.v_full_f32[offset * kv_dim..(offset + b) * kv_dim],
                        b,
                        attn.wv.row_offset,
                        kv_dim,
                        &mut arena.lora_scratch,
                    );
                }
            }
            if let Some(ref bq) = attn.bq {
                for t in 0..b {
                    let row = &mut arena.q_rope[(offset + t) * q_dim..(offset + t + 1) * q_dim];
                    for (y, &bi) in row.iter_mut().zip(bq.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref bk) = attn.bk {
                for t in 0..b {
                    let row =
                        &mut arena.k_rope_f32[(offset + t) * kv_dim..(offset + t + 1) * kv_dim];
                    for (y, &bi) in row.iter_mut().zip(bk.iter()) {
                        *y += bi;
                    }
                }
            }
            if let Some(ref bv) = attn.bv {
                for t in 0..b {
                    let row =
                        &mut arena.v_full_f32[(offset + t) * kv_dim..(offset + t + 1) * kv_dim];
                    for (y, &bi) in row.iter_mut().zip(bv.iter()) {
                        *y += bi;
                    }
                }
            }

            for t in 0..b {
                let pos = offset + t;
                let qt = &mut arena.q_rope[pos * q_dim..(pos + 1) * q_dim];
                let kt_v = &mut arena.k_rope_f32[pos * kv_dim..(pos + 1) * kv_dim];
                let vt_v = &arena.v_full_f32[pos * kv_dim..(pos + 1) * kv_dim];
                if c.use_rope {
                    rope_compute_cos_sin(rope_freqs, pos, &mut arena.rope_cos, &mut arena.rope_sin);
                    rope_inplace_tables(
                        qt,
                        attn.n_head,
                        hd,
                        c.rope_dim,
                        &arena.rope_cos,
                        &arena.rope_sin,
                        c.rope_type,
                    );
                    rope_inplace_tables(
                        kt_v,
                        attn.n_head_kv,
                        hd,
                        c.rope_dim,
                        &arena.rope_cos,
                        &arena.rope_sin,
                        c.rope_type,
                    );
                }
                if !exact_f32_kv {
                    let kv_off = pos * kv_dim;
                    (kt.f32_to_f16_slice)(kt_v, &mut arena.k_f16[kv_off..kv_off + kv_dim]);
                    (kt.f32_to_f16_slice)(vt_v, &mut arena.v_f16[kv_off..kv_off + kv_dim]);
                }
            }

            if exact_f32_kv {
                gqa_f32_exact(
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.k_rope_f32[..(offset + b) * kv_dim],
                    &arena.v_full_f32[..(offset + b) * kv_dim],
                    attn.n_head,
                    attn.n_head_kv,
                    hd,
                    offset,
                    b,
                    c.attn_scale as f64,
                );
            } else if single {
                if c.custom_attn_scale {
                    gqa_attention_f16_kv_ext_parallel(
                        pool,
                        kt,
                        &mut arena.attn_out[offset * q_dim..(offset + 1) * q_dim],
                        &arena.q_rope[offset * q_dim..(offset + 1) * q_dim],
                        &arena.k_f16,
                        &arena.v_f16,
                        attn.n_head,
                        attn.n_head_kv,
                        hd,
                        offset,
                        c.attn_scale,
                        None,
                        arena.max_seq,
                        &mut arena.attn_scores,
                    );
                } else {
                    gqa_attention_f16_kv_parallel(
                        pool,
                        kt,
                        &mut arena.attn_out[offset * q_dim..(offset + 1) * q_dim],
                        &arena.q_rope[offset * q_dim..(offset + 1) * q_dim],
                        &arena.k_f16,
                        &arena.v_f16,
                        attn.n_head,
                        attn.n_head_kv,
                        hd,
                        offset + 1,
                        arena.max_seq,
                        &mut arena.attn_scores,
                    );
                }
            } else if c.custom_attn_scale {
                gqa_attention_f16_kv_ext_batch_parallel(
                    pool,
                    kt,
                    &mut arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                    &arena.q_rope[offset * q_dim..(offset + b) * q_dim],
                    &arena.k_f16,
                    &arena.v_f16,
                    attn.n_head,
                    attn.n_head_kv,
                    hd,
                    offset,
                    b,
                    c.attn_scale,
                    None,
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
                    attn.n_head,
                    attn.n_head_kv,
                    hd,
                    offset,
                    b,
                    arena.max_seq,
                    &mut arena.attn_scores,
                );
            }

            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &attn.wo,
                bound,
                slot_wo,
                &arena.attn_out[offset * q_dim..(offset + b) * q_dim],
                &mut arena.xb2_chunk[offset * d..(offset + b) * d],
                b,
                single,
                &mut arena.mv_scratch,
                &mut arena.lora_scratch,
            );
            if let Some(ref bo) = attn.bo {
                for t in 0..b {
                    let row = &mut arena.xb2_chunk[(offset + t) * d..(offset + t + 1) * d];
                    for (y, &bi) in row.iter_mut().zip(bo.iter()) {
                        *y += bi;
                    }
                }
            }
            offset += b;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_attn(
        model: &Mamba2Model<'_>,
        attn: &Mamba2AttnWeights<'_>,
        adapter: &LoraAdapter,
        grads: &mut LoraGradStore,
        arena: &mut Mamba2TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        seq_len: usize,
        exact_f32_kv: bool,
        d_branch_in: &[f32],
        d_x_norm_out: &mut [f32],
        accumulate_dx: bool,
    ) {
        let c = &model.cfg;
        let d = c.n_embd;
        let hd = c.head_dim;
        let q_dim = attn.n_head * hd;
        let kv_dim = attn.n_head_kv * hd;

        linear_bwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            &attn.wo,
            adapter,
            grads,
            &arena.attn_out[..seq_len * q_dim],
            d_branch_in,
            &mut arena.d_attn_out[..seq_len * q_dim],
            seq_len,
            false,
            &mut arena.lora_h,
            &mut arena.lora_dh,
            &mut arena.lora_bt,
        );

        if exact_f32_kv {
            gqa_f32_bwd_exact(
                kt,
                &mut arena.dq[..seq_len * q_dim],
                &mut arena.dk[..seq_len * kv_dim],
                &mut arena.dv[..seq_len * kv_dim],
                &arena.q_rope[..seq_len * q_dim],
                &arena.k_rope_f32[..seq_len * kv_dim],
                &arena.v_full_f32[..seq_len * kv_dim],
                &arena.d_attn_out[..seq_len * q_dim],
                attn.n_head,
                attn.n_head_kv,
                hd,
                seq_len,
                c.attn_scale as f64,
                &mut arena.bwd_probs,
                &mut arena.bwd_dp,
            );
        } else {
            gqa_attention_f16_kv_bwd_ext_f32(
                kt,
                &mut arena.dq[..seq_len * q_dim],
                &mut arena.dk[..seq_len * kv_dim],
                &mut arena.dv[..seq_len * kv_dim],
                &arena.q_rope[..seq_len * q_dim],
                &arena.k_f16[..seq_len * kv_dim],
                &arena.v_f16[..seq_len * kv_dim],
                &arena.d_attn_out[..seq_len * q_dim],
                attn.n_head,
                attn.n_head_kv,
                hd,
                seq_len,
                c.attn_scale,
                None,
            );
        }

        if c.use_rope {
            let rope_freqs = model.rope_inv_freqs_for_ctx(arena.max_seq);
            for t in 0..seq_len {
                rope_compute_cos_sin(rope_freqs, t, &mut arena.rope_cos, &mut arena.rope_sin);
                rope_bwd_inplace_tables(
                    &mut arena.dq[t * q_dim..(t + 1) * q_dim],
                    attn.n_head,
                    hd,
                    c.rope_dim,
                    &arena.rope_cos,
                    &arena.rope_sin,
                    c.rope_type,
                );
                rope_bwd_inplace_tables(
                    &mut arena.dk[t * kv_dim..(t + 1) * kv_dim],
                    attn.n_head_kv,
                    hd,
                    c.rope_dim,
                    &arena.rope_cos,
                    &arena.rope_sin,
                    c.rope_type,
                );
            }
        }

        linear_bwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            &attn.wq,
            adapter,
            grads,
            &arena.x_attn_norm[..seq_len * d],
            &arena.dq[..seq_len * q_dim],
            d_x_norm_out,
            seq_len,
            accumulate_dx,
            &mut arena.lora_h,
            &mut arena.lora_dh,
            &mut arena.lora_bt,
        );
        linear_bwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            &attn.wk,
            adapter,
            grads,
            &arena.x_attn_norm[..seq_len * d],
            &arena.dk[..seq_len * kv_dim],
            d_x_norm_out,
            seq_len,
            true,
            &mut arena.lora_h,
            &mut arena.lora_dh,
            &mut arena.lora_bt,
        );
        linear_bwd_with_accel(
            accel,
            pool,
            kt,
            &attn.wv,
            adapter,
            grads,
            &arena.x_attn_norm[..seq_len * d],
            &arena.dv[..seq_len * kv_dim],
            d_x_norm_out,
            seq_len,
            true,
            &mut arena.lora_h,
            &mut arena.lora_dh,
            &mut arena.lora_bt,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_ffn_forward(
        model: &Mamba2Model<'_>,
        ffn: &Mamba2FfnWeights<'_>,
        bound: &BoundLora,
        arena: &mut Mamba2TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        use_ffn_norm_input: bool,
        seq_len: usize,
    ) {
        let d = model.cfg.n_embd;
        match ffn {
            Mamba2FfnWeights::SwiGlu {
                n_ff,
                w_gate,
                w_up,
                w_down,
                b_gate,
                b_up,
                b_down,
            } => {
                let ff = *n_ff;
                let slot_wg = bound.slot_for(w_gate.name);
                let slot_wu = bound.slot_for(w_up.name);
                let slot_wd = bound.slot_for(w_down.name);
                let mut offset = 0usize;
                while offset < seq_len {
                    let b = (seq_len - offset).min(MAX_CPU_BATCH);
                    let single = b == 1;
                    let xin = if use_ffn_norm_input {
                        &arena.x_ffn_norm[offset * d..(offset + b) * d]
                    } else {
                        &arena.x_attn_norm[offset * d..(offset + b) * d]
                    };
                    if accel.is_some() {
                        linear_fwd_with_accel(
                            accel.as_deref_mut(),
                            pool,
                            kt,
                            w_gate,
                            bound,
                            slot_wg,
                            xin,
                            &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                            b,
                            single,
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
                            xin,
                            &mut arena.up_pre[offset * ff..(offset + b) * ff],
                            b,
                            single,
                            &mut arena.mv_scratch,
                            &mut arena.lora_scratch,
                        );
                    } else {
                        if single {
                            matvec_pair_f32(
                                pool,
                                kt,
                                w_gate.dtype,
                                w_gate.bytes,
                                ff,
                                &mut arena.gate_pre[offset * ff..(offset + 1) * ff],
                                w_up.dtype,
                                w_up.bytes,
                                ff,
                                &mut arena.up_pre[offset * ff..(offset + 1) * ff],
                                xin,
                                &mut arena.mv_scratch,
                                d,
                            );
                        } else {
                            matmul_pair_f32(
                                pool,
                                kt,
                                w_gate.dtype,
                                w_gate.bytes,
                                ff,
                                &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                                w_up.dtype,
                                w_up.bytes,
                                ff,
                                &mut arena.up_pre[offset * ff..(offset + b) * ff],
                                xin,
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
                                xin,
                                &mut arena.gate_pre[offset * ff..(offset + b) * ff],
                                b,
                                &mut arena.lora_scratch,
                            );
                            bound.apply_linear(
                                slot_wu,
                                pool,
                                kt,
                                xin,
                                &mut arena.up_pre[offset * ff..(offset + b) * ff],
                                b,
                                &mut arena.lora_scratch,
                            );
                        }
                    }
                    if let Some(bg) = b_gate {
                        for t in 0..b {
                            let row = &mut arena.gate_pre[(offset + t) * ff..(offset + t + 1) * ff];
                            for (y, &bi) in row.iter_mut().zip(bg.iter()) {
                                *y += bi;
                            }
                        }
                    }
                    if let Some(bu) = b_up {
                        for t in 0..b {
                            let row = &mut arena.up_pre[(offset + t) * ff..(offset + t + 1) * ff];
                            for (y, &bi) in row.iter_mut().zip(bu.iter()) {
                                *y += bi;
                            }
                        }
                    }
                    arena.ffn_act[offset * ff..(offset + b) * ff]
                        .copy_from_slice(&arena.gate_pre[offset * ff..(offset + b) * ff]);
                    swiglu_inplace_parallel(
                        pool,
                        &mut arena.ffn_act[offset * ff..(offset + b) * ff],
                        &arena.up_pre[offset * ff..(offset + b) * ff],
                    );
                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        w_down,
                        bound,
                        slot_wd,
                        &arena.ffn_act[offset * ff..(offset + b) * ff],
                        &mut arena.xb2_chunk[offset * d..(offset + b) * d],
                        b,
                        single,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );
                    if let Some(bd) = b_down {
                        for t in 0..b {
                            let row = &mut arena.xb2_chunk[(offset + t) * d..(offset + t + 1) * d];
                            for (y, &bi) in row.iter_mut().zip(bd.iter()) {
                                *y += bi;
                            }
                        }
                    }
                    offset += b;
                }
            }
            Mamba2FfnWeights::Relu2 {
                n_ff,
                w_up,
                w_down,
                b_up,
                b_down,
            } => {
                let ff = *n_ff;
                let slot_wu = bound.slot_for(w_up.name);
                let slot_wd = bound.slot_for(w_down.name);
                let mut offset = 0usize;
                while offset < seq_len {
                    let b = (seq_len - offset).min(MAX_CPU_BATCH);
                    let single = b == 1;
                    let xin = if use_ffn_norm_input {
                        &arena.x_ffn_norm[offset * d..(offset + b) * d]
                    } else {
                        &arena.x_attn_norm[offset * d..(offset + b) * d]
                    };
                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        w_up,
                        bound,
                        slot_wu,
                        xin,
                        &mut arena.up_pre[offset * ff..(offset + b) * ff],
                        b,
                        single,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );
                    if let Some(bu) = b_up {
                        for t in 0..b {
                            let row = &mut arena.up_pre[(offset + t) * ff..(offset + t + 1) * ff];
                            for (y, &bi) in row.iter_mut().zip(bu.iter()) {
                                *y += bi;
                            }
                        }
                    }
                    arena.ffn_act[offset * ff..(offset + b) * ff]
                        .copy_from_slice(&arena.up_pre[offset * ff..(offset + b) * ff]);
                    relu2_inplace_parallel(
                        pool,
                        &mut arena.ffn_act[offset * ff..(offset + b) * ff],
                    );
                    linear_fwd_with_accel(
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        w_down,
                        bound,
                        slot_wd,
                        &arena.ffn_act[offset * ff..(offset + b) * ff],
                        &mut arena.xb2_chunk[offset * d..(offset + b) * d],
                        b,
                        single,
                        &mut arena.mv_scratch,
                        &mut arena.lora_scratch,
                    );
                    if let Some(bd) = b_down {
                        for t in 0..b {
                            let row = &mut arena.xb2_chunk[(offset + t) * d..(offset + t + 1) * d];
                            for (y, &bi) in row.iter_mut().zip(bd.iter()) {
                                *y += bi;
                            }
                        }
                    }
                    offset += b;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_ffn(
        model: &Mamba2Model<'_>,
        ffn: &Mamba2FfnWeights<'_>,
        adapter: &LoraAdapter,
        grads: &mut LoraGradStore,
        arena: &mut Mamba2TrainArena,
        mut accel: Option<&mut (dyn LinearAccel + '_)>,
        pool: &ThreadPool,
        kt: &KernelTable,
        use_ffn_norm_input: bool,
        seq_len: usize,
        d_branch_in: &[f32],
    ) {
        let d = model.cfg.n_embd;
        let xin = if use_ffn_norm_input {
            &arena.x_ffn_norm[..seq_len * d]
        } else {
            &arena.x_attn_norm[..seq_len * d]
        };
        match ffn {
            Mamba2FfnWeights::SwiGlu {
                n_ff,
                w_gate,
                w_up,
                w_down,
                ..
            } => {
                let ff = *n_ff;
                linear_bwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    w_down,
                    adapter,
                    grads,
                    &arena.ffn_act[..seq_len * ff],
                    d_branch_in,
                    &mut arena.d_ffn_act[..seq_len * ff],
                    seq_len,
                    false,
                    &mut arena.lora_h,
                    &mut arena.lora_dh,
                    &mut arena.lora_bt,
                );
                swiglu_bwd_f32(
                    &mut arena.d_gate[..seq_len * ff],
                    &mut arena.d_up[..seq_len * ff],
                    &arena.gate_pre[..seq_len * ff],
                    &arena.up_pre[..seq_len * ff],
                    &arena.d_ffn_act[..seq_len * ff],
                );
                linear_bwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    w_gate,
                    adapter,
                    grads,
                    xin,
                    &arena.d_gate[..seq_len * ff],
                    &mut arena.d_x_norm[..seq_len * d],
                    seq_len,
                    false,
                    &mut arena.lora_h,
                    &mut arena.lora_dh,
                    &mut arena.lora_bt,
                );
                linear_bwd_with_accel(
                    accel,
                    pool,
                    kt,
                    w_up,
                    adapter,
                    grads,
                    xin,
                    &arena.d_up[..seq_len * ff],
                    &mut arena.d_x_norm[..seq_len * d],
                    seq_len,
                    true,
                    &mut arena.lora_h,
                    &mut arena.lora_dh,
                    &mut arena.lora_bt,
                );
            }
            Mamba2FfnWeights::Relu2 {
                n_ff, w_up, w_down, ..
            } => {
                let ff = *n_ff;
                linear_bwd_with_accel(
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    w_down,
                    adapter,
                    grads,
                    &arena.ffn_act[..seq_len * ff],
                    d_branch_in,
                    &mut arena.d_ffn_act[..seq_len * ff],
                    seq_len,
                    false,
                    &mut arena.lora_h,
                    &mut arena.lora_dh,
                    &mut arena.lora_bt,
                );
                relu2_bwd_f32(
                    &mut arena.d_up[..seq_len * ff],
                    &arena.up_pre[..seq_len * ff],
                    &arena.d_ffn_act[..seq_len * ff],
                );
                linear_bwd_with_accel(
                    accel,
                    pool,
                    kt,
                    w_up,
                    adapter,
                    grads,
                    xin,
                    &arena.d_up[..seq_len * ff],
                    &mut arena.d_x_norm[..seq_len * d],
                    seq_len,
                    false,
                    &mut arena.lora_h,
                    &mut arena.lora_dh,
                    &mut arena.lora_bt,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_forward(
        model: &Mamba2Model<'_>,
        bound: &BoundLora,
        arena: &mut Mamba2TrainArena,
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
        let res_scale = c.residual_scale;

        {
            let x_src = if from_ckpt_input {
                &arena.x_layer_in[..seq_len * d]
            } else {
                &arena.x_cur[..seq_len * d]
            };
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut arena.x_attn_norm[..seq_len * d],
                x_src,
                &lw.attn_norm,
                c.rms_eps,
                d,
                seq_len,
            );
        }

        match &lw.block {
            Mamba2Block::SsmOnly { ssm } => {
                Self::run_ssm_forward(
                    model,
                    ssm,
                    bound,
                    arena,
                    accel,
                    pool,
                    kt,
                    seq_len,
                    save_recompute,
                );
                if !save_recompute {
                    for i in 0..(seq_len * d) {
                        arena.x_cur[i] += res_scale * arena.xb2_chunk[i];
                    }
                }
            }
            Mamba2Block::AttnOnly { attn } => {
                Self::run_attn_forward(
                    model,
                    attn,
                    bound,
                    arena,
                    accel,
                    pool,
                    kt,
                    seq_len,
                    exact_f32_kv,
                );
                if !save_recompute {
                    for i in 0..(seq_len * d) {
                        arena.x_cur[i] += res_scale * arena.xb2_chunk[i];
                    }
                }
            }
            Mamba2Block::FfnOnly { ffn } => {
                Self::run_ffn_forward(model, ffn, bound, arena, accel, pool, kt, false, seq_len);
                if !save_recompute {
                    for i in 0..(seq_len * d) {
                        arena.x_cur[i] += res_scale * arena.xb2_chunk[i];
                    }
                }
            }
            Mamba2Block::SsmAndFfn {
                ssm, ffn_norm, ffn, ..
            } => {
                Self::run_ssm_forward(
                    model,
                    ssm,
                    bound,
                    arena,
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    seq_len,
                    save_recompute,
                );
                for i in 0..(seq_len * d) {
                    let xin = if from_ckpt_input {
                        arena.x_layer_in[i]
                    } else {
                        arena.x_cur[i]
                    };
                    arena.x_mid[i] = xin + res_scale * arena.xb2_chunk[i];
                }
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_ffn_norm[..seq_len * d],
                    &arena.x_mid[..seq_len * d],
                    ffn_norm,
                    c.rms_eps,
                    d,
                    seq_len,
                );
                Self::run_ffn_forward(model, ffn, bound, arena, accel, pool, kt, true, seq_len);
                if !save_recompute {
                    for i in 0..(seq_len * d) {
                        arena.x_cur[i] = arena.x_mid[i] + res_scale * arena.xb2_chunk[i];
                    }
                }
            }
            Mamba2Block::AttnAndFfn {
                attn,
                ffn_norm,
                ffn,
            } => {
                Self::run_attn_forward(
                    model,
                    attn,
                    bound,
                    arena,
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    seq_len,
                    exact_f32_kv,
                );
                for i in 0..(seq_len * d) {
                    let xin = if from_ckpt_input {
                        arena.x_layer_in[i]
                    } else {
                        arena.x_cur[i]
                    };
                    arena.x_mid[i] = xin + res_scale * arena.xb2_chunk[i];
                }
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_ffn_norm[..seq_len * d],
                    &arena.x_mid[..seq_len * d],
                    ffn_norm,
                    c.rms_eps,
                    d,
                    seq_len,
                );
                Self::run_ffn_forward(model, ffn, bound, arena, accel, pool, kt, true, seq_len);
                if !save_recompute {
                    for i in 0..(seq_len * d) {
                        arena.x_cur[i] = arena.x_mid[i] + res_scale * arena.xb2_chunk[i];
                    }
                }
            }
            Mamba2Block::ParallelAttnSsmAndFfn {
                attn,
                ssm,
                ffn_norm,
                ffn,
            } => {
                Self::run_attn_forward(
                    model,
                    attn,
                    bound,
                    arena,
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    seq_len,
                    exact_f32_kv,
                );
                for i in 0..(seq_len * d) {
                    let xin = if from_ckpt_input {
                        arena.x_layer_in[i]
                    } else {
                        arena.x_cur[i]
                    };
                    arena.x_mid[i] = xin + res_scale * arena.xb2_chunk[i];
                }
                Self::run_ssm_forward(
                    model,
                    ssm,
                    bound,
                    arena,
                    accel.as_deref_mut(),
                    pool,
                    kt,
                    seq_len,
                    save_recompute,
                );
                for i in 0..(seq_len * d) {
                    arena.x_mid[i] += res_scale * arena.xb2_chunk[i];
                }
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut arena.x_ffn_norm[..seq_len * d],
                    &arena.x_mid[..seq_len * d],
                    ffn_norm,
                    c.rms_eps,
                    d,
                    seq_len,
                );
                Self::run_ffn_forward(model, ffn, bound, arena, accel, pool, kt, true, seq_len);
                if !save_recompute {
                    for i in 0..(seq_len * d) {
                        arena.x_cur[i] = arena.x_mid[i] + res_scale * arena.xb2_chunk[i];
                    }
                }
            }
        }
    }

    /// Evaluates the training forward pass and writes final-token logits (`0-ULP` parity gate!).
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
        let c = &self.model.cfg;
        let d = c.n_embd;
        let n_tok = token_ids.len();

        // Notice `Mamba2Model::forward_batch` chunks by `MAX_CPU_BATCH` across ALL layers;
        // for `n_tok <= MAX_CPU_BATCH` (standard test/microbatch), layer-by-layer is 0-ULP identical!
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
            c.rms_eps,
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
        if (c.inv_logit_scale - 1.0).abs() > 1e-12 {
            let s = c.inv_logit_scale;
            for v in out_last_logits.iter_mut() {
                *v *= s;
            }
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

        self.sync_bound_lora(adapter);
        let kt = active_kernels();
        let c = &self.model.cfg;
        let d = c.n_embd;
        let vocab = c.vocab_size;
        let res_scale = c.residual_scale;

        // 1. Forward pass with per-layer residual checkpoints
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

        // 2. Masked Chunked Cross-Entropy (`CE_CHUNK_TOKENS = 16`)
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
                let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f64 = row.iter().map(|&lv| ((lv - max_l) as f64).exp()).sum();
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

        // 3. Reverse layer backward (`l = (0..n_layers).rev()`)
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
            let mut d_branch = std::mem::take(&mut self.arena.d_branch);
            let mut d_x_norm = std::mem::take(&mut self.arena.d_x_norm);

            match &lw.block {
                Mamba2Block::SsmOnly { ssm } => {
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    Self::backward_ssm(
                        self.model,
                        ssm,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        seq_len,
                        &d_branch[..seq_len * d],
                        &mut d_x_norm[..seq_len * d],
                        false,
                    );
                }
                Mamba2Block::AttnOnly { attn } => {
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    Self::backward_attn(
                        self.model,
                        attn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        seq_len,
                        self.exact_f32_kv_gradcheck,
                        &d_branch[..seq_len * d],
                        &mut d_x_norm[..seq_len * d],
                        false,
                    );
                }
                Mamba2Block::FfnOnly { ffn } => {
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    self.arena.d_x_norm = d_x_norm;
                    Self::backward_ffn(
                        self.model,
                        ffn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        false,
                        seq_len,
                        &d_branch[..seq_len * d],
                    );
                    d_x_norm = std::mem::take(&mut self.arena.d_x_norm);
                }
                Mamba2Block::SsmAndFfn {
                    ssm, ffn_norm, ffn, ..
                } => {
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    self.arena.d_x_norm = d_x_norm;
                    Self::backward_ffn(
                        self.model,
                        ffn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        true,
                        seq_len,
                        &d_branch[..seq_len * d],
                    );
                    d_x_norm = std::mem::take(&mut self.arena.d_x_norm);
                    for t in 0..seq_len {
                        rms_norm_bwd_f32(
                            &mut self.arena.tmp_embd,
                            None,
                            &self.arena.x_mid[t * d..(t + 1) * d],
                            ffn_norm,
                            &d_x_norm[t * d..(t + 1) * d],
                            c.rms_eps,
                        );
                        (kt.axpy_f32)(
                            &mut self.arena.dx[t * d..(t + 1) * d],
                            1.0,
                            &self.arena.tmp_embd,
                        );
                    }
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    Self::backward_ssm(
                        self.model,
                        ssm,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        seq_len,
                        &d_branch[..seq_len * d],
                        &mut d_x_norm[..seq_len * d],
                        false,
                    );
                }
                Mamba2Block::AttnAndFfn {
                    attn,
                    ffn_norm,
                    ffn,
                } => {
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    self.arena.d_x_norm = d_x_norm;
                    Self::backward_ffn(
                        self.model,
                        ffn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        true,
                        seq_len,
                        &d_branch[..seq_len * d],
                    );
                    d_x_norm = std::mem::take(&mut self.arena.d_x_norm);
                    for t in 0..seq_len {
                        rms_norm_bwd_f32(
                            &mut self.arena.tmp_embd,
                            None,
                            &self.arena.x_mid[t * d..(t + 1) * d],
                            ffn_norm,
                            &d_x_norm[t * d..(t + 1) * d],
                            c.rms_eps,
                        );
                        (kt.axpy_f32)(
                            &mut self.arena.dx[t * d..(t + 1) * d],
                            1.0,
                            &self.arena.tmp_embd,
                        );
                    }
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    Self::backward_attn(
                        self.model,
                        attn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        seq_len,
                        self.exact_f32_kv_gradcheck,
                        &d_branch[..seq_len * d],
                        &mut d_x_norm[..seq_len * d],
                        false,
                    );
                }
                Mamba2Block::ParallelAttnSsmAndFfn {
                    attn,
                    ssm,
                    ffn_norm,
                    ffn,
                } => {
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    self.arena.d_x_norm = d_x_norm;
                    Self::backward_ffn(
                        self.model,
                        ffn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        true,
                        seq_len,
                        &d_branch[..seq_len * d],
                    );
                    d_x_norm = std::mem::take(&mut self.arena.d_x_norm);
                    for t in 0..seq_len {
                        rms_norm_bwd_f32(
                            &mut self.arena.tmp_embd,
                            None,
                            &self.arena.x_mid[t * d..(t + 1) * d],
                            ffn_norm,
                            &d_x_norm[t * d..(t + 1) * d],
                            c.rms_eps,
                        );
                        (kt.axpy_f32)(
                            &mut self.arena.dx[t * d..(t + 1) * d],
                            1.0,
                            &self.arena.tmp_embd,
                        );
                    }
                    for (dst, &dxv) in d_branch[..seq_len * d]
                        .iter_mut()
                        .zip(&self.arena.dx[..seq_len * d])
                    {
                        *dst = res_scale * dxv;
                    }
                    Self::backward_ssm(
                        self.model,
                        ssm,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        seq_len,
                        &d_branch[..seq_len * d],
                        &mut d_x_norm[..seq_len * d],
                        false,
                    );
                    let _ = &self.arena.d_x_norm_par;
                    Self::backward_attn(
                        self.model,
                        attn,
                        adapter,
                        grads,
                        &mut self.arena,
                        accel.as_deref_mut(),
                        pool,
                        kt,
                        seq_len,
                        self.exact_f32_kv_gradcheck,
                        &d_branch[..seq_len * d],
                        &mut d_x_norm[..seq_len * d],
                        true,
                    );
                }
            }

            for t in 0..seq_len {
                rms_norm_bwd_f32(
                    &mut self.arena.tmp_embd,
                    None,
                    &self.arena.x_layer_in[t * d..(t + 1) * d],
                    &lw.attn_norm,
                    &d_x_norm[t * d..(t + 1) * d],
                    c.rms_eps,
                );
                (kt.axpy_f32)(
                    &mut self.arena.dx[t * d..(t + 1) * d],
                    1.0,
                    &self.arena.tmp_embd,
                );
            }

            self.arena.d_branch = d_branch;
            self.arena.d_x_norm = d_x_norm;
        }

        // 4. Optional `token_embd.weight` LoRA backward
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

        ((loss_sum / (n_valid as f64)) as f32, n_valid)
    }

    /// Exact `f64` masked cross-entropy loss for finite-difference Jacobian verification.
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
fn gqa_f32_exact(
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
) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let group = n_heads / n_kv_heads;
    for t in 0..n_tok {
        let win_len = start_pos + t + 1;
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let oh = &mut out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let mut scores = vec![0.0f64; win_len];
            let mut max_s = f64::NEG_INFINITY;
            for (step, sc) in scores.iter_mut().enumerate() {
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
                for (step, &e) in scores.iter().enumerate() {
                    acc += (e * inv) * (v_all[step * kv_dim + kv_h * head_dim + d] as f64);
                }
                oh[d] = acc as f32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gqa_f32_bwd_exact(
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
        let win_len = t + 1;
        let p = &mut probs[..win_len];
        let d_prob = &mut dp[..win_len];
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let doh = &d_out[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            let mut max_s = f64::NEG_INFINITY;
            for (step, ps) in p.iter_mut().enumerate() {
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
            for step in 0..win_len {
                let pw = p[step];
                let v_off = step * kv_dim + kv_h * head_dim;
                let vh = &v_all[v_off..v_off + head_dim];
                let dvh = &mut dv[v_off..v_off + head_dim];
                let mut dot_dov = 0.0f64;
                for d in 0..head_dim {
                    dvh[d] += (doh[d] as f64 * pw) as f32;
                    dot_dov += (doh[d] as f64) * (vh[d] as f64);
                }
                d_prob[step] = dot_dov;
                sum_p_dp += pw * dot_dov;
            }
            let dqh = &mut dq[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
            for step in 0..win_len {
                let ds = (p[step] * (d_prob[step] - sum_p_dp) * scale) as f32;
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
