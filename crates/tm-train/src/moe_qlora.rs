//! Shared Sparse Mixture-of-Experts (`SparseMoeFfn`) QLoRA forward & exact analytical backward
//! engine (`ADR-018`) used by both [`crate::LlamaTrainer`] (`olmoe`, `granitemoe`, `qwen2moe`,
//! `qwen3moe`, `bailingmoe2`) and [`crate::Mamba2Trainer`] (`granitehybrid` / `granitemoehybrid`).
//!
//! Guarantees:
//! 1. **0-ULP Forward Parity Gate**: Uses the exact same `tm-cpu` `matvec`/`matmul`, `compute_moe_routing_row`,
//!    and `0..n_expert_used` reduction order as `tm_models::eval_sparse_moe_ffn`.
//! 2. **Exact Analytical Backward**: Backpropagates upstream `d_out [seq_len, n_embd]` through:
//!    - Active routed experts `slot in 0..k` (`down_exps^T -> SwiGLU/SiLU bwd -> gate_exps^T & up_exps^T`),
//!    - Router `ffn_gate_inp` (`weights_scale`, `norm_weights` quotient rule with `6.103515625e-5` clamp,
//!      and `Softmax` / `Sigmoid` / `SoftmaxWeight` / `SqrtSoftplus` Jacobians + router LoRA `dA`/`dB`),
//!    - Optional shared expert `ffn_{gate,up,down}_shexp` + optional `qwen2moe` 1D sigmoid gate
//!      `ffn_gate_inp_shexp.weight` (including LoRA `dA`/`dB` on every projection).
//! 3. **Zero Heap Allocations on Hot Path (`AGENTS.md` Rule #4)** & `#![forbid(unsafe_code)]`.

use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, ThreadPool, matmul_pair_f32,
    matmul_q_transposed_f32, matvec_f32, matvec_pair_f32, softplus_f32, swiglu_bwd_f32,
    swiglu_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch};
use tm_models::{ExpertGatingFunc, MOE_NORM_CLAMP_F16_MIN, SparseMoeFfn, compute_moe_routing_row};

use crate::accel::{LinearAccel, linear_bwd_with_accel, linear_fwd_with_accel};
use crate::optim::LoraGradStore;

/// Pre-allocated single-layer MoE recomputation & backward workspace (`AGENTS.md` Rule #4).
pub(crate) struct MoeTrainScratch {
    // Router & per-token routing scratch
    pub router_logits: Vec<f32>,
    pub d_router_logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub sel_probs: Vec<f32>,
    pub top_ids: Vec<usize>,
    pub top_weights: Vec<f32>,
    pub probs_f64: Vec<f64>,
    pub dp_f64: Vec<f64>,
    pub v_f64: Vec<f64>,
    pub dv_f64: Vec<f64>,
    pub d_top_w: Vec<f64>,

    // Active routed expert activations (`[max_seq * max_k, ...]`) & temporary single-expert buffers
    pub exp_gate_pre: Vec<f32>,
    pub exp_up_pre: Vec<f32>,
    pub exp_act: Vec<f32>,
    pub exp_out: Vec<f32>,
    pub tmp_exp_gate: Vec<f32>,
    pub tmp_exp_up: Vec<f32>,
    pub tmp_exp_out: Vec<f32>,
    pub dy_exp: Vec<f32>,
    pub d_exp_act: Vec<f32>,
    pub d_exp_gate: Vec<f32>,
    pub d_exp_up: Vec<f32>,

    // Shared expert recompute & backward buffers (`[max_seq, max_ff_shexp]`)
    pub shexp_gate_pre: Vec<f32>,
    pub shexp_up_pre: Vec<f32>,
    pub shexp_act: Vec<f32>,
    pub shexp_down_out: Vec<f32>,
    pub shexp_gate_logit: Vec<f32>,
    pub d_shexp_down: Vec<f32>,
    pub d_shexp_act: Vec<f32>,
    pub d_shexp_gate: Vec<f32>,
    pub d_shexp_up: Vec<f32>,
    pub d_shexp_gate_logit: Vec<f32>,
}

impl MoeTrainScratch {
    pub fn empty() -> Self {
        Self {
            router_logits: Vec::new(),
            d_router_logits: Vec::new(),
            probs: Vec::new(),
            sel_probs: Vec::new(),
            top_ids: Vec::new(),
            top_weights: Vec::new(),
            probs_f64: Vec::new(),
            dp_f64: Vec::new(),
            v_f64: Vec::new(),
            dv_f64: Vec::new(),
            d_top_w: Vec::new(),
            exp_gate_pre: Vec::new(),
            exp_up_pre: Vec::new(),
            exp_act: Vec::new(),
            exp_out: Vec::new(),
            tmp_exp_gate: Vec::new(),
            tmp_exp_up: Vec::new(),
            tmp_exp_out: Vec::new(),
            dy_exp: Vec::new(),
            d_exp_act: Vec::new(),
            d_exp_gate: Vec::new(),
            d_exp_up: Vec::new(),
            shexp_gate_pre: Vec::new(),
            shexp_up_pre: Vec::new(),
            shexp_act: Vec::new(),
            shexp_down_out: Vec::new(),
            shexp_gate_logit: Vec::new(),
            d_shexp_down: Vec::new(),
            d_shexp_act: Vec::new(),
            d_shexp_gate: Vec::new(),
            d_shexp_up: Vec::new(),
            d_shexp_gate_logit: Vec::new(),
        }
    }

    pub fn new(
        max_seq: usize,
        n_embd: usize,
        max_n_exp: usize,
        max_k: usize,
        max_ff_exp: usize,
        max_ff_shexp: usize,
    ) -> Self {
        if max_n_exp == 0 || max_k == 0 || max_ff_exp == 0 {
            return Self::empty();
        }
        let sk = max_seq * max_k;
        Self {
            router_logits: vec![0.0; max_seq * max_n_exp],
            d_router_logits: vec![0.0; max_seq * max_n_exp],
            probs: vec![0.0; max_n_exp],
            sel_probs: vec![0.0; max_n_exp],
            top_ids: vec![0; sk],
            top_weights: vec![0.0; sk],
            probs_f64: vec![0.0; max_n_exp],
            dp_f64: vec![0.0; max_n_exp],
            v_f64: vec![0.0; max_k],
            dv_f64: vec![0.0; max_k],
            d_top_w: vec![0.0; max_k],
            exp_gate_pre: vec![0.0; sk * max_ff_exp],
            exp_up_pre: vec![0.0; sk * max_ff_exp],
            exp_act: vec![0.0; sk * max_ff_exp],
            exp_out: vec![0.0; sk * n_embd],
            tmp_exp_gate: vec![0.0; max_ff_exp],
            tmp_exp_up: vec![0.0; max_ff_exp],
            tmp_exp_out: vec![0.0; n_embd],
            dy_exp: vec![0.0; n_embd],
            d_exp_act: vec![0.0; max_ff_exp],
            d_exp_gate: vec![0.0; max_ff_exp],
            d_exp_up: vec![0.0; max_ff_exp],
            shexp_gate_pre: vec![0.0; max_seq * max_ff_shexp],
            shexp_up_pre: vec![0.0; max_seq * max_ff_shexp],
            shexp_act: vec![0.0; max_seq * max_ff_shexp],
            shexp_down_out: if max_ff_shexp > 0 {
                vec![0.0; max_seq * n_embd]
            } else {
                Vec::new()
            },
            shexp_gate_logit: if max_ff_shexp > 0 {
                vec![0.0; max_seq]
            } else {
                Vec::new()
            },
            d_shexp_down: if max_ff_shexp > 0 {
                vec![0.0; max_seq * n_embd]
            } else {
                Vec::new()
            },
            d_shexp_act: vec![0.0; max_seq * max_ff_shexp],
            d_shexp_gate: vec![0.0; max_seq * max_ff_shexp],
            d_shexp_up: vec![0.0; max_seq * max_ff_shexp],
            d_shexp_gate_logit: if max_ff_shexp > 0 {
                vec![0.0; max_seq]
            } else {
                Vec::new()
            },
        }
    }
}

/// Evaluates `SparseMoeFfn` for a chunk `[offset .. offset + b]` (`1 <= b <= MAX_CPU_BATCH`),
/// writing `[b, n_embd]` into `out_chunk` (`0-ULP` bit-identical to `tm_models::eval_sparse_moe_ffn`).
///
/// When `save_recompute == true`, stores intermediate router, expert, and shared-expert activations
/// at global token indices `offset .. offset + b` inside `scratch`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_moe_chunk(
    moe: &SparseMoeFfn<'_>,
    bound: &BoundLora,
    mut accel: Option<&mut (dyn LinearAccel + '_)>,
    pool: &ThreadPool,
    kt: &KernelTable,
    x_norm_chunk: &[f32],
    out_chunk: &mut [f32],
    n_embd: usize,
    offset: usize,
    b: usize,
    save_recompute: bool,
    mv_scratch: &mut MatvecScratch,
    lora_scratch: &mut LoraScratch,
    scratch: &mut MoeTrainScratch,
) {
    debug_assert!((1..=MAX_CPU_BATCH).contains(&b));
    let n_exp = moe.cfg.n_expert;
    let k = moe.cfg.n_expert_used;
    let n_ff_exp = moe.cfg.n_ff_exp;
    let single_chunk = b == 1;

    // 1. Router projection (`[b, n_embd] -> [b, n_expert]`)
    let slot_gi = bound.slot_for(moe.gate_inp.name);
    let rl_dst = &mut scratch.router_logits[offset * n_exp..(offset + b) * n_exp];
    linear_fwd_with_accel(
        accel.as_deref_mut(),
        pool,
        kt,
        &moe.gate_inp,
        bound,
        slot_gi,
        x_norm_chunk,
        rl_dst,
        b,
        single_chunk,
        mv_scratch,
        lora_scratch,
    );
    if let Some(ref gb) = moe.gate_inp_bias {
        for t in 0..b {
            let row = &mut rl_dst[t * n_exp..(t + 1) * n_exp];
            for (y, &bi) in row.iter_mut().zip(gb.iter()) {
                *y += bi;
            }
        }
    }

    // 2. Top-K routing + deterministic `0..k` expert evaluation per token
    out_chunk[..b * n_embd].fill(0.0);
    for t in 0..b {
        let g_t = offset + t;
        let l_row = &scratch.router_logits[g_t * n_exp..(g_t + 1) * n_exp];
        let ids_t = &mut scratch.top_ids[g_t * k..(g_t + 1) * k];
        let w_t = &mut scratch.top_weights[g_t * k..(g_t + 1) * k];
        compute_moe_routing_row(
            &moe.cfg,
            l_row,
            moe.exp_probs_b.as_deref(),
            &mut scratch.probs[..n_exp],
            &mut scratch.sel_probs[..n_exp],
            ids_t,
            w_t,
        );

        let xt = &x_norm_chunk[t * n_embd..(t + 1) * n_embd];
        let out_t = &mut out_chunk[t * n_embd..(t + 1) * n_embd];
        for slot in 0..k {
            let exp_id = ids_t[slot];
            let weight = w_t[slot];
            let (gate_opt, (up_dt, up_bytes), (down_dt, down_bytes)) =
                moe.experts.expert_slices(exp_id);
            let slot_idx = g_t * k + slot;
            if let Some((gate_dt, gate_bytes)) = gate_opt {
                matvec_pair_f32(
                    pool,
                    kt,
                    gate_dt,
                    gate_bytes,
                    n_ff_exp,
                    &mut scratch.tmp_exp_gate[..n_ff_exp],
                    up_dt,
                    up_bytes,
                    n_ff_exp,
                    &mut scratch.tmp_exp_up[..n_ff_exp],
                    xt,
                    mv_scratch,
                    n_embd,
                );
                if save_recompute {
                    let f_off = slot_idx * n_ff_exp;
                    scratch.exp_gate_pre[f_off..f_off + n_ff_exp]
                        .copy_from_slice(&scratch.tmp_exp_gate[..n_ff_exp]);
                    scratch.exp_up_pre[f_off..f_off + n_ff_exp]
                        .copy_from_slice(&scratch.tmp_exp_up[..n_ff_exp]);
                }
                swiglu_inplace_parallel(
                    pool,
                    &mut scratch.tmp_exp_gate[..n_ff_exp],
                    &scratch.tmp_exp_up[..n_ff_exp],
                );
            } else {
                matvec_f32(
                    pool,
                    kt,
                    up_dt,
                    up_bytes,
                    xt,
                    mv_scratch,
                    &mut scratch.tmp_exp_gate[..n_ff_exp],
                    n_ff_exp,
                    n_embd,
                );
                if save_recompute {
                    let f_off = slot_idx * n_ff_exp;
                    scratch.exp_gate_pre[f_off..f_off + n_ff_exp]
                        .copy_from_slice(&scratch.tmp_exp_gate[..n_ff_exp]);
                }
                for v in &mut scratch.tmp_exp_gate[..n_ff_exp] {
                    let x = *v;
                    *v = x / (1.0 + (-x).exp());
                }
            }
            if save_recompute {
                let f_off = slot_idx * n_ff_exp;
                scratch.exp_act[f_off..f_off + n_ff_exp]
                    .copy_from_slice(&scratch.tmp_exp_gate[..n_ff_exp]);
            }
            matvec_f32(
                pool,
                kt,
                down_dt,
                down_bytes,
                &scratch.tmp_exp_gate[..n_ff_exp],
                mv_scratch,
                &mut scratch.tmp_exp_out[..n_embd],
                n_embd,
                n_ff_exp,
            );
            if save_recompute {
                let o_off = slot_idx * n_embd;
                scratch.exp_out[o_off..o_off + n_embd]
                    .copy_from_slice(&scratch.tmp_exp_out[..n_embd]);
            }
            (kt.axpy_f32)(out_t, weight, &scratch.tmp_exp_out[..n_embd]);
        }
    }

    // 3. Optional shared expert (`ffn_{gate,up,down}_shexp` + optional 1D `ffn_gate_inp_shexp`)
    if let Some(ref sh) = moe.shared_expert {
        let ff_sh = sh.n_ff_shexp;
        let slot_sh_g = bound.slot_for(sh.w_gate.name);
        let slot_sh_u = bound.slot_for(sh.w_up.name);
        let slot_sh_d = bound.slot_for(sh.w_down.name);

        let g_slice = &mut scratch.shexp_gate_pre[offset * ff_sh..(offset + b) * ff_sh];
        let u_slice = &mut scratch.shexp_up_pre[offset * ff_sh..(offset + b) * ff_sh];
        if accel.is_some() {
            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &sh.w_gate,
                bound,
                slot_sh_g,
                x_norm_chunk,
                g_slice,
                b,
                single_chunk,
                mv_scratch,
                lora_scratch,
            );
            linear_fwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &sh.w_up,
                bound,
                slot_sh_u,
                x_norm_chunk,
                u_slice,
                b,
                single_chunk,
                mv_scratch,
                lora_scratch,
            );
        } else {
            if single_chunk {
                matvec_pair_f32(
                    pool,
                    kt,
                    sh.w_gate.dtype,
                    sh.w_gate.bytes,
                    ff_sh,
                    g_slice,
                    sh.w_up.dtype,
                    sh.w_up.bytes,
                    ff_sh,
                    u_slice,
                    x_norm_chunk,
                    mv_scratch,
                    n_embd,
                );
            } else {
                matmul_pair_f32(
                    pool,
                    kt,
                    sh.w_gate.dtype,
                    sh.w_gate.bytes,
                    ff_sh,
                    g_slice,
                    sh.w_up.dtype,
                    sh.w_up.bytes,
                    ff_sh,
                    u_slice,
                    x_norm_chunk,
                    mv_scratch,
                    n_embd,
                    b,
                );
            }
            if !bound.is_empty() {
                bound.apply_linear(slot_sh_g, pool, kt, x_norm_chunk, g_slice, b, lora_scratch);
                bound.apply_linear(slot_sh_u, pool, kt, x_norm_chunk, u_slice, b, lora_scratch);
            }
        }

        let act_slice = &mut scratch.shexp_act[offset * ff_sh..(offset + b) * ff_sh];
        act_slice.copy_from_slice(g_slice);
        swiglu_inplace_parallel(pool, act_slice, u_slice);

        let down_slice = &mut scratch.shexp_down_out[offset * n_embd..(offset + b) * n_embd];
        linear_fwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            &sh.w_down,
            bound,
            slot_sh_d,
            act_slice,
            down_slice,
            b,
            single_chunk,
            mv_scratch,
            lora_scratch,
        );

        if let Some(ref gi) = sh.w_gate_inp_shexp {
            let slot_gi_sh = bound.slot_for(gi.name);
            let gl_slice = &mut scratch.shexp_gate_logit[offset..offset + b];
            linear_fwd_with_accel(
                accel,
                pool,
                kt,
                gi,
                bound,
                slot_gi_sh,
                x_norm_chunk,
                gl_slice,
                b,
                single_chunk,
                mv_scratch,
                lora_scratch,
            );
            for t in 0..b {
                let g = 1.0 / (1.0 + (-gl_slice[t]).exp());
                (kt.axpy_f32)(
                    &mut out_chunk[t * n_embd..(t + 1) * n_embd],
                    g,
                    &down_slice[t * n_embd..(t + 1) * n_embd],
                );
            }
        } else {
            (kt.axpy_f32)(out_chunk, 1.0, down_slice);
        }
    }
}

/// Exact analytical backward pass for `SparseMoeFfn` over `seq_len` tokens.
///
/// Writes total gradient w.r.t. `x_norm [seq_len, n_embd]` into `d_x_norm` (overwriting) and
/// accumulates all active LoRA parameter gradients into `grads`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_moe_ffn(
    moe: &SparseMoeFfn<'_>,
    adapter: &LoraAdapter,
    grads: &mut LoraGradStore,
    mut accel: Option<&mut (dyn LinearAccel + '_)>,
    pool: &ThreadPool,
    kt: &KernelTable,
    x_norm: &[f32],
    d_out: &[f32],
    d_x_norm: &mut [f32],
    n_embd: usize,
    seq_len: usize,
    scratch: &mut MoeTrainScratch,
    lora_h: &mut [f32],
    lora_dh: &mut [f32],
    lora_bt: &mut [f32],
) {
    let n_exp = moe.cfg.n_expert;
    let k = moe.cfg.n_expert_used;
    let n_ff_exp = moe.cfg.n_ff_exp;
    d_x_norm[..seq_len * n_embd].fill(0.0);

    let scale_factor =
        if moe.cfg.weights_scale != 0.0 && (moe.cfg.weights_scale - 1.0).abs() > 1e-12 {
            moe.cfg.weights_scale as f64
        } else {
            1.0
        };

    // 1. Active routed experts `slot in 0..k` + per-token router Jacobian `d_router_logits`
    for t in 0..seq_len {
        let d_out_t = &d_out[t * n_embd..(t + 1) * n_embd];
        let dx_t = &mut d_x_norm[t * n_embd..(t + 1) * n_embd];

        for slot in 0..k {
            let slot_idx = t * k + slot;
            let exp_id = scratch.top_ids[slot_idx];
            let weight = scratch.top_weights[slot_idx];
            let o_off = slot_idx * n_embd;
            let f_off = slot_idx * n_ff_exp;

            let mut dw = 0.0f64;
            for (i, &dy_i) in d_out_t.iter().enumerate() {
                dw += (dy_i as f64) * (scratch.exp_out[o_off + i] as f64);
                scratch.dy_exp[i] = weight * dy_i;
            }
            scratch.d_top_w[slot] = dw;

            let (gate_opt, (up_dt, up_bytes), (down_dt, down_bytes)) =
                moe.experts.expert_slices(exp_id);

            matmul_q_transposed_f32(
                pool,
                kt,
                down_dt,
                down_bytes,
                &scratch.dy_exp[..n_embd],
                &mut scratch.d_exp_act[..n_ff_exp],
                n_embd,
                n_ff_exp,
                1,
                false,
            );

            if let Some((gate_dt, gate_bytes)) = gate_opt {
                swiglu_bwd_f32(
                    &mut scratch.d_exp_gate[..n_ff_exp],
                    &mut scratch.d_exp_up[..n_ff_exp],
                    &scratch.exp_gate_pre[f_off..f_off + n_ff_exp],
                    &scratch.exp_up_pre[f_off..f_off + n_ff_exp],
                    &scratch.d_exp_act[..n_ff_exp],
                );
                matmul_q_transposed_f32(
                    pool,
                    kt,
                    gate_dt,
                    gate_bytes,
                    &scratch.d_exp_gate[..n_ff_exp],
                    dx_t,
                    n_ff_exp,
                    n_embd,
                    1,
                    true,
                );
                matmul_q_transposed_f32(
                    pool,
                    kt,
                    up_dt,
                    up_bytes,
                    &scratch.d_exp_up[..n_ff_exp],
                    dx_t,
                    n_ff_exp,
                    n_embd,
                    1,
                    true,
                );
            } else {
                for j in 0..n_ff_exp {
                    let u = scratch.exp_gate_pre[f_off + j] as f64;
                    let sig = 1.0 / (1.0 + (-u).exp());
                    scratch.d_exp_up[j] =
                        ((scratch.d_exp_act[j] as f64) * sig * (1.0 + u * (1.0 - sig))) as f32;
                }
                matmul_q_transposed_f32(
                    pool,
                    kt,
                    up_dt,
                    up_bytes,
                    &scratch.d_exp_up[..n_ff_exp],
                    dx_t,
                    n_ff_exp,
                    n_embd,
                    1,
                    true,
                );
            }
        }

        // Exact analytical router Jacobian `d_top_w[0..k] -> d_router_logits[t, 0..n_exp]`
        let logits_t = &scratch.router_logits[t * n_exp..(t + 1) * n_exp];
        let p_f64 = &mut scratch.probs_f64[..n_exp];
        match moe.cfg.gating_func {
            ExpertGatingFunc::Softmax => {
                let max_l = logits_t.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let mut sum = 0.0f64;
                for (p, &l) in p_f64.iter_mut().zip(logits_t.iter()) {
                    let e = ((l as f64) - max_l).exp();
                    *p = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for p in p_f64.iter_mut() {
                    *p *= inv;
                }
            }
            ExpertGatingFunc::Sigmoid => {
                for (p, &l) in p_f64.iter_mut().zip(logits_t.iter()) {
                    *p = 1.0 / (1.0 + (-(l as f64)).exp());
                }
            }
            ExpertGatingFunc::SoftmaxWeight => {
                for (p, &l) in p_f64.iter_mut().zip(logits_t.iter()) {
                    *p = l as f64;
                }
            }
            ExpertGatingFunc::SqrtSoftplus => {
                for (p, &l) in p_f64.iter_mut().zip(logits_t.iter()) {
                    *p = (softplus_f32(l) as f64).sqrt();
                }
            }
        }

        let ids_t = &scratch.top_ids[t * k..(t + 1) * k];
        for (slot, &eid) in ids_t.iter().enumerate() {
            scratch.v_f64[slot] = p_f64[eid];
        }
        if moe.cfg.gating_func == ExpertGatingFunc::SoftmaxWeight {
            let max_u = scratch.v_f64[..k]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for v in &mut scratch.v_f64[..k] {
                *v = (*v - max_u).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in &mut scratch.v_f64[..k] {
                *v *= inv;
            }
        }

        // Step 5 & 4 reverse (`weights_scale` + `norm_weights`)
        if moe.cfg.norm_weights {
            let sum_v: f64 = scratch.v_f64[..k].iter().sum();
            let eps = moe.cfg.norm_weights_eps as f64;
            let clamp_min = MOE_NORM_CLAMP_F16_MIN as f64;
            if eps > 0.0 || sum_v > clamp_min {
                let denom = if eps > 0.0 { sum_v + eps } else { sum_v };
                let mut dot_dn_v = 0.0f64;
                for slot in 0..k {
                    dot_dn_v += (scratch.d_top_w[slot] * scale_factor) * scratch.v_f64[slot];
                }
                let inv_d2 = 1.0 / (denom * denom);
                for slot in 0..k {
                    let dn = scratch.d_top_w[slot] * scale_factor;
                    scratch.dv_f64[slot] = (dn * denom - dot_dn_v) * inv_d2;
                }
            } else {
                let inv_c = 1.0 / clamp_min;
                for slot in 0..k {
                    scratch.dv_f64[slot] = (scratch.d_top_w[slot] * scale_factor) * inv_c;
                }
            }
        } else {
            for slot in 0..k {
                scratch.dv_f64[slot] = scratch.d_top_w[slot] * scale_factor;
            }
        }

        // Step 3 & 2 reverse (`SoftmaxWeight` -> `dp_f64`)
        scratch.dp_f64[..n_exp].fill(0.0);
        if moe.cfg.gating_func == ExpertGatingFunc::SoftmaxWeight {
            let mut dot_dv_v = 0.0f64;
            for slot in 0..k {
                dot_dv_v += scratch.dv_f64[slot] * scratch.v_f64[slot];
            }
            for (slot, &eid) in ids_t.iter().enumerate() {
                scratch.dp_f64[eid] = scratch.v_f64[slot] * (scratch.dv_f64[slot] - dot_dv_v);
            }
        } else {
            for (slot, &eid) in ids_t.iter().enumerate() {
                scratch.dp_f64[eid] = scratch.dv_f64[slot];
            }
        }

        // Step 1 reverse (`dp_f64 -> d_router_logits`)
        let dl_row = &mut scratch.d_router_logits[t * n_exp..(t + 1) * n_exp];
        match moe.cfg.gating_func {
            ExpertGatingFunc::Softmax => {
                let mut dot_dp_p = 0.0f64;
                for (&dp_e, &p_e) in scratch.dp_f64[..n_exp].iter().zip(p_f64.iter()) {
                    dot_dp_p += dp_e * p_e;
                }
                for (e, dst) in dl_row.iter_mut().enumerate() {
                    *dst = (p_f64[e] * (scratch.dp_f64[e] - dot_dp_p)) as f32;
                }
            }
            ExpertGatingFunc::Sigmoid => {
                for (e, dst) in dl_row.iter_mut().enumerate() {
                    let p = p_f64[e];
                    *dst = (scratch.dp_f64[e] * p * (1.0 - p)) as f32;
                }
            }
            ExpertGatingFunc::SoftmaxWeight => {
                for (e, dst) in dl_row.iter_mut().enumerate() {
                    *dst = scratch.dp_f64[e] as f32;
                }
            }
            ExpertGatingFunc::SqrtSoftplus => {
                for (e, dst) in dl_row.iter_mut().enumerate() {
                    let l = logits_t[e] as f64;
                    let sig = 1.0 / (1.0 + (-l).exp());
                    let denom = 2.0 * p_f64[e].max(1e-12);
                    *dst = (scratch.dp_f64[e] * sig / denom) as f32;
                }
            }
        }
    }

    linear_bwd_with_accel(
        accel.as_deref_mut(),
        pool,
        kt,
        &moe.gate_inp,
        adapter,
        grads,
        x_norm,
        &scratch.d_router_logits[..seq_len * n_exp],
        d_x_norm,
        seq_len,
        true,
        lora_h,
        lora_dh,
        lora_bt,
    );

    // 2. Optional shared expert backward (`ffn_gate_inp_shexp` + `w_down` -> `SwiGLU` -> `w_gate`/`w_up`)
    if let Some(ref sh) = moe.shared_expert {
        let ff_sh = sh.n_ff_shexp;
        if let Some(ref gi) = sh.w_gate_inp_shexp {
            for t in 0..seq_len {
                let gl = scratch.shexp_gate_logit[t] as f64;
                let g = 1.0 / (1.0 + (-gl).exp());
                let mut dg = 0.0f64;
                for i in 0..n_embd {
                    let dy_i = d_out[t * n_embd + i];
                    dg += (dy_i as f64) * (scratch.shexp_down_out[t * n_embd + i] as f64);
                    scratch.d_shexp_down[t * n_embd + i] = ((dy_i as f64) * g) as f32;
                }
                scratch.d_shexp_gate_logit[t] = (dg * g * (1.0 - g)) as f32;
            }
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                gi,
                adapter,
                grads,
                x_norm,
                &scratch.d_shexp_gate_logit[..seq_len],
                d_x_norm,
                seq_len,
                true,
                lora_h,
                lora_dh,
                lora_bt,
            );
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &sh.w_down,
                adapter,
                grads,
                &scratch.shexp_act[..seq_len * ff_sh],
                &scratch.d_shexp_down[..seq_len * n_embd],
                &mut scratch.d_shexp_act[..seq_len * ff_sh],
                seq_len,
                false,
                lora_h,
                lora_dh,
                lora_bt,
            );
        } else {
            linear_bwd_with_accel(
                accel.as_deref_mut(),
                pool,
                kt,
                &sh.w_down,
                adapter,
                grads,
                &scratch.shexp_act[..seq_len * ff_sh],
                d_out,
                &mut scratch.d_shexp_act[..seq_len * ff_sh],
                seq_len,
                false,
                lora_h,
                lora_dh,
                lora_bt,
            );
        }

        swiglu_bwd_f32(
            &mut scratch.d_shexp_gate[..seq_len * ff_sh],
            &mut scratch.d_shexp_up[..seq_len * ff_sh],
            &scratch.shexp_gate_pre[..seq_len * ff_sh],
            &scratch.shexp_up_pre[..seq_len * ff_sh],
            &scratch.d_shexp_act[..seq_len * ff_sh],
        );
        linear_bwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            &sh.w_gate,
            adapter,
            grads,
            x_norm,
            &scratch.d_shexp_gate[..seq_len * ff_sh],
            d_x_norm,
            seq_len,
            true,
            lora_h,
            lora_dh,
            lora_bt,
        );
        linear_bwd_with_accel(
            accel,
            pool,
            kt,
            &sh.w_up,
            adapter,
            grads,
            x_norm,
            &scratch.d_shexp_up[..seq_len * ff_sh],
            d_x_norm,
            seq_len,
            true,
            lora_h,
            lora_dh,
            lora_bt,
        );
    }
}
