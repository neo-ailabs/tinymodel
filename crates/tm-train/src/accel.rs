//! Safe GPU/CPU linear acceleration trait (`LinearAccel`) and shared zero-allocation
//! forward/backward linear + LoRA + masked chunked cross-entropy helpers (`ADR-015` T3/T4).

use tm_core::DType;
use tm_cpu::{
    KernelTable, MatvecScratch, ThreadPool, matmul_f32, matmul_q_transposed_f32, matvec_f32,
    rms_norm, rms_norm_bwd_f32, softcap_inplace_parallel,
};
use tm_lora::{BoundLora, LoraAdapter, LoraScratch};
use tm_models::RawMatView;

use crate::optim::LoraGradStore;
use crate::qlora::CE_CHUNK_TOKENS;

/// Optional GPU hardware accelerator (`MetalLinearAccel` / `VulkanLinearAccel`) for frozen
/// quantized base-model forward (`W * x`) and transposed backward (`dy * W`) GEMMs (`Phase T4`).
///
/// Keeping this trait in `tm-train` preserves `#![forbid(unsafe_code)]` across all 5 architecture
/// trainers while allowing Apple Metal and Vulkan GPUs to offload the heavy quantized matrix ops.
pub trait LinearAccel {
    fn backend_name(&self) -> &str;
    fn device_name(&self) -> &str;

    /// Computes `out[n_tok, rows] = x[n_tok, cols] * W^T`.
    /// Returns `Ok(true)` if executed on the accelerator, or `Ok(false)` to fall back to `tm-cpu`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_fwd(
        &mut self,
        name: &str,
        dtype: DType,
        w_bytes: &[u8],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        n_tok: usize,
    ) -> Result<bool, String>;

    /// Computes `dx[n_tok, cols] = (accum ? dx : 0) + dy[n_tok, rows] * W[rows, cols]`.
    /// Returns `Ok(true)` if executed on the accelerator, or `Ok(false)` to fall back to `tm-cpu`.
    #[allow(clippy::too_many_arguments)]
    fn matmul_bwd_transposed(
        &mut self,
        name: &str,
        dtype: DType,
        w_bytes: &[u8],
        dy: &[f32],
        dx: &mut [f32],
        rows: usize,
        cols: usize,
        n_tok: usize,
        accum: bool,
    ) -> Result<bool, String>;
}

/// Evaluates `out = W_base * x + LoRA(x)` for `n_tok` tokens (`single_tok_mode` selects `matvec_f32`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_fwd_with_accel(
    accel: Option<&mut (dyn LinearAccel + '_)>,
    pool: &ThreadPool,
    kt: &KernelTable,
    w: &RawMatView<'_>,
    bound: &BoundLora,
    slot: Option<usize>,
    x: &[f32],
    out: &mut [f32],
    n_tok: usize,
    single_tok_mode: bool,
    mv_scratch: &mut MatvecScratch,
    lora_scratch: &mut LoraScratch,
) {
    let mut handled = false;
    if let Some(acc) = accel
        && let Ok(true) = acc.matmul_fwd(w.name, w.dtype, w.bytes, x, out, w.rows, w.cols, n_tok)
    {
        handled = true;
    }
    if !handled {
        if single_tok_mode && n_tok == 1 {
            matvec_f32(
                pool, kt, w.dtype, w.bytes, x, mv_scratch, out, w.rows, w.cols,
            );
        } else {
            matmul_f32(
                pool, kt, w.dtype, w.bytes, x, mv_scratch, out, w.rows, w.cols, n_tok,
            );
        }
    }
    bound.apply_linear(slot, pool, kt, x, out, n_tok, lora_scratch);
}

/// Symmetric backward for `y = W_base * x + scale * (x * A^T) * B^T` (`AGENTS.md` Rule #5):
/// 1. Offloads `dx = (accum ? dx : 0) + dy * W_base` to `accel` when present, else `matmul_q_transposed_f32`.
/// 2. Accumulates `(dA, dB)` and `out_dx += scale * (dy * B) * A` in deterministic token order (`t = 0..n_tok`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_bwd_with_accel(
    accel: Option<&mut (dyn LinearAccel + '_)>,
    pool: &ThreadPool,
    kt: &KernelTable,
    w: &RawMatView<'_>,
    adapter: &LoraAdapter,
    grads: &mut LoraGradStore,
    x: &[f32],
    dy: &[f32],
    out_dx: &mut [f32],
    n_tok: usize,
    accum: bool,
    lora_h: &mut [f32],
    lora_dh: &mut [f32],
    lora_bt: &mut [f32],
) {
    let mut handled = false;
    if let Some(acc) = accel
        && let Ok(true) = acc.matmul_bwd_transposed(
            w.name, w.dtype, w.bytes, dy, out_dx, w.rows, w.cols, n_tok, accum,
        )
    {
        handled = true;
    }
    if !handled {
        matmul_q_transposed_f32(
            pool, kt, w.dtype, w.bytes, dy, out_dx, w.rows, w.cols, n_tok, accum,
        );
    }

    let Some(pair) = adapter.pairs.get(w.name) else {
        return;
    };
    if pair.is_token_embd {
        return;
    }
    let scale = pair.effective_scale(adapter.alpha, 1.0);
    if scale == 0.0 {
        return;
    }

    let r = pair.rank;
    let d_in = pair.d_in;
    let d_out = pair.d_out;
    let h = &mut lora_h[..n_tok * r];
    let dh = &mut lora_dh[..n_tok * r];
    let b_t = &mut lora_bt[..r * d_out];

    for o in 0..d_out {
        let src = &pair.b[o * r..(o + 1) * r];
        for (r_idx, &val) in src.iter().enumerate() {
            b_t[r_idx * d_out + o] = val;
        }
    }

    let g = grads.grads.get_mut(w.name).expect("lora grad slot");

    for t in 0..n_tok {
        let xt = &x[t * d_in..(t + 1) * d_in];
        let dyt = &dy[t * d_out..(t + 1) * d_out];
        let ht = &mut h[t * r..(t + 1) * r];
        let dht = &mut dh[t * r..(t + 1) * r];

        for r_idx in 0..r {
            let ar = &pair.a[r_idx * d_in..(r_idx + 1) * d_in];
            let mut sum_ax = 0.0f64;
            for (&ai, &xi) in ar.iter().zip(xt.iter()) {
                sum_ax += (ai as f64) * (xi as f64);
            }
            ht[r_idx] = sum_ax as f32;

            let bt_r = &b_t[r_idx * d_out..(r_idx + 1) * d_out];
            let mut sum_dyb = 0.0f64;
            for (&dyo, &bo) in dyt.iter().zip(bt_r.iter()) {
                sum_dyb += (dyo as f64) * (bo as f64);
            }
            dht[r_idx] = (sum_dyb * (scale as f64)) as f32;
        }

        for (o, &dyo) in dyt.iter().enumerate() {
            if dyo == 0.0 {
                continue;
            }
            let s_dyo = dyo * scale;
            let db_row = &mut g.db[o * r..(o + 1) * r];
            for (db_v, &h_v) in db_row.iter_mut().zip(ht.iter()) {
                *db_v = s_dyo.mul_add(h_v, *db_v);
            }
        }

        let dxt = &mut out_dx[t * d_in..(t + 1) * d_in];
        for (r_idx, &dhr) in dht.iter().enumerate() {
            if dhr == 0.0 {
                continue;
            }
            let ar = &pair.a[r_idx * d_in..(r_idx + 1) * d_in];
            (kt.axpy_f32)(&mut g.da[r_idx * d_in..(r_idx + 1) * d_in], dhr, xt);
            (kt.axpy_f32)(dxt, dhr, ar);
        }
    }
}

/// Optional `token_embd.weight` LoRA backward (`A: [vocab, r]`, `B: [d, r]`), supporting
/// post-embedding `emb_scale` (`1.0` for Llama/Qwen3.5/LFM2, `sqrt(n_embd)` for Gemma 4).
pub(crate) fn embed_lora_backward(
    embed_name: &str,
    adapter: &LoraAdapter,
    grads: &mut LoraGradStore,
    input_toks: &[u32],
    dx: &[f32],
    d: usize,
    emb_scale: f32,
) {
    let Some(pair) = adapter.pairs.get(embed_name) else {
        return;
    };
    if !pair.is_token_embd {
        return;
    }
    let scale = pair.effective_scale(adapter.alpha, 1.0) * emb_scale;
    if scale == 0.0 {
        return;
    }
    let r = pair.rank;
    let g = grads.grads.get_mut(embed_name).expect("embed grad slot");
    for (t, &tok) in input_toks.iter().enumerate() {
        let tid = tok as usize;
        let dxt = &dx[t * d..(t + 1) * d];
        let a_row = &pair.a[tid * r..(tid + 1) * r];
        for (r_idx, &ar_val) in a_row.iter().enumerate() {
            let mut dot_dxb = 0.0f64;
            for (o, &dx_o) in dxt.iter().enumerate() {
                dot_dxb += (dx_o as f64) * (pair.b[o * r + r_idx] as f64);
                g.db[o * r + r_idx] = (dx_o * scale).mul_add(ar_val, g.db[o * r + r_idx]);
            }
            g.da[tid * r + r_idx] += (dot_dxb * (scale as f64)) as f32;
        }
    }
}

/// Shared masked chunked cross-entropy (`CE_CHUNK_TOKENS = 16`) forward + backward through
/// `output_norm`, `output` (`lm_head` + optional LoRA), and optional `logit_softcap` (`Gemma 4`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn masked_chunked_ce_fwd_bwd(
    mut accel: Option<&mut (dyn LinearAccel + '_)>,
    pool: &ThreadPool,
    kt: &KernelTable,
    output_norm: &[f32],
    rms_eps: f32,
    output_mat: &RawMatView<'_>,
    logit_softcap: Option<f32>,
    bound_lora: &BoundLora,
    adapter: &LoraAdapter,
    grads: &mut LoraGradStore,
    x_final: &[f32],
    out_dx: &mut [f32],
    ce_pos: &[(usize, u32)],
    grad_scale: f32,
    d: usize,
    vocab: usize,
    ce_x_sub: &mut [f32],
    ce_x_norm: &mut [f32],
    ce_logits: &mut [f32],
    ce_dx_norm: &mut [f32],
    mv_scratch: &mut MatvecScratch,
    lora_scratch: &mut LoraScratch,
    lora_h: &mut [f32],
    lora_dh: &mut [f32],
    lora_bt: &mut [f32],
) -> f32 {
    let n_valid = ce_pos.len();
    out_dx.fill(0.0);
    if n_valid == 0 {
        return 0.0;
    }

    let mut loss_sum = 0.0f64;
    let slot_out = bound_lora.slot_for(output_mat.name);
    let inv_valid = (grad_scale as f64) / (n_valid as f64);

    let mut c0 = 0usize;
    while c0 < n_valid {
        let b = (n_valid - c0).min(CE_CHUNK_TOKENS);
        for k in 0..b {
            let t = ce_pos[c0 + k].0;
            let x_t = &x_final[t * d..(t + 1) * d];
            ce_x_sub[k * d..(k + 1) * d].copy_from_slice(x_t);
            rms_norm(
                &mut ce_x_norm[k * d..(k + 1) * d],
                x_t,
                output_norm,
                rms_eps,
            );
        }

        linear_fwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            output_mat,
            bound_lora,
            slot_out,
            &ce_x_norm[..b * d],
            &mut ce_logits[..b * vocab],
            b,
            false,
            mv_scratch,
            lora_scratch,
        );
        if let Some(cap) = logit_softcap {
            softcap_inplace_parallel(pool, &mut ce_logits[..b * vocab], cap);
        }

        for k in 0..b {
            let target = ce_pos[c0 + k].1 as usize;
            let row = &mut ce_logits[k * vocab..(k + 1) * vocab];
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

            if let Some(cap) = logit_softcap {
                let inv_cap = 1.0 / (cap as f64);
                for (v_idx, lv) in row.iter_mut().enumerate() {
                    let capped = *lv as f64;
                    let p = (capped - log_z).exp();
                    let dl = (p - if v_idx == target { 1.0 } else { 0.0 }) * inv_valid;
                    let t_val = capped * inv_cap;
                    *lv = (dl * (1.0 - t_val * t_val)) as f32;
                }
            } else {
                for (v_idx, lv) in row.iter_mut().enumerate() {
                    let p = ((*lv as f64) - log_z).exp();
                    let dl = (p - if v_idx == target { 1.0 } else { 0.0 }) * inv_valid;
                    *lv = dl as f32;
                }
            }
        }

        linear_bwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            output_mat,
            adapter,
            grads,
            &ce_x_norm[..b * d],
            &ce_logits[..b * vocab],
            &mut ce_dx_norm[..b * d],
            b,
            false,
            lora_h,
            lora_dh,
            lora_bt,
        );

        for k in 0..b {
            let t = ce_pos[c0 + k].0;
            rms_norm_bwd_f32(
                &mut out_dx[t * d..(t + 1) * d],
                None,
                &ce_x_sub[k * d..(k + 1) * d],
                output_norm,
                &ce_dx_norm[k * d..(k + 1) * d],
                rms_eps,
            );
        }

        c0 += b;
    }

    (loss_sum / (n_valid as f64)) as f32
}

/// Evaluates exact `f64` masked cross-entropy loss (forward only) for `< 1e-4` finite-difference checks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn masked_chunked_ce_loss_f64(
    mut accel: Option<&mut (dyn LinearAccel + '_)>,
    pool: &ThreadPool,
    kt: &KernelTable,
    output_norm: &[f32],
    rms_eps: f32,
    output_mat: &RawMatView<'_>,
    logit_softcap: Option<f32>,
    bound_lora: &BoundLora,
    x_final: &[f32],
    ce_pos: &[(usize, u32)],
    d: usize,
    vocab: usize,
    ce_x_norm: &mut [f32],
    ce_logits: &mut [f32],
    mv_scratch: &mut MatvecScratch,
    lora_scratch: &mut LoraScratch,
) -> f64 {
    let n_valid = ce_pos.len();
    if n_valid == 0 {
        return 0.0;
    }
    let mut loss_sum = 0.0f64;
    let slot_out = bound_lora.slot_for(output_mat.name);
    let mut c0 = 0usize;
    while c0 < n_valid {
        let b = (n_valid - c0).min(CE_CHUNK_TOKENS);
        for k in 0..b {
            let t = ce_pos[c0 + k].0;
            rms_norm(
                &mut ce_x_norm[k * d..(k + 1) * d],
                &x_final[t * d..(t + 1) * d],
                output_norm,
                rms_eps,
            );
        }
        linear_fwd_with_accel(
            accel.as_deref_mut(),
            pool,
            kt,
            output_mat,
            bound_lora,
            slot_out,
            &ce_x_norm[..b * d],
            &mut ce_logits[..b * vocab],
            b,
            false,
            mv_scratch,
            lora_scratch,
        );
        if let Some(cap) = logit_softcap {
            softcap_inplace_parallel(pool, &mut ce_logits[..b * vocab], cap);
        }
        for k in 0..b {
            let target = ce_pos[c0 + k].1 as usize;
            let row = &ce_logits[k * vocab..(k + 1) * vocab];
            let max_l = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = row.iter().map(|&lv| ((lv - max_l) as f64).exp()).sum();
            let log_z = (max_l as f64) + sum_exp.ln();
            loss_sum += log_z - (row[target] as f64);
        }
        c0 += b;
    }
    loss_sum / (n_valid as f64)
}
