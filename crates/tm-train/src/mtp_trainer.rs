//! Phase `T2a`: Trains a `LinearMtpHead` (`general.type = "mtp"`) on top of any frozen
//! `CpuModel` (`Llama`, `Qwen3`, `Qwen3.5`, `Gemma 4`, `LFM2`) (`ADR-015` §3 & `ARCHITECTURE.md` §5.5).
//!
//! Position alignment (`ARCHITECTURE.md` §5.5):
//! - Base model forward at index `t` (`0 <= t <= T - 3`) produces `hidden[t]`.
//! - `LinearMtpHead` fuses `(hidden[t], embed(tokens[t + 1])) -> mtp_hidden[t + 1]`.
//! - Projecting `mtp_hidden[t + 1]` through frozen `output_norm` + `output` (`lm_head`)
//!   predicts `target = tokens[t + 2]`.

use std::time::Instant;

use tm_core::{DType, quantize_row_to_bytes};
use tm_cpu::{
    MatvecScratch, ThreadPool, active_kernels, matmul_f32, matmul_q_transposed_f32, rms_norm,
    rms_norm_bwd_f32,
};
use tm_runtime::{CpuModel, LinearMtpHead};

use crate::TrainError;
use crate::dataset::TrainDataset;
use crate::optim::{SplitMix64, lr_at_step};

#[derive(Debug, Clone)]
pub struct MtpTrainConfig {
    pub steps: usize,
    pub seq_len: usize,
    pub lr: f32,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub out_dtype: DType,
    pub seed: u64,
    pub log_every: usize,
}

impl Default for MtpTrainConfig {
    fn default() -> Self {
        Self {
            steps: 60,
            seq_len: 128,
            lr: 1e-3,
            warmup_steps: 6,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            out_dtype: DType::F16,
            seed: 42,
            log_every: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MtpTrainReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub initial_acc1: f32,
    pub final_acc1: f32,
    pub total_mtp_tokens: usize,
    pub elapsed_sec: f64,
}

/// Trains a [`LinearMtpHead`] on `dataset` using frozen `model` forward states + frozen `lm_head`
/// transposed backward (`matmul_q_transposed_f32`).
pub fn train_mtp_head<F>(
    model: &CpuModel<'_>,
    pool: &ThreadPool,
    dataset: &TrainDataset,
    cfg: &MtpTrainConfig,
    mut on_step: F,
) -> Result<(LinearMtpHead, MtpTrainReport), TrainError>
where
    F: FnMut(usize, f32, f32, f32),
{
    if cfg.steps == 0 {
        return Err(TrainError::InvalidConfig(
            "train-mtp steps must be >= 1".to_string(),
        ));
    }
    let d = model.n_embd();
    let vocab = model.vocab_size();
    let (base_out_norm, rms_eps, out_dtype, out_bytes, softcap) = model.output_head_raw();
    let kt = active_kernels();

    // Initialize `h_norm = 1`, `e_norm = 1`, and `W_proj: [d, 2*d]` with `0.5 * I_d` residual
    // pass-through on `hidden[t]` plus small Kaiming noise so step 0 starts near base hidden space.
    let mut rng = SplitMix64::new(cfg.seed);
    let mut h_norm = vec![1.0f32; d];
    let mut e_norm = vec![1.0f32; d];
    let bound = 0.02f32 / ((2 * d) as f32).sqrt();
    let mut w_proj = vec![0.0f32; d * (2 * d)];
    for r in 0..d {
        for c in 0..(2 * d) {
            w_proj[r * 2 * d + c] = rng.uniform_sym(bound);
        }
        w_proj[r * 2 * d + r] += 0.5;
    }

    // AdamW moments for `(h_norm, e_norm, w_proj)`
    let mut m_hn = vec![0.0f32; d];
    let mut v_hn = vec![0.0f32; d];
    let mut m_en = vec![0.0f32; d];
    let mut v_en = vec![0.0f32; d];
    let mut m_wp = vec![0.0f32; w_proj.len()];
    let mut v_wp = vec![0.0f32; w_proj.len()];

    let mut d_hn = vec![0.0f32; d];
    let mut d_en = vec![0.0f32; d];
    let mut d_wp = vec![0.0f32; w_proj.len()];

    let max_seq = cfg.seq_len.max(4);
    let mut state = model.new_state(max_seq);
    let mut mv_scratch = MatvecScratch::new(d.max(2 * d));

    const CHUNK: usize = 16;
    let mut hidden_seq = vec![0.0f32; max_seq * d];
    let mut emb_seq = vec![0.0f32; max_seq * d];
    let mut concat_chunk = vec![0.0f32; CHUNK * 2 * d];
    let mut mtp_h_chunk = vec![0.0f32; CHUNK * d];
    let mut x_norm_chunk = vec![0.0f32; CHUNK * d];
    let mut logits_chunk = vec![0.0f32; CHUNK * vocab];
    let mut raw_logits_chunk = if softcap.is_some() {
        vec![0.0f32; CHUNK * vocab]
    } else {
        Vec::new()
    };
    let mut d_x_norm_chunk = vec![0.0f32; CHUNK * d];
    let mut d_mtp_h = vec![0.0f32; d];
    let mut d_concat = vec![0.0f32; 2 * d];
    let mut d_tmp = vec![0.0f32; d];

    // Pre-filter samples with length >= 3 (MTP requires `(t, t+1, t+2)`)
    let usable: Vec<usize> = dataset
        .samples
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            let t_lim = s.tokens.len().min(max_seq);
            t_lim >= 3 && s.loss_mask[2..t_lim].iter().any(|&m| m != 0)
        })
        .map(|(i, _)| i)
        .collect();
    if usable.is_empty() {
        return Err(TrainError::Dataset {
            path: "mtp".to_string(),
            detail: "no sequences with length >= 3 and supervised token[t+2]".to_string(),
        });
    }

    let t_start = Instant::now();
    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let mut initial_acc1 = 0.0f32;
    let mut final_acc1 = 0.0f32;
    let mut total_mtp_tokens = 0usize;

    for step in 0..cfg.steps {
        let sample = &dataset.samples[usable[step % usable.len()]];
        let seq_len = sample.tokens.len().min(max_seq);
        let toks = &sample.tokens[..seq_len];
        let mask = &sample.loss_mask[..seq_len];

        // 1. Collect `hidden[t]` (`t in 0..seq_len - 2`) from frozen base model
        state.reset();
        let n_base = seq_len - 2;
        for (t, &tid) in toks[..n_base].iter().enumerate() {
            model.forward_token(&mut state, pool, kt, tid, t, false)?;
            hidden_seq[t * d..(t + 1) * d].copy_from_slice(state.last_hidden());
            model.embed_token(toks[t + 1], &mut emb_seq[t * d..(t + 1) * d]);
        }

        let valid_pos: Vec<(usize, u32)> = (0..n_base)
            .filter(|&t| mask[t + 2] != 0)
            .map(|t| (t, toks[t + 2]))
            .collect();
        let n_valid = valid_pos.len();
        if n_valid == 0 {
            continue;
        }
        total_mtp_tokens += n_valid;

        d_hn.fill(0.0);
        d_en.fill(0.0);
        d_wp.fill(0.0);

        let mut loss_sum = 0.0f64;
        let mut hits = 0usize;

        // 2. Chunked MTP forward + frozen `lm_head` CE + exact MTP backward
        let mut c0 = 0usize;
        while c0 < n_valid {
            let b = (n_valid - c0).min(CHUNK);
            for k in 0..b {
                let t = valid_pos[c0 + k].0;
                let h_in = &hidden_seq[t * d..(t + 1) * d];
                let e_in = &emb_seq[t * d..(t + 1) * d];
                let cat = &mut concat_chunk[k * 2 * d..(k + 1) * 2 * d];
                rms_norm(&mut cat[..d], h_in, &h_norm, rms_eps);
                rms_norm(&mut cat[d..2 * d], e_in, &e_norm, rms_eps);

                let mh = &mut mtp_h_chunk[k * d..(k + 1) * d];
                for (r, mh_r) in mh.iter_mut().enumerate() {
                    let row = &w_proj[r * 2 * d..(r + 1) * 2 * d];
                    let mut sum = 0.0f32;
                    for (&wi, &ci) in row.iter().zip(cat.iter()) {
                        sum = wi.mul_add(ci, sum);
                    }
                    *mh_r = sum;
                }
                rms_norm(
                    &mut x_norm_chunk[k * d..(k + 1) * d],
                    mh,
                    base_out_norm,
                    rms_eps,
                );
            }

            matmul_f32(
                pool,
                kt,
                out_dtype,
                out_bytes,
                &x_norm_chunk[..b * d],
                &mut mv_scratch,
                &mut logits_chunk[..b * vocab],
                vocab,
                d,
                b,
            );
            if let Some(cap) = softcap {
                raw_logits_chunk[..b * vocab].copy_from_slice(&logits_chunk[..b * vocab]);
                let inv_cap = 1.0 / cap;
                for l in &mut logits_chunk[..b * vocab] {
                    *l = (*l * inv_cap).tanh() * cap;
                }
            }

            let inv_n = 1.0 / (n_valid as f64);
            for k in 0..b {
                let target = valid_pos[c0 + k].1 as usize;
                let row = &mut logits_chunk[k * vocab..(k + 1) * vocab];
                let mut best_v = 0usize;
                let mut max_l = f32::NEG_INFINITY;
                for (v_idx, &lv) in row.iter().enumerate() {
                    if lv > max_l {
                        max_l = lv;
                        best_v = v_idx;
                    }
                }
                if best_v == target {
                    hits += 1;
                }
                let mut sum_exp = 0.0f64;
                for &lv in row.iter() {
                    sum_exp += ((lv - max_l) as f64).exp();
                }
                let log_z = (max_l as f64) + sum_exp.ln();
                loss_sum += log_z - (row[target] as f64);

                for (v_idx, lv) in row.iter_mut().enumerate() {
                    let p = ((*lv as f64) - log_z).exp();
                    let mut dl = (p - if v_idx == target { 1.0 } else { 0.0 }) * inv_n;
                    if let Some(cap) = softcap {
                        let raw_l = raw_logits_chunk[k * vocab + v_idx] as f64;
                        let th = (raw_l / (cap as f64)).tanh();
                        dl *= 1.0 - th * th;
                    }
                    *lv = dl as f32;
                }
            }

            // `d_x_norm_chunk = d_logits * W_out` (zero-copy transposed quantized matmul!)
            matmul_q_transposed_f32(
                pool,
                kt,
                out_dtype,
                out_bytes,
                &logits_chunk[..b * vocab],
                &mut d_x_norm_chunk[..b * d],
                vocab,
                d,
                b,
                false,
            );

            for k in 0..b {
                let t = valid_pos[c0 + k].0;
                let h_in = &hidden_seq[t * d..(t + 1) * d];
                let e_in = &emb_seq[t * d..(t + 1) * d];
                let cat = &concat_chunk[k * 2 * d..(k + 1) * 2 * d];
                let mh = &mtp_h_chunk[k * d..(k + 1) * d];
                let dxn = &d_x_norm_chunk[k * d..(k + 1) * d];

                rms_norm_bwd_f32(&mut d_mtp_h, None, mh, base_out_norm, dxn, rms_eps);

                d_concat.fill(0.0);
                for (r, &dy_r) in d_mtp_h.iter().enumerate() {
                    if dy_r == 0.0 {
                        continue;
                    }
                    (kt.axpy_f32)(&mut d_wp[r * 2 * d..(r + 1) * 2 * d], dy_r, cat);
                    (kt.axpy_f32)(&mut d_concat, dy_r, &w_proj[r * 2 * d..(r + 1) * 2 * d]);
                }

                rms_norm_bwd_f32(
                    &mut d_tmp,
                    Some(&mut d_hn),
                    h_in,
                    &h_norm,
                    &d_concat[..d],
                    rms_eps,
                );
                rms_norm_bwd_f32(
                    &mut d_tmp,
                    Some(&mut d_en),
                    e_in,
                    &e_norm,
                    &d_concat[d..2 * d],
                    rms_eps,
                );
            }

            c0 += b;
        }

        // Global L2 gradient clipping + AdamW update
        let mut g_sq = 0.0f64;
        for &g in d_hn.iter().chain(d_en.iter()).chain(d_wp.iter()) {
            let gf = g as f64;
            g_sq += gf * gf;
        }
        let g_norm = g_sq.sqrt() as f32;
        if cfg.max_grad_norm > 0.0 && g_norm > cfg.max_grad_norm {
            let sc = cfg.max_grad_norm / (g_norm + 1e-12);
            for g in &mut d_hn {
                *g *= sc;
            }
            for g in &mut d_en {
                *g *= sc;
            }
            for g in &mut d_wp {
                *g *= sc;
            }
        }

        let cur_lr = lr_at_step(step, cfg.steps, cfg.warmup_steps, cfg.lr, 0.1);
        let t_step = (step + 1) as i32;
        let b1 = 0.9f64;
        let b2 = 0.999f64;
        let bc1 = 1.0 - b1.powi(t_step);
        let bc2 = 1.0 - b2.powi(t_step);
        let wd = cfg.weight_decay as f64;
        let lr_f = cur_lr as f64;

        let adam_apply = |w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], decay: f64| {
            for i in 0..w.len() {
                let gi = g[i] as f64;
                let mi = b1 * (m[i] as f64) + (1.0 - b1) * gi;
                let vi = b2 * (v[i] as f64) + (1.0 - b2) * gi * gi;
                m[i] = mi as f32;
                v[i] = vi as f32;
                let mh = mi / bc1;
                let vh = vi / bc2;
                w[i] =
                    ((w[i] as f64) * (1.0 - lr_f * decay) - lr_f * mh / (vh.sqrt() + 1e-8)) as f32;
            }
        };
        adam_apply(&mut h_norm, &d_hn, &mut m_hn, &mut v_hn, 0.0);
        adam_apply(&mut e_norm, &d_en, &mut m_en, &mut v_en, 0.0);
        adam_apply(&mut w_proj, &d_wp, &mut m_wp, &mut v_wp, wd);

        let mean_loss = (loss_sum / (n_valid as f64)) as f32;
        let acc1 = (hits as f32) / (n_valid as f32);
        if step == 0 {
            initial_loss = mean_loss;
            initial_acc1 = acc1;
        }
        final_loss = mean_loss;
        final_acc1 = acc1;

        on_step(step + 1, mean_loss, acc1, cur_lr);
    }

    let rb = cfg.out_dtype.row_bytes(2 * d);
    let mut proj_bytes = vec![0u8; d * rb];
    for r in 0..d {
        quantize_row_to_bytes(
            cfg.out_dtype,
            &w_proj[r * 2 * d..(r + 1) * 2 * d],
            &mut proj_bytes[r * rb..(r + 1) * rb],
        );
    }
    let head = LinearMtpHead::new(d, rms_eps, h_norm, e_norm, cfg.out_dtype, proj_bytes);
    let report = MtpTrainReport {
        initial_loss,
        final_loss,
        initial_acc1,
        final_acc1,
        total_mtp_tokens,
        elapsed_sec: t_start.elapsed().as_secs_f64(),
    };
    Ok((head, report))
}
