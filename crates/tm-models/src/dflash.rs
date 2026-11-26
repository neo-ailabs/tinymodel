//! Thinking Machines Lab `DSpark` / `DFlash` (`general.architecture == "dflash"`)
//! 3-layer non-causal block speculative drafter + recurrent Markov head for `Inkling-2-Mini`.
//!
//! Upstream algorithmic reference: `ggml-org/llama.cpp` `src/models/dflash.cpp` &
//! `src/models/dspark.cpp` (`PR #32218`, MIT License, Copyright (c) 2023-2026 Georgi Gerganov
//! and contributors).
//!
//! Strictly obeys `AGENTS.md`:
//! - `#![forbid(unsafe_code)]` (inherited from `tm-models` root — 0 `unsafe` blocks);
//! - Full dtype dispatch (`Rule #5`): `Q8_0` projections, `BF16` Markov / hidden-out weights,
//!   and `F32` norms / fusion / biases all dispatch by their own `DType`;
//! - Zero heap allocations on `inject_token` and `draft_block` (`Rule #4`);
//! - Borrows `token_embd`, `token_embd_norm`, and `output` (`lm_head`) from the target
//!   `Inkling-2-Mini` model (`CPU` or `Vulkan` VRAM), saving `938 MiB` of duplicate weights.

use core::marker::PhantomData;
use tm_core::{DType, dequantize_row_bytes, f16_to_f32};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool, matmul_f32, matvec_f32,
    rms_norm, rms_norm_batch_parallel, rope_compute_cos_sin, rope_inplace_tables,
    swiglu_inplace_parallel,
};
use tm_gguf::{DirectTensorReader, GgufError, GgufFile, PageAlignedSlot};

use crate::llama::ModelError;
use crate::sampler::argmax;
use crate::state::decode_embedding_row;

/// Maximum supported draft block size (`dflash.block_size == 8` on `Inkling-2-Mini`).
pub const MAX_DFLASH_BLOCK_SIZE: usize = 16;

/// Parsed hyperparameters for an `Inkling-2` `DSpark` (`dflash`) drafter GGUF.
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    pub n_layers: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    /// Effective sliding-window span (`dflash.attention.sliding_window + 1` per `dflash.cpp`:
    /// includes the anchor plus `sliding_window` preceding positions).
    pub n_swa_eff: usize,
    pub is_swa: Vec<bool>,
    pub markov_rank: usize,
    pub target_layers: Vec<usize>,
    pub vocab_padded: usize,
    pub vocab_size: usize,
    pub final_logit_softcapping: f32,
}

impl DFlashConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?;
        if arch != "dflash" {
            return Err(ModelError::UnsupportedArch(arch.to_string()));
        }
        let n_layers = gguf.req_usize("dflash.block_count")?;
        let block_size = gguf.req_usize("dflash.block_size")?;
        if !(2..=MAX_DFLASH_BLOCK_SIZE).contains(&block_size) || block_size > MAX_CPU_BATCH {
            return Err(ModelError::InvalidConfig(format!(
                "dflash.block_size={block_size} outside 2..={MAX_DFLASH_BLOCK_SIZE}"
            )));
        }
        let n_embd = gguf.req_usize("dflash.embedding_length")?;
        let n_ff = gguf.req_usize("dflash.feed_forward_length")?;
        let n_head = gguf.req_usize("dflash.attention.head_count")?;
        let n_head_kv = gguf
            .get_meta("dflash.attention.head_count_kv")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim = gguf
            .get_meta("dflash.attention.key_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head.max(1));
        let rope_dim = gguf
            .get_meta("dflash.rope.dimension_count")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf
            .get_meta("dflash.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let rms_eps = gguf
            .get_meta("dflash.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let raw_swa = gguf.req_usize("dflash.attention.sliding_window")?;
        if raw_swa == 0 {
            return Err(ModelError::InvalidConfig(
                "Inkling2 DSpark requires dflash.attention.sliding_window > 0".to_string(),
            ));
        }
        let n_swa_eff = raw_swa + 1;
        let is_swa = gguf
            .get_meta("dflash.attention.sliding_window_pattern")
            .and_then(|v| v.as_bool_array())
            .unwrap_or_else(|| vec![true; n_layers]);
        if is_swa.len() != n_layers {
            return Err(ModelError::InvalidConfig(format!(
                "dflash sliding_window_pattern len {} != n_layers {n_layers}",
                is_swa.len()
            )));
        }
        let target_layers = gguf
            .get_meta("dflash.target_layers")
            .and_then(|v| v.as_usize_array())
            .ok_or_else(|| {
                ModelError::InvalidConfig("missing dflash.target_layers array".to_string())
            })?;
        if target_layers.is_empty() {
            return Err(ModelError::InvalidConfig(
                "dflash.target_layers cannot be empty".to_string(),
            ));
        }
        let sample_from_anchor = gguf
            .get_meta("dflash.sample_from_anchor")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if sample_from_anchor {
            return Err(ModelError::InvalidConfig(
                "Inkling2 DSpark expects dflash.sample_from_anchor == false".to_string(),
            ));
        }
        let causal = gguf
            .get_meta("dflash.attention.causal")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if causal {
            return Err(ModelError::InvalidConfig(
                "Inkling2 DSpark expects non-causal block attention".to_string(),
            ));
        }
        let markov_w1_info =
            gguf.tensor_info("markov_w1.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf.path().display().to_string(),
                    tensor: "markov_w1.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let markov_rank = gguf
            .get_meta("dflash.dspark.markov_rank")
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| markov_w1_info.dims.first().copied().unwrap_or(0));
        if markov_rank == 0 || markov_rank > n_embd {
            return Err(ModelError::InvalidConfig(format!(
                "invalid dspark markov_rank={markov_rank}"
            )));
        }
        let vocab_padded = gguf
            .get_meta("dflash.vocab_size")
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| markov_w1_info.dims.get(1).copied().unwrap_or(0));
        let vocab_size = gguf
            .get_meta("dflash.vocab_size_unpadded")
            .and_then(|v| v.as_usize())
            .unwrap_or(vocab_padded);
        let final_logit_softcapping = gguf
            .get_meta("dflash.final_logit_softcapping")
            .and_then(|v| v.as_f32())
            .unwrap_or(30.0);
        if final_logit_softcapping <= 0.0 {
            return Err(ModelError::InvalidConfig(
                "Inkling2 DSpark requires positive final_logit_softcapping".to_string(),
            ));
        }

        Ok(Self {
            n_layers,
            block_size,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_eps,
            n_swa_eff,
            is_swa,
            markov_rank,
            target_layers,
            vocab_padded,
            vocab_size,
            final_logit_softcapping,
        })
    }

    #[inline]
    pub const fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    #[inline]
    pub const fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    #[inline]
    pub const fn max_draft_tokens(&self) -> usize {
        self.block_size.saturating_sub(1)
    }
}

#[derive(Clone)]
struct DFlashMat {
    bytes: Vec<u8>,
    dtype: DType,
}

struct DFlashLayerWeights {
    dflare_norm: Vec<f32>,
    attn_k_ctx: DFlashMat,
    attn_v_ctx: DFlashMat,
    attn_k_ctx_norm: Vec<f32>,
    attn_norm: Vec<f32>,
    wq: DFlashMat,
    wk: DFlashMat,
    wv: DFlashMat,
    wo: DFlashMat,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: DFlashMat,
    ffn_up: DFlashMat,
    ffn_down: DFlashMat,
}

/// Resident `DSpark` (`dflash`) block speculative drafter model (`383.6 MiB` in `Q8_0`).
///
/// Owns its `383.6 MiB` weight buffers in anonymous RAM (populated via `O_DIRECT` when backed
/// by a file on disk) so Linux `kswapd` can never evict `DSpark` pages under tight RAM!
pub struct DFlashModel<'a> {
    pub cfg: DFlashConfig,
    /// Pre-softmaxed rows `[n_layers + 1][n_target_taps]` of `dspark_fuse.weight`.
    fuse_probs: Vec<Vec<f32>>,
    dspark_norm: Vec<f32>,
    mask_embd: Vec<f32>,
    output_norm: Vec<f32>,
    markov_w1: DFlashMat,
    joint_proj: DFlashMat,
    joint_proj_b: Vec<f32>,
    hidden_out: DFlashMat,
    hidden_out_b: Vec<f32>,
    rope_inv_freqs: Vec<f64>,
    layers: Vec<DFlashLayerWeights>,
    _marker: PhantomData<&'a ()>,
}

/// Pre-allocated per-session state for `DFlashModel` (0 heap allocations on draft / inject path).
pub struct DFlashState {
    pub max_ctx: usize,
    /// Committed F16 K cache `[n_layers][max_ctx * kv_dim]`.
    k_cache: Vec<Vec<u16>>,
    /// Committed F16 V cache `[n_layers][max_ctx * kv_dim]`.
    v_cache: Vec<Vec<u16>>,
    /// Ring buffer of `dspark_norm(fused_taps)` for the most recent `ring_cap = block_size + 2` positions.
    ctx_feat_ring: Vec<f32>,
    ctx_feat_pos: Vec<isize>,
    ring_cap: usize,
    /// Scratch buffers for `inject_token` and `draft_block` (`block_size` rows).
    mix_tmp: Vec<f32>,
    inj_norm: Vec<f32>,
    inj_k: Vec<f32>,
    inj_v: Vec<f32>,
    inj_head_tmp: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    inp_l: Vec<f32>,
    norm_blk: Vec<f32>,
    q_blk: Vec<f32>,
    k_blk: Vec<f32>,
    v_blk: Vec<f32>,
    attn_blk: Vec<f32>,
    proj_blk: Vec<f32>,
    ffn_gate_blk: Vec<f32>,
    ffn_up_blk: Vec<f32>,
    attn_head_scratch: Vec<(Vec<f32>, Vec<f32>)>,
    kv_f16_tmp: Vec<u16>,
    hidden_blk: Vec<f32>,
    rnn_state: Vec<f32>,
    prev_embd: Vec<f32>,
    joint_in: Vec<f32>,
    joint_out: Vec<f32>,
    latent: Vec<f32>,
    correction: Vec<f32>,
    corr_hidden: Vec<f32>,
    logits_buf: Vec<f32>,
    mv_scratch: MatvecScratch,
}

fn load_f32_vec(
    gguf: &GgufFile,
    direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    expected_dims: &[usize],
) -> Result<Vec<f32>, ModelError> {
    let n: usize = expected_dims.iter().product();
    let mut out = vec![0.0f32; n];
    if let Some((rdr, slot)) = direct {
        let (info, f_off, b_len) = gguf.req_tensor_file_range(name, layer, expected_dims)?;
        let mut raw = vec![0u8; b_len];
        rdr.read_exact_uncached(f_off, &mut raw, slot)?;
        dequantize_row_bytes(info.dtype, &raw, &mut out);
    } else {
        let (info, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
        dequantize_row_bytes(info.dtype, bytes, &mut out);
    }
    Ok(out)
}

fn load_mat(
    gguf: &GgufFile,
    direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<DFlashMat, ModelError> {
    if let Some((rdr, slot)) = direct {
        let (info, f_off, b_len) = gguf.req_tensor_file_range(name, layer, &[cols, rows])?;
        let mut owned = vec![0u8; b_len];
        rdr.read_exact_uncached(f_off, &mut owned, slot)?;
        Ok(DFlashMat {
            bytes: owned,
            dtype: info.dtype,
        })
    } else {
        let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
        Ok(DFlashMat {
            bytes: bytes.to_vec(),
            dtype: info.dtype,
        })
    }
}

impl<'a> DFlashModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let cfg = DFlashConfig::from_gguf(gguf)?;
        let direct_reader = if gguf.path().exists() {
            DirectTensorReader::open(gguf.path()).ok()
        } else {
            None
        };
        let mut io_slot = PageAlignedSlot::new(8 * 1024 * 1024);
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hd = cfg.head_dim;
        let rank = cfg.markov_rank;
        let n_taps = cfg.target_layers.len();

        let fuse_raw = load_f32_vec(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_fuse.weight",
            "global",
            &[n_taps, cfg.n_layers + 1],
        )?;
        let mut fuse_probs = Vec::with_capacity(cfg.n_layers + 1);
        for r in 0..=cfg.n_layers {
            let row = &fuse_raw[r * n_taps..(r + 1) * n_taps];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0.0f32; n_taps];
            let mut sum = 0.0f32;
            for (p, &v) in probs.iter_mut().zip(row.iter()) {
                let e = (v - max_v).exp();
                *p = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for p in &mut probs {
                *p *= inv;
            }
            fuse_probs.push(probs);
        }

        let dspark_norm = load_f32_vec(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_norm.weight",
            "global",
            &[n_embd],
        )?;
        let mask_embd = load_f32_vec(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "mask_embd.weight",
            "global",
            &[n_embd],
        )?;
        let output_norm = load_f32_vec(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output_norm.weight",
            "global",
            &[n_embd],
        )?;
        let markov_w1 = load_mat(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "markov_w1.weight",
            "global",
            rank,
            cfg.vocab_padded,
        )?;
        let joint_proj = load_mat(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "joint_proj.weight",
            "global",
            2 * n_embd + 2 * rank,
            3 * rank,
        )?;
        let joint_proj_b = load_f32_vec(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "joint_proj.bias",
            "global",
            &[3 * rank],
        )?;
        let hidden_out = load_mat(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_hidden_out.weight",
            "global",
            rank,
            n_embd,
        )?;
        let hidden_out_b = load_f32_vec(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_hidden_out.bias",
            "global",
            &[n_embd],
        )?;

        let half_rope = cfg.rope_dim / 2;
        let mut rope_inv_freqs = vec![0.0f64; half_rope];
        let base = cfg.rope_freq_base as f64;
        for (i, slot) in rope_inv_freqs.iter_mut().enumerate() {
            *slot = base.powf(-((2 * i) as f64) / (cfg.rope_dim as f64));
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let ltag = format!("blk.{il}");
            layers.push(DFlashLayerWeights {
                dflare_norm: load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.dflare_norm.weight"),
                    &ltag,
                    &[n_embd],
                )?,
                attn_k_ctx: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_k_ctx.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                attn_v_ctx: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_v_ctx.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                attn_k_ctx_norm: load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_k_ctx_norm.weight"),
                    &ltag,
                    &[hd],
                )?,
                attn_norm: load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_norm.weight"),
                    &ltag,
                    &[n_embd],
                )?,
                wq: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_q.weight"),
                    &ltag,
                    n_embd,
                    q_dim,
                )?,
                wk: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wv: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_v.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wo: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                )?,
                attn_q_norm: load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_q_norm.weight"),
                    &ltag,
                    &[hd],
                )?,
                attn_k_norm: load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.attn_k_norm.weight"),
                    &ltag,
                    &[hd],
                )?,
                ffn_norm: load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.ffn_norm.weight"),
                    &ltag,
                    &[n_embd],
                )?,
                ffn_gate: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.ffn_gate.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                ffn_up: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.ffn_up.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                ffn_down: load_mat(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{il}.ffn_down.weight"),
                    &ltag,
                    n_ff,
                    n_embd,
                )?,
            });
        }

        Ok(Self {
            cfg,
            fuse_probs,
            dspark_norm,
            mask_embd,
            output_norm,
            markov_w1,
            joint_proj,
            joint_proj_b,
            hidden_out,
            hidden_out_b,
            rope_inv_freqs,
            layers,
            _marker: PhantomData,
        })
    }

    fn mix_target_features(
        &self,
        taps: &[f32],
        row: usize,
        norm_weight: &[f32],
        mix_tmp: &mut [f32],
        out: &mut [f32],
    ) {
        let n_embd = self.cfg.n_embd;
        let weights = &self.fuse_probs[row];
        debug_assert_eq!(taps.len(), weights.len() * n_embd);
        mix_tmp[..n_embd].fill(0.0);
        for (t, &w) in weights.iter().enumerate() {
            let tap = &taps[t * n_embd..(t + 1) * n_embd];
            for (dst, &v) in mix_tmp[..n_embd].iter_mut().zip(tap.iter()) {
                *dst = v.mul_add(w, *dst);
            }
        }
        rms_norm(
            &mut out[..n_embd],
            &mix_tmp[..n_embd],
            norm_weight,
            self.cfg.rms_eps,
        );
    }

    /// Injects committed target model feature taps (`[target_layers.len() * n_embd]`) at `pos`
    /// into the 3-layer draft KV cache and stores `dspark_norm` context features for `pos`.
    pub fn inject_token(
        &self,
        state: &mut DFlashState,
        pool: &ThreadPool,
        kt: &KernelTable,
        pos: usize,
        target_taps: &[f32],
    ) -> Result<(), ModelError> {
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        let n_embd = self.cfg.n_embd;
        let kv_dim = self.cfg.kv_dim();
        let hd = self.cfg.head_dim;
        assert_eq!(target_taps.len(), self.cfg.target_layers.len() * n_embd);

        rope_compute_cos_sin(
            &self.rope_inv_freqs,
            pos,
            &mut state.rope_cos,
            &mut state.rope_sin,
        );

        for (il, layer) in self.layers.iter().enumerate() {
            self.mix_target_features(
                target_taps,
                il,
                &layer.dflare_norm,
                &mut state.mix_tmp,
                &mut state.inj_norm,
            );
            matvec_f32(
                pool,
                kt,
                layer.attn_k_ctx.dtype,
                &layer.attn_k_ctx.bytes,
                &state.inj_norm[..n_embd],
                &mut state.mv_scratch,
                &mut state.inj_k[..kv_dim],
                kv_dim,
                n_embd,
            );
            matvec_f32(
                pool,
                kt,
                layer.attn_v_ctx.dtype,
                &layer.attn_v_ctx.bytes,
                &state.inj_norm[..n_embd],
                &mut state.mv_scratch,
                &mut state.inj_v[..kv_dim],
                kv_dim,
                n_embd,
            );

            for h in 0..self.cfg.n_head_kv {
                let s = h * hd;
                let e = s + hd;
                rms_norm(
                    &mut state.inj_head_tmp[..hd],
                    &state.inj_k[s..e],
                    &layer.attn_k_ctx_norm,
                    self.cfg.rms_eps,
                );
                state.inj_k[s..e].copy_from_slice(&state.inj_head_tmp[..hd]);
            }
            rope_inplace_tables(
                &mut state.inj_k[..kv_dim],
                self.cfg.n_head_kv,
                hd,
                self.cfg.rope_dim,
                &state.rope_cos,
                &state.rope_sin,
                RopeType::Neox,
            );

            let kv_off = pos * kv_dim;
            (kt.f32_to_f16_slice)(
                &state.inj_k[..kv_dim],
                &mut state.k_cache[il][kv_off..kv_off + kv_dim],
            );
            (kt.f32_to_f16_slice)(
                &state.inj_v[..kv_dim],
                &mut state.v_cache[il][kv_off..kv_off + kv_dim],
            );
        }

        let ring_slot = pos % state.ring_cap;
        let feat_off = ring_slot * n_embd;
        self.mix_target_features(
            target_taps,
            self.cfg.n_layers,
            &self.dspark_norm,
            &mut state.mix_tmp,
            &mut state.ctx_feat_ring[feat_off..feat_off + n_embd],
        );
        state.ctx_feat_pos[ring_slot] = pos as isize;
        Ok(())
    }

    /// Injects `count` consecutive committed tokens starting at `start_pos`.
    /// `taps_batch` has length `count * target_layers.len() * n_embd`.
    pub fn inject_batch(
        &self,
        state: &mut DFlashState,
        pool: &ThreadPool,
        kt: &KernelTable,
        start_pos: usize,
        count: usize,
        taps_batch: &[f32],
    ) -> Result<(), ModelError> {
        let stride = self.cfg.target_layers.len() * self.cfg.n_embd;
        assert_eq!(taps_batch.len(), count * stride);
        for i in 0..count {
            self.inject_token(
                state,
                pool,
                kt,
                start_pos + i,
                &taps_batch[i * stride..(i + 1) * stride],
            )?;
        }
        Ok(())
    }

    /// Drafts `k_drafts` (`1 <= k_drafts <= block_size - 1`) speculative tokens starting after
    /// `anchor_token_id` at position `pos0` (`pos0 >= 1`, where `pos0 - 1` was already injected!).
    ///
    /// - `embed_anchor_normed(anchor_token_id, &mut [f32; n_embd])` computes
    ///   `rms_norm(target.tok_embd[anchor], target.token_embd_norm, target.rms_eps)`.
    /// - `project_lm_head(&[f32; n_embd], &mut [f32; vocab_size])` computes unpadded raw logits
    ///   `(target.output * x)[..vocab_size]` (on CPU or GPU VRAM!), before `DSpark` softcapping.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_block<E, P>(
        &self,
        state: &mut DFlashState,
        pool: &ThreadPool,
        kt: &KernelTable,
        pos0: usize,
        anchor_token_id: u32,
        k_drafts: usize,
        embed_anchor_normed: E,
        mut project_lm_head: P,
        out_drafts: &mut [u32],
    ) -> Result<usize, ModelError>
    where
        E: FnMut(u32, &mut [f32]),
        P: FnMut(&[f32], &mut [f32]),
    {
        let vocab_size = self.cfg.vocab_size;
        let mut logits_tmp = core::mem::take(&mut state.logits_buf);
        let res = self.draft_block_greedy(
            state,
            pool,
            kt,
            pos0,
            anchor_token_id,
            k_drafts,
            embed_anchor_normed,
            |corr_hidden, cap| {
                project_lm_head(corr_hidden, &mut logits_tmp[..vocab_size]);
                let inv_cap = 1.0 / cap;
                for lg in &mut logits_tmp[..vocab_size] {
                    *lg = cap * (*lg * inv_cap).tanh();
                }
                argmax(&logits_tmp[..vocab_size])
            },
            out_drafts,
        );
        state.logits_buf = logits_tmp;
        res
    }

    /// Greedy block drafting variant where `(target.output * corr_hidden)[..vocab_size] -> softcap -> argmax`
    /// is evaluated directly by `project_and_argmax` (e.g., on GPU VRAM in `< 1.5 ms` returning only `4` bytes!).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_block_greedy<E, A>(
        &self,
        state: &mut DFlashState,
        pool: &ThreadPool,
        kt: &KernelTable,
        pos0: usize,
        anchor_token_id: u32,
        k_drafts: usize,
        mut embed_anchor_normed: E,
        mut project_and_argmax: A,
        out_drafts: &mut [u32],
    ) -> Result<usize, ModelError>
    where
        E: FnMut(u32, &mut [f32]),
        A: FnMut(&[f32], f32) -> u32,
    {
        let bs = self.cfg.block_size;
        let max_k = self.cfg.max_draft_tokens();
        let k = k_drafts.min(max_k).min(out_drafts.len());
        if k == 0 || pos0 == 0 || pos0 + bs > state.max_ctx {
            return Ok(0);
        }
        let prev_pos = pos0 - 1;
        let ring_slot = prev_pos % state.ring_cap;
        if state.ctx_feat_pos[ring_slot] != prev_pos as isize {
            return Ok(0);
        }
        if (anchor_token_id as usize) >= self.cfg.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token_id: anchor_token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }

        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let q_dim = self.cfg.q_dim();
        let kv_dim = self.cfg.kv_dim();
        let nh = self.cfg.n_head;
        let nh_kv = self.cfg.n_head_kv;
        let hd = self.cfg.head_dim;
        let gqa_ratio = nh / nh_kv;
        let attn_scale = 1.0 / (hd as f32).sqrt();

        // 1. Build 8-row block input embeddings `[rms_norm(tok_embd[anchor], tok_norm), mask_embd * 7]`
        embed_anchor_normed(anchor_token_id, &mut state.inp_l[..n_embd]);
        for r in 1..bs {
            state.inp_l[r * n_embd..(r + 1) * n_embd].copy_from_slice(&self.mask_embd);
        }

        // 2. Run 3 non-causal block-attention Transformer stages (`il in 0..n_layers`)
        for (il, layer) in self.layers.iter().enumerate() {
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.norm_blk[..bs * n_embd],
                &state.inp_l[..bs * n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
                n_embd,
                bs,
            );
            matmul_f32(
                pool,
                kt,
                layer.wq.dtype,
                &layer.wq.bytes,
                &state.norm_blk[..bs * n_embd],
                &mut state.mv_scratch,
                &mut state.q_blk[..bs * q_dim],
                q_dim,
                n_embd,
                bs,
            );
            matmul_f32(
                pool,
                kt,
                layer.wk.dtype,
                &layer.wk.bytes,
                &state.norm_blk[..bs * n_embd],
                &mut state.mv_scratch,
                &mut state.k_blk[..bs * kv_dim],
                kv_dim,
                n_embd,
                bs,
            );
            matmul_f32(
                pool,
                kt,
                layer.wv.dtype,
                &layer.wv.bytes,
                &state.norm_blk[..bs * n_embd],
                &mut state.mv_scratch,
                &mut state.v_blk[..bs * kv_dim],
                kv_dim,
                n_embd,
                bs,
            );

            // Per-head Q/K RMSNorm + NeoX RoPE at `pos0 + r`
            for r in 0..bs {
                rope_compute_cos_sin(
                    &self.rope_inv_freqs,
                    pos0 + r,
                    &mut state.rope_cos,
                    &mut state.rope_sin,
                );
                let q_row = &mut state.q_blk[r * q_dim..(r + 1) * q_dim];
                for h in 0..nh {
                    let s = h * hd;
                    let e = s + hd;
                    rms_norm(
                        &mut state.inj_head_tmp[..hd],
                        &q_row[s..e],
                        &layer.attn_q_norm,
                        self.cfg.rms_eps,
                    );
                    q_row[s..e].copy_from_slice(&state.inj_head_tmp[..hd]);
                }
                rope_inplace_tables(
                    q_row,
                    nh,
                    hd,
                    self.cfg.rope_dim,
                    &state.rope_cos,
                    &state.rope_sin,
                    RopeType::Neox,
                );

                let k_row = &mut state.k_blk[r * kv_dim..(r + 1) * kv_dim];
                for h in 0..nh_kv {
                    let s = h * hd;
                    let e = s + hd;
                    rms_norm(
                        &mut state.inj_head_tmp[..hd],
                        &k_row[s..e],
                        &layer.attn_k_norm,
                        self.cfg.rms_eps,
                    );
                    k_row[s..e].copy_from_slice(&state.inj_head_tmp[..hd]);
                }
                rope_inplace_tables(
                    k_row,
                    nh_kv,
                    hd,
                    self.cfg.rope_dim,
                    &state.rope_cos,
                    &state.rope_sin,
                    RopeType::Neox,
                );
            }

            // Round-trip block K/V through F16 so block K/V precision matches ggml's F16 KV cache!
            (kt.f32_to_f16_slice)(
                &state.k_blk[..bs * kv_dim],
                &mut state.kv_f16_tmp[..bs * kv_dim],
            );
            for (dst, &h16) in state.k_blk[..bs * kv_dim]
                .iter_mut()
                .zip(&state.kv_f16_tmp[..bs * kv_dim])
            {
                *dst = f16_to_f32(h16);
            }
            (kt.f32_to_f16_slice)(
                &state.v_blk[..bs * kv_dim],
                &mut state.kv_f16_tmp[..bs * kv_dim],
            );
            for (dst, &h16) in state.v_blk[..bs * kv_dim]
                .iter_mut()
                .zip(&state.kv_f16_tmp[..bs * kv_dim])
            {
                *dst = f16_to_f32(h16);
            }

            // Non-causal block attention over committed cache `[ctx_start .. pos0)` + all `bs` block keys!
            // Note (`dspark.cpp:997`): `block_uniform = true` -> whole block shares the SWA window of
            // the anchor (`pos0`), where `n_swa_eff = raw_swa + 1` -> `ctx_start = (pos0 + 1).saturating_sub(n_swa_eff)`.
            let ctx_start = if self.cfg.is_swa[il] {
                (pos0 + 1).saturating_sub(self.cfg.n_swa_eff)
            } else {
                0
            };
            let ctx_len = pos0 - ctx_start;
            let total_kv = ctx_len + bs;
            let k_cache_il = &state.k_cache[il];
            let v_cache_il = &state.v_cache[il];
            let q_blk_ref = &state.q_blk[..bs * q_dim];
            let k_blk_ref = &state.k_blk[..bs * kv_dim];
            let v_blk_ref = &state.v_blk[..bs * kv_dim];

            pool.parallel_for_each_mut(
                &mut state.attn_head_scratch[..bs * nh],
                |rh, (out_h, sc_buf)| {
                    let r = rh / nh;
                    let h = rh % nh;
                    let kvh = h / gqa_ratio;
                    let q_h = &q_blk_ref[r * q_dim + h * hd..r * q_dim + (h + 1) * hd];
                    let sc = &mut sc_buf[..total_kv];
                    let mut max_s = f32::NEG_INFINITY;

                    for (idx, kv_pos) in (ctx_start..pos0).enumerate() {
                        let base = kv_pos * kv_dim + kvh * hd;
                        let s = (kt.dot_f32_f16)(q_h, &k_cache_il[base..base + hd]) * attn_scale;
                        sc[idx] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    for br in 0..bs {
                        let k_bh = &k_blk_ref[br * kv_dim + kvh * hd..br * kv_dim + (kvh + 1) * hd];
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot = q_h[d].mul_add(k_bh[d], dot);
                        }
                        let s = dot * attn_scale;
                        sc[ctx_len + br] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }

                    let mut sum_e = 0.0f32;
                    for s in sc.iter_mut() {
                        let e = (*s - max_s).exp();
                        *s = e;
                        sum_e += e;
                    }
                    let inv_sum = 1.0 / sum_e;

                    out_h[..hd].fill(0.0);
                    for (idx, kv_pos) in (ctx_start..pos0).enumerate() {
                        let p = sc[idx] * inv_sum;
                        let base = kv_pos * kv_dim + kvh * hd;
                        (kt.axpy_f32_f16)(&mut out_h[..hd], p, &v_cache_il[base..base + hd]);
                    }
                    for br in 0..bs {
                        let p = sc[ctx_len + br] * inv_sum;
                        let v_bh = &v_blk_ref[br * kv_dim + kvh * hd..br * kv_dim + (kvh + 1) * hd];
                        for d in 0..hd {
                            out_h[d] = v_bh[d].mul_add(p, out_h[d]);
                        }
                    }
                },
            );
            for (rh, (out_h, _)) in state.attn_head_scratch[..bs * nh].iter().enumerate() {
                let r = rh / nh;
                let h = rh % nh;
                let dst_off = r * q_dim + h * hd;
                state.attn_blk[dst_off..dst_off + hd].copy_from_slice(&out_h[..hd]);
            }

            matmul_f32(
                pool,
                kt,
                layer.wo.dtype,
                &layer.wo.bytes,
                &state.attn_blk[..bs * q_dim],
                &mut state.mv_scratch,
                &mut state.proj_blk[..bs * n_embd],
                n_embd,
                q_dim,
                bs,
            );
            for i in 0..(bs * n_embd) {
                state.inp_l[i] += state.proj_blk[i];
            }

            // SwiGLU FFN (`bs` rows)
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.norm_blk[..bs * n_embd],
                &state.inp_l[..bs * n_embd],
                &layer.ffn_norm,
                self.cfg.rms_eps,
                n_embd,
                bs,
            );
            matmul_f32(
                pool,
                kt,
                layer.ffn_gate.dtype,
                &layer.ffn_gate.bytes,
                &state.norm_blk[..bs * n_embd],
                &mut state.mv_scratch,
                &mut state.ffn_gate_blk[..bs * n_ff],
                n_ff,
                n_embd,
                bs,
            );
            matmul_f32(
                pool,
                kt,
                layer.ffn_up.dtype,
                &layer.ffn_up.bytes,
                &state.norm_blk[..bs * n_embd],
                &mut state.mv_scratch,
                &mut state.ffn_up_blk[..bs * n_ff],
                n_ff,
                n_embd,
                bs,
            );
            swiglu_inplace_parallel(
                pool,
                &mut state.ffn_gate_blk[..bs * n_ff],
                &state.ffn_up_blk[..bs * n_ff],
            );
            matmul_f32(
                pool,
                kt,
                layer.ffn_down.dtype,
                &layer.ffn_down.bytes,
                &state.ffn_gate_blk[..bs * n_ff],
                &mut state.mv_scratch,
                &mut state.proj_blk[..bs * n_embd],
                n_embd,
                n_ff,
                bs,
            );
            for i in 0..(bs * n_embd) {
                state.inp_l[i] += state.proj_blk[i];
            }
        }

        // 3. Final decoder RMSNorm (`hidden_blk`: `[bs, n_embd]`)
        rms_norm_batch_parallel(
            pool,
            kt,
            &mut state.hidden_blk[..bs * n_embd],
            &state.inp_l[..bs * n_embd],
            &self.output_norm,
            self.cfg.rms_eps,
            n_embd,
            bs,
        );

        // 4. Chained Recurrent Markov Head (`dspark.cpp:1001-1030`, `1154-1174`)
        let rank = self.cfg.markov_rank;
        let cap = self.cfg.final_logit_softcapping;
        let ctx_off = ring_slot * n_embd;

        state.rnn_state[..rank].fill(0.0);
        let mut prev_tok = anchor_token_id;

        for (step, dst_tok) in out_drafts.iter_mut().take(k).enumerate() {
            decode_embedding_row(
                self.markov_w1.dtype,
                &self.markov_w1.bytes,
                rank,
                prev_tok as usize,
                &mut state.prev_embd[..rank],
            );

            // `joint_in = [rnn_state(rank) ; prev_embd(rank) ; hidden_blk[step + 1](n_embd) ; ctx_feat(n_embd)]`
            let row_off = (step + 1) * n_embd;
            state.joint_in[..rank].copy_from_slice(&state.rnn_state[..rank]);
            state.joint_in[rank..2 * rank].copy_from_slice(&state.prev_embd[..rank]);
            state.joint_in[2 * rank..2 * rank + n_embd]
                .copy_from_slice(&state.hidden_blk[row_off..row_off + n_embd]);
            state.joint_in[2 * rank + n_embd..2 * rank + 2 * n_embd]
                .copy_from_slice(&state.ctx_feat_ring[ctx_off..ctx_off + n_embd]);

            matvec_f32(
                pool,
                kt,
                self.joint_proj.dtype,
                &self.joint_proj.bytes,
                &state.joint_in[..2 * rank + 2 * n_embd],
                &mut state.mv_scratch,
                &mut state.joint_out[..3 * rank],
                3 * rank,
                2 * rank + 2 * n_embd,
            );
            for i in 0..(3 * rank) {
                state.joint_out[i] += self.joint_proj_b[i];
            }

            for j in 0..rank {
                let gate = 1.0 / (1.0 + (-state.joint_out[j]).exp());
                let cand = state.joint_out[rank + j].tanh();
                let st = state.rnn_state[j];
                state.rnn_state[j] = cand + gate * (st - cand);
                state.latent[j] = state.joint_out[2 * rank + j].tanh();
            }

            matvec_f32(
                pool,
                kt,
                self.hidden_out.dtype,
                &self.hidden_out.bytes,
                &state.latent[..rank],
                &mut state.mv_scratch,
                &mut state.correction[..n_embd],
                n_embd,
                rank,
            );
            for c in 0..n_embd {
                state.corr_hidden[c] =
                    state.hidden_blk[row_off + c] + state.correction[c] + self.hidden_out_b[c];
            }

            let picked = project_and_argmax(&state.corr_hidden[..n_embd], cap);
            *dst_tok = picked;
            prev_tok = picked;
        }

        Ok(k)
    }
}

impl DFlashState {
    pub fn new(model: &DFlashModel<'_>, max_ctx: usize) -> Self {
        let cfg = &model.cfg;
        let bs = cfg.block_size;
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let rank = cfg.markov_rank;
        let ring_cap = bs + 4;
        let max_cols = (2 * n_embd + 2 * rank).max(n_ff).max(q_dim);

        Self {
            max_ctx,
            k_cache: (0..cfg.n_layers)
                .map(|_| vec![0u16; max_ctx * kv_dim])
                .collect(),
            v_cache: (0..cfg.n_layers)
                .map(|_| vec![0u16; max_ctx * kv_dim])
                .collect(),
            ctx_feat_ring: vec![0.0; ring_cap * n_embd],
            ctx_feat_pos: vec![-1; ring_cap],
            ring_cap,
            mix_tmp: vec![0.0; n_embd],
            inj_norm: vec![0.0; n_embd],
            inj_k: vec![0.0; kv_dim],
            inj_v: vec![0.0; kv_dim],
            inj_head_tmp: vec![0.0; cfg.head_dim],
            rope_cos: vec![0.0; cfg.rope_dim / 2],
            rope_sin: vec![0.0; cfg.rope_dim / 2],
            inp_l: vec![0.0; bs * n_embd],
            norm_blk: vec![0.0; bs * n_embd],
            q_blk: vec![0.0; bs * q_dim],
            k_blk: vec![0.0; bs * kv_dim],
            v_blk: vec![0.0; bs * kv_dim],
            attn_blk: vec![0.0; bs * q_dim],
            proj_blk: vec![0.0; bs * n_embd],
            ffn_gate_blk: vec![0.0; bs * n_ff],
            ffn_up_blk: vec![0.0; bs * n_ff],
            attn_head_scratch: (0..bs * cfg.n_head)
                .map(|_| (vec![0.0; cfg.head_dim], vec![0.0; max_ctx + bs]))
                .collect(),
            kv_f16_tmp: vec![0; bs * kv_dim],
            hidden_blk: vec![0.0; bs * n_embd],
            rnn_state: vec![0.0; rank],
            prev_embd: vec![0.0; rank],
            joint_in: vec![0.0; 2 * rank + 2 * n_embd],
            joint_out: vec![0.0; 3 * rank],
            latent: vec![0.0; rank],
            correction: vec![0.0; n_embd],
            corr_hidden: vec![0.0; n_embd],
            logits_buf: vec![0.0; cfg.vocab_size],
            mv_scratch: MatvecScratch::new(max_cols),
        }
    }

    pub fn reset(&mut self) {
        for k in &mut self.k_cache {
            k.fill(0);
        }
        for v in &mut self.v_cache {
            v.fill(0);
        }
        self.ctx_feat_ring.fill(0.0);
        self.ctx_feat_pos.fill(-1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tm_cpu::active_kernels;

    #[test]
    fn loads_real_dspark_inkling2_mini_if_present() {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf");
        if !p.exists() {
            return;
        }
        let gguf = GgufFile::open(&p).unwrap();
        let model = DFlashModel::from_gguf(&gguf).unwrap();
        assert_eq!(model.cfg.n_layers, 3);
        assert_eq!(model.cfg.block_size, 8);
        assert_eq!(model.cfg.n_embd, 2304);
        assert_eq!(model.cfg.n_ff, 9216);
        assert_eq!(model.cfg.markov_rank, 256);
        assert_eq!(model.cfg.target_layers, vec![5, 23, 41, 48]);
        assert_eq!(model.cfg.n_swa_eff, 1025);
        assert_eq!(model.cfg.vocab_padded, 201024);
        assert_eq!(model.cfg.vocab_size, 200058);

        let mut state = DFlashState::new(&model, 64);
        let pool = ThreadPool::new(4);
        let kt = active_kernels();
        let fake_taps = vec![0.05f32; 4 * 2304];
        model
            .inject_token(&mut state, &pool, kt, 0, &fake_taps)
            .unwrap();
        model
            .inject_token(&mut state, &pool, kt, 1, &fake_taps)
            .unwrap();

        let mut drafts = [0u32; 3];
        let n = model
            .draft_block(
                &mut state,
                &pool,
                kt,
                2,
                42,
                3,
                |tid, out| {
                    for (i, v) in out.iter_mut().enumerate() {
                        *v = ((tid as usize + i) % 17) as f32 * 0.02;
                    }
                },
                |hid, logits| {
                    for (v_idx, lg) in logits.iter_mut().enumerate().take(256) {
                        *lg = hid[v_idx % hid.len()] * ((v_idx % 7) as f32 * 0.1 + 0.1);
                    }
                    logits[256..].fill(-5.0);
                },
                &mut drafts,
            )
            .unwrap();
        assert_eq!(n, 3);
        for &d in &drafts {
            assert!((d as usize) < model.cfg.vocab_size);
        }
    }
}
