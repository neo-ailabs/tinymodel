//! Llama / SmolLM2 / Qwen3 GQA Transformer forward pass.
//!
//! Supports both `llama` (Norm RoPE, optional `rope_freqs.weight`) and
//! `qwen3` (NeoX RoPE + per-head QK-Norm `attn_q_norm` / `attn_k_norm`).

use thiserror::Error;
use tm_core::{DType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool,
    gqa_attention_f16_kv_batch_parallel, gqa_attention_f16_kv_parallel, matmul_f32, matvec_f32,
    matvec_pair_f32, matvec_triple_f32, rms_norm, rms_norm_batch_parallel, rope_compute_cos_sin,
    rope_inplace_tables, swiglu_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("unsupported model architecture '{0}' (supports 'llama', 'qwen3', and 'qwen35')")]
    UnsupportedArch(String),
    #[error("position {pos} exceeds configured max_ctx {max_ctx}")]
    ContextOverflow { pos: usize, max_ctx: usize },
    #[error("token id {token_id} out of vocab range {vocab_size}")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },
}

/// Model hyperparameters parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub vocab_size: usize,
    pub max_ctx_train: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub rope_type: RopeType,
}

/// Zero-copy weight matrix view carrying its own `DType` (per `AGENTS.md` rule #5).
#[derive(Clone, Copy)]
struct MatView<'a> {
    bytes: &'a [u8],
    dtype: DType,
    rows: usize,
    cols: usize,
}

struct LayerWeights<'a> {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wq: MatView<'a>,
    wk: MatView<'a>,
    wv: MatView<'a>,
    wo: MatView<'a>,
    w_gate: MatView<'a>,
    w_up: MatView<'a>,
    w_down: MatView<'a>,
}

/// Loaded Llama / Qwen3 model borrowing quantized weight slices directly from `GgufFile`.
pub struct LlamaModel<'a> {
    pub cfg: LlamaConfig,
    token_embd: MatView<'a>,
    output: MatView<'a>,
    output_norm: Vec<f32>,
    /// Precomputed `freq_base^(-2i / rope_dim) / rope_freqs[i]` for `i in 0..rope_dim/2`.
    rope_inv_freqs: Vec<f64>,
    layers: Vec<LayerWeights<'a>>,
}

/// Pre-allocated per-session inference state (KV cache + scratch buffers).
/// Zero heap allocations occur inside `LlamaModel::forward_token`.
pub struct LlamaState {
    pub max_ctx: usize,
    /// Per-layer F16 K cache: `layers[l]` has `[max_ctx, n_head_kv * head_dim]`.
    k_cache: Vec<Vec<u16>>,
    /// Per-layer F16 V cache: `layers[l]` has `[max_ctx, n_head_kv * head_dim]`.
    v_cache: Vec<Vec<u16>>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    mv_scratch: MatvecScratch,
}

impl LlamaState {
    /// Allocates KV cache and all working buffers up front for `max_ctx` tokens.
    pub fn new(cfg: &LlamaConfig, max_ctx: usize) -> Self {
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let q_dim = cfg.n_head * cfg.head_dim;
        let k_cache = (0..cfg.n_layers)
            .map(|_| vec![0u16; max_ctx * kv_dim])
            .collect();
        let v_cache = (0..cfg.n_layers)
            .map(|_| vec![0u16; max_ctx * kv_dim])
            .collect();
        let max_cols = cfg.n_embd.max(cfg.n_ff).max(q_dim);

        Self {
            max_ctx,
            k_cache,
            v_cache,
            x: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb: vec![0.0; MAX_CPU_BATCH * cfg.n_embd.max(q_dim)],
            xb2: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            q: vec![0.0; MAX_CPU_BATCH * q_dim],
            k: vec![0.0; MAX_CPU_BATCH * kv_dim],
            v: vec![0.0; MAX_CPU_BATCH * kv_dim],
            hb: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            hb2: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            rope_cos: vec![0.0; cfg.rope_dim / 2],
            rope_sin: vec![0.0; cfg.rope_dim / 2],
            scores: vec![0.0; cfg.n_head * max_ctx],
            logits: vec![0.0; cfg.vocab_size],
            mv_scratch: MatvecScratch::new(max_cols),
        }
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

fn load_f32_vec(
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<Vec<f32>, ModelError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[len])?;
    let mut out = vec![0.0f32; len];
    dequantize_row_bytes(info.dtype, bytes, &mut out);
    Ok(out)
}

fn load_mat<'a>(
    gguf: &'a GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<MatView<'a>, ModelError> {
    // GGML stores [cols, rows] (ne0 = cols, ne1 = rows).
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    Ok(MatView {
        bytes,
        dtype: info.dtype,
        rows,
        cols,
    })
}

impl<'a> LlamaModel<'a> {
    /// Loads and strictly validates a `llama` or `qwen3` GGUF model (zero-copy weight views).
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "llama" && arch != "qwen3" {
            return Err(ModelError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        let n_layers = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
        let n_ff = gguf.req_usize(&format!("{pfx}.feed_forward_length"))?;
        let n_head = gguf.req_usize(&format!("{pfx}.attention.head_count"))?;
        let n_head_kv = gguf
            .get_meta(&format!("{pfx}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim = gguf
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head);
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(2048);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "qwen3" { 1_000_000.0 } else { 10000.0 });
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "qwen3" { 1e-6 } else { 1e-5 });
        let rope_type = if pfx == "qwen3" {
            RopeType::Neox
        } else {
            RopeType::Norm
        };

        let embd_info =
            gguf.tensor_info("token_embd.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf.path().display().to_string(),
                    tensor: "token_embd.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let vocab_size = gguf
            .get_meta(&format!("{pfx}.vocab_size"))
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| embd_info.dims.get(1).copied().unwrap_or(0));

        let cfg = LlamaConfig {
            arch,
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            vocab_size,
            max_ctx_train,
            rope_freq_base,
            rms_eps,
            rope_type,
        };

        let token_embd = load_mat(gguf, "token_embd.weight", "global", n_embd, vocab_size)?;
        let output = if gguf.tensor_info("output.weight").is_some() {
            load_mat(gguf, "output.weight", "global", n_embd, vocab_size)?
        } else {
            token_embd
        };
        let output_norm = load_f32_vec(gguf, "output_norm.weight", "global", n_embd)?;
        let rope_freqs = if gguf.tensor_info("rope_freqs.weight").is_some() {
            Some(load_f32_vec(
                gguf,
                "rope_freqs.weight",
                "global",
                rope_dim / 2,
            )?)
        } else {
            None
        };

        // Precompute RoPE inverse frequencies once at load time (eliminates powf() per token).
        let half = rope_dim / 2;
        let base_f = rope_freq_base as f64;
        let dim_f = rope_dim as f64;
        let mut rope_inv_freqs = Vec::with_capacity(half);
        for i in 0..half {
            let mut freq = base_f.powf(-2.0 * (i as f64) / dim_f);
            if let Some(ref factors) = rope_freqs {
                freq /= factors[i] as f64;
            }
            rope_inv_freqs.push(freq);
        }

        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let mut layers = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let qn_name = format!("{ltag}.attn_q_norm.weight");
            let kn_name = format!("{ltag}.attn_k_norm.weight");
            let q_norm = if gguf.tensor_info(&qn_name).is_some() {
                Some(load_f32_vec(gguf, &qn_name, &ltag, head_dim)?)
            } else {
                None
            };
            let k_norm = if gguf.tensor_info(&kn_name).is_some() {
                Some(load_f32_vec(gguf, &kn_name, &ltag, head_dim)?)
            } else {
                None
            };

            layers.push(LayerWeights {
                attn_norm: load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, n_embd)?,
                ffn_norm: load_f32_vec(gguf, &format!("{ltag}.ffn_norm.weight"), &ltag, n_embd)?,
                q_norm,
                k_norm,
                wq: load_mat(gguf, &format!("{ltag}.attn_q.weight"), &ltag, n_embd, q_dim)?,
                wk: load_mat(
                    gguf,
                    &format!("{ltag}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wv: load_mat(
                    gguf,
                    &format!("{ltag}.attn_v.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wo: load_mat(
                    gguf,
                    &format!("{ltag}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                )?,
                w_gate: load_mat(
                    gguf,
                    &format!("{ltag}.ffn_gate.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_up: load_mat(gguf, &format!("{ltag}.ffn_up.weight"), &ltag, n_embd, n_ff)?,
                w_down: load_mat(
                    gguf,
                    &format!("{ltag}.ffn_down.weight"),
                    &ltag,
                    n_ff,
                    n_embd,
                )?,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            output,
            output_norm,
            rope_inv_freqs,
            layers,
        })
    }

    /// Runs a single-token forward pass at sequence position `pos`.
    ///
    /// Performs **zero heap allocations**. If `compute_logits` is `false` (during prompt
    /// prefill for non-final tokens), skips the expensive LM head projection.
    pub fn forward_token(
        &self,
        state: &mut LlamaState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        let tid = token_id as usize;
        if tid >= self.cfg.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }

        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;

        // Precompute RoPE cos/sin once for this position across all layers & heads.
        rope_compute_cos_sin(
            &self.rope_inv_freqs,
            pos,
            &mut state.rope_cos,
            &mut state.rope_sin,
        );

        // 1. Token embedding lookup (zero-copy row slice -> f32)
        let row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        let emb_row = &self.token_embd.bytes[tid * row_bytes..(tid + 1) * row_bytes];
        dequantize_row_bytes(self.token_embd.dtype, emb_row, &mut state.x[..n_embd]);

        // 2. Transformer layers
        for (l, layer) in self.layers.iter().enumerate() {
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
            );

            matvec_triple_f32(
                pool,
                kt,
                layer.wq.dtype,
                layer.wq.bytes,
                layer.wq.rows,
                &mut state.q[..q_dim],
                layer.wk.dtype,
                layer.wk.bytes,
                layer.wk.rows,
                &mut state.k[..kv_dim],
                layer.wv.dtype,
                layer.wv.bytes,
                layer.wv.rows,
                &mut state.v[..kv_dim],
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                n_embd,
            );

            // Optional per-head QK-Norm (Qwen3) applied before RoPE
            if let Some(ref qn) = layer.q_norm {
                for h in 0..self.cfg.n_head {
                    let s = h * self.cfg.head_dim;
                    let e = s + self.cfg.head_dim;
                    rms_norm(&mut state.xb[s..e], &state.q[s..e], qn, self.cfg.rms_eps);
                    state.q[s..e].copy_from_slice(&state.xb[s..e]);
                }
            }
            if let Some(ref kn) = layer.k_norm {
                for h in 0..self.cfg.n_head_kv {
                    let s = h * self.cfg.head_dim;
                    let e = s + self.cfg.head_dim;
                    rms_norm(&mut state.xb[s..e], &state.k[s..e], kn, self.cfg.rms_eps);
                    state.k[s..e].copy_from_slice(&state.xb[s..e]);
                }
            }

            rope_inplace_tables(
                &mut state.q[..q_dim],
                self.cfg.n_head,
                self.cfg.head_dim,
                self.cfg.rope_dim,
                &state.rope_cos,
                &state.rope_sin,
                self.cfg.rope_type,
            );
            rope_inplace_tables(
                &mut state.k[..kv_dim],
                self.cfg.n_head_kv,
                self.cfg.head_dim,
                self.cfg.rope_dim,
                &state.rope_cos,
                &state.rope_sin,
                self.cfg.rope_type,
            );

            // Store K, V into F16 KV cache at position `pos` via ISA-dispatched `fcvtn`
            let kv_off = pos * kv_dim;
            (kt.f32_to_f16_slice)(
                &state.k[..kv_dim],
                &mut state.k_cache[l][kv_off..kv_off + kv_dim],
            );
            (kt.f32_to_f16_slice)(
                &state.v[..kv_dim],
                &mut state.v_cache[l][kv_off..kv_off + kv_dim],
            );

            // GQA attention over 0..=pos (hardware NEON `fcvt` + parallel across heads for long ctx)
            gqa_attention_f16_kv_parallel(
                pool,
                kt,
                &mut state.xb[..q_dim],
                &state.q[..q_dim],
                &state.k_cache[l],
                &state.v_cache[l],
                self.cfg.n_head,
                self.cfg.n_head_kv,
                self.cfg.head_dim,
                pos + 1,
                state.max_ctx,
                &mut state.scores,
            );

            // Output projection + residual
            matvec_f32(
                pool,
                kt,
                layer.wo.dtype,
                layer.wo.bytes,
                &state.xb[..q_dim],
                &mut state.mv_scratch,
                &mut state.xb2[..n_embd],
                layer.wo.rows,
                layer.wo.cols,
            );
            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
            }

            // FFN (SwiGLU)
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.ffn_norm,
                self.cfg.rms_eps,
            );
            matvec_pair_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                layer.w_gate.rows,
                &mut state.hb[..n_ff],
                layer.w_up.dtype,
                layer.w_up.bytes,
                layer.w_up.rows,
                &mut state.hb2[..n_ff],
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                n_embd,
            );
            swiglu_inplace_parallel(pool, &mut state.hb[..n_ff], &state.hb2[..n_ff]);
            matvec_f32(
                pool,
                kt,
                layer.w_down.dtype,
                layer.w_down.bytes,
                &state.hb[..n_ff],
                &mut state.mv_scratch,
                &mut state.xb2[..n_embd],
                layer.w_down.rows,
                layer.w_down.cols,
            );
            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
            }
        }

        // 3. Final norm + LM head
        if compute_logits {
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &self.output_norm,
                self.cfg.rms_eps,
            );
            matvec_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes,
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                &mut state.logits,
                self.output.rows,
                self.output.cols,
            );
        }

        Ok(())
    }

    /// Runs a batched prompt prefill on CPU in chunks of up to `MAX_CPU_BATCH` tokens,
    /// reusing L2-cached weight tiles and `i8mm`/`dotprod`/`neon` GEMM microkernels.
    ///
    /// Performs **zero heap allocations** and preserves exact batch-width invariance.
    pub fn forward_batch(
        &self,
        state: &mut LlamaState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        self.forward_batch_inner(
            state,
            pool,
            kt,
            token_ids,
            start_pos,
            compute_last_logits,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch_profiled(
        &self,
        state: &mut LlamaState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        prof: &mut CpuDecodeProfile,
    ) -> Result<(), ModelError> {
        self.forward_batch_inner(
            state,
            pool,
            kt,
            token_ids,
            start_pos,
            compute_last_logits,
            Some(prof),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_inner(
        &self,
        state: &mut LlamaState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        mut prof: Option<&mut CpuDecodeProfile>,
    ) -> Result<(), ModelError> {
        use std::time::Instant;

        if token_ids.is_empty() {
            return Ok(());
        }
        let end_pos = start_pos + token_ids.len() - 1;
        if end_pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: end_pos,
                max_ctx: state.max_ctx,
            });
        }
        for &tid in token_ids {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }
        if token_ids.len() == 1 {
            if let Some(p) = prof {
                return self.forward_token_profiled(
                    state,
                    pool,
                    kt,
                    token_ids[0],
                    start_pos,
                    compute_last_logits,
                    p,
                );
            }
            return self.forward_token(
                state,
                pool,
                kt,
                token_ids[0],
                start_pos,
                compute_last_logits,
            );
        }

        let t_total = prof.as_ref().map(|_| Instant::now());
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let row_bytes = self.token_embd.dtype.row_bytes(n_embd);

        let total = token_ids.len();
        let mut offset = 0usize;
        while offset < total {
            let b = (total - offset).min(MAX_CPU_BATCH);
            let chunk = &token_ids[offset..offset + b];
            let chunk_pos = start_pos + offset;
            let is_last = offset + b == total;

            // 1. Batched embedding lookup
            let t0 = prof.as_ref().map(|_| Instant::now());
            for (t, &tok) in chunk.iter().enumerate() {
                let tid = tok as usize;
                let emb_row = &self.token_embd.bytes[tid * row_bytes..(tid + 1) * row_bytes];
                dequantize_row_bytes(
                    self.token_embd.dtype,
                    emb_row,
                    &mut state.x[t * n_embd..(t + 1) * n_embd],
                );
            }
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                p.embed_ns += ts.elapsed().as_nanos() as u64;
            }

            // 2. Transformer layers with multi-token GEMM
            for (l, layer) in self.layers.iter().enumerate() {
                let t0 = prof.as_ref().map(|_| Instant::now());
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..b * n_embd],
                    &state.x[..b * n_embd],
                    &layer.attn_norm,
                    self.cfg.rms_eps,
                    n_embd,
                    b,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.attn_norm_ns += ts.elapsed().as_nanos() as u64;
                }

                let t0 = prof.as_ref().map(|_| Instant::now());
                matmul_f32(
                    pool,
                    kt,
                    layer.wq.dtype,
                    layer.wq.bytes,
                    &state.xb[..b * n_embd],
                    &mut state.mv_scratch,
                    &mut state.q[..b * q_dim],
                    q_dim,
                    n_embd,
                    b,
                );
                matmul_f32(
                    pool,
                    kt,
                    layer.wk.dtype,
                    layer.wk.bytes,
                    &state.xb[..b * n_embd],
                    &mut state.mv_scratch,
                    &mut state.k[..b * kv_dim],
                    kv_dim,
                    n_embd,
                    b,
                );
                matmul_f32(
                    pool,
                    kt,
                    layer.wv.dtype,
                    layer.wv.bytes,
                    &state.xb[..b * n_embd],
                    &mut state.mv_scratch,
                    &mut state.v[..b * kv_dim],
                    kv_dim,
                    n_embd,
                    b,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.attn_qkv_ns += ts.elapsed().as_nanos() as u64;
                }

                // Per-token QK-Norm, RoPE, and F16 KV store (so all `b` keys/values are in cache)
                let t0 = prof.as_ref().map(|_| Instant::now());
                for t in 0..b {
                    let pos = chunk_pos + t;
                    let qt = &mut state.q[t * q_dim..(t + 1) * q_dim];
                    let kt_vec = &mut state.k[t * kv_dim..(t + 1) * kv_dim];
                    let vt_vec = &state.v[t * kv_dim..(t + 1) * kv_dim];

                    if let Some(ref qn) = layer.q_norm {
                        for h in 0..self.cfg.n_head {
                            let s = h * self.cfg.head_dim;
                            let e = s + self.cfg.head_dim;
                            rms_norm(&mut state.xb[s..e], &qt[s..e], qn, self.cfg.rms_eps);
                            qt[s..e].copy_from_slice(&state.xb[s..e]);
                        }
                    }
                    if let Some(ref kn) = layer.k_norm {
                        for h in 0..self.cfg.n_head_kv {
                            let s = h * self.cfg.head_dim;
                            let e = s + self.cfg.head_dim;
                            rms_norm(&mut state.xb[s..e], &kt_vec[s..e], kn, self.cfg.rms_eps);
                            kt_vec[s..e].copy_from_slice(&state.xb[s..e]);
                        }
                    }

                    rope_compute_cos_sin(
                        &self.rope_inv_freqs,
                        pos,
                        &mut state.rope_cos,
                        &mut state.rope_sin,
                    );
                    rope_inplace_tables(
                        qt,
                        self.cfg.n_head,
                        self.cfg.head_dim,
                        self.cfg.rope_dim,
                        &state.rope_cos,
                        &state.rope_sin,
                        self.cfg.rope_type,
                    );
                    rope_inplace_tables(
                        kt_vec,
                        self.cfg.n_head_kv,
                        self.cfg.head_dim,
                        self.cfg.rope_dim,
                        &state.rope_cos,
                        &state.rope_sin,
                        self.cfg.rope_type,
                    );

                    let kv_off = pos * kv_dim;
                    (kt.f32_to_f16_slice)(kt_vec, &mut state.k_cache[l][kv_off..kv_off + kv_dim]);
                    (kt.f32_to_f16_slice)(vt_vec, &mut state.v_cache[l][kv_off..kv_off + kv_dim]);
                }
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.rope_kv_ns += ts.elapsed().as_nanos() as u64;
                }

                // Causal GQA attention for all `b` tokens in a single parallel dispatch
                let t0 = prof.as_ref().map(|_| Instant::now());
                gqa_attention_f16_kv_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..b * q_dim],
                    &state.q[..b * q_dim],
                    &state.k_cache[l],
                    &state.v_cache[l],
                    self.cfg.n_head,
                    self.cfg.n_head_kv,
                    self.cfg.head_dim,
                    chunk_pos,
                    b,
                    state.max_ctx,
                    &mut state.scores,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.gqa_attn_ns += ts.elapsed().as_nanos() as u64;
                }

                // Batched Wo + residual
                let t0 = prof.as_ref().map(|_| Instant::now());
                matmul_f32(
                    pool,
                    kt,
                    layer.wo.dtype,
                    layer.wo.bytes,
                    &state.xb[..b * q_dim],
                    &mut state.mv_scratch,
                    &mut state.xb2[..b * n_embd],
                    n_embd,
                    q_dim,
                    b,
                );
                for i in 0..(b * n_embd) {
                    state.x[i] += state.xb2[i];
                }
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.attn_wo_ns += ts.elapsed().as_nanos() as u64;
                }

                // Batched FFN
                let t0 = prof.as_ref().map(|_| Instant::now());
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..b * n_embd],
                    &state.x[..b * n_embd],
                    &layer.ffn_norm,
                    self.cfg.rms_eps,
                    n_embd,
                    b,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.ffn_norm_ns += ts.elapsed().as_nanos() as u64;
                }

                let t0 = prof.as_ref().map(|_| Instant::now());
                matmul_f32(
                    pool,
                    kt,
                    layer.w_gate.dtype,
                    layer.w_gate.bytes,
                    &state.xb[..b * n_embd],
                    &mut state.mv_scratch,
                    &mut state.hb[..b * n_ff],
                    n_ff,
                    n_embd,
                    b,
                );
                matmul_f32(
                    pool,
                    kt,
                    layer.w_up.dtype,
                    layer.w_up.bytes,
                    &state.xb[..b * n_embd],
                    &mut state.mv_scratch,
                    &mut state.hb2[..b * n_ff],
                    n_ff,
                    n_embd,
                    b,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.ffn_gate_up_ns += ts.elapsed().as_nanos() as u64;
                }

                let t0 = prof.as_ref().map(|_| Instant::now());
                swiglu_inplace_parallel(pool, &mut state.hb[..b * n_ff], &state.hb2[..b * n_ff]);
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.swiglu_ns += ts.elapsed().as_nanos() as u64;
                }

                let t0 = prof.as_ref().map(|_| Instant::now());
                matmul_f32(
                    pool,
                    kt,
                    layer.w_down.dtype,
                    layer.w_down.bytes,
                    &state.hb[..b * n_ff],
                    &mut state.mv_scratch,
                    &mut state.xb2[..b * n_embd],
                    n_embd,
                    n_ff,
                    b,
                );
                for i in 0..(b * n_embd) {
                    state.x[i] += state.xb2[i];
                }
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.ffn_down_ns += ts.elapsed().as_nanos() as u64;
                }
            }

            if is_last && compute_last_logits {
                let t0 = prof.as_ref().map(|_| Instant::now());
                let last_x = &state.x[(b - 1) * n_embd..b * n_embd];
                rms_norm(
                    &mut state.xb[..n_embd],
                    last_x,
                    &self.output_norm,
                    self.cfg.rms_eps,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.output_norm_ns += ts.elapsed().as_nanos() as u64;
                }
                let t0 = prof.as_ref().map(|_| Instant::now());
                matvec_f32(
                    pool,
                    kt,
                    self.output.dtype,
                    self.output.bytes,
                    &state.xb[..n_embd],
                    &mut state.mv_scratch,
                    &mut state.logits,
                    self.output.rows,
                    self.output.cols,
                );
                if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t0) {
                    p.lm_head_ns += ts.elapsed().as_nanos() as u64;
                }
            }

            offset += b;
        }

        if let (Some(p), Some(tt)) = (prof, t_total) {
            p.runs += total;
            p.total_ns += tt.elapsed().as_nanos() as u64;
        }

        Ok(())
    }

    /// Instrumented single-token forward pass that accumulates nanosecond timings and
    /// exact byte counts per operator stage into `prof` (zero heap allocations).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_profiled(
        &self,
        state: &mut LlamaState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        prof: &mut CpuDecodeProfile,
    ) -> Result<(), ModelError> {
        use std::time::Instant;

        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        let tid = token_id as usize;
        if tid >= self.cfg.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }

        let t_total = Instant::now();
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;

        let t0 = Instant::now();
        rope_compute_cos_sin(
            &self.rope_inv_freqs,
            pos,
            &mut state.rope_cos,
            &mut state.rope_sin,
        );
        let row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        let emb_row = &self.token_embd.bytes[tid * row_bytes..(tid + 1) * row_bytes];
        dequantize_row_bytes(self.token_embd.dtype, emb_row, &mut state.x[..n_embd]);
        prof.embed_ns += t0.elapsed().as_nanos() as u64;
        prof.embed_bytes += (row_bytes + n_embd * 4) as u64;

        for (l, layer) in self.layers.iter().enumerate() {
            let t0 = Instant::now();
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
            );
            prof.attn_norm_ns += t0.elapsed().as_nanos() as u64;

            let t0 = Instant::now();
            matvec_triple_f32(
                pool,
                kt,
                layer.wq.dtype,
                layer.wq.bytes,
                layer.wq.rows,
                &mut state.q[..q_dim],
                layer.wk.dtype,
                layer.wk.bytes,
                layer.wk.rows,
                &mut state.k[..kv_dim],
                layer.wv.dtype,
                layer.wv.bytes,
                layer.wv.rows,
                &mut state.v[..kv_dim],
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                n_embd,
            );
            prof.attn_qkv_ns += t0.elapsed().as_nanos() as u64;
            prof.attn_qkv_bytes +=
                (layer.wq.bytes.len() + layer.wk.bytes.len() + layer.wv.bytes.len()) as u64;

            if layer.q_norm.is_some() || layer.k_norm.is_some() {
                let t0 = Instant::now();
                if let Some(ref qn) = layer.q_norm {
                    for h in 0..self.cfg.n_head {
                        let s = h * self.cfg.head_dim;
                        let e = s + self.cfg.head_dim;
                        rms_norm(&mut state.xb[s..e], &state.q[s..e], qn, self.cfg.rms_eps);
                        state.q[s..e].copy_from_slice(&state.xb[s..e]);
                    }
                }
                if let Some(ref kn) = layer.k_norm {
                    for h in 0..self.cfg.n_head_kv {
                        let s = h * self.cfg.head_dim;
                        let e = s + self.cfg.head_dim;
                        rms_norm(&mut state.xb[s..e], &state.k[s..e], kn, self.cfg.rms_eps);
                        state.k[s..e].copy_from_slice(&state.xb[s..e]);
                    }
                }
                prof.qk_norm_ns += t0.elapsed().as_nanos() as u64;
            }

            let t0 = Instant::now();
            rope_inplace_tables(
                &mut state.q[..q_dim],
                self.cfg.n_head,
                self.cfg.head_dim,
                self.cfg.rope_dim,
                &state.rope_cos,
                &state.rope_sin,
                self.cfg.rope_type,
            );
            rope_inplace_tables(
                &mut state.k[..kv_dim],
                self.cfg.n_head_kv,
                self.cfg.head_dim,
                self.cfg.rope_dim,
                &state.rope_cos,
                &state.rope_sin,
                self.cfg.rope_type,
            );
            let kv_off = pos * kv_dim;
            (kt.f32_to_f16_slice)(
                &state.k[..kv_dim],
                &mut state.k_cache[l][kv_off..kv_off + kv_dim],
            );
            (kt.f32_to_f16_slice)(
                &state.v[..kv_dim],
                &mut state.v_cache[l][kv_off..kv_off + kv_dim],
            );
            prof.rope_kv_ns += t0.elapsed().as_nanos() as u64;

            let t0 = Instant::now();
            gqa_attention_f16_kv_parallel(
                pool,
                kt,
                &mut state.xb[..q_dim],
                &state.q[..q_dim],
                &state.k_cache[l],
                &state.v_cache[l],
                self.cfg.n_head,
                self.cfg.n_head_kv,
                self.cfg.head_dim,
                pos + 1,
                state.max_ctx,
                &mut state.scores,
            );
            prof.gqa_attn_ns += t0.elapsed().as_nanos() as u64;
            prof.gqa_kv_bytes += ((pos + 1) * kv_dim * 2 * 2) as u64;

            let t0 = Instant::now();
            matvec_f32(
                pool,
                kt,
                layer.wo.dtype,
                layer.wo.bytes,
                &state.xb[..q_dim],
                &mut state.mv_scratch,
                &mut state.xb2[..n_embd],
                layer.wo.rows,
                layer.wo.cols,
            );
            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
            }
            prof.attn_wo_ns += t0.elapsed().as_nanos() as u64;
            prof.attn_wo_bytes += layer.wo.bytes.len() as u64;

            let t0 = Instant::now();
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.ffn_norm,
                self.cfg.rms_eps,
            );
            prof.ffn_norm_ns += t0.elapsed().as_nanos() as u64;

            let t0 = Instant::now();
            matvec_pair_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                layer.w_gate.rows,
                &mut state.hb[..n_ff],
                layer.w_up.dtype,
                layer.w_up.bytes,
                layer.w_up.rows,
                &mut state.hb2[..n_ff],
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                n_embd,
            );
            prof.ffn_gate_up_ns += t0.elapsed().as_nanos() as u64;
            prof.ffn_gate_up_bytes += (layer.w_gate.bytes.len() + layer.w_up.bytes.len()) as u64;

            let t0 = Instant::now();
            swiglu_inplace_parallel(pool, &mut state.hb[..n_ff], &state.hb2[..n_ff]);
            prof.swiglu_ns += t0.elapsed().as_nanos() as u64;

            let t0 = Instant::now();
            matvec_f32(
                pool,
                kt,
                layer.w_down.dtype,
                layer.w_down.bytes,
                &state.hb[..n_ff],
                &mut state.mv_scratch,
                &mut state.xb2[..n_embd],
                layer.w_down.rows,
                layer.w_down.cols,
            );
            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
            }
            prof.ffn_down_ns += t0.elapsed().as_nanos() as u64;
            prof.ffn_down_bytes += layer.w_down.bytes.len() as u64;
        }

        if compute_logits {
            let t0 = Instant::now();
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &self.output_norm,
                self.cfg.rms_eps,
            );
            prof.output_norm_ns += t0.elapsed().as_nanos() as u64;

            let t0 = Instant::now();
            matvec_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes,
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                &mut state.logits,
                self.output.rows,
                self.output.cols,
            );
            prof.lm_head_ns += t0.elapsed().as_nanos() as u64;
            prof.lm_head_bytes += self.output.bytes.len() as u64;
        }

        prof.runs += 1;
        prof.total_ns += t_total.elapsed().as_nanos() as u64;
        Ok(())
    }

    pub fn lm_head_dtype(&self) -> DType {
        self.output.dtype
    }
}

/// Cumulative CPU operator profiling counters across `runs` token steps.
#[derive(Debug, Clone, Default)]
pub struct CpuDecodeProfile {
    pub runs: usize,
    pub total_ns: u64,
    pub embed_ns: u64,
    pub attn_norm_ns: u64,
    pub attn_qkv_ns: u64,
    pub qk_norm_ns: u64,
    pub rope_kv_ns: u64,
    pub gqa_attn_ns: u64,
    pub attn_wo_ns: u64,
    pub ffn_norm_ns: u64,
    pub ffn_gate_up_ns: u64,
    pub swiglu_ns: u64,
    pub ffn_down_ns: u64,
    pub output_norm_ns: u64,
    pub lm_head_ns: u64,
    pub embed_bytes: u64,
    pub attn_qkv_bytes: u64,
    pub gqa_kv_bytes: u64,
    pub attn_wo_bytes: u64,
    pub ffn_gate_up_bytes: u64,
    pub ffn_down_bytes: u64,
    pub lm_head_bytes: u64,
}
