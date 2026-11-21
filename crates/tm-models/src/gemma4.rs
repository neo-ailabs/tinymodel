//! Gemma 4 (`general.architecture == "gemma4"`) forward pass with Per-Layer Embeddings (PLE),
//! interleaved Sliding-Window Attention (SWA) + Full Proportional-RoPE Attention,
//! cross-layer KV-cache sharing (`shared_kv_layers`), variable-width GeGLU FFN,
//! unit-weight V RMSNorm, and final logit soft-capping.
//!
//! Reference: `third_party/llama.cpp/src/models/gemma4.cpp` (MIT License).
//! Strictly obeys `AGENTS.md`:
//! - `#![forbid(unsafe_code)]` (inherited from crate root);
//! - Zero heap allocations on `forward_token` / `forward_batch`;
//! - On-demand per-row PLE lookup directly from `mmap` (`per_layer_token_embd.weight` 1.32 GB
//!   table is NEVER scanned or unpacked upfront — only active token row(s) are read);
//! - State safety (`Rule #6`): validates all context bounds and token IDs BEFORE mutating `LayerState`.

use tm_core::{DType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool, geglu_inplace_parallel,
    gelu_tanh_f32, gqa_attention_f16_kv_ext_batch_parallel, gqa_attention_f16_kv_ext_parallel,
    matmul_f32, matvec_f32, matvec_pair_f32, matvec_triple_f32, rms_norm, rms_norm_batch_parallel,
    rms_norm_unit_inplace, rope_compute_cos_sin, rope_inplace_tables, softcap_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};

use crate::llama::ModelError;
use crate::state::LayerState;

/// Hyperparameters for `general.architecture == "gemma4"`.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff_per_layer: Vec<usize>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim_full: usize,
    pub head_dim_swa: usize,
    pub rope_dim_full: usize,
    pub rope_dim_swa: usize,
    pub freq_base_full: f32,
    pub freq_base_swa: f32,
    pub n_swa: usize,
    pub is_swa: Vec<bool>,
    pub n_layer_kv_from_start: usize,
    pub kv_source: Vec<usize>,
    pub n_embd_per_layer: usize,
    pub vocab_size: usize,
    pub max_ctx_train: usize,
    pub rms_eps: f32,
    pub final_logit_softcapping: Option<f32>,
}

impl Gemma4Config {
    #[inline]
    pub fn head_dim(&self, il: usize) -> usize {
        if self.is_swa[il] {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    #[inline]
    pub fn rope_dim(&self, il: usize) -> usize {
        if self.is_swa[il] {
            self.rope_dim_swa
        } else {
            self.rope_dim_full
        }
    }

    #[inline]
    pub fn has_kv(&self, il: usize) -> bool {
        il < self.n_layer_kv_from_start
    }

    #[inline]
    pub fn max_n_ff(&self) -> usize {
        self.n_ff_per_layer.iter().copied().max().unwrap_or(0)
    }

    #[inline]
    pub fn max_head_dim(&self) -> usize {
        self.head_dim_full.max(self.head_dim_swa)
    }
}

#[derive(Clone, Copy)]
struct MatView<'a> {
    bytes: &'a [u8],
    dtype: DType,
    rows: usize,
    cols: usize,
}

struct PleGlobalWeights<'a> {
    tok_embd: MatView<'a>,
    model_proj: MatView<'a>,
    /// Pre-scaled by `1 / sqrt(2)` at load time (`gemma4.cpp:543`).
    proj_norm_scaled: Vec<f32>,
}

struct PleLayerWeights<'a> {
    inp_gate: MatView<'a>,
    proj: MatView<'a>,
    post_norm: Vec<f32>,
}

struct KvLayerWeights<'a> {
    wk: MatView<'a>,
    wv: Option<MatView<'a>>,
    k_norm: Vec<f32>,
}

struct Gemma4LayerWeights<'a> {
    attn_norm: Vec<f32>,
    wq: MatView<'a>,
    q_norm: Vec<f32>,
    kv: Option<KvLayerWeights<'a>>,
    wo: MatView<'a>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: MatView<'a>,
    w_up: MatView<'a>,
    w_down: MatView<'a>,
    post_ffw_norm: Vec<f32>,
    ple: Option<PleLayerWeights<'a>>,
    out_scale: Option<f32>,
}

/// Loaded Gemma 4 model borrowing quantized weight slices zero-copy from `GgufFile`.
pub struct Gemma4Model<'a> {
    pub cfg: Gemma4Config,
    token_embd: MatView<'a>,
    output: MatView<'a>,
    output_norm: Vec<f32>,
    rope_inv_freqs_full: Vec<f64>,
    rope_inv_freqs_swa: Vec<f64>,
    ple_global: Option<PleGlobalWeights<'a>>,
    layers: Vec<Gemma4LayerWeights<'a>>,
}

/// Pre-allocated inference state for Gemma 4 (`LayerState::Kv` on own-KV layers,
/// `LayerState::SharedKv` on KV-shared layers — zero duplicate KV allocation).
pub struct Gemma4State {
    pub max_ctx: usize,
    pub layers: Vec<LayerState>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    attn_res: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    /// Per-token PLE inputs `[MAX_CPU_BATCH, n_layers * n_embd_per_layer]`.
    ple_inputs: Vec<f32>,
    ple_proj_scratch: Vec<f32>,
    ple_gate_scratch: Vec<f32>,
    rope_cos_full: Vec<f32>,
    rope_sin_full: Vec<f32>,
    rope_cos_swa: Vec<f32>,
    rope_sin_swa: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    mv_scratch: MatvecScratch,
}

impl Gemma4State {
    pub fn new(model: &Gemma4Model<'_>, max_ctx: usize) -> Self {
        let cfg = &model.cfg;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let hd = cfg.head_dim(il);
            let kv_dim = cfg.n_head_kv * hd;
            let window = if cfg.is_swa[il] {
                Some(cfg.n_swa)
            } else {
                None
            };
            if cfg.has_kv(il) {
                layers.push(LayerState::Kv {
                    k_cache: vec![0u16; max_ctx * kv_dim],
                    v_cache: vec![0u16; max_ctx * kv_dim],
                    kv_dim,
                    window,
                });
            } else {
                layers.push(LayerState::SharedKv {
                    source_layer: cfg.kv_source[il],
                    window,
                });
            }
        }

        let max_hd = cfg.max_head_dim();
        let max_q_dim = cfg.n_head * max_hd;
        let max_kv_dim = cfg.n_head_kv * max_hd;
        let max_n_ff = cfg.max_n_ff();
        let ple_total = cfg.n_layers * cfg.n_embd_per_layer;
        let max_cols = cfg
            .n_embd
            .max(max_n_ff)
            .max(max_q_dim)
            .max(cfg.n_embd_per_layer);

        Self {
            max_ctx,
            layers,
            x: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb2: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            attn_res: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            q_buf: vec![0.0; MAX_CPU_BATCH * max_q_dim],
            k_buf: vec![0.0; MAX_CPU_BATCH * max_kv_dim],
            v_buf: vec![0.0; MAX_CPU_BATCH * max_kv_dim],
            attn_out: vec![0.0; MAX_CPU_BATCH * max_q_dim],
            hb: vec![0.0; MAX_CPU_BATCH * max_n_ff],
            hb2: vec![0.0; MAX_CPU_BATCH * max_n_ff],
            ple_inputs: vec![0.0; MAX_CPU_BATCH * ple_total.max(1)],
            ple_proj_scratch: vec![0.0; MAX_CPU_BATCH * ple_total.max(1)],
            ple_gate_scratch: vec![0.0; MAX_CPU_BATCH * cfg.n_embd_per_layer.max(1)],
            rope_cos_full: vec![0.0; cfg.rope_dim_full / 2],
            rope_sin_full: vec![0.0; cfg.rope_dim_full / 2],
            rope_cos_swa: vec![0.0; cfg.rope_dim_swa / 2],
            rope_sin_swa: vec![0.0; cfg.rope_dim_swa / 2],
            scores: vec![0.0; cfg.n_head * max_ctx],
            logits: vec![0.0; cfg.vocab_size],
            mv_scratch: MatvecScratch::new(max_cols),
        }
    }

    pub fn reset(&mut self) {
        for ls in &mut self.layers {
            ls.reset();
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
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    Ok(MatView {
        bytes,
        dtype: info.dtype,
        rows,
        cols,
    })
}

impl<'a> Gemma4Model<'a> {
    /// Loads and validates a `gemma4` GGUF model.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "gemma4" {
            return Err(ModelError::UnsupportedArch(arch));
        }

        let pfx = "gemma4";
        let n_layers = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;

        let ff_key = format!("{pfx}.feed_forward_length");
        let n_ff_per_layer = if let Some(mv) = gguf.get_meta(&ff_key) {
            if let Some(arr) = mv.as_usize_array() {
                if arr.len() != n_layers {
                    return Err(GgufError::MetadataTypeMismatch {
                        path: gguf.path().display().to_string(),
                        key: ff_key,
                        expected: "usize array matching block_count",
                    }
                    .into());
                }
                arr
            } else if let Some(single) = mv.as_usize() {
                vec![single; n_layers]
            } else {
                return Err(GgufError::MetadataTypeMismatch {
                    path: gguf.path().display().to_string(),
                    key: ff_key,
                    expected: "usize or usize[]",
                }
                .into());
            }
        } else {
            return Err(GgufError::MissingMetadata {
                path: gguf.path().display().to_string(),
                key: ff_key,
            }
            .into());
        };

        let n_head = gguf.req_usize(&format!("{pfx}.attention.head_count"))?;
        let n_head_kv = gguf
            .get_meta(&format!("{pfx}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim_full = gguf
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head);
        let head_dim_swa = gguf
            .get_meta(&format!("{pfx}.attention.key_length_swa"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_full);
        let rope_dim_full = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_full);
        let rope_dim_swa = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count_swa"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_swa);
        let freq_base_full = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let freq_base_swa = gguf
            .get_meta(&format!("{pfx}.rope.freq_base_swa"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);
        let n_swa = gguf.req_usize(&format!("{pfx}.attention.sliding_window"))?;

        let swa_pat_key = format!("{pfx}.attention.sliding_window_pattern");
        let is_swa = gguf
            .get_meta(&swa_pat_key)
            .and_then(|v| v.as_bool_array())
            .filter(|v| v.len() == n_layers)
            .ok_or_else(|| GgufError::MissingMetadata {
                path: gguf.path().display().to_string(),
                key: swa_pat_key,
            })?;

        let shared_kv_layers = gguf
            .get_meta(&format!("{pfx}.attention.shared_kv_layers"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let n_layer_kv_from_start = n_layers.saturating_sub(shared_kv_layers);

        let mut kv_source = vec![0usize; n_layers];
        for il in 0..n_layers {
            if il < n_layer_kv_from_start {
                kv_source[il] = il;
            } else {
                // Match `gemma4.cpp:17`: last layer `< n_layer_kv_from_start` with same SWA type
                let mut src = n_layer_kv_from_start.saturating_sub(1);
                for cand in (0..n_layer_kv_from_start).rev() {
                    if is_swa[cand] == is_swa[il] {
                        src = cand;
                        break;
                    }
                }
                kv_source[il] = src;
            }
        }

        let n_embd_per_layer = gguf
            .get_meta(&format!("{pfx}.embedding_length_per_layer_input"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(8192);
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let final_logit_softcapping = gguf
            .get_meta(&format!("{pfx}.final_logit_softcapping"))
            .and_then(|v| v.as_f32())
            .filter(|&c| c > 0.0);

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

        let token_embd = load_mat(gguf, "token_embd.weight", "global", n_embd, vocab_size)?;
        let output = if gguf.tensor_info("output.weight").is_some() {
            load_mat(gguf, "output.weight", "global", n_embd, vocab_size)?
        } else {
            token_embd
        };
        let output_norm = load_f32_vec(gguf, "output_norm.weight", "global", n_embd)?;

        let rope_freqs_full = if gguf.tensor_info("rope_freqs.weight").is_some() {
            Some(load_f32_vec(
                gguf,
                "rope_freqs.weight",
                "global",
                rope_dim_full / 2,
            )?)
        } else {
            None
        };

        let rope_inv_freqs_full: Vec<f64> = (0..(rope_dim_full / 2))
            .map(|i| {
                let mut f =
                    (freq_base_full as f64).powf(-2.0 * (i as f64) / (rope_dim_full as f64));
                if let Some(ref factors) = rope_freqs_full {
                    f /= factors[i] as f64;
                }
                f
            })
            .collect();

        let rope_inv_freqs_swa: Vec<f64> = (0..(rope_dim_swa / 2))
            .map(|i| (freq_base_swa as f64).powf(-2.0 * (i as f64) / (rope_dim_swa as f64)))
            .collect();

        let ple_global = if n_embd_per_layer > 0 {
            let ple_total = n_embd_per_layer * n_layers;
            let tok_embd = load_mat(
                gguf,
                "per_layer_token_embd.weight",
                "global",
                ple_total,
                vocab_size,
            )?;
            let model_proj = load_mat(
                gguf,
                "per_layer_model_proj.weight",
                "global",
                n_embd,
                ple_total,
            )?;
            let mut proj_norm_scaled = load_f32_vec(
                gguf,
                "per_layer_proj_norm.weight",
                "global",
                n_embd_per_layer,
            )?;
            let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
            for v in &mut proj_norm_scaled {
                *v *= inv_sqrt2;
            }
            Some(PleGlobalWeights {
                tok_embd,
                model_proj,
                proj_norm_scaled,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let ltag = format!("blk.{il}");
            let hd = if is_swa[il] {
                head_dim_swa
            } else {
                head_dim_full
            };
            let q_dim = n_head * hd;
            let kv_dim = n_head_kv * hd;
            let n_ff = n_ff_per_layer[il];

            let kv = if il < n_layer_kv_from_start {
                let wk = load_mat(
                    gguf,
                    &format!("{ltag}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?;
                let wv = if gguf.tensor_info(&format!("{ltag}.attn_v.weight")).is_some() {
                    Some(load_mat(
                        gguf,
                        &format!("{ltag}.attn_v.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?)
                } else {
                    None
                };
                let k_norm = load_f32_vec(gguf, &format!("{ltag}.attn_k_norm.weight"), &ltag, hd)?;
                Some(KvLayerWeights { wk, wv, k_norm })
            } else {
                None
            };

            let ple = if n_embd_per_layer > 0 {
                Some(PleLayerWeights {
                    inp_gate: load_mat(
                        gguf,
                        &format!("{ltag}.inp_gate.weight"),
                        &ltag,
                        n_embd,
                        n_embd_per_layer,
                    )?,
                    proj: load_mat(
                        gguf,
                        &format!("{ltag}.proj.weight"),
                        &ltag,
                        n_embd_per_layer,
                        n_embd,
                    )?,
                    post_norm: load_f32_vec(
                        gguf,
                        &format!("{ltag}.post_norm.weight"),
                        &ltag,
                        n_embd,
                    )?,
                })
            } else {
                None
            };

            let out_scale = if gguf
                .tensor_info(&format!("{ltag}.layer_output_scale.weight"))
                .is_some()
            {
                let v = load_f32_vec(gguf, &format!("{ltag}.layer_output_scale.weight"), &ltag, 1)?;
                Some(v[0])
            } else {
                None
            };

            layers.push(Gemma4LayerWeights {
                attn_norm: load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, n_embd)?,
                wq: load_mat(gguf, &format!("{ltag}.attn_q.weight"), &ltag, n_embd, q_dim)?,
                q_norm: load_f32_vec(gguf, &format!("{ltag}.attn_q_norm.weight"), &ltag, hd)?,
                kv,
                wo: load_mat(
                    gguf,
                    &format!("{ltag}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                )?,
                post_attn_norm: load_f32_vec(
                    gguf,
                    &format!("{ltag}.post_attention_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                ffn_norm: load_f32_vec(gguf, &format!("{ltag}.ffn_norm.weight"), &ltag, n_embd)?,
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
                post_ffw_norm: load_f32_vec(
                    gguf,
                    &format!("{ltag}.post_ffw_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                ple,
                out_scale,
            });
        }

        let cfg = Gemma4Config {
            arch,
            n_layers,
            n_embd,
            n_ff_per_layer,
            n_head,
            n_head_kv,
            head_dim_full,
            head_dim_swa,
            rope_dim_full,
            rope_dim_swa,
            freq_base_full,
            freq_base_swa,
            n_swa,
            is_swa,
            n_layer_kv_from_start,
            kv_source,
            n_embd_per_layer,
            vocab_size,
            max_ctx_train,
            rms_eps,
            final_logit_softcapping,
        };

        Ok(Self {
            cfg,
            token_embd,
            output,
            output_norm,
            rope_inv_freqs_full,
            rope_inv_freqs_swa,
            ple_global,
            layers,
        })
    }

    /// Single-token forward pass. Validates bounds BEFORE mutating any `LayerState`.
    pub fn forward_token(
        &self,
        state: &mut Gemma4State,
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
        let ple_dim = self.cfg.n_embd_per_layer;
        let ple_total = self.cfg.n_layers * ple_dim;

        // 1.Scaled input token embedding (`sqrt(n_embd)`)
        let emb_row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        let emb_row = &self.token_embd.bytes[tid * emb_row_bytes..(tid + 1) * emb_row_bytes];
        dequantize_row_bytes(self.token_embd.dtype, emb_row, &mut state.x[..n_embd]);
        let emb_scale = (n_embd as f32).sqrt();
        for v in &mut state.x[..n_embd] {
            *v *= emb_scale;
        }

        // 2. Per-Layer Embeddings (PLE) on-demand row read + projection (`gemma4.cpp:189-198 & 537-560`)
        if let Some(ref ple_g) = self.ple_global {
            let ple_row_bytes = ple_g.tok_embd.dtype.row_bytes(ple_total);
            let ple_row = &ple_g.tok_embd.bytes[tid * ple_row_bytes..(tid + 1) * ple_row_bytes];
            dequantize_row_bytes(
                ple_g.tok_embd.dtype,
                ple_row,
                &mut state.ple_inputs[..ple_total],
            );
            let tok_ple_scale = (ple_dim as f32).sqrt() / 2.0f32.sqrt();
            for v in &mut state.ple_inputs[..ple_total] {
                *v *= tok_ple_scale;
            }

            matvec_f32(
                pool,
                kt,
                ple_g.model_proj.dtype,
                ple_g.model_proj.bytes,
                &state.x[..n_embd],
                &mut state.mv_scratch,
                &mut state.ple_proj_scratch[..ple_total],
                ple_total,
                n_embd,
            );

            let proj_eps = self.cfg.rms_eps * (n_embd as f32);
            for il in 0..self.cfg.n_layers {
                let s = il * ple_dim;
                let e = s + ple_dim;
                rms_norm(
                    &mut state.ple_gate_scratch[..ple_dim],
                    &state.ple_proj_scratch[s..e],
                    &ple_g.proj_norm_scaled,
                    proj_eps,
                );
                for d in 0..ple_dim {
                    state.ple_inputs[s + d] += state.ple_gate_scratch[d];
                }
            }
        }

        // 3. Precompute RoPE cos/sin for both SWA and Full-Attention layers at `pos`
        rope_compute_cos_sin(
            &self.rope_inv_freqs_full,
            pos,
            &mut state.rope_cos_full,
            &mut state.rope_sin_full,
        );
        rope_compute_cos_sin(
            &self.rope_inv_freqs_swa,
            pos,
            &mut state.rope_cos_swa,
            &mut state.rope_sin_swa,
        );

        // 4. Transformer layers (own-KV `0..n_layer_kv_from_start` + KV-shared upper layers)
        for (il, layer) in self.layers.iter().enumerate() {
            let hd = self.cfg.head_dim(il);
            let rope_dim = self.cfg.rope_dim(il);
            let q_dim = self.cfg.n_head * hd;
            let kv_dim = self.cfg.n_head_kv * hd;
            let is_swa = self.cfg.is_swa[il];
            let window = if is_swa { Some(self.cfg.n_swa) } else { None };
            let n_ff = self.cfg.n_ff_per_layer[il];

            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
            );

            let mut did_triple_qkv = false;
            if let Some(ref kv_w) = layer.kv
                && let Some(ref wv) = kv_w.wv
            {
                matvec_triple_f32(
                    pool,
                    kt,
                    layer.wq.dtype,
                    layer.wq.bytes,
                    q_dim,
                    &mut state.q_buf[..q_dim],
                    kv_w.wk.dtype,
                    kv_w.wk.bytes,
                    kv_dim,
                    &mut state.k_buf[..kv_dim],
                    wv.dtype,
                    wv.bytes,
                    kv_dim,
                    &mut state.v_buf[..kv_dim],
                    &state.xb[..n_embd],
                    &mut state.mv_scratch,
                    n_embd,
                );
                did_triple_qkv = true;
            } else {
                matvec_f32(
                    pool,
                    kt,
                    layer.wq.dtype,
                    layer.wq.bytes,
                    &state.xb[..n_embd],
                    &mut state.mv_scratch,
                    &mut state.q_buf[..q_dim],
                    q_dim,
                    n_embd,
                );
            }
            for h in 0..self.cfg.n_head {
                let s = h * hd;
                let e = s + hd;
                rms_norm(
                    &mut state.xb2[..hd],
                    &state.q_buf[s..e],
                    &layer.q_norm,
                    self.cfg.rms_eps,
                );
                state.q_buf[s..e].copy_from_slice(&state.xb2[..hd]);
            }
            let (cos_s, sin_s) = if is_swa {
                (&state.rope_cos_swa[..], &state.rope_sin_swa[..])
            } else {
                (&state.rope_cos_full[..], &state.rope_sin_full[..])
            };
            rope_inplace_tables(
                &mut state.q_buf[..q_dim],
                self.cfg.n_head,
                hd,
                rope_dim,
                cos_s,
                sin_s,
                RopeType::Neox,
            );

            if let Some(ref kv_w) = layer.kv {
                if !did_triple_qkv {
                    if let Some(ref wv) = kv_w.wv {
                        matvec_pair_f32(
                            pool,
                            kt,
                            kv_w.wk.dtype,
                            kv_w.wk.bytes,
                            kv_dim,
                            &mut state.k_buf[..kv_dim],
                            wv.dtype,
                            wv.bytes,
                            kv_dim,
                            &mut state.v_buf[..kv_dim],
                            &state.xb[..n_embd],
                            &mut state.mv_scratch,
                            n_embd,
                        );
                    } else {
                        matvec_f32(
                            pool,
                            kt,
                            kv_w.wk.dtype,
                            kv_w.wk.bytes,
                            &state.xb[..n_embd],
                            &mut state.mv_scratch,
                            &mut state.k_buf[..kv_dim],
                            kv_dim,
                            n_embd,
                        );
                        state.v_buf[..kv_dim].copy_from_slice(&state.k_buf[..kv_dim]);
                    }
                }

                for h in 0..self.cfg.n_head_kv {
                    let s = h * hd;
                    let e = s + hd;
                    rms_norm(
                        &mut state.xb2[..hd],
                        &state.k_buf[s..e],
                        &kv_w.k_norm,
                        self.cfg.rms_eps,
                    );
                    state.k_buf[s..e].copy_from_slice(&state.xb2[..hd]);
                    rms_norm_unit_inplace(&mut state.v_buf[s..e], self.cfg.rms_eps);
                }

                rope_inplace_tables(
                    &mut state.k_buf[..kv_dim],
                    self.cfg.n_head_kv,
                    hd,
                    rope_dim,
                    cos_s,
                    sin_s,
                    RopeType::Neox,
                );

                if let LayerState::Kv {
                    ref mut k_cache,
                    ref mut v_cache,
                    ..
                } = state.layers[il]
                {
                    let off = pos * kv_dim;
                    (kt.f32_to_f16_slice)(&state.k_buf[..kv_dim], &mut k_cache[off..off + kv_dim]);
                    (kt.f32_to_f16_slice)(&state.v_buf[..kv_dim], &mut v_cache[off..off + kv_dim]);
                }
            }

            let kv_src = self.cfg.kv_source[il];
            let (k_cache_ref, v_cache_ref) = match state.layers[kv_src] {
                LayerState::Kv {
                    ref k_cache,
                    ref v_cache,
                    ..
                } => (k_cache.as_slice(), v_cache.as_slice()),
                _ => unreachable!("kv_source always points to a LayerState::Kv layer"),
            };

            // Gemma 4 uses `f_attention_scale = 1.0` (`gemma4.cpp:12`)
            gqa_attention_f16_kv_ext_parallel(
                pool,
                kt,
                &mut state.attn_out[..q_dim],
                &state.q_buf[..q_dim],
                k_cache_ref,
                v_cache_ref,
                self.cfg.n_head,
                self.cfg.n_head_kv,
                hd,
                pos,
                1.0,
                window,
                state.max_ctx,
                &mut state.scores,
            );

            matvec_f32(
                pool,
                kt,
                layer.wo.dtype,
                layer.wo.bytes,
                &state.attn_out[..q_dim],
                &mut state.mv_scratch,
                &mut state.xb2[..n_embd],
                n_embd,
                q_dim,
            );
            rms_norm(
                &mut state.xb[..n_embd],
                &state.xb2[..n_embd],
                &layer.post_attn_norm,
                self.cfg.rms_eps,
            );
            for i in 0..n_embd {
                state.attn_res[i] = state.x[i] + state.xb[i];
            }

            // GeGLU FFN
            rms_norm(
                &mut state.xb[..n_embd],
                &state.attn_res[..n_embd],
                &layer.ffn_norm,
                self.cfg.rms_eps,
            );
            matvec_pair_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                n_ff,
                &mut state.hb[..n_ff],
                layer.w_up.dtype,
                layer.w_up.bytes,
                n_ff,
                &mut state.hb2[..n_ff],
                &state.xb[..n_embd],
                &mut state.mv_scratch,
                n_embd,
            );
            geglu_inplace_parallel(pool, &mut state.hb[..n_ff], &state.hb2[..n_ff]);
            matvec_f32(
                pool,
                kt,
                layer.w_down.dtype,
                layer.w_down.bytes,
                &state.hb[..n_ff],
                &mut state.mv_scratch,
                &mut state.xb2[..n_embd],
                n_embd,
                n_ff,
            );
            rms_norm(
                &mut state.xb[..n_embd],
                &state.xb2[..n_embd],
                &layer.post_ffw_norm,
                self.cfg.rms_eps,
            );
            for i in 0..n_embd {
                state.x[i] = state.attn_res[i] + state.xb[i];
            }

            // Per-layer embedding (PLE) gated residual (`gemma4.cpp:400-420`)
            if let Some(ref ple_l) = layer.ple {
                matvec_f32(
                    pool,
                    kt,
                    ple_l.inp_gate.dtype,
                    ple_l.inp_gate.bytes,
                    &state.x[..n_embd],
                    &mut state.mv_scratch,
                    &mut state.ple_gate_scratch[..ple_dim],
                    ple_dim,
                    n_embd,
                );
                let ple_this = &state.ple_inputs[il * ple_dim..(il + 1) * ple_dim];
                for (g, &p_in) in state.ple_gate_scratch[..ple_dim].iter_mut().zip(ple_this) {
                    *g = gelu_tanh_f32(*g) * p_in;
                }
                matvec_f32(
                    pool,
                    kt,
                    ple_l.proj.dtype,
                    ple_l.proj.bytes,
                    &state.ple_gate_scratch[..ple_dim],
                    &mut state.mv_scratch,
                    &mut state.xb2[..n_embd],
                    n_embd,
                    ple_dim,
                );
                rms_norm(
                    &mut state.xb[..n_embd],
                    &state.xb2[..n_embd],
                    &ple_l.post_norm,
                    self.cfg.rms_eps,
                );
                for i in 0..n_embd {
                    state.x[i] += state.xb[i];
                }
            }

            if let Some(scale) = layer.out_scale {
                for v in &mut state.x[..n_embd] {
                    *v *= scale;
                }
            }
        }

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
            if let Some(cap) = self.cfg.final_logit_softcapping {
                softcap_inplace_parallel(pool, &mut state.logits, cap);
            }
        }

        Ok(())
    }

    /// Batched prefill forward pass for `token_ids` starting at `start_pos`.
    /// Validates all context bounds and token IDs BEFORE mutating `LayerState` (`Rule #6`).
    pub fn forward_batch(
        &self,
        state: &mut Gemma4State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let end_pos = start_pos.saturating_add(token_ids.len());
        if end_pos > state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: end_pos - 1,
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
            return self.forward_token(
                state,
                pool,
                kt,
                token_ids[0],
                start_pos,
                compute_last_logits,
            );
        }

        let mut offset = 0usize;
        while offset < token_ids.len() {
            let b = (token_ids.len() - offset).min(MAX_CPU_BATCH);
            let is_last = offset + b == token_ids.len();
            self.forward_chunk(
                state,
                pool,
                kt,
                &token_ids[offset..offset + b],
                start_pos + offset,
                is_last && compute_last_logits,
            );
            offset += b;
        }
        Ok(())
    }

    fn forward_chunk(
        &self,
        state: &mut Gemma4State,
        pool: &ThreadPool,
        kt: &KernelTable,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) {
        let b = tokens.len();
        let n_embd = self.cfg.n_embd;
        let ple_dim = self.cfg.n_embd_per_layer;
        let ple_total = self.cfg.n_layers * ple_dim;

        // 1. Batched input embedding + sqrt(n_embd)
        let emb_row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        let emb_scale = (n_embd as f32).sqrt();
        for (t, &tok) in tokens.iter().enumerate() {
            let tid = tok as usize;
            let emb_row = &self.token_embd.bytes[tid * emb_row_bytes..(tid + 1) * emb_row_bytes];
            let dst = &mut state.x[t * n_embd..(t + 1) * n_embd];
            dequantize_row_bytes(self.token_embd.dtype, emb_row, dst);
            for v in dst {
                *v *= emb_scale;
            }
        }

        // 2. Batched PLE row lookup + single-pass BF16 `per_layer_model_proj` GEMM
        if let Some(ref ple_g) = self.ple_global {
            let ple_row_bytes = ple_g.tok_embd.dtype.row_bytes(ple_total);
            let tok_ple_scale = (ple_dim as f32).sqrt() / 2.0f32.sqrt();
            for (t, &tok) in tokens.iter().enumerate() {
                let tid = tok as usize;
                let row = &ple_g.tok_embd.bytes[tid * ple_row_bytes..(tid + 1) * ple_row_bytes];
                let dst = &mut state.ple_inputs[t * ple_total..(t + 1) * ple_total];
                dequantize_row_bytes(ple_g.tok_embd.dtype, row, dst);
                for v in dst {
                    *v *= tok_ple_scale;
                }
            }

            matmul_f32(
                pool,
                kt,
                ple_g.model_proj.dtype,
                ple_g.model_proj.bytes,
                &state.x[..b * n_embd],
                &mut state.mv_scratch,
                &mut state.ple_proj_scratch[..b * ple_total],
                ple_total,
                n_embd,
                b,
            );

            let proj_eps = self.cfg.rms_eps * (n_embd as f32);
            for t in 0..b {
                for il in 0..self.cfg.n_layers {
                    let s = t * ple_total + il * ple_dim;
                    let e = s + ple_dim;
                    rms_norm(
                        &mut state.ple_gate_scratch[..ple_dim],
                        &state.ple_proj_scratch[s..e],
                        &ple_g.proj_norm_scaled,
                        proj_eps,
                    );
                    for d in 0..ple_dim {
                        state.ple_inputs[s + d] += state.ple_gate_scratch[d];
                    }
                }
            }
        }

        // 3. Layers (Note: for KV-shared layers `il >= n_layer_kv_from_start`, if `!compute_last_logits`
        //    is false on the final chunk we still compute all `b` rows unless pruning; if `!compute_last_logits`
        //    on intermediate chunks of a long prompt, KV-shared layers don't write KV cache at all!!
        //    Wait: `gemma4.cpp:201` prunes non-output rows at `il == n_layer_kv_from_start`!
        //    If `!compute_last_logits`, `n_outputs == 0` so layers `15..34` have ZERO side effects
        //    on KV cache and can be skipped completely on intermediate chunks, and on the last chunk
        //    only the last token `b - 1` needs to run through layers `15..34`!!).
        let mut active_rows = b;
        let mut row_pos_offset = start_pos;

        for (il, layer) in self.layers.iter().enumerate() {
            if il == self.cfg.n_layer_kv_from_start
                && self.cfg.n_layer_kv_from_start < self.cfg.n_layers
            {
                if !compute_last_logits {
                    // Intermediate prefill chunk: layers `n_layer_kv_from_start..n_layers` write NO state!
                    return;
                }
                if active_rows > 1 {
                    // Prune to only the final output token `(b - 1)` (`gemma4.cpp:211-228`)!
                    let last = b - 1;
                    state.x.copy_within(last * n_embd..(last + 1) * n_embd, 0);
                    if ple_total > 0 {
                        state
                            .ple_inputs
                            .copy_within(last * ple_total..(last + 1) * ple_total, 0);
                    }
                    active_rows = 1;
                    row_pos_offset = start_pos + last;
                }
            }

            let hd = self.cfg.head_dim(il);
            let rope_dim = self.cfg.rope_dim(il);
            let q_dim = self.cfg.n_head * hd;
            let kv_dim = self.cfg.n_head_kv * hd;
            let is_swa = self.cfg.is_swa[il];
            let window = if is_swa { Some(self.cfg.n_swa) } else { None };
            let n_ff = self.cfg.n_ff_per_layer[il];

            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..active_rows * n_embd],
                &state.x[..active_rows * n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
                n_embd,
                active_rows,
            );

            matmul_f32(
                pool,
                kt,
                layer.wq.dtype,
                layer.wq.bytes,
                &state.xb[..active_rows * n_embd],
                &mut state.mv_scratch,
                &mut state.q_buf[..active_rows * q_dim],
                q_dim,
                n_embd,
                active_rows,
            );

            if let Some(ref kv_w) = layer.kv {
                matmul_f32(
                    pool,
                    kt,
                    kv_w.wk.dtype,
                    kv_w.wk.bytes,
                    &state.xb[..active_rows * n_embd],
                    &mut state.mv_scratch,
                    &mut state.k_buf[..active_rows * kv_dim],
                    kv_dim,
                    n_embd,
                    active_rows,
                );
                if let Some(ref wv) = kv_w.wv {
                    matmul_f32(
                        pool,
                        kt,
                        wv.dtype,
                        wv.bytes,
                        &state.xb[..active_rows * n_embd],
                        &mut state.mv_scratch,
                        &mut state.v_buf[..active_rows * kv_dim],
                        kv_dim,
                        n_embd,
                        active_rows,
                    );
                } else {
                    state.v_buf[..active_rows * kv_dim]
                        .copy_from_slice(&state.k_buf[..active_rows * kv_dim]);
                }
            }

            for t in 0..active_rows {
                let pos = row_pos_offset + t;
                let qt = &mut state.q_buf[t * q_dim..(t + 1) * q_dim];
                for h in 0..self.cfg.n_head {
                    let s = h * hd;
                    let e = s + hd;
                    rms_norm(
                        &mut state.xb2[..hd],
                        &qt[s..e],
                        &layer.q_norm,
                        self.cfg.rms_eps,
                    );
                    qt[s..e].copy_from_slice(&state.xb2[..hd]);
                }

                if is_swa {
                    rope_compute_cos_sin(
                        &self.rope_inv_freqs_swa,
                        pos,
                        &mut state.rope_cos_swa,
                        &mut state.rope_sin_swa,
                    );
                    rope_inplace_tables(
                        qt,
                        self.cfg.n_head,
                        hd,
                        rope_dim,
                        &state.rope_cos_swa,
                        &state.rope_sin_swa,
                        RopeType::Neox,
                    );
                } else {
                    rope_compute_cos_sin(
                        &self.rope_inv_freqs_full,
                        pos,
                        &mut state.rope_cos_full,
                        &mut state.rope_sin_full,
                    );
                    rope_inplace_tables(
                        qt,
                        self.cfg.n_head,
                        hd,
                        rope_dim,
                        &state.rope_cos_full,
                        &state.rope_sin_full,
                        RopeType::Neox,
                    );
                }

                if let Some(ref kv_w) = layer.kv {
                    let kt_vec = &mut state.k_buf[t * kv_dim..(t + 1) * kv_dim];
                    let vt_vec = &mut state.v_buf[t * kv_dim..(t + 1) * kv_dim];
                    for h in 0..self.cfg.n_head_kv {
                        let s = h * hd;
                        let e = s + hd;
                        rms_norm(
                            &mut state.xb2[..hd],
                            &kt_vec[s..e],
                            &kv_w.k_norm,
                            self.cfg.rms_eps,
                        );
                        kt_vec[s..e].copy_from_slice(&state.xb2[..hd]);
                        rms_norm_unit_inplace(&mut vt_vec[s..e], self.cfg.rms_eps);
                    }

                    let (cos_s, sin_s) = if is_swa {
                        (&state.rope_cos_swa[..], &state.rope_sin_swa[..])
                    } else {
                        (&state.rope_cos_full[..], &state.rope_sin_full[..])
                    };
                    rope_inplace_tables(
                        kt_vec,
                        self.cfg.n_head_kv,
                        hd,
                        rope_dim,
                        cos_s,
                        sin_s,
                        RopeType::Neox,
                    );

                    if let LayerState::Kv {
                        ref mut k_cache,
                        ref mut v_cache,
                        ..
                    } = state.layers[il]
                    {
                        let off = pos * kv_dim;
                        (kt.f32_to_f16_slice)(kt_vec, &mut k_cache[off..off + kv_dim]);
                        (kt.f32_to_f16_slice)(vt_vec, &mut v_cache[off..off + kv_dim]);
                    }
                }
            }

            let kv_src = self.cfg.kv_source[il];
            let (k_cache_ref, v_cache_ref) = match state.layers[kv_src] {
                LayerState::Kv {
                    ref k_cache,
                    ref v_cache,
                    ..
                } => (k_cache.as_slice(), v_cache.as_slice()),
                _ => unreachable!("kv_source always points to LayerState::Kv"),
            };

            gqa_attention_f16_kv_ext_batch_parallel(
                pool,
                kt,
                &mut state.attn_out[..active_rows * q_dim],
                &state.q_buf[..active_rows * q_dim],
                k_cache_ref,
                v_cache_ref,
                self.cfg.n_head,
                self.cfg.n_head_kv,
                hd,
                row_pos_offset,
                active_rows,
                1.0,
                window,
                state.max_ctx,
                &mut state.scores,
            );

            matmul_f32(
                pool,
                kt,
                layer.wo.dtype,
                layer.wo.bytes,
                &state.attn_out[..active_rows * q_dim],
                &mut state.mv_scratch,
                &mut state.xb2[..active_rows * n_embd],
                n_embd,
                q_dim,
                active_rows,
            );
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..active_rows * n_embd],
                &state.xb2[..active_rows * n_embd],
                &layer.post_attn_norm,
                self.cfg.rms_eps,
                n_embd,
                active_rows,
            );
            for i in 0..active_rows * n_embd {
                state.attn_res[i] = state.x[i] + state.xb[i];
            }

            // GeGLU FFN
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..active_rows * n_embd],
                &state.attn_res[..active_rows * n_embd],
                &layer.ffn_norm,
                self.cfg.rms_eps,
                n_embd,
                active_rows,
            );
            matmul_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                &state.xb[..active_rows * n_embd],
                &mut state.mv_scratch,
                &mut state.hb[..active_rows * n_ff],
                n_ff,
                n_embd,
                active_rows,
            );
            matmul_f32(
                pool,
                kt,
                layer.w_up.dtype,
                layer.w_up.bytes,
                &state.xb[..active_rows * n_embd],
                &mut state.mv_scratch,
                &mut state.hb2[..active_rows * n_ff],
                n_ff,
                n_embd,
                active_rows,
            );
            geglu_inplace_parallel(
                pool,
                &mut state.hb[..active_rows * n_ff],
                &state.hb2[..active_rows * n_ff],
            );
            matmul_f32(
                pool,
                kt,
                layer.w_down.dtype,
                layer.w_down.bytes,
                &state.hb[..active_rows * n_ff],
                &mut state.mv_scratch,
                &mut state.xb2[..active_rows * n_embd],
                n_embd,
                n_ff,
                active_rows,
            );
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..active_rows * n_embd],
                &state.xb2[..active_rows * n_embd],
                &layer.post_ffw_norm,
                self.cfg.rms_eps,
                n_embd,
                active_rows,
            );
            for i in 0..active_rows * n_embd {
                state.x[i] = state.attn_res[i] + state.xb[i];
            }

            if let Some(ref ple_l) = layer.ple {
                matmul_f32(
                    pool,
                    kt,
                    ple_l.inp_gate.dtype,
                    ple_l.inp_gate.bytes,
                    &state.x[..active_rows * n_embd],
                    &mut state.mv_scratch,
                    &mut state.ple_gate_scratch[..active_rows * ple_dim],
                    ple_dim,
                    n_embd,
                    active_rows,
                );
                for t in 0..active_rows {
                    let ple_this = &state.ple_inputs
                        [t * ple_total + il * ple_dim..t * ple_total + (il + 1) * ple_dim];
                    let g_slice = &mut state.ple_gate_scratch[t * ple_dim..(t + 1) * ple_dim];
                    for d in 0..ple_dim {
                        g_slice[d] = gelu_tanh_f32(g_slice[d]) * ple_this[d];
                    }
                }
                matmul_f32(
                    pool,
                    kt,
                    ple_l.proj.dtype,
                    ple_l.proj.bytes,
                    &state.ple_gate_scratch[..active_rows * ple_dim],
                    &mut state.mv_scratch,
                    &mut state.xb2[..active_rows * n_embd],
                    n_embd,
                    ple_dim,
                    active_rows,
                );
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..active_rows * n_embd],
                    &state.xb2[..active_rows * n_embd],
                    &ple_l.post_norm,
                    self.cfg.rms_eps,
                    n_embd,
                    active_rows,
                );
                for i in 0..active_rows * n_embd {
                    state.x[i] += state.xb[i];
                }
            }

            if let Some(scale) = layer.out_scale {
                for v in &mut state.x[..active_rows * n_embd] {
                    *v *= scale;
                }
            }
        }

        if compute_last_logits {
            let last_off = (active_rows - 1) * n_embd;
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[last_off..last_off + n_embd],
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
            if let Some(cap) = self.cfg.final_logit_softcapping {
                softcap_inplace_parallel(pool, &mut state.logits, cap);
            }
        }
    }
}
