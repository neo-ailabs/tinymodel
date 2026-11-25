//! Liquid AI LFM2 / LFM2.5 (`general.architecture == "lfm2"`) hybrid Gated ShortConv + GQA forward pass.
//!
//! Reference: `third_party/llama.cpp/src/models/lfm2.cpp` & `lfm-base.cpp` (MIT License).
//! Layer dispatch rule (`lfm2.cpp:15`):
//! - Layers with `head_count_kv[il] == 0` are `ShortConv` layers:
//!   `bcx = in_proj * cur` (`[3 * n_embd]`), split into `b, c, x` (`[n_embd]` each),
//!   `bx = b * x`, causal depthwise convolution of kernel `l_cache` (`hist_len = l_cache - 1`
//!   in `LayerState::ShortConv`) with `shortconv.conv.weight` (`[l_cache, n_embd]`),
//!   `y = c * conv_out`, `out = out_proj * y`.
//! - Layers with `head_count_kv[il] > 0` are GQA attention layers with per-head `attn_q_norm`
//!   and `attn_k_norm`, NeoX RoPE, and optional standard sliding window (`n_swa`).
//! - Output RMSNorm tensor is `token_embd_norm.weight` (with fallback to `output_norm.weight`).

use tm_core::{DType, KvType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool, matmul_f32, matmul_pair_f32,
    matmul_triple_f32, matvec_f32, matvec_pair_f32, matvec_triple_f32, rms_norm,
    rms_norm_batch_parallel, rope_compute_cos_sin, rope_inplace_tables, swiglu_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};
use tm_lora::{BoundLora, LoraScratch, LoraSet};

use crate::llama::{ModelError, RawMatView};
use crate::state::{LayerState, MAX_SPEC_WINDOW, SpecVerifyBuf, decode_embedding_row};

/// Hyperparameters for `general.architecture == "lfm2"` or `"lfm3"`.
#[derive(Debug, Clone)]
pub struct Lfm2Config {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv_per_layer: Vec<usize>,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub shortconv_l_cache: usize,
    pub n_swa: Option<usize>,
    pub is_swa: Vec<bool>,
    pub vocab_size: usize,
    pub max_ctx_train: usize,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rms_eps: f32,
}

impl Lfm2Config {
    #[inline]
    pub fn is_shortconv(&self, il: usize) -> bool {
        self.n_head_kv_per_layer[il] == 0
    }

    #[inline]
    pub fn layer_window(&self, il: usize) -> Option<usize> {
        if self.is_swa.get(il).copied().unwrap_or(false) {
            self.n_swa
        } else {
            None
        }
    }

    #[inline]
    pub fn max_n_head_kv(&self) -> usize {
        self.n_head_kv_per_layer.iter().copied().max().unwrap_or(0)
    }
}

type MatView<'a> = RawMatView<'a>;

#[allow(clippy::large_enum_variant)]
pub enum Lfm2OpWeights<'a> {
    ShortConv {
        in_proj: RawMatView<'a>,
        /// Depthwise causal conv weights `[n_embd, l_cache]` (`l_cache` contiguous floats per channel).
        conv: Vec<f32>,
        out_proj: RawMatView<'a>,
    },
    Gqa {
        n_head_kv: usize,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        wq: RawMatView<'a>,
        wk: RawMatView<'a>,
        wv: RawMatView<'a>,
        w_attn_gate: Option<RawMatView<'a>>,
        wo: RawMatView<'a>,
    },
}

pub struct Lfm2LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub op: Lfm2OpWeights<'a>,
    pub post_attn_norm: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: RawMatView<'a>,
    pub w_up: RawMatView<'a>,
    pub w_down: RawMatView<'a>,
    pub post_ffw_norm: Option<Vec<f32>>,
}

/// Loaded LFM2 / LFM3 hybrid model borrowing quantized weight slices zero-copy from `GgufFile`.
pub struct Lfm2Model<'a> {
    pub cfg: Lfm2Config,
    pub token_embd: RawMatView<'a>,
    pub output: RawMatView<'a>,
    pub output_norm: Vec<f32>,
    pub rope_inv_freqs: Vec<f64>,
    pub rope_inv_freqs_swa: Vec<f64>,
    pub layers: Vec<Lfm2LayerWeights<'a>>,
    pub lora: BoundLora,
}

/// Pre-allocated per-session inference state for LFM2 / LFM3 (`LayerState::ShortConv` + `LayerState::Kv`).
pub struct Lfm2State {
    pub max_ctx: usize,
    pub layers: Vec<LayerState>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    bcx_buf: Vec<f32>,
    conv_y: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    attn_out: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    rope_cos_swa: Vec<f32>,
    rope_sin_swa: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    pub last_hidden: Vec<f32>,
    mv_scratch: MatvecScratch,
    pub lora_scratch: LoraScratch,
}

impl Lfm2State {
    pub fn new(model: &Lfm2Model<'_>, max_ctx: usize) -> Self {
        Self::new_with_kv_type(model, max_ctx, KvType::F16)
    }

    pub fn new_with_kv_type(model: &Lfm2Model<'_>, max_ctx: usize, kv_type: KvType) -> Self {
        let cfg = &model.cfg;
        let hist_len = cfg.shortconv_l_cache.saturating_sub(1);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            if cfg.is_shortconv(il) {
                layers.push(LayerState::ShortConv {
                    conv_hist: vec![0.0f32; cfg.n_embd * hist_len],
                });
            } else {
                let kv_dim = cfg.n_head_kv_per_layer[il] * cfg.head_dim;
                layers.push(LayerState::new_kv(
                    max_ctx,
                    kv_dim,
                    cfg.layer_window(il),
                    kv_type,
                ));
            }
        }

        let q_dim = cfg.n_head * cfg.head_dim;
        let max_kv_dim = cfg.max_n_head_kv() * cfg.head_dim;
        let max_cols = cfg.n_embd.max(cfg.n_ff).max(q_dim).max(3 * cfg.n_embd);

        Self {
            max_ctx,
            layers,
            x: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb2: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            bcx_buf: vec![0.0; MAX_CPU_BATCH * 3 * cfg.n_embd],
            conv_y: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            q_buf: vec![0.0; MAX_CPU_BATCH * q_dim.max(1)],
            k_buf: vec![0.0; MAX_CPU_BATCH * max_kv_dim.max(1)],
            v_buf: vec![0.0; MAX_CPU_BATCH * max_kv_dim.max(1)],
            gate_buf: vec![0.0; MAX_CPU_BATCH * cfg.n_head.max(1)],
            attn_out: vec![0.0; MAX_CPU_BATCH * q_dim.max(1)],
            hb: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            hb2: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            rope_cos: vec![0.0; cfg.rope_dim / 2],
            rope_sin: vec![0.0; cfg.rope_dim / 2],
            rope_cos_swa: vec![0.0; cfg.rope_dim / 2],
            rope_sin_swa: vec![0.0; cfg.rope_dim / 2],
            scores: vec![0.0; cfg.n_head.max(1) * max_ctx],
            logits: vec![0.0; cfg.vocab_size],
            last_hidden: vec![0.0; cfg.n_embd],
            mv_scratch: MatvecScratch::new(max_cols),
            lora_scratch: LoraScratch::new(64),
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
    expected_dims: &[usize],
) -> Result<Vec<f32>, ModelError> {
    let (info, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
    let len: usize = expected_dims.iter().product();
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
        name: info.name.as_str(),
        bytes,
        dtype: info.dtype,
        rows,
        cols,
        row_offset: 0,
        is_sliced: false,
        lora_slot: None,
    })
}

impl<'a> Lfm2Model<'a> {
    /// Loads and strictly validates an `lfm2` or `lfm3` GGUF model.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "lfm2" && arch != "lfm3" {
            return Err(ModelError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        let n_layers = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
        let n_ff = if let Some(mv) = gguf.get_meta(&format!("{pfx}.feed_forward_length")) {
            mv.as_usize()
                .or_else(|| mv.as_usize_array().and_then(|v| v.first().copied()))
                .ok_or_else(|| GgufError::MetadataTypeMismatch {
                    path: gguf.path().display().to_string(),
                    key: format!("{pfx}.feed_forward_length"),
                    expected: "usize or usize[]",
                })?
        } else {
            return Err(GgufError::MissingMetadata {
                path: gguf.path().display().to_string(),
                key: format!("{pfx}.feed_forward_length"),
            }
            .into());
        };

        let n_head = if let Some(mv) = gguf.get_meta(&format!("{pfx}.attention.head_count")) {
            mv.as_usize()
                .or_else(|| mv.as_usize_array().and_then(|v| v.iter().copied().max()))
                .unwrap_or(1)
        } else {
            1
        };

        let hkv_key = format!("{pfx}.attention.head_count_kv");
        let n_head_kv_per_layer = if let Some(mv) = gguf.get_meta(&hkv_key) {
            if let Some(arr) = mv.as_usize_array() {
                if arr.len() != n_layers {
                    return Err(GgufError::MetadataTypeMismatch {
                        path: gguf.path().display().to_string(),
                        key: hkv_key,
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
                    key: hkv_key,
                    expected: "usize or usize[]",
                }
                .into());
            }
        } else {
            return Err(GgufError::MissingMetadata {
                path: gguf.path().display().to_string(),
                key: hkv_key,
            }
            .into());
        };

        let head_dim = gguf
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head.max(1));
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let shortconv_l_cache = gguf.req_usize(&format!("{pfx}.shortconv.l_cache"))?;
        let n_swa = gguf
            .get_meta(&format!("{pfx}.attention.sliding_window"))
            .and_then(|v| v.as_usize())
            .filter(|&w| w > 0);
        let swa_pat_key = format!("{pfx}.attention.sliding_window_pattern");
        let is_swa = if let Some(arr) = gguf
            .get_meta(&swa_pat_key)
            .and_then(|v| v.as_bool_array())
            .filter(|v| v.len() == n_layers)
        {
            arr
        } else if pfx == "lfm3" {
            return Err(GgufError::MissingMetadata {
                path: gguf.path().display().to_string(),
                key: swa_pat_key,
            }
            .into());
        } else {
            (0..n_layers)
                .map(|il| n_swa.is_some() && n_head_kv_per_layer[il] > 0)
                .collect()
        };
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let rope_freq_base_swa = gguf
            .get_meta(&format!("{pfx}.rope.freq_base_swa"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "lfm3" {
                10_000.0
            } else {
                rope_freq_base
            });
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

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
            MatView {
                name: "output.weight",
                ..token_embd
            }
        };
        // LFM2 stores output RMSNorm under `token_embd_norm.weight` (`llama-arch.cpp:535`)
        let output_norm = if gguf.tensor_info("token_embd_norm.weight").is_some() {
            load_f32_vec(gguf, "token_embd_norm.weight", "global", &[n_embd])?
        } else {
            load_f32_vec(gguf, "output_norm.weight", "global", &[n_embd])?
        };

        let mut rope_freqs: Option<Vec<f32>> = if gguf.tensor_info("rope_freqs.weight").is_some() {
            Some(load_f32_vec(
                gguf,
                "rope_freqs.weight",
                "global",
                &[rope_dim / 2],
            )?)
        } else {
            None
        };
        if rope_freqs.is_none() {
            for il in 0..n_layers {
                if n_head_kv_per_layer[il] > 0 && !is_swa[il] {
                    let rf_name = format!("blk.{il}.rope_freqs.weight");
                    if gguf.tensor_info(&rf_name).is_some() {
                        rope_freqs = Some(load_f32_vec(
                            gguf,
                            &rf_name,
                            &format!("blk.{il}"),
                            &[rope_dim / 2],
                        )?);
                        break;
                    }
                }
            }
        }

        let rope_inv_freqs: Vec<f64> = (0..(rope_dim / 2))
            .map(|i| {
                let mut f = (rope_freq_base as f64).powf(-2.0 * (i as f64) / (rope_dim as f64));
                if let Some(ref rf) = rope_freqs {
                    f /= rf[i] as f64;
                }
                f
            })
            .collect();
        let rope_inv_freqs_swa: Vec<f64> = (0..(rope_dim / 2))
            .map(|i| (rope_freq_base_swa as f64).powf(-2.0 * (i as f64) / (rope_dim as f64)))
            .collect();

        let q_dim = n_head * head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for (il, &n_head_kv) in n_head_kv_per_layer.iter().enumerate() {
            let ltag = format!("blk.{il}");
            let attn_norm =
                load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, &[n_embd])?;
            let ffn_norm =
                load_f32_vec(gguf, &format!("{ltag}.ffn_norm.weight"), &ltag, &[n_embd])?;
            let post_attn_name = format!("{ltag}.post_attention_norm.weight");
            let post_attn_norm = if gguf.tensor_info(&post_attn_name).is_some() {
                Some(load_f32_vec(gguf, &post_attn_name, &ltag, &[n_embd])?)
            } else {
                None
            };
            let post_ffw_name = format!("{ltag}.post_ffw_norm.weight");
            let post_ffw_norm = if gguf.tensor_info(&post_ffw_name).is_some() {
                Some(load_f32_vec(gguf, &post_ffw_name, &ltag, &[n_embd])?)
            } else {
                None
            };

            let op = if n_head_kv == 0 {
                let in_proj = load_mat(
                    gguf,
                    &format!("{ltag}.shortconv.in_proj.weight"),
                    &ltag,
                    n_embd,
                    3 * n_embd,
                )?;
                let conv_name = format!("{ltag}.shortconv.conv.weight");
                let conv = if let Some(info) = gguf.tensor_info(&conv_name)
                    && info.dims.len() == 3
                {
                    load_f32_vec(gguf, &conv_name, &ltag, &[shortconv_l_cache, 1, n_embd])?
                } else {
                    load_f32_vec(gguf, &conv_name, &ltag, &[shortconv_l_cache, n_embd])?
                };
                let out_proj = load_mat(
                    gguf,
                    &format!("{ltag}.shortconv.out_proj.weight"),
                    &ltag,
                    n_embd,
                    n_embd,
                )?;
                Lfm2OpWeights::ShortConv {
                    in_proj,
                    conv,
                    out_proj,
                }
            } else {
                let kv_dim = n_head_kv * head_dim;
                Lfm2OpWeights::Gqa {
                    n_head_kv,
                    q_norm: load_f32_vec(
                        gguf,
                        &format!("{ltag}.attn_q_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    k_norm: load_f32_vec(
                        gguf,
                        &format!("{ltag}.attn_k_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
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
                    w_attn_gate: if gguf
                        .tensor_info(&format!("{ltag}.attn_gate.weight"))
                        .is_some()
                    {
                        Some(load_mat(
                            gguf,
                            &format!("{ltag}.attn_gate.weight"),
                            &ltag,
                            n_embd,
                            n_head,
                        )?)
                    } else {
                        None
                    },
                    wo: load_mat(
                        gguf,
                        &format!("{ltag}.attn_output.weight"),
                        &ltag,
                        q_dim,
                        n_embd,
                    )?,
                }
            };

            layers.push(Lfm2LayerWeights {
                attn_norm,
                op,
                post_attn_norm,
                ffn_norm,
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
                post_ffw_norm,
            });
        }

        let cfg = Lfm2Config {
            arch,
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv_per_layer,
            head_dim,
            rope_dim,
            shortconv_l_cache,
            n_swa,
            is_swa,
            vocab_size,
            max_ctx_train,
            rope_freq_base,
            rope_freq_base_swa,
            rms_eps,
        };

        Ok(Self {
            cfg,
            token_embd,
            output,
            output_norm,
            rope_inv_freqs,
            rope_inv_freqs_swa,
            layers,
            lora: BoundLora::default(),
        })
    }

    pub fn bind_lora(&mut self, set: &LoraSet) {
        let bound = BoundLora::from_lora_set(set);
        self.token_embd.lora_slot = bound.slot_for(self.token_embd.name);
        self.output.lora_slot = bound.slot_for(self.output.name);
        for l in &mut self.layers {
            match &mut l.op {
                Lfm2OpWeights::ShortConv {
                    in_proj, out_proj, ..
                } => {
                    in_proj.lora_slot = bound.slot_for(in_proj.name);
                    out_proj.lora_slot = bound.slot_for(out_proj.name);
                }
                Lfm2OpWeights::Gqa {
                    wq,
                    wk,
                    wv,
                    w_attn_gate,
                    wo,
                    ..
                } => {
                    wq.lora_slot = bound.slot_for(wq.name);
                    wk.lora_slot = bound.slot_for(wk.name);
                    wv.lora_slot = bound.slot_for(wv.name);
                    if let Some(wg) = w_attn_gate {
                        wg.lora_slot = bound.slot_for(wg.name);
                    }
                    wo.lora_slot = bound.slot_for(wo.name);
                }
            }
            l.w_gate.lora_slot = bound.slot_for(l.w_gate.name);
            l.w_up.lora_slot = bound.slot_for(l.w_up.name);
            l.w_down.lora_slot = bound.slot_for(l.w_down.name);
        }
        self.lora = bound;
    }

    pub fn num_shortconv_layers(&self) -> usize {
        (0..self.cfg.n_layers)
            .filter(|&il| self.cfg.is_shortconv(il))
            .count()
    }

    /// Single-token forward pass. Validates bounds BEFORE mutating `LayerState` (`Rule #6`).
    pub fn forward_token(
        &self,
        state: &mut Lfm2State,
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

        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes,
            n_embd,
            tid,
            &mut state.x[..n_embd],
        );
        self.lora
            .apply_embed_row(self.token_embd.lora_slot, kt, tid, &mut state.x[..n_embd]);

        self.forward_from_x(state, pool, kt, pos, compute_logits)
    }

    /// Runs a single-position forward pass from a precomputed input embedding (`embed.len() == n_embd`),
    /// used for multimodal vision token injection (`ADR-019`).
    pub fn forward_embed(
        &self,
        state: &mut Lfm2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        embed: &[f32],
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        let n_embd = self.cfg.n_embd;
        if embed.len() != n_embd {
            return Err(ModelError::InvalidConfig(format!(
                "forward_embed length mismatch: expected {n_embd}, got {}",
                embed.len()
            )));
        }
        state.x[..n_embd].copy_from_slice(embed);
        self.forward_from_x(state, pool, kt, pos, compute_logits)
    }

    fn forward_from_x(
        &self,
        state: &mut Lfm2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let hd = self.cfg.head_dim;
        let q_dim = self.cfg.n_head * hd;
        let l_cache = self.cfg.shortconv_l_cache;
        let hist_len = l_cache.saturating_sub(1);
        let attn_scale = 1.0 / (hd as f32).sqrt();

        rope_compute_cos_sin(
            &self.rope_inv_freqs,
            pos,
            &mut state.rope_cos,
            &mut state.rope_sin,
        );
        rope_compute_cos_sin(
            &self.rope_inv_freqs_swa,
            pos,
            &mut state.rope_cos_swa,
            &mut state.rope_sin_swa,
        );

        for (il, layer) in self.layers.iter().enumerate() {
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
            );

            match (&layer.op, &mut state.layers[il]) {
                (
                    Lfm2OpWeights::ShortConv {
                        in_proj,
                        conv,
                        out_proj,
                    },
                    LayerState::ShortConv { conv_hist },
                ) => {
                    matvec_f32(
                        pool,
                        kt,
                        in_proj.dtype,
                        in_proj.bytes,
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        &mut state.bcx_buf[..3 * n_embd],
                        3 * n_embd,
                        n_embd,
                    );
                    self.lora.apply_linear(
                        in_proj.lora_slot,
                        pool,
                        kt,
                        &state.xb[..n_embd],
                        &mut state.bcx_buf[..3 * n_embd],
                        1,
                        &mut state.lora_scratch,
                    );

                    // `bcx` layout (`lfm-base.cpp:28-60`): `[b (n_embd), c (n_embd), x (n_embd)]`
                    for ch in 0..n_embd {
                        let b_val = state.bcx_buf[ch];
                        let c_val = state.bcx_buf[n_embd + ch];
                        let x_val = state.bcx_buf[2 * n_embd + ch];
                        let bx_val = b_val * x_val;

                        let h_slice = &mut conv_hist[ch * hist_len..(ch + 1) * hist_len];
                        let w_slice = &conv[ch * l_cache..(ch + 1) * l_cache];
                        let mut sum = 0.0f32;
                        for tap in 0..hist_len {
                            sum += h_slice[tap] * w_slice[tap];
                        }
                        sum += bx_val * w_slice[hist_len];
                        for tap in 0..hist_len.saturating_sub(1) {
                            h_slice[tap] = h_slice[tap + 1];
                        }
                        if hist_len > 0 {
                            h_slice[hist_len - 1] = bx_val;
                        }
                        state.conv_y[ch] = c_val * sum;
                    }

                    matvec_f32(
                        pool,
                        kt,
                        out_proj.dtype,
                        out_proj.bytes,
                        &state.conv_y[..n_embd],
                        &mut state.mv_scratch,
                        &mut state.xb2[..n_embd],
                        n_embd,
                        n_embd,
                    );
                    self.lora.apply_linear(
                        out_proj.lora_slot,
                        pool,
                        kt,
                        &state.conv_y[..n_embd],
                        &mut state.xb2[..n_embd],
                        1,
                        &mut state.lora_scratch,
                    );
                }
                (
                    Lfm2OpWeights::Gqa {
                        n_head_kv,
                        q_norm,
                        k_norm,
                        wq,
                        wk,
                        wv,
                        w_attn_gate,
                        wo,
                    },
                    kv_state @ LayerState::Kv { .. },
                ) => {
                    let kv_dim = *n_head_kv * hd;
                    matvec_triple_f32(
                        pool,
                        kt,
                        wq.dtype,
                        wq.bytes,
                        q_dim,
                        &mut state.q_buf[..q_dim],
                        wk.dtype,
                        wk.bytes,
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
                    if !self.lora.is_empty() {
                        self.lora.apply_linear(
                            wq.lora_slot,
                            pool,
                            kt,
                            &state.xb[..n_embd],
                            &mut state.q_buf[..q_dim],
                            1,
                            &mut state.lora_scratch,
                        );
                        self.lora.apply_linear(
                            wk.lora_slot,
                            pool,
                            kt,
                            &state.xb[..n_embd],
                            &mut state.k_buf[..kv_dim],
                            1,
                            &mut state.lora_scratch,
                        );
                        self.lora.apply_linear(
                            wv.lora_slot,
                            pool,
                            kt,
                            &state.xb[..n_embd],
                            &mut state.v_buf[..kv_dim],
                            1,
                            &mut state.lora_scratch,
                        );
                    }
                    if let Some(wg) = w_attn_gate {
                        matvec_f32(
                            pool,
                            kt,
                            wg.dtype,
                            wg.bytes,
                            &state.xb[..n_embd],
                            &mut state.mv_scratch,
                            &mut state.gate_buf[..self.cfg.n_head],
                            self.cfg.n_head,
                            n_embd,
                        );
                        self.lora.apply_linear(
                            wg.lora_slot,
                            pool,
                            kt,
                            &state.xb[..n_embd],
                            &mut state.gate_buf[..self.cfg.n_head],
                            1,
                            &mut state.lora_scratch,
                        );
                    }

                    for h in 0..self.cfg.n_head {
                        let s = h * hd;
                        let e = s + hd;
                        rms_norm(
                            &mut state.xb2[..hd],
                            &state.q_buf[s..e],
                            q_norm,
                            self.cfg.rms_eps,
                        );
                        state.q_buf[s..e].copy_from_slice(&state.xb2[..hd]);
                    }
                    for h in 0..*n_head_kv {
                        let s = h * hd;
                        let e = s + hd;
                        rms_norm(
                            &mut state.xb2[..hd],
                            &state.k_buf[s..e],
                            k_norm,
                            self.cfg.rms_eps,
                        );
                        state.k_buf[s..e].copy_from_slice(&state.xb2[..hd]);
                    }

                    let (cos_s, sin_s) = if self.cfg.is_swa[il] {
                        (&state.rope_cos_swa[..], &state.rope_sin_swa[..])
                    } else {
                        (&state.rope_cos[..], &state.rope_sin[..])
                    };
                    rope_inplace_tables(
                        &mut state.q_buf[..q_dim],
                        self.cfg.n_head,
                        hd,
                        self.cfg.rope_dim,
                        cos_s,
                        sin_s,
                        RopeType::Neox,
                    );
                    rope_inplace_tables(
                        &mut state.k_buf[..kv_dim],
                        *n_head_kv,
                        hd,
                        self.cfg.rope_dim,
                        cos_s,
                        sin_s,
                        RopeType::Neox,
                    );

                    kv_state.store_kv_at(kt, pos, &state.k_buf[..kv_dim], &state.v_buf[..kv_dim]);

                    kv_state.attend_single(
                        pool,
                        kt,
                        &mut state.attn_out[..q_dim],
                        &state.q_buf[..q_dim],
                        self.cfg.n_head,
                        *n_head_kv,
                        hd,
                        pos,
                        attn_scale,
                        None,
                        state.max_ctx,
                        &mut state.scores,
                    );
                    if w_attn_gate.is_some() {
                        for h in 0..self.cfg.n_head {
                            let g = 1.0 / (1.0 + (-state.gate_buf[h]).exp());
                            for v in &mut state.attn_out[h * hd..(h + 1) * hd] {
                                *v *= g;
                            }
                        }
                    }

                    matvec_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.attn_out[..q_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..n_embd],
                        n_embd,
                        q_dim,
                    );
                    self.lora.apply_linear(
                        wo.lora_slot,
                        pool,
                        kt,
                        &state.attn_out[..q_dim],
                        &mut state.xb2[..n_embd],
                        1,
                        &mut state.lora_scratch,
                    );
                }
                _ => unreachable!("LayerState variant always matches Lfm2OpWeights"),
            }

            if let Some(ref pan) = layer.post_attn_norm {
                rms_norm(
                    &mut state.xb[..n_embd],
                    &state.xb2[..n_embd],
                    pan,
                    self.cfg.rms_eps,
                );
                for i in 0..n_embd {
                    state.x[i] += state.xb[i];
                }
            } else {
                for i in 0..n_embd {
                    state.x[i] += state.xb2[i];
                }
            }

            // SwiGLU dense FFN (`lfm-base.cpp:95-105`)
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
            if !self.lora.is_empty() {
                self.lora.apply_linear(
                    layer.w_gate.lora_slot,
                    pool,
                    kt,
                    &state.xb[..n_embd],
                    &mut state.hb[..n_ff],
                    1,
                    &mut state.lora_scratch,
                );
                self.lora.apply_linear(
                    layer.w_up.lora_slot,
                    pool,
                    kt,
                    &state.xb[..n_embd],
                    &mut state.hb2[..n_ff],
                    1,
                    &mut state.lora_scratch,
                );
            }
            swiglu_inplace_parallel(pool, &mut state.hb[..n_ff], &state.hb2[..n_ff]);
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
            self.lora.apply_linear(
                layer.w_down.lora_slot,
                pool,
                kt,
                &state.hb[..n_ff],
                &mut state.xb2[..n_embd],
                1,
                &mut state.lora_scratch,
            );
            if let Some(ref pfn) = layer.post_ffw_norm {
                rms_norm(
                    &mut state.xb[..n_embd],
                    &state.xb2[..n_embd],
                    pfn,
                    self.cfg.rms_eps,
                );
                for i in 0..n_embd {
                    state.x[i] += state.xb[i];
                }
            } else {
                for i in 0..n_embd {
                    state.x[i] += state.xb2[i];
                }
            }
        }

        state.last_hidden.copy_from_slice(&state.x[..n_embd]);

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
            self.lora.apply_linear(
                self.output.lora_slot,
                pool,
                kt,
                &state.xb[..n_embd],
                &mut state.logits,
                1,
                &mut state.lora_scratch,
            );
        }

        Ok(())
    }

    /// Batched prefill forward pass for `token_ids` starting at `start_pos`.
    /// Validates all bounds BEFORE mutating any `LayerState` (`Rule #6`).
    pub fn forward_batch(
        &self,
        state: &mut Lfm2State,
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
        state: &mut Lfm2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) {
        let b = tokens.len();
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let hd = self.cfg.head_dim;
        let q_dim = self.cfg.n_head * hd;
        let l_cache = self.cfg.shortconv_l_cache;
        let hist_len = l_cache.saturating_sub(1);
        let attn_scale = 1.0 / (hd as f32).sqrt();

        for (t, &tok) in tokens.iter().enumerate() {
            decode_embedding_row(
                self.token_embd.dtype,
                self.token_embd.bytes,
                n_embd,
                tok as usize,
                &mut state.x[t * n_embd..(t + 1) * n_embd],
            );
            self.lora.apply_embed_row(
                self.token_embd.lora_slot,
                kt,
                tok as usize,
                &mut state.x[t * n_embd..(t + 1) * n_embd],
            );
        }

        for (il, layer) in self.layers.iter().enumerate() {
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

            match (&layer.op, &mut state.layers[il]) {
                (
                    Lfm2OpWeights::ShortConv {
                        in_proj,
                        conv,
                        out_proj,
                    },
                    LayerState::ShortConv { conv_hist },
                ) => {
                    matmul_f32(
                        pool,
                        kt,
                        in_proj.dtype,
                        in_proj.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.bcx_buf[..b * 3 * n_embd],
                        3 * n_embd,
                        n_embd,
                        b,
                    );
                    self.lora.apply_linear(
                        in_proj.lora_slot,
                        pool,
                        kt,
                        &state.xb[..b * n_embd],
                        &mut state.bcx_buf[..b * 3 * n_embd],
                        b,
                        &mut state.lora_scratch,
                    );

                    for t in 0..b {
                        let bcx_t = &state.bcx_buf[t * 3 * n_embd..(t + 1) * 3 * n_embd];
                        let y_t = &mut state.conv_y[t * n_embd..(t + 1) * n_embd];
                        for ch in 0..n_embd {
                            let bx_val = bcx_t[ch] * bcx_t[2 * n_embd + ch];
                            let c_val = bcx_t[n_embd + ch];
                            let h_slice = &mut conv_hist[ch * hist_len..(ch + 1) * hist_len];
                            let w_slice = &conv[ch * l_cache..(ch + 1) * l_cache];
                            let mut sum = 0.0f32;
                            for tap in 0..hist_len {
                                sum += h_slice[tap] * w_slice[tap];
                            }
                            sum += bx_val * w_slice[hist_len];
                            for tap in 0..hist_len.saturating_sub(1) {
                                h_slice[tap] = h_slice[tap + 1];
                            }
                            if hist_len > 0 {
                                h_slice[hist_len - 1] = bx_val;
                            }
                            y_t[ch] = c_val * sum;
                        }
                    }

                    matmul_f32(
                        pool,
                        kt,
                        out_proj.dtype,
                        out_proj.bytes,
                        &state.conv_y[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.xb2[..b * n_embd],
                        n_embd,
                        n_embd,
                        b,
                    );
                    self.lora.apply_linear(
                        out_proj.lora_slot,
                        pool,
                        kt,
                        &state.conv_y[..b * n_embd],
                        &mut state.xb2[..b * n_embd],
                        b,
                        &mut state.lora_scratch,
                    );
                }
                (
                    Lfm2OpWeights::Gqa {
                        n_head_kv,
                        q_norm,
                        k_norm,
                        wq,
                        wk,
                        wv,
                        w_attn_gate,
                        wo,
                    },
                    kv_state @ LayerState::Kv { .. },
                ) => {
                    let kv_dim = *n_head_kv * hd;
                    matmul_triple_f32(
                        pool,
                        kt,
                        wq.dtype,
                        wq.bytes,
                        q_dim,
                        &mut state.q_buf[..b * q_dim],
                        wk.dtype,
                        wk.bytes,
                        kv_dim,
                        &mut state.k_buf[..b * kv_dim],
                        wv.dtype,
                        wv.bytes,
                        kv_dim,
                        &mut state.v_buf[..b * kv_dim],
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                        b,
                    );
                    if !self.lora.is_empty() {
                        self.lora.apply_linear(
                            wq.lora_slot,
                            pool,
                            kt,
                            &state.xb[..b * n_embd],
                            &mut state.q_buf[..b * q_dim],
                            b,
                            &mut state.lora_scratch,
                        );
                        self.lora.apply_linear(
                            wk.lora_slot,
                            pool,
                            kt,
                            &state.xb[..b * n_embd],
                            &mut state.k_buf[..b * kv_dim],
                            b,
                            &mut state.lora_scratch,
                        );
                        self.lora.apply_linear(
                            wv.lora_slot,
                            pool,
                            kt,
                            &state.xb[..b * n_embd],
                            &mut state.v_buf[..b * kv_dim],
                            b,
                            &mut state.lora_scratch,
                        );
                    }
                    if let Some(wg) = w_attn_gate {
                        matmul_f32(
                            pool,
                            kt,
                            wg.dtype,
                            wg.bytes,
                            &state.xb[..b * n_embd],
                            &mut state.mv_scratch,
                            &mut state.gate_buf[..b * self.cfg.n_head],
                            self.cfg.n_head,
                            n_embd,
                            b,
                        );
                        self.lora.apply_linear(
                            wg.lora_slot,
                            pool,
                            kt,
                            &state.xb[..b * n_embd],
                            &mut state.gate_buf[..b * self.cfg.n_head],
                            b,
                            &mut state.lora_scratch,
                        );
                    }

                    for t in 0..b {
                        let pos = start_pos + t;
                        let qt = &mut state.q_buf[t * q_dim..(t + 1) * q_dim];
                        let kt_vec = &mut state.k_buf[t * kv_dim..(t + 1) * kv_dim];
                        let vt_vec = &state.v_buf[t * kv_dim..(t + 1) * kv_dim];

                        for h in 0..self.cfg.n_head {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(&mut state.xb2[..hd], &qt[s..e], q_norm, self.cfg.rms_eps);
                            qt[s..e].copy_from_slice(&state.xb2[..hd]);
                        }
                        for h in 0..*n_head_kv {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(
                                &mut state.xb2[..hd],
                                &kt_vec[s..e],
                                k_norm,
                                self.cfg.rms_eps,
                            );
                            kt_vec[s..e].copy_from_slice(&state.xb2[..hd]);
                        }

                        let inv_freqs = if self.cfg.is_swa[il] {
                            &self.rope_inv_freqs_swa
                        } else {
                            &self.rope_inv_freqs
                        };
                        rope_compute_cos_sin(
                            inv_freqs,
                            pos,
                            &mut state.rope_cos,
                            &mut state.rope_sin,
                        );
                        rope_inplace_tables(
                            qt,
                            self.cfg.n_head,
                            hd,
                            self.cfg.rope_dim,
                            &state.rope_cos,
                            &state.rope_sin,
                            RopeType::Neox,
                        );
                        rope_inplace_tables(
                            kt_vec,
                            *n_head_kv,
                            hd,
                            self.cfg.rope_dim,
                            &state.rope_cos,
                            &state.rope_sin,
                            RopeType::Neox,
                        );

                        kv_state.store_kv_at(kt, pos, kt_vec, vt_vec);
                    }

                    kv_state.attend_batch(
                        pool,
                        kt,
                        &mut state.attn_out[..b * q_dim],
                        &state.q_buf[..b * q_dim],
                        self.cfg.n_head,
                        *n_head_kv,
                        hd,
                        start_pos,
                        b,
                        attn_scale,
                        None,
                        state.max_ctx,
                        &mut state.scores,
                    );
                    if w_attn_gate.is_some() {
                        for t in 0..b {
                            for h in 0..self.cfg.n_head {
                                let g =
                                    1.0 / (1.0 + (-state.gate_buf[t * self.cfg.n_head + h]).exp());
                                let off = t * q_dim + h * hd;
                                for v in &mut state.attn_out[off..off + hd] {
                                    *v *= g;
                                }
                            }
                        }
                    }

                    matmul_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.attn_out[..b * q_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..b * n_embd],
                        n_embd,
                        q_dim,
                        b,
                    );
                    self.lora.apply_linear(
                        wo.lora_slot,
                        pool,
                        kt,
                        &state.attn_out[..b * q_dim],
                        &mut state.xb2[..b * n_embd],
                        b,
                        &mut state.lora_scratch,
                    );
                }
                _ => unreachable!("LayerState variant always matches Lfm2OpWeights"),
            }

            if let Some(ref pan) = layer.post_attn_norm {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..b * n_embd],
                    &state.xb2[..b * n_embd],
                    pan,
                    self.cfg.rms_eps,
                    n_embd,
                    b,
                );
                for i in 0..b * n_embd {
                    state.x[i] += state.xb[i];
                }
            } else {
                for i in 0..b * n_embd {
                    state.x[i] += state.xb2[i];
                }
            }

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
            matmul_pair_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                n_ff,
                &mut state.hb[..b * n_ff],
                layer.w_up.dtype,
                layer.w_up.bytes,
                n_ff,
                &mut state.hb2[..b * n_ff],
                &state.xb[..b * n_embd],
                &mut state.mv_scratch,
                n_embd,
                b,
            );
            if !self.lora.is_empty() {
                self.lora.apply_linear(
                    layer.w_gate.lora_slot,
                    pool,
                    kt,
                    &state.xb[..b * n_embd],
                    &mut state.hb[..b * n_ff],
                    b,
                    &mut state.lora_scratch,
                );
                self.lora.apply_linear(
                    layer.w_up.lora_slot,
                    pool,
                    kt,
                    &state.xb[..b * n_embd],
                    &mut state.hb2[..b * n_ff],
                    b,
                    &mut state.lora_scratch,
                );
            }
            swiglu_inplace_parallel(pool, &mut state.hb[..b * n_ff], &state.hb2[..b * n_ff]);
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
            self.lora.apply_linear(
                layer.w_down.lora_slot,
                pool,
                kt,
                &state.hb[..b * n_ff],
                &mut state.xb2[..b * n_embd],
                b,
                &mut state.lora_scratch,
            );
            if let Some(ref pfn) = layer.post_ffw_norm {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..b * n_embd],
                    &state.xb2[..b * n_embd],
                    pfn,
                    self.cfg.rms_eps,
                    n_embd,
                    b,
                );
                for i in 0..b * n_embd {
                    state.x[i] += state.xb[i];
                }
            } else {
                for i in 0..b * n_embd {
                    state.x[i] += state.xb2[i];
                }
            }
        }

        let last = (b - 1) * n_embd;
        state
            .last_hidden
            .copy_from_slice(&state.x[last..last + n_embd]);

        if compute_last_logits {
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[last..last + n_embd],
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
            self.lora.apply_linear(
                self.output.lora_slot,
                pool,
                kt,
                &state.xb[..n_embd],
                &mut state.logits,
                1,
                &mut state.lora_scratch,
            );
        }
    }

    /// Evaluates a speculative verification window `window_tokens` (`1 <= W <= MAX_SPEC_WINDOW`)
    /// while keeping `LayerState::ShortConv` **read-only** (`ADR-005`).
    pub fn verify_speculative(
        &self,
        state: &mut Lfm2State,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        window_tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), ModelError> {
        let w = window_tokens.len();
        assert!((1..=MAX_SPEC_WINDOW).contains(&w));
        let end_pos = start_pos.saturating_add(w);
        if end_pos > state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: end_pos - 1,
                max_ctx: state.max_ctx,
            });
        }
        for &tid in window_tokens {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }

        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let hd = self.cfg.head_dim;
        let q_dim = self.cfg.n_head * hd;
        let l_cache = self.cfg.shortconv_l_cache;
        let hist_len = l_cache.saturating_sub(1);
        let attn_scale = 1.0 / (hd as f32).sqrt();
        let vocab = self.cfg.vocab_size;

        for (t, &tok) in window_tokens.iter().enumerate() {
            decode_embedding_row(
                self.token_embd.dtype,
                self.token_embd.bytes,
                n_embd,
                tok as usize,
                &mut state.x[t * n_embd..(t + 1) * n_embd],
            );
            self.lora.apply_embed_row(
                self.token_embd.lora_slot,
                kt,
                tok as usize,
                &mut state.x[t * n_embd..(t + 1) * n_embd],
            );
        }

        let mut sc_idx = 0usize;
        for (il, layer) in self.layers.iter().enumerate() {
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..w * n_embd],
                &state.x[..w * n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
                n_embd,
                w,
            );

            match (&layer.op, &mut state.layers[il]) {
                (
                    Lfm2OpWeights::ShortConv {
                        in_proj,
                        conv,
                        out_proj,
                    },
                    LayerState::ShortConv { conv_hist },
                ) => {
                    spec_buf.tmp_conv_hist[..conv_hist.len()].copy_from_slice(conv_hist);
                    matmul_f32(
                        pool,
                        kt,
                        in_proj.dtype,
                        in_proj.bytes,
                        &state.xb[..w * n_embd],
                        &mut state.mv_scratch,
                        &mut state.bcx_buf[..w * 3 * n_embd],
                        3 * n_embd,
                        n_embd,
                        w,
                    );
                    self.lora.apply_linear(
                        in_proj.lora_slot,
                        pool,
                        kt,
                        &state.xb[..w * n_embd],
                        &mut state.bcx_buf[..w * 3 * n_embd],
                        w,
                        &mut state.lora_scratch,
                    );

                    let bx_layer_off = sc_idx * MAX_SPEC_WINDOW * n_embd;
                    for t in 0..w {
                        let bcx_t = &state.bcx_buf[t * 3 * n_embd..(t + 1) * 3 * n_embd];
                        let y_t = &mut state.conv_y[t * n_embd..(t + 1) * n_embd];
                        for ch in 0..n_embd {
                            let bx_val = bcx_t[ch] * bcx_t[2 * n_embd + ch];
                            spec_buf.sc_bx_win[bx_layer_off + t * n_embd + ch] = bx_val;
                            let c_val = bcx_t[n_embd + ch];
                            let h_slice =
                                &mut spec_buf.tmp_conv_hist[ch * hist_len..(ch + 1) * hist_len];
                            let w_slice = &conv[ch * l_cache..(ch + 1) * l_cache];
                            let mut sum = 0.0f32;
                            for tap in 0..hist_len {
                                sum += h_slice[tap] * w_slice[tap];
                            }
                            sum += bx_val * w_slice[hist_len];
                            for tap in 0..hist_len.saturating_sub(1) {
                                h_slice[tap] = h_slice[tap + 1];
                            }
                            if hist_len > 0 {
                                h_slice[hist_len - 1] = bx_val;
                            }
                            y_t[ch] = c_val * sum;
                        }
                    }
                    sc_idx += 1;

                    matmul_f32(
                        pool,
                        kt,
                        out_proj.dtype,
                        out_proj.bytes,
                        &state.conv_y[..w * n_embd],
                        &mut state.mv_scratch,
                        &mut state.xb2[..w * n_embd],
                        n_embd,
                        n_embd,
                        w,
                    );
                    self.lora.apply_linear(
                        out_proj.lora_slot,
                        pool,
                        kt,
                        &state.conv_y[..w * n_embd],
                        &mut state.xb2[..w * n_embd],
                        w,
                        &mut state.lora_scratch,
                    );
                }
                (
                    Lfm2OpWeights::Gqa {
                        n_head_kv,
                        q_norm,
                        k_norm,
                        wq,
                        wk,
                        wv,
                        w_attn_gate,
                        wo,
                    },
                    kv_state @ LayerState::Kv { .. },
                ) => {
                    let kv_dim = *n_head_kv * hd;
                    matmul_triple_f32(
                        pool,
                        kt,
                        wq.dtype,
                        wq.bytes,
                        q_dim,
                        &mut state.q_buf[..w * q_dim],
                        wk.dtype,
                        wk.bytes,
                        kv_dim,
                        &mut state.k_buf[..w * kv_dim],
                        wv.dtype,
                        wv.bytes,
                        kv_dim,
                        &mut state.v_buf[..w * kv_dim],
                        &state.xb[..w * n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                        w,
                    );
                    if !self.lora.is_empty() {
                        self.lora.apply_linear(
                            wq.lora_slot,
                            pool,
                            kt,
                            &state.xb[..w * n_embd],
                            &mut state.q_buf[..w * q_dim],
                            w,
                            &mut state.lora_scratch,
                        );
                        self.lora.apply_linear(
                            wk.lora_slot,
                            pool,
                            kt,
                            &state.xb[..w * n_embd],
                            &mut state.k_buf[..w * kv_dim],
                            w,
                            &mut state.lora_scratch,
                        );
                        self.lora.apply_linear(
                            wv.lora_slot,
                            pool,
                            kt,
                            &state.xb[..w * n_embd],
                            &mut state.v_buf[..w * kv_dim],
                            w,
                            &mut state.lora_scratch,
                        );
                    }
                    if let Some(wg) = w_attn_gate {
                        matmul_f32(
                            pool,
                            kt,
                            wg.dtype,
                            wg.bytes,
                            &state.xb[..w * n_embd],
                            &mut state.mv_scratch,
                            &mut state.gate_buf[..w * self.cfg.n_head],
                            self.cfg.n_head,
                            n_embd,
                            w,
                        );
                        self.lora.apply_linear(
                            wg.lora_slot,
                            pool,
                            kt,
                            &state.xb[..w * n_embd],
                            &mut state.gate_buf[..w * self.cfg.n_head],
                            w,
                            &mut state.lora_scratch,
                        );
                    }

                    for t in 0..w {
                        let pos = start_pos + t;
                        let qt = &mut state.q_buf[t * q_dim..(t + 1) * q_dim];
                        let kt_vec = &mut state.k_buf[t * kv_dim..(t + 1) * kv_dim];
                        let vt_vec = &state.v_buf[t * kv_dim..(t + 1) * kv_dim];

                        for h in 0..self.cfg.n_head {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(&mut state.xb2[..hd], &qt[s..e], q_norm, self.cfg.rms_eps);
                            qt[s..e].copy_from_slice(&state.xb2[..hd]);
                        }
                        for h in 0..*n_head_kv {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(
                                &mut state.xb2[..hd],
                                &kt_vec[s..e],
                                k_norm,
                                self.cfg.rms_eps,
                            );
                            kt_vec[s..e].copy_from_slice(&state.xb2[..hd]);
                        }

                        let inv_freqs = if self.cfg.is_swa[il] {
                            &self.rope_inv_freqs_swa
                        } else {
                            &self.rope_inv_freqs
                        };
                        rope_compute_cos_sin(
                            inv_freqs,
                            pos,
                            &mut state.rope_cos,
                            &mut state.rope_sin,
                        );
                        rope_inplace_tables(
                            qt,
                            self.cfg.n_head,
                            hd,
                            self.cfg.rope_dim,
                            &state.rope_cos,
                            &state.rope_sin,
                            RopeType::Neox,
                        );
                        rope_inplace_tables(
                            kt_vec,
                            *n_head_kv,
                            hd,
                            self.cfg.rope_dim,
                            &state.rope_cos,
                            &state.rope_sin,
                            RopeType::Neox,
                        );

                        kv_state.store_kv_at(kt, pos, kt_vec, vt_vec);
                    }

                    kv_state.attend_batch(
                        pool,
                        kt,
                        &mut state.attn_out[..w * q_dim],
                        &state.q_buf[..w * q_dim],
                        self.cfg.n_head,
                        *n_head_kv,
                        hd,
                        start_pos,
                        w,
                        attn_scale,
                        None,
                        state.max_ctx,
                        &mut state.scores,
                    );
                    if w_attn_gate.is_some() {
                        for t in 0..w {
                            for h in 0..self.cfg.n_head {
                                let g =
                                    1.0 / (1.0 + (-state.gate_buf[t * self.cfg.n_head + h]).exp());
                                let off = t * q_dim + h * hd;
                                for v in &mut state.attn_out[off..off + hd] {
                                    *v *= g;
                                }
                            }
                        }
                    }

                    matmul_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.attn_out[..w * q_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..w * n_embd],
                        n_embd,
                        q_dim,
                        w,
                    );
                    self.lora.apply_linear(
                        wo.lora_slot,
                        pool,
                        kt,
                        &state.attn_out[..w * q_dim],
                        &mut state.xb2[..w * n_embd],
                        w,
                        &mut state.lora_scratch,
                    );
                }
                _ => unreachable!("LayerState variant always matches Lfm2OpWeights"),
            }

            if let Some(ref pan) = layer.post_attn_norm {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..w * n_embd],
                    &state.xb2[..w * n_embd],
                    pan,
                    self.cfg.rms_eps,
                    n_embd,
                    w,
                );
                for i in 0..w * n_embd {
                    state.x[i] += state.xb[i];
                }
            } else {
                for i in 0..w * n_embd {
                    state.x[i] += state.xb2[i];
                }
            }

            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..w * n_embd],
                &state.x[..w * n_embd],
                &layer.ffn_norm,
                self.cfg.rms_eps,
                n_embd,
                w,
            );
            matmul_pair_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                n_ff,
                &mut state.hb[..w * n_ff],
                layer.w_up.dtype,
                layer.w_up.bytes,
                n_ff,
                &mut state.hb2[..w * n_ff],
                &state.xb[..w * n_embd],
                &mut state.mv_scratch,
                n_embd,
                w,
            );
            if !self.lora.is_empty() {
                self.lora.apply_linear(
                    layer.w_gate.lora_slot,
                    pool,
                    kt,
                    &state.xb[..w * n_embd],
                    &mut state.hb[..w * n_ff],
                    w,
                    &mut state.lora_scratch,
                );
                self.lora.apply_linear(
                    layer.w_up.lora_slot,
                    pool,
                    kt,
                    &state.xb[..w * n_embd],
                    &mut state.hb2[..w * n_ff],
                    w,
                    &mut state.lora_scratch,
                );
            }
            swiglu_inplace_parallel(pool, &mut state.hb[..w * n_ff], &state.hb2[..w * n_ff]);
            matmul_f32(
                pool,
                kt,
                layer.w_down.dtype,
                layer.w_down.bytes,
                &state.hb[..w * n_ff],
                &mut state.mv_scratch,
                &mut state.xb2[..w * n_embd],
                n_embd,
                n_ff,
                w,
            );
            self.lora.apply_linear(
                layer.w_down.lora_slot,
                pool,
                kt,
                &state.hb[..w * n_ff],
                &mut state.xb2[..w * n_embd],
                w,
                &mut state.lora_scratch,
            );
            if let Some(ref pfn) = layer.post_ffw_norm {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..w * n_embd],
                    &state.xb2[..w * n_embd],
                    pfn,
                    self.cfg.rms_eps,
                    n_embd,
                    w,
                );
                for i in 0..w * n_embd {
                    state.x[i] += state.xb[i];
                }
            } else {
                for i in 0..w * n_embd {
                    state.x[i] += state.xb2[i];
                }
            }
        }

        spec_buf.last_window_len = w;
        spec_buf.hidden_win[..w * n_embd].copy_from_slice(&state.x[..w * n_embd]);
        rms_norm_batch_parallel(
            pool,
            kt,
            &mut spec_buf.norm_win[..w * n_embd],
            &spec_buf.hidden_win[..w * n_embd],
            &self.output_norm,
            self.cfg.rms_eps,
            n_embd,
            w,
        );
        matmul_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes,
            &spec_buf.norm_win[..w * n_embd],
            &mut state.mv_scratch,
            &mut spec_buf.logits_win[..w * vocab],
            self.output.rows,
            self.output.cols,
            w,
        );
        self.lora.apply_linear(
            self.output.lora_slot,
            pool,
            kt,
            &spec_buf.norm_win[..w * n_embd],
            &mut spec_buf.logits_win[..w * vocab],
            w,
            &mut state.lora_scratch,
        );
        Ok(())
    }

    /// Commits `committed` (`1 <= committed <= W`) tokens from `verify_speculative` by shifting
    /// each `LayerState::ShortConv` history by `0..committed` (`ADR-005`).
    pub fn commit_speculative(
        &self,
        state: &mut Lfm2State,
        spec_buf: &SpecVerifyBuf,
        committed: usize,
    ) {
        assert!((1..=spec_buf.last_window_len).contains(&committed));
        let n_embd = self.cfg.n_embd;
        let hist_len = self.cfg.shortconv_l_cache.saturating_sub(1);
        let mut sc_idx = 0usize;
        if hist_len > 0 {
            for ls in &mut state.layers {
                if let LayerState::ShortConv { conv_hist } = ls {
                    let bx_layer_off = sc_idx * MAX_SPEC_WINDOW * n_embd;
                    for t in 0..committed {
                        for ch in 0..n_embd {
                            let bx_val = spec_buf.sc_bx_win[bx_layer_off + t * n_embd + ch];
                            let h_slice = &mut conv_hist[ch * hist_len..(ch + 1) * hist_len];
                            for tap in 0..hist_len - 1 {
                                h_slice[tap] = h_slice[tap + 1];
                            }
                            h_slice[hist_len - 1] = bx_val;
                        }
                    }
                    sc_idx += 1;
                }
            }
        }
        let idx = committed - 1;
        state
            .logits
            .copy_from_slice(spec_buf.logits_at(idx, self.cfg.vocab_size));
        state
            .last_hidden
            .copy_from_slice(spec_buf.hidden_at(idx, n_embd));
    }

    pub fn project_logits_from_hidden(
        &self,
        state: &mut Lfm2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        hidden: &[f32],
        out_logits: &mut [f32],
    ) {
        let n_embd = self.cfg.n_embd;
        rms_norm(
            &mut state.xb[..n_embd],
            hidden,
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
            out_logits,
            self.output.rows,
            self.output.cols,
        );
        self.lora.apply_linear(
            self.output.lora_slot,
            pool,
            kt,
            &state.xb[..n_embd],
            out_logits,
            1,
            &mut state.lora_scratch,
        );
    }

    pub fn embed_token(&self, token_id: u32, out_emb: &mut [f32]) {
        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes,
            self.cfg.n_embd,
            token_id as usize,
            out_emb,
        );
        if !self.lora.is_empty() {
            let kt = tm_cpu::active_kernels();
            self.lora
                .apply_embed_row(self.token_embd.lora_slot, kt, token_id as usize, out_emb);
        }
    }

    pub fn output_head_raw(&self) -> (&[f32], f32, DType, &'a [u8], Option<f32>) {
        (
            &self.output_norm,
            self.cfg.rms_eps,
            self.output.dtype,
            self.output.bytes,
            None,
        )
    }
}
