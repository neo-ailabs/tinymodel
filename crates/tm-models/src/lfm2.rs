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

use tm_core::{DType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool,
    gqa_attention_f16_kv_ext_batch_parallel, gqa_attention_f16_kv_ext_parallel, matmul_f32,
    matvec_f32, matvec_pair_f32, matvec_triple_f32, rms_norm, rms_norm_batch_parallel,
    rope_compute_cos_sin, rope_inplace_tables, swiglu_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};

use crate::llama::ModelError;
use crate::state::LayerState;

/// Hyperparameters for `general.architecture == "lfm2"`.
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
    pub vocab_size: usize,
    pub max_ctx_train: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
}

impl Lfm2Config {
    #[inline]
    pub fn is_shortconv(&self, il: usize) -> bool {
        self.n_head_kv_per_layer[il] == 0
    }

    #[inline]
    pub fn max_n_head_kv(&self) -> usize {
        self.n_head_kv_per_layer.iter().copied().max().unwrap_or(0)
    }
}

#[derive(Clone, Copy)]
struct MatView<'a> {
    bytes: &'a [u8],
    dtype: DType,
    rows: usize,
    cols: usize,
}

enum Lfm2OpWeights<'a> {
    ShortConv {
        in_proj: MatView<'a>,
        /// Depthwise causal conv weights `[n_embd, l_cache]` (`l_cache` contiguous floats per channel).
        conv: Vec<f32>,
        out_proj: MatView<'a>,
    },
    Gqa {
        n_head_kv: usize,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        wq: MatView<'a>,
        wk: MatView<'a>,
        wv: MatView<'a>,
        wo: MatView<'a>,
    },
}

struct Lfm2LayerWeights<'a> {
    attn_norm: Vec<f32>,
    op: Lfm2OpWeights<'a>,
    ffn_norm: Vec<f32>,
    w_gate: MatView<'a>,
    w_up: MatView<'a>,
    w_down: MatView<'a>,
}

/// Loaded LFM2 hybrid model borrowing quantized weight slices zero-copy from `GgufFile`.
pub struct Lfm2Model<'a> {
    pub cfg: Lfm2Config,
    token_embd: MatView<'a>,
    output: MatView<'a>,
    output_norm: Vec<f32>,
    rope_inv_freqs: Vec<f64>,
    layers: Vec<Lfm2LayerWeights<'a>>,
}

/// Pre-allocated per-session inference state for LFM2 (`LayerState::ShortConv` + `LayerState::Kv`).
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
    attn_out: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    mv_scratch: MatvecScratch,
}

impl Lfm2State {
    pub fn new(model: &Lfm2Model<'_>, max_ctx: usize) -> Self {
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
                layers.push(LayerState::Kv {
                    k_cache: vec![0u16; max_ctx * kv_dim],
                    v_cache: vec![0u16; max_ctx * kv_dim],
                    kv_dim,
                    window: cfg.n_swa,
                });
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
            attn_out: vec![0.0; MAX_CPU_BATCH * q_dim.max(1)],
            hb: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            hb2: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            rope_cos: vec![0.0; cfg.rope_dim / 2],
            rope_sin: vec![0.0; cfg.rope_dim / 2],
            scores: vec![0.0; cfg.n_head.max(1) * max_ctx],
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
        bytes,
        dtype: info.dtype,
        rows,
        cols,
    })
}

impl<'a> Lfm2Model<'a> {
    /// Loads and strictly validates an `lfm2` GGUF model.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "lfm2" {
            return Err(ModelError::UnsupportedArch(arch));
        }

        let pfx = "lfm2";
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
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
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
            token_embd
        };
        // LFM2 stores output RMSNorm under `token_embd_norm.weight` (`llama-arch.cpp:535`)
        let output_norm = if gguf.tensor_info("token_embd_norm.weight").is_some() {
            load_f32_vec(gguf, "token_embd_norm.weight", "global", &[n_embd])?
        } else {
            load_f32_vec(gguf, "output_norm.weight", "global", &[n_embd])?
        };

        let rope_inv_freqs: Vec<f64> = (0..(rope_dim / 2))
            .map(|i| (rope_freq_base as f64).powf(-2.0 * (i as f64) / (rope_dim as f64)))
            .collect();

        let q_dim = n_head * head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for (il, &n_head_kv) in n_head_kv_per_layer.iter().enumerate() {
            let ltag = format!("blk.{il}");
            let attn_norm =
                load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, &[n_embd])?;
            let ffn_norm =
                load_f32_vec(gguf, &format!("{ltag}.ffn_norm.weight"), &ltag, &[n_embd])?;

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
            vocab_size,
            max_ctx_train,
            rope_freq_base,
            rms_eps,
        };

        Ok(Self {
            cfg,
            token_embd,
            output,
            output_norm,
            rope_inv_freqs,
            layers,
        })
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

        let emb_row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        let emb_row = &self.token_embd.bytes[tid * emb_row_bytes..(tid + 1) * emb_row_bytes];
        dequantize_row_bytes(self.token_embd.dtype, emb_row, &mut state.x[..n_embd]);

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
                }
                (
                    Lfm2OpWeights::Gqa {
                        n_head_kv,
                        q_norm,
                        k_norm,
                        wq,
                        wk,
                        wv,
                        wo,
                    },
                    LayerState::Kv {
                        k_cache,
                        v_cache,
                        window,
                        ..
                    },
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

                    rope_inplace_tables(
                        &mut state.q_buf[..q_dim],
                        self.cfg.n_head,
                        hd,
                        self.cfg.rope_dim,
                        &state.rope_cos,
                        &state.rope_sin,
                        RopeType::Neox,
                    );
                    rope_inplace_tables(
                        &mut state.k_buf[..kv_dim],
                        *n_head_kv,
                        hd,
                        self.cfg.rope_dim,
                        &state.rope_cos,
                        &state.rope_sin,
                        RopeType::Neox,
                    );

                    let cache_off = pos * kv_dim;
                    (kt.f32_to_f16_slice)(
                        &state.k_buf[..kv_dim],
                        &mut k_cache[cache_off..cache_off + kv_dim],
                    );
                    (kt.f32_to_f16_slice)(
                        &state.v_buf[..kv_dim],
                        &mut v_cache[cache_off..cache_off + kv_dim],
                    );

                    gqa_attention_f16_kv_ext_parallel(
                        pool,
                        kt,
                        &mut state.attn_out[..q_dim],
                        &state.q_buf[..q_dim],
                        k_cache,
                        v_cache,
                        self.cfg.n_head,
                        *n_head_kv,
                        hd,
                        pos,
                        attn_scale,
                        *window,
                        state.max_ctx,
                        &mut state.scores,
                    );

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
                }
                _ => unreachable!("LayerState variant always matches Lfm2OpWeights"),
            }

            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
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
            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
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

        let emb_row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        for (t, &tok) in tokens.iter().enumerate() {
            let tid = tok as usize;
            let emb_row = &self.token_embd.bytes[tid * emb_row_bytes..(tid + 1) * emb_row_bytes];
            dequantize_row_bytes(
                self.token_embd.dtype,
                emb_row,
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
                }
                (
                    Lfm2OpWeights::Gqa {
                        n_head_kv,
                        q_norm,
                        k_norm,
                        wq,
                        wk,
                        wv,
                        wo,
                    },
                    LayerState::Kv {
                        k_cache,
                        v_cache,
                        window,
                        ..
                    },
                ) => {
                    let kv_dim = *n_head_kv * hd;
                    matmul_f32(
                        pool,
                        kt,
                        wq.dtype,
                        wq.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.q_buf[..b * q_dim],
                        q_dim,
                        n_embd,
                        b,
                    );
                    matmul_f32(
                        pool,
                        kt,
                        wk.dtype,
                        wk.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.k_buf[..b * kv_dim],
                        kv_dim,
                        n_embd,
                        b,
                    );
                    matmul_f32(
                        pool,
                        kt,
                        wv.dtype,
                        wv.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.v_buf[..b * kv_dim],
                        kv_dim,
                        n_embd,
                        b,
                    );

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

                        rope_compute_cos_sin(
                            &self.rope_inv_freqs,
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

                        let cache_off = pos * kv_dim;
                        (kt.f32_to_f16_slice)(kt_vec, &mut k_cache[cache_off..cache_off + kv_dim]);
                        (kt.f32_to_f16_slice)(vt_vec, &mut v_cache[cache_off..cache_off + kv_dim]);
                    }

                    gqa_attention_f16_kv_ext_batch_parallel(
                        pool,
                        kt,
                        &mut state.attn_out[..b * q_dim],
                        &state.q_buf[..b * q_dim],
                        k_cache,
                        v_cache,
                        self.cfg.n_head,
                        *n_head_kv,
                        hd,
                        start_pos,
                        b,
                        attn_scale,
                        *window,
                        state.max_ctx,
                        &mut state.scores,
                    );

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
                }
                _ => unreachable!("LayerState variant always matches Lfm2OpWeights"),
            }

            for i in 0..b * n_embd {
                state.x[i] += state.xb2[i];
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
            for i in 0..b * n_embd {
                state.x[i] += state.xb2[i];
            }
        }

        if compute_last_logits {
            let last = (b - 1) * n_embd;
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
        }
    }
}
