//! Qwen3.5 hybrid Gated DeltaNet (GDN) + Gated Full-Attention forward pass.
//!
//! Strictly implements `Strata/docs/GDN_CONTRACT.md`, `third_party/llama.cpp/src/models/qwen35.cpp`,
//! and all invariants in `AGENTS.md` §5 & §9:
//! - Hybrid layer schedule: GDN linear attention (`attn_qkv` + `ssm_*`) interleaved with
//!   gated full GQA attention (`attn_q` outputting interleaved `[Q_h, Gate_h]` per head,
//!   partial NeoX M-RoPE on `rope_dim = 64`, `sigmoid` output gate).
//! - GDN invariants:
//!   1. Causal `conv1d` over `qkv` followed by **SiLU BEFORE splitting** `q`, `k`, `v`.
//!   2. Per-head L2-normalization on `q` and `k` ONLY (**never** normalize `v`).
//!   3. Tiled head grouping `kh = vh % n_k_heads` (**never** contiguous repeat).
//!   4. Output query scaled by `1 / sqrt(head_v_dim)` on the updated state.
//!   5. Per-head gated RMSNorm with **SiLU** on `z` (`rms_norm(o_h, ssm_norm) * silu(z_h)`).
//! - Zero heap allocations on `forward_token` / `forward_batch`.
//! - State safety (`AGENTS.md` Rule #6): all bounds and token IDs are validated BEFORE
//!   mutating any `LayerState` (`Kv` or `DeltaNet`).

use std::time::Instant;

use tm_core::{DType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool,
    gdn_conv1d_silu_split_parallel, gdn_recurrence_gated_parallel,
    gqa_attention_f16_kv_batch_parallel, gqa_attention_f16_kv_parallel, matmul_f32, matvec_f32,
    matvec_pair_f32, matvec_triple_f32, rms_norm, rms_norm_batch_parallel, rope_compute_cos_sin,
    rope_inplace_tables, swiglu_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};

use crate::llama::ModelError;

/// Hyperparameters for `general.architecture == "qwen35"`.
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    /// Full-attention query heads (`qwen35.attention.head_count`, e.g. 8).
    pub n_head: usize,
    /// Full-attention KV heads (`qwen35.attention.head_count_kv`, e.g. 2).
    pub n_head_kv: usize,
    /// Full-attention head dimension (`qwen35.attention.key_length`, e.g. 256).
    pub head_dim: usize,
    /// Partial RoPE rotated dimension (`qwen35.rope.dimension_count`, e.g. 64).
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub full_attention_interval: usize,
    /// GDN conv kernel width (`qwen35.ssm.conv_kernel`, e.g. 4).
    pub ssm_conv_kernel: usize,
    /// GDN state & K/V head dimension (`qwen35.ssm.state_size`, e.g. 128).
    pub ssm_state_size: usize,
    /// GDN K/Q head count (`qwen35.ssm.group_count`, e.g. 16).
    pub ssm_n_group: usize,
    /// GDN V head count (`qwen35.ssm.time_step_rank`, e.g. 16).
    pub ssm_dt_rank: usize,
    /// GDN inner size (`qwen35.ssm.inner_size`, e.g. 2048 = `ssm_dt_rank * ssm_state_size`).
    pub ssm_d_inner: usize,
    pub vocab_size: usize,
    pub max_ctx_train: usize,
}

impl Qwen35Config {
    /// Total channels in `attn_qkv` and `ssm_conv1d`: `2 * HK * S + d_inner`.
    #[inline]
    pub const fn conv_dim(&self) -> usize {
        2 * self.ssm_n_group * self.ssm_state_size + self.ssm_d_inner
    }

    /// Value head dimension `S_v = d_inner / HV` (equals `ssm_state_size` in Qwen3.5).
    #[inline]
    pub const fn ssm_head_v_dim(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank
    }
}

/// Per-layer heterogeneous inference state (`docs/ARCHITECTURE.md` §5.2).
#[derive(Clone)]
pub enum LayerState {
    /// Full GQA attention KV cache (`[max_ctx, n_head_kv * head_dim]` in IEEE-754 F16).
    Kv {
        k_cache: Vec<u16>,
        v_cache: Vec<u16>,
    },
    /// Gated DeltaNet recurrent state (`[n_v_heads, S, S]` transposed) + causal Conv1d history
    /// (`[conv_dim, conv_kernel - 1]` storing the `conv_kernel - 1` most recent raw `qkv` steps).
    DeltaNet {
        state: Vec<f32>,
        conv_hist: Vec<f32>,
    },
}

impl LayerState {
    /// Zeroes out the layer state in-place without reallocating.
    pub fn reset(&mut self) {
        match self {
            LayerState::Kv { k_cache, v_cache } => {
                k_cache.fill(0);
                v_cache.fill(0);
            }
            LayerState::DeltaNet { state, conv_hist } => {
                state.fill(0.0);
                conv_hist.fill(0.0);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct MatView<'a> {
    bytes: &'a [u8],
    dtype: DType,
    rows: usize,
    cols: usize,
}

enum AttnWeights<'a> {
    FullAttn {
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        /// Joint Q+Gate projection `[n_embd -> n_head * head_dim * 2]`.
        wqg: MatView<'a>,
        wk: MatView<'a>,
        wv: MatView<'a>,
        wo: MatView<'a>,
    },
    Gdn {
        /// Joint QKV projection `[n_embd -> conv_dim]`.
        wqkv: MatView<'a>,
        /// Output gate `z` projection `[n_embd -> ssm_d_inner]`.
        wz: MatView<'a>,
        w_alpha: MatView<'a>,
        w_beta: MatView<'a>,
        /// Depthwise causal conv weights `[conv_dim, conv_kernel]` (`conv_kernel` contiguous floats per channel).
        conv1d: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_dt_bias: Vec<f32>,
        ssm_norm: Vec<f32>,
        w_out: MatView<'a>,
    },
}

struct Qwen35LayerWeights<'a> {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    attn: AttnWeights<'a>,
    w_gate: MatView<'a>,
    w_up: MatView<'a>,
    w_down: MatView<'a>,
}

/// Loaded Qwen3.5 hybrid model borrowing quantized weight slices directly from `GgufFile`.
pub struct Qwen35Model<'a> {
    pub cfg: Qwen35Config,
    token_embd: MatView<'a>,
    output: MatView<'a>,
    output_norm: Vec<f32>,
    rope_inv_freqs: Vec<f64>,
    layers: Vec<Qwen35LayerWeights<'a>>,
}

/// Pre-allocated per-session inference state for Qwen3.5 (`LayerState` + Arena-like scratch).
/// Zero heap allocations occur inside `forward_token` or `forward_batch`.
pub struct Qwen35State {
    pub max_ctx: usize,
    pub layers: Vec<LayerState>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    /// Holds `qg_full` (`[B, n_head * head_dim * 2]`) on full-attn layers or `qkv` (`[B, conv_dim]`) on GDN layers.
    qkv_buf: Vec<f32>,
    /// Holds `q` (`[B, q_dim]`) on full-attn layers or `q_conv` (`[B, HK * S]`) on GDN layers.
    q_buf: Vec<f32>,
    /// Holds `k` (`[B, kv_dim]`) on full-attn layers or `k_conv` (`[B, HK * S]`) on GDN layers.
    k_buf: Vec<f32>,
    /// Holds `v` (`[B, kv_dim]`) on full-attn layers or `v_conv` (`[B, HV * S]`) on GDN layers.
    v_buf: Vec<f32>,
    /// Holds `attn_gate` (`[B, q_dim]`) on full-attn layers or `z` (`[B, d_inner]`) on GDN layers.
    gate_buf: Vec<f32>,
    /// Holds attention / GDN pre-projection output (`[B, max(q_dim, d_inner)]`).
    attn_out: Vec<f32>,
    alpha_buf: Vec<f32>,
    beta_buf: Vec<f32>,
    g_exp_buf: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    mv_scratch: MatvecScratch,
}

impl Qwen35State {
    /// Allocates heterogeneous `LayerState`s (`Kv` only on full-attention layers, `DeltaNet`
    /// only on GDN layers) and all scratch buffers up front for `max_ctx` tokens.
    pub fn new(model: &Qwen35Model<'_>, max_ctx: usize) -> Self {
        let cfg = &model.cfg;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let q_dim = cfg.n_head * cfg.head_dim;
        let qg_dim = q_dim * 2;
        let conv_dim = cfg.conv_dim();
        let k_gdn_dim = cfg.ssm_n_group * cfg.ssm_state_size;
        let v_gdn_dim = cfg.ssm_d_inner;
        let hist_len = cfg.ssm_conv_kernel.saturating_sub(1);
        let state_elems = cfg.ssm_dt_rank * cfg.ssm_state_size * cfg.ssm_head_v_dim();

        let layers = model
            .layers
            .iter()
            .map(|lw| match lw.attn {
                AttnWeights::FullAttn { .. } => LayerState::Kv {
                    k_cache: vec![0u16; max_ctx * kv_dim],
                    v_cache: vec![0u16; max_ctx * kv_dim],
                },
                AttnWeights::Gdn { .. } => LayerState::DeltaNet {
                    state: vec![0.0f32; state_elems],
                    conv_hist: vec![0.0f32; conv_dim * hist_len],
                },
            })
            .collect();

        let max_cols = cfg.n_embd.max(cfg.n_ff).max(q_dim).max(v_gdn_dim);

        Self {
            max_ctx,
            layers,
            x: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            xb2: vec![0.0; MAX_CPU_BATCH * cfg.n_embd],
            qkv_buf: vec![0.0; MAX_CPU_BATCH * qg_dim.max(conv_dim)],
            q_buf: vec![0.0; MAX_CPU_BATCH * q_dim.max(k_gdn_dim)],
            k_buf: vec![0.0; MAX_CPU_BATCH * kv_dim.max(k_gdn_dim)],
            v_buf: vec![0.0; MAX_CPU_BATCH * kv_dim.max(v_gdn_dim)],
            gate_buf: vec![0.0; MAX_CPU_BATCH * q_dim.max(v_gdn_dim)],
            attn_out: vec![0.0; MAX_CPU_BATCH * q_dim.max(v_gdn_dim)],
            alpha_buf: vec![0.0; MAX_CPU_BATCH * cfg.ssm_dt_rank],
            beta_buf: vec![0.0; MAX_CPU_BATCH * cfg.ssm_dt_rank],
            g_exp_buf: vec![0.0; MAX_CPU_BATCH * cfg.ssm_dt_rank],
            hb: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            hb2: vec![0.0; MAX_CPU_BATCH * cfg.n_ff],
            rope_cos: vec![0.0; cfg.rope_dim / 2],
            rope_sin: vec![0.0; cfg.rope_dim / 2],
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

/// CPU operator timing breakdown for Qwen3.5 hybrid decode.
#[derive(Debug, Default, Clone, Copy)]
pub struct Qwen35DecodeProfile {
    pub n_steps: usize,
    pub embed_ns: u64,
    pub rms_norm_ns: u64,
    pub gdn_proj_ns: u64,
    pub gdn_conv_norm_ns: u64,
    pub gdn_recur_ns: u64,
    pub gdn_out_ns: u64,
    pub full_qkv_ns: u64,
    pub full_attn_ns: u64,
    pub full_wo_ns: u64,
    pub ffn_gate_up_ns: u64,
    pub ffn_down_ns: u64,
    pub lm_head_ns: u64,
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

#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl<'a> Qwen35Model<'a> {
    /// Loads and strictly validates a `qwen35` GGUF model (zero-copy weight views).
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "qwen35" {
            return Err(ModelError::UnsupportedArch(arch));
        }

        let pfx = "qwen35";
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
            .unwrap_or(256);
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(64);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000_000.0);
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let full_attention_interval = gguf
            .get_meta(&format!("{pfx}.full_attention_interval"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4);
        let ssm_conv_kernel = gguf.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_state_size = gguf.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_group = gguf.req_usize(&format!("{pfx}.ssm.group_count"))?;
        let ssm_dt_rank = gguf.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_d_inner = gguf.req_usize(&format!("{pfx}.ssm.inner_size"))?;
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(32768);

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

        let cfg = Qwen35Config {
            arch,
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_eps,
            full_attention_interval,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            vocab_size,
            max_ctx_train,
        };

        let token_embd = load_mat(gguf, "token_embd.weight", "global", n_embd, vocab_size)?;
        let output = if gguf.tensor_info("output.weight").is_some() {
            load_mat(gguf, "output.weight", "global", n_embd, vocab_size)?
        } else {
            token_embd
        };
        let output_norm = load_f32_vec(gguf, "output_norm.weight", "global", &[n_embd])?;

        let half = rope_dim / 2;
        let base_f = rope_freq_base as f64;
        let dim_f = rope_dim as f64;
        let mut rope_inv_freqs = Vec::with_capacity(half);
        for i in 0..half {
            rope_inv_freqs.push(base_f.powf(-2.0 * (i as f64) / dim_f));
        }

        let q_dim = n_head * head_dim;
        let qg_dim = q_dim * 2;
        let kv_dim = n_head_kv * head_dim;
        let conv_dim = cfg.conv_dim();
        let head_v_dim = cfg.ssm_head_v_dim();

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm =
                load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, &[n_embd])?;
            let post_norm_name = if gguf
                .tensor_info(&format!("{ltag}.post_attention_norm.weight"))
                .is_some()
            {
                format!("{ltag}.post_attention_norm.weight")
            } else {
                format!("{ltag}.ffn_norm.weight")
            };
            let post_attn_norm = load_f32_vec(gguf, &post_norm_name, &ltag, &[n_embd])?;

            let is_gdn = gguf
                .tensor_info(&format!("{ltag}.attn_qkv.weight"))
                .is_some();
            let attn = if is_gdn {
                AttnWeights::Gdn {
                    wqkv: load_mat(
                        gguf,
                        &format!("{ltag}.attn_qkv.weight"),
                        &ltag,
                        n_embd,
                        conv_dim,
                    )?,
                    wz: load_mat(
                        gguf,
                        &format!("{ltag}.attn_gate.weight"),
                        &ltag,
                        n_embd,
                        ssm_d_inner,
                    )?,
                    w_alpha: load_mat(
                        gguf,
                        &format!("{ltag}.ssm_alpha.weight"),
                        &ltag,
                        n_embd,
                        ssm_dt_rank,
                    )?,
                    w_beta: load_mat(
                        gguf,
                        &format!("{ltag}.ssm_beta.weight"),
                        &ltag,
                        n_embd,
                        ssm_dt_rank,
                    )?,
                    conv1d: load_f32_vec(
                        gguf,
                        &format!("{ltag}.ssm_conv1d.weight"),
                        &ltag,
                        &[ssm_conv_kernel, conv_dim],
                    )?,
                    ssm_a: load_f32_vec(gguf, &format!("{ltag}.ssm_a"), &ltag, &[ssm_dt_rank])?,
                    ssm_dt_bias: load_f32_vec(
                        gguf,
                        &format!("{ltag}.ssm_dt.bias"),
                        &ltag,
                        &[ssm_dt_rank],
                    )?,
                    ssm_norm: load_f32_vec(
                        gguf,
                        &format!("{ltag}.ssm_norm.weight"),
                        &ltag,
                        &[head_v_dim],
                    )?,
                    w_out: load_mat(
                        gguf,
                        &format!("{ltag}.ssm_out.weight"),
                        &ltag,
                        ssm_d_inner,
                        n_embd,
                    )?,
                }
            } else {
                AttnWeights::FullAttn {
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
                    wqg: load_mat(
                        gguf,
                        &format!("{ltag}.attn_q.weight"),
                        &ltag,
                        n_embd,
                        qg_dim,
                    )?,
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

            layers.push(Qwen35LayerWeights {
                attn_norm,
                post_attn_norm,
                attn,
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

    /// Number of GDN linear-attention layers in this model.
    pub fn num_gdn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.attn, AttnWeights::Gdn { .. }))
            .count()
    }

    /// Number of full GQA attention layers in this model.
    pub fn num_full_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.attn, AttnWeights::FullAttn { .. }))
            .count()
    }

    /// Single-token forward pass at sequence position `pos` (zero heap allocations).
    ///
    /// Validates `pos` and `token_id` **before** modifying any `LayerState` (`AGENTS.md` Rule #6).
    pub fn forward_token(
        &self,
        state: &mut Qwen35State,
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
        let qg_dim = q_dim * 2;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let conv_dim = self.cfg.conv_dim();
        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = self.cfg.ssm_d_inner;
        let d_conv = self.cfg.ssm_conv_kernel;

        rope_compute_cos_sin(
            &self.rope_inv_freqs,
            pos,
            &mut state.rope_cos,
            &mut state.rope_sin,
        );

        let row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        let emb_row = &self.token_embd.bytes[tid * row_bytes..(tid + 1) * row_bytes];
        dequantize_row_bytes(self.token_embd.dtype, emb_row, &mut state.x[..n_embd]);

        for (l, layer) in self.layers.iter().enumerate() {
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
            );

            match (&layer.attn, &mut state.layers[l]) {
                (
                    AttnWeights::Gdn {
                        wqkv,
                        wz,
                        w_alpha,
                        w_beta,
                        conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        w_out,
                    },
                    LayerState::DeltaNet {
                        state: ssm_state,
                        conv_hist,
                    },
                ) => {
                    // 1. Fused QKV + Z projection (both Q4_0 -> single quantize + barrier)
                    matvec_pair_f32(
                        pool,
                        kt,
                        wqkv.dtype,
                        wqkv.bytes,
                        wqkv.rows,
                        &mut state.qkv_buf[..conv_dim],
                        wz.dtype,
                        wz.bytes,
                        wz.rows,
                        &mut state.gate_buf[..v_gdn_dim],
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                    );
                    matvec_pair_f32(
                        pool,
                        kt,
                        w_alpha.dtype,
                        w_alpha.bytes,
                        w_alpha.rows,
                        &mut state.alpha_buf[..hv],
                        w_beta.dtype,
                        w_beta.bytes,
                        w_beta.rows,
                        &mut state.beta_buf[..hv],
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                    );

                    // 2. Compute GDN scalar gates per value head
                    for vh in 0..hv {
                        let sp = softplus_f32(state.alpha_buf[vh] + ssm_dt_bias[vh]);
                        let gate = sp * ssm_a[vh];
                        state.g_exp_buf[vh] = gate.exp();
                        state.beta_buf[vh] = sigmoid_f32(state.beta_buf[vh]);
                    }

                    // 3 & 4. Parallel causal Conv1d + SiLU BEFORE q/k/v split + per-head q/k L2-norm (`AGENTS.md` §9)
                    gdn_conv1d_silu_split_parallel(
                        pool,
                        kt,
                        &state.qkv_buf[..conv_dim],
                        conv_hist,
                        conv1d,
                        &mut state.q_buf[..k_gdn_dim],
                        &mut state.k_buf[..k_gdn_dim],
                        &mut state.v_buf[..v_gdn_dim],
                        d_conv,
                        hk,
                        s_dim,
                        v_gdn_dim,
                        self.cfg.rms_eps,
                        1,
                    );

                    // 5 & 6. Parallel NEON GDN recurrence + L1-fused gated RMSNorm * SiLU(z) (`kh = vh % hk`)
                    gdn_recurrence_gated_parallel(
                        pool,
                        kt,
                        &state.q_buf[..k_gdn_dim],
                        &state.k_buf[..k_gdn_dim],
                        &state.v_buf[..v_gdn_dim],
                        &state.g_exp_buf[..hv],
                        &state.beta_buf[..hv],
                        ssm_state,
                        &mut state.attn_out[..v_gdn_dim],
                        &state.gate_buf[..v_gdn_dim],
                        ssm_norm,
                        self.cfg.rms_eps,
                        hk,
                        hv,
                        s_dim,
                        1,
                    );

                    // 7. Output projection (`ssm_out`, Q5_K) + residual add
                    matvec_f32(
                        pool,
                        kt,
                        w_out.dtype,
                        w_out.bytes,
                        &state.attn_out[..v_gdn_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..n_embd],
                        w_out.rows,
                        w_out.cols,
                    );
                }
                (
                    AttnWeights::FullAttn {
                        q_norm,
                        k_norm,
                        wqg,
                        wk,
                        wv,
                        wo,
                    },
                    LayerState::Kv { k_cache, v_cache },
                ) => {
                    matvec_triple_f32(
                        pool,
                        kt,
                        wqg.dtype,
                        wqg.bytes,
                        wqg.rows,
                        &mut state.qkv_buf[..qg_dim],
                        wk.dtype,
                        wk.bytes,
                        wk.rows,
                        &mut state.k_buf[..kv_dim],
                        wv.dtype,
                        wv.bytes,
                        wv.rows,
                        &mut state.v_buf[..kv_dim],
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                    );

                    // De-interleave `[Q_h (hd), Gate_h (hd)]` per head + apply per-head Q/K RMSNorm
                    let hd = self.cfg.head_dim;
                    for h in 0..self.cfg.n_head {
                        let src_q = &state.qkv_buf[h * 2 * hd..h * 2 * hd + hd];
                        let src_g = &state.qkv_buf[h * 2 * hd + hd..(h + 1) * 2 * hd];
                        rms_norm(
                            &mut state.q_buf[h * hd..(h + 1) * hd],
                            src_q,
                            q_norm,
                            self.cfg.rms_eps,
                        );
                        state.gate_buf[h * hd..(h + 1) * hd].copy_from_slice(src_g);
                    }
                    for h in 0..self.cfg.n_head_kv {
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

                    // Partial NeoX RoPE on first `rope_dim` elements of each Q and K head
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
                        self.cfg.n_head_kv,
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

                    gqa_attention_f16_kv_parallel(
                        pool,
                        kt,
                        &mut state.attn_out[..q_dim],
                        &state.q_buf[..q_dim],
                        k_cache,
                        v_cache,
                        self.cfg.n_head,
                        self.cfg.n_head_kv,
                        hd,
                        pos + 1,
                        state.max_ctx,
                        &mut state.scores,
                    );

                    // Full-attention output gate uses `sigmoid` (`qwen35.cpp:345`)
                    for (o, &g) in state.attn_out[..q_dim]
                        .iter_mut()
                        .zip(state.gate_buf[..q_dim].iter())
                    {
                        *o *= sigmoid_f32(g);
                    }

                    matvec_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.attn_out[..q_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..n_embd],
                        wo.rows,
                        wo.cols,
                    );
                }
                _ => unreachable!("LayerState variant always matches Qwen35LayerWeights::attn"),
            }

            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
            }

            // Post-attention RMSNorm + SwiGLU FFN
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.post_attn_norm,
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
    ///
    /// Uses `smmla`/`sdot` batched `matmul_f32` (`MAX_CPU_BATCH = 32` per chunk) and
    /// validates the entire `token_ids` slice and context bounds BEFORE mutating state.
    pub fn forward_batch(
        &self,
        state: &mut Qwen35State,
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
        state: &mut Qwen35State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        prof: &mut Qwen35DecodeProfile,
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
        state: &mut Qwen35State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        mut prof: Option<&mut Qwen35DecodeProfile>,
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
        for &token_id in token_ids {
            if (token_id as usize) >= self.cfg.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token_id,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }

        if token_ids.len() == 1 {
            if let Some(p) = prof {
                return self.forward_token_profiled(state, pool, kt, token_ids[0], start_pos, p);
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
                prof.as_deref_mut(),
            );
            offset += b;
        }
        if let Some(p) = prof {
            p.n_steps += token_ids.len();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk(
        &self,
        state: &mut Qwen35State,
        pool: &ThreadPool,
        kt: &KernelTable,
        tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        mut prof: Option<&mut Qwen35DecodeProfile>,
    ) {
        let b = tokens.len();
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let hd = self.cfg.head_dim;
        let q_dim = self.cfg.n_head * hd;
        let qg_dim = q_dim * 2;
        let kv_dim = self.cfg.n_head_kv * hd;
        let conv_dim = self.cfg.conv_dim();
        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = self.cfg.ssm_d_inner;
        let d_conv = self.cfg.ssm_conv_kernel;

        let t_emb = prof.as_ref().map(|_| Instant::now());
        let row_bytes = self.token_embd.dtype.row_bytes(n_embd);
        for (t, &tok) in tokens.iter().enumerate() {
            let tid = tok as usize;
            let emb_row = &self.token_embd.bytes[tid * row_bytes..(tid + 1) * row_bytes];
            dequantize_row_bytes(
                self.token_embd.dtype,
                emb_row,
                &mut state.x[t * n_embd..(t + 1) * n_embd],
            );
        }
        if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_emb) {
            p.embed_ns += ts.elapsed().as_nanos() as u64;
        }

        for (l, layer) in self.layers.iter().enumerate() {
            let t_rn = prof.as_ref().map(|_| Instant::now());
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
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_rn) {
                p.rms_norm_ns += ts.elapsed().as_nanos() as u64;
            }

            match (&layer.attn, &mut state.layers[l]) {
                (
                    AttnWeights::Gdn {
                        wqkv,
                        wz,
                        w_alpha,
                        w_beta,
                        conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        w_out,
                    },
                    LayerState::DeltaNet {
                        state: ssm_state,
                        conv_hist,
                    },
                ) => {
                    let tp = prof.as_ref().map(|_| Instant::now());
                    matmul_f32(
                        pool,
                        kt,
                        wqkv.dtype,
                        wqkv.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.qkv_buf[..b * conv_dim],
                        wqkv.rows,
                        n_embd,
                        b,
                    );
                    matmul_f32(
                        pool,
                        kt,
                        wz.dtype,
                        wz.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.gate_buf[..b * v_gdn_dim],
                        wz.rows,
                        n_embd,
                        b,
                    );
                    matmul_f32(
                        pool,
                        kt,
                        w_alpha.dtype,
                        w_alpha.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.alpha_buf[..b * hv],
                        w_alpha.rows,
                        n_embd,
                        b,
                    );
                    matmul_f32(
                        pool,
                        kt,
                        w_beta.dtype,
                        w_beta.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.beta_buf[..b * hv],
                        w_beta.rows,
                        n_embd,
                        b,
                    );
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), tp) {
                        p.gdn_proj_ns += ts.elapsed().as_nanos() as u64;
                    }

                    let tc = prof.as_ref().map(|_| Instant::now());
                    for t in 0..b {
                        for vh in 0..hv {
                            let idx = t * hv + vh;
                            let sp = softplus_f32(state.alpha_buf[idx] + ssm_dt_bias[vh]);
                            state.g_exp_buf[idx] = (sp * ssm_a[vh]).exp();
                            state.beta_buf[idx] = sigmoid_f32(state.beta_buf[idx]);
                        }
                    }
                    gdn_conv1d_silu_split_parallel(
                        pool,
                        kt,
                        &state.qkv_buf[..b * conv_dim],
                        conv_hist,
                        conv1d,
                        &mut state.q_buf[..b * k_gdn_dim],
                        &mut state.k_buf[..b * k_gdn_dim],
                        &mut state.v_buf[..b * v_gdn_dim],
                        d_conv,
                        hk,
                        s_dim,
                        v_gdn_dim,
                        self.cfg.rms_eps,
                        b,
                    );
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), tc) {
                        p.gdn_conv_norm_ns += ts.elapsed().as_nanos() as u64;
                    }

                    // Run multi-token NEON GDN recurrence + L1-fused gated RMSNorm * SiLU(z)!
                    let tr = prof.as_ref().map(|_| Instant::now());
                    gdn_recurrence_gated_parallel(
                        pool,
                        kt,
                        &state.q_buf[..b * k_gdn_dim],
                        &state.k_buf[..b * k_gdn_dim],
                        &state.v_buf[..b * v_gdn_dim],
                        &state.g_exp_buf[..b * hv],
                        &state.beta_buf[..b * hv],
                        ssm_state,
                        &mut state.attn_out[..b * v_gdn_dim],
                        &state.gate_buf[..b * v_gdn_dim],
                        ssm_norm,
                        self.cfg.rms_eps,
                        hk,
                        hv,
                        s_dim,
                        b,
                    );
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), tr) {
                        p.gdn_recur_ns += ts.elapsed().as_nanos() as u64;
                    }

                    let to = prof.as_ref().map(|_| Instant::now());
                    matmul_f32(
                        pool,
                        kt,
                        w_out.dtype,
                        w_out.bytes,
                        &state.attn_out[..b * v_gdn_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..b * n_embd],
                        w_out.rows,
                        w_out.cols,
                        b,
                    );
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), to) {
                        p.gdn_out_ns += ts.elapsed().as_nanos() as u64;
                    }
                }
                (
                    AttnWeights::FullAttn {
                        q_norm,
                        k_norm,
                        wqg,
                        wk,
                        wv,
                        wo,
                    },
                    LayerState::Kv { k_cache, v_cache },
                ) => {
                    let tq = prof.as_ref().map(|_| Instant::now());
                    matmul_f32(
                        pool,
                        kt,
                        wqg.dtype,
                        wqg.bytes,
                        &state.xb[..b * n_embd],
                        &mut state.mv_scratch,
                        &mut state.qkv_buf[..b * qg_dim],
                        wqg.rows,
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
                        wk.rows,
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
                        wv.rows,
                        n_embd,
                        b,
                    );
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), tq) {
                        p.full_qkv_ns += ts.elapsed().as_nanos() as u64;
                    }

                    let ta = prof.as_ref().map(|_| Instant::now());
                    for t in 0..b {
                        let pos = start_pos + t;
                        let qg_t = &state.qkv_buf[t * qg_dim..(t + 1) * qg_dim];
                        let q_t = &mut state.q_buf[t * q_dim..(t + 1) * q_dim];
                        let g_t = &mut state.gate_buf[t * q_dim..(t + 1) * q_dim];
                        let k_t = &mut state.k_buf[t * kv_dim..(t + 1) * kv_dim];
                        let v_t = &state.v_buf[t * kv_dim..(t + 1) * kv_dim];

                        for h in 0..self.cfg.n_head {
                            rms_norm(
                                &mut q_t[h * hd..(h + 1) * hd],
                                &qg_t[h * 2 * hd..h * 2 * hd + hd],
                                q_norm,
                                self.cfg.rms_eps,
                            );
                            g_t[h * hd..(h + 1) * hd]
                                .copy_from_slice(&qg_t[h * 2 * hd + hd..(h + 1) * 2 * hd]);
                        }
                        for h in 0..self.cfg.n_head_kv {
                            let s = h * hd;
                            let e = s + hd;
                            rms_norm(&mut state.xb2[..hd], &k_t[s..e], k_norm, self.cfg.rms_eps);
                            k_t[s..e].copy_from_slice(&state.xb2[..hd]);
                        }

                        rope_compute_cos_sin(
                            &self.rope_inv_freqs,
                            pos,
                            &mut state.rope_cos,
                            &mut state.rope_sin,
                        );
                        rope_inplace_tables(
                            q_t,
                            self.cfg.n_head,
                            hd,
                            self.cfg.rope_dim,
                            &state.rope_cos,
                            &state.rope_sin,
                            RopeType::Neox,
                        );
                        rope_inplace_tables(
                            k_t,
                            self.cfg.n_head_kv,
                            hd,
                            self.cfg.rope_dim,
                            &state.rope_cos,
                            &state.rope_sin,
                            RopeType::Neox,
                        );

                        let cache_off = pos * kv_dim;
                        (kt.f32_to_f16_slice)(k_t, &mut k_cache[cache_off..cache_off + kv_dim]);
                        (kt.f32_to_f16_slice)(v_t, &mut v_cache[cache_off..cache_off + kv_dim]);
                    }

                    gqa_attention_f16_kv_batch_parallel(
                        pool,
                        kt,
                        &mut state.attn_out[..b * q_dim],
                        &state.q_buf[..b * q_dim],
                        k_cache,
                        v_cache,
                        self.cfg.n_head,
                        self.cfg.n_head_kv,
                        hd,
                        start_pos,
                        b,
                        state.max_ctx,
                        &mut state.scores,
                    );
                    for (o, &g) in state.attn_out[..b * q_dim]
                        .iter_mut()
                        .zip(state.gate_buf[..b * q_dim].iter())
                    {
                        *o *= sigmoid_f32(g);
                    }
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), ta) {
                        p.full_attn_ns += ts.elapsed().as_nanos() as u64;
                    }

                    let two = prof.as_ref().map(|_| Instant::now());
                    matmul_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.attn_out[..b * q_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..b * n_embd],
                        wo.rows,
                        wo.cols,
                        b,
                    );
                    if let (Some(p), Some(ts)) = (prof.as_deref_mut(), two) {
                        p.full_wo_ns += ts.elapsed().as_nanos() as u64;
                    }
                }
                _ => unreachable!("LayerState variant always matches Qwen35LayerWeights::attn"),
            }

            for i in 0..b * n_embd {
                state.x[i] += state.xb2[i];
            }

            let t_pn = prof.as_ref().map(|_| Instant::now());
            rms_norm_batch_parallel(
                pool,
                kt,
                &mut state.xb[..b * n_embd],
                &state.x[..b * n_embd],
                &layer.post_attn_norm,
                self.cfg.rms_eps,
                n_embd,
                b,
            );
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_pn) {
                p.rms_norm_ns += ts.elapsed().as_nanos() as u64;
            }
            let t_gu = prof.as_ref().map(|_| Instant::now());
            matmul_f32(
                pool,
                kt,
                layer.w_gate.dtype,
                layer.w_gate.bytes,
                &state.xb[..b * n_embd],
                &mut state.mv_scratch,
                &mut state.hb[..b * n_ff],
                layer.w_gate.rows,
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
                layer.w_up.rows,
                n_embd,
                b,
            );
            swiglu_inplace_parallel(pool, &mut state.hb[..b * n_ff], &state.hb2[..b * n_ff]);
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_gu) {
                p.ffn_gate_up_ns += ts.elapsed().as_nanos() as u64;
            }
            let t_fd = prof.as_ref().map(|_| Instant::now());
            matmul_f32(
                pool,
                kt,
                layer.w_down.dtype,
                layer.w_down.bytes,
                &state.hb[..b * n_ff],
                &mut state.mv_scratch,
                &mut state.xb2[..b * n_embd],
                layer.w_down.rows,
                layer.w_down.cols,
                b,
            );
            for i in 0..b * n_embd {
                state.x[i] += state.xb2[i];
            }
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_fd) {
                p.ffn_down_ns += ts.elapsed().as_nanos() as u64;
            }
        }

        if compute_last_logits {
            let t_rn = prof.as_ref().map(|_| Instant::now());
            let last = (b - 1) * n_embd;
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[last..last + n_embd],
                &self.output_norm,
                self.cfg.rms_eps,
            );
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_rn) {
                p.rms_norm_ns += ts.elapsed().as_nanos() as u64;
            }
            let t_lm = prof.as_ref().map(|_| Instant::now());
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
            if let (Some(p), Some(ts)) = (prof, t_lm) {
                p.lm_head_ns += ts.elapsed().as_nanos() as u64;
            }
        }
    }

    /// Profiled single-token decode pass recording per-operator nanoseconds.
    pub fn forward_token_profiled(
        &self,
        state: &mut Qwen35State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        prof: &mut Qwen35DecodeProfile,
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
        let qg_dim = q_dim * 2;
        let kv_dim = self.cfg.n_head_kv * hd;
        let conv_dim = self.cfg.conv_dim();
        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = self.cfg.ssm_d_inner;
        let d_conv = self.cfg.ssm_conv_kernel;

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

        for (l, layer) in self.layers.iter().enumerate() {
            let tn = Instant::now();
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.attn_norm,
                self.cfg.rms_eps,
            );
            prof.rms_norm_ns += tn.elapsed().as_nanos() as u64;

            match (&layer.attn, &mut state.layers[l]) {
                (
                    AttnWeights::Gdn {
                        wqkv,
                        wz,
                        w_alpha,
                        w_beta,
                        conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        w_out,
                    },
                    LayerState::DeltaNet {
                        state: ssm_state,
                        conv_hist,
                    },
                ) => {
                    let tp = Instant::now();
                    matvec_pair_f32(
                        pool,
                        kt,
                        wqkv.dtype,
                        wqkv.bytes,
                        wqkv.rows,
                        &mut state.qkv_buf[..conv_dim],
                        wz.dtype,
                        wz.bytes,
                        wz.rows,
                        &mut state.gate_buf[..v_gdn_dim],
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                    );
                    matvec_pair_f32(
                        pool,
                        kt,
                        w_alpha.dtype,
                        w_alpha.bytes,
                        w_alpha.rows,
                        &mut state.alpha_buf[..hv],
                        w_beta.dtype,
                        w_beta.bytes,
                        w_beta.rows,
                        &mut state.beta_buf[..hv],
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                    );
                    prof.gdn_proj_ns += tp.elapsed().as_nanos() as u64;

                    let tc = Instant::now();
                    for vh in 0..hv {
                        let sp = softplus_f32(state.alpha_buf[vh] + ssm_dt_bias[vh]);
                        state.g_exp_buf[vh] = (sp * ssm_a[vh]).exp();
                        state.beta_buf[vh] = sigmoid_f32(state.beta_buf[vh]);
                    }
                    gdn_conv1d_silu_split_parallel(
                        pool,
                        kt,
                        &state.qkv_buf[..conv_dim],
                        conv_hist,
                        conv1d,
                        &mut state.q_buf[..k_gdn_dim],
                        &mut state.k_buf[..k_gdn_dim],
                        &mut state.v_buf[..v_gdn_dim],
                        d_conv,
                        hk,
                        s_dim,
                        v_gdn_dim,
                        self.cfg.rms_eps,
                        1,
                    );
                    prof.gdn_conv_norm_ns += tc.elapsed().as_nanos() as u64;

                    let tr = Instant::now();
                    gdn_recurrence_gated_parallel(
                        pool,
                        kt,
                        &state.q_buf[..k_gdn_dim],
                        &state.k_buf[..k_gdn_dim],
                        &state.v_buf[..v_gdn_dim],
                        &state.g_exp_buf[..hv],
                        &state.beta_buf[..hv],
                        ssm_state,
                        &mut state.attn_out[..v_gdn_dim],
                        &state.gate_buf[..v_gdn_dim],
                        ssm_norm,
                        self.cfg.rms_eps,
                        hk,
                        hv,
                        s_dim,
                        1,
                    );
                    prof.gdn_recur_ns += tr.elapsed().as_nanos() as u64;

                    let to = Instant::now();
                    matvec_f32(
                        pool,
                        kt,
                        w_out.dtype,
                        w_out.bytes,
                        &state.attn_out[..v_gdn_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..n_embd],
                        w_out.rows,
                        w_out.cols,
                    );
                    prof.gdn_out_ns += to.elapsed().as_nanos() as u64;
                }
                (
                    AttnWeights::FullAttn {
                        q_norm,
                        k_norm,
                        wqg,
                        wk,
                        wv,
                        wo,
                    },
                    LayerState::Kv { k_cache, v_cache },
                ) => {
                    let tq = Instant::now();
                    matvec_triple_f32(
                        pool,
                        kt,
                        wqg.dtype,
                        wqg.bytes,
                        wqg.rows,
                        &mut state.qkv_buf[..qg_dim],
                        wk.dtype,
                        wk.bytes,
                        wk.rows,
                        &mut state.k_buf[..kv_dim],
                        wv.dtype,
                        wv.bytes,
                        wv.rows,
                        &mut state.v_buf[..kv_dim],
                        &state.xb[..n_embd],
                        &mut state.mv_scratch,
                        n_embd,
                    );
                    prof.full_qkv_ns += tq.elapsed().as_nanos() as u64;

                    let ta = Instant::now();
                    for h in 0..self.cfg.n_head {
                        rms_norm(
                            &mut state.q_buf[h * hd..(h + 1) * hd],
                            &state.qkv_buf[h * 2 * hd..h * 2 * hd + hd],
                            q_norm,
                            self.cfg.rms_eps,
                        );
                        state.gate_buf[h * hd..(h + 1) * hd]
                            .copy_from_slice(&state.qkv_buf[h * 2 * hd + hd..(h + 1) * 2 * hd]);
                    }
                    for h in 0..self.cfg.n_head_kv {
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
                        self.cfg.n_head_kv,
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
                    gqa_attention_f16_kv_parallel(
                        pool,
                        kt,
                        &mut state.attn_out[..q_dim],
                        &state.q_buf[..q_dim],
                        k_cache,
                        v_cache,
                        self.cfg.n_head,
                        self.cfg.n_head_kv,
                        hd,
                        pos + 1,
                        state.max_ctx,
                        &mut state.scores,
                    );
                    for (o, &g) in state.attn_out[..q_dim]
                        .iter_mut()
                        .zip(state.gate_buf[..q_dim].iter())
                    {
                        *o *= sigmoid_f32(g);
                    }
                    prof.full_attn_ns += ta.elapsed().as_nanos() as u64;

                    let tw = Instant::now();
                    matvec_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.attn_out[..q_dim],
                        &mut state.mv_scratch,
                        &mut state.xb2[..n_embd],
                        wo.rows,
                        wo.cols,
                    );
                    prof.full_wo_ns += tw.elapsed().as_nanos() as u64;
                }
                _ => unreachable!("LayerState variant always matches Qwen35LayerWeights::attn"),
            }

            for i in 0..n_embd {
                state.x[i] += state.xb2[i];
            }

            let tn2 = Instant::now();
            rms_norm(
                &mut state.xb[..n_embd],
                &state.x[..n_embd],
                &layer.post_attn_norm,
                self.cfg.rms_eps,
            );
            prof.rms_norm_ns += tn2.elapsed().as_nanos() as u64;

            let tf1 = Instant::now();
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
            prof.ffn_gate_up_ns += tf1.elapsed().as_nanos() as u64;

            let tf2 = Instant::now();
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
            prof.ffn_down_ns += tf2.elapsed().as_nanos() as u64;
        }

        let tlm = Instant::now();
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
        prof.lm_head_ns += tlm.elapsed().as_nanos() as u64;
        prof.n_steps += 1;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tm_cpu::{active_kernels, gdn_recurrence_parallel};

    #[test]
    fn gdn_recurrence_matches_fp64_reference_and_batch_bit_exact() {
        // Choose HV = 6 > HK = 2 so tiled `kh = vh % HK` vs contiguous `vh / 3` would diverge!
        let hk = 2usize;
        let hv = 6usize;
        let hd = 16usize;
        let n_tok = 7usize;

        let q: Vec<f32> = (0..n_tok * hk * hd)
            .map(|i| (((i * 17 + 3) % 97) as f32 - 48.0) * 0.02)
            .collect();
        let k: Vec<f32> = (0..n_tok * hk * hd)
            .map(|i| (((i * 31 + 7) % 89) as f32 - 44.0) * 0.02)
            .collect();
        let v: Vec<f32> = (0..n_tok * hv * hd)
            .map(|i| (((i * 13 + 11) % 83) as f32 - 41.0) * 0.05)
            .collect();
        let g_exp: Vec<f32> = (0..n_tok * hv)
            .map(|i| 0.85 + ((i % 7) as f32) * 0.02)
            .collect();
        let beta: Vec<f32> = (0..n_tok * hv)
            .map(|i| 0.25 + ((i % 5) as f32) * 0.1)
            .collect();

        // 1. Textbook FP64 reference (non-transposed S[vh][i][j])
        let mut s_ref = vec![vec![vec![0.0f64; hd]; hd]; hv];
        let mut out_ref = vec![0.0f64; n_tok * hv * hd];
        let scale64 = 1.0 / (hd as f64).sqrt();

        for t in 0..n_tok {
            for vh in 0..hv {
                let kh = vh % hk;
                let g = g_exp[t * hv + vh] as f64;
                let b = beta[t * hv + vh] as f64;
                let q_h = &q[t * hk * hd + kh * hd..t * hk * hd + (kh + 1) * hd];
                let k_h = &k[t * hk * hd + kh * hd..t * hk * hd + (kh + 1) * hd];
                let v_h = &v[t * hv * hd + vh * hd..t * hv * hd + (vh + 1) * hd];

                // kv[j] = sum_i (g * S[i][j]) * k[i]
                let mut delta = vec![0.0f64; hd];
                for j in 0..hd {
                    let mut kv_j = 0.0f64;
                    for i in 0..hd {
                        s_ref[vh][i][j] *= g;
                        kv_j += s_ref[vh][i][j] * (k_h[i] as f64);
                    }
                    delta[j] = ((v_h[j] as f64) - kv_j) * b;
                }
                // S[i][j] += k[i] * delta[j]
                for i in 0..hd {
                    for j in 0..hd {
                        s_ref[vh][i][j] += (k_h[i] as f64) * delta[j];
                    }
                }
                // o[j] = (sum_i S[i][j] * q[i]) / sqrt(hd)
                for j in 0..hd {
                    let mut o_j = 0.0f64;
                    for i in 0..hd {
                        o_j += s_ref[vh][i][j] * (q_h[i] as f64);
                    }
                    out_ref[(t * hv + vh) * hd + j] = o_j * scale64;
                }
            }
        }

        // 2. Parallel batched execution (`n_tok = 7`)
        let pool = ThreadPool::new(4);
        let mut s_batch = vec![0.0f32; hv * hd * hd];
        let mut out_batch = vec![0.0f32; n_tok * hv * hd];
        gdn_recurrence_parallel(
            &pool,
            &q,
            &k,
            &v,
            &g_exp,
            &beta,
            &mut s_batch,
            &mut out_batch,
            hk,
            hv,
            hd,
            n_tok,
        );

        // 3. Step-by-step single-token execution (`n_tok = 1` x 7)
        let mut s_step = vec![0.0f32; hv * hd * hd];
        let mut out_step = vec![0.0f32; n_tok * hv * hd];
        for t in 0..n_tok {
            gdn_recurrence_parallel(
                &pool,
                &q[t * hk * hd..(t + 1) * hk * hd],
                &k[t * hk * hd..(t + 1) * hk * hd],
                &v[t * hv * hd..(t + 1) * hv * hd],
                &g_exp[t * hv..(t + 1) * hv],
                &beta[t * hv..(t + 1) * hv],
                &mut s_step,
                &mut out_step[t * hv * hd..(t + 1) * hv * hd],
                hk,
                hv,
                hd,
                1,
            );
        }

        // Verify bit-exact determinism between batch and single-token (`AGENTS.md` §9)
        for i in 0..out_batch.len() {
            assert_eq!(
                out_batch[i].to_bits(),
                out_step[i].to_bits(),
                "GDN output batch vs step bit mismatch at {i}"
            );
        }
        for i in 0..s_batch.len() {
            assert_eq!(
                s_batch[i].to_bits(),
                s_step[i].to_bits(),
                "GDN state batch vs step bit mismatch at {i}"
            );
        }

        // Verify tight agreement against FP64 reference (`docs/TESTING.md` L3)
        for i in 0..out_batch.len() {
            let err = (out_batch[i] as f64 - out_ref[i]).abs();
            assert!(
                err < 1e-6,
                "GDN output vs FP64 mismatch at {i}: got={} ref={} err={err}",
                out_batch[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn qwen35_state_atomicity_on_error() {
        let model_path = std::path::Path::new("../../models/Qwen3.5-0.8B-Q4_0.gguf");
        if !model_path.exists() {
            return;
        }
        let gguf = GgufFile::open(model_path).expect("open Qwen3.5-0.8B");
        let model = Qwen35Model::from_gguf(&gguf).expect("load Qwen3.5-0.8B");
        let mut state = Qwen35State::new(&model, 8);
        let pool = ThreadPool::new(4);
        let kt = active_kernels();

        model
            .forward_token(&mut state, &pool, kt, 760, 0, false)
            .expect("step 0");
        model
            .forward_token(&mut state, &pool, kt, 469, 1, true)
            .expect("step 1");
        let snap = state.layers.clone();

        // Out-of-bounds pos and out-of-range token_id MUST return Err BEFORE mutating LayerState!
        assert!(
            model
                .forward_token(&mut state, &pool, kt, 760, 8, false)
                .is_err()
        );
        assert!(
            model
                .forward_token(&mut state, &pool, kt, 999_999, 2, false)
                .is_err()
        );
        assert!(
            model
                .forward_batch(&mut state, &pool, kt, &[760, 469, 999_999], 2, false)
                .is_err()
        );

        for (i, (a, b)) in state.layers.iter().zip(snap.iter()).enumerate() {
            match (a, b) {
                (
                    LayerState::Kv {
                        k_cache: ka,
                        v_cache: va,
                    },
                    LayerState::Kv {
                        k_cache: kb,
                        v_cache: vb,
                    },
                ) => {
                    assert_eq!(ka, kb, "KV k_cache mutated on error at layer {i}");
                    assert_eq!(va, vb, "KV v_cache mutated on error at layer {i}");
                }
                (
                    LayerState::DeltaNet {
                        state: sa,
                        conv_hist: ca,
                    },
                    LayerState::DeltaNet {
                        state: sb,
                        conv_hist: cb,
                    },
                ) => {
                    assert_eq!(sa, sb, "GDN state mutated on error at layer {i}");
                    assert_eq!(ca, cb, "GDN conv_hist mutated on error at layer {i}");
                }
                _ => panic!("LayerState variant mismatch"),
            }
        }
    }
}
