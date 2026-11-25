//! Unified Mamba-2 SSM & Hybrid forward pass (`ADR-017`).
//!
//! Covers all 4 upstream `llama.cpp` Mamba-2 GGUF architectures:
//! - `"mamba2"` (Pure Mamba-2 SSM blocks)
//! - `"nemotron_h"` (Heterogeneous `Mamba-2` | `NoRoPE GQA` | `Squared-ReLU FFN` blocks)
//! - `"falcon-h1"` (Parallel `NeoX-RoPE GQA + Mamba-2` mixer + `SwiGLU FFN` blocks)
//! - `"granitehybrid"` (Interleaved `Mamba-2 | Norm-RoPE GQA` mixer + `SwiGLU FFN` + muP scaling)

use tm_core::{DType, KvType, QK8_0, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool,
    gqa_attention_f16_kv_batch_parallel, gqa_attention_f16_kv_ext_batch_parallel,
    gqa_attention_f16_kv_ext_parallel, gqa_attention_f16_kv_parallel,
    gqa_attention_q8_0_kv_ext_batch_parallel, gqa_attention_q8_0_kv_ext_parallel,
    mamba2_scan_parallel, matmul_f32, matmul_pair_f32, matmul_triple_f32, matvec_f32,
    matvec_pair_f32, matvec_triple_f32, relu2_inplace_parallel, rms_norm, rms_norm_batch_parallel,
    rope_compute_cos_sin, rope_inplace_tables, swiglu_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};
use tm_lora::{BoundLora, LoraScratch, LoraSet};

use crate::llama::{ModelError, RawMatView};
use crate::moe::{
    MoeScratch, SparseMoeFfn, eval_sparse_moe_ffn, load_layer_moe_ffn, parse_moe_arch_params,
};
use crate::state::{LayerState, MAX_SPEC_WINDOW, SpecVerifyBuf, decode_embedding_row};

/// Model hyperparameters for `mamba2`, `nemotron_h`, `falcon-h1`, and `granitehybrid`.
#[derive(Debug, Clone)]
pub struct Mamba2Config {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub max_ctx_train: usize,
    pub rms_eps: f32,
    // Mamba-2 SSM parameters
    pub ssm_conv_kernel: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_n_head: usize,
    pub ssm_n_group: usize,
    pub ssm_norm_eps: f32,
    // Optional GQA attention parameters (hybrid architectures)
    pub n_head_per_layer: Vec<usize>,
    pub n_head_kv_per_layer: Vec<usize>,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rope_type: RopeType,
    pub use_rope: bool,
    pub rope_orig_ctx: usize,
    // Optional FFN parameters
    pub n_ff_per_layer: Vec<usize>,
    // Optional muP scaling (`granitehybrid`)
    pub embedding_scale: f32,
    pub attn_scale: f32,
    pub custom_attn_scale: bool,
    pub residual_scale: f32,
    pub inv_logit_scale: f32,
    // Optional Sparse MoE parameters (`granitehybrid` / `granitemoehybrid` with `expert_count > 0`)
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub max_n_ff_exp: usize,
    pub max_n_ff_shexp: usize,
}

impl Mamba2Config {
    #[inline]
    pub fn ssm_head_dim(&self) -> usize {
        self.ssm_inner_size / self.ssm_n_head
    }

    #[inline]
    pub fn bc_dim(&self) -> usize {
        self.ssm_n_group * self.ssm_state_size
    }

    #[inline]
    pub fn conv_dim(&self) -> usize {
        self.ssm_inner_size + 2 * self.bc_dim()
    }

    #[inline]
    pub fn d_in_proj(&self) -> usize {
        2 * self.ssm_inner_size + 2 * self.bc_dim() + self.ssm_n_head
    }

    #[inline]
    pub fn ssm_group_dim(&self) -> usize {
        self.ssm_inner_size / self.ssm_n_group
    }

    #[inline]
    pub fn max_q_dim(&self) -> usize {
        self.n_head_per_layer.iter().copied().max().unwrap_or(0) * self.head_dim
    }

    #[inline]
    pub fn max_kv_dim(&self) -> usize {
        self.n_head_kv_per_layer.iter().copied().max().unwrap_or(0) * self.head_dim
    }

    #[inline]
    pub fn max_n_ff(&self) -> usize {
        self.n_ff_per_layer.iter().copied().max().unwrap_or(0)
    }
}

/// Weights for a Mamba-2 selective state space mixer (`blk.{l}.ssm_*`).
pub struct Mamba2SsmWeights<'a> {
    pub ssm_spec_idx: usize,
    pub ssm_in: RawMatView<'a>,
    /// Row-major `[conv_dim, d_conv]` (`d_conv` contiguous taps per channel).
    pub conv_w: Vec<f32>,
    pub conv_b: Option<Vec<f32>>,
    pub dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_d: Vec<f32>,
    /// Optional grouped RMSNorm weight `[d_inner]` (`n_group` groups of `d_inner / n_group`).
    pub ssm_norm: Option<Vec<f32>>,
    pub ssm_out: RawMatView<'a>,
}

/// Weights for a GQA attention mixer in hybrid Mamba-2 architectures.
pub struct Mamba2AttnWeights<'a> {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub wq: RawMatView<'a>,
    pub wk: RawMatView<'a>,
    pub wv: RawMatView<'a>,
    pub wo: RawMatView<'a>,
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub bo: Option<Vec<f32>>,
}

/// FFN sub-block weights (`SwiGLU`, `Relu2`, or fine-grained `SparseMoe`).
#[allow(clippy::large_enum_variant)]
pub enum Mamba2FfnWeights<'a> {
    SwiGlu {
        n_ff: usize,
        w_gate: RawMatView<'a>,
        w_up: RawMatView<'a>,
        w_down: RawMatView<'a>,
        b_gate: Option<Vec<f32>>,
        b_up: Option<Vec<f32>>,
        b_down: Option<Vec<f32>>,
    },
    Relu2 {
        n_ff: usize,
        w_up: RawMatView<'a>,
        w_down: RawMatView<'a>,
        b_up: Option<Vec<f32>>,
        b_down: Option<Vec<f32>>,
    },
    SparseMoe(SparseMoeFfn<'a>),
}

/// Heterogeneous block topology across `mamba2`, `nemotron_h`, `falcon-h1`, and `granitehybrid`.
#[allow(clippy::large_enum_variant)]
pub enum Mamba2Block<'a> {
    /// `mamba2` or `nemotron_h` SSM-only layer: `x += res_scale * Ssm(rms_norm(x, attn_norm))`.
    SsmOnly { ssm: Mamba2SsmWeights<'a> },
    /// `nemotron_h` Attention-only layer: `x += res_scale * Attn(rms_norm(x, attn_norm))`.
    AttnOnly { attn: Mamba2AttnWeights<'a> },
    /// `nemotron_h` FFN-only layer: `x += res_scale * Relu2Ffn(rms_norm(x, attn_norm))`.
    FfnOnly { ffn: Mamba2FfnWeights<'a> },
    /// `granitehybrid` sequential SSM + SwiGLU FFN layer.
    SsmAndFfn {
        ssm: Mamba2SsmWeights<'a>,
        ffn_norm: Vec<f32>,
        ffn: Mamba2FfnWeights<'a>,
    },
    /// `granitehybrid` sequential Attention + SwiGLU FFN layer.
    AttnAndFfn {
        attn: Mamba2AttnWeights<'a>,
        ffn_norm: Vec<f32>,
        ffn: Mamba2FfnWeights<'a>,
    },
    /// `falcon-h1` parallel `(Attn(xn) + Ssm(xn))` mixer + sequential `SwiGLU` FFN layer.
    ParallelAttnSsmAndFfn {
        attn: Mamba2AttnWeights<'a>,
        ssm: Mamba2SsmWeights<'a>,
        ffn_norm: Vec<f32>,
        ffn: Mamba2FfnWeights<'a>,
    },
}

/// Per-layer weights and `LayerState` slot indices for `Mamba2Model`.
pub struct Mamba2LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub kv_slot: Option<usize>,
    pub ssm_slot: Option<usize>,
    pub block: Mamba2Block<'a>,
}

/// Loaded unified Mamba-2 / Hybrid model (`#![forbid(unsafe_code)]`).
pub struct Mamba2Model<'a> {
    pub cfg: Mamba2Config,
    pub token_embd: RawMatView<'a>,
    pub output: RawMatView<'a>,
    pub output_bias: Option<Vec<f32>>,
    pub output_norm: Vec<f32>,
    pub rope_inv_freqs: Vec<f64>,
    pub rope_inv_freqs_long: Option<Vec<f64>>,
    pub layers: Vec<Mamba2LayerWeights<'a>>,
    pub n_state_slots: usize,
    pub n_ssm_layers: usize,
    pub lora: BoundLora,
}

/// Pre-allocated per-session inference state for `Mamba2Model` (zero heap allocations in `forward_*`).
pub struct Mamba2State {
    pub max_ctx: usize,
    pub kv_type: KvType,
    pub layers: Vec<LayerState>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    xb_par: Vec<f32>,
    in_proj: Vec<f32>,
    ssm_z: Vec<f32>,
    ssm_xbc: Vec<f32>,
    ssm_dt: Vec<f32>,
    ssm_x: Vec<f32>,
    ssm_b: Vec<f32>,
    ssm_c: Vec<f32>,
    ssm_y: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    pub last_hidden: Vec<f32>,
    mv_scratch: MatvecScratch,
    pub lora_scratch: LoraScratch,
    pub moe_scratch: MoeScratch,
}

impl Mamba2State {
    pub fn new(model: &Mamba2Model<'_>, max_ctx: usize) -> Self {
        Self::new_with_kv_type(model, max_ctx, KvType::F16)
    }

    pub fn new_with_kv_type(model: &Mamba2Model<'_>, max_ctx: usize, kv_type: KvType) -> Self {
        let c = &model.cfg;
        let d_inner = c.ssm_inner_size;
        let bc_dim = c.bc_dim();
        let conv_dim = c.conv_dim();
        let d_in_proj = c.d_in_proj();
        let hist_len = c.ssm_conv_kernel.saturating_sub(1);
        let state_len = d_inner * c.ssm_state_size;

        let mut layers = Vec::with_capacity(model.n_state_slots);
        for _ in 0..model.n_state_slots {
            layers.push(LayerState::SharedKv {
                source_layer: 0,
                window: None,
            });
        }
        for lw in &model.layers {
            if let Some(ks) = lw.kv_slot {
                let n_kv = match &lw.block {
                    Mamba2Block::AttnOnly { attn }
                    | Mamba2Block::AttnAndFfn { attn, .. }
                    | Mamba2Block::ParallelAttnSsmAndFfn { attn, .. } => attn.n_head_kv,
                    _ => 0,
                };
                layers[ks] = LayerState::new_kv(max_ctx, n_kv * c.head_dim, None, kv_type);
            }
            if let Some(ss) = lw.ssm_slot {
                layers[ss] = LayerState::Ssm {
                    state: vec![0.0; state_len],
                    conv_hist: vec![0.0; conv_dim * hist_len],
                };
            }
        }

        let max_q = c.max_q_dim();
        let max_kv = c.max_kv_dim();
        let max_ff = c.max_n_ff();
        let max_n_head = c.n_head_per_layer.iter().copied().max().unwrap_or(0);
        let max_cols = c
            .n_embd
            .max(d_inner)
            .max(d_in_proj)
            .max(max_q)
            .max(max_ff)
            .max(c.max_n_ff_exp)
            .max(c.max_n_ff_shexp)
            .max(c.vocab_size);

        Self {
            max_ctx,
            kv_type,
            layers,
            x: vec![0.0; MAX_CPU_BATCH * c.n_embd],
            xb: vec![0.0; MAX_CPU_BATCH * c.n_embd.max(max_q).max(d_inner)],
            xb2: vec![0.0; MAX_CPU_BATCH * c.n_embd],
            xb_par: if c.arch == "falcon-h1" {
                vec![0.0; MAX_CPU_BATCH * c.n_embd]
            } else {
                Vec::new()
            },
            in_proj: vec![0.0; MAX_CPU_BATCH * d_in_proj],
            ssm_z: vec![0.0; MAX_CPU_BATCH * d_inner],
            ssm_xbc: vec![0.0; MAX_CPU_BATCH * conv_dim],
            ssm_dt: vec![0.0; MAX_CPU_BATCH * c.ssm_n_head],
            ssm_x: vec![0.0; MAX_CPU_BATCH * d_inner],
            ssm_b: vec![0.0; MAX_CPU_BATCH * bc_dim],
            ssm_c: vec![0.0; MAX_CPU_BATCH * bc_dim],
            ssm_y: vec![0.0; MAX_CPU_BATCH * d_inner],
            q: vec![0.0; MAX_CPU_BATCH * max_q],
            k: vec![0.0; MAX_CPU_BATCH * max_kv],
            v: vec![0.0; MAX_CPU_BATCH * max_kv],
            hb: vec![0.0; MAX_CPU_BATCH * max_ff],
            hb2: vec![0.0; MAX_CPU_BATCH * max_ff],
            rope_cos: vec![0.0; c.rope_dim / 2],
            rope_sin: vec![0.0; c.rope_dim / 2],
            scores: vec![0.0; max_n_head * max_ctx],
            logits: vec![0.0; c.vocab_size],
            last_hidden: vec![0.0; c.n_embd],
            mv_scratch: MatvecScratch::new(max_cols),
            lora_scratch: LoraScratch::new(64),
            moe_scratch: MoeScratch::new(
                c.n_embd,
                c.n_expert,
                c.n_expert_used,
                c.max_n_ff_exp,
                c.max_n_ff_shexp,
            ),
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

fn load_f32_vec_1d_or_2d(
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    d0: usize,
    d1: usize,
) -> Result<Vec<f32>, ModelError> {
    let total = d0 * d1;
    let Some(info) = gguf.tensor_info(name) else {
        return Err(GgufError::MissingTensor {
            path: gguf.path().display().to_string(),
            tensor: name.to_string(),
            layer: layer.to_string(),
        }
        .into());
    };
    let (_, bytes) = if info.dims == [total] {
        gguf.req_tensor(name, layer, &[total])?
    } else {
        gguf.req_tensor(name, layer, &[d0, d1])?
    };
    let mut out = vec![0.0f32; total];
    dequantize_row_bytes(info.dtype, bytes, &mut out);
    Ok(out)
}

fn load_opt_f32_vec(
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<Option<Vec<f32>>, ModelError> {
    if gguf.tensor_info(name).is_some() {
        Ok(Some(load_f32_vec(gguf, name, layer, len)?))
    } else {
        Ok(None)
    }
}

fn load_f32_vec_any(
    gguf: &GgufFile,
    names: &[&str],
    layer: &str,
    len: usize,
) -> Result<Vec<f32>, ModelError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return load_f32_vec(gguf, name, layer, len);
        }
    }
    load_f32_vec(gguf, names[0], layer, len)
}

fn load_opt_f32_vec_any(
    gguf: &GgufFile,
    names: &[&str],
    layer: &str,
    len: usize,
) -> Result<Option<Vec<f32>>, ModelError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return Ok(Some(load_f32_vec(gguf, name, layer, len)?));
        }
    }
    Ok(None)
}

fn load_mat<'a>(
    gguf: &'a GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<RawMatView<'a>, ModelError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    Ok(RawMatView {
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

fn load_mat_slice<'a>(
    gguf: &'a GgufFile,
    fused_name: &str,
    layer: &str,
    cols: usize,
    total_rows: usize,
    row_start: usize,
    rows: usize,
) -> Result<RawMatView<'a>, ModelError> {
    let (info, bytes) = gguf.req_tensor(fused_name, layer, &[cols, total_rows])?;
    if info.dtype == DType::Q4_0x4 && (!row_start.is_multiple_of(4) || !rows.is_multiple_of(4)) {
        return Err(ModelError::InvalidConfig(format!(
            "cannot slice Q4_0x4 tensor '{fused_name}' at non-4-aligned row range {row_start}..{}",
            row_start + rows
        )));
    }
    let row_bytes = info.dtype.row_bytes(cols);
    let byte_start = row_start * row_bytes;
    let byte_end = (row_start + rows) * row_bytes;
    Ok(RawMatView {
        name: info.name.as_str(),
        bytes: &bytes[byte_start..byte_end],
        dtype: info.dtype,
        rows,
        cols,
        row_offset: row_start,
        is_sliced: true,
        lora_slot: None,
    })
}

fn meta_usize_per_layer(gguf: &GgufFile, key: &str, n_layers: usize, default: usize) -> Vec<usize> {
    if let Some(v) = gguf.get_meta(key) {
        if let Some(arr) = v.as_usize_array()
            && arr.len() >= n_layers
        {
            return arr[..n_layers].to_vec();
        }
        if let Some(sc) = v.as_usize() {
            return vec![sc; n_layers];
        }
    }
    vec![default; n_layers]
}

fn load_ssm_weights<'a>(
    gguf: &'a GgufFile,
    ltag: &str,
    n_embd: usize,
    cfg: &Mamba2Config,
    ssm_spec_idx: usize,
    norm_required: bool,
) -> Result<Mamba2SsmWeights<'a>, ModelError> {
    let d_inner = cfg.ssm_inner_size;
    let d_conv = cfg.ssm_conv_kernel;
    let conv_dim = cfg.conv_dim();
    let d_in_proj = cfg.d_in_proj();
    let n_heads = cfg.ssm_n_head;
    let n_group = cfg.ssm_n_group;
    let group_dim = cfg.ssm_group_dim();

    let ssm_in = load_mat(
        gguf,
        &format!("{ltag}.ssm_in.weight"),
        ltag,
        n_embd,
        d_in_proj,
    )?;

    let conv_name = format!("{ltag}.ssm_conv1d.weight");
    let (c_info, c_bytes) = gguf.req_tensor(&conv_name, ltag, &[d_conv, conv_dim])?;
    let mut conv_w = vec![0.0f32; conv_dim * d_conv];
    let rb = c_info.dtype.row_bytes(d_conv);
    for r in 0..conv_dim {
        dequantize_row_bytes(
            c_info.dtype,
            &c_bytes[r * rb..(r + 1) * rb],
            &mut conv_w[r * d_conv..(r + 1) * d_conv],
        );
    }
    let conv_b = load_opt_f32_vec(gguf, &format!("{ltag}.ssm_conv1d.bias"), ltag, conv_dim)?;
    let dt_bias = load_f32_vec(gguf, &format!("{ltag}.ssm_dt.bias"), ltag, n_heads)?;
    let ssm_a = load_f32_vec_1d_or_2d(gguf, &format!("{ltag}.ssm_a"), ltag, 1, n_heads)?;
    let ssm_d = load_f32_vec_1d_or_2d(gguf, &format!("{ltag}.ssm_d"), ltag, 1, n_heads)?;

    let norm_name = format!("{ltag}.ssm_norm.weight");
    let ssm_norm = if gguf.tensor_info(&norm_name).is_some() || norm_required {
        Some(load_f32_vec_1d_or_2d(
            gguf, &norm_name, ltag, group_dim, n_group,
        )?)
    } else {
        None
    };

    let ssm_out = load_mat(
        gguf,
        &format!("{ltag}.ssm_out.weight"),
        ltag,
        d_inner,
        n_embd,
    )?;

    Ok(Mamba2SsmWeights {
        ssm_spec_idx,
        ssm_in,
        conv_w,
        conv_b,
        dt_bias,
        ssm_a,
        ssm_d,
        ssm_norm,
        ssm_out,
    })
}

fn load_attn_weights<'a>(
    gguf: &'a GgufFile,
    ltag: &str,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Result<Mamba2AttnWeights<'a>, ModelError> {
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let q_name = format!("{ltag}.attn_q.weight");
    let qkv_name = format!("{ltag}.attn_qkv.weight");
    let (wq, wk, wv) = if gguf.tensor_info(&q_name).is_some() {
        (
            load_mat(gguf, &q_name, ltag, n_embd, q_dim)?,
            load_mat(gguf, &format!("{ltag}.attn_k.weight"), ltag, n_embd, kv_dim)?,
            load_mat(gguf, &format!("{ltag}.attn_v.weight"), ltag, n_embd, kv_dim)?,
        )
    } else {
        let total = q_dim + 2 * kv_dim;
        (
            load_mat_slice(gguf, &qkv_name, ltag, n_embd, total, 0, q_dim)?,
            load_mat_slice(gguf, &qkv_name, ltag, n_embd, total, q_dim, kv_dim)?,
            load_mat_slice(gguf, &qkv_name, ltag, n_embd, total, q_dim + kv_dim, kv_dim)?,
        )
    };
    let qkv_bias_name = format!("{ltag}.attn_qkv.bias");
    let (bq, bk, bv) = if gguf.tensor_info(&qkv_bias_name).is_some() {
        let full = load_f32_vec(gguf, &qkv_bias_name, ltag, q_dim + 2 * kv_dim)?;
        (
            Some(full[..q_dim].to_vec()),
            Some(full[q_dim..q_dim + kv_dim].to_vec()),
            Some(full[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec()),
        )
    } else {
        (
            load_opt_f32_vec(gguf, &format!("{ltag}.attn_q.bias"), ltag, q_dim)?,
            load_opt_f32_vec(gguf, &format!("{ltag}.attn_k.bias"), ltag, kv_dim)?,
            load_opt_f32_vec(gguf, &format!("{ltag}.attn_v.bias"), ltag, kv_dim)?,
        )
    };
    let wo = load_mat(
        gguf,
        &format!("{ltag}.attn_output.weight"),
        ltag,
        q_dim,
        n_embd,
    )?;
    let bo = load_opt_f32_vec(gguf, &format!("{ltag}.attn_output.bias"), ltag, n_embd)?;
    Ok(Mamba2AttnWeights {
        n_head,
        n_head_kv,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
    })
}

fn load_swiglu_ffn<'a>(
    gguf: &'a GgufFile,
    ltag: &str,
    n_embd: usize,
    n_ff: usize,
) -> Result<Mamba2FfnWeights<'a>, ModelError> {
    let w_gate = load_mat(gguf, &format!("{ltag}.ffn_gate.weight"), ltag, n_embd, n_ff)?;
    let w_up = load_mat(gguf, &format!("{ltag}.ffn_up.weight"), ltag, n_embd, n_ff)?;
    let w_down = load_mat(gguf, &format!("{ltag}.ffn_down.weight"), ltag, n_ff, n_embd)?;
    let b_gate = load_opt_f32_vec(gguf, &format!("{ltag}.ffn_gate.bias"), ltag, n_ff)?;
    let b_up = load_opt_f32_vec(gguf, &format!("{ltag}.ffn_up.bias"), ltag, n_ff)?;
    let b_down = load_opt_f32_vec(gguf, &format!("{ltag}.ffn_down.bias"), ltag, n_embd)?;
    Ok(Mamba2FfnWeights::SwiGlu {
        n_ff,
        w_gate,
        w_up,
        w_down,
        b_gate,
        b_up,
        b_down,
    })
}

#[inline]
fn add_bias_rows(dst: &mut [f32], bias: &[f32], rows: usize) {
    let cols = bias.len();
    for r in 0..rows {
        let row = &mut dst[r * cols..(r + 1) * cols];
        for (y, &b) in row.iter_mut().zip(bias.iter()) {
            *y += b;
        }
    }
}

#[inline]
fn scale_slice_inplace(dst: &mut [f32], scale: f32) {
    if (scale - 1.0).abs() > 1e-12 {
        for v in dst.iter_mut() {
            *v *= scale;
        }
    }
}

/// Causal depthwise Conv1d + optional bias + `SiLU` + split `xBC [n_tok, conv_dim]` into
/// `x_out [n_tok, d_inner]`, `b_out [n_tok, bc_dim]`, and `c_out [n_tok, bc_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_conv1d_silu_split(
    xbc_in: &[f32],
    conv_hist: &mut [f32],
    conv_w: &[f32],
    conv_b: Option<&[f32]>,
    x_out: &mut [f32],
    b_out: &mut [f32],
    c_out: &mut [f32],
    d_inner: usize,
    bc_dim: usize,
    d_conv: usize,
    n_tok: usize,
) {
    let conv_dim = d_inner + 2 * bc_dim;
    let hist_len = d_conv.saturating_sub(1);
    for t in 0..n_tok {
        let row = &xbc_in[t * conv_dim..(t + 1) * conv_dim];
        let x_t = &mut x_out[t * d_inner..(t + 1) * d_inner];
        let b_t = &mut b_out[t * bc_dim..(t + 1) * bc_dim];
        let c_t = &mut c_out[t * bc_dim..(t + 1) * bc_dim];

        for ch in 0..conv_dim {
            let w_ch = &conv_w[ch * d_conv..(ch + 1) * d_conv];
            let mut acc = conv_b.map_or(0.0f32, |b| b[ch]);
            if hist_len > 0 {
                let h_ch = &mut conv_hist[ch * hist_len..(ch + 1) * hist_len];
                for k in 0..hist_len {
                    acc = h_ch[k].mul_add(w_ch[k], acc);
                }
                acc = row[ch].mul_add(w_ch[hist_len], acc);
                h_ch.copy_within(1..hist_len, 0);
                h_ch[hist_len - 1] = row[ch];
            } else if d_conv == 1 {
                acc = row[ch].mul_add(w_ch[0], acc);
            }
            let val = acc / (1.0 + (-acc).exp());
            if ch < d_inner {
                x_t[ch] = val;
            } else if ch < d_inner + bc_dim {
                b_t[ch - d_inner] = val;
            } else {
                c_t[ch - d_inner - bc_dim] = val;
            }
        }
    }
}

/// Applies `y = silu(z) * y` followed by optional per-group RMSNorm (`n_group` groups of `d_inner / n_group`).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_gate_and_grouped_norm(
    kt: &KernelTable,
    y: &mut [f32],
    z: &[f32],
    ssm_norm: Option<&[f32]>,
    scratch: &mut [f32],
    d_inner: usize,
    n_group: usize,
    eps: f32,
    n_tok: usize,
) {
    let total = n_tok * d_inner;
    for i in 0..total {
        let zv = z[i];
        y[i] *= zv / (1.0 + (-zv).exp());
    }
    if let Some(nw) = ssm_norm {
        let group_dim = d_inner / n_group;
        for t in 0..n_tok {
            let yt = &mut y[t * d_inner..(t + 1) * d_inner];
            for g in 0..n_group {
                let s = g * group_dim;
                let e = s + group_dim;
                (kt.rms_norm_f32)(&mut scratch[..group_dim], &yt[s..e], &nw[s..e], eps);
                yt[s..e].copy_from_slice(&scratch[..group_dim]);
            }
        }
    }
}

impl<'a> Mamba2Model<'a> {
    /// Loads and validates a `mamba2`, `nemotron_h`, `falcon-h1`, `granitehybrid`, or `granitemoehybrid` GGUF model.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if !matches!(
            arch.as_str(),
            "mamba2" | "nemotron_h" | "falcon-h1" | "granitehybrid" | "granitemoehybrid"
        ) {
            return Err(ModelError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        let n_layers = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.get_meta(&format!("{pfx}.attention.layer_norm_epsilon")))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        let ssm_conv_kernel = gguf.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_inner_size = gguf.req_usize(&format!("{pfx}.ssm.inner_size"))?;
        let ssm_state_size = gguf.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_head = gguf.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_n_group = gguf
            .get_meta(&format!("{pfx}.ssm.group_count"))
            .and_then(|v| v.as_usize())
            .filter(|&g| g > 0)
            .unwrap_or(1);
        let ssm_norm_eps = gguf
            .get_meta(&format!("{pfx}.ssm.norm_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(rms_eps);

        if ssm_inner_size == 0
            || ssm_n_head == 0
            || ssm_state_size == 0
            || !ssm_inner_size.is_multiple_of(ssm_n_head)
            || !ssm_n_head.is_multiple_of(ssm_n_group)
            || !ssm_inner_size.is_multiple_of(ssm_n_group)
        {
            return Err(ModelError::InvalidConfig(format!(
                "invalid Mamba-2 dimensions: inner={ssm_inner_size}, heads={ssm_n_head}, groups={ssm_n_group}, state={ssm_state_size}"
            )));
        }

        let n_head_per_layer =
            meta_usize_per_layer(gguf, &format!("{pfx}.attention.head_count"), n_layers, 0);
        let default_kv = n_head_per_layer.first().copied().unwrap_or(0);
        let n_head_kv_per_layer = meta_usize_per_layer(
            gguf,
            &format!("{pfx}.attention.head_count_kv"),
            n_layers,
            default_kv,
        );
        let n_ff_per_layer =
            meta_usize_per_layer(gguf, &format!("{pfx}.feed_forward_length"), n_layers, 0);

        let max_n_head = n_head_per_layer.iter().copied().max().unwrap_or(0);
        let head_dim = gguf
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| n_embd.checked_div(max_n_head).unwrap_or(0));
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);
        let rope_type = if pfx == "falcon-h1" {
            RopeType::Neox
        } else {
            RopeType::Norm
        };
        let rope_finetuned = gguf
            .get_meta(&format!("{pfx}.rope.scaling.finetuned"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let use_rope = pfx != "nemotron_h" && rope_finetuned && rope_dim > 0;
        let rope_orig_ctx = gguf
            .get_meta(&format!("{pfx}.rope.scaling.original_context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(max_ctx_train);

        let embedding_scale = gguf
            .get_meta(&format!("{pfx}.embedding_scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);
        let default_attn_scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };
        let (attn_scale, custom_attn_scale) = if let Some(s) = gguf
            .get_meta(&format!("{pfx}.attention.scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
        {
            (s, (s - default_attn_scale).abs() > 1e-12)
        } else {
            (default_attn_scale, false)
        };
        let residual_scale = gguf
            .get_meta(&format!("{pfx}.residual_scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);
        let inv_logit_scale = gguf
            .get_meta(&format!("{pfx}.logit_scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .map_or(1.0, |s| 1.0 / s);

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

        let max_n_ff_meta = n_ff_per_layer.iter().copied().max().unwrap_or(0);
        let moe_params = parse_moe_arch_params(gguf, pfx, &arch, n_layers, max_n_ff_meta)?;

        let mut cfg = Mamba2Config {
            arch: arch.clone(),
            n_layers,
            n_embd,
            vocab_size,
            max_ctx_train,
            rms_eps,
            ssm_conv_kernel,
            ssm_inner_size,
            ssm_state_size,
            ssm_n_head,
            ssm_n_group,
            ssm_norm_eps,
            n_head_per_layer,
            n_head_kv_per_layer,
            head_dim,
            rope_dim,
            rope_freq_base,
            rope_type,
            use_rope,
            rope_orig_ctx,
            n_ff_per_layer,
            embedding_scale,
            attn_scale,
            custom_attn_scale,
            residual_scale,
            inv_logit_scale,
            n_expert: moe_params.as_ref().map_or(0, |m| m.n_expert),
            n_expert_used: moe_params.as_ref().map_or(0, |m| m.n_expert_used),
            max_n_ff_exp: 0,
            max_n_ff_shexp: 0,
        };

        let token_embd = load_mat(gguf, "token_embd.weight", "global", n_embd, vocab_size)?;
        let output = if gguf.tensor_info("output.weight").is_some() {
            load_mat(gguf, "output.weight", "global", n_embd, vocab_size)?
        } else {
            RawMatView {
                name: "output.weight",
                ..token_embd
            }
        };
        let output_bias = load_opt_f32_vec(gguf, "output.bias", "global", vocab_size)?;
        let output_norm = load_f32_vec_any(
            gguf,
            &["output_norm.weight", "norm.weight"],
            "global",
            n_embd,
        )?;

        let half = rope_dim / 2;
        let rope_freqs_short = if pfx == "falcon-h1" || half == 0 {
            None
        } else {
            load_opt_f32_vec_any(
                gguf,
                &[
                    "rope_freqs.weight",
                    "rope_factors_short.weight",
                    "blk.0.rope_factors_short.weight",
                ],
                "global",
                half,
            )?
        };
        let rope_freqs_long = if pfx == "falcon-h1" || half == 0 {
            None
        } else {
            load_opt_f32_vec_any(
                gguf,
                &["rope_factors_long.weight", "blk.0.rope_factors_long.weight"],
                "global",
                half,
            )?
        };
        let build_inv_freqs = |factors: Option<&[f32]>| -> Vec<f64> {
            let mut out = Vec::with_capacity(half);
            let dim_f = rope_dim as f64;
            let base_f = rope_freq_base as f64;
            for i in 0..half {
                let mut freq = base_f.powf(-2.0 * (i as f64) / dim_f);
                if let Some(f) = factors {
                    freq /= f[i] as f64;
                }
                out.push(freq);
            }
            out
        };
        let rope_inv_freqs = build_inv_freqs(rope_freqs_short.as_deref());
        let rope_inv_freqs_long = rope_freqs_long.as_deref().map(|f| build_inv_freqs(Some(f)));

        let mut layers = Vec::with_capacity(n_layers);
        let mut n_state_slots = 0usize;
        let mut n_ssm_layers = 0usize;

        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm = load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, n_embd)?;

            match pfx {
                "mamba2" => {
                    let ssm_idx = n_ssm_layers;
                    n_ssm_layers += 1;
                    let slot = n_state_slots;
                    n_state_slots += 1;
                    let ssm = load_ssm_weights(gguf, &ltag, n_embd, &cfg, ssm_idx, true)?;
                    layers.push(Mamba2LayerWeights {
                        attn_norm,
                        kv_slot: None,
                        ssm_slot: Some(slot),
                        block: Mamba2Block::SsmOnly { ssm },
                    });
                }
                "nemotron_h" => {
                    let nh_kv = cfg.n_head_kv_per_layer[l];
                    let n_ff = cfg.n_ff_per_layer[l];
                    let slot = n_state_slots;
                    n_state_slots += 1;
                    if nh_kv == 0 && n_ff == 0 {
                        let ssm_idx = n_ssm_layers;
                        n_ssm_layers += 1;
                        let ssm = load_ssm_weights(gguf, &ltag, n_embd, &cfg, ssm_idx, true)?;
                        layers.push(Mamba2LayerWeights {
                            attn_norm,
                            kv_slot: None,
                            ssm_slot: Some(slot),
                            block: Mamba2Block::SsmOnly { ssm },
                        });
                    } else if n_ff == 0 {
                        let nh = cfg.n_head_per_layer[l];
                        let attn = load_attn_weights(gguf, &ltag, n_embd, nh, nh_kv, cfg.head_dim)?;
                        layers.push(Mamba2LayerWeights {
                            attn_norm,
                            kv_slot: Some(slot),
                            ssm_slot: None,
                            block: Mamba2Block::AttnOnly { attn },
                        });
                    } else {
                        let w_up =
                            load_mat(gguf, &format!("{ltag}.ffn_up.weight"), &ltag, n_embd, n_ff)?;
                        let w_down = load_mat(
                            gguf,
                            &format!("{ltag}.ffn_down.weight"),
                            &ltag,
                            n_ff,
                            n_embd,
                        )?;
                        let b_up =
                            load_opt_f32_vec(gguf, &format!("{ltag}.ffn_up.bias"), &ltag, n_ff)?;
                        let b_down = load_opt_f32_vec(
                            gguf,
                            &format!("{ltag}.ffn_down.bias"),
                            &ltag,
                            n_embd,
                        )?;
                        layers.push(Mamba2LayerWeights {
                            attn_norm,
                            kv_slot: None,
                            ssm_slot: None,
                            block: Mamba2Block::FfnOnly {
                                ffn: Mamba2FfnWeights::Relu2 {
                                    n_ff,
                                    w_up,
                                    w_down,
                                    b_up,
                                    b_down,
                                },
                            },
                        });
                    }
                }
                "granitehybrid" | "granitemoehybrid" => {
                    let nh_kv = cfg.n_head_kv_per_layer[l];
                    let n_ff = cfg.n_ff_per_layer[l];
                    let ffn_norm =
                        load_f32_vec(gguf, &format!("{ltag}.ffn_norm.weight"), &ltag, n_embd)?;
                    let ffn = if let Some(ref mp) = moe_params
                        && let Some(moe) = load_layer_moe_ffn(gguf, l, &ltag, n_embd, mp)?
                    {
                        cfg.max_n_ff_exp = cfg.max_n_ff_exp.max(moe.cfg.n_ff_exp);
                        cfg.max_n_ff_shexp = cfg.max_n_ff_shexp.max(moe.cfg.n_ff_shexp);
                        Mamba2FfnWeights::SparseMoe(moe)
                    } else {
                        load_swiglu_ffn(gguf, &ltag, n_embd, n_ff)?
                    };
                    let slot = n_state_slots;
                    n_state_slots += 1;
                    if nh_kv == 0 {
                        let ssm_idx = n_ssm_layers;
                        n_ssm_layers += 1;
                        let ssm = load_ssm_weights(gguf, &ltag, n_embd, &cfg, ssm_idx, true)?;
                        layers.push(Mamba2LayerWeights {
                            attn_norm,
                            kv_slot: None,
                            ssm_slot: Some(slot),
                            block: Mamba2Block::SsmAndFfn { ssm, ffn_norm, ffn },
                        });
                    } else {
                        let nh = cfg.n_head_per_layer[l];
                        let attn = load_attn_weights(gguf, &ltag, n_embd, nh, nh_kv, cfg.head_dim)?;
                        layers.push(Mamba2LayerWeights {
                            attn_norm,
                            kv_slot: Some(slot),
                            ssm_slot: None,
                            block: Mamba2Block::AttnAndFfn {
                                attn,
                                ffn_norm,
                                ffn,
                            },
                        });
                    }
                }
                "falcon-h1" => {
                    let nh = cfg.n_head_per_layer[l];
                    let nh_kv = cfg.n_head_kv_per_layer[l];
                    let n_ff = cfg.n_ff_per_layer[l];
                    let kv_slot = n_state_slots;
                    let ssm_slot = n_state_slots + 1;
                    n_state_slots += 2;
                    let ssm_idx = n_ssm_layers;
                    n_ssm_layers += 1;

                    let attn = load_attn_weights(gguf, &ltag, n_embd, nh, nh_kv, cfg.head_dim)?;
                    let ssm = load_ssm_weights(gguf, &ltag, n_embd, &cfg, ssm_idx, false)?;
                    let ffn_norm = load_f32_vec_any(
                        gguf,
                        &[
                            &format!("{ltag}.ffn_norm.weight"),
                            &format!("{ltag}.ffn_norm"),
                        ],
                        &ltag,
                        n_embd,
                    )?;
                    let ffn = load_swiglu_ffn(gguf, &ltag, n_embd, n_ff)?;
                    layers.push(Mamba2LayerWeights {
                        attn_norm,
                        kv_slot: Some(kv_slot),
                        ssm_slot: Some(ssm_slot),
                        block: Mamba2Block::ParallelAttnSsmAndFfn {
                            attn,
                            ssm,
                            ffn_norm,
                            ffn,
                        },
                    });
                }
                _ => unreachable!(),
            }
        }

        Ok(Self {
            cfg,
            token_embd,
            output,
            output_bias,
            output_norm,
            rope_inv_freqs,
            rope_inv_freqs_long,
            layers,
            n_state_slots,
            n_ssm_layers,
            lora: BoundLora::default(),
        })
    }

    #[inline]
    pub fn rope_inv_freqs_for_ctx(&self, max_ctx: usize) -> &[f64] {
        if max_ctx > self.cfg.rope_orig_ctx
            && let Some(ref long) = self.rope_inv_freqs_long
        {
            return long.as_slice();
        }
        self.rope_inv_freqs.as_slice()
    }

    pub fn bind_lora(&mut self, set: &LoraSet) {
        let bound = BoundLora::from_lora_set(set);
        self.token_embd.lora_slot = bound.slot_for(self.token_embd.name);
        self.output.lora_slot = bound.slot_for(self.output.name);
        let bind_ssm = |ssm: &mut Mamba2SsmWeights<'_>| {
            ssm.ssm_in.lora_slot = bound.slot_for(ssm.ssm_in.name);
            ssm.ssm_out.lora_slot = bound.slot_for(ssm.ssm_out.name);
        };
        let bind_attn = |attn: &mut Mamba2AttnWeights<'_>| {
            attn.wq.lora_slot = bound.slot_for(attn.wq.name);
            attn.wk.lora_slot = bound.slot_for(attn.wk.name);
            attn.wv.lora_slot = bound.slot_for(attn.wv.name);
            attn.wo.lora_slot = bound.slot_for(attn.wo.name);
        };
        let bind_ffn = |ffn: &mut Mamba2FfnWeights<'_>| match ffn {
            Mamba2FfnWeights::SwiGlu {
                w_gate,
                w_up,
                w_down,
                ..
            } => {
                w_gate.lora_slot = bound.slot_for(w_gate.name);
                w_up.lora_slot = bound.slot_for(w_up.name);
                w_down.lora_slot = bound.slot_for(w_down.name);
            }
            Mamba2FfnWeights::Relu2 { w_up, w_down, .. } => {
                w_up.lora_slot = bound.slot_for(w_up.name);
                w_down.lora_slot = bound.slot_for(w_down.name);
            }
            Mamba2FfnWeights::SparseMoe(moe) => moe.bind_lora(&bound),
        };
        for lw in &mut self.layers {
            match &mut lw.block {
                Mamba2Block::SsmOnly { ssm } => bind_ssm(ssm),
                Mamba2Block::AttnOnly { attn } => bind_attn(attn),
                Mamba2Block::FfnOnly { ffn } => bind_ffn(ffn),
                Mamba2Block::SsmAndFfn { ssm, ffn, .. } => {
                    bind_ssm(ssm);
                    bind_ffn(ffn);
                }
                Mamba2Block::AttnAndFfn { attn, ffn, .. } => {
                    bind_attn(attn);
                    bind_ffn(ffn);
                }
                Mamba2Block::ParallelAttnSsmAndFfn { attn, ssm, ffn, .. } => {
                    bind_attn(attn);
                    bind_ssm(ssm);
                    bind_ffn(ffn);
                }
            }
        }
        self.lora = bound;
    }

    pub fn forward_token(
        &self,
        state: &mut Mamba2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        let toks = [token_id];
        self.forward_chunk_internal(state, None, pool, kt, &toks, pos, compute_logits)
    }

    /// Runs a single-position forward pass from a precomputed input embedding (`embed.len() == n_embd`),
    /// used for multimodal vision token injection (`ADR-019`).
    pub fn forward_embed(
        &self,
        state: &mut Mamba2State,
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
        let d = self.cfg.n_embd;
        if embed.len() != d {
            return Err(ModelError::InvalidConfig(format!(
                "forward_embed length mismatch: expected {d}, got {}",
                embed.len()
            )));
        }
        state.x[..d].copy_from_slice(embed);
        self.forward_chunk_from_x(state, None, pool, kt, 1, pos, compute_logits)
    }

    pub fn forward_batch(
        &self,
        state: &mut Mamba2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
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

        let total = token_ids.len();
        let mut offset = 0usize;
        while offset < total {
            let b = (total - offset).min(MAX_CPU_BATCH);
            let is_last = offset + b == total;
            self.forward_chunk_internal(
                state,
                None,
                pool,
                kt,
                &token_ids[offset..offset + b],
                start_pos + offset,
                is_last && compute_last_logits,
            )?;
            offset += b;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_internal(
        &self,
        state: &mut Mamba2State,
        spec_buf: Option<&mut SpecVerifyBuf>,
        pool: &ThreadPool,
        kt: &KernelTable,
        chunk: &[u32],
        chunk_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        let b = chunk.len();
        let end_pos = chunk_pos + b - 1;
        if end_pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: end_pos,
                max_ctx: state.max_ctx,
            });
        }
        for &tid in chunk {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }

        let c = &self.cfg;
        let d = c.n_embd;

        for (t, &tok) in chunk.iter().enumerate() {
            let dst = &mut state.x[t * d..(t + 1) * d];
            decode_embedding_row(
                self.token_embd.dtype,
                self.token_embd.bytes,
                d,
                tok as usize,
                dst,
            );
            self.lora
                .apply_embed_row(self.token_embd.lora_slot, kt, tok as usize, dst);
        }
        scale_slice_inplace(&mut state.x[..b * d], c.embedding_scale);

        self.forward_chunk_from_x(state, spec_buf, pool, kt, b, chunk_pos, compute_last_logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk_from_x(
        &self,
        state: &mut Mamba2State,
        mut spec_buf: Option<&mut SpecVerifyBuf>,
        pool: &ThreadPool,
        kt: &KernelTable,
        b: usize,
        chunk_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        let c = &self.cfg;
        let d = c.n_embd;
        let single_tok = b == 1;
        let res_scale = c.residual_scale;

        for lw in &self.layers {
            if single_tok {
                rms_norm(&mut state.xb[..d], &state.x[..d], &lw.attn_norm, c.rms_eps);
            } else {
                rms_norm_batch_parallel(
                    pool,
                    kt,
                    &mut state.xb[..b * d],
                    &state.x[..b * d],
                    &lw.attn_norm,
                    c.rms_eps,
                    d,
                    b,
                );
            }

            match &lw.block {
                Mamba2Block::SsmOnly { ssm } => {
                    let ssm_slot = lw.ssm_slot.expect("ssm_slot");
                    self.run_ssm_branch(
                        ssm,
                        ssm_slot,
                        state,
                        spec_buf.as_deref_mut(),
                        pool,
                        kt,
                        b,
                        single_tok,
                    );
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                }
                Mamba2Block::AttnOnly { attn } => {
                    let kv_slot = lw.kv_slot.expect("kv_slot");
                    self.run_attn_branch(attn, kv_slot, state, pool, kt, chunk_pos, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                }
                Mamba2Block::FfnOnly { ffn } => {
                    self.run_ffn_branch(ffn, state, pool, kt, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                }
                Mamba2Block::SsmAndFfn { ssm, ffn_norm, ffn } => {
                    let ssm_slot = lw.ssm_slot.expect("ssm_slot");
                    self.run_ssm_branch(
                        ssm,
                        ssm_slot,
                        state,
                        spec_buf.as_deref_mut(),
                        pool,
                        kt,
                        b,
                        single_tok,
                    );
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                    if single_tok {
                        rms_norm(&mut state.xb[..d], &state.x[..d], ffn_norm, c.rms_eps);
                    } else {
                        rms_norm_batch_parallel(
                            pool,
                            kt,
                            &mut state.xb[..b * d],
                            &state.x[..b * d],
                            ffn_norm,
                            c.rms_eps,
                            d,
                            b,
                        );
                    }
                    self.run_ffn_branch(ffn, state, pool, kt, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                }
                Mamba2Block::AttnAndFfn {
                    attn,
                    ffn_norm,
                    ffn,
                } => {
                    let kv_slot = lw.kv_slot.expect("kv_slot");
                    self.run_attn_branch(attn, kv_slot, state, pool, kt, chunk_pos, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                    if single_tok {
                        rms_norm(&mut state.xb[..d], &state.x[..d], ffn_norm, c.rms_eps);
                    } else {
                        rms_norm_batch_parallel(
                            pool,
                            kt,
                            &mut state.xb[..b * d],
                            &state.x[..b * d],
                            ffn_norm,
                            c.rms_eps,
                            d,
                            b,
                        );
                    }
                    self.run_ffn_branch(ffn, state, pool, kt, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                }
                Mamba2Block::ParallelAttnSsmAndFfn {
                    attn,
                    ssm,
                    ffn_norm,
                    ffn,
                } => {
                    // Save `attn_norm(x)` in `xb_par` because `run_attn_branch` reuses `xb` for attention output.
                    state.xb_par[..b * d].copy_from_slice(&state.xb[..b * d]);
                    let kv_slot = lw.kv_slot.expect("kv_slot");
                    let ssm_slot = lw.ssm_slot.expect("ssm_slot");
                    self.run_attn_branch(attn, kv_slot, state, pool, kt, chunk_pos, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                    state.xb[..b * d].copy_from_slice(&state.xb_par[..b * d]);
                    self.run_ssm_branch(
                        ssm,
                        ssm_slot,
                        state,
                        spec_buf.as_deref_mut(),
                        pool,
                        kt,
                        b,
                        single_tok,
                    );
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                    if single_tok {
                        rms_norm(&mut state.xb[..d], &state.x[..d], ffn_norm, c.rms_eps);
                    } else {
                        rms_norm_batch_parallel(
                            pool,
                            kt,
                            &mut state.xb[..b * d],
                            &state.x[..b * d],
                            ffn_norm,
                            c.rms_eps,
                            d,
                            b,
                        );
                    }
                    self.run_ffn_branch(ffn, state, pool, kt, b, single_tok);
                    for i in 0..(b * d) {
                        state.x[i] += res_scale * state.xb2[i];
                    }
                }
            }
        }

        state
            .last_hidden
            .copy_from_slice(&state.x[(b - 1) * d..b * d]);
        if compute_last_logits {
            let last_x = &state.x[(b - 1) * d..b * d];
            rms_norm(&mut state.xb[..d], last_x, &self.output_norm, c.rms_eps);
            matvec_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes,
                &state.xb[..d],
                &mut state.mv_scratch,
                &mut state.logits,
                self.output.rows,
                self.output.cols,
            );
            self.lora.apply_linear(
                self.output.lora_slot,
                pool,
                kt,
                &state.xb[..d],
                &mut state.logits,
                1,
                &mut state.lora_scratch,
            );
            if let Some(ref ob) = self.output_bias {
                add_bias_rows(&mut state.logits, ob, 1);
            }
            scale_slice_inplace(&mut state.logits, c.inv_logit_scale);
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_ssm_branch(
        &self,
        ssm: &Mamba2SsmWeights<'_>,
        ssm_slot: usize,
        state: &mut Mamba2State,
        spec_buf: Option<&mut SpecVerifyBuf>,
        pool: &ThreadPool,
        kt: &KernelTable,
        b: usize,
        single_tok: bool,
    ) {
        let c = &self.cfg;
        let d = c.n_embd;
        let d_inner = c.ssm_inner_size;
        let bc_dim = c.bc_dim();
        let conv_dim = c.conv_dim();
        let d_in_proj = c.d_in_proj();
        let n_heads = c.ssm_n_head;

        if single_tok {
            matvec_f32(
                pool,
                kt,
                ssm.ssm_in.dtype,
                ssm.ssm_in.bytes,
                &state.xb[..d],
                &mut state.mv_scratch,
                &mut state.in_proj[..d_in_proj],
                d_in_proj,
                d,
            );
        } else {
            matmul_f32(
                pool,
                kt,
                ssm.ssm_in.dtype,
                ssm.ssm_in.bytes,
                &state.xb[..b * d],
                &mut state.mv_scratch,
                &mut state.in_proj[..b * d_in_proj],
                d_in_proj,
                d,
                b,
            );
        }
        self.lora.apply_linear(
            ssm.ssm_in.lora_slot,
            pool,
            kt,
            &state.xb[..b * d],
            &mut state.in_proj[..b * d_in_proj],
            b,
            &mut state.lora_scratch,
        );

        for t in 0..b {
            let row = &state.in_proj[t * d_in_proj..(t + 1) * d_in_proj];
            state.ssm_z[t * d_inner..(t + 1) * d_inner].copy_from_slice(&row[..d_inner]);
            state.ssm_xbc[t * conv_dim..(t + 1) * conv_dim]
                .copy_from_slice(&row[d_inner..d_inner + conv_dim]);
            state.ssm_dt[t * n_heads..(t + 1) * n_heads]
                .copy_from_slice(&row[d_inner + conv_dim..d_in_proj]);
        }

        if let Some(sb) = spec_buf {
            let idx = ssm.ssm_spec_idx;
            let xbc_off = idx * MAX_SPEC_WINDOW * conv_dim;
            let dt_off = idx * MAX_SPEC_WINDOW * n_heads;
            sb.ssm_xbc_win[xbc_off..xbc_off + b * conv_dim]
                .copy_from_slice(&state.ssm_xbc[..b * conv_dim]);
            sb.ssm_dt_win[dt_off..dt_off + b * n_heads]
                .copy_from_slice(&state.ssm_dt[..b * n_heads]);

            let LayerState::Ssm {
                state: ref perm_s,
                conv_hist: ref perm_h,
            } = state.layers[ssm_slot]
            else {
                unreachable!("expected LayerState::Ssm");
            };
            sb.tmp_ssm_state.copy_from_slice(perm_s);
            sb.tmp_conv_hist.copy_from_slice(perm_h);

            mamba2_conv1d_silu_split(
                &state.ssm_xbc[..b * conv_dim],
                &mut sb.tmp_conv_hist,
                &ssm.conv_w,
                ssm.conv_b.as_deref(),
                &mut state.ssm_x[..b * d_inner],
                &mut state.ssm_b[..b * bc_dim],
                &mut state.ssm_c[..b * bc_dim],
                d_inner,
                bc_dim,
                c.ssm_conv_kernel,
                b,
            );
            mamba2_scan_parallel(
                pool,
                kt,
                &state.ssm_x[..b * d_inner],
                &state.ssm_b[..b * bc_dim],
                &state.ssm_c[..b * bc_dim],
                &state.ssm_dt[..b * n_heads],
                &ssm.dt_bias,
                &ssm.ssm_a,
                &ssm.ssm_d,
                &mut sb.tmp_ssm_state,
                &mut state.ssm_y[..b * d_inner],
                n_heads,
                c.ssm_n_group,
                c.ssm_head_dim(),
                c.ssm_state_size,
                b,
            );
        } else {
            let LayerState::Ssm {
                state: ref mut ssm_s,
                conv_hist: ref mut ssm_h,
            } = state.layers[ssm_slot]
            else {
                unreachable!("expected LayerState::Ssm");
            };
            mamba2_conv1d_silu_split(
                &state.ssm_xbc[..b * conv_dim],
                ssm_h,
                &ssm.conv_w,
                ssm.conv_b.as_deref(),
                &mut state.ssm_x[..b * d_inner],
                &mut state.ssm_b[..b * bc_dim],
                &mut state.ssm_c[..b * bc_dim],
                d_inner,
                bc_dim,
                c.ssm_conv_kernel,
                b,
            );
            mamba2_scan_parallel(
                pool,
                kt,
                &state.ssm_x[..b * d_inner],
                &state.ssm_b[..b * bc_dim],
                &state.ssm_c[..b * bc_dim],
                &state.ssm_dt[..b * n_heads],
                &ssm.dt_bias,
                &ssm.ssm_a,
                &ssm.ssm_d,
                ssm_s,
                &mut state.ssm_y[..b * d_inner],
                n_heads,
                c.ssm_n_group,
                c.ssm_head_dim(),
                c.ssm_state_size,
                b,
            );
        }

        mamba2_gate_and_grouped_norm(
            kt,
            &mut state.ssm_y[..b * d_inner],
            &state.ssm_z[..b * d_inner],
            ssm.ssm_norm.as_deref(),
            &mut state.xb[..c.ssm_group_dim()],
            d_inner,
            c.ssm_n_group,
            c.ssm_norm_eps,
            b,
        );

        if single_tok {
            matvec_f32(
                pool,
                kt,
                ssm.ssm_out.dtype,
                ssm.ssm_out.bytes,
                &state.ssm_y[..d_inner],
                &mut state.mv_scratch,
                &mut state.xb2[..d],
                d,
                d_inner,
            );
        } else {
            matmul_f32(
                pool,
                kt,
                ssm.ssm_out.dtype,
                ssm.ssm_out.bytes,
                &state.ssm_y[..b * d_inner],
                &mut state.mv_scratch,
                &mut state.xb2[..b * d],
                d,
                d_inner,
                b,
            );
        }
        self.lora.apply_linear(
            ssm.ssm_out.lora_slot,
            pool,
            kt,
            &state.ssm_y[..b * d_inner],
            &mut state.xb2[..b * d],
            b,
            &mut state.lora_scratch,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attn_branch(
        &self,
        attn: &Mamba2AttnWeights<'_>,
        kv_slot: usize,
        state: &mut Mamba2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        chunk_pos: usize,
        b: usize,
        single_tok: bool,
    ) {
        let c = &self.cfg;
        let d = c.n_embd;
        let hd = c.head_dim;
        let q_dim = attn.n_head * hd;
        let kv_dim = attn.n_head_kv * hd;

        if single_tok {
            matvec_triple_f32(
                pool,
                kt,
                attn.wq.dtype,
                attn.wq.bytes,
                q_dim,
                &mut state.q[..q_dim],
                attn.wk.dtype,
                attn.wk.bytes,
                kv_dim,
                &mut state.k[..kv_dim],
                attn.wv.dtype,
                attn.wv.bytes,
                kv_dim,
                &mut state.v[..kv_dim],
                &state.xb[..d],
                &mut state.mv_scratch,
                d,
            );
        } else {
            matmul_triple_f32(
                pool,
                kt,
                attn.wq.dtype,
                attn.wq.bytes,
                q_dim,
                &mut state.q[..b * q_dim],
                attn.wk.dtype,
                attn.wk.bytes,
                kv_dim,
                &mut state.k[..b * kv_dim],
                attn.wv.dtype,
                attn.wv.bytes,
                kv_dim,
                &mut state.v[..b * kv_dim],
                &state.xb[..b * d],
                &mut state.mv_scratch,
                d,
                b,
            );
        }
        if !self.lora.is_empty() {
            self.lora.apply_linear_slice(
                attn.wq.lora_slot,
                pool,
                kt,
                &state.xb[..b * d],
                &mut state.q[..b * q_dim],
                b,
                attn.wq.row_offset,
                q_dim,
                &mut state.lora_scratch,
            );
            self.lora.apply_linear_slice(
                attn.wk.lora_slot,
                pool,
                kt,
                &state.xb[..b * d],
                &mut state.k[..b * kv_dim],
                b,
                attn.wk.row_offset,
                kv_dim,
                &mut state.lora_scratch,
            );
            self.lora.apply_linear_slice(
                attn.wv.lora_slot,
                pool,
                kt,
                &state.xb[..b * d],
                &mut state.v[..b * kv_dim],
                b,
                attn.wv.row_offset,
                kv_dim,
                &mut state.lora_scratch,
            );
        }
        if let Some(ref bq) = attn.bq {
            add_bias_rows(&mut state.q[..b * q_dim], bq, b);
        }
        if let Some(ref bk) = attn.bk {
            add_bias_rows(&mut state.k[..b * kv_dim], bk, b);
        }
        if let Some(ref bv) = attn.bv {
            add_bias_rows(&mut state.v[..b * kv_dim], bv, b);
        }

        let rope_freqs = self.rope_inv_freqs_for_ctx(state.max_ctx);
        for t in 0..b {
            let pos = chunk_pos + t;
            let qt = &mut state.q[t * q_dim..(t + 1) * q_dim];
            let kt_vec = &mut state.k[t * kv_dim..(t + 1) * kv_dim];
            let vt_vec = &state.v[t * kv_dim..(t + 1) * kv_dim];
            if c.use_rope {
                rope_compute_cos_sin(rope_freqs, pos, &mut state.rope_cos, &mut state.rope_sin);
                rope_inplace_tables(
                    qt,
                    attn.n_head,
                    hd,
                    c.rope_dim,
                    &state.rope_cos,
                    &state.rope_sin,
                    c.rope_type,
                );
                rope_inplace_tables(
                    kt_vec,
                    attn.n_head_kv,
                    hd,
                    c.rope_dim,
                    &state.rope_cos,
                    &state.rope_sin,
                    c.rope_type,
                );
            }
            state.layers[kv_slot].store_kv_at(kt, pos, kt_vec, vt_vec);
        }

        let LayerState::Kv {
            kv_type,
            ref k_cache,
            ref v_cache,
            ref k_q8,
            ref v_q8,
            ..
        } = state.layers[kv_slot]
        else {
            unreachable!("expected LayerState::Kv");
        };

        match kv_type {
            KvType::F16 => {
                if single_tok {
                    if c.custom_attn_scale {
                        gqa_attention_f16_kv_ext_parallel(
                            pool,
                            kt,
                            &mut state.xb[..q_dim],
                            &state.q[..q_dim],
                            k_cache,
                            v_cache,
                            attn.n_head,
                            attn.n_head_kv,
                            hd,
                            chunk_pos,
                            c.attn_scale,
                            None,
                            state.max_ctx,
                            &mut state.scores,
                        );
                    } else {
                        gqa_attention_f16_kv_parallel(
                            pool,
                            kt,
                            &mut state.xb[..q_dim],
                            &state.q[..q_dim],
                            k_cache,
                            v_cache,
                            attn.n_head,
                            attn.n_head_kv,
                            hd,
                            chunk_pos + 1,
                            state.max_ctx,
                            &mut state.scores,
                        );
                    }
                } else if c.custom_attn_scale {
                    gqa_attention_f16_kv_ext_batch_parallel(
                        pool,
                        kt,
                        &mut state.xb[..b * q_dim],
                        &state.q[..b * q_dim],
                        k_cache,
                        v_cache,
                        attn.n_head,
                        attn.n_head_kv,
                        hd,
                        chunk_pos,
                        b,
                        c.attn_scale,
                        None,
                        state.max_ctx,
                        &mut state.scores,
                    );
                } else {
                    gqa_attention_f16_kv_batch_parallel(
                        pool,
                        kt,
                        &mut state.xb[..b * q_dim],
                        &state.q[..b * q_dim],
                        k_cache,
                        v_cache,
                        attn.n_head,
                        attn.n_head_kv,
                        hd,
                        chunk_pos,
                        b,
                        state.max_ctx,
                        &mut state.scores,
                    );
                }
            }
            KvType::Q8_0 => {
                let _ = QK8_0;
                if single_tok {
                    gqa_attention_q8_0_kv_ext_parallel(
                        pool,
                        kt,
                        &mut state.xb[..q_dim],
                        &state.q[..q_dim],
                        k_q8,
                        v_q8,
                        attn.n_head,
                        attn.n_head_kv,
                        hd,
                        chunk_pos,
                        c.attn_scale,
                        None,
                        state.max_ctx,
                        &mut state.scores,
                    );
                } else {
                    gqa_attention_q8_0_kv_ext_batch_parallel(
                        pool,
                        kt,
                        &mut state.xb[..b * q_dim],
                        &state.q[..b * q_dim],
                        k_q8,
                        v_q8,
                        attn.n_head,
                        attn.n_head_kv,
                        hd,
                        chunk_pos,
                        b,
                        c.attn_scale,
                        None,
                        state.max_ctx,
                        &mut state.scores,
                    );
                }
            }
        }

        if single_tok {
            matvec_f32(
                pool,
                kt,
                attn.wo.dtype,
                attn.wo.bytes,
                &state.xb[..q_dim],
                &mut state.mv_scratch,
                &mut state.xb2[..d],
                d,
                q_dim,
            );
        } else {
            matmul_f32(
                pool,
                kt,
                attn.wo.dtype,
                attn.wo.bytes,
                &state.xb[..b * q_dim],
                &mut state.mv_scratch,
                &mut state.xb2[..b * d],
                d,
                q_dim,
                b,
            );
        }
        self.lora.apply_linear(
            attn.wo.lora_slot,
            pool,
            kt,
            &state.xb[..b * q_dim],
            &mut state.xb2[..b * d],
            b,
            &mut state.lora_scratch,
        );
        if let Some(ref bo) = attn.bo {
            add_bias_rows(&mut state.xb2[..b * d], bo, b);
        }
    }

    fn run_ffn_branch(
        &self,
        ffn: &Mamba2FfnWeights<'_>,
        state: &mut Mamba2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        b: usize,
        single_tok: bool,
    ) {
        let d = self.cfg.n_embd;
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
                if single_tok {
                    matvec_pair_f32(
                        pool,
                        kt,
                        w_gate.dtype,
                        w_gate.bytes,
                        ff,
                        &mut state.hb[..ff],
                        w_up.dtype,
                        w_up.bytes,
                        ff,
                        &mut state.hb2[..ff],
                        &state.xb[..d],
                        &mut state.mv_scratch,
                        d,
                    );
                } else {
                    matmul_pair_f32(
                        pool,
                        kt,
                        w_gate.dtype,
                        w_gate.bytes,
                        ff,
                        &mut state.hb[..b * ff],
                        w_up.dtype,
                        w_up.bytes,
                        ff,
                        &mut state.hb2[..b * ff],
                        &state.xb[..b * d],
                        &mut state.mv_scratch,
                        d,
                        b,
                    );
                }
                if !self.lora.is_empty() {
                    self.lora.apply_linear(
                        w_gate.lora_slot,
                        pool,
                        kt,
                        &state.xb[..b * d],
                        &mut state.hb[..b * ff],
                        b,
                        &mut state.lora_scratch,
                    );
                    self.lora.apply_linear(
                        w_up.lora_slot,
                        pool,
                        kt,
                        &state.xb[..b * d],
                        &mut state.hb2[..b * ff],
                        b,
                        &mut state.lora_scratch,
                    );
                }
                if let Some(bg) = b_gate {
                    add_bias_rows(&mut state.hb[..b * ff], bg, b);
                }
                if let Some(bu) = b_up {
                    add_bias_rows(&mut state.hb2[..b * ff], bu, b);
                }
                swiglu_inplace_parallel(pool, &mut state.hb[..b * ff], &state.hb2[..b * ff]);
                if single_tok {
                    matvec_f32(
                        pool,
                        kt,
                        w_down.dtype,
                        w_down.bytes,
                        &state.hb[..ff],
                        &mut state.mv_scratch,
                        &mut state.xb2[..d],
                        d,
                        ff,
                    );
                } else {
                    matmul_f32(
                        pool,
                        kt,
                        w_down.dtype,
                        w_down.bytes,
                        &state.hb[..b * ff],
                        &mut state.mv_scratch,
                        &mut state.xb2[..b * d],
                        d,
                        ff,
                        b,
                    );
                }
                self.lora.apply_linear(
                    w_down.lora_slot,
                    pool,
                    kt,
                    &state.hb[..b * ff],
                    &mut state.xb2[..b * d],
                    b,
                    &mut state.lora_scratch,
                );
                if let Some(bd) = b_down {
                    add_bias_rows(&mut state.xb2[..b * d], bd, b);
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
                if single_tok {
                    matvec_f32(
                        pool,
                        kt,
                        w_up.dtype,
                        w_up.bytes,
                        &state.xb[..d],
                        &mut state.mv_scratch,
                        &mut state.hb[..ff],
                        ff,
                        d,
                    );
                } else {
                    matmul_f32(
                        pool,
                        kt,
                        w_up.dtype,
                        w_up.bytes,
                        &state.xb[..b * d],
                        &mut state.mv_scratch,
                        &mut state.hb[..b * ff],
                        ff,
                        d,
                        b,
                    );
                }
                self.lora.apply_linear(
                    w_up.lora_slot,
                    pool,
                    kt,
                    &state.xb[..b * d],
                    &mut state.hb[..b * ff],
                    b,
                    &mut state.lora_scratch,
                );
                if let Some(bu) = b_up {
                    add_bias_rows(&mut state.hb[..b * ff], bu, b);
                }
                relu2_inplace_parallel(pool, &mut state.hb[..b * ff]);
                if single_tok {
                    matvec_f32(
                        pool,
                        kt,
                        w_down.dtype,
                        w_down.bytes,
                        &state.hb[..ff],
                        &mut state.mv_scratch,
                        &mut state.xb2[..d],
                        d,
                        ff,
                    );
                } else {
                    matmul_f32(
                        pool,
                        kt,
                        w_down.dtype,
                        w_down.bytes,
                        &state.hb[..b * ff],
                        &mut state.mv_scratch,
                        &mut state.xb2[..b * d],
                        d,
                        ff,
                        b,
                    );
                }
                self.lora.apply_linear(
                    w_down.lora_slot,
                    pool,
                    kt,
                    &state.hb[..b * ff],
                    &mut state.xb2[..b * d],
                    b,
                    &mut state.lora_scratch,
                );
                if let Some(bd) = b_down {
                    add_bias_rows(&mut state.xb2[..b * d], bd, b);
                }
            }
            Mamba2FfnWeights::SparseMoe(moe) => {
                eval_sparse_moe_ffn(
                    moe,
                    &self.lora,
                    pool,
                    kt,
                    &state.xb[..b * d],
                    &mut state.xb2[..b * d],
                    d,
                    b,
                    &mut state.mv_scratch,
                    &mut state.lora_scratch,
                    &mut state.moe_scratch,
                );
            }
        }
    }

    /// Read-only speculative verification over `window_tokens` (`1 <= W <= MAX_SPEC_WINDOW`):
    /// leaves permanent `LayerState::Ssm` untouched and records `xBC` / `dt_raw` for `commit_speculative`.
    pub fn verify_speculative(
        &self,
        state: &mut Mamba2State,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        window_tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), ModelError> {
        let w = window_tokens.len();
        assert!((1..=MAX_SPEC_WINDOW).contains(&w));
        self.forward_chunk_internal(
            state,
            Some(spec_buf),
            pool,
            kt,
            window_tokens,
            start_pos,
            false,
        )?;
        let d = self.cfg.n_embd;
        let vocab = self.cfg.vocab_size;
        spec_buf.last_window_len = w;
        spec_buf.hidden_win[..w * d].copy_from_slice(&state.x[..w * d]);

        rms_norm_batch_parallel(
            pool,
            kt,
            &mut spec_buf.norm_win[..w * d],
            &spec_buf.hidden_win[..w * d],
            &self.output_norm,
            self.cfg.rms_eps,
            d,
            w,
        );
        matmul_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes,
            &spec_buf.norm_win[..w * d],
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
            &spec_buf.norm_win[..w * d],
            &mut spec_buf.logits_win[..w * vocab],
            w,
            &mut state.lora_scratch,
        );
        if let Some(ref ob) = self.output_bias {
            add_bias_rows(&mut spec_buf.logits_win[..w * vocab], ob, w);
        }
        scale_slice_inplace(
            &mut spec_buf.logits_win[..w * vocab],
            self.cfg.inv_logit_scale,
        );
        Ok(())
    }

    /// Replays ONLY the `committed` (`1 <= committed <= W`) accepted steps into permanent `LayerState::Ssm`.
    pub fn commit_speculative(
        &self,
        state: &mut Mamba2State,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        committed: usize,
    ) {
        assert!((1..=spec_buf.last_window_len).contains(&committed));
        let c = &self.cfg;
        let d_inner = c.ssm_inner_size;
        let bc_dim = c.bc_dim();
        let conv_dim = c.conv_dim();
        let n_heads = c.ssm_n_head;

        for lw in &self.layers {
            let ssm = match &lw.block {
                Mamba2Block::SsmOnly { ssm }
                | Mamba2Block::SsmAndFfn { ssm, .. }
                | Mamba2Block::ParallelAttnSsmAndFfn { ssm, .. } => ssm,
                _ => continue,
            };
            let ssm_slot = lw.ssm_slot.expect("ssm_slot");
            let idx = ssm.ssm_spec_idx;
            let xbc_off = idx * MAX_SPEC_WINDOW * conv_dim;
            let dt_off = idx * MAX_SPEC_WINDOW * n_heads;

            let LayerState::Ssm {
                state: ref mut ssm_s,
                conv_hist: ref mut ssm_h,
            } = state.layers[ssm_slot]
            else {
                unreachable!("expected LayerState::Ssm");
            };

            mamba2_conv1d_silu_split(
                &spec_buf.ssm_xbc_win[xbc_off..xbc_off + committed * conv_dim],
                ssm_h,
                &ssm.conv_w,
                ssm.conv_b.as_deref(),
                &mut spec_buf.replay_q[..committed * d_inner],
                &mut spec_buf.replay_k[..committed * bc_dim],
                &mut spec_buf.replay_v[..committed * bc_dim],
                d_inner,
                bc_dim,
                c.ssm_conv_kernel,
                committed,
            );
            mamba2_scan_parallel(
                pool,
                kt,
                &spec_buf.replay_q[..committed * d_inner],
                &spec_buf.replay_k[..committed * bc_dim],
                &spec_buf.replay_v[..committed * bc_dim],
                &spec_buf.ssm_dt_win[dt_off..dt_off + committed * n_heads],
                &ssm.dt_bias,
                &ssm.ssm_a,
                &ssm.ssm_d,
                ssm_s,
                &mut spec_buf.replay_out[..committed * d_inner],
                n_heads,
                c.ssm_n_group,
                c.ssm_head_dim(),
                c.ssm_state_size,
                committed,
            );
        }

        let last = committed - 1;
        state
            .logits
            .copy_from_slice(spec_buf.logits_at(last, c.vocab_size));
        state
            .last_hidden
            .copy_from_slice(spec_buf.hidden_at(last, c.n_embd));
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
        scale_slice_inplace(out_emb, self.cfg.embedding_scale);
    }

    pub fn output_head_raw(&self) -> (&[f32], f32, DType, &[u8], Option<f32>) {
        (
            &self.output_norm,
            self.cfg.rms_eps,
            self.output.dtype,
            self.output.bytes,
            None,
        )
    }

    pub fn project_logits_from_hidden(
        &self,
        state: &mut Mamba2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        hidden: &[f32],
        out_logits: &mut [f32],
    ) {
        let d = self.cfg.n_embd;
        rms_norm(
            &mut state.xb[..d],
            hidden,
            &self.output_norm,
            self.cfg.rms_eps,
        );
        matvec_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes,
            &state.xb[..d],
            &mut state.mv_scratch,
            out_logits,
            self.output.rows,
            self.output.cols,
        );
        self.lora.apply_linear(
            self.output.lora_slot,
            pool,
            kt,
            &state.xb[..d],
            out_logits,
            1,
            &mut state.lora_scratch,
        );
        if let Some(ref ob) = self.output_bias {
            add_bias_rows(out_logits, ob, 1);
        }
        scale_slice_inplace(out_logits, self.cfg.inv_logit_scale);
    }
}
