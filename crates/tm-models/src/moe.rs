//! Reusable fine-grained Sparse Mixture-of-Experts FFN (`SparseMoeFfn`, `ADR-018`).
//!
//! Covers all upstream `llama.cpp` sub-8B sparse MoE GGUF architectures across both
//! standard Attention (`LlamaModel`) and Mamba-2 Hybrid (`Mamba2Model`) backbones:
//! - `"olmoe"` (`OLMoE-1B-7B`: 64 experts, top-8, Softmax, `norm_w = false`)
//! - `"granitemoe"` / `"granite"` (`Granite-3.1-3B-A800M`: Softmax, `norm_w = true`, optional shared expert)
//! - `"granitehybrid"` / `"granitemoehybrid"` (`Granite-4.0-H-Tiny 7B-A1B`: Mamba-2 + GQA + MoE)
//! - `"qwen2moe"` (`Qwen1.5/2-MoE-A2.7B`: Softmax, `norm_w = false`, sigmoid-gated shared expert)
//! - `"qwen3moe"` (`Qwen3-MoE`: Softmax, `norm_w = true`, optional shared expert)
//! - `"bailingmoe2"` (`Ling-mini` / `Ling-3.0-tiny`: leading dense layers + Softmax/Sigmoid MoE + `exp_probs_b` + shared expert)
//!
//! Strictly obeys `AGENTS.md`:
//! - `#![forbid(unsafe_code)]` (inherited from `tm-models` crate root);
//! - Zero heap allocations on token generation / batch prefill paths (`MoeScratch` pre-allocated);
//! - Full per-tensor `DType` dispatch (`Rule #5`);
//! - Deterministic `0..n_expert_used` reduction order so single-token, multi-thread, and batched
//!   prefill match to **0 ULP (`to_bits()`)** (`§9`).

use tm_core::{DType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, ThreadPool, matmul_f32, matvec_f32, matvec_pair_f32,
    softplus_f32, swiglu_inplace_parallel,
};
use tm_gguf::{GgufError, GgufFile};
use tm_lora::{BoundLora, LoraScratch};

use crate::llama::{ModelError, RawMatView};

/// Minimum F16 normal (`2^-14`), used by `llama-graph.cpp:2513` when normalizing router weights.
pub const MOE_NORM_CLAMP_F16_MIN: f32 = 6.103_515_6e-5;

/// Router gating function (`enum llama_expert_gating_func_type` in `llama-hparams.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertGatingFunc {
    Softmax,
    Sigmoid,
    SoftmaxWeight,
    SqrtSoftplus,
}

impl ExpertGatingFunc {
    pub fn from_gguf_u32(val: u32) -> Result<Self, ModelError> {
        match val {
            0 | 1 => Ok(Self::Softmax),
            2 => Ok(Self::Sigmoid),
            3 => Ok(Self::SoftmaxWeight),
            4 => Ok(Self::SqrtSoftplus),
            other => Err(ModelError::InvalidConfig(format!(
                "unsupported expert_gating_func = {other}"
            ))),
        }
    }
}

/// Hyperparameters for a single layer's `SparseMoeFfn`.
#[derive(Debug, Clone)]
pub struct SparseMoeConfig {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub gating_func: ExpertGatingFunc,
    pub norm_weights: bool,
    pub norm_weights_eps: f32,
    pub weights_scale: f32,
}

/// Model-wide MoE metadata parsed from GGUF header.
#[derive(Debug, Clone)]
pub struct MoeArchParams {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_layer_dense_lead: usize,
    pub n_ff_exp_per_layer: Vec<usize>,
    pub gating_func: ExpertGatingFunc,
    pub norm_weights: bool,
    pub norm_weights_eps: f32,
    pub weights_scale: f32,
}

impl MoeArchParams {
    #[inline]
    pub fn max_n_ff_exp(&self) -> usize {
        self.n_ff_exp_per_layer.iter().copied().max().unwrap_or(0)
    }
}

/// Zero-copy 3D expert weight slices (`[cols, rows, n_expert]` in GGML layout).
pub enum PackedExperts<'a> {
    /// Standard separate `ffn_gate_exps.weight` (optional for non-gated SiLU) + `ffn_up_exps.weight`
    /// + `ffn_down_exps.weight`.
    Separate {
        gate_exps: Option<&'a [u8]>,
        gate_dtype: DType,
        gate_exp_bytes: usize,
        up_exps: &'a [u8],
        up_dtype: DType,
        up_exp_bytes: usize,
        down_exps: &'a [u8],
        down_dtype: DType,
        down_exp_bytes: usize,
    },
    /// Fused `ffn_gate_up_exps.weight` (`[n_embd, 2 * n_ff_exp, n_expert]`, gate rows `[0..n_ff_exp]`
    /// followed by up rows `[n_ff_exp..2*n_ff_exp]`) + `ffn_down_exps.weight`.
    FusedGateUp {
        gate_up_exps: &'a [u8],
        gate_up_dtype: DType,
        gate_up_exp_bytes: usize,
        half_exp_bytes: usize,
        down_exps: &'a [u8],
        down_dtype: DType,
        down_exp_bytes: usize,
    },
}

impl<'a> PackedExperts<'a> {
    /// Returns `((gate_dtype, gate_bytes_opt), (up_dtype, up_bytes), (down_dtype, down_bytes))` for expert `exp_id`.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn expert_slices(
        &self,
        exp_id: usize,
    ) -> (
        Option<(DType, &'a [u8])>,
        (DType, &'a [u8]),
        (DType, &'a [u8]),
    ) {
        match *self {
            PackedExperts::Separate {
                gate_exps,
                gate_dtype,
                gate_exp_bytes,
                up_exps,
                up_dtype,
                up_exp_bytes,
                down_exps,
                down_dtype,
                down_exp_bytes,
            } => {
                let g = gate_exps.map(|gb| {
                    let s = exp_id * gate_exp_bytes;
                    (gate_dtype, &gb[s..s + gate_exp_bytes])
                });
                let u_s = exp_id * up_exp_bytes;
                let d_s = exp_id * down_exp_bytes;
                (
                    g,
                    (up_dtype, &up_exps[u_s..u_s + up_exp_bytes]),
                    (down_dtype, &down_exps[d_s..d_s + down_exp_bytes]),
                )
            }
            PackedExperts::FusedGateUp {
                gate_up_exps,
                gate_up_dtype,
                gate_up_exp_bytes,
                half_exp_bytes,
                down_exps,
                down_dtype,
                down_exp_bytes,
            } => {
                let gu_s = exp_id * gate_up_exp_bytes;
                let d_s = exp_id * down_exp_bytes;
                (
                    Some((gate_up_dtype, &gate_up_exps[gu_s..gu_s + half_exp_bytes])),
                    (
                        gate_up_dtype,
                        &gate_up_exps[gu_s + half_exp_bytes..gu_s + gate_up_exp_bytes],
                    ),
                    (down_dtype, &down_exps[d_s..d_s + down_exp_bytes]),
                )
            }
        }
    }

    pub fn active_bytes_per_token(&self, n_expert_used: usize) -> u64 {
        let one = match *self {
            PackedExperts::Separate {
                gate_exp_bytes,
                up_exp_bytes,
                down_exp_bytes,
                ..
            } => gate_exp_bytes + up_exp_bytes + down_exp_bytes,
            PackedExperts::FusedGateUp {
                gate_up_exp_bytes,
                down_exp_bytes,
                ..
            } => gate_up_exp_bytes + down_exp_bytes,
        };
        (one * n_expert_used) as u64
    }
}

/// Optional shared expert branch (`granitemoe`, `granitehybrid`, `qwen2moe`, `qwen3moe`, `bailingmoe2`).
pub struct SharedExpertWeights<'a> {
    pub n_ff_shexp: usize,
    pub w_gate: RawMatView<'a>,
    pub w_up: RawMatView<'a>,
    pub w_down: RawMatView<'a>,
    /// Optional scalar gate `blk.{l}.ffn_gate_inp_shexp.weight` (`[n_embd]` -> `sigmoid(dot)` in `qwen2moe`).
    pub w_gate_inp_shexp: Option<RawMatView<'a>>,
}

/// Complete per-layer Sparse Mixture-of-Experts FFN weights.
pub struct SparseMoeFfn<'a> {
    pub cfg: SparseMoeConfig,
    pub gate_inp: RawMatView<'a>,
    pub gate_inp_bias: Option<Vec<f32>>,
    pub exp_probs_b: Option<Vec<f32>>,
    pub experts: PackedExperts<'a>,
    pub shared_expert: Option<SharedExpertWeights<'a>>,
}

impl<'a> SparseMoeFfn<'a> {
    pub fn bind_lora(&mut self, bound: &BoundLora) {
        self.gate_inp.lora_slot = bound.slot_for(self.gate_inp.name);
        if let Some(ref mut sh) = self.shared_expert {
            sh.w_gate.lora_slot = bound.slot_for(sh.w_gate.name);
            sh.w_up.lora_slot = bound.slot_for(sh.w_up.name);
            sh.w_down.lora_slot = bound.slot_for(sh.w_down.name);
            if let Some(ref mut gi) = sh.w_gate_inp_shexp {
                gi.lora_slot = bound.slot_for(gi.name);
            }
        }
    }
}

/// Pre-allocated working buffers for `SparseMoeFfn` (`0` allocations on token & batch paths).
pub struct MoeScratch {
    pub router_logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub sel_probs: Vec<f32>,
    pub top_ids: Vec<usize>,
    pub top_weights: Vec<f32>,
    pub exp_gate: Vec<f32>,
    pub exp_up: Vec<f32>,
    pub exp_out: Vec<f32>,
    pub shexp_gate_scalar: Vec<f32>,
}

impl MoeScratch {
    pub fn empty() -> Self {
        Self {
            router_logits: Vec::new(),
            probs: Vec::new(),
            sel_probs: Vec::new(),
            top_ids: Vec::new(),
            top_weights: Vec::new(),
            exp_gate: Vec::new(),
            exp_up: Vec::new(),
            exp_out: Vec::new(),
            shexp_gate_scalar: Vec::new(),
        }
    }

    pub fn new(
        n_embd: usize,
        n_expert: usize,
        n_expert_used: usize,
        max_n_ff_exp: usize,
        max_n_ff_shexp: usize,
    ) -> Self {
        if n_expert == 0 {
            return Self::empty();
        }
        let max_ff = max_n_ff_exp.max(max_n_ff_shexp).max(1);
        Self {
            router_logits: vec![0.0; MAX_CPU_BATCH * n_expert],
            probs: vec![0.0; n_expert],
            sel_probs: vec![0.0; n_expert],
            top_ids: vec![0; MAX_CPU_BATCH * n_expert_used],
            top_weights: vec![0.0; MAX_CPU_BATCH * n_expert_used],
            exp_gate: vec![0.0; MAX_CPU_BATCH * max_ff],
            exp_up: vec![0.0; MAX_CPU_BATCH * max_ff],
            exp_out: vec![0.0; MAX_CPU_BATCH * n_embd],
            shexp_gate_scalar: vec![0.0; MAX_CPU_BATCH],
        }
    }
}

/// Parses optional MoE hyperparameters from GGUF metadata. Returns `Ok(None)` when `expert_count == 0`.
pub fn parse_moe_arch_params(
    gguf: &GgufFile,
    pfx: &str,
    arch: &str,
    n_layers: usize,
    n_ff_default: usize,
) -> Result<Option<MoeArchParams>, ModelError> {
    let n_expert = gguf
        .get_meta(&format!("{pfx}.expert_count"))
        .and_then(|v| v.as_usize())
        .unwrap_or(0);
    if n_expert == 0 {
        return Ok(None);
    }
    let n_expert_used = gguf.req_usize(&format!("{pfx}.expert_used_count"))?;
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(ModelError::InvalidConfig(format!(
            "invalid expert_used_count={n_expert_used} for expert_count={n_expert}"
        )));
    }
    let n_layer_dense_lead = gguf
        .get_meta(&format!("{pfx}.leading_dense_block_count"))
        .and_then(|v| v.as_usize())
        .unwrap_or(0);

    // Per `qwen2moe.cpp:45` & `qwen3moe.cpp:50`, Qwen MoEs fallback to `n_ff / n_expert_used` when
    // `expert_feed_forward_length` is omitted; `olmoe` / `granite*` store `n_ff_exp` in `feed_forward_length`.
    let fallback_ff_exp = if matches!(arch, "qwen2moe" | "qwen3moe" | "qwen35moe" | "qwen4exp")
        && n_ff_default >= n_expert_used
    {
        n_ff_default / n_expert_used
    } else {
        n_ff_default
    };
    let mut n_ff_exp_per_layer = vec![fallback_ff_exp; n_layers];
    if let Some(mv) = gguf.get_meta(&format!("{pfx}.expert_feed_forward_length")) {
        if let Some(arr) = mv.as_usize_array()
            && arr.len() >= n_layers
        {
            for (l, &v) in arr[..n_layers].iter().enumerate() {
                if v > 0 {
                    n_ff_exp_per_layer[l] = v;
                }
            }
        } else if let Some(sc) = mv.as_usize()
            && sc > 0
        {
            n_ff_exp_per_layer.fill(sc);
        }
    }

    let gating_raw = gguf
        .get_meta(&format!("{pfx}.expert_gating_func"))
        .and_then(|v| v.as_usize())
        .unwrap_or(1) as u32;
    let gating_func = ExpertGatingFunc::from_gguf_u32(gating_raw)?;

    let default_norm_w = matches!(
        arch,
        "granite"
            | "granitemoe"
            | "granitehybrid"
            | "granitemoehybrid"
            | "qwen3moe"
            | "qwen35moe"
            | "qwen4exp"
    );
    let norm_weights = gguf
        .get_meta(&format!("{pfx}.expert_weights_norm"))
        .and_then(|v| v.as_bool())
        .unwrap_or(default_norm_w);
    let norm_weights_eps = gguf
        .get_meta(&format!("{pfx}.expert_weights_norm_eps"))
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let weights_scale = gguf
        .get_meta(&format!("{pfx}.expert_weights_scale"))
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);

    Ok(Some(MoeArchParams {
        n_expert,
        n_expert_used,
        n_layer_dense_lead,
        n_ff_exp_per_layer,
        gating_func,
        norm_weights,
        norm_weights_eps,
        weights_scale,
    }))
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

fn load_mat_2d<'a>(
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

fn load_mat_1row<'a>(
    gguf: &'a GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
) -> Result<RawMatView<'a>, ModelError> {
    let (info, bytes) = gguf.tensor_bytes(name, layer)?;
    if info.dims.as_slice() != [cols] && info.dims.as_slice() != [cols, 1] {
        return Err(GgufError::TensorShapeMismatch {
            path: gguf.path().display().to_string(),
            tensor: name.to_string(),
            layer: layer.to_string(),
            expected: vec![cols],
            actual: info.dims.clone(),
        }
        .into());
    }
    Ok(RawMatView {
        name: info.name.as_str(),
        bytes,
        dtype: info.dtype,
        rows: 1,
        cols,
        row_offset: 0,
        is_sliced: false,
        lora_slot: None,
    })
}

/// Loads `SparseMoeFfn` for layer `l` if `blk.{l}.ffn_gate_inp.weight` is present and `l >= n_layer_dense_lead`.
pub fn load_layer_moe_ffn<'a>(
    gguf: &'a GgufFile,
    l: usize,
    ltag: &str,
    n_embd: usize,
    params: &MoeArchParams,
) -> Result<Option<SparseMoeFfn<'a>>, ModelError> {
    if l < params.n_layer_dense_lead {
        return Ok(None);
    }
    let gate_inp_name = format!("{ltag}.ffn_gate_inp.weight");
    if gguf.tensor_info(&gate_inp_name).is_some() {
        // Proceed to load MoE layer
    } else {
        return Ok(None);
    }

    let n_expert = params.n_expert;
    let n_expert_used = params.n_expert_used;
    let gate_inp = load_mat_2d(gguf, &gate_inp_name, ltag, n_embd, n_expert)?;
    let gate_inp_bias =
        load_opt_f32_vec(gguf, &format!("{ltag}.ffn_gate_inp.bias"), ltag, n_expert)?;
    let exp_probs_b = load_opt_f32_vec(gguf, &format!("{ltag}.exp_probs_b.bias"), ltag, n_expert)?;

    // Determine `n_ff_exp` from metadata or directly from `ffn_down_exps.weight` `dims[0]` when metadata was 0.
    let down_exps_name = format!("{ltag}.ffn_down_exps.weight");
    let down_info_probe =
        gguf.tensor_info(&down_exps_name)
            .ok_or_else(|| GgufError::MissingTensor {
                path: gguf.path().display().to_string(),
                tensor: down_exps_name.clone(),
                layer: ltag.to_string(),
            })?;
    let n_ff_exp = if params.n_ff_exp_per_layer[l] > 0 {
        params.n_ff_exp_per_layer[l]
    } else {
        down_info_probe.dims.first().copied().unwrap_or(0)
    };
    if n_ff_exp == 0 {
        return Err(ModelError::InvalidConfig(format!(
            "{ltag}: expert_feed_forward_length must be > 0"
        )));
    }

    let (down_info, down_bytes) =
        gguf.req_tensor(&down_exps_name, ltag, &[n_ff_exp, n_embd, n_expert])?;
    let down_exp_bytes = n_embd * down_info.dtype.row_bytes(n_ff_exp);

    let fused_gu_name = format!("{ltag}.ffn_gate_up_exps.weight");
    let experts = if gguf.tensor_info(&fused_gu_name).is_some() {
        let (gu_info, gu_bytes) =
            gguf.req_tensor(&fused_gu_name, ltag, &[n_embd, 2 * n_ff_exp, n_expert])?;
        if gu_info.dtype == DType::Q4_0x4 && !n_ff_exp.is_multiple_of(4) {
            return Err(ModelError::InvalidConfig(format!(
                "{ltag}: Q4_0x4 fused ffn_gate_up_exps requires n_ff_exp ({n_ff_exp}) % 4 == 0"
            )));
        }
        let half_exp_bytes = n_ff_exp * gu_info.dtype.row_bytes(n_embd);
        PackedExperts::FusedGateUp {
            gate_up_exps: gu_bytes,
            gate_up_dtype: gu_info.dtype,
            gate_up_exp_bytes: 2 * half_exp_bytes,
            half_exp_bytes,
            down_exps: down_bytes,
            down_dtype: down_info.dtype,
            down_exp_bytes,
        }
    } else {
        let up_name = format!("{ltag}.ffn_up_exps.weight");
        let (up_info, up_bytes) = gguf.req_tensor(&up_name, ltag, &[n_embd, n_ff_exp, n_expert])?;
        let up_exp_bytes = n_ff_exp * up_info.dtype.row_bytes(n_embd);

        let gate_name = format!("{ltag}.ffn_gate_exps.weight");
        let (gate_exps, gate_dtype, gate_exp_bytes) = if gguf.tensor_info(&gate_name).is_some() {
            let (g_info, g_bytes) =
                gguf.req_tensor(&gate_name, ltag, &[n_embd, n_ff_exp, n_expert])?;
            (
                Some(g_bytes),
                g_info.dtype,
                n_ff_exp * g_info.dtype.row_bytes(n_embd),
            )
        } else {
            (None, up_info.dtype, 0)
        };

        PackedExperts::Separate {
            gate_exps,
            gate_dtype,
            gate_exp_bytes,
            up_exps: up_bytes,
            up_dtype: up_info.dtype,
            up_exp_bytes,
            down_exps: down_bytes,
            down_dtype: down_info.dtype,
            down_exp_bytes,
        }
    };

    // Optional shared expert (`ffn_{gate,up,down}_shexp.weight` + optional `ffn_gate_inp_shexp.weight`)
    let shexp_gate_name = format!("{ltag}.ffn_gate_shexp.weight");
    let (shared_expert, n_ff_shexp) = if let Some(sh_info) = gguf.tensor_info(&shexp_gate_name) {
        let ff_sh = sh_info.dims.get(1).copied().unwrap_or(0);
        if ff_sh == 0 {
            return Err(ModelError::InvalidConfig(format!(
                "{ltag}: invalid ffn_gate_shexp shape {:?}",
                sh_info.dims
            )));
        }
        let w_gate = load_mat_2d(gguf, &shexp_gate_name, ltag, n_embd, ff_sh)?;
        let w_up = load_mat_2d(
            gguf,
            &format!("{ltag}.ffn_up_shexp.weight"),
            ltag,
            n_embd,
            ff_sh,
        )?;
        let w_down = load_mat_2d(
            gguf,
            &format!("{ltag}.ffn_down_shexp.weight"),
            ltag,
            ff_sh,
            n_embd,
        )?;
        let gi_name = format!("{ltag}.ffn_gate_inp_shexp.weight");
        let w_gate_inp_shexp = if gguf.tensor_info(&gi_name).is_some() {
            Some(load_mat_1row(gguf, &gi_name, ltag, n_embd)?)
        } else {
            None
        };
        (
            Some(SharedExpertWeights {
                n_ff_shexp: ff_sh,
                w_gate,
                w_up,
                w_down,
                w_gate_inp_shexp,
            }),
            ff_sh,
        )
    } else {
        (None, 0)
    };

    Ok(Some(SparseMoeFfn {
        cfg: SparseMoeConfig {
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_ff_shexp,
            gating_func: params.gating_func,
            norm_weights: params.norm_weights,
            norm_weights_eps: params.norm_weights_eps,
            weights_scale: params.weights_scale,
        },
        gate_inp,
        gate_inp_bias,
        exp_probs_b,
        experts,
        shared_expert,
    }))
}

/// Computes exact `llama.cpp` `build_moe_ffn` routing (`top_ids[..k]` and `top_weights[..k]`)
/// for a single token's router `logits[..n_expert]`.
pub fn compute_moe_routing_row(
    cfg: &SparseMoeConfig,
    logits: &[f32],
    exp_probs_b: Option<&[f32]>,
    probs_buf: &mut [f32],
    sel_buf: &mut [f32],
    out_ids: &mut [usize],
    out_weights: &mut [f32],
) {
    let n_exp = cfg.n_expert;
    let k = cfg.n_expert_used;
    debug_assert_eq!(logits.len(), n_exp);

    match cfg.gating_func {
        ExpertGatingFunc::Softmax => {
            let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (p, &l) in probs_buf[..n_exp].iter_mut().zip(logits.iter()) {
                let e = (l - max_l).exp();
                *p = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for p in &mut probs_buf[..n_exp] {
                *p *= inv;
            }
        }
        ExpertGatingFunc::Sigmoid => {
            for (p, &l) in probs_buf[..n_exp].iter_mut().zip(logits.iter()) {
                *p = 1.0 / (1.0 + (-l).exp());
            }
        }
        ExpertGatingFunc::SoftmaxWeight => {
            probs_buf[..n_exp].copy_from_slice(logits);
        }
        ExpertGatingFunc::SqrtSoftplus => {
            for (p, &l) in probs_buf[..n_exp].iter_mut().zip(logits.iter()) {
                *p = softplus_f32(l).sqrt();
            }
        }
    }

    if let Some(b) = exp_probs_b {
        for ((s, &p), &bi) in sel_buf[..n_exp]
            .iter_mut()
            .zip(probs_buf[..n_exp].iter())
            .zip(b.iter())
        {
            *s = p + bi;
        }
    } else {
        sel_buf[..n_exp].copy_from_slice(&probs_buf[..n_exp]);
    }

    // Partial selection sort for `k <= n_exp` (`k` is typically 2..8): deterministic tie-break `e_a < e_b`.
    for slot in 0..k {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (e, &sv) in sel_buf[..n_exp].iter().enumerate() {
            let mut already = false;
            for &prev in &out_ids[..slot] {
                if prev == e {
                    already = true;
                    break;
                }
            }
            if !already && sv > best_val {
                best_val = sv;
                best_idx = e;
            }
        }
        out_ids[slot] = best_idx;
        out_weights[slot] = probs_buf[best_idx];
    }

    if cfg.gating_func == ExpertGatingFunc::SoftmaxWeight {
        let max_w = out_weights[..k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for w in &mut out_weights[..k] {
            *w = (*w - max_w).exp();
            sum += *w;
        }
        let inv = 1.0 / sum;
        for w in &mut out_weights[..k] {
            *w *= inv;
        }
    }

    if cfg.norm_weights {
        let sum_w: f32 = out_weights[..k].iter().sum();
        let denom = if cfg.norm_weights_eps > 0.0 {
            sum_w + cfg.norm_weights_eps
        } else {
            sum_w.max(MOE_NORM_CLAMP_F16_MIN)
        };
        let inv = 1.0 / denom;
        for w in &mut out_weights[..k] {
            *w *= inv;
        }
    }

    if cfg.weights_scale != 0.0 && (cfg.weights_scale - 1.0).abs() > 1e-12 {
        for w in &mut out_weights[..k] {
            *w *= cfg.weights_scale;
        }
    }
}

/// Evaluates `SparseMoeFfn` over `n_tok` (`1 <= n_tok <= MAX_CPU_BATCH`) normalized input rows
/// `x_norm [n_tok, n_embd]`, writing `[n_tok, n_embd]` into `out`.
///
/// Uses identical per-token expert `matvec` reduction order for `n_tok == 1` and `n_tok > 1` so
/// `forward_batch` and `forward_token` are **0-ULP (`to_bits()`)** identical (`AGENTS.md` §9).
#[allow(clippy::too_many_arguments)]
pub fn eval_sparse_moe_ffn(
    moe: &SparseMoeFfn<'_>,
    lora: &BoundLora,
    pool: &ThreadPool,
    kt: &KernelTable,
    x_norm: &[f32],
    out: &mut [f32],
    n_embd: usize,
    n_tok: usize,
    mv_scratch: &mut MatvecScratch,
    lora_scratch: &mut LoraScratch,
    scratch: &mut MoeScratch,
) {
    let n_exp = moe.cfg.n_expert;
    let k = moe.cfg.n_expert_used;
    let n_ff_exp = moe.cfg.n_ff_exp;

    // 1. Router projection (`[n_tok, n_embd] -> [n_tok, n_expert]`)
    if n_tok == 1 {
        matvec_f32(
            pool,
            kt,
            moe.gate_inp.dtype,
            moe.gate_inp.bytes,
            &x_norm[..n_embd],
            mv_scratch,
            &mut scratch.router_logits[..n_exp],
            n_exp,
            n_embd,
        );
    } else {
        matmul_f32(
            pool,
            kt,
            moe.gate_inp.dtype,
            moe.gate_inp.bytes,
            &x_norm[..n_tok * n_embd],
            mv_scratch,
            &mut scratch.router_logits[..n_tok * n_exp],
            n_exp,
            n_embd,
            n_tok,
        );
    }
    lora.apply_linear(
        moe.gate_inp.lora_slot,
        pool,
        kt,
        &x_norm[..n_tok * n_embd],
        &mut scratch.router_logits[..n_tok * n_exp],
        n_tok,
        lora_scratch,
    );
    if let Some(ref gb) = moe.gate_inp_bias {
        for t in 0..n_tok {
            let row = &mut scratch.router_logits[t * n_exp..(t + 1) * n_exp];
            for (y, &b) in row.iter_mut().zip(gb.iter()) {
                *y += b;
            }
        }
    }

    // 2. Top-K routing + deterministic `0..k` expert evaluation per token
    out[..n_tok * n_embd].fill(0.0);
    for t in 0..n_tok {
        let l_row = &scratch.router_logits[t * n_exp..(t + 1) * n_exp];
        let ids_t = &mut scratch.top_ids[t * k..(t + 1) * k];
        let w_t = &mut scratch.top_weights[t * k..(t + 1) * k];
        compute_moe_routing_row(
            &moe.cfg,
            l_row,
            moe.exp_probs_b.as_deref(),
            &mut scratch.probs,
            &mut scratch.sel_probs,
            ids_t,
            w_t,
        );

        let xt = &x_norm[t * n_embd..(t + 1) * n_embd];
        let out_t = &mut out[t * n_embd..(t + 1) * n_embd];
        for slot in 0..k {
            let exp_id = ids_t[slot];
            let weight = w_t[slot];
            let (gate_opt, (up_dt, up_bytes), (down_dt, down_bytes)) =
                moe.experts.expert_slices(exp_id);
            if let Some((gate_dt, gate_bytes)) = gate_opt {
                matvec_pair_f32(
                    pool,
                    kt,
                    gate_dt,
                    gate_bytes,
                    n_ff_exp,
                    &mut scratch.exp_gate[..n_ff_exp],
                    up_dt,
                    up_bytes,
                    n_ff_exp,
                    &mut scratch.exp_up[..n_ff_exp],
                    xt,
                    mv_scratch,
                    n_embd,
                );
                swiglu_inplace_parallel(
                    pool,
                    &mut scratch.exp_gate[..n_ff_exp],
                    &scratch.exp_up[..n_ff_exp],
                );
            } else {
                matvec_f32(
                    pool,
                    kt,
                    up_dt,
                    up_bytes,
                    xt,
                    mv_scratch,
                    &mut scratch.exp_gate[..n_ff_exp],
                    n_ff_exp,
                    n_embd,
                );
                for v in &mut scratch.exp_gate[..n_ff_exp] {
                    let x = *v;
                    *v = x / (1.0 + (-x).exp());
                }
            }
            matvec_f32(
                pool,
                kt,
                down_dt,
                down_bytes,
                &scratch.exp_gate[..n_ff_exp],
                mv_scratch,
                &mut scratch.exp_out[..n_embd],
                n_embd,
                n_ff_exp,
            );
            (kt.axpy_f32)(out_t, weight, &scratch.exp_out[..n_embd]);
        }
    }

    // 3. Optional shared expert (`granitemoe`, `granitehybrid`, `qwen2moe`, `qwen3moe`, `bailingmoe2`)
    if let Some(ref sh) = moe.shared_expert {
        let ff_sh = sh.n_ff_shexp;
        if n_tok == 1 {
            matvec_pair_f32(
                pool,
                kt,
                sh.w_gate.dtype,
                sh.w_gate.bytes,
                ff_sh,
                &mut scratch.exp_gate[..ff_sh],
                sh.w_up.dtype,
                sh.w_up.bytes,
                ff_sh,
                &mut scratch.exp_up[..ff_sh],
                &x_norm[..n_embd],
                mv_scratch,
                n_embd,
            );
        } else {
            tm_cpu::matmul_pair_f32(
                pool,
                kt,
                sh.w_gate.dtype,
                sh.w_gate.bytes,
                ff_sh,
                &mut scratch.exp_gate[..n_tok * ff_sh],
                sh.w_up.dtype,
                sh.w_up.bytes,
                ff_sh,
                &mut scratch.exp_up[..n_tok * ff_sh],
                &x_norm[..n_tok * n_embd],
                mv_scratch,
                n_embd,
                n_tok,
            );
        }
        if !lora.is_empty() {
            lora.apply_linear(
                sh.w_gate.lora_slot,
                pool,
                kt,
                &x_norm[..n_tok * n_embd],
                &mut scratch.exp_gate[..n_tok * ff_sh],
                n_tok,
                lora_scratch,
            );
            lora.apply_linear(
                sh.w_up.lora_slot,
                pool,
                kt,
                &x_norm[..n_tok * n_embd],
                &mut scratch.exp_up[..n_tok * ff_sh],
                n_tok,
                lora_scratch,
            );
        }
        swiglu_inplace_parallel(
            pool,
            &mut scratch.exp_gate[..n_tok * ff_sh],
            &scratch.exp_up[..n_tok * ff_sh],
        );

        if n_tok == 1 {
            matvec_f32(
                pool,
                kt,
                sh.w_down.dtype,
                sh.w_down.bytes,
                &scratch.exp_gate[..ff_sh],
                mv_scratch,
                &mut scratch.exp_out[..n_embd],
                n_embd,
                ff_sh,
            );
        } else {
            matmul_f32(
                pool,
                kt,
                sh.w_down.dtype,
                sh.w_down.bytes,
                &scratch.exp_gate[..n_tok * ff_sh],
                mv_scratch,
                &mut scratch.exp_out[..n_tok * n_embd],
                n_embd,
                ff_sh,
                n_tok,
            );
        }
        lora.apply_linear(
            sh.w_down.lora_slot,
            pool,
            kt,
            &scratch.exp_gate[..n_tok * ff_sh],
            &mut scratch.exp_out[..n_tok * n_embd],
            n_tok,
            lora_scratch,
        );

        if let Some(ref gi) = sh.w_gate_inp_shexp {
            if n_tok == 1 {
                matvec_f32(
                    pool,
                    kt,
                    gi.dtype,
                    gi.bytes,
                    &x_norm[..n_embd],
                    mv_scratch,
                    &mut scratch.shexp_gate_scalar[..1],
                    1,
                    n_embd,
                );
            } else {
                matmul_f32(
                    pool,
                    kt,
                    gi.dtype,
                    gi.bytes,
                    &x_norm[..n_tok * n_embd],
                    mv_scratch,
                    &mut scratch.shexp_gate_scalar[..n_tok],
                    1,
                    n_embd,
                    n_tok,
                );
            }
            lora.apply_linear(
                gi.lora_slot,
                pool,
                kt,
                &x_norm[..n_tok * n_embd],
                &mut scratch.shexp_gate_scalar[..n_tok],
                n_tok,
                lora_scratch,
            );
            for t in 0..n_tok {
                let g = 1.0 / (1.0 + (-scratch.shexp_gate_scalar[t]).exp());
                (kt.axpy_f32)(
                    &mut out[t * n_embd..(t + 1) * n_embd],
                    g,
                    &scratch.exp_out[t * n_embd..(t + 1) * n_embd],
                );
            }
        } else {
            (kt.axpy_f32)(
                &mut out[..n_tok * n_embd],
                1.0,
                &scratch.exp_out[..n_tok * n_embd],
            );
        }
    }
}
