//! Shared Apple Metal Sparse MoE GPU engine (`GpuMoeEngine` + `GpuSparseMoe`, `ADR-018`).
//!
//! Executes fine-grained Top-K routing + indexed 3D expert projections + optional shared expert
//! **100% inside a single `MTLCommandBuffer`** with zero CPU-GPU roundtrips and deterministic
//! `0..n_expert_used` slot reduction.
//!
//! Every compute pipeline validates `maxTotalThreadsPerThreadgroup` (`AGENTS.md` §9), and every
//! `unsafe` Metal FFI call carries an explicit `// SAFETY:` comment.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::MetalError;
use crate::lora_gpu::MetalLoraEngine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertGatingFunc {
    Softmax,
    Sigmoid,
    SoftmaxWeight,
    SqrtSoftplus,
}

impl ExpertGatingFunc {
    pub fn from_gguf_u32(val: u32) -> Result<Self, MetalError> {
        match val {
            0 | 1 => Ok(Self::Softmax),
            2 => Ok(Self::Sigmoid),
            3 => Ok(Self::SoftmaxWeight),
            4 => Ok(Self::SqrtSoftplus),
            other => Err(MetalError::InvalidDimensions(format!(
                "unsupported expert_gating_func = {other}"
            ))),
        }
    }
}

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

pub fn parse_moe_arch_params(
    gguf: &GgufFile,
    pfx: &str,
    arch: &str,
    n_layers: usize,
    n_ff_default: usize,
) -> Result<Option<MoeArchParams>, MetalError> {
    let n_expert = gguf
        .get_meta(&format!("{pfx}.expert_count"))
        .and_then(|v| v.as_usize())
        .unwrap_or(0);
    if n_expert == 0 {
        return Ok(None);
    }
    let n_expert_used = gguf.req_usize(&format!("{pfx}.expert_used_count"))?;
    if n_expert_used == 0 || n_expert_used > n_expert {
        return Err(MetalError::InvalidDimensions(format!(
            "invalid expert_used_count={n_expert_used} for expert_count={n_expert}"
        )));
    }
    let n_layer_dense_lead = gguf
        .get_meta(&format!("{pfx}.leading_dense_block_count"))
        .and_then(|v| v.as_usize())
        .unwrap_or(0);

    let fallback_ff_exp =
        if matches!(arch, "qwen2moe" | "qwen3moe") && n_ff_default >= n_expert_used {
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
        "granite" | "granitemoe" | "granitehybrid" | "granitemoehybrid" | "qwen3moe"
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

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[repr(C)]
struct GemvArgs {
    n_in: u32,
    n_out: u32,
    add_residual: u32,
    row_bytes: u32,
}

#[repr(C)]
struct GemmArgs {
    n_in: u32,
    n_out: u32,
    n_tok: u32,
    add_residual: u32,
    row_bytes: u32,
}

#[repr(C)]
struct AddBiasArgs {
    cols: u32,
    n_tok: u32,
}

#[repr(C)]
struct ScaleArgs {
    n: u32,
    val: f32,
}

#[repr(C)]
struct MoeRouterArgs {
    n_expert: u32,
    n_expert_used: u32,
    n_tok: u32,
    gating_func: u32,
    norm_weights: u32,
    has_exp_probs_b: u32,
    norm_eps: f32,
    weights_scale: f32,
}

#[repr(C)]
struct MoeIndexedGemvArgs {
    n_in: u32,
    n_out: u32,
    row_bytes: u32,
    exp_stride_bytes: u32,
    intra_exp_offset_bytes: u32,
    k: u32,
    x_tok_stride: u32,
    x_slot_stride: u32,
    scale_by_router: u32,
}

#[repr(C)]
struct MoeReduceArgs {
    n_embd: u32,
    k: u32,
    n_tok: u32,
}

#[repr(C)]
struct MoeShexpGateArgs {
    n_embd: u32,
    n_tok: u32,
}

#[derive(Clone)]
pub struct MoeGpuMat {
    pub name: String,
    pub buf: MtlBuf,
    pub offset: usize,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
}

#[derive(Clone)]
pub enum GpuPackedExperts {
    Separate {
        gate: Option<(MtlBuf, usize, DType, usize, usize)>, // (buf, offset, dtype, row_bytes, exp_bytes)
        up: (MtlBuf, usize, DType, usize, usize),
        down: (MtlBuf, usize, DType, usize, usize),
    },
    FusedGateUp {
        gate_up: (MtlBuf, usize, DType, usize, usize, usize), // (buf, offset, dtype, row_bytes, exp_bytes, half_exp_bytes)
        down: (MtlBuf, usize, DType, usize, usize),
    },
}

pub struct GpuSharedExpert {
    pub n_ff_shexp: usize,
    pub w_gate: MoeGpuMat,
    pub w_up: MoeGpuMat,
    pub w_down: MoeGpuMat,
    pub w_gate_inp_shexp: Option<MoeGpuMat>,
}

pub struct GpuSparseMoe {
    pub cfg: SparseMoeConfig,
    pub gate_inp: MoeGpuMat,
    pub gate_inp_bias: Option<MtlBuf>,
    pub exp_probs_b: Option<MtlBuf>,
    pub experts: GpuPackedExperts,
    pub shared_expert: Option<GpuSharedExpert>,
}

impl GpuSparseMoe {
    pub fn active_bytes_per_token(&self) -> u64 {
        let k = self.cfg.n_expert_used as u64;
        let one_exp = match &self.experts {
            GpuPackedExperts::Separate { gate, up, down } => {
                gate.as_ref().map_or(0, |g| g.4) + up.4 + down.4
            }
            GpuPackedExperts::FusedGateUp { gate_up, down } => gate_up.4 + down.4,
        } as u64;
        let sh = self.shared_expert.as_ref().map_or(0u64, |s| {
            (s.w_gate.rows * s.w_gate.row_bytes
                + s.w_up.rows * s.w_up.row_bytes
                + s.w_down.rows * s.w_down.row_bytes) as u64
        });
        (self.gate_inp.rows * self.gate_inp.row_bytes) as u64 + k * one_exp + sh
    }
}

/// Reusable GPU pipelines and pre-allocated buffers for `SparseMoeFfn`.
pub struct GpuMoeEngine {
    router_topk: MtlPipe,
    silu_mul: MtlPipe,
    silu_inplace: MtlPipe,
    add_bias: MtlPipe,
    scale_axpy: MtlPipe,
    reduce_slots: MtlPipe,
    shexp_gated_add: MtlPipe,
    idx_q4_0: MtlPipe,
    idx_q4_1: MtlPipe,
    idx_q8_0: MtlPipe,
    idx_q5_k: MtlPipe,
    idx_q6_k: MtlPipe,
    idx_bf16: MtlPipe,
    idx_f32: MtlPipe,
    gemv_q4_0: MtlPipe,
    gemv_q4_1: MtlPipe,
    gemv_q8_0: MtlPipe,
    gemv_q5_k: MtlPipe,
    gemv_q6_k: MtlPipe,
    gemv_bf16: MtlPipe,
    gemv_f32: MtlPipe,
    gemm_q4_0: MtlPipe,
    gemm_q4_1: MtlPipe,
    gemm_q8_0: MtlPipe,
    gemm_q5_k: MtlPipe,
    gemm_q6_k: MtlPipe,
    gemm_bf16: MtlPipe,
    gemm_f32: MtlPipe,
    dummy_f32: MtlBuf,
    router_logits: MtlBuf,
    router_probs: MtlBuf,
    router_sel: MtlBuf,
    top_ids: MtlBuf,
    top_weights: MtlBuf,
    exp_gate: MtlBuf,
    exp_up: MtlBuf,
    slot_out: MtlBuf,
    shexp_out: MtlBuf,
    shexp_gate_scalar: MtlBuf,
}

fn require_fit(
    pipe: &ProtocolObject<dyn MTLComputePipelineState>,
    name: &'static str,
    nt: usize,
) -> Result<(), MetalError> {
    let limit = pipe.maxTotalThreadsPerThreadgroup();
    if nt > limit {
        return Err(MetalError::ThreadgroupTooLarge {
            kernel: name,
            requested: nt,
            limit,
        });
    }
    Ok(())
}

fn build_pipe(
    device: &ProtocolObject<dyn MTLDevice>,
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
    max_tg: usize,
) -> Result<MtlPipe, MetalError> {
    let ns = NSString::from_str(name);
    let func = lib
        .newFunctionWithName(&ns)
        .ok_or_else(|| MetalError::FunctionNotFound(name.to_string()))?;
    let pipe = device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| MetalError::PipelineCreation {
            name: name.to_string(),
            reason: e.localizedDescription().to_string(),
        })?;
    require_fit(&pipe, name, max_tg)?;
    Ok(pipe)
}

fn alloc_buf(
    device: &ProtocolObject<dyn MTLDevice>,
    byte_len: usize,
) -> Result<MtlBuf, MetalError> {
    let len = byte_len.max(16);
    device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAlloc { bytes: len })
}

fn upload_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<MtlBuf, MetalError> {
    let len = bytes.len().max(4);
    let ptr = NonNull::new(bytes.as_ptr() as *mut c_void)
        .ok_or(MetalError::BufferAlloc { bytes: len })?;
    // SAFETY: `bytes` is valid for `bytes.len()` readable bytes; Metal copies into shared storage.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAlloc { bytes: len })
}

fn upload_f32(device: &ProtocolObject<dyn MTLDevice>, vals: &[f32]) -> Result<MtlBuf, MetalError> {
    let byte_len = core::mem::size_of_val(vals).max(4);
    let ptr = NonNull::new(vals.as_ptr() as *mut c_void)
        .ok_or(MetalError::BufferAlloc { bytes: byte_len })?;
    // SAFETY: `vals` is valid for `vals.len() * 4` readable bytes.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            core::mem::size_of_val(vals),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAlloc { bytes: byte_len })
}

fn ensure_supported_dtype(dt: DType) -> Result<(), MetalError> {
    match dt {
        DType::Q4_0
        | DType::Q4_1
        | DType::Q8_0
        | DType::Q5K
        | DType::Q6K
        | DType::BF16
        | DType::F32 => Ok(()),
        other => Err(MetalError::UnsupportedDType(other)),
    }
}

fn load_opt_f32_vec(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<Option<MtlBuf>, MetalError> {
    let Some(info) = gguf.tensor_info(name) else {
        return Ok(None);
    };
    let (_, bytes) = gguf.req_tensor(name, layer, &[len])?;
    let mut vals = vec![0.0f32; info.dims[0]];
    dequantize_row_bytes(info.dtype, bytes, &mut vals);
    Ok(Some(upload_f32(device, &vals)?))
}

fn load_mat_2d(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    shared_weights: Option<&MtlBuf>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<MoeGpuMat, MetalError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    ensure_supported_dtype(info.dtype)?;
    let row_bytes = info.dtype.row_bytes(cols);
    let (buf, offset) = match shared_weights {
        Some(sb) => (sb.clone(), gguf.data_offset() + info.offset),
        None => (upload_bytes(device, bytes)?, 0),
    };
    Ok(MoeGpuMat {
        name: name.to_string(),
        buf,
        offset,
        dtype: info.dtype,
        rows,
        cols,
        row_bytes,
    })
}

fn load_mat_1row(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    shared_weights: Option<&MtlBuf>,
    name: &str,
    layer: &str,
    cols: usize,
) -> Result<MoeGpuMat, MetalError> {
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
    ensure_supported_dtype(info.dtype)?;
    let row_bytes = info.dtype.row_bytes(cols);
    let (buf, offset) = match shared_weights {
        Some(sb) => (sb.clone(), gguf.data_offset() + info.offset),
        None => (upload_bytes(device, bytes)?, 0),
    };
    Ok(MoeGpuMat {
        name: name.to_string(),
        buf,
        offset,
        dtype: info.dtype,
        rows: 1,
        cols,
        row_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_3d_exps(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    shared_weights: Option<&MtlBuf>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
    n_expert: usize,
) -> Result<(MtlBuf, usize, DType, usize, usize), MetalError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows, n_expert])?;
    ensure_supported_dtype(info.dtype)?;
    let row_bytes = info.dtype.row_bytes(cols);
    let exp_bytes = rows * row_bytes;
    let (buf, offset) = match shared_weights {
        Some(sb) => (sb.clone(), gguf.data_offset() + info.offset),
        None => (upload_bytes(device, bytes)?, 0),
    };
    Ok((buf, offset, info.dtype, row_bytes, exp_bytes))
}

/// Loads `GpuSparseMoe` for layer `l` if `blk.{l}.ffn_gate_inp.weight` exists and `l >= n_layer_dense_lead`.
pub fn load_gpu_layer_moe(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    shared_weights: Option<&MtlBuf>,
    l: usize,
    ltag: &str,
    n_embd: usize,
    params: &MoeArchParams,
) -> Result<Option<GpuSparseMoe>, MetalError> {
    if l < params.n_layer_dense_lead {
        return Ok(None);
    }
    let gate_inp_name = format!("{ltag}.ffn_gate_inp.weight");
    if gguf.tensor_info(&gate_inp_name).is_none() {
        return Ok(None);
    }

    let n_expert = params.n_expert;
    let n_expert_used = params.n_expert_used;
    let gate_inp = load_mat_2d(
        device,
        gguf,
        shared_weights,
        &gate_inp_name,
        ltag,
        n_embd,
        n_expert,
    )?;
    let gate_inp_bias = load_opt_f32_vec(
        device,
        gguf,
        &format!("{ltag}.ffn_gate_inp.bias"),
        ltag,
        n_expert,
    )?;
    let exp_probs_b = load_opt_f32_vec(
        device,
        gguf,
        &format!("{ltag}.exp_probs_b.bias"),
        ltag,
        n_expert,
    )?;

    let down_name = format!("{ltag}.ffn_down_exps.weight");
    let down_probe = gguf
        .tensor_info(&down_name)
        .ok_or_else(|| GgufError::MissingTensor {
            path: gguf.path().display().to_string(),
            tensor: down_name.clone(),
            layer: ltag.to_string(),
        })?;
    let n_ff_exp = if params.n_ff_exp_per_layer[l] > 0 {
        params.n_ff_exp_per_layer[l]
    } else {
        down_probe.dims.first().copied().unwrap_or(0)
    };
    if n_ff_exp == 0 {
        return Err(MetalError::InvalidDimensions(format!(
            "{ltag}: expert_feed_forward_length must be > 0"
        )));
    }

    let down = load_3d_exps(
        device,
        gguf,
        shared_weights,
        &down_name,
        ltag,
        n_ff_exp,
        n_embd,
        n_expert,
    )?;

    let fused_name = format!("{ltag}.ffn_gate_up_exps.weight");
    let experts = if gguf.tensor_info(&fused_name).is_some() {
        let (buf, offset, dt, rb, exp_bytes) = load_3d_exps(
            device,
            gguf,
            shared_weights,
            &fused_name,
            ltag,
            n_embd,
            2 * n_ff_exp,
            n_expert,
        )?;
        GpuPackedExperts::FusedGateUp {
            gate_up: (buf, offset, dt, rb, exp_bytes, n_ff_exp * rb),
            down,
        }
    } else {
        let up = load_3d_exps(
            device,
            gguf,
            shared_weights,
            &format!("{ltag}.ffn_up_exps.weight"),
            ltag,
            n_embd,
            n_ff_exp,
            n_expert,
        )?;
        let gate_name = format!("{ltag}.ffn_gate_exps.weight");
        let gate = if gguf.tensor_info(&gate_name).is_some() {
            Some(load_3d_exps(
                device,
                gguf,
                shared_weights,
                &gate_name,
                ltag,
                n_embd,
                n_ff_exp,
                n_expert,
            )?)
        } else {
            None
        };
        GpuPackedExperts::Separate { gate, up, down }
    };

    let shexp_gate_name = format!("{ltag}.ffn_gate_shexp.weight");
    let (shared_expert, n_ff_shexp) = if let Some(sh_info) = gguf.tensor_info(&shexp_gate_name) {
        let ff_sh = sh_info.dims.get(1).copied().unwrap_or(0);
        if ff_sh == 0 {
            return Err(MetalError::InvalidDimensions(format!(
                "{ltag}: invalid ffn_gate_shexp dims {:?}",
                sh_info.dims
            )));
        }
        let w_gate = load_mat_2d(
            device,
            gguf,
            shared_weights,
            &shexp_gate_name,
            ltag,
            n_embd,
            ff_sh,
        )?;
        let w_up = load_mat_2d(
            device,
            gguf,
            shared_weights,
            &format!("{ltag}.ffn_up_shexp.weight"),
            ltag,
            n_embd,
            ff_sh,
        )?;
        let w_down = load_mat_2d(
            device,
            gguf,
            shared_weights,
            &format!("{ltag}.ffn_down_shexp.weight"),
            ltag,
            ff_sh,
            n_embd,
        )?;
        let gi_name = format!("{ltag}.ffn_gate_inp_shexp.weight");
        let w_gate_inp_shexp = if gguf.tensor_info(&gi_name).is_some() {
            Some(load_mat_1row(
                device,
                gguf,
                shared_weights,
                &gi_name,
                ltag,
                n_embd,
            )?)
        } else {
            None
        };
        (
            Some(GpuSharedExpert {
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

    Ok(Some(GpuSparseMoe {
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

impl GpuMoeEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        lib: &ProtocolObject<dyn MTLLibrary>,
        max_batch: usize,
        n_embd: usize,
        n_expert: usize,
        n_expert_used: usize,
        max_n_ff_exp: usize,
        max_n_ff_shexp: usize,
    ) -> Result<Option<Self>, MetalError> {
        if n_expert == 0 {
            return Ok(None);
        }
        let max_tk = max_batch * n_expert_used;
        let max_ff_bytes = (max_tk * max_n_ff_exp).max(max_batch * max_n_ff_shexp) * 4;
        Ok(Some(Self {
            router_topk: build_pipe(device, lib, "moe_router_topk_f32", 64)?,
            silu_mul: build_pipe(device, lib, "silu_mul_f32", 256)?,
            silu_inplace: build_pipe(device, lib, "silu_inplace_f32", 256)?,
            add_bias: build_pipe(device, lib, "add_bias_f32", 256)?,
            scale_axpy: build_pipe(device, lib, "scale_axpy_f32", 256)?,
            reduce_slots: build_pipe(device, lib, "moe_reduce_slots_f32", 256)?,
            shexp_gated_add: build_pipe(device, lib, "moe_shexp_gated_add_f32", 256)?,
            idx_q4_0: build_pipe(device, lib, "moe_indexed_gemv_q4_0", 128)?,
            idx_q4_1: build_pipe(device, lib, "moe_indexed_gemv_q4_1", 128)?,
            idx_q8_0: build_pipe(device, lib, "moe_indexed_gemv_q8_0", 128)?,
            idx_q5_k: build_pipe(device, lib, "moe_indexed_gemv_q5_k", 128)?,
            idx_q6_k: build_pipe(device, lib, "moe_indexed_gemv_q6_k", 128)?,
            idx_bf16: build_pipe(device, lib, "moe_indexed_gemv_bf16", 128)?,
            idx_f32: build_pipe(device, lib, "moe_indexed_gemv_f32", 128)?,
            gemv_q4_0: build_pipe(device, lib, "gemv_q4_0_res", 128)?,
            gemv_q4_1: build_pipe(device, lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(device, lib, "gemv_q8_0_res", 128)?,
            gemv_q5_k: build_pipe(device, lib, "gemv_q5_k_res", 128)?,
            gemv_q6_k: build_pipe(device, lib, "gemv_q6_k_res", 128)?,
            gemv_bf16: build_pipe(device, lib, "gemv_bf16_res", 128)?,
            gemv_f32: build_pipe(device, lib, "gemv_f32_res", 128)?,
            gemm_q4_0: build_pipe(device, lib, "gemm_q4_0_res", 128)?,
            gemm_q4_1: build_pipe(device, lib, "gemm_q4_1_res", 128)?,
            gemm_q8_0: build_pipe(device, lib, "gemm_q8_0_res", 128)?,
            gemm_q5_k: build_pipe(device, lib, "gemm_q5_k_res", 128)?,
            gemm_q6_k: build_pipe(device, lib, "gemm_q6_k_res", 128)?,
            gemm_bf16: build_pipe(device, lib, "gemm_bf16_res", 128)?,
            gemm_f32: build_pipe(device, lib, "gemm_f32_res", 128)?,
            dummy_f32: upload_f32(device, &[0.0f32])?,
            router_logits: alloc_buf(device, max_batch * n_expert * 4)?,
            router_probs: alloc_buf(device, max_batch * n_expert * 4)?,
            router_sel: alloc_buf(device, max_batch * n_expert * 4)?,
            top_ids: alloc_buf(device, max_tk * 4)?,
            top_weights: alloc_buf(device, max_tk * 4)?,
            exp_gate: alloc_buf(device, max_ff_bytes)?,
            exp_up: alloc_buf(device, max_ff_bytes)?,
            slot_out: alloc_buf(device, max_tk * n_embd * 4)?,
            shexp_out: alloc_buf(device, max_batch * n_embd * 4)?,
            shexp_gate_scalar: alloc_buf(device, max_batch * 4)?,
        }))
    }

    fn encode_mat(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        lora: &MetalLoraEngine,
        mat: &MoeGpuMat,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        n_tok: usize,
    ) -> Result<(), MetalError> {
        const THREADS: usize = 128;
        if n_tok == 1 {
            let (pipe, name, rows_per_tg): (&MtlPipe, &'static str, usize) = match mat.dtype {
                DType::Q4_0 => (&self.gemv_q4_0, "gemv_q4_0_res", 32),
                DType::Q4_1 => (&self.gemv_q4_1, "gemv_q4_1_res", 32),
                DType::Q8_0 => (&self.gemv_q8_0, "gemv_q8_0_res", 16),
                DType::Q5K => (&self.gemv_q5_k, "gemv_q5_k_res", 8),
                DType::Q6K => (&self.gemv_q6_k, "gemv_q6_k_res", 16),
                DType::BF16 => (&self.gemv_bf16, "gemv_bf16_res", 16),
                DType::F32 => (&self.gemv_f32, "gemv_f32_res", 8),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            require_fit(pipe, name, THREADS)?;
            let args = GemvArgs {
                n_in: mat.cols as u32,
                n_out: mat.rows as u32,
                add_residual: 0,
                row_bytes: mat.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers remain valid until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(y_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<GemvArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: mat.rows.div_ceil(rows_per_tg),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            let (pipe, name, rows_per_tg, toks_per_tg): (&MtlPipe, &'static str, usize, usize) =
                match mat.dtype {
                    DType::Q4_0 => (&self.gemm_q4_0, "gemm_q4_0_res", 32, 32),
                    DType::Q4_1 => (&self.gemm_q4_1, "gemm_q4_1_res", 32, 32),
                    DType::Q8_0 => (&self.gemm_q8_0, "gemm_q8_0_res", 4, 8),
                    DType::Q5K => (&self.gemm_q5_k, "gemm_q5_k_res", 4, 8),
                    DType::Q6K => (&self.gemm_q6_k, "gemm_q6_k_res", 4, 8),
                    DType::BF16 => (&self.gemm_bf16, "gemm_bf16_res", 32, 32),
                    DType::F32 => (&self.gemm_f32, "gemm_f32_res", 4, 8),
                    other => return Err(MetalError::UnsupportedDType(other)),
                };
            require_fit(pipe, name, THREADS)?;
            let args = GemmArgs {
                n_in: mat.cols as u32,
                n_out: mat.rows as u32,
                n_tok: n_tok as u32,
                add_residual: 0,
                row_bytes: mat.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers remain valid until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(y_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<GemmArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: mat.rows.div_ceil(rows_per_tg),
                    height: n_tok.div_ceil(toks_per_tg),
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        }
        lora.encode_linear(enc, &mat.name, x_buf, 0, y_buf, 0, n_tok);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_indexed_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        exps_buf: &ProtocolObject<dyn MTLBuffer>,
        exps_offset: usize,
        dtype: DType,
        n_in: usize,
        n_out: usize,
        row_bytes: usize,
        exp_stride_bytes: usize,
        intra_exp_offset_bytes: usize,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        x_tok_stride: usize,
        x_slot_stride: usize,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        k: usize,
        n_tok: usize,
        scale_by_router: bool,
    ) -> Result<(), MetalError> {
        const THREADS: usize = 128;
        let (pipe, name, rows_per_tg): (&MtlPipe, &'static str, usize) = match dtype {
            DType::Q4_0 => (&self.idx_q4_0, "moe_indexed_gemv_q4_0", 32),
            DType::Q4_1 => (&self.idx_q4_1, "moe_indexed_gemv_q4_1", 32),
            DType::Q8_0 => (&self.idx_q8_0, "moe_indexed_gemv_q8_0", 16),
            DType::Q5K => (&self.idx_q5_k, "moe_indexed_gemv_q5_k", 8),
            DType::Q6K => (&self.idx_q6_k, "moe_indexed_gemv_q6_k", 16),
            DType::BF16 => (&self.idx_bf16, "moe_indexed_gemv_bf16", 16),
            DType::F32 => (&self.idx_f32, "moe_indexed_gemv_f32", 8),
            other => return Err(MetalError::UnsupportedDType(other)),
        };
        require_fit(pipe, name, THREADS)?;
        let args = MoeIndexedGemvArgs {
            n_in: n_in as u32,
            n_out: n_out as u32,
            row_bytes: row_bytes as u32,
            exp_stride_bytes: exp_stride_bytes as u32,
            intra_exp_offset_bytes: intra_exp_offset_bytes as u32,
            k: k as u32,
            x_tok_stride: x_tok_stride as u32,
            x_slot_stride: x_slot_stride as u32,
            scale_by_router: u32::from(scale_by_router),
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers remain valid until command buffer completion.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(exps_buf), exps_offset, 0);
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.top_ids), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.top_weights), 0, 3);
            enc.setBuffer_offset_atIndex(Some(y_buf), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<MoeIndexedGemvArgs>(),
                5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_out.div_ceil(rows_per_tg),
                height: n_tok * k,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Encodes a complete `GpuSparseMoe` layer (`router -> indexed experts -> deterministic reduce -> shared expert`)
    /// writing `[n_tok, n_embd]` into `out_buf`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_ffn(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        lora: &MetalLoraEngine,
        moe: &GpuSparseMoe,
        x_norm_buf: &ProtocolObject<dyn MTLBuffer>,
        out_buf: &ProtocolObject<dyn MTLBuffer>,
        n_embd: usize,
        n_tok: usize,
    ) -> Result<(), MetalError> {
        let n_exp = moe.cfg.n_expert;
        let k = moe.cfg.n_expert_used;
        let ff_exp = moe.cfg.n_ff_exp;

        // 1. Router projection + optional bias + Top-K selection
        self.encode_mat(
            enc,
            lora,
            &moe.gate_inp,
            x_norm_buf,
            &self.router_logits,
            n_tok,
        )?;
        if let Some(ref gb) = moe.gate_inp_bias {
            const TG: usize = 256;
            require_fit(&self.add_bias, "add_bias_f32", TG)?;
            let bargs = AddBiasArgs {
                cols: n_exp as u32,
                n_tok: n_tok as u32,
            };
            enc.setComputePipelineState(&self.add_bias);
            // SAFETY: Buffers remain valid until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.router_logits), 0, 0);
                enc.setBuffer_offset_atIndex(Some(gb), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::from(&bargs).cast::<c_void>(),
                    core::mem::size_of::<AddBiasArgs>(),
                    2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n_tok * n_exp).div_ceil(TG),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        {
            const TG: usize = 64;
            require_fit(&self.router_topk, "moe_router_topk_f32", TG)?;
            let gating_func = match moe.cfg.gating_func {
                ExpertGatingFunc::Softmax => 0,
                ExpertGatingFunc::Sigmoid => 1,
                ExpertGatingFunc::SoftmaxWeight => 2,
                ExpertGatingFunc::SqrtSoftplus => 3,
            };
            let rargs = MoeRouterArgs {
                n_expert: n_exp as u32,
                n_expert_used: k as u32,
                n_tok: n_tok as u32,
                gating_func,
                norm_weights: u32::from(moe.cfg.norm_weights),
                has_exp_probs_b: u32::from(moe.exp_probs_b.is_some()),
                norm_eps: moe.cfg.norm_weights_eps,
                weights_scale: moe.cfg.weights_scale,
            };
            let pb = moe.exp_probs_b.as_ref().unwrap_or(&self.dummy_f32);
            enc.setComputePipelineState(&self.router_topk);
            // SAFETY: Buffers remain valid until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.router_logits), 0, 0);
                enc.setBuffer_offset_atIndex(Some(pb), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.router_probs), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.router_sel), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.top_ids), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&self.top_weights), 0, 5);
                enc.setBytes_length_atIndex(
                    NonNull::from(&rargs).cast::<c_void>(),
                    core::mem::size_of::<MoeRouterArgs>(),
                    6,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_tok.div_ceil(TG),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // 2. Indexed 3D expert evaluation across all `n_tok * k` active pairs
        let tk_ff = n_tok * k * ff_exp;
        match &moe.experts {
            GpuPackedExperts::Separate { gate, up, down } => {
                if let Some((g_buf, g_off, g_dt, g_rb, g_stride)) = gate {
                    self.encode_indexed_gemv(
                        enc,
                        g_buf,
                        *g_off,
                        *g_dt,
                        n_embd,
                        ff_exp,
                        *g_rb,
                        *g_stride,
                        0,
                        x_norm_buf,
                        n_embd,
                        0,
                        &self.exp_gate,
                        k,
                        n_tok,
                        false,
                    )?;
                    self.encode_indexed_gemv(
                        enc,
                        &up.0,
                        up.1,
                        up.2,
                        n_embd,
                        ff_exp,
                        up.3,
                        up.4,
                        0,
                        x_norm_buf,
                        n_embd,
                        0,
                        &self.exp_up,
                        k,
                        n_tok,
                        false,
                    )?;
                    const TG: usize = 256;
                    require_fit(&self.silu_mul, "silu_mul_f32", TG)?;
                    let n_u32 = tk_ff as u32;
                    enc.setComputePipelineState(&self.silu_mul);
                    // SAFETY: Buffers remain valid until command buffer completion.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.exp_gate), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.exp_up), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&n_u32).cast::<c_void>(),
                            core::mem::size_of::<u32>(),
                            2,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: tk_ff.div_ceil(TG),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                } else {
                    self.encode_indexed_gemv(
                        enc,
                        &up.0,
                        up.1,
                        up.2,
                        n_embd,
                        ff_exp,
                        up.3,
                        up.4,
                        0,
                        x_norm_buf,
                        n_embd,
                        0,
                        &self.exp_gate,
                        k,
                        n_tok,
                        false,
                    )?;
                    const TG: usize = 256;
                    require_fit(&self.silu_inplace, "silu_inplace_f32", TG)?;
                    let n_u32 = tk_ff as u32;
                    enc.setComputePipelineState(&self.silu_inplace);
                    // SAFETY: Buffers remain valid until command buffer completion.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.exp_gate), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&n_u32).cast::<c_void>(),
                            core::mem::size_of::<u32>(),
                            1,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: tk_ff.div_ceil(TG),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: TG,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                self.encode_indexed_gemv(
                    enc,
                    &down.0,
                    down.1,
                    down.2,
                    ff_exp,
                    n_embd,
                    down.3,
                    down.4,
                    0,
                    &self.exp_gate,
                    k * ff_exp,
                    ff_exp,
                    &self.slot_out,
                    k,
                    n_tok,
                    true,
                )?;
            }
            GpuPackedExperts::FusedGateUp { gate_up, down } => {
                self.encode_indexed_gemv(
                    enc,
                    &gate_up.0,
                    gate_up.1,
                    gate_up.2,
                    n_embd,
                    ff_exp,
                    gate_up.3,
                    gate_up.4,
                    0,
                    x_norm_buf,
                    n_embd,
                    0,
                    &self.exp_gate,
                    k,
                    n_tok,
                    false,
                )?;
                self.encode_indexed_gemv(
                    enc,
                    &gate_up.0,
                    gate_up.1,
                    gate_up.2,
                    n_embd,
                    ff_exp,
                    gate_up.3,
                    gate_up.4,
                    gate_up.5,
                    x_norm_buf,
                    n_embd,
                    0,
                    &self.exp_up,
                    k,
                    n_tok,
                    false,
                )?;
                const TG: usize = 256;
                require_fit(&self.silu_mul, "silu_mul_f32", TG)?;
                let n_u32 = tk_ff as u32;
                enc.setComputePipelineState(&self.silu_mul);
                // SAFETY: Buffers remain valid until command buffer completion.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.exp_gate), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.exp_up), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&n_u32).cast::<c_void>(),
                        core::mem::size_of::<u32>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: tk_ff.div_ceil(TG),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: TG,
                        height: 1,
                        depth: 1,
                    },
                );
                self.encode_indexed_gemv(
                    enc,
                    &down.0,
                    down.1,
                    down.2,
                    ff_exp,
                    n_embd,
                    down.3,
                    down.4,
                    0,
                    &self.exp_gate,
                    k * ff_exp,
                    ff_exp,
                    &self.slot_out,
                    k,
                    n_tok,
                    true,
                )?;
            }
        }

        // 3. Deterministic `0..k` reduction -> `out_buf [n_tok, n_embd]`
        {
            const TG: usize = 256;
            require_fit(&self.reduce_slots, "moe_reduce_slots_f32", TG)?;
            let rargs = MoeReduceArgs {
                n_embd: n_embd as u32,
                k: k as u32,
                n_tok: n_tok as u32,
            };
            enc.setComputePipelineState(&self.reduce_slots);
            // SAFETY: Buffers remain valid until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.slot_out), 0, 0);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::from(&rargs).cast::<c_void>(),
                    core::mem::size_of::<MoeReduceArgs>(),
                    2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (n_tok * n_embd).div_ceil(TG),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // 4. Optional shared expert
        if let Some(ref sh) = moe.shared_expert {
            let ff_sh = sh.n_ff_shexp;
            self.encode_mat(enc, lora, &sh.w_gate, x_norm_buf, &self.exp_gate, n_tok)?;
            self.encode_mat(enc, lora, &sh.w_up, x_norm_buf, &self.exp_up, n_tok)?;
            {
                const TG: usize = 256;
                require_fit(&self.silu_mul, "silu_mul_f32", TG)?;
                let n_u32 = (n_tok * ff_sh) as u32;
                enc.setComputePipelineState(&self.silu_mul);
                // SAFETY: Buffers remain valid until command buffer completion.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.exp_gate), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.exp_up), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&n_u32).cast::<c_void>(),
                        core::mem::size_of::<u32>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (n_tok * ff_sh).div_ceil(TG),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            self.encode_mat(
                enc,
                lora,
                &sh.w_down,
                &self.exp_gate,
                &self.shexp_out,
                n_tok,
            )?;
            if let Some(ref gi) = sh.w_gate_inp_shexp {
                self.encode_mat(enc, lora, gi, x_norm_buf, &self.shexp_gate_scalar, n_tok)?;
                const TG: usize = 256;
                require_fit(&self.shexp_gated_add, "moe_shexp_gated_add_f32", TG)?;
                let gargs = MoeShexpGateArgs {
                    n_embd: n_embd as u32,
                    n_tok: n_tok as u32,
                };
                enc.setComputePipelineState(&self.shexp_gated_add);
                // SAFETY: Buffers remain valid until command buffer completion.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.shexp_out), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.shexp_gate_scalar), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&gargs).cast::<c_void>(),
                        core::mem::size_of::<MoeShexpGateArgs>(),
                        3,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (n_tok * n_embd).div_ceil(TG),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: TG,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                const TG: usize = 256;
                require_fit(&self.scale_axpy, "scale_axpy_f32", TG)?;
                let sargs = ScaleArgs {
                    n: (n_tok * n_embd) as u32,
                    val: 1.0,
                };
                enc.setComputePipelineState(&self.scale_axpy);
                // SAFETY: Buffers remain valid until command buffer completion.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.shexp_out), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&sargs).cast::<c_void>(),
                        core::mem::size_of::<ScaleArgs>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (n_tok * n_embd).div_ceil(TG),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        Ok(())
    }
}
