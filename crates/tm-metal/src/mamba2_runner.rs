//! End-to-end Apple Metal GPU runner for `mamba2`, `nemotron_h`, `falcon-h1`, and `granitehybrid` (`ADR-017`).
//!
//! All `unsafe` Metal calls include `// SAFETY:` comments, and every pipeline/dispatch checks
//! `maxTotalThreadsPerThreadgroup` (`AGENTS.md` §9).

use core::ffi::c_void;
use core::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::MetalError;
use crate::lora_gpu::MetalLoraEngine;

const LLAMA_OPS_MSL: &str = include_str!("shaders/llama_ops.metal");
const MAX_BATCH: usize = 64;
const METAL_PAGE_SIZE: usize = 16384;

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvArgs {
    n_in: u32,
    n_out: u32,
    add_residual: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmArgs {
    n_in: u32,
    n_out: u32,
    n_tok: u32,
    add_residual: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormArgs {
    n: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedArgs {
    token_id: u32,
    n_embd: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BiasArgs {
    d_out: u32,
    n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleVecArgs {
    n: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeKvArgs {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    rope_dim: u32,
    pos: u32,
    rope_type: u32,
    has_freq_factors: u32,
    freq_base: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnExtArgs {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    start_pos: u32,
    n_tok: u32,
    window: u32,
    attn_scale: f32,
    attn_softcap: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Split3RowsArgs {
    d0: u32,
    d1: u32,
    d2: u32,
    n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mamba2ConvArgs {
    d_inner: u32,
    bc_dim: u32,
    d_conv: u32,
    n_tok: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mamba2ScanArgs {
    n_heads: u32,
    n_groups: u32,
    head_dim: u32,
    d_state: u32,
    n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Mamba2GateNormArgs {
    d_inner: u32,
    n_groups: u32,
    has_norm: u32,
    eps: f32,
}

#[derive(Debug, Clone)]
pub struct Mamba2MetalConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub rms_eps: f32,
    pub ssm_conv_kernel: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_n_head: usize,
    pub ssm_n_group: usize,
    pub ssm_norm_eps: f32,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rope_type: u32,
    pub use_rope: bool,
    pub embedding_scale: f32,
    pub attn_scale: f32,
    pub residual_scale: f32,
    pub inv_logit_scale: f32,
}

struct GpuMat {
    name: String,
    buf: MtlBuf,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

struct GpuSsm {
    ssm_in: GpuMat,
    conv_w: MtlBuf,
    conv_b: Option<MtlBuf>,
    dt_bias: MtlBuf,
    ssm_a: MtlBuf,
    ssm_d: MtlBuf,
    ssm_norm: Option<MtlBuf>,
    ssm_out: GpuMat,
    ssm_state: MtlBuf,
    ssm_state_bytes: usize,
    conv_hist: MtlBuf,
    conv_hist_bytes: usize,
}

struct GpuAttn {
    n_head: usize,
    n_head_kv: usize,
    wq: GpuMat,
    wk: GpuMat,
    wv: GpuMat,
    wo: GpuMat,
    bq: Option<MtlBuf>,
    bk: Option<MtlBuf>,
    bv: Option<MtlBuf>,
    bo: Option<MtlBuf>,
    k_cache: MtlBuf,
    v_cache: MtlBuf,
}

#[allow(clippy::large_enum_variant)]
enum GpuFfn {
    SwiGlu {
        n_ff: usize,
        w_gate: GpuMat,
        w_up: GpuMat,
        w_down: GpuMat,
        b_gate: Option<MtlBuf>,
        b_up: Option<MtlBuf>,
        b_down: Option<MtlBuf>,
    },
    Relu2 {
        n_ff: usize,
        w_up: GpuMat,
        w_down: GpuMat,
        b_up: Option<MtlBuf>,
        b_down: Option<MtlBuf>,
    },
    SparseMoe(crate::moe_gpu::GpuSparseMoe),
}

#[allow(clippy::large_enum_variant)]
enum GpuMamba2Block {
    SsmOnly {
        ssm: GpuSsm,
    },
    AttnOnly {
        attn: GpuAttn,
    },
    FfnOnly {
        ffn: GpuFfn,
    },
    SsmAndFfn {
        ssm: GpuSsm,
        ffn_norm: MtlBuf,
        ffn: GpuFfn,
    },
    AttnAndFfn {
        attn: GpuAttn,
        ffn_norm: MtlBuf,
        ffn: GpuFfn,
    },
    ParallelAttnSsmAndFfn {
        attn: GpuAttn,
        ssm: GpuSsm,
        ffn_norm: MtlBuf,
        ffn: GpuFfn,
    },
}

struct GpuMamba2Layer {
    attn_norm: MtlBuf,
    block: GpuMamba2Block,
}

struct Pipelines {
    embed_q4_0: MtlPipe,
    embed_q8_0: MtlPipe,
    embed_q6_k: MtlPipe,
    embed_f32: MtlPipe,
    embed_q4_0_batch: MtlPipe,
    embed_q8_0_batch: MtlPipe,
    embed_q6_k_batch: MtlPipe,
    embed_f32_batch: MtlPipe,
    rms_norm: MtlPipe,
    rms_norm_batch: MtlPipe,
    add_bias: MtlPipe,
    copy_f32: MtlPipe,
    scale_inplace: MtlPipe,
    scale_axpy: MtlPipe,
    silu_mul: MtlPipe,
    relu2_inplace: MtlPipe,
    rope_kv: MtlPipe,
    rope_kv_batch: MtlPipe,
    gqa_attn_ext_batch: MtlPipe,
    split3_rows: MtlPipe,
    mamba2_conv: MtlPipe,
    mamba2_scan: MtlPipe,
    mamba2_gate_norm: MtlPipe,
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
    argmax: MtlPipe,
}

pub struct Mamba2MetalRunner {
    pub cfg: Mamba2MetalConfig,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Pipelines,
    token_embd: GpuMat,
    output_tied: bool,
    output_separate: Option<GpuMat>,
    output_bias: Option<MtlBuf>,
    output_norm: MtlBuf,
    rope_freqs: MtlBuf,
    has_rope_freqs: bool,
    dummy_f32: MtlBuf,
    layers: Vec<GpuMamba2Layer>,
    token_ids_buf: MtlBuf,
    argmax_buf: MtlBuf,
    x: MtlBuf,
    xb: MtlBuf,
    xb_par: MtlBuf,
    xb2: MtlBuf,
    in_proj: MtlBuf,
    ssm_z: MtlBuf,
    ssm_xbc: MtlBuf,
    ssm_dt: MtlBuf,
    ssm_x: MtlBuf,
    ssm_b: MtlBuf,
    ssm_c: MtlBuf,
    ssm_y: MtlBuf,
    q: MtlBuf,
    k: MtlBuf,
    v: MtlBuf,
    hb: MtlBuf,
    hb2: MtlBuf,
    logits: MtlBuf,
    moe_engine: Option<crate::moe_gpu::GpuMoeEngine>,
    lora: MetalLoraEngine,
    zero_copy_weights: bool,
    _gguf: Arc<GgufFile>,
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

fn upload_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<MtlBuf, MetalError> {
    let len = bytes.len().max(4);
    let ptr = NonNull::new(bytes.as_ptr() as *mut c_void)
        .ok_or(MetalError::BufferAlloc { bytes: len })?;
    // SAFETY: `bytes` is readable for `bytes.len()` bytes; Metal copies into shared storage.
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
    let src = if vals.is_empty() { &[0.0f32][..] } else { vals };
    let ptr = NonNull::new(src.as_ptr() as *mut c_void)
        .ok_or(MetalError::BufferAlloc { bytes: byte_len })?;
    // SAFETY: `src` is valid for `size_of_val(src)` readable bytes.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            core::mem::size_of_val(src),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalError::BufferAlloc { bytes: byte_len })
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

fn try_zero_copy_gguf_buf(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
) -> Option<MtlBuf> {
    let raw = gguf.as_bytes();
    if raw.is_empty() || !(raw.as_ptr() as usize).is_multiple_of(METAL_PAGE_SIZE) {
        return None;
    }
    let aligned_len = raw.len().div_ceil(METAL_PAGE_SIZE) * METAL_PAGE_SIZE;
    let non_null = NonNull::new(raw.as_ptr() as *mut c_void)?;
    // SAFETY: `raw.as_ptr()` is page-aligned and kept alive by `_gguf: Arc<GgufFile>` (last field).
    unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            non_null,
            aligned_len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

fn load_gpu_f32_vec(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<MtlBuf, MetalError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[len])?;
    let mut vals = vec![0.0f32; len];
    dequantize_row_bytes(info.dtype, bytes, &mut vals);
    upload_f32(device, &vals)
}

fn load_gpu_f32_vec_1d_or_2d(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    d0: usize,
    d1: usize,
) -> Result<MtlBuf, MetalError> {
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
    let mut vals = vec![0.0f32; total];
    dequantize_row_bytes(info.dtype, bytes, &mut vals);
    upload_f32(device, &vals)
}

fn load_opt_gpu_f32_vec(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<Option<MtlBuf>, MetalError> {
    if gguf.tensor_info(name).is_some() {
        Ok(Some(load_gpu_f32_vec(device, gguf, name, layer, len)?))
    } else {
        Ok(None)
    }
}

fn load_gpu_f32_vec_any(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    names: &[&str],
    layer: &str,
    len: usize,
) -> Result<MtlBuf, MetalError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return load_gpu_f32_vec(device, gguf, name, layer, len);
        }
    }
    load_gpu_f32_vec(device, gguf, names[0], layer, len)
}

fn load_opt_gpu_f32_vec_any(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    names: &[&str],
    layer: &str,
    len: usize,
) -> Result<Option<MtlBuf>, MetalError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return Ok(Some(load_gpu_f32_vec(device, gguf, name, layer, len)?));
        }
    }
    Ok(None)
}

fn load_gpu_mat(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    shared_weights: Option<&MtlBuf>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<GpuMat, MetalError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    match info.dtype {
        DType::Q4_0
        | DType::Q4_1
        | DType::Q8_0
        | DType::Q5K
        | DType::Q6K
        | DType::BF16
        | DType::F32 => {}
        other => return Err(MetalError::UnsupportedDType(other)),
    }
    let (buf, offset) = if let Some(shared) = shared_weights {
        (shared.clone(), gguf.tensor_file_offset(info))
    } else {
        (upload_bytes(device, bytes)?, 0)
    };
    Ok(GpuMat {
        name: name.to_string(),
        buf,
        offset,
        dtype: info.dtype,
        rows,
        cols,
        row_bytes: info.dtype.row_bytes(cols),
    })
}

#[allow(clippy::too_many_arguments)]
fn load_gpu_mat_slice(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    shared_weights: Option<&MtlBuf>,
    fused_name: &str,
    logical_name: String,
    layer: &str,
    cols: usize,
    total_rows: usize,
    row_start: usize,
    rows: usize,
) -> Result<GpuMat, MetalError> {
    let (info, bytes) = gguf.req_tensor(fused_name, layer, &[cols, total_rows])?;
    match info.dtype {
        DType::Q4_0
        | DType::Q4_1
        | DType::Q8_0
        | DType::Q5K
        | DType::Q6K
        | DType::BF16
        | DType::F32 => {}
        other => return Err(MetalError::UnsupportedDType(other)),
    }
    let row_bytes = info.dtype.row_bytes(cols);
    let byte_off = row_start * row_bytes;
    let byte_len = rows * row_bytes;
    let (buf, offset) = if let Some(shared) = shared_weights {
        (shared.clone(), gguf.tensor_file_offset(info) + byte_off)
    } else {
        (
            upload_bytes(device, &bytes[byte_off..byte_off + byte_len])?,
            0,
        )
    };
    Ok(GpuMat {
        name: logical_name,
        buf,
        offset,
        dtype: info.dtype,
        rows,
        cols,
        row_bytes,
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

#[allow(clippy::too_many_arguments)]
fn load_gpu_ssm(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    sw: Option<&MtlBuf>,
    ltag: &str,
    n_embd: usize,
    cfg: &Mamba2MetalConfig,
    norm_required: bool,
) -> Result<GpuSsm, MetalError> {
    let d_inner = cfg.ssm_inner_size;
    let bc_dim = cfg.ssm_n_group * cfg.ssm_state_size;
    let conv_dim = d_inner + 2 * bc_dim;
    let d_in_proj = 2 * d_inner + 2 * bc_dim + cfg.ssm_n_head;
    let d_conv = cfg.ssm_conv_kernel;
    let n_heads = cfg.ssm_n_head;
    let n_group = cfg.ssm_n_group;
    let group_dim = d_inner / n_group;

    let ssm_in = load_gpu_mat(
        device,
        gguf,
        sw,
        &format!("{ltag}.ssm_in.weight"),
        ltag,
        n_embd,
        d_in_proj,
    )?;

    let conv_name = format!("{ltag}.ssm_conv1d.weight");
    let (c_info, c_bytes) = gguf.req_tensor(&conv_name, ltag, &[d_conv, conv_dim])?;
    let mut cw = vec![0.0f32; conv_dim * d_conv];
    let rb = c_info.dtype.row_bytes(d_conv);
    for r in 0..conv_dim {
        dequantize_row_bytes(
            c_info.dtype,
            &c_bytes[r * rb..(r + 1) * rb],
            &mut cw[r * d_conv..(r + 1) * d_conv],
        );
    }
    let conv_w = upload_f32(device, &cw)?;
    let conv_b = load_opt_gpu_f32_vec(
        device,
        gguf,
        &format!("{ltag}.ssm_conv1d.bias"),
        ltag,
        conv_dim,
    )?;
    let dt_bias = load_gpu_f32_vec(device, gguf, &format!("{ltag}.ssm_dt.bias"), ltag, n_heads)?;
    let ssm_a =
        load_gpu_f32_vec_1d_or_2d(device, gguf, &format!("{ltag}.ssm_a"), ltag, 1, n_heads)?;
    let ssm_d =
        load_gpu_f32_vec_1d_or_2d(device, gguf, &format!("{ltag}.ssm_d"), ltag, 1, n_heads)?;

    let norm_name = format!("{ltag}.ssm_norm.weight");
    let ssm_norm = if gguf.tensor_info(&norm_name).is_some() || norm_required {
        Some(load_gpu_f32_vec_1d_or_2d(
            device, gguf, &norm_name, ltag, group_dim, n_group,
        )?)
    } else {
        None
    };
    let ssm_out = load_gpu_mat(
        device,
        gguf,
        sw,
        &format!("{ltag}.ssm_out.weight"),
        ltag,
        d_inner,
        n_embd,
    )?;

    let ssm_state_bytes = d_inner * cfg.ssm_state_size * 4;
    let conv_hist_bytes = conv_dim * d_conv.saturating_sub(1) * 4;
    let ssm_state = alloc_buf(device, ssm_state_bytes)?;
    let conv_hist = alloc_buf(device, conv_hist_bytes)?;
    // Zero-init recurrent buffers explicitly
    // SAFETY: Shared buffers are freshly allocated and idle.
    unsafe {
        if ssm_state_bytes > 0 {
            core::ptr::write_bytes(ssm_state.contents().as_ptr() as *mut u8, 0, ssm_state_bytes);
        }
        if conv_hist_bytes > 0 {
            core::ptr::write_bytes(conv_hist.contents().as_ptr() as *mut u8, 0, conv_hist_bytes);
        }
    }

    Ok(GpuSsm {
        ssm_in,
        conv_w,
        conv_b,
        dt_bias,
        ssm_a,
        ssm_d,
        ssm_norm,
        ssm_out,
        ssm_state,
        ssm_state_bytes,
        conv_hist,
        conv_hist_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_gpu_attn(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    sw: Option<&MtlBuf>,
    ltag: &str,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    max_ctx: usize,
) -> Result<GpuAttn, MetalError> {
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let q_name = format!("{ltag}.attn_q.weight");
    let qkv_name = format!("{ltag}.attn_qkv.weight");
    let (wq, wk, wv) = if gguf.tensor_info(&q_name).is_some() {
        (
            load_gpu_mat(device, gguf, sw, &q_name, ltag, n_embd, q_dim)?,
            load_gpu_mat(
                device,
                gguf,
                sw,
                &format!("{ltag}.attn_k.weight"),
                ltag,
                n_embd,
                kv_dim,
            )?,
            load_gpu_mat(
                device,
                gguf,
                sw,
                &format!("{ltag}.attn_v.weight"),
                ltag,
                n_embd,
                kv_dim,
            )?,
        )
    } else {
        let total = q_dim + 2 * kv_dim;
        (
            load_gpu_mat_slice(
                device, gguf, sw, &qkv_name, q_name, ltag, n_embd, total, 0, q_dim,
            )?,
            load_gpu_mat_slice(
                device,
                gguf,
                sw,
                &qkv_name,
                format!("{ltag}.attn_k.weight"),
                ltag,
                n_embd,
                total,
                q_dim,
                kv_dim,
            )?,
            load_gpu_mat_slice(
                device,
                gguf,
                sw,
                &qkv_name,
                format!("{ltag}.attn_v.weight"),
                ltag,
                n_embd,
                total,
                q_dim + kv_dim,
                kv_dim,
            )?,
        )
    };
    let qkv_bias_name = format!("{ltag}.attn_qkv.bias");
    let (bq, bk, bv) = if gguf.tensor_info(&qkv_bias_name).is_some() {
        let (info, bytes) = gguf.req_tensor(&qkv_bias_name, ltag, &[q_dim + 2 * kv_dim])?;
        let mut full = vec![0.0f32; q_dim + 2 * kv_dim];
        dequantize_row_bytes(info.dtype, bytes, &mut full);
        (
            Some(upload_f32(device, &full[..q_dim])?),
            Some(upload_f32(device, &full[q_dim..q_dim + kv_dim])?),
            Some(upload_f32(
                device,
                &full[q_dim + kv_dim..q_dim + 2 * kv_dim],
            )?),
        )
    } else {
        (
            load_opt_gpu_f32_vec(device, gguf, &format!("{ltag}.attn_q.bias"), ltag, q_dim)?,
            load_opt_gpu_f32_vec(device, gguf, &format!("{ltag}.attn_k.bias"), ltag, kv_dim)?,
            load_opt_gpu_f32_vec(device, gguf, &format!("{ltag}.attn_v.bias"), ltag, kv_dim)?,
        )
    };
    let wo = load_gpu_mat(
        device,
        gguf,
        sw,
        &format!("{ltag}.attn_output.weight"),
        ltag,
        q_dim,
        n_embd,
    )?;
    let bo = load_opt_gpu_f32_vec(
        device,
        gguf,
        &format!("{ltag}.attn_output.bias"),
        ltag,
        n_embd,
    )?;
    let kv_bytes = max_ctx * kv_dim * 2;
    Ok(GpuAttn {
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
        k_cache: alloc_buf(device, kv_bytes)?,
        v_cache: alloc_buf(device, kv_bytes)?,
    })
}

fn load_gpu_swiglu(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    sw: Option<&MtlBuf>,
    ltag: &str,
    n_embd: usize,
    n_ff: usize,
) -> Result<GpuFfn, MetalError> {
    let w_gate = load_gpu_mat(
        device,
        gguf,
        sw,
        &format!("{ltag}.ffn_gate.weight"),
        ltag,
        n_embd,
        n_ff,
    )?;
    let w_up = load_gpu_mat(
        device,
        gguf,
        sw,
        &format!("{ltag}.ffn_up.weight"),
        ltag,
        n_embd,
        n_ff,
    )?;
    let w_down = load_gpu_mat(
        device,
        gguf,
        sw,
        &format!("{ltag}.ffn_down.weight"),
        ltag,
        n_ff,
        n_embd,
    )?;
    let b_gate = load_opt_gpu_f32_vec(device, gguf, &format!("{ltag}.ffn_gate.bias"), ltag, n_ff)?;
    let b_up = load_opt_gpu_f32_vec(device, gguf, &format!("{ltag}.ffn_up.bias"), ltag, n_ff)?;
    let b_down =
        load_opt_gpu_f32_vec(device, gguf, &format!("{ltag}.ffn_down.bias"), ltag, n_embd)?;
    Ok(GpuFfn::SwiGlu {
        n_ff,
        w_gate,
        w_up,
        w_down,
        b_gate,
        b_up,
        b_down,
    })
}

impl Mamba2MetalRunner {
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, MetalError> {
        let gguf_ref: &GgufFile = &gguf;
        let arch = gguf_ref.req_str("general.architecture")?.to_string();
        if !matches!(
            arch.as_str(),
            "mamba2" | "nemotron_h" | "falcon-h1" | "granitehybrid" | "granitemoehybrid"
        ) {
            return Err(MetalError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        let n_layers = gguf_ref.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf_ref.req_usize(&format!("{pfx}.embedding_length"))?;
        let max_ctx_train = gguf_ref
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let rms_eps = gguf_ref
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf_ref.get_meta(&format!("{pfx}.attention.layer_norm_epsilon")))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        let ssm_conv_kernel = gguf_ref.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_inner_size = gguf_ref.req_usize(&format!("{pfx}.ssm.inner_size"))?;
        let ssm_state_size = gguf_ref.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_head = gguf_ref.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_n_group = gguf_ref
            .get_meta(&format!("{pfx}.ssm.group_count"))
            .and_then(|v| v.as_usize())
            .filter(|&g| g > 0)
            .unwrap_or(1);
        let ssm_norm_eps = gguf_ref
            .get_meta(&format!("{pfx}.ssm.norm_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(rms_eps);

        let n_head_per_layer = meta_usize_per_layer(
            gguf_ref,
            &format!("{pfx}.attention.head_count"),
            n_layers,
            0,
        );
        let default_kv = n_head_per_layer.first().copied().unwrap_or(0);
        let n_head_kv_per_layer = meta_usize_per_layer(
            gguf_ref,
            &format!("{pfx}.attention.head_count_kv"),
            n_layers,
            default_kv,
        );
        let n_ff_per_layer =
            meta_usize_per_layer(gguf_ref, &format!("{pfx}.feed_forward_length"), n_layers, 0);

        let max_n_head = n_head_per_layer.iter().copied().max().unwrap_or(0);
        let max_n_head_kv = n_head_kv_per_layer.iter().copied().max().unwrap_or(0);
        let max_n_ff = n_ff_per_layer.iter().copied().max().unwrap_or(0);
        let head_dim = gguf_ref
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| n_embd.checked_div(max_n_head).unwrap_or(0));
        let rope_dim = gguf_ref
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf_ref
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);
        let rope_type = u32::from(pfx == "falcon-h1");
        let rope_finetuned = gguf_ref
            .get_meta(&format!("{pfx}.rope.scaling.finetuned"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let use_rope = pfx != "nemotron_h" && rope_finetuned && rope_dim > 0;
        let rope_orig_ctx = gguf_ref
            .get_meta(&format!("{pfx}.rope.scaling.original_context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(max_ctx_train);

        let embedding_scale = gguf_ref
            .get_meta(&format!("{pfx}.embedding_scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);
        let attn_scale = gguf_ref
            .get_meta(&format!("{pfx}.attention.scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .unwrap_or(if head_dim > 0 {
                1.0 / (head_dim as f32).sqrt()
            } else {
                1.0
            });
        let residual_scale = gguf_ref
            .get_meta(&format!("{pfx}.residual_scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .unwrap_or(1.0);
        let inv_logit_scale = gguf_ref
            .get_meta(&format!("{pfx}.logit_scale"))
            .and_then(|v| v.as_f32())
            .filter(|&s| s != 0.0)
            .map_or(1.0, |s| 1.0 / s);

        let embd_info =
            gguf_ref
                .tensor_info("token_embd.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf_ref.path().display().to_string(),
                    tensor: "token_embd.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let vocab_size = gguf_ref
            .get_meta(&format!("{pfx}.vocab_size"))
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| embd_info.dims.get(1).copied().unwrap_or(0));

        let cfg = Mamba2MetalConfig {
            arch: arch.clone(),
            n_layers,
            n_embd,
            vocab_size,
            max_ctx,
            rms_eps,
            ssm_conv_kernel,
            ssm_inner_size,
            ssm_state_size,
            ssm_n_head,
            ssm_n_group,
            ssm_norm_eps,
            head_dim,
            rope_dim,
            rope_freq_base,
            rope_type,
            use_rope,
            embedding_scale,
            attn_scale,
            residual_scale,
            inv_logit_scale,
        };

        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueFailed)?;
        let src = NSString::from_str(LLAMA_OPS_MSL);
        let lib = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let pipes = Pipelines {
            embed_q4_0: build_pipe(&device, &lib, "embed_q4_0", 64)?,
            embed_q8_0: build_pipe(&device, &lib, "embed_q8_0", 64)?,
            embed_q6_k: build_pipe(&device, &lib, "embed_q6_k", 64)?,
            embed_f32: build_pipe(&device, &lib, "embed_f32", 64)?,
            embed_q4_0_batch: build_pipe(&device, &lib, "embed_q4_0_batch", 64)?,
            embed_q8_0_batch: build_pipe(&device, &lib, "embed_q8_0_batch", 64)?,
            embed_q6_k_batch: build_pipe(&device, &lib, "embed_q6_k_batch", 64)?,
            embed_f32_batch: build_pipe(&device, &lib, "embed_f32_batch", 64)?,
            rms_norm: build_pipe(&device, &lib, "rms_norm_f32", 256)?,
            rms_norm_batch: build_pipe(&device, &lib, "rms_norm_f32_batch", 256)?,
            add_bias: build_pipe(&device, &lib, "add_bias_f32", 256)?,
            copy_f32: build_pipe(&device, &lib, "copy_f32", 256)?,
            scale_inplace: build_pipe(&device, &lib, "scale_inplace_f32", 256)?,
            scale_axpy: build_pipe(&device, &lib, "scale_axpy_f32", 256)?,
            silu_mul: build_pipe(&device, &lib, "silu_mul_f32", 256)?,
            relu2_inplace: build_pipe(&device, &lib, "relu2_inplace_f32", 256)?,
            rope_kv: build_pipe(&device, &lib, "rope_and_kv_store_f16", 64)?,
            rope_kv_batch: build_pipe(&device, &lib, "rope_and_kv_store_f16_batch", 64)?,
            gqa_attn_ext_batch: build_pipe(&device, &lib, "gqa_attn_f16_kv_ext_batch", 128)?,
            split3_rows: build_pipe(&device, &lib, "split3_rows_f32", 256)?,
            mamba2_conv: build_pipe(&device, &lib, "mamba2_conv1d_silu_f32", 256)?,
            mamba2_scan: build_pipe(&device, &lib, "mamba2_scan_f32", 128)?,
            mamba2_gate_norm: build_pipe(&device, &lib, "mamba2_gate_grouped_norm_f32", 256)?,
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res", 128)?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_res", 128)?,
            gemv_q5_k: build_pipe(&device, &lib, "gemv_q5_k_res", 128)?,
            gemv_q6_k: build_pipe(&device, &lib, "gemv_q6_k_res", 128)?,
            gemv_bf16: build_pipe(&device, &lib, "gemv_bf16_res", 128)?,
            gemv_f32: build_pipe(&device, &lib, "gemv_f32_res", 128)?,
            gemm_q4_0: build_pipe(&device, &lib, "gemm_q4_0_res", 128)?,
            gemm_q4_1: build_pipe(&device, &lib, "gemm_q4_1_res", 128)?,
            gemm_q8_0: build_pipe(&device, &lib, "gemm_q8_0_res", 128)?,
            gemm_q5_k: build_pipe(&device, &lib, "gemm_q5_k_res", 128)?,
            gemm_q6_k: build_pipe(&device, &lib, "gemm_q6_k_res", 128)?,
            gemm_bf16: build_pipe(&device, &lib, "gemm_bf16_res", 128)?,
            gemm_f32: build_pipe(&device, &lib, "gemm_f32_res", 128)?,
            argmax: build_pipe(&device, &lib, "argmax_f32", 256)?,
        };
        let lora = MetalLoraEngine::new(&device, &lib)?;

        let shared_weights = try_zero_copy_gguf_buf(&device, gguf_ref);
        let zero_copy_weights = shared_weights.is_some();
        let sw = shared_weights.as_ref();

        let token_embd = load_gpu_mat(
            &device,
            gguf_ref,
            sw,
            "token_embd.weight",
            "global",
            n_embd,
            vocab_size,
        )?;
        let (output_tied, output_separate) = if gguf_ref.tensor_info("output.weight").is_some() {
            (
                false,
                Some(load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    "output.weight",
                    "global",
                    n_embd,
                    vocab_size,
                )?),
            )
        } else {
            (true, None)
        };
        let output_bias =
            load_opt_gpu_f32_vec(&device, gguf_ref, "output.bias", "global", vocab_size)?;
        let output_norm = load_gpu_f32_vec_any(
            &device,
            gguf_ref,
            &["output_norm.weight", "norm.weight"],
            "global",
            n_embd,
        )?;

        let half = rope_dim / 2;
        let rope_freq_buf = if pfx == "falcon-h1" || half == 0 {
            None
        } else if max_ctx > rope_orig_ctx {
            load_opt_gpu_f32_vec_any(
                &device,
                gguf_ref,
                &[
                    "rope_factors_long.weight",
                    "blk.0.rope_factors_long.weight",
                    "rope_freqs.weight",
                    "rope_factors_short.weight",
                    "blk.0.rope_factors_short.weight",
                ],
                "global",
                half,
            )?
        } else {
            load_opt_gpu_f32_vec_any(
                &device,
                gguf_ref,
                &[
                    "rope_freqs.weight",
                    "rope_factors_short.weight",
                    "blk.0.rope_factors_short.weight",
                ],
                "global",
                half,
            )?
        };
        let dummy_f32 = upload_f32(&device, &[1.0f32])?;
        let (has_rope_freqs, rope_freqs) = match rope_freq_buf {
            Some(b) => (true, b),
            None => (false, dummy_f32.clone()),
        };

        let moe_params =
            crate::moe_gpu::parse_moe_arch_params(gguf_ref, pfx, &arch, n_layers, max_n_ff)?;
        let mut max_n_exp = 0usize;
        let mut max_k_exp = 0usize;
        let mut max_ff_exp = 0usize;
        let mut max_ff_shexp = 0usize;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm = load_gpu_f32_vec(
                &device,
                gguf_ref,
                &format!("{ltag}.attn_norm.weight"),
                &ltag,
                n_embd,
            )?;
            let block = match pfx {
                "mamba2" => {
                    let ssm = load_gpu_ssm(&device, gguf_ref, sw, &ltag, n_embd, &cfg, true)?;
                    GpuMamba2Block::SsmOnly { ssm }
                }
                "nemotron_h" => {
                    let nh_kv = n_head_kv_per_layer[l];
                    let n_ff = n_ff_per_layer[l];
                    if nh_kv == 0 && n_ff == 0 {
                        let ssm = load_gpu_ssm(&device, gguf_ref, sw, &ltag, n_embd, &cfg, true)?;
                        GpuMamba2Block::SsmOnly { ssm }
                    } else if n_ff == 0 {
                        let attn = load_gpu_attn(
                            &device,
                            gguf_ref,
                            sw,
                            &ltag,
                            n_embd,
                            n_head_per_layer[l],
                            nh_kv,
                            head_dim,
                            max_ctx,
                        )?;
                        GpuMamba2Block::AttnOnly { attn }
                    } else {
                        let w_up = load_gpu_mat(
                            &device,
                            gguf_ref,
                            sw,
                            &format!("{ltag}.ffn_up.weight"),
                            &ltag,
                            n_embd,
                            n_ff,
                        )?;
                        let w_down = load_gpu_mat(
                            &device,
                            gguf_ref,
                            sw,
                            &format!("{ltag}.ffn_down.weight"),
                            &ltag,
                            n_ff,
                            n_embd,
                        )?;
                        let b_up = load_opt_gpu_f32_vec(
                            &device,
                            gguf_ref,
                            &format!("{ltag}.ffn_up.bias"),
                            &ltag,
                            n_ff,
                        )?;
                        let b_down = load_opt_gpu_f32_vec(
                            &device,
                            gguf_ref,
                            &format!("{ltag}.ffn_down.bias"),
                            &ltag,
                            n_embd,
                        )?;
                        GpuMamba2Block::FfnOnly {
                            ffn: GpuFfn::Relu2 {
                                n_ff,
                                w_up,
                                w_down,
                                b_up,
                                b_down,
                            },
                        }
                    }
                }
                "granitehybrid" | "granitemoehybrid" => {
                    let nh_kv = n_head_kv_per_layer[l];
                    let n_ff = n_ff_per_layer[l];
                    let ffn_norm = load_gpu_f32_vec(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.ffn_norm.weight"),
                        &ltag,
                        n_embd,
                    )?;
                    let ffn = if let Some(ref mp) = moe_params
                        && let Some(moe) = crate::moe_gpu::load_gpu_layer_moe(
                            &device, gguf_ref, sw, l, &ltag, n_embd, mp,
                        )? {
                        max_n_exp = max_n_exp.max(moe.cfg.n_expert);
                        max_k_exp = max_k_exp.max(moe.cfg.n_expert_used);
                        max_ff_exp = max_ff_exp.max(moe.cfg.n_ff_exp);
                        max_ff_shexp = max_ff_shexp.max(moe.cfg.n_ff_shexp);
                        GpuFfn::SparseMoe(moe)
                    } else {
                        load_gpu_swiglu(&device, gguf_ref, sw, &ltag, n_embd, n_ff)?
                    };
                    if nh_kv == 0 {
                        let ssm = load_gpu_ssm(&device, gguf_ref, sw, &ltag, n_embd, &cfg, true)?;
                        GpuMamba2Block::SsmAndFfn { ssm, ffn_norm, ffn }
                    } else {
                        let attn = load_gpu_attn(
                            &device,
                            gguf_ref,
                            sw,
                            &ltag,
                            n_embd,
                            n_head_per_layer[l],
                            nh_kv,
                            head_dim,
                            max_ctx,
                        )?;
                        GpuMamba2Block::AttnAndFfn {
                            attn,
                            ffn_norm,
                            ffn,
                        }
                    }
                }
                "falcon-h1" => {
                    let attn = load_gpu_attn(
                        &device,
                        gguf_ref,
                        sw,
                        &ltag,
                        n_embd,
                        n_head_per_layer[l],
                        n_head_kv_per_layer[l],
                        head_dim,
                        max_ctx,
                    )?;
                    let ssm = load_gpu_ssm(&device, gguf_ref, sw, &ltag, n_embd, &cfg, false)?;
                    let ffn_norm = load_gpu_f32_vec_any(
                        &device,
                        gguf_ref,
                        &[
                            &format!("{ltag}.ffn_norm.weight"),
                            &format!("{ltag}.ffn_norm"),
                        ],
                        &ltag,
                        n_embd,
                    )?;
                    let ffn =
                        load_gpu_swiglu(&device, gguf_ref, sw, &ltag, n_embd, n_ff_per_layer[l])?;
                    GpuMamba2Block::ParallelAttnSsmAndFfn {
                        attn,
                        ssm,
                        ffn_norm,
                        ffn,
                    }
                }
                _ => unreachable!(),
            };
            layers.push(GpuMamba2Layer { attn_norm, block });
        }

        let d_inner = ssm_inner_size;
        let bc_dim = ssm_n_group * ssm_state_size;
        let conv_dim = d_inner + 2 * bc_dim;
        let d_in_proj = 2 * d_inner + 2 * bc_dim + ssm_n_head;
        let max_q = max_n_head * head_dim;
        let max_kv = max_n_head_kv * head_dim;
        let f32_sz = 4usize;

        let token_ids_buf = alloc_buf(&device, MAX_BATCH * 4)?;
        let argmax_buf = alloc_buf(&device, 16)?;
        let x = alloc_buf(&device, MAX_BATCH * n_embd * f32_sz)?;
        let xb = alloc_buf(&device, MAX_BATCH * n_embd.max(max_q) * f32_sz)?;
        let xb_par = alloc_buf(&device, MAX_BATCH * n_embd * f32_sz)?;
        let xb2 = alloc_buf(&device, MAX_BATCH * n_embd * f32_sz)?;
        let in_proj = alloc_buf(&device, MAX_BATCH * d_in_proj * f32_sz)?;
        let ssm_z = alloc_buf(&device, MAX_BATCH * d_inner * f32_sz)?;
        let ssm_xbc = alloc_buf(&device, MAX_BATCH * conv_dim * f32_sz)?;
        let ssm_dt = alloc_buf(&device, MAX_BATCH * ssm_n_head * f32_sz)?;
        let ssm_x = alloc_buf(&device, MAX_BATCH * d_inner * f32_sz)?;
        let ssm_b = alloc_buf(&device, MAX_BATCH * bc_dim * f32_sz)?;
        let ssm_c = alloc_buf(&device, MAX_BATCH * bc_dim * f32_sz)?;
        let ssm_y = alloc_buf(&device, MAX_BATCH * d_inner * f32_sz)?;
        let q = alloc_buf(&device, MAX_BATCH * max_q * f32_sz)?;
        let k = alloc_buf(&device, MAX_BATCH * max_kv * f32_sz)?;
        let v = alloc_buf(&device, MAX_BATCH * max_kv * f32_sz)?;
        let hb = alloc_buf(&device, MAX_BATCH * max_n_ff * f32_sz)?;
        let hb2 = alloc_buf(&device, MAX_BATCH * max_n_ff * f32_sz)?;
        let logits = alloc_buf(&device, vocab_size * f32_sz)?;
        let moe_engine = crate::moe_gpu::GpuMoeEngine::new(
            &device,
            &lib,
            MAX_BATCH,
            n_embd,
            max_n_exp,
            max_k_exp,
            max_ff_exp,
            max_ff_shexp,
        )?;

        Ok(Self {
            cfg,
            device,
            queue,
            pipes,
            token_embd,
            output_tied,
            output_separate,
            output_bias,
            output_norm,
            rope_freqs,
            has_rope_freqs,
            dummy_f32,
            layers,
            token_ids_buf,
            argmax_buf,
            x,
            xb,
            xb_par,
            xb2,
            in_proj,
            ssm_z,
            ssm_xbc,
            ssm_dt,
            ssm_x,
            ssm_b,
            ssm_c,
            ssm_y,
            q,
            k,
            v,
            hb,
            hb2,
            logits,
            moe_engine,
            lora,
            zero_copy_weights,
            _gguf: gguf,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn is_zero_copy_weights(&self) -> bool {
        self.zero_copy_weights
    }

    pub fn bind_lora(&mut self, loras: &tm_lora::LoraSet) -> Result<(), MetalError> {
        self.lora.bind(&self.device, &self._gguf, loras, MAX_BATCH)
    }

    pub fn reset_state(&mut self) {
        let zero_ssm = |ssm: &GpuSsm| {
            // SAFETY: No command buffer is in flight when `reset_state(&mut self)` is called.
            unsafe {
                if ssm.ssm_state_bytes > 0 {
                    core::ptr::write_bytes(
                        ssm.ssm_state.contents().as_ptr() as *mut u8,
                        0,
                        ssm.ssm_state_bytes,
                    );
                }
                if ssm.conv_hist_bytes > 0 {
                    core::ptr::write_bytes(
                        ssm.conv_hist.contents().as_ptr() as *mut u8,
                        0,
                        ssm.conv_hist_bytes,
                    );
                }
            }
        };
        for lw in &self.layers {
            match &lw.block {
                GpuMamba2Block::SsmOnly { ssm }
                | GpuMamba2Block::SsmAndFfn { ssm, .. }
                | GpuMamba2Block::ParallelAttnSsmAndFfn { ssm, .. } => zero_ssm(ssm),
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_linear(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        lora_name: &str,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        n_tok: usize,
        add_residual: bool,
    ) -> Result<(), MetalError> {
        const THREADS: usize = 128;
        if n_tok == 1 {
            let (pipe, name, rows_per_tg): (&MtlPipe, &'static str, usize) = match mat.dtype {
                DType::Q4_0 => (&self.pipes.gemv_q4_0, "gemv_q4_0_res", 32),
                DType::Q4_1 => (&self.pipes.gemv_q4_1, "gemv_q4_1_res", 32),
                DType::Q8_0 => (&self.pipes.gemv_q8_0, "gemv_q8_0_res", 16),
                DType::Q5K => (&self.pipes.gemv_q5_k, "gemv_q5_k_res", 8),
                DType::Q6K => (&self.pipes.gemv_q6_k, "gemv_q6_k_res", 16),
                DType::BF16 => (&self.pipes.gemv_bf16, "gemv_bf16_res", 16),
                DType::F32 => (&self.pipes.gemv_f32, "gemv_f32_res", 8),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            require_fit(pipe, name, THREADS)?;
            let args = GemvArgs {
                n_in: mat.cols as u32,
                n_out: mat.rows as u32,
                add_residual: u32::from(add_residual),
                row_bytes: mat.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
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
                    DType::Q4_0 => (&self.pipes.gemm_q4_0, "gemm_q4_0_res", 32, 32),
                    DType::Q4_1 => (&self.pipes.gemm_q4_1, "gemm_q4_1_res", 32, 32),
                    DType::Q8_0 => (&self.pipes.gemm_q8_0, "gemm_q8_0_res", 4, 8),
                    DType::Q5K => (&self.pipes.gemm_q5_k, "gemm_q5_k_res", 4, 8),
                    DType::Q6K => (&self.pipes.gemm_q6_k, "gemm_q6_k_res", 4, 8),
                    DType::BF16 => (&self.pipes.gemm_bf16, "gemm_bf16_res", 32, 32),
                    DType::F32 => (&self.pipes.gemm_f32, "gemm_f32_res", 4, 8),
                    other => return Err(MetalError::UnsupportedDType(other)),
                };
            require_fit(pipe, name, THREADS)?;
            let args = GemmArgs {
                n_in: mat.cols as u32,
                n_out: mat.rows as u32,
                n_tok: n_tok as u32,
                add_residual: u32::from(add_residual),
                row_bytes: mat.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
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
        self.lora
            .encode_linear(enc, lora_name, x_buf, 0, y_buf, 0, n_tok);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rms_norm_rows(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        x_off: usize,
        w_buf: &ProtocolObject<dyn MTLBuffer>,
        out_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
        n_tok: usize,
    ) -> Result<(), MetalError> {
        const TG: usize = 256;
        let args = NormArgs {
            n: n as u32,
            eps: self.cfg.rms_eps,
        };
        let pipe = if n_tok == 1 {
            require_fit(&self.pipes.rms_norm, "rms_norm_f32", TG)?;
            &self.pipes.rms_norm
        } else {
            require_fit(&self.pipes.rms_norm_batch, "rms_norm_f32_batch", TG)?;
            &self.pipes.rms_norm_batch
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x_buf), x_off, 0);
            enc.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<NormArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_tok,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_add_bias(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        bias_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        d_out: usize,
        n_tok: usize,
    ) -> Result<(), MetalError> {
        const TG: usize = 256;
        require_fit(&self.pipes.add_bias, "add_bias_f32", TG)?;
        let bargs = BiasArgs {
            d_out: d_out as u32,
            n_tok: n_tok as u32,
        };
        enc.setComputePipelineState(&self.pipes.add_bias);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(bias_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(y_buf), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&bargs).cast::<c_void>(),
                core::mem::size_of::<BiasArgs>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: d_out.div_ceil(TG),
                height: n_tok,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_copy_f32(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src: &ProtocolObject<dyn MTLBuffer>,
        dst: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
    ) -> Result<(), MetalError> {
        const TG: usize = 256;
        require_fit(&self.pipes.copy_f32, "copy_f32", TG)?;
        let n_u32 = n as u32;
        enc.setComputePipelineState(&self.pipes.copy_f32);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&n_u32).cast::<c_void>(),
                core::mem::size_of::<u32>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(TG),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_scale_inplace(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
        scale: f32,
    ) -> Result<(), MetalError> {
        if (scale - 1.0).abs() <= 1e-12 {
            return Ok(());
        }
        const TG: usize = 256;
        require_fit(&self.pipes.scale_inplace, "scale_inplace_f32", TG)?;
        let sargs = ScaleVecArgs { n: n as u32, scale };
        enc.setComputePipelineState(&self.pipes.scale_inplace);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(y_buf), 0, 0);
            enc.setBytes_length_atIndex(
                NonNull::from(&sargs).cast::<c_void>(),
                core::mem::size_of::<ScaleVecArgs>(),
                1,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(TG),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_scale_axpy(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        src_buf: &ProtocolObject<dyn MTLBuffer>,
        dst_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
        scale: f32,
    ) -> Result<(), MetalError> {
        const TG: usize = 256;
        require_fit(&self.pipes.scale_axpy, "scale_axpy_f32", TG)?;
        let sargs = ScaleVecArgs { n: n as u32, scale };
        enc.setComputePipelineState(&self.pipes.scale_axpy);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst_buf), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&sargs).cast::<c_void>(),
                core::mem::size_of::<ScaleVecArgs>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(TG),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn encode_ssm(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        ssm: &GpuSsm,
        in_buf: &ProtocolObject<dyn MTLBuffer>,
        b: usize,
    ) -> Result<(), MetalError> {
        let d = self.cfg.n_embd;
        let d_inner = self.cfg.ssm_inner_size;
        let bc_dim = self.cfg.ssm_n_group * self.cfg.ssm_state_size;
        let conv_dim = d_inner + 2 * bc_dim;
        let n_heads = self.cfg.ssm_n_head;
        let d_in_proj = d_inner + conv_dim + n_heads;

        self.encode_linear(
            enc,
            &ssm.ssm_in,
            &ssm.ssm_in.name,
            in_buf,
            &self.in_proj,
            b,
            false,
        )?;

        // Split `in_proj` into `ssm_z`, `ssm_xbc`, `ssm_dt`
        {
            const TG: usize = 256;
            require_fit(&self.pipes.split3_rows, "split3_rows_f32", TG)?;
            let sargs = Split3RowsArgs {
                d0: d_inner as u32,
                d1: conv_dim as u32,
                d2: n_heads as u32,
                n_tok: b as u32,
            };
            enc.setComputePipelineState(&self.pipes.split3_rows);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.in_proj), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_z), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_xbc), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_dt), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&sargs).cast::<c_void>(),
                    core::mem::size_of::<Split3RowsArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: d_in_proj.div_ceil(TG),
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // Causal depthwise Conv1d + SiLU + split `x`, `B`, `C`
        {
            let tg = conv_dim.clamp(1, 256);
            require_fit(&self.pipes.mamba2_conv, "mamba2_conv1d_silu_f32", tg)?;
            let cargs = Mamba2ConvArgs {
                d_inner: d_inner as u32,
                bc_dim: bc_dim as u32,
                d_conv: self.cfg.ssm_conv_kernel as u32,
                n_tok: b as u32,
                has_bias: u32::from(ssm.conv_b.is_some()),
            };
            let cb_buf = ssm.conv_b.as_ref().unwrap_or(&self.dummy_f32);
            enc.setComputePipelineState(&self.pipes.mamba2_conv);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.ssm_xbc), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&ssm.conv_w), 0, 1);
                enc.setBuffer_offset_atIndex(Some(cb_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&ssm.conv_hist), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_x), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_b), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_c), 0, 6);
                enc.setBytes_length_atIndex(
                    NonNull::from(&cargs).cast::<c_void>(),
                    core::mem::size_of::<Mamba2ConvArgs>(),
                    7,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: conv_dim.div_ceil(tg),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // Selective State Scan (`4` `(h, j)` rows per threadgroup of `128` threads)
        {
            const TG: usize = 128;
            require_fit(&self.pipes.mamba2_scan, "mamba2_scan_f32", TG)?;
            let sargs = Mamba2ScanArgs {
                n_heads: n_heads as u32,
                n_groups: self.cfg.ssm_n_group as u32,
                head_dim: (d_inner / n_heads) as u32,
                d_state: self.cfg.ssm_state_size as u32,
                n_tok: b as u32,
            };
            enc.setComputePipelineState(&self.pipes.mamba2_scan);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.ssm_x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_c), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_dt), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&ssm.dt_bias), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&ssm.ssm_a), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&ssm.ssm_d), 0, 6);
                enc.setBuffer_offset_atIndex(Some(&ssm.ssm_state), 0, 7);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_y), 0, 8);
                enc.setBytes_length_atIndex(
                    NonNull::from(&sargs).cast::<c_void>(),
                    core::mem::size_of::<Mamba2ScanArgs>(),
                    9,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: d_inner.div_ceil(4),
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

        // `silu(z) * y` + optional grouped `ssm_norm`
        {
            const TG: usize = 256;
            require_fit(
                &self.pipes.mamba2_gate_norm,
                "mamba2_gate_grouped_norm_f32",
                TG,
            )?;
            let gargs = Mamba2GateNormArgs {
                d_inner: d_inner as u32,
                n_groups: self.cfg.ssm_n_group as u32,
                has_norm: u32::from(ssm.ssm_norm.is_some()),
                eps: self.cfg.ssm_norm_eps,
            };
            let nw_buf = ssm.ssm_norm.as_ref().unwrap_or(&self.dummy_f32);
            enc.setComputePipelineState(&self.pipes.mamba2_gate_norm);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.ssm_y), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.ssm_z), 0, 1);
                enc.setBuffer_offset_atIndex(Some(nw_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&gargs).cast::<c_void>(),
                    core::mem::size_of::<Mamba2GateNormArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.cfg.ssm_n_group,
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // `ssm_out` -> `xb2` + residual into `x`
        let _ = d;
        self.encode_linear(
            enc,
            &ssm.ssm_out,
            &ssm.ssm_out.name,
            &self.ssm_y,
            &self.xb2,
            b,
            false,
        )?;
        self.encode_scale_axpy(
            enc,
            &self.xb2,
            &self.x,
            b * self.cfg.n_embd,
            self.cfg.residual_scale,
        )?;
        Ok(())
    }

    fn encode_attn(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        attn: &GpuAttn,
        in_buf: &ProtocolObject<dyn MTLBuffer>,
        chunk_pos: usize,
        b: usize,
    ) -> Result<(), MetalError> {
        let hd = self.cfg.head_dim;
        let q_dim = attn.n_head * hd;
        let kv_dim = attn.n_head_kv * hd;

        self.encode_linear(enc, &attn.wq, &attn.wq.name, in_buf, &self.q, b, false)?;
        self.encode_linear(enc, &attn.wk, &attn.wk.name, in_buf, &self.k, b, false)?;
        self.encode_linear(enc, &attn.wv, &attn.wv.name, in_buf, &self.v, b, false)?;
        if let Some(ref bq) = attn.bq {
            self.encode_add_bias(enc, bq, &self.q, q_dim, b)?;
        }
        if let Some(ref bk) = attn.bk {
            self.encode_add_bias(enc, bk, &self.k, kv_dim, b)?;
        }
        if let Some(ref bv) = attn.bv {
            self.encode_add_bias(enc, bv, &self.v, kv_dim, b)?;
        }

        // RoPE (disabled when `rope_dim == 0` for `nemotron_h`!) + F16 KV store
        {
            const TG: usize = 64;
            let eff_rope_dim = if self.cfg.use_rope {
                self.cfg.rope_dim
            } else {
                0
            };
            let rope_pairs = (attn.n_head + attn.n_head_kv) * (eff_rope_dim / 2);
            let rope_total = rope_pairs.max(kv_dim).max(1);
            let rargs = RopeKvArgs {
                n_head: attn.n_head as u32,
                n_head_kv: attn.n_head_kv as u32,
                head_dim: hd as u32,
                rope_dim: eff_rope_dim as u32,
                pos: chunk_pos as u32,
                rope_type: self.cfg.rope_type,
                has_freq_factors: u32::from(self.has_rope_freqs),
                freq_base: self.cfg.rope_freq_base,
            };
            let pipe = if b == 1 {
                require_fit(&self.pipes.rope_kv, "rope_and_kv_store_f16", TG)?;
                &self.pipes.rope_kv
            } else {
                require_fit(&self.pipes.rope_kv_batch, "rope_and_kv_store_f16_batch", TG)?;
                &self.pipes.rope_kv_batch
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.k), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.v), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&attn.k_cache), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&attn.v_cache), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&self.rope_freqs), 0, 5);
                enc.setBytes_length_atIndex(
                    NonNull::from(&rargs).cast::<c_void>(),
                    core::mem::size_of::<RopeKvArgs>(),
                    6,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rope_total.div_ceil(TG),
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // GQA attention (`gqa_attn_f16_kv_ext_batch` handles arbitrary `attn_scale` and `b >= 1`)
        {
            const TG: usize = 128;
            require_fit(
                &self.pipes.gqa_attn_ext_batch,
                "gqa_attn_f16_kv_ext_batch",
                TG,
            )?;
            let aargs = AttnExtArgs {
                n_head: attn.n_head as u32,
                n_head_kv: attn.n_head_kv as u32,
                head_dim: hd as u32,
                start_pos: chunk_pos as u32,
                n_tok: b as u32,
                window: 0,
                attn_scale: self.cfg.attn_scale,
                attn_softcap: 0.0,
            };
            enc.setComputePipelineState(&self.pipes.gqa_attn_ext_batch);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&attn.k_cache), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&attn.v_cache), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&aargs).cast::<c_void>(),
                    core::mem::size_of::<AttnExtArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: attn.n_head,
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        }

        self.encode_linear(enc, &attn.wo, &attn.wo.name, &self.xb, &self.xb2, b, false)?;
        if let Some(ref bo) = attn.bo {
            self.encode_add_bias(enc, bo, &self.xb2, self.cfg.n_embd, b)?;
        }
        self.encode_scale_axpy(
            enc,
            &self.xb2,
            &self.x,
            b * self.cfg.n_embd,
            self.cfg.residual_scale,
        )?;
        Ok(())
    }

    fn encode_ffn(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        ffn: &GpuFfn,
        b: usize,
    ) -> Result<(), MetalError> {
        match ffn {
            GpuFfn::SwiGlu {
                n_ff,
                w_gate,
                w_up,
                w_down,
                b_gate,
                b_up,
                b_down,
            } => {
                let ff = *n_ff;
                self.encode_linear(enc, w_gate, &w_gate.name, &self.xb, &self.hb, b, false)?;
                self.encode_linear(enc, w_up, &w_up.name, &self.xb, &self.hb2, b, false)?;
                if let Some(bg) = b_gate {
                    self.encode_add_bias(enc, bg, &self.hb, ff, b)?;
                }
                if let Some(bu) = b_up {
                    self.encode_add_bias(enc, bu, &self.hb2, ff, b)?;
                }
                {
                    const TG: usize = 256;
                    require_fit(&self.pipes.silu_mul, "silu_mul_f32", TG)?;
                    let total_ff = (b * ff) as u32;
                    enc.setComputePipelineState(&self.pipes.silu_mul);
                    // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.hb), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.hb2), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&total_ff).cast::<c_void>(),
                            core::mem::size_of::<u32>(),
                            2,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: (total_ff as usize).div_ceil(TG),
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
                self.encode_linear(enc, w_down, &w_down.name, &self.hb, &self.xb2, b, false)?;
                if let Some(bd) = b_down {
                    self.encode_add_bias(enc, bd, &self.xb2, self.cfg.n_embd, b)?;
                }
            }
            GpuFfn::Relu2 {
                n_ff,
                w_up,
                w_down,
                b_up,
                b_down,
            } => {
                let ff = *n_ff;
                self.encode_linear(enc, w_up, &w_up.name, &self.xb, &self.hb, b, false)?;
                if let Some(bu) = b_up {
                    self.encode_add_bias(enc, bu, &self.hb, ff, b)?;
                }
                {
                    const TG: usize = 256;
                    require_fit(&self.pipes.relu2_inplace, "relu2_inplace_f32", TG)?;
                    let total_ff = (b * ff) as u32;
                    enc.setComputePipelineState(&self.pipes.relu2_inplace);
                    // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.hb), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&total_ff).cast::<c_void>(),
                            core::mem::size_of::<u32>(),
                            1,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: (total_ff as usize).div_ceil(TG),
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
                self.encode_linear(enc, w_down, &w_down.name, &self.hb, &self.xb2, b, false)?;
                if let Some(bd) = b_down {
                    self.encode_add_bias(enc, bd, &self.xb2, self.cfg.n_embd, b)?;
                }
            }
            GpuFfn::SparseMoe(moe) => {
                if let Some(ref eng) = self.moe_engine {
                    eng.encode_moe_ffn(
                        enc,
                        &self.lora,
                        moe,
                        &self.xb,
                        &self.xb2,
                        self.cfg.n_embd,
                        b,
                    )?;
                }
            }
        }
        self.encode_scale_axpy(
            enc,
            &self.xb2,
            &self.x,
            b * self.cfg.n_embd,
            self.cfg.residual_scale,
        )?;
        Ok(())
    }

    fn encode_chunk_into(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        single_token: Option<u32>,
        chunk_pos: usize,
        b: usize,
        compute_logits: bool,
    ) -> Result<(), MetalError> {
        let d = self.cfg.n_embd;

        // 1. Embedding
        if let Some(tok) = single_token {
            let (pipe, name, n_items): (&MtlPipe, &'static str, usize) = match self.token_embd.dtype
            {
                DType::Q4_0 => (&self.pipes.embed_q4_0, "embed_q4_0", d / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0, "embed_q8_0", d / 32),
                DType::Q6K => (&self.pipes.embed_q6_k, "embed_q6_k", d / 256),
                DType::F32 => (&self.pipes.embed_f32, "embed_f32", d),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            let tg_w = n_items.clamp(1, 64);
            require_fit(pipe, name, tg_w)?;
            let eargs = EmbedArgs {
                token_id: tok,
                n_embd: d as u32,
                row_bytes: self.token_embd.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.token_embd.buf), self.token_embd.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&self.x), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::from(&eargs).cast::<c_void>(),
                    core::mem::size_of::<EmbedArgs>(),
                    2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_items.div_ceil(tg_w),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
            self.lora.encode_embed_single(
                enc,
                "token_embd.weight",
                &self.token_ids_buf,
                tok,
                &self.x,
            );
        } else {
            let (pipe, name, n_items): (&MtlPipe, &'static str, usize) = match self.token_embd.dtype
            {
                DType::Q4_0 => (&self.pipes.embed_q4_0_batch, "embed_q4_0_batch", d / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0_batch, "embed_q8_0_batch", d / 32),
                DType::Q6K => (&self.pipes.embed_q6_k_batch, "embed_q6_k_batch", d / 256),
                DType::F32 => (&self.pipes.embed_f32_batch, "embed_f32_batch", d),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            let tg_w = n_items.clamp(1, 64);
            require_fit(pipe, name, tg_w)?;
            let eargs = EmbedArgs {
                token_id: b as u32,
                n_embd: d as u32,
                row_bytes: self.token_embd.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.token_embd.buf), self.token_embd.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&self.token_ids_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.x), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&eargs).cast::<c_void>(),
                    core::mem::size_of::<EmbedArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_items.div_ceil(tg_w),
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
            self.lora
                .encode_embed_batch(enc, "token_embd.weight", &self.token_ids_buf, &self.x, b);
        }
        self.encode_scale_inplace(enc, &self.x, b * d, self.cfg.embedding_scale)?;

        // 2. Blocks
        for lw in &self.layers {
            self.encode_rms_norm_rows(enc, &self.x, 0, &lw.attn_norm, &self.xb, d, b)?;
            match &lw.block {
                GpuMamba2Block::SsmOnly { ssm } => {
                    self.encode_ssm(enc, ssm, &self.xb, b)?;
                }
                GpuMamba2Block::AttnOnly { attn } => {
                    self.encode_attn(enc, attn, &self.xb, chunk_pos, b)?;
                }
                GpuMamba2Block::FfnOnly { ffn } => {
                    self.encode_ffn(enc, ffn, b)?;
                }
                GpuMamba2Block::SsmAndFfn { ssm, ffn_norm, ffn } => {
                    self.encode_ssm(enc, ssm, &self.xb, b)?;
                    self.encode_rms_norm_rows(enc, &self.x, 0, ffn_norm, &self.xb, d, b)?;
                    self.encode_ffn(enc, ffn, b)?;
                }
                GpuMamba2Block::AttnAndFfn {
                    attn,
                    ffn_norm,
                    ffn,
                } => {
                    self.encode_attn(enc, attn, &self.xb, chunk_pos, b)?;
                    self.encode_rms_norm_rows(enc, &self.x, 0, ffn_norm, &self.xb, d, b)?;
                    self.encode_ffn(enc, ffn, b)?;
                }
                GpuMamba2Block::ParallelAttnSsmAndFfn {
                    attn,
                    ssm,
                    ffn_norm,
                    ffn,
                } => {
                    self.encode_copy_f32(enc, &self.xb, &self.xb_par, b * d)?;
                    self.encode_attn(enc, attn, &self.xb, chunk_pos, b)?;
                    self.encode_ssm(enc, ssm, &self.xb_par, b)?;
                    self.encode_rms_norm_rows(enc, &self.x, 0, ffn_norm, &self.xb, d, b)?;
                    self.encode_ffn(enc, ffn, b)?;
                }
            }
        }

        // 3. Final norm + LM head
        if compute_logits {
            let last_off = (b - 1) * d * 4;
            self.encode_rms_norm_rows(enc, &self.x, last_off, &self.output_norm, &self.xb, d, 1)?;
            let out_mat = if self.output_tied {
                &self.token_embd
            } else {
                self.output_separate.as_ref().unwrap_or(&self.token_embd)
            };
            self.encode_linear(
                enc,
                out_mat,
                "output.weight",
                &self.xb,
                &self.logits,
                1,
                false,
            )?;
            if let Some(ref ob) = self.output_bias {
                self.encode_add_bias(enc, ob, &self.logits, self.cfg.vocab_size, 1)?;
            }
            self.encode_scale_inplace(
                enc,
                &self.logits,
                self.cfg.vocab_size,
                self.cfg.inv_logit_scale,
            )?;
        }

        Ok(())
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], MetalError> {
        if pos >= self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "pos {pos} >= max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(MetalError::InvalidDimensions(format!(
                "token_id {token_id} >= vocab_size {}",
                self.cfg.vocab_size
            )));
        }

        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        if let Err(e) = self.encode_chunk_into(&enc, Some(token_id), pos, 1, compute_logits) {
            enc.endEncoding();
            return Err(e);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        Ok(self.logits())
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, MetalError> {
        if pos >= self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "pos {pos} >= max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(MetalError::InvalidDimensions(format!(
                "token_id {token_id} >= vocab_size {}",
                self.cfg.vocab_size
            )));
        }

        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        if let Err(e) = self.encode_chunk_into(&enc, Some(token_id), pos, 1, true) {
            enc.endEncoding();
            return Err(e);
        }
        const TG: usize = 256;
        require_fit(&self.pipes.argmax, "argmax_f32", TG)?;
        let vocab_u32 = self.cfg.vocab_size as u32;
        enc.setComputePipelineState(&self.pipes.argmax);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.logits), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.argmax_buf), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&vocab_u32).cast::<c_void>(),
                core::mem::size_of::<u32>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: TG,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ptr = self.argmax_buf.contents().as_ptr() as *const u32;
        // SAFETY: `waitUntilCompleted` finished; `argmax_buf` holds at least 1 `u32`.
        Ok(unsafe { *ptr })
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], MetalError> {
        if token_ids.is_empty() {
            return Err(MetalError::InvalidDimensions(
                "forward_batch requires at least 1 token".to_string(),
            ));
        }
        if start_pos + token_ids.len() > self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "start_pos {} + len {} > max_ctx {}",
                start_pos,
                token_ids.len(),
                self.cfg.max_ctx
            )));
        }
        for &tok in token_ids {
            if (tok as usize) >= self.cfg.vocab_size {
                return Err(MetalError::InvalidDimensions(format!(
                    "token_id {tok} >= vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
        }
        if token_ids.len() == 1 {
            return self.forward_token(token_ids[0], start_pos, compute_last_logits);
        }

        let total = token_ids.len();
        let mut offset = 0usize;
        while offset < total {
            let b = (total - offset).min(MAX_BATCH);
            let chunk = &token_ids[offset..offset + b];
            let is_last = offset + b == total;
            {
                let dst = self.token_ids_buf.contents().as_ptr() as *mut u32;
                // SAFETY: `token_ids_buf` is idle and holds `MAX_BATCH` u32s.
                unsafe {
                    core::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, b);
                }
            }
            let cmd = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncodeFailed)?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncodeFailed)?;
            if let Err(e) = self.encode_chunk_into(
                &enc,
                None,
                start_pos + offset,
                b,
                is_last && compute_last_logits,
            ) {
                enc.endEncoding();
                return Err(e);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            offset += b;
        }
        Ok(self.logits())
    }

    pub fn logits(&self) -> &[f32] {
        let ptr = self.logits.contents().as_ptr() as *const f32;
        // SAFETY: `logits` is `StorageModeShared` with `vocab_size` `f32` values.
        unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) }
    }

    pub fn profile(
        &mut self,
        decode_runs: usize,
        prefill_len: usize,
    ) -> Result<crate::MetalProfileReport, MetalError> {
        self.reset_state();
        let n_dec = decode_runs.max(1).min(self.cfg.max_ctx);
        let mut wall_sum = 0.0f64;
        let mut gpu_sum = 0.0f64;
        for step in 0..n_dec {
            let t0 = std::time::Instant::now();
            let cmd = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncodeFailed)?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncodeFailed)?;
            self.encode_chunk_into(&enc, Some(1), step, 1, true)?;
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            wall_sum += t0.elapsed().as_secs_f64();
            gpu_sum += (cmd.GPUEndTime() - cmd.GPUStartTime()).max(0.0);
        }

        self.reset_state();
        let n_pp = prefill_len.clamp(1, self.cfg.max_ctx);
        let pp_ids = vec![1u32; n_pp];
        let t_pp = std::time::Instant::now();
        let _ = self.forward_batch(&pp_ids, 0, false)?;
        let prefill_wall_ms = t_pp.elapsed().as_secs_f64() * 1e3;

        Ok(crate::MetalProfileReport {
            decode_runs: n_dec,
            decode_single_cb_wall_us: (wall_sum * 1e6) / (n_dec as f64),
            decode_single_cb_gpu_us: (gpu_sum * 1e6) / (n_dec as f64),
            decode_stages: Vec::new(),
            prefill_tokens: n_pp,
            prefill_wall_ms,
            prefill_chunk_tokens: n_pp.min(MAX_BATCH),
            prefill_chunk_stages: Vec::new(),
        })
    }
}
