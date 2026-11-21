//! End-to-end Apple Metal GPU runner for Liquid AI `lfm2` (hybrid Gated `ShortConv`
//! + GQA Attention with QK-Norm & optional Sliding Window + SwiGLU FFN + `token_embd_norm`).
//!
//! Encodes the entire hybrid forward pass into a **single `MTLCommandBuffer`** per decode token
//! with zero intermediate host syncs and zero-copy GGUF mmap weights (`newBufferWithBytesNoCopy`).

use core::ffi::c_void;
use core::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

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
use crate::llama_runner::{MetalProfileReport, MetalStageStat};

const LLAMA_OPS_MSL: &str = include_str!("shaders/llama_ops.metal");
const MAX_BATCH: usize = 128;
const METAL_PAGE_SIZE: usize = 16384;

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
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ShortConvArgs {
    n_embd: u32,
    l_cache: u32,
    n_tok: u32,
    _pad: u32,
}

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Hyperparameters for `Lfm2MetalRunner`.
#[derive(Debug, Clone)]
pub struct Lfm2MetalConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv_per_layer: Vec<usize>,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub shortconv_l_cache: usize,
    pub sliding_window: Option<usize>,
    pub vocab_size: usize,
    pub max_ctx: usize,
}

struct GpuMat {
    buf: MtlBuf,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

enum GpuLfm2Op {
    ShortConv {
        in_proj: GpuMat,
        conv_w: MtlBuf,
        out_proj: GpuMat,
        conv_hist: MtlBuf,
        conv_hist_bytes: usize,
    },
    Gqa {
        n_head_kv: usize,
        q_norm: MtlBuf,
        k_norm: MtlBuf,
        wq: GpuMat,
        wk: GpuMat,
        wv: GpuMat,
        wo: GpuMat,
        k_cache: MtlBuf,
        v_cache: MtlBuf,
        kv_bytes: usize,
    },
}

struct GpuLfm2Layer {
    attn_norm: MtlBuf,
    op: GpuLfm2Op,
    ffn_norm: MtlBuf,
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
}

struct Lfm2Pipelines {
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
    qk_norm_batch: MtlPipe,
    rope_kv: MtlPipe,
    rope_kv_batch: MtlPipe,
    gqa_attn_ext_batch: MtlPipe,
    shortconv: MtlPipe,
    gemv_q4_0: MtlPipe,
    gemv_q4_1: MtlPipe,
    gemv_q8_0: MtlPipe,
    gemv_q5_k: MtlPipe,
    gemv_q6_k: MtlPipe,
    gemv_bf16: MtlPipe,
    gemv_f32: MtlPipe,
    gemv_pair_silu_q4_0: MtlPipe,
    gemm_q4_0: MtlPipe,
    gemm_q4_1: MtlPipe,
    gemm_q8_0: MtlPipe,
    gemm_q5_k: MtlPipe,
    gemm_q6_k: MtlPipe,
    gemm_bf16: MtlPipe,
    gemm_f32: MtlPipe,
    silu_mul: MtlPipe,
    argmax: MtlPipe,
}

/// Complete Apple Metal GPU inference runner for `lfm2` hybrid models.
pub struct Lfm2MetalRunner {
    pub cfg: Lfm2MetalConfig,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Lfm2Pipelines,
    token_embd: GpuMat,
    output_separate: Option<GpuMat>,
    output_norm: MtlBuf,
    dummy_rope_freqs: MtlBuf,
    layers: Vec<GpuLfm2Layer>,
    token_ids_buf: MtlBuf,
    argmax_buf: MtlBuf,
    x: MtlBuf,
    xb: MtlBuf,
    q_buf: MtlBuf,
    k_buf: MtlBuf,
    v_buf: MtlBuf,
    attn_out: MtlBuf,
    bcx_buf: MtlBuf,
    conv_out: MtlBuf,
    hb: MtlBuf,
    hb2: MtlBuf,
    logits: MtlBuf,
    zero_copy_weights: bool,
    // IMPORTANT: Must remain LAST so all zero-copy `MTLBuffer`s drop before `_gguf` unmaps.
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

fn alloc_buf(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<MtlBuf, MetalError> {
    let len = bytes.max(16);
    let buf = device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAlloc { bytes: len })?;
    // SAFETY: `contents()` is valid for `len` bytes on `StorageModeShared` and not yet in use by GPU.
    unsafe {
        core::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, len);
    }
    Ok(buf)
}

fn upload_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<MtlBuf, MetalError> {
    let len = bytes.len().max(16);
    let buf = device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAlloc { bytes: len })?;
    if !bytes.is_empty() {
        // SAFETY: `buf` is freshly allocated with length >= `bytes.len()`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.contents().as_ptr() as *mut u8,
                bytes.len(),
            );
        }
    }
    Ok(buf)
}

fn upload_f32(device: &ProtocolObject<dyn MTLDevice>, vals: &[f32]) -> Result<MtlBuf, MetalError> {
    // SAFETY: `f32` is POD; byte slice has exact length `vals.len() * 4`.
    let raw = unsafe { core::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4) };
    upload_bytes(device, raw)
}

fn load_gpu_f32_tensor(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    expected_dims: &[usize],
) -> Result<MtlBuf, MetalError> {
    let (info, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
    let len: usize = expected_dims.iter().product();
    let mut vals = vec![0.0f32; len];
    dequantize_row_bytes(info.dtype, bytes, &mut vals);
    upload_f32(device, &vals)
}

fn try_zero_copy_gguf_buf(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
) -> Option<MtlBuf> {
    let raw = gguf.as_bytes();
    if raw.is_empty() {
        return None;
    }
    let ptr = raw.as_ptr();
    if !(ptr as usize).is_multiple_of(METAL_PAGE_SIZE) {
        return None;
    }
    let aligned_len = raw.len().div_ceil(METAL_PAGE_SIZE) * METAL_PAGE_SIZE;
    let non_null = NonNull::new(ptr as *mut c_void)?;
    // SAFETY: `ptr` is page-aligned and backed by `GgufFile`'s read-only `Mmap` for
    // `aligned_len` bytes. `Lfm2MetalRunner` stores `_gguf: Arc<GgufFile>` as its
    // last field so this `MTLBuffer` is always dropped before unmapping.
    unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            non_null,
            aligned_len,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
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
        buf,
        offset,
        dtype: info.dtype,
        rows,
        cols,
        row_bytes: info.dtype.row_bytes(cols),
    })
}

impl Lfm2MetalRunner {
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, MetalError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "lfm2" {
            return Err(MetalError::UnsupportedArch(arch));
        }

        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueFailed)?;

        let src = NSString::from_str(LLAMA_OPS_MSL);
        let lib = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let pipes = Lfm2Pipelines {
            embed_q4_0: build_pipe(&device, &lib, "embed_q4_0", 64)?,
            embed_q8_0: build_pipe(&device, &lib, "embed_q8_0", 64)?,
            embed_q6_k: build_pipe(&device, &lib, "embed_q6_k", 64)?,
            embed_f32: build_pipe(&device, &lib, "embed_f32", 256)?,
            embed_q4_0_batch: build_pipe(&device, &lib, "embed_q4_0_batch", 64)?,
            embed_q8_0_batch: build_pipe(&device, &lib, "embed_q8_0_batch", 64)?,
            embed_q6_k_batch: build_pipe(&device, &lib, "embed_q6_k_batch", 64)?,
            embed_f32_batch: build_pipe(&device, &lib, "embed_f32_batch", 64)?,
            rms_norm: build_pipe(&device, &lib, "rms_norm_f32", 256)?,
            rms_norm_batch: build_pipe(&device, &lib, "rms_norm_f32_batch", 256)?,
            qk_norm_batch: build_pipe(&device, &lib, "qk_norm_f32_batch", 32)?,
            rope_kv: build_pipe(&device, &lib, "rope_and_kv_store_f16", 256)?,
            rope_kv_batch: build_pipe(&device, &lib, "rope_and_kv_store_f16_batch", 64)?,
            gqa_attn_ext_batch: build_pipe(&device, &lib, "gqa_attn_f16_kv_ext_batch", 128)?,
            shortconv: build_pipe(&device, &lib, "lfm2_shortconv_f32", 256)?,
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res", 128)?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_fast", 128)?,
            gemv_q5_k: build_pipe(&device, &lib, "gemv_q5_k_res", 128)?,
            gemv_q6_k: build_pipe(&device, &lib, "gemv_q6_k_res", 128)?,
            gemv_bf16: build_pipe(&device, &lib, "gemv_bf16_res", 128)?,
            gemv_f32: build_pipe(&device, &lib, "gemv_f32_res", 128)?,
            gemv_pair_silu_q4_0: build_pipe(&device, &lib, "gemv_pair_silu_q4_0", 128)?,
            gemm_q4_0: build_pipe(&device, &lib, "gemm_q4_0_res", 128)?,
            gemm_q4_1: build_pipe(&device, &lib, "gemm_q4_1_res", 128)?,
            gemm_q8_0: build_pipe(&device, &lib, "gemm_q8_0_res", 128)?,
            gemm_q5_k: build_pipe(&device, &lib, "gemm_q5_k_res", 128)?,
            gemm_q6_k: build_pipe(&device, &lib, "gemm_q6_k_res", 128)?,
            gemm_bf16: build_pipe(&device, &lib, "gemm_bf16_res", 128)?,
            gemm_f32: build_pipe(&device, &lib, "gemm_f32_res", 128)?,
            silu_mul: build_pipe(&device, &lib, "silu_mul_f32", 256)?,
            argmax: build_pipe(&device, &lib, "argmax_f32", 256)?,
        };

        let n_layers = gguf.req_usize("lfm2.block_count")?;
        let n_embd = gguf.req_usize("lfm2.embedding_length")?;
        let n_ff = if let Some(arr) = gguf
            .get_meta("lfm2.feed_forward_length")
            .and_then(|v| v.as_usize_array())
        {
            arr[0]
        } else {
            gguf.req_usize("lfm2.feed_forward_length")?
        };
        let n_head = if let Some(arr) = gguf
            .get_meta("lfm2.attention.head_count")
            .and_then(|v| v.as_usize_array())
        {
            arr[0]
        } else {
            gguf.req_usize("lfm2.attention.head_count")?
        };
        let n_head_kv_per_layer = if let Some(arr) = gguf
            .get_meta("lfm2.attention.head_count_kv")
            .and_then(|v| v.as_usize_array())
        {
            arr
        } else {
            vec![
                gguf.get_meta("lfm2.attention.head_count_kv")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(n_head);
                n_layers
            ]
        };

        let head_dim = gguf
            .get_meta("lfm2.attention.key_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head);
        let rope_dim = gguf
            .get_meta("lfm2.rope.dimension_count")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf
            .get_meta("lfm2.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let rms_eps = gguf.req_f32("lfm2.attention.layer_norm_rms_epsilon")?;
        let shortconv_l_cache = gguf.req_usize("lfm2.shortconv.l_cache")?;
        if shortconv_l_cache > 17 {
            return Err(MetalError::InvalidDimensions(format!(
                "shortconv.l_cache ({shortconv_l_cache}) exceeds Metal kernel max (17)"
            )));
        }
        let sliding_window = gguf
            .get_meta("lfm2.attention.sliding_window")
            .and_then(|v| v.as_usize())
            .filter(|&w| w > 0);

        let file_ctx = gguf
            .get_meta("lfm2.context_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(max_ctx);
        let max_ctx = max_ctx.min(file_ctx).max(1);

        let emb_info =
            gguf.tensor_info("token_embd.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf.path().display().to_string(),
                    tensor: "token_embd.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let vocab_size = emb_info.dims[1];

        let cfg = Lfm2MetalConfig {
            arch,
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv_per_layer: n_head_kv_per_layer.clone(),
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_eps,
            shortconv_l_cache,
            sliding_window,
            vocab_size,
            max_ctx,
        };

        let shared_gguf_buf = try_zero_copy_gguf_buf(&device, &gguf);
        let zero_copy_weights = shared_gguf_buf.is_some();
        let sw = shared_gguf_buf.as_ref();

        let token_embd = load_gpu_mat(
            &device,
            &gguf,
            sw,
            "token_embd.weight",
            "global",
            n_embd,
            vocab_size,
        )?;
        let output_separate = if gguf.tensor_info("output.weight").is_some() {
            Some(load_gpu_mat(
                &device,
                &gguf,
                sw,
                "output.weight",
                "global",
                n_embd,
                vocab_size,
            )?)
        } else {
            None
        };
        let output_norm = if gguf.tensor_info("token_embd_norm.weight").is_some() {
            load_gpu_f32_tensor(
                &device,
                &gguf,
                "token_embd_norm.weight",
                "global",
                &[n_embd],
            )?
        } else {
            load_gpu_f32_tensor(&device, &gguf, "output_norm.weight", "global", &[n_embd])?
        };
        let dummy_rope_freqs = upload_f32(&device, &[1.0f32; 256])?;

        let q_dim = n_head * head_dim;
        let hist_len = shortconv_l_cache.saturating_sub(1);
        let mut layers = Vec::with_capacity(n_layers);
        for (il, &n_head_kv) in n_head_kv_per_layer.iter().enumerate() {
            let ltag = format!("blk.{il}");
            let attn_norm = load_gpu_f32_tensor(
                &device,
                &gguf,
                &format!("{ltag}.attn_norm.weight"),
                &ltag,
                &[n_embd],
            )?;
            let ffn_norm = load_gpu_f32_tensor(
                &device,
                &gguf,
                &format!("{ltag}.ffn_norm.weight"),
                &ltag,
                &[n_embd],
            )?;

            let op = if n_head_kv == 0 {
                let in_proj = load_gpu_mat(
                    &device,
                    &gguf,
                    sw,
                    &format!("{ltag}.shortconv.in_proj.weight"),
                    &ltag,
                    n_embd,
                    3 * n_embd,
                )?;
                let conv_name = format!("{ltag}.shortconv.conv.weight");
                let conv_w = if let Some(info) = gguf.tensor_info(&conv_name)
                    && info.dims.len() == 3
                {
                    load_gpu_f32_tensor(
                        &device,
                        &gguf,
                        &conv_name,
                        &ltag,
                        &[shortconv_l_cache, 1, n_embd],
                    )?
                } else {
                    load_gpu_f32_tensor(
                        &device,
                        &gguf,
                        &conv_name,
                        &ltag,
                        &[shortconv_l_cache, n_embd],
                    )?
                };
                let out_proj = load_gpu_mat(
                    &device,
                    &gguf,
                    sw,
                    &format!("{ltag}.shortconv.out_proj.weight"),
                    &ltag,
                    n_embd,
                    n_embd,
                )?;
                let conv_hist_bytes = n_embd * hist_len * 4;
                GpuLfm2Op::ShortConv {
                    in_proj,
                    conv_w,
                    out_proj,
                    conv_hist: alloc_buf(&device, conv_hist_bytes)?,
                    conv_hist_bytes,
                }
            } else {
                let kv_dim = n_head_kv * head_dim;
                let kv_bytes = max_ctx * kv_dim * 2;
                GpuLfm2Op::Gqa {
                    n_head_kv,
                    q_norm: load_gpu_f32_tensor(
                        &device,
                        &gguf,
                        &format!("{ltag}.attn_q_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    k_norm: load_gpu_f32_tensor(
                        &device,
                        &gguf,
                        &format!("{ltag}.attn_k_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    wq: load_gpu_mat(
                        &device,
                        &gguf,
                        sw,
                        &format!("{ltag}.attn_q.weight"),
                        &ltag,
                        n_embd,
                        q_dim,
                    )?,
                    wk: load_gpu_mat(
                        &device,
                        &gguf,
                        sw,
                        &format!("{ltag}.attn_k.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?,
                    wv: load_gpu_mat(
                        &device,
                        &gguf,
                        sw,
                        &format!("{ltag}.attn_v.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?,
                    wo: load_gpu_mat(
                        &device,
                        &gguf,
                        sw,
                        &format!("{ltag}.attn_output.weight"),
                        &ltag,
                        q_dim,
                        n_embd,
                    )?,
                    k_cache: alloc_buf(&device, kv_bytes)?,
                    v_cache: alloc_buf(&device, kv_bytes)?,
                    kv_bytes,
                }
            };

            let w_gate = load_gpu_mat(
                &device,
                &gguf,
                sw,
                &format!("{ltag}.ffn_gate.weight"),
                &ltag,
                n_embd,
                n_ff,
            )?;
            let w_up = load_gpu_mat(
                &device,
                &gguf,
                sw,
                &format!("{ltag}.ffn_up.weight"),
                &ltag,
                n_embd,
                n_ff,
            )?;
            let w_down = load_gpu_mat(
                &device,
                &gguf,
                sw,
                &format!("{ltag}.ffn_down.weight"),
                &ltag,
                n_ff,
                n_embd,
            )?;

            layers.push(GpuLfm2Layer {
                attn_norm,
                op,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        let max_n_head_kv = n_head_kv_per_layer.iter().copied().max().unwrap_or(0);
        let max_kv_dim = max_n_head_kv.max(1) * head_dim;

        let token_ids_buf = alloc_buf(&device, MAX_BATCH * 4)?;
        let argmax_buf = alloc_buf(&device, 16)?;
        let x = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let xb = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let q_buf = alloc_buf(&device, MAX_BATCH * q_dim * 4)?;
        let k_buf = alloc_buf(&device, MAX_BATCH * max_kv_dim * 4)?;
        let v_buf = alloc_buf(&device, MAX_BATCH * max_kv_dim * 4)?;
        let attn_out = alloc_buf(&device, MAX_BATCH * q_dim * 4)?;
        let bcx_buf = alloc_buf(&device, MAX_BATCH * 3 * n_embd * 4)?;
        let conv_out = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let hb = alloc_buf(&device, MAX_BATCH * n_ff * 4)?;
        let hb2 = alloc_buf(&device, MAX_BATCH * n_ff * 4)?;
        let logits = alloc_buf(&device, vocab_size * 4)?;

        Ok(Self {
            cfg,
            device,
            queue,
            pipes,
            token_embd,
            output_separate,
            output_norm,
            dummy_rope_freqs,
            layers,
            token_ids_buf,
            argmax_buf,
            x,
            xb,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            bcx_buf,
            conv_out,
            hb,
            hb2,
            logits,
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

    pub fn reset_state(&mut self) {
        for layer in &self.layers {
            match &layer.op {
                GpuLfm2Op::ShortConv {
                    conv_hist,
                    conv_hist_bytes,
                    ..
                } => {
                    if *conv_hist_bytes > 0 {
                        // SAFETY: Exclusive `&mut self` guarantees no active GPU command buffer is using `conv_hist`.
                        unsafe {
                            core::ptr::write_bytes(
                                conv_hist.contents().as_ptr() as *mut u8,
                                0,
                                *conv_hist_bytes,
                            );
                        }
                    }
                }
                GpuLfm2Op::Gqa {
                    k_cache,
                    v_cache,
                    kv_bytes,
                    ..
                } => {
                    // SAFETY: Exclusive `&mut self` guarantees no active GPU command buffer is using KV cache.
                    unsafe {
                        core::ptr::write_bytes(
                            k_cache.contents().as_ptr() as *mut u8,
                            0,
                            *kv_bytes,
                        );
                        core::ptr::write_bytes(
                            v_cache.contents().as_ptr() as *mut u8,
                            0,
                            *kv_bytes,
                        );
                    }
                }
            }
        }
    }

    pub fn logits(&self) -> &[f32] {
        let ptr = self.logits.contents().as_ptr() as *const f32;
        // SAFETY: `self.logits` is `StorageModeShared` of size `vocab_size * 4` and GPU work completed.
        unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) }
    }

    #[inline]
    fn output_mat(&self) -> &GpuMat {
        self.output_separate.as_ref().unwrap_or(&self.token_embd)
    }

    fn enc_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &GpuMat,
        x: &MtlBuf,
        y: &MtlBuf,
        add_residual: bool,
    ) {
        let (pipe, rows_per_tg): (&MtlPipe, usize) = match w.dtype {
            DType::Q4_0 => (&self.pipes.gemv_q4_0, 32),
            DType::Q4_1 => (&self.pipes.gemv_q4_1, 32),
            DType::Q8_0 => (&self.pipes.gemv_q8_0, 16),
            DType::Q6K => (&self.pipes.gemv_q6_k, 16),
            DType::Q5K => (&self.pipes.gemv_q5_k, 8),
            DType::BF16 => (&self.pipes.gemv_bf16, 16),
            DType::F32 => (&self.pipes.gemv_f32, 8),
            other => unreachable!("validated dtype at load: {other:?}"),
        };
        let args = GemvArgs {
            n_in: w.cols as u32,
            n_out: w.rows as u32,
            add_residual: u32::from(add_residual),
            row_bytes: w.row_bytes as u32,
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.buf), w.offset, 0);
            enc.setBuffer_offset_atIndex(Some(x), 0, 1);
            enc.setBuffer_offset_atIndex(Some(y), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<GemvArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: w.rows.div_ceil(rows_per_tg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    fn enc_gemm(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &GpuMat,
        x: &MtlBuf,
        y: &MtlBuf,
        n_tok: usize,
        add_residual: bool,
    ) {
        if n_tok == 1 {
            self.enc_gemv(enc, w, x, y, add_residual);
            return;
        }
        let (pipe, r_tile, t_tile): (&MtlPipe, usize, usize) = match w.dtype {
            DType::Q4_0 => (&self.pipes.gemm_q4_0, 32, 32),
            DType::Q4_1 => (&self.pipes.gemm_q4_1, 32, 32),
            DType::Q8_0 => (&self.pipes.gemm_q8_0, 4, 8),
            DType::Q5K => (&self.pipes.gemm_q5_k, 4, 8),
            DType::Q6K => (&self.pipes.gemm_q6_k, 4, 8),
            DType::BF16 => (&self.pipes.gemm_bf16, 32, 32),
            DType::F32 => (&self.pipes.gemm_f32, 4, 8),
            other => unreachable!("validated dtype at load: {other:?}"),
        };
        let args = GemmArgs {
            n_in: w.cols as u32,
            n_out: w.rows as u32,
            n_tok: n_tok as u32,
            add_residual: u32::from(add_residual),
            row_bytes: w.row_bytes as u32,
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.buf), w.offset, 0);
            enc.setBuffer_offset_atIndex(Some(x), 0, 1);
            enc.setBuffer_offset_atIndex(Some(y), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<GemmArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: w.rows.div_ceil(r_tile),
                height: n_tok.div_ceil(t_tile),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_rms_norm_batch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MtlBuf,
        w: &MtlBuf,
        out: &MtlBuf,
        n: usize,
        eps: f32,
        n_tok: usize,
    ) {
        let args = NormArgs { n: n as u32, eps };
        enc.setComputePipelineState(&self.pipes.rms_norm_batch);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x), 0, 0);
            enc.setBuffer_offset_atIndex(Some(w), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out), 0, 2);
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
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_chunk(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        n_tok: usize,
        start_pos: usize,
        single_token_id: Option<u32>,
        compute_logits: bool,
        greedy_argmax: bool,
    ) {
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let hd = self.cfg.head_dim;
        let rope_dim = self.cfg.rope_dim;
        let window = self.cfg.sliding_window.unwrap_or(0) as u32;
        let attn_scale = 1.0f32 / (hd as f32).sqrt();

        // 1. Input embedding
        if let Some(tid) = single_token_id {
            let args = EmbedArgs {
                token_id: tid,
                n_embd: n_embd as u32,
                row_bytes: self.token_embd.row_bytes as u32,
            };
            let (pipe, units): (&MtlPipe, usize) = match self.token_embd.dtype {
                DType::Q4_0 => (&self.pipes.embed_q4_0, n_embd / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0, n_embd / 32),
                DType::Q6K => (&self.pipes.embed_q6_k, n_embd / 256),
                DType::F32 => (&self.pipes.embed_f32, n_embd),
                other => unreachable!("unsupported embed dtype: {other:?}"),
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers outlive command buffer execution.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.token_embd.buf), self.token_embd.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&self.x), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<EmbedArgs>(),
                    2,
                );
            }
            let tg = 64usize.min(units.max(1));
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: units.div_ceil(tg),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            let args = EmbedArgs {
                token_id: 0,
                n_embd: n_embd as u32,
                row_bytes: self.token_embd.row_bytes as u32,
            };
            let (pipe, units): (&MtlPipe, usize) = match self.token_embd.dtype {
                DType::Q4_0 => (&self.pipes.embed_q4_0_batch, n_embd / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0_batch, n_embd / 32),
                DType::Q6K => (&self.pipes.embed_q6_k_batch, n_embd / 256),
                DType::F32 => (&self.pipes.embed_f32_batch, n_embd),
                other => unreachable!("unsupported embed dtype: {other:?}"),
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers outlive command buffer execution.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.token_embd.buf), self.token_embd.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&self.token_ids_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.x), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<EmbedArgs>(),
                    3,
                );
            }
            let tg_x = 64usize.min(units.max(1));
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: units.div_ceil(tg_x),
                    height: n_tok,
                    depth: 1,
                },
                MTLSize {
                    width: tg_x,
                    height: 1,
                    depth: 1,
                },
            );
        }

        // 2. Hybrid `ShortConv` / `GQA` + `SwiGLU` layers
        for layer in &self.layers {
            self.enc_rms_norm_batch(
                enc,
                &self.x,
                &layer.attn_norm,
                &self.xb,
                n_embd,
                self.cfg.rms_eps,
                n_tok,
            );

            match &layer.op {
                GpuLfm2Op::ShortConv {
                    in_proj,
                    conv_w,
                    out_proj,
                    conv_hist,
                    ..
                } => {
                    self.enc_gemm(enc, in_proj, &self.xb, &self.bcx_buf, n_tok, false);
                    let sc_args = ShortConvArgs {
                        n_embd: n_embd as u32,
                        l_cache: self.cfg.shortconv_l_cache as u32,
                        n_tok: n_tok as u32,
                        _pad: 0,
                    };
                    enc.setComputePipelineState(&self.pipes.shortconv);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.bcx_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(conv_w), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(conv_hist), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&self.conv_out), 0, 3);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&sc_args).cast::<c_void>(),
                            core::mem::size_of::<ShortConvArgs>(),
                            4,
                        );
                    }
                    let tg = 256usize.min(n_embd.max(1));
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: n_embd.div_ceil(tg),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: tg,
                            height: 1,
                            depth: 1,
                        },
                    );
                    self.enc_gemm(enc, out_proj, &self.conv_out, &self.x, n_tok, true);
                }
                GpuLfm2Op::Gqa {
                    n_head_kv,
                    q_norm,
                    k_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    k_cache,
                    v_cache,
                    ..
                } => {
                    let kv_dim = *n_head_kv * hd;
                    self.enc_gemm(enc, wq, &self.xb, &self.q_buf, n_tok, false);
                    self.enc_gemm(enc, wk, &self.xb, &self.k_buf, n_tok, false);
                    self.enc_gemm(enc, wv, &self.xb, &self.v_buf, n_tok, false);

                    let qn_args = NormArgs {
                        n: hd as u32,
                        eps: self.cfg.rms_eps,
                    };
                    enc.setComputePipelineState(&self.pipes.qk_norm_batch);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(q_norm), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&qn_args).cast::<c_void>(),
                            core::mem::size_of::<NormArgs>(),
                            2,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: self.cfg.n_head,
                            height: n_tok,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );

                    enc.setComputePipelineState(&self.pipes.qk_norm_batch);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(k_norm), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&qn_args).cast::<c_void>(),
                            core::mem::size_of::<NormArgs>(),
                            2,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: *n_head_kv,
                            height: n_tok,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );

                    let rk_args = RopeKvArgs {
                        n_head: self.cfg.n_head as u32,
                        n_head_kv: *n_head_kv as u32,
                        head_dim: hd as u32,
                        rope_dim: rope_dim as u32,
                        pos: start_pos as u32,
                        rope_type: 2, // NeoX
                        has_freq_factors: 0,
                        freq_base: self.cfg.rope_freq_base,
                    };
                    let max_threads_x =
                        ((self.cfg.n_head + *n_head_kv) * (rope_dim / 2)).max(kv_dim);
                    if n_tok == 1 {
                        enc.setComputePipelineState(&self.pipes.rope_kv);
                        // SAFETY: Buffers outlive command buffer execution.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(k_cache), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(v_cache), 0, 4);
                            enc.setBuffer_offset_atIndex(Some(&self.dummy_rope_freqs), 0, 5);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&rk_args).cast::<c_void>(),
                                core::mem::size_of::<RopeKvArgs>(),
                                6,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: max_threads_x.div_ceil(256),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 256,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        enc.setComputePipelineState(&self.pipes.rope_kv_batch);
                        // SAFETY: Buffers outlive command buffer execution.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 1);
                            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(k_cache), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(v_cache), 0, 4);
                            enc.setBuffer_offset_atIndex(Some(&self.dummy_rope_freqs), 0, 5);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&rk_args).cast::<c_void>(),
                                core::mem::size_of::<RopeKvArgs>(),
                                6,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: max_threads_x.div_ceil(64),
                                height: n_tok,
                                depth: 1,
                            },
                            MTLSize {
                                width: 64,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }

                    let attn_args = AttnExtArgs {
                        n_head: self.cfg.n_head as u32,
                        n_head_kv: *n_head_kv as u32,
                        head_dim: hd as u32,
                        start_pos: start_pos as u32,
                        n_tok: n_tok as u32,
                        window,
                        attn_scale,
                        _pad: 0,
                    };
                    enc.setComputePipelineState(&self.pipes.gqa_attn_ext_batch);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(k_cache), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(v_cache), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 3);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&attn_args).cast::<c_void>(),
                            core::mem::size_of::<AttnExtArgs>(),
                            4,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: self.cfg.n_head,
                            height: n_tok,
                            depth: 1,
                        },
                        MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );

                    self.enc_gemm(enc, wo, &self.attn_out, &self.x, n_tok, true);
                }
            }

            // SwiGLU FFN + residual add into `x`
            self.enc_rms_norm_batch(
                enc,
                &self.x,
                &layer.ffn_norm,
                &self.xb,
                n_embd,
                self.cfg.rms_eps,
                n_tok,
            );
            if n_tok == 1 && layer.w_gate.dtype == DType::Q4_0 && layer.w_up.dtype == DType::Q4_0 {
                let g_args = GemvArgs {
                    n_in: n_embd as u32,
                    n_out: n_ff as u32,
                    add_residual: 0,
                    row_bytes: layer.w_gate.row_bytes as u32,
                };
                enc.setComputePipelineState(&self.pipes.gemv_pair_silu_q4_0);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&layer.w_gate.buf), layer.w_gate.offset, 0);
                    enc.setBuffer_offset_atIndex(Some(&layer.w_up.buf), layer.w_up.offset, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.hb), 0, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&g_args).cast::<c_void>(),
                        core::mem::size_of::<GemvArgs>(),
                        4,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n_ff.div_ceil(16),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                self.enc_gemm(enc, &layer.w_gate, &self.xb, &self.hb, n_tok, false);
                self.enc_gemm(enc, &layer.w_up, &self.xb, &self.hb2, n_tok, false);
                let total_ff = (n_tok * n_ff) as u32;
                enc.setComputePipelineState(&self.pipes.silu_mul);
                // SAFETY: Buffers outlive command buffer execution.
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
                        width: (n_tok * n_ff).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            self.enc_gemm(enc, &layer.w_down, &self.hb, &self.x, n_tok, true);
        }

        // 3. Final `token_embd_norm` + LM head + optional `argmax`
        if compute_logits {
            let last_x_off = (n_tok - 1) * n_embd * 4;
            let norm_args = NormArgs {
                n: n_embd as u32,
                eps: self.cfg.rms_eps,
            };
            enc.setComputePipelineState(&self.pipes.rms_norm);
            // SAFETY: Buffers outlive command buffer execution.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.x), last_x_off, 0);
                enc.setBuffer_offset_atIndex(Some(&self.output_norm), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&norm_args).cast::<c_void>(),
                    core::mem::size_of::<NormArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            self.enc_gemv(enc, self.output_mat(), &self.xb, &self.logits, false);

            if greedy_argmax {
                let n_vocab = self.cfg.vocab_size as u32;
                enc.setComputePipelineState(&self.pipes.argmax);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.logits), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.argmax_buf), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&n_vocab).cast::<c_void>(),
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
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], MetalError> {
        if pos >= self.cfg.max_ctx || (token_id as usize) >= self.cfg.vocab_size {
            return Err(MetalError::InvalidDimensions(format!(
                "invalid token_id={token_id} or pos={pos}"
            )));
        }
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        self.encode_chunk(&enc, 1, pos, Some(token_id), compute_logits, false);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(self.logits())
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, MetalError> {
        if pos >= self.cfg.max_ctx || (token_id as usize) >= self.cfg.vocab_size {
            return Err(MetalError::InvalidDimensions(format!(
                "invalid token_id={token_id} or pos={pos}"
            )));
        }
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        self.encode_chunk(&enc, 1, pos, Some(token_id), true, true);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        // SAFETY: `self.argmax_buf` is 16-byte `StorageModeShared` and GPU work completed.
        let id = unsafe { *(self.argmax_buf.contents().as_ptr() as *const u32) };
        Ok(id)
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], MetalError> {
        if token_ids.is_empty() {
            return Ok(self.logits());
        }
        if start_pos + token_ids.len() > self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "batch end {} > max_ctx {}",
                start_pos + token_ids.len(),
                self.cfg.max_ctx
            )));
        }
        for &tid in token_ids {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(MetalError::InvalidDimensions(format!(
                    "token_id {tid} >= vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
        }
        if token_ids.len() == 1 {
            return self.forward_token(token_ids[0], start_pos, compute_last_logits);
        }

        let mut offset = 0usize;
        while offset < token_ids.len() {
            let b = (token_ids.len() - offset).min(MAX_BATCH);
            let is_last = offset + b == token_ids.len();
            // SAFETY: Exclusive `&mut self` and previous chunk completed.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    token_ids[offset..offset + b].as_ptr(),
                    self.token_ids_buf.contents().as_ptr() as *mut u32,
                    b,
                );
            }
            let cmd = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncodeFailed)?;
            let enc = cmd
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncodeFailed)?;
            self.encode_chunk(
                &enc,
                b,
                start_pos + offset,
                None,
                is_last && compute_last_logits,
                false,
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            offset += b;
        }
        Ok(self.logits())
    }

    pub fn profile(
        &mut self,
        n_runs: usize,
        ctx_pos: usize,
    ) -> Result<MetalProfileReport, MetalError> {
        self.reset_state();
        let prefill_len = ctx_pos
            .min(self.cfg.max_ctx.saturating_sub(n_runs + 2))
            .max(1);
        let pp: Vec<u32> = (0..prefill_len)
            .map(|i| ((i % (self.cfg.vocab_size - 1)) + 1) as u32)
            .collect();
        let t_pp = Instant::now();
        self.forward_batch(&pp, 0, true)?;
        let prefill_wall_ms = t_pp.elapsed().as_secs_f64() * 1e3;

        let mut tok = 1u32;
        let t0 = Instant::now();
        for step in 0..n_runs {
            tok = self.forward_token_greedy(tok, prefill_len + step)?;
        }
        let wall_us = (t0.elapsed().as_secs_f64() * 1e6) / (n_runs.max(1) as f64);

        let mut bytes_per_tok = 0usize;
        for l in &self.layers {
            match &l.op {
                GpuLfm2Op::ShortConv {
                    in_proj, out_proj, ..
                } => {
                    bytes_per_tok +=
                        in_proj.rows * in_proj.row_bytes + out_proj.rows * out_proj.row_bytes;
                }
                GpuLfm2Op::Gqa { wq, wk, wv, wo, .. } => {
                    bytes_per_tok += wq.rows * wq.row_bytes
                        + wk.rows * wk.row_bytes
                        + wv.rows * wv.row_bytes
                        + wo.rows * wo.row_bytes;
                }
            }
            bytes_per_tok += l.w_gate.rows * l.w_gate.row_bytes
                + l.w_up.rows * l.w_up.row_bytes
                + l.w_down.rows * l.w_down.row_bytes;
        }
        let out_m = self.output_mat();
        bytes_per_tok += out_m.rows * out_m.row_bytes;

        Ok(MetalProfileReport {
            decode_runs: n_runs,
            decode_single_cb_wall_us: wall_us,
            decode_single_cb_gpu_us: wall_us,
            decode_stages: vec![MetalStageStat {
                name: "lfm2_fused_e2e_cmdbuf",
                gpu_us: wall_us,
                bytes: bytes_per_tok as u64,
                flops: 0,
            }],
            prefill_tokens: prefill_len,
            prefill_wall_ms,
            prefill_chunk_tokens: prefill_len.min(MAX_BATCH),
            prefill_chunk_stages: Vec::new(),
        })
    }
}
