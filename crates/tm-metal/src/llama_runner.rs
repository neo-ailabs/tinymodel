//! End-to-end Apple Metal GPU runner for `llama` and `qwen3` GGUF models.
//!
//! Encodes the entire token forward pass (embedding lookup -> N layers of
//! RMSNorm, QKV GEMV, RoPE+KV-store, Flash-style online-softmax GQA, Wo+residual,
//! FFN RMSNorm, fused Gate+Up+SwiGLU, Wdown+residual -> output norm -> LM head)
//! into a **single `MTLCommandBuffer`** per token with zero intermediate host syncs.

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
/// Maximum number of tokens processed per batched prefill command buffer.
const MAX_BATCH: usize = 128;
/// Apple Silicon VM page size (16 KiB; also a multiple of 4 KiB on x86_64).
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
struct AttnArgs {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    seq_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnArgs {
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    seq_len: u32,
    n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct QkvGemvArgs {
    n_in: u32,
    q_rows: u32,
    k_rows: u32,
    v_rows: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EmbedArgs {
    token_id: u32,
    n_embd: u32,
    row_bytes: u32,
}

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Hyperparameters for `LlamaMetalRunner`.
#[derive(Debug, Clone)]
pub struct LlamaMetalConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub rope_type: u32, // 0 = Norm, 1 = Neox
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

struct GpuLayer {
    attn_norm: MtlBuf,
    ffn_norm: MtlBuf,
    q_norm: Option<MtlBuf>,
    k_norm: Option<MtlBuf>,
    wq: GpuMat,
    wk: GpuMat,
    wv: GpuMat,
    wo: GpuMat,
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
    k_cache: MtlBuf,
    v_cache: MtlBuf,
}

struct Pipelines {
    // Single-token decode pipelines
    embed_q4_0: MtlPipe,
    embed_q8_0: MtlPipe,
    embed_q6_k: MtlPipe,
    embed_f32: MtlPipe,
    rms_norm: MtlPipe,
    qk_norm: MtlPipe,
    rope_kv: MtlPipe,
    gqa_attn: MtlPipe,
    gemv_q4_0: MtlPipe,
    gemv_qkv_q4_0: MtlPipe,
    gemv_pair_silu_q4_0: MtlPipe,
    gemv_q4_1: MtlPipe,
    gemv_q8_0: MtlPipe,
    gemv_q5_k: MtlPipe,
    gemv_q6_k: MtlPipe,
    gemv_bf16: MtlPipe,
    gemv_f32: MtlPipe,
    argmax: MtlPipe,
    // Batched prefill pipelines
    embed_q4_0_batch: MtlPipe,
    embed_q8_0_batch: MtlPipe,
    embed_q6_k_batch: MtlPipe,
    embed_f32_batch: MtlPipe,
    rms_norm_batch: MtlPipe,
    qk_norm_batch: MtlPipe,
    rope_kv_batch: MtlPipe,
    gqa_attn_batch: MtlPipe,
    fa_gqa_batch: MtlPipe,
    fa_gqa_batch_hd256: MtlPipe,
    gemm_q4_0: MtlPipe,
    silu_mul: MtlPipe,
    gemm_q4_1: MtlPipe,
    gemm_q8_0: MtlPipe,
    gemm_q5_k: MtlPipe,
    gemm_q6_k: MtlPipe,
    gemm_bf16: MtlPipe,
    gemm_f32: MtlPipe,
}

/// Complete Apple Metal GPU inference runner for `llama` and `qwen3` models.
pub struct LlamaMetalRunner {
    pub cfg: LlamaMetalConfig,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Pipelines,
    token_embd: GpuMat,
    output_tied: bool,
    output_separate: Option<GpuMat>,
    output_norm: MtlBuf,
    rope_freqs: MtlBuf,
    has_rope_freqs: bool,
    layers: Vec<GpuLayer>,
    // Pre-allocated activation buffers (sized for MAX_BATCH, zero allocation during inference)
    token_ids_buf: MtlBuf,
    argmax_buf: MtlBuf,
    x: MtlBuf,
    xb: MtlBuf,
    q: MtlBuf,
    k: MtlBuf,
    v: MtlBuf,
    hb: MtlBuf,
    hb2: MtlBuf,
    logits: MtlBuf,
    lora: MetalLoraEngine,
    zero_copy_weights: bool,
    // IMPORTANT: Must remain LAST so all `MTLBuffer` fields (including zero-copy views
    // backed by the GGUF mmap) are dropped BEFORE `_gguf` unmaps the file.
    _gguf: Arc<GgufFile>,
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
    // Verify `maxTotalThreadsPerThreadgroup` up front (`AGENTS.md` rule #9) so
    // dispatches never silently no-op AND hot-path encoding makes zero ObjC queries.
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
    let byte_len = core::mem::size_of_val(vals);
    let ptr = NonNull::new(vals.as_ptr() as *mut c_void)
        .ok_or(MetalError::BufferAlloc { bytes: byte_len })?;
    // SAFETY: `vals` is valid for `byte_len` readable bytes.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            byte_len,
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
    // `Mmap::map` maps whole OS pages (zero-padded to page boundary); Metal requires
    // `newBufferWithBytesNoCopy` length to be page-aligned.
    let aligned_len = raw.len().div_ceil(METAL_PAGE_SIZE) * METAL_PAGE_SIZE;
    let non_null = NonNull::new(ptr as *mut c_void)?;
    // SAFETY: `ptr` is page-aligned and backed by `GgufFile`'s read-only `Mmap` for
    // `aligned_len` bytes. `LlamaMetalRunner` stores `_gguf: Arc<GgufFile>` as its
    // last field so this `MTLBuffer` is always dropped before the mapping is unmapped.
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
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K | DType::F32 => {}
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

impl LlamaMetalRunner {
    /// Loads a `llama` or `qwen3` GGUF model onto the default Apple Metal GPU.
    ///
    /// Uses zero-copy `newBufferWithBytesNoCopy` over `gguf`'s memory map whenever
    /// the mapping is page-aligned (standard for all `GgufFile::open` files on Apple Silicon).
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, MetalError> {
        let gguf_ref: &GgufFile = &gguf;
        let arch = gguf_ref.req_str("general.architecture")?.to_string();
        if arch != "llama" && arch != "qwen3" {
            return Err(MetalError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        let n_layers = gguf_ref.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf_ref.req_usize(&format!("{pfx}.embedding_length"))?;
        let n_ff = gguf_ref.req_usize(&format!("{pfx}.feed_forward_length"))?;
        let n_head = gguf_ref.req_usize(&format!("{pfx}.attention.head_count"))?;
        let n_head_kv = gguf_ref
            .get_meta(&format!("{pfx}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim = gguf_ref
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head);
        let rope_dim = gguf_ref
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf_ref
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "qwen3" { 1_000_000.0 } else { 10000.0 });
        let rms_eps = gguf_ref
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "qwen3" { 1e-6 } else { 1e-5 });
        let rope_type = if pfx == "qwen3" { 1 } else { 0 };

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
            rms_norm: build_pipe(&device, &lib, "rms_norm_f32", 256)?,
            qk_norm: build_pipe(&device, &lib, "qk_norm_f32", 32)?,
            rope_kv: build_pipe(&device, &lib, "rope_and_kv_store_f16", 64)?,
            gqa_attn: build_pipe(&device, &lib, "gqa_attn_f16_kv", 128)?,
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res", 128)?,
            gemv_qkv_q4_0: build_pipe(&device, &lib, "gemv_qkv_q4_0", 128)?,
            gemv_pair_silu_q4_0: build_pipe(&device, &lib, "gemv_pair_silu_q4_0", 128)?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_res", 128)?,
            gemv_q5_k: build_pipe(&device, &lib, "gemv_q5_k_res", 128)?,
            gemv_q6_k: build_pipe(&device, &lib, "gemv_q6_k_res", 128)?,
            gemv_bf16: build_pipe(&device, &lib, "gemv_bf16_res", 128)?,
            gemv_f32: build_pipe(&device, &lib, "gemv_f32_res", 128)?,
            argmax: build_pipe(&device, &lib, "argmax_f32", 256)?,
            embed_q4_0_batch: build_pipe(&device, &lib, "embed_q4_0_batch", 64)?,
            embed_q8_0_batch: build_pipe(&device, &lib, "embed_q8_0_batch", 64)?,
            embed_q6_k_batch: build_pipe(&device, &lib, "embed_q6_k_batch", 64)?,
            embed_f32_batch: build_pipe(&device, &lib, "embed_f32_batch", 64)?,
            rms_norm_batch: build_pipe(&device, &lib, "rms_norm_f32_batch", 256)?,
            qk_norm_batch: build_pipe(&device, &lib, "qk_norm_f32_batch", 32)?,
            rope_kv_batch: build_pipe(&device, &lib, "rope_and_kv_store_f16_batch", 64)?,
            gqa_attn_batch: build_pipe(&device, &lib, "gqa_attn_f16_kv_batch", 128)?,
            fa_gqa_batch: build_pipe(&device, &lib, "fa_gqa_f16_kv_batch", 128)?,
            fa_gqa_batch_hd256: build_pipe(&device, &lib, "fa_gqa_f16_kv_batch_hd256", 128)?,
            gemm_q4_0: build_pipe(&device, &lib, "gemm_q4_0_res", 128)?,
            silu_mul: build_pipe(&device, &lib, "silu_mul_f32", 256)?,
            gemm_q4_1: build_pipe(&device, &lib, "gemm_q4_1_res", 128)?,
            gemm_q8_0: build_pipe(&device, &lib, "gemm_q8_0_res", 128)?,
            gemm_q5_k: build_pipe(&device, &lib, "gemm_q5_k_res", 128)?,
            gemm_q6_k: build_pipe(&device, &lib, "gemm_q6_k_res", 128)?,
            gemm_bf16: build_pipe(&device, &lib, "gemm_bf16_res", 128)?,
            gemm_f32: build_pipe(&device, &lib, "gemm_f32_res", 128)?,
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
        let output_norm =
            load_gpu_f32_vec(&device, gguf_ref, "output_norm.weight", "global", n_embd)?;
        let (has_rope_freqs, rope_freqs) = if gguf_ref.tensor_info("rope_freqs.weight").is_some() {
            (
                true,
                load_gpu_f32_vec(
                    &device,
                    gguf_ref,
                    "rope_freqs.weight",
                    "global",
                    rope_dim / 2,
                )?,
            )
        } else {
            (false, upload_f32(&device, &[1.0f32])?)
        };

        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let kv_bytes = max_ctx * kv_dim * core::mem::size_of::<u16>();

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let qn_name = format!("{ltag}.attn_q_norm.weight");
            let kn_name = format!("{ltag}.attn_k_norm.weight");
            let q_norm = if gguf_ref.tensor_info(&qn_name).is_some() {
                Some(load_gpu_f32_vec(
                    &device, gguf_ref, &qn_name, &ltag, head_dim,
                )?)
            } else {
                None
            };
            let k_norm = if gguf_ref.tensor_info(&kn_name).is_some() {
                Some(load_gpu_f32_vec(
                    &device, gguf_ref, &kn_name, &ltag, head_dim,
                )?)
            } else {
                None
            };

            layers.push(GpuLayer {
                attn_norm: load_gpu_f32_vec(
                    &device,
                    gguf_ref,
                    &format!("{ltag}.attn_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                ffn_norm: load_gpu_f32_vec(
                    &device,
                    gguf_ref,
                    &format!("{ltag}.ffn_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                q_norm,
                k_norm,
                wq: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.attn_q.weight"),
                    &ltag,
                    n_embd,
                    q_dim,
                )?,
                wk: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wv: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.attn_v.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wo: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                )?,
                w_gate: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.ffn_gate.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_up: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.ffn_up.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_down: load_gpu_mat(
                    &device,
                    gguf_ref,
                    sw,
                    &format!("{ltag}.ffn_down.weight"),
                    &ltag,
                    n_ff,
                    n_embd,
                )?,
                k_cache: alloc_buf(&device, kv_bytes)?,
                v_cache: alloc_buf(&device, kv_bytes)?,
            });
        }

        let f32_sz = core::mem::size_of::<f32>();
        let token_ids_buf = alloc_buf(&device, MAX_BATCH * core::mem::size_of::<u32>())?;
        let argmax_buf = alloc_buf(&device, 16)?;
        let x = alloc_buf(&device, MAX_BATCH * n_embd * f32_sz)?;
        let xb = alloc_buf(&device, MAX_BATCH * n_embd.max(q_dim) * f32_sz)?;
        let q = alloc_buf(&device, MAX_BATCH * q_dim * f32_sz)?;
        let k = alloc_buf(&device, MAX_BATCH * kv_dim * f32_sz)?;
        let v = alloc_buf(&device, MAX_BATCH * kv_dim * f32_sz)?;
        let hb = alloc_buf(&device, MAX_BATCH * n_ff * f32_sz)?;
        let hb2 = alloc_buf(&device, MAX_BATCH * n_ff * f32_sz)?;
        let logits = alloc_buf(&device, vocab_size * f32_sz)?;

        Ok(Self {
            cfg: LlamaMetalConfig {
                arch,
                n_layers,
                n_embd,
                n_ff,
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                vocab_size,
                max_ctx,
                rope_freq_base,
                rms_eps,
                rope_type,
            },
            device,
            queue,
            pipes,
            token_embd,
            output_tied,
            output_separate,
            output_norm,
            rope_freqs,
            has_rope_freqs,
            layers,
            token_ids_buf,
            argmax_buf,
            x,
            xb,
            q,
            k,
            v,
            hb,
            hb2,
            logits,
            lora,
            zero_copy_weights,
            _gguf: gguf,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Binds dynamic GGUF LoRA adapters onto this Metal runner (`Phase T4`).
    pub fn bind_lora(&mut self, loras: &tm_lora::LoraSet) -> Result<(), MetalError> {
        self.lora.bind(&self.device, &self._gguf, loras, MAX_BATCH)
    }

    /// Returns `true` if model weights are bound directly from the GGUF mmap via
    /// `newBufferWithBytesNoCopy` (zero host-to-GPU copy).
    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy_weights
    }

    fn encode_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        add_residual: bool,
    ) -> Result<(), MetalError> {
        self.encode_gemv_named(enc, mat, &mat.name, x_buf, y_buf, add_residual)
    }

    fn encode_gemv_named(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        lora_name: &str,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        add_residual: bool,
    ) -> Result<(), MetalError> {
        let (pipe, rows_per_tg): (&MtlPipe, usize) = match mat.dtype {
            DType::Q4_0 => (&self.pipes.gemv_q4_0, 32),
            DType::Q4_1 => (&self.pipes.gemv_q4_1, 32),
            DType::Q8_0 => (&self.pipes.gemv_q8_0, 16),
            DType::Q5K => (&self.pipes.gemv_q5_k, 8),
            DType::Q6K => (&self.pipes.gemv_q6_k, 16),
            DType::BF16 => (&self.pipes.gemv_bf16, 16),
            DType::F32 => (&self.pipes.gemv_f32, 8),
            other => return Err(MetalError::UnsupportedDType(other)),
        };

        const THREADS: usize = 128;
        let args = GemvArgs {
            n_in: mat.cols as u32,
            n_out: mat.rows as u32,
            add_residual: u32::from(add_residual),
            row_bytes: mat.row_bytes as u32,
        };

        enc.setComputePipelineState(pipe);
        // SAFETY: All buffers live on `self` for the full command buffer lifetime.
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
        self.lora
            .encode_linear(enc, lora_name, x_buf, 0, y_buf, 0, 1);
        Ok(())
    }

    fn encode_qkv_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        layer: &GpuLayer,
    ) -> Result<(), MetalError> {
        if layer.wq.dtype == DType::Q4_0
            && layer.wk.dtype == DType::Q4_0
            && layer.wv.dtype == DType::Q4_0
            && layer.wq.rows.is_multiple_of(32)
            && layer.wk.rows.is_multiple_of(32)
        {
            const THREADS: usize = 128;
            const ROWS_PER_TG: usize = 32;
            let total_rows = layer.wq.rows + layer.wk.rows + layer.wv.rows;
            let qargs = QkvGemvArgs {
                n_in: layer.wq.cols as u32,
                q_rows: layer.wq.rows as u32,
                k_rows: layer.wk.rows as u32,
                v_rows: layer.wv.rows as u32,
                row_bytes: layer.wq.row_bytes as u32,
            };
            enc.setComputePipelineState(&self.pipes.gemv_qkv_q4_0);
            // SAFETY: All buffers live on `self` for the full command buffer lifetime.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&layer.wq.buf), layer.wq.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&layer.wk.buf), layer.wk.offset, 1);
                enc.setBuffer_offset_atIndex(Some(&layer.wv.buf), layer.wv.offset, 2);
                enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&self.q), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&self.k), 0, 5);
                enc.setBuffer_offset_atIndex(Some(&self.v), 0, 6);
                enc.setBytes_length_atIndex(
                    NonNull::from(&qargs).cast::<c_void>(),
                    core::mem::size_of::<QkvGemvArgs>(),
                    7,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: total_rows.div_ceil(ROWS_PER_TG),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            self.lora
                .encode_linear(enc, &layer.wq.name, &self.xb, 0, &self.q, 0, 1);
            self.lora
                .encode_linear(enc, &layer.wk.name, &self.xb, 0, &self.k, 0, 1);
            self.lora
                .encode_linear(enc, &layer.wv.name, &self.xb, 0, &self.v, 0, 1);
            Ok(())
        } else {
            self.encode_gemv(enc, &layer.wq, &self.xb, &self.q, false)?;
            self.encode_gemv(enc, &layer.wk, &self.xb, &self.k, false)?;
            self.encode_gemv(enc, &layer.wv, &self.xb, &self.v, false)?;
            Ok(())
        }
    }

    fn encode_decode_attn(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        layer: &GpuLayer,
        seq_len: usize,
    ) {
        let aargs = AttnArgs {
            n_head: self.cfg.n_head as u32,
            n_head_kv: self.cfg.n_head_kv as u32,
            head_dim: self.cfg.head_dim as u32,
            seq_len: seq_len as u32,
        };
        enc.setComputePipelineState(&self.pipes.gqa_attn);
        // SAFETY: Buffers remain alive until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::from(&aargs).cast::<c_void>(),
                core::mem::size_of::<AttnArgs>(),
                4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.n_head,
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

    fn encode_prefill_attn(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        layer: &GpuLayer,
        chunk_pos: usize,
        b: usize,
    ) {
        const TG: usize = 128;
        if self.cfg.head_dim <= 256 && self.cfg.head_dim.is_multiple_of(8) {
            const BQ: usize = 32;
            let fa = FlashAttnArgs {
                n_head: self.cfg.n_head as u32,
                n_head_kv: self.cfg.n_head_kv as u32,
                head_dim: self.cfg.head_dim as u32,
                seq_len: (chunk_pos + 1) as u32,
                n_tok: b as u32,
            };
            let fa_pipe = if self.cfg.head_dim <= 128 {
                &self.pipes.fa_gqa_batch
            } else {
                &self.pipes.fa_gqa_batch_hd256
            };
            enc.setComputePipelineState(fa_pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&fa).cast::<c_void>(),
                    core::mem::size_of::<FlashAttnArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.cfg.n_head,
                    height: b.div_ceil(BQ),
                    depth: 1,
                },
                MTLSize {
                    width: TG,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            let aargs = AttnArgs {
                n_head: self.cfg.n_head as u32,
                n_head_kv: self.cfg.n_head_kv as u32,
                head_dim: self.cfg.head_dim as u32,
                seq_len: (chunk_pos + 1) as u32,
            };
            enc.setComputePipelineState(&self.pipes.gqa_attn_batch);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&aargs).cast::<c_void>(),
                    core::mem::size_of::<AttnArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.cfg.n_head,
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
    }

    fn encode_gemm(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        n_tok: usize,
        add_residual: bool,
    ) -> Result<(), MetalError> {
        let (pipe, rows_per_tg, toks_per_tg): (&MtlPipe, usize, usize) = match mat.dtype {
            DType::Q4_0 => (&self.pipes.gemm_q4_0, 32, 32),
            DType::Q4_1 => (&self.pipes.gemm_q4_1, 32, 32),
            DType::Q8_0 => (&self.pipes.gemm_q8_0, 4, 8),
            DType::Q5K => (&self.pipes.gemm_q5_k, 4, 8),
            DType::Q6K => (&self.pipes.gemm_q6_k, 4, 8),
            DType::BF16 => (&self.pipes.gemm_bf16, 32, 32),
            DType::F32 => (&self.pipes.gemm_f32, 4, 8),
            other => return Err(MetalError::UnsupportedDType(other)),
        };

        const THREADS: usize = 128;
        let args = GemmArgs {
            n_in: mat.cols as u32,
            n_out: mat.rows as u32,
            n_tok: n_tok as u32,
            add_residual: u32::from(add_residual),
            row_bytes: mat.row_bytes as u32,
        };

        enc.setComputePipelineState(pipe);
        // SAFETY: All buffers live on `self` for the full command buffer lifetime.
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
        self.lora
            .encode_linear(enc, &mat.name, x_buf, 0, y_buf, 0, n_tok);
        Ok(())
    }

    fn encode_rms_norm_at(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        x_byte_offset: usize,
        w_buf: &ProtocolObject<dyn MTLBuffer>,
        out_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
    ) -> Result<(), MetalError> {
        const THREADS: usize = 256;
        let args = NormArgs {
            n: n as u32,
            eps: self.cfg.rms_eps,
        };
        enc.setComputePipelineState(&self.pipes.rms_norm);
        // SAFETY: Buffers remain alive until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x_buf), x_byte_offset, 0);
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
                width: 1,
                height: 1,
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

    fn encode_rms_norm(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        w_buf: &ProtocolObject<dyn MTLBuffer>,
        out_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
    ) -> Result<(), MetalError> {
        self.encode_rms_norm_at(enc, x_buf, 0, w_buf, out_buf, n)
    }

    fn encode_rms_norm_batch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        w_buf: &ProtocolObject<dyn MTLBuffer>,
        out_buf: &ProtocolObject<dyn MTLBuffer>,
        n: usize,
        n_tok: usize,
    ) -> Result<(), MetalError> {
        const THREADS: usize = 256;
        let args = NormArgs {
            n: n as u32,
            eps: self.cfg.rms_eps,
        };
        enc.setComputePipelineState(&self.pipes.rms_norm_batch);
        // SAFETY: Buffers remain alive until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
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
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Executes a single-token forward pass completely on Metal GPU in ONE command buffer.
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

        if let Err(e) = self.encode_single_token_into(&enc, token_id, pos, compute_logits) {
            enc.endEncoding();
            return Err(e);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ptr = self.logits.contents().as_ptr() as *const f32;
        // SAFETY: `waitUntilCompleted` finished; `self.logits` is `StorageModeShared`
        // and holds `self.cfg.vocab_size` `f32` values.
        Ok(unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) })
    }

    /// Executes a single-token forward pass AND greedy argmax in ONE Metal command buffer,
    /// reading back only the 4-byte winning token ID instead of scanning `vocab_size` floats on CPU.
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

        if let Err(e) = self.encode_single_token_into(&enc, token_id, pos, true) {
            enc.endEncoding();
            return Err(e);
        }

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
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ptr = self.argmax_buf.contents().as_ptr() as *const u32;
        // SAFETY: `waitUntilCompleted` finished; `argmax_buf` holds at least one `u32`.
        Ok(unsafe { *ptr })
    }

    /// Executes a batched prompt prefill on Metal GPU in chunks of up to `MAX_BATCH` tokens,
    /// encoding each chunk into a single `MTLCommandBuffer` and sharing weight reads across
    /// `BT = 8` tokens in registers.
    ///
    /// Every quantized GEMM preserves the exact per-token accumulation and reduction order of
    /// `forward_token`, guaranteeing bit-identical outputs regardless of batch size.
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

        let half_rope = self.cfg.rope_dim / 2;
        let rope_threads = (self.cfg.n_head + self.cfg.n_head_kv) * half_rope;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let rope_total = rope_threads.max(kv_dim);

        let total = token_ids.len();
        let mut offset = 0usize;
        while offset < total {
            let b = (total - offset).min(MAX_BATCH);
            let chunk = &token_ids[offset..offset + b];
            let chunk_pos = start_pos + offset;
            let is_last_chunk = offset + b == total;

            // Copy chunk token IDs into shared GPU buffer (<= 256 bytes).
            {
                let dst = self.token_ids_buf.contents().as_ptr() as *mut u32;
                // SAFETY: `token_ids_buf` is `StorageModeShared` with capacity `MAX_BATCH` u32s
                // and no GPU command buffer is currently in flight.
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

            // 1. Batched embedding lookup
            {
                let (pipe, name, n_items): (&MtlPipe, &'static str, usize) =
                    match self.token_embd.dtype {
                        DType::Q4_0 => (
                            &self.pipes.embed_q4_0_batch,
                            "embed_q4_0_batch",
                            self.cfg.n_embd / 32,
                        ),
                        DType::Q8_0 => (
                            &self.pipes.embed_q8_0_batch,
                            "embed_q8_0_batch",
                            self.cfg.n_embd / 32,
                        ),
                        DType::Q6K => (
                            &self.pipes.embed_q6_k_batch,
                            "embed_q6_k_batch",
                            self.cfg.n_embd / 256,
                        ),
                        DType::F32 => (
                            &self.pipes.embed_f32_batch,
                            "embed_f32_batch",
                            self.cfg.n_embd,
                        ),
                        other => return Err(MetalError::UnsupportedDType(other)),
                    };
                let tg_w = n_items.clamp(1, 64);
                require_fit(pipe, name, tg_w)?;
                let eargs = EmbedArgs {
                    token_id: b as u32,
                    n_embd: self.cfg.n_embd as u32,
                    row_bytes: self.token_embd.row_bytes as u32,
                };
                enc.setComputePipelineState(pipe);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(
                        Some(&self.token_embd.buf),
                        self.token_embd.offset,
                        0,
                    );
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
                self.lora.encode_embed_batch(
                    &enc,
                    "token_embd.weight",
                    &self.token_ids_buf,
                    &self.x,
                    b,
                );
            }

            // 2. Transformer layers for all `b` tokens in ONE command buffer
            for layer in &self.layers {
                self.encode_rms_norm_batch(
                    &enc,
                    &self.x,
                    &layer.attn_norm,
                    &self.xb,
                    self.cfg.n_embd,
                    b,
                )?;

                self.encode_gemm(&enc, &layer.wq, &self.xb, &self.q, b, false)?;
                self.encode_gemm(&enc, &layer.wk, &self.xb, &self.k, b, false)?;
                self.encode_gemm(&enc, &layer.wv, &self.xb, &self.v, b, false)?;

                if let Some(ref qn) = layer.q_norm {
                    require_fit(&self.pipes.qk_norm_batch, "qk_norm_f32_batch", 32)?;
                    let na = NormArgs {
                        n: self.cfg.head_dim as u32,
                        eps: self.cfg.rms_eps,
                    };
                    enc.setComputePipelineState(&self.pipes.qk_norm_batch);
                    // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(qn), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&na).cast::<c_void>(),
                            core::mem::size_of::<NormArgs>(),
                            2,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: self.cfg.n_head,
                            height: b,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                if let Some(ref kn) = layer.k_norm {
                    require_fit(&self.pipes.qk_norm_batch, "qk_norm_f32_batch", 32)?;
                    let na = NormArgs {
                        n: self.cfg.head_dim as u32,
                        eps: self.cfg.rms_eps,
                    };
                    enc.setComputePipelineState(&self.pipes.qk_norm_batch);
                    // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.k), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(kn), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&na).cast::<c_void>(),
                            core::mem::size_of::<NormArgs>(),
                            2,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: self.cfg.n_head_kv,
                            height: b,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                // Batched RoPE + F16 KV store
                {
                    const TG: usize = 64;
                    require_fit(&self.pipes.rope_kv_batch, "rope_and_kv_store_f16_batch", TG)?;
                    let rargs = RopeKvArgs {
                        n_head: self.cfg.n_head as u32,
                        n_head_kv: self.cfg.n_head_kv as u32,
                        head_dim: self.cfg.head_dim as u32,
                        rope_dim: self.cfg.rope_dim as u32,
                        pos: chunk_pos as u32,
                        rope_type: self.cfg.rope_type,
                        has_freq_factors: u32::from(self.has_rope_freqs),
                        freq_base: self.cfg.rope_freq_base,
                    };
                    enc.setComputePipelineState(&self.pipes.rope_kv_batch);
                    // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.k), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&self.v), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 4);
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

                // Batched causal FlashAttention (`simdgroup_mma` tiled)
                self.encode_prefill_attn(&enc, layer, chunk_pos, b);

                // Wo + residual add
                self.encode_gemm(&enc, &layer.wo, &self.xb, &self.x, b, true)?;

                // FFN RMSNorm
                self.encode_rms_norm_batch(
                    &enc,
                    &self.x,
                    &layer.ffn_norm,
                    &self.xb,
                    self.cfg.n_embd,
                    b,
                )?;

                // Batched Gate + Up via hardware simdgroup_float8x8 GEMM + SwiGLU
                self.encode_gemm(&enc, &layer.w_gate, &self.xb, &self.hb, b, false)?;
                self.encode_gemm(&enc, &layer.w_up, &self.xb, &self.hb2, b, false)?;
                {
                    const TG: usize = 256;
                    let total_ff = (b * self.cfg.n_ff) as u32;
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

                // W_down + residual add
                self.encode_gemm(&enc, &layer.w_down, &self.hb, &self.x, b, true)?;
            }

            // 3. Output norm + LM head only for the final token of the final chunk
            if is_last_chunk && compute_last_logits {
                let last_x_off = (b - 1) * self.cfg.n_embd * core::mem::size_of::<f32>();
                self.encode_rms_norm_at(
                    &enc,
                    &self.x,
                    last_x_off,
                    &self.output_norm,
                    &self.xb,
                    self.cfg.n_embd,
                )?;
                let out_mat = if self.output_tied {
                    &self.token_embd
                } else {
                    self.output_separate.as_ref().unwrap_or(&self.token_embd)
                };
                self.encode_gemv_named(
                    &enc,
                    out_mat,
                    "output.weight",
                    &self.xb,
                    &self.logits,
                    false,
                )?;
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            offset += b;
        }

        let ptr = self.logits.contents().as_ptr() as *const f32;
        // SAFETY: `waitUntilCompleted` finished; `self.logits` is `StorageModeShared`
        // and holds `self.cfg.vocab_size` `f32` values.
        Ok(unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) })
    }

    /// Runs a single-command-buffer token step and returns `(wall_sec, gpu_sec)`.
    fn measure_single_cb_token(
        &mut self,
        token_id: u32,
        pos: usize,
    ) -> Result<(f64, f64), MetalError> {
        let t_wall = std::time::Instant::now();
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        self.encode_single_token_into(&enc, token_id, pos, true)?;
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        let wall_s = t_wall.elapsed().as_secs_f64();
        let gpu_s = (cmd.GPUEndTime() - cmd.GPUStartTime()).max(0.0);
        Ok((wall_s, gpu_s))
    }

    fn encode_single_token_into(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), MetalError> {
        // 1. Embedding
        {
            let (pipe, n_items): (&MtlPipe, usize) = match self.token_embd.dtype {
                DType::Q4_0 => (&self.pipes.embed_q4_0, self.cfg.n_embd / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0, self.cfg.n_embd / 32),
                DType::Q6K => (&self.pipes.embed_q6_k, self.cfg.n_embd / 256),
                DType::F32 => (&self.pipes.embed_f32, self.cfg.n_embd),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            let tg_w = n_items.clamp(1, 64);
            let eargs = EmbedArgs {
                token_id,
                n_embd: self.cfg.n_embd as u32,
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
                token_id,
                &self.x,
            );
        }

        let half_rope = self.cfg.rope_dim / 2;
        let rope_threads = (self.cfg.n_head + self.cfg.n_head_kv) * half_rope;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let rope_total = rope_threads.max(kv_dim);

        for layer in &self.layers {
            self.encode_rms_norm(enc, &self.x, &layer.attn_norm, &self.xb, self.cfg.n_embd)?;
            self.encode_qkv_gemv(enc, layer)?;

            if let Some(ref qn) = layer.q_norm {
                let na = NormArgs {
                    n: self.cfg.head_dim as u32,
                    eps: self.cfg.rms_eps,
                };
                enc.setComputePipelineState(&self.pipes.qk_norm);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(qn), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&na).cast::<c_void>(),
                        core::mem::size_of::<NormArgs>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: self.cfg.n_head,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            if let Some(ref kn) = layer.k_norm {
                let na = NormArgs {
                    n: self.cfg.head_dim as u32,
                    eps: self.cfg.rms_eps,
                };
                enc.setComputePipelineState(&self.pipes.qk_norm);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.k), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(kn), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&na).cast::<c_void>(),
                        core::mem::size_of::<NormArgs>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: self.cfg.n_head_kv,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            {
                const TG: usize = 64;
                let rargs = RopeKvArgs {
                    n_head: self.cfg.n_head as u32,
                    n_head_kv: self.cfg.n_head_kv as u32,
                    head_dim: self.cfg.head_dim as u32,
                    rope_dim: self.cfg.rope_dim as u32,
                    pos: pos as u32,
                    rope_type: self.cfg.rope_type,
                    has_freq_factors: u32::from(self.has_rope_freqs),
                    freq_base: self.cfg.rope_freq_base,
                };
                enc.setComputePipelineState(&self.pipes.rope_kv);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.k), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.v), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 4);
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

            self.encode_decode_attn(enc, layer, pos + 1);

            self.encode_gemv(enc, &layer.wo, &self.xb, &self.x, true)?;
            self.encode_rms_norm(enc, &self.x, &layer.ffn_norm, &self.xb, self.cfg.n_embd)?;

            if layer.w_gate.dtype == DType::Q4_0
                && layer.w_up.dtype == DType::Q4_0
                && !self.lora.has(&layer.w_gate.name)
                && !self.lora.has(&layer.w_up.name)
            {
                const THREADS: usize = 128;
                const ROWS_PER_TG: usize = 16;
                let gargs = GemvArgs {
                    n_in: layer.w_gate.cols as u32,
                    n_out: layer.w_gate.rows as u32,
                    add_residual: 0,
                    row_bytes: layer.w_gate.row_bytes as u32,
                };
                enc.setComputePipelineState(&self.pipes.gemv_pair_silu_q4_0);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&layer.w_gate.buf), layer.w_gate.offset, 0);
                    enc.setBuffer_offset_atIndex(Some(&layer.w_up.buf), layer.w_up.offset, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.hb), 0, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&gargs).cast::<c_void>(),
                        core::mem::size_of::<GemvArgs>(),
                        4,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: layer.w_gate.rows.div_ceil(ROWS_PER_TG),
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
                self.encode_gemv(enc, &layer.w_gate, &self.xb, &self.hb, false)?;
                self.encode_gemv(enc, &layer.w_up, &self.xb, &self.hb2, false)?;
                const TG: usize = 256;
                let total_ff = self.cfg.n_ff as u32;
                require_fit(&self.pipes.silu_mul, "silu_mul", TG)?;
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

            self.encode_gemv(enc, &layer.w_down, &self.hb, &self.x, true)?;
        }

        if compute_logits {
            self.encode_rms_norm(enc, &self.x, &self.output_norm, &self.xb, self.cfg.n_embd)?;
            let out_mat = if self.output_tied {
                &self.token_embd
            } else {
                self.output_separate.as_ref().unwrap_or(&self.token_embd)
            };
            self.encode_gemv_named(enc, out_mat, "output.weight", &self.xb, &self.logits, false)?;
        }
        Ok(())
    }

    /// Profiles Metal GPU execution using hardware `GPUStartTime()` / `GPUEndTime()` timestamps
    /// for both single-token decode (`decode_runs` tokens) and batched prefill (`prefill_len` tokens).
    pub fn profile(
        &mut self,
        decode_runs: usize,
        prefill_len: usize,
    ) -> Result<MetalProfileReport, MetalError> {
        let pqueue = self
            .device
            .newCommandQueueWithMaxCommandBufferCount(1024)
            .ok_or(MetalError::CommandQueueFailed)?;

        // Warmup GPU clocks
        for i in 0..4 {
            let _ = self.measure_single_cb_token(1, i)?;
        }
        let warm_ids = [1u32; 32];
        let _ = self.forward_batch(&warm_ids, 0, false)?;

        // 1. Measure true 1-CommandBuffer-per-token decode (Wall vs GPU hardware time)
        let mut wall_sum = 0.0f64;
        let mut gpu_sum = 0.0f64;
        let n_dec = decode_runs.max(1);
        for step in 0..n_dec {
            let (w, g) = self.measure_single_cb_token(1, step)?;
            wall_sum += w;
            gpu_sum += g;
        }
        let decode_single_cb_wall_us = (wall_sum * 1e6) / (n_dec as f64);
        let decode_single_cb_gpu_us = (gpu_sum * 1e6) / (n_dec as f64);

        // 2. Per-stage hardware GPU timestamps for Decode (back-to-back pipelined CBs, drained every 4 layers)
        const N_DEC_STAGES: usize = 13;
        let dec_names: [&'static str; N_DEC_STAGES] = [
            "embed",
            "attn_norm",
            "attn_qkv",
            "attn_wk",
            "attn_wv",
            "qk_norm",
            "rope_kv",
            "gqa_attn",
            "attn_wo+res",
            "ffn_norm",
            "ffn_gate_up_silu",
            "ffn_down+res",
            "lm_head+norm",
        ];
        let mut dec_gpu_sec = [0.0f64; N_DEC_STAGES];
        let mut dec_bytes = [0u64; N_DEC_STAGES];
        let mut dec_flops = [0u64; N_DEC_STAGES];

        let n_embd = self.cfg.n_embd;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let half_rope = self.cfg.rope_dim / 2;
        let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope).max(kv_dim);
        let out_mat_bytes = if self.output_tied {
            self.token_embd.row_bytes * self.token_embd.rows
        } else {
            self.output_separate
                .as_ref()
                .map_or(0, |m| m.row_bytes * m.rows)
        };
        let out_mat_flops = (2 * n_embd * self.cfg.vocab_size) as u64;

        dec_bytes[0] = (self.token_embd.row_bytes + n_embd * 4) as u64;
        dec_bytes[1] = (self.cfg.n_layers * n_embd * 12) as u64;
        for l in &self.layers {
            dec_bytes[2] += (l.wq.row_bytes * l.wq.rows
                + l.wk.row_bytes * l.wk.rows
                + l.wv.row_bytes * l.wv.rows) as u64;
            dec_flops[2] += (2
                * (l.wq.rows * l.wq.cols + l.wk.rows * l.wk.cols + l.wv.rows * l.wv.cols))
                as u64;
            dec_bytes[8] += (l.wo.row_bytes * l.wo.rows) as u64;
            dec_flops[8] += (2 * l.wo.rows * l.wo.cols) as u64;
            dec_bytes[10] +=
                (l.w_gate.row_bytes * l.w_gate.rows + l.w_up.row_bytes * l.w_up.rows) as u64;
            dec_flops[10] += (4 * l.w_gate.rows * l.w_gate.cols) as u64;
            dec_bytes[11] += (l.w_down.row_bytes * l.w_down.rows) as u64;
            dec_flops[11] += (2 * l.w_down.rows * l.w_down.cols) as u64;
        }
        dec_bytes[9] = (self.cfg.n_layers * n_embd * 12) as u64;
        dec_bytes[12] = out_mat_bytes as u64;
        dec_flops[12] = out_mat_flops;

        let mut pending: Vec<(usize, Retained<ProtocolObject<dyn MTLCommandBuffer>>)> =
            Vec::with_capacity(64);
        let drain = |pending: &mut Vec<(usize, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
                     acc: &mut [f64; N_DEC_STAGES]| {
            if let Some((_, last)) = pending.last() {
                last.waitUntilCompleted();
            }
            for (idx, cb) in pending.drain(..) {
                acc[idx] += (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0);
            }
        };

        let mut kv_bytes_sum = 0u64;
        for step in 0..n_dec {
            let pos = step;
            kv_bytes_sum += (self.cfg.n_layers * (pos + 1) * kv_dim * 2 * 2) as u64;

            // Stage 0: embed
            {
                let cb = pqueue
                    .commandBuffer()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let (pipe, name, n_items): (&MtlPipe, &'static str, usize) =
                    match self.token_embd.dtype {
                        DType::Q4_0 => (&self.pipes.embed_q4_0, "embed_q4_0", n_embd / 32),
                        DType::Q8_0 => (&self.pipes.embed_q8_0, "embed_q8_0", n_embd / 32),
                        DType::Q6K => (&self.pipes.embed_q6_k, "embed_q6_k", n_embd / 256),
                        DType::F32 => (&self.pipes.embed_f32, "embed_f32", n_embd),
                        other => return Err(MetalError::UnsupportedDType(other)),
                    };
                let tg_w = n_items.clamp(1, 64);
                require_fit(pipe, name, tg_w)?;
                let eargs = EmbedArgs {
                    token_id: 1,
                    n_embd: n_embd as u32,
                    row_bytes: self.token_embd.row_bytes as u32,
                };
                enc.setComputePipelineState(pipe);
                // SAFETY: Buffers live on `self` until `drain` waits.
                unsafe {
                    enc.setBuffer_offset_atIndex(
                        Some(&self.token_embd.buf),
                        self.token_embd.offset,
                        0,
                    );
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
                enc.endEncoding();
                cb.commit();
                pending.push((0, cb));
            }

            for (li, layer) in self.layers.iter().enumerate() {
                // 1: attn_norm
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_rms_norm(&enc, &self.x, &layer.attn_norm, &self.xb, n_embd)?;
                    enc.endEncoding();
                    cb.commit();
                    pending.push((1, cb));
                }
                // 2: fused attn_qkv
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_qkv_gemv(&enc, layer)?;
                    enc.endEncoding();
                    cb.commit();
                    pending.push((2, cb));
                }
                // 5: qk_norm (optional)
                if layer.q_norm.is_some() || layer.k_norm.is_some() {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let na = NormArgs {
                        n: self.cfg.head_dim as u32,
                        eps: self.cfg.rms_eps,
                    };
                    if let Some(ref qn) = layer.q_norm {
                        enc.setComputePipelineState(&self.pipes.qk_norm);
                        // SAFETY: Buffers live until `drain`.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(qn), 0, 1);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&na).cast::<c_void>(),
                                core::mem::size_of::<NormArgs>(),
                                2,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: self.cfg.n_head,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 32,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    if let Some(ref kn) = layer.k_norm {
                        enc.setComputePipelineState(&self.pipes.qk_norm);
                        // SAFETY: Buffers live until `drain`.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&self.k), 0, 0);
                            enc.setBuffer_offset_atIndex(Some(kn), 0, 1);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&na).cast::<c_void>(),
                                core::mem::size_of::<NormArgs>(),
                                2,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: self.cfg.n_head_kv,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 32,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    enc.endEncoding();
                    cb.commit();
                    pending.push((5, cb));
                }
                // 6: rope_kv
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    const TG: usize = 64;
                    let rargs = RopeKvArgs {
                        n_head: self.cfg.n_head as u32,
                        n_head_kv: self.cfg.n_head_kv as u32,
                        head_dim: self.cfg.head_dim as u32,
                        rope_dim: self.cfg.rope_dim as u32,
                        pos: pos as u32,
                        rope_type: self.cfg.rope_type,
                        has_freq_factors: u32::from(self.has_rope_freqs),
                        freq_base: self.cfg.rope_freq_base,
                    };
                    enc.setComputePipelineState(&self.pipes.rope_kv);
                    // SAFETY: Buffers live until `drain`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.k), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&self.v), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 4);
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
                    cb.commit();
                    pending.push((6, cb));
                }
                // 7: gqa_attn
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_decode_attn(&enc, layer, pos + 1);
                    enc.endEncoding();
                    cb.commit();
                    pending.push((7, cb));
                }
                // 8: wo + res
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemv(&enc, &layer.wo, &self.xb, &self.x, true)?;
                    enc.endEncoding();
                    cb.commit();
                    pending.push((8, cb));
                }
                // 9: ffn_norm
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_rms_norm(&enc, &self.x, &layer.ffn_norm, &self.xb, n_embd)?;
                    enc.endEncoding();
                    cb.commit();
                    pending.push((9, cb));
                }
                // 10: ffn_gate_up_silu
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    const THREADS: usize = 128;
                    const ROWS_PER_TG: usize = 16;
                    let gargs = GemvArgs {
                        n_in: layer.w_gate.cols as u32,
                        n_out: layer.w_gate.rows as u32,
                        add_residual: 0,
                        row_bytes: layer.w_gate.row_bytes as u32,
                    };
                    enc.setComputePipelineState(&self.pipes.gemv_pair_silu_q4_0);
                    // SAFETY: Buffers live until `drain`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(
                            Some(&layer.w_gate.buf),
                            layer.w_gate.offset,
                            0,
                        );
                        enc.setBuffer_offset_atIndex(Some(&layer.w_up.buf), layer.w_up.offset, 1);
                        enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&self.hb), 0, 3);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&gargs).cast::<c_void>(),
                            core::mem::size_of::<GemvArgs>(),
                            4,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: layer.w_gate.rows.div_ceil(ROWS_PER_TG),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: THREADS,
                            height: 1,
                            depth: 1,
                        },
                    );
                    enc.endEncoding();
                    cb.commit();
                    pending.push((10, cb));
                }
                // 11: ffn_down + res
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemv(&enc, &layer.w_down, &self.hb, &self.x, true)?;
                    enc.endEncoding();
                    cb.commit();
                    pending.push((11, cb));
                }

                if (li + 1) % 4 == 0 {
                    drain(&mut pending, &mut dec_gpu_sec);
                }
            }

            // 12: output_norm + lm_head
            {
                let cb = pqueue
                    .commandBuffer()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                self.encode_rms_norm(&enc, &self.x, &self.output_norm, &self.xb, n_embd)?;
                let out_mat = if self.output_tied {
                    &self.token_embd
                } else {
                    self.output_separate.as_ref().unwrap_or(&self.token_embd)
                };
                self.encode_gemv_named(
                    &enc,
                    out_mat,
                    "output.weight",
                    &self.xb,
                    &self.logits,
                    false,
                )?;
                enc.endEncoding();
                cb.commit();
                pending.push((12, cb));
            }
            drain(&mut pending, &mut dec_gpu_sec);
        }
        dec_bytes[7] = kv_bytes_sum / (n_dec as u64);

        let decode_stages: Vec<MetalStageStat> = (0..N_DEC_STAGES)
            .filter(|&i| dec_gpu_sec[i] > 0.0)
            .map(|i| MetalStageStat {
                name: dec_names[i],
                gpu_us: (dec_gpu_sec[i] * 1e6) / (n_dec as f64),
                bytes: dec_bytes[i],
                flops: dec_flops[i],
            })
            .collect();

        // 3. Prefill end-to-end + per-stage hardware GPU timestamps for `prefill_len`
        let n_pp = prefill_len.clamp(32, self.cfg.max_ctx);
        let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
        let t_pp_wall = std::time::Instant::now();
        let _ = self.forward_batch(&pp_ids, 0, false)?;
        let prefill_wall_ms = t_pp_wall.elapsed().as_secs_f64() * 1e3;

        // Profile 1 chunk of `b = MAX_BATCH.min(n_pp)` (at `chunk_pos = n_pp - b` so attention sees realistic KV!)
        let b = MAX_BATCH.min(n_pp);
        let chunk_pos = n_pp - b;
        const N_PP_STAGES: usize = 12;
        let pp_names: [&'static str; N_PP_STAGES] = [
            "embed_batch",
            "attn_norm_batch",
            "wq_gemm",
            "wk_gemm",
            "wv_gemm",
            "rope_kv_batch",
            "gqa_attn_batch",
            "wo_gemm+res",
            "ffn_norm_batch",
            "w_gate_gemm",
            "w_up_gemm+silu",
            "w_down_gemm+res",
        ];
        let mut pp_gpu_sec = [0.0f64; N_PP_STAGES];
        let mut pp_flops = [0u64; N_PP_STAGES];
        let mut pp_bytes = [0u64; N_PP_STAGES];
        for l in &self.layers {
            pp_flops[2] += (2 * b * l.wq.rows * l.wq.cols) as u64;
            pp_bytes[2] += (l.wq.row_bytes * l.wq.rows) as u64;
            pp_flops[3] += (2 * b * l.wk.rows * l.wk.cols) as u64;
            pp_bytes[3] += (l.wk.row_bytes * l.wk.rows) as u64;
            pp_flops[4] += (2 * b * l.wv.rows * l.wv.cols) as u64;
            pp_bytes[4] += (l.wv.row_bytes * l.wv.rows) as u64;
            pp_flops[7] += (2 * b * l.wo.rows * l.wo.cols) as u64;
            pp_bytes[7] += (l.wo.row_bytes * l.wo.rows) as u64;
            pp_flops[9] += (2 * b * l.w_gate.rows * l.w_gate.cols) as u64;
            pp_bytes[9] += (l.w_gate.row_bytes * l.w_gate.rows) as u64;
            pp_flops[10] += (2 * b * l.w_up.rows * l.w_up.cols) as u64;
            pp_bytes[10] += (l.w_up.row_bytes * l.w_up.rows) as u64;
            pp_flops[11] += (2 * b * l.w_down.rows * l.w_down.cols) as u64;
            pp_bytes[11] += (l.w_down.row_bytes * l.w_down.rows) as u64;
        }
        // Attention FLOPs across `b` queries at `chunk_pos .. chunk_pos + b`
        let avg_seq = chunk_pos + b / 2;
        pp_flops[6] =
            (self.cfg.n_layers * b * self.cfg.n_head * avg_seq * self.cfg.head_dim * 4) as u64;
        pp_bytes[6] = (self.cfg.n_layers
            * b.div_ceil(32)
            * self.cfg.n_head
            * avg_seq
            * self.cfg.head_dim
            * 4) as u64;

        let mut pp_pending: Vec<(usize, Retained<ProtocolObject<dyn MTLCommandBuffer>>)> =
            Vec::with_capacity(64);
        let drain_pp =
            |pending: &mut Vec<(usize, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
             acc: &mut [f64; N_PP_STAGES]| {
                if let Some((_, last)) = pending.last() {
                    last.waitUntilCompleted();
                }
                for (idx, cb) in pending.drain(..) {
                    acc[idx] += (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0);
                }
            };

        let pp_iters = 4usize;
        for _ in 0..pp_iters {
            // 0: embed_batch
            {
                let cb = pqueue
                    .commandBuffer()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let (pipe, _, n_items): (&MtlPipe, &'static str, usize) =
                    match self.token_embd.dtype {
                        DType::Q4_0 => (
                            &self.pipes.embed_q4_0_batch,
                            "embed_q4_0_batch",
                            n_embd / 32,
                        ),
                        DType::Q8_0 => (
                            &self.pipes.embed_q8_0_batch,
                            "embed_q8_0_batch",
                            n_embd / 32,
                        ),
                        DType::Q6K => (
                            &self.pipes.embed_q6_k_batch,
                            "embed_q6_k_batch",
                            n_embd / 256,
                        ),
                        DType::F32 => (&self.pipes.embed_f32_batch, "embed_f32_batch", n_embd),
                        other => return Err(MetalError::UnsupportedDType(other)),
                    };
                let tg_w = n_items.clamp(1, 64);
                let eargs = EmbedArgs {
                    token_id: b as u32,
                    n_embd: n_embd as u32,
                    row_bytes: self.token_embd.row_bytes as u32,
                };
                enc.setComputePipelineState(pipe);
                // SAFETY: Buffers live until `drain_pp`.
                unsafe {
                    enc.setBuffer_offset_atIndex(
                        Some(&self.token_embd.buf),
                        self.token_embd.offset,
                        0,
                    );
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
                enc.endEncoding();
                cb.commit();
                pp_pending.push((0, cb));
            }

            for (li, layer) in self.layers.iter().enumerate() {
                // 1: attn_norm_batch
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_rms_norm_batch(
                        &enc,
                        &self.x,
                        &layer.attn_norm,
                        &self.xb,
                        n_embd,
                        b,
                    )?;
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((1, cb));
                }
                // 2, 3, 4: wq, wk, wv gemm
                for (st, mat, out_b) in [
                    (2usize, &layer.wq, &self.q),
                    (3usize, &layer.wk, &self.k),
                    (4usize, &layer.wv, &self.v),
                ] {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemm(&enc, mat, &self.xb, out_b, b, false)?;
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((st, cb));
                }
                // 5: rope_kv_batch
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    const TG: usize = 64;
                    let rargs = RopeKvArgs {
                        n_head: self.cfg.n_head as u32,
                        n_head_kv: self.cfg.n_head_kv as u32,
                        head_dim: self.cfg.head_dim as u32,
                        rope_dim: self.cfg.rope_dim as u32,
                        pos: chunk_pos as u32,
                        rope_type: self.cfg.rope_type,
                        has_freq_factors: u32::from(self.has_rope_freqs),
                        freq_base: self.cfg.rope_freq_base,
                    };
                    enc.setComputePipelineState(&self.pipes.rope_kv_batch);
                    // SAFETY: Buffers live until `drain_pp`.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.k), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&self.v), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&layer.k_cache), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(&layer.v_cache), 0, 4);
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
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((5, cb));
                }
                // 6: fa_gqa_batch (tiled FlashAttention)
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_prefill_attn(&enc, layer, chunk_pos, b);
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((6, cb));
                }
                // 7: wo_gemm + res
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemm(&enc, &layer.wo, &self.xb, &self.x, b, true)?;
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((7, cb));
                }
                // 8: ffn_norm_batch
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_rms_norm_batch(
                        &enc,
                        &self.x,
                        &layer.ffn_norm,
                        &self.xb,
                        n_embd,
                        b,
                    )?;
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((8, cb));
                }
                // 9: w_gate_gemm
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemm(&enc, &layer.w_gate, &self.xb, &self.hb, b, false)?;
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((9, cb));
                }
                // 10: w_up_gemm + silu_mul
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemm(&enc, &layer.w_up, &self.xb, &self.hb2, b, false)?;
                    const TG: usize = 256;
                    let total_ff = (b * self.cfg.n_ff) as u32;
                    enc.setComputePipelineState(&self.pipes.silu_mul);
                    // SAFETY: Buffers live until `drain_pp`.
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
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((10, cb));
                }
                // 11: w_down_gemm + res
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_gemm(&enc, &layer.w_down, &self.hb, &self.x, b, true)?;
                    enc.endEncoding();
                    cb.commit();
                    pp_pending.push((11, cb));
                }

                if (li + 1) % 4 == 0 {
                    drain_pp(&mut pp_pending, &mut pp_gpu_sec);
                }
            }
            drain_pp(&mut pp_pending, &mut pp_gpu_sec);
        }

        let prefill_chunk_stages: Vec<MetalStageStat> = (0..N_PP_STAGES)
            .filter(|&i| pp_gpu_sec[i] > 0.0)
            .map(|i| MetalStageStat {
                name: pp_names[i],
                gpu_us: (pp_gpu_sec[i] * 1e6) / (pp_iters as f64),
                bytes: pp_bytes[i],
                flops: pp_flops[i],
            })
            .collect();

        Ok(MetalProfileReport {
            decode_runs: n_dec,
            decode_single_cb_wall_us,
            decode_single_cb_gpu_us,
            decode_stages,
            prefill_tokens: n_pp,
            prefill_wall_ms,
            prefill_chunk_tokens: b,
            prefill_chunk_stages,
        })
    }
}

/// Hardware GPU timing and throughput metrics for one operator stage.
#[derive(Debug, Clone)]
pub struct MetalStageStat {
    pub name: &'static str,
    pub gpu_us: f64,
    pub bytes: u64,
    pub flops: u64,
}

/// Complete Metal decode + prefill hardware-timestamp profile.
#[derive(Debug, Clone)]
pub struct MetalProfileReport {
    pub decode_runs: usize,
    pub decode_single_cb_wall_us: f64,
    pub decode_single_cb_gpu_us: f64,
    pub decode_stages: Vec<MetalStageStat>,
    pub prefill_tokens: usize,
    pub prefill_wall_ms: f64,
    pub prefill_chunk_tokens: usize,
    pub prefill_chunk_stages: Vec<MetalStageStat>,
}
