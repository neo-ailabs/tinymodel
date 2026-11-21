//! End-to-end Apple Metal GPU runner for `gemma4` (Per-Layer Embeddings `PLE`,
//! `BF16` `per_layer_model_proj`, 4:1 Sliding-Window + Proportional-RoPE Full Attention,
//! cross-layer KV sharing `shared_kv_layers`, unit-weight V-Norm, GeGLU FFN, and softcapping).
//!
//! Encodes the entire 35-layer decode pass (including on-demand GPU 1-row PLE lookup,
//! BF16 model projection, SWA/Full attention, GeGLU, LM head, and softcapping/argmax)
//! into a **single `MTLCommandBuffer`** per decode token with zero host syncs and zero-copy
//! GGUF mmap weights (`newBufferWithBytesNoCopy`).

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
struct ScaleArgs {
    n: u32,
    val: f32,
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
struct RopeQOnlyArgs {
    n_head: u32,
    head_dim: u32,
    rope_dim: u32,
    start_pos: u32,
    rope_type: u32,
    has_freq_factors: u32,
    n_tok: u32,
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
struct QkvGemvArgs {
    n_in: u32,
    q_rows: u32,
    k_rows: u32,
    v_rows: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PleGateArgs {
    ple_dim: u32,
    ple_total: u32,
    layer_idx: u32,
    n_tok: u32,
}

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Hyperparameters for `Gemma4MetalRunner`.
#[derive(Debug, Clone)]
pub struct Gemma4MetalConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_layer_kv_from_start: usize,
    pub n_embd: usize,
    pub n_embd_per_layer: usize,
    pub n_ff_per_layer: Vec<usize>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim_full: usize,
    pub head_dim_swa: usize,
    pub rope_dim_full: usize,
    pub rope_dim_swa: usize,
    pub rope_freq_base_full: f32,
    pub rope_freq_base_swa: f32,
    pub n_swa: usize,
    pub is_swa: Vec<bool>,
    pub kv_source_layer: Vec<usize>,
    pub rms_eps: f32,
    pub final_logit_softcapping: Option<f32>,
    pub vocab_size: usize,
    pub max_ctx: usize,
}

impl Gemma4MetalConfig {
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
    pub fn max_head_dim(&self) -> usize {
        self.head_dim_full.max(self.head_dim_swa)
    }

    #[inline]
    pub fn max_n_ff(&self) -> usize {
        self.n_ff_per_layer.iter().copied().max().unwrap_or(0)
    }
}

struct GpuMat {
    buf: MtlBuf,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

struct GpuGemma4PleGlobal {
    tok_embd: GpuMat,
    model_proj: GpuMat,
    proj_norm_scaled: MtlBuf,
}

struct GpuGemma4PleLayer {
    inp_gate: GpuMat,
    proj: GpuMat,
    post_norm: MtlBuf,
}

struct GpuGemma4KvOwn {
    k_norm: MtlBuf,
    wk: GpuMat,
    wv: Option<GpuMat>,
    k_cache: MtlBuf,
    v_cache: MtlBuf,
    kv_bytes: usize,
}

struct GpuGemma4Layer {
    attn_norm: MtlBuf,
    q_norm: MtlBuf,
    wq: GpuMat,
    kv: Option<GpuGemma4KvOwn>,
    wo: GpuMat,
    post_attention_norm: MtlBuf,
    ffn_norm: MtlBuf,
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
    post_ffw_norm: MtlBuf,
    ple: Option<GpuGemma4PleLayer>,
    out_scale: Option<f32>,
}

struct Gemma4Pipelines {
    embed_q4_0: MtlPipe,
    embed_q8_0: MtlPipe,
    embed_q6_k: MtlPipe,
    embed_f32: MtlPipe,
    embed_q4_0_batch: MtlPipe,
    embed_q8_0_batch: MtlPipe,
    embed_q6_k_batch: MtlPipe,
    embed_f32_batch: MtlPipe,
    scale_inplace: MtlPipe,
    softcap_inplace: MtlPipe,
    softcap_and_argmax: MtlPipe,
    rms_norm: MtlPipe,
    rms_norm_batch: MtlPipe,
    rms_norm_add_batch: MtlPipe,
    ple_combine_proj: MtlPipe,
    ple_gate_gelu_mul: MtlPipe,
    qk_norm_batch: MtlPipe,
    qk_norm_unit_batch: MtlPipe,
    rope_kv: MtlPipe,
    rope_kv_batch: MtlPipe,
    rope_q_only_batch: MtlPipe,
    gqa_attn_ext_1sg: MtlPipe,
    gqa_attn_ext_batch: MtlPipe,
    gemv_q4_0: MtlPipe,
    gemv_qkv_q4_0: MtlPipe,
    gemv_q4_1: MtlPipe,
    gemv_q8_0: MtlPipe,
    gemv_q5_k: MtlPipe,
    gemv_q6_k: MtlPipe,
    gemv_bf16: MtlPipe,
    gemv_f32: MtlPipe,
    gemv_pair_geglu_q4_0: MtlPipe,
    gemm_q4_0: MtlPipe,
    gemm_q4_1: MtlPipe,
    gemm_q8_0: MtlPipe,
    gemm_q5_k: MtlPipe,
    gemm_q6_k: MtlPipe,
    gemm_bf16: MtlPipe,
    gemm_f32: MtlPipe,
    geglu_mul: MtlPipe,
    argmax: MtlPipe,
}

/// Complete Apple Metal GPU inference runner for `gemma4` models.
pub struct Gemma4MetalRunner {
    pub cfg: Gemma4MetalConfig,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Gemma4Pipelines,
    token_embd: GpuMat,
    output_tied: bool,
    output_separate: Option<GpuMat>,
    output_norm: MtlBuf,
    rope_freqs_full: Option<MtlBuf>,
    dummy_rope_freqs: MtlBuf,
    ple_global: Option<GpuGemma4PleGlobal>,
    layers: Vec<GpuGemma4Layer>,
    // Pre-allocated scratch buffers (zero heap/GPU allocation during inference)
    token_ids_buf: MtlBuf,
    argmax_buf: MtlBuf,
    x: MtlBuf,
    xb: MtlBuf,
    xb2: MtlBuf,
    q_buf: MtlBuf,
    k_buf: MtlBuf,
    v_buf: MtlBuf,
    attn_out: MtlBuf,
    hb: MtlBuf,
    hb2: MtlBuf,
    ple_inputs: MtlBuf,
    ple_proj_scratch: MtlBuf,
    ple_gate_scratch: MtlBuf,
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
    // `aligned_len` bytes. `Gemma4MetalRunner` stores `_gguf: Arc<GgufFile>` as its
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

impl Gemma4MetalRunner {
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, MetalError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "gemma4" {
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

        let pipes = Gemma4Pipelines {
            embed_q4_0: build_pipe(&device, &lib, "embed_q4_0", 64)?,
            embed_q8_0: build_pipe(&device, &lib, "embed_q8_0", 64)?,
            embed_q6_k: build_pipe(&device, &lib, "embed_q6_k", 64)?,
            embed_f32: build_pipe(&device, &lib, "embed_f32", 256)?,
            embed_q4_0_batch: build_pipe(&device, &lib, "embed_q4_0_batch", 64)?,
            embed_q8_0_batch: build_pipe(&device, &lib, "embed_q8_0_batch", 64)?,
            embed_q6_k_batch: build_pipe(&device, &lib, "embed_q6_k_batch", 64)?,
            embed_f32_batch: build_pipe(&device, &lib, "embed_f32_batch", 64)?,
            scale_inplace: build_pipe(&device, &lib, "scale_inplace_f32", 256)?,
            softcap_inplace: build_pipe(&device, &lib, "softcap_inplace_f32", 256)?,
            softcap_and_argmax: build_pipe(&device, &lib, "softcap_and_argmax_f32", 256)?,
            rms_norm: build_pipe(&device, &lib, "rms_norm_f32", 256)?,
            rms_norm_batch: build_pipe(&device, &lib, "rms_norm_f32_batch", 256)?,
            rms_norm_add_batch: build_pipe(&device, &lib, "rms_norm_add_f32_batch", 128)?,
            ple_combine_proj: build_pipe(&device, &lib, "ple_combine_proj_f32", 32)?,
            ple_gate_gelu_mul: build_pipe(&device, &lib, "ple_gate_gelu_mul_f32", 256)?,
            qk_norm_batch: build_pipe(&device, &lib, "qk_norm_f32_batch", 32)?,
            qk_norm_unit_batch: build_pipe(&device, &lib, "qk_norm_unit_f32_batch", 32)?,
            rope_kv: build_pipe(&device, &lib, "rope_and_kv_store_f16", 256)?,
            rope_kv_batch: build_pipe(&device, &lib, "rope_and_kv_store_f16_batch", 64)?,
            rope_q_only_batch: build_pipe(&device, &lib, "rope_q_only_f32_batch", 256)?,
            gqa_attn_ext_1sg: build_pipe(&device, &lib, "gqa_attn_f16_kv_ext_1sg", 32)?,
            gqa_attn_ext_batch: build_pipe(&device, &lib, "gqa_attn_f16_kv_ext_batch", 128)?,
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res", 128)?,
            gemv_qkv_q4_0: build_pipe(&device, &lib, "gemv_qkv_q4_0", 128)?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_fast", 128)?,
            gemv_q5_k: build_pipe(&device, &lib, "gemv_q5_k_res", 128)?,
            gemv_q6_k: build_pipe(&device, &lib, "gemv_q6_k_res", 128)?,
            gemv_bf16: build_pipe(&device, &lib, "gemv_bf16_res", 128)?,
            gemv_f32: build_pipe(&device, &lib, "gemv_f32_res", 128)?,
            gemv_pair_geglu_q4_0: build_pipe(&device, &lib, "gemv_pair_geglu_q4_0", 128)?,
            gemm_q4_0: build_pipe(&device, &lib, "gemm_q4_0_res", 128)?,
            gemm_q4_1: build_pipe(&device, &lib, "gemm_q4_1_res", 128)?,
            gemm_q8_0: build_pipe(&device, &lib, "gemm_q8_0_res", 128)?,
            gemm_q5_k: build_pipe(&device, &lib, "gemm_q5_k_res", 128)?,
            gemm_q6_k: build_pipe(&device, &lib, "gemm_q6_k_res", 128)?,
            gemm_bf16: build_pipe(&device, &lib, "gemm_bf16_res", 128)?,
            gemm_f32: build_pipe(&device, &lib, "gemm_f32_res", 128)?,
            geglu_mul: build_pipe(&device, &lib, "geglu_mul_f32", 256)?,
            argmax: build_pipe(&device, &lib, "argmax_f32", 256)?,
        };

        let n_layers = gguf.req_usize("gemma4.block_count")?;
        let n_embd = gguf.req_usize("gemma4.embedding_length")?;
        let n_embd_per_layer = gguf
            .get_meta("gemma4.embedding_length_per_layer_input")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let shared_kv_layers = gguf
            .get_meta("gemma4.attention.shared_kv_layers")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        if shared_kv_layers >= n_layers {
            return Err(MetalError::InvalidDimensions(format!(
                "shared_kv_layers ({shared_kv_layers}) >= n_layers ({n_layers})"
            )));
        }
        let n_layer_kv_from_start = n_layers - shared_kv_layers;

        let n_ff_per_layer = if let Some(arr) = gguf
            .get_meta("gemma4.feed_forward_length")
            .and_then(|v| v.as_usize_array())
        {
            arr
        } else {
            vec![gguf.req_usize("gemma4.feed_forward_length")?; n_layers]
        };

        let n_head = gguf.req_usize("gemma4.attention.head_count")?;
        let n_head_kv = gguf
            .get_meta("gemma4.attention.head_count_kv")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim_full = gguf
            .get_meta("gemma4.attention.key_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head);
        let head_dim_swa = gguf
            .get_meta("gemma4.attention.key_length_swa")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_full);
        if head_dim_full > 512 || head_dim_swa > 512 {
            return Err(MetalError::InvalidDimensions(format!(
                "Gemma4MetalRunner supports head_dim <= 512 (got full={head_dim_full}, swa={head_dim_swa})"
            )));
        }

        let rope_dim_full = gguf
            .get_meta("gemma4.rope.dimension_count")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_full);
        let rope_dim_swa = gguf
            .get_meta("gemma4.rope.dimension_count_swa")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim_swa);
        let rope_freq_base_full = gguf
            .get_meta("gemma4.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let rope_freq_base_swa = gguf
            .get_meta("gemma4.rope.freq_base_swa")
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);
        let n_swa = gguf
            .get_meta("gemma4.attention.sliding_window")
            .and_then(|v| v.as_usize())
            .unwrap_or(512);
        let is_swa = if let Some(arr) = gguf
            .get_meta("gemma4.attention.sliding_window_pattern")
            .and_then(|v| v.as_bool_array())
        {
            arr
        } else {
            (0..n_layers).map(|il| (il % 5) < 4).collect()
        };

        let mut kv_source_layer = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            if il < n_layer_kv_from_start {
                kv_source_layer.push(il);
            } else {
                let want_swa = is_swa[il];
                let src = (0..n_layer_kv_from_start)
                    .rev()
                    .find(|&j| is_swa[j] == want_swa)
                    .ok_or_else(|| {
                        MetalError::InvalidDimensions(format!(
                            "no KV source layer found for shared layer {il}"
                        ))
                    })?;
                kv_source_layer.push(src);
            }
        }

        let rms_eps = gguf.req_f32("gemma4.attention.layer_norm_rms_epsilon")?;
        let final_logit_softcapping = gguf
            .get_meta("gemma4.final_logit_softcapping")
            .and_then(|v| v.as_f32())
            .filter(|&v| v > 0.0);
        let file_ctx = gguf
            .get_meta("gemma4.context_length")
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

        let cfg = Gemma4MetalConfig {
            arch,
            n_layers,
            n_layer_kv_from_start,
            n_embd,
            n_embd_per_layer,
            n_ff_per_layer,
            n_head,
            n_head_kv,
            head_dim_full,
            head_dim_swa,
            rope_dim_full,
            rope_dim_swa,
            rope_freq_base_full,
            rope_freq_base_swa,
            n_swa,
            is_swa,
            kv_source_layer,
            rms_eps,
            final_logit_softcapping,
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
        let (output_tied, output_separate) = if gguf.tensor_info("output.weight").is_some() {
            (
                false,
                Some(load_gpu_mat(
                    &device,
                    &gguf,
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
            load_gpu_f32_tensor(&device, &gguf, "output_norm.weight", "global", &[n_embd])?;

        let rope_freqs_full = if gguf.tensor_info("rope_freqs.weight").is_some() {
            Some(load_gpu_f32_tensor(
                &device,
                &gguf,
                "rope_freqs.weight",
                "global",
                &[rope_dim_full / 2],
            )?)
        } else {
            None
        };
        let dummy_rope_freqs = upload_f32(&device, &[1.0f32; 256])?;

        let ple_total = n_layers * n_embd_per_layer;
        let ple_global = if n_embd_per_layer > 0 {
            let tok_embd = load_gpu_mat(
                &device,
                &gguf,
                sw,
                "per_layer_token_embd.weight",
                "global",
                ple_total,
                vocab_size,
            )?;
            let model_proj = load_gpu_mat(
                &device,
                &gguf,
                sw,
                "per_layer_model_proj.weight",
                "global",
                n_embd,
                ple_total,
            )?;
            let (pinfo, pbytes) =
                gguf.req_tensor("per_layer_proj_norm.weight", "global", &[n_embd_per_layer])?;
            let mut pnorm = vec![0.0f32; n_embd_per_layer];
            dequantize_row_bytes(pinfo.dtype, pbytes, &mut pnorm);
            let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
            for v in &mut pnorm {
                *v *= inv_sqrt2;
            }
            Some(GpuGemma4PleGlobal {
                tok_embd,
                model_proj,
                proj_norm_scaled: upload_f32(&device, &pnorm)?,
            })
        } else {
            None
        };

        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let ltag = format!("blk.{il}");
            let hd = cfg.head_dim(il);
            let q_dim = n_head * hd;
            let kv_dim = n_head_kv * hd;
            let n_ff = cfg.n_ff_per_layer[il];

            let attn_norm = load_gpu_f32_tensor(
                &device,
                &gguf,
                &format!("{ltag}.attn_norm.weight"),
                &ltag,
                &[n_embd],
            )?;
            let q_norm = load_gpu_f32_tensor(
                &device,
                &gguf,
                &format!("{ltag}.attn_q_norm.weight"),
                &ltag,
                &[hd],
            )?;
            let wq = load_gpu_mat(
                &device,
                &gguf,
                sw,
                &format!("{ltag}.attn_q.weight"),
                &ltag,
                n_embd,
                q_dim,
            )?;

            let kv = if il < n_layer_kv_from_start {
                let k_norm = load_gpu_f32_tensor(
                    &device,
                    &gguf,
                    &format!("{ltag}.attn_k_norm.weight"),
                    &ltag,
                    &[hd],
                )?;
                let wk = load_gpu_mat(
                    &device,
                    &gguf,
                    sw,
                    &format!("{ltag}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?;
                let v_name = format!("{ltag}.attn_v.weight");
                let wv = if gguf.tensor_info(&v_name).is_some() {
                    Some(load_gpu_mat(
                        &device, &gguf, sw, &v_name, &ltag, n_embd, kv_dim,
                    )?)
                } else {
                    None
                };
                let kv_bytes = max_ctx * kv_dim * 2;
                Some(GpuGemma4KvOwn {
                    k_norm,
                    wk,
                    wv,
                    k_cache: alloc_buf(&device, kv_bytes)?,
                    v_cache: alloc_buf(&device, kv_bytes)?,
                    kv_bytes,
                })
            } else {
                None
            };

            let wo = load_gpu_mat(
                &device,
                &gguf,
                sw,
                &format!("{ltag}.attn_output.weight"),
                &ltag,
                q_dim,
                n_embd,
            )?;
            let post_attention_norm = load_gpu_f32_tensor(
                &device,
                &gguf,
                &format!("{ltag}.post_attention_norm.weight"),
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
            let post_ffw_norm = load_gpu_f32_tensor(
                &device,
                &gguf,
                &format!("{ltag}.post_ffw_norm.weight"),
                &ltag,
                &[n_embd],
            )?;

            let ple = if n_embd_per_layer > 0 {
                Some(GpuGemma4PleLayer {
                    inp_gate: load_gpu_mat(
                        &device,
                        &gguf,
                        sw,
                        &format!("{ltag}.inp_gate.weight"),
                        &ltag,
                        n_embd,
                        n_embd_per_layer,
                    )?,
                    proj: load_gpu_mat(
                        &device,
                        &gguf,
                        sw,
                        &format!("{ltag}.proj.weight"),
                        &ltag,
                        n_embd_per_layer,
                        n_embd,
                    )?,
                    post_norm: load_gpu_f32_tensor(
                        &device,
                        &gguf,
                        &format!("{ltag}.post_norm.weight"),
                        &ltag,
                        &[n_embd],
                    )?,
                })
            } else {
                None
            };

            let scale_name = format!("{ltag}.layer_output_scale.weight");
            let out_scale = if gguf.tensor_info(&scale_name).is_some() {
                let (sinfo, sbytes) = gguf.req_tensor(&scale_name, &ltag, &[1])?;
                let mut sv = [1.0f32; 1];
                dequantize_row_bytes(sinfo.dtype, sbytes, &mut sv);
                Some(sv[0])
            } else {
                None
            };

            layers.push(GpuGemma4Layer {
                attn_norm,
                q_norm,
                wq,
                kv,
                wo,
                post_attention_norm,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
                post_ffw_norm,
                ple,
                out_scale,
            });
        }

        let max_hd = cfg.max_head_dim();
        let max_q_dim = n_head * max_hd;
        let max_kv_dim = n_head_kv * max_hd;
        let max_ff = cfg.max_n_ff();

        let token_ids_buf = alloc_buf(&device, MAX_BATCH * 4)?;
        let argmax_buf = alloc_buf(&device, 16)?;
        let x = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let xb = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let xb2 = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let q_buf = alloc_buf(&device, MAX_BATCH * max_q_dim * 4)?;
        let k_buf = alloc_buf(&device, MAX_BATCH * max_kv_dim * 4)?;
        let v_buf = alloc_buf(&device, MAX_BATCH * max_kv_dim * 4)?;
        let attn_out = alloc_buf(&device, MAX_BATCH * max_q_dim * 4)?;
        let hb = alloc_buf(&device, MAX_BATCH * max_ff * 4)?;
        let hb2 = alloc_buf(&device, MAX_BATCH * max_ff * 4)?;
        let ple_inputs = alloc_buf(&device, MAX_BATCH * ple_total.max(4) * 4)?;
        let ple_proj_scratch = alloc_buf(&device, MAX_BATCH * ple_total.max(4) * 4)?;
        let ple_gate_scratch = alloc_buf(&device, MAX_BATCH * n_embd_per_layer.max(4) * 4)?;
        let logits = alloc_buf(&device, vocab_size * 4)?;

        Ok(Self {
            cfg,
            device,
            queue,
            pipes,
            token_embd,
            output_tied,
            output_separate,
            output_norm,
            rope_freqs_full,
            dummy_rope_freqs,
            ple_global,
            layers,
            token_ids_buf,
            argmax_buf,
            x,
            xb,
            xb2,
            q_buf,
            k_buf,
            v_buf,
            attn_out,
            hb,
            hb2,
            ple_inputs,
            ple_proj_scratch,
            ple_gate_scratch,
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
            if let Some(ref kv) = layer.kv {
                // SAFETY: Exclusive `&mut self` guarantees no active GPU command buffer is touching KV buffers.
                unsafe {
                    core::ptr::write_bytes(
                        kv.k_cache.contents().as_ptr() as *mut u8,
                        0,
                        kv.kv_bytes,
                    );
                    core::ptr::write_bytes(
                        kv.v_cache.contents().as_ptr() as *mut u8,
                        0,
                        kv.kv_bytes,
                    );
                }
            }
        }
    }

    pub fn logits(&self) -> &[f32] {
        let ptr = self.logits.contents().as_ptr() as *const f32;
        // SAFETY: `self.logits` is `StorageModeShared` of size `vocab_size * 4` and GPU work has completed.
        unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) }
    }

    #[inline]
    fn output_mat(&self) -> &GpuMat {
        self.output_separate.as_ref().unwrap_or(&self.token_embd)
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &GpuMat,
        x: &MtlBuf,
        x_off: usize,
        y: &MtlBuf,
        y_off: usize,
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
        // SAFETY: Buffers belong to `self` and outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.buf), w.offset, 0);
            enc.setBuffer_offset_atIndex(Some(x), x_off, 1);
            enc.setBuffer_offset_atIndex(Some(y), y_off, 2);
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

    #[allow(clippy::too_many_arguments)]
    fn enc_gemm(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        w: &GpuMat,
        x: &MtlBuf,
        x_off: usize,
        y: &MtlBuf,
        y_off: usize,
        n_tok: usize,
        add_residual: bool,
    ) {
        if n_tok == 1 {
            self.enc_gemv(enc, w, x, x_off, y, y_off, add_residual);
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
        // SAFETY: Buffers belong to `self` and outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.buf), w.offset, 0);
            enc.setBuffer_offset_atIndex(Some(x), x_off, 1);
            enc.setBuffer_offset_atIndex(Some(y), y_off, 2);
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

    fn enc_embed_row(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        token_id: u32,
        dst: &MtlBuf,
    ) {
        let args = EmbedArgs {
            token_id,
            n_embd: mat.cols as u32,
            row_bytes: mat.row_bytes as u32,
        };
        let (pipe, units): (&MtlPipe, usize) = match mat.dtype {
            DType::Q4_0 => (&self.pipes.embed_q4_0, mat.cols / 32),
            DType::Q8_0 => (&self.pipes.embed_q8_0, mat.cols / 32),
            DType::Q6K => (&self.pipes.embed_q6_k, mat.cols / 256),
            DType::F32 => (&self.pipes.embed_f32, mat.cols),
            other => unreachable!("unsupported embed dtype: {other:?}"),
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
            enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
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
    }

    fn enc_embed_batch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        n_tok: usize,
        dst: &MtlBuf,
    ) {
        let args = EmbedArgs {
            token_id: 0,
            n_embd: mat.cols as u32,
            row_bytes: mat.row_bytes as u32,
        };
        let (pipe, units): (&MtlPipe, usize) = match mat.dtype {
            DType::Q4_0 => (&self.pipes.embed_q4_0_batch, mat.cols / 32),
            DType::Q8_0 => (&self.pipes.embed_q8_0_batch, mat.cols / 32),
            DType::Q6K => (&self.pipes.embed_q6_k_batch, mat.cols / 256),
            DType::F32 => (&self.pipes.embed_f32_batch, mat.cols),
            other => unreachable!("unsupported embed dtype: {other:?}"),
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
            enc.setBuffer_offset_atIndex(Some(&self.token_ids_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(dst), 0, 2);
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

    fn enc_scale(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        buf: &MtlBuf,
        off: usize,
        n: usize,
        val: f32,
    ) {
        let args = ScaleArgs { n: n as u32, val };
        enc.setComputePipelineState(&self.pipes.scale_inplace);
        // SAFETY: Buffer outlives command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(buf), off, 0);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<ScaleArgs>(),
                1,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(256),
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

    #[allow(clippy::too_many_arguments)]
    fn enc_rms_norm_batch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        x: &MtlBuf,
        x_off: usize,
        w: &MtlBuf,
        out: &MtlBuf,
        out_off: usize,
        n: usize,
        eps: f32,
        n_tok: usize,
    ) {
        let args = NormArgs { n: n as u32, eps };
        enc.setComputePipelineState(&self.pipes.rms_norm_batch);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x), x_off, 0);
            enc.setBuffer_offset_atIndex(Some(w), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out), out_off, 2);
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

    #[allow(clippy::too_many_arguments)]
    fn enc_rms_norm_add_batch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        dst_res: &MtlBuf,
        dst_off: usize,
        x: &MtlBuf,
        x_off: usize,
        w: &MtlBuf,
        n: usize,
        eps: f32,
        n_tok: usize,
    ) {
        let args = NormArgs { n: n as u32, eps };
        enc.setComputePipelineState(&self.pipes.rms_norm_add_batch);
        // SAFETY: Buffers outlive command buffer execution.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(dst_res), dst_off, 0);
            enc.setBuffer_offset_atIndex(Some(x), x_off, 1);
            enc.setBuffer_offset_atIndex(Some(w), 0, 2);
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
                width: 128,
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
        let ple_dim = self.cfg.n_embd_per_layer;
        let ple_total = self.cfg.n_layers * ple_dim;

        // 1. Input token embedding + sqrt(n_embd)
        if let Some(tid) = single_token_id {
            self.enc_embed_row(enc, &self.token_embd, tid, &self.x);
        } else {
            self.enc_embed_batch(enc, &self.token_embd, n_tok, &self.x);
        }
        self.enc_scale(enc, &self.x, 0, n_tok * n_embd, (n_embd as f32).sqrt());

        // 2. Global Per-Layer Embeddings (`PLE`) lookup + BF16 `per_layer_model_proj`
        if let Some(ref ple_g) = self.ple_global {
            if let Some(tid) = single_token_id {
                self.enc_embed_row(enc, &ple_g.tok_embd, tid, &self.ple_inputs);
            } else {
                self.enc_embed_batch(enc, &ple_g.tok_embd, n_tok, &self.ple_inputs);
            }
            let tok_ple_scale = (ple_dim as f32).sqrt() / 2.0f32.sqrt();
            self.enc_scale(enc, &self.ple_inputs, 0, n_tok * ple_total, tok_ple_scale);

            self.enc_gemm(
                enc,
                &ple_g.model_proj,
                &self.x,
                0,
                &self.ple_proj_scratch,
                0,
                n_tok,
                false,
            );
            let proj_args = NormArgs {
                n: ple_dim as u32,
                eps: self.cfg.rms_eps * (n_embd as f32),
            };
            enc.setComputePipelineState(&self.pipes.ple_combine_proj);
            // SAFETY: Buffers outlive command buffer execution.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.ple_inputs), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&self.ple_proj_scratch), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&ple_g.proj_norm_scaled), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&proj_args).cast::<c_void>(),
                    core::mem::size_of::<NormArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_tok * self.cfg.n_layers,
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

        // 3. Transformer blocks (`0..n_layers`), pruning non-output rows at `n_layer_kv_from_start`!
        let mut active_rows = n_tok;
        let mut x_byte_off = 0usize;
        let mut ple_byte_off = 0usize;
        let mut row_pos_offset = start_pos;

        for (il, layer) in self.layers.iter().enumerate() {
            if il == self.cfg.n_layer_kv_from_start
                && self.cfg.n_layer_kv_from_start < self.cfg.n_layers
            {
                if !compute_logits {
                    return;
                }
                if active_rows > 1 {
                    let last = n_tok - 1;
                    x_byte_off = last * n_embd * 4;
                    ple_byte_off = last * ple_total * 4;
                    row_pos_offset = start_pos + last;
                    active_rows = 1;
                }
            }

            let hd = self.cfg.head_dim(il);
            let rope_dim = self.cfg.rope_dim(il);
            let kv_dim = self.cfg.n_head_kv * hd;
            let is_swa = self.cfg.is_swa[il];
            let freq_base = if is_swa {
                self.cfg.rope_freq_base_swa
            } else {
                self.cfg.rope_freq_base_full
            };
            let (freq_buf, has_freq) = if !is_swa {
                if let Some(ref rf) = self.rope_freqs_full {
                    (rf, 1u32)
                } else {
                    (&self.dummy_rope_freqs, 0u32)
                }
            } else {
                (&self.dummy_rope_freqs, 0u32)
            };
            let window = if is_swa { self.cfg.n_swa as u32 } else { 0u32 };
            let n_ff = self.cfg.n_ff_per_layer[il];

            // Pre-attention RMSNorm -> `xb`
            self.enc_rms_norm_batch(
                enc,
                &self.x,
                x_byte_off,
                &layer.attn_norm,
                &self.xb,
                0,
                n_embd,
                self.cfg.rms_eps,
                active_rows,
            );

            // QKV projection (fused single-dispatch `gemv_qkv_q4_0` on decode when Q4_0!) + per-head `q_norm`
            let qn_args = NormArgs {
                n: hd as u32,
                eps: self.cfg.rms_eps,
            };
            let mut did_fused_qkv = false;
            if active_rows == 1
                && let Some(ref kv_own) = layer.kv
                && let Some(ref wv) = kv_own.wv
                && layer.wq.dtype == DType::Q4_0
                && kv_own.wk.dtype == DType::Q4_0
                && wv.dtype == DType::Q4_0
            {
                let qkv_args = QkvGemvArgs {
                    n_in: n_embd as u32,
                    q_rows: layer.wq.rows as u32,
                    k_rows: kv_own.wk.rows as u32,
                    v_rows: wv.rows as u32,
                    row_bytes: layer.wq.row_bytes as u32,
                };
                let total_qkv = layer.wq.rows + kv_own.wk.rows + wv.rows;
                enc.setComputePipelineState(&self.pipes.gemv_qkv_q4_0);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&layer.wq.buf), layer.wq.offset, 0);
                    enc.setBuffer_offset_atIndex(Some(&kv_own.wk.buf), kv_own.wk.offset, 1);
                    enc.setBuffer_offset_atIndex(Some(&wv.buf), wv.offset, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 6);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&qkv_args).cast::<c_void>(),
                        core::mem::size_of::<QkvGemvArgs>(),
                        7,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: total_qkv.div_ceil(32),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
                did_fused_qkv = true;
            }
            if !did_fused_qkv {
                self.enc_gemm(
                    enc,
                    &layer.wq,
                    &self.xb,
                    0,
                    &self.q_buf,
                    0,
                    active_rows,
                    false,
                );
            }

            enc.setComputePipelineState(&self.pipes.qk_norm_batch);
            // SAFETY: Buffers outlive command buffer execution.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&layer.q_norm), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::from(&qn_args).cast::<c_void>(),
                    core::mem::size_of::<NormArgs>(),
                    2,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.cfg.n_head,
                    height: active_rows,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );

            // Own KV vs Shared KV
            if let Some(ref kv_own) = layer.kv {
                if !did_fused_qkv {
                    self.enc_gemm(
                        enc,
                        &kv_own.wk,
                        &self.xb,
                        0,
                        &self.k_buf,
                        0,
                        active_rows,
                        false,
                    );
                    if let Some(ref wv) = kv_own.wv {
                        self.enc_gemm(enc, wv, &self.xb, 0, &self.v_buf, 0, active_rows, false);
                    } else {
                        // K=V mode: project/copy unrotated K into `v_buf` before RoPE mutates `k_buf`
                        self.enc_gemm(
                            enc,
                            &kv_own.wk,
                            &self.xb,
                            0,
                            &self.v_buf,
                            0,
                            active_rows,
                            false,
                        );
                    }
                }

                // K RMSNorm (weighted) + V RMSNorm (unit-weight `1.0`!)
                enc.setComputePipelineState(&self.pipes.qk_norm_batch);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&kv_own.k_norm), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&qn_args).cast::<c_void>(),
                        core::mem::size_of::<NormArgs>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: self.cfg.n_head_kv,
                        height: active_rows,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );

                enc.setComputePipelineState(&self.pipes.qk_norm_unit_batch);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&qn_args).cast::<c_void>(),
                        core::mem::size_of::<NormArgs>(),
                        1,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: active_rows * self.cfg.n_head_kv,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );

                // NeoX RoPE on Q & K + store K/V into F16 cache
                let rk_args = RopeKvArgs {
                    n_head: self.cfg.n_head as u32,
                    n_head_kv: self.cfg.n_head_kv as u32,
                    head_dim: hd as u32,
                    rope_dim: rope_dim as u32,
                    pos: row_pos_offset as u32,
                    rope_type: 2, // NeoX
                    has_freq_factors: has_freq,
                    freq_base,
                };
                let max_threads_x =
                    ((self.cfg.n_head + self.cfg.n_head_kv) * (rope_dim / 2)).max(kv_dim);
                if active_rows == 1 {
                    enc.setComputePipelineState(&self.pipes.rope_kv);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 1);
                        enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 2);
                        enc.setBuffer_offset_atIndex(Some(&kv_own.k_cache), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(&kv_own.v_cache), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(freq_buf), 0, 5);
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
                        enc.setBuffer_offset_atIndex(Some(&kv_own.k_cache), 0, 3);
                        enc.setBuffer_offset_atIndex(Some(&kv_own.v_cache), 0, 4);
                        enc.setBuffer_offset_atIndex(Some(freq_buf), 0, 5);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&rk_args).cast::<c_void>(),
                            core::mem::size_of::<RopeKvArgs>(),
                            6,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: max_threads_x.div_ceil(64),
                            height: active_rows,
                            depth: 1,
                        },
                        MTLSize {
                            width: 64,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
            } else {
                // KV-shared layer: Q-only NeoX RoPE
                let rq_args = RopeQOnlyArgs {
                    n_head: self.cfg.n_head as u32,
                    head_dim: hd as u32,
                    rope_dim: rope_dim as u32,
                    start_pos: row_pos_offset as u32,
                    rope_type: 2, // NeoX
                    has_freq_factors: has_freq,
                    n_tok: active_rows as u32,
                    freq_base,
                };
                let total_q = active_rows * self.cfg.n_head * (rope_dim / 2);
                enc.setComputePipelineState(&self.pipes.rope_q_only_batch);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(freq_buf), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&rq_args).cast::<c_void>(),
                        core::mem::size_of::<RopeQOnlyArgs>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: total_q.div_ceil(256),
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

            // GQA Attention (`attn_scale = 1.0`, SWA `window` or Full `0`, reading `kv_source_layer[il]`)
            let src_il = self.cfg.kv_source_layer[il];
            let src_kv = self.layers[src_il]
                .kv
                .as_ref()
                .expect("kv_source_layer always owns KV");
            let attn_args = AttnExtArgs {
                n_head: self.cfg.n_head as u32,
                n_head_kv: self.cfg.n_head_kv as u32,
                head_dim: hd as u32,
                start_pos: row_pos_offset as u32,
                n_tok: active_rows as u32,
                window,
                attn_scale: 1.0,
                _pad: 0,
            };
            let seq_len_now = row_pos_offset + 1;
            let eff_kv_len = if window > 0 {
                seq_len_now.min(window as usize)
            } else {
                seq_len_now
            };
            if active_rows == 1 && eff_kv_len <= 128 {
                enc.setComputePipelineState(&self.pipes.gqa_attn_ext_1sg);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&src_kv.k_cache), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&src_kv.v_cache), 0, 2);
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
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                enc.setComputePipelineState(&self.pipes.gqa_attn_ext_batch);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&src_kv.k_cache), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&src_kv.v_cache), 0, 2);
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
                        height: active_rows,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            // `wo` -> `xb2`, then fused `x += rms_norm(xb2, post_attention_norm)`
            self.enc_gemm(
                enc,
                &layer.wo,
                &self.attn_out,
                0,
                &self.xb2,
                0,
                active_rows,
                false,
            );
            self.enc_rms_norm_add_batch(
                enc,
                &self.x,
                x_byte_off,
                &self.xb2,
                0,
                &layer.post_attention_norm,
                n_embd,
                self.cfg.rms_eps,
                active_rows,
            );

            // FFN: `ffn_norm(x)` -> `xb`, fused `GeGLU(gate, up)` -> `hb`, `w_down` -> `xb2`, `x += post_ffw_norm(xb2)`
            self.enc_rms_norm_batch(
                enc,
                &self.x,
                x_byte_off,
                &layer.ffn_norm,
                &self.xb,
                0,
                n_embd,
                self.cfg.rms_eps,
                active_rows,
            );
            if active_rows == 1
                && layer.w_gate.dtype == DType::Q4_0
                && layer.w_up.dtype == DType::Q4_0
            {
                let g_args = GemvArgs {
                    n_in: n_embd as u32,
                    n_out: n_ff as u32,
                    add_residual: 0,
                    row_bytes: layer.w_gate.row_bytes as u32,
                };
                enc.setComputePipelineState(&self.pipes.gemv_pair_geglu_q4_0);
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
                self.enc_gemm(
                    enc,
                    &layer.w_gate,
                    &self.xb,
                    0,
                    &self.hb,
                    0,
                    active_rows,
                    false,
                );
                self.enc_gemm(
                    enc,
                    &layer.w_up,
                    &self.xb,
                    0,
                    &self.hb2,
                    0,
                    active_rows,
                    false,
                );
                let total_ff = (active_rows * n_ff) as u32;
                enc.setComputePipelineState(&self.pipes.geglu_mul);
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
                        width: (active_rows * n_ff).div_ceil(256),
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
            self.enc_gemm(
                enc,
                &layer.w_down,
                &self.hb,
                0,
                &self.xb2,
                0,
                active_rows,
                false,
            );
            self.enc_rms_norm_add_batch(
                enc,
                &self.x,
                x_byte_off,
                &self.xb2,
                0,
                &layer.post_ffw_norm,
                n_embd,
                self.cfg.rms_eps,
                active_rows,
            );

            // Per-Layer Embedding (`PLE`) residual block
            if let Some(ref ple_l) = layer.ple {
                self.enc_gemm(
                    enc,
                    &ple_l.inp_gate,
                    &self.x,
                    x_byte_off,
                    &self.ple_gate_scratch,
                    0,
                    active_rows,
                    false,
                );
                let pg_args = PleGateArgs {
                    ple_dim: ple_dim as u32,
                    ple_total: ple_total as u32,
                    layer_idx: il as u32,
                    n_tok: active_rows as u32,
                };
                enc.setComputePipelineState(&self.pipes.ple_gate_gelu_mul);
                // SAFETY: Buffers outlive command buffer execution.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.ple_gate_scratch), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.ple_inputs), ple_byte_off, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&pg_args).cast::<c_void>(),
                        core::mem::size_of::<PleGateArgs>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (active_rows * ple_dim).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                self.enc_gemm(
                    enc,
                    &ple_l.proj,
                    &self.ple_gate_scratch,
                    0,
                    &self.xb2,
                    0,
                    active_rows,
                    false,
                );
                self.enc_rms_norm_add_batch(
                    enc,
                    &self.x,
                    x_byte_off,
                    &self.xb2,
                    0,
                    &ple_l.post_norm,
                    n_embd,
                    self.cfg.rms_eps,
                    active_rows,
                );
            }

            if let Some(scale) = layer.out_scale {
                self.enc_scale(enc, &self.x, x_byte_off, active_rows * n_embd, scale);
            }
        }

        // 4. Final `output_norm` + LM head + optional `final_logit_softcapping` + optional GPU `argmax`
        if compute_logits {
            let last_x_off = x_byte_off + (active_rows - 1) * n_embd * 4;
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

            self.enc_gemv(enc, self.output_mat(), &self.xb, 0, &self.logits, 0, false);

            if let Some(cap) = self.cfg.final_logit_softcapping {
                let sc_args = ScaleArgs {
                    n: self.cfg.vocab_size as u32,
                    val: cap,
                };
                if greedy_argmax {
                    enc.setComputePipelineState(&self.pipes.softcap_and_argmax);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.logits), 0, 0);
                        enc.setBuffer_offset_atIndex(Some(&self.argmax_buf), 0, 1);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&sc_args).cast::<c_void>(),
                            core::mem::size_of::<ScaleArgs>(),
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
                } else {
                    enc.setComputePipelineState(&self.pipes.softcap_inplace);
                    // SAFETY: Buffers outlive command buffer execution.
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&self.logits), 0, 0);
                        enc.setBytes_length_atIndex(
                            NonNull::from(&sc_args).cast::<c_void>(),
                            core::mem::size_of::<ScaleArgs>(),
                            1,
                        );
                    }
                    enc.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: self.cfg.vocab_size.div_ceil(256),
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
            } else if greedy_argmax {
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
            // SAFETY: Exclusive `&mut self` and previous chunk waited until completed.
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
        if let Some(ref ple_g) = self.ple_global {
            bytes_per_tok +=
                ple_g.tok_embd.row_bytes + ple_g.model_proj.rows * ple_g.model_proj.row_bytes;
        }
        for (il, l) in self.layers.iter().enumerate() {
            bytes_per_tok += l.wq.rows * l.wq.row_bytes + l.wo.rows * l.wo.row_bytes;
            if let Some(ref kv) = l.kv {
                bytes_per_tok += kv.wk.rows * kv.wk.row_bytes;
                if let Some(ref wv) = kv.wv {
                    bytes_per_tok += wv.rows * wv.row_bytes;
                }
            }
            bytes_per_tok += l.w_gate.rows * l.w_gate.row_bytes
                + l.w_up.rows * l.w_up.row_bytes
                + l.w_down.rows * l.w_down.row_bytes;
            if let Some(ref p) = l.ple {
                bytes_per_tok +=
                    p.inp_gate.rows * p.inp_gate.row_bytes + p.proj.rows * p.proj.row_bytes;
            }
            let seq = (prefill_len + n_runs / 2).min(if self.cfg.is_swa[il] {
                self.cfg.n_swa
            } else {
                usize::MAX
            });
            bytes_per_tok += 2 * seq * self.cfg.n_head_kv * self.cfg.head_dim(il) * 2;
        }
        let out_m = self.output_mat();
        bytes_per_tok += out_m.rows * out_m.row_bytes;

        let _ = self.output_tied;
        Ok(MetalProfileReport {
            decode_runs: n_runs,
            decode_single_cb_wall_us: wall_us,
            decode_single_cb_gpu_us: wall_us,
            decode_stages: vec![MetalStageStat {
                name: "gemma4_fused_e2e_cmdbuf",
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
