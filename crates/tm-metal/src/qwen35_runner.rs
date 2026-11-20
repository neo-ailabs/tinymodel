//! End-to-end Apple Metal GPU runner for `qwen35` (hybrid Gated DeltaNet + Full Attention).
//!
//! Encodes the entire hybrid forward pass (18 GDN linear-attention blocks + 6 gated
//! `head_dim = 256` full-attention blocks + SwiGLU FFN + 248k-vocab Q6_K LM head)
//! into a **single `MTLCommandBuffer`** per decode token with zero intermediate host syncs,
//! zero-copy GGUF mmap weights (`newBufferWithBytesNoCopy`), and `simdgroup_mma` batched prefill.

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
struct PairGemvArgs {
    n_in: u32,
    a_rows: u32,
    b_rows: u32,
    row_bytes: u32,
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
struct SplitQgArgs {
    n_head: u32,
    head_dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnGatesArgs {
    n_v_heads: u32,
    n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnConvArgs {
    n_k_heads: u32,
    n_v_heads: u32,
    s_dim: u32,
    n_tok: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnRecurArgs {
    n_k_heads: u32,
    n_v_heads: u32,
    s_dim: u32,
    n_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GdnNormArgs {
    n_v_heads: u32,
    s_dim: u32,
    eps: f32,
}

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Hyperparameters for `Qwen35MetalRunner`.
#[derive(Debug, Clone)]
pub struct Qwen35MetalConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
}

impl Qwen35MetalConfig {
    #[inline]
    pub const fn conv_dim(&self) -> usize {
        2 * self.ssm_n_group * self.ssm_state_size + self.ssm_d_inner
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

enum GpuAttnLayer {
    FullAttn {
        q_norm: MtlBuf,
        k_norm: MtlBuf,
        wqg: GpuMat,
        wk: GpuMat,
        wv: GpuMat,
        wo: GpuMat,
        k_cache: MtlBuf,
        v_cache: MtlBuf,
    },
    Gdn {
        wqkv: GpuMat,
        wz: GpuMat,
        w_alpha: GpuMat,
        w_beta: GpuMat,
        conv1d: MtlBuf,
        ssm_a: MtlBuf,
        ssm_dt_bias: MtlBuf,
        ssm_norm: MtlBuf,
        w_out: GpuMat,
        /// Transposed recurrent state `[n_v_heads, S, S]` in `StorageModeShared` F32.
        state: MtlBuf,
        state_bytes: usize,
        /// Causal Conv1d sliding history `[conv_dim, 3]` in `StorageModeShared` F32.
        conv_hist: MtlBuf,
        conv_hist_bytes: usize,
    },
}

struct GpuQwen35Layer {
    attn_norm: MtlBuf,
    post_attn_norm: MtlBuf,
    attn: GpuAttnLayer,
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
}

struct Qwen35Pipelines {
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
    qk_norm: MtlPipe,
    qk_norm_batch: MtlPipe,
    rope_kv: MtlPipe,
    rope_kv_batch: MtlPipe,
    gqa_attn: MtlPipe,
    gqa_attn_batch: MtlPipe,
    fa_gqa_batch: MtlPipe,
    fa_gqa_batch_hd256: MtlPipe,
    split_qg_qnorm: MtlPipe,
    sigmoid_mul: MtlPipe,
    gdn_gates: MtlPipe,
    gdn_conv_silu_l2: MtlPipe,
    gdn_recurrence: MtlPipe,
    gdn_norm_gated_silu: MtlPipe,
    gemv_q4_0: MtlPipe,
    gemv_qkv_q4_0: MtlPipe,
    gemv_pair_q4_0: MtlPipe,
    gemv_pair_q8_0: MtlPipe,
    gemv_pair_silu_q4_0: MtlPipe,
    gemv_q4_1: MtlPipe,
    gemv_q8_0: MtlPipe,
    gemv_q5_k: MtlPipe,
    gemv_q6_k: MtlPipe,
    gemm_q4_0: MtlPipe,
    gemm_q4_1: MtlPipe,
    gemm_q8_0: MtlPipe,
    gemm_q5_k: MtlPipe,
    gemm_q6_k: MtlPipe,
    silu_mul: MtlPipe,
    argmax: MtlPipe,
}

/// Complete Apple Metal GPU inference runner for `qwen35` hybrid GDN models.
pub struct Qwen35MetalRunner {
    pub cfg: Qwen35MetalConfig,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: Qwen35Pipelines,
    token_embd: GpuMat,
    output_tied: bool,
    output_separate: Option<GpuMat>,
    output_norm: MtlBuf,
    dummy_rope_freqs: MtlBuf,
    layers: Vec<GpuQwen35Layer>,
    // Pre-allocated scratch buffers (sized for MAX_BATCH, zero heap/GPU allocation during inference)
    token_ids_buf: MtlBuf,
    argmax_buf: MtlBuf,
    x: MtlBuf,
    xb: MtlBuf,
    qkv_buf: MtlBuf,
    q_buf: MtlBuf,
    k_buf: MtlBuf,
    v_buf: MtlBuf,
    gate_buf: MtlBuf,
    attn_out: MtlBuf,
    alpha_buf: MtlBuf,
    beta_buf: MtlBuf,
    g_exp_buf: MtlBuf,
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
    // Zero-initialize stateful / scratch buffers deterministically.
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
    // `aligned_len` bytes. `Qwen35MetalRunner` stores `_gguf: Arc<GgufFile>` as its
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
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K | DType::F32 => {}
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

impl Qwen35MetalRunner {
    /// Loads a `qwen35` hybrid GGUF model onto the default Apple Metal GPU.
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, MetalError> {
        let gguf_ref: &GgufFile = &gguf;
        let arch = gguf_ref.req_str("general.architecture")?.to_string();
        if arch != "qwen35" {
            return Err(MetalError::UnsupportedArch(arch));
        }

        let pfx = "qwen35";
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
            .unwrap_or(256);
        let rope_dim = gguf_ref
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(64);
        let rope_freq_base = gguf_ref
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000_000.0);
        let rms_eps = gguf_ref
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let ssm_conv_kernel = gguf_ref.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_state_size = gguf_ref.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_group = gguf_ref.req_usize(&format!("{pfx}.ssm.group_count"))?;
        let ssm_dt_rank = gguf_ref.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_d_inner = gguf_ref.req_usize(&format!("{pfx}.ssm.inner_size"))?;

        if ssm_conv_kernel != 4
            || ssm_state_size != 128
            || ssm_d_inner != ssm_dt_rank * ssm_state_size
        {
            return Err(MetalError::InvalidDimensions(format!(
                "Qwen3.5 Metal runner currently expects conv_kernel=4, state_size=128 (got conv_kernel={ssm_conv_kernel}, state_size={ssm_state_size}, d_inner={ssm_d_inner})"
            )));
        }

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

        let cfg = Qwen35MetalConfig {
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
            ssm_conv_kernel,
            ssm_state_size,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            vocab_size,
            max_ctx,
        };

        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueFailed)?;

        let src = NSString::from_str(LLAMA_OPS_MSL);
        let lib = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let pipes = Qwen35Pipelines {
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
            qk_norm: build_pipe(&device, &lib, "qk_norm_f32", 32)?,
            qk_norm_batch: build_pipe(&device, &lib, "qk_norm_f32_batch", 32)?,
            rope_kv: build_pipe(&device, &lib, "rope_and_kv_store_f16", 64)?,
            rope_kv_batch: build_pipe(&device, &lib, "rope_and_kv_store_f16_batch", 64)?,
            gqa_attn: build_pipe(&device, &lib, "gqa_attn_f16_kv", 128)?,
            gqa_attn_batch: build_pipe(&device, &lib, "gqa_attn_f16_kv_batch", 128)?,
            fa_gqa_batch: build_pipe(&device, &lib, "fa_gqa_f16_kv_batch", 128)?,
            fa_gqa_batch_hd256: build_pipe(&device, &lib, "fa_gqa_f16_kv_batch_hd256", 128)?,
            split_qg_qnorm: build_pipe(&device, &lib, "qwen35_split_qg_qnorm_f32", 32)?,
            sigmoid_mul: build_pipe(&device, &lib, "sigmoid_mul_f32", 256)?,
            gdn_gates: build_pipe(&device, &lib, "gdn_gates_f32", 64)?,
            gdn_conv_silu_l2: build_pipe(&device, &lib, "gdn_conv_silu_l2_f32", 128)?,
            gdn_recurrence: build_pipe(&device, &lib, "gdn_recurrence_f32", 128)?,
            gdn_norm_gated_silu: build_pipe(&device, &lib, "gdn_norm_gated_silu_f32", 32)?,
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res", 128)?,
            gemv_qkv_q4_0: build_pipe(&device, &lib, "gemv_qkv_q4_0", 128)?,
            gemv_pair_q4_0: build_pipe(&device, &lib, "gemv_pair_q4_0", 128)?,
            gemv_pair_q8_0: build_pipe(&device, &lib, "gemv_pair_q8_0", 128)?,
            gemv_pair_silu_q4_0: build_pipe(&device, &lib, "gemv_pair_silu_q4_0", 128)?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_res", 128)?,
            gemv_q5_k: build_pipe(&device, &lib, "gemv_q5_k_res", 128)?,
            gemv_q6_k: build_pipe(&device, &lib, "gemv_q6_k_res", 128)?,
            gemm_q4_0: build_pipe(&device, &lib, "gemm_q4_0_res", 128)?,
            gemm_q4_1: build_pipe(&device, &lib, "gemm_q4_1_res", 128)?,
            gemm_q8_0: build_pipe(&device, &lib, "gemm_q8_0_res", 128)?,
            gemm_q5_k: build_pipe(&device, &lib, "gemm_q5_k_res", 128)?,
            gemm_q6_k: build_pipe(&device, &lib, "gemm_q6_k_res", 128)?,
            silu_mul: build_pipe(&device, &lib, "silu_mul_f32", 256)?,
            argmax: build_pipe(&device, &lib, "argmax_f32", 256)?,
        };

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
            load_gpu_f32_tensor(&device, gguf_ref, "output_norm.weight", "global", &[n_embd])?;
        let dummy_rope_freqs = alloc_buf(&device, 16)?;

        let q_dim = n_head * head_dim;
        let qg_dim = q_dim * 2;
        let kv_dim = n_head_kv * head_dim;
        let conv_dim = cfg.conv_dim();
        let k_gdn_dim = ssm_n_group * ssm_state_size;
        let v_gdn_dim = ssm_d_inner;
        let kv_cache_bytes = max_ctx * kv_dim * 2; // F16
        let state_bytes = ssm_dt_rank * ssm_state_size * ssm_state_size * 4;
        let conv_hist_bytes = conv_dim * (ssm_conv_kernel - 1) * 4;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm = load_gpu_f32_tensor(
                &device,
                gguf_ref,
                &format!("{ltag}.attn_norm.weight"),
                &ltag,
                &[n_embd],
            )?;
            let post_name = if gguf_ref
                .tensor_info(&format!("{ltag}.post_attention_norm.weight"))
                .is_some()
            {
                format!("{ltag}.post_attention_norm.weight")
            } else {
                format!("{ltag}.ffn_norm.weight")
            };
            let post_attn_norm =
                load_gpu_f32_tensor(&device, gguf_ref, &post_name, &ltag, &[n_embd])?;

            let is_gdn = gguf_ref
                .tensor_info(&format!("{ltag}.attn_qkv.weight"))
                .is_some();
            let attn = if is_gdn {
                GpuAttnLayer::Gdn {
                    wqkv: load_gpu_mat(
                        &device,
                        gguf_ref,
                        sw,
                        &format!("{ltag}.attn_qkv.weight"),
                        &ltag,
                        n_embd,
                        conv_dim,
                    )?,
                    wz: load_gpu_mat(
                        &device,
                        gguf_ref,
                        sw,
                        &format!("{ltag}.attn_gate.weight"),
                        &ltag,
                        n_embd,
                        v_gdn_dim,
                    )?,
                    w_alpha: load_gpu_mat(
                        &device,
                        gguf_ref,
                        sw,
                        &format!("{ltag}.ssm_alpha.weight"),
                        &ltag,
                        n_embd,
                        ssm_dt_rank,
                    )?,
                    w_beta: load_gpu_mat(
                        &device,
                        gguf_ref,
                        sw,
                        &format!("{ltag}.ssm_beta.weight"),
                        &ltag,
                        n_embd,
                        ssm_dt_rank,
                    )?,
                    conv1d: load_gpu_f32_tensor(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.ssm_conv1d.weight"),
                        &ltag,
                        &[ssm_conv_kernel, conv_dim],
                    )?,
                    ssm_a: load_gpu_f32_tensor(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.ssm_a"),
                        &ltag,
                        &[ssm_dt_rank],
                    )?,
                    ssm_dt_bias: load_gpu_f32_tensor(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.ssm_dt.bias"),
                        &ltag,
                        &[ssm_dt_rank],
                    )?,
                    ssm_norm: load_gpu_f32_tensor(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.ssm_norm.weight"),
                        &ltag,
                        &[ssm_state_size],
                    )?,
                    w_out: load_gpu_mat(
                        &device,
                        gguf_ref,
                        sw,
                        &format!("{ltag}.ssm_out.weight"),
                        &ltag,
                        v_gdn_dim,
                        n_embd,
                    )?,
                    state: alloc_buf(&device, state_bytes)?,
                    state_bytes,
                    conv_hist: alloc_buf(&device, conv_hist_bytes)?,
                    conv_hist_bytes,
                }
            } else {
                GpuAttnLayer::FullAttn {
                    q_norm: load_gpu_f32_tensor(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.attn_q_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    k_norm: load_gpu_f32_tensor(
                        &device,
                        gguf_ref,
                        &format!("{ltag}.attn_k_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    wqg: load_gpu_mat(
                        &device,
                        gguf_ref,
                        sw,
                        &format!("{ltag}.attn_q.weight"),
                        &ltag,
                        n_embd,
                        qg_dim,
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
                    k_cache: alloc_buf(&device, kv_cache_bytes)?,
                    v_cache: alloc_buf(&device, kv_cache_bytes)?,
                }
            };

            layers.push(GpuQwen35Layer {
                attn_norm,
                post_attn_norm,
                attn,
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
            });
        }

        let token_ids_buf = alloc_buf(&device, MAX_BATCH * 4)?;
        let argmax_buf = alloc_buf(&device, 16)?;
        let x = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let xb = alloc_buf(&device, MAX_BATCH * n_embd * 4)?;
        let qkv_buf = alloc_buf(&device, MAX_BATCH * qg_dim.max(conv_dim) * 4)?;
        let q_buf = alloc_buf(&device, MAX_BATCH * q_dim.max(k_gdn_dim) * 4)?;
        let k_buf = alloc_buf(&device, MAX_BATCH * kv_dim.max(k_gdn_dim) * 4)?;
        let v_buf = alloc_buf(&device, MAX_BATCH * kv_dim.max(v_gdn_dim) * 4)?;
        let gate_buf = alloc_buf(&device, MAX_BATCH * q_dim.max(v_gdn_dim) * 4)?;
        let attn_out = alloc_buf(&device, MAX_BATCH * q_dim.max(v_gdn_dim) * 4)?;
        let alpha_buf = alloc_buf(&device, MAX_BATCH * ssm_dt_rank * 4)?;
        let beta_buf = alloc_buf(&device, MAX_BATCH * ssm_dt_rank * 4)?;
        let g_exp_buf = alloc_buf(&device, MAX_BATCH * ssm_dt_rank * 4)?;
        let hb = alloc_buf(&device, MAX_BATCH * n_ff * 4)?;
        let hb2 = alloc_buf(&device, MAX_BATCH * n_ff * 4)?;
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
            dummy_rope_freqs,
            layers,
            token_ids_buf,
            argmax_buf,
            x,
            xb,
            qkv_buf,
            q_buf,
            k_buf,
            v_buf,
            gate_buf,
            attn_out,
            alpha_buf,
            beta_buf,
            g_exp_buf,
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

    pub const fn is_zero_copy_weights(&self) -> bool {
        self.zero_copy_weights
    }

    /// Resets all recurrent GDN states + Conv1d histories (and zeroes KV caches) in-place.
    pub fn reset_state(&mut self) {
        for layer in &self.layers {
            if let GpuAttnLayer::Gdn {
                state,
                state_bytes,
                conv_hist,
                conv_hist_bytes,
                ..
            } = &layer.attn
            {
                // SAFETY: `StorageModeShared` buffers are idle while holding `&mut self`.
                unsafe {
                    core::ptr::write_bytes(state.contents().as_ptr() as *mut u8, 0, *state_bytes);
                    core::ptr::write_bytes(
                        conv_hist.contents().as_ptr() as *mut u8,
                        0,
                        *conv_hist_bytes,
                    );
                }
            }
        }
    }

    pub fn logits(&self) -> &[f32] {
        // SAFETY: `self.logits` is `StorageModeShared` of `vocab_size * 4` bytes and idle on `&self`.
        unsafe {
            core::slice::from_raw_parts(
                self.logits.contents().as_ptr() as *const f32,
                self.cfg.vocab_size,
            )
        }
    }

    fn out_mat(&self) -> &GpuMat {
        if self.output_tied {
            &self.token_embd
        } else {
            self.output_separate.as_ref().unwrap_or(&self.token_embd)
        }
    }

    fn encode_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
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
            other => return Err(MetalError::UnsupportedDType(other)),
        };

        let args = GemvArgs {
            n_in: mat.cols as u32,
            n_out: mat.rows as u32,
            add_residual: u32::from(add_residual),
            row_bytes: mat.row_bytes as u32,
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers live on `self` for the full command buffer lifetime.
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
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
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
            other => return Err(MetalError::UnsupportedDType(other)),
        };

        let args = GemmArgs {
            n_in: mat.cols as u32,
            n_out: mat.rows as u32,
            n_tok: n_tok as u32,
            add_residual: u32::from(add_residual),
            row_bytes: mat.row_bytes as u32,
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers live on `self` for the full command buffer lifetime.
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
                width: 128,
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
        n_tok: usize,
    ) {
        let nargs = NormArgs {
            n: n as u32,
            eps: self.cfg.rms_eps,
        };
        if n_tok == 1 {
            enc.setComputePipelineState(&self.pipes.rms_norm);
        } else {
            enc.setComputePipelineState(&self.pipes.rms_norm_batch);
        }
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(w_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&nargs).cast::<c_void>(),
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
    fn encode_gdn_core(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        conv1d: &MtlBuf,
        ssm_a: &MtlBuf,
        ssm_dt_bias: &MtlBuf,
        ssm_norm: &MtlBuf,
        state: &MtlBuf,
        conv_hist: &MtlBuf,
        n_tok: usize,
    ) {
        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;

        // 1. Gates (`g_exp`, `beta`)
        let gargs = GdnGatesArgs {
            n_v_heads: hv as u32,
            n_tok: n_tok as u32,
        };
        let total_gates = n_tok * hv;
        let tg_gates = total_gates.clamp(1, 64);
        enc.setComputePipelineState(&self.pipes.gdn_gates);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.alpha_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.beta_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(ssm_dt_bias), 0, 2);
            enc.setBuffer_offset_atIndex(Some(ssm_a), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.g_exp_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.beta_buf), 0, 5);
            enc.setBytes_length_atIndex(
                NonNull::from(&gargs).cast::<c_void>(),
                core::mem::size_of::<GdnGatesArgs>(),
                6,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: total_gates.div_ceil(tg_gates),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg_gates,
                height: 1,
                depth: 1,
            },
        );

        // 2. Fused Causal Conv1d + SiLU + Split Q/K/V + per-head L2-Norm on Q & K ONLY
        let cargs = GdnConvArgs {
            n_k_heads: hk as u32,
            n_v_heads: hv as u32,
            s_dim: s_dim as u32,
            n_tok: n_tok as u32,
            eps: self.cfg.rms_eps,
        };
        enc.setComputePipelineState(&self.pipes.gdn_conv_silu_l2);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.qkv_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(conv1d), 0, 1);
            enc.setBuffer_offset_atIndex(Some(conv_hist), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 5);
            enc.setBytes_length_atIndex(
                NonNull::from(&cargs).cast::<c_void>(),
                core::mem::size_of::<GdnConvArgs>(),
                6,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 2 * hk + hv,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        // 3. Register-resident 64-TG GDN recurrence (`kh = vh % HK`, `1/sqrt(S)`)
        let rargs = GdnRecurArgs {
            n_k_heads: hk as u32,
            n_v_heads: hv as u32,
            s_dim: s_dim as u32,
            n_tok: n_tok as u32,
        };
        enc.setComputePipelineState(&self.pipes.gdn_recurrence);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.g_exp_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&self.beta_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(state), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 6);
            enc.setBytes_length_atIndex(
                NonNull::from(&rargs).cast::<c_void>(),
                core::mem::size_of::<GdnRecurArgs>(),
                7,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: s_dim / 32,
                height: hv,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        // 4. Per-head gated RMSNorm * SiLU(z)
        let nargs = GdnNormArgs {
            n_v_heads: hv as u32,
            s_dim: s_dim as u32,
            eps: self.cfg.rms_eps,
        };
        enc.setComputePipelineState(&self.pipes.gdn_norm_gated_silu);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 0);
            enc.setBuffer_offset_atIndex(Some(ssm_norm), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.gate_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&nargs).cast::<c_void>(),
                core::mem::size_of::<GdnNormArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: hv,
                height: n_tok,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_full_attn_core(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        q_norm: &MtlBuf,
        k_norm: &MtlBuf,
        k_cache: &MtlBuf,
        v_cache: &MtlBuf,
        start_pos: usize,
        n_tok: usize,
    ) {
        let hd = self.cfg.head_dim;
        let q_dim = self.cfg.n_head * hd;
        let kv_dim = self.cfg.n_head_kv * hd;
        let half_rope = self.cfg.rope_dim / 2;

        // 1. De-interleave `[Q_h, Gate_h]` + fused `q_norm`
        let sargs = SplitQgArgs {
            n_head: self.cfg.n_head as u32,
            head_dim: hd as u32,
            eps: self.cfg.rms_eps,
        };
        enc.setComputePipelineState(&self.pipes.split_qg_qnorm);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.qkv_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(q_norm), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.gate_buf), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::from(&sargs).cast::<c_void>(),
                core::mem::size_of::<SplitQgArgs>(),
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
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        // 2. Per-head K RMSNorm
        let kn_args = NormArgs {
            n: hd as u32,
            eps: self.cfg.rms_eps,
        };
        if n_tok == 1 {
            enc.setComputePipelineState(&self.pipes.qk_norm);
        } else {
            enc.setComputePipelineState(&self.pipes.qk_norm_batch);
        }
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(k_norm), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&kn_args).cast::<c_void>(),
                core::mem::size_of::<NormArgs>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.cfg.n_head_kv,
                height: n_tok,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        // 3. Partial NeoX RoPE (`rope_dim = 64`) + F16 KV cache store (including unrotated `64..256` tail!)
        let rargs = RopeKvArgs {
            n_head: self.cfg.n_head as u32,
            n_head_kv: self.cfg.n_head_kv as u32,
            head_dim: hd as u32,
            rope_dim: self.cfg.rope_dim as u32,
            pos: start_pos as u32,
            rope_type: 1, // Neox / M-RoPE on 1D text positions
            has_freq_factors: 0,
            freq_base: self.cfg.rope_freq_base,
        };
        let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope).max(kv_dim);
        let rope_tg = rope_total.clamp(1, 64);
        if n_tok == 1 {
            enc.setComputePipelineState(&self.pipes.rope_kv);
        } else {
            enc.setComputePipelineState(&self.pipes.rope_kv_batch);
        }
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(k_cache), 0, 3);
            enc.setBuffer_offset_atIndex(Some(v_cache), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&self.dummy_rope_freqs), 0, 5);
            enc.setBytes_length_atIndex(
                NonNull::from(&rargs).cast::<c_void>(),
                core::mem::size_of::<RopeKvArgs>(),
                6,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: rope_total.div_ceil(rope_tg),
                height: n_tok,
                depth: 1,
            },
            MTLSize {
                width: rope_tg,
                height: 1,
                depth: 1,
            },
        );

        // 4. GQA Attention (`gqa_attn_f16_kv` for decode, `fa_gqa_f16_kv_batch_hd256` for prefill!)
        if n_tok == 1 {
            let aargs = AttnArgs {
                n_head: self.cfg.n_head as u32,
                n_head_kv: self.cfg.n_head_kv as u32,
                head_dim: hd as u32,
                seq_len: (start_pos + 1) as u32,
            };
            enc.setComputePipelineState(&self.pipes.gqa_attn);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(k_cache), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v_cache), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 3);
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
        } else if hd <= 256 && hd.is_multiple_of(8) {
            let fa = FlashAttnArgs {
                n_head: self.cfg.n_head as u32,
                n_head_kv: self.cfg.n_head_kv as u32,
                head_dim: hd as u32,
                seq_len: (start_pos + 1) as u32,
                n_tok: n_tok as u32,
            };
            let fa_pipe = if hd <= 128 {
                &self.pipes.fa_gqa_batch
            } else {
                &self.pipes.fa_gqa_batch_hd256
            };
            enc.setComputePipelineState(fa_pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(k_cache), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v_cache), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&fa).cast::<c_void>(),
                    core::mem::size_of::<FlashAttnArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: self.cfg.n_head,
                    height: n_tok.div_ceil(32),
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            let aargs = AttnArgs {
                n_head: self.cfg.n_head as u32,
                n_head_kv: self.cfg.n_head_kv as u32,
                head_dim: hd as u32,
                seq_len: (start_pos + 1) as u32,
            };
            enc.setComputePipelineState(&self.pipes.gqa_attn_batch);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.q_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(k_cache), 0, 1);
                enc.setBuffer_offset_atIndex(Some(v_cache), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&aargs).cast::<c_void>(),
                    core::mem::size_of::<AttnArgs>(),
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
        }

        // 5. Full-attention output gate: `attn_out *= sigmoid(gate_buf)` (`qwen35.cpp:345`)
        let total_q = (n_tok * q_dim) as u32;
        enc.setComputePipelineState(&self.pipes.sigmoid_mul);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.attn_out), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.gate_buf), 0, 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&total_q).cast::<c_void>(),
                core::mem::size_of::<u32>(),
                2,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (total_q as usize).div_ceil(256),
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

    fn encode_decode_pass(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), MetalError> {
        let n_embd = self.cfg.n_embd;

        // 1. Token embedding
        let (epipe, n_items): (&MtlPipe, usize) = match self.token_embd.dtype {
            DType::Q4_0 => (&self.pipes.embed_q4_0, n_embd / 32),
            DType::Q8_0 => (&self.pipes.embed_q8_0, n_embd / 32),
            DType::Q6K => (&self.pipes.embed_q6_k, n_embd / 256),
            DType::F32 => (&self.pipes.embed_f32, n_embd),
            other => return Err(MetalError::UnsupportedDType(other)),
        };
        let eargs = EmbedArgs {
            token_id,
            n_embd: n_embd as u32,
            row_bytes: self.token_embd.row_bytes as u32,
        };
        let etg = n_items.clamp(1, 64);
        enc.setComputePipelineState(epipe);
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
                width: n_items.div_ceil(etg),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: etg,
                height: 1,
                depth: 1,
            },
        );

        // 2. Hybrid layers
        for layer in &self.layers {
            self.encode_rms_norm(enc, &self.x, &layer.attn_norm, &self.xb, n_embd, 1);

            match &layer.attn {
                GpuAttnLayer::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                    state,
                    conv_hist,
                    ..
                } => {
                    // Fused Q4_0 `wqkv + wz` in 1 dispatch when both are Q4_0
                    if wqkv.dtype == DType::Q4_0
                        && wz.dtype == DType::Q4_0
                        && wqkv.rows.is_multiple_of(32)
                    {
                        let pargs = PairGemvArgs {
                            n_in: n_embd as u32,
                            a_rows: wqkv.rows as u32,
                            b_rows: wz.rows as u32,
                            row_bytes: wqkv.row_bytes as u32,
                        };
                        enc.setComputePipelineState(&self.pipes.gemv_pair_q4_0);
                        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&wqkv.buf), wqkv.offset, 0);
                            enc.setBuffer_offset_atIndex(Some(&wz.buf), wz.offset, 1);
                            enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(&self.qkv_buf), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(&self.gate_buf), 0, 4);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&pargs).cast::<c_void>(),
                                core::mem::size_of::<PairGemvArgs>(),
                                5,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: (wqkv.rows + wz.rows).div_ceil(32),
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
                        self.encode_gemv(enc, wqkv, &self.xb, &self.qkv_buf, false)?;
                        self.encode_gemv(enc, wz, &self.xb, &self.gate_buf, false)?;
                    }

                    // Fused Q8_0 `w_alpha + w_beta` in 1 dispatch when both are Q8_0
                    if w_alpha.dtype == DType::Q8_0 && w_beta.dtype == DType::Q8_0 {
                        let pargs = PairGemvArgs {
                            n_in: n_embd as u32,
                            a_rows: w_alpha.rows as u32,
                            b_rows: w_beta.rows as u32,
                            row_bytes: w_alpha.row_bytes as u32,
                        };
                        enc.setComputePipelineState(&self.pipes.gemv_pair_q8_0);
                        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&w_alpha.buf), w_alpha.offset, 0);
                            enc.setBuffer_offset_atIndex(Some(&w_beta.buf), w_beta.offset, 1);
                            enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                            enc.setBuffer_offset_atIndex(Some(&self.alpha_buf), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(&self.beta_buf), 0, 4);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&pargs).cast::<c_void>(),
                                core::mem::size_of::<PairGemvArgs>(),
                                5,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: (w_alpha.rows + w_beta.rows).div_ceil(4),
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
                        self.encode_gemv(enc, w_alpha, &self.xb, &self.alpha_buf, false)?;
                        self.encode_gemv(enc, w_beta, &self.xb, &self.beta_buf, false)?;
                    }

                    self.encode_gdn_core(
                        enc,
                        conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        state,
                        conv_hist,
                        1,
                    );
                    self.encode_gemv(enc, w_out, &self.attn_out, &self.x, true)?;
                }
                GpuAttnLayer::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                    k_cache,
                    v_cache,
                } => {
                    if wqg.dtype == DType::Q4_0
                        && wk.dtype == DType::Q4_0
                        && wv.dtype == DType::Q4_0
                        && wqg.rows.is_multiple_of(32)
                        && wk.rows.is_multiple_of(32)
                    {
                        let qargs = QkvGemvArgs {
                            n_in: n_embd as u32,
                            q_rows: wqg.rows as u32,
                            k_rows: wk.rows as u32,
                            v_rows: wv.rows as u32,
                            row_bytes: wqg.row_bytes as u32,
                        };
                        enc.setComputePipelineState(&self.pipes.gemv_qkv_q4_0);
                        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                        unsafe {
                            enc.setBuffer_offset_atIndex(Some(&wqg.buf), wqg.offset, 0);
                            enc.setBuffer_offset_atIndex(Some(&wk.buf), wk.offset, 1);
                            enc.setBuffer_offset_atIndex(Some(&wv.buf), wv.offset, 2);
                            enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 3);
                            enc.setBuffer_offset_atIndex(Some(&self.qkv_buf), 0, 4);
                            enc.setBuffer_offset_atIndex(Some(&self.k_buf), 0, 5);
                            enc.setBuffer_offset_atIndex(Some(&self.v_buf), 0, 6);
                            enc.setBytes_length_atIndex(
                                NonNull::from(&qargs).cast::<c_void>(),
                                core::mem::size_of::<QkvGemvArgs>(),
                                7,
                            );
                        }
                        enc.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: (wqg.rows + wk.rows + wv.rows).div_ceil(32),
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
                        self.encode_gemv(enc, wqg, &self.xb, &self.qkv_buf, false)?;
                        self.encode_gemv(enc, wk, &self.xb, &self.k_buf, false)?;
                        self.encode_gemv(enc, wv, &self.xb, &self.v_buf, false)?;
                    }

                    self.encode_full_attn_core(enc, q_norm, k_norm, k_cache, v_cache, pos, 1);
                    self.encode_gemv(enc, wo, &self.attn_out, &self.x, true)?;
                }
            }

            // Post-attention norm + FFN
            self.encode_rms_norm(enc, &self.x, &layer.post_attn_norm, &self.xb, n_embd, 1);
            if layer.w_gate.dtype == DType::Q4_0 && layer.w_up.dtype == DType::Q4_0 {
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
                        width: layer.w_gate.rows.div_ceil(16),
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
                self.encode_gemv(enc, &layer.w_gate, &self.xb, &self.hb, false)?;
                self.encode_gemv(enc, &layer.w_up, &self.xb, &self.hb2, false)?;
                let n_ff_u32 = self.cfg.n_ff as u32;
                enc.setComputePipelineState(&self.pipes.silu_mul);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.hb), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.hb2), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&n_ff_u32).cast::<c_void>(),
                        core::mem::size_of::<u32>(),
                        2,
                    );
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: self.cfg.n_ff.div_ceil(256),
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
            self.encode_gemv(enc, &layer.w_down, &self.hb, &self.x, true)?;
        }

        if compute_logits {
            self.encode_rms_norm(enc, &self.x, &self.output_norm, &self.xb, n_embd, 1);
            self.encode_gemv(enc, self.out_mat(), &self.xb, &self.logits, false)?;
            if compute_argmax {
                let n_vocab = self.cfg.vocab_size as u32;
                enc.setComputePipelineState(&self.pipes.argmax);
                // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
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
        Ok(())
    }

    /// Single-token forward pass at `pos`, writing `self.logits` when `compute_logits` is true.
    ///
    /// Validates `pos` and `token_id` **before** encoding any GPU work (`AGENTS.md` Rule #6).
    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), MetalError> {
        if pos >= self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "position {pos} exceeds max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(MetalError::InvalidDimensions(format!(
                "token_id {token_id} out of vocab range {}",
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
        self.encode_decode_pass(&enc, token_id, pos, compute_logits, false)?;
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Single-token forward pass + GPU-side `argmax_f32` over Qwen3.5's 248k vocabulary,
    /// reading back only 4 bytes instead of 970 KB of FP32 logits per step!
    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, MetalError> {
        if pos >= self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "position {pos} exceeds max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(MetalError::InvalidDimensions(format!(
                "token_id {token_id} out of vocab range {}",
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
        self.encode_decode_pass(&enc, token_id, pos, true, true)?;
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // SAFETY: `argmax_buf` is `StorageModeShared` (>= 4 bytes) and `cmd` has completed.
        let next_id = unsafe { *(self.argmax_buf.contents().as_ptr() as *const u32) };
        Ok(next_id)
    }

    /// Batched prefill forward pass for `token_ids` starting at `start_pos`.
    ///
    /// Validates all bounds and token IDs **before** modifying GPU state (`AGENTS.md` Rule #6).
    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), MetalError> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let end_pos = start_pos.saturating_add(token_ids.len());
        if end_pos > self.cfg.max_ctx {
            return Err(MetalError::InvalidDimensions(format!(
                "batch end_pos {end_pos} exceeds max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        for &tok in token_ids {
            if (tok as usize) >= self.cfg.vocab_size {
                return Err(MetalError::InvalidDimensions(format!(
                    "token_id {tok} out of vocab range {}",
                    self.cfg.vocab_size
                )));
            }
        }

        if token_ids.len() == 1 {
            return self.forward_token(token_ids[0], start_pos, compute_last_logits);
        }

        let n_embd = self.cfg.n_embd;
        let mut offset = 0usize;
        while offset < token_ids.len() {
            let b = (token_ids.len() - offset).min(MAX_BATCH);
            let chunk = &token_ids[offset..offset + b];
            let chunk_pos = start_pos + offset;
            let is_last = offset + b == token_ids.len();

            // SAFETY: Previous command buffer completed; `token_ids_buf` holds `MAX_BATCH` u32s.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
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

            let (epipe, n_items): (&MtlPipe, usize) = match self.token_embd.dtype {
                DType::Q4_0 => (&self.pipes.embed_q4_0_batch, n_embd / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0_batch, n_embd / 32),
                DType::Q6K => (&self.pipes.embed_q6_k_batch, n_embd / 256),
                DType::F32 => (&self.pipes.embed_f32_batch, n_embd),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            let eargs = EmbedArgs {
                token_id: 0,
                n_embd: n_embd as u32,
                row_bytes: self.token_embd.row_bytes as u32,
            };
            let etg = n_items.clamp(1, 64);
            enc.setComputePipelineState(epipe);
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
                    width: n_items.div_ceil(etg),
                    height: b,
                    depth: 1,
                },
                MTLSize {
                    width: etg,
                    height: 1,
                    depth: 1,
                },
            );

            for layer in &self.layers {
                self.encode_rms_norm(&enc, &self.x, &layer.attn_norm, &self.xb, n_embd, b);

                match &layer.attn {
                    GpuAttnLayer::Gdn {
                        wqkv,
                        wz,
                        w_alpha,
                        w_beta,
                        conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        w_out,
                        state,
                        conv_hist,
                        ..
                    } => {
                        self.encode_gemm(&enc, wqkv, &self.xb, &self.qkv_buf, b, false)?;
                        self.encode_gemm(&enc, wz, &self.xb, &self.gate_buf, b, false)?;
                        self.encode_gemm(&enc, w_alpha, &self.xb, &self.alpha_buf, b, false)?;
                        self.encode_gemm(&enc, w_beta, &self.xb, &self.beta_buf, b, false)?;
                        self.encode_gdn_core(
                            &enc,
                            conv1d,
                            ssm_a,
                            ssm_dt_bias,
                            ssm_norm,
                            state,
                            conv_hist,
                            b,
                        );
                        self.encode_gemm(&enc, w_out, &self.attn_out, &self.x, b, true)?;
                    }
                    GpuAttnLayer::FullAttn {
                        q_norm,
                        k_norm,
                        wqg,
                        wk,
                        wv,
                        wo,
                        k_cache,
                        v_cache,
                    } => {
                        self.encode_gemm(&enc, wqg, &self.xb, &self.qkv_buf, b, false)?;
                        self.encode_gemm(&enc, wk, &self.xb, &self.k_buf, b, false)?;
                        self.encode_gemm(&enc, wv, &self.xb, &self.v_buf, b, false)?;
                        self.encode_full_attn_core(
                            &enc, q_norm, k_norm, k_cache, v_cache, chunk_pos, b,
                        );
                        self.encode_gemm(&enc, wo, &self.attn_out, &self.x, b, true)?;
                    }
                }

                self.encode_rms_norm(&enc, &self.x, &layer.post_attn_norm, &self.xb, n_embd, b);
                self.encode_gemm(&enc, &layer.w_gate, &self.xb, &self.hb, b, false)?;
                self.encode_gemm(&enc, &layer.w_up, &self.xb, &self.hb2, b, false)?;

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
                        width: (total_ff as usize).div_ceil(256),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );

                self.encode_gemm(&enc, &layer.w_down, &self.hb, &self.x, b, true)?;
            }

            if is_last && compute_last_logits {
                let last_byte_off = (b - 1) * n_embd * 4;
                let nargs = NormArgs {
                    n: n_embd as u32,
                    eps: self.cfg.rms_eps,
                };
                enc.setComputePipelineState(&self.pipes.rms_norm);
                // SAFETY: `last_byte_off + n_embd * 4 <= MAX_BATCH * n_embd * 4`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.x), last_byte_off, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.output_norm), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.xb), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::from(&nargs).cast::<c_void>(),
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
                self.encode_gemv(&enc, self.out_mat(), &self.xb, &self.logits, false)?;
            }

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            offset += b;
        }
        Ok(())
    }

    /// Profiles Qwen3.5 Metal GPU decode & batched prefill using hardware `GPUStartTime()` / `GPUEndTime()`.
    pub fn profile(
        &mut self,
        decode_runs: usize,
        prefill_len: usize,
    ) -> Result<MetalProfileReport, MetalError> {
        let pqueue = self
            .device
            .newCommandQueueWithMaxCommandBufferCount(1024)
            .ok_or(MetalError::CommandQueueFailed)?;

        self.reset_state();
        for i in 0..4 {
            self.forward_token_greedy((i + 1) as u32, i)?;
        }

        // 1. End-to-end single-CB decode timing
        let n_dec = decode_runs.max(1).min(self.cfg.max_ctx);
        self.reset_state();
        let mut gpu_total_s = 0.0f64;
        let t_wall = Instant::now();
        for step in 0..n_dec {
            let cb = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandEncodeFailed)?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or(MetalError::CommandEncodeFailed)?;
            self.encode_decode_pass(&enc, ((step % 64) + 1) as u32, step, true, true)?;
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            gpu_total_s += (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0);
        }
        let decode_single_cb_wall_us = (t_wall.elapsed().as_secs_f64() * 1e6) / (n_dec as f64);
        let decode_single_cb_gpu_us = (gpu_total_s * 1e6) / (n_dec as f64);

        // 2. Per-stage GPU hardware timestamp breakdown
        const N_STG: usize = 10;
        let names: [&'static str; N_STG] = [
            "embed",
            "rms_norm",
            "gdn_proj (qkv+z+ab)",
            "gdn_core (conv+rec+gnorm)",
            "gdn_ssm_out (Q5_K)",
            "full_qkv_proj",
            "full_attn_core (hd=256)",
            "full_wo",
            "ffn (gate+up+down)",
            "lm_head (Q6_K 248k)",
        ];
        let mut stg_sec = [0.0f64; N_STG];
        let mut stg_bytes = [0u64; N_STG];
        let mut stg_flops = [0u64; N_STG];

        let n_embd = self.cfg.n_embd;
        let state_elems = self.cfg.ssm_dt_rank * self.cfg.ssm_state_size * self.cfg.ssm_state_size;
        stg_bytes[0] = (self.token_embd.row_bytes + n_embd * 4) as u64;
        stg_bytes[1] = ((2 * self.cfg.n_layers + 1) * n_embd * 12) as u64;
        for l in &self.layers {
            match &l.attn {
                GpuAttnLayer::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    w_out,
                    ..
                } => {
                    stg_bytes[2] += (wqkv.row_bytes * wqkv.rows
                        + wz.row_bytes * wz.rows
                        + w_alpha.row_bytes * w_alpha.rows
                        + w_beta.row_bytes * w_beta.rows)
                        as u64;
                    stg_flops[2] +=
                        (2 * (wqkv.rows + wz.rows + w_alpha.rows + w_beta.rows) * n_embd) as u64;
                    // State read + write (2 * 1 MB per GDN layer)
                    stg_bytes[3] += (2 * state_elems * 4) as u64;
                    stg_flops[3] += (6 * state_elems) as u64;
                    stg_bytes[4] += (w_out.row_bytes * w_out.rows) as u64;
                    stg_flops[4] += (2 * w_out.rows * w_out.cols) as u64;
                }
                GpuAttnLayer::FullAttn {
                    wqg, wk, wv, wo, ..
                } => {
                    stg_bytes[5] += (wqg.row_bytes * wqg.rows
                        + wk.row_bytes * wk.rows
                        + wv.row_bytes * wv.rows) as u64;
                    stg_flops[5] += (2 * (wqg.rows + wk.rows + wv.rows) * n_embd) as u64;
                    stg_bytes[7] += (wo.row_bytes * wo.rows) as u64;
                    stg_flops[7] += (2 * wo.rows * wo.cols) as u64;
                }
            }
            stg_bytes[8] += (l.w_gate.row_bytes * l.w_gate.rows
                + l.w_up.row_bytes * l.w_up.rows
                + l.w_down.row_bytes * l.w_down.rows) as u64;
            stg_flops[8] += (2
                * (l.w_gate.rows * l.w_gate.cols
                    + l.w_up.rows * l.w_up.cols
                    + l.w_down.rows * l.w_down.cols)) as u64;
        }
        let out_m = self.out_mat();
        stg_bytes[9] = (out_m.row_bytes * out_m.rows) as u64;
        stg_flops[9] = (2 * out_m.rows * out_m.cols) as u64;

        self.reset_state();
        let mut pending: Vec<(usize, Retained<ProtocolObject<dyn MTLCommandBuffer>>)> =
            Vec::with_capacity(64);
        let drain = |pending: &mut Vec<(usize, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
                     acc: &mut [f64; N_STG]| {
            if let Some((_, last)) = pending.last() {
                last.waitUntilCompleted();
            }
            for (idx, cb) in pending.drain(..) {
                acc[idx] += (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0);
            }
        };

        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let n_full = self
            .layers
            .iter()
            .filter(|l| matches!(l.attn, GpuAttnLayer::FullAttn { .. }))
            .count();
        let mut kv_bytes_sum = 0u64;

        for step in 0..n_dec {
            kv_bytes_sum += (n_full * (step + 1) * kv_dim * 4) as u64;
            // Embed
            {
                let cb = pqueue
                    .commandBuffer()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let (epipe, n_items): (&MtlPipe, usize) = match self.token_embd.dtype {
                    DType::Q4_0 => (&self.pipes.embed_q4_0, n_embd / 32),
                    DType::Q8_0 => (&self.pipes.embed_q8_0, n_embd / 32),
                    DType::Q6K => (&self.pipes.embed_q6_k, n_embd / 256),
                    DType::F32 => (&self.pipes.embed_f32, n_embd),
                    other => return Err(MetalError::UnsupportedDType(other)),
                };
                let eargs = EmbedArgs {
                    token_id: ((step % 64) + 1) as u32,
                    n_embd: n_embd as u32,
                    row_bytes: self.token_embd.row_bytes as u32,
                };
                let etg = n_items.clamp(1, 64);
                enc.setComputePipelineState(epipe);
                // SAFETY: Buffers live on `self` until `drain`.
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
                        width: n_items.div_ceil(etg),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: etg,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
                cb.commit();
                pending.push((0, cb));
            }

            for (li, layer) in self.layers.iter().enumerate() {
                // Pre-attn norm
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_rms_norm(&enc, &self.x, &layer.attn_norm, &self.xb, n_embd, 1);
                    enc.endEncoding();
                    cb.commit();
                    pending.push((1, cb));
                }

                match &layer.attn {
                    GpuAttnLayer::Gdn {
                        wqkv,
                        wz,
                        w_alpha,
                        w_beta,
                        conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        w_out,
                        state,
                        conv_hist,
                        ..
                    } => {
                        {
                            let cb = pqueue
                                .commandBuffer()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            let enc = cb
                                .computeCommandEncoder()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            self.encode_gemv(&enc, wqkv, &self.xb, &self.qkv_buf, false)?;
                            self.encode_gemv(&enc, wz, &self.xb, &self.gate_buf, false)?;
                            self.encode_gemv(&enc, w_alpha, &self.xb, &self.alpha_buf, false)?;
                            self.encode_gemv(&enc, w_beta, &self.xb, &self.beta_buf, false)?;
                            enc.endEncoding();
                            cb.commit();
                            pending.push((2, cb));
                        }
                        {
                            let cb = pqueue
                                .commandBuffer()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            let enc = cb
                                .computeCommandEncoder()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            self.encode_gdn_core(
                                &enc,
                                conv1d,
                                ssm_a,
                                ssm_dt_bias,
                                ssm_norm,
                                state,
                                conv_hist,
                                1,
                            );
                            enc.endEncoding();
                            cb.commit();
                            pending.push((3, cb));
                        }
                        {
                            let cb = pqueue
                                .commandBuffer()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            let enc = cb
                                .computeCommandEncoder()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            self.encode_gemv(&enc, w_out, &self.attn_out, &self.x, true)?;
                            enc.endEncoding();
                            cb.commit();
                            pending.push((4, cb));
                        }
                    }
                    GpuAttnLayer::FullAttn {
                        q_norm,
                        k_norm,
                        wqg,
                        wk,
                        wv,
                        wo,
                        k_cache,
                        v_cache,
                    } => {
                        {
                            let cb = pqueue
                                .commandBuffer()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            let enc = cb
                                .computeCommandEncoder()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            self.encode_gemv(&enc, wqg, &self.xb, &self.qkv_buf, false)?;
                            self.encode_gemv(&enc, wk, &self.xb, &self.k_buf, false)?;
                            self.encode_gemv(&enc, wv, &self.xb, &self.v_buf, false)?;
                            enc.endEncoding();
                            cb.commit();
                            pending.push((5, cb));
                        }
                        {
                            let cb = pqueue
                                .commandBuffer()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            let enc = cb
                                .computeCommandEncoder()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            self.encode_full_attn_core(
                                &enc, q_norm, k_norm, k_cache, v_cache, step, 1,
                            );
                            enc.endEncoding();
                            cb.commit();
                            pending.push((6, cb));
                        }
                        {
                            let cb = pqueue
                                .commandBuffer()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            let enc = cb
                                .computeCommandEncoder()
                                .ok_or(MetalError::CommandEncodeFailed)?;
                            self.encode_gemv(&enc, wo, &self.attn_out, &self.x, true)?;
                            enc.endEncoding();
                            cb.commit();
                            pending.push((7, cb));
                        }
                    }
                }

                // Post-attn norm
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    self.encode_rms_norm(&enc, &self.x, &layer.post_attn_norm, &self.xb, n_embd, 1);
                    enc.endEncoding();
                    cb.commit();
                    pending.push((1, cb));
                }

                // FFN
                {
                    let cb = pqueue
                        .commandBuffer()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let enc = cb
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandEncodeFailed)?;
                    let gargs = GemvArgs {
                        n_in: layer.w_gate.cols as u32,
                        n_out: layer.w_gate.rows as u32,
                        add_residual: 0,
                        row_bytes: layer.w_gate.row_bytes as u32,
                    };
                    enc.setComputePipelineState(&self.pipes.gemv_pair_silu_q4_0);
                    // SAFETY: Buffers live on `self` until `drain`.
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
                            width: layer.w_gate.rows.div_ceil(16),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                    self.encode_gemv(&enc, &layer.w_down, &self.hb, &self.x, true)?;
                    enc.endEncoding();
                    cb.commit();
                    pending.push((8, cb));
                }

                if (li + 1) % 6 == 0 {
                    drain(&mut pending, &mut stg_sec);
                }
            }

            // Output norm + LM head
            {
                let cb = pqueue
                    .commandBuffer()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                self.encode_rms_norm(&enc, &self.x, &self.output_norm, &self.xb, n_embd, 1);
                enc.endEncoding();
                cb.commit();
                pending.push((1, cb));
            }
            {
                let cb = pqueue
                    .commandBuffer()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandEncodeFailed)?;
                self.encode_gemv(&enc, self.out_mat(), &self.xb, &self.logits, false)?;
                enc.endEncoding();
                cb.commit();
                pending.push((9, cb));
            }
            drain(&mut pending, &mut stg_sec);
        }

        stg_bytes[6] = kv_bytes_sum / (n_dec as u64);
        let decode_stages: Vec<MetalStageStat> = (0..N_STG)
            .filter(|&i| stg_sec[i] > 0.0)
            .map(|i| MetalStageStat {
                name: names[i],
                gpu_us: (stg_sec[i] * 1e6) / (n_dec as f64),
                bytes: stg_bytes[i],
                flops: stg_flops[i],
            })
            .collect();

        // 3. End-to-end prefill wall timing
        let n_pp = prefill_len.max(1).min(self.cfg.max_ctx);
        let pp_tokens: Vec<u32> = (0..n_pp).map(|i| ((i % 64) + 1) as u32).collect();
        self.reset_state();
        let t_pp = Instant::now();
        self.forward_batch(&pp_tokens, 0, true)?;
        let prefill_wall_ms = t_pp.elapsed().as_secs_f64() * 1e3;

        Ok(MetalProfileReport {
            decode_runs: n_dec,
            decode_single_cb_wall_us,
            decode_single_cb_gpu_us,
            decode_stages,
            prefill_tokens: n_pp,
            prefill_wall_ms,
            prefill_chunk_tokens: n_pp.min(MAX_BATCH),
            prefill_chunk_stages: Vec::new(),
        })
    }
}
