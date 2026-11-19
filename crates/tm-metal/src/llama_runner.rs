//! End-to-end Apple Metal GPU runner for `llama` and `qwen3` GGUF models.
//!
//! Encodes the entire token forward pass (embedding lookup -> N layers of
//! RMSNorm, QKV GEMV, RoPE+KV-store, Flash-style online-softmax GQA, Wo+residual,
//! FFN RMSNorm, fused Gate+Up+SwiGLU, Wdown+residual -> output norm -> LM head)
//! into a **single `MTLCommandBuffer`** per token with zero intermediate host syncs.

use core::ffi::c_void;
use core::ptr::NonNull;

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

const LLAMA_OPS_MSL: &str = include_str!("shaders/llama_ops.metal");

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
    buf: MtlBuf,
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
    embed_q4_0: MtlPipe,
    embed_q8_0: MtlPipe,
    embed_q6_k: MtlPipe,
    embed_f32: MtlPipe,
    rms_norm: MtlPipe,
    qk_norm: MtlPipe,
    rope_kv: MtlPipe,
    gqa_attn: MtlPipe,
    gemv_q4_0: MtlPipe,
    gemv_pair_silu_q4_0: MtlPipe,
    gemv_q4_1: MtlPipe,
    gemv_q8_0: MtlPipe,
    gemv_q6_k: MtlPipe,
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
    // Pre-allocated activation buffers (zero allocation during inference)
    x: MtlBuf,
    xb: MtlBuf,
    q: MtlBuf,
    k: MtlBuf,
    v: MtlBuf,
    hb: MtlBuf,
    hb2: MtlBuf,
    logits: MtlBuf,
}

fn build_pipe(
    device: &ProtocolObject<dyn MTLDevice>,
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<MtlPipe, MetalError> {
    let ns = NSString::from_str(name);
    let func = lib
        .newFunctionWithName(&ns)
        .ok_or_else(|| MetalError::FunctionNotFound(name.to_string()))?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| MetalError::PipelineCreation {
            name: name.to_string(),
            reason: e.localizedDescription().to_string(),
        })
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

fn load_gpu_mat(
    device: &ProtocolObject<dyn MTLDevice>,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<GpuMat, MetalError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    match info.dtype {
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q6K | DType::F32 => {}
        other => return Err(MetalError::UnsupportedDType(other)),
    }
    let buf = upload_bytes(device, bytes)?;
    Ok(GpuMat {
        buf,
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
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, MetalError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "llama" && arch != "qwen3" {
            return Err(MetalError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
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
            .unwrap_or(n_embd / n_head);
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "qwen3" { 1_000_000.0 } else { 10000.0 });
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if pfx == "qwen3" { 1e-6 } else { 1e-5 });
        let rope_type = if pfx == "qwen3" { 1 } else { 0 };

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

        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueFailed)?;

        let src = NSString::from_str(LLAMA_OPS_MSL);
        let lib = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let pipes = Pipelines {
            embed_q4_0: build_pipe(&device, &lib, "embed_q4_0")?,
            embed_q8_0: build_pipe(&device, &lib, "embed_q8_0")?,
            embed_q6_k: build_pipe(&device, &lib, "embed_q6_k")?,
            embed_f32: build_pipe(&device, &lib, "embed_f32")?,
            rms_norm: build_pipe(&device, &lib, "rms_norm_f32")?,
            qk_norm: build_pipe(&device, &lib, "qk_norm_f32")?,
            rope_kv: build_pipe(&device, &lib, "rope_and_kv_store_f16")?,
            gqa_attn: build_pipe(&device, &lib, "gqa_attn_f16_kv")?,
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res")?,
            gemv_pair_silu_q4_0: build_pipe(&device, &lib, "gemv_pair_silu_q4_0")?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res")?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_res")?,
            gemv_q6_k: build_pipe(&device, &lib, "gemv_q6_k_res")?,
        };

        let token_embd = load_gpu_mat(
            &device,
            gguf,
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
                    gguf,
                    "output.weight",
                    "global",
                    n_embd,
                    vocab_size,
                )?),
            )
        } else {
            (true, None)
        };
        let output_norm = load_gpu_f32_vec(&device, gguf, "output_norm.weight", "global", n_embd)?;
        let (has_rope_freqs, rope_freqs) = if gguf.tensor_info("rope_freqs.weight").is_some() {
            (
                true,
                load_gpu_f32_vec(&device, gguf, "rope_freqs.weight", "global", rope_dim / 2)?,
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
            let q_norm = if gguf.tensor_info(&qn_name).is_some() {
                Some(load_gpu_f32_vec(&device, gguf, &qn_name, &ltag, head_dim)?)
            } else {
                None
            };
            let k_norm = if gguf.tensor_info(&kn_name).is_some() {
                Some(load_gpu_f32_vec(&device, gguf, &kn_name, &ltag, head_dim)?)
            } else {
                None
            };

            layers.push(GpuLayer {
                attn_norm: load_gpu_f32_vec(
                    &device,
                    gguf,
                    &format!("{ltag}.attn_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                ffn_norm: load_gpu_f32_vec(
                    &device,
                    gguf,
                    &format!("{ltag}.ffn_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                q_norm,
                k_norm,
                wq: load_gpu_mat(
                    &device,
                    gguf,
                    &format!("{ltag}.attn_q.weight"),
                    &ltag,
                    n_embd,
                    q_dim,
                )?,
                wk: load_gpu_mat(
                    &device,
                    gguf,
                    &format!("{ltag}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wv: load_gpu_mat(
                    &device,
                    gguf,
                    &format!("{ltag}.attn_v.weight"),
                    &ltag,
                    n_embd,
                    kv_dim,
                )?,
                wo: load_gpu_mat(
                    &device,
                    gguf,
                    &format!("{ltag}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                )?,
                w_gate: load_gpu_mat(
                    &device,
                    gguf,
                    &format!("{ltag}.ffn_gate.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_up: load_gpu_mat(
                    &device,
                    gguf,
                    &format!("{ltag}.ffn_up.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_down: load_gpu_mat(
                    &device,
                    gguf,
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
        let x = alloc_buf(&device, n_embd * f32_sz)?;
        let xb = alloc_buf(&device, n_embd.max(q_dim) * f32_sz)?;
        let q = alloc_buf(&device, q_dim * f32_sz)?;
        let k = alloc_buf(&device, kv_dim * f32_sz)?;
        let v = alloc_buf(&device, kv_dim * f32_sz)?;
        let hb = alloc_buf(&device, n_ff * f32_sz)?;
        let hb2 = alloc_buf(&device, n_ff * f32_sz)?;
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
            x,
            xb,
            q,
            k,
            v,
            hb,
            hb2,
            logits,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    fn encode_gemv(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        mat: &GpuMat,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        add_residual: bool,
    ) -> Result<(), MetalError> {
        let (pipe, name, rows_per_tg): (&MtlPipe, &'static str, usize) = match mat.dtype {
            DType::Q4_0 => (&self.pipes.gemv_q4_0, "gemv_q4_0_res", 32),
            DType::Q4_1 => (&self.pipes.gemv_q4_1, "gemv_q4_1_res", 32),
            DType::Q8_0 => (&self.pipes.gemv_q8_0, "gemv_q8_0_res", 16),
            DType::Q6K => (&self.pipes.gemv_q6_k, "gemv_q6_k_res", 16),
            other => return Err(MetalError::UnsupportedDType(other)),
        };

        const THREADS: usize = 128;
        require_fit(pipe, name, THREADS)?;

        let args = GemvArgs {
            n_in: mat.cols as u32,
            n_out: mat.rows as u32,
            add_residual: u32::from(add_residual),
            row_bytes: mat.row_bytes as u32,
        };

        enc.setComputePipelineState(pipe);
        // SAFETY: All buffers live on `self` for the full command buffer lifetime.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mat.buf), 0, 0);
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
        const THREADS: usize = 256;
        require_fit(&self.pipes.rms_norm, "rms_norm_f32", THREADS)?;
        let args = NormArgs {
            n: n as u32,
            eps: self.cfg.rms_eps,
        };
        enc.setComputePipelineState(&self.pipes.rms_norm);
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

        // 1. Embedding lookup on GPU
        {
            let (pipe, name, n_items): (&MtlPipe, &'static str, usize) = match self.token_embd.dtype
            {
                DType::Q4_0 => (&self.pipes.embed_q4_0, "embed_q4_0", self.cfg.n_embd / 32),
                DType::Q8_0 => (&self.pipes.embed_q8_0, "embed_q8_0", self.cfg.n_embd / 32),
                DType::Q6K => (&self.pipes.embed_q6_k, "embed_q6_k", self.cfg.n_embd / 256),
                DType::F32 => (&self.pipes.embed_f32, "embed_f32", self.cfg.n_embd),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
            let tg_w = n_items.clamp(1, 64);
            require_fit(pipe, name, tg_w)?;
            let eargs = EmbedArgs {
                token_id,
                n_embd: self.cfg.n_embd as u32,
                row_bytes: self.token_embd.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&self.token_embd.buf), 0, 0);
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
        }

        // 2. Transformer layers (all in the same command buffer!)
        let half_rope = self.cfg.rope_dim / 2;
        let rope_threads = (self.cfg.n_head + self.cfg.n_head_kv) * half_rope;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let rope_total = rope_threads.max(kv_dim);

        for layer in &self.layers {
            self.encode_rms_norm(&enc, &self.x, &layer.attn_norm, &self.xb, self.cfg.n_embd)?;

            self.encode_gemv(&enc, &layer.wq, &self.xb, &self.q, false)?;
            self.encode_gemv(&enc, &layer.wk, &self.xb, &self.k, false)?;
            self.encode_gemv(&enc, &layer.wv, &self.xb, &self.v, false)?;

            // Optional Qwen3 per-head QK-Norm
            if let Some(ref qn) = layer.q_norm {
                require_fit(&self.pipes.qk_norm, "qk_norm_f32", 32)?;
                let na = NormArgs {
                    n: self.cfg.head_dim as u32,
                    eps: self.cfg.rms_eps,
                };
                enc.setComputePipelineState(&self.pipes.qk_norm);
                // SAFETY: Buffers remain alive until `waitUntilCompleted`.
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
                require_fit(&self.pipes.qk_norm, "qk_norm_f32", 32)?;
                let na = NormArgs {
                    n: self.cfg.head_dim as u32,
                    eps: self.cfg.rms_eps,
                };
                enc.setComputePipelineState(&self.pipes.qk_norm);
                // SAFETY: Buffers remain alive until `waitUntilCompleted`.
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

            // Fused RoPE + F16 KV store
            {
                const TG: usize = 64;
                require_fit(&self.pipes.rope_kv, "rope_and_kv_store_f16", TG)?;
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
                // SAFETY: Buffers remain alive until `waitUntilCompleted`.
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

            // GQA attention over F16 KV cache
            {
                const TG: usize = 128;
                require_fit(&self.pipes.gqa_attn, "gqa_attn_f16_kv", TG)?;
                let aargs = AttnArgs {
                    n_head: self.cfg.n_head as u32,
                    n_head_kv: self.cfg.n_head_kv as u32,
                    head_dim: self.cfg.head_dim as u32,
                    seq_len: (pos + 1) as u32,
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
                        width: TG,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            // Wo + residual add directly into `self.x`
            self.encode_gemv(&enc, &layer.wo, &self.xb, &self.x, true)?;

            // FFN RMSNorm
            self.encode_rms_norm(&enc, &self.x, &layer.ffn_norm, &self.xb, self.cfg.n_embd)?;

            // Fused Gate + Up + SwiGLU (single dispatch when both are Q4_0!)
            if layer.w_gate.dtype == DType::Q4_0 && layer.w_up.dtype == DType::Q4_0 {
                const THREADS: usize = 128;
                const ROWS_PER_TG: usize = 16;
                require_fit(
                    &self.pipes.gemv_pair_silu_q4_0,
                    "gemv_pair_silu_q4_0",
                    THREADS,
                )?;
                let gargs = GemvArgs {
                    n_in: layer.w_gate.cols as u32,
                    n_out: layer.w_gate.rows as u32,
                    add_residual: 0,
                    row_bytes: layer.w_gate.row_bytes as u32,
                };
                enc.setComputePipelineState(&self.pipes.gemv_pair_silu_q4_0);
                // SAFETY: Buffers remain alive until `waitUntilCompleted`.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&layer.w_gate.buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&layer.w_up.buf), 0, 1);
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
                self.encode_gemv(&enc, &layer.w_gate, &self.xb, &self.hb, false)?;
                self.encode_gemv(&enc, &layer.w_up, &self.xb, &self.hb2, false)?;
                // Fallback if gate/up are non-Q4_0 (not needed for Q4_0 models)
                return Err(MetalError::UnsupportedDType(layer.w_gate.dtype));
            }

            // W_down + residual add directly into `self.x`
            self.encode_gemv(&enc, &layer.w_down, &self.hb, &self.x, true)?;
        }

        // 3. Output norm + LM head
        if compute_logits {
            self.encode_rms_norm(&enc, &self.x, &self.output_norm, &self.xb, self.cfg.n_embd)?;
            let out_mat = if self.output_tied {
                &self.token_embd
            } else {
                self.output_separate.as_ref().unwrap_or(&self.token_embd)
            };
            self.encode_gemv(&enc, out_mat, &self.xb, &self.logits, false)?;
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let ptr = self.logits.contents().as_ptr() as *const f32;
        // SAFETY: `waitUntilCompleted` finished; `self.logits` is `StorageModeShared`
        // and holds `self.cfg.vocab_size` `f32` values.
        Ok(unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) })
    }
}
