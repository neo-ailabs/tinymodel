//! Apple Metal GPU forward & transposed-backward quantized GEMM accelerator (`Phase T4`)
//! for `tinymodel finetune --metal`.
//!
//! Shares zero-copy GGUF `mmap` weight pages via `newBufferWithBytesNoCopy` on Apple Silicon
//! unified memory and executes heavy `W * x` forward and `dy * W` transposed backward projections
//! on hardware `simdgroup_float8x8` tensor cores without dequantizing weights to FP32.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::collections::HashMap;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use tm_core::{DType, QK4_0};
use tm_gguf::GgufFile;

use crate::MetalError;

const LLAMA_OPS_MSL: &str = include_str!("shaders/llama_ops.metal");
const METAL_PAGE_SIZE: usize = 16384;
/// Default minimum MAC count (`n_tok * rows * cols`) to amortize synchronous Metal CB roundtrip.
pub const DEFAULT_METAL_MIN_MACS: usize = 4_194_304;

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

struct AccelMat {
    buf: MtlBuf,
    offset: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

struct AccelPipes {
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
    bwd_q4_0: MtlPipe,
    bwd_q4_1: MtlPipe,
    bwd_q8_0: MtlPipe,
    bwd_q5_k: MtlPipe,
    bwd_q6_k: MtlPipe,
    bwd_bf16: MtlPipe,
    bwd_f32: MtlPipe,
}

/// Zero-copy Apple Metal quantized GEMM accelerator for QLoRA fine-tuning.
pub struct MetalGpuAccel {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipes: AccelPipes,
    mats: HashMap<String, AccelMat>,
    in_buf: MtlBuf,
    out_buf: MtlBuf,
    min_macs: usize,
    zero_copy_weights: bool,
    // Must remain LAST so zero-copy `MTLBuffer` views drop before `GgufFile` unmaps.
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
    let limit = pipe.maxTotalThreadsPerThreadgroup();
    if max_tg > limit {
        return Err(MetalError::ThreadgroupTooLarge {
            kernel: name,
            requested: max_tg,
            limit,
        });
    }
    Ok(pipe)
}

fn alloc_shared_buf(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<MtlBuf, MetalError> {
    let len = bytes.max(16);
    device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAlloc { bytes: len })
}

fn upload_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<MtlBuf, MetalError> {
    let len = bytes.len().max(16);
    let buf = alloc_shared_buf(device, len)?;
    if !bytes.is_empty() {
        // SAFETY: `buf` has capacity `len >= bytes.len()` in `StorageModeShared`.
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

fn uninterleave_q4_0x4_to_q4_0(bytes: &[u8], rows: usize, cols: usize) -> Option<Vec<u8>> {
    if !rows.is_multiple_of(4) || !cols.is_multiple_of(QK4_0) {
        return None;
    }
    let cols_blocks = cols / QK4_0;
    let row_stride = cols_blocks * 18;
    let mut dst = vec![0u8; rows * row_stride];
    for rg in 0..(rows / 4) {
        let src_g = &bytes[rg * cols_blocks * 72..(rg + 1) * cols_blocks * 72];
        for ib in 0..cols_blocks {
            let sb = &src_g[ib * 72..(ib + 1) * 72];
            let off0 = (rg * 4) * row_stride + ib * 18;
            let off1 = (rg * 4 + 1) * row_stride + ib * 18;
            let off2 = (rg * 4 + 2) * row_stride + ib * 18;
            let off3 = (rg * 4 + 3) * row_stride + ib * 18;
            dst[off0..off0 + 2].copy_from_slice(&sb[0..2]);
            dst[off0 + 2..off0 + 10].copy_from_slice(&sb[8..16]);
            dst[off0 + 10..off0 + 18].copy_from_slice(&sb[24..32]);
            dst[off1..off1 + 2].copy_from_slice(&sb[2..4]);
            dst[off1 + 2..off1 + 10].copy_from_slice(&sb[16..24]);
            dst[off1 + 10..off1 + 18].copy_from_slice(&sb[32..40]);
            dst[off2..off2 + 2].copy_from_slice(&sb[4..6]);
            dst[off2 + 2..off2 + 10].copy_from_slice(&sb[40..48]);
            dst[off2 + 10..off2 + 18].copy_from_slice(&sb[56..64]);
            dst[off3..off3 + 2].copy_from_slice(&sb[6..8]);
            dst[off3 + 2..off3 + 10].copy_from_slice(&sb[48..56]);
            dst[off3 + 10..off3 + 18].copy_from_slice(&sb[64..72]);
        }
    }
    Some(dst)
}

impl MetalGpuAccel {
    /// Initializes a zero-copy Metal training accelerator over all supported 2D matrices in `gguf`.
    pub fn from_gguf(gguf: Arc<GgufFile>, max_seq: usize) -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueFailed)?;

        let src = NSString::from_str(LLAMA_OPS_MSL);
        let lib = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let pipes = AccelPipes {
            gemv_q4_0: build_pipe(&device, &lib, "gemv_q4_0_res", 128)?,
            gemv_q4_1: build_pipe(&device, &lib, "gemv_q4_1_res", 128)?,
            gemv_q8_0: build_pipe(&device, &lib, "gemv_q8_0_fast", 128)?,
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
            bwd_q4_0: build_pipe(&device, &lib, "gemm_q4_0_transposed", 128)?,
            bwd_q4_1: build_pipe(&device, &lib, "gemm_q4_1_transposed", 128)?,
            bwd_q8_0: build_pipe(&device, &lib, "gemm_q8_0_transposed", 128)?,
            bwd_q5_k: build_pipe(&device, &lib, "gemm_q5_k_transposed", 128)?,
            bwd_q6_k: build_pipe(&device, &lib, "gemm_q6_k_transposed", 128)?,
            bwd_bf16: build_pipe(&device, &lib, "gemm_bf16_transposed", 128)?,
            bwd_f32: build_pipe(&device, &lib, "gemm_f32_transposed", 128)?,
        };

        let shared_buf = {
            let raw = gguf.as_bytes();
            let ptr = raw.as_ptr();
            if !raw.is_empty() && (ptr as usize).is_multiple_of(METAL_PAGE_SIZE) {
                let aligned_len = raw.len().div_ceil(METAL_PAGE_SIZE) * METAL_PAGE_SIZE;
                NonNull::new(ptr as *mut c_void).and_then(|nn| {
                    // SAFETY: `ptr` is page-aligned and backed by `gguf`'s read-only `Mmap` for
                    // `aligned_len` bytes; `_gguf` is the last field of `MetalGpuAccel`.
                    unsafe {
                        device.newBufferWithBytesNoCopy_length_options_deallocator(
                            nn,
                            aligned_len,
                            MTLResourceOptions::StorageModeShared,
                            None,
                        )
                    }
                })
            } else {
                None
            }
        };
        let zero_copy_weights = shared_buf.is_some();

        let mut mats = HashMap::new();
        let mut max_dim = 4096usize;
        for tname in gguf.tensor_names() {
            let Some(info) = gguf.tensor_info(tname) else {
                continue;
            };
            if info.dims.len() != 2 {
                continue;
            }
            let cols = info.dims[0];
            let rows = info.dims[1];
            max_dim = max_dim.max(cols).max(rows);

            let Ok((_, raw_bytes)) = gguf.tensor_bytes(tname, "accel") else {
                continue;
            };
            match info.dtype {
                DType::Q4_0
                | DType::Q4_1
                | DType::Q8_0
                | DType::Q5K
                | DType::Q6K
                | DType::BF16
                | DType::F32 => {
                    let (buf, offset) = if let Some(ref sb) = shared_buf {
                        (sb.clone(), gguf.tensor_file_offset(info))
                    } else {
                        (upload_bytes(&device, raw_bytes)?, 0)
                    };
                    mats.insert(
                        tname.clone(),
                        AccelMat {
                            buf,
                            offset,
                            dtype: info.dtype,
                            rows,
                            cols,
                            row_bytes: info.dtype.row_bytes(cols),
                        },
                    );
                }
                DType::Q4_0x4 => {
                    if let Some(q4) = uninterleave_q4_0x4_to_q4_0(raw_bytes, rows, cols) {
                        let buf = upload_bytes(&device, &q4)?;
                        mats.insert(
                            tname.clone(),
                            AccelMat {
                                buf,
                                offset: 0,
                                dtype: DType::Q4_0,
                                rows,
                                cols,
                                row_bytes: DType::Q4_0.row_bytes(cols),
                            },
                        );
                    }
                }
                _ => {}
            }
        }

        // Also map tied `"output.weight"` -> `"token_embd.weight"` when `output.weight` is absent.
        if !mats.contains_key("output.weight")
            && let Some(emb) = mats.get("token_embd.weight")
        {
            mats.insert(
                "output.weight".to_string(),
                AccelMat {
                    buf: emb.buf.clone(),
                    offset: emb.offset,
                    dtype: emb.dtype,
                    rows: emb.rows,
                    cols: emb.cols,
                    row_bytes: emb.row_bytes,
                },
            );
        }

        let scratch_elems = max_seq.max(1) * max_dim.max(1);
        let in_buf = alloc_shared_buf(&device, scratch_elems * 4)?;
        let out_buf = alloc_shared_buf(&device, scratch_elems * 4)?;

        Ok(Self {
            device,
            queue,
            pipes,
            mats,
            in_buf,
            out_buf,
            min_macs: DEFAULT_METAL_MIN_MACS,
            zero_copy_weights,
            _gguf: gguf,
        })
    }

    pub fn with_min_macs(mut self, min_macs: usize) -> Self {
        self.min_macs = min_macs;
        self
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn is_zero_copy_weights(&self) -> bool {
        self.zero_copy_weights
    }

    fn ensure_scratch(&mut self, in_bytes: usize, out_bytes: usize) -> bool {
        if self.in_buf.length() < in_bytes {
            let Ok(b) = alloc_shared_buf(&self.device, in_bytes) else {
                return false;
            };
            self.in_buf = b;
        }
        if self.out_buf.length() < out_bytes {
            let Ok(b) = alloc_shared_buf(&self.device, out_bytes) else {
                return false;
            };
            self.out_buf = b;
        }
        true
    }

    /// Forward `y[n_tok, rows] = x[n_tok, cols] @ W[rows, cols]^T` on Apple Metal GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_fwd(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        x: &[f32],
        y: &mut [f32],
        n_tok: usize,
    ) -> bool {
        if n_tok == 0 || rows == 0 || cols == 0 || n_tok * rows * cols < self.min_macs {
            return false;
        }
        let in_bytes = n_tok * cols * 4;
        let out_bytes = n_tok * rows * 4;
        if x.len() < n_tok * cols || y.len() < n_tok * rows {
            return false;
        }
        if !self.ensure_scratch(in_bytes, out_bytes) {
            return false;
        }
        let Some(mat) = self.mats.get(tensor_name) else {
            return false;
        };
        if mat.rows != rows || mat.cols != cols {
            return false;
        }

        // SAFETY: `self.in_buf` is idle and has `>= in_bytes` in `StorageModeShared`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                x.as_ptr(),
                self.in_buf.contents().as_ptr() as *mut f32,
                n_tok * cols,
            );
        }

        let Some(cmd) = self.queue.commandBuffer() else {
            return false;
        };
        let Some(enc) = cmd.computeCommandEncoder() else {
            return false;
        };

        if n_tok == 1 {
            let (pipe, r_per_tg): (&MtlPipe, usize) = match mat.dtype {
                DType::Q4_0 => (&self.pipes.gemv_q4_0, 32),
                DType::Q4_1 => (&self.pipes.gemv_q4_1, 32),
                DType::Q8_0 => (&self.pipes.gemv_q8_0, 16),
                DType::Q5K => (&self.pipes.gemv_q5_k, 8),
                DType::Q6K => (&self.pipes.gemv_q6_k, 16),
                DType::BF16 => (&self.pipes.gemv_bf16, 16),
                DType::F32 => (&self.pipes.gemv_f32, 8),
                _ => {
                    enc.endEncoding();
                    return false;
                }
            };
            let args = GemvArgs {
                n_in: cols as u32,
                n_out: rows as u32,
                add_residual: 0,
                row_bytes: mat.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&self.in_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.out_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<GemvArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows.div_ceil(r_per_tg),
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
            let (pipe, r_tile, t_tile): (&MtlPipe, usize, usize) = match mat.dtype {
                DType::Q4_0 => (&self.pipes.gemm_q4_0, 32, 32),
                DType::Q4_1 => (&self.pipes.gemm_q4_1, 32, 32),
                DType::Q8_0 => (&self.pipes.gemm_q8_0, 4, 8),
                DType::Q5K => (&self.pipes.gemm_q5_k, 4, 8),
                DType::Q6K => (&self.pipes.gemm_q6_k, 4, 8),
                DType::BF16 => (&self.pipes.gemm_bf16, 32, 32),
                DType::F32 => (&self.pipes.gemm_f32, 4, 8),
                _ => {
                    enc.endEncoding();
                    return false;
                }
            };
            let args = GemmArgs {
                n_in: cols as u32,
                n_out: rows as u32,
                n_tok: n_tok as u32,
                add_residual: 0,
                row_bytes: mat.row_bytes as u32,
            };
            enc.setComputePipelineState(pipe);
            // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
                enc.setBuffer_offset_atIndex(Some(&self.in_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&self.out_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<GemmArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows.div_ceil(r_tile),
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

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // SAFETY: GPU completed; `self.out_buf` holds `n_tok * rows` floats in `StorageModeShared`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.out_buf.contents().as_ptr() as *const f32,
                y.as_mut_ptr(),
                n_tok * rows,
            );
        }
        true
    }

    /// Transposed backward `dx[n_tok, cols] = (accum ? dx : 0) + dy[n_tok, rows] @ W[rows, cols]`
    /// on Apple Metal `simdgroup_float8x8` hardware tensor cores!
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_bwd_transposed(
        &mut self,
        tensor_name: &str,
        rows: usize,
        cols: usize,
        dy: &[f32],
        dx: &mut [f32],
        n_tok: usize,
        accum: bool,
    ) -> bool {
        if n_tok == 0 || rows == 0 || cols == 0 || n_tok * rows * cols < self.min_macs {
            return false;
        }
        let in_bytes = n_tok * rows * 4;
        let out_bytes = n_tok * cols * 4;
        if dy.len() < n_tok * rows || dx.len() < n_tok * cols {
            return false;
        }
        if !self.ensure_scratch(in_bytes, out_bytes) {
            return false;
        }
        let Some(mat) = self.mats.get(tensor_name) else {
            return false;
        };
        if mat.rows != rows || mat.cols != cols {
            return false;
        }

        let pipe: &MtlPipe = match mat.dtype {
            DType::Q4_0 => &self.pipes.bwd_q4_0,
            DType::Q4_1 => &self.pipes.bwd_q4_1,
            DType::Q8_0 => &self.pipes.bwd_q8_0,
            DType::Q5K => &self.pipes.bwd_q5_k,
            DType::Q6K => &self.pipes.bwd_q6_k,
            DType::BF16 => &self.pipes.bwd_bf16,
            DType::F32 => &self.pipes.bwd_f32,
            _ => return false,
        };

        // SAFETY: `in_buf` and `out_buf` are idle `StorageModeShared` buffers of sufficient size.
        unsafe {
            core::ptr::copy_nonoverlapping(
                dy.as_ptr(),
                self.in_buf.contents().as_ptr() as *mut f32,
                n_tok * rows,
            );
            if accum {
                core::ptr::copy_nonoverlapping(
                    dx.as_ptr(),
                    self.out_buf.contents().as_ptr() as *mut f32,
                    n_tok * cols,
                );
            }
        }

        let Some(cmd) = self.queue.commandBuffer() else {
            return false;
        };
        let Some(enc) = cmd.computeCommandEncoder() else {
            return false;
        };

        let args = GemmArgs {
            n_in: cols as u32,
            n_out: rows as u32,
            n_tok: n_tok as u32,
            add_residual: u32::from(accum),
            row_bytes: mat.row_bytes as u32,
        };
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers live on `self` until `waitUntilCompleted`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&mat.buf), mat.offset, 0);
            enc.setBuffer_offset_atIndex(Some(&self.in_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.out_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<GemmArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: cols.div_ceil(32),
                height: n_tok.div_ceil(32),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        // SAFETY: GPU completed; `self.out_buf` holds `n_tok * cols` floats.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self.out_buf.contents().as_ptr() as *const f32,
                dx.as_mut_ptr(),
                n_tok * cols,
            );
        }
        true
    }
}
