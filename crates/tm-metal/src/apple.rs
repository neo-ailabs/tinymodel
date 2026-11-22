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
use tm_core::{BlockQ4_0, BlockQ5K, DType, QK_K, QK4_0};

use crate::MetalError;
use crate::lora_gpu::{LoraEmbedArgs, LoraLinearArgs};

const GEMV_Q4_0_MSL: &str = include_str!("shaders/gemv_q4_0.metal");
const LLAMA_OPS_MSL: &str = include_str!("shaders/llama_ops.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct OpsGemvArgs {
    n_in: u32,
    n_out: u32,
    add_residual: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OpsGemmArgs {
    n_in: u32,
    n_out: u32,
    n_tok: u32,
    add_residual: u32,
    row_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvArgs {
    n_in: u32,
    n_out: u32,
    n_tok: u32,
    row_bytes: u32,
}

/// Shared-memory Metal buffer (unified memory on Apple Silicon: CPU & GPU share the same pages).
pub struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    byte_len: usize,
}

impl MetalBuffer {
    /// Length of the buffer in bytes.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Reads the buffer contents as a slice of `f32`.
    ///
    /// Caller must ensure no GPU command buffer is concurrently writing to this buffer.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.byte_len % core::mem::size_of::<f32>(), 0);
        let ptr = self.raw.contents().as_ptr() as *const f32;
        let len = self.byte_len / core::mem::size_of::<f32>();
        // SAFETY: Buffer is allocated with `StorageModeShared`, has `byte_len` bytes,
        // and `MTLBuffer` base addresses are at least 256-byte aligned (satisfies `f32` align).
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }

    /// Mutable view of the buffer contents as `f32`.
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.byte_len % core::mem::size_of::<f32>(), 0);
        let ptr = self.raw.contents().as_ptr() as *mut f32;
        let len = self.byte_len / core::mem::size_of::<f32>();
        // SAFETY: `&mut self` guarantees exclusive CPU access; `MTLBuffer` is 256-byte aligned.
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }
}

/// Active Apple Metal device, command queue, and compiled compute pipelines.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipe_gemv_q4_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_gemm_q4_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_gemv_q5_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_gemm_q5_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_q4_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_q4_1: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_q8_0: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_q5_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_q6_k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_bwd_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_lora_down: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_lora_up: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipe_lora_embed: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalContext {
    /// Initializes the default Metal GPU device and compiles the embedded MSL kernels.
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueFailed)?;

        let src = NSString::from_str(GEMV_Q4_0_MSL);
        let library = device
            .newLibraryWithSource_options_error(&src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let ops_src = NSString::from_str(LLAMA_OPS_MSL);
        let ops_lib = device
            .newLibraryWithSource_options_error(&ops_src, None)
            .map_err(|e| MetalError::ShaderCompile(e.localizedDescription().to_string()))?;

        let pipe_gemv_q4_0 = Self::build_pipeline(&device, &library, "gemv_q4_0_f32_lpr8_nr2")?;
        let pipe_gemm_q4_0 = Self::build_pipeline(&device, &library, "gemm_q4_0_f32_lpr8")?;
        let pipe_gemv_q5_k = Self::build_pipeline(&device, &ops_lib, "gemv_q5_k_res")?;
        let pipe_gemm_q5_k = Self::build_pipeline(&device, &ops_lib, "gemm_q5_k_res")?;
        let pipe_bwd_q4_0 = Self::build_pipeline(&device, &ops_lib, "gemm_q4_0_transposed")?;
        let pipe_bwd_q4_1 = Self::build_pipeline(&device, &ops_lib, "gemm_q4_1_transposed")?;
        let pipe_bwd_q8_0 = Self::build_pipeline(&device, &ops_lib, "gemm_q8_0_transposed")?;
        let pipe_bwd_q5_k = Self::build_pipeline(&device, &ops_lib, "gemm_q5_k_transposed")?;
        let pipe_bwd_q6_k = Self::build_pipeline(&device, &ops_lib, "gemm_q6_k_transposed")?;
        let pipe_bwd_bf16 = Self::build_pipeline(&device, &ops_lib, "gemm_bf16_transposed")?;
        let pipe_bwd_f32 = Self::build_pipeline(&device, &ops_lib, "gemm_f32_transposed")?;
        let pipe_lora_down = Self::build_pipeline(&device, &ops_lib, "lora_down_f32")?;
        let pipe_lora_up = Self::build_pipeline(&device, &ops_lib, "lora_up_axpy_f32")?;
        let pipe_lora_embed = Self::build_pipeline(&device, &ops_lib, "lora_embed_f32")?;

        Ok(Self {
            device,
            queue,
            pipe_gemv_q4_0,
            pipe_gemm_q4_0,
            pipe_gemv_q5_k,
            pipe_gemm_q5_k,
            pipe_bwd_q4_0,
            pipe_bwd_q4_1,
            pipe_bwd_q8_0,
            pipe_bwd_q5_k,
            pipe_bwd_q6_k,
            pipe_bwd_bf16,
            pipe_bwd_f32,
            pipe_lora_down,
            pipe_lora_up,
            pipe_lora_embed,
        })
    }

    fn build_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let fn_name = NSString::from_str(name);
        let func = library
            .newFunctionWithName(&fn_name)
            .ok_or_else(|| MetalError::FunctionNotFound(name.to_string()))?;
        device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| MetalError::PipelineCreation {
                name: name.to_string(),
                reason: e.localizedDescription().to_string(),
            })
    }

    /// Returns the Metal GPU device name (e.g., `"Apple M4"`).
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Returns the recommended maximum working set size in bytes.
    pub fn recommended_max_working_set_bytes(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// Allocates a shared-memory buffer and copies `data` into it.
    pub fn buffer_from_slice<T: Copy>(&self, data: &[T]) -> Result<MetalBuffer, MetalError> {
        let byte_len = core::mem::size_of_val(data);
        let alloc_len = byte_len.max(4);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void)
            .ok_or(MetalError::BufferAlloc { bytes: alloc_len })?;
        // SAFETY: `ptr` points to `byte_len` readable bytes from `data`; Metal copies
        // the bytes into the newly allocated `StorageModeShared` buffer.
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAlloc { bytes: byte_len })?;
        Ok(MetalBuffer { raw, byte_len })
    }

    /// Allocates an uninitialized (zeroed) shared-memory buffer of `byte_len` bytes.
    pub fn alloc_buffer(&self, byte_len: usize) -> Result<MetalBuffer, MetalError> {
        let raw = self
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAlloc { bytes: byte_len })?;
        Ok(MetalBuffer { raw, byte_len })
    }

    fn require_fit(
        pipe: &ProtocolObject<dyn MTLComputePipelineState>,
        kernel: &'static str,
        threads_per_tg: usize,
    ) -> Result<(), MetalError> {
        let limit = pipe.maxTotalThreadsPerThreadgroup();
        if threads_per_tg > limit {
            return Err(MetalError::ThreadgroupTooLarge {
                kernel,
                requested: threads_per_tg,
                limit,
            });
        }
        Ok(())
    }

    /// Runs single-token `Q4_0 x F32` GEMV on Metal GPU: `y = W * x`.
    /// Executes `repeat` dispatches inside a single command buffer (use `repeat = 1` for normal inference).
    pub fn gemv_q4_0_f32_repeat(
        &self,
        w: &MetalBuffer,
        x: &MetalBuffer,
        y: &mut MetalBuffer,
        rows: usize,
        cols: usize,
        repeat: usize,
    ) -> Result<(), MetalError> {
        if cols == 0 || !cols.is_multiple_of(QK4_0) || rows == 0 {
            return Err(MetalError::InvalidDimensions(format!(
                "rows={rows}, cols={cols} (cols must be non-zero multiple of {QK4_0})"
            )));
        }
        let cols_blocks = cols / QK4_0;
        let row_bytes = cols_blocks * core::mem::size_of::<BlockQ4_0>();
        if w.byte_len < rows * row_bytes
            || x.byte_len < cols * core::mem::size_of::<f32>()
            || y.byte_len < rows * core::mem::size_of::<f32>()
        {
            return Err(MetalError::InvalidDimensions(
                "buffer too small for GEMV dimensions".to_string(),
            ));
        }

        const THREADS_PER_TG: usize = 128; // SG=4 * 32 lanes
        const ROWS_PER_TG: usize = 32; // SG=4 * RG=4 * NR=2
        Self::require_fit(
            &self.pipe_gemv_q4_0,
            "gemv_q4_0_f32_lpr8_nr2",
            THREADS_PER_TG,
        )?;

        let args = GemvArgs {
            n_in: cols as u32,
            n_out: rows as u32,
            n_tok: 1,
            row_bytes: row_bytes as u32,
        };

        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;

        enc.setComputePipelineState(&self.pipe_gemv_q4_0);
        // SAFETY: Buffers `w`, `x`, `y` remain alive until `waitUntilCompleted` returns below;
        // `args` is copied inline by `setBytes_length_atIndex`.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&x.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&y.raw), 0, 2);
            let args_ptr = NonNull::from(&args).cast::<c_void>();
            enc.setBytes_length_atIndex(args_ptr, core::mem::size_of::<GemvArgs>(), 3);
        }

        let num_tgs = rows.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: num_tgs,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };

        for _ in 0..repeat.max(1) {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Single-dispatch convenience wrapper for `gemv_q4_0_f32`.
    pub fn gemv_q4_0_f32(
        &self,
        w: &MetalBuffer,
        x: &MetalBuffer,
        y: &mut MetalBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<(), MetalError> {
        self.gemv_q4_0_f32_repeat(w, x, y, rows, cols, 1)
    }

    /// Runs multi-token (`1..=8`) shared-weight `Q4_0 x F32` GEMV on Metal GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_0_f32_repeat(
        &self,
        w: &MetalBuffer,
        x: &MetalBuffer,
        y: &mut MetalBuffer,
        rows: usize,
        cols: usize,
        n_tokens: usize,
        repeat: usize,
    ) -> Result<(), MetalError> {
        if cols == 0 || !cols.is_multiple_of(QK4_0) || rows == 0 || n_tokens == 0 || n_tokens > 8 {
            return Err(MetalError::InvalidDimensions(format!(
                "rows={rows}, cols={cols}, n_tokens={n_tokens} (n_tokens must be 1..=8)"
            )));
        }
        let cols_blocks = cols / QK4_0;
        let row_bytes = cols_blocks * core::mem::size_of::<BlockQ4_0>();
        if w.byte_len < rows * row_bytes
            || x.byte_len < n_tokens * cols * core::mem::size_of::<f32>()
            || y.byte_len < n_tokens * rows * core::mem::size_of::<f32>()
        {
            return Err(MetalError::InvalidDimensions(
                "buffer too small for batched GEMV dimensions".to_string(),
            ));
        }

        const THREADS_PER_TG: usize = 128; // SG=4 * 32 lanes
        const ROWS_PER_TG: usize = 16; // SG=4 * RG=4
        Self::require_fit(&self.pipe_gemm_q4_0, "gemm_q4_0_f32_lpr8", THREADS_PER_TG)?;

        let args = GemvArgs {
            n_in: cols as u32,
            n_out: rows as u32,
            n_tok: n_tokens as u32,
            row_bytes: row_bytes as u32,
        };

        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;

        enc.setComputePipelineState(&self.pipe_gemm_q4_0);
        // SAFETY: Buffers remain alive until `waitUntilCompleted` returns below.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&x.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&y.raw), 0, 2);
            let args_ptr = NonNull::from(&args).cast::<c_void>();
            enc.setBytes_length_atIndex(args_ptr, core::mem::size_of::<GemvArgs>(), 3);
        }

        let num_tgs = rows.div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: num_tgs,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };

        for _ in 0..repeat.max(1) {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Runs `Q5_K x F32` GEMV (`n_tokens == 1`) or batched GEMM (`n_tokens > 1`) on Metal GPU.
    pub fn gemm_q5_k_f32(
        &self,
        w: &MetalBuffer,
        x: &MetalBuffer,
        y: &mut MetalBuffer,
        rows: usize,
        cols: usize,
        n_tokens: usize,
    ) -> Result<(), MetalError> {
        if cols == 0 || !cols.is_multiple_of(QK_K) || rows == 0 || n_tokens == 0 {
            return Err(MetalError::InvalidDimensions(format!(
                "rows={rows}, cols={cols}, n_tokens={n_tokens}"
            )));
        }
        let row_bytes = (cols / QK_K) * core::mem::size_of::<BlockQ5K>();
        if w.byte_len < rows * row_bytes
            || x.byte_len < n_tokens * cols * 4
            || y.byte_len < n_tokens * rows * 4
        {
            return Err(MetalError::InvalidDimensions(
                "buffer too small for Q5_K GEMV/GEMM".to_string(),
            ));
        }

        const THREADS: usize = 128;
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;

        if n_tokens == 1 {
            Self::require_fit(&self.pipe_gemv_q5_k, "gemv_q5_k_res", THREADS)?;
            let args = OpsGemvArgs {
                n_in: cols as u32,
                n_out: rows as u32,
                add_residual: 0,
                row_bytes: row_bytes as u32,
            };
            enc.setComputePipelineState(&self.pipe_gemv_q5_k);
            // SAFETY: Buffers remain alive until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&w.raw), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&x.raw), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&y.raw), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<OpsGemvArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows.div_ceil(8),
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
            Self::require_fit(&self.pipe_gemm_q5_k, "gemm_q5_k_res", THREADS)?;
            let args = OpsGemmArgs {
                n_in: cols as u32,
                n_out: rows as u32,
                n_tok: n_tokens as u32,
                add_residual: 0,
                row_bytes: row_bytes as u32,
            };
            enc.setComputePipelineState(&self.pipe_gemm_q5_k);
            // SAFETY: Buffers remain alive until `waitUntilCompleted`.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&w.raw), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&x.raw), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&y.raw), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<OpsGemmArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: rows.div_ceil(4),
                    height: n_tokens.div_ceil(8),
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Runs hardware-accelerated transposed quantized GEMM on Metal GPU (`Phase T4`):
    /// `dx[n_tokens, cols] = (accum ? dx : 0) + dy[n_tokens, rows] * W[rows, cols]`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_transposed_f32_repeat(
        &self,
        dtype: DType,
        w: &MetalBuffer,
        dy: &MetalBuffer,
        dx: &mut MetalBuffer,
        rows: usize,
        cols: usize,
        n_tokens: usize,
        accum: bool,
        repeat: usize,
    ) -> Result<(), MetalError> {
        if rows == 0 || cols == 0 || n_tokens == 0 {
            return Err(MetalError::InvalidDimensions(format!(
                "rows={rows}, cols={cols}, n_tokens={n_tokens} must be non-zero"
            )));
        }
        let align_req = match dtype {
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => 32,
            DType::Q5K | DType::Q6K => QK_K,
            DType::BF16 | DType::F32 => 1,
            other => return Err(MetalError::UnsupportedDType(other)),
        };
        if !cols.is_multiple_of(align_req) {
            return Err(MetalError::InvalidDimensions(format!(
                "cols={cols} must be a multiple of {align_req} for {dtype:?}"
            )));
        }
        let row_bytes = dtype.row_bytes(cols);
        if w.byte_len < rows * row_bytes
            || dy.byte_len < n_tokens * rows * 4
            || dx.byte_len < n_tokens * cols * 4
        {
            return Err(MetalError::InvalidDimensions(
                "buffer too small for transposed GEMM".to_string(),
            ));
        }

        let (pipe, kname): (&ProtocolObject<dyn MTLComputePipelineState>, &'static str) =
            match dtype {
                DType::Q4_0 => (&self.pipe_bwd_q4_0, "gemm_q4_0_transposed"),
                DType::Q4_1 => (&self.pipe_bwd_q4_1, "gemm_q4_1_transposed"),
                DType::Q8_0 => (&self.pipe_bwd_q8_0, "gemm_q8_0_transposed"),
                DType::Q5K => (&self.pipe_bwd_q5_k, "gemm_q5_k_transposed"),
                DType::Q6K => (&self.pipe_bwd_q6_k, "gemm_q6_k_transposed"),
                DType::BF16 => (&self.pipe_bwd_bf16, "gemm_bf16_transposed"),
                DType::F32 => (&self.pipe_bwd_f32, "gemm_f32_transposed"),
                other => return Err(MetalError::UnsupportedDType(other)),
            };
        Self::require_fit(pipe, kname, 128)?;

        let args = OpsGemmArgs {
            n_in: cols as u32,
            n_out: rows as u32,
            n_tok: n_tokens as u32,
            add_residual: u32::from(accum),
            row_bytes: row_bytes as u32,
        };

        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        enc.setComputePipelineState(pipe);
        // SAFETY: Buffers remain alive until `waitUntilCompleted` below.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&w.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&dy.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&dx.raw), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<OpsGemmArgs>(),
                3,
            );
        }
        let grid = MTLSize {
            width: cols.div_ceil(32),
            height: n_tokens.div_ceil(32),
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        for _ in 0..repeat.max(1) {
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Single-dispatch wrapper for [`Self::matmul_transposed_f32_repeat`].
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_transposed_f32(
        &self,
        dtype: DType,
        w: &MetalBuffer,
        dy: &MetalBuffer,
        dx: &mut MetalBuffer,
        rows: usize,
        cols: usize,
        n_tokens: usize,
        accum: bool,
    ) -> Result<(), MetalError> {
        self.matmul_transposed_f32_repeat(dtype, w, dy, dx, rows, cols, n_tokens, accum, 1)
    }

    /// Executes dynamic GPU linear LoRA `y += scale * (x @ A^T) @ B^T` (`B_T` has layout `[rank, d_out]`).
    #[allow(clippy::too_many_arguments)]
    pub fn lora_linear_f32(
        &self,
        a: &MetalBuffer,
        b_t: &MetalBuffer,
        x: &MetalBuffer,
        h_scratch: &mut MetalBuffer,
        y: &mut MetalBuffer,
        d_in: usize,
        d_out: usize,
        rank: usize,
        n_tok: usize,
        scale: f32,
    ) -> Result<(), MetalError> {
        if d_in == 0 || d_out == 0 || rank == 0 || n_tok == 0 {
            return Err(MetalError::InvalidDimensions(
                "lora_linear_f32 requires non-zero dimensions".to_string(),
            ));
        }
        Self::require_fit(&self.pipe_lora_down, "lora_down_f32", 128)?;
        let tg_up = d_out.clamp(1, 256);
        Self::require_fit(&self.pipe_lora_up, "lora_up_axpy_f32", tg_up)?;

        let args = LoraLinearArgs {
            d_in: d_in as u32,
            d_out: d_out as u32,
            rank: rank as u32,
            n_tok: n_tok as u32,
            scale,
        };
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;

        enc.setComputePipelineState(&self.pipe_lora_down);
        // SAFETY: Buffers remain alive until `waitUntilCompleted` below.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&x.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&h_scratch.raw), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<LoraLinearArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n_tok * rank).div_ceil(4),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        enc.setComputePipelineState(&self.pipe_lora_up);
        // SAFETY: Buffers remain alive until `waitUntilCompleted` below.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&b_t.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&h_scratch.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&y.raw), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<LoraLinearArgs>(),
                3,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: d_out.div_ceil(tg_up),
                height: n_tok,
                depth: 1,
            },
            MTLSize {
                width: tg_up,
                height: 1,
                depth: 1,
            },
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Executes dynamic GPU embedding LoRA `x[t] += scale * A[tok[t]] @ B_T`.
    #[allow(clippy::too_many_arguments)]
    pub fn lora_embed_f32(
        &self,
        a: &MetalBuffer,
        b_t: &MetalBuffer,
        token_ids: &MetalBuffer,
        x: &mut MetalBuffer,
        d_out: usize,
        rank: usize,
        n_tok: usize,
        scale: f32,
    ) -> Result<(), MetalError> {
        let tg_x = d_out.clamp(1, 256);
        Self::require_fit(&self.pipe_lora_embed, "lora_embed_f32", tg_x)?;
        let args = LoraEmbedArgs {
            d_out: d_out as u32,
            rank: rank as u32,
            n_tok: n_tok as u32,
            token_id: 0,
            use_tok_buf: 1,
            scale,
        };
        let cmd = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::CommandEncodeFailed)?;
        let enc = cmd
            .computeCommandEncoder()
            .ok_or(MetalError::CommandEncodeFailed)?;
        enc.setComputePipelineState(&self.pipe_lora_embed);
        // SAFETY: Buffers remain alive until `waitUntilCompleted` below.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a.raw), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_t.raw), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&token_ids.raw), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&x.raw), 0, 3);
            enc.setBytes_length_atIndex(
                NonNull::from(&args).cast::<c_void>(),
                core::mem::size_of::<LoraEmbedArgs>(),
                4,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: d_out.div_ceil(tg_x),
                height: n_tok,
                depth: 1,
            },
            MTLSize {
                width: tg_x,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }
}
