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
use tm_core::{BlockQ4_0, QK4_0};

use crate::MetalError;

const GEMV_Q4_0_MSL: &str = include_str!("shaders/gemv_q4_0.metal");

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

        let pipe_gemv_q4_0 = Self::build_pipeline(&device, &library, "gemv_q4_0_f32_lpr8_nr2")?;
        let pipe_gemm_q4_0 = Self::build_pipeline(&device, &library, "gemm_q4_0_f32_lpr8")?;

        Ok(Self {
            device,
            queue,
            pipe_gemv_q4_0,
            pipe_gemm_q4_0,
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
}
