//! Vulkan GPU forward & transposed-backward quantized GEMM accelerator (`Phase T4`)
//! for `tinymodel finetune --vulkan` across AMD RDNA4, Intel Xe/UHD, Arm Mali-G715/G610, and lavapipe.
//!
//! Caches normalized `VkBuffer` weight matrices on demand (supporting `Q4_0`, `Q4_1`, `Q8_0`,
//! `Q5_K`, `Q6_K`, `F32`, lossless `Q4_0x4 -> Q4_0`, and `BF16`/`F16`/`Q2_0`/`TQ2_0`/`IQ* -> F32`)
//! and dispatches Cooperative Matrix (`16x16x16`) or `32x64x32` LDS-tiled forward + transposed
//! backward GEMMs using pre-allocated buffers.

use std::collections::HashMap;
use std::sync::Arc;

use tm_core::DType;
use tm_gguf::GgufFile;

use crate::VulkanError;
use crate::context::{VkBuffer, VulkanContext};
use crate::kernels::normalize_vk_mat_bytes;

/// Default minimum MAC count (`n_tok * rows * cols`) to amortize Vulkan queue submit + fence wait.
pub const DEFAULT_VK_MIN_MACS: usize = 4_194_304;

struct CachedVkMat {
    buf: VkBuffer,
    dtype: DType,
    rows: usize,
    cols: usize,
}

/// Cross-vendor Vulkan GPU quantized GEMM accelerator for QLoRA fine-tuning.
pub struct VulkanGpuAccel {
    ctx: VulkanContext,
    mats: HashMap<String, Option<CachedVkMat>>,
    in_buf: VkBuffer,
    out_buf: VkBuffer,
    min_macs: usize,
    gguf: Arc<GgufFile>,
}

impl VulkanGpuAccel {
    pub fn from_gguf(gguf: Arc<GgufFile>, max_seq: usize) -> Result<Self, VulkanError> {
        Self::from_gguf_with_context(VulkanContext::new()?, gguf, max_seq)
    }

    pub fn from_gguf_with_context(
        ctx: VulkanContext,
        gguf: Arc<GgufFile>,
        max_seq: usize,
    ) -> Result<Self, VulkanError> {
        let mut max_dim = 4096usize;
        for tname in gguf.tensor_names() {
            if let Some(info) = gguf.tensor_info(tname)
                && info.dims.len() == 2
            {
                max_dim = max_dim.max(info.dims[0]).max(info.dims[1]);
            }
        }
        let scratch_bytes = ((max_seq.max(1) * max_dim.max(1) * 4) as u64).max(64);
        let in_buf = ctx.create_device_slab(scratch_bytes)?;
        let out_buf = ctx.create_device_slab(scratch_bytes)?;
        Ok(Self {
            ctx,
            mats: HashMap::new(),
            in_buf,
            out_buf,
            min_macs: DEFAULT_VK_MIN_MACS,
            gguf,
        })
    }

    pub fn with_min_macs(mut self, min_macs: usize) -> Self {
        self.min_macs = min_macs;
        self
    }

    pub fn device_name(&self) -> String {
        self.ctx.device_name().to_string()
    }

    pub fn driver_info(&self) -> String {
        self.ctx.info().driver_info.clone()
    }

    fn ensure_scratch(&mut self, in_bytes: u64, out_bytes: u64) -> bool {
        if self.in_buf.byte_len() < in_bytes {
            let Ok(b) = self.ctx.create_device_slab(in_bytes) else {
                return false;
            };
            self.in_buf = b;
        }
        if self.out_buf.byte_len() < out_bytes {
            let Ok(b) = self.ctx.create_device_slab(out_bytes) else {
                return false;
            };
            self.out_buf = b;
        }
        true
    }

    fn ensure_mat_loaded(&mut self, tensor_name: &str, rows: usize, cols: usize) -> bool {
        if !self.mats.contains_key(tensor_name) {
            let lookup = if tensor_name == "output.weight"
                && self.gguf.tensor_info("output.weight").is_none()
            {
                "token_embd.weight"
            } else {
                tensor_name
            };
            let entry = (|| -> Option<CachedVkMat> {
                let (info, raw_bytes) = self.gguf.tensor_bytes(lookup, "vk_accel").ok()?;
                if info.dims.len() != 2 || info.dims[0] != cols || info.dims[1] != rows {
                    return None;
                }
                let (eff_dt, norm_bytes) =
                    normalize_vk_mat_bytes(info.dtype, raw_bytes, rows, cols).ok()?;
                let buf = self.ctx.create_device_slab(norm_bytes.len() as u64).ok()?;
                self.ctx.write_bytes(&buf, 0, &norm_bytes).ok()?;
                Some(CachedVkMat {
                    buf,
                    dtype: eff_dt,
                    rows,
                    cols,
                })
            })();
            self.mats.insert(tensor_name.to_string(), entry);
        }
        matches!(
            self.mats.get(tensor_name),
            Some(Some(m)) if m.rows == rows && m.cols == cols
        )
    }

    /// Forward `y[n_tok, rows] = x[n_tok, cols] @ W[rows, cols]^T` on Vulkan GPU.
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
        if x.len() < n_tok * cols || y.len() < n_tok * rows {
            return false;
        }
        let in_bytes = (n_tok * cols * 4) as u64;
        let out_bytes = (n_tok * rows * 4) as u64;
        if !self.ensure_scratch(in_bytes, out_bytes)
            || !self.ensure_mat_loaded(tensor_name, rows, cols)
        {
            return false;
        }

        // SAFETY: `f32` slice viewed as bytes for `in_bytes` bytes.
        let x_bytes =
            unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, in_bytes as usize) };
        if self.ctx.write_bytes(&self.in_buf, 0, x_bytes).is_err() {
            return false;
        }

        let Some(Some(mat)) = self.mats.get(tensor_name) else {
            return false;
        };
        let sg = self.ctx.info().subgroup_size;
        let use_cm = self.ctx.info().coopmat_f16_f32acc && (sg == 32 || sg == 64) && n_tok >= 16;
        let ok = if use_cm {
            let bn = if rows >= 512 && (8 * sg) <= self.ctx.info().max_workgroup_invocations {
                128
            } else {
                64
            };
            self.ctx
                .gemm_coopmat_f16_raw(
                    mat.dtype,
                    &mat.buf,
                    &self.in_buf,
                    &self.out_buf,
                    rows,
                    cols,
                    n_tok,
                    bn,
                    false,
                    1,
                )
                .is_ok()
        } else {
            self.ctx
                .gemm_fallback_raw(
                    mat.dtype,
                    &mat.buf,
                    &self.in_buf,
                    &self.out_buf,
                    rows,
                    cols,
                    n_tok,
                    false,
                    1,
                )
                .is_ok()
        };
        if !ok {
            return false;
        }

        let Ok(host_out) = self.ctx.read_f32(&self.out_buf, n_tok * rows) else {
            return false;
        };
        y[..n_tok * rows].copy_from_slice(&host_out);
        true
    }

    /// Transposed backward `dx[n_tok, cols] = (accum ? dx : 0) + dy[n_tok, rows] @ W[rows, cols]`
    /// on Vulkan GPU!
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
        if dy.len() < n_tok * rows || dx.len() < n_tok * cols {
            return false;
        }
        let in_bytes = (n_tok * rows * 4) as u64;
        let out_bytes = (n_tok * cols * 4) as u64;
        if !self.ensure_scratch(in_bytes, out_bytes)
            || !self.ensure_mat_loaded(tensor_name, rows, cols)
        {
            return false;
        }

        // SAFETY: `f32` slice viewed as bytes for `in_bytes` bytes.
        let dy_bytes =
            unsafe { core::slice::from_raw_parts(dy.as_ptr() as *const u8, in_bytes as usize) };
        if self.ctx.write_bytes(&self.in_buf, 0, dy_bytes).is_err() {
            return false;
        }
        if accum {
            // SAFETY: `dx` has at least `n_tok * cols` floats (`out_bytes` bytes).
            let dx_bytes = unsafe {
                core::slice::from_raw_parts(dx.as_ptr() as *const u8, out_bytes as usize)
            };
            if self.ctx.write_bytes(&self.out_buf, 0, dx_bytes).is_err() {
                return false;
            }
        }

        let Some(Some(mat)) = self.mats.get(tensor_name) else {
            return false;
        };
        if self
            .ctx
            .gemm_transposed_raw(
                mat.dtype,
                &mat.buf,
                &self.in_buf,
                &self.out_buf,
                rows,
                cols,
                n_tok,
                accum,
                1,
            )
            .is_err()
        {
            return false;
        }

        let Ok(host_dx) = self.ctx.read_f32(&self.out_buf, n_tok * cols) else {
            return false;
        };
        dx[..n_tok * cols].copy_from_slice(&host_dx);
        true
    }
}
