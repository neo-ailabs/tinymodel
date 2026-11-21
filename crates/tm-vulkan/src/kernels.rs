//! Quantized matrix kernels (V1: Q4_0 x F32 GEMV / shared-weight multi-token GEMV).

use tm_core::{BlockQ4_0, QK4_0};

use crate::VulkanError;
use crate::context::{VkBuffer, VulkanContext};

const SPV_Q4_0_SOA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_0_soa.spv"));
const SPV_Q4_0_AOS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_0_aos.spv"));

/// Maximum tokens per shared-weight GEMV dispatch (matches the Metal backend).
pub const MAX_NTOK: usize = 8;
/// Default Q4_0 GEMV geometry: workgroup size and rows per subgroup.
pub const DEFAULT_GEMV_WG: u32 = 128;
pub const DEFAULT_GEMV_NR: u32 = 2;

/// GPU-side layout of a Q4_0 weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4Layout {
    /// Raw ggml `BlockQ4_0` array (18-byte blocks); allows zero-copy import.
    Aos,
    /// Repacked at upload: 16-byte nibble plane (uvec4 loads) + fp16 scale plane.
    Soa,
}

/// A Q4_0 weight matrix resident on the GPU.
pub struct VkMatQ4_0 {
    layout: Q4Layout,
    rows: usize,
    cols: usize,
    q: VkBuffer,
    d: Option<VkBuffer>,
}

impl VkMatQ4_0 {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn layout(&self) -> Q4Layout {
        self.layout
    }
    /// Bytes read from VRAM per full pass over the matrix.
    pub fn weight_bytes(&self) -> u64 {
        (self.rows * self.cols / QK4_0 * core::mem::size_of::<BlockQ4_0>()) as u64
    }
}

impl VulkanContext {
    /// Uploads a row-major `rows x cols` Q4_0 matrix (`cols % 32 == 0`).
    pub fn upload_q4_0(
        &self,
        blocks: &[BlockQ4_0],
        rows: usize,
        cols: usize,
        layout: Q4Layout,
    ) -> Result<VkMatQ4_0, VulkanError> {
        if !cols.is_multiple_of(QK4_0) || blocks.len() != rows * cols / QK4_0 || rows == 0 {
            return Err(VulkanError::InvalidDimensions(format!(
                "Q4_0 matrix {rows}x{cols} with {} blocks",
                blocks.len()
            )));
        }
        match layout {
            Q4Layout::Aos => Ok(VkMatQ4_0 {
                layout,
                rows,
                cols,
                q: self.upload(blocks)?,
                d: None,
            }),
            Q4Layout::Soa => {
                let mut qs = Vec::with_capacity(blocks.len() * 16);
                let mut ds = Vec::with_capacity(blocks.len());
                for b in blocks {
                    qs.extend_from_slice(&b.qs);
                    ds.push(b.d);
                }
                Ok(VkMatQ4_0 {
                    layout,
                    rows,
                    cols,
                    q: self.upload_bytes(&qs)?,
                    d: Some(self.upload(&ds)?),
                })
            }
        }
    }

    /// Overrides the Q4_0 GEMV geometry (benchmark/tuning hook). `wg` must be a
    /// multiple of the subgroup size; `nr` is 1..=4.
    pub fn set_q4_0_gemv_config(&self, wg: u32, nr: u32) -> Result<(), VulkanError> {
        if !(1..=4).contains(&nr) || wg == 0 || !wg.is_multiple_of(self.info().subgroup_size) {
            return Err(VulkanError::InvalidDimensions(format!(
                "gemv config wg={wg} nr={nr} (subgroup {})",
                self.info().subgroup_size
            )));
        }
        if let Ok(mut c) = self.gemv_cfg.lock() {
            *c = (wg, nr);
        }
        Ok(())
    }

    fn gemv_config(&self) -> (u32, u32) {
        self.gemv_cfg
            .lock()
            .map(|c| *c)
            .unwrap_or((DEFAULT_GEMV_WG, DEFAULT_GEMV_NR))
    }

    /// `y[t * rows + r] = dot(W[r], x[t])` for `t < n_tok` (`n_tok <= 8`), dispatched
    /// `repeat` times in one submission (`repeat = 1` for normal use; >1 for benchmarks).
    pub fn gemm_q4_0_f32(
        &self,
        w: &VkMatQ4_0,
        x: &VkBuffer,
        y: &VkBuffer,
        n_tok: usize,
        repeat: usize,
    ) -> Result<(), VulkanError> {
        if n_tok == 0 || n_tok > MAX_NTOK {
            return Err(VulkanError::InvalidDimensions(format!(
                "n_tok={n_tok} (must be 1..={MAX_NTOK})"
            )));
        }
        let need_x = (n_tok * w.cols * 4) as u64;
        let need_y = (n_tok * w.rows * 4) as u64;
        if x.byte_len() < need_x || y.byte_len() < need_y {
            return Err(VulkanError::InvalidDimensions(format!(
                "x/y buffers too small: x {} < {need_x} or y {} < {need_y}",
                x.byte_len(),
                y.byte_len()
            )));
        }
        let (wg, nr) = self.gemv_config();
        let spec = [wg, nr, n_tok as u32];
        let key = n_tok as u32 | (nr << 8) | (wg << 16);
        let (pipe, bufs): (_, Vec<&VkBuffer>) = match (w.layout, &w.d) {
            (Q4Layout::Soa, Some(d)) => (
                self.pipeline("gemv_q4_0_soa", key, SPV_Q4_0_SOA, 4, 8, &spec)?,
                vec![&w.q, d, x, y],
            ),
            _ => (
                self.pipeline("gemv_q4_0_aos", key, SPV_Q4_0_AOS, 3, 8, &spec)?,
                vec![&w.q, x, y],
            ),
        };
        let rows_per_wg = (wg / self.info().subgroup_size).max(1) * nr;
        let groups = (w.rows as u32).div_ceil(rows_per_wg);
        let push = [w.cols as u32, w.rows as u32];
        self.run(|rec| {
            for _ in 0..repeat.max(1) {
                rec.dispatch(&pipe, &bufs, &push, groups)?;
            }
            Ok(())
        })
    }
}
