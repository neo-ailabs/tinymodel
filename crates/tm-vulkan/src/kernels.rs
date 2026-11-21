//! Quantized matrix and attention kernels (V1 GEMV + V3 Cooperative Matrix & tiled FlashAttention).

use tm_core::{BlockQ4_0, DType, QK4_0};

use crate::VulkanError;
use crate::context::{VkBuffer, VulkanContext};

const SPV_Q4_0_SOA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_0_soa.spv"));
const SPV_Q4_0_AOS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_0_aos.spv"));
const SPV_GEMM_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_tiled.spv"));
const SPV_GEMM_KQUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_kquant.spv"));
const SPV_GEMM_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_f16.spv"));
const SPV_GEMM_PAIR_SILU_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_pair_silu_coopmat_f16.spv"));
const SPV_QUANTIZE_Q8_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantize_q8_1.spv"));
const SPV_GEMM_COOPMAT_I8: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_i8.spv"));
const SPV_FA_COOPMAT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_coopmat.spv"));
const SPV_FA_GQA_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_gqa_tiled.spv"));

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

pub(crate) fn dtype_code(dt: DType) -> Result<u32, VulkanError> {
    match dt {
        DType::Q4_0 => Ok(0),
        DType::Q4_1 => Ok(1),
        DType::Q8_0 => Ok(2),
        DType::Q5K => Ok(3),
        DType::Q6K => Ok(4),
        DType::F32 => Ok(5),
        other => Err(VulkanError::UnsupportedDType(other)),
    }
}

/// Normalizes a 2D `[rows, cols]` GGUF weight slice into a GPU-native dtype:
/// - `Q4_0 / Q4_1 / Q8_0 / Q5K / Q6K / F32`: zero-copy borrowed slice.
/// - `Q4_0x4` (M3 offline 4-row interleaved `Q4_0`): 0-ULP lossless byte uninterleave back to `Q4_0`
///   so `CoopMatF16`, `CoopMatI8`, and fused QKV/SwiGLU GEMV run at full native speed.
/// - `F16 / BF16 / Q2_0 / TQ2_0 / Iq2Xxs / Iq3Xxs / Iq4Nl`: load-time dequantization to `F32`
///   so M2/M4 converted or pruned GGUFs also execute seamlessly on Vulkan.
pub(crate) fn normalize_vk_mat_bytes<'a>(
    dtype: DType,
    bytes: &'a [u8],
    rows: usize,
    cols: usize,
) -> Result<(DType, std::borrow::Cow<'a, [u8]>), VulkanError> {
    match dtype {
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K | DType::F32 => {
            Ok((dtype, std::borrow::Cow::Borrowed(bytes)))
        }
        DType::Q4_0x4 => {
            if !rows.is_multiple_of(4) || !cols.is_multiple_of(QK4_0) {
                return Err(VulkanError::InvalidDimensions(format!(
                    "Q4_0x4 matrix {rows}x{cols} must have rows%4==0 and cols%32==0"
                )));
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
            Ok((DType::Q4_0, std::borrow::Cow::Owned(dst)))
        }
        DType::F16
        | DType::BF16
        | DType::Q2_0
        | DType::TQ2_0
        | DType::Iq2Xxs
        | DType::Iq3Xxs
        | DType::Iq4Nl => {
            let rb = dtype.row_bytes(cols);
            let mut f32_vals = vec![0.0f32; rows * cols];
            for r in 0..rows {
                tm_core::dequantize_row_bytes(
                    dtype,
                    &bytes[r * rb..(r + 1) * rb],
                    &mut f32_vals[r * cols..(r + 1) * cols],
                );
            }
            let mut out = Vec::with_capacity(f32_vals.len() * 4);
            for &v in &f32_vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok((DType::F32, std::borrow::Cow::Owned(out)))
        }
        other => Err(VulkanError::UnsupportedDType(other)),
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

    /// Portable LDS/subgroup batched GEMM fallback (`gemm_tiled` / `gemm_kquant`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fallback_raw(
        &self,
        dtype: DType,
        w: &VkBuffer,
        x: &VkBuffer,
        y: &VkBuffer,
        rows: usize,
        cols: usize,
        n_tok: usize,
        add_residual: bool,
        repeat: usize,
    ) -> Result<(), VulkanError> {
        let dt = dtype_code(dtype)?;
        let row_bytes = dtype.row_bytes(cols);
        let sg = self.info().subgroup_size;
        let (pipe, gx, gy) = match dtype {
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => (
                self.pipeline("gemm_tiled", dt, SPV_GEMM_TILED, 3, 20, &[128, dt])?,
                (rows as u32).div_ceil(64),
                (n_tok as u32).div_ceil(32),
            ),
            DType::Q5K | DType::Q6K => {
                let rows_per_wg = (128 / sg).max(1);
                (
                    self.pipeline("gemm_kquant", dt, SPV_GEMM_KQUANT, 3, 20, &[128, dt])?,
                    (rows as u32).div_ceil(rows_per_wg),
                    (n_tok as u32).div_ceil(8),
                )
            }
            other => return Err(VulkanError::UnsupportedDType(other)),
        };
        let push = [
            cols as u32,
            rows as u32,
            n_tok as u32,
            u32::from(add_residual),
            (row_bytes / 2) as u32,
        ];
        self.run(|rec| {
            for _ in 0..repeat.max(1) {
                rec.dispatch_xy(&pipe, &[w, x, y], &push, gx, gy)?;
            }
            Ok(())
        })
    }

    /// `16x16x16 f16*f16+f32->f32` Cooperative Matrix batched GEMM (`bn` in `{64, 128}`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_coopmat_f16_raw(
        &self,
        dtype: DType,
        w: &VkBuffer,
        x: &VkBuffer,
        y: &VkBuffer,
        rows: usize,
        cols: usize,
        n_tok: usize,
        bn: u32,
        add_residual: bool,
        repeat: usize,
    ) -> Result<(), VulkanError> {
        if !self.info().coopmat_f16_f32acc {
            return Err(VulkanError::MissingFeature {
                device: self.info().name.clone(),
                feature: "16x16x16 f16*f16+f32->f32 cooperativeMatrix",
            });
        }
        if bn != 64 && bn != 128 {
            return Err(VulkanError::InvalidDimensions(format!(
                "coopmat_f16 bn={bn} (must be 64 or 128)"
            )));
        }
        let dt = dtype_code(dtype)?;
        let (wg, bm, eff_bn) = if bn == 128 {
            (8 * self.info().subgroup_size, 128u32, 128u32)
        } else {
            (4 * self.info().subgroup_size, 64u32, 64u32)
        };
        let key = dt | (eff_bn << 8) | (bm << 16) | (wg << 24);
        let pipe = self.pipeline(
            "gemm_coopmat_f16",
            key,
            SPV_GEMM_COOPMAT_F16,
            3,
            20,
            &[wg, dt, bm, eff_bn],
        )?;
        let row_bytes = dtype.row_bytes(cols);
        let push = [
            cols as u32,
            rows as u32,
            n_tok as u32,
            u32::from(add_residual),
            (row_bytes / 2) as u32,
        ];
        let gx = (rows as u32).div_ceil(eff_bn);
        let gy = (n_tok as u32).div_ceil(bm);
        self.run(|rec| {
            for _ in 0..repeat.max(1) {
                rec.dispatch_xy(&pipe, &[w, x, y], &push, gx, gy)?;
            }
            Ok(())
        })
    }

    /// Fused `Gate + Up + SwiGLU` `16x16x16 f16*f16+f32->f32` Cooperative Matrix GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_pair_silu_coopmat_f16_raw(
        &self,
        dtype: DType,
        w_gate: &VkBuffer,
        w_up: &VkBuffer,
        x: &VkBuffer,
        y: &VkBuffer,
        rows: usize,
        cols: usize,
        n_tok: usize,
        repeat: usize,
    ) -> Result<(), VulkanError> {
        if !self.info().coopmat_f16_f32acc {
            return Err(VulkanError::MissingFeature {
                device: self.info().name.clone(),
                feature: "16x16x16 f16*f16+f32->f32 cooperativeMatrix",
            });
        }
        if !matches!(dtype, DType::Q4_0 | DType::Q4_1 | DType::Q8_0) {
            return Err(VulkanError::UnsupportedDType(dtype));
        }
        let dt = dtype_code(dtype)?;
        let wg = 8 * self.info().subgroup_size;
        let key = dt | (wg << 8);
        let pipe = self.pipeline(
            "gemm_pair_silu_coopmat_f16",
            key,
            SPV_GEMM_PAIR_SILU_COOPMAT_F16,
            4,
            16,
            &[wg, dt],
        )?;
        let row_bytes = dtype.row_bytes(cols);
        let push = [
            cols as u32,
            rows as u32,
            n_tok as u32,
            (row_bytes / 2) as u32,
        ];
        let gx = (rows as u32).div_ceil(64);
        let gy = (n_tok as u32).div_ceil(128);
        self.run(|rec| {
            for _ in 0..repeat.max(1) {
                rec.dispatch_xy(&pipe, &[w_gate, w_up, x, y], &push, gx, gy)?;
            }
            Ok(())
        })
    }

    /// Dynamic `FP32 -> Q8_1` activation quantization + `16x16x16 i8*i8+i32->i32` Cooperative Matrix GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_coopmat_i8_raw(
        &self,
        dtype: DType,
        w: &VkBuffer,
        x: &VkBuffer,
        x_q8: &VkBuffer,
        x_ds: &VkBuffer,
        y: &VkBuffer,
        rows: usize,
        cols: usize,
        n_tok: usize,
        add_residual: bool,
        repeat: usize,
    ) -> Result<(), VulkanError> {
        if !self.info().coopmat_i8_i32acc {
            return Err(VulkanError::MissingFeature {
                device: self.info().name.clone(),
                feature: "16x16x16 i8*i8+i32->i32 cooperativeMatrix",
            });
        }
        if !matches!(dtype, DType::Q4_0 | DType::Q4_1 | DType::Q8_0) {
            return Err(VulkanError::UnsupportedDType(dtype));
        }
        let dt = dtype_code(dtype)?;
        let wg = 4 * self.info().subgroup_size;
        let q_pipe = self.pipeline("quantize_q8_1", 256, SPV_QUANTIZE_Q8_1, 3, 4, &[256])?;
        let m_pipe = self.pipeline(
            "gemm_coopmat_i8",
            dt | (wg << 8),
            SPV_GEMM_COOPMAT_I8,
            4,
            20,
            &[wg, dt],
        )?;
        let n_blocks = ((n_tok * cols) / 32) as u32;
        let q_groups = n_blocks.div_ceil(8);
        let row_bytes = dtype.row_bytes(cols);
        let push = [
            cols as u32,
            rows as u32,
            n_tok as u32,
            u32::from(add_residual),
            (row_bytes / 2) as u32,
        ];
        let gx = (rows as u32).div_ceil(64);
        let gy = (n_tok as u32).div_ceil(64);
        self.run(|rec| {
            for _ in 0..repeat.max(1) {
                rec.dispatch(&q_pipe, &[x, x_q8, x_ds], &[n_blocks], q_groups)?;
                rec.dispatch_xy(&m_pipe, &[w, x_q8, x_ds, y], &push, gx, gy)?;
            }
            Ok(())
        })
    }

    /// Batched causal GQA FlashAttention (`use_coopmat = true` -> `fa_coopmat`, `false` -> `fa_gqa_tiled`).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_batch_raw(
        &self,
        q: &VkBuffer,
        k_cache: &VkBuffer,
        v_cache: &VkBuffer,
        out: &VkBuffer,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        base_pos: usize,
        n_tok: usize,
        use_coopmat: bool,
        repeat: usize,
    ) -> Result<(), VulkanError> {
        let sg = self.info().subgroup_size;
        let push = [
            n_head as u32,
            n_head_kv as u32,
            head_dim as u32,
            base_pos as u32,
            n_tok as u32,
        ];
        let (pipe, gx, gy) = if use_coopmat {
            if !self.info().coopmat_f16_f32acc || sg != 32 || !head_dim.is_multiple_of(16) {
                return Err(VulkanError::MissingFeature {
                    device: self.info().name.clone(),
                    feature: "wave32 16x16x16 f16 cooperativeMatrix FlashAttention",
                });
            }
            let wg = 128u32;
            (
                self.pipeline("fa_coopmat", wg, SPV_FA_COOPMAT, 4, 20, &[wg])?,
                n_head as u32,
                (n_tok as u32).div_ceil(64),
            )
        } else {
            let wg = 8 * sg;
            let n_vec = ((head_dim / 4).div_ceil(sg as usize)).max(1) as u32;
            (
                self.pipeline(
                    "fa_gqa_tiled",
                    wg | (n_vec << 16),
                    SPV_FA_GQA_TILED,
                    4,
                    20,
                    &[wg, n_vec],
                )?,
                n_head as u32,
                (n_tok as u32).div_ceil(8),
            )
        };
        self.run(|rec| {
            for _ in 0..repeat.max(1) {
                rec.dispatch_xy(&pipe, &[q, k_cache, v_cache, out], &push, gx, gy)?;
            }
            Ok(())
        })
    }
}
