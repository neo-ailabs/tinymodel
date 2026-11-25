//! Generic Multimodal Vision Encoder (`mmproj` GGUF, `ADR-019`).
//!
//! Pure safe Rust (`#![forbid(unsafe_code)]`, inherited from `tm-models`) with zero external
//! dependencies, compatible with `llama.cpp` / `mtmd` `mmproj` GGUF files across all major
//! sub-8B VLM vision backbones:
//!
//! - **SigLIP / SigLIP-2 ViT**: `gemma3`, `gemma4`, `smolvlm`, `lfm2`, `mlp`, `linear`
//!   (2D patch conv, optional learned position embeddings with bilinear interpolation,
//!   Pre-LayerNorm or Pre-RMSNorm bidirectional self-attention, GELU/SiLU/SwiGLU/GeGLU FFN,
//!   2D spatial average pooling / pixel-shuffle / MLP projector).
//! - **Qwen-VL ViT**: `qwen2vl_merger`, `qwen2.5vl_merger`, `qwen3vl_merger`, `qwen35_merger`
//!   (2D or temporal-2 3D patch conv, 2D Vision RoPE `(row, col)`, Pre-RMSNorm / Pre-LayerNorm,
//!   2×2 spatial patch merger + 2-layer GELU MLP projector, 3D M-RoPE position metadata).
//! - **Pixtral / Mistral-3 ViT**: `pixtral`
//!   (2D patch conv, 2D Vision RoPE, Pre-RMSNorm, SwiGLU FFN, spatial patch merger + MLP).
//!
//! Also includes zero-dependency image decoding (PPM `P5`/`P6`, uncompressed 24/32-bit BMP),
//! Keys bicubic (`a = -0.5`) & bilinear resampling, and Qwen-VL `smart_resize_dims`.

use std::path::Path;

use tm_core::{DType, dequantize_row_bytes};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, ThreadPool, gelu_tanh_f32, matmul_f32, matvec_f32,
    rms_norm, softmax_inplace,
};
use tm_gguf::{GgufError, GgufFile};

use crate::llama::{ModelError, RawMatView};

// ============================================================================
// Image Container, Zero-Dependency Decoders & Preprocessing
// ============================================================================

/// Interleaved 8-bit RGB image (`data.len() == width * height * 3`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Constructs a validated RGB image from interleaved `[R, G, B, ...]` bytes.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, ModelError> {
        if width == 0 || height == 0 {
            return Err(ModelError::InvalidConfig(format!(
                "image dimensions must be non-zero, got {width}x{height}"
            )));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| ModelError::InvalidConfig("image dimensions overflow".into()))?;
        if data.len() != expected {
            return Err(ModelError::InvalidConfig(format!(
                "RGB image byte length mismatch: expected {expected} ({width}x{height}x3), got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Decodes a binary PPM (`P6` RGB) or PGM (`P5` grayscale) image from raw bytes.
    pub fn from_ppm_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        let mut pos = 0usize;
        let next_token = |pos: &mut usize| -> Option<String> {
            while *pos < bytes.len() {
                let b = bytes[*pos];
                if b.is_ascii_whitespace() {
                    *pos += 1;
                } else if b == b'#' {
                    while *pos < bytes.len() && bytes[*pos] != b'\n' {
                        *pos += 1;
                    }
                } else {
                    break;
                }
            }
            if *pos >= bytes.len() {
                return None;
            }
            let start = *pos;
            while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() && bytes[*pos] != b'#' {
                *pos += 1;
            }
            Some(String::from_utf8_lossy(&bytes[start..*pos]).into_owned())
        };

        let magic = next_token(&mut pos)
            .ok_or_else(|| ModelError::InvalidConfig("empty PPM/PGM image".into()))?;
        let is_rgb = match magic.as_str() {
            "P6" => true,
            "P5" => false,
            other => {
                return Err(ModelError::InvalidConfig(format!(
                    "unsupported PNM magic '{other}' (expected P6 or P5)"
                )));
            }
        };
        let width: usize = next_token(&mut pos)
            .ok_or_else(|| ModelError::InvalidConfig("PPM missing width".into()))?
            .parse()
            .map_err(|_| ModelError::InvalidConfig("invalid PPM width".into()))?;
        let height: usize = next_token(&mut pos)
            .ok_or_else(|| ModelError::InvalidConfig("PPM missing height".into()))?
            .parse()
            .map_err(|_| ModelError::InvalidConfig("invalid PPM height".into()))?;
        let maxval: usize = next_token(&mut pos)
            .ok_or_else(|| ModelError::InvalidConfig("PPM missing maxval".into()))?
            .parse()
            .map_err(|_| ModelError::InvalidConfig("invalid PPM maxval".into()))?;
        if maxval == 0 || maxval > 255 {
            return Err(ModelError::InvalidConfig(format!(
                "unsupported PPM maxval {maxval} (expected 1..=255)"
            )));
        }
        // Exactly one whitespace byte separates the header from binary pixel data.
        if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
            return Err(ModelError::InvalidConfig(
                "PPM missing whitespace separator before pixel data".into(),
            ));
        }
        pos += 1;

        let n_px = width
            .checked_mul(height)
            .ok_or_else(|| ModelError::InvalidConfig("PPM dimensions overflow".into()))?;
        let payload = &bytes[pos..];
        if is_rgb {
            let need = n_px * 3;
            if payload.len() < need {
                return Err(ModelError::InvalidConfig(format!(
                    "truncated P6 PPM payload: need {need} bytes, got {}",
                    payload.len()
                )));
            }
            let mut data = payload[..need].to_vec();
            if maxval != 255 {
                for v in &mut data {
                    *v = ((*v as usize * 255 + maxval / 2) / maxval).min(255) as u8;
                }
            }
            Self::new(width, height, data)
        } else {
            if payload.len() < n_px {
                return Err(ModelError::InvalidConfig(format!(
                    "truncated P5 PGM payload: need {n_px} bytes, got {}",
                    payload.len()
                )));
            }
            let mut data = Vec::with_capacity(n_px * 3);
            for &g in &payload[..n_px] {
                let v = if maxval == 255 {
                    g
                } else {
                    ((g as usize * 255 + maxval / 2) / maxval).min(255) as u8
                };
                data.push(v);
                data.push(v);
                data.push(v);
            }
            Self::new(width, height, data)
        }
    }

    /// Decodes an uncompressed 24-bit or 32-bit BMP image from raw bytes.
    pub fn from_bmp_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < 54 || &bytes[0..2] != b"BM" {
            return Err(ModelError::InvalidConfig(
                "invalid BMP header (expected 'BM' and >= 54 bytes)".into(),
            ));
        }
        let data_offset = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
        let width_i32 = i32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);
        let height_i32 = i32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]);
        let bpp = u16::from_le_bytes([bytes[28], bytes[29]]);
        let compression = u32::from_le_bytes([bytes[30], bytes[31], bytes[32], bytes[33]]);
        if width_i32 <= 0 || height_i32 == 0 {
            return Err(ModelError::InvalidConfig(format!(
                "invalid BMP dimensions {width_i32}x{height_i32}"
            )));
        }
        if compression != 0 && compression != 3 {
            return Err(ModelError::InvalidConfig(format!(
                "unsupported BMP compression {compression} (only uncompressed BI_RGB supported)"
            )));
        }
        if bpp != 24 && bpp != 32 {
            return Err(ModelError::InvalidConfig(format!(
                "unsupported BMP bit depth {bpp} (expected 24 or 32)"
            )));
        }
        let width = width_i32 as usize;
        let top_down = height_i32 < 0;
        let height = height_i32.unsigned_abs() as usize;
        let bytes_per_px = (bpp / 8) as usize;
        let row_stride = (width * bytes_per_px).div_ceil(4) * 4;
        let need = data_offset + row_stride * height;
        if bytes.len() < need {
            return Err(ModelError::InvalidConfig(format!(
                "truncated BMP pixel data: need {need} bytes, got {}",
                bytes.len()
            )));
        }
        let mut data = vec![0u8; width * height * 3];
        for y in 0..height {
            let src_y = if top_down { y } else { height - 1 - y };
            let src_row = &bytes[data_offset + src_y * row_stride..];
            let dst_row = &mut data[y * width * 3..(y + 1) * width * 3];
            for x in 0..width {
                let s = x * bytes_per_px;
                let d = x * 3;
                // BMP stores BGR(A)
                dst_row[d] = src_row[s + 2];
                dst_row[d + 1] = src_row[s + 1];
                dst_row[d + 2] = src_row[s];
            }
        }
        Self::new(width, height, data)
    }

    /// Loads an RGB image from disk (supports PPM `P6`, PGM `P5`, and uncompressed BMP).
    pub fn from_file(path: &Path) -> Result<Self, ModelError> {
        let bytes = std::fs::read(path).map_err(|e| {
            ModelError::InvalidConfig(format!("failed to read image '{}': {e}", path.display()))
        })?;
        if bytes.len() >= 2 && (&bytes[0..2] == b"P6" || &bytes[0..2] == b"P5") {
            Self::from_ppm_bytes(&bytes)
        } else if bytes.len() >= 2 && &bytes[0..2] == b"BM" {
            Self::from_bmp_bytes(&bytes)
        } else {
            Err(ModelError::InvalidConfig(format!(
                "unsupported image format for '{}' (native decoder supports PPM P6/P5 and BMP; convert PNG/JPEG via CLI)",
                path.display()
            )))
        }
    }

    /// Resizes using bilinear interpolation, returning interleaved RGB `f32` in `[0.0, 255.0]`.
    pub fn resize_bilinear_f32(&self, dst_w: usize, dst_h: usize) -> Vec<f32> {
        assert!(dst_w > 0 && dst_h > 0);
        if dst_w == self.width && dst_h == self.height {
            return self.data.iter().map(|&v| v as f32).collect();
        }
        let mut out = vec![0.0f32; dst_w * dst_h * 3];
        let scale_x = self.width as f64 / dst_w as f64;
        let scale_y = self.height as f64 / dst_h as f64;
        for dy in 0..dst_h {
            let sy = ((dy as f64 + 0.5) * scale_y - 0.5).clamp(0.0, (self.height - 1) as f64);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let wy = (sy - y0 as f64) as f32;
            for dx in 0..dst_w {
                let sx = ((dx as f64 + 0.5) * scale_x - 0.5).clamp(0.0, (self.width - 1) as f64);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let wx = (sx - x0 as f64) as f32;
                let dst = (dy * dst_w + dx) * 3;
                let p00 = (y0 * self.width + x0) * 3;
                let p01 = (y0 * self.width + x1) * 3;
                let p10 = (y1 * self.width + x0) * 3;
                let p11 = (y1 * self.width + x1) * 3;
                for c in 0..3 {
                    let top =
                        self.data[p00 + c] as f32 * (1.0 - wx) + self.data[p01 + c] as f32 * wx;
                    let bot =
                        self.data[p10 + c] as f32 * (1.0 - wx) + self.data[p11 + c] as f32 * wx;
                    out[dst + c] = (top * (1.0 - wy) + bot * wy).clamp(0.0, 255.0);
                }
            }
        }
        out
    }

    /// Resizes using separable Keys bicubic (`a = -0.5`, matching PIL/PyTorch `BICUBIC`),
    /// returning interleaved RGB `f32` in `[0.0, 255.0]`.
    pub fn resize_bicubic_f32(&self, dst_w: usize, dst_h: usize) -> Vec<f32> {
        assert!(dst_w > 0 && dst_h > 0);
        if dst_w == self.width && dst_h == self.height {
            return self.data.iter().map(|&v| v as f32).collect();
        }
        #[inline]
        fn cubic_weight(x: f64) -> f64 {
            let ax = x.abs();
            if ax < 1.0 {
                (1.5 * ax - 2.5) * ax * ax + 1.0
            } else if ax < 2.0 {
                ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0
            } else {
                0.0
            }
        }

        // Horizontal pass: [src_h, dst_w, 3]
        let mut tmp = vec![0.0f64; self.height * dst_w * 3];
        let scale_x = self.width as f64 / dst_w as f64;
        for dx in 0..dst_w {
            let sx = (dx as f64 + 0.5) * scale_x - 0.5;
            let ix = sx.floor() as isize;
            let mut ws = [0.0f64; 4];
            let mut wsum = 0.0f64;
            for (k, w_slot) in ws.iter_mut().enumerate() {
                let w = cubic_weight(sx - (ix - 1 + k as isize) as f64);
                *w_slot = w;
                wsum += w;
            }
            if wsum.abs() > 1e-12 {
                for w in &mut ws {
                    *w /= wsum;
                }
            }
            let xs = [
                (ix - 1).clamp(0, self.width as isize - 1) as usize,
                ix.clamp(0, self.width as isize - 1) as usize,
                (ix + 1).clamp(0, self.width as isize - 1) as usize,
                (ix + 2).clamp(0, self.width as isize - 1) as usize,
            ];
            for y in 0..self.height {
                let row_off = y * self.width * 3;
                let dst = (y * dst_w + dx) * 3;
                for c in 0..3 {
                    let mut acc = 0.0f64;
                    for k in 0..4 {
                        acc += ws[k] * self.data[row_off + xs[k] * 3 + c] as f64;
                    }
                    tmp[dst + c] = acc;
                }
            }
        }

        // Vertical pass: [dst_h, dst_w, 3]
        let mut out = vec![0.0f32; dst_w * dst_h * 3];
        let scale_y = self.height as f64 / dst_h as f64;
        for dy in 0..dst_h {
            let sy = (dy as f64 + 0.5) * scale_y - 0.5;
            let iy = sy.floor() as isize;
            let mut ws = [0.0f64; 4];
            let mut wsum = 0.0f64;
            for (k, w_slot) in ws.iter_mut().enumerate() {
                let w = cubic_weight(sy - (iy - 1 + k as isize) as f64);
                *w_slot = w;
                wsum += w;
            }
            if wsum.abs() > 1e-12 {
                for w in &mut ws {
                    *w /= wsum;
                }
            }
            let ys = [
                (iy - 1).clamp(0, self.height as isize - 1) as usize,
                iy.clamp(0, self.height as isize - 1) as usize,
                (iy + 1).clamp(0, self.height as isize - 1) as usize,
                (iy + 2).clamp(0, self.height as isize - 1) as usize,
            ];
            for dx in 0..dst_w {
                let dst = (dy * dst_w + dx) * 3;
                for c in 0..3 {
                    let mut acc = 0.0f64;
                    for k in 0..4 {
                        acc += ws[k] * tmp[(ys[k] * dst_w + dx) * 3 + c];
                    }
                    out[dst + c] = acc.clamp(0.0, 255.0) as f32;
                }
            }
        }
        out
    }

    /// Preprocesses this RGB image according to `cfg` into normalized planar CHW floats.
    pub fn preprocess(&self, cfg: &VisionConfig) -> PreprocessedImage {
        let (target_h, target_w) = cfg.target_image_dims(self.height, self.width);
        let resized = self.resize_bicubic_f32(target_w, target_h);
        let hw = target_h * target_w;
        let mut pixels_chw = vec![0.0f32; 3 * hw];
        let inv255 = 1.0f32 / 255.0f32;
        for i in 0..hw {
            for c in 0..3 {
                let v = resized[i * 3 + c] * inv255;
                pixels_chw[c * hw + i] = (v - cfg.image_mean[c]) / cfg.image_std[c];
            }
        }
        PreprocessedImage {
            width: target_w,
            height: target_h,
            grid_h: target_h / cfg.patch_size,
            grid_w: target_w / cfg.patch_size,
            pixels_chw,
        }
    }
}

/// Computes Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Pixtral dynamic resolution `(target_h, target_w)`
/// rounded to multiples of `factor` (`patch_size * merge_size`) and clamped to `[min_pixels, max_pixels]`.
pub fn smart_resize_dims(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let f = factor.max(1) as f64;
    let h = height.max(1) as f64;
    let w = width.max(1) as f64;
    let mut h_bar = (h / f).round().max(1.0) * f;
    let mut w_bar = (w / f).round().max(1.0) * f;
    let max_p = max_pixels.max(factor * factor) as f64;
    let min_p = min_pixels.max(factor * factor) as f64;
    if h_bar * w_bar > max_p {
        let beta = ((h * w) / max_p).sqrt();
        h_bar = ((h / beta / f).floor().max(1.0)) * f;
        w_bar = ((w / beta / f).floor().max(1.0)) * f;
    } else if h_bar * w_bar < min_p {
        let beta = (min_p / (h * w)).sqrt();
        h_bar = ((h * beta / f).ceil().max(1.0)) * f;
        w_bar = ((w * beta / f).ceil().max(1.0)) * f;
    }
    (h_bar as usize, w_bar as usize)
}

/// Normalized planar CHW `[3, height, width]` image ready for `VisionModel::encode`.
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub width: usize,
    pub height: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub pixels_chw: Vec<f32>,
}

// ============================================================================
// Vision Configuration & Projector Types
// ============================================================================

/// Supported `mmproj` projector families (matching `clip.projector_type` in `llama.cpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionProjectorType {
    /// Gemma 3 / Gemma 3n: 2D avg-pool -> `mm.soft_emb_norm` (RMSNorm) -> `mm.input_projection`.
    Gemma3,
    /// Gemma 4: 2D avg-pool -> `mm.soft_emb_norm` (RMSNorm) -> `mm.input_projection`.
    Gemma4,
    /// Qwen2-VL: 3D temporal-2 conv, 2D RoPE, 2×2 spatial patch merger + 2-layer GELU MLP.
    Qwen2Vl,
    /// Qwen2.5-VL: 3D temporal-2 conv, 2D RoPE, Pre-RMSNorm + SwiGLU ViT, 2×2 merger + MLP.
    Qwen25Vl,
    /// Qwen3-VL / Qwen3.5: 3D temporal-2 conv, 2D RoPE + learned pos emb, 2×2 merger + MLP.
    Qwen3Vl,
    /// Pixtral / Mistral-3: 2D RoPE, Pre-RMSNorm + SwiGLU ViT, spatial merger + MLP.
    Pixtral,
    /// SmolVLM / Idefics3 / LFM2-VL: SigLIP ViT + pixel shuffle + 2-layer MLP / linear.
    SmolVlm,
    /// Generic SigLIP / CLIP ViT + linear or 2-layer MLP projector.
    Mlp,
}

impl VisionProjectorType {
    pub fn from_gguf_str(s: &str) -> Result<Self, ModelError> {
        match s {
            "gemma3" | "gemma3n" => Ok(Self::Gemma3),
            "gemma4" => Ok(Self::Gemma4),
            "qwen2vl_merger" | "qwen2vl" => Ok(Self::Qwen2Vl),
            "qwen2.5vl_merger" | "qwen25vl_merger" | "qwen2.5vl" => Ok(Self::Qwen25Vl),
            "qwen3vl_merger" | "qwen35_merger" | "qwen3vl" | "qwen35" => Ok(Self::Qwen3Vl),
            "pixtral" | "mistral3" => Ok(Self::Pixtral),
            "smolvlm" | "idefics3" | "lfm2" => Ok(Self::SmolVlm),
            "mlp" | "siglip" | "linear" | "ldp" | "ldpv2" => Ok(Self::Mlp),
            other => Err(ModelError::UnsupportedArch(format!(
                "unsupported clip.projector_type '{other}'"
            ))),
        }
    }

    /// True if this vision encoder uses 2D Vision RoPE `(row, col)`.
    #[inline]
    pub fn uses_2d_rope(self) -> bool {
        matches!(
            self,
            Self::Qwen2Vl | Self::Qwen25Vl | Self::Qwen3Vl | Self::Pixtral
        )
    }

    /// True if this vision encoder uses dynamic aspect-preserving `smart_resize_dims`.
    #[inline]
    pub fn uses_dynamic_resolution(self) -> bool {
        matches!(
            self,
            Self::Qwen2Vl | Self::Qwen25Vl | Self::Qwen3Vl | Self::Pixtral
        )
    }
}

/// Activation function used inside ViT FFN / MLP projector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionAct {
    GeluTanh,
    GeluQuick,
    GeluErf,
    Silu,
}

impl VisionAct {
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::GeluTanh => gelu_tanh_f32(x),
            Self::GeluQuick => x / (1.0 + (-1.702 * x).exp()),
            Self::GeluErf => 0.5 * x * (1.0 + erf_approx_f32(x * core::f32::consts::FRAC_1_SQRT_2)),
            Self::Silu => x / (1.0 + (-x).exp()),
        }
    }
}

/// Abramowitz & Stegun 7.1.26 erf approximation (`max error < 1.5e-7`, matches `libm::erff`).
#[inline]
fn erf_approx_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f64 } else { 1.0f64 };
    let ax = (x as f64).abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    (sign * (1.0 - poly * (-ax * ax).exp())) as f32
}

/// Parsed `mmproj` GGUF vision encoder configuration.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub projector_type: VisionProjectorType,
    pub image_size: usize,
    pub patch_size: usize,
    pub d_vit: usize,
    pub d_ff: usize,
    pub d_proj: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub merge_size: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub ffn_act: VisionAct,
    pub proj_act: VisionAct,
    pub use_2d_rope: bool,
    pub dynamic_resolution: bool,
}

impl VisionConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ModelError> {
        let proj_str = gguf
            .get_meta("clip.projector_type")
            .and_then(|v| v.as_str())
            .unwrap_or("mlp");
        let projector_type = VisionProjectorType::from_gguf_str(proj_str)?;

        let patch_size = gguf
            .get_meta("clip.vision.patch_size")
            .and_then(|v| v.as_usize())
            .unwrap_or(14);
        let image_size = gguf
            .get_meta("clip.vision.image_size")
            .and_then(|v| v.as_usize())
            .unwrap_or(if matches!(projector_type, VisionProjectorType::Gemma3) {
                896
            } else {
                384
            });
        let d_vit = gguf.req_usize("clip.vision.embedding_length")?;
        let d_ff = gguf.req_usize("clip.vision.feed_forward_length")?;
        let d_proj = gguf.req_usize("clip.vision.projection_dim")?;
        let n_layers = gguf.req_usize("clip.vision.block_count")?;
        let n_heads = gguf.req_usize("clip.vision.attention.head_count")?;
        if n_heads == 0 || !d_vit.is_multiple_of(n_heads) {
            return Err(ModelError::InvalidConfig(format!(
                "clip.vision.embedding_length ({d_vit}) must be divisible by head_count ({n_heads})"
            )));
        }
        let head_dim = d_vit / n_heads;
        let norm_eps = gguf
            .get_meta("clip.vision.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let rope_theta = gguf
            .get_meta("clip.vision.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);

        let default_merge = match projector_type {
            VisionProjectorType::Gemma3 => 4,
            VisionProjectorType::Gemma4 | VisionProjectorType::SmolVlm => 3,
            VisionProjectorType::Qwen2Vl
            | VisionProjectorType::Qwen25Vl
            | VisionProjectorType::Qwen3Vl
            | VisionProjectorType::Pixtral => 2,
            VisionProjectorType::Mlp => 1,
        };
        let merge_size = gguf
            .get_meta("clip.vision.spatial_merge_size")
            .and_then(|v| v.as_usize())
            .or_else(|| {
                gguf.get_meta("clip.vision.projector.scale_factor")
                    .and_then(|v| v.as_usize())
            })
            .unwrap_or(default_merge)
            .max(1);

        let default_mean = match projector_type {
            VisionProjectorType::Gemma3
            | VisionProjectorType::Gemma4
            | VisionProjectorType::SmolVlm => [0.5, 0.5, 0.5],
            _ => [0.481_454_66, 0.457_827_5, 0.408_210_73],
        };
        let default_std = match projector_type {
            VisionProjectorType::Gemma3
            | VisionProjectorType::Gemma4
            | VisionProjectorType::SmolVlm => [0.5, 0.5, 0.5],
            _ => [0.268_629_54, 0.261_302_6, 0.275_777_1],
        };
        let image_mean = gguf
            .get_meta("clip.vision.image_mean")
            .and_then(|v| v.as_f32_slice())
            .and_then(|s| (s.len() >= 3).then(|| [s[0], s[1], s[2]]))
            .unwrap_or(default_mean);
        let image_std = gguf
            .get_meta("clip.vision.image_std")
            .and_then(|v| v.as_f32_slice())
            .and_then(|s| (s.len() >= 3).then(|| [s[0], s[1], s[2]]))
            .unwrap_or(default_std);

        let min_pixels = gguf
            .get_meta("clip.vision.image_min_pixels")
            .and_then(|v| v.as_usize())
            .unwrap_or((patch_size * merge_size).pow(2) * 4);
        let max_pixels = gguf
            .get_meta("clip.vision.image_max_pixels")
            .and_then(|v| v.as_usize())
            .unwrap_or(image_size * image_size);

        let use_gelu = gguf
            .get_meta("clip.use_gelu")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let use_silu = gguf
            .get_meta("clip.use_silu")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let ffn_act = if use_silu {
            VisionAct::Silu
        } else if use_gelu {
            VisionAct::GeluTanh
        } else {
            match projector_type {
                VisionProjectorType::Qwen25Vl | VisionProjectorType::Pixtral => VisionAct::Silu,
                VisionProjectorType::Qwen2Vl => VisionAct::GeluQuick,
                _ => VisionAct::GeluTanh,
            }
        };
        let proj_act = match projector_type {
            VisionProjectorType::Qwen2Vl
            | VisionProjectorType::Qwen25Vl
            | VisionProjectorType::Qwen3Vl
            | VisionProjectorType::Pixtral => VisionAct::GeluErf,
            _ => ffn_act,
        };

        let use_2d_rope = gguf
            .get_meta("clip.vision.use_2d_rope")
            .and_then(|v| v.as_bool())
            .unwrap_or(projector_type.uses_2d_rope());
        let dynamic_resolution = gguf
            .get_meta("clip.vision.dynamic_resolution")
            .and_then(|v| v.as_bool())
            .unwrap_or(projector_type.uses_dynamic_resolution());

        Ok(Self {
            projector_type,
            image_size,
            patch_size,
            d_vit,
            d_ff,
            d_proj,
            n_layers,
            n_heads,
            head_dim,
            norm_eps,
            rope_theta,
            merge_size,
            image_mean,
            image_std,
            min_pixels,
            max_pixels,
            ffn_act,
            proj_act,
            use_2d_rope,
            dynamic_resolution,
        })
    }

    /// Computes `(target_h, target_w)` for preprocessing an input image.
    pub fn target_image_dims(&self, src_h: usize, src_w: usize) -> (usize, usize) {
        if self.dynamic_resolution {
            smart_resize_dims(
                src_h,
                src_w,
                self.patch_size * self.merge_size,
                self.min_pixels,
                self.max_pixels,
            )
        } else {
            (self.image_size, self.image_size)
        }
    }
}

// ============================================================================
// Zero-Copy Vision Weights
// ============================================================================

/// Attention projection layout inside a ViT block.
#[allow(clippy::large_enum_variant)]
pub enum VisionAttnProj<'a> {
    Separate {
        wq: RawMatView<'a>,
        wk: RawMatView<'a>,
        wv: RawMatView<'a>,
        bq: Option<Vec<f32>>,
        bk: Option<Vec<f32>>,
        bv: Option<Vec<f32>>,
    },
    FusedQkv {
        wqkv: RawMatView<'a>,
        bqkv: Option<Vec<f32>>,
    },
}

/// FFN layout inside a ViT block.
pub enum VisionFfn<'a> {
    /// Standard 2-layer MLP: `down(act(up(x)))`.
    Dense {
        w_up: RawMatView<'a>,
        b_up: Option<Vec<f32>>,
        w_down: RawMatView<'a>,
        b_down: Option<Vec<f32>>,
    },
    /// Gated 3-layer GLU (SwiGLU / GeGLU): `down(act(gate(x)) * up(x))`.
    Gated {
        w_gate: RawMatView<'a>,
        b_gate: Option<Vec<f32>>,
        w_up: RawMatView<'a>,
        b_up: Option<Vec<f32>>,
        w_down: RawMatView<'a>,
        b_down: Option<Vec<f32>>,
    },
}

/// Per-layer weights for a ViT transformer block (`v.blk.{i}.*`).
pub struct VisionLayerWeights<'a> {
    pub ln1_w: Vec<f32>,
    /// `Some` => LayerNorm; `None` => RMSNorm.
    pub ln1_b: Option<Vec<f32>>,
    pub attn: VisionAttnProj<'a>,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub wo: RawMatView<'a>,
    pub bo: Option<Vec<f32>>,
    pub ln2_w: Vec<f32>,
    /// `Some` => LayerNorm; `None` => RMSNorm.
    pub ln2_b: Option<Vec<f32>>,
    pub ffn: VisionFfn<'a>,
}

/// Multimodal projector mapping ViT patch embeddings into LLM hidden space (`d_proj`).
pub enum VisionProjector<'a> {
    /// Gemma 3 / Gemma 4: `scale×scale` spatial average pool -> RMSNorm -> Linear projection.
    GemmaPoolProj {
        scale_factor: usize,
        soft_emb_norm_w: Vec<f32>,
        input_proj: RawMatView<'a>,
        /// True if GGUF stored `mm.input_projection.weight` as `[d_proj, d_vit]` (rows=d_vit, cols=d_proj).
        transposed: bool,
    },
    /// Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5: `merge×merge` spatial concat + 2-layer MLP.
    QwenVlMerger {
        merge_size: usize,
        fc1_w: RawMatView<'a>,
        fc1_b: Option<Vec<f32>>,
        fc2_w: RawMatView<'a>,
        fc2_b: Option<Vec<f32>>,
        act: VisionAct,
    },
    /// Pixtral: optional RMSNorm + spatial patch merger linear (`mm.patch_merger`) + 2-layer MLP.
    PixtralMerger {
        merge_size: usize,
        merging_w: Option<RawMatView<'a>>,
        fc1_w: RawMatView<'a>,
        fc1_b: Option<Vec<f32>>,
        fc2_w: RawMatView<'a>,
        fc2_b: Option<Vec<f32>>,
        act: VisionAct,
    },
    /// SmolVLM / LFM2-VL / Generic MLP: optional `scale×scale` pixel-shuffle + 1- or 2-layer MLP.
    MlpProj {
        scale_factor: usize,
        fc1_w: RawMatView<'a>,
        fc1_b: Option<Vec<f32>>,
        fc2_w: Option<RawMatView<'a>>,
        fc2_b: Option<Vec<f32>>,
        act: VisionAct,
    },
}

/// Output of `VisionModel::encode`: projected visual token embeddings in LLM hidden space.
#[derive(Debug, Clone)]
pub struct VisionEmbeddings {
    /// Flattened `[n_tokens, d_llm]` embeddings ready for LLM `forward_embed`.
    pub embeddings: Vec<f32>,
    /// Number of visual tokens produced for the image.
    pub n_tokens: usize,
    /// LLM embedding dimension (`d_proj`).
    pub d_llm: usize,
    /// Post-merge spatial height in tokens.
    pub grid_h: usize,
    /// Post-merge spatial width in tokens.
    pub grid_w: usize,
    /// Per-token `(t, h, w)` 3D M-RoPE relative coordinates for Qwen-VL / Qwen3.5.
    pub mrope_positions: Vec<[usize; 3]>,
}

impl VisionEmbeddings {
    #[inline]
    pub fn token_embed(&self, idx: usize) -> &[f32] {
        let off = idx * self.d_llm;
        &self.embeddings[off..off + self.d_llm]
    }
}

/// Zero-copy multimodal Vision Encoder loaded from an `mmproj` GGUF file.
pub struct VisionModel<'a> {
    pub cfg: VisionConfig,
    /// Flattened `[d_vit, 3 * patch_size * patch_size]` patch embedding kernel.
    pub patch_embd_w: RawMatView<'a>,
    /// Optional second temporal kernel (`v.patch_embd.weight.1`) for Qwen-VL 3D conv.
    pub patch_embd_w1: Option<RawMatView<'a>>,
    pub patch_embd_b: Option<Vec<f32>>,
    pub class_embd: Option<Vec<f32>>,
    /// Learned position embeddings `[n_pos, d_vit]`.
    pub position_embd: Option<Vec<f32>>,
    pub pos_grid_h: usize,
    pub pos_grid_w: usize,
    pub pre_ln_w: Option<Vec<f32>>,
    pub pre_ln_b: Option<Vec<f32>>,
    pub layers: Vec<VisionLayerWeights<'a>>,
    pub post_ln_w: Option<Vec<f32>>,
    pub post_ln_b: Option<Vec<f32>>,
    pub projector: VisionProjector<'a>,
}

// ============================================================================
// GGUF Tensor Helpers
// ============================================================================

fn load_vec_1d(gguf: &GgufFile, name: &str, expected: usize) -> Result<Vec<f32>, ModelError> {
    let (info, bytes) = gguf.tensor_bytes(name, "vision")?;
    let total: usize = info.dims.iter().product();
    if total != expected {
        return Err(GgufError::TensorShapeMismatch {
            path: gguf.path().display().to_string(),
            tensor: name.to_string(),
            layer: "vision".to_string(),
            expected: vec![expected],
            actual: info.dims.clone(),
        }
        .into());
    }
    let mut out = vec![0.0f32; expected];
    dequantize_row_bytes(info.dtype, bytes, &mut out);
    Ok(out)
}

fn opt_vec_1d(
    gguf: &GgufFile,
    name: &str,
    expected: usize,
) -> Result<Option<Vec<f32>>, ModelError> {
    if gguf.tensor_info(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_vec_1d(gguf, name, expected)?))
}

fn load_mat_2d<'a>(
    gguf: &'a GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<RawMatView<'a>, ModelError> {
    let (info, bytes) = gguf.req_tensor(name, "vision", &[cols, rows])?;
    Ok(RawMatView {
        name: info.name.as_str(),
        bytes,
        dtype: info.dtype,
        rows,
        cols,
        row_offset: 0,
        is_sliced: false,
        lora_slot: None,
    })
}

fn load_mat_2d_any<'a>(
    gguf: &'a GgufFile,
    names: &[&str],
    rows: usize,
    cols: usize,
) -> Result<RawMatView<'a>, ModelError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return load_mat_2d(gguf, name, rows, cols);
        }
    }
    Err(GgufError::MissingTensor {
        path: gguf.path().display().to_string(),
        tensor: names.join(" | "),
        layer: "vision".to_string(),
    }
    .into())
}

fn opt_vec_1d_any(
    gguf: &GgufFile,
    names: &[&str],
    expected: usize,
) -> Result<Option<Vec<f32>>, ModelError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return Ok(Some(load_vec_1d(gguf, name, expected)?));
        }
    }
    Ok(None)
}

fn load_patch_conv<'a>(
    gguf: &'a GgufFile,
    name: &str,
    d_vit: usize,
    patch_size: usize,
) -> Result<RawMatView<'a>, ModelError> {
    let (info, bytes) = gguf.tensor_bytes(name, "vision")?;
    let patch_dim = 3 * patch_size * patch_size;
    let total: usize = info.dims.iter().product();
    if total != d_vit * patch_dim {
        return Err(GgufError::TensorShapeMismatch {
            path: gguf.path().display().to_string(),
            tensor: name.to_string(),
            layer: "vision".to_string(),
            expected: vec![patch_size, patch_size, 3, d_vit],
            actual: info.dims.clone(),
        }
        .into());
    }
    Ok(RawMatView {
        name: info.name.as_str(),
        bytes,
        dtype: info.dtype,
        rows: d_vit,
        cols: patch_dim,
        row_offset: 0,
        is_sliced: false,
        lora_slot: None,
    })
}

impl<'a> VisionModel<'a> {
    /// Loads and validates a zero-copy `VisionModel` from an `mmproj` GGUF file.
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let cfg = VisionConfig::from_gguf(gguf)?;
        let d_vit = cfg.d_vit;
        let d_ff = cfg.d_ff;
        let d_proj = cfg.d_proj;

        let patch_embd_w = load_patch_conv(gguf, "v.patch_embd.weight", d_vit, cfg.patch_size)?;
        let patch_embd_w1 = if gguf.tensor_info("v.patch_embd.weight.1").is_some() {
            Some(load_patch_conv(
                gguf,
                "v.patch_embd.weight.1",
                d_vit,
                cfg.patch_size,
            )?)
        } else {
            None
        };
        let patch_embd_b = opt_vec_1d(gguf, "v.patch_embd.bias", d_vit)?;
        let class_embd = opt_vec_1d(gguf, "v.class_embd", d_vit)?;

        let (position_embd, pos_grid_h, pos_grid_w) =
            if gguf.tensor_info("v.position_embd.weight").is_some() {
                let (info, bytes) = gguf.tensor_bytes("v.position_embd.weight", "vision")?;
                let total: usize = info.dims.iter().product();
                if !total.is_multiple_of(d_vit) {
                    return Err(ModelError::InvalidConfig(format!(
                        "v.position_embd.weight total {total} not divisible by d_vit {d_vit}"
                    )));
                }
                let n_pos = total / d_vit;
                let n_spatial = if class_embd.is_some() && n_pos > 1 {
                    let s = ((n_pos - 1) as f64).sqrt().round() as usize;
                    if s * s == n_pos - 1 { n_pos - 1 } else { n_pos }
                } else {
                    n_pos
                };
                let g = (n_spatial as f64).sqrt().round() as usize;
                let (gh, gw) = if g * g == n_spatial {
                    (g, g)
                } else {
                    let g0 = cfg.image_size / cfg.patch_size;
                    (g0, n_spatial / g0.max(1))
                };
                let mut pos_vec = vec![0.0f32; total];
                let rb = info.dtype.row_bytes(d_vit);
                for r in 0..n_pos {
                    dequantize_row_bytes(
                        info.dtype,
                        &bytes[r * rb..(r + 1) * rb],
                        &mut pos_vec[r * d_vit..(r + 1) * d_vit],
                    );
                }
                (Some(pos_vec), gh, gw)
            } else {
                (None, 0, 0)
            };

        let pre_ln_w = opt_vec_1d(gguf, "v.pre_ln.weight", d_vit)?;
        let pre_ln_b = opt_vec_1d(gguf, "v.pre_ln.bias", d_vit)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let ln1_w = load_vec_1d(gguf, &format!("v.blk.{il}.ln1.weight"), d_vit)?;
            let ln1_b = opt_vec_1d(gguf, &format!("v.blk.{il}.ln1.bias"), d_vit)?;

            let qkv_name = format!("v.blk.{il}.attn_qkv.weight");
            let attn = if gguf.tensor_info(&qkv_name).is_some() {
                let wqkv = load_mat_2d(gguf, &qkv_name, 3 * d_vit, d_vit)?;
                let bqkv = opt_vec_1d(gguf, &format!("v.blk.{il}.attn_qkv.bias"), 3 * d_vit)?;
                VisionAttnProj::FusedQkv { wqkv, bqkv }
            } else {
                let wq = load_mat_2d(gguf, &format!("v.blk.{il}.attn_q.weight"), d_vit, d_vit)?;
                let wk = load_mat_2d(gguf, &format!("v.blk.{il}.attn_k.weight"), d_vit, d_vit)?;
                let wv = load_mat_2d(gguf, &format!("v.blk.{il}.attn_v.weight"), d_vit, d_vit)?;
                let bq = opt_vec_1d(gguf, &format!("v.blk.{il}.attn_q.bias"), d_vit)?;
                let bk = opt_vec_1d(gguf, &format!("v.blk.{il}.attn_k.bias"), d_vit)?;
                let bv = opt_vec_1d(gguf, &format!("v.blk.{il}.attn_v.bias"), d_vit)?;
                VisionAttnProj::Separate {
                    wq,
                    wk,
                    wv,
                    bq,
                    bk,
                    bv,
                }
            };

            let q_norm = opt_vec_1d(
                gguf,
                &format!("v.blk.{il}.attn_q_norm.weight"),
                cfg.head_dim,
            )?;
            let k_norm = opt_vec_1d(
                gguf,
                &format!("v.blk.{il}.attn_k_norm.weight"),
                cfg.head_dim,
            )?;
            let wo = load_mat_2d(gguf, &format!("v.blk.{il}.attn_out.weight"), d_vit, d_vit)?;
            let bo = opt_vec_1d(gguf, &format!("v.blk.{il}.attn_out.bias"), d_vit)?;

            let ln2_w = load_vec_1d(gguf, &format!("v.blk.{il}.ln2.weight"), d_vit)?;
            let ln2_b = opt_vec_1d(gguf, &format!("v.blk.{il}.ln2.bias"), d_vit)?;

            let gate_name = format!("v.blk.{il}.ffn_gate.weight");
            let ffn = if gguf.tensor_info(&gate_name).is_some() {
                let w_gate = load_mat_2d(gguf, &gate_name, d_ff, d_vit)?;
                let b_gate = opt_vec_1d(gguf, &format!("v.blk.{il}.ffn_gate.bias"), d_ff)?;
                let w_up = load_mat_2d(gguf, &format!("v.blk.{il}.ffn_up.weight"), d_ff, d_vit)?;
                let b_up = opt_vec_1d(gguf, &format!("v.blk.{il}.ffn_up.bias"), d_ff)?;
                let w_down =
                    load_mat_2d(gguf, &format!("v.blk.{il}.ffn_down.weight"), d_vit, d_ff)?;
                let b_down = opt_vec_1d(gguf, &format!("v.blk.{il}.ffn_down.bias"), d_vit)?;
                VisionFfn::Gated {
                    w_gate,
                    b_gate,
                    w_up,
                    b_up,
                    w_down,
                    b_down,
                }
            } else {
                let w_up = load_mat_2d(gguf, &format!("v.blk.{il}.ffn_up.weight"), d_ff, d_vit)?;
                let b_up = opt_vec_1d(gguf, &format!("v.blk.{il}.ffn_up.bias"), d_ff)?;
                let w_down =
                    load_mat_2d(gguf, &format!("v.blk.{il}.ffn_down.weight"), d_vit, d_ff)?;
                let b_down = opt_vec_1d(gguf, &format!("v.blk.{il}.ffn_down.bias"), d_vit)?;
                VisionFfn::Dense {
                    w_up,
                    b_up,
                    w_down,
                    b_down,
                }
            };

            layers.push(VisionLayerWeights {
                ln1_w,
                ln1_b,
                attn,
                q_norm,
                k_norm,
                wo,
                bo,
                ln2_w,
                ln2_b,
                ffn,
            });
        }

        let post_ln_w = opt_vec_1d(gguf, "v.post_ln.weight", d_vit)?;
        let post_ln_b = opt_vec_1d(gguf, "v.post_ln.bias", d_vit)?;

        let projector = match cfg.projector_type {
            VisionProjectorType::Gemma3 | VisionProjectorType::Gemma4 => {
                let soft_emb_norm_w = load_vec_1d(gguf, "mm.soft_emb_norm.weight", d_vit)?;
                let (proj_info, proj_bytes) =
                    gguf.tensor_bytes("mm.input_projection.weight", "vision")?;
                let (input_proj, transposed) = if proj_info.dims == [d_vit, d_proj] {
                    (
                        RawMatView {
                            name: proj_info.name.as_str(),
                            bytes: proj_bytes,
                            dtype: proj_info.dtype,
                            rows: d_proj,
                            cols: d_vit,
                            row_offset: 0,
                            is_sliced: false,
                            lora_slot: None,
                        },
                        false,
                    )
                } else if proj_info.dims == [d_proj, d_vit] {
                    (
                        RawMatView {
                            name: proj_info.name.as_str(),
                            bytes: proj_bytes,
                            dtype: proj_info.dtype,
                            rows: d_vit,
                            cols: d_proj,
                            row_offset: 0,
                            is_sliced: false,
                            lora_slot: None,
                        },
                        true,
                    )
                } else {
                    return Err(GgufError::TensorShapeMismatch {
                        path: gguf.path().display().to_string(),
                        tensor: "mm.input_projection.weight".to_string(),
                        layer: "vision".to_string(),
                        expected: vec![d_vit, d_proj],
                        actual: proj_info.dims.clone(),
                    }
                    .into());
                };
                VisionProjector::GemmaPoolProj {
                    scale_factor: cfg.merge_size,
                    soft_emb_norm_w,
                    input_proj,
                    transposed,
                }
            }
            VisionProjectorType::Qwen2Vl
            | VisionProjectorType::Qwen25Vl
            | VisionProjectorType::Qwen3Vl => {
                let merged_dim = d_vit * cfg.merge_size * cfg.merge_size;
                let fc1_w = load_mat_2d(gguf, "mm.0.weight", merged_dim, merged_dim)?;
                let fc1_b = opt_vec_1d(gguf, "mm.0.bias", merged_dim)?;
                let fc2_w = load_mat_2d(gguf, "mm.2.weight", d_proj, merged_dim)?;
                let fc2_b = opt_vec_1d(gguf, "mm.2.bias", d_proj)?;
                VisionProjector::QwenVlMerger {
                    merge_size: cfg.merge_size,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    act: cfg.proj_act,
                }
            }
            VisionProjectorType::Pixtral => {
                let merged_dim = d_vit * cfg.merge_size * cfg.merge_size;
                let merging_w = if gguf.tensor_info("mm.patch_merger.weight").is_some() {
                    Some(load_mat_2d(
                        gguf,
                        "mm.patch_merger.weight",
                        d_vit,
                        merged_dim,
                    )?)
                } else {
                    None
                };
                let fc1_in = if merging_w.is_some() {
                    d_vit
                } else {
                    merged_dim
                };
                let fc1_w = load_mat_2d_any(gguf, &["mm.1.weight", "mm.0.weight"], d_proj, fc1_in)?;
                let fc1_b = opt_vec_1d_any(gguf, &["mm.1.bias", "mm.0.bias"], d_proj)?;
                let fc2_w = load_mat_2d_any(gguf, &["mm.2.weight", "mm.3.weight"], d_proj, d_proj)?;
                let fc2_b = opt_vec_1d_any(gguf, &["mm.2.bias", "mm.3.bias"], d_proj)?;
                VisionProjector::PixtralMerger {
                    merge_size: cfg.merge_size,
                    merging_w,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    act: cfg.proj_act,
                }
            }
            VisionProjectorType::SmolVlm | VisionProjectorType::Mlp => {
                let scale = cfg.merge_size;
                let in_dim = d_vit * scale * scale;
                let fc1_name = if gguf.tensor_info("mm.0.weight").is_some() {
                    "mm.0.weight"
                } else {
                    "mm.input_projection.weight"
                };
                let (fc1_info, fc1_bytes) = gguf.tensor_bytes(fc1_name, "vision")?;
                if fc1_info.dims.len() != 2 || fc1_info.dims[0] != in_dim {
                    return Err(GgufError::TensorShapeMismatch {
                        path: gguf.path().display().to_string(),
                        tensor: fc1_info.name.clone(),
                        layer: "vision".to_string(),
                        expected: vec![in_dim, d_proj],
                        actual: fc1_info.dims.clone(),
                    }
                    .into());
                }
                let fc1_out = fc1_info.dims[1];
                let fc1_w = RawMatView {
                    name: fc1_info.name.as_str(),
                    bytes: fc1_bytes,
                    dtype: fc1_info.dtype,
                    rows: fc1_out,
                    cols: in_dim,
                    row_offset: 0,
                    is_sliced: false,
                    lora_slot: None,
                };
                let fc1_b = opt_vec_1d(gguf, "mm.0.bias", fc1_out)?;
                let (fc2_w, fc2_b) = if gguf.tensor_info("mm.2.weight").is_some() {
                    (
                        Some(load_mat_2d(gguf, "mm.2.weight", d_proj, fc1_out)?),
                        opt_vec_1d(gguf, "mm.2.bias", d_proj)?,
                    )
                } else {
                    (None, None)
                };
                VisionProjector::MlpProj {
                    scale_factor: scale,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    act: cfg.proj_act,
                }
            }
        };

        Ok(Self {
            cfg,
            patch_embd_w,
            patch_embd_w1,
            patch_embd_b,
            class_embd,
            position_embd,
            pos_grid_h,
            pos_grid_w,
            pre_ln_w,
            pre_ln_b,
            layers,
            post_ln_w,
            post_ln_b,
            projector,
        })
    }

    /// Encodes a `PreprocessedImage` into LLM-space `VisionEmbeddings`.
    ///
    /// Deterministic across CPU ISAs (`generic` / `neon` / `dotprod` / `i8mm` / `avx2`).
    pub fn encode(
        &self,
        img: &PreprocessedImage,
        pool: &ThreadPool,
        kt: &KernelTable,
    ) -> Result<VisionEmbeddings, ModelError> {
        let p = self.cfg.patch_size;
        if img.height == 0
            || img.width == 0
            || !img.height.is_multiple_of(p)
            || !img.width.is_multiple_of(p)
        {
            return Err(ModelError::InvalidConfig(format!(
                "preprocessed image {}x{} must be divisible by patch_size {p}",
                img.width, img.height
            )));
        }
        let gh = img.grid_h;
        let gw = img.grid_w;
        let n_patches = gh * gw;
        let d_vit = self.cfg.d_vit;
        let d_ff = self.cfg.d_ff;
        let n_heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim;
        let patch_dim = 3 * p * p;

        let max_cols = d_vit
            .max(d_ff)
            .max(patch_dim)
            .max(d_vit * self.cfg.merge_size * self.cfg.merge_size)
            .max(self.cfg.d_proj);
        let mut scratch = MatvecScratch::new(max_cols.next_multiple_of(32));

        // 1. Extract patches in [C, pH, pW] order matching GGML conv2d [pW, pH, C, d_vit]
        let mut patch_buf = vec![0.0f32; n_patches * patch_dim];
        let hw = img.height * img.width;
        for py in 0..gh {
            for px in 0..gw {
                let pidx = py * gw + px;
                let dst = &mut patch_buf[pidx * patch_dim..(pidx + 1) * patch_dim];
                let mut k = 0usize;
                for c in 0..3 {
                    let ch_off = c * hw;
                    for dy in 0..p {
                        let row_off = ch_off + (py * p + dy) * img.width + px * p;
                        for dx in 0..p {
                            dst[k] = img.pixels_chw[row_off + dx];
                            k += 1;
                        }
                    }
                }
            }
        }

        // 2. Patch embedding projection (+ optional temporal-2 weight.1 + bias)
        let has_cls = self.class_embd.is_some();
        let n_seq = n_patches + usize::from(has_cls);
        let mut hidden = vec![0.0f32; n_seq * d_vit];
        if let Some(cls) = &self.class_embd {
            hidden[..d_vit].copy_from_slice(cls);
        }
        let patch_out = &mut hidden[usize::from(has_cls) * d_vit..];
        vision_linear(
            pool,
            kt,
            &self.patch_embd_w,
            &patch_buf,
            patch_out,
            n_patches,
            &mut scratch,
        );
        if let Some(w1) = &self.patch_embd_w1 {
            let mut tmp1 = vec![0.0f32; n_patches * d_vit];
            vision_linear(pool, kt, w1, &patch_buf, &mut tmp1, n_patches, &mut scratch);
            for (a, &b) in patch_out.iter_mut().zip(tmp1.iter()) {
                *a += b;
            }
        }
        if let Some(bias) = &self.patch_embd_b {
            add_bias_rows(patch_out, bias, n_patches, d_vit);
        }

        // 3. Learned position embedding (with bilinear interpolation if grid differs)
        if let Some(pos_emb) = &self.position_embd {
            let n_pos = pos_emb.len() / d_vit;
            if has_cls && n_pos == self.pos_grid_h * self.pos_grid_w + 1 {
                for j in 0..d_vit {
                    hidden[j] += pos_emb[j];
                }
                add_pos_emb_2d(
                    &mut hidden[d_vit..],
                    &pos_emb[d_vit..],
                    self.pos_grid_h,
                    self.pos_grid_w,
                    gh,
                    gw,
                    d_vit,
                );
            } else {
                add_pos_emb_2d(
                    &mut hidden[usize::from(has_cls) * d_vit..],
                    pos_emb,
                    self.pos_grid_h,
                    self.pos_grid_w,
                    gh,
                    gw,
                    d_vit,
                );
            }
        }

        // Optional pre-LN
        if let Some(w) = &self.pre_ln_w {
            let mut normed = vec![0.0f32; n_seq * d_vit];
            apply_norm_rows(
                &hidden,
                &mut normed,
                w,
                self.pre_ln_b.as_deref(),
                self.cfg.norm_eps,
                n_seq,
                d_vit,
            );
            hidden = normed;
        }

        // Precompute 2D Vision RoPE tables if enabled (half-dim row, half-dim col)
        let rope_2d = if self.cfg.use_2d_rope {
            Some(build_vision_rope_2d(
                gh,
                gw,
                has_cls,
                hd,
                self.cfg.rope_theta,
            ))
        } else {
            None
        };

        // 4. Transformer encoder blocks
        let mut norm_buf = vec![0.0f32; n_seq * d_vit];
        let mut q_buf = vec![0.0f32; n_seq * d_vit];
        let mut k_buf = vec![0.0f32; n_seq * d_vit];
        let mut v_buf = vec![0.0f32; n_seq * d_vit];
        let mut qkv_buf = Vec::new();
        let mut attn_out = vec![0.0f32; n_seq * d_vit];
        let mut proj_buf = vec![0.0f32; n_seq * d_vit];
        let mut ff_up = vec![0.0f32; n_seq * d_ff];
        let mut ff_gate = Vec::new();
        let mut scores = vec![0.0f32; n_seq];
        let attn_scale = 1.0f32 / (hd as f32).sqrt();

        for layer in &self.layers {
            // 4a. Pre-norm 1
            apply_norm_rows(
                &hidden,
                &mut norm_buf,
                &layer.ln1_w,
                layer.ln1_b.as_deref(),
                self.cfg.norm_eps,
                n_seq,
                d_vit,
            );

            // 4b. QKV projection
            match &layer.attn {
                VisionAttnProj::Separate {
                    wq,
                    wk,
                    wv,
                    bq,
                    bk,
                    bv,
                } => {
                    vision_linear(pool, kt, wq, &norm_buf, &mut q_buf, n_seq, &mut scratch);
                    vision_linear(pool, kt, wk, &norm_buf, &mut k_buf, n_seq, &mut scratch);
                    vision_linear(pool, kt, wv, &norm_buf, &mut v_buf, n_seq, &mut scratch);
                    if let Some(b) = bq {
                        add_bias_rows(&mut q_buf, b, n_seq, d_vit);
                    }
                    if let Some(b) = bk {
                        add_bias_rows(&mut k_buf, b, n_seq, d_vit);
                    }
                    if let Some(b) = bv {
                        add_bias_rows(&mut v_buf, b, n_seq, d_vit);
                    }
                }
                VisionAttnProj::FusedQkv { wqkv, bqkv } => {
                    if qkv_buf.len() != n_seq * 3 * d_vit {
                        qkv_buf.resize(n_seq * 3 * d_vit, 0.0);
                    }
                    vision_linear(pool, kt, wqkv, &norm_buf, &mut qkv_buf, n_seq, &mut scratch);
                    if let Some(b) = bqkv {
                        add_bias_rows(&mut qkv_buf, b, n_seq, 3 * d_vit);
                    }
                    for t in 0..n_seq {
                        let src = &qkv_buf[t * 3 * d_vit..(t + 1) * 3 * d_vit];
                        q_buf[t * d_vit..(t + 1) * d_vit].copy_from_slice(&src[..d_vit]);
                        k_buf[t * d_vit..(t + 1) * d_vit].copy_from_slice(&src[d_vit..2 * d_vit]);
                        v_buf[t * d_vit..(t + 1) * d_vit]
                            .copy_from_slice(&src[2 * d_vit..3 * d_vit]);
                    }
                }
            }

            // Optional per-head QK-Norm
            if let Some(qn) = &layer.q_norm {
                let mut tmp = vec![0.0f32; hd];
                for t in 0..n_seq {
                    for h in 0..n_heads {
                        let s = &mut q_buf[t * d_vit + h * hd..t * d_vit + (h + 1) * hd];
                        rms_norm(&mut tmp, s, qn, self.cfg.norm_eps);
                        s.copy_from_slice(&tmp);
                    }
                }
            }
            if let Some(kn) = &layer.k_norm {
                let mut tmp = vec![0.0f32; hd];
                for t in 0..n_seq {
                    for h in 0..n_heads {
                        let s = &mut k_buf[t * d_vit + h * hd..t * d_vit + (h + 1) * hd];
                        rms_norm(&mut tmp, s, kn, self.cfg.norm_eps);
                        s.copy_from_slice(&tmp);
                    }
                }
            }

            // Optional 2D Vision RoPE
            if let Some((cos_tbl, sin_tbl)) = &rope_2d {
                apply_vision_rope_2d(&mut q_buf, cos_tbl, sin_tbl, n_seq, n_heads, hd);
                apply_vision_rope_2d(&mut k_buf, cos_tbl, sin_tbl, n_seq, n_heads, hd);
            }

            // 4c. Bidirectional Full Multi-Head Self-Attention
            attn_out.fill(0.0);
            for h in 0..n_heads {
                let h_off = h * hd;
                for qi in 0..n_seq {
                    let q_vec = &q_buf[qi * d_vit + h_off..qi * d_vit + h_off + hd];
                    for ki in 0..n_seq {
                        let k_vec = &k_buf[ki * d_vit + h_off..ki * d_vit + h_off + hd];
                        let mut dot = 0.0f32;
                        for j in 0..hd {
                            dot += q_vec[j] * k_vec[j];
                        }
                        scores[ki] = dot * attn_scale;
                    }
                    softmax_inplace(&mut scores[..n_seq]);
                    let out_vec = &mut attn_out[qi * d_vit + h_off..qi * d_vit + h_off + hd];
                    for ki in 0..n_seq {
                        let w = scores[ki];
                        let v_vec = &v_buf[ki * d_vit + h_off..ki * d_vit + h_off + hd];
                        for j in 0..hd {
                            out_vec[j] += w * v_vec[j];
                        }
                    }
                }
            }

            // 4d. Output projection + residual
            vision_linear(
                pool,
                kt,
                &layer.wo,
                &attn_out,
                &mut proj_buf,
                n_seq,
                &mut scratch,
            );
            if let Some(b) = &layer.bo {
                add_bias_rows(&mut proj_buf, b, n_seq, d_vit);
            }
            for (h, &p_val) in hidden.iter_mut().zip(proj_buf.iter()) {
                *h += p_val;
            }

            // 4e. Pre-norm 2 + FFN + residual
            apply_norm_rows(
                &hidden,
                &mut norm_buf,
                &layer.ln2_w,
                layer.ln2_b.as_deref(),
                self.cfg.norm_eps,
                n_seq,
                d_vit,
            );
            match &layer.ffn {
                VisionFfn::Dense {
                    w_up,
                    b_up,
                    w_down,
                    b_down,
                } => {
                    vision_linear(pool, kt, w_up, &norm_buf, &mut ff_up, n_seq, &mut scratch);
                    if let Some(b) = b_up {
                        add_bias_rows(&mut ff_up, b, n_seq, d_ff);
                    }
                    for v in &mut ff_up {
                        *v = self.cfg.ffn_act.apply(*v);
                    }
                    vision_linear(pool, kt, w_down, &ff_up, &mut proj_buf, n_seq, &mut scratch);
                    if let Some(b) = b_down {
                        add_bias_rows(&mut proj_buf, b, n_seq, d_vit);
                    }
                }
                VisionFfn::Gated {
                    w_gate,
                    b_gate,
                    w_up,
                    b_up,
                    w_down,
                    b_down,
                } => {
                    if ff_gate.len() != n_seq * d_ff {
                        ff_gate.resize(n_seq * d_ff, 0.0);
                    }
                    vision_linear(
                        pool,
                        kt,
                        w_gate,
                        &norm_buf,
                        &mut ff_gate,
                        n_seq,
                        &mut scratch,
                    );
                    if let Some(b) = b_gate {
                        add_bias_rows(&mut ff_gate, b, n_seq, d_ff);
                    }
                    vision_linear(pool, kt, w_up, &norm_buf, &mut ff_up, n_seq, &mut scratch);
                    if let Some(b) = b_up {
                        add_bias_rows(&mut ff_up, b, n_seq, d_ff);
                    }
                    for (g, &u) in ff_gate.iter_mut().zip(ff_up.iter()) {
                        *g = self.cfg.ffn_act.apply(*g) * u;
                    }
                    vision_linear(
                        pool,
                        kt,
                        w_down,
                        &ff_gate,
                        &mut proj_buf,
                        n_seq,
                        &mut scratch,
                    );
                    if let Some(b) = b_down {
                        add_bias_rows(&mut proj_buf, b, n_seq, d_vit);
                    }
                }
            }
            for (h, &p_val) in hidden.iter_mut().zip(proj_buf.iter()) {
                *h += p_val;
            }
        }

        // 5. Post-LN (strip CLS token before spatial projection if present)
        let spatial_raw = &hidden[usize::from(has_cls) * d_vit..];
        let spatial = if let Some(w) = &self.post_ln_w {
            let mut out = vec![0.0f32; n_patches * d_vit];
            apply_norm_rows(
                spatial_raw,
                &mut out,
                w,
                self.post_ln_b.as_deref(),
                self.cfg.norm_eps,
                n_patches,
                d_vit,
            );
            out
        } else {
            spatial_raw.to_vec()
        };

        // 6. Multimodal Projector -> LLM embedding space
        let (embeddings, out_gh, out_gw) =
            self.run_projector(&spatial, gh, gw, pool, kt, &mut scratch)?;
        let n_tokens = out_gh * out_gw;
        let mut mrope_positions = Vec::with_capacity(n_tokens);
        for r in 0..out_gh {
            for c in 0..out_gw {
                mrope_positions.push([0, r, c]);
            }
        }

        Ok(VisionEmbeddings {
            embeddings,
            n_tokens,
            d_llm: self.cfg.d_proj,
            grid_h: out_gh,
            grid_w: out_gw,
            mrope_positions,
        })
    }

    fn run_projector(
        &self,
        spatial: &[f32],
        gh: usize,
        gw: usize,
        pool: &ThreadPool,
        kt: &KernelTable,
        scratch: &mut MatvecScratch,
    ) -> Result<(Vec<f32>, usize, usize), ModelError> {
        let d_vit = self.cfg.d_vit;
        let d_proj = self.cfg.d_proj;

        match &self.projector {
            VisionProjector::GemmaPoolProj {
                scale_factor,
                soft_emb_norm_w,
                input_proj,
                transposed,
            } => {
                let sf = (*scale_factor).max(1);
                if !gh.is_multiple_of(sf) || !gw.is_multiple_of(sf) {
                    return Err(ModelError::InvalidConfig(format!(
                        "Gemma vision grid {gh}x{gw} must be divisible by scale_factor {sf}"
                    )));
                }
                let out_gh = gh / sf;
                let out_gw = gw / sf;
                let n_out = out_gh * out_gw;
                let inv_area = 1.0f32 / (sf * sf) as f32;
                let mut pooled = vec![0.0f32; n_out * d_vit];
                for oy in 0..out_gh {
                    for ox in 0..out_gw {
                        let dst =
                            &mut pooled[(oy * out_gw + ox) * d_vit..(oy * out_gw + ox + 1) * d_vit];
                        for dy in 0..sf {
                            for dx in 0..sf {
                                let sy = oy * sf + dy;
                                let sx = ox * sf + dx;
                                let src =
                                    &spatial[(sy * gw + sx) * d_vit..(sy * gw + sx + 1) * d_vit];
                                for j in 0..d_vit {
                                    dst[j] += src[j];
                                }
                            }
                        }
                        for v in dst.iter_mut() {
                            *v *= inv_area;
                        }
                    }
                }
                let mut normed = vec![0.0f32; n_out * d_vit];
                apply_norm_rows(
                    &pooled,
                    &mut normed,
                    soft_emb_norm_w,
                    None,
                    self.cfg.norm_eps,
                    n_out,
                    d_vit,
                );
                let mut out = vec![0.0f32; n_out * d_proj];
                if *transposed {
                    vision_linear_transposed(input_proj, &normed, &mut out, n_out);
                } else {
                    vision_linear(pool, kt, input_proj, &normed, &mut out, n_out, scratch);
                }
                Ok((out, out_gh, out_gw))
            }
            VisionProjector::QwenVlMerger {
                merge_size,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                act,
            } => {
                let ms = (*merge_size).max(1);
                if !gh.is_multiple_of(ms) || !gw.is_multiple_of(ms) {
                    return Err(ModelError::InvalidConfig(format!(
                        "Qwen-VL grid {gh}x{gw} must be divisible by merge_size {ms}"
                    )));
                }
                let out_gh = gh / ms;
                let out_gw = gw / ms;
                let n_out = out_gh * out_gw;
                let merged_dim = d_vit * ms * ms;
                let merged = spatial_merge_concat(spatial, gh, gw, d_vit, ms);
                let mut h1 = vec![0.0f32; n_out * merged_dim];
                vision_linear(pool, kt, fc1_w, &merged, &mut h1, n_out, scratch);
                if let Some(b) = fc1_b {
                    add_bias_rows(&mut h1, b, n_out, merged_dim);
                }
                for v in &mut h1 {
                    *v = act.apply(*v);
                }
                let mut out = vec![0.0f32; n_out * d_proj];
                vision_linear(pool, kt, fc2_w, &h1, &mut out, n_out, scratch);
                if let Some(b) = fc2_b {
                    add_bias_rows(&mut out, b, n_out, d_proj);
                }
                Ok((out, out_gh, out_gw))
            }
            VisionProjector::PixtralMerger {
                merge_size,
                merging_w,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                act,
            } => {
                let ms = (*merge_size).max(1);
                if !gh.is_multiple_of(ms) || !gw.is_multiple_of(ms) {
                    return Err(ModelError::InvalidConfig(format!(
                        "Pixtral grid {gh}x{gw} must be divisible by merge_size {ms}"
                    )));
                }
                let out_gh = gh / ms;
                let out_gw = gw / ms;
                let n_out = out_gh * out_gw;
                let merged = spatial_merge_concat(spatial, gh, gw, d_vit, ms);
                let fc1_in = if let Some(mw) = merging_w {
                    let mut m_out = vec![0.0f32; n_out * d_vit];
                    vision_linear(pool, kt, mw, &merged, &mut m_out, n_out, scratch);
                    m_out
                } else {
                    merged
                };
                let mut h1 = vec![0.0f32; n_out * d_proj];
                vision_linear(pool, kt, fc1_w, &fc1_in, &mut h1, n_out, scratch);
                if let Some(b) = fc1_b {
                    add_bias_rows(&mut h1, b, n_out, d_proj);
                }
                for v in &mut h1 {
                    *v = act.apply(*v);
                }
                let mut out = vec![0.0f32; n_out * d_proj];
                vision_linear(pool, kt, fc2_w, &h1, &mut out, n_out, scratch);
                if let Some(b) = fc2_b {
                    add_bias_rows(&mut out, b, n_out, d_proj);
                }
                Ok((out, out_gh, out_gw))
            }
            VisionProjector::MlpProj {
                scale_factor,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                act,
            } => {
                let sf = (*scale_factor).max(1);
                if !gh.is_multiple_of(sf) || !gw.is_multiple_of(sf) {
                    return Err(ModelError::InvalidConfig(format!(
                        "MLP projector grid {gh}x{gw} must be divisible by scale_factor {sf}"
                    )));
                }
                let out_gh = gh / sf;
                let out_gw = gw / sf;
                let n_out = out_gh * out_gw;
                let inp = if sf == 1 {
                    spatial.to_vec()
                } else {
                    spatial_merge_concat(spatial, gh, gw, d_vit, sf)
                };
                let mut h1 = vec![0.0f32; n_out * fc1_w.rows];
                vision_linear(pool, kt, fc1_w, &inp, &mut h1, n_out, scratch);
                if let Some(b) = fc1_b {
                    add_bias_rows(&mut h1, b, n_out, fc1_w.rows);
                }
                if let Some(w2) = fc2_w {
                    for v in &mut h1 {
                        *v = act.apply(*v);
                    }
                    let mut out = vec![0.0f32; n_out * d_proj];
                    vision_linear(pool, kt, w2, &h1, &mut out, n_out, scratch);
                    if let Some(b) = fc2_b {
                        add_bias_rows(&mut out, b, n_out, d_proj);
                    }
                    Ok((out, out_gh, out_gw))
                } else {
                    Ok((h1, out_gh, out_gw))
                }
            }
        }
    }
}

// ============================================================================
// Vision Math Helpers (Deterministic, Zero-Unsafe)
// ============================================================================

/// Linear projection `out[t] = W @ x[t]` for `n_tok` tokens.
/// Uses `tm-cpu` SIMD/quantized `matmul_f32` when `cols % 32 == 0` and falls back to
/// exact per-row dequantized dot product for non-32-aligned dimensions (e.g. `3*14*14=588`).
fn vision_linear(
    pool: &ThreadPool,
    kt: &KernelTable,
    w: &RawMatView<'_>,
    x: &[f32],
    out: &mut [f32],
    n_tok: usize,
    scratch: &mut MatvecScratch,
) {
    let rows = w.rows;
    let cols = w.cols;
    debug_assert_eq!(x.len(), n_tok * cols);
    debug_assert_eq!(out.len(), n_tok * rows);
    if n_tok == 0 || rows == 0 {
        return;
    }
    if cols.is_multiple_of(32) {
        let mut off = 0usize;
        while off < n_tok {
            let b = (n_tok - off).min(MAX_CPU_BATCH);
            if b == 1 {
                matvec_f32(
                    pool,
                    kt,
                    w.dtype,
                    w.bytes,
                    &x[off * cols..(off + 1) * cols],
                    scratch,
                    &mut out[off * rows..(off + 1) * rows],
                    rows,
                    cols,
                );
            } else {
                matmul_f32(
                    pool,
                    kt,
                    w.dtype,
                    w.bytes,
                    &x[off * cols..(off + b) * cols],
                    scratch,
                    &mut out[off * rows..(off + b) * rows],
                    rows,
                    cols,
                    b,
                );
            }
            off += b;
        }
        return;
    }

    // Fallback for non-32-aligned `cols` (e.g. patch_embd `3 * 14 * 14 = 588` or tiny tests).
    let rb = w.dtype.row_bytes(cols);
    let mut row_f32 = vec![0.0f32; cols];
    for r in 0..rows {
        let w_row = if w.dtype == DType::F32 {
            // Decode F32 slice via dequantize_row_bytes (handles alignment cleanly without unsafe)
            dequantize_row_bytes(DType::F32, &w.bytes[r * rb..(r + 1) * rb], &mut row_f32);
            &row_f32[..]
        } else {
            dequantize_row_bytes(w.dtype, &w.bytes[r * rb..(r + 1) * rb], &mut row_f32);
            &row_f32[..]
        };
        for t in 0..n_tok {
            let x_row = &x[t * cols..(t + 1) * cols];
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += w_row[c] * x_row[c];
            }
            out[t * rows + r] = acc;
        }
    }
}

/// Transposed linear projection `out[t, o] = sum_i W[i, o] * x[t, i]` where `W` has shape `[in_dim, out_dim]`.
fn vision_linear_transposed(w: &RawMatView<'_>, x: &[f32], out: &mut [f32], n_tok: usize) {
    let in_dim = w.rows;
    let out_dim = w.cols;
    debug_assert_eq!(x.len(), n_tok * in_dim);
    debug_assert_eq!(out.len(), n_tok * out_dim);
    out.fill(0.0);
    let rb = w.dtype.row_bytes(out_dim);
    let mut row_f32 = vec![0.0f32; out_dim];
    for i in 0..in_dim {
        dequantize_row_bytes(w.dtype, &w.bytes[i * rb..(i + 1) * rb], &mut row_f32);
        for t in 0..n_tok {
            let xv = x[t * in_dim + i];
            let dst = &mut out[t * out_dim..(t + 1) * out_dim];
            for o in 0..out_dim {
                dst[o] += xv * row_f32[o];
            }
        }
    }
}

fn add_bias_rows(buf: &mut [f32], bias: &[f32], n_rows: usize, dim: usize) {
    for r in 0..n_rows {
        let row = &mut buf[r * dim..(r + 1) * dim];
        for j in 0..dim {
            row[j] += bias[j];
        }
    }
}

/// Applies LayerNorm (when `bias.is_some()`) or RMSNorm (when `bias.is_none()`) row-wise.
fn apply_norm_rows(
    src: &[f32],
    dst: &mut [f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    n_rows: usize,
    dim: usize,
) {
    if let Some(b) = bias {
        for r in 0..n_rows {
            let x = &src[r * dim..(r + 1) * dim];
            let y = &mut dst[r * dim..(r + 1) * dim];
            let mut mean = 0.0f32;
            for &v in x {
                mean += v;
            }
            mean /= dim as f32;
            let mut var = 0.0f32;
            for &v in x {
                let d = v - mean;
                var += d * d;
            }
            let inv_std = 1.0f32 / (var / dim as f32 + eps).sqrt();
            for j in 0..dim {
                y[j] = (x[j] - mean) * inv_std * weight[j] + b[j];
            }
        }
    } else {
        for r in 0..n_rows {
            let x = &src[r * dim..(r + 1) * dim];
            let y = &mut dst[r * dim..(r + 1) * dim];
            rms_norm(y, x, weight, eps);
        }
    }
}

/// Adds 2D learned position embeddings `[base_gh * base_gw, d_vit]` to `[gh * gw, d_vit]`,
/// performing bilinear interpolation if `(gh, gw) != (base_gh, base_gw)`.
fn add_pos_emb_2d(
    dst: &mut [f32],
    pos_emb: &[f32],
    base_gh: usize,
    base_gw: usize,
    gh: usize,
    gw: usize,
    d_vit: usize,
) {
    if gh == base_gh && gw == base_gw {
        for (a, &b) in dst.iter_mut().zip(pos_emb.iter()) {
            *a += b;
        }
        return;
    }
    let scale_y = if gh > 1 {
        (base_gh - 1) as f32 / (gh - 1) as f32
    } else {
        0.0
    };
    let scale_x = if gw > 1 {
        (base_gw - 1) as f32 / (gw - 1) as f32
    } else {
        0.0
    };
    for y in 0..gh {
        let sy = (y as f32 * scale_y).clamp(0.0, (base_gh - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(base_gh - 1);
        let wy = sy - y0 as f32;
        for x in 0..gw {
            let sx = (x as f32 * scale_x).clamp(0.0, (base_gw - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(base_gw - 1);
            let wx = sx - x0 as f32;
            let out = &mut dst[(y * gw + x) * d_vit..(y * gw + x + 1) * d_vit];
            let p00 = &pos_emb[(y0 * base_gw + x0) * d_vit..];
            let p01 = &pos_emb[(y0 * base_gw + x1) * d_vit..];
            let p10 = &pos_emb[(y1 * base_gw + x0) * d_vit..];
            let p11 = &pos_emb[(y1 * base_gw + x1) * d_vit..];
            let w00 = (1.0 - wy) * (1.0 - wx);
            let w01 = (1.0 - wy) * wx;
            let w10 = wy * (1.0 - wx);
            let w11 = wy * wx;
            for j in 0..d_vit {
                out[j] += w00 * p00[j] + w01 * p01[j] + w10 * p10[j] + w11 * p11[j];
            }
        }
    }
}

/// Builds 2D Vision RoPE `(cos, sin)` tables of shape `[n_seq, head_dim / 2]`.
/// First `head_dim / 4` pairs encode row (`y`), next `head_dim / 4` pairs encode col (`x`).
fn build_vision_rope_2d(
    gh: usize,
    gw: usize,
    has_cls: bool,
    head_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let quarter = half / 2;
    let n_seq = gh * gw + usize::from(has_cls);
    let mut cos_tbl = vec![1.0f32; n_seq * half];
    let mut sin_tbl = vec![0.0f32; n_seq * half];
    if quarter == 0 {
        return (cos_tbl, sin_tbl);
    }
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|i| 1.0f32 / theta.powf((2 * i) as f32 / half as f32))
        .collect();
    let offset = usize::from(has_cls);
    for r in 0..gh {
        for c in 0..gw {
            let t = offset + r * gw + c;
            let cos_row = &mut cos_tbl[t * half..(t + 1) * half];
            let sin_row = &mut sin_tbl[t * half..(t + 1) * half];
            for i in 0..quarter {
                let ang_r = r as f32 * inv_freq[i];
                cos_row[i] = ang_r.cos();
                sin_row[i] = ang_r.sin();
                let ang_c = c as f32 * inv_freq[i];
                cos_row[quarter + i] = ang_c.cos();
                sin_row[quarter + i] = ang_c.sin();
            }
        }
    }
    (cos_tbl, sin_tbl)
}

/// Applies NeoX-style split-half 2D Vision RoPE in-place to `buf` (`[n_seq, n_heads * head_dim]`).
fn apply_vision_rope_2d(
    buf: &mut [f32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    n_seq: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    let d_vit = n_heads * head_dim;
    for t in 0..n_seq {
        let cos_r = &cos_tbl[t * half..(t + 1) * half];
        let sin_r = &sin_tbl[t * half..(t + 1) * half];
        for h in 0..n_heads {
            let v = &mut buf[t * d_vit + h * head_dim..t * d_vit + (h + 1) * head_dim];
            for i in 0..half {
                let x0 = v[i];
                let x1 = v[i + half];
                let c = cos_r[i];
                let s = sin_r[i];
                v[i] = x0 * c - x1 * s;
                v[i + half] = x0 * s + x1 * c;
            }
        }
    }
}

/// Groups `ms × ms` spatially adjacent patches into `[gh/ms, gw/ms, ms*ms*d_vit]`.
fn spatial_merge_concat(
    spatial: &[f32],
    gh: usize,
    gw: usize,
    d_vit: usize,
    ms: usize,
) -> Vec<f32> {
    let out_gh = gh / ms;
    let out_gw = gw / ms;
    let merged_dim = d_vit * ms * ms;
    let mut out = vec![0.0f32; out_gh * out_gw * merged_dim];
    for oy in 0..out_gh {
        for ox in 0..out_gw {
            let dst =
                &mut out[(oy * out_gw + ox) * merged_dim..(oy * out_gw + ox + 1) * merged_dim];
            let mut k = 0usize;
            for dy in 0..ms {
                for dx in 0..ms {
                    let sy = oy * ms + dy;
                    let sx = ox * ms + dx;
                    let src = &spatial[(sy * gw + sx) * d_vit..(sy * gw + sx + 1) * d_vit];
                    dst[k * d_vit..(k + 1) * d_vit].copy_from_slice(src);
                    k += 1;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_and_bmp_roundtrip() {
        // P6 2x2 RGB
        let mut ppm = b"P6\n# comment\n2 2\n255\n".to_vec();
        let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 64, 32];
        ppm.extend_from_slice(&pixels);
        let img = RgbImage::from_ppm_bytes(&ppm).expect("valid P6 PPM");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.data, pixels);

        // P5 2x1 grayscale
        let pgm = b"P5 2 1 255\n\x10\xf0";
        let g = RgbImage::from_ppm_bytes(pgm).expect("valid P5 PGM");
        assert_eq!(g.data, vec![0x10, 0x10, 0x10, 0xf0, 0xf0, 0xf0]);

        // Uncompressed 24-bit 2x1 BMP (row stride padded to 8 bytes)
        let mut bmp = vec![0u8; 54 + 8];
        bmp[0..2].copy_from_slice(b"BM");
        bmp[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp[14..18].copy_from_slice(&40u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&2i32.to_le_bytes());
        bmp[22..26].copy_from_slice(&1i32.to_le_bytes());
        bmp[26..28].copy_from_slice(&1u16.to_le_bytes());
        bmp[28..30].copy_from_slice(&24u16.to_le_bytes());
        // Pixel 0: BGR = (10, 20, 30) -> RGB (30, 20, 10)
        // Pixel 1: BGR = (40, 50, 60) -> RGB (60, 50, 40)
        bmp[54..60].copy_from_slice(&[10, 20, 30, 40, 50, 60]);
        let b = RgbImage::from_bmp_bytes(&bmp).expect("valid BMP");
        assert_eq!(b.data, vec![30, 20, 10, 60, 50, 40]);
    }

    #[test]
    fn smart_resize_preserves_factor_and_bounds() {
        let (h, w) = smart_resize_dims(1000, 500, 28, 28 * 28 * 4, 28 * 28 * 256);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w <= 28 * 28 * 256);
        assert!(h * w >= 28 * 28 * 4);
        assert!(h > w);
    }
}
