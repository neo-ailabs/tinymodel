//! Vulkan GPU Multimodal Vision Encoder (`VisionVkRunner`, `P1-B2`).
//!
//! Executes the full `mmproj` ViT + Multimodal Projector pipeline on Vulkan GPUs
//! (NVIDIA Turing+, AMD RDNA, Intel ANV, Arm Mali Pixel 8 / RK3588, Pi 5 V3D, lavapipe)
//! with zero `unsafe` code inside this module (reusing `VulkanContext` safe dispatch helpers).
//!
//! Supported `clip.projector_type` families (100% parity with `tm-models::VisionModel`):
//! - `gemma3`, `gemma4`: SigLIP ViT + 2D avg-pool + `mm.soft_emb_norm` RMSNorm + projection
//! - `qwen2vl`, `qwen2.5vl`, `qwen3vl`, `qwen35`: 2D/3D patch conv + 2D RoPE + 2×2 merger + MLP
//! - `pixtral`, `mistral3`: 2D RoPE + Pre-RMSNorm + SwiGLU + spatial merger + MLP
//! - `smolvlm`, `idefics3`, `lfm2`, `mlp`, `siglip`, `linear`: SigLIP ViT + pixel-shuffle + MLP

use std::sync::Arc;

use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::VulkanError;
use crate::context::{Pipeline, Recorder, VkBuffer, VulkanContext};
use crate::kernels::{dtype_code, normalize_vk_mat_bytes};

const SPV_RMS_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv"));
const SPV_QK_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm.spv"));
const SPV_GEMM_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_tiled.spv"));
const SPV_GEMM_TILED_DP4A: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_tiled_dp4a.spv"));
const SPV_GEMM_KQUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_kquant.spv"));
const SPV_QUANTIZE_Q8_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantize_q8_1.spv"));
const SPV_VISION_OPS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vision_ops.spv"));
const SPV_VISION_BIDIR_ATTN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/vision_bidir_attn.spv"));
const SPV_VISION_GEMM_F32: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vision_gemm_f32.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionVkProjectorType {
    Gemma3,
    Gemma4,
    Qwen2Vl,
    Qwen25Vl,
    Qwen3Vl,
    Pixtral,
    SmolVlm,
    Mlp,
}

impl VisionVkProjectorType {
    fn from_gguf_str(s: &str) -> Result<Self, VulkanError> {
        match s {
            "gemma3" | "gemma3n" => Ok(Self::Gemma3),
            "gemma4" => Ok(Self::Gemma4),
            "qwen2vl_merger" | "qwen2vl" => Ok(Self::Qwen2Vl),
            "qwen2.5vl_merger" | "qwen25vl_merger" | "qwen2.5vl" => Ok(Self::Qwen25Vl),
            "qwen3vl_merger" | "qwen35_merger" | "qwen35moe_merger" | "qwen3vl" | "qwen35"
            | "qwen35moe" => Ok(Self::Qwen3Vl),
            "pixtral" | "mistral3" => Ok(Self::Pixtral),
            "smolvlm" | "idefics3" | "lfm2" => Ok(Self::SmolVlm),
            "mlp" | "siglip" | "linear" | "ldp" | "ldpv2" => Ok(Self::Mlp),
            other => Err(VulkanError::UnsupportedArch(format!(
                "unsupported clip.projector_type '{other}'"
            ))),
        }
    }

    fn uses_2d_rope(self) -> bool {
        matches!(
            self,
            Self::Qwen2Vl | Self::Qwen25Vl | Self::Qwen3Vl | Self::Pixtral
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisionActKind {
    GeluTanh = 0,
    GeluQuick = 1,
    GeluErf = 2,
    Silu = 3,
}

/// Parsed configuration for [`VisionVkRunner`].
#[derive(Debug, Clone)]
pub struct VisionVkConfig {
    pub projector_type: VisionVkProjectorType,
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
    pub use_2d_rope: bool,
    ffn_act: VisionActKind,
}

/// Projected visual embeddings produced on GPU by [`VisionVkRunner::encode_preprocessed`].
#[derive(Debug, Clone)]
pub struct VisionVkEmbeddings {
    pub embeddings: Vec<f32>,
    pub n_tokens: usize,
    pub d_llm: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub mrope_positions: Vec<[usize; 3]>,
}

struct VkVisionMat {
    buf: VkBuffer,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    aligned32: bool,
}

#[allow(clippy::large_enum_variant)]
enum VkAttnProj {
    Separate {
        wq: VkVisionMat,
        wk: VkVisionMat,
        wv: VkVisionMat,
        bq: Option<VkBuffer>,
        bk: Option<VkBuffer>,
        bv: Option<VkBuffer>,
    },
    FusedQkv {
        wqkv: VkVisionMat,
        bqkv: Option<VkBuffer>,
    },
}

enum VkVisionFfn {
    Dense {
        w_up: VkVisionMat,
        b_up: Option<VkBuffer>,
        w_down: VkVisionMat,
        b_down: Option<VkBuffer>,
    },
    Gated {
        w_gate: VkVisionMat,
        b_gate: Option<VkBuffer>,
        w_up: VkVisionMat,
        b_up: Option<VkBuffer>,
        w_down: VkVisionMat,
        b_down: Option<VkBuffer>,
    },
}

struct VkVisionLayer {
    ln1_w: VkBuffer,
    ln1_b: Option<VkBuffer>,
    attn: VkAttnProj,
    q_norm: Option<VkBuffer>,
    k_norm: Option<VkBuffer>,
    wo: VkVisionMat,
    bo: Option<VkBuffer>,
    ln2_w: VkBuffer,
    ln2_b: Option<VkBuffer>,
    ffn: VkVisionFfn,
}

enum VkProjector {
    GemmaPoolProj {
        scale_factor: usize,
        soft_emb_norm_w: VkBuffer,
        input_proj: VkVisionMat,
    },
    QwenVlMerger {
        merge_size: usize,
        fc1_w: VkVisionMat,
        fc1_b: Option<VkBuffer>,
        fc2_w: VkVisionMat,
        fc2_b: Option<VkBuffer>,
        act: VisionActKind,
    },
    PixtralMerger {
        merge_size: usize,
        merging_w: Option<VkVisionMat>,
        fc1_w: VkVisionMat,
        fc1_b: Option<VkBuffer>,
        fc2_w: VkVisionMat,
        fc2_b: Option<VkBuffer>,
        act: VisionActKind,
    },
    MlpProj {
        scale_factor: usize,
        fc1_w: VkVisionMat,
        fc1_b: Option<VkBuffer>,
        fc2_w: Option<VkVisionMat>,
        fc2_b: Option<VkBuffer>,
        act: VisionActKind,
    },
}

struct VisionPipes {
    rms_norm: Arc<Pipeline>,
    qk_norm: Arc<Pipeline>,
    layer_norm: Arc<Pipeline>,
    add_bias: Arc<Pipeline>,
    split_qkv: Arc<Pipeline>,
    rope_2d: Arc<Pipeline>,
    act_inplace: Arc<Pipeline>,
    gated_act_mul: Arc<Pipeline>,
    residual_add: Arc<Pipeline>,
    bidir_attn: Arc<Pipeline>,
    gemm_f32_gen: Arc<Pipeline>,
    quant_q8_1: Option<Arc<Pipeline>>,
}

/// Complete Vulkan GPU Vision Encoder for `mmproj` GGUF files.
pub struct VisionVkRunner {
    pub cfg: VisionVkConfig,
    ctx: VulkanContext,
    use_dp4a: bool,
    pipes: VisionPipes,
    dummy_buf: VkBuffer,
    patch_embd_w: VkVisionMat,
    patch_embd_w1: Option<VkVisionMat>,
    patch_embd_b: Option<VkBuffer>,
    class_embd: Option<Vec<f32>>,
    position_embd: Option<Vec<f32>>,
    pos_grid_h: usize,
    pos_grid_w: usize,
    pre_ln_w: Option<VkBuffer>,
    pre_ln_b: Option<VkBuffer>,
    layers: Vec<VkVisionLayer>,
    post_ln_w: Option<VkBuffer>,
    post_ln_b: Option<VkBuffer>,
    projector: VkProjector,
}

fn load_vec_1d(gguf: &GgufFile, name: &str, expected: usize) -> Result<Vec<f32>, VulkanError> {
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

fn upload_vec_1d(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    name: &str,
    expected: usize,
) -> Result<VkBuffer, VulkanError> {
    let v = load_vec_1d(gguf, name, expected)?;
    ctx.upload(&v)
}

fn opt_upload_vec_1d(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    name: &str,
    expected: usize,
) -> Result<Option<VkBuffer>, VulkanError> {
    if gguf.tensor_info(name).is_none() {
        return Ok(None);
    }
    Ok(Some(upload_vec_1d(ctx, gguf, name, expected)?))
}

fn opt_upload_vec_1d_any(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    names: &[&str],
    expected: usize,
) -> Result<Option<VkBuffer>, VulkanError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return Ok(Some(upload_vec_1d(ctx, gguf, name, expected)?));
        }
    }
    Ok(None)
}

fn upload_mat_from_raw(
    ctx: &VulkanContext,
    dtype: DType,
    raw_bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<VkVisionMat, VulkanError> {
    if cols.is_multiple_of(32) {
        let (eff_dtype, bytes) = normalize_vk_mat_bytes(dtype, raw_bytes, rows, cols)?;
        let _ = dtype_code(eff_dtype)?;
        let row_bytes = eff_dtype.row_bytes(cols);
        let buf = ctx.upload_bytes(&bytes)?;
        Ok(VkVisionMat {
            buf,
            dtype: eff_dtype,
            rows,
            cols,
            row_bytes,
            aligned32: true,
        })
    } else {
        // Dequantize non-32-aligned weights (e.g. patch_embd 588 or SigLIP d_ff 4304) to F32
        // so `vision_gemm_f32` executes them on GPU with zero CPU fallback.
        let rb = dtype.row_bytes(cols);
        let mut f32_data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            dequantize_row_bytes(
                dtype,
                &raw_bytes[r * rb..(r + 1) * rb],
                &mut f32_data[r * cols..(r + 1) * cols],
            );
        }
        let buf = ctx.upload(&f32_data)?;
        Ok(VkVisionMat {
            buf,
            dtype: DType::F32,
            rows,
            cols,
            row_bytes: cols * 4,
            aligned32: false,
        })
    }
}

fn upload_mat_2d(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<VkVisionMat, VulkanError> {
    let (info, bytes) = gguf.req_tensor(name, "vision", &[cols, rows])?;
    upload_mat_from_raw(ctx, info.dtype, bytes, rows, cols)
}

fn upload_mat_2d_any(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    names: &[&str],
    rows: usize,
    cols: usize,
) -> Result<VkVisionMat, VulkanError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return upload_mat_2d(ctx, gguf, name, rows, cols);
        }
    }
    Err(GgufError::MissingTensor {
        path: gguf.path().display().to_string(),
        tensor: names.join(" | "),
        layer: "vision".to_string(),
    }
    .into())
}

fn upload_patch_conv(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    name: &str,
    d_vit: usize,
    patch_size: usize,
) -> Result<VkVisionMat, VulkanError> {
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
    upload_mat_from_raw(ctx, info.dtype, bytes, d_vit, patch_dim)
}

impl VisionVkRunner {
    /// Loads and validates an `mmproj` GGUF vision encoder onto the default Vulkan device.
    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Self, VulkanError> {
        Self::from_gguf_with_context(VulkanContext::new()?, gguf)
    }

    /// Loads and validates an `mmproj` GGUF vision encoder onto an explicit [`VulkanContext`].
    pub fn from_gguf_with_context(
        ctx: VulkanContext,
        gguf: Arc<GgufFile>,
    ) -> Result<Self, VulkanError> {
        let gguf_ref: &GgufFile = &gguf;
        let proj_str = gguf_ref
            .get_meta("clip.projector_type")
            .and_then(|v| v.as_str())
            .unwrap_or("mlp");
        let projector_type = VisionVkProjectorType::from_gguf_str(proj_str)?;

        let patch_size = gguf_ref
            .get_meta("clip.vision.patch_size")
            .and_then(|v| v.as_usize())
            .unwrap_or(14);
        let image_size = gguf_ref
            .get_meta("clip.vision.image_size")
            .and_then(|v| v.as_usize())
            .unwrap_or(if projector_type == VisionVkProjectorType::Gemma3 {
                896
            } else {
                384
            });
        let d_vit = gguf_ref.req_usize("clip.vision.embedding_length")?;
        let d_ff = gguf_ref.req_usize("clip.vision.feed_forward_length")?;
        let d_proj = gguf_ref.req_usize("clip.vision.projection_dim")?;
        let n_layers = gguf_ref.req_usize("clip.vision.block_count")?;
        let n_heads = gguf_ref.req_usize("clip.vision.attention.head_count")?;
        if n_heads == 0 || !d_vit.is_multiple_of(n_heads) {
            return Err(VulkanError::InvalidDimensions(format!(
                "clip.vision.embedding_length ({d_vit}) not divisible by head_count ({n_heads})"
            )));
        }
        let head_dim = d_vit / n_heads;
        if head_dim > 256 || !head_dim.is_multiple_of(4) {
            return Err(VulkanError::InvalidDimensions(format!(
                "vision head_dim ({head_dim}) must be a multiple of 4 and <= 256"
            )));
        }
        let norm_eps = gguf_ref
            .get_meta("clip.vision.attention.layer_norm_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let rope_theta = gguf_ref
            .get_meta("clip.vision.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);

        let default_merge = match projector_type {
            VisionVkProjectorType::Gemma3 => 4,
            VisionVkProjectorType::Gemma4 | VisionVkProjectorType::SmolVlm => 3,
            VisionVkProjectorType::Qwen2Vl
            | VisionVkProjectorType::Qwen25Vl
            | VisionVkProjectorType::Qwen3Vl
            | VisionVkProjectorType::Pixtral => 2,
            VisionVkProjectorType::Mlp => 1,
        };
        let merge_size = gguf_ref
            .get_meta("clip.vision.spatial_merge_size")
            .and_then(|v| v.as_usize())
            .or_else(|| {
                gguf_ref
                    .get_meta("clip.vision.projector.scale_factor")
                    .and_then(|v| v.as_usize())
            })
            .unwrap_or(default_merge)
            .max(1);

        let use_gelu = gguf_ref
            .get_meta("clip.use_gelu")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let use_silu = gguf_ref
            .get_meta("clip.use_silu")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let ffn_act = if use_silu {
            VisionActKind::Silu
        } else if use_gelu {
            VisionActKind::GeluTanh
        } else {
            match projector_type {
                VisionVkProjectorType::Qwen25Vl | VisionVkProjectorType::Pixtral => {
                    VisionActKind::Silu
                }
                VisionVkProjectorType::Qwen2Vl => VisionActKind::GeluQuick,
                _ => VisionActKind::GeluTanh,
            }
        };
        let proj_act = match projector_type {
            VisionVkProjectorType::Qwen2Vl
            | VisionVkProjectorType::Qwen25Vl
            | VisionVkProjectorType::Qwen3Vl
            | VisionVkProjectorType::Pixtral => VisionActKind::GeluErf,
            _ => ffn_act,
        };
        let use_2d_rope = gguf_ref
            .get_meta("clip.vision.use_2d_rope")
            .and_then(|v| v.as_bool())
            .unwrap_or(projector_type.uses_2d_rope());

        let cfg = VisionVkConfig {
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
            use_2d_rope,
            ffn_act,
        };

        let sg = ctx.info().subgroup_size;
        let attn_wg = 8 * sg;
        let attn_n_vec = ((head_dim / 4).div_ceil(sg as usize)).max(1) as u32;
        let use_dp4a = ctx.info().int8_dot_accelerated;

        let pipes = VisionPipes {
            rms_norm: ctx.pipeline("rms_norm", 0, SPV_RMS_NORM, 3, 12, &[256])?,
            qk_norm: ctx.pipeline("qk_norm", 0, SPV_QK_NORM, 2, 8, &[64])?,
            layer_norm: ctx.pipeline("vision_ln", 0, SPV_VISION_OPS, 4, 16, &[256, 0])?,
            add_bias: ctx.pipeline("vision_bias", 1, SPV_VISION_OPS, 4, 16, &[256, 1])?,
            split_qkv: ctx.pipeline("vision_split_qkv", 2, SPV_VISION_OPS, 4, 16, &[256, 2])?,
            rope_2d: ctx.pipeline("vision_rope2d", 3, SPV_VISION_OPS, 4, 16, &[256, 3])?,
            act_inplace: ctx.pipeline("vision_act", 4, SPV_VISION_OPS, 4, 16, &[256, 4])?,
            gated_act_mul: ctx.pipeline("vision_gated_act", 5, SPV_VISION_OPS, 4, 16, &[256, 5])?,
            residual_add: ctx.pipeline("vision_res_add", 6, SPV_VISION_OPS, 4, 16, &[256, 6])?,
            bidir_attn: ctx.pipeline(
                "vision_bidir_attn",
                attn_wg | (attn_n_vec << 16),
                SPV_VISION_BIDIR_ATTN,
                4,
                12,
                &[attn_wg, attn_n_vec],
            )?,
            gemm_f32_gen: ctx.pipeline(
                "vision_gemm_f32",
                256,
                SPV_VISION_GEMM_F32,
                3,
                16,
                &[256],
            )?,
            quant_q8_1: if use_dp4a {
                Some(ctx.pipeline("quantize_q8_1", 256, SPV_QUANTIZE_Q8_1, 3, 4, &[256])?)
            } else {
                None
            },
        };
        let dummy_buf = ctx.upload(&[0.0f32; 4])?;

        // Upload patch embeddings + optional pos/cls embeddings
        let patch_embd_w =
            upload_patch_conv(&ctx, gguf_ref, "v.patch_embd.weight", d_vit, patch_size)?;
        let patch_embd_w1 = if gguf_ref.tensor_info("v.patch_embd.weight.1").is_some() {
            Some(upload_patch_conv(
                &ctx,
                gguf_ref,
                "v.patch_embd.weight.1",
                d_vit,
                patch_size,
            )?)
        } else {
            None
        };
        let patch_embd_b = opt_upload_vec_1d(&ctx, gguf_ref, "v.patch_embd.bias", d_vit)?;
        let class_embd = if gguf_ref.tensor_info("v.class_embd").is_some() {
            Some(load_vec_1d(gguf_ref, "v.class_embd", d_vit)?)
        } else {
            None
        };

        let (position_embd, pos_grid_h, pos_grid_w) =
            if gguf_ref.tensor_info("v.position_embd.weight").is_some() {
                let (info, bytes) = gguf_ref.tensor_bytes("v.position_embd.weight", "vision")?;
                let total: usize = info.dims.iter().product();
                if !total.is_multiple_of(d_vit) {
                    return Err(VulkanError::InvalidDimensions(format!(
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
                    let g0 = image_size / patch_size;
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

        let pre_ln_w = opt_upload_vec_1d(&ctx, gguf_ref, "v.pre_ln.weight", d_vit)?;
        let pre_ln_b = opt_upload_vec_1d(&ctx, gguf_ref, "v.pre_ln.bias", d_vit)?;

        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let ln1_w = upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ln1.weight"), d_vit)?;
            let ln1_b = opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ln1.bias"), d_vit)?;

            let qkv_name = format!("v.blk.{il}.attn_qkv.weight");
            let attn = if gguf_ref.tensor_info(&qkv_name).is_some() {
                let wqkv = upload_mat_2d(&ctx, gguf_ref, &qkv_name, 3 * d_vit, d_vit)?;
                let bqkv = opt_upload_vec_1d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.attn_qkv.bias"),
                    3 * d_vit,
                )?;
                VkAttnProj::FusedQkv { wqkv, bqkv }
            } else {
                let wq = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.attn_q.weight"),
                    d_vit,
                    d_vit,
                )?;
                let wk = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.attn_k.weight"),
                    d_vit,
                    d_vit,
                )?;
                let wv = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.attn_v.weight"),
                    d_vit,
                    d_vit,
                )?;
                let bq =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.attn_q.bias"), d_vit)?;
                let bk =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.attn_k.bias"), d_vit)?;
                let bv =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.attn_v.bias"), d_vit)?;
                VkAttnProj::Separate {
                    wq,
                    wk,
                    wv,
                    bq,
                    bk,
                    bv,
                }
            };

            let q_norm = opt_upload_vec_1d(
                &ctx,
                gguf_ref,
                &format!("v.blk.{il}.attn_q_norm.weight"),
                head_dim,
            )?;
            let k_norm = opt_upload_vec_1d(
                &ctx,
                gguf_ref,
                &format!("v.blk.{il}.attn_k_norm.weight"),
                head_dim,
            )?;
            let wo = upload_mat_2d(
                &ctx,
                gguf_ref,
                &format!("v.blk.{il}.attn_out.weight"),
                d_vit,
                d_vit,
            )?;
            let bo =
                opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.attn_out.bias"), d_vit)?;

            let ln2_w = upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ln2.weight"), d_vit)?;
            let ln2_b = opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ln2.bias"), d_vit)?;

            let gate_name = format!("v.blk.{il}.ffn_gate.weight");
            let ffn = if gguf_ref.tensor_info(&gate_name).is_some() {
                let w_gate = upload_mat_2d(&ctx, gguf_ref, &gate_name, d_ff, d_vit)?;
                let b_gate =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ffn_gate.bias"), d_ff)?;
                let w_up = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.ffn_up.weight"),
                    d_ff,
                    d_vit,
                )?;
                let b_up =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ffn_up.bias"), d_ff)?;
                let w_down = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.ffn_down.weight"),
                    d_vit,
                    d_ff,
                )?;
                let b_down =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ffn_down.bias"), d_vit)?;
                VkVisionFfn::Gated {
                    w_gate,
                    b_gate,
                    w_up,
                    b_up,
                    w_down,
                    b_down,
                }
            } else {
                let w_up = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.ffn_up.weight"),
                    d_ff,
                    d_vit,
                )?;
                let b_up =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ffn_up.bias"), d_ff)?;
                let w_down = upload_mat_2d(
                    &ctx,
                    gguf_ref,
                    &format!("v.blk.{il}.ffn_down.weight"),
                    d_vit,
                    d_ff,
                )?;
                let b_down =
                    opt_upload_vec_1d(&ctx, gguf_ref, &format!("v.blk.{il}.ffn_down.bias"), d_vit)?;
                VkVisionFfn::Dense {
                    w_up,
                    b_up,
                    w_down,
                    b_down,
                }
            };

            layers.push(VkVisionLayer {
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

        let post_ln_w = opt_upload_vec_1d(&ctx, gguf_ref, "v.post_ln.weight", d_vit)?;
        let post_ln_b = opt_upload_vec_1d(&ctx, gguf_ref, "v.post_ln.bias", d_vit)?;

        let projector = match projector_type {
            VisionVkProjectorType::Gemma3 | VisionVkProjectorType::Gemma4 => {
                let soft_emb_norm_w =
                    upload_vec_1d(&ctx, gguf_ref, "mm.soft_emb_norm.weight", d_vit)?;
                let (proj_info, proj_bytes) =
                    gguf_ref.tensor_bytes("mm.input_projection.weight", "vision")?;
                let input_proj = if proj_info.dims == [d_vit, d_proj] {
                    upload_mat_from_raw(&ctx, proj_info.dtype, proj_bytes, d_proj, d_vit)?
                } else if proj_info.dims == [d_proj, d_vit] {
                    // Transpose [d_vit, d_proj] -> [d_proj, d_vit] F32 at load time.
                    let rb = proj_info.dtype.row_bytes(d_proj);
                    let mut row = vec![0.0f32; d_proj];
                    let mut transposed = vec![0.0f32; d_proj * d_vit];
                    for i in 0..d_vit {
                        dequantize_row_bytes(
                            proj_info.dtype,
                            &proj_bytes[i * rb..(i + 1) * rb],
                            &mut row,
                        );
                        for o in 0..d_proj {
                            transposed[o * d_vit + i] = row[o];
                        }
                    }
                    let buf = ctx.upload(&transposed)?;
                    VkVisionMat {
                        buf,
                        dtype: DType::F32,
                        rows: d_proj,
                        cols: d_vit,
                        row_bytes: d_vit * 4,
                        aligned32: d_vit.is_multiple_of(32),
                    }
                } else {
                    return Err(GgufError::TensorShapeMismatch {
                        path: gguf_ref.path().display().to_string(),
                        tensor: "mm.input_projection.weight".to_string(),
                        layer: "vision".to_string(),
                        expected: vec![d_vit, d_proj],
                        actual: proj_info.dims.clone(),
                    }
                    .into());
                };
                VkProjector::GemmaPoolProj {
                    scale_factor: merge_size,
                    soft_emb_norm_w,
                    input_proj,
                }
            }
            VisionVkProjectorType::Qwen2Vl
            | VisionVkProjectorType::Qwen25Vl
            | VisionVkProjectorType::Qwen3Vl => {
                let merged_dim = d_vit * merge_size * merge_size;
                let fc1_w = upload_mat_2d(&ctx, gguf_ref, "mm.0.weight", merged_dim, merged_dim)?;
                let fc1_b = opt_upload_vec_1d(&ctx, gguf_ref, "mm.0.bias", merged_dim)?;
                let fc2_w = upload_mat_2d(&ctx, gguf_ref, "mm.2.weight", d_proj, merged_dim)?;
                let fc2_b = opt_upload_vec_1d(&ctx, gguf_ref, "mm.2.bias", d_proj)?;
                VkProjector::QwenVlMerger {
                    merge_size,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    act: proj_act,
                }
            }
            VisionVkProjectorType::Pixtral => {
                let merged_dim = d_vit * merge_size * merge_size;
                let merging_w = if gguf_ref.tensor_info("mm.patch_merger.weight").is_some() {
                    Some(upload_mat_2d(
                        &ctx,
                        gguf_ref,
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
                let fc1_w = upload_mat_2d_any(
                    &ctx,
                    gguf_ref,
                    &["mm.1.weight", "mm.0.weight"],
                    d_proj,
                    fc1_in,
                )?;
                let fc1_b =
                    opt_upload_vec_1d_any(&ctx, gguf_ref, &["mm.1.bias", "mm.0.bias"], d_proj)?;
                let fc2_w = upload_mat_2d_any(
                    &ctx,
                    gguf_ref,
                    &["mm.2.weight", "mm.3.weight"],
                    d_proj,
                    d_proj,
                )?;
                let fc2_b =
                    opt_upload_vec_1d_any(&ctx, gguf_ref, &["mm.2.bias", "mm.3.bias"], d_proj)?;
                VkProjector::PixtralMerger {
                    merge_size,
                    merging_w,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    act: proj_act,
                }
            }
            VisionVkProjectorType::SmolVlm | VisionVkProjectorType::Mlp => {
                let scale = merge_size;
                let in_dim = d_vit * scale * scale;
                let fc1_name = if gguf_ref.tensor_info("mm.0.weight").is_some() {
                    "mm.0.weight"
                } else {
                    "mm.input_projection.weight"
                };
                let (fc1_info, fc1_bytes) = gguf_ref.tensor_bytes(fc1_name, "vision")?;
                if fc1_info.dims.len() != 2 || fc1_info.dims[0] != in_dim {
                    return Err(GgufError::TensorShapeMismatch {
                        path: gguf_ref.path().display().to_string(),
                        tensor: fc1_info.name.clone(),
                        layer: "vision".to_string(),
                        expected: vec![in_dim, d_proj],
                        actual: fc1_info.dims.clone(),
                    }
                    .into());
                }
                let fc1_out = fc1_info.dims[1];
                let fc1_w = upload_mat_from_raw(&ctx, fc1_info.dtype, fc1_bytes, fc1_out, in_dim)?;
                let fc1_b = opt_upload_vec_1d(&ctx, gguf_ref, "mm.0.bias", fc1_out)?;
                let (fc2_w, fc2_b) = if gguf_ref.tensor_info("mm.2.weight").is_some() {
                    (
                        Some(upload_mat_2d(
                            &ctx,
                            gguf_ref,
                            "mm.2.weight",
                            d_proj,
                            fc1_out,
                        )?),
                        opt_upload_vec_1d(&ctx, gguf_ref, "mm.2.bias", d_proj)?,
                    )
                } else {
                    (None, None)
                };
                VkProjector::MlpProj {
                    scale_factor: scale,
                    fc1_w,
                    fc1_b,
                    fc2_w,
                    fc2_b,
                    act: proj_act,
                }
            }
        };

        Ok(Self {
            cfg,
            ctx,
            use_dp4a,
            pipes,
            dummy_buf,
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

    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    #[allow(clippy::too_many_arguments)]
    fn record_linear(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkVisionMat,
        x: &VkBuffer,
        y: &VkBuffer,
        n_tok: usize,
        add_residual: bool,
        xq_xd: Option<(&VkBuffer, &VkBuffer)>,
    ) -> Result<(), VulkanError> {
        let add_res = u32::from(add_residual);
        if !mat.aligned32 {
            let push = [mat.cols as u32, mat.rows as u32, n_tok as u32, add_res];
            rec.dispatch_xy(
                &self.pipes.gemm_f32_gen,
                &[&mat.buf, x, y],
                &push,
                (mat.rows as u32).div_ceil(16),
                (n_tok as u32).div_ceil(16),
            )?;
            return Ok(());
        }
        let dt = dtype_code(mat.dtype)?;
        let push = [
            mat.cols as u32,
            mat.rows as u32,
            n_tok as u32,
            add_res,
            (mat.row_bytes / 2) as u32,
        ];
        if self.use_dp4a
            && matches!(
                mat.dtype,
                DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K
            )
            && let (Some(q_pipe), Some((xq, xd))) = (&self.pipes.quant_q8_1, xq_xd)
        {
            let total_blocks = (n_tok * (mat.cols / 32)) as u32;
            rec.dispatch(
                q_pipe,
                &[x, xq, xd],
                &[total_blocks],
                total_blocks.div_ceil(8),
            )?;
            let pipe = self.ctx.pipeline(
                "gemm_tiled_dp4a",
                dt,
                SPV_GEMM_TILED_DP4A,
                4,
                20,
                &[128, dt],
            )?;
            rec.dispatch_xy(
                &pipe,
                &[&mat.buf, xq, xd, y],
                &push,
                (mat.rows as u32).div_ceil(64),
                (n_tok as u32).div_ceil(32),
            )?;
            return Ok(());
        }
        match mat.dtype {
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => {
                let pipe =
                    self.ctx
                        .pipeline("gemm_tiled", dt, SPV_GEMM_TILED, 3, 20, &[128, dt])?;
                rec.dispatch_xy(
                    &pipe,
                    &[&mat.buf, x, y],
                    &push,
                    (mat.rows as u32).div_ceil(64),
                    (n_tok as u32).div_ceil(32),
                )?;
            }
            DType::Q5K | DType::Q6K => {
                let wg = 128u32;
                let sg = self.ctx.info().subgroup_size;
                let rows_per_wg = (wg / sg).max(1) * 2;
                let pipe = self.ctx.pipeline(
                    "gemm_kquant",
                    dt | (wg << 8),
                    SPV_GEMM_KQUANT,
                    3,
                    20,
                    &[wg, dt],
                )?;
                rec.dispatch_xy(
                    &pipe,
                    &[&mat.buf, x, y],
                    &push,
                    (mat.rows as u32).div_ceil(rows_per_wg),
                    n_tok as u32,
                )?;
            }
            other => return Err(VulkanError::UnsupportedDType(other)),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_norm(
        &self,
        rec: &mut Recorder<'_>,
        x: &VkBuffer,
        w: &VkBuffer,
        b: Option<&VkBuffer>,
        y: &VkBuffer,
        n_tok: usize,
        dim: usize,
    ) -> Result<(), VulkanError> {
        let eps_bits = self.cfg.norm_eps.to_bits();
        if let Some(bias) = b {
            rec.dispatch(
                &self.pipes.layer_norm,
                &[x, w, bias, y],
                &[dim as u32, eps_bits, 0, 0],
                n_tok as u32,
            )
        } else {
            rec.dispatch(
                &self.pipes.rms_norm,
                &[x, w, y],
                &[dim as u32, 0, eps_bits],
                n_tok as u32,
            )
        }
    }

    fn record_bias(
        &self,
        rec: &mut Recorder<'_>,
        y: &VkBuffer,
        b: &VkBuffer,
        n_tok: usize,
        dim: usize,
    ) -> Result<(), VulkanError> {
        let total = (n_tok * dim) as u32;
        rec.dispatch(
            &self.pipes.add_bias,
            &[y, b, &self.dummy_buf, &self.dummy_buf],
            &[dim as u32, total, 0, 0],
            total.div_ceil(256),
        )
    }

    /// Encodes a preprocessed planar CHW `[3, height, width]` image on Vulkan GPU.
    pub fn encode_preprocessed(
        &mut self,
        pixels_chw: &[f32],
        width: usize,
        height: usize,
    ) -> Result<VisionVkEmbeddings, VulkanError> {
        let p = self.cfg.patch_size;
        if width == 0 || height == 0 || !width.is_multiple_of(p) || !height.is_multiple_of(p) {
            return Err(VulkanError::InvalidDimensions(format!(
                "preprocessed image {width}x{height} must be divisible by patch_size {p}"
            )));
        }
        if pixels_chw.len() != 3 * width * height {
            return Err(VulkanError::InvalidDimensions(format!(
                "pixels_chw len {} != 3 * {width} * {height}",
                pixels_chw.len()
            )));
        }
        let gh = height / p;
        let gw = width / p;
        let n_patches = gh * gw;
        let d_vit = self.cfg.d_vit;
        let d_ff = self.cfg.d_ff;
        let n_heads = self.cfg.n_heads;
        let hd = self.cfg.head_dim;
        let patch_dim = 3 * p * p;

        // 1. Extract patches on CPU ([C, pH, pW]) and run patch_embd on GPU.
        let mut patch_buf = vec![0.0f32; n_patches * patch_dim];
        let hw = height * width;
        for py in 0..gh {
            for px in 0..gw {
                let pidx = py * gw + px;
                let dst = &mut patch_buf[pidx * patch_dim..(pidx + 1) * patch_dim];
                let mut k = 0usize;
                for c in 0..3 {
                    let ch_off = c * hw;
                    for dy in 0..p {
                        let row_off = ch_off + (py * p + dy) * width + px * p;
                        for dx in 0..p {
                            dst[k] = pixels_chw[row_off + dx];
                            k += 1;
                        }
                    }
                }
            }
        }

        let gpu_patches = self.ctx.upload(&patch_buf)?;
        let gpu_patch_out = self.ctx.alloc_buffer((n_patches * d_vit * 4) as u64)?;
        let max_q8_cols = d_vit
            .max(d_ff)
            .max(patch_dim)
            .max(d_vit * self.cfg.merge_size * self.cfg.merge_size)
            .max(self.cfg.d_proj);
        let q8_scratch = if self.use_dp4a {
            let max_blocks = max_q8_cols.div_ceil(32);
            let xq = self
                .ctx
                .alloc_buffer(((n_patches + 1) * max_blocks * 32) as u64)?;
            let xd = self
                .ctx
                .alloc_buffer(((n_patches + 1) * max_blocks * 8) as u64)?;
            Some((xq, xd))
        } else {
            None
        };
        let q8_ref = q8_scratch.as_ref().map(|(a, b)| (a, b));

        self.ctx.run(|rec| {
            self.record_linear(
                rec,
                &self.patch_embd_w,
                &gpu_patches,
                &gpu_patch_out,
                n_patches,
                false,
                q8_ref,
            )?;
            if let Some(ref w1) = self.patch_embd_w1 {
                self.record_linear(
                    rec,
                    w1,
                    &gpu_patches,
                    &gpu_patch_out,
                    n_patches,
                    true,
                    q8_ref,
                )?;
            }
            if let Some(ref b) = self.patch_embd_b {
                self.record_bias(rec, &gpu_patch_out, b, n_patches, d_vit)?;
            }
            Ok(())
        })?;

        // 2. Assemble CLS + learned PosEmb (bilinearly interpolated if grid differs).
        let mut patch_emb = self.ctx.read_f32(&gpu_patch_out, n_patches * d_vit)?;
        let has_cls = self.class_embd.is_some();
        let n_seq = n_patches + usize::from(has_cls);
        let mut hidden_host = vec![0.0f32; n_seq * d_vit];
        if let Some(ref cls) = self.class_embd {
            hidden_host[..d_vit].copy_from_slice(cls);
        }
        hidden_host[usize::from(has_cls) * d_vit..].copy_from_slice(&patch_emb);
        patch_emb.clear();

        if let Some(ref pos_emb) = self.position_embd {
            let n_pos = pos_emb.len() / d_vit;
            if has_cls && n_pos == self.pos_grid_h * self.pos_grid_w + 1 {
                for j in 0..d_vit {
                    hidden_host[j] += pos_emb[j];
                }
                add_pos_emb_2d(
                    &mut hidden_host[d_vit..],
                    &pos_emb[d_vit..],
                    self.pos_grid_h,
                    self.pos_grid_w,
                    gh,
                    gw,
                    d_vit,
                );
            } else {
                add_pos_emb_2d(
                    &mut hidden_host[usize::from(has_cls) * d_vit..],
                    pos_emb,
                    self.pos_grid_h,
                    self.pos_grid_w,
                    gh,
                    gw,
                    d_vit,
                );
            }
        }

        // Upload initial hidden sequence + optional 2D Vision RoPE tables.
        let hidden_buf = self.ctx.upload(&hidden_host)?;
        let norm_buf = self.ctx.alloc_buffer((n_seq * d_vit * 4) as u64)?;
        let q_buf = self.ctx.alloc_buffer((n_seq * d_vit * 4) as u64)?;
        let k_buf = self.ctx.alloc_buffer((n_seq * d_vit * 4) as u64)?;
        let v_buf = self.ctx.alloc_buffer((n_seq * d_vit * 4) as u64)?;
        let any_fused_qkv = self
            .layers
            .iter()
            .any(|l| matches!(l.attn, VkAttnProj::FusedQkv { .. }));
        let qkv_buf = if any_fused_qkv {
            Some(self.ctx.alloc_buffer((n_seq * 3 * d_vit * 4) as u64)?)
        } else {
            None
        };
        let attn_out_buf = self.ctx.alloc_buffer((n_seq * d_vit * 4) as u64)?;
        let proj_buf = self.ctx.alloc_buffer((n_seq * d_vit * 4) as u64)?;
        let ff_up_buf = self.ctx.alloc_buffer((n_seq * d_ff * 4) as u64)?;
        let any_gated = self
            .layers
            .iter()
            .any(|l| matches!(l.ffn, VkVisionFfn::Gated { .. }));
        let ff_gate_buf = if any_gated {
            Some(self.ctx.alloc_buffer((n_seq * d_ff * 4) as u64)?)
        } else {
            None
        };

        let rope_bufs = if self.cfg.use_2d_rope {
            let (cos_tbl, sin_tbl) = build_vision_rope_2d(gh, gw, has_cls, hd, self.cfg.rope_theta);
            Some((self.ctx.upload(&cos_tbl)?, self.ctx.upload(&sin_tbl)?))
        } else {
            None
        };

        // Optional pre-LN
        if let Some(ref w) = self.pre_ln_w {
            self.ctx.run(|rec| {
                self.record_norm(
                    rec,
                    &hidden_buf,
                    w,
                    self.pre_ln_b.as_ref(),
                    &norm_buf,
                    n_seq,
                    d_vit,
                )
            })?;
            // Copy normed -> hidden via residual or swap: swap is even cleaner, or just copy on GPU:
            let normed_host = self.ctx.read_f32(&norm_buf, n_seq * d_vit)?;
            self.ctx
                .write_bytes(&hidden_buf, 0, &bytemuck_f32_bytes(&normed_host))?;
        }

        // 3. ViT Transformer blocks (1 layer per GPU submit -> zero desc_pool overflow & TDR-safe).
        let eps_bits = self.cfg.norm_eps.to_bits();
        let ffn_act_u32 = self.cfg.ffn_act as u32;
        for layer in &self.layers {
            self.ctx.run(|rec| {
                // Pre-norm 1
                self.record_norm(
                    rec,
                    &hidden_buf,
                    &layer.ln1_w,
                    layer.ln1_b.as_ref(),
                    &norm_buf,
                    n_seq,
                    d_vit,
                )?;

                // QKV projection
                match &layer.attn {
                    VkAttnProj::Separate {
                        wq,
                        wk,
                        wv,
                        bq,
                        bk,
                        bv,
                    } => {
                        self.record_linear(rec, wq, &norm_buf, &q_buf, n_seq, false, q8_ref)?;
                        self.record_linear(rec, wk, &norm_buf, &k_buf, n_seq, false, q8_ref)?;
                        self.record_linear(rec, wv, &norm_buf, &v_buf, n_seq, false, q8_ref)?;
                        if let Some(b) = bq {
                            self.record_bias(rec, &q_buf, b, n_seq, d_vit)?;
                        }
                        if let Some(b) = bk {
                            self.record_bias(rec, &k_buf, b, n_seq, d_vit)?;
                        }
                        if let Some(b) = bv {
                            self.record_bias(rec, &v_buf, b, n_seq, d_vit)?;
                        }
                    }
                    VkAttnProj::FusedQkv { wqkv, bqkv } => {
                        let qkv = qkv_buf.as_ref().expect("qkv_buf allocated");
                        self.record_linear(rec, wqkv, &norm_buf, qkv, n_seq, false, q8_ref)?;
                        if let Some(b) = bqkv {
                            self.record_bias(rec, qkv, b, n_seq, 3 * d_vit)?;
                        }
                        let total = (n_seq * d_vit) as u32;
                        rec.dispatch(
                            &self.pipes.split_qkv,
                            &[qkv, &q_buf, &k_buf, &v_buf],
                            &[d_vit as u32, total, 0, 0],
                            total.div_ceil(256),
                        )?;
                    }
                }

                // Optional per-head QK-Norm
                if let Some(ref qn) = layer.q_norm {
                    rec.dispatch_xy(
                        &self.pipes.qk_norm,
                        &[&q_buf, qn],
                        &[hd as u32, eps_bits],
                        n_heads as u32,
                        n_seq as u32,
                    )?;
                }
                if let Some(ref kn) = layer.k_norm {
                    rec.dispatch_xy(
                        &self.pipes.qk_norm,
                        &[&k_buf, kn],
                        &[hd as u32, eps_bits],
                        n_heads as u32,
                        n_seq as u32,
                    )?;
                }

                // Optional 2D Vision RoPE
                if let Some((ref cos_b, ref sin_b)) = rope_bufs {
                    let total_pairs = (n_seq * n_heads * (hd / 2)) as u32;
                    let push = [n_heads as u32, hd as u32, total_pairs, 0];
                    rec.dispatch(
                        &self.pipes.rope_2d,
                        &[&q_buf, cos_b, sin_b, &self.dummy_buf],
                        &push,
                        total_pairs.div_ceil(256),
                    )?;
                    rec.dispatch(
                        &self.pipes.rope_2d,
                        &[&k_buf, cos_b, sin_b, &self.dummy_buf],
                        &push,
                        total_pairs.div_ceil(256),
                    )?;
                }

                // Bidirectional Full Multi-Head Self-Attention
                rec.dispatch_xy(
                    &self.pipes.bidir_attn,
                    &[&q_buf, &k_buf, &v_buf, &attn_out_buf],
                    &[n_heads as u32, hd as u32, n_seq as u32],
                    n_heads as u32,
                    (n_seq as u32).div_ceil(8),
                )?;

                // Output projection + bias + residual
                self.record_linear(
                    rec,
                    &layer.wo,
                    &attn_out_buf,
                    &proj_buf,
                    n_seq,
                    false,
                    q8_ref,
                )?;
                if let Some(ref b) = layer.bo {
                    self.record_bias(rec, &proj_buf, b, n_seq, d_vit)?;
                }
                let tot_vit = (n_seq * d_vit) as u32;
                rec.dispatch(
                    &self.pipes.residual_add,
                    &[&hidden_buf, &proj_buf, &self.dummy_buf, &self.dummy_buf],
                    &[tot_vit, 0, 0, 0],
                    tot_vit.div_ceil(256),
                )?;

                // Pre-norm 2 + FFN + bias + residual
                self.record_norm(
                    rec,
                    &hidden_buf,
                    &layer.ln2_w,
                    layer.ln2_b.as_ref(),
                    &norm_buf,
                    n_seq,
                    d_vit,
                )?;
                let tot_ff = (n_seq * d_ff) as u32;
                match &layer.ffn {
                    VkVisionFfn::Dense {
                        w_up,
                        b_up,
                        w_down,
                        b_down,
                    } => {
                        self.record_linear(rec, w_up, &norm_buf, &ff_up_buf, n_seq, false, q8_ref)?;
                        if let Some(b) = b_up {
                            self.record_bias(rec, &ff_up_buf, b, n_seq, d_ff)?;
                        }
                        rec.dispatch(
                            &self.pipes.act_inplace,
                            &[
                                &ff_up_buf,
                                &self.dummy_buf,
                                &self.dummy_buf,
                                &self.dummy_buf,
                            ],
                            &[tot_ff, ffn_act_u32, 0, 0],
                            tot_ff.div_ceil(256),
                        )?;
                        self.record_linear(
                            rec, w_down, &ff_up_buf, &proj_buf, n_seq, false, q8_ref,
                        )?;
                        if let Some(b) = b_down {
                            self.record_bias(rec, &proj_buf, b, n_seq, d_vit)?;
                        }
                    }
                    VkVisionFfn::Gated {
                        w_gate,
                        b_gate,
                        w_up,
                        b_up,
                        w_down,
                        b_down,
                    } => {
                        let fg = ff_gate_buf.as_ref().expect("ff_gate_buf allocated");
                        self.record_linear(rec, w_gate, &norm_buf, fg, n_seq, false, q8_ref)?;
                        if let Some(b) = b_gate {
                            self.record_bias(rec, fg, b, n_seq, d_ff)?;
                        }
                        self.record_linear(rec, w_up, &norm_buf, &ff_up_buf, n_seq, false, q8_ref)?;
                        if let Some(b) = b_up {
                            self.record_bias(rec, &ff_up_buf, b, n_seq, d_ff)?;
                        }
                        rec.dispatch(
                            &self.pipes.gated_act_mul,
                            &[fg, &ff_up_buf, &self.dummy_buf, &self.dummy_buf],
                            &[tot_ff, ffn_act_u32, 0, 0],
                            tot_ff.div_ceil(256),
                        )?;
                        self.record_linear(rec, w_down, fg, &proj_buf, n_seq, false, q8_ref)?;
                        if let Some(b) = b_down {
                            self.record_bias(rec, &proj_buf, b, n_seq, d_vit)?;
                        }
                    }
                }
                rec.dispatch(
                    &self.pipes.residual_add,
                    &[&hidden_buf, &proj_buf, &self.dummy_buf, &self.dummy_buf],
                    &[tot_vit, 0, 0, 0],
                    tot_vit.div_ceil(256),
                )?;
                Ok(())
            })?;
        }

        // 4. Optional Post-LN on GPU (strip CLS token if present)
        let spatial = if let Some(ref w) = self.post_ln_w {
            self.ctx.run(|rec| {
                self.record_norm(
                    rec,
                    &hidden_buf,
                    w,
                    self.post_ln_b.as_ref(),
                    &norm_buf,
                    n_seq,
                    d_vit,
                )
            })?;
            let all = self.ctx.read_f32(&norm_buf, n_seq * d_vit)?;
            all[usize::from(has_cls) * d_vit..].to_vec()
        } else {
            let all = self.ctx.read_f32(&hidden_buf, n_seq * d_vit)?;
            all[usize::from(has_cls) * d_vit..].to_vec()
        };

        // 5. Multimodal Projector (spatial pool/merge on CPU, linear/MLP layers on GPU)
        let (embeddings, out_gh, out_gw) = self.run_projector_gpu(&spatial, gh, gw, q8_ref)?;
        let n_tokens = out_gh * out_gw;
        let mut mrope_positions = Vec::with_capacity(n_tokens);
        for r in 0..out_gh {
            for c in 0..out_gw {
                mrope_positions.push([0, r, c]);
            }
        }
        Ok(VisionVkEmbeddings {
            embeddings,
            n_tokens,
            d_llm: self.cfg.d_proj,
            grid_h: out_gh,
            grid_w: out_gw,
            mrope_positions,
        })
    }

    fn run_projector_gpu(
        &self,
        spatial: &[f32],
        gh: usize,
        gw: usize,
        q8_ref: Option<(&VkBuffer, &VkBuffer)>,
    ) -> Result<(Vec<f32>, usize, usize), VulkanError> {
        let d_vit = self.cfg.d_vit;
        let d_proj = self.cfg.d_proj;

        match &self.projector {
            VkProjector::GemmaPoolProj {
                scale_factor,
                soft_emb_norm_w,
                input_proj,
            } => {
                let sf = (*scale_factor).max(1);
                if !gh.is_multiple_of(sf) || !gw.is_multiple_of(sf) {
                    return Err(VulkanError::InvalidDimensions(format!(
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
                                let src = &spatial[((oy * sf + dy) * gw + (ox * sf + dx)) * d_vit
                                    ..((oy * sf + dy) * gw + (ox * sf + dx) + 1) * d_vit];
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
                let in_b = self.ctx.upload(&pooled)?;
                let norm_b = self.ctx.alloc_buffer((n_out * d_vit * 4) as u64)?;
                let out_b = self.ctx.alloc_buffer((n_out * d_proj * 4) as u64)?;
                self.ctx.run(|rec| {
                    self.record_norm(rec, &in_b, soft_emb_norm_w, None, &norm_b, n_out, d_vit)?;
                    self.record_linear(rec, input_proj, &norm_b, &out_b, n_out, false, q8_ref)
                })?;
                Ok((self.ctx.read_f32(&out_b, n_out * d_proj)?, out_gh, out_gw))
            }
            VkProjector::QwenVlMerger {
                merge_size,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                act,
            } => {
                let ms = (*merge_size).max(1);
                if !gh.is_multiple_of(ms) || !gw.is_multiple_of(ms) {
                    return Err(VulkanError::InvalidDimensions(format!(
                        "Qwen-VL grid {gh}x{gw} must be divisible by merge_size {ms}"
                    )));
                }
                let out_gh = gh / ms;
                let out_gw = gw / ms;
                let n_out = out_gh * out_gw;
                let merged_dim = d_vit * ms * ms;
                let merged = spatial_merge_concat(spatial, gh, gw, d_vit, ms);
                let in_b = self.ctx.upload(&merged)?;
                let h1_b = self.ctx.alloc_buffer((n_out * merged_dim * 4) as u64)?;
                let out_b = self.ctx.alloc_buffer((n_out * d_proj * 4) as u64)?;
                let tot_h1 = (n_out * merged_dim) as u32;
                self.ctx.run(|rec| {
                    self.record_linear(rec, fc1_w, &in_b, &h1_b, n_out, false, q8_ref)?;
                    if let Some(b) = fc1_b {
                        self.record_bias(rec, &h1_b, b, n_out, merged_dim)?;
                    }
                    rec.dispatch(
                        &self.pipes.act_inplace,
                        &[&h1_b, &self.dummy_buf, &self.dummy_buf, &self.dummy_buf],
                        &[tot_h1, *act as u32, 0, 0],
                        tot_h1.div_ceil(256),
                    )?;
                    self.record_linear(rec, fc2_w, &h1_b, &out_b, n_out, false, q8_ref)?;
                    if let Some(b) = fc2_b {
                        self.record_bias(rec, &out_b, b, n_out, d_proj)?;
                    }
                    Ok(())
                })?;
                Ok((self.ctx.read_f32(&out_b, n_out * d_proj)?, out_gh, out_gw))
            }
            VkProjector::PixtralMerger {
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
                    return Err(VulkanError::InvalidDimensions(format!(
                        "Pixtral grid {gh}x{gw} must be divisible by merge_size {ms}"
                    )));
                }
                let out_gh = gh / ms;
                let out_gw = gw / ms;
                let n_out = out_gh * out_gw;
                let merged = spatial_merge_concat(spatial, gh, gw, d_vit, ms);
                let in_b = self.ctx.upload(&merged)?;
                let m_b = if merging_w.is_some() {
                    Some(self.ctx.alloc_buffer((n_out * d_vit * 4) as u64)?)
                } else {
                    None
                };
                let h1_b = self.ctx.alloc_buffer((n_out * d_proj * 4) as u64)?;
                let out_b = self.ctx.alloc_buffer((n_out * d_proj * 4) as u64)?;
                let tot_h1 = (n_out * d_proj) as u32;
                self.ctx.run(|rec| {
                    let fc1_in = if let (Some(mw), Some(mb)) = (merging_w, &m_b) {
                        self.record_linear(rec, mw, &in_b, mb, n_out, false, q8_ref)?;
                        mb
                    } else {
                        &in_b
                    };
                    self.record_linear(rec, fc1_w, fc1_in, &h1_b, n_out, false, q8_ref)?;
                    if let Some(b) = fc1_b {
                        self.record_bias(rec, &h1_b, b, n_out, d_proj)?;
                    }
                    rec.dispatch(
                        &self.pipes.act_inplace,
                        &[&h1_b, &self.dummy_buf, &self.dummy_buf, &self.dummy_buf],
                        &[tot_h1, *act as u32, 0, 0],
                        tot_h1.div_ceil(256),
                    )?;
                    self.record_linear(rec, fc2_w, &h1_b, &out_b, n_out, false, q8_ref)?;
                    if let Some(b) = fc2_b {
                        self.record_bias(rec, &out_b, b, n_out, d_proj)?;
                    }
                    Ok(())
                })?;
                Ok((self.ctx.read_f32(&out_b, n_out * d_proj)?, out_gh, out_gw))
            }
            VkProjector::MlpProj {
                scale_factor,
                fc1_w,
                fc1_b,
                fc2_w,
                fc2_b,
                act,
            } => {
                let sf = (*scale_factor).max(1);
                if !gh.is_multiple_of(sf) || !gw.is_multiple_of(sf) {
                    return Err(VulkanError::InvalidDimensions(format!(
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
                let in_b = self.ctx.upload(&inp)?;
                let h1_b = self.ctx.alloc_buffer((n_out * fc1_w.rows * 4) as u64)?;
                if let Some(w2) = fc2_w {
                    let out_b = self.ctx.alloc_buffer((n_out * d_proj * 4) as u64)?;
                    let tot_h1 = (n_out * fc1_w.rows) as u32;
                    self.ctx.run(|rec| {
                        self.record_linear(rec, fc1_w, &in_b, &h1_b, n_out, false, q8_ref)?;
                        if let Some(b) = fc1_b {
                            self.record_bias(rec, &h1_b, b, n_out, fc1_w.rows)?;
                        }
                        rec.dispatch(
                            &self.pipes.act_inplace,
                            &[&h1_b, &self.dummy_buf, &self.dummy_buf, &self.dummy_buf],
                            &[tot_h1, *act as u32, 0, 0],
                            tot_h1.div_ceil(256),
                        )?;
                        self.record_linear(rec, w2, &h1_b, &out_b, n_out, false, q8_ref)?;
                        if let Some(b) = fc2_b {
                            self.record_bias(rec, &out_b, b, n_out, d_proj)?;
                        }
                        Ok(())
                    })?;
                    Ok((self.ctx.read_f32(&out_b, n_out * d_proj)?, out_gh, out_gw))
                } else {
                    self.ctx.run(|rec| {
                        self.record_linear(rec, fc1_w, &in_b, &h1_b, n_out, false, q8_ref)?;
                        if let Some(b) = fc1_b {
                            self.record_bias(rec, &h1_b, b, n_out, fc1_w.rows)?;
                        }
                        Ok(())
                    })?;
                    Ok((
                        self.ctx.read_f32(&h1_b, n_out * fc1_w.rows)?,
                        out_gh,
                        out_gw,
                    ))
                }
            }
        }
    }
}

fn bytemuck_f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

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
