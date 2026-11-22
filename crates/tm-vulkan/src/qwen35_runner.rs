//! End-to-end Vulkan GPU inference runner for `qwen35` hybrid Gated DeltaNet (GDN) + Gated
//! Full-Attention GGUF models.
//!
//! Strictly enforces `AGENTS.md` §5 & §9 and `GDN_CONTRACT`:
//! 1. Causal `conv1d` over `qkv` followed by **SiLU BEFORE splitting** `q`, `k`, `v`.
//! 2. Per-head L2-normalization on `q` and `k` ONLY (**never** normalize `v`).
//! 3. Tiled head grouping `kh = vh % n_k_heads` (**never** contiguous repeat).
//! 4. Query scaled by `1 / sqrt(s_dim)` on the updated recurrent state.
//! 5. Per-head gated RMSNorm with **SiLU** on `z` (`rms_norm(o_h, ssm_norm) * silu(z_h)`),
//!    while full-attention layers use `sigmoid` output gating on de-interleaved `[Q_h, Gate_h]`.
//! - **Slabbed Device Arena**: Upfront planning + `OutOfDeviceMemory` check before mutating state.
//! - **Pre-recorded Decode Command Buffers**: Zero heap allocations and single `vkQueueSubmit` per token.
//! - **V3 Cooperative Matrix Prefill + Portable Fallback**: `16×16×16` CoopMat GEMM + fused SwiGLU +
//!   `fa_coopmat` (`head_dim <= 256`), with automatic subgroup/LDS fallback on non-CoopMat GPUs.

use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::context::{Pipeline, VkBuffer, VkPrefillMode, VulkanContext};
use crate::kernels::{dtype_code, normalize_vk_mat_bytes};
use crate::llama_runner::{VkProfileReport, VkStageStat};
use crate::lora_gpu::{VkEmbedSlot, VkLinearSlot, VkLoraEngine};
use crate::{VkResultExt, VulkanError};

const SPV_EMBED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv"));
const SPV_RMS_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv"));
const SPV_QK_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm.spv"));
const SPV_SPLIT_QG_QNORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/split_qg_qnorm.spv"));
const SPV_SIGMOID_MUL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sigmoid_mul.spv"));
const SPV_ROPE_KV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_kv.spv"));
const SPV_GQA_ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gqa_attn.spv"));
const SPV_FA_COOPMAT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_coopmat.spv"));
const SPV_FA_GQA_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_gqa_tiled.spv"));
const SPV_GEMV_RES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_res.spv"));
const SPV_GEMV_PAIR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_pair.spv"));
const SPV_GEMV_QKV_Q4_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_qkv_q4_0.spv"));
const SPV_GEMV_PAIR_SILU_Q4_0: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemv_pair_silu_q4_0.spv"));
const SPV_SILU_MUL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul.spv"));
const SPV_GDN_GATES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_gates.spv"));
const SPV_GDN_CONV_SILU_L2: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_conv_silu_l2.spv"));
const SPV_GDN_RECURRENCE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_recurrence.spv"));
const SPV_GDN_RECURRENCE_LDS: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_recurrence_lds.spv"));
const SPV_GDN_NORM_GATED_SILU: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_norm_gated_silu.spv"));
const SPV_GEMM_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_tiled.spv"));
const SPV_GEMM_KQUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_kquant.spv"));
const SPV_GEMM_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_f16.spv"));
const SPV_GEMM_PAIR_SILU_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_pair_silu_coopmat_f16.spv"));
const SPV_QUANTIZE_Q8_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantize_q8_1.spv"));
const SPV_GEMM_COOPMAT_I8: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_i8.spv"));
const SPV_ARGMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));

const PREFERRED_MAX_BATCH: usize = 512;
const TARGET_SLAB_BYTES: u64 = 512 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedPrefill {
    Fallback,
    CoopMatF16,
    CoopMatI8,
}

impl ResolvedPrefill {
    fn label(self) -> &'static str {
        match self {
            Self::Fallback => "tiled-fallback",
            Self::CoopMatF16 => "coopmat-f16",
            Self::CoopMatI8 => "coopmat-i8",
        }
    }
}

/// Model hyperparameters for [`Qwen35VkRunner`].
#[derive(Debug, Clone)]
pub struct Qwen35VkConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_gdn_layers: usize,
    pub n_full_attn_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
}

impl Qwen35VkConfig {
    #[inline]
    pub const fn conv_dim(&self) -> usize {
        2 * self.ssm_n_group * self.ssm_state_size + self.ssm_d_inner
    }
}

#[derive(Clone, Copy)]
struct VkSlice {
    slab: usize,
    offset: u64,
    len: u64,
}

#[derive(Clone, Copy)]
struct VkMat {
    slice: VkSlice,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

#[derive(Clone, Copy)]
enum GemmFlavor {
    Tiled32,
    KQuantSubgroup,
    CoopMatF16 { bm: u32, bn: u32 },
    CoopMatI8,
}

struct BatchGemmEntry {
    pipe: Arc<Pipeline>,
    set: vk::DescriptorSet,
    mat: VkMat,
    flavor: GemmFlavor,
}

enum BatchFfnPlan {
    FusedPairSiluCoopMat {
        pipe: Arc<Pipeline>,
        set: vk::DescriptorSet,
        cols: usize,
        rows: usize,
        row_bytes: usize,
    },
    Separate {
        gate: BatchGemmEntry,
        up: BatchGemmEntry,
        silu_set: vk::DescriptorSet,
    },
}

#[derive(Clone, Copy)]
enum BatchAttnKind {
    CoopMat64,
    Tiled8,
    Legacy1,
}

struct ArenaPlanner {
    raw_align: u64,
    align: u64,
    slab_cap: u64,
    max_range: u64,
    slab_base: usize,
    imported_limit: Option<u64>,
    imported_bytes: u64,
    slab_sizes: Vec<u64>,
}

impl ArenaPlanner {
    fn new(align: u64, slab_cap: u64, max_range: u64, imported_limit: Option<u64>) -> Self {
        Self {
            raw_align: align.max(1),
            align: align.max(256),
            slab_cap,
            max_range,
            slab_base: usize::from(imported_limit.is_some()),
            imported_limit,
            imported_bytes: 0,
            slab_sizes: Vec::new(),
        }
    }

    fn reserve(&mut self, byte_len: u64) -> Result<VkSlice, VulkanError> {
        let len = byte_len.max(16);
        if len > self.max_range {
            return Err(VulkanError::BufferTooLarge {
                bytes: len,
                limit: self.max_range,
            });
        }
        let aligned = len.div_ceil(self.align) * self.align;
        if let Some(last) = self.slab_sizes.last_mut()
            && *last + aligned <= self.slab_cap
        {
            let offset = *last;
            *last += aligned;
            return Ok(VkSlice {
                slab: self.slab_base + self.slab_sizes.len() - 1,
                offset,
                len,
            });
        }
        let slab = self.slab_base + self.slab_sizes.len();
        self.slab_sizes.push(aligned);
        Ok(VkSlice {
            slab,
            offset: 0,
            len,
        })
    }

    fn try_import_slice(&mut self, file_offset: u64, byte_len: u64) -> Option<VkSlice> {
        let limit = self.imported_limit?;
        if byte_len == 0
            || byte_len > self.max_range
            || file_offset + byte_len > limit
            || !file_offset.is_multiple_of(self.raw_align)
        {
            return None;
        }
        self.imported_bytes += byte_len;
        Some(VkSlice {
            slab: 0,
            offset: file_offset,
            len: byte_len,
        })
    }

    fn total_bytes(&self) -> u64 {
        self.imported_bytes + self.slab_sizes.iter().sum::<u64>()
    }
}

fn gemv_nr(dt: DType) -> u32 {
    match dt {
        DType::Q8_0 | DType::Q6K => 4,
        _ => 2,
    }
}

fn embed_items(dt: DType, n_embd: usize) -> Result<u32, VulkanError> {
    let items = match dt {
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => n_embd / 32,
        DType::Q5K | DType::Q6K => n_embd / 256,
        DType::F32 => n_embd,
        other => return Err(VulkanError::UnsupportedDType(other)),
    };
    Ok(items as u32)
}

type GemvCall = (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]);

enum DecodePairSets {
    Fused {
        pipe: Arc<Pipeline>,
        set: vk::DescriptorSet,
        groups_x: u32,
        push: [u32; 4],
    },
    Separate {
        a: GemvCall,
        b: GemvCall,
    },
}

enum DecodeQkvSets {
    Fused {
        pipe: Arc<Pipeline>,
        set: vk::DescriptorSet,
        groups_x: u32,
        push: [u32; 4],
    },
    Separate {
        wqg: GemvCall,
        wk: GemvCall,
        wv: GemvCall,
    },
}

enum DecodeFfnSets {
    FusedPairSilu {
        pipe: Arc<Pipeline>,
        set: vk::DescriptorSet,
        groups_x: u32,
        push: [u32; 2],
    },
    Separate {
        gate: GemvCall,
        up: GemvCall,
        silu_set: vk::DescriptorSet,
    },
}

enum AttnLayerPlan {
    Gdn {
        qkv_z_dec: DecodePairSets,
        alpha_beta_dec: DecodePairSets,
        wqkv_batch: BatchGemmEntry,
        wz_batch: BatchGemmEntry,
        w_alpha_batch: BatchGemmEntry,
        w_beta_batch: BatchGemmEntry,
        gates_set: vk::DescriptorSet,
        conv_set: vk::DescriptorSet,
        recur_set: vk::DescriptorSet,
        gnorm_set: vk::DescriptorSet,
        w_out_dec: GemvCall,
        w_out_batch: BatchGemmEntry,
        state_slice: VkSlice,
        conv_hist_slice: VkSlice,
        proj_weight_bytes: u64,
        out_weight_bytes: u64,
    },
    FullAttn {
        qkv_dec: DecodeQkvSets,
        wqg_batch: BatchGemmEntry,
        wk_batch: BatchGemmEntry,
        wv_batch: BatchGemmEntry,
        split_qg_set: vk::DescriptorSet,
        k_norm_set: vk::DescriptorSet,
        rope_dec_set: vk::DescriptorSet,
        rope_batch_set: vk::DescriptorSet,
        attn_dec_set: vk::DescriptorSet,
        attn_batch_set: vk::DescriptorSet,
        sigmoid_set: vk::DescriptorSet,
        wo_dec: GemvCall,
        wo_batch: BatchGemmEntry,
        k_cache_slice: VkSlice,
        v_cache_slice: VkSlice,
        qkv_weight_bytes: u64,
        wo_weight_bytes: u64,
    },
}

struct Qwen35LayerPlan {
    idx: usize,
    attn_norm_set: vk::DescriptorSet,
    attn: AttnLayerPlan,
    post_attn_norm_set: vk::DescriptorSet,
    ffn_dec: DecodeFfnSets,
    ffn_dec_sep: (GemvCall, GemvCall, vk::DescriptorSet),
    ffn_batch: BatchFfnPlan,
    ffn_batch_sep: (BatchGemmEntry, BatchGemmEntry, vk::DescriptorSet),
    down_dec: GemvCall,
    down_batch: BatchGemmEntry,
    gate_up_weight_bytes: u64,
    down_weight_bytes: u64,
}

struct RunnerPipes {
    embed_dec: Arc<Pipeline>,
    embed_batch: Arc<Pipeline>,
    rms_norm: Arc<Pipeline>,
    qk_norm: Arc<Pipeline>,
    split_qg_qnorm: Arc<Pipeline>,
    sigmoid_mul: Arc<Pipeline>,
    rope_dec: Arc<Pipeline>,
    rope_batch: Arc<Pipeline>,
    attn_dec: Arc<Pipeline>,
    attn_batch: Arc<Pipeline>,
    attn_batch_kind: BatchAttnKind,
    gdn_gates: Arc<Pipeline>,
    gdn_conv_silu_l2: Arc<Pipeline>,
    gdn_conv_batch: Arc<Pipeline>,
    gdn_conv_state_upd: Arc<Pipeline>,
    gdn_recurrence: Arc<Pipeline>,
    gdn_norm_gated_silu: Arc<Pipeline>,
    silu_mul: Arc<Pipeline>,
    quant_q8_1: Option<Arc<Pipeline>>,
    argmax: Arc<Pipeline>,
}

/// Complete Vulkan GPU inference runner for `qwen35` hybrid GDN + full-attention models.
pub struct Qwen35VkRunner {
    pub cfg: Qwen35VkConfig,
    ctx: VulkanContext,
    prefill_kind: ResolvedPrefill,
    max_batch: usize,
    slabs: Vec<VkBuffer>,
    step_buf: VkBuffer,
    token_ids_buf: VkBuffer,
    argmax_buf: VkBuffer,
    logits_host: VkBuffer,
    logits_dev: VkSlice,
    token_embd: VkMat,
    out_mat: VkMat,
    pipes: RunnerPipes,
    embed_dec_set: vk::DescriptorSet,
    embed_batch_set: vk::DescriptorSet,
    quant_xb_set: Option<vk::DescriptorSet>,
    quant_attn_out_set: Option<vk::DescriptorSet>,
    quant_hb_set: Option<vk::DescriptorSet>,
    output_norm_set: vk::DescriptorSet,
    lm_head_dec: GemvCall,
    argmax_set: vk::DescriptorSet,
    layers: Vec<Qwen35LayerPlan>,
    desc_pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    cb_decode_no_logits: vk::CommandBuffer,
    cb_decode_logits: vk::CommandBuffer,
    cb_decode_greedy: vk::CommandBuffer,
    cb_batch: vk::CommandBuffer,
    fence: vk::Fence,
    lora: VkLoraEngine,
    linear_slots: Vec<VkLinearSlot>,
    embed_slots: Vec<VkEmbedSlot>,
    zero_copy_weights: bool,
    used_lds_recurrence: bool,
    // Kept AFTER `slabs` so the GGUF `mmap` outlives any imported `VK_EXT_external_memory_host` buffer.
    _gguf: Arc<GgufFile>,
}

struct PendingMat<'a> {
    slice: VkSlice,
    bytes: std::borrow::Cow<'a, [u8]>,
    imported: bool,
    mat: VkMat,
}

struct PendingF32 {
    slice: VkSlice,
    vals: Vec<f32>,
}

fn plan_f32_tensor(
    planner: &mut ArenaPlanner,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    expected_dims: &[usize],
) -> Result<PendingF32, VulkanError> {
    let (info, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
    let len: usize = expected_dims.iter().product();
    let mut vals = vec![0.0f32; len];
    dequantize_row_bytes(info.dtype, bytes, &mut vals);
    let slice = planner.reserve((len * 4) as u64)?;
    Ok(PendingF32 { slice, vals })
}

fn plan_mat<'a>(
    planner: &mut ArenaPlanner,
    gguf: &'a GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<PendingMat<'a>, VulkanError> {
    let (info, raw_bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    let (eff_dtype, bytes) = normalize_vk_mat_bytes(info.dtype, raw_bytes, rows, cols)?;
    let _ = dtype_code(eff_dtype)?;
    let row_bytes = eff_dtype.row_bytes(cols);
    let file_off = gguf.tensor_file_offset(info) as u64;
    let can_import = matches!(&bytes, std::borrow::Cow::Borrowed(_));
    let (slice, imported) = if can_import {
        match planner.try_import_slice(file_off, bytes.len() as u64) {
            Some(s) => (s, true),
            None => (planner.reserve(bytes.len() as u64)?, false),
        }
    } else {
        (planner.reserve(bytes.len() as u64)?, false)
    };
    Ok(PendingMat {
        slice,
        bytes,
        imported,
        mat: VkMat {
            slice,
            dtype: eff_dtype,
            rows,
            cols,
            row_bytes,
        },
    })
}

impl Qwen35VkRunner {
    /// Loads a `qwen35` GGUF model onto the selected Vulkan device.
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, VulkanError> {
        Self::from_gguf_with_context(VulkanContext::new()?, gguf, max_ctx)
    }

    /// Loads a `qwen35` GGUF model onto an explicit [`VulkanContext`].
    pub fn from_gguf_with_context(
        ctx: VulkanContext,
        gguf: Arc<GgufFile>,
        max_ctx: usize,
    ) -> Result<Self, VulkanError> {
        if max_ctx == 0 {
            return Err(VulkanError::InvalidDimensions(
                "max_ctx must be >= 1".to_string(),
            ));
        }
        let gguf_ref: &GgufFile = &gguf;
        let arch = gguf_ref.req_str("general.architecture")?.to_string();
        if arch != "qwen35" {
            return Err(VulkanError::UnsupportedArch(arch));
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
        let full_attention_interval = gguf_ref
            .get_meta(&format!("{pfx}.full_attention_interval"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4);
        let ssm_conv_kernel = gguf_ref.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_state_size = gguf_ref.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_group = gguf_ref.req_usize(&format!("{pfx}.ssm.group_count"))?;
        let ssm_dt_rank = gguf_ref.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_d_inner = gguf_ref.req_usize(&format!("{pfx}.ssm.inner_size"))?;

        if head_dim > 256 {
            return Err(VulkanError::InvalidDimensions(format!(
                "qwen35 head_dim {head_dim} > 256 not supported"
            )));
        }
        if !(2..=8).contains(&ssm_conv_kernel) {
            return Err(VulkanError::InvalidDimensions(format!(
                "qwen35 ssm_conv_kernel {ssm_conv_kernel} must be in 2..=8"
            )));
        }
        if ssm_state_size == 0 || ssm_state_size > 128 || !ssm_state_size.is_multiple_of(16) {
            return Err(VulkanError::InvalidDimensions(format!(
                "qwen35 ssm_state_size {ssm_state_size} must be a multiple of 16 in 16..=128"
            )));
        }
        if ssm_d_inner != ssm_dt_rank * ssm_state_size {
            return Err(VulkanError::InvalidDimensions(format!(
                "qwen35 ssm_d_inner {ssm_d_inner} != ssm_dt_rank ({ssm_dt_rank}) * ssm_state_size ({ssm_state_size})"
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

        let info = ctx.info().clone();
        let sg = info.subgroup_size;
        let can_cm_f16 = info.coopmat_f16_f32acc && (sg == 32 || sg == 64);
        let can_cm_i8 = info.coopmat_i8_i32acc && (sg == 32 || sg == 64);
        let prefill_kind = match ctx.prefill_mode() {
            VkPrefillMode::Fallback => ResolvedPrefill::Fallback,
            VkPrefillMode::CoopMatF16 => {
                if !can_cm_f16 {
                    return Err(VulkanError::MissingFeature {
                        device: info.name.clone(),
                        feature: "16x16x16 f16*f16+f32->f32 cooperativeMatrix",
                    });
                }
                ResolvedPrefill::CoopMatF16
            }
            VkPrefillMode::CoopMatI8 => {
                if !can_cm_i8 {
                    return Err(VulkanError::MissingFeature {
                        device: info.name.clone(),
                        feature: "16x16x16 i8*i8+i32->i32 cooperativeMatrix",
                    });
                }
                ResolvedPrefill::CoopMatI8
            }
            VkPrefillMode::Auto => {
                if can_cm_f16 {
                    ResolvedPrefill::CoopMatF16
                } else {
                    ResolvedPrefill::Fallback
                }
            }
        };

        let umap_buf = ctx.try_import_host_readonly(gguf_ref.as_bytes(), gguf_ref.is_mmap());
        let slab_cap = TARGET_SLAB_BYTES
            .min(info.max_alloc_bytes / 2)
            .max(64 << 20);
        let mut planner = ArenaPlanner::new(
            info.min_storage_buffer_offset_alignment,
            slab_cap,
            info.max_storage_buffer_range,
            umap_buf.as_ref().map(VkBuffer::byte_len),
        );

        // 1. Plan every device allocation up front BEFORE mutating state (`AGENTS.md` MUST #4 & #6).
        let p_token_embd = plan_mat(
            &mut planner,
            gguf_ref,
            "token_embd.weight",
            "global",
            n_embd,
            vocab_size,
        )?;
        let p_output = if gguf_ref.tensor_info("output.weight").is_some() {
            Some(plan_mat(
                &mut planner,
                gguf_ref,
                "output.weight",
                "global",
                n_embd,
                vocab_size,
            )?)
        } else {
            None
        };
        let p_output_norm = plan_f32_tensor(
            &mut planner,
            gguf_ref,
            "output_norm.weight",
            "global",
            &[n_embd],
        )?;
        let dummy_rope_freqs = PendingF32 {
            slice: planner.reserve(16)?,
            vals: vec![1.0f32],
        };

        let q_dim = n_head * head_dim;
        let qg_dim = q_dim * 2;
        let kv_dim = n_head_kv * head_dim;
        let conv_dim = 2 * ssm_n_group * ssm_state_size + ssm_d_inner;
        let k_gdn_dim = ssm_n_group * ssm_state_size;
        let v_gdn_dim = ssm_d_inner;
        let hist_len = ssm_conv_kernel - 1;
        let kv_bytes = (max_ctx * kv_dim * 2) as u64;
        let state_bytes = (ssm_dt_rank * ssm_state_size * ssm_state_size * 4) as u64;
        let conv_hist_bytes = (conv_dim * hist_len * 4) as u64;

        #[allow(clippy::large_enum_variant)]
        enum PendingAttn<'a> {
            Gdn {
                wqkv: PendingMat<'a>,
                wz: PendingMat<'a>,
                w_alpha: PendingMat<'a>,
                w_beta: PendingMat<'a>,
                conv1d: PendingF32,
                ssm_a: PendingF32,
                ssm_dt_bias: PendingF32,
                ssm_norm: PendingF32,
                w_out: PendingMat<'a>,
                state_slice: VkSlice,
                conv_hist_slice: VkSlice,
            },
            FullAttn {
                q_norm: PendingF32,
                k_norm: PendingF32,
                wqg: PendingMat<'a>,
                wk: PendingMat<'a>,
                wv: PendingMat<'a>,
                wo: PendingMat<'a>,
                k_cache: VkSlice,
                v_cache: VkSlice,
            },
        }

        struct PendingLayer<'a> {
            attn_norm: PendingF32,
            post_attn_norm: PendingF32,
            attn: PendingAttn<'a>,
            w_gate: PendingMat<'a>,
            w_up: PendingMat<'a>,
            w_down: PendingMat<'a>,
        }

        let mut p_layers = Vec::with_capacity(n_layers);
        let mut n_gdn_layers = 0usize;
        let mut n_full_attn_layers = 0usize;
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm = plan_f32_tensor(
                &mut planner,
                gguf_ref,
                &format!("{ltag}.attn_norm.weight"),
                &ltag,
                &[n_embd],
            )?;
            let post_norm_name = if gguf_ref
                .tensor_info(&format!("{ltag}.post_attention_norm.weight"))
                .is_some()
            {
                format!("{ltag}.post_attention_norm.weight")
            } else {
                format!("{ltag}.ffn_norm.weight")
            };
            let post_attn_norm =
                plan_f32_tensor(&mut planner, gguf_ref, &post_norm_name, &ltag, &[n_embd])?;

            let is_gdn = gguf_ref
                .tensor_info(&format!("{ltag}.attn_qkv.weight"))
                .is_some();
            let attn = if is_gdn {
                n_gdn_layers += 1;
                PendingAttn::Gdn {
                    wqkv: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_qkv.weight"),
                        &ltag,
                        n_embd,
                        conv_dim,
                    )?,
                    wz: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_gate.weight"),
                        &ltag,
                        n_embd,
                        v_gdn_dim,
                    )?,
                    w_alpha: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_alpha.weight"),
                        &ltag,
                        n_embd,
                        ssm_dt_rank,
                    )?,
                    w_beta: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_beta.weight"),
                        &ltag,
                        n_embd,
                        ssm_dt_rank,
                    )?,
                    conv1d: plan_f32_tensor(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_conv1d.weight"),
                        &ltag,
                        &[ssm_conv_kernel, conv_dim],
                    )?,
                    ssm_a: plan_f32_tensor(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_a"),
                        &ltag,
                        &[ssm_dt_rank],
                    )?,
                    ssm_dt_bias: plan_f32_tensor(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_dt.bias"),
                        &ltag,
                        &[ssm_dt_rank],
                    )?,
                    ssm_norm: plan_f32_tensor(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_norm.weight"),
                        &ltag,
                        &[ssm_state_size],
                    )?,
                    w_out: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.ssm_out.weight"),
                        &ltag,
                        v_gdn_dim,
                        n_embd,
                    )?,
                    state_slice: planner.reserve(state_bytes)?,
                    conv_hist_slice: planner.reserve(conv_hist_bytes)?,
                }
            } else {
                n_full_attn_layers += 1;
                PendingAttn::FullAttn {
                    q_norm: plan_f32_tensor(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_q_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    k_norm: plan_f32_tensor(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_k_norm.weight"),
                        &ltag,
                        &[head_dim],
                    )?,
                    wqg: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_q.weight"),
                        &ltag,
                        n_embd,
                        qg_dim,
                    )?,
                    wk: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_k.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?,
                    wv: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_v.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?,
                    wo: plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_output.weight"),
                        &ltag,
                        q_dim,
                        n_embd,
                    )?,
                    k_cache: planner.reserve(kv_bytes)?,
                    v_cache: planner.reserve(kv_bytes)?,
                }
            };

            p_layers.push(PendingLayer {
                attn_norm,
                post_attn_norm,
                attn,
                w_gate: plan_mat(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.ffn_gate.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_up: plan_mat(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.ffn_up.weight"),
                    &ltag,
                    n_embd,
                    n_ff,
                )?,
                w_down: plan_mat(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.ffn_down.weight"),
                    &ltag,
                    n_ff,
                    n_embd,
                )?,
            });
        }

        let budget = if info.unified_memory {
            info.device_local_bytes * 90 / 100
        } else {
            info.device_local_bytes * 97 / 100
        };
        let qkv_cap = qg_dim.max(conv_dim);
        let q_cap = q_dim.max(k_gdn_dim);
        let k_cap = kv_dim.max(k_gdn_dim);
        let v_cap = kv_dim.max(v_gdn_dim);
        let gate_cap = q_dim.max(v_gdn_dim);
        let max_dim = n_ff.max(n_embd).max(qkv_cap);
        let max_i8_cols = n_embd.max(n_ff).max(gate_cap);
        let range_batch_cap =
            ((info.max_storage_buffer_range as usize) / (max_dim * 4 * 2)).max(32);
        let base_bytes = planner.total_bytes();
        let per_tok_scratch = ((2 * n_embd
            + qkv_cap
            + q_cap
            + k_cap
            + v_cap
            + 2 * gate_cap
            + 3 * ssm_dt_rank
            + 2 * n_ff
            + max_i8_cols)
            * 4) as u64;
        // Cap single-submission batch to 64 on non-CoopMat UMA GPUs (Intel i915, Android Mali,
        // RK3588 Panthor, Pi 5 V3D) so per-submit latency stays well below 500..2000 ms kernel
        // GPU TDR watchdogs (`VK_ERROR_DEVICE_LOST`) while saving 8x activation scratch RAM.
        let preferred_batch = if info.unified_memory && prefill_kind == ResolvedPrefill::Fallback {
            64usize
        } else {
            PREFERRED_MAX_BATCH
        };
        let mut max_batch = preferred_batch.min(range_batch_cap);
        if max_batch > 128 && base_bytes + (max_batch as u64) * per_tok_scratch > budget {
            max_batch = 128.min(range_batch_cap);
        }

        let s_x = planner.reserve((max_batch * n_embd * 4) as u64)?;
        let s_xb = planner.reserve((max_batch * n_embd * 4) as u64)?;
        let s_qkv = planner.reserve((max_batch * qkv_cap * 4) as u64)?;
        let s_q = planner.reserve((max_batch * q_cap * 4) as u64)?;
        let s_k = planner.reserve((max_batch * k_cap * 4) as u64)?;
        let s_v = planner.reserve((max_batch * v_cap * 4) as u64)?;
        let s_gate = planner.reserve((max_batch * gate_cap * 4) as u64)?;
        let s_attn_out = planner.reserve((max_batch * gate_cap * 4) as u64)?;
        let s_alpha = planner.reserve((max_batch * ssm_dt_rank * 4) as u64)?;
        let s_beta = planner.reserve((max_batch * ssm_dt_rank * 4) as u64)?;
        let s_g_exp = planner.reserve((max_batch * ssm_dt_rank * 4) as u64)?;
        let s_hb = planner.reserve((max_batch * n_ff * 4) as u64)?;
        let s_hb2 = planner.reserve((max_batch * n_ff * 4) as u64)?;
        let (s_xq, s_xd) = if prefill_kind == ResolvedPrefill::CoopMatI8 {
            let max_blocks = (max_batch * max_i8_cols) / 32;
            (
                Some(planner.reserve((max_blocks * 32) as u64)?),
                Some(planner.reserve((max_blocks * 8) as u64)?),
            )
        } else {
            (None, None)
        };
        let logits_dev = planner.reserve((vocab_size * 4) as u64)?;

        let total_needed = planner.total_bytes();
        if total_needed > budget {
            return Err(VulkanError::OutOfDeviceMemory {
                needed_mib: total_needed as f64 / (1024.0 * 1024.0),
                budget_mib: budget as f64 / (1024.0 * 1024.0),
                max_ctx,
                device: info.name.clone(),
            });
        }

        // 2. Allocate mutable slabs, zero-initialize state, and upload any non-imported weights.
        let zero_copy_weights = umap_buf.is_some() && planner.imported_bytes > 0;
        let mut slabs = Vec::with_capacity(planner.slab_base + planner.slab_sizes.len());
        if let Some(ubuf) = umap_buf {
            slabs.push(ubuf);
        }
        for &sz in &planner.slab_sizes {
            slabs.push(ctx.create_device_slab(sz)?);
        }
        if !planner.slab_sizes.is_empty() {
            ctx.run(|rec| {
                for slab in &slabs[planner.slab_base..] {
                    // SAFETY: `slab` is a newly created mutable transfer-dst buffer idle on the device.
                    unsafe {
                        ctx.inner
                            .device
                            .cmd_fill_buffer(rec.cb, slab.buf, 0, vk::WHOLE_SIZE, 0);
                    }
                }
                Ok(())
            })?;
        }

        let write_f32 = |pf: &PendingF32| -> Result<(), VulkanError> {
            // SAFETY: `f32` slice viewed as plain bytes for staging copy.
            let bytes = unsafe {
                core::slice::from_raw_parts(pf.vals.as_ptr() as *const u8, pf.vals.len() * 4)
            };
            ctx.write_bytes(&slabs[pf.slice.slab], pf.slice.offset, bytes)
        };
        let write_mat = |pm: &PendingMat<'_>| -> Result<(), VulkanError> {
            if pm.imported {
                return Ok(());
            }
            ctx.write_bytes(&slabs[pm.slice.slab], pm.slice.offset, &pm.bytes)
        };

        write_mat(&p_token_embd)?;
        if let Some(ref po) = p_output {
            write_mat(po)?;
        }
        write_f32(&p_output_norm)?;
        write_f32(&dummy_rope_freqs)?;
        for pl in &p_layers {
            write_f32(&pl.attn_norm)?;
            write_f32(&pl.post_attn_norm)?;
            match &pl.attn {
                PendingAttn::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                    ..
                } => {
                    write_mat(wqkv)?;
                    write_mat(wz)?;
                    write_mat(w_alpha)?;
                    write_mat(w_beta)?;
                    write_f32(conv1d)?;
                    write_f32(ssm_a)?;
                    write_f32(ssm_dt_bias)?;
                    write_f32(ssm_norm)?;
                    write_mat(w_out)?;
                }
                PendingAttn::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                    ..
                } => {
                    write_f32(q_norm)?;
                    write_f32(k_norm)?;
                    write_mat(wqg)?;
                    write_mat(wk)?;
                    write_mat(wv)?;
                    write_mat(wo)?;
                }
            }
            write_mat(&pl.w_gate)?;
            write_mat(&pl.w_up)?;
            write_mat(&pl.w_down)?;
        }

        let step_buf = ctx.create_host_buffer(32)?;
        let token_ids_buf = ctx.create_host_buffer((max_batch * 4) as u64)?;
        let argmax_buf = ctx.create_host_buffer(16)?;
        let logits_host = ctx.create_host_buffer((vocab_size * 4) as u64)?;

        // 3. Build compute pipelines.
        let emb_dt = dtype_code(p_token_embd.mat.dtype)?;
        let attn_wg = 8 * sg;
        let attn_n_vec = ((head_dim / 4).div_ceil(sg as usize)).max(1) as u32;
        let (attn_batch, attn_batch_kind) = if prefill_kind != ResolvedPrefill::Fallback
            && info.coopmat_f16_f32acc
            && sg == 32
            && head_dim.is_multiple_of(16)
            && head_dim <= 256
        {
            (
                ctx.pipeline("fa_coopmat", 128, SPV_FA_COOPMAT, 4, 20, &[128])?,
                BatchAttnKind::CoopMat64,
            )
        } else if head_dim.is_multiple_of(4) && head_dim <= 256 {
            (
                ctx.pipeline(
                    "fa_gqa_tiled",
                    attn_wg | (attn_n_vec << 16),
                    SPV_FA_GQA_TILED,
                    4,
                    20,
                    &[attn_wg, attn_n_vec],
                )?,
                BatchAttnKind::Tiled8,
            )
        } else {
            (
                ctx.pipeline(
                    "attn_batch",
                    attn_wg | 0x1000 | (attn_n_vec << 16),
                    SPV_GQA_ATTN,
                    5,
                    16,
                    &[attn_wg, 1, attn_n_vec],
                )?,
                BatchAttnKind::Legacy1,
            )
        };

        let quant_q8_1 = if prefill_kind == ResolvedPrefill::CoopMatI8 {
            Some(ctx.pipeline("quantize_q8_1", 256, SPV_QUANTIZE_Q8_1, 3, 4, &[256])?)
        } else {
            None
        };
        let used_lds_recurrence = !info.subgroup_shuffle || ctx.force_lds_recurrence();
        let gdn_recurrence_pipe = if used_lds_recurrence {
            ctx.pipeline(
                "gdn_recurrence_lds",
                128,
                SPV_GDN_RECURRENCE_LDS,
                7,
                16,
                &[128],
            )?
        } else {
            ctx.pipeline("gdn_recurrence", 128, SPV_GDN_RECURRENCE, 7, 16, &[128])?
        };

        let pipes = RunnerPipes {
            embed_dec: ctx.pipeline("embed_dec", emb_dt, SPV_EMBED, 3, 8, &[64, emb_dt, 0])?,
            embed_batch: ctx.pipeline(
                "embed_batch",
                emb_dt | 0x100,
                SPV_EMBED,
                3,
                8,
                &[64, emb_dt, 1],
            )?,
            rms_norm: ctx.pipeline("rms_norm", 0, SPV_RMS_NORM, 3, 12, &[256])?,
            qk_norm: ctx.pipeline("qk_norm", 0, SPV_QK_NORM, 2, 8, &[64])?,
            split_qg_qnorm: ctx.pipeline("split_qg_qnorm", 64, SPV_SPLIT_QG_QNORM, 4, 12, &[64])?,
            sigmoid_mul: ctx.pipeline("sigmoid_mul", 256, SPV_SIGMOID_MUL, 2, 4, &[256])?,
            rope_dec: ctx.pipeline("rope_dec", 0, SPV_ROPE_KV, 7, 32, &[64, 0])?,
            rope_batch: ctx.pipeline("rope_batch", 1, SPV_ROPE_KV, 7, 32, &[64, 1])?,
            attn_dec: ctx.pipeline(
                "attn_dec",
                attn_wg | (attn_n_vec << 16),
                SPV_GQA_ATTN,
                5,
                16,
                &[attn_wg, 0, attn_n_vec],
            )?,
            attn_batch,
            attn_batch_kind,
            gdn_gates: ctx.pipeline("gdn_gates", 64, SPV_GDN_GATES, 5, 8, &[64])?,
            gdn_conv_silu_l2: ctx.pipeline(
                "gdn_conv_silu_l2",
                128,
                SPV_GDN_CONV_SILU_L2,
                6,
                24,
                &[128, 0],
            )?,
            gdn_conv_batch: ctx.pipeline(
                "gdn_conv_batch",
                128 | 0x100,
                SPV_GDN_CONV_SILU_L2,
                6,
                24,
                &[128, 1],
            )?,
            gdn_conv_state_upd: ctx.pipeline(
                "gdn_conv_state_upd",
                128 | 0x200,
                SPV_GDN_CONV_SILU_L2,
                6,
                24,
                &[128, 2],
            )?,
            gdn_recurrence: gdn_recurrence_pipe,
            gdn_norm_gated_silu: ctx.pipeline(
                "gdn_norm_gated_silu",
                64,
                SPV_GDN_NORM_GATED_SILU,
                3,
                12,
                &[64],
            )?,
            silu_mul: ctx.pipeline("silu_mul", 0, SPV_SILU_MUL, 2, 4, &[256])?,
            quant_q8_1,
            argmax: ctx.pipeline("argmax", 0, SPV_ARGMAX, 2, 4, &[256])?,
        };

        let gemv_pipe_for = |mat: &VkMat| -> Result<(Arc<Pipeline>, u32, [u32; 4]), VulkanError> {
            let dt = dtype_code(mat.dtype)?;
            let nr = gemv_nr(mat.dtype);
            let wg = 128u32;
            let key = dt | (nr << 8) | (wg << 16);
            let p = ctx.pipeline("gemv_res", key, SPV_GEMV_RES, 3, 16, &[wg, dt, nr])?;
            let rows_per_wg = (wg / sg).max(1) * nr;
            let groups = (mat.rows as u32).div_ceil(rows_per_wg);
            let push = [
                mat.cols as u32,
                mat.rows as u32,
                0,
                (mat.row_bytes / 2) as u32,
            ];
            Ok((p, groups, push))
        };

        // 4. Allocate static descriptor pool, command pool, pre-bound descriptor sets.
        let max_sets = (n_layers as u32) * 40 + 32;
        let max_desc = max_sets * 8;
        let d = &ctx.inner.device;
        // SAFETY: valid device handle; pools and fence destroyed in `Drop` after `device_wait_idle`.
        let (desc_pool, cmd_pool, cbs, fence) = unsafe {
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: max_desc,
            }];
            let dp = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .ctx("vkCreateDescriptorPool")?;
            let cp = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(ctx.inner.queue_family),
                    None,
                )
                .ctx("vkCreateCommandPool")?;
            let cbs = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cp)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(4),
                )
                .ctx("vkAllocateCommandBuffers")?;
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .ctx("vkCreateFence")?;
            (dp, cp, cbs, fence)
        };

        enum Binding<'b> {
            Arena(VkSlice),
            Raw(&'b VkBuffer),
        }

        let alloc_set =
            |pipe: &Pipeline, binds: &[Binding<'_>]| -> Result<vk::DescriptorSet, VulkanError> {
                let infos: Vec<vk::DescriptorBufferInfo> = binds
                    .iter()
                    .map(|b| match *b {
                        Binding::Arena(s) => vk::DescriptorBufferInfo {
                            buffer: slabs[s.slab].buf,
                            offset: s.offset,
                            range: s.len,
                        },
                        Binding::Raw(buf) => vk::DescriptorBufferInfo {
                            buffer: buf.buf,
                            offset: 0,
                            range: buf.byte_len(),
                        },
                    })
                    .collect();
                // SAFETY: `desc_pool`, `pipe.set_layout`, and buffer handles outlive the runner.
                unsafe {
                    let layouts = [pipe.set_layout];
                    let set = d
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(desc_pool)
                                .set_layouts(&layouts),
                        )
                        .ctx("vkAllocateDescriptorSets")?[0];
                    let writes: Vec<vk::WriteDescriptorSet> = infos
                        .iter()
                        .enumerate()
                        .map(|(i, info)| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(i as u32)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(core::slice::from_ref(info))
                        })
                        .collect();
                    d.update_descriptor_sets(&writes, &[]);
                    Ok(set)
                }
            };

        let make_pair_dec = |ma: VkMat,
                             mb: VkMat,
                             x_slice: VkSlice,
                             ya_slice: VkSlice,
                             yb_slice: VkSlice|
         -> Result<DecodePairSets, VulkanError> {
            if ma.dtype == mb.dtype && matches!(ma.dtype, DType::Q4_0 | DType::Q8_0) {
                let dt = dtype_code(ma.dtype)?;
                let nr = if ma.dtype == DType::Q4_0 { 2u32 } else { 1u32 };
                if ma.rows.is_multiple_of(nr as usize) {
                    let wg = 128u32;
                    let key = dt | (nr << 8) | (wg << 16);
                    let pipe =
                        ctx.pipeline("gemv_pair", key, SPV_GEMV_PAIR, 5, 16, &[wg, dt, nr])?;
                    let set = alloc_set(
                        &pipe,
                        &[
                            Binding::Arena(ma.slice),
                            Binding::Arena(mb.slice),
                            Binding::Arena(x_slice),
                            Binding::Arena(ya_slice),
                            Binding::Arena(yb_slice),
                        ],
                    )?;
                    let rows_per_wg = (wg / sg).max(1) * nr;
                    let total_rows = (ma.rows + mb.rows) as u32;
                    return Ok(DecodePairSets::Fused {
                        pipe,
                        set,
                        groups_x: total_rows.div_ceil(rows_per_wg),
                        push: [
                            ma.cols as u32,
                            ma.rows as u32,
                            mb.rows as u32,
                            (ma.row_bytes / 2) as u32,
                        ],
                    });
                }
            }
            let (pa, ga, push_a) = gemv_pipe_for(&ma)?;
            let sa = alloc_set(
                &pa,
                &[
                    Binding::Arena(ma.slice),
                    Binding::Arena(x_slice),
                    Binding::Arena(ya_slice),
                ],
            )?;
            let (pb, gb, push_b) = gemv_pipe_for(&mb)?;
            let sb = alloc_set(
                &pb,
                &[
                    Binding::Arena(mb.slice),
                    Binding::Arena(x_slice),
                    Binding::Arena(yb_slice),
                ],
            )?;
            Ok(DecodePairSets::Separate {
                a: (pa, sa, ga, push_a),
                b: (pb, sb, gb, push_b),
            })
        };

        let make_batch_gemm = |mat: VkMat,
                               x_slice: VkSlice,
                               y_slice: VkSlice|
         -> Result<BatchGemmEntry, VulkanError> {
            let dt = dtype_code(mat.dtype)?;
            let cm_wg = 4 * sg;
            if prefill_kind == ResolvedPrefill::CoopMatI8
                && matches!(mat.dtype, DType::Q4_0 | DType::Q4_1 | DType::Q8_0)
                && let (Some(xq), Some(xd)) = (s_xq, s_xd)
            {
                let pipe = ctx.pipeline(
                    "gemm_coopmat_i8",
                    dt | (cm_wg << 8),
                    SPV_GEMM_COOPMAT_I8,
                    4,
                    20,
                    &[cm_wg, dt],
                )?;
                let set = alloc_set(
                    &pipe,
                    &[
                        Binding::Arena(mat.slice),
                        Binding::Arena(xq),
                        Binding::Arena(xd),
                        Binding::Arena(y_slice),
                    ],
                )?;
                return Ok(BatchGemmEntry {
                    pipe,
                    set,
                    mat,
                    flavor: GemmFlavor::CoopMatI8,
                });
            }

            if prefill_kind != ResolvedPrefill::Fallback && can_cm_f16 {
                let (wg_cm, bm, bn) =
                    if mat.rows >= 512 && (8 * sg) <= info.max_workgroup_invocations {
                        (8 * sg, 128u32, 128u32)
                    } else {
                        (4 * sg, 64u32, 64u32)
                    };
                let key = dt | (bn << 8) | (bm << 16) | (wg_cm << 24);
                let pipe = ctx.pipeline(
                    "gemm_coopmat_f16",
                    key,
                    SPV_GEMM_COOPMAT_F16,
                    3,
                    20,
                    &[wg_cm, dt, bm, bn],
                )?;
                let set = alloc_set(
                    &pipe,
                    &[
                        Binding::Arena(mat.slice),
                        Binding::Arena(x_slice),
                        Binding::Arena(y_slice),
                    ],
                )?;
                return Ok(BatchGemmEntry {
                    pipe,
                    set,
                    mat,
                    flavor: GemmFlavor::CoopMatF16 { bm, bn },
                });
            }

            match mat.dtype {
                DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => {
                    let pipe = ctx.pipeline("gemm_tiled", dt, SPV_GEMM_TILED, 3, 20, &[128, dt])?;
                    let set = alloc_set(
                        &pipe,
                        &[
                            Binding::Arena(mat.slice),
                            Binding::Arena(x_slice),
                            Binding::Arena(y_slice),
                        ],
                    )?;
                    Ok(BatchGemmEntry {
                        pipe,
                        set,
                        mat,
                        flavor: GemmFlavor::Tiled32,
                    })
                }
                DType::Q5K | DType::Q6K => {
                    let pipe =
                        ctx.pipeline("gemm_kquant", dt, SPV_GEMM_KQUANT, 3, 20, &[128, dt])?;
                    let set = alloc_set(
                        &pipe,
                        &[
                            Binding::Arena(mat.slice),
                            Binding::Arena(x_slice),
                            Binding::Arena(y_slice),
                        ],
                    )?;
                    Ok(BatchGemmEntry {
                        pipe,
                        set,
                        mat,
                        flavor: GemmFlavor::KQuantSubgroup,
                    })
                }
                other => Err(VulkanError::UnsupportedDType(other)),
            }
        };

        let (quant_xb_set, quant_attn_out_set, quant_hb_set) =
            if let (Some(qp), Some(xq), Some(xd)) = (&pipes.quant_q8_1, s_xq, s_xd) {
                (
                    Some(alloc_set(
                        qp,
                        &[Binding::Arena(s_xb), Binding::Arena(xq), Binding::Arena(xd)],
                    )?),
                    Some(alloc_set(
                        qp,
                        &[
                            Binding::Arena(s_attn_out),
                            Binding::Arena(xq),
                            Binding::Arena(xd),
                        ],
                    )?),
                    Some(alloc_set(
                        qp,
                        &[Binding::Arena(s_hb), Binding::Arena(xq), Binding::Arena(xd)],
                    )?),
                )
            } else {
                (None, None, None)
            };

        let token_embd = p_token_embd.mat;
        let out_mat = p_output.as_ref().map(|p| p.mat).unwrap_or(token_embd);

        let embed_dec_set = alloc_set(
            &pipes.embed_dec,
            &[
                Binding::Arena(token_embd.slice),
                Binding::Raw(&step_buf),
                Binding::Arena(s_x),
            ],
        )?;
        let embed_batch_set = alloc_set(
            &pipes.embed_batch,
            &[
                Binding::Arena(token_embd.slice),
                Binding::Raw(&token_ids_buf),
                Binding::Arena(s_x),
            ],
        )?;
        let output_norm_set = alloc_set(
            &pipes.rms_norm,
            &[
                Binding::Arena(s_x),
                Binding::Arena(p_output_norm.slice),
                Binding::Arena(s_xb),
            ],
        )?;
        let (lm_pipe, lm_groups, lm_push) = gemv_pipe_for(&out_mat)?;
        let lm_set = alloc_set(
            &lm_pipe,
            &[
                Binding::Arena(out_mat.slice),
                Binding::Arena(s_xb),
                Binding::Arena(logits_dev),
            ],
        )?;
        let lm_head_dec = (lm_pipe, lm_set, lm_groups, lm_push);
        let argmax_set = alloc_set(
            &pipes.argmax,
            &[Binding::Arena(logits_dev), Binding::Raw(&argmax_buf)],
        )?;

        let mk_lin_slot = |name: String, x_s: VkSlice, y_s: VkSlice| VkLinearSlot {
            name,
            x_buf: slabs[x_s.slab].buf,
            x_off: x_s.offset,
            x_len: x_s.len,
            y_buf: slabs[y_s.slab].buf,
            y_off: y_s.offset,
            y_len: y_s.len,
        };
        let embed_slots = vec![VkEmbedSlot {
            name: "token_embd.weight".to_string(),
            dec_tok_buf: step_buf.buf,
            dec_tok_len: step_buf.byte_len(),
            batch_tok_buf: token_ids_buf.buf,
            batch_tok_len: token_ids_buf.byte_len(),
            x_buf: slabs[s_x.slab].buf,
            x_off: s_x.offset,
            x_len: s_x.len,
        }];
        let mut linear_slots = Vec::with_capacity(n_layers * 8 + 1);
        linear_slots.push(mk_lin_slot("output.weight".to_string(), s_xb, logits_dev));

        let mut layers = Vec::with_capacity(n_layers);
        for (il, pl) in p_layers.iter().enumerate() {
            let ltag = format!("blk.{il}");
            match &pl.attn {
                PendingAttn::Gdn { .. } => {
                    linear_slots.push(mk_lin_slot(format!("{ltag}.attn_qkv.weight"), s_xb, s_qkv));
                    linear_slots.push(mk_lin_slot(
                        format!("{ltag}.attn_gate.weight"),
                        s_xb,
                        s_gate,
                    ));
                    linear_slots.push(mk_lin_slot(
                        format!("{ltag}.ssm_alpha.weight"),
                        s_xb,
                        s_alpha,
                    ));
                    linear_slots.push(mk_lin_slot(format!("{ltag}.ssm_beta.weight"), s_xb, s_beta));
                    linear_slots.push(mk_lin_slot(
                        format!("{ltag}.ssm_out.weight"),
                        s_attn_out,
                        s_x,
                    ));
                }
                PendingAttn::FullAttn { .. } => {
                    linear_slots.push(mk_lin_slot(format!("{ltag}.attn_q.weight"), s_xb, s_qkv));
                    linear_slots.push(mk_lin_slot(format!("{ltag}.attn_k.weight"), s_xb, s_k));
                    linear_slots.push(mk_lin_slot(format!("{ltag}.attn_v.weight"), s_xb, s_v));
                    linear_slots.push(mk_lin_slot(
                        format!("{ltag}.attn_output.weight"),
                        s_attn_out,
                        s_x,
                    ));
                }
            }
            linear_slots.push(mk_lin_slot(format!("{ltag}.ffn_gate.weight"), s_xb, s_hb));
            linear_slots.push(mk_lin_slot(format!("{ltag}.ffn_up.weight"), s_xb, s_hb2));
            linear_slots.push(mk_lin_slot(format!("{ltag}.ffn_down.weight"), s_hb, s_x));
            let attn_norm_set = alloc_set(
                &pipes.rms_norm,
                &[
                    Binding::Arena(s_x),
                    Binding::Arena(pl.attn_norm.slice),
                    Binding::Arena(s_xb),
                ],
            )?;

            let attn_plan = match &pl.attn {
                PendingAttn::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                    state_slice,
                    conv_hist_slice,
                } => {
                    let qkv_z_dec = make_pair_dec(wqkv.mat, wz.mat, s_xb, s_qkv, s_gate)?;
                    let alpha_beta_dec =
                        make_pair_dec(w_alpha.mat, w_beta.mat, s_xb, s_alpha, s_beta)?;
                    let wqkv_batch = make_batch_gemm(wqkv.mat, s_xb, s_qkv)?;
                    let wz_batch = make_batch_gemm(wz.mat, s_xb, s_gate)?;
                    let w_alpha_batch = make_batch_gemm(w_alpha.mat, s_xb, s_alpha)?;
                    let w_beta_batch = make_batch_gemm(w_beta.mat, s_xb, s_beta)?;

                    let gates_set = alloc_set(
                        &pipes.gdn_gates,
                        &[
                            Binding::Arena(s_alpha),
                            Binding::Arena(s_beta),
                            Binding::Arena(ssm_dt_bias.slice),
                            Binding::Arena(ssm_a.slice),
                            Binding::Arena(s_g_exp),
                        ],
                    )?;
                    let conv_set = alloc_set(
                        &pipes.gdn_conv_silu_l2,
                        &[
                            Binding::Arena(s_qkv),
                            Binding::Arena(conv1d.slice),
                            Binding::Arena(*conv_hist_slice),
                            Binding::Arena(s_q),
                            Binding::Arena(s_k),
                            Binding::Arena(s_v),
                        ],
                    )?;
                    let recur_set = alloc_set(
                        &pipes.gdn_recurrence,
                        &[
                            Binding::Arena(s_q),
                            Binding::Arena(s_k),
                            Binding::Arena(s_v),
                            Binding::Arena(s_g_exp),
                            Binding::Arena(s_beta),
                            Binding::Arena(*state_slice),
                            Binding::Arena(s_attn_out),
                        ],
                    )?;
                    let gnorm_set = alloc_set(
                        &pipes.gdn_norm_gated_silu,
                        &[
                            Binding::Arena(s_attn_out),
                            Binding::Arena(ssm_norm.slice),
                            Binding::Arena(s_gate),
                        ],
                    )?;

                    let (p_out, g_out, mut push_out) = gemv_pipe_for(&w_out.mat)?;
                    push_out[2] = 1; // fused residual add into `s_x`
                    let s_out = alloc_set(
                        &p_out,
                        &[
                            Binding::Arena(w_out.slice),
                            Binding::Arena(s_attn_out),
                            Binding::Arena(s_x),
                        ],
                    )?;
                    let w_out_batch = make_batch_gemm(w_out.mat, s_attn_out, s_x)?;

                    AttnLayerPlan::Gdn {
                        qkv_z_dec,
                        alpha_beta_dec,
                        wqkv_batch,
                        wz_batch,
                        w_alpha_batch,
                        w_beta_batch,
                        gates_set,
                        conv_set,
                        recur_set,
                        gnorm_set,
                        w_out_dec: (p_out, s_out, g_out, push_out),
                        w_out_batch,
                        state_slice: *state_slice,
                        conv_hist_slice: *conv_hist_slice,
                        proj_weight_bytes: wqkv.slice.len
                            + wz.slice.len
                            + w_alpha.slice.len
                            + w_beta.slice.len,
                        out_weight_bytes: w_out.slice.len,
                    }
                }
                PendingAttn::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                    k_cache,
                    v_cache,
                } => {
                    let qkv_dec = if wqg.mat.dtype == DType::Q4_0
                        && wk.mat.dtype == DType::Q4_0
                        && wv.mat.dtype == DType::Q4_0
                        && wqg.mat.rows.is_multiple_of(2)
                        && wk.mat.rows.is_multiple_of(2)
                    {
                        let wg = 128u32;
                        let nr = 2u32;
                        let pipe =
                            ctx.pipeline("gemv_qkv_q4_0", wg, SPV_GEMV_QKV_Q4_0, 7, 16, &[wg, nr])?;
                        let set = alloc_set(
                            &pipe,
                            &[
                                Binding::Arena(wqg.slice),
                                Binding::Arena(wk.slice),
                                Binding::Arena(wv.slice),
                                Binding::Arena(s_xb),
                                Binding::Arena(s_qkv),
                                Binding::Arena(s_k),
                                Binding::Arena(s_v),
                            ],
                        )?;
                        let total_rows = (wqg.mat.rows + wk.mat.rows + wv.mat.rows) as u32;
                        let rows_per_wg = (wg / sg).max(1) * nr;
                        DecodeQkvSets::Fused {
                            pipe,
                            set,
                            groups_x: total_rows.div_ceil(rows_per_wg),
                            push: [
                                wqg.mat.cols as u32,
                                wqg.mat.rows as u32,
                                wk.mat.rows as u32,
                                wv.mat.rows as u32,
                            ],
                        }
                    } else {
                        let (pq, gq, push_q) = gemv_pipe_for(&wqg.mat)?;
                        let sq = alloc_set(
                            &pq,
                            &[
                                Binding::Arena(wqg.slice),
                                Binding::Arena(s_xb),
                                Binding::Arena(s_qkv),
                            ],
                        )?;
                        let (pk, gk, push_k) = gemv_pipe_for(&wk.mat)?;
                        let sk = alloc_set(
                            &pk,
                            &[
                                Binding::Arena(wk.slice),
                                Binding::Arena(s_xb),
                                Binding::Arena(s_k),
                            ],
                        )?;
                        let (pv, gv, push_v) = gemv_pipe_for(&wv.mat)?;
                        let sv = alloc_set(
                            &pv,
                            &[
                                Binding::Arena(wv.slice),
                                Binding::Arena(s_xb),
                                Binding::Arena(s_v),
                            ],
                        )?;
                        DecodeQkvSets::Separate {
                            wqg: (pq, sq, gq, push_q),
                            wk: (pk, sk, gk, push_k),
                            wv: (pv, sv, gv, push_v),
                        }
                    };

                    let wqg_batch = make_batch_gemm(wqg.mat, s_xb, s_qkv)?;
                    let wk_batch = make_batch_gemm(wk.mat, s_xb, s_k)?;
                    let wv_batch = make_batch_gemm(wv.mat, s_xb, s_v)?;

                    let split_qg_set = alloc_set(
                        &pipes.split_qg_qnorm,
                        &[
                            Binding::Arena(s_qkv),
                            Binding::Arena(q_norm.slice),
                            Binding::Arena(s_q),
                            Binding::Arena(s_gate),
                        ],
                    )?;
                    let k_norm_set = alloc_set(
                        &pipes.qk_norm,
                        &[Binding::Arena(s_k), Binding::Arena(k_norm.slice)],
                    )?;

                    let rope_binds = [
                        Binding::Arena(s_q),
                        Binding::Arena(s_k),
                        Binding::Arena(s_v),
                        Binding::Arena(*k_cache),
                        Binding::Arena(*v_cache),
                        Binding::Arena(dummy_rope_freqs.slice),
                        Binding::Raw(&step_buf),
                    ];
                    let rope_dec_set = alloc_set(&pipes.rope_dec, &rope_binds)?;
                    let rope_batch_set = alloc_set(&pipes.rope_batch, &rope_binds)?;

                    let attn_dec_set = alloc_set(
                        &pipes.attn_dec,
                        &[
                            Binding::Arena(s_q),
                            Binding::Arena(*k_cache),
                            Binding::Arena(*v_cache),
                            Binding::Arena(s_attn_out),
                            Binding::Raw(&step_buf),
                        ],
                    )?;
                    let attn_batch_set = match pipes.attn_batch_kind {
                        BatchAttnKind::CoopMat64 | BatchAttnKind::Tiled8 => alloc_set(
                            &pipes.attn_batch,
                            &[
                                Binding::Arena(s_q),
                                Binding::Arena(*k_cache),
                                Binding::Arena(*v_cache),
                                Binding::Arena(s_attn_out),
                            ],
                        )?,
                        BatchAttnKind::Legacy1 => alloc_set(
                            &pipes.attn_batch,
                            &[
                                Binding::Arena(s_q),
                                Binding::Arena(*k_cache),
                                Binding::Arena(*v_cache),
                                Binding::Arena(s_attn_out),
                                Binding::Raw(&step_buf),
                            ],
                        )?,
                    };

                    let sigmoid_set = alloc_set(
                        &pipes.sigmoid_mul,
                        &[Binding::Arena(s_attn_out), Binding::Arena(s_gate)],
                    )?;

                    let (p_wo, g_wo, mut push_wo) = gemv_pipe_for(&wo.mat)?;
                    push_wo[2] = 1;
                    let s_wo = alloc_set(
                        &p_wo,
                        &[
                            Binding::Arena(wo.slice),
                            Binding::Arena(s_attn_out),
                            Binding::Arena(s_x),
                        ],
                    )?;
                    let wo_batch = make_batch_gemm(wo.mat, s_attn_out, s_x)?;

                    AttnLayerPlan::FullAttn {
                        qkv_dec,
                        wqg_batch,
                        wk_batch,
                        wv_batch,
                        split_qg_set,
                        k_norm_set,
                        rope_dec_set,
                        rope_batch_set,
                        attn_dec_set,
                        attn_batch_set,
                        sigmoid_set,
                        wo_dec: (p_wo, s_wo, g_wo, push_wo),
                        wo_batch,
                        k_cache_slice: *k_cache,
                        v_cache_slice: *v_cache,
                        qkv_weight_bytes: wqg.slice.len + wk.slice.len + wv.slice.len,
                        wo_weight_bytes: wo.slice.len,
                    }
                }
            };

            let post_attn_norm_set = alloc_set(
                &pipes.rms_norm,
                &[
                    Binding::Arena(s_x),
                    Binding::Arena(pl.post_attn_norm.slice),
                    Binding::Arena(s_xb),
                ],
            )?;

            let ffn_dec_sep = {
                let (pg, gg, push_g) = gemv_pipe_for(&pl.w_gate.mat)?;
                let sg_set = alloc_set(
                    &pg,
                    &[
                        Binding::Arena(pl.w_gate.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_hb),
                    ],
                )?;
                let (pu, gu, push_u) = gemv_pipe_for(&pl.w_up.mat)?;
                let su_set = alloc_set(
                    &pu,
                    &[
                        Binding::Arena(pl.w_up.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_hb2),
                    ],
                )?;
                let silu_set = alloc_set(
                    &pipes.silu_mul,
                    &[Binding::Arena(s_hb), Binding::Arena(s_hb2)],
                )?;
                ((pg, sg_set, gg, push_g), (pu, su_set, gu, push_u), silu_set)
            };

            let ffn_dec = if pl.w_gate.mat.dtype == DType::Q4_0 && pl.w_up.mat.dtype == DType::Q4_0
            {
                let wg = 128u32;
                let pipe = ctx.pipeline(
                    "gemv_pair_silu_q4_0",
                    wg,
                    SPV_GEMV_PAIR_SILU_Q4_0,
                    4,
                    8,
                    &[wg],
                )?;
                let set = alloc_set(
                    &pipe,
                    &[
                        Binding::Arena(pl.w_gate.slice),
                        Binding::Arena(pl.w_up.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_hb),
                    ],
                )?;
                let rows_per_wg = (wg / sg).max(1);
                DecodeFfnSets::FusedPairSilu {
                    pipe,
                    set,
                    groups_x: (pl.w_gate.mat.rows as u32).div_ceil(rows_per_wg),
                    push: [pl.w_gate.mat.cols as u32, pl.w_gate.mat.rows as u32],
                }
            } else {
                DecodeFfnSets::Separate {
                    gate: ffn_dec_sep.0.clone(),
                    up: ffn_dec_sep.1.clone(),
                    silu_set: ffn_dec_sep.2,
                }
            };

            let ffn_batch_sep = {
                let gate = make_batch_gemm(pl.w_gate.mat, s_xb, s_hb)?;
                let up = make_batch_gemm(pl.w_up.mat, s_xb, s_hb2)?;
                let silu_set = alloc_set(
                    &pipes.silu_mul,
                    &[Binding::Arena(s_hb), Binding::Arena(s_hb2)],
                )?;
                (gate, up, silu_set)
            };

            let ffn_batch = if prefill_kind == ResolvedPrefill::CoopMatF16
                && pl.w_gate.mat.dtype == pl.w_up.mat.dtype
                && matches!(pl.w_gate.mat.dtype, DType::Q4_0 | DType::Q4_1 | DType::Q8_0)
                && (8 * sg) <= info.max_workgroup_invocations
            {
                let dt = dtype_code(pl.w_gate.mat.dtype)?;
                let cm_wg = 8 * sg;
                let pipe = ctx.pipeline(
                    "gemm_pair_silu_coopmat_f16",
                    dt | (cm_wg << 8),
                    SPV_GEMM_PAIR_SILU_COOPMAT_F16,
                    4,
                    16,
                    &[cm_wg, dt],
                )?;
                let set = alloc_set(
                    &pipe,
                    &[
                        Binding::Arena(pl.w_gate.slice),
                        Binding::Arena(pl.w_up.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_hb),
                    ],
                )?;
                BatchFfnPlan::FusedPairSiluCoopMat {
                    pipe,
                    set,
                    cols: pl.w_gate.mat.cols,
                    rows: pl.w_gate.mat.rows,
                    row_bytes: pl.w_gate.mat.row_bytes,
                }
            } else {
                let gate = make_batch_gemm(pl.w_gate.mat, s_xb, s_hb)?;
                let up = make_batch_gemm(pl.w_up.mat, s_xb, s_hb2)?;
                BatchFfnPlan::Separate {
                    gate,
                    up,
                    silu_set: ffn_batch_sep.2,
                }
            };

            let (p_dn, g_dn, mut push_dn) = gemv_pipe_for(&pl.w_down.mat)?;
            push_dn[2] = 1;
            let s_dn = alloc_set(
                &p_dn,
                &[
                    Binding::Arena(pl.w_down.slice),
                    Binding::Arena(s_hb),
                    Binding::Arena(s_x),
                ],
            )?;
            let down_batch = make_batch_gemm(pl.w_down.mat, s_hb, s_x)?;

            layers.push(Qwen35LayerPlan {
                idx: il,
                attn_norm_set,
                attn: attn_plan,
                post_attn_norm_set,
                ffn_dec,
                ffn_dec_sep,
                ffn_batch,
                ffn_batch_sep,
                down_dec: (p_dn, s_dn, g_dn, push_dn),
                down_batch,
                gate_up_weight_bytes: pl.w_gate.slice.len + pl.w_up.slice.len,
                down_weight_bytes: pl.w_down.slice.len,
            });
        }

        let lora = VkLoraEngine::new(&ctx)?;

        let mut runner = Self {
            cfg: Qwen35VkConfig {
                arch,
                n_layers,
                n_gdn_layers,
                n_full_attn_layers,
                n_embd,
                n_ff,
                n_head,
                n_head_kv,
                head_dim,
                rope_dim,
                rope_freq_base,
                rms_eps,
                full_attention_interval,
                ssm_conv_kernel,
                ssm_state_size,
                ssm_n_group,
                ssm_dt_rank,
                ssm_d_inner,
                vocab_size,
                max_ctx,
            },
            ctx,
            prefill_kind,
            max_batch,
            slabs,
            step_buf,
            token_ids_buf,
            argmax_buf,
            logits_host,
            logits_dev,
            token_embd,
            out_mat,
            pipes,
            embed_dec_set,
            embed_batch_set,
            quant_xb_set,
            quant_attn_out_set,
            quant_hb_set,
            output_norm_set,
            lm_head_dec,
            argmax_set,
            layers,
            desc_pool,
            cmd_pool,
            cb_decode_no_logits: cbs[0],
            cb_decode_logits: cbs[1],
            cb_decode_greedy: cbs[2],
            cb_batch: cbs[3],
            fence,
            lora,
            linear_slots,
            embed_slots,
            zero_copy_weights,
            used_lds_recurrence,
            _gguf: gguf.clone(),
        };
        runner.prerecord_decode_command_buffers()?;
        Ok(runner)
    }

    /// Binds dynamic GGUF LoRA adapters onto this Qwen3.5 Vulkan runner and re-records static decode CBs (`Phase T4`).
    pub fn bind_lora(&mut self, loras: &tm_lora::LoraSet) -> Result<(), VulkanError> {
        self.lora.bind(
            &self.ctx,
            &self._gguf,
            loras,
            self.max_batch,
            &self.linear_slots,
            &self.embed_slots,
        )?;
        self.prerecord_decode_command_buffers()
    }

    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    pub fn driver_info(&self) -> &str {
        &self.ctx.info().driver_info
    }

    pub fn active_prefill_backend(&self) -> &'static str {
        self.prefill_kind.label()
    }

    pub fn is_zero_copy(&self) -> bool {
        self.zero_copy_weights
    }

    pub fn uses_lds_recurrence(&self) -> bool {
        self.used_lds_recurrence
    }

    pub fn weights_mode(&self) -> &'static str {
        if self.zero_copy_weights {
            "zero-copy-umap"
        } else if self.ctx.info().unified_memory {
            "uma-mapped"
        } else {
            "vram-slab"
        }
    }

    /// Borrowed view of the most recently computed `[vocab_size]` FP32 logits.
    pub fn logits(&self) -> &[f32] {
        let ptr = self.logits_host.mapped_ptr().expect("logits_host mapped") as *const f32;
        // SAFETY: `logits_host` is persistently mapped (`vocab_size * 4` bytes, 256B aligned)
        // and only read on the host after fence synchronization.
        unsafe { core::slice::from_raw_parts(ptr, self.cfg.vocab_size) }
    }

    /// Zeroes all recurrent GDN states, Conv1d histories, and KV caches on the GPU in-place.
    pub fn reset_state(&mut self) -> Result<(), VulkanError> {
        let d = &self.ctx.inner.device;
        let cb = self.cb_batch;
        // SAFETY: `cb` is idle while holding `&mut self`; all state slices are 256B-aligned.
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .ctx("vkResetCommandBuffer")?;
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .ctx("vkBeginCommandBuffer")?;
            for layer in &self.layers {
                match &layer.attn {
                    AttnLayerPlan::Gdn {
                        state_slice,
                        conv_hist_slice,
                        ..
                    } => {
                        let s_len = (state_slice.len + 3) & !3;
                        let h_len = (conv_hist_slice.len + 3) & !3;
                        d.cmd_fill_buffer(
                            cb,
                            self.slabs[state_slice.slab].buf,
                            state_slice.offset,
                            s_len,
                            0,
                        );
                        d.cmd_fill_buffer(
                            cb,
                            self.slabs[conv_hist_slice.slab].buf,
                            conv_hist_slice.offset,
                            h_len,
                            0,
                        );
                    }
                    AttnLayerPlan::FullAttn {
                        k_cache_slice,
                        v_cache_slice,
                        ..
                    } => {
                        let k_len = (k_cache_slice.len + 3) & !3;
                        let v_len = (v_cache_slice.len + 3) & !3;
                        d.cmd_fill_buffer(
                            cb,
                            self.slabs[k_cache_slice.slab].buf,
                            k_cache_slice.offset,
                            k_len,
                            0,
                        );
                        d.cmd_fill_buffer(
                            cb,
                            self.slabs[v_cache_slice.slab].buf,
                            v_cache_slice.offset,
                            v_len,
                            0,
                        );
                    }
                }
            }
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[bar],
                &[],
                &[],
            );
            d.end_command_buffer(cb).ctx("vkEndCommandBuffer")?;
        }
        self.submit_and_wait(cb)
    }

    fn check_groups(&self, kernel: &'static str, gx: u32, gy: u32) -> Result<(), VulkanError> {
        let lim = self.ctx.info().max_workgroup_count;
        if gx > lim[0] || gy > lim[1] {
            return Err(VulkanError::InvalidDimensions(format!(
                "dispatch ({gx}, {gy}, 1) for '{kernel}' exceeds maxComputeWorkGroupCount {:?}",
                lim
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn cmd_dispatch_set(
        &self,
        cb: vk::CommandBuffer,
        pipe: &Pipeline,
        set: vk::DescriptorSet,
        push_u32: &[u32],
        gx: u32,
        gy: u32,
        barrier_after: bool,
    ) -> Result<(), VulkanError> {
        self.check_groups(pipe.name, gx, gy)?;
        let d = &self.ctx.inner.device;
        let mut push_bytes = [0u8; 32];
        let byte_len = push_u32.len() * 4;
        for (i, &w) in push_u32.iter().enumerate() {
            push_bytes[i * 4..(i + 1) * 4].copy_from_slice(&w.to_ne_bytes());
        }
        // SAFETY: `cb` is in the recording state; `pipe` and `set` outlive execution.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            if byte_len > 0 {
                d.cmd_push_constants(
                    cb,
                    pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push_bytes[..byte_len],
                );
            }
            d.cmd_dispatch(cb, gx, gy, 1);
            if barrier_after {
                Self::cmd_compute_barrier(d, cb);
            }
        }
        Ok(())
    }

    fn cmd_compute_barrier(d: &ash::Device, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: valid command buffer in recording state.
        unsafe {
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    fn cmd_decode_pair(
        &self,
        cb: vk::CommandBuffer,
        pair: &DecodePairSets,
        barrier_after: bool,
    ) -> Result<(), VulkanError> {
        match pair {
            DecodePairSets::Fused {
                pipe,
                set,
                groups_x,
                push,
            } => self.cmd_dispatch_set(cb, pipe, *set, push, *groups_x, 1, barrier_after),
            DecodePairSets::Separate { a, b } => {
                self.cmd_dispatch_set(cb, &a.0, a.1, &a.3, a.2, 1, false)?;
                self.cmd_dispatch_set(cb, &b.0, b.1, &b.3, b.2, 1, barrier_after)
            }
        }
    }

    fn record_decode_into(
        &self,
        cb: vk::CommandBuffer,
        compute_logits: bool,
        greedy_argmax: bool,
    ) -> Result<(), VulkanError> {
        let d = &self.ctx.inner.device;
        // SAFETY: `cb` is idle and owned by `self.cmd_pool`.
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .ctx("vkResetCommandBuffer")?;
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .ctx("vkBeginCommandBuffer")?;
        }

        let emb_items = embed_items(self.token_embd.dtype, self.cfg.n_embd)?;
        self.cmd_dispatch_set(
            cb,
            &self.pipes.embed_dec,
            self.embed_dec_set,
            &[
                self.cfg.n_embd as u32,
                (self.token_embd.row_bytes / 2) as u32,
            ],
            emb_items.div_ceil(64),
            1,
            true,
        )?;
        self.lora
            .cmd_record_embed(&self.ctx, cb, "token_embd.weight", 1, false)?;

        let eps_bits = self.cfg.rms_eps.to_bits();
        let norm_push = [self.cfg.n_embd as u32, 0, eps_bits];
        let split_push = [self.cfg.n_head as u32, self.cfg.head_dim as u32, eps_bits];
        let qk_push = [self.cfg.head_dim as u32, eps_bits];
        let half_rope = self.cfg.rope_dim / 2;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope).max(kv_dim);
        let rope_groups = (rope_total as u32).div_ceil(64);
        let rope_push = [
            self.cfg.n_head as u32,
            self.cfg.n_head_kv as u32,
            self.cfg.head_dim as u32,
            self.cfg.rope_dim as u32,
            0,
            1, // NeoX RoPE on first `rope_dim` elements
            0, // no freq_factors
            self.cfg.rope_freq_base.to_bits(),
        ];
        let attn_push = [
            self.cfg.n_head as u32,
            self.cfg.n_head_kv as u32,
            self.cfg.head_dim as u32,
            0,
        ];

        let hk = self.cfg.ssm_n_group as u32;
        let hv = self.cfg.ssm_dt_rank as u32;
        let s_dim = self.cfg.ssm_state_size as u32;
        let d_conv = self.cfg.ssm_conv_kernel as u32;
        let gates_push = [hv, 1];
        let conv_push = [hk, hv, s_dim, d_conv, 1, eps_bits];
        let recur_push = [hk, hv, s_dim, 1];
        let gnorm_push = [hv, s_dim, eps_bits];

        for layer in &self.layers {
            let ltag = format!("blk.{}", layer.idx);
            let wg_n = format!("{ltag}.ffn_gate.weight");
            let wu_n = format!("{ltag}.ffn_up.weight");
            let wd_n = format!("{ltag}.ffn_down.weight");

            self.cmd_dispatch_set(
                cb,
                &self.pipes.rms_norm,
                layer.attn_norm_set,
                &norm_push,
                1,
                1,
                true,
            )?;

            match &layer.attn {
                AttnLayerPlan::Gdn {
                    qkv_z_dec,
                    alpha_beta_dec,
                    gates_set,
                    conv_set,
                    recur_set,
                    gnorm_set,
                    w_out_dec,
                    ..
                } => {
                    self.cmd_decode_pair(cb, qkv_z_dec, false)?;
                    self.cmd_decode_pair(cb, alpha_beta_dec, true)?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.attn_qkv.weight"),
                        1,
                    )?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.attn_gate.weight"),
                        1,
                    )?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.ssm_alpha.weight"),
                        1,
                    )?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.ssm_beta.weight"),
                        1,
                    )?;

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.gdn_gates,
                        *gates_set,
                        &gates_push,
                        hv.div_ceil(64),
                        1,
                        false,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.gdn_conv_silu_l2,
                        *conv_set,
                        &conv_push,
                        2 * hk + hv,
                        1,
                        true,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.gdn_recurrence,
                        *recur_set,
                        &recur_push,
                        s_dim.div_ceil(32),
                        hv,
                        true,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.gdn_norm_gated_silu,
                        *gnorm_set,
                        &gnorm_push,
                        hv,
                        1,
                        true,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &w_out_dec.0,
                        w_out_dec.1,
                        &w_out_dec.3,
                        w_out_dec.2,
                        1,
                        true,
                    )?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.ssm_out.weight"),
                        1,
                    )?;
                }
                AttnLayerPlan::FullAttn {
                    qkv_dec,
                    split_qg_set,
                    k_norm_set,
                    rope_dec_set,
                    attn_dec_set,
                    sigmoid_set,
                    wo_dec,
                    ..
                } => {
                    match qkv_dec {
                        DecodeQkvSets::Fused {
                            pipe,
                            set,
                            groups_x,
                            push,
                        } => {
                            self.cmd_dispatch_set(cb, pipe, *set, push, *groups_x, 1, true)?;
                        }
                        DecodeQkvSets::Separate { wqg, wk, wv } => {
                            self.cmd_dispatch_set(cb, &wqg.0, wqg.1, &wqg.3, wqg.2, 1, false)?;
                            self.cmd_dispatch_set(cb, &wk.0, wk.1, &wk.3, wk.2, 1, false)?;
                            self.cmd_dispatch_set(cb, &wv.0, wv.1, &wv.3, wv.2, 1, true)?;
                        }
                    }
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.attn_q.weight"),
                        1,
                    )?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.attn_k.weight"),
                        1,
                    )?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.attn_v.weight"),
                        1,
                    )?;

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.split_qg_qnorm,
                        *split_qg_set,
                        &split_push,
                        self.cfg.n_head as u32,
                        1,
                        false,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.qk_norm,
                        *k_norm_set,
                        &qk_push,
                        self.cfg.n_head_kv as u32,
                        1,
                        true,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rope_dec,
                        *rope_dec_set,
                        &rope_push,
                        rope_groups,
                        1,
                        true,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.attn_dec,
                        *attn_dec_set,
                        &attn_push,
                        self.cfg.n_head as u32,
                        1,
                        true,
                    )?;
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.sigmoid_mul,
                        *sigmoid_set,
                        &[q_dim as u32],
                        (q_dim as u32).div_ceil(256),
                        1,
                        true,
                    )?;
                    self.cmd_dispatch_set(cb, &wo_dec.0, wo_dec.1, &wo_dec.3, wo_dec.2, 1, true)?;
                    self.lora.cmd_record_linear(
                        &self.ctx,
                        cb,
                        &format!("{ltag}.attn_output.weight"),
                        1,
                    )?;
                }
            }

            self.cmd_dispatch_set(
                cb,
                &self.pipes.rms_norm,
                layer.post_attn_norm_set,
                &norm_push,
                1,
                1,
                true,
            )?;

            let force_sep = self.lora.has_linear(&wg_n) || self.lora.has_linear(&wu_n);
            if force_sep {
                let (ref gate, ref up, silu_set) = layer.ffn_dec_sep;
                self.cmd_dispatch_set(cb, &gate.0, gate.1, &gate.3, gate.2, 1, false)?;
                self.cmd_dispatch_set(cb, &up.0, up.1, &up.3, up.2, 1, true)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wg_n, 1)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wu_n, 1)?;
                let gx = (self.cfg.n_ff as u32).div_ceil(256).min(65535);
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.silu_mul,
                    silu_set,
                    &[self.cfg.n_ff as u32],
                    gx,
                    1,
                    true,
                )?;
            } else {
                match &layer.ffn_dec {
                    DecodeFfnSets::FusedPairSilu {
                        pipe,
                        set,
                        groups_x,
                        push,
                    } => {
                        self.cmd_dispatch_set(cb, pipe, *set, push, *groups_x, 1, true)?;
                    }
                    DecodeFfnSets::Separate { gate, up, silu_set } => {
                        self.cmd_dispatch_set(cb, &gate.0, gate.1, &gate.3, gate.2, 1, false)?;
                        self.cmd_dispatch_set(cb, &up.0, up.1, &up.3, up.2, 1, true)?;
                        let gx = (self.cfg.n_ff as u32).div_ceil(256).min(65535);
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.silu_mul,
                            *silu_set,
                            &[self.cfg.n_ff as u32],
                            gx,
                            1,
                            true,
                        )?;
                    }
                }
            }

            self.cmd_dispatch_set(
                cb,
                &layer.down_dec.0,
                layer.down_dec.1,
                &layer.down_dec.3,
                layer.down_dec.2,
                1,
                true,
            )?;
            self.lora.cmd_record_linear(&self.ctx, cb, &wd_n, 1)?;
        }

        if compute_logits {
            self.cmd_dispatch_set(
                cb,
                &self.pipes.rms_norm,
                self.output_norm_set,
                &norm_push,
                1,
                1,
                true,
            )?;
            let lm = &self.lm_head_dec;
            if greedy_argmax {
                self.cmd_dispatch_set(cb, &lm.0, lm.1, &lm.3, lm.2, 1, true)?;
                self.lora
                    .cmd_record_linear(&self.ctx, cb, "output.weight", 1)?;
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.argmax,
                    self.argmax_set,
                    &[self.cfg.vocab_size as u32],
                    1,
                    1,
                    false,
                )?;
                let host_bar = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                // SAFETY: valid recording command buffer.
                unsafe {
                    d.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[host_bar],
                        &[],
                        &[],
                    );
                }
            } else {
                self.cmd_dispatch_set(
                    cb,
                    &lm.0,
                    lm.1,
                    &lm.3,
                    lm.2,
                    1,
                    self.lora.has_linear("output.weight"),
                )?;
                self.lora
                    .cmd_record_linear(&self.ctx, cb, "output.weight", 1)?;
                let to_xfer = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                let to_host = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                // SAFETY: valid buffers and ranges.
                unsafe {
                    d.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[to_xfer],
                        &[],
                        &[],
                    );
                    d.cmd_copy_buffer(
                        cb,
                        self.slabs[self.logits_dev.slab].buf,
                        self.logits_host.buf,
                        &[vk::BufferCopy {
                            src_offset: self.logits_dev.offset,
                            dst_offset: 0,
                            size: (self.cfg.vocab_size * 4) as u64,
                        }],
                    );
                    d.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[to_host],
                        &[],
                        &[],
                    );
                }
            }
        }

        // SAFETY: valid command buffer in recording state.
        unsafe { d.end_command_buffer(cb) }.ctx("vkEndCommandBuffer")?;
        Ok(())
    }

    fn prerecord_decode_command_buffers(&mut self) -> Result<(), VulkanError> {
        self.record_decode_into(self.cb_decode_no_logits, false, false)?;
        self.record_decode_into(self.cb_decode_logits, true, false)?;
        self.record_decode_into(self.cb_decode_greedy, true, true)?;
        Ok(())
    }

    fn write_step_params(&self, token_id: u32, pos: usize) {
        let ptr = self.step_buf.mapped_ptr().expect("step_buf mapped") as *mut u32;
        let words = [token_id, pos as u32, (pos + 1) as u32, 0];
        // SAFETY: `step_buf` is host-visible coherent (32 bytes) and idle before submission.
        unsafe {
            core::ptr::copy_nonoverlapping(words.as_ptr(), ptr, 4);
        }
    }

    fn submit_and_wait(&self, cb: vk::CommandBuffer) -> Result<(), VulkanError> {
        let d = &self.ctx.inner.device;
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        // SAFETY: `cb` is executable, `self.fence` is unsignaled, and `&mut self` callers
        // serialize all uses of this runner's command buffers and recurrent state.
        unsafe {
            d.queue_submit(self.ctx.inner.queue, &submit, self.fence)
                .ctx("vkQueueSubmit")?;
            d.wait_for_fences(&[self.fence], true, u64::MAX)
                .ctx("vkWaitForFences")?;
            d.reset_fences(&[self.fence]).ctx("vkResetFences")?;
        }
        Ok(())
    }

    /// Executes a single-token forward pass using the pre-recorded Vulkan command buffer.
    /// Zero heap allocations occur on this path (`AGENTS.md` MUST #4 & #6).
    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        if pos >= self.cfg.max_ctx {
            return Err(VulkanError::InvalidDimensions(format!(
                "pos {pos} >= max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(VulkanError::InvalidDimensions(format!(
                "token_id {token_id} >= vocab_size {}",
                self.cfg.vocab_size
            )));
        }
        self.write_step_params(token_id, pos);
        let cb = if compute_logits {
            self.cb_decode_logits
        } else {
            self.cb_decode_no_logits
        };
        self.submit_and_wait(cb)?;
        Ok(self.logits())
    }

    /// Executes a single-token forward pass + GPU argmax in one pre-recorded command buffer,
    /// reading back only the 4-byte winning token ID.
    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, VulkanError> {
        if pos >= self.cfg.max_ctx {
            return Err(VulkanError::InvalidDimensions(format!(
                "pos {pos} >= max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(VulkanError::InvalidDimensions(format!(
                "token_id {token_id} >= vocab_size {}",
                self.cfg.vocab_size
            )));
        }
        self.write_step_params(token_id, pos);
        self.submit_and_wait(self.cb_decode_greedy)?;
        let ptr = self.argmax_buf.mapped_ptr().expect("argmax_buf mapped") as *const u32;
        // SAFETY: `submit_and_wait` completed and flushed `SHADER_WRITE -> HOST_READ`.
        Ok(unsafe { *ptr })
    }

    fn cmd_quant_if_needed(
        &self,
        cb: vk::CommandBuffer,
        set_opt: Option<vk::DescriptorSet>,
        n_tok: u32,
        cols: usize,
        needs_i8: bool,
    ) -> Result<(), VulkanError> {
        if needs_i8 && let (Some(qp), Some(set)) = (&self.pipes.quant_q8_1, set_opt) {
            let n_blocks = n_tok * (cols as u32) / 32;
            let gx = n_blocks.div_ceil(8);
            self.cmd_dispatch_set(cb, qp, set, &[n_blocks], gx, 1, true)?;
        }
        Ok(())
    }

    fn cmd_gemm(
        &self,
        cb: vk::CommandBuffer,
        entry: &BatchGemmEntry,
        n_tok: u32,
        add_residual: bool,
        barrier_after: bool,
    ) -> Result<(), VulkanError> {
        let sg = self.ctx.info().subgroup_size;
        let mat = entry.mat;
        let (gx, gy) = match entry.flavor {
            GemmFlavor::Tiled32 => ((mat.rows as u32).div_ceil(64), n_tok.div_ceil(32)),
            GemmFlavor::KQuantSubgroup => {
                let rows_per_wg = (128 / sg).max(1);
                ((mat.rows as u32).div_ceil(rows_per_wg), n_tok.div_ceil(8))
            }
            GemmFlavor::CoopMatF16 { bm, bn } => {
                ((mat.rows as u32).div_ceil(bn), n_tok.div_ceil(bm))
            }
            GemmFlavor::CoopMatI8 => ((mat.rows as u32).div_ceil(64), n_tok.div_ceil(64)),
        };
        let push = [
            mat.cols as u32,
            mat.rows as u32,
            n_tok,
            u32::from(add_residual),
            (mat.row_bytes / 2) as u32,
        ];
        self.cmd_dispatch_set(cb, &entry.pipe, entry.set, &push, gx, gy, barrier_after)
    }

    fn cmd_attn_batch(
        &self,
        cb: vk::CommandBuffer,
        set: vk::DescriptorSet,
        chunk_pos: usize,
        n_tok: u32,
    ) -> Result<(), VulkanError> {
        match self.pipes.attn_batch_kind {
            BatchAttnKind::CoopMat64 => {
                let push = [
                    self.cfg.n_head as u32,
                    self.cfg.n_head_kv as u32,
                    self.cfg.head_dim as u32,
                    chunk_pos as u32,
                    n_tok,
                ];
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.attn_batch,
                    set,
                    &push,
                    self.cfg.n_head as u32,
                    n_tok.div_ceil(64),
                    true,
                )
            }
            BatchAttnKind::Tiled8 => {
                let push = [
                    self.cfg.n_head as u32,
                    self.cfg.n_head_kv as u32,
                    self.cfg.head_dim as u32,
                    chunk_pos as u32,
                    n_tok,
                ];
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.attn_batch,
                    set,
                    &push,
                    self.cfg.n_head as u32,
                    n_tok.div_ceil(8),
                    true,
                )
            }
            BatchAttnKind::Legacy1 => {
                let push = [
                    self.cfg.n_head as u32,
                    self.cfg.n_head_kv as u32,
                    self.cfg.head_dim as u32,
                    (chunk_pos + 1) as u32,
                ];
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.attn_batch,
                    set,
                    &push,
                    self.cfg.n_head as u32,
                    n_tok,
                    true,
                )
            }
        }
    }

    /// Executes a batched prompt prefill on Vulkan GPU in chunks of up to `max_batch` tokens.
    /// Validates all token IDs and context bounds BEFORE mutating any layer state (`MUST #4 & #6`).
    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        if token_ids.is_empty() {
            return Err(VulkanError::InvalidDimensions(
                "forward_batch requires at least 1 token".to_string(),
            ));
        }
        if start_pos + token_ids.len() > self.cfg.max_ctx {
            return Err(VulkanError::InvalidDimensions(format!(
                "start_pos {} + len {} > max_ctx {}",
                start_pos,
                token_ids.len(),
                self.cfg.max_ctx
            )));
        }
        for &tok in token_ids {
            if (tok as usize) >= self.cfg.vocab_size {
                return Err(VulkanError::InvalidDimensions(format!(
                    "token_id {tok} >= vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
        }
        if token_ids.len() == 1 {
            return self.forward_token(token_ids[0], start_pos, compute_last_logits);
        }

        let emb_items = embed_items(self.token_embd.dtype, self.cfg.n_embd)?;
        let eps_bits = self.cfg.rms_eps.to_bits();
        let norm_push = [self.cfg.n_embd as u32, 0, eps_bits];
        let split_push = [self.cfg.n_head as u32, self.cfg.head_dim as u32, eps_bits];
        let qk_push = [self.cfg.head_dim as u32, eps_bits];
        let half_rope = self.cfg.rope_dim / 2;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope).max(kv_dim);
        let rope_groups_x = (rope_total as u32).div_ceil(64);
        let hk = self.cfg.ssm_n_group as u32;
        let hv = self.cfg.ssm_dt_rank as u32;
        let s_dim = self.cfg.ssm_state_size as u32;
        let d_conv = self.cfg.ssm_conv_kernel as u32;

        let d = &self.ctx.inner.device;
        let cb = self.cb_batch;
        let total = token_ids.len();
        let mut offset = 0usize;

        while offset < total {
            let b = (total - offset).min(self.max_batch);
            let b_u32 = b as u32;
            let chunk = &token_ids[offset..offset + b];
            let chunk_pos = start_pos + offset;
            let is_last_chunk = offset + b == total;

            {
                let dst = self
                    .token_ids_buf
                    .mapped_ptr()
                    .expect("token_ids_buf mapped") as *mut u32;
                // SAFETY: `token_ids_buf` holds `self.max_batch` u32s and GPU is idle.
                unsafe {
                    core::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, b);
                }
            }

            // SAFETY: `cb` is idle (waited after every chunk).
            unsafe {
                d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                    .ctx("vkResetCommandBuffer")?;
                d.begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .ctx("vkBeginCommandBuffer")?;
            }

            self.cmd_dispatch_set(
                cb,
                &self.pipes.embed_batch,
                self.embed_batch_set,
                &[
                    self.cfg.n_embd as u32,
                    (self.token_embd.row_bytes / 2) as u32,
                ],
                emb_items.div_ceil(64),
                b_u32,
                true,
            )?;
            self.lora
                .cmd_record_embed(&self.ctx, cb, "token_embd.weight", b_u32, true)?;

            let rope_push = [
                self.cfg.n_head as u32,
                self.cfg.n_head_kv as u32,
                self.cfg.head_dim as u32,
                self.cfg.rope_dim as u32,
                chunk_pos as u32,
                1,
                0,
                self.cfg.rope_freq_base.to_bits(),
            ];
            let gates_push = [hv, b_u32];
            let conv_push = [hk, hv, s_dim, d_conv, b_u32, eps_bits];
            let recur_push = [hk, hv, s_dim, b_u32];
            let gnorm_push = [hv, s_dim, eps_bits];

            for layer in &self.layers {
                let ltag = format!("blk.{}", layer.idx);
                let wg_n = format!("{ltag}.ffn_gate.weight");
                let wu_n = format!("{ltag}.ffn_up.weight");
                let wd_n = format!("{ltag}.ffn_down.weight");

                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rms_norm,
                    layer.attn_norm_set,
                    &norm_push,
                    b_u32,
                    1,
                    true,
                )?;

                match &layer.attn {
                    AttnLayerPlan::Gdn {
                        wqkv_batch,
                        wz_batch,
                        w_alpha_batch,
                        w_beta_batch,
                        gates_set,
                        conv_set,
                        recur_set,
                        gnorm_set,
                        w_out_batch,
                        ..
                    } => {
                        let proj_i8 = matches!(wqkv_batch.flavor, GemmFlavor::CoopMatI8)
                            || matches!(wz_batch.flavor, GemmFlavor::CoopMatI8)
                            || matches!(w_alpha_batch.flavor, GemmFlavor::CoopMatI8)
                            || matches!(w_beta_batch.flavor, GemmFlavor::CoopMatI8);
                        self.cmd_quant_if_needed(
                            cb,
                            self.quant_xb_set,
                            b_u32,
                            self.cfg.n_embd,
                            proj_i8,
                        )?;
                        self.cmd_gemm(cb, wqkv_batch, b_u32, false, false)?;
                        self.cmd_gemm(cb, wz_batch, b_u32, false, false)?;
                        self.cmd_gemm(cb, w_alpha_batch, b_u32, false, false)?;
                        self.cmd_gemm(cb, w_beta_batch, b_u32, false, true)?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.attn_qkv.weight"),
                            b_u32,
                        )?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.attn_gate.weight"),
                            b_u32,
                        )?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.ssm_alpha.weight"),
                            b_u32,
                        )?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.ssm_beta.weight"),
                            b_u32,
                        )?;

                        let total_gates = b_u32 * hv;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.gdn_gates,
                            *gates_set,
                            &gates_push,
                            total_gates.div_ceil(64).min(65535),
                            1,
                            false,
                        )?;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.gdn_conv_batch,
                            *conv_set,
                            &conv_push,
                            2 * hk + hv,
                            b_u32,
                            true,
                        )?;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.gdn_conv_state_upd,
                            *conv_set,
                            &conv_push,
                            2 * hk + hv,
                            1,
                            false,
                        )?;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.gdn_recurrence,
                            *recur_set,
                            &recur_push,
                            s_dim.div_ceil(32),
                            hv,
                            true,
                        )?;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.gdn_norm_gated_silu,
                            *gnorm_set,
                            &gnorm_push,
                            hv,
                            b_u32,
                            true,
                        )?;

                        self.cmd_quant_if_needed(
                            cb,
                            self.quant_attn_out_set,
                            b_u32,
                            self.cfg.ssm_d_inner,
                            matches!(w_out_batch.flavor, GemmFlavor::CoopMatI8),
                        )?;
                        self.cmd_gemm(cb, w_out_batch, b_u32, true, true)?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.ssm_out.weight"),
                            b_u32,
                        )?;
                    }
                    AttnLayerPlan::FullAttn {
                        wqg_batch,
                        wk_batch,
                        wv_batch,
                        split_qg_set,
                        k_norm_set,
                        rope_batch_set,
                        attn_batch_set,
                        sigmoid_set,
                        wo_batch,
                        ..
                    } => {
                        let qkv_i8 = matches!(wqg_batch.flavor, GemmFlavor::CoopMatI8)
                            || matches!(wk_batch.flavor, GemmFlavor::CoopMatI8)
                            || matches!(wv_batch.flavor, GemmFlavor::CoopMatI8);
                        self.cmd_quant_if_needed(
                            cb,
                            self.quant_xb_set,
                            b_u32,
                            self.cfg.n_embd,
                            qkv_i8,
                        )?;
                        self.cmd_gemm(cb, wqg_batch, b_u32, false, false)?;
                        self.cmd_gemm(cb, wk_batch, b_u32, false, false)?;
                        self.cmd_gemm(cb, wv_batch, b_u32, false, true)?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.attn_q.weight"),
                            b_u32,
                        )?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.attn_k.weight"),
                            b_u32,
                        )?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.attn_v.weight"),
                            b_u32,
                        )?;

                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.split_qg_qnorm,
                            *split_qg_set,
                            &split_push,
                            self.cfg.n_head as u32,
                            b_u32,
                            false,
                        )?;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.qk_norm,
                            *k_norm_set,
                            &qk_push,
                            self.cfg.n_head_kv as u32,
                            b_u32,
                            true,
                        )?;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.rope_batch,
                            *rope_batch_set,
                            &rope_push,
                            rope_groups_x,
                            b_u32,
                            true,
                        )?;
                        self.cmd_attn_batch(cb, *attn_batch_set, chunk_pos, b_u32)?;

                        let total_q = (b * q_dim) as u32;
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.sigmoid_mul,
                            *sigmoid_set,
                            &[total_q],
                            total_q.div_ceil(256).min(65535),
                            1,
                            true,
                        )?;

                        self.cmd_quant_if_needed(
                            cb,
                            self.quant_attn_out_set,
                            b_u32,
                            q_dim,
                            matches!(wo_batch.flavor, GemmFlavor::CoopMatI8),
                        )?;
                        self.cmd_gemm(cb, wo_batch, b_u32, true, true)?;
                        self.lora.cmd_record_linear(
                            &self.ctx,
                            cb,
                            &format!("{ltag}.attn_output.weight"),
                            b_u32,
                        )?;
                    }
                }

                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rms_norm,
                    layer.post_attn_norm_set,
                    &norm_push,
                    b_u32,
                    1,
                    true,
                )?;

                let force_sep = self.lora.has_linear(&wg_n) || self.lora.has_linear(&wu_n);
                if force_sep {
                    let (ref gate, ref up, silu_set) = layer.ffn_batch_sep;
                    let gu_i8 = matches!(gate.flavor, GemmFlavor::CoopMatI8)
                        || matches!(up.flavor, GemmFlavor::CoopMatI8);
                    self.cmd_quant_if_needed(cb, self.quant_xb_set, b_u32, self.cfg.n_embd, gu_i8)?;
                    self.cmd_gemm(cb, gate, b_u32, false, false)?;
                    self.cmd_gemm(cb, up, b_u32, false, true)?;
                    self.lora.cmd_record_linear(&self.ctx, cb, &wg_n, b_u32)?;
                    self.lora.cmd_record_linear(&self.ctx, cb, &wu_n, b_u32)?;
                    let total_ff = (b * self.cfg.n_ff) as u32;
                    let gx = total_ff.div_ceil(256).min(65535);
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.silu_mul,
                        silu_set,
                        &[total_ff],
                        gx,
                        1,
                        true,
                    )?;
                } else {
                    match &layer.ffn_batch {
                        BatchFfnPlan::FusedPairSiluCoopMat {
                            pipe,
                            set,
                            cols,
                            rows,
                            row_bytes,
                        } => {
                            let push = [*cols as u32, *rows as u32, b_u32, (*row_bytes / 2) as u32];
                            self.cmd_dispatch_set(
                                cb,
                                pipe,
                                *set,
                                &push,
                                (*rows as u32).div_ceil(64),
                                b_u32.div_ceil(128),
                                true,
                            )?;
                        }
                        BatchFfnPlan::Separate { gate, up, silu_set } => {
                            let gu_i8 = matches!(gate.flavor, GemmFlavor::CoopMatI8)
                                || matches!(up.flavor, GemmFlavor::CoopMatI8);
                            self.cmd_quant_if_needed(
                                cb,
                                self.quant_xb_set,
                                b_u32,
                                self.cfg.n_embd,
                                gu_i8,
                            )?;
                            self.cmd_gemm(cb, gate, b_u32, false, false)?;
                            self.cmd_gemm(cb, up, b_u32, false, true)?;
                            let total_ff = (b * self.cfg.n_ff) as u32;
                            let gx = total_ff.div_ceil(256).min(65535);
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.silu_mul,
                                *silu_set,
                                &[total_ff],
                                gx,
                                1,
                                true,
                            )?;
                        }
                    }
                }

                self.cmd_quant_if_needed(
                    cb,
                    self.quant_hb_set,
                    b_u32,
                    self.cfg.n_ff,
                    matches!(layer.down_batch.flavor, GemmFlavor::CoopMatI8),
                )?;
                self.cmd_gemm(cb, &layer.down_batch, b_u32, true, true)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wd_n, b_u32)?;
            }

            if is_last_chunk && compute_last_logits {
                let last_x_off = ((b - 1) * self.cfg.n_embd) as u32;
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rms_norm,
                    self.output_norm_set,
                    &[self.cfg.n_embd as u32, last_x_off, eps_bits],
                    1,
                    1,
                    true,
                )?;
                let lm = &self.lm_head_dec;
                self.cmd_dispatch_set(
                    cb,
                    &lm.0,
                    lm.1,
                    &lm.3,
                    lm.2,
                    1,
                    self.lora.has_linear("output.weight"),
                )?;
                self.lora
                    .cmd_record_linear(&self.ctx, cb, "output.weight", 1)?;

                let to_xfer = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                let to_host = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                // SAFETY: valid buffers and ranges.
                unsafe {
                    d.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[to_xfer],
                        &[],
                        &[],
                    );
                    d.cmd_copy_buffer(
                        cb,
                        self.slabs[self.logits_dev.slab].buf,
                        self.logits_host.buf,
                        &[vk::BufferCopy {
                            src_offset: self.logits_dev.offset,
                            dst_offset: 0,
                            size: (self.cfg.vocab_size * 4) as u64,
                        }],
                    );
                    d.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[to_host],
                        &[],
                        &[],
                    );
                }
            }

            // SAFETY: valid recording command buffer.
            unsafe { d.end_command_buffer(cb) }.ctx("vkEndCommandBuffer")?;
            self.submit_and_wait(cb)?;

            offset += b;
        }

        Ok(self.logits())
    }

    /// Profiles single-command-buffer decode and batched prefill on Qwen3.5 using Vulkan
    /// hardware timestamp queries (`vkCmdWriteTimestamp`).
    pub fn profile(
        &mut self,
        decode_runs: usize,
        prefill_len: usize,
    ) -> Result<VkProfileReport, VulkanError> {
        let n_pp = prefill_len
            .min(self.cfg.max_ctx.saturating_sub(decode_runs + 4))
            .max(1);
        let n_dec = decode_runs.max(1);

        let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
        let t_warm = Instant::now();
        let mut warm_passes = 0usize;
        while warm_passes < 2 || t_warm.elapsed().as_secs_f64() < 0.35 {
            self.reset_state()?;
            if n_pp >= 2 {
                let _ = self.forward_batch(&pp_ids, 0, true)?;
            } else {
                let _ = self.forward_token_greedy(1, 0)?;
            }
            warm_passes += 1;
        }
        for i in 0..4 {
            let _ = self.forward_token_greedy(1, n_pp + i)?;
        }

        // 1. Measure clean prefill wall time after resetting recurrent state
        self.reset_state()?;
        let t_pp = Instant::now();
        let _ = self.forward_batch(&pp_ids, 0, true)?;
        let prefill_wall_ms = t_pp.elapsed().as_secs_f64() * 1e3;

        // 2. Measure true 1-CB-per-token greedy decode wall time
        let has_ts = self.ctx.info().timestamp_compute;
        let ns_per_tick = self.ctx.info().timestamp_period_ns as f64;

        let mut wall_sum_s = 0.0f64;
        for step in 0..n_dec {
            let t0 = Instant::now();
            let _ = self.forward_token_greedy(1, n_pp + step)?;
            wall_sum_s += t0.elapsed().as_secs_f64();
        }
        let decode_single_cb_wall_us = (wall_sum_s * 1e6) / (n_dec as f64);
        let inner = self.ctx.inner.clone();
        let d = &inner.device;

        const N_STAGES: usize = 13;
        let stage_names: [&'static str; N_STAGES] = [
            "embed",
            "attn_norm",
            "gdn_proj",
            "gdn_conv_l2",
            "gdn_recur+gnorm",
            "gdn_ssm_out+res",
            "full_qkv+split+rope",
            "full_attn+gate",
            "full_wo+res",
            "ffn_norm",
            "ffn_gate_up_silu",
            "ffn_down+res",
            "lm_head+norm",
        ];

        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let conv_dim = self.cfg.conv_dim();
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let v_gdn_dim = self.cfg.ssm_d_inner;
        let avg_seq = (n_pp + n_dec / 2) as u64;
        let state_rw_bytes = (2 * hv * s_dim * s_dim * 4) as u64;
        let conv_rw_bytes = (2 * conv_dim * (self.cfg.ssm_conv_kernel - 1) * 4) as u64;

        let mut stage_bytes = [0u64; N_STAGES];
        let mut stage_flops = [0u64; N_STAGES];
        stage_bytes[0] = (self.token_embd.row_bytes + self.cfg.n_embd * 4) as u64;
        for l in &self.layers {
            stage_bytes[1] += (self.cfg.n_embd * 12) as u64;
            match &l.attn {
                AttnLayerPlan::Gdn {
                    proj_weight_bytes,
                    out_weight_bytes,
                    ..
                } => {
                    stage_bytes[2] += *proj_weight_bytes;
                    stage_flops[2] +=
                        (2 * self.cfg.n_embd * (conv_dim + v_gdn_dim + 2 * hv)) as u64;
                    stage_bytes[3] += conv_rw_bytes + ((conv_dim * 2) * 4) as u64;
                    stage_bytes[4] += state_rw_bytes;
                    stage_flops[4] += (6 * hv * s_dim * s_dim) as u64;
                    stage_bytes[5] += *out_weight_bytes;
                    stage_flops[5] += (2 * v_gdn_dim * self.cfg.n_embd) as u64;
                }
                AttnLayerPlan::FullAttn {
                    qkv_weight_bytes,
                    wo_weight_bytes,
                    ..
                } => {
                    stage_bytes[6] += *qkv_weight_bytes;
                    stage_flops[6] += (2 * self.cfg.n_embd * (2 * q_dim + 2 * kv_dim)) as u64;
                    stage_bytes[7] += avg_seq * (kv_dim as u64) * 4;
                    stage_flops[7] += 4 * avg_seq * (q_dim as u64);
                    stage_bytes[8] += *wo_weight_bytes;
                    stage_flops[8] += (2 * q_dim * self.cfg.n_embd) as u64;
                }
            }
            stage_bytes[9] += (self.cfg.n_embd * 12) as u64;
            stage_bytes[10] += l.gate_up_weight_bytes;
            stage_flops[10] += (4 * self.cfg.n_embd * self.cfg.n_ff) as u64;
            stage_bytes[11] += l.down_weight_bytes;
            stage_flops[11] += (2 * self.cfg.n_ff * self.cfg.n_embd) as u64;
        }
        stage_bytes[12] = self.out_mat.slice.len;
        stage_flops[12] = (2 * self.cfg.n_embd * self.cfg.vocab_size) as u64;

        let mut stage_gpu_us = [0.0f64; N_STAGES];
        let mut decode_single_cb_gpu_us = decode_single_cb_wall_us;
        let mut prefill_chunk_stages = Vec::new();

        if has_ts {
            let max_queries = (self.cfg.n_layers as u32) * 14 + 16;
            // SAFETY: valid device; query pool is destroyed before returning.
            let qp = unsafe {
                d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(max_queries),
                    None,
                )
            }
            .ctx("vkCreateQueryPool")?;

            let cb = self.cb_batch;
            let mut total_gpu_ns = 0.0f64;
            let mut stage_gpu_ns = [0.0f64; N_STAGES];

            let emb_items = embed_items(self.token_embd.dtype, self.cfg.n_embd)?;
            let eps_bits = self.cfg.rms_eps.to_bits();
            let norm_push = [self.cfg.n_embd as u32, 0, eps_bits];
            let split_push = [self.cfg.n_head as u32, self.cfg.head_dim as u32, eps_bits];
            let qk_push = [self.cfg.head_dim as u32, eps_bits];
            let half_rope = self.cfg.rope_dim / 2;
            let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope).max(kv_dim);
            let rope_groups = (rope_total as u32).div_ceil(64);
            let rope_push = [
                self.cfg.n_head as u32,
                self.cfg.n_head_kv as u32,
                self.cfg.head_dim as u32,
                self.cfg.rope_dim as u32,
                0,
                1,
                0,
                self.cfg.rope_freq_base.to_bits(),
            ];
            let attn_push = [
                self.cfg.n_head as u32,
                self.cfg.n_head_kv as u32,
                self.cfg.head_dim as u32,
                0,
            ];
            let hk_u32 = self.cfg.ssm_n_group as u32;
            let hv_u32 = self.cfg.ssm_dt_rank as u32;
            let s_u32 = self.cfg.ssm_state_size as u32;
            let dc_u32 = self.cfg.ssm_conv_kernel as u32;

            self.reset_state()?;
            let _ = self.forward_batch(&pp_ids, 0, true)?;

            for step in 0..n_dec {
                let pos = n_pp + step;
                self.write_step_params(1, pos);

                let mut q_idx = 0u32;
                let mut spans: Vec<(usize, u32, u32)> =
                    Vec::with_capacity(self.cfg.n_layers * 9 + 4);

                // SAFETY: `cb` is idle, `qp` has `max_queries` slots.
                unsafe {
                    d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                        .ctx("vkResetCommandBuffer")?;
                    d.begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .ctx("vkBeginCommandBuffer")?;
                    d.cmd_reset_query_pool(cb, qp, 0, max_queries);
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, q_idx);
                }
                let q_cb_start = q_idx;
                q_idx += 1;

                let mut mark = |cb: vk::CommandBuffer| -> u32 {
                    let cur = q_idx;
                    q_idx += 1;
                    // SAFETY: `cur < max_queries`.
                    unsafe {
                        d.cmd_write_timestamp(cb, vk::PipelineStageFlags::COMPUTE_SHADER, qp, cur);
                    }
                    cur
                };

                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.embed_dec,
                    self.embed_dec_set,
                    &[
                        self.cfg.n_embd as u32,
                        (self.token_embd.row_bytes / 2) as u32,
                    ],
                    emb_items.div_ceil(64),
                    1,
                    true,
                )?;
                let mut prev = mark(cb);
                spans.push((0, q_cb_start, prev));

                for layer in &self.layers {
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rms_norm,
                        layer.attn_norm_set,
                        &norm_push,
                        1,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((1, prev, cur));
                    prev = cur;

                    match &layer.attn {
                        AttnLayerPlan::Gdn {
                            qkv_z_dec,
                            alpha_beta_dec,
                            gates_set,
                            conv_set,
                            recur_set,
                            gnorm_set,
                            w_out_dec,
                            ..
                        } => {
                            self.cmd_decode_pair(cb, qkv_z_dec, false)?;
                            self.cmd_decode_pair(cb, alpha_beta_dec, true)?;
                            let cur = mark(cb);
                            spans.push((2, prev, cur));
                            prev = cur;

                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_gates,
                                *gates_set,
                                &[hv_u32, 1],
                                hv_u32.div_ceil(64),
                                1,
                                false,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_conv_silu_l2,
                                *conv_set,
                                &[hk_u32, hv_u32, s_u32, dc_u32, 1, eps_bits],
                                2 * hk_u32 + hv_u32,
                                1,
                                true,
                            )?;
                            let cur = mark(cb);
                            spans.push((3, prev, cur));
                            prev = cur;

                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_recurrence,
                                *recur_set,
                                &[hk_u32, hv_u32, s_u32, 1],
                                s_u32.div_ceil(32),
                                hv_u32,
                                true,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_norm_gated_silu,
                                *gnorm_set,
                                &[hv_u32, s_u32, eps_bits],
                                hv_u32,
                                1,
                                true,
                            )?;
                            let cur = mark(cb);
                            spans.push((4, prev, cur));
                            prev = cur;

                            self.cmd_dispatch_set(
                                cb,
                                &w_out_dec.0,
                                w_out_dec.1,
                                &w_out_dec.3,
                                w_out_dec.2,
                                1,
                                true,
                            )?;
                            let cur = mark(cb);
                            spans.push((5, prev, cur));
                            prev = cur;
                        }
                        AttnLayerPlan::FullAttn {
                            qkv_dec,
                            split_qg_set,
                            k_norm_set,
                            rope_dec_set,
                            attn_dec_set,
                            sigmoid_set,
                            wo_dec,
                            ..
                        } => {
                            match qkv_dec {
                                DecodeQkvSets::Fused {
                                    pipe,
                                    set,
                                    groups_x,
                                    push,
                                } => {
                                    self.cmd_dispatch_set(
                                        cb, pipe, *set, push, *groups_x, 1, true,
                                    )?;
                                }
                                DecodeQkvSets::Separate { wqg, wk, wv } => {
                                    self.cmd_dispatch_set(
                                        cb, &wqg.0, wqg.1, &wqg.3, wqg.2, 1, false,
                                    )?;
                                    self.cmd_dispatch_set(cb, &wk.0, wk.1, &wk.3, wk.2, 1, false)?;
                                    self.cmd_dispatch_set(cb, &wv.0, wv.1, &wv.3, wv.2, 1, true)?;
                                }
                            }
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.split_qg_qnorm,
                                *split_qg_set,
                                &split_push,
                                self.cfg.n_head as u32,
                                1,
                                false,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.qk_norm,
                                *k_norm_set,
                                &qk_push,
                                self.cfg.n_head_kv as u32,
                                1,
                                true,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.rope_dec,
                                *rope_dec_set,
                                &rope_push,
                                rope_groups,
                                1,
                                true,
                            )?;
                            let cur = mark(cb);
                            spans.push((6, prev, cur));
                            prev = cur;

                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.attn_dec,
                                *attn_dec_set,
                                &attn_push,
                                self.cfg.n_head as u32,
                                1,
                                true,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.sigmoid_mul,
                                *sigmoid_set,
                                &[q_dim as u32],
                                (q_dim as u32).div_ceil(256),
                                1,
                                true,
                            )?;
                            let cur = mark(cb);
                            spans.push((7, prev, cur));
                            prev = cur;

                            self.cmd_dispatch_set(
                                cb, &wo_dec.0, wo_dec.1, &wo_dec.3, wo_dec.2, 1, true,
                            )?;
                            let cur = mark(cb);
                            spans.push((8, prev, cur));
                            prev = cur;
                        }
                    }

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rms_norm,
                        layer.post_attn_norm_set,
                        &norm_push,
                        1,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((9, prev, cur));
                    prev = cur;

                    match &layer.ffn_dec {
                        DecodeFfnSets::FusedPairSilu {
                            pipe,
                            set,
                            groups_x,
                            push,
                        } => {
                            self.cmd_dispatch_set(cb, pipe, *set, push, *groups_x, 1, true)?;
                        }
                        DecodeFfnSets::Separate { gate, up, silu_set } => {
                            self.cmd_dispatch_set(cb, &gate.0, gate.1, &gate.3, gate.2, 1, false)?;
                            self.cmd_dispatch_set(cb, &up.0, up.1, &up.3, up.2, 1, true)?;
                            let gx = (self.cfg.n_ff as u32).div_ceil(256).min(65535);
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.silu_mul,
                                *silu_set,
                                &[self.cfg.n_ff as u32],
                                gx,
                                1,
                                true,
                            )?;
                        }
                    }
                    let cur = mark(cb);
                    spans.push((10, prev, cur));
                    prev = cur;

                    self.cmd_dispatch_set(
                        cb,
                        &layer.down_dec.0,
                        layer.down_dec.1,
                        &layer.down_dec.3,
                        layer.down_dec.2,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((11, prev, cur));
                    prev = cur;
                }

                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rms_norm,
                    self.output_norm_set,
                    &norm_push,
                    1,
                    1,
                    true,
                )?;
                let lm = &self.lm_head_dec;
                self.cmd_dispatch_set(cb, &lm.0, lm.1, &lm.3, lm.2, 1, true)?;
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.argmax,
                    self.argmax_set,
                    &[self.cfg.vocab_size as u32],
                    1,
                    1,
                    false,
                )?;
                let cur = mark(cb);
                spans.push((12, prev, cur));

                // SAFETY: valid recording CB.
                unsafe { d.end_command_buffer(cb) }.ctx("vkEndCommandBuffer")?;
                self.submit_and_wait(cb)?;

                let mut ts = vec![0u64; q_idx as usize];
                // SAFETY: `submit_and_wait` completed; all `q_idx` queries are ready.
                unsafe {
                    d.get_query_pool_results(
                        qp,
                        0,
                        &mut ts,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .ctx("vkGetQueryPoolResults")?;
                }
                total_gpu_ns +=
                    (ts[cur as usize].saturating_sub(ts[q_cb_start as usize])) as f64 * ns_per_tick;
                for (st, a, b_q) in spans {
                    stage_gpu_ns[st] +=
                        (ts[b_q as usize].saturating_sub(ts[a as usize])) as f64 * ns_per_tick;
                }
            }

            decode_single_cb_gpu_us = (total_gpu_ns * 1e-3) / (n_dec as f64);
            for i in 0..N_STAGES {
                stage_gpu_us[i] = (stage_gpu_ns[i] * 1e-3) / (n_dec as f64);
            }

            // 3. Timestamp-profile one batched prefill chunk (`min(n_pp, self.max_batch)`)
            let b_pp = n_pp.min(self.max_batch);
            if b_pp >= 2 {
                self.reset_state()?;
                let b_u32 = b_pp as u32;
                {
                    let dst = self
                        .token_ids_buf
                        .mapped_ptr()
                        .expect("token_ids_buf mapped") as *mut u32;
                    // SAFETY: `b_pp <= self.max_batch`, GPU is idle.
                    unsafe {
                        core::ptr::copy_nonoverlapping(pp_ids.as_ptr(), dst, b_pp);
                    }
                }
                let mut q_idx = 0u32;
                let mut pp_spans: Vec<(usize, u32, u32)> =
                    Vec::with_capacity(self.cfg.n_layers * 9 + 4);
                // SAFETY: `cb` is idle, `qp` has `max_queries` slots.
                unsafe {
                    d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                        .ctx("vkResetCommandBuffer")?;
                    d.begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .ctx("vkBeginCommandBuffer")?;
                    d.cmd_reset_query_pool(cb, qp, 0, max_queries);
                    d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, q_idx);
                }
                let q_start = q_idx;
                q_idx += 1;
                let mut mark_pp = |cb: vk::CommandBuffer| -> u32 {
                    let cur = q_idx;
                    q_idx += 1;
                    // SAFETY: `cur < max_queries`.
                    unsafe {
                        d.cmd_write_timestamp(cb, vk::PipelineStageFlags::COMPUTE_SHADER, qp, cur);
                    }
                    cur
                };

                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.embed_batch,
                    self.embed_batch_set,
                    &[
                        self.cfg.n_embd as u32,
                        (self.token_embd.row_bytes / 2) as u32,
                    ],
                    emb_items.div_ceil(64),
                    b_u32,
                    true,
                )?;
                let mut prev = mark_pp(cb);
                pp_spans.push((0, q_start, prev));

                for layer in &self.layers {
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rms_norm,
                        layer.attn_norm_set,
                        &norm_push,
                        b_u32,
                        1,
                        true,
                    )?;
                    let cur = mark_pp(cb);
                    pp_spans.push((1, prev, cur));
                    prev = cur;

                    match &layer.attn {
                        AttnLayerPlan::Gdn {
                            wqkv_batch,
                            wz_batch,
                            w_alpha_batch,
                            w_beta_batch,
                            gates_set,
                            conv_set,
                            recur_set,
                            gnorm_set,
                            w_out_batch,
                            ..
                        } => {
                            let proj_i8 = matches!(wqkv_batch.flavor, GemmFlavor::CoopMatI8)
                                || matches!(wz_batch.flavor, GemmFlavor::CoopMatI8)
                                || matches!(w_alpha_batch.flavor, GemmFlavor::CoopMatI8)
                                || matches!(w_beta_batch.flavor, GemmFlavor::CoopMatI8);
                            self.cmd_quant_if_needed(
                                cb,
                                self.quant_xb_set,
                                b_u32,
                                self.cfg.n_embd,
                                proj_i8,
                            )?;
                            self.cmd_gemm(cb, wqkv_batch, b_u32, false, false)?;
                            self.cmd_gemm(cb, wz_batch, b_u32, false, false)?;
                            self.cmd_gemm(cb, w_alpha_batch, b_u32, false, false)?;
                            self.cmd_gemm(cb, w_beta_batch, b_u32, false, true)?;
                            let cur = mark_pp(cb);
                            pp_spans.push((2, prev, cur));
                            prev = cur;

                            let total_gates = b_u32 * hv_u32;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_gates,
                                *gates_set,
                                &[hv_u32, b_u32],
                                total_gates.div_ceil(64).min(65535),
                                1,
                                false,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_conv_batch,
                                *conv_set,
                                &[hk_u32, hv_u32, s_u32, dc_u32, b_u32, eps_bits],
                                2 * hk_u32 + hv_u32,
                                b_u32,
                                true,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_conv_state_upd,
                                *conv_set,
                                &[hk_u32, hv_u32, s_u32, dc_u32, b_u32, eps_bits],
                                2 * hk_u32 + hv_u32,
                                1,
                                false,
                            )?;
                            let cur = mark_pp(cb);
                            pp_spans.push((3, prev, cur));
                            prev = cur;

                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_recurrence,
                                *recur_set,
                                &[hk_u32, hv_u32, s_u32, b_u32],
                                s_u32.div_ceil(32),
                                hv_u32,
                                true,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.gdn_norm_gated_silu,
                                *gnorm_set,
                                &[hv_u32, s_u32, eps_bits],
                                hv_u32,
                                b_u32,
                                true,
                            )?;
                            let cur = mark_pp(cb);
                            pp_spans.push((4, prev, cur));
                            prev = cur;

                            self.cmd_quant_if_needed(
                                cb,
                                self.quant_attn_out_set,
                                b_u32,
                                self.cfg.ssm_d_inner,
                                matches!(w_out_batch.flavor, GemmFlavor::CoopMatI8),
                            )?;
                            self.cmd_gemm(cb, w_out_batch, b_u32, true, true)?;
                            let cur = mark_pp(cb);
                            pp_spans.push((5, prev, cur));
                            prev = cur;
                        }
                        AttnLayerPlan::FullAttn {
                            wqg_batch,
                            wk_batch,
                            wv_batch,
                            split_qg_set,
                            k_norm_set,
                            rope_batch_set,
                            attn_batch_set,
                            sigmoid_set,
                            wo_batch,
                            ..
                        } => {
                            let qkv_i8 = matches!(wqg_batch.flavor, GemmFlavor::CoopMatI8)
                                || matches!(wk_batch.flavor, GemmFlavor::CoopMatI8)
                                || matches!(wv_batch.flavor, GemmFlavor::CoopMatI8);
                            self.cmd_quant_if_needed(
                                cb,
                                self.quant_xb_set,
                                b_u32,
                                self.cfg.n_embd,
                                qkv_i8,
                            )?;
                            self.cmd_gemm(cb, wqg_batch, b_u32, false, false)?;
                            self.cmd_gemm(cb, wk_batch, b_u32, false, false)?;
                            self.cmd_gemm(cb, wv_batch, b_u32, false, true)?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.split_qg_qnorm,
                                *split_qg_set,
                                &split_push,
                                self.cfg.n_head as u32,
                                b_u32,
                                false,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.qk_norm,
                                *k_norm_set,
                                &qk_push,
                                self.cfg.n_head_kv as u32,
                                b_u32,
                                true,
                            )?;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.rope_batch,
                                *rope_batch_set,
                                &rope_push,
                                rope_groups,
                                b_u32,
                                true,
                            )?;
                            let cur = mark_pp(cb);
                            pp_spans.push((6, prev, cur));
                            prev = cur;

                            self.cmd_attn_batch(cb, *attn_batch_set, 0, b_u32)?;
                            let total_q = (b_pp * q_dim) as u32;
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.sigmoid_mul,
                                *sigmoid_set,
                                &[total_q],
                                total_q.div_ceil(256).min(65535),
                                1,
                                true,
                            )?;
                            let cur = mark_pp(cb);
                            pp_spans.push((7, prev, cur));
                            prev = cur;

                            self.cmd_quant_if_needed(
                                cb,
                                self.quant_attn_out_set,
                                b_u32,
                                q_dim,
                                matches!(wo_batch.flavor, GemmFlavor::CoopMatI8),
                            )?;
                            self.cmd_gemm(cb, wo_batch, b_u32, true, true)?;
                            let cur = mark_pp(cb);
                            pp_spans.push((8, prev, cur));
                            prev = cur;
                        }
                    }

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rms_norm,
                        layer.post_attn_norm_set,
                        &norm_push,
                        b_u32,
                        1,
                        true,
                    )?;
                    let cur = mark_pp(cb);
                    pp_spans.push((9, prev, cur));
                    prev = cur;

                    match &layer.ffn_batch {
                        BatchFfnPlan::FusedPairSiluCoopMat {
                            pipe,
                            set,
                            cols,
                            rows,
                            row_bytes,
                        } => {
                            let push = [*cols as u32, *rows as u32, b_u32, (*row_bytes / 2) as u32];
                            self.cmd_dispatch_set(
                                cb,
                                pipe,
                                *set,
                                &push,
                                (*rows as u32).div_ceil(64),
                                b_u32.div_ceil(128),
                                true,
                            )?;
                        }
                        BatchFfnPlan::Separate { gate, up, silu_set } => {
                            let gu_i8 = matches!(gate.flavor, GemmFlavor::CoopMatI8)
                                || matches!(up.flavor, GemmFlavor::CoopMatI8);
                            self.cmd_quant_if_needed(
                                cb,
                                self.quant_xb_set,
                                b_u32,
                                self.cfg.n_embd,
                                gu_i8,
                            )?;
                            self.cmd_gemm(cb, gate, b_u32, false, false)?;
                            self.cmd_gemm(cb, up, b_u32, false, true)?;
                            let total_ff = (b_pp * self.cfg.n_ff) as u32;
                            let gx = total_ff.div_ceil(256).min(65535);
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.silu_mul,
                                *silu_set,
                                &[total_ff],
                                gx,
                                1,
                                true,
                            )?;
                        }
                    }
                    let cur = mark_pp(cb);
                    pp_spans.push((10, prev, cur));
                    prev = cur;

                    self.cmd_quant_if_needed(
                        cb,
                        self.quant_hb_set,
                        b_u32,
                        self.cfg.n_ff,
                        matches!(layer.down_batch.flavor, GemmFlavor::CoopMatI8),
                    )?;
                    self.cmd_gemm(cb, &layer.down_batch, b_u32, true, true)?;
                    let cur = mark_pp(cb);
                    pp_spans.push((11, prev, cur));
                    prev = cur;
                }

                let last_x_off = ((b_pp - 1) * self.cfg.n_embd) as u32;
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rms_norm,
                    self.output_norm_set,
                    &[self.cfg.n_embd as u32, last_x_off, eps_bits],
                    1,
                    1,
                    true,
                )?;
                let lm = &self.lm_head_dec;
                self.cmd_dispatch_set(cb, &lm.0, lm.1, &lm.3, lm.2, 1, false)?;
                let cur = mark_pp(cb);
                pp_spans.push((12, prev, cur));

                // SAFETY: valid recording CB.
                unsafe { d.end_command_buffer(cb) }.ctx("vkEndCommandBuffer")?;
                self.submit_and_wait(cb)?;

                let mut ts = vec![0u64; q_idx as usize];
                // SAFETY: `submit_and_wait` completed; all `q_idx` queries are ready.
                unsafe {
                    d.get_query_pool_results(
                        qp,
                        0,
                        &mut ts,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .ctx("vkGetQueryPoolResults")?;
                }

                let mut pp_ns = [0.0f64; N_STAGES];
                for (st, a, b_q) in pp_spans {
                    pp_ns[st] +=
                        (ts[b_q as usize].saturating_sub(ts[a as usize])) as f64 * ns_per_tick;
                }
                let b_u64 = b_pp as u64;
                let causal_pairs = b_u64 * (b_u64 + 1) / 2;
                let pp_stage_names: [&'static str; N_STAGES] = [
                    "embed",
                    "attn_norm",
                    "gdn_proj_gemm",
                    "gdn_conv_l2",
                    "gdn_recur+gnorm",
                    "gdn_ssm_out_gemm+res",
                    "full_qkv_gemm+rope",
                    "full_flash_attn+gate",
                    "full_wo_gemm+res",
                    "ffn_norm",
                    "ffn_gate_up_silu",
                    "ffn_down_gemm+res",
                    "lm_head+norm",
                ];
                let mut pp_bytes = [0u64; N_STAGES];
                let mut pp_flops = [0u64; N_STAGES];
                pp_bytes[0] = b_u64 * (self.token_embd.row_bytes + self.cfg.n_embd * 4) as u64;
                for l in &self.layers {
                    pp_bytes[1] += b_u64 * (self.cfg.n_embd * 8) as u64;
                    match &l.attn {
                        AttnLayerPlan::Gdn {
                            proj_weight_bytes,
                            out_weight_bytes,
                            ..
                        } => {
                            pp_bytes[2] += *proj_weight_bytes
                                + b_u64
                                    * ((self.cfg.n_embd + conv_dim + v_gdn_dim + 2 * hv) * 4)
                                        as u64;
                            pp_flops[2] += b_u64
                                * (2 * self.cfg.n_embd * (conv_dim + v_gdn_dim + 2 * hv)) as u64;
                            pp_bytes[3] += conv_rw_bytes + b_u64 * (conv_dim * 8) as u64;
                            pp_bytes[4] += state_rw_bytes
                                + b_u64
                                    * ((2 * self.cfg.ssm_n_group * s_dim + 3 * v_gdn_dim) * 4)
                                        as u64;
                            pp_flops[4] += b_u64 * (6 * hv * s_dim * s_dim) as u64;
                            pp_bytes[5] += *out_weight_bytes
                                + b_u64 * ((v_gdn_dim + 2 * self.cfg.n_embd) * 4) as u64;
                            pp_flops[5] += b_u64 * (2 * v_gdn_dim * self.cfg.n_embd) as u64;
                        }
                        AttnLayerPlan::FullAttn {
                            qkv_weight_bytes,
                            wo_weight_bytes,
                            ..
                        } => {
                            pp_bytes[6] += *qkv_weight_bytes
                                + b_u64 * ((self.cfg.n_embd + 2 * q_dim + 2 * kv_dim) * 4) as u64;
                            pp_flops[6] +=
                                b_u64 * (2 * self.cfg.n_embd * (2 * q_dim + 2 * kv_dim)) as u64;
                            pp_bytes[7] += b_u64 * ((3 * q_dim * 4 + 2 * kv_dim * 2) as u64);
                            pp_flops[7] += 4 * causal_pairs * (q_dim as u64);
                            pp_bytes[8] += *wo_weight_bytes
                                + b_u64 * ((q_dim + 2 * self.cfg.n_embd) * 4) as u64;
                            pp_flops[8] += b_u64 * (2 * q_dim * self.cfg.n_embd) as u64;
                        }
                    }
                    pp_bytes[9] += b_u64 * (self.cfg.n_embd * 8) as u64;
                    pp_bytes[10] += l.gate_up_weight_bytes
                        + b_u64 * ((self.cfg.n_embd + self.cfg.n_ff) * 4) as u64;
                    pp_flops[10] += b_u64 * (4 * self.cfg.n_embd * self.cfg.n_ff) as u64;
                    pp_bytes[11] += l.down_weight_bytes
                        + b_u64 * ((self.cfg.n_ff + 2 * self.cfg.n_embd) * 4) as u64;
                    pp_flops[11] += b_u64 * (2 * self.cfg.n_ff * self.cfg.n_embd) as u64;
                }
                pp_bytes[12] = self.out_mat.slice.len;
                pp_flops[12] = (2 * self.cfg.n_embd * self.cfg.vocab_size) as u64;

                for i in 0..N_STAGES {
                    if pp_ns[i] > 0.0 || pp_bytes[i] > 0 {
                        prefill_chunk_stages.push(VkStageStat {
                            name: pp_stage_names[i],
                            gpu_us: pp_ns[i] * 1e-3,
                            bytes: pp_bytes[i],
                            flops: pp_flops[i],
                        });
                    }
                }
            }

            // SAFETY: GPU is idle after `submit_and_wait`.
            unsafe { d.destroy_query_pool(qp, None) };
        }

        let mut decode_stages = Vec::new();
        for i in 0..N_STAGES {
            if stage_gpu_us[i] > 0.0 || stage_bytes[i] > 0 {
                decode_stages.push(VkStageStat {
                    name: stage_names[i],
                    gpu_us: stage_gpu_us[i],
                    bytes: stage_bytes[i],
                    flops: stage_flops[i],
                });
            }
        }

        Ok(VkProfileReport {
            decode_single_cb_wall_us,
            decode_single_cb_gpu_us,
            decode_stages,
            prefill_backend: self.active_prefill_backend(),
            prefill_tokens: n_pp,
            prefill_wall_ms,
            prefill_chunk_stages,
        })
    }
}

impl Drop for Qwen35VkRunner {
    fn drop(&mut self) {
        let d = &self.ctx.inner.device;
        // SAFETY: wait idle first, then destroy pools and fence before `slabs` and `ctx` drop.
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.cmd_pool, None);
            d.destroy_descriptor_pool(self.desc_pool, None);
        }
    }
}
