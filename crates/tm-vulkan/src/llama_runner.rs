//! End-to-end Vulkan GPU inference runner for `llama`, `SmolLM2`, and `qwen3` GGUF models.
//!
//! Key architectural properties (`AGENTS.md` §5 & `docs/research/11-amdgpu.md` §4):
//! - **Slabbed Device Arena**: All weights, F16 KV caches, and working buffers are planned up
//!   front (`OutOfDeviceMemory` returned before mutating state if VRAM budget is exceeded) and
//!   sub-allocated into ≤512 MiB `VkBuffer` slabs so 8B+ models (>4 GiB total) never exceed
//!   RADV/Vulkan's 4 GiB per-buffer `maxStorageBufferRange`.
//! - **Pre-recorded Decode Command Buffers**: Static descriptor sets and three reusable decode
//!   command buffers (`no_logits`, `logits`, `greedy`) are recorded once at load time. Each
//!   decode token writes 12 bytes (`token_id`, `pos`, `seq_len`) to a mapped step buffer and
//!   calls `vkQueueSubmit` once — zero heap allocations, zero descriptor updates, and near-zero
//!   host launch overhead.
//! - **V3 Cooperative Matrix Prefill + Portable Fallback**: Automatically dispatches RDNA4 /
//!   Vulkan `16×16×16` Cooperative Matrix (`VK_KHR_cooperative_matrix`) GEMM (including fused
//!   `Gate+Up+SwiGLU`) and tiled causal FlashAttention (`fa_coopmat`), with automatic runtime
//!   fallback to portable subgroup/LDS tiled kernels (`gemm_tiled` / `gemm_kquant` / `fa_gqa_tiled`)
//!   on GPUs without cooperative matrix (`Intel ANV`, `lavapipe`, `Mali`).

use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::context::{Pipeline, VkBuffer, VkPrefillMode, VulkanContext};
use crate::kernels::{dtype_code, normalize_vk_mat_bytes};
use crate::lora_gpu::{VkEmbedSlot, VkLinearSlot, VkLoraEngine};
use crate::{VkResultExt, VulkanError};

const SPV_EMBED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv"));
const SPV_RMS_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.spv"));
const SPV_QK_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm.spv"));
const SPV_ROPE_KV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope_kv.spv"));
const SPV_GQA_ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gqa_attn.spv"));
const SPV_FA_COOPMAT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_coopmat.spv"));
const SPV_FA_GQA_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fa_gqa_tiled.spv"));
const SPV_GEMV_RES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_res.spv"));
const SPV_GEMV_QKV_Q4_0: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_qkv_q4_0.spv"));
const SPV_GEMV_PAIR_SILU_Q4_0: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemv_pair_silu_q4_0.spv"));
const SPV_SILU_MUL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul.spv"));
const SPV_GEMM_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_tiled.spv"));
const SPV_GEMM_KQUANT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_kquant.spv"));
const SPV_GEMM_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_f16.spv"));
const SPV_GEMM_PAIR_SILU_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_pair_silu_coopmat_f16.spv"));
const SPV_QUANTIZE_Q8_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantize_q8_1.spv"));
const SPV_GEMM_COOPMAT_I8: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_i8.spv"));
const SPV_ARGMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
const SPV_ADD_BIAS_AXPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lora_up_axpy.spv"));

/// Preferred maximum tokens per batched prefill submission (`pp512` executes in a single pass!).
const PREFERRED_MAX_BATCH: usize = 512;
/// Target maximum size per device-local `VkBuffer` slab (512 MiB).
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

/// Model hyperparameters for [`LlamaVkRunner`].
#[derive(Debug, Clone)]
pub struct LlamaVkConfig {
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
    pub rope_type: u32, // 0 = Norm, 1 = NeoX
    pub sliding_window: Option<usize>,
    pub use_rope: Vec<bool>,
}

impl LlamaVkConfig {
    #[inline]
    pub fn layer_uses_rope(&self, layer: usize) -> bool {
        self.use_rope.get(layer).copied().unwrap_or(true)
    }
}

/// Hardware GPU timing for one pipeline stage.
#[derive(Debug, Clone)]
pub struct VkStageStat {
    pub name: &'static str,
    pub gpu_us: f64,
    pub bytes: u64,
    pub flops: u64,
}

/// End-to-end decode and prefill GPU profile report.
#[derive(Debug, Clone)]
pub struct VkProfileReport {
    pub decode_single_cb_wall_us: f64,
    pub decode_single_cb_gpu_us: f64,
    pub decode_stages: Vec<VkStageStat>,
    pub prefill_backend: &'static str,
    pub prefill_tokens: usize,
    pub prefill_wall_ms: f64,
    pub prefill_chunk_stages: Vec<VkStageStat>,
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

enum DecodeQkvSets {
    Fused {
        pipe: Arc<Pipeline>,
        set: vk::DescriptorSet,
        groups_x: u32,
        push: [u32; 4],
    },
    Separate {
        wq: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
        wk: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
        wv: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
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
        gate: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
        up: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
        silu_set: vk::DescriptorSet,
    },
}

struct LayerPlan {
    idx: usize,
    attn_norm_set: vk::DescriptorSet,
    qkv_dec: DecodeQkvSets,
    wq_batch: BatchGemmEntry,
    wk_batch: BatchGemmEntry,
    wv_batch: BatchGemmEntry,
    bq_set: Option<vk::DescriptorSet>,
    bk_set: Option<vk::DescriptorSet>,
    bv_set: Option<vk::DescriptorSet>,
    bo_set: Option<vk::DescriptorSet>,
    q_norm_set: Option<vk::DescriptorSet>,
    k_norm_set: Option<vk::DescriptorSet>,
    rope_dec_set: vk::DescriptorSet,
    rope_batch_set: vk::DescriptorSet,
    attn_dec_set: vk::DescriptorSet,
    attn_batch_set: vk::DescriptorSet,
    wo_dec: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
    wo_batch: BatchGemmEntry,
    ffn_norm_set: vk::DescriptorSet,
    ffn_dec: DecodeFfnSets,
    #[allow(clippy::type_complexity)]
    ffn_dec_sep: (
        (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
        (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
        vk::DescriptorSet,
    ),
    ffn_batch: BatchFfnPlan,
    ffn_batch_sep: (BatchGemmEntry, BatchGemmEntry, vk::DescriptorSet),
    down_dec: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
    down_batch: BatchGemmEntry,
    // Byte sizes for stage bandwidth accounting in `profile`
    qkv_weight_bytes: u64,
    wo_weight_bytes: u64,
    gate_up_weight_bytes: u64,
    down_weight_bytes: u64,
}

struct RunnerPipes {
    embed_dec: Arc<Pipeline>,
    embed_batch: Arc<Pipeline>,
    rms_norm: Arc<Pipeline>,
    qk_norm: Arc<Pipeline>,
    add_bias: Arc<Pipeline>,
    rope_dec: Arc<Pipeline>,
    rope_batch: Arc<Pipeline>,
    attn_dec: Arc<Pipeline>,
    attn_batch: Arc<Pipeline>,
    attn_batch_kind: BatchAttnKind,
    silu_mul: Arc<Pipeline>,
    quant_q8_1: Option<Arc<Pipeline>>,
    argmax: Arc<Pipeline>,
}

/// Complete Vulkan GPU inference runner for `llama`, `SmolLM2`, and `qwen3` GGUF models.
pub struct LlamaVkRunner {
    pub cfg: LlamaVkConfig,
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
    quant_hb_set: Option<vk::DescriptorSet>,
    output_norm_set: vk::DescriptorSet,
    lm_head_dec: (Arc<Pipeline>, vk::DescriptorSet, u32, [u32; 4]),
    output_bias_set: Option<vk::DescriptorSet>,
    argmax_set: vk::DescriptorSet,
    layers: Vec<LayerPlan>,
    has_rope_freqs: bool,
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

fn plan_f32_vec(
    planner: &mut ArenaPlanner,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<PendingF32, VulkanError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[len])?;
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

#[allow(clippy::too_many_arguments)]
fn plan_mat_slice<'a>(
    planner: &mut ArenaPlanner,
    gguf: &'a GgufFile,
    fused_name: &str,
    layer: &str,
    cols: usize,
    total_rows: usize,
    row_start: usize,
    rows: usize,
) -> Result<PendingMat<'a>, VulkanError> {
    let (info, raw_bytes) = gguf.req_tensor(fused_name, layer, &[cols, total_rows])?;
    if info.dtype == DType::Q4_0x4 && (!row_start.is_multiple_of(4) || !rows.is_multiple_of(4)) {
        return Err(VulkanError::InvalidDimensions(format!(
            "cannot slice Q4_0x4 tensor '{fused_name}' at non-4-aligned row range {row_start}..{}",
            row_start + rows
        )));
    }
    let raw_row_bytes = info.dtype.row_bytes(cols);
    let byte_off = row_start * raw_row_bytes;
    let byte_end = (row_start + rows) * raw_row_bytes;
    let sub_raw = &raw_bytes[byte_off..byte_end];
    let (eff_dtype, bytes) = normalize_vk_mat_bytes(info.dtype, sub_raw, rows, cols)?;
    let _ = dtype_code(eff_dtype)?;
    let row_bytes = eff_dtype.row_bytes(cols);
    let file_off = gguf.tensor_file_offset(info) as u64 + byte_off as u64;
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

fn plan_opt_f32_vec_any(
    planner: &mut ArenaPlanner,
    gguf: &GgufFile,
    names: &[&str],
    layer: &str,
    len: usize,
) -> Result<Option<PendingF32>, VulkanError> {
    for &name in names {
        if gguf.tensor_info(name).is_some() {
            return Ok(Some(plan_f32_vec(planner, gguf, name, layer, len)?));
        }
    }
    Ok(None)
}

fn plan_opt_f32_vec(
    planner: &mut ArenaPlanner,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<Option<PendingF32>, VulkanError> {
    if gguf.tensor_info(name).is_some() {
        Ok(Some(plan_f32_vec(planner, gguf, name, layer, len)?))
    } else {
        Ok(None)
    }
}

impl LlamaVkRunner {
    /// Loads a `llama`, `qwen2` (Qwen2 & Qwen2.5), or `qwen3` GGUF model onto the selected Vulkan device.
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, VulkanError> {
        Self::from_gguf_with_context(VulkanContext::new()?, gguf, max_ctx)
    }

    /// Loads a `llama`, `qwen2` (Qwen2 & Qwen2.5), or `qwen3` GGUF model onto an explicit [`VulkanContext`].
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
        if !matches!(
            arch.as_str(),
            "llama"
                | "qwen2"
                | "qwen3"
                | "smollm3"
                | "phi3"
                | "mistral"
                | "mistral3"
                | "olmo2"
                | "olmo3"
                | "granite"
                | "exaone"
        ) {
            return Err(VulkanError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        if gguf_ref
            .get_meta(&format!("{pfx}.expert_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0)
            > 0
        {
            return Err(VulkanError::InvalidDimensions(format!(
                "sparse MoE variant of '{arch}' requires MoE shader path on Vulkan LlamaVkRunner"
            )));
        }
        if matches!(pfx, "olmo2" | "olmo3") {
            return Err(VulkanError::InvalidDimensions(
                "olmo2/olmo3 pure Post-Norm + full-vector QK-Norm requires dedicated shaders on Vulkan LlamaVkRunner"
                    .to_string(),
            ));
        }
        for key in [
            format!("{pfx}.embedding_scale"),
            format!("{pfx}.attention.scale"),
            format!("{pfx}.residual_scale"),
            format!("{pfx}.logit_scale"),
        ] {
            if let Some(s) = gguf_ref.get_meta(&key).and_then(|v| v.as_f32())
                && s != 0.0
                && (s - 1.0).abs() > 1e-6
            {
                return Err(VulkanError::InvalidDimensions(format!(
                    "non-unit muP multiplier '{key}' = {s} requires scale shader on Vulkan LlamaVkRunner"
                )));
            }
        }
        let is_qwen = matches!(pfx, "qwen2" | "qwen3");
        let is_neox = matches!(
            pfx,
            "qwen2" | "qwen3" | "phi3" | "olmo2" | "olmo3" | "exaone"
        );
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
            .unwrap_or(n_embd / n_head);
        let rope_dim = gguf_ref
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let max_ctx_train = gguf_ref
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(2048);
        let default_freq_base = if is_qwen {
            1_000_000.0
        } else if pfx == "smollm3" {
            5_000_000.0
        } else {
            10_000.0
        };
        let rope_freq_base = gguf_ref
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(default_freq_base);
        let rms_eps = gguf_ref
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(if is_qwen || pfx == "smollm3" {
                1e-6
            } else {
                1e-5
            });
        let rope_type = if is_neox { 1 } else { 0 };
        let use_rope = if let Some(pat) = gguf_ref
            .get_meta(&format!("{pfx}.rope.pattern"))
            .and_then(|v| v.as_bool_array())
            && pat.len() == n_layers
        {
            pat
        } else if let Some(step) = gguf_ref
            .get_meta(&format!("{pfx}.rope.no_rope_layer_step"))
            .and_then(|v| v.as_usize())
            && step > 0
        {
            (0..n_layers).map(|l| (l + 1) % step != 0).collect()
        } else if pfx == "smollm3" {
            (0..n_layers).map(|l| (l + 1) % 4 != 0).collect()
        } else {
            vec![true; n_layers]
        };
        let sliding_window = if pfx == "phi3" {
            None
        } else {
            gguf_ref
                .get_meta(&format!("{pfx}.attention.sliding_window"))
                .and_then(|v| v.as_usize())
                .filter(|&w| w > 0)
        };
        if let Some(w) = sliding_window
            && max_ctx > w
        {
            return Err(VulkanError::InvalidDimensions(format!(
                "sliding_window ({w}) < max_ctx ({max_ctx}) requires SWA shader on Vulkan LlamaVkRunner"
            )));
        }
        let rope_orig_ctx = gguf_ref
            .get_meta(&format!("{pfx}.rope.scaling.original_context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(max_ctx_train);

        if head_dim > 256 {
            return Err(VulkanError::InvalidDimensions(format!(
                "head_dim {head_dim} > 256 not supported"
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

        // 1. Plan every device allocation up front BEFORE touching GPU memory (`MUST #4 & #6`).
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
        let p_output_bias =
            plan_opt_f32_vec(&mut planner, gguf_ref, "output.bias", "global", vocab_size)?;
        let p_output_norm = plan_f32_vec(
            &mut planner,
            gguf_ref,
            "output_norm.weight",
            "global",
            n_embd,
        )?;
        let rope_freq_opt = if max_ctx > rope_orig_ctx {
            plan_opt_f32_vec_any(
                &mut planner,
                gguf_ref,
                &[
                    "rope_factors_long.weight",
                    "blk.0.rope_factors_long.weight",
                    "rope_freqs.weight",
                    "rope_factors_short.weight",
                    "blk.0.rope_factors_short.weight",
                ],
                "global",
                rope_dim / 2,
            )?
        } else {
            plan_opt_f32_vec_any(
                &mut planner,
                gguf_ref,
                &[
                    "rope_freqs.weight",
                    "rope_factors_short.weight",
                    "blk.0.rope_factors_short.weight",
                ],
                "global",
                rope_dim / 2,
            )?
        };
        let (has_rope_freqs, p_rope_freqs) = match rope_freq_opt {
            Some(p) => (true, p),
            None => {
                let slice = planner.reserve(16)?;
                (
                    false,
                    PendingF32 {
                        slice,
                        vals: vec![1.0f32],
                    },
                )
            }
        };

        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let kv_bytes = (max_ctx * kv_dim * 2) as u64;

        struct PendingLayer<'a> {
            attn_norm: PendingF32,
            ffn_norm: PendingF32,
            q_norm: Option<PendingF32>,
            k_norm: Option<PendingF32>,
            bq: Option<PendingF32>,
            bk: Option<PendingF32>,
            bv: Option<PendingF32>,
            bo: Option<PendingF32>,
            wq: PendingMat<'a>,
            wk: PendingMat<'a>,
            wv: PendingMat<'a>,
            wo: PendingMat<'a>,
            w_gate: PendingMat<'a>,
            w_up: PendingMat<'a>,
            w_down: PendingMat<'a>,
            k_cache: VkSlice,
            v_cache: VkSlice,
        }

        let mut p_layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let q_norm = plan_opt_f32_vec(
                &mut planner,
                gguf_ref,
                &format!("{ltag}.attn_q_norm.weight"),
                &ltag,
                head_dim,
            )?;
            let k_norm = plan_opt_f32_vec(
                &mut planner,
                gguf_ref,
                &format!("{ltag}.attn_k_norm.weight"),
                &ltag,
                head_dim,
            )?;
            let qkv_bias_name = format!("{ltag}.attn_qkv.bias");
            let (bq, bk, bv) = if gguf_ref.tensor_info(&qkv_bias_name).is_some() {
                let (info, bytes) =
                    gguf_ref.req_tensor(&qkv_bias_name, &ltag, &[q_dim + 2 * kv_dim])?;
                let mut full = vec![0.0f32; q_dim + 2 * kv_dim];
                dequantize_row_bytes(info.dtype, bytes, &mut full);
                let s_q = planner.reserve((q_dim * 4) as u64)?;
                let s_k = planner.reserve((kv_dim * 4) as u64)?;
                let s_v = planner.reserve((kv_dim * 4) as u64)?;
                (
                    Some(PendingF32 {
                        slice: s_q,
                        vals: full[..q_dim].to_vec(),
                    }),
                    Some(PendingF32 {
                        slice: s_k,
                        vals: full[q_dim..q_dim + kv_dim].to_vec(),
                    }),
                    Some(PendingF32 {
                        slice: s_v,
                        vals: full[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec(),
                    }),
                )
            } else {
                (
                    plan_opt_f32_vec(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_q.bias"),
                        &ltag,
                        q_dim,
                    )?,
                    plan_opt_f32_vec(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_k.bias"),
                        &ltag,
                        kv_dim,
                    )?,
                    plan_opt_f32_vec(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_v.bias"),
                        &ltag,
                        kv_dim,
                    )?,
                )
            };
            let bo = plan_opt_f32_vec(
                &mut planner,
                gguf_ref,
                &format!("{ltag}.attn_output.bias"),
                &ltag,
                n_embd,
            )?;
            let q_name = format!("{ltag}.attn_q.weight");
            let qkv_name = format!("{ltag}.attn_qkv.weight");
            let (wq, wk, wv) = if gguf_ref.tensor_info(&q_name).is_some() {
                (
                    plan_mat(&mut planner, gguf_ref, &q_name, &ltag, n_embd, q_dim)?,
                    plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_k.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?,
                    plan_mat(
                        &mut planner,
                        gguf_ref,
                        &format!("{ltag}.attn_v.weight"),
                        &ltag,
                        n_embd,
                        kv_dim,
                    )?,
                )
            } else {
                let total_qkv = q_dim + 2 * kv_dim;
                (
                    plan_mat_slice(
                        &mut planner,
                        gguf_ref,
                        &qkv_name,
                        &ltag,
                        n_embd,
                        total_qkv,
                        0,
                        q_dim,
                    )?,
                    plan_mat_slice(
                        &mut planner,
                        gguf_ref,
                        &qkv_name,
                        &ltag,
                        n_embd,
                        total_qkv,
                        q_dim,
                        kv_dim,
                    )?,
                    plan_mat_slice(
                        &mut planner,
                        gguf_ref,
                        &qkv_name,
                        &ltag,
                        n_embd,
                        total_qkv,
                        q_dim + kv_dim,
                        kv_dim,
                    )?,
                )
            };
            let gate_name = format!("{ltag}.ffn_gate.weight");
            let up_name = format!("{ltag}.ffn_up.weight");
            let (w_gate, w_up) = if gguf_ref.tensor_info(&gate_name).is_some() {
                (
                    plan_mat(&mut planner, gguf_ref, &gate_name, &ltag, n_embd, n_ff)?,
                    plan_mat(&mut planner, gguf_ref, &up_name, &ltag, n_embd, n_ff)?,
                )
            } else {
                (
                    plan_mat_slice(
                        &mut planner,
                        gguf_ref,
                        &up_name,
                        &ltag,
                        n_embd,
                        2 * n_ff,
                        0,
                        n_ff,
                    )?,
                    plan_mat_slice(
                        &mut planner,
                        gguf_ref,
                        &up_name,
                        &ltag,
                        n_embd,
                        2 * n_ff,
                        n_ff,
                        n_ff,
                    )?,
                )
            };
            p_layers.push(PendingLayer {
                attn_norm: plan_f32_vec(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.attn_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                ffn_norm: plan_f32_vec(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.ffn_norm.weight"),
                    &ltag,
                    n_embd,
                )?,
                q_norm,
                k_norm,
                bq,
                bk,
                bv,
                bo,
                wq,
                wk,
                wv,
                wo: plan_mat(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                )?,
                w_gate,
                w_up,
                w_down: plan_mat(
                    &mut planner,
                    gguf_ref,
                    &format!("{ltag}.ffn_down.weight"),
                    &ltag,
                    n_ff,
                    n_embd,
                )?,
                k_cache: planner.reserve(kv_bytes)?,
                v_cache: planner.reserve(kv_bytes)?,
            });
        }

        let budget = if info.unified_memory {
            info.device_local_bytes * 90 / 100
        } else {
            info.device_local_bytes * 97 / 100
        };
        let max_dim = n_ff.max(n_embd).max(q_dim);
        let range_batch_cap =
            ((info.max_storage_buffer_range as usize) / (max_dim * 4 * 2)).max(32);
        let base_bytes = planner.total_bytes();
        let per_tok_scratch =
            ((n_embd + n_embd.max(q_dim) + q_dim + 2 * kv_dim + 2 * n_ff + max_dim) * 4) as u64;
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
        let s_xb = planner.reserve((max_batch * n_embd.max(q_dim) * 4) as u64)?;
        let s_q = planner.reserve((max_batch * q_dim * 4) as u64)?;
        let s_k = planner.reserve((max_batch * kv_dim * 4) as u64)?;
        let s_v = planner.reserve((max_batch * kv_dim * 4) as u64)?;
        let s_hb = planner.reserve((max_batch * n_ff * 4) as u64)?;
        let s_hb2 = planner.reserve((max_batch * n_ff * 4) as u64)?;
        let (s_xq, s_xd) = if prefill_kind == ResolvedPrefill::CoopMatI8 {
            let max_blocks = (max_batch * max_dim) / 32;
            (
                Some(planner.reserve((max_blocks * 32) as u64)?),
                Some(planner.reserve((max_blocks * 8) as u64)?),
            )
        } else {
            (None, None)
        };
        let logits_dev = planner.reserve((vocab_size * 4) as u64)?;
        let p_bias_ones = {
            let slice = planner.reserve((max_batch * 4) as u64)?;
            PendingF32 {
                slice,
                vals: vec![1.0f32; max_batch],
            }
        };

        let total_needed = planner.total_bytes();
        if total_needed > budget {
            return Err(VulkanError::OutOfDeviceMemory {
                needed_mib: total_needed as f64 / (1024.0 * 1024.0),
                budget_mib: budget as f64 / (1024.0 * 1024.0),
                max_ctx,
                device: info.name.clone(),
            });
        }

        // 2. Allocate mutable slabs and upload any non-imported weights.
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
                    // SAFETY: `slab` is a newly allocated mutable transfer-dst buffer idle on the device.
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
        if let Some(ref pob) = p_output_bias {
            write_f32(pob)?;
        }
        write_f32(&p_output_norm)?;
        write_f32(&p_rope_freqs)?;
        write_f32(&p_bias_ones)?;
        for pl in &p_layers {
            write_f32(&pl.attn_norm)?;
            write_f32(&pl.ffn_norm)?;
            if let Some(ref qn) = pl.q_norm {
                write_f32(qn)?;
            }
            if let Some(ref kn) = pl.k_norm {
                write_f32(kn)?;
            }
            if let Some(ref bq) = pl.bq {
                write_f32(bq)?;
            }
            if let Some(ref bk) = pl.bk {
                write_f32(bk)?;
            }
            if let Some(ref bv) = pl.bv {
                write_f32(bv)?;
            }
            if let Some(ref bo) = pl.bo {
                write_f32(bo)?;
            }
            write_mat(&pl.wq)?;
            write_mat(&pl.wk)?;
            write_mat(&pl.wv)?;
            write_mat(&pl.wo)?;
            write_mat(&pl.w_gate)?;
            write_mat(&pl.w_up)?;
            write_mat(&pl.w_down)?;
        }

        let step_buf = ctx.create_host_buffer(32)?;
        let token_ids_buf = ctx.create_host_buffer((max_batch * 4) as u64)?;
        let argmax_buf = ctx.create_host_buffer(16)?;
        let logits_host = ctx.create_host_buffer((vocab_size * 4) as u64)?;

        // 3. Build pipelines (each validates workgroup size against device limits).
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
            add_bias: ctx.pipeline("add_bias", 256, SPV_ADD_BIAS_AXPY, 3, 20, &[256])?,
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
        let max_sets = (n_layers as u32) * 40 + 40;
        let max_desc = max_sets * 8;
        let d = &ctx.inner.device;
        // SAFETY: `d` is a valid logical device and `ctx.inner.queue_family` is its compute queue family.
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
                // SAFETY: `desc_pool`, `pipe.set_layout`, and all buffer handles outlive the runner.
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

        let (quant_xb_set, quant_hb_set) =
            if let (Some(qp), Some(xq), Some(xd)) = (&pipes.quant_q8_1, s_xq, s_xd) {
                (
                    Some(alloc_set(
                        qp,
                        &[Binding::Arena(s_xb), Binding::Arena(xq), Binding::Arena(xd)],
                    )?),
                    Some(alloc_set(
                        qp,
                        &[Binding::Arena(s_hb), Binding::Arena(xq), Binding::Arena(xd)],
                    )?),
                )
            } else {
                (None, None)
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
        let output_bias_set = match p_output_bias {
            Some(ref pob) => Some(alloc_set(
                &pipes.add_bias,
                &[
                    Binding::Arena(pob.slice),
                    Binding::Arena(p_bias_ones.slice),
                    Binding::Arena(logits_dev),
                ],
            )?),
            None => None,
        };
        let argmax_set = alloc_set(
            &pipes.argmax,
            &[Binding::Arena(logits_dev), Binding::Raw(&argmax_buf)],
        )?;

        let mk_lin_slot = |name: String, xs: VkSlice, ys: VkSlice| VkLinearSlot {
            name,
            x_buf: slabs[xs.slab].buf,
            x_off: xs.offset,
            x_len: xs.len,
            y_buf: slabs[ys.slab].buf,
            y_off: ys.offset,
            y_len: ys.len,
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
        let mut linear_slots = Vec::with_capacity(n_layers * 7 + 1);
        linear_slots.push(mk_lin_slot("output.weight".to_string(), s_xb, logits_dev));

        let mut layers = Vec::with_capacity(n_layers);
        for (il, pl) in p_layers.iter().enumerate() {
            let ltag = format!("blk.{il}");
            linear_slots.push(mk_lin_slot(format!("{ltag}.attn_q.weight"), s_xb, s_q));
            linear_slots.push(mk_lin_slot(format!("{ltag}.attn_k.weight"), s_xb, s_k));
            linear_slots.push(mk_lin_slot(format!("{ltag}.attn_v.weight"), s_xb, s_v));
            linear_slots.push(mk_lin_slot(format!("{ltag}.attn_output.weight"), s_xb, s_x));
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

            let qkv_dec = if pl.wq.mat.dtype == DType::Q4_0
                && pl.wk.mat.dtype == DType::Q4_0
                && pl.wv.mat.dtype == DType::Q4_0
                && pl.wq.mat.rows.is_multiple_of(2)
                && pl.wk.mat.rows.is_multiple_of(2)
            {
                let wg = 128u32;
                let nr = 2u32;
                let pipe =
                    ctx.pipeline("gemv_qkv_q4_0", wg, SPV_GEMV_QKV_Q4_0, 7, 16, &[wg, nr])?;
                let set = alloc_set(
                    &pipe,
                    &[
                        Binding::Arena(pl.wq.slice),
                        Binding::Arena(pl.wk.slice),
                        Binding::Arena(pl.wv.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_q),
                        Binding::Arena(s_k),
                        Binding::Arena(s_v),
                    ],
                )?;
                let total_rows = (pl.wq.mat.rows + pl.wk.mat.rows + pl.wv.mat.rows) as u32;
                let rows_per_wg = (wg / sg).max(1) * nr;
                DecodeQkvSets::Fused {
                    pipe,
                    set,
                    groups_x: total_rows.div_ceil(rows_per_wg),
                    push: [
                        pl.wq.mat.cols as u32,
                        pl.wq.mat.rows as u32,
                        pl.wk.mat.rows as u32,
                        pl.wv.mat.rows as u32,
                    ],
                }
            } else {
                let (pq, gq, push_q) = gemv_pipe_for(&pl.wq.mat)?;
                let sq = alloc_set(
                    &pq,
                    &[
                        Binding::Arena(pl.wq.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_q),
                    ],
                )?;
                let (pk, gk, push_k) = gemv_pipe_for(&pl.wk.mat)?;
                let sk = alloc_set(
                    &pk,
                    &[
                        Binding::Arena(pl.wk.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_k),
                    ],
                )?;
                let (pv, gv, push_v) = gemv_pipe_for(&pl.wv.mat)?;
                let sv = alloc_set(
                    &pv,
                    &[
                        Binding::Arena(pl.wv.slice),
                        Binding::Arena(s_xb),
                        Binding::Arena(s_v),
                    ],
                )?;
                DecodeQkvSets::Separate {
                    wq: (pq, sq, gq, push_q),
                    wk: (pk, sk, gk, push_k),
                    wv: (pv, sv, gv, push_v),
                }
            };

            let wq_batch = make_batch_gemm(pl.wq.mat, s_xb, s_q)?;
            let wk_batch = make_batch_gemm(pl.wk.mat, s_xb, s_k)?;
            let wv_batch = make_batch_gemm(pl.wv.mat, s_xb, s_v)?;

            let mk_bias_set = |pb: &Option<PendingF32>,
                               dst: VkSlice|
             -> Result<Option<vk::DescriptorSet>, VulkanError> {
                match pb {
                    Some(b) => Ok(Some(alloc_set(
                        &pipes.add_bias,
                        &[
                            Binding::Arena(b.slice),
                            Binding::Arena(p_bias_ones.slice),
                            Binding::Arena(dst),
                        ],
                    )?)),
                    None => Ok(None),
                }
            };
            let bq_set = mk_bias_set(&pl.bq, s_q)?;
            let bk_set = mk_bias_set(&pl.bk, s_k)?;
            let bv_set = mk_bias_set(&pl.bv, s_v)?;
            let bo_set = mk_bias_set(&pl.bo, s_x)?;

            let q_norm_set = match pl.q_norm {
                Some(ref qn) => Some(alloc_set(
                    &pipes.qk_norm,
                    &[Binding::Arena(s_q), Binding::Arena(qn.slice)],
                )?),
                None => None,
            };
            let k_norm_set = match pl.k_norm {
                Some(ref kn) => Some(alloc_set(
                    &pipes.qk_norm,
                    &[Binding::Arena(s_k), Binding::Arena(kn.slice)],
                )?),
                None => None,
            };

            let rope_binds = [
                Binding::Arena(s_q),
                Binding::Arena(s_k),
                Binding::Arena(s_v),
                Binding::Arena(pl.k_cache),
                Binding::Arena(pl.v_cache),
                Binding::Arena(p_rope_freqs.slice),
                Binding::Raw(&step_buf),
            ];
            let rope_dec_set = alloc_set(&pipes.rope_dec, &rope_binds)?;
            let rope_batch_set = alloc_set(&pipes.rope_batch, &rope_binds)?;

            let attn_dec_set = alloc_set(
                &pipes.attn_dec,
                &[
                    Binding::Arena(s_q),
                    Binding::Arena(pl.k_cache),
                    Binding::Arena(pl.v_cache),
                    Binding::Arena(s_xb),
                    Binding::Raw(&step_buf),
                ],
            )?;
            let attn_batch_set = match pipes.attn_batch_kind {
                BatchAttnKind::CoopMat64 | BatchAttnKind::Tiled8 => alloc_set(
                    &pipes.attn_batch,
                    &[
                        Binding::Arena(s_q),
                        Binding::Arena(pl.k_cache),
                        Binding::Arena(pl.v_cache),
                        Binding::Arena(s_xb),
                    ],
                )?,
                BatchAttnKind::Legacy1 => alloc_set(
                    &pipes.attn_batch,
                    &[
                        Binding::Arena(s_q),
                        Binding::Arena(pl.k_cache),
                        Binding::Arena(pl.v_cache),
                        Binding::Arena(s_xb),
                        Binding::Raw(&step_buf),
                    ],
                )?,
            };

            let (p_wo, g_wo, mut push_wo) = gemv_pipe_for(&pl.wo.mat)?;
            push_wo[2] = 1;
            let s_wo = alloc_set(
                &p_wo,
                &[
                    Binding::Arena(pl.wo.slice),
                    Binding::Arena(s_xb),
                    Binding::Arena(s_x),
                ],
            )?;
            let wo_batch = make_batch_gemm(pl.wo.mat, s_xb, s_x)?;

            let ffn_norm_set = alloc_set(
                &pipes.rms_norm,
                &[
                    Binding::Arena(s_x),
                    Binding::Arena(pl.ffn_norm.slice),
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

            layers.push(LayerPlan {
                idx: il,
                attn_norm_set,
                qkv_dec,
                wq_batch,
                wk_batch,
                wv_batch,
                bq_set,
                bk_set,
                bv_set,
                bo_set,
                q_norm_set,
                k_norm_set,
                rope_dec_set,
                rope_batch_set,
                attn_dec_set,
                attn_batch_set,
                wo_dec: (p_wo, s_wo, g_wo, push_wo),
                wo_batch,
                ffn_norm_set,
                ffn_dec,
                ffn_dec_sep,
                ffn_batch,
                ffn_batch_sep,
                down_dec: (p_dn, s_dn, g_dn, push_dn),
                down_batch,
                qkv_weight_bytes: pl.wq.slice.len + pl.wk.slice.len + pl.wv.slice.len,
                wo_weight_bytes: pl.wo.slice.len,
                gate_up_weight_bytes: pl.w_gate.slice.len + pl.w_up.slice.len,
                down_weight_bytes: pl.w_down.slice.len,
            });
        }

        let lora = VkLoraEngine::new(&ctx)?;

        let mut runner = Self {
            cfg: LlamaVkConfig {
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
                sliding_window,
                use_rope,
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
            quant_hb_set,
            output_norm_set,
            lm_head_dec,
            output_bias_set,
            argmax_set,
            layers,
            has_rope_freqs,
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
            _gguf: gguf.clone(),
        };
        runner.prerecord_decode_command_buffers()?;
        Ok(runner)
    }

    /// Binds dynamic GGUF LoRA adapters onto this Vulkan runner and re-records static decode CBs (`Phase T4`).
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

    fn cmd_add_bias(
        &self,
        cb: vk::CommandBuffer,
        bias_set: vk::DescriptorSet,
        d_out: usize,
        n_tok: u32,
        barrier_after: bool,
    ) -> Result<(), VulkanError> {
        self.cmd_dispatch_set(
            cb,
            &self.pipes.add_bias,
            bias_set,
            &[0, d_out as u32, 1, n_tok, 1.0f32.to_bits()],
            (d_out as u32).div_ceil(256),
            n_tok,
            barrier_after,
        )
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
        let qk_push = [self.cfg.head_dim as u32, eps_bits];
        let half_rope = self.cfg.rope_dim / 2;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope)
            .max(self.cfg.n_head_kv * self.cfg.head_dim);
        let rope_groups = (rope_total as u32).div_ceil(64);
        let rope_push = [
            self.cfg.n_head as u32,
            self.cfg.n_head_kv as u32,
            self.cfg.head_dim as u32,
            self.cfg.rope_dim as u32,
            0,
            self.cfg.rope_type,
            u32::from(self.has_rope_freqs),
            self.cfg.rope_freq_base.to_bits(),
        ];
        let attn_push = [
            self.cfg.n_head as u32,
            self.cfg.n_head_kv as u32,
            self.cfg.head_dim as u32,
            0,
        ];

        for layer in &self.layers {
            let ltag = format!("blk.{}", layer.idx);
            let wq_n = format!("{ltag}.attn_q.weight");
            let wk_n = format!("{ltag}.attn_k.weight");
            let wv_n = format!("{ltag}.attn_v.weight");
            let wo_n = format!("{ltag}.attn_output.weight");
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

            match &layer.qkv_dec {
                DecodeQkvSets::Fused {
                    pipe,
                    set,
                    groups_x,
                    push,
                } => {
                    self.cmd_dispatch_set(cb, pipe, *set, push, *groups_x, 1, true)?;
                }
                DecodeQkvSets::Separate { wq, wk, wv } => {
                    self.cmd_dispatch_set(cb, &wq.0, wq.1, &wq.3, wq.2, 1, false)?;
                    self.cmd_dispatch_set(cb, &wk.0, wk.1, &wk.3, wk.2, 1, false)?;
                    self.cmd_dispatch_set(cb, &wv.0, wv.1, &wv.3, wv.2, 1, true)?;
                }
            }
            self.lora.cmd_record_linear(&self.ctx, cb, &wq_n, 1)?;
            self.lora.cmd_record_linear(&self.ctx, cb, &wk_n, 1)?;
            self.lora.cmd_record_linear(&self.ctx, cb, &wv_n, 1)?;
            if layer.bq_set.is_some() || layer.bk_set.is_some() || layer.bv_set.is_some() {
                if let Some(bqs) = layer.bq_set {
                    self.cmd_add_bias(cb, bqs, q_dim, 1, false)?;
                }
                if let Some(bks) = layer.bk_set {
                    self.cmd_add_bias(cb, bks, kv_dim, 1, false)?;
                }
                if let Some(bvs) = layer.bv_set {
                    self.cmd_add_bias(cb, bvs, kv_dim, 1, false)?;
                }
                Self::cmd_compute_barrier(d, cb);
            }

            if layer.q_norm_set.is_some() || layer.k_norm_set.is_some() {
                if let Some(qns) = layer.q_norm_set {
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.qk_norm,
                        qns,
                        &qk_push,
                        self.cfg.n_head as u32,
                        1,
                        false,
                    )?;
                }
                if let Some(kns) = layer.k_norm_set {
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.qk_norm,
                        kns,
                        &qk_push,
                        self.cfg.n_head_kv as u32,
                        1,
                        false,
                    )?;
                }
                Self::cmd_compute_barrier(d, cb);
            }

            let mut l_rope_push = rope_push;
            if !self.cfg.layer_uses_rope(layer.idx) {
                l_rope_push[3] = 0;
            }
            self.cmd_dispatch_set(
                cb,
                &self.pipes.rope_dec,
                layer.rope_dec_set,
                &l_rope_push,
                rope_groups,
                1,
                true,
            )?;

            self.cmd_dispatch_set(
                cb,
                &self.pipes.attn_dec,
                layer.attn_dec_set,
                &attn_push,
                self.cfg.n_head as u32,
                1,
                true,
            )?;

            self.cmd_dispatch_set(
                cb,
                &layer.wo_dec.0,
                layer.wo_dec.1,
                &layer.wo_dec.3,
                layer.wo_dec.2,
                1,
                true,
            )?;
            self.lora.cmd_record_linear(&self.ctx, cb, &wo_n, 1)?;
            if let Some(bos) = layer.bo_set {
                self.cmd_add_bias(cb, bos, self.cfg.n_embd, 1, true)?;
            }

            self.cmd_dispatch_set(
                cb,
                &self.pipes.rms_norm,
                layer.ffn_norm_set,
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
                if let Some(obs) = self.output_bias_set {
                    self.cmd_add_bias(cb, obs, self.cfg.vocab_size, 1, true)?;
                }
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
                let need_post_lm_bar =
                    self.lora.has_linear("output.weight") || self.output_bias_set.is_some();
                self.cmd_dispatch_set(cb, &lm.0, lm.1, &lm.3, lm.2, 1, need_post_lm_bar)?;
                self.lora
                    .cmd_record_linear(&self.ctx, cb, "output.weight", 1)?;
                if let Some(obs) = self.output_bias_set {
                    self.cmd_add_bias(cb, obs, self.cfg.vocab_size, 1, false)?;
                }
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
        // serialize all uses of this runner's command buffers.
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
    /// Zero heap allocations occur on this path (`AGENTS.md` rule #4).
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
    /// Zero heap allocations occur on this path (`AGENTS.md` rule #4).
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
        let qk_push = [self.cfg.head_dim as u32, eps_bits];
        let half_rope = self.cfg.rope_dim / 2;
        let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope)
            .max(self.cfg.n_head_kv * self.cfg.head_dim);
        let rope_groups_x = (rope_total as u32).div_ceil(64);
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;

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
                self.cfg.rope_type,
                u32::from(self.has_rope_freqs),
                self.cfg.rope_freq_base.to_bits(),
            ];

            for layer in &self.layers {
                let ltag = format!("blk.{}", layer.idx);
                let wq_n = format!("{ltag}.attn_q.weight");
                let wk_n = format!("{ltag}.attn_k.weight");
                let wv_n = format!("{ltag}.attn_v.weight");
                let wo_n = format!("{ltag}.attn_output.weight");
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

                let qkv_needs_i8 = matches!(layer.wq_batch.flavor, GemmFlavor::CoopMatI8)
                    || matches!(layer.wk_batch.flavor, GemmFlavor::CoopMatI8)
                    || matches!(layer.wv_batch.flavor, GemmFlavor::CoopMatI8);
                self.cmd_quant_if_needed(
                    cb,
                    self.quant_xb_set,
                    b_u32,
                    self.cfg.n_embd,
                    qkv_needs_i8,
                )?;

                self.cmd_gemm(cb, &layer.wq_batch, b_u32, false, false)?;
                self.cmd_gemm(cb, &layer.wk_batch, b_u32, false, false)?;
                self.cmd_gemm(cb, &layer.wv_batch, b_u32, false, true)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wq_n, b_u32)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wk_n, b_u32)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wv_n, b_u32)?;
                if layer.bq_set.is_some() || layer.bk_set.is_some() || layer.bv_set.is_some() {
                    if let Some(bqs) = layer.bq_set {
                        self.cmd_add_bias(cb, bqs, q_dim, b_u32, false)?;
                    }
                    if let Some(bks) = layer.bk_set {
                        self.cmd_add_bias(cb, bks, kv_dim, b_u32, false)?;
                    }
                    if let Some(bvs) = layer.bv_set {
                        self.cmd_add_bias(cb, bvs, kv_dim, b_u32, false)?;
                    }
                    Self::cmd_compute_barrier(d, cb);
                }

                if layer.q_norm_set.is_some() || layer.k_norm_set.is_some() {
                    if let Some(qns) = layer.q_norm_set {
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.qk_norm,
                            qns,
                            &qk_push,
                            self.cfg.n_head as u32,
                            b_u32,
                            false,
                        )?;
                    }
                    if let Some(kns) = layer.k_norm_set {
                        self.cmd_dispatch_set(
                            cb,
                            &self.pipes.qk_norm,
                            kns,
                            &qk_push,
                            self.cfg.n_head_kv as u32,
                            b_u32,
                            false,
                        )?;
                    }
                    Self::cmd_compute_barrier(d, cb);
                }

                let mut l_rope_push = rope_push;
                if !self.cfg.layer_uses_rope(layer.idx) {
                    l_rope_push[3] = 0;
                }
                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rope_batch,
                    layer.rope_batch_set,
                    &l_rope_push,
                    rope_groups_x,
                    b_u32,
                    true,
                )?;

                self.cmd_attn_batch(cb, layer.attn_batch_set, chunk_pos, b_u32)?;

                self.cmd_quant_if_needed(
                    cb,
                    self.quant_xb_set,
                    b_u32,
                    q_dim,
                    matches!(layer.wo_batch.flavor, GemmFlavor::CoopMatI8),
                )?;
                self.cmd_gemm(cb, &layer.wo_batch, b_u32, true, true)?;
                self.lora.cmd_record_linear(&self.ctx, cb, &wo_n, b_u32)?;
                if let Some(bos) = layer.bo_set {
                    self.cmd_add_bias(cb, bos, self.cfg.n_embd, b_u32, true)?;
                }

                self.cmd_dispatch_set(
                    cb,
                    &self.pipes.rms_norm,
                    layer.ffn_norm_set,
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
                let need_post_lm_bar =
                    self.lora.has_linear("output.weight") || self.output_bias_set.is_some();
                self.cmd_dispatch_set(cb, &lm.0, lm.1, &lm.3, lm.2, 1, need_post_lm_bar)?;
                self.lora
                    .cmd_record_linear(&self.ctx, cb, "output.weight", 1)?;
                if let Some(obs) = self.output_bias_set {
                    self.cmd_add_bias(cb, obs, self.cfg.vocab_size, 1, false)?;
                }

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

    /// Profiles single-command-buffer decode and batched prefill using Vulkan hardware
    /// timestamp queries (`vkCmdWriteTimestamp`).
    pub fn profile(
        &mut self,
        decode_runs: usize,
        prefill_len: usize,
    ) -> Result<VkProfileReport, VulkanError> {
        let n_pp = prefill_len
            .min(self.cfg.max_ctx.saturating_sub(decode_runs + 4))
            .max(1);
        let n_dec = decode_runs.max(1);

        // Warmup GPU SMU boost clocks (>= 2 passes / >= 350 ms sustained load so RDNA4 exits idle P-state)
        let pp_ids: Vec<u32> = (0..n_pp).map(|i| ((i % 1000) + 1) as u32).collect();
        let t_warm = Instant::now();
        let mut warm_passes = 0usize;
        while warm_passes < 2 || t_warm.elapsed().as_secs_f64() < 0.35 {
            if n_pp >= 2 {
                let _ = self.forward_batch(&pp_ids, 0, true)?;
            } else {
                let _ = self.forward_token_greedy(1, 0)?;
            }
            warm_passes += 1;
        }
        for i in 0..4 {
            let _ = self.forward_token_greedy(1, i)?;
        }

        // 1. Measure prefill wall time
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
        let d = &self.ctx.inner.device;

        const N_STAGES: usize = 11;
        let stage_names: [&'static str; N_STAGES] = [
            "embed",
            "attn_norm",
            "attn_qkv",
            "qk_norm",
            "rope_kv",
            "gqa_attn",
            "attn_wo+res",
            "ffn_norm",
            "ffn_gate_up_silu",
            "ffn_down+res",
            "lm_head+norm",
        ];

        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let avg_seq = (n_pp + n_dec / 2) as u64;
        let mut stage_bytes = [0u64; N_STAGES];
        let mut stage_flops = [0u64; N_STAGES];
        stage_bytes[0] = (self.token_embd.row_bytes + self.cfg.n_embd * 4) as u64;
        for l in &self.layers {
            stage_bytes[1] += (self.cfg.n_embd * 12) as u64;
            stage_bytes[2] += l.qkv_weight_bytes;
            stage_flops[2] += (2 * self.cfg.n_embd * (q_dim + 2 * kv_dim)) as u64;
            if l.q_norm_set.is_some() || l.k_norm_set.is_some() {
                stage_bytes[3] += ((q_dim + kv_dim) * 8) as u64;
            }
            stage_bytes[4] += ((q_dim + 2 * kv_dim) * 4 + 2 * kv_dim * 2) as u64;
            stage_bytes[5] += avg_seq * (kv_dim as u64) * 4;
            stage_flops[5] += 4 * avg_seq * (q_dim as u64);
            stage_bytes[6] += l.wo_weight_bytes;
            stage_flops[6] += (2 * q_dim * self.cfg.n_embd) as u64;
            stage_bytes[7] += (self.cfg.n_embd * 12) as u64;
            stage_bytes[8] += l.gate_up_weight_bytes;
            stage_flops[8] += (4 * self.cfg.n_embd * self.cfg.n_ff) as u64;
            stage_bytes[9] += l.down_weight_bytes;
            stage_flops[9] += (2 * self.cfg.n_ff * self.cfg.n_embd) as u64;
        }
        stage_bytes[10] = self.out_mat.slice.len;
        stage_flops[10] = (2 * self.cfg.n_embd * self.cfg.vocab_size) as u64;

        let mut stage_gpu_us = [0.0f64; N_STAGES];
        let mut decode_single_cb_gpu_us = decode_single_cb_wall_us;
        let mut prefill_chunk_stages = Vec::new();

        if has_ts {
            let max_queries = (self.cfg.n_layers as u32) * 18 + 8;
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
            let qk_push = [self.cfg.head_dim as u32, eps_bits];
            let half_rope = self.cfg.rope_dim / 2;
            let rope_total = ((self.cfg.n_head + self.cfg.n_head_kv) * half_rope)
                .max(self.cfg.n_head_kv * self.cfg.head_dim);
            let rope_groups = (rope_total as u32).div_ceil(64);
            let rope_push = [
                self.cfg.n_head as u32,
                self.cfg.n_head_kv as u32,
                self.cfg.head_dim as u32,
                self.cfg.rope_dim as u32,
                0,
                self.cfg.rope_type,
                u32::from(self.has_rope_freqs),
                self.cfg.rope_freq_base.to_bits(),
            ];
            let attn_push = [
                self.cfg.n_head as u32,
                self.cfg.n_head_kv as u32,
                self.cfg.head_dim as u32,
                0,
            ];

            for step in 0..n_dec {
                let pos = n_pp + step;
                self.write_step_params(1, pos);

                let mut q_idx = 0u32;
                let mut spans: Vec<(usize, u32, u32)> =
                    Vec::with_capacity(self.cfg.n_layers * 9 + 2);

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

                    match &layer.qkv_dec {
                        DecodeQkvSets::Fused {
                            pipe,
                            set,
                            groups_x,
                            push,
                        } => {
                            self.cmd_dispatch_set(cb, pipe, *set, push, *groups_x, 1, true)?;
                        }
                        DecodeQkvSets::Separate { wq, wk, wv } => {
                            self.cmd_dispatch_set(cb, &wq.0, wq.1, &wq.3, wq.2, 1, false)?;
                            self.cmd_dispatch_set(cb, &wk.0, wk.1, &wk.3, wk.2, 1, false)?;
                            self.cmd_dispatch_set(cb, &wv.0, wv.1, &wv.3, wv.2, 1, true)?;
                        }
                    }
                    if layer.bq_set.is_some() || layer.bk_set.is_some() || layer.bv_set.is_some() {
                        if let Some(bqs) = layer.bq_set {
                            self.cmd_add_bias(cb, bqs, q_dim, 1, false)?;
                        }
                        if let Some(bks) = layer.bk_set {
                            self.cmd_add_bias(cb, bks, kv_dim, 1, false)?;
                        }
                        if let Some(bvs) = layer.bv_set {
                            self.cmd_add_bias(cb, bvs, kv_dim, 1, false)?;
                        }
                        Self::cmd_compute_barrier(d, cb);
                    }
                    let cur = mark(cb);
                    spans.push((2, prev, cur));
                    prev = cur;

                    if layer.q_norm_set.is_some() || layer.k_norm_set.is_some() {
                        if let Some(qns) = layer.q_norm_set {
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.qk_norm,
                                qns,
                                &qk_push,
                                self.cfg.n_head as u32,
                                1,
                                false,
                            )?;
                        }
                        if let Some(kns) = layer.k_norm_set {
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.qk_norm,
                                kns,
                                &qk_push,
                                self.cfg.n_head_kv as u32,
                                1,
                                false,
                            )?;
                        }
                        Self::cmd_compute_barrier(d, cb);
                        let cur = mark(cb);
                        spans.push((3, prev, cur));
                        prev = cur;
                    }

                    let mut l_rope_push = rope_push;
                    if !self.cfg.layer_uses_rope(layer.idx) {
                        l_rope_push[3] = 0;
                    }
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rope_dec,
                        layer.rope_dec_set,
                        &l_rope_push,
                        rope_groups,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((4, prev, cur));
                    prev = cur;

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.attn_dec,
                        layer.attn_dec_set,
                        &attn_push,
                        self.cfg.n_head as u32,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((5, prev, cur));
                    prev = cur;

                    self.cmd_dispatch_set(
                        cb,
                        &layer.wo_dec.0,
                        layer.wo_dec.1,
                        &layer.wo_dec.3,
                        layer.wo_dec.2,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((6, prev, cur));
                    prev = cur;

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rms_norm,
                        layer.ffn_norm_set,
                        &norm_push,
                        1,
                        1,
                        true,
                    )?;
                    let cur = mark(cb);
                    spans.push((7, prev, cur));
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
                    spans.push((8, prev, cur));
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
                    spans.push((9, prev, cur));
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
                spans.push((10, prev, cur));

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
                    Vec::with_capacity(self.cfg.n_layers * 9 + 2);
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

                    let qkv_i8 = matches!(layer.wq_batch.flavor, GemmFlavor::CoopMatI8)
                        || matches!(layer.wk_batch.flavor, GemmFlavor::CoopMatI8)
                        || matches!(layer.wv_batch.flavor, GemmFlavor::CoopMatI8);
                    self.cmd_quant_if_needed(
                        cb,
                        self.quant_xb_set,
                        b_u32,
                        self.cfg.n_embd,
                        qkv_i8,
                    )?;
                    self.cmd_gemm(cb, &layer.wq_batch, b_u32, false, false)?;
                    self.cmd_gemm(cb, &layer.wk_batch, b_u32, false, false)?;
                    self.cmd_gemm(cb, &layer.wv_batch, b_u32, false, true)?;
                    if layer.bq_set.is_some() || layer.bk_set.is_some() || layer.bv_set.is_some() {
                        if let Some(bqs) = layer.bq_set {
                            self.cmd_add_bias(cb, bqs, q_dim, b_u32, false)?;
                        }
                        if let Some(bks) = layer.bk_set {
                            self.cmd_add_bias(cb, bks, kv_dim, b_u32, false)?;
                        }
                        if let Some(bvs) = layer.bv_set {
                            self.cmd_add_bias(cb, bvs, kv_dim, b_u32, false)?;
                        }
                        Self::cmd_compute_barrier(d, cb);
                    }
                    let cur = mark_pp(cb);
                    pp_spans.push((2, prev, cur));
                    prev = cur;

                    if layer.q_norm_set.is_some() || layer.k_norm_set.is_some() {
                        if let Some(qns) = layer.q_norm_set {
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.qk_norm,
                                qns,
                                &qk_push,
                                self.cfg.n_head as u32,
                                b_u32,
                                false,
                            )?;
                        }
                        if let Some(kns) = layer.k_norm_set {
                            self.cmd_dispatch_set(
                                cb,
                                &self.pipes.qk_norm,
                                kns,
                                &qk_push,
                                self.cfg.n_head_kv as u32,
                                b_u32,
                                false,
                            )?;
                        }
                        Self::cmd_compute_barrier(d, cb);
                        let cur = mark_pp(cb);
                        pp_spans.push((3, prev, cur));
                        prev = cur;
                    }

                    let mut l_rope_push = rope_push;
                    if !self.cfg.layer_uses_rope(layer.idx) {
                        l_rope_push[3] = 0;
                    }
                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rope_batch,
                        layer.rope_batch_set,
                        &l_rope_push,
                        rope_groups,
                        b_u32,
                        true,
                    )?;
                    let cur = mark_pp(cb);
                    pp_spans.push((4, prev, cur));
                    prev = cur;

                    self.cmd_attn_batch(cb, layer.attn_batch_set, 0, b_u32)?;
                    let cur = mark_pp(cb);
                    pp_spans.push((5, prev, cur));
                    prev = cur;

                    self.cmd_quant_if_needed(
                        cb,
                        self.quant_xb_set,
                        b_u32,
                        q_dim,
                        matches!(layer.wo_batch.flavor, GemmFlavor::CoopMatI8),
                    )?;
                    self.cmd_gemm(cb, &layer.wo_batch, b_u32, true, true)?;
                    let cur = mark_pp(cb);
                    pp_spans.push((6, prev, cur));
                    prev = cur;

                    self.cmd_dispatch_set(
                        cb,
                        &self.pipes.rms_norm,
                        layer.ffn_norm_set,
                        &norm_push,
                        b_u32,
                        1,
                        true,
                    )?;
                    let cur = mark_pp(cb);
                    pp_spans.push((7, prev, cur));
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
                                b_u32.div_ceil(64),
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
                    pp_spans.push((8, prev, cur));
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
                    pp_spans.push((9, prev, cur));
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
                pp_spans.push((10, prev, cur));

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
                    "attn_qkv_gemm",
                    "qk_norm",
                    "rope_kv",
                    "flash_attn",
                    "attn_wo_gemm+res",
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
                    pp_bytes[2] += l.qkv_weight_bytes
                        + b_u64 * ((self.cfg.n_embd + q_dim + 2 * kv_dim) * 4) as u64;
                    pp_flops[2] += b_u64 * (2 * self.cfg.n_embd * (q_dim + 2 * kv_dim)) as u64;
                    if l.q_norm_set.is_some() || l.k_norm_set.is_some() {
                        pp_bytes[3] += b_u64 * ((q_dim + kv_dim) * 8) as u64;
                    }
                    pp_bytes[4] += b_u64 * ((q_dim + 2 * kv_dim) * 4 + 2 * kv_dim * 2) as u64;
                    pp_bytes[5] += b_u64 * ((2 * q_dim * 4 + 2 * kv_dim * 2) as u64);
                    pp_flops[5] += 4 * causal_pairs * (q_dim as u64);
                    pp_bytes[6] +=
                        l.wo_weight_bytes + b_u64 * ((q_dim + 2 * self.cfg.n_embd) * 4) as u64;
                    pp_flops[6] += b_u64 * (2 * q_dim * self.cfg.n_embd) as u64;
                    pp_bytes[7] += b_u64 * (self.cfg.n_embd * 8) as u64;
                    pp_bytes[8] += l.gate_up_weight_bytes
                        + b_u64 * ((self.cfg.n_embd + self.cfg.n_ff) * 4) as u64;
                    pp_flops[8] += b_u64 * (4 * self.cfg.n_embd * self.cfg.n_ff) as u64;
                    pp_bytes[9] += l.down_weight_bytes
                        + b_u64 * ((self.cfg.n_ff + 2 * self.cfg.n_embd) * 4) as u64;
                    pp_flops[9] += b_u64 * (2 * self.cfg.n_ff * self.cfg.n_embd) as u64;
                }
                pp_bytes[10] = self.out_mat.slice.len;
                pp_flops[10] = (2 * self.cfg.n_embd * self.cfg.vocab_size) as u64;

                for i in 0..N_STAGES {
                    if i == 3 && pp_bytes[i] == 0 {
                        continue;
                    }
                    prefill_chunk_stages.push(VkStageStat {
                        name: pp_stage_names[i],
                        gpu_us: pp_ns[i] * 1e-3,
                        bytes: pp_bytes[i],
                        flops: pp_flops[i],
                    });
                }
            }

            // SAFETY: GPU is idle after `submit_and_wait`.
            unsafe { d.destroy_query_pool(qp, None) };
        }

        let mut decode_stages = Vec::new();
        for i in 0..N_STAGES {
            if stage_gpu_us[i] > 0.0 || stage_bytes[i] > 0 {
                if i == 3 && stage_bytes[i] == 0 {
                    continue;
                }
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

impl Drop for LlamaVkRunner {
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
