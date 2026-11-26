//! Loop-Aware Vulkan GPU + Host CPU/NVMe Hybrid Inference Runner for `inkling2` & `inkling2_loop`
//! (`Thinking Machines Lab Inkling-2-Mini`, `ADR-015`).
//!
//! Supports BOTH:
//! 1. **100% Pure-GPU Execution**: When all MoE expert layers fit in Vulkan device memory (e.g.
//!    32 GB+ unified/discrete GPUs or smaller MoE checkpoints), the entire 48-pass looped forward
//!    graph (`embed`, `mix`, `control`, `ShortConv`, relative-bias + zero-sink GQA attention,
//!    indexed `Q8_0` `mul_mat_id` MoE experts, `SiTU-GLU`, and soft-capped `lm_head`) executes
//!    end-to-end on the Vulkan GPU in a single command-buffer submission per token!
//! 2. **Loop-Aware 3-Tier `VRAM + Host RAM + NVMe O_DIRECT` Hybrid Execution**: On 8 GB discrete
//!    GPUs (e.g. Radeon RX 9060 XT 8 GB + 16 GB Host RAM), `Inkling2VkRunner` places 100% of dense
//!    projections (`24` layers), `token_embd`, `output.weight` (`201k` vocab), all `48` KV +
//!    ShortConv states, AND as many `3×` looped core MoE layers (`blk.6..10`, `~5.04 GiB`) as fit
//!    into `DEVICE_LOCAL` VRAM via uncached `O_DIRECT` streaming (`0` page-cache bytes), while
//!    bridging only `9 KiB` (`[n_embd]` floats) through persistently mapped coherent memory for
//!    remaining host RAM / NVMe `O_DIRECT` MoE layers!

use std::sync::Arc;

use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{DirectTensorReader, GgufError, GgufFile, PageAlignedSlot};

use crate::VulkanError;
use crate::context::{Pipeline, Recorder, VkBuffer, VulkanContext};
use crate::kernels::{dtype_code, normalize_vk_mat_bytes};

const SPV_EMBED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv"));
const SPV_GEMV_RES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_res.spv"));
const SPV_SIGMOID_MUL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sigmoid_mul.spv"));
const SPV_ARGMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
const SPV_INKLING2_OPS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_ops.spv"));
const SPV_INKLING2_REL_ATTN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_rel_attn.spv"));
const SPV_INKLING2_MOE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_moe.spv"));

const LOOP_CTRL_Q: usize = 0;
const LOOP_CTRL_FFN: usize = 3;
const LOOP_CTRL_COUNT: usize = 4;
const LOOP_CTRL_NAMES: [&str; LOOP_CTRL_COUNT] = ["q", "k", "v", "ffn"];

/// Model hyperparameters for [`Inkling2VkRunner`].
#[derive(Debug, Clone)]
pub struct Inkling2VkConfig {
    pub arch: String,
    pub n_layers_phys: usize,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff_exp: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_weights_scale: f32,
    pub n_head_per_layer: Vec<usize>,
    pub n_head_kv_per_layer: Vec<usize>,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub n_swa: usize,
    pub is_swa: Vec<bool>,
    pub shortconv_l_cache: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub attn_res_block_size: usize,
    pub final_logit_softcapping: f32,
    pub n_relative: usize,
    pub n_relative_extent: usize,
    pub qk_clip: f32,
    pub relative_clip: f32,
    pub log_scale_floor: usize,
    pub log_scale_alpha: f32,
    pub vocab_padded: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub n_passes: usize,
    pub gate_rank: usize,
}

impl Inkling2VkConfig {
    #[inline]
    pub fn is_looped(&self) -> bool {
        self.n_passes > 1 && self.core_end > self.core_start
    }

    #[inline]
    pub fn layer_phys(&self, il: usize) -> usize {
        if il < self.n_layers_phys {
            il
        } else {
            self.core_start + (il - self.n_layers_phys) % (self.core_end - self.core_start)
        }
    }

    #[inline]
    pub fn n_head(&self, ip: usize) -> usize {
        self.n_head_per_layer[ip]
    }

    #[inline]
    pub fn n_head_kv(&self, ip: usize) -> usize {
        self.n_head_kv_per_layer[ip]
    }

    #[inline]
    pub fn q_dim(&self, ip: usize) -> usize {
        self.n_head(ip) * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self, ip: usize) -> usize {
        self.n_head_kv(ip) * self.head_dim
    }

    #[inline]
    pub fn rel_extent_for(&self, ip: usize) -> usize {
        if self.is_swa[ip] {
            self.n_swa
        } else {
            self.n_relative_extent
        }
    }

    #[inline]
    pub fn max_depth_blocks(&self) -> usize {
        self.n_layers_phys.div_ceil(self.attn_res_block_size) + 2
    }
}

struct VkDenseMat {
    buf: VkBuffer,
    dtype_code: u32,
    rows: usize,
    cols: usize,
    row_u16s: u32,
}

struct VkVramExperts {
    gate_up: VkBuffer,
    gate_up_row_u16s: u32,
    gate_up_expert_u16s: u32,
    down: VkBuffer,
    down_row_u16s: u32,
    down_expert_u16s: u32,
}

struct VkLoopLayer {
    msg_query_off: u32,
    ctrl_down: [VkDenseMat; LOOP_CTRL_COUNT],
    ctrl_gain: [VkDenseMat; LOOP_CTRL_COUNT],
    ctrl_forget: [VkDenseMat; LOOP_CTRL_COUNT],
}

struct VkInklingLayer {
    wq: VkDenseMat,
    wk: VkDenseMat,
    wv: VkDenseMat,
    wo: VkDenseMat,
    wqkv_gate: VkDenseMat,
    attn_sink_proj: VkDenseMat,
    attn_rel: VkDenseMat,
    attn_k_conv_proj: Option<VkDenseMat>,
    attn_v_conv_proj: Option<VkDenseMat>,
    ffn_gate_inp: VkDenseMat,
    q_clip_off: u32,
    rel_bias_off: u32,
    rel_clip_off: u32,
    k_conv_off: u32,
    v_conv_off: u32,
    attn_post_conv_off: u32,
    ffn_post_conv_off: u32,
    attn_norm_off: u32,
    q_inp_norm_off: u32,
    k_inp_norm_off: u32,
    v_inp_norm_off: u32,
    attn_res_query_off: Option<u32>,
    ffn_res_query_off: u32,
    ffn_norm_off: u32,
    exp_probs_b_off: Option<u32>,
    vram_experts: Option<VkVramExperts>,
}

/// Maximum number of tokens evaluated in lockstep per `forward_window_hybrid` pass
/// (covers `32`-token prompt prefill chunks and `DSpark` `block_size == 8` verification windows!).
pub const MAX_INKLING2_VK_WINDOW: usize = 128;
/// Maximum speculative rollback window with `save_conv_snapshots == true` (`DSpark` `block_size == 8`).
pub const MAX_INKLING2_SPEC_WINDOW: usize = 8;

/// Callback signature invoked for MoE layers offloaded to Host RAM / NVMe `O_DIRECT`:
/// `(ip, cur_normed, top_ids, top_weights, moe_out)`.
pub type HostMoeFn<'a> = dyn FnMut(usize, &[f32], &[usize], &[f32], &mut [f32]) + 'a;

/// Windowed callback invoked for MoE layers offloaded to Host RAM / NVMe `O_DIRECT` across `1..=8` tokens:
/// `(ip, window_len, cur_normed_win, top_ids_win, top_weights_win, moe_out_win)`.
pub type HostMoeWindowFn<'a> = dyn FnMut(usize, usize, &[f32], &[usize], &[f32], &mut [f32]) + 'a;

/// End-to-end / Hybrid Vulkan runner for `inkling2` & `inkling2_loop`.
pub struct Inkling2VkRunner {
    ctx: VulkanContext,
    pub cfg: Inkling2VkConfig,
    vram_expert_mask: Vec<bool>,
    vram_total_bytes: u64,
    weights_mode_label: String,
    f32_params: VkBuffer,
    tok_embd: VkDenseMat,
    tok_norm_off: u32,
    out_res_query_off: u32,
    out_norm_off: u32,
    output: VkDenseMat,
    layers: Vec<VkInklingLayer>,
    loop_layers: Vec<VkLoopLayer>,
    // Device state (`48` logical KV + ShortConv slots)
    k_cache: VkBuffer,
    v_cache: VkBuffer,
    slot_kv_vec4_bases: Vec<u32>,
    slot_kv_u32_bases: Vec<u32>,
    conv_hist: VkBuffer,
    slot_conv_bases: Vec<u32>,
    total_conv_f32s: usize,
    last_window_len: usize,
    // Multi-token window state & `O(1)` speculative `ShortConv` snapshot buffers
    win_state_buf: VkBuffer,
    per_tok_state_f32s: usize,
    conv_snapshots_buf: VkBuffer,
    dspark_taps_buf: VkBuffer,
    win_argmax_buf: VkBuffer,
    step_tok0_buf: VkBuffer,
    top_ids_tmp_buf: VkBuffer,
    top_weights_tmp_buf: VkBuffer,
    // Working device buffers
    step_buf: VkBuffer,
    cur_buf: VkBuffer,
    inp_sa_buf: VkBuffer,
    ffn_inp_buf: VkBuffer,
    prelude_buf: VkBuffer,
    bank_out_buf: VkBuffer,
    blocks_buf: VkBuffer,
    prelude_blocks_buf: VkBuffer,
    bank_buf: VkBuffer,
    inputs_buf: VkBuffer,
    q_in_buf: VkBuffer,
    k_in_buf: VkBuffer,
    v_in_buf: VkBuffer,
    messages_buf: VkBuffer,
    ctrl_in_buf: VkBuffer,
    ctrl_g_buf: VkBuffer,
    ctrl_gain_buf: VkBuffer,
    ctrl_forget_buf: VkBuffer,
    q_buf: VkBuffer,
    k_buf: VkBuffer,
    v_buf: VkBuffer,
    r_buf: VkBuffer,
    k_dyn_buf: VkBuffer,
    v_dyn_buf: VkBuffer,
    sinks_buf: VkBuffer,
    attn_vec_buf: VkBuffer,
    gate_buf: VkBuffer,
    wo_out_buf: VkBuffer,
    router_logits_buf: VkBuffer,
    exp_gate_up_buf: VkBuffer,
    exp_hidden_buf: VkBuffer,
    exp_down_buf: VkBuffer,
    logits_padded_buf: VkBuffer,
    logits_buf: VkBuffer,
    argmax_buf: VkBuffer,
    dummy_buf: VkBuffer,
    // Persistently mapped coherent host-GPU bridge buffers (`9 KiB` activation transfer!)
    ffn_normed_bridge: VkBuffer,
    top_ids_bridge: VkBuffer,
    top_weights_bridge: VkBuffer,
    moe_out_bridge: VkBuffer,
    logits_host: Vec<f32>,
    host_cur_scratch: Vec<f32>,
    host_ids_scratch: Vec<usize>,
    host_weights_scratch: Vec<f32>,
    host_moe_out_scratch: Vec<f32>,
    // Cached pipelines
    pipe_embed: Arc<Pipeline>,
    pipe_ops: Arc<Pipeline>,
    pipe_rel_attn: Arc<Pipeline>,
    pipe_sigmoid_mul: Arc<Pipeline>,
    pipe_moe_ops: Arc<Pipeline>,
    pipe_argmax: Arc<Pipeline>,
    ops_wg: u32,
    gemv_wg: u32,
    gemv_nr: u32,
    moe_wg: u32,
    moe_nr: u32,
}

fn read_u_or_arr(
    gguf: &GgufFile,
    key: &str,
    expected_len: usize,
) -> Result<Vec<usize>, VulkanError> {
    let mv = gguf
        .get_meta(key)
        .ok_or_else(|| GgufError::MissingMetadata {
            path: gguf.path().display().to_string(),
            key: key.to_string(),
        })?;
    if let Some(arr) = mv.as_usize_array() {
        if arr.len() != expected_len {
            return Err(VulkanError::InvalidDimensions(format!(
                "{key}: expected {expected_len} entries, got {}",
                arr.len()
            )));
        }
        Ok(arr)
    } else if let Some(single) = mv.as_usize() {
        Ok(vec![single; expected_len])
    } else {
        Err(VulkanError::InvalidDimensions(format!(
            "{key}: expected integer or integer[]"
        )))
    }
}

fn parse_vk_config(gguf: &GgufFile, max_ctx: usize) -> Result<Inkling2VkConfig, VulkanError> {
    let arch = gguf.req_str("general.architecture")?.to_string();
    if arch != "inkling2" && arch != "inkling2_loop" {
        return Err(VulkanError::UnsupportedArch(arch));
    }
    let pfx = arch.as_str();
    let n_layers_phys = gguf.req_usize(&format!("{pfx}.block_count"))?;
    let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
    let n_ff_exp = read_u_or_arr(
        gguf,
        &format!("{pfx}.expert_feed_forward_length"),
        n_layers_phys,
    )?[0];
    let n_expert = gguf.req_usize(&format!("{pfx}.expert_count"))?;
    let n_expert_used = read_u_or_arr(gguf, &format!("{pfx}.expert_used_count"), n_layers_phys)?[0];
    let expert_weights_scale = gguf
        .get_meta(&format!("{pfx}.expert_weights_scale"))
        .and_then(|v| v.as_f32())
        .unwrap_or(1.0);
    let n_head_per_layer =
        read_u_or_arr(gguf, &format!("{pfx}.attention.head_count"), n_layers_phys)?;
    let n_head_kv_per_layer = read_u_or_arr(
        gguf,
        &format!("{pfx}.attention.head_count_kv"),
        n_layers_phys,
    )?;
    let head_dim = gguf.req_usize(&format!("{pfx}.attention.key_length"))?;
    if !head_dim.is_multiple_of(4) || head_dim > 256 {
        return Err(VulkanError::InvalidDimensions(format!(
            "inkling2 head_dim {head_dim} must be a multiple of 4 and <= 256"
        )));
    }
    let rms_eps = gguf.req_f32(&format!("{pfx}.attention.layer_norm_rms_epsilon"))?;
    let n_swa = gguf.req_usize(&format!("{pfx}.attention.sliding_window"))?;
    let swa_key = format!("{pfx}.attention.sliding_window_pattern");
    let is_swa = gguf
        .get_meta(&swa_key)
        .and_then(|v| v.as_bool_array())
        .ok_or_else(|| GgufError::MissingMetadata {
            path: gguf.path().display().to_string(),
            key: swa_key,
        })?;
    let shortconv_l_cache = gguf.req_usize(&format!("{pfx}.shortconv.l_cache"))?;
    let situ_beta = gguf.req_f32(&format!("{pfx}.activation.situ_beta"))?;
    let situ_linear_beta = gguf.req_f32(&format!("{pfx}.activation.situ_linear_beta"))?;
    let attn_res_block_size = gguf.req_usize(&format!("{pfx}.attn_res.block_size"))?;
    let final_logit_softcapping = gguf
        .get_meta(&format!("{pfx}.final_logit_softcapping"))
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let n_relative = gguf.req_usize(&format!("{pfx}.attention.relative_length"))?;
    let n_relative_extent = gguf.req_usize(&format!("{pfx}.attention.relative_extent"))?;
    let qk_clip = gguf.req_f32(&format!("{pfx}.attention.qk_clip"))?;
    let relative_clip = gguf.req_f32(&format!("{pfx}.attention.relative_clip"))?;
    let log_scale_floor = gguf.req_usize(&format!("{pfx}.attention.log_scale_floor"))?;
    let log_scale_alpha = gguf.req_f32(&format!("{pfx}.attention.log_scale_alpha"))?;
    let vocab_padded = gguf.req_usize(&format!("{pfx}.vocab_size"))?;
    let vocab_size = gguf.req_usize(&format!("{pfx}.vocab_size_unpadded"))?;
    if n_expert > 256 {
        return Err(VulkanError::InvalidDimensions(format!(
            "inkling2 n_expert {n_expert} exceeds Vulkan router shared-memory cap 256"
        )));
    }

    let (core_start, core_end, n_passes, gate_rank, n_layers) = if arch == "inkling2_loop" {
        let cs = gguf.req_usize(&format!("{pfx}.loop.start"))?;
        let ce = gguf.req_usize(&format!("{pfx}.loop.end"))?;
        let np = gguf.req_usize(&format!("{pfx}.loop.count"))?;
        let gr = gguf.req_usize(&format!("{pfx}.loop.gate_rank"))?;
        (cs, ce, np, gr, n_layers_phys + (np - 1) * (ce - cs))
    } else {
        (0, 0, 1, 0, n_layers_phys)
    };

    Ok(Inkling2VkConfig {
        arch,
        n_layers_phys,
        n_layers,
        n_embd,
        n_ff_exp,
        n_expert,
        n_expert_used,
        expert_weights_scale,
        n_head_per_layer,
        n_head_kv_per_layer,
        head_dim,
        rms_eps,
        n_swa,
        is_swa,
        shortconv_l_cache,
        situ_beta,
        situ_linear_beta,
        attn_res_block_size,
        final_logit_softcapping,
        n_relative,
        n_relative_extent,
        qk_clip,
        relative_clip,
        log_scale_floor,
        log_scale_alpha,
        vocab_padded,
        vocab_size,
        max_ctx,
        core_start,
        core_end,
        n_passes,
        gate_rank,
    })
}

fn push_f32_param(
    gguf: &GgufFile,
    mut direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    dims: &[usize],
    pool: &mut Vec<f32>,
) -> Result<u32, VulkanError> {
    let off = pool.len() as u32;
    let n_elem: usize = dims.iter().product();
    pool.resize(pool.len() + n_elem, 0.0);
    let dst = &mut pool[off as usize..off as usize + n_elem];
    if let Some((rdr, ref mut slot)) = direct {
        let (info, f_off, b_len) = gguf.req_tensor_file_range(name, layer, dims)?;
        let mut raw = vec![0u8; b_len];
        rdr.read_exact_uncached(f_off, &mut raw, slot)?;
        dequantize_row_bytes(info.dtype, &raw, dst);
    } else {
        let (info, bytes) = gguf.req_tensor(name, layer, dims)?;
        dequantize_row_bytes(info.dtype, bytes, dst);
    }
    Ok(off)
}

#[allow(clippy::too_many_arguments)]
fn upload_dense_mat(
    ctx: &VulkanContext,
    gguf: &GgufFile,
    mut direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
    vram_acc: &mut u64,
) -> Result<VkDenseMat, VulkanError> {
    if let Some((rdr, ref mut slot)) = direct {
        let (info, f_off, b_len) = gguf.req_tensor_file_range(name, layer, &[cols, rows])?;
        let orig_dt = info.dtype;
        if matches!(
            orig_dt,
            DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K | DType::F32
        ) {
            let dt_code = dtype_code(orig_dt)?;
            let row_u16s = (orig_dt.row_bytes(cols) / 2) as u32;
            let buf = upload_range_uncached(ctx, rdr, f_off, b_len, slot)?;
            *vram_acc += b_len as u64;
            return Ok(VkDenseMat {
                buf,
                dtype_code: dt_code,
                rows,
                cols,
                row_u16s,
            });
        }
        let mut raw = vec![0u8; b_len];
        rdr.read_exact_uncached(f_off, &mut raw, slot)?;
        let (norm_dt, norm_bytes) = normalize_vk_mat_bytes(orig_dt, &raw, rows, cols)?;
        let dt_code = dtype_code(norm_dt)?;
        let row_u16s = (norm_dt.row_bytes(cols) / 2) as u32;
        let buf = ctx.upload_bytes(&norm_bytes)?;
        *vram_acc += norm_bytes.len() as u64;
        Ok(VkDenseMat {
            buf,
            dtype_code: dt_code,
            rows,
            cols,
            row_u16s,
        })
    } else {
        let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
        let (norm_dt, norm_bytes) = normalize_vk_mat_bytes(info.dtype, bytes, rows, cols)?;
        let dt_code = dtype_code(norm_dt)?;
        let row_u16s = (norm_dt.row_bytes(cols) / 2) as u32;
        let buf = ctx.upload_bytes(&norm_bytes)?;
        *vram_acc += norm_bytes.len() as u64;
        Ok(VkDenseMat {
            buf,
            dtype_code: dt_code,
            rows,
            cols,
            row_u16s,
        })
    }
}

fn upload_range_uncached(
    ctx: &VulkanContext,
    rdr: &DirectTensorReader,
    file_offset: u64,
    byte_len: usize,
    slot: &mut PageAlignedSlot,
) -> Result<VkBuffer, VulkanError> {
    let dst = ctx.create_device_slab(byte_len as u64)?;
    let max_chunk = 16 * 1024 * 1024usize;
    let mut done = 0usize;
    while done < byte_len {
        let take = (byte_len - done).min(max_chunk);
        let slice = rdr.read_into_slot(file_offset + done as u64, take, slot)?;
        ctx.write_bytes(&dst, done as u64, slice)?;
        done += take;
    }
    Ok(dst)
}

impl Inkling2VkRunner {
    /// Loads `Inkling2VkRunner` with automatic loop-aware VRAM expert placement (`ADR-015`).
    pub fn from_gguf(gguf: Arc<GgufFile>, max_ctx: usize) -> Result<Self, VulkanError> {
        Self::from_gguf_with_vram_budget(gguf, max_ctx, None)
    }

    /// Loads `Inkling2VkRunner` with an optional explicit VRAM budget override (in bytes).
    pub fn from_gguf_with_vram_budget(
        gguf: Arc<GgufFile>,
        max_ctx: usize,
        vram_budget_override: Option<u64>,
    ) -> Result<Self, VulkanError> {
        let ctx = VulkanContext::new()?;
        let info = ctx.info().clone();
        let cfg = parse_vk_config(&gguf, max_ctx)?;
        let n_phys = cfg.n_layers_phys;
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff_exp;
        let n_exp = cfg.n_expert;

        let mut total_file_bytes = 0u64;
        let mut dense_est_bytes = 0u64;
        let mut expert_layer_bytes = vec![0u64; n_phys];
        let mut max_single_expert_tensor = 0u64;
        for name in gguf.tensor_names() {
            if let Some(ti) = gguf.tensor_info(name) {
                total_file_bytes += ti.byte_len as u64;
                if let Some(rest) = name.strip_prefix("blk.")
                    && let Some((idx_s, suf)) = rest.split_once('.')
                    && let Ok(ip) = idx_s.parse::<usize>()
                    && suf.ends_with("_exps.weight")
                {
                    if ti.dtype != DType::Q8_0 {
                        return Err(VulkanError::UnsupportedDType(ti.dtype));
                    }
                    expert_layer_bytes[ip] += ti.byte_len as u64;
                    max_single_expert_tensor = max_single_expert_tensor.max(ti.byte_len as u64);
                } else {
                    dense_est_bytes += ti.byte_len as u64;
                }
            }
        }

        // State bytes in VRAM (`k_cache` + `v_cache` F16 + `conv_hist` F32 + scratch)
        let mut total_kv_u16s = 0usize;
        let mut total_conv_f32s = 0usize;
        let hist_len = cfg.shortconv_l_cache - 1;
        for il in 0..cfg.n_layers {
            let ip = cfg.layer_phys(il);
            let kv_dim = cfg.kv_dim(ip);
            total_kv_u16s += max_ctx * kv_dim;
            total_conv_f32s += (2 * kv_dim + 2 * n_embd) * hist_len;
        }
        let scratch_est =
            ((24 * n_embd + 4 * cfg.n_expert_used * n_ff + 2 * cfg.vocab_padded + 65536) * 4)
                as u64;
        let state_vram_bytes = (2 * total_kv_u16s * 2 + total_conv_f32s * 4) as u64 + scratch_est;

        let env_vram_mib = std::env::var("TM_VK_VRAM_MIB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        let headroom = if info.unified_memory {
            2048 * 1024 * 1024u64
        } else {
            680 * 1024 * 1024u64
        };
        let default_vram_budget = info.device_local_bytes.saturating_sub(headroom);
        let vram_budget = vram_budget_override
            .or_else(|| env_vram_mib.map(|m| m * 1024 * 1024))
            .unwrap_or(default_vram_budget);

        let base_needed = dense_est_bytes.saturating_add(state_vram_bytes);
        if base_needed > vram_budget {
            return Err(VulkanError::OutOfDeviceMemory {
                needed_mib: base_needed as f64 / 1048576.0,
                budget_mib: vram_budget as f64 / 1048576.0,
                max_ctx,
                device: info.name.clone(),
            });
        }

        // Decide which physical layers' MoE experts fit into VRAM:
        // On UMA devices (`unified_memory == true`), only place experts into `VkBuffer`s when the
        // whole model fits in `vram_budget` (otherwise host RAM + `O_DIRECT` avoids UMA duplication!).
        // On discrete GPUs (`RX 9060 XT`), greedily pack highest-reuse `3×` core layers (`core_start..core_end`)
        // first, then `1×` outer layers!
        let buf_cap = info.max_alloc_bytes.min(info.max_storage_buffer_range);
        let mut vram_expert_mask = vec![false; n_phys];
        let all_fit = total_file_bytes.saturating_add(state_vram_bytes) <= vram_budget
            && max_single_expert_tensor <= buf_cap;
        if all_fit {
            vram_expert_mask.fill(true);
        } else if !info.unified_memory && max_single_expert_tensor <= buf_cap {
            let mut rem = vram_budget.saturating_sub(base_needed);
            let mut order: Vec<usize> = (cfg.core_start..cfg.core_end).collect();
            order.extend(0..cfg.core_start);
            order.extend(cfg.core_end..n_phys);
            for ip in order {
                let need = expert_layer_bytes[ip];
                if need <= rem {
                    vram_expert_mask[ip] = true;
                    rem -= need;
                }
            }
        }

        // Use `DirectTensorReader` (`O_DIRECT`) whenever the model doesn't trivially fit in host RAM,
        // so uploading ~6.7 GiB to VRAM touches 0 bytes of Linux page cache!
        let use_direct = !all_fit && gguf.path().exists();
        let direct_reader = if use_direct {
            Some(DirectTensorReader::open(gguf.path())?)
        } else {
            None
        };
        let mut io_slot = PageAlignedSlot::new(16 * 1024 * 1024 + 8192);

        let mut vram_total_bytes = 0u64;
        let mut f32_pool: Vec<f32> = Vec::with_capacity(2 * 1024 * 1024);

        let tok_embd = upload_dense_mat(
            &ctx,
            &gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "token_embd.weight",
            "global",
            n_embd,
            cfg.vocab_padded,
            &mut vram_total_bytes,
        )?;
        let tok_norm_off = push_f32_param(
            &gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "token_embd_norm.weight",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;
        let out_res_query_off = push_f32_param(
            &gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output_res_query.weight",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;
        let out_norm_off = push_f32_param(
            &gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output_norm.weight",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;
        let output = upload_dense_mat(
            &ctx,
            &gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output.weight",
            "global",
            n_embd,
            cfg.vocab_padded,
            &mut vram_total_bytes,
        )?;

        let mut layers = Vec::with_capacity(n_phys);
        for (ip, &in_vram) in vram_expert_mask.iter().enumerate() {
            let ltag = format!("blk.{ip}");
            let swa = cfg.is_swa[ip];
            let q_dim = cfg.q_dim(ip);
            let kv_dim = cfg.kv_dim(ip);
            let nh = cfg.n_head(ip);
            let nh_kv = cfg.n_head_kv(ip);
            let hd = cfg.head_dim;
            let n_rel = cfg.n_relative;
            let n_conv = cfg.shortconv_l_cache;
            let extent = cfg.rel_extent_for(ip);

            let wq = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_q.weight"),
                &ltag,
                n_embd,
                q_dim,
                &mut vram_total_bytes,
            )?;
            let wk = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_k.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut vram_total_bytes,
            )?;
            let wv = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_v.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut vram_total_bytes,
            )?;
            let wo = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_output.weight"),
                &ltag,
                q_dim,
                n_embd,
                &mut vram_total_bytes,
            )?;
            let wqkv_gate = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_gate.weight"),
                &ltag,
                n_embd,
                q_dim,
                &mut vram_total_bytes,
            )?;
            let attn_sink_proj = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_sink_proj.weight"),
                &ltag,
                n_embd,
                nh,
                &mut vram_total_bytes,
            )?;
            let attn_rel = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel.weight"),
                &ltag,
                n_embd,
                n_rel * nh,
                &mut vram_total_bytes,
            )?;

            let q_clip_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_q_clip.weight"),
                &ltag,
                &[hd, nh],
                &mut f32_pool,
            )?;
            let rel_bias_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel_bias.weight"),
                &ltag,
                &[n_rel, extent],
                &mut f32_pool,
            )?;
            let rel_clip_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel_clip.weight"),
                &ltag,
                &[n_rel, nh],
                &mut f32_pool,
            )?;
            let k_conv_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_k_conv.weight"),
                &ltag,
                &[n_conv, kv_dim],
                &mut f32_pool,
            )?;
            let v_conv_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_v_conv.weight"),
                &ltag,
                &[n_conv, kv_dim],
                &mut f32_pool,
            )?;
            let attn_post_conv_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_post_conv.weight"),
                &ltag,
                &[n_conv, n_embd],
                &mut f32_pool,
            )?;
            let ffn_post_conv_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_post_conv.weight"),
                &ltag,
                &[n_conv, n_embd],
                &mut f32_pool,
            )?;

            let (
                attn_norm_off,
                q_inp_norm_off,
                k_inp_norm_off,
                v_inp_norm_off,
                attn_k_conv_proj,
                attn_v_conv_proj,
            ) = if swa {
                let an = push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_norm.weight"),
                    &ltag,
                    &[n_embd],
                    &mut f32_pool,
                )?;
                (an, 0, 0, 0, None, None)
            } else {
                let qn = push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_q_inp_norm.weight"),
                    &ltag,
                    &[n_embd],
                    &mut f32_pool,
                )?;
                let kn = push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_k_inp_norm.weight"),
                    &ltag,
                    &[n_embd],
                    &mut f32_pool,
                )?;
                let vn = push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_v_inp_norm.weight"),
                    &ltag,
                    &[n_embd],
                    &mut f32_pool,
                )?;
                let kp = upload_dense_mat(
                    &ctx,
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_k_conv_proj.weight"),
                    &ltag,
                    n_embd,
                    n_conv * nh_kv,
                    &mut vram_total_bytes,
                )?;
                let vp = upload_dense_mat(
                    &ctx,
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_v_conv_proj.weight"),
                    &ltag,
                    n_embd,
                    n_conv * nh_kv,
                    &mut vram_total_bytes,
                )?;
                (0, qn, kn, vn, Some(kp), Some(vp))
            };

            let attn_res_query_off = if ip > 0 {
                let dims: &[usize] = if swa { &[n_embd] } else { &[n_embd, 3] };
                Some(push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_res_query.weight"),
                    &ltag,
                    dims,
                    &mut f32_pool,
                )?)
            } else {
                None
            };
            let ffn_res_query_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_res_query.weight"),
                &ltag,
                &[n_embd],
                &mut f32_pool,
            )?;
            let ffn_norm_off = push_f32_param(
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_norm.weight"),
                &ltag,
                &[n_embd],
                &mut f32_pool,
            )?;
            let ffn_gate_inp = upload_dense_mat(
                &ctx,
                &gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_gate_inp.weight"),
                &ltag,
                n_embd,
                n_exp,
                &mut vram_total_bytes,
            )?;
            let exp_probs_b_off = if gguf
                .tensor_info(&format!("blk.{ip}.exp_probs_b.bias"))
                .is_some()
            {
                Some(push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.exp_probs_b.bias"),
                    &ltag,
                    &[n_exp],
                    &mut f32_pool,
                )?)
            } else {
                None
            };

            let vram_experts = if in_vram {
                let gu_row_u16s = (DType::Q8_0.row_bytes(n_embd) / 2) as u32;
                let gu_exp_u16s = gu_row_u16s * (2 * n_ff as u32);
                let dn_row_u16s = (DType::Q8_0.row_bytes(n_ff) / 2) as u32;
                let dn_exp_u16s = dn_row_u16s * (n_embd as u32);
                let (gate_up, down) = if let Some(ref rdr) = direct_reader {
                    let (_, gu_off, gu_len) = gguf.req_tensor_file_range(
                        &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                        &ltag,
                        &[n_embd, 2 * n_ff, n_exp],
                    )?;
                    let (_, dn_off, dn_len) = gguf.req_tensor_file_range(
                        &format!("blk.{ip}.ffn_down_exps.weight"),
                        &ltag,
                        &[n_ff, n_embd, n_exp],
                    )?;
                    let gu_b = upload_range_uncached(&ctx, rdr, gu_off, gu_len, &mut io_slot)?;
                    let dn_b = upload_range_uncached(&ctx, rdr, dn_off, dn_len, &mut io_slot)?;
                    vram_total_bytes += (gu_len + dn_len) as u64;
                    (gu_b, dn_b)
                } else {
                    let (_, gu_bytes) = gguf.req_tensor(
                        &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                        &ltag,
                        &[n_embd, 2 * n_ff, n_exp],
                    )?;
                    let (_, dn_bytes) = gguf.req_tensor(
                        &format!("blk.{ip}.ffn_down_exps.weight"),
                        &ltag,
                        &[n_ff, n_embd, n_exp],
                    )?;
                    let gu_b = ctx.upload_bytes(gu_bytes)?;
                    let dn_b = ctx.upload_bytes(dn_bytes)?;
                    vram_total_bytes += (gu_bytes.len() + dn_bytes.len()) as u64;
                    (gu_b, dn_b)
                };
                Some(VkVramExperts {
                    gate_up,
                    gate_up_row_u16s: gu_row_u16s,
                    gate_up_expert_u16s: gu_exp_u16s,
                    down,
                    down_row_u16s: dn_row_u16s,
                    down_expert_u16s: dn_exp_u16s,
                })
            } else {
                None
            };

            layers.push(VkInklingLayer {
                wq,
                wk,
                wv,
                wo,
                wqkv_gate,
                attn_sink_proj,
                attn_rel,
                attn_k_conv_proj,
                attn_v_conv_proj,
                ffn_gate_inp,
                q_clip_off,
                rel_bias_off,
                rel_clip_off,
                k_conv_off,
                v_conv_off,
                attn_post_conv_off,
                ffn_post_conv_off,
                attn_norm_off,
                q_inp_norm_off,
                k_inp_norm_off,
                v_inp_norm_off,
                attn_res_query_off,
                ffn_res_query_off,
                ffn_norm_off,
                exp_probs_b_off,
                vram_experts,
            });
        }

        let mut loop_layers = Vec::new();
        if cfg.is_looped() {
            let rank = cfg.gate_rank;
            for ip in cfg.core_start..cfg.core_end {
                let ltag = format!("blk.{ip}");
                let msg_query_off = push_f32_param(
                    &gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.loop_msg_query.weight"),
                    &ltag,
                    &[n_embd, LOOP_CTRL_COUNT],
                    &mut f32_pool,
                )?;
                let mut cd_vec = Vec::with_capacity(LOOP_CTRL_COUNT);
                let mut cg_vec = Vec::with_capacity(LOOP_CTRL_COUNT);
                let mut cf_vec = Vec::with_capacity(LOOP_CTRL_COUNT);
                for cname in LOOP_CTRL_NAMES {
                    cd_vec.push(upload_dense_mat(
                        &ctx,
                        &gguf,
                        direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_down.weight"),
                        &ltag,
                        2 * n_embd,
                        rank,
                        &mut vram_total_bytes,
                    )?);
                    cg_vec.push(upload_dense_mat(
                        &ctx,
                        &gguf,
                        direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_gain.weight"),
                        &ltag,
                        rank,
                        n_embd,
                        &mut vram_total_bytes,
                    )?);
                    cf_vec.push(upload_dense_mat(
                        &ctx,
                        &gguf,
                        direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_forget.weight"),
                        &ltag,
                        rank,
                        n_embd,
                        &mut vram_total_bytes,
                    )?);
                }
                let ctrl_down: [VkDenseMat; LOOP_CTRL_COUNT] = cd_vec
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("LOOP_CTRL_COUNT"));
                let ctrl_gain: [VkDenseMat; LOOP_CTRL_COUNT] = cg_vec
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("LOOP_CTRL_COUNT"));
                let ctrl_forget: [VkDenseMat; LOOP_CTRL_COUNT] = cf_vec
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("LOOP_CTRL_COUNT"));
                loop_layers.push(VkLoopLayer {
                    msg_query_off,
                    ctrl_down,
                    ctrl_gain,
                    ctrl_forget,
                });
            }
        }

        let f32_params = ctx.upload(&f32_pool)?;
        vram_total_bytes += (f32_pool.len() * 4) as u64;

        // Allocate 48-slot KV caches (`F16` = 2 bytes per element) + ShortConv histories (`F32`)
        let mut slot_kv_vec4_bases = Vec::with_capacity(cfg.n_layers);
        let mut slot_kv_u32_bases = Vec::with_capacity(cfg.n_layers);
        let mut slot_conv_bases = Vec::with_capacity(cfg.n_layers);
        let mut kv_u16_cursor = 0usize;
        let mut conv_f32_cursor = 0usize;
        for il in 0..cfg.n_layers {
            let ip = cfg.layer_phys(il);
            let kv_dim = cfg.kv_dim(ip);
            slot_kv_u32_bases.push((kv_u16_cursor / 2) as u32);
            slot_kv_vec4_bases.push((kv_u16_cursor / 4) as u32);
            kv_u16_cursor += max_ctx * kv_dim;
            slot_conv_bases.push(conv_f32_cursor as u32);
            conv_f32_cursor += (2 * kv_dim + 2 * n_embd) * hist_len;
        }
        let kv_bytes = (kv_u16_cursor * 2).max(16) as u64;
        let k_cache = ctx.alloc_buffer(kv_bytes)?;
        let v_cache = ctx.alloc_buffer(kv_bytes)?;
        let conv_hist = ctx.alloc_buffer((conv_f32_cursor * 4).max(16) as u64)?;
        vram_total_bytes += 2 * kv_bytes + (conv_f32_cursor * 4) as u64;

        let max_nh = cfg.n_head_per_layer.iter().copied().max().unwrap_or(1);
        let max_nh_kv = cfg.n_head_kv_per_layer.iter().copied().max().unwrap_or(1);
        let max_q_dim = max_nh * cfg.head_dim;
        let max_kv_dim = max_nh_kv * cfg.head_dim;
        let max_blocks = cfg.max_depth_blocks();
        let per_tok_state_f32s = (4 + 3 * max_blocks) * n_embd;

        let step_buf = ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * 4).max(16) as u64)?;
        let step_tok0_buf = ctx.alloc_buffer(16)?;
        let win_state_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * per_tok_state_f32s * 4).max(16) as u64)?;
        let conv_snapshots_buf =
            ctx.alloc_buffer((MAX_INKLING2_SPEC_WINDOW * conv_f32_cursor * 4).max(16) as u64)?;
        let dspark_taps_buf =
            ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * 8 * n_embd * 4).max(16) as u64)?;
        let win_argmax_buf = ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * 4).max(16) as u64)?;
        let top_ids_tmp_buf = ctx.alloc_buffer((cfg.n_expert_used * 4).max(16) as u64)?;
        let top_weights_tmp_buf = ctx.alloc_buffer((cfg.n_expert_used * 4).max(16) as u64)?;
        let cur_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let inp_sa_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let ffn_inp_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let prelude_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let bank_out_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let blocks_buf = ctx.alloc_buffer((max_blocks * n_embd * 4) as u64)?;
        let prelude_blocks_buf = ctx.alloc_buffer((max_blocks * n_embd * 4) as u64)?;
        let bank_buf = ctx.alloc_buffer((max_blocks * n_embd * 4) as u64)?;
        let inputs_buf = ctx.alloc_buffer((3 * n_embd * 4) as u64)?;
        let q_in_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let k_in_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let v_in_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let messages_buf = ctx.alloc_buffer((LOOP_CTRL_COUNT * n_embd * 4) as u64)?;
        let ctrl_in_buf = ctx.alloc_buffer((2 * n_embd * 4) as u64)?;
        let ctrl_g_buf = ctx.alloc_buffer((cfg.gate_rank.max(1) * 4) as u64)?;
        let ctrl_gain_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let ctrl_forget_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let q_buf = ctx.alloc_buffer((max_q_dim * 4) as u64)?;
        let k_buf = ctx.alloc_buffer((max_kv_dim * 4) as u64)?;
        let v_buf = ctx.alloc_buffer((max_kv_dim * 4) as u64)?;
        let r_buf = ctx.alloc_buffer((max_nh * cfg.n_relative * 4) as u64)?;
        let k_dyn_buf = ctx.alloc_buffer((cfg.shortconv_l_cache * max_nh_kv * 4) as u64)?;
        let v_dyn_buf = ctx.alloc_buffer((cfg.shortconv_l_cache * max_nh_kv * 4) as u64)?;
        let sinks_buf = ctx.alloc_buffer((max_nh * 4) as u64)?;
        let attn_vec_buf = ctx.alloc_buffer((max_q_dim * 4) as u64)?;
        let gate_buf = ctx.alloc_buffer((max_q_dim * 4) as u64)?;
        let wo_out_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let router_logits_buf = ctx.alloc_buffer((n_exp * 4) as u64)?;
        let exp_gate_up_buf = ctx.alloc_buffer((cfg.n_expert_used * 2 * n_ff * 4) as u64)?;
        let exp_hidden_buf = ctx.alloc_buffer((cfg.n_expert_used * n_ff * 4) as u64)?;
        let exp_down_buf = ctx.alloc_buffer((cfg.n_expert_used * n_embd * 4) as u64)?;
        let logits_padded_buf = ctx.alloc_buffer((cfg.vocab_padded * 4) as u64)?;
        let logits_buf = ctx.create_host_buffer((cfg.vocab_size * 4) as u64)?;
        let argmax_buf = ctx.create_host_buffer(16)?;
        let dummy_buf = ctx.alloc_buffer(64)?;

        let ffn_normed_bridge =
            ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let top_ids_bridge = ctx
            .create_host_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 4).max(16) as u64)?;
        let top_weights_bridge = ctx
            .create_host_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 4).max(16) as u64)?;
        let moe_out_bridge =
            ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;

        // Build compute pipelines
        let sg = info.subgroup_size;
        let max_wg = info.max_workgroup_invocations;
        let ops_wg = 128u32.max(sg).min(max_wg);
        let gemv_wg = 128u32.max(sg).min(max_wg);
        let gemv_nr = 4u32;
        let moe_wg = gemv_wg;
        let moe_nr = 4u32;
        let attn_sg_cnt = if 8 * sg <= max_wg { 8u32 } else { 4u32 };
        let attn_wg = attn_sg_cnt * sg;
        let attn_n_vec = ((cfg.head_dim / 4).div_ceil(sg as usize)).max(1) as u32;

        let pipe_embed = ctx.pipeline(
            "inkling2_embed",
            tok_embd.dtype_code,
            SPV_EMBED,
            3,
            8,
            &[64u32.max(sg), tok_embd.dtype_code, 0],
        )?;
        let pipe_ops = ctx.pipeline("inkling2_ops", ops_wg, SPV_INKLING2_OPS, 6, 32, &[ops_wg])?;
        let pipe_rel_attn = ctx.pipeline(
            "inkling2_rel_attn",
            attn_wg | (attn_n_vec << 16),
            SPV_INKLING2_REL_ATTN,
            7,
            28,
            &[attn_wg, attn_sg_cnt, attn_n_vec],
        )?;
        let pipe_sigmoid_mul =
            ctx.pipeline("inkling2_sig_mul", ops_wg, SPV_SIGMOID_MUL, 2, 4, &[ops_wg])?;
        let pipe_moe_ops = ctx.pipeline(
            "inkling2_moe",
            moe_wg | (moe_nr << 16),
            SPV_INKLING2_MOE,
            6,
            32,
            &[moe_wg, moe_nr],
        )?;
        let pipe_argmax = ctx.pipeline("inkling2_argmax", 256, SPV_ARGMAX, 2, 4, &[256])?;

        // Warm-cache GEMV pipelines for every encountered dtype (`Q8_0`, `F32`, etc.)
        for dt_code in [0u32, 1, 2, 3, 4, 5] {
            let _ = ctx.pipeline(
                "gemv_res",
                (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_wg, dt_code, gemv_nr],
            )?;
        }

        let n_vram_exps = vram_expert_mask.iter().filter(|&&b| b).count();
        let weights_mode_label = if n_vram_exps == n_phys {
            "vram-full".to_string()
        } else {
            format!("hybrid-vram{n_vram_exps}/{n_phys}exps")
        };

        Ok(Self {
            ctx,
            vram_expert_mask,
            vram_total_bytes,
            weights_mode_label,
            f32_params,
            tok_embd,
            tok_norm_off,
            out_res_query_off,
            out_norm_off,
            output,
            layers,
            loop_layers,
            k_cache,
            v_cache,
            slot_kv_vec4_bases,
            slot_kv_u32_bases,
            conv_hist,
            slot_conv_bases,
            total_conv_f32s: conv_f32_cursor,
            last_window_len: 0,
            win_state_buf,
            per_tok_state_f32s,
            conv_snapshots_buf,
            dspark_taps_buf,
            win_argmax_buf,
            step_tok0_buf,
            top_ids_tmp_buf,
            top_weights_tmp_buf,
            step_buf,
            cur_buf,
            inp_sa_buf,
            ffn_inp_buf,
            prelude_buf,
            bank_out_buf,
            blocks_buf,
            prelude_blocks_buf,
            bank_buf,
            inputs_buf,
            q_in_buf,
            k_in_buf,
            v_in_buf,
            messages_buf,
            ctrl_in_buf,
            ctrl_g_buf,
            ctrl_gain_buf,
            ctrl_forget_buf,
            q_buf,
            k_buf,
            v_buf,
            r_buf,
            k_dyn_buf,
            v_dyn_buf,
            sinks_buf,
            attn_vec_buf,
            gate_buf,
            wo_out_buf,
            router_logits_buf,
            exp_gate_up_buf,
            exp_hidden_buf,
            exp_down_buf,
            logits_padded_buf,
            logits_buf,
            argmax_buf,
            dummy_buf,
            ffn_normed_bridge,
            top_ids_bridge,
            top_weights_bridge,
            moe_out_bridge,
            logits_host: vec![0.0; cfg.vocab_size],
            host_cur_scratch: vec![0.0; MAX_INKLING2_VK_WINDOW * n_embd],
            host_ids_scratch: vec![0; MAX_INKLING2_VK_WINDOW * cfg.n_expert_used],
            host_weights_scratch: vec![0.0; MAX_INKLING2_VK_WINDOW * cfg.n_expert_used],
            host_moe_out_scratch: vec![0.0; MAX_INKLING2_VK_WINDOW * n_embd],
            pipe_embed,
            pipe_ops,
            pipe_rel_attn,
            pipe_sigmoid_mul,
            pipe_moe_ops,
            pipe_argmax,
            ops_wg,
            gemv_wg,
            gemv_nr,
            moe_wg,
            moe_nr,
            cfg,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.ctx.info().name
    }

    pub fn driver_info(&self) -> &str {
        &self.ctx.info().driver_info
    }

    pub fn weights_mode(&self) -> &str {
        &self.weights_mode_label
    }

    pub fn vram_expert_mask(&self) -> &[bool] {
        &self.vram_expert_mask
    }

    pub fn vram_total_bytes(&self) -> u64 {
        self.vram_total_bytes
    }

    pub fn has_host_offload_layers(&self) -> bool {
        self.vram_expert_mask.iter().any(|&v| !v)
    }

    /// Zeroes out all 48 KV + ShortConv states on the GPU.
    pub fn reset_state(&mut self) -> Result<(), VulkanError> {
        let k_buf = self.k_cache.buf;
        let v_buf = self.v_cache.buf;
        let c_buf = self.conv_hist.buf;
        self.ctx.run(|rec| {
            let d = &rec.ctx.inner.device;
            // SAFETY: Valid device buffers owned by `self`; `ctx.run` waits on fence before return.
            unsafe {
                d.cmd_fill_buffer(rec.cb, k_buf, 0, ash::vk::WHOLE_SIZE, 0);
                d.cmd_fill_buffer(rec.cb, v_buf, 0, ash::vk::WHOLE_SIZE, 0);
                d.cmd_fill_buffer(rec.cb, c_buf, 0, ash::vk::WHOLE_SIZE, 0);
            }
            Ok(())
        })
    }

    /// Pure-GPU single-token forward pass (requires `!self.has_host_offload_layers()`).
    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        self.forward_token_hybrid(token_id, pos, compute_logits, None)
    }

    /// Pure-GPU greedy token forward pass returning argmax token ID directly from GPU.
    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, VulkanError> {
        self.forward_token_greedy_hybrid(token_id, pos, None)
    }

    /// Hybrid GPU + Host RAM/NVMe single-token forward pass returning `[0..vocab_size)` logits.
    pub fn forward_token_hybrid(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        mut host_moe: Option<&mut HostMoeFn<'_>>,
    ) -> Result<&[f32], VulkanError> {
        self.run_token_internal(token_id, pos, compute_logits, false, &mut host_moe)?;
        if compute_logits {
            read_mapped_f32(&self.ctx, &self.logits_buf, &mut self.logits_host)?;
        }
        Ok(&self.logits_host)
    }

    /// Hybrid GPU + Host RAM/NVMe single-token greedy forward pass returning GPU `argmax` ID.
    pub fn forward_token_greedy_hybrid(
        &mut self,
        token_id: u32,
        pos: usize,
        mut host_moe: Option<&mut HostMoeFn<'_>>,
    ) -> Result<u32, VulkanError> {
        self.run_token_internal(token_id, pos, true, true, &mut host_moe)?;
        let mut id_u32 = [0u32; 1];
        read_mapped_u32(&self.argmax_buf, &mut id_u32)?;
        Ok(id_u32[0])
    }

    /// Batched / multi-token evaluation with optional hybrid host MoE evaluator.
    pub fn forward_batch_hybrid(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        mut host_moe: Option<&mut HostMoeFn<'_>>,
    ) -> Result<&[f32], VulkanError> {
        if token_ids.is_empty() {
            return Ok(&self.logits_host);
        }
        let last = token_ids.len() - 1;
        for (i, &tid) in token_ids.iter().enumerate() {
            let want = compute_last_logits && (i == last);
            self.run_token_internal(tid, start_pos + i, want, false, &mut host_moe)?;
        }
        if compute_last_logits {
            read_mapped_f32(&self.ctx, &self.logits_buf, &mut self.logits_host)?;
        }
        Ok(&self.logits_host)
    }

    /// Evaluates a window of `1 <= W <= MAX_INKLING2_VK_WINDOW` tokens in lockstep across all
    /// 48 visits (sharing GPU submissions and grouping unique Host RAM + NVMe `O_DIRECT` experts!),
    /// optionally snapshotting `ShortConv` states for `O(1)` speculative rollback and capturing
    /// `dspark_target_visits` (`e.g. &[5, 23, 41, 48]`).
    ///
    /// If `compute_last_logits` is true, populates `self.logits_host` with token `W - 1`'s logits.
    /// If `out_argmax` is `Some`, populates `out_argmax[0..W]` with GPU `argmax` token IDs for ALL `W` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_window_hybrid(
        &mut self,
        window_tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        save_conv_snapshots: bool,
        dspark_target_visits: &[usize],
        mut host_moe_win: Option<&mut HostMoeWindowFn<'_>>,
        out_argmax: Option<&mut [u32]>,
    ) -> Result<&[f32], VulkanError> {
        let w = window_tokens.len();
        if w == 0 || w > MAX_INKLING2_VK_WINDOW {
            return Err(VulkanError::InvalidDimensions(format!(
                "window length {w} outside 1..={MAX_INKLING2_VK_WINDOW}"
            )));
        }
        if save_conv_snapshots && w > MAX_INKLING2_SPEC_WINDOW {
            return Err(VulkanError::InvalidDimensions(format!(
                "speculative snapshot window {w} exceeds MAX_INKLING2_SPEC_WINDOW={MAX_INKLING2_SPEC_WINDOW}"
            )));
        }
        if dspark_target_visits.len() > 8 {
            return Err(VulkanError::InvalidDimensions(format!(
                "dspark_target_visits len {} exceeds cap 8",
                dspark_target_visits.len()
            )));
        }
        let end_pos = start_pos.saturating_add(w);
        if end_pos > self.cfg.max_ctx {
            return Err(VulkanError::InvalidDimensions(format!(
                "window end position {} exceeds max_ctx {}",
                end_pos - 1,
                self.cfg.max_ctx
            )));
        }
        for &tid in window_tokens {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(VulkanError::InvalidDimensions(format!(
                    "token_id {tid} exceeds vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let want_all_argmax = out_argmax.is_some();
        if let Some(ref slice) = out_argmax
            && slice.len() < w
        {
            return Err(VulkanError::InvalidDimensions(
                "out_argmax slice shorter than window_tokens".to_string(),
            ));
        }

        self.run_window_internal(
            window_tokens,
            start_pos,
            compute_last_logits,
            want_all_argmax,
            save_conv_snapshots,
            dspark_target_visits,
            &mut host_moe_win,
        )?;
        self.last_window_len = w;

        if let Some(dst_argmax) = out_argmax {
            read_mapped_u32(&self.win_argmax_buf, &mut dst_argmax[..w])?;
        }
        if compute_last_logits {
            read_mapped_f32(&self.ctx, &self.logits_buf, &mut self.logits_host)?;
        }
        Ok(&self.logits_host)
    }

    /// Commits `1 <= committed <= last_window_len` accepted tokens from the preceding
    /// `forward_window_hybrid(..., save_conv_snapshots = true, ...)` call in `O(1)` (`~0.015 ms`).
    pub fn commit_speculative_window(&mut self, committed: usize) -> Result<(), VulkanError> {
        if committed == 0 || committed > self.last_window_len {
            return Err(VulkanError::InvalidDimensions(format!(
                "commit_speculative_window({committed}) outside 1..={}",
                self.last_window_len
            )));
        }
        if committed < self.last_window_len && self.total_conv_f32s > 0 {
            let bytes = (self.total_conv_f32s * 4) as u64;
            let src_off = ((committed - 1) * self.total_conv_f32s * 4) as u64;
            let src_buf = self.conv_snapshots_buf.buf;
            let dst_buf = self.conv_hist.buf;
            self.ctx.run(|rec| {
                // SAFETY: Valid device buffers of sufficient size; `ctx.run` waits on fence.
                unsafe {
                    rec.ctx.inner.device.cmd_copy_buffer(
                        rec.cb,
                        src_buf,
                        dst_buf,
                        &[ash::vk::BufferCopy {
                            src_offset: src_off,
                            dst_offset: 0,
                            size: bytes,
                        }],
                    );
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Reads captured `DSpark` target visit taps (`count_tokens * n_taps * n_embd` floats)
    /// from the coherent host-visible tap buffer.
    pub fn read_dspark_taps(
        &self,
        count_tokens: usize,
        n_taps: usize,
        dst: &mut [f32],
    ) -> Result<(), VulkanError> {
        let need = count_tokens * n_taps * self.cfg.n_embd;
        if dst.len() < need {
            return Err(VulkanError::InvalidDimensions(
                "read_dspark_taps destination slice too small".to_string(),
            ));
        }
        read_mapped_f32(&self.ctx, &self.dspark_taps_buf, &mut dst[..need])
    }

    /// Computes `rms_norm(tok_embd[token_id], tok_norm, rms_eps)` on GPU VRAM (`9 KiB` readback)
    /// so `DFlashModel` (`DSpark`) borrows `tok_embd` directly from VRAM with 0 duplicate host RAM.
    pub fn embed_token_normed(
        &mut self,
        token_id: u32,
        out_normed: &mut [f32],
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        if out_normed.len() < n_embd {
            return Err(VulkanError::InvalidDimensions(
                "embed_token_normed slice shorter than n_embd".to_string(),
            ));
        }
        write_mapped_u32(&self.ctx, &self.step_buf, &[token_id])?;
        self.ctx.run(|rec| {
            self.record_embed_and_norm(rec)?;
            self.record_vec_copy(rec, &self.moe_out_bridge, 0, &self.cur_buf, 0, n_embd)
        })?;
        read_mapped_f32(&self.ctx, &self.moe_out_bridge, &mut out_normed[..n_embd])
    }

    /// Evaluates `(output * corr_hidden)[..vocab_size] -> softcap -> argmax` on GPU VRAM and
    /// returns the 4-byte greedy token ID (`< 1.5 ms` per draft token on RX 9060 XT!).
    pub fn project_dspark_greedy(
        &mut self,
        corr_hidden: &[f32],
        softcap: f32,
    ) -> Result<u32, VulkanError> {
        let n_embd = self.cfg.n_embd;
        if corr_hidden.len() != n_embd {
            return Err(VulkanError::InvalidDimensions(
                "project_dspark_greedy input length != n_embd".to_string(),
            ));
        }
        write_mapped_f32(&self.ctx, &self.moe_out_bridge, corr_hidden)?;
        self.ctx.run(|rec| {
            // Copy 9 KiB `corr_hidden` from host-coherent GTT bridge into `DEVICE_LOCAL` VRAM
            // (`wo_out_buf`) first so 201k LM-head subgroups hit RDNA4 Infinity Cache instead of PCIe!
            self.record_vec_copy(rec, &self.wo_out_buf, 0, &self.moe_out_bridge, 0, n_embd)?;
            self.dispatch_gemv(rec, &self.output, &self.wo_out_buf, &self.logits_padded_buf)?;
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.logits_buf,
                    &self.logits_padded_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[
                    6,
                    self.cfg.vocab_size as u32,
                    2,
                    0,
                    0,
                    softcap.to_bits(),
                    0,
                    0,
                ],
                self.elem_groups(self.cfg.vocab_size),
            )?;
            rec.dispatch(
                &self.pipe_argmax,
                &[&self.logits_buf, &self.argmax_buf],
                &[self.cfg.vocab_size as u32],
                1,
            )
        })?;
        let mut id = [0u32; 1];
        read_mapped_u32(&self.argmax_buf, &mut id)?;
        Ok(id[0])
    }

    fn gemv_pipe(&self, dt_code: u32) -> Result<Arc<Pipeline>, VulkanError> {
        self.ctx.pipeline(
            "gemv_res",
            (self.gemv_wg << 8) | (self.gemv_nr << 4) | dt_code,
            SPV_GEMV_RES,
            3,
            16,
            &[self.gemv_wg, dt_code, self.gemv_nr],
        )
    }

    #[inline]
    fn gemv_groups(&self, n_out: usize) -> u32 {
        let sg = self.ctx.info().subgroup_size;
        let rows_per_wg = ((self.gemv_wg / sg) * self.gemv_nr) as usize;
        n_out.div_ceil(rows_per_wg) as u32
    }

    #[inline]
    fn elem_groups(&self, n: usize) -> u32 {
        n.div_ceil(self.ops_wg as usize) as u32
    }

    fn dispatch_gemv(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkDenseMat,
        x: &VkBuffer,
        y: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let p = self.gemv_pipe(mat.dtype_code)?;
        rec.dispatch(
            &p,
            &[&mat.buf, x, y],
            &[mat.cols as u32, mat.rows as u32, 0, mat.row_u16s],
            self.gemv_groups(mat.rows),
        )
    }

    fn run_token_internal(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        compute_argmax: bool,
        host_moe: &mut Option<&mut HostMoeFn<'_>>,
    ) -> Result<(), VulkanError> {
        if pos >= self.cfg.max_ctx {
            return Err(VulkanError::InvalidDimensions(format!(
                "position {pos} exceeds configured max_ctx {}",
                self.cfg.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(VulkanError::InvalidDimensions(format!(
                "token_id {token_id} exceeds vocab_size {}",
                self.cfg.vocab_size
            )));
        }

        // Write `token_id` to mapped `step_buf`
        if let Some(p) = self.step_buf.mapped_ptr() {
            // SAFETY: `step_buf` is 16-byte persistently mapped coherent buffer, idle on GPU.
            unsafe {
                *(p as *mut u32) = token_id;
            }
        } else {
            self.ctx
                .write_bytes(&self.step_buf, 0, &token_id.to_le_bytes())?;
        }

        let ratio = ((pos + 1) as f32) / (self.cfg.log_scale_floor as f32);
        let tau = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();
        let n_embd = self.cfg.n_embd;

        // Build flat execution schedule of `(il, ip, pass, reset_to_prelude_before_this_step)`
        // so pure-GPU runs submit ALL 48 passes in ONE single `ctx.run` command buffer, while
        // hybrid runs batch consecutive GPU work between host-offloaded MoE layers!
        #[derive(Clone, Copy)]
        struct ExecStep {
            il: usize,
            ip: usize,
            pass: usize,
            save_prelude_before: bool,
            reset_pass_before: bool,
        }

        let mut schedule = Vec::with_capacity(self.cfg.n_layers);
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                schedule.push(ExecStep {
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
            }
        } else {
            let cs = self.cfg.core_start;
            let ce = self.cfg.core_end;
            for il in 0..cs {
                schedule.push(ExecStep {
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
            }
            for pass in 0..self.cfg.n_passes {
                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    schedule.push(ExecStep {
                        il,
                        ip,
                        pass,
                        save_prelude_before: pass == 0 && ip == cs,
                        reset_pass_before: pass > 0 && ip == cs,
                    });
                }
            }
            for il in ce..self.cfg.n_layers_phys {
                schedule.push(ExecStep {
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
            }
        }

        // Fast path: if NO layers require host MoE offload, record the ENTIRE token (embed + all 48
        // passes + output_norm + lm_head + softcap + argmax) in ONE single `ctx.run` submission!
        if !self.has_host_offload_layers() {
            let mut blocks_len = 0usize;
            let mut prelude_blocks_len = 0usize;
            let mut bank_len = 0usize;
            self.ctx.run(|rec| {
                self.record_embed_and_norm(rec)?;
                for st in &schedule {
                    self.record_pre_step_transitions(
                        rec,
                        st.save_prelude_before,
                        st.reset_pass_before,
                        &mut blocks_len,
                        &mut prelude_blocks_len,
                        &mut bank_len,
                    )?;
                    self.record_layer_pre_moe(
                        rec,
                        st.il,
                        st.ip,
                        pos,
                        tau,
                        st.pass > 0,
                        &mut blocks_len,
                        bank_len,
                    )?;
                    self.record_vram_moe(rec, st.ip)?;
                    self.record_layer_post_moe(rec, st.il, st.ip, true)?;
                }
                self.record_output_stage(rec, blocks_len, compute_logits, compute_argmax)
            })?;
            return Ok(());
        }

        // Hybrid 3-tier execution: batch consecutive GPU operations into single command-buffer
        // submissions between host-offloaded MoE layers (`blk.ip` where `vram_experts.is_none()`).
        let mut blocks_len = 0usize;
        let mut prelude_blocks_len = 0usize;
        let mut bank_len = 0usize;
        let mut cursor = 0usize;
        let mut need_embed = true;
        let mut pending_post_moe: Option<(usize, usize)> = None;

        while cursor < schedule.len() {
            let mut stop_for_host_ip: Option<(usize, usize)> = None;
            self.ctx.run(|rec| {
                if need_embed {
                    self.record_embed_and_norm(rec)?;
                    need_embed = false;
                }
                if let Some((prev_il, prev_ip)) = pending_post_moe.take() {
                    self.record_layer_post_moe(rec, prev_il, prev_ip, false)?;
                }
                while cursor < schedule.len() {
                    let st = schedule[cursor];
                    self.record_pre_step_transitions(
                        rec,
                        st.save_prelude_before,
                        st.reset_pass_before,
                        &mut blocks_len,
                        &mut prelude_blocks_len,
                        &mut bank_len,
                    )?;
                    self.record_layer_pre_moe(
                        rec,
                        st.il,
                        st.ip,
                        pos,
                        tau,
                        st.pass > 0,
                        &mut blocks_len,
                        bank_len,
                    )?;
                    cursor += 1;
                    if self.layers[st.ip].vram_experts.is_some() {
                        self.record_vram_moe(rec, st.ip)?;
                        self.record_layer_post_moe(rec, st.il, st.ip, true)?;
                    } else {
                        stop_for_host_ip = Some((st.il, st.ip));
                        break;
                    }
                }
                if stop_for_host_ip.is_none() {
                    self.record_output_stage(rec, blocks_len, compute_logits, compute_argmax)?;
                }
                Ok(())
            })?;

            if let Some((h_il, h_ip)) = stop_for_host_ip {
                let eval_fn = host_moe.as_deref_mut().ok_or_else(|| {
                    VulkanError::InvalidDimensions(format!(
                        "layer {h_ip} MoE experts are offloaded from VRAM, but no HostMoeFn was supplied"
                    ))
                })?;
                let top_k = self.cfg.n_expert_used;
                let mut raw_ids = [0u32; 32];
                read_mapped_f32(
                    &self.ctx,
                    &self.ffn_normed_bridge,
                    &mut self.host_cur_scratch[..n_embd],
                )?;
                read_mapped_u32(&self.top_ids_bridge, &mut raw_ids[..top_k])?;
                read_mapped_f32(
                    &self.ctx,
                    &self.top_weights_bridge,
                    &mut self.host_weights_scratch[..top_k],
                )?;
                for (dst_id, &raw_id) in self.host_ids_scratch[..top_k]
                    .iter_mut()
                    .zip(&raw_ids[..top_k])
                {
                    *dst_id = raw_id as usize;
                }
                eval_fn(
                    h_ip,
                    &self.host_cur_scratch[..n_embd],
                    &self.host_ids_scratch[..top_k],
                    &self.host_weights_scratch[..top_k],
                    &mut self.host_moe_out_scratch[..n_embd],
                );
                write_mapped_f32(
                    &self.ctx,
                    &self.moe_out_bridge,
                    &self.host_moe_out_scratch[..n_embd],
                )?;
                pending_post_moe = Some((h_il, h_ip));
            }
        }

        if let Some((last_il, last_ip)) = pending_post_moe.take() {
            self.ctx.run(|rec| {
                self.record_layer_post_moe(rec, last_il, last_ip, false)?;
                self.record_output_stage(rec, blocks_len, compute_logits, compute_argmax)
            })?;
        }

        Ok(())
    }

    fn record_embed_and_norm(&self, rec: &mut Recorder<'_>) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd as u32;
        let emb_wg = self.pipe_embed.wg_size;
        rec.dispatch(
            &self.pipe_embed,
            &[&self.tok_embd.buf, &self.step_buf, &self.inp_sa_buf],
            &[n_embd, self.tok_embd.row_u16s],
            n_embd.div_ceil(emb_wg),
        )?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.inp_sa_buf,
                &self.cur_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                0,
                n_embd,
                0,
                0,
                1,
                self.cfg.rms_eps.to_bits(),
                0,
                self.tok_norm_off,
            ],
            1,
        )
    }

    fn record_vec_copy(
        &self,
        rec: &mut Recorder<'_>,
        dst: &VkBuffer,
        dst_off: u32,
        src: &VkBuffer,
        src_off: u32,
        len: usize,
    ) -> Result<(), VulkanError> {
        if len == 0 {
            return Ok(());
        }
        rec.dispatch(
            &self.pipe_ops,
            &[
                dst,
                src,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[6, len as u32, 1, dst_off, src_off, 0, 0, 0],
            self.elem_groups(len),
        )
    }

    fn record_vec_add(
        &self,
        rec: &mut Recorder<'_>,
        dst: &VkBuffer,
        a: &VkBuffer,
        b: &VkBuffer,
        len: usize,
    ) -> Result<(), VulkanError> {
        rec.dispatch(
            &self.pipe_ops,
            &[dst, a, b, &self.dummy_buf, &self.dummy_buf, &self.dummy_buf],
            &[6, len as u32, 0, 0, 0, 0, 0, 0],
            self.elem_groups(len),
        )
    }

    fn record_pre_step_transitions(
        &self,
        rec: &mut Recorder<'_>,
        save_prelude: bool,
        reset_pass: bool,
        blocks_len: &mut usize,
        prelude_blocks_len: &mut usize,
        bank_len: &mut usize,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        if save_prelude {
            *prelude_blocks_len = *blocks_len;
            self.record_vec_copy(rec, &self.prelude_buf, 0, &self.cur_buf, 0, n_embd)?;
            self.record_vec_copy(
                rec,
                &self.prelude_blocks_buf,
                0,
                &self.blocks_buf,
                0,
                *prelude_blocks_len * n_embd,
            )?;
        }
        if reset_pass {
            let bank_start = *prelude_blocks_len + 1;
            *bank_len = blocks_len.saturating_sub(bank_start);
            if *bank_len > 0 {
                self.record_vec_copy(
                    rec,
                    &self.bank_buf,
                    0,
                    &self.blocks_buf,
                    (bank_start * n_embd) as u32,
                    *bank_len * n_embd,
                )?;
            }
            self.record_vec_copy(rec, &self.bank_out_buf, 0, &self.cur_buf, 0, n_embd)?;
            self.record_vec_copy(rec, &self.cur_buf, 0, &self.prelude_buf, 0, n_embd)?;
            *blocks_len = *prelude_blocks_len;
            self.record_vec_copy(
                rec,
                &self.blocks_buf,
                0,
                &self.prelude_blocks_buf,
                0,
                *prelude_blocks_len * n_embd,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_control(
        &self,
        rec: &mut Recorder<'_>,
        loop_w: &VkLoopLayer,
        ctrl: usize,
        cur_buf: &VkBuffer,
        cur_off: u32,
        msg_off: u32,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd as u32;
        // `ctrl_in[0..n_embd] = rms_norm_unit(cur, 1e-6)`
        rec.dispatch(
            &self.pipe_ops,
            &[
                cur_buf,
                &self.ctrl_in_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, n_embd, cur_off, 0, 0, 1e-6f32.to_bits(), 0, 0],
            1,
        )?;
        // `ctrl_in[n_embd..2*n_embd] = rms_norm_unit(msg, 1e-6)`
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.messages_buf,
                &self.ctrl_in_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, n_embd, msg_off, n_embd, 0, 1e-6f32.to_bits(), 0, 0],
            1,
        )?;
        self.dispatch_gemv(
            rec,
            &loop_w.ctrl_down[ctrl],
            &self.ctrl_in_buf,
            &self.ctrl_g_buf,
        )?;
        self.dispatch_gemv(
            rec,
            &loop_w.ctrl_gain[ctrl],
            &self.ctrl_g_buf,
            &self.ctrl_gain_buf,
        )?;
        self.dispatch_gemv(
            rec,
            &loop_w.ctrl_forget[ctrl],
            &self.ctrl_g_buf,
            &self.ctrl_forget_buf,
        )?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                cur_buf,
                &self.messages_buf,
                &self.ctrl_gain_buf,
                &self.ctrl_forget_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[4, n_embd, cur_off, msg_off, 0, 0, 0, 0],
            self.elem_groups(self.cfg.n_embd),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_layer_pre_moe(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        has_messages: bool,
        blocks_len: &mut usize,
        bank_len: usize,
    ) -> Result<(), VulkanError> {
        self.record_layer_pre_moe_into(
            rec,
            il,
            ip,
            pos,
            tau,
            has_messages,
            blocks_len,
            bank_len,
            &self.ffn_normed_bridge,
            &self.top_ids_bridge,
            &self.top_weights_bridge,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_layer_pre_moe_into(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        has_messages: bool,
        blocks_len: &mut usize,
        bank_len: usize,
        ffn_norm_dst: &VkBuffer,
        top_ids_dst: &VkBuffer,
        top_w_dst: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let swa = self.cfg.is_swa[ip];
        let nh = self.cfg.n_head(ip);
        let nh_kv = self.cfg.n_head_kv(ip);
        let hd = self.cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nh_kv * hd;
        let n_rel = self.cfg.n_relative;
        let n_conv = self.cfg.shortconv_l_cache;
        let extent = self.cfg.rel_extent_for(ip);
        let slot_conv_base = self.slot_conv_bases[il];
        let layer = &self.layers[ip];

        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.bank_buf,
                    &self.bank_out_buf,
                    &self.f32_params,
                    &self.messages_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[
                    3,
                    n_embd as u32,
                    bank_len as u32,
                    loop_w.msg_query_off,
                    0,
                    0,
                    0,
                    0,
                ],
                LOOP_CTRL_COUNT as u32,
            )?;
        }

        self.record_vec_copy(rec, &self.inp_sa_buf, 0, &self.cur_buf, 0, n_embd)?;

        if let Some(rq_off) = layer.attn_res_query_off {
            if swa {
                rec.dispatch(
                    &self.pipe_ops,
                    &[
                        &self.blocks_buf,
                        &self.cur_buf,
                        &self.f32_params,
                        &self.inputs_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[3, n_embd as u32, *blocks_len as u32, rq_off, 0, 0, 0, 0],
                    1,
                )?;
                self.record_vec_copy(
                    rec,
                    &self.inputs_buf,
                    n_embd as u32,
                    &self.inputs_buf,
                    0,
                    n_embd,
                )?;
                self.record_vec_copy(
                    rec,
                    &self.inputs_buf,
                    (2 * n_embd) as u32,
                    &self.inputs_buf,
                    0,
                    n_embd,
                )?;
            } else {
                rec.dispatch(
                    &self.pipe_ops,
                    &[
                        &self.blocks_buf,
                        &self.cur_buf,
                        &self.f32_params,
                        &self.inputs_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[3, n_embd as u32, *blocks_len as u32, rq_off, 0, 0, 0, 0],
                    3,
                )?;
            }
        } else {
            for j in 0..3 {
                self.record_vec_copy(
                    rec,
                    &self.inputs_buf,
                    (j * n_embd) as u32,
                    &self.cur_buf,
                    0,
                    n_embd,
                )?;
            }
        }

        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            for j in 0..3 {
                self.record_control(
                    rec,
                    loop_w,
                    LOOP_CTRL_Q + j,
                    &self.inputs_buf,
                    (j * n_embd) as u32,
                    (j * n_embd) as u32,
                )?;
            }
        }

        // Input RMSNorm into `q_in_buf`, `k_in_buf`, `v_in_buf`
        if swa && has_messages {
            for (j, dst_b) in [&self.q_in_buf, &self.k_in_buf, &self.v_in_buf]
                .into_iter()
                .enumerate()
            {
                rec.dispatch(
                    &self.pipe_ops,
                    &[
                        &self.inputs_buf,
                        dst_b,
                        &self.f32_params,
                        &self.dummy_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[
                        0,
                        n_embd as u32,
                        (j * n_embd) as u32,
                        0,
                        1,
                        self.cfg.rms_eps.to_bits(),
                        0,
                        layer.attn_norm_off,
                    ],
                    1,
                )?;
            }
        } else if swa {
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.inputs_buf,
                    &self.q_in_buf,
                    &self.f32_params,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[
                    0,
                    n_embd as u32,
                    0,
                    0,
                    1,
                    self.cfg.rms_eps.to_bits(),
                    0,
                    layer.attn_norm_off,
                ],
                1,
            )?;
            self.record_vec_copy(rec, &self.k_in_buf, 0, &self.q_in_buf, 0, n_embd)?;
            self.record_vec_copy(rec, &self.v_in_buf, 0, &self.q_in_buf, 0, n_embd)?;
        } else {
            let pairs = [
                (0usize, &self.q_in_buf, layer.q_inp_norm_off),
                (1usize, &self.k_in_buf, layer.k_inp_norm_off),
                (2usize, &self.v_in_buf, layer.v_inp_norm_off),
            ];
            for (j, dst_b, norm_off) in pairs {
                rec.dispatch(
                    &self.pipe_ops,
                    &[
                        &self.inputs_buf,
                        dst_b,
                        &self.f32_params,
                        &self.dummy_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[
                        0,
                        n_embd as u32,
                        (j * n_embd) as u32,
                        0,
                        1,
                        self.cfg.rms_eps.to_bits(),
                        0,
                        norm_off,
                    ],
                    1,
                )?;
            }
        }

        // Self-Attention projections
        self.dispatch_gemv(rec, &layer.wq, &self.q_in_buf, &self.q_buf)?;
        self.dispatch_gemv(rec, &layer.wk, &self.k_in_buf, &self.k_buf)?;
        self.dispatch_gemv(rec, &layer.wv, &self.v_in_buf, &self.v_buf)?;
        self.dispatch_gemv(rec, &layer.attn_rel, &self.q_in_buf, &self.r_buf)?;
        self.dispatch_gemv(rec, &layer.attn_sink_proj, &self.q_in_buf, &self.sinks_buf)?;
        self.dispatch_gemv(rec, &layer.wqkv_gate, &self.q_in_buf, &self.gate_buf)?;

        if !swa {
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            self.dispatch_gemv(rec, kp, &self.k_in_buf, &self.k_dyn_buf)?;
            self.dispatch_gemv(rec, vp, &self.v_in_buf, &self.v_dyn_buf)?;
        }

        let dyn_heads = if swa { 0u32 } else { nh_kv as u32 };
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.k_buf,
                &self.f32_params,
                &self.k_dyn_buf,
                &self.conv_hist,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                2,
                kv_dim as u32,
                n_conv as u32,
                0,
                dyn_heads,
                layer.k_conv_off,
                0,
                slot_conv_base,
            ],
            self.elem_groups(kv_dim),
        )?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.v_buf,
                &self.f32_params,
                &self.v_dyn_buf,
                &self.conv_hist,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                2,
                kv_dim as u32,
                n_conv as u32,
                kv_dim as u32,
                dyn_heads,
                layer.v_conv_off,
                0,
                slot_conv_base,
            ],
            self.elem_groups(kv_dim),
        )?;

        let tau_eff = if swa { 1.0f32 } else { tau };
        // Qcur clip + optional tau
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.q_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                1,
                hd as u32,
                1,
                layer.q_clip_off,
                0,
                self.cfg.qk_clip.to_bits(),
                tau_eff.to_bits(),
                0,
            ],
            nh as u32,
        )?;
        // Kcur clip
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.k_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                1,
                hd as u32,
                0,
                0,
                0,
                self.cfg.qk_clip.to_bits(),
                1.0f32.to_bits(),
                0,
            ],
            nh_kv as u32,
        )?;
        // Vcur per-head unit RMSNorm (`v_buf` -> `v_in_buf` scratch, then pack to F16 KV cache!)
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.v_buf,
                &self.v_in_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, hd as u32, 0, 0, 0, self.cfg.rms_eps.to_bits(), 0, 0],
            nh_kv as u32,
        )?;
        // Rcur clip + optional tau
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.r_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                1,
                n_rel as u32,
                1,
                layer.rel_clip_off,
                0,
                self.cfg.relative_clip.to_bits(),
                tau_eff.to_bits(),
                0,
            ],
            nh as u32,
        )?;

        // Pack `k_buf` and `v_in_buf` (unit-normed Vcur) into F16 `k_cache` & `v_cache` at `(il, pos)`
        let kv_pairs = kv_dim / 2;
        let dst_u32_off = self.slot_kv_u32_bases[il] + (pos * kv_pairs) as u32;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.k_buf,
                &self.v_in_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.k_cache,
                &self.v_cache,
            ],
            &[5, kv_pairs as u32, dst_u32_off, 0, 0, 0, 0, 0],
            self.elem_groups(kv_pairs),
        )?;

        // Relative-bias + zero-sink GQA attention
        let kv_start = if swa {
            (pos + 1).saturating_sub(self.cfg.n_swa)
        } else {
            0
        };
        rec.dispatch(
            &self.pipe_rel_attn,
            &[
                &self.q_buf,
                &self.r_buf,
                &self.f32_params,
                &self.sinks_buf,
                &self.k_cache,
                &self.v_cache,
                &self.attn_vec_buf,
            ],
            &[
                (nh as u32) | ((nh_kv as u32) << 16),
                (hd as u32) | ((n_rel as u32) << 16),
                extent as u32,
                pos as u32,
                kv_start as u32,
                self.slot_kv_vec4_bases[il],
                layer.rel_bias_off,
            ],
            nh as u32,
        )?;

        // Sigmoid output gate + `wo` + `attn_post_conv`
        rec.dispatch(
            &self.pipe_sigmoid_mul,
            &[&self.attn_vec_buf, &self.gate_buf],
            &[q_dim as u32],
            self.elem_groups(q_dim),
        )?;
        self.dispatch_gemv(rec, &layer.wo, &self.attn_vec_buf, &self.wo_out_buf)?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.wo_out_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.conv_hist,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                2,
                n_embd as u32,
                n_conv as u32,
                (2 * kv_dim) as u32,
                0,
                layer.attn_post_conv_off,
                0,
                slot_conv_base,
            ],
            self.elem_groups(n_embd),
        )?;

        // Block residual stream boundary (`ip % block_size == 0`)
        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            self.record_vec_copy(
                rec,
                &self.blocks_buf,
                (*blocks_len * n_embd) as u32,
                &self.inp_sa_buf,
                0,
                n_embd,
            )?;
            *blocks_len += 1;
            self.record_vec_copy(rec, &self.cur_buf, 0, &self.wo_out_buf, 0, n_embd)?;
        } else {
            self.record_vec_add(
                rec,
                &self.cur_buf,
                &self.inp_sa_buf,
                &self.wo_out_buf,
                n_embd,
            )?;
        }

        self.record_vec_copy(rec, &self.ffn_inp_buf, 0, &self.cur_buf, 0, n_embd)?;

        // FFN `mix` + optional `control(FFN)` + `ffn_norm` -> writes directly into `ffn_normed_bridge`!
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.blocks_buf,
                &self.cur_buf,
                &self.f32_params,
                &self.inp_sa_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                3,
                n_embd as u32,
                *blocks_len as u32,
                layer.ffn_res_query_off,
                0,
                0,
                0,
                0,
            ],
            1,
        )?;
        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            self.record_control(
                rec,
                loop_w,
                LOOP_CTRL_FFN,
                &self.inp_sa_buf,
                0,
                (LOOP_CTRL_FFN * n_embd) as u32,
            )?;
        }
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.inp_sa_buf,
                ffn_norm_dst,
                &self.f32_params,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                0,
                n_embd as u32,
                0,
                0,
                1,
                self.cfg.rms_eps.to_bits(),
                0,
                layer.ffn_norm_off,
            ],
            1,
        )?;

        // Router GEMV + GPU top-k selection
        self.dispatch_gemv(
            rec,
            &layer.ffn_gate_inp,
            ffn_norm_dst,
            &self.router_logits_buf,
        )?;
        let (has_b, b_off) = match layer.exp_probs_b_off {
            Some(off) => (1u32, off),
            None => (0u32, 0u32),
        };
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.router_logits_buf,
                &self.f32_params,
                top_ids_dst,
                top_w_dst,
                &self.dummy_buf,
            ],
            &[
                0,
                self.cfg.n_expert as u32,
                self.cfg.n_expert_used as u32,
                has_b,
                b_off,
                0,
                self.cfg.expert_weights_scale.to_bits(),
                0,
            ],
            1,
        )
    }

    fn record_vram_moe(&self, rec: &mut Recorder<'_>, ip: usize) -> Result<(), VulkanError> {
        let vexp = self.layers[ip]
            .vram_experts
            .as_ref()
            .expect("record_vram_moe requires vram_experts");
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        let top_k = self.cfg.n_expert_used;
        let sg = self.ctx.info().subgroup_size;
        let rows_per_wg = ((self.moe_wg / sg) * self.moe_nr) as usize;

        // 1. Indexed `gate_up` `Q8_0` GEMV (`[2 * n_ff]` rows x `top_k` experts)
        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &vexp.gate_up,
                &self.ffn_normed_bridge,
                &self.dummy_buf,
                &self.top_ids_bridge,
                &self.top_weights_bridge,
                &self.exp_gate_up_buf,
            ],
            &[
                1,
                n_embd as u32,
                (2 * n_ff) as u32,
                vexp.gate_up_row_u16s,
                vexp.gate_up_expert_u16s,
                1,
                0,
                0,
            ],
            (2 * n_ff).div_ceil(rows_per_wg) as u32,
            top_k as u32,
        )?;
        // 2. SiTU-GLU (`[top_k * n_ff]`)
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_gate_up_buf,
                &self.dummy_buf,
                &self.top_ids_bridge,
                &self.top_weights_bridge,
                &self.exp_hidden_buf,
            ],
            &[
                2,
                (top_k * n_ff) as u32,
                n_ff as u32,
                0,
                0,
                0,
                self.cfg.situ_beta.to_bits(),
                self.cfg.situ_linear_beta.to_bits(),
            ],
            (top_k * n_ff).div_ceil(self.moe_wg as usize) as u32,
        )?;
        // 3. Indexed `down` `Q8_0` GEMV (`[n_embd]` rows x `top_k` experts)
        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &vexp.down,
                &self.exp_hidden_buf,
                &self.dummy_buf,
                &self.top_ids_bridge,
                &self.top_weights_bridge,
                &self.exp_down_buf,
            ],
            &[
                1,
                n_ff as u32,
                n_embd as u32,
                vexp.down_row_u16s,
                vexp.down_expert_u16s,
                0,
                0,
                0,
            ],
            n_embd.div_ceil(rows_per_wg) as u32,
            top_k as u32,
        )?;
        // 4. Weighted reduction into `wo_out_buf`
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_down_buf,
                &self.dummy_buf,
                &self.top_ids_bridge,
                &self.top_weights_bridge,
                &self.wo_out_buf,
            ],
            &[3, n_embd as u32, top_k as u32, 0, 0, 0, 0, 0],
            n_embd.div_ceil(self.moe_wg as usize) as u32,
        )
    }

    fn record_layer_post_moe(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        moe_in_wo_out: bool,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let kv_dim = self.cfg.kv_dim(ip);
        let n_conv = self.cfg.shortconv_l_cache;
        let slot_conv_base = self.slot_conv_bases[il];
        let layer = &self.layers[ip];

        let moe_buf = if moe_in_wo_out {
            &self.wo_out_buf
        } else {
            &self.moe_out_bridge
        };
        rec.dispatch(
            &self.pipe_ops,
            &[
                moe_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.conv_hist,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                2,
                n_embd as u32,
                n_conv as u32,
                (2 * kv_dim + n_embd) as u32,
                0,
                layer.ffn_post_conv_off,
                0,
                slot_conv_base,
            ],
            self.elem_groups(n_embd),
        )?;
        self.record_vec_add(rec, &self.cur_buf, &self.ffn_inp_buf, moe_buf, n_embd)
    }

    fn record_output_stage(
        &self,
        rec: &mut Recorder<'_>,
        blocks_len: usize,
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.blocks_buf,
                &self.cur_buf,
                &self.f32_params,
                &self.inp_sa_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                3,
                n_embd as u32,
                blocks_len as u32,
                self.out_res_query_off,
                0,
                0,
                0,
                0,
            ],
            1,
        )?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.inp_sa_buf,
                &self.wo_out_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                0,
                n_embd as u32,
                0,
                0,
                1,
                self.cfg.rms_eps.to_bits(),
                0,
                self.out_norm_off,
            ],
            1,
        )?;
        if compute_logits {
            self.dispatch_gemv(rec, &self.output, &self.wo_out_buf, &self.logits_padded_buf)?;
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.logits_buf,
                    &self.logits_padded_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[
                    6,
                    self.cfg.vocab_size as u32,
                    2,
                    0,
                    0,
                    self.cfg.final_logit_softcapping.to_bits(),
                    0,
                    0,
                ],
                self.elem_groups(self.cfg.vocab_size),
            )?;
            if compute_argmax {
                rec.dispatch(
                    &self.pipe_argmax,
                    &[&self.logits_buf, &self.argmax_buf],
                    &[self.cfg.vocab_size as u32],
                    1,
                )?;
            }
        }
        Ok(())
    }

    fn record_buf_copy_bytes(
        &self,
        rec: &mut Recorder<'_>,
        dst: &VkBuffer,
        dst_byte_off: u64,
        src: &VkBuffer,
        src_byte_off: u64,
        byte_len: u64,
    ) {
        if byte_len == 0 {
            return;
        }
        let d = &rec.ctx.inner.device;
        // SAFETY: Valid buffers and ranges owned by `self`; explicit transfer->compute/transfer barrier follows.
        unsafe {
            d.cmd_copy_buffer(
                rec.cb,
                src.buf,
                dst.buf,
                &[ash::vk::BufferCopy {
                    src_offset: src_byte_off,
                    dst_offset: dst_byte_off,
                    size: byte_len,
                }],
            );
            let barrier = [ash::vk::MemoryBarrier::default()
                .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    ash::vk::AccessFlags::SHADER_READ
                        | ash::vk::AccessFlags::SHADER_WRITE
                        | ash::vk::AccessFlags::TRANSFER_READ,
                )];
            d.cmd_pipeline_barrier(
                rec.cb,
                ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::PipelineStageFlags::COMPUTE_SHADER | ash::vk::PipelineStageFlags::TRANSFER,
                ash::vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        }
    }

    fn record_load_tok_state(
        &self,
        rec: &mut Recorder<'_>,
        tok_i: usize,
        blocks_len: usize,
        prelude_blocks_len: usize,
        bank_len: usize,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let max_b = self.cfg.max_depth_blocks();
        let base = (tok_i * self.per_tok_state_f32s) as u32;
        self.record_vec_copy(rec, &self.cur_buf, 0, &self.win_state_buf, base, n_embd)?;
        self.record_vec_copy(
            rec,
            &self.ffn_inp_buf,
            0,
            &self.win_state_buf,
            base + n_embd as u32,
            n_embd,
        )?;
        self.record_vec_copy(
            rec,
            &self.prelude_buf,
            0,
            &self.win_state_buf,
            base + (2 * n_embd) as u32,
            n_embd,
        )?;
        self.record_vec_copy(
            rec,
            &self.bank_out_buf,
            0,
            &self.win_state_buf,
            base + (3 * n_embd) as u32,
            n_embd,
        )?;
        if blocks_len > 0 {
            self.record_vec_copy(
                rec,
                &self.blocks_buf,
                0,
                &self.win_state_buf,
                base + (4 * n_embd) as u32,
                blocks_len * n_embd,
            )?;
        }
        if prelude_blocks_len > 0 {
            self.record_vec_copy(
                rec,
                &self.prelude_blocks_buf,
                0,
                &self.win_state_buf,
                base + ((4 + max_b) * n_embd) as u32,
                prelude_blocks_len * n_embd,
            )?;
        }
        if bank_len > 0 {
            self.record_vec_copy(
                rec,
                &self.bank_buf,
                0,
                &self.win_state_buf,
                base + ((4 + 2 * max_b) * n_embd) as u32,
                bank_len * n_embd,
            )?;
        }
        Ok(())
    }

    fn record_save_tok_state(
        &self,
        rec: &mut Recorder<'_>,
        tok_i: usize,
        blocks_len: usize,
        prelude_blocks_len: usize,
        bank_len: usize,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let max_b = self.cfg.max_depth_blocks();
        let base = (tok_i * self.per_tok_state_f32s) as u32;
        self.record_vec_copy(rec, &self.win_state_buf, base, &self.cur_buf, 0, n_embd)?;
        self.record_vec_copy(
            rec,
            &self.win_state_buf,
            base + n_embd as u32,
            &self.ffn_inp_buf,
            0,
            n_embd,
        )?;
        self.record_vec_copy(
            rec,
            &self.win_state_buf,
            base + (2 * n_embd) as u32,
            &self.prelude_buf,
            0,
            n_embd,
        )?;
        self.record_vec_copy(
            rec,
            &self.win_state_buf,
            base + (3 * n_embd) as u32,
            &self.bank_out_buf,
            0,
            n_embd,
        )?;
        if blocks_len > 0 {
            self.record_vec_copy(
                rec,
                &self.win_state_buf,
                base + (4 * n_embd) as u32,
                &self.blocks_buf,
                0,
                blocks_len * n_embd,
            )?;
        }
        if prelude_blocks_len > 0 {
            self.record_vec_copy(
                rec,
                &self.win_state_buf,
                base + ((4 + max_b) * n_embd) as u32,
                &self.prelude_blocks_buf,
                0,
                prelude_blocks_len * n_embd,
            )?;
        }
        if bank_len > 0 {
            self.record_vec_copy(
                rec,
                &self.win_state_buf,
                base + ((4 + 2 * max_b) * n_embd) as u32,
                &self.bank_buf,
                0,
                bank_len * n_embd,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_window_internal(
        &mut self,
        window_tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        compute_all_argmax: bool,
        save_conv_snapshots: bool,
        dspark_target_visits: &[usize],
        host_moe_win: &mut Option<&mut HostMoeWindowFn<'_>>,
    ) -> Result<(), VulkanError> {
        let w = window_tokens.len();
        let n_embd = self.cfg.n_embd;
        let top_k = self.cfg.n_expert_used;
        let hist_len = self.cfg.shortconv_l_cache - 1;
        let n_taps = dspark_target_visits.len();

        write_mapped_u32(&self.ctx, &self.step_buf, window_tokens)?;

        #[derive(Clone, Copy)]
        struct ExecStep {
            visit: usize,
            il: usize,
            ip: usize,
            pass: usize,
            save_prelude_before: bool,
            reset_pass_before: bool,
        }

        let mut schedule = Vec::with_capacity(self.cfg.n_layers);
        let mut visit_ctr = 0usize;
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                schedule.push(ExecStep {
                    visit: visit_ctr,
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
                visit_ctr += 1;
            }
        } else {
            let cs = self.cfg.core_start;
            let ce = self.cfg.core_end;
            for il in 0..cs {
                schedule.push(ExecStep {
                    visit: visit_ctr,
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
                visit_ctr += 1;
            }
            for pass in 0..self.cfg.n_passes {
                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    schedule.push(ExecStep {
                        visit: visit_ctr,
                        il,
                        ip,
                        pass,
                        save_prelude_before: pass == 0 && ip == cs,
                        reset_pass_before: pass > 0 && ip == cs,
                    });
                    visit_ctr += 1;
                }
            }
            for il in ce..self.cfg.n_layers_phys {
                schedule.push(ExecStep {
                    visit: visit_ctr,
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
                visit_ctr += 1;
            }
        }

        let mut taus = [1.0f32; MAX_INKLING2_VK_WINDOW];
        for (tok_i, tau_slot) in taus.iter_mut().enumerate().take(w) {
            let ratio = ((start_pos + tok_i + 1) as f32) / (self.cfg.log_scale_floor as f32);
            *tau_slot = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();
        }

        let mut blocks_len = 0usize;
        let mut prelude_blocks_len = 0usize;
        let mut bank_len = 0usize;
        let mut cursor = 0usize;
        let mut need_embed = true;
        let mut pending_post_moe: Option<(usize, usize)> = None;

        while cursor < schedule.len() {
            let mut stop_for_host_ip: Option<(usize, usize)> = None;
            self.ctx.run(|rec| {
                if need_embed {
                    for tok_i in 0..w {
                        self.record_buf_copy_bytes(
                            rec,
                            &self.step_tok0_buf,
                            0,
                            &self.step_buf,
                            (tok_i * 4) as u64,
                            4,
                        );
                        let emb_wg = self.pipe_embed.wg_size;
                        rec.dispatch(
                            &self.pipe_embed,
                            &[&self.tok_embd.buf, &self.step_tok0_buf, &self.inp_sa_buf],
                            &[n_embd as u32, self.tok_embd.row_u16s],
                            (n_embd as u32).div_ceil(emb_wg),
                        )?;
                        rec.dispatch(
                            &self.pipe_ops,
                            &[
                                &self.inp_sa_buf,
                                &self.cur_buf,
                                &self.f32_params,
                                &self.dummy_buf,
                                &self.dummy_buf,
                                &self.dummy_buf,
                            ],
                            &[
                                0,
                                n_embd as u32,
                                0,
                                0,
                                1,
                                self.cfg.rms_eps.to_bits(),
                                0,
                                self.tok_norm_off,
                            ],
                            1,
                        )?;
                        let base = (tok_i * self.per_tok_state_f32s) as u32;
                        self.record_vec_copy(
                            rec,
                            &self.win_state_buf,
                            base,
                            &self.cur_buf,
                            0,
                            n_embd,
                        )?;
                    }
                    need_embed = false;
                }

                if let Some((prev_il, prev_ip)) = pending_post_moe.take() {
                    let prev_kv_dim = self.cfg.kv_dim(prev_ip);
                    let ffn_conv_off_f32 = self.slot_conv_bases[prev_il] as usize
                        + (2 * prev_kv_dim + n_embd) * hist_len;
                    let ffn_conv_len_f32 = n_embd * hist_len;
                    for tok_i in 0..w {
                        let base = (tok_i * self.per_tok_state_f32s) as u32;
                        self.record_vec_copy(
                            rec,
                            &self.ffn_inp_buf,
                            0,
                            &self.win_state_buf,
                            base + n_embd as u32,
                            n_embd,
                        )?;
                        self.record_vec_copy(
                            rec,
                            &self.wo_out_buf,
                            0,
                            &self.moe_out_bridge,
                            (tok_i * n_embd) as u32,
                            n_embd,
                        )?;
                        self.record_layer_post_moe(rec, prev_il, prev_ip, true)?;
                        if save_conv_snapshots && ffn_conv_len_f32 > 0 {
                            let snap_dst = tok_i * self.total_conv_f32s + ffn_conv_off_f32;
                            self.record_vec_copy(
                                rec,
                                &self.conv_snapshots_buf,
                                snap_dst as u32,
                                &self.conv_hist,
                                ffn_conv_off_f32 as u32,
                                ffn_conv_len_f32,
                            )?;
                        }
                        self.record_vec_copy(
                            rec,
                            &self.win_state_buf,
                            base,
                            &self.cur_buf,
                            0,
                            n_embd,
                        )?;
                    }
                }

                let max_visits_per_cb = (256 / w.max(1)).clamp(1, 64);
                let mut visits_in_cb = 0usize;
                while cursor < schedule.len() {
                    if visits_in_cb >= max_visits_per_cb {
                        break;
                    }
                    let st = schedule[cursor];
                    let in_vram = self.layers[st.ip].vram_experts.is_some();
                    let kv_dim = self.cfg.kv_dim(st.ip);
                    let attn_conv_off_f32 = self.slot_conv_bases[st.il] as usize;
                    let attn_conv_len_f32 = (2 * kv_dim + n_embd) * hist_len;
                    let ffn_conv_off_f32 = attn_conv_off_f32 + attn_conv_len_f32;
                    let ffn_conv_len_f32 = n_embd * hist_len;
                    let tap_slot = dspark_target_visits.iter().position(|&v| v == st.visit);

                    let mut next_b = blocks_len;
                    let mut next_pb = prelude_blocks_len;
                    let mut next_bk = bank_len;

                    for (tok_i, &tau_t) in taus.iter().enumerate().take(w) {
                        let mut b_t = blocks_len;
                        let mut pb_t = prelude_blocks_len;
                        let mut bk_t = bank_len;

                        self.record_load_tok_state(rec, tok_i, b_t, pb_t, bk_t)?;
                        self.record_pre_step_transitions(
                            rec,
                            st.save_prelude_before,
                            st.reset_pass_before,
                            &mut b_t,
                            &mut pb_t,
                            &mut bk_t,
                        )?;

                        if let Some(t_pos) = tap_slot {
                            self.record_vec_copy(
                                rec,
                                &self.dspark_taps_buf,
                                ((tok_i * n_taps + t_pos) * n_embd) as u32,
                                &self.cur_buf,
                                0,
                                n_embd,
                            )?;
                        }

                        if in_vram {
                            self.record_layer_pre_moe(
                                rec,
                                st.il,
                                st.ip,
                                start_pos + tok_i,
                                tau_t,
                                st.pass > 0,
                                &mut b_t,
                                bk_t,
                            )?;
                            if save_conv_snapshots && attn_conv_len_f32 > 0 {
                                let snap_dst = tok_i * self.total_conv_f32s + attn_conv_off_f32;
                                self.record_vec_copy(
                                    rec,
                                    &self.conv_snapshots_buf,
                                    snap_dst as u32,
                                    &self.conv_hist,
                                    attn_conv_off_f32 as u32,
                                    attn_conv_len_f32,
                                )?;
                            }
                            self.record_vram_moe(rec, st.ip)?;
                            self.record_layer_post_moe(rec, st.il, st.ip, true)?;
                            if save_conv_snapshots && ffn_conv_len_f32 > 0 {
                                let snap_dst = tok_i * self.total_conv_f32s + ffn_conv_off_f32;
                                self.record_vec_copy(
                                    rec,
                                    &self.conv_snapshots_buf,
                                    snap_dst as u32,
                                    &self.conv_hist,
                                    ffn_conv_off_f32 as u32,
                                    ffn_conv_len_f32,
                                )?;
                            }
                        } else {
                            self.record_layer_pre_moe_into(
                                rec,
                                st.il,
                                st.ip,
                                start_pos + tok_i,
                                tau_t,
                                st.pass > 0,
                                &mut b_t,
                                bk_t,
                                &self.wo_out_buf,
                                &self.top_ids_tmp_buf,
                                &self.top_weights_tmp_buf,
                            )?;
                            if save_conv_snapshots && attn_conv_len_f32 > 0 {
                                let snap_dst = tok_i * self.total_conv_f32s + attn_conv_off_f32;
                                self.record_vec_copy(
                                    rec,
                                    &self.conv_snapshots_buf,
                                    snap_dst as u32,
                                    &self.conv_hist,
                                    attn_conv_off_f32 as u32,
                                    attn_conv_len_f32,
                                )?;
                            }
                            self.record_vec_copy(
                                rec,
                                &self.ffn_normed_bridge,
                                (tok_i * n_embd) as u32,
                                &self.wo_out_buf,
                                0,
                                n_embd,
                            )?;
                            self.record_buf_copy_bytes(
                                rec,
                                &self.top_ids_bridge,
                                (tok_i * top_k * 4) as u64,
                                &self.top_ids_tmp_buf,
                                0,
                                (top_k * 4) as u64,
                            );
                            self.record_buf_copy_bytes(
                                rec,
                                &self.top_weights_bridge,
                                (tok_i * top_k * 4) as u64,
                                &self.top_weights_tmp_buf,
                                0,
                                (top_k * 4) as u64,
                            );
                        }

                        self.record_save_tok_state(rec, tok_i, b_t, pb_t, bk_t)?;
                        next_b = b_t;
                        next_pb = pb_t;
                        next_bk = bk_t;
                    }

                    blocks_len = next_b;
                    prelude_blocks_len = next_pb;
                    bank_len = next_bk;
                    cursor += 1;
                    visits_in_cb += 1;

                    if !in_vram {
                        stop_for_host_ip = Some((st.il, st.ip));
                        break;
                    }
                }

                if cursor == schedule.len() && stop_for_host_ip.is_none() {
                    self.record_window_outputs(
                        rec,
                        w,
                        blocks_len,
                        compute_last_logits,
                        compute_all_argmax,
                        dspark_target_visits,
                    )?;
                }
                Ok(())
            })?;

            if let Some((h_il, h_ip)) = stop_for_host_ip {
                let eval_win = host_moe_win.as_deref_mut().ok_or_else(|| {
                    VulkanError::InvalidDimensions(format!(
                        "layer {h_ip} MoE experts are offloaded from VRAM, but no HostMoeWindowFn was supplied"
                    ))
                })?;
                let mut raw_ids = [0u32; MAX_INKLING2_VK_WINDOW * 32];
                read_mapped_f32(
                    &self.ctx,
                    &self.ffn_normed_bridge,
                    &mut self.host_cur_scratch[..w * n_embd],
                )?;
                read_mapped_u32(&self.top_ids_bridge, &mut raw_ids[..w * top_k])?;
                read_mapped_f32(
                    &self.ctx,
                    &self.top_weights_bridge,
                    &mut self.host_weights_scratch[..w * top_k],
                )?;
                for (dst_id, &raw_id) in self.host_ids_scratch[..w * top_k]
                    .iter_mut()
                    .zip(&raw_ids[..w * top_k])
                {
                    *dst_id = raw_id as usize;
                }
                eval_win(
                    h_ip,
                    w,
                    &self.host_cur_scratch[..w * n_embd],
                    &self.host_ids_scratch[..w * top_k],
                    &self.host_weights_scratch[..w * top_k],
                    &mut self.host_moe_out_scratch[..w * n_embd],
                );
                write_mapped_f32(
                    &self.ctx,
                    &self.moe_out_bridge,
                    &self.host_moe_out_scratch[..w * n_embd],
                )?;
                pending_post_moe = Some((h_il, h_ip));
            }
        }

        if let Some((last_il, last_ip)) = pending_post_moe.take() {
            let prev_kv_dim = self.cfg.kv_dim(last_ip);
            let ffn_conv_off_f32 =
                self.slot_conv_bases[last_il] as usize + (2 * prev_kv_dim + n_embd) * hist_len;
            let ffn_conv_len_f32 = n_embd * hist_len;
            self.ctx.run(|rec| {
                for tok_i in 0..w {
                    let base = (tok_i * self.per_tok_state_f32s) as u32;
                    self.record_vec_copy(
                        rec,
                        &self.ffn_inp_buf,
                        0,
                        &self.win_state_buf,
                        base + n_embd as u32,
                        n_embd,
                    )?;
                    self.record_vec_copy(
                        rec,
                        &self.wo_out_buf,
                        0,
                        &self.moe_out_bridge,
                        (tok_i * n_embd) as u32,
                        n_embd,
                    )?;
                    self.record_layer_post_moe(rec, last_il, last_ip, true)?;
                    if save_conv_snapshots && ffn_conv_len_f32 > 0 {
                        let snap_dst = tok_i * self.total_conv_f32s + ffn_conv_off_f32;
                        self.record_vec_copy(
                            rec,
                            &self.conv_snapshots_buf,
                            snap_dst as u32,
                            &self.conv_hist,
                            ffn_conv_off_f32 as u32,
                            ffn_conv_len_f32,
                        )?;
                    }
                    self.record_vec_copy(rec, &self.win_state_buf, base, &self.cur_buf, 0, n_embd)?;
                }
                self.record_window_outputs(
                    rec,
                    w,
                    blocks_len,
                    compute_last_logits,
                    compute_all_argmax,
                    dspark_target_visits,
                )
            })?;
        }

        Ok(())
    }

    fn record_window_outputs(
        &self,
        rec: &mut Recorder<'_>,
        w: usize,
        blocks_len: usize,
        compute_last_logits: bool,
        compute_all_argmax: bool,
        dspark_target_visits: &[usize],
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let n_taps = dspark_target_visits.len();
        let final_tap = dspark_target_visits
            .iter()
            .position(|&v| v == self.cfg.n_layers);

        for tok_i in 0..w {
            let is_last = tok_i + 1 == w;
            let want_lg = compute_all_argmax || (compute_last_logits && is_last);
            let want_am = compute_all_argmax;
            if !want_lg && final_tap.is_none() {
                continue;
            }
            let base = (tok_i * self.per_tok_state_f32s) as u32;
            self.record_vec_copy(rec, &self.cur_buf, 0, &self.win_state_buf, base, n_embd)?;
            if blocks_len > 0 {
                self.record_vec_copy(
                    rec,
                    &self.blocks_buf,
                    0,
                    &self.win_state_buf,
                    base + (4 * n_embd) as u32,
                    blocks_len * n_embd,
                )?;
            }
            self.record_output_stage(rec, blocks_len, want_lg, want_am)?;
            if let Some(t_pos) = final_tap {
                self.record_vec_copy(
                    rec,
                    &self.dspark_taps_buf,
                    ((tok_i * n_taps + t_pos) * n_embd) as u32,
                    &self.wo_out_buf,
                    0,
                    n_embd,
                )?;
            }
            if want_am {
                self.record_buf_copy_bytes(
                    rec,
                    &self.win_argmax_buf,
                    (tok_i * 4) as u64,
                    &self.argmax_buf,
                    0,
                    4,
                );
            }
        }
        Ok(())
    }
}

fn write_mapped_u32(ctx: &VulkanContext, buf: &VkBuffer, src: &[u32]) -> Result<(), VulkanError> {
    if let Some(ptr) = buf.mapped_ptr() {
        // SAFETY: `buf` is persistently mapped coherent memory, idle before next `ctx.run`.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut u32, src.len());
        }
        Ok(())
    } else {
        // SAFETY: POD `u32` slice viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(src.as_ptr() as *const u8, core::mem::size_of_val(src))
        };
        ctx.write_bytes(buf, 0, bytes)
    }
}

fn read_mapped_f32(
    ctx: &VulkanContext,
    buf: &VkBuffer,
    dst: &mut [f32],
) -> Result<(), VulkanError> {
    if let Some(ptr) = buf.mapped_ptr() {
        // SAFETY: `buf` is persistently mapped `HOST_VISIBLE | HOST_COHERENT` and idle after
        // `ctx.run` waited on its fence; `dst.len() * 4 <= buf.byte_len()`.
        unsafe {
            core::ptr::copy_nonoverlapping(ptr as *const f32, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    } else {
        let v = ctx.read_f32(buf, dst.len())?;
        dst.copy_from_slice(&v);
        Ok(())
    }
}

fn read_mapped_u32(buf: &VkBuffer, dst: &mut [u32]) -> Result<(), VulkanError> {
    let ptr = buf
        .mapped_ptr()
        .ok_or(VulkanError::NoMemoryType("mapped u32 bridge"))?;
    // SAFETY: `buf` is persistently mapped coherent memory, idle after fence wait.
    unsafe {
        core::ptr::copy_nonoverlapping(ptr as *const u32, dst.as_mut_ptr(), dst.len());
    }
    Ok(())
}

fn write_mapped_f32(ctx: &VulkanContext, buf: &VkBuffer, src: &[f32]) -> Result<(), VulkanError> {
    if let Some(ptr) = buf.mapped_ptr() {
        // SAFETY: `buf` is persistently mapped coherent memory, idle before next `ctx.run`.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut f32, src.len());
        }
        Ok(())
    } else {
        // SAFETY: POD `f32` slice viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(src.as_ptr() as *const u8, core::mem::size_of_val(src))
        };
        ctx.write_bytes(buf, 0, bytes)
    }
}
