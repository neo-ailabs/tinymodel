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
use crate::context::{BakedCb, Pipeline, Recorder, VkBuffer, VkPrefillMode, VulkanContext};
use crate::kernels::{dtype_code, normalize_vk_mat_bytes};

const SPV_EMBED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv"));
const SPV_GEMV_RES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_res.spv"));
const SPV_GEMV_DP4A: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_dp4a.spv"));
const SPV_QUANTIZE_Q8_1: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quantize_q8_1.spv"));
const SPV_GEMM_TILED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_tiled.spv"));
const SPV_GEMM_COOPMAT_F16: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_coopmat_f16.spv"));
const SPV_SIGMOID_MUL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sigmoid_mul.spv"));
const SPV_ARGMAX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argmax.spv"));
const SPV_INKLING2_OPS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_ops.spv"));
const SPV_INKLING2_REL_ATTN: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_rel_attn.spv"));
const SPV_INKLING2_MOE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_moe.spv"));
const SPV_INKLING2_MOE_DP4A: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_moe_dp4a.spv"));
const SPV_INKLING2_MOE_GROUPED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_moe_grouped.spv"));
const SPV_INKLING2_MOE_GROUPED_CM: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/inkling2_moe_grouped_cm.spv"));

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

pub(crate) struct VkDenseMat {
    pub(crate) buf: VkBuffer,
    pub(crate) dtype_code: u32,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_u16s: u32,
}

struct VkVramExperts {
    gate_up: VkBuffer,
    gate_up_row_u16s: u32,
    gate_up_expert_u16s: u32,
    down: VkBuffer,
    down_row_u16s: u32,
    down_expert_u16s: u32,
}

struct VkImportedExperts {
    // Field drop order is safety-critical: `vexp` (`VkBuffer`s) MUST drop BEFORE `_gu_pin`/`_dn_pin`!
    vexp: VkVramExperts,
    _gu_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
    _dn_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
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
    host_imported_experts: Option<VkImportedExperts>,
}

/// Maximum number of tokens evaluated in lockstep per `forward_window_hybrid` pass
/// (covers `256`-token prompt prefill chunks and `DSpark` `block_size == 8` verification windows!).
pub const MAX_INKLING2_VK_WINDOW: usize = 256;
/// Maximum speculative rollback window with `save_conv_snapshots == true` (`DSpark` `block_size == 8`).
pub const MAX_INKLING2_SPEC_WINDOW: usize = 8;

/// Callback signature invoked for MoE layers offloaded to Host RAM / NVMe `O_DIRECT`:
/// `(ip, cur_normed, top_ids, top_weights, moe_out)`.
pub type HostMoeFn<'a> = dyn FnMut(usize, &[f32], &[usize], &[f32], &mut [f32]) + 'a;

/// Windowed callback invoked for MoE layers offloaded to Host RAM / NVMe `O_DIRECT` across `1..=8` tokens:
/// `(ip, window_len, cur_normed_win, top_ids_win, top_weights_win, moe_out_win)`.
pub type HostMoeWindowFn<'a> = dyn FnMut(usize, usize, &[f32], &[usize], &[f32], &mut [f32]) + 'a;

const N_PROF_STAGES: usize = 7;
const PROF_STAGE_NAMES: [&str; N_PROF_STAGES] = [
    "embed_state",
    "loop_mix_norm",
    "attn_dense",
    "ffn_router",
    "vram_moe",
    "post_moe",
    "lm_head",
];
const STAGE_EMBED_STATE: usize = 0;
const STAGE_LOOP_MIX_NORM: usize = 1;
const STAGE_ATTN_DENSE: usize = 2;
const STAGE_FFN_ROUTER: usize = 3;
const STAGE_VRAM_MOE: usize = 4;
const STAGE_POST_MOE: usize = 5;
const STAGE_LM_HEAD: usize = 6;
const MAX_PROF_QUERIES: u32 = 8192;

/// Wall-clock + hardware timestamp phase accumulators for `TM_VK_PROF=1`.
#[derive(Debug, Default, Clone, Copy)]
struct Inkling2Prof {
    tokens: u64,
    gpu_submits: u64,
    /// CPU time spent recording command buffers (dispatch + descriptor-set allocation).
    gpu_record_ns: u64,
    /// `vkQueueSubmit` + GPU execution + `vkWaitForFences` (submit wall minus record).
    gpu_wait_ns: u64,
    /// Total GPU hardware execution span (`BOTTOM_OF_PIPE - TOP_OF_PIPE` via `vkCmdWriteTimestamp`).
    gpu_hw_ns: u64,
    /// Per-stage GPU hardware timestamps (`PROF_STAGE_NAMES`).
    stage_gpu_ns: [u64; N_PROF_STAGES],
    host_moe_calls: u64,
    host_moe_ns: u64,
    /// Mapped-memory bridge reads/writes around host MoE.
    bridge_ns: u64,
}

struct TsRecorder<'a> {
    qp: ash::vk::QueryPool,
    next_q: u32,
    prev_q: u32,
    spans: &'a mut Vec<(usize, u32, u32)>,
}

impl<'a> TsRecorder<'a> {
    fn begin(
        rec: &Recorder<'_>,
        qp: Option<ash::vk::QueryPool>,
        spans: &'a mut Vec<(usize, u32, u32)>,
    ) -> Option<Self> {
        let qp = qp?;
        spans.clear();
        let d = &rec.ctx.inner.device;
        // SAFETY: `qp` has `MAX_PROF_QUERIES` slots and `rec.cb` is actively recording.
        unsafe {
            d.cmd_reset_query_pool(rec.cb, qp, 0, MAX_PROF_QUERIES);
            d.cmd_write_timestamp(rec.cb, ash::vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
        }
        Some(Self {
            qp,
            next_q: 1,
            prev_q: 0,
            spans,
        })
    }

    #[inline]
    fn mark(opt: &mut Option<Self>, rec: &Recorder<'_>, stage: usize) {
        if let Some(t) = opt.as_mut()
            && t.next_q < MAX_PROF_QUERIES
        {
            let cur = t.next_q;
            t.next_q += 1;
            // SAFETY: `cur < MAX_PROF_QUERIES` and `rec.cb` is actively recording.
            unsafe {
                rec.ctx.inner.device.cmd_write_timestamp(
                    rec.cb,
                    ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    t.qp,
                    cur,
                );
            }
            t.spans.push((stage, t.prev_q, cur));
            t.prev_q = cur;
        }
    }
}

#[derive(Clone, Copy)]
struct InklingExecStep {
    visit: usize,
    il: usize,
    ip: usize,
    pass: usize,
    save_prelude_before: bool,
    reset_pass_before: bool,
}

#[derive(Clone, Copy)]
struct GpuSegSpec {
    need_embed: bool,
    prev_host: Option<(usize, usize)>,
    start_cur: usize,
    end_cur: usize,
    stop_host: Option<(usize, usize)>,
    run_outputs: bool,
    init_b: usize,
    init_pb: usize,
    init_bk: usize,
    end_b: usize,
}

/// End-to-end / Hybrid Vulkan runner for `inkling2` & `inkling2_loop`.
pub struct Inkling2VkRunner {
    pub(crate) ctx: VulkanContext,
    pub cfg: Inkling2VkConfig,
    schedule: Vec<InklingExecStep>,
    prof_on: bool,
    prof: Inkling2Prof,
    prof_qp: Option<ash::vk::QueryPool>,
    prof_ts_buf: Vec<u64>,
    prof_spans_slots: [Vec<(usize, u32, u32)>; 2],
    vram_expert_mask: Vec<bool>,
    vram_total_bytes: u64,
    weights_mode_label: String,
    f32_params: VkBuffer,
    pub(crate) tok_embd: VkDenseMat,
    tok_norm_off: u32,
    out_res_query_off: u32,
    out_norm_off: u32,
    pub(crate) output: VkDenseMat,
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
    pub(crate) dspark_taps_buf: VkBuffer,
    win_argmax_buf: VkBuffer,
    step_tok0_buf: VkBuffer,
    top_ids_tmp_buf: VkBuffer,
    top_weights_tmp_buf: VkBuffer,
    // Working device buffers
    pub(crate) step_buf: VkBuffer,
    dyn_pos_buf: VkBuffer,
    pub(crate) cur_buf: VkBuffer,
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
    ctrl_in_bufs: [VkBuffer; 3],
    ctrl_g_bufs: [VkBuffer; 3],
    ctrl_gain_bufs: [VkBuffer; 3],
    ctrl_forget_bufs: [VkBuffer; 3],
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
    // Batched `w`-token prefill `DEVICE_LOCAL` buffers (`CoopMat`/`gemm_tiled` + grouped MoE)
    win_taus_buf: VkBuffer,
    win_inputs_buf: VkBuffer,
    win_messages_buf: VkBuffer,
    win_ctrl_in_bufs: [VkBuffer; 3],
    win_ctrl_g_bufs: [VkBuffer; 3],
    win_ctrl_gain_bufs: [VkBuffer; 3],
    win_ctrl_forget_bufs: [VkBuffer; 3],
    win_q_in_buf: VkBuffer,
    win_k_in_buf: VkBuffer,
    win_v_in_buf: VkBuffer,
    win_q_buf: VkBuffer,
    win_k_buf: VkBuffer,
    win_v_buf: VkBuffer,
    win_v_norm_buf: VkBuffer,
    win_r_buf: VkBuffer,
    win_sinks_buf: VkBuffer,
    win_gate_buf: VkBuffer,
    win_k_dyn_buf: VkBuffer,
    win_v_dyn_buf: VkBuffer,
    win_attn_vec_buf: VkBuffer,
    win_wo_out_buf: VkBuffer,
    win_ffn_norm_buf: VkBuffer,
    win_router_logits_buf: VkBuffer,
    win_top_ids_dev_buf: VkBuffer,
    win_top_weights_dev_buf: VkBuffer,
    win_moe_out_dev_buf: VkBuffer,
    moe_tiles_buf: VkBuffer,
    exp_gate_up_buf: VkBuffer,
    exp_hidden_buf: VkBuffer,
    exp_down_buf: VkBuffer,
    pub(crate) logits_padded_buf: VkBuffer,
    logits_buf: VkBuffer,
    win_logits_padded_buf: VkBuffer,
    win_logits_dev_buf: VkBuffer,
    argmax_buf: VkBuffer,
    pub(crate) dummy_buf: VkBuffer,
    xq_buf: VkBuffer,
    xd_buf: VkBuffer,
    ext_xq_buf: VkBuffer,
    ext_xd_buf: VkBuffer,
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
    pipe_embed_batch: Arc<Pipeline>,
    pipe_ops: Arc<Pipeline>,
    pipe_rel_attn: Arc<Pipeline>,
    pipe_sigmoid_mul: Arc<Pipeline>,
    pipe_moe_ops: Arc<Pipeline>,
    pipe_moe_dp4a: Option<Arc<Pipeline>>,
    pipe_quant_q8_1: Option<Arc<Pipeline>>,
    pipe_moe_grouped: Option<Arc<Pipeline>>,
    moe_grouped_bm: usize,
    moe_grouped_bn: usize,
    pipe_argmax: Arc<Pipeline>,
    gemv_pipes: [Arc<Pipeline>; 6],
    gemv_small_pipes: [Arc<Pipeline>; 6],
    gemv_m4_pipes: [Arc<Pipeline>; 6],
    gemv_m8_pipes: [Arc<Pipeline>; 6],
    gemv_small_m4_pipes: [Arc<Pipeline>; 6],
    gemv_small_m8_pipes: [Arc<Pipeline>; 6],
    gemv_dp4a_pipes: Option<[Arc<Pipeline>; 4]>,
    gemv_dp4a_small_pipes: Option<[Arc<Pipeline>; 4]>,
    gemv_dp4a_m4_pipes: Option<[Arc<Pipeline>; 4]>,
    gemv_dp4a_m8_pipes: Option<[Arc<Pipeline>; 4]>,
    gemv_dp4a_small_m4_pipes: Option<[Arc<Pipeline>; 4]>,
    gemv_dp4a_small_m8_pipes: Option<[Arc<Pipeline>; 4]>,
    dp4a_min_dim: usize,
    dp4a_q8_enabled: bool,
    gemm_pipes: [Option<Arc<Pipeline>>; 6],
    gemm_tile_mn: (usize, usize),
    ops_wg: u32,
    gemv_wg: u32,
    gemv_nr: u32,
    gemv_small_wg: u32,
    moe_wg: u32,
    moe_nr: u32,
    hybrid_gpu_exps: usize,
    baked_decode_cbs: [Option<Vec<BakedCb>>; 3],
    baked_ext_moe_cbs: Vec<Option<BakedCb>>,
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

pub(crate) fn push_f32_param(
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
pub(crate) fn upload_dense_mat(
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

pub(crate) fn upload_range_uncached(
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
        Self::from_gguf_with_options(gguf, max_ctx, None, 0)
    }

    /// Loads `Inkling2VkRunner` while reserving `draft_reserve_bytes` of VRAM slack for
    /// [`crate::DFlashVkRunner`] when placing MoE expert layers on discrete GPUs.
    pub fn from_gguf_with_draft_reserve(
        gguf: Arc<GgufFile>,
        max_ctx: usize,
        draft_reserve_bytes: u64,
    ) -> Result<Self, VulkanError> {
        Self::from_gguf_with_options(gguf, max_ctx, None, draft_reserve_bytes)
    }

    /// Loads `Inkling2VkRunner` with an optional explicit VRAM budget override (in bytes).
    pub fn from_gguf_with_vram_budget(
        gguf: Arc<GgufFile>,
        max_ctx: usize,
        vram_budget_override: Option<u64>,
    ) -> Result<Self, VulkanError> {
        Self::from_gguf_with_options(gguf, max_ctx, vram_budget_override, 0)
    }

    fn from_gguf_with_options(
        gguf: Arc<GgufFile>,
        max_ctx: usize,
        vram_budget_override: Option<u64>,
        draft_reserve_bytes: u64,
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
        let all_fit = total_file_bytes
            .saturating_add(state_vram_bytes)
            .saturating_add(draft_reserve_bytes)
            <= vram_budget
            && max_single_expert_tensor <= buf_cap;
        if all_fit {
            vram_expert_mask.fill(true);
        } else if !info.unified_memory && max_single_expert_tensor <= buf_cap {
            let mut rem = vram_budget
                .saturating_sub(base_needed)
                .saturating_sub(draft_reserve_bytes);
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
                host_imported_experts: None,
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
        let dyn_pos_buf = ctx.create_host_buffer(16)?;
        let step_tok0_buf = ctx.alloc_buffer(16)?;
        let win_state_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * per_tok_state_f32s * 4).max(16) as u64)?;
        let conv_snapshots_buf =
            ctx.alloc_buffer((MAX_INKLING2_SPEC_WINDOW * conv_f32_cursor * 4).max(16) as u64)?;
        let dspark_taps_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * 8 * n_embd * 4).max(16) as u64)?;
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
        let ctrl_in_bufs = [
            ctx.alloc_buffer((2 * n_embd * 4) as u64)?,
            ctx.alloc_buffer((2 * n_embd * 4) as u64)?,
            ctx.alloc_buffer((2 * n_embd * 4) as u64)?,
        ];
        let ctrl_g_bufs = [
            ctx.alloc_buffer((cfg.gate_rank.max(1) * 4) as u64)?,
            ctx.alloc_buffer((cfg.gate_rank.max(1) * 4) as u64)?,
            ctx.alloc_buffer((cfg.gate_rank.max(1) * 4) as u64)?,
        ];
        let ctrl_gain_bufs = [
            ctx.alloc_buffer((n_embd * 4) as u64)?,
            ctx.alloc_buffer((n_embd * 4) as u64)?,
            ctx.alloc_buffer((n_embd * 4) as u64)?,
        ];
        let ctrl_forget_bufs = [
            ctx.alloc_buffer((n_embd * 4) as u64)?,
            ctx.alloc_buffer((n_embd * 4) as u64)?,
            ctx.alloc_buffer((n_embd * 4) as u64)?,
        ];
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
        let win_taus_buf = ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * 4).max(16) as u64)?;
        let win_inputs_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * 3 * n_embd * 4).max(16) as u64)?;
        let win_messages_buf = ctx
            .alloc_buffer((MAX_INKLING2_VK_WINDOW * LOOP_CTRL_COUNT * n_embd * 4).max(16) as u64)?;
        let win_ctrl_in_bufs = [
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * 2 * n_embd * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * 2 * n_embd * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * 2 * n_embd * 4).max(16) as u64)?,
        ];
        let win_ctrl_g_bufs = [
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.gate_rank.max(1) * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.gate_rank.max(1) * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.gate_rank.max(1) * 4).max(16) as u64)?,
        ];
        let win_ctrl_gain_bufs = [
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4).max(16) as u64)?,
        ];
        let win_ctrl_forget_bufs = [
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4).max(16) as u64)?,
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4).max(16) as u64)?,
        ];
        let win_q_in_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let win_k_in_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let win_v_in_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let win_q_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_q_dim * 4) as u64)?;
        let win_k_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_kv_dim * 4) as u64)?;
        let win_v_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_kv_dim * 4) as u64)?;
        let win_v_norm_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_kv_dim * 4) as u64)?;
        let win_r_buf = ctx
            .alloc_buffer((MAX_INKLING2_VK_WINDOW * max_nh * cfg.n_relative * 4).max(16) as u64)?;
        let win_sinks_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_nh * 4).max(16) as u64)?;
        let win_gate_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_q_dim * 4) as u64)?;
        let win_k_dyn_buf = ctx.alloc_buffer(
            (MAX_INKLING2_VK_WINDOW * cfg.shortconv_l_cache * max_nh_kv * 4).max(16) as u64,
        )?;
        let win_v_dyn_buf = ctx.alloc_buffer(
            (MAX_INKLING2_VK_WINDOW * cfg.shortconv_l_cache * max_nh_kv * 4).max(16) as u64,
        )?;
        let win_attn_vec_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * max_q_dim * 4) as u64)?;
        let win_wo_out_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let win_ffn_norm_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let win_router_logits_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_exp * 4).max(16) as u64)?;
        let win_top_ids_dev_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 4).max(16) as u64)?;
        let win_top_weights_dev_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 4).max(16) as u64)?;
        let win_moe_out_dev_buf = ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let moe_tiles_buf = ctx.alloc_buffer(65536)?;
        let exp_gate_up_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 2 * n_ff * 4) as u64)?;
        let exp_hidden_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * n_ff * 4) as u64)?;
        let exp_down_buf =
            ctx.alloc_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * n_embd * 4) as u64)?;
        let logits_padded_buf = ctx.alloc_buffer((cfg.vocab_padded * 4) as u64)?;
        let logits_buf = ctx.create_host_buffer((cfg.vocab_size * 4) as u64)?;
        let win_logits_padded_buf =
            ctx.alloc_buffer((MAX_INKLING2_SPEC_WINDOW * cfg.vocab_padded * 4).max(16) as u64)?;
        let win_logits_dev_buf =
            ctx.alloc_buffer((MAX_INKLING2_SPEC_WINDOW * cfg.vocab_size * 4).max(16) as u64)?;
        let argmax_buf = ctx.create_host_buffer(16)?;
        let dummy_buf = ctx.alloc_buffer(64)?;
        let max_q8_elems = MAX_INKLING2_VK_WINDOW
            * (cfg.n_expert_used * n_ff)
                .max(4 * n_embd + 1024)
                .max(max_q_dim)
                .max(32);
        let max_q8_blks = max_q8_elems.div_ceil(32);
        let xq_buf = ctx.alloc_buffer((max_q8_blks * 32).max(32) as u64)?;
        let xd_buf = ctx.alloc_buffer((max_q8_blks * 8).max(16) as u64)?;
        let ext_q8_elems =
            (MAX_INKLING2_SPEC_WINDOW * (cfg.n_expert_used * n_ff).max(n_embd)).max(32);
        let ext_q8_blks = ext_q8_elems.div_ceil(32);
        let ext_xq_buf = ctx.alloc_buffer((ext_q8_blks * 32).max(32) as u64)?;
        let ext_xd_buf = ctx.alloc_buffer((ext_q8_blks * 8).max(16) as u64)?;

        let ffn_normed_bridge =
            ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;
        let top_ids_bridge = ctx
            .create_host_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 4).max(16) as u64)?;
        let top_weights_bridge = ctx
            .create_host_buffer((MAX_INKLING2_VK_WINDOW * cfg.n_expert_used * 4).max(16) as u64)?;
        let moe_out_bridge =
            ctx.create_host_buffer((MAX_INKLING2_VK_WINDOW * n_embd * 4) as u64)?;

        // Build flat execution schedule once at load time (zero per-token heap allocation!)
        let mut schedule = Vec::with_capacity(cfg.n_layers);
        let mut visit_ctr = 0usize;
        if !cfg.is_looped() {
            for il in 0..cfg.n_layers_phys {
                schedule.push(InklingExecStep {
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
            let cs = cfg.core_start;
            let ce = cfg.core_end;
            for il in 0..cs {
                schedule.push(InklingExecStep {
                    visit: visit_ctr,
                    il,
                    ip: il,
                    pass: 0,
                    save_prelude_before: false,
                    reset_pass_before: false,
                });
                visit_ctr += 1;
            }
            for pass in 0..cfg.n_passes {
                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    schedule.push(InklingExecStep {
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
            for il in ce..cfg.n_layers_phys {
                schedule.push(InklingExecStep {
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

        // Build compute pipelines
        let sg = info.subgroup_size;
        let max_wg = info.max_workgroup_invocations;
        let ops_wg = 128u32.max(sg).min(max_wg);
        let gemv_wg = 128u32.max(sg).min(max_wg);
        let gemv_nr = 4u32;
        let gemv_small_wg = 64u32.max(sg).min(max_wg);
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
        let pipe_embed_batch = ctx.pipeline(
            "inkling2_embed_batch",
            tok_embd.dtype_code | 0x100,
            SPV_EMBED,
            3,
            8,
            &[64u32.max(sg), tok_embd.dtype_code, 1],
        )?;
        let pipe_ops = ctx.pipeline("inkling2_ops", ops_wg, SPV_INKLING2_OPS, 6, 32, &[ops_wg])?;
        let pipe_rel_attn = ctx.pipeline(
            "inkling2_rel_attn",
            attn_wg | (attn_n_vec << 16),
            SPV_INKLING2_REL_ATTN,
            8,
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

        // Pre-cache large-row (`NR=4`) and small-row (`WG=64, NR=1`) GEMV pipelines (`M=1, 4, 8`)
        let mut gemv_vec = Vec::with_capacity(6);
        let mut gemv_small_vec = Vec::with_capacity(6);
        let mut gemv_m4_vec = Vec::with_capacity(6);
        let mut gemv_m8_vec = Vec::with_capacity(6);
        let mut gemv_small_m4_vec = Vec::with_capacity(6);
        let mut gemv_small_m8_vec = Vec::with_capacity(6);
        for dt_code in 0u32..6 {
            gemv_vec.push(ctx.pipeline(
                "gemv_res",
                (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_wg, dt_code, gemv_nr, 1],
            )?);
            gemv_small_vec.push(ctx.pipeline(
                "gemv_res",
                (gemv_small_wg << 8) | (1u32 << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_small_wg, dt_code, 1, 1],
            )?);
            gemv_m4_vec.push(ctx.pipeline(
                "gemv_res",
                (4u32 << 20) | (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_wg, dt_code, gemv_nr, 4],
            )?);
            gemv_m8_vec.push(ctx.pipeline(
                "gemv_res",
                (8u32 << 20) | (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_wg, dt_code, gemv_nr, 8],
            )?);
            gemv_small_m4_vec.push(ctx.pipeline(
                "gemv_res",
                (4u32 << 20) | (gemv_small_wg << 8) | (1u32 << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_small_wg, dt_code, 1, 4],
            )?);
            gemv_small_m8_vec.push(ctx.pipeline(
                "gemv_res",
                (8u32 << 20) | (gemv_small_wg << 8) | (1u32 << 4) | dt_code,
                SPV_GEMV_RES,
                3,
                16,
                &[gemv_small_wg, dt_code, 1, 8],
            )?);
        }
        let gemv_pipes: [Arc<Pipeline>; 6] =
            gemv_vec.try_into().unwrap_or_else(|_| unreachable!("6"));
        let gemv_small_pipes: [Arc<Pipeline>; 6] = gemv_small_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("6"));
        let gemv_m4_pipes: [Arc<Pipeline>; 6] =
            gemv_m4_vec.try_into().unwrap_or_else(|_| unreachable!("6"));
        let gemv_m8_pipes: [Arc<Pipeline>; 6] =
            gemv_m8_vec.try_into().unwrap_or_else(|_| unreachable!("6"));
        let gemv_small_m4_pipes: [Arc<Pipeline>; 6] = gemv_small_m4_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("6"));
        let gemv_small_m8_pipes: [Arc<Pipeline>; 6] = gemv_small_m8_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("6"));

        let (dp4a_min_dim, dp4a_q8_enabled) = if !info.int8_dot_accelerated || max_wg < 256 {
            (usize::MAX, false)
        } else {
            match std::env::var("TM_VK_DP4A").ok().as_deref().map(str::trim) {
                Some("0" | "false" | "off") => (usize::MAX, false),
                Some("force") => (32, true),
                Some("1" | "true" | "on") => (256, true),
                _ => {
                    if VkPrefillMode::from_env() == VkPrefillMode::Fallback {
                        (usize::MAX, false)
                    } else {
                        // In `auto` mode, enable `dp4a` for 4/5-bit (`Q4_0`/`Q4_1`/`Q5_K`) matrices while
                        // keeping `Q8_0` on zero-barrier 8-lane coalesced `w32` GEMV (avoids 336 unfused
                        // `quantize_q8_1` barriers per token across 48 loop visits; measured +2.0 tok/s on L4).
                        (256, false)
                    }
                }
            }
        };
        let (
            pipe_quant_q8_1,
            pipe_moe_dp4a,
            gemv_dp4a_pipes,
            gemv_dp4a_small_pipes,
            gemv_dp4a_m4_pipes,
            gemv_dp4a_m8_pipes,
            gemv_dp4a_small_m4_pipes,
            gemv_dp4a_small_m8_pipes,
        ) = if dp4a_min_dim < usize::MAX {
            let p_q = ctx.pipeline("quantize_q8_1", 256, SPV_QUANTIZE_Q8_1, 3, 4, &[256])?;
            let p_moe = ctx.pipeline(
                "inkling2_moe_dp4a",
                moe_wg | (moe_nr << 16),
                SPV_INKLING2_MOE_DP4A,
                5,
                32,
                &[moe_wg, moe_nr],
            )?;
            let mut dv = Vec::with_capacity(4);
            let mut ds = Vec::with_capacity(4);
            let mut dm4 = Vec::with_capacity(4);
            let mut dm8 = Vec::with_capacity(4);
            let mut dsm4 = Vec::with_capacity(4);
            let mut dsm8 = Vec::with_capacity(4);
            for dt_code in 0u32..4 {
                dv.push(ctx.pipeline(
                    "gemv_dp4a",
                    (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                    SPV_GEMV_DP4A,
                    4,
                    16,
                    &[gemv_wg, dt_code, gemv_nr, 1],
                )?);
                ds.push(ctx.pipeline(
                    "gemv_dp4a",
                    (gemv_small_wg << 8) | (1u32 << 4) | dt_code,
                    SPV_GEMV_DP4A,
                    4,
                    16,
                    &[gemv_small_wg, dt_code, 1, 1],
                )?);
                dm4.push(ctx.pipeline(
                    "gemv_dp4a",
                    (4u32 << 20) | (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                    SPV_GEMV_DP4A,
                    4,
                    16,
                    &[gemv_wg, dt_code, gemv_nr, 4],
                )?);
                dm8.push(ctx.pipeline(
                    "gemv_dp4a",
                    (8u32 << 20) | (gemv_wg << 8) | (gemv_nr << 4) | dt_code,
                    SPV_GEMV_DP4A,
                    4,
                    16,
                    &[gemv_wg, dt_code, gemv_nr, 8],
                )?);
                dsm4.push(ctx.pipeline(
                    "gemv_dp4a",
                    (4u32 << 20) | (gemv_small_wg << 8) | (1u32 << 4) | dt_code,
                    SPV_GEMV_DP4A,
                    4,
                    16,
                    &[gemv_small_wg, dt_code, 1, 4],
                )?);
                dsm8.push(ctx.pipeline(
                    "gemv_dp4a",
                    (8u32 << 20) | (gemv_small_wg << 8) | (1u32 << 4) | dt_code,
                    SPV_GEMV_DP4A,
                    4,
                    16,
                    &[gemv_small_wg, dt_code, 1, 8],
                )?);
            }
            let cv = |v: Vec<Arc<Pipeline>>| -> [Arc<Pipeline>; 4] {
                v.try_into().unwrap_or_else(|_| unreachable!("4"))
            };
            (
                Some(p_q),
                Some(p_moe),
                Some(cv(dv)),
                Some(cv(ds)),
                Some(cv(dm4)),
                Some(cv(dm8)),
                Some(cv(dsm4)),
                Some(cv(dsm8)),
            )
        } else {
            (None, None, None, None, None, None, None, None)
        };

        let use_cm_f16 = VkPrefillMode::from_env() != VkPrefillMode::Fallback
            && info.coopmat_f16_f32acc
            && sg == 32
            && max_wg >= 128;
        let mut gemm_pipes: [Option<Arc<Pipeline>>; 6] = Default::default();
        let gemm_tile_mn = if use_cm_f16 {
            for (dt_code, slot) in gemm_pipes.iter_mut().enumerate() {
                let dt = dt_code as u32;
                *slot = Some(ctx.pipeline(
                    "gemm_coopmat_f16",
                    (64 << 16) | (64 << 8) | dt,
                    SPV_GEMM_COOPMAT_F16,
                    3,
                    20,
                    &[128, dt, 64, 64],
                )?);
            }
            (64usize, 64usize)
        } else if 128 >= sg && 128 <= max_wg {
            for dt in [0u32, 1, 2, 5] {
                gemm_pipes[dt as usize] =
                    Some(ctx.pipeline("gemm_tiled", dt, SPV_GEMM_TILED, 3, 20, &[128, dt])?);
            }
            (32usize, 64usize)
        } else {
            (32usize, 64usize)
        };

        let (pipe_moe_grouped, moe_grouped_bm, moe_grouped_bn) = if use_cm_f16 {
            (
                Some(ctx.pipeline(
                    "inkling2_moe_grp_cm",
                    128,
                    SPV_INKLING2_MOE_GROUPED_CM,
                    4,
                    20,
                    &[128],
                )?),
                64usize,
                64usize,
            )
        } else if 128 >= sg && 128 <= max_wg {
            (
                Some(ctx.pipeline(
                    "inkling2_moe_grp",
                    128,
                    SPV_INKLING2_MOE_GROUPED,
                    4,
                    20,
                    &[128],
                )?),
                16usize,
                64usize,
            )
        } else {
            (None, 16usize, 64usize)
        };

        let n_vram_exps = vram_expert_mask.iter().filter(|&&b| b).count();
        let weights_mode_label = if n_vram_exps == n_phys {
            "vram-full".to_string()
        } else {
            format!("hybrid-vram{n_vram_exps}/{n_phys}exps")
        };

        let prof_level = std::env::var("TM_VK_PROF")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0);
        let prof_on = prof_level >= 1;
        let prof_qp = if prof_level >= 2 && info.timestamp_compute {
            // SAFETY: Valid device; destroyed in `Inkling2VkRunner::drop`.
            unsafe {
                ctx.inner
                    .device
                    .create_query_pool(
                        &ash::vk::QueryPoolCreateInfo::default()
                            .query_type(ash::vk::QueryType::TIMESTAMP)
                            .query_count(MAX_PROF_QUERIES),
                        None,
                    )
                    .ok()
            }
        } else {
            None
        };
        let default_hybrid_gpu_exps = if !info.unified_memory && cfg.n_expert_used >= 4 {
            (cfg.n_expert_used / 4).clamp(1, 3)
        } else {
            0
        };
        let hybrid_gpu_exps = std::env::var("TM_VK_HYBRID_GPU_EXPS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(default_hybrid_gpu_exps)
            .min(cfg.n_expert_used);
        let baked_ext_moe_cbs: Vec<Option<BakedCb>> = (0..n_phys).map(|_| None).collect();
        let prof_ts_buf = if prof_qp.is_some() {
            vec![0u64; MAX_PROF_QUERIES as usize]
        } else {
            Vec::new()
        };
        let prof_spans_slots = if prof_qp.is_some() {
            [
                Vec::with_capacity(MAX_PROF_QUERIES as usize),
                Vec::with_capacity(MAX_PROF_QUERIES as usize),
            ]
        } else {
            [Vec::new(), Vec::new()]
        };

        Ok(Self {
            ctx,
            schedule,
            prof_on,
            prof: Inkling2Prof::default(),
            prof_qp,
            prof_ts_buf,
            prof_spans_slots,
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
            dyn_pos_buf,
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
            ctrl_in_bufs,
            ctrl_g_bufs,
            ctrl_gain_bufs,
            ctrl_forget_bufs,
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
            win_taus_buf,
            win_inputs_buf,
            win_messages_buf,
            win_ctrl_in_bufs,
            win_ctrl_g_bufs,
            win_ctrl_gain_bufs,
            win_ctrl_forget_bufs,
            win_q_in_buf,
            win_k_in_buf,
            win_v_in_buf,
            win_q_buf,
            win_k_buf,
            win_v_buf,
            win_v_norm_buf,
            win_r_buf,
            win_sinks_buf,
            win_gate_buf,
            win_k_dyn_buf,
            win_v_dyn_buf,
            win_attn_vec_buf,
            win_wo_out_buf,
            win_ffn_norm_buf,
            win_router_logits_buf,
            win_top_ids_dev_buf,
            win_top_weights_dev_buf,
            win_moe_out_dev_buf,
            moe_tiles_buf,
            exp_gate_up_buf,
            exp_hidden_buf,
            exp_down_buf,
            logits_padded_buf,
            logits_buf,
            win_logits_padded_buf,
            win_logits_dev_buf,
            argmax_buf,
            dummy_buf,
            xq_buf,
            xd_buf,
            ext_xq_buf,
            ext_xd_buf,
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
            pipe_embed_batch,
            pipe_ops,
            pipe_rel_attn,
            pipe_sigmoid_mul,
            pipe_moe_ops,
            pipe_moe_dp4a,
            pipe_quant_q8_1,
            pipe_moe_grouped,
            moe_grouped_bm,
            moe_grouped_bn,
            pipe_argmax,
            gemv_pipes,
            gemv_small_pipes,
            gemv_m4_pipes,
            gemv_m8_pipes,
            gemv_small_m4_pipes,
            gemv_small_m8_pipes,
            gemv_dp4a_pipes,
            gemv_dp4a_small_pipes,
            gemv_dp4a_m4_pipes,
            gemv_dp4a_m8_pipes,
            gemv_dp4a_small_m4_pipes,
            gemv_dp4a_small_m8_pipes,
            dp4a_min_dim,
            dp4a_q8_enabled,
            gemm_pipes,
            gemm_tile_mn,
            ops_wg,
            gemv_wg,
            gemv_nr,
            gemv_small_wg,
            moe_wg,
            moe_nr,
            hybrid_gpu_exps,
            baked_decode_cbs: [None, None, None],
            baked_ext_moe_cbs,
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

    /// Clears `TM_VK_PROF=1` phase accumulators.
    pub fn reset_prof(&mut self) {
        self.prof = Inkling2Prof::default();
    }

    /// Returns a per-token phase breakdown since the last [`Self::reset_prof`] when `TM_VK_PROF=1`.
    pub fn prof_report(&self) -> Option<String> {
        if !self.prof_on || self.prof.tokens == 0 {
            return None;
        }
        let p = self.prof;
        let t = p.tokens as f64;
        let ms = |ns: u64| ns as f64 / 1e6 / t;
        let total = p.gpu_record_ns + p.gpu_wait_ns + p.host_moe_ns + p.bridge_ns;
        let mut out = format!(
            "[inkling2 prof] {} tok: submits/tok={:.1} | gpu_record={:.2} ms | gpu_submit+exec+wait={:.2} ms | host_moe={:.2} ms ({:.1} calls/tok) | bridge={:.2} ms | sum={:.2} ms/tok",
            p.tokens,
            p.gpu_submits as f64 / t,
            ms(p.gpu_record_ns),
            ms(p.gpu_wait_ns),
            ms(p.host_moe_ns),
            p.host_moe_calls as f64 / t,
            ms(p.bridge_ns),
            ms(total),
        );
        if p.gpu_hw_ns > 0 {
            let gap_ns = p.gpu_wait_ns.saturating_sub(p.gpu_hw_ns);
            out.push_str(&format!(
                "\n  [gpu_hw] exec={:.2} ms/tok (submit+fence_gap={:.2} ms)",
                ms(p.gpu_hw_ns),
                ms(gap_ns),
            ));
            for (name, &ns) in PROF_STAGE_NAMES.iter().zip(p.stage_gpu_ns.iter()) {
                out.push_str(&format!(" | {name}={:.2}", ms(ns)));
            }
            out.push_str(" ms/tok");
        }
        Some(out)
    }

    /// Accumulates one `ctx.run_timed` submission + optional hardware timestamp spans into `self.prof`.
    fn prof_finish_submit(
        &mut self,
        (rec_ns, total_ns): (u64, u64),
        next_q: u32,
        spans: &[(usize, u32, u32)],
    ) {
        if !self.prof_on {
            return;
        }
        self.prof.gpu_submits += 1;
        self.prof.gpu_record_ns += rec_ns;
        self.prof.gpu_wait_ns += total_ns.saturating_sub(rec_ns);
        if let Some(qp) = self.prof_qp
            && next_q > 1
        {
            let q_cnt = next_q as usize;
            // SAFETY: `ctx.run_timed` already waited on the submission fence, and `q_cnt <= MAX_PROF_QUERIES`.
            let ok = unsafe {
                self.ctx
                    .inner
                    .device
                    .get_query_pool_results(
                        qp,
                        0,
                        &mut self.prof_ts_buf[..q_cnt],
                        ash::vk::QueryResultFlags::TYPE_64 | ash::vk::QueryResultFlags::WAIT,
                    )
                    .is_ok()
            };
            if ok {
                let period = self.ctx.info().timestamp_period_ns as f64;
                let total_ticks = self.prof_ts_buf[q_cnt - 1].wrapping_sub(self.prof_ts_buf[0]);
                self.prof.gpu_hw_ns += ((total_ticks as f64) * period) as u64;
                for &(st, q0, q1) in spans {
                    let dt =
                        self.prof_ts_buf[q1 as usize].wrapping_sub(self.prof_ts_buf[q0 as usize]);
                    self.prof.stage_gpu_ns[st] += ((dt as f64) * period) as u64;
                }
            }
        }
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

    /// Zero-copy imports a host-RAM `HugePageBytes` `Q8_0` expert layer (`ip`) into Vulkan via
    /// `VK_EXT_external_memory_host` so multi-token prefill (`W >= 8`, `!save_conv_snapshots`)
    /// executes `inkling2_moe_grouped_cm` / `inkling2_moe_grouped` directly over PCIe/UMA (`0`
    /// extra host RAM, `0` `DEVICE_LOCAL` VRAM, `1` command buffer across all 48 visits!), while
    /// single-token decode (`W == 1`) continues using high-bandwidth host DDR4 `AVX2` kernels.
    pub fn register_host_moe_layer(
        &mut self,
        ip: usize,
        gate_up_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
        gate_up_logical_bytes: usize,
        down_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
        down_logical_bytes: usize,
    ) -> bool {
        if ip >= self.cfg.n_layers_phys || self.layers[ip].vram_experts.is_some() {
            return false;
        }
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        let n_exp = self.cfg.n_expert;
        let gu_row_bytes = DType::Q8_0.row_bytes(n_embd);
        let dn_row_bytes = DType::Q8_0.row_bytes(n_ff);
        let exp_gu_bytes = gu_row_bytes * (2 * n_ff) * n_exp;
        let exp_dn_bytes = dn_row_bytes * n_embd * n_exp;
        if gate_up_logical_bytes != exp_gu_bytes || down_logical_bytes != exp_dn_bytes {
            return false;
        }
        let gu_aligned = (*gate_up_pin).as_ref();
        let dn_aligned = (*down_pin).as_ref();
        if gu_aligned.len() < exp_gu_bytes || dn_aligned.len() < exp_dn_bytes {
            return false;
        }
        let Some(gu_buf) = self.ctx.try_import_host_rw_pinned(gu_aligned) else {
            return false;
        };
        let Some(dn_buf) = self.ctx.try_import_host_rw_pinned(dn_aligned) else {
            return false;
        };
        let gu_row_u16s = (gu_row_bytes / 2) as u32;
        let dn_row_u16s = (dn_row_bytes / 2) as u32;
        self.layers[ip].host_imported_experts = Some(VkImportedExperts {
            vexp: VkVramExperts {
                gate_up: gu_buf,
                gate_up_row_u16s: gu_row_u16s,
                gate_up_expert_u16s: gu_row_u16s * (2 * n_ff as u32),
                down: dn_buf,
                down_row_u16s: dn_row_u16s,
                down_expert_u16s: dn_row_u16s * (n_embd as u32),
            },
            _gu_pin: gate_up_pin,
            _dn_pin: down_pin,
        });
        let n_vram = self.vram_expert_mask.iter().filter(|&&b| b).count();
        let n_ext = self
            .layers
            .iter()
            .filter(|l| l.host_imported_experts.is_some())
            .count();
        if n_ext > 0 {
            self.weights_mode_label = format!(
                "hybrid-vram{n_vram}+ext{n_ext}/{}exps",
                self.cfg.n_layers_phys
            );
        }
        self.invalidate_baked_cbs();
        true
    }

    /// Overrides the number of per-token MoE experts (`0..=n_expert_used`) evaluated concurrently
    /// on the GPU over zero-copy PCIe while host CPU threads evaluate the remaining experts.
    pub fn set_hybrid_gpu_experts(&mut self, count: usize) {
        let clamped = count.min(self.cfg.n_expert_used);
        if self.hybrid_gpu_exps != clamped {
            self.hybrid_gpu_exps = clamped;
            self.invalidate_baked_cbs();
        }
    }

    fn invalidate_baked_cbs(&mut self) {
        self.baked_decode_cbs = [None, None, None];
        for slot in &mut self.baked_ext_moe_cbs {
            *slot = None;
        }
    }

    #[inline]
    fn layer_gpu_ext_experts(&self, ip: usize) -> usize {
        if self.layers[ip].vram_experts.is_some() || self.layers[ip].host_imported_experts.is_none()
        {
            0
        } else {
            self.hybrid_gpu_exps.min(self.cfg.n_expert_used)
        }
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

    #[inline]
    fn select_gemv(&self, n_out: usize, dt_code: u32) -> (&Pipeline, u32) {
        let sg = self.ctx.info().subgroup_size;
        if n_out <= 1024 {
            let rows_per_wg = (self.gemv_small_wg / sg) as usize;
            (
                &self.gemv_small_pipes[dt_code as usize],
                n_out.div_ceil(rows_per_wg) as u32,
            )
        } else {
            let rows_per_wg = ((self.gemv_wg / sg) * self.gemv_nr) as usize;
            (
                &self.gemv_pipes[dt_code as usize],
                n_out.div_ceil(rows_per_wg) as u32,
            )
        }
    }

    #[inline]
    fn select_gemv_batch(&self, n_out: usize, dt_code: u32, m_cap: usize) -> (&Pipeline, u32) {
        let sg = self.ctx.info().subgroup_size;
        if n_out <= 1024 {
            let rows_per_wg = (self.gemv_small_wg / sg) as usize;
            let pipes = if m_cap <= 4 {
                &self.gemv_small_m4_pipes
            } else {
                &self.gemv_small_m8_pipes
            };
            (&pipes[dt_code as usize], n_out.div_ceil(rows_per_wg) as u32)
        } else {
            let rows_per_wg = ((self.gemv_wg / sg) * self.gemv_nr) as usize;
            let pipes = if m_cap <= 4 {
                &self.gemv_m4_pipes
            } else {
                &self.gemv_m8_pipes
            };
            (&pipes[dt_code as usize], n_out.div_ceil(rows_per_wg) as u32)
        }
    }

    #[inline]
    fn elem_groups(&self, n: usize) -> u32 {
        n.div_ceil(self.ops_wg as usize) as u32
    }

    #[inline]
    fn advance_depth_counts(
        &self,
        st: InklingExecStep,
        b: &mut usize,
        pb: &mut usize,
        bk: &mut usize,
    ) {
        if st.save_prelude_before {
            *pb = *b;
        }
        if st.reset_pass_before {
            let bank_start = *pb + 1;
            *bk = b.saturating_sub(bank_start);
            *b = *pb;
        }
        if st.ip.is_multiple_of(self.cfg.attn_res_block_size) {
            *b += 1;
        }
    }

    fn record_quant_q8_1(
        &self,
        rec: &mut Recorder<'_>,
        x: &VkBuffer,
        xq: &VkBuffer,
        xd: &VkBuffer,
        n_elems: usize,
    ) -> Result<(), VulkanError> {
        let p_q = self
            .pipe_quant_q8_1
            .as_ref()
            .expect("record_quant_q8_1 requires pipe_quant_q8_1");
        let n_blocks = (n_elems / 32) as u32;
        rec.dispatch(p_q, &[x, xq, xd], &[n_blocks], n_blocks.div_ceil(8))
    }

    #[inline]
    fn select_gemv_dp4a(&self, n_out: usize, dt_code: u32) -> Option<(&Pipeline, u32)> {
        let sg = self.ctx.info().subgroup_size;
        if n_out <= 1024 {
            let pipes = self.gemv_dp4a_small_pipes.as_ref()?;
            let rows_per_wg = (self.gemv_small_wg / sg) as usize;
            Some((&pipes[dt_code as usize], n_out.div_ceil(rows_per_wg) as u32))
        } else {
            let pipes = self.gemv_dp4a_pipes.as_ref()?;
            let rows_per_wg = ((self.gemv_wg / sg) * self.gemv_nr) as usize;
            Some((&pipes[dt_code as usize], n_out.div_ceil(rows_per_wg) as u32))
        }
    }

    #[inline]
    fn select_gemv_dp4a_batch(
        &self,
        n_out: usize,
        dt_code: u32,
        m_cap: usize,
    ) -> Option<(&Pipeline, u32)> {
        let sg = self.ctx.info().subgroup_size;
        if n_out <= 1024 {
            let pipes = if m_cap <= 4 {
                self.gemv_dp4a_small_m4_pipes.as_ref()?
            } else {
                self.gemv_dp4a_small_m8_pipes.as_ref()?
            };
            let rows_per_wg = (self.gemv_small_wg / sg) as usize;
            Some((&pipes[dt_code as usize], n_out.div_ceil(rows_per_wg) as u32))
        } else {
            let pipes = if m_cap <= 4 {
                self.gemv_dp4a_m4_pipes.as_ref()?
            } else {
                self.gemv_dp4a_m8_pipes.as_ref()?
            };
            let rows_per_wg = ((self.gemv_wg / sg) * self.gemv_nr) as usize;
            Some((&pipes[dt_code as usize], n_out.div_ceil(rows_per_wg) as u32))
        }
    }

    #[inline]
    fn use_mat_dp4a(&self, mat: &VkDenseMat) -> bool {
        (mat.dtype_code <= 1
            || mat.dtype_code == 3
            || (mat.dtype_code == 2 && self.dp4a_q8_enabled))
            && mat.cols >= self.dp4a_min_dim
            && mat.rows >= self.dp4a_min_dim
            && mat.cols.is_multiple_of(32)
    }

    pub(crate) fn dispatch_gemv(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkDenseMat,
        x: &VkBuffer,
        y: &VkBuffer,
    ) -> Result<(), VulkanError> {
        if self.use_mat_dp4a(mat)
            && let Some((p, gx)) = self.select_gemv_dp4a(mat.rows, mat.dtype_code)
        {
            self.record_quant_q8_1(rec, x, &self.xq_buf, &self.xd_buf, mat.cols)?;
            return rec.dispatch(
                p,
                &[&mat.buf, &self.xq_buf, &self.xd_buf, y],
                &[mat.cols as u32, mat.rows as u32, 0, mat.row_u16s],
                gx,
            );
        }
        let (p, gx) = self.select_gemv(mat.rows, mat.dtype_code);
        rec.dispatch(
            p,
            &[&mat.buf, x, y],
            &[mat.cols as u32, mat.rows as u32, 0, mat.row_u16s],
            gx,
        )
    }

    fn dispatch_gemv_nb(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkDenseMat,
        x: &VkBuffer,
        y: &VkBuffer,
    ) -> Result<(), VulkanError> {
        if self.use_mat_dp4a(mat)
            && let Some((p, gx)) = self.select_gemv_dp4a(mat.rows, mat.dtype_code)
        {
            self.record_quant_q8_1(rec, x, &self.xq_buf, &self.xd_buf, mat.cols)?;
            return rec.dispatch(
                p,
                &[&mat.buf, &self.xq_buf, &self.xd_buf, y],
                &[mat.cols as u32, mat.rows as u32, 0, mat.row_u16s],
                gx,
            );
        }
        let (p, gx) = self.select_gemv(mat.rows, mat.dtype_code);
        rec.dispatch_nb(
            p,
            &[&mat.buf, x, y],
            &[mat.cols as u32, mat.rows as u32, 0, mat.row_u16s],
            gx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm_inner(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkDenseMat,
        x_win: &VkBuffer,
        y_win: &VkBuffer,
        n_tok: usize,
        use_tc: bool,
        emit_barrier: bool,
    ) -> Result<(), VulkanError> {
        if use_tc
            && mat.cols.is_multiple_of(32)
            && let Some(ref p) = self.gemm_pipes[mat.dtype_code as usize]
        {
            let (bm, bn) = self.gemm_tile_mn;
            let gx = mat.rows.div_ceil(bn) as u32;
            let gy = n_tok.div_ceil(bm) as u32;
            let push = [
                mat.cols as u32,
                mat.rows as u32,
                n_tok as u32,
                0,
                mat.row_u16s,
            ];
            if emit_barrier {
                rec.dispatch_xy(p, &[&mat.buf, x_win, y_win], &push, gx, gy)
            } else {
                rec.dispatch_xy_nb(p, &[&mat.buf, x_win, y_win], &push, gx, gy)
            }
        } else if (2..=8).contains(&n_tok)
            && mat.dtype_code <= 2
            && self.use_mat_dp4a(mat)
            && let Some((p, gx)) = self.select_gemv_dp4a_batch(mat.rows, mat.dtype_code, n_tok)
        {
            self.record_quant_q8_1(rec, x_win, &self.xq_buf, &self.xd_buf, n_tok * mat.cols)?;
            let push = [
                mat.cols as u32,
                mat.rows as u32,
                (n_tok as u32) << 8,
                mat.row_u16s,
            ];
            rec.dispatch(p, &[&mat.buf, &self.xq_buf, &self.xd_buf, y_win], &push, gx)
        } else if (2..=8).contains(&n_tok) && matches!(mat.dtype_code, 0 | 1 | 2 | 5) {
            // Register-batched `M = 4` / `M = 8` `gemv_res`: loads `W` ONCE across all `n_tok`
            // tokens while remaining `0-ULP` bit-exact with `n_tok = 1`!
            let (p, gx) = self.select_gemv_batch(mat.rows, mat.dtype_code, n_tok);
            let push = [
                mat.cols as u32,
                mat.rows as u32,
                (n_tok as u32) << 8,
                mat.row_u16s,
            ];
            if emit_barrier {
                rec.dispatch(p, &[&mat.buf, x_win, y_win], &push, gx)
            } else {
                rec.dispatch_nb(p, &[&mat.buf, x_win, y_win], &push, gx)
            }
        } else {
            let (p, gx) = self.select_gemv(mat.rows, mat.dtype_code);
            let push = [mat.cols as u32, mat.rows as u32, 0, mat.row_u16s];
            if emit_barrier {
                rec.dispatch_xy(p, &[&mat.buf, x_win, y_win], &push, gx, n_tok as u32)
            } else {
                rec.dispatch_xy_nb(p, &[&mat.buf, x_win, y_win], &push, gx, n_tok as u32)
            }
        }
    }

    pub(crate) fn dispatch_gemm(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkDenseMat,
        x_win: &VkBuffer,
        y_win: &VkBuffer,
        n_tok: usize,
        use_tc: bool,
    ) -> Result<(), VulkanError> {
        self.dispatch_gemm_inner(rec, mat, x_win, y_win, n_tok, use_tc, true)
    }

    pub(crate) fn dispatch_gemm_nb(
        &self,
        rec: &mut Recorder<'_>,
        mat: &VkDenseMat,
        x_win: &VkBuffer,
        y_win: &VkBuffer,
        n_tok: usize,
        use_tc: bool,
    ) -> Result<(), VulkanError> {
        self.dispatch_gemm_inner(rec, mat, x_win, y_win, n_tok, use_tc, false)
    }

    fn build_decode_segments(&self, out: &mut [GpuSegSpec; 64]) -> usize {
        let mut n_segs = 0usize;
        let mut cursor = 0usize;
        let mut need_embed = true;
        let mut pending_post_moe: Option<(usize, usize)> = None;
        let mut b = 0usize;
        let mut pb = 0usize;
        let mut bk = 0usize;

        while cursor < self.schedule.len() {
            let init_b = b;
            let init_pb = pb;
            let init_bk = bk;
            let start_cur = cursor;
            let mut stop_host = None;
            while cursor < self.schedule.len() {
                let st = self.schedule[cursor];
                self.advance_depth_counts(st, &mut b, &mut pb, &mut bk);
                cursor += 1;
                if self.layers[st.ip].vram_experts.is_none() {
                    stop_host = Some((st.il, st.ip));
                    break;
                }
            }
            let run_outputs = cursor == self.schedule.len() && stop_host.is_none();
            out[n_segs] = GpuSegSpec {
                need_embed,
                prev_host: pending_post_moe.take(),
                start_cur,
                end_cur: cursor,
                stop_host,
                run_outputs,
                init_b,
                init_pb,
                init_bk,
                end_b: b,
            };
            n_segs += 1;
            need_embed = false;
            pending_post_moe = stop_host;
        }

        if let Some(last_host) = pending_post_moe.take() {
            out[n_segs] = GpuSegSpec {
                need_embed: false,
                prev_host: Some(last_host),
                start_cur: cursor,
                end_cur: cursor,
                stop_host: None,
                run_outputs: true,
                init_b: b,
                init_pb: pb,
                init_bk: bk,
                end_b: b,
            };
            n_segs += 1;
        }
        n_segs
    }

    #[allow(clippy::too_many_arguments)]
    fn record_decode_segment(
        &self,
        rec: &mut Recorder<'_>,
        seg: GpuSegSpec,
        pos: usize,
        tau: f32,
        dynamic_pos: bool,
        compute_logits: bool,
        compute_argmax: bool,
        ts: &mut Option<TsRecorder<'_>>,
    ) -> Result<(), VulkanError> {
        if seg.need_embed {
            self.record_embed_and_norm(rec)?;
            TsRecorder::mark(ts, rec, STAGE_EMBED_STATE);
        }
        if let Some((prev_il, prev_ip)) = seg.prev_host {
            let k_gpu = self.layer_gpu_ext_experts(prev_ip);
            if k_gpu > 0 {
                let k_cpu = self.cfg.n_expert_used - k_gpu;
                if k_cpu > 0 {
                    self.record_vec_copy(
                        rec,
                        &self.wo_out_buf,
                        0,
                        &self.moe_out_bridge,
                        0,
                        self.cfg.n_embd,
                    )?;
                }
                self.record_ext_moe_reduce(
                    rec,
                    1,
                    k_cpu,
                    k_gpu,
                    &self.top_weights_tmp_buf,
                    &self.wo_out_buf,
                )?;
                self.record_layer_post_moe(rec, prev_il, prev_ip, true)?;
            } else {
                self.record_layer_post_moe(rec, prev_il, prev_ip, false)?;
            }
            TsRecorder::mark(ts, rec, STAGE_POST_MOE);
        }
        let mut b = seg.init_b;
        let mut pb = seg.init_pb;
        let mut bk = seg.init_bk;
        for &st in &self.schedule[seg.start_cur..seg.end_cur] {
            let in_vram = self.layers[st.ip].vram_experts.is_some();
            self.record_pre_step_transitions(
                rec,
                st.save_prelude_before,
                st.reset_pass_before,
                &mut b,
                &mut pb,
                &mut bk,
            )?;
            if in_vram {
                self.record_layer_pre_moe_into(
                    rec,
                    st.il,
                    st.ip,
                    pos,
                    tau,
                    dynamic_pos,
                    st.pass > 0,
                    &mut b,
                    bk,
                    &self.wo_out_buf,
                    &self.top_ids_tmp_buf,
                    &self.top_weights_tmp_buf,
                    ts,
                )?;
                self.record_vram_moe_from(
                    rec,
                    st.ip,
                    &self.wo_out_buf,
                    &self.top_ids_tmp_buf,
                    &self.top_weights_tmp_buf,
                )?;
                TsRecorder::mark(ts, rec, STAGE_VRAM_MOE);
                self.record_layer_post_moe(rec, st.il, st.ip, true)?;
                TsRecorder::mark(ts, rec, STAGE_POST_MOE);
            } else {
                self.record_layer_pre_moe(
                    rec,
                    st.il,
                    st.ip,
                    pos,
                    tau,
                    dynamic_pos,
                    st.pass > 0,
                    &mut b,
                    bk,
                    ts,
                )?;
            }
        }
        if seg.run_outputs {
            self.record_output_stage(rec, seg.end_b, compute_logits, compute_argmax)?;
            TsRecorder::mark(ts, rec, STAGE_LM_HEAD);
        }
        Ok(())
    }

    fn ensure_baked_decode_mode(
        &mut self,
        mode_idx: usize,
        segs: &[GpuSegSpec],
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<u64, VulkanError> {
        let mut rec_total_ns = 0u64;
        if self.baked_decode_cbs[mode_idx].is_none() {
            let mut vec = Vec::with_capacity(segs.len());
            for &seg in segs {
                let (baked, r_ns) = self.ctx.bake_cb(|rec| {
                    let mut no_ts = None;
                    self.record_decode_segment(
                        rec,
                        seg,
                        0,
                        1.0,
                        true,
                        compute_logits,
                        compute_argmax,
                        &mut no_ts,
                    )
                })?;
                rec_total_ns += r_ns;
                vec.push(baked);
            }
            self.baked_decode_cbs[mode_idx] = Some(vec);
        }
        for seg in segs {
            if let Some((_, h_ip)) = seg.stop_host {
                let k_gpu = self.layer_gpu_ext_experts(h_ip);
                if k_gpu > 0 && self.baked_ext_moe_cbs[h_ip].is_none() {
                    let k_cpu = self.cfg.n_expert_used - k_gpu;
                    let (baked_ext, r_ns) = self.ctx.bake_cb(|rec| {
                        self.record_ext_moe_tail(
                            rec,
                            h_ip,
                            1,
                            k_cpu,
                            k_gpu,
                            &self.wo_out_buf,
                            &self.top_ids_tmp_buf,
                        )
                    })?;
                    rec_total_ns += r_ns;
                    self.baked_ext_moe_cbs[h_ip] = Some(baked_ext);
                }
            }
        }
        Ok(rec_total_ns)
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

        let ratio = ((pos + 1) as f32) / (self.cfg.log_scale_floor as f32);
        let tau = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();
        write_mapped_u32(&self.ctx, &self.step_buf, &[token_id])?;
        write_mapped_u32(&self.ctx, &self.dyn_pos_buf, &[pos as u32])?;
        write_mapped_f32(&self.ctx, &self.win_taus_buf, &[tau])?;

        let n_embd = self.cfg.n_embd;
        let top_k = self.cfg.n_expert_used;
        if self.prof_on {
            self.prof.tokens += 1;
        }

        let mut segs = [GpuSegSpec {
            need_embed: false,
            prev_host: None,
            start_cur: 0,
            end_cur: 0,
            stop_host: None,
            run_outputs: false,
            init_b: 0,
            init_pb: 0,
            init_bk: 0,
            end_b: 0,
        }; 64];
        let n_segs = self.build_decode_segments(&mut segs);
        if n_segs == 0 {
            return Ok(());
        }

        // Fast zero-record path (`0.00 ms/tok` CB recording + concurrent GPU PCIe + CPU DDR MoE):
        if self.prof_qp.is_none() {
            let mode_idx = match (compute_logits, compute_argmax) {
                (false, _) => 0,
                (true, false) => 1,
                (true, true) => 2,
            };
            let bake_rec_ns = self.ensure_baked_decode_mode(
                mode_idx,
                &segs[..n_segs],
                compute_logits,
                compute_argmax,
            )?;
            if self.prof_on {
                self.prof.gpu_record_ns += bake_rec_ns;
            }

            let mut inflight = {
                let seg_cbs = self.baked_decode_cbs[mode_idx]
                    .as_ref()
                    .expect("baked_decode_cbs initialized");
                Some(self.ctx.submit_baked(&seg_cbs[0])?)
            };

            for idx in 0..n_segs {
                let inf = inflight.take().expect("active baked segment");
                let wait_ns = self.ctx.wait_baked(inf)?;
                if self.prof_on {
                    self.prof.gpu_submits += 1;
                    self.prof.gpu_wait_ns += wait_ns;
                }

                if let Some((_, h_ip)) = segs[idx].stop_host {
                    let k_gpu = self.layer_gpu_ext_experts(h_ip);
                    let k_cpu = top_k - k_gpu;
                    let ext_inf = if k_gpu > 0 {
                        let ext_cb = self.baked_ext_moe_cbs[h_ip]
                            .as_ref()
                            .expect("baked_ext_moe_cbs initialized");
                        Some(self.ctx.submit_baked(ext_cb)?)
                    } else {
                        None
                    };

                    if k_cpu > 0 {
                        let t_br0 = std::time::Instant::now();
                        let eval_fn = match host_moe.as_deref_mut() {
                            Some(f) => f,
                            None => {
                                if let Some(ei) = ext_inf {
                                    let _ = self.ctx.wait_baked(ei);
                                }
                                return Err(VulkanError::InvalidDimensions(format!(
                                    "layer {h_ip} MoE experts are offloaded from VRAM, but no HostMoeFn was supplied"
                                )));
                            }
                        };
                        let mut raw_ids = [0u32; 32];
                        if let Err(e) = read_mapped_f32(
                            &self.ctx,
                            &self.ffn_normed_bridge,
                            &mut self.host_cur_scratch[..n_embd],
                        )
                        .and_then(|()| read_mapped_u32(&self.top_ids_bridge, &mut raw_ids[..k_cpu]))
                        .and_then(|()| {
                            read_mapped_f32(
                                &self.ctx,
                                &self.top_weights_bridge,
                                &mut self.host_weights_scratch[..k_cpu],
                            )
                        }) {
                            if let Some(ei) = ext_inf {
                                let _ = self.ctx.wait_baked(ei);
                            }
                            return Err(e);
                        }
                        for (dst_id, &raw_id) in self.host_ids_scratch[..k_cpu]
                            .iter_mut()
                            .zip(&raw_ids[..k_cpu])
                        {
                            *dst_id = raw_id as usize;
                        }
                        let t_moe0 = std::time::Instant::now();
                        eval_fn(
                            h_ip,
                            &self.host_cur_scratch[..n_embd],
                            &self.host_ids_scratch[..k_cpu],
                            &self.host_weights_scratch[..k_cpu],
                            &mut self.host_moe_out_scratch[..n_embd],
                        );
                        let t_moe1 = std::time::Instant::now();
                        if let Err(e) = write_mapped_f32(
                            &self.ctx,
                            &self.moe_out_bridge,
                            &self.host_moe_out_scratch[..n_embd],
                        ) {
                            if let Some(ei) = ext_inf {
                                let _ = self.ctx.wait_baked(ei);
                            }
                            return Err(e);
                        }
                        if self.prof_on {
                            let moe_ns = (t_moe1 - t_moe0).as_nanos() as u64;
                            self.prof.host_moe_calls += 1;
                            self.prof.host_moe_ns += moe_ns;
                            self.prof.bridge_ns +=
                                (t_br0.elapsed().as_nanos() as u64).saturating_sub(moe_ns);
                        }
                    }

                    if let Some(ei) = ext_inf {
                        let ext_wait_ns = self.ctx.wait_baked(ei)?;
                        if self.prof_on {
                            self.prof.gpu_submits += 1;
                            self.prof.gpu_wait_ns += ext_wait_ns;
                        }
                    }
                }

                if idx + 1 < n_segs {
                    let seg_cbs = self.baked_decode_cbs[mode_idx]
                        .as_ref()
                        .expect("baked_decode_cbs initialized");
                    inflight = Some(self.ctx.submit_baked(&seg_cbs[idx + 1])?);
                }
            }
            return Ok(());
        }

        // Detailed `TM_VK_PROF=2` hardware timestamp path (2-slot pipelined dynamic CB recording):
        let qp = self.prof_qp;
        let mut spans_slots = core::mem::take(&mut self.prof_spans_slots);
        let mut n_queries_slots = [0u32; 2];

        let prep0 = self.ctx.prepare_cb(0, |rec| {
            let mut ts = TsRecorder::begin(rec, qp, &mut spans_slots[0]);
            self.record_decode_segment(
                rec,
                segs[0],
                pos,
                tau,
                false,
                compute_logits,
                compute_argmax,
                &mut ts,
            )?;
            n_queries_slots[0] = ts.map_or(0, |t| t.next_q);
            Ok(())
        });
        let prep0 = match prep0 {
            Ok(p) => p,
            Err(e) => {
                self.prof_spans_slots = spans_slots;
                return Err(e);
            }
        };
        let mut inflight = Some(self.ctx.submit_prepared(prep0)?);

        let mut loop_res: Result<(), VulkanError> = Ok(());
        for idx in 0..n_segs {
            let cur_slot = idx & 1;
            let next_slot = (idx + 1) & 1;
            let next_prep = if idx + 1 < n_segs {
                match self.ctx.prepare_cb(next_slot, |rec| {
                    let mut ts = TsRecorder::begin(rec, qp, &mut spans_slots[next_slot]);
                    self.record_decode_segment(
                        rec,
                        segs[idx + 1],
                        pos,
                        tau,
                        false,
                        compute_logits,
                        compute_argmax,
                        &mut ts,
                    )?;
                    n_queries_slots[next_slot] = ts.map_or(0, |t| t.next_q);
                    Ok(())
                }) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        if let Some(inf) = inflight.take() {
                            let _ = self.ctx.wait_in_flight(inf);
                        }
                        loop_res = Err(e);
                        break;
                    }
                }
            } else {
                None
            };

            let inf = inflight.take().expect("active in-flight segment");
            let sub_t = match self.ctx.wait_in_flight(inf) {
                Ok(t) => t,
                Err(e) => {
                    loop_res = Err(e);
                    break;
                }
            };
            self.prof_finish_submit(sub_t, n_queries_slots[cur_slot], &spans_slots[cur_slot]);

            if let Some((_, h_ip)) = segs[idx].stop_host {
                let k_gpu = self.layer_gpu_ext_experts(h_ip);
                let k_cpu = top_k - k_gpu;
                let ext_inf = if k_gpu > 0 {
                    match self
                        .ctx
                        .prepare_cb(cur_slot, |rec| {
                            self.record_ext_moe_tail(
                                rec,
                                h_ip,
                                1,
                                k_cpu,
                                k_gpu,
                                &self.wo_out_buf,
                                &self.top_ids_tmp_buf,
                            )
                        })
                        .and_then(|p| self.ctx.submit_prepared(p))
                    {
                        Ok(ei) => Some(ei),
                        Err(e) => {
                            loop_res = Err(e);
                            break;
                        }
                    }
                } else {
                    None
                };

                if k_cpu > 0 {
                    let t_br0 = std::time::Instant::now();
                    let eval_fn = match host_moe.as_deref_mut() {
                        Some(f) => f,
                        None => {
                            if let Some(ei) = ext_inf {
                                let _ = self.ctx.wait_in_flight(ei);
                            }
                            loop_res = Err(VulkanError::InvalidDimensions(format!(
                                "layer {h_ip} MoE experts are offloaded from VRAM, but no HostMoeFn was supplied"
                            )));
                            break;
                        }
                    };
                    let mut raw_ids = [0u32; 32];
                    if let Err(e) = read_mapped_f32(
                        &self.ctx,
                        &self.ffn_normed_bridge,
                        &mut self.host_cur_scratch[..n_embd],
                    )
                    .and_then(|()| read_mapped_u32(&self.top_ids_bridge, &mut raw_ids[..k_cpu]))
                    .and_then(|()| {
                        read_mapped_f32(
                            &self.ctx,
                            &self.top_weights_bridge,
                            &mut self.host_weights_scratch[..k_cpu],
                        )
                    }) {
                        if let Some(ei) = ext_inf {
                            let _ = self.ctx.wait_in_flight(ei);
                        }
                        loop_res = Err(e);
                        break;
                    }
                    for (dst_id, &raw_id) in self.host_ids_scratch[..k_cpu]
                        .iter_mut()
                        .zip(&raw_ids[..k_cpu])
                    {
                        *dst_id = raw_id as usize;
                    }
                    let t_moe0 = std::time::Instant::now();
                    eval_fn(
                        h_ip,
                        &self.host_cur_scratch[..n_embd],
                        &self.host_ids_scratch[..k_cpu],
                        &self.host_weights_scratch[..k_cpu],
                        &mut self.host_moe_out_scratch[..n_embd],
                    );
                    let t_moe1 = std::time::Instant::now();
                    if let Err(e) = write_mapped_f32(
                        &self.ctx,
                        &self.moe_out_bridge,
                        &self.host_moe_out_scratch[..n_embd],
                    ) {
                        if let Some(ei) = ext_inf {
                            let _ = self.ctx.wait_in_flight(ei);
                        }
                        loop_res = Err(e);
                        break;
                    }
                    if self.prof_on {
                        let moe_ns = (t_moe1 - t_moe0).as_nanos() as u64;
                        self.prof.host_moe_calls += 1;
                        self.prof.host_moe_ns += moe_ns;
                        self.prof.bridge_ns +=
                            (t_br0.elapsed().as_nanos() as u64).saturating_sub(moe_ns);
                    }
                }

                if let Some(ei) = ext_inf {
                    match self.ctx.wait_in_flight(ei) {
                        Ok(ext_t) => self.prof_finish_submit(ext_t, 0, &[]),
                        Err(e) => {
                            loop_res = Err(e);
                            break;
                        }
                    }
                }
            }

            if let Some(prep) = next_prep {
                match self.ctx.submit_prepared(prep) {
                    Ok(inf_next) => inflight = Some(inf_next),
                    Err(e) => {
                        loop_res = Err(e);
                        break;
                    }
                }
            }
        }

        self.prof_spans_slots = spans_slots;
        loop_res
    }

    pub(crate) fn record_embed_and_norm(&self, rec: &mut Recorder<'_>) -> Result<(), VulkanError> {
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

    fn record_vec_copy_nb(
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
        rec.dispatch_nb(
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
        let cin = &self.ctrl_in_bufs[0];
        let cg = &self.ctrl_g_bufs[0];
        let cgain = &self.ctrl_gain_bufs[0];
        let cforget = &self.ctrl_forget_bufs[0];
        // `ctrl_in[0..n_embd] = rms_norm_unit(cur, 1e-6)` and `ctrl_in[n_embd..2*n_embd] = rms_norm_unit(msg, 1e-6)`
        rec.dispatch_nb(
            &self.pipe_ops,
            &[
                cur_buf,
                cin,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, n_embd, cur_off, 0, 0, 1e-6f32.to_bits(), 0, 0],
            1,
        )?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.messages_buf,
                cin,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, n_embd, msg_off, n_embd, 0, 1e-6f32.to_bits(), 0, 0],
            1,
        )?;
        self.dispatch_gemv(rec, &loop_w.ctrl_down[ctrl], cin, cg)?;
        self.dispatch_gemv_nb(rec, &loop_w.ctrl_gain[ctrl], cg, cgain)?;
        self.dispatch_gemv(rec, &loop_w.ctrl_forget[ctrl], cg, cforget)?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                cur_buf,
                &self.messages_buf,
                cgain,
                cforget,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[4, n_embd, cur_off, msg_off, 0, 0, 0, 0],
            self.elem_groups(self.cfg.n_embd),
        )
    }

    /// Executes all 3 cross-pass loop controllers (`Q, K, V`) concurrently in 4 barrier waves
    /// instead of 12 serial waves (`3x` SM occupancy on `ctrl_down`).
    fn record_controls_qkv(
        &self,
        rec: &mut Recorder<'_>,
        loop_w: &VkLoopLayer,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd as u32;
        for j in 0..3 {
            let off = (j as u32) * n_embd;
            let cin = &self.ctrl_in_bufs[j];
            rec.dispatch_nb(
                &self.pipe_ops,
                &[
                    &self.inputs_buf,
                    cin,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[0, n_embd, off, 0, 0, 1e-6f32.to_bits(), 0, 0],
                1,
            )?;
            let args_m = [0, n_embd, off, n_embd, 0, 1e-6f32.to_bits(), 0, 0];
            let bufs_m = [
                &self.messages_buf,
                cin,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ];
            if j < 2 {
                rec.dispatch_nb(&self.pipe_ops, &bufs_m, &args_m, 1)?;
            } else {
                rec.dispatch(&self.pipe_ops, &bufs_m, &args_m, 1)?;
            }
        }
        for j in 0..3 {
            let cd = &loop_w.ctrl_down[LOOP_CTRL_Q + j];
            if j < 2 {
                self.dispatch_gemv_nb(rec, cd, &self.ctrl_in_bufs[j], &self.ctrl_g_bufs[j])?;
            } else {
                self.dispatch_gemv(rec, cd, &self.ctrl_in_bufs[j], &self.ctrl_g_bufs[j])?;
            }
        }
        for j in 0..3 {
            let cg = &loop_w.ctrl_gain[LOOP_CTRL_Q + j];
            let cf = &loop_w.ctrl_forget[LOOP_CTRL_Q + j];
            self.dispatch_gemv_nb(rec, cg, &self.ctrl_g_bufs[j], &self.ctrl_gain_bufs[j])?;
            if j < 2 {
                self.dispatch_gemv_nb(rec, cf, &self.ctrl_g_bufs[j], &self.ctrl_forget_bufs[j])?;
            } else {
                self.dispatch_gemv(rec, cf, &self.ctrl_g_bufs[j], &self.ctrl_forget_bufs[j])?;
            }
        }
        let groups = self.elem_groups(self.cfg.n_embd);
        for j in 0..3 {
            let off = (j as u32) * n_embd;
            let bufs = [
                &self.inputs_buf,
                &self.messages_buf,
                &self.ctrl_gain_bufs[j],
                &self.ctrl_forget_bufs[j],
                &self.dummy_buf,
                &self.dummy_buf,
            ];
            let args = [4, n_embd, off, off, 0, 0, 0, 0];
            if j < 2 {
                rec.dispatch_nb(&self.pipe_ops, &bufs, &args, groups)?;
            } else {
                rec.dispatch(&self.pipe_ops, &bufs, &args, groups)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_layer_pre_moe(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        dynamic_pos: bool,
        has_messages: bool,
        blocks_len: &mut usize,
        bank_len: usize,
        ts: &mut Option<TsRecorder<'_>>,
    ) -> Result<(), VulkanError> {
        let k_gpu = self.layer_gpu_ext_experts(ip);
        if k_gpu > 0 {
            self.record_layer_pre_moe_into(
                rec,
                il,
                ip,
                pos,
                tau,
                dynamic_pos,
                has_messages,
                blocks_len,
                bank_len,
                &self.wo_out_buf,
                &self.top_ids_tmp_buf,
                &self.top_weights_tmp_buf,
                ts,
            )?;
            let k_cpu = self.cfg.n_expert_used - k_gpu;
            if k_cpu > 0 {
                self.record_vec_copy(
                    rec,
                    &self.ffn_normed_bridge,
                    0,
                    &self.wo_out_buf,
                    0,
                    self.cfg.n_embd,
                )?;
                self.record_buf_copy_bytes(
                    rec,
                    &self.top_ids_bridge,
                    0,
                    &self.top_ids_tmp_buf,
                    0,
                    (k_cpu * 4) as u64,
                );
                self.record_buf_copy_bytes(
                    rec,
                    &self.top_weights_bridge,
                    0,
                    &self.top_weights_tmp_buf,
                    0,
                    (k_cpu * 4) as u64,
                );
            }
            Ok(())
        } else {
            self.record_layer_pre_moe_into(
                rec,
                il,
                ip,
                pos,
                tau,
                dynamic_pos,
                has_messages,
                blocks_len,
                bank_len,
                &self.ffn_normed_bridge,
                &self.top_ids_bridge,
                &self.top_weights_bridge,
                ts,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_layer_pre_moe_into(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        dynamic_pos: bool,
        has_messages: bool,
        blocks_len: &mut usize,
        bank_len: usize,
        ffn_norm_dst: &VkBuffer,
        top_ids_dst: &VkBuffer,
        top_w_dst: &VkBuffer,
        ts: &mut Option<TsRecorder<'_>>,
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
                self.record_vec_copy_nb(
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
            self.record_vec_copy_nb(rec, &self.inputs_buf, 0, &self.cur_buf, 0, n_embd)?;
            self.record_vec_copy_nb(
                rec,
                &self.inputs_buf,
                n_embd as u32,
                &self.cur_buf,
                0,
                n_embd,
            )?;
            self.record_vec_copy(
                rec,
                &self.inputs_buf,
                (2 * n_embd) as u32,
                &self.cur_buf,
                0,
                n_embd,
            )?;
        }

        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            self.record_controls_qkv(rec, loop_w)?;
        }

        // Input RMSNorm into `q_in_buf`, `k_in_buf`, `v_in_buf`
        if swa && has_messages {
            for (j, dst_b) in [&self.q_in_buf, &self.k_in_buf, &self.v_in_buf]
                .into_iter()
                .enumerate()
            {
                let args = [
                    0,
                    n_embd as u32,
                    (j * n_embd) as u32,
                    0,
                    1,
                    self.cfg.rms_eps.to_bits(),
                    0,
                    layer.attn_norm_off,
                ];
                let bufs = [
                    &self.inputs_buf,
                    dst_b,
                    &self.f32_params,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ];
                if j < 2 {
                    rec.dispatch_nb(&self.pipe_ops, &bufs, &args, 1)?;
                } else {
                    rec.dispatch(&self.pipe_ops, &bufs, &args, 1)?;
                }
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
            self.record_vec_copy_nb(rec, &self.k_in_buf, 0, &self.q_in_buf, 0, n_embd)?;
            self.record_vec_copy(rec, &self.v_in_buf, 0, &self.q_in_buf, 0, n_embd)?;
        } else {
            let pairs = [
                (0usize, &self.q_in_buf, layer.q_inp_norm_off),
                (1usize, &self.k_in_buf, layer.k_inp_norm_off),
                (2usize, &self.v_in_buf, layer.v_inp_norm_off),
            ];
            for (j, dst_b, norm_off) in pairs {
                let args = [
                    0,
                    n_embd as u32,
                    (j * n_embd) as u32,
                    0,
                    1,
                    self.cfg.rms_eps.to_bits(),
                    0,
                    norm_off,
                ];
                let bufs = [
                    &self.inputs_buf,
                    dst_b,
                    &self.f32_params,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ];
                if j < 2 {
                    rec.dispatch_nb(&self.pipe_ops, &bufs, &args, 1)?;
                } else {
                    rec.dispatch(&self.pipe_ops, &bufs, &args, 1)?;
                }
            }
        }
        TsRecorder::mark(ts, rec, STAGE_LOOP_MIX_NORM);

        // Self-Attention projections (all read `q_in`/`k_in`/`v_in` and write disjoint buffers)
        self.dispatch_gemv_nb(rec, &layer.wq, &self.q_in_buf, &self.q_buf)?;
        self.dispatch_gemv_nb(rec, &layer.wk, &self.k_in_buf, &self.k_buf)?;
        self.dispatch_gemv_nb(rec, &layer.wv, &self.v_in_buf, &self.v_buf)?;
        self.dispatch_gemv_nb(rec, &layer.attn_rel, &self.q_in_buf, &self.r_buf)?;
        self.dispatch_gemv_nb(rec, &layer.attn_sink_proj, &self.q_in_buf, &self.sinks_buf)?;
        if swa {
            self.dispatch_gemv(rec, &layer.wqkv_gate, &self.q_in_buf, &self.gate_buf)?;
        } else {
            self.dispatch_gemv_nb(rec, &layer.wqkv_gate, &self.q_in_buf, &self.gate_buf)?;
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            self.dispatch_gemv_nb(rec, kp, &self.k_in_buf, &self.k_dyn_buf)?;
            self.dispatch_gemv(rec, vp, &self.v_in_buf, &self.v_dyn_buf)?;
        }

        let dyn_heads = if swa { 0u32 } else { nh_kv as u32 };
        rec.dispatch_nb(
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
        let use_tau_buf = u32::from(dynamic_pos && !swa);
        let tau_buf_ref = if use_tau_buf != 0 {
            &self.win_taus_buf
        } else {
            &self.dummy_buf
        };
        // Qcur clip + Kcur clip + Vcur unit-RMSNorm + Rcur clip (4 disjoint buffers, 1 barrier)
        rec.dispatch_nb(
            &self.pipe_ops,
            &[
                &self.q_buf,
                &self.f32_params,
                tau_buf_ref,
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
                use_tau_buf,
            ],
            nh as u32,
        )?;
        rec.dispatch_nb(
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
        rec.dispatch_nb(
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
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.r_buf,
                &self.f32_params,
                tau_buf_ref,
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
                use_tau_buf,
            ],
            nh as u32,
        )?;

        // Pack `k_buf` and `v_in_buf` (unit-normed Vcur) into F16 `k_cache` & `v_cache` at `(il, pos)`
        let kv_pairs = kv_dim / 2;
        let (dst_u32_off, pack_dyn, pack_pos_buf) = if dynamic_pos {
            (self.slot_kv_u32_bases[il], 1u32, &self.dyn_pos_buf)
        } else {
            (
                self.slot_kv_u32_bases[il] + (pos * kv_pairs) as u32,
                0u32,
                &self.dummy_buf,
            )
        };
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.k_buf,
                &self.v_in_buf,
                pack_pos_buf,
                &self.dummy_buf,
                &self.k_cache,
                &self.v_cache,
            ],
            &[5, kv_pairs as u32, dst_u32_off, pack_dyn, 0, 0, 0, 0],
            self.elem_groups(kv_pairs),
        )?;

        // Relative-bias + zero-sink GQA attention
        let (pc_pos, pc_kv_start) = if dynamic_pos {
            (
                0xFFFF_FFFFu32,
                if swa { self.cfg.n_swa as u32 } else { 0u32 },
            )
        } else {
            let kv_s = if swa {
                (pos + 1).saturating_sub(self.cfg.n_swa)
            } else {
                0
            };
            (pos as u32, kv_s as u32)
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
                &self.dyn_pos_buf,
            ],
            &[
                (nh as u32) | ((nh_kv as u32) << 16),
                (hd as u32) | ((n_rel as u32) << 16),
                extent as u32,
                pc_pos,
                pc_kv_start,
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

        // Block residual stream boundary (`ip % block_size == 0`) — save `cur_buf` BEFORE overwriting it!
        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            self.record_vec_copy(
                rec,
                &self.blocks_buf,
                (*blocks_len * n_embd) as u32,
                &self.cur_buf,
                0,
                n_embd,
            )?;
            *blocks_len += 1;
            self.record_vec_copy_nb(rec, &self.cur_buf, 0, &self.wo_out_buf, 0, n_embd)?;
            self.record_vec_copy(rec, &self.ffn_inp_buf, 0, &self.wo_out_buf, 0, n_embd)?;
        } else {
            self.record_vec_add(rec, &self.cur_buf, &self.cur_buf, &self.wo_out_buf, n_embd)?;
            self.record_vec_copy(rec, &self.ffn_inp_buf, 0, &self.cur_buf, 0, n_embd)?;
        }
        TsRecorder::mark(ts, rec, STAGE_ATTN_DENSE);

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
        )?;
        TsRecorder::mark(ts, rec, STAGE_FFN_ROUTER);
        Ok(())
    }

    fn record_vram_moe_from(
        &self,
        rec: &mut Recorder<'_>,
        ip: usize,
        x_buf: &VkBuffer,
        top_ids_buf: &VkBuffer,
        top_w_buf: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let vexp = self.layers[ip]
            .vram_experts
            .as_ref()
            .expect("record_vram_moe requires vram_experts");
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        let top_k = self.cfg.n_expert_used;
        let sg = self.ctx.info().subgroup_size;
        let rows_per_wg = ((self.moe_wg / sg) * self.moe_nr) as usize;

        let use_moe_dp4a = self.dp4a_q8_enabled
            && n_embd >= self.dp4a_min_dim
            && n_ff >= self.dp4a_min_dim
            && n_embd.is_multiple_of(32)
            && n_ff.is_multiple_of(32);

        // 1. Indexed `gate_up` `Q8_0` GEMV (`[2 * n_ff]` rows x `top_k` experts)
        if use_moe_dp4a && let Some(ref p_dp4a) = self.pipe_moe_dp4a {
            self.record_quant_q8_1(rec, x_buf, &self.xq_buf, &self.xd_buf, n_embd)?;
            rec.dispatch_xy(
                p_dp4a,
                &[
                    &vexp.gate_up,
                    &self.xq_buf,
                    &self.xd_buf,
                    top_ids_buf,
                    &self.exp_gate_up_buf,
                ],
                &[
                    1,
                    n_embd as u32,
                    (2 * n_ff) as u32,
                    vexp.gate_up_row_u16s,
                    vexp.gate_up_expert_u16s,
                    top_k as u32,
                    0,
                    0,
                ],
                (2 * n_ff).div_ceil(rows_per_wg) as u32,
                top_k as u32,
            )?;
        } else {
            rec.dispatch_xy(
                &self.pipe_moe_ops,
                &[
                    &vexp.gate_up,
                    x_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    &self.exp_gate_up_buf,
                ],
                &[
                    1,
                    n_embd as u32,
                    (2 * n_ff) as u32,
                    vexp.gate_up_row_u16s,
                    vexp.gate_up_expert_u16s,
                    top_k as u32,
                    0,
                    0,
                ],
                (2 * n_ff).div_ceil(rows_per_wg) as u32,
                top_k as u32,
            )?;
        }
        // 2. SiTU-GLU (`[top_k * n_ff]`)
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_gate_up_buf,
                &self.dummy_buf,
                top_ids_buf,
                top_w_buf,
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
        if use_moe_dp4a && let Some(ref p_dp4a) = self.pipe_moe_dp4a {
            self.record_quant_q8_1(
                rec,
                &self.exp_hidden_buf,
                &self.xq_buf,
                &self.xd_buf,
                top_k * n_ff,
            )?;
            rec.dispatch_xy(
                p_dp4a,
                &[
                    &vexp.down,
                    &self.xq_buf,
                    &self.xd_buf,
                    top_ids_buf,
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
        } else {
            rec.dispatch_xy(
                &self.pipe_moe_ops,
                &[
                    &vexp.down,
                    &self.exp_hidden_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
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
        }
        // 4. Weighted reduction into `wo_out_buf`
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_down_buf,
                &self.dummy_buf,
                top_ids_buf,
                top_w_buf,
                &self.wo_out_buf,
            ],
            &[3, n_embd as u32, top_k as u32, 0, 0, 0, 0, 0],
            n_embd.div_ceil(self.moe_wg as usize) as u32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_ext_moe_tail(
        &self,
        rec: &mut Recorder<'_>,
        ip: usize,
        w: usize,
        k_cpu: usize,
        k_gpu: usize,
        x_buf: &VkBuffer,
        top_ids_buf: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let vexp = &self.layers[ip]
            .host_imported_experts
            .as_ref()
            .expect("record_ext_moe_tail requires host_imported_experts")
            .vexp;
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        let top_k = self.cfg.n_expert_used;
        let total_gpu_k = w * k_gpu;
        let pack_u7 = ((top_k as u32) << 16) | (k_gpu as u32);
        let sg = self.ctx.info().subgroup_size;
        let rows_per_wg = ((self.moe_wg / sg) * self.moe_nr) as usize;
        let use_moe_dp4a = self.dp4a_q8_enabled
            && n_embd >= self.dp4a_min_dim
            && n_ff >= self.dp4a_min_dim
            && n_embd.is_multiple_of(32)
            && n_ff.is_multiple_of(32);

        if use_moe_dp4a && let Some(ref p_dp4a) = self.pipe_moe_dp4a {
            self.record_quant_q8_1(rec, x_buf, &self.ext_xq_buf, &self.ext_xd_buf, w * n_embd)?;
            rec.dispatch_xy(
                p_dp4a,
                &[
                    &vexp.gate_up,
                    &self.ext_xq_buf,
                    &self.ext_xd_buf,
                    top_ids_buf,
                    &self.exp_gate_up_buf,
                ],
                &[
                    1,
                    n_embd as u32,
                    (2 * n_ff) as u32,
                    vexp.gate_up_row_u16s,
                    vexp.gate_up_expert_u16s,
                    top_k as u32,
                    k_cpu as u32,
                    pack_u7,
                ],
                (2 * n_ff).div_ceil(rows_per_wg) as u32,
                total_gpu_k as u32,
            )?;
        } else {
            rec.dispatch_xy(
                &self.pipe_moe_ops,
                &[
                    &vexp.gate_up,
                    x_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    &self.dummy_buf,
                    &self.exp_gate_up_buf,
                ],
                &[
                    1,
                    n_embd as u32,
                    (2 * n_ff) as u32,
                    vexp.gate_up_row_u16s,
                    vexp.gate_up_expert_u16s,
                    top_k as u32,
                    k_cpu as u32,
                    pack_u7,
                ],
                (2 * n_ff).div_ceil(rows_per_wg) as u32,
                total_gpu_k as u32,
            )?;
        }
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_gate_up_buf,
                &self.dummy_buf,
                top_ids_buf,
                &self.dummy_buf,
                &self.exp_hidden_buf,
            ],
            &[
                2,
                (total_gpu_k * n_ff) as u32,
                n_ff as u32,
                0,
                0,
                0,
                self.cfg.situ_beta.to_bits(),
                self.cfg.situ_linear_beta.to_bits(),
            ],
            (total_gpu_k * n_ff).div_ceil(self.moe_wg as usize) as u32,
        )?;
        if use_moe_dp4a && let Some(ref p_dp4a) = self.pipe_moe_dp4a {
            self.record_quant_q8_1(
                rec,
                &self.exp_hidden_buf,
                &self.ext_xq_buf,
                &self.ext_xd_buf,
                total_gpu_k * n_ff,
            )?;
            rec.dispatch_xy(
                p_dp4a,
                &[
                    &vexp.down,
                    &self.ext_xq_buf,
                    &self.ext_xd_buf,
                    top_ids_buf,
                    &self.exp_down_buf,
                ],
                &[
                    1,
                    n_ff as u32,
                    n_embd as u32,
                    vexp.down_row_u16s,
                    vexp.down_expert_u16s,
                    0,
                    k_cpu as u32,
                    pack_u7,
                ],
                n_embd.div_ceil(rows_per_wg) as u32,
                total_gpu_k as u32,
            )
        } else {
            rec.dispatch_xy(
                &self.pipe_moe_ops,
                &[
                    &vexp.down,
                    &self.exp_hidden_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    &self.dummy_buf,
                    &self.exp_down_buf,
                ],
                &[
                    1,
                    n_ff as u32,
                    n_embd as u32,
                    vexp.down_row_u16s,
                    vexp.down_expert_u16s,
                    0,
                    k_cpu as u32,
                    pack_u7,
                ],
                n_embd.div_ceil(rows_per_wg) as u32,
                total_gpu_k as u32,
            )
        }
    }

    fn record_ext_moe_reduce(
        &self,
        rec: &mut Recorder<'_>,
        w: usize,
        k_cpu: usize,
        k_gpu: usize,
        top_w_buf: &VkBuffer,
        dst_buf: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let add_to_y = u32::from(k_cpu > 0);
        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_down_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                top_w_buf,
                dst_buf,
            ],
            &[
                3,
                n_embd as u32,
                k_gpu as u32,
                add_to_y,
                0,
                0,
                k_cpu as u32,
                self.cfg.n_expert_used as u32,
            ],
            n_embd.div_ceil(self.moe_wg as usize) as u32,
            w as u32,
        )
    }

    #[inline]
    fn window_gpu_experts(&self, ip: usize, allow_host_imported: bool) -> Option<&VkVramExperts> {
        let layer = &self.layers[ip];
        layer.vram_experts.as_ref().or_else(|| {
            if allow_host_imported {
                layer.host_imported_experts.as_ref().map(|h| &h.vexp)
            } else {
                None
            }
        })
    }

    /// Batched `w`-token VRAM / Zero-Copy PCIe MoE using GPU expert-grouped `BM=16` TensorCore
    /// (`CoopMat F16`) or LDS-tiled `Q8_0` GEMM when `w > 8`, or `0-ULP` register-grouped GEMV (`w <= 8`).
    #[allow(clippy::too_many_arguments)]
    fn record_vram_moe_window(
        &self,
        rec: &mut Recorder<'_>,
        ip: usize,
        w: usize,
        use_grouped: bool,
        x_win_buf: &VkBuffer,
        top_ids_buf: &VkBuffer,
        top_w_buf: &VkBuffer,
        out_win_buf: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let vexp = self.layers[ip]
            .vram_experts
            .as_ref()
            .expect("record_vram_moe_window requires vram_experts");
        self.record_moe_window_with(
            rec,
            vexp,
            w,
            use_grouped,
            x_win_buf,
            top_ids_buf,
            top_w_buf,
            out_win_buf,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_moe_window_with(
        &self,
        rec: &mut Recorder<'_>,
        vexp: &VkVramExperts,
        w: usize,
        use_grouped: bool,
        x_win_buf: &VkBuffer,
        top_ids_buf: &VkBuffer,
        top_w_buf: &VkBuffer,
        out_win_buf: &VkBuffer,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        let top_k = self.cfg.n_expert_used;
        let total_k = w * top_k;

        if use_grouped
            && w > 8
            && n_embd.is_multiple_of(32)
            && n_ff.is_multiple_of(32)
            && let Some(ref p_grp) = self.pipe_moe_grouped
        {
            let bm = self.moe_grouped_bm;
            let bn = self.moe_grouped_bn;
            let max_tiles = total_k.min(self.cfg.n_expert + total_k.div_ceil(bm)) as u32;

            // 1. Build `BM` expert-grouped tile descriptors on GPU (`1` workgroup)
            rec.dispatch(
                &self.pipe_moe_ops,
                &[
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    &self.moe_tiles_buf,
                ],
                &[
                    4,
                    self.cfg.n_expert as u32,
                    total_k as u32,
                    bm as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                1,
            )?;
            // 2. Expert-grouped `gate_up` (`CoopMat F16` on T4/RDNA4, `LDS Tiled` fallback)
            rec.dispatch_xy(
                p_grp,
                &[
                    &vexp.gate_up,
                    x_win_buf,
                    &self.moe_tiles_buf,
                    &self.exp_gate_up_buf,
                ],
                &[
                    n_embd as u32,
                    (2 * n_ff) as u32,
                    vexp.gate_up_row_u16s,
                    vexp.gate_up_expert_u16s,
                    top_k as u32,
                ],
                (2 * n_ff).div_ceil(bn) as u32,
                max_tiles,
            )?;
            // 3. Batched SiTU-GLU (`[w * top_k * n_ff]`)
            rec.dispatch(
                &self.pipe_moe_ops,
                &[
                    &self.dummy_buf,
                    &self.exp_gate_up_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    &self.exp_hidden_buf,
                ],
                &[
                    2,
                    (total_k * n_ff) as u32,
                    n_ff as u32,
                    0,
                    0,
                    0,
                    self.cfg.situ_beta.to_bits(),
                    self.cfg.situ_linear_beta.to_bits(),
                ],
                (total_k * n_ff).div_ceil(self.moe_wg as usize) as u32,
            )?;
            // 4. Expert-grouped `down` (`CoopMat F16` on T4/RDNA4, `LDS Tiled` fallback)
            rec.dispatch_xy(
                p_grp,
                &[
                    &vexp.down,
                    &self.exp_hidden_buf,
                    &self.moe_tiles_buf,
                    &self.exp_down_buf,
                ],
                &[
                    n_ff as u32,
                    n_embd as u32,
                    vexp.down_row_u16s,
                    vexp.down_expert_u16s,
                    0,
                ],
                n_embd.div_ceil(bn) as u32,
                max_tiles,
            )?;
            // 5. Batched deterministic `0..top_k` reduction into `out_win_buf` (`[w, n_embd]`)
            return rec.dispatch_xy(
                &self.pipe_moe_ops,
                &[
                    &self.dummy_buf,
                    &self.exp_down_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    out_win_buf,
                ],
                &[3, n_embd as u32, top_k as u32, 0, 0, 0, 0, 0],
                n_embd.div_ceil(self.moe_wg as usize) as u32,
                w as u32,
            );
        }

        let sg = self.ctx.info().subgroup_size;
        let rows_per_wg = ((self.moe_wg / sg) * self.moe_nr) as usize;

        if (2..=8).contains(&w) {
            // Exact `0-ULP` expert-deduplicated register GEMV (`op=4` + `op=5`, `tile_cnt <= 8`):
            // reads every shared expert ONCE across the `w`-token verify window while remaining
            // bit-for-bit identical to single-token `op=1`!
            let use_moe_dp4a = self.dp4a_q8_enabled
                && n_embd >= self.dp4a_min_dim
                && n_ff >= self.dp4a_min_dim
                && n_embd.is_multiple_of(32)
                && n_ff.is_multiple_of(32);
            let max_tiles = total_k.min(self.cfg.n_expert + total_k.div_ceil(8)) as u32;
            rec.dispatch(
                &self.pipe_moe_ops,
                &[
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    &self.moe_tiles_buf,
                ],
                &[4, self.cfg.n_expert as u32, total_k as u32, 8, 0, 0, 0, 0],
                1,
            )?;
            if use_moe_dp4a && let Some(ref p_dp4a) = self.pipe_moe_dp4a {
                self.record_quant_q8_1(rec, x_win_buf, &self.xq_buf, &self.xd_buf, w * n_embd)?;
                rec.dispatch_xy(
                    p_dp4a,
                    &[
                        &vexp.gate_up,
                        &self.xq_buf,
                        &self.xd_buf,
                        &self.moe_tiles_buf,
                        &self.exp_gate_up_buf,
                    ],
                    &[
                        5,
                        n_embd as u32,
                        (2 * n_ff) as u32,
                        vexp.gate_up_row_u16s,
                        vexp.gate_up_expert_u16s,
                        top_k as u32,
                        0,
                        0,
                    ],
                    (2 * n_ff).div_ceil(rows_per_wg) as u32,
                    max_tiles,
                )?;
            } else {
                rec.dispatch_xy(
                    &self.pipe_moe_ops,
                    &[
                        &vexp.gate_up,
                        x_win_buf,
                        &self.dummy_buf,
                        &self.moe_tiles_buf,
                        top_w_buf,
                        &self.exp_gate_up_buf,
                    ],
                    &[
                        5,
                        n_embd as u32,
                        (2 * n_ff) as u32,
                        vexp.gate_up_row_u16s,
                        vexp.gate_up_expert_u16s,
                        top_k as u32,
                        0,
                        0,
                    ],
                    (2 * n_ff).div_ceil(rows_per_wg) as u32,
                    max_tiles,
                )?;
            }
            rec.dispatch(
                &self.pipe_moe_ops,
                &[
                    &self.dummy_buf,
                    &self.exp_gate_up_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    &self.exp_hidden_buf,
                ],
                &[
                    2,
                    (total_k * n_ff) as u32,
                    n_ff as u32,
                    0,
                    0,
                    0,
                    self.cfg.situ_beta.to_bits(),
                    self.cfg.situ_linear_beta.to_bits(),
                ],
                (total_k * n_ff).div_ceil(self.moe_wg as usize) as u32,
            )?;
            if use_moe_dp4a && let Some(ref p_dp4a) = self.pipe_moe_dp4a {
                self.record_quant_q8_1(
                    rec,
                    &self.exp_hidden_buf,
                    &self.xq_buf,
                    &self.xd_buf,
                    total_k * n_ff,
                )?;
                rec.dispatch_xy(
                    p_dp4a,
                    &[
                        &vexp.down,
                        &self.xq_buf,
                        &self.xd_buf,
                        &self.moe_tiles_buf,
                        &self.exp_down_buf,
                    ],
                    &[
                        5,
                        n_ff as u32,
                        n_embd as u32,
                        vexp.down_row_u16s,
                        vexp.down_expert_u16s,
                        0,
                        0,
                        0,
                    ],
                    n_embd.div_ceil(rows_per_wg) as u32,
                    max_tiles,
                )?;
            } else {
                rec.dispatch_xy(
                    &self.pipe_moe_ops,
                    &[
                        &vexp.down,
                        &self.exp_hidden_buf,
                        &self.dummy_buf,
                        &self.moe_tiles_buf,
                        top_w_buf,
                        &self.exp_down_buf,
                    ],
                    &[
                        5,
                        n_ff as u32,
                        n_embd as u32,
                        vexp.down_row_u16s,
                        vexp.down_expert_u16s,
                        0,
                        0,
                        0,
                    ],
                    n_embd.div_ceil(rows_per_wg) as u32,
                    max_tiles,
                )?;
            }
            return rec.dispatch_xy(
                &self.pipe_moe_ops,
                &[
                    &self.dummy_buf,
                    &self.exp_down_buf,
                    &self.dummy_buf,
                    top_ids_buf,
                    top_w_buf,
                    out_win_buf,
                ],
                &[3, n_embd as u32, top_k as u32, 0, 0, 0, 0, 0],
                n_embd.div_ceil(self.moe_wg as usize) as u32,
                w as u32,
            );
        }

        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &vexp.gate_up,
                x_win_buf,
                &self.dummy_buf,
                top_ids_buf,
                top_w_buf,
                &self.exp_gate_up_buf,
            ],
            &[
                1,
                n_embd as u32,
                (2 * n_ff) as u32,
                vexp.gate_up_row_u16s,
                vexp.gate_up_expert_u16s,
                top_k as u32,
                0,
                0,
            ],
            (2 * n_ff).div_ceil(rows_per_wg) as u32,
            total_k as u32,
        )?;
        rec.dispatch(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_gate_up_buf,
                &self.dummy_buf,
                top_ids_buf,
                top_w_buf,
                &self.exp_hidden_buf,
            ],
            &[
                2,
                (total_k * n_ff) as u32,
                n_ff as u32,
                0,
                0,
                0,
                self.cfg.situ_beta.to_bits(),
                self.cfg.situ_linear_beta.to_bits(),
            ],
            (total_k * n_ff).div_ceil(self.moe_wg as usize) as u32,
        )?;
        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &vexp.down,
                &self.exp_hidden_buf,
                &self.dummy_buf,
                top_ids_buf,
                top_w_buf,
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
            total_k as u32,
        )?;
        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.exp_down_buf,
                &self.dummy_buf,
                top_ids_buf,
                top_w_buf,
                out_win_buf,
            ],
            &[3, n_embd as u32, top_k as u32, 0, 0, 0, 0, 0],
            n_embd.div_ceil(self.moe_wg as usize) as u32,
            w as u32,
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
        self.record_vec_copy_nb(rec, &self.cur_buf, 0, &self.win_state_buf, base, n_embd)?;
        self.record_vec_copy_nb(
            rec,
            &self.ffn_inp_buf,
            0,
            &self.win_state_buf,
            base + n_embd as u32,
            n_embd,
        )?;
        self.record_vec_copy_nb(
            rec,
            &self.prelude_buf,
            0,
            &self.win_state_buf,
            base + (2 * n_embd) as u32,
            n_embd,
        )?;
        self.record_vec_copy_nb(
            rec,
            &self.bank_out_buf,
            0,
            &self.win_state_buf,
            base + (3 * n_embd) as u32,
            n_embd,
        )?;
        if blocks_len > 0 {
            self.record_vec_copy_nb(
                rec,
                &self.blocks_buf,
                0,
                &self.win_state_buf,
                base + (4 * n_embd) as u32,
                blocks_len * n_embd,
            )?;
        }
        if prelude_blocks_len > 0 {
            self.record_vec_copy_nb(
                rec,
                &self.prelude_blocks_buf,
                0,
                &self.win_state_buf,
                base + ((4 + max_b) * n_embd) as u32,
                prelude_blocks_len * n_embd,
            )?;
        }
        if bank_len > 0 {
            self.record_vec_copy_nb(
                rec,
                &self.bank_buf,
                0,
                &self.win_state_buf,
                base + ((4 + 2 * max_b) * n_embd) as u32,
                bank_len * n_embd,
            )?;
        }
        rec.barrier();
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
        self.record_vec_copy_nb(rec, &self.win_state_buf, base, &self.cur_buf, 0, n_embd)?;
        self.record_vec_copy_nb(
            rec,
            &self.win_state_buf,
            base + n_embd as u32,
            &self.ffn_inp_buf,
            0,
            n_embd,
        )?;
        self.record_vec_copy_nb(
            rec,
            &self.win_state_buf,
            base + (2 * n_embd) as u32,
            &self.prelude_buf,
            0,
            n_embd,
        )?;
        self.record_vec_copy_nb(
            rec,
            &self.win_state_buf,
            base + (3 * n_embd) as u32,
            &self.bank_out_buf,
            0,
            n_embd,
        )?;
        if blocks_len > 0 {
            self.record_vec_copy_nb(
                rec,
                &self.win_state_buf,
                base + (4 * n_embd) as u32,
                &self.blocks_buf,
                0,
                blocks_len * n_embd,
            )?;
        }
        if prelude_blocks_len > 0 {
            self.record_vec_copy_nb(
                rec,
                &self.win_state_buf,
                base + ((4 + max_b) * n_embd) as u32,
                &self.prelude_blocks_buf,
                0,
                prelude_blocks_len * n_embd,
            )?;
        }
        if bank_len > 0 {
            self.record_vec_copy_nb(
                rec,
                &self.win_state_buf,
                base + ((4 + 2 * max_b) * n_embd) as u32,
                &self.bank_buf,
                0,
                bank_len * n_embd,
            )?;
        }
        rec.barrier();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_strided_copy(
        &self,
        rec: &mut Recorder<'_>,
        dst: &VkBuffer,
        dst_off: u32,
        dst_stride: u32,
        src: &VkBuffer,
        src_off: u32,
        src_stride: u32,
        len: usize,
        w: usize,
        emit_barrier: bool,
    ) -> Result<(), VulkanError> {
        if len == 0 || w == 0 {
            return Ok(());
        }
        let bufs = [
            dst,
            src,
            &self.dummy_buf,
            &self.dummy_buf,
            &self.dummy_buf,
            &self.dummy_buf,
        ];
        let args = [
            7, len as u32, 1, dst_off, src_off, src_stride, 0, dst_stride,
        ];
        let gx = self.elem_groups(len);
        if emit_barrier {
            rec.dispatch_xy(&self.pipe_ops, &bufs, &args, gx, w as u32)
        } else {
            rec.dispatch_xy_nb(&self.pipe_ops, &bufs, &args, gx, w as u32)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_strided_add(
        &self,
        rec: &mut Recorder<'_>,
        dst: &VkBuffer,
        dst_off: u32,
        dst_stride: u32,
        a: &VkBuffer,
        a_off: u32,
        a_stride: u32,
        b: &VkBuffer,
        b_off: u32,
        len: usize,
        w: usize,
    ) -> Result<(), VulkanError> {
        if len == 0 || w == 0 {
            return Ok(());
        }
        rec.dispatch_xy(
            &self.pipe_ops,
            &[dst, a, b, &self.dummy_buf, &self.dummy_buf, &self.dummy_buf],
            &[
                7, len as u32, 0, dst_off, a_off, a_stride, b_off, dst_stride,
            ],
            self.elem_groups(len),
            w as u32,
        )
    }

    /// Batched `post_moe` (`ffn_post_conv` across `w` tokens + residual add into `win_state_buf`)
    /// in 2 GPU dispatches instead of `4 * w`!
    fn record_win_post_moe(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        w: usize,
        moe_in_dev: bool,
        save_conv_snapshots: bool,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let kv_dim = self.cfg.kv_dim(ip);
        let n_conv = self.cfg.shortconv_l_cache;
        let slot_conv_base = self.slot_conv_bases[il];
        let layer = &self.layers[ip];
        let stride = self.per_tok_state_f32s as u32;
        let conv_p1 = if save_conv_snapshots {
            ((self.total_conv_f32s as u32) << 8) | (n_conv as u32)
        } else {
            n_conv as u32
        };
        let snap_buf = if save_conv_snapshots {
            &self.conv_snapshots_buf
        } else {
            &self.dummy_buf
        };

        if !moe_in_dev {
            self.record_vec_copy(
                rec,
                &self.win_moe_out_dev_buf,
                0,
                &self.moe_out_bridge,
                0,
                w * n_embd,
            )?;
        }
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.win_moe_out_dev_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.conv_hist,
                snap_buf,
                &self.dummy_buf,
            ],
            &[
                8,
                n_embd as u32,
                conv_p1,
                (2 * kv_dim + n_embd) as u32,
                0,
                layer.ffn_post_conv_off,
                w as u32,
                slot_conv_base,
            ],
            self.elem_groups(n_embd),
        )?;
        self.record_strided_add(
            rec,
            &self.win_state_buf,
            0,
            stride,
            &self.win_state_buf,
            n_embd as u32,
            stride,
            &self.win_moe_out_dev_buf,
            0,
            n_embd,
            w,
        )
    }

    fn build_window_segments(
        &self,
        allow_host_imported: bool,
        max_visits_per_cb: usize,
        out: &mut [GpuSegSpec; 64],
    ) -> usize {
        let mut n_segs = 0usize;
        let mut cursor = 0usize;
        let mut need_embed = true;
        let mut pending_post_moe: Option<(usize, usize)> = None;
        let mut b = 0usize;
        let mut pb = 0usize;
        let mut bk = 0usize;

        while cursor < self.schedule.len() {
            let init_b = b;
            let init_pb = pb;
            let init_bk = bk;
            let start_cur = cursor;
            let mut stop_host = None;
            let mut visits_in_cb = 0usize;
            while cursor < self.schedule.len() && visits_in_cb < max_visits_per_cb {
                let st = self.schedule[cursor];
                self.advance_depth_counts(st, &mut b, &mut pb, &mut bk);
                cursor += 1;
                visits_in_cb += 1;
                if self
                    .window_gpu_experts(st.ip, allow_host_imported)
                    .is_none()
                {
                    stop_host = Some((st.il, st.ip));
                    break;
                }
            }
            let run_outputs = cursor == self.schedule.len() && stop_host.is_none();
            out[n_segs] = GpuSegSpec {
                need_embed,
                prev_host: pending_post_moe.take(),
                start_cur,
                end_cur: cursor,
                stop_host,
                run_outputs,
                init_b,
                init_pb,
                init_bk,
                end_b: b,
            };
            n_segs += 1;
            need_embed = false;
            pending_post_moe = stop_host;
        }

        if let Some(last_host) = pending_post_moe.take() {
            out[n_segs] = GpuSegSpec {
                need_embed: false,
                prev_host: Some(last_host),
                start_cur: cursor,
                end_cur: cursor,
                stop_host: None,
                run_outputs: true,
                init_b: b,
                init_pb: pb,
                init_bk: bk,
                end_b: b,
            };
            n_segs += 1;
        }
        n_segs
    }

    #[allow(clippy::too_many_arguments)]
    fn record_window_segment(
        &self,
        rec: &mut Recorder<'_>,
        seg: GpuSegSpec,
        w: usize,
        start_pos: usize,
        taus: &[f32],
        can_batch_gemm: bool,
        compute_last_logits: bool,
        compute_all_argmax: bool,
        save_conv_snapshots: bool,
        dspark_target_visits: &[usize],
        ts: &mut Option<TsRecorder<'_>>,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let top_k = self.cfg.n_expert_used;
        let hist_len = self.cfg.shortconv_l_cache - 1;
        let n_taps = dspark_target_visits.len();

        if seg.need_embed {
            if w > 1 {
                let emb_wg = self.pipe_embed_batch.wg_size;
                rec.dispatch_xy(
                    &self.pipe_embed_batch,
                    &[&self.tok_embd.buf, &self.step_buf, &self.win_q_in_buf],
                    &[n_embd as u32, self.tok_embd.row_u16s],
                    (n_embd as u32).div_ceil(emb_wg),
                    w as u32,
                )?;
                rec.dispatch_xy(
                    &self.pipe_ops,
                    &[
                        &self.win_q_in_buf,
                        &self.win_k_in_buf,
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
                        n_embd as u32,
                        self.tok_norm_off,
                    ],
                    1,
                    w as u32,
                )?;
                self.record_strided_copy(
                    rec,
                    &self.win_state_buf,
                    0,
                    self.per_tok_state_f32s as u32,
                    &self.win_k_in_buf,
                    0,
                    n_embd as u32,
                    n_embd,
                    w,
                    true,
                )?;
            } else {
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
                    self.record_vec_copy(rec, &self.win_state_buf, base, &self.cur_buf, 0, n_embd)?;
                }
            }
            TsRecorder::mark(ts, rec, STAGE_EMBED_STATE);
        }

        if let Some((prev_il, prev_ip)) = seg.prev_host {
            let k_gpu = self.layer_gpu_ext_experts(prev_ip);
            let k_cpu = top_k - k_gpu;
            if can_batch_gemm {
                if k_gpu > 0 {
                    if k_cpu > 0 {
                        self.record_vec_copy(
                            rec,
                            &self.win_moe_out_dev_buf,
                            0,
                            &self.moe_out_bridge,
                            0,
                            w * n_embd,
                        )?;
                    }
                    self.record_ext_moe_reduce(
                        rec,
                        w,
                        k_cpu,
                        k_gpu,
                        &self.top_weights_bridge,
                        &self.win_moe_out_dev_buf,
                    )?;
                    self.record_win_post_moe(rec, prev_il, prev_ip, w, true, save_conv_snapshots)?;
                } else {
                    self.record_win_post_moe(rec, prev_il, prev_ip, w, false, save_conv_snapshots)?;
                }
            } else {
                let moe_src = if k_gpu > 0 {
                    if k_cpu > 0 {
                        self.record_vec_copy(
                            rec,
                            &self.win_moe_out_dev_buf,
                            0,
                            &self.moe_out_bridge,
                            0,
                            w * n_embd,
                        )?;
                    }
                    self.record_ext_moe_reduce(
                        rec,
                        w,
                        k_cpu,
                        k_gpu,
                        &self.top_weights_bridge,
                        &self.win_moe_out_dev_buf,
                    )?;
                    &self.win_moe_out_dev_buf
                } else {
                    &self.moe_out_bridge
                };
                let prev_kv_dim = self.cfg.kv_dim(prev_ip);
                let ffn_conv_off_f32 =
                    self.slot_conv_bases[prev_il] as usize + (2 * prev_kv_dim + n_embd) * hist_len;
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
                        moe_src,
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
                    self.record_vec_copy(rec, &self.win_state_buf, base, &self.cur_buf, 0, n_embd)?;
                }
            }
            TsRecorder::mark(ts, rec, STAGE_POST_MOE);
        }

        let mut blocks_len = seg.init_b;
        let mut prelude_blocks_len = seg.init_pb;
        let mut bank_len = seg.init_bk;

        let use_tc = w > 8 && !save_conv_snapshots;
        let allow_host_imported = can_batch_gemm
            && use_tc
            && self.pipe_moe_grouped.is_some()
            && self.cfg.n_ff_exp.is_multiple_of(32);

        for &st in &self.schedule[seg.start_cur..seg.end_cur] {
            let in_vram = self.layers[st.ip].vram_experts.is_some();
            let batch_gpu_vexp = self.window_gpu_experts(
                st.ip,
                allow_host_imported && self.cfg.q_dim(st.ip).is_multiple_of(32),
            );
            let kv_dim = self.cfg.kv_dim(st.ip);
            let attn_conv_off_f32 = self.slot_conv_bases[st.il] as usize;
            let attn_conv_len_f32 = (2 * kv_dim + n_embd) * hist_len;
            let ffn_conv_off_f32 = attn_conv_off_f32 + attn_conv_len_f32;
            let ffn_conv_len_f32 = n_embd * hist_len;
            let tap_slot = dspark_target_visits.iter().position(|&v| v == st.visit);

            let mut next_b = blocks_len;
            let mut next_pb = prelude_blocks_len;
            let mut next_bk = bank_len;

            if can_batch_gemm && (!use_tc || self.cfg.q_dim(st.ip).is_multiple_of(32)) {
                self.record_visit_batched_gemm(
                    rec,
                    st.il,
                    st.ip,
                    st.pass > 0,
                    st.save_prelude_before,
                    st.reset_pass_before,
                    w,
                    start_pos,
                    tap_slot,
                    n_taps,
                    batch_gpu_vexp.is_some(),
                    use_tc,
                    save_conv_snapshots,
                    &mut next_b,
                    &mut next_pb,
                    &mut next_bk,
                    ts,
                )?;
                if let Some(vexp) = batch_gpu_vexp {
                    self.record_moe_window_with(
                        rec,
                        vexp,
                        w,
                        !save_conv_snapshots,
                        &self.win_ffn_norm_buf,
                        &self.win_top_ids_dev_buf,
                        &self.win_top_weights_dev_buf,
                        &self.win_moe_out_dev_buf,
                    )?;
                    TsRecorder::mark(ts, rec, STAGE_VRAM_MOE);
                    self.record_win_post_moe(rec, st.il, st.ip, w, true, save_conv_snapshots)?;
                    TsRecorder::mark(ts, rec, STAGE_POST_MOE);
                }
            } else {
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
                    TsRecorder::mark(ts, rec, STAGE_EMBED_STATE);

                    if in_vram && w == 1 {
                        self.record_layer_pre_moe_into(
                            rec,
                            st.il,
                            st.ip,
                            start_pos + tok_i,
                            tau_t,
                            false,
                            st.pass > 0,
                            &mut b_t,
                            bk_t,
                            &self.wo_out_buf,
                            &self.top_ids_tmp_buf,
                            &self.top_weights_tmp_buf,
                            ts,
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
                        self.record_vram_moe_from(
                            rec,
                            st.ip,
                            &self.wo_out_buf,
                            &self.top_ids_tmp_buf,
                            &self.top_weights_tmp_buf,
                        )?;
                        TsRecorder::mark(ts, rec, STAGE_VRAM_MOE);
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
                        self.record_save_tok_state(rec, tok_i, b_t, pb_t, bk_t)?;
                        TsRecorder::mark(ts, rec, STAGE_POST_MOE);
                    } else {
                        self.record_layer_pre_moe_into(
                            rec,
                            st.il,
                            st.ip,
                            start_pos + tok_i,
                            tau_t,
                            false,
                            st.pass > 0,
                            &mut b_t,
                            bk_t,
                            &self.wo_out_buf,
                            &self.top_ids_tmp_buf,
                            &self.top_weights_tmp_buf,
                            ts,
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
                        self.record_save_tok_state(rec, tok_i, b_t, pb_t, bk_t)?;
                        TsRecorder::mark(ts, rec, STAGE_EMBED_STATE);
                    }

                    next_b = b_t;
                    next_pb = pb_t;
                    next_bk = bk_t;
                }

                if in_vram && w > 1 {
                    self.record_vram_moe_window(
                        rec,
                        st.ip,
                        w,
                        !save_conv_snapshots,
                        &self.ffn_normed_bridge,
                        &self.top_ids_bridge,
                        &self.top_weights_bridge,
                        &self.moe_out_bridge,
                    )?;
                    TsRecorder::mark(ts, rec, STAGE_VRAM_MOE);
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
                        self.record_vec_copy(
                            rec,
                            &self.win_state_buf,
                            base,
                            &self.cur_buf,
                            0,
                            n_embd,
                        )?;
                    }
                    TsRecorder::mark(ts, rec, STAGE_POST_MOE);
                }
            }

            blocks_len = next_b;
            prelude_blocks_len = next_pb;
            bank_len = next_bk;
        }

        if seg.run_outputs {
            self.record_window_outputs(
                rec,
                w,
                seg.end_b,
                compute_last_logits,
                compute_all_argmax,
                dspark_target_visits,
            )?;
            TsRecorder::mark(ts, rec, STAGE_LM_HEAD);
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
        if self.prof_on {
            self.prof.tokens += w as u64;
        }

        let mut taus = [1.0f32; MAX_INKLING2_VK_WINDOW];
        for (tok_i, tau_slot) in taus.iter_mut().enumerate().take(w) {
            let ratio = ((start_pos + tok_i + 1) as f32) / (self.cfg.log_scale_floor as f32);
            *tau_slot = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();
        }
        write_mapped_u32(&self.ctx, &self.step_buf, window_tokens)?;
        write_mapped_f32(&self.ctx, &self.win_taus_buf, &taus[..w])?;

        let can_batch_gemm = w >= 2
            && (!save_conv_snapshots
                || (w <= MAX_INKLING2_SPEC_WINDOW && self.total_conv_f32s < (1 << 24)))
            && n_embd.is_multiple_of(32)
            && self.cfg.shortconv_l_cache <= 9
            && self.ctx.info().subgroup_size <= 128
            && self.ctx.info().max_workgroup_invocations >= 128;
        let allow_host_imported = can_batch_gemm
            && w > 8
            && !save_conv_snapshots
            && self.pipe_moe_grouped.is_some()
            && self.cfg.n_ff_exp.is_multiple_of(32)
            && (0..self.cfg.n_layers_phys).all(|ip| self.cfg.q_dim(ip).is_multiple_of(32));
        let max_visits_per_cb = if can_batch_gemm {
            64usize
        } else {
            (256 / w.max(1)).clamp(1, 64)
        };

        let mut segs = [GpuSegSpec {
            need_embed: false,
            prev_host: None,
            start_cur: 0,
            end_cur: 0,
            stop_host: None,
            run_outputs: false,
            init_b: 0,
            init_pb: 0,
            init_bk: 0,
            end_b: 0,
        }; 64];
        let n_segs = self.build_window_segments(allow_host_imported, max_visits_per_cb, &mut segs);
        if n_segs == 0 {
            return Ok(());
        }

        let qp = self.prof_qp;
        let mut spans_slots = core::mem::take(&mut self.prof_spans_slots);
        let mut n_queries_slots = [0u32; 2];

        let prep0 = self.ctx.prepare_cb(0, |rec| {
            let mut ts = TsRecorder::begin(rec, qp, &mut spans_slots[0]);
            self.record_window_segment(
                rec,
                segs[0],
                w,
                start_pos,
                &taus[..w],
                can_batch_gemm,
                compute_last_logits,
                compute_all_argmax,
                save_conv_snapshots,
                dspark_target_visits,
                &mut ts,
            )?;
            n_queries_slots[0] = ts.map_or(0, |t| t.next_q);
            Ok(())
        });
        let prep0 = match prep0 {
            Ok(p) => p,
            Err(e) => {
                self.prof_spans_slots = spans_slots;
                return Err(e);
            }
        };
        let mut inflight = Some(self.ctx.submit_prepared(prep0)?);

        let mut loop_res: Result<(), VulkanError> = Ok(());
        for idx in 0..n_segs {
            let cur_slot = idx & 1;
            let next_slot = (idx + 1) & 1;
            let next_prep = if idx + 1 < n_segs {
                match self.ctx.prepare_cb(next_slot, |rec| {
                    let mut ts = TsRecorder::begin(rec, qp, &mut spans_slots[next_slot]);
                    self.record_window_segment(
                        rec,
                        segs[idx + 1],
                        w,
                        start_pos,
                        &taus[..w],
                        can_batch_gemm,
                        compute_last_logits,
                        compute_all_argmax,
                        save_conv_snapshots,
                        dspark_target_visits,
                        &mut ts,
                    )?;
                    n_queries_slots[next_slot] = ts.map_or(0, |t| t.next_q);
                    Ok(())
                }) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        if let Some(inf) = inflight.take() {
                            let _ = self.ctx.wait_in_flight(inf);
                        }
                        loop_res = Err(e);
                        break;
                    }
                }
            } else {
                None
            };

            let inf = inflight.take().expect("active in-flight window segment");
            let sub_t = match self.ctx.wait_in_flight(inf) {
                Ok(t) => t,
                Err(e) => {
                    loop_res = Err(e);
                    break;
                }
            };
            self.prof_finish_submit(sub_t, n_queries_slots[cur_slot], &spans_slots[cur_slot]);

            if let Some((_, h_ip)) = segs[idx].stop_host {
                let k_gpu = self.layer_gpu_ext_experts(h_ip);
                let k_cpu = top_k - k_gpu;
                let ext_x_buf = if can_batch_gemm {
                    &self.win_ffn_norm_buf
                } else {
                    &self.ffn_normed_bridge
                };
                let ext_inf = if k_gpu > 0 {
                    match self
                        .ctx
                        .prepare_cb(cur_slot, |rec| {
                            self.record_ext_moe_tail(
                                rec,
                                h_ip,
                                w,
                                k_cpu,
                                k_gpu,
                                ext_x_buf,
                                &self.top_ids_bridge,
                            )
                        })
                        .and_then(|p| self.ctx.submit_prepared(p))
                    {
                        Ok(ei) => Some(ei),
                        Err(e) => {
                            loop_res = Err(e);
                            break;
                        }
                    }
                } else {
                    None
                };

                if k_cpu > 0 {
                    let t_br0 = std::time::Instant::now();
                    let eval_win = match host_moe_win.as_deref_mut() {
                        Some(f) => f,
                        None => {
                            if let Some(ei) = ext_inf {
                                let _ = self.ctx.wait_in_flight(ei);
                            }
                            loop_res = Err(VulkanError::InvalidDimensions(format!(
                                "layer {h_ip} MoE experts are offloaded from VRAM, but no HostMoeWindowFn was supplied"
                            )));
                            break;
                        }
                    };
                    let mut raw_ids = [0u32; MAX_INKLING2_VK_WINDOW * 32];
                    let mut raw_w = [0.0f32; MAX_INKLING2_VK_WINDOW * 32];
                    if let Err(e) = read_mapped_f32(
                        &self.ctx,
                        &self.ffn_normed_bridge,
                        &mut self.host_cur_scratch[..w * n_embd],
                    )
                    .and_then(|()| read_mapped_u32(&self.top_ids_bridge, &mut raw_ids[..w * top_k]))
                    .and_then(|()| {
                        read_mapped_f32(
                            &self.ctx,
                            &self.top_weights_bridge,
                            &mut raw_w[..w * top_k],
                        )
                    }) {
                        if let Some(ei) = ext_inf {
                            let _ = self.ctx.wait_in_flight(ei);
                        }
                        loop_res = Err(e);
                        break;
                    }
                    for tok_i in 0..w {
                        for j in 0..k_cpu {
                            self.host_ids_scratch[tok_i * k_cpu + j] =
                                raw_ids[tok_i * top_k + j] as usize;
                            self.host_weights_scratch[tok_i * k_cpu + j] = raw_w[tok_i * top_k + j];
                        }
                    }
                    let t_moe0 = std::time::Instant::now();
                    eval_win(
                        h_ip,
                        w,
                        &self.host_cur_scratch[..w * n_embd],
                        &self.host_ids_scratch[..w * k_cpu],
                        &self.host_weights_scratch[..w * k_cpu],
                        &mut self.host_moe_out_scratch[..w * n_embd],
                    );
                    let t_moe1 = std::time::Instant::now();
                    if let Err(e) = write_mapped_f32(
                        &self.ctx,
                        &self.moe_out_bridge,
                        &self.host_moe_out_scratch[..w * n_embd],
                    ) {
                        if let Some(ei) = ext_inf {
                            let _ = self.ctx.wait_in_flight(ei);
                        }
                        loop_res = Err(e);
                        break;
                    }
                    if self.prof_on {
                        let moe_ns = (t_moe1 - t_moe0).as_nanos() as u64;
                        self.prof.host_moe_calls += 1;
                        self.prof.host_moe_ns += moe_ns;
                        self.prof.bridge_ns +=
                            (t_br0.elapsed().as_nanos() as u64).saturating_sub(moe_ns);
                    }
                }

                if let Some(ei) = ext_inf {
                    match self.ctx.wait_in_flight(ei) {
                        Ok(ext_t) => self.prof_finish_submit(ext_t, 0, &[]),
                        Err(e) => {
                            loop_res = Err(e);
                            break;
                        }
                    }
                }
            }

            if let Some(prep) = next_prep {
                match self.ctx.submit_prepared(prep) {
                    Ok(inf_next) => inflight = Some(inf_next),
                    Err(e) => {
                        loop_res = Err(e);
                        break;
                    }
                }
            }
        }

        self.prof_spans_slots = spans_slots;
        loop_res
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

        if (2..=MAX_INKLING2_SPEC_WINDOW).contains(&w) && n_embd.is_multiple_of(32) {
            if !compute_all_argmax && !compute_last_logits && final_tap.is_none() {
                return Ok(());
            }
            let wy = w as u32;
            let stride = self.per_tok_state_f32s as u32;
            let off_blocks = (4 * n_embd) as u32;
            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.win_state_buf,
                    &self.win_state_buf,
                    &self.f32_params,
                    &self.win_q_in_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[
                    3,
                    n_embd as u32,
                    blocks_len as u32,
                    self.out_res_query_off,
                    0,
                    off_blocks,
                    0,
                    stride,
                ],
                1,
                wy,
            )?;
            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.win_q_in_buf,
                    &self.win_wo_out_buf,
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
                    n_embd as u32,
                    self.out_norm_off,
                ],
                1,
                wy,
            )?;
            if let Some(t_pos) = final_tap {
                self.record_strided_copy(
                    rec,
                    &self.dspark_taps_buf,
                    (t_pos * n_embd) as u32,
                    (n_taps * n_embd) as u32,
                    &self.win_wo_out_buf,
                    0,
                    n_embd as u32,
                    n_embd,
                    w,
                    true,
                )?;
            }
            if compute_all_argmax {
                // Single-read `M = 4` / `M = 8` `output.weight` projection across ALL `w` tokens!
                self.dispatch_gemm(
                    rec,
                    &self.output,
                    &self.win_wo_out_buf,
                    &self.win_logits_padded_buf,
                    w,
                    false,
                )?;
                rec.dispatch_xy(
                    &self.pipe_ops,
                    &[
                        &self.win_logits_dev_buf,
                        &self.win_logits_padded_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[
                        7,
                        self.cfg.vocab_size as u32,
                        2,
                        0,
                        0,
                        self.cfg.vocab_padded as u32,
                        self.cfg.final_logit_softcapping.to_bits(),
                        self.cfg.vocab_size as u32,
                    ],
                    self.elem_groups(self.cfg.vocab_size),
                    wy,
                )?;
                rec.dispatch_xy(
                    &self.pipe_argmax,
                    &[&self.win_logits_dev_buf, &self.win_argmax_buf],
                    &[self.cfg.vocab_size as u32],
                    1,
                    wy,
                )?;
                if compute_last_logits {
                    self.record_vec_copy(
                        rec,
                        &self.logits_buf,
                        0,
                        &self.win_logits_dev_buf,
                        ((w - 1) * self.cfg.vocab_size) as u32,
                        self.cfg.vocab_size,
                    )?;
                }
            } else if compute_last_logits {
                self.record_vec_copy(
                    rec,
                    &self.wo_out_buf,
                    0,
                    &self.win_wo_out_buf,
                    ((w - 1) * n_embd) as u32,
                    n_embd,
                )?;
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
            }
            return Ok(());
        }

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

    fn record_win_controls_qkv(
        &self,
        rec: &mut Recorder<'_>,
        loop_w: &VkLoopLayer,
        w: usize,
        use_tc: bool,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd as u32;
        let wy = w as u32;
        let out_m2 = 2u32 << 24;
        for j in 0..3 {
            let off = (j as u32) * n_embd;
            let cin = &self.win_ctrl_in_bufs[j];
            rec.dispatch_xy_nb(
                &self.pipe_ops,
                &[
                    &self.win_inputs_buf,
                    cin,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[0, n_embd, off, out_m2, 0, 1e-6f32.to_bits(), 3 * n_embd, 0],
                1,
                wy,
            )?;
            let args_m = [
                0,
                n_embd,
                off,
                out_m2 | n_embd,
                0,
                1e-6f32.to_bits(),
                4 * n_embd,
                0,
            ];
            let bufs_m = [
                &self.win_messages_buf,
                cin,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ];
            if j < 2 {
                rec.dispatch_xy_nb(&self.pipe_ops, &bufs_m, &args_m, 1, wy)?;
            } else {
                rec.dispatch_xy(&self.pipe_ops, &bufs_m, &args_m, 1, wy)?;
            }
        }
        for j in 0..3 {
            let cd = &loop_w.ctrl_down[LOOP_CTRL_Q + j];
            if j < 2 {
                self.dispatch_gemm_nb(
                    rec,
                    cd,
                    &self.win_ctrl_in_bufs[j],
                    &self.win_ctrl_g_bufs[j],
                    w,
                    use_tc,
                )?;
            } else {
                self.dispatch_gemm(
                    rec,
                    cd,
                    &self.win_ctrl_in_bufs[j],
                    &self.win_ctrl_g_bufs[j],
                    w,
                    use_tc,
                )?;
            }
        }
        for j in 0..3 {
            let cg = &loop_w.ctrl_gain[LOOP_CTRL_Q + j];
            let cf = &loop_w.ctrl_forget[LOOP_CTRL_Q + j];
            self.dispatch_gemm_nb(
                rec,
                cg,
                &self.win_ctrl_g_bufs[j],
                &self.win_ctrl_gain_bufs[j],
                w,
                use_tc,
            )?;
            if j < 2 {
                self.dispatch_gemm_nb(
                    rec,
                    cf,
                    &self.win_ctrl_g_bufs[j],
                    &self.win_ctrl_forget_bufs[j],
                    w,
                    use_tc,
                )?;
            } else {
                self.dispatch_gemm(
                    rec,
                    cf,
                    &self.win_ctrl_g_bufs[j],
                    &self.win_ctrl_forget_bufs[j],
                    w,
                    use_tc,
                )?;
            }
        }
        let gx = self.elem_groups(self.cfg.n_embd);
        for j in 0..3 {
            let off = (j as u32) * n_embd;
            let bufs = [
                &self.win_inputs_buf,
                &self.win_messages_buf,
                &self.win_ctrl_gain_bufs[j],
                &self.win_ctrl_forget_bufs[j],
                &self.dummy_buf,
                &self.dummy_buf,
            ];
            let args = [4, n_embd, off, off, 3 * n_embd, 4 * n_embd, 0, 0];
            if j < 2 {
                rec.dispatch_xy_nb(&self.pipe_ops, &bufs, &args, gx, wy)?;
            } else {
                rec.dispatch_xy(&self.pipe_ops, &bufs, &args, gx, wy)?;
            }
        }
        Ok(())
    }

    fn record_win_control_ffn(
        &self,
        rec: &mut Recorder<'_>,
        loop_w: &VkLoopLayer,
        cur_win_buf: &VkBuffer,
        w: usize,
        use_tc: bool,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd as u32;
        let wy = w as u32;
        let out_m2 = 2u32 << 24;
        let cin = &self.win_ctrl_in_bufs[0];
        let cg = &self.win_ctrl_g_bufs[0];
        let cgain = &self.win_ctrl_gain_bufs[0];
        let cforget = &self.win_ctrl_forget_bufs[0];
        let msg_off = (LOOP_CTRL_FFN as u32) * n_embd;

        rec.dispatch_xy_nb(
            &self.pipe_ops,
            &[
                cur_win_buf,
                cin,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, n_embd, 0, out_m2, 0, 1e-6f32.to_bits(), n_embd, 0],
            1,
            wy,
        )?;
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &self.win_messages_buf,
                cin,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                0,
                n_embd,
                msg_off,
                out_m2 | n_embd,
                0,
                1e-6f32.to_bits(),
                4 * n_embd,
                0,
            ],
            1,
            wy,
        )?;
        self.dispatch_gemm(rec, &loop_w.ctrl_down[LOOP_CTRL_FFN], cin, cg, w, use_tc)?;
        self.dispatch_gemm_nb(rec, &loop_w.ctrl_gain[LOOP_CTRL_FFN], cg, cgain, w, use_tc)?;
        self.dispatch_gemm(
            rec,
            &loop_w.ctrl_forget[LOOP_CTRL_FFN],
            cg,
            cforget,
            w,
            use_tc,
        )?;
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                cur_win_buf,
                &self.win_messages_buf,
                cgain,
                cforget,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[4, n_embd, 0, msg_off, n_embd, 4 * n_embd, 0, 0],
            self.elem_groups(self.cfg.n_embd),
            wy,
        )
    }

    /// Evaluates pre-MoE dense, ShortConv, RelAttn, and Loop Controller operations across `w >= 2`
    /// tokens with ZERO per-token CPU/GPU loops (`CoopMat F16` / `gemm_tiled` for `w >= 8` prefill,
    /// `0-ULP` bit-exact 2D `gemv_res` + `op=8` ShortConv snapshots for `DSpark` speculative verify).
    #[allow(clippy::too_many_arguments)]
    fn record_visit_batched_gemm(
        &self,
        rec: &mut Recorder<'_>,
        il: usize,
        ip: usize,
        has_messages: bool,
        save_prelude_before: bool,
        reset_pass_before: bool,
        w: usize,
        start_pos: usize,
        tap_slot: Option<usize>,
        n_taps: usize,
        in_vram: bool,
        use_tc: bool,
        save_conv_snapshots: bool,
        blocks_len: &mut usize,
        prelude_blocks_len: &mut usize,
        bank_len: &mut usize,
        ts: &mut Option<TsRecorder<'_>>,
    ) -> Result<(), VulkanError> {
        let n_embd = self.cfg.n_embd;
        let max_b = self.cfg.max_depth_blocks();
        let stride = self.per_tok_state_f32s as u32;
        let off_cur = 0u32;
        let off_ffn_inp = n_embd as u32;
        let off_prelude = (2 * n_embd) as u32;
        let off_bank_out = (3 * n_embd) as u32;
        let off_blocks = (4 * n_embd) as u32;
        let off_prelude_blocks = ((4 + max_b) * n_embd) as u32;
        let off_bank = ((4 + 2 * max_b) * n_embd) as u32;

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
        let wy = w as u32;

        // Phase 1: Strided 2D pre-step transitions + 2D depth mix + batched GEMM controllers + 2D RMSNorm
        if save_prelude_before {
            *prelude_blocks_len = *blocks_len;
            let pb_len = *prelude_blocks_len * n_embd;
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_prelude,
                stride,
                &self.win_state_buf,
                off_cur,
                stride,
                n_embd,
                w,
                pb_len == 0,
            )?;
            if pb_len > 0 {
                self.record_strided_copy(
                    rec,
                    &self.win_state_buf,
                    off_prelude_blocks,
                    stride,
                    &self.win_state_buf,
                    off_blocks,
                    stride,
                    pb_len,
                    w,
                    true,
                )?;
            }
        }
        if reset_pass_before {
            let bank_start = *prelude_blocks_len + 1;
            *bank_len = blocks_len.saturating_sub(bank_start);
            if *bank_len > 0 {
                self.record_strided_copy(
                    rec,
                    &self.win_state_buf,
                    off_bank,
                    stride,
                    &self.win_state_buf,
                    off_blocks + (bank_start * n_embd) as u32,
                    stride,
                    *bank_len * n_embd,
                    w,
                    true,
                )?;
            }
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_bank_out,
                stride,
                &self.win_state_buf,
                off_cur,
                stride,
                n_embd,
                w,
                true,
            )?;
            let pb_len = *prelude_blocks_len * n_embd;
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_cur,
                stride,
                &self.win_state_buf,
                off_prelude,
                stride,
                n_embd,
                w,
                pb_len == 0,
            )?;
            *blocks_len = *prelude_blocks_len;
            if pb_len > 0 {
                self.record_strided_copy(
                    rec,
                    &self.win_state_buf,
                    off_blocks,
                    stride,
                    &self.win_state_buf,
                    off_prelude_blocks,
                    stride,
                    pb_len,
                    w,
                    true,
                )?;
            }
        }

        if let Some(t_pos) = tap_slot {
            self.record_strided_copy(
                rec,
                &self.dspark_taps_buf,
                (t_pos * n_embd) as u32,
                (n_taps * n_embd) as u32,
                &self.win_state_buf,
                off_cur,
                stride,
                n_embd,
                w,
                false,
            )?;
        }

        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            rec.dispatch_xy_nb(
                &self.pipe_ops,
                &[
                    &self.win_state_buf,
                    &self.win_state_buf,
                    &self.f32_params,
                    &self.win_messages_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ],
                &[
                    3,
                    n_embd as u32,
                    *bank_len as u32,
                    loop_w.msg_query_off,
                    0,
                    off_bank,
                    off_bank_out,
                    stride,
                ],
                LOOP_CTRL_COUNT as u32,
                wy,
            )?;
        }

        // Stash `inp_sa = cur` inside `OFF_FFN_INP` until Phase 5
        self.record_strided_copy(
            rec,
            &self.win_state_buf,
            off_ffn_inp,
            stride,
            &self.win_state_buf,
            off_cur,
            stride,
            n_embd,
            w,
            true,
        )?;

        if let Some(rq_off) = layer.attn_res_query_off {
            if swa {
                rec.dispatch_xy(
                    &self.pipe_ops,
                    &[
                        &self.win_state_buf,
                        &self.win_state_buf,
                        &self.f32_params,
                        &self.win_q_in_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[
                        3,
                        n_embd as u32,
                        *blocks_len as u32,
                        rq_off,
                        0,
                        off_blocks,
                        off_cur,
                        stride,
                    ],
                    1,
                    wy,
                )?;
                for j in 0..3 {
                    self.record_strided_copy(
                        rec,
                        &self.win_inputs_buf,
                        (j * n_embd) as u32,
                        (3 * n_embd) as u32,
                        &self.win_q_in_buf,
                        0,
                        n_embd as u32,
                        n_embd,
                        w,
                        j == 2,
                    )?;
                }
            } else {
                rec.dispatch_xy(
                    &self.pipe_ops,
                    &[
                        &self.win_state_buf,
                        &self.win_state_buf,
                        &self.f32_params,
                        &self.win_inputs_buf,
                        &self.dummy_buf,
                        &self.dummy_buf,
                    ],
                    &[
                        3,
                        n_embd as u32,
                        *blocks_len as u32,
                        rq_off,
                        0,
                        off_blocks,
                        off_cur,
                        stride,
                    ],
                    3,
                    wy,
                )?;
            }
        } else {
            for j in 0..3 {
                self.record_strided_copy(
                    rec,
                    &self.win_inputs_buf,
                    (j * n_embd) as u32,
                    (3 * n_embd) as u32,
                    &self.win_state_buf,
                    off_cur,
                    stride,
                    n_embd,
                    w,
                    j == 2,
                )?;
            }
        }

        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            self.record_win_controls_qkv(rec, loop_w, w, use_tc)?;
        }

        if swa && has_messages {
            for (j, dst_b) in [&self.win_q_in_buf, &self.win_k_in_buf, &self.win_v_in_buf]
                .into_iter()
                .enumerate()
            {
                let args = [
                    0,
                    n_embd as u32,
                    (j * n_embd) as u32,
                    0,
                    1,
                    self.cfg.rms_eps.to_bits(),
                    (3 * n_embd) as u32,
                    layer.attn_norm_off,
                ];
                let bufs = [
                    &self.win_inputs_buf,
                    dst_b,
                    &self.f32_params,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ];
                if j < 2 {
                    rec.dispatch_xy_nb(&self.pipe_ops, &bufs, &args, 1, wy)?;
                } else {
                    rec.dispatch_xy(&self.pipe_ops, &bufs, &args, 1, wy)?;
                }
            }
        } else if swa {
            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.win_inputs_buf,
                    &self.win_q_in_buf,
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
                    (3 * n_embd) as u32,
                    layer.attn_norm_off,
                ],
                1,
                wy,
            )?;
            self.record_vec_copy_nb(
                rec,
                &self.win_k_in_buf,
                0,
                &self.win_q_in_buf,
                0,
                w * n_embd,
            )?;
            self.record_vec_copy(
                rec,
                &self.win_v_in_buf,
                0,
                &self.win_q_in_buf,
                0,
                w * n_embd,
            )?;
        } else {
            let pairs = [
                (0usize, &self.win_q_in_buf, layer.q_inp_norm_off),
                (1usize, &self.win_k_in_buf, layer.k_inp_norm_off),
                (2usize, &self.win_v_in_buf, layer.v_inp_norm_off),
            ];
            for (j, dst_b, norm_off) in pairs {
                let args = [
                    0,
                    n_embd as u32,
                    (j * n_embd) as u32,
                    0,
                    1,
                    self.cfg.rms_eps.to_bits(),
                    (3 * n_embd) as u32,
                    norm_off,
                ];
                let bufs = [
                    &self.win_inputs_buf,
                    dst_b,
                    &self.f32_params,
                    &self.dummy_buf,
                    &self.dummy_buf,
                    &self.dummy_buf,
                ];
                if j < 2 {
                    rec.dispatch_xy_nb(&self.pipe_ops, &bufs, &args, 1, wy)?;
                } else {
                    rec.dispatch_xy(&self.pipe_ops, &bufs, &args, 1, wy)?;
                }
            }
        }
        TsRecorder::mark(ts, rec, STAGE_LOOP_MIX_NORM);

        let conv_p1 = if save_conv_snapshots {
            ((self.total_conv_f32s as u32) << 8) | (n_conv as u32)
        } else {
            n_conv as u32
        };
        let snap_buf = if save_conv_snapshots {
            &self.conv_snapshots_buf
        } else {
            &self.dummy_buf
        };

        // Phase 2: Batched QKV / Rel / Sink / Gate (+ dynamic ShortConv tap) GEMMs across ALL `w` tokens!
        self.dispatch_gemm_nb(
            rec,
            &layer.wq,
            &self.win_q_in_buf,
            &self.win_q_buf,
            w,
            use_tc,
        )?;
        self.dispatch_gemm_nb(
            rec,
            &layer.wk,
            &self.win_k_in_buf,
            &self.win_k_buf,
            w,
            use_tc,
        )?;
        self.dispatch_gemm_nb(
            rec,
            &layer.wv,
            &self.win_v_in_buf,
            &self.win_v_buf,
            w,
            use_tc,
        )?;
        self.dispatch_gemm_nb(
            rec,
            &layer.attn_rel,
            &self.win_q_in_buf,
            &self.win_r_buf,
            w,
            use_tc,
        )?;
        self.dispatch_gemm_nb(
            rec,
            &layer.attn_sink_proj,
            &self.win_q_in_buf,
            &self.win_sinks_buf,
            w,
            use_tc,
        )?;
        if swa {
            self.dispatch_gemm(
                rec,
                &layer.wqkv_gate,
                &self.win_q_in_buf,
                &self.win_gate_buf,
                w,
                use_tc,
            )?;
        } else {
            self.dispatch_gemm_nb(
                rec,
                &layer.wqkv_gate,
                &self.win_q_in_buf,
                &self.win_gate_buf,
                w,
                use_tc,
            )?;
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            self.dispatch_gemm_nb(rec, kp, &self.win_k_in_buf, &self.win_k_dyn_buf, w, use_tc)?;
            self.dispatch_gemm(rec, vp, &self.win_v_in_buf, &self.win_v_dyn_buf, w, use_tc)?;
        }

        // Phase 3: Multi-token ShortConv (`op=8`) + 2D Clip/Norm + 2D KV Pack + 2D RelAttn (`8` dispatches total!)
        let dyn_heads = if swa { 0u32 } else { nh_kv as u32 };
        rec.dispatch_nb(
            &self.pipe_ops,
            &[
                &self.win_k_buf,
                &self.f32_params,
                &self.win_k_dyn_buf,
                &self.conv_hist,
                snap_buf,
                &self.dummy_buf,
            ],
            &[
                8,
                kv_dim as u32,
                conv_p1,
                0,
                dyn_heads,
                layer.k_conv_off,
                wy,
                slot_conv_base,
            ],
            self.elem_groups(kv_dim),
        )?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.win_v_buf,
                &self.f32_params,
                &self.win_v_dyn_buf,
                &self.conv_hist,
                snap_buf,
                &self.dummy_buf,
            ],
            &[
                8,
                kv_dim as u32,
                conv_p1,
                kv_dim as u32,
                dyn_heads,
                layer.v_conv_off,
                wy,
                slot_conv_base,
            ],
            self.elem_groups(kv_dim),
        )?;

        let use_tau_buf = u32::from(!swa);
        rec.dispatch_xy_nb(
            &self.pipe_ops,
            &[
                &self.win_q_buf,
                &self.f32_params,
                &self.win_taus_buf,
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
                1.0f32.to_bits(),
                use_tau_buf,
            ],
            nh as u32,
            wy,
        )?;
        rec.dispatch_xy_nb(
            &self.pipe_ops,
            &[
                &self.win_k_buf,
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
            wy,
        )?;
        rec.dispatch_xy_nb(
            &self.pipe_ops,
            &[
                &self.win_v_buf,
                &self.win_v_norm_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[0, hd as u32, 0, 0, 0, self.cfg.rms_eps.to_bits(), 0, 0],
            nh_kv as u32,
            wy,
        )?;
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &self.win_r_buf,
                &self.f32_params,
                &self.win_taus_buf,
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
                1.0f32.to_bits(),
                use_tau_buf,
            ],
            nh as u32,
            wy,
        )?;

        let kv_pairs = kv_dim / 2;
        let dst_u32_off = self.slot_kv_u32_bases[il] + (start_pos * kv_pairs) as u32;
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &self.win_k_buf,
                &self.win_v_norm_buf,
                &self.dummy_buf,
                &self.dummy_buf,
                &self.k_cache,
                &self.v_cache,
            ],
            &[5, kv_pairs as u32, dst_u32_off, 0, 0, 0, 0, 0],
            self.elem_groups(kv_pairs),
            wy,
        )?;

        let swa_win = if swa { self.cfg.n_swa as u32 } else { 0 };
        rec.dispatch_xy(
            &self.pipe_rel_attn,
            &[
                &self.win_q_buf,
                &self.win_r_buf,
                &self.f32_params,
                &self.win_sinks_buf,
                &self.k_cache,
                &self.v_cache,
                &self.win_attn_vec_buf,
                &self.dyn_pos_buf,
            ],
            &[
                (nh as u32) | ((nh_kv as u32) << 16),
                (hd as u32) | ((n_rel as u32) << 16),
                extent as u32,
                start_pos as u32,
                swa_win,
                self.slot_kv_vec4_bases[il],
                layer.rel_bias_off,
            ],
            nh as u32,
            wy,
        )?;

        // Phase 4: Batched `sigmoid_mul` + Batched `wo` GEMM across ALL `w` tokens!
        rec.dispatch(
            &self.pipe_sigmoid_mul,
            &[&self.win_attn_vec_buf, &self.win_gate_buf],
            &[(w * q_dim) as u32],
            self.elem_groups(w * q_dim),
        )?;
        self.dispatch_gemm(
            rec,
            &layer.wo,
            &self.win_attn_vec_buf,
            &self.win_wo_out_buf,
            w,
            use_tc,
        )?;
        TsRecorder::mark(ts, rec, STAGE_ATTN_DENSE);

        // Phase 5: Multi-token `attn_post_conv` (`op=8`) + 2D Block Residual + 2D FFN Mix/Ctrl/Norm
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.win_wo_out_buf,
                &self.f32_params,
                &self.dummy_buf,
                &self.conv_hist,
                snap_buf,
                &self.dummy_buf,
            ],
            &[
                8,
                n_embd as u32,
                conv_p1,
                (2 * kv_dim) as u32,
                0,
                layer.attn_post_conv_off,
                wy,
                slot_conv_base,
            ],
            self.elem_groups(n_embd),
        )?;

        // `OFF_FFN_INP` currently holds `inp_sa` saved during Phase 1 — barrier before overwriting `OFF_FFN_INP`!
        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_blocks + (*blocks_len * n_embd) as u32,
                stride,
                &self.win_state_buf,
                off_ffn_inp,
                stride,
                n_embd,
                w,
                true,
            )?;
            *blocks_len += 1;
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_cur,
                stride,
                &self.win_wo_out_buf,
                0,
                n_embd as u32,
                n_embd,
                w,
                false,
            )?;
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_ffn_inp,
                stride,
                &self.win_wo_out_buf,
                0,
                n_embd as u32,
                n_embd,
                w,
                true,
            )?;
        } else {
            self.record_strided_add(
                rec,
                &self.win_state_buf,
                off_cur,
                stride,
                &self.win_state_buf,
                off_ffn_inp,
                stride,
                &self.win_wo_out_buf,
                0,
                n_embd,
                w,
            )?;
            self.record_strided_copy(
                rec,
                &self.win_state_buf,
                off_ffn_inp,
                stride,
                &self.win_state_buf,
                off_cur,
                stride,
                n_embd,
                w,
                true,
            )?;
        }

        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &self.win_state_buf,
                &self.win_state_buf,
                &self.f32_params,
                &self.win_q_in_buf,
                &self.dummy_buf,
                &self.dummy_buf,
            ],
            &[
                3,
                n_embd as u32,
                *blocks_len as u32,
                layer.ffn_res_query_off,
                0,
                off_blocks,
                off_cur,
                stride,
            ],
            1,
            wy,
        )?;
        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            self.record_win_control_ffn(rec, loop_w, &self.win_q_in_buf, w, use_tc)?;
        }
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &self.win_q_in_buf,
                &self.win_ffn_norm_buf,
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
                n_embd as u32,
                layer.ffn_norm_off,
            ],
            1,
            wy,
        )?;

        // Phase 6: Batched `ffn_gate_inp` Router GEMM + single 2D `top_k` dispatch across ALL `w` tokens!
        self.dispatch_gemm(
            rec,
            &layer.ffn_gate_inp,
            &self.win_ffn_norm_buf,
            &self.win_router_logits_buf,
            w,
            use_tc,
        )?;
        let (has_b, b_off) = match layer.exp_probs_b_off {
            Some(off) => (1u32, off),
            None => (0u32, 0u32),
        };
        let (top_ids_dst, top_w_dst) = if in_vram {
            (&self.win_top_ids_dev_buf, &self.win_top_weights_dev_buf)
        } else {
            (&self.top_ids_bridge, &self.top_weights_bridge)
        };
        rec.dispatch_xy(
            &self.pipe_moe_ops,
            &[
                &self.dummy_buf,
                &self.win_router_logits_buf,
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
            wy,
        )?;
        if !in_vram {
            self.record_vec_copy(
                rec,
                &self.ffn_normed_bridge,
                0,
                &self.win_ffn_norm_buf,
                0,
                w * n_embd,
            )?;
        }
        TsRecorder::mark(ts, rec, STAGE_FFN_ROUTER);
        Ok(())
    }
}

impl Drop for Inkling2VkRunner {
    fn drop(&mut self) {
        if let Some(qp) = self.prof_qp.take() {
            // SAFETY: Wait idle before destroying query pool created from `self.ctx.inner.device`.
            unsafe {
                let _ = self.ctx.inner.device.device_wait_idle();
                self.ctx.inner.device.destroy_query_pool(qp, None);
            }
        }
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
