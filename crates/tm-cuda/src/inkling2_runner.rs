//! NVIDIA CUDA GPU runner for Thinking Machines Lab `Inkling-2-Mini` (`inkling2` / `inkling2_loop`)
//! and 100%-GPU `DSpark` (`dflash`) block speculative drafter + recurrent Markov head.

use std::ffi::c_void;
use std::sync::Arc;

use tm_core::{DType, KvType, dequantize_row_bytes};
use tm_gguf::{DirectTensorReader, GgufFile, PageAlignedSlot};
use tm_models::{DFlashConfig, Inkling2Config, LOOP_CTRL_COUNT, LOOP_CTRL_FFN, RawMatView};

use crate::CudaError;
use crate::driver::{
    CUdeviceptr, CUfunction, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec, CudaHostRegistered,
    CudaPinnedBuffer,
};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat, dtype_cuda_tag};
use crate::llama_runner::{
    CudaProfileReport, CudaStageStat, MAX_CUDA_BATCH, is_q8_1_moe_dtype, upload_mat,
};

/// Callback for host-side single-token MoE evaluation (`ip, cur_normed, top_ids, top_weights, moe_out`).
pub type CudaHostMoeFn<'a> = dyn FnMut(usize, &[f32], &[usize], &[f32], &mut [f32]) + 'a;

/// Callback for host-side multi-token window MoE evaluation (`ip, w, cur_win, ids_win, ws_win, out_win`).
pub type CudaHostMoeWindowFn<'a> =
    dyn FnMut(usize, usize, &[f32], &[usize], &[f32], &mut [f32]) + 'a;

struct CudaImportedHostExperts {
    // Field drop order: unregister `CUdeviceptr` mappings BEFORE dropping `Arc` host pins!
    gu_reg: CudaHostRegistered,
    dn_reg: CudaHostRegistered,
    _gu_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
    _dn_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
}

const LOOP_CTRL_NAMES: [&str; LOOP_CTRL_COUNT] = ["q", "k", "v", "ffn"];

/// Maximum speculative verification window with `ShortConv` history snapshots (`DSpark` `block_size == 8`).
pub const MAX_INKLING2_CUDA_SPEC_WINDOW: usize = 16;

struct Inkling2CudaKernels {
    rms_norm_unit: CUfunction,
    mix_depth: CUfunction,
    control_prep: CUfunction,
    control_apply: CUfunction,
    shortconv: CUfunction,
    clip_heads: CUfunction,
    situ_glu_fused: CUfunction,
    rel_attn_f16: CUfunction,
    softcap_inplace: CUfunction,
    softcap_argmax: CUfunction,
}

impl Inkling2CudaKernels {
    fn load(ctx: &CudaContext) -> Result<Self, CudaError> {
        Ok(Self {
            rms_norm_unit: ctx.get_function("inkling2_rms_norm_unit_f32")?,
            mix_depth: ctx.get_function("inkling2_mix_depth_f32")?,
            control_prep: ctx.get_function("inkling2_control_prep_f32")?,
            control_apply: ctx.get_function("inkling2_control_apply_f32")?,
            shortconv: ctx.get_function("inkling2_shortconv_f32")?,
            clip_heads: ctx.get_function("inkling2_clip_heads_f32")?,
            situ_glu_fused: ctx.get_function("inkling2_situ_glu_fused_f32")?,
            rel_attn_f16: ctx.get_function("inkling2_rel_attn_f16_kv")?,
            softcap_inplace: ctx.get_function("softcap_inplace_f32")?,
            softcap_argmax: ctx.get_function("softcap_argmax_f32")?,
        })
    }
}

struct CudaLoopLayer {
    msg_query: CudaBuffer,
    ctrl_down: [CudaMat; LOOP_CTRL_COUNT],
    ctrl_gain: [CudaMat; LOOP_CTRL_COUNT],
    ctrl_forget: [CudaMat; LOOP_CTRL_COUNT],
}

struct CudaInklingLayer {
    wq: CudaMat,
    wk: CudaMat,
    wv: CudaMat,
    wo: CudaMat,
    wqkv_gate: CudaMat,
    attn_sink_proj: CudaMat,
    attn_rel: CudaMat,
    attn_k_conv_proj: Option<CudaMat>,
    attn_v_conv_proj: Option<CudaMat>,
    attn_q_clip: CudaBuffer,
    attn_rel_bias: CudaBuffer,
    attn_rel_clip: CudaBuffer,
    attn_k_conv: CudaBuffer,
    attn_v_conv: CudaBuffer,
    attn_post_conv: CudaBuffer,
    ffn_post_conv: CudaBuffer,
    attn_norm: Option<CudaBuffer>,
    attn_q_inp_norm: Option<CudaBuffer>,
    attn_k_inp_norm: Option<CudaBuffer>,
    attn_v_inp_norm: Option<CudaBuffer>,
    attn_res_query: Option<CudaBuffer>,
    ffn_res_query: CudaBuffer,
    ffn_norm: CudaBuffer,
    ffn_gate_inp: CudaMat,
    exp_probs_b: Option<CudaBuffer>,
    gate_up_exps: Option<CudaBuffer>,
    gate_up_dtype: i32,
    gate_up_row_bytes: usize,
    gate_up_exp_bytes: usize,
    gate_up_scales: Option<CudaBuffer>,
    down_exps: Option<CudaBuffer>,
    down_dtype: i32,
    down_row_bytes: usize,
    down_exp_bytes: usize,
    down_scales: Option<CudaBuffer>,
    host_imported_experts: Option<CudaImportedHostExperts>,
}

fn load_f32_dev(
    ctx: &CudaContext,
    gguf: &GgufFile,
    mut direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    expected_dims: &[usize],
) -> Result<CudaBuffer, CudaError> {
    let n_elem: usize = expected_dims.iter().product();
    let mut f32_vec = vec![0.0f32; n_elem];
    if let Some((rdr, ref mut slot)) = direct {
        let (ti, off, len) = gguf.req_tensor_file_range(name, layer, expected_dims)?;
        let mut raw = vec![0u8; len];
        rdr.read_exact_uncached(off, &mut raw, slot)?;
        dequantize_row_bytes(ti.dtype, &raw, &mut f32_vec);
    } else {
        let (ti, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
        dequantize_row_bytes(ti.dtype, bytes, &mut f32_vec);
    }
    ctx.alloc_from_f32(&f32_vec)
}

fn load_mat_dev(
    ctx: &CudaContext,
    gguf: &GgufFile,
    mut direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<(CudaMat, usize), CudaError> {
    if let Some((rdr, ref mut slot)) = direct {
        let (ti, off, len) = gguf.req_tensor_file_range(name, layer, &[cols, rows])?;
        let mut raw = vec![0u8; len];
        rdr.read_exact_uncached(off, &mut raw, slot)?;
        let view = RawMatView {
            name: "",
            bytes: &raw,
            dtype: ti.dtype,
            rows,
            cols,
            row_offset: 0,
            is_sliced: false,
            lora_slot: None,
        };
        Ok((upload_mat(ctx, &view)?, len))
    } else {
        let (ti, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
        let view = RawMatView {
            name: "",
            bytes,
            dtype: ti.dtype,
            rows,
            cols,
            row_offset: 0,
            is_sliced: false,
            lora_slot: None,
        };
        Ok((upload_mat(ctx, &view)?, bytes.len()))
    }
}

/// All-VRAM or Loop-Aware 3-Tier (`VRAM + Zero-Copy DEVICEMAP Host RAM + NVMe O_DIRECT`)
/// CUDA runner for `Inkling-2-Mini` (`inkling2` / `inkling2_loop`).
pub struct Inkling2CudaRunner {
    pub ctx: CudaContext,
    pub kernels: CudaKernels,
    ik: Inkling2CudaKernels,
    pub cfg: Inkling2Config,
    pub max_ctx: usize,
    pub batch_cap: usize,
    pub vram_bytes: usize,
    pub model_weight_bytes: usize,
    pub active_decode_bytes: usize,
    pub host_offloaded_moe_layers: usize,
    vram_expert_mask: Vec<bool>,
    hybrid_gpu_exps: usize,
    core_vram_prefix: usize,

    pub(crate) token_embd: CudaMat,
    pub(crate) token_embd_norm: CudaBuffer,
    output_res_query: CudaBuffer,
    output_norm: CudaBuffer,
    pub(crate) output: CudaMat,
    layers: Vec<CudaInklingLayer>,
    loop_layers: Vec<CudaLoopLayer>,

    // Per-logical-slot state (`n_layers = 48` on `Inkling-2-Mini`)
    k_caches: Vec<CudaBuffer>,
    v_caches: Vec<CudaBuffer>,
    shortconv_hist: CudaBuffer,
    slot_hist_offs: Vec<usize>,
    total_hist_floats: usize,
    shortconv_snap: CudaBuffer,

    // Pre-allocated Arena scratch on GPU (`0` allocations on forward/decode/verify!)
    pinned_ids: CudaPinnedBuffer,
    pinned_pos: CudaPinnedBuffer,
    pinned_moe_x: CudaPinnedBuffer,
    pinned_moe_ids: CudaPinnedBuffer,
    pinned_moe_ws: CudaPinnedBuffer,
    pinned_moe_out: CudaPinnedBuffer,
    ev_dtoh: CudaEvent,
    host_top_ids_scratch: Vec<usize>,
    host_top_ws_scratch: Vec<f32>,
    host_x_scratch: Vec<f32>,
    host_out_scratch: Vec<f32>,
    pub(crate) token_ids_buf: CudaBuffer,
    pos_buf: CudaBuffer,
    decode_graphs: [Option<CudaGraphExec>; 3],
    core_prefix_graphs: [Option<CudaGraphExec>; 3],
    pub(crate) cur_buf: CudaBuffer,
    inp_sa_buf: CudaBuffer,
    ffn_inp_buf: CudaBuffer,
    prelude_buf: CudaBuffer,
    bank_out_buf: CudaBuffer,
    blocks_buf: CudaBuffer,
    prelude_blocks_buf: CudaBuffer,
    bank_buf: CudaBuffer,
    inputs_buf: CudaBuffer,
    messages_buf: CudaBuffer,
    ctrl_in_buf: CudaBuffer,
    ctrl_g_buf: CudaBuffer,
    ctrl_gain_buf: CudaBuffer,
    ctrl_forget_buf: CudaBuffer,
    q_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    r_buf: CudaBuffer,
    sinks_buf: CudaBuffer,
    gate_buf: CudaBuffer,
    k_dyn_buf: CudaBuffer,
    v_dyn_buf: CudaBuffer,
    attn_vec_buf: CudaBuffer,
    wo_out_buf: CudaBuffer,
    xq_scratch: CudaBuffer,
    moe_logits: CudaBuffer,
    moe_top_ids: CudaBuffer,
    moe_top_weights: CudaBuffer,
    moe_exp_offs: CudaBuffer,
    moe_tile_offs: CudaBuffer,
    moe_sorted_ts: CudaBuffer,
    moe_exp_gu: CudaBuffer,
    moe_exp_hid: CudaBuffer,
    moe_slot_outs: CudaBuffer,
    moe_exp_xq: CudaBuffer,
    moe_out_buf: CudaBuffer,
    pub(crate) dspark_taps_buf: CudaBuffer,
    dspark_tap_slot: usize,
    pub(crate) logits_padded_buf: CudaBuffer,
    argmax_dev: CudaBuffer,
    logits_host: Vec<f32>,
}

impl Inkling2CudaRunner {
    pub fn from_gguf_with_kv_type(
        gguf: &GgufFile,
        max_ctx: usize,
        kv_type: KvType,
    ) -> Result<Self, CudaError> {
        Self::from_gguf_with_draft_reserve(gguf, max_ctx, kv_type, 0)
    }

    pub fn from_gguf_with_draft_reserve(
        gguf: &GgufFile,
        max_ctx: usize,
        _kv_type: KvType,
        draft_reserve_bytes: usize,
    ) -> Result<Self, CudaError> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let ctx = CudaContext::new(None)?;
        let kernels = CudaKernels::load(&ctx)?;
        let ik = Inkling2CudaKernels::load(&ctx)?;

        let n_phys = cfg.n_layers_phys;
        let mut total_file_bytes = 0usize;
        let mut dense_file_bytes = 0usize;
        let mut expert_layer_bytes = vec![0usize; n_phys];
        for tname in gguf.tensor_names() {
            if let Some(ti) = gguf.tensor_info(tname) {
                total_file_bytes += ti.byte_len;
                if let Some(rest) = tname.strip_prefix("blk.")
                    && let Some((idx_s, suf)) = rest.split_once('.')
                    && let Ok(ip) = idx_s.parse::<usize>()
                    && ip < n_phys
                    && suf.ends_with("_exps.weight")
                {
                    expert_layer_bytes[ip] += ti.byte_len;
                } else {
                    dense_file_bytes += ti.byte_len;
                }
            }
        }
        let mut kv_total_bytes = 0usize;
        for il in 0..cfg.n_layers {
            let ip = cfg.layer_phys(il);
            kv_total_bytes += 2 * max_ctx * cfg.kv_dim(ip) * 2;
        }
        let (free_vram, _) = ctx.mem_info()?;
        let env_vram_mib: Option<usize> = std::env::var("TM_CUDA_INKLING2_VRAM_MIB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());

        let all_fit = env_vram_mib.is_none()
            && total_file_bytes
                .saturating_add(kv_total_bytes)
                .saturating_add(192 * 1024 * 1024)
                .saturating_add(draft_reserve_bytes)
                <= free_vram;

        let mut vram_expert_mask = vec![false; n_phys];
        let batch_cap = if all_fit {
            vram_expert_mask.fill(true);
            MAX_CUDA_BATCH
        } else {
            let headroom = 340 * 1024 * 1024usize;
            let force_zero_vram_exps = env_vram_mib == Some(0);
            let vram_budget = match env_vram_mib {
                Some(0) | None => free_vram.saturating_sub(headroom),
                Some(m) => m * 1024 * 1024,
            };
            let scratch_est = 220 * 1024 * 1024usize;
            let base_needed = dense_file_bytes
                .saturating_add(kv_total_bytes)
                .saturating_add(scratch_est)
                .saturating_add(draft_reserve_bytes);
            if base_needed > vram_budget {
                return Err(CudaError::OutOfDeviceMemory {
                    needed_mib: base_needed as f64 / (1024.0 * 1024.0),
                    free_mib: vram_budget as f64 / (1024.0 * 1024.0),
                    max_ctx,
                    device: ctx.info().name.clone(),
                });
            }
            let mut rem = if force_zero_vram_exps {
                0
            } else {
                vram_budget - base_needed
            };
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
            128usize.min(MAX_CUDA_BATCH)
        };
        let default_hybrid_gpu_exps = (cfg.n_expert_used / 4).clamp(1, 4);
        let hybrid_gpu_exps = std::env::var("TM_CUDA_HYBRID_GPU_EXPS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(default_hybrid_gpu_exps)
            .min(cfg.n_expert_used);

        let sys_avail = tm_core::probe_system_memory()
            .map(|b| b.available_bytes as usize)
            .unwrap_or(32 * 1024 * 1024 * 1024);
        let use_direct =
            gguf.path().exists() && (!all_fit || total_file_bytes + (4 << 30) > sys_avail);
        let direct_rdr = if use_direct {
            DirectTensorReader::open(gguf.path()).ok()
        } else {
            None
        };
        let mut io_slot = PageAlignedSlot::new(16 * 1024 * 1024 + 8192);

        let d = cfg.n_embd;
        let n_ff = cfg.n_ff_exp;
        let n_exp = cfg.n_expert;
        let n_conv = cfg.shortconv_l_cache;

        let (token_embd, emb_b) = load_mat_dev(
            &ctx,
            gguf,
            direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
            "token_embd.weight",
            "global",
            d,
            cfg.vocab_padded,
        )?;
        let token_embd_norm = load_f32_dev(
            &ctx,
            gguf,
            direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
            "token_embd_norm.weight",
            "global",
            &[d],
        )?;
        let output_res_query = load_f32_dev(
            &ctx,
            gguf,
            direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
            "output_res_query.weight",
            "global",
            &[d],
        )?;
        let output_norm = load_f32_dev(
            &ctx,
            gguf,
            direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
            "output_norm.weight",
            "global",
            &[d],
        )?;
        let (output, out_b) = load_mat_dev(
            &ctx,
            gguf,
            direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
            "output.weight",
            "global",
            d,
            cfg.vocab_padded,
        )?;

        let mut model_weight_bytes = emb_b + out_b;
        let mut active_decode_bytes = out_b;
        let mut layers = Vec::with_capacity(cfg.n_layers_phys);

        for (ip, &in_vram) in vram_expert_mask.iter().enumerate() {
            let ltag = format!("blk.{ip}");
            let q_dim = cfg.q_dim(ip);
            let kv_dim = cfg.kv_dim(ip);
            let nh = cfg.n_head(ip);
            let nh_kv = cfg.n_head_kv(ip);
            let swa = cfg.is_swa[ip];
            let extent = cfg.rel_extent_for(ip);
            let n_rel = cfg.n_relative;
            let visits = if cfg.is_looped() && (cfg.core_start..cfg.core_end).contains(&ip) {
                cfg.n_passes
            } else {
                1
            };

            let (wq, b_wq) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_q.weight"),
                &ltag,
                d,
                q_dim,
            )?;
            let (wk, b_wk) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_k.weight"),
                &ltag,
                d,
                kv_dim,
            )?;
            let (wv, b_wv) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_v.weight"),
                &ltag,
                d,
                kv_dim,
            )?;
            let (wo, b_wo) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_output.weight"),
                &ltag,
                q_dim,
                d,
            )?;
            let (wqkv_gate, b_wg) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_gate.weight"),
                &ltag,
                d,
                q_dim,
            )?;
            let (attn_sink_proj, b_sp) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_sink_proj.weight"),
                &ltag,
                d,
                nh,
            )?;
            let (attn_rel, b_ar) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel.weight"),
                &ltag,
                d,
                nh * n_rel,
            )?;

            let (attn_k_conv_proj, attn_v_conv_proj, b_dyn) = if !swa {
                let (kp, b1) = load_mat_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_k_conv_proj.weight"),
                    &ltag,
                    d,
                    n_conv * nh_kv,
                )?;
                let (vp, b2) = load_mat_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_v_conv_proj.weight"),
                    &ltag,
                    d,
                    n_conv * nh_kv,
                )?;
                (Some(kp), Some(vp), b1 + b2)
            } else {
                (None, None, 0)
            };

            let attn_q_clip = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_q_clip.weight"),
                &ltag,
                &[cfg.head_dim, nh],
            )?;
            let attn_rel_bias = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel_bias.weight"),
                &ltag,
                &[n_rel, extent],
            )?;
            let attn_rel_clip = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel_clip.weight"),
                &ltag,
                &[n_rel, nh],
            )?;
            let attn_k_conv = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_k_conv.weight"),
                &ltag,
                &[n_conv, kv_dim],
            )?;
            let attn_v_conv = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_v_conv.weight"),
                &ltag,
                &[n_conv, kv_dim],
            )?;
            let attn_post_conv = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_post_conv.weight"),
                &ltag,
                &[n_conv, d],
            )?;
            let ffn_post_conv = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_post_conv.weight"),
                &ltag,
                &[n_conv, d],
            )?;

            let (attn_norm, attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm) = if swa {
                let an = load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_norm.weight"),
                    &ltag,
                    &[d],
                )?;
                (Some(an), None, None, None)
            } else {
                let qn = load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_q_inp_norm.weight"),
                    &ltag,
                    &[d],
                )?;
                let kn = load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_k_inp_norm.weight"),
                    &ltag,
                    &[d],
                )?;
                let vn = load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_v_inp_norm.weight"),
                    &ltag,
                    &[d],
                )?;
                (None, Some(qn), Some(kn), Some(vn))
            };

            let attn_res_query = if ip > 0 {
                let res_dims: &[usize] = if swa { &[d] } else { &[d, 3] };
                Some(load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_res_query.weight"),
                    &ltag,
                    res_dims,
                )?)
            } else {
                None
            };
            let ffn_res_query = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_res_query.weight"),
                &ltag,
                &[d],
            )?;
            let ffn_norm = load_f32_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_norm.weight"),
                &ltag,
                &[d],
            )?;
            let (ffn_gate_inp, b_gi) = load_mat_dev(
                &ctx,
                gguf,
                direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_gate_inp.weight"),
                &ltag,
                d,
                n_exp,
            )?;
            let exp_probs_b = if gguf
                .tensor_info(&format!("blk.{ip}.exp_probs_b.bias"))
                .is_some()
            {
                Some(load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.exp_probs_b.bias"),
                    &ltag,
                    &[n_exp],
                )?)
            } else {
                None
            };
            let gate_up_scales = if gguf
                .tensor_info(&format!("blk.{ip}.ffn_gate_up_exps.scale"))
                .is_some()
            {
                Some(load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.ffn_gate_up_exps.scale"),
                    &ltag,
                    &[n_exp],
                )?)
            } else {
                None
            };
            let down_scales = if gguf
                .tensor_info(&format!("blk.{ip}.ffn_down_exps.scale"))
                .is_some()
            {
                Some(load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.ffn_down_exps.scale"),
                    &ltag,
                    &[n_exp],
                )?)
            } else {
                None
            };

            let (gu_ti, gu_off, gu_len) = gguf.req_tensor_file_range(
                &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                &ltag,
                &[d, 2 * n_ff, n_exp],
            )?;
            let (dn_ti, dn_off, dn_len) = gguf.req_tensor_file_range(
                &format!("blk.{ip}.ffn_down_exps.weight"),
                &ltag,
                &[n_ff, d, n_exp],
            )?;
            let gu_dt = gu_ti.dtype;
            let dn_dt = dn_ti.dtype;
            let (gu_buf, dn_buf) = if in_vram {
                // Live VRAM guard: if CUDA driver page rounding or display server pressure left less
                // than `gu_len + dn_len + kv_total_bytes + draft_reserve_bytes + 280 MiB` free,
                // gracefully spill this layer to host `DEVICEMAP` instead of failing `cuMemAlloc_v2`!
                let live_ok = if all_fit {
                    true
                } else {
                    let (free_now, _) = ctx.mem_info()?;
                    let rem_layers = n_phys.saturating_sub(ip + 1);
                    let tail_need = kv_total_bytes
                        .saturating_add(draft_reserve_bytes)
                        .saturating_add(rem_layers * 42 * 1024 * 1024)
                        .saturating_add(240 * 1024 * 1024);
                    free_now >= gu_len.saturating_add(dn_len).saturating_add(tail_need)
                };
                if live_ok {
                    let gb = if let Some(ref rdr) = direct_rdr {
                        let mut raw = vec![0u8; gu_len];
                        rdr.read_exact_uncached(gu_off, &mut raw, &mut io_slot)?;
                        ctx.alloc_from_bytes(&raw)?
                    } else {
                        let (_, bytes) = gguf.req_tensor(
                            &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                            &ltag,
                            &[d, 2 * n_ff, n_exp],
                        )?;
                        ctx.alloc_from_bytes(bytes)?
                    };
                    let db = if let Some(ref rdr) = direct_rdr {
                        let mut raw = vec![0u8; dn_len];
                        rdr.read_exact_uncached(dn_off, &mut raw, &mut io_slot)?;
                        ctx.alloc_from_bytes(&raw)?
                    } else {
                        let (_, bytes) = gguf.req_tensor(
                            &format!("blk.{ip}.ffn_down_exps.weight"),
                            &ltag,
                            &[n_ff, d, n_exp],
                        )?;
                        ctx.alloc_from_bytes(bytes)?
                    };
                    (Some(gb), Some(db))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let dense_b = b_wq + b_wk + b_wv + b_wo + b_wg + b_sp + b_ar + b_dyn + b_gi;
            model_weight_bytes += dense_b + gu_len + dn_len;
            active_decode_bytes +=
                visits * (dense_b + cfg.n_expert_used * (gu_len / n_exp + dn_len / n_exp));

            layers.push(CudaInklingLayer {
                wq,
                wk,
                wv,
                wo,
                wqkv_gate,
                attn_sink_proj,
                attn_rel,
                attn_k_conv_proj,
                attn_v_conv_proj,
                attn_q_clip,
                attn_rel_bias,
                attn_rel_clip,
                attn_k_conv,
                attn_v_conv,
                attn_post_conv,
                ffn_post_conv,
                attn_norm,
                attn_q_inp_norm,
                attn_k_inp_norm,
                attn_v_inp_norm,
                attn_res_query,
                ffn_res_query,
                ffn_norm,
                ffn_gate_inp,
                exp_probs_b,
                gate_up_exps: gu_buf,
                gate_up_dtype: dtype_cuda_tag(gu_dt)?,
                gate_up_row_bytes: gu_dt.row_bytes(d),
                gate_up_exp_bytes: gu_len / n_exp,
                gate_up_scales,
                down_exps: dn_buf,
                down_dtype: dtype_cuda_tag(dn_dt)?,
                down_row_bytes: dn_dt.row_bytes(n_ff),
                down_exp_bytes: dn_len / n_exp,
                down_scales,
                host_imported_experts: None,
            });
        }
        for (ip, l) in layers.iter().enumerate() {
            vram_expert_mask[ip] = l.gate_up_exps.is_some();
        }
        let host_offloaded_moe_layers = vram_expert_mask.iter().filter(|&&v| !v).count();
        let mut core_vram_prefix = 0usize;
        if cfg.is_looped() {
            for &in_vram in &vram_expert_mask[cfg.core_start..cfg.core_end] {
                if in_vram {
                    core_vram_prefix += 1;
                } else {
                    break;
                }
            }
        }

        let mut loop_layers = Vec::new();
        if cfg.is_looped() {
            let rank = cfg.gate_rank;
            for ip in cfg.core_start..cfg.core_end {
                let ltag = format!("blk.{ip}");
                let msg_query = load_f32_dev(
                    &ctx,
                    gguf,
                    direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.loop_msg_query.weight"),
                    &ltag,
                    &[d, LOOP_CTRL_COUNT],
                )?;
                let mut downs = Vec::with_capacity(LOOP_CTRL_COUNT);
                let mut gains = Vec::with_capacity(LOOP_CTRL_COUNT);
                let mut forgets = Vec::with_capacity(LOOP_CTRL_COUNT);
                for cname in LOOP_CTRL_NAMES {
                    let (md, b1) = load_mat_dev(
                        &ctx,
                        gguf,
                        direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_down.weight"),
                        &ltag,
                        2 * d,
                        rank,
                    )?;
                    let (mg, b2) = load_mat_dev(
                        &ctx,
                        gguf,
                        direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_gain.weight"),
                        &ltag,
                        rank,
                        d,
                    )?;
                    let (mf, b3) = load_mat_dev(
                        &ctx,
                        gguf,
                        direct_rdr.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_forget.weight"),
                        &ltag,
                        rank,
                        d,
                    )?;
                    let ctrl_b = b1 + b2 + b3;
                    model_weight_bytes += ctrl_b;
                    active_decode_bytes += (cfg.n_passes - 1) * ctrl_b;
                    downs.push(md);
                    gains.push(mg);
                    forgets.push(mf);
                }
                loop_layers.push(CudaLoopLayer {
                    msg_query,
                    ctrl_down: downs.try_into().ok().expect("4 loop ctrl_down"),
                    ctrl_gain: gains.try_into().ok().expect("4 loop ctrl_gain"),
                    ctrl_forget: forgets.try_into().ok().expect("4 loop ctrl_forget"),
                });
            }
        }

        let mut k_caches = Vec::with_capacity(cfg.n_layers);
        let mut v_caches = Vec::with_capacity(cfg.n_layers);
        let hist_len = n_conv.saturating_sub(1);
        let mut slot_hist_offs = Vec::with_capacity(cfg.n_layers);
        let mut total_hist_floats = 0usize;
        for il in 0..cfg.n_layers {
            let ip = cfg.layer_phys(il);
            let kv_b = max_ctx * cfg.kv_dim(ip) * 2;
            k_caches.push(ctx.alloc_zeroed(kv_b)?);
            v_caches.push(ctx.alloc_zeroed(kv_b)?);
            slot_hist_offs.push(total_hist_floats);
            total_hist_floats += (2 * cfg.kv_dim(ip) + 2 * d) * hist_len;
        }
        let shortconv_hist = ctx.alloc_zeroed(total_hist_floats.max(4) * 4)?;
        let shortconv_snap =
            ctx.alloc_zeroed(MAX_INKLING2_CUDA_SPEC_WINDOW * total_hist_floats.max(4) * 4)?;

        let max_nh = cfg.max_n_head();
        let max_nh_kv = cfg.max_n_head_kv();
        let max_q_dim = max_nh * cfg.head_dim;
        let max_kv_dim = max_nh_kv * cfg.head_dim;
        let max_blocks = cfg.max_depth_blocks();
        let ku = cfg.n_expert_used;
        let max_cols = (2 * d).max(max_q_dim).max(n_ff).next_multiple_of(32);

        let pinned_ids = ctx.alloc_pinned(batch_cap * 4)?;
        let pinned_pos = ctx.alloc_pinned(4)?;
        let moe_pin_cap = if host_offloaded_moe_layers > 0 {
            batch_cap
        } else {
            1
        };
        let pinned_moe_x = ctx.alloc_pinned(moe_pin_cap * d * 4)?;
        let pinned_moe_ids = ctx.alloc_pinned(moe_pin_cap * ku * 4)?;
        let pinned_moe_ws = ctx.alloc_pinned(moe_pin_cap * ku * 4)?;
        let pinned_moe_out = ctx.alloc_pinned(moe_pin_cap * d * 4)?;
        let ev_dtoh = CudaEvent::new(&ctx)?;
        let token_ids_buf = ctx.alloc_zeroed(batch_cap * 4)?;
        let pos_buf = ctx.alloc_zeroed(4)?;
        let cur_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let inp_sa_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let ffn_inp_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let prelude_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let bank_out_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let blocks_buf = ctx.alloc_zeroed(batch_cap * max_blocks * d * 4)?;
        let prelude_blocks_buf = ctx.alloc_zeroed(batch_cap * max_blocks * d * 4)?;
        let bank_buf = ctx.alloc_zeroed(batch_cap * max_blocks * d * 4)?;
        let inputs_buf = ctx.alloc_zeroed(batch_cap * 3 * d * 4)?;
        let messages_buf = ctx.alloc_zeroed(batch_cap * LOOP_CTRL_COUNT * d * 4)?;
        let ctrl_in_buf = ctx.alloc_zeroed(batch_cap * 2 * d * 4)?;
        let ctrl_g_buf = ctx.alloc_zeroed(batch_cap * cfg.gate_rank.max(1) * 4)?;
        let ctrl_gain_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let ctrl_forget_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let q_buf = ctx.alloc_zeroed(batch_cap * max_q_dim * 4)?;
        let k_buf = ctx.alloc_zeroed(batch_cap * max_kv_dim * 4)?;
        let v_buf = ctx.alloc_zeroed(batch_cap * max_kv_dim * 4)?;
        let r_buf = ctx.alloc_zeroed(batch_cap * max_nh * cfg.n_relative * 4)?;
        let sinks_buf = ctx.alloc_zeroed(batch_cap * max_nh * 4)?;
        let gate_buf = ctx.alloc_zeroed(batch_cap * max_q_dim * 4)?;
        let k_dyn_buf = ctx.alloc_zeroed(batch_cap * n_conv * max_nh_kv * 4)?;
        let v_dyn_buf = ctx.alloc_zeroed(batch_cap * n_conv * max_nh_kv * 4)?;
        let attn_vec_buf = ctx.alloc_zeroed(batch_cap * max_q_dim * 4)?;
        let wo_out_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let xq_scratch = ctx.alloc_zeroed(batch_cap * (max_cols / 32) * BLOCK_Q8_1_BYTES)?;
        let moe_logits = ctx.alloc_zeroed(batch_cap * n_exp * 4)?;
        let moe_top_ids = ctx.alloc_zeroed(batch_cap * ku * 4)?;
        let moe_top_weights = ctx.alloc_zeroed(batch_cap * ku * 4)?;
        let moe_exp_offs = ctx.alloc_zeroed((n_exp.max(1) + 2) * 4)?;
        let moe_tile_offs = ctx.alloc_zeroed((n_exp.max(1) + 2) * 4)?;
        let moe_sorted_ts = ctx.alloc_zeroed(batch_cap * ku * 4)?;
        let moe_exp_gu = ctx.alloc_zeroed(batch_cap * ku * 2 * n_ff * 4)?;
        let moe_exp_hid = ctx.alloc_zeroed(batch_cap * ku * n_ff * 4)?;
        let moe_slot_outs = ctx.alloc_zeroed(batch_cap * ku * d * 4)?;
        let moe_exp_xq =
            ctx.alloc_zeroed(batch_cap * ku * d.max(n_ff).div_ceil(32) * BLOCK_Q8_1_BYTES)?;
        let moe_out_buf = ctx.alloc_zeroed(batch_cap * d * 4)?;
        let dspark_taps_buf = ctx.alloc_zeroed(batch_cap * 8 * d * 4)?;
        let logits_padded_buf =
            ctx.alloc_zeroed(MAX_INKLING2_CUDA_SPEC_WINDOW * cfg.vocab_padded * 4)?;
        let argmax_dev = ctx.alloc_zeroed(MAX_INKLING2_CUDA_SPEC_WINDOW * 4)?;
        ctx.synchronize()?;

        let (free_after, _) = ctx.mem_info()?;
        let vram_bytes = free_vram.saturating_sub(free_after);

        Ok(Self {
            ctx,
            kernels,
            ik,
            cfg: cfg.clone(),
            max_ctx,
            batch_cap,
            vram_bytes,
            model_weight_bytes,
            active_decode_bytes,
            host_offloaded_moe_layers,
            vram_expert_mask,
            hybrid_gpu_exps,
            core_vram_prefix,
            token_embd,
            token_embd_norm,
            output_res_query,
            output_norm,
            output,
            layers,
            loop_layers,
            k_caches,
            v_caches,
            shortconv_hist,
            slot_hist_offs,
            total_hist_floats,
            shortconv_snap,
            pinned_ids,
            pinned_pos,
            pinned_moe_x,
            pinned_moe_ids,
            pinned_moe_ws,
            pinned_moe_out,
            ev_dtoh,
            host_top_ids_scratch: vec![0usize; moe_pin_cap * ku],
            host_top_ws_scratch: vec![0.0f32; moe_pin_cap * ku],
            host_x_scratch: vec![0.0f32; moe_pin_cap * d],
            host_out_scratch: vec![0.0f32; moe_pin_cap * d],
            token_ids_buf,
            pos_buf,
            decode_graphs: [None, None, None],
            core_prefix_graphs: [None, None, None],
            cur_buf,
            inp_sa_buf,
            ffn_inp_buf,
            prelude_buf,
            bank_out_buf,
            blocks_buf,
            prelude_blocks_buf,
            bank_buf,
            inputs_buf,
            messages_buf,
            ctrl_in_buf,
            ctrl_g_buf,
            ctrl_gain_buf,
            ctrl_forget_buf,
            q_buf,
            k_buf,
            v_buf,
            r_buf,
            sinks_buf,
            gate_buf,
            k_dyn_buf,
            v_dyn_buf,
            attn_vec_buf,
            wo_out_buf,
            xq_scratch,
            moe_logits,
            moe_top_ids,
            moe_top_weights,
            moe_exp_offs,
            moe_tile_offs,
            moe_sorted_ts,
            moe_exp_gu,
            moe_exp_hid,
            moe_slot_outs,
            moe_exp_xq,
            moe_out_buf,
            dspark_taps_buf,
            dspark_tap_slot: 0,
            logits_padded_buf,
            argmax_dev,
            logits_host: vec![0.0f32; cfg.vocab_size],
        })
    }

    pub fn set_dspark_tap_slot(&mut self, slot: usize) {
        self.dspark_tap_slot = slot;
    }

    #[inline]
    pub fn has_host_offload_layers(&self) -> bool {
        self.host_offloaded_moe_layers > 0
    }

    #[inline]
    pub fn vram_expert_mask(&self) -> &[bool] {
        &self.vram_expert_mask
    }

    pub fn weights_mode(&self) -> String {
        if self.host_offloaded_moe_layers == 0 {
            return "device-vram".to_string();
        }
        let mut n_vram = 0usize;
        let mut n_devmap = 0usize;
        let mut n_host = 0usize;
        for l in &self.layers {
            if l.gate_up_exps.is_some() {
                n_vram += 1;
            } else if l.host_imported_experts.is_some() {
                n_devmap += 1;
            } else {
                n_host += 1;
            }
        }
        format!("tiered_cuda(vram={n_vram},devmap={n_devmap},nvme={n_host})")
    }

    /// Best-effort zero-copy `CU_MEMHOSTREGISTER_DEVICEMAP` registration of a host RAM MoE layer (`P1-2`).
    /// Shares the exact same `HugePageBytes` allocation with CPU `AVX2` workers (`0` duplicate RAM!).
    pub fn register_host_moe_layer(
        &mut self,
        ip: usize,
        gu_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
        gu_len: usize,
        dn_pin: Arc<dyn AsRef<[u8]> + Send + Sync>,
        dn_len: usize,
    ) -> Result<bool, CudaError> {
        let Some(layer) = self.layers.get_mut(ip) else {
            return Ok(false);
        };
        if layer.gate_up_exps.is_some() || layer.host_imported_experts.is_some() {
            return Ok(false);
        }
        let gu_slice = (*gu_pin).as_ref();
        let dn_slice = (*dn_pin).as_ref();
        if gu_slice.len() < gu_len || dn_slice.len() < dn_len || gu_len == 0 || dn_len == 0 {
            return Ok(false);
        }
        let gu_ptr = gu_slice.as_ptr() as *mut u8;
        let dn_ptr = dn_slice.as_ptr() as *mut u8;
        if !(gu_ptr as usize).is_multiple_of(4096) || !(dn_ptr as usize).is_multiple_of(4096) {
            return Ok(false);
        }
        let gu_aligned = gu_slice.len() & !4095usize;
        let dn_aligned = dn_slice.len() & !4095usize;
        if gu_aligned < gu_len || dn_aligned < dn_len {
            return Ok(false);
        }
        // SAFETY: `gu_pin` and `dn_pin` are `Arc`-pinned `HugePageBytes` stored inside `CudaImportedHostExperts`
        // AFTER `gu_reg`/`dn_reg`, guaranteeing `Drop` unregisters `CUdeviceptr` before freeing host pages.
        let gu_reg = match unsafe { self.ctx.register_host_devicemap(gu_ptr, gu_aligned) } {
            Ok(r) => r,
            Err(_) => return Ok(false),
        };
        let dn_reg = match unsafe { self.ctx.register_host_devicemap(dn_ptr, dn_aligned) } {
            Ok(r) => r,
            Err(_) => return Ok(false),
        };
        layer.host_imported_experts = Some(CudaImportedHostExperts {
            gu_reg,
            dn_reg,
            _gu_pin: gu_pin,
            _dn_pin: dn_pin,
        });
        Ok(true)
    }

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        for kc in &self.k_caches {
            kc.zero()?;
        }
        for vc in &self.v_caches {
            vc.zero()?;
        }
        self.shortconv_hist.zero()?;
        self.ctx.synchronize()
    }

    /// Commits `committed` (`1..=w`) tokens from a speculative window by restoring the exact
    /// post-step `ShortConv` history snapshot at `committed - 1`.
    pub fn commit_speculative_window(&mut self, committed: usize) -> Result<(), CudaError> {
        if committed == 0 || self.total_hist_floats == 0 {
            return Ok(());
        }
        let snap_src = self.shortconv_snap.device_ptr()
            + ((committed - 1) * self.total_hist_floats * 4) as u64;
        self.kernels.launch_copy_2d_rows(
            &self.ctx,
            self.shortconv_hist.device_ptr(),
            snap_src,
            self.total_hist_floats,
            self.total_hist_floats,
            self.total_hist_floats,
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_mix_depth(
        &self,
        blocks_ptr: CUdeviceptr,
        cur_ptr: CUdeviceptr,
        queries_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        n_blocks: usize,
        n_queries: usize,
        blocks_tok_stride: usize,
        cur_tok_stride: usize,
        out_q_stride: usize,
        replicate_count: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut pb = blocks_ptr;
        let mut pc = cur_ptr;
        let mut pq = queries_ptr;
        let mut po = out_ptr;
        let mut c_d = self.cfg.n_embd as i32;
        let mut c_nb = n_blocks as i32;
        let mut c_bs = blocks_tok_stride as i32;
        let mut c_cs = cur_tok_stride as i32;
        let mut c_os = out_q_stride as i32;
        let mut c_rc = replicate_count as i32;
        let mut args: [*mut c_void; 10] = [
            (&mut pb as *mut CUdeviceptr).cast(),
            (&mut pc as *mut CUdeviceptr).cast(),
            (&mut pq as *mut CUdeviceptr).cast(),
            (&mut po as *mut CUdeviceptr).cast(),
            (&mut c_d as *mut i32).cast(),
            (&mut c_nb as *mut i32).cast(),
            (&mut c_bs as *mut i32).cast(),
            (&mut c_cs as *mut i32).cast(),
            (&mut c_os as *mut i32).cast(),
            (&mut c_rc as *mut i32).cast(),
        ];
        // SAFETY: Matches `inkling2_mix_depth_f32`.
        unsafe {
            self.ctx.launch(
                "inkling2_mix_depth_f32",
                self.ik.mix_depth,
                (n_queries as u32, n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_control(
        &self,
        cur_ptr: CUdeviceptr,
        msg_ptr: CUdeviceptr,
        loop_w: &CudaLoopLayer,
        ctrl_idx: usize,
        cur_tok_stride: usize,
        msg_tok_stride: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        {
            let mut pc = cur_ptr;
            let mut pm = msg_ptr;
            let mut pci = self.ctrl_in_buf.device_ptr();
            let mut cd = d as i32;
            let mut cs = cur_tok_stride as i32;
            let mut ms = msg_tok_stride as i32;
            let mut args: [*mut c_void; 6] = [
                (&mut pc as *mut CUdeviceptr).cast(),
                (&mut pm as *mut CUdeviceptr).cast(),
                (&mut pci as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut cs as *mut i32).cast(),
                (&mut ms as *mut i32).cast(),
            ];
            // SAFETY: Matches `inkling2_control_prep_f32`.
            unsafe {
                self.ctx.launch(
                    "inkling2_control_prep_f32",
                    self.ik.control_prep,
                    (2, n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }
        self.kernels.launch_linear(
            &self.ctx,
            &loop_w.ctrl_down[ctrl_idx],
            self.ctrl_in_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.ctrl_g_buf.device_ptr(),
            n_tok,
            true,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &loop_w.ctrl_gain[ctrl_idx],
            self.ctrl_g_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.ctrl_gain_buf.device_ptr(),
            n_tok,
            true,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &loop_w.ctrl_forget[ctrl_idx],
            self.ctrl_g_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.ctrl_forget_buf.device_ptr(),
            n_tok,
            true,
        )?;
        {
            let mut pc = cur_ptr;
            let mut pm = msg_ptr;
            let mut pg = self.ctrl_gain_buf.device_ptr();
            let mut pf = self.ctrl_forget_buf.device_ptr();
            let mut cd = d as i32;
            let mut cs = cur_tok_stride as i32;
            let mut ms = msg_tok_stride as i32;
            let mut args: [*mut c_void; 7] = [
                (&mut pc as *mut CUdeviceptr).cast(),
                (&mut pm as *mut CUdeviceptr).cast(),
                (&mut pg as *mut CUdeviceptr).cast(),
                (&mut pf as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut cs as *mut i32).cast(),
                (&mut ms as *mut i32).cast(),
            ];
            // SAFETY: Matches `inkling2_control_apply_f32`.
            unsafe {
                self.ctx.launch(
                    "inkling2_control_apply_f32",
                    self.ik.control_apply,
                    ((d as u32).div_ceil(256), n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_shortconv(
        &self,
        il: usize,
        cur_ptr: CUdeviceptr,
        weight_ptr: CUdeviceptr,
        dyn_taps_ptr: CUdeviceptr,
        n_channels: usize,
        chan_offset: usize,
        dyn_heads: usize,
        save_snapshots: bool,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let slot_base_floats = self.slot_hist_offs[il];
        let mut pc = cur_ptr;
        let mut pw = weight_ptr;
        let mut pd = dyn_taps_ptr;
        let mut ph = self.shortconv_hist.device_ptr() + (slot_base_floats * 4) as u64;
        let mut ps = if save_snapshots {
            self.shortconv_snap.device_ptr()
        } else {
            0
        };
        let mut c_nc = n_channels as i32;
        let mut c_cv = self.cfg.shortconv_l_cache as i32;
        let mut c_co = chan_offset as i32;
        let mut c_dh = dyn_heads as i32;
        let mut c_sb = slot_base_floats as i32;
        let mut c_st = self.total_hist_floats as i32;
        let mut c_nt = n_tok as i32;
        let mut args: [*mut c_void; 12] = [
            (&mut pc as *mut CUdeviceptr).cast(),
            (&mut pw as *mut CUdeviceptr).cast(),
            (&mut pd as *mut CUdeviceptr).cast(),
            (&mut ph as *mut CUdeviceptr).cast(),
            (&mut ps as *mut CUdeviceptr).cast(),
            (&mut c_nc as *mut i32).cast(),
            (&mut c_cv as *mut i32).cast(),
            (&mut c_co as *mut i32).cast(),
            (&mut c_dh as *mut i32).cast(),
            (&mut c_sb as *mut i32).cast(),
            (&mut c_st as *mut i32).cast(),
            (&mut c_nt as *mut i32).cast(),
        ];
        // SAFETY: Matches `inkling2_shortconv_f32`.
        unsafe {
            self.ctx.launch(
                "inkling2_shortconv_f32",
                self.ik.shortconv,
                ((n_channels as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_clip_heads(
        &self,
        vec_ptr: CUdeviceptr,
        scale_ptr: CUdeviceptr,
        n_heads: usize,
        dim: usize,
        thresh: f32,
        apply_tau: bool,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let mut pv = vec_ptr;
        let mut ps = scale_ptr;
        let mut nh = n_heads as i32;
        let mut cd = dim as i32;
        let mut ct = thresh;
        let mut at = i32::from(apply_tau);
        let mut lf = self.cfg.log_scale_floor as i32;
        let mut la = self.cfg.log_scale_alpha;
        let mut sp = start_pos as i32;
        let mut pp = pos_dev_ptr;
        let mut args: [*mut c_void; 10] = [
            (&mut pv as *mut CUdeviceptr).cast(),
            (&mut ps as *mut CUdeviceptr).cast(),
            (&mut nh as *mut i32).cast(),
            (&mut cd as *mut i32).cast(),
            (&mut ct as *mut f32).cast(),
            (&mut at as *mut i32).cast(),
            (&mut lf as *mut i32).cast(),
            (&mut la as *mut f32).cast(),
            (&mut sp as *mut i32).cast(),
            (&mut pp as *mut CUdeviceptr).cast(),
        ];
        // SAFETY: Matches `inkling2_clip_heads_f32`.
        unsafe {
            self.ctx.launch(
                "inkling2_clip_heads_f32",
                self.ik.clip_heads,
                (n_heads as u32, n_tok as u32, 1),
                (128, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer(
        &self,
        il: usize,
        ip: usize,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        has_messages: bool,
        save_snapshots: bool,
        blocks_len: &mut usize,
    ) -> Result<(), CudaError> {
        self.encode_layer_pre_moe(
            il,
            ip,
            n_tok,
            start_pos,
            pos_dev_ptr,
            has_messages,
            save_snapshots,
            blocks_len,
        )?;
        let layer = &self.layers[ip];
        let gu_ptr = layer
            .gate_up_exps
            .as_ref()
            .expect("VRAM gate_up_exps required in encode_layer")
            .device_ptr();
        let dn_ptr = layer
            .down_exps
            .as_ref()
            .expect("VRAM down_exps required in encode_layer")
            .device_ptr();
        self.encode_moe_from_ptrs(
            ip,
            gu_ptr,
            dn_ptr,
            n_tok,
            self.cfg.n_expert_used,
            self.cfg.n_expert,
            false,
        )?;
        self.encode_layer_post_moe(il, ip, n_tok, save_snapshots)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_pre_moe(
        &self,
        il: usize,
        ip: usize,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        has_messages: bool,
        save_snapshots: bool,
        blocks_len: &mut usize,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let swa = self.cfg.is_swa[ip];
        let nh = self.cfg.n_head(ip);
        let nh_kv = self.cfg.n_head_kv(ip);
        let hd = self.cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nh_kv * hd;
        let n_rel = self.cfg.n_relative;
        let extent = self.cfg.rel_extent_for(ip);
        let max_blocks = self.cfg.max_depth_blocks();
        let blk_tok_stride = max_blocks * d;
        let layer = &self.layers[ip];

        // `inp_sa = cur`
        self.kernels.launch_copy_2d_rows(
            &self.ctx,
            self.inp_sa_buf.device_ptr(),
            self.cur_buf.device_ptr(),
            n_tok * d,
            n_tok * d,
            n_tok * d,
            1,
        )?;

        // 1. Depth-residual mix -> `inputs_buf [n_tok, 3, d]`
        if let Some(ref rq) = layer.attn_res_query {
            if swa {
                self.launch_mix_depth(
                    self.blocks_buf.device_ptr(),
                    self.cur_buf.device_ptr(),
                    rq.device_ptr(),
                    self.inputs_buf.device_ptr(),
                    *blocks_len,
                    1,
                    blk_tok_stride,
                    d,
                    3,
                    3,
                    n_tok,
                )?;
            } else {
                self.launch_mix_depth(
                    self.blocks_buf.device_ptr(),
                    self.cur_buf.device_ptr(),
                    rq.device_ptr(),
                    self.inputs_buf.device_ptr(),
                    *blocks_len,
                    3,
                    blk_tok_stride,
                    d,
                    3,
                    1,
                    n_tok,
                )?;
            }
        } else {
            for j in 0..3 {
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    self.inputs_buf.device_ptr() + (j * d * 4) as u64,
                    self.cur_buf.device_ptr(),
                    d,
                    3 * d,
                    d,
                    n_tok,
                )?;
            }
        }

        // 2. Cross-pass loop controllers (`Q, K, V`) + input RMSNorm
        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            for j in 0..3 {
                self.launch_control(
                    self.inputs_buf.device_ptr() + (j * d * 4) as u64,
                    self.messages_buf.device_ptr() + (j * d * 4) as u64,
                    loop_w,
                    j,
                    3 * d,
                    LOOP_CTRL_COUNT * d,
                    n_tok,
                )?;
            }
        }

        // Copy `inputs[t, 0..3]` into contiguous `[n_tok, d]` slices (`ctrl_gain_buf`, `ctrl_forget_buf`, `wo_out_buf`)
        // after RMSNorm so all `launch_linear` calls see contiguous `[n_tok, d]` matrices!
        let q_in_ptr = self.ctrl_gain_buf.device_ptr();
        let k_in_ptr = self.ctrl_forget_buf.device_ptr();
        let v_in_ptr = self.wo_out_buf.device_ptr();

        if swa && !has_messages {
            let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                self.moe_out_buf.device_ptr(),
                self.inputs_buf.device_ptr(),
                d,
                d,
                3 * d,
                n_tok,
            )?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.moe_out_buf.device_ptr(),
                an.device_ptr(),
                q_in_ptr,
                d,
                n_tok,
                self.cfg.rms_eps,
            )?;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                k_in_ptr,
                q_in_ptr,
                n_tok * d,
                n_tok * d,
                n_tok * d,
                1,
            )?;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                v_in_ptr,
                q_in_ptr,
                n_tok * d,
                n_tok * d,
                n_tok * d,
                1,
            )?;
        } else {
            let norms: [&CudaBuffer; 3] = if swa {
                let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
                [an, an, an]
            } else {
                [
                    layer.attn_q_inp_norm.as_ref().expect("q_inp_norm"),
                    layer.attn_k_inp_norm.as_ref().expect("k_inp_norm"),
                    layer.attn_v_inp_norm.as_ref().expect("v_inp_norm"),
                ]
            };
            let dsts = [q_in_ptr, k_in_ptr, v_in_ptr];
            for j in 0..3 {
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    self.moe_out_buf.device_ptr(),
                    self.inputs_buf.device_ptr() + (j * d * 4) as u64,
                    d,
                    d,
                    3 * d,
                    n_tok,
                )?;
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.moe_out_buf.device_ptr(),
                    norms[j].device_ptr(),
                    dsts[j],
                    d,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
            }
        }

        // 3. Attention projections (`q_in_ptr` shared across `wq`, `attn_rel`, `attn_sink_proj`, `wqkv_gate`!)
        let can_share_q8 = n_tok <= crate::kernels::MMVQ_MAX_TOK && d.is_multiple_of(32);
        let q_uses_q8 = matches!(
            layer.wq.dtype,
            DType::Q2_0 | DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
        );
        if can_share_q8 && q_uses_q8 {
            self.kernels.launch_quantize_q8_1(
                &self.ctx,
                q_in_ptr,
                self.xq_scratch.device_ptr(),
                d,
                n_tok,
            )?;
        }
        self.kernels.launch_linear(
            &self.ctx,
            &layer.wq,
            q_in_ptr,
            self.xq_scratch.device_ptr(),
            can_share_q8 && q_uses_q8,
            self.q_buf.device_ptr(),
            n_tok,
            false,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &layer.wqkv_gate,
            q_in_ptr,
            self.xq_scratch.device_ptr(),
            can_share_q8 && q_uses_q8,
            self.gate_buf.device_ptr(),
            n_tok,
            false,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &layer.attn_rel,
            q_in_ptr,
            self.xq_scratch.device_ptr(),
            can_share_q8
                && q_uses_q8
                && matches!(
                    layer.attn_rel.dtype,
                    DType::Q2_0 | DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
                ),
            self.r_buf.device_ptr(),
            n_tok,
            true,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &layer.attn_sink_proj,
            q_in_ptr,
            self.xq_scratch.device_ptr(),
            can_share_q8
                && q_uses_q8
                && matches!(
                    layer.attn_sink_proj.dtype,
                    DType::Q2_0 | DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
                ),
            self.sinks_buf.device_ptr(),
            n_tok,
            true,
        )?;

        self.kernels.launch_linear(
            &self.ctx,
            &layer.wk,
            k_in_ptr,
            self.xq_scratch.device_ptr(),
            false,
            self.k_buf.device_ptr(),
            n_tok,
            false,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &layer.wv,
            v_in_ptr,
            self.xq_scratch.device_ptr(),
            false,
            self.v_buf.device_ptr(),
            n_tok,
            false,
        )?;

        if !swa {
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            self.kernels.launch_linear(
                &self.ctx,
                kp,
                k_in_ptr,
                self.xq_scratch.device_ptr(),
                false,
                self.k_dyn_buf.device_ptr(),
                n_tok,
                true,
            )?;
            self.kernels.launch_linear(
                &self.ctx,
                vp,
                v_in_ptr,
                self.xq_scratch.device_ptr(),
                false,
                self.v_dyn_buf.device_ptr(),
                n_tok,
                true,
            )?;
        }

        // K & V causal ShortConvs
        self.launch_shortconv(
            il,
            self.k_buf.device_ptr(),
            layer.attn_k_conv.device_ptr(),
            if swa { 0 } else { self.k_dyn_buf.device_ptr() },
            kv_dim,
            0,
            if swa { 0 } else { nh_kv },
            save_snapshots,
            n_tok,
        )?;
        self.launch_shortconv(
            il,
            self.v_buf.device_ptr(),
            layer.attn_v_conv.device_ptr(),
            if swa { 0 } else { self.v_dyn_buf.device_ptr() },
            kv_dim,
            kv_dim,
            if swa { 0 } else { nh_kv },
            save_snapshots,
            n_tok,
        )?;

        // Q/K/R clipping + V per-head unit RMSNorm
        self.launch_clip_heads(
            self.q_buf.device_ptr(),
            layer.attn_q_clip.device_ptr(),
            nh,
            hd,
            self.cfg.qk_clip,
            !swa,
            start_pos,
            pos_dev_ptr,
            n_tok,
        )?;
        self.launch_clip_heads(
            self.k_buf.device_ptr(),
            0,
            nh_kv,
            hd,
            self.cfg.qk_clip,
            false,
            start_pos,
            pos_dev_ptr,
            n_tok,
        )?;
        {
            let mut pv = self.v_buf.device_ptr();
            let mut po = self.v_buf.device_ptr();
            let mut chd = hd as i32;
            let mut eps = self.cfg.rms_eps;
            let mut args: [*mut c_void; 4] = [
                (&mut pv as *mut CUdeviceptr).cast(),
                (&mut po as *mut CUdeviceptr).cast(),
                (&mut chd as *mut i32).cast(),
                (&mut eps as *mut f32).cast(),
            ];
            // SAFETY: Matches `inkling2_rms_norm_unit_f32`.
            unsafe {
                self.ctx.launch(
                    "inkling2_rms_norm_unit_f32",
                    self.ik.rms_norm_unit,
                    ((n_tok * nh_kv) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }
        self.launch_clip_heads(
            self.r_buf.device_ptr(),
            layer.attn_rel_clip.device_ptr(),
            nh,
            n_rel,
            self.cfg.relative_clip,
            !swa,
            start_pos,
            pos_dev_ptr,
            n_tok,
        )?;

        // Write F16 KV cache for logical slot `il`
        self.kernels.launch_write_kv_f16(
            &self.ctx,
            self.k_buf.device_ptr(),
            self.v_buf.device_ptr(),
            self.k_caches[il].device_ptr(),
            self.v_caches[il].device_ptr(),
            kv_dim,
            start_pos,
            pos_dev_ptr,
            n_tok,
        )?;

        // Relative-bias + zero-sink GQA attention
        {
            let mut pq = self.q_buf.device_ptr();
            let mut pr = self.r_buf.device_ptr();
            let mut prb = layer.attn_rel_bias.device_ptr();
            let mut ps = self.sinks_buf.device_ptr();
            let mut pkc = self.k_caches[il].device_ptr();
            let mut pvc = self.v_caches[il].device_ptr();
            let mut po = self.attn_vec_buf.device_ptr();
            let mut c_nh = nh as i32;
            let mut c_nhkv = nh_kv as i32;
            let mut c_hd = hd as i32;
            let mut c_nr = n_rel as i32;
            let mut c_ex = extent as i32;
            let mut c_swa = if swa { self.cfg.n_swa as i32 } else { 0 };
            let mut c_sp = start_pos as i32;
            let mut pp = pos_dev_ptr;
            let mut args: [*mut c_void; 15] = [
                (&mut pq as *mut CUdeviceptr).cast(),
                (&mut pr as *mut CUdeviceptr).cast(),
                (&mut prb as *mut CUdeviceptr).cast(),
                (&mut ps as *mut CUdeviceptr).cast(),
                (&mut pkc as *mut CUdeviceptr).cast(),
                (&mut pvc as *mut CUdeviceptr).cast(),
                (&mut po as *mut CUdeviceptr).cast(),
                (&mut c_nh as *mut i32).cast(),
                (&mut c_nhkv as *mut i32).cast(),
                (&mut c_hd as *mut i32).cast(),
                (&mut c_nr as *mut i32).cast(),
                (&mut c_ex as *mut i32).cast(),
                (&mut c_swa as *mut i32).cast(),
                (&mut c_sp as *mut i32).cast(),
                (&mut pp as *mut CUdeviceptr).cast(),
            ];
            // SAFETY: Matches `inkling2_rel_attn_f16_kv`.
            unsafe {
                self.ctx.launch(
                    "inkling2_rel_attn_f16_kv",
                    self.ik.rel_attn_f16,
                    (nh as u32, n_tok as u32, 1),
                    (32, 8, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        // Sigmoid gate + `wo` + `attn_post_conv`
        self.kernels.launch_sigmoid_gate_mul(
            &self.ctx,
            self.attn_vec_buf.device_ptr(),
            self.gate_buf.device_ptr(),
            n_tok * q_dim,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &layer.wo,
            self.attn_vec_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.wo_out_buf.device_ptr(),
            n_tok,
            false,
        )?;
        self.launch_shortconv(
            il,
            self.wo_out_buf.device_ptr(),
            layer.attn_post_conv.device_ptr(),
            0,
            d,
            2 * kv_dim,
            0,
            save_snapshots,
            n_tok,
        )?;

        // Block residual stream boundary (`ip % attn_res_block_size == 0`)
        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                self.blocks_buf.device_ptr() + (*blocks_len * d * 4) as u64,
                self.inp_sa_buf.device_ptr(),
                d,
                blk_tok_stride,
                d,
                n_tok,
            )?;
            *blocks_len += 1;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                self.cur_buf.device_ptr(),
                self.wo_out_buf.device_ptr(),
                n_tok * d,
                n_tok * d,
                n_tok * d,
                1,
            )?;
        } else {
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                self.cur_buf.device_ptr(),
                self.inp_sa_buf.device_ptr(),
                n_tok * d,
                n_tok * d,
                n_tok * d,
                1,
            )?;
            self.kernels.launch_add_scaled(
                &self.ctx,
                self.cur_buf.device_ptr(),
                self.wo_out_buf.device_ptr(),
                1.0,
                n_tok * d,
            )?;
        }

        // `ffn_inp = cur`
        self.kernels.launch_copy_2d_rows(
            &self.ctx,
            self.ffn_inp_buf.device_ptr(),
            self.cur_buf.device_ptr(),
            n_tok * d,
            n_tok * d,
            n_tok * d,
            1,
        )?;

        // 4. MoE FFN (`mix` -> optional `control(FFN)` -> `ffn_norm` -> 128-choose-8 `SiTU` MoE -> `ffn_post_conv`)
        self.launch_mix_depth(
            self.blocks_buf.device_ptr(),
            self.cur_buf.device_ptr(),
            layer.ffn_res_query.device_ptr(),
            self.inp_sa_buf.device_ptr(),
            *blocks_len,
            1,
            blk_tok_stride,
            d,
            1,
            1,
            n_tok,
        )?;
        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            self.launch_control(
                self.inp_sa_buf.device_ptr(),
                self.messages_buf.device_ptr() + (LOOP_CTRL_FFN * d * 4) as u64,
                loop_w,
                LOOP_CTRL_FFN,
                d,
                LOOP_CTRL_COUNT * d,
                n_tok,
            )?;
        }
        self.kernels.launch_rms_norm(
            &self.ctx,
            self.inp_sa_buf.device_ptr(),
            layer.ffn_norm.device_ptr(),
            self.wo_out_buf.device_ptr(),
            d,
            n_tok,
            self.cfg.rms_eps,
        )?;

        // Router (`Sigmoid` + `exp_probs_b`, `norm_weights = true`, `norm_eps = 1e-20`, `weights_scale`)
        self.kernels.launch_linear(
            &self.ctx,
            &layer.ffn_gate_inp,
            self.wo_out_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.moe_logits.device_ptr(),
            n_tok,
            false,
        )?;
        let ne = self.cfg.n_expert;
        let ku = self.cfg.n_expert_used;
        self.kernels.launch_moe_router_topk(
            &self.ctx,
            self.moe_logits.device_ptr(),
            layer.exp_probs_b.as_ref().map_or(0, |b| b.device_ptr()),
            self.moe_top_ids.device_ptr(),
            self.moe_top_weights.device_ptr(),
            ne,
            ku,
            2, // Sigmoid
            true,
            1.0e-20,
            self.cfg.expert_weights_scale,
            n_tok,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_moe_from_ptrs(
        &self,
        ip: usize,
        gu_exps_ptr: CUdeviceptr,
        dn_exps_ptr: CUdeviceptr,
        n_tok: usize,
        active_slots: usize,
        active_n_exp: usize,
        zero_slots_first: bool,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let ku = self.cfg.n_expert_used;
        let n_ff = self.cfg.n_ff_exp;
        let layer = &self.layers[ip];
        let gu_scales_ptr = layer.gate_up_scales.as_ref().map_or(0, |b| b.device_ptr());
        let dn_scales_ptr = layer.down_scales.as_ref().map_or(0, |b| b.device_ptr());

        if n_tok >= 16
            && active_n_exp <= 512
            && self.ctx.info().has_fp16_wmma()
            && d.is_multiple_of(32)
            && n_ff.is_multiple_of(32)
        {
            let rows_slots = n_tok * ku;
            if zero_slots_first {
                self.moe_slot_outs.zero_prefix(rows_slots * d * 4)?;
            }
            self.kernels.launch_moe_build_expert_buckets(
                &self.ctx,
                self.moe_top_ids.device_ptr(),
                self.moe_exp_offs.device_ptr(),
                self.moe_tile_offs.device_ptr(),
                self.moe_sorted_ts.device_ptr(),
                rows_slots,
                active_n_exp,
            )?;
            self.kernels.launch_wmma_moe_grouped(
                &self.ctx,
                gu_exps_ptr,
                0,
                0,
                self.wo_out_buf.device_ptr(),
                self.moe_exp_offs.device_ptr(),
                self.moe_tile_offs.device_ptr(),
                self.moe_sorted_ts.device_ptr(),
                self.moe_top_weights.device_ptr(),
                gu_scales_ptr,
                self.moe_exp_gu.device_ptr(),
                layer.gate_up_dtype,
                2 * n_ff,
                d,
                layer.gate_up_row_bytes,
                layer.gate_up_exp_bytes,
                0,
                active_n_exp,
                ku,
                n_tok,
                0,
                0,
            )?;
            let total = (rows_slots * n_ff) as u32;
            let mut pgu = self.moe_exp_gu.device_ptr();
            let mut po = self.moe_exp_hid.device_ptr();
            let mut c_nff = n_ff as i32;
            let mut c_nr = rows_slots as i32;
            let mut beta = self.cfg.situ_beta;
            let mut lb = self.cfg.situ_linear_beta;
            let mut args: [*mut c_void; 6] = [
                (&mut pgu as *mut CUdeviceptr).cast(),
                (&mut po as *mut CUdeviceptr).cast(),
                (&mut c_nff as *mut i32).cast(),
                (&mut c_nr as *mut i32).cast(),
                (&mut beta as *mut f32).cast(),
                (&mut lb as *mut f32).cast(),
            ];
            // SAFETY: Matches `inkling2_situ_glu_fused_f32`.
            unsafe {
                self.ctx.launch(
                    "inkling2_situ_glu_fused_f32",
                    self.ik.situ_glu_fused,
                    (total.div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
            self.kernels.launch_wmma_moe_grouped(
                &self.ctx,
                dn_exps_ptr,
                0,
                0,
                self.moe_exp_hid.device_ptr(),
                self.moe_exp_offs.device_ptr(),
                self.moe_tile_offs.device_ptr(),
                self.moe_sorted_ts.device_ptr(),
                self.moe_top_weights.device_ptr(),
                dn_scales_ptr,
                self.moe_slot_outs.device_ptr(),
                layer.down_dtype,
                d,
                n_ff,
                layer.down_row_bytes,
                layer.down_exp_bytes,
                0,
                active_n_exp,
                ku,
                n_tok,
                1,
                1,
            )?;
            self.kernels.launch_moe_reduce_slots(
                &self.ctx,
                self.moe_slot_outs.device_ptr(),
                self.moe_out_buf.device_ptr(),
                d,
                ku,
                n_tok,
            )
        } else {
            let eff_slots = active_slots;
            let rows_slots = n_tok * eff_slots;
            let in_xq_ptr = if is_q8_1_moe_dtype(layer.gate_up_dtype, d) {
                self.kernels.launch_quantize_q8_1(
                    &self.ctx,
                    self.wo_out_buf.device_ptr(),
                    self.moe_exp_xq.device_ptr(),
                    d,
                    n_tok,
                )?;
                self.moe_exp_xq.device_ptr()
            } else {
                0
            };

            self.kernels.launch_moe_indexed_dual_gu_act(
                &self.ctx,
                gu_exps_ptr,
                gu_exps_ptr,
                0,
                0,
                0,
                in_xq_ptr,
                self.wo_out_buf.device_ptr(),
                self.moe_top_ids.device_ptr(),
                gu_scales_ptr,
                gu_scales_ptr,
                self.moe_exp_hid.device_ptr(),
                layer.gate_up_dtype,
                layer.gate_up_dtype,
                n_ff,
                d,
                layer.gate_up_row_bytes,
                layer.gate_up_row_bytes,
                layer.gate_up_exp_bytes,
                layer.gate_up_exp_bytes,
                n_ff * layer.gate_up_row_bytes,
                eff_slots,
                n_tok,
                1,
                self.cfg.situ_beta,
                self.cfg.situ_linear_beta,
            )?;

            let dn_uses_q8 = is_q8_1_moe_dtype(layer.down_dtype, n_ff);
            let dn_xq_ptr = if dn_uses_q8 {
                self.kernels.launch_quantize_q8_1(
                    &self.ctx,
                    self.moe_exp_hid.device_ptr(),
                    self.moe_exp_xq.device_ptr(),
                    n_ff,
                    rows_slots,
                )?;
                self.moe_exp_xq.device_ptr()
            } else {
                0
            };

            self.kernels.launch_moe_indexed_gemv_q8_1(
                &self.ctx,
                dn_exps_ptr,
                0,
                0,
                dn_xq_ptr,
                self.moe_exp_hid.device_ptr(),
                self.moe_top_ids.device_ptr(),
                self.moe_top_weights.device_ptr(),
                dn_scales_ptr,
                self.moe_slot_outs.device_ptr(),
                layer.down_dtype,
                d,
                n_ff,
                layer.down_row_bytes,
                layer.down_exp_bytes,
                eff_slots,
                n_tok,
                1,
                1,
            )?;
            self.kernels.launch_moe_reduce_slots(
                &self.ctx,
                self.moe_slot_outs.device_ptr(),
                self.moe_out_buf.device_ptr(),
                d,
                eff_slots,
                n_tok,
            )
        }
    }

    fn encode_layer_post_moe(
        &self,
        il: usize,
        ip: usize,
        n_tok: usize,
        save_snapshots: bool,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let kv_dim = self.cfg.kv_dim(ip);
        let layer = &self.layers[ip];
        self.launch_shortconv(
            il,
            self.moe_out_buf.device_ptr(),
            layer.ffn_post_conv.device_ptr(),
            0,
            d,
            2 * kv_dim + d,
            0,
            save_snapshots,
            n_tok,
        )?;

        self.kernels.launch_copy_2d_rows(
            &self.ctx,
            self.cur_buf.device_ptr(),
            self.ffn_inp_buf.device_ptr(),
            n_tok * d,
            n_tok * d,
            n_tok * d,
            1,
        )?;
        self.kernels.launch_add_scaled(
            &self.ctx,
            self.cur_buf.device_ptr(),
            self.moe_out_buf.device_ptr(),
            1.0,
            n_tok * d,
        )
    }

    fn maybe_capture_visit_tap(
        &self,
        visit: usize,
        src_ptr: CUdeviceptr,
        dspark_visits: &[usize],
        n_tok: usize,
    ) -> Result<(), CudaError> {
        if dspark_visits.is_empty() {
            return Ok(());
        }
        let d = self.cfg.n_embd;
        let n_taps = dspark_visits.len();
        for (tap_idx, &v) in dspark_visits.iter().enumerate() {
            if v == visit {
                let dst = self.dspark_taps_buf.device_ptr()
                    + (((self.dspark_tap_slot * n_taps + tap_idx) * d) * 4) as u64;
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    dst,
                    src_ptr,
                    d,
                    n_taps * d,
                    d,
                    n_tok,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_chunk(
        &self,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        preloaded_embed: bool,
        compute_last_logits: bool,
        compute_all_argmax: bool,
        save_snapshots: bool,
        dspark_visits: &[usize],
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let max_blocks = self.cfg.max_depth_blocks();
        let blk_tok_stride = max_blocks * d;

        if !preloaded_embed {
            self.kernels.launch_embed(
                &self.ctx,
                &self.token_embd,
                self.token_ids_buf.device_ptr(),
                self.inp_sa_buf.device_ptr(),
                n_tok,
                1.0,
            )?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.inp_sa_buf.device_ptr(),
                self.token_embd_norm.device_ptr(),
                self.cur_buf.device_ptr(),
                d,
                n_tok,
                self.cfg.rms_eps,
            )?;
        }

        let mut blocks_len = 0usize;
        let mut visit = 0usize;
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                self.maybe_capture_visit_tap(
                    visit,
                    self.cur_buf.device_ptr(),
                    dspark_visits,
                    n_tok,
                )?;
                visit += 1;
                self.encode_layer(
                    il,
                    il,
                    n_tok,
                    start_pos,
                    pos_dev_ptr,
                    false,
                    save_snapshots,
                    &mut blocks_len,
                )?;
            }
        } else {
            let cs = self.cfg.core_start;
            let ce = self.cfg.core_end;
            for il in 0..cs {
                self.maybe_capture_visit_tap(
                    visit,
                    self.cur_buf.device_ptr(),
                    dspark_visits,
                    n_tok,
                )?;
                visit += 1;
                self.encode_layer(
                    il,
                    il,
                    n_tok,
                    start_pos,
                    pos_dev_ptr,
                    false,
                    save_snapshots,
                    &mut blocks_len,
                )?;
            }

            let prelude_blocks_len = blocks_len;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                self.prelude_buf.device_ptr(),
                self.cur_buf.device_ptr(),
                n_tok * d,
                n_tok * d,
                n_tok * d,
                1,
            )?;
            if prelude_blocks_len > 0 {
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    self.prelude_blocks_buf.device_ptr(),
                    self.blocks_buf.device_ptr(),
                    prelude_blocks_len * d,
                    blk_tok_stride,
                    blk_tok_stride,
                    n_tok,
                )?;
            }

            let mut bank_len = 0usize;
            for pass in 0..self.cfg.n_passes {
                if pass > 0 {
                    let bank_start = prelude_blocks_len + 1;
                    bank_len = blocks_len.saturating_sub(bank_start);
                    if bank_len > 0 {
                        self.kernels.launch_copy_2d_rows(
                            &self.ctx,
                            self.bank_buf.device_ptr(),
                            self.blocks_buf.device_ptr() + (bank_start * d * 4) as u64,
                            bank_len * d,
                            blk_tok_stride,
                            blk_tok_stride,
                            n_tok,
                        )?;
                    }
                    self.kernels.launch_copy_2d_rows(
                        &self.ctx,
                        self.bank_out_buf.device_ptr(),
                        self.cur_buf.device_ptr(),
                        n_tok * d,
                        n_tok * d,
                        n_tok * d,
                        1,
                    )?;
                    self.kernels.launch_copy_2d_rows(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        self.prelude_buf.device_ptr(),
                        n_tok * d,
                        n_tok * d,
                        n_tok * d,
                        1,
                    )?;
                    blocks_len = prelude_blocks_len;
                    if prelude_blocks_len > 0 {
                        self.kernels.launch_copy_2d_rows(
                            &self.ctx,
                            self.blocks_buf.device_ptr(),
                            self.prelude_blocks_buf.device_ptr(),
                            prelude_blocks_len * d,
                            blk_tok_stride,
                            blk_tok_stride,
                            n_tok,
                        )?;
                    }
                }

                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    self.maybe_capture_visit_tap(
                        visit,
                        self.cur_buf.device_ptr(),
                        dspark_visits,
                        n_tok,
                    )?;
                    visit += 1;
                    if pass > 0 {
                        let loop_w = &self.loop_layers[ip - cs];
                        self.launch_mix_depth(
                            self.bank_buf.device_ptr(),
                            self.bank_out_buf.device_ptr(),
                            loop_w.msg_query.device_ptr(),
                            self.messages_buf.device_ptr(),
                            bank_len,
                            LOOP_CTRL_COUNT,
                            blk_tok_stride,
                            d,
                            LOOP_CTRL_COUNT,
                            1,
                            n_tok,
                        )?;
                    }
                    self.encode_layer(
                        il,
                        ip,
                        n_tok,
                        start_pos,
                        pos_dev_ptr,
                        pass > 0,
                        save_snapshots,
                        &mut blocks_len,
                    )?;
                }
            }

            for il in ce..self.cfg.n_layers_phys {
                self.maybe_capture_visit_tap(
                    visit,
                    self.cur_buf.device_ptr(),
                    dspark_visits,
                    n_tok,
                )?;
                visit += 1;
                self.encode_layer(
                    il,
                    il,
                    n_tok,
                    start_pos,
                    pos_dev_ptr,
                    false,
                    save_snapshots,
                    &mut blocks_len,
                )?;
            }
        }

        // Final output depth mix + output RMSNorm
        self.launch_mix_depth(
            self.blocks_buf.device_ptr(),
            self.cur_buf.device_ptr(),
            self.output_res_query.device_ptr(),
            self.inp_sa_buf.device_ptr(),
            blocks_len,
            1,
            blk_tok_stride,
            d,
            1,
            1,
            n_tok,
        )?;
        self.kernels.launch_rms_norm(
            &self.ctx,
            self.inp_sa_buf.device_ptr(),
            self.output_norm.device_ptr(),
            self.cur_buf.device_ptr(),
            d,
            n_tok,
            self.cfg.rms_eps,
        )?;
        self.maybe_capture_visit_tap(
            self.cfg.n_layers,
            self.cur_buf.device_ptr(),
            dspark_visits,
            n_tok,
        )?;

        let cap = self.cfg.final_logit_softcapping.unwrap_or(0.0);
        if compute_all_argmax {
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                self.cur_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.logits_padded_buf.device_ptr(),
                n_tok,
                false,
            )?;
            let mut pl = self.logits_padded_buf.device_ptr();
            let mut pm: CUdeviceptr = 0;
            let mut pids = self.argmax_dev.device_ptr();
            let mut ppr: CUdeviceptr = 0;
            let mut nv = self.cfg.vocab_size as i32;
            let mut rs = self.cfg.vocab_padded as i32;
            let mut off = 0i32;
            let mut c_cap = cap;
            let mut args: [*mut c_void; 8] = [
                (&mut pl as *mut CUdeviceptr).cast(),
                (&mut pm as *mut CUdeviceptr).cast(),
                (&mut pids as *mut CUdeviceptr).cast(),
                (&mut ppr as *mut CUdeviceptr).cast(),
                (&mut nv as *mut i32).cast(),
                (&mut rs as *mut i32).cast(),
                (&mut off as *mut i32).cast(),
                (&mut c_cap as *mut f32).cast(),
            ];
            // SAFETY: Matches `softcap_argmax_f32`.
            unsafe {
                self.ctx.launch(
                    "softcap_argmax_f32",
                    self.ik.softcap_argmax,
                    (1, n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        } else if compute_last_logits {
            let last_x = self.cur_buf.device_ptr() + ((n_tok - 1) * d * 4) as u64;
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                last_x,
                self.xq_scratch.device_ptr(),
                false,
                self.logits_padded_buf.device_ptr(),
                1,
                false,
            )?;
            if cap > 0.0 {
                let mut px = self.logits_padded_buf.device_ptr();
                let mut c_cap = cap;
                let mut len = self.cfg.vocab_size;
                let mut args: [*mut c_void; 3] = [
                    (&mut px as *mut CUdeviceptr).cast(),
                    (&mut c_cap as *mut f32).cast(),
                    (&mut len as *mut usize).cast(),
                ];
                // SAFETY: Matches `softcap_inplace_f32`.
                unsafe {
                    self.ctx.launch(
                        "softcap_inplace_f32",
                        self.ik.softcap_inplace,
                        ((self.cfg.vocab_size as u32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn run_decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), CudaError> {
        self.pinned_ids.write_u32_at(0, &[token_id]);
        self.pinned_pos.write_u32_at(0, &[pos as u32]);
        self.token_ids_buf
            .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;
        self.pos_buf
            .copy_from_pinned_async(0, &self.pinned_pos, 0, 4)?;
        let slot = if !compute_logits && !compute_argmax {
            0
        } else if !compute_argmax {
            1
        } else {
            2
        };
        if self.decode_graphs[slot].is_none() {
            let pos_ptr = self.pos_buf.device_ptr();
            let ctx = self.ctx.clone();
            let g = CudaGraphExec::capture(&ctx, || {
                self.encode_chunk(
                    1,
                    0,
                    pos_ptr,
                    false,
                    compute_logits && !compute_argmax,
                    compute_argmax,
                    false,
                    &[],
                )
            })?;
            self.decode_graphs[slot] = Some(g);
        }
        self.decode_graphs[slot]
            .as_ref()
            .expect("captured inkling2 decode graph")
            .launch()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_hybrid(
        &mut self,
        il: usize,
        ip: usize,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        has_messages: bool,
        save_snapshots: bool,
        blocks_len: &mut usize,
        mut host_single: Option<&mut CudaHostMoeFn<'_>>,
        mut host_win: Option<&mut CudaHostMoeWindowFn<'_>>,
    ) -> Result<(), CudaError> {
        if self.layers[ip].gate_up_exps.is_some() {
            return self.encode_layer(
                il,
                ip,
                n_tok,
                start_pos,
                pos_dev_ptr,
                has_messages,
                save_snapshots,
                blocks_len,
            );
        }

        // If `k_gpu == ku` on a `DEVICEMAP` layer (`TM_CUDA_HYBRID_GPU_EXPS=8`), run 100% on GPU!
        let ku = self.cfg.n_expert_used;
        let ne = self.cfg.n_expert;
        let d = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        if n_tok == 1
            && self.hybrid_gpu_exps >= ku
            && let Some(ref ext) = self.layers[ip].host_imported_experts
        {
            let gu_p = ext.gu_reg.device_ptr();
            let dn_p = ext.dn_reg.device_ptr();
            self.encode_layer_pre_moe(
                il,
                ip,
                1,
                start_pos,
                pos_dev_ptr,
                has_messages,
                save_snapshots,
                blocks_len,
            )?;
            self.encode_moe_from_ptrs(ip, gu_p, dn_p, 1, ku, ne, false)?;
            return self.encode_layer_post_moe(il, ip, 1, save_snapshots);
        }

        self.encode_layer_pre_moe(
            il,
            ip,
            n_tok,
            start_pos,
            pos_dev_ptr,
            has_messages,
            save_snapshots,
            blocks_len,
        )?;

        // Async D2H of `[n_tok, d]` normed input + `[n_tok, ku]` router ids/weights, followed by `ev_dtoh`!
        self.wo_out_buf
            .copy_to_pinned_async(0, &mut self.pinned_moe_x, 0, n_tok * d * 4)?;
        self.moe_top_ids
            .copy_to_pinned_async(0, &mut self.pinned_moe_ids, 0, n_tok * ku * 4)?;
        self.moe_top_weights
            .copy_to_pinned_async(0, &mut self.pinned_moe_ws, 0, n_tok * ku * 4)?;
        self.ev_dtoh.record()?;

        // Enqueue GPU's zero-copy PCIe `DEVICEMAP` share immediately on `stream` so GPU executes
        // concurrently while CPU wakes up on `ev_dtoh`!
        let mut k_gpu = 0usize;
        let mut split_e = 0usize;
        if let Some(ref ext) = self.layers[ip].host_imported_experts {
            let gu_p = ext.gu_reg.device_ptr();
            let dn_p = ext.dn_reg.device_ptr();
            if n_tok == 1 && self.hybrid_gpu_exps > 0 {
                k_gpu = self.hybrid_gpu_exps.min(ku);
                self.encode_moe_from_ptrs(ip, gu_p, dn_p, 1, k_gpu, ne, false)?;
            } else if n_tok >= 16
                && ne <= 512
                && self.ctx.info().has_fp16_wmma()
                && d.is_multiple_of(32)
                && n_ff.is_multiple_of(32)
            {
                split_e = (ne / 4).max(1);
                self.encode_moe_from_ptrs(ip, gu_p, dn_p, n_tok, ku, split_e, true)?;
            }
        }

        // Wait ONLY for the tiny D2H transfer (`ev_dtoh`), NOT the PCIe GPU MoE kernels queued after it!
        self.ev_dtoh.synchronize()?;

        self.host_x_scratch[..n_tok * d].copy_from_slice(self.pinned_moe_x.as_f32_slice(n_tok * d));
        for (dst, &src) in self.host_top_ids_scratch[..n_tok * ku]
            .iter_mut()
            .zip(self.pinned_moe_ids.as_u32_slice(n_tok * ku))
        {
            *dst = src as usize;
        }
        self.host_top_ws_scratch[..n_tok * ku]
            .copy_from_slice(self.pinned_moe_ws.as_f32_slice(n_tok * ku));

        if n_tok == 1 {
            if k_gpu < ku {
                if let Some(ref mut cb) = host_single {
                    cb(
                        ip,
                        &self.host_x_scratch[..d],
                        &self.host_top_ids_scratch[k_gpu..ku],
                        &self.host_top_ws_scratch[k_gpu..ku],
                        &mut self.host_out_scratch[..d],
                    );
                } else if let Some(ref mut cb_w) = host_win {
                    cb_w(
                        ip,
                        1,
                        &self.host_x_scratch[..d],
                        &self.host_top_ids_scratch[k_gpu..ku],
                        &self.host_top_ws_scratch[k_gpu..ku],
                        &mut self.host_out_scratch[..d],
                    );
                } else {
                    return Err(CudaError::InvalidDimensions(format!(
                        "layer {ip} MoE experts are offloaded to host, but no host callback was provided"
                    )));
                }
                self.pinned_moe_out
                    .as_f32_slice_mut(d)
                    .copy_from_slice(&self.host_out_scratch[..d]);
                if k_gpu == 0 {
                    self.moe_out_buf
                        .copy_from_pinned_async(0, &self.pinned_moe_out, 0, d * 4)?;
                } else {
                    self.inp_sa_buf
                        .copy_from_pinned_async(0, &self.pinned_moe_out, 0, d * 4)?;
                    self.kernels.launch_add_scaled(
                        &self.ctx,
                        self.moe_out_buf.device_ptr(),
                        self.inp_sa_buf.device_ptr(),
                        1.0,
                        d,
                    )?;
                }
            }
        } else {
            if split_e > 0 {
                for idx in 0..n_tok * ku {
                    if self.host_top_ids_scratch[idx] < split_e {
                        self.host_top_ws_scratch[idx] = 0.0;
                    }
                }
            }
            if let Some(ref mut cb_w) = host_win {
                cb_w(
                    ip,
                    n_tok,
                    &self.host_x_scratch[..n_tok * d],
                    &self.host_top_ids_scratch[..n_tok * ku],
                    &self.host_top_ws_scratch[..n_tok * ku],
                    &mut self.host_out_scratch[..n_tok * d],
                );
            } else if let Some(ref mut cb) = host_single {
                for t in 0..n_tok {
                    cb(
                        ip,
                        &self.host_x_scratch[t * d..(t + 1) * d],
                        &self.host_top_ids_scratch[t * ku..(t + 1) * ku],
                        &self.host_top_ws_scratch[t * ku..(t + 1) * ku],
                        &mut self.host_out_scratch[t * d..(t + 1) * d],
                    );
                }
            } else {
                return Err(CudaError::InvalidDimensions(format!(
                    "layer {ip} MoE experts are offloaded to host, but no host window callback was provided"
                )));
            }
            self.pinned_moe_out
                .as_f32_slice_mut(n_tok * d)
                .copy_from_slice(&self.host_out_scratch[..n_tok * d]);
            if split_e == 0 {
                self.moe_out_buf.copy_from_pinned_async(
                    0,
                    &self.pinned_moe_out,
                    0,
                    n_tok * d * 4,
                )?;
            } else {
                self.inp_sa_buf.copy_from_pinned_async(
                    0,
                    &self.pinned_moe_out,
                    0,
                    n_tok * d * 4,
                )?;
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    self.moe_out_buf.device_ptr(),
                    self.inp_sa_buf.device_ptr(),
                    1.0,
                    n_tok * d,
                )?;
            }
        }

        self.encode_layer_post_moe(il, ip, n_tok, save_snapshots)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_chunk_hybrid(
        &mut self,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        preloaded_embed: bool,
        compute_last_logits: bool,
        compute_all_argmax: bool,
        save_snapshots: bool,
        dspark_visits: &[usize],
        mut host_single: Option<&mut CudaHostMoeFn<'_>>,
        mut host_win: Option<&mut CudaHostMoeWindowFn<'_>>,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let max_blocks = self.cfg.max_depth_blocks();
        let blk_tok_stride = max_blocks * d;

        if !preloaded_embed {
            self.kernels.launch_embed(
                &self.ctx,
                &self.token_embd,
                self.token_ids_buf.device_ptr(),
                self.inp_sa_buf.device_ptr(),
                n_tok,
                1.0,
            )?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.inp_sa_buf.device_ptr(),
                self.token_embd_norm.device_ptr(),
                self.cur_buf.device_ptr(),
                d,
                n_tok,
                self.cfg.rms_eps,
            )?;
        }

        let mut blocks_len = 0usize;
        let mut visit = 0usize;
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                self.maybe_capture_visit_tap(
                    visit,
                    self.cur_buf.device_ptr(),
                    dspark_visits,
                    n_tok,
                )?;
                visit += 1;
                self.run_layer_hybrid(
                    il,
                    il,
                    n_tok,
                    start_pos,
                    pos_dev_ptr,
                    false,
                    save_snapshots,
                    &mut blocks_len,
                    host_single.as_deref_mut(),
                    host_win.as_deref_mut(),
                )?;
            }
        } else {
            let cs = self.cfg.core_start;
            let ce = self.cfg.core_end;
            for il in 0..cs {
                self.maybe_capture_visit_tap(
                    visit,
                    self.cur_buf.device_ptr(),
                    dspark_visits,
                    n_tok,
                )?;
                visit += 1;
                self.run_layer_hybrid(
                    il,
                    il,
                    n_tok,
                    start_pos,
                    pos_dev_ptr,
                    false,
                    save_snapshots,
                    &mut blocks_len,
                    host_single.as_deref_mut(),
                    host_win.as_deref_mut(),
                )?;
            }

            let prelude_blocks_len = blocks_len;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                self.prelude_buf.device_ptr(),
                self.cur_buf.device_ptr(),
                n_tok * d,
                n_tok * d,
                n_tok * d,
                1,
            )?;
            if prelude_blocks_len > 0 {
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    self.prelude_blocks_buf.device_ptr(),
                    self.blocks_buf.device_ptr(),
                    prelude_blocks_len * d,
                    blk_tok_stride,
                    blk_tok_stride,
                    n_tok,
                )?;
            }

            let can_graph_core = n_tok == 1
                && pos_dev_ptr != 0
                && !save_snapshots
                && dspark_visits.is_empty()
                && self.core_vram_prefix > 0
                && self.cfg.n_passes <= 3;

            let mut bank_len = 0usize;
            for pass in 0..self.cfg.n_passes {
                if pass > 0 {
                    let bank_start = prelude_blocks_len + 1;
                    bank_len = blocks_len.saturating_sub(bank_start);
                    if bank_len > 0 {
                        self.kernels.launch_copy_2d_rows(
                            &self.ctx,
                            self.bank_buf.device_ptr(),
                            self.blocks_buf.device_ptr() + (bank_start * d * 4) as u64,
                            bank_len * d,
                            blk_tok_stride,
                            blk_tok_stride,
                            n_tok,
                        )?;
                    }
                    self.kernels.launch_copy_2d_rows(
                        &self.ctx,
                        self.bank_out_buf.device_ptr(),
                        self.cur_buf.device_ptr(),
                        n_tok * d,
                        n_tok * d,
                        n_tok * d,
                        1,
                    )?;
                    self.kernels.launch_copy_2d_rows(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        self.prelude_buf.device_ptr(),
                        n_tok * d,
                        n_tok * d,
                        n_tok * d,
                        1,
                    )?;
                    blocks_len = prelude_blocks_len;
                    if prelude_blocks_len > 0 {
                        self.kernels.launch_copy_2d_rows(
                            &self.ctx,
                            self.blocks_buf.device_ptr(),
                            self.prelude_blocks_buf.device_ptr(),
                            prelude_blocks_len * d,
                            blk_tok_stride,
                            blk_tok_stride,
                            n_tok,
                        )?;
                    }
                }

                let graph_end = if can_graph_core {
                    cs + self.core_vram_prefix
                } else {
                    cs
                };
                if graph_end > cs {
                    if self.core_prefix_graphs[pass].is_none() {
                        let ctx = self.ctx.clone();
                        let mut cap_blks = blocks_len;
                        let g = CudaGraphExec::capture(&ctx, || {
                            for ip in cs..graph_end {
                                let il = if pass == 0 {
                                    ip
                                } else {
                                    self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                                };
                                if pass > 0 {
                                    let loop_w = &self.loop_layers[ip - cs];
                                    self.launch_mix_depth(
                                        self.bank_buf.device_ptr(),
                                        self.bank_out_buf.device_ptr(),
                                        loop_w.msg_query.device_ptr(),
                                        self.messages_buf.device_ptr(),
                                        bank_len,
                                        LOOP_CTRL_COUNT,
                                        blk_tok_stride,
                                        d,
                                        LOOP_CTRL_COUNT,
                                        1,
                                        1,
                                    )?;
                                }
                                self.encode_layer(
                                    il,
                                    ip,
                                    1,
                                    0,
                                    pos_dev_ptr,
                                    pass > 0,
                                    false,
                                    &mut cap_blks,
                                )?;
                            }
                            Ok(())
                        })?;
                        self.core_prefix_graphs[pass] = Some(g);
                    }
                    self.core_prefix_graphs[pass]
                        .as_ref()
                        .expect("captured core prefix CUDA graph")
                        .launch()?;
                    for ip in cs..graph_end {
                        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
                            blocks_len += 1;
                        }
                        visit += 1;
                    }
                }

                for ip in graph_end..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    self.maybe_capture_visit_tap(
                        visit,
                        self.cur_buf.device_ptr(),
                        dspark_visits,
                        n_tok,
                    )?;
                    visit += 1;
                    if pass > 0 {
                        let loop_w = &self.loop_layers[ip - cs];
                        self.launch_mix_depth(
                            self.bank_buf.device_ptr(),
                            self.bank_out_buf.device_ptr(),
                            loop_w.msg_query.device_ptr(),
                            self.messages_buf.device_ptr(),
                            bank_len,
                            LOOP_CTRL_COUNT,
                            blk_tok_stride,
                            d,
                            LOOP_CTRL_COUNT,
                            1,
                            n_tok,
                        )?;
                    }
                    self.run_layer_hybrid(
                        il,
                        ip,
                        n_tok,
                        start_pos,
                        pos_dev_ptr,
                        pass > 0,
                        save_snapshots,
                        &mut blocks_len,
                        host_single.as_deref_mut(),
                        host_win.as_deref_mut(),
                    )?;
                }
            }

            for il in ce..self.cfg.n_layers_phys {
                self.maybe_capture_visit_tap(
                    visit,
                    self.cur_buf.device_ptr(),
                    dspark_visits,
                    n_tok,
                )?;
                visit += 1;
                self.run_layer_hybrid(
                    il,
                    il,
                    n_tok,
                    start_pos,
                    pos_dev_ptr,
                    false,
                    save_snapshots,
                    &mut blocks_len,
                    host_single.as_deref_mut(),
                    host_win.as_deref_mut(),
                )?;
            }
        }

        self.launch_mix_depth(
            self.blocks_buf.device_ptr(),
            self.cur_buf.device_ptr(),
            self.output_res_query.device_ptr(),
            self.inp_sa_buf.device_ptr(),
            blocks_len,
            1,
            blk_tok_stride,
            d,
            1,
            1,
            n_tok,
        )?;
        self.kernels.launch_rms_norm(
            &self.ctx,
            self.inp_sa_buf.device_ptr(),
            self.output_norm.device_ptr(),
            self.cur_buf.device_ptr(),
            d,
            n_tok,
            self.cfg.rms_eps,
        )?;
        self.maybe_capture_visit_tap(
            self.cfg.n_layers,
            self.cur_buf.device_ptr(),
            dspark_visits,
            n_tok,
        )?;

        let cap = self.cfg.final_logit_softcapping.unwrap_or(0.0);
        if compute_all_argmax {
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                self.cur_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.logits_padded_buf.device_ptr(),
                n_tok,
                false,
            )?;
            let mut pl = self.logits_padded_buf.device_ptr();
            let mut pm: CUdeviceptr = 0;
            let mut pids = self.argmax_dev.device_ptr();
            let mut ppr: CUdeviceptr = 0;
            let mut nv = self.cfg.vocab_size as i32;
            let mut rs = self.cfg.vocab_padded as i32;
            let mut off = 0i32;
            let mut c_cap = cap;
            let mut args: [*mut c_void; 8] = [
                (&mut pl as *mut CUdeviceptr).cast(),
                (&mut pm as *mut CUdeviceptr).cast(),
                (&mut pids as *mut CUdeviceptr).cast(),
                (&mut ppr as *mut CUdeviceptr).cast(),
                (&mut nv as *mut i32).cast(),
                (&mut rs as *mut i32).cast(),
                (&mut off as *mut i32).cast(),
                (&mut c_cap as *mut f32).cast(),
            ];
            // SAFETY: Matches `softcap_argmax_f32`.
            unsafe {
                self.ctx.launch(
                    "softcap_argmax_f32",
                    self.ik.softcap_argmax,
                    (1, n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        } else if compute_last_logits {
            let last_x = self.cur_buf.device_ptr() + ((n_tok - 1) * d * 4) as u64;
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                last_x,
                self.xq_scratch.device_ptr(),
                false,
                self.logits_padded_buf.device_ptr(),
                1,
                false,
            )?;
            if cap > 0.0 {
                let mut px = self.logits_padded_buf.device_ptr();
                let mut c_cap = cap;
                let mut len = self.cfg.vocab_size;
                let mut args: [*mut c_void; 3] = [
                    (&mut px as *mut CUdeviceptr).cast(),
                    (&mut c_cap as *mut f32).cast(),
                    (&mut len as *mut usize).cast(),
                ];
                // SAFETY: Matches `softcap_inplace_f32`.
                unsafe {
                    self.ctx.launch(
                        "softcap_inplace_f32",
                        self.ik.softcap_inplace,
                        ((self.cfg.vocab_size as u32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        self.forward_token_hybrid(token_id, pos, compute_logits, None)
    }

    pub fn forward_token_hybrid(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        host_moe: Option<&mut CudaHostMoeFn<'_>>,
    ) -> Result<&[f32], CudaError> {
        if pos >= self.max_ctx || (token_id as usize) >= self.cfg.vocab_size {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid token_id={token_id} or pos={pos}"
            )));
        }
        if self.host_offloaded_moe_layers == 0 {
            self.run_decode_step(token_id, pos, compute_logits, false)?;
        } else {
            self.pinned_ids.write_u32_at(0, &[token_id]);
            self.pinned_pos.write_u32_at(0, &[pos as u32]);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;
            self.pos_buf
                .copy_from_pinned_async(0, &self.pinned_pos, 0, 4)?;
            let pos_ptr = self.pos_buf.device_ptr();
            self.encode_chunk_hybrid(
                1,
                0,
                pos_ptr,
                false,
                compute_logits,
                false,
                false,
                &[],
                host_moe,
                None,
            )?;
        }
        if compute_logits {
            self.logits_padded_buf.copy_to_f32(&mut self.logits_host)?;
        } else {
            self.ctx.synchronize()?;
        }
        Ok(&self.logits_host)
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, CudaError> {
        self.forward_token_greedy_hybrid(token_id, pos, None)
    }

    pub fn forward_token_greedy_hybrid(
        &mut self,
        token_id: u32,
        pos: usize,
        host_moe: Option<&mut CudaHostMoeFn<'_>>,
    ) -> Result<u32, CudaError> {
        if pos >= self.max_ctx || (token_id as usize) >= self.cfg.vocab_size {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid token_id={token_id} or pos={pos}"
            )));
        }
        if self.host_offloaded_moe_layers == 0 {
            self.run_decode_step(token_id, pos, false, true)?;
        } else {
            self.pinned_ids.write_u32_at(0, &[token_id]);
            self.pinned_pos.write_u32_at(0, &[pos as u32]);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;
            self.pos_buf
                .copy_from_pinned_async(0, &self.pinned_pos, 0, 4)?;
            let pos_ptr = self.pos_buf.device_ptr();
            self.encode_chunk_hybrid(
                1,
                0,
                pos_ptr,
                false,
                false,
                true,
                false,
                &[],
                host_moe,
                None,
            )?;
        }
        self.argmax_dev.read_u32()
    }

    /// Evaluates a multi-token verification or prefill window (`1..=batch_cap`), optionally
    /// saving `ShortConv` rollback snapshots, capturing `DSpark` visit taps, and returning greedy
    /// argmax token IDs for every window position!
    #[allow(clippy::too_many_arguments)]
    pub fn forward_window_with_taps(
        &mut self,
        window_tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        save_conv_snapshots: bool,
        dspark_visits: &[usize],
        out_argmax: Option<&mut [u32]>,
    ) -> Result<&[f32], CudaError> {
        self.forward_window_hybrid(
            window_tokens,
            start_pos,
            compute_last_logits,
            save_conv_snapshots,
            dspark_visits,
            None,
            out_argmax,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_window_hybrid(
        &mut self,
        window_tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        save_conv_snapshots: bool,
        dspark_visits: &[usize],
        host_win: Option<&mut CudaHostMoeWindowFn<'_>>,
        mut out_argmax: Option<&mut [u32]>,
    ) -> Result<&[f32], CudaError> {
        if window_tokens.is_empty() {
            return Ok(&self.logits_host);
        }
        let w = window_tokens.len();
        if w > self.batch_cap || start_pos + w > self.max_ctx {
            return Err(CudaError::InvalidDimensions(format!(
                "window len {w} at pos {start_pos} exceeds limits (batch_cap={})",
                self.batch_cap
            )));
        }
        for &tid in window_tokens {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(CudaError::InvalidDimensions(format!(
                    "token_id {tid} >= vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let want_all_argmax = out_argmax.is_some();
        self.pinned_ids.write_u32_at(0, window_tokens);
        self.token_ids_buf
            .copy_from_pinned_async(0, &self.pinned_ids, 0, w * 4)?;
        if self.host_offloaded_moe_layers == 0 {
            self.encode_chunk(
                w,
                start_pos,
                0,
                false,
                compute_last_logits && !want_all_argmax,
                want_all_argmax,
                save_conv_snapshots,
                dspark_visits,
            )?;
        } else {
            self.encode_chunk_hybrid(
                w,
                start_pos,
                0,
                false,
                compute_last_logits && !want_all_argmax,
                want_all_argmax,
                save_conv_snapshots,
                dspark_visits,
                None,
                host_win,
            )?;
        }
        if let Some(ref mut dst) = out_argmax {
            // SAFETY: `[u32]` is POD; `w <= MAX_INKLING2_CUDA_SPEC_WINDOW`.
            let bytes =
                unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), w * 4) };
            self.argmax_dev.copy_to_bytes(bytes)?;
        } else if compute_last_logits {
            self.logits_padded_buf.copy_to_f32(&mut self.logits_host)?;
        } else {
            self.ctx.synchronize()?;
        }
        Ok(&self.logits_host)
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], CudaError> {
        if token_ids.is_empty() {
            return Ok(&self.logits_host);
        }
        let mut off = 0usize;
        while off < token_ids.len() {
            let b = (token_ids.len() - off).min(self.batch_cap);
            let is_last = off + b == token_ids.len();
            let _ = self.forward_window_with_taps(
                &token_ids[off..off + b],
                start_pos + off,
                is_last && compute_last_logits,
                false,
                &[],
                None,
            )?;
            off += b;
        }
        Ok(&self.logits_host)
    }

    pub fn forward_embed(
        &mut self,
        embed: &[f32],
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        if pos >= self.max_ctx || embed.len() != self.cfg.n_embd {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid forward_embed len={} or pos={pos}",
                embed.len()
            )));
        }
        self.inp_sa_buf.copy_from_f32(embed)?;
        self.kernels.launch_rms_norm(
            &self.ctx,
            self.inp_sa_buf.device_ptr(),
            self.token_embd_norm.device_ptr(),
            self.cur_buf.device_ptr(),
            self.cfg.n_embd,
            1,
            self.cfg.rms_eps,
        )?;
        self.encode_chunk(1, pos, 0, true, compute_logits, false, false, &[])?;
        if compute_logits {
            self.logits_padded_buf.copy_to_f32(&mut self.logits_host)?;
        } else {
            self.ctx.synchronize()?;
        }
        Ok(&self.logits_host)
    }

    pub fn profile(
        &mut self,
        n_runs: usize,
        ctx_pos: usize,
    ) -> Result<CudaProfileReport, CudaError> {
        let runs = n_runs.max(1);
        let pos = ctx_pos.min(self.max_ctx.saturating_sub(1));
        let _ = self.forward_token_greedy(1, pos)?;

        let ev0 = CudaEvent::new(&self.ctx)?;
        let ev1 = CudaEvent::new(&self.ctx)?;
        self.token_ids_buf.copy_from_u32(&[1])?;
        self.pos_buf.copy_from_u32(&[pos as u32])?;
        ev0.record()?;
        if let Some(ref g) = self.decode_graphs[2] {
            for _ in 0..runs {
                g.launch()?;
            }
        } else {
            for _ in 0..runs {
                self.encode_chunk(1, pos, 0, false, false, true, false, &[])?;
            }
        }
        ev1.record()?;
        ev1.synchronize()?;
        let total_ms = ev1.elapsed_ms(&ev0)? as f64 / runs as f64;
        let tps = 1000.0 / total_ms.max(1e-9);
        let eff_bw = (self.active_decode_bytes as f64 / 1e9) / (total_ms / 1e3).max(1e-12);
        let peak_bw = self.ctx.info().peak_mem_bw_gbps.max(1.0);

        Ok(CudaProfileReport {
            device_name: self.ctx.info().name.clone(),
            arch: self.cfg.arch.clone(),
            total_ms_per_tok: total_ms,
            tok_per_sec: tps,
            model_weight_bytes: self.active_decode_bytes,
            effective_bw_gbps: eff_bw,
            peak_bw_gbps: peak_bw,
            bw_utilization_pct: 100.0 * eff_bw / peak_bw,
            stages: vec![CudaStageStat {
                name: "full_decode_step_gpu",
                ms_per_tok: total_ms,
                pct: 100.0,
            }],
        })
    }
}

// ============================================================================
// 100%-GPU `DSpark` / `DFlash` Speculative Drafter for `Inkling2CudaRunner`
// ============================================================================

struct DFlashCudaKernels {
    mix_rmsnorm: CUfunction,
    k_ctx_norm_rope_kv: CUfunction,
    init_block: CUfunction,
    qk_norm_rope_f16round: CUfunction,
    noncausal_attn: CUfunction,
    markov_prep: CUfunction,
    markov_gate_latent: CUfunction,
    markov_corr: CUfunction,
    indexed_q8_0_q8_1_gemv: CUfunction,
}

impl DFlashCudaKernels {
    fn load(ctx: &CudaContext) -> Result<Self, CudaError> {
        Ok(Self {
            mix_rmsnorm: ctx.get_function("dflash_mix_rmsnorm_f32")?,
            k_ctx_norm_rope_kv: ctx.get_function("dflash_k_ctx_norm_rope_kv_f16")?,
            init_block: ctx.get_function("dflash_init_block_f32")?,
            qk_norm_rope_f16round: ctx.get_function("dflash_qk_norm_rope_f16round_f32")?,
            noncausal_attn: ctx.get_function("dflash_noncausal_attn_f32")?,
            markov_prep: ctx.get_function("dflash_markov_prep_f32")?,
            markov_gate_latent: ctx.get_function("dflash_markov_gate_latent_f32")?,
            markov_corr: ctx.get_function("dflash_markov_corr_f32")?,
            indexed_q8_0_q8_1_gemv: ctx.get_function("dflash_indexed_q8_0_q8_1_gemv")?,
        })
    }
}

struct CudaDFlashLayer {
    dflare_norm: CudaBuffer,
    attn_k_ctx: CudaMat,
    attn_v_ctx: CudaMat,
    attn_k_ctx_norm: CudaBuffer,
    attn_norm: CudaBuffer,
    wq: CudaMat,
    wk: CudaMat,
    wv: CudaMat,
    wo: CudaMat,
    attn_q_norm: CudaBuffer,
    attn_k_norm: CudaBuffer,
    ffn_norm: CudaBuffer,
    ffn_gate: CudaMat,
    ffn_up: CudaMat,
    ffn_down: CudaMat,
    k_cache: CudaBuffer,
    v_cache: CudaBuffer,
}

/// 100%-CUDA `DSpark` (`dflash`) block speculative drafter sharing `Inkling2CudaRunner`'s
/// `token_embd`, `token_embd_norm`, `output`, and `dspark_taps_buf`.
pub struct DFlashCudaRunner {
    pub cfg: DFlashConfig,
    pub max_ctx: usize,
    dfk: DFlashCudaKernels,
    fuse_probs: Vec<CudaBuffer>,
    dspark_norm: CudaBuffer,
    mask_embd: CudaBuffer,
    output_norm: CudaBuffer,
    joint_proj: CudaMat,
    joint_proj_b: CudaBuffer,
    hidden_out: CudaMat,
    hidden_out_b: CudaBuffer,
    markov_w1_bf16: CudaBuffer,
    rope_inv_freqs: CudaBuffer,
    layers: Vec<CudaDFlashLayer>,
    vocab_map: Option<(CudaBuffer, usize)>,

    ring_cap: usize,
    ctx_feat_ring: CudaBuffer,
    ctx_feat_pos: Vec<isize>,

    inj_norm_buf: CudaBuffer,
    inj_k_buf: CudaBuffer,
    inj_v_buf: CudaBuffer,
    inp_l_buf: CudaBuffer,
    norm_blk_buf: CudaBuffer,
    q_blk_buf: CudaBuffer,
    k_blk_buf: CudaBuffer,
    v_blk_buf: CudaBuffer,
    attn_blk_buf: CudaBuffer,
    proj_blk_buf: CudaBuffer,
    ffn_gate_blk_buf: CudaBuffer,
    ffn_up_blk_buf: CudaBuffer,
    hidden_blk_buf: CudaBuffer,
    rnn_state_buf: CudaBuffer,
    joint_in_buf: CudaBuffer,
    joint_out_buf: CudaBuffer,
    latent_buf: CudaBuffer,
    correction_buf: CudaBuffer,
    corr_hidden_buf: CudaBuffer,
    xq_scratch: CudaBuffer,
    draft_toks_buf: CudaBuffer,
    draft_probs_buf: CudaBuffer,
}

impl DFlashCudaRunner {
    pub fn from_gguf(
        target: &Inkling2CudaRunner,
        gguf: &GgufFile,
        max_ctx: usize,
    ) -> Result<Self, CudaError> {
        let cfg = DFlashConfig::from_gguf(gguf)?;
        if cfg.n_embd != target.cfg.n_embd || cfg.vocab_size != target.cfg.vocab_size {
            return Err(CudaError::InvalidDimensions(format!(
                "dflash (n_embd={}, vocab={}) mismatches target (n_embd={}, vocab={})",
                cfg.n_embd, cfg.vocab_size, target.cfg.n_embd, target.cfg.vocab_size
            )));
        }
        let ctx = &target.ctx;
        let dfk = DFlashCudaKernels::load(ctx)?;
        let d = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let rank = cfg.markov_rank;
        let n_taps = cfg.target_layers.len();
        let bs = cfg.block_size;

        let (fuse_ti, fuse_bytes) =
            gguf.req_tensor("dspark_fuse.weight", "global", &[n_taps, cfg.n_layers + 1])?;
        let mut fuse_raw = vec![0.0f32; n_taps * (cfg.n_layers + 1)];
        dequantize_row_bytes(fuse_ti.dtype, fuse_bytes, &mut fuse_raw);
        let mut fuse_probs = Vec::with_capacity(cfg.n_layers + 1);
        for r in 0..=cfg.n_layers {
            let row = &mut fuse_raw[r * n_taps..(r + 1) * n_taps];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - mx).exp();
                s += *v;
            }
            for v in row.iter_mut() {
                *v /= s;
            }
            fuse_probs.push(ctx.alloc_from_f32(row)?);
        }

        let dspark_norm = load_f32_dev(ctx, gguf, None, "dspark_norm.weight", "global", &[d])?;
        let mask_embd = load_f32_dev(ctx, gguf, None, "mask_embd.weight", "global", &[d])?;
        let output_norm = load_f32_dev(ctx, gguf, None, "output_norm.weight", "global", &[d])?;
        let (joint_proj, _) = load_mat_dev(
            ctx,
            gguf,
            None,
            "joint_proj.weight",
            "global",
            2 * rank + 2 * d,
            3 * rank,
        )?;
        let joint_proj_b = load_f32_dev(ctx, gguf, None, "joint_proj.bias", "global", &[3 * rank])?;
        let (hidden_out, _) = load_mat_dev(
            ctx,
            gguf,
            None,
            "dspark_hidden_out.weight",
            "global",
            rank,
            d,
        )?;
        let hidden_out_b = load_f32_dev(ctx, gguf, None, "dspark_hidden_out.bias", "global", &[d])?;

        let half_rope = cfg.rope_dim / 2;
        let mut inv_f32 = Vec::with_capacity(half_rope);
        let base = cfg.rope_freq_base as f64;
        for i in 0..half_rope {
            inv_f32.push(base.powf(-((2 * i) as f64) / (cfg.rope_dim as f64)) as f32);
        }
        let rope_inv_freqs = ctx.alloc_from_f32(&inv_f32)?;

        let (mw1_ti, mw1_bytes) =
            gguf.req_tensor("markov_w1.weight", "global", &[rank, cfg.vocab_padded])?;
        let mut w1_norm2 = vec![0.0f32; cfg.vocab_size];
        let markov_w1_bf16 = if mw1_ti.dtype == DType::BF16 {
            for (v, n2) in w1_norm2.iter_mut().enumerate() {
                let row = &mw1_bytes[v * rank * 2..(v + 1) * rank * 2];
                let mut s = 0.0f32;
                for c in row.as_chunks::<2>().0 {
                    let bits = u16::from_le_bytes(*c);
                    let f = f32::from_bits((bits as u32) << 16);
                    s += f * f;
                }
                *n2 = s;
            }
            ctx.alloc_from_bytes(mw1_bytes)?
        } else {
            let mut bf16_bytes = vec![0u8; cfg.vocab_padded * rank * 2];
            let mut row_f = vec![0.0f32; rank];
            let rb = mw1_ti.dtype.row_bytes(rank);
            for r in 0..cfg.vocab_padded {
                dequantize_row_bytes(mw1_ti.dtype, &mw1_bytes[r * rb..(r + 1) * rb], &mut row_f);
                let mut s = 0.0f32;
                for (j, &f) in row_f.iter().enumerate() {
                    let bf = ((f.to_bits().wrapping_add(0x7FFF)) >> 16) as u16;
                    bf16_bytes[(r * rank + j) * 2..(r * rank + j + 1) * 2]
                        .copy_from_slice(&bf.to_le_bytes());
                    s += f * f;
                }
                if r < cfg.vocab_size {
                    w1_norm2[r] = s;
                }
            }
            ctx.alloc_from_bytes(&bf16_bytes)?
        };

        // Importance-ranked draft vocabulary pruning (`65,536` rows on `Q8_0` `output.weight` -> 3.1x faster draft!)
        let vocab_lim: usize = std::env::var("TM_DSPARK_VOCAB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(65_536);
        let vocab_map = if target.output.dtype == DType::Q8_0
            && d.is_multiple_of(32)
            && vocab_lim > 0
            && vocab_lim < cfg.vocab_size
        {
            let mut idx: Vec<u32> = (0..cfg.vocab_size as u32).collect();
            idx.sort_unstable_by(|&a, &b| {
                w1_norm2[b as usize]
                    .total_cmp(&w1_norm2[a as usize])
                    .then_with(|| a.cmp(&b))
            });
            idx.truncate(vocab_lim);
            idx.sort_unstable();
            // SAFETY: `[u32]` is POD.
            let map_bytes =
                unsafe { core::slice::from_raw_parts(idx.as_ptr().cast::<u8>(), idx.len() * 4) };
            Some((ctx.alloc_from_bytes(map_bytes)?, idx.len()))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let ltag = format!("blk.{il}");
            let dflare_norm = load_f32_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.dflare_norm.weight"),
                &ltag,
                &[d],
            )?;
            let (attn_k_ctx, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_k_ctx.weight"),
                &ltag,
                d,
                kv_dim,
            )?;
            let (attn_v_ctx, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_v_ctx.weight"),
                &ltag,
                d,
                kv_dim,
            )?;
            let attn_k_ctx_norm = load_f32_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_k_ctx_norm.weight"),
                &ltag,
                &[cfg.head_dim],
            )?;
            let attn_norm = load_f32_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_norm.weight"),
                &ltag,
                &[d],
            )?;
            let (wq, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_q.weight"),
                &ltag,
                d,
                q_dim,
            )?;
            let (wk, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_k.weight"),
                &ltag,
                d,
                kv_dim,
            )?;
            let (wv, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_v.weight"),
                &ltag,
                d,
                kv_dim,
            )?;
            let (wo, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_output.weight"),
                &ltag,
                q_dim,
                d,
            )?;
            let attn_q_norm = load_f32_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_q_norm.weight"),
                &ltag,
                &[cfg.head_dim],
            )?;
            let attn_k_norm = load_f32_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.attn_k_norm.weight"),
                &ltag,
                &[cfg.head_dim],
            )?;
            let ffn_norm = load_f32_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.ffn_norm.weight"),
                &ltag,
                &[d],
            )?;
            let (ffn_gate, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.ffn_gate.weight"),
                &ltag,
                d,
                n_ff,
            )?;
            let (ffn_up, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.ffn_up.weight"),
                &ltag,
                d,
                n_ff,
            )?;
            let (ffn_down, _) = load_mat_dev(
                ctx,
                gguf,
                None,
                &format!("blk.{il}.ffn_down.weight"),
                &ltag,
                n_ff,
                d,
            )?;
            let kv_b = max_ctx * kv_dim * 2;
            layers.push(CudaDFlashLayer {
                dflare_norm,
                attn_k_ctx,
                attn_v_ctx,
                attn_k_ctx_norm,
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                attn_q_norm,
                attn_k_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                k_cache: ctx.alloc_zeroed(kv_b)?,
                v_cache: ctx.alloc_zeroed(kv_b)?,
            });
        }

        let ring_cap = 64usize;
        let ctx_feat_ring = ctx.alloc_zeroed(ring_cap * d * 4)?;
        let max_cols = (2 * rank + 2 * d).max(n_ff).max(q_dim).next_multiple_of(32);

        Ok(Self {
            cfg,
            max_ctx,
            dfk,
            fuse_probs,
            dspark_norm,
            mask_embd,
            output_norm,
            joint_proj,
            joint_proj_b,
            hidden_out,
            hidden_out_b,
            markov_w1_bf16,
            rope_inv_freqs,
            layers,
            vocab_map,
            ring_cap,
            ctx_feat_ring,
            ctx_feat_pos: vec![-1; ring_cap],
            inj_norm_buf: ctx.alloc_zeroed(target.batch_cap.max(bs) * d * 4)?,
            inj_k_buf: ctx.alloc_zeroed(target.batch_cap.max(bs) * kv_dim * 4)?,
            inj_v_buf: ctx.alloc_zeroed(target.batch_cap.max(bs) * kv_dim * 4)?,
            inp_l_buf: ctx.alloc_zeroed(bs * d * 4)?,
            norm_blk_buf: ctx.alloc_zeroed(bs * d * 4)?,
            q_blk_buf: ctx.alloc_zeroed(bs * q_dim * 4)?,
            k_blk_buf: ctx.alloc_zeroed(bs * kv_dim * 4)?,
            v_blk_buf: ctx.alloc_zeroed(bs * kv_dim * 4)?,
            attn_blk_buf: ctx.alloc_zeroed(bs * q_dim * 4)?,
            proj_blk_buf: ctx.alloc_zeroed(bs * d * 4)?,
            ffn_gate_blk_buf: ctx.alloc_zeroed(bs * n_ff * 4)?,
            ffn_up_blk_buf: ctx.alloc_zeroed(bs * n_ff * 4)?,
            hidden_blk_buf: ctx.alloc_zeroed(bs * d * 4)?,
            rnn_state_buf: ctx.alloc_zeroed(rank * 4)?,
            joint_in_buf: ctx.alloc_zeroed((2 * rank + 2 * d) * 4)?,
            joint_out_buf: ctx.alloc_zeroed(3 * rank * 4)?,
            latent_buf: ctx.alloc_zeroed(rank * 4)?,
            correction_buf: ctx.alloc_zeroed(d * 4)?,
            corr_hidden_buf: ctx.alloc_zeroed(d * 4)?,
            xq_scratch: ctx
                .alloc_zeroed(target.batch_cap.max(bs) * (max_cols / 32) * BLOCK_Q8_1_BYTES)?,
            draft_toks_buf: ctx.alloc_zeroed((bs + 4) * 4)?,
            draft_probs_buf: ctx.alloc_zeroed((bs + 4) * 4)?,
        })
    }

    pub fn reset_state(&mut self, target: &Inkling2CudaRunner) -> Result<(), CudaError> {
        for l in &self.layers {
            l.k_cache.zero()?;
            l.v_cache.zero()?;
        }
        self.ctx_feat_ring.zero()?;
        self.ctx_feat_pos.fill(-1);
        target.ctx.synchronize()
    }

    fn encode_inject_batch(
        &mut self,
        target: &Inkling2CudaRunner,
        start_pos: usize,
        count: usize,
    ) -> Result<(), CudaError> {
        if count == 0 {
            return Ok(());
        }
        let d = self.cfg.n_embd;
        let n_taps = self.cfg.target_layers.len();
        let nh_kv = self.cfg.n_head_kv;
        let hd = self.cfg.head_dim;

        for il in 0..self.cfg.n_layers {
            let l = &self.layers[il];
            {
                let mut pt = target.dspark_taps_buf.device_ptr();
                let mut pf = self.fuse_probs[il].device_ptr();
                let mut pn = l.dflare_norm.device_ptr();
                let mut po = self.inj_norm_buf.device_ptr();
                let mut cd = d as i32;
                let mut ct = n_taps as i32;
                let mut eps = self.cfg.rms_eps;
                let mut sp = start_pos as i32;
                let mut ts = 0i32;
                let mut rc = 0i32;
                let mut args: [*mut c_void; 10] = [
                    (&mut pt as *mut CUdeviceptr).cast(),
                    (&mut pf as *mut CUdeviceptr).cast(),
                    (&mut pn as *mut CUdeviceptr).cast(),
                    (&mut po as *mut CUdeviceptr).cast(),
                    (&mut cd as *mut i32).cast(),
                    (&mut ct as *mut i32).cast(),
                    (&mut eps as *mut f32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut ts as *mut i32).cast(),
                    (&mut rc as *mut i32).cast(),
                ];
                // SAFETY: Matches `dflash_mix_rmsnorm_f32`.
                unsafe {
                    target.ctx.launch(
                        "dflash_mix_rmsnorm_f32",
                        self.dfk.mix_rmsnorm,
                        (1, count as u32, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            target.kernels.launch_linear(
                &target.ctx,
                &l.attn_k_ctx,
                self.inj_norm_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.inj_k_buf.device_ptr(),
                count,
                false,
            )?;
            target.kernels.launch_linear(
                &target.ctx,
                &l.attn_v_ctx,
                self.inj_norm_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.inj_v_buf.device_ptr(),
                count,
                false,
            )?;
            {
                let mut pk = self.inj_k_buf.device_ptr();
                let mut pv = self.inj_v_buf.device_ptr();
                let mut pn = l.attn_k_ctx_norm.device_ptr();
                let mut pf = self.rope_inv_freqs.device_ptr();
                let mut pkc = l.k_cache.device_ptr();
                let mut pvc = l.v_cache.device_ptr();
                let mut nhkv = nh_kv as i32;
                let mut chd = hd as i32;
                let mut crd = self.cfg.rope_dim as i32;
                let mut sp = start_pos as i32;
                let mut eps = self.cfg.rms_eps;
                let mut args: [*mut c_void; 11] = [
                    (&mut pk as *mut CUdeviceptr).cast(),
                    (&mut pv as *mut CUdeviceptr).cast(),
                    (&mut pn as *mut CUdeviceptr).cast(),
                    (&mut pf as *mut CUdeviceptr).cast(),
                    (&mut pkc as *mut CUdeviceptr).cast(),
                    (&mut pvc as *mut CUdeviceptr).cast(),
                    (&mut nhkv as *mut i32).cast(),
                    (&mut chd as *mut i32).cast(),
                    (&mut crd as *mut i32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut eps as *mut f32).cast(),
                ];
                // SAFETY: Matches `dflash_k_ctx_norm_rope_kv_f16`.
                unsafe {
                    target.ctx.launch(
                        "dflash_k_ctx_norm_rope_kv_f16",
                        self.dfk.k_ctx_norm_rope_kv,
                        (nh_kv as u32, count as u32, 1),
                        (128, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
        }

        let ring_n = count.min(self.ring_cap);
        let tok_skip = count - ring_n;
        {
            let mut pt = target.dspark_taps_buf.device_ptr();
            let mut pf = self.fuse_probs[self.cfg.n_layers].device_ptr();
            let mut pn = self.dspark_norm.device_ptr();
            let mut po = self.ctx_feat_ring.device_ptr();
            let mut cd = d as i32;
            let mut ct = n_taps as i32;
            let mut eps = self.cfg.rms_eps;
            let mut sp = start_pos as i32;
            let mut ts = tok_skip as i32;
            let mut rc = self.ring_cap as i32;
            let mut args: [*mut c_void; 10] = [
                (&mut pt as *mut CUdeviceptr).cast(),
                (&mut pf as *mut CUdeviceptr).cast(),
                (&mut pn as *mut CUdeviceptr).cast(),
                (&mut po as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut ct as *mut i32).cast(),
                (&mut eps as *mut f32).cast(),
                (&mut sp as *mut i32).cast(),
                (&mut ts as *mut i32).cast(),
                (&mut rc as *mut i32).cast(),
            ];
            // SAFETY: Matches `dflash_mix_rmsnorm_f32`.
            unsafe {
                target.ctx.launch(
                    "dflash_mix_rmsnorm_f32",
                    self.dfk.mix_rmsnorm,
                    (1, ring_n as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }
        for t in 0..ring_n {
            let p = start_pos + tok_skip + t;
            self.ctx_feat_pos[p % self.ring_cap] = p as isize;
        }
        Ok(())
    }

    pub fn inject_batch(
        &mut self,
        target: &Inkling2CudaRunner,
        start_pos: usize,
        count: usize,
    ) -> Result<(), CudaError> {
        if count == 0 {
            return Ok(());
        }
        self.encode_inject_batch(target, start_pos, count)?;
        target.ctx.synchronize()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inject_and_draft_greedy_with_probs(
        &mut self,
        target: &mut Inkling2CudaRunner,
        inject_start_pos: usize,
        inject_count: usize,
        draft_pos0: usize,
        anchor_token_id: u32,
        k_drafts: usize,
        out_drafts: &mut [u32],
        out_probs: &mut [f32],
    ) -> Result<usize, CudaError> {
        let bs = self.cfg.block_size;
        let max_d = bs.saturating_sub(1);
        let k = k_drafts
            .min(max_d)
            .min(out_drafts.len())
            .min(out_probs.len());

        let prev_in_inj = inject_count > 0
            && draft_pos0 > inject_start_pos
            && (draft_pos0 - 1) < inject_start_pos + inject_count;
        let has_prev = draft_pos0 > 0
            && (prev_in_inj
                || self.ctx_feat_pos[(draft_pos0 - 1) % self.ring_cap]
                    == (draft_pos0 - 1) as isize);
        let run_k = if k == 0 || !has_prev || draft_pos0 + bs > self.max_ctx {
            0
        } else {
            k
        };
        if inject_count == 0 && run_k == 0 {
            return Ok(0);
        }

        if inject_count > 0 {
            self.encode_inject_batch(target, inject_start_pos, inject_count)?;
        }
        if run_k == 0 {
            target.ctx.synchronize()?;
            return Ok(0);
        }

        let d = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let nh = self.cfg.n_head;
        let nh_kv = self.cfg.n_head_kv;
        let hd = self.cfg.head_dim;
        let rank = self.cfg.markov_rank;

        // 1. Embed `anchor_token_id` + `token_embd_norm` -> `inj_norm_buf[0..d]`, init `[bs, d]` `inp_l_buf` + zero `rnn_state`
        self.draft_toks_buf.copy_from_u32(&[anchor_token_id])?;
        target.kernels.launch_embed(
            &target.ctx,
            &target.token_embd,
            self.draft_toks_buf.device_ptr(),
            self.correction_buf.device_ptr(),
            1,
            1.0,
        )?;
        target.kernels.launch_rms_norm(
            &target.ctx,
            self.correction_buf.device_ptr(),
            target.token_embd_norm.device_ptr(),
            self.inj_norm_buf.device_ptr(),
            d,
            1,
            target.cfg.rms_eps,
        )?;
        {
            let mut pinp = self.inp_l_buf.device_ptr();
            let mut panc = self.inj_norm_buf.device_ptr();
            let mut pmsk = self.mask_embd.device_ptr();
            let mut prnn = self.rnn_state_buf.device_ptr();
            let mut cd = d as i32;
            let mut cbs = bs as i32;
            let mut crk = rank as i32;
            let mut args: [*mut c_void; 7] = [
                (&mut pinp as *mut CUdeviceptr).cast(),
                (&mut panc as *mut CUdeviceptr).cast(),
                (&mut pmsk as *mut CUdeviceptr).cast(),
                (&mut prnn as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut cbs as *mut i32).cast(),
                (&mut crk as *mut i32).cast(),
            ];
            // SAFETY: Matches `dflash_init_block_f32`.
            unsafe {
                target.ctx.launch(
                    "dflash_init_block_f32",
                    self.dfk.init_block,
                    (((bs * d).max(rank) as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        // 2. 3 non-causal block-attention Transformer stages (`bs = 8` rows via MMVQ `__dp4a`!)
        let can_share_q8 = bs <= crate::kernels::MMVQ_MAX_TOK && d.is_multiple_of(32);
        for il in 0..self.cfg.n_layers {
            let l = &self.layers[il];
            target.kernels.launch_rms_norm(
                &target.ctx,
                self.inp_l_buf.device_ptr(),
                l.attn_norm.device_ptr(),
                self.norm_blk_buf.device_ptr(),
                d,
                bs,
                self.cfg.rms_eps,
            )?;
            let uses_q8 = matches!(
                l.wq.dtype,
                DType::Q8_0 | DType::Q4_0 | DType::Q5K | DType::Q6K
            );
            if can_share_q8 && uses_q8 {
                target.kernels.launch_quantize_q8_1(
                    &target.ctx,
                    self.norm_blk_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    d,
                    bs,
                )?;
            }
            target.kernels.launch_linear(
                &target.ctx,
                &l.wq,
                self.norm_blk_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                can_share_q8 && uses_q8,
                self.q_blk_buf.device_ptr(),
                bs,
                false,
            )?;
            target.kernels.launch_linear(
                &target.ctx,
                &l.wk,
                self.norm_blk_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                can_share_q8 && uses_q8,
                self.k_blk_buf.device_ptr(),
                bs,
                false,
            )?;
            target.kernels.launch_linear(
                &target.ctx,
                &l.wv,
                self.norm_blk_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                can_share_q8 && uses_q8,
                self.v_blk_buf.device_ptr(),
                bs,
                false,
            )?;

            {
                let mut pq = self.q_blk_buf.device_ptr();
                let mut pk = self.k_blk_buf.device_ptr();
                let mut pv = self.v_blk_buf.device_ptr();
                let mut pqn = l.attn_q_norm.device_ptr();
                let mut pkn = l.attn_k_norm.device_ptr();
                let mut pf = self.rope_inv_freqs.device_ptr();
                let mut cnh = nh as i32;
                let mut cnhkv = nh_kv as i32;
                let mut chd = hd as i32;
                let mut crd = self.cfg.rope_dim as i32;
                let mut p0 = draft_pos0 as i32;
                let mut eps = self.cfg.rms_eps;
                let mut args: [*mut c_void; 12] = [
                    (&mut pq as *mut CUdeviceptr).cast(),
                    (&mut pk as *mut CUdeviceptr).cast(),
                    (&mut pv as *mut CUdeviceptr).cast(),
                    (&mut pqn as *mut CUdeviceptr).cast(),
                    (&mut pkn as *mut CUdeviceptr).cast(),
                    (&mut pf as *mut CUdeviceptr).cast(),
                    (&mut cnh as *mut i32).cast(),
                    (&mut cnhkv as *mut i32).cast(),
                    (&mut chd as *mut i32).cast(),
                    (&mut crd as *mut i32).cast(),
                    (&mut p0 as *mut i32).cast(),
                    (&mut eps as *mut f32).cast(),
                ];
                // SAFETY: Matches `dflash_qk_norm_rope_f16round_f32`.
                unsafe {
                    target.ctx.launch(
                        "dflash_qk_norm_rope_f16round_f32",
                        self.dfk.qk_norm_rope_f16round,
                        ((nh + nh_kv) as u32, bs as u32, 1),
                        (128, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            let ctx_start = if self.cfg.is_swa[il] {
                (draft_pos0 + 1).saturating_sub(self.cfg.n_swa_eff)
            } else {
                0
            };
            {
                let mut pq = self.q_blk_buf.device_ptr();
                let mut pk = self.k_blk_buf.device_ptr();
                let mut pv = self.v_blk_buf.device_ptr();
                let mut pkc = l.k_cache.device_ptr();
                let mut pvc = l.v_cache.device_ptr();
                let mut po = self.attn_blk_buf.device_ptr();
                let mut cnh = nh as i32;
                let mut cnhkv = nh_kv as i32;
                let mut chd = hd as i32;
                let mut cbs = bs as i32;
                let mut cs = ctx_start as i32;
                let mut p0 = draft_pos0 as i32;
                let mut args: [*mut c_void; 12] = [
                    (&mut pq as *mut CUdeviceptr).cast(),
                    (&mut pk as *mut CUdeviceptr).cast(),
                    (&mut pv as *mut CUdeviceptr).cast(),
                    (&mut pkc as *mut CUdeviceptr).cast(),
                    (&mut pvc as *mut CUdeviceptr).cast(),
                    (&mut po as *mut CUdeviceptr).cast(),
                    (&mut cnh as *mut i32).cast(),
                    (&mut cnhkv as *mut i32).cast(),
                    (&mut chd as *mut i32).cast(),
                    (&mut cbs as *mut i32).cast(),
                    (&mut cs as *mut i32).cast(),
                    (&mut p0 as *mut i32).cast(),
                ];
                // SAFETY: Matches `dflash_noncausal_attn_f32`.
                unsafe {
                    target.ctx.launch(
                        "dflash_noncausal_attn_f32",
                        self.dfk.noncausal_attn,
                        (nh as u32, bs as u32, 1),
                        (32, 8, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            target.kernels.launch_linear(
                &target.ctx,
                &l.wo,
                self.attn_blk_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.proj_blk_buf.device_ptr(),
                bs,
                false,
            )?;
            target.kernels.launch_add_scaled(
                &target.ctx,
                self.inp_l_buf.device_ptr(),
                self.proj_blk_buf.device_ptr(),
                1.0,
                bs * d,
            )?;

            target.kernels.launch_rms_norm(
                &target.ctx,
                self.inp_l_buf.device_ptr(),
                l.ffn_norm.device_ptr(),
                self.norm_blk_buf.device_ptr(),
                d,
                bs,
                self.cfg.rms_eps,
            )?;
            let ffn_q8 = matches!(
                l.ffn_gate.dtype,
                DType::Q8_0 | DType::Q4_0 | DType::Q5K | DType::Q6K
            );
            if can_share_q8 && ffn_q8 {
                target.kernels.launch_quantize_q8_1(
                    &target.ctx,
                    self.norm_blk_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    d,
                    bs,
                )?;
            }
            target.kernels.launch_linear(
                &target.ctx,
                &l.ffn_gate,
                self.norm_blk_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                can_share_q8 && ffn_q8,
                self.ffn_gate_blk_buf.device_ptr(),
                bs,
                false,
            )?;
            target.kernels.launch_linear(
                &target.ctx,
                &l.ffn_up,
                self.norm_blk_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                can_share_q8 && ffn_q8,
                self.ffn_up_blk_buf.device_ptr(),
                bs,
                false,
            )?;
            if can_share_q8
                && matches!(
                    l.ffn_down.dtype,
                    DType::Q8_0 | DType::Q4_0 | DType::Q5K | DType::Q6K
                )
                && n_ff.is_multiple_of(32)
            {
                target.kernels.launch_swiglu_quantize_q8_1(
                    &target.ctx,
                    self.ffn_gate_blk_buf.device_ptr(),
                    self.ffn_up_blk_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    n_ff,
                    bs,
                )?;
                target.kernels.launch_linear(
                    &target.ctx,
                    &l.ffn_down,
                    self.ffn_gate_blk_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    true,
                    self.proj_blk_buf.device_ptr(),
                    bs,
                    false,
                )?;
            } else {
                target.kernels.launch_swiglu(
                    &target.ctx,
                    self.ffn_gate_blk_buf.device_ptr(),
                    self.ffn_up_blk_buf.device_ptr(),
                    bs * n_ff,
                )?;
                target.kernels.launch_linear(
                    &target.ctx,
                    &l.ffn_down,
                    self.ffn_gate_blk_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    false,
                    self.proj_blk_buf.device_ptr(),
                    bs,
                    false,
                )?;
            }
            target.kernels.launch_add_scaled(
                &target.ctx,
                self.inp_l_buf.device_ptr(),
                self.proj_blk_buf.device_ptr(),
                1.0,
                bs * d,
            )?;
        }

        // 3. Output RMSNorm -> `hidden_blk_buf [bs, d]`
        target.kernels.launch_rms_norm(
            &target.ctx,
            self.inp_l_buf.device_ptr(),
            self.output_norm.device_ptr(),
            self.hidden_blk_buf.device_ptr(),
            d,
            bs,
            self.cfg.rms_eps,
        )?;

        // 4. Chained `run_k`-step Recurrent Markov Head + Pruned `__dp4a` LM Head + Softcap-Argmax!
        let ring_slot = (draft_pos0 - 1) % self.ring_cap;
        let ctx_ring_off = (ring_slot * d) as i32;
        for step in 0..run_k {
            {
                let mut pji = self.joint_in_buf.device_ptr();
                let mut prnn = self.rnn_state_buf.device_ptr();
                let mut pmw = self.markov_w1_bf16.device_ptr();
                let mut phid = self.hidden_blk_buf.device_ptr();
                let mut pdt = self.draft_toks_buf.device_ptr();
                let mut pcr = self.ctx_feat_ring.device_ptr();
                let mut crk = rank as i32;
                let mut cd = d as i32;
                let mut st = step as i32;
                let mut cro = ctx_ring_off;
                let mut args: [*mut c_void; 10] = [
                    (&mut pji as *mut CUdeviceptr).cast(),
                    (&mut prnn as *mut CUdeviceptr).cast(),
                    (&mut pmw as *mut CUdeviceptr).cast(),
                    (&mut phid as *mut CUdeviceptr).cast(),
                    (&mut pdt as *mut CUdeviceptr).cast(),
                    (&mut pcr as *mut CUdeviceptr).cast(),
                    (&mut crk as *mut i32).cast(),
                    (&mut cd as *mut i32).cast(),
                    (&mut st as *mut i32).cast(),
                    (&mut cro as *mut i32).cast(),
                ];
                // SAFETY: Matches `dflash_markov_prep_f32`.
                unsafe {
                    target.ctx.launch(
                        "dflash_markov_prep_f32",
                        self.dfk.markov_prep,
                        (((2 * rank + 2 * d) as u32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            target.kernels.launch_linear(
                &target.ctx,
                &self.joint_proj,
                self.joint_in_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.joint_out_buf.device_ptr(),
                1,
                true,
            )?;
            {
                let mut pjo = self.joint_out_buf.device_ptr();
                let mut pjb = self.joint_proj_b.device_ptr();
                let mut prnn = self.rnn_state_buf.device_ptr();
                let mut plat = self.latent_buf.device_ptr();
                let mut crk = rank as i32;
                let mut args: [*mut c_void; 5] = [
                    (&mut pjo as *mut CUdeviceptr).cast(),
                    (&mut pjb as *mut CUdeviceptr).cast(),
                    (&mut prnn as *mut CUdeviceptr).cast(),
                    (&mut plat as *mut CUdeviceptr).cast(),
                    (&mut crk as *mut i32).cast(),
                ];
                // SAFETY: Matches `dflash_markov_gate_latent_f32`.
                unsafe {
                    target.ctx.launch(
                        "dflash_markov_gate_latent_f32",
                        self.dfk.markov_gate_latent,
                        ((rank as u32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            target.kernels.launch_linear(
                &target.ctx,
                &self.hidden_out,
                self.latent_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.correction_buf.device_ptr(),
                1,
                true,
            )?;
            {
                let mut pch = self.corr_hidden_buf.device_ptr();
                let mut phb = self.hidden_blk_buf.device_ptr();
                let mut pco = self.correction_buf.device_ptr();
                let mut phob = self.hidden_out_b.device_ptr();
                let mut cd = d as i32;
                let mut st = step as i32;
                let mut args: [*mut c_void; 6] = [
                    (&mut pch as *mut CUdeviceptr).cast(),
                    (&mut phb as *mut CUdeviceptr).cast(),
                    (&mut pco as *mut CUdeviceptr).cast(),
                    (&mut phob as *mut CUdeviceptr).cast(),
                    (&mut cd as *mut i32).cast(),
                    (&mut st as *mut i32).cast(),
                ];
                // SAFETY: Matches `dflash_markov_corr_f32`.
                unsafe {
                    target.ctx.launch(
                        "dflash_markov_corr_f32",
                        self.dfk.markov_corr,
                        ((d as u32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            if let Some((ref map_buf, n_pruned)) = self.vocab_map {
                target.kernels.launch_quantize_q8_1(
                    &target.ctx,
                    self.corr_hidden_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    d,
                    1,
                )?;
                let mut pw = target.output.ptr();
                let mut pxq = self.xq_scratch.device_ptr();
                let mut pmp = map_buf.device_ptr();
                let mut pol = target.logits_padded_buf.device_ptr();
                let mut np = n_pruned as i32;
                let mut cd = d as i32;
                let mut args: [*mut c_void; 6] = [
                    (&mut pw as *mut CUdeviceptr).cast(),
                    (&mut pxq as *mut CUdeviceptr).cast(),
                    (&mut pmp as *mut CUdeviceptr).cast(),
                    (&mut pol as *mut CUdeviceptr).cast(),
                    (&mut np as *mut i32).cast(),
                    (&mut cd as *mut i32).cast(),
                ];
                // SAFETY: Matches `dflash_indexed_q8_0_q8_1_gemv`.
                unsafe {
                    target.ctx.launch(
                        "dflash_indexed_q8_0_q8_1_gemv",
                        self.dfk.indexed_q8_0_q8_1_gemv,
                        ((n_pruned as u32).div_ceil(4), 1, 1),
                        (32, 4, 1),
                        0,
                        &mut args,
                    )?;
                }
                let mut pl = target.logits_padded_buf.device_ptr();
                let mut pm = map_buf.device_ptr();
                let mut pids = self.draft_toks_buf.device_ptr();
                let mut ppr = self.draft_probs_buf.device_ptr();
                let mut nv = n_pruned as i32;
                let mut rs = n_pruned as i32;
                let mut slot = (step + 1) as i32;
                let mut cap = self.cfg.final_logit_softcapping;
                let mut args2: [*mut c_void; 8] = [
                    (&mut pl as *mut CUdeviceptr).cast(),
                    (&mut pm as *mut CUdeviceptr).cast(),
                    (&mut pids as *mut CUdeviceptr).cast(),
                    (&mut ppr as *mut CUdeviceptr).cast(),
                    (&mut nv as *mut i32).cast(),
                    (&mut rs as *mut i32).cast(),
                    (&mut slot as *mut i32).cast(),
                    (&mut cap as *mut f32).cast(),
                ];
                // SAFETY: Matches `softcap_argmax_f32`.
                unsafe {
                    target.ctx.launch(
                        "softcap_argmax_f32",
                        target.ik.softcap_argmax,
                        (1, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args2,
                    )?;
                }
            } else {
                target.kernels.launch_linear(
                    &target.ctx,
                    &target.output,
                    self.corr_hidden_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    false,
                    target.logits_padded_buf.device_ptr(),
                    1,
                    false,
                )?;
                let mut pl = target.logits_padded_buf.device_ptr();
                let mut pm: CUdeviceptr = 0;
                let mut pids = self.draft_toks_buf.device_ptr();
                let mut ppr = self.draft_probs_buf.device_ptr();
                let mut nv = self.cfg.vocab_size as i32;
                let mut rs = self.cfg.vocab_padded as i32;
                let mut slot = (step + 1) as i32;
                let mut cap = self.cfg.final_logit_softcapping;
                let mut args2: [*mut c_void; 8] = [
                    (&mut pl as *mut CUdeviceptr).cast(),
                    (&mut pm as *mut CUdeviceptr).cast(),
                    (&mut pids as *mut CUdeviceptr).cast(),
                    (&mut ppr as *mut CUdeviceptr).cast(),
                    (&mut nv as *mut i32).cast(),
                    (&mut rs as *mut i32).cast(),
                    (&mut slot as *mut i32).cast(),
                    (&mut cap as *mut f32).cast(),
                ];
                // SAFETY: Matches `softcap_argmax_f32`.
                unsafe {
                    target.ctx.launch(
                        "softcap_argmax_f32",
                        target.ik.softcap_argmax,
                        (1, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args2,
                    )?;
                }
            }
        }

        let mut raw_toks = [0u32; 20];
        let mut raw_probs = [0.0f32; 20];
        // SAFETY: POD slices of length `1 + run_k`.
        unsafe {
            let tb = core::slice::from_raw_parts_mut(
                raw_toks.as_mut_ptr().cast::<u8>(),
                (1 + run_k) * 4,
            );
            self.draft_toks_buf.copy_to_bytes(tb)?;
        }
        self.draft_probs_buf
            .copy_to_f32(&mut raw_probs[..1 + run_k])?;
        out_drafts[..run_k].copy_from_slice(&raw_toks[1..1 + run_k]);
        out_probs[..run_k].copy_from_slice(&raw_probs[1..1 + run_k]);
        Ok(run_k)
    }
}
