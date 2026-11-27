// Adapted in part from `Strata` (`engine/src/hy4flash/cpu/hy4_cpu.cpp` & `engine/src/hy4flash/run/`,
// MIT License, Copyright (c) 2026 Agention AI).

//! NVIDIA CUDA GPU runner for `Tencent Hy4-Flash` (`.strata` v1.0 / v1.1 `S_K` + `Q8_ROW` + `MLA` + `LatentMoE`).
//!
//! - Streams Section 1 (`5.92 GB` trunk) and Section 4 (`56.04 GB` `S_K` expert arena) directly from disk
//!   into VRAM in `64 MiB` chunks so host RSS stays `< 512 MiB`!
//! - Stages the 4 Engram modules' normalized `1536`-dim inputs (`4 × 6 KiB = 24 KiB/tok`) + `any_valid[4]`
//!   from the Section 2 hot seed / external `Hy4-Flash-Engram-Q8.strata` page-LRU, then launches a single
//!   captured `CudaGraphExec` across all 42 `MLA + LatentMoE + iHC` layers!

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use tm_core::DType;
use tm_models::MAX_SPEC_WINDOW;
use tm_models::hy4::{
    FMT_F32, FMT_Q8ROW, Hy4Config, Hy4EngramCfg, Hy4EngramReader, Hy4Manifest, Hy4TensorInfo,
    discover_sibling_engram_strata, read_expert_directory, read_strata_header, read_strata_section,
};

use crate::CudaError;
use crate::driver::{
    CUdeviceptr, CUfunction, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec, CudaPinnedBuffer,
};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat};
use crate::llama_runner::{CudaProfileReport, CudaStageStat, MAX_CUDA_BATCH};

const BLOCK_Q8_1_G64_BYTES: usize = 72;
const STREAM_CHUNK_BYTES: usize = 64 * 1024 * 1024;
/// Default `DFly2` draft block length (`K = 4`, evaluating `B = K + 1 = 5` slots `[x_a, mask..mask]`).
pub const DFLY2_DEFAULT_DRAFT_K: usize = 4;

struct Hy4CudaKernels {
    quantize_g64: CUfunction,
    mmvq_sk_g64: CUfunction,
    mmvq_q8row: CUfunction,
    embed_sk: CUfunction,
    hc_apply_pre: CUfunction,
    hc_post: CUfunction,
    engram_gate_add: CUfunction,
    mla_prep_latent_kv: CUfunction,
    mla_absorb_q: CUfunction,
    mla_latent_attn: CUfunction,
    mla_vup_gate: CUfunction,
    latmoe_indexed_sk: CUfunction,
    latmoe_dual_gu_sq: CUfunction,
}

impl Hy4CudaKernels {
    fn load(ctx: &CudaContext) -> Result<Self, CudaError> {
        Ok(Self {
            quantize_g64: ctx.get_function("quantize_q8_1_g64_f32")?,
            mmvq_sk_g64: ctx.get_function("mmvq_sk_q8_1_g64")?,
            mmvq_q8row: ctx.get_function("mmvq_q8row_q8_1")?,
            embed_sk: ctx.get_function("hy4_embed_sk_f32")?,
            hc_apply_pre: ctx.get_function("hy4_hc_apply_pre_f32")?,
            hc_post: ctx.get_function("hy4_hc_post_f32")?,
            engram_gate_add: ctx.get_function("hy4_engram_gate_add_f32")?,
            mla_prep_latent_kv: ctx.get_function("hy4_mla_prep_latent_kv_f16")?,
            mla_absorb_q: ctx.get_function("hy4_mla_absorb_q")?,
            mla_latent_attn: ctx.get_function("hy4_mla_latent_attn_f16_with_sink")?,
            mla_vup_gate: ctx.get_function("hy4_mla_vup_gate")?,
            latmoe_indexed_sk: ctx.get_function("hy4_latmoe_indexed_sk_q8_1_g64")?,
            latmoe_dual_gu_sq: ctx.get_function("hy4_latmoe_dual_gu_swiglu_sq_f32")?,
        })
    }
}

#[derive(Clone)]
struct TrunkWeight {
    ptr: CUdeviceptr,
    fmt: i32,
    rows: usize,
    cols: usize,
}

struct CudaEngramLayer {
    mod_idx: usize,
    key_projs: Vec<TrunkWeight>,
    key_biases: Vec<TrunkWeight>,
    val_proj: TrunkWeight,
    val_bias: TrunkWeight,
    norm_key: CudaBuffer,   // [hc, hidden] contiguous
    norm_query: CudaBuffer, // [hc, hidden] contiguous
    hc_scales: CudaBuffer,  // [hc, hidden] contiguous
    pre_gamma: TrunkWeight,
}

struct CudaHy4HcPre {
    hc_fn: TrunkWeight,
    hc_scale: TrunkWeight,
    hc_base: TrunkWeight,
}

enum CudaHy4Mlp {
    Dense {
        gate_proj: TrunkWeight,
        up_proj: TrunkWeight,
        down_proj: TrunkWeight,
        post_fc2_norm: TrunkWeight,
    },
    LatentMoe {
        moe_idx: usize,
        gate: TrunkWeight,
        e_bias: TrunkWeight,
        routed_down: TrunkWeight,
        routed_up: TrunkWeight,
        sh_gate: TrunkWeight,
        sh_up: TrunkWeight,
        sh_down: TrunkWeight,
        sh_ff: usize,
        combine_norm: TrunkWeight,
    },
}

struct CudaHy4Layer {
    engram: Option<CudaEngramLayer>,
    hc_attn: CudaHy4HcPre,
    input_ln: TrunkWeight,
    is_swa: bool,
    n_head: usize,
    ring_cap: usize,
    q_a_proj: TrunkWeight,
    q_a_ln: TrunkWeight,
    q_b_proj: TrunkWeight,
    kv_a_proj: TrunkWeight,
    kv_a_ln: TrunkWeight,
    kv_b_proj: TrunkWeight,
    sinks: TrunkWeight,
    linear_gate: TrunkWeight,
    o_proj: TrunkWeight,
    post_proj_norm: TrunkWeight,
    /// Matrix-Absorbed MLA compressed latent cache: `[cap, kv_lora + qk_rope]` in FP16 (`P1-1`).
    latent_kv_cache: CudaBuffer,
    hc_mlp: CudaHy4HcPre,
    post_attn_ln: TrunkWeight,
    mlp: CudaHy4Mlp,
}

struct CudaDflyLayer {
    target_layer_idx: usize,
    input_ln: TrunkWeight,
    q_a_proj: TrunkWeight,
    q_a_ln: TrunkWeight,
    q_b_proj: TrunkWeight,
    kv_a_proj: TrunkWeight,
    kv_a_ln: TrunkWeight,
    kv_b_proj: TrunkWeight,
    sinks: TrunkWeight,
    linear_gate: TrunkWeight,
    o_proj: TrunkWeight,
    post_proj_norm: TrunkWeight,
    post_attn_ln: TrunkWeight,
    gate_proj: TrunkWeight,
    up_proj: TrunkWeight,
    down_proj: TrunkWeight,
    mlp_ff: usize,
}

struct CudaHy4DFly2 {
    fusion_enorm: TrunkWeight,
    fusion_hnorm: TrunkWeight,
    fusion_eh_proj: TrunkWeight,
    fusion_gate: TrunkWeight,
    layers: Vec<CudaDflyLayer>,
    final_norm: TrunkWeight,
    corr_hidden_norm: TrunkWeight,
    corr_embed_norm: TrunkWeight,
    corr_gate_proj: TrunkWeight,
    corr_up_proj: TrunkWeight,
    corr_down_proj: TrunkWeight,
    corr_ff: usize,
    conf_proj: TrunkWeight,
    conf_bias: TrunkWeight,
    /// Post-`model.norm` hidden state at position `a - 1` (`[hidden]` F32).
    hprev_f32: CudaBuffer,
    h_buf: CudaBuffer,          // [MAX_SPEC_WINDOW + 1, hidden]
    hf_buf: CudaBuffer,         // [MAX_SPEC_WINDOW + 1, hidden]
    hcin_buf: CudaBuffer,       // [2 * hidden]
    ec_buf: CudaBuffer,         // [hidden]
    hcorr_buf: CudaBuffer,      // [hidden]
    conf_logit_buf: CudaBuffer, // [1]
    chain_ids_dev: CudaBuffer,  // [MAX_SPEC_WINDOW + 1] u32
    conf_probs_dev: CudaBuffer, // [MAX_SPEC_WINDOW + 1] f32
}

pub struct Hy4CudaRunner {
    pub ctx: CudaContext,
    pub kernels: CudaKernels,
    hk: Hy4CudaKernels,
    pub cfg: Hy4Config,
    pub engram_cfg: Hy4EngramCfg,
    pub max_ctx: usize,
    pub vram_bytes: usize,
    pub model_weight_bytes: usize,
    pub active_decode_bytes: usize,

    trunk_arena: Arc<CudaBuffer>,
    sec4_arena: CudaBuffer,
    expert_dir_dev: CudaBuffer,
    rank_of_dev: CudaBuffer,

    embed_tokens: TrunkWeight,
    hc_head_fn: TrunkWeight,
    hc_head_scale: TrunkWeight,
    hc_head_base: TrunkWeight,
    final_norm: TrunkWeight,
    lm_head: TrunkWeight,
    unit_ones_hc: CudaBuffer,
    inv_freqs_sparse: CudaBuffer,
    inv_freqs_swa: CudaBuffer,
    layers: Vec<CudaHy4Layer>,
    dfly2: Option<CudaHy4DFly2>,

    // Host Engram state + pinned async staging (`P1-2`)
    engram_reader: Hy4EngramReader,
    engram_ln_weights: Vec<Vec<f32>>,
    seq_tokens: Vec<u32>,
    pinned_ids: CudaPinnedBuffer,
    pinned_pos: CudaPinnedBuffer,
    pinned_engram_r: CudaPinnedBuffer,
    pinned_engram_valid: CudaPinnedBuffer,

    // Device execution buffers + captured CUDA Graphs
    token_ids_buf: CudaBuffer,
    pos_buf: CudaBuffer,
    engram_r_dev: CudaBuffer,     // [n_mod, MAX_CUDA_BATCH, engram_dim]
    engram_valid_dev: CudaBuffer, // [n_mod, MAX_CUDA_BATCH]
    engram_kv_buf: CudaBuffer,    // [MAX_CUDA_BATCH, 5, hidden]
    engram_tmp_buf: CudaBuffer,   // [MAX_CUDA_BATCH, hidden]
    decode_graphs: [Option<CudaGraphExec>; 3],

    x_hc_buf: CudaBuffer,
    fl_buf: CudaBuffer,
    mix_buf: CudaBuffer,
    post_buf: CudaBuffer,
    cur_buf: CudaBuffer,
    curn_buf: CudaBuffer,
    hf_verify_buf: CudaBuffer,
    sub_buf: CudaBuffer,
    qa_buf: CudaBuffer,
    qr_buf: CudaBuffer,
    q_buf: CudaBuffer,
    q_abs_buf: CudaBuffer,
    kva_buf: CudaBuffer,
    ctx_lat_buf: CudaBuffer,
    attn_buf: CudaBuffer,
    gate_buf: CudaBuffer,
    ff1_buf: CudaBuffer,
    ff2_buf: CudaBuffer,
    router_logits_buf: CudaBuffer,
    top_ids_buf: CudaBuffer,
    top_weights_buf: CudaBuffer,
    lat_z_buf: CudaBuffer,
    lat_act_f32_buf: CudaBuffer,
    lat_slots_buf: CudaBuffer,
    xq_q8_1_buf: CudaBuffer,
    xq_g64_buf: CudaBuffer,
    slot_xq_g64_buf: CudaBuffer,
    logits_dev: CudaBuffer,
    argmax_dev: CudaBuffer,
    logits_host: Vec<f32>,
}

fn stream_file_region_to_vram(
    ctx: &CudaContext,
    f: &mut File,
    file_offset: u64,
    byte_length: u64,
    pinned_staging: &mut CudaPinnedBuffer,
) -> Result<CudaBuffer, CudaError> {
    if byte_length == 0 {
        return ctx.alloc_zeroed(64);
    }
    let buf = ctx.alloc_uninit(byte_length as usize)?;
    f.seek(SeekFrom::Start(file_offset)).map_err(|e| {
        CudaError::InvalidDimensions(format!("seek .strata offset {file_offset}: {e}"))
    })?;
    let total = byte_length as usize;
    let half_cap = (pinned_staging.len() / 2).max(4096);
    if total <= half_cap {
        let dst = &mut pinned_staging.as_bytes_mut()[..total];
        f.read_exact(dst)
            .map_err(|e| CudaError::InvalidDimensions(format!("read .strata: {e}")))?;
        buf.copy_from_pinned_async(0, pinned_staging, 0, total)?;
        ctx.synchronize()?;
        return Ok(buf);
    }

    // `P1-2`: Double-buffered disk read + async PCIe DMA overlap!
    let evs = [CudaEvent::new(ctx)?, CudaEvent::new(ctx)?];
    let mut copied = 0usize;
    let mut slot = 0usize;
    let mut slot_used = [false; 2];
    while copied < total {
        if slot_used[slot] {
            evs[slot].synchronize()?;
        }
        let want = (total - copied).min(half_cap);
        let pin_off = slot * half_cap;
        f.read_exact(&mut pinned_staging.as_bytes_mut()[pin_off..pin_off + want])
            .map_err(|e| CudaError::InvalidDimensions(format!("read .strata at +{copied}: {e}")))?;
        buf.copy_from_pinned_async(copied, pinned_staging, pin_off, want)?;
        evs[slot].record()?;
        slot_used[slot] = true;
        copied += want;
        slot ^= 1;
    }
    ctx.synchronize()?;
    Ok(buf)
}

impl Hy4CudaRunner {
    pub fn from_strata(
        model_path: &Path,
        engram_path: Option<&Path>,
        max_ctx: usize,
    ) -> Result<(Self, String), CudaError> {
        let mut f = File::open(model_path).map_err(|e| {
            CudaError::InvalidDimensions(format!("open '{}': {e}", model_path.display()))
        })?;
        let hdr = read_strata_header(&mut f)?;
        let (_, man) = Hy4Manifest::from_strata_path(model_path, true)?;
        let cfg = man.cfg.clone();
        let ecfg = man.engram.clone();

        let ctx = CudaContext::new(None)?;
        let kernels = CudaKernels::load(&ctx)?;
        let hk = Hy4CudaKernels::load(&ctx)?;

        let sec1_bytes = hdr.sections[1].byte_length as usize;
        let sec4_bytes = hdr.sections[4].byte_length as usize;
        let model_weight_bytes = sec1_bytes + sec4_bytes;

        let kv_lat = cfg.kv_lora + cfg.qk_rope;
        let mut kv_total_bytes = 0usize;
        for il in 0..cfg.n_layers {
            let cap = if cfg.is_swa[il] && cfg.window > 0 {
                (cfg.window + MAX_CUDA_BATCH).min(max_ctx)
            } else {
                max_ctx
            };
            kv_total_bytes += cap * kv_lat * 2;
        }

        let (free_vram, _) = ctx.mem_info()?;
        let needed_bytes = model_weight_bytes + kv_total_bytes + 192 * 1024 * 1024;
        if needed_bytes > free_vram {
            return Err(CudaError::OutOfDeviceMemory {
                needed_mib: needed_bytes as f64 / (1024.0 * 1024.0),
                free_mib: free_vram as f64 / (1024.0 * 1024.0),
                max_ctx,
                device: ctx.info().name.clone(),
            });
        }

        // Stream Section 1 (trunk) and Section 4 (expert bodies) via double-buffered pinned host DMA (`P1-2`)!
        let pin_bytes = (2 * STREAM_CHUNK_BYTES).min(2 * sec1_bytes.max(sec4_bytes).max(4096));
        let mut pinned_staging = ctx.alloc_pinned(pin_bytes)?;
        let trunk_arena = Arc::new(stream_file_region_to_vram(
            &ctx,
            &mut f,
            hdr.sections[1].file_offset,
            hdr.sections[1].byte_length,
            &mut pinned_staging,
        )?);
        let sec4_arena = stream_file_region_to_vram(
            &ctx,
            &mut f,
            hdr.sections[4].file_offset,
            hdr.sections[4].byte_length,
            &mut pinned_staging,
        )?;
        drop(pinned_staging);

        // Read Section 2 (64 MiB hot Engram seed) + Section 3 (expert directory)
        let seed = read_strata_section(&mut f, &hdr, 2)?;
        let expert_dir = read_expert_directory(&mut f, &hdr)?;
        let mut dir_bytes = Vec::with_capacity(expert_dir.len() * 32);
        for ent in &expert_dir {
            dir_bytes.extend_from_slice(&ent.to_le_bytes());
        }
        let expert_dir_dev = if dir_bytes.is_empty() {
            ctx.alloc_zeroed(64)?
        } else {
            ctx.alloc_from_bytes(&dir_bytes)?
        };

        let ne = cfg.n_experts;
        let n_moe = cfg.n_layers.saturating_sub(1);
        let mut rank_of = vec![-1i32; (n_moe * ne).max(1)];
        for (i, ent) in expert_dir.iter().enumerate() {
            let ml = i / ne.max(1);
            if ml < n_moe && (ent.expert_id as usize) < ne {
                rank_of[ml * ne + ent.expert_id as usize] = (i % ne) as i32;
            }
        }
        // SAFETY: `[i32]` is POD.
        let rank_bytes = unsafe {
            core::slice::from_raw_parts(rank_of.as_ptr().cast::<u8>(), rank_of.len() * 4)
        };
        let rank_of_dev = ctx.alloc_from_bytes(rank_bytes)?;

        // Read tiny Engram per-head norm weights & stacked `[hc, hidden]` norm/scale vectors from Section 1 on disk
        let mut read_trunk_f32 = |name: &str| -> Result<Vec<f32>, CudaError> {
            let ti = man.req_tensor(name)?;
            let mut raw = vec![0u8; ti.bytes as usize];
            f.seek(SeekFrom::Start(hdr.sections[1].file_offset + ti.offset))
                .map_err(|e| CudaError::InvalidDimensions(format!("seek {name}: {e}")))?;
            f.read_exact(&mut raw)
                .map_err(|e| CudaError::InvalidDimensions(format!("read {name}: {e}")))?;
            let n: usize = ti.shape.iter().product();
            let mut out = vec![0.0f32; n];
            tm_models::dequant_hy4_row(ti.fmt, &raw, 1, n, 0, &mut out);
            Ok(out)
        };

        let mut engram_ln_weights = Vec::with_capacity(ecfg.n_mod);
        for &il in &ecfg.layers {
            let mut lnw = vec![0.0f32; ecfg.engram_dim];
            let rd = ecfg.row_dim;
            for ch in 0..ecfg.n_cols {
                let v = read_trunk_f32(&format!(
                    "model.layers.{il}.engram_layer.lookup_per_head_norms.{ch}.weight"
                ))?;
                lnw[ch * rd..(ch + 1) * rd].copy_from_slice(&v);
            }
            engram_ln_weights.push(lnw);
        }

        let tw = |name: &str| -> Result<TrunkWeight, CudaError> {
            let ti: &Hy4TensorInfo = man.req_tensor(name)?;
            Ok(TrunkWeight {
                ptr: trunk_arena.device_ptr() + ti.offset,
                fmt: ti.fmt,
                rows: ti.rows(),
                cols: ti.cols(),
            })
        };

        let embed_tokens = tw("model.embed_tokens.weight")?;
        let hc_head_fn = tw("model.hc_head.hc_head_fn")?;
        let hc_head_scale = tw("model.hc_head.hc_head_scale")?;
        let hc_head_base = tw("model.hc_head.hc_head_base")?;
        let final_norm = tw("model.norm.weight")?;
        let lm_head = tw("lm_head.weight")?;

        let h = cfg.n_embd;
        let hc = cfg.hc_count;
        let hc_dim = hc * h;
        let unit_ones_hc = ctx.alloc_from_f32(&vec![1.0f32; hc_dim])?;

        let half_rope = cfg.qk_rope / 2;
        let inv_sp: Vec<f32> = (0..half_rope)
            .map(|i| {
                cfg.theta_sparse
                    .powf(-2.0 * (i as f64) / (cfg.qk_rope as f64)) as f32
            })
            .collect();
        let inv_sw: Vec<f32> = (0..half_rope)
            .map(|i| cfg.theta_swa.powf(-2.0 * (i as f64) / (cfg.qk_rope as f64)) as f32)
            .collect();
        let inv_freqs_sparse = ctx.alloc_from_f32(&inv_sp)?;
        let inv_freqs_swa = ctx.alloc_from_f32(&inv_sw)?;

        let mut max_sh_ff = cfg.lat_dim.max(1);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for il in 0..cfg.n_layers {
            let engram = if let Some(mod_idx) = ecfg.layers.iter().position(|&l| l == il) {
                let p = format!("model.layers.{il}.engram_layer");
                let mut key_projs = Vec::with_capacity(hc);
                let mut key_biases = Vec::with_capacity(hc);
                let mut nk_cat = Vec::with_capacity(hc * h);
                let mut nq_cat = Vec::with_capacity(hc * h);
                let mut hs_cat = Vec::with_capacity(hc * h);
                for j in 0..hc {
                    key_projs.push(tw(&format!("{p}.key_projs.{j}.weight"))?);
                    key_biases.push(tw(&format!("{p}.key_projs.{j}.bias"))?);
                    nk_cat.extend_from_slice(&read_trunk_f32(&format!("{p}.norm_key.{j}.weight"))?);
                    nq_cat
                        .extend_from_slice(&read_trunk_f32(&format!("{p}.norm_query.{j}.weight"))?);
                    hs_cat.extend_from_slice(&read_trunk_f32(&format!(
                        "{p}.unified_per_hc_scales.{j}"
                    ))?);
                }
                Some(CudaEngramLayer {
                    mod_idx,
                    key_projs,
                    key_biases,
                    val_proj: tw(&format!("{p}.value_proj.weight"))?,
                    val_bias: tw(&format!("{p}.value_proj.bias"))?,
                    norm_key: ctx.alloc_from_f32(&nk_cat)?,
                    norm_query: ctx.alloc_from_f32(&nq_cat)?,
                    hc_scales: ctx.alloc_from_f32(&hs_cat)?,
                    pre_gamma: tw(&format!("{p}.unified_pre_gamma"))?,
                })
            } else {
                None
            };

            let lp = format!("model.layers.{il}");
            let hc_attn = CudaHy4HcPre {
                hc_fn: tw(&format!("{lp}.hc_attn_layer.hc_pre.hc_fn"))?,
                hc_scale: tw(&format!("{lp}.hc_attn_layer.hc_pre.hc_scale"))?,
                hc_base: tw(&format!("{lp}.hc_attn_layer.hc_pre.hc_base"))?,
            };
            let hc_mlp = CudaHy4HcPre {
                hc_fn: tw(&format!("{lp}.hc_mlp_layer.hc_pre.hc_fn"))?,
                hc_scale: tw(&format!("{lp}.hc_mlp_layer.hc_pre.hc_scale"))?,
                hc_base: tw(&format!("{lp}.hc_mlp_layer.hc_pre.hc_base"))?,
            };

            let swa = cfg.is_swa[il];
            let nh = if swa { cfg.heads_swa } else { cfg.heads_sparse };
            let cap = if swa && cfg.window > 0 {
                (cfg.window + MAX_CUDA_BATCH).min(max_ctx)
            } else {
                max_ctx
            };
            let ring_cap = if swa && cfg.window > 0 && cap < max_ctx {
                cap
            } else {
                0
            };
            let ap = format!("{lp}.self_attn");

            let mp = format!("{lp}.mlp");
            let mlp = if il == 0 {
                CudaHy4Mlp::Dense {
                    gate_proj: tw(&format!("{mp}.gate_proj.weight"))?,
                    up_proj: tw(&format!("{mp}.up_proj.weight"))?,
                    down_proj: tw(&format!("{mp}.down_proj.weight"))?,
                    post_fc2_norm: tw(&format!("{mp}.post_fc2_norm.weight"))?,
                }
            } else {
                let sh_gate = tw(&format!("{mp}.shared_experts.gate_proj.weight"))?;
                let sh_ff = sh_gate.rows;
                max_sh_ff = max_sh_ff.max(sh_ff);
                CudaHy4Mlp::LatentMoe {
                    moe_idx: il - 1,
                    gate: tw(&format!("{mp}.gate.weight"))?,
                    e_bias: tw(&format!("{mp}.gate.e_score_correction_bias"))?,
                    routed_down: tw(&format!("{mp}.routed_expert_down_proj.weight"))?,
                    routed_up: tw(&format!("{mp}.routed_expert_up_proj.weight"))?,
                    sh_gate,
                    sh_up: tw(&format!("{mp}.shared_experts.up_proj.weight"))?,
                    sh_down: tw(&format!("{mp}.shared_experts.down_proj.weight"))?,
                    sh_ff,
                    combine_norm: tw(&format!("{mp}.combine_output_norm.weight"))?,
                }
            };

            layers.push(CudaHy4Layer {
                engram,
                hc_attn,
                input_ln: tw(&format!("{lp}.input_layernorm.weight"))?,
                is_swa: swa,
                n_head: nh,
                ring_cap,
                q_a_proj: tw(&format!("{ap}.q_a_proj.weight"))?,
                q_a_ln: tw(&format!("{ap}.q_a_layernorm.weight"))?,
                q_b_proj: tw(&format!("{ap}.q_b_proj.weight"))?,
                kv_a_proj: tw(&format!("{ap}.kv_a_proj_with_mqa.weight"))?,
                kv_a_ln: tw(&format!("{ap}.kv_a_layernorm.weight"))?,
                kv_b_proj: tw(&format!("{ap}.kv_b_proj.weight"))?,
                sinks: tw(&format!("{ap}.learnable_sink_param"))?,
                linear_gate: tw(&format!("{ap}.linear_gate.weight"))?,
                o_proj: tw(&format!("{ap}.o_proj.weight"))?,
                post_proj_norm: tw(&format!("{ap}.post_proj_norm.weight"))?,
                latent_kv_cache: ctx.alloc_zeroed(cap * kv_lat * 2)?,
                hc_mlp,
                post_attn_ln: tw(&format!("{lp}.post_attention_layernorm.weight"))?,
                mlp,
            });
        }

        // Optional built-in 5-layer `DFly2` speculative drafter (`model.dfly_layers.*` in Section 1)
        let spec_rows = MAX_SPEC_WINDOW + 1;
        let mut max_dfly_ff = 0usize;
        let dfly2 = if !cfg.dfly_layers.is_empty()
            && man
                .tensors
                .contains_key("model.dfly_layers.mtp_fusion.eh_proj.weight")
        {
            let pf = "model.dfly_layers";
            let mut dfly_start = u64::MAX;
            let mut dfly_end = 0u64;
            for (nm, ti) in &man.tensors {
                if nm.starts_with(pf) {
                    dfly_start = dfly_start.min(ti.offset);
                    dfly_end = dfly_end.max(ti.offset + ti.bytes);
                }
            }
            if dfly_end > dfly_start {
                ctx.pin_l2_persisting(
                    trunk_arena.device_ptr() + dfly_start,
                    (dfly_end - dfly_start) as usize,
                );
            }

            let mut d_layers = Vec::with_capacity(cfg.dfly_layers.len());
            for (d_idx, &target_l) in cfg.dfly_layers.iter().enumerate() {
                let dp = format!("{pf}.{d_idx}");
                let da = format!("{dp}.self_attn");
                let dm = format!("{dp}.mlp");
                let gate_proj = tw(&format!("{dm}.gate_proj.weight"))?;
                let mlp_ff = gate_proj.rows;
                max_dfly_ff = max_dfly_ff.max(mlp_ff);
                d_layers.push(CudaDflyLayer {
                    target_layer_idx: target_l,
                    input_ln: tw(&format!("{dp}.input_layernorm.weight"))?,
                    q_a_proj: tw(&format!("{da}.q_a_proj.weight"))?,
                    q_a_ln: tw(&format!("{da}.q_a_layernorm.weight"))?,
                    q_b_proj: tw(&format!("{da}.q_b_proj.weight"))?,
                    kv_a_proj: tw(&format!("{da}.kv_a_proj_with_mqa.weight"))?,
                    kv_a_ln: tw(&format!("{da}.kv_a_layernorm.weight"))?,
                    kv_b_proj: tw(&format!("{da}.kv_b_proj.weight"))?,
                    sinks: tw(&format!("{da}.learnable_sink_param"))?,
                    linear_gate: tw(&format!("{da}.linear_gate.weight"))?,
                    o_proj: tw(&format!("{da}.o_proj.weight"))?,
                    post_proj_norm: tw(&format!("{da}.post_proj_norm.weight"))?,
                    post_attn_ln: tw(&format!("{dp}.post_attention_layernorm.weight"))?,
                    gate_proj,
                    up_proj: tw(&format!("{dm}.up_proj.weight"))?,
                    down_proj: tw(&format!("{dm}.down_proj.weight"))?,
                    mlp_ff,
                });
            }
            let corr_gate_proj = tw(&format!("{pf}.hidden_correction.gate_proj.weight"))?;
            let corr_ff = corr_gate_proj.rows;
            max_dfly_ff = max_dfly_ff.max(corr_ff);
            Some(CudaHy4DFly2 {
                fusion_enorm: tw(&format!("{pf}.mtp_fusion.enorm.weight"))?,
                fusion_hnorm: tw(&format!("{pf}.mtp_fusion.hnorm.weight"))?,
                fusion_eh_proj: tw(&format!("{pf}.mtp_fusion.eh_proj.weight"))?,
                fusion_gate: tw(&format!("{pf}.mtp_fusion.gate"))?,
                layers: d_layers,
                final_norm: tw(&format!("{pf}.final_norm.weight"))?,
                corr_hidden_norm: tw(&format!("{pf}.hidden_correction.hidden_norm.weight"))?,
                corr_embed_norm: tw(&format!("{pf}.hidden_correction.embed_norm.weight"))?,
                corr_gate_proj,
                corr_up_proj: tw(&format!("{pf}.hidden_correction.up_proj.weight"))?,
                corr_down_proj: tw(&format!("{pf}.hidden_correction.down_proj.weight"))?,
                corr_ff,
                conf_proj: tw(&format!("{pf}.confidence_head.proj.weight"))?,
                conf_bias: tw(&format!("{pf}.confidence_head.proj.bias"))?,
                hprev_f32: ctx.alloc_zeroed(h * 4)?,
                h_buf: ctx.alloc_zeroed(spec_rows * h * 4)?,
                hf_buf: ctx.alloc_zeroed(spec_rows * h * 4)?,
                hcin_buf: ctx.alloc_zeroed(2 * h * 4)?,
                ec_buf: ctx.alloc_zeroed(h * 4)?,
                hcorr_buf: ctx.alloc_zeroed(h * 4)?,
                conf_logit_buf: ctx.alloc_zeroed(4)?,
                chain_ids_dev: ctx.alloc_zeroed(spec_rows * 4)?,
                conf_probs_dev: ctx.alloc_zeroed(spec_rows * 4)?,
            })
        } else {
            None
        };

        let sibling_eng = if engram_path.is_none() {
            discover_sibling_engram_strata(model_path)
        } else {
            None
        };
        let eng_ref = engram_path.or(sibling_eng.as_deref());
        let engram_reader = Hy4EngramReader::open(eng_ref, &seed, ecfg.row_dim, 65_536)?;

        let max_nh = cfg.heads_swa.max(cfg.heads_sparse);
        let max_qd = max_nh * cfg.qk_head;
        let max_vd = max_nh * cfg.v_head;
        let max_ff = cfg.dense_ff.max(max_sh_ff).max(max_dfly_ff).max(h);
        let max_cols = [
            h,
            2 * h,
            hc_dim,
            cfg.q_lora,
            cfg.kv_lora,
            max_vd,
            max_ff,
            cfg.lat_dim,
            ecfg.engram_dim,
        ]
        .into_iter()
        .max()
        .unwrap_or(h)
        .next_multiple_of(256);

        let n_mod = ecfg.n_mod.max(1);
        let ed = ecfg.engram_dim.max(1);
        let rows_slots = MAX_CUDA_BATCH * cfg.top_k.max(1);

        let pinned_ids = ctx.alloc_pinned(MAX_CUDA_BATCH * 4)?;
        let pinned_pos = ctx.alloc_pinned(4)?;
        let pinned_engram_r = ctx.alloc_pinned(n_mod * MAX_CUDA_BATCH * ed * 4)?;
        let pinned_engram_valid = ctx.alloc_pinned(n_mod * MAX_CUDA_BATCH * 4)?;
        let token_ids_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * 4)?;
        let pos_buf = ctx.alloc_zeroed(4)?;
        let engram_r_dev = ctx.alloc_zeroed(n_mod * MAX_CUDA_BATCH * ed * 4)?;
        let engram_valid_dev = ctx.alloc_zeroed(n_mod * MAX_CUDA_BATCH * 4)?;
        let engram_kv_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * (hc + 1) * h * 4)?;
        let engram_tmp_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * h * 4)?;

        let x_hc_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let fl_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let mix_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * 2 * hc * 4)?;
        let post_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc * 4)?;
        let cur_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * h * 4)?;
        let curn_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * h * 4)?;
        let hf_verify_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * h * 4)?;
        let sub_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * h * 4)?;
        let qa_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.q_lora * 4)?;
        let qr_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.q_lora * 4)?;
        let q_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_qd * 4)?;
        let q_abs_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_nh * kv_lat * 4)?;
        let kva_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * kv_lat * 4)?;
        let ctx_lat_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_nh * cfg.kv_lora * 4)?;
        let attn_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_vd * 4)?;
        let gate_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_vd * 4)?;
        let ff1_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_ff * 4)?;
        let ff2_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_ff * 4)?;
        let router_logits_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_experts.max(1) * 4)?;
        let top_ids_buf = ctx.alloc_zeroed(rows_slots * 4)?;
        let top_weights_buf = ctx.alloc_zeroed(rows_slots * 4)?;
        let lat = cfg.lat_dim.max(64);
        let lat_z_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * lat * 4)?;
        let lat_act_f32_buf = ctx.alloc_zeroed(rows_slots * lat * 4)?;
        let lat_slots_buf = ctx.alloc_zeroed(rows_slots * lat * 4)?;
        let max_ctx_rows = MAX_CUDA_BATCH * max_nh;
        let xq_q8_1_buf = ctx.alloc_zeroed(
            (MAX_CUDA_BATCH * (max_cols / 32)).max(max_ctx_rows * cfg.kv_lora.div_ceil(32))
                * BLOCK_Q8_1_BYTES,
        )?;
        let xq_g64_buf = ctx.alloc_zeroed(
            (MAX_CUDA_BATCH * (max_cols / 64)).max(max_ctx_rows * cfg.kv_lora.div_ceil(64))
                * BLOCK_Q8_1_G64_BYTES,
        )?;
        let slot_xq_g64_buf =
            ctx.alloc_zeroed(rows_slots * (lat.div_ceil(64)) * BLOCK_Q8_1_G64_BYTES)?;
        let logits_dev = ctx.alloc_zeroed(spec_rows * cfg.vocab_size * 4)?;
        let argmax_dev = ctx.alloc_zeroed(spec_rows * 4)?;
        ctx.synchronize()?;

        let active_exp_bytes = (sec4_bytes / cfg.n_experts.max(1)) * cfg.top_k;
        let active_decode_bytes = sec1_bytes + active_exp_bytes;
        let (free_after, _) = ctx.mem_info()?;
        let vram_bytes = free_vram.saturating_sub(free_after);

        Ok((
            Self {
                ctx,
                kernels,
                hk,
                cfg: cfg.clone(),
                engram_cfg: ecfg.clone(),
                max_ctx,
                vram_bytes,
                model_weight_bytes,
                active_decode_bytes,
                trunk_arena,
                sec4_arena,
                expert_dir_dev,
                rank_of_dev,
                embed_tokens,
                hc_head_fn,
                hc_head_scale,
                hc_head_base,
                final_norm,
                lm_head,
                unit_ones_hc,
                inv_freqs_sparse,
                inv_freqs_swa,
                layers,
                dfly2,
                engram_reader,
                engram_ln_weights,
                seq_tokens: vec![ecfg.pad_id as u32; max_ctx.max(1)],
                pinned_ids,
                pinned_pos,
                pinned_engram_r,
                pinned_engram_valid,
                token_ids_buf,
                pos_buf,
                engram_r_dev,
                engram_valid_dev,
                engram_kv_buf,
                engram_tmp_buf,
                decode_graphs: [None, None, None],
                x_hc_buf,
                fl_buf,
                mix_buf,
                post_buf,
                cur_buf,
                curn_buf,
                hf_verify_buf,
                sub_buf,
                qa_buf,
                qr_buf,
                q_buf,
                q_abs_buf,
                kva_buf,
                ctx_lat_buf,
                attn_buf,
                gate_buf,
                ff1_buf,
                ff2_buf,
                router_logits_buf,
                top_ids_buf,
                top_weights_buf,
                lat_z_buf,
                lat_act_f32_buf,
                lat_slots_buf,
                xq_q8_1_buf,
                xq_g64_buf,
                slot_xq_g64_buf,
                logits_dev,
                argmax_dev,
                logits_host: vec![0.0; cfg.vocab_size],
            },
            man.tokenizer_json,
        ))
    }

    #[inline]
    pub fn has_builtin_dfly2(&self) -> bool {
        self.dfly2.is_some()
    }

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        for l in &self.layers {
            l.latent_kv_cache.zero()?;
        }
        if let Some(ref dfly) = self.dfly2 {
            dfly.hprev_f32.zero()?;
        }
        self.ctx.synchronize()
    }

    fn stage_engram_to_gpu(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
    ) -> Result<(), CudaError> {
        for (i, &tid) in token_ids.iter().enumerate() {
            self.seq_tokens[start_pos + i] = tid;
        }
        let n_mod = self.engram_cfg.n_mod;
        if n_mod == 0 {
            return Ok(());
        }
        let n_tok = token_ids.len();
        let ed = self.engram_cfg.engram_dim;
        let r_len = n_mod * n_tok * ed;
        let v_len = n_mod * n_tok;
        {
            let r_host = self.pinned_engram_r.as_f32_slice_mut(r_len);
            let v_host = self.pinned_engram_valid.as_f32_slice_mut(v_len);
            for m_idx in 0..n_mod {
                let r_slice = &mut r_host[m_idx * n_tok * ed..(m_idx + 1) * n_tok * ed];
                let v_slice = &mut v_host[m_idx * n_tok..(m_idx + 1) * n_tok];
                self.engram_reader.compute_module_input(
                    &self.engram_cfg,
                    &self.engram_ln_weights[m_idx],
                    &self.seq_tokens,
                    token_ids,
                    start_pos,
                    m_idx,
                    r_slice,
                    v_slice,
                )?;
            }
        }
        self.engram_r_dev
            .copy_from_pinned_async(0, &self.pinned_engram_r, 0, r_len * 4)?;
        self.engram_valid_dev
            .copy_from_pinned_async(0, &self.pinned_engram_valid, 0, v_len * 4)
    }

    fn launch_quantize_g64(
        &self,
        x_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        cols: usize,
        n_rows: usize,
    ) -> Result<(), CudaError> {
        let mut px = x_ptr;
        let mut po = out_ptr;
        let mut cc = cols as i32;
        let mut nr = n_rows as i32;
        let mut args: [*mut c_void; 4] = [
            (&mut px as *mut CUdeviceptr).cast(),
            (&mut po as *mut CUdeviceptr).cast(),
            (&mut cc as *mut i32).cast(),
            (&mut nr as *mut i32).cast(),
        ];
        // SAFETY: Matches `quantize_q8_1_g64_f32`.
        unsafe {
            self.ctx.launch(
                "quantize_q8_1_g64_f32",
                self.hk.quantize_g64,
                (((cols / 64) as u32).div_ceil(2), n_rows as u32, 1),
                (32, 2, 1),
                0,
                &mut args,
            )
        }
    }

    /// Launches a `.strata` trunk linear projection (`F32`, `Q8_ROW`, or `S_K`) across arbitrary `n_tok`
    /// by looping in `MMVQ_MAX_TOK` (`8`-token) micro-batches when `n_tok > 8`!
    fn launch_hy4_linear(
        &self,
        w: &TrunkWeight,
        x_ptr: CUdeviceptr,
        y_ptr: CUdeviceptr,
        n_tok: usize,
        g64_already_quantized: bool,
    ) -> Result<(), CudaError> {
        if w.fmt == FMT_F32 {
            let mat = CudaMat {
                buf: Arc::clone(&self.trunk_arena),
                offset: (w.ptr - self.trunk_arena.device_ptr()) as usize,
                dtype: DType::F32,
                rows: w.rows,
                cols: w.cols,
                row_bytes: w.cols * 4,
            };
            return self.kernels.launch_linear(
                &self.ctx,
                &mat,
                x_ptr,
                self.xq_q8_1_buf.device_ptr(),
                false,
                y_ptr,
                n_tok,
                true,
            );
        }

        if w.fmt == FMT_Q8ROW {
            let mut off = 0usize;
            while off < n_tok {
                let b = (n_tok - off).min(crate::kernels::MMVQ_MAX_TOK);
                let xb = x_ptr + (off * w.cols * 4) as u64;
                let yb = y_ptr + (off * w.rows * 4) as u64;
                self.kernels.launch_quantize_q8_1(
                    &self.ctx,
                    xb,
                    self.xq_q8_1_buf.device_ptr(),
                    w.cols,
                    b,
                )?;
                let mut pw = w.ptr;
                let mut pxq = self.xq_q8_1_buf.device_ptr();
                let mut py = yb;
                let mut cr = w.rows as i32;
                let mut cc = w.cols as i32;
                let mut nt = b as i32;
                let mut args: [*mut c_void; 6] = [
                    (&mut pw as *mut CUdeviceptr).cast(),
                    (&mut pxq as *mut CUdeviceptr).cast(),
                    (&mut py as *mut CUdeviceptr).cast(),
                    (&mut cr as *mut i32).cast(),
                    (&mut cc as *mut i32).cast(),
                    (&mut nt as *mut i32).cast(),
                ];
                // SAFETY: Matches `mmvq_q8row_q8_1`.
                unsafe {
                    self.ctx.launch(
                        "mmvq_q8row_q8_1",
                        self.hk.mmvq_q8row,
                        ((w.rows as u32).div_ceil(4), 1, 1),
                        (32, 4, 1),
                        0,
                        &mut args,
                    )?;
                }
                off += b;
            }
            return Ok(());
        }

        // `S_K` (`0..=8`)
        if !g64_already_quantized {
            self.launch_quantize_g64(x_ptr, self.xq_g64_buf.device_ptr(), w.cols, n_tok)?;
        }
        let n_b64 = w.cols / 64;
        let mut off = 0usize;
        while off < n_tok {
            let b = (n_tok - off).min(crate::kernels::MMVQ_MAX_TOK);
            let xqb = self.xq_g64_buf.device_ptr() + (off * n_b64 * BLOCK_Q8_1_G64_BYTES) as u64;
            let yb = y_ptr + (off * w.rows * 4) as u64;
            let mut pw = w.ptr;
            let mut pxq = xqb;
            let mut py = yb;
            let mut sc = w.fmt;
            let mut cr = w.rows as i32;
            let mut cc = w.cols as i32;
            let mut nt = b as i32;
            let mut args: [*mut c_void; 7] = [
                (&mut pw as *mut CUdeviceptr).cast(),
                (&mut pxq as *mut CUdeviceptr).cast(),
                (&mut py as *mut CUdeviceptr).cast(),
                (&mut sc as *mut i32).cast(),
                (&mut cr as *mut i32).cast(),
                (&mut cc as *mut i32).cast(),
                (&mut nt as *mut i32).cast(),
            ];
            let grid_x = if b == 1 && w.cols >= 1024 {
                w.rows as u32
            } else {
                (w.rows as u32).div_ceil(4)
            };
            // SAFETY: Matches `mmvq_sk_q8_1_g64`.
            unsafe {
                self.ctx.launch(
                    "mmvq_sk_q8_1_g64",
                    self.hk.mmvq_sk_g64,
                    (grid_x, 1, 1),
                    (32, 4, 1),
                    0,
                    &mut args,
                )?;
            }
            off += b;
        }
        Ok(())
    }

    fn launch_hc_pre(
        &self,
        hc_w: &CudaHy4HcPre,
        cur_ptr: CUdeviceptr,
        post_ptr: Option<CUdeviceptr>,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let h = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * h;
        let mix_stride = if post_ptr.is_some() { 2 * hc } else { hc };

        self.kernels.launch_rms_norm(
            &self.ctx,
            self.x_hc_buf.device_ptr(),
            self.unit_ones_hc.device_ptr(),
            self.fl_buf.device_ptr(),
            hc_dim,
            n_tok,
            self.cfg.rms_eps,
        )?;
        self.launch_hy4_linear(
            &hc_w.hc_fn,
            self.fl_buf.device_ptr(),
            self.mix_buf.device_ptr(),
            n_tok,
            false,
        )?;

        let mut px = self.x_hc_buf.device_ptr();
        let mut pm = self.mix_buf.device_ptr();
        let mut psc = hc_w.hc_scale.ptr;
        let mut pb = hc_w.hc_base.ptr;
        let mut pc = cur_ptr;
        let mut pp = post_ptr.unwrap_or(0);
        let mut ch = h as i32;
        let mut chc = hc as i32;
        let mut cms = mix_stride as i32;
        let mut ceps = self.cfg.hc_eps;
        let mut cmag = self.cfg.hc_mag;
        let mut args: [*mut c_void; 11] = [
            (&mut px as *mut CUdeviceptr).cast(),
            (&mut pm as *mut CUdeviceptr).cast(),
            (&mut psc as *mut CUdeviceptr).cast(),
            (&mut pb as *mut CUdeviceptr).cast(),
            (&mut pc as *mut CUdeviceptr).cast(),
            (&mut pp as *mut CUdeviceptr).cast(),
            (&mut ch as *mut i32).cast(),
            (&mut chc as *mut i32).cast(),
            (&mut cms as *mut i32).cast(),
            (&mut ceps as *mut f32).cast(),
            (&mut cmag as *mut f32).cast(),
        ];
        // SAFETY: Matches `hy4_hc_apply_pre_f32`.
        unsafe {
            self.ctx.launch(
                "hy4_hc_apply_pre_f32",
                self.hk.hc_apply_pre,
                ((h as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    fn launch_hc_post(&self, sub_ptr: CUdeviceptr, n_tok: usize) -> Result<(), CudaError> {
        let h = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let mut px = self.x_hc_buf.device_ptr();
        let mut ps = sub_ptr;
        let mut pp = self.post_buf.device_ptr();
        let mut ch = h as i32;
        let mut chc = hc as i32;
        let mut args: [*mut c_void; 5] = [
            (&mut px as *mut CUdeviceptr).cast(),
            (&mut ps as *mut CUdeviceptr).cast(),
            (&mut pp as *mut CUdeviceptr).cast(),
            (&mut ch as *mut i32).cast(),
            (&mut chc as *mut i32).cast(),
        ];
        // SAFETY: Matches `hy4_hc_post_f32`.
        unsafe {
            self.ctx.launch(
                "hy4_hc_post_f32",
                self.hk.hc_post,
                (((hc * h) as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    fn encode_chunk(
        &self,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        compute_last_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), CudaError> {
        let h = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let ed = self.engram_cfg.engram_dim;

        // 1. Token embeddings -> `x_hc [n_tok, hc, h]`
        {
            let mut pemb = self.embed_tokens.ptr;
            let mut pids = self.token_ids_buf.device_ptr();
            let mut px = self.x_hc_buf.device_ptr();
            let mut sc = self.embed_tokens.fmt;
            let mut ch = h as i32;
            let mut chc = hc as i32;
            let mut cv = self.embed_tokens.rows as i32;
            let mut args: [*mut c_void; 7] = [
                (&mut pemb as *mut CUdeviceptr).cast(),
                (&mut pids as *mut CUdeviceptr).cast(),
                (&mut px as *mut CUdeviceptr).cast(),
                (&mut sc as *mut i32).cast(),
                (&mut ch as *mut i32).cast(),
                (&mut chc as *mut i32).cast(),
                (&mut cv as *mut i32).cast(),
            ];
            // SAFETY: Matches `hy4_embed_sk_f32`.
            unsafe {
                self.ctx.launch(
                    "hy4_embed_sk_f32",
                    self.hk.embed_sk,
                    ((h / 256) as u32, n_tok as u32, 1),
                    (32, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        for l in &self.layers {
            // Optional Engram module
            if let Some(ref eng) = l.engram {
                let r_ptr = self.engram_r_dev.device_ptr() + (eng.mod_idx * n_tok * ed * 4) as u64;
                let v_ptr = self.engram_valid_dev.device_ptr() + (eng.mod_idx * n_tok * 4) as u64;
                let is_sk = eng.val_proj.fmt < 100;
                if is_sk {
                    self.launch_quantize_g64(r_ptr, self.xq_g64_buf.device_ptr(), ed, n_tok)?;
                }
                for j in 0..(hc + 1) {
                    let (w, b) = if j < hc {
                        (&eng.key_projs[j], &eng.key_biases[j])
                    } else {
                        (&eng.val_proj, &eng.val_bias)
                    };
                    self.launch_hy4_linear(
                        w,
                        r_ptr,
                        self.engram_tmp_buf.device_ptr(),
                        n_tok,
                        is_sk && w.fmt < 100,
                    )?;
                    self.kernels.launch_add_bias(
                        &self.ctx,
                        self.engram_tmp_buf.device_ptr(),
                        b.ptr,
                        h,
                        n_tok,
                    )?;
                    // Copy `[n_tok, h]` into `engram_kv_buf[:, j, :]` (`dst_stride = (hc + 1) * h`)
                    self.kernels.launch_copy_2d_rows(
                        &self.ctx,
                        self.engram_kv_buf.device_ptr() + (j * h * 4) as u64,
                        self.engram_tmp_buf.device_ptr(),
                        h,
                        (hc + 1) * h,
                        h,
                        n_tok,
                    )?;
                }
                {
                    let mut px = self.x_hc_buf.device_ptr();
                    let mut pkv = self.engram_kv_buf.device_ptr();
                    let mut pnk = eng.norm_key.device_ptr();
                    let mut pnq = eng.norm_query.device_ptr();
                    let mut phs = eng.hc_scales.device_ptr();
                    let mut ppg = eng.pre_gamma.ptr;
                    let mut pav = v_ptr;
                    let mut ch = h as i32;
                    let mut chc = hc as i32;
                    let mut args: [*mut c_void; 9] = [
                        (&mut px as *mut CUdeviceptr).cast(),
                        (&mut pkv as *mut CUdeviceptr).cast(),
                        (&mut pnk as *mut CUdeviceptr).cast(),
                        (&mut pnq as *mut CUdeviceptr).cast(),
                        (&mut phs as *mut CUdeviceptr).cast(),
                        (&mut ppg as *mut CUdeviceptr).cast(),
                        (&mut pav as *mut CUdeviceptr).cast(),
                        (&mut ch as *mut i32).cast(),
                        (&mut chc as *mut i32).cast(),
                    ];
                    // SAFETY: Matches `hy4_engram_gate_add_f32`.
                    unsafe {
                        self.ctx.launch(
                            "hy4_engram_gate_add_f32",
                            self.hk.engram_gate_add,
                            (hc as u32, n_tok as u32, 1),
                            (256, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }
            }

            // 1. Attention HC Pre + Input RMSNorm
            self.launch_hc_pre(
                &l.hc_attn,
                self.cur_buf.device_ptr(),
                Some(self.post_buf.device_ptr()),
                n_tok,
            )?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.cur_buf.device_ptr(),
                l.input_ln.ptr,
                self.curn_buf.device_ptr(),
                h,
                n_tok,
                self.cfg.rms_eps,
            )?;

            // `P1-1`: Matrix-Absorbed MLA (`latent_kv_cache` `[cap, kv_lora + qk_rope]` FP16 — `28x` smaller KV!)
            self.encode_absorbed_mla(
                &l.q_a_proj,
                &l.q_a_ln,
                &l.q_b_proj,
                &l.kv_a_proj,
                &l.kv_a_ln,
                &l.kv_b_proj,
                &l.sinks,
                &l.linear_gate,
                &l.o_proj,
                l.latent_kv_cache.device_ptr(),
                l.n_head,
                l.is_swa,
                l.ring_cap,
                self.curn_buf.device_ptr(),
                self.cur_buf.device_ptr(),
                n_tok,
                start_pos,
                pos_dev_ptr,
            )?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.cur_buf.device_ptr(),
                l.post_proj_norm.ptr,
                self.sub_buf.device_ptr(),
                h,
                n_tok,
                self.cfg.rms_eps,
            )?;
            self.launch_hc_post(self.sub_buf.device_ptr(), n_tok)?;

            // 2. MLP / LatentMoE HC Pre + Post-Attn RMSNorm
            self.launch_hc_pre(
                &l.hc_mlp,
                self.cur_buf.device_ptr(),
                Some(self.post_buf.device_ptr()),
                n_tok,
            )?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.cur_buf.device_ptr(),
                l.post_attn_ln.ptr,
                self.curn_buf.device_ptr(),
                h,
                n_tok,
                self.cfg.rms_eps,
            )?;

            match &l.mlp {
                CudaHy4Mlp::Dense {
                    gate_proj,
                    up_proj,
                    down_proj,
                    post_fc2_norm,
                } => {
                    let ff = self.cfg.dense_ff;
                    let sk = gate_proj.fmt < 100 || up_proj.fmt < 100;
                    if sk {
                        self.launch_quantize_g64(
                            self.curn_buf.device_ptr(),
                            self.xq_g64_buf.device_ptr(),
                            h,
                            n_tok,
                        )?;
                    }
                    self.launch_hy4_linear(
                        gate_proj,
                        self.curn_buf.device_ptr(),
                        self.ff1_buf.device_ptr(),
                        n_tok,
                        sk && gate_proj.fmt < 100,
                    )?;
                    self.launch_hy4_linear(
                        up_proj,
                        self.curn_buf.device_ptr(),
                        self.ff2_buf.device_ptr(),
                        n_tok,
                        sk && up_proj.fmt < 100,
                    )?;
                    self.kernels.launch_swiglu(
                        &self.ctx,
                        self.ff1_buf.device_ptr(),
                        self.ff2_buf.device_ptr(),
                        n_tok * ff,
                    )?;
                    self.launch_hy4_linear(
                        down_proj,
                        self.ff1_buf.device_ptr(),
                        self.cur_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_rms_norm(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        post_fc2_norm.ptr,
                        self.sub_buf.device_ptr(),
                        h,
                        n_tok,
                        self.cfg.rms_eps,
                    )?;
                }
                CudaHy4Mlp::LatentMoe {
                    moe_idx,
                    gate,
                    e_bias,
                    routed_down,
                    routed_up,
                    sh_gate,
                    sh_up,
                    sh_down,
                    sh_ff,
                    combine_norm,
                } => {
                    let ne = self.cfg.n_experts;
                    let ku = self.cfg.top_k;
                    let lat = self.cfg.lat_dim;
                    self.launch_hy4_linear(
                        gate,
                        self.curn_buf.device_ptr(),
                        self.router_logits_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_moe_router_topk(
                        &self.ctx,
                        self.router_logits_buf.device_ptr(),
                        e_bias.ptr,
                        self.top_ids_buf.device_ptr(),
                        self.top_weights_buf.device_ptr(),
                        ne,
                        ku,
                        2, // Sigmoid
                        true,
                        1e-20,
                        self.cfg.routed_scale,
                        n_tok,
                    )?;

                    let curn_sk = routed_down.fmt < 100 || sh_gate.fmt < 100 || sh_up.fmt < 100;
                    if curn_sk {
                        self.launch_quantize_g64(
                            self.curn_buf.device_ptr(),
                            self.xq_g64_buf.device_ptr(),
                            h,
                            n_tok,
                        )?;
                    }
                    self.launch_hy4_linear(
                        routed_down,
                        self.curn_buf.device_ptr(),
                        self.lat_z_buf.device_ptr(),
                        n_tok,
                        curn_sk && routed_down.fmt < 100,
                    )?;
                    self.launch_hy4_linear(
                        sh_gate,
                        self.curn_buf.device_ptr(),
                        self.ff1_buf.device_ptr(),
                        n_tok,
                        curn_sk && sh_gate.fmt < 100,
                    )?;
                    self.launch_hy4_linear(
                        sh_up,
                        self.curn_buf.device_ptr(),
                        self.ff2_buf.device_ptr(),
                        n_tok,
                        curn_sk && sh_up.fmt < 100,
                    )?;

                    // `P0-3`: Fused dual `Gate + Up + clamped SwiGLU + SmoothQuant sq16` + `BlockQ8_1_G64` quantize!
                    self.launch_quantize_g64(
                        self.lat_z_buf.device_ptr(),
                        self.xq_g64_buf.device_ptr(),
                        lat,
                        n_tok,
                    )?;
                    let layer_dir_ptr =
                        self.expert_dir_dev.device_ptr() + (*moe_idx * ne * 32) as u64;
                    let rank_layer_ptr = self.rank_of_dev.device_ptr() + (*moe_idx * ne * 4) as u64;
                    {
                        let mut ps4 = self.sec4_arena.device_ptr();
                        let mut pld = layer_dir_ptr;
                        let mut prk = rank_layer_ptr;
                        let mut pxq = self.xq_g64_buf.device_ptr();
                        let mut pxf = self.lat_z_buf.device_ptr();
                        let mut pid = self.top_ids_buf.device_ptr();
                        let mut po = self.lat_act_f32_buf.device_ptr();
                        let mut cl = lat as i32;
                        let mut cc = lat as i32;
                        let mut csc = ku as i32;
                        let mut clim = self.cfg.swiglu_limit;
                        let mut args: [*mut c_void; 11] = [
                            (&mut ps4 as *mut CUdeviceptr).cast(),
                            (&mut pld as *mut CUdeviceptr).cast(),
                            (&mut prk as *mut CUdeviceptr).cast(),
                            (&mut pxq as *mut CUdeviceptr).cast(),
                            (&mut pxf as *mut CUdeviceptr).cast(),
                            (&mut pid as *mut CUdeviceptr).cast(),
                            (&mut po as *mut CUdeviceptr).cast(),
                            (&mut cl as *mut i32).cast(),
                            (&mut cc as *mut i32).cast(),
                            (&mut csc as *mut i32).cast(),
                            (&mut clim as *mut f32).cast(),
                        ];
                        // SAFETY: Matches `hy4_latmoe_dual_gu_swiglu_sq_f32`.
                        unsafe {
                            self.ctx.launch(
                                "hy4_latmoe_dual_gu_swiglu_sq_f32",
                                self.hk.latmoe_dual_gu_sq,
                                ((lat as u32).div_ceil(4), ku as u32, n_tok as u32),
                                (32, 4, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }
                    let n_rows = n_tok * ku;
                    self.launch_quantize_g64(
                        self.lat_act_f32_buf.device_ptr(),
                        self.slot_xq_g64_buf.device_ptr(),
                        lat,
                        n_rows,
                    )?;

                    // Parallel `ku`-slot `S_K` down + ordered slot reduction!
                    {
                        let mut ps4 = self.sec4_arena.device_ptr();
                        let mut pld = layer_dir_ptr;
                        let mut prk = rank_layer_ptr;
                        let mut pxq = self.slot_xq_g64_buf.device_ptr();
                        let mut pxf = self.lat_act_f32_buf.device_ptr();
                        let mut pid = self.top_ids_buf.device_ptr();
                        let mut ptw = self.top_weights_buf.device_ptr();
                        let mut po = self.lat_slots_buf.device_ptr();
                        let mut isd = 1i32;
                        let mut cr = lat as i32;
                        let mut cc = lat as i32;
                        let mut csc = ku as i32;
                        let mut args: [*mut c_void; 12] = [
                            (&mut ps4 as *mut CUdeviceptr).cast(),
                            (&mut pld as *mut CUdeviceptr).cast(),
                            (&mut prk as *mut CUdeviceptr).cast(),
                            (&mut pxq as *mut CUdeviceptr).cast(),
                            (&mut pxf as *mut CUdeviceptr).cast(),
                            (&mut pid as *mut CUdeviceptr).cast(),
                            (&mut ptw as *mut CUdeviceptr).cast(),
                            (&mut po as *mut CUdeviceptr).cast(),
                            (&mut isd as *mut i32).cast(),
                            (&mut cr as *mut i32).cast(),
                            (&mut cc as *mut i32).cast(),
                            (&mut csc as *mut i32).cast(),
                        ];
                        // SAFETY: Matches `hy4_latmoe_indexed_sk_q8_1_g64`.
                        unsafe {
                            self.ctx.launch(
                                "hy4_latmoe_indexed_sk_q8_1_g64",
                                self.hk.latmoe_indexed_sk,
                                ((lat as u32).div_ceil(4), ku as u32, n_tok as u32),
                                (32, 4, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }

                    self.kernels.launch_moe_reduce_slots(
                        &self.ctx,
                        self.lat_slots_buf.device_ptr(),
                        self.lat_z_buf.device_ptr(),
                        lat,
                        ku,
                        n_tok,
                    )?;
                    self.launch_hy4_linear(
                        routed_up,
                        self.lat_z_buf.device_ptr(),
                        self.cur_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;

                    // Shared expert + residual add + `combine_output_norm`
                    self.kernels.launch_swiglu(
                        &self.ctx,
                        self.ff1_buf.device_ptr(),
                        self.ff2_buf.device_ptr(),
                        n_tok * *sh_ff,
                    )?;
                    self.launch_hy4_linear(
                        sh_down,
                        self.ff1_buf.device_ptr(),
                        self.engram_tmp_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_add_scaled(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        self.engram_tmp_buf.device_ptr(),
                        1.0,
                        n_tok * h,
                    )?;
                    self.kernels.launch_rms_norm(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        combine_norm.ptr,
                        self.sub_buf.device_ptr(),
                        h,
                        n_tok,
                        self.cfg.rms_eps,
                    )?;
                }
            }

            self.launch_hc_post(self.sub_buf.device_ptr(), n_tok)?;
        }

        if compute_last_logits || self.dfly2.is_some() {
            let head_hc = CudaHy4HcPre {
                hc_fn: self.hc_head_fn.clone(),
                hc_scale: self.hc_head_scale.clone(),
                hc_base: self.hc_head_base.clone(),
            };
            self.launch_hc_pre(&head_hc, self.cur_buf.device_ptr(), None, n_tok)?;
            if self.dfly2.is_some() {
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.cur_buf.device_ptr(),
                    self.final_norm.ptr,
                    self.hf_verify_buf.device_ptr(),
                    h,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
            }
            if compute_last_logits {
                let normed_last = if self.dfly2.is_some() {
                    self.hf_verify_buf.device_ptr() + ((n_tok - 1) * h * 4) as u64
                } else {
                    let last_cur = self.cur_buf.device_ptr() + ((n_tok - 1) * h * 4) as u64;
                    self.kernels.launch_rms_norm(
                        &self.ctx,
                        last_cur,
                        self.final_norm.ptr,
                        self.curn_buf.device_ptr(),
                        h,
                        1,
                        self.cfg.rms_eps,
                    )?;
                    self.curn_buf.device_ptr()
                };
                self.launch_hy4_linear(
                    &self.lm_head,
                    normed_last,
                    self.logits_dev.device_ptr(),
                    1,
                    false,
                )?;
                if compute_argmax {
                    self.kernels.launch_argmax(
                        &self.ctx,
                        self.logits_dev.device_ptr(),
                        self.argmax_dev.device_ptr(),
                        self.cfg.vocab_size,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_absorbed_mla(
        &self,
        q_a_proj: &TrunkWeight,
        q_a_ln: &TrunkWeight,
        q_b_proj: &TrunkWeight,
        kv_a_proj: &TrunkWeight,
        kv_a_ln: &TrunkWeight,
        kv_b_proj: &TrunkWeight,
        sinks: &TrunkWeight,
        linear_gate: &TrunkWeight,
        o_proj: &TrunkWeight,
        latent_kv_ptr: CUdeviceptr,
        n_head: usize,
        is_swa: bool,
        ring_cap: usize,
        in_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
    ) -> Result<(), CudaError> {
        let h = self.cfg.n_embd;
        let curn_sk = q_a_proj.fmt < 100 || kv_a_proj.fmt < 100 || linear_gate.fmt < 100;
        if curn_sk {
            self.launch_quantize_g64(in_ptr, self.xq_g64_buf.device_ptr(), h, n_tok)?;
        }
        self.launch_hy4_linear(
            q_a_proj,
            in_ptr,
            self.qa_buf.device_ptr(),
            n_tok,
            curn_sk && q_a_proj.fmt < 100,
        )?;
        self.launch_hy4_linear(
            kv_a_proj,
            in_ptr,
            self.kva_buf.device_ptr(),
            n_tok,
            curn_sk && kv_a_proj.fmt < 100,
        )?;
        self.launch_hy4_linear(
            linear_gate,
            in_ptr,
            self.gate_buf.device_ptr(),
            n_tok,
            curn_sk && linear_gate.fmt < 100,
        )?;

        self.kernels.launch_rms_norm(
            &self.ctx,
            self.qa_buf.device_ptr(),
            q_a_ln.ptr,
            self.qr_buf.device_ptr(),
            self.cfg.q_lora,
            n_tok,
            self.cfg.rms_eps,
        )?;
        self.launch_hy4_linear(
            q_b_proj,
            self.qr_buf.device_ptr(),
            self.q_buf.device_ptr(),
            n_tok,
            false,
        )?;

        let inv_ptr = if is_swa {
            self.inv_freqs_swa.device_ptr()
        } else {
            self.inv_freqs_sparse.device_ptr()
        };
        let kv_lat = self.cfg.kv_lora + self.cfg.qk_rope;
        let q_row_floats = n_head * self.cfg.qk_head;
        let qa_row_floats = n_head * kv_lat;
        let ctx_row_floats = n_head * self.cfg.kv_lora;
        // When `ring_cap > 0 && n_tok > ring_cap` (small-window SWA prefill), step in `<= ring_cap` chunks
        // so `pos % ring_cap` never wraps onto a slot before earlier tokens in the batch attend to it!
        // Note: on `CudaGraphExec` decode (`n_tok == 1`) and real `Hy4-Flash` (`ring_cap == 1024 >= 64`),
        // `sub_step == n_tok` -> executes in a single pass with zero extra launches!
        let max_sub = if is_swa && ring_cap > 0 {
            ring_cap
        } else {
            n_tok
        };
        let mut t_off = 0usize;
        while t_off < n_tok {
            let sub_sp = start_pos + t_off;
            let step_cap = if is_swa && ring_cap > 0 {
                if sub_sp < ring_cap {
                    ring_cap - sub_sp
                } else {
                    ring_cap.saturating_sub(self.cfg.window) + 1
                }
            } else {
                max_sub
            };
            let sub_nt = (n_tok - t_off).min(step_cap.max(1));
            let q_sub = self.q_buf.device_ptr() + (t_off * q_row_floats * 4) as u64;
            let kva_sub = self.kva_buf.device_ptr() + (t_off * kv_lat * 4) as u64;
            let qa_sub = self.q_abs_buf.device_ptr() + (t_off * qa_row_floats * 4) as u64;
            let ctx_sub = self.ctx_lat_buf.device_ptr() + (t_off * ctx_row_floats * 4) as u64;

            // 1. RoPE on `q_pe` + RMSNorm(`kva[0..kv_lora]`) + RoPE(`k_pe`) -> write FP16 `latent_kv_cache`!
            {
                let mut pq = q_sub;
                let mut pkva = kva_sub;
                let mut plnw = kv_a_ln.ptr;
                let mut pif = inv_ptr;
                let mut plc = latent_kv_ptr;
                let mut nh = n_head as i32;
                let mut qkn = self.cfg.qk_nope as i32;
                let mut qkr = self.cfg.qk_rope as i32;
                let mut kvl = self.cfg.kv_lora as i32;
                let mut eps = self.cfg.rms_eps;
                let mut rc = ring_cap as i32;
                let mut sp = sub_sp as i32;
                let mut pp = pos_dev_ptr;
                let mut args: [*mut c_void; 13] = [
                    (&mut pq as *mut CUdeviceptr).cast(),
                    (&mut pkva as *mut CUdeviceptr).cast(),
                    (&mut plnw as *mut CUdeviceptr).cast(),
                    (&mut pif as *mut CUdeviceptr).cast(),
                    (&mut plc as *mut CUdeviceptr).cast(),
                    (&mut nh as *mut i32).cast(),
                    (&mut qkn as *mut i32).cast(),
                    (&mut qkr as *mut i32).cast(),
                    (&mut kvl as *mut i32).cast(),
                    (&mut eps as *mut f32).cast(),
                    (&mut rc as *mut i32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut pp as *mut CUdeviceptr).cast(),
                ];
                // SAFETY: Matches `hy4_mla_prep_latent_kv_f16`.
                unsafe {
                    self.ctx.launch(
                        "hy4_mla_prep_latent_kv_f16",
                        self.hk.mla_prep_latent_kv,
                        ((n_head + 1) as u32, sub_nt as u32, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            // 2. Absorb `W_kb` into `q_nope` -> `qa_sub [sub_nt, n_head, kv_lora + qk_rope]`
            {
                let mut pq = q_sub;
                let mut pw = kv_b_proj.ptr;
                let mut pqa = qa_sub;
                let mut sc = kv_b_proj.fmt;
                let mut nh = n_head as i32;
                let mut qkn = self.cfg.qk_nope as i32;
                let mut qkr = self.cfg.qk_rope as i32;
                let mut vh = self.cfg.v_head as i32;
                let mut kvl = self.cfg.kv_lora as i32;
                let mut nt = sub_nt as i32;
                let mut args: [*mut c_void; 10] = [
                    (&mut pq as *mut CUdeviceptr).cast(),
                    (&mut pw as *mut CUdeviceptr).cast(),
                    (&mut pqa as *mut CUdeviceptr).cast(),
                    (&mut sc as *mut i32).cast(),
                    (&mut nh as *mut i32).cast(),
                    (&mut qkn as *mut i32).cast(),
                    (&mut qkr as *mut i32).cast(),
                    (&mut vh as *mut i32).cast(),
                    (&mut kvl as *mut i32).cast(),
                    (&mut nt as *mut i32).cast(),
                ];
                // SAFETY: Matches `hy4_mla_absorb_q`.
                unsafe {
                    self.ctx.launch(
                        "hy4_mla_absorb_q",
                        self.hk.mla_absorb_q,
                        (
                            (self.cfg.kv_lora as u32).div_ceil(256),
                            n_head as u32,
                            (sub_nt as u32).div_ceil(4),
                        ),
                        (32, 4, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            // 3. MQA over shared `latent_kv_cache` with `learnable_sink_param` -> `ctx_sub [sub_nt, n_head, kv_lora]`
            {
                let mut pqa = qa_sub;
                let mut plc = latent_kv_ptr;
                let mut psink = sinks.ptr;
                let mut pctx = ctx_sub;
                let mut nh = n_head as i32;
                let mut kvl = self.cfg.kv_lora as i32;
                let mut qkr = self.cfg.qk_rope as i32;
                let mut qkh = self.cfg.qk_head as i32;
                let mut nswa = if is_swa { self.cfg.window as i32 } else { 0 };
                let mut rc = ring_cap as i32;
                let mut sp = sub_sp as i32;
                let mut pp = pos_dev_ptr;
                let mut args: [*mut c_void; 12] = [
                    (&mut pqa as *mut CUdeviceptr).cast(),
                    (&mut plc as *mut CUdeviceptr).cast(),
                    (&mut psink as *mut CUdeviceptr).cast(),
                    (&mut pctx as *mut CUdeviceptr).cast(),
                    (&mut nh as *mut i32).cast(),
                    (&mut kvl as *mut i32).cast(),
                    (&mut qkr as *mut i32).cast(),
                    (&mut qkh as *mut i32).cast(),
                    (&mut nswa as *mut i32).cast(),
                    (&mut rc as *mut i32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut pp as *mut CUdeviceptr).cast(),
                ];
                // SAFETY: Matches `hy4_mla_latent_attn_f16_with_sink`.
                unsafe {
                    self.ctx.launch(
                        "hy4_mla_latent_attn_f16_with_sink",
                        self.hk.mla_latent_attn,
                        (n_head as u32, sub_nt as u32, 1),
                        (32, 8, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            t_off += sub_nt;
        }

        // 4. Quantize `ctx_lat_buf` (for `S_K` or `Q8_ROW`) + fused `W_vb * ctx * sigmoid(gate)`!
        let n_ctx_rows = n_tok * n_head;
        if kv_b_proj.fmt < 100 {
            self.launch_quantize_g64(
                self.ctx_lat_buf.device_ptr(),
                self.xq_g64_buf.device_ptr(),
                self.cfg.kv_lora,
                n_ctx_rows,
            )?;
        } else if kv_b_proj.fmt == FMT_Q8ROW {
            self.kernels.launch_quantize_q8_1(
                &self.ctx,
                self.ctx_lat_buf.device_ptr(),
                self.xq_q8_1_buf.device_ptr(),
                self.cfg.kv_lora,
                n_ctx_rows,
            )?;
        }
        {
            let mut pw = kv_b_proj.ptr;
            let mut pg64 = self.xq_g64_buf.device_ptr();
            let mut pq8 = self.xq_q8_1_buf.device_ptr();
            let mut pf32 = self.ctx_lat_buf.device_ptr();
            let mut pgate = self.gate_buf.device_ptr();
            let mut po = self.attn_buf.device_ptr();
            let mut sc = kv_b_proj.fmt;
            let mut nh = n_head as i32;
            let mut qkn = self.cfg.qk_nope as i32;
            let mut vh = self.cfg.v_head as i32;
            let mut kvl = self.cfg.kv_lora as i32;
            let mut args: [*mut c_void; 11] = [
                (&mut pw as *mut CUdeviceptr).cast(),
                (&mut pg64 as *mut CUdeviceptr).cast(),
                (&mut pq8 as *mut CUdeviceptr).cast(),
                (&mut pf32 as *mut CUdeviceptr).cast(),
                (&mut pgate as *mut CUdeviceptr).cast(),
                (&mut po as *mut CUdeviceptr).cast(),
                (&mut sc as *mut i32).cast(),
                (&mut nh as *mut i32).cast(),
                (&mut qkn as *mut i32).cast(),
                (&mut vh as *mut i32).cast(),
                (&mut kvl as *mut i32).cast(),
            ];
            // SAFETY: Matches `hy4_mla_vup_gate`.
            unsafe {
                self.ctx.launch(
                    "hy4_mla_vup_gate",
                    self.hk.mla_vup_gate,
                    (
                        (self.cfg.v_head as u32).div_ceil(4),
                        n_head as u32,
                        n_tok as u32,
                    ),
                    (32, 4, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        self.launch_hy4_linear(o_proj, self.attn_buf.device_ptr(), out_ptr, n_tok, false)
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
        self.stage_engram_to_gpu(&[token_id], pos)?;
        let slot = if !compute_logits {
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
                self.encode_chunk(1, 0, pos_ptr, compute_logits, compute_argmax)
            })?;
            self.decode_graphs[slot] = Some(g);
        }
        self.decode_graphs[slot]
            .as_ref()
            .expect("captured hy4 decode graph")
            .launch()?;
        if let Some(ref dfly) = self.dfly2 {
            dfly.hprev_f32
                .copy_from_device(&self.hf_verify_buf, self.cfg.n_embd * 4)?;
        }
        Ok(())
    }

    /// Evaluates `[x_tok, drafts...]` (`w = 1 + drafts.len()`) on the `Hy4` target model, overwriting
    /// any speculative entries written into `latent_kv_cache[pos .. pos + w - 1]` and updating `dfly.hprev_f32`!
    pub fn verify_window_greedy(
        &mut self,
        x_tok: u32,
        drafts: &[u32],
        pos: usize,
        out_committed: &mut Vec<u32>,
    ) -> Result<(usize, usize), CudaError> {
        if drafts.is_empty() {
            let next = self.forward_token_greedy(x_tok, pos)?;
            out_committed.clear();
            out_committed.push(next);
            return Ok((1, 0));
        }
        let d_len = drafts.len().min(MAX_SPEC_WINDOW);
        let w = 1 + d_len;
        if pos + w > self.max_ctx {
            let next = self.forward_token_greedy(x_tok, pos)?;
            out_committed.clear();
            out_committed.push(next);
            return Ok((1, 0));
        }

        let mut win = [0u32; MAX_SPEC_WINDOW + 1];
        win[0] = x_tok;
        win[1..w].copy_from_slice(&drafts[..d_len]);
        self.pinned_ids.write_u32_at(0, &win[..w]);
        self.token_ids_buf
            .copy_from_pinned_async(0, &self.pinned_ids, 0, w * 4)?;
        self.stage_engram_to_gpu(&win[..w], pos)?;
        self.encode_chunk(w, pos, 0, false, false)?;

        let h = self.cfg.n_embd;
        if self.dfly2.is_none() {
            let head_hc = CudaHy4HcPre {
                hc_fn: self.hc_head_fn.clone(),
                hc_scale: self.hc_head_scale.clone(),
                hc_base: self.hc_head_base.clone(),
            };
            self.launch_hc_pre(&head_hc, self.cur_buf.device_ptr(), None, w)?;
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.cur_buf.device_ptr(),
                self.final_norm.ptr,
                self.hf_verify_buf.device_ptr(),
                h,
                w,
                self.cfg.rms_eps,
            )?;
        }

        self.launch_hy4_linear(
            &self.lm_head,
            self.hf_verify_buf.device_ptr(),
            self.logits_dev.device_ptr(),
            w,
            false,
        )?;
        self.kernels.launch_argmax_rows(
            &self.ctx,
            self.logits_dev.device_ptr(),
            self.argmax_dev.device_ptr(),
            self.cfg.vocab_size,
            w,
        )?;

        let mut picks = [0u32; MAX_SPEC_WINDOW + 1];
        // SAFETY: `[u32]` is POD; `w <= MAX_SPEC_WINDOW + 1`.
        let pick_bytes =
            unsafe { core::slice::from_raw_parts_mut(picks.as_mut_ptr().cast::<u8>(), w * 4) };
        self.argmax_dev.copy_to_bytes(pick_bytes)?;

        let mut accepted = 0usize;
        while accepted < d_len && picks[accepted] == drafts[accepted] {
            accepted += 1;
        }
        let committed = accepted + 1;
        out_committed.clear();
        out_committed.extend_from_slice(&picks[..committed]);

        if committed < w {
            for p in (pos + committed)..(pos + w) {
                if p < self.seq_tokens.len() {
                    self.seq_tokens[p] = self.engram_cfg.pad_id as u32;
                }
            }
        }

        if let Some(ref dfly) = self.dfly2 {
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                dfly.hprev_f32.device_ptr(),
                self.hf_verify_buf.device_ptr() + (accepted * h * 4) as u64,
                h,
                h,
                h,
                1,
            )?;
        }
        Ok((committed, accepted))
    }

    /// Runs a full `P0-1` built-in 5-layer `DFly2` parallel-masked + chained-correction speculative step!
    /// Dynamically truncates the draft prefix when cumulative `confidence_head` probability drops `< conf_thresh`.
    pub fn spec_step_dfly2_greedy(
        &mut self,
        x_tok: u32,
        pos: usize,
        draft_k: usize,
        conf_thresh: f32,
        out_committed: &mut Vec<u32>,
    ) -> Result<(usize, usize), CudaError> {
        let k_max = draft_k.min(MAX_SPEC_WINDOW);
        if k_max == 0 || self.dfly2.is_none() || pos + 1 + k_max > self.max_ctx {
            let next = self.forward_token_greedy(x_tok, pos)?;
            out_committed.clear();
            out_committed.push(next);
            return Ok((0, 0));
        }

        let b_len = 1 + k_max;
        let h = self.cfg.n_embd;
        let mut drafts = [0u32; MAX_SPEC_WINDOW];
        let mut eff_k = k_max;

        {
            let dfly = self.dfly2.as_ref().expect("checked is_some");
            // 1. Upload `[x_tok, mask, ..., mask]` (`B = K + 1` tokens) + seed `chain_ids_dev[0] = x_tok`
            let mut bids = [self.cfg.dfly_mask_id; MAX_SPEC_WINDOW + 1];
            bids[0] = x_tok;
            self.pinned_ids.write_u32_at(0, &bids[..b_len]);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, b_len * 4)?;
            dfly.chain_ids_dev
                .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;

            // 2. Embed `B` rows (`hc = 1`) into `dfly.h_buf [B, h]`
            {
                let mut pemb = self.embed_tokens.ptr;
                let mut pids = self.token_ids_buf.device_ptr();
                let mut px = dfly.h_buf.device_ptr();
                let mut sc = self.embed_tokens.fmt;
                let mut ch = h as i32;
                let mut chc = 1i32;
                let mut cv = self.embed_tokens.rows as i32;
                let mut args: [*mut c_void; 7] = [
                    (&mut pemb as *mut CUdeviceptr).cast(),
                    (&mut pids as *mut CUdeviceptr).cast(),
                    (&mut px as *mut CUdeviceptr).cast(),
                    (&mut sc as *mut i32).cast(),
                    (&mut ch as *mut i32).cast(),
                    (&mut chc as *mut i32).cast(),
                    (&mut cv as *mut i32).cast(),
                ];
                // SAFETY: Matches `hy4_embed_sk_f32`.
                unsafe {
                    self.ctx.launch(
                        "hy4_embed_sk_f32",
                        self.hk.embed_sk,
                        ((h / 256) as u32, b_len as u32, 1),
                        (32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            // 3. Slot 0 `mtp_fusion`: `H[0] += gate * eh_proj([rms_norm(H[0], enorm) ; rms_norm(hprev, hnorm)])`
            self.kernels.launch_mtp_rms_norm_cat2(
                &self.ctx,
                dfly.h_buf.device_ptr(),
                dfly.fusion_enorm.ptr,
                dfly.hprev_f32.device_ptr(),
                dfly.fusion_hnorm.ptr,
                dfly.hcin_buf.device_ptr(),
                h,
                self.cfg.rms_eps,
                1,
            )?;
            self.launch_hy4_linear(
                &dfly.fusion_eh_proj,
                dfly.hcin_buf.device_ptr(),
                self.cur_buf.device_ptr(),
                1,
                false,
            )?;
            self.kernels.launch_elemwise_mul_add(
                &self.ctx,
                dfly.h_buf.device_ptr(),
                dfly.fusion_gate.ptr,
                self.cur_buf.device_ptr(),
                h,
            )?;

            // 4. Run 5 `DFly2` sparse MLA + dense SwiGLU layers sharing target `latent_kv_cache[L]`!
            for dl in &dfly.layers {
                let target_l = &self.layers[dl.target_layer_idx];
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    dfly.h_buf.device_ptr(),
                    dl.input_ln.ptr,
                    self.curn_buf.device_ptr(),
                    h,
                    b_len,
                    self.cfg.rms_eps,
                )?;
                self.encode_absorbed_mla(
                    &dl.q_a_proj,
                    &dl.q_a_ln,
                    &dl.q_b_proj,
                    &dl.kv_a_proj,
                    &dl.kv_a_ln,
                    &dl.kv_b_proj,
                    &dl.sinks,
                    &dl.linear_gate,
                    &dl.o_proj,
                    target_l.latent_kv_cache.device_ptr(),
                    self.cfg.heads_sparse,
                    false,
                    0,
                    self.curn_buf.device_ptr(),
                    self.cur_buf.device_ptr(),
                    b_len,
                    pos,
                    0,
                )?;
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.cur_buf.device_ptr(),
                    dl.post_proj_norm.ptr,
                    self.sub_buf.device_ptr(),
                    h,
                    b_len,
                    self.cfg.rms_eps,
                )?;
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    dfly.h_buf.device_ptr(),
                    self.sub_buf.device_ptr(),
                    1.0,
                    b_len * h,
                )?;

                self.kernels.launch_rms_norm(
                    &self.ctx,
                    dfly.h_buf.device_ptr(),
                    dl.post_attn_ln.ptr,
                    self.curn_buf.device_ptr(),
                    h,
                    b_len,
                    self.cfg.rms_eps,
                )?;
                let sk = dl.gate_proj.fmt < 100 || dl.up_proj.fmt < 100;
                if sk {
                    self.launch_quantize_g64(
                        self.curn_buf.device_ptr(),
                        self.xq_g64_buf.device_ptr(),
                        h,
                        b_len,
                    )?;
                }
                self.launch_hy4_linear(
                    &dl.gate_proj,
                    self.curn_buf.device_ptr(),
                    self.ff1_buf.device_ptr(),
                    b_len,
                    sk && dl.gate_proj.fmt < 100,
                )?;
                self.launch_hy4_linear(
                    &dl.up_proj,
                    self.curn_buf.device_ptr(),
                    self.ff2_buf.device_ptr(),
                    b_len,
                    sk && dl.up_proj.fmt < 100,
                )?;
                self.kernels.launch_swiglu(
                    &self.ctx,
                    self.ff1_buf.device_ptr(),
                    self.ff2_buf.device_ptr(),
                    b_len * dl.mlp_ff,
                )?;
                self.launch_hy4_linear(
                    &dl.down_proj,
                    self.ff1_buf.device_ptr(),
                    self.cur_buf.device_ptr(),
                    b_len,
                    false,
                )?;
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    dfly.h_buf.device_ptr(),
                    self.cur_buf.device_ptr(),
                    1.0,
                    b_len * h,
                )?;
            }

            // 5. Final norm `Hf = rms_norm(H, dfly.final_norm)`
            self.kernels.launch_rms_norm(
                &self.ctx,
                dfly.h_buf.device_ptr(),
                dfly.final_norm.ptr,
                dfly.hf_buf.device_ptr(),
                h,
                b_len,
                self.cfg.rms_eps,
            )?;

            // 6. Chained `hidden_correction` + `confidence_head` + `lm_head` argmax across `j = 1..=K` (`0` host syncs!)
            for j in 1..=k_max {
                {
                    let mut pemb = self.embed_tokens.ptr;
                    let mut pids = dfly.chain_ids_dev.device_ptr() + ((j - 1) * 4) as u64;
                    let mut px = dfly.ec_buf.device_ptr();
                    let mut sc = self.embed_tokens.fmt;
                    let mut ch = h as i32;
                    let mut chc = 1i32;
                    let mut cv = self.embed_tokens.rows as i32;
                    let mut args: [*mut c_void; 7] = [
                        (&mut pemb as *mut CUdeviceptr).cast(),
                        (&mut pids as *mut CUdeviceptr).cast(),
                        (&mut px as *mut CUdeviceptr).cast(),
                        (&mut sc as *mut i32).cast(),
                        (&mut ch as *mut i32).cast(),
                        (&mut chc as *mut i32).cast(),
                        (&mut cv as *mut i32).cast(),
                    ];
                    // SAFETY: Matches `hy4_embed_sk_f32`.
                    unsafe {
                        self.ctx.launch(
                            "hy4_embed_sk_f32",
                            self.hk.embed_sk,
                            ((h / 256) as u32, 1, 1),
                            (32, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }

                let hf_j = dfly.hf_buf.device_ptr() + (j * h * 4) as u64;
                self.kernels.launch_mtp_rms_norm_cat2(
                    &self.ctx,
                    hf_j,
                    dfly.corr_hidden_norm.ptr,
                    dfly.ec_buf.device_ptr(),
                    dfly.corr_embed_norm.ptr,
                    dfly.hcin_buf.device_ptr(),
                    h,
                    self.cfg.rms_eps,
                    1,
                )?;
                let sk = dfly.corr_gate_proj.fmt < 100 || dfly.corr_up_proj.fmt < 100;
                if sk {
                    self.launch_quantize_g64(
                        dfly.hcin_buf.device_ptr(),
                        self.xq_g64_buf.device_ptr(),
                        2 * h,
                        1,
                    )?;
                }
                self.launch_hy4_linear(
                    &dfly.corr_gate_proj,
                    dfly.hcin_buf.device_ptr(),
                    self.ff1_buf.device_ptr(),
                    1,
                    sk && dfly.corr_gate_proj.fmt < 100,
                )?;
                self.launch_hy4_linear(
                    &dfly.corr_up_proj,
                    dfly.hcin_buf.device_ptr(),
                    self.ff2_buf.device_ptr(),
                    1,
                    sk && dfly.corr_up_proj.fmt < 100,
                )?;
                self.kernels.launch_swiglu(
                    &self.ctx,
                    self.ff1_buf.device_ptr(),
                    self.ff2_buf.device_ptr(),
                    dfly.corr_ff,
                )?;
                self.launch_hy4_linear(
                    &dfly.corr_down_proj,
                    self.ff1_buf.device_ptr(),
                    dfly.hcorr_buf.device_ptr(),
                    1,
                    false,
                )?;
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    dfly.hcorr_buf.device_ptr(),
                    hf_j,
                    1.0,
                    h,
                )?;

                self.launch_hy4_linear(
                    &dfly.conf_proj,
                    dfly.hcorr_buf.device_ptr(),
                    dfly.conf_logit_buf.device_ptr(),
                    1,
                    false,
                )?;
                self.kernels.launch_dfly_bias_sigmoid_slot(
                    &self.ctx,
                    dfly.conf_logit_buf.device_ptr(),
                    dfly.conf_bias.ptr,
                    dfly.conf_probs_dev.device_ptr(),
                    j,
                )?;

                self.launch_hy4_linear(
                    &self.lm_head,
                    dfly.hcorr_buf.device_ptr(),
                    self.logits_dev.device_ptr(),
                    1,
                    false,
                )?;
                self.kernels.launch_argmax(
                    &self.ctx,
                    self.logits_dev.device_ptr(),
                    dfly.chain_ids_dev.device_ptr() + (j * 4) as u64,
                    self.cfg.vocab_size,
                )?;
            }

            // Single 40-byte readback of `chain_ids[0..=K]` + `conf_probs[0..=K]`!
            let mut chain_host = [0u32; MAX_SPEC_WINDOW + 1];
            let mut conf_host = [0.0f32; MAX_SPEC_WINDOW + 1];
            // SAFETY: `[u32]` and `[f32]` are POD; `b_len <= MAX_SPEC_WINDOW + 1`.
            unsafe {
                dfly.chain_ids_dev
                    .copy_to_bytes(core::slice::from_raw_parts_mut(
                        chain_host.as_mut_ptr().cast::<u8>(),
                        b_len * 4,
                    ))?;
                dfly.conf_probs_dev
                    .copy_to_bytes(core::slice::from_raw_parts_mut(
                        conf_host.as_mut_ptr().cast::<u8>(),
                        b_len * 4,
                    ))?;
            }
            drafts[..k_max].copy_from_slice(&chain_host[1..b_len]);

            if conf_thresh > 0.0 {
                let mut cum = 1.0f32;
                let mut keep = 0usize;
                for (j, &cp) in conf_host.iter().enumerate().take(k_max + 1).skip(1) {
                    cum *= cp;
                    if keep == 0 || cum >= conf_thresh {
                        keep = j;
                    } else {
                        break;
                    }
                }
                eff_k = keep.max(1);
            }
        }

        // 7. Verify `[x_tok, drafts[0..eff_k]]` on the 42-layer target `Hy4` trunk; return `(drafted, accepted)`.
        let (_committed, accepted) =
            self.verify_window_greedy(x_tok, &drafts[..eff_k], pos, out_committed)?;
        Ok((eff_k, accepted))
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        if pos >= self.max_ctx || (token_id as usize) >= self.cfg.vocab_size {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid token_id={token_id} or pos={pos}"
            )));
        }
        self.run_decode_step(token_id, pos, compute_logits, false)?;
        if compute_logits {
            self.logits_dev.copy_to_f32(&mut self.logits_host)?;
        } else {
            self.ctx.synchronize()?;
        }
        Ok(&self.logits_host)
    }

    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, CudaError> {
        if pos >= self.max_ctx || (token_id as usize) >= self.cfg.vocab_size {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid token_id={token_id} or pos={pos}"
            )));
        }
        self.run_decode_step(token_id, pos, true, true)?;
        self.argmax_dev.read_u32()
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
        if start_pos + token_ids.len() > self.max_ctx {
            return Err(CudaError::InvalidDimensions(format!(
                "batch end {} exceeds max_ctx {}",
                start_pos + token_ids.len(),
                self.max_ctx
            )));
        }
        let mut off = 0usize;
        while off < token_ids.len() {
            let chunk = &token_ids[off..(off + MAX_CUDA_BATCH).min(token_ids.len())];
            let is_last = off + chunk.len() == token_ids.len();
            self.pinned_ids.write_u32_at(0, chunk);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, chunk.len() * 4)?;
            self.stage_engram_to_gpu(chunk, start_pos + off)?;
            self.encode_chunk(
                chunk.len(),
                start_pos + off,
                0,
                compute_last_logits && is_last,
                false,
            )?;
            if is_last && let Some(ref dfly) = self.dfly2 {
                let h = self.cfg.n_embd;
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    dfly.hprev_f32.device_ptr(),
                    self.hf_verify_buf.device_ptr() + ((chunk.len() - 1) * h * 4) as u64,
                    h,
                    h,
                    h,
                    1,
                )?;
            }
            off += chunk.len();
        }
        if compute_last_logits {
            self.logits_dev.copy_to_f32(&mut self.logits_host)?;
        } else {
            self.ctx.synchronize()?;
        }
        Ok(&self.logits_host)
    }

    pub fn forward_embed(
        &mut self,
        _embed: &[f32],
        _pos: usize,
        _compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        Err(CudaError::UnsupportedArch(
            "forward_embed not yet implemented for hy4flash".to_string(),
        ))
    }

    pub fn profile(
        &mut self,
        n_runs: usize,
        ctx_pos: usize,
    ) -> Result<CudaProfileReport, CudaError> {
        let runs = n_runs.max(1);
        let pos = ctx_pos.min(self.max_ctx.saturating_sub(1));
        let sample_tok = self.engram_cfg.pad_id.min(self.cfg.vocab_size - 1) as u32;
        let _ = self.forward_token_greedy(sample_tok, pos)?;

        let ev0 = CudaEvent::new(&self.ctx)?;
        let ev1 = CudaEvent::new(&self.ctx)?;
        ev0.record()?;
        for _ in 0..runs {
            self.run_decode_step(sample_tok, pos, true, true)?;
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
                name: "full_hy4_decode_step_gpu",
                ms_per_tok: total_ms,
                pct: 100.0,
            }],
        })
    }
}
