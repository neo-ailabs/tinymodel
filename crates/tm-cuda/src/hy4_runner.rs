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
use tm_models::hy4::{
    FMT_F32, FMT_Q8ROW, Hy4Config, Hy4EngramCfg, Hy4EngramReader, Hy4Manifest, Hy4TensorInfo,
    discover_sibling_engram_strata, read_expert_directory, read_strata_header, read_strata_section,
};

use crate::CudaError;
use crate::driver::{CUdeviceptr, CUfunction, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat};
use crate::llama_runner::{CudaProfileReport, CudaStageStat, MAX_CUDA_BATCH};

const BLOCK_Q8_1_G64_BYTES: usize = 72;
const STREAM_CHUNK_BYTES: usize = 64 * 1024 * 1024;

struct Hy4CudaKernels {
    quantize_g64: CUfunction,
    mmvq_sk_g64: CUfunction,
    mmvq_q8row: CUfunction,
    embed_sk: CUfunction,
    hc_apply_pre: CUfunction,
    hc_post: CUfunction,
    engram_gate_add: CUfunction,
    mla_prep_rope: CUfunction,
    pack_mla_kv: CUfunction,
    attn_with_sink: CUfunction,
    latmoe_indexed_sk: CUfunction,
    latmoe_swiglu_sq: CUfunction,
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
            mla_prep_rope: ctx.get_function("hy4_mla_prep_rope_f32")?,
            pack_mla_kv: ctx.get_function("hy4_pack_mla_kv_f16")?,
            attn_with_sink: ctx.get_function("hy4_attn_f16_kv_with_sink")?,
            latmoe_indexed_sk: ctx.get_function("hy4_latmoe_indexed_sk_q8_1_g64")?,
            latmoe_swiglu_sq: ctx.get_function("hy4_latmoe_swiglu_sq_quantize_g64")?,
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
    k_cache: CudaBuffer,
    v_cache: CudaBuffer,
    hc_mlp: CudaHy4HcPre,
    post_attn_ln: TrunkWeight,
    mlp: CudaHy4Mlp,
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

    // Host Engram state
    engram_reader: Hy4EngramReader,
    engram_ln_weights: Vec<Vec<f32>>,
    seq_tokens: Vec<u32>,
    engram_r_host: Vec<f32>,
    engram_valid_host: Vec<f32>,

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
    sub_buf: CudaBuffer,
    qa_buf: CudaBuffer,
    qr_buf: CudaBuffer,
    q_buf: CudaBuffer,
    kva_buf: CudaBuffer,
    kvc_buf: CudaBuffer,
    kpe_buf: CudaBuffer,
    kvx_buf: CudaBuffer,
    attn_buf: CudaBuffer,
    gate_buf: CudaBuffer,
    ff1_buf: CudaBuffer,
    ff2_buf: CudaBuffer,
    router_logits_buf: CudaBuffer,
    top_ids_buf: CudaBuffer,
    top_weights_buf: CudaBuffer,
    lat_z_buf: CudaBuffer,
    lat_gu_buf: CudaBuffer,
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
    staging: &mut [u8],
) -> Result<CudaBuffer, CudaError> {
    if byte_length == 0 {
        return ctx.alloc_zeroed(64);
    }
    let buf = ctx.alloc_uninit(byte_length as usize)?;
    f.seek(SeekFrom::Start(file_offset)).map_err(|e| {
        CudaError::InvalidDimensions(format!("seek .strata offset {file_offset}: {e}"))
    })?;
    let mut copied = 0usize;
    let total = byte_length as usize;
    while copied < total {
        let want = (total - copied).min(staging.len());
        f.read_exact(&mut staging[..want])
            .map_err(|e| CudaError::InvalidDimensions(format!("read .strata at +{copied}: {e}")))?;
        buf.copy_from_bytes_at(copied, &staging[..want])?;
        copied += want;
    }
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

        let mut kv_total_bytes = 0usize;
        for il in 0..cfg.n_layers {
            let nh = if cfg.is_swa[il] {
                cfg.heads_swa
            } else {
                cfg.heads_sparse
            };
            let cap = if cfg.is_swa[il] && cfg.window > 0 {
                (cfg.window + MAX_CUDA_BATCH).min(max_ctx)
            } else {
                max_ctx
            };
            kv_total_bytes += cap * nh * (cfg.qk_head + cfg.v_head) * 2;
        }

        let (free_vram, _) = ctx.mem_info()?;
        let needed_bytes = model_weight_bytes + kv_total_bytes + 256 * 1024 * 1024;
        if needed_bytes > free_vram {
            return Err(CudaError::OutOfDeviceMemory {
                needed_mib: needed_bytes as f64 / (1024.0 * 1024.0),
                free_mib: free_vram as f64 / (1024.0 * 1024.0),
                max_ctx,
                device: ctx.info().name.clone(),
            });
        }

        // Stream Section 1 (trunk) and Section 4 (expert bodies) in 64 MiB chunks directly to GPU!
        let mut staging = vec![0u8; STREAM_CHUNK_BYTES.min(sec1_bytes.max(sec4_bytes).max(4096))];
        let trunk_arena = Arc::new(stream_file_region_to_vram(
            &ctx,
            &mut f,
            hdr.sections[1].file_offset,
            hdr.sections[1].byte_length,
            &mut staging,
        )?);
        let sec4_arena = stream_file_region_to_vram(
            &ctx,
            &mut f,
            hdr.sections[4].file_offset,
            hdr.sections[4].byte_length,
            &mut staging,
        )?;
        drop(staging);

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
                k_cache: ctx.alloc_zeroed(cap * nh * cfg.qk_head * 2)?,
                v_cache: ctx.alloc_zeroed(cap * nh * cfg.v_head * 2)?,
                hc_mlp,
                post_attn_ln: tw(&format!("{lp}.post_attention_layernorm.weight"))?,
                mlp,
            });
        }

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
        let max_ex = max_nh * (cfg.qk_nope + cfg.v_head);
        let max_ff = cfg.dense_ff.max(max_sh_ff).max(h);
        let max_cols = [
            h,
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
        let sub_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * h * 4)?;
        let qa_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.q_lora * 4)?;
        let qr_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.q_lora * 4)?;
        let q_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_qd * 4)?;
        let kva_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * (cfg.kv_lora + cfg.qk_rope) * 4)?;
        let kvc_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.kv_lora * 4)?;
        let kpe_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.qk_rope * 4)?;
        let kvx_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_ex * 4)?;
        let attn_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_vd * 4)?;
        let gate_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_vd * 4)?;
        let ff1_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_ff * 4)?;
        let ff2_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_ff * 4)?;
        let router_logits_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_experts.max(1) * 4)?;
        let top_ids_buf = ctx.alloc_zeroed(rows_slots * 4)?;
        let top_weights_buf = ctx.alloc_zeroed(rows_slots * 4)?;
        let lat = cfg.lat_dim.max(64);
        let lat_z_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * lat * 4)?;
        let lat_gu_buf = ctx.alloc_zeroed(rows_slots * 2 * lat * 4)?;
        let lat_act_f32_buf = ctx.alloc_zeroed(rows_slots * lat * 4)?;
        let lat_slots_buf = ctx.alloc_zeroed(rows_slots * lat * 4)?;
        let xq_q8_1_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * (max_cols / 32) * BLOCK_Q8_1_BYTES)?;
        let xq_g64_buf =
            ctx.alloc_zeroed(MAX_CUDA_BATCH * (max_cols / 64) * BLOCK_Q8_1_G64_BYTES)?;
        let slot_xq_g64_buf =
            ctx.alloc_zeroed(rows_slots * (lat.div_ceil(64)) * BLOCK_Q8_1_G64_BYTES)?;
        let logits_dev = ctx.alloc_zeroed(cfg.vocab_size * 4)?;
        let argmax_dev = ctx.alloc_zeroed(4)?;
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
                engram_reader,
                engram_ln_weights,
                seq_tokens: vec![ecfg.pad_id as u32; max_ctx.max(1)],
                engram_r_host: vec![0.0; n_mod * MAX_CUDA_BATCH * ed],
                engram_valid_host: vec![0.0; n_mod * MAX_CUDA_BATCH],
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
                sub_buf,
                qa_buf,
                qr_buf,
                q_buf,
                kva_buf,
                kvc_buf,
                kpe_buf,
                kvx_buf,
                attn_buf,
                gate_buf,
                ff1_buf,
                ff2_buf,
                router_logits_buf,
                top_ids_buf,
                top_weights_buf,
                lat_z_buf,
                lat_gu_buf,
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

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        for l in &self.layers {
            l.k_cache.zero()?;
            l.v_cache.zero()?;
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
        for m_idx in 0..n_mod {
            let r_slice = &mut self.engram_r_host[m_idx * n_tok * ed..(m_idx + 1) * n_tok * ed];
            let v_slice = &mut self.engram_valid_host[m_idx * n_tok..(m_idx + 1) * n_tok];
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
        // Upload only the active `n_mod * n_tok` slices (`24 KiB` for decode instead of `1.57 MiB`!).
        let r_len = n_mod * n_tok * ed;
        let v_len = n_mod * n_tok;
        // SAFETY: `[f32]` is plain old data.
        let r_bytes = unsafe {
            core::slice::from_raw_parts(self.engram_r_host.as_ptr().cast::<u8>(), r_len * 4)
        };
        // SAFETY: `[f32]` is plain old data.
        let v_bytes = unsafe {
            core::slice::from_raw_parts(self.engram_valid_host.as_ptr().cast::<u8>(), v_len * 4)
        };
        self.engram_r_dev.copy_from_bytes_at(0, r_bytes)?;
        self.engram_valid_dev.copy_from_bytes_at(0, v_bytes)
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

            // Non-absorbed MLA (`q_a_proj` -> `q_a_ln` -> `q_b_proj`; `kv_a_proj` -> RoPE + `kv_a_ln` -> `kv_b_proj`)
            let curn_sk = l.q_a_proj.fmt < 100 || l.kv_a_proj.fmt < 100 || l.linear_gate.fmt < 100;
            if curn_sk {
                self.launch_quantize_g64(
                    self.curn_buf.device_ptr(),
                    self.xq_g64_buf.device_ptr(),
                    h,
                    n_tok,
                )?;
            }
            self.launch_hy4_linear(
                &l.q_a_proj,
                self.curn_buf.device_ptr(),
                self.qa_buf.device_ptr(),
                n_tok,
                curn_sk && l.q_a_proj.fmt < 100,
            )?;
            self.launch_hy4_linear(
                &l.kv_a_proj,
                self.curn_buf.device_ptr(),
                self.kva_buf.device_ptr(),
                n_tok,
                curn_sk && l.kv_a_proj.fmt < 100,
            )?;
            self.launch_hy4_linear(
                &l.linear_gate,
                self.curn_buf.device_ptr(),
                self.gate_buf.device_ptr(),
                n_tok,
                curn_sk && l.linear_gate.fmt < 100,
            )?;

            self.kernels.launch_rms_norm(
                &self.ctx,
                self.qa_buf.device_ptr(),
                l.q_a_ln.ptr,
                self.qr_buf.device_ptr(),
                self.cfg.q_lora,
                n_tok,
                self.cfg.rms_eps,
            )?;
            self.launch_hy4_linear(
                &l.q_b_proj,
                self.qr_buf.device_ptr(),
                self.q_buf.device_ptr(),
                n_tok,
                false,
            )?;

            let inv_ptr = if l.is_swa {
                self.inv_freqs_swa.device_ptr()
            } else {
                self.inv_freqs_sparse.device_ptr()
            };
            {
                let mut pq = self.q_buf.device_ptr();
                let mut pkva = self.kva_buf.device_ptr();
                let mut pcm = self.qa_buf.device_ptr(); // reuse `qa_buf` (`>= kv_lora`) for `kv_cmpr_in`!
                let mut pkpe = self.kpe_buf.device_ptr();
                let mut pif = inv_ptr;
                let mut nh = l.n_head as i32;
                let mut qkn = self.cfg.qk_nope as i32;
                let mut qkr = self.cfg.qk_rope as i32;
                let mut kvl = self.cfg.kv_lora as i32;
                let mut sp = start_pos as i32;
                let mut pp = pos_dev_ptr;
                let mut args: [*mut c_void; 11] = [
                    (&mut pq as *mut CUdeviceptr).cast(),
                    (&mut pkva as *mut CUdeviceptr).cast(),
                    (&mut pcm as *mut CUdeviceptr).cast(),
                    (&mut pkpe as *mut CUdeviceptr).cast(),
                    (&mut pif as *mut CUdeviceptr).cast(),
                    (&mut nh as *mut i32).cast(),
                    (&mut qkn as *mut i32).cast(),
                    (&mut qkr as *mut i32).cast(),
                    (&mut kvl as *mut i32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut pp as *mut CUdeviceptr).cast(),
                ];
                // SAFETY: Matches `hy4_mla_prep_rope_f32`.
                unsafe {
                    self.ctx.launch(
                        "hy4_mla_prep_rope_f32",
                        self.hk.mla_prep_rope,
                        ((l.n_head + 1) as u32, n_tok as u32, 1),
                        (128, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            self.kernels.launch_rms_norm(
                &self.ctx,
                self.qa_buf.device_ptr(),
                l.kv_a_ln.ptr,
                self.kvc_buf.device_ptr(),
                self.cfg.kv_lora,
                n_tok,
                self.cfg.rms_eps,
            )?;
            self.launch_hy4_linear(
                &l.kv_b_proj,
                self.kvc_buf.device_ptr(),
                self.kvx_buf.device_ptr(),
                n_tok,
                false,
            )?;

            {
                let mut pkvx = self.kvx_buf.device_ptr();
                let mut pkpe = self.kpe_buf.device_ptr();
                let mut pkc = l.k_cache.device_ptr();
                let mut pvc = l.v_cache.device_ptr();
                let mut nh = l.n_head as i32;
                let mut qkn = self.cfg.qk_nope as i32;
                let mut qkr = self.cfg.qk_rope as i32;
                let mut vh = self.cfg.v_head as i32;
                let mut rc = l.ring_cap as i32;
                let mut sp = start_pos as i32;
                let mut pp = pos_dev_ptr;
                let mut args: [*mut c_void; 11] = [
                    (&mut pkvx as *mut CUdeviceptr).cast(),
                    (&mut pkpe as *mut CUdeviceptr).cast(),
                    (&mut pkc as *mut CUdeviceptr).cast(),
                    (&mut pvc as *mut CUdeviceptr).cast(),
                    (&mut nh as *mut i32).cast(),
                    (&mut qkn as *mut i32).cast(),
                    (&mut qkr as *mut i32).cast(),
                    (&mut vh as *mut i32).cast(),
                    (&mut rc as *mut i32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut pp as *mut CUdeviceptr).cast(),
                ];
                // SAFETY: Matches `hy4_pack_mla_kv_f16` (note `Block: (320, 1, 1)` when `qk_head > 256`!).
                let blk_t = self.cfg.qk_head.max(self.cfg.v_head).next_multiple_of(32) as u32;
                unsafe {
                    self.ctx.launch(
                        "hy4_pack_mla_kv_f16",
                        self.hk.pack_mla_kv,
                        (l.n_head as u32, n_tok as u32, 1),
                        (blk_t, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            {
                let mut pq = self.q_buf.device_ptr();
                let mut pkc = l.k_cache.device_ptr();
                let mut pvc = l.v_cache.device_ptr();
                let mut psink = l.sinks.ptr;
                let mut po = self.attn_buf.device_ptr();
                let mut nh = l.n_head as i32;
                let mut qkh = self.cfg.qk_head as i32;
                let mut vh = self.cfg.v_head as i32;
                let mut nswa = if l.is_swa { self.cfg.window as i32 } else { 0 };
                let mut rc = l.ring_cap as i32;
                let mut sp = start_pos as i32;
                let mut pp = pos_dev_ptr;
                let mut args: [*mut c_void; 12] = [
                    (&mut pq as *mut CUdeviceptr).cast(),
                    (&mut pkc as *mut CUdeviceptr).cast(),
                    (&mut pvc as *mut CUdeviceptr).cast(),
                    (&mut psink as *mut CUdeviceptr).cast(),
                    (&mut po as *mut CUdeviceptr).cast(),
                    (&mut nh as *mut i32).cast(),
                    (&mut qkh as *mut i32).cast(),
                    (&mut vh as *mut i32).cast(),
                    (&mut nswa as *mut i32).cast(),
                    (&mut rc as *mut i32).cast(),
                    (&mut sp as *mut i32).cast(),
                    (&mut pp as *mut CUdeviceptr).cast(),
                ];
                // SAFETY: Matches `hy4_attn_f16_kv_with_sink`.
                unsafe {
                    self.ctx.launch(
                        "hy4_attn_f16_kv_with_sink",
                        self.hk.attn_with_sink,
                        (l.n_head as u32, n_tok as u32, 1),
                        (32, 8, 1),
                        0,
                        &mut args,
                    )?;
                }
            }

            let vd = l.n_head * self.cfg.v_head;
            self.kernels.launch_sigmoid_gate_mul(
                &self.ctx,
                self.attn_buf.device_ptr(),
                self.gate_buf.device_ptr(),
                n_tok * vd,
            )?;
            self.launch_hy4_linear(
                &l.o_proj,
                self.attn_buf.device_ptr(),
                self.cur_buf.device_ptr(),
                n_tok,
                false,
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

                    // Quantize `lat_z [n_tok, lat]` into `xq_g64_buf` and run parallel `ku`-slot `S_K` gate_up!
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
                        let mut ptw = self.top_weights_buf.device_ptr();
                        let mut po = self.lat_gu_buf.device_ptr();
                        let mut isd = 0i32;
                        let mut cr = (2 * lat) as i32;
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
                                (((2 * lat) as u32).div_ceil(4), ku as u32, n_tok as u32),
                                (32, 4, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }

                    // Clipped SwiGLU + SmoothQuant `sq16` divide + `BlockQ8_1_G64` quantize!
                    let n_rows = n_tok * ku;
                    {
                        let mut pgu = self.lat_gu_buf.device_ptr();
                        let mut ps4 = self.sec4_arena.device_ptr();
                        let mut pld = layer_dir_ptr;
                        let mut prk = rank_layer_ptr;
                        let mut pid = self.top_ids_buf.device_ptr();
                        let mut pxq = self.slot_xq_g64_buf.device_ptr();
                        let mut pxf = self.lat_act_f32_buf.device_ptr();
                        let mut cl = lat as i32;
                        let mut cnr = n_rows as i32;
                        let mut clim = self.cfg.swiglu_limit;
                        let mut args: [*mut c_void; 10] = [
                            (&mut pgu as *mut CUdeviceptr).cast(),
                            (&mut ps4 as *mut CUdeviceptr).cast(),
                            (&mut pld as *mut CUdeviceptr).cast(),
                            (&mut prk as *mut CUdeviceptr).cast(),
                            (&mut pid as *mut CUdeviceptr).cast(),
                            (&mut pxq as *mut CUdeviceptr).cast(),
                            (&mut pxf as *mut CUdeviceptr).cast(),
                            (&mut cl as *mut i32).cast(),
                            (&mut cnr as *mut i32).cast(),
                            (&mut clim as *mut f32).cast(),
                        ];
                        // SAFETY: Matches `hy4_latmoe_swiglu_sq_quantize_g64`.
                        unsafe {
                            self.ctx.launch(
                                "hy4_latmoe_swiglu_sq_quantize_g64",
                                self.hk.latmoe_swiglu_sq,
                                (((lat / 64) as u32).div_ceil(2), n_rows as u32, 1),
                                (32, 2, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }

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

        if compute_last_logits {
            let head_hc = CudaHy4HcPre {
                hc_fn: self.hc_head_fn.clone(),
                hc_scale: self.hc_head_scale.clone(),
                hc_base: self.hc_head_base.clone(),
            };
            self.launch_hc_pre(&head_hc, self.cur_buf.device_ptr(), None, n_tok)?;
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
            self.launch_hy4_linear(
                &self.lm_head,
                self.curn_buf.device_ptr(),
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
        Ok(())
    }

    fn run_decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), CudaError> {
        self.token_ids_buf.copy_from_u32(&[token_id])?;
        self.pos_buf.copy_from_u32(&[pos as u32])?;
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
            .launch()
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
            self.token_ids_buf.copy_from_u32(chunk)?;
            self.stage_engram_to_gpu(chunk, start_pos + off)?;
            self.encode_chunk(
                chunk.len(),
                start_pos + off,
                0,
                compute_last_logits && is_last,
                false,
            )?;
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
