//! NVIDIA CUDA GPU runner for `Qwen3.8-Flash-Next` (`general.architecture == "qwen4exp"`).
//!
//! - Uploads all 48 trunk + 512-expert `Q2_0` / `GSQ-RCO` mixed-precision weights into VRAM
//!   while keeping the `26.8 GiB` `per_layer_token_embd.weight` N-gram table in zero-copy host `mmap`
//!   (gathering only `16 × 160 = 10 KiB/tok` per decode step!).
//! - Pre-populates `(token_ids_buf, pos_buf, ple_emb_buf)` before launching a single captured
//!   `CudaGraphExec` across all 48 layers!

use std::ffi::c_void;

use tm_core::{KvType, dequantize_row_bytes};
use tm_gguf::GgufFile;
use tm_models::{
    ExpertGatingFunc, MAX_SPEC_WINDOW, PackedExperts, Qwen4ExpAttnWeights, Qwen4ExpConfig,
    Qwen4ExpHcWeights, Qwen4ExpModel, Qwen4ExpPleWeights, RawMatView, SharedExpertWeights,
    SparseMoeConfig, SparseMoeFfn,
};

use crate::CudaError;
use crate::driver::{
    CUdeviceptr, CUfunction, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec, CudaPinnedBuffer,
};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat, is_q8_1_mmvq_dtype};
use crate::llama_runner::{
    CudaMoeLayer, CudaMoeScratch, CudaProfileReport, CudaStageStat, MAX_CUDA_BATCH,
    moe_expert_only_bytes, moe_weight_bytes, plan_moe_host_offload_count, upload_mat,
};

struct Qwen4ExpCudaKernels {
    init_hc: CUfunction,
    hc_rms_norm: CUfunction,
    scale_silu: CUfunction,
    hc_mix_gate: CUfunction,
    hc_combine: CUfunction,
    mtp_add_streams: CUfunction,
    ple_gate_norm: CUfunction,
    ple_dilated_conv: CUfunction,
    gdn_norm_gated_sigmoid: CUfunction,
    qsa_idx_prep_pool: CUfunction,
    qsa_select_blocks: CUfunction,
    qsa_sparse_attn_f16: CUfunction,
    qsa_sparse_attn_q8_0: CUfunction,
}

impl Qwen4ExpCudaKernels {
    fn load(ctx: &CudaContext) -> Result<Self, CudaError> {
        Ok(Self {
            init_hc: ctx.get_function("qwen4exp_init_hc_f32")?,
            hc_rms_norm: ctx.get_function("qwen4exp_hc_rms_norm_f32")?,
            scale_silu: ctx.get_function("qwen4exp_scale_silu_f32")?,
            hc_mix_gate: ctx.get_function("qwen4exp_hc_mix_gate_f32")?,
            hc_combine: ctx.get_function("qwen4exp_hc_combine_f32")?,
            mtp_add_streams: ctx.get_function("qwen4exp_mtp_add_streams_f32")?,
            ple_gate_norm: ctx.get_function("qwen4exp_ple_gate_norm_f32")?,
            ple_dilated_conv: ctx.get_function("qwen4exp_ple_dilated_conv_add_f32")?,
            gdn_norm_gated_sigmoid: ctx.get_function("gdn_rms_norm_gated_sigmoid_f32")?,
            qsa_idx_prep_pool: ctx.get_function("qwen4exp_qsa_idx_prep_pool_f32")?,
            qsa_select_blocks: ctx.get_function("qwen4exp_qsa_select_blocks_f32")?,
            qsa_sparse_attn_f16: ctx.get_function("qwen4exp_qsa_sparse_attn_f16_kv")?,
            qsa_sparse_attn_q8_0: ctx.get_function("qwen4exp_qsa_sparse_attn_q8_0_kv")?,
        })
    }
}

struct CudaHc {
    norm: CudaBuffer,
    down: CudaMat,
    up: CudaMat,
    inject: Option<CudaMat>,
}

impl CudaHc {
    fn upload(ctx: &CudaContext, h: &Qwen4ExpHcWeights<'_>) -> Result<Self, CudaError> {
        Ok(Self {
            norm: ctx.alloc_from_f32(&h.norm)?,
            down: upload_mat(ctx, &h.down)?,
            up: upload_mat(ctx, &h.up)?,
            inject: match &h.inject {
                Some(inj) => Some(upload_mat(ctx, inj)?),
                None => None,
            },
        })
    }
}

struct CudaPle {
    key: CudaMat,
    value: CudaMat,
    norm_key: CudaBuffer,
    norm_query: CudaBuffer,
    norm_conv: CudaBuffer,
    conv1d: CudaBuffer,
}

impl CudaPle {
    fn upload(ctx: &CudaContext, p: &Qwen4ExpPleWeights<'_>) -> Result<Self, CudaError> {
        Ok(Self {
            key: upload_mat(ctx, &p.key)?,
            value: upload_mat(ctx, &p.value)?,
            norm_key: ctx.alloc_from_f32(&p.norm_key)?,
            norm_query: ctx.alloc_from_f32(&p.norm_query)?,
            norm_conv: ctx.alloc_from_f32(&p.norm_conv)?,
            conv1d: ctx.alloc_from_f32(&p.conv1d)?,
        })
    }
}

#[allow(clippy::large_enum_variant)]
enum CudaQwen4Attn {
    FullQsa {
        q_norm: CudaBuffer,
        k_norm: CudaBuffer,
        wqg: CudaMat,
        wk: CudaMat,
        wv: CudaMat,
        wo: CudaMat,
        idx_q_proj: CudaMat,
        idx_k_proj: CudaMat,
        idx_q_norm: CudaBuffer,
        idx_k_norm: CudaBuffer,
        pool_ratio: usize,
        k_cache: CudaBuffer,
        v_cache: CudaBuffer,
        idx_k_cache: CudaBuffer,
        idx_pooled: CudaBuffer,
    },
    GdnSigmoid {
        wqkv: CudaMat,
        wz: CudaMat,
        w_alpha: CudaMat,
        w_beta: CudaMat,
        conv1d: CudaBuffer,
        ssm_a: CudaBuffer,
        ssm_dt_bias: CudaBuffer,
        ssm_norm: CudaBuffer,
        w_out: CudaMat,
        conv_hist: CudaBuffer,
        ssm_state: CudaBuffer,
        conv_snap: CudaBuffer,
        ssm_snap: CudaBuffer,
        spec_qkv: CudaBuffer,
        spec_alpha: CudaBuffer,
        spec_beta: CudaBuffer,
    },
}

struct CudaQwen4Layer {
    hc_attn: CudaHc,
    hc_ffn: CudaHc,
    ple: Option<CudaPle>,
    attn: CudaQwen4Attn,
    moe: CudaMoeLayer,
}

struct CudaQwen4Mtp {
    norm_emb: CudaBuffer,      // [d] F32 (`w + 1.0`)
    norm_hid: CudaBuffer,      // [hc_dim] F32 (`w + 1.0`)
    fc_emb: CudaMat,           // [d, d]
    fc_hid: CudaMat,           // [d, d]
    mix_hc: CudaHc,            // `mtp.hyper_connection_mixer.*`
    layer: CudaQwen4Layer,     // 1 QSA + 512-expert MoE layer
    res_hc_buf: CudaBuffer,    // [MAX_CUDA_BATCH, hc_dim]
    trunk_hc_hist: CudaBuffer, // [MAX_CUDA_BATCH, hc_dim] saved after trunk layer 47
    hlast_hc: CudaBuffer,      // [hc_dim] last committed trunk 4-stream state
    last_mtp_hc: CudaBuffer,   // [hc_dim] last chained MTP 4-stream state
    fix_seeds: CudaBuffer,     // [MAX_SPEC_WINDOW, hc_dim]
    mtp_seeds: CudaBuffer,     // [MAX_CUDA_BATCH, hc_dim]
    mtp_e2_buf: CudaBuffer,    // [MAX_CUDA_BATCH, d]
    mtp_hn_buf: CudaBuffer,    // [MAX_CUDA_BATCH, hc_dim]
    mtp_draft_ids: CudaBuffer, // [MAX_SPEC_WINDOW] u32
    fix_tokens: [u32; MAX_SPEC_WINDOW],
    n_fix: usize,
}

pub struct Qwen4ExpCudaRunner {
    pub ctx: CudaContext,
    pub kernels: CudaKernels,
    qk: Qwen4ExpCudaKernels,
    pub cfg: Qwen4ExpConfig,
    pub max_ctx: usize,
    pub kv_type: KvType,
    pub vram_bytes: usize,
    pub model_weight_bytes: usize,
    pub active_decode_bytes: usize,
    pub host_offloaded_moe_layers: usize,

    token_embd: CudaMat,
    hc_head: CudaHc,
    output: CudaMat,
    rope_inv_freqs: CudaBuffer,
    layers: Vec<CudaQwen4Layer>,
    mtp: Option<CudaQwen4Mtp>,

    // Host PLE N-gram state (owns read-only `GgufFile` mmap handle — zero RAM copy, 100% safe!)
    ple_gguf: Option<(GgufFile, tm_core::DType, usize)>,
    seq_tokens: Vec<u32>,
    pinned_ids: CudaPinnedBuffer,
    pinned_pos: CudaPinnedBuffer,
    pinned_ple: CudaPinnedBuffer,

    token_ids_buf: CudaBuffer,
    pos_buf: CudaBuffer,
    decode_graphs: [Option<CudaGraphExec>; 3],
    verify_graphs: [Option<CudaGraphExec>; MAX_SPEC_WINDOW],
    ple_emb_buf: CudaBuffer,
    ple_conv_hist: CudaBuffer,
    ple_conv_snap: CudaBuffer,
    spec_ple_normed: CudaBuffer,
    ple_key_buf: CudaBuffer,
    ple_val_buf: CudaBuffer,
    ple_gated_buf: CudaBuffer,
    ple_normed_buf: CudaBuffer,
    res_hc_buf: CudaBuffer,
    hc_xn_buf: CudaBuffer,
    hc_lo_buf: CudaBuffer,
    hc_gate_buf: CudaBuffer,
    hc_inject_buf: CudaBuffer,
    cur_buf: CudaBuffer,
    sub_out_buf: CudaBuffer,
    qkv_buf: CudaBuffer,
    q_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    z_buf: CudaBuffer,
    alpha_buf: CudaBuffer,
    beta_buf: CudaBuffer,
    gdn_g_buf: CudaBuffer,
    gdn_b_buf: CudaBuffer,
    idx_q_buf: CudaBuffer,
    idx_k_buf: CudaBuffer,
    qsa_scores_buf: CudaBuffer,
    qsa_allowed_buf: CudaBuffer,
    max_qsa_blocks: usize,
    hb_buf: CudaBuffer,
    hb2_buf: CudaBuffer,
    xq_scratch: CudaBuffer,
    moe_scratch: CudaMoeScratch,
    logits_dev: CudaBuffer,
    argmax_dev: CudaBuffer,
    logits_host: Vec<f32>,
}

fn hc_bytes(h: &Qwen4ExpHcWeights<'_>) -> usize {
    h.norm.len() * 4 + h.down.bytes.len() + h.up.bytes.len() + h.inject.map_or(0, |i| i.bytes.len())
}

impl Qwen4ExpCudaRunner {
    pub fn from_gguf_with_kv_type(
        gguf: &GgufFile,
        max_ctx: usize,
        kv_type: KvType,
    ) -> Result<Self, CudaError> {
        let cpu_m = Qwen4ExpModel::from_gguf(gguf)?;
        let cfg = cpu_m.cfg.clone();
        let ctx = CudaContext::new(None)?;
        let kernels = CudaKernels::load(&ctx)?;
        let qk = Qwen4ExpCudaKernels::load(&ctx)?;

        let d = cfg.n_embd;
        let hc = cfg.hc_count;
        let hc_dim = cfg.hc_dim();
        let hc_lr = cfg.hc_low_rank;
        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let conv_dim = cfg.conv_dim();
        let idx_d = cfg.indexer_key_len;

        let kv_layer_bytes = match kv_type {
            KvType::F16 => max_ctx * kv_dim * 2,
            KvType::Q8_0 => max_ctx * (kv_dim / 32) * 34,
        };
        let n_full = cpu_m.num_full_attn_layers();
        let n_gdn = cpu_m.num_gdn_layers();
        let total_state_bytes = n_full * (2 * kv_layer_bytes + max_ctx * idx_d * 2 * 2)
            + n_gdn
                * ((cfg.ssm_conv_kernel.saturating_sub(1)) * conv_dim * 4
                    + cfg.ssm_dt_rank * cfg.ssm_state_size * cfg.ssm_head_v_dim() * 4);

        let tied_out = std::ptr::eq(cpu_m.token_embd.bytes.as_ptr(), cpu_m.output.bytes.as_ptr());
        let mut model_weight_bytes = cpu_m.token_embd.bytes.len()
            + if tied_out {
                0
            } else {
                cpu_m.output.bytes.len()
            }
            + hc_bytes(&cpu_m.hc_head);
        let mut active_decode_bytes = if tied_out {
            cpu_m.output.bytes.len()
        } else {
            cpu_m.output.bytes.len() + cpu_m.token_embd.dtype.row_bytes(d)
        } + hc_bytes(&cpu_m.hc_head);

        let mut moe_exp_sizes = Vec::new();
        for lw in &cpu_m.layers {
            let hb = hc_bytes(&lw.hc_attn) + hc_bytes(&lw.hc_ffn);
            let ab = match &lw.attn {
                Qwen4ExpAttnWeights::FullQsa {
                    wqg,
                    wk,
                    wv,
                    wo,
                    idx_q_proj,
                    idx_k_proj,
                    ..
                } => {
                    wqg.bytes.len()
                        + wk.bytes.len()
                        + wv.bytes.len()
                        + wo.bytes.len()
                        + idx_q_proj.bytes.len()
                        + idx_k_proj.bytes.len()
                }
                Qwen4ExpAttnWeights::GdnSigmoid {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    w_out,
                    ..
                } => {
                    wqkv.bytes.len()
                        + wz.bytes.len()
                        + w_alpha.bytes.len()
                        + w_beta.bytes.len()
                        + w_out.bytes.len()
                }
            };
            let pb = lw
                .ple
                .as_ref()
                .map_or(0, |p| p.key.bytes.len() + p.value.bytes.len());
            let mb = moe_weight_bytes(&lw.moe);
            model_weight_bytes += hb + ab + pb + mb;

            let exp_only = moe_expert_only_bytes(&lw.moe);
            moe_exp_sizes.push(exp_only);
            let non_exp_mb = mb.saturating_sub(exp_only);
            let active_exp = (exp_only / cfg.n_expert.max(1)) * cfg.n_expert_used;
            active_decode_bytes += hb + ab + pb + non_exp_mb + active_exp;
        }

        let (free_vram, _) = ctx.mem_info()?;
        let needed_bytes = model_weight_bytes + total_state_bytes + 128 * 1024 * 1024;
        let offload_moe_count =
            plan_moe_host_offload_count(free_vram, needed_bytes, &moe_exp_sizes);
        let offloaded_bytes: usize = moe_exp_sizes.iter().rev().take(offload_moe_count).sum();
        if needed_bytes.saturating_sub(offloaded_bytes) > free_vram {
            return Err(CudaError::OutOfDeviceMemory {
                needed_mib: needed_bytes as f64 / (1024.0 * 1024.0),
                free_mib: free_vram as f64 / (1024.0 * 1024.0),
                max_ctx,
                device: ctx.info().name.clone(),
            });
        }

        let token_embd = upload_mat(&ctx, &cpu_m.token_embd)?;
        let output = if tied_out {
            token_embd.clone()
        } else {
            upload_mat(&ctx, &cpu_m.output)?
        };
        let hc_head = CudaHc::upload(&ctx, &cpu_m.hc_head)?;
        let inv_f32: Vec<f32> = cpu_m.rope_inv_freqs.iter().map(|&v| v as f32).collect();
        let rope_inv_freqs = ctx.alloc_from_f32(&inv_f32)?;

        let ple_gguf = match cpu_m.per_layer_token_embd {
            Some(RawMatView { dtype, cols, .. }) => {
                let pf = GgufFile::open(gguf.path())?;
                if let Ok((_, ple_b)) = pf.tensor_bytes("per_layer_token_embd.weight", "global")
                    && let Ok(mem) = tm_core::probe_system_memory()
                    && mem.available_bytes > (ple_b.len() as u64).saturating_add(8u64 << 30)
                    && ple_b.len() >= (64 << 20)
                {
                    // Sequentially fault large PLE mmap (`26.8 GiB`) into host page cache using 16 parallel readers
                    // so random 64-way N-gram lookups hit DDR RAM (~5 us/tok) instead of NVMe page faults (~26 ms/tok)!
                    let n_workers = 16usize;
                    let chunk = ple_b.len().div_ceil(n_workers);
                    std::thread::scope(|s| {
                        for w_i in 0..n_workers {
                            let start = (w_i * chunk).min(ple_b.len());
                            let end = ((w_i + 1) * chunk).min(ple_b.len());
                            let sub = &ple_b[start..end];
                            s.spawn(move || {
                                let mut acc = 0u8;
                                for b in sub.iter().step_by(4096) {
                                    acc ^= *b;
                                }
                                std::hint::black_box(acc);
                            });
                        }
                    });
                }
                Some((pf, dtype, cols))
            }
            None => None,
        };

        let spec_rows = MAX_SPEC_WINDOW + 1;
        let total_moe_layers = moe_exp_sizes.len();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for (il, lw) in cpu_m.layers.iter().enumerate() {
            let hc_attn = CudaHc::upload(&ctx, &lw.hc_attn)?;
            let hc_ffn = CudaHc::upload(&ctx, &lw.hc_ffn)?;
            let ple = match &lw.ple {
                Some(p) => Some(CudaPle::upload(&ctx, p)?),
                None => None,
            };
            let attn = match &lw.attn {
                Qwen4ExpAttnWeights::FullQsa {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                    idx_q_proj,
                    idx_k_proj,
                    idx_q_norm,
                    idx_k_norm,
                } => {
                    let pool_r = cfg.compress_ratios.get(il).copied().unwrap_or(4).max(1);
                    CudaQwen4Attn::FullQsa {
                        q_norm: ctx.alloc_from_f32(q_norm)?,
                        k_norm: ctx.alloc_from_f32(k_norm)?,
                        wqg: upload_mat(&ctx, wqg)?,
                        wk: upload_mat(&ctx, wk)?,
                        wv: upload_mat(&ctx, wv)?,
                        wo: upload_mat(&ctx, wo)?,
                        idx_q_proj: upload_mat(&ctx, idx_q_proj)?,
                        idx_k_proj: upload_mat(&ctx, idx_k_proj)?,
                        idx_q_norm: ctx.alloc_from_f32(idx_q_norm)?,
                        idx_k_norm: ctx.alloc_from_f32(idx_k_norm)?,
                        pool_ratio: pool_r,
                        k_cache: ctx.alloc_zeroed(kv_layer_bytes)?,
                        v_cache: ctx.alloc_zeroed(kv_layer_bytes)?,
                        idx_k_cache: ctx.alloc_zeroed(max_ctx * idx_d * 2)?,
                        idx_pooled: ctx.alloc_zeroed((max_ctx / pool_r + 2) * idx_d * 2)?,
                    }
                }
                Qwen4ExpAttnWeights::GdnSigmoid {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                } => {
                    let hist_b = cfg.ssm_conv_kernel.saturating_sub(1).max(1) * conv_dim * 4;
                    let state_b = cfg.ssm_dt_rank * cfg.ssm_state_size * cfg.ssm_head_v_dim() * 4;
                    CudaQwen4Attn::GdnSigmoid {
                        wqkv: upload_mat(&ctx, wqkv)?,
                        wz: upload_mat(&ctx, wz)?,
                        w_alpha: upload_mat(&ctx, w_alpha)?,
                        w_beta: upload_mat(&ctx, w_beta)?,
                        conv1d: ctx.alloc_from_f32(conv1d)?,
                        ssm_a: ctx.alloc_from_f32(ssm_a)?,
                        ssm_dt_bias: ctx.alloc_from_f32(ssm_dt_bias)?,
                        ssm_norm: ctx.alloc_from_f32(ssm_norm)?,
                        w_out: upload_mat(&ctx, w_out)?,
                        conv_hist: ctx.alloc_zeroed(hist_b)?,
                        ssm_state: ctx.alloc_zeroed(state_b)?,
                        conv_snap: ctx.alloc_zeroed(hist_b)?,
                        ssm_snap: ctx.alloc_zeroed(state_b)?,
                        spec_qkv: ctx.alloc_zeroed(spec_rows * conv_dim * 4)?,
                        spec_alpha: ctx.alloc_zeroed(spec_rows * cfg.ssm_dt_rank * 4)?,
                        spec_beta: ctx.alloc_zeroed(spec_rows * cfg.ssm_dt_rank * 4)?,
                    }
                }
            };
            let host_map_moe = il + offload_moe_count >= total_moe_layers;
            let moe = CudaMoeLayer::upload_with_offload(&ctx, &lw.moe, d, host_map_moe)?;
            layers.push(CudaQwen4Layer {
                hc_attn,
                hc_ffn,
                ple,
                attn,
                moe,
            });
        }

        let max_cols = [
            d,
            hc_dim,
            hc_lr,
            conv_dim,
            cfg.ssm_d_inner,
            2 * q_dim,
            cfg.max_n_ff_exp,
            cfg.max_n_ff_shexp,
        ]
        .into_iter()
        .max()
        .unwrap_or(d)
        .next_multiple_of(256);

        let pinned_ids = ctx.alloc_pinned(MAX_CUDA_BATCH * 4)?;
        let pinned_pos = ctx.alloc_pinned(4)?;
        let pinned_ple = ctx.alloc_pinned(MAX_CUDA_BATCH * d * 4)?;
        let token_ids_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * 4)?;
        let pos_buf = ctx.alloc_zeroed(4)?;
        let ple_emb_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * d * 4)?;
        let ple_hist_bytes = cfg.ple_hist_len().max(1) * hc_dim * 4;
        let ple_conv_hist = ctx.alloc_zeroed(ple_hist_bytes)?;
        let ple_conv_snap = ctx.alloc_zeroed(ple_hist_bytes)?;
        let n_ple_layers = layers.iter().filter(|l| l.ple.is_some()).count().max(1);
        let spec_ple_normed = ctx.alloc_zeroed(n_ple_layers * spec_rows * hc_dim * 4)?;
        let ple_key_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let ple_val_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * d * 4)?;
        let ple_gated_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let ple_normed_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let res_hc_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let hc_xn_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let hc_lo_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_lr * 4)?;
        let hc_gate_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?;
        let hc_inject_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hc * 4)?;
        let cur_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * d * 4)?;
        let sub_out_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * d * 4)?;
        let qkv_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * conv_dim.max(2 * q_dim) * 4)?;
        let q_buf = ctx
            .alloc_zeroed(MAX_CUDA_BATCH * (cfg.ssm_n_group * cfg.ssm_state_size).max(q_dim) * 4)?;
        let k_buf = ctx.alloc_zeroed(
            MAX_CUDA_BATCH * (cfg.ssm_n_group * cfg.ssm_state_size).max(kv_dim) * 4,
        )?;
        let v_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.ssm_d_inner.max(kv_dim) * 4)?;
        let z_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.ssm_d_inner.max(q_dim) * 4)?;
        let alpha_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.ssm_dt_rank * 4)?;
        let beta_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.ssm_dt_rank * 4)?;
        let gdn_g_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.ssm_dt_rank * 4)?;
        let gdn_b_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.ssm_dt_rank * 4)?;
        let idx_q_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.indexer_n_head * idx_d.max(1) * 4)?;
        let idx_k_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * idx_d.max(1) * 4)?;
        let max_qsa_blocks = max_ctx + 4;
        let qsa_scores_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_qsa_blocks * 4)?;
        let qsa_allowed_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_qsa_blocks)?;
        let max_shexp = cfg.max_n_ff_shexp.max(cfg.ssm_d_inner).max(d);
        let hb_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_shexp * 4)?;
        let hb2_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * max_shexp * 4)?;
        let xq_scratch = ctx.alloc_zeroed(MAX_CUDA_BATCH * (max_cols / 32) * BLOCK_Q8_1_BYTES)?;
        let moe_scratch =
            CudaMoeScratch::alloc(&ctx, d, cfg.n_expert, cfg.n_expert_used, cfg.max_n_ff_exp)?;
        let logits_dev = ctx.alloc_zeroed(spec_rows * cfg.vocab_size * 4)?;
        let argmax_dev = ctx.alloc_zeroed(spec_rows * 4)?;
        ctx.synchronize()?;

        let (free_after, _) = ctx.mem_info()?;
        let vram_bytes = free_vram.saturating_sub(free_after);

        let mut runner = Self {
            ctx,
            kernels,
            qk,
            cfg: cfg.clone(),
            max_ctx,
            kv_type,
            vram_bytes,
            model_weight_bytes,
            active_decode_bytes,
            host_offloaded_moe_layers: offload_moe_count,
            token_embd,
            hc_head,
            output,
            rope_inv_freqs,
            layers,
            mtp: None,
            ple_gguf,
            seq_tokens: vec![cfg.ple_eos_token_id; max_ctx.max(1)],
            pinned_ids,
            pinned_pos,
            pinned_ple,
            token_ids_buf,
            pos_buf,
            decode_graphs: [None, None, None],
            verify_graphs: std::array::from_fn(|_| None),
            ple_emb_buf,
            ple_conv_hist,
            ple_conv_snap,
            spec_ple_normed,
            ple_key_buf,
            ple_val_buf,
            ple_gated_buf,
            ple_normed_buf,
            res_hc_buf,
            hc_xn_buf,
            hc_lo_buf,
            hc_gate_buf,
            hc_inject_buf,
            cur_buf,
            sub_out_buf,
            qkv_buf,
            q_buf,
            k_buf,
            v_buf,
            z_buf,
            alpha_buf,
            beta_buf,
            gdn_g_buf,
            gdn_b_buf,
            idx_q_buf,
            idx_k_buf,
            qsa_scores_buf,
            qsa_allowed_buf,
            max_qsa_blocks,
            hb_buf,
            hb2_buf,
            xq_scratch,
            moe_scratch,
            logits_dev,
            argmax_dev,
            logits_host: vec![0.0; cfg.vocab_size],
        };

        // Auto-discover sibling `mtp-*.gguf` (`qwen4exp-mtp`) if present alongside the main GGUF!
        if let Some(parent) = gguf.path().parent() {
            for cand in [
                "mtp-q2_0.gguf",
                "mtp-q4_0.gguf",
                "mtp-q8_0.gguf",
                "mtp.gguf",
            ] {
                let p = parent.join(cand);
                if p.is_file()
                    && let Ok(mg) = GgufFile::open(&p)
                    && mg.req_str("general.architecture").ok() == Some("qwen4exp-mtp")
                {
                    let _ = runner.attach_mtp_gguf(&mg);
                    break;
                }
            }
        }

        Ok(runner)
    }

    #[inline]
    pub fn has_builtin_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Attaches a packed `qwen4exp-mtp` GGUF (`Strata/terminal/tools/mtp_pack.py`, 31 `mtp.*` tensors).
    pub fn attach_mtp_gguf(&mut self, mtp_gguf: &GgufFile) -> Result<(), CudaError> {
        let arch = mtp_gguf.req_str("general.architecture")?;
        if arch != "qwen4exp-mtp" {
            return Err(CudaError::UnsupportedArch(format!(
                "expected 'qwen4exp-mtp' GGUF for Qwen4Exp MTP, got '{arch}'"
            )));
        }
        let (free_before, _) = self.ctx.mem_info()?;
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = self.cfg.hc_dim();
        let hc_lr = self.cfg.hc_low_rank;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let idx_nh = self.cfg.indexer_n_head;
        let idx_d = self.cfg.indexer_key_len;
        let ne = self.cfg.n_expert;

        let load_gemma_norm = |name: &str, len: usize| -> Result<Vec<f32>, CudaError> {
            let (ti, bytes) = mtp_gguf.tensor_bytes(name, "mtp")?;
            let actual: usize = ti.dims.iter().product();
            if actual != len {
                return Err(CudaError::InvalidDimensions(format!(
                    "MTP norm '{name}' expected {len} elems, got {:?}",
                    ti.dims
                )));
            }
            let mut v = vec![0.0f32; len];
            dequantize_row_bytes(ti.dtype, bytes, &mut v);
            for x in &mut v {
                *x += 1.0;
            }
            Ok(v)
        };
        let load_mat_view =
            |name: &'static str, cols: usize, rows: usize| -> Result<RawMatView<'_>, CudaError> {
                let (ti, bytes) = mtp_gguf.req_tensor(name, "mtp", &[cols, rows])?;
                Ok(RawMatView {
                    name: ti.name.as_str(),
                    bytes,
                    dtype: ti.dtype,
                    rows,
                    cols,
                    row_offset: 0,
                    is_sliced: false,
                    lora_slot: None,
                })
            };
        let upload_mtp_hc = |pfx: &str, with_inject: bool| -> Result<CudaHc, CudaError> {
            let norm = load_gemma_norm(&format!("{pfx}.hc_norm.weight"), hc_dim)?;
            let n_down = format!("{pfx}.input_mix_weight_down.weight");
            let (di, db) = mtp_gguf.req_tensor(&n_down, "mtp", &[hc_dim, hc_lr])?;
            let down = upload_mat(
                &self.ctx,
                &RawMatView {
                    name: di.name.as_str(),
                    bytes: db,
                    dtype: di.dtype,
                    rows: hc_lr,
                    cols: hc_dim,
                    row_offset: 0,
                    is_sliced: false,
                    lora_slot: None,
                },
            )?;
            let n_up = format!("{pfx}.input_mix_weight_up.weight");
            let (ui, ub) = mtp_gguf.req_tensor(&n_up, "mtp", &[hc_lr, hc_dim])?;
            let up = upload_mat(
                &self.ctx,
                &RawMatView {
                    name: ui.name.as_str(),
                    bytes: ub,
                    dtype: ui.dtype,
                    rows: hc_dim,
                    cols: hc_lr,
                    row_offset: 0,
                    is_sliced: false,
                    lora_slot: None,
                },
            )?;
            let inject = if with_inject {
                let n_inj = format!("{pfx}.block_inject_weight.weight");
                let (ii, ib) = mtp_gguf.req_tensor(&n_inj, "mtp", &[hc_dim, hc])?;
                Some(upload_mat(
                    &self.ctx,
                    &RawMatView {
                        name: ii.name.as_str(),
                        bytes: ib,
                        dtype: ii.dtype,
                        rows: hc,
                        cols: hc_dim,
                        row_offset: 0,
                        is_sliced: false,
                        lora_slot: None,
                    },
                )?)
            } else {
                None
            };
            Ok(CudaHc {
                norm: self.ctx.alloc_from_f32(&norm)?,
                down,
                up,
                inject,
            })
        };

        let norm_emb = self
            .ctx
            .alloc_from_f32(&load_gemma_norm("mtp.pre_fc_norm_embedding.weight", d)?)?;
        let norm_hid = self
            .ctx
            .alloc_from_f32(&load_gemma_norm("mtp.pre_fc_norm_hidden.weight", hc_dim)?)?;
        let fc_emb = upload_mat(&self.ctx, &load_mat_view("mtp.fc_embedding.weight", d, d)?)?;
        let fc_hid = upload_mat(&self.ctx, &load_mat_view("mtp.fc_hidden.weight", d, d)?)?;
        let mix_hc = upload_mtp_hc("mtp.hyper_connection_mixer", false)?;
        let hc_attn = upload_mtp_hc("mtp.layers.0.attn_hyper_connection", true)?;
        let hc_ffn = upload_mtp_hc("mtp.layers.0.mlp_hyper_connection", true)?;

        let q_norm = self.ctx.alloc_from_f32(&load_gemma_norm(
            "mtp.layers.0.self_attn.q_norm.weight",
            self.cfg.head_dim,
        )?)?;
        let k_norm = self.ctx.alloc_from_f32(&load_gemma_norm(
            "mtp.layers.0.self_attn.k_norm.weight",
            self.cfg.head_dim,
        )?)?;
        let wqg = upload_mat(
            &self.ctx,
            &load_mat_view("mtp.layers.0.self_attn.q_proj.weight", d, 2 * q_dim)?,
        )?;
        let wk = upload_mat(
            &self.ctx,
            &load_mat_view("mtp.layers.0.self_attn.k_proj.weight", d, kv_dim)?,
        )?;
        let wv = upload_mat(
            &self.ctx,
            &load_mat_view("mtp.layers.0.self_attn.v_proj.weight", d, kv_dim)?,
        )?;
        let wo = upload_mat(
            &self.ctx,
            &load_mat_view("mtp.layers.0.self_attn.o_proj.weight", q_dim, d)?,
        )?;

        let (iqk_info, iqk_bytes) = mtp_gguf.req_tensor(
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            "mtp",
            &[d, (idx_nh + 1) * idx_d],
        )?;
        let iqk_rb = iqk_info.dtype.row_bytes(d);
        let q_rows = idx_nh * idx_d;
        let idx_q_proj = upload_mat(
            &self.ctx,
            &RawMatView {
                name: iqk_info.name.as_str(),
                bytes: &iqk_bytes[..q_rows * iqk_rb],
                dtype: iqk_info.dtype,
                rows: q_rows,
                cols: d,
                row_offset: 0,
                is_sliced: true,
                lora_slot: None,
            },
        )?;
        let idx_k_proj = upload_mat(
            &self.ctx,
            &RawMatView {
                name: iqk_info.name.as_str(),
                bytes: &iqk_bytes[q_rows * iqk_rb..(q_rows + idx_d) * iqk_rb],
                dtype: iqk_info.dtype,
                rows: idx_d,
                cols: d,
                row_offset: q_rows,
                is_sliced: true,
                lora_slot: None,
            },
        )?;
        let idx_q_norm = self.ctx.alloc_from_f32(&load_gemma_norm(
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
            idx_d,
        )?)?;
        let idx_k_norm = self.ctx.alloc_from_f32(&load_gemma_norm(
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
            idx_d,
        )?)?;

        let kv_layer_bytes = match self.kv_type {
            KvType::F16 => self.max_ctx * kv_dim * 2,
            KvType::Q8_0 => self.max_ctx * (kv_dim / 32) * 34,
        };
        let pool_ratio = 4usize;
        let attn = CudaQwen4Attn::FullQsa {
            q_norm,
            k_norm,
            wqg,
            wk,
            wv,
            wo,
            idx_q_proj,
            idx_k_proj,
            idx_q_norm,
            idx_k_norm,
            pool_ratio,
            k_cache: self.ctx.alloc_zeroed(kv_layer_bytes)?,
            v_cache: self.ctx.alloc_zeroed(kv_layer_bytes)?,
            idx_k_cache: self.ctx.alloc_zeroed(self.max_ctx * idx_d * 2)?,
            idx_pooled: self
                .ctx
                .alloc_zeroed((self.max_ctx / pool_ratio + 2) * idx_d * 2)?,
        };

        // Load 512-expert `FusedGateUp` + Sigmoid-gated shared expert
        let gate_inp = load_mat_view("mtp.layers.0.mlp.gate.weight", d, ne)?;
        let (gu_info, gu_bytes) =
            mtp_gguf.tensor_bytes("mtp.layers.0.mlp.experts.gate_up_proj", "mtp")?;
        if gu_info.dims.len() != 3 || gu_info.dims[0] != d || gu_info.dims[2] != ne {
            return Err(CudaError::InvalidDimensions(format!(
                "MTP gate_up_proj unexpected dims {:?}",
                gu_info.dims
            )));
        }
        let n_ff_exp = gu_info.dims[1] / 2;
        let (dn_info, dn_bytes) = mtp_gguf.req_tensor(
            "mtp.layers.0.mlp.experts.down_proj",
            "mtp",
            &[n_ff_exp, d, ne],
        )?;
        let gu_rb = gu_info.dtype.row_bytes(d);
        let dn_rb = dn_info.dtype.row_bytes(n_ff_exp);
        let sh_gate = load_mat_view(
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            d,
            self.cfg.max_n_ff_shexp,
        )?;
        let sh_up = load_mat_view(
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            d,
            self.cfg.max_n_ff_shexp,
        )?;
        let sh_down = load_mat_view(
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            self.cfg.max_n_ff_shexp,
            d,
        )?;
        let sh_gate_inp = load_mat_view("mtp.layers.0.mlp.shared_expert_gate.weight", d, 1)?;
        let moe_view = SparseMoeFfn {
            cfg: SparseMoeConfig {
                n_expert: ne,
                n_expert_used: self.cfg.n_expert_used,
                n_ff_exp,
                n_ff_shexp: self.cfg.max_n_ff_shexp,
                gating_func: ExpertGatingFunc::Softmax,
                norm_weights: true,
                norm_weights_eps: 0.0,
                weights_scale: 1.0,
            },
            gate_inp,
            gate_inp_bias: None,
            exp_probs_b: None,
            experts: PackedExperts::FusedGateUp {
                gate_up_exps: gu_bytes,
                gate_up_dtype: gu_info.dtype,
                gate_up_exp_bytes: 2 * n_ff_exp * gu_rb,
                half_exp_bytes: n_ff_exp * gu_rb,
                down_exps: dn_bytes,
                down_dtype: dn_info.dtype,
                down_exp_bytes: d * dn_rb,
            },
            shared_expert: Some(SharedExpertWeights {
                n_ff_shexp: self.cfg.max_n_ff_shexp,
                w_gate: sh_gate,
                w_up: sh_up,
                w_down: sh_down,
                w_gate_inp_shexp: Some(sh_gate_inp),
            }),
        };
        let moe = CudaMoeLayer::upload_with_offload(&self.ctx, &moe_view, d, false)?;

        let mtp = CudaQwen4Mtp {
            norm_emb,
            norm_hid,
            fc_emb,
            fc_hid,
            mix_hc,
            layer: CudaQwen4Layer {
                hc_attn,
                hc_ffn,
                ple: None,
                attn,
                moe,
            },
            res_hc_buf: self.ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?,
            trunk_hc_hist: self.ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?,
            hlast_hc: self.ctx.alloc_zeroed(hc_dim * 4)?,
            last_mtp_hc: self.ctx.alloc_zeroed(hc_dim * 4)?,
            fix_seeds: self.ctx.alloc_zeroed(MAX_SPEC_WINDOW * hc_dim * 4)?,
            mtp_seeds: self.ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?,
            mtp_e2_buf: self.ctx.alloc_zeroed(MAX_CUDA_BATCH * d * 4)?,
            mtp_hn_buf: self.ctx.alloc_zeroed(MAX_CUDA_BATCH * hc_dim * 4)?,
            mtp_draft_ids: self.ctx.alloc_zeroed(MAX_SPEC_WINDOW * 4)?,
            fix_tokens: [0; MAX_SPEC_WINDOW],
            n_fix: 0,
        };
        self.ctx.synchronize()?;
        let (free_after, _) = self.ctx.mem_info()?;
        self.vram_bytes += free_before.saturating_sub(free_after);
        self.decode_graphs = [None, None, None];
        self.verify_graphs = std::array::from_fn(|_| None);
        self.mtp = Some(mtp);
        Ok(())
    }

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        self.ple_conv_hist.zero()?;
        let zero_attn = |attn: &mut CudaQwen4Attn| -> Result<(), CudaError> {
            match attn {
                CudaQwen4Attn::FullQsa {
                    k_cache,
                    v_cache,
                    idx_k_cache,
                    idx_pooled,
                    ..
                } => {
                    k_cache.zero()?;
                    v_cache.zero()?;
                    idx_k_cache.zero()?;
                    idx_pooled.zero()?;
                }
                CudaQwen4Attn::GdnSigmoid {
                    conv_hist,
                    ssm_state,
                    ..
                } => {
                    conv_hist.zero()?;
                    ssm_state.zero()?;
                }
            }
            Ok(())
        };
        for l in &mut self.layers {
            zero_attn(&mut l.attn)?;
        }
        if let Some(ref mut mtp) = self.mtp {
            zero_attn(&mut mtp.layer.attn)?;
            mtp.hlast_hc.zero()?;
            mtp.last_mtp_hc.zero()?;
            mtp.n_fix = 0;
        }
        self.ctx.synchronize()
    }

    fn stage_ple_to_gpu(&mut self, token_ids: &[u32], start_pos: usize) -> Result<(), CudaError> {
        for (i, &tid) in token_ids.iter().enumerate() {
            self.seq_tokens[start_pos + i] = tid;
        }
        let Some((ref ple_f, dtype, hd)) = self.ple_gguf else {
            return Ok(());
        };
        let (_, ple_bytes) = ple_f.tensor_bytes("per_layer_token_embd.weight", "global")?;
        let n_tok = token_ids.len();
        let ng = self.cfg.ple_ngram_size;
        let per_g = self.cfg.ple_heads_per_ngram;
        let eos = self.cfg.ple_eos_token_id;
        let rb = dtype.row_bytes(hd);
        let d = self.cfg.n_embd;
        let mut ctx_toks = [0u64; 8];

        for (i, &tok) in token_ids.iter().enumerate().take(n_tok) {
            let pos = start_pos + i;
            ctx_toks[0] = tok as u64;
            let mut cut = false;
            for (s, ct) in ctx_toks.iter_mut().enumerate().take(ng).skip(1) {
                let prev = if cut || pos < s {
                    None
                } else {
                    Some(self.seq_tokens[pos - s])
                };
                match prev {
                    None => {
                        cut = true;
                        *ct = eos as u64;
                    }
                    Some(pt) => {
                        if pt == eos {
                            cut = true;
                        }
                        *ct = if cut { eos as u64 } else { pt as u64 };
                    }
                }
            }
            let dst_all = self.pinned_ple.as_f32_slice_mut(n_tok * d);
            let dst = &mut dst_all[i * d..(i + 1) * d];
            for n in 2..=ng {
                let mut mixed = ctx_toks[0].wrapping_mul(self.cfg.ple_layer_multipliers[0]);
                for (j, &cj) in ctx_toks.iter().enumerate().take(n).skip(1) {
                    mixed ^= cj.wrapping_mul(self.cfg.ple_layer_multipliers[j]);
                }
                let base = (n - 2) * per_g;
                for g in 0..per_g {
                    let h_i = base + g;
                    let row = (mixed % self.cfg.ple_head_vocab_sizes[h_i]
                        + self.cfg.ple_head_offsets[h_i]) as usize;
                    tm_core::dequantize_row_bytes(
                        dtype,
                        &ple_bytes[row * rb..(row + 1) * rb],
                        &mut dst[h_i * hd..(h_i + 1) * hd],
                    );
                }
            }
        }
        self.ple_emb_buf
            .copy_from_pinned_async(0, &self.pinned_ple, 0, n_tok * d * 4)
    }

    fn launch_hc_mix_at(
        &self,
        res_hc_ptr: CUdeviceptr,
        hc_w: &CudaHc,
        mixed_ptr: CUdeviceptr,
        inject_ptr: Option<CUdeviceptr>,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * d;
        let hc_lr = self.cfg.hc_low_rank;

        {
            let mut pr = res_hc_ptr;
            let mut pw = hc_w.norm.device_ptr();
            let mut px = self.hc_xn_buf.device_ptr();
            let mut cd = d as i32;
            let mut ch = hc as i32;
            let mut eps = self.cfg.rms_eps;
            let mut args: [*mut c_void; 6] = [
                (&mut pr as *mut CUdeviceptr).cast(),
                (&mut pw as *mut CUdeviceptr).cast(),
                (&mut px as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut ch as *mut i32).cast(),
                (&mut eps as *mut f32).cast(),
            ];
            // SAFETY: Matches `qwen4exp_hc_rms_norm_f32`.
            unsafe {
                self.ctx.launch(
                    "qwen4exp_hc_rms_norm_f32",
                    self.qk.hc_rms_norm,
                    (hc as u32, n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        let can_share_q8 = n_tok <= crate::kernels::MMVQ_MAX_TOK && hc_dim.is_multiple_of(32);
        let down_q8 = is_q8_1_mmvq_dtype(hc_w.down.dtype);
        let inj_q8 = hc_w
            .inject
            .as_ref()
            .is_some_and(|m| is_q8_1_mmvq_dtype(m.dtype));
        if can_share_q8 && (down_q8 || inj_q8) {
            self.kernels.launch_quantize_q8_1(
                &self.ctx,
                self.hc_xn_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                hc_dim,
                n_tok,
            )?;
        }
        self.kernels.launch_linear(
            &self.ctx,
            &hc_w.down,
            self.hc_xn_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            can_share_q8 && down_q8,
            self.hc_lo_buf.device_ptr(),
            n_tok,
            false,
        )?;
        if let (Some(w_inj), Some(p_inj)) = (&hc_w.inject, inject_ptr) {
            self.kernels.launch_linear(
                &self.ctx,
                w_inj,
                self.hc_xn_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                can_share_q8 && inj_q8,
                p_inj,
                n_tok,
                false,
            )?;
        }

        {
            let mut pl = self.hc_lo_buf.device_ptr();
            let mut sc = 1.0 / (hc as f32);
            let mut n = n_tok * hc_lr;
            let mut args: [*mut c_void; 3] = [
                (&mut pl as *mut CUdeviceptr).cast(),
                (&mut sc as *mut f32).cast(),
                (&mut n as *mut usize).cast(),
            ];
            // SAFETY: Matches `qwen4exp_scale_silu_f32`.
            unsafe {
                self.ctx.launch(
                    "qwen4exp_scale_silu_f32",
                    self.qk.scale_silu,
                    (((n_tok * hc_lr) as u32).div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        self.kernels.launch_linear(
            &self.ctx,
            &hc_w.up,
            self.hc_lo_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.hc_gate_buf.device_ptr(),
            n_tok,
            false,
        )?;

        {
            let mut px = self.hc_xn_buf.device_ptr();
            let mut pg = self.hc_gate_buf.device_ptr();
            let mut pm = mixed_ptr;
            let mut cd = d as i32;
            let mut ch = hc as i32;
            let mut args: [*mut c_void; 5] = [
                (&mut px as *mut CUdeviceptr).cast(),
                (&mut pg as *mut CUdeviceptr).cast(),
                (&mut pm as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut ch as *mut i32).cast(),
            ];
            // SAFETY: Matches `qwen4exp_hc_mix_gate_f32`.
            unsafe {
                self.ctx.launch(
                    "qwen4exp_hc_mix_gate_f32",
                    self.qk.hc_mix_gate,
                    ((d as u32).div_ceil(256), n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )
            }
        }
    }

    fn launch_hc_mix(
        &self,
        hc_w: &CudaHc,
        mixed_ptr: CUdeviceptr,
        inject_ptr: Option<CUdeviceptr>,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        self.launch_hc_mix_at(
            self.res_hc_buf.device_ptr(),
            hc_w,
            mixed_ptr,
            inject_ptr,
            n_tok,
        )
    }

    fn launch_hc_combine_at(
        &self,
        res_hc_ptr: CUdeviceptr,
        sub_ptr: CUdeviceptr,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let mut pr = res_hc_ptr;
        let mut ps = sub_ptr;
        let mut pi = self.hc_inject_buf.device_ptr();
        let mut cd = d as i32;
        let mut ch = hc as i32;
        let mut args: [*mut c_void; 5] = [
            (&mut pr as *mut CUdeviceptr).cast(),
            (&mut ps as *mut CUdeviceptr).cast(),
            (&mut pi as *mut CUdeviceptr).cast(),
            (&mut cd as *mut i32).cast(),
            (&mut ch as *mut i32).cast(),
        ];
        // SAFETY: Matches `qwen4exp_hc_combine_f32`.
        unsafe {
            self.ctx.launch(
                "qwen4exp_hc_combine_f32",
                self.qk.hc_combine,
                (((hc * d) as u32).div_ceil(256), n_tok as u32, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_one_qwen4_layer(
        &self,
        l: &CudaQwen4Layer,
        res_hc_ptr: CUdeviceptr,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        save_spec_state: bool,
        ple_ordinal: &mut usize,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * d;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let dv_dim = self.cfg.ssm_head_v_dim();
        let conv_dim = self.cfg.conv_dim();
        let idx_nh = self.cfg.indexer_n_head;
        let idx_d = self.cfg.indexer_key_len;

        if let Some(ref ple) = l.ple {
            self.kernels.launch_linear(
                &self.ctx,
                &ple.key,
                self.ple_emb_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.ple_key_buf.device_ptr(),
                n_tok,
                false,
            )?;
            self.kernels.launch_linear(
                &self.ctx,
                &ple.value,
                self.ple_emb_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.ple_val_buf.device_ptr(),
                n_tok,
                false,
            )?;
            {
                let mut pr = res_hc_ptr;
                let mut pk = self.ple_key_buf.device_ptr();
                let mut pv = self.ple_val_buf.device_ptr();
                let mut pnk = ple.norm_key.device_ptr();
                let mut pnq = ple.norm_query.device_ptr();
                let mut pnc = ple.norm_conv.device_ptr();
                let mut pg = self.ple_gated_buf.device_ptr();
                let mut pn = self.ple_normed_buf.device_ptr();
                let mut cd = d as i32;
                let mut ch = hc as i32;
                let mut eps = self.cfg.rms_eps;
                let mut args: [*mut c_void; 11] = [
                    (&mut pr as *mut CUdeviceptr).cast(),
                    (&mut pk as *mut CUdeviceptr).cast(),
                    (&mut pv as *mut CUdeviceptr).cast(),
                    (&mut pnk as *mut CUdeviceptr).cast(),
                    (&mut pnq as *mut CUdeviceptr).cast(),
                    (&mut pnc as *mut CUdeviceptr).cast(),
                    (&mut pg as *mut CUdeviceptr).cast(),
                    (&mut pn as *mut CUdeviceptr).cast(),
                    (&mut cd as *mut i32).cast(),
                    (&mut ch as *mut i32).cast(),
                    (&mut eps as *mut f32).cast(),
                ];
                // SAFETY: Matches `qwen4exp_ple_gate_norm_f32`.
                unsafe {
                    self.ctx.launch(
                        "qwen4exp_ple_gate_norm_f32",
                        self.qk.ple_gate_norm,
                        (hc as u32, n_tok as u32, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
            if save_spec_state {
                let dst_off = (*ple_ordinal) * (MAX_SPEC_WINDOW + 1) * hc_dim * 4;
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    self.spec_ple_normed.device_ptr() + dst_off as u64,
                    self.ple_normed_buf.device_ptr(),
                    n_tok * hc_dim,
                    n_tok * hc_dim,
                    n_tok * hc_dim,
                    1,
                )?;
            }
            *ple_ordinal += 1;
            {
                let mut pr = res_hc_ptr;
                let mut pg = self.ple_gated_buf.device_ptr();
                let mut pn = self.ple_normed_buf.device_ptr();
                let mut pcw = ple.conv1d.device_ptr();
                let mut ph = self.ple_conv_hist.device_ptr();
                let mut chd = hc_dim as i32;
                let mut ck = self.cfg.ple_conv_kernel as i32;
                let mut cs = self.cfg.ple_ngram_size as i32;
                let mut nt = n_tok as i32;
                let mut args: [*mut c_void; 9] = [
                    (&mut pr as *mut CUdeviceptr).cast(),
                    (&mut pg as *mut CUdeviceptr).cast(),
                    (&mut pn as *mut CUdeviceptr).cast(),
                    (&mut pcw as *mut CUdeviceptr).cast(),
                    (&mut ph as *mut CUdeviceptr).cast(),
                    (&mut chd as *mut i32).cast(),
                    (&mut ck as *mut i32).cast(),
                    (&mut cs as *mut i32).cast(),
                    (&mut nt as *mut i32).cast(),
                ];
                // SAFETY: Matches `qwen4exp_ple_dilated_conv_add_f32`.
                unsafe {
                    self.ctx.launch(
                        "qwen4exp_ple_dilated_conv_add_f32",
                        self.qk.ple_dilated_conv,
                        ((hc_dim as u32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
            }
        }

        // 1. Attention / GDN HC Mix
        self.launch_hc_mix_at(
            res_hc_ptr,
            &l.hc_attn,
            self.cur_buf.device_ptr(),
            Some(self.hc_inject_buf.device_ptr()),
            n_tok,
        )?;

        let can_share_q8 = n_tok <= crate::kernels::MMVQ_MAX_TOK && d.is_multiple_of(32);
        match &l.attn {
            CudaQwen4Attn::GdnSigmoid {
                wqkv,
                wz,
                w_alpha,
                w_beta,
                conv1d,
                ssm_a,
                ssm_dt_bias,
                ssm_norm,
                w_out,
                conv_hist,
                ssm_state,
                conv_snap,
                ssm_snap,
                spec_qkv,
                spec_alpha,
                spec_beta,
            } => {
                let qkv_q8 = is_q8_1_mmvq_dtype(wqkv.dtype);
                let z_q8 = is_q8_1_mmvq_dtype(wz.dtype);
                if can_share_q8 && (qkv_q8 || z_q8) {
                    self.kernels.launch_quantize_q8_1(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        d,
                        n_tok,
                    )?;
                }
                self.kernels.launch_linear(
                    &self.ctx,
                    wqkv,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && qkv_q8,
                    self.qkv_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    wz,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && z_q8,
                    self.z_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    w_alpha,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && (qkv_q8 || z_q8) && is_q8_1_mmvq_dtype(w_alpha.dtype),
                    self.alpha_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    w_beta,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && (qkv_q8 || z_q8) && is_q8_1_mmvq_dtype(w_beta.dtype),
                    self.beta_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                if save_spec_state {
                    spec_qkv.copy_from_device(&self.qkv_buf, n_tok * conv_dim * 4)?;
                    spec_alpha.copy_from_device(&self.alpha_buf, n_tok * hv * 4)?;
                    spec_beta.copy_from_device(&self.beta_buf, n_tok * hv * 4)?;
                }

                if n_tok > 1 || save_spec_state {
                    self.kernels.launch_gdn_multi_step(
                        &self.ctx,
                        self.qkv_buf.device_ptr(),
                        self.z_buf.device_ptr(),
                        self.alpha_buf.device_ptr(),
                        self.beta_buf.device_ptr(),
                        conv_hist.device_ptr(),
                        conv_hist.device_ptr(),
                        if save_spec_state {
                            conv_snap.device_ptr()
                        } else {
                            0
                        },
                        conv1d.device_ptr(),
                        ssm_a.device_ptr(),
                        ssm_dt_bias.device_ptr(),
                        ssm_norm.device_ptr(),
                        ssm_state.device_ptr(),
                        ssm_state.device_ptr(),
                        if save_spec_state {
                            ssm_snap.device_ptr()
                        } else {
                            0
                        },
                        self.q_buf.device_ptr(),
                        self.k_buf.device_ptr(),
                        self.hb_buf.device_ptr(),
                        self.gdn_g_buf.device_ptr(),
                        self.gdn_b_buf.device_ptr(),
                        self.v_buf.device_ptr(),
                        conv_dim,
                        self.cfg.ssm_conv_kernel,
                        hk,
                        hv,
                        s_dim,
                        dv_dim,
                        self.cfg.rms_eps,
                        1, // Sigmoid gated RMSNorm (`qwen4exp`)
                        n_tok,
                    )?;
                } else {
                    let qkv_t = self.qkv_buf.device_ptr();
                    let z_t = self.z_buf.device_ptr();
                    let a_t = self.alpha_buf.device_ptr();
                    let b_t = self.beta_buf.device_ptr();
                    let out_t = self.v_buf.device_ptr();

                    // 1. Causal Conv1d + SiLU BEFORE split
                    {
                        let mut pqkv = qkv_t;
                        let mut phist = conv_hist.device_ptr();
                        let mut pcw = conv1d.device_ptr();
                        let mut pq = self.q_buf.device_ptr();
                        let mut pk = self.k_buf.device_ptr();
                        let mut pv = self.hb_buf.device_ptr();
                        let mut cd = conv_dim as i32;
                        let mut dc = self.cfg.ssm_conv_kernel as i32;
                        let mut kd = (hk * s_dim) as i32;
                        let mut args: [*mut c_void; 9] = [
                            (&mut pqkv as *mut CUdeviceptr).cast(),
                            (&mut phist as *mut CUdeviceptr).cast(),
                            (&mut pcw as *mut CUdeviceptr).cast(),
                            (&mut pq as *mut CUdeviceptr).cast(),
                            (&mut pk as *mut CUdeviceptr).cast(),
                            (&mut pv as *mut CUdeviceptr).cast(),
                            (&mut cd as *mut i32).cast(),
                            (&mut dc as *mut i32).cast(),
                            (&mut kd as *mut i32).cast(),
                        ];
                        // SAFETY: Matches `gdn_conv1d_silu_split_step_f32`.
                        unsafe {
                            self.ctx.launch(
                                "gdn_conv1d_silu_split_step_f32",
                                self.kernels.gdn_conv1d_silu_split_step_f32,
                                ((conv_dim as u32).div_ceil(256), 1, 1),
                                (256, 1, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }
                    // 2. Per-head Q (`* 1/sqrt(S)`) & K L2 norm
                    {
                        let mut pq = self.q_buf.device_ptr();
                        let mut pk = self.k_buf.device_ptr();
                        let mut sd = s_dim as i32;
                        let mut eps = self.cfg.rms_eps;
                        let mut args: [*mut c_void; 4] = [
                            (&mut pq as *mut CUdeviceptr).cast(),
                            (&mut pk as *mut CUdeviceptr).cast(),
                            (&mut sd as *mut i32).cast(),
                            (&mut eps as *mut f32).cast(),
                        ];
                        // SAFETY: Matches `gdn_l2_norm_qk_f32`.
                        unsafe {
                            self.ctx.launch(
                                "gdn_l2_norm_qk_f32",
                                self.kernels.gdn_l2_norm_qk_f32,
                                (hk as u32, 1, 1),
                                (32, 1, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }
                    // 3. Gates `g` and `beta`
                    {
                        let mut pa = a_t;
                        let mut pb = b_t;
                        let mut psa = ssm_a.device_ptr();
                        let mut pdb = ssm_dt_bias.device_ptr();
                        let mut pg = self.gdn_g_buf.device_ptr();
                        let mut pbo = self.gdn_b_buf.device_ptr();
                        let mut chv = hv as i32;
                        let mut args: [*mut c_void; 7] = [
                            (&mut pa as *mut CUdeviceptr).cast(),
                            (&mut pb as *mut CUdeviceptr).cast(),
                            (&mut psa as *mut CUdeviceptr).cast(),
                            (&mut pdb as *mut CUdeviceptr).cast(),
                            (&mut pg as *mut CUdeviceptr).cast(),
                            (&mut pbo as *mut CUdeviceptr).cast(),
                            (&mut chv as *mut i32).cast(),
                        ];
                        // SAFETY: Matches `gdn_gates_f32`.
                        unsafe {
                            self.ctx.launch(
                                "gdn_gates_f32",
                                self.kernels.gdn_gates_f32,
                                ((hv as u32).div_ceil(64), 1, 1),
                                (64, 1, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }
                    // 4. Warp-sharded GDN state recurrence (`vh % HK`)
                    {
                        let mut pst = ssm_state.device_ptr();
                        let mut pq = self.q_buf.device_ptr();
                        let mut pk = self.k_buf.device_ptr();
                        let mut pv = self.hb_buf.device_ptr();
                        let mut pg = self.gdn_g_buf.device_ptr();
                        let mut pb = self.gdn_b_buf.device_ptr();
                        let mut py = out_t;
                        let mut chk = hk as i32;
                        let mut chv = hv as i32;
                        let mut csd = s_dim as i32;
                        let mut cdv = dv_dim as i32;
                        let mut args: [*mut c_void; 11] = [
                            (&mut pst as *mut CUdeviceptr).cast(),
                            (&mut pq as *mut CUdeviceptr).cast(),
                            (&mut pk as *mut CUdeviceptr).cast(),
                            (&mut pv as *mut CUdeviceptr).cast(),
                            (&mut pg as *mut CUdeviceptr).cast(),
                            (&mut pb as *mut CUdeviceptr).cast(),
                            (&mut py as *mut CUdeviceptr).cast(),
                            (&mut chk as *mut i32).cast(),
                            (&mut chv as *mut i32).cast(),
                            (&mut csd as *mut i32).cast(),
                            (&mut cdv as *mut i32).cast(),
                        ];
                        // SAFETY: Matches `gdn_recurrence_step_f32`.
                        unsafe {
                            self.ctx.launch(
                                "gdn_recurrence_step_f32",
                                self.kernels.gdn_recurrence_step_f32,
                                (hv as u32, (dv_dim as u32).div_ceil(4), 1),
                                (32, 4, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }
                    // Sigmoid-gated RMSNorm (`qwen4exp.cpp:667`)!
                    {
                        let mut po = out_t;
                        let mut pz = z_t;
                        let mut pn = ssm_norm.device_ptr();
                        let mut dv = dv_dim as i32;
                        let mut eps = self.cfg.rms_eps;
                        let mut args: [*mut c_void; 5] = [
                            (&mut po as *mut CUdeviceptr).cast(),
                            (&mut pz as *mut CUdeviceptr).cast(),
                            (&mut pn as *mut CUdeviceptr).cast(),
                            (&mut dv as *mut i32).cast(),
                            (&mut eps as *mut f32).cast(),
                        ];
                        // SAFETY: Matches `gdn_rms_norm_gated_sigmoid_f32`.
                        unsafe {
                            self.ctx.launch(
                                "gdn_rms_norm_gated_sigmoid_f32",
                                self.qk.gdn_norm_gated_sigmoid,
                                (hv as u32, 1, 1),
                                (32, 1, 1),
                                0,
                                &mut args,
                            )?;
                        }
                    }
                }

                self.kernels.launch_linear(
                    &self.ctx,
                    w_out,
                    self.v_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    false,
                    self.sub_out_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
            }
            CudaQwen4Attn::FullQsa {
                q_norm,
                k_norm,
                wqg,
                wk,
                wv,
                wo,
                idx_q_proj,
                idx_k_proj,
                idx_q_norm,
                idx_k_norm,
                pool_ratio,
                k_cache,
                v_cache,
                idx_k_cache,
                idx_pooled,
            } => {
                let any_q8 = is_q8_1_mmvq_dtype(wqg.dtype)
                    || is_q8_1_mmvq_dtype(wk.dtype)
                    || is_q8_1_mmvq_dtype(wv.dtype);
                if can_share_q8 && any_q8 {
                    self.kernels.launch_quantize_q8_1(
                        &self.ctx,
                        self.cur_buf.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        d,
                        n_tok,
                    )?;
                }
                self.kernels.launch_linear(
                    &self.ctx,
                    wqg,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && any_q8 && is_q8_1_mmvq_dtype(wqg.dtype),
                    self.qkv_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    wk,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && any_q8 && is_q8_1_mmvq_dtype(wk.dtype),
                    self.k_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    wv,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && any_q8 && is_q8_1_mmvq_dtype(wv.dtype),
                    self.v_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    idx_q_proj,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && any_q8 && is_q8_1_mmvq_dtype(idx_q_proj.dtype),
                    self.idx_q_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    idx_k_proj,
                    self.cur_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    can_share_q8 && any_q8 && is_q8_1_mmvq_dtype(idx_k_proj.dtype),
                    self.idx_k_buf.device_ptr(),
                    n_tok,
                    false,
                )?;

                // Indexer Q norm+RoPE & raw K F16 cache + block-`pool_ratio` mean pool
                {
                    let mut pq = self.idx_q_buf.device_ptr();
                    let mut pk = self.idx_k_buf.device_ptr();
                    let mut pqn = idx_q_norm.device_ptr();
                    let mut pkn = idx_k_norm.device_ptr();
                    let mut pif = self.rope_inv_freqs.device_ptr();
                    let mut pkc = idx_k_cache.device_ptr();
                    let mut ppl = idx_pooled.device_ptr();
                    let mut nh = idx_nh as i32;
                    let mut idm = idx_d as i32;
                    let mut rd = self.cfg.rope_dim as i32;
                    let mut pr = *pool_ratio as i32;
                    let mut eps = self.cfg.rms_eps;
                    let mut sp = start_pos as i32;
                    let mut pp = pos_dev_ptr;
                    let mut args: [*mut c_void; 14] = [
                        (&mut pq as *mut CUdeviceptr).cast(),
                        (&mut pk as *mut CUdeviceptr).cast(),
                        (&mut pqn as *mut CUdeviceptr).cast(),
                        (&mut pkn as *mut CUdeviceptr).cast(),
                        (&mut pif as *mut CUdeviceptr).cast(),
                        (&mut pkc as *mut CUdeviceptr).cast(),
                        (&mut ppl as *mut CUdeviceptr).cast(),
                        (&mut nh as *mut i32).cast(),
                        (&mut idm as *mut i32).cast(),
                        (&mut rd as *mut i32).cast(),
                        (&mut pr as *mut i32).cast(),
                        (&mut eps as *mut f32).cast(),
                        (&mut sp as *mut i32).cast(),
                        (&mut pp as *mut CUdeviceptr).cast(),
                    ];
                    // SAFETY: Matches `qwen4exp_qsa_idx_prep_pool_f32`.
                    unsafe {
                        self.ctx.launch(
                            "qwen4exp_qsa_idx_prep_pool_f32",
                            self.qk.qsa_idx_prep_pool,
                            (n_tok as u32, (idx_nh + 1) as u32, 1),
                            (128, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }

                // Split interleaved `[q_h, gate_h]` per head
                {
                    let total = (n_tok * q_dim) as u32;
                    let mut pqg = self.qkv_buf.device_ptr();
                    let mut pq = self.q_buf.device_ptr();
                    let mut pg = self.z_buf.device_ptr();
                    let mut nh = self.cfg.n_head as i32;
                    let mut hd = self.cfg.head_dim as i32;
                    let mut nt = n_tok as i32;
                    let mut args: [*mut c_void; 6] = [
                        (&mut pqg as *mut CUdeviceptr).cast(),
                        (&mut pq as *mut CUdeviceptr).cast(),
                        (&mut pg as *mut CUdeviceptr).cast(),
                        (&mut nh as *mut i32).cast(),
                        (&mut hd as *mut i32).cast(),
                        (&mut nt as *mut i32).cast(),
                    ];
                    // SAFETY: Matches `split_qwen35_q_gate_f32`.
                    unsafe {
                        self.ctx.launch(
                            "split_qwen35_q_gate_f32",
                            self.kernels.split_qwen35_q_gate_f32,
                            (total.div_ceil(256), 1, 1),
                            (256, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }
                self.kernels.launch_qk_rms_norm(
                    &self.ctx,
                    self.q_buf.device_ptr(),
                    q_norm.device_ptr(),
                    self.cfg.n_head,
                    self.cfg.head_dim,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
                self.kernels.launch_qk_rms_norm(
                    &self.ctx,
                    self.k_buf.device_ptr(),
                    k_norm.device_ptr(),
                    self.cfg.n_head_kv,
                    self.cfg.head_dim,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
                self.kernels.launch_rope_qk(
                    &self.ctx,
                    self.q_buf.device_ptr(),
                    self.k_buf.device_ptr(),
                    self.rope_inv_freqs.device_ptr(),
                    self.cfg.n_head,
                    self.cfg.n_head_kv,
                    self.cfg.head_dim,
                    self.cfg.rope_dim,
                    2, // NeoX partial RoPE
                    start_pos,
                    pos_dev_ptr,
                    n_tok,
                )?;

                let top_blocks = self
                    .cfg
                    .indexer_top_k
                    .checked_div(*pool_ratio)
                    .map_or(i32::MAX, |v| v as i32);
                {
                    let mut pq = self.idx_q_buf.device_ptr();
                    let mut ppl = idx_pooled.device_ptr();
                    let mut psc = self.qsa_scores_buf.device_ptr();
                    let mut pal = self.qsa_allowed_buf.device_ptr();
                    let mut nh = idx_nh as i32;
                    let mut idm = idx_d as i32;
                    let mut pr = *pool_ratio as i32;
                    let mut tb = top_blocks;
                    let mut mb = self.max_qsa_blocks as i32;
                    let mut sp = start_pos as i32;
                    let mut pp = pos_dev_ptr;
                    let mut args: [*mut c_void; 11] = [
                        (&mut pq as *mut CUdeviceptr).cast(),
                        (&mut ppl as *mut CUdeviceptr).cast(),
                        (&mut psc as *mut CUdeviceptr).cast(),
                        (&mut pal as *mut CUdeviceptr).cast(),
                        (&mut nh as *mut i32).cast(),
                        (&mut idm as *mut i32).cast(),
                        (&mut pr as *mut i32).cast(),
                        (&mut tb as *mut i32).cast(),
                        (&mut mb as *mut i32).cast(),
                        (&mut sp as *mut i32).cast(),
                        (&mut pp as *mut CUdeviceptr).cast(),
                    ];
                    // SAFETY: Matches `qwen4exp_qsa_select_blocks_f32`.
                    unsafe {
                        self.ctx.launch(
                            "qwen4exp_qsa_select_blocks_f32",
                            self.qk.qsa_select_blocks,
                            (n_tok as u32, 1, 1),
                            (128, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }

                let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();
                let is_q8_kv = self.kv_type == KvType::Q8_0;
                if is_q8_kv {
                    self.kernels.launch_write_kv_q8_0(
                        &self.ctx,
                        self.k_buf.device_ptr(),
                        self.v_buf.device_ptr(),
                        k_cache.device_ptr(),
                        v_cache.device_ptr(),
                        kv_dim,
                        start_pos,
                        pos_dev_ptr,
                        n_tok,
                    )?;
                } else {
                    self.kernels.launch_write_kv_f16(
                        &self.ctx,
                        self.k_buf.device_ptr(),
                        self.v_buf.device_ptr(),
                        k_cache.device_ptr(),
                        v_cache.device_ptr(),
                        kv_dim,
                        start_pos,
                        pos_dev_ptr,
                        n_tok,
                    )?;
                }

                {
                    let mut pq = self.q_buf.device_ptr();
                    let mut pkc = k_cache.device_ptr();
                    let mut pvc = v_cache.device_ptr();
                    let mut pal = self.qsa_allowed_buf.device_ptr();
                    let mut po = self.qkv_buf.device_ptr();
                    let mut nh = self.cfg.n_head as i32;
                    let mut nhkv = self.cfg.n_head_kv as i32;
                    let mut hd = self.cfg.head_dim as i32;
                    let mut pr = *pool_ratio as i32;
                    let mut tb = top_blocks;
                    let mut mb = self.max_qsa_blocks as i32;
                    let mut sc = scale;
                    let mut sp = start_pos as i32;
                    let mut pp = pos_dev_ptr;
                    let mut args: [*mut c_void; 14] = [
                        (&mut pq as *mut CUdeviceptr).cast(),
                        (&mut pkc as *mut CUdeviceptr).cast(),
                        (&mut pvc as *mut CUdeviceptr).cast(),
                        (&mut pal as *mut CUdeviceptr).cast(),
                        (&mut po as *mut CUdeviceptr).cast(),
                        (&mut nh as *mut i32).cast(),
                        (&mut nhkv as *mut i32).cast(),
                        (&mut hd as *mut i32).cast(),
                        (&mut pr as *mut i32).cast(),
                        (&mut tb as *mut i32).cast(),
                        (&mut mb as *mut i32).cast(),
                        (&mut sc as *mut f32).cast(),
                        (&mut sp as *mut i32).cast(),
                        (&mut pp as *mut CUdeviceptr).cast(),
                    ];
                    let (func, kname) = if is_q8_kv {
                        (
                            self.qk.qsa_sparse_attn_q8_0,
                            "qwen4exp_qsa_sparse_attn_q8_0_kv",
                        )
                    } else {
                        (
                            self.qk.qsa_sparse_attn_f16,
                            "qwen4exp_qsa_sparse_attn_f16_kv",
                        )
                    };
                    // SAFETY: Matches `qwen4exp_qsa_sparse_attn_{f16,q8_0}_kv`.
                    unsafe {
                        self.ctx.launch(
                            kname,
                            func,
                            (self.cfg.n_head as u32, n_tok as u32, 1),
                            (32, 4, 1),
                            0,
                            &mut args,
                        )?;
                    }
                }

                self.kernels.launch_sigmoid_gate_mul(
                    &self.ctx,
                    self.qkv_buf.device_ptr(),
                    self.z_buf.device_ptr(),
                    n_tok * q_dim,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    wo,
                    self.qkv_buf.device_ptr(),
                    self.xq_scratch.device_ptr(),
                    false,
                    self.sub_out_buf.device_ptr(),
                    n_tok,
                    false,
                )?;
            }
        }

        self.launch_hc_combine_at(res_hc_ptr, self.sub_out_buf.device_ptr(), n_tok)?;

        // 2. MoE FFN + gated shared expert
        self.launch_hc_mix_at(
            res_hc_ptr,
            &l.hc_ffn,
            self.cur_buf.device_ptr(),
            Some(self.hc_inject_buf.device_ptr()),
            n_tok,
        )?;
        l.moe.encode(
            &self.ctx,
            &self.kernels,
            &self.moe_scratch,
            self.xq_scratch.device_ptr(),
            self.hb_buf.device_ptr(),
            self.hb2_buf.device_ptr(),
            self.cur_buf.device_ptr(),
            self.sub_out_buf.device_ptr(),
            d,
            n_tok,
        )?;
        self.launch_hc_combine_at(res_hc_ptr, self.sub_out_buf.device_ptr(), n_tok)
    }

    fn encode_chunk(
        &self,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        preloaded_embed: bool,
        compute_last_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), CudaError> {
        self.encode_chunk_impl(
            n_tok,
            start_pos,
            pos_dev_ptr,
            preloaded_embed,
            compute_last_logits,
            compute_argmax,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_chunk_impl(
        &self,
        n_tok: usize,
        start_pos: usize,
        pos_dev_ptr: CUdeviceptr,
        preloaded_embed: bool,
        compute_last_logits: bool,
        compute_argmax: bool,
        save_spec_state: bool,
    ) -> Result<(), CudaError> {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * d;

        if !preloaded_embed {
            self.kernels.launch_embed(
                &self.ctx,
                &self.token_embd,
                self.token_ids_buf.device_ptr(),
                self.cur_buf.device_ptr(),
                n_tok,
                1.0,
            )?;
        }

        // Initialize 4-stream `res_hc [n_tok, hc, d]`
        {
            let total = (n_tok * hc_dim) as u32;
            let mut px = self.cur_buf.device_ptr();
            let mut pr = self.res_hc_buf.device_ptr();
            let mut cd = d as i32;
            let mut ch = hc as i32;
            let mut nt = n_tok as i32;
            let mut args: [*mut c_void; 5] = [
                (&mut px as *mut CUdeviceptr).cast(),
                (&mut pr as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut ch as *mut i32).cast(),
                (&mut nt as *mut i32).cast(),
            ];
            // SAFETY: Matches `qwen4exp_init_hc_f32`.
            unsafe {
                self.ctx.launch(
                    "qwen4exp_init_hc_f32",
                    self.qk.init_hc,
                    (total.div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        let mut ple_ordinal = 0usize;
        for l in &self.layers {
            self.encode_one_qwen4_layer(
                l,
                self.res_hc_buf.device_ptr(),
                n_tok,
                start_pos,
                pos_dev_ptr,
                save_spec_state,
                &mut ple_ordinal,
            )?;
        }

        if let Some(ref mtp) = self.mtp {
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                mtp.trunk_hc_hist.device_ptr(),
                self.res_hc_buf.device_ptr(),
                n_tok * hc_dim,
                n_tok * hc_dim,
                n_tok * hc_dim,
                1,
            )?;
        }

        // Output head HC Mix + `output.weight`
        if save_spec_state {
            self.launch_hc_mix(&self.hc_head, self.cur_buf.device_ptr(), None, n_tok)?;
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                self.cur_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.logits_dev.device_ptr(),
                n_tok,
                false,
            )?;
            self.kernels.launch_argmax_rows(
                &self.ctx,
                self.logits_dev.device_ptr(),
                self.argmax_dev.device_ptr(),
                self.cfg.vocab_size,
                n_tok,
            )?;
        } else if compute_last_logits {
            self.launch_hc_mix(&self.hc_head, self.cur_buf.device_ptr(), None, n_tok)?;
            let last_x = self.cur_buf.device_ptr() + ((n_tok - 1) * d * 4) as u64;
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                last_x,
                self.xq_scratch.device_ptr(),
                false,
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

    fn encode_mtp_chunk(
        &self,
        mtp: &CudaQwen4Mtp,
        seeds_ptr: CUdeviceptr,
        tokens_ptr: CUdeviceptr,
        n_tok: usize,
        start_pos: usize,
        argmax_out_ptr: CUdeviceptr,
    ) -> Result<(), CudaError> {
        if n_tok == 0 {
            return Ok(());
        }
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * d;

        // 1. Token embedding + `mtp.pre_fc_norm_embedding` + `mtp.fc_embedding` -> `mtp_e2_buf [n_tok, d]`
        self.kernels.launch_embed(
            &self.ctx,
            &self.token_embd,
            tokens_ptr,
            self.cur_buf.device_ptr(),
            n_tok,
            1.0,
        )?;
        self.kernels.launch_rms_norm(
            &self.ctx,
            self.cur_buf.device_ptr(),
            mtp.norm_emb.device_ptr(),
            self.sub_out_buf.device_ptr(),
            d,
            n_tok,
            self.cfg.rms_eps,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &mtp.fc_emb,
            self.sub_out_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            mtp.mtp_e2_buf.device_ptr(),
            n_tok,
            false,
        )?;

        // 2. Full 4-stream `mtp.pre_fc_norm_hidden` (`[n_tok, hc_dim]`) + per-stream `mtp.fc_hidden` (`[n_tok * hc, d]`)
        self.kernels.launch_rms_norm(
            &self.ctx,
            seeds_ptr,
            mtp.norm_hid.device_ptr(),
            mtp.mtp_hn_buf.device_ptr(),
            hc_dim,
            n_tok,
            self.cfg.rms_eps,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &mtp.fc_hid,
            mtp.mtp_hn_buf.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            mtp.res_hc_buf.device_ptr(),
            n_tok * hc,
            false,
        )?;

        // 3. Broadcast-add `mtp_e2_buf [n_tok, d]` across all 4 streams of `mtp.res_hc_buf [n_tok, hc, d]`
        {
            let mut pr = mtp.res_hc_buf.device_ptr();
            let mut pe = mtp.mtp_e2_buf.device_ptr();
            let mut cd = d as i32;
            let mut chd = hc_dim as i32;
            let mut args: [*mut c_void; 4] = [
                (&mut pr as *mut CUdeviceptr).cast(),
                (&mut pe as *mut CUdeviceptr).cast(),
                (&mut cd as *mut i32).cast(),
                (&mut chd as *mut i32).cast(),
            ];
            // SAFETY: Matches `qwen4exp_mtp_add_streams_f32`.
            unsafe {
                self.ctx.launch(
                    "qwen4exp_mtp_add_streams_f32",
                    self.qk.mtp_add_streams,
                    ((hc_dim as u32).div_ceil(256), n_tok as u32, 1),
                    (256, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        // 4. Single MTP QSA + 512-expert MoE + shared-expert layer
        let mut dummy_ple = 0usize;
        self.encode_one_qwen4_layer(
            &mtp.layer,
            mtp.res_hc_buf.device_ptr(),
            n_tok,
            start_pos,
            0,
            false,
            &mut dummy_ple,
        )?;

        // 5. Optional draft head (`mtp.hyper_connection_mixer` + trunk `output.weight` + `argmax`)
        if argmax_out_ptr != 0 {
            let last_hc_ptr = mtp.res_hc_buf.device_ptr() + ((n_tok - 1) * hc_dim * 4) as u64;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                mtp.last_mtp_hc.device_ptr(),
                last_hc_ptr,
                hc_dim,
                hc_dim,
                hc_dim,
                1,
            )?;
            self.launch_hc_mix_at(last_hc_ptr, &mtp.mix_hc, self.cur_buf.device_ptr(), None, 1)?;
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                self.cur_buf.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.logits_dev.device_ptr(),
                1,
                false,
            )?;
            self.kernels.launch_argmax(
                &self.ctx,
                self.logits_dev.device_ptr(),
                argmax_out_ptr,
                self.cfg.vocab_size,
            )?;
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
        let hc_dim = self.cfg.hc_count * self.cfg.n_embd;
        if pos > 0 {
            if let Some(ref mtp) = self.mtp
                && mtp.n_fix == MAX_SPEC_WINDOW
            {
                let flush_n = mtp.n_fix;
                self.pinned_ids.write_u32_at(0, &mtp.fix_tokens[..flush_n]);
                self.token_ids_buf
                    .copy_from_pinned_async(0, &self.pinned_ids, 0, flush_n * 4)?;
                self.encode_mtp_chunk(
                    mtp,
                    mtp.fix_seeds.device_ptr(),
                    self.token_ids_buf.device_ptr(),
                    flush_n,
                    (pos - 1).saturating_sub(flush_n),
                    0,
                )?;
            }
            if let Some(ref mut m) = self.mtp {
                if m.n_fix == MAX_SPEC_WINDOW {
                    m.n_fix = 0;
                }
                let idx = m.n_fix;
                m.fix_tokens[idx] = token_id;
                let dst = m.fix_seeds.device_ptr() + (idx * hc_dim * 4) as u64;
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    dst,
                    m.hlast_hc.device_ptr(),
                    hc_dim,
                    hc_dim,
                    hc_dim,
                    1,
                )?;
                m.n_fix = idx + 1;
            }
        } else if let Some(ref mut m) = self.mtp {
            m.n_fix = 0;
        }

        self.pinned_ids.write_u32_at(0, &[token_id]);
        self.pinned_pos.write_u32_at(0, &[pos as u32]);
        self.token_ids_buf
            .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;
        self.pos_buf
            .copy_from_pinned_async(0, &self.pinned_pos, 0, 4)?;
        self.stage_ple_to_gpu(&[token_id], pos)?;
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
                self.encode_chunk(1, 0, pos_ptr, false, compute_logits, compute_argmax)
            })?;
            self.decode_graphs[slot] = Some(g);
        }
        self.decode_graphs[slot]
            .as_ref()
            .expect("captured qwen4exp decode graph")
            .launch()?;

        if let Some(ref m) = self.mtp {
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                m.hlast_hc.device_ptr(),
                m.trunk_hc_hist.device_ptr(),
                hc_dim,
                hc_dim,
                hc_dim,
                1,
            )?;
        }
        Ok(())
    }

    /// Evaluates `[x_tok, drafts...]` (`w = 1 + drafts.len()`) at `pos`, writes the `1 + accepted`
    /// committed greedy predictions into `out_committed`, and rolls back `ple_conv_hist`,
    /// `seq_tokens`, and all `GdnSigmoid` states to `pos + committed` without re-running matmuls.
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
        let start_pos = pos;
        let mut win_buf = [0u32; MAX_SPEC_WINDOW + 1];
        win_buf[0] = x_tok;
        win_buf[1..w].copy_from_slice(&drafts[..d_len]);
        let win_toks = &win_buf[..w];
        let mut picks = [0u32; MAX_SPEC_WINDOW + 1];

        // Snapshot recurrent PLE state; GDN conv/SSM states are snapshotted inline by `launch_gdn_multi_step`.
        self.ple_conv_snap
            .copy_from_device(&self.ple_conv_hist, self.ple_conv_hist.byte_len())?;

        self.pinned_ids.write_u32_at(0, win_toks);
        self.pinned_pos.write_u32_at(0, &[start_pos as u32]);
        self.token_ids_buf
            .copy_from_pinned_async(0, &self.pinned_ids, 0, w * 4)?;
        self.pos_buf
            .copy_from_pinned_async(0, &self.pinned_pos, 0, 4)?;
        self.stage_ple_to_gpu(win_toks, start_pos)?;
        let g_slot = d_len - 1;
        if self.verify_graphs[g_slot].is_none() {
            let pos_ptr = self.pos_buf.device_ptr();
            let ctx = self.ctx.clone();
            let g = CudaGraphExec::capture(&ctx, || {
                self.encode_chunk_impl(w, 0, pos_ptr, false, false, false, true)
            })?;
            self.verify_graphs[g_slot] = Some(g);
        }
        self.verify_graphs[g_slot]
            .as_ref()
            .expect("captured qwen4exp verify graph")
            .launch()?;
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

        // Record true trunk 4-stream hidden states for neural MTP (`trunk_hc_hist[0..committed]`).
        let hc_dim = self.cfg.hc_count * self.cfg.n_embd;
        if let Some(ref mut mtp) = self.mtp {
            let n_fix = committed.saturating_sub(1);
            mtp.n_fix = n_fix;
            mtp.fix_tokens[..n_fix].copy_from_slice(&picks[..n_fix]);
            if n_fix > 0 {
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    mtp.fix_seeds.device_ptr(),
                    mtp.trunk_hc_hist.device_ptr(),
                    n_fix * hc_dim,
                    n_fix * hc_dim,
                    n_fix * hc_dim,
                    1,
                )?;
            }
            let last_src = mtp.trunk_hc_hist.device_ptr() + ((committed - 1) * hc_dim * 4) as u64;
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                mtp.hlast_hc.device_ptr(),
                last_src,
                hc_dim,
                hc_dim,
                hc_dim,
                1,
            )?;
        }

        if committed < w {
            // Clear rejected `seq_tokens` slots so future PLE N-gram hashes see clean history.
            for p in (start_pos + committed)..(start_pos + w) {
                if p < self.seq_tokens.len() {
                    self.seq_tokens[p] = self.cfg.ple_eos_token_id;
                }
            }
            // Replay exact `committed` PLE dilated-conv steps from `ple_conv_snap` + `spec_ple_normed`.
            self.ple_conv_hist
                .copy_from_device(&self.ple_conv_snap, self.ple_conv_hist.byte_len())?;
            let mut ple_ord = 0usize;
            for l in &self.layers {
                if let Some(ref ple) = l.ple {
                    let src_off = ple_ord * (MAX_SPEC_WINDOW + 1) * hc_dim * 4;
                    let mut pr = self.res_hc_buf.device_ptr();
                    let mut pg = self.ple_gated_buf.device_ptr();
                    let mut pn = self.spec_ple_normed.device_ptr() + src_off as u64;
                    let mut pcw = ple.conv1d.device_ptr();
                    let mut ph = self.ple_conv_hist.device_ptr();
                    let mut chd = hc_dim as i32;
                    let mut ck = self.cfg.ple_conv_kernel as i32;
                    let mut cs = self.cfg.ple_ngram_size as i32;
                    let mut nt = committed as i32;
                    let mut args: [*mut c_void; 9] = [
                        (&mut pr as *mut CUdeviceptr).cast(),
                        (&mut pg as *mut CUdeviceptr).cast(),
                        (&mut pn as *mut CUdeviceptr).cast(),
                        (&mut pcw as *mut CUdeviceptr).cast(),
                        (&mut ph as *mut CUdeviceptr).cast(),
                        (&mut chd as *mut i32).cast(),
                        (&mut ck as *mut i32).cast(),
                        (&mut cs as *mut i32).cast(),
                        (&mut nt as *mut i32).cast(),
                    ];
                    // SAFETY: Replays `committed` steps of `ple_conv_hist` into scratch `res_hc_buf`.
                    unsafe {
                        self.ctx.launch(
                            "qwen4exp_ple_dilated_conv_add_f32",
                            self.qk.ple_dilated_conv,
                            ((hc_dim as u32).div_ceil(256), 1, 1),
                            (256, 1, 1),
                            0,
                            &mut args,
                        )?;
                    }
                    ple_ord += 1;
                }
            }

            // Replay exact `committed` GDN recurrence steps in 1 fused multi-step pass per layer!
            let hk = self.cfg.ssm_n_group;
            let hv = self.cfg.ssm_dt_rank;
            let s_dim = self.cfg.ssm_state_size;
            let dv_dim = self.cfg.ssm_head_v_dim();
            let conv_dim = self.cfg.conv_dim();
            for l in &self.layers {
                if let CudaQwen4Attn::GdnSigmoid {
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    conv_hist,
                    ssm_state,
                    conv_snap,
                    ssm_snap,
                    spec_qkv,
                    spec_alpha,
                    spec_beta,
                    ..
                } = &l.attn
                {
                    self.kernels.launch_gdn_multi_step(
                        &self.ctx,
                        spec_qkv.device_ptr(),
                        0,
                        spec_alpha.device_ptr(),
                        spec_beta.device_ptr(),
                        conv_snap.device_ptr(),
                        conv_hist.device_ptr(),
                        0,
                        conv1d.device_ptr(),
                        ssm_a.device_ptr(),
                        ssm_dt_bias.device_ptr(),
                        ssm_norm.device_ptr(),
                        ssm_snap.device_ptr(),
                        ssm_state.device_ptr(),
                        0,
                        self.q_buf.device_ptr(),
                        self.k_buf.device_ptr(),
                        self.hb_buf.device_ptr(),
                        self.gdn_g_buf.device_ptr(),
                        self.gdn_b_buf.device_ptr(),
                        0, // state-only replay!
                        conv_dim,
                        self.cfg.ssm_conv_kernel,
                        hk,
                        hv,
                        s_dim,
                        dv_dim,
                        self.cfg.rms_eps,
                        1,
                        committed,
                    )?;
                }
            }
            self.ctx.synchronize()?;
        } else if self.mtp.is_some() {
            self.ctx.synchronize()?;
        }

        Ok((committed, accepted))
    }

    /// Runs `1..=draft_max` neural MTP draft steps (`qwen4exp-mtp`) followed by batched 48-layer
    /// `Qwen4Exp` target verification (`1 + draft_max` tokens) in greedy mode.
    pub fn spec_step_mtp_greedy(
        &mut self,
        x_tok: u32,
        pos: usize,
        draft_max: usize,
        out_committed: &mut Vec<u32>,
    ) -> Result<(usize, usize), CudaError> {
        let dmax = draft_max.min(MAX_SPEC_WINDOW);
        if dmax == 0 || self.mtp.is_none() || pos == 0 || pos + 1 + dmax > self.max_ctx {
            let next = self.forward_token_greedy(x_tok, pos)?;
            out_committed.clear();
            out_committed.push(next);
            return Ok((1, 0));
        }

        let hc_dim = self.cfg.hc_count * self.cfg.n_embd;
        let mut drafts = [0u32; MAX_SPEC_WINDOW];
        {
            let mtp = self.mtp.as_ref().expect("mtp checked");
            let n_fix = mtp.n_fix;
            let batch_n = n_fix + 1;
            let mut mtp_toks = [0u32; MAX_SPEC_WINDOW + 1];
            mtp_toks[..n_fix].copy_from_slice(&mtp.fix_tokens[..n_fix]);
            mtp_toks[n_fix] = x_tok;
            self.pinned_ids.write_u32_at(0, &mtp_toks[..batch_n]);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, batch_n * 4)?;

            if n_fix > 0 {
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    mtp.mtp_seeds.device_ptr(),
                    mtp.fix_seeds.device_ptr(),
                    n_fix * hc_dim,
                    n_fix * hc_dim,
                    n_fix * hc_dim,
                    1,
                )?;
            }
            self.kernels.launch_copy_2d_rows(
                &self.ctx,
                mtp.mtp_seeds.device_ptr() + (n_fix * hc_dim * 4) as u64,
                mtp.hlast_hc.device_ptr(),
                hc_dim,
                hc_dim,
                hc_dim,
                1,
            )?;

            // Step 0: repair `n_fix` previously accepted MTP KV slots + draft `d_0` at MTP pos `pos - 1`!
            let mtp_start = (pos - 1).saturating_sub(n_fix);
            self.encode_mtp_chunk(
                mtp,
                mtp.mtp_seeds.device_ptr(),
                self.token_ids_buf.device_ptr(),
                batch_n,
                mtp_start,
                mtp.mtp_draft_ids.device_ptr(),
            )?;

            // Chained steps `k = 1..dmax`: feed `(last_mtp_hc, mtp_draft_ids[k - 1])` directly on GPU!
            for k in 1..dmax {
                let prev_tok_ptr = mtp.mtp_draft_ids.device_ptr() + ((k - 1) * 4) as u64;
                let next_tok_ptr = mtp.mtp_draft_ids.device_ptr() + (k * 4) as u64;
                self.encode_mtp_chunk(
                    mtp,
                    mtp.last_mtp_hc.device_ptr(),
                    prev_tok_ptr,
                    1,
                    pos - 1 + k,
                    next_tok_ptr,
                )?;
            }

            // SAFETY: `[u32]` is POD; `dmax <= MAX_SPEC_WINDOW`.
            let draft_bytes = unsafe {
                core::slice::from_raw_parts_mut(drafts.as_mut_ptr().cast::<u8>(), dmax * 4)
            };
            mtp.mtp_draft_ids.copy_to_bytes(draft_bytes)?;
        }
        if let Some(ref mut mtp) = self.mtp {
            mtp.n_fix = 0;
        }

        self.verify_window_greedy(x_tok, &drafts[..dmax], pos, out_committed)
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
        for &tid in token_ids {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(CudaError::InvalidDimensions(format!(
                    "token_id {tid} >= vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let hc_dim = self.cfg.hc_count * self.cfg.n_embd;
        let mut off = 0usize;
        while off < token_ids.len() {
            let chunk = &token_ids[off..(off + MAX_CUDA_BATCH).min(token_ids.len())];
            let is_last = off + chunk.len() == token_ids.len();
            self.pinned_ids.write_u32_at(0, chunk);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, chunk.len() * 4)?;
            self.stage_ple_to_gpu(chunk, start_pos + off)?;
            self.encode_chunk(
                chunk.len(),
                start_pos + off,
                0,
                false,
                compute_last_logits && is_last,
                false,
            )?;

            if let Some(ref mtp) = self.mtp {
                // Warm MTP QSA KV cache with `(H_{p}, tok_{p + 1})` pairs across prefill chunks!
                if start_pos + off > 0 {
                    // Pair `(hlast_hc = H_{start_pos + off - 1}, chunk[0])` at MTP pos `start_pos + off - 1`.
                    if mtp.n_fix > 0 {
                        let nf = mtp.n_fix;
                        self.pinned_ids.write_u32_at(0, &mtp.fix_tokens[..nf]);
                        self.token_ids_buf.copy_from_pinned_async(
                            0,
                            &self.pinned_ids,
                            0,
                            nf * 4,
                        )?;
                        self.encode_mtp_chunk(
                            mtp,
                            mtp.fix_seeds.device_ptr(),
                            self.token_ids_buf.device_ptr(),
                            nf,
                            (start_pos + off - 1).saturating_sub(nf),
                            0,
                        )?;
                    }
                    self.pinned_ids.write_u32_at(0, &chunk[..1]);
                    self.token_ids_buf
                        .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;
                    self.encode_mtp_chunk(
                        mtp,
                        mtp.hlast_hc.device_ptr(),
                        self.token_ids_buf.device_ptr(),
                        1,
                        start_pos + off - 1,
                        0,
                    )?;
                }
                if chunk.len() >= 2 {
                    self.pinned_ids.write_u32_at(0, &chunk[1..]);
                    self.token_ids_buf.copy_from_pinned_async(
                        0,
                        &self.pinned_ids,
                        0,
                        (chunk.len() - 1) * 4,
                    )?;
                    self.encode_mtp_chunk(
                        mtp,
                        mtp.trunk_hc_hist.device_ptr(),
                        self.token_ids_buf.device_ptr(),
                        chunk.len() - 1,
                        start_pos + off,
                        0,
                    )?;
                }
                let last_src =
                    mtp.trunk_hc_hist.device_ptr() + ((chunk.len() - 1) * hc_dim * 4) as u64;
                self.kernels.launch_copy_2d_rows(
                    &self.ctx,
                    mtp.hlast_hc.device_ptr(),
                    last_src,
                    hc_dim,
                    hc_dim,
                    hc_dim,
                    1,
                )?;
            }
            if let Some(ref mut mtp) = self.mtp {
                mtp.n_fix = 0;
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
        embed: &[f32],
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        if pos >= self.max_ctx || embed.len() != self.cfg.n_embd {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid embed len {} or pos {pos}",
                embed.len()
            )));
        }
        self.cur_buf.copy_from_f32(embed)?;
        self.stage_ple_to_gpu(&[self.cfg.ple_image_token_id], pos)?;
        self.encode_chunk(1, pos, 0, true, compute_logits, false)?;
        if compute_logits {
            self.logits_dev.copy_to_f32(&mut self.logits_host)?;
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
        ev0.record()?;
        for _ in 0..runs {
            self.run_decode_step(1, pos, true, true)?;
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
                name: "full_qwen4exp_decode_step_gpu",
                ms_per_tok: total_ms,
                pct: 100.0,
            }],
        })
    }
}
