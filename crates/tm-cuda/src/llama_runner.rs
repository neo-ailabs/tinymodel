//! CUDA GPU runner for all `LlamaModel` dense & fine-grained MoE architectures:
//! `llama`, `qwen2`, `qwen3`, `smollm3`, `phi3`, `mistral`, `mistral3`, `olmo2`, `olmo3`,
//! `granite`, `exaone`, `olmoe`, `granitemoe`, `qwen2moe`, `qwen3moe`, `bailingmoe2`.

use std::sync::Arc;

use tm_core::{DType, KvType, dequantize_q4_0x4_single_row, dequantize_row_bytes, f32_to_f16};
use tm_cpu::RopeType;
use tm_gguf::GgufFile;
use tm_models::{
    ExpertGatingFunc, LlamaConfig, LlamaModel, MAX_SPEC_WINDOW, PackedExperts, RawMatView,
    SharedExpertWeights, SparseMoeConfig, SparseMoeFfn,
};

use crate::CudaError;
use crate::driver::{
    CUdeviceptr, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec, CudaPinnedBuffer,
};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat, dtype_cuda_tag};

/// Maximum batch chunk processed in a single prefill pass on GPU (`512` tokens).
pub const MAX_CUDA_BATCH: usize = 512;

#[derive(Debug, Clone)]
pub struct CudaStageStat {
    pub name: &'static str,
    pub ms_per_tok: f64,
    pub pct: f64,
}

#[derive(Debug, Clone)]
pub struct CudaProfileReport {
    pub device_name: String,
    pub arch: String,
    pub total_ms_per_tok: f64,
    pub tok_per_sec: f64,
    pub model_weight_bytes: usize,
    pub effective_bw_gbps: f64,
    pub peak_bw_gbps: f64,
    pub bw_utilization_pct: f64,
    pub stages: Vec<CudaStageStat>,
}

/// Uploads a `RawMatView` into CUDA device memory, converting `Q4_0x4` or exotic low-bit types
/// cleanly into `F16` if not directly supported by `cuda_ops.ptx`.
pub(crate) fn upload_mat(ctx: &CudaContext, m: &RawMatView<'_>) -> Result<CudaMat, CudaError> {
    if dtype_cuda_tag(m.dtype).is_ok() {
        let rb = m.dtype.row_bytes(m.cols);
        let buf = Arc::new(ctx.alloc_from_bytes(m.bytes)?);
        return Ok(CudaMat {
            buf,
            offset: 0,
            dtype: m.dtype,
            rows: m.rows,
            cols: m.cols,
            row_bytes: rb,
        });
    }

    // Fallback load-time conversion to F16 (e.g. `Q4_0x4` or `IQ2_XXS`/`TQ2_0`)
    let mut f16_bytes = vec![0u8; m.rows * m.cols * 2];
    let mut row_f32 = vec![0.0f32; m.cols];
    for r in 0..m.rows {
        if m.dtype == DType::Q4_0x4 {
            dequantize_q4_0x4_single_row(m.bytes, m.cols, r, &mut row_f32);
        } else {
            let rb = m.dtype.row_bytes(m.cols);
            dequantize_row_bytes(m.dtype, &m.bytes[r * rb..(r + 1) * rb], &mut row_f32);
        }
        let dst_row = &mut f16_bytes[r * m.cols * 2..(r + 1) * m.cols * 2];
        for (c, &v) in row_f32.iter().enumerate() {
            let h = f32_to_f16(v);
            dst_row[2 * c..2 * c + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    let buf = Arc::new(ctx.alloc_from_bytes(&f16_bytes)?);
    Ok(CudaMat {
        buf,
        offset: 0,
        dtype: DType::F16,
        rows: m.rows,
        cols: m.cols,
        row_bytes: m.cols * 2,
    })
}

pub(crate) struct CudaSharedExpert {
    pub n_ff_shexp: usize,
    pub w_gate: CudaMat,
    pub w_up: CudaMat,
    pub w_down: CudaMat,
    pub w_gate_inp_shexp: Option<CudaMat>,
}

pub(crate) enum CudaPackedExperts {
    Separate {
        gate_exps: Option<CudaBuffer>,
        gate_dtype: i32,
        gate_row_bytes: usize,
        gate_exp_bytes: usize,
        up_exps: CudaBuffer,
        up_dtype: i32,
        up_row_bytes: usize,
        up_exp_bytes: usize,
        down_exps: CudaBuffer,
        down_dtype: i32,
        down_row_bytes: usize,
        down_exp_bytes: usize,
    },
    FusedGateUp {
        gate_up_exps: CudaBuffer,
        gate_up_dtype: i32,
        gate_up_row_bytes: usize,
        gate_up_exp_bytes: usize,
        #[allow(dead_code)]
        half_exp_bytes: usize,
        down_exps: CudaBuffer,
        down_dtype: i32,
        down_row_bytes: usize,
        down_exp_bytes: usize,
    },
}

pub(crate) struct CudaMoeVramCache {
    pub cache_slots: usize,
    pub vram_gate_cache: Option<CudaBuffer>,
    pub vram_up_cache: CudaBuffer,
    pub vram_down_cache: CudaBuffer,
    pub slot_of_exp: CudaBuffer,
    pub exp_of_slot: CudaBuffer,
    pub hit_counts: CudaBuffer,
}

pub(crate) struct CudaMoeLayer {
    pub cfg: SparseMoeConfig,
    pub gate_inp: CudaMat,
    pub gate_inp_bias: Option<CudaBuffer>,
    pub exp_probs_b: Option<CudaBuffer>,
    pub experts: CudaPackedExperts,
    pub vram_cache: Option<CudaMoeVramCache>,
    pub shared_expert: Option<CudaSharedExpert>,
}

pub(crate) struct CudaMoeScratch {
    pub logits: CudaBuffer,
    pub top_ids: CudaBuffer,
    pub top_weights: CudaBuffer,
    pub exp_offs: CudaBuffer,
    pub tile_offs: CudaBuffer,
    pub sorted_ts: CudaBuffer,
    pub exp_gate: CudaBuffer,
    pub exp_up: CudaBuffer,
    pub slot_outs: CudaBuffer,
    pub exp_xq: CudaBuffer,
    pub shexp_gate: CudaBuffer,
}

impl CudaMoeScratch {
    pub(crate) fn alloc(
        ctx: &CudaContext,
        n_embd: usize,
        n_expert: usize,
        n_expert_used: usize,
        max_n_ff_exp: usize,
    ) -> Result<Self, CudaError> {
        let ne = n_expert.max(1);
        let ku = n_expert_used.max(1);
        let n_ff_exp = max_n_ff_exp.max(1);
        let max_q_cols = n_embd.max(n_ff_exp).next_multiple_of(32);
        let rows_slots = MAX_CUDA_BATCH * ku;
        Ok(Self {
            logits: ctx.alloc_zeroed(MAX_CUDA_BATCH * ne * 4)?,
            top_ids: ctx.alloc_zeroed(rows_slots * 4)?,
            top_weights: ctx.alloc_zeroed(rows_slots * 4)?,
            exp_offs: ctx.alloc_zeroed((ne + 2) * 4)?,
            tile_offs: ctx.alloc_zeroed((ne + 2) * 4)?,
            sorted_ts: ctx.alloc_zeroed(rows_slots * 4)?,
            exp_gate: ctx.alloc_zeroed(rows_slots * 2 * n_ff_exp * 4)?,
            exp_up: ctx.alloc_zeroed(rows_slots * n_ff_exp * 4)?,
            slot_outs: ctx.alloc_zeroed(rows_slots * n_embd.max(1) * 4)?,
            exp_xq: ctx.alloc_zeroed(rows_slots * (max_q_cols / 32) * BLOCK_Q8_1_BYTES)?,
            shexp_gate: ctx.alloc_zeroed(MAX_CUDA_BATCH * 4)?,
        })
    }
}

#[inline]
pub(crate) fn is_q8_1_moe_dtype(dtype_tag: i32, cols: usize) -> bool {
    match dtype_tag {
        2 | 3 | 6 | 8 | 20 => cols.is_multiple_of(32),
        40 | 42 => cols.is_multiple_of(64),
        11..=14 | 23 => cols.is_multiple_of(256),
        _ => false,
    }
}

pub(crate) fn moe_weight_bytes(moe: &SparseMoeFfn<'_>) -> usize {
    let mut bytes = moe.gate_inp.bytes.len();
    match &moe.experts {
        PackedExperts::Separate {
            gate_exps,
            up_exps,
            down_exps,
            ..
        } => {
            bytes += gate_exps.map_or(0, |b| b.len()) + up_exps.len() + down_exps.len();
        }
        PackedExperts::FusedGateUp {
            gate_up_exps,
            down_exps,
            ..
        } => {
            bytes += gate_up_exps.len() + down_exps.len();
        }
    }
    if let Some(ref sh) = moe.shared_expert {
        bytes += sh.w_gate.bytes.len() + sh.w_up.bytes.len() + sh.w_down.bytes.len();
        if let Some(ref wgi) = sh.w_gate_inp_shexp {
            bytes += wgi.bytes.len();
        }
    }
    bytes
}

pub(crate) fn moe_expert_only_bytes(moe: &SparseMoeFfn<'_>) -> usize {
    match &moe.experts {
        PackedExperts::Separate {
            gate_exps,
            up_exps,
            down_exps,
            ..
        } => gate_exps.map_or(0, |b| b.len()) + up_exps.len() + down_exps.len(),
        PackedExperts::FusedGateUp {
            gate_up_exps,
            down_exps,
            ..
        } => gate_up_exps.len() + down_exps.len(),
    }
}

/// Decides how many trailing MoE layers (`0..=moe_exp_sizes.len()`) should place their expert
/// weight matrices in zero-copy device-mapped pinned DDR (`P2`: `TM_CUDA_MOE_HOST_OFFLOAD` or
/// automatic fallback when `needed_bytes > free_vram`).
pub(crate) fn plan_moe_host_offload_count(
    free_vram: usize,
    needed_bytes: usize,
    moe_exp_sizes: &[usize],
) -> usize {
    if moe_exp_sizes.is_empty() {
        return 0;
    }
    if let Ok(raw) = std::env::var("TM_CUDA_MOE_HOST_OFFLOAD") {
        let s = raw.trim().to_ascii_lowercase();
        if matches!(s.as_str(), "true" | "all" | "yes") {
            return moe_exp_sizes.len();
        }
        if let Ok(n) = s.parse::<usize>() {
            return n.min(moe_exp_sizes.len());
        }
    }
    if needed_bytes <= free_vram {
        return 0;
    }
    let deficit = needed_bytes - free_vram;
    let mut saved = 0usize;
    let mut count = 0usize;
    for &sz in moe_exp_sizes.iter().rev() {
        saved += sz;
        count += 1;
        if saved >= deficit {
            return count;
        }
    }
    0
}

fn alloc_expert_buf(
    ctx: &CudaContext,
    bytes: &[u8],
    host_mapped: bool,
) -> Result<CudaBuffer, CudaError> {
    if host_mapped {
        ctx.alloc_host_mapped_from_bytes(bytes)
    } else {
        ctx.alloc_from_bytes(bytes)
    }
}

fn plan_moe_vram_cache_slots(
    ctx: &CudaContext,
    moe: &SparseMoeFfn<'_>,
    host_mapped: bool,
) -> usize {
    if !host_mapped || moe.cfg.n_expert == 0 {
        return 0;
    }
    if let Ok(raw) = std::env::var("TM_CUDA_MOE_CACHE_EXPERTS") {
        let s = raw.trim().to_ascii_lowercase();
        if matches!(s.as_str(), "0" | "off" | "false" | "no") {
            return 0;
        }
        if let Ok(n) = s.parse::<usize>() {
            return n.min(moe.cfg.n_expert);
        }
    }
    let exp_total = moe_expert_only_bytes(moe);
    let per_exp = (exp_total / moe.cfg.n_expert.max(1)).max(1);
    let default_slots = (moe.cfg.n_expert / 4).clamp(4, 64).min(moe.cfg.n_expert);
    if let Ok((free_vram, _)) = ctx.mem_info() {
        let reserve = 256 * 1024 * 1024usize;
        if free_vram > reserve {
            let max_fit = (free_vram - reserve) / (per_exp * 4).max(1);
            return default_slots.min(max_fit);
        }
    }
    0
}

fn build_moe_vram_cache(
    ctx: &CudaContext,
    moe: &SparseMoeFfn<'_>,
    host_mapped: bool,
) -> Result<Option<CudaMoeVramCache>, CudaError> {
    let slots = plan_moe_vram_cache_slots(ctx, moe, host_mapped);
    let ne = moe.cfg.n_expert;
    if slots == 0 || slots >= ne {
        return Ok(None);
    }

    // Seed initial cache slots with the highest-bias experts (`exp_probs_b`) or `0..slots`.
    let mut order: Vec<usize> = (0..ne).collect();
    if let Some(ref b) = moe.exp_probs_b
        && b.len() >= ne
    {
        order.sort_by(|&a, &c| b[c].total_cmp(&b[a]).then_with(|| a.cmp(&c)));
    }
    let chosen = &order[..slots];
    let mut slot_of_exp_host = vec![-1i32; ne];
    let mut exp_of_slot_host = vec![0u32; slots];
    for (s, &e) in chosen.iter().enumerate() {
        slot_of_exp_host[e] = s as i32;
        exp_of_slot_host[s] = e as u32;
    }

    let copy_slots = |src: &[u8], exp_bytes: usize| -> Result<CudaBuffer, CudaError> {
        let mut init = Vec::with_capacity(slots * exp_bytes);
        for &e in chosen {
            init.extend_from_slice(&src[e * exp_bytes..(e + 1) * exp_bytes]);
        }
        ctx.alloc_from_bytes(&init)
    };

    let (vram_gate_cache, vram_up_cache, vram_down_cache) = match &moe.experts {
        PackedExperts::Separate {
            gate_exps,
            gate_exp_bytes,
            up_exps,
            up_exp_bytes,
            down_exps,
            down_exp_bytes,
            ..
        } => {
            let vg = match gate_exps {
                Some(ge) => Some(copy_slots(ge, *gate_exp_bytes)?),
                None => None,
            };
            let vu = copy_slots(up_exps, *up_exp_bytes)?;
            let vd = copy_slots(down_exps, *down_exp_bytes)?;
            (vg, vu, vd)
        }
        PackedExperts::FusedGateUp {
            gate_up_exps,
            gate_up_exp_bytes,
            down_exps,
            down_exp_bytes,
            ..
        } => {
            let vgu = copy_slots(gate_up_exps, *gate_up_exp_bytes)?;
            let vd = copy_slots(down_exps, *down_exp_bytes)?;
            (None, vgu, vd)
        }
    };

    // SAFETY: `[i32]` is POD.
    let slot_bytes = unsafe {
        core::slice::from_raw_parts(
            slot_of_exp_host.as_ptr().cast::<u8>(),
            slot_of_exp_host.len() * 4,
        )
    };
    let slot_of_exp = ctx.alloc_from_bytes(slot_bytes)?;
    // SAFETY: `[u32]` is POD.
    let exp_bytes = unsafe {
        core::slice::from_raw_parts(
            exp_of_slot_host.as_ptr().cast::<u8>(),
            exp_of_slot_host.len() * 4,
        )
    };
    let exp_of_slot = ctx.alloc_from_bytes(exp_bytes)?;
    let hit_counts = ctx.alloc_zeroed((ne + 1) * 4)?;

    Ok(Some(CudaMoeVramCache {
        cache_slots: slots,
        vram_gate_cache,
        vram_up_cache,
        vram_down_cache,
        slot_of_exp,
        exp_of_slot,
        hit_counts,
    }))
}

impl CudaMoeLayer {
    #[allow(dead_code)]
    pub(crate) fn upload(
        ctx: &CudaContext,
        moe: &SparseMoeFfn<'_>,
        n_embd: usize,
    ) -> Result<Self, CudaError> {
        Self::upload_with_offload(ctx, moe, n_embd, false)
    }

    pub(crate) fn upload_with_offload(
        ctx: &CudaContext,
        moe: &SparseMoeFfn<'_>,
        n_embd: usize,
        host_mapped: bool,
    ) -> Result<Self, CudaError> {
        let gate_inp = upload_mat(ctx, &moe.gate_inp)?;
        let gate_inp_bias = match &moe.gate_inp_bias {
            Some(b) => Some(ctx.alloc_from_f32(b)?),
            None => None,
        };
        let exp_probs_b = match &moe.exp_probs_b {
            Some(b) => Some(ctx.alloc_from_f32(b)?),
            None => None,
        };
        let n_ff_exp = moe.cfg.n_ff_exp;
        let experts = match &moe.experts {
            PackedExperts::Separate {
                gate_exps,
                gate_dtype,
                gate_exp_bytes,
                up_exps,
                up_dtype,
                up_exp_bytes,
                down_exps,
                down_dtype,
                down_exp_bytes,
            } => CudaPackedExperts::Separate {
                gate_exps: match gate_exps {
                    Some(b) => Some(alloc_expert_buf(ctx, b, host_mapped)?),
                    None => None,
                },
                gate_dtype: dtype_cuda_tag(*gate_dtype)?,
                gate_row_bytes: gate_dtype.row_bytes(n_embd),
                gate_exp_bytes: *gate_exp_bytes,
                up_exps: alloc_expert_buf(ctx, up_exps, host_mapped)?,
                up_dtype: dtype_cuda_tag(*up_dtype)?,
                up_row_bytes: up_dtype.row_bytes(n_embd),
                up_exp_bytes: *up_exp_bytes,
                down_exps: alloc_expert_buf(ctx, down_exps, host_mapped)?,
                down_dtype: dtype_cuda_tag(*down_dtype)?,
                down_row_bytes: down_dtype.row_bytes(n_ff_exp),
                down_exp_bytes: *down_exp_bytes,
            },
            PackedExperts::FusedGateUp {
                gate_up_exps,
                gate_up_dtype,
                gate_up_exp_bytes,
                half_exp_bytes,
                down_exps,
                down_dtype,
                down_exp_bytes,
            } => CudaPackedExperts::FusedGateUp {
                gate_up_exps: alloc_expert_buf(ctx, gate_up_exps, host_mapped)?,
                gate_up_dtype: dtype_cuda_tag(*gate_up_dtype)?,
                gate_up_row_bytes: gate_up_dtype.row_bytes(n_embd),
                gate_up_exp_bytes: *gate_up_exp_bytes,
                half_exp_bytes: *half_exp_bytes,
                down_exps: alloc_expert_buf(ctx, down_exps, host_mapped)?,
                down_dtype: dtype_cuda_tag(*down_dtype)?,
                down_row_bytes: down_dtype.row_bytes(n_ff_exp),
                down_exp_bytes: *down_exp_bytes,
            },
        };
        let vram_cache = build_moe_vram_cache(ctx, moe, host_mapped)?;
        let shared_expert = match &moe.shared_expert {
            Some(SharedExpertWeights {
                n_ff_shexp,
                w_gate,
                w_up,
                w_down,
                w_gate_inp_shexp,
            }) => Some(CudaSharedExpert {
                n_ff_shexp: *n_ff_shexp,
                w_gate: upload_mat(ctx, w_gate)?,
                w_up: upload_mat(ctx, w_up)?,
                w_down: upload_mat(ctx, w_down)?,
                w_gate_inp_shexp: match w_gate_inp_shexp {
                    Some(w) => Some(upload_mat(ctx, w)?),
                    None => None,
                },
            }),
            None => None,
        };
        Ok(Self {
            cfg: moe.cfg.clone(),
            gate_inp,
            gate_inp_bias,
            exp_probs_b,
            experts,
            vram_cache,
            shared_expert,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        ctx: &CudaContext,
        kernels: &CudaKernels,
        scratch: &CudaMoeScratch,
        xq_scratch_ptr: CUdeviceptr,
        hb_ptr: CUdeviceptr,
        hb2_ptr: CUdeviceptr,
        ffn_in_ptr: CUdeviceptr,
        ffn_out_ptr: CUdeviceptr,
        d: usize,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let ne = self.cfg.n_expert;
        let ku = self.cfg.n_expert_used;
        let n_ff_exp = self.cfg.n_ff_exp;
        let rows_slots = n_tok * ku;

        // 1. Router logits `[n_tok, n_expert]` (`force_mmvq = false` so `n_tok > 8` uses Tensor Cores!)
        kernels.launch_linear(
            ctx,
            &self.gate_inp,
            ffn_in_ptr,
            xq_scratch_ptr,
            false,
            scratch.logits.device_ptr(),
            n_tok,
            false,
        )?;
        if let Some(ref b) = self.gate_inp_bias {
            kernels.launch_add_bias(ctx, scratch.logits.device_ptr(), b.device_ptr(), ne, n_tok)?;
        }

        // 2. Parallel 64-thread Top-K routing on GPU (`n_expert <= 512`)
        let gating_code = match self.cfg.gating_func {
            ExpertGatingFunc::Softmax => 1i32,
            ExpertGatingFunc::Sigmoid => 2i32,
            ExpertGatingFunc::SoftmaxWeight => 3i32,
            ExpertGatingFunc::SqrtSoftplus => 4i32,
        };
        kernels.launch_moe_router_topk(
            ctx,
            scratch.logits.device_ptr(),
            self.exp_probs_b.as_ref().map_or(0, |b| b.device_ptr()),
            scratch.top_ids.device_ptr(),
            scratch.top_weights.device_ptr(),
            ne,
            ku,
            gating_code,
            self.cfg.norm_weights,
            self.cfg.norm_weights_eps,
            self.cfg.weights_scale,
            n_tok,
        )?;

        // `P2`: Capture-safe on-GPU frequency-aged hot-expert VRAM cache promotion!
        let (vram_g_ptr, vram_u_ptr, vram_d_ptr, slot_of_exp_ptr) =
            if let Some(ref vc) = self.vram_cache {
                let vg = vc.vram_gate_cache.as_ref().map_or(0, |b| b.device_ptr());
                let vu = vc.vram_up_cache.device_ptr();
                let vd = vc.vram_down_cache.device_ptr();
                let soe = vc.slot_of_exp.device_ptr();
                let (hg, gb, hu, ub, hd, db) = match &self.experts {
                    CudaPackedExperts::Separate {
                        gate_exps,
                        gate_exp_bytes,
                        up_exps,
                        up_exp_bytes,
                        down_exps,
                        down_exp_bytes,
                        ..
                    } => (
                        gate_exps.as_ref().map_or(0, |b| b.device_ptr()),
                        if gate_exps.is_some() {
                            *gate_exp_bytes
                        } else {
                            0
                        },
                        up_exps.device_ptr(),
                        *up_exp_bytes,
                        down_exps.device_ptr(),
                        *down_exp_bytes,
                    ),
                    CudaPackedExperts::FusedGateUp {
                        gate_up_exps,
                        gate_up_exp_bytes,
                        down_exps,
                        down_exp_bytes,
                        ..
                    } => (
                        0,
                        0,
                        gate_up_exps.device_ptr(),
                        *gate_up_exp_bytes,
                        down_exps.device_ptr(),
                        *down_exp_bytes,
                    ),
                };
                kernels.launch_moe_cache_tick_promote(
                    ctx,
                    scratch.top_ids.device_ptr(),
                    vc.hit_counts.device_ptr(),
                    soe,
                    vc.exp_of_slot.device_ptr(),
                    hg,
                    vg,
                    gb,
                    hu,
                    vu,
                    ub,
                    hd,
                    vd,
                    db,
                    rows_slots,
                    ne,
                    vc.cache_slots,
                )?;
                (vg, vu, vd, soe)
            } else {
                (0, 0, 0, 0)
            };

        // 3a. `P0-A`: Grouped `wmma` Tensor Core prefill GEMM when `n_tok >= 16`!
        let use_grouped_wmma = n_tok >= 16
            && ne <= 512
            && ctx.info().has_fp16_wmma()
            && d.is_multiple_of(32)
            && n_ff_exp.is_multiple_of(32);
        if use_grouped_wmma {
            kernels.launch_moe_build_expert_buckets(
                ctx,
                scratch.top_ids.device_ptr(),
                scratch.exp_offs.device_ptr(),
                scratch.tile_offs.device_ptr(),
                scratch.sorted_ts.device_ptr(),
                rows_slots,
                ne,
            )?;
            match &self.experts {
                CudaPackedExperts::Separate {
                    gate_exps,
                    gate_dtype,
                    gate_row_bytes,
                    gate_exp_bytes,
                    up_exps,
                    up_dtype,
                    up_row_bytes,
                    up_exp_bytes,
                    down_exps,
                    down_dtype,
                    down_row_bytes,
                    down_exp_bytes,
                } => {
                    kernels.launch_wmma_moe_grouped(
                        ctx,
                        up_exps.device_ptr(),
                        vram_u_ptr,
                        slot_of_exp_ptr,
                        ffn_in_ptr,
                        scratch.exp_offs.device_ptr(),
                        scratch.tile_offs.device_ptr(),
                        scratch.sorted_ts.device_ptr(),
                        scratch.top_weights.device_ptr(),
                        0,
                        scratch.exp_up.device_ptr(),
                        *up_dtype,
                        n_ff_exp,
                        d,
                        *up_row_bytes,
                        *up_exp_bytes,
                        0,
                        ne,
                        ku,
                        n_tok,
                        0,
                        0,
                    )?;
                    let down_f32_ptr = if let Some(ge) = gate_exps {
                        kernels.launch_wmma_moe_grouped(
                            ctx,
                            ge.device_ptr(),
                            vram_g_ptr,
                            slot_of_exp_ptr,
                            ffn_in_ptr,
                            scratch.exp_offs.device_ptr(),
                            scratch.tile_offs.device_ptr(),
                            scratch.sorted_ts.device_ptr(),
                            scratch.top_weights.device_ptr(),
                            0,
                            scratch.exp_gate.device_ptr(),
                            *gate_dtype,
                            n_ff_exp,
                            d,
                            *gate_row_bytes,
                            *gate_exp_bytes,
                            0,
                            ne,
                            ku,
                            n_tok,
                            0,
                            0,
                        )?;
                        kernels.launch_swiglu(
                            ctx,
                            scratch.exp_gate.device_ptr(),
                            scratch.exp_up.device_ptr(),
                            rows_slots * n_ff_exp,
                        )?;
                        scratch.exp_gate.device_ptr()
                    } else {
                        scratch.exp_up.device_ptr()
                    };
                    kernels.launch_wmma_moe_grouped(
                        ctx,
                        down_exps.device_ptr(),
                        vram_d_ptr,
                        slot_of_exp_ptr,
                        down_f32_ptr,
                        scratch.exp_offs.device_ptr(),
                        scratch.tile_offs.device_ptr(),
                        scratch.sorted_ts.device_ptr(),
                        scratch.top_weights.device_ptr(),
                        0,
                        scratch.slot_outs.device_ptr(),
                        *down_dtype,
                        d,
                        n_ff_exp,
                        *down_row_bytes,
                        *down_exp_bytes,
                        0,
                        ne,
                        ku,
                        n_tok,
                        1,
                        1,
                    )?;
                }
                CudaPackedExperts::FusedGateUp {
                    gate_up_exps,
                    gate_up_dtype,
                    gate_up_row_bytes,
                    gate_up_exp_bytes,
                    down_exps,
                    down_dtype,
                    down_row_bytes,
                    down_exp_bytes,
                    ..
                } => {
                    kernels.launch_wmma_moe_grouped(
                        ctx,
                        gate_up_exps.device_ptr(),
                        vram_u_ptr,
                        slot_of_exp_ptr,
                        ffn_in_ptr,
                        scratch.exp_offs.device_ptr(),
                        scratch.tile_offs.device_ptr(),
                        scratch.sorted_ts.device_ptr(),
                        scratch.top_weights.device_ptr(),
                        0,
                        scratch.exp_gate.device_ptr(),
                        *gate_up_dtype,
                        2 * n_ff_exp,
                        d,
                        *gate_up_row_bytes,
                        *gate_up_exp_bytes,
                        0,
                        ne,
                        ku,
                        n_tok,
                        0,
                        0,
                    )?;
                    kernels.launch_swiglu_fused(
                        ctx,
                        scratch.exp_gate.device_ptr(),
                        scratch.exp_up.device_ptr(),
                        n_ff_exp,
                        rows_slots,
                    )?;
                    kernels.launch_wmma_moe_grouped(
                        ctx,
                        down_exps.device_ptr(),
                        vram_d_ptr,
                        slot_of_exp_ptr,
                        scratch.exp_up.device_ptr(),
                        scratch.exp_offs.device_ptr(),
                        scratch.tile_offs.device_ptr(),
                        scratch.sorted_ts.device_ptr(),
                        scratch.top_weights.device_ptr(),
                        0,
                        scratch.slot_outs.device_ptr(),
                        *down_dtype,
                        d,
                        n_ff_exp,
                        *down_row_bytes,
                        *down_exp_bytes,
                        0,
                        ne,
                        ku,
                        n_tok,
                        1,
                        1,
                    )?;
                }
            }
        } else {
            // 3b. `P0-3` + `P1`: Fused dual `Gate + Up + SwiGLU` `__dp4a` decode/verify path
            // supporting heterogeneous `(gate_dtype, up_dtype)` (e.g. `Qwen3.8` `GSQ-RCO`)!
            let need_in_xq = match &self.experts {
                CudaPackedExperts::Separate {
                    gate_exps,
                    gate_dtype,
                    up_dtype,
                    ..
                } => {
                    is_q8_1_moe_dtype(*up_dtype, d)
                        || (gate_exps.is_some() && is_q8_1_moe_dtype(*gate_dtype, d))
                }
                CudaPackedExperts::FusedGateUp { gate_up_dtype, .. } => {
                    is_q8_1_moe_dtype(*gate_up_dtype, d)
                }
            };
            let in_xq_ptr = if need_in_xq {
                kernels.launch_quantize_q8_1(
                    ctx,
                    ffn_in_ptr,
                    scratch.exp_xq.device_ptr(),
                    d,
                    n_tok,
                )?;
                scratch.exp_xq.device_ptr()
            } else {
                0
            };

            match &self.experts {
                CudaPackedExperts::Separate {
                    gate_exps,
                    gate_dtype,
                    gate_row_bytes,
                    gate_exp_bytes,
                    up_exps,
                    up_dtype,
                    up_row_bytes,
                    up_exp_bytes,
                    down_exps,
                    down_dtype,
                    down_row_bytes,
                    down_exp_bytes,
                } => {
                    let down_uses_q8 = is_q8_1_moe_dtype(*down_dtype, n_ff_exp);
                    let (down_xq_ptr, down_f32_ptr) = if let Some(ge) = gate_exps {
                        kernels.launch_moe_indexed_dual_gu_act(
                            ctx,
                            ge.device_ptr(),
                            up_exps.device_ptr(),
                            vram_g_ptr,
                            vram_u_ptr,
                            slot_of_exp_ptr,
                            in_xq_ptr,
                            ffn_in_ptr,
                            scratch.top_ids.device_ptr(),
                            0,
                            0,
                            scratch.exp_up.device_ptr(),
                            *gate_dtype,
                            *up_dtype,
                            n_ff_exp,
                            d,
                            *gate_row_bytes,
                            *up_row_bytes,
                            *gate_exp_bytes,
                            *up_exp_bytes,
                            0,
                            ku,
                            n_tok,
                            0,
                            1.0,
                            0.0,
                        )?;
                        if down_uses_q8 {
                            kernels.launch_quantize_q8_1(
                                ctx,
                                scratch.exp_up.device_ptr(),
                                scratch.exp_xq.device_ptr(),
                                n_ff_exp,
                                rows_slots,
                            )?;
                            (scratch.exp_xq.device_ptr(), scratch.exp_up.device_ptr())
                        } else {
                            (0, scratch.exp_up.device_ptr())
                        }
                    } else {
                        kernels.launch_moe_indexed_gemv_q8_1(
                            ctx,
                            up_exps.device_ptr(),
                            vram_u_ptr,
                            slot_of_exp_ptr,
                            in_xq_ptr,
                            ffn_in_ptr,
                            scratch.top_ids.device_ptr(),
                            scratch.top_weights.device_ptr(),
                            0,
                            scratch.exp_up.device_ptr(),
                            *up_dtype,
                            n_ff_exp,
                            d,
                            *up_row_bytes,
                            *up_exp_bytes,
                            ku,
                            n_tok,
                            0,
                            0,
                        )?;
                        if down_uses_q8 {
                            kernels.launch_quantize_q8_1(
                                ctx,
                                scratch.exp_up.device_ptr(),
                                scratch.exp_xq.device_ptr(),
                                n_ff_exp,
                                rows_slots,
                            )?;
                            (scratch.exp_xq.device_ptr(), scratch.exp_up.device_ptr())
                        } else {
                            (0, scratch.exp_up.device_ptr())
                        }
                    };
                    kernels.launch_moe_indexed_gemv_q8_1(
                        ctx,
                        down_exps.device_ptr(),
                        vram_d_ptr,
                        slot_of_exp_ptr,
                        down_xq_ptr,
                        down_f32_ptr,
                        scratch.top_ids.device_ptr(),
                        scratch.top_weights.device_ptr(),
                        0,
                        scratch.slot_outs.device_ptr(),
                        *down_dtype,
                        d,
                        n_ff_exp,
                        *down_row_bytes,
                        *down_exp_bytes,
                        ku,
                        n_tok,
                        1,
                        1,
                    )?;
                }
                CudaPackedExperts::FusedGateUp {
                    gate_up_exps,
                    gate_up_dtype,
                    gate_up_row_bytes,
                    gate_up_exp_bytes,
                    half_exp_bytes,
                    down_exps,
                    down_dtype,
                    down_row_bytes,
                    down_exp_bytes,
                } => {
                    kernels.launch_moe_indexed_dual_gu_act(
                        ctx,
                        gate_up_exps.device_ptr(),
                        gate_up_exps.device_ptr(),
                        vram_u_ptr,
                        vram_u_ptr,
                        slot_of_exp_ptr,
                        in_xq_ptr,
                        ffn_in_ptr,
                        scratch.top_ids.device_ptr(),
                        0,
                        0,
                        scratch.exp_up.device_ptr(),
                        *gate_up_dtype,
                        *gate_up_dtype,
                        n_ff_exp,
                        d,
                        *gate_up_row_bytes,
                        *gate_up_row_bytes,
                        *gate_up_exp_bytes,
                        *gate_up_exp_bytes,
                        *half_exp_bytes,
                        ku,
                        n_tok,
                        0,
                        1.0,
                        0.0,
                    )?;
                    let down_uses_q8 = is_q8_1_moe_dtype(*down_dtype, n_ff_exp);
                    let down_xq_ptr = if down_uses_q8 {
                        kernels.launch_quantize_q8_1(
                            ctx,
                            scratch.exp_up.device_ptr(),
                            scratch.exp_xq.device_ptr(),
                            n_ff_exp,
                            rows_slots,
                        )?;
                        scratch.exp_xq.device_ptr()
                    } else {
                        0
                    };
                    kernels.launch_moe_indexed_gemv_q8_1(
                        ctx,
                        down_exps.device_ptr(),
                        vram_d_ptr,
                        slot_of_exp_ptr,
                        down_xq_ptr,
                        scratch.exp_up.device_ptr(),
                        scratch.top_ids.device_ptr(),
                        scratch.top_weights.device_ptr(),
                        0,
                        scratch.slot_outs.device_ptr(),
                        *down_dtype,
                        d,
                        n_ff_exp,
                        *down_row_bytes,
                        *down_exp_bytes,
                        ku,
                        n_tok,
                        1,
                        1,
                    )?;
                }
            }
        }

        // Deterministic `0..ku` slot reduction (`[n_tok, ku, d] -> [n_tok, d]`, zero atomicAdd!)
        kernels.launch_moe_reduce_slots(
            ctx,
            scratch.slot_outs.device_ptr(),
            ffn_out_ptr,
            d,
            ku,
            n_tok,
        )?;

        // 4. Optional shared expert (100% stream-ordered, CUDA-Graph safe!)
        if let Some(ref sh) = self.shared_expert {
            let can_share_xq = n_tok <= crate::kernels::MMVQ_MAX_TOK && d.is_multiple_of(32);
            let sh_uses_q8 = crate::kernels::is_q8_1_mmvq_dtype(sh.w_gate.dtype);
            let up_uses_q8 = crate::kernels::is_q8_1_mmvq_dtype(sh.w_up.dtype);
            if can_share_xq && (sh_uses_q8 || up_uses_q8) {
                kernels.launch_quantize_q8_1(ctx, ffn_in_ptr, xq_scratch_ptr, d, n_tok)?;
            }
            kernels.launch_linear(
                ctx,
                &sh.w_gate,
                ffn_in_ptr,
                xq_scratch_ptr,
                can_share_xq && sh_uses_q8,
                hb_ptr,
                n_tok,
                false,
            )?;
            kernels.launch_linear(
                ctx,
                &sh.w_up,
                ffn_in_ptr,
                xq_scratch_ptr,
                can_share_xq && up_uses_q8,
                hb2_ptr,
                n_tok,
                false,
            )?;
            if let Some(ref w_gi) = sh.w_gate_inp_shexp {
                kernels.launch_linear(
                    ctx,
                    w_gi,
                    ffn_in_ptr,
                    xq_scratch_ptr,
                    can_share_xq
                        && (sh_uses_q8 || up_uses_q8)
                        && crate::kernels::is_q8_1_mmvq_dtype(w_gi.dtype),
                    scratch.shexp_gate.device_ptr(),
                    n_tok,
                    true,
                )?;
            }
            let down_uses_q8 = crate::kernels::is_q8_1_mmvq_dtype(sh.w_down.dtype);
            if n_tok <= crate::kernels::MMVQ_MAX_TOK
                && down_uses_q8
                && sh.n_ff_shexp.is_multiple_of(32)
            {
                kernels.launch_swiglu_quantize_q8_1(
                    ctx,
                    hb_ptr,
                    hb2_ptr,
                    xq_scratch_ptr,
                    sh.n_ff_shexp,
                    n_tok,
                )?;
                kernels.launch_linear(
                    ctx,
                    &sh.w_down,
                    hb_ptr,
                    xq_scratch_ptr,
                    true,
                    hb2_ptr,
                    n_tok,
                    false,
                )?;
            } else {
                kernels.launch_swiglu(ctx, hb_ptr, hb2_ptr, n_tok * sh.n_ff_shexp)?;
                kernels.launch_linear(
                    ctx,
                    &sh.w_down,
                    hb_ptr,
                    xq_scratch_ptr,
                    false,
                    hb2_ptr,
                    n_tok,
                    false,
                )?;
            }
            if sh.w_gate_inp_shexp.is_some() {
                kernels.launch_scale_rows_by_sigmoid_scalar(
                    ctx,
                    hb2_ptr,
                    scratch.shexp_gate.device_ptr(),
                    d,
                    n_tok,
                )?;
            }
            kernels.launch_add_scaled(ctx, ffn_out_ptr, hb2_ptr, 1.0, n_tok * d)?;
        }

        Ok(())
    }
}

struct CudaLlamaLayer {
    attn_norm: Option<CudaBuffer>,
    ffn_norm: Option<CudaBuffer>,
    post_attn_norm: Option<CudaBuffer>,
    post_ffn_norm: Option<CudaBuffer>,
    q_norm: Option<(CudaBuffer, usize, usize)>, // (buf, n_heads_for_norm, norm_dim)
    k_norm: Option<(CudaBuffer, usize, usize)>,
    bq: Option<CudaBuffer>,
    bk: Option<CudaBuffer>,
    bv: Option<CudaBuffer>,
    bo: Option<CudaBuffer>,
    b_gate: Option<CudaBuffer>,
    b_up: Option<CudaBuffer>,
    b_down: Option<CudaBuffer>,
    wq: CudaMat,
    wk: CudaMat,
    wv: CudaMat,
    wo: CudaMat,
    w_gate: Option<CudaMat>,
    w_up: Option<CudaMat>,
    w_down: Option<CudaMat>,
    moe: Option<CudaMoeLayer>,
    k_cache: CudaBuffer,
    v_cache: CudaBuffer,
}

/// High-throughput CUDA runner for dense & sparse-MoE `LlamaModel` families.
pub struct LlamaCudaRunner {
    pub ctx: CudaContext,
    pub kernels: CudaKernels,
    pub cfg: LlamaConfig,
    pub max_ctx: usize,
    pub kv_type: KvType,
    pub vram_bytes: usize,
    pub model_weight_bytes: usize,

    token_embd: CudaMat,
    output_norm: CudaBuffer,
    output: CudaMat,
    output_bias: Option<CudaBuffer>,
    rope_inv_freqs: CudaBuffer,
    rope_inv_freqs_swa: CudaBuffer,
    layers: Vec<CudaLlamaLayer>,

    // Pre-allocated Arena scratch on GPU (zero allocations on forward/decode!)
    pinned_ids: CudaPinnedBuffer,
    pinned_pos: CudaPinnedBuffer,
    token_ids_buf: CudaBuffer,
    pos_buf: CudaBuffer,
    decode_graphs: [Option<CudaGraphExec>; 3], // [no_logits, logits, greedy]
    x: CudaBuffer,
    xb: CudaBuffer,
    xb2: CudaBuffer,
    q: CudaBuffer,
    k: CudaBuffer,
    v: CudaBuffer,
    hb: CudaBuffer,
    hb2: CudaBuffer,
    xq_scratch: CudaBuffer,
    moe_scratch: Option<CudaMoeScratch>,
    logits_dev: CudaBuffer,
    argmax_dev: CudaBuffer,
    logits_host: Vec<f32>,
    pub host_offloaded_moe_layers: usize,
}

impl LlamaCudaRunner {
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, CudaError> {
        Self::from_gguf_with_kv_type(gguf, max_ctx, KvType::F16)
    }

    pub fn from_gguf_with_kv_type(
        gguf: &GgufFile,
        max_ctx: usize,
        kv_type: KvType,
    ) -> Result<Self, CudaError> {
        let cpu_m = LlamaModel::from_gguf(gguf)?;
        let cfg = cpu_m.cfg.clone();
        let ctx = CudaContext::new(None)?;
        let kernels = CudaKernels::load(&ctx)?;

        let q_dim = cfg.n_head * cfg.head_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        if kv_type == KvType::Q8_0 && !kv_dim.is_multiple_of(32) {
            return Err(CudaError::InvalidDimensions(format!(
                "Q8_0 KV cache requires kv_dim ({kv_dim}) divisible by 32"
            )));
        }

        let kv_layer_bytes = match kv_type {
            KvType::F16 => max_ctx * kv_dim * 2,
            KvType::Q8_0 => max_ctx * (kv_dim / 32) * 34,
        };
        let total_kv_bytes = cfg.n_layers * 2 * kv_layer_bytes;

        // Estimate weight + KV + scratch VRAM BEFORE allocating (`AGENTS.md` Rule #6)
        let tied_output =
            std::ptr::eq(cpu_m.token_embd.bytes.as_ptr(), cpu_m.output.bytes.as_ptr());
        let mut weight_bytes = cpu_m.token_embd.bytes.len()
            + if tied_output {
                0
            } else {
                cpu_m.output.bytes.len()
            };
        let mut moe_exp_sizes = Vec::new();
        for l in &cpu_m.layers {
            weight_bytes +=
                l.wq.bytes.len() + l.wk.bytes.len() + l.wv.bytes.len() + l.wo.bytes.len();
            weight_bytes += l.w_gate.bytes.len() + l.w_up.bytes.len() + l.w_down.bytes.len();
            if let Some(ref moe) = l.moe {
                weight_bytes += moe_weight_bytes(moe);
                moe_exp_sizes.push(moe_expert_only_bytes(moe));
            }
        }
        let (free_vram, _) = ctx.mem_info()?;
        let needed_bytes = weight_bytes + total_kv_bytes + 64 * 1024 * 1024;
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
        let output_norm = ctx.alloc_from_f32(&cpu_m.output_norm)?;
        let output = if tied_output {
            token_embd.clone()
        } else {
            upload_mat(&ctx, &cpu_m.output)?
        };
        let output_bias = match &cpu_m.output_bias {
            Some(b) => Some(ctx.alloc_from_f32(b)?),
            None => None,
        };

        let inv_f32: Vec<f32> = if max_ctx > cfg.rope_orig_ctx {
            cpu_m
                .rope_inv_freqs_long
                .as_ref()
                .unwrap_or(&cpu_m.rope_inv_freqs)
                .iter()
                .map(|&v| v as f32)
                .collect()
        } else {
            cpu_m.rope_inv_freqs.iter().map(|&v| v as f32).collect()
        };
        let inv_swa_f32: Vec<f32> = cpu_m.rope_inv_freqs_swa.iter().map(|&v| v as f32).collect();
        let rope_inv_freqs = ctx.alloc_from_f32(&inv_f32)?;
        let rope_inv_freqs_swa = ctx.alloc_from_f32(&inv_swa_f32)?;

        let opt_f32 = |v: &Option<Vec<f32>>| -> Result<Option<CudaBuffer>, CudaError> {
            match v {
                Some(s) => Ok(Some(ctx.alloc_from_f32(s)?)),
                None => Ok(None),
            }
        };

        let total_moe_layers = moe_exp_sizes.len();
        let mut moe_seen = 0usize;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in &cpu_m.layers {
            let q_norm = match &l.q_norm {
                Some(qn) => {
                    let (nh, nd) = if qn.len() == q_dim {
                        (1, q_dim)
                    } else {
                        (cfg.n_head, cfg.head_dim)
                    };
                    Some((ctx.alloc_from_f32(qn)?, nh, nd))
                }
                None => None,
            };
            let k_norm = match &l.k_norm {
                Some(kn) => {
                    let (nh, nd) = if kn.len() == kv_dim {
                        (1, kv_dim)
                    } else {
                        (cfg.n_head_kv, cfg.head_dim)
                    };
                    Some((ctx.alloc_from_f32(kn)?, nh, nd))
                }
                None => None,
            };

            let has_dense_ffn = l.w_up.rows > 0;
            let moe = match &l.moe {
                Some(m) => {
                    let host_map = moe_seen + offload_moe_count >= total_moe_layers;
                    moe_seen += 1;
                    Some(CudaMoeLayer::upload_with_offload(
                        &ctx, m, cfg.n_embd, host_map,
                    )?)
                }
                None => None,
            };

            layers.push(CudaLlamaLayer {
                attn_norm: opt_f32(&l.attn_norm)?,
                ffn_norm: opt_f32(&l.ffn_norm)?,
                post_attn_norm: opt_f32(&l.post_attn_norm)?,
                post_ffn_norm: opt_f32(&l.post_ffn_norm)?,
                q_norm,
                k_norm,
                bq: opt_f32(&l.bq)?,
                bk: opt_f32(&l.bk)?,
                bv: opt_f32(&l.bv)?,
                bo: opt_f32(&l.bo)?,
                b_gate: opt_f32(&l.b_gate)?,
                b_up: opt_f32(&l.b_up)?,
                b_down: opt_f32(&l.b_down)?,
                wq: upload_mat(&ctx, &l.wq)?,
                wk: upload_mat(&ctx, &l.wk)?,
                wv: upload_mat(&ctx, &l.wv)?,
                wo: upload_mat(&ctx, &l.wo)?,
                w_gate: if has_dense_ffn {
                    Some(upload_mat(&ctx, &l.w_gate)?)
                } else {
                    None
                },
                w_up: if has_dense_ffn {
                    Some(upload_mat(&ctx, &l.w_up)?)
                } else {
                    None
                },
                w_down: if has_dense_ffn {
                    Some(upload_mat(&ctx, &l.w_down)?)
                } else {
                    None
                },
                moe,
                k_cache: ctx.alloc_zeroed(kv_layer_bytes)?,
                v_cache: ctx.alloc_zeroed(kv_layer_bytes)?,
            });
        }

        let ff_max = cfg
            .n_ff
            .max(cfg.max_n_ff_exp)
            .max(cfg.max_n_ff_shexp)
            .max(cfg.n_embd);
        let max_cols = cfg.n_embd.max(ff_max).max(q_dim).next_multiple_of(32);

        let pinned_ids = ctx.alloc_pinned(MAX_CUDA_BATCH * 4)?;
        let pinned_pos = ctx.alloc_pinned(4)?;
        let token_ids_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * 4)?;
        let pos_buf = ctx.alloc_zeroed(4)?;
        let x = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_embd * 4)?;
        let xb = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_embd.max(q_dim) * 4)?;
        let xb2 = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_embd * 4)?;
        let q = ctx.alloc_zeroed(MAX_CUDA_BATCH * q_dim * 4)?;
        let k = ctx.alloc_zeroed(MAX_CUDA_BATCH * kv_dim * 4)?;
        let v = ctx.alloc_zeroed(MAX_CUDA_BATCH * kv_dim * 4)?;
        let hb = ctx.alloc_zeroed(MAX_CUDA_BATCH * ff_max.max(1) * 4)?;
        let hb2 = ctx.alloc_zeroed(MAX_CUDA_BATCH * ff_max.max(1) * 4)?;
        let xq_scratch = ctx.alloc_zeroed(MAX_CUDA_BATCH * (max_cols / 32) * BLOCK_Q8_1_BYTES)?;

        let moe_scratch = if cfg.n_expert > 0 {
            Some(CudaMoeScratch::alloc(
                &ctx,
                cfg.n_embd,
                cfg.n_expert,
                cfg.n_expert_used,
                cfg.max_n_ff_exp,
            )?)
        } else {
            None
        };

        let spec_rows = MAX_SPEC_WINDOW + 1;
        let logits_dev = ctx.alloc_zeroed(spec_rows * cfg.vocab_size * 4)?;
        let argmax_dev = ctx.alloc_zeroed(spec_rows * 4)?;
        ctx.synchronize()?;

        let (free_after, _) = ctx.mem_info()?;
        let vram_bytes = free_vram.saturating_sub(free_after);

        Ok(Self {
            ctx,
            kernels,
            cfg: cfg.clone(),
            max_ctx,
            kv_type,
            vram_bytes,
            model_weight_bytes: weight_bytes,
            token_embd,
            output_norm,
            output,
            output_bias,
            rope_inv_freqs,
            rope_inv_freqs_swa,
            layers,
            pinned_ids,
            pinned_pos,
            token_ids_buf,
            pos_buf,
            decode_graphs: [None, None, None],
            x,
            xb,
            xb2,
            q,
            k,
            v,
            hb,
            hb2,
            xq_scratch,
            moe_scratch,
            logits_dev,
            argmax_dev,
            logits_host: vec![0.0f32; cfg.vocab_size],
            host_offloaded_moe_layers: offload_moe_count,
        })
    }

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        for l in &self.layers {
            l.k_cache.zero()?;
            l.v_cache.zero()?;
        }
        self.ctx.synchronize()
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits_host
    }

    #[inline]
    fn rope_type_int(&self, layer_idx: usize) -> i32 {
        if !self.cfg.layer_uses_rope(layer_idx) {
            -1
        } else {
            match self.cfg.rope_type {
                RopeType::Norm => 0,
                RopeType::Neox => 2,
            }
        }
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
        let d = self.cfg.n_embd;
        let q_dim = self.cfg.n_head * self.cfg.head_dim;
        let kv_dim = self.cfg.n_head_kv * self.cfg.head_dim;
        let is_q8_kv = self.kv_type == KvType::Q8_0;
        let can_share_xq = n_tok <= crate::kernels::MMVQ_MAX_TOK && d.is_multiple_of(32);

        // 1. Embedding lookup
        if !preloaded_embed {
            self.kernels.launch_embed(
                &self.ctx,
                &self.token_embd,
                self.token_ids_buf.device_ptr(),
                self.x.device_ptr(),
                n_tok,
                self.cfg.embedding_scale,
            )?;
        }

        // 2. Transformer blocks
        for (il, layer) in self.layers.iter().enumerate() {
            let attn_in_ptr = if let Some(ref an) = layer.attn_norm {
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.x.device_ptr(),
                    an.device_ptr(),
                    self.xb.device_ptr(),
                    d,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
                self.xb.device_ptr()
            } else {
                self.x.device_ptr()
            };

            // Pre-quantize `attn_in` once for Q, K, V if using MMVQ (`n_tok <= 8`)!
            let qkv_uses_q8 = matches!(
                layer.wq.dtype,
                DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
            );
            if can_share_xq && qkv_uses_q8 {
                self.kernels.launch_quantize_q8_1(
                    &self.ctx,
                    attn_in_ptr,
                    self.xq_scratch.device_ptr(),
                    d,
                    n_tok,
                )?;
            }

            self.kernels.launch_linear(
                &self.ctx,
                &layer.wq,
                attn_in_ptr,
                self.xq_scratch.device_ptr(),
                can_share_xq && qkv_uses_q8,
                self.q.device_ptr(),
                n_tok,
                false,
            )?;
            self.kernels.launch_linear(
                &self.ctx,
                &layer.wk,
                attn_in_ptr,
                self.xq_scratch.device_ptr(),
                can_share_xq && qkv_uses_q8,
                self.k.device_ptr(),
                n_tok,
                false,
            )?;
            self.kernels.launch_linear(
                &self.ctx,
                &layer.wv,
                attn_in_ptr,
                self.xq_scratch.device_ptr(),
                can_share_xq && qkv_uses_q8,
                self.v.device_ptr(),
                n_tok,
                false,
            )?;

            if let Some(ref bq) = layer.bq {
                self.kernels.launch_add_bias(
                    &self.ctx,
                    self.q.device_ptr(),
                    bq.device_ptr(),
                    q_dim,
                    n_tok,
                )?;
            }
            if let Some(ref bk) = layer.bk {
                self.kernels.launch_add_bias(
                    &self.ctx,
                    self.k.device_ptr(),
                    bk.device_ptr(),
                    kv_dim,
                    n_tok,
                )?;
            }
            if let Some(ref bv) = layer.bv {
                self.kernels.launch_add_bias(
                    &self.ctx,
                    self.v.device_ptr(),
                    bv.device_ptr(),
                    kv_dim,
                    n_tok,
                )?;
            }

            if let Some((ref qn, nh, nd)) = layer.q_norm {
                self.kernels.launch_qk_rms_norm(
                    &self.ctx,
                    self.q.device_ptr(),
                    qn.device_ptr(),
                    nh,
                    nd,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
            }
            if let Some((ref kn, nh, nd)) = layer.k_norm {
                self.kernels.launch_qk_rms_norm(
                    &self.ctx,
                    self.k.device_ptr(),
                    kn.device_ptr(),
                    nh,
                    nd,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
            }

            let rope_buf = if self.cfg.has_dual_rope_swa() && self.cfg.layer_window(il).is_some() {
                &self.rope_inv_freqs_swa
            } else {
                &self.rope_inv_freqs
            };
            self.kernels.launch_rope_qk(
                &self.ctx,
                self.q.device_ptr(),
                self.k.device_ptr(),
                rope_buf.device_ptr(),
                self.cfg.n_head,
                self.cfg.n_head_kv,
                self.cfg.head_dim,
                self.cfg.rope_dim,
                self.rope_type_int(il),
                start_pos,
                pos_dev_ptr,
                n_tok,
            )?;

            if is_q8_kv {
                self.kernels.launch_write_kv_q8_0(
                    &self.ctx,
                    self.k.device_ptr(),
                    self.v.device_ptr(),
                    layer.k_cache.device_ptr(),
                    layer.v_cache.device_ptr(),
                    kv_dim,
                    start_pos,
                    pos_dev_ptr,
                    n_tok,
                )?;
            } else {
                self.kernels.launch_write_kv_f16(
                    &self.ctx,
                    self.k.device_ptr(),
                    self.v.device_ptr(),
                    layer.k_cache.device_ptr(),
                    layer.v_cache.device_ptr(),
                    kv_dim,
                    start_pos,
                    pos_dev_ptr,
                    n_tok,
                )?;
            }

            let swa_win = self.cfg.layer_window(il).unwrap_or(0);
            self.kernels.launch_flash_attn_gqa(
                &self.ctx,
                self.q.device_ptr(),
                layer.k_cache.device_ptr(),
                layer.v_cache.device_ptr(),
                self.xb.device_ptr(),
                is_q8_kv,
                self.cfg.n_head,
                self.cfg.n_head_kv,
                self.cfg.head_dim,
                start_pos,
                pos_dev_ptr,
                n_tok,
                swa_win,
                self.cfg.attn_scale,
                0.0,
            )?;

            self.kernels.launch_linear(
                &self.ctx,
                &layer.wo,
                self.xb.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.xb2.device_ptr(),
                n_tok,
                false,
            )?;
            if let Some(ref bo) = layer.bo {
                self.kernels.launch_add_bias(
                    &self.ctx,
                    self.xb2.device_ptr(),
                    bo.device_ptr(),
                    d,
                    n_tok,
                )?;
            }
            if let Some(ref pan) = layer.post_attn_norm {
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.xb2.device_ptr(),
                    pan.device_ptr(),
                    self.xb.device_ptr(),
                    d,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    self.x.device_ptr(),
                    self.xb.device_ptr(),
                    self.cfg.residual_scale,
                    n_tok * d,
                )?;
            } else {
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    self.x.device_ptr(),
                    self.xb2.device_ptr(),
                    self.cfg.residual_scale,
                    n_tok * d,
                )?;
            }

            // FFN (Dense SwiGLU or Sparse MoE)
            let ffn_in_ptr = if let Some(ref fn_norm) = layer.ffn_norm {
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.x.device_ptr(),
                    fn_norm.device_ptr(),
                    self.xb.device_ptr(),
                    d,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
                self.xb.device_ptr()
            } else {
                self.x.device_ptr()
            };

            if let Some(ref moe) = layer.moe {
                self.encode_moe_ffn(moe, ffn_in_ptr, self.xb2.device_ptr(), n_tok)?;
            } else {
                let wg = layer.w_gate.as_ref().expect("dense w_gate");
                let wu = layer.w_up.as_ref().expect("dense w_up");
                let wd = layer.w_down.as_ref().expect("dense w_down");
                let ffn_uses_q8 = matches!(
                    wg.dtype,
                    DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
                );
                if can_share_xq && ffn_uses_q8 {
                    self.kernels.launch_quantize_q8_1(
                        &self.ctx,
                        ffn_in_ptr,
                        self.xq_scratch.device_ptr(),
                        d,
                        n_tok,
                    )?;
                }
                self.kernels.launch_linear(
                    &self.ctx,
                    wg,
                    ffn_in_ptr,
                    self.xq_scratch.device_ptr(),
                    can_share_xq && ffn_uses_q8,
                    self.hb.device_ptr(),
                    n_tok,
                    false,
                )?;
                self.kernels.launch_linear(
                    &self.ctx,
                    wu,
                    ffn_in_ptr,
                    self.xq_scratch.device_ptr(),
                    can_share_xq && ffn_uses_q8,
                    self.hb2.device_ptr(),
                    n_tok,
                    false,
                )?;
                if let Some(ref bg) = layer.b_gate {
                    self.kernels.launch_add_bias(
                        &self.ctx,
                        self.hb.device_ptr(),
                        bg.device_ptr(),
                        self.cfg.n_ff,
                        n_tok,
                    )?;
                }
                if let Some(ref bu) = layer.b_up {
                    self.kernels.launch_add_bias(
                        &self.ctx,
                        self.hb2.device_ptr(),
                        bu.device_ptr(),
                        self.cfg.n_ff,
                        n_tok,
                    )?;
                }
                let down_uses_q8 = matches!(
                    wd.dtype,
                    DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
                );
                if n_tok <= crate::kernels::MMVQ_MAX_TOK
                    && down_uses_q8
                    && self.cfg.n_ff.is_multiple_of(32)
                {
                    self.kernels.launch_swiglu_quantize_q8_1(
                        &self.ctx,
                        self.hb.device_ptr(),
                        self.hb2.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        self.cfg.n_ff,
                        n_tok,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        wd,
                        self.hb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        true,
                        self.xb2.device_ptr(),
                        n_tok,
                        false,
                    )?;
                } else {
                    self.kernels.launch_swiglu(
                        &self.ctx,
                        self.hb.device_ptr(),
                        self.hb2.device_ptr(),
                        n_tok * self.cfg.n_ff,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        wd,
                        self.hb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        false,
                        self.xb2.device_ptr(),
                        n_tok,
                        false,
                    )?;
                }
                if let Some(ref bd) = layer.b_down {
                    self.kernels.launch_add_bias(
                        &self.ctx,
                        self.xb2.device_ptr(),
                        bd.device_ptr(),
                        d,
                        n_tok,
                    )?;
                }
            }

            if let Some(ref pfn) = layer.post_ffn_norm {
                self.kernels.launch_rms_norm(
                    &self.ctx,
                    self.xb2.device_ptr(),
                    pfn.device_ptr(),
                    self.xb.device_ptr(),
                    d,
                    n_tok,
                    self.cfg.rms_eps,
                )?;
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    self.x.device_ptr(),
                    self.xb.device_ptr(),
                    self.cfg.residual_scale,
                    n_tok * d,
                )?;
            } else {
                self.kernels.launch_add_scaled(
                    &self.ctx,
                    self.x.device_ptr(),
                    self.xb2.device_ptr(),
                    self.cfg.residual_scale,
                    n_tok * d,
                )?;
            }
        }

        // 3. Final RMSNorm + LM Head on the final token if requested
        if compute_last_logits {
            let last_x = self.x.device_ptr() + ((n_tok - 1) * d * 4) as u64;
            self.kernels.launch_rms_norm(
                &self.ctx,
                last_x,
                self.output_norm.device_ptr(),
                self.xb.device_ptr(),
                d,
                1,
                self.cfg.rms_eps,
            )?;
            self.kernels.launch_linear(
                &self.ctx,
                &self.output,
                self.xb.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.logits_dev.device_ptr(),
                1,
                false,
            )?;
            if let Some(ref ob) = self.output_bias {
                self.kernels.launch_add_bias(
                    &self.ctx,
                    self.logits_dev.device_ptr(),
                    ob.device_ptr(),
                    self.cfg.vocab_size,
                    1,
                )?;
            }
            self.kernels.launch_scale_inplace(
                &self.ctx,
                self.logits_dev.device_ptr(),
                self.cfg.inv_logit_scale,
                self.cfg.vocab_size,
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

    fn encode_moe_ffn(
        &self,
        moe: &CudaMoeLayer,
        ffn_in_ptr: CUdeviceptr,
        ffn_out_ptr: CUdeviceptr,
        n_tok: usize,
    ) -> Result<(), CudaError> {
        let scratch = self.moe_scratch.as_ref().ok_or_else(|| {
            CudaError::InvalidDimensions("MoE layer present without CudaMoeScratch".to_string())
        })?;
        moe.encode(
            &self.ctx,
            &self.kernels,
            scratch,
            self.xq_scratch.device_ptr(),
            self.hb.device_ptr(),
            self.hb2.device_ptr(),
            ffn_in_ptr,
            ffn_out_ptr,
            self.cfg.n_embd,
            n_tok,
        )
    }

    fn can_use_decode_graph(&self) -> bool {
        true
    }

    fn run_decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), CudaError> {
        self.pinned_ids.write_u32_at(0, &[token_id]);
        self.token_ids_buf
            .copy_from_pinned_async(0, &self.pinned_ids, 0, 4)?;
        if self.can_use_decode_graph() {
            self.pinned_pos.write_u32_at(0, &[pos as u32]);
            self.pos_buf
                .copy_from_pinned_async(0, &self.pinned_pos, 0, 4)?;
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
                .expect("captured decode graph")
                .launch()
        } else {
            self.encode_chunk(1, pos, 0, false, compute_logits, compute_argmax)
        }
    }

    /// Evaluates `[x_tok, drafts...]` (`w = 1 + drafts.len() <= MAX_SPEC_WINDOW + 1`) in one pass
    /// and returns `(committed_count, accepted_drafts)` with committed greedy picks in `out_committed`.
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
        self.encode_chunk(w, pos, 0, false, false, false)?;

        let d = self.cfg.n_embd;
        self.kernels.launch_rms_norm(
            &self.ctx,
            self.x.device_ptr(),
            self.output_norm.device_ptr(),
            self.xb.device_ptr(),
            d,
            w,
            self.cfg.rms_eps,
        )?;
        self.kernels.launch_linear(
            &self.ctx,
            &self.output,
            self.xb.device_ptr(),
            self.xq_scratch.device_ptr(),
            false,
            self.logits_dev.device_ptr(),
            w,
            false,
        )?;
        if let Some(ref ob) = self.output_bias {
            self.kernels.launch_add_bias(
                &self.ctx,
                self.logits_dev.device_ptr(),
                ob.device_ptr(),
                self.cfg.vocab_size,
                w,
            )?;
        }
        self.kernels.launch_scale_inplace(
            &self.ctx,
            self.logits_dev.device_ptr(),
            self.cfg.inv_logit_scale,
            w * self.cfg.vocab_size,
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
        Ok((committed, accepted))
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], CudaError> {
        if pos >= self.max_ctx {
            return Err(CudaError::InvalidDimensions(format!(
                "pos {pos} >= max_ctx {}",
                self.max_ctx
            )));
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(CudaError::InvalidDimensions(format!(
                "token_id {token_id} >= vocab_size {}",
                self.cfg.vocab_size
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

    /// Runs a full single-token decode step AND greedy argmax on the GPU, reading back ONLY
    /// 4 bytes (`u32` token ID) over PCIe instead of `vocab_size * 4` bytes!
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
        let end_pos = start_pos + token_ids.len() - 1;
        if end_pos >= self.max_ctx {
            return Err(CudaError::InvalidDimensions(format!(
                "end_pos {end_pos} >= max_ctx {}",
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

        let mut off = 0usize;
        while off < token_ids.len() {
            let b = (token_ids.len() - off).min(MAX_CUDA_BATCH);
            let is_last = off + b == token_ids.len();
            self.pinned_ids.write_u32_at(0, &token_ids[off..off + b]);
            self.token_ids_buf
                .copy_from_pinned_async(0, &self.pinned_ids, 0, b * 4)?;
            self.encode_chunk(
                b,
                start_pos + off,
                0,
                false,
                is_last && compute_last_logits,
                false,
            )?;
            off += b;
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
        if pos >= self.max_ctx {
            return Err(CudaError::InvalidDimensions(format!(
                "pos {pos} >= max_ctx {}",
                self.max_ctx
            )));
        }
        if embed.len() != self.cfg.n_embd {
            return Err(CudaError::InvalidDimensions(format!(
                "forward_embed length {} != n_embd {}",
                embed.len(),
                self.cfg.n_embd
            )));
        }
        self.x.copy_from_f32(embed)?;
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
        self.token_ids_buf.copy_from_u32(&[1])?;
        self.pos_buf.copy_from_u32(&[pos as u32])?;
        ev0.record()?;
        if let Some(ref g) = self.decode_graphs[2] {
            for _ in 0..runs {
                g.launch()?;
            }
        } else {
            for _ in 0..runs {
                self.encode_chunk(1, pos, 0, false, true, true)?;
            }
        }
        ev1.record()?;
        ev1.synchronize()?;
        let total_ms = ev1.elapsed_ms(&ev0)? as f64 / runs as f64;
        let tps = 1000.0 / total_ms.max(1e-9);
        let eff_bw = (self.model_weight_bytes as f64 / 1e9) / (total_ms / 1e3).max(1e-12);
        let peak_bw = self.ctx.info().peak_mem_bw_gbps.max(1.0);

        Ok(CudaProfileReport {
            device_name: self.ctx.info().name.clone(),
            arch: self.cfg.arch.clone(),
            total_ms_per_tok: total_ms,
            tok_per_sec: tps,
            model_weight_bytes: self.model_weight_bytes,
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
