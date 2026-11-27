//! CUDA GPU runner for all `LlamaModel` dense & fine-grained MoE architectures:
//! `llama`, `qwen2`, `qwen3`, `smollm3`, `phi3`, `mistral`, `mistral3`, `olmo2`, `olmo3`,
//! `granite`, `exaone`, `olmoe`, `granitemoe`, `qwen2moe`, `qwen3moe`, `bailingmoe2`.

use std::ffi::c_void;
use std::sync::Arc;

use tm_core::{DType, KvType, dequantize_q4_0x4_single_row, dequantize_row_bytes, f32_to_f16};
use tm_cpu::RopeType;
use tm_gguf::GgufFile;
use tm_models::{
    ExpertGatingFunc, LlamaConfig, LlamaModel, PackedExperts, RawMatView, SharedExpertWeights,
    SparseMoeConfig, SparseMoeFfn,
};

use crate::CudaError;
use crate::driver::{CUdeviceptr, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat, dtype_cuda_tag};

/// Maximum batch chunk processed in a single prefill pass on GPU (`128` tokens).
pub const MAX_CUDA_BATCH: usize = 128;

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

struct CudaSharedExpert {
    n_ff_shexp: usize,
    w_gate: CudaMat,
    w_up: CudaMat,
    w_down: CudaMat,
    w_gate_inp_shexp: Option<CudaMat>,
}

enum CudaPackedExperts {
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

struct CudaMoeLayer {
    cfg: SparseMoeConfig,
    gate_inp: CudaMat,
    gate_inp_bias: Option<CudaBuffer>,
    exp_probs_b: Option<CudaBuffer>,
    experts: CudaPackedExperts,
    shared_expert: Option<CudaSharedExpert>,
}

impl CudaMoeLayer {
    fn upload(ctx: &CudaContext, moe: &SparseMoeFfn<'_>, n_embd: usize) -> Result<Self, CudaError> {
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
                    Some(b) => Some(ctx.alloc_from_bytes(b)?),
                    None => None,
                },
                gate_dtype: dtype_cuda_tag(*gate_dtype)?,
                gate_row_bytes: gate_dtype.row_bytes(n_embd),
                gate_exp_bytes: *gate_exp_bytes,
                up_exps: ctx.alloc_from_bytes(up_exps)?,
                up_dtype: dtype_cuda_tag(*up_dtype)?,
                up_row_bytes: up_dtype.row_bytes(n_embd),
                up_exp_bytes: *up_exp_bytes,
                down_exps: ctx.alloc_from_bytes(down_exps)?,
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
                gate_up_exps: ctx.alloc_from_bytes(gate_up_exps)?,
                gate_up_dtype: dtype_cuda_tag(*gate_up_dtype)?,
                gate_up_row_bytes: gate_up_dtype.row_bytes(n_embd),
                gate_up_exp_bytes: *gate_up_exp_bytes,
                half_exp_bytes: *half_exp_bytes,
                down_exps: ctx.alloc_from_bytes(down_exps)?,
                down_dtype: dtype_cuda_tag(*down_dtype)?,
                down_row_bytes: down_dtype.row_bytes(n_ff_exp),
                down_exp_bytes: *down_exp_bytes,
            },
        };
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
            shared_expert,
        })
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
    moe_logits: CudaBuffer,
    moe_top_ids: CudaBuffer,
    moe_top_weights: CudaBuffer,
    moe_exp_gate: CudaBuffer,
    moe_exp_up: CudaBuffer,
    moe_shexp_gate: CudaBuffer,
    logits_dev: CudaBuffer,
    argmax_dev: CudaBuffer,
    logits_host: Vec<f32>,
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
        for l in &cpu_m.layers {
            weight_bytes +=
                l.wq.bytes.len() + l.wk.bytes.len() + l.wv.bytes.len() + l.wo.bytes.len();
            weight_bytes += l.w_gate.bytes.len() + l.w_up.bytes.len() + l.w_down.bytes.len();
            if let Some(ref moe) = l.moe {
                weight_bytes += moe.gate_inp.bytes.len();
                match &moe.experts {
                    PackedExperts::Separate {
                        gate_exps,
                        up_exps,
                        down_exps,
                        ..
                    } => {
                        weight_bytes +=
                            gate_exps.map_or(0, |b| b.len()) + up_exps.len() + down_exps.len();
                    }
                    PackedExperts::FusedGateUp {
                        gate_up_exps,
                        down_exps,
                        ..
                    } => {
                        weight_bytes += gate_up_exps.len() + down_exps.len();
                    }
                }
            }
        }
        let (free_vram, _) = ctx.mem_info()?;
        let needed_bytes = weight_bytes + total_kv_bytes + 64 * 1024 * 1024;
        if needed_bytes > free_vram {
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
                Some(m) => Some(CudaMoeLayer::upload(&ctx, m, cfg.n_embd)?),
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

        let ne = cfg.n_expert.max(1);
        let ku = cfg.n_expert_used.max(1);
        let n_ff_exp = cfg.max_n_ff_exp.max(1);
        let moe_logits = ctx.alloc_zeroed(MAX_CUDA_BATCH * ne * 4)?;
        let moe_top_ids = ctx.alloc_zeroed(MAX_CUDA_BATCH * ku * 4)?;
        let moe_top_weights = ctx.alloc_zeroed(MAX_CUDA_BATCH * ku * 4)?;
        let moe_exp_gate = ctx.alloc_zeroed(2 * n_ff_exp * 4)?;
        let moe_exp_up = ctx.alloc_zeroed(n_ff_exp * 4)?;
        let moe_shexp_gate = ctx.alloc_zeroed(MAX_CUDA_BATCH * 4)?;

        let logits_dev = ctx.alloc_zeroed(cfg.vocab_size * 4)?;
        let argmax_dev = ctx.alloc_zeroed(4)?;
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
            moe_logits,
            moe_top_ids,
            moe_top_weights,
            moe_exp_gate,
            moe_exp_up,
            moe_shexp_gate,
            logits_dev,
            argmax_dev,
            logits_host: vec![0.0f32; cfg.vocab_size],
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
        let d = self.cfg.n_embd;
        let ne = moe.cfg.n_expert;
        let ku = moe.cfg.n_expert_used;
        let n_ff_exp = moe.cfg.n_ff_exp;

        // 1. Router logits `[n_tok, n_expert]`
        self.kernels.launch_linear(
            &self.ctx,
            &moe.gate_inp,
            ffn_in_ptr,
            self.xq_scratch.device_ptr(),
            false,
            self.moe_logits.device_ptr(),
            n_tok,
            true,
        )?;
        if let Some(ref b) = moe.gate_inp_bias {
            self.kernels.launch_add_bias(
                &self.ctx,
                self.moe_logits.device_ptr(),
                b.device_ptr(),
                ne,
                n_tok,
            )?;
        }

        // 2. Top-K routing on GPU
        let gating_code = match moe.cfg.gating_func {
            ExpertGatingFunc::Softmax => 1i32,
            ExpertGatingFunc::Sigmoid => 2i32,
            ExpertGatingFunc::SoftmaxWeight => 3i32,
            ExpertGatingFunc::SqrtSoftplus => 4i32,
        };
        {
            let mut pl = self.moe_logits.device_ptr();
            let mut pb = moe.exp_probs_b.as_ref().map_or(0, |b| b.device_ptr());
            let mut pids = self.moe_top_ids.device_ptr();
            let mut pw = self.moe_top_weights.device_ptr();
            let mut c_ne = ne as i32;
            let mut c_ku = ku as i32;
            let mut c_gf = gating_code;
            let mut c_nw = i32::from(moe.cfg.norm_weights);
            let mut c_eps = moe.cfg.norm_weights_eps;
            let mut c_sc = moe.cfg.weights_scale;
            let mut args: [*mut c_void; 10] = [
                (&mut pl as *mut CUdeviceptr).cast(),
                (&mut pb as *mut CUdeviceptr).cast(),
                (&mut pids as *mut CUdeviceptr).cast(),
                (&mut pw as *mut CUdeviceptr).cast(),
                (&mut c_ne as *mut i32).cast(),
                (&mut c_ku as *mut i32).cast(),
                (&mut c_gf as *mut i32).cast(),
                (&mut c_nw as *mut i32).cast(),
                (&mut c_eps as *mut f32).cast(),
                (&mut c_sc as *mut f32).cast(),
            ];
            // SAFETY: Matches `moe_router_topk_f32`.
            unsafe {
                self.ctx.launch(
                    "moe_router_topk_f32",
                    self.kernels.moe_router_topk_f32,
                    (n_tok as u32, 1, 1),
                    (32, 1, 1),
                    0,
                    &mut args,
                )?;
            }
        }

        // Zero output accumulator
        // SAFETY: `cuMemsetD8_v2` via scale_inplace with 0.0f is stream-ordered!
        self.kernels
            .launch_scale_inplace(&self.ctx, ffn_out_ptr, 0.0, n_tok * d)?;

        // 3. Deterministic `0..k_used` expert evaluation per token
        for t in 0..n_tok {
            let x_t = ffn_in_ptr + (t * d * 4) as u64;
            let out_t = ffn_out_ptr + (t * d * 4) as u64;
            let ids_t = self.moe_top_ids.device_ptr() + (t * ku * 4) as u64;
            let ws_t = self.moe_top_weights.device_ptr() + (t * ku * 4) as u64;

            for slot in 0..ku {
                match &moe.experts {
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
                        self.launch_indexed_exp(
                            up_exps.device_ptr(),
                            x_t,
                            ids_t,
                            ws_t,
                            self.moe_exp_up.device_ptr(),
                            *up_dtype,
                            n_ff_exp,
                            d,
                            *up_row_bytes,
                            *up_exp_bytes,
                            slot,
                            false,
                        )?;
                        let down_in = if let Some(ge) = gate_exps {
                            self.launch_indexed_exp(
                                ge.device_ptr(),
                                x_t,
                                ids_t,
                                ws_t,
                                self.moe_exp_gate.device_ptr(),
                                *gate_dtype,
                                n_ff_exp,
                                d,
                                *gate_row_bytes,
                                *gate_exp_bytes,
                                slot,
                                false,
                            )?;
                            self.kernels.launch_swiglu(
                                &self.ctx,
                                self.moe_exp_gate.device_ptr(),
                                self.moe_exp_up.device_ptr(),
                                n_ff_exp,
                            )?;
                            self.moe_exp_gate.device_ptr()
                        } else {
                            self.moe_exp_up.device_ptr()
                        };
                        self.launch_indexed_exp(
                            down_exps.device_ptr(),
                            down_in,
                            ids_t,
                            ws_t,
                            out_t,
                            *down_dtype,
                            d,
                            n_ff_exp,
                            *down_row_bytes,
                            *down_exp_bytes,
                            slot,
                            true,
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
                        self.launch_indexed_exp(
                            gate_up_exps.device_ptr(),
                            x_t,
                            ids_t,
                            ws_t,
                            self.moe_exp_gate.device_ptr(),
                            *gate_up_dtype,
                            2 * n_ff_exp,
                            d,
                            *gate_up_row_bytes,
                            *gate_up_exp_bytes,
                            slot,
                            false,
                        )?;
                        self.kernels.launch_swiglu_fused(
                            &self.ctx,
                            self.moe_exp_gate.device_ptr(),
                            self.moe_exp_up.device_ptr(),
                            n_ff_exp,
                            1,
                        )?;
                        self.launch_indexed_exp(
                            down_exps.device_ptr(),
                            self.moe_exp_up.device_ptr(),
                            ids_t,
                            ws_t,
                            out_t,
                            *down_dtype,
                            d,
                            n_ff_exp,
                            *down_row_bytes,
                            *down_exp_bytes,
                            slot,
                            true,
                        )?;
                    }
                }
            }
        }

        // 4. Optional shared expert
        if let Some(ref sh) = moe.shared_expert {
            self.kernels.launch_linear(
                &self.ctx,
                &sh.w_gate,
                ffn_in_ptr,
                self.xq_scratch.device_ptr(),
                false,
                self.hb.device_ptr(),
                n_tok,
                false,
            )?;
            self.kernels.launch_linear(
                &self.ctx,
                &sh.w_up,
                ffn_in_ptr,
                self.xq_scratch.device_ptr(),
                false,
                self.hb2.device_ptr(),
                n_tok,
                false,
            )?;
            self.kernels.launch_swiglu(
                &self.ctx,
                self.hb.device_ptr(),
                self.hb2.device_ptr(),
                n_tok * sh.n_ff_shexp,
            )?;
            // Project shared expert down into `moe_logits` or `hb2` (reuse `hb2` of size `MAX_CUDA_BATCH * ff_max >= n_tok * d`)
            self.kernels.launch_linear(
                &self.ctx,
                &sh.w_down,
                self.hb.device_ptr(),
                self.xq_scratch.device_ptr(),
                false,
                self.hb2.device_ptr(),
                n_tok,
                false,
            )?;
            if let Some(ref w_gi) = sh.w_gate_inp_shexp {
                self.kernels.launch_linear(
                    &self.ctx,
                    w_gi,
                    ffn_in_ptr,
                    self.xq_scratch.device_ptr(),
                    false,
                    self.moe_shexp_gate.device_ptr(),
                    n_tok,
                    true,
                )?;
                for t in 0..n_tok {
                    let row_ptr = self.hb2.device_ptr() + (t * d * 4) as u64;
                    // Broadcast scalar gate via `sigmoid_gate_mul` on 1 element? Wait: `moe_shexp_gate[t]` scales all `d` floats!
                    // Read scalar on host only if `w_gate_inp_shexp` present (qwen2moe):
                    let mut g = [0.0f32; 1];
                    let tmp_slice =
                        unsafe { core::slice::from_raw_parts_mut(g.as_mut_ptr().cast::<u8>(), 4) };
                    // Sync & read single scalar for qwen2moe
                    self.ctx.synchronize()?;
                    let sub = CudaBufferSliceRead {
                        ctx: &self.ctx,
                        ptr: self.moe_shexp_gate.device_ptr() + (t * 4) as u64,
                    };
                    sub.read_4b(tmp_slice)?;
                    let sig = 1.0f32 / (1.0f32 + (-g[0]).exp());
                    self.kernels
                        .launch_scale_inplace(&self.ctx, row_ptr, sig, d)?;
                }
            }
            self.kernels.launch_add_scaled(
                &self.ctx,
                ffn_out_ptr,
                self.hb2.device_ptr(),
                1.0,
                n_tok * d,
            )?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_indexed_exp(
        &self,
        w_ptr: CUdeviceptr,
        x_ptr: CUdeviceptr,
        ids_ptr: CUdeviceptr,
        ws_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        dtype: i32,
        rows: usize,
        cols: usize,
        row_bytes: usize,
        exp_stride: usize,
        slot: usize,
        accum_weighted: bool,
    ) -> Result<(), CudaError> {
        let mut pw = w_ptr;
        let mut px = x_ptr;
        let mut pids = ids_ptr;
        let mut pws = ws_ptr;
        let mut pout = out_ptr;
        let mut dt = dtype;
        let mut r = rows as i32;
        let mut c = cols as i32;
        let mut rb = row_bytes;
        let mut es = exp_stride;
        let mut sl = slot as i32;
        let mut aw = i32::from(accum_weighted);
        let mut args: [*mut c_void; 12] = [
            (&mut pw as *mut CUdeviceptr).cast(),
            (&mut px as *mut CUdeviceptr).cast(),
            (&mut pids as *mut CUdeviceptr).cast(),
            (&mut pws as *mut CUdeviceptr).cast(),
            (&mut pout as *mut CUdeviceptr).cast(),
            (&mut dt as *mut i32).cast(),
            (&mut r as *mut i32).cast(),
            (&mut c as *mut i32).cast(),
            (&mut rb as *mut usize).cast(),
            (&mut es as *mut usize).cast(),
            (&mut sl as *mut i32).cast(),
            (&mut aw as *mut i32).cast(),
        ];
        // SAFETY: Matches `moe_indexed_gemv_f32`.
        unsafe {
            self.ctx.launch(
                "moe_indexed_gemv_f32",
                self.kernels.moe_indexed_gemv_f32,
                ((rows as u32).div_ceil(4), 1, 1),
                (32, 4, 1),
                0,
                &mut args,
            )
        }
    }

    fn can_use_decode_graph(&self) -> bool {
        // Dense models and MoE models without host scalar readback (`qwen2moe` `w_gate_inp_shexp`)
        // have 100% static GPU kernel topology on single-token decode!
        !self.layers.iter().any(|l| {
            l.moe
                .as_ref()
                .and_then(|m| m.shared_expert.as_ref())
                .and_then(|sh| sh.w_gate_inp_shexp.as_ref())
                .is_some()
        })
    }

    fn run_decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
        compute_argmax: bool,
    ) -> Result<(), CudaError> {
        self.token_ids_buf.copy_from_u32(&[token_id])?;
        if self.can_use_decode_graph() {
            self.pos_buf.copy_from_u32(&[pos as u32])?;
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
            self.token_ids_buf.copy_from_u32(&token_ids[off..off + b])?;
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

struct CudaBufferSliceRead<'a> {
    ctx: &'a CudaContext,
    ptr: CUdeviceptr,
}

impl<'a> CudaBufferSliceRead<'a> {
    fn read_4b(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        let tmp = self.ctx.alloc_zeroed(16)?;
        // Copy 4B via D2D then D2H using safe CudaBuffer methods
        self.ctx.bind_current()?;
        // Simple 4-byte synchronous DtoH via a tiny helper on `CudaContext`? Let's expose `dtoh_at` in `driver.rs`!
        let _ = tmp;
        self.ctx.copy_dtoh_raw(self.ptr, dst)
    }
}
