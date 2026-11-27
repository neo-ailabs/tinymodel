//! CUDA GPU runner for `Qwen3.5` hybrid Gated DeltaNet + Full Attention (`GDN_CONTRACT` compliant).

use std::ffi::c_void;

use tm_core::{DType, KvType};
use tm_gguf::GgufFile;
use tm_models::{Qwen35AttnWeights, Qwen35Config, Qwen35FfnWeights, Qwen35Model};

use crate::CudaError;
use crate::driver::{CUdeviceptr, CudaBuffer, CudaContext, CudaEvent, CudaGraphExec};
use crate::kernels::{BLOCK_Q8_1_BYTES, CudaKernels, CudaMat};
use crate::llama_runner::{
    CudaMoeLayer, CudaMoeScratch, CudaProfileReport, CudaStageStat, MAX_CUDA_BATCH,
    moe_weight_bytes, upload_mat,
};

enum CudaQwen35Attn {
    FullAttn {
        q_norm: CudaBuffer,
        k_norm: CudaBuffer,
        wqg: CudaMat,
        wk: CudaMat,
        wv: CudaMat,
        wo: CudaMat,
        k_cache: CudaBuffer,
        v_cache: CudaBuffer,
    },
    Gdn {
        wqkv: CudaMat,
        wz: CudaMat,
        w_alpha: CudaMat,
        w_beta: CudaMat,
        conv1d: CudaBuffer,
        ssm_a: CudaBuffer,
        ssm_dt_bias: CudaBuffer,
        ssm_norm: CudaBuffer,
        w_out: CudaMat,
        conv_hist: CudaBuffer, // [hist_len, conv_dim]
        ssm_state: CudaBuffer, // [hv, s_dim, dv_dim]
    },
}

enum CudaQwen35Ffn {
    Dense {
        w_gate: CudaMat,
        w_up: CudaMat,
        w_down: CudaMat,
    },
    Moe(Box<CudaMoeLayer>),
}

struct CudaQwen35Layer {
    attn_norm: CudaBuffer,
    post_attn_norm: CudaBuffer,
    attn: CudaQwen35Attn,
    ffn: CudaQwen35Ffn,
}

/// High-throughput CUDA runner for `Qwen3.5` (`qwen35` & `qwen35moe`, e.g. `Qwen3.6-35B-A3B`).
pub struct Qwen35CudaRunner {
    pub ctx: CudaContext,
    pub kernels: CudaKernels,
    pub cfg: Qwen35Config,
    pub max_ctx: usize,
    pub kv_type: KvType,
    pub vram_bytes: usize,
    pub model_weight_bytes: usize,

    token_embd: CudaMat,
    output_norm: CudaBuffer,
    output: CudaMat,
    rope_inv_freqs: CudaBuffer,
    layers: Vec<CudaQwen35Layer>,

    // Pre-allocated Arena scratch buffers on GPU
    token_ids_buf: CudaBuffer,
    pos_buf: CudaBuffer,
    decode_graphs: [Option<CudaGraphExec>; 3],
    x: CudaBuffer,
    xb: CudaBuffer,
    xb2: CudaBuffer,
    qg_or_qkv: CudaBuffer,
    q_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    gate_buf: CudaBuffer,
    alpha_buf: CudaBuffer,
    beta_buf: CudaBuffer,
    beta_act_buf: CudaBuffer,
    g_buf: CudaBuffer,
    hb: CudaBuffer,
    hb2: CudaBuffer,
    xq_scratch: CudaBuffer,
    moe_scratch: Option<CudaMoeScratch>,
    logits_dev: CudaBuffer,
    argmax_dev: CudaBuffer,
    logits_host: Vec<f32>,
}

impl Qwen35CudaRunner {
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, CudaError> {
        Self::from_gguf_with_kv_type(gguf, max_ctx, KvType::F16)
    }

    pub fn from_gguf_with_kv_type(
        gguf: &GgufFile,
        max_ctx: usize,
        kv_type: KvType,
    ) -> Result<Self, CudaError> {
        let cpu_m = Qwen35Model::from_gguf(gguf)?;
        let cfg = cpu_m.cfg.clone();
        let ctx = CudaContext::new(None)?;
        let kernels = CudaKernels::load(&ctx)?;

        let q_dim = cfg.n_head * cfg.head_dim;
        let qg_dim = 2 * q_dim;
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let conv_dim = cfg.conv_dim();
        let hk = cfg.ssm_n_group;
        let hv = cfg.ssm_dt_rank;
        let s_dim = cfg.ssm_state_size;
        let dv_dim = cfg.ssm_head_v_dim();
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = cfg.ssm_d_inner;
        let hist_len = cfg.ssm_conv_kernel.saturating_sub(1);

        if kv_type == KvType::Q8_0 && !kv_dim.is_multiple_of(32) {
            return Err(CudaError::InvalidDimensions(format!(
                "Q8_0 KV cache requires kv_dim ({kv_dim}) divisible by 32"
            )));
        }
        let kv_layer_bytes = match kv_type {
            KvType::F16 => max_ctx * kv_dim * 2,
            KvType::Q8_0 => max_ctx * (kv_dim / 32) * 34,
        };

        let tied_output =
            std::ptr::eq(cpu_m.token_embd.bytes.as_ptr(), cpu_m.output.bytes.as_ptr());
        let mut weight_bytes = cpu_m.token_embd.bytes.len()
            + if tied_output {
                0
            } else {
                cpu_m.output.bytes.len()
            };
        let mut state_bytes = 0usize;
        for l in &cpu_m.layers {
            match &l.ffn {
                Qwen35FfnWeights::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => {
                    weight_bytes += w_gate.bytes.len() + w_up.bytes.len() + w_down.bytes.len();
                }
                Qwen35FfnWeights::Moe(moe) => {
                    weight_bytes += moe_weight_bytes(moe);
                }
            }
            match &l.attn {
                Qwen35AttnWeights::FullAttn {
                    wqg, wk, wv, wo, ..
                } => {
                    weight_bytes +=
                        wqg.bytes.len() + wk.bytes.len() + wv.bytes.len() + wo.bytes.len();
                    state_bytes += 2 * kv_layer_bytes;
                }
                Qwen35AttnWeights::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    w_out,
                    ..
                } => {
                    weight_bytes += wqkv.bytes.len()
                        + wz.bytes.len()
                        + w_alpha.bytes.len()
                        + w_beta.bytes.len()
                        + w_out.bytes.len();
                    state_bytes += (hist_len * conv_dim + hv * s_dim * dv_dim) * 4;
                }
            }
        }
        let (free_vram, _) = ctx.mem_info()?;
        let needed_bytes = weight_bytes + state_bytes + 64 * 1024 * 1024;
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
        let inv_f32: Vec<f32> = cpu_m.rope_inv_freqs.iter().map(|&v| v as f32).collect();
        let rope_inv_freqs = ctx.alloc_from_f32(&inv_f32)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in &cpu_m.layers {
            let attn = match &l.attn {
                Qwen35AttnWeights::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                } => CudaQwen35Attn::FullAttn {
                    q_norm: ctx.alloc_from_f32(q_norm)?,
                    k_norm: ctx.alloc_from_f32(k_norm)?,
                    wqg: upload_mat(&ctx, wqg)?,
                    wk: upload_mat(&ctx, wk)?,
                    wv: upload_mat(&ctx, wv)?,
                    wo: upload_mat(&ctx, wo)?,
                    k_cache: ctx.alloc_zeroed(kv_layer_bytes)?,
                    v_cache: ctx.alloc_zeroed(kv_layer_bytes)?,
                },
                Qwen35AttnWeights::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    w_out,
                } => CudaQwen35Attn::Gdn {
                    wqkv: upload_mat(&ctx, wqkv)?,
                    wz: upload_mat(&ctx, wz)?,
                    w_alpha: upload_mat(&ctx, w_alpha)?,
                    w_beta: upload_mat(&ctx, w_beta)?,
                    conv1d: ctx.alloc_from_f32(conv1d)?,
                    ssm_a: ctx.alloc_from_f32(ssm_a)?,
                    ssm_dt_bias: ctx.alloc_from_f32(ssm_dt_bias)?,
                    ssm_norm: ctx.alloc_from_f32(ssm_norm)?,
                    w_out: upload_mat(&ctx, w_out)?,
                    conv_hist: ctx.alloc_zeroed(hist_len.max(1) * conv_dim * 4)?,
                    ssm_state: ctx.alloc_zeroed(hv * s_dim * dv_dim * 4)?,
                },
            };
            let ffn = match &l.ffn {
                Qwen35FfnWeights::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => CudaQwen35Ffn::Dense {
                    w_gate: upload_mat(&ctx, w_gate)?,
                    w_up: upload_mat(&ctx, w_up)?,
                    w_down: upload_mat(&ctx, w_down)?,
                },
                Qwen35FfnWeights::Moe(moe) => {
                    CudaQwen35Ffn::Moe(Box::new(CudaMoeLayer::upload(&ctx, moe, cfg.n_embd)?))
                }
            };
            layers.push(CudaQwen35Layer {
                attn_norm: ctx.alloc_from_f32(&l.attn_norm)?,
                post_attn_norm: ctx.alloc_from_f32(&l.post_attn_norm)?,
                attn,
                ffn,
            });
        }

        let ff_max = cfg
            .n_ff
            .max(cfg.max_n_ff_exp)
            .max(cfg.max_n_ff_shexp)
            .max(cfg.n_embd);
        let max_cols = cfg
            .n_embd
            .max(ff_max)
            .max(q_dim)
            .max(v_gdn_dim)
            .next_multiple_of(32);

        let token_ids_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * 4)?;
        let pos_buf = ctx.alloc_zeroed(4)?;
        let x = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_embd * 4)?;
        let xb = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_embd.max(q_dim).max(v_gdn_dim) * 4)?;
        let xb2 = ctx.alloc_zeroed(MAX_CUDA_BATCH * cfg.n_embd * 4)?;
        let qg_or_qkv = ctx.alloc_zeroed(MAX_CUDA_BATCH * qg_dim.max(conv_dim) * 4)?;
        let q_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * q_dim.max(k_gdn_dim) * 4)?;
        let k_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * kv_dim.max(k_gdn_dim) * 4)?;
        let v_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * kv_dim.max(v_gdn_dim) * 4)?;
        let gate_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * q_dim.max(v_gdn_dim) * 4)?;
        let alpha_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hv * 4)?;
        let beta_buf = ctx.alloc_zeroed(MAX_CUDA_BATCH * hv * 4)?;
        let beta_act_buf = ctx.alloc_zeroed(hv * 4)?;
        let g_buf = ctx.alloc_zeroed(hv * 4)?;
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
            rope_inv_freqs,
            layers,
            token_ids_buf,
            pos_buf,
            decode_graphs: [None, None, None],
            x,
            xb,
            xb2,
            qg_or_qkv,
            q_buf,
            k_buf,
            v_buf,
            gate_buf,
            alpha_buf,
            beta_buf,
            beta_act_buf,
            g_buf,
            hb,
            hb2,
            xq_scratch,
            moe_scratch,
            logits_dev,
            argmax_dev,
            logits_host: vec![0.0f32; cfg.vocab_size],
        })
    }

    pub fn reset_state(&mut self) -> Result<(), CudaError> {
        for l in &self.layers {
            match &l.attn {
                CudaQwen35Attn::FullAttn {
                    k_cache, v_cache, ..
                } => {
                    k_cache.zero()?;
                    v_cache.zero()?;
                }
                CudaQwen35Attn::Gdn {
                    conv_hist,
                    ssm_state,
                    ..
                } => {
                    conv_hist.zero()?;
                    ssm_state.zero()?;
                }
            }
        }
        self.ctx.synchronize()
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
        let conv_dim = self.cfg.conv_dim();
        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let dv_dim = self.cfg.ssm_head_v_dim();
        let k_gdn_dim = hk * s_dim;
        let v_gdn_dim = self.cfg.ssm_d_inner;
        let d_conv = self.cfg.ssm_conv_kernel;
        let is_q8_kv = self.kv_type == KvType::Q8_0;
        let attn_scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
        let can_share_xq = n_tok <= crate::kernels::MMVQ_MAX_TOK && d.is_multiple_of(32);

        if !preloaded_embed {
            self.kernels.launch_embed(
                &self.ctx,
                &self.token_embd,
                self.token_ids_buf.device_ptr(),
                self.x.device_ptr(),
                n_tok,
                1.0,
            )?;
        }

        for layer in &self.layers {
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.x.device_ptr(),
                layer.attn_norm.device_ptr(),
                self.xb.device_ptr(),
                d,
                n_tok,
                self.cfg.rms_eps,
            )?;

            match &layer.attn {
                CudaQwen35Attn::FullAttn {
                    q_norm,
                    k_norm,
                    wqg,
                    wk,
                    wv,
                    wo,
                    k_cache,
                    v_cache,
                } => {
                    let uses_q8 = matches!(
                        wqg.dtype,
                        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
                    );
                    if can_share_xq && uses_q8 {
                        self.kernels.launch_quantize_q8_1(
                            &self.ctx,
                            self.xb.device_ptr(),
                            self.xq_scratch.device_ptr(),
                            d,
                            n_tok,
                        )?;
                    }
                    self.kernels.launch_linear(
                        &self.ctx,
                        wqg,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.qg_or_qkv.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        wk,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.k_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        wv,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.v_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;

                    // Split interleaved `[q_h, gate_h]` per head
                    {
                        let total = (n_tok * q_dim) as u32;
                        let mut pqg = self.qg_or_qkv.device_ptr();
                        let mut pq = self.q_buf.device_ptr();
                        let mut pg = self.gate_buf.device_ptr();
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

                    self.kernels.launch_flash_attn_gqa(
                        &self.ctx,
                        self.q_buf.device_ptr(),
                        k_cache.device_ptr(),
                        v_cache.device_ptr(),
                        self.xb.device_ptr(),
                        is_q8_kv,
                        self.cfg.n_head,
                        self.cfg.n_head_kv,
                        self.cfg.head_dim,
                        start_pos,
                        pos_dev_ptr,
                        n_tok,
                        0,
                        attn_scale,
                        0.0,
                    )?;

                    // Multiply attention output by `sigmoid(gate)` (`Qwen3.5` FullAttn output gate!)
                    self.kernels.launch_sigmoid_gate_mul(
                        &self.ctx,
                        self.xb.device_ptr(),
                        self.gate_buf.device_ptr(),
                        n_tok * q_dim,
                    )?;

                    self.kernels.launch_linear(
                        &self.ctx,
                        wo,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        false,
                        self.xb2.device_ptr(),
                        n_tok,
                        false,
                    )?;
                }
                CudaQwen35Attn::Gdn {
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
                } => {
                    // Batched projections (`wmma` Tensor Core for prefill, MMVQ for decode!)
                    let uses_q8 = matches!(
                        wqkv.dtype,
                        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::Q5K | DType::Q6K
                    );
                    if can_share_xq && uses_q8 {
                        self.kernels.launch_quantize_q8_1(
                            &self.ctx,
                            self.xb.device_ptr(),
                            self.xq_scratch.device_ptr(),
                            d,
                            n_tok,
                        )?;
                    }
                    self.kernels.launch_linear(
                        &self.ctx,
                        wqkv,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.qg_or_qkv.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        wz,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.gate_buf.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    // `w_alpha` and `w_beta` have only `hv` (e.g. 16) rows -> `force_mmvq = true` is ultra fast!
                    self.kernels.launch_linear(
                        &self.ctx,
                        w_alpha,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.alpha_buf.device_ptr(),
                        n_tok,
                        true,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        w_beta,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && uses_q8,
                        self.beta_buf.device_ptr(),
                        n_tok,
                        true,
                    )?;

                    // Sequential causal GDN recurrence across `t = 0..n_tok` (preserves 0-ULP batch determinism!)
                    for t in 0..n_tok {
                        let qkv_t = self.qg_or_qkv.device_ptr() + (t * conv_dim * 4) as u64;
                        let z_t = self.gate_buf.device_ptr() + (t * v_gdn_dim * 4) as u64;
                        let alpha_t = self.alpha_buf.device_ptr() + (t * hv * 4) as u64;
                        let beta_t = self.beta_buf.device_ptr() + (t * hv * 4) as u64;
                        let y_t = self.xb.device_ptr() + (t * v_gdn_dim * 4) as u64;

                        // 1. Causal Conv1d + SiLU BEFORE split (`GDN_CONTRACT` #1)
                        {
                            let mut pqkv = qkv_t;
                            let mut phist = conv_hist.device_ptr();
                            let mut pcw = conv1d.device_ptr();
                            let mut pq = self.q_buf.device_ptr();
                            let mut pk = self.k_buf.device_ptr();
                            let mut pv = self.v_buf.device_ptr();
                            let mut cd = conv_dim as i32;
                            let mut dc = d_conv as i32;
                            let mut kd = k_gdn_dim as i32;
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

                        // 2. Per-head Q (`* 1/sqrt(S)`) & K L2 norm (`GDN_CONTRACT` #2 & #3)
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
                            let mut pa = alpha_t;
                            let mut pb = beta_t;
                            let mut psa = ssm_a.device_ptr();
                            let mut pdb = ssm_dt_bias.device_ptr();
                            let mut pg = self.g_buf.device_ptr();
                            let mut pbo = self.beta_act_buf.device_ptr();
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

                        // 4. Warp-sharded GDN state recurrence (`vh % HK`, `GDN_CONTRACT` #4)
                        {
                            let mut pst = ssm_state.device_ptr();
                            let mut pq = self.q_buf.device_ptr();
                            let mut pk = self.k_buf.device_ptr();
                            let mut pv = self.v_buf.device_ptr();
                            let mut pg = self.g_buf.device_ptr();
                            let mut pb = self.beta_act_buf.device_ptr();
                            let mut py = y_t;
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

                        // 5. Per-head gated RMSNorm `rms_norm(y) * silu(z)` (`GDN_CONTRACT` #5)
                        {
                            let mut py = y_t;
                            let mut pz = z_t;
                            let mut pnw = ssm_norm.device_ptr();
                            let mut cdv = dv_dim as i32;
                            let mut eps = self.cfg.rms_eps;
                            let mut args: [*mut c_void; 5] = [
                                (&mut py as *mut CUdeviceptr).cast(),
                                (&mut pz as *mut CUdeviceptr).cast(),
                                (&mut pnw as *mut CUdeviceptr).cast(),
                                (&mut cdv as *mut i32).cast(),
                                (&mut eps as *mut f32).cast(),
                            ];
                            // SAFETY: Matches `gdn_rms_norm_gated_f32`.
                            unsafe {
                                self.ctx.launch(
                                    "gdn_rms_norm_gated_f32",
                                    self.kernels.gdn_rms_norm_gated_f32,
                                    (hv as u32, 1, 1),
                                    (32, 1, 1),
                                    0,
                                    &mut args,
                                )?;
                            }
                        }
                    }

                    // Output projection `[n_tok, v_gdn_dim] -> [n_tok, n_embd]`
                    self.kernels.launch_linear(
                        &self.ctx,
                        w_out,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        false,
                        self.xb2.device_ptr(),
                        n_tok,
                        false,
                    )?;
                }
            }

            // Residual after attention/GDN
            self.kernels.launch_add_scaled(
                &self.ctx,
                self.x.device_ptr(),
                self.xb2.device_ptr(),
                1.0,
                n_tok * d,
            )?;

            // Pre-FFN RMSNorm (`post_attn_norm` in Qwen3.5 GGUF naming) + Dense/MoE FFN + residual
            self.kernels.launch_rms_norm(
                &self.ctx,
                self.x.device_ptr(),
                layer.post_attn_norm.device_ptr(),
                self.xb.device_ptr(),
                d,
                n_tok,
                self.cfg.rms_eps,
            )?;
            match &layer.ffn {
                CudaQwen35Ffn::Dense {
                    w_gate,
                    w_up,
                    w_down,
                } => {
                    let ffn_uses_q8 = matches!(
                        w_gate.dtype,
                        DType::Q2_0
                            | DType::Q4_0
                            | DType::Q4_1
                            | DType::Q8_0
                            | DType::Q5K
                            | DType::Q6K
                    );
                    if can_share_xq && ffn_uses_q8 {
                        self.kernels.launch_quantize_q8_1(
                            &self.ctx,
                            self.xb.device_ptr(),
                            self.xq_scratch.device_ptr(),
                            d,
                            n_tok,
                        )?;
                    }
                    self.kernels.launch_linear(
                        &self.ctx,
                        w_gate,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && ffn_uses_q8,
                        self.hb.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    self.kernels.launch_linear(
                        &self.ctx,
                        w_up,
                        self.xb.device_ptr(),
                        self.xq_scratch.device_ptr(),
                        can_share_xq && ffn_uses_q8,
                        self.hb2.device_ptr(),
                        n_tok,
                        false,
                    )?;
                    let down_uses_q8 = matches!(
                        w_down.dtype,
                        DType::Q2_0
                            | DType::Q4_0
                            | DType::Q4_1
                            | DType::Q8_0
                            | DType::Q5K
                            | DType::Q6K
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
                            w_down,
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
                            w_down,
                            self.hb.device_ptr(),
                            self.xq_scratch.device_ptr(),
                            false,
                            self.xb2.device_ptr(),
                            n_tok,
                            false,
                        )?;
                    }
                }
                CudaQwen35Ffn::Moe(moe) => {
                    let scratch = self.moe_scratch.as_ref().ok_or_else(|| {
                        CudaError::InvalidDimensions(
                            "Qwen3.5 MoE layer present without CudaMoeScratch".to_string(),
                        )
                    })?;
                    moe.encode(
                        &self.ctx,
                        &self.kernels,
                        scratch,
                        self.xq_scratch.device_ptr(),
                        self.hb.device_ptr(),
                        self.hb2.device_ptr(),
                        self.xb.device_ptr(),
                        self.xb2.device_ptr(),
                        d,
                        n_tok,
                    )?;
                }
            }
            self.kernels.launch_add_scaled(
                &self.ctx,
                self.x.device_ptr(),
                self.xb2.device_ptr(),
                1.0,
                n_tok * d,
            )?;
        }

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
            .expect("captured qwen35 decode graph")
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
        if pos >= self.max_ctx || embed.len() != self.cfg.n_embd {
            return Err(CudaError::InvalidDimensions(format!(
                "invalid forward_embed len={} or pos={pos}",
                embed.len()
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
