//! Single-command-buffer Vulkan runner for `llama` and `qwen3` architectures
//! (`SmolLM2`, `Llama-3.2`, `Qwen3`), supporting all GGUF quantizations (`Q4_0`, `Q4_0x4`,
//! `Q4_1`, `Q8_0`, `Q5_K`, `Q6_K`, `F16`, `BF16`, `F32`) with pre-baked `VkDescriptorSet`s
//! and zero heap allocations on the token generation path (`AGENTS.md` Rule #4).

use std::sync::Arc;

use ash::vk;
use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::context::{DescriptorArena, GpuBuffer, VulkanContext, VulkanError, as_bytes};
use crate::pipelines::{
    ArgmaxPush, AttnPush, ElemPush, EmbedPush, GemmPush, GemvPush, NormPush, QkNormPush,
    RopeKvPush, VulkanPipelines,
};

const MAX_VK_BATCH: usize = 64;

#[derive(Debug, Clone)]
pub struct LlamaVulkanConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rope_type: u32,
    pub rms_eps: f32,
    pub vocab_size: usize,
    pub max_ctx: usize,
}

struct VkLinear {
    _buf: Arc<GpuBuffer>,
    dtype: DType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    dset_gemv: vk::DescriptorSet,
    dset_gemm: vk::DescriptorSet,
}

struct VkLlamaLayer {
    _attn_norm: GpuBuffer,
    dset_attn_norm: vk::DescriptorSet,
    wq: VkLinear,
    wk: VkLinear,
    wv: VkLinear,
    _q_norm: Option<GpuBuffer>,
    dset_q_norm: Option<vk::DescriptorSet>,
    _k_norm: Option<GpuBuffer>,
    dset_k_norm: Option<vk::DescriptorSet>,
    k_cache: GpuBuffer,
    v_cache: GpuBuffer,
    dset_rope_kv: vk::DescriptorSet,
    dset_gqa_attn: vk::DescriptorSet,
    wo: VkLinear,
    _ffn_norm: GpuBuffer,
    dset_ffn_norm: vk::DescriptorSet,
    w_gate: VkLinear,
    w_up: VkLinear,
    dset_swiglu: vk::DescriptorSet,
    w_down: VkLinear,
}

/// End-to-end Vulkan GPU runner for `llama` and `qwen3` models.
pub struct LlamaVulkanRunner {
    pub ctx: Arc<VulkanContext>,
    pub cfg: LlamaVulkanConfig,
    pipes: VulkanPipelines,
    _desc_arena: DescriptorArena,
    _token_embd: Arc<GpuBuffer>,
    token_embd_dtype: DType,
    token_embd_row_bytes: usize,
    dset_embed: vk::DescriptorSet,
    _rope_freqs: GpuBuffer,
    has_rope_freqs: bool,
    layers: Vec<VkLlamaLayer>,
    _output_norm: GpuBuffer,
    dset_output_norm: vk::DescriptorSet,
    output: VkLinear,
    dset_argmax: vk::DescriptorSet,
    tok_ids_buf: GpuBuffer,
    _x_buf: GpuBuffer,
    _xb_buf: GpuBuffer,
    _q_buf: GpuBuffer,
    _k_buf: GpuBuffer,
    _v_buf: GpuBuffer,
    _hb_buf: GpuBuffer,
    _hb2_buf: GpuBuffer,
    logits_buf: GpuBuffer,
    argmax_buf: GpuBuffer,
    host_logits: Vec<f32>,
}

fn load_f32_vec(
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    len: usize,
) -> Result<Vec<f32>, VulkanError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[len])?;
    let mut out = vec![0.0f32; len];
    dequantize_row_bytes(info.dtype, bytes, &mut out);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn upload_linear(
    ctx: &Arc<VulkanContext>,
    pipes: &VulkanPipelines,
    arena: &DescriptorArena,
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
    in_buf: &GpuBuffer,
    out_buf: &GpuBuffer,
    shared_buf: Option<Arc<GpuBuffer>>,
) -> Result<(VkLinear, Arc<GpuBuffer>), VulkanError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    // Ensure dtype is supported by Vulkan GEMV + GEMM before allocating
    let gv = pipes.gemv_for_dtype(info.dtype, name)?;
    let gm = pipes.gemm_for_dtype(info.dtype, name)?;
    let buf = match shared_buf {
        Some(existing) => existing,
        None => Arc::new(GpuBuffer::from_bytes_device_optimal(ctx, bytes)?),
    };
    let dset_gemv = arena.alloc_and_write(gv, &[&buf, in_buf, out_buf])?;
    let dset_gemm = arena.alloc_and_write(gm, &[&buf, in_buf, out_buf])?;
    let row_bytes = info.dtype.row_bytes(cols);
    Ok((
        VkLinear {
            _buf: Arc::clone(&buf),
            dtype: info.dtype,
            rows,
            cols,
            row_bytes,
            dset_gemv,
            dset_gemm,
        },
        buf,
    ))
}

impl LlamaVulkanRunner {
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, VulkanError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "llama" && arch != "qwen3" {
            return Err(VulkanError::UnsupportedArch(arch));
        }

        let pfx = arch.as_str();
        let n_layers = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
        let n_ff = gguf.req_usize(&format!("{pfx}.feed_forward_length"))?;
        let n_head = gguf.req_usize(&format!("{pfx}.attention.head_count"))?;
        let n_head_kv = gguf
            .get_meta(&format!("{pfx}.attention.head_count_kv"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim = gguf
            .get_meta(&format!("{pfx}.attention.key_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head);
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000.0);
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);
        // Qwen3 uses NeoX (`2`); Llama uses Normal (`0`)
        let rope_type = if arch == "qwen3" { 2u32 } else { 0u32 };

        let embd_info =
            gguf.tensor_info("token_embd.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf.path().display().to_string(),
                    tensor: "token_embd.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let vocab_size = gguf
            .get_meta(&format!("{pfx}.vocab_size"))
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| embd_info.dims.get(1).copied().unwrap_or(0));

        let ctx = VulkanContext::new()?;
        let pipes = VulkanPipelines::new(&ctx)?;

        // Pre-size static DescriptorArena (`~20` sets per layer + `16` global)
        let max_sets = (n_layers as u32) * 24 + 32;
        let desc_arena = DescriptorArena::new(&ctx, max_sets, max_sets * 6)?;

        let q_dim = n_head * head_dim;
        let kv_dim = n_head_kv * head_dim;
        let tok_ids_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * 4)?;
        let x_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_embd * 4)?;
        let xb_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_embd.max(q_dim) * 4)?;
        let q_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * q_dim * 4)?;
        let k_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * kv_dim * 4)?;
        let v_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * kv_dim * 4)?;
        let hb_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_ff * 4)?;
        let hb2_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_ff * 4)?;
        let logits_buf = GpuBuffer::new_host_visible(&ctx, vocab_size * 4)?;
        let argmax_buf = GpuBuffer::new_host_visible(&ctx, 16)?;

        let (emb_ti, emb_bytes) =
            gguf.req_tensor("token_embd.weight", "global", &[n_embd, vocab_size])?;
        let token_embd = Arc::new(GpuBuffer::from_bytes_device_optimal(&ctx, emb_bytes)?);
        let token_embd_dtype = emb_ti.dtype;
        let token_embd_row_bytes = emb_ti.dtype.row_bytes(n_embd);
        let dset_embed =
            desc_arena.alloc_and_write(&pipes.embed, &[&token_embd, &tok_ids_buf, &x_buf])?;

        let (rope_freqs, has_rope_freqs) = if gguf.tensor_info("rope_freqs.weight").is_some() {
            let v = load_f32_vec(gguf, "rope_freqs.weight", "global", rope_dim / 2)?;
            (GpuBuffer::from_f32_slice(&ctx, &v)?, true)
        } else {
            (GpuBuffer::from_f32_slice(&ctx, &[1.0f32; 4])?, false)
        };

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm_v =
                load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, n_embd)?;
            let attn_norm = GpuBuffer::from_f32_slice(&ctx, &attn_norm_v)?;
            let dset_attn_norm =
                desc_arena.alloc_and_write(&pipes.rms_norm, &[&x_buf, &attn_norm, &xb_buf])?;

            let (wq, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.attn_q.weight"),
                &ltag,
                n_embd,
                q_dim,
                &xb_buf,
                &q_buf,
                None,
            )?;
            let (wk, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.attn_k.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &xb_buf,
                &k_buf,
                None,
            )?;
            let (wv, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.attn_v.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &xb_buf,
                &v_buf,
                None,
            )?;

            let qn_name = format!("{ltag}.attn_q_norm.weight");
            let (q_norm, dset_q_norm) = if gguf.tensor_info(&qn_name).is_some() {
                let qn_v = load_f32_vec(gguf, &qn_name, &ltag, head_dim)?;
                let qn_b = GpuBuffer::from_f32_slice(&ctx, &qn_v)?;
                let ds = desc_arena.alloc_and_write(&pipes.qk_norm, &[&q_buf, &qn_b])?;
                (Some(qn_b), Some(ds))
            } else {
                (None, None)
            };

            let kn_name = format!("{ltag}.attn_k_norm.weight");
            let (k_norm, dset_k_norm) = if gguf.tensor_info(&kn_name).is_some() {
                let kn_v = load_f32_vec(gguf, &kn_name, &ltag, head_dim)?;
                let kn_b = GpuBuffer::from_f32_slice(&ctx, &kn_v)?;
                let ds = desc_arena.alloc_and_write(&pipes.qk_norm, &[&k_buf, &kn_b])?;
                (Some(kn_b), Some(ds))
            } else {
                (None, None)
            };

            let kv_cache_bytes = max_ctx * kv_dim * 2; // F16 = 2 bytes per element
            let k_cache = GpuBuffer::new_host_visible(&ctx, kv_cache_bytes)?;
            let v_cache = GpuBuffer::new_host_visible(&ctx, kv_cache_bytes)?;
            let dset_rope_kv = desc_arena.alloc_and_write(
                &pipes.rope_kv,
                &[&q_buf, &k_buf, &v_buf, &k_cache, &v_cache, &rope_freqs],
            )?;
            let dset_gqa_attn = desc_arena
                .alloc_and_write(&pipes.gqa_attn, &[&q_buf, &k_cache, &v_cache, &xb_buf])?;

            let (wo, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.attn_output.weight"),
                &ltag,
                q_dim,
                n_embd,
                &xb_buf,
                &x_buf,
                None,
            )?;

            let ffn_norm_v = load_f32_vec(gguf, &format!("{ltag}.ffn_norm.weight"), &ltag, n_embd)?;
            let ffn_norm = GpuBuffer::from_f32_slice(&ctx, &ffn_norm_v)?;
            let dset_ffn_norm =
                desc_arena.alloc_and_write(&pipes.rms_norm, &[&x_buf, &ffn_norm, &xb_buf])?;

            let (w_gate, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.ffn_gate.weight"),
                &ltag,
                n_embd,
                n_ff,
                &xb_buf,
                &hb_buf,
                None,
            )?;
            let (w_up, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.ffn_up.weight"),
                &ltag,
                n_embd,
                n_ff,
                &xb_buf,
                &hb2_buf,
                None,
            )?;
            let dset_swiglu = desc_arena.alloc_and_write(&pipes.swiglu, &[&hb_buf, &hb2_buf])?;
            let (w_down, _) = upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                &format!("{ltag}.ffn_down.weight"),
                &ltag,
                n_ff,
                n_embd,
                &hb_buf,
                &x_buf,
                None,
            )?;

            layers.push(VkLlamaLayer {
                _attn_norm: attn_norm,
                dset_attn_norm,
                wq,
                wk,
                wv,
                _q_norm: q_norm,
                dset_q_norm,
                _k_norm: k_norm,
                dset_k_norm,
                k_cache,
                v_cache,
                dset_rope_kv,
                dset_gqa_attn,
                wo,
                _ffn_norm: ffn_norm,
                dset_ffn_norm,
                w_gate,
                w_up,
                dset_swiglu,
                w_down,
            });
        }

        let out_norm_v = load_f32_vec(gguf, "output_norm.weight", "global", n_embd)?;
        let output_norm = GpuBuffer::from_f32_slice(&ctx, &out_norm_v)?;
        let dset_output_norm =
            desc_arena.alloc_and_write(&pipes.rms_norm, &[&x_buf, &output_norm, &xb_buf])?;

        let (output, _) = if gguf.tensor_info("output.weight").is_some() {
            upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                "output.weight",
                "global",
                n_embd,
                vocab_size,
                &xb_buf,
                &logits_buf,
                None,
            )?
        } else {
            upload_linear(
                &ctx,
                &pipes,
                &desc_arena,
                gguf,
                "token_embd.weight",
                "global",
                n_embd,
                vocab_size,
                &xb_buf,
                &logits_buf,
                Some(Arc::clone(&token_embd)),
            )?
        };
        let dset_argmax = desc_arena.alloc_and_write(&pipes.argmax, &[&logits_buf, &argmax_buf])?;

        let cfg = LlamaVulkanConfig {
            arch,
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rope_type,
            rms_eps,
            vocab_size,
            max_ctx,
        };

        Ok(Self {
            ctx,
            cfg,
            pipes,
            _desc_arena: desc_arena,
            _token_embd: token_embd,
            token_embd_dtype,
            token_embd_row_bytes,
            dset_embed,
            _rope_freqs: rope_freqs,
            has_rope_freqs,
            layers,
            _output_norm: output_norm,
            dset_output_norm,
            output,
            dset_argmax,
            tok_ids_buf,
            _x_buf: x_buf,
            _xb_buf: xb_buf,
            _q_buf: q_buf,
            _k_buf: k_buf,
            _v_buf: v_buf,
            _hb_buf: hb_buf,
            _hb2_buf: hb2_buf,
            logits_buf,
            argmax_buf,
            host_logits: vec![0.0; vocab_size],
        })
    }

    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.k_cache.zero_fill();
            layer.v_cache.zero_fill();
        }
    }

    fn dispatch_linear(
        &self,
        cmd: vk::CommandBuffer,
        lin: &VkLinear,
        n_tok: usize,
        add_residual: bool,
    ) -> Result<(), VulkanError> {
        let groups_x = (lin.rows as u32).div_ceil(4);
        if n_tok == 1 {
            let pipe = self.pipes.gemv_for_dtype(lin.dtype, "linear")?;
            let pc = GemvPush {
                n_in: lin.cols as u32,
                n_out: lin.rows as u32,
                row_bytes: lin.row_bytes as u32,
                add_residual: u32::from(add_residual),
                out_offset: 0,
            };
            pipe.cmd_dispatch(cmd, lin.dset_gemv, as_bytes(&pc), groups_x, 1, 1)
        } else {
            let pipe = self.pipes.gemm_for_dtype(lin.dtype, "linear")?;
            let pc = GemmPush {
                n_in: lin.cols as u32,
                n_out: lin.rows as u32,
                n_tok: n_tok as u32,
                row_bytes: lin.row_bytes as u32,
                add_residual: u32::from(add_residual),
                out_stride: lin.rows as u32,
                out_col_offset: 0,
            };
            let groups_y = (n_tok as u32).div_ceil(8);
            pipe.cmd_dispatch(cmd, lin.dset_gemm, as_bytes(&pc), groups_x, groups_y, 1)
        }
    }

    fn run_chunk_internal(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        compute_logits: bool,
        run_argmax: bool,
    ) -> Result<(), VulkanError> {
        let b = tokens.len();
        assert!((1..=MAX_VK_BATCH).contains(&b));
        self.tok_ids_buf.write_u32_slice(0, tokens);

        let n_embd = self.cfg.n_embd as u32;
        let n_ff = self.cfg.n_ff as u32;
        let hd = self.cfg.head_dim as u32;
        let bt = b as u32;
        let attn_scale = 1.0 / (self.cfg.head_dim as f32).sqrt();

        let ctx = Arc::clone(&self.ctx);
        ctx.submit_and_wait(|cmd| {
            let emb_pc = EmbedPush {
                n_embd,
                row_bytes: self.token_embd_row_bytes as u32,
                dtype_code: self.token_embd_dtype.to_ggml_type(),
                n_tok: bt,
            };
            self.pipes.embed.cmd_dispatch(
                cmd,
                self.dset_embed,
                as_bytes(&emb_pc),
                n_embd.div_ceil(64),
                bt,
                1,
            )?;
            ctx.cmd_compute_barrier(cmd);

            for layer in &self.layers {
                let norm_pc = NormPush {
                    n: n_embd,
                    in_offset: 0,
                    eps: self.cfg.rms_eps,
                };
                self.pipes.rms_norm.cmd_dispatch(
                    cmd,
                    layer.dset_attn_norm,
                    as_bytes(&norm_pc),
                    bt,
                    1,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                // Concurrent Q, K, V projections (disjoint output buffers `q_buf`, `k_buf`, `v_buf`)
                self.dispatch_linear(cmd, &layer.wq, b, false)?;
                self.dispatch_linear(cmd, &layer.wk, b, false)?;
                self.dispatch_linear(cmd, &layer.wv, b, false)?;
                ctx.cmd_compute_barrier(cmd);

                if layer.dset_q_norm.is_some() || layer.dset_k_norm.is_some() {
                    if let Some(ds_qn) = layer.dset_q_norm {
                        let qk_pc = QkNormPush {
                            n_heads: self.cfg.n_head as u32,
                            head_dim: hd,
                            eps: self.cfg.rms_eps,
                        };
                        self.pipes.qk_norm.cmd_dispatch(
                            cmd,
                            ds_qn,
                            as_bytes(&qk_pc),
                            self.cfg.n_head as u32,
                            bt,
                            1,
                        )?;
                    }
                    if let Some(ds_kn) = layer.dset_k_norm {
                        let qk_pc = QkNormPush {
                            n_heads: self.cfg.n_head_kv as u32,
                            head_dim: hd,
                            eps: self.cfg.rms_eps,
                        };
                        self.pipes.qk_norm.cmd_dispatch(
                            cmd,
                            ds_kn,
                            as_bytes(&qk_pc),
                            self.cfg.n_head_kv as u32,
                            bt,
                            1,
                        )?;
                    }
                    ctx.cmd_compute_barrier(cmd);
                }

                let rkv_pc = RopeKvPush {
                    n_head: self.cfg.n_head as u32,
                    n_head_kv: self.cfg.n_head_kv as u32,
                    head_dim: hd,
                    rope_dim: self.cfg.rope_dim as u32,
                    start_pos: start_pos as u32,
                    n_tok: bt,
                    rope_type: self.cfg.rope_type,
                    has_freq_factors: u32::from(self.has_rope_freqs),
                    freq_base: self.cfg.rope_freq_base,
                };
                self.pipes.rope_kv.cmd_dispatch(
                    cmd,
                    layer.dset_rope_kv,
                    as_bytes(&rkv_pc),
                    self.cfg.n_head as u32,
                    bt,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                let attn_pc = AttnPush {
                    n_head: self.cfg.n_head as u32,
                    n_head_kv: self.cfg.n_head_kv as u32,
                    head_dim: hd,
                    start_pos: start_pos as u32,
                    n_tok: bt,
                    scale: attn_scale,
                };
                self.pipes.gqa_attn.cmd_dispatch(
                    cmd,
                    layer.dset_gqa_attn,
                    as_bytes(&attn_pc),
                    self.cfg.n_head as u32,
                    bt,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                // `wo` adds directly onto `x_buf` (`add_residual = true`)!
                self.dispatch_linear(cmd, &layer.wo, b, true)?;
                ctx.cmd_compute_barrier(cmd);

                self.pipes.rms_norm.cmd_dispatch(
                    cmd,
                    layer.dset_ffn_norm,
                    as_bytes(&norm_pc),
                    bt,
                    1,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                // Concurrent `w_gate` + `w_up`
                self.dispatch_linear(cmd, &layer.w_gate, b, false)?;
                self.dispatch_linear(cmd, &layer.w_up, b, false)?;
                ctx.cmd_compute_barrier(cmd);

                let swiglu_elems = bt * n_ff;
                let el_pc = ElemPush {
                    total_elems: swiglu_elems,
                };
                self.pipes.swiglu.cmd_dispatch(
                    cmd,
                    layer.dset_swiglu,
                    as_bytes(&el_pc),
                    swiglu_elems.div_ceil(256),
                    1,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                // `w_down` adds directly onto `x_buf` (`add_residual = true`)!
                self.dispatch_linear(cmd, &layer.w_down, b, true)?;
                ctx.cmd_compute_barrier(cmd);
            }

            if compute_logits {
                let last_in_off = ((b - 1) as u32) * n_embd;
                let out_norm_pc = NormPush {
                    n: n_embd,
                    in_offset: last_in_off,
                    eps: self.cfg.rms_eps,
                };
                self.pipes.rms_norm.cmd_dispatch(
                    cmd,
                    self.dset_output_norm,
                    as_bytes(&out_norm_pc),
                    1,
                    1,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                self.dispatch_linear(cmd, &self.output, 1, false)?;
                ctx.cmd_compute_barrier(cmd);

                if run_argmax {
                    let am_pc = ArgmaxPush {
                        vocab_size: self.cfg.vocab_size as u32,
                        logits_offset: 0,
                    };
                    self.pipes.argmax.cmd_dispatch(
                        cmd,
                        self.dset_argmax,
                        as_bytes(&am_pc),
                        1,
                        1,
                        1,
                    )?;
                }
            }
            Ok(())
        })
    }

    pub fn forward_token(
        &mut self,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        if pos >= self.cfg.max_ctx {
            return Err(VulkanError::ContextOverflow {
                pos,
                max_ctx: self.cfg.max_ctx,
            });
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(VulkanError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }
        self.run_chunk_internal(&[token_id], pos, compute_logits, false)?;
        if compute_logits {
            self.logits_buf.read_f32_slice(0, &mut self.host_logits);
        }
        Ok(&self.host_logits)
    }

    /// Fast single-command-buffer greedy decode step: runs GPU `argmax` and reads back ONLY 4 bytes!
    pub fn forward_token_greedy(&mut self, token_id: u32, pos: usize) -> Result<u32, VulkanError> {
        if pos >= self.cfg.max_ctx {
            return Err(VulkanError::ContextOverflow {
                pos,
                max_ctx: self.cfg.max_ctx,
            });
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(VulkanError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }
        self.run_chunk_internal(&[token_id], pos, true, true)?;
        Ok(self.argmax_buf.read_u32(0))
    }

    pub fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<&[f32], VulkanError> {
        if token_ids.is_empty() {
            return Ok(&self.host_logits);
        }
        let end_pos = start_pos.saturating_add(token_ids.len());
        if end_pos > self.cfg.max_ctx {
            return Err(VulkanError::ContextOverflow {
                pos: end_pos - 1,
                max_ctx: self.cfg.max_ctx,
            });
        }
        for &tid in token_ids {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(VulkanError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }

        let mut offset = 0usize;
        while offset < token_ids.len() {
            let b = (token_ids.len() - offset).min(MAX_VK_BATCH);
            let is_last = offset + b == token_ids.len();
            self.run_chunk_internal(
                &token_ids[offset..offset + b],
                start_pos + offset,
                is_last && compute_last_logits,
                false,
            )?;
            offset += b;
        }
        if compute_last_logits {
            self.logits_buf.read_f32_slice(0, &mut self.host_logits);
        }
        Ok(&self.host_logits)
    }
}
