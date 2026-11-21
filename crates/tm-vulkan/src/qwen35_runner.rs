//! Single-command-buffer Vulkan runner for `qwen35` (hybrid Gated DeltaNet + Full Attention).
//!
//! Strictly enforces all 5 GDN invariants (`AGENTS.md` §9 & `Strata/docs/GDN_CONTRACT.md`):
//! 1. Causal `conv1d` followed by **SiLU BEFORE splitting** `q`, `k`, `v`.
//! 2. Per-head L2-normalization on `q` and `k` ONLY (**never** `v`).
//! 3. Tiled head grouping `kh = vh % n_k_heads`.
//! 4. Query scaled by `1 / sqrt(head_v_dim)` on the updated state.
//! 5. Per-head gated `RMSNorm(o_h, ssm_norm) * SiLU(z_h)` on GDN layers, and `sigmoid` gate on FullAttn layers.

use std::sync::Arc;

use ash::vk;
use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufError, GgufFile};

use crate::context::{DescriptorArena, GpuBuffer, VulkanContext, VulkanError, as_bytes};
use crate::pipelines::{
    ArgmaxPush, AttnPush, ElemPush, EmbedPush, GdnConvPush, GdnGatesPush, GdnRecurPush, GemmPush,
    GemvPush, NormPush, QkNormPush, RopeKvPush, SplitQgPush, VulkanPipelines,
};

const MAX_VK_BATCH: usize = 64;

#[derive(Debug, Clone)]
pub struct Qwen35VulkanConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
}

impl Qwen35VulkanConfig {
    #[inline]
    pub const fn conv_dim(&self) -> usize {
        2 * self.ssm_n_group * self.ssm_state_size + self.ssm_d_inner
    }
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

enum VkQwen35Attn {
    FullAttn {
        wqg: VkLinear,
        wk: VkLinear,
        wv: VkLinear,
        _q_norm: GpuBuffer,
        dset_split_qg: vk::DescriptorSet,
        _k_norm: GpuBuffer,
        dset_k_norm: vk::DescriptorSet,
        k_cache: GpuBuffer,
        v_cache: GpuBuffer,
        dset_rope_kv: vk::DescriptorSet,
        dset_gqa_attn: vk::DescriptorSet,
        dset_sigmoid_gate: vk::DescriptorSet,
        wo: VkLinear,
    },
    Gdn {
        wqkv: VkLinear,
        wz: VkLinear,
        w_alpha: VkLinear,
        w_beta: VkLinear,
        _conv1d: GpuBuffer,
        _ssm_a: GpuBuffer,
        _ssm_dt_bias: GpuBuffer,
        _ssm_norm: GpuBuffer,
        ssm_state: GpuBuffer,
        conv_hist: GpuBuffer,
        dset_gates: vk::DescriptorSet,
        dset_conv: vk::DescriptorSet,
        dset_recur: vk::DescriptorSet,
        w_out: VkLinear,
    },
}

struct VkQwen35Layer {
    _attn_norm: GpuBuffer,
    dset_attn_norm: vk::DescriptorSet,
    attn: VkQwen35Attn,
    _post_attn_norm: GpuBuffer,
    dset_post_attn_norm: vk::DescriptorSet,
    w_gate: VkLinear,
    w_up: VkLinear,
    dset_swiglu: vk::DescriptorSet,
    w_down: VkLinear,
}

/// End-to-end Vulkan GPU runner for `qwen35` hybrid models.
pub struct Qwen35VulkanRunner {
    pub ctx: Arc<VulkanContext>,
    pub cfg: Qwen35VulkanConfig,
    pipes: VulkanPipelines,
    _desc_arena: DescriptorArena,
    _token_embd: Arc<GpuBuffer>,
    token_embd_dtype: DType,
    token_embd_row_bytes: usize,
    dset_embed: vk::DescriptorSet,
    _dummy_freqs: GpuBuffer,
    layers: Vec<VkQwen35Layer>,
    _output_norm: GpuBuffer,
    dset_output_norm: vk::DescriptorSet,
    output: VkLinear,
    dset_argmax: vk::DescriptorSet,
    tok_ids_buf: GpuBuffer,
    _x_buf: GpuBuffer,
    _xb_buf: GpuBuffer,
    _qkv_buf: GpuBuffer,
    _q_buf: GpuBuffer,
    _k_buf: GpuBuffer,
    _v_buf: GpuBuffer,
    _gate_buf: GpuBuffer,
    _attn_out_buf: GpuBuffer,
    _alpha_buf: GpuBuffer,
    _beta_buf: GpuBuffer,
    _g_exp_buf: GpuBuffer,
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
    expected_dims: &[usize],
) -> Result<Vec<f32>, VulkanError> {
    let (info, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
    let len: usize = expected_dims.iter().product();
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

impl Qwen35VulkanRunner {
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, VulkanError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "qwen35" {
            return Err(VulkanError::UnsupportedArch(arch));
        }

        let pfx = "qwen35";
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
            .unwrap_or(256);
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(64);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000_000.0);
        let rms_eps = gguf
            .get_meta(&format!("{pfx}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let ssm_conv_kernel = gguf.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_state_size = gguf.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_group = gguf.req_usize(&format!("{pfx}.ssm.group_count"))?;
        let ssm_dt_rank = gguf.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_d_inner = gguf.req_usize(&format!("{pfx}.ssm.inner_size"))?;

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

        let cfg = Qwen35VulkanConfig {
            arch,
            n_layers,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_eps,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            vocab_size,
            max_ctx,
        };

        let ctx = VulkanContext::new()?;
        let pipes = VulkanPipelines::new(&ctx)?;
        let max_sets = (n_layers as u32) * 32 + 32;
        let desc_arena = DescriptorArena::new(&ctx, max_sets, max_sets * 9)?;

        let q_dim = n_head * head_dim;
        let qg_dim = q_dim * 2;
        let kv_dim = n_head_kv * head_dim;
        let conv_dim = cfg.conv_dim();
        let k_gdn_dim = ssm_n_group * ssm_state_size;
        let v_gdn_dim = ssm_d_inner;
        let hist_len = ssm_conv_kernel.saturating_sub(1);
        let head_v_dim = ssm_d_inner / ssm_dt_rank;

        let tok_ids_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * 4)?;
        let x_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_embd * 4)?;
        let xb_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_embd * 4)?;
        let qkv_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * qg_dim.max(conv_dim) * 4)?;
        let q_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * q_dim.max(k_gdn_dim) * 4)?;
        let k_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * kv_dim.max(k_gdn_dim) * 4)?;
        let v_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * kv_dim.max(v_gdn_dim) * 4)?;
        let gate_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * q_dim.max(v_gdn_dim) * 4)?;
        let attn_out_buf =
            GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * q_dim.max(v_gdn_dim) * 4)?;
        let alpha_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * ssm_dt_rank * 4)?;
        let beta_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * ssm_dt_rank * 4)?;
        let g_exp_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * ssm_dt_rank * 4)?;
        let hb_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_ff * 4)?;
        let hb2_buf = GpuBuffer::new_host_visible(&ctx, MAX_VK_BATCH * n_ff * 4)?;
        let logits_buf = GpuBuffer::new_host_visible(&ctx, vocab_size * 4)?;
        let argmax_buf = GpuBuffer::new_host_visible(&ctx, 16)?;
        let dummy_freqs = GpuBuffer::from_f32_slice(&ctx, &[1.0f32; 4])?;

        let (emb_ti, emb_bytes) =
            gguf.req_tensor("token_embd.weight", "global", &[n_embd, vocab_size])?;
        let token_embd = Arc::new(GpuBuffer::from_bytes_device_optimal(&ctx, emb_bytes)?);
        let token_embd_dtype = emb_ti.dtype;
        let token_embd_row_bytes = emb_ti.dtype.row_bytes(n_embd);
        let dset_embed =
            desc_arena.alloc_and_write(&pipes.embed, &[&token_embd, &tok_ids_buf, &x_buf])?;

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let ltag = format!("blk.{l}");
            let attn_norm_v =
                load_f32_vec(gguf, &format!("{ltag}.attn_norm.weight"), &ltag, &[n_embd])?;
            let attn_norm = GpuBuffer::from_f32_slice(&ctx, &attn_norm_v)?;
            let dset_attn_norm =
                desc_arena.alloc_and_write(&pipes.rms_norm, &[&x_buf, &attn_norm, &xb_buf])?;

            let post_norm_name = if gguf
                .tensor_info(&format!("{ltag}.post_attention_norm.weight"))
                .is_some()
            {
                format!("{ltag}.post_attention_norm.weight")
            } else {
                format!("{ltag}.ffn_norm.weight")
            };
            let post_norm_v = load_f32_vec(gguf, &post_norm_name, &ltag, &[n_embd])?;
            let post_attn_norm = GpuBuffer::from_f32_slice(&ctx, &post_norm_v)?;
            let dset_post_attn_norm =
                desc_arena.alloc_and_write(&pipes.rms_norm, &[&x_buf, &post_attn_norm, &xb_buf])?;

            let is_gdn = gguf
                .tensor_info(&format!("{ltag}.attn_qkv.weight"))
                .is_some();
            let attn = if is_gdn {
                let (wqkv, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.attn_qkv.weight"),
                    &ltag,
                    n_embd,
                    conv_dim,
                    &xb_buf,
                    &qkv_buf,
                    None,
                )?;
                let (wz, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.attn_gate.weight"),
                    &ltag,
                    n_embd,
                    v_gdn_dim,
                    &xb_buf,
                    &gate_buf,
                    None,
                )?;
                let (w_alpha, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.ssm_alpha.weight"),
                    &ltag,
                    n_embd,
                    ssm_dt_rank,
                    &xb_buf,
                    &alpha_buf,
                    None,
                )?;
                let (w_beta, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.ssm_beta.weight"),
                    &ltag,
                    n_embd,
                    ssm_dt_rank,
                    &xb_buf,
                    &beta_buf,
                    None,
                )?;

                let conv1d_v = load_f32_vec(
                    gguf,
                    &format!("{ltag}.ssm_conv1d.weight"),
                    &ltag,
                    &[ssm_conv_kernel, conv_dim],
                )?;
                let conv1d = GpuBuffer::from_f32_slice(&ctx, &conv1d_v)?;
                let ssm_a_v = load_f32_vec(gguf, &format!("{ltag}.ssm_a"), &ltag, &[ssm_dt_rank])?;
                let ssm_a = GpuBuffer::from_f32_slice(&ctx, &ssm_a_v)?;
                let dt_bias_v =
                    load_f32_vec(gguf, &format!("{ltag}.ssm_dt.bias"), &ltag, &[ssm_dt_rank])?;
                let ssm_dt_bias = GpuBuffer::from_f32_slice(&ctx, &dt_bias_v)?;
                let ssm_norm_v = load_f32_vec(
                    gguf,
                    &format!("{ltag}.ssm_norm.weight"),
                    &ltag,
                    &[head_v_dim],
                )?;
                let ssm_norm = GpuBuffer::from_f32_slice(&ctx, &ssm_norm_v)?;

                let ssm_state = GpuBuffer::new_host_visible(
                    &ctx,
                    ssm_dt_rank * ssm_state_size * head_v_dim * 4,
                )?;
                let conv_hist = GpuBuffer::new_host_visible(&ctx, conv_dim * hist_len * 4)?;

                let dset_gates = desc_arena.alloc_and_write(
                    &pipes.gdn_gates,
                    &[&alpha_buf, &beta_buf, &ssm_a, &ssm_dt_bias, &g_exp_buf],
                )?;
                let dset_conv = desc_arena.alloc_and_write(
                    &pipes.gdn_conv_silu_l2,
                    &[&qkv_buf, &conv_hist, &conv1d, &q_buf, &k_buf, &v_buf],
                )?;
                let dset_recur = desc_arena.alloc_and_write(
                    &pipes.gdn_recur_gated,
                    &[
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        &g_exp_buf,
                        &beta_buf,
                        &ssm_state,
                        &gate_buf,
                        &ssm_norm,
                        &attn_out_buf,
                    ],
                )?;

                let (w_out, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.ssm_out.weight"),
                    &ltag,
                    v_gdn_dim,
                    n_embd,
                    &attn_out_buf,
                    &x_buf,
                    None,
                )?;

                VkQwen35Attn::Gdn {
                    wqkv,
                    wz,
                    w_alpha,
                    w_beta,
                    _conv1d: conv1d,
                    _ssm_a: ssm_a,
                    _ssm_dt_bias: ssm_dt_bias,
                    _ssm_norm: ssm_norm,
                    ssm_state,
                    conv_hist,
                    dset_gates,
                    dset_conv,
                    dset_recur,
                    w_out,
                }
            } else {
                let (wqg, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.attn_q.weight"),
                    &ltag,
                    n_embd,
                    qg_dim,
                    &xb_buf,
                    &qkv_buf,
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

                let q_norm_v = load_f32_vec(
                    gguf,
                    &format!("{ltag}.attn_q_norm.weight"),
                    &ltag,
                    &[head_dim],
                )?;
                let q_norm = GpuBuffer::from_f32_slice(&ctx, &q_norm_v)?;
                let dset_split_qg = desc_arena.alloc_and_write(
                    &pipes.split_qg_norm,
                    &[&qkv_buf, &q_norm, &q_buf, &gate_buf],
                )?;

                let k_norm_v = load_f32_vec(
                    gguf,
                    &format!("{ltag}.attn_k_norm.weight"),
                    &ltag,
                    &[head_dim],
                )?;
                let k_norm = GpuBuffer::from_f32_slice(&ctx, &k_norm_v)?;
                let dset_k_norm = desc_arena.alloc_and_write(&pipes.qk_norm, &[&k_buf, &k_norm])?;

                let kv_bytes = max_ctx * kv_dim * 2;
                let k_cache = GpuBuffer::new_host_visible(&ctx, kv_bytes)?;
                let v_cache = GpuBuffer::new_host_visible(&ctx, kv_bytes)?;
                let dset_rope_kv = desc_arena.alloc_and_write(
                    &pipes.rope_kv,
                    &[&q_buf, &k_buf, &v_buf, &k_cache, &v_cache, &dummy_freqs],
                )?;
                let dset_gqa_attn = desc_arena.alloc_and_write(
                    &pipes.gqa_attn,
                    &[&q_buf, &k_cache, &v_cache, &attn_out_buf],
                )?;
                let dset_sigmoid_gate =
                    desc_arena.alloc_and_write(&pipes.sigmoid_gate, &[&attn_out_buf, &gate_buf])?;

                let (wo, _) = upload_linear(
                    &ctx,
                    &pipes,
                    &desc_arena,
                    gguf,
                    &format!("{ltag}.attn_output.weight"),
                    &ltag,
                    q_dim,
                    n_embd,
                    &attn_out_buf,
                    &x_buf,
                    None,
                )?;

                VkQwen35Attn::FullAttn {
                    wqg,
                    wk,
                    wv,
                    _q_norm: q_norm,
                    dset_split_qg,
                    _k_norm: k_norm,
                    dset_k_norm,
                    k_cache,
                    v_cache,
                    dset_rope_kv,
                    dset_gqa_attn,
                    dset_sigmoid_gate,
                    wo,
                }
            };

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

            layers.push(VkQwen35Layer {
                _attn_norm: attn_norm,
                dset_attn_norm,
                attn,
                _post_attn_norm: post_attn_norm,
                dset_post_attn_norm,
                w_gate,
                w_up,
                dset_swiglu,
                w_down,
            });
        }

        let out_norm_v = load_f32_vec(gguf, "output_norm.weight", "global", &[n_embd])?;
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

        Ok(Self {
            ctx,
            cfg,
            pipes,
            _desc_arena: desc_arena,
            _token_embd: token_embd,
            token_embd_dtype,
            token_embd_row_bytes,
            dset_embed,
            _dummy_freqs: dummy_freqs,
            layers,
            _output_norm: output_norm,
            dset_output_norm,
            output,
            dset_argmax,
            tok_ids_buf,
            _x_buf: x_buf,
            _xb_buf: xb_buf,
            _qkv_buf: qkv_buf,
            _q_buf: q_buf,
            _k_buf: k_buf,
            _v_buf: v_buf,
            _gate_buf: gate_buf,
            _attn_out_buf: attn_out_buf,
            _alpha_buf: alpha_buf,
            _beta_buf: beta_buf,
            _g_exp_buf: g_exp_buf,
            _hb_buf: hb_buf,
            _hb2_buf: hb2_buf,
            logits_buf,
            argmax_buf,
            host_logits: vec![0.0; vocab_size],
        })
    }

    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.attn {
                VkQwen35Attn::FullAttn {
                    k_cache, v_cache, ..
                } => {
                    k_cache.zero_fill();
                    v_cache.zero_fill();
                }
                VkQwen35Attn::Gdn {
                    ssm_state,
                    conv_hist,
                    ..
                } => {
                    ssm_state.zero_fill();
                    conv_hist.zero_fill();
                }
            }
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

        let bt = b as u32;
        let n_embd = self.cfg.n_embd as u32;
        let n_ff = self.cfg.n_ff as u32;
        let hd = self.cfg.head_dim as u32;
        let q_dim = (self.cfg.n_head * self.cfg.head_dim) as u32;
        let hk = self.cfg.ssm_n_group as u32;
        let hv = self.cfg.ssm_dt_rank as u32;
        let s_dim = self.cfg.ssm_state_size as u32;
        let d_conv = self.cfg.ssm_conv_kernel as u32;
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

                match &layer.attn {
                    VkQwen35Attn::Gdn {
                        wqkv,
                        wz,
                        w_alpha,
                        w_beta,
                        dset_gates,
                        dset_conv,
                        dset_recur,
                        w_out,
                        ..
                    } => {
                        // All 4 input projections read `xb_buf` and write disjoint buffers concurrently!
                        self.dispatch_linear(cmd, wqkv, b, false)?;
                        self.dispatch_linear(cmd, wz, b, false)?;
                        self.dispatch_linear(cmd, w_alpha, b, false)?;
                        self.dispatch_linear(cmd, w_beta, b, false)?;
                        ctx.cmd_compute_barrier(cmd);

                        // Scalar gates + causal Conv1d+SiLU+L2Norm are ALSO independent!
                        let gates_pc = GdnGatesPush {
                            n_v_heads: hv,
                            n_tok: bt,
                        };
                        self.pipes.gdn_gates.cmd_dispatch(
                            cmd,
                            *dset_gates,
                            as_bytes(&gates_pc),
                            (hv * bt).div_ceil(64),
                            1,
                            1,
                        )?;

                        let conv_pc = GdnConvPush {
                            n_k_heads: hk,
                            n_v_heads: hv,
                            s_dim,
                            d_conv,
                            n_tok: bt,
                            eps: self.cfg.rms_eps,
                        };
                        self.pipes.gdn_conv_silu_l2.cmd_dispatch(
                            cmd,
                            *dset_conv,
                            as_bytes(&conv_pc),
                            2 * hk + hv,
                            1,
                            1,
                        )?;
                        ctx.cmd_compute_barrier(cmd);

                        let recur_pc = GdnRecurPush {
                            n_k_heads: hk,
                            n_v_heads: hv,
                            s_dim,
                            n_tok: bt,
                            eps: self.cfg.rms_eps,
                        };
                        self.pipes.gdn_recur_gated.cmd_dispatch(
                            cmd,
                            *dset_recur,
                            as_bytes(&recur_pc),
                            hv,
                            1,
                            1,
                        )?;
                        ctx.cmd_compute_barrier(cmd);

                        self.dispatch_linear(cmd, w_out, b, true)?;
                        ctx.cmd_compute_barrier(cmd);
                    }
                    VkQwen35Attn::FullAttn {
                        wqg,
                        wk,
                        wv,
                        dset_split_qg,
                        dset_k_norm,
                        dset_rope_kv,
                        dset_gqa_attn,
                        dset_sigmoid_gate,
                        wo,
                        ..
                    } => {
                        self.dispatch_linear(cmd, wqg, b, false)?;
                        self.dispatch_linear(cmd, wk, b, false)?;
                        self.dispatch_linear(cmd, wv, b, false)?;
                        ctx.cmd_compute_barrier(cmd);

                        let sqg_pc = SplitQgPush {
                            n_head: self.cfg.n_head as u32,
                            head_dim: hd,
                            eps: self.cfg.rms_eps,
                        };
                        self.pipes.split_qg_norm.cmd_dispatch(
                            cmd,
                            *dset_split_qg,
                            as_bytes(&sqg_pc),
                            self.cfg.n_head as u32,
                            bt,
                            1,
                        )?;

                        let kn_pc = QkNormPush {
                            n_heads: self.cfg.n_head_kv as u32,
                            head_dim: hd,
                            eps: self.cfg.rms_eps,
                        };
                        self.pipes.qk_norm.cmd_dispatch(
                            cmd,
                            *dset_k_norm,
                            as_bytes(&kn_pc),
                            self.cfg.n_head_kv as u32,
                            bt,
                            1,
                        )?;
                        ctx.cmd_compute_barrier(cmd);

                        let rkv_pc = RopeKvPush {
                            n_head: self.cfg.n_head as u32,
                            n_head_kv: self.cfg.n_head_kv as u32,
                            head_dim: hd,
                            rope_dim: self.cfg.rope_dim as u32,
                            start_pos: start_pos as u32,
                            n_tok: bt,
                            rope_type: 2, // NeoX partial M-RoPE
                            has_freq_factors: 0,
                            freq_base: self.cfg.rope_freq_base,
                        };
                        self.pipes.rope_kv.cmd_dispatch(
                            cmd,
                            *dset_rope_kv,
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
                            *dset_gqa_attn,
                            as_bytes(&attn_pc),
                            self.cfg.n_head as u32,
                            bt,
                            1,
                        )?;
                        ctx.cmd_compute_barrier(cmd);

                        let gate_elems = bt * q_dim;
                        let sig_pc = ElemPush {
                            total_elems: gate_elems,
                        };
                        self.pipes.sigmoid_gate.cmd_dispatch(
                            cmd,
                            *dset_sigmoid_gate,
                            as_bytes(&sig_pc),
                            gate_elems.div_ceil(256),
                            1,
                            1,
                        )?;
                        ctx.cmd_compute_barrier(cmd);

                        self.dispatch_linear(cmd, wo, b, true)?;
                        ctx.cmd_compute_barrier(cmd);
                    }
                }

                self.pipes.rms_norm.cmd_dispatch(
                    cmd,
                    layer.dset_post_attn_norm,
                    as_bytes(&norm_pc),
                    bt,
                    1,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

                self.dispatch_linear(cmd, &layer.w_gate, b, false)?;
                self.dispatch_linear(cmd, &layer.w_up, b, false)?;
                ctx.cmd_compute_barrier(cmd);

                let sw_elems = bt * n_ff;
                let sw_pc = ElemPush {
                    total_elems: sw_elems,
                };
                self.pipes.swiglu.cmd_dispatch(
                    cmd,
                    layer.dset_swiglu,
                    as_bytes(&sw_pc),
                    sw_elems.div_ceil(256),
                    1,
                    1,
                )?;
                ctx.cmd_compute_barrier(cmd);

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
