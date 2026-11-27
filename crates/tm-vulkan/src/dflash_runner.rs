//! 100% Vulkan GPU `DSpark` / `DFlash` (`general.architecture == "dflash"`)
//! 3-layer non-causal block speculative drafter + chained recurrent Markov head!
//!
//! Executes BOTH committed target-tap KV injection (`inject_batch`) AND the entire `bs = 8`
//! non-causal block Transformer + `k`-step recurrent Markov head + `lm_head` + softcap-argmax
//! (`draft_block_greedy` / `inject_and_draft_greedy`) in **ONE single Vulkan command buffer**
//! with zero per-step CPU-GPU sync roundtrips, borrowing `tok_embd` and `output` (`lm_head`)
//! directly from [`crate::Inkling2VkRunner`] (`938 MiB` duplicate VRAM saved!).

use std::sync::Arc;

use tm_core::DType;
use tm_gguf::{DirectTensorReader, GgufError, GgufFile, PageAlignedSlot};

use crate::VulkanError;
use crate::context::{Pipeline, Recorder, VkBuffer};
use crate::inkling2_runner::{Inkling2VkRunner, VkDenseMat, push_f32_param, upload_dense_mat};

const SPV_DFLASH_OPS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dflash_ops.spv"));
const SPV_DFLASH_ATTN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dflash_attn.spv"));
const SPV_SILU_MUL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul.spv"));

const MAX_DFLASH_VK_BLOCK: usize = 16;

/// Parsed hyperparameters for [`DFlashVkRunner`].
#[derive(Debug, Clone)]
pub struct DFlashVkConfig {
    pub n_layers: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub n_swa_eff: usize,
    pub is_swa: Vec<bool>,
    pub markov_rank: usize,
    pub target_layers: Vec<usize>,
    pub vocab_padded: usize,
    pub vocab_size: usize,
    pub final_logit_softcapping: f32,
    pub max_ctx: usize,
}

impl DFlashVkConfig {
    pub fn from_gguf(gguf: &GgufFile, max_ctx: usize) -> Result<Self, VulkanError> {
        let arch = gguf.req_str("general.architecture")?;
        if arch != "dflash" {
            return Err(VulkanError::UnsupportedArch(arch.to_string()));
        }
        let n_layers = gguf.req_usize("dflash.block_count")?;
        let block_size = gguf.req_usize("dflash.block_size")?;
        if !(2..=MAX_DFLASH_VK_BLOCK).contains(&block_size) {
            return Err(VulkanError::InvalidDimensions(format!(
                "dflash.block_size={block_size} outside 2..={MAX_DFLASH_VK_BLOCK}"
            )));
        }
        let n_embd = gguf.req_usize("dflash.embedding_length")?;
        let n_ff = gguf.req_usize("dflash.feed_forward_length")?;
        let n_head = gguf.req_usize("dflash.attention.head_count")?;
        let n_head_kv = gguf
            .get_meta("dflash.attention.head_count_kv")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_head);
        let head_dim = gguf
            .get_meta("dflash.attention.key_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(n_embd / n_head.max(1));
        if !head_dim.is_multiple_of(4) || head_dim > 256 {
            return Err(VulkanError::InvalidDimensions(format!(
                "dflash head_dim={head_dim} must be a multiple of 4 and <= 256"
            )));
        }
        let rope_dim = gguf
            .get_meta("dflash.rope.dimension_count")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        let rope_freq_base = gguf
            .get_meta("dflash.rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(1_000_000.0);
        let rms_eps = gguf
            .get_meta("dflash.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let raw_swa = gguf.req_usize("dflash.attention.sliding_window")?;
        let n_swa_eff = raw_swa + 1;
        let is_swa = gguf
            .get_meta("dflash.attention.sliding_window_pattern")
            .and_then(|v| v.as_bool_array())
            .unwrap_or_else(|| vec![true; n_layers]);
        let target_layers = gguf
            .get_meta("dflash.target_layers")
            .and_then(|v| v.as_usize_array())
            .ok_or_else(|| {
                VulkanError::InvalidDimensions("missing dflash.target_layers".to_string())
            })?;
        let markov_w1_info =
            gguf.tensor_info("markov_w1.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf.path().display().to_string(),
                    tensor: "markov_w1.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let markov_rank = gguf
            .get_meta("dflash.dspark.markov_rank")
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| markov_w1_info.dims.first().copied().unwrap_or(0));
        let vocab_padded = gguf
            .get_meta("dflash.vocab_size")
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| markov_w1_info.dims.get(1).copied().unwrap_or(0));
        let vocab_size = gguf
            .get_meta("dflash.vocab_size_unpadded")
            .and_then(|v| v.as_usize())
            .unwrap_or(vocab_padded);
        let final_logit_softcapping = gguf
            .get_meta("dflash.final_logit_softcapping")
            .and_then(|v| v.as_f32())
            .unwrap_or(30.0);

        Ok(Self {
            n_layers,
            block_size,
            n_embd,
            n_ff,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_eps,
            n_swa_eff,
            is_swa,
            markov_rank,
            target_layers,
            vocab_padded,
            vocab_size,
            final_logit_softcapping,
            max_ctx,
        })
    }

    #[inline]
    pub const fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    #[inline]
    pub const fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    #[inline]
    pub const fn max_draft_tokens(&self) -> usize {
        self.block_size.saturating_sub(1)
    }
}

struct VkDFlashLayer {
    dflare_norm_off: u32,
    attn_k_ctx: VkDenseMat,
    attn_v_ctx: VkDenseMat,
    attn_k_ctx_norm_off: u32,
    attn_norm_off: u32,
    wq: VkDenseMat,
    wk: VkDenseMat,
    wv: VkDenseMat,
    wo: VkDenseMat,
    attn_q_norm_off: u32,
    attn_k_norm_off: u32,
    ffn_norm_off: u32,
    ffn_gate: VkDenseMat,
    ffn_up: VkDenseMat,
    ffn_down: VkDenseMat,
    kv_u32_base: u32,
    kv_vec4_base: u32,
}

/// 100% GPU `DSpark` (`dflash`) block speculative drafter sharing [`Inkling2VkRunner`]'s
/// `VulkanContext`, `tok_embd`, `output` (`lm_head`), and `dspark_taps_buf`.
pub struct DFlashVkRunner {
    pub cfg: DFlashVkConfig,
    vram_bytes: u64,
    f32_params: VkBuffer,
    fuse_prob_offs: Vec<u32>,
    dspark_norm_off: u32,
    mask_embd_off: u32,
    output_norm_off: u32,
    joint_proj_b_off: u32,
    hidden_out_b_off: u32,
    rope_freq_off: u32,
    markov_w1_bf16: VkBuffer,
    joint_proj: VkDenseMat,
    hidden_out: VkDenseMat,
    layers: Vec<VkDFlashLayer>,
    // State buffers
    k_cache: VkBuffer,
    v_cache: VkBuffer,
    ring_cap: usize,
    ctx_feat_ring: VkBuffer,
    ctx_feat_pos: Vec<isize>,
    // Scratch buffers
    inj_norm_buf: VkBuffer,
    inj_k_buf: VkBuffer,
    inj_v_buf: VkBuffer,
    inp_l_buf: VkBuffer,
    norm_blk_buf: VkBuffer,
    q_blk_buf: VkBuffer,
    k_blk_buf: VkBuffer,
    v_blk_buf: VkBuffer,
    attn_blk_buf: VkBuffer,
    proj_blk_buf: VkBuffer,
    ffn_gate_blk_buf: VkBuffer,
    ffn_up_blk_buf: VkBuffer,
    hidden_blk_buf: VkBuffer,
    rnn_state_buf: VkBuffer,
    joint_in_buf: VkBuffer,
    joint_out_buf: VkBuffer,
    latent_buf: VkBuffer,
    correction_buf: VkBuffer,
    corr_hidden_buf: VkBuffer,
    xq_buf: VkBuffer,
    xd_buf: VkBuffer,
    draft_toks_buf: VkBuffer,
    vocab_map: Option<(VkBuffer, usize)>,
    // Pipelines
    pipe_ops: Arc<Pipeline>,
    pipe_argmax_cap: Arc<Pipeline>,
    pipe_idx_lm: Arc<Pipeline>,
    pipe_attn: Arc<Pipeline>,
    pipe_silu_mul: Arc<Pipeline>,
    ops_wg: u32,
    idx_lm_wg: u32,
    idx_lm_nr: u32,
}

impl DFlashVkRunner {
    /// Uploads `DFlash` (`DSpark`) weights onto `target`'s Vulkan device.
    pub fn from_gguf(
        target: &Inkling2VkRunner,
        gguf: &GgufFile,
        max_ctx: usize,
    ) -> Result<Self, VulkanError> {
        let ctx = &target.ctx;
        let info = ctx.info();
        let cfg = DFlashVkConfig::from_gguf(gguf, max_ctx)?;
        if cfg.n_embd != target.cfg.n_embd || cfg.vocab_size != target.cfg.vocab_size {
            return Err(VulkanError::InvalidDimensions(format!(
                "dflash (n_embd={}, vocab={}) mismatches target (n_embd={}, vocab={})",
                cfg.n_embd, cfg.vocab_size, target.cfg.n_embd, target.cfg.vocab_size
            )));
        }

        let direct_reader = if gguf.path().exists() {
            DirectTensorReader::open(gguf.path()).ok()
        } else {
            None
        };
        let mut io_slot = PageAlignedSlot::new(16 * 1024 * 1024 + 8192);
        let mut vram_bytes = 0u64;
        let mut f32_pool: Vec<f32> = Vec::with_capacity(131072);

        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hd = cfg.head_dim;
        let rank = cfg.markov_rank;
        let n_taps = cfg.target_layers.len();

        let fuse_raw_off = push_f32_param(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_fuse.weight",
            "global",
            &[n_taps, cfg.n_layers + 1],
            &mut f32_pool,
        )? as usize;
        let mut fuse_prob_offs = Vec::with_capacity(cfg.n_layers + 1);
        for r in 0..=cfg.n_layers {
            let s = fuse_raw_off + r * n_taps;
            let row = &mut f32_pool[s..s + n_taps];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                let e = (*v - max_v).exp();
                *v = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
            fuse_prob_offs.push(s as u32);
        }

        let dspark_norm_off = push_f32_param(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_norm.weight",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;
        let mask_embd_off = push_f32_param(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "mask_embd.weight",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;
        let output_norm_off = push_f32_param(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output_norm.weight",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;
        let joint_proj_b_off = push_f32_param(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "joint_proj.bias",
            "global",
            &[3 * rank],
            &mut f32_pool,
        )?;
        let hidden_out_b_off = push_f32_param(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_hidden_out.bias",
            "global",
            &[n_embd],
            &mut f32_pool,
        )?;

        let half_rope = cfg.rope_dim / 2;
        let rope_freq_off = f32_pool.len() as u32;
        let base = cfg.rope_freq_base as f64;
        for i in 0..half_rope {
            f32_pool.push(base.powf(-((2 * i) as f64) / (cfg.rope_dim as f64)) as f32);
        }

        // Upload `markov_w1.weight` (`BF16` -> raw `u16` bits; `F32`/`F16` converted cleanly to `BF16` `u16` bits)
        // and compute per-token L2 norms `w1_norm2[0..vocab_size]` for importance-ranked draft vocab pruning.
        let mut w1_norm2 = vec![0.0f32; cfg.vocab_size];
        let mw1_buf = if let Some(ref rdr) = direct_reader {
            let (ti, f_off, b_len) = gguf.req_tensor_file_range(
                "markov_w1.weight",
                "global",
                &[rank, cfg.vocab_padded],
            )?;
            let mut raw = vec![0u8; b_len];
            rdr.read_exact_uncached(f_off, &mut raw, &mut io_slot)?;
            if ti.dtype == DType::BF16 {
                for (id, dst_n) in w1_norm2.iter_mut().enumerate() {
                    let row_b = &raw[id * rank * 2..(id + 1) * rank * 2];
                    let mut s = 0.0f32;
                    for &c in row_b.as_chunks::<2>().0 {
                        let u = u16::from_le_bytes(c) as u32;
                        let v = f32::from_bits(u << 16);
                        s += v * v;
                    }
                    *dst_n = s;
                }
                let b = ctx.upload_bytes(&raw)?;
                vram_bytes += b_len as u64;
                b
            } else {
                let mut tmp_f32 = vec![0.0f32; rank * cfg.vocab_padded];
                tm_core::dequantize_row_bytes(ti.dtype, &raw, &mut tmp_f32);
                for (id, dst_n) in w1_norm2.iter_mut().enumerate() {
                    let row = &tmp_f32[id * rank..(id + 1) * rank];
                    *dst_n = row.iter().map(|&v| v * v).sum();
                }
                let bf16_u16: Vec<u16> = tmp_f32
                    .iter()
                    .map(|&v| {
                        ((v.to_bits().wrapping_add(0x7FFF + ((v.to_bits() >> 16) & 1))) >> 16)
                            as u16
                    })
                    .collect();
                let b = ctx.upload(&bf16_u16)?;
                vram_bytes += (bf16_u16.len() * 2) as u64;
                b
            }
        } else {
            let (ti, raw) =
                gguf.req_tensor("markov_w1.weight", "global", &[rank, cfg.vocab_padded])?;
            if ti.dtype == DType::BF16 {
                for (id, dst_n) in w1_norm2.iter_mut().enumerate() {
                    let row_b = &raw[id * rank * 2..(id + 1) * rank * 2];
                    let mut s = 0.0f32;
                    for &c in row_b.as_chunks::<2>().0 {
                        let u = u16::from_le_bytes(c) as u32;
                        let v = f32::from_bits(u << 16);
                        s += v * v;
                    }
                    *dst_n = s;
                }
                let b = ctx.upload_bytes(raw)?;
                vram_bytes += raw.len() as u64;
                b
            } else {
                let mut tmp_f32 = vec![0.0f32; rank * cfg.vocab_padded];
                tm_core::dequantize_row_bytes(ti.dtype, raw, &mut tmp_f32);
                for (id, dst_n) in w1_norm2.iter_mut().enumerate() {
                    let row = &tmp_f32[id * rank..(id + 1) * rank];
                    *dst_n = row.iter().map(|&v| v * v).sum();
                }
                let bf16_u16: Vec<u16> = tmp_f32
                    .iter()
                    .map(|&v| {
                        ((v.to_bits().wrapping_add(0x7FFF + ((v.to_bits() >> 16) & 1))) >> 16)
                            as u16
                    })
                    .collect();
                let b = ctx.upload(&bf16_u16)?;
                vram_bytes += (bf16_u16.len() * 2) as u64;
                b
            }
        };

        let joint_proj = upload_dense_mat(
            ctx,
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "joint_proj.weight",
            "global",
            2 * n_embd + 2 * rank,
            3 * rank,
            &mut vram_bytes,
        )?;
        let hidden_out = upload_dense_mat(
            ctx,
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "dspark_hidden_out.weight",
            "global",
            rank,
            n_embd,
            &mut vram_bytes,
        )?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut kv_u16_cursor = 0usize;
        for il in 0..cfg.n_layers {
            let ltag = format!("blk.{il}");
            let dflare_norm_off = push_f32_param(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.dflare_norm.weight"),
                &ltag,
                &[n_embd],
                &mut f32_pool,
            )?;
            let attn_k_ctx = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_k_ctx.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut vram_bytes,
            )?;
            let attn_v_ctx = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_v_ctx.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut vram_bytes,
            )?;
            let attn_k_ctx_norm_off = push_f32_param(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_k_ctx_norm.weight"),
                &ltag,
                &[hd],
                &mut f32_pool,
            )?;
            let attn_norm_off = push_f32_param(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_norm.weight"),
                &ltag,
                &[n_embd],
                &mut f32_pool,
            )?;
            let wq = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_q.weight"),
                &ltag,
                n_embd,
                q_dim,
                &mut vram_bytes,
            )?;
            let wk = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_k.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut vram_bytes,
            )?;
            let wv = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_v.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut vram_bytes,
            )?;
            let wo = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_output.weight"),
                &ltag,
                q_dim,
                n_embd,
                &mut vram_bytes,
            )?;
            let attn_q_norm_off = push_f32_param(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_q_norm.weight"),
                &ltag,
                &[hd],
                &mut f32_pool,
            )?;
            let attn_k_norm_off = push_f32_param(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.attn_k_norm.weight"),
                &ltag,
                &[hd],
                &mut f32_pool,
            )?;
            let ffn_norm_off = push_f32_param(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.ffn_norm.weight"),
                &ltag,
                &[n_embd],
                &mut f32_pool,
            )?;
            let ffn_gate = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.ffn_gate.weight"),
                &ltag,
                n_embd,
                n_ff,
                &mut vram_bytes,
            )?;
            let ffn_up = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.ffn_up.weight"),
                &ltag,
                n_embd,
                n_ff,
                &mut vram_bytes,
            )?;
            let ffn_down = upload_dense_mat(
                ctx,
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{il}.ffn_down.weight"),
                &ltag,
                n_ff,
                n_embd,
                &mut vram_bytes,
            )?;

            let kv_u32_base = (kv_u16_cursor / 2) as u32;
            let kv_vec4_base = (kv_u16_cursor / 4) as u32;
            kv_u16_cursor += max_ctx * kv_dim;

            layers.push(VkDFlashLayer {
                dflare_norm_off,
                attn_k_ctx,
                attn_v_ctx,
                attn_k_ctx_norm_off,
                attn_norm_off,
                wq,
                wk,
                wv,
                wo,
                attn_q_norm_off,
                attn_k_norm_off,
                ffn_norm_off,
                ffn_gate,
                ffn_up,
                ffn_down,
                kv_u32_base,
                kv_vec4_base,
            });
        }

        let f32_params = ctx.upload(&f32_pool)?;
        vram_bytes += (f32_pool.len() * 4) as u64;

        let kv_bytes = (kv_u16_cursor * 2).max(16) as u64;
        let k_cache = ctx.alloc_buffer(kv_bytes)?;
        let v_cache = ctx.alloc_buffer(kv_bytes)?;
        vram_bytes += 2 * kv_bytes;

        let ring_cap = cfg.block_size + 4;
        let ctx_feat_ring = ctx.alloc_buffer((ring_cap * n_embd * 4).max(16) as u64)?;
        let max_inj = crate::MAX_INKLING2_VK_WINDOW;
        let bs = cfg.block_size;

        let inj_norm_buf = ctx.alloc_buffer((max_inj * n_embd * 4) as u64)?;
        let inj_k_buf = ctx.alloc_buffer((max_inj * kv_dim * 4) as u64)?;
        let inj_v_buf = ctx.alloc_buffer((max_inj * kv_dim * 4) as u64)?;
        let inp_l_buf = ctx.alloc_buffer((bs * n_embd * 4) as u64)?;
        let norm_blk_buf = ctx.alloc_buffer((bs * n_embd * 4) as u64)?;
        let q_blk_buf = ctx.alloc_buffer((bs * q_dim * 4) as u64)?;
        let k_blk_buf = ctx.alloc_buffer((bs * kv_dim * 4) as u64)?;
        let v_blk_buf = ctx.alloc_buffer((bs * kv_dim * 4) as u64)?;
        let attn_blk_buf = ctx.alloc_buffer((bs * q_dim * 4) as u64)?;
        let proj_blk_buf = ctx.alloc_buffer((bs * n_embd * 4) as u64)?;
        let ffn_gate_blk_buf = ctx.alloc_buffer((bs * n_ff * 4) as u64)?;
        let ffn_up_blk_buf = ctx.alloc_buffer((bs * n_ff * 4) as u64)?;
        let hidden_blk_buf = ctx.alloc_buffer((bs * n_embd * 4) as u64)?;
        let rnn_state_buf = ctx.alloc_buffer((rank * 4).max(16) as u64)?;
        let joint_in_buf = ctx.alloc_buffer(((2 * rank + 2 * n_embd) * 4) as u64)?;
        let joint_out_buf = ctx.alloc_buffer(((3 * rank) * 4).max(16) as u64)?;
        let latent_buf = ctx.alloc_buffer((rank * 4).max(16) as u64)?;
        let correction_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let corr_hidden_buf = ctx.alloc_buffer((n_embd * 4) as u64)?;
        let q8_blks = (bs * n_embd.max(n_ff).max(q_dim)).div_ceil(32).max(1);
        let xq_buf = ctx.alloc_buffer((q8_blks * 32) as u64)?;
        let xd_buf = ctx.alloc_buffer((q8_blks * 8).max(16) as u64)?;
        let draft_toks_buf = ctx.create_host_buffer(((2 * bs + 8) * 4) as u64)?;

        // Optional draft vocabulary pruning (`TM_VK_DRAFT_VOCAB=<N>`, ranking active code/text
        // tokens by learned `||markov_w1[id]||_2` + byte/ASCII head + special control tail tokens)
        let draft_vocab_limit = std::env::var("TM_VK_DRAFT_VOCAB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let vocab_map = if draft_vocab_limit > 0
            && draft_vocab_limit < cfg.vocab_size
            && target.output.dtype_code == 2
        {
            let tail = (draft_vocab_limit / 16)
                .clamp(256, 2048)
                .min(draft_vocab_limit);
            let head_bytes = 512usize.min(draft_vocab_limit.saturating_sub(tail));
            let tail_start = (cfg.vocab_size - tail).max(head_bytes);
            let rem_budget =
                draft_vocab_limit.saturating_sub(head_bytes + (cfg.vocab_size - tail_start));

            let mut mid_cands: Vec<u32> = (head_bytes as u32..tail_start as u32).collect();
            mid_cands.sort_unstable_by(|&a, &b| {
                w1_norm2[b as usize]
                    .total_cmp(&w1_norm2[a as usize])
                    .then_with(|| a.cmp(&b))
            });
            mid_cands.truncate(rem_budget);

            let mut ids = Vec::with_capacity(draft_vocab_limit);
            for id in 0..head_bytes {
                ids.push(id as u32);
            }
            ids.extend_from_slice(&mid_cands);
            for id in tail_start..cfg.vocab_size {
                ids.push(id as u32);
            }
            ids.sort_unstable();
            let n_v = ids.len();
            let map_buf = ctx.upload(&ids)?;
            Some((map_buf, n_v))
        } else {
            None
        };

        let sg = info.subgroup_size;
        let max_wg = info.max_workgroup_invocations;
        let ops_wg = 128u32.max(sg).min(max_wg);
        let idx_lm_wg = 128u32.max(sg).min(max_wg);
        let idx_lm_nr = 4u32;
        let attn_sg = if 8 * sg <= max_wg { 8u32 } else { 4u32 };
        let attn_wg = attn_sg * sg;
        let attn_nvec = ((hd / 4).div_ceil(sg as usize)).max(1) as u32;

        let pipe_ops = ctx.pipeline("dflash_ops", ops_wg, SPV_DFLASH_OPS, 6, 32, &[ops_wg, 1])?;
        let pipe_argmax_cap = ctx.pipeline(
            "dflash_ops_argmax",
            256 | 0x1000,
            SPV_DFLASH_OPS,
            6,
            32,
            &[256, 1],
        )?;
        let pipe_idx_lm = ctx.pipeline(
            "dflash_ops_idxlm",
            idx_lm_wg | (idx_lm_nr << 16) | 0x2000,
            SPV_DFLASH_OPS,
            6,
            32,
            &[idx_lm_wg, idx_lm_nr],
        )?;
        let pipe_attn = ctx.pipeline(
            "dflash_attn",
            attn_wg | (attn_nvec << 16),
            SPV_DFLASH_ATTN,
            6,
            20,
            &[attn_wg, attn_sg, attn_nvec],
        )?;
        let silu_wg = 256u32.min(max_wg);
        let pipe_silu_mul =
            ctx.pipeline("dflash_silu_mul", silu_wg, SPV_SILU_MUL, 2, 4, &[silu_wg])?;

        Ok(Self {
            cfg,
            vram_bytes,
            f32_params,
            fuse_prob_offs,
            dspark_norm_off,
            mask_embd_off,
            output_norm_off,
            joint_proj_b_off,
            hidden_out_b_off,
            rope_freq_off,
            markov_w1_bf16: mw1_buf,
            joint_proj,
            hidden_out,
            layers,
            k_cache,
            v_cache,
            ring_cap,
            ctx_feat_ring,
            ctx_feat_pos: vec![-1; ring_cap],
            inj_norm_buf,
            inj_k_buf,
            inj_v_buf,
            inp_l_buf,
            norm_blk_buf,
            q_blk_buf,
            k_blk_buf,
            v_blk_buf,
            attn_blk_buf,
            proj_blk_buf,
            ffn_gate_blk_buf,
            ffn_up_blk_buf,
            hidden_blk_buf,
            rnn_state_buf,
            joint_in_buf,
            joint_out_buf,
            latent_buf,
            correction_buf,
            corr_hidden_buf,
            xq_buf,
            xd_buf,
            draft_toks_buf,
            vocab_map,
            pipe_ops,
            pipe_argmax_cap,
            pipe_idx_lm,
            pipe_attn,
            pipe_silu_mul,
            ops_wg,
            idx_lm_wg,
            idx_lm_nr,
        })
    }

    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub fn reset_state(&mut self, target: &Inkling2VkRunner) -> Result<(), VulkanError> {
        let k_buf = self.k_cache.buf;
        let v_buf = self.v_cache.buf;
        let r_buf = self.ctx_feat_ring.buf;
        target.ctx.run(|rec| {
            let d = &rec.ctx.inner.device;
            // SAFETY: Valid device buffers owned by `self`; `ctx.run` waits on fence before returning.
            unsafe {
                d.cmd_fill_buffer(rec.cb, k_buf, 0, ash::vk::WHOLE_SIZE, 0);
                d.cmd_fill_buffer(rec.cb, v_buf, 0, ash::vk::WHOLE_SIZE, 0);
                d.cmd_fill_buffer(rec.cb, r_buf, 0, ash::vk::WHOLE_SIZE, 0);
            }
            Ok(())
        })?;
        self.ctx_feat_pos.fill(-1);
        Ok(())
    }

    #[inline]
    fn elem_groups(&self, n: usize) -> u32 {
        n.div_ceil(self.ops_wg as usize).min(65535) as u32
    }

    fn record_inject_batch(
        &mut self,
        rec: &mut Recorder<'_>,
        target: &Inkling2VkRunner,
        start_pos: usize,
        count: usize,
    ) -> Result<(), VulkanError> {
        if count == 0 {
            return Ok(());
        }
        if start_pos + count > self.cfg.max_ctx {
            return Err(VulkanError::InvalidDimensions(format!(
                "DFlashVkRunner::inject_batch end_pos {} > max_ctx {}",
                start_pos + count,
                self.cfg.max_ctx
            )));
        }
        let n_embd = self.cfg.n_embd;
        let n_taps = self.cfg.target_layers.len();
        let hd = self.cfg.head_dim;
        let nh_kv = self.cfg.n_head_kv;
        let eps_bits = self.cfg.rms_eps.to_bits();
        let use_tc = count >= 8;

        for il in 0..self.cfg.n_layers {
            let layer = &self.layers[il];
            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &target.dspark_taps_buf,
                    &self.f32_params,
                    &self.inj_norm_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                ],
                &[
                    0,
                    n_embd as u32,
                    n_taps as u32,
                    self.fuse_prob_offs[il],
                    layer.dflare_norm_off,
                    eps_bits,
                    start_pos as u32,
                    0,
                ],
                1,
                count as u32,
            )?;
            target.dispatch_gemm_nb(
                rec,
                &layer.attn_k_ctx,
                &self.inj_norm_buf,
                &self.inj_k_buf,
                count,
                use_tc,
            )?;
            target.dispatch_gemm(
                rec,
                &layer.attn_v_ctx,
                &self.inj_norm_buf,
                &self.inj_v_buf,
                count,
                use_tc,
            )?;
            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.inj_k_buf,
                    &self.inj_v_buf,
                    &self.f32_params,
                    &target.dummy_buf,
                    &self.k_cache,
                    &self.v_cache,
                ],
                &[
                    1,
                    hd as u32,
                    self.cfg.rope_dim as u32,
                    start_pos as u32,
                    layer.attn_k_ctx_norm_off,
                    eps_bits,
                    self.rope_freq_off,
                    layer.kv_u32_base,
                ],
                nh_kv as u32,
                count as u32,
            )?;
        }

        // Write only the most recent `ring_n = count.min(ring_cap)` positions into `ctx_feat_ring`
        // so `(start_pos + tok_skip + t) % ring_cap` is strictly 1-to-1 race-free even when `count == 256`!
        let ring_n = count.min(self.ring_cap);
        let tok_skip = count - ring_n;
        let packed_p4 = ((tok_skip as u32) << 8) | (self.ring_cap as u32);
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &target.dspark_taps_buf,
                &self.f32_params,
                &self.ctx_feat_ring,
                &target.dummy_buf,
                &target.dummy_buf,
                &target.dummy_buf,
            ],
            &[
                0,
                n_embd as u32,
                n_taps as u32,
                self.fuse_prob_offs[self.cfg.n_layers],
                self.dspark_norm_off,
                eps_bits,
                start_pos as u32,
                packed_p4,
            ],
            1,
            ring_n as u32,
        )?;
        for t in 0..ring_n {
            let p = start_pos + tok_skip + t;
            self.ctx_feat_pos[p % self.ring_cap] = p as isize;
        }
        Ok(())
    }

    /// Injects `count` committed target tokens (`[start_pos .. start_pos + count)`) directly
    /// from `target.dspark_taps_buf` on GPU VRAM (`0` CPU readback bytes!).
    pub fn inject_batch(
        &mut self,
        target: &Inkling2VkRunner,
        start_pos: usize,
        count: usize,
    ) -> Result<(), VulkanError> {
        if count == 0 {
            return Ok(());
        }
        let mut err = None;
        target.ctx.run(|rec| {
            if let Err(e) = self.record_inject_batch(rec, target, start_pos, count) {
                err = Some(e);
            }
            Ok(())
        })?;
        if let Some(e) = err {
            return Err(e);
        }
        Ok(())
    }

    fn record_draft_block_greedy(
        &self,
        rec: &mut Recorder<'_>,
        target: &Inkling2VkRunner,
        pos0: usize,
        k: usize,
    ) -> Result<(), VulkanError> {
        let bs = self.cfg.block_size;
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff;
        let nh = self.cfg.n_head;
        let nh_kv = self.cfg.n_head_kv;
        let hd = self.cfg.head_dim;
        let rank = self.cfg.markov_rank;
        let eps_bits = self.cfg.rms_eps.to_bits();
        let cap_bits = self.cfg.final_logit_softcapping.to_bits();
        let prev_pos = pos0 - 1;
        let ring_slot = prev_pos % self.ring_cap;

        // 1. Embed `anchor_token_id` (`target.step_buf[0]`) -> `target.cur_buf`, then build `[bs, n_embd]` `inp_l` + zero `rnn_state`
        target.record_embed_and_norm(rec)?;
        rec.dispatch(
            &self.pipe_ops,
            &[
                &self.inp_l_buf,
                &target.cur_buf,
                &self.f32_params,
                &self.rnn_state_buf,
                &target.dummy_buf,
                &target.dummy_buf,
            ],
            &[
                2,
                n_embd as u32,
                bs as u32,
                self.mask_embd_off,
                rank as u32,
                0,
                0,
                0,
            ],
            self.elem_groups(bs * n_embd),
        )?;

        let fuse_q8 = target.fused_q8_active()
            && self.ops_wg == 128
            && n_embd.is_multiple_of(32)
            && n_ff.is_multiple_of(32);
        let fuse_flag = u32::from(fuse_q8);
        let (q8_xq, q8_xd) = if fuse_q8 {
            (&self.xq_buf, &self.xd_buf)
        } else {
            (&target.dummy_buf, &target.dummy_buf)
        };

        // 2. 3 non-causal block-attention Transformer stages (`bs = 8` rows in lockstep via `M = 8` single-read GEMV!)
        for il in 0..self.cfg.n_layers {
            let layer = &self.layers[il];
            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.inp_l_buf,
                    &self.norm_blk_buf,
                    &self.f32_params,
                    &target.dummy_buf,
                    q8_xq,
                    q8_xd,
                ],
                &[
                    4,
                    n_embd as u32,
                    layer.attn_norm_off,
                    fuse_flag,
                    0,
                    eps_bits,
                    0,
                    0,
                ],
                1,
                bs as u32,
            )?;
            if fuse_q8 {
                target.dispatch_gemm_prequant(
                    rec,
                    &layer.wq,
                    &self.norm_blk_buf,
                    &self.xq_buf,
                    &self.xd_buf,
                    &self.q_blk_buf,
                    bs,
                    false,
                    false,
                )?;
                target.dispatch_gemm_prequant(
                    rec,
                    &layer.wk,
                    &self.norm_blk_buf,
                    &self.xq_buf,
                    &self.xd_buf,
                    &self.k_blk_buf,
                    bs,
                    false,
                    false,
                )?;
                target.dispatch_gemm_prequant(
                    rec,
                    &layer.wv,
                    &self.norm_blk_buf,
                    &self.xq_buf,
                    &self.xd_buf,
                    &self.v_blk_buf,
                    bs,
                    false,
                    true,
                )?;
            } else {
                target.dispatch_gemm_nb(
                    rec,
                    &layer.wq,
                    &self.norm_blk_buf,
                    &self.q_blk_buf,
                    bs,
                    false,
                )?;
                target.dispatch_gemm_nb(
                    rec,
                    &layer.wk,
                    &self.norm_blk_buf,
                    &self.k_blk_buf,
                    bs,
                    false,
                )?;
                target.dispatch_gemm(
                    rec,
                    &layer.wv,
                    &self.norm_blk_buf,
                    &self.v_blk_buf,
                    bs,
                    false,
                )?;
            }

            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.q_blk_buf,
                    &self.k_blk_buf,
                    &self.v_blk_buf,
                    &self.f32_params,
                    &target.dummy_buf,
                    &target.dummy_buf,
                ],
                &[
                    3,
                    (nh as u32) | ((nh_kv as u32) << 16),
                    (hd as u32) | ((self.cfg.rope_dim as u32) << 16),
                    pos0 as u32,
                    layer.attn_q_norm_off,
                    eps_bits,
                    self.rope_freq_off,
                    layer.attn_k_norm_off,
                ],
                (nh + nh_kv) as u32,
                bs as u32,
            )?;

            let ctx_start = if self.cfg.is_swa[il] {
                (pos0 + 1).saturating_sub(self.cfg.n_swa_eff)
            } else {
                0
            };
            rec.dispatch_xy(
                &self.pipe_attn,
                &[
                    &self.q_blk_buf,
                    &self.k_blk_buf,
                    &self.v_blk_buf,
                    &self.k_cache,
                    &self.v_cache,
                    &self.attn_blk_buf,
                ],
                &[
                    (nh as u32) | ((nh_kv as u32) << 16),
                    (hd as u32) | ((bs as u32) << 16),
                    ctx_start as u32,
                    pos0 as u32,
                    layer.kv_vec4_base,
                ],
                nh as u32,
                bs as u32,
            )?;

            target.dispatch_gemm(
                rec,
                &layer.wo,
                &self.attn_blk_buf,
                &self.proj_blk_buf,
                bs,
                false,
            )?;
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.inp_l_buf,
                    &self.proj_blk_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                ],
                &[5, (bs * n_embd) as u32, 0, 0, 0, 0, 0, 0],
                self.elem_groups(bs * n_embd),
            )?;

            rec.dispatch_xy(
                &self.pipe_ops,
                &[
                    &self.inp_l_buf,
                    &self.norm_blk_buf,
                    &self.f32_params,
                    &target.dummy_buf,
                    q8_xq,
                    q8_xd,
                ],
                &[
                    4,
                    n_embd as u32,
                    layer.ffn_norm_off,
                    fuse_flag,
                    0,
                    eps_bits,
                    0,
                    0,
                ],
                1,
                bs as u32,
            )?;
            if fuse_q8 {
                target.dispatch_gemm_prequant(
                    rec,
                    &layer.ffn_gate,
                    &self.norm_blk_buf,
                    &self.xq_buf,
                    &self.xd_buf,
                    &self.ffn_gate_blk_buf,
                    bs,
                    false,
                    false,
                )?;
                target.dispatch_gemm_prequant(
                    rec,
                    &layer.ffn_up,
                    &self.norm_blk_buf,
                    &self.xq_buf,
                    &self.xd_buf,
                    &self.ffn_up_blk_buf,
                    bs,
                    false,
                    true,
                )?;
            } else {
                target.dispatch_gemm_nb(
                    rec,
                    &layer.ffn_gate,
                    &self.norm_blk_buf,
                    &self.ffn_gate_blk_buf,
                    bs,
                    false,
                )?;
                target.dispatch_gemm(
                    rec,
                    &layer.ffn_up,
                    &self.norm_blk_buf,
                    &self.ffn_up_blk_buf,
                    bs,
                    false,
                )?;
            }
            let total_ff = bs * n_ff;
            rec.dispatch(
                &self.pipe_silu_mul,
                &[&self.ffn_gate_blk_buf, &self.ffn_up_blk_buf],
                &[total_ff as u32],
                total_ff
                    .div_ceil(self.pipe_silu_mul.wg_size as usize)
                    .min(65535) as u32,
            )?;
            target.dispatch_gemm(
                rec,
                &layer.ffn_down,
                &self.ffn_gate_blk_buf,
                &self.proj_blk_buf,
                bs,
                false,
            )?;
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.inp_l_buf,
                    &self.proj_blk_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                    &target.dummy_buf,
                ],
                &[5, (bs * n_embd) as u32, 0, 0, 0, 0, 0, 0],
                self.elem_groups(bs * n_embd),
            )?;
        }

        // 3. Final decoder RMSNorm -> `hidden_blk_buf [bs, n_embd]`
        rec.dispatch_xy(
            &self.pipe_ops,
            &[
                &self.inp_l_buf,
                &self.hidden_blk_buf,
                &self.f32_params,
                &target.dummy_buf,
                &target.dummy_buf,
                &target.dummy_buf,
            ],
            &[4, n_embd as u32, self.output_norm_off, 0, 0, eps_bits, 0, 0],
            1,
            bs as u32,
        )?;

        // 4. Chained `k`-step Recurrent Markov Head + `lm_head` + Fused Softcap-Argmax (all inside ONE CB!)
        let ctx_ring_off = (ring_slot * n_embd) as u32;
        let prob_slot_base = (bs + 2) as u32;
        let sg = target.ctx.info().subgroup_size;
        let idx_rows_per_wg = ((self.idx_lm_wg / sg) * self.idx_lm_nr) as usize;

        for step in 0..k {
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.joint_in_buf,
                    &self.rnn_state_buf,
                    &self.markov_w1_bf16,
                    &self.hidden_blk_buf,
                    &self.draft_toks_buf,
                    &self.ctx_feat_ring,
                ],
                &[
                    6,
                    rank as u32,
                    n_embd as u32,
                    step as u32,
                    ctx_ring_off,
                    0,
                    0,
                    0,
                ],
                self.elem_groups(2 * rank + 2 * n_embd),
            )?;
            target.dispatch_gemv(
                rec,
                &self.joint_proj,
                &self.joint_in_buf,
                &self.joint_out_buf,
            )?;
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.joint_out_buf,
                    &self.rnn_state_buf,
                    &self.latent_buf,
                    &self.f32_params,
                    &target.dummy_buf,
                    &target.dummy_buf,
                ],
                &[7, rank as u32, self.joint_proj_b_off, 0, 0, 0, 0, 0],
                self.elem_groups(rank),
            )?;
            target.dispatch_gemv(
                rec,
                &self.hidden_out,
                &self.latent_buf,
                &self.correction_buf,
            )?;
            let corr_groups = if fuse_q8 {
                (n_embd / 32).div_ceil(4) as u32
            } else {
                self.elem_groups(n_embd)
            };
            rec.dispatch(
                &self.pipe_ops,
                &[
                    &self.corr_hidden_buf,
                    &self.hidden_blk_buf,
                    &self.correction_buf,
                    &self.f32_params,
                    q8_xq,
                    q8_xd,
                ],
                &[
                    8,
                    n_embd as u32,
                    step as u32,
                    self.hidden_out_b_off,
                    fuse_flag,
                    0,
                    0,
                    0,
                ],
                corr_groups,
            )?;

            if let Some((ref map_buf, n_pruned)) = self.vocab_map {
                rec.dispatch(
                    &self.pipe_idx_lm,
                    &[
                        &target.output.buf,
                        &self.corr_hidden_buf,
                        &target.dummy_buf,
                        &target.logits_padded_buf,
                        map_buf,
                        &target.dummy_buf,
                    ],
                    &[
                        10,
                        n_embd as u32,
                        n_pruned as u32,
                        target.output.row_u16s,
                        0,
                        0,
                        0,
                        0,
                    ],
                    n_pruned.div_ceil(idx_rows_per_wg) as u32,
                )?;
                rec.dispatch(
                    &self.pipe_argmax_cap,
                    &[
                        &target.logits_padded_buf,
                        &target.dummy_buf,
                        &target.dummy_buf,
                        &target.dummy_buf,
                        &self.draft_toks_buf,
                        map_buf,
                    ],
                    &[
                        9,
                        n_pruned as u32,
                        (step + 1) as u32,
                        1,
                        prob_slot_base,
                        cap_bits,
                        0,
                        0,
                    ],
                    1,
                )?;
            } else {
                if fuse_q8 {
                    target.dispatch_gemv_prequant(
                        rec,
                        &target.output,
                        &self.corr_hidden_buf,
                        &self.xq_buf,
                        &self.xd_buf,
                        &target.logits_padded_buf,
                        true,
                    )?;
                } else {
                    target.dispatch_gemv(
                        rec,
                        &target.output,
                        &self.corr_hidden_buf,
                        &target.logits_padded_buf,
                    )?;
                }
                rec.dispatch(
                    &self.pipe_argmax_cap,
                    &[
                        &target.logits_padded_buf,
                        &target.dummy_buf,
                        &target.dummy_buf,
                        &target.dummy_buf,
                        &self.draft_toks_buf,
                        &target.dummy_buf,
                    ],
                    &[
                        9,
                        self.cfg.vocab_size as u32,
                        (step + 1) as u32,
                        0,
                        prob_slot_base,
                        cap_bits,
                        0,
                        0,
                    ],
                    1,
                )?;
            }
        }
        Ok(())
    }

    /// Drafts `k_drafts` greedy tokens at `pos0` after `anchor_token_id` in 1 GPU submission.
    pub fn draft_block_greedy(
        &mut self,
        target: &mut Inkling2VkRunner,
        pos0: usize,
        anchor_token_id: u32,
        k_drafts: usize,
        out_drafts: &mut [u32],
    ) -> Result<usize, VulkanError> {
        self.inject_and_draft_greedy(target, 0, 0, pos0, anchor_token_id, k_drafts, out_drafts)
    }

    /// Fuses `inject_batch(inject_start_pos, inject_count)` (for the just-committed window) AND
    /// `draft_block_greedy(draft_pos0, anchor_token_id, k_drafts)` (for the next speculative block)
    /// into **ONE single Vulkan command buffer submission**!
    #[allow(clippy::too_many_arguments)]
    pub fn inject_and_draft_greedy(
        &mut self,
        target: &mut Inkling2VkRunner,
        inject_start_pos: usize,
        inject_count: usize,
        draft_pos0: usize,
        anchor_token_id: u32,
        k_drafts: usize,
        out_drafts: &mut [u32],
    ) -> Result<usize, VulkanError> {
        let mut probs = [0.0f32; MAX_DFLASH_VK_BLOCK];
        let cap = k_drafts.min(MAX_DFLASH_VK_BLOCK);
        self.inject_and_draft_greedy_with_probs(
            target,
            inject_start_pos,
            inject_count,
            draft_pos0,
            anchor_token_id,
            k_drafts,
            out_drafts,
            &mut probs[..cap],
        )
    }

    /// Same as [`Self::inject_and_draft_greedy`], additionally returning exact per-step softmax
    /// top-1 confidence probabilities (`p_top1 in (0, 1]`) for confidence-gated early stopping.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_and_draft_greedy_with_probs(
        &mut self,
        target: &mut Inkling2VkRunner,
        inject_start_pos: usize,
        inject_count: usize,
        draft_pos0: usize,
        anchor_token_id: u32,
        k_drafts: usize,
        out_drafts: &mut [u32],
        out_probs: &mut [f32],
    ) -> Result<usize, VulkanError> {
        let bs = self.cfg.block_size;
        let k = k_drafts
            .min(self.cfg.max_draft_tokens())
            .min(out_drafts.len())
            .min(out_probs.len());

        // Check `can_draft` AFTER accounting for `inject_count` updating `ctx_feat_pos`!
        let prev_in_inj = inject_count > 0
            && draft_pos0 > inject_start_pos
            && (draft_pos0 - 1) < inject_start_pos + inject_count;
        let has_prev = draft_pos0 > 0
            && (prev_in_inj
                || self.ctx_feat_pos[(draft_pos0 - 1) % self.ring_cap]
                    == (draft_pos0 - 1) as isize);
        let run_k = if k == 0 || !has_prev || draft_pos0 + bs > self.cfg.max_ctx {
            0
        } else {
            k
        };
        if inject_count == 0 && run_k == 0 {
            return Ok(0);
        }
        if run_k > 0 {
            if (anchor_token_id as usize) >= self.cfg.vocab_size {
                return Err(VulkanError::InvalidDimensions(format!(
                    "anchor_token_id {anchor_token_id} >= vocab_size {}",
                    self.cfg.vocab_size
                )));
            }
            target
                .ctx
                .write_bytes(&target.step_buf, 0, &anchor_token_id.to_le_bytes())?;
            target
                .ctx
                .write_bytes(&self.draft_toks_buf, 0, &anchor_token_id.to_le_bytes())?;
        }

        let mut inner_err = None;
        target.ctx.run(|rec| {
            if inject_count > 0
                && let Err(e) =
                    self.record_inject_batch(rec, target, inject_start_pos, inject_count)
            {
                inner_err = Some(e);
                return Ok(());
            }
            if run_k > 0
                && let Err(e) = self.record_draft_block_greedy(rec, target, draft_pos0, run_k)
            {
                inner_err = Some(e);
            }
            Ok(())
        })?;
        if let Some(e) = inner_err {
            return Err(e);
        }

        if run_k > 0 {
            let ptr = self
                .draft_toks_buf
                .mapped_ptr()
                .ok_or(VulkanError::NoMemoryType("draft_toks_buf mapped"))?;
            let prob_base = bs + 3;
            // SAFETY: `draft_toks_buf` (`(2 * bs + 8) * 4` bytes) is persistently mapped coherent
            // memory, idle after `ctx.run`, with `1 + run_k` and `prob_base + run_k` in bounds.
            unsafe {
                let u32_ptr = ptr as *const u32;
                core::ptr::copy_nonoverlapping(u32_ptr.add(1), out_drafts.as_mut_ptr(), run_k);
                for (i, dst_p) in out_probs.iter_mut().enumerate().take(run_k) {
                    *dst_p = f32::from_bits(*u32_ptr.add(prob_base + i));
                }
            }
        }
        Ok(run_k)
    }
}
