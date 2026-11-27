//! `Qwen3.8-Flash-Next` (`general.architecture == "qwen4exp"`) forward pass (`#![forbid(unsafe_code)]`).
//!
//! Reference implementation matching `Strata` (`qwen4exp.cpp` & `qsa.cuh`, MIT License):
//! - 4-stream Hyper-Connections (`hc_count = 4`, `hc_low_rank = 320`): per-stream RMSNorm,
//!   low-rank SiLU bottleneck (`1 / hc` scaled), Sigmoid stream mix (`mean` across streams),
//!   and `2 * sigmoid(inject / hc)` residual scatter; `output_hc` serves as the final norm.
//! - PLE N-gram hash embedding (`per_layer_token_embd.weight` kept zero-copy in host `mmap`):
//!   `n = 2..=ngram_size` XOR-mul rolling hash with EOS window reset, per-stream signed-sqrt
//!   Sigmoid gate, and dilated causal `conv1d` (`dilation = ngram_size`) + SiLU residual.
//! - Hybrid Sigmoid-gated GDN (`3:1` interval, **Sigmoid** output gate on `z` instead of SiLU!)
//!   interleaved with QSA (`Query-Sparse Attention`: interleaved `[Q, Gate]`, NeoX partial RoPE,
//!   block-4 mean-pooled F16 indexer keys rotated to block-start position, ReLU dot-product top-k).
//! - Fine-grained 512-expert `Q2_0` MoE (`top_k = 10`, Softmax, `norm_weights = true`) +
//!   Sigmoid-gated shared expert (`ffn_gate_inp_shexp.weight`).

use tm_core::{DType, KvType, dequantize_row_bytes, f16_to_f32, f32_to_f16};
use tm_cpu::{
    KernelTable, MAX_CPU_BATCH, MatvecScratch, RopeType, ThreadPool,
    gdn_conv1d_silu_split_parallel, gdn_recurrence_parallel, matmul_f32, matvec_f32, rms_norm,
    rope_inplace, softplus_f32,
};
use tm_gguf::{GgufError, GgufFile};
use tm_lora::{BoundLora, LoraScratch, LoraSet};

use crate::llama::{ModelError, RawMatView};
use crate::moe::{
    MoeScratch, SparseMoeFfn, eval_sparse_moe_ffn, load_layer_moe_ffn, parse_moe_arch_params,
};
use crate::state::{LayerState, SpecVerifyBuf, decode_embedding_row};

/// Hyperparameters for `general.architecture == "qwen4exp"`.
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub hc_count: usize,
    pub hc_low_rank: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub max_n_ff_exp: usize,
    pub max_n_ff_shexp: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub full_attention_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_d_inner: usize,
    pub indexer_n_head: usize,
    pub indexer_key_len: usize,
    pub indexer_top_k: usize,
    pub compress_ratios: Vec<usize>,
    pub ple_layer: Option<usize>,
    pub ple_ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_n_heads: usize,
    pub ple_head_dim: usize,
    pub ple_conv_kernel: usize,
    pub ple_eos_token_id: u32,
    pub ple_image_token_id: u32,
    pub ple_layer_multipliers: Vec<u64>,
    pub ple_head_offsets: Vec<u64>,
    pub ple_head_vocab_sizes: Vec<u64>,
    pub vocab_size: usize,
    pub max_ctx_train: usize,
}

impl Qwen4ExpConfig {
    #[inline]
    pub const fn hc_dim(&self) -> usize {
        self.hc_count * self.n_embd
    }

    #[inline]
    pub const fn conv_dim(&self) -> usize {
        2 * self.ssm_n_group * self.ssm_state_size + self.ssm_d_inner
    }

    #[inline]
    pub const fn ssm_head_v_dim(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank
    }

    #[inline]
    pub const fn is_recurrent(&self, il: usize) -> bool {
        !(il + 1).is_multiple_of(self.full_attention_interval)
    }

    #[inline]
    pub const fn ple_hist_len(&self) -> usize {
        if self.ple_conv_kernel > 1 {
            (self.ple_conv_kernel - 1) * self.ple_ngram_size
        } else {
            0
        }
    }
}

pub struct Qwen4ExpHcWeights<'a> {
    pub norm: Vec<f32>, // [hc_dim]
    pub down: RawMatView<'a>,
    pub up: RawMatView<'a>,
    pub inject: Option<RawMatView<'a>>,
}

pub struct Qwen4ExpPleWeights<'a> {
    pub key: RawMatView<'a>,
    pub value: RawMatView<'a>,
    pub norm_key: Vec<f32>,
    pub norm_query: Vec<f32>,
    pub norm_conv: Vec<f32>,
    /// Row-major `[hc_dim, ple_conv_kernel]` (`kern` contiguous floats per channel).
    pub conv1d: Vec<f32>,
}

pub enum Qwen4ExpAttnWeights<'a> {
    FullQsa {
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        wqg: RawMatView<'a>,
        wk: RawMatView<'a>,
        wv: RawMatView<'a>,
        wo: RawMatView<'a>,
        idx_q_proj: RawMatView<'a>,
        idx_k_proj: RawMatView<'a>,
        idx_q_norm: Vec<f32>,
        idx_k_norm: Vec<f32>,
    },
    GdnSigmoid {
        wqkv: RawMatView<'a>,
        wz: RawMatView<'a>,
        w_alpha: RawMatView<'a>,
        w_beta: RawMatView<'a>,
        conv1d: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_dt_bias: Vec<f32>,
        ssm_norm: Vec<f32>,
        w_out: RawMatView<'a>,
    },
}

pub struct Qwen4ExpLayerWeights<'a> {
    pub hc_attn: Qwen4ExpHcWeights<'a>,
    pub hc_ffn: Qwen4ExpHcWeights<'a>,
    pub ple: Option<Qwen4ExpPleWeights<'a>>,
    pub attn: Qwen4ExpAttnWeights<'a>,
    pub moe: SparseMoeFfn<'a>,
}

pub struct Qwen4ExpModel<'a> {
    pub cfg: Qwen4ExpConfig,
    pub token_embd: RawMatView<'a>,
    pub per_layer_token_embd: Option<RawMatView<'a>>,
    pub hc_head: Qwen4ExpHcWeights<'a>,
    pub output: RawMatView<'a>,
    pub rope_inv_freqs: Vec<f64>,
    pub layers: Vec<Qwen4ExpLayerWeights<'a>>,
    pub lora: BoundLora,
}

pub struct Qwen4ExpState {
    pub max_ctx: usize,
    pub layers: Vec<LayerState>,
    /// Per-QSA-layer raw indexer key cache `[max_ctx, idx_dim]` in IEEE F16 (`u16`).
    pub idx_k_caches: Vec<Vec<u16>>,
    /// Per-QSA-layer mean-pooled + normed + rotated block keys `[max_ctx / pool + 1, idx_dim]` in F16 (`u16`).
    pub idx_pooled_caches: Vec<Vec<u16>>,
    pub qsa_slot_of_layer: Vec<Option<usize>>,
    /// Token history `[max_ctx]` for PLE N-gram hash lookup.
    pub seq_tokens: Vec<u32>,
    /// PLE dilated causal conv history `[ple_hist_len * hc_dim]`.
    pub ple_conv_hist: Vec<f32>,
    pub res_hc: Vec<f32>,
    pub hc_xn: Vec<f32>,
    pub hc_lo: Vec<f32>,
    pub hc_gate: Vec<f32>,
    pub hc_inject: Vec<f32>,
    pub cur: Vec<f32>,
    pub sub_out: Vec<f32>,
    pub ple_emb: Vec<f32>,
    pub ple_key_buf: Vec<f32>,
    pub ple_val_buf: Vec<f32>,
    pub ple_gated: Vec<f32>,
    pub ple_normed: Vec<f32>,
    pub qkv_buf: Vec<f32>,
    pub q_buf: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub z_buf: Vec<f32>,
    pub alpha_buf: Vec<f32>,
    pub beta_buf: Vec<f32>,
    pub idx_q_buf: Vec<f32>,
    pub idx_k_buf: Vec<f32>,
    pub idx_scores: Vec<f32>,
    pub block_allowed: Vec<bool>,
    pub norm_tmp: Vec<f32>,
    pub norm_tmp2: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub mv_scratch: MatvecScratch,
    pub moe_scratch: MoeScratch,
    pub lora_scratch: LoraScratch,
    pub last_hidden: Vec<f32>,
    logits: Vec<f32>,
}

fn load_f32_flex(
    gguf: &GgufFile,
    name: &str,
    layer: &str,
    expected_elems: usize,
) -> Result<Vec<f32>, ModelError> {
    let (info, bytes) = gguf.tensor_bytes(name, layer)?;
    let actual_elems: usize = info.dims.iter().product();
    if actual_elems != expected_elems {
        return Err(GgufError::TensorShapeMismatch {
            path: gguf.path().display().to_string(),
            tensor: name.to_string(),
            layer: layer.to_string(),
            expected: vec![expected_elems],
            actual: info.dims.clone(),
        }
        .into());
    }
    let mut out = vec![0.0f32; expected_elems];
    dequantize_row_bytes(info.dtype, bytes, &mut out);
    Ok(out)
}

fn load_mat<'a>(
    gguf: &'a GgufFile,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
) -> Result<RawMatView<'a>, ModelError> {
    let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
    Ok(RawMatView {
        name: info.name.as_str(),
        bytes,
        dtype: info.dtype,
        rows,
        cols,
        row_offset: 0,
        is_sliced: false,
        lora_slot: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_hc_weights<'a>(
    gguf: &'a GgufFile,
    norm_name: &str,
    down_name: &str,
    up_name: &str,
    inject_name: Option<&str>,
    layer: &str,
    hc: usize,
    hc_dim: usize,
    hc_lr: usize,
) -> Result<Qwen4ExpHcWeights<'a>, ModelError> {
    let norm = load_f32_flex(gguf, norm_name, layer, hc_dim)?;
    let down = load_mat(gguf, down_name, layer, hc_dim, hc_lr)?;
    let up = load_mat(gguf, up_name, layer, hc_lr, hc_dim)?;
    let inject = match inject_name {
        Some(in_nm) => Some(load_mat(gguf, in_nm, layer, hc_dim, hc)?),
        None => None,
    };
    Ok(Qwen4ExpHcWeights {
        norm,
        down,
        up,
        inject,
    })
}

impl<'a> Qwen4ExpModel<'a> {
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "qwen4exp" {
            return Err(ModelError::UnsupportedArch(arch));
        }
        let pfx = "qwen4exp";
        let n_layers = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
        let hc_count = gguf.req_usize(&format!("{pfx}.hyper_connection.count"))?;
        let hc_low_rank = gguf.req_usize(&format!("{pfx}.hyper_connection.low_rank"))?;
        if hc_count <= 1 || hc_low_rank == 0 {
            return Err(ModelError::InvalidConfig(format!(
                "qwen4exp invalid hc_count={hc_count} hc_low_rank={hc_low_rank}"
            )));
        }
        let hc_dim = hc_count * n_embd;

        let n_head = gguf.req_usize(&format!("{pfx}.attention.head_count"))?;
        let n_head_kv = gguf.req_usize(&format!("{pfx}.attention.head_count_kv"))?;
        let head_dim = gguf.req_usize(&format!("{pfx}.attention.key_length"))?;
        let rope_dim = gguf
            .get_meta(&format!("{pfx}.rope.dimension_count"))
            .and_then(|v| v.as_usize())
            .unwrap_or(64);
        let rope_freq_base = gguf
            .get_meta(&format!("{pfx}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .unwrap_or(10_000_000.0);
        let rms_eps = gguf.req_f32(&format!("{pfx}.attention.layer_norm_rms_epsilon"))?;
        let full_attention_interval = gguf
            .get_meta(&format!("{pfx}.full_attention_interval"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4);

        let ssm_conv_kernel = gguf.req_usize(&format!("{pfx}.ssm.conv_kernel"))?;
        let ssm_state_size = gguf.req_usize(&format!("{pfx}.ssm.state_size"))?;
        let ssm_n_group = gguf.req_usize(&format!("{pfx}.ssm.group_count"))?;
        let ssm_dt_rank = gguf.req_usize(&format!("{pfx}.ssm.time_step_rank"))?;
        let ssm_d_inner = gguf.req_usize(&format!("{pfx}.ssm.inner_size"))?;
        if ssm_conv_kernel == 0
            || ssm_state_size == 0
            || ssm_n_group == 0
            || ssm_dt_rank == 0
            || !ssm_dt_rank.is_multiple_of(ssm_n_group)
            || ssm_d_inner != ssm_dt_rank * ssm_state_size
        {
            return Err(ModelError::InvalidConfig(
                "qwen4exp invalid SSM/GDN dimensions".to_string(),
            ));
        }

        let indexer_n_head = gguf.req_usize(&format!("{pfx}.attention.indexer.head_count"))?;
        let indexer_key_len = gguf.req_usize(&format!("{pfx}.attention.indexer.key_length"))?;
        let indexer_top_k = gguf.req_usize(&format!("{pfx}.attention.indexer.top_k"))?;
        let compress_ratios = match gguf.get_meta(&format!("{pfx}.attention.compress_ratios")) {
            Some(mv) => mv
                .as_usize_array()
                .or_else(|| mv.as_usize().map(|v| vec![v; n_layers]))
                .unwrap_or_else(|| vec![4; n_layers]),
            None => vec![4; n_layers],
        };

        let n_ff_default = gguf
            .get_meta(&format!("{pfx}.feed_forward_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(0);
        let moe_params = parse_moe_arch_params(gguf, pfx, &arch, n_layers, n_ff_default)?
            .ok_or_else(|| {
                ModelError::InvalidConfig("qwen4exp requires MoE expert_count > 0".to_string())
            })?;
        let n_ff_shexp = gguf
            .get_meta(&format!("{pfx}.expert_shared_feed_forward_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(moe_params.max_n_ff_exp());

        // Optional PLE N-gram hash embeddings
        let ple_layers_opt = gguf
            .get_meta(&format!("{pfx}.ple.layers"))
            .and_then(|v| v.as_usize_array());
        let (
            ple_layer,
            ple_ngram_size,
            ple_heads_per_ngram,
            ple_n_heads,
            ple_head_dim,
            ple_conv_kernel,
            ple_eos_token_id,
            ple_image_token_id,
            ple_layer_multipliers,
            ple_head_offsets,
            ple_head_vocab_sizes,
        ) = if let Some(pl) = ple_layers_opt
            && !pl.is_empty()
        {
            if pl.len() != 1 || pl[0] >= n_layers {
                return Err(ModelError::InvalidConfig(format!(
                    "qwen4exp invalid ple.layers = {pl:?}"
                )));
            }
            let ng = gguf.req_usize(&format!("{pfx}.ple.ngram_size"))?;
            let hpg = gguf.req_usize(&format!("{pfx}.ple.heads_per_ngram"))?;
            let nh = (ng.saturating_sub(1)) * hpg;
            let hd = gguf.req_usize(&format!("{pfx}.embedding_length_per_layer_input"))?;
            let ck = gguf.req_usize(&format!("{pfx}.ple.conv_kernel"))?;
            let eos = gguf.req_usize(&format!("{pfx}.ple.eos_token_id"))? as u32;
            let img = gguf
                .get_meta(&format!("{pfx}.ple.image_token_id"))
                .and_then(|v| v.as_usize())
                .map_or(eos, |v| v as u32);
            let mults = gguf
                .get_meta(&format!("{pfx}.ple.layer_multipliers"))
                .and_then(|v| v.as_u64_array())
                .ok_or_else(|| GgufError::MissingMetadata {
                    path: gguf.path().display().to_string(),
                    key: format!("{pfx}.ple.layer_multipliers"),
                })?;
            let offs = gguf
                .get_meta(&format!("{pfx}.ple.head_offsets"))
                .and_then(|v| v.as_u64_array())
                .ok_or_else(|| GgufError::MissingMetadata {
                    path: gguf.path().display().to_string(),
                    key: format!("{pfx}.ple.head_offsets"),
                })?;
            let vszs = gguf
                .get_meta(&format!("{pfx}.ple.head_vocab_sizes"))
                .and_then(|v| v.as_u64_array())
                .ok_or_else(|| GgufError::MissingMetadata {
                    path: gguf.path().display().to_string(),
                    key: format!("{pfx}.ple.head_vocab_sizes"),
                })?;
            if mults.len() < ng || offs.len() < nh || vszs.len() < nh || nh * hd != n_embd {
                return Err(ModelError::InvalidConfig(
                    "qwen4exp PLE array lengths or ple_n_heads * ple_head_dim != n_embd"
                        .to_string(),
                ));
            }
            (
                Some(pl[0]),
                ng,
                hpg,
                nh,
                hd,
                ck,
                eos,
                img,
                mults,
                offs,
                vszs,
            )
        } else {
            (
                None,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };

        let emb_info =
            gguf.tensor_info("token_embd.weight")
                .ok_or_else(|| GgufError::MissingTensor {
                    path: gguf.path().display().to_string(),
                    tensor: "token_embd.weight".to_string(),
                    layer: "global".to_string(),
                })?;
        let vocab_size = gguf
            .get_meta(&format!("{pfx}.vocab_size"))
            .and_then(|v| v.as_usize())
            .or_else(|| emb_info.dims.get(1).copied())
            .ok_or_else(|| ModelError::InvalidConfig("qwen4exp missing vocab_size".to_string()))?;

        let cfg = Qwen4ExpConfig {
            arch,
            n_layers,
            n_embd,
            hc_count,
            hc_low_rank,
            n_expert: moe_params.n_expert,
            n_expert_used: moe_params.n_expert_used,
            max_n_ff_exp: moe_params.max_n_ff_exp(),
            max_n_ff_shexp: n_ff_shexp,
            n_head,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_eps,
            full_attention_interval,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            indexer_n_head,
            indexer_key_len,
            indexer_top_k,
            compress_ratios,
            ple_layer,
            ple_ngram_size,
            ple_heads_per_ngram,
            ple_n_heads,
            ple_head_dim,
            ple_conv_kernel,
            ple_eos_token_id,
            ple_image_token_id,
            ple_layer_multipliers,
            ple_head_offsets,
            ple_head_vocab_sizes,
            vocab_size,
            max_ctx_train,
        };

        let token_embd = load_mat(gguf, "token_embd.weight", "global", n_embd, vocab_size)?;
        let output = if gguf.tensor_info("output.weight").is_some() {
            load_mat(gguf, "output.weight", "global", n_embd, vocab_size)?
        } else {
            token_embd
        };
        let hc_head = load_hc_weights(
            gguf,
            "output_hc_norm.weight",
            "output_hc_down.weight",
            "output_hc_up.weight",
            None,
            "global",
            hc_count,
            hc_dim,
            hc_low_rank,
        )?;

        let per_layer_token_embd = if ple_n_heads > 0 {
            let (ti, bytes) = gguf.tensor_bytes("per_layer_token_embd.weight", "global")?;
            if ti.dims.len() != 2 || ti.dims[0] != ple_head_dim {
                return Err(GgufError::TensorShapeMismatch {
                    path: gguf.path().display().to_string(),
                    tensor: "per_layer_token_embd.weight".to_string(),
                    layer: "global".to_string(),
                    expected: vec![ple_head_dim, 0],
                    actual: ti.dims.clone(),
                }
                .into());
            }
            Some(RawMatView {
                name: ti.name.as_str(),
                bytes,
                dtype: ti.dtype,
                rows: ti.dims[1],
                cols: ti.dims[0],
                row_offset: 0,
                is_sliced: false,
                lora_slot: None,
            })
        } else {
            None
        };

        let half_rope = rope_dim / 2;
        let rope_inv_freqs: Vec<f64> = (0..half_rope)
            .map(|i| (rope_freq_base as f64).powf(-2.0 * (i as f64) / (rope_dim as f64)))
            .collect();

        let conv_dim = cfg.conv_dim();
        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let ltag = format!("blk.{il}");
            let hc_attn = load_hc_weights(
                gguf,
                &format!("blk.{il}.hc_attn_norm.weight"),
                &format!("blk.{il}.hc_attn_down.weight"),
                &format!("blk.{il}.hc_attn_up.weight"),
                Some(&format!("blk.{il}.hc_attn_inject.weight")),
                &ltag,
                hc_count,
                hc_dim,
                hc_low_rank,
            )?;
            let hc_ffn = load_hc_weights(
                gguf,
                &format!("blk.{il}.hc_ffn_norm.weight"),
                &format!("blk.{il}.hc_ffn_down.weight"),
                &format!("blk.{il}.hc_ffn_up.weight"),
                Some(&format!("blk.{il}.hc_ffn_inject.weight")),
                &ltag,
                hc_count,
                hc_dim,
                hc_low_rank,
            )?;

            let attn = if cfg.is_recurrent(il) {
                let wqkv = load_mat(
                    gguf,
                    &format!("blk.{il}.attn_qkv.weight"),
                    &ltag,
                    n_embd,
                    conv_dim,
                )?;
                let wz = load_mat(
                    gguf,
                    &format!("blk.{il}.attn_gate.weight"),
                    &ltag,
                    n_embd,
                    ssm_d_inner,
                )?;
                let w_alpha = load_mat(
                    gguf,
                    &format!("blk.{il}.ssm_alpha.weight"),
                    &ltag,
                    n_embd,
                    ssm_dt_rank,
                )?;
                let w_beta = load_mat(
                    gguf,
                    &format!("blk.{il}.ssm_beta.weight"),
                    &ltag,
                    n_embd,
                    ssm_dt_rank,
                )?;
                let (ci, cb) = gguf.req_tensor(
                    &format!("blk.{il}.ssm_conv1d.weight"),
                    &ltag,
                    &[ssm_conv_kernel, conv_dim],
                )?;
                let mut conv1d = vec![0.0f32; conv_dim * ssm_conv_kernel];
                dequantize_row_bytes(ci.dtype, cb, &mut conv1d);
                let ssm_a = load_f32_flex(gguf, &format!("blk.{il}.ssm_a"), &ltag, ssm_dt_rank)?;
                let ssm_dt_bias =
                    load_f32_flex(gguf, &format!("blk.{il}.ssm_dt.bias"), &ltag, ssm_dt_rank)?;
                let ssm_norm = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.ssm_norm.weight"),
                    &ltag,
                    ssm_state_size,
                )?;
                let w_out = load_mat(
                    gguf,
                    &format!("blk.{il}.ssm_out.weight"),
                    &ltag,
                    ssm_d_inner,
                    n_embd,
                )?;
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
                }
            } else {
                let q_norm = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.attn_q_norm.weight"),
                    &ltag,
                    head_dim,
                )?;
                let k_norm = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.attn_k_norm.weight"),
                    &ltag,
                    head_dim,
                )?;
                let wqg = load_mat(
                    gguf,
                    &format!("blk.{il}.attn_q.weight"),
                    &ltag,
                    n_embd,
                    n_head * head_dim * 2,
                )?;
                let wk = load_mat(
                    gguf,
                    &format!("blk.{il}.attn_k.weight"),
                    &ltag,
                    n_embd,
                    n_head_kv * head_dim,
                )?;
                let wv = load_mat(
                    gguf,
                    &format!("blk.{il}.attn_v.weight"),
                    &ltag,
                    n_embd,
                    n_head_kv * head_dim,
                )?;
                let wo = load_mat(
                    gguf,
                    &format!("blk.{il}.attn_output.weight"),
                    &ltag,
                    n_head * head_dim,
                    n_embd,
                )?;
                let idx_q_proj = load_mat(
                    gguf,
                    &format!("blk.{il}.indexer.q_proj.weight"),
                    &ltag,
                    n_embd,
                    indexer_n_head * indexer_key_len,
                )?;
                let idx_k_proj = load_mat(
                    gguf,
                    &format!("blk.{il}.indexer.k_proj.weight"),
                    &ltag,
                    n_embd,
                    indexer_key_len,
                )?;
                let idx_q_norm = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.indexer.q_norm.weight"),
                    &ltag,
                    indexer_key_len,
                )?;
                let idx_k_norm = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.indexer.k_norm.weight"),
                    &ltag,
                    indexer_key_len,
                )?;
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
                }
            };

            let ple = if ple_layer == Some(il) {
                let key = load_mat(
                    gguf,
                    &format!("blk.{il}.ple_key.weight"),
                    &ltag,
                    n_embd,
                    hc_dim,
                )?;
                let value = load_mat(
                    gguf,
                    &format!("blk.{il}.ple_value.weight"),
                    &ltag,
                    n_embd,
                    n_embd,
                )?;
                let norm_key = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.ple_norm_key.weight"),
                    &ltag,
                    hc_dim,
                )?;
                let norm_query = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.ple_norm_query.weight"),
                    &ltag,
                    hc_dim,
                )?;
                let norm_conv = load_f32_flex(
                    gguf,
                    &format!("blk.{il}.ple_norm_conv.weight"),
                    &ltag,
                    hc_dim,
                )?;
                let (pci, pcb) = gguf.req_tensor(
                    &format!("blk.{il}.ple_conv1d.weight"),
                    &ltag,
                    &[ple_conv_kernel, hc_dim],
                )?;
                let mut conv1d = vec![0.0f32; hc_dim * ple_conv_kernel];
                dequantize_row_bytes(pci.dtype, pcb, &mut conv1d);
                Some(Qwen4ExpPleWeights {
                    key,
                    value,
                    norm_key,
                    norm_query,
                    norm_conv,
                    conv1d,
                })
            } else {
                None
            };

            let moe =
                load_layer_moe_ffn(gguf, il, &ltag, n_embd, &moe_params)?.ok_or_else(|| {
                    ModelError::InvalidConfig(format!("qwen4exp blk.{il} missing MoE weights"))
                })?;
            layers.push(Qwen4ExpLayerWeights {
                hc_attn,
                hc_ffn,
                ple,
                attn,
                moe,
            });
        }

        Ok(Self {
            cfg,
            token_embd,
            per_layer_token_embd,
            hc_head,
            output,
            rope_inv_freqs,
            layers,
            lora: BoundLora::default(),
        })
    }

    pub fn bind_lora(&mut self, _loras: &LoraSet) {}

    pub fn num_full_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.attn, Qwen4ExpAttnWeights::FullQsa { .. }))
            .count()
    }

    pub fn num_gdn_layers(&self) -> usize {
        self.cfg.n_layers - self.num_full_attn_layers()
    }

    pub fn embed_token(&self, token_id: u32, out_emb: &mut [f32]) {
        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes,
            self.cfg.n_embd,
            token_id as usize,
            out_emb,
        );
    }

    pub fn output_head_raw(&self) -> (&[f32], f32, DType, &[u8], Option<f32>) {
        (
            &self.hc_head.norm[..self.cfg.n_embd],
            self.cfg.rms_eps,
            self.output.dtype,
            self.output.bytes,
            None,
        )
    }

    pub fn project_logits_from_hidden(
        &self,
        state: &mut Qwen4ExpState,
        pool: &ThreadPool,
        kt: &KernelTable,
        hidden: &[f32],
        out_logits: &mut [f32],
    ) {
        matvec_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes,
            hidden,
            &mut state.mv_scratch,
            out_logits,
            self.cfg.vocab_size,
            self.cfg.n_embd,
        );
    }

    /// Gathers and dequantizes `ple_n_heads` N-gram rows (`[n_tok, n_embd]`) from host `mmap`!
    pub fn gather_ple_embeddings(
        &self,
        seq_tokens: &[u32],
        token_ids: &[u32],
        start_pos: usize,
        ple_out: &mut [f32],
    ) {
        let Some(ref ple_tab) = self.per_layer_token_embd else {
            return;
        };
        let n_tok = token_ids.len();
        let ng = self.cfg.ple_ngram_size;
        let per_g = self.cfg.ple_heads_per_ngram;
        let nh = self.cfg.ple_n_heads;
        let hd = self.cfg.ple_head_dim;
        let eos = self.cfg.ple_eos_token_id;
        let rb = ple_tab.dtype.row_bytes(hd);
        let mut ctx = [0u64; 8];

        for (i, &tok) in token_ids.iter().enumerate().take(n_tok) {
            let pos = start_pos + i;
            ctx[0] = tok as u64;
            let mut cut = false;
            for s in 1..ng {
                let prev = if cut || pos < s {
                    None
                } else if pos - s >= start_pos {
                    Some(token_ids[pos - s - start_pos])
                } else {
                    Some(seq_tokens[pos - s])
                };
                match prev {
                    None => {
                        cut = true;
                        ctx[s] = eos as u64;
                    }
                    Some(pt) => {
                        if pt == eos {
                            cut = true;
                        }
                        ctx[s] = if cut { eos as u64 } else { pt as u64 };
                    }
                }
            }

            let dst_tok = &mut ple_out[i * self.cfg.n_embd..(i + 1) * self.cfg.n_embd];
            for n in 2..=ng {
                let mut mixed = ctx[0].wrapping_mul(self.cfg.ple_layer_multipliers[0]);
                for (j, &cj) in ctx.iter().enumerate().take(n).skip(1) {
                    mixed ^= cj.wrapping_mul(self.cfg.ple_layer_multipliers[j]);
                }
                let base = (n - 2) * per_g;
                for g in 0..per_g {
                    let h_i = base + g;
                    let row = (mixed % self.cfg.ple_head_vocab_sizes[h_i]
                        + self.cfg.ple_head_offsets[h_i]) as usize;
                    let src = &ple_tab.bytes[row * rb..(row + 1) * rb];
                    dequantize_row_bytes(
                        ple_tab.dtype,
                        src,
                        &mut dst_tok[h_i * hd..(h_i + 1) * hd],
                    );
                }
            }
            debug_assert_eq!(nh * hd, self.cfg.n_embd);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_hc_mix_single(
        &self,
        hc_w: &Qwen4ExpHcWeights<'_>,
        res_hc: &[f32],
        xn: &mut [f32],
        lo: &mut [f32],
        gate: &mut [f32],
        mixed: &mut [f32],
        inject_out: Option<&mut [f32]>,
        mv_scratch: &mut MatvecScratch,
        pool: &ThreadPool,
        kt: &KernelTable,
    ) {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * d;
        let hc_lr = self.cfg.hc_low_rank;
        let inv_hc = 1.0 / (hc as f32);

        for c in 0..hc {
            rms_norm(
                &mut xn[c * d..(c + 1) * d],
                &res_hc[c * d..(c + 1) * d],
                &hc_w.norm[c * d..(c + 1) * d],
                self.cfg.rms_eps,
            );
        }
        matvec_f32(
            pool,
            kt,
            hc_w.down.dtype,
            hc_w.down.bytes,
            &xn[..hc_dim],
            mv_scratch,
            &mut lo[..hc_lr],
            hc_lr,
            hc_dim,
        );
        for v in &mut lo[..hc_lr] {
            let x = *v * inv_hc;
            *v = x / (1.0 + (-x).exp());
        }
        matvec_f32(
            pool,
            kt,
            hc_w.up.dtype,
            hc_w.up.bytes,
            &lo[..hc_lr],
            mv_scratch,
            &mut gate[..hc_dim],
            hc_dim,
            hc_lr,
        );
        mixed[..d].fill(0.0);
        for c in 0..hc {
            for j in 0..d {
                let g = gate[c * d + j];
                let sig = 1.0 / (1.0 + (-g).exp());
                mixed[j] += xn[c * d + j] * sig;
            }
        }
        for v in &mut mixed[..d] {
            *v *= inv_hc;
        }
        if let (Some(w_inj), Some(inj_dst)) = (&hc_w.inject, inject_out) {
            matvec_f32(
                pool,
                kt,
                w_inj.dtype,
                w_inj.bytes,
                &xn[..hc_dim],
                mv_scratch,
                &mut inj_dst[..hc],
                hc,
                hc_dim,
            );
        }
    }

    fn apply_hc_combine_single(&self, res_hc: &mut [f32], sub_out: &[f32], inject: &[f32]) {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let inv_hc = 1.0 / (hc as f32);
        for c in 0..hc {
            let w = 2.0 / (1.0 + (-inject[c] * inv_hc).exp());
            let dst = &mut res_hc[c * d..(c + 1) * d];
            for j in 0..d {
                dst[j] += sub_out[j] * w;
            }
        }
    }

    fn apply_ple_single(
        &self,
        ple: &Qwen4ExpPleWeights<'_>,
        state: &mut Qwen4ExpState,
        pool: &ThreadPool,
        kt: &KernelTable,
    ) {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        let hc_dim = hc * d;
        let eps = self.cfg.rms_eps;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        matvec_f32(
            pool,
            kt,
            ple.key.dtype,
            ple.key.bytes,
            &state.ple_emb[..d],
            &mut state.mv_scratch,
            &mut state.ple_key_buf[..hc_dim],
            hc_dim,
            d,
        );
        matvec_f32(
            pool,
            kt,
            ple.value.dtype,
            ple.value.bytes,
            &state.ple_emb[..d],
            &mut state.mv_scratch,
            &mut state.ple_val_buf[..d],
            d,
            d,
        );

        for c in 0..hc {
            let off = c * d;
            rms_norm(
                &mut state.hc_xn[off..off + d],
                &state.ple_key_buf[off..off + d],
                &ple.norm_key[off..off + d],
                eps,
            );
            rms_norm(
                &mut state.hc_gate[off..off + d],
                &state.res_hc[off..off + d],
                &ple.norm_query[off..off + d],
                eps,
            );
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += state.hc_xn[off + j] * state.hc_gate[off + j];
            }
            let s = dot * inv_sqrt_d;
            let mag = s.abs().max(1e-6).sqrt();
            let signed_mag = if s > 0.0 {
                mag
            } else if s < 0.0 {
                -mag
            } else {
                0.0
            };
            let gate = 1.0 / (1.0 + (-signed_mag).exp());
            for j in 0..d {
                state.ple_gated[off + j] = state.ple_val_buf[j] * gate;
            }
            rms_norm(
                &mut state.ple_normed[off..off + d],
                &state.ple_gated[off..off + d],
                &ple.norm_conv[off..off + d],
                eps,
            );
        }

        let kern = self.cfg.ple_conv_kernel;
        let dil = self.cfg.ple_ngram_size;
        let hist_len = self.cfg.ple_hist_len();
        for ch in 0..hc_dim {
            let w_ch = &ple.conv1d[ch * kern..(ch + 1) * kern];
            let mut acc = state.ple_normed[ch] * w_ch[kern - 1];
            for (k, &wk) in w_ch[..kern - 1].iter().enumerate() {
                let lag = (kern - 1 - k) * dil;
                acc += state.ple_conv_hist[(hist_len - lag) * hc_dim + ch] * wk;
            }
            for h in 0..hist_len.saturating_sub(1) {
                state.ple_conv_hist[h * hc_dim + ch] = state.ple_conv_hist[(h + 1) * hc_dim + ch];
            }
            if hist_len > 0 {
                state.ple_conv_hist[(hist_len - 1) * hc_dim + ch] = state.ple_normed[ch];
            }
            let conv_act = acc / (1.0 + (-acc).exp());
            state.res_hc[ch] += state.ple_gated[ch] + conv_act;
        }
    }

    pub fn forward_token(
        &self,
        state: &mut Qwen4ExpState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }
        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes,
            self.cfg.n_embd,
            token_id as usize,
            &mut state.cur,
        );
        state.seq_tokens[pos] = token_id;
        if self.cfg.ple_layer.is_some() {
            self.gather_ple_embeddings(&state.seq_tokens, &[token_id], pos, &mut state.ple_emb);
        }
        self.run_single_from_cur(state, pool, kt, pos, compute_logits)
    }

    pub fn forward_embed(
        &self,
        state: &mut Qwen4ExpState,
        pool: &ThreadPool,
        kt: &KernelTable,
        embed: &[f32],
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        state.cur[..self.cfg.n_embd].copy_from_slice(&embed[..self.cfg.n_embd]);
        let stand_in = self.cfg.ple_image_token_id;
        state.seq_tokens[pos] = stand_in;
        if self.cfg.ple_layer.is_some() {
            self.gather_ple_embeddings(&state.seq_tokens, &[stand_in], pos, &mut state.ple_emb);
        }
        self.run_single_from_cur(state, pool, kt, pos, compute_logits)
    }

    fn run_single_from_cur(
        &self,
        state: &mut Qwen4ExpState,
        pool: &ThreadPool,
        kt: &KernelTable,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        let d = self.cfg.n_embd;
        let hc = self.cfg.hc_count;
        for c in 0..hc {
            state.res_hc[c * d..(c + 1) * d].copy_from_slice(&state.cur[..d]);
        }

        let hk = self.cfg.ssm_n_group;
        let hv = self.cfg.ssm_dt_rank;
        let s_dim = self.cfg.ssm_state_size;
        let d_inner = self.cfg.ssm_d_inner;
        let conv_dim = self.cfg.conv_dim();

        for (il, layer) in self.layers.iter().enumerate() {
            if let Some(ref ple) = layer.ple {
                self.apply_ple_single(ple, state, pool, kt);
            }

            self.apply_hc_mix_single(
                &layer.hc_attn,
                &state.res_hc,
                &mut state.hc_xn,
                &mut state.hc_lo,
                &mut state.hc_gate,
                &mut state.cur,
                Some(&mut state.hc_inject),
                &mut state.mv_scratch,
                pool,
                kt,
            );

            match &layer.attn {
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
                    matvec_f32(
                        pool,
                        kt,
                        wqkv.dtype,
                        wqkv.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.qkv_buf[..conv_dim],
                        conv_dim,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        wz.dtype,
                        wz.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.z_buf[..d_inner],
                        d_inner,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        w_alpha.dtype,
                        w_alpha.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.alpha_buf[..hv],
                        hv,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        w_beta.dtype,
                        w_beta.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.beta_buf[..hv],
                        hv,
                        d,
                    );

                    if let LayerState::DeltaNet {
                        conv_hist,
                        state: ssm_state,
                    } = &mut state.layers[il]
                    {
                        for h in 0..hv {
                            let sp = softplus_f32(state.alpha_buf[h] + ssm_dt_bias[h]);
                            state.alpha_buf[h] = (ssm_a[h] * sp).exp();
                            state.beta_buf[h] = 1.0 / (1.0 + (-state.beta_buf[h]).exp());
                        }
                        gdn_conv1d_silu_split_parallel(
                            pool,
                            kt,
                            &state.qkv_buf[..conv_dim],
                            conv_hist,
                            conv1d,
                            &mut state.q_buf[..hk * s_dim],
                            &mut state.k_buf[..hk * s_dim],
                            &mut state.v_buf[..d_inner],
                            self.cfg.ssm_conv_kernel,
                            hk,
                            s_dim,
                            d_inner,
                            self.cfg.rms_eps,
                            1,
                        );
                        gdn_recurrence_parallel(
                            pool,
                            &state.q_buf[..hk * s_dim],
                            &state.k_buf[..hk * s_dim],
                            &state.v_buf[..d_inner],
                            &state.alpha_buf[..hv],
                            &state.beta_buf[..hv],
                            ssm_state,
                            &mut state.qkv_buf[..d_inner],
                            hk,
                            hv,
                            s_dim,
                            1,
                        );
                        // Sigmoid output gate (`qwen4exp.cpp:667`)!
                        for h in 0..hv {
                            let y = &state.qkv_buf[h * s_dim..(h + 1) * s_dim];
                            let z = &state.z_buf[h * s_dim..(h + 1) * s_dim];
                            let dst = &mut state.v_buf[h * s_dim..(h + 1) * s_dim];
                            rms_norm(dst, y, ssm_norm, self.cfg.rms_eps);
                            for j in 0..s_dim {
                                dst[j] *= 1.0 / (1.0 + (-z[j]).exp());
                            }
                        }
                        matvec_f32(
                            pool,
                            kt,
                            w_out.dtype,
                            w_out.bytes,
                            &state.v_buf[..d_inner],
                            &mut state.mv_scratch,
                            &mut state.sub_out[..d],
                            d,
                            d_inner,
                        );
                    }
                }
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
                    let nh = self.cfg.n_head;
                    let nh_kv = self.cfg.n_head_kv;
                    let hd = self.cfg.head_dim;
                    let q_dim = nh * hd;
                    let kv_dim = nh_kv * hd;
                    let idx_nh = self.cfg.indexer_n_head;
                    let idx_d = self.cfg.indexer_key_len;
                    let pool_r = self.cfg.compress_ratios.get(il).copied().unwrap_or(4);
                    let qsa_slot = state.qsa_slot_of_layer[il].expect("QSA slot");

                    matvec_f32(
                        pool,
                        kt,
                        wqg.dtype,
                        wqg.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.qkv_buf[..2 * q_dim],
                        2 * q_dim,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        wk.dtype,
                        wk.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.k_buf[..kv_dim],
                        kv_dim,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        wv.dtype,
                        wv.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.v_buf[..kv_dim],
                        kv_dim,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        idx_q_proj.dtype,
                        idx_q_proj.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.idx_q_buf[..idx_nh * idx_d],
                        idx_nh * idx_d,
                        d,
                    );
                    matvec_f32(
                        pool,
                        kt,
                        idx_k_proj.dtype,
                        idx_k_proj.bytes,
                        &state.cur[..d],
                        &mut state.mv_scratch,
                        &mut state.idx_k_buf[..idx_d],
                        idx_d,
                        d,
                    );

                    for h in 0..nh {
                        let q_src = &state.qkv_buf[h * 2 * hd..h * 2 * hd + hd];
                        let g_src = &state.qkv_buf[h * 2 * hd + hd..(h + 1) * 2 * hd];
                        rms_norm(
                            &mut state.q_buf[h * hd..(h + 1) * hd],
                            q_src,
                            q_norm,
                            self.cfg.rms_eps,
                        );
                        state.z_buf[h * hd..(h + 1) * hd].copy_from_slice(g_src);
                    }
                    for h in 0..nh_kv {
                        state.norm_tmp[..hd].copy_from_slice(&state.k_buf[h * hd..(h + 1) * hd]);
                        rms_norm(
                            &mut state.k_buf[h * hd..(h + 1) * hd],
                            &state.norm_tmp[..hd],
                            k_norm,
                            self.cfg.rms_eps,
                        );
                    }
                    rope_inplace(
                        &mut state.q_buf[..q_dim],
                        nh,
                        hd,
                        self.cfg.rope_dim,
                        pos,
                        self.cfg.rope_freq_base,
                        None,
                        RopeType::Neox,
                    );
                    rope_inplace(
                        &mut state.k_buf[..kv_dim],
                        nh_kv,
                        hd,
                        self.cfg.rope_dim,
                        pos,
                        self.cfg.rope_freq_base,
                        None,
                        RopeType::Neox,
                    );
                    state.layers[il].store_kv_at(
                        kt,
                        pos,
                        &state.k_buf[..kv_dim],
                        &state.v_buf[..kv_dim],
                    );

                    // Indexer Q norm + NeoX RoPE, raw K F16 store + block-`pool_r` mean pool
                    for h in 0..idx_nh {
                        state.norm_tmp[..idx_d]
                            .copy_from_slice(&state.idx_q_buf[h * idx_d..(h + 1) * idx_d]);
                        rms_norm(
                            &mut state.idx_q_buf[h * idx_d..(h + 1) * idx_d],
                            &state.norm_tmp[..idx_d],
                            idx_q_norm,
                            self.cfg.rms_eps,
                        );
                    }
                    rope_inplace(
                        &mut state.idx_q_buf[..idx_nh * idx_d],
                        idx_nh,
                        idx_d,
                        self.cfg.rope_dim,
                        pos,
                        self.cfg.rope_freq_base,
                        None,
                        RopeType::Neox,
                    );
                    for j in 0..idx_d {
                        state.idx_k_caches[qsa_slot][pos * idx_d + j] =
                            f32_to_f16(state.idx_k_buf[j]);
                    }
                    if pool_r > 0 && (pos + 1).is_multiple_of(pool_r) {
                        let p0 = pos + 1 - pool_r;
                        state.norm_tmp[..idx_d].fill(0.0);
                        for m in 0..pool_r {
                            let row = &state.idx_k_caches[qsa_slot]
                                [(p0 + m) * idx_d..(p0 + m + 1) * idx_d];
                            for (dst, &rv) in state.norm_tmp[..idx_d].iter_mut().zip(row.iter()) {
                                *dst += f16_to_f32(rv);
                            }
                        }
                        let inv_r = 1.0 / (pool_r as f32);
                        for v in &mut state.norm_tmp[..idx_d] {
                            *v *= inv_r;
                        }
                        rms_norm(
                            &mut state.norm_tmp2[..idx_d],
                            &state.norm_tmp[..idx_d],
                            idx_k_norm,
                            self.cfg.rms_eps,
                        );
                        rope_inplace(
                            &mut state.norm_tmp2[..idx_d],
                            1,
                            idx_d,
                            self.cfg.rope_dim,
                            p0,
                            self.cfg.rope_freq_base,
                            None,
                            RopeType::Neox,
                        );
                        let blk_j = (pos + 1) / pool_r - 1;
                        for j in 0..idx_d {
                            state.idx_pooled_caches[qsa_slot][blk_j * idx_d + j] =
                                f32_to_f16(state.norm_tmp2[j]);
                        }
                    }

                    // QSA block selection (`n_vis_blocks <= top_blocks` -> all positions allowed!)
                    let n_vis_blocks = (pos + 1).checked_div(pool_r).unwrap_or(0);
                    let top_blocks = self
                        .cfg
                        .indexer_top_k
                        .checked_div(pool_r)
                        .unwrap_or(usize::MAX);
                    let use_sparse = pool_r > 0 && n_vis_blocks > top_blocks;
                    if use_sparse {
                        state.block_allowed[..n_vis_blocks].fill(false);
                        let idx_scale = 1.0 / (idx_d as f32).sqrt();
                        for b in 0..n_vis_blocks {
                            let pk = &state.idx_pooled_caches[qsa_slot][b * idx_d..(b + 1) * idx_d];
                            let mut sc = 0.0f32;
                            for h in 0..idx_nh {
                                let iq = &state.idx_q_buf[h * idx_d..(h + 1) * idx_d];
                                let dot = (kt.dot_f32_f16)(iq, pk);
                                sc += dot.max(0.0) * idx_scale;
                            }
                            state.idx_scores[b] = sc;
                        }
                        for _ in 0..top_blocks {
                            let mut best_b = 0usize;
                            let mut best_v = f32::NEG_INFINITY;
                            for (b, &sv) in state.idx_scores[..n_vis_blocks].iter().enumerate() {
                                if !state.block_allowed[b] && sv > best_v {
                                    best_v = sv;
                                    best_b = b;
                                }
                            }
                            state.block_allowed[best_b] = true;
                        }
                    }

                    let group = nh / nh_kv;
                    let scale = 1.0 / (hd as f32).sqrt();
                    match &state.layers[il] {
                        LayerState::Kv {
                            kv_type: KvType::F16,
                            k_cache,
                            v_cache,
                            ..
                        } => {
                            for h in 0..nh {
                                let kvh = h / group;
                                let qh = &state.q_buf[h * hd..(h + 1) * hd];
                                let mut mx = f32::NEG_INFINITY;
                                for t in 0..=pos {
                                    if use_sparse {
                                        let blk = t / pool_r;
                                        if blk < n_vis_blocks && !state.block_allowed[blk] {
                                            state.attn_scores[t] = f32::NEG_INFINITY;
                                            continue;
                                        }
                                    }
                                    let ko = t * kv_dim + kvh * hd;
                                    let s = (kt.dot_f32_f16)(qh, &k_cache[ko..ko + hd]) * scale;
                                    state.attn_scores[t] = s;
                                    if s > mx {
                                        mx = s;
                                    }
                                }
                                let mut den = 0.0f32;
                                for s in &mut state.attn_scores[..=pos] {
                                    if *s == f32::NEG_INFINITY {
                                        *s = 0.0;
                                    } else {
                                        *s = (*s - mx).exp();
                                        den += *s;
                                    }
                                }
                                let inv_den = 1.0 / den;
                                let oh = &mut state.qkv_buf[h * hd..(h + 1) * hd];
                                oh.fill(0.0);
                                for t in 0..=pos {
                                    let p = state.attn_scores[t] * inv_den;
                                    if p > 0.0 {
                                        let vo = t * kv_dim + kvh * hd;
                                        (kt.axpy_f32_f16)(oh, p, &v_cache[vo..vo + hd]);
                                    }
                                }
                            }
                        }
                        LayerState::Kv {
                            kv_type: KvType::Q8_0,
                            k_q8,
                            v_q8,
                            ..
                        } => {
                            let kv_blks = kv_dim / 32;
                            let hd_blks = hd / 32;
                            for h in 0..nh {
                                let kvh = h / group;
                                let qh = &state.q_buf[h * hd..(h + 1) * hd];
                                let mut mx = f32::NEG_INFINITY;
                                for t in 0..=pos {
                                    if use_sparse {
                                        let blk = t / pool_r;
                                        if blk < n_vis_blocks && !state.block_allowed[blk] {
                                            state.attn_scores[t] = f32::NEG_INFINITY;
                                            continue;
                                        }
                                    }
                                    let ko = t * kv_blks + kvh * hd_blks;
                                    let s = (kt.dot_f32_q8_0)(qh, &k_q8[ko..ko + hd_blks]) * scale;
                                    state.attn_scores[t] = s;
                                    if s > mx {
                                        mx = s;
                                    }
                                }
                                let mut den = 0.0f32;
                                for s in &mut state.attn_scores[..=pos] {
                                    if *s == f32::NEG_INFINITY {
                                        *s = 0.0;
                                    } else {
                                        *s = (*s - mx).exp();
                                        den += *s;
                                    }
                                }
                                let inv_den = 1.0 / den;
                                let oh = &mut state.qkv_buf[h * hd..(h + 1) * hd];
                                oh.fill(0.0);
                                for t in 0..=pos {
                                    let p = state.attn_scores[t] * inv_den;
                                    if p > 0.0 {
                                        let vo = t * kv_blks + kvh * hd_blks;
                                        (kt.axpy_f32_q8_0)(oh, p, &v_q8[vo..vo + hd_blks]);
                                    }
                                }
                            }
                        }
                        _ => unreachable!(),
                    }

                    for i in 0..q_dim {
                        state.qkv_buf[i] *= 1.0 / (1.0 + (-state.z_buf[i]).exp());
                    }
                    matvec_f32(
                        pool,
                        kt,
                        wo.dtype,
                        wo.bytes,
                        &state.qkv_buf[..q_dim],
                        &mut state.mv_scratch,
                        &mut state.sub_out[..d],
                        d,
                        q_dim,
                    );
                }
            }

            self.apply_hc_combine_single(&mut state.res_hc, &state.sub_out, &state.hc_inject);

            // MoE FFN + gated shared expert
            self.apply_hc_mix_single(
                &layer.hc_ffn,
                &state.res_hc,
                &mut state.hc_xn,
                &mut state.hc_lo,
                &mut state.hc_gate,
                &mut state.cur,
                Some(&mut state.hc_inject),
                &mut state.mv_scratch,
                pool,
                kt,
            );
            eval_sparse_moe_ffn(
                &layer.moe,
                &self.lora,
                pool,
                kt,
                &state.cur[..d],
                &mut state.sub_out[..d],
                d,
                1,
                &mut state.mv_scratch,
                &mut state.lora_scratch,
                &mut state.moe_scratch,
            );
            self.apply_hc_combine_single(&mut state.res_hc, &state.sub_out, &state.hc_inject);
        }

        // Final output HC mix + `output.weight`
        self.apply_hc_mix_single(
            &self.hc_head,
            &state.res_hc,
            &mut state.hc_xn,
            &mut state.hc_lo,
            &mut state.hc_gate,
            &mut state.last_hidden,
            None,
            &mut state.mv_scratch,
            pool,
            kt,
        );
        if compute_logits {
            matmul_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes,
                &state.last_hidden[..d],
                &mut state.mv_scratch,
                &mut state.logits,
                self.cfg.vocab_size,
                d,
                1,
            );
        }
        Ok(())
    }

    pub fn forward_batch(
        &self,
        state: &mut Qwen4ExpState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        if token_ids.is_empty() {
            return Ok(());
        }
        if start_pos + token_ids.len() > state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: start_pos + token_ids.len() - 1,
                max_ctx: state.max_ctx,
            });
        }
        for (i, &tid) in token_ids.iter().enumerate() {
            let is_last = i + 1 == token_ids.len();
            self.forward_token(
                state,
                pool,
                kt,
                tid,
                start_pos + i,
                compute_last_logits && is_last,
            )?;
        }
        Ok(())
    }

    pub fn verify_speculative(
        &self,
        state: &mut Qwen4ExpState,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        window_tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), ModelError> {
        spec_buf.last_window_len = window_tokens.len();
        for (i, &tid) in window_tokens.iter().enumerate() {
            self.forward_token(state, pool, kt, tid, start_pos + i, true)?;
            let v = self.cfg.vocab_size;
            spec_buf.logits_win[i * v..(i + 1) * v].copy_from_slice(&state.logits);
            let d = self.cfg.n_embd;
            spec_buf.hidden_win[i * d..(i + 1) * d].copy_from_slice(&state.last_hidden);
        }
        Ok(())
    }

    pub fn commit_speculative(
        &self,
        _state: &mut Qwen4ExpState,
        _spec_buf: &mut SpecVerifyBuf,
        _pool: &ThreadPool,
        _kt: &KernelTable,
        _committed: usize,
    ) {
    }
}

impl Qwen4ExpState {
    pub fn new(model: &Qwen4ExpModel<'_>, max_ctx: usize) -> Self {
        Self::new_with_kv_type(model, max_ctx, KvType::F16)
    }

    pub fn new_with_kv_type(model: &Qwen4ExpModel<'_>, max_ctx: usize, kv_type: KvType) -> Self {
        let cfg = &model.cfg;
        let d = cfg.n_embd;
        let hc_dim = cfg.hc_dim();
        let conv_dim = cfg.conv_dim();
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let idx_d = cfg.indexer_key_len;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut idx_k_caches = Vec::new();
        let mut idx_pooled_caches = Vec::new();
        let mut qsa_slot_of_layer = vec![None; cfg.n_layers];

        for (il, lw) in model.layers.iter().enumerate() {
            match lw.attn {
                Qwen4ExpAttnWeights::GdnSigmoid { .. } => {
                    layers.push(LayerState::DeltaNet {
                        state: vec![
                            0.0;
                            cfg.ssm_dt_rank * cfg.ssm_state_size * cfg.ssm_head_v_dim()
                        ],
                        conv_hist: vec![0.0; cfg.ssm_conv_kernel.saturating_sub(1) * conv_dim],
                    });
                }
                Qwen4ExpAttnWeights::FullQsa { .. } => {
                    layers.push(LayerState::new_kv(max_ctx, kv_dim, None, kv_type));
                    let slot = idx_k_caches.len();
                    qsa_slot_of_layer[il] = Some(slot);
                    let r = cfg.compress_ratios.get(il).copied().unwrap_or(4).max(1);
                    idx_k_caches.push(vec![0u16; max_ctx * idx_d]);
                    idx_pooled_caches.push(vec![0u16; (max_ctx / r + 2) * idx_d]);
                }
            }
        }

        let max_cols = [
            d,
            hc_dim,
            cfg.hc_low_rank,
            conv_dim,
            cfg.ssm_d_inner,
            cfg.n_head * cfg.head_dim * 2,
            cfg.max_n_ff_exp,
            cfg.max_n_ff_shexp,
        ]
        .into_iter()
        .max()
        .unwrap_or(d);

        Self {
            max_ctx,
            layers,
            idx_k_caches,
            idx_pooled_caches,
            qsa_slot_of_layer,
            seq_tokens: vec![cfg.ple_eos_token_id; max_ctx.max(1)],
            ple_conv_hist: vec![0.0; cfg.ple_hist_len() * hc_dim],
            res_hc: vec![0.0; hc_dim],
            hc_xn: vec![0.0; hc_dim],
            hc_lo: vec![0.0; cfg.hc_low_rank],
            hc_gate: vec![0.0; hc_dim],
            hc_inject: vec![0.0; cfg.hc_count],
            cur: vec![0.0; d],
            sub_out: vec![0.0; d],
            ple_emb: vec![0.0; d],
            ple_key_buf: vec![0.0; hc_dim],
            ple_val_buf: vec![0.0; d],
            ple_gated: vec![0.0; hc_dim],
            ple_normed: vec![0.0; hc_dim],
            qkv_buf: vec![0.0; conv_dim.max(2 * cfg.n_head * cfg.head_dim)],
            q_buf: vec![0.0; (cfg.ssm_n_group * cfg.ssm_state_size).max(cfg.n_head * cfg.head_dim)],
            k_buf: vec![0.0; (cfg.ssm_n_group * cfg.ssm_state_size).max(kv_dim)],
            v_buf: vec![0.0; cfg.ssm_d_inner.max(kv_dim)],
            z_buf: vec![0.0; cfg.ssm_d_inner.max(cfg.n_head * cfg.head_dim)],
            alpha_buf: vec![0.0; cfg.ssm_dt_rank],
            beta_buf: vec![0.0; cfg.ssm_dt_rank],
            idx_q_buf: vec![0.0; cfg.indexer_n_head * idx_d],
            idx_k_buf: vec![0.0; idx_d],
            idx_scores: vec![0.0; max_ctx + 2],
            block_allowed: vec![false; max_ctx + 2],
            norm_tmp: vec![0.0; cfg.head_dim.max(idx_d).max(1)],
            norm_tmp2: vec![0.0; cfg.head_dim.max(idx_d).max(1)],
            attn_scores: vec![0.0; max_ctx.max(MAX_CPU_BATCH)],
            mv_scratch: MatvecScratch::new(max_cols),
            moe_scratch: MoeScratch::new(
                d,
                cfg.n_expert,
                cfg.n_expert_used,
                cfg.max_n_ff_exp,
                cfg.max_n_ff_shexp,
            ),
            lora_scratch: LoraScratch::new(0),
            last_hidden: vec![0.0; d],
            logits: vec![0.0; cfg.vocab_size],
        }
    }

    pub fn reset(&mut self) {
        for ls in &mut self.layers {
            ls.reset();
        }
        for kc in &mut self.idx_k_caches {
            kc.fill(0);
        }
        for pc in &mut self.idx_pooled_caches {
            pc.fill(0);
        }
        self.ple_conv_hist.fill(0.0);
    }

    #[inline]
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}
