//! Unified per-layer inference state (`docs/ARCHITECTURE.md` §5.2).
//!
//! Covers standard/sliding-window KV caches (`F16` and `Q8_0` quantized `F-14`),
//! KV-shared reference layers (Gemma 4), Gated DeltaNet recurrent + conv history states (Qwen3.5),
//! and ShortConv states (LFM2).

use tm_core::{
    BlockQ8_0, DType, KvType, QK8_0, dequantize_q4_0x4_single_row, dequantize_row_bytes,
};
use tm_cpu::{
    KernelTable, ThreadPool, gqa_attention_f16_kv_batch_parallel,
    gqa_attention_f16_kv_ext_batch_parallel, gqa_attention_f16_kv_ext_parallel,
    gqa_attention_f16_kv_parallel, gqa_attention_q8_0_kv_ext_batch_parallel,
    gqa_attention_q8_0_kv_ext_parallel,
};

/// Maximum speculative verification window (`1` committed token + up to `8` draft tokens).
pub const MAX_SPEC_WINDOW: usize = 9;

/// Decodes a single embedding row `row_idx` from `tensor_bytes` into `dst` (supports `Q4_0x4` and all standard `DType`s).
#[inline]
pub fn decode_embedding_row(
    dtype: DType,
    tensor_bytes: &[u8],
    cols: usize,
    row_idx: usize,
    dst: &mut [f32],
) {
    if dtype == DType::Q4_0x4 {
        dequantize_q4_0x4_single_row(tensor_bytes, cols, row_idx, dst);
    } else {
        let rb = dtype.row_bytes(cols);
        let row = &tensor_bytes[row_idx * rb..(row_idx + 1) * rb];
        dequantize_row_bytes(dtype, row, dst);
    }
}

/// Pre-allocated scratch buffers for multi-token speculative verification & replay commit (`ADR-005`).
///
/// Allocated ONCE when speculative decoding is enabled; zero heap allocations occur on the token path,
/// and normal single-token sessions pay zero extra memory when speculation is disabled.
pub struct SpecVerifyBuf {
    pub logits_win: Vec<f32>,
    pub hidden_win: Vec<f32>,
    pub norm_win: Vec<f32>,
    pub gdn_qkv_win: Vec<f32>,
    pub gdn_g_win: Vec<f32>,
    pub gdn_beta_win: Vec<f32>,
    pub sc_bx_win: Vec<f32>,
    pub tmp_ssm_state: Vec<f32>,
    pub tmp_conv_hist: Vec<f32>,
    pub replay_q: Vec<f32>,
    pub replay_k: Vec<f32>,
    pub replay_v: Vec<f32>,
    pub replay_out: Vec<f32>,
    pub last_window_len: usize,
}

impl SpecVerifyBuf {
    pub fn new_kv_only(vocab_size: usize, n_embd: usize) -> Self {
        Self {
            logits_win: vec![0.0; MAX_SPEC_WINDOW * vocab_size],
            hidden_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            norm_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            gdn_qkv_win: Vec::new(),
            gdn_g_win: Vec::new(),
            gdn_beta_win: Vec::new(),
            sc_bx_win: Vec::new(),
            tmp_ssm_state: Vec::new(),
            tmp_conv_hist: Vec::new(),
            replay_q: Vec::new(),
            replay_k: Vec::new(),
            replay_v: Vec::new(),
            replay_out: Vec::new(),
            last_window_len: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_qwen35(
        vocab_size: usize,
        n_embd: usize,
        n_gdn_layers: usize,
        conv_dim: usize,
        d_conv: usize,
        hk: usize,
        hv: usize,
        s_dim: usize,
    ) -> Self {
        let hist_len = d_conv.saturating_sub(1);
        let k_dim = hk * s_dim;
        let v_dim = hv * s_dim;
        Self {
            logits_win: vec![0.0; MAX_SPEC_WINDOW * vocab_size],
            hidden_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            norm_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            gdn_qkv_win: vec![0.0; n_gdn_layers * MAX_SPEC_WINDOW * conv_dim],
            gdn_g_win: vec![0.0; n_gdn_layers * MAX_SPEC_WINDOW * hv],
            gdn_beta_win: vec![0.0; n_gdn_layers * MAX_SPEC_WINDOW * hv],
            sc_bx_win: Vec::new(),
            tmp_ssm_state: vec![0.0; hv * s_dim * s_dim],
            tmp_conv_hist: vec![0.0; conv_dim * hist_len],
            replay_q: vec![0.0; MAX_SPEC_WINDOW * k_dim],
            replay_k: vec![0.0; MAX_SPEC_WINDOW * k_dim],
            replay_v: vec![0.0; MAX_SPEC_WINDOW * v_dim],
            replay_out: vec![0.0; MAX_SPEC_WINDOW * v_dim],
            last_window_len: 0,
        }
    }

    pub fn new_lfm2(vocab_size: usize, n_embd: usize, n_sc_layers: usize, l_cache: usize) -> Self {
        let hist_len = l_cache.saturating_sub(1);
        Self {
            logits_win: vec![0.0; MAX_SPEC_WINDOW * vocab_size],
            hidden_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            norm_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            gdn_qkv_win: Vec::new(),
            gdn_g_win: Vec::new(),
            gdn_beta_win: Vec::new(),
            sc_bx_win: vec![0.0; n_sc_layers * MAX_SPEC_WINDOW * n_embd],
            tmp_ssm_state: Vec::new(),
            tmp_conv_hist: vec![0.0; n_embd * hist_len],
            replay_q: Vec::new(),
            replay_k: Vec::new(),
            replay_v: Vec::new(),
            replay_out: Vec::new(),
            last_window_len: 0,
        }
    }

    pub fn new_inkling2(vocab_size: usize, n_embd: usize, total_conv_floats: usize) -> Self {
        Self {
            logits_win: vec![0.0; MAX_SPEC_WINDOW * vocab_size],
            hidden_win: vec![0.0; MAX_SPEC_WINDOW * n_embd],
            norm_win: Vec::new(),
            gdn_qkv_win: Vec::new(),
            gdn_g_win: Vec::new(),
            gdn_beta_win: Vec::new(),
            sc_bx_win: vec![0.0; MAX_SPEC_WINDOW * total_conv_floats],
            tmp_ssm_state: Vec::new(),
            tmp_conv_hist: vec![0.0; total_conv_floats],
            replay_q: Vec::new(),
            replay_k: Vec::new(),
            replay_v: Vec::new(),
            replay_out: Vec::new(),
            last_window_len: 0,
        }
    }

    #[inline]
    pub fn logits_at(&self, step: usize, vocab_size: usize) -> &[f32] {
        &self.logits_win[step * vocab_size..(step + 1) * vocab_size]
    }

    #[inline]
    pub fn hidden_at(&self, step: usize, n_embd: usize) -> &[f32] {
        &self.hidden_win[step * n_embd..(step + 1) * n_embd]
    }
}

/// Per-layer heterogeneous inference state (`docs/ARCHITECTURE.md` §5.2).
#[derive(Clone, Debug, PartialEq)]
pub enum LayerState {
    /// Full or sliding-window GQA attention KV cache (`[max_ctx, kv_dim]` in `F16` or `BlockQ8_0`).
    Kv {
        kv_type: KvType,
        k_cache: Vec<u16>,
        v_cache: Vec<u16>,
        k_q8: Vec<BlockQ8_0>,
        v_q8: Vec<BlockQ8_0>,
        kv_dim: usize,
        window: Option<usize>,
    },
    /// KV-shared layer (e.g. Gemma 4 upper layers) reading an earlier layer's `LayerState::Kv`.
    SharedKv {
        source_layer: usize,
        window: Option<usize>,
    },
    /// Gated DeltaNet recurrent state (`[n_v_heads, S, S]` transposed) + causal Conv1d history.
    DeltaNet {
        state: Vec<f32>,
        conv_hist: Vec<f32>,
    },
    /// LFM2 gated short convolution history (`[n_embd, l_cache - 1]`).
    ShortConv { conv_hist: Vec<f32> },
}

impl LayerState {
    /// Allocates a new `LayerState::Kv` with the requested `KvType` (`F16` or `Q8_0`).
    /// Only the active precision's backing buffers are allocated (zero wasted memory).
    pub fn new_kv(max_ctx: usize, kv_dim: usize, window: Option<usize>, kv_type: KvType) -> Self {
        match kv_type {
            KvType::F16 => LayerState::Kv {
                kv_type: KvType::F16,
                k_cache: vec![0u16; max_ctx * kv_dim],
                v_cache: vec![0u16; max_ctx * kv_dim],
                k_q8: Vec::new(),
                v_q8: Vec::new(),
                kv_dim,
                window,
            },
            KvType::Q8_0 => {
                assert_eq!(kv_dim % QK8_0, 0, "Q8_0 KV cache requires kv_dim % 32 == 0");
                let n_blk = max_ctx * (kv_dim / QK8_0);
                LayerState::Kv {
                    kv_type: KvType::Q8_0,
                    k_cache: Vec::new(),
                    v_cache: Vec::new(),
                    k_q8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n_blk],
                    v_q8: vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n_blk],
                    kv_dim,
                    window,
                }
            }
        }
    }

    /// Zeroes out the layer state in-place without reallocating.
    pub fn reset(&mut self) {
        match self {
            LayerState::Kv {
                k_cache,
                v_cache,
                k_q8,
                v_q8,
                ..
            } => {
                k_cache.fill(0);
                v_cache.fill(0);
                k_q8.fill(BlockQ8_0 { d: 0, qs: [0; 32] });
                v_q8.fill(BlockQ8_0 { d: 0, qs: [0; 32] });
            }
            LayerState::SharedKv { .. } => {}
            LayerState::DeltaNet { state, conv_hist } => {
                state.fill(0.0);
                conv_hist.fill(0.0);
            }
            LayerState::ShortConv { conv_hist } => {
                conv_hist.fill(0.0);
            }
        }
    }

    /// Returns true if this layer contains non-truncatable recurrent/convolutional state
    /// (requiring exact checkpoint restoration in `PrefixCache` and read-only verify + replay commit in speculation).
    #[inline]
    pub fn is_recurrent(&self) -> bool {
        matches!(
            self,
            LayerState::DeltaNet { .. } | LayerState::ShortConv { .. }
        )
    }

    /// Writes `k_vec` and `v_vec` (`[kv_dim]` floats) into this `LayerState::Kv` at sequence position `pos`.
    #[inline]
    pub fn store_kv_at(&mut self, kt: &KernelTable, pos: usize, k_vec: &[f32], v_vec: &[f32]) {
        if let LayerState::Kv {
            kv_type,
            k_cache,
            v_cache,
            k_q8,
            v_q8,
            kv_dim,
            ..
        } = self
        {
            match kv_type {
                KvType::F16 => {
                    let off = pos * *kv_dim;
                    (kt.f32_to_f16_slice)(k_vec, &mut k_cache[off..off + *kv_dim]);
                    (kt.f32_to_f16_slice)(v_vec, &mut v_cache[off..off + *kv_dim]);
                }
                KvType::Q8_0 => {
                    let blks = *kv_dim / QK8_0;
                    let off = pos * blks;
                    (kt.quantize_q8_0)(k_vec, &mut k_q8[off..off + blks]);
                    (kt.quantize_q8_0)(v_vec, &mut v_q8[off..off + blks]);
                }
            }
        }
    }

    /// Single-token GQA attention reading from this `LayerState::Kv` (`F16` or `Q8_0`).
    #[allow(clippy::too_many_arguments)]
    pub fn attend_single(
        &self,
        pool: &ThreadPool,
        kt: &KernelTable,
        out: &mut [f32],
        q: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        pos: usize,
        attn_scale: f32,
        window_override: Option<usize>,
        max_ctx: usize,
        scores: &mut [f32],
    ) {
        match self {
            LayerState::Kv {
                kv_type: KvType::F16,
                k_cache,
                v_cache,
                window,
                ..
            } => {
                let win = window_override.or(*window);
                if win.is_none() && (attn_scale - 1.0 / (head_dim as f32).sqrt()).abs() < 1e-7 {
                    gqa_attention_f16_kv_parallel(
                        pool,
                        kt,
                        out,
                        q,
                        k_cache,
                        v_cache,
                        n_head,
                        n_head_kv,
                        head_dim,
                        pos + 1,
                        max_ctx,
                        scores,
                    );
                } else {
                    gqa_attention_f16_kv_ext_parallel(
                        pool, kt, out, q, k_cache, v_cache, n_head, n_head_kv, head_dim, pos,
                        attn_scale, win, max_ctx, scores,
                    );
                }
            }
            LayerState::Kv {
                kv_type: KvType::Q8_0,
                k_q8,
                v_q8,
                window,
                ..
            } => {
                let win = window_override.or(*window);
                gqa_attention_q8_0_kv_ext_parallel(
                    pool, kt, out, q, k_q8, v_q8, n_head, n_head_kv, head_dim, pos, attn_scale,
                    win, max_ctx, scores,
                );
            }
            _ => unreachable!("attend_single requires LayerState::Kv"),
        }
    }

    /// Multi-token causal GQA attention reading from this `LayerState::Kv` (`F16` or `Q8_0`).
    #[allow(clippy::too_many_arguments)]
    pub fn attend_batch(
        &self,
        pool: &ThreadPool,
        kt: &KernelTable,
        out: &mut [f32],
        q: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        start_pos: usize,
        n_tok: usize,
        attn_scale: f32,
        window_override: Option<usize>,
        max_ctx: usize,
        scores: &mut [f32],
    ) {
        match self {
            LayerState::Kv {
                kv_type: KvType::F16,
                k_cache,
                v_cache,
                window,
                ..
            } => {
                let win = window_override.or(*window);
                if win.is_none() && (attn_scale - 1.0 / (head_dim as f32).sqrt()).abs() < 1e-7 {
                    gqa_attention_f16_kv_batch_parallel(
                        pool, kt, out, q, k_cache, v_cache, n_head, n_head_kv, head_dim, start_pos,
                        n_tok, max_ctx, scores,
                    );
                } else {
                    gqa_attention_f16_kv_ext_batch_parallel(
                        pool, kt, out, q, k_cache, v_cache, n_head, n_head_kv, head_dim, start_pos,
                        n_tok, attn_scale, win, max_ctx, scores,
                    );
                }
            }
            LayerState::Kv {
                kv_type: KvType::Q8_0,
                k_q8,
                v_q8,
                window,
                ..
            } => {
                let win = window_override.or(*window);
                gqa_attention_q8_0_kv_ext_batch_parallel(
                    pool, kt, out, q, k_q8, v_q8, n_head, n_head_kv, head_dim, start_pos, n_tok,
                    attn_scale, win, max_ctx, scores,
                );
            }
            _ => unreachable!("attend_batch requires LayerState::Kv"),
        }
    }
}
