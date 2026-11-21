//! Unified per-layer inference state (`docs/ARCHITECTURE.md` §5.2).
//!
//! Covers standard/sliding-window KV caches, KV-shared reference layers (Gemma 4),
//! Gated DeltaNet recurrent + conv history states (Qwen3.5), and ShortConv states (LFM2).

/// Per-layer heterogeneous inference state (`docs/ARCHITECTURE.md` §5.2).
#[derive(Clone, Debug, PartialEq)]
pub enum LayerState {
    /// Full or sliding-window GQA attention KV cache (`[max_ctx, kv_dim]` in IEEE-754 F16).
    Kv {
        k_cache: Vec<u16>,
        v_cache: Vec<u16>,
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
    /// Zeroes out the layer state in-place without reallocating.
    pub fn reset(&mut self) {
        match self {
            LayerState::Kv {
                k_cache, v_cache, ..
            } => {
                k_cache.fill(0);
                v_cache.fill(0);
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
    /// (requiring exact checkpoint restoration in `PrefixCache`).
    #[inline]
    pub fn is_recurrent(&self) -> bool {
        matches!(
            self,
            LayerState::DeltaNet { .. } | LayerState::ShortConv { .. }
        )
    }
}
