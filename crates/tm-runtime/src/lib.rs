#![forbid(unsafe_code)]

//! Unified CPU inference `Session`, polymorphic model/state dispatch across all supported
//! architectures (`llama`, `qwen3`, `qwen35`, `gemma4`, `lfm2`), and LRU `PrefixCache` (`F-13`).

pub mod prefix_cache;

use tm_cpu::{KernelTable, ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{
    Gemma4Model, Gemma4State, LayerState, Lfm2Model, Lfm2State, LlamaModel, LlamaState, ModelError,
    Qwen35Model, Qwen35State,
};

pub use prefix_cache::{CachedLayer, PrefixCache};

/// Unified CPU model wrapper supporting all `tinymodel` architectures.
pub enum CpuModel<'a> {
    Llama(LlamaModel<'a>),
    Qwen35(Qwen35Model<'a>),
    Gemma4(Gemma4Model<'a>),
    Lfm2(Lfm2Model<'a>),
}

/// Unified pre-allocated inference state matching `CpuModel`.
pub enum CpuState {
    Llama(LlamaState),
    Qwen35(Qwen35State),
    Gemma4(Gemma4State),
    Lfm2(Lfm2State),
}

impl<'a> CpuModel<'a> {
    /// Loads and validates any supported GGUF model (`llama`, `qwen3`, `qwen35`, `gemma4`, `lfm2`).
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?;
        match arch {
            "llama" | "qwen3" => Ok(CpuModel::Llama(LlamaModel::from_gguf(gguf)?)),
            "qwen35" => Ok(CpuModel::Qwen35(Qwen35Model::from_gguf(gguf)?)),
            "gemma4" => Ok(CpuModel::Gemma4(Gemma4Model::from_gguf(gguf)?)),
            "lfm2" => Ok(CpuModel::Lfm2(Lfm2Model::from_gguf(gguf)?)),
            other => Err(ModelError::UnsupportedArch(other.to_string())),
        }
    }

    pub fn arch(&self) -> &str {
        match self {
            CpuModel::Llama(m) => &m.cfg.arch,
            CpuModel::Qwen35(m) => &m.cfg.arch,
            CpuModel::Gemma4(m) => &m.cfg.arch,
            CpuModel::Lfm2(m) => &m.cfg.arch,
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            CpuModel::Llama(m) => m.cfg.vocab_size,
            CpuModel::Qwen35(m) => m.cfg.vocab_size,
            CpuModel::Gemma4(m) => m.cfg.vocab_size,
            CpuModel::Lfm2(m) => m.cfg.vocab_size,
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            CpuModel::Llama(m) => m.cfg.n_layers,
            CpuModel::Qwen35(m) => m.cfg.n_layers,
            CpuModel::Gemma4(m) => m.cfg.n_layers,
            CpuModel::Lfm2(m) => m.cfg.n_layers,
        }
    }

    /// Estimates anonymous dirty RAM required for `CpuState` at `max_ctx` (for Linux/sbc safety checks).
    pub fn estimate_dirty_bytes(&self, max_ctx: usize) -> u64 {
        match self {
            CpuModel::Llama(m) => {
                let c = &m.cfg;
                let kv = (2 * c.n_layers * max_ctx * c.n_head_kv * c.head_dim * 2) as u64;
                let sc = ((c.vocab_size + 36 * (c.n_embd + c.n_ff)) * 4) as u64;
                kv + sc
            }
            CpuModel::Qwen35(m) => {
                let c = &m.cfg;
                let kv =
                    (2 * m.num_full_attn_layers() * max_ctx * c.n_head_kv * c.head_dim * 2) as u64;
                let gdn = (m.num_gdn_layers()
                    * (c.ssm_dt_rank * c.ssm_state_size * c.ssm_head_v_dim()
                        + c.conv_dim() * c.ssm_conv_kernel)
                    * 4) as u64;
                let sc = ((c.vocab_size + 40 * (c.n_embd + c.n_ff + c.conv_dim())) * 4) as u64;
                kv + gdn + sc
            }
            CpuModel::Gemma4(m) => {
                let c = &m.cfg;
                let mut kv = 0u64;
                for il in 0..c.n_layer_kv_from_start {
                    kv += (2 * max_ctx * c.n_head_kv * c.head_dim(il) * 2) as u64;
                }
                let sc = ((c.vocab_size + 40 * (c.n_embd + c.max_n_ff())) * 4) as u64;
                kv + sc
            }
            CpuModel::Lfm2(m) => {
                let c = &m.cfg;
                let mut kv = 0u64;
                let mut conv = 0u64;
                for il in 0..c.n_layers {
                    if c.is_shortconv(il) {
                        conv += (c.n_embd * c.shortconv_l_cache * 4) as u64;
                    } else {
                        kv += (2 * max_ctx * c.n_head_kv_per_layer[il] * c.head_dim * 2) as u64;
                    }
                }
                let sc = ((c.vocab_size + 36 * (c.n_embd + c.n_ff)) * 4) as u64;
                kv + conv + sc
            }
        }
    }

    pub fn new_state(&self, max_ctx: usize) -> CpuState {
        match self {
            CpuModel::Llama(m) => CpuState::Llama(LlamaState::new(&m.cfg, max_ctx)),
            CpuModel::Qwen35(m) => CpuState::Qwen35(Qwen35State::new(m, max_ctx)),
            CpuModel::Gemma4(m) => CpuState::Gemma4(Gemma4State::new(m, max_ctx)),
            CpuModel::Lfm2(m) => CpuState::Lfm2(Lfm2State::new(m, max_ctx)),
        }
    }

    pub fn forward_token(
        &self,
        state: &mut CpuState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        match (self, state) {
            (CpuModel::Llama(m), CpuState::Llama(s)) => {
                m.forward_token(s, pool, kt, token_id, pos, compute_logits)
            }
            (CpuModel::Qwen35(m), CpuState::Qwen35(s)) => {
                m.forward_token(s, pool, kt, token_id, pos, compute_logits)
            }
            (CpuModel::Gemma4(m), CpuState::Gemma4(s)) => {
                m.forward_token(s, pool, kt, token_id, pos, compute_logits)
            }
            (CpuModel::Lfm2(m), CpuState::Lfm2(s)) => {
                m.forward_token(s, pool, kt, token_id, pos, compute_logits)
            }
            _ => unreachable!("CpuModel and CpuState variants always match"),
        }
    }

    pub fn forward_batch(
        &self,
        state: &mut CpuState,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        match (self, state) {
            (CpuModel::Llama(m), CpuState::Llama(s)) => {
                m.forward_batch(s, pool, kt, token_ids, start_pos, compute_last_logits)
            }
            (CpuModel::Qwen35(m), CpuState::Qwen35(s)) => {
                m.forward_batch(s, pool, kt, token_ids, start_pos, compute_last_logits)
            }
            (CpuModel::Gemma4(m), CpuState::Gemma4(s)) => {
                m.forward_batch(s, pool, kt, token_ids, start_pos, compute_last_logits)
            }
            (CpuModel::Lfm2(m), CpuState::Lfm2(s)) => {
                m.forward_batch(s, pool, kt, token_ids, start_pos, compute_last_logits)
            }
            _ => unreachable!("CpuModel and CpuState variants always match"),
        }
    }
}

impl CpuState {
    pub fn max_ctx(&self) -> usize {
        match self {
            CpuState::Llama(s) => s.max_ctx,
            CpuState::Qwen35(s) => s.max_ctx,
            CpuState::Gemma4(s) => s.max_ctx,
            CpuState::Lfm2(s) => s.max_ctx,
        }
    }

    pub fn reset(&mut self) {
        match self {
            CpuState::Llama(s) => s.reset(),
            CpuState::Qwen35(s) => s.reset(),
            CpuState::Gemma4(s) => s.reset(),
            CpuState::Lfm2(s) => s.reset(),
        }
    }

    pub fn logits(&self) -> &[f32] {
        match self {
            CpuState::Llama(s) => s.logits(),
            CpuState::Qwen35(s) => s.logits(),
            CpuState::Gemma4(s) => s.logits(),
            CpuState::Lfm2(s) => s.logits(),
        }
    }

    pub fn layers_slice(&self) -> Option<&[LayerState]> {
        match self {
            CpuState::Llama(_) => None,
            CpuState::Qwen35(s) => Some(&s.layers),
            CpuState::Gemma4(s) => Some(&s.layers),
            CpuState::Lfm2(s) => Some(&s.layers),
        }
    }

    pub fn layers_mut(&mut self) -> Option<&mut [LayerState]> {
        match self {
            CpuState::Llama(_) => None,
            CpuState::Qwen35(s) => Some(&mut s.layers),
            CpuState::Gemma4(s) => Some(&mut s.layers),
            CpuState::Lfm2(s) => Some(&mut s.layers),
        }
    }
}

/// Stateful inference session combining `CpuModel`, `CpuState`, `ThreadPool`, and `PrefixCache`.
pub struct Session<'a> {
    pub model: &'a CpuModel<'a>,
    pub state: CpuState,
    pub cache: PrefixCache,
    pos: usize,
}

/// Statistics returned by `Session::prefill_with_cache`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillCacheReport {
    pub total_tokens: usize,
    pub cached_tokens: usize,
    pub evaluated_tokens: usize,
}

impl<'a> Session<'a> {
    pub fn new(model: &'a CpuModel<'a>, max_ctx: usize, prefix_cache_bytes: usize) -> Self {
        Self {
            model,
            state: model.new_state(max_ctx),
            cache: PrefixCache::new(prefix_cache_bytes),
            pos: 0,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn logits(&self) -> &[f32] {
        self.state.logits()
    }

    /// Saves an explicit checkpoint for `prefix_tokens` (must match current session state at `prefix_tokens.len()`).
    pub fn save_checkpoint(&mut self, prefix_tokens: &[u32]) {
        match &self.state {
            CpuState::Llama(s) => self.cache.save_llama(prefix_tokens, s),
            other => {
                if let Some(layers) = other.layers_slice() {
                    self.cache
                        .save_layers(prefix_tokens, other.max_ctx(), layers);
                }
            }
        }
    }

    /// Evaluates `prompt_tokens` from position 0, reusing the longest cached prefix/checkpoint
    /// (`<= prompt_tokens.len() - 1`), saving a checkpoint at `prompt_tokens.len() - 1` (and optional
    /// `checkpoint_every` intervals), and computing exact final logits at `prompt_tokens.len() - 1`.
    ///
    /// Guarantees **100% bit-for-bit identical logits** to uncached `forward_batch` across
    /// both KV/SWA (`Llama`, `Qwen3`, `Gemma 4`) and recurrent (`Qwen3.5`, `LFM2`) models!
    pub fn prefill_with_cache(
        &mut self,
        pool: &ThreadPool,
        prompt_tokens: &[u32],
        checkpoint_every: Option<usize>,
    ) -> Result<PrefillCacheReport, ModelError> {
        let kt = active_kernels();
        let total = prompt_tokens.len();
        if total == 0 {
            return Ok(PrefillCacheReport {
                total_tokens: 0,
                cached_tokens: 0,
                evaluated_tokens: 0,
            });
        }
        if total > self.state.max_ctx() {
            return Err(ModelError::ContextOverflow {
                pos: total - 1,
                max_ctx: self.state.max_ctx(),
            });
        }
        for &tid in prompt_tokens {
            if (tid as usize) >= self.model.vocab_size() {
                return Err(ModelError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.model.vocab_size(),
                });
            }
        }

        // Restore up to `total - 1` so at least 1 final token runs to populate `state.logits`!
        let max_restore = total.saturating_sub(1);
        self.state.reset();
        let matched = match &mut self.state {
            CpuState::Llama(s) => self.cache.restore_llama(prompt_tokens, max_restore, s),
            other => {
                let mctx = other.max_ctx();
                if let Some(layers) = other.layers_mut() {
                    self.cache
                        .restore_layers(prompt_tokens, max_restore, mctx, layers)
                } else {
                    0
                }
            }
        };

        if total == 1 {
            self.model
                .forward_token(&mut self.state, pool, kt, prompt_tokens[0], 0, true)?;
            self.pos = 1;
            return Ok(PrefillCacheReport {
                total_tokens: 1,
                cached_tokens: 0,
                evaluated_tokens: 1,
            });
        }

        let pre_end = total - 1;
        if matched < pre_end {
            if let Some(step_k) = checkpoint_every.filter(|&k| k > 0) {
                let mut cur = matched;
                while cur < pre_end {
                    let next_boundary = ((cur / step_k) + 1) * step_k;
                    let seg_end = next_boundary.min(pre_end);
                    self.model.forward_batch(
                        &mut self.state,
                        pool,
                        kt,
                        &prompt_tokens[cur..seg_end],
                        cur,
                        false,
                    )?;
                    cur = seg_end;
                    if cur < pre_end && cur.is_multiple_of(step_k) {
                        self.save_checkpoint(&prompt_tokens[..cur]);
                    }
                }
            } else {
                self.model.forward_batch(
                    &mut self.state,
                    pool,
                    kt,
                    &prompt_tokens[matched..pre_end],
                    matched,
                    false,
                )?;
            }
            self.save_checkpoint(&prompt_tokens[..pre_end]);
        }

        // Final token step produces exact `logits` at `total - 1` and advances state to `total`
        self.model.forward_token(
            &mut self.state,
            pool,
            kt,
            prompt_tokens[pre_end],
            pre_end,
            true,
        )?;
        self.pos = total;

        Ok(PrefillCacheReport {
            total_tokens: total,
            cached_tokens: matched,
            evaluated_tokens: total - matched,
        })
    }

    /// Advances the session by 1 generated token at `self.pos`.
    pub fn decode_step(
        &mut self,
        pool: &ThreadPool,
        token_id: u32,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        let kt = active_kernels();
        let cur_pos = self.pos;
        self.model
            .forward_token(&mut self.state, pool, kt, token_id, cur_pos, compute_logits)?;
        self.pos = cur_pos + 1;
        Ok(())
    }
}
