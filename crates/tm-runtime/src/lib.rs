#![forbid(unsafe_code)]

//! Unified CPU inference `Session`, polymorphic model/state dispatch across all supported
//! architectures (`llama`, `qwen3`, `qwen35`, `gemma4`, `lfm2`), `Q8_0` KV cache (`F-14`),
//! speculative decoding (`F-12`, `ADR-005`), and LRU `PrefixCache` (`F-13`).

pub mod prefix_cache;
pub mod spec;

use std::time::Instant;

pub use tm_core::KvType;
use tm_cpu::{KernelTable, ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{
    Gemma4Model, Gemma4State, LayerState, Lfm2Model, Lfm2State, LlamaModel, LlamaState,
    MAX_SPEC_WINDOW, ModelError, Qwen35Model, Qwen35State, SpecVerifyBuf, argmax,
};

pub use prefix_cache::{CachedLayer, PrefixCache};
pub use spec::{
    DraftChoice, LinearMtpHead, MAX_DRAFT_TOKENS, MIN_NET_GAIN_THRESHOLD, MtpDrafter,
    SpecController, SpecMode, SpecStats, SuffixDrafter,
};

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

    pub fn n_embd(&self) -> usize {
        match self {
            CpuModel::Llama(m) => m.cfg.n_embd,
            CpuModel::Qwen35(m) => m.cfg.n_embd,
            CpuModel::Gemma4(m) => m.cfg.n_embd,
            CpuModel::Lfm2(m) => m.cfg.n_embd,
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

    /// Estimates anonymous dirty RAM required for `CpuState` at `max_ctx` with `F16` KV cache.
    pub fn estimate_dirty_bytes(&self, max_ctx: usize) -> u64 {
        self.estimate_dirty_bytes_with_kv(max_ctx, KvType::F16)
    }

    /// Estimates anonymous dirty RAM required for `CpuState` at `max_ctx` and `kv_type` (`F16` or `Q8_0`).
    pub fn estimate_dirty_bytes_with_kv(&self, max_ctx: usize, kv_type: KvType) -> u64 {
        let kv_bytes_for = |n_layers: usize, kv_dim: usize| -> u64 {
            match kv_type {
                KvType::F16 => (2 * n_layers * max_ctx * kv_dim * 2) as u64,
                KvType::Q8_0 => (2 * n_layers * max_ctx * (kv_dim / 32) * 34) as u64,
            }
        };
        match self {
            CpuModel::Llama(m) => {
                let c = &m.cfg;
                let kv = kv_bytes_for(c.n_layers, c.n_head_kv * c.head_dim);
                let sc = ((c.vocab_size + 36 * (c.n_embd + c.n_ff)) * 4) as u64;
                kv + sc
            }
            CpuModel::Qwen35(m) => {
                let c = &m.cfg;
                let kv = kv_bytes_for(m.num_full_attn_layers(), c.n_head_kv * c.head_dim);
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
                    kv += kv_bytes_for(1, c.n_head_kv * c.head_dim(il));
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
                        kv += kv_bytes_for(1, c.n_head_kv_per_layer[il] * c.head_dim);
                    }
                }
                let sc = ((c.vocab_size + 36 * (c.n_embd + c.n_ff)) * 4) as u64;
                kv + conv + sc
            }
        }
    }

    pub fn new_state(&self, max_ctx: usize) -> CpuState {
        self.new_state_with_kv(max_ctx, KvType::F16)
    }

    pub fn new_state_with_kv(&self, max_ctx: usize, kv_type: KvType) -> CpuState {
        match self {
            CpuModel::Llama(m) => {
                CpuState::Llama(LlamaState::new_with_kv_type(&m.cfg, max_ctx, kv_type))
            }
            CpuModel::Qwen35(m) => {
                CpuState::Qwen35(Qwen35State::new_with_kv_type(m, max_ctx, kv_type))
            }
            CpuModel::Gemma4(m) => {
                CpuState::Gemma4(Gemma4State::new_with_kv_type(m, max_ctx, kv_type))
            }
            CpuModel::Lfm2(m) => CpuState::Lfm2(Lfm2State::new_with_kv_type(m, max_ctx, kv_type)),
        }
    }

    /// Allocates architecture-matched `SpecVerifyBuf` for speculative verification & replay commit.
    pub fn new_spec_buf(&self) -> SpecVerifyBuf {
        match self {
            CpuModel::Llama(m) => SpecVerifyBuf::new_kv_only(m.cfg.vocab_size, m.cfg.n_embd),
            CpuModel::Gemma4(m) => SpecVerifyBuf::new_kv_only(m.cfg.vocab_size, m.cfg.n_embd),
            CpuModel::Qwen35(m) => SpecVerifyBuf::new_qwen35(
                m.cfg.vocab_size,
                m.cfg.n_embd,
                m.num_gdn_layers(),
                m.cfg.conv_dim(),
                m.cfg.ssm_conv_kernel,
                m.cfg.ssm_n_group,
                m.cfg.ssm_dt_rank,
                m.cfg.ssm_state_size,
            ),
            CpuModel::Lfm2(m) => SpecVerifyBuf::new_lfm2(
                m.cfg.vocab_size,
                m.cfg.n_embd,
                m.num_shortconv_layers(),
                m.cfg.shortconv_l_cache,
            ),
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

    pub fn verify_speculative(
        &self,
        state: &mut CpuState,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        window_tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), ModelError> {
        match (self, state) {
            (CpuModel::Llama(m), CpuState::Llama(s)) => {
                m.verify_speculative(s, spec_buf, pool, kt, window_tokens, start_pos)
            }
            (CpuModel::Qwen35(m), CpuState::Qwen35(s)) => {
                m.verify_speculative(s, spec_buf, pool, kt, window_tokens, start_pos)
            }
            (CpuModel::Gemma4(m), CpuState::Gemma4(s)) => {
                m.verify_speculative(s, spec_buf, pool, kt, window_tokens, start_pos)
            }
            (CpuModel::Lfm2(m), CpuState::Lfm2(s)) => {
                m.verify_speculative(s, spec_buf, pool, kt, window_tokens, start_pos)
            }
            _ => unreachable!("CpuModel and CpuState variants always match"),
        }
    }

    pub fn commit_speculative(
        &self,
        state: &mut CpuState,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        committed: usize,
    ) {
        match (self, state) {
            (CpuModel::Llama(m), CpuState::Llama(s)) => {
                m.commit_speculative(s, spec_buf, committed);
            }
            (CpuModel::Qwen35(m), CpuState::Qwen35(s)) => {
                m.commit_speculative(s, spec_buf, pool, kt, committed);
            }
            (CpuModel::Gemma4(m), CpuState::Gemma4(s)) => {
                m.commit_speculative(s, spec_buf, committed);
            }
            (CpuModel::Lfm2(m), CpuState::Lfm2(s)) => {
                m.commit_speculative(s, spec_buf, committed);
            }
            _ => unreachable!("CpuModel and CpuState variants always match"),
        }
    }

    pub fn embed_token(&self, token_id: u32, out_emb: &mut [f32]) {
        match self {
            CpuModel::Llama(m) => m.embed_token(token_id, out_emb),
            CpuModel::Qwen35(m) => m.embed_token(token_id, out_emb),
            CpuModel::Gemma4(m) => m.embed_token(token_id, out_emb),
            CpuModel::Lfm2(m) => m.embed_token(token_id, out_emb),
        }
    }

    pub fn project_logits_from_hidden(
        &self,
        state: &mut CpuState,
        pool: &ThreadPool,
        kt: &KernelTable,
        hidden: &[f32],
        out_logits: &mut [f32],
    ) {
        match (self, state) {
            (CpuModel::Llama(m), CpuState::Llama(s)) => {
                m.project_logits_from_hidden(s, pool, kt, hidden, out_logits);
            }
            (CpuModel::Qwen35(m), CpuState::Qwen35(s)) => {
                m.project_logits_from_hidden(s, pool, kt, hidden, out_logits);
            }
            (CpuModel::Gemma4(m), CpuState::Gemma4(s)) => {
                m.project_logits_from_hidden(s, pool, kt, hidden, out_logits);
            }
            (CpuModel::Lfm2(m), CpuState::Lfm2(s)) => {
                m.project_logits_from_hidden(s, pool, kt, hidden, out_logits);
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

    pub fn last_hidden(&self) -> &[f32] {
        match self {
            CpuState::Llama(s) => s.last_hidden(),
            CpuState::Qwen35(s) => &s.last_hidden,
            CpuState::Gemma4(s) => &s.last_hidden,
            CpuState::Lfm2(s) => &s.last_hidden,
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

/// Stateful inference session combining `CpuModel`, `CpuState`, `PrefixCache`, and optional `SpecVerifyBuf`.
pub struct Session<'a> {
    pub model: &'a CpuModel<'a>,
    pub state: CpuState,
    pub cache: PrefixCache,
    spec_buf: Option<SpecVerifyBuf>,
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
        Self::new_with_options(model, max_ctx, prefix_cache_bytes, KvType::F16)
    }

    pub fn new_with_options(
        model: &'a CpuModel<'a>,
        max_ctx: usize,
        prefix_cache_bytes: usize,
        kv_type: KvType,
    ) -> Self {
        Self {
            model,
            state: model.new_state_with_kv(max_ctx, kv_type),
            cache: PrefixCache::new(prefix_cache_bytes),
            spec_buf: None,
            pos: 0,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn logits(&self) -> &[f32] {
        self.state.logits()
    }

    /// Pre-allocates `SpecVerifyBuf` up front (zero heap allocations during speculative generation).
    pub fn enable_speculation(&mut self) {
        if self.spec_buf.is_none() {
            self.spec_buf = Some(self.model.new_spec_buf());
        }
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

    /// Generates up to `max_new_tokens` greedy tokens after prompt prefill (`self.pos == prompt_tokens.len()`)
    /// using speculative decoding (`SuffixDrafter`, optional `MtpDrafter`, and `SpecController`).
    ///
    /// Guarantees **100% deterministic token-exact parity** with single-token greedy decoding
    /// across all architectures (`Llama`, `Qwen3`, `Qwen3.5` GDN, `Gemma 4` PLE/SWA, `LFM2` ShortConv)!
    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_speculative<F>(
        &mut self,
        pool: &ThreadPool,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        mode: SpecMode,
        suffix: &mut SuffixDrafter,
        mut mtp: Option<&mut dyn MtpDrafter>,
        controller: &mut SpecController,
        mut should_stop: F,
    ) -> Result<(Vec<u32>, SpecStats), ModelError>
    where
        F: FnMut(&[u32], u32) -> bool,
    {
        let mut stats = SpecStats::default();
        if max_new_tokens == 0 {
            return Ok((Vec::new(), stats));
        }
        if mode != SpecMode::Off {
            self.enable_speculation();
        }

        let kt = active_kernels();
        let vocab = self.model.vocab_size();
        let n_embd = self.model.n_embd();
        let max_ctx = self.state.max_ctx();

        let first_tok = argmax(self.state.logits());
        let mut generated = Vec::with_capacity(max_new_tokens);
        generated.push(first_tok);
        if should_stop(&generated, first_tok) || max_new_tokens == 1 {
            return Ok((generated, stats));
        }

        let mut history = Vec::with_capacity(prompt_tokens.len() + max_new_tokens);
        history.extend_from_slice(prompt_tokens);
        suffix.observe_sequence(&history);
        suffix.observe_step(&history, first_tok);
        history.push(first_tok);

        // Pre-allocated MTP scratch (allocated once before generation loop — zero per-token heap allocs)
        let (mut mtp_prev_h, mut mtp_next_h, mut mtp_emb, mut mtp_logits) = if mtp.is_some() {
            (
                vec![0.0f32; n_embd],
                vec![0.0f32; n_embd],
                vec![0.0f32; n_embd],
                vec![0.0f32; vocab],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        let mut cur_tok = first_tok;
        let mut draft_buf = [0u32; MAX_DRAFT_TOKENS];
        let mut win_buf = [0u32; MAX_SPEC_WINDOW];

        while generated.len() < max_new_tokens {
            let rem_tokens = max_new_tokens - generated.len();
            let rem_ctx = max_ctx.saturating_sub(self.pos + 1);
            let max_k = rem_tokens.min(rem_ctx).min(MAX_DRAFT_TOKENS);

            let mut probe_one = [0u32; 1];
            let suffix_ready = max_k > 0 && suffix.draft(&history, 1, &mut probe_one) > 0;
            let raw_choice = controller.choose(mode, mtp.is_some(), suffix_ready);
            let choice = raw_choice.clamp_len(max_k);

            if mode == SpecMode::Auto
                && matches!(choice, DraftChoice::None)
                && (suffix_ready || mtp.is_some())
            {
                stats.auto_disabled_steps += 1;
            }

            let mut drafted_len = 0usize;
            let mut draft_ns = 0u64;
            let mut used_mtp = false;

            match choice {
                DraftChoice::None => {}
                DraftChoice::Suffix(k) => {
                    drafted_len = suffix.draft(&history, k, &mut draft_buf);
                }
                DraftChoice::Mtp(k) => {
                    if let Some(mtp_head) = mtp.as_deref_mut() {
                        used_mtp = true;
                        let td = Instant::now();
                        mtp_prev_h.copy_from_slice(self.state.last_hidden());
                        let mut prev_tok_for_mtp = cur_tok;
                        for slot in draft_buf.iter_mut().take(k) {
                            self.model.embed_token(prev_tok_for_mtp, &mut mtp_emb);
                            mtp_head.draft_hidden(pool, kt, &mtp_prev_h, &mtp_emb, &mut mtp_next_h);
                            self.model.project_logits_from_hidden(
                                &mut self.state,
                                pool,
                                kt,
                                &mtp_next_h,
                                &mut mtp_logits,
                            );
                            let pred = argmax(&mtp_logits);
                            *slot = pred;
                            mtp_prev_h.copy_from_slice(&mtp_next_h);
                            prev_tok_for_mtp = pred;
                            drafted_len += 1;
                        }
                        draft_ns = td.elapsed().as_nanos() as u64;
                    }
                }
            }

            if drafted_len == 0 {
                let t0 = Instant::now();
                let step_pos = self.pos;
                self.model
                    .forward_token(&mut self.state, pool, kt, cur_tok, step_pos, true)?;
                self.pos = step_pos + 1;
                controller.record_base_step(t0.elapsed().as_nanos() as u64);
                stats.single_steps += 1;
                stats.committed_tokens += 1;

                let next_tok = argmax(self.state.logits());
                generated.push(next_tok);
                suffix.observe_step(&history, next_tok);
                history.push(next_tok);
                cur_tok = next_tok;

                if should_stop(&generated, next_tok) {
                    break;
                }
                continue;
            }

            let w = 1 + drafted_len;
            win_buf[0] = cur_tok;
            win_buf[1..w].copy_from_slice(&draft_buf[..drafted_len]);

            let spec_buf = self
                .spec_buf
                .as_mut()
                .expect("spec_buf initialized via enable_speculation");
            let start_pos = self.pos;
            let tv = Instant::now();
            self.model.verify_speculative(
                &mut self.state,
                spec_buf,
                pool,
                kt,
                &win_buf[..w],
                start_pos,
            )?;
            let verify_ns = tv.elapsed().as_nanos() as u64;

            let mut accepted = 0usize;
            let mut committed = 0usize;
            let mut stop_hit = false;

            for j in 0..=drafted_len {
                let target_tok = argmax(spec_buf.logits_at(j, vocab));
                if j < drafted_len && target_tok == win_buf[j + 1] {
                    accepted += 1;
                    generated.push(target_tok);
                    suffix.observe_step(&history, target_tok);
                    history.push(target_tok);
                    cur_tok = target_tok;

                    if should_stop(&generated, target_tok) || generated.len() >= max_new_tokens {
                        committed = j + 1;
                        stop_hit = true;
                        break;
                    }
                } else {
                    // Either draft `j + 1` was rejected OR all `drafted_len` drafts matched (`j == drafted_len`):
                    // `target_tok` is the exact greedy bonus/correction token!
                    committed = j + 1;
                    generated.push(target_tok);
                    suffix.observe_step(&history, target_tok);
                    history.push(target_tok);
                    cur_tok = target_tok;
                    if should_stop(&generated, target_tok) {
                        stop_hit = true;
                    }
                    break;
                }
            }

            self.model
                .commit_speculative(&mut self.state, spec_buf, pool, kt, committed);
            self.pos = start_pos + committed;

            stats.verify_calls += 1;
            stats.drafted_tokens += drafted_len;
            stats.accepted_tokens += accepted;
            stats.committed_tokens += committed;

            if used_mtp {
                controller.record_mtp_verify(drafted_len, accepted, draft_ns, verify_ns);
            } else {
                controller.record_suffix_verify(drafted_len, accepted, verify_ns);
            }

            if stop_hit {
                break;
            }
        }

        Ok((generated, stats))
    }
}
