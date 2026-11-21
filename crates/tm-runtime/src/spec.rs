// Portions adapted from Strata (`Strata/src/spec/suffix_drafter.cpp` & `controller.cpp`),
// Copyright (c) 2026 Niko1221 and the Strata contributors, licensed under the MIT License.

//! Speculative decoding subsystem (`docs/ARCHITECTURE.md` §5.5 & `docs/REQUIREMENTS.md` F-12):
//! - `SuffixDrafter`: bounded-memory 4-way set-associative trigram + bigram lookup table;
//! - `MtpDrafter` + `LinearMtpHead`: Multi-Token Prediction (`hidden[i] + embed(token[i+1]) -> logits[i+2]`);
//! - `SpecController`: online EMA acceptance/latency cost model that picks `(source, k in 1..=8)`
//!   and automatically disables speculation whenever expected net gain `< 5%`.

use core::str::FromStr;

use tm_core::DType;
use tm_cpu::{KernelTable, MatvecScratch, ThreadPool, matvec_f32, rms_norm};

/// Maximum draft tokens proposed per verification step (`MAX_SPEC_WINDOW - 1`).
pub const MAX_DRAFT_TOKENS: usize = 8;

/// Minimum net throughput gain (`5%`) required to keep speculative decoding active (`F-12`).
pub const MIN_NET_GAIN_THRESHOLD: f64 = 0.05;

/// User-facing speculative decoding policy (`--spec off|auto|suffix|mtp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecMode {
    #[default]
    Off,
    /// Online EMA controller selects `None`, `Suffix(k)`, or `Mtp(k)` (`< 5%` gain auto-disables).
    Auto,
    /// Forces `SuffixDrafter` with adaptive `k` (`1..=MAX_DRAFT_TOKENS`) when matches exist.
    Suffix,
    /// Forces `MtpDrafter` when an MTP head is attached (falls back to single-token otherwise).
    Mtp,
}

impl SpecMode {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            SpecMode::Off => "off",
            SpecMode::Auto => "auto",
            SpecMode::Suffix => "suffix",
            SpecMode::Mtp => "mtp",
        }
    }
}

impl FromStr for SpecMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "false" | "0" => Ok(SpecMode::Off),
            "auto" | "on" | "true" | "1" => Ok(SpecMode::Auto),
            "suffix" | "ngram" => Ok(SpecMode::Suffix),
            "mtp" => Ok(SpecMode::Mtp),
            other => Err(format!(
                "unsupported --spec '{other}' (expected 'off', 'auto', 'suffix', or 'mtp')"
            )),
        }
    }
}

/// Selected draft source and draft length `k` (`1 <= k <= MAX_DRAFT_TOKENS`) for the next step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftChoice {
    None,
    Suffix(usize),
    Mtp(usize),
}

impl DraftChoice {
    #[inline]
    pub fn clamp_len(self, max_k: usize) -> Self {
        if max_k == 0 {
            return DraftChoice::None;
        }
        match self {
            DraftChoice::None => DraftChoice::None,
            DraftChoice::Suffix(k) => DraftChoice::Suffix(k.min(max_k).max(1)),
            DraftChoice::Mtp(k) => DraftChoice::Mtp(k.min(max_k).max(1)),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct NgramSlot {
    key: u64,
    next_tok: u32,
    count: u16,
    age: u16,
}

/// Bounded-capacity 4-way set-associative 4-gram + trigram + bigram suffix drafter (`F-12`).
///
/// Memory footprint is strictly fixed at construction (`num_sets * 3 * 4 * 16` bytes; default `2048`
/// sets = `384 KiB`), with zero heap allocations during `observe_*` or `draft`.
pub struct SuffixDrafter {
    mask: usize,
    fourgrams: Vec<[NgramSlot; 4]>,
    trigrams: Vec<[NgramSlot; 4]>,
    bigrams: Vec<[NgramSlot; 4]>,
    tick: u16,
}

impl Default for SuffixDrafter {
    fn default() -> Self {
        Self::with_sets(2048)
    }
}

impl SuffixDrafter {
    /// Creates a bounded `SuffixDrafter` with `num_sets` (rounded up to a power of two).
    pub fn with_sets(num_sets: usize) -> Self {
        let sets = num_sets.max(64).next_power_of_two();
        Self {
            mask: sets - 1,
            fourgrams: vec![[NgramSlot::default(); 4]; sets],
            trigrams: vec![[NgramSlot::default(); 4]; sets],
            bigrams: vec![[NgramSlot::default(); 4]; sets],
            tick: 0,
        }
    }

    pub fn clear(&mut self) {
        self.fourgrams.fill([NgramSlot::default(); 4]);
        self.trigrams.fill([NgramSlot::default(); 4]);
        self.bigrams.fill([NgramSlot::default(); 4]);
        self.tick = 0;
    }

    #[inline]
    fn hash_key(key: u64) -> usize {
        let mut x = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (x ^ (x >> 31)) as usize
    }

    #[inline]
    fn fourgram_key(t0: u32, t1: u32, t2: u32) -> u64 {
        let h01 =
            (t0 as u64).wrapping_mul(0x9E37_79B1_85EB_CA87) ^ ((t1 as u64) << 21) ^ (t1 as u64);
        (h01.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ (t2 as u64)) | 0xC000_0000_0000_0000
    }

    #[inline]
    fn trigram_key(t0: u32, t1: u32) -> u64 {
        (((t0 as u64) << 32) | (t1 as u64)) & !0xC000_0000_0000_0000 | 0x8000_0000_0000_0000
    }

    #[inline]
    fn bigram_key(t0: u32) -> u64 {
        (t0 as u64) | 0x4000_0000_0000_0000
    }

    fn insert_slot(table: &mut [[NgramSlot; 4]], mask: usize, key: u64, next_tok: u32, tick: u16) {
        let set_idx = Self::hash_key(key) & mask;
        let ways = &mut table[set_idx];

        // 1. Exact `(key, next_tok)` match -> bump frequency + recency
        for w in ways.iter_mut() {
            if w.count > 0 && w.key == key && w.next_tok == next_tok {
                w.count = w.count.saturating_add(1);
                w.age = tick;
                return;
            }
        }
        // 2. Empty way available
        for w in ways.iter_mut() {
            if w.count == 0 {
                *w = NgramSlot {
                    key,
                    next_tok,
                    count: 1,
                    age: tick,
                };
                return;
            }
        }
        // 3. Same `key` with stale single-hit successor -> overwrite with latest continuation
        for w in ways.iter_mut() {
            if w.key == key && w.count <= 1 {
                *w = NgramSlot {
                    key,
                    next_tok,
                    count: 1,
                    age: tick,
                };
                return;
            }
        }
        // 4. Evict lowest-count (tie-broken by oldest `age`) way
        let mut victim = 0usize;
        let mut best_score = u32::MAX;
        for (i, w) in ways.iter().enumerate() {
            let recency = tick.wrapping_sub(w.age);
            let score = ((w.count as u32) << 16) | (u16::MAX.wrapping_sub(recency) as u32);
            if score < best_score {
                best_score = score;
                victim = i;
            }
        }
        ways[victim] = NgramSlot {
            key,
            next_tok,
            count: 1,
            age: tick,
        };
    }

    #[inline]
    fn lookup_best(
        table: &[[NgramSlot; 4]],
        mask: usize,
        key: u64,
        tick: u16,
    ) -> Option<(u32, u16)> {
        let set_idx = Self::hash_key(key) & mask;
        let ways = &table[set_idx];
        let mut best: Option<(u32, u16, u16)> = None;
        for w in ways {
            if w.count > 0 && w.key == key {
                let recency = tick.wrapping_sub(w.age);
                let replace = match best {
                    None => true,
                    Some((_, bc, br)) => w.count > bc || (w.count == bc && recency < br),
                };
                if replace {
                    best = Some((w.next_tok, w.count, recency));
                }
            }
        }
        best.map(|(tok, cnt, _)| (tok, cnt))
    }

    /// Observes all adjacent 2-gram and 3-gram transitions in `tokens`.
    pub fn observe_sequence(&mut self, tokens: &[u32]) {
        if tokens.len() < 2 {
            return;
        }
        for i in 1..tokens.len() {
            self.tick = self.tick.wrapping_add(1);
            let prev1 = tokens[i - 1];
            let cur = tokens[i];
            Self::insert_slot(
                &mut self.bigrams,
                self.mask,
                Self::bigram_key(prev1),
                cur,
                self.tick,
            );
            if i >= 2 {
                let prev2 = tokens[i - 2];
                Self::insert_slot(
                    &mut self.trigrams,
                    self.mask,
                    Self::trigram_key(prev2, prev1),
                    cur,
                    self.tick,
                );
            }
            if i >= 3 {
                let prev3 = tokens[i - 3];
                let prev2 = tokens[i - 2];
                Self::insert_slot(
                    &mut self.fourgrams,
                    self.mask,
                    Self::fourgram_key(prev3, prev2, prev1),
                    cur,
                    self.tick,
                );
            }
        }
    }

    /// Observes a single newly confirmed token `next_tok` appended after `prev_context`.
    #[inline]
    pub fn observe_step(&mut self, prev_context: &[u32], next_tok: u32) {
        let n = prev_context.len();
        if n == 0 {
            return;
        }
        self.tick = self.tick.wrapping_add(1);
        let prev1 = prev_context[n - 1];
        Self::insert_slot(
            &mut self.bigrams,
            self.mask,
            Self::bigram_key(prev1),
            next_tok,
            self.tick,
        );
        if n >= 2 {
            let prev2 = prev_context[n - 2];
            Self::insert_slot(
                &mut self.trigrams,
                self.mask,
                Self::trigram_key(prev2, prev1),
                next_tok,
                self.tick,
            );
        }
        if n >= 3 {
            let prev3 = prev_context[n - 3];
            let prev2 = prev_context[n - 2];
            Self::insert_slot(
                &mut self.fourgrams,
                self.mask,
                Self::fourgram_key(prev3, prev2, prev1),
                next_tok,
                self.tick,
            );
        }
    }

    /// Proposes up to `k` (`<= out.len()`) draft tokens from `context` (zero heap allocations).
    /// Prefers 4-gram `(t_{-3}, t_{-2}, t_{-1}) -> next`, then trigram `(t_{-2}, t_{-1}) -> next`,
    /// falling back to bigram `t_{-1} -> next` when observed at least twice (or on step 0).
    pub fn draft(&self, context: &[u32], k: usize, out: &mut [u32]) -> usize {
        let limit = k.min(out.len()).min(MAX_DRAFT_TOKENS);
        if limit == 0 || context.is_empty() {
            return 0;
        }

        let mut prev1 = context[context.len() - 1];
        let mut prev2 = if context.len() >= 2 {
            Some(context[context.len() - 2])
        } else {
            None
        };
        let mut prev3 = if context.len() >= 3 {
            Some(context[context.len() - 3])
        } else {
            None
        };

        let mut drafted = 0usize;
        let mut repeat_run = 0usize;
        while drafted < limit {
            let four_hit = prev3.zip(prev2).and_then(|(p3, p2)| {
                Self::lookup_best(
                    &self.fourgrams,
                    self.mask,
                    Self::fourgram_key(p3, p2, prev1),
                    self.tick,
                )
            });
            let tri_hit = if four_hit.is_none() {
                prev2.and_then(|p2| {
                    Self::lookup_best(
                        &self.trigrams,
                        self.mask,
                        Self::trigram_key(p2, prev1),
                        self.tick,
                    )
                })
            } else {
                None
            };
            let chosen = if let Some((tok, _cnt)) = four_hit {
                Some(tok)
            } else if let Some((tok, _cnt)) = tri_hit {
                Some(tok)
            } else if let Some((tok, cnt)) =
                Self::lookup_best(&self.bigrams, self.mask, Self::bigram_key(prev1), self.tick)
            {
                // Require count >= 2 for unanchored multi-step bigram chains to avoid noisy speculation
                if drafted == 0 || cnt >= 2 {
                    Some(tok)
                } else {
                    None
                }
            } else {
                None
            };

            let Some(next_tok) = chosen else {
                break;
            };
            if next_tok == prev1 {
                repeat_run += 1;
                if repeat_run > 2 {
                    break;
                }
            } else {
                repeat_run = 0;
            }

            out[drafted] = next_tok;
            prev3 = prev2;
            prev2 = Some(prev1);
            prev1 = next_tok;
            drafted += 1;
        }
        drafted
    }
}

/// Multi-Token Prediction (MTP) cell interface (`docs/ARCHITECTURE.md` §5.5).
///
/// Position-alignment contract (`AGENTS.md` / `ARCHITECTURE.md` §5.5):
/// - Base model step at `i` produces `hidden[i]` and `token[i + 1]`.
/// - `draft_hidden` consumes `(hidden[i], embed(token[i + 1]))` and writes `mtp_hidden[i + 1]`,
///   which the caller projects through the base model's `output_norm` + `lm_head` to obtain
///   `logits[i + 2]` -> `token[i + 2]`!
/// - Chaining step `d` feeds `(mtp_hidden[i + d - 1], embed(token[i + d])) -> mtp_hidden[i + d]`.
pub trait MtpDrafter {
    /// Computes `out_hidden` (`[n_embd]`) from `prev_hidden` (`hidden[i]`) and `next_token_emb` (`embed(token[i+1])`).
    fn draft_hidden(
        &mut self,
        pool: &ThreadPool,
        kt: &KernelTable,
        prev_hidden: &[f32],
        next_token_emb: &[f32],
        out_hidden: &mut [f32],
    );
}

/// Lightweight RMSNorm + `[2 * n_embd -> n_embd]` fusion MTP cell (`ARCHITECTURE.md` §5.5)
/// sharing `token_embd` and `lm_head` with the base model. Zero heap allocations on `draft_hidden`.
pub struct LinearMtpHead {
    n_embd: usize,
    rms_eps: f32,
    h_norm: Vec<f32>,
    e_norm: Vec<f32>,
    proj_dtype: DType,
    proj_bytes: Vec<u8>,
    concat_buf: Vec<f32>,
    mv_scratch: MatvecScratch,
}

impl LinearMtpHead {
    pub fn new(
        n_embd: usize,
        rms_eps: f32,
        h_norm: Vec<f32>,
        e_norm: Vec<f32>,
        proj_dtype: DType,
        proj_bytes: Vec<u8>,
    ) -> Self {
        assert_eq!(h_norm.len(), n_embd);
        assert_eq!(e_norm.len(), n_embd);
        assert_eq!(proj_bytes.len(), proj_dtype.row_bytes(2 * n_embd) * n_embd);
        Self {
            n_embd,
            rms_eps,
            h_norm,
            e_norm,
            proj_dtype,
            proj_bytes,
            concat_buf: vec![0.0; 2 * n_embd],
            mv_scratch: MatvecScratch::new(2 * n_embd),
        }
    }
}

impl MtpDrafter for LinearMtpHead {
    fn draft_hidden(
        &mut self,
        pool: &ThreadPool,
        kt: &KernelTable,
        prev_hidden: &[f32],
        next_token_emb: &[f32],
        out_hidden: &mut [f32],
    ) {
        let d = self.n_embd;
        let (h_part, e_part) = self.concat_buf.split_at_mut(d);
        rms_norm(h_part, &prev_hidden[..d], &self.h_norm, self.rms_eps);
        rms_norm(e_part, &next_token_emb[..d], &self.e_norm, self.rms_eps);
        matvec_f32(
            pool,
            kt,
            self.proj_dtype,
            &self.proj_bytes,
            &self.concat_buf,
            &mut self.mv_scratch,
            &mut out_hidden[..d],
            d,
            2 * d,
        );
    }
}

/// Cumulative runtime telemetry for speculative decoding (`F-12`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub verify_calls: usize,
    pub single_steps: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub committed_tokens: usize,
    pub auto_disabled_steps: usize,
}

impl SpecStats {
    #[inline]
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            (self.accepted_tokens as f64) / (self.drafted_tokens as f64)
        }
    }

    #[inline]
    pub fn mean_tokens_per_verify(&self) -> f64 {
        if self.verify_calls == 0 {
            1.0
        } else {
            (self.accepted_tokens + self.verify_calls) as f64 / (self.verify_calls as f64)
        }
    }
}

/// Online EMA controller (`docs/ARCHITECTURE.md` §5.5) choosing `(source, draft_len)` per step
/// and automatically disabling speculation whenever net throughput gain `< 5%` (`F-12`).
#[derive(Debug, Clone)]
pub struct SpecController {
    alpha: f64,
    max_draft: usize,
    ema_base_ns: f64,
    ema_suffix_accept: f64,
    ema_suffix_overhead_per_tok: f64,
    ema_mtp_accept: f64,
    ema_mtp_overhead_per_tok: f64,
    ema_mtp_draft_ratio: f64,
    suffix_trials: usize,
    mtp_trials: usize,
    cooldown_remaining: usize,
    probe_interval: usize,
}

impl Default for SpecController {
    fn default() -> Self {
        Self::new(MAX_DRAFT_TOKENS)
    }
}

impl SpecController {
    pub fn new(max_draft: usize) -> Self {
        Self {
            alpha: 0.25,
            max_draft: max_draft.clamp(1, MAX_DRAFT_TOKENS),
            ema_base_ns: 0.0,
            ema_suffix_accept: 0.55,
            // Shared-weight GEMM typically adds ~14% wall time per extra verified token in the window
            ema_suffix_overhead_per_tok: 0.14,
            ema_mtp_accept: 0.60,
            ema_mtp_overhead_per_tok: 0.14,
            ema_mtp_draft_ratio: 0.08,
            suffix_trials: 0,
            mtp_trials: 0,
            cooldown_remaining: 0,
            probe_interval: 32,
        }
    }

    #[inline]
    pub fn suffix_accept_ema(&self) -> f64 {
        self.ema_suffix_accept
    }

    #[inline]
    pub fn mtp_accept_ema(&self) -> f64 {
        self.ema_mtp_accept
    }

    /// Expected net speedup (`gain = E[tokens] / E[cost_ratio] - 1.0`) for `k` draft tokens.
    #[inline]
    pub fn expected_gain(
        accept_prob: f64,
        verify_overhead_per_tok: f64,
        draft_cost_ratio_per_tok: f64,
        k: usize,
    ) -> f64 {
        let p = accept_prob.clamp(0.0, 0.99);
        let mut expected_tokens = 1.0;
        let mut p_pow = 1.0;
        for _ in 0..k {
            p_pow *= p;
            expected_tokens += p_pow;
        }
        let cost_ratio =
            1.0 + (k as f64) * (verify_overhead_per_tok.max(0.02) + draft_cost_ratio_per_tok);
        (expected_tokens / cost_ratio) - 1.0
    }

    fn best_k_and_gain(
        &self,
        accept_prob: f64,
        verify_overhead_per_tok: f64,
        draft_cost_ratio_per_tok: f64,
    ) -> (usize, f64) {
        let mut best_k = 1usize;
        let mut best_gain = f64::NEG_INFINITY;
        for k in 1..=self.max_draft {
            let g = Self::expected_gain(
                accept_prob,
                verify_overhead_per_tok,
                draft_cost_ratio_per_tok,
                k,
            );
            if g > best_gain {
                best_gain = g;
                best_k = k;
            }
        }
        (best_k, best_gain)
    }

    /// Selects the next draft strategy (`None`, `Suffix(k)`, or `Mtp(k)`).
    pub fn choose(
        &mut self,
        mode: SpecMode,
        has_mtp: bool,
        suffix_has_candidate: bool,
    ) -> DraftChoice {
        match mode {
            SpecMode::Off => DraftChoice::None,
            SpecMode::Suffix => {
                if !suffix_has_candidate {
                    DraftChoice::None
                } else {
                    let (k, _) = self.best_k_and_gain(
                        self.ema_suffix_accept.max(0.35),
                        self.ema_suffix_overhead_per_tok,
                        0.0,
                    );
                    DraftChoice::Suffix(k.clamp(2, self.max_draft))
                }
            }
            SpecMode::Mtp => {
                if !has_mtp {
                    DraftChoice::None
                } else {
                    let (k, _) = self.best_k_and_gain(
                        self.ema_mtp_accept.max(0.35),
                        self.ema_mtp_overhead_per_tok,
                        self.ema_mtp_draft_ratio,
                    );
                    DraftChoice::Mtp(k.clamp(1, self.max_draft))
                }
            }
            SpecMode::Auto => {
                if self.cooldown_remaining > 0 {
                    self.cooldown_remaining -= 1;
                    return DraftChoice::None;
                }

                let mut best_choice = DraftChoice::None;
                let mut best_gain = MIN_NET_GAIN_THRESHOLD;

                if suffix_has_candidate {
                    // During initial 3 trials, allow probing `Suffix(4)` to calibrate acceptance EMA
                    if self.suffix_trials < 3 {
                        return DraftChoice::Suffix(4.min(self.max_draft));
                    }
                    let (k, gain) = self.best_k_and_gain(
                        self.ema_suffix_accept,
                        self.ema_suffix_overhead_per_tok,
                        0.0,
                    );
                    if gain >= best_gain {
                        best_gain = gain;
                        best_choice = DraftChoice::Suffix(k);
                    }
                }

                if has_mtp {
                    if self.mtp_trials < 3 && matches!(best_choice, DraftChoice::None) {
                        return DraftChoice::Mtp(2.min(self.max_draft));
                    }
                    let (k, gain) = self.best_k_and_gain(
                        self.ema_mtp_accept,
                        self.ema_mtp_overhead_per_tok,
                        self.ema_mtp_draft_ratio,
                    );
                    if gain >= best_gain {
                        best_choice = DraftChoice::Mtp(k);
                    }
                }

                // If we have enough observations and every source yields < 5% net gain, enter cooldown
                // and periodically re-probe every `probe_interval` tokens!
                if matches!(best_choice, DraftChoice::None)
                    && (suffix_has_candidate || has_mtp)
                    && (self.suffix_trials >= 3 || self.mtp_trials >= 3)
                {
                    self.cooldown_remaining = self.probe_interval;
                    // On cooldown expiry (`cooldown_remaining` was 0 when we entered), allow 1 probe
                    // every `probe_interval` steps by resetting trial window slightly:
                    if suffix_has_candidate
                        && self.suffix_trials.is_multiple_of(self.probe_interval)
                    {
                        self.cooldown_remaining = 0;
                        return DraftChoice::Suffix(2.min(self.max_draft));
                    }
                }

                best_choice
            }
        }
    }

    /// Updates baseline single-token step time EMA (`ns`).
    #[inline]
    pub fn record_base_step(&mut self, step_ns: u64) {
        let ns = step_ns as f64;
        if ns <= 0.0 {
            return;
        }
        if self.ema_base_ns <= 0.0 {
            self.ema_base_ns = ns;
        } else {
            self.ema_base_ns = (1.0 - self.alpha) * self.ema_base_ns + self.alpha * ns;
        }
    }

    /// Updates EMA acceptance rate and per-token verification overhead after a suffix verify call.
    pub fn record_suffix_verify(&mut self, drafted: usize, accepted: usize, verify_ns: u64) {
        if drafted == 0 {
            return;
        }
        self.suffix_trials += 1;
        let obs_acc = (accepted as f64) / (drafted as f64);
        self.ema_suffix_accept = (1.0 - self.alpha) * self.ema_suffix_accept + self.alpha * obs_acc;

        if self.ema_base_ns > 0.0 && verify_ns > 0 {
            let ratio = (verify_ns as f64) / self.ema_base_ns;
            let per_tok = ((ratio - 1.0) / (drafted as f64)).clamp(0.02, 1.5);
            self.ema_suffix_overhead_per_tok =
                (1.0 - self.alpha) * self.ema_suffix_overhead_per_tok + self.alpha * per_tok;
        }
    }

    /// Updates EMA acceptance rate, draft cost, and verification overhead after an MTP verify call.
    pub fn record_mtp_verify(
        &mut self,
        drafted: usize,
        accepted: usize,
        draft_ns: u64,
        verify_ns: u64,
    ) {
        if drafted == 0 {
            return;
        }
        self.mtp_trials += 1;
        let obs_acc = (accepted as f64) / (drafted as f64);
        self.ema_mtp_accept = (1.0 - self.alpha) * self.ema_mtp_accept + self.alpha * obs_acc;

        if self.ema_base_ns > 0.0 {
            if draft_ns > 0 {
                let d_ratio =
                    ((draft_ns as f64) / self.ema_base_ns / (drafted as f64)).clamp(0.005, 1.0);
                self.ema_mtp_draft_ratio =
                    (1.0 - self.alpha) * self.ema_mtp_draft_ratio + self.alpha * d_ratio;
            }
            if verify_ns > 0 {
                let ratio = (verify_ns as f64) / self.ema_base_ns;
                let per_tok = ((ratio - 1.0) / (drafted as f64)).clamp(0.02, 1.5);
                self.ema_mtp_overhead_per_tok =
                    (1.0 - self.alpha) * self.ema_mtp_overhead_per_tok + self.alpha * per_tok;
            }
        }
    }
}
