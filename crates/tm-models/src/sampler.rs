//! Token sampler supporting greedy, temperature, top-k, top-p, min-p,
//! and repeat penalty with deterministic PRNG seeding.

/// Sampling hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: 42,
        }
    }
}

/// Reusable token sampler with pre-allocated candidate scratch buffer.
pub struct Sampler {
    cfg: SamplerConfig,
    rng_state: u64,
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig, vocab_size: usize) -> Self {
        let seed = if cfg.seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            cfg.seed
        };
        Self {
            cfg,
            rng_state: seed,
            candidates: Vec::with_capacity(vocab_size),
        }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    /// Samples the next token ID from `logits` (optionally penalizing `recent_tokens`).
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        if self.cfg.temperature <= 0.0 && (self.cfg.repeat_penalty - 1.0).abs() < 1e-6 {
            return argmax(logits);
        }

        self.candidates.clear();
        for (id, &val) in logits.iter().enumerate() {
            self.candidates.push((id as u32, val));
        }

        // Apply repeat penalty (matches llama.cpp convention: divide positive, multiply negative).
        if (self.cfg.repeat_penalty - 1.0).abs() >= 1e-6 && !recent_tokens.is_empty() {
            let pen = self.cfg.repeat_penalty;
            for &tid in recent_tokens {
                if let Some(entry) = self.candidates.get_mut(tid as usize) {
                    if entry.1 > 0.0 {
                        entry.1 /= pen;
                    } else {
                        entry.1 *= pen;
                    }
                }
            }
        }

        if self.cfg.temperature <= 0.0 {
            let mut best_id = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for &(id, v) in &self.candidates {
                if v > best_val {
                    best_val = v;
                    best_id = id;
                }
            }
            return best_id;
        }

        // Sort descending by logit
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Top-k truncation
        if self.cfg.top_k > 0 && self.cfg.top_k < self.candidates.len() {
            self.candidates.truncate(self.cfg.top_k);
        }

        // Temperature + softmax
        let inv_temp = 1.0 / self.cfg.temperature;
        let max_logit = self.candidates[0].1;
        let mut sum = 0.0f32;
        for entry in &mut self.candidates {
            let p = ((entry.1 - max_logit) * inv_temp).exp();
            entry.1 = p;
            sum += p;
        }
        let inv_sum = 1.0 / sum;
        for entry in &mut self.candidates {
            entry.1 *= inv_sum;
        }

        // Min-p filtering
        if self.cfg.min_p > 0.0 {
            let thresh = self.candidates[0].1 * self.cfg.min_p;
            let mut keep = 1usize;
            while keep < self.candidates.len() && self.candidates[keep].1 >= thresh {
                keep += 1;
            }
            self.candidates.truncate(keep);
        }

        // Top-p (nucleus) filtering
        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for entry in &self.candidates {
                cum += entry.1;
                keep += 1;
                if cum >= self.cfg.top_p {
                    break;
                }
            }
            self.candidates.truncate(keep.max(1));
        }

        // Renormalize & sample
        let total: f32 = self.candidates.iter().map(|e| e.1).sum();
        let target = self.next_f32() * total;
        let mut acc = 0.0f32;
        for &(id, p) in &self.candidates {
            acc += p;
            if acc >= target {
                return id;
            }
        }
        self.candidates.last().map(|e| e.0).unwrap_or(0)
    }
}

/// Greedy argmax over logits.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded_determinism() {
        let logits = [0.1f32, 2.5, 1.8, -0.5, 2.4];
        let mut s_greedy = Sampler::new(SamplerConfig::default(), logits.len());
        assert_eq!(s_greedy.sample(&logits, &[]), 1);

        let cfg = SamplerConfig {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            seed: 12345,
        };
        let mut s1 = Sampler::new(cfg, logits.len());
        let mut s2 = Sampler::new(cfg, logits.len());
        let seq1: Vec<u32> = (0..16).map(|_| s1.sample(&logits, &[1])).collect();
        let seq2: Vec<u32> = (0..16).map(|_| s2.sample(&logits, &[1])).collect();
        assert_eq!(seq1, seq2);
    }
}
