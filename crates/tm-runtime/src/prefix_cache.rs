//! Memory-bounded LRU Prefix Cache (`docs/REQUIREMENTS.md` F-13 & `docs/ARCHITECTURE.md` §5.2).
//!
//! Supports BOTH:
//! 1. **Prefix-truncatable KV/SWA models** (`Llama`, `Qwen3`, `Gemma 4`): any cached entry
//!    can satisfy a *shorter* or *longer* common prefix `0..common_len` by slicing `[0..common_len]`
//!    of each `LayerState::Kv` (`SharedKv` layers cost 0 bytes!).
//! 2. **Hybrid recurrent models** (`Qwen3.5` with `DeltaNet`, `LFM2` with `ShortConv`):
//!    recurrent states cannot be truncated arbitrarily, so checkpoints saved at system-prompt
//!    end, turn boundaries, or every `K` tokens are restored only at exact checkpoint length `L <= query.len()`.

use tm_models::{LayerState, LlamaState};

/// Compact snapshot of a single layer at `prefix_len` tokens.
#[derive(Clone, Debug)]
pub enum CachedLayer {
    Kv {
        k_prefix: Vec<u16>,
        v_prefix: Vec<u16>,
        kv_dim: usize,
    },
    SharedKv,
    DeltaNet {
        state: Vec<f32>,
        conv_hist: Vec<f32>,
    },
    ShortConv {
        conv_hist: Vec<f32>,
    },
}

impl CachedLayer {
    pub fn byte_size(&self) -> usize {
        match self {
            CachedLayer::Kv {
                k_prefix, v_prefix, ..
            } => (k_prefix.len() + v_prefix.len()) * core::mem::size_of::<u16>(),
            CachedLayer::SharedKv => 0,
            CachedLayer::DeltaNet { state, conv_hist } => {
                (state.len() + conv_hist.len()) * core::mem::size_of::<f32>()
            }
            CachedLayer::ShortConv { conv_hist } => conv_hist.len() * core::mem::size_of::<f32>(),
        }
    }
}

#[derive(Clone, Debug)]
struct CacheEntry {
    tokens: Vec<u32>,
    has_recurrent: bool,
    layers: Vec<CachedLayer>,
    bytes: usize,
    last_used_tick: u64,
}

/// Memory-bounded LRU Prefix Cache (`F-13`).
pub struct PrefixCache {
    max_bytes: usize,
    used_bytes: usize,
    tick: u64,
    entries: Vec<CacheEntry>,
}

impl PrefixCache {
    /// Creates a new `PrefixCache` bounded by `max_bytes` of total snapshot RAM.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            used_bytes: 0,
            tick: 0,
            entries: Vec::new(),
        }
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.used_bytes = 0;
    }

    /// Snapshots `LlamaState` at `tokens[..prefix_len]` into the LRU cache.
    pub fn save_llama(&mut self, tokens: &[u32], state: &LlamaState) {
        let prefix_len = tokens.len();
        if prefix_len == 0 || prefix_len > state.max_ctx || self.max_bytes == 0 {
            return;
        }
        let kv_dim = state.kv_dim;
        let elems = prefix_len * kv_dim;
        let layers: Vec<CachedLayer> = state
            .k_cache
            .iter()
            .zip(state.v_cache.iter())
            .map(|(k, v)| CachedLayer::Kv {
                k_prefix: k[..elems].to_vec(),
                v_prefix: v[..elems].to_vec(),
                kv_dim,
            })
            .collect();
        self.insert_entry(tokens.to_vec(), false, layers);
    }

    /// Snapshots heterogeneous `&[LayerState]` (`Qwen3.5`, `Gemma 4`, `LFM2`) at `tokens` into the LRU cache.
    pub fn save_layers(&mut self, tokens: &[u32], max_ctx: usize, layers_state: &[LayerState]) {
        let prefix_len = tokens.len();
        if prefix_len == 0 || prefix_len > max_ctx || self.max_bytes == 0 {
            return;
        }
        let has_recurrent = layers_state.iter().any(|l| l.is_recurrent());
        let layers: Vec<CachedLayer> = layers_state
            .iter()
            .map(|ls| match ls {
                LayerState::Kv {
                    k_cache,
                    v_cache,
                    kv_dim,
                    ..
                } => {
                    let elems = prefix_len * *kv_dim;
                    CachedLayer::Kv {
                        k_prefix: k_cache[..elems].to_vec(),
                        v_prefix: v_cache[..elems].to_vec(),
                        kv_dim: *kv_dim,
                    }
                }
                LayerState::SharedKv { .. } => CachedLayer::SharedKv,
                LayerState::DeltaNet { state, conv_hist } => CachedLayer::DeltaNet {
                    state: state.clone(),
                    conv_hist: conv_hist.clone(),
                },
                LayerState::ShortConv { conv_hist } => CachedLayer::ShortConv {
                    conv_hist: conv_hist.clone(),
                },
            })
            .collect();
        self.insert_entry(tokens.to_vec(), has_recurrent, layers);
    }

    /// Restores the longest cached prefix `<= max_restore_len` into `LlamaState`, returning `matched_len`.
    pub fn restore_llama(
        &mut self,
        query_tokens: &[u32],
        max_restore_len: usize,
        state: &mut LlamaState,
    ) -> usize {
        let cap = max_restore_len.min(query_tokens.len()).min(state.max_ctx);
        if cap == 0 {
            return 0;
        }
        let Some((best_idx, matched_len)) = self.find_best_match(query_tokens, cap) else {
            return 0;
        };
        self.tick += 1;
        let entry = &mut self.entries[best_idx];
        entry.last_used_tick = self.tick;

        for (il, cl) in entry.layers.iter().enumerate() {
            if let CachedLayer::Kv {
                k_prefix,
                v_prefix,
                kv_dim,
            } = cl
            {
                let elems = matched_len * *kv_dim;
                state.k_cache[il][..elems].copy_from_slice(&k_prefix[..elems]);
                state.v_cache[il][..elems].copy_from_slice(&v_prefix[..elems]);
            }
        }
        matched_len
    }

    /// Restores the longest valid cached prefix/checkpoint `<= max_restore_len` into `&mut [LayerState]`.
    pub fn restore_layers(
        &mut self,
        query_tokens: &[u32],
        max_restore_len: usize,
        max_ctx: usize,
        layers_state: &mut [LayerState],
    ) -> usize {
        let cap = max_restore_len.min(query_tokens.len()).min(max_ctx);
        if cap == 0 {
            return 0;
        }
        let Some((best_idx, matched_len)) = self.find_best_match(query_tokens, cap) else {
            return 0;
        };
        self.tick += 1;
        let entry = &mut self.entries[best_idx];
        entry.last_used_tick = self.tick;

        for (ls, cl) in layers_state.iter_mut().zip(entry.layers.iter()) {
            match (ls, cl) {
                (
                    LayerState::Kv {
                        k_cache,
                        v_cache,
                        kv_dim,
                        ..
                    },
                    CachedLayer::Kv {
                        k_prefix, v_prefix, ..
                    },
                ) => {
                    let elems = matched_len * *kv_dim;
                    k_cache[..elems].copy_from_slice(&k_prefix[..elems]);
                    v_cache[..elems].copy_from_slice(&v_prefix[..elems]);
                }
                (LayerState::SharedKv { .. }, CachedLayer::SharedKv) => {}
                (
                    LayerState::DeltaNet { state, conv_hist },
                    CachedLayer::DeltaNet {
                        state: cs,
                        conv_hist: ch,
                    },
                ) => {
                    state.copy_from_slice(cs);
                    conv_hist.copy_from_slice(ch);
                }
                (LayerState::ShortConv { conv_hist }, CachedLayer::ShortConv { conv_hist: ch }) => {
                    conv_hist.copy_from_slice(ch);
                }
                _ => {}
            }
        }
        matched_len
    }

    fn find_best_match(
        &self,
        query_tokens: &[u32],
        max_restore_len: usize,
    ) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, u64)> = None;
        for (idx, entry) in self.entries.iter().enumerate() {
            let mlen = if entry.has_recurrent {
                // Recurrent states (`DeltaNet`, `ShortConv`) can ONLY restore at exact checkpoint length!
                let elen = entry.tokens.len();
                if elen <= max_restore_len && query_tokens[..elen] == entry.tokens[..] {
                    elen
                } else {
                    0
                }
            } else {
                // Pure KV / SWA models can truncate any cached prefix down to common prefix `<= max_restore_len`!
                let limit = entry.tokens.len().min(max_restore_len);
                common_prefix_len(&entry.tokens[..limit], &query_tokens[..limit])
            };

            if mlen > 0 {
                let replace = match best {
                    None => true,
                    Some((_, best_len, best_tick)) => {
                        mlen > best_len || (mlen == best_len && entry.last_used_tick > best_tick)
                    }
                };
                if replace {
                    best = Some((idx, mlen, entry.last_used_tick));
                }
            }
        }
        best.map(|(i, len, _)| (i, len))
    }

    fn insert_entry(&mut self, tokens: Vec<u32>, has_recurrent: bool, layers: Vec<CachedLayer>) {
        let layer_bytes: usize = layers.iter().map(|l| l.byte_size()).sum();
        let entry_bytes = layer_bytes + tokens.len() * core::mem::size_of::<u32>();
        if entry_bytes > self.max_bytes {
            return;
        }

        // Deduplicate exact same token sequence if already present
        if let Some(existing_idx) = self.entries.iter().position(|e| e.tokens == tokens) {
            let removed = self.entries.swap_remove(existing_idx);
            self.used_bytes = self.used_bytes.saturating_sub(removed.bytes);
        }

        // Evict least-recently-used entries until `used_bytes + entry_bytes <= max_bytes`
        while !self.entries.is_empty() && self.used_bytes + entry_bytes > self.max_bytes {
            if let Some((lru_idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used_tick)
            {
                let evicted = self.entries.swap_remove(lru_idx);
                self.used_bytes = self.used_bytes.saturating_sub(evicted.bytes);
            }
        }

        self.tick += 1;
        self.used_bytes += entry_bytes;
        self.entries.push(CacheEntry {
            tokens,
            has_recurrent,
            layers,
            bytes: entry_bytes,
            last_used_tick: self.tick,
        });
    }
}

#[inline]
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
