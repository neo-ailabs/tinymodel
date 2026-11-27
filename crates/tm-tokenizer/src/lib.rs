#![forbid(unsafe_code)]

//! Lightweight, flat-arena pure-Rust GPT-2 byte-level BPE tokenizer loaded from GGUF metadata.
//!
//! Stores the entire vocabulary in a single contiguous `String` arena (`vocab_pool` +
//! `vocab_offsets`) with an open-addressed `u32` hash index and integer-pair BPE merges
//! (`(id_a, id_b) -> (rank, id_merged)`), eliminating ~1M per-token heap allocations and
//! saving ~60 MiB of dirty RAM on 248k-vocab models (such as Qwen3.5) for Pi Zero 2W / Pi 4B.

use std::collections::HashMap;

use thiserror::Error;
use tm_gguf::GgufFile;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("GGUF missing tokenizer metadata key '{0}'")]
    MissingKey(&'static str),
    #[error("unsupported tokenizer model '{0}' (currently supported: 'gpt2', 'llama', 'gemma4')")]
    UnsupportedModel(String),
    #[error("invalid HuggingFace tokenizer.json: {0}")]
    InvalidJson(String),
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerMode {
    Gpt2ByteBpe,
    Gpt4oByteBpe,
    DeepSeek3ByteBpe,
    Gemma4SpmBpe,
    LlamaSpm,
}

const EMPTY_SLOT: u32 = u32::MAX;

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Lightweight BPE tokenizer (covers GPT-2 byte-level BPE for SmolLM2/Llama-3/Qwen/LFM2
/// and SPM-style raw UTF-8 BPE for Gemma 4).
pub struct Tokenizer {
    mode: TokenizerMode,
    real_vocab_size: usize,
    vocab_pool: String,
    vocab_offsets: Vec<u32>,
    /// Open-addressed hash table (size = power of 2, `EMPTY_SLOT` = empty) mapping string -> token ID.
    vocab_table: Vec<u32>,
    vocab_mask: usize,
    /// Packed `(id_a << 32) | id_b` -> `(merge_rank, merged_id)`.
    merges: HashMap<u64, (u32, u32)>,
    /// Optional SentencePiece scores (`tokenizer.ggml.scores` for `model == "llama"`).
    scores: Vec<f32>,
    add_space_prefix: bool,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub eot_id: Option<u32>,
    pub add_bos: bool,
    max_digits: usize,
    /// Control / user-defined special tokens (`token_type == 3 | 4`), sorted by byte length descending.
    special_tokens: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Loads tokenizer tables directly from an opened `GgufFile` into a flat contiguous arena.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = gguf
            .get_meta("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt2");
        let pre = gguf
            .get_meta("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let arch = gguf
            .get_meta("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let mode = if model == "gemma4" || pre == "gemma4" {
            TokenizerMode::Gemma4SpmBpe
        } else if model == "llama" {
            TokenizerMode::LlamaSpm
        } else if matches!(
            pre,
            "inkling2" | "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2"
        ) {
            TokenizerMode::Gpt4oByteBpe
        } else if model == "gpt2" {
            TokenizerMode::Gpt2ByteBpe
        } else {
            return Err(TokenizerError::UnsupportedModel(model.to_string()));
        };

        let (total_bytes, token_iter) = gguf
            .string_array_iter("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.tokens"))?;

        let real_vocab_size = token_iter.len();

        let mut vocab_pool = String::with_capacity(total_bytes + 4096);
        let mut vocab_offsets = Vec::with_capacity(real_vocab_size + 256);
        vocab_offsets.push(0u32);

        // Size open-addressed table for <= 50% load factor (including any extra interned merge pieces)
        let table_cap = ((real_vocab_size + 4096) * 2).next_power_of_two().max(64);
        let vocab_mask = table_cap - 1;
        let mut vocab_table = vec![EMPTY_SLOT; table_cap];

        for (id, s) in token_iter.enumerate() {
            vocab_pool.push_str(s);
            vocab_offsets.push(vocab_pool.len() as u32);
            table_upsert(
                &mut vocab_table,
                vocab_mask,
                &vocab_pool,
                &vocab_offsets,
                s,
                id as u32,
            );
        }

        let mut merges = HashMap::new();
        if let Some((_, mlist)) = gguf.string_array_iter("tokenizer.ggml.merges") {
            merges.reserve(mlist.len());
            let mut merged_scratch = String::with_capacity(64);
            for (rank, line) in mlist.enumerate() {
                if let Some((a, b)) = line.split_once(' ') {
                    let id_a = intern_or_lookup(
                        &mut vocab_table,
                        vocab_mask,
                        &mut vocab_pool,
                        &mut vocab_offsets,
                        a,
                    );
                    let id_b = intern_or_lookup(
                        &mut vocab_table,
                        vocab_mask,
                        &mut vocab_pool,
                        &mut vocab_offsets,
                        b,
                    );
                    merged_scratch.clear();
                    merged_scratch.push_str(a);
                    merged_scratch.push_str(b);
                    let id_ab = intern_or_lookup(
                        &mut vocab_table,
                        vocab_mask,
                        &mut vocab_pool,
                        &mut vocab_offsets,
                        &merged_scratch,
                    );
                    let pair_key = ((id_a as u64) << 32) | (id_b as u64);
                    merges.entry(pair_key).or_insert((rank as u32, id_ab));
                }
            }
        }

        let mut bos_id = gguf
            .get_meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        if pre == "inkling2" {
            // Matches `llama-vocab.cpp:2674`: `inkling2` sets `special_bos_id = LLAMA_TOKEN_NULL`.
            bos_id = None;
        }
        let eos_id = gguf
            .get_meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let mut eot_id = gguf
            .get_meta("tokenizer.ggml.eot_token_id")
            .and_then(|v| v.as_u32());
        if eot_id.is_none() {
            for eog_str in ["<turn|>", "<end_of_turn>", "<|eot_id|>", "<|im_end|>"] {
                if let Some(id) = table_lookup(
                    &vocab_table,
                    vocab_mask,
                    &vocab_pool,
                    &vocab_offsets,
                    eog_str,
                ) && (id as usize) < real_vocab_size
                {
                    eot_id = Some(id);
                    break;
                }
            }
        }
        let mut add_bos = gguf
            .get_meta("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(mode == TokenizerMode::LlamaSpm);
        if mode == TokenizerMode::Gemma4SpmBpe {
            add_bos = true;
        }
        let add_space_prefix = gguf
            .get_meta("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| mode == TokenizerMode::LlamaSpm && !arch.starts_with("gemma"));
        let scores = if mode == TokenizerMode::LlamaSpm {
            gguf.get_meta("tokenizer.ggml.scores")
                .and_then(|v| v.as_f32_slice())
                .map(|s| s.to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        // Qwen2 / Qwen3 / Qwen3.5 BPE regex uses single-digit `\p{N}` instead of `\p{N}{1,3}`.
        let max_digits = if pre.starts_with("qwen") { 1 } else { 3 };

        let mut special_tokens = Vec::new();
        if let Some(ttypes) = gguf
            .get_meta("tokenizer.ggml.token_type")
            .and_then(|v| v.as_i32_array())
        {
            for (id, &tt) in ttypes.iter().enumerate().take(real_vocab_size) {
                // `LLAMA_TOKEN_TYPE_CONTROL == 3`, `LLAMA_TOKEN_TYPE_USER_DEFINED == 4`
                if tt == 3 || tt == 4 {
                    let s_off = vocab_offsets[id] as usize;
                    let e_off = vocab_offsets[id + 1] as usize;
                    let tok_s = &vocab_pool[s_off..e_off];
                    if tok_s.len() >= 2 {
                        special_tokens.push((tok_s.to_string(), id as u32));
                    }
                }
            }
            // Longest special token first so greedy prefix scan matches exact longest control token.
            special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_maps();

        Ok(Self {
            mode,
            real_vocab_size,
            vocab_pool,
            vocab_offsets,
            vocab_table,
            vocab_mask,
            merges,
            scores,
            add_space_prefix,
            byte_to_char,
            char_to_byte,
            bos_id,
            eos_id,
            eot_id,
            add_bos,
            max_digits,
            special_tokens,
        })
    }

    /// Loads a GPT-2 / DeepSeek-3 `ByteLevel` BPE tokenizer from a HuggingFace `tokenizer.json` string
    /// (e.g. `Section 0` `tokenizer_json` in `.strata` v1.0 artifacts for `Tencent Hy4-Flash`).
    pub fn from_hf_tokenizer_json(
        json_str: &str,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        let parsed = parse_hf_bpe_json(json_str)?;
        let real_vocab_size = parsed.tokens.len();
        let total_bytes: usize = parsed.tokens.iter().map(|s| s.len()).sum();

        let mut vocab_pool = String::with_capacity(total_bytes + 4096);
        let mut vocab_offsets = Vec::with_capacity(real_vocab_size + 256);
        vocab_offsets.push(0u32);

        let table_cap = ((real_vocab_size + 4096) * 2).next_power_of_two().max(64);
        let vocab_mask = table_cap - 1;
        let mut vocab_table = vec![EMPTY_SLOT; table_cap];

        for (id, s) in parsed.tokens.iter().enumerate() {
            vocab_pool.push_str(s);
            vocab_offsets.push(vocab_pool.len() as u32);
            if !s.is_empty() {
                table_upsert(
                    &mut vocab_table,
                    vocab_mask,
                    &vocab_pool,
                    &vocab_offsets,
                    s,
                    id as u32,
                );
            }
        }

        let mut merges = HashMap::with_capacity(parsed.merges.len());
        let mut merged_scratch = String::with_capacity(64);
        for (rank, (a, b)) in parsed.merges.iter().enumerate() {
            let id_a = intern_or_lookup(
                &mut vocab_table,
                vocab_mask,
                &mut vocab_pool,
                &mut vocab_offsets,
                a,
            );
            let id_b = intern_or_lookup(
                &mut vocab_table,
                vocab_mask,
                &mut vocab_pool,
                &mut vocab_offsets,
                b,
            );
            merged_scratch.clear();
            merged_scratch.push_str(a);
            merged_scratch.push_str(b);
            let id_ab = intern_or_lookup(
                &mut vocab_table,
                vocab_mask,
                &mut vocab_pool,
                &mut vocab_offsets,
                &merged_scratch,
            );
            let pair_key = ((id_a as u64) << 32) | (id_b as u64);
            merges.entry(pair_key).or_insert((rank as u32, id_ab));
        }

        let mut special_tokens = parsed.special_tokens;
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));

        let mut eot_id = None;
        for eog_str in [
            "<|eos|>",
            "<｜end▁of▁sentence｜>",
            "<|im_end|>",
            "<|eot_id|>",
            "<end_of_turn>",
        ] {
            if let Some(id) = table_lookup(
                &vocab_table,
                vocab_mask,
                &vocab_pool,
                &vocab_offsets,
                eog_str,
            ) && (id as usize) < real_vocab_size
            {
                eot_id = Some(id);
                break;
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_maps();
        let mode = if parsed.is_deepseek3 {
            TokenizerMode::DeepSeek3ByteBpe
        } else {
            TokenizerMode::Gpt2ByteBpe
        };

        Ok(Self {
            mode,
            real_vocab_size,
            vocab_pool,
            vocab_offsets,
            vocab_table,
            vocab_mask,
            merges,
            scores: Vec::new(),
            add_space_prefix: false,
            byte_to_char,
            char_to_byte,
            bos_id,
            eos_id,
            eot_id: eot_id.or(eos_id),
            add_bos: bos_id.is_some(),
            max_digits: 3,
            special_tokens,
        })
    }

    #[inline]
    pub fn is_stop_token(&self, id: u32) -> bool {
        self.eos_id == Some(id) || self.eot_id == Some(id)
    }

    pub fn vocab_size(&self) -> usize {
        self.real_vocab_size
    }

    #[inline]
    fn token_str(&self, id: u32) -> Option<&str> {
        let idx = id as usize;
        let start = *self.vocab_offsets.get(idx)? as usize;
        let end = *self.vocab_offsets.get(idx + 1)? as usize;
        self.vocab_pool.get(start..end)
    }

    #[inline]
    fn lookup_id(&self, s: &str) -> Option<u32> {
        table_lookup(
            &self.vocab_table,
            self.vocab_mask,
            &self.vocab_pool,
            &self.vocab_offsets,
            s,
        )
    }

    /// Encodes `text` into token IDs (prepending BOS if `add_bos` is true).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos && let Some(bos) = self.bos_id {
            ids.push(bos);
        }

        if matches!(
            self.mode,
            TokenizerMode::Gemma4SpmBpe | TokenizerMode::LlamaSpm
        ) {
            self.encode_with_specials(text, &mut ids, true);
            return ids;
        }

        self.encode_with_specials(text, &mut ids, false);
        ids
    }

    fn encode_with_specials(&self, text: &str, ids: &mut Vec<u32>, is_spm: bool) {
        if self.special_tokens.is_empty() {
            if is_spm {
                self.encode_spm_span(text, ids, true);
            } else {
                self.encode_byte_bpe_span(text, ids);
            }
            return;
        }

        let mut first_byte_mask = [0u64; 4];
        for (st, _) in &self.special_tokens {
            let b = st.as_bytes()[0];
            first_byte_mask[(b >> 6) as usize] |= 1u64 << (b & 63);
        }

        let bytes = text.as_bytes();
        let mut span_start = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            let b = bytes[i];
            let maybe_special = (first_byte_mask[(b >> 6) as usize] & (1u64 << (b & 63))) != 0;
            if maybe_special && text.is_char_boundary(i) {
                let rem = &text[i..];
                let mut matched: Option<(usize, u32)> = None;
                for (st, st_id) in &self.special_tokens {
                    if rem.starts_with(st.as_str()) {
                        matched = Some((st.len(), *st_id));
                        break;
                    }
                }
                if let Some((st_len, st_id)) = matched {
                    if span_start < i {
                        let sub = &text[span_start..i];
                        if is_spm {
                            self.encode_spm_span(sub, ids, span_start == 0);
                        } else {
                            self.encode_byte_bpe_span(sub, ids);
                        }
                    }
                    ids.push(st_id);
                    i += st_len;
                    span_start = i;
                    continue;
                }
            }
            i += 1;
        }

        if span_start < text.len() {
            let sub = &text[span_start..];
            if is_spm {
                self.encode_spm_span(sub, ids, span_start == 0);
            } else {
                self.encode_byte_bpe_span(sub, ids);
            }
        }
    }

    fn encode_byte_bpe_span(&self, text: &str, ids: &mut Vec<u32>) {
        let mut encoded = String::with_capacity(64);
        let mut sym_ids: Vec<u32> = Vec::with_capacity(64);
        let mut char_buf = [0u8; 4];

        let chunks = match self.mode {
            TokenizerMode::Gpt4oByteBpe => pretokenize_gpt4o(text),
            TokenizerMode::DeepSeek3ByteBpe => pretokenize_deepseek3(text),
            _ => pretokenize_bpe(text, self.max_digits),
        };

        for chunk in chunks {
            encoded.clear();
            for &b in chunk.as_bytes() {
                encoded.push(self.byte_to_char[b as usize]);
            }

            if let Some(id) = self.lookup_id(&encoded)
                && (id as usize) < self.real_vocab_size
            {
                ids.push(id);
                continue;
            }

            sym_ids.clear();
            let mut all_found = true;
            for ch in encoded.chars() {
                let s = ch.encode_utf8(&mut char_buf);
                if let Some(id) = self.lookup_id(s) {
                    sym_ids.push(id);
                } else {
                    all_found = false;
                    break;
                }
            }
            if !all_found {
                continue;
            }

            self.apply_bpe_merges(&mut sym_ids);

            for &id in &sym_ids {
                if (id as usize) < self.real_vocab_size {
                    ids.push(id);
                }
            }
        }
    }

    fn apply_bpe_merges(&self, sym_ids: &mut Vec<u32>) {
        while sym_ids.len() >= 2 {
            let mut best_rank = u32::MAX;
            let mut best_idx = 0usize;
            let mut best_merged = 0u32;
            for i in 0..(sym_ids.len() - 1) {
                let pair_key = ((sym_ids[i] as u64) << 32) | (sym_ids[i + 1] as u64);
                if let Some(&(rank, merged_id)) = self.merges.get(&pair_key)
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_idx = i;
                    best_merged = merged_id;
                }
            }
            if best_rank == u32::MAX {
                break;
            }
            sym_ids[best_idx] = best_merged;
            sym_ids.remove(best_idx + 1);
        }
    }

    fn encode_spm_span(&self, text: &str, ids: &mut Vec<u32>, is_first_span: bool) {
        if text.is_empty() {
            return;
        }
        if self.mode == TokenizerMode::Gemma4SpmBpe {
            self.encode_gemma4_into(text, ids);
            return;
        }
        // SentencePiece BPE (`tokenizer.ggml.model == "llama"` for Gemma 2/3, Phi-3, Mistral, Llama-2):
        let mut escaped = String::with_capacity(text.len() + 4);
        if is_first_span && self.add_space_prefix && !text.starts_with(' ') {
            escaped.push('\u{2581}');
        }
        for ch in text.chars() {
            escaped.push(if ch == ' ' { '\u{2581}' } else { ch });
        }
        let mut sym_ids: Vec<u32> = Vec::with_capacity(escaped.len());
        let mut char_buf = [0u8; 4];
        for ch in escaped.chars() {
            let s = ch.encode_utf8(&mut char_buf);
            if let Some(id) = self.lookup_id(s)
                && (id as usize) < self.real_vocab_size
            {
                sym_ids.push(id);
            } else {
                for &b in s.as_bytes() {
                    let hex_tok = format!("<0x{b:02X}>");
                    if let Some(bid) = self.lookup_id(&hex_tok) {
                        sym_ids.push(bid);
                    }
                }
            }
        }
        if !self.merges.is_empty() {
            self.apply_bpe_merges(&mut sym_ids);
        } else if !self.scores.is_empty() {
            let mut merged_scratch = String::with_capacity(64);
            while sym_ids.len() >= 2 {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_idx = usize::MAX;
                let mut best_id = 0u32;
                for i in 0..(sym_ids.len() - 1) {
                    let Some(sa) = self.token_str(sym_ids[i]) else {
                        continue;
                    };
                    let Some(sb) = self.token_str(sym_ids[i + 1]) else {
                        continue;
                    };
                    merged_scratch.clear();
                    merged_scratch.push_str(sa);
                    merged_scratch.push_str(sb);
                    if let Some(mid) = self.lookup_id(&merged_scratch)
                        && (mid as usize) < self.scores.len()
                    {
                        let sc = self.scores[mid as usize];
                        if sc > best_score {
                            best_score = sc;
                            best_idx = i;
                            best_id = mid;
                        }
                    }
                }
                if best_idx == usize::MAX {
                    break;
                }
                sym_ids[best_idx] = best_id;
                sym_ids.remove(best_idx + 1);
            }
        }
        for &id in &sym_ids {
            if (id as usize) < self.real_vocab_size {
                ids.push(id);
            }
        }
    }

    fn encode_gemma4_into(&self, text: &str, ids: &mut Vec<u32>) {
        // Gemma 4 SPM-style BPE (`llama-vocab.cpp`: `LLAMA_VOCAB_PRE_TYPE_GEMMA4`):
        // 1. Split only on `[^\n]+|[\n]+`;
        // 2. Replace `' '` with `'▁'` (`U+2581`), raw UTF-8 without GPT-2 byte encoding;
        // 3. Direct vocab lookup for pure-newline runs, else UTF-8 chars with `<0xXX>` fallback + BPE merges.
        let mut escaped = String::with_capacity(128);
        let mut sym_ids: Vec<u32> = Vec::with_capacity(128);
        let mut char_buf = [0u8; 4];

        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            let is_nl = bytes[i] == b'\n';
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i] == b'\n') == is_nl {
                i += 1;
            }
            let chunk = &text[start..i];
            if is_nl
                && let Some(id) = self.lookup_id(chunk)
                && (id as usize) < self.real_vocab_size
            {
                ids.push(id);
                continue;
            }

            escaped.clear();
            for ch in chunk.chars() {
                if ch == ' ' {
                    escaped.push('\u{2581}');
                } else {
                    escaped.push(ch);
                }
            }

            sym_ids.clear();
            for ch in escaped.chars() {
                let s = ch.encode_utf8(&mut char_buf);
                if let Some(id) = self.lookup_id(s) {
                    sym_ids.push(id);
                } else {
                    for &b in s.as_bytes() {
                        let hex_tok = format!("<0x{b:02X}>");
                        if let Some(bid) = self.lookup_id(&hex_tok) {
                            sym_ids.push(bid);
                        }
                    }
                }
            }

            self.apply_bpe_merges(&mut sym_ids);
            for &id in &sym_ids {
                if (id as usize) < self.real_vocab_size {
                    ids.push(id);
                }
            }
        }
    }

    /// Decodes a single token ID into raw bytes.
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        if (id as usize) >= self.real_vocab_size {
            return Vec::new();
        }
        let Some(tok) = self.token_str(id) else {
            return Vec::new();
        };
        if matches!(
            self.mode,
            TokenizerMode::Gemma4SpmBpe | TokenizerMode::LlamaSpm
        ) {
            if tok.len() == 6
                && tok.starts_with("<0x")
                && tok.ends_with('>')
                && let Ok(b) = u8::from_str_radix(&tok[3..5], 16)
            {
                return vec![b];
            }
            let mut out = Vec::with_capacity(tok.len());
            let mut buf = [0u8; 4];
            for ch in tok.chars() {
                if ch == '\u{2581}' {
                    out.push(b' ');
                } else {
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
            return out;
        }
        let mut out = Vec::with_capacity(tok.len());
        for ch in tok.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                out.push(b);
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        out
    }

    /// Decodes a sequence of token IDs into a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decode_token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[inline]
fn table_lookup(table: &[u32], mask: usize, pool: &str, offsets: &[u32], key: &str) -> Option<u32> {
    let mut slot = (fnv1a_64(key.as_bytes()) as usize) & mask;
    loop {
        let id = table[slot];
        if id == EMPTY_SLOT {
            return None;
        }
        let idx = id as usize;
        let s0 = offsets[idx] as usize;
        let s1 = offsets[idx + 1] as usize;
        if &pool[s0..s1] == key {
            return Some(id);
        }
        slot = (slot + 1) & mask;
    }
}

#[inline]
fn table_upsert(
    table: &mut [u32],
    mask: usize,
    pool: &str,
    offsets: &[u32],
    key: &str,
    new_id: u32,
) {
    let mut slot = (fnv1a_64(key.as_bytes()) as usize) & mask;
    loop {
        let cur = table[slot];
        if cur == EMPTY_SLOT {
            table[slot] = new_id;
            return;
        }
        let idx = cur as usize;
        let s0 = offsets[idx] as usize;
        let s1 = offsets[idx + 1] as usize;
        if &pool[s0..s1] == key {
            table[slot] = new_id;
            return;
        }
        slot = (slot + 1) & mask;
    }
}

#[inline]
fn intern_or_lookup(
    table: &mut [u32],
    mask: usize,
    pool: &mut String,
    offsets: &mut Vec<u32>,
    key: &str,
) -> u32 {
    if let Some(id) = table_lookup(table, mask, pool, offsets, key) {
        return id;
    }
    // Only intern extra intermediate pieces while keeping load factor < 85%
    let new_id = (offsets.len() - 1) as u32;
    pool.push_str(key);
    offsets.push(pool.len() as u32);
    if offsets.len() < (table.len() * 7) / 8 {
        table_upsert(table, mask, pool, offsets, key, new_id);
    }
    new_id
}

/// Standard OpenAI GPT-2 `bytes_to_unicode()` bijection.
fn build_gpt2_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut is_direct = [false; 256];
    for b in b'!'..=b'~' {
        is_direct[b as usize] = true;
    }
    for b in 0xA1u8..=0xACu8 {
        is_direct[b as usize] = true;
    }
    for b in 0xAEu8..=0xFFu8 {
        is_direct[b as usize] = true;
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0u16..256 {
        let ch = if is_direct[b as usize] {
            char::from_u32(b as u32).unwrap_or('\0')
        } else {
            let c = char::from_u32(256 + n).unwrap_or('\0');
            n += 1;
            c
        };
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

/// Dependency-free pre-tokenizer matching Llama-3 / SmolLM / Qwen2-3.5 BPE regex:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,max_digits}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn pretokenize_bpe(text: &str, max_digits: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        let rest = &text[i..];

        // 1. English contractions: 's, 't, 're, 've, 'm, 'll, 'd (case-insensitive)
        if rest.starts_with('\'') {
            let lower = rest.get(1..3).unwrap_or("").to_ascii_lowercase();
            let lower1 = rest.get(1..2).unwrap_or("").to_ascii_lowercase();
            if matches!(lower.as_str(), "re" | "ve" | "ll") {
                chunks.push(&text[i..i + 3]);
                i += 3;
                continue;
            }
            if matches!(lower1.as_str(), "s" | "t" | "m" | "d") {
                chunks.push(&text[i..i + 2]);
                i += 2;
                continue;
            }
        }

        let mut chars = rest.char_indices();
        let Some((_, c0)) = chars.next() else {
            break;
        };

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if c0.is_alphabetic() {
            let mut end = c0.len_utf8();
            for (idx, c) in chars {
                if c.is_alphabetic() {
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }
        if c0 != '\r'
            && c0 != '\n'
            && !c0.is_numeric()
            && let Some((idx1, c1)) = chars.clone().next()
            && c1.is_alphabetic()
        {
            let mut end = idx1 + c1.len_utf8();
            for (idx, c) in chars.skip(1) {
                if c.is_alphabetic() {
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 3. \p{N}{1,max_digits}
        if c0.is_numeric() {
            let mut end = c0.len_utf8();
            let mut count = 1usize;
            for (idx, c) in chars {
                if c.is_numeric() && count < max_digits {
                    end = idx + c.len_utf8();
                    count += 1;
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]*
        let is_punct = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
        if is_punct(c0) || (c0 == ' ' && chars.clone().next().is_some_and(|(_, c1)| is_punct(c1))) {
            let mut end = c0.len_utf8();
            let mut iter = chars;
            if c0 == ' '
                && let Some((idx1, c1)) = iter.next()
            {
                end = idx1 + c1.len_utf8();
            }
            let mut in_nl = false;
            for (idx, c) in iter {
                if !in_nl && is_punct(c) {
                    end = idx + c.len_utf8();
                } else if c == '\r' || c == '\n' {
                    in_nl = true;
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 5. Whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
        if c0.is_whitespace() {
            let mut end = c0.len_utf8();
            let mut last_nl_end = if c0 == '\r' || c0 == '\n' {
                Some(end)
            } else {
                None
            };
            let mut last_ws_start = 0usize;
            let mut hit_non_ws = false;

            for (idx, c) in chars {
                if c.is_whitespace() {
                    last_ws_start = idx;
                    end = idx + c.len_utf8();
                    if c == '\r' || c == '\n' {
                        last_nl_end = Some(end);
                    }
                } else {
                    hit_non_ws = true;
                    break;
                }
            }

            if let Some(nl_end) = last_nl_end {
                chunks.push(&text[i..i + nl_end]);
                i += nl_end;
            } else if hit_non_ws && last_ws_start > 0 {
                // \s+(?!\S): leave the last space for the following word/punct token
                chunks.push(&text[i..i + last_ws_start]);
                i += last_ws_start;
            } else {
                chunks.push(&text[i..i + end]);
                i += end;
            }
            continue;
        }

        // Fallback single char
        chunks.push(&text[i..i + c0.len_utf8()]);
        i += c0.len_utf8();
    }

    chunks
}

#[inline]
fn is_cjk_or_kana(c: char) -> bool {
    let u = c as u32;
    (0x4E00..=0x9FA5).contains(&u)
        || (0x3040..=0x309F).contains(&u)
        || (0x30A0..=0x30FF).contains(&u)
}

/// Pre-tokenizer matching `Hy4-Flash` / `DeepSeek-V3` three-stage isolated splits:
/// 1. `\p{N}{1,3}`
/// 2. `[一-龥぀-ゟ゠-ヿ]+`
/// 3. `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn pretokenize_deepseek3(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let len = text.len();
    let mut i = 0usize;

    let is_ascii_punct = |c: char| c.is_ascii_punctuation();
    let is_punct_or_sym =
        |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() && !is_cjk_or_kana(c);
    let is_non_cjk_letter = |c: char| c.is_alphabetic() && !is_cjk_or_kana(c);

    while i < len {
        let rest = &text[i..];
        let mut chars = rest.char_indices();
        let Some((_, c0)) = chars.next() else {
            break;
        };

        // Stage 1: `\p{N}{1,3}`
        if c0.is_numeric() {
            let mut end = c0.len_utf8();
            let mut cnt = 1usize;
            for (idx, c) in chars {
                if c.is_numeric() && cnt < 3 {
                    end = idx + c.len_utf8();
                    cnt += 1;
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // Stage 2: `[一-龥぀-ゟ゠-ヿ]+`
        if is_cjk_or_kana(c0) {
            let mut end = c0.len_utf8();
            for (idx, c) in chars {
                if is_cjk_or_kana(c) {
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // Stage 3a: `[ascii_punct][A-Za-z]+`
        if is_ascii_punct(c0)
            && let Some((idx1, c1)) = chars.clone().next()
            && c1.is_ascii_alphabetic()
        {
            let mut end = idx1 + c1.len_utf8();
            for (idx, c) in chars.skip(1) {
                if c.is_ascii_alphabetic() {
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // Stage 3b: `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` (excluding CJK/Kana isolated in Stage 2)
        if is_non_cjk_letter(c0) {
            let mut end = c0.len_utf8();
            for (idx, c) in chars {
                if is_non_cjk_letter(c) {
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }
        if c0 != '\r'
            && c0 != '\n'
            && !is_punct_or_sym(c0)
            && !c0.is_numeric()
            && let Some((idx1, c1)) = chars.clone().next()
            && is_non_cjk_letter(c1)
        {
            let mut end = idx1 + c1.len_utf8();
            for (idx, c) in chars.skip(1) {
                if is_non_cjk_letter(c) {
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // Stage 3c: ` ?[\p{P}\p{S}]+[\r\n]*`
        if is_punct_or_sym(c0)
            || (c0 == ' '
                && chars
                    .clone()
                    .next()
                    .is_some_and(|(_, c1)| is_punct_or_sym(c1)))
        {
            let mut end = c0.len_utf8();
            let mut iter = chars;
            if c0 == ' '
                && let Some((idx1, c1)) = iter.next()
            {
                end = idx1 + c1.len_utf8();
            }
            let mut in_nl = false;
            for (idx, c) in iter {
                if !in_nl && is_punct_or_sym(c) {
                    end = idx + c.len_utf8();
                } else if c == '\r' || c == '\n' {
                    in_nl = true;
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // Stage 3d: `\s*[\r\n]+ | \s+(?!\S) | \s+`
        if c0.is_whitespace() {
            let mut end = c0.len_utf8();
            let mut last_nl_end = if c0 == '\r' || c0 == '\n' {
                Some(end)
            } else {
                None
            };
            let mut last_ws_start = 0usize;
            let mut hit_non_ws = false;
            for (idx, c) in chars {
                if c.is_whitespace() {
                    last_ws_start = idx;
                    end = idx + c.len_utf8();
                    if c == '\r' || c == '\n' {
                        last_nl_end = Some(end);
                    }
                } else {
                    hit_non_ws = true;
                    break;
                }
            }
            if let Some(nl_end) = last_nl_end {
                chunks.push(&text[i..i + nl_end]);
                i += nl_end;
            } else if hit_non_ws && last_ws_start > 0 {
                chunks.push(&text[i..i + last_ws_start]);
                i += last_ws_start;
            } else {
                chunks.push(&text[i..i + end]);
                i += end;
            }
            continue;
        }

        chunks.push(&text[i..i + c0.len_utf8()]);
        i += c0.len_utf8();
    }

    chunks
}

struct ParsedHfBpe {
    tokens: Vec<String>,
    merges: Vec<(String, String)>,
    special_tokens: Vec<(String, u32)>,
    is_deepseek3: bool,
}

struct JsonCursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonCursor<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, ch: u8) -> Result<(), TokenizerError> {
        self.skip_ws();
        if self.b.get(self.i).copied() == Some(ch) {
            self.i += 1;
            Ok(())
        } else {
            Err(TokenizerError::InvalidJson(format!(
                "expected '{}' at byte {}",
                ch as char, self.i
            )))
        }
    }

    fn parse_str(&mut self) -> Result<String, TokenizerError> {
        self.expect(b'"')?;
        let mut out = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b'"' {
                self.i += 1;
                return Ok(out);
            }
            if c != b'\\' {
                let start = self.i;
                self.i += 1;
                while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                    self.i += 1;
                }
                let s = core::str::from_utf8(&self.b[start..self.i])
                    .map_err(|e| TokenizerError::InvalidJson(e.to_string()))?;
                out.push_str(s);
                continue;
            }
            self.i += 1;
            let esc = *self
                .b
                .get(self.i)
                .ok_or_else(|| TokenizerError::InvalidJson("EOF in string escape".to_string()))?;
            self.i += 1;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000C}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let u1 = self.parse_hex4()?;
                    if (0xD800..=0xDBFF).contains(&u1)
                        && self.b.get(self.i..self.i + 2) == Some(b"\\u")
                    {
                        self.i += 2;
                        let u2 = self.parse_hex4()?;
                        let cp = 0x10000 + (((u1 as u32 - 0xD800) << 10) | (u2 as u32 - 0xDC00));
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                        }
                    } else if let Some(ch) = char::from_u32(u1 as u32) {
                        out.push(ch);
                    }
                }
                other => {
                    return Err(TokenizerError::InvalidJson(format!(
                        "bad escape \\{}",
                        other as char
                    )));
                }
            }
        }
        Err(TokenizerError::InvalidJson(
            "unterminated string".to_string(),
        ))
    }

    fn parse_hex4(&mut self) -> Result<u16, TokenizerError> {
        let slice = self
            .b
            .get(self.i..self.i + 4)
            .ok_or_else(|| TokenizerError::InvalidJson("short \\u escape".to_string()))?;
        self.i += 4;
        let s =
            core::str::from_utf8(slice).map_err(|e| TokenizerError::InvalidJson(e.to_string()))?;
        u16::from_str_radix(s, 16).map_err(|e| TokenizerError::InvalidJson(e.to_string()))
    }

    fn parse_u32(&mut self) -> Result<u32, TokenizerError> {
        self.skip_ws();
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        let s = core::str::from_utf8(&self.b[start..self.i])
            .map_err(|e| TokenizerError::InvalidJson(e.to_string()))?;
        s.parse::<u32>()
            .map_err(|e| TokenizerError::InvalidJson(e.to_string()))
    }

    fn skip_value(&mut self) -> Result<(), TokenizerError> {
        let Some(c) = self.peek() else {
            return Err(TokenizerError::InvalidJson("unexpected EOF".to_string()));
        };
        match c {
            b'"' => {
                let _ = self.parse_str()?;
                Ok(())
            }
            b'{' => {
                self.i += 1;
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    let _ = self.parse_str()?;
                    self.expect(b':')?;
                    self.skip_value()?;
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        self.expect(b'}')?;
                        return Ok(());
                    }
                }
            }
            b'[' => {
                self.i += 1;
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        self.expect(b']')?;
                        return Ok(());
                    }
                }
            }
            _ => {
                while self.i < self.b.len()
                    && !matches!(
                        self.b[self.i],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n'
                    )
                {
                    self.i += 1;
                }
                Ok(())
            }
        }
    }
}

fn parse_hf_bpe_json(json_str: &str) -> Result<ParsedHfBpe, TokenizerError> {
    let mut cur = JsonCursor {
        b: json_str.as_bytes(),
        i: 0,
    };
    let is_deepseek3 = json_str.contains("一-龥");
    let mut vocab_pairs: Vec<(u32, String)> = Vec::new();
    let mut added_list: Vec<(u32, String)> = Vec::new();
    let mut merges: Vec<(String, String)> = Vec::new();

    cur.expect(b'{')?;
    if cur.peek() != Some(b'}') {
        loop {
            let key = cur.parse_str()?;
            cur.expect(b':')?;
            match key.as_str() {
                "added_tokens" => {
                    if cur.peek() == Some(b'[') {
                        cur.i += 1;
                        if cur.peek() != Some(b']') {
                            loop {
                                cur.expect(b'{')?;
                                let mut id = None;
                                let mut content = None;
                                if cur.peek() != Some(b'}') {
                                    loop {
                                        let fk = cur.parse_str()?;
                                        cur.expect(b':')?;
                                        match fk.as_str() {
                                            "id" => id = Some(cur.parse_u32()?),
                                            "content" => content = Some(cur.parse_str()?),
                                            _ => cur.skip_value()?,
                                        }
                                        if cur.peek() == Some(b',') {
                                            cur.i += 1;
                                        } else {
                                            cur.expect(b'}')?;
                                            break;
                                        }
                                    }
                                } else {
                                    cur.i += 1;
                                }
                                if let (Some(tid), Some(tok_s)) = (id, content) {
                                    added_list.push((tid, tok_s));
                                }
                                if cur.peek() == Some(b',') {
                                    cur.i += 1;
                                } else {
                                    cur.expect(b']')?;
                                    break;
                                }
                            }
                        } else {
                            cur.i += 1;
                        }
                    } else {
                        cur.skip_value()?;
                    }
                }
                "model" => {
                    cur.expect(b'{')?;
                    if cur.peek() != Some(b'}') {
                        loop {
                            let mk = cur.parse_str()?;
                            cur.expect(b':')?;
                            match mk.as_str() {
                                "vocab" => {
                                    cur.expect(b'{')?;
                                    if cur.peek() != Some(b'}') {
                                        loop {
                                            let ts = cur.parse_str()?;
                                            cur.expect(b':')?;
                                            let tid = cur.parse_u32()?;
                                            vocab_pairs.push((tid, ts));
                                            if cur.peek() == Some(b',') {
                                                cur.i += 1;
                                            } else {
                                                cur.expect(b'}')?;
                                                break;
                                            }
                                        }
                                    } else {
                                        cur.i += 1;
                                    }
                                }
                                "merges" => {
                                    cur.expect(b'[')?;
                                    if cur.peek() != Some(b']') {
                                        loop {
                                            if cur.peek() == Some(b'[') {
                                                cur.i += 1;
                                                let a = cur.parse_str()?;
                                                cur.expect(b',')?;
                                                let b = cur.parse_str()?;
                                                cur.expect(b']')?;
                                                merges.push((a, b));
                                            } else {
                                                let line = cur.parse_str()?;
                                                if let Some((a, b)) = line.split_once(' ') {
                                                    merges.push((a.to_string(), b.to_string()));
                                                }
                                            }
                                            if cur.peek() == Some(b',') {
                                                cur.i += 1;
                                            } else {
                                                cur.expect(b']')?;
                                                break;
                                            }
                                        }
                                    } else {
                                        cur.i += 1;
                                    }
                                }
                                _ => cur.skip_value()?,
                            }
                            if cur.peek() == Some(b',') {
                                cur.i += 1;
                            } else {
                                cur.expect(b'}')?;
                                break;
                            }
                        }
                    } else {
                        cur.i += 1;
                    }
                }
                _ => cur.skip_value()?,
            }
            if cur.peek() == Some(b',') {
                cur.i += 1;
            } else {
                cur.expect(b'}')?;
                break;
            }
        }
    }

    let max_id = vocab_pairs
        .iter()
        .map(|(i, _)| *i)
        .chain(added_list.iter().map(|(i, _)| *i))
        .max()
        .ok_or_else(|| TokenizerError::InvalidJson("empty vocabulary".to_string()))?;
    let mut tokens = vec![String::new(); (max_id as usize) + 1];
    for (id, s) in vocab_pairs {
        tokens[id as usize] = s;
    }
    let mut special_tokens = Vec::new();
    for (id, s) in added_list {
        if s.len() >= 2 {
            special_tokens.push((s.clone(), id));
        }
        tokens[id as usize] = s;
    }

    Ok(ParsedHfBpe {
        tokens,
        merges,
        special_tokens,
        is_deepseek3,
    })
}

fn match_contraction_suffix(s: &str) -> usize {
    if !s.starts_with('\'') {
        return 0;
    }
    let lower2 = s.get(1..3).unwrap_or("").to_ascii_lowercase();
    if matches!(lower2.as_str(), "re" | "ve" | "ll") {
        return 3;
    }
    let lower1 = s.get(1..2).unwrap_or("").to_ascii_lowercase();
    if matches!(lower1.as_str(), "s" | "t" | "m" | "d") {
        return 2;
    }
    0
}

/// Dependency-free pre-tokenizer matching `LLAMA_VOCAB_PRE_TYPE_GPT4O` (`inkling2` / `o200k_base`):
/// 1. `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
/// 2. `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`
/// 3. `\p{N}{1,3}`
/// 4. `' '?[^\s\p{L}\p{N}]+[\r\n/]*`
/// 5. `\s*[\r\n]+ | \s+(?!\S) | \s+`
fn pretokenize_gpt4o(text: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let len = text.len();
    let mut i = 0usize;

    while i < len {
        let rest = &text[i..];
        let mut chars = rest.char_indices();
        let Some((_, c0)) = chars.next() else {
            break;
        };

        // 1 & 2: Optional non-CRLF/non-alnum prefix + camelCase-aware letter run + optional contraction
        let letter_start = if c0.is_alphabetic() {
            Some((0usize, c0))
        } else if c0 != '\r'
            && c0 != '\n'
            && !c0.is_numeric()
            && let Some((idx1, c1)) = chars.clone().next()
            && c1.is_alphabetic()
        {
            Some((idx1, c1))
        } else {
            None
        };

        if let Some((l_off, _)) = letter_start {
            // Branch 1: `[upper]*[lower]+` vs Branch 2: `[upper]+[lower]*`
            // In `llama-vocab.cpp:460`, `[^a-z]` acts as upper/caseless and `[^A-Z]` as lower/caseless.
            let mut end1 = 0usize;
            {
                let mut seen_lower = false;
                for (off, ch) in rest[l_off..].char_indices() {
                    if !ch.is_alphabetic() {
                        break;
                    }
                    let is_ascii_upper = ch.is_ascii_uppercase();
                    if !seen_lower {
                        if !is_ascii_upper {
                            seen_lower = true;
                            end1 = l_off + off + ch.len_utf8();
                        }
                    } else if !is_ascii_upper {
                        end1 = l_off + off + ch.len_utf8();
                    } else {
                        break;
                    }
                }
            }
            let end2 = {
                let mut e = 0usize;
                let mut in_lower_tail = false;
                let mut has_upper = false;
                for (off, ch) in rest[l_off..].char_indices() {
                    if !ch.is_alphabetic() {
                        break;
                    }
                    let is_ascii_lower = ch.is_ascii_lowercase();
                    let is_ascii_upper = ch.is_ascii_uppercase();
                    if !in_lower_tail {
                        if !is_ascii_lower {
                            has_upper = true;
                            e = l_off + off + ch.len_utf8();
                        } else if has_upper {
                            in_lower_tail = true;
                            e = l_off + off + ch.len_utf8();
                        } else {
                            break;
                        }
                    } else if !is_ascii_upper {
                        e = l_off + off + ch.len_utf8();
                    } else {
                        break;
                    }
                }
                e
            };

            let mut end = if end1 > 0 { end1 } else { end2 };
            if end > 0 {
                end += match_contraction_suffix(&rest[end..]);
                chunks.push(&text[i..i + end]);
                i += end;
                continue;
            }
        }

        // 3. \p{N}{1,3}
        if c0.is_numeric() {
            let mut end = c0.len_utf8();
            let mut count = 1usize;
            for (idx, c) in chars {
                if c.is_numeric() && count < 3 {
                    end = idx + c.len_utf8();
                    count += 1;
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 4. ' '?[^\s\p{L}\p{N}]+[\r\n/]*
        let is_punct = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
        if is_punct(c0) || (c0 == ' ' && chars.clone().next().is_some_and(|(_, c1)| is_punct(c1))) {
            let mut end = c0.len_utf8();
            let mut iter = chars;
            if c0 == ' '
                && let Some((idx1, c1)) = iter.next()
            {
                end = idx1 + c1.len_utf8();
            }
            let mut in_tail = false;
            for (idx, c) in iter {
                if !in_tail && is_punct(c) {
                    end = idx + c.len_utf8();
                } else if c == '\r' || c == '\n' || c == '/' {
                    in_tail = true;
                    end = idx + c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 5. Whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
        if c0.is_whitespace() {
            let mut end = c0.len_utf8();
            let mut last_nl_end = if c0 == '\r' || c0 == '\n' {
                Some(end)
            } else {
                None
            };
            let mut last_ws_start = 0usize;
            let mut hit_non_ws = false;

            for (idx, c) in chars {
                if c.is_whitespace() {
                    last_ws_start = idx;
                    end = idx + c.len_utf8();
                    if c == '\r' || c == '\n' {
                        last_nl_end = Some(end);
                    }
                } else {
                    hit_non_ws = true;
                    break;
                }
            }

            if let Some(nl_end) = last_nl_end {
                chunks.push(&text[i..i + nl_end]);
                i += nl_end;
            } else if hit_non_ws && last_ws_start > 0 {
                chunks.push(&text[i..i + last_ws_start]);
                i += last_ws_start;
            } else {
                chunks.push(&text[i..i + end]);
                i += end;
            }
            continue;
        }

        chunks.push(&text[i..i + c0.len_utf8()]);
        i += c0.len_utf8();
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenize_digit_splitting_qwen_vs_llama() {
        assert_eq!(
            pretokenize_bpe("Year 2026!", 1),
            vec!["Year", " ", "2", "0", "2", "6", "!"]
        );
        assert_eq!(
            pretokenize_bpe("Year 2026!", 3),
            vec!["Year", " ", "202", "6", "!"]
        );
    }

    #[test]
    fn gemma4_spm_bpe_loads_and_roundtrips() {
        let model_path = std::path::Path::new("../../models/gemma-4-E2B-it-Q4_0.gguf");
        if !model_path.exists() {
            return;
        }
        let gguf = GgufFile::open(model_path).expect("open Gemma 4 E2B");
        let tok = Tokenizer::from_gguf(&gguf).expect("load Gemma 4 tokenizer");
        assert_eq!(tok.vocab_size(), 262_144);
        assert!(tok.add_bos);
        assert_eq!(tok.bos_id, Some(2));
        let text =
            "Gemma 4 uses Per-Layer Embeddings (PLE) & sliding window: 2 + 2 = 4!\nNext line.";
        let ids = tok.encode(text, false);
        assert!(ids.len() > 5);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn inkling2_200k_vocab_special_tokens_and_gpt4o_roundtrip() {
        let model_path =
            std::path::Path::new("../../models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf");
        if !model_path.exists() {
            return;
        }
        let gguf = GgufFile::open(model_path).expect("open dspark Inkling-2-Mini");
        let tok = Tokenizer::from_gguf(&gguf).expect("load inkling2 tokenizer");
        assert_eq!(tok.vocab_size(), 200_058);
        assert_eq!(tok.bos_id, None);
        assert!(!tok.add_bos);
        assert_eq!(tok.eos_id, Some(200_006));
        assert_eq!(tok.eot_id, Some(200_006));

        // Verify exact special control token partitioning (`<|begin_of_text|>` == 200028)
        let prompt = "<|begin_of_text|>HelloWorld don't 2026 中文测试!<|content_invoke_tool_json|>";
        let ids = tok.encode(prompt, true);
        assert_eq!(ids.first().copied(), Some(200_028));
        assert_eq!(ids.last().copied(), Some(200_049));
        let inner = &ids[1..ids.len() - 1];
        assert_eq!(tok.decode(inner), "HelloWorld don't 2026 中文测试!");
    }
}
