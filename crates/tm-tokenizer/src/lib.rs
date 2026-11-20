#![forbid(unsafe_code)]

//! Lightweight pure-Rust GPT-2 byte-level BPE tokenizer loaded directly from GGUF metadata.

use std::collections::HashMap;

use thiserror::Error;
use tm_gguf::GgufFile;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("GGUF missing tokenizer metadata key '{0}'")]
    MissingKey(&'static str),
    #[error("unsupported tokenizer model '{0}' (currently supported: 'gpt2')")]
    UnsupportedModel(String),
}

/// GPT-2 byte-level BPE tokenizer (covers SmolLM2, Llama 3.x, Qwen2/3/3.5).
pub struct Tokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub add_bos: bool,
    max_digits: usize,
}

impl Tokenizer {
    /// Loads tokenizer tables directly from an opened `GgufFile`.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = gguf
            .get_meta("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt2");
        if model != "gpt2" {
            return Err(TokenizerError::UnsupportedModel(model.to_string()));
        }

        let tokens_val = gguf
            .get_meta("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.tokens"))?;
        let token_strs = tokens_val
            .as_string_array()
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.tokens"))?;

        let mut vocab = Vec::with_capacity(token_strs.len());
        let mut token_to_id = HashMap::with_capacity(token_strs.len());
        for (id, s) in token_strs.into_iter().enumerate() {
            let owned = s.to_string();
            token_to_id.insert(owned.clone(), id as u32);
            vocab.push(owned);
        }

        let mut merges = HashMap::new();
        if let Some(mval) = gguf.get_meta("tokenizer.ggml.merges")
            && let Some(mlist) = mval.as_string_array()
        {
            merges.reserve(mlist.len());
            for (rank, line) in mlist.into_iter().enumerate() {
                if let Some((a, b)) = line.split_once(' ') {
                    merges.insert((a.to_string(), b.to_string()), rank);
                }
            }
        }

        let bos_id = gguf
            .get_meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_id = gguf
            .get_meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let add_bos = gguf
            .get_meta("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let pre = gguf
            .get_meta("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // Qwen2 / Qwen3 / Qwen3.5 BPE regex uses single-digit `\p{N}` instead of `\p{N}{1,3}`.
        let max_digits = if pre.starts_with("qwen") { 1 } else { 3 };

        let (byte_to_char, char_to_byte) = build_gpt2_byte_maps();

        Ok(Self {
            vocab,
            token_to_id,
            merges,
            byte_to_char,
            char_to_byte,
            bos_id,
            eos_id,
            add_bos,
            max_digits,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encodes `text` into token IDs (prepending BOS if `add_bos_override` or `self.add_bos` is true).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos && let Some(bos) = self.bos_id {
            ids.push(bos);
        }

        for chunk in pretokenize_bpe(text, self.max_digits) {
            let encoded: String = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize])
                .collect();

            if let Some(&id) = self.token_to_id.get(&encoded) {
                ids.push(id);
                continue;
            }

            let mut symbols: Vec<String> = encoded.chars().map(|c| c.to_string()).collect();
            while symbols.len() >= 2 {
                let mut best_rank = usize::MAX;
                let mut best_idx = None;
                for i in 0..(symbols.len() - 1) {
                    // Look up merge rank without allocating when possible
                    if let Some(&rank) = self
                        .merges
                        .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                        && rank < best_rank
                    {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
                let Some(idx) = best_idx else {
                    break;
                };
                let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                symbols[idx] = merged;
                symbols.remove(idx + 1);
            }

            for sym in symbols {
                if let Some(&id) = self.token_to_id.get(&sym) {
                    ids.push(id);
                }
            }
        }

        ids
    }

    /// Decodes a single token ID into raw bytes (exact GPT-2 byte-level decoding).
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.vocab.get(id as usize) else {
            return Vec::new();
        };
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
