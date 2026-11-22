//! SFT & causal LM dataset loader (`TMSFT` v1 pre-tokenized binary, JSONL, and plain `.txt`).
//!
//! `loss_mask` convention (1-to-1 aligned with `tokens`):
//! - For causal next-token prediction `tokens[t] -> tokens[t + 1]` (`t in 0..tokens.len() - 1`),
//!   `loss_mask[t + 1] != 0` marks `tokens[t + 1]` as a supervised completion token, while
//!   `loss_mask[t + 1] == 0` marks an unsupervised prompt/chat-template token (`label = -100` in HF).

use std::fs;
use std::io::Write;
use std::path::Path;

use tm_tokenizer::Tokenizer;

use crate::TrainError;

pub const TMSFT_MAGIC: &[u8; 4] = b"TMSF";
pub const TMSFT_VERSION: u32 = 1;

/// One training sequence of length `2..=max_seq_len` with token-aligned `loss_mask`.
#[derive(Debug, Clone)]
pub struct TrainSample {
    pub tokens: Vec<u32>,
    pub loss_mask: Vec<u8>,
}

impl TrainSample {
    /// Number of supervised target tokens (`loss_mask[1..] != 0`).
    pub fn supervised_count(&self) -> usize {
        if self.tokens.len() < 2 {
            return 0;
        }
        self.loss_mask[1..].iter().filter(|&&m| m != 0).count()
    }
}

/// Loaded collection of training samples ready for deterministic epoch/step cycling.
#[derive(Debug, Clone)]
pub struct TrainDataset {
    pub samples: Vec<TrainSample>,
    pub total_tokens: usize,
    pub supervised_tokens: usize,
}

impl TrainDataset {
    /// Loads a dataset from `.tmsft`/`.bin` (`TMSF` magic), `.jsonl`, or plain `.txt`.
    pub fn load<P: AsRef<Path>>(
        path: P,
        tokenizer: Option<&Tokenizer>,
        max_seq_len: usize,
    ) -> Result<Self, TrainError> {
        let p = path.as_ref();
        let path_str = p.display().to_string();
        if max_seq_len < 2 {
            return Err(TrainError::InvalidConfig(format!(
                "max_seq_len must be >= 2, got {max_seq_len}"
            )));
        }
        let raw = fs::read(p).map_err(|e| TrainError::Io {
            path: path_str.clone(),
            message: e.to_string(),
        })?;
        if raw.len() >= 4 && &raw[..4] == TMSFT_MAGIC {
            return Self::from_tmsft_bytes(&path_str, &raw, max_seq_len);
        }

        let text = String::from_utf8(raw).map_err(|e| TrainError::Dataset {
            path: path_str.clone(),
            detail: format!("not a valid TMSFT binary or UTF-8 text file: {e}"),
        })?;

        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let looks_jsonl = ext == "jsonl"
            || text
                .lines()
                .find(|l| !l.trim().is_empty())
                .is_some_and(|l| l.trim_start().starts_with('{'));

        if looks_jsonl {
            Self::from_jsonl(&path_str, &text, tokenizer, max_seq_len)
        } else {
            let tok = tokenizer.ok_or_else(|| TrainError::Dataset {
                path: path_str.clone(),
                detail: "plain-text dataset requires a GGUF Tokenizer".to_string(),
            })?;
            Self::from_plain_text(&path_str, &text, tok, max_seq_len)
        }
    }

    pub fn from_samples(path_str: &str, mut samples: Vec<TrainSample>) -> Result<Self, TrainError> {
        samples.retain(|s| s.tokens.len() >= 2 && s.supervised_count() > 0);
        if samples.is_empty() {
            return Err(TrainError::Dataset {
                path: path_str.to_string(),
                detail: "dataset contains 0 valid supervised sequences (len >= 2)".to_string(),
            });
        }
        let total_tokens = samples.iter().map(|s| s.tokens.len()).sum();
        let supervised_tokens = samples.iter().map(|s| s.supervised_count()).sum();
        Ok(Self {
            samples,
            total_tokens,
            supervised_tokens,
        })
    }

    fn from_tmsft_bytes(
        path_str: &str,
        raw: &[u8],
        max_seq_len: usize,
    ) -> Result<Self, TrainError> {
        if raw.len() < 12 {
            return Err(TrainError::Dataset {
                path: path_str.to_string(),
                detail: "truncated TMSFT header".to_string(),
            });
        }
        let ver = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
        if ver != TMSFT_VERSION {
            return Err(TrainError::Dataset {
                path: path_str.to_string(),
                detail: format!("unsupported TMSFT version {ver}"),
            });
        }
        let n_samples = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
        let mut off = 12usize;
        let mut samples = Vec::with_capacity(n_samples);

        for idx in 0..n_samples {
            if off + 4 > raw.len() {
                return Err(TrainError::Dataset {
                    path: path_str.to_string(),
                    detail: format!("truncated TMSFT record {idx}"),
                });
            }
            let len =
                u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as usize;
            off += 4;
            let need = len.checked_mul(5).ok_or_else(|| TrainError::Dataset {
                path: path_str.to_string(),
                detail: "TMSFT sample length overflow".to_string(),
            })?;
            if off + need > raw.len() {
                return Err(TrainError::Dataset {
                    path: path_str.to_string(),
                    detail: format!("truncated TMSFT payload at record {idx}"),
                });
            }
            let mut toks = Vec::with_capacity(len);
            for i in 0..len {
                let p = off + i * 4;
                toks.push(u32::from_le_bytes([
                    raw[p],
                    raw[p + 1],
                    raw[p + 2],
                    raw[p + 3],
                ]));
            }
            off += len * 4;
            let mask = raw[off..off + len].to_vec();
            off += len;

            let keep = len.min(max_seq_len);
            if keep >= 2 {
                samples.push(TrainSample {
                    tokens: toks[..keep].to_vec(),
                    loss_mask: mask[..keep].to_vec(),
                });
            }
        }
        Self::from_samples(path_str, samples)
    }

    fn from_plain_text(
        path_str: &str,
        text: &str,
        tok: &Tokenizer,
        max_seq_len: usize,
    ) -> Result<Self, TrainError> {
        let ids = tok.encode(text, tok.add_bos);
        if ids.len() < 2 {
            return Err(TrainError::Dataset {
                path: path_str.to_string(),
                detail: format!("text encoded to {} tokens (need >= 2)", ids.len()),
            });
        }
        let mut samples = Vec::new();
        let stride = (max_seq_len - 1).max(1);
        let mut start = 0usize;
        while start + 1 < ids.len() {
            let end = (start + max_seq_len).min(ids.len());
            let slice = ids[start..end].to_vec();
            let mut mask = vec![1u8; slice.len()];
            mask[0] = 0;
            samples.push(TrainSample {
                tokens: slice,
                loss_mask: mask,
            });
            if end == ids.len() {
                break;
            }
            start += stride;
        }
        Self::from_samples(path_str, samples)
    }

    fn from_jsonl(
        path_str: &str,
        text: &str,
        tokenizer: Option<&Tokenizer>,
        max_seq_len: usize,
    ) -> Result<Self, TrainError> {
        let mut samples = Vec::new();
        for (line_no, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            let obj = parse_flat_json_line(line).map_err(|detail| TrainError::Dataset {
                path: path_str.to_string(),
                detail: format!("line {}: {detail}", line_no + 1),
            })?;

            if let Some(toks) = obj.int_arrays.get("tokens") {
                if toks.len() < 2 {
                    continue;
                }
                let mut mask: Vec<u8> = if let Some(m) = obj.int_arrays.get("loss_mask") {
                    if m.len() != toks.len() {
                        return Err(TrainError::Dataset {
                            path: path_str.to_string(),
                            detail: format!(
                                "line {}: loss_mask len {} != tokens len {}",
                                line_no + 1,
                                m.len(),
                                toks.len()
                            ),
                        });
                    }
                    m.iter().map(|&v| u8::from(v != 0)).collect()
                } else {
                    let mut m = vec![1u8; toks.len()];
                    m[0] = 0;
                    m
                };
                mask[0] = 0;
                let keep = toks.len().min(max_seq_len);
                let toks_u32: Vec<u32> = toks[..keep].iter().map(|&v| v as u32).collect();
                samples.push(TrainSample {
                    tokens: toks_u32,
                    loss_mask: mask[..keep].to_vec(),
                });
                continue;
            }

            let tok = tokenizer.ok_or_else(|| TrainError::Dataset {
                path: path_str.to_string(),
                detail: format!(
                    "line {}: text/prompt JSONL requires a GGUF Tokenizer",
                    line_no + 1
                ),
            })?;

            if let (Some(prompt), Some(comp)) =
                (obj.strings.get("prompt"), obj.strings.get("completion"))
            {
                let mut p_ids = tok.encode(prompt, tok.add_bos);
                let mut c_ids = tok.encode(comp, false);
                if let Some(eos) = tok.eos_id
                    && c_ids.last().copied() != Some(eos)
                {
                    c_ids.push(eos);
                }
                if p_ids.is_empty() && c_ids.len() < 2 {
                    continue;
                }
                // Keep at least 1 completion token inside `max_seq_len` (left-truncate prompt if needed)
                if p_ids.len() >= max_seq_len {
                    let keep_p = max_seq_len / 2;
                    p_ids = p_ids[p_ids.len() - keep_p..].to_vec();
                }
                let rem = max_seq_len.saturating_sub(p_ids.len());
                if c_ids.len() > rem {
                    c_ids.truncate(rem);
                }
                let mut full = Vec::with_capacity(p_ids.len() + c_ids.len());
                let mut mask = Vec::with_capacity(p_ids.len() + c_ids.len());
                full.extend_from_slice(&p_ids);
                mask.resize(p_ids.len(), 0u8);
                full.extend_from_slice(&c_ids);
                mask.resize(full.len(), 1u8);
                if !mask.is_empty() {
                    mask[0] = 0;
                }
                samples.push(TrainSample {
                    tokens: full,
                    loss_mask: mask,
                });
            } else if let Some(txt) = obj.strings.get("text") {
                let mut ids = tok.encode(txt, tok.add_bos);
                if ids.len() > max_seq_len {
                    ids.truncate(max_seq_len);
                }
                if ids.len() >= 2 {
                    let mut mask = vec![1u8; ids.len()];
                    mask[0] = 0;
                    samples.push(TrainSample {
                        tokens: ids,
                        loss_mask: mask,
                    });
                }
            } else {
                return Err(TrainError::Dataset {
                    path: path_str.to_string(),
                    detail: format!(
                        "line {}: expected ('tokens'), ('prompt'+'completion'), or ('text')",
                        line_no + 1
                    ),
                });
            }
        }
        Self::from_samples(path_str, samples)
    }
}

/// Writes pre-tokenized `TrainSample` records into `TMSFT` v1 binary format.
pub fn write_tmsft_file<P: AsRef<Path>>(
    dst: P,
    samples: &[TrainSample],
) -> Result<usize, TrainError> {
    let p = dst.as_ref();
    let mut f = fs::File::create(p).map_err(|e| TrainError::Io {
        path: p.display().to_string(),
        message: e.to_string(),
    })?;
    let mut written = 0usize;
    f.write_all(TMSFT_MAGIC).map_err(|e| TrainError::Io {
        path: p.display().to_string(),
        message: e.to_string(),
    })?;
    written += 4;
    f.write_all(&TMSFT_VERSION.to_le_bytes())
        .map_err(|e| TrainError::Io {
            path: p.display().to_string(),
            message: e.to_string(),
        })?;
    written += 4;
    f.write_all(&(samples.len() as u32).to_le_bytes())
        .map_err(|e| TrainError::Io {
            path: p.display().to_string(),
            message: e.to_string(),
        })?;
    written += 4;
    for s in samples {
        assert_eq!(s.tokens.len(), s.loss_mask.len());
        let len = s.tokens.len() as u32;
        f.write_all(&len.to_le_bytes())
            .map_err(|e| TrainError::Io {
                path: p.display().to_string(),
                message: e.to_string(),
            })?;
        written += 4;
        for &t in &s.tokens {
            f.write_all(&t.to_le_bytes()).map_err(|e| TrainError::Io {
                path: p.display().to_string(),
                message: e.to_string(),
            })?;
            written += 4;
        }
        f.write_all(&s.loss_mask).map_err(|e| TrainError::Io {
            path: p.display().to_string(),
            message: e.to_string(),
        })?;
        written += s.loss_mask.len();
    }
    Ok(written)
}

#[derive(Default)]
struct FlatJsonObj {
    strings: std::collections::HashMap<String, String>,
    int_arrays: std::collections::HashMap<String, Vec<i64>>,
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n') {
        *i += 1;
    }
}

fn parse_json_string(s: &str, i: &mut usize) -> Result<String, String> {
    let b = s.as_bytes();
    skip_ws(b, i);
    if *i >= b.len() || b[*i] != b'"' {
        return Err("expected '\"'".to_string());
    }
    *i += 1;
    let mut out = String::new();
    while *i < b.len() {
        let ch = s[*i..].chars().next().ok_or("invalid UTF-8")?;
        *i += ch.len_utf8();
        if ch == '"' {
            return Ok(out);
        }
        if ch == '\\' {
            if *i >= b.len() {
                return Err("unterminated escape".to_string());
            }
            let esc = b[*i];
            *i += 1;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000c}'),
                b'u' => {
                    if *i + 4 > b.len() {
                        return Err("truncated \\uXXXX".to_string());
                    }
                    let hex = &s[*i..*i + 4];
                    *i += 4;
                    let code =
                        u32::from_str_radix(hex, 16).map_err(|_| format!("invalid \\u{hex}"))?;
                    if let Some(c) = char::from_u32(code) {
                        out.push(c);
                    }
                }
                other => return Err(format!("unsupported escape '\\{}'", other as char)),
            }
        } else {
            out.push(ch);
        }
    }
    Err("unterminated JSON string".to_string())
}

fn skip_json_value(s: &str, i: &mut usize) -> Result<(), String> {
    let b = s.as_bytes();
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected EOF".to_string());
    }
    match b[*i] {
        b'"' => {
            let _ = parse_json_string(s, i)?;
            Ok(())
        }
        b'{' | b'[' => {
            let open = b[*i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            while *i < b.len() {
                if b[*i] == b'"' {
                    let _ = parse_json_string(s, i)?;
                    continue;
                }
                if b[*i] == open {
                    depth += 1;
                } else if b[*i] == close {
                    depth -= 1;
                    if depth == 0 {
                        *i += 1;
                        return Ok(());
                    }
                }
                *i += 1;
            }
            Err("unterminated nested JSON container".to_string())
        }
        _ => {
            while *i < b.len()
                && !matches!(b[*i], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
            {
                *i += 1;
            }
            Ok(())
        }
    }
}

fn parse_json_int_array(s: &str, i: &mut usize) -> Result<Vec<i64>, String> {
    let b = s.as_bytes();
    skip_ws(b, i);
    if *i >= b.len() || b[*i] != b'[' {
        return Err("expected '['".to_string());
    }
    *i += 1;
    let mut vals = Vec::new();
    loop {
        skip_ws(b, i);
        if *i >= b.len() {
            return Err("unterminated '['".to_string());
        }
        if b[*i] == b']' {
            *i += 1;
            return Ok(vals);
        }
        let start = *i;
        while *i < b.len() && !matches!(b[*i], b',' | b']' | b' ' | b'\t' | b'\r' | b'\n') {
            *i += 1;
        }
        let tok_str = &s[start..*i];
        let v: i64 = tok_str
            .parse()
            .map_err(|_| format!("invalid integer '{tok_str}'"))?;
        vals.push(v);
        skip_ws(b, i);
        if *i < b.len() && b[*i] == b',' {
            *i += 1;
        }
    }
}

fn parse_flat_json_line(s: &str) -> Result<FlatJsonObj, String> {
    let b = s.as_bytes();
    let mut i = 0usize;
    skip_ws(b, &mut i);
    if i >= b.len() || b[i] != b'{' {
        return Err("expected '{'".to_string());
    }
    i += 1;
    let mut obj = FlatJsonObj::default();
    loop {
        skip_ws(b, &mut i);
        if i >= b.len() {
            return Err("unterminated '{'".to_string());
        }
        if b[i] == b'}' {
            return Ok(obj);
        }
        let key = parse_json_string(s, &mut i)?;
        skip_ws(b, &mut i);
        if i >= b.len() || b[i] != b':' {
            return Err("expected ':'".to_string());
        }
        i += 1;
        skip_ws(b, &mut i);
        if i >= b.len() {
            return Err("unexpected EOF after ':'".to_string());
        }
        match b[i] {
            b'"' => {
                let val = parse_json_string(s, &mut i)?;
                obj.strings.insert(key, val);
            }
            b'[' if key == "tokens" || key == "loss_mask" => {
                let arr = parse_json_int_array(s, &mut i)?;
                obj.int_arrays.insert(key, arr);
            }
            _ => {
                skip_json_value(s, &mut i)?;
            }
        }
        skip_ws(b, &mut i);
        if i < b.len() && b[i] == b',' {
            i += 1;
        }
    }
}
