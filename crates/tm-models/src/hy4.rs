// Adapted in part from `Strata` (`engine/src/artifact/strata_reader.cpp`, `engine/src/ngram/hy4_engram_reader.cpp`,
// and `engine/src/hy4flash/cpu/hy4_cpu.cpp`, MIT License, Copyright (c) 2026 Agention AI).

//! `Tencent Hy4-Flash` (`.strata` v1.0 / v1.1 container) reader, `S_K` / `Q8_ROW` codecs,
//! 4-module 8-bit Engram reader, and safe CPU reference forward pass (`#![forbid(unsafe_code)]`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tm_core::{f16_to_f32, f32_to_f16};
use tm_cpu::{KernelTable, RopeType, ThreadPool, rms_norm, rope_inplace};

use crate::llama::ModelError;

pub const HY4_MAGIC: &[u8; 8] = b"STRATA\x00\x01";
pub const STRATA_PAGE_SIZE: usize = 4096;
pub const STRATA_NUM_SECTIONS: usize = 6;
pub const HY4_ENGRAM_PAGE_BYTES: u64 = 4096;
pub const HY4_ENGRAM_ROWS_PER_PAGE: u64 = 40;

pub const FMT_S125: i32 = 0;
pub const FMT_S15: i32 = 1;
pub const FMT_S2: i32 = 2;
pub const FMT_S3: i32 = 3;
pub const FMT_S4: i32 = 4;
pub const FMT_S5: i32 = 5;
pub const FMT_S6: i32 = 6;
pub const FMT_S8: i32 = 8;
pub const FMT_Q8ROW: i32 = 100;
pub const FMT_F32: i32 = 101;

const S125_CODEBOOK: [[i8; 4]; 32] = [
    [-1, -1, -1, -1],
    [-1, -1, -1, 1],
    [-1, -1, 1, -1],
    [-1, -1, 1, 1],
    [-1, 1, -1, -1],
    [-1, 1, -1, 1],
    [-1, 1, 1, -1],
    [-1, 1, 1, 1],
    [1, -1, -1, -1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, -1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [1, 1, 1, 1],
    [2, 0, 0, 0],
    [-2, 0, 0, 0],
    [0, 2, 0, 0],
    [0, -2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, -2, 0],
    [0, 0, 0, 2],
    [0, 0, 0, -2],
    [0, 0, 0, 0],
    [2, 2, 0, 0],
    [-2, -2, 0, 0],
    [0, 0, 2, 2],
    [0, 0, -2, -2],
    [2, -2, 0, 0],
    [0, 0, 2, -2],
    [2, 0, 2, 0],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrataProfile {
    Tiered = 0,
    Unified = 1,
    EngramShard = 2,
    VisionShard = 3,
}

impl StrataProfile {
    fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Self::Tiered),
            1 => Some(Self::Unified),
            2 => Some(Self::EngramShard),
            3 => Some(Self::VisionShard),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StrataSectionEntry {
    pub file_offset: u64,
    pub byte_length: u64,
    pub sha256: [u8; 32],
    pub section_id: u32,
    pub flags: u32,
    pub reserved: u64,
}

#[derive(Debug, Clone)]
pub struct StrataHeader {
    pub flags: u32,
    pub profile: StrataProfile,
    pub num_sections: u16,
    pub sections: [StrataSectionEntry; STRATA_NUM_SECTIONS],
    pub engram_shard_sha256: [u8; 32],
}

/// 32-byte `.strata` Section 3 expert directory entry (`strata_reader.hpp:StrataExpertEntry`).
#[derive(Debug, Clone, Copy)]
pub struct StrataExpertEntry {
    pub expert_id: u16,
    pub gu_code_id: u8,
    pub dn_code_id: u8,
    pub cluster_id: u16,
    pub flags: u16,
    pub body_offset: u64,
    pub body_bytes: u32,
    pub dn_offset: u32,
    pub aux_offsets: [u32; 2],
}

impl StrataExpertEntry {
    pub fn from_le_bytes(b: &[u8; 32]) -> Self {
        Self {
            expert_id: u16::from_le_bytes([b[0], b[1]]),
            gu_code_id: b[2],
            dn_code_id: b[3],
            cluster_id: u16::from_le_bytes([b[4], b[5]]),
            flags: u16::from_le_bytes([b[6], b[7]]),
            body_offset: u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]),
            body_bytes: u32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            dn_offset: u32::from_le_bytes([b[20], b[21], b[22], b[23]]),
            aux_offsets: [
                u32::from_le_bytes([b[24], b[25], b[26], b[27]]),
                u32::from_le_bytes([b[28], b[29], b[30], b[31]]),
            ],
        }
    }

    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..2].copy_from_slice(&self.expert_id.to_le_bytes());
        out[2] = self.gu_code_id;
        out[3] = self.dn_code_id;
        out[4..6].copy_from_slice(&self.cluster_id.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..16].copy_from_slice(&self.body_offset.to_le_bytes());
        out[16..20].copy_from_slice(&self.body_bytes.to_le_bytes());
        out[20..24].copy_from_slice(&self.dn_offset.to_le_bytes());
        out[24..28].copy_from_slice(&self.aux_offsets[0].to_le_bytes());
        out[28..32].copy_from_slice(&self.aux_offsets[1].to_le_bytes());
        out
    }
}

/// Checks if `path` starts with `STRATA\x00\x01` magic or has `.strata` extension.
pub fn is_strata_file(path: &Path) -> bool {
    if path.extension().and_then(|s| s.to_str()) == Some("strata") {
        return true;
    }
    if let Ok(mut f) = File::open(path) {
        let mut magic = [0u8; 8];
        if f.read_exact(&mut magic).is_ok() {
            return &magic == HY4_MAGIC;
        }
    }
    false
}

/// Auto-discovers a sibling `*Engram*.strata` or `*engram*.strata` in the same directory as `model_path`.
pub fn discover_sibling_engram_strata(model_path: &Path) -> Option<PathBuf> {
    let dir = model_path.parent()?;
    let entries = std::fs::read_dir(dir).ok()?;
    for ent in entries.flatten() {
        let p = ent.path();
        if p == model_path {
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) == Some("strata")
            && let Some(nm) = p.file_name().and_then(|s| s.to_str())
            && nm.to_ascii_lowercase().contains("engram")
        {
            return Some(p);
        }
    }
    None
}

pub fn read_strata_header(f: &mut File) -> Result<StrataHeader, ModelError> {
    f.seek(SeekFrom::Start(0))
        .map_err(|e| ModelError::InvalidConfig(format!("strata seek header: {e}")))?;
    let mut frame = vec![0u8; STRATA_PAGE_SIZE];
    f.read_exact(&mut frame)
        .map_err(|e| ModelError::InvalidConfig(format!("strata read 4 KiB header: {e}")))?;
    if &frame[0..8] != HY4_MAGIC {
        return Err(ModelError::InvalidConfig(
            "invalid .strata magic (expected STRATA\\x00\\x01)".to_string(),
        ));
    }
    let flags = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
    let prof_u16 = u16::from_le_bytes([frame[12], frame[13]]);
    let profile = StrataProfile::from_u16(prof_u16).ok_or_else(|| {
        ModelError::InvalidConfig(format!("unknown .strata profile id {prof_u16}"))
    })?;
    let num_sections = u16::from_le_bytes([frame[14], frame[15]]);
    if num_sections as usize != STRATA_NUM_SECTIONS {
        return Err(ModelError::InvalidConfig(format!(
            "unexpected .strata num_sections {num_sections}"
        )));
    }
    let dummy_sec = StrataSectionEntry {
        file_offset: 0,
        byte_length: 0,
        sha256: [0u8; 32],
        section_id: 0,
        flags: 0,
        reserved: 0,
    };
    let mut sections = [dummy_sec; STRATA_NUM_SECTIONS];
    for (i, sec) in sections.iter_mut().enumerate() {
        let b = &frame[16 + i * 64..16 + (i + 1) * 64];
        let mut sha = [0u8; 32];
        sha.copy_from_slice(&b[16..48]);
        *sec = StrataSectionEntry {
            file_offset: u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            byte_length: u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]),
            sha256: sha,
            section_id: u32::from_le_bytes([b[48], b[49], b[50], b[51]]),
            flags: u32::from_le_bytes([b[52], b[53], b[54], b[55]]),
            reserved: u64::from_le_bytes([b[56], b[57], b[58], b[59], b[60], b[61], b[62], b[63]]),
        };
    }
    let mut engram_shard_sha256 = [0u8; 32];
    engram_shard_sha256.copy_from_slice(&frame[0x190..0x1B0]);
    Ok(StrataHeader {
        flags,
        profile,
        num_sections,
        sections,
        engram_shard_sha256,
    })
}

pub fn read_strata_section(
    f: &mut File,
    hdr: &StrataHeader,
    sec_idx: usize,
) -> Result<Vec<u8>, ModelError> {
    let sec = &hdr.sections[sec_idx];
    if sec.byte_length == 0 {
        return Ok(Vec::new());
    }
    let mut buf = vec![0u8; sec.byte_length as usize];
    f.seek(SeekFrom::Start(sec.file_offset))
        .map_err(|e| ModelError::InvalidConfig(format!("strata seek sec {sec_idx}: {e}")))?;
    f.read_exact(&mut buf)
        .map_err(|e| ModelError::InvalidConfig(format!("strata read sec {sec_idx}: {e}")))?;
    Ok(buf)
}

pub fn read_expert_directory(
    f: &mut File,
    hdr: &StrataHeader,
) -> Result<Vec<StrataExpertEntry>, ModelError> {
    let s3 = read_strata_section(f, hdr, 3)?;
    if s3.is_empty() {
        return Ok(Vec::new());
    }
    let has_vram_order = (hdr.sections[3].flags & 1) != 0;
    let n = if has_vram_order {
        if !s3.len().is_multiple_of(34) {
            return Err(ModelError::InvalidConfig(format!(
                "Section 3 v1.1 byte_length {} not divisible by 34",
                s3.len()
            )));
        }
        s3.len() / 34
    } else {
        if !s3.len().is_multiple_of(32) {
            return Err(ModelError::InvalidConfig(format!(
                "Section 3 byte_length {} not divisible by 32",
                s3.len()
            )));
        }
        s3.len() / 32
    };
    let mut out = Vec::with_capacity(n);
    for chunk in s3[..n * 32].as_chunks::<32>().0 {
        out.push(StrataExpertEntry::from_le_bytes(chunk));
    }
    Ok(out)
}

fn decompress_section0_manifest(raw: &[u8]) -> Result<String, ModelError> {
    if raw.is_empty() {
        return Err(ModelError::InvalidConfig(
            "empty Section 0 manifest".to_string(),
        ));
    }
    // Support uncompressed JSON directly for synthetic tests!
    let trimmed = raw
        .iter()
        .position(|b| !matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        .map_or(raw, |i| &raw[i..]);
    if trimmed.starts_with(b"{") {
        let end = raw.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
        return String::from_utf8(raw[..end].to_vec())
            .map_err(|e| ModelError::InvalidConfig(format!("Section 0 UTF-8 error: {e}")));
    }

    // ZSTD frame (`0xFD2FB528` little-endian = `[0x28, 0xB5, 0x2F, 0xFD]`):
    // Pipe through `zstd -d -c` with a dedicated writer thread so large manifests never deadlock pipe buffers.
    let try_cmd = |prog: &str, args: &[&str]| -> Option<Vec<u8>> {
        let mut child = Command::new(prog)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let mut stdin = child.stdin.take()?;
        let payload = raw.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(&payload);
        });
        let out = child.wait_with_output().ok()?;
        let _ = writer.join();
        if out.status.success() && !out.stdout.is_empty() {
            Some(out.stdout)
        } else {
            None
        }
    };

    let bytes = try_cmd("zstd", &["-d", "-c"])
        .or_else(|| {
            try_cmd(
                "python3",
                &[
                    "-c",
                    "import sys, zstandard; sys.stdout.buffer.write(zstandard.ZstdDecompressor().decompress(sys.stdin.buffer.read(), max_output_size=1<<30))",
                ],
            )
        })
        .ok_or_else(|| {
            ModelError::InvalidConfig(
                "failed to decompress ZSTD Section 0 manifest (install `zstd` CLI)".to_string(),
            )
        })?;
    String::from_utf8(bytes)
        .map_err(|e| ModelError::InvalidConfig(format!("manifest UTF-8 error: {e}")))
}

#[derive(Debug, Clone)]
pub struct Hy4TensorInfo {
    pub name: String,
    pub fmt: i32,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: u64,
}

impl Hy4TensorInfo {
    #[inline]
    pub fn rows(&self) -> usize {
        if self.shape.len() > 1 {
            self.shape[0]
        } else {
            1
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape.last().copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct Hy4Config {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub hc_count: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub hc_eps: f32,
    pub hc_mag: f32,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub qk_head: usize,
    pub v_head: usize,
    pub window: usize,
    pub kpool: usize,
    pub index_topk: usize,
    pub n_idx_head: usize,
    pub idx_dim: usize,
    pub swiglu_limit: f32,
    pub routed_scale: f32,
    pub n_experts: usize,
    pub top_k: usize,
    pub heads_sparse: usize,
    pub heads_swa: usize,
    pub theta_sparse: f64,
    pub theta_swa: f64,
    pub is_swa: Vec<bool>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub dense_ff: usize,
    pub lat_dim: usize,
    pub dfly_mask_id: u32,
    pub dfly_layers: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Hy4EngramCfg {
    pub n_mod: usize,
    pub n_cols: usize,
    pub n_head: usize,
    pub n_gram: usize,
    pub pad_id: usize,
    pub row_dim: usize,
    pub engram_dim: usize,
    pub layers: Vec<usize>,
    pub token_map: Vec<i64>,
    pub mult: Vec<u64>,
    pub primes: Vec<u64>,
    pub offs: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct Hy4Manifest {
    pub cfg: Hy4Config,
    pub engram: Hy4EngramCfg,
    pub tensors: HashMap<String, Hy4TensorInfo>,
    pub tokenizer_json: String,
    pub chat_template: String,
}

impl Hy4Manifest {
    pub fn from_strata_path(
        path: &Path,
        keep_tokenizer: bool,
    ) -> Result<(StrataHeader, Self), ModelError> {
        let mut f = File::open(path)
            .map_err(|e| ModelError::InvalidConfig(format!("open '{}': {e}", path.display())))?;
        let hdr = read_strata_header(&mut f)?;
        let s0 = read_strata_section(&mut f, &hdr, 0)?;
        let json_text = decompress_section0_manifest(&s0)?;
        let man = Self::parse_json(&json_text, keep_tokenizer)?;
        Ok((hdr, man))
    }

    pub fn req_tensor(&self, name: &str) -> Result<&Hy4TensorInfo, ModelError> {
        self.tensors
            .get(name)
            .ok_or_else(|| ModelError::InvalidConfig(format!("missing .strata tensor '{name}'")))
    }

    pub fn parse_json(json_text: &str, keep_tokenizer: bool) -> Result<Self, ModelError> {
        let root = parse_json_value(json_text)?;
        let tc = root.req_key("config")?.req_key("text_config")?;

        let n_layers = tc.req_key("num_hidden_layers")?.as_usize()?;
        let n_embd = tc.req_key("hidden_size")?.as_usize()?;
        let hc_count = tc.req_key("hc_mult")?.as_usize()?;
        let vocab_size = tc.req_key("vocab_size")?.as_usize()?;
        let rms_eps = tc.req_key("rms_norm_eps")?.as_f64()? as f32;
        let hc_eps = tc.req_key("hc_eps")?.as_f64()? as f32;
        let hc_mag = tc.req_key("hc_magnitude")?.as_f64()? as f32;
        let q_lora = tc.req_key("q_lora_rank")?.as_usize()?;
        let kv_lora = tc.req_key("kv_lora_rank")?.as_usize()?;
        let qk_nope = tc.req_key("qk_nope_head_dim")?.as_usize()?;
        let qk_rope = tc.req_key("qk_rope_head_dim")?.as_usize()?;
        let qk_head = tc.req_key("qk_head_dim")?.as_usize()?;
        let v_head = tc.req_key("v_head_dim")?.as_usize()?;
        let window = tc.req_key("sliding_window")?.as_usize()?;
        let kpool = tc
            .req_key("compressed_indexer_compress_ratio")?
            .as_usize()?;
        let index_topk = tc.req_key("index_topk")?.as_usize()?;
        let n_idx_head = tc
            .req_key("compressed_indexer_num_attention_heads")?
            .as_usize()?;
        let idx_dim = tc.req_key("compressed_indexer_head_dim")?.as_usize()?;
        let swiglu_limit = tc.req_key("swiglu_limit")?.as_f64()? as f32;
        let routed_scale = tc.req_key("routed_scaling_factor")?.as_f64()? as f32;
        let n_experts = tc.req_key("n_routed_experts")?.as_usize()?;
        let top_k = tc.req_key("num_experts_per_tok")?.as_usize()?;
        let heads_sparse = tc.req_key("num_attention_heads")?.as_usize()?;
        let heads_swa = tc.req_key("swa_num_attention_heads")?.as_usize()?;
        let bos_id = tc
            .get_key("bos_token_id")
            .map_or(120_000, |v| v.as_usize().unwrap_or(120_000) as u32);
        let eos_id = tc
            .get_key("eos_token_id")
            .map_or(120_025, |v| v.as_usize().unwrap_or(120_025) as u32);
        let rp = tc.req_key("rope_parameters")?;
        let theta_sparse = rp
            .req_key("deepseek_sparse_attention")?
            .req_key("rope_theta")?
            .as_f64()?;
        let theta_swa = rp
            .req_key("sliding_attention")?
            .req_key("rope_theta")?
            .as_f64()?;
        let lt = tc.req_key("layer_types")?.as_array()?;
        let mut is_swa = Vec::with_capacity(n_layers);
        for v in lt.iter().take(n_layers) {
            is_swa.push(v.as_str()? == "sliding_attention");
        }

        let em = root.req_key("engram_meta")?;
        let n_gram = em.req_key("ple_ngram_size")?.as_usize()?;
        let n_head = em.req_key("ple_heads_per_ngram")?.as_usize()?;
        let n_cols = (n_gram.saturating_sub(1)) * n_head;
        let pad_id = em.req_key("engram_pad_id")?.as_usize()?;
        let layers: Vec<usize> = em
            .req_key("ple_layers")?
            .as_array()?
            .iter()
            .map(|v| v.as_usize())
            .collect::<Result<_, _>>()?;
        let n_mod = layers.len();
        let token_map: Vec<i64> = em
            .req_key("engram_token_map")?
            .as_array()?
            .iter()
            .map(|v| v.as_i64())
            .collect::<Result<_, _>>()?;
        let mult: Vec<u64> = em
            .req_key("ple_layer_multipliers")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64())
            .collect::<Result<_, _>>()?;
        let primes: Vec<u64> = em
            .req_key("ple_head_vocab_sizes")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64())
            .collect::<Result<_, _>>()?;
        let offs: Vec<i64> = em
            .req_key("ple_head_offsets")?
            .as_array()?
            .iter()
            .map(|v| v.as_i64())
            .collect::<Result<_, _>>()?;

        let mut tensors = HashMap::new();
        for t in root.req_key("trunk_tensors")?.as_array()? {
            let name = t.req_key("name")?.as_str()?.to_string();
            let fmt = fmt_from_str(t.req_key("format")?.as_str()?)?;
            let shape: Vec<usize> = t
                .req_key("shape")?
                .as_array()?
                .iter()
                .map(|d| d.as_usize())
                .collect::<Result<_, _>>()?;
            let offset = t.req_key("offset")?.as_u64()?;
            let bytes = t.req_key("bytes")?.as_u64()?;
            tensors.insert(
                name.clone(),
                Hy4TensorInfo {
                    name,
                    fmt,
                    shape,
                    offset,
                    bytes,
                },
            );
        }

        let dense_ff = tensors
            .get("model.layers.0.mlp.gate_proj.weight")
            .map_or(13_824, |t| t.rows());
        let lat_dim = tensors
            .get("model.layers.1.mlp.routed_expert_down_proj.weight")
            .or_else(|| tensors.get("model.layers.0.mlp.routed_expert_down_proj.weight"))
            .map_or(1536, |t| t.rows());
        let engram_dim = layers
            .first()
            .and_then(|&il| {
                tensors.get(&format!("model.layers.{il}.engram_layer.value_proj.weight"))
            })
            .map_or(1536, |t| t.cols());
        let row_dim = engram_dim.checked_div(n_cols).unwrap_or(96);
        let dfly_mask_id = tc
            .get_key("dfly_mask_id")
            .and_then(|v| v.as_usize().ok())
            .map(|x| x as u32)
            .unwrap_or_else(|| {
                if vocab_size > 120_009 {
                    120_009
                } else {
                    vocab_size.saturating_sub(1) as u32
                }
            });
        let dfly_layers = if tensors.contains_key("model.dfly_layers.0.input_layernorm.weight") {
            let mut n_dfly = 0usize;
            while tensors.contains_key(&format!(
                "model.dfly_layers.{n_dfly}.input_layernorm.weight"
            )) {
                n_dfly += 1;
            }
            if let Some(arr) = tc.get_key("dfly_layers").and_then(|v| v.as_array().ok()) {
                arr.iter()
                    .take(n_dfly)
                    .filter_map(|v| v.as_usize().ok())
                    .collect()
            } else if n_layers >= 42 && n_dfly == 5 {
                vec![25, 29, 33, 37, 41]
            } else {
                let mut sparse: Vec<usize> = (0..n_layers).filter(|&l| !is_swa[l]).collect();
                if sparse.len() >= n_dfly {
                    sparse.split_off(sparse.len() - n_dfly)
                } else {
                    (0..n_dfly).map(|d| d % n_layers.max(1)).collect()
                }
            }
        } else {
            Vec::new()
        };

        let tokenizer_json = if keep_tokenizer {
            root.get_key("tokenizer_json")
                .and_then(|v| v.as_str().ok())
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };
        let chat_template = if keep_tokenizer {
            root.get_key("chat_template")
                .and_then(|v| v.as_str().ok())
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };

        Ok(Self {
            cfg: Hy4Config {
                arch: "hy4flash".to_string(),
                n_layers,
                n_embd,
                hc_count,
                vocab_size,
                rms_eps,
                hc_eps,
                hc_mag,
                q_lora,
                kv_lora,
                qk_nope,
                qk_rope,
                qk_head,
                v_head,
                window,
                kpool,
                index_topk,
                n_idx_head,
                idx_dim,
                swiglu_limit,
                routed_scale,
                n_experts,
                top_k,
                heads_sparse,
                heads_swa,
                theta_sparse,
                theta_swa,
                is_swa,
                bos_id,
                eos_id,
                dense_ff,
                lat_dim,
                dfly_mask_id,
                dfly_layers,
            },
            engram: Hy4EngramCfg {
                n_mod,
                n_cols,
                n_head,
                n_gram,
                pad_id,
                row_dim,
                engram_dim,
                layers,
                token_map,
                mult,
                primes,
                offs,
            },
            tensors,
            tokenizer_json,
            chat_template,
        })
    }
}

pub fn fmt_from_str(s: &str) -> Result<i32, ModelError> {
    match s {
        "F32" => Ok(FMT_F32),
        "Q8_ROW" => Ok(FMT_Q8ROW),
        "S1.25" | "S1_25" => Ok(FMT_S125),
        "S1.5" | "S1_5" => Ok(FMT_S15),
        "S2_K" => Ok(FMT_S2),
        "S3_K" => Ok(FMT_S3),
        "S4_K" => Ok(FMT_S4),
        "S5_K" => Ok(FMT_S5),
        "S6_K" => Ok(FMT_S6),
        "S8_K" => Ok(FMT_S8),
        other => Err(ModelError::InvalidConfig(format!(
            "unknown .strata weight format '{other}'"
        ))),
    }
}

#[inline]
pub const fn sk_superblock_bytes(code: i32) -> usize {
    match code {
        0 => 48,
        1 => 58,
        2 => 70,
        3 => 102,
        4 => 134,
        5 => 166,
        6 => 198,
        8 => 262,
        _ => 0,
    }
}

/// Dequantizes a single row of `cols` elements in `.strata` `fmt` (`F32`, `Q8_ROW`, or `S_K`).
/// Note: for `Q8_ROW`, `tensor_bytes` must point to the start of the whole `Q8_ROW` tensor (`[rows, cols]`)
/// and `row_idx` selects the row; for `F32` and `S_K`, `tensor_bytes` also starts at the tensor base and `row_idx` selects the row!
pub fn dequant_hy4_row(
    fmt: i32,
    tensor_bytes: &[u8],
    rows: usize,
    cols: usize,
    row_idx: usize,
    dst: &mut [f32],
) {
    if fmt == FMT_F32 {
        let row = &tensor_bytes[row_idx * cols * 4..(row_idx + 1) * cols * 4];
        for (d, &c) in dst[..cols].iter_mut().zip(row.as_chunks::<4>().0.iter()) {
            *d = f32::from_le_bytes(c);
        }
        return;
    }
    if fmt == FMT_Q8ROW {
        let sc_b = &tensor_bytes[row_idx * 4..(row_idx + 1) * 4];
        let sc = f32::from_le_bytes([sc_b[0], sc_b[1], sc_b[2], sc_b[3]]);
        let q_off = rows * 4 + row_idx * cols;
        let qr = &tensor_bytes[q_off..q_off + cols];
        for j in 0..cols {
            dst[j] = (qr[j] as i8 as f32) * sc;
        }
        return;
    }

    let sb_bytes = sk_superblock_bytes(fmt);
    let n_sb = cols / 256;
    let row = &tensor_bytes[row_idx * n_sb * sb_bytes..(row_idx + 1) * n_sb * sb_bytes];
    for sb in 0..n_sb {
        let sbp = &row[sb * sb_bytes..(sb + 1) * sb_bytes];
        let d_sb = f16_to_f32(u16::from_le_bytes([sbp[0], sbp[1]]));
        let pl = &sbp[6..];
        let out_sb = &mut dst[sb * 256..(sb + 1) * 256];
        for lane in 0..32 {
            let g = lane >> 3;
            let elem0 = lane * 8;
            let mut eff = d_sb * ((sbp[2 + g] as f32) * (1.0 / 255.0));
            match fmt {
                8 => {
                    for j in 0..8 {
                        out_sb[elem0 + j] = (pl[elem0 + j] as i8 as f32) * eff;
                    }
                }
                4 => {
                    for j in 0..4 {
                        let b = pl[(elem0 >> 1) + j];
                        out_sb[elem0 + 2 * j] = ((b & 0x0F) as f32 - 7.5) * eff;
                        out_sb[elem0 + 2 * j + 1] = ((b >> 4) as f32 - 7.5) * eff;
                    }
                }
                5 => {
                    let h = pl[128 + lane];
                    for j in 0..4 {
                        let b = pl[(elem0 >> 1) + j];
                        let q0 = (b & 0x0F) | (((h >> (2 * j)) & 1) << 4);
                        let q1 = (b >> 4) | (((h >> (2 * j + 1)) & 1) << 4);
                        out_sb[elem0 + 2 * j] = (q0 as f32 - 15.5) * eff;
                        out_sb[elem0 + 2 * j + 1] = (q1 as f32 - 15.5) * eff;
                    }
                }
                6 => {
                    let h0 = pl[128 + 2 * lane];
                    let h1 = pl[128 + 2 * lane + 1];
                    for j in 0..4 {
                        let b = pl[(elem0 >> 1) + j];
                        let hi_pair = if j < 2 {
                            h0 >> (4 * j)
                        } else {
                            h1 >> (4 * (j - 2))
                        };
                        let q0 = (b & 0x0F) | ((hi_pair & 0x03) << 4);
                        let q1 = (b >> 4) | (((hi_pair >> 2) & 0x03) << 4);
                        out_sb[elem0 + 2 * j] = (q0 as f32 - 31.5) * eff;
                        out_sb[elem0 + 2 * j + 1] = (q1 as f32 - 31.5) * eff;
                    }
                }
                2 => {
                    let b0 = pl[2 * lane];
                    let b1 = pl[2 * lane + 1];
                    for j in 0..4 {
                        out_sb[elem0 + j] = (((b0 >> (2 * j)) & 0x03) as f32 - 1.5) * eff;
                        out_sb[elem0 + 4 + j] = (((b1 >> (2 * j)) & 0x03) as f32 - 1.5) * eff;
                    }
                }
                3 => {
                    let b0 = pl[2 * lane];
                    let b1 = pl[2 * lane + 1];
                    let h = pl[64 + lane];
                    for j in 0..4 {
                        let q0 = ((b0 >> (2 * j)) & 0x03) | (((h >> j) & 1) << 2);
                        let q1 = ((b1 >> (2 * j)) & 0x03) | (((h >> (4 + j)) & 1) << 2);
                        out_sb[elem0 + j] = (q0 as f32 - 3.5) * eff;
                        out_sb[elem0 + 4 + j] = (q1 as f32 - 3.5) * eff;
                    }
                }
                1 => {
                    for j in 0..8 {
                        let idx = elem0 + j;
                        let mut x = pl[idx / 5] as i32;
                        for _ in 0..(idx % 5) {
                            x /= 3;
                        }
                        out_sb[idx] = ((x % 3 - 1) as f32) * eff;
                    }
                }
                0 => {
                    let fb = pl[40 + (g >> 1)];
                    let nib = if (g & 1) != 0 { fb >> 4 } else { fb & 0x0F };
                    eff *= 0.5 + (nib as f32) * (1.0 / 15.0);
                    for v in 0..2 {
                        let vec_idx = 2 * lane + v;
                        let g40 = vec_idx >> 3;
                        let j5 = vec_idx & 7;
                        let p5 = &pl[g40 * 5..g40 * 5 + 5];
                        let bits40 = (p5[0] as u64)
                            | ((p5[1] as u64) << 8)
                            | ((p5[2] as u64) << 16)
                            | ((p5[3] as u64) << 24)
                            | ((p5[4] as u64) << 32);
                        let cb = S125_CODEBOOK[((bits40 >> (5 * j5)) & 0x1F) as usize];
                        for k in 0..4 {
                            out_sb[elem0 + 4 * v + k] = (cb[k] as f32) * eff;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SeedEntry {
    row_id: u64,
    slot: u32,
}

/// 4-Module 8-Bit Engram reader (Section 2 hot seed + external `EngramShard` Section 4 4KiB page reader + clock LRU).
pub struct Hy4EngramReader {
    row_dim: usize,
    row_bytes: usize,
    rows_per_page: u64,
    seed_index: Vec<SeedEntry>,
    seed_rows: Vec<u8>,
    shard_file: Option<File>,
    sec4_offset: u64,
    lru_cap: usize,
    lru_clock: usize,
    lru_keys: Vec<u64>,
    lru_ref: Vec<u8>,
    lru_data: Vec<u8>,
    lru_map: HashMap<u64, usize>,
}

impl Hy4EngramReader {
    pub fn open(
        shard_path: Option<&Path>,
        section2_seed: &[u8],
        row_dim: usize,
        lru_capacity_rows: usize,
    ) -> Result<Self, ModelError> {
        let row_bytes = 4 + row_dim;
        let rows_per_page = (STRATA_PAGE_SIZE / row_bytes).max(1) as u64;

        let mut shard_file = None;
        let mut sec4_offset = 0u64;
        let mut shard_sha256: Option<[u8; 32]> = None;

        if let Some(sp) = shard_path {
            let mut sf = File::open(sp).map_err(|e| {
                ModelError::InvalidConfig(format!("open Engram shard '{}': {e}", sp.display()))
            })?;
            let shdr = read_strata_header(&mut sf)?;
            if shdr.profile != StrataProfile::EngramShard {
                return Err(ModelError::InvalidConfig(format!(
                    "'{}' is not an EngramShard .strata (got {:?})",
                    sp.display(),
                    shdr.profile
                )));
            }
            sec4_offset = shdr.sections[4].file_offset;
            shard_sha256 = Some(shdr.sections[4].sha256);
            let sec4_len = shdr.sections[4].byte_length;
            if sec4_len >= (64 << 20)
                && let Ok(mem) = tm_core::probe_system_memory()
                && mem.available_bytes > sec4_len.saturating_add(16u64 << 30)
            {
                // Host RAM comfortably holds the Engram shard (e.g., 177 GiB > 63 GiB on server):
                // sequentially pre-warm OS page cache across 16 reader handles so random 4 KiB page reads hit RAM!
                let n_workers = 16u64;
                let chunk = sec4_len.div_ceil(n_workers);
                std::thread::scope(|s| {
                    for w_i in 0..n_workers {
                        let start = sec4_offset + (w_i * chunk).min(sec4_len);
                        let end = sec4_offset + ((w_i + 1) * chunk).min(sec4_len);
                        let p_clone = sp.to_path_buf();
                        s.spawn(move || {
                            if let Ok(mut rf) = File::open(&p_clone) {
                                let mut buf = vec![0u8; 4 * 1024 * 1024];
                                if rf.seek(SeekFrom::Start(start)).is_ok() {
                                    let mut pos = start;
                                    while pos < end {
                                        let want = ((end - pos) as usize).min(buf.len());
                                        if rf.read_exact(&mut buf[..want]).is_err() {
                                            break;
                                        }
                                        pos += want as u64;
                                    }
                                }
                            }
                        });
                    }
                });
            }
            shard_file = Some(sf);
        }

        let mut seed_index = Vec::new();
        let mut seed_rows = Vec::new();
        if section2_seed.len() >= 64 && &section2_seed[0..4] == b"ENGR" {
            let fmt_id = u32::from_le_bytes([
                section2_seed[4],
                section2_seed[5],
                section2_seed[6],
                section2_seed[7],
            ]);
            let n_rows = u32::from_le_bytes([
                section2_seed[8],
                section2_seed[9],
                section2_seed[10],
                section2_seed[11],
            ]) as usize;
            let row_b = u32::from_le_bytes([
                section2_seed[12],
                section2_seed[13],
                section2_seed[14],
                section2_seed[15],
            ]) as usize;
            let sha_ok = shard_sha256.is_none_or(|sh| section2_seed[16..48] == sh);
            if fmt_id == 8 && row_b == row_bytes && sha_ok {
                let idx_bytes = n_rows * 12;
                let data_off = (64 + idx_bytes).next_multiple_of(64);
                let data_bytes = n_rows * row_bytes;
                if data_off + data_bytes <= section2_seed.len() {
                    seed_index.reserve(n_rows);
                    for i in 0..n_rows {
                        let p = &section2_seed[64 + i * 12..64 + (i + 1) * 12];
                        seed_index.push(SeedEntry {
                            row_id: u64::from_le_bytes([
                                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                            ]),
                            slot: u32::from_le_bytes([p[8], p[9], p[10], p[11]]),
                        });
                    }
                    seed_rows.extend_from_slice(&section2_seed[data_off..data_off + data_bytes]);
                }
            }
        }

        let lru_cap = if shard_file.is_some() {
            lru_capacity_rows
        } else {
            0
        };
        Ok(Self {
            row_dim,
            row_bytes,
            rows_per_page,
            seed_index,
            seed_rows,
            shard_file,
            sec4_offset,
            lru_cap,
            lru_clock: 0,
            lru_keys: vec![u64::MAX; lru_cap],
            lru_ref: vec![0u8; lru_cap],
            lru_data: vec![0u8; lru_cap * row_bytes],
            lru_map: HashMap::with_capacity(lru_cap),
        })
    }

    fn decode_raw(&self, raw: &[u8], dst: &mut [f32]) {
        let sc = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        for i in 0..self.row_dim {
            dst[i] = (raw[4 + i] as i8 as f32) * sc;
        }
    }

    fn lookup_seed(&self, row_id: u64) -> Option<&[u8]> {
        let idx = self
            .seed_index
            .binary_search_by_key(&row_id, |e| e.row_id)
            .ok()?;
        let slot = self.seed_index[idx].slot as usize;
        Some(&self.seed_rows[slot * self.row_bytes..(slot + 1) * self.row_bytes])
    }

    pub fn fetch_rows_f32(
        &mut self,
        row_ids: &[u64],
        out_f32: &mut [f32],
    ) -> Result<(), ModelError> {
        let rd = self.row_dim;
        let rb = self.row_bytes;
        let mut misses: Vec<(u64, usize, usize, u64)> = Vec::new();

        for (i, &rid) in row_ids.iter().enumerate() {
            if let Some(raw) = self.lookup_seed(rid) {
                self.decode_raw(raw, &mut out_f32[i * rd..(i + 1) * rd]);
                continue;
            }
            if self.lru_cap > 0
                && let Some(&slot) = self.lru_map.get(&rid)
            {
                self.lru_ref[slot] = 1;
                let sc = f32::from_le_bytes([
                    self.lru_data[slot * rb],
                    self.lru_data[slot * rb + 1],
                    self.lru_data[slot * rb + 2],
                    self.lru_data[slot * rb + 3],
                ]);
                for k in 0..rd {
                    out_f32[i * rd + k] = (self.lru_data[slot * rb + 4 + k] as i8 as f32) * sc;
                }
                continue;
            }
            misses.push((
                rid / self.rows_per_page,
                (rid % self.rows_per_page) as usize,
                i,
                rid,
            ));
        }

        if misses.is_empty() {
            return Ok(());
        }

        let Some(ref mut sf) = self.shard_file else {
            return Err(ModelError::InvalidConfig(
                "Hy4 Engram lookup missed Section 2 seed and no external *Engram*.strata shard was provided (pass `--engram <Hy4-Flash-Engram-Q8.strata>` or place it next to the model)".to_string(),
            ));
        };

        misses.sort_by_key(|m| m.0);
        let mut page_buf = [0u8; STRATA_PAGE_SIZE];
        let mut cur_page = u64::MAX;

        for (page_id, in_page, out_idx, rid) in misses {
            if page_id != cur_page {
                sf.seek(SeekFrom::Start(
                    self.sec4_offset + page_id * HY4_ENGRAM_PAGE_BYTES,
                ))
                .map_err(|e| {
                    ModelError::InvalidConfig(format!("Engram seek page {page_id}: {e}"))
                })?;
                sf.read_exact(&mut page_buf).map_err(|e| {
                    ModelError::InvalidConfig(format!("Engram read page {page_id}: {e}"))
                })?;
                cur_page = page_id;
            }
            let raw = &page_buf[in_page * rb..(in_page + 1) * rb];
            let sc = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            for k in 0..rd {
                out_f32[out_idx * rd + k] = (raw[4 + k] as i8 as f32) * sc;
            }

            if self.lru_cap > 0 && !self.lru_map.contains_key(&rid) {
                while self.lru_ref[self.lru_clock] != 0 {
                    self.lru_ref[self.lru_clock] = 0;
                    self.lru_clock = (self.lru_clock + 1) % self.lru_cap;
                }
                let slot = self.lru_clock;
                if self.lru_keys[slot] != u64::MAX {
                    self.lru_map.remove(&self.lru_keys[slot]);
                }
                self.lru_keys[slot] = rid;
                self.lru_ref[slot] = 1;
                self.lru_data[slot * rb..(slot + 1) * rb].copy_from_slice(raw);
                self.lru_map.insert(rid, slot);
                self.lru_clock = (self.lru_clock + 1) % self.lru_cap;
            }
        }
        Ok(())
    }

    /// Computes normalized 16-head Engram input `r_out[n_tok, engram_dim]` + `any_valid_out[n_tok]` for module `mod_idx` (`0..4`).
    #[allow(clippy::too_many_arguments)]
    pub fn compute_module_input(
        &mut self,
        ecfg: &Hy4EngramCfg,
        ln_weights: &[f32], // [n_cols * row_dim]
        seq_tokens: &[u32],
        token_ids: &[u32],
        start_pos: usize,
        mod_idx: usize,
        r_out: &mut [f32],
        any_valid_out: &mut [f32],
    ) -> Result<(), ModelError> {
        let n_tok = token_ids.len();
        let nc = ecfg.n_cols;
        let nh = ecfg.n_head;
        let ng = ecfg.n_gram;
        let rd = ecfg.row_dim;
        let ed = ecfg.engram_dim;
        let pad_cid = ecfg.token_map[ecfg.pad_id];
        let mult = &ecfg.mult[mod_idx * ng..(mod_idx + 1) * ng];
        let primes = &ecfg.primes[mod_idx * nc..(mod_idx + 1) * nc];
        let offs = &ecfg.offs[mod_idx * nc..(mod_idx + 1) * nc];

        let map_cid = |tid: u32| -> i64 { ecfg.token_map.get(tid as usize).copied().unwrap_or(-1) };

        let mut ids = vec![0u64; n_tok * nc];
        let mut valid = vec![1.0f32; n_tok * nc];

        for (i, &tok) in token_ids.iter().enumerate() {
            let pos = start_pos + i;
            let c0 = map_cid(tok);
            let dead = c0 < 0;
            let t0 = (if dead { pad_cid } else { c0 }) as u64;
            let prev1 = if pos >= 1 {
                let p = if pos > start_pos {
                    token_ids[pos - 1 - start_pos]
                } else {
                    seq_tokens[pos - 1]
                };
                map_cid(p)
            } else {
                -1
            };
            let prev2 = if pos >= 2 {
                let p = if pos - 2 >= start_pos {
                    token_ids[pos - 2 - start_pos]
                } else {
                    seq_tokens[pos - 2]
                };
                map_cid(p)
            } else {
                -1
            };
            let t1 = (if prev1 < 0 { pad_cid } else { prev1 }) as u64;
            let t2 = (if prev2 < 0 { pad_cid } else { prev2 }) as u64;
            let r1 = t0.wrapping_mul(mult[0]) ^ t1.wrapping_mul(mult[1]);
            let r2 = r1 ^ t2.wrapping_mul(mult[2]);
            for h in 0..nh {
                ids[i * nc + h] = (r1 % primes[h]).wrapping_add(offs[h] as u64);
                ids[i * nc + nh + h] = (r2 % primes[nh + h]).wrapping_add(offs[nh + h] as u64);
            }
            if dead {
                valid[i * nc..(i + 1) * nc].fill(0.0);
                any_valid_out[i] = 0.0;
            } else {
                any_valid_out[i] = 1.0;
            }
        }

        let mut raw_rows = vec![0.0f32; n_tok * nc * rd];
        self.fetch_rows_f32(&ids, &mut raw_rows)?;

        const EPS: f32 = 1.192_092_9e-7;
        for t in 0..n_tok {
            for ch in 0..nc {
                let row = &raw_rows[(t * nc + ch) * rd..(t * nc + ch + 1) * rd];
                let mut ss = 0.0f64;
                for &v in row {
                    ss += (v as f64) * (v as f64);
                }
                let inv = 1.0f32 / (((ss / rd as f64) as f32) + EPS).sqrt();
                let v_flag = valid[t * nc + ch];
                let w_ch = &ln_weights[ch * rd..(ch + 1) * rd];
                let dst = &mut r_out[t * ed + ch * rd..t * ed + (ch + 1) * rd];
                for k in 0..rd {
                    dst[k] = row[k] * inv * w_ch[k] * v_flag;
                }
            }
        }
        Ok(())
    }
}

/// Safe CPU reference model for `Hy4-Flash` (used for CPU-vs-CUDA multi-token parity tests).
pub struct Hy4Model {
    pub hdr: StrataHeader,
    pub man: Hy4Manifest,
    pub trunk: Vec<u8>,
    pub expert_dir: Vec<StrataExpertEntry>,
    pub rank_of: Vec<i32>,
    pub sec4_arena: Vec<u8>,
    pub engram_ln_weights: Vec<Vec<f32>>,
}

pub struct Hy4State {
    pub max_ctx: usize,
    pub engram: Hy4EngramReader,
    pub seq_tokens: Vec<u32>,
    /// Matrix-Absorbed MLA compressed latent cache per layer: `[cap, kv_lora + qk_rope]` in FP16 (`28x` smaller than expanded KV!).
    pub latent_kv_caches: Vec<Vec<u16>>,
    logits: Vec<f32>,
}

impl Hy4Model {
    pub fn open(
        model_path: &Path,
        engram_path: Option<&Path>,
    ) -> Result<(Self, Hy4EngramReader), ModelError> {
        let mut f = File::open(model_path).map_err(|e| {
            ModelError::InvalidConfig(format!("open '{}': {e}", model_path.display()))
        })?;
        let hdr = read_strata_header(&mut f)?;
        let s0 = read_strata_section(&mut f, &hdr, 0)?;
        let json_text = decompress_section0_manifest(&s0)?;
        let man = Hy4Manifest::parse_json(&json_text, true)?;
        let trunk = read_strata_section(&mut f, &hdr, 1)?;
        let seed = read_strata_section(&mut f, &hdr, 2)?;
        let expert_dir = read_expert_directory(&mut f, &hdr)?;
        let sec4_arena = read_strata_section(&mut f, &hdr, 4)?;

        let ne = man.cfg.n_experts;
        let n_moe = man.cfg.n_layers.saturating_sub(1);
        let mut rank_of = vec![-1i32; n_moe * ne];
        for (i, ent) in expert_dir.iter().enumerate() {
            let ml = i / ne;
            if ml < n_moe && (ent.expert_id as usize) < ne {
                rank_of[ml * ne + ent.expert_id as usize] = (i % ne) as i32;
            }
        }

        let mut engram_ln_weights = Vec::with_capacity(man.engram.n_mod);
        for &il in &man.engram.layers {
            let mut lnw = vec![0.0f32; man.engram.engram_dim];
            let rd = man.engram.row_dim;
            for ch in 0..man.engram.n_cols {
                let nm =
                    format!("model.layers.{il}.engram_layer.lookup_per_head_norms.{ch}.weight");
                let ti = man.req_tensor(&nm)?;
                let tb = &trunk[ti.offset as usize..(ti.offset + ti.bytes) as usize];
                dequant_hy4_row(ti.fmt, tb, 1, rd, 0, &mut lnw[ch * rd..(ch + 1) * rd]);
            }
            engram_ln_weights.push(lnw);
        }

        let sibling_eng = if engram_path.is_none() {
            discover_sibling_engram_strata(model_path)
        } else {
            None
        };
        let eng_ref = engram_path.or(sibling_eng.as_deref());
        let reader = Hy4EngramReader::open(eng_ref, &seed, man.engram.row_dim, 16_384)?;

        Ok((
            Self {
                hdr,
                man,
                trunk,
                expert_dir,
                rank_of,
                sec4_arena,
                engram_ln_weights,
            },
            reader,
        ))
    }

    pub fn new_state(&self, reader: Hy4EngramReader, max_ctx: usize) -> Hy4State {
        let c = &self.man.cfg;
        let kv_lat = c.kv_lora + c.qk_rope;
        let mut latent_kv_caches = Vec::with_capacity(c.n_layers);
        for il in 0..c.n_layers {
            let cap = if c.is_swa[il] && c.window > 0 {
                c.window.min(max_ctx)
            } else {
                max_ctx
            };
            latent_kv_caches.push(vec![0u16; cap * kv_lat]);
        }
        Hy4State {
            max_ctx,
            engram: reader,
            seq_tokens: vec![self.man.engram.pad_id as u32; max_ctx.max(1)],
            latent_kv_caches,
            logits: vec![0.0; c.vocab_size],
        }
    }

    fn load_vec(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let ti = self.man.req_tensor(name)?;
        let n: usize = ti.shape.iter().product();
        let mut out = vec![0.0f32; n];
        let tb = &self.trunk[ti.offset as usize..(ti.offset + ti.bytes) as usize];
        if ti.fmt == FMT_F32 {
            for (d, &c) in out.iter_mut().zip(tb.as_chunks::<4>().0.iter()) {
                *d = f32::from_le_bytes(c);
            }
        } else {
            let rows = ti.rows();
            let cols = ti.cols();
            for r in 0..rows {
                dequant_hy4_row(
                    ti.fmt,
                    tb,
                    rows,
                    cols,
                    r,
                    &mut out[r * cols..(r + 1) * cols],
                );
            }
        }
        Ok(out)
    }

    fn linear_trunk(
        &self,
        pool: &ThreadPool,
        name: &str,
        x: &[f32],
        out: &mut [f32],
        n_tok: usize,
    ) -> Result<(), ModelError> {
        let ti = self.man.req_tensor(name)?;
        let rows = ti.rows();
        let cols = ti.cols();
        let tb = &self.trunk[ti.offset as usize..(ti.offset + ti.bytes) as usize];
        self.linear_raw(pool, ti.fmt, tb, rows, cols, x, out, n_tok);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_raw(
        &self,
        pool: &ThreadPool,
        fmt: i32,
        w_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
        n_tok: usize,
    ) {
        let mut w_row = vec![0.0f32; cols];
        let _ = pool;
        for r in 0..rows {
            dequant_hy4_row(fmt, w_bytes, rows, cols, r, &mut w_row);
            for t in 0..n_tok {
                let xt = &x[t * cols..(t + 1) * cols];
                let mut acc = 0.0f32;
                for k in 0..cols {
                    acc += w_row[k] * xt[k];
                }
                out[t * rows + r] = acc;
            }
        }
    }

    pub fn forward_tokens(
        &self,
        state: &mut Hy4State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        let _ = kt;
        let n_tok = token_ids.len();
        if n_tok == 0 {
            return Ok(());
        }
        if start_pos + n_tok > state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: start_pos + n_tok - 1,
                max_ctx: state.max_ctx,
            });
        }
        let c = &self.man.cfg;
        let h = c.n_embd;
        let hc = c.hc_count;
        let hc_dim = hc * h;

        for (i, &tid) in token_ids.iter().enumerate() {
            state.seq_tokens[start_pos + i] = tid;
        }

        // 1. Token embeddings -> replicated across `hc` streams `[n_tok, hc, h]`
        let emb_ti = self.man.req_tensor("model.embed_tokens.weight")?;
        let emb_tb = &self.trunk[emb_ti.offset as usize..(emb_ti.offset + emb_ti.bytes) as usize];
        let mut x_hc = vec![0.0f32; n_tok * hc_dim];
        let mut emb_row = vec![0.0f32; h];
        for (t, &tid) in token_ids.iter().enumerate() {
            dequant_hy4_row(
                emb_ti.fmt,
                emb_tb,
                emb_ti.rows(),
                h,
                tid as usize,
                &mut emb_row,
            );
            for s in 0..hc {
                x_hc[(t * hc + s) * h..(t * hc + s + 1) * h].copy_from_slice(&emb_row);
            }
        }

        let unit_ones = vec![1.0f32; hc_dim];
        let mut fl = vec![0.0f32; n_tok * hc_dim];
        let mut cur = vec![0.0f32; n_tok * h];
        let mut curn = vec![0.0f32; n_tok * h];
        let mut post = vec![0.0f32; n_tok * hc];
        let mut sub = vec![0.0f32; n_tok * h];

        for il in 0..c.n_layers {
            // Optional Engram module
            if let Some(mod_idx) = self.man.engram.layers.iter().position(|&l| l == il) {
                let ed = self.man.engram.engram_dim;
                let mut r_in = vec![0.0f32; n_tok * ed];
                let mut any_v = vec![0.0f32; n_tok];
                state.engram.compute_module_input(
                    &self.man.engram,
                    &self.engram_ln_weights[mod_idx],
                    &state.seq_tokens,
                    token_ids,
                    start_pos,
                    mod_idx,
                    &mut r_in,
                    &mut any_v,
                )?;
                let p = format!("model.layers.{il}.engram_layer");
                let mut kv_proj = vec![0.0f32; n_tok * (hc + 1) * h];
                let mut yj = vec![0.0f32; n_tok * h];
                for j in 0..(hc + 1) {
                    let wnm = if j < hc {
                        format!("{p}.key_projs.{j}.weight")
                    } else {
                        format!("{p}.value_proj.weight")
                    };
                    let bnm = if j < hc {
                        format!("{p}.key_projs.{j}.bias")
                    } else {
                        format!("{p}.value_proj.bias")
                    };
                    self.linear_trunk(pool, &wnm, &r_in, &mut yj, n_tok)?;
                    let b = self.load_vec(&bnm)?;
                    for t in 0..n_tok {
                        for d in 0..h {
                            kv_proj[(t * (hc + 1) + j) * h + d] = yj[t * h + d] + b[d];
                        }
                    }
                }
                let pre_gamma = self.load_vec(&format!("{p}.unified_pre_gamma"))?;
                let mut nk = Vec::with_capacity(hc);
                let mut nq = Vec::with_capacity(hc);
                let mut hs = Vec::with_capacity(hc);
                for j in 0..hc {
                    nk.push(self.load_vec(&format!("{p}.norm_key.{j}.weight"))?);
                    nq.push(self.load_vec(&format!("{p}.norm_query.{j}.weight"))?);
                    hs.push(self.load_vec(&format!("{p}.unified_per_hc_scales.{j}"))?);
                }
                const EPS: f32 = 1.192_092_9e-7;
                for t in 0..n_tok {
                    if any_v[t] <= 0.0 {
                        continue;
                    }
                    let val = &kv_proj[(t * (hc + 1) + hc) * h..(t * (hc + 1) + hc + 1) * h];
                    let sv: f64 = val.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let vinv = 1.0f32 / (((sv / h as f64) as f32) + EPS).sqrt();
                    for j in 0..hc {
                        let key = &kv_proj[(t * (hc + 1) + j) * h..(t * (hc + 1) + j + 1) * h];
                        let xs = &mut x_hc[(t * hc + j) * h..(t * hc + j + 1) * h];
                        let sx: f64 = xs.iter().map(|&v| (v as f64) * (v as f64)).sum();
                        let sk: f64 = key.iter().map(|&v| (v as f64) * (v as f64)).sum();
                        let xinv = 1.0f32 / (((sx / h as f64) as f32) + EPS).sqrt();
                        let kinv = 1.0f32 / (((sk / h as f64) as f32) + EPS).sqrt();
                        let mut dot = 0.0f64;
                        for d in 0..h {
                            dot +=
                                ((xs[d] * xinv) * (key[d] * kinv) * (nk[j][d] * nq[j][d])) as f64;
                        }
                        let dotf = (dot / (h as f64).sqrt()) as f32;
                        let mag = dotf.abs().max(1e-6).sqrt();
                        let s_mag = if dotf < 0.0 { -mag } else { mag };
                        let gate = 1.0 / (1.0 + (-s_mag).exp());
                        for d in 0..h {
                            xs[d] += (val[d] * vinv * pre_gamma[d]) * (gate + hs[j][d]) * any_v[t];
                        }
                    }
                }
            }

            // Attention HC Pre
            let l_pfx = format!("model.layers.{il}");
            self.apply_hc_pre(
                pool,
                &format!("{l_pfx}.hc_attn_layer"),
                &unit_ones,
                &x_hc,
                &mut fl,
                &mut cur,
                &mut post,
                n_tok,
            )?;
            let in_ln = self.load_vec(&format!("{l_pfx}.input_layernorm.weight"))?;
            for t in 0..n_tok {
                rms_norm(
                    &mut curn[t * h..(t + 1) * h],
                    &cur[t * h..(t + 1) * h],
                    &in_ln,
                    c.rms_eps,
                );
            }
            self.attn_sublayer(pool, state, il, &curn, start_pos, n_tok, &mut sub)?;
            for t in 0..n_tok {
                for s in 0..hc {
                    let p_s = post[t * hc + s];
                    let dst = &mut x_hc[(t * hc + s) * h..(t * hc + s + 1) * h];
                    for d in 0..h {
                        dst[d] += sub[t * h + d] * p_s;
                    }
                }
            }

            // MLP / LatentMoE HC Pre
            self.apply_hc_pre(
                pool,
                &format!("{l_pfx}.hc_mlp_layer"),
                &unit_ones,
                &x_hc,
                &mut fl,
                &mut cur,
                &mut post,
                n_tok,
            )?;
            let post_ln = self.load_vec(&format!("{l_pfx}.post_attention_layernorm.weight"))?;
            for t in 0..n_tok {
                rms_norm(
                    &mut curn[t * h..(t + 1) * h],
                    &cur[t * h..(t + 1) * h],
                    &post_ln,
                    c.rms_eps,
                );
            }
            self.mlp_sublayer(pool, il, &curn, n_tok, &mut sub)?;
            for t in 0..n_tok {
                for s in 0..hc {
                    let p_s = post[t * hc + s];
                    let dst = &mut x_hc[(t * hc + s) * h..(t * hc + s + 1) * h];
                    for d in 0..h {
                        dst[d] += sub[t * h + d] * p_s;
                    }
                }
            }
        }

        if compute_last_logits {
            let last_x = &x_hc[(n_tok - 1) * hc_dim..n_tok * hc_dim];
            let mut fl_last = vec![0.0f32; hc_dim];
            rms_norm(&mut fl_last, last_x, &unit_ones, c.rms_eps);
            let fn_w = self.load_vec("model.hc_head.hc_head_fn")?;
            let sc_w = self.load_vec("model.hc_head.hc_head_scale")?;
            let base_w = self.load_vec("model.hc_head.hc_head_base")?;
            let norm_w = self.load_vec("model.norm.weight")?;
            let mut hh = vec![0.0f32; h];
            for s in 0..hc {
                let mut mix = 0.0f32;
                for k in 0..hc_dim {
                    mix += fn_w[s * hc_dim + k] * fl_last[k];
                }
                let pre = 1.0 / (1.0 + (-(mix * sc_w[0] + base_w[s])).exp()) + c.hc_eps;
                for d in 0..h {
                    hh[d] += last_x[s * h + d] * pre;
                }
            }
            let mut hf = vec![0.0f32; h];
            rms_norm(&mut hf, &hh, &norm_w, c.rms_eps);
            self.linear_trunk(pool, "lm_head.weight", &hf, &mut state.logits, 1)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_hc_pre(
        &self,
        pool: &ThreadPool,
        prefix: &str,
        unit_ones: &[f32],
        x_hc: &[f32],
        fl: &mut [f32],
        cur: &mut [f32],
        post: &mut [f32],
        n_tok: usize,
    ) -> Result<(), ModelError> {
        let _ = pool;
        let c = &self.man.cfg;
        let h = c.n_embd;
        let hc = c.hc_count;
        let hc_dim = hc * h;
        let fn_w = self.load_vec(&format!("{prefix}.hc_pre.hc_fn"))?;
        let sc_w = self.load_vec(&format!("{prefix}.hc_pre.hc_scale"))?;
        let base_w = self.load_vec(&format!("{prefix}.hc_pre.hc_base"))?;

        for t in 0..n_tok {
            rms_norm(
                &mut fl[t * hc_dim..(t + 1) * hc_dim],
                &x_hc[t * hc_dim..(t + 1) * hc_dim],
                unit_ones,
                c.rms_eps,
            );
            let mut mix = vec![0.0f32; 2 * hc];
            for r in 0..(2 * hc) {
                let mut acc = 0.0f32;
                for k in 0..hc_dim {
                    acc += fn_w[r * hc_dim + k] * fl[t * hc_dim + k];
                }
                mix[r] = acc;
            }
            cur[t * h..(t + 1) * h].fill(0.0);
            for s in 0..hc {
                let pre_s = 1.0 / (1.0 + (-(mix[s] * sc_w[0] + base_w[s])).exp()) + c.hc_eps;
                post[t * hc + s] = c.hc_mag
                    * (1.0 / (1.0 + (-(mix[hc + s] * sc_w[1] + base_w[hc + s])).exp()))
                    + c.hc_eps;
                for d in 0..h {
                    cur[t * h + d] += x_hc[(t * hc + s) * h + d] * pre_s;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_sublayer(
        &self,
        pool: &ThreadPool,
        state: &mut Hy4State,
        il: usize,
        curn: &[f32],
        start_pos: usize,
        n_tok: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let c = &self.man.cfg;
        let swa = c.is_swa[il];
        let nh = if swa { c.heads_swa } else { c.heads_sparse };
        let theta = (if swa { c.theta_swa } else { c.theta_sparse }) as f32;
        let p = format!("model.layers.{il}.self_attn");

        let mut qa = vec![0.0f32; n_tok * c.q_lora];
        let mut qr = vec![0.0f32; n_tok * c.q_lora];
        self.linear_trunk(pool, &format!("{p}.q_a_proj.weight"), curn, &mut qa, n_tok)?;
        let q_ln = self.load_vec(&format!("{p}.q_a_layernorm.weight"))?;
        for t in 0..n_tok {
            rms_norm(
                &mut qr[t * c.q_lora..(t + 1) * c.q_lora],
                &qa[t * c.q_lora..(t + 1) * c.q_lora],
                &q_ln,
                c.rms_eps,
            );
        }
        let qd = nh * c.qk_head;
        let mut q = vec![0.0f32; n_tok * qd];
        self.linear_trunk(pool, &format!("{p}.q_b_proj.weight"), &qr, &mut q, n_tok)?;
        for t in 0..n_tok {
            for h_i in 0..nh {
                let qp =
                    &mut q[(t * nh + h_i) * c.qk_head + c.qk_nope..(t * nh + h_i + 1) * c.qk_head];
                rope_inplace(
                    qp,
                    1,
                    c.qk_rope,
                    c.qk_rope,
                    start_pos + t,
                    theta,
                    None,
                    RopeType::Norm,
                );
            }
        }

        let kva_dim = c.kv_lora + c.qk_rope;
        let mut kva = vec![0.0f32; n_tok * kva_dim];
        self.linear_trunk(
            pool,
            &format!("{p}.kv_a_proj_with_mqa.weight"),
            curn,
            &mut kva,
            n_tok,
        )?;
        let kv_ln = self.load_vec(&format!("{p}.kv_a_layernorm.weight"))?;
        let mut kvc = vec![0.0f32; n_tok * c.kv_lora];
        let mut kpe = vec![0.0f32; n_tok * c.qk_rope];
        for t in 0..n_tok {
            rms_norm(
                &mut kvc[t * c.kv_lora..(t + 1) * c.kv_lora],
                &kva[t * kva_dim..t * kva_dim + c.kv_lora],
                &kv_ln,
                c.rms_eps,
            );
            kpe[t * c.qk_rope..(t + 1) * c.qk_rope]
                .copy_from_slice(&kva[t * kva_dim + c.kv_lora..(t + 1) * kva_dim]);
            rope_inplace(
                &mut kpe[t * c.qk_rope..(t + 1) * c.qk_rope],
                1,
                c.qk_rope,
                c.qk_rope,
                start_pos + t,
                theta,
                None,
                RopeType::Norm,
            );
        }

        let ring_cap = if swa && c.window > 0 {
            c.window.min(state.max_ctx)
        } else {
            0
        };
        let kv_lat = c.kv_lora + c.qk_rope;
        // Matrix-Absorbed MLA: dequantize `kv_b_proj` (`[nh * (qk_nope + v_head), kv_lora]`) once per layer call
        let kvb_ti = self.man.req_tensor(&format!("{p}.kv_b_proj.weight"))?;
        let kvb_tb = &self.trunk[kvb_ti.offset as usize..(kvb_ti.offset + kvb_ti.bytes) as usize];
        let ex = c.qk_nope + c.v_head;
        let total_rows = nh * ex;
        let mut w_kvb = vec![0.0f32; total_rows * c.kv_lora];
        for r in 0..total_rows {
            dequant_hy4_row(
                kvb_ti.fmt,
                kvb_tb,
                total_rows,
                c.kv_lora,
                r,
                &mut w_kvb[r * c.kv_lora..(r + 1) * c.kv_lora],
            );
        }

        let sinks = self.load_vec(&format!("{p}.learnable_sink_param"))?;
        let scale = 1.0f32 / (c.qk_head as f32).sqrt();
        let vd = nh * c.v_head;
        let mut attn = vec![0.0f32; n_tok * vd];
        let mut qa_t = vec![0.0f32; kv_lat];
        let mut ctx_t = vec![0.0f32; c.kv_lora];

        // Causal token-by-token latent KV write + absorbed MQA so `n_tok > ring_cap` SWA wrap never clobbers earlier slots!
        for t in 0..n_tok {
            let pos = start_pos + t;
            let slot = if ring_cap > 0 { pos % ring_cap } else { pos };
            {
                let dst_lat = &mut state.latent_kv_caches[il][slot * kv_lat..(slot + 1) * kv_lat];
                for d in 0..c.kv_lora {
                    dst_lat[d] = f32_to_f16(kvc[t * c.kv_lora + d]);
                }
                for d in 0..c.qk_rope {
                    dst_lat[c.kv_lora + d] = f32_to_f16(kpe[t * c.qk_rope + d]);
                }
            }
            let kv_start = if swa && c.window > 0 && (pos + 1) > c.window {
                pos + 1 - c.window
            } else {
                0
            };

            for h_i in 0..nh {
                qa_t.fill(0.0);
                let qh = &q[(t * nh + h_i) * c.qk_head..(t * nh + h_i + 1) * c.qk_head];
                qa_t[c.kv_lora..kv_lat].copy_from_slice(&qh[c.qk_nope..c.qk_head]);
                for i in 0..c.qk_nope {
                    let qv = qh[i];
                    let wr = &w_kvb[(h_i * ex + i) * c.kv_lora..(h_i * ex + i + 1) * c.kv_lora];
                    for l in 0..c.kv_lora {
                        qa_t[l] += qv * wr[l];
                    }
                }

                let mut mx = sinks[h_i];
                let mut scores = Vec::with_capacity(pos - kv_start + 1);
                for p_t in kv_start..=pos {
                    let s_slot = if ring_cap > 0 { p_t % ring_cap } else { p_t };
                    let lat_s = &state.latent_kv_caches[il][s_slot * kv_lat..(s_slot + 1) * kv_lat];
                    let mut dot = 0.0f32;
                    for d in 0..kv_lat {
                        dot += qa_t[d] * f16_to_f32(lat_s[d]);
                    }
                    let sc = dot * scale;
                    mx = mx.max(sc);
                    scores.push(sc);
                }
                let mut denom = ((sinks[h_i] - mx) as f64).exp();
                for sc in &mut scores {
                    *sc = (*sc - mx).exp();
                    denom += *sc as f64;
                }
                ctx_t.fill(0.0);
                for (idx_p, &sc) in scores.iter().enumerate() {
                    let p_t = kv_start + idx_p;
                    let s_slot = if ring_cap > 0 { p_t % ring_cap } else { p_t };
                    let lat_s =
                        &state.latent_kv_caches[il][s_slot * kv_lat..s_slot * kv_lat + c.kv_lora];
                    let wt = (sc as f64 / denom) as f32;
                    for d in 0..c.kv_lora {
                        ctx_t[d] += wt * f16_to_f32(lat_s[d]);
                    }
                }

                for j in 0..c.v_head {
                    let wr = &w_kvb[(h_i * ex + c.qk_nope + j) * c.kv_lora
                        ..(h_i * ex + c.qk_nope + j + 1) * c.kv_lora];
                    let mut acc = 0.0f32;
                    for l in 0..c.kv_lora {
                        acc += wr[l] * ctx_t[l];
                    }
                    attn[(t * nh + h_i) * c.v_head + j] = acc;
                }
            }
        }

        let mut gate = vec![0.0f32; n_tok * vd];
        self.linear_trunk(
            pool,
            &format!("{p}.linear_gate.weight"),
            curn,
            &mut gate,
            n_tok,
        )?;
        for (a, &g) in attn.iter_mut().zip(gate.iter()) {
            *a *= 1.0 / (1.0 + (-g).exp());
        }
        let mut o_raw = vec![0.0f32; n_tok * c.n_embd];
        self.linear_trunk(
            pool,
            &format!("{p}.o_proj.weight"),
            &attn,
            &mut o_raw,
            n_tok,
        )?;
        let post_n = self.load_vec(&format!("{p}.post_proj_norm.weight"))?;
        for t in 0..n_tok {
            rms_norm(
                &mut out[t * c.n_embd..(t + 1) * c.n_embd],
                &o_raw[t * c.n_embd..(t + 1) * c.n_embd],
                &post_n,
                c.rms_eps,
            );
        }
        Ok(())
    }

    fn mlp_sublayer(
        &self,
        pool: &ThreadPool,
        il: usize,
        curn: &[f32],
        n_tok: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let c = &self.man.cfg;
        let h = c.n_embd;
        let p = format!("model.layers.{il}.mlp");

        if il == 0 {
            let ff = c.dense_ff;
            let mut g = vec![0.0f32; n_tok * ff];
            let mut u = vec![0.0f32; n_tok * ff];
            self.linear_trunk(pool, &format!("{p}.gate_proj.weight"), curn, &mut g, n_tok)?;
            self.linear_trunk(pool, &format!("{p}.up_proj.weight"), curn, &mut u, n_tok)?;
            for (gv, &uv) in g.iter_mut().zip(u.iter()) {
                *gv = (*gv / (1.0 + (-*gv).exp())) * uv;
            }
            let mut d = vec![0.0f32; n_tok * h];
            self.linear_trunk(pool, &format!("{p}.down_proj.weight"), &g, &mut d, n_tok)?;
            let fc2_n = self.load_vec(&format!("{p}.post_fc2_norm.weight"))?;
            for t in 0..n_tok {
                rms_norm(
                    &mut out[t * h..(t + 1) * h],
                    &d[t * h..(t + 1) * h],
                    &fc2_n,
                    c.rms_eps,
                );
            }
            return Ok(());
        }

        let ne = c.n_experts;
        let ku = c.top_k;
        let lat = c.lat_dim;
        let mut r_logits = vec![0.0f32; n_tok * ne];
        self.linear_trunk(
            pool,
            &format!("{p}.gate.weight"),
            curn,
            &mut r_logits,
            n_tok,
        )?;
        let e_bias = self.load_vec(&format!("{p}.gate.e_score_correction_bias"))?;

        let mut z = vec![0.0f32; n_tok * lat];
        self.linear_trunk(
            pool,
            &format!("{p}.routed_expert_down_proj.weight"),
            curn,
            &mut z,
            n_tok,
        )?;
        let mut z_out = vec![0.0f32; n_tok * lat];

        let moe_l = il - 1;
        let mut gu = vec![0.0f32; 2 * lat];
        let mut act = vec![0.0f32; lat];
        let mut ye = vec![0.0f32; lat];

        for t in 0..n_tok {
            let mut probs = vec![0.0f32; ne];
            let mut sel = vec![0.0f32; ne];
            for e in 0..ne {
                let pr = 1.0 / (1.0 + (-r_logits[t * ne + e]).exp());
                probs[e] = pr;
                sel[e] = pr + e_bias[e];
            }
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then_with(|| a.cmp(&b)));
            let top = &order[..ku];
            let sum_p: f32 = top.iter().map(|&e| probs[e]).sum();
            let z_t = &z[t * lat..(t + 1) * lat];

            for &exp_id in top {
                let wt = (probs[exp_id] / (sum_p + 1e-20)) * c.routed_scale;
                let rank = self.rank_of[moe_l * ne + exp_id] as usize;
                let ent = self.expert_dir[moe_l * ne + rank];
                let base_off = ent.body_offset as usize;
                let sq_bytes = lat * 2;
                let sq_raw = &self.sec4_arena[base_off..base_off + sq_bytes];
                let gu_off = base_off + sq_bytes;
                let dn_off = base_off + ent.dn_offset as usize;

                self.linear_raw(
                    pool,
                    ent.gu_code_id as i32,
                    &self.sec4_arena[gu_off..dn_off],
                    2 * lat,
                    lat,
                    z_t,
                    &mut gu,
                    1,
                );
                for d in 0..lat {
                    let sq_d = f16_to_f32(u16::from_le_bytes([sq_raw[2 * d], sq_raw[2 * d + 1]]));
                    let gv = gu[d];
                    let uv = gu[lat + d];
                    let sv = (gv / (1.0 + (-gv).exp())).min(c.swiglu_limit);
                    let uc = uv.clamp(-c.swiglu_limit, c.swiglu_limit);
                    act[d] = (sv * uc) / sq_d;
                }
                self.linear_raw(
                    pool,
                    ent.dn_code_id as i32,
                    &self.sec4_arena[dn_off..base_off + ent.body_bytes as usize],
                    lat,
                    lat,
                    &act,
                    &mut ye,
                    1,
                );
                for d in 0..lat {
                    z_out[t * lat + d] += wt * ye[d];
                }
            }
        }

        let mut moe_out = vec![0.0f32; n_tok * h];
        self.linear_trunk(
            pool,
            &format!("{p}.routed_expert_up_proj.weight"),
            &z_out,
            &mut moe_out,
            n_tok,
        )?;
        let sh_ff = self
            .man
            .req_tensor(&format!("{p}.shared_experts.gate_proj.weight"))?
            .rows();
        let mut sg = vec![0.0f32; n_tok * sh_ff];
        let mut su = vec![0.0f32; n_tok * sh_ff];
        let mut sd = vec![0.0f32; n_tok * h];
        self.linear_trunk(
            pool,
            &format!("{p}.shared_experts.gate_proj.weight"),
            curn,
            &mut sg,
            n_tok,
        )?;
        self.linear_trunk(
            pool,
            &format!("{p}.shared_experts.up_proj.weight"),
            curn,
            &mut su,
            n_tok,
        )?;
        for (gv, &uv) in sg.iter_mut().zip(su.iter()) {
            *gv = (*gv / (1.0 + (-*gv).exp())) * uv;
        }
        self.linear_trunk(
            pool,
            &format!("{p}.shared_experts.down_proj.weight"),
            &sg,
            &mut sd,
            n_tok,
        )?;
        for (m_v, &s_v) in moe_out.iter_mut().zip(sd.iter()) {
            *m_v += s_v;
        }
        let comb_n = self.load_vec(&format!("{p}.combine_output_norm.weight"))?;
        for t in 0..n_tok {
            rms_norm(
                &mut out[t * h..(t + 1) * h],
                &moe_out[t * h..(t + 1) * h],
                &comb_n,
                c.rms_eps,
            );
        }
        Ok(())
    }
}

impl Hy4State {
    pub fn reset(&mut self) {
        for lc in &mut self.latent_kv_caches {
            lc.fill(0);
        }
    }

    #[inline]
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

// ============================================================================
// Minimal zero-dependency JSON parser (`#![forbid(unsafe_code)]`)
// ============================================================================

enum JsonVal {
    Null,
    #[allow(dead_code)]
    Bool(bool),
    Num(f64, Option<i64>, Option<u64>),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(HashMap<String, JsonVal>),
}

impl JsonVal {
    fn get_key(&self, k: &str) -> Option<&JsonVal> {
        match self {
            JsonVal::Obj(m) => m.get(k),
            _ => None,
        }
    }

    fn req_key(&self, k: &str) -> Result<&JsonVal, ModelError> {
        self.get_key(k)
            .ok_or_else(|| ModelError::InvalidConfig(format!("manifest missing JSON key '{k}'")))
    }

    fn as_str(&self) -> Result<&str, ModelError> {
        match self {
            JsonVal::Str(s) => Ok(s),
            _ => Err(ModelError::InvalidConfig(
                "expected JSON string".to_string(),
            )),
        }
    }

    fn as_array(&self) -> Result<&[JsonVal], ModelError> {
        match self {
            JsonVal::Arr(a) => Ok(a),
            _ => Err(ModelError::InvalidConfig("expected JSON array".to_string())),
        }
    }

    fn as_f64(&self) -> Result<f64, ModelError> {
        match self {
            JsonVal::Num(f, _, _) => Ok(*f),
            _ => Err(ModelError::InvalidConfig(
                "expected JSON number".to_string(),
            )),
        }
    }

    fn as_i64(&self) -> Result<i64, ModelError> {
        match self {
            JsonVal::Num(_f, Some(i), _) => Ok(*i),
            JsonVal::Num(_f, None, Some(u)) => Ok(*u as i64),
            JsonVal::Num(f, _, _) => Ok(*f as i64),
            _ => Err(ModelError::InvalidConfig(
                "expected JSON integer".to_string(),
            )),
        }
    }

    fn as_u64(&self) -> Result<u64, ModelError> {
        match self {
            JsonVal::Num(_, _, Some(u)) => Ok(*u),
            JsonVal::Num(_, Some(i), _) => Ok(*i as u64),
            JsonVal::Num(f, _, _) => Ok(*f as u64),
            _ => Err(ModelError::InvalidConfig("expected JSON u64".to_string())),
        }
    }

    fn as_usize(&self) -> Result<usize, ModelError> {
        Ok(self.as_u64()? as usize)
    }
}

struct JCur<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JCur<'a> {
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), ModelError> {
        self.skip_ws();
        if self.b.get(self.i).copied() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(ModelError::InvalidConfig(format!(
                "manifest JSON expected '{}' at byte {}",
                c as char, self.i
            )))
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, ModelError> {
        let sl = self
            .b
            .get(self.i..self.i + 4)
            .ok_or_else(|| ModelError::InvalidConfig("short \\u escape".to_string()))?;
        self.i += 4;
        let s = core::str::from_utf8(sl)
            .map_err(|e| ModelError::InvalidConfig(format!("bad \\u escape: {e}")))?;
        u16::from_str_radix(s, 16)
            .map_err(|e| ModelError::InvalidConfig(format!("bad \\u hex: {e}")))
    }

    fn parse_str(&mut self) -> Result<String, ModelError> {
        self.expect(b'"')?;
        let mut out = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c == b'"' {
                self.i += 1;
                return Ok(out);
            }
            if c != b'\\' {
                let st = self.i;
                self.i += 1;
                while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                    self.i += 1;
                }
                let s = core::str::from_utf8(&self.b[st..self.i])
                    .map_err(|e| ModelError::InvalidConfig(e.to_string()))?;
                out.push_str(s);
                continue;
            }
            self.i += 1;
            let esc = *self
                .b
                .get(self.i)
                .ok_or_else(|| ModelError::InvalidConfig("EOF in escape".to_string()))?;
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
                    return Err(ModelError::InvalidConfig(format!(
                        "bad escape \\{}",
                        other as char
                    )));
                }
            }
        }
        Err(ModelError::InvalidConfig(
            "unterminated JSON string".to_string(),
        ))
    }

    fn parse_val(&mut self) -> Result<JsonVal, ModelError> {
        let Some(c) = self.peek() else {
            return Err(ModelError::InvalidConfig("unexpected JSON EOF".to_string()));
        };
        match c {
            b'"' => Ok(JsonVal::Str(self.parse_str()?)),
            b'{' => {
                self.i += 1;
                let mut m = HashMap::new();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(JsonVal::Obj(m));
                }
                loop {
                    let k = self.parse_str()?;
                    self.expect(b':')?;
                    let v = self.parse_val()?;
                    m.insert(k, v);
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        self.expect(b'}')?;
                        return Ok(JsonVal::Obj(m));
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(JsonVal::Arr(a));
                }
                loop {
                    a.push(self.parse_val()?);
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        self.expect(b']')?;
                        return Ok(JsonVal::Arr(a));
                    }
                }
            }
            b't' if self.b.get(self.i..self.i + 4) == Some(b"true") => {
                self.i += 4;
                Ok(JsonVal::Bool(true))
            }
            b'f' if self.b.get(self.i..self.i + 5) == Some(b"false") => {
                self.i += 5;
                Ok(JsonVal::Bool(false))
            }
            b'n' if self.b.get(self.i..self.i + 4) == Some(b"null") => {
                self.i += 4;
                Ok(JsonVal::Null)
            }
            _ => {
                let st = self.i;
                while self.i < self.b.len()
                    && !matches!(
                        self.b[self.i],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n'
                    )
                {
                    self.i += 1;
                }
                let s = core::str::from_utf8(&self.b[st..self.i])
                    .map_err(|e| ModelError::InvalidConfig(e.to_string()))?;
                let ui = s.parse::<u64>().ok();
                let si = s.parse::<i64>().ok();
                let fl = s.parse::<f64>().map_err(|e| {
                    ModelError::InvalidConfig(format!("bad JSON number '{s}': {e}"))
                })?;
                Ok(JsonVal::Num(fl, si, ui))
            }
        }
    }
}

fn parse_json_value(json_text: &str) -> Result<JsonVal, ModelError> {
    let mut cur = JCur {
        b: json_text.as_bytes(),
        i: 0,
    };
    cur.parse_val()
}
