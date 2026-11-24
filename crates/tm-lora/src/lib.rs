#![forbid(unsafe_code)]

//! GGUF LoRA adapter loading, multi-adapter scaling, dynamic side-branch execution,
//! and lossless/quantized offline merging (`ADR-015`).
//!
//! Binary schema matches upstream `llama.cpp` (`src/llama-adapter.cpp` & `convert_lora_to_gguf.py`):
//! - Metadata:
//!   - `general.type = "adapter"`
//!   - `general.architecture = "<arch>"`
//!   - `adapter.type = "lora"`
//!   - `adapter.lora.alpha = <f32>` (`0.0` / absent means `scale = adapter_scale`)
//! - Tensor pairs for base 2D weight `<name>` (`dims = [d_in, d_out]`):
//!   - `<name>.lora_a` with `dims = [d_in, r]` (`r` rows of `d_in` contiguous columns)
//!   - `<name>.lora_b` with `dims = [r, d_out]` (`d_out` rows of `r` contiguous columns)
//!   - Exception (`token_embd.weight`): `lora_a` has `dims = [r, vocab]`, `lora_b` has `dims = [r, d_in]`.

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;
use tm_core::{
    DType, dequantize_q4_0x4_single_row, dequantize_row_bytes, f32_to_f16, quantize_row_to_bytes,
    repack_q4_0_bytes_to_q4_0x4,
};
use tm_cpu::{KernelTable, MAX_CPU_BATCH, ThreadPool};
use tm_gguf::{GgufError, GgufFile, MetadataValue, NewGgufTensor, write_gguf_file_atomic};

/// Errors produced when loading, binding, saving, or merging GGUF LoRA adapters.
#[derive(Debug, Error)]
pub enum LoraError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error("LoRA '{path}': expected general.type='adapter', got '{got}'")]
    NotAnAdapter { path: String, got: String },
    #[error("LoRA '{path}': expected adapter.type='lora', got '{got}'")]
    UnsupportedAdapterType { path: String, got: String },
    #[error("LoRA '{path}': architecture mismatch (base='{base}', adapter='{adapter}')")]
    ArchMismatch {
        path: String,
        base: String,
        adapter: String,
    },
    #[error("LoRA '{path}': unexpected tensor suffix on '{tensor}' (expected .lora_a or .lora_b)")]
    UnexpectedTensorSuffix { path: String, tensor: String },
    #[error("LoRA '{path}': incomplete pair for base tensor '{base_tensor}' (missing {missing})")]
    IncompletePair {
        path: String,
        base_tensor: String,
        missing: &'static str,
    },
    #[error(
        "LoRA '{path}': invalid shape for '{base_tensor}' (lora_a={a_dims:?}, lora_b={b_dims:?})"
    )]
    InvalidPairShape {
        path: String,
        base_tensor: String,
        a_dims: Vec<usize>,
        b_dims: Vec<usize>,
    },
    #[error("LoRA '{path}': target tensor '{base_tensor}' does not exist in base model")]
    UnknownBaseTensor { path: String, base_tensor: String },
    #[error(
        "LoRA '{path}': shape mismatch against base tensor '{base_tensor}' (base={base_dims:?}, lora_in={d_in}, lora_out={d_out})"
    )]
    BaseShapeMismatch {
        path: String,
        base_tensor: String,
        base_dims: Vec<usize>,
        d_in: usize,
        d_out: usize,
    },
    #[error("LoRA dynamic binding is not yet supported on architecture '{0}'")]
    UnsupportedBaseArch(String),
}

/// Decoded `(A, B)` factor pair for a single base weight tensor.
#[derive(Debug, Clone)]
pub struct LoraWeightPair {
    /// Base model tensor name (e.g. `"blk.0.attn_q.weight"` or `"token_embd.weight"`).
    pub base_name: String,
    /// True when `base_name == "token_embd.weight"` (embedding lookup layout: `a=[vocab, r]`, `b=[d_in, r]`).
    pub is_token_embd: bool,
    /// LoRA bottleneck rank `r`.
    pub rank: usize,
    /// Input feature dimension (`d_in` for linear weights; `vocab_size` for `token_embd.weight`).
    pub d_in: usize,
    /// Output feature dimension (`d_out` for linear weights; `n_embd` for `token_embd.weight`).
    pub d_out: usize,
    /// Row-major `FP32` `lora_a`:
    /// - Linear weight: `[rank, d_in]` (`rank` rows of `d_in` floats).
    /// - `token_embd.weight`: `[vocab, rank]` (`vocab` rows of `rank` floats).
    pub a: Vec<f32>,
    /// Row-major `FP32` `lora_b`: `[d_out, rank]` (`d_out` rows of `rank` floats).
    pub b: Vec<f32>,
}

impl LoraWeightPair {
    /// Effective LoRA scaling factor matching `llama_adapter_lora_weight::get_scale` (`llama-adapter.h:53`):
    /// `if alpha != 0.0 { adapter_scale * alpha / rank } else { adapter_scale }`.
    #[inline]
    pub fn effective_scale(&self, alpha: f32, adapter_scale: f32) -> f32 {
        if alpha != 0.0 {
            adapter_scale * alpha / (self.rank as f32)
        } else {
            adapter_scale
        }
    }
}

/// A parsed GGUF LoRA adapter (`general.type == "adapter"`, `adapter.type == "lora"`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub arch: String,
    pub alpha: f32,
    pub name: Option<String>,
    pub pairs: HashMap<String, LoraWeightPair>,
    pub pair_order: Vec<String>,
}

fn decode_gguf_2d_to_f32(
    gguf: &GgufFile,
    tensor_name: &str,
    cols: usize,
    rows: usize,
) -> Result<Vec<f32>, GgufError> {
    let (info, bytes) = gguf.tensor_bytes(tensor_name, "lora")?;
    let mut out = vec![0.0f32; rows * cols];
    if info.dtype == DType::Q4_0x4 {
        for r in 0..rows {
            dequantize_q4_0x4_single_row(bytes, cols, r, &mut out[r * cols..(r + 1) * cols]);
        }
    } else {
        let rb = info.dtype.row_bytes(cols);
        for r in 0..rows {
            dequantize_row_bytes(
                info.dtype,
                &bytes[r * rb..(r + 1) * rb],
                &mut out[r * cols..(r + 1) * cols],
            );
        }
    }
    Ok(out)
}

impl LoraAdapter {
    /// Opens and validates a GGUF LoRA adapter from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LoraError> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Parses a [`LoraAdapter`] from an already-opened [`GgufFile`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, LoraError> {
        let path_str = gguf.path().display().to_string();
        let general_type = gguf
            .get_meta("general.type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if general_type != "adapter" {
            return Err(LoraError::NotAnAdapter {
                path: path_str,
                got: general_type.to_string(),
            });
        }
        let adapter_type = gguf
            .get_meta("adapter.type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if adapter_type != "lora" {
            return Err(LoraError::UnsupportedAdapterType {
                path: path_str,
                got: adapter_type.to_string(),
            });
        }
        let arch = gguf.req_str("general.architecture")?.to_string();
        let alpha = gguf
            .get_meta("adapter.lora.alpha")
            .and_then(|v| v.as_f32())
            .unwrap_or(0.0);
        let name = gguf
            .get_meta("general.name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        struct RawSlot {
            a_name: Option<String>,
            b_name: Option<String>,
        }
        let mut raw_map: HashMap<String, RawSlot> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        for tname in gguf.tensor_names() {
            if let Some(base) = tname.strip_suffix(".lora_a") {
                let entry = raw_map.entry(base.to_string()).or_insert_with(|| {
                    order.push(base.to_string());
                    RawSlot {
                        a_name: None,
                        b_name: None,
                    }
                });
                entry.a_name = Some(tname.clone());
            } else if let Some(base) = tname.strip_suffix(".lora_b") {
                let entry = raw_map.entry(base.to_string()).or_insert_with(|| {
                    order.push(base.to_string());
                    RawSlot {
                        a_name: None,
                        b_name: None,
                    }
                });
                entry.b_name = Some(tname.clone());
            } else if tname.ends_with("_norm.weight") {
                // Match `llama-adapter.cpp:290`: optional norm vectors ignored cleanly.
                continue;
            } else {
                return Err(LoraError::UnexpectedTensorSuffix {
                    path: path_str,
                    tensor: tname.clone(),
                });
            }
        }

        let mut pairs = HashMap::with_capacity(order.len());
        for base in &order {
            let slot = &raw_map[base];
            let a_name = slot
                .a_name
                .as_deref()
                .ok_or_else(|| LoraError::IncompletePair {
                    path: path_str.clone(),
                    base_tensor: base.clone(),
                    missing: "lora_a",
                })?;
            let b_name = slot
                .b_name
                .as_deref()
                .ok_or_else(|| LoraError::IncompletePair {
                    path: path_str.clone(),
                    base_tensor: base.clone(),
                    missing: "lora_b",
                })?;
            let a_info = gguf.tensor_info(a_name).expect("a_name from tensor_names");
            let b_info = gguf.tensor_info(b_name).expect("b_name from tensor_names");
            if a_info.dims.len() != 2 || b_info.dims.len() != 2 {
                return Err(LoraError::InvalidPairShape {
                    path: path_str,
                    base_tensor: base.clone(),
                    a_dims: a_info.dims.clone(),
                    b_dims: b_info.dims.clone(),
                });
            }

            let is_token_embd = base.ends_with("token_embd.weight");
            if is_token_embd {
                // `llama-adapter.cpp:365`: `a` is `[r, vocab]`, `b` is `[r, d_in]`.
                let rank = a_info.dims[0];
                let vocab = a_info.dims[1];
                if rank == 0 || b_info.dims[0] != rank {
                    return Err(LoraError::InvalidPairShape {
                        path: path_str,
                        base_tensor: base.clone(),
                        a_dims: a_info.dims.clone(),
                        b_dims: b_info.dims.clone(),
                    });
                }
                let d_out = b_info.dims[1];
                let a = decode_gguf_2d_to_f32(gguf, a_name, rank, vocab)?;
                let b = decode_gguf_2d_to_f32(gguf, b_name, rank, d_out)?;
                pairs.insert(
                    base.clone(),
                    LoraWeightPair {
                        base_name: base.clone(),
                        is_token_embd: true,
                        rank,
                        d_in: vocab,
                        d_out,
                        a,
                        b,
                    },
                );
            } else {
                // `llama-adapter.cpp:370`: `a` is `[d_in, r]`, `b` is `[r, d_out]`.
                let d_in = a_info.dims[0];
                let rank = a_info.dims[1];
                let b_r = b_info.dims[0];
                let d_out = b_info.dims[1];
                if rank == 0 || rank != b_r || d_in == 0 || d_out == 0 {
                    return Err(LoraError::InvalidPairShape {
                        path: path_str,
                        base_tensor: base.clone(),
                        a_dims: a_info.dims.clone(),
                        b_dims: b_info.dims.clone(),
                    });
                }
                let a = decode_gguf_2d_to_f32(gguf, a_name, d_in, rank)?;
                let b = decode_gguf_2d_to_f32(gguf, b_name, rank, d_out)?;
                pairs.insert(
                    base.clone(),
                    LoraWeightPair {
                        base_name: base.clone(),
                        is_token_embd: false,
                        rank,
                        d_in,
                        d_out,
                        a,
                        b,
                    },
                );
            }
        }

        Ok(Self {
            arch,
            alpha,
            name,
            pairs,
            pair_order: order,
        })
    }

    /// Writes this adapter to a crash-safe `llama.cpp`-compatible GGUF v3 file (`F32` or `F16` tensors).
    pub fn save_gguf<P: AsRef<Path>>(
        &self,
        dst_path: P,
        use_f16: bool,
    ) -> Result<usize, LoraError> {
        let mut meta = vec![
            (
                "general.type".to_string(),
                MetadataValue::String("adapter".to_string()),
            ),
            (
                "general.architecture".to_string(),
                MetadataValue::String(self.arch.clone()),
            ),
            (
                "adapter.type".to_string(),
                MetadataValue::String("lora".to_string()),
            ),
            (
                "adapter.lora.alpha".to_string(),
                MetadataValue::F32(self.alpha),
            ),
        ];
        if let Some(ref n) = self.name {
            meta.push(("general.name".to_string(), MetadataValue::String(n.clone())));
        }

        let out_dtype = if use_f16 { DType::F16 } else { DType::F32 };
        let encode_slice = |vals: &[f32]| -> Vec<u8> {
            if use_f16 {
                let mut buf = Vec::with_capacity(vals.len() * 2);
                for &v in vals {
                    buf.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                }
                buf
            } else {
                let mut buf = Vec::with_capacity(vals.len() * 4);
                for &v in vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            }
        };

        struct Prepared {
            name: String,
            dims: [usize; 2],
            bytes: Vec<u8>,
        }
        let mut prepared = Vec::with_capacity(self.pair_order.len() * 2);
        for base in &self.pair_order {
            let p = &self.pairs[base];
            let a_dims = if p.is_token_embd {
                [p.rank, p.d_in]
            } else {
                [p.d_in, p.rank]
            };
            let b_dims = [p.rank, p.d_out];
            prepared.push(Prepared {
                name: format!("{base}.lora_a"),
                dims: a_dims,
                bytes: encode_slice(&p.a),
            });
            prepared.push(Prepared {
                name: format!("{base}.lora_b"),
                dims: b_dims,
                bytes: encode_slice(&p.b),
            });
        }

        let views: Vec<NewGgufTensor<'_>> = prepared
            .iter()
            .map(|t| NewGgufTensor {
                name: &t.name,
                dims: &t.dims,
                dtype: out_dtype,
                data: &t.bytes,
            })
            .collect();
        Ok(write_gguf_file_atomic(dst_path, &meta, &views, 32)?)
    }
}

/// Parse CLI `--lora <path>[:<scale>]` specification (supports Windows drive letters `C:\...`).
pub fn parse_lora_spec(spec: &str) -> Result<(String, f32), String> {
    if let Some((left, right)) = spec.rsplit_once(':')
        && !left.is_empty()
        && let Ok(scale) = right.parse::<f32>()
    {
        if !scale.is_finite() {
            return Err(format!("invalid non-finite LoRA scale in '{spec}'"));
        }
        return Ok((left.to_string(), scale));
    }
    Ok((spec.to_string(), 1.0))
}

/// Ordered collection of `(LoraAdapter, user_scale)` pairs applied additively at runtime.
#[derive(Debug, Clone, Default)]
pub struct LoraSet {
    pub adapters: Vec<(LoraAdapter, f32)>,
}

impl LoraSet {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    pub fn push(&mut self, adapter: LoraAdapter, scale: f32) {
        self.adapters.push((adapter, scale));
    }

    /// Validates all loaded adapters against `base_gguf` architecture and tensor dimensions.
    pub fn validate_against_gguf(&self, base_gguf: &GgufFile) -> Result<(), LoraError> {
        let base_arch = base_gguf.req_str("general.architecture")?;
        let has_separate_output = base_gguf.tensor_info("output.weight").is_some();

        for (adapter, _) in &self.adapters {
            if adapter.arch != base_arch {
                return Err(LoraError::ArchMismatch {
                    path: base_gguf.path().display().to_string(),
                    base: base_arch.to_string(),
                    adapter: adapter.arch.clone(),
                });
            }
            for base_name in &adapter.pair_order {
                let p = &adapter.pairs[base_name];
                let lookup_name = if base_name == "output.weight" && !has_separate_output {
                    "token_embd.weight"
                } else {
                    base_name.as_str()
                };
                let Some(info) = base_gguf.tensor_info(lookup_name) else {
                    return Err(LoraError::UnknownBaseTensor {
                        path: base_gguf.path().display().to_string(),
                        base_tensor: base_name.clone(),
                    });
                };
                if info.dims.len() != 2 {
                    return Err(LoraError::BaseShapeMismatch {
                        path: base_gguf.path().display().to_string(),
                        base_tensor: base_name.clone(),
                        base_dims: info.dims.clone(),
                        d_in: p.d_in,
                        d_out: p.d_out,
                    });
                }
                let (exp_in, exp_out) = if p.is_token_embd {
                    (info.dims[1], info.dims[0])
                } else {
                    (info.dims[0], info.dims[1])
                };
                if p.d_in != exp_in || p.d_out != exp_out {
                    return Err(LoraError::BaseShapeMismatch {
                        path: base_gguf.path().display().to_string(),
                        base_tensor: base_name.clone(),
                        base_dims: info.dims.clone(),
                        d_in: p.d_in,
                        d_out: p.d_out,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Pre-bound runtime entry for one `(A, B, effective_scale)` factor on a specific model slot.
#[derive(Debug, Clone)]
pub struct BoundLoraEntry {
    pub scale: f32,
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub is_token_embd: bool,
    pub a: Vec<f32>,
    /// Stored transposed as `[rank, d_out]` (`rank` rows of `d_out` contiguous floats!) so
    /// `y += sum_r h[r] * b_t[r, 0..d_out]` executes contiguous SIMD `(kt.axpy_f32)`!
    pub b_t: Vec<f32>,
}

/// O(1) indexed LoRA chains bound to a loaded model's weight slots.
#[derive(Debug, Clone, Default)]
pub struct BoundLora {
    pub chains: Vec<Vec<BoundLoraEntry>>,
    pub name_to_slot: HashMap<String, usize>,
    pub max_rank: usize,
}

/// Pre-allocated scratch buffer for dynamic LoRA execution (zero heap allocation on hot path!).
#[derive(Debug, Clone, Default)]
pub struct LoraScratch {
    pub hidden: Vec<f32>,
}

impl LoraScratch {
    pub fn new(max_rank: usize) -> Self {
        Self {
            hidden: vec![0.0; MAX_CPU_BATCH * max_rank.max(1)],
        }
    }

    pub fn ensure_rank(&mut self, max_rank: usize) {
        let need = MAX_CPU_BATCH * max_rank.max(1);
        if self.hidden.len() < need {
            self.hidden.resize(need, 0.0);
        }
    }
}

impl BoundLora {
    /// Compiles a [`LoraSet`] into contiguous SIMD-friendly chains indexed by tensor name.
    pub fn from_lora_set(set: &LoraSet) -> Self {
        if set.is_empty() {
            return Self::default();
        }
        let mut chains: Vec<Vec<BoundLoraEntry>> = Vec::new();
        let mut name_to_slot: HashMap<String, usize> = HashMap::new();
        let mut max_rank = 0usize;

        for (adapter, user_scale) in &set.adapters {
            if *user_scale == 0.0 {
                continue;
            }
            for base_name in &adapter.pair_order {
                let p = &adapter.pairs[base_name];
                let scale = p.effective_scale(adapter.alpha, *user_scale);
                if scale == 0.0 {
                    continue;
                }
                max_rank = max_rank.max(p.rank);
                // Transpose `b` from `[d_out, rank]` -> `b_t: [rank, d_out]` once at bind time
                // so runtime `y += h[r] * b_t[r]` streams `d_out` floats contiguously via `axpy_f32`!
                let mut b_t = vec![0.0f32; p.rank * p.d_out];
                for o in 0..p.d_out {
                    let src_row = &p.b[o * p.rank..(o + 1) * p.rank];
                    for (r, &val) in src_row.iter().enumerate() {
                        b_t[r * p.d_out + o] = val;
                    }
                }
                let slot = *name_to_slot.entry(base_name.clone()).or_insert_with(|| {
                    let idx = chains.len();
                    chains.push(Vec::new());
                    idx
                });
                chains[slot].push(BoundLoraEntry {
                    scale,
                    rank: p.rank,
                    d_in: p.d_in,
                    d_out: p.d_out,
                    is_token_embd: p.is_token_embd,
                    a: p.a.clone(),
                    b_t,
                });
            }
        }

        Self {
            chains,
            name_to_slot,
            max_rank,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    #[inline]
    pub fn slot_for(&self, tensor_name: &str) -> Option<usize> {
        self.name_to_slot.get(tensor_name).copied()
    }

    /// Applies `token_embd.weight` LoRA delta (`out_embd += scale * B * A[:, tok]`) for 1 token.
    /// Zero heap allocations.
    #[inline]
    pub fn apply_embed_row(
        &self,
        slot: Option<usize>,
        kt: &KernelTable,
        tok: usize,
        out_embd: &mut [f32],
    ) {
        let Some(idx) = slot else {
            return;
        };
        for e in &self.chains[idx] {
            if !e.is_token_embd {
                continue;
            }
            debug_assert!(tok < e.d_in);
            debug_assert_eq!(out_embd.len(), e.d_out);
            let a_row = &e.a[tok * e.rank..(tok + 1) * e.rank];
            for (r, &ar) in a_row.iter().enumerate() {
                let w = ar * e.scale;
                if w != 0.0 {
                    (kt.axpy_f32)(out_embd, w, &e.b_t[r * e.d_out..(r + 1) * e.d_out]);
                }
            }
        }
    }

    /// Applies linear LoRA delta (`y += scale * (x @ A^T) @ B^T`) for `n_tok` (`1..=MAX_CPU_BATCH`) tokens.
    ///
    /// Preserves **0-ULP batch-width invariance (`L1b`)** and **thread-count determinism (`N-10`)**:
    /// each token's `h[r] = dot(a[r], x_t)` uses identical sequential FMA order, and `y_t` accumulates
    /// `r = 0..rank` via `(kt.axpy_f32)`. Zero heap allocations!
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_linear(
        &self,
        slot: Option<usize>,
        pool: &ThreadPool,
        kt: &KernelTable,
        x: &[f32],
        y: &mut [f32],
        n_tok: usize,
        scratch: &mut LoraScratch,
    ) {
        let slice_rows = y.len().checked_div(n_tok).unwrap_or(0);
        self.apply_linear_slice(slot, pool, kt, x, y, n_tok, 0, slice_rows, scratch);
    }

    /// Applies linear LoRA delta to an output row sub-slice `row_offset .. row_offset + slice_rows`
    /// of a (potentially fused, e.g. `attn_qkv.weight` or `ffn_up.weight`) base weight tensor.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn apply_linear_slice(
        &self,
        slot: Option<usize>,
        _pool: &ThreadPool,
        kt: &KernelTable,
        x: &[f32],
        y: &mut [f32],
        n_tok: usize,
        row_offset: usize,
        slice_rows: usize,
        scratch: &mut LoraScratch,
    ) {
        let Some(idx) = slot else {
            return;
        };
        debug_assert!(n_tok <= MAX_CPU_BATCH);
        debug_assert_eq!(y.len(), n_tok * slice_rows);
        for e in &self.chains[idx] {
            if e.is_token_embd {
                continue;
            }
            debug_assert_eq!(x.len(), n_tok * e.d_in);
            debug_assert!(row_offset + slice_rows <= e.d_out);
            let h_all = &mut scratch.hidden[..n_tok * e.rank];
            for t in 0..n_tok {
                let xt = &x[t * e.d_in..(t + 1) * e.d_in];
                let ht = &mut h_all[t * e.rank..(t + 1) * e.rank];
                for (r, hr_dst) in ht.iter_mut().enumerate() {
                    let ar = &e.a[r * e.d_in..(r + 1) * e.d_in];
                    let mut sum = 0.0f32;
                    for (&ai, &xi) in ar.iter().zip(xt.iter()) {
                        sum = ai.mul_add(xi, sum);
                    }
                    *hr_dst = sum * e.scale;
                }
                let yt = &mut y[t * slice_rows..(t + 1) * slice_rows];
                for (r, &hr) in ht.iter().enumerate() {
                    if hr != 0.0 {
                        let b_off = r * e.d_out + row_offset;
                        (kt.axpy_f32)(yt, hr, &e.b_t[b_off..b_off + slice_rows]);
                    }
                }
            }
        }
    }
}

/// Options for offline merging (`tinymodel lora-merge`).
#[derive(Debug, Clone, Default)]
pub struct LoraMergeOptions {
    /// Optional target quantization for 2D weights (defaults to preserving each base tensor's `DType`).
    pub quant: Option<DType>,
    /// Optional separate quantization for `output.weight` / tied `lm_head`.
    pub lm_head_quant: Option<DType>,
}

/// Summary returned by [`merge_lora_gguf`].
#[derive(Debug, Clone)]
pub struct LoraMergeReport {
    pub total_tensors: usize,
    pub merged_tensors: usize,
    pub untied_output: bool,
    pub output_file_bytes: usize,
}

fn choose_merged_dtype(
    req: Option<DType>,
    cols: usize,
    rows: usize,
    orig: DType,
    has_lora_delta: bool,
    is_token_embd: bool,
) -> DType {
    let target = match req {
        Some(q) => {
            // Leave untouched `token_embd.weight` in its base dtype unless lossless `F32`/`Q4_0` requested
            if is_token_embd && !has_lora_delta && !matches!(q, DType::F32 | DType::Q4_0) {
                return orig;
            }
            match q {
                DType::Q4_1 | DType::Q5K | DType::Q6K => DType::Q8_0,
                DType::BF16 => DType::F16,
                other => other,
            }
        }
        None => {
            if !has_lora_delta {
                return orig;
            }
            if is_token_embd {
                match orig {
                    DType::Q4_1 | DType::Q5K | DType::Q6K => DType::Q8_0,
                    DType::BF16 => DType::F16,
                    DType::Q4_0x4 => DType::Q4_0,
                    other => other,
                }
            } else {
                // Promote LoRA-modified sub-8-bit linear weights to `Q8_0` by default to prevent
                // 4-bit double-quantization loss while keeping CPU/Metal/Vulkan compatibility.
                match orig {
                    DType::F32 | DType::F16 => orig,
                    DType::BF16 => DType::F16,
                    _ => DType::Q8_0,
                }
            }
        }
    };
    match target {
        DType::Q4_0x4 if !(cols.is_multiple_of(32) && rows.is_multiple_of(4)) => DType::Q4_0,
        DType::Q2_0 if !cols.is_multiple_of(64) => DType::Iq4Nl,
        DType::TQ2_0 | DType::Iq2Xxs | DType::Iq3Xxs | DType::Q5K | DType::Q6K
            if !cols.is_multiple_of(256) =>
        {
            if cols.is_multiple_of(64) {
                DType::Q2_0
            } else {
                DType::Q4_0
            }
        }
        other => other,
    }
}

/// Merges a [`LoraSet`] offline into `base_gguf` (`W' = W + sum_k s_k B_k A_k`) and writes a
/// standalone GGUF v3 model to `dst_path` with zero runtime overhead.
///
/// If `base_gguf` has tied embeddings (no `output.weight`) and `loras` modifies `"output.weight"`
/// differently from `"token_embd.weight"`, `output.weight` is automatically materialized (untied).
pub fn merge_lora_gguf<P: AsRef<Path>>(
    base_gguf: &GgufFile,
    loras: &LoraSet,
    dst_path: P,
    opts: &LoraMergeOptions,
) -> Result<LoraMergeReport, LoraError> {
    loras.validate_against_gguf(base_gguf)?;
    let bound = BoundLora::from_lora_set(loras);
    let kt = tm_cpu::active_kernels();

    let has_separate_output = base_gguf.tensor_info("output.weight").is_some();
    let untie_output = !has_separate_output && bound.slot_for("output.weight").is_some();

    let mut out_names: Vec<String> = base_gguf.tensor_names().to_vec();
    if untie_output {
        out_names.push("output.weight".to_string());
    }

    // Materialize metadata (cloning string arrays cleanly so `write_gguf_file_atomic` needs no raw backing)
    let mut meta_vec: Vec<(String, MetadataValue)> = Vec::with_capacity(base_gguf.metadata().len());
    for (k, v) in base_gguf.metadata() {
        let cloned = match v {
            MetadataValue::StringSpans { .. } => {
                if let Some((_, it)) = base_gguf.string_array_iter(k) {
                    MetadataValue::Array(it.map(|s| MetadataValue::String(s.to_string())).collect())
                } else {
                    continue;
                }
            }
            other => other.clone(),
        };
        meta_vec.push((k.clone(), cloned));
    }

    struct BuiltTensor {
        name: String,
        dims: Vec<usize>,
        dtype: DType,
        bytes: Vec<u8>,
    }

    let mut built: Vec<BuiltTensor> = Vec::with_capacity(out_names.len());
    let mut merged_tensors = 0usize;
    let mut f32_row: Vec<f32> = Vec::new();

    for name in &out_names {
        let src_lookup = if name == "output.weight" && untie_output {
            "token_embd.weight"
        } else {
            name.as_str()
        };
        let (info, src_bytes) = base_gguf.tensor_bytes(src_lookup, "merge")?;

        let slot = bound.slot_for(name);
        let is_2d = info.dims.len() == 2 && name != "per_layer_token_embd.weight";
        let is_head = name == "output.weight"
            || (name == "token_embd.weight" && !has_separate_output && !untie_output);
        let req_dt = if is_2d {
            if is_head {
                opts.lm_head_quant.or(opts.quant)
            } else {
                opts.quant
            }
        } else {
            None
        };
        let dst_dtype = if is_2d {
            choose_merged_dtype(
                req_dt,
                info.dims[0],
                info.dims[1],
                info.dtype,
                slot.is_some(),
                name.ends_with("token_embd.weight"),
            )
        } else {
            info.dtype
        };

        if slot.is_none() && dst_dtype == info.dtype {
            built.push(BuiltTensor {
                name: name.clone(),
                dims: info.dims.clone(),
                dtype: info.dtype,
                bytes: src_bytes.to_vec(),
            });
            continue;
        }

        if slot.is_some() {
            merged_tensors += 1;
        }
        let cols = info.dims[0];
        let rows = info.dims[1];
        f32_row.resize(cols, 0.0);

        let decode_row = |r: usize, dst: &mut [f32]| {
            if info.dtype == DType::Q4_0x4 {
                dequantize_q4_0x4_single_row(src_bytes, cols, r, dst);
            } else {
                let s_rb = info.dtype.row_bytes(cols);
                dequantize_row_bytes(info.dtype, &src_bytes[r * s_rb..(r + 1) * s_rb], dst);
            }
        };

        let add_lora_to_row = |r: usize, row_vals: &mut [f32]| {
            let Some(s_idx) = slot else {
                return;
            };
            for e in &bound.chains[s_idx] {
                if e.is_token_embd {
                    // `token_embd.weight` row `r` (`r` is vocab token ID, `row_vals` has length `d_out = cols`!):
                    let a_tok = &e.a[r * e.rank..(r + 1) * e.rank];
                    for (rk, &ar) in a_tok.iter().enumerate() {
                        let w = ar * e.scale;
                        for (c, v) in row_vals.iter_mut().enumerate() {
                            *v = e.b_t[rk * e.d_out + c].mul_add(w, *v);
                        }
                    }
                } else {
                    // Linear weight row `r` (`0..d_out`), `row_vals` has length `d_in = cols`:
                    // `W'[r, :] += scale * sum_{rk} B[r, rk] * A[rk, :]` (`B[r, rk] == b_t[rk * d_out + r]`).
                    for rk in 0..e.rank {
                        let b_val = e.b_t[rk * e.d_out + r] * e.scale;
                        if b_val != 0.0 {
                            (kt.axpy_f32)(row_vals, b_val, &e.a[rk * e.d_in..(rk + 1) * e.d_in]);
                        }
                    }
                }
            }
        };

        let out_bytes = if dst_dtype == DType::Q4_0x4 {
            let row_q4_bytes = DType::Q4_0.row_bytes(cols);
            let group_bytes = row_q4_bytes * 4;
            let mut raw_g = vec![0u8; group_bytes];
            let mut rep_g = vec![0u8; group_bytes];
            let mut all = Vec::with_capacity((rows / 4) * group_bytes);
            for rg in 0..(rows / 4) {
                for m in 0..4 {
                    let r = rg * 4 + m;
                    decode_row(r, &mut f32_row);
                    add_lora_to_row(r, &mut f32_row);
                    quantize_row_to_bytes(
                        DType::Q4_0,
                        &f32_row,
                        &mut raw_g[m * row_q4_bytes..(m + 1) * row_q4_bytes],
                    );
                }
                repack_q4_0_bytes_to_q4_0x4(&raw_g, &mut rep_g, 4, cols);
                all.extend_from_slice(&rep_g);
            }
            all
        } else {
            let out_rb = dst_dtype.row_bytes(cols);
            let mut row_buf = vec![0u8; out_rb];
            let mut all = Vec::with_capacity(rows * out_rb);
            for r in 0..rows {
                decode_row(r, &mut f32_row);
                add_lora_to_row(r, &mut f32_row);
                quantize_row_to_bytes(dst_dtype, &f32_row, &mut row_buf);
                all.extend_from_slice(&row_buf);
            }
            all
        };

        built.push(BuiltTensor {
            name: name.clone(),
            dims: info.dims.clone(),
            dtype: dst_dtype,
            bytes: out_bytes,
        });
    }

    let views: Vec<NewGgufTensor<'_>> = built
        .iter()
        .map(|t| NewGgufTensor {
            name: &t.name,
            dims: &t.dims,
            dtype: t.dtype,
            data: &t.bytes,
        })
        .collect();
    let output_file_bytes =
        write_gguf_file_atomic(dst_path, &meta_vec, &views, base_gguf.alignment())?;

    Ok(LoraMergeReport {
        total_tensors: built.len(),
        merged_tensors,
        untied_output: untie_output,
        output_file_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_gguf_roundtrip_and_bound_batch_invariance() {
        let mut pairs = HashMap::new();
        pairs.insert(
            "blk.0.attn_q.weight".to_string(),
            LoraWeightPair {
                base_name: "blk.0.attn_q.weight".to_string(),
                is_token_embd: false,
                rank: 4,
                d_in: 32,
                d_out: 32,
                a: (0..4 * 32).map(|i| ((i as f32) - 60.0) * 0.01).collect(),
                b: (0..32 * 4).map(|i| ((i as f32) - 50.0) * 0.02).collect(),
            },
        );
        let adapter = LoraAdapter {
            arch: "llama".to_string(),
            alpha: 8.0,
            name: Some("unit-test-lora".to_string()),
            pairs,
            pair_order: vec!["blk.0.attn_q.weight".to_string()],
        };

        let tmp = std::env::temp_dir().join(format!("tm_lora_test_{}.gguf", std::process::id()));
        adapter.save_gguf(&tmp, false).expect("save_gguf");
        let loaded = LoraAdapter::open(&tmp).expect("load_gguf");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(loaded.arch, "llama");
        assert_eq!(loaded.alpha, 8.0);
        assert_eq!(loaded.pairs["blk.0.attn_q.weight"].rank, 4);
        assert_eq!(
            loaded.pairs["blk.0.attn_q.weight"].a,
            adapter.pairs["blk.0.attn_q.weight"].a
        );

        let mut set = LoraSet::new();
        set.push(loaded, 0.75);
        let bound = BoundLora::from_lora_set(&set);
        let slot = bound.slot_for("blk.0.attn_q.weight");
        assert!(slot.is_some());

        let pool = ThreadPool::new(2);
        let kt = tm_cpu::active_kernels();
        let mut scratch = LoraScratch::new(bound.max_rank);
        let x: Vec<f32> = (0..5 * 32).map(|i| (i as f32) * 0.03 - 1.5).collect();
        let mut y_batch = vec![1.25f32; 5 * 32];
        bound.apply_linear(slot, &pool, kt, &x, &mut y_batch, 5, &mut scratch);

        for t in 0..5 {
            let mut y_single = vec![1.25f32; 32];
            bound.apply_linear(
                slot,
                &pool,
                kt,
                &x[t * 32..(t + 1) * 32],
                &mut y_single,
                1,
                &mut scratch,
            );
            for c in 0..32 {
                assert_eq!(y_single[c].to_bits(), y_batch[t * 32 + c].to_bits());
            }
        }
    }
}
