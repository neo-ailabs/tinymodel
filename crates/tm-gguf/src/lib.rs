//! Zero-copy GGUF v2/v3 parser and memory-mapped tensor loader.
//!
//! `unsafe` is restricted strictly to `memmap2::Mmap::map` with a `// SAFETY:` comment.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;
use tm_core::{
    DType, QK4_0, dequantize_q4_0x4_single_row, dequantize_row_bytes, quantize_row_to_bytes,
    repack_q4_0_bytes_to_q4_0x4,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

/// Errors raised while opening, parsing, or validating a GGUF file.
#[derive(Debug, Error)]
pub enum GgufError {
    #[error("failed to open GGUF file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("GGUF '{path}': invalid magic bytes (expected 'GGUF')")]
    InvalidMagic { path: String },
    #[error("GGUF '{path}': unsupported GGUF version {version} (supported: 2, 3)")]
    UnsupportedVersion { path: String, version: u32 },
    #[error("GGUF '{path}': unexpected EOF at offset {offset}")]
    UnexpectedEof { path: String, offset: usize },
    #[error("GGUF '{path}': invalid UTF-8 string at offset {offset}")]
    InvalidUtf8 { path: String, offset: usize },
    #[error("GGUF '{path}': unknown metadata value type {val_type} for key '{key}'")]
    UnknownValueType {
        path: String,
        key: String,
        val_type: u32,
    },
    #[error("GGUF '{path}': invalid alignment {alignment} (must be non-zero power of two)")]
    InvalidAlignment { path: String, alignment: usize },
    #[error("GGUF '{path}': tensor '{tensor}' has unsupported ggml_type {ggml_type}")]
    UnsupportedDType {
        path: String,
        tensor: String,
        ggml_type: u32,
    },
    #[error(
        "GGUF '{path}': tensor '{tensor}' byte range [{start}..{end}) exceeds file size {file_len}"
    )]
    TensorOutOfBounds {
        path: String,
        tensor: String,
        start: usize,
        end: usize,
        file_len: usize,
    },
    #[error("GGUF '{path}': missing required metadata key '{key}'")]
    MissingMetadata { path: String, key: String },
    #[error("GGUF '{path}': metadata key '{key}' has unexpected type (expected {expected})")]
    MetadataTypeMismatch {
        path: String,
        key: String,
        expected: &'static str,
    },
    #[error("GGUF '{path}': missing required tensor '{tensor}' (layer {layer})")]
    MissingTensor {
        path: String,
        tensor: String,
        layer: String,
    },
    #[error(
        "GGUF '{path}': tensor '{tensor}' (layer {layer}) shape mismatch: expected {expected:?}, got {actual:?}"
    )]
    TensorShapeMismatch {
        path: String,
        tensor: String,
        layer: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("GGUF '{path}': invalid convert option: {detail}")]
    InvalidConvertOption { path: String, detail: String },
}

/// Typed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// Zero-copy string array storing `(byte_offset, byte_len)` spans inside `GgufFile`'s mmap.
    /// Eliminates ~500k `String` heap allocations and ~45 MiB dirty RAM on 248k-vocab models.
    StringSpans {
        spans: Vec<(u32, u32)>,
        total_bytes: usize,
    },
    U32Array(Vec<u32>),
    I32Array(Vec<i32>),
    F32Array(Vec<f32>),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            MetadataValue::U32(v) => Some(v),
            MetadataValue::I32(v) => u32::try_from(v).ok(),
            MetadataValue::U64(v) => u32::try_from(v).ok(),
            MetadataValue::I64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            MetadataValue::U32(v) => Some(v as usize),
            MetadataValue::I32(v) => usize::try_from(v).ok(),
            MetadataValue::U64(v) => usize::try_from(v).ok(),
            MetadataValue::I64(v) => usize::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            MetadataValue::F32(v) => Some(v),
            MetadataValue::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            MetadataValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            MetadataValue::Array(items) => items.iter().map(|v| v.as_str()).collect(),
            _ => None,
        }
    }

    pub fn as_i32_array(&self) -> Option<Vec<i32>> {
        match self {
            MetadataValue::I32Array(v) => Some(v.clone()),
            MetadataValue::U32Array(v) => v.iter().map(|&x| i32::try_from(x).ok()).collect(),
            MetadataValue::Array(items) => items
                .iter()
                .map(|v| match *v {
                    MetadataValue::I32(x) => Some(x),
                    MetadataValue::U32(x) => i32::try_from(x).ok(),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            MetadataValue::F32Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_usize_array(&self) -> Option<Vec<usize>> {
        match self {
            MetadataValue::U32Array(v) => Some(v.iter().map(|&x| x as usize).collect()),
            MetadataValue::I32Array(v) => v.iter().map(|&x| usize::try_from(x).ok()).collect(),
            MetadataValue::Array(items) => items.iter().map(|v| v.as_usize()).collect(),
            _ => None,
        }
    }

    pub fn as_bool_array(&self) -> Option<Vec<bool>> {
        match self {
            MetadataValue::Array(items) => items.iter().map(|v| v.as_bool()).collect(),
            _ => None,
        }
    }
}

/// Zero-copy iterator over a GGUF metadata string array (`StringSpans` or `Array`).
pub enum StringArrayIter<'a> {
    Spans {
        data: &'a [u8],
        iter: core::slice::Iter<'a, (u32, u32)>,
    },
    Values(core::slice::Iter<'a, MetadataValue>),
}

impl<'a> Iterator for StringArrayIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            StringArrayIter::Spans { data, iter } => {
                let &(start, len) = iter.next()?;
                let s = start as usize;
                let e = s + (len as usize);
                std::str::from_utf8(&data[s..e]).ok()
            }
            StringArrayIter::Values(iter) => iter.next()?.as_str(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for StringArrayIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            StringArrayIter::Spans { iter, .. } => iter.len(),
            StringArrayIter::Values(iter) => iter.len(),
        }
    }
}

/// Metadata entry for a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML dimension order: `dims[0]` is innermost (contiguous columns / `ne0`),
    /// `dims[1]` is rows (`ne1`), etc.
    pub dims: Vec<usize>,
    pub dtype: DType,
    /// Byte offset relative to the start of the GGUF tensor data section.
    pub offset: usize,
    /// Total size in bytes of this tensor's raw data.
    pub byte_len: usize,
    /// Byte offset of the 4-byte `ggml_type` field inside the GGUF header (for zero-overhead offline conversion).
    pub header_type_offset: usize,
}

enum Backing {
    Mmap(Mmap),
    #[cfg(test)]
    Owned(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m.as_ref(),
            #[cfg(test)]
            Backing::Owned(v) => v.as_slice(),
        }
    }
}

/// Memory-mapped GGUF file with parsed metadata and tensor directory.
pub struct GgufFile {
    path: PathBuf,
    version: u32,
    alignment: usize,
    data_offset: usize,
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, TensorInfo>,
    tensor_order: Vec<String>,
    backing: Backing,
}

impl GgufFile {
    /// Opens and memory-maps a GGUF file from disk, validating its header and tensor table.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let path_buf = path.as_ref().to_path_buf();
        let path_str = path_buf.display().to_string();
        let file = File::open(&path_buf).map_err(|source| GgufError::Io {
            path: path_str.clone(),
            source,
        })?;
        // SAFETY: The file is opened read-only and we treat the mapped bytes as immutable
        // for the lifetime of `GgufFile`. Callers must not mutate the underlying file on disk.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| GgufError::Io {
            path: path_str.clone(),
            source,
        })?;
        // Hint sequential page-cache readahead on Unix (accelerates cold SD/SSD load on Pi 4B / Zero 2W).
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);
        Self::parse(path_buf, Backing::Mmap(mmap))
    }

    #[cfg(test)]
    pub(crate) fn from_bytes(name: &str, bytes: Vec<u8>) -> Result<Self, GgufError> {
        Self::parse(PathBuf::from(name), Backing::Owned(bytes))
    }

    fn parse(path: PathBuf, backing: Backing) -> Result<Self, GgufError> {
        let path_str = path.display().to_string();
        let data = backing.as_slice();
        let mut r = Cursor {
            data,
            pos: 0,
            path: &path_str,
        };

        let magic = r.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { path: path_str });
        }

        let version = r.read_u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion {
                path: path_str,
                version,
            });
        }

        let n_tensors = usize::try_from(r.read_u64()?).map_err(|_| GgufError::UnexpectedEof {
            path: path_str.clone(),
            offset: r.pos,
        })?;
        let n_kv = usize::try_from(r.read_u64()?).map_err(|_| GgufError::UnexpectedEof {
            path: path_str.clone(),
            offset: r.pos,
        })?;

        let mut metadata = HashMap::with_capacity(n_kv.min(4096));
        for _ in 0..n_kv {
            let key = r.read_string()?;
            let val_type = r.read_u32()?;
            let val = r.read_value(&key, val_type)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_usize())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment {
                path: path_str,
                alignment,
            });
        }

        let mut raw_tensors = Vec::with_capacity(n_tensors.min(4096));
        for _ in 0..n_tensors {
            let name = r.read_string()?;
            let n_dims = r.read_u32()? as usize;
            if n_dims == 0 || n_dims > 8 {
                return Err(GgufError::UnexpectedEof {
                    path: path_str.clone(),
                    offset: r.pos,
                });
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let d = usize::try_from(r.read_u64()?).map_err(|_| GgufError::UnexpectedEof {
                    path: path_str.clone(),
                    offset: r.pos,
                })?;
                dims.push(d);
            }
            let header_type_offset = r.pos;
            let ggml_type = r.read_u32()?;
            let offset = usize::try_from(r.read_u64()?).map_err(|_| GgufError::UnexpectedEof {
                path: path_str.clone(),
                offset: r.pos,
            })?;

            let dtype =
                DType::from_ggml_type(ggml_type).ok_or_else(|| GgufError::UnsupportedDType {
                    path: path_str.clone(),
                    tensor: name.clone(),
                    ggml_type,
                })?;

            let n_elems = dims
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| GgufError::UnexpectedEof {
                    path: path_str.clone(),
                    offset: r.pos,
                })?;
            let blk = dtype.block_elements();
            if !n_elems.is_multiple_of(blk) {
                return Err(GgufError::TensorShapeMismatch {
                    path: path_str.clone(),
                    tensor: name,
                    layer: "header".to_string(),
                    expected: vec![blk],
                    actual: dims,
                });
            }
            let byte_len = (n_elems / blk)
                .checked_mul(dtype.block_bytes())
                .ok_or_else(|| GgufError::UnexpectedEof {
                    path: path_str.clone(),
                    offset: r.pos,
                })?;

            raw_tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
                byte_len,
                header_type_offset,
            });
        }

        let rem = r.pos % alignment;
        let data_offset = if rem == 0 {
            r.pos
        } else {
            r.pos
                .checked_add(alignment - rem)
                .ok_or_else(|| GgufError::UnexpectedEof {
                    path: path_str.clone(),
                    offset: r.pos,
                })?
        };

        let mut tensors = HashMap::with_capacity(raw_tensors.len());
        let mut tensor_order = Vec::with_capacity(raw_tensors.len());
        for info in raw_tensors {
            let start = data_offset.checked_add(info.offset).ok_or_else(|| {
                GgufError::TensorOutOfBounds {
                    path: path_str.clone(),
                    tensor: info.name.clone(),
                    start: data_offset,
                    end: usize::MAX,
                    file_len: data.len(),
                }
            })?;
            let end =
                start
                    .checked_add(info.byte_len)
                    .ok_or_else(|| GgufError::TensorOutOfBounds {
                        path: path_str.clone(),
                        tensor: info.name.clone(),
                        start,
                        end: usize::MAX,
                        file_len: data.len(),
                    })?;
            if end > data.len() {
                return Err(GgufError::TensorOutOfBounds {
                    path: path_str.clone(),
                    tensor: info.name.clone(),
                    start,
                    end,
                    file_len: data.len(),
                });
            }
            tensor_order.push(info.name.clone());
            tensors.insert(info.name.clone(), info);
        }

        Ok(Self {
            path,
            version,
            alignment,
            data_offset,
            metadata,
            tensors,
            tensor_order,
            backing,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Byte offset where the tensor data section begins in the file.
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Raw read-only view of the entire memory-mapped GGUF file.
    pub fn as_bytes(&self) -> &[u8] {
        self.backing.as_slice()
    }

    /// Returns `true` when backed by an OS page-aligned read-only `mmap` (vs in-memory test buffer).
    pub fn is_mmap(&self) -> bool {
        #[cfg(test)]
        {
            matches!(self.backing, Backing::Mmap(_))
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    /// Absolute byte offset of a tensor from the start of the GGUF file.
    pub fn tensor_file_offset(&self, info: &TensorInfo) -> usize {
        self.data_offset + info.offset
    }

    pub fn metadata(&self) -> &HashMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn get_meta(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Returns `(total_utf8_bytes, zero_copy_iter)` for a GGUF string array metadata entry.
    pub fn string_array_iter(&self, key: &str) -> Option<(usize, StringArrayIter<'_>)> {
        match self.metadata.get(key)? {
            MetadataValue::StringSpans { spans, total_bytes } => Some((
                *total_bytes,
                StringArrayIter::Spans {
                    data: self.backing.as_slice(),
                    iter: spans.iter(),
                },
            )),
            MetadataValue::Array(items) => {
                let mut total = 0usize;
                for item in items {
                    total += item.as_str()?.len();
                }
                Some((total, StringArrayIter::Values(items.iter())))
            }
            _ => None,
        }
    }

    pub fn req_str(&self, key: &str) -> Result<&str, GgufError> {
        let v = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata {
                path: self.path.display().to_string(),
                key: key.to_string(),
            })?;
        v.as_str().ok_or_else(|| GgufError::MetadataTypeMismatch {
            path: self.path.display().to_string(),
            key: key.to_string(),
            expected: "string",
        })
    }

    pub fn req_usize(&self, key: &str) -> Result<usize, GgufError> {
        let v = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata {
                path: self.path.display().to_string(),
                key: key.to_string(),
            })?;
        v.as_usize().ok_or_else(|| GgufError::MetadataTypeMismatch {
            path: self.path.display().to_string(),
            key: key.to_string(),
            expected: "integer",
        })
    }

    pub fn req_f32(&self, key: &str) -> Result<f32, GgufError> {
        let v = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata {
                path: self.path.display().to_string(),
                key: key.to_string(),
            })?;
        v.as_f32().ok_or_else(|| GgufError::MetadataTypeMismatch {
            path: self.path.display().to_string(),
            key: key.to_string(),
            expected: "f32",
        })
    }

    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_order
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Returns `(TensorInfo, absolute_file_offset, byte_len)` after validating `expected_dims`,
    /// WITHOUT touching the `mmap` pages of the tensor (allowing `DirectTensorReader` to load
    /// multi-GiB tensors via `O_DIRECT` without polluting Linux page cache).
    pub fn req_tensor_file_range(
        &self,
        name: &str,
        layer: &str,
        expected_dims: &[usize],
    ) -> Result<(&TensorInfo, u64, usize), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor {
                path: self.path.display().to_string(),
                tensor: name.to_string(),
                layer: layer.to_string(),
            })?;
        if info.dims.as_slice() != expected_dims {
            return Err(GgufError::TensorShapeMismatch {
                path: self.path.display().to_string(),
                tensor: name.to_string(),
                layer: layer.to_string(),
                expected: expected_dims.to_vec(),
                actual: info.dims.clone(),
            });
        }
        let abs_offset = (self.data_offset + info.offset) as u64;
        Ok((info, abs_offset, info.byte_len))
    }

    /// Borrows the raw mmap bytes for tensor `name` (zero copy).
    pub fn tensor_bytes(&self, name: &str, layer: &str) -> Result<(&TensorInfo, &[u8]), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor {
                path: self.path.display().to_string(),
                tensor: name.to_string(),
                layer: layer.to_string(),
            })?;
        let start = self.data_offset + info.offset;
        let end = start + info.byte_len;
        Ok((info, &self.backing.as_slice()[start..end]))
    }

    /// Borrows a tensor and strictly validates its expected dimensions.
    pub fn req_tensor(
        &self,
        name: &str,
        layer: &str,
        expected_dims: &[usize],
    ) -> Result<(&TensorInfo, &[u8]), GgufError> {
        let (info, bytes) = self.tensor_bytes(name, layer)?;
        if info.dims.as_slice() != expected_dims {
            return Err(GgufError::TensorShapeMismatch {
                path: self.path.display().to_string(),
                tensor: name.to_string(),
                layer: layer.to_string(),
                expected: expected_dims.to_vec(),
                actual: info.dims.clone(),
            });
        }
        Ok((info, bytes))
    }

    /// Streams this GGUF file to `dst_path`, repacking eligible 2D `Q4_0` projection matrices
    /// (`rows % 4 == 0`, `cols % 32 == 0`) into zero-overhead `DType::Q4_0x4` (`Q4_0_4x8`)
    /// while preserving exact file size, alignment, metadata, and zero-copy `mmap` compatibility (`F-04`).
    pub fn convert_to_q4_0x4<P: AsRef<Path>>(
        &self,
        dst_path: P,
    ) -> Result<ConvertStats, GgufError> {
        let out_path = dst_path.as_ref();
        let out_str = out_path.display().to_string();
        let raw = self.backing.as_slice();
        let has_separate_output = self.tensors.contains_key("output.weight");

        // 1. Copy header `[0..data_offset)` and patch 4-byte `ggml_type` on eligible tensors.
        let mut header = raw[..self.data_offset].to_vec();
        let mut repacked_spans: Vec<(usize, usize, usize, usize)> = Vec::new(); // (file_start, file_end, rows, cols)
        let mut repacked_tensors = 0usize;
        let mut repacked_bytes = 0usize;

        for name in &self.tensor_order {
            let info = &self.tensors[name];
            if info.dtype != DType::Q4_0 || info.dims.len() != 2 {
                continue;
            }
            if info.name == "per_layer_token_embd.weight" {
                continue;
            }
            if info.name == "token_embd.weight" && has_separate_output {
                continue;
            }
            let cols = info.dims[0];
            let rows = info.dims[1];
            if !cols.is_multiple_of(QK4_0) || !rows.is_multiple_of(4) {
                continue;
            }

            let new_type = DType::Q4_0x4.to_ggml_type().to_le_bytes();
            let off = info.header_type_offset;
            header[off..off + 4].copy_from_slice(&new_type);

            let f_start = self.data_offset + info.offset;
            let f_end = f_start + info.byte_len;
            repacked_spans.push((f_start, f_end, rows, cols));
            repacked_tensors += 1;
            repacked_bytes += info.byte_len;
        }

        repacked_spans.sort_by_key(|s| s.0);

        let file = File::create(out_path).map_err(|source| GgufError::Io {
            path: out_str.clone(),
            source,
        })?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        writer.write_all(&header).map_err(|source| GgufError::Io {
            path: out_str.clone(),
            source,
        })?;

        // 2. Stream data section `[data_offset..raw.len())`, repacking matched tensors row-group by row-group
        //    with a tiny reusable scratch buffer (< 256 KiB peak heap).
        let mut cursor = self.data_offset;
        let mut group_scratch: Vec<u8> = Vec::new();
        for (f_start, f_end, rows, cols) in repacked_spans {
            if cursor < f_start {
                writer
                    .write_all(&raw[cursor..f_start])
                    .map_err(|source| GgufError::Io {
                        path: out_str.clone(),
                        source,
                    })?;
            }
            let cols_blocks = cols / QK4_0;
            let group_bytes = cols_blocks * 72;
            if group_scratch.len() < group_bytes {
                group_scratch.resize(group_bytes, 0);
            }
            let src_tensor = &raw[f_start..f_end];
            for rg in 0..(rows / 4) {
                let s0 = rg * group_bytes;
                repack_q4_0_bytes_to_q4_0x4(
                    &src_tensor[s0..s0 + group_bytes],
                    &mut group_scratch[..group_bytes],
                    4,
                    cols,
                );
                writer
                    .write_all(&group_scratch[..group_bytes])
                    .map_err(|source| GgufError::Io {
                        path: out_str.clone(),
                        source,
                    })?;
            }
            cursor = f_end;
        }

        if cursor < raw.len() {
            writer
                .write_all(&raw[cursor..])
                .map_err(|source| GgufError::Io {
                    path: out_str.clone(),
                    source,
                })?;
        }
        writer.flush().map_err(|source| GgufError::Io {
            path: out_str,
            source,
        })?;

        Ok(ConvertStats {
            total_tensors: self.tensor_order.len(),
            repacked_tensors,
            repacked_bytes,
            total_file_bytes: raw.len(),
        })
    }
}

/// Summary statistics returned by `GgufFile::convert_to_q4_0x4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertStats {
    pub total_tensors: usize,
    pub repacked_tensors: usize,
    pub repacked_bytes: usize,
    pub total_file_bytes: usize,
}

/// Options for offline GGUF weight quantization, layout repacking, and BPE-closed vocabulary pruning (`F-04`).
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Target quantization for 2D linear projection weights (`None` defaults to `Q4_0x4` repacking).
    pub quant: Option<DType>,
    /// Optional separate quantization format for the LM head (`output.weight` or tied `token_embd.weight`).
    pub lm_head_quant: Option<DType>,
    /// Optional target vocabulary size `N` (`0 < N < old_vocab_size`) for BPE-closed vocabulary pruning.
    pub prune_vocab: Option<usize>,
    /// Optional UTF-8 domain corpus whose BPE subwords (and recursive merge ancestors) are prioritized.
    pub vocab_corpus: Option<String>,
}

/// Summary report returned by `GgufFile::convert_with_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertReport {
    pub total_tensors: usize,
    pub converted_tensors: usize,
    pub old_vocab_size: usize,
    pub new_vocab_size: usize,
    pub old_merges: usize,
    pub kept_merges: usize,
    pub input_file_bytes: usize,
    pub output_file_bytes: usize,
}

struct VocabPrunePlan {
    kept_old_ids: Vec<usize>,
    old_to_new: Vec<Option<u32>>,
    kept_tokens: Vec<String>,
    kept_merges: Vec<String>,
}

fn build_gpt2_byte_to_char_array() -> [char; 256] {
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
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0u16..256 {
        out[b as usize] = if is_direct[b as usize] {
            char::from_u32(b as u32).unwrap_or('\0')
        } else {
            let c = char::from_u32(256 + n).unwrap_or('\0');
            n += 1;
            c
        };
    }
    out
}

fn resolve_quant_dtype(req: DType, cols: usize, rows: usize, orig: DType) -> DType {
    if req == DType::Q4_0x4 {
        if cols.is_multiple_of(QK4_0) && rows.is_multiple_of(4) {
            return DType::Q4_0x4;
        }
        if cols.is_multiple_of(QK4_0) {
            return DType::Q4_0;
        }
        return orig;
    }
    if cols.is_multiple_of(req.block_elements()) {
        return req;
    }
    // Automatic per-tensor fallback when `cols % 256 != 0` (e.g. `SmolLM2-360M` `n_embd = 960`):
    match req {
        DType::TQ2_0 | DType::Iq2Xxs => {
            if cols.is_multiple_of(64) {
                DType::Q2_0
            } else if cols.is_multiple_of(32) {
                DType::Iq4Nl
            } else {
                orig
            }
        }
        DType::Iq3Xxs => {
            if cols.is_multiple_of(32) {
                DType::Iq4Nl
            } else {
                orig
            }
        }
        _ => {
            if cols.is_multiple_of(32) {
                DType::Q4_0
            } else {
                orig
            }
        }
    }
}

fn write_gguf_str<W: Write>(w: &mut W, s: &str, out_path: &str) -> Result<(), GgufError> {
    w.write_all(&(s.len() as u64).to_le_bytes())
        .and_then(|_| w.write_all(s.as_bytes()))
        .map_err(|source| GgufError::Io {
            path: out_path.to_string(),
            source,
        })
}

fn write_gguf_val<W: Write>(
    w: &mut W,
    val: &MetadataValue,
    backing_bytes: &[u8],
    write_tag: bool,
    out_path: &str,
) -> Result<(), GgufError> {
    let io_err = |source: std::io::Error| GgufError::Io {
        path: out_path.to_string(),
        source,
    };
    let tag: u32 = match val {
        MetadataValue::U8(_) => 0,
        MetadataValue::I8(_) => 1,
        MetadataValue::U16(_) => 2,
        MetadataValue::I16(_) => 3,
        MetadataValue::U32(_) => 4,
        MetadataValue::I32(_) => 5,
        MetadataValue::F32(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => 8,
        MetadataValue::StringSpans { .. }
        | MetadataValue::U32Array(_)
        | MetadataValue::I32Array(_)
        | MetadataValue::F32Array(_)
        | MetadataValue::Array(_) => 9,
        MetadataValue::U64(_) => 10,
        MetadataValue::I64(_) => 11,
        MetadataValue::F64(_) => 12,
    };
    if write_tag {
        w.write_all(&tag.to_le_bytes()).map_err(io_err)?;
    }
    match val {
        MetadataValue::U8(v) => w.write_all(&[*v]).map_err(io_err),
        MetadataValue::I8(v) => w.write_all(&[*v as u8]).map_err(io_err),
        MetadataValue::U16(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::I16(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::U32(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::I32(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::F32(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::Bool(v) => w.write_all(&[u8::from(*v)]).map_err(io_err),
        MetadataValue::String(s) => write_gguf_str(w, s, out_path),
        MetadataValue::U64(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::I64(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::F64(v) => w.write_all(&v.to_le_bytes()).map_err(io_err),
        MetadataValue::StringSpans { spans, .. } => {
            w.write_all(&8u32.to_le_bytes()).map_err(io_err)?;
            w.write_all(&(spans.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            for &(start, len) in spans {
                let s = start as usize;
                let l = len as usize;
                w.write_all(&(l as u64).to_le_bytes()).map_err(io_err)?;
                w.write_all(&backing_bytes[s..s + l]).map_err(io_err)?;
            }
            Ok(())
        }
        MetadataValue::U32Array(arr) => {
            w.write_all(&4u32.to_le_bytes()).map_err(io_err)?;
            w.write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            for &x in arr {
                w.write_all(&x.to_le_bytes()).map_err(io_err)?;
            }
            Ok(())
        }
        MetadataValue::I32Array(arr) => {
            w.write_all(&5u32.to_le_bytes()).map_err(io_err)?;
            w.write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            for &x in arr {
                w.write_all(&x.to_le_bytes()).map_err(io_err)?;
            }
            Ok(())
        }
        MetadataValue::F32Array(arr) => {
            w.write_all(&6u32.to_le_bytes()).map_err(io_err)?;
            w.write_all(&(arr.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            for &x in arr {
                w.write_all(&x.to_le_bytes()).map_err(io_err)?;
            }
            Ok(())
        }
        MetadataValue::Array(items) => {
            let elem_tag: u32 = match items.first() {
                Some(MetadataValue::U8(_)) => 0,
                Some(MetadataValue::I8(_)) => 1,
                Some(MetadataValue::U16(_)) => 2,
                Some(MetadataValue::I16(_)) => 3,
                Some(MetadataValue::U32(_)) => 4,
                Some(MetadataValue::I32(_)) => 5,
                Some(MetadataValue::F32(_)) => 6,
                Some(MetadataValue::Bool(_)) => 7,
                Some(MetadataValue::String(_)) | None => 8,
                Some(MetadataValue::U64(_)) => 10,
                Some(MetadataValue::I64(_)) => 11,
                Some(MetadataValue::F64(_)) => 12,
                _ => 9,
            };
            w.write_all(&elem_tag.to_le_bytes()).map_err(io_err)?;
            w.write_all(&(items.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            for item in items {
                write_gguf_val(w, item, backing_bytes, false, out_path)?;
            }
            Ok(())
        }
    }
}

impl GgufFile {
    fn build_vocab_prune_plan(
        &self,
        target_vocab: usize,
        corpus: Option<&str>,
    ) -> Result<VocabPrunePlan, GgufError> {
        let path_str = self.path.display().to_string();
        let (_, tok_iter) = self
            .string_array_iter("tokenizer.ggml.tokens")
            .ok_or_else(|| GgufError::MissingMetadata {
                path: path_str.clone(),
                key: "tokenizer.ggml.tokens".to_string(),
            })?;
        let tokens: Vec<&str> = tok_iter.collect();
        let v_old = tokens.len();

        let embd_rows = self
            .tensors
            .get("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .unwrap_or(v_old);
        let max_valid = v_old.min(embd_rows);

        if target_vocab < 264 || target_vocab >= max_valid {
            return Err(GgufError::InvalidConvertOption {
                path: path_str,
                detail: format!(
                    "--prune-vocab {target_vocab} must be in 264..{max_valid} (current vocab={v_old})"
                ),
            });
        }

        let mut str_to_id: HashMap<&str, usize> = HashMap::with_capacity(v_old * 2);
        for (idx, &s) in tokens.iter().enumerate() {
            if idx < max_valid {
                str_to_id.entry(s).or_insert(idx);
            }
        }

        let mut raw_merges: Vec<&str> = Vec::new();
        if let Some((_, m_iter)) = self.string_array_iter("tokenizer.ggml.merges") {
            raw_merges = m_iter.collect();
        }

        let mut parsed_merges: Vec<(usize, usize, usize, usize)> =
            Vec::with_capacity(raw_merges.len()); // (line_idx, id_a, id_b, id_ab)
        let mut parent_of: Vec<Option<(usize, usize)>> = vec![None; max_valid];
        let mut ab_buf = String::with_capacity(64);
        for (m_idx, &line) in raw_merges.iter().enumerate() {
            if let Some((a, b)) = line.split_once(' ')
                && let (Some(&ia), Some(&ib)) = (str_to_id.get(a), str_to_id.get(b))
            {
                ab_buf.clear();
                ab_buf.push_str(a);
                ab_buf.push_str(b);
                if let Some(&iab) = str_to_id.get(ab_buf.as_str()) {
                    parsed_merges.push((m_idx, ia, ib, iab));
                    if parent_of[iab].is_none() {
                        parent_of[iab] = Some((ia, ib));
                    }
                }
            }
        }

        let mut keep = vec![false; max_valid];
        let mut kept_count = 0usize;

        let mark_plain = |id: usize, keep: &mut [bool], cnt: &mut usize, cap: usize| {
            if id < keep.len() && !keep[id] && *cnt < cap {
                keep[id] = true;
                *cnt += 1;
            }
        };

        // 1. Mandatory special IDs from GGUF metadata
        for meta_key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.sep_token_id",
            "tokenizer.ggml.nl_token_id",
        ] {
            if let Some(id) = self.metadata.get(meta_key).and_then(|v| v.as_usize()) {
                mark_plain(id, &mut keep, &mut kept_count, target_vocab);
            }
        }

        // 2. Mandatory base byte fallback tokens (`<= 300` slots) + control/chat markers
        if let Some(ttypes) = self
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_i32_array())
        {
            // Keep BYTE (`6`), UNKNOWN (`2`), CONTROL (`3`), USER_DEFINED (`4`) up to half of `target_vocab`
            let special_cap = (target_vocab / 2).max(260);
            for (idx, &tt) in ttypes.iter().enumerate().take(max_valid) {
                if matches!(tt, 2 | 3 | 4 | 6) && kept_count < special_cap {
                    mark_plain(idx, &mut keep, &mut kept_count, target_vocab);
                }
            }
        }
        for (idx, &s) in tokens.iter().enumerate().take(max_valid) {
            let mut chars = s.chars();
            let is_base_byte_char = match (chars.next(), chars.next()) {
                (Some(c), None) => (c as u32) <= 0x024F || c == '\u{2581}',
                _ => false,
            };
            let is_angle_special = s.starts_with('<') && s.ends_with('>') && s.len() <= 32;
            if is_base_byte_char || is_angle_special {
                mark_plain(idx, &mut keep, &mut kept_count, target_vocab);
            }
        }

        // Helper: mark `id` together with ALL recursive BPE merge ancestors so BPE merge closure holds!
        let mark_bpe_closed = |root: usize,
                               keep: &mut [bool],
                               cnt: &mut usize,
                               parents: &[Option<(usize, usize)>],
                               cap: usize| {
            if root >= keep.len() || keep[root] {
                return;
            }
            let mut stack = vec![root];
            let mut to_add: Vec<usize> = Vec::new();
            while let Some(cur) = stack.pop() {
                if cur >= keep.len() || keep[cur] || to_add.contains(&cur) {
                    continue;
                }
                to_add.push(cur);
                if let Some((pa, pb)) = parents[cur] {
                    stack.push(pa);
                    stack.push(pb);
                }
            }
            if *cnt + to_add.len() <= cap {
                for id in to_add {
                    keep[id] = true;
                    *cnt += 1;
                }
            }
        };

        // 3. Optional domain corpus (`--vocab-corpus`): simulate full BPE merges on corpus words & rank by frequency
        if let Some(text) = corpus
            && !text.is_empty()
        {
            let tok_model = self
                .metadata
                .get("tokenizer.ggml.model")
                .and_then(|v| v.as_str())
                .unwrap_or("gpt2");
            let byte_to_char = build_gpt2_byte_to_char_array();
            let mut pair_map: HashMap<(usize, usize), (usize, usize)> =
                HashMap::with_capacity(parsed_merges.len());
            for &(rank, ia, ib, iab) in &parsed_merges {
                pair_map.entry((ia, ib)).or_insert((rank, iab));
            }

            let mut word_counts: HashMap<String, usize> = HashMap::new();
            let mut cur = String::new();
            let mut prev_ws = false;
            for ch in text.chars() {
                let ws = ch.is_whitespace();
                if !cur.is_empty() && ws != prev_ws && !prev_ws {
                    *word_counts.entry(std::mem::take(&mut cur)).or_insert(0) += 1;
                }
                cur.push(ch);
                prev_ws = ws;
                if cur.len() >= 48 {
                    *word_counts.entry(std::mem::take(&mut cur)).or_insert(0) += 1;
                }
            }
            if !cur.is_empty() {
                *word_counts.entry(cur).or_insert(0) += 1;
            }

            let mut freq = vec![0usize; max_valid];
            let mut cbuf = [0u8; 4];
            for (piece, occ) in word_counts {
                let mut syms: Vec<usize> = Vec::with_capacity(piece.len());
                if tok_model == "gemma4" {
                    for ch in piece.chars() {
                        let mapped = if ch == ' ' { '\u{2581}' } else { ch };
                        let s = mapped.encode_utf8(&mut cbuf);
                        if let Some(&id) = str_to_id.get(s) {
                            syms.push(id);
                        }
                    }
                } else {
                    for &b in piece.as_bytes() {
                        let c = byte_to_char[b as usize];
                        let s = c.encode_utf8(&mut cbuf);
                        if let Some(&id) = str_to_id.get(s) {
                            syms.push(id);
                        }
                    }
                }
                while syms.len() >= 2 {
                    let mut best_rank = usize::MAX;
                    let mut best_pos = 0usize;
                    let mut best_id = 0usize;
                    for i in 0..(syms.len() - 1) {
                        if let Some(&(r, mid)) = pair_map.get(&(syms[i], syms[i + 1]))
                            && r < best_rank
                        {
                            best_rank = r;
                            best_pos = i;
                            best_id = mid;
                        }
                    }
                    if best_rank == usize::MAX {
                        break;
                    }
                    syms[best_pos] = best_id;
                    syms.remove(best_pos + 1);
                }
                for id in syms {
                    freq[id] += occ;
                }
            }

            let mut ranked: Vec<(usize, usize)> = freq
                .into_iter()
                .enumerate()
                .filter(|&(_, f)| f > 0)
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (id, _) in ranked {
                if kept_count >= target_vocab {
                    break;
                }
                mark_bpe_closed(id, &mut keep, &mut kept_count, &parent_of, target_vocab);
            }
        }

        // 4. Fill remaining budget in BPE merge rank order (preserves highest-frequency subwords & closure)
        for &(_, ia, ib, iab) in &parsed_merges {
            if kept_count >= target_vocab {
                break;
            }
            mark_bpe_closed(ia, &mut keep, &mut kept_count, &parent_of, target_vocab);
            mark_bpe_closed(ib, &mut keep, &mut kept_count, &parent_of, target_vocab);
            mark_bpe_closed(iab, &mut keep, &mut kept_count, &parent_of, target_vocab);
        }

        // 5. Top up any remaining slots in ascending token ID order
        for id in 0..max_valid {
            if kept_count >= target_vocab {
                break;
            }
            mark_plain(id, &mut keep, &mut kept_count, target_vocab);
        }

        let mut kept_old_ids = Vec::with_capacity(target_vocab);
        let mut old_to_new: Vec<Option<u32>> = vec![None; embd_rows.max(v_old)];
        for (old_id, &is_kept) in keep.iter().enumerate() {
            if is_kept {
                let new_id = kept_old_ids.len() as u32;
                kept_old_ids.push(old_id);
                old_to_new[old_id] = Some(new_id);
            }
        }

        let kept_tokens: Vec<String> = kept_old_ids
            .iter()
            .map(|&old_id| tokens[old_id].to_string())
            .collect();

        let mut kept_merges: Vec<String> = Vec::new();
        for &(m_idx, ia, ib, iab) in &parsed_merges {
            if old_to_new[ia].is_some() && old_to_new[ib].is_some() && old_to_new[iab].is_some() {
                kept_merges.push(raw_merges[m_idx].to_string());
            }
        }

        Ok(VocabPrunePlan {
            kept_old_ids,
            old_to_new,
            kept_tokens,
            kept_merges,
        })
    }

    /// Streams and converts this GGUF model to `dst_path` with optional low-bit quantization
    /// (`Q2_0`, `TQ2_0`, `IQ2_XXS`, `IQ3_XXS`, `IQ4_NL`, `Q4_0x4`, `Q4_0`, `Q8_0`), separate
    /// `--lm-head-quant`, and BPE-closed `--prune-vocab` (`F-04`, `ARCHITECTURE.md` §5.6).
    pub fn convert_with_options<P: AsRef<Path>>(
        &self,
        dst_path: P,
        opts: &ConvertOptions,
    ) -> Result<ConvertReport, GgufError> {
        let raw = self.backing.as_slice();
        let old_vocab = self
            .string_array_iter("tokenizer.ggml.tokens")
            .map(|(_, it)| it.len())
            .or_else(|| {
                self.tensors
                    .get("token_embd.weight")
                    .and_then(|t| t.dims.get(1).copied())
            })
            .unwrap_or(0);
        let old_merges = self
            .string_array_iter("tokenizer.ggml.merges")
            .map(|(_, it)| it.len())
            .unwrap_or(0);

        // Fast zero-overhead path when only repacking existing `Q4_0 -> Q4_0x4` without vocab pruning:
        let is_pure_q4x4 = opts.prune_vocab.is_none()
            && opts.lm_head_quant.is_none()
            && matches!(opts.quant, None | Some(DType::Q4_0x4))
            && self
                .tensors
                .values()
                .filter(|t| t.dims.len() == 2 && t.name != "per_layer_token_embd.weight")
                .any(|t| t.dtype == DType::Q4_0);
        if is_pure_q4x4 && opts.quant.is_none() {
            let st = self.convert_to_q4_0x4(&dst_path)?;
            return Ok(ConvertReport {
                total_tensors: st.total_tensors,
                converted_tensors: st.repacked_tensors,
                old_vocab_size: old_vocab,
                new_vocab_size: old_vocab,
                old_merges,
                kept_merges: old_merges,
                input_file_bytes: st.total_file_bytes,
                output_file_bytes: st.total_file_bytes,
            });
        }

        let out_path = dst_path.as_ref();
        let out_str = out_path.display().to_string();
        let prune_plan = match opts.prune_vocab {
            Some(n) => Some(self.build_vocab_prune_plan(n, opts.vocab_corpus.as_deref())?),
            None => None,
        };
        let new_vocab = prune_plan
            .as_ref()
            .map(|p| p.kept_old_ids.len())
            .unwrap_or(old_vocab);
        let kept_merges_cnt = prune_plan
            .as_ref()
            .map(|p| p.kept_merges.len())
            .unwrap_or(old_merges);

        // 1. Prepare updated metadata dictionary
        let mut meta = self.metadata.clone();
        meta.insert(
            "general.alignment".to_string(),
            MetadataValue::U32(self.alignment as u32),
        );
        if let Some(ref plan) = prune_plan {
            meta.insert(
                "tokenizer.ggml.tokens".to_string(),
                MetadataValue::Array(
                    plan.kept_tokens
                        .iter()
                        .cloned()
                        .map(MetadataValue::String)
                        .collect(),
                ),
            );
            if meta.contains_key("tokenizer.ggml.merges") {
                meta.insert(
                    "tokenizer.ggml.merges".to_string(),
                    MetadataValue::Array(
                        plan.kept_merges
                            .iter()
                            .cloned()
                            .map(MetadataValue::String)
                            .collect(),
                    ),
                );
            }
            if let Some(MetadataValue::F32Array(scores)) =
                meta.get("tokenizer.ggml.scores").cloned()
            {
                let new_scores: Vec<f32> = plan
                    .kept_old_ids
                    .iter()
                    .map(|&i| scores.get(i).copied().unwrap_or(0.0))
                    .collect();
                meta.insert(
                    "tokenizer.ggml.scores".to_string(),
                    MetadataValue::F32Array(new_scores),
                );
            }
            if let Some(tt_arr) = meta
                .get("tokenizer.ggml.token_type")
                .and_then(|v| v.as_i32_array())
            {
                let new_tt: Vec<i32> = plan
                    .kept_old_ids
                    .iter()
                    .map(|&i| tt_arr.get(i).copied().unwrap_or(1))
                    .collect();
                meta.insert(
                    "tokenizer.ggml.token_type".to_string(),
                    MetadataValue::I32Array(new_tt),
                );
            }
            for id_key in [
                "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.eot_token_id",
                "tokenizer.ggml.unknown_token_id",
                "tokenizer.ggml.padding_token_id",
                "tokenizer.ggml.sep_token_id",
                "tokenizer.ggml.nl_token_id",
            ] {
                if let Some(old_id) = meta.get(id_key).and_then(|v| v.as_usize()) {
                    if let Some(Some(remapped)) = plan.old_to_new.get(old_id) {
                        meta.insert(id_key.to_string(), MetadataValue::U32(*remapped));
                    } else {
                        meta.remove(id_key);
                    }
                }
            }
            if let Some(arch) = meta
                .get("general.architecture")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
            {
                let arch_vocab_key = format!("{arch}.vocab_size");
                if meta.contains_key(&arch_vocab_key) {
                    meta.insert(arch_vocab_key, MetadataValue::U32(new_vocab as u32));
                }
            }
        }

        // 2. Plan output tensors (`dims`, `dtype`, `byte_len`, `aligned_offset`)
        struct OutTensorPlan {
            name: String,
            dims: Vec<usize>,
            new_dtype: DType,
            byte_len: usize,
            offset: usize,
            vocab_sliced: bool,
        }

        let has_separate_output = self.tensors.contains_key("output.weight");
        let default_quant = opts.quant;
        let mut plans: Vec<OutTensorPlan> = Vec::with_capacity(self.tensor_order.len());
        let mut cur_data_off = 0usize;
        let mut converted_tensors = 0usize;

        for name in &self.tensor_order {
            let info = &self.tensors[name];
            let is_vocab_tensor = matches!(
                name.as_str(),
                "token_embd.weight" | "output.weight" | "per_layer_token_embd.weight"
            );
            let vocab_sliced = prune_plan.is_some() && is_vocab_tensor && info.dims.len() == 2;
            let mut new_dims = info.dims.clone();
            if let Some(ref plan) = prune_plan
                && vocab_sliced
            {
                new_dims[1] = plan.kept_old_ids.len();
            }

            let new_dtype = if info.dims.len() == 2 {
                let cols = new_dims[0];
                let rows = new_dims[1];
                if name == "per_layer_token_embd.weight" {
                    info.dtype
                } else if name == "output.weight"
                    || (name == "token_embd.weight" && !has_separate_output)
                {
                    match opts.lm_head_quant.or(default_quant) {
                        Some(req) => resolve_quant_dtype(req, cols, rows, info.dtype),
                        None => info.dtype,
                    }
                } else if name == "token_embd.weight" && has_separate_output {
                    // Untied input embedding: keep compact `<= 4.5 bpw` types untouched; if wider (`F16`/`Q8_0`), compress to `IQ4_NL` when `--quant` is low-bit.
                    match default_quant {
                        Some(req)
                            if matches!(
                                req,
                                DType::Q2_0
                                    | DType::TQ2_0
                                    | DType::Iq2Xxs
                                    | DType::Iq3Xxs
                                    | DType::Iq4Nl
                            ) && info.dtype.block_bytes() * 32
                                > 18 * info.dtype.block_elements() =>
                        {
                            resolve_quant_dtype(DType::Iq4Nl, cols, rows, info.dtype)
                        }
                        _ => info.dtype,
                    }
                } else {
                    match default_quant {
                        Some(req) => resolve_quant_dtype(req, cols, rows, info.dtype),
                        None => info.dtype,
                    }
                }
            } else {
                info.dtype
            };

            if new_dtype != info.dtype || vocab_sliced {
                converted_tensors += 1;
            }

            let n_elems: usize = new_dims.iter().product();
            let byte_len = (n_elems / new_dtype.block_elements()) * new_dtype.block_bytes();
            let aligned_off = cur_data_off.next_multiple_of(self.alignment);
            cur_data_off = aligned_off + byte_len;

            plans.push(OutTensorPlan {
                name: name.clone(),
                dims: new_dims,
                new_dtype,
                byte_len,
                offset: aligned_off,
                vocab_sliced,
            });
        }

        // 3. Serialize GGUF v3 header + sorted KV metadata + tensor directory into `hdr_buf`
        let mut hdr_buf: Vec<u8> = Vec::with_capacity(self.data_offset + 4096);
        let io_err = |source: std::io::Error| GgufError::Io {
            path: out_str.clone(),
            source,
        };
        hdr_buf.extend_from_slice(GGUF_MAGIC);
        hdr_buf.extend_from_slice(&3u32.to_le_bytes());
        hdr_buf.extend_from_slice(&(plans.len() as u64).to_le_bytes());
        hdr_buf.extend_from_slice(&(meta.len() as u64).to_le_bytes());

        let mut sorted_keys: Vec<&String> = meta.keys().collect();
        sorted_keys.sort();
        for k in sorted_keys {
            write_gguf_str(&mut hdr_buf, k, &out_str)?;
            write_gguf_val(&mut hdr_buf, &meta[k], raw, true, &out_str)?;
        }

        for p in &plans {
            write_gguf_str(&mut hdr_buf, &p.name, &out_str)?;
            hdr_buf.extend_from_slice(&(p.dims.len() as u32).to_le_bytes());
            for &d in &p.dims {
                hdr_buf.extend_from_slice(&(d as u64).to_le_bytes());
            }
            hdr_buf.extend_from_slice(&p.new_dtype.to_ggml_type().to_le_bytes());
            hdr_buf.extend_from_slice(&(p.offset as u64).to_le_bytes());
        }

        let new_data_offset = hdr_buf.len().next_multiple_of(self.alignment);
        hdr_buf.resize(new_data_offset, 0u8);

        // 4. Stream header + converted/sliced tensor payloads to `out_path`
        let file = File::create(out_path).map_err(io_err)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        writer.write_all(&hdr_buf).map_err(io_err)?;

        let mut written_data = 0usize;
        let mut f32_row: Vec<f32> = Vec::new();
        let mut out_row_buf: Vec<u8> = Vec::new();
        let mut q4_group_raw: Vec<u8> = Vec::new();
        let mut q4_group_rep: Vec<u8> = Vec::new();
        let zero_pad = [0u8; 64];

        for p in &plans {
            while written_data < p.offset {
                let pad_n = (p.offset - written_data).min(zero_pad.len());
                writer.write_all(&zero_pad[..pad_n]).map_err(io_err)?;
                written_data += pad_n;
            }

            let src_info = &self.tensors[&p.name];
            let src_bytes = &raw[self.data_offset + src_info.offset
                ..self.data_offset + src_info.offset + src_info.byte_len];

            if !p.vocab_sliced && p.new_dtype == src_info.dtype {
                // Direct zero-copy byte stream
                writer.write_all(src_bytes).map_err(io_err)?;
            } else if !p.vocab_sliced
                && src_info.dtype == DType::Q4_0
                && p.new_dtype == DType::Q4_0x4
            {
                // Lossless `Q4_0 -> Q4_0x4` super-block repack
                let cols = p.dims[0];
                let rows = p.dims[1];
                let group_bytes = (cols / QK4_0) * 72;
                q4_group_rep.resize(group_bytes, 0);
                for rg in 0..(rows / 4) {
                    let s0 = rg * group_bytes;
                    repack_q4_0_bytes_to_q4_0x4(
                        &src_bytes[s0..s0 + group_bytes],
                        &mut q4_group_rep,
                        4,
                        cols,
                    );
                    writer.write_all(&q4_group_rep).map_err(io_err)?;
                }
            } else if p.vocab_sliced
                && p.new_dtype == src_info.dtype
                && src_info.dtype != DType::Q4_0x4
            {
                // Lossless row-slicing without re-quantization!
                let cols = p.dims[0];
                let rb = src_info.dtype.row_bytes(cols);
                let kept = &prune_plan
                    .as_ref()
                    .expect("vocab_sliced implies plan")
                    .kept_old_ids;
                for &old_r in kept {
                    writer
                        .write_all(&src_bytes[old_r * rb..(old_r + 1) * rb])
                        .map_err(io_err)?;
                }
            } else {
                // General row-streaming dequantize -> quantize (and optional 4-row `Q4_0x4` repack)
                let cols = p.dims[0];
                let out_rows = p.dims[1];
                f32_row.resize(cols, 0.0);

                let decode_src_row = |old_r: usize, dst_f32: &mut [f32]| {
                    if src_info.dtype == DType::Q4_0x4 {
                        dequantize_q4_0x4_single_row(src_bytes, cols, old_r, dst_f32);
                    } else {
                        let s_rb = src_info.dtype.row_bytes(cols);
                        dequantize_row_bytes(
                            src_info.dtype,
                            &src_bytes[old_r * s_rb..(old_r + 1) * s_rb],
                            dst_f32,
                        );
                    }
                };

                if p.new_dtype == DType::Q4_0x4 {
                    let row_q4_bytes = DType::Q4_0.row_bytes(cols);
                    let group_bytes = row_q4_bytes * 4;
                    q4_group_raw.resize(group_bytes, 0);
                    q4_group_rep.resize(group_bytes, 0);
                    for rg in 0..(out_rows / 4) {
                        for m in 0..4 {
                            let logical_r = rg * 4 + m;
                            let old_r = match prune_plan {
                                Some(ref plan) if p.vocab_sliced => plan.kept_old_ids[logical_r],
                                _ => logical_r,
                            };
                            decode_src_row(old_r, &mut f32_row);
                            quantize_row_to_bytes(
                                DType::Q4_0,
                                &f32_row,
                                &mut q4_group_raw[m * row_q4_bytes..(m + 1) * row_q4_bytes],
                            );
                        }
                        repack_q4_0_bytes_to_q4_0x4(&q4_group_raw, &mut q4_group_rep, 4, cols);
                        writer.write_all(&q4_group_rep).map_err(io_err)?;
                    }
                } else {
                    let out_rb = p.new_dtype.row_bytes(cols);
                    out_row_buf.resize(out_rb, 0);
                    for logical_r in 0..out_rows {
                        let old_r = match prune_plan {
                            Some(ref plan) if p.vocab_sliced => plan.kept_old_ids[logical_r],
                            _ => logical_r,
                        };
                        decode_src_row(old_r, &mut f32_row);
                        quantize_row_to_bytes(p.new_dtype, &f32_row, &mut out_row_buf);
                        writer.write_all(&out_row_buf).map_err(io_err)?;
                    }
                }
            }

            written_data += p.byte_len;
        }

        writer.flush().map_err(io_err)?;
        let output_file_bytes = new_data_offset + written_data;

        Ok(ConvertReport {
            total_tensors: self.tensor_order.len(),
            converted_tensors,
            old_vocab_size: old_vocab,
            new_vocab_size: new_vocab,
            old_merges,
            kept_merges: kept_merges_cnt,
            input_file_bytes: raw.len(),
            output_file_bytes,
        })
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    path: &'a str,
}

impl<'a> Cursor<'a> {
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GgufError::UnexpectedEof {
                path: self.path.to_string(),
                offset: self.pos,
            })?;
        if end > self.data.len() {
            return Err(GgufError::UnexpectedEof {
                path: self.path.to_string(),
                offset: self.pos,
            });
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }
    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let start = self.pos;
        let len = usize::try_from(self.read_u64()?).map_err(|_| GgufError::UnexpectedEof {
            path: self.path.to_string(),
            offset: start,
        })?;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| GgufError::InvalidUtf8 {
                path: self.path.to_string(),
                offset: start,
            })
    }

    fn read_value(&mut self, key: &str, val_type: u32) -> Result<MetadataValue, GgufError> {
        match val_type {
            0 => Ok(MetadataValue::U8(self.read_u8()?)),
            1 => Ok(MetadataValue::I8(self.read_i8()?)),
            2 => Ok(MetadataValue::U16(self.read_u16()?)),
            3 => Ok(MetadataValue::I16(self.read_i16()?)),
            4 => Ok(MetadataValue::U32(self.read_u32()?)),
            5 => Ok(MetadataValue::I32(self.read_i32()?)),
            6 => Ok(MetadataValue::F32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_u8()? != 0)),
            8 => Ok(MetadataValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let len =
                    usize::try_from(self.read_u64()?).map_err(|_| GgufError::UnexpectedEof {
                        path: self.path.to_string(),
                        offset: self.pos,
                    })?;
                // Guard against absurdly large crafted lengths before allocation.
                if len > self.data.len() {
                    return Err(GgufError::UnexpectedEof {
                        path: self.path.to_string(),
                        offset: self.pos,
                    });
                }
                match elem_type {
                    4 => {
                        let mut v = Vec::with_capacity(len.min(1_048_576));
                        for _ in 0..len {
                            v.push(self.read_u32()?);
                        }
                        Ok(MetadataValue::U32Array(v))
                    }
                    5 => {
                        let mut v = Vec::with_capacity(len.min(1_048_576));
                        for _ in 0..len {
                            v.push(self.read_i32()?);
                        }
                        Ok(MetadataValue::I32Array(v))
                    }
                    6 => {
                        let mut v = Vec::with_capacity(len.min(1_048_576));
                        for _ in 0..len {
                            v.push(self.read_f32()?);
                        }
                        Ok(MetadataValue::F32Array(v))
                    }
                    8 if self.pos < (u32::MAX as usize / 2) => {
                        let mut spans = Vec::with_capacity(len.min(1_048_576));
                        let mut total_bytes = 0usize;
                        for _ in 0..len {
                            let hdr_pos = self.pos;
                            let slen = usize::try_from(self.read_u64()?).map_err(|_| {
                                GgufError::UnexpectedEof {
                                    path: self.path.to_string(),
                                    offset: hdr_pos,
                                }
                            })?;
                            let str_start = self.pos;
                            let bytes = self.read_bytes(slen)?;
                            if str_start > u32::MAX as usize || slen > u32::MAX as usize {
                                return Err(GgufError::UnexpectedEof {
                                    path: self.path.to_string(),
                                    offset: hdr_pos,
                                });
                            }
                            std::str::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8 {
                                path: self.path.to_string(),
                                offset: hdr_pos,
                            })?;
                            spans.push((str_start as u32, slen as u32));
                            total_bytes += slen;
                        }
                        Ok(MetadataValue::StringSpans { spans, total_bytes })
                    }
                    _ => {
                        let mut items = Vec::with_capacity(len.min(1_048_576));
                        for _ in 0..len {
                            items.push(self.read_value(key, elem_type)?);
                        }
                        Ok(MetadataValue::Array(items))
                    }
                }
            }
            10 => Ok(MetadataValue::U64(self.read_u64()?)),
            11 => Ok(MetadataValue::I64(self.read_i64()?)),
            12 => Ok(MetadataValue::F64(self.read_f64()?)),
            other => Err(GgufError::UnknownValueType {
                path: self.path.to_string(),
                key: key.to_string(),
                val_type: other,
            }),
        }
    }
}

/// Pre-allocated 4096-byte page-aligned buffer for zero-allocation `O_DIRECT` NVMe reads.
///
/// Implemented in 100% safe Rust (zero `unsafe` blocks) by over-allocating `4096` bytes and
/// computing the runtime alignment offset once at construction.
pub struct PageAlignedSlot {
    buf: Vec<u8>,
    base_pad: usize,
    cap_aligned: usize,
}

impl PageAlignedSlot {
    pub fn new(min_bytes: usize) -> Self {
        // Add 2 extra 4 KiB pages for unaligned head/tail file offsets + 1 page for base pointer alignment.
        let cap_aligned = (min_bytes + 8192).next_multiple_of(4096);
        let buf = vec![0u8; cap_aligned + 4096];
        let addr = buf.as_ptr() as usize;
        let base_pad = (4096 - (addr & 4095)) & 4095;
        debug_assert_eq!((addr + base_pad) & 4095, 0);
        Self {
            buf,
            base_pad,
            cap_aligned,
        }
    }

    #[inline]
    pub fn aligned_mut(&mut self, aligned_len: usize) -> &mut [u8] {
        assert!(aligned_len <= self.cap_aligned && aligned_len.is_multiple_of(4096));
        &mut self.buf[self.base_pad..self.base_pad + aligned_len]
    }

    #[inline]
    pub fn subslice(&self, offset: usize, len: usize) -> &[u8] {
        &self.buf[self.base_pad + offset..self.base_pad + offset + len]
    }
}

/// Uncached (`O_DIRECT` on Linux when supported, standard `pread` fallback) random-access GGUF
/// tensor & MoE expert reader (`ADR-015`).
///
/// Keeps multi-GiB Tier 0 / Tier 1 / Tier 2 expert reads completely out of the OS page cache
/// while performing zero heap allocations on the token generation path.
pub struct DirectTensorReader {
    path: String,
    #[cfg(unix)]
    direct_file: Option<File>,
    fallback_file: File,
}

impl DirectTensorReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let p = path.as_ref();
        let path_str = p.display().to_string();
        let fallback_file = File::open(p).map_err(|source| GgufError::Io {
            path: path_str.clone(),
            source,
        })?;

        #[cfg(target_os = "linux")]
        let direct_file = {
            use std::os::unix::fs::OpenOptionsExt;
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            const O_DIRECT_FLAG: i32 = 0x10000;
            #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
            const O_DIRECT_FLAG: i32 = 0x4000;

            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(O_DIRECT_FLAG)
                .open(p)
                .ok()
        };
        #[cfg(all(unix, not(target_os = "linux")))]
        let direct_file: Option<File> = None;

        Ok(Self {
            path: path_str,
            #[cfg(unix)]
            direct_file,
            fallback_file,
        })
    }

    pub fn is_o_direct(&self) -> bool {
        #[cfg(unix)]
        {
            self.direct_file.is_some()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    /// Reads `[file_offset .. file_offset + len)` directly into `slot` and returns the exact
    /// sub-slice `&[u8]` inside `slot` (preserving 32-byte GGUF block alignment without extra copies!).
    pub fn read_into_slot<'a>(
        &self,
        file_offset: u64,
        len: usize,
        slot: &'a mut PageAlignedSlot,
    ) -> Result<&'a [u8], GgufError> {
        let aligned_start = file_offset & !4095u64;
        let head_pad = (file_offset - aligned_start) as usize;
        let aligned_len = (head_pad + len).next_multiple_of(4096);

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let dst = slot.aligned_mut(aligned_len);
            if let Some(ref df) = self.direct_file {
                let mut read_total = 0usize;
                let mut ok = true;
                while read_total < aligned_len {
                    match df.read_at(&mut dst[read_total..], aligned_start + read_total as u64) {
                        Ok(0) => break,
                        Ok(n) => read_total += n,
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && read_total >= head_pad + len {
                    return Ok(slot.subslice(head_pad, len));
                }
            }
            // Fallback (e.g., EOF tail page shorter than 4096 bytes or filesystem without O_DIRECT)
            self.fallback_file
                .read_exact_at(&mut dst[head_pad..head_pad + len], file_offset)
                .map_err(|source| GgufError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            Ok(slot.subslice(head_pad, len))
        }

        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = self
                .fallback_file
                .try_clone()
                .map_err(|source| GgufError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            f.seek(SeekFrom::Start(file_offset))
                .map_err(|source| GgufError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            let dst = slot.aligned_mut(aligned_len);
            f.read_exact(&mut dst[head_pad..head_pad + len])
                .map_err(|source| GgufError::Io {
                    path: self.path.clone(),
                    source,
                })?;
            Ok(slot.subslice(head_pad, len))
        }
    }

    /// Streams a large tensor range `[file_offset .. file_offset + dst.len())` into `dst` in
    /// reusable `O_DIRECT` chunks (`slot`), leaving 0 bytes of clean file cache behind in RAM.
    pub fn read_exact_uncached(
        &self,
        file_offset: u64,
        dst: &mut [u8],
        slot: &mut PageAlignedSlot,
    ) -> Result<(), GgufError> {
        let max_chunk = slot.cap_aligned.saturating_sub(8192).max(4096);
        let mut done = 0usize;
        while done < dst.len() {
            let take = (dst.len() - done).min(max_chunk);
            let s = self.read_into_slot(file_offset + done as u64, take, slot)?;
            dst[done..done + take].copy_from_slice(s);
            done += take;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_inputs_cleanly() {
        // Empty / truncated
        let err = GgufFile::from_bytes("empty.gguf", vec![]).err().unwrap();
        assert!(err.to_string().contains("empty.gguf"));

        // Bad magic
        let err = GgufFile::from_bytes("bad.gguf", b"NOPE1234".to_vec())
            .err()
            .unwrap();
        assert!(matches!(err, GgufError::InvalidMagic { .. }));

        // Unsupported version 99
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let err = GgufFile::from_bytes("v99.gguf", buf).err().unwrap();
        assert!(matches!(
            err,
            GgufError::UnsupportedVersion { version: 99, .. }
        ));

        // Crafted huge n_kv (must not OOM, must return UnexpectedEof)
        let mut buf = Vec::new();
        buf.extend_from_slice(&b"GGUF"[..]);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&u64::MAX.to_le_bytes());
        let err = GgufFile::from_bytes("huge_kv.gguf", buf).err().unwrap();
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }

    #[test]
    fn direct_tensor_reader_matches_mmap_bytes_and_preserves_alignment() {
        let mut slot = PageAlignedSlot::new(65536);
        let ptr = slot.aligned_mut(4096).as_ptr() as usize;
        assert_eq!(ptr % 4096, 0, "PageAlignedSlot base must be 4096-aligned");

        // Test against a temporary file with non-4096-aligned 32-byte-aligned offsets & EOF tail
        let tmp_path =
            std::env::temp_dir().join(format!("tm_direct_reader_test_{}.bin", std::process::id()));
        let total = 20_000usize; // deliberately not a multiple of 4096 to exercise EOF fallback too
        let expected: Vec<u8> = (0..total)
            .map(|i| ((i.wrapping_mul(131).wrapping_add(17)) & 0xFF) as u8)
            .collect();
        std::fs::write(&tmp_path, &expected).unwrap();

        let reader = DirectTensorReader::open(&tmp_path).unwrap();
        for &(off, len) in &[(0usize, 4096usize), (96, 8192), (4128, 5024), (16032, 3968)] {
            let got = reader.read_into_slot(off as u64, len, &mut slot).unwrap();
            assert_eq!(
                got,
                &expected[off..off + len],
                "mismatch at off={off} len={len}"
            );
            assert_eq!(
                (got.as_ptr() as usize) % 32,
                off % 32,
                "32-byte GGUF alignment modulo must be preserved"
            );
        }

        let mut full_copy = vec![0u8; 12_345];
        reader
            .read_exact_uncached(320, &mut full_copy, &mut slot)
            .unwrap();
        assert_eq!(&full_copy[..], &expected[320..320 + 12_345]);
        let _ = std::fs::remove_file(&tmp_path);

        // If dspark GGUF is present locally, also verify req_tensor_file_range vs req_tensor_bytes
        let dspark = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf");
        if dspark.exists() {
            let gguf = GgufFile::open(&dspark).unwrap();
            let dreader = DirectTensorReader::open(&dspark).unwrap();
            let mut dslot = PageAlignedSlot::new(16 * 1024 * 1024);
            for tname in gguf.tensor_names().iter().take(4) {
                let (info, mmap_bytes) = gguf.tensor_bytes(tname, "test").unwrap();
                let (_, f_off, b_len) = gguf
                    .req_tensor_file_range(tname, "test", &info.dims)
                    .unwrap();
                assert_eq!(b_len, mmap_bytes.len());
                let direct_bytes = dreader.read_into_slot(f_off, b_len, &mut dslot).unwrap();
                assert_eq!(
                    direct_bytes, mmap_bytes,
                    "DirectTensorReader mismatch on {tname}"
                );
            }
        }
    }
}
