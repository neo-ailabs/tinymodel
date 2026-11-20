//! Zero-copy GGUF v2/v3 parser and memory-mapped tensor loader.
//!
//! `unsafe` is restricted strictly to `memmap2::Mmap::map` with a `// SAFETY:` comment.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;
use tm_core::DType;

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
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&u64::MAX.to_le_bytes());
        let err = GgufFile::from_bytes("huge_kv.gguf", buf).err().unwrap();
        assert!(matches!(err, GgufError::UnexpectedEof { .. }));
    }
}
