//! Quantization block layouts and data types.
//!
//! Block layouts (`BlockQ4_0`, `BlockQ4_1`, `BlockQ8_0`, `BlockQ8_1`, `BlockQ5K`, `BlockQ6K`, `BlockQ8K`)
//! match ggml's binary GGUF format:
//! Copyright (c) 2023-2026 The ggml authors (MIT License).
//! Upstream reference: llama.cpp `ggml/src/ggml-common.h` (commit `3cf03257` / `b001d7b`).

use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::iq_tables::{IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_FP4, KVALUES_IQ4NL};

/// Number of weights per `BlockQ2_0` (2.25 bpw).
pub const QK2_0: usize = 64;
/// Number of weights per `BlockNvFp4` (4.50 bpw, 4 sub-blocks of 16 elements).
pub const QK_NVFP4: usize = 64;
/// Sub-block size per `UE4M3` scale in `BlockNvFp4`.
pub const QK_NVFP4_SUB: usize = 16;
/// Number of weights per `BlockQ4_0` / `BlockQ4_1`.
pub const QK4_0: usize = 32;
/// Number of weights per `BlockIQ4NL` (4.50 bpw non-linear codebook).
pub const QK4_NL: usize = 32;
/// Number of weights per `BlockQ4_0x4` (4 output rows x 32 columns).
pub const QK4_0X4: usize = 128;
pub const QK4_1: usize = 32;
/// Number of weights per `BlockQ8_0` / `BlockQ8_1`.
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
/// Number of weights per K-quant / ternary / i-quant super-block (`256`).
pub const QK_K: usize = 256;
/// Custom `tinymodel` GGUF tensor type ID for 4-row interleaved `BlockQ4_0x4` (`Q4_0_4x8`).
pub const GGML_TYPE_TM_Q4_0X4: u32 = 100;

/// KV cache storage precision (`F-14`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvType {
    /// IEEE 754 binary16 (`2.0` bytes/element, default).
    #[default]
    F16,
    /// Symmetric 8-bit block quantization (`BlockQ8_0`, `1.0625` bytes/element — ~47% KV savings).
    Q8_0,
}

impl KvType {
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }
}

impl core::str::FromStr for KvType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("f16") || s.eq_ignore_ascii_case("fp16") {
            Ok(KvType::F16)
        } else if s.eq_ignore_ascii_case("q8_0") || s.eq_ignore_ascii_case("q8") {
            Ok(KvType::Q8_0)
        } else {
            Err("unsupported --kv-type (expected 'f16' or 'q8_0')")
        }
    }
}

/// Supported tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q2_0,
    TQ2_0,
    Iq2Xxs,
    Iq3Xxs,
    Iq4Nl,
    Iq4Xs,
    NvFp4,
    Q4_0,
    /// Offline 4-row x 32-col interleaved `Q4_0` layout (`72` bytes = `4 * 18` bytes, zero overhead).
    Q4_0x4,
    Q4_1,
    Q5_0,
    Q8_0,
    Q8_1,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl DType {
    /// Maps a GGUF / ggml `ggml_type` integer ID to `DType`.
    pub const fn from_ggml_type(id: u32) -> Option<Self> {
        match id {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            3 => Some(DType::Q4_1),
            6 => Some(DType::Q5_0),
            8 => Some(DType::Q8_0),
            9 => Some(DType::Q8_1),
            11 => Some(DType::Q3K),
            12 => Some(DType::Q4K),
            13 => Some(DType::Q5K),
            14 => Some(DType::Q6K),
            15 => Some(DType::Q8K),
            16 => Some(DType::Iq2Xxs),
            18 => Some(DType::Iq3Xxs),
            20 => Some(DType::Iq4Nl),
            23 => Some(DType::Iq4Xs),
            30 => Some(DType::BF16),
            35 => Some(DType::TQ2_0),
            40 => Some(DType::NvFp4),
            42 => Some(DType::Q2_0),
            GGML_TYPE_TM_Q4_0X4 => Some(DType::Q4_0x4),
            _ => None,
        }
    }

    /// Maps `DType` back to its GGUF `ggml_type` integer ID.
    pub const fn to_ggml_type(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q4_0 => 2,
            DType::Q4_1 => 3,
            DType::Q5_0 => 6,
            DType::Q8_0 => 8,
            DType::Q8_1 => 9,
            DType::Q3K => 11,
            DType::Q4K => 12,
            DType::Q5K => 13,
            DType::Q6K => 14,
            DType::Q8K => 15,
            DType::Iq2Xxs => 16,
            DType::Iq3Xxs => 18,
            DType::Iq4Nl => 20,
            DType::Iq4Xs => 23,
            DType::BF16 => 30,
            DType::TQ2_0 => 35,
            DType::NvFp4 => 40,
            DType::Q2_0 => 42,
            DType::Q4_0x4 => GGML_TYPE_TM_Q4_0X4,
        }
    }

    /// Canonical lowercase CLI name for this `DType`.
    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::Q2_0 => "q2_0",
            DType::TQ2_0 => "tq2_0",
            DType::Iq2Xxs => "iq2_xxs",
            DType::Iq3Xxs => "iq3_xxs",
            DType::Iq4Nl => "iq4_nl",
            DType::Iq4Xs => "iq4_xs",
            DType::NvFp4 => "nvfp4",
            DType::Q4_0 => "q4_0",
            DType::Q4_0x4 => "q4_0x4",
            DType::Q4_1 => "q4_1",
            DType::Q5_0 => "q5_0",
            DType::Q8_0 => "q8_0",
            DType::Q8_1 => "q8_1",
            DType::Q3K => "q3_k",
            DType::Q4K => "q4_k",
            DType::Q5K => "q5_k",
            DType::Q6K => "q6_k",
            DType::Q8K => "q8_k",
        }
    }

    /// Parses a CLI quantization target name (`q4_0x4`, `q2_0`, `tq2_0`, `iq2_xxs`, `iq3_xxs`, `iq4_nl`, `nvfp4`, `q4_0`, `q8_0`).
    pub fn from_cli_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" => Some(DType::F32),
            "f16" | "fp16" => Some(DType::F16),
            "bf16" => Some(DType::BF16),
            "q2_0" | "q2" => Some(DType::Q2_0),
            "tq2_0" | "tq2" | "ternary" => Some(DType::TQ2_0),
            "iq2_xxs" | "iq2" => Some(DType::Iq2Xxs),
            "iq3_xxs" | "iq3" => Some(DType::Iq3Xxs),
            "iq4_nl" | "iq4" => Some(DType::Iq4Nl),
            "iq4_xs" => Some(DType::Iq4Xs),
            "nvfp4" | "fp4" => Some(DType::NvFp4),
            "q4_0" => Some(DType::Q4_0),
            "q4_0x4" | "q4_0_4x8" => Some(DType::Q4_0x4),
            "q4_1" => Some(DType::Q4_1),
            "q5_0" => Some(DType::Q5_0),
            "q8_0" | "q8" => Some(DType::Q8_0),
            "q3_k" => Some(DType::Q3K),
            "q4_k" => Some(DType::Q4K),
            "q5_k" => Some(DType::Q5K),
            "q6_k" => Some(DType::Q6K),
            _ => None,
        }
    }

    /// Block size in elements.
    pub const fn block_elements(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q2_0 => QK2_0,
            DType::NvFp4 => QK_NVFP4,
            DType::Q4_0 | DType::Q5_0 => QK4_0,
            DType::Iq4Nl => QK4_NL,
            DType::Q4_0x4 => QK4_0X4,
            DType::Q4_1 => QK4_1,
            DType::Q8_0 => QK8_0,
            DType::Q8_1 => QK8_1,
            DType::TQ2_0
            | DType::Iq2Xxs
            | DType::Iq3Xxs
            | DType::Iq4Xs
            | DType::Q3K
            | DType::Q4K
            | DType::Q5K
            | DType::Q6K
            | DType::Q8K => QK_K,
        }
    }

    /// Size of one block in bytes.
    pub const fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q2_0 => core::mem::size_of::<BlockQ2_0>(),
            DType::TQ2_0 => core::mem::size_of::<BlockTQ2_0>(),
            DType::Iq2Xxs => core::mem::size_of::<BlockIQ2XXS>(),
            DType::Iq3Xxs => core::mem::size_of::<BlockIQ3XXS>(),
            DType::Iq4Nl => core::mem::size_of::<BlockIQ4NL>(),
            DType::Iq4Xs => 136,
            DType::NvFp4 => core::mem::size_of::<BlockNvFp4>(),
            DType::Q4_0 => core::mem::size_of::<BlockQ4_0>(),
            DType::Q4_0x4 => core::mem::size_of::<BlockQ4_0x4>(),
            DType::Q4_1 => core::mem::size_of::<BlockQ4_1>(),
            DType::Q5_0 => 22,
            DType::Q8_0 => core::mem::size_of::<BlockQ8_0>(),
            DType::Q8_1 => core::mem::size_of::<BlockQ8_1>(),
            DType::Q3K => 110,
            DType::Q4K => 144,
            DType::Q5K => core::mem::size_of::<BlockQ5K>(),
            DType::Q6K => core::mem::size_of::<BlockQ6K>(),
            DType::Q8K => core::mem::size_of::<BlockQ8K>(),
        }
    }

    /// Average byte size per logical row of `n_elements` columns.
    /// For `Q4_0x4`, 4 rows occupy `(n_elements / 32) * 72` bytes, so 1 row averages `(n_elements / 32) * 18` bytes.
    pub const fn row_bytes(self, n_elements: usize) -> usize {
        match self {
            DType::Q4_0x4 => (n_elements / QK4_0) * core::mem::size_of::<BlockQ4_0>(),
            _ => (n_elements / self.block_elements()) * self.block_bytes(),
        }
    }
}

/// 4-row x 32-column interleaved 4-bit symmetric quantization super-block (128 weights in 72 bytes).
///
/// Layout (`align = 2`, exact `4 * sizeof(BlockQ4_0)` = 72 bytes — zero byte overhead):
/// - `d: [u16; 4]`: IEEE-754 `f16` scales for `[r0, r1, r2, r3]`.
/// - `qs[0..32]` (rows `r0, r1` pre-zipped for zero-shuffle `smmla` / `sdot`):
///   - `qs[0..8]`   = `r0.qs[0..8]`  (low nibbles `k=0..8`,  high nibbles `k=16..24`)
///   - `qs[8..16]`  = `r1.qs[0..8]`  (low nibbles `k=0..8`,  high nibbles `k=16..24`)
///   - `qs[16..24]` = `r0.qs[8..16]` (low nibbles `k=8..16`, high nibbles `k=24..32`)
///   - `qs[24..32]` = `r1.qs[8..16]` (low nibbles `k=8..16`, high nibbles `k=24..32`)
/// - `qs[32..64]` (rows `r2, r3` in identical order).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4_0x4 {
    pub d: [u16; 4],
    pub qs: [u8; 64],
}

/// 2-bit asymmetric-center quantization block (`ggml_type = 42`, 64 weights in 18 bytes = 2.25 bpw).
/// Each 2-bit code `c in 0..4` decodes to `((c as i32) - 1) * d` in `{-d, 0, +d, +2d}`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ2_0 {
    pub d: u16,
    pub qs: [u8; QK2_0 / 4],
}

/// BitNet b1.58 ternary quantization block (`ggml_type = 35`, 256 weights in 66 bytes = 2.0625 bpw).
/// Note field order matches `ggml-common.h`: `qs` (`[u8; 64]`) precedes `d` (`f16` at offset 64).
/// Each 2-bit code `c in 0..3` decodes to `((c as i32) - 1) * d` in `{-d, 0, +d}`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTQ2_0 {
    pub qs: [u8; QK_K / 4],
    pub d: u16,
}

/// Importance-matrix 2-bit codebook block (`ggml_type = 16`, 256 weights in 66 bytes = 2.0625 bpw).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ2XXS {
    pub d: u16,
    pub qs: [u16; QK_K / 8],
}

/// Importance-matrix 3-bit codebook block (`ggml_type = 18`, 256 weights in 98 bytes = 3.0625 bpw).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ3XXS {
    pub d: u16,
    pub qs: [u8; 3 * QK_K / 8],
}

/// Non-linear 4-bit codebook block (`ggml_type = 20`, 32 weights in 18 bytes = 4.50 bpw).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ4NL {
    pub d: u16,
    pub qs: [u8; QK4_NL / 2],
}

/// NVIDIA ModelOpt 4-bit FP4 (`E2M1`) super-block (`ggml_type = 40`, 64 weights in 36 bytes = 4.50 bpw).
/// Contains 4 unsigned `UE4M3` sub-block scales (one per 16 elements) and 32 packed nibble bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockNvFp4 {
    pub d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    pub qs: [u8; QK_NVFP4 / 2],
}

/// Converts an unsigned `UE4M3` scale byte to `f32 * 0.5` matching `ggml_ue4m3_to_fp32` bit-for-bit
/// (`0.5` compensates `KVALUES_FP4 = 2 * E2M1_float`).
#[inline]
pub const fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0x0F) as i32;
    let man = (x & 0x07) as i32;
    if exp == 0 {
        (man as f32) * (1.0 / 1024.0)
    } else {
        let pow2 = f32::from_bits(((127 + exp - 8) as u32) << 23);
        (1.0 + (man as f32) * 0.125) * pow2
    }
}

/// Precomputed 256-entry `UE4M3 -> f32 * 0.5` lookup table (`1 KiB` L1-resident).
pub const UE4M3_TO_F32_LUT: [f32; 256] = {
    let mut tbl = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        tbl[i] = ue4m3_to_f32(i as u8);
        i += 1;
    }
    tbl
};

/// Encodes a non-negative `f32` scale (`amax / 6.0`) into `UE4M3` matching `ggml_fp32_to_ue4m3`.
#[inline]
pub fn f32_to_ue4m3(mut x: f32) -> u8 {
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    if x > 448.0 {
        x = 448.0;
    }
    let bits = x.to_bits();
    let fp32_exp = (((bits >> 23) & 0xFF) as i32) - 127;
    let fp32_man = ((bits >> 20) & 0x07) as i32;
    let ue_exp = fp32_exp + 7;
    if ue_exp <= 0 {
        // Subnormal UE4M3 (`val = man * 2^-9`)
        let sub = (x * 512.0).round() as i32;
        return sub.clamp(0, 7) as u8;
    }
    if ue_exp >= 15 {
        // Clamp away from `0x7F` NaN encoding (`max finite = 0x7E = 448.0`)
        let man_clamped = fp32_man.min(6);
        return ((15 << 3) | man_clamped) as u8;
    }
    let round_bit = (bits >> 19) & 1;
    let mut code = ((ue_exp << 3) | fp32_man) + (round_bit as i32);
    if code >= 0x7F {
        code = 0x7E;
    }
    code as u8
}

/// 4-bit symmetric quantization block (32 weights in 18 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; QK4_0 / 2],
}

/// 4-bit asymmetric quantization block (32 weights in 20 bytes): `w = q * d + m`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4_1 {
    pub d: u16,
    pub m: u16,
    pub qs: [u8; QK4_1 / 2],
}

/// 8-bit symmetric quantization block (32 weights in 34 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

/// 8-bit quantization block with precomputed `s = d * sum(qs)` (32 weights in 36 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ8_1 {
    pub d: u16,
    pub s: u16,
    pub qs: [i8; QK8_1],
}

/// 5-bit K-quant super-block (256 weights in 176 bytes): `w = d * sc * q5 - dmin * m`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ5K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

/// Unpacks the 12-byte 6-bit packed `scales` array of `BlockQ4_K` / `BlockQ5K`
/// into 8 6-bit sub-block scales (`sc[0..8]`) and 8 6-bit sub-block mins (`m[0..8]`).
#[inline]
pub const fn unpack_q5k_scales_mins(s: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let sc = [
        s[0] & 63,
        s[1] & 63,
        s[2] & 63,
        s[3] & 63,
        (s[8] & 0x0F) | ((s[0] >> 6) << 4),
        (s[9] & 0x0F) | ((s[1] >> 6) << 4),
        (s[10] & 0x0F) | ((s[2] >> 6) << 4),
        (s[11] & 0x0F) | ((s[3] >> 6) << 4),
    ];
    let m = [
        s[4] & 63,
        s[5] & 63,
        s[6] & 63,
        s[7] & 63,
        (s[8] >> 4) | ((s[4] >> 6) << 4),
        (s[9] >> 4) | ((s[5] >> 6) << 4),
        (s[10] >> 4) | ((s[6] >> 6) << 4),
        (s[11] >> 4) | ((s[7] >> 6) << 4),
    ];
    (sc, m)
}

/// 6-bit K-quant super-block (256 weights in 210 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: u16,
}

/// 8-bit K-quant activation super-block (256 weights in 292 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

const _: () = assert!(core::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(core::mem::size_of::<BlockQ4_0x4>() == 72);
const _: () = assert!(core::mem::size_of::<BlockQ4_1>() == 20);
const _: () = assert!(core::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(core::mem::size_of::<BlockQ8_1>() == 36);
const _: () = assert!(core::mem::size_of::<BlockQ5K>() == 176);
const _: () = assert!(core::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(core::mem::size_of::<BlockQ8K>() == 292);

/// Repacks `[rows, cols_blocks]` standard `BlockQ4_0` blocks (`rows % 4 == 0`) into
/// `[rows / 4, cols_blocks]` interleaved `BlockQ4_0x4` super-blocks (zero byte overhead).
pub fn repack_q4_0_to_q4_0x4(
    src: &[BlockQ4_0],
    dst: &mut [BlockQ4_0x4],
    rows: usize,
    cols_blocks: usize,
) {
    assert_eq!(rows % 4, 0, "Q4_0x4 requires rows divisible by 4");
    assert_eq!(src.len(), rows * cols_blocks);
    assert_eq!(dst.len(), (rows / 4) * cols_blocks);

    for r_group in 0..(rows / 4) {
        let r0 = &src[(r_group * 4) * cols_blocks..(r_group * 4 + 1) * cols_blocks];
        let r1 = &src[(r_group * 4 + 1) * cols_blocks..(r_group * 4 + 2) * cols_blocks];
        let r2 = &src[(r_group * 4 + 2) * cols_blocks..(r_group * 4 + 3) * cols_blocks];
        let r3 = &src[(r_group * 4 + 3) * cols_blocks..(r_group * 4 + 4) * cols_blocks];
        let dst_row = &mut dst[r_group * cols_blocks..(r_group + 1) * cols_blocks];

        for ib in 0..cols_blocks {
            let b0 = &r0[ib];
            let b1 = &r1[ib];
            let b2 = &r2[ib];
            let b3 = &r3[ib];
            let sb = &mut dst_row[ib];
            sb.d = [b0.d, b1.d, b2.d, b3.d];
            sb.qs[0..8].copy_from_slice(&b0.qs[0..8]);
            sb.qs[8..16].copy_from_slice(&b1.qs[0..8]);
            sb.qs[16..24].copy_from_slice(&b0.qs[8..16]);
            sb.qs[24..32].copy_from_slice(&b1.qs[8..16]);
            sb.qs[32..40].copy_from_slice(&b2.qs[0..8]);
            sb.qs[40..48].copy_from_slice(&b3.qs[0..8]);
            sb.qs[48..56].copy_from_slice(&b2.qs[8..16]);
            sb.qs[56..64].copy_from_slice(&b3.qs[8..16]);
        }
    }
}

/// Safe byte-slice version of `repack_q4_0_to_q4_0x4` for offline GGUF conversion (`#forbid(unsafe_code)`).
pub fn repack_q4_0_bytes_to_q4_0x4(src: &[u8], dst: &mut [u8], rows: usize, cols: usize) {
    assert_eq!(rows % 4, 0, "Q4_0x4 requires rows divisible by 4");
    assert_eq!(cols % QK4_0, 0, "Q4_0x4 requires cols divisible by 32");
    let cols_blocks = cols / QK4_0;
    let row_stride = cols_blocks * 18;
    assert_eq!(src.len(), rows * row_stride);
    assert_eq!(dst.len(), src.len());

    for r_group in 0..(rows / 4) {
        let r0 = &src[(r_group * 4) * row_stride..(r_group * 4 + 1) * row_stride];
        let r1 = &src[(r_group * 4 + 1) * row_stride..(r_group * 4 + 2) * row_stride];
        let r2 = &src[(r_group * 4 + 2) * row_stride..(r_group * 4 + 3) * row_stride];
        let r3 = &src[(r_group * 4 + 3) * row_stride..(r_group * 4 + 4) * row_stride];
        let dst_group = &mut dst[r_group * cols_blocks * 72..(r_group + 1) * cols_blocks * 72];

        for ib in 0..cols_blocks {
            let b0 = &r0[ib * 18..(ib + 1) * 18];
            let b1 = &r1[ib * 18..(ib + 1) * 18];
            let b2 = &r2[ib * 18..(ib + 1) * 18];
            let b3 = &r3[ib * 18..(ib + 1) * 18];
            let sb = &mut dst_group[ib * 72..(ib + 1) * 72];
            sb[0..2].copy_from_slice(&b0[0..2]);
            sb[2..4].copy_from_slice(&b1[0..2]);
            sb[4..6].copy_from_slice(&b2[0..2]);
            sb[6..8].copy_from_slice(&b3[0..2]);
            sb[8..16].copy_from_slice(&b0[2..10]);
            sb[16..24].copy_from_slice(&b1[2..10]);
            sb[24..32].copy_from_slice(&b0[10..18]);
            sb[32..40].copy_from_slice(&b1[10..18]);
            sb[40..48].copy_from_slice(&b2[2..10]);
            sb[48..56].copy_from_slice(&b3[2..10]);
            sb[56..64].copy_from_slice(&b2[10..18]);
            sb[64..72].copy_from_slice(&b3[10..18]);
        }
    }
}

/// Dequantizes a single row `row_idx` directly from a `DType::Q4_0x4` tensor (`[rows, cols]`).
/// Allows zero-copy embedding lookups even when `token_embd.weight` is tied with `output.weight` and repacked!
pub fn dequantize_q4_0x4_single_row(
    tensor_bytes: &[u8],
    cols: usize,
    row_idx: usize,
    dst: &mut [f32],
) {
    assert_eq!(cols % QK4_0, 0);
    assert_eq!(dst.len(), cols);
    let cols_blocks = cols / QK4_0;
    let r_group = row_idx / 4;
    let sub = row_idx % 4;
    let pair_off = if sub < 2 { 0 } else { 32 };
    let lane_off = if sub.is_multiple_of(2) { 0 } else { 8 };
    let group_bytes = &tensor_bytes[r_group * cols_blocks * 72..(r_group + 1) * cols_blocks * 72];

    for (ib, sb) in group_bytes.as_chunks::<72>().0.iter().enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([sb[sub * 2], sb[sub * 2 + 1]]));
        let qs = &sb[8..72];
        let base = ib * QK4_0;
        for j in 0..8 {
            let b_lo = qs[pair_off + lane_off + j];
            let b_hi = qs[pair_off + 16 + lane_off + j];
            dst[base + j] = (((b_lo & 0x0F) as i32) - 8) as f32 * d;
            dst[base + 8 + j] = (((b_hi & 0x0F) as i32) - 8) as f32 * d;
            dst[base + 16 + j] = (((b_lo >> 4) as i32) - 8) as f32 * d;
            dst[base + 24 + j] = (((b_hi >> 4) as i32) - 8) as f32 * d;
        }
    }
}

/// Quantizes `src` (`f32`, length multiple of 32) into `dst` (`BlockQ8_0`).
pub fn quantize_row_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    assert_eq!(src.len() % QK8_0, 0);
    let nb = src.len() / QK8_0;
    assert_eq!(dst.len(), nb);

    for (ib, block) in dst.iter_mut().enumerate() {
        let base = ib * QK8_0;
        let slice = &src[base..base + QK8_0];

        let mut amax = 0.0f32;
        for &v in slice {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f32_to_f16(d);

        for (j, &v) in slice.iter().enumerate() {
            block.qs[j] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Quantizes `src` (`f32`, length multiple of 32) into `dst` (`BlockQ8_1`).
pub fn quantize_row_q8_1(src: &[f32], dst: &mut [BlockQ8_1]) {
    assert_eq!(src.len() % QK8_1, 0);
    let nb = src.len() / QK8_1;
    assert_eq!(dst.len(), nb);

    for (ib, block) in dst.iter_mut().enumerate() {
        let base = ib * QK8_1;
        let slice = &src[base..base + QK8_1];

        let mut amax = 0.0f32;
        for &v in slice {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut sum: i32 = 0;
        for (j, &v) in slice.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            block.qs[j] = q;
            sum += q as i32;
        }
        block.d = f32_to_f16(d);
        block.s = f32_to_f16((sum as f32) * d);
    }
}

/// Quantizes `src` (`f32`, length multiple of 256) into `dst` (`BlockQ8K`).
pub fn quantize_row_q8_k(src: &[f32], dst: &mut [BlockQ8K]) {
    assert_eq!(src.len() % QK_K, 0);
    let nb = src.len() / QK_K;
    assert_eq!(dst.len(), nb);

    for (ib, block) in dst.iter_mut().enumerate() {
        let base = ib * QK_K;
        let slice = &src[base..base + QK_K];

        let mut max_v = 0.0f32;
        let mut amax = 0.0f32;
        for &v in slice {
            let av = v.abs();
            if av > amax {
                amax = av;
                max_v = v;
            }
        }

        if amax == 0.0 {
            block.d = 0.0;
            block.qs = [0; QK_K];
            block.bsums = [0; QK_K / 16];
            continue;
        }

        let iscale = -127.0 / max_v;
        for (q, &v) in block.qs.iter_mut().zip(slice.iter()) {
            *q = (iscale * v).round().clamp(-128.0, 127.0) as i8;
        }
        for j in 0..(QK_K / 16) {
            let mut sum: i32 = 0;
            for l in 0..16 {
                sum += block.qs[j * 16 + l] as i32;
            }
            block.bsums[j] = sum as i16;
        }
        block.d = 1.0 / iscale;
    }
}

/// Quantizes `src` (`f32`, length multiple of 32) into `dst` (`BlockQ4_0`).
pub fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    assert_eq!(src.len() % QK4_0, 0);
    let nb = src.len() / QK4_0;
    assert_eq!(dst.len(), nb);

    for (ib, block) in dst.iter_mut().enumerate() {
        let base = ib * QK4_0;
        let slice = &src[base..base + QK4_0];

        let mut max_v = 0.0f32;
        let mut amax = 0.0f32;
        for &v in slice {
            let av = v.abs();
            if av > amax {
                amax = av;
                max_v = v;
            }
        }

        let d = max_v / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f32_to_f16(d);

        for j in 0..(QK4_0 / 2) {
            let x0 = slice[j] * id;
            let x1 = slice[j + QK4_0 / 2] * id;

            let xi0 = ((x0 + 8.5) as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5) as i32).clamp(0, 15) as u8;

            block.qs[j] = xi0 | (xi1 << 4);
        }
    }
}

/// Dequantizes `src` (`BlockQ4_0`) into `dst` (`f32`).
pub fn dequantize_row_q4_0(src: &[BlockQ4_0], dst: &mut [f32]) {
    assert_eq!(dst.len(), src.len() * QK4_0);
    for (ib, block) in src.iter().enumerate() {
        let d = f16_to_f32(block.d);
        let base = ib * QK4_0;
        for j in 0..(QK4_0 / 2) {
            let x0 = ((block.qs[j] & 0x0F) as i32) - 8;
            let x1 = ((block.qs[j] >> 4) as i32) - 8;
            dst[base + j] = (x0 as f32) * d;
            dst[base + j + QK4_0 / 2] = (x1 as f32) * d;
        }
    }
}

/// Dequantizes raw little-endian bytes of any supported `DType` into `dst` (`f32`).
/// Used for zero-copy embedding table lookups without unsafe pointer casts.
pub fn dequantize_row_bytes(dtype: DType, src: &[u8], dst: &mut [f32]) {
    let n = dst.len();
    assert_eq!(n % dtype.block_elements(), 0);
    assert_eq!(src.len(), dtype.row_bytes(n));

    match dtype {
        DType::F32 => {
            for (d, &chunk) in dst.iter_mut().zip(src.as_chunks::<4>().0.iter()) {
                *d = f32::from_le_bytes(chunk);
            }
        }
        DType::F16 => {
            for (d, &chunk) in dst.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                *d = f16_to_f32(u16::from_le_bytes(chunk));
            }
        }
        DType::BF16 => {
            for (d, &chunk) in dst.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                let bits = (u16::from_le_bytes(chunk) as u32) << 16;
                *d = f32::from_bits(bits);
            }
        }
        DType::Q4_0 => {
            for (ib, blk) in src.as_chunks::<18>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let base = ib * QK4_0;
                for j in 0..16 {
                    let b = blk[2 + j];
                    dst[base + j] = (((b & 0x0F) as i32) - 8) as f32 * d;
                    dst[base + j + 16] = (((b >> 4) as i32) - 8) as f32 * d;
                }
            }
        }
        DType::Q4_0x4 => {
            // `src` holds `[n / 4` rows or 4-row groups]; dequantizes sequentially in logical row order
            // when `n` is `4 * cols` (`n % 128 == 0`).
            let cols = n / 4;
            for sub in 0..4 {
                dequantize_q4_0x4_single_row(
                    src,
                    cols,
                    sub,
                    &mut dst[sub * cols..(sub + 1) * cols],
                );
            }
        }
        DType::Q4_1 => {
            for (ib, blk) in src.as_chunks::<20>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let base = ib * QK4_1;
                for j in 0..16 {
                    let b = blk[4 + j];
                    dst[base + j] = ((b & 0x0F) as f32) * d + m;
                    dst[base + j + 16] = ((b >> 4) as f32) * d + m;
                }
            }
        }
        DType::Q8_0 => {
            for (ib, blk) in src.as_chunks::<34>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let base = ib * QK8_0;
                for j in 0..32 {
                    dst[base + j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        DType::Q8_1 => {
            for (ib, blk) in src.as_chunks::<36>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let base = ib * QK8_1;
                for j in 0..32 {
                    dst[base + j] = (blk[4 + j] as i8) as f32 * d;
                }
            }
        }
        DType::Q5_0 => {
            for (ib, blk) in src.as_chunks::<22>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                let base = ib * QK4_0;
                for j in 0..16 {
                    let xh0 = (((qh >> j) & 1) << 4) as u8;
                    let xh1 = (((qh >> (j + 16)) & 1) << 4) as u8;
                    let qs = blk[6 + j];
                    dst[base + j] = ((((qs & 0x0F) | xh0) as i32) - 16) as f32 * d;
                    dst[base + j + 16] = ((((qs >> 4) | xh1) as i32) - 16) as f32 * d;
                }
            }
        }
        DType::Q3K => {
            const KMASK1: u32 = 0x0303_0303;
            const KMASK2: u32 = 0x0F0F_0F0F;
            for (ib, blk) in src.as_chunks::<110>().0.iter().enumerate() {
                let hm = &blk[0..32];
                let qs = &blk[32..96];
                let sc_raw = &blk[96..108];
                let d_all = f16_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
                let a0 = u32::from_le_bytes([sc_raw[0], sc_raw[1], sc_raw[2], sc_raw[3]]);
                let a1 = u32::from_le_bytes([sc_raw[4], sc_raw[5], sc_raw[6], sc_raw[7]]);
                let tmp = u32::from_le_bytes([sc_raw[8], sc_raw[9], sc_raw[10], sc_raw[11]]);
                let aux = [
                    (a0 & KMASK2) | ((tmp & KMASK1) << 4),
                    (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
                    ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
                    ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
                ];
                let mut scales = [0i8; 16];
                for (w, &u) in aux.iter().enumerate() {
                    let b = u.to_le_bytes();
                    for k in 0..4 {
                        scales[w * 4 + k] = b[k] as i8;
                    }
                }
                let mut out_idx = ib * QK_K;
                let mut is = 0usize;
                let mut m = 1u8;
                for half128 in 0..2 {
                    let q = &qs[half128 * 32..(half128 + 1) * 32];
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let dl0 = d_all * ((scales[is] as i32 - 32) as f32);
                        is += 1;
                        for l in 0..16 {
                            let qv = (((q[l] >> shift) & 3) as i32)
                                - if (hm[l] & m) != 0 { 0 } else { 4 };
                            dst[out_idx] = dl0 * (qv as f32);
                            out_idx += 1;
                        }
                        let dl1 = d_all * ((scales[is] as i32 - 32) as f32);
                        is += 1;
                        for l in 0..16 {
                            let qv = (((q[l + 16] >> shift) & 3) as i32)
                                - if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                            dst[out_idx] = dl1 * (qv as f32);
                            out_idx += 1;
                        }
                        shift += 2;
                        m = m.wrapping_shl(1);
                    }
                }
            }
        }
        DType::Q4K => {
            for (ib, blk) in src.as_chunks::<144>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let mut scales_12 = [0u8; 12];
                scales_12.copy_from_slice(&blk[4..16]);
                let (sc, m) = unpack_q5k_scales_mins(&scales_12);
                let qs = &blk[16..144];
                let base = ib * QK_K;
                for chunk in 0..4 {
                    let d1 = d * (sc[2 * chunk] as f32);
                    let m1 = dmin * (m[2 * chunk] as f32);
                    let d2 = d * (sc[2 * chunk + 1] as f32);
                    let m2 = dmin * (m[2 * chunk + 1] as f32);
                    let ql = &qs[chunk * 32..(chunk + 1) * 32];
                    let out = &mut dst[base + chunk * 64..base + (chunk + 1) * 64];
                    for l in 0..32 {
                        out[l] = d1 * ((ql[l] & 0x0F) as f32) - m1;
                        out[l + 32] = d2 * ((ql[l] >> 4) as f32) - m2;
                    }
                }
            }
        }
        DType::Iq4Xs => {
            for (ib, blk) in src.as_chunks::<136>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let scales_h = u16::from_le_bytes([blk[2], blk[3]]) as u32;
                let scales_l = &blk[4..8];
                let qs_all = &blk[8..136];
                let base = ib * QK_K;
                for ib32 in 0..8 {
                    let ls = (((scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0x0F) as u32)
                        | (((scales_h >> (2 * ib32)) & 3) << 4);
                    let dl = d * ((ls as i32 - 32) as f32);
                    let qs = &qs_all[ib32 * 16..(ib32 + 1) * 16];
                    let out = &mut dst[base + ib32 * 32..base + (ib32 + 1) * 32];
                    for j in 0..16 {
                        out[j] = dl * (KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32);
                        out[j + 16] = dl * (KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32);
                    }
                }
            }
        }
        DType::Q5K => {
            for (ib, blk) in src.as_chunks::<176>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let mut scales_12 = [0u8; 12];
                scales_12.copy_from_slice(&blk[4..16]);
                let (sc, m) = unpack_q5k_scales_mins(&scales_12);
                let qh = &blk[16..48];
                let qs = &blk[48..176];

                let base = ib * QK_K;
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for chunk in 0..4 {
                    let d1 = d * (sc[2 * chunk] as f32);
                    let m1 = dmin * (m[2 * chunk] as f32);
                    let d2 = d * (sc[2 * chunk + 1] as f32);
                    let m2 = dmin * (m[2 * chunk + 1] as f32);
                    let ql = &qs[chunk * 32..(chunk + 1) * 32];
                    let out = &mut dst[base + chunk * 64..base + (chunk + 1) * 64];
                    for l in 0..32 {
                        let q1 = (ql[l] & 0x0F) + if (qh[l] & u1) != 0 { 16 } else { 0 };
                        let q2 = (ql[l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
                        out[l] = d1 * (q1 as f32) - m1;
                        out[l + 32] = d2 * (q2 as f32) - m2;
                    }
                    u1 = u1.wrapping_shl(2);
                    u2 = u2.wrapping_shl(2);
                }
            }
        }
        DType::Q6K => {
            for (ib, blk) in src.as_chunks::<210>().0.iter().enumerate() {
                let ql_all = &blk[0..128];
                let qh_all = &blk[128..192];
                let sc_all = &blk[192..208];
                let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));

                let base = ib * QK_K;
                for n128 in 0..2 {
                    let ql = &ql_all[n128 * 64..(n128 + 1) * 64];
                    let qh = &qh_all[n128 * 32..(n128 + 1) * 32];
                    let sc = &sc_all[n128 * 8..(n128 + 1) * 8];
                    let out = &mut dst[base + n128 * 128..base + (n128 + 1) * 128];

                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32) - 32;
                        let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32) - 32;
                        let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32) - 32;
                        let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32) - 32;

                        out[l] = d * ((sc[is] as i8) as f32) * (q1 as f32);
                        out[l + 32] = d * ((sc[is + 2] as i8) as f32) * (q2 as f32);
                        out[l + 64] = d * ((sc[is + 4] as i8) as f32) * (q3 as f32);
                        out[l + 96] = d * ((sc[is + 6] as i8) as f32) * (q4 as f32);
                    }
                }
            }
        }
        DType::Q8K => {
            for (ib, blk) in src.as_chunks::<292>().0.iter().enumerate() {
                let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                let base = ib * QK_K;
                for j in 0..QK_K {
                    dst[base + j] = (blk[4 + j] as i8) as f32 * d;
                }
            }
        }
        DType::Q2_0 => {
            for (ib, blk) in src.as_chunks::<18>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let base = ib * QK2_0;
                for b in 0..16 {
                    let byte = blk[2 + b];
                    dst[base + b * 4] = (((byte & 3) as i32) - 1) as f32 * d;
                    dst[base + b * 4 + 1] = ((((byte >> 2) & 3) as i32) - 1) as f32 * d;
                    dst[base + b * 4 + 2] = ((((byte >> 4) & 3) as i32) - 1) as f32 * d;
                    dst[base + b * 4 + 3] = ((((byte >> 6) & 3) as i32) - 1) as f32 * d;
                }
            }
        }
        DType::TQ2_0 => {
            for (ib, blk) in src.as_chunks::<66>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[64], blk[65]]));
                let base = ib * QK_K;
                for half in 0..2 {
                    let qs = &blk[half * 32..(half + 1) * 32];
                    let out = &mut dst[base + half * 128..base + (half + 1) * 128];
                    for l in 0..4 {
                        let sh = l * 2;
                        for (m, &qb) in qs.iter().enumerate() {
                            let q = (((qb >> sh) & 3) as i32) - 1;
                            out[l * 32 + m] = (q as f32) * d;
                        }
                    }
                }
            }
        }
        DType::Iq2Xxs => {
            for (ib, blk) in src.as_chunks::<66>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qs_bytes = &blk[2..66];
                let base = ib * QK_K;
                for ib32 in 0..8 {
                    let sb = &qs_bytes[ib32 * 8..(ib32 + 1) * 8];
                    let aux1 = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);
                    let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                    for l in 0..4 {
                        let grid_bytes = IQ2XXS_GRID[sb[l] as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        let out =
                            &mut dst[base + ib32 * 32 + l * 8..base + ib32 * 32 + (l + 1) * 8];
                        for j in 0..8 {
                            let s = if (signs & (1u8 << j)) != 0 {
                                -1.0f32
                            } else {
                                1.0f32
                            };
                            out[j] = db * (grid_bytes[j] as f32) * s;
                        }
                    }
                }
            }
        }
        DType::Iq3Xxs => {
            for (ib, blk) in src.as_chunks::<98>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let q3 = &blk[2..66];
                let gas = &blk[66..98];
                let base = ib * QK_K;
                for ib32 in 0..8 {
                    let aux32 = u32::from_le_bytes([
                        gas[ib32 * 4],
                        gas[ib32 * 4 + 1],
                        gas[ib32 * 4 + 2],
                        gas[ib32 * 4 + 3],
                    ]);
                    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                    let q3_sub = &q3[ib32 * 8..(ib32 + 1) * 8];
                    for l in 0..4 {
                        let g1 = IQ3XXS_GRID[q3_sub[2 * l] as usize].to_le_bytes();
                        let g2 = IQ3XXS_GRID[q3_sub[2 * l + 1] as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        let out =
                            &mut dst[base + ib32 * 32 + l * 8..base + ib32 * 32 + (l + 1) * 8];
                        for j in 0..4 {
                            let s0 = if (signs & (1u8 << j)) != 0 {
                                -1.0f32
                            } else {
                                1.0f32
                            };
                            let s1 = if (signs & (1u8 << (j + 4))) != 0 {
                                -1.0f32
                            } else {
                                1.0f32
                            };
                            out[j] = db * (g1[j] as f32) * s0;
                            out[j + 4] = db * (g2[j] as f32) * s1;
                        }
                    }
                }
            }
        }
        DType::Iq4Nl => {
            for (ib, blk) in src.as_chunks::<18>().0.iter().enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let base = ib * QK4_NL;
                for j in 0..16 {
                    let b = blk[2 + j];
                    dst[base + j] = d * (KVALUES_IQ4NL[(b & 0x0F) as usize] as f32);
                    dst[base + j + 16] = d * (KVALUES_IQ4NL[(b >> 4) as usize] as f32);
                }
            }
        }
        DType::NvFp4 => {
            for (ib, blk) in src.as_chunks::<36>().0.iter().enumerate() {
                let base = ib * QK_NVFP4;
                for s in 0..4 {
                    let d = UE4M3_TO_F32_LUT[blk[s] as usize];
                    let sub_base = base + s * QK_NVFP4_SUB;
                    for j in 0..(QK_NVFP4_SUB / 2) {
                        let b = blk[4 + s * (QK_NVFP4_SUB / 2) + j];
                        dst[sub_base + j] = d * (KVALUES_FP4[(b & 0x0F) as usize] as f32);
                        dst[sub_base + j + QK_NVFP4_SUB / 2] =
                            d * (KVALUES_FP4[(b >> 4) as usize] as f32);
                    }
                }
            }
        }
    }
}

/// Quantizes `src` (`f32`, multiple of 64) into `BlockQ2_0` (`w = (code - 1) * d`, `code in 0..=3`).
pub fn quantize_row_q2_0(src: &[f32], dst: &mut [BlockQ2_0]) {
    assert_eq!(src.len() % QK2_0, 0);
    assert_eq!(dst.len(), src.len() / QK2_0);
    for (ib, blk) in dst.iter_mut().enumerate() {
        let s = &src[ib * QK2_0..(ib + 1) * QK2_0];
        let mut min_v = 0.0f32;
        let mut max_v = 0.0f32;
        for &v in s {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        // Codes represent `{-1, 0, +1, +2} * d`. Pick `d >= 0` minimizing clipping on both ends,
        // or negate `d` when the negative tail dominates (`{-2, -1, 0, +1} * |d|`).
        let d_pos = (max_v * 0.5).max(-min_v);
        let d_neg = (-min_v * 0.5).max(max_v);
        let d_init = if d_neg < d_pos { -d_neg } else { d_pos };
        let d_f16 = f32_to_f16(d_init);
        let d = f16_to_f32(d_f16);
        let id = if d.abs() > 1e-30 { 1.0 / d } else { 0.0 };

        // 1-step least-squares scale refinement (`d* = sum(q_i * x_i) / sum(q_i^2)`)
        let mut codes = [1u8; QK2_0];
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (i, &v) in s.iter().enumerate() {
            let q = (v * id).round().clamp(-1.0, 2.0) as i32;
            codes[i] = (q + 1) as u8;
            num += (q as f32) * v;
            den += (q * q) as f32;
        }
        let d_ref = if den > 0.0 { num / den } else { d_init };
        let d_final_f16 = f32_to_f16(d_ref);
        let d_final = f16_to_f32(d_final_f16);
        let id_final = if d_final.abs() > 1e-30 {
            1.0 / d_final
        } else {
            0.0
        };
        blk.d = d_final_f16;
        for b in 0..16 {
            let mut packed = 0u8;
            for c in 0..4 {
                let q = (s[b * 4 + c] * id_final).round().clamp(-1.0, 2.0) as i32;
                packed |= ((q + 1) as u8) << (c * 2);
            }
            blk.qs[b] = packed;
        }
    }
}

/// Quantizes `src` (`f32`, multiple of 256) into BitNet b1.58 ternary `BlockTQ2_0` (`{-d, 0, +d}`).
pub fn quantize_row_tq2_0(src: &[f32], dst: &mut [BlockTQ2_0]) {
    assert_eq!(src.len() % QK_K, 0);
    assert_eq!(dst.len(), src.len() / QK_K);
    for (ib, blk) in dst.iter_mut().enumerate() {
        let s = &src[ib * QK_K..(ib + 1) * QK_K];
        let mut amax = 0.0f32;
        for &v in s {
            amax = amax.max(v.abs());
        }
        if amax <= 1e-30 {
            blk.d = 0;
            blk.qs = [0x55u8; QK_K / 4]; // `code = 1` -> `0` everywhere
            continue;
        }
        // Search optimal L2 ternary threshold `tau in [0.25..0.65] * amax` (`d* = sum_{|x|>=tau} |x| / N_nz`)
        let mut best_err = f32::MAX;
        let mut best_d = amax * 0.7;
        for step in 5..=14 {
            let tau = amax * (step as f32 * 0.05);
            let mut sum_abs = 0.0f32;
            let mut cnt = 0usize;
            for &v in s {
                if v.abs() >= tau {
                    sum_abs += v.abs();
                    cnt += 1;
                }
            }
            if cnt == 0 {
                continue;
            }
            let d_cand = f16_to_f32(f32_to_f16(sum_abs / (cnt as f32)));
            if d_cand <= 0.0 {
                continue;
            }
            let half_d = 0.5 * d_cand;
            let mut err = 0.0f32;
            for &v in s {
                let q = if v > half_d {
                    d_cand
                } else if v < -half_d {
                    -d_cand
                } else {
                    0.0
                };
                let diff = v - q;
                err += diff * diff;
            }
            if err < best_err {
                best_err = err;
                best_d = d_cand;
            }
        }
        let d_f16 = f32_to_f16(best_d);
        let d = f16_to_f32(d_f16);
        let thresh = 0.5 * d;
        blk.d = d_f16;
        for half in 0..2 {
            let sub = &s[half * 128..(half + 1) * 128];
            for m in 0..32 {
                let mut packed = 0u8;
                for l in 0..4 {
                    let v = sub[l * 32 + m];
                    let code: u8 = if v > thresh {
                        2
                    } else if v < -thresh {
                        0
                    } else {
                        1
                    };
                    packed |= code << (l * 2);
                }
                blk.qs[half * 32 + m] = packed;
            }
        }
    }
}

/// Computes the valid 7-bit `KSIGNS_IQ2XS` sign index (even popcount over 8 lanes) and positive magnitudes.
#[inline]
fn pack_even_signs_8(group: &[f32; 8]) -> (u8, [f32; 8]) {
    let mut mask = 0u8;
    let mut abs_v = [0.0f32; 8];
    let mut min_idx = 0usize;
    let mut min_abs = f32::MAX;
    for j in 0..8 {
        let v = group[j];
        let av = v.abs();
        abs_v[j] = av;
        if v < 0.0 {
            mask |= 1u8 << j;
        }
        if av < min_abs {
            min_abs = av;
            min_idx = j;
        }
    }
    if !mask.count_ones().is_multiple_of(2) {
        mask ^= 1u8 << min_idx;
        // Flipped lane opposes original sign, so its effective projection onto `+grid[j]` is negative!
        abs_v[min_idx] = -abs_v[min_idx];
    }
    (mask & 0x7F, abs_v)
}

/// Quantizes `src` (`f32`, multiple of 256) into `BlockIQ2XXS` (`ggml_type = 16`, 2.0625 bpw).
pub fn quantize_row_iq2_xxs(src: &[f32], dst: &mut [BlockIQ2XXS]) {
    assert_eq!(src.len() % QK_K, 0);
    assert_eq!(dst.len(), src.len() / QK_K);
    for (ib, blk) in dst.iter_mut().enumerate() {
        let s = &src[ib * QK_K..(ib + 1) * QK_K];
        // Max decoded value in `IQ2_XXS` is `d * (31 * 0.125) * 43 = 166.625 * d`
        let mut amax = 0.0f32;
        for &v in s {
            amax = amax.max(v.abs());
        }
        if amax <= 1e-30 {
            blk.d = 0;
            blk.qs = [0u16; QK_K / 8];
            continue;
        }
        let d_f16 = f32_to_f16(amax / 150.0);
        let d = f16_to_f32(d_f16);
        blk.d = d_f16;

        for ib32 in 0..8 {
            let sub32 = &s[ib32 * 32..(ib32 + 1) * 32];
            let mut sub_amax = 0.0f32;
            for &v in sub32 {
                sub_amax = sub_amax.max(v.abs());
            }
            // Initial 4-bit `ls_idx in 0..=15` (`db = d * (2*ls_idx + 1) * 0.125`, max grid = 43)
            let ls_est = (((sub_amax / (d * 43.0 * 0.125)) - 1.0) * 0.5)
                .round()
                .clamp(0.0, 15.0) as u32;

            let mut grid_idx = [0u8; 4];
            let mut sign_idx = [0u8; 4];
            let mut sum_gx = 0.0f32;
            let mut sum_gg = 0.0f32;

            for l in 0..4 {
                let mut grp = [0.0f32; 8];
                grp.copy_from_slice(&sub32[l * 8..(l + 1) * 8]);
                let (s7, signed_abs) = pack_even_signs_8(&grp);
                sign_idx[l] = s7;

                // Pick `g in 0..256` minimizing `||signed_abs - db_est * grid||^2`
                let db_est = d * ((2 * ls_est + 1) as f32) * 0.125;
                let mut best_g = 0u8;
                let mut best_err = f32::MAX;
                let mut best_gx = 0.0f32;
                let mut best_gg = 0.0f32;
                for (g, &word) in IQ2XXS_GRID.iter().enumerate() {
                    let gb = word.to_le_bytes();
                    let mut err = 0.0f32;
                    let mut gx = 0.0f32;
                    let mut gg = 0.0f32;
                    for j in 0..8 {
                        let gj = gb[j] as f32;
                        let diff = signed_abs[j] - db_est * gj;
                        err += diff * diff;
                        gx += gj * signed_abs[j];
                        gg += gj * gj;
                    }
                    if err < best_err {
                        best_err = err;
                        best_g = g as u8;
                        best_gx = gx;
                        best_gg = gg;
                    }
                }
                grid_idx[l] = best_g;
                sum_gx += best_gx;
                sum_gg += best_gg;
            }

            // Refine 4-bit `ls_idx` by least squares (`db* = sum_gx / sum_gg = d * (2*ls + 1) * 0.125`)
            let ls_final = if sum_gg > 0.0 && d > 0.0 {
                ((((sum_gx / sum_gg) / (d * 0.125)) - 1.0) * 0.5)
                    .round()
                    .clamp(0.0, 15.0) as u32
            } else {
                ls_est
            };
            let aux0 = u32::from_le_bytes(grid_idx);
            let aux1 = (sign_idx[0] as u32)
                | ((sign_idx[1] as u32) << 7)
                | ((sign_idx[2] as u32) << 14)
                | ((sign_idx[3] as u32) << 21)
                | (ls_final << 28);
            blk.qs[ib32 * 4] = (aux0 & 0xFFFF) as u16;
            blk.qs[ib32 * 4 + 1] = (aux0 >> 16) as u16;
            blk.qs[ib32 * 4 + 2] = (aux1 & 0xFFFF) as u16;
            blk.qs[ib32 * 4 + 3] = (aux1 >> 16) as u16;
        }
    }
}

/// Quantizes `src` (`f32`, multiple of 256) into `BlockIQ3XXS` (`ggml_type = 18`, 3.0625 bpw).
pub fn quantize_row_iq3_xxs(src: &[f32], dst: &mut [BlockIQ3XXS]) {
    assert_eq!(src.len() % QK_K, 0);
    assert_eq!(dst.len(), src.len() / QK_K);
    for (ib, blk) in dst.iter_mut().enumerate() {
        let s = &src[ib * QK_K..(ib + 1) * QK_K];
        // Max decoded value in `IQ3_XXS` is `d * (31 * 0.25) * 62 = 480.5 * d`
        let mut amax = 0.0f32;
        for &v in s {
            amax = amax.max(v.abs());
        }
        if amax <= 1e-30 {
            blk.d = 0;
            blk.qs = [0u8; 3 * QK_K / 8];
            continue;
        }
        let d_f16 = f32_to_f16(amax / 450.0);
        let d = f16_to_f32(d_f16);
        blk.d = d_f16;

        for ib32 in 0..8 {
            let sub32 = &s[ib32 * 32..(ib32 + 1) * 32];
            let mut sub_amax = 0.0f32;
            for &v in sub32 {
                sub_amax = sub_amax.max(v.abs());
            }
            let ls_est = (((sub_amax / (d * 62.0 * 0.25)) - 1.0) * 0.5)
                .round()
                .clamp(0.0, 15.0) as u32;
            let db_est = d * ((2 * ls_est + 1) as f32) * 0.25;

            let mut sign_idx = [0u8; 4];
            let mut sum_gx = 0.0f32;
            let mut sum_gg = 0.0f32;

            for l in 0..4 {
                let mut grp = [0.0f32; 8];
                grp.copy_from_slice(&sub32[l * 8..(l + 1) * 8]);
                let (s7, signed_abs) = pack_even_signs_8(&grp);
                sign_idx[l] = s7;

                for half4 in 0..2 {
                    let target4 = &signed_abs[half4 * 4..(half4 + 1) * 4];
                    let mut best_g = 0u8;
                    let mut best_err = f32::MAX;
                    let mut best_gx = 0.0f32;
                    let mut best_gg = 0.0f32;
                    for (g, &word) in IQ3XXS_GRID.iter().enumerate() {
                        let gb = word.to_le_bytes();
                        let mut err = 0.0f32;
                        let mut gx = 0.0f32;
                        let mut gg = 0.0f32;
                        for j in 0..4 {
                            let gj = gb[j] as f32;
                            let diff = target4[j] - db_est * gj;
                            err += diff * diff;
                            gx += gj * target4[j];
                            gg += gj * gj;
                        }
                        if err < best_err {
                            best_err = err;
                            best_g = g as u8;
                            best_gx = gx;
                            best_gg = gg;
                        }
                    }
                    blk.qs[ib32 * 8 + 2 * l + half4] = best_g;
                    sum_gx += best_gx;
                    sum_gg += best_gg;
                }
            }

            let ls_final = if sum_gg > 0.0 && d > 0.0 {
                ((((sum_gx / sum_gg) / (d * 0.25)) - 1.0) * 0.5)
                    .round()
                    .clamp(0.0, 15.0) as u32
            } else {
                ls_est
            };
            let aux32 = (sign_idx[0] as u32)
                | ((sign_idx[1] as u32) << 7)
                | ((sign_idx[2] as u32) << 14)
                | ((sign_idx[3] as u32) << 21)
                | (ls_final << 28);
            let gas_off = 64 + ib32 * 4;
            blk.qs[gas_off..gas_off + 4].copy_from_slice(&aux32.to_le_bytes());
        }
    }
}

/// Quantizes `src` (`f32`, multiple of 32) into non-linear 4-bit `BlockIQ4NL` (`ggml_type = 20`).
pub fn quantize_row_iq4_nl(src: &[f32], dst: &mut [BlockIQ4NL]) {
    assert_eq!(src.len() % QK4_NL, 0);
    assert_eq!(dst.len(), src.len() / QK4_NL);
    for (ib, blk) in dst.iter_mut().enumerate() {
        let s = &src[ib * QK4_NL..(ib + 1) * QK4_NL];
        let mut max_v = 0.0f32;
        let mut amax = 0.0f32;
        for &v in s {
            let av = v.abs();
            if av > amax {
                amax = av;
                max_v = v;
            }
        }
        let d_init = max_v / -127.0;
        let d_f16 = f32_to_f16(d_init);
        let d = f16_to_f32(d_f16);
        blk.d = d_f16;
        if d.abs() <= 1e-30 {
            blk.qs = [0x88u8; QK4_NL / 2];
            continue;
        }
        let id = 1.0 / d;
        let nearest = |val: f32| -> u8 {
            let scaled = val * id;
            let mut best = 0u8;
            let mut best_d = f32::MAX;
            for (idx, &kv) in KVALUES_IQ4NL.iter().enumerate() {
                let df = (scaled - kv as f32).abs();
                if df < best_d {
                    best_d = df;
                    best = idx as u8;
                }
            }
            best
        };
        for j in 0..16 {
            let lo = nearest(s[j]);
            let hi = nearest(s[j + 16]);
            blk.qs[j] = lo | (hi << 4);
        }
    }
}

/// Quantizes `src` (`f32`, multiple of 64) into `BlockNvFp4` (`ggml_type = 40`).
pub fn quantize_row_nvfp4(src: &[f32], dst: &mut [BlockNvFp4]) {
    assert_eq!(src.len() % QK_NVFP4, 0);
    assert_eq!(dst.len(), src.len() / QK_NVFP4);
    for (ib, blk) in dst.iter_mut().enumerate() {
        for s in 0..4 {
            let xb = &src[ib * QK_NVFP4 + s * QK_NVFP4_SUB..ib * QK_NVFP4 + (s + 1) * QK_NVFP4_SUB];
            let mut amax = 0.0f32;
            for &v in xb {
                let av = v.abs();
                if av > amax {
                    amax = av;
                }
            }
            let ue = f32_to_ue4m3(amax / 6.0);
            blk.d[s] = ue;
            let d = UE4M3_TO_F32_LUT[ue as usize];
            let nearest = |val: f32| -> u8 {
                let mut best = 0u8;
                let mut best_err = ((KVALUES_FP4[0] as f32) * d - val).abs();
                for (idx, &kv) in KVALUES_FP4.iter().enumerate().skip(1) {
                    let err = ((kv as f32) * d - val).abs();
                    if err < best_err {
                        best_err = err;
                        best = idx as u8;
                    }
                }
                best
            };
            for j in 0..(QK_NVFP4_SUB / 2) {
                let x0 = nearest(xb[j]);
                let x1 = nearest(xb[QK_NVFP4_SUB / 2 + j]);
                blk.qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

/// Quantizes `src` (`f32` row of length `src.len()`) into raw little-endian bytes of `target_dtype`.
pub fn quantize_row_to_bytes(target_dtype: DType, src: &[f32], dst: &mut [u8]) {
    let n = src.len();
    assert_eq!(n % target_dtype.block_elements(), 0);
    assert_eq!(dst.len(), target_dtype.row_bytes(n));
    match target_dtype {
        DType::Q2_0 => {
            let nb = n / QK2_0;
            let mut blks = vec![
                BlockQ2_0 {
                    d: 0,
                    qs: [0; QK2_0 / 4]
                };
                nb
            ];
            quantize_row_q2_0(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 18..i * 18 + 2].copy_from_slice(&b.d.to_le_bytes());
                dst[i * 18 + 2..i * 18 + 18].copy_from_slice(&b.qs);
            }
        }
        DType::TQ2_0 => {
            let nb = n / QK_K;
            let mut blks = vec![
                BlockTQ2_0 {
                    qs: [0; QK_K / 4],
                    d: 0
                };
                nb
            ];
            quantize_row_tq2_0(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 66..i * 66 + 64].copy_from_slice(&b.qs);
                dst[i * 66 + 64..i * 66 + 66].copy_from_slice(&b.d.to_le_bytes());
            }
        }
        DType::Iq2Xxs => {
            let nb = n / QK_K;
            let mut blks = vec![
                BlockIQ2XXS {
                    d: 0,
                    qs: [0; QK_K / 8]
                };
                nb
            ];
            quantize_row_iq2_xxs(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 66..i * 66 + 2].copy_from_slice(&b.d.to_le_bytes());
                for (w, &u) in b.qs.iter().enumerate() {
                    dst[i * 66 + 2 + w * 2..i * 66 + 4 + w * 2].copy_from_slice(&u.to_le_bytes());
                }
            }
        }
        DType::Iq3Xxs => {
            let nb = n / QK_K;
            let mut blks = vec![
                BlockIQ3XXS {
                    d: 0,
                    qs: [0; 3 * QK_K / 8]
                };
                nb
            ];
            quantize_row_iq3_xxs(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 98..i * 98 + 2].copy_from_slice(&b.d.to_le_bytes());
                dst[i * 98 + 2..i * 98 + 98].copy_from_slice(&b.qs);
            }
        }
        DType::Iq4Nl => {
            let nb = n / QK4_NL;
            let mut blks = vec![
                BlockIQ4NL {
                    d: 0,
                    qs: [0; QK4_NL / 2]
                };
                nb
            ];
            quantize_row_iq4_nl(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 18..i * 18 + 2].copy_from_slice(&b.d.to_le_bytes());
                dst[i * 18 + 2..i * 18 + 18].copy_from_slice(&b.qs);
            }
        }
        DType::NvFp4 => {
            let nb = n / QK_NVFP4;
            let mut blks = vec![
                BlockNvFp4 {
                    d: [0; 4],
                    qs: [0; QK_NVFP4 / 2]
                };
                nb
            ];
            quantize_row_nvfp4(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 36..i * 36 + 4].copy_from_slice(&b.d);
                dst[i * 36 + 4..i * 36 + 36].copy_from_slice(&b.qs);
            }
        }
        DType::Q4_0 => {
            let nb = n / QK4_0;
            let mut blks = vec![BlockQ4_0 { d: 0, qs: [0; 16] }; nb];
            quantize_row_q4_0(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 18..i * 18 + 2].copy_from_slice(&b.d.to_le_bytes());
                dst[i * 18 + 2..i * 18 + 18].copy_from_slice(&b.qs);
            }
        }
        DType::Q8_0 => {
            let nb = n / QK8_0;
            let mut blks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; nb];
            quantize_row_q8_0(src, &mut blks);
            for (i, b) in blks.iter().enumerate() {
                dst[i * 34..i * 34 + 2].copy_from_slice(&b.d.to_le_bytes());
                for (j, &q) in b.qs.iter().enumerate() {
                    dst[i * 34 + 2 + j] = q as u8;
                }
            }
        }
        DType::F16 => {
            for (i, &v) in src.iter().enumerate() {
                dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        }
        DType::F32 => {
            for (i, &v) in src.iter().enumerate() {
                dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        other => panic!("quantize_row_to_bytes: unsupported target dtype {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_quant_dequant_smoke() {
        let mut src = [0.0f32; 32];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.1;
        }
        let mut blk = [BlockQ4_0 { d: 0, qs: [0; 16] }];
        quantize_row_q4_0(&src, &mut blk);
        let mut out = [0.0f32; 32];
        dequantize_row_q4_0(&blk, &mut out);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.25, "a={a}, b={b}");
        }
    }

    #[test]
    fn q4_0x4_repack_and_dequant_bit_exact() {
        let rows = 8usize;
        let cols = 64usize;
        let nb = cols / QK4_0;
        let mut raw_blocks = Vec::with_capacity(rows * nb);
        for r in 0..rows {
            for ib in 0..nb {
                let mut qs = [0u8; 16];
                for (j, q) in qs.iter_mut().enumerate() {
                    let lo = ((r * 3 + ib * 5 + j) & 0x0F) as u8;
                    let hi = ((15 - ((r + ib * 7 + j * 3) & 0x0F)) & 0x0F) as u8;
                    *q = lo | (hi << 4);
                }
                raw_blocks.push(BlockQ4_0 {
                    d: f32_to_f16((r as f32 - 3.5) * 0.0625 + (ib as f32) * 0.03125),
                    qs,
                });
            }
        }

        let mut raw_bytes = Vec::with_capacity(rows * nb * 18);
        for b in &raw_blocks {
            raw_bytes.extend_from_slice(&b.d.to_le_bytes());
            raw_bytes.extend_from_slice(&b.qs);
        }

        let mut repacked_bytes = vec![0u8; raw_bytes.len()];
        repack_q4_0_bytes_to_q4_0x4(&raw_bytes, &mut repacked_bytes, rows, cols);

        let mut repacked_blocks = vec![
            BlockQ4_0x4 {
                d: [0; 4],
                qs: [0; 64]
            };
            (rows / 4) * nb
        ];
        repack_q4_0_to_q4_0x4(&raw_blocks, &mut repacked_blocks, rows, nb);

        for (i, sb) in repacked_blocks.iter().enumerate() {
            let chunk = &repacked_bytes[i * 72..(i + 1) * 72];
            for m in 0..4 {
                assert_eq!(
                    sb.d[m],
                    u16::from_le_bytes([chunk[m * 2], chunk[m * 2 + 1]])
                );
            }
            assert_eq!(&sb.qs[..], &chunk[8..72]);
        }

        for r in 0..rows {
            let mut ref_row = vec![0.0f32; cols];
            dequantize_row_q4_0(&raw_blocks[r * nb..(r + 1) * nb], &mut ref_row);
            let mut got_row = vec![0.0f32; cols];
            dequantize_q4_0x4_single_row(&repacked_bytes, cols, r, &mut got_row);
            assert_eq!(
                got_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {r} dequant mismatch"
            );
        }
    }

    #[test]
    fn m4_lowbit_quant_dequant_roundtrip() {
        let n = 256usize;
        let src: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.37).sin() * 1.4) + ((i as f32 * 0.11).cos() * 0.4))
            .collect();

        for (dt, min_cos) in [
            (DType::Q2_0, 0.88f32),
            (DType::TQ2_0, 0.84f32),
            (DType::Iq2Xxs, 0.90f32),
            (DType::Iq3Xxs, 0.96f32),
            (DType::Iq4Nl, 0.99f32),
            (DType::NvFp4, 0.985f32),
        ] {
            let mut raw = vec![0u8; dt.row_bytes(n)];
            quantize_row_to_bytes(dt, &src, &mut raw);
            let mut dec = vec![0.0f32; n];
            dequantize_row_bytes(dt, &raw, &mut dec);
            let dot: f32 = src.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let na: f32 = src.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nb: f32 = dec.iter().map(|b| b * b).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(
                cos >= min_cos,
                "{dt:?} cosine similarity {cos:.4} < {min_cos}"
            );
        }
    }
}
