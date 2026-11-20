//! Quantization block layouts and data types.
//!
//! Block layouts (`BlockQ4_0`, `BlockQ4_1`, `BlockQ8_0`, `BlockQ8_1`, `BlockQ5K`, `BlockQ6K`, `BlockQ8K`)
//! match ggml's binary GGUF format:
//! Copyright (c) 2023-2026 The ggml authors (MIT License).
//! Upstream reference: llama.cpp `ggml/src/ggml-common.h` (commit `3cf03257` / `b001d7b`).

use crate::fp16::{f16_to_f32, f32_to_f16};

/// Number of weights per `BlockQ4_0` / `BlockQ4_1`.
pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
/// Number of weights per `BlockQ8_0` / `BlockQ8_1`.
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;
/// Number of weights per K-quant super-block (`BlockQ5K`, `BlockQ6K`, `BlockQ8K`).
pub const QK_K: usize = 256;

/// Supported tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q8_0,
    Q8_1,
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
            8 => Some(DType::Q8_0),
            9 => Some(DType::Q8_1),
            13 => Some(DType::Q5K),
            14 => Some(DType::Q6K),
            15 => Some(DType::Q8K),
            30 => Some(DType::BF16),
            _ => None,
        }
    }

    /// Block size in elements.
    pub const fn block_elements(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q4_0 => QK4_0,
            DType::Q4_1 => QK4_1,
            DType::Q8_0 => QK8_0,
            DType::Q8_1 => QK8_1,
            DType::Q5K | DType::Q6K | DType::Q8K => QK_K,
        }
    }

    /// Size of one block in bytes.
    pub const fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4_0 => core::mem::size_of::<BlockQ4_0>(),
            DType::Q4_1 => core::mem::size_of::<BlockQ4_1>(),
            DType::Q8_0 => core::mem::size_of::<BlockQ8_0>(),
            DType::Q8_1 => core::mem::size_of::<BlockQ8_1>(),
            DType::Q5K => core::mem::size_of::<BlockQ5K>(),
            DType::Q6K => core::mem::size_of::<BlockQ6K>(),
            DType::Q8K => core::mem::size_of::<BlockQ8K>(),
        }
    }

    /// Byte size of `n_elements` stored in this `DType` (requires `n_elements % block_elements == 0`).
    pub const fn row_bytes(self, n_elements: usize) -> usize {
        (n_elements / self.block_elements()) * self.block_bytes()
    }
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
const _: () = assert!(core::mem::size_of::<BlockQ4_1>() == 20);
const _: () = assert!(core::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(core::mem::size_of::<BlockQ8_1>() == 36);
const _: () = assert!(core::mem::size_of::<BlockQ5K>() == 176);
const _: () = assert!(core::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(core::mem::size_of::<BlockQ8K>() == 292);

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
}
