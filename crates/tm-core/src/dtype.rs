//! Quantization block layouts and data types.
//!
//! Block layouts (`BlockQ4_0`, `BlockQ8_0`) match ggml's binary GGUF format:
//! Copyright (c) 2023-2026 The ggml authors (MIT License).
//! Upstream reference: llama.cpp `ggml/src/ggml-common.h` (commit `3cf03257` / `b001d7b`).

use crate::fp16::{f16_to_f32, f32_to_f16};

/// Number of weights per `BlockQ4_0`.
pub const QK4_0: usize = 32;
/// Number of weights per `BlockQ8_0`.
pub const QK8_0: usize = 32;

/// Supported tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
}

impl DType {
    /// Block size in elements.
    pub const fn block_elements(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q4_0 => QK4_0,
            DType::Q8_0 => QK8_0,
        }
    }

    /// Size of one block in bytes.
    pub const fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4_0 => core::mem::size_of::<BlockQ4_0>(),
            DType::Q8_0 => core::mem::size_of::<BlockQ8_0>(),
        }
    }
}

/// 4-bit symmetric quantization block (32 weights in 18 bytes).
///
/// Layout matches `block_q4_0` in ggml:
/// - `d`: IEEE 754 binary16 scale (little-endian `u16`).
/// - `qs[j]` low nibble (`& 0x0F`) stores weight `j` (0..16), offset by +8.
/// - `qs[j]` high nibble (`>> 4`) stores weight `j + 16` (16..32), offset by +8.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; QK4_0 / 2],
}

/// 8-bit symmetric quantization block (32 weights in 34 bytes).
///
/// Layout matches `block_q8_0` in ggml:
/// - `d`: IEEE 754 binary16 scale (little-endian `u16`).
/// - `qs[0..32]`: signed 8-bit quantized values.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(core::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(core::mem::size_of::<BlockQ8_0>() == 34);

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
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            block.qs[j] = q;
        }
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
