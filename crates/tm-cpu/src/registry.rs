//! Runtime kernel registry and ISA dispatch table.
//!
//! Design inspired by Strata `src/quant/registry.hpp` and `dispatch.cpp`
//! (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors).

use tm_core::{
    BlockIQ2XXS, BlockIQ3XXS, BlockIQ4NL, BlockQ2_0, BlockQ4_0, BlockQ4_0x4, BlockQ4_1, BlockQ5K,
    BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, BlockTQ2_0,
};

use crate::generic;
use crate::isa::{Isa, detect_caps, effective_isa, isa_supported};

/// Function-pointer types for CPU compute kernels.
pub type VecDotQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0]) -> f32;
pub type GemvQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0], &mut [f32], usize, usize, usize);
pub type GemvQ4_0x4Fn = fn(&[BlockQ4_0x4], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmQ4_0x4Fn = fn(&[BlockQ4_0x4], &[BlockQ8_0], &mut [f32], usize, usize, usize);
pub type VecDotQ4_1Fn = fn(&[BlockQ4_1], &[BlockQ8_1]) -> f32;
pub type GemvQ4_1Fn = fn(&[BlockQ4_1], &[BlockQ8_1], &mut [f32], usize, usize);
pub type VecDotQ8_0Fn = fn(&[BlockQ8_0], &[BlockQ8_0]) -> f32;
pub type GemvQ8_0Fn = fn(&[BlockQ8_0], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmQ8_0Fn = fn(&[BlockQ8_0], &[BlockQ8_0], &mut [f32], usize, usize, usize);
pub type VecDotQ5KFn = fn(&[BlockQ5K], &[BlockQ8K]) -> f32;
pub type GemvQ5KFn = fn(&[BlockQ5K], &[BlockQ8K], &mut [f32], usize, usize);
pub type VecDotQ6KFn = fn(&[BlockQ6K], &[BlockQ8K]) -> f32;
pub type GemvQ6KFn = fn(&[BlockQ6K], &[BlockQ8K], &mut [f32], usize, usize);
pub type GemmQ6KFn = fn(&[BlockQ6K], &[BlockQ8K], &mut [f32], usize, usize, usize);
pub type VecDotQ2_0Fn = fn(&[BlockQ2_0], &[BlockQ8_0]) -> f32;
pub type GemvQ2_0Fn = fn(&[BlockQ2_0], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmQ2_0Fn = fn(&[BlockQ2_0], &[BlockQ8_0], &mut [f32], usize, usize, usize);
pub type VecDotTQ2_0Fn = fn(&[BlockTQ2_0], &[BlockQ8K]) -> f32;
pub type GemvTQ2_0Fn = fn(&[BlockTQ2_0], &[BlockQ8K], &mut [f32], usize, usize);
pub type GemmTQ2_0Fn = fn(&[BlockTQ2_0], &[BlockQ8K], &mut [f32], usize, usize, usize);
pub type VecDotIQ2XXSFn = fn(&[BlockIQ2XXS], &[BlockQ8K]) -> f32;
pub type GemvIQ2XXSFn = fn(&[BlockIQ2XXS], &[BlockQ8K], &mut [f32], usize, usize);
pub type GemmIQ2XXSFn = fn(&[BlockIQ2XXS], &[BlockQ8K], &mut [f32], usize, usize, usize);
pub type VecDotIQ3XXSFn = fn(&[BlockIQ3XXS], &[BlockQ8K]) -> f32;
pub type GemvIQ3XXSFn = fn(&[BlockIQ3XXS], &[BlockQ8K], &mut [f32], usize, usize);
pub type GemmIQ3XXSFn = fn(&[BlockIQ3XXS], &[BlockQ8K], &mut [f32], usize, usize, usize);
pub type VecDotIQ4NLFn = fn(&[BlockIQ4NL], &[BlockQ8_0]) -> f32;
pub type GemvIQ4NLFn = fn(&[BlockIQ4NL], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmIQ4NLFn = fn(&[BlockIQ4NL], &[BlockQ8_0], &mut [f32], usize, usize, usize);
pub type DotF32F16Fn = fn(&[f32], &[u16]) -> f32;
pub type DotF32Bf16Fn = fn(&[f32], &[u16]) -> f32;
pub type DotF32Q8_0Fn = fn(&[f32], &[BlockQ8_0]) -> f32;
pub type GemvF32Q8_0Fn = fn(&[BlockQ8_0], &[f32], &mut [f32], usize, usize);
pub type GemmF32Q8_0Fn = fn(&[BlockQ8_0], &[f32], &mut [f32], usize, usize, usize);
pub type AxpyF32Fn = fn(&mut [f32], f32, &[f32]);
pub type AxpyF32F16Fn = fn(&mut [f32], f32, &[u16]);
pub type AxpyF32Q8_0Fn = fn(&mut [f32], f32, &[BlockQ8_0]);
pub type GemmQ4_0TransposedTileFn =
    fn(&[BlockQ4_0], usize, usize, &[f32], &mut [f32], usize, usize);
pub type F32ToF16SliceFn = fn(&[f32], &mut [u16]);
pub type QuantQ8_0Fn = fn(&[f32], &mut [BlockQ8_0]);
pub type QuantQ8_1Fn = fn(&[f32], &mut [BlockQ8_1]);
pub type QuantQ8KFn = fn(&[f32], &mut [BlockQ8K]);
pub type RmsNormF32Fn = fn(&mut [f32], &[f32], &[f32], f32);
pub type L2NormF32Fn = fn(&mut [f32], f32);
pub type GdnRecurrenceHeadFn =
    fn(&[f32], &[f32], &[f32], f32, f32, f32, &mut [f32], &mut [f32], usize);
pub type Mamba2ScanHeadFn =
    fn(&[f32], &[f32], &[f32], f32, f32, f32, &mut [f32], &mut [f32], usize, usize);

/// Function-pointer table for CPU compute kernels at a given ISA tier.
#[derive(Clone, Copy)]
pub struct KernelTable {
    pub isa: Isa,
    pub vec_dot_q4_0_q8_0: VecDotQ4_0Fn,
    pub gemv_q4_0_q8_0: GemvQ4_0Fn,
    pub gemm_q4_0_q8_0: GemmQ4_0Fn,
    pub gemv_q4_0x4_q8_0: GemvQ4_0x4Fn,
    pub gemm_q4_0x4_q8_0: GemmQ4_0x4Fn,
    pub vec_dot_q4_1_q8_1: VecDotQ4_1Fn,
    pub gemv_q4_1_q8_1: GemvQ4_1Fn,
    pub vec_dot_q8_0_q8_0: VecDotQ8_0Fn,
    pub gemv_q8_0_q8_0: GemvQ8_0Fn,
    pub gemm_q8_0_q8_0: GemmQ8_0Fn,
    pub vec_dot_q5_k_q8_k: VecDotQ5KFn,
    pub gemv_q5_k_q8_k: GemvQ5KFn,
    pub vec_dot_q6_k_q8_k: VecDotQ6KFn,
    pub gemv_q6_k_q8_k: GemvQ6KFn,
    pub gemm_q6_k_q8_k: GemmQ6KFn,
    pub vec_dot_q2_0_q8_0: VecDotQ2_0Fn,
    pub gemv_q2_0_q8_0: GemvQ2_0Fn,
    pub gemm_q2_0_q8_0: GemmQ2_0Fn,
    pub vec_dot_tq2_0_q8_k: VecDotTQ2_0Fn,
    pub gemv_tq2_0_q8_k: GemvTQ2_0Fn,
    pub gemm_tq2_0_q8_k: GemmTQ2_0Fn,
    pub vec_dot_iq2_xxs_q8_k: VecDotIQ2XXSFn,
    pub gemv_iq2_xxs_q8_k: GemvIQ2XXSFn,
    pub gemm_iq2_xxs_q8_k: GemmIQ2XXSFn,
    pub vec_dot_iq3_xxs_q8_k: VecDotIQ3XXSFn,
    pub gemv_iq3_xxs_q8_k: GemvIQ3XXSFn,
    pub gemm_iq3_xxs_q8_k: GemmIQ3XXSFn,
    pub vec_dot_iq4_nl_q8_0: VecDotIQ4NLFn,
    pub gemv_iq4_nl_q8_0: GemvIQ4NLFn,
    pub gemm_iq4_nl_q8_0: GemmIQ4NLFn,
    pub dot_f32_f16: DotF32F16Fn,
    pub dot_f32_bf16: DotF32Bf16Fn,
    pub dot_f32_q8_0: DotF32Q8_0Fn,
    pub gemv_f32_q8_0: GemvF32Q8_0Fn,
    pub gemm_f32_q8_0: GemmF32Q8_0Fn,
    pub axpy_f32: AxpyF32Fn,
    pub axpy_f32_f16: AxpyF32F16Fn,
    pub axpy_f32_q8_0: AxpyF32Q8_0Fn,
    pub gemm_q4_0_transposed_tile: GemmQ4_0TransposedTileFn,
    pub f32_to_f16_slice: F32ToF16SliceFn,
    pub quantize_q8_0: QuantQ8_0Fn,
    pub quantize_q8_1: QuantQ8_1Fn,
    pub quantize_q8_k: QuantQ8KFn,
    pub rms_norm_f32: RmsNormF32Fn,
    pub l2_norm_f32: L2NormF32Fn,
    pub gdn_recurrence_head: GdnRecurrenceHeadFn,
    pub mamba2_scan_head: Mamba2ScanHeadFn,
}

const GENERIC_TABLE: KernelTable = KernelTable {
    isa: Isa::Generic,
    vec_dot_q4_0_q8_0: generic::vec_dot_q4_0_q8_0,
    gemv_q4_0_q8_0: generic::gemv_q4_0_q8_0,
    gemm_q4_0_q8_0: generic::gemm_q4_0_q8_0,
    gemv_q4_0x4_q8_0: generic::gemv_q4_0x4_q8_0,
    gemm_q4_0x4_q8_0: generic::gemm_q4_0x4_q8_0,
    vec_dot_q4_1_q8_1: generic::vec_dot_q4_1_q8_1,
    gemv_q4_1_q8_1: generic::gemv_q4_1_q8_1,
    vec_dot_q8_0_q8_0: generic::vec_dot_q8_0_q8_0,
    gemv_q8_0_q8_0: generic::gemv_q8_0_q8_0,
    gemm_q8_0_q8_0: generic::gemm_q8_0_q8_0,
    vec_dot_q5_k_q8_k: generic::vec_dot_q5_k_q8_k,
    gemv_q5_k_q8_k: generic::gemv_q5_k_q8_k,
    vec_dot_q6_k_q8_k: generic::vec_dot_q6_k_q8_k,
    gemv_q6_k_q8_k: generic::gemv_q6_k_q8_k,
    gemm_q6_k_q8_k: generic::gemm_q6_k_q8_k,
    vec_dot_q2_0_q8_0: generic::vec_dot_q2_0_q8_0,
    gemv_q2_0_q8_0: generic::gemv_q2_0_q8_0,
    gemm_q2_0_q8_0: generic::gemm_q2_0_q8_0,
    vec_dot_tq2_0_q8_k: generic::vec_dot_tq2_0_q8_k,
    gemv_tq2_0_q8_k: generic::gemv_tq2_0_q8_k,
    gemm_tq2_0_q8_k: generic::gemm_tq2_0_q8_k,
    vec_dot_iq2_xxs_q8_k: generic::vec_dot_iq2_xxs_q8_k,
    gemv_iq2_xxs_q8_k: generic::gemv_iq2_xxs_q8_k,
    gemm_iq2_xxs_q8_k: generic::gemm_iq2_xxs_q8_k,
    vec_dot_iq3_xxs_q8_k: generic::vec_dot_iq3_xxs_q8_k,
    gemv_iq3_xxs_q8_k: generic::gemv_iq3_xxs_q8_k,
    gemm_iq3_xxs_q8_k: generic::gemm_iq3_xxs_q8_k,
    vec_dot_iq4_nl_q8_0: generic::vec_dot_iq4_nl_q8_0,
    gemv_iq4_nl_q8_0: generic::gemv_iq4_nl_q8_0,
    gemm_iq4_nl_q8_0: generic::gemm_iq4_nl_q8_0,
    dot_f32_f16: generic::dot_f32_f16,
    dot_f32_bf16: generic::dot_f32_bf16,
    dot_f32_q8_0: generic::dot_f32_q8_0,
    gemv_f32_q8_0: generic::gemv_f32_q8_0,
    gemm_f32_q8_0: generic::gemm_f32_q8_0,
    axpy_f32: generic::axpy_f32,
    axpy_f32_f16: generic::axpy_f32_f16,
    axpy_f32_q8_0: generic::axpy_f32_q8_0,
    gemm_q4_0_transposed_tile: generic::gemm_q4_0_transposed_tile,
    f32_to_f16_slice: generic::f32_to_f16_slice,
    quantize_q8_0: generic::quantize_q8_0,
    quantize_q8_1: generic::quantize_q8_1,
    quantize_q8_k: generic::quantize_q8_k,
    rms_norm_f32: generic::rms_norm_f32,
    l2_norm_f32: generic::l2_norm_f32,
    gdn_recurrence_head: generic::gdn_recurrence_head,
    mamba2_scan_head: generic::mamba2_scan_head,
};

#[cfg(target_arch = "aarch64")]
mod aarch64_wrappers {
    use super::*;
    #[cfg(feature = "dotprod")]
    use crate::aarch64::dotprod;
    #[cfg(feature = "i8mm")]
    use crate::aarch64::i8mm;
    #[cfg(feature = "lut")]
    use crate::aarch64::lut;
    use crate::aarch64::neon_v80;

    fn neon_v80_vec_dot(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `NEON_V80_TABLE` is only selected when NEON is supported (always on aarch64).
        unsafe { neon_v80::vec_dot_q4_0_q8_0(x, y) }
    }

    fn neon_v80_gemv(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `NEON_V80_TABLE` is only selected when NEON is supported.
        unsafe { neon_v80::gemv_q4_0_q8_0(w, x, out, rows, cols_blocks) }
    }

    fn neon_v80_gemm(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `NEON_V80_TABLE` is only selected when NEON is supported.
        unsafe { neon_v80::gemm_q4_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::vec_dot_q4_0_q8_0(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemv_q4_0_q8_0(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemm_q4_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_q4_1(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::vec_dot_q4_1_q8_1(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::vec_dot_q8_0_q8_0(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemv_q8_0_q8_0(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_q5_k(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::vec_dot_q5_k_q8_k(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_q5_k(
        w: &[BlockQ5K],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemv_q5_k_q8_k(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_q6_k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::vec_dot_q6_k_q8_k(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_q6_k(
        w: &[BlockQ6K],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemv_q6_k_q8_k(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "i8mm")]
    fn i8mm_vec_dot(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `I8MM_TABLE` is only selected after runtime `i8mm` + `dotprod` detection.
        unsafe { i8mm::vec_dot_q4_0_q8_0(x, y) }
    }

    #[cfg(feature = "i8mm")]
    fn i8mm_gemv(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `I8MM_TABLE` is only selected after runtime `i8mm` + `dotprod` detection.
        unsafe { i8mm::gemv_q4_0_q8_0(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "i8mm")]
    fn i8mm_gemm(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `I8MM_TABLE` is only selected after runtime `i8mm` + `dotprod` detection.
        unsafe { i8mm::gemm_q4_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    fn neon_dot_f32_f16(q: &[f32], k_f16: &[u16]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::dot_f32_f16(q, k_f16) }
    }

    fn neon_axpy_f32(out: &mut [f32], w: f32, x: &[f32]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::axpy_f32(out, w, x) }
    }

    fn neon_axpy_f32_f16(out: &mut [f32], w: f32, v_f16: &[u16]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::axpy_f32_f16(out, w, v_f16) }
    }

    fn neon_gemm_q4_0_transposed_tile(
        w: &[BlockQ4_0],
        cols_blocks: usize,
        cb: usize,
        dy: &[f32],
        dx_tile: &mut [f32],
        rows: usize,
        n_tok: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemm_q4_0_transposed_tile(w, cols_blocks, cb, dy, dx_tile, rows, n_tok) }
    }

    fn neon_f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::f32_to_f16_slice(src, dst) }
    }

    fn neon_quant_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::quantize_q8_0(src, dst) }
    }

    fn neon_quant_q8_1(src: &[f32], dst: &mut [BlockQ8_1]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::quantize_q8_1(src, dst) }
    }

    fn neon_v80_vec_dot_q4_1(x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::vec_dot_q4_1_q8_1(x, y) }
    }

    fn neon_v80_gemv_q4_1(
        w: &[BlockQ4_1],
        x: &[BlockQ8_1],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemv_q4_1_q8_1(w, x, out, rows, cols_blocks) }
    }

    fn neon_v80_vec_dot_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::vec_dot_q8_0_q8_0(x, y) }
    }

    fn neon_v80_gemv_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemv_q8_0_q8_0(w, x, out, rows, cols_blocks) }
    }

    fn neon_v80_vec_dot_q5_k(x: &[BlockQ5K], y: &[BlockQ8K]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::vec_dot_q5_k_q8_k(x, y) }
    }

    fn neon_v80_gemv_q5_k(
        w: &[BlockQ5K],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemv_q5_k_q8_k(w, x, out, rows, cols_blocks) }
    }

    fn neon_v80_vec_dot_q6_k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::vec_dot_q6_k_q8_k(x, y) }
    }

    fn neon_v80_gemv_q6_k(
        w: &[BlockQ6K],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemv_q6_k_q8_k(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_q4_1(
        w: &[BlockQ4_1],
        x: &[BlockQ8_1],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemv_q4_1_q8_1(w, x, out, rows, cols_blocks) }
    }

    fn neon_rms_norm_f32(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::rms_norm_f32(out, x, weight, eps) }
    }

    fn neon_dot_f32_bf16(x: &[f32], w_bf16: &[u16]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::dot_f32_bf16(x, w_bf16) }
    }

    fn neon_l2_norm_f32(x: &mut [f32], eps: f32) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::l2_norm_f32(x, eps) }
    }

    #[allow(clippy::too_many_arguments)]
    fn neon_gdn_recurrence_head(
        q_h: &[f32],
        k_h: &[f32],
        v_h: &[f32],
        g: f32,
        beta: f32,
        scale: f32,
        s_vh: &mut [f32],
        o_h: &mut [f32],
        head_dim: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gdn_recurrence_head(q_h, k_h, v_h, g, beta, scale, s_vh, o_h, head_dim) }
    }

    #[allow(clippy::too_many_arguments)]
    fn neon_mamba2_scan_head(
        x_h: &[f32],
        b_g: &[f32],
        c_g: &[f32],
        da: f32,
        dt: f32,
        d_h: f32,
        s_h: &mut [f32],
        y_h: &mut [f32],
        head_dim: usize,
        d_state: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            neon_v80::mamba2_scan_head(x_h, b_g, c_g, da, dt, d_h, s_h, y_h, head_dim, d_state)
        }
    }

    fn neon_v80_gemv_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks) }
    }

    fn neon_v80_gemm_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemm_q4_0x4_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    fn neon_v80_gemm_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemm_q8_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    fn neon_v80_gemm_q6_k(
        w: &[BlockQ6K],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::gemm_q6_k_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemm_q4_0x4_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemm_q8_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm_q6_k(
        w: &[BlockQ6K],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { dotprod::gemm_q6_k_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "i8mm")]
    fn i8mm_gemv_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `I8MM_TABLE` is only selected after runtime `i8mm` + `dotprod` detection.
        unsafe { i8mm::gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "i8mm")]
    fn i8mm_gemm_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `I8MM_TABLE` is only selected after runtime `i8mm` + `dotprod` detection.
        unsafe { i8mm::gemm_q4_0x4_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    fn neon_quant_q8_k(src: &[f32], dst: &mut [BlockQ8K]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::quantize_q8_k(src, dst) }
    }

    fn neon_dot_f32_q8_0(q: &[f32], k_q8: &[BlockQ8_0]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::dot_f32_q8_0(q, k_q8) }
    }

    fn neon_axpy_f32_q8_0(out: &mut [f32], w: f32, v_q8: &[BlockQ8_0]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::axpy_f32_q8_0(out, w, v_q8) }
    }

    #[cfg(feature = "lut")]
    fn lut_vec_dot_q2_0(x: &[BlockQ2_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::vec_dot_q2_0_q8_0(x, y) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::vec_dot_q2_0_q8_0 as lut_vec_dot_q2_0;

    #[cfg(feature = "lut")]
    fn lut_gemv_q2_0(
        w: &[BlockQ2_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemv_q2_0_q8_0(w, x, out, rows, cols_blocks) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemv_q2_0_q8_0 as lut_gemv_q2_0;

    #[cfg(feature = "lut")]
    fn lut_gemm_q2_0(
        w: &[BlockQ2_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemm_q2_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemm_q2_0_q8_0 as lut_gemm_q2_0;

    #[cfg(feature = "lut")]
    fn lut_vec_dot_tq2_0(x: &[BlockTQ2_0], y: &[BlockQ8K]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::vec_dot_tq2_0_q8_k(x, y) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::vec_dot_tq2_0_q8_k as lut_vec_dot_tq2_0;

    #[cfg(feature = "lut")]
    fn lut_gemv_tq2_0(
        w: &[BlockTQ2_0],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemv_tq2_0_q8_k(w, x, out, rows, cols_blocks) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemv_tq2_0_q8_k as lut_gemv_tq2_0;

    #[cfg(feature = "lut")]
    fn lut_gemm_tq2_0(
        w: &[BlockTQ2_0],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemm_tq2_0_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemm_tq2_0_q8_k as lut_gemm_tq2_0;

    #[cfg(feature = "lut")]
    fn lut_vec_dot_iq2_xxs(x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::vec_dot_iq2_xxs_q8_k(x, y) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::vec_dot_iq2_xxs_q8_k as lut_vec_dot_iq2_xxs;

    #[cfg(feature = "lut")]
    fn lut_gemv_iq2_xxs(
        w: &[BlockIQ2XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemv_iq2_xxs_q8_k(w, x, out, rows, cols_blocks) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemv_iq2_xxs_q8_k as lut_gemv_iq2_xxs;

    #[cfg(feature = "lut")]
    fn lut_gemm_iq2_xxs(
        w: &[BlockIQ2XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemm_iq2_xxs_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemm_iq2_xxs_q8_k as lut_gemm_iq2_xxs;

    #[cfg(feature = "lut")]
    fn lut_vec_dot_iq3_xxs(x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::vec_dot_iq3_xxs_q8_k(x, y) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::vec_dot_iq3_xxs_q8_k as lut_vec_dot_iq3_xxs;

    #[cfg(feature = "lut")]
    fn lut_gemv_iq3_xxs(
        w: &[BlockIQ3XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemv_iq3_xxs_q8_k(w, x, out, rows, cols_blocks) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemv_iq3_xxs_q8_k as lut_gemv_iq3_xxs;

    #[cfg(feature = "lut")]
    fn lut_gemm_iq3_xxs(
        w: &[BlockIQ3XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemm_iq3_xxs_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemm_iq3_xxs_q8_k as lut_gemm_iq3_xxs;

    #[cfg(feature = "lut")]
    fn lut_vec_dot_iq4_nl(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::vec_dot_iq4_nl_q8_0(x, y) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::vec_dot_iq4_nl_q8_0 as lut_vec_dot_iq4_nl;

    #[cfg(feature = "lut")]
    fn lut_gemv_iq4_nl(
        w: &[BlockIQ4NL],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemv_iq4_nl_q8_0(w, x, out, rows, cols_blocks) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemv_iq4_nl_q8_0 as lut_gemv_iq4_nl;

    #[cfg(feature = "lut")]
    fn lut_gemm_iq4_nl(
        w: &[BlockIQ4NL],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { lut::gemm_iq4_nl_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    #[cfg(not(feature = "lut"))]
    use generic::gemm_iq4_nl_q8_0 as lut_gemm_iq4_nl;

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_iq2_xxs(x: &[BlockIQ2XXS], y: &[BlockQ8K]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_vec_dot_iq2_xxs_q8_k(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_iq2_xxs(
        w: &[BlockIQ2XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_gemv_iq2_xxs_q8_k(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm_iq2_xxs(
        w: &[BlockIQ2XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_gemm_iq2_xxs_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_iq3_xxs(x: &[BlockIQ3XXS], y: &[BlockQ8K]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_vec_dot_iq3_xxs_q8_k(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_iq3_xxs(
        w: &[BlockIQ3XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_gemv_iq3_xxs_q8_k(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm_iq3_xxs(
        w: &[BlockIQ3XXS],
        x: &[BlockQ8K],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_gemm_iq3_xxs_q8_k(w, x, out, rows, cols_blocks, n_tokens) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_vec_dot_iq4_nl(x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_vec_dot_iq4_nl_q8_0(x, y) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemv_iq4_nl(
        w: &[BlockIQ4NL],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_gemv_iq4_nl_q8_0(w, x, out, rows, cols_blocks) }
    }

    #[cfg(feature = "dotprod")]
    fn dotprod_gemm_iq4_nl(
        w: &[BlockIQ4NL],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `DOTPROD_TABLE` is only selected after runtime `dotprod` detection.
        unsafe { lut::dotprod_gemm_iq4_nl_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    fn neon_gemv_f32_q8_0(
        w: &[BlockQ8_0],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        assert_eq!(w.len(), rows * cols_blocks);
        assert_eq!(out.len(), rows);
        for (r, dst) in out.iter_mut().enumerate() {
            *dst = neon_dot_f32_q8_0(x, &w[r * cols_blocks..(r + 1) * cols_blocks]);
        }
    }

    fn neon_gemm_f32_q8_0(
        w: &[BlockQ8_0],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        let cols = cols_blocks * tm_core::QK8_0;
        assert_eq!(x.len(), n_tokens * cols);
        assert_eq!(out.len(), n_tokens * rows);
        for t in 0..n_tokens {
            neon_gemv_f32_q8_0(
                w,
                &x[t * cols..(t + 1) * cols],
                &mut out[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }
    }

    pub const NEON_V80_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonV80,
        vec_dot_q4_0_q8_0: neon_v80_vec_dot,
        gemv_q4_0_q8_0: neon_v80_gemv,
        gemm_q4_0_q8_0: neon_v80_gemm,
        gemv_q4_0x4_q8_0: neon_v80_gemv_q4_0x4,
        gemm_q4_0x4_q8_0: neon_v80_gemm_q4_0x4,
        vec_dot_q4_1_q8_1: neon_v80_vec_dot_q4_1,
        gemv_q4_1_q8_1: neon_v80_gemv_q4_1,
        vec_dot_q8_0_q8_0: neon_v80_vec_dot_q8_0,
        gemv_q8_0_q8_0: neon_v80_gemv_q8_0,
        gemm_q8_0_q8_0: neon_v80_gemm_q8_0,
        vec_dot_q5_k_q8_k: neon_v80_vec_dot_q5_k,
        gemv_q5_k_q8_k: neon_v80_gemv_q5_k,
        vec_dot_q6_k_q8_k: neon_v80_vec_dot_q6_k,
        gemv_q6_k_q8_k: neon_v80_gemv_q6_k,
        gemm_q6_k_q8_k: neon_v80_gemm_q6_k,
        vec_dot_q2_0_q8_0: lut_vec_dot_q2_0,
        gemv_q2_0_q8_0: lut_gemv_q2_0,
        gemm_q2_0_q8_0: lut_gemm_q2_0,
        vec_dot_tq2_0_q8_k: lut_vec_dot_tq2_0,
        gemv_tq2_0_q8_k: lut_gemv_tq2_0,
        gemm_tq2_0_q8_k: lut_gemm_tq2_0,
        vec_dot_iq2_xxs_q8_k: lut_vec_dot_iq2_xxs,
        gemv_iq2_xxs_q8_k: lut_gemv_iq2_xxs,
        gemm_iq2_xxs_q8_k: lut_gemm_iq2_xxs,
        vec_dot_iq3_xxs_q8_k: lut_vec_dot_iq3_xxs,
        gemv_iq3_xxs_q8_k: lut_gemv_iq3_xxs,
        gemm_iq3_xxs_q8_k: lut_gemm_iq3_xxs,
        vec_dot_iq4_nl_q8_0: lut_vec_dot_iq4_nl,
        gemv_iq4_nl_q8_0: lut_gemv_iq4_nl,
        gemm_iq4_nl_q8_0: lut_gemm_iq4_nl,
        dot_f32_f16: neon_dot_f32_f16,
        dot_f32_bf16: neon_dot_f32_bf16,
        dot_f32_q8_0: neon_dot_f32_q8_0,
        gemv_f32_q8_0: neon_gemv_f32_q8_0,
        gemm_f32_q8_0: neon_gemm_f32_q8_0,
        axpy_f32: neon_axpy_f32,
        axpy_f32_f16: neon_axpy_f32_f16,
        axpy_f32_q8_0: neon_axpy_f32_q8_0,
        gemm_q4_0_transposed_tile: neon_gemm_q4_0_transposed_tile,
        f32_to_f16_slice: neon_f32_to_f16_slice,
        quantize_q8_0: neon_quant_q8_0,
        quantize_q8_1: neon_quant_q8_1,
        quantize_q8_k: neon_quant_q8_k,
        rms_norm_f32: neon_rms_norm_f32,
        l2_norm_f32: neon_l2_norm_f32,
        gdn_recurrence_head: neon_gdn_recurrence_head,
        mamba2_scan_head: neon_mamba2_scan_head,
    };

    #[cfg(feature = "dotprod")]
    pub const DOTPROD_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonDotprod,
        vec_dot_q4_0_q8_0: dotprod_vec_dot,
        gemv_q4_0_q8_0: dotprod_gemv,
        gemm_q4_0_q8_0: dotprod_gemm,
        gemv_q4_0x4_q8_0: dotprod_gemv_q4_0x4,
        gemm_q4_0x4_q8_0: dotprod_gemm_q4_0x4,
        vec_dot_q4_1_q8_1: dotprod_vec_dot_q4_1,
        gemv_q4_1_q8_1: dotprod_gemv_q4_1,
        vec_dot_q8_0_q8_0: dotprod_vec_dot_q8_0,
        gemv_q8_0_q8_0: dotprod_gemv_q8_0,
        gemm_q8_0_q8_0: dotprod_gemm_q8_0,
        vec_dot_q5_k_q8_k: dotprod_vec_dot_q5_k,
        gemv_q5_k_q8_k: dotprod_gemv_q5_k,
        vec_dot_q6_k_q8_k: dotprod_vec_dot_q6_k,
        gemv_q6_k_q8_k: dotprod_gemv_q6_k,
        gemm_q6_k_q8_k: dotprod_gemm_q6_k,
        vec_dot_q2_0_q8_0: lut_vec_dot_q2_0,
        gemv_q2_0_q8_0: lut_gemv_q2_0,
        gemm_q2_0_q8_0: lut_gemm_q2_0,
        vec_dot_tq2_0_q8_k: lut_vec_dot_tq2_0,
        gemv_tq2_0_q8_k: lut_gemv_tq2_0,
        gemm_tq2_0_q8_k: lut_gemm_tq2_0,
        vec_dot_iq2_xxs_q8_k: dotprod_vec_dot_iq2_xxs,
        gemv_iq2_xxs_q8_k: dotprod_gemv_iq2_xxs,
        gemm_iq2_xxs_q8_k: dotprod_gemm_iq2_xxs,
        vec_dot_iq3_xxs_q8_k: dotprod_vec_dot_iq3_xxs,
        gemv_iq3_xxs_q8_k: dotprod_gemv_iq3_xxs,
        gemm_iq3_xxs_q8_k: dotprod_gemm_iq3_xxs,
        vec_dot_iq4_nl_q8_0: dotprod_vec_dot_iq4_nl,
        gemv_iq4_nl_q8_0: dotprod_gemv_iq4_nl,
        gemm_iq4_nl_q8_0: dotprod_gemm_iq4_nl,
        dot_f32_f16: neon_dot_f32_f16,
        dot_f32_bf16: neon_dot_f32_bf16,
        dot_f32_q8_0: neon_dot_f32_q8_0,
        gemv_f32_q8_0: neon_gemv_f32_q8_0,
        gemm_f32_q8_0: neon_gemm_f32_q8_0,
        axpy_f32: neon_axpy_f32,
        axpy_f32_f16: neon_axpy_f32_f16,
        axpy_f32_q8_0: neon_axpy_f32_q8_0,
        gemm_q4_0_transposed_tile: neon_gemm_q4_0_transposed_tile,
        f32_to_f16_slice: neon_f32_to_f16_slice,
        quantize_q8_0: neon_quant_q8_0,
        quantize_q8_1: neon_quant_q8_1,
        quantize_q8_k: neon_quant_q8_k,
        rms_norm_f32: neon_rms_norm_f32,
        l2_norm_f32: neon_l2_norm_f32,
        gdn_recurrence_head: neon_gdn_recurrence_head,
        mamba2_scan_head: neon_mamba2_scan_head,
    };

    #[cfg(feature = "i8mm")]
    pub const I8MM_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonI8mm,
        vec_dot_q4_0_q8_0: i8mm_vec_dot,
        gemv_q4_0_q8_0: i8mm_gemv,
        gemm_q4_0_q8_0: i8mm_gemm,
        gemv_q4_0x4_q8_0: i8mm_gemv_q4_0x4,
        gemm_q4_0x4_q8_0: i8mm_gemm_q4_0x4,
        vec_dot_q4_1_q8_1: dotprod_vec_dot_q4_1,
        gemv_q4_1_q8_1: dotprod_gemv_q4_1,
        vec_dot_q8_0_q8_0: dotprod_vec_dot_q8_0,
        gemv_q8_0_q8_0: dotprod_gemv_q8_0,
        gemm_q8_0_q8_0: dotprod_gemm_q8_0,
        vec_dot_q5_k_q8_k: dotprod_vec_dot_q5_k,
        gemv_q5_k_q8_k: dotprod_gemv_q5_k,
        vec_dot_q6_k_q8_k: dotprod_vec_dot_q6_k,
        gemv_q6_k_q8_k: dotprod_gemv_q6_k,
        gemm_q6_k_q8_k: dotprod_gemm_q6_k,
        vec_dot_q2_0_q8_0: lut_vec_dot_q2_0,
        gemv_q2_0_q8_0: lut_gemv_q2_0,
        gemm_q2_0_q8_0: lut_gemm_q2_0,
        vec_dot_tq2_0_q8_k: lut_vec_dot_tq2_0,
        gemv_tq2_0_q8_k: lut_gemv_tq2_0,
        gemm_tq2_0_q8_k: lut_gemm_tq2_0,
        vec_dot_iq2_xxs_q8_k: dotprod_vec_dot_iq2_xxs,
        gemv_iq2_xxs_q8_k: dotprod_gemv_iq2_xxs,
        gemm_iq2_xxs_q8_k: dotprod_gemm_iq2_xxs,
        vec_dot_iq3_xxs_q8_k: dotprod_vec_dot_iq3_xxs,
        gemv_iq3_xxs_q8_k: dotprod_gemv_iq3_xxs,
        gemm_iq3_xxs_q8_k: dotprod_gemm_iq3_xxs,
        vec_dot_iq4_nl_q8_0: dotprod_vec_dot_iq4_nl,
        gemv_iq4_nl_q8_0: dotprod_gemv_iq4_nl,
        gemm_iq4_nl_q8_0: dotprod_gemm_iq4_nl,
        dot_f32_f16: neon_dot_f32_f16,
        dot_f32_bf16: neon_dot_f32_bf16,
        dot_f32_q8_0: neon_dot_f32_q8_0,
        gemv_f32_q8_0: neon_gemv_f32_q8_0,
        gemm_f32_q8_0: neon_gemm_f32_q8_0,
        axpy_f32: neon_axpy_f32,
        axpy_f32_f16: neon_axpy_f32_f16,
        axpy_f32_q8_0: neon_axpy_f32_q8_0,
        gemm_q4_0_transposed_tile: neon_gemm_q4_0_transposed_tile,
        f32_to_f16_slice: neon_f32_to_f16_slice,
        quantize_q8_0: neon_quant_q8_0,
        quantize_q8_1: neon_quant_q8_1,
        quantize_q8_k: neon_quant_q8_k,
        rms_norm_f32: neon_rms_norm_f32,
        l2_norm_f32: neon_l2_norm_f32,
        gdn_recurrence_head: neon_gdn_recurrence_head,
        mamba2_scan_head: neon_mamba2_scan_head,
    };
}

/// Selects the kernel table for a requested `Isa`, falling back to the widest
/// compiled tier if the target CPU or feature set does not include it.
#[cfg(target_arch = "x86_64")]
mod x86_64_wrappers {
    use super::*;
    use crate::x86_64::{avx2, avx512};

    fn avx512_vec_dot_q4_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl+fma` detection.
        unsafe { avx512::vec_dot_q4_0_q8_0(x, y) }
    }
    fn avx512_gemv_q4_0(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl+fma` detection.
        unsafe { avx512::gemv_q4_0_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx512_gemm_q4_0(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl+fma` detection.
        unsafe { avx512::gemm_q4_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    fn avx512_vec_dot_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl` detection.
        unsafe { avx512::vec_dot_q8_0_q8_0(x, y) }
    }
    fn avx512_gemv_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl` detection.
        unsafe { avx512::gemv_q8_0_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx512_gemm_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl` detection.
        unsafe { avx512::gemm_q8_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    fn avx512_gemv_f32_q8_0(
        w: &[BlockQ8_0],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl+fma` detection.
        unsafe { avx512::gemv_f32_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx512_gemm_f32_q8_0(
        w: &[BlockQ8_0],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX512_TABLE` is only selected after runtime `avx512f+bw+vl+fma` detection.
        unsafe { avx512::gemm_f32_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }

    fn avx2_vec_dot_q4_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::vec_dot_q4_0_q8_0(x, y) }
    }
    fn avx2_gemv_q4_0(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemv_q4_0_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx2_gemm_q4_0(
        w: &[BlockQ4_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemm_q4_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    fn avx2_gemv_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemv_q4_0x4_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx2_gemm_q4_0x4(
        w: &[BlockQ4_0x4],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemm_q4_0x4_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    fn avx2_vec_dot_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::vec_dot_q8_0_q8_0(x, y) }
    }
    fn avx2_gemv_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemv_q8_0_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx2_gemm_q8_0(
        w: &[BlockQ8_0],
        x: &[BlockQ8_0],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemm_q8_0_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    fn avx2_dot_f32_bf16(x: &[f32], w_bf16: &[u16]) -> f32 {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::dot_f32_bf16(x, w_bf16) }
    }
    fn avx2_dot_f32_q8_0(q: &[f32], k_q8: &[BlockQ8_0]) -> f32 {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::dot_f32_q8_0(q, k_q8) }
    }
    fn avx2_gemv_f32_q8_0(
        w: &[BlockQ8_0],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemv_f32_q8_0(w, x, out, rows, cols_blocks) }
    }
    fn avx2_gemm_f32_q8_0(
        w: &[BlockQ8_0],
        x: &[f32],
        out: &mut [f32],
        rows: usize,
        cols_blocks: usize,
        n_tokens: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gemm_f32_q8_0(w, x, out, rows, cols_blocks, n_tokens) }
    }
    fn avx2_axpy_f32_q8_0(out: &mut [f32], w: f32, v_q8: &[BlockQ8_0]) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::axpy_f32_q8_0(out, w, v_q8) }
    }
    fn avx2_rms_norm_f32(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::rms_norm_f32(out, x, weight, eps) }
    }
    fn avx2_l2_norm_f32(x: &mut [f32], eps: f32) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::l2_norm_f32(x, eps) }
    }
    #[allow(clippy::too_many_arguments)]
    fn avx2_gdn_recurrence_head(
        q_h: &[f32],
        k_h: &[f32],
        v_h: &[f32],
        g: f32,
        beta: f32,
        scale: f32,
        s_vh: &mut [f32],
        o_h: &mut [f32],
        head_dim: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma` detection.
        unsafe { avx2::gdn_recurrence_head(q_h, k_h, v_h, g, beta, scale, s_vh, o_h, head_dim) }
    }

    #[allow(clippy::too_many_arguments)]
    fn avx2_mamba2_scan_head(
        x_h: &[f32],
        b_g: &[f32],
        c_g: &[f32],
        da: f32,
        dt: f32,
        d_h: f32,
        s_h: &mut [f32],
        y_h: &mut [f32],
        head_dim: usize,
        d_state: usize,
    ) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma + f16c` detection.
        unsafe { avx2::mamba2_scan_head(x_h, b_g, c_g, da, dt, d_h, s_h, y_h, head_dim, d_state) }
    }
    fn avx2_dot_f32_f16(q: &[f32], k_f16: &[u16]) -> f32 {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma + f16c` detection.
        unsafe { avx2::dot_f32_f16(q, k_f16) }
    }
    fn avx2_axpy_f32_f16(out: &mut [f32], w: f32, v_f16: &[u16]) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma + f16c` detection.
        unsafe { avx2::axpy_f32_f16(out, w, v_f16) }
    }
    fn avx2_f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
        // SAFETY: `AVX2_TABLE` is only selected after runtime `avx2 + fma + f16c` detection.
        unsafe { avx2::f32_to_f16_slice(src, dst) }
    }

    pub const AVX2_TABLE: KernelTable = KernelTable {
        isa: Isa::Avx2,
        vec_dot_q4_0_q8_0: avx2_vec_dot_q4_0,
        gemv_q4_0_q8_0: avx2_gemv_q4_0,
        gemm_q4_0_q8_0: avx2_gemm_q4_0,
        gemv_q4_0x4_q8_0: avx2_gemv_q4_0x4,
        gemm_q4_0x4_q8_0: avx2_gemm_q4_0x4,
        vec_dot_q4_1_q8_1: generic::vec_dot_q4_1_q8_1,
        gemv_q4_1_q8_1: generic::gemv_q4_1_q8_1,
        vec_dot_q8_0_q8_0: avx2_vec_dot_q8_0,
        gemv_q8_0_q8_0: avx2_gemv_q8_0,
        gemm_q8_0_q8_0: avx2_gemm_q8_0,
        vec_dot_q5_k_q8_k: generic::vec_dot_q5_k_q8_k,
        gemv_q5_k_q8_k: generic::gemv_q5_k_q8_k,
        vec_dot_q6_k_q8_k: generic::vec_dot_q6_k_q8_k,
        gemv_q6_k_q8_k: generic::gemv_q6_k_q8_k,
        gemm_q6_k_q8_k: generic::gemm_q6_k_q8_k,
        vec_dot_q2_0_q8_0: generic::vec_dot_q2_0_q8_0,
        gemv_q2_0_q8_0: generic::gemv_q2_0_q8_0,
        gemm_q2_0_q8_0: generic::gemm_q2_0_q8_0,
        vec_dot_tq2_0_q8_k: generic::vec_dot_tq2_0_q8_k,
        gemv_tq2_0_q8_k: generic::gemv_tq2_0_q8_k,
        gemm_tq2_0_q8_k: generic::gemm_tq2_0_q8_k,
        vec_dot_iq2_xxs_q8_k: generic::vec_dot_iq2_xxs_q8_k,
        gemv_iq2_xxs_q8_k: generic::gemv_iq2_xxs_q8_k,
        gemm_iq2_xxs_q8_k: generic::gemm_iq2_xxs_q8_k,
        vec_dot_iq3_xxs_q8_k: generic::vec_dot_iq3_xxs_q8_k,
        gemv_iq3_xxs_q8_k: generic::gemv_iq3_xxs_q8_k,
        gemm_iq3_xxs_q8_k: generic::gemm_iq3_xxs_q8_k,
        vec_dot_iq4_nl_q8_0: generic::vec_dot_iq4_nl_q8_0,
        gemv_iq4_nl_q8_0: generic::gemv_iq4_nl_q8_0,
        gemm_iq4_nl_q8_0: generic::gemm_iq4_nl_q8_0,
        dot_f32_f16: avx2_dot_f32_f16,
        dot_f32_bf16: avx2_dot_f32_bf16,
        dot_f32_q8_0: avx2_dot_f32_q8_0,
        gemv_f32_q8_0: avx2_gemv_f32_q8_0,
        gemm_f32_q8_0: avx2_gemm_f32_q8_0,
        axpy_f32: generic::axpy_f32,
        axpy_f32_f16: avx2_axpy_f32_f16,
        axpy_f32_q8_0: avx2_axpy_f32_q8_0,
        gemm_q4_0_transposed_tile: generic::gemm_q4_0_transposed_tile,
        f32_to_f16_slice: avx2_f32_to_f16_slice,
        quantize_q8_0: generic::quantize_q8_0,
        quantize_q8_1: generic::quantize_q8_1,
        quantize_q8_k: generic::quantize_q8_k,
        rms_norm_f32: avx2_rms_norm_f32,
        l2_norm_f32: avx2_l2_norm_f32,
        gdn_recurrence_head: avx2_gdn_recurrence_head,
        mamba2_scan_head: avx2_mamba2_scan_head,
    };

    pub const AVX512_TABLE: KernelTable = KernelTable {
        isa: Isa::Avx512,
        vec_dot_q4_0_q8_0: avx512_vec_dot_q4_0,
        gemv_q4_0_q8_0: avx512_gemv_q4_0,
        gemm_q4_0_q8_0: avx512_gemm_q4_0,
        vec_dot_q8_0_q8_0: avx512_vec_dot_q8_0,
        gemv_q8_0_q8_0: avx512_gemv_q8_0,
        gemm_q8_0_q8_0: avx512_gemm_q8_0,
        gemv_f32_q8_0: avx512_gemv_f32_q8_0,
        gemm_f32_q8_0: avx512_gemm_f32_q8_0,
        ..AVX2_TABLE
    };
}

pub fn select_kernels(isa: Isa) -> &'static KernelTable {
    let caps = detect_caps();
    if !isa_supported(isa, caps) {
        return &GENERIC_TABLE;
    }

    match isa {
        Isa::Generic => &GENERIC_TABLE,
        #[cfg(target_arch = "aarch64")]
        Isa::NeonV80 => &aarch64_wrappers::NEON_V80_TABLE,
        #[cfg(all(target_arch = "aarch64", feature = "dotprod"))]
        Isa::NeonDotprod => &aarch64_wrappers::DOTPROD_TABLE,
        #[cfg(all(target_arch = "aarch64", not(feature = "dotprod")))]
        Isa::NeonDotprod => &aarch64_wrappers::NEON_V80_TABLE,
        #[cfg(all(target_arch = "aarch64", feature = "i8mm"))]
        Isa::NeonI8mm => &aarch64_wrappers::I8MM_TABLE,
        #[cfg(all(target_arch = "aarch64", not(feature = "i8mm"), feature = "dotprod"))]
        Isa::NeonI8mm => &aarch64_wrappers::DOTPROD_TABLE,
        #[cfg(all(
            target_arch = "aarch64",
            not(feature = "i8mm"),
            not(feature = "dotprod")
        ))]
        Isa::NeonI8mm => &aarch64_wrappers::NEON_V80_TABLE,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => &x86_64_wrappers::AVX2_TABLE,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => &x86_64_wrappers::AVX512_TABLE,
        _ => &GENERIC_TABLE,
    }
}

/// Returns the active kernel table for the current process (respects `TM_FORCE_ISA`).
pub fn active_kernels() -> &'static KernelTable {
    select_kernels(effective_isa())
}
