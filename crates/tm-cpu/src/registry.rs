//! Runtime kernel registry and ISA dispatch table.
//!
//! Design inspired by Strata `src/quant/registry.hpp` and `dispatch.cpp`
//! (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors).

use tm_core::{BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K};

use crate::generic;
use crate::isa::{Isa, detect_caps, effective_isa, isa_supported};

/// Function-pointer types for CPU compute kernels.
pub type VecDotQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0]) -> f32;
pub type GemvQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0], &mut [f32], usize, usize, usize);
pub type VecDotQ4_1Fn = fn(&[BlockQ4_1], &[BlockQ8_1]) -> f32;
pub type GemvQ4_1Fn = fn(&[BlockQ4_1], &[BlockQ8_1], &mut [f32], usize, usize);
pub type VecDotQ8_0Fn = fn(&[BlockQ8_0], &[BlockQ8_0]) -> f32;
pub type GemvQ8_0Fn = fn(&[BlockQ8_0], &[BlockQ8_0], &mut [f32], usize, usize);
pub type VecDotQ5KFn = fn(&[BlockQ5K], &[BlockQ8K]) -> f32;
pub type GemvQ5KFn = fn(&[BlockQ5K], &[BlockQ8K], &mut [f32], usize, usize);
pub type VecDotQ6KFn = fn(&[BlockQ6K], &[BlockQ8K]) -> f32;
pub type GemvQ6KFn = fn(&[BlockQ6K], &[BlockQ8K], &mut [f32], usize, usize);
pub type DotF32F16Fn = fn(&[f32], &[u16]) -> f32;
pub type AxpyF32F16Fn = fn(&mut [f32], f32, &[u16]);
pub type F32ToF16SliceFn = fn(&[f32], &mut [u16]);
pub type QuantQ8_0Fn = fn(&[f32], &mut [BlockQ8_0]);
pub type QuantQ8_1Fn = fn(&[f32], &mut [BlockQ8_1]);

/// Function-pointer table for CPU compute kernels at a given ISA tier.
#[derive(Clone, Copy)]
pub struct KernelTable {
    pub isa: Isa,
    pub vec_dot_q4_0_q8_0: VecDotQ4_0Fn,
    pub gemv_q4_0_q8_0: GemvQ4_0Fn,
    pub gemm_q4_0_q8_0: GemmQ4_0Fn,
    pub vec_dot_q4_1_q8_1: VecDotQ4_1Fn,
    pub gemv_q4_1_q8_1: GemvQ4_1Fn,
    pub vec_dot_q8_0_q8_0: VecDotQ8_0Fn,
    pub gemv_q8_0_q8_0: GemvQ8_0Fn,
    pub vec_dot_q5_k_q8_k: VecDotQ5KFn,
    pub gemv_q5_k_q8_k: GemvQ5KFn,
    pub vec_dot_q6_k_q8_k: VecDotQ6KFn,
    pub gemv_q6_k_q8_k: GemvQ6KFn,
    pub dot_f32_f16: DotF32F16Fn,
    pub axpy_f32_f16: AxpyF32F16Fn,
    pub f32_to_f16_slice: F32ToF16SliceFn,
    pub quantize_q8_0: QuantQ8_0Fn,
    pub quantize_q8_1: QuantQ8_1Fn,
}

const GENERIC_TABLE: KernelTable = KernelTable {
    isa: Isa::Generic,
    vec_dot_q4_0_q8_0: generic::vec_dot_q4_0_q8_0,
    gemv_q4_0_q8_0: generic::gemv_q4_0_q8_0,
    gemm_q4_0_q8_0: generic::gemm_q4_0_q8_0,
    vec_dot_q4_1_q8_1: generic::vec_dot_q4_1_q8_1,
    gemv_q4_1_q8_1: generic::gemv_q4_1_q8_1,
    vec_dot_q8_0_q8_0: generic::vec_dot_q8_0_q8_0,
    gemv_q8_0_q8_0: generic::gemv_q8_0_q8_0,
    vec_dot_q5_k_q8_k: generic::vec_dot_q5_k_q8_k,
    gemv_q5_k_q8_k: generic::gemv_q5_k_q8_k,
    vec_dot_q6_k_q8_k: generic::vec_dot_q6_k_q8_k,
    gemv_q6_k_q8_k: generic::gemv_q6_k_q8_k,
    dot_f32_f16: generic::dot_f32_f16,
    axpy_f32_f16: generic::axpy_f32_f16,
    f32_to_f16_slice: generic::f32_to_f16_slice,
    quantize_q8_0: generic::quantize_q8_0,
    quantize_q8_1: generic::quantize_q8_1,
};

#[cfg(target_arch = "aarch64")]
mod aarch64_wrappers {
    use super::*;
    #[cfg(feature = "dotprod")]
    use crate::aarch64::dotprod;
    #[cfg(feature = "i8mm")]
    use crate::aarch64::i8mm;
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

    fn neon_axpy_f32_f16(out: &mut [f32], w: f32, v_f16: &[u16]) {
        // SAFETY: NEON is always available on aarch64.
        unsafe { neon_v80::axpy_f32_f16(out, w, v_f16) }
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

    pub const NEON_V80_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonV80,
        vec_dot_q4_0_q8_0: neon_v80_vec_dot,
        gemv_q4_0_q8_0: neon_v80_gemv,
        gemm_q4_0_q8_0: neon_v80_gemm,
        vec_dot_q4_1_q8_1: generic::vec_dot_q4_1_q8_1,
        gemv_q4_1_q8_1: generic::gemv_q4_1_q8_1,
        vec_dot_q8_0_q8_0: generic::vec_dot_q8_0_q8_0,
        gemv_q8_0_q8_0: generic::gemv_q8_0_q8_0,
        vec_dot_q5_k_q8_k: generic::vec_dot_q5_k_q8_k,
        gemv_q5_k_q8_k: generic::gemv_q5_k_q8_k,
        vec_dot_q6_k_q8_k: generic::vec_dot_q6_k_q8_k,
        gemv_q6_k_q8_k: generic::gemv_q6_k_q8_k,
        dot_f32_f16: neon_dot_f32_f16,
        axpy_f32_f16: neon_axpy_f32_f16,
        f32_to_f16_slice: neon_f32_to_f16_slice,
        quantize_q8_0: neon_quant_q8_0,
        quantize_q8_1: neon_quant_q8_1,
    };

    #[cfg(feature = "dotprod")]
    pub const DOTPROD_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonDotprod,
        vec_dot_q4_0_q8_0: dotprod_vec_dot,
        gemv_q4_0_q8_0: dotprod_gemv,
        gemm_q4_0_q8_0: dotprod_gemm,
        vec_dot_q4_1_q8_1: dotprod_vec_dot_q4_1,
        gemv_q4_1_q8_1: dotprod_gemv_q4_1,
        vec_dot_q8_0_q8_0: dotprod_vec_dot_q8_0,
        gemv_q8_0_q8_0: dotprod_gemv_q8_0,
        vec_dot_q5_k_q8_k: dotprod_vec_dot_q5_k,
        gemv_q5_k_q8_k: dotprod_gemv_q5_k,
        vec_dot_q6_k_q8_k: dotprod_vec_dot_q6_k,
        gemv_q6_k_q8_k: dotprod_gemv_q6_k,
        dot_f32_f16: neon_dot_f32_f16,
        axpy_f32_f16: neon_axpy_f32_f16,
        f32_to_f16_slice: neon_f32_to_f16_slice,
        quantize_q8_0: neon_quant_q8_0,
        quantize_q8_1: neon_quant_q8_1,
    };

    #[cfg(feature = "i8mm")]
    pub const I8MM_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonI8mm,
        vec_dot_q4_0_q8_0: dotprod_vec_dot,
        gemv_q4_0_q8_0: dotprod_gemv,
        gemm_q4_0_q8_0: i8mm_gemm,
        vec_dot_q4_1_q8_1: dotprod_vec_dot_q4_1,
        gemv_q4_1_q8_1: dotprod_gemv_q4_1,
        vec_dot_q8_0_q8_0: dotprod_vec_dot_q8_0,
        gemv_q8_0_q8_0: dotprod_gemv_q8_0,
        vec_dot_q5_k_q8_k: dotprod_vec_dot_q5_k,
        gemv_q5_k_q8_k: dotprod_gemv_q5_k,
        vec_dot_q6_k_q8_k: dotprod_vec_dot_q6_k,
        gemv_q6_k_q8_k: dotprod_gemv_q6_k,
        dot_f32_f16: neon_dot_f32_f16,
        axpy_f32_f16: neon_axpy_f32_f16,
        f32_to_f16_slice: neon_f32_to_f16_slice,
        quantize_q8_0: neon_quant_q8_0,
        quantize_q8_1: neon_quant_q8_1,
    };
}

/// Selects the kernel table for a requested `Isa`, falling back to the widest
/// compiled tier if the target CPU or feature set does not include it.
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
        _ => &GENERIC_TABLE,
    }
}

/// Returns the active kernel table for the current process (respects `TM_FORCE_ISA`).
pub fn active_kernels() -> &'static KernelTable {
    select_kernels(effective_isa())
}
