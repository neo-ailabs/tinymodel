//! Runtime kernel registry and ISA dispatch table.
//!
//! Design inspired by Strata `src/quant/registry.hpp` and `dispatch.cpp`
//! (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors).

use tm_core::{BlockQ4_0, BlockQ8_0};

use crate::generic;
use crate::isa::{Isa, detect_caps, effective_isa, isa_supported};

/// Function-pointer types for CPU compute kernels.
pub type VecDotQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0]) -> f32;
pub type GemvQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0], &mut [f32], usize, usize);
pub type GemmQ4_0Fn = fn(&[BlockQ4_0], &[BlockQ8_0], &mut [f32], usize, usize, usize);

/// Function-pointer table for CPU compute kernels at a given ISA tier.
#[derive(Clone, Copy)]
pub struct KernelTable {
    pub isa: Isa,
    pub vec_dot_q4_0_q8_0: VecDotQ4_0Fn,
    pub gemv_q4_0_q8_0: GemvQ4_0Fn,
    pub gemm_q4_0_q8_0: GemmQ4_0Fn,
}

const GENERIC_TABLE: KernelTable = KernelTable {
    isa: Isa::Generic,
    vec_dot_q4_0_q8_0: generic::vec_dot_q4_0_q8_0,
    gemv_q4_0_q8_0: generic::gemv_q4_0_q8_0,
    gemm_q4_0_q8_0: generic::gemm_q4_0_q8_0,
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

    pub const NEON_V80_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonV80,
        vec_dot_q4_0_q8_0: neon_v80_vec_dot,
        gemv_q4_0_q8_0: neon_v80_gemv,
        gemm_q4_0_q8_0: neon_v80_gemm,
    };

    #[cfg(feature = "dotprod")]
    pub const DOTPROD_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonDotprod,
        vec_dot_q4_0_q8_0: dotprod_vec_dot,
        gemv_q4_0_q8_0: dotprod_gemv,
        gemm_q4_0_q8_0: dotprod_gemm,
    };

    #[cfg(feature = "i8mm")]
    pub const I8MM_TABLE: KernelTable = KernelTable {
        isa: Isa::NeonI8mm,
        vec_dot_q4_0_q8_0: dotprod_vec_dot,
        gemv_q4_0_q8_0: dotprod_gemv,
        gemm_q4_0_q8_0: i8mm_gemm,
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
