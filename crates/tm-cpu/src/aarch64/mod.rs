//! AArch64 SIMD and inline-assembly kernels.

pub mod asm;
#[cfg(feature = "dotprod")]
pub mod dotprod;
#[cfg(feature = "i8mm")]
pub mod i8mm;
#[cfg(feature = "lut")]
pub mod lut;
pub mod neon_v80;
