//! CPU backend for `tinymodel`: runtime ISA detection, kernel registry,
//! scalar reference implementations, SIMD/asm kernels, and thread pool.

pub mod generic;
pub mod isa;
pub mod pool;
pub mod registry;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub use isa::{Caps, Isa, best_isa, detect_caps, effective_isa};
pub use pool::ThreadPool;
pub use registry::{KernelTable, active_kernels, select_kernels};
