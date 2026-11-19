//! CPU backend for `tinymodel`: runtime ISA detection, kernel registry,
//! scalar reference implementations, SIMD/asm kernels, and thread pool.

pub mod generic;
pub mod isa;
pub mod ops;
pub mod pool;
pub mod registry;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub use isa::{Caps, Isa, best_isa, detect_caps, effective_isa};
pub use ops::{
    MAX_CPU_BATCH, MatvecScratch, RopeType, gqa_attention_f16_kv, gqa_attention_f16_kv_parallel,
    matmul_f32, matvec_f32, matvec_pair_f32, matvec_triple_f32, rms_norm, rope_compute_cos_sin,
    rope_inplace, rope_inplace_tables, softmax_inplace, swiglu_inplace,
};
pub use pool::ThreadPool;
pub use registry::{KernelTable, active_kernels, select_kernels};
