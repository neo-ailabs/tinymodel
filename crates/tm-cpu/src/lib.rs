//! CPU backend for `tinymodel`: runtime ISA detection, kernel registry,
//! scalar reference implementations, SIMD/asm kernels, and thread pool.

pub mod affinity;
pub mod generic;
pub mod isa;
pub mod ops;
pub mod pool;
pub mod registry;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub use affinity::{CpuTopology, detected_topology, recommended_threads};
pub use isa::{Caps, Isa, best_isa, detect_caps, effective_isa};
pub use ops::{
    MAX_CPU_BATCH, MatvecScratch, RopeType, gdn_conv1d_silu_split_parallel,
    gdn_recurrence_gated_parallel, gdn_recurrence_parallel, geglu_inplace, geglu_inplace_parallel,
    gelu_tanh_f32, gqa_attention_f16_kv, gqa_attention_f16_kv_batch_parallel,
    gqa_attention_f16_kv_ext_batch_parallel, gqa_attention_f16_kv_ext_parallel,
    gqa_attention_f16_kv_parallel, gqa_attention_q8_0_kv_ext_batch_parallel,
    gqa_attention_q8_0_kv_ext_parallel, l2_norm_inplace, matmul_f32, matmul_pair_f32,
    matmul_triple_f32, matvec_f32, matvec_pair_f32, matvec_triple_f32, rms_norm,
    rms_norm_batch_parallel, rms_norm_unit_inplace, rope_compute_cos_sin, rope_inplace,
    rope_inplace_tables, softcap_inplace_parallel, softmax_inplace, swiglu_inplace,
    swiglu_inplace_parallel,
};
pub use pool::ThreadPool;
pub use registry::{KernelTable, active_kernels, select_kernels};
