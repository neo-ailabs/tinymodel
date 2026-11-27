#![forbid(unsafe_code)]

//! Core data types, quantization block layouts, error types, and static arena
//! for `tinymodel`. Intentionally has zero external dependencies.

pub mod arena;
pub mod dtype;
pub mod error;
pub mod fp16;
pub mod iq_tables;
pub mod memprobe;

pub use arena::Arena;
pub use dtype::{
    BlockIQ2XXS, BlockIQ3XXS, BlockIQ4NL, BlockNvFp4, BlockQ2_0, BlockQ4_0, BlockQ4_0x4, BlockQ4_1,
    BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, BlockTQ2_0, DType, GGML_TYPE_TM_Q4_0X4,
    KvType, QK_K, QK_NVFP4, QK_NVFP4_SUB, QK2_0, QK4_0, QK4_0X4, QK4_1, QK4_NL, QK8_0, QK8_1,
    UE4M3_TO_F32_LUT, dequantize_q4_0x4_single_row, dequantize_row_bytes, dequantize_row_q4_0,
    f32_to_ue4m3, quantize_row_iq2_xxs, quantize_row_iq3_xxs, quantize_row_iq4_nl,
    quantize_row_nvfp4, quantize_row_q2_0, quantize_row_q4_0, quantize_row_q8_0, quantize_row_q8_1,
    quantize_row_q8_k, quantize_row_to_bytes, quantize_row_tq2_0, repack_q4_0_bytes_to_q4_0x4,
    repack_q4_0_to_q4_0x4, ue4m3_to_f32, unpack_q5k_scales_mins,
};
pub use error::CoreError;
pub use fp16::{f16_to_f32, f32_to_f16};
pub use iq_tables::{IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KSIGNS64, KVALUES_FP4, KVALUES_IQ4NL};
pub use memprobe::{
    MemProbeError, MemoryBudget, ProcessMemoryStats, probe_process_memory_stats,
    probe_process_rss_bytes, probe_system_memory,
};
