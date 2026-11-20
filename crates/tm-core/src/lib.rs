#![forbid(unsafe_code)]

//! Core data types, quantization block layouts, error types, and static arena
//! for `tinymodel`. Intentionally has zero external dependencies.

pub mod arena;
pub mod dtype;
pub mod error;
pub mod fp16;
pub mod memprobe;

pub use arena::Arena;
pub use dtype::{
    BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, DType, QK_K, QK4_0,
    QK4_1, QK8_0, QK8_1, dequantize_row_bytes, dequantize_row_q4_0, quantize_row_q4_0,
    quantize_row_q8_0, quantize_row_q8_1, quantize_row_q8_k, unpack_q5k_scales_mins,
};
pub use error::CoreError;
pub use fp16::{f16_to_f32, f32_to_f16};
pub use memprobe::{MemProbeError, MemoryBudget, probe_process_rss_bytes, probe_system_memory};
