#![forbid(unsafe_code)]

//! Core data types, quantization block layouts, error types, and static arena
//! for `tinymodel`. Intentionally has zero external dependencies.

pub mod arena;
pub mod dtype;
pub mod error;
pub mod fp16;

pub use arena::Arena;
pub use dtype::{
    BlockQ4_0, BlockQ8_0, DType, QK4_0, QK8_0, dequantize_row_q4_0, quantize_row_q4_0,
    quantize_row_q8_0,
};
pub use error::CoreError;
pub use fp16::{f16_to_f32, f32_to_f16};
