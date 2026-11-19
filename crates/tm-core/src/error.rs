//! Core error types (zero external dependencies).

use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidShape { expected: usize, actual: usize },
    OutOfMemory { requested: usize, available: usize },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::InvalidShape { expected, actual } => {
                write!(f, "invalid shape: expected {expected}, got {actual}")
            }
            CoreError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "arena out of memory: requested {requested} bytes, available {available} bytes"
                )
            }
        }
    }
}

impl std::error::Error for CoreError {}
