//! Simple bump-allocated scratch arena for zero-allocation token generation.

use crate::error::CoreError;

/// Contiguous byte arena aligned to 64-byte cache lines.
pub struct Arena {
    buf: Vec<u8>,
    offset: usize,
}

impl Arena {
    /// Allocates an arena of `capacity` bytes up front.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Resets the bump pointer without freeing backing memory.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Reserves `bytes` aligned to `align` (must be power of two) and returns a mutable slice.
    pub fn alloc_bytes(&mut self, bytes: usize, align: usize) -> Result<&mut [u8], CoreError> {
        debug_assert!(align.is_power_of_two());
        let aligned = (self.offset + (align - 1)) & !(align - 1);
        let end = aligned.checked_add(bytes).ok_or(CoreError::OutOfMemory {
            requested: bytes,
            available: self.buf.len().saturating_sub(self.offset),
        })?;
        if end > self.buf.len() {
            return Err(CoreError::OutOfMemory {
                requested: bytes,
                available: self.buf.len().saturating_sub(aligned),
            });
        }
        self.offset = end;
        Ok(&mut self.buf[aligned..end])
    }

    #[inline]
    pub fn used(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }
}
