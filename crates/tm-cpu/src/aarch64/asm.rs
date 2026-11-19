//! Inline assembly helpers for AArch64 instructions whose `std::arch` intrinsics
//! are not yet stabilized in Rust stable (or for single-instruction FP16 conversion).
//!
//! Per `AGENTS.md` rule #1, unstable ISA instructions (`i8mm`, `fp16` arithmetic,
//! `sme`) live exclusively in this module using `core::arch::asm!`.

use core::arch::aarch64::{int8x16_t, int32x4_t};
use core::arch::asm;

/// Signed 8-bit 2x8 * 8x2 matrix multiply-accumulate (`smmla` from ARMv8.6 `i8mm`).
///
/// Computes `acc[2x2] += a[2x8] * b^T[2x8]` where `a` and `b` are 16-byte vectors
/// storing two 8-element `i8` rows each, and `acc` holds four `i32` accumulators.
///
/// # Safety
/// Caller must guarantee the CPU supports the `i8mm` feature.
#[target_feature(enable = "i8mm")]
#[inline]
pub unsafe fn vmmlaq_s32(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: `i8mm` target feature is enabled on this function and verified by caller;
    // `smmla` only reads/writes the provided SIMD vector registers with no memory side effects.
    unsafe {
        asm!(
            "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// Hardware IEEE 754 binary16 (`u16`) -> `f32` conversion via base ARMv8.0 `fcvt s, h`.
///
/// Base ARMv8.0-A guarantees FP16 storage conversion (`fcvt s, h`), including on
/// Cortex-A53 (Pi Zero 2W), Cortex-A76 (Pi 5), and Apple Silicon.
#[inline(always)]
pub fn f16_to_f32_hw(bits: u16) -> f32 {
    let out: f32;
    // SAFETY: `fmov h, w` + `fcvt s, h` are mandatory base ARMv8.0 FP instructions
    // operating purely on registers with no memory access.
    unsafe {
        asm!(
            "fmov {tmp:h}, {inp:w}",
            "fcvt {out:s}, {tmp:h}",
            inp = in(reg) bits as u32,
            tmp = out(vreg) _,
            out = out(vreg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}
