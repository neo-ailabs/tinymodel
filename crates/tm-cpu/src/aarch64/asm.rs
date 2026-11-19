//! Inline assembly helpers for AArch64 instructions whose `std::arch` intrinsics
//! are not yet stabilized in Rust stable (or for single-instruction FP16 conversion).
//!
//! Per `AGENTS.md` rule #1, unstable ISA instructions (`i8mm`, `fp16` arithmetic,
//! `sme`) live exclusively in this module using `core::arch::asm!`.

use core::arch::aarch64::{float32x4_t, uint16x8_t};
#[cfg(feature = "i8mm")]
use core::arch::aarch64::{int8x16_t, int32x4_t};
use core::arch::asm;

/// Signed 8-bit 2x8 * 8x2 matrix multiply-accumulate (`smmla` from ARMv8.6 `i8mm`).
///
/// Computes `acc[2x2] += a[2x8] * b^T[2x8]` where `a` and `b` are 16-byte vectors
/// storing two 8-element `i8` rows each, and `acc` holds four `i32` accumulators.
///
/// # Safety
/// Caller must guarantee the CPU supports the `i8mm` feature.
#[cfg(feature = "i8mm")]
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

/// Hardware IEEE 754 binary16 (`u16`) -> `f32` conversion via base ARMv8.0 `ins` + `fcvt s, h`.
///
/// Both `ins v.h[0], w` and `fcvt s, h` are mandatory base ARMv8.0-A instructions
/// (unlike `fmov h, w` which requires `fullfp16`), so this works on Cortex-A53
/// (Pi Zero 2W), Cortex-A72 (Pi 4), Cortex-A76 (Pi 5), and Apple Silicon.
#[inline(always)]
pub fn f16_to_f32_hw(bits: u16) -> f32 {
    let out: f32;
    // SAFETY: `ins v.h[0], w` + `fcvt s, h` are mandatory base ARMv8.0 NEON/FP instructions
    // operating purely on registers with no memory access.
    unsafe {
        asm!(
            "ins {tmp:v}.h[0], {inp:w}",
            "fcvt {out:s}, {tmp:h}",
            inp = in(reg) bits as u32,
            tmp = out(vreg) _,
            out = out(vreg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}

/// Hardware `f32` -> IEEE 754 binary16 (`u16`) conversion via base ARMv8.0 `fcvt h, s` + `umov`.
/// Mandatory in base ARMv8.0-A (works on Cortex-A53, Cortex-A76, and Apple Silicon).
#[inline(always)]
pub fn f32_to_f16_hw(val: f32) -> u16 {
    let out: u32;
    // SAFETY: `fcvt h, s` and `umov w, v.h[0]` are mandatory base ARMv8.0 NEON/FP instructions
    // operating purely on registers with no memory side effects.
    unsafe {
        asm!(
            "fcvt {tmp:h}, {inp:s}",
            "umov {out:w}, {tmp:v}.h[0]",
            inp = in(vreg) val,
            tmp = out(vreg) _,
            out = out(reg) out,
            options(pure, nomem, nostack),
        );
    }
    out as u16
}

/// Converts 8 packed IEEE 754 `f16` lanes (`uint16x8_t`) into two `float32x4_t` vectors
/// via base ARMv8.0 `fcvt v.4s, v.4h` and `fcvt2 v.4s, v.8h` (2 single-cycle instructions).
///
/// # Safety
/// Caller must ensure NEON is available (always true on `aarch64`).
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cvt_f16x8_to_f32x4x2(bits: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let lo: float32x4_t;
    let hi: float32x4_t;
    // SAFETY: `fcvtl` and `fcvtl2` are mandatory base ARMv8.0-A NEON instructions.
    unsafe {
        asm!(
            "fcvtl {lo:v}.4s, {inp:v}.4h",
            "fcvtl2 {hi:v}.4s, {inp:v}.8h",
            inp = in(vreg) bits,
            lo = out(vreg) lo,
            hi = out(vreg) hi,
            options(pure, nomem, nostack),
        );
    }
    (lo, hi)
}

/// Converts two `float32x4_t` vectors into 8 packed IEEE 754 `f16` lanes (`uint16x8_t`)
/// via base ARMv8.0 `fcvtn v.4h, v.4s` and `fcvtn2 v.8h, v.4s`.
///
/// # Safety
/// Caller must ensure NEON is available (always true on `aarch64`).
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cvt_f32x4x2_to_f16x8(lo: float32x4_t, hi: float32x4_t) -> uint16x8_t {
    let mut out: uint16x8_t;
    // SAFETY: `fcvtn` and `fcvtn2` are mandatory base ARMv8.0-A NEON instructions.
    unsafe {
        asm!(
            "fcvtn {out:v}.4h, {lo:v}.4s",
            "fcvtn2 {out:v}.8h, {hi:v}.4s",
            lo = in(vreg) lo,
            hi = in(vreg) hi,
            out = out(vreg) out,
            options(pure, nomem, nostack),
        );
    }
    out
}
