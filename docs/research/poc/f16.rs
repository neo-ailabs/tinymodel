use std::arch::aarch64::*;
#[target_feature(enable = "fp16")]
unsafe fn f(a: float16x8_t) -> float16x8_t { vfmaq_f16(a, a, a) }
fn main(){ unsafe { let _ = f(vdupq_n_f16(1.0)); println!("f16 ok"); } }
