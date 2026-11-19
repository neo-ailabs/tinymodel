use std::arch::aarch64::*;
#[target_feature(enable = "i8mm")]
unsafe fn m(a: int8x16_t, b: int8x16_t) -> int32x4_t { vmmlaq_s32(vdupq_n_s32(0), a, b) }
fn main(){ unsafe { let a=vdupq_n_s8(2); let r=m(a,a); println!("i8mm {}", vgetq_lane_s32::<0>(r)); } }
