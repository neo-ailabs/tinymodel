use std::arch::aarch64::*;
#[target_feature(enable = "dotprod")]
unsafe fn d(a: int8x16_t, b: int8x16_t) -> int32x4_t { vdotq_s32(vdupq_n_s32(0), a, b) }
fn main(){ unsafe { let a=vdupq_n_s8(2); let r=d(a,a); println!("dotprod {}", vgetq_lane_s32::<0>(r)); } }
