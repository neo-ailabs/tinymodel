use std::arch::{aarch64::*, asm};
#[target_feature(enable = "dotprod")]
unsafe fn d(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
  asm!("sdot {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) acc, in(vreg) a, in(vreg) b, options(pure, nomem, nostack)); acc }
fn main(){ unsafe { let a=vdupq_n_s8(2); let r=d(vdupq_n_s32(0),a,a); println!("asm sdot {} (feat={})", vgetq_lane_s32::<0>(r), std::arch::is_aarch64_feature_detected!("dotprod")); } }
