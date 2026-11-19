//! IEEE 754 binary16 (FP16) <-> binary32 (FP32) conversion in pure safe Rust.
//! Exact for all finite, subnormal, zero, infinity, and NaN bit patterns.

/// Converts an IEEE 754 binary16 bit pattern (`u16`) to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) >> 10) & 0x1F;
    let mant = (bits as u32) & 0x03FF;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal binary16 -> normalized binary32
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let exp32 = ((e + 127) as u32) << 23;
            sign | exp32 | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        // Normal
        let exp32 = (exp + (127 - 15)) << 23;
        sign | exp32 | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Converts an `f32` to IEEE 754 binary16 bit pattern (`u16`) with round-to-nearest-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf or NaN
        if mant == 0 {
            return sign | 0x7C00;
        }
        let nan_mant = ((mant >> 13) as u16) | 1;
        return sign | 0x7C00 | nan_mant;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow to Inf
        return sign | 0x7C00;
    }

    if unbiased < -14 {
        // Subnormal or underflow to zero
        if unbiased < -25 {
            return sign;
        }
        let shift = (-14 - unbiased) as u32 + 13;
        let full_mant = mant | 0x0080_0000;
        let half_mant = full_mant >> shift;
        let rem_mask = (1u32 << shift) - 1;
        let rem = full_mant & rem_mask;
        let halfway = 1u32 << (shift - 1);
        let rounded = if rem > halfway || (rem == halfway && (half_mant & 1) != 0) {
            half_mant + 1
        } else {
            half_mant
        };
        return sign | (rounded as u16);
    }

    // Normal number with round-to-nearest-even
    let exp16 = ((unbiased + 15) as u32) << 10;
    let mant16 = mant >> 13;
    let rem = mant & 0x1FFF;
    let mut res = exp16 | mant16;
    if rem > 0x1000 || (rem == 0x1000 && (mant16 & 1) != 0) {
        res += 1;
    }
    sign | (res as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_special_and_powers_of_two() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 0.5, -0.25, 2.0, 65504.0, -65504.0] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert_eq!(v.to_bits(), back.to_bits());
        }
        assert!(f16_to_f32(0x7C00).is_infinite() && f16_to_f32(0x7C00).is_sign_positive());
        assert!(f16_to_f32(0xFC00).is_infinite() && f16_to_f32(0xFC00).is_sign_negative());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn roundtrip_all_finite_f16() {
        for bits in 0u16..=u16::MAX {
            let exp = (bits >> 10) & 0x1F;
            if exp == 31 {
                continue;
            }
            let f = f16_to_f32(bits);
            let back = f32_to_f16(f);
            assert_eq!(bits, back, "mismatch for f16 bits 0x{bits:04x} (f={f})");
        }
    }
}
