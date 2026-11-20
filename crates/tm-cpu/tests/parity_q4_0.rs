//! L1 / L1b kernel parity and batch-width invariance tests for `Q4_0 x Q8_0`.
//!
//! Follows `docs/TESTING.md`:
//! - Power-of-two scales: SIMD tiers must be **bit-for-bit identical** to `generic`.
//! - Arbitrary f16 scales: relative error <= 1e-5 against `generic`.
//! - Batch-width invariance: single-token vs 1..=8-token batching produces
//!   consistent results (bit-exact on power-of-two scales for all tiers, and
//!   bit-exact on all scales for `generic`, `neon`, `dotprod`).

use tm_core::{BlockQ4_0, BlockQ8_0, f32_to_f16};
use tm_cpu::isa::{Isa, supported_isas};
use tm_cpu::{ThreadPool, generic, select_kernels};

/// Simple deterministic xorshift64 PRNG (zero external dependencies).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }

    fn next_i8(&mut self) -> i8 {
        self.next_u8() as i8
    }

    fn pow2_scale(&mut self) -> u16 {
        // Exponents in [-6, -2] keep sums well inside exact f32 integer mantissa (24 bits).
        const POW2: [f32; 8] = [
            0.015625, -0.015625, 0.03125, -0.03125, 0.0625, -0.0625, 0.125, -0.125,
        ];
        let idx = (self.next_u64() as usize) % POW2.len();
        f32_to_f16(POW2[idx])
    }

    fn arbitrary_scale(&mut self) -> u16 {
        let r = ((self.next_u64() % 2001) as f32 - 1000.0) / 1000.0;
        f32_to_f16(r * 0.25)
    }
}

fn gen_q4_blocks(rng: &mut Rng, n: usize, pow2: bool) -> Vec<BlockQ4_0> {
    (0..n)
        .map(|i| {
            let mut qs = [0u8; 16];
            if i == 0 {
                // Guarantee all 16 nibble values (0..15) appear in both low and high nibbles.
                for (j, b) in qs.iter_mut().enumerate() {
                    *b = (j as u8) | (((15 - j) as u8) << 4);
                }
            } else {
                for b in &mut qs {
                    *b = rng.next_u8();
                }
            }
            BlockQ4_0 {
                d: if pow2 {
                    rng.pow2_scale()
                } else {
                    rng.arbitrary_scale()
                },
                qs,
            }
        })
        .collect()
}

fn gen_q8_blocks(rng: &mut Rng, n: usize, pow2: bool) -> Vec<BlockQ8_0> {
    (0..n)
        .map(|_| {
            let mut qs = [0i8; 32];
            for q in &mut qs {
                *q = rng.next_i8();
            }
            BlockQ8_0 {
                d: if pow2 {
                    rng.pow2_scale()
                } else {
                    rng.arbitrary_scale()
                },
                qs,
            }
        })
        .collect()
}

#[test]
fn l1_parity_pow2_bit_exact_all_isas() {
    let mut rng = Rng(0x5EED_CAFE_1234_5678);
    let isas = supported_isas();

    for &cols_blocks in &[1usize, 2, 3, 7, 16, 33, 64] {
        for &rows in &[1usize, 2, 3, 4, 5, 7, 12] {
            let w = gen_q4_blocks(&mut rng, rows * cols_blocks, true);
            let x = gen_q8_blocks(&mut rng, cols_blocks, true);

            let mut ref_out = vec![0.0f32; rows];
            generic::gemv_q4_0_q8_0(&w, &x, &mut ref_out, rows, cols_blocks);

            for &isa in &isas {
                let kt = select_kernels(isa);

                // vec_dot parity
                for r in 0..rows {
                    let row = &w[r * cols_blocks..(r + 1) * cols_blocks];
                    let got = (kt.vec_dot_q4_0_q8_0)(row, &x);
                    assert_eq!(
                        got.to_bits(),
                        ref_out[r].to_bits(),
                        "vec_dot bit mismatch: isa={} rows={} cols_blocks={} r={} got={} ref={}",
                        isa.name(),
                        rows,
                        cols_blocks,
                        r,
                        got,
                        ref_out[r]
                    );
                }

                // gemv parity
                let mut got_out = vec![0.0f32; rows];
                (kt.gemv_q4_0_q8_0)(&w, &x, &mut got_out, rows, cols_blocks);
                for r in 0..rows {
                    assert_eq!(
                        got_out[r].to_bits(),
                        ref_out[r].to_bits(),
                        "gemv bit mismatch: isa={} rows={} cols_blocks={} r={} got={} ref={}",
                        isa.name(),
                        rows,
                        cols_blocks,
                        r,
                        got_out[r],
                        ref_out[r]
                    );
                }
            }
        }
    }
}

#[test]
fn l1_parity_arbitrary_scales_within_1e5() {
    let mut rng = Rng(0xDEAD_BEEF_9988_7766);
    let isas = supported_isas();

    let rows = 19;
    let cols_blocks = 65; // 2080 elements (odd block count exercises tail path)
    let n_tokens = 5;

    let w = gen_q4_blocks(&mut rng, rows * cols_blocks, false);
    let x = gen_q8_blocks(&mut rng, n_tokens * cols_blocks, false);

    let mut ref_out = vec![0.0f32; n_tokens * rows];
    generic::gemm_q4_0_q8_0(&w, &x, &mut ref_out, rows, cols_blocks, n_tokens);

    // Compute L1 norm of block contributions for well-conditioned backward error bound
    // (random signed blocks cancel heavily, so dividing by |final_sum| amplifies 1-ULP noise).
    let mut l1_norms = vec![0.0f32; n_tokens * rows];
    for t in 0..n_tokens {
        for r in 0..rows {
            let mut acc = 0.0f32;
            for ib in 0..cols_blocks {
                let bx = &w[r * cols_blocks + ib];
                let by = &x[t * cols_blocks + ib];
                let mut sumi = 0i32;
                for j in 0..16 {
                    let v0 = ((bx.qs[j] & 0x0F) as i32) - 8;
                    let v1 = ((bx.qs[j] >> 4) as i32) - 8;
                    sumi += v0 * (by.qs[j] as i32) + v1 * (by.qs[j + 16] as i32);
                }
                acc +=
                    ((sumi as f32) * tm_core::f16_to_f32(bx.d) * tm_core::f16_to_f32(by.d)).abs();
            }
            l1_norms[t * rows + r] = acc.max(1.0);
        }
    }

    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got_out = vec![0.0f32; n_tokens * rows];
        (kt.gemm_q4_0_q8_0)(&w, &x, &mut got_out, rows, cols_blocks, n_tokens);

        for (i, (&got, &expected)) in got_out.iter().zip(ref_out.iter()).enumerate() {
            let rel = (got - expected).abs() / l1_norms[i];
            assert!(
                rel <= 1e-5,
                "gemm arbitrary-scale mismatch: isa={} idx={} got={} ref={} l1={} rel={}",
                isa.name(),
                i,
                got,
                expected,
                l1_norms[i],
                rel
            );
        }
    }
}

#[test]
fn l1b_batch_width_invariance() {
    let mut rng = Rng(0x1234_5678_ABCD_EF01);
    let isas = supported_isas();

    let rows = 11;
    let cols_blocks = 33;
    let max_tokens = 8;

    let w_pow2 = gen_q4_blocks(&mut rng, rows * cols_blocks, true);
    let x_pow2 = gen_q8_blocks(&mut rng, max_tokens * cols_blocks, true);

    let w_arb = gen_q4_blocks(&mut rng, rows * cols_blocks, false);
    let x_arb = gen_q8_blocks(&mut rng, max_tokens * cols_blocks, false);

    for &isa in &isas {
        let kt = select_kernels(isa);

        // Compute 1-token reference outputs for each token individually.
        let mut single_pow2 = vec![0.0f32; max_tokens * rows];
        let mut single_arb = vec![0.0f32; max_tokens * rows];
        for t in 0..max_tokens {
            (kt.gemv_q4_0_q8_0)(
                &w_pow2,
                &x_pow2[t * cols_blocks..(t + 1) * cols_blocks],
                &mut single_pow2[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
            (kt.gemv_q4_0_q8_0)(
                &w_arb,
                &x_arb[t * cols_blocks..(t + 1) * cols_blocks],
                &mut single_arb[t * rows..(t + 1) * rows],
                rows,
                cols_blocks,
            );
        }

        for n_tok in 1..=max_tokens {
            let mut batch_pow2 = vec![0.0f32; n_tok * rows];
            (kt.gemm_q4_0_q8_0)(
                &w_pow2,
                &x_pow2[..n_tok * cols_blocks],
                &mut batch_pow2,
                rows,
                cols_blocks,
                n_tok,
            );
            assert_eq!(
                batch_pow2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                single_pow2[..n_tok * rows]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "pow2 batch-width invariance failed for isa={} n_tok={}",
                isa.name(),
                n_tok
            );

            let mut batch_arb = vec![0.0f32; n_tok * rows];
            (kt.gemm_q4_0_q8_0)(
                &w_arb,
                &x_arb[..n_tok * cols_blocks],
                &mut batch_arb,
                rows,
                cols_blocks,
                n_tok,
            );
            if isa != Isa::NeonI8mm {
                // generic, neon_v80, dotprod preserve exact bit-level reduction order across batch widths.
                assert_eq!(
                    batch_arb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    single_arb[..n_tok * rows]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "arbitrary-scale batch-width invariance failed for isa={} n_tok={}",
                    isa.name(),
                    n_tok
                );
            }
        }
    }
}

#[test]
fn thread_pool_matches_single_thread_bit_exact() {
    let mut rng = Rng(0xCAFE_BABE_1122_3344);
    let rows = 256;
    let cols_blocks = 64;
    let w = gen_q4_blocks(&mut rng, rows * cols_blocks, false);
    let x = gen_q8_blocks(&mut rng, cols_blocks, false);

    let kt = tm_cpu::active_kernels();
    let mut single = vec![0.0f32; rows];
    (kt.gemv_q4_0_q8_0)(&w, &x, &mut single, rows, cols_blocks);

    let pool = ThreadPool::new(4);
    let mut parallel = vec![0.0f32; rows];
    let out_ptr = parallel.as_mut_ptr() as usize;

    pool.parallel_for(rows, 16, |r0, r1| {
        let len = r1 - r0;
        let w_slice = &w[r0 * cols_blocks..r1 * cols_blocks];
        // SAFETY: Disjoint row ranges `[r0..r1)` are owned exclusively by each chunk.
        let out_slice =
            unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
        (kt.gemv_q4_0_q8_0)(w_slice, &x, out_slice, len, cols_blocks);
    });

    assert_eq!(
        parallel.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        single.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn l1_parity_q4_1_q8_0_q5_k_q6_k_pow2_bit_exact() {
    use tm_core::{
        BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_1, BlockQ8K, DType, dequantize_row_bytes,
    };

    let mut rng = Rng(0x7777_8888_9999_AAAA);
    let isas = supported_isas();

    // 1. Q4_1 x Q8_1
    let nb32 = 33;
    let x41: Vec<BlockQ4_1> = (0..nb32)
        .map(|_| {
            let mut qs = [0u8; 16];
            for b in &mut qs {
                *b = rng.next_u8();
            }
            BlockQ4_1 {
                d: rng.pow2_scale(),
                m: rng.pow2_scale(),
                qs,
            }
        })
        .collect();
    let y81: Vec<BlockQ8_1> = (0..nb32)
        .map(|_| {
            let mut qs = [0i8; 32];
            for q in &mut qs {
                *q = rng.next_i8();
            }
            BlockQ8_1 {
                d: rng.pow2_scale(),
                s: rng.pow2_scale(),
                qs,
            }
        })
        .collect();
    let ref_41 = generic::vec_dot_q4_1_q8_1(&x41, &y81);

    // 2. Q8_0 x Q8_0 (explicitly include all `-128 * -128` block to guard against i16 overflow!)
    let mut x80 = gen_q8_blocks(&mut rng, nb32, true);
    let mut y80 = gen_q8_blocks(&mut rng, nb32, true);
    x80[0].qs = [i8::MIN; 32];
    y80[0].qs = [i8::MIN; 32];
    let ref_80 = generic::vec_dot_q8_0_q8_0(&x80, &y80);

    // 3. Q6_K x Q8_K
    let nb256 = 7;
    let x6k: Vec<BlockQ6K> = (0..nb256)
        .map(|_| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for b in &mut ql {
                *b = rng.next_u8();
            }
            for b in &mut qh {
                *b = rng.next_u8();
            }
            for s in &mut scales {
                *s = ((rng.next_u8() % 15) as i8) - 7;
            }
            BlockQ6K {
                ql,
                qh,
                scales,
                d: rng.pow2_scale(),
            }
        })
        .collect();
    let y8k: Vec<BlockQ8K> = (0..nb256)
        .map(|_| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for (j, bsum) in bsums.iter_mut().enumerate() {
                let mut s = 0i16;
                for l in 0..16 {
                    let q = ((rng.next_u8() % 63) as i8) - 31;
                    qs[j * 16 + l] = q;
                    s += q as i16;
                }
                *bsum = s;
            }
            BlockQ8K {
                d: 0.03125,
                qs,
                bsums,
            }
        })
        .collect();
    let ref_6k = generic::vec_dot_q6_k_q8_k(&x6k, &y8k);

    // 3b. Q5_K x Q8_K (and exact agreement with dequantize_row_bytes(DType::Q5K))
    let x5k: Vec<BlockQ5K> = (0..nb256)
        .map(|_| {
            let mut scales = [0u8; 12];
            let mut qh = [0u8; 32];
            let mut qs = [0u8; 128];
            for b in &mut scales {
                // Keep 6-bit unpacked scales/mins small (0..15) so integer dot fits comfortably in exact f32 mantissa
                *b = rng.next_u8() & 0x33;
            }
            for b in &mut qh {
                *b = rng.next_u8();
            }
            for b in &mut qs {
                *b = rng.next_u8();
            }
            BlockQ5K {
                d: rng.pow2_scale(),
                dmin: rng.pow2_scale(),
                scales,
                qh,
                qs,
            }
        })
        .collect();
    let ref_5k = generic::vec_dot_q5_k_q8_k(&x5k, &y8k);

    // Verify `dequantize_row_bytes(DType::Q5K)` matches `vec_dot_q5_k_q8_k` bit-for-bit on pow2 scales!
    // SAFETY: `BlockQ5K` is `#[repr(C)]` POD of size 176 bytes with no padding.
    let x5k_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(x5k.as_ptr() as *const u8, nb256 * 176) };
    let mut deq_5k = vec![0.0f32; nb256 * 256];
    dequantize_row_bytes(DType::Q5K, x5k_bytes, &mut deq_5k);
    let mut deq_dot = 0.0f32;
    for (ib, by) in y8k.iter().enumerate() {
        let mut block_sum = 0.0f32;
        for l in 0..256 {
            block_sum += deq_5k[ib * 256 + l] * (by.qs[l] as f32) * by.d;
        }
        deq_dot += block_sum;
    }
    assert_eq!(
        deq_dot.to_bits(),
        ref_5k.to_bits(),
        "Q5_K dequantize_row_bytes vs vec_dot_q5_k_q8_k mismatch: deq={deq_dot} vec_dot={ref_5k}"
    );

    for &isa in &isas {
        let kt = select_kernels(isa);
        assert_eq!(
            (kt.vec_dot_q4_1_q8_1)(&x41, &y81).to_bits(),
            ref_41.to_bits(),
            "Q4_1 bit mismatch for isa={}",
            isa.name()
        );
        assert_eq!(
            (kt.vec_dot_q8_0_q8_0)(&x80, &y80).to_bits(),
            ref_80.to_bits(),
            "Q8_0 bit mismatch for isa={}",
            isa.name()
        );
        assert_eq!(
            (kt.vec_dot_q5_k_q8_k)(&x5k, &y8k).to_bits(),
            ref_5k.to_bits(),
            "Q5_K bit mismatch for isa={}",
            isa.name()
        );
        assert_eq!(
            (kt.vec_dot_q6_k_q8_k)(&x6k, &y8k).to_bits(),
            ref_6k.to_bits(),
            "Q6_K bit mismatch for isa={}",
            isa.name()
        );
    }

    // 4. Multi-row GEMV parity for Q8_0 (rows=7), Q5_K (rows=5), and Q6_K (rows=5)
    let w80 = gen_q8_blocks(&mut rng, 7 * nb32, true);
    let mut ref_g80 = vec![0.0f32; 7];
    generic::gemv_q8_0_q8_0(&w80, &y80, &mut ref_g80, 7, nb32);

    let w5k: Vec<BlockQ5K> = (0..5 * nb256).map(|i| x5k[i % nb256]).collect();
    let mut ref_g5k = vec![0.0f32; 5];
    generic::gemv_q5_k_q8_k(&w5k, &y8k, &mut ref_g5k, 5, nb256);

    let w6k: Vec<BlockQ6K> = (0..5 * nb256).map(|i| x6k[i % nb256]).collect();
    let mut ref_g6k = vec![0.0f32; 5];
    generic::gemv_q6_k_q8_k(&w6k, &y8k, &mut ref_g6k, 5, nb256);

    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got_g80 = vec![0.0f32; 7];
        (kt.gemv_q8_0_q8_0)(&w80, &y80, &mut got_g80, 7, nb32);
        assert_eq!(
            got_g80.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_g80.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gemv Q8_0 bit mismatch for isa={}",
            isa.name()
        );

        let mut got_g5k = vec![0.0f32; 5];
        (kt.gemv_q5_k_q8_k)(&w5k, &y8k, &mut got_g5k, 5, nb256);
        assert_eq!(
            got_g5k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_g5k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gemv Q5_K bit mismatch for isa={}",
            isa.name()
        );

        let mut got_g6k = vec![0.0f32; 5];
        (kt.gemv_q6_k_q8_k)(&w6k, &y8k, &mut got_g6k, 5, nb256);
        assert_eq!(
            got_g6k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_g6k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gemv Q6_K bit mismatch for isa={}",
            isa.name()
        );
    }

    // 5. Multi-row GEMV parity for Q4_1 (rows=7 -> tests 4-row loop + 3-row tail)
    let w41: Vec<BlockQ4_1> = (0..7 * nb32).map(|i| x41[i % nb32]).collect();
    let mut ref_g41 = vec![0.0f32; 7];
    generic::gemv_q4_1_q8_1(&w41, &y81, &mut ref_g41, 7, nb32);
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got_g41 = vec![0.0f32; 7];
        (kt.gemv_q4_1_q8_1)(&w41, &y81, &mut got_g41, 7, nb32);
        assert_eq!(
            got_g41.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_g41.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gemv Q4_1 bit mismatch for isa={}",
            isa.name()
        );
    }

    // 6. Bit-exact parity for NEON quantize_q8_0, quantize_q8_1, f32_to_f16_slice,
    //    and power-of-two bit-exact parity for dot_f32_f16 & axpy_f32_f16
    let src_f32: Vec<f32> = (0..256).map(|i| ((i - 128) as f32) * 0.03125).collect();
    let mut ref_q80 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; 8];
    let mut ref_q81 = vec![
        BlockQ8_1 {
            d: 0,
            s: 0,
            qs: [0; 32]
        };
        8
    ];
    generic::quantize_q8_0(&src_f32, &mut ref_q80);
    generic::quantize_q8_1(&src_f32, &mut ref_q81);

    let mut ref_f16 = vec![0u16; 67];
    generic::f32_to_f16_slice(&src_f32[..67], &mut ref_f16);
    let ref_dot16 = generic::dot_f32_f16(&src_f32[..67], &ref_f16);
    let mut ref_axpy = vec![0.25f32; 67];
    generic::axpy_f32_f16(&mut ref_axpy, 0.5, &ref_f16);

    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got_q80 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; 8];
        let mut got_q81 = vec![
            BlockQ8_1 {
                d: 0,
                s: 0,
                qs: [0; 32]
            };
            8
        ];
        (kt.quantize_q8_0)(&src_f32, &mut got_q80);
        (kt.quantize_q8_1)(&src_f32, &mut got_q81);
        for b in 0..8 {
            assert_eq!(got_q80[b].d, ref_q80[b].d);
            assert_eq!(got_q80[b].qs, ref_q80[b].qs);
            assert_eq!(got_q81[b].d, ref_q81[b].d);
            assert_eq!(got_q81[b].s, ref_q81[b].s);
            assert_eq!(got_q81[b].qs, ref_q81[b].qs);
        }

        let mut got_f16 = vec![0u16; 67];
        (kt.f32_to_f16_slice)(&src_f32[..67], &mut got_f16);
        assert_eq!(
            got_f16,
            ref_f16,
            "f32_to_f16_slice mismatch for {}",
            isa.name()
        );

        let got_dot16 = (kt.dot_f32_f16)(&src_f32[..67], &got_f16);
        assert_eq!(
            got_dot16.to_bits(),
            ref_dot16.to_bits(),
            "dot_f32_f16 pow2 bit mismatch for {}",
            isa.name()
        );

        let mut got_axpy = vec![0.25f32; 67];
        (kt.axpy_f32_f16)(&mut got_axpy, 0.5, &got_f16);
        assert_eq!(
            got_axpy.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_axpy.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "axpy_f32_f16 pow2 bit mismatch for {}",
            isa.name()
        );
    }
}
