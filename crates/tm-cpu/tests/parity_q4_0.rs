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
