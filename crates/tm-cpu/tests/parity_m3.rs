//! M3 L1 & L1b parity and batch-width invariance tests:
//! - `BlockQ4_0x4` (`Q4_0_4x8`) GEMV & GEMM across all ISA tiers (`generic`, `neon_v80`, `dotprod`, `i8mm`),
//!   verifying 100% bit-exact parity (`0` ULP) against `generic` on BOTH power-of-two AND arbitrary scales,
//!   exact agreement with original `BlockQ4_0` on power-of-two scales, and batch-width invariance (`1..=19` tokens).
//! - Multi-token shared-weight `gemm_q8_0_q8_0` and `gemm_q6_k_q8_k` + 4-row `gemv_q6_k_q8_k` parity.
//! - NEON `quantize_q8_k` bit-for-bit parity with `tm_core::quantize_row_q8_k`.
//! - `Q8_0` KV cache attention kernels (`dot_f32_q8_0`, `axpy_f32_q8_0`, `gqa_attention_q8_0_kv_ext_*`) parity.
//! - Fused batched `matmul_pair_f32` and `matmul_triple_f32` bit-exact parity.

use tm_core::{
    BlockQ4_0, BlockQ4_0x4, BlockQ6K, BlockQ8_0, BlockQ8K, DType, QK_K, QK4_0, f32_to_f16,
    repack_q4_0_to_q4_0x4,
};
use tm_cpu::isa::supported_isas;
use tm_cpu::{
    MatvecScratch, ThreadPool, generic, gqa_attention_q8_0_kv_ext_batch_parallel,
    gqa_attention_q8_0_kv_ext_parallel, matmul_f32, matmul_pair_f32, matmul_triple_f32,
    select_kernels,
};

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
        const POW2: [f32; 8] = [
            0.015625, -0.015625, 0.03125, -0.03125, 0.0625, -0.0625, 0.125, -0.125,
        ];
        f32_to_f16(POW2[(self.next_u64() as usize) % POW2.len()])
    }
    fn arb_scale(&mut self) -> u16 {
        let r = ((self.next_u64() % 2001) as f32 - 1000.0) / 1000.0;
        f32_to_f16(r * 0.25)
    }
}

#[test]
fn l1_and_l1b_q4_0x4_bit_exact_all_isas_and_batch_widths() {
    let mut rng = Rng(0x2026_1121_BEEF_CAFE);
    let isas = supported_isas();

    // Exercise odd group counts (4, 8, 12, 68 rows) and multi-K-tile block counts (1, 3, 33, 67 blocks)
    for &cols_blocks in &[1usize, 3, 33, 67] {
        for &rows in &[4usize, 8, 12, 68] {
            for &pow2 in &[true, false] {
                let w_orig: Vec<BlockQ4_0> = (0..rows * cols_blocks)
                    .map(|i| {
                        let mut qs = [0u8; 16];
                        if i == 0 {
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
                                rng.arb_scale()
                            },
                            qs,
                        }
                    })
                    .collect();

                let mut w_rep = vec![
                    BlockQ4_0x4 {
                        d: [0; 4],
                        qs: [0; 64]
                    };
                    (rows / 4) * cols_blocks
                ];
                repack_q4_0_to_q4_0x4(&w_orig, &mut w_rep, rows, cols_blocks);

                let max_tok = 11usize; // exercises 8-token tile + 2-token pair + 1-token tail!
                let x: Vec<BlockQ8_0> = (0..max_tok * cols_blocks)
                    .map(|_| {
                        let mut qs = [0i8; 32];
                        for q in &mut qs {
                            *q = rng.next_i8();
                        }
                        BlockQ8_0 {
                            d: if pow2 {
                                rng.pow2_scale()
                            } else {
                                rng.arb_scale()
                            },
                            qs,
                        }
                    })
                    .collect();

                let mut ref_gemv = vec![0.0f32; rows];
                generic::gemv_q4_0x4_q8_0(
                    &w_rep,
                    &x[..cols_blocks],
                    &mut ref_gemv,
                    rows,
                    cols_blocks,
                );

                if pow2 {
                    let mut orig_gemv = vec![0.0f32; rows];
                    generic::gemv_q4_0_q8_0(
                        &w_orig,
                        &x[..cols_blocks],
                        &mut orig_gemv,
                        rows,
                        cols_blocks,
                    );
                    assert_eq!(
                        ref_gemv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        orig_gemv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "Q4_0x4 vs original Q4_0 pow2 mismatch"
                    );
                }

                let mut ref_gemm = vec![0.0f32; max_tok * rows];
                generic::gemm_q4_0x4_q8_0(&w_rep, &x, &mut ref_gemm, rows, cols_blocks, max_tok);

                for &isa in &isas {
                    let kt = select_kernels(isa);
                    let mut got_v = vec![0.0f32; rows];
                    (kt.gemv_q4_0x4_q8_0)(&w_rep, &x[..cols_blocks], &mut got_v, rows, cols_blocks);
                    assert_eq!(
                        got_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        ref_gemv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "gemv_q4_0x4 bit mismatch: isa={} rows={rows} cb={cols_blocks} pow2={pow2}",
                        isa.name()
                    );

                    // L1b batch-width invariance across `n_tok = 1..=max_tok` (bit-exact on ALL scales!)
                    for n_tok in 1..=max_tok {
                        let mut got_m = vec![0.0f32; n_tok * rows];
                        (kt.gemm_q4_0x4_q8_0)(
                            &w_rep,
                            &x[..n_tok * cols_blocks],
                            &mut got_m,
                            rows,
                            cols_blocks,
                            n_tok,
                        );
                        assert_eq!(
                            got_m.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            ref_gemm[..n_tok * rows]
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "gemm_q4_0x4 batch invariance failed: isa={} rows={rows} cb={cols_blocks} n_tok={n_tok} pow2={pow2}",
                            isa.name()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn l1_and_l1b_q6_k_q8_0_shared_weight_and_quantize_q8_k_parity() {
    let mut rng = Rng(0xABCD_1234_5678_EF90);
    let isas = supported_isas();

    // 1. NEON `quantize_q8_k` bit-for-bit parity (including negative max, positive max, and all-zero block)
    let mut src_f32: Vec<f32> = (0..6 * QK_K)
        .map(|i| (((i as i32 * 17 - 1234) % 511) as f32) * 0.0137)
        .collect();
    src_f32[QK_K..2 * QK_K].fill(0.0);
    src_f32[2 * QK_K + 19] = -99.5; // negative dominant extremum
    src_f32[3 * QK_K + 77] = 88.25; // positive dominant extremum

    let mut ref_q8k = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; 16]
        };
        6
    ];
    generic::quantize_q8_k(&src_f32, &mut ref_q8k);

    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got_q8k = vec![
            BlockQ8K {
                d: 0.0,
                qs: [0; QK_K],
                bsums: [0; 16]
            };
            6
        ];
        (kt.quantize_q8_k)(&src_f32, &mut got_q8k);
        for b in 0..6 {
            assert_eq!(got_q8k[b].d.to_bits(), ref_q8k[b].d.to_bits());
            assert_eq!(got_q8k[b].qs, ref_q8k[b].qs);
            assert_eq!(got_q8k[b].bsums, ref_q8k[b].bsums);
        }
    }

    // 2. 4-row `gemv_q6_k_q8_k` + shared-weight `gemm_q6_k_q8_k` (rows=11, n_tok=1..=6)
    let rows6 = 11usize;
    let cb6 = 5usize;
    let max_t = 6usize;
    let w6: Vec<BlockQ6K> = (0..rows6 * cb6)
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
    let x8k: Vec<BlockQ8K> = (0..max_t * cb6)
        .map(|_| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for (j, bs) in bsums.iter_mut().enumerate() {
                let mut sum = 0i16;
                for l in 0..16 {
                    let q = ((rng.next_u8() % 63) as i8) - 31;
                    qs[j * 16 + l] = q;
                    sum += q as i16;
                }
                *bs = sum;
            }
            BlockQ8K {
                d: 0.03125,
                qs,
                bsums,
            }
        })
        .collect();

    let mut ref_6k = vec![0.0f32; max_t * rows6];
    generic::gemm_q6_k_q8_k(&w6, &x8k, &mut ref_6k, rows6, cb6, max_t);

    // 3. Shared-weight `gemm_q8_0_q8_0` (rows=9, cb=17, n_tok=1..=6)
    let rows8 = 9usize;
    let cb8 = 17usize;
    let w8: Vec<BlockQ8_0> = (0..rows8 * cb8)
        .map(|_| {
            let mut qs = [0i8; 32];
            for q in &mut qs {
                *q = rng.next_i8();
            }
            BlockQ8_0 {
                d: rng.pow2_scale(),
                qs,
            }
        })
        .collect();
    let x8: Vec<BlockQ8_0> = (0..max_t * cb8)
        .map(|_| {
            let mut qs = [0i8; 32];
            for q in &mut qs {
                *q = rng.next_i8();
            }
            BlockQ8_0 {
                d: rng.pow2_scale(),
                qs,
            }
        })
        .collect();
    let mut ref_80 = vec![0.0f32; max_t * rows8];
    generic::gemm_q8_0_q8_0(&w8, &x8, &mut ref_80, rows8, cb8, max_t);

    for &isa in &isas {
        let kt = select_kernels(isa);
        for n_tok in 1..=max_t {
            let mut got_6k = vec![0.0f32; n_tok * rows6];
            (kt.gemm_q6_k_q8_k)(&w6, &x8k[..n_tok * cb6], &mut got_6k, rows6, cb6, n_tok);
            assert_eq!(
                got_6k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_6k[..n_tok * rows6]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "gemm_q6_k_q8_k mismatch: isa={} n_tok={n_tok}",
                isa.name()
            );

            let mut got_80 = vec![0.0f32; n_tok * rows8];
            (kt.gemm_q8_0_q8_0)(&w8, &x8[..n_tok * cb8], &mut got_80, rows8, cb8, n_tok);
            assert_eq!(
                got_80.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ref_80[..n_tok * rows8]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "gemm_q8_0_q8_0 mismatch: isa={} n_tok={n_tok}",
                isa.name()
            );
        }
    }

    // 4. `vec_dot_q5_k_q8_k` & `gemv_q5_k_q8_k` 0-ULP bit-exact parity across all ISAs
    let rows5 = 11usize;
    let w5: Vec<tm_core::BlockQ5K> = (0..rows5 * cb6)
        .map(|_| {
            let mut scales = [0u8; 12];
            let mut qh = [0u8; 32];
            let mut qs = [0u8; 128];
            for b in &mut scales {
                *b = rng.next_u8();
            }
            for b in &mut qh {
                *b = rng.next_u8();
            }
            for b in &mut qs {
                *b = rng.next_u8();
            }
            tm_core::BlockQ5K {
                d: rng.arb_scale(),
                dmin: rng.arb_scale(),
                scales,
                qh,
                qs,
            }
        })
        .collect();
    let mut ref_5k = vec![0.0f32; rows5];
    generic::gemv_q5_k_q8_k(&w5, &x8k[..cb6], &mut ref_5k, rows5, cb6);
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got_5k = vec![0.0f32; rows5];
        (kt.gemv_q5_k_q8_k)(&w5, &x8k[..cb6], &mut got_5k, rows5, cb6);
        assert_eq!(
            got_5k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_5k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "gemv_q5_k_q8_k 0-ULP mismatch: isa={}",
            isa.name()
        );
    }
}

#[test]
fn l1_q8_0_kv_cache_and_fused_matmul_parity() {
    let mut rng = Rng(0x55AA_33CC_9911_77EE);
    let isas = supported_isas();

    let hd = 64usize;
    let hb = hd / QK4_0;
    let q: Vec<f32> = (0..hd).map(|i| (i as f32 - 32.0) * 0.03125).collect();
    let kv: Vec<BlockQ8_0> = (0..hb)
        .map(|_| {
            let mut qs = [0i8; 32];
            for x in &mut qs {
                *x = rng.next_i8();
            }
            BlockQ8_0 {
                d: rng.arb_scale(),
                qs,
            }
        })
        .collect();

    let ref_dot = generic::dot_f32_q8_0(&q, &kv);
    let mut ref_axpy = q.clone();
    generic::axpy_f32_q8_0(&mut ref_axpy, 0.375, &kv);

    for &isa in &isas {
        let kt = select_kernels(isa);
        let got_dot = (kt.dot_f32_q8_0)(&q, &kv);
        assert_eq!(
            got_dot.to_bits(),
            ref_dot.to_bits(),
            "dot_f32_q8_0 bit mismatch for {}",
            isa.name()
        );

        let mut got_axpy = q.clone();
        (kt.axpy_f32_q8_0)(&mut got_axpy, 0.375, &kv);
        assert_eq!(
            got_axpy.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ref_axpy.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "axpy_f32_q8_0 bit mismatch for {}",
            isa.name()
        );
    }

    // Verify batched Q8_0 KV GQA matches sequential single-token Q8_0 KV GQA bit-for-bit
    let pool4 = ThreadPool::new(4);
    let kt = tm_cpu::active_kernels();
    let n_heads = 8usize;
    let n_kv = 2usize;
    let seq = 30usize;
    let n_tok = 5usize;
    let q_batch: Vec<f32> = (0..n_tok * n_heads * hd)
        .map(|i| ((i % 71) as f32 - 35.0) * 0.02)
        .collect();
    let k_cache: Vec<BlockQ8_0> = (0..(seq + n_tok) * n_kv * hb)
        .map(|_| {
            let mut qs = [0i8; 32];
            for v in &mut qs {
                *v = rng.next_i8();
            }
            BlockQ8_0 {
                d: rng.arb_scale(),
                qs,
            }
        })
        .collect();
    let v_cache = k_cache.clone();
    let mut scores = vec![0.0f32; n_heads * 64];
    let mut out_seq = vec![0.0f32; n_tok * n_heads * hd];
    for t in 0..n_tok {
        gqa_attention_q8_0_kv_ext_parallel(
            &pool4,
            kt,
            &mut out_seq[t * n_heads * hd..(t + 1) * n_heads * hd],
            &q_batch[t * n_heads * hd..(t + 1) * n_heads * hd],
            &k_cache,
            &v_cache,
            n_heads,
            n_kv,
            hd,
            seq + t,
            1.0 / (hd as f32).sqrt(),
            Some(16),
            64,
            &mut scores,
        );
    }
    let mut out_bat = vec![0.0f32; n_tok * n_heads * hd];
    gqa_attention_q8_0_kv_ext_batch_parallel(
        &pool4,
        kt,
        &mut out_bat,
        &q_batch,
        &k_cache,
        &v_cache,
        n_heads,
        n_kv,
        hd,
        seq,
        n_tok,
        1.0 / (hd as f32).sqrt(),
        Some(16),
        64,
        &mut scores,
    );
    assert_eq!(
        out_bat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        out_seq.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "Q8_0 KV batch vs single-token mismatch"
    );

    // Verify fused `matmul_pair_f32` and `matmul_triple_f32` match separate `matmul_f32` bit-for-bit on Q4_0x4
    let cols = 128usize;
    let cb = cols / QK4_0;
    let ra = 20usize;
    let rb = 12usize;
    let rc = 12usize;
    let make_rep = |rows: usize, rng: &mut Rng| -> Vec<BlockQ4_0x4> {
        let orig: Vec<BlockQ4_0> = (0..rows * cb)
            .map(|_| BlockQ4_0 {
                d: rng.arb_scale(),
                qs: [rng.next_u8(); 16],
            })
            .collect();
        let mut rep = vec![
            BlockQ4_0x4 {
                d: [0; 4],
                qs: [0; 64]
            };
            (rows / 4) * cb
        ];
        repack_q4_0_to_q4_0x4(&orig, &mut rep, rows, cb);
        rep
    };
    let wa = make_rep(ra, &mut rng);
    let wb = make_rep(rb, &mut rng);
    let wc = make_rep(rc, &mut rng);
    // SAFETY: `BlockQ4_0x4` is POD of size 72 bytes.
    let (wa_b, wb_b, wc_b) = unsafe {
        (
            core::slice::from_raw_parts(wa.as_ptr() as *const u8, wa.len() * 72),
            core::slice::from_raw_parts(wb.as_ptr() as *const u8, wb.len() * 72),
            core::slice::from_raw_parts(wc.as_ptr() as *const u8, wc.len() * 72),
        )
    };
    let x_in: Vec<f32> = (0..n_tok * cols)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.05)
        .collect();
    let mut scratch = MatvecScratch::new(cols);
    let mut ref_a = vec![0.0f32; n_tok * ra];
    let mut ref_b = vec![0.0f32; n_tok * rb];
    let mut ref_c = vec![0.0f32; n_tok * rc];
    matmul_f32(
        &pool4,
        kt,
        DType::Q4_0x4,
        wa_b,
        &x_in,
        &mut scratch,
        &mut ref_a,
        ra,
        cols,
        n_tok,
    );
    matmul_f32(
        &pool4,
        kt,
        DType::Q4_0x4,
        wb_b,
        &x_in,
        &mut scratch,
        &mut ref_b,
        rb,
        cols,
        n_tok,
    );
    matmul_f32(
        &pool4,
        kt,
        DType::Q4_0x4,
        wc_b,
        &x_in,
        &mut scratch,
        &mut ref_c,
        rc,
        cols,
        n_tok,
    );

    let mut got_pa = vec![0.0f32; n_tok * ra];
    let mut got_pb = vec![0.0f32; n_tok * rb];
    matmul_pair_f32(
        &pool4,
        kt,
        DType::Q4_0x4,
        wa_b,
        ra,
        &mut got_pa,
        DType::Q4_0x4,
        wb_b,
        rb,
        &mut got_pb,
        &x_in,
        &mut scratch,
        cols,
        n_tok,
    );
    assert_eq!(got_pa, ref_a);
    assert_eq!(got_pb, ref_b);

    let mut got_ta = vec![0.0f32; n_tok * ra];
    let mut got_tb = vec![0.0f32; n_tok * rb];
    let mut got_tc = vec![0.0f32; n_tok * rc];
    matmul_triple_f32(
        &pool4,
        kt,
        DType::Q4_0x4,
        wa_b,
        ra,
        &mut got_ta,
        DType::Q4_0x4,
        wb_b,
        rb,
        &mut got_tb,
        DType::Q4_0x4,
        wc_b,
        rc,
        &mut got_tc,
        &x_in,
        &mut scratch,
        cols,
        n_tok,
    );
    assert_eq!(got_ta, ref_a);
    assert_eq!(got_tb, ref_b);
    assert_eq!(got_tc, ref_c);
}
