//! Milestone M4 L1 & L1b bit-exact parity tests (`0` ULP vs `generic` scalar reference)
//! for `Q2_0`, `TQ2_0` (16-row T-MAC `vqtbl1q` activation-LUT), `IQ2_XXS`, `IQ3_XXS`, and `IQ4_NL`.

use tm_core::{BlockQ8_0, DType, QK8_0, f32_to_f16, quantize_row_to_bytes};
use tm_cpu::isa::supported_isas;
use tm_cpu::{MatvecScratch, ThreadPool, generic, matmul_f32, matvec_f32, select_kernels};

fn make_f32_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..rows * cols)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s & 0xFFFF) as f32) / 32768.0 - 1.0;
            if i % 97 == 0 {
                u * 3.5
            } else if i % 53 == 0 {
                0.0
            } else {
                u * 0.65
            }
        })
        .collect()
}

fn quantize_matrix_aligned(
    dtype: DType,
    src: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<u8>, usize, usize) {
    let rb = dtype.row_bytes(cols);
    let total = rows * rb;
    let mut backing = vec![0u8; total + 32];
    let off = (32 - (backing.as_ptr() as usize % 32)) % 32;
    let dst = &mut backing[off..off + total];
    for r in 0..rows {
        quantize_row_to_bytes(
            dtype,
            &src[r * cols..(r + 1) * cols],
            &mut dst[r * rb..(r + 1) * rb],
        );
    }
    (backing, off, total)
}

#[test]
fn l1_lowbit_lut_and_iq_kernels_bit_exact_across_all_isas() {
    let isas = supported_isas();
    assert!(!isas.is_empty());

    // Include row counts that exercise:
    // - pure tail (`1`, `15`),
    // - exact 16-row T-MAC LUT groups (`16`, `32`),
    // - mixed 16-row T-MAC LUT + tail (`17`, `35`),
    // and column counts crossing `K_TILE` boundaries (`cols = 512` and `cols = 1280` > `K_TILE=4*256=1024`).
    let shapes = [
        (1usize, 256usize),
        (15, 512),
        (16, 768),
        (17, 1280),
        (35, 512),
    ];
    let dtypes = [
        DType::Q2_0,
        DType::TQ2_0,
        DType::Iq2Xxs,
        DType::Iq3Xxs,
        DType::Iq4Nl,
    ];

    let pool1 = ThreadPool::new(1);
    let pool4 = ThreadPool::new(4);
    let kt_ref = select_kernels(tm_cpu::Isa::Generic);

    for &dtype in &dtypes {
        for &(rows, cols) in &shapes {
            let w_f32 = make_f32_matrix(
                rows,
                cols,
                0xCAFE_BABE_1234_5678 ^ (dtype.to_ggml_type() as u64),
            );
            let x_f32 = make_f32_matrix(1, cols, 0x9E37_79B9_7F4A_7C15 ^ (cols as u64));
            let (backing, off, total_bytes) = quantize_matrix_aligned(dtype, &w_f32, rows, cols);
            let w_bytes = &backing[off..off + total_bytes];

            let mut scratch_ref = MatvecScratch::new(cols);
            let mut out_ref = vec![0.0f32; rows];
            matvec_f32(
                &pool1,
                kt_ref,
                dtype,
                w_bytes,
                &x_f32,
                &mut scratch_ref,
                &mut out_ref,
                rows,
                cols,
            );

            for &isa in &isas {
                let kt = select_kernels(isa);
                for pool in [&pool1, &pool4] {
                    let mut scratch = MatvecScratch::new(cols);
                    let mut out_got = vec![0.0f32; rows];
                    matvec_f32(
                        pool,
                        kt,
                        dtype,
                        w_bytes,
                        &x_f32,
                        &mut scratch,
                        &mut out_got,
                        rows,
                        cols,
                    );
                    for r in 0..rows {
                        assert_eq!(
                            out_got[r].to_bits(),
                            out_ref[r].to_bits(),
                            "L1 0-ULP mismatch for dtype={} isa={} thr={} shape={}x{} row={}: got={} (0x{:08x}) != ref={} (0x{:08x})",
                            dtype.as_str(),
                            isa.name(),
                            pool.num_threads(),
                            rows,
                            cols,
                            r,
                            out_got[r],
                            out_got[r].to_bits(),
                            out_ref[r],
                            out_ref[r].to_bits(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn l1_extreme_i8_min_and_negative_scales_bit_exact() {
    // Specifically stress `i8::MIN = -128` activations where `q0 = +2, q1 = +2` gives `val = -512`
    // (`hi = -2, lo = 0`) and `q0 = -1, q1 = -1` gives `val = +256` (`hi = +1, lo = 0`)!
    let isas = supported_isas();
    let cols_blocks = 3usize;
    let rows = 19usize;
    let d_neg = f32_to_f16(-0.03125);
    let d_pos = f32_to_f16(0.0625);

    let w_q20: Vec<tm_core::BlockQ2_0> = (0..rows * cols_blocks)
        .map(|i| tm_core::BlockQ2_0 {
            d: if i % 2 == 0 { d_neg } else { d_pos },
            qs: [if i % 3 == 0 {
                0xFF
            } else if i % 3 == 1 {
                0x00
            } else {
                0xE4
            }; 16],
        })
        .collect();
    let x_q80: Vec<BlockQ8_0> = (0..cols_blocks * 2)
        .map(|i| BlockQ8_0 {
            d: if i % 2 == 0 { d_pos } else { d_neg },
            qs: [if i % 2 == 0 { i8::MIN } else { i8::MAX }; QK8_0],
        })
        .collect();

    let mut ref_out = vec![0.0f32; rows];
    generic::gemv_q2_0_q8_0(&w_q20, &x_q80, &mut ref_out, rows, cols_blocks);

    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut got = vec![0.0f32; rows];
        (kt.gemv_q2_0_q8_0)(&w_q20, &x_q80, &mut got, rows, cols_blocks);
        for r in 0..rows {
            assert_eq!(
                got[r].to_bits(),
                ref_out[r].to_bits(),
                "Extreme i8::MIN Q2_0 LUT mismatch on isa={} row={}",
                isa.name(),
                r
            );
        }
    }
}

#[test]
fn l1b_batch_width_invariance_m4_formats() {
    let isas = supported_isas();
    let rows = 35usize;
    let cols = 512usize;
    let batch_sizes = [1usize, 2, 7, 16, 33, 64];
    let x_all = make_f32_matrix(64, cols, 0xABCD_EF01_2345_6789);
    let pool4 = ThreadPool::new(4);

    for &dtype in &[
        DType::Q2_0,
        DType::TQ2_0,
        DType::Iq2Xxs,
        DType::Iq3Xxs,
        DType::Iq4Nl,
    ] {
        let w_f32 = make_f32_matrix(
            rows,
            cols,
            0x5151_5151_A2A2_A2A2 ^ (dtype.to_ggml_type() as u64),
        );
        let (backing, off, total_bytes) = quantize_matrix_aligned(dtype, &w_f32, rows, cols);
        let w_bytes = &backing[off..off + total_bytes];

        let kt_ref = select_kernels(tm_cpu::Isa::Generic);
        let mut single_ref = vec![0.0f32; 64 * rows];
        let mut scratch = MatvecScratch::new(cols);
        for t in 0..64 {
            matvec_f32(
                &pool4,
                kt_ref,
                dtype,
                w_bytes,
                &x_all[t * cols..(t + 1) * cols],
                &mut scratch,
                &mut single_ref[t * rows..(t + 1) * rows],
                rows,
                cols,
            );
        }

        for &isa in &isas {
            let kt = select_kernels(isa);
            for &b in &batch_sizes {
                let mut out_batch = vec![0.0f32; b * rows];
                matmul_f32(
                    &pool4,
                    kt,
                    dtype,
                    w_bytes,
                    &x_all[..b * cols],
                    &mut scratch,
                    &mut out_batch,
                    rows,
                    cols,
                    b,
                );
                for i in 0..b * rows {
                    assert_eq!(
                        out_batch[i].to_bits(),
                        single_ref[i].to_bits(),
                        "L1b batch-width invariance failed for dtype={} isa={} batch={} idx={}",
                        dtype.as_str(),
                        isa.name(),
                        b,
                        i
                    );
                }
            }
        }
    }
}
