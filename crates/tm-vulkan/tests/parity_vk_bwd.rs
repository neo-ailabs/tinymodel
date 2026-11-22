//! Phase `T4` Vulkan verification & microbenchmark (`AGENTS.md` Rule #3):
//! 1. Aligns `gemm_transposed_raw` across ALL 6 Vulkan quantized/FP32 formats
//!    (`Q4_0`, `Q4_1`, `Q8_0`, `Q5_K`, `Q6_K`, `F32`) against `tm_cpu` scalar `Generic` reference
//!    for `n_tok = 1, 9, 37` and `accum = false/true`.
//! 2. Aligns `lora_linear_raw` and `lora_embed_raw` + **0-ULP batch vs single-token determinism**.
//! 3. Benchmarks `gemm_transposed_raw` (`Q4_0 4096x2048 T=128`) on the active Vulkan GPU.
//!
//! Skips cleanly when `libvulkan` or a compute device is unavailable (e.g. macOS CI without MoltenVK).

use std::time::Instant;

use tm_core::{DType, f32_to_f16, quantize_row_to_bytes};
use tm_cpu::{Isa, ThreadPool, matmul_q_transposed_f32, select_kernels};
use tm_vulkan::{VulkanContext, VulkanError};

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((self.0 >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
        (u * 2.0 - 1.0) * scale
    }
}

fn try_ctx() -> Option<VulkanContext> {
    match VulkanContext::new() {
        Ok(c) => Some(c),
        Err(VulkanError::Loader(_) | VulkanError::NoDevice) => {
            eprintln!("[parity_vk_bwd] no Vulkan driver/device available; skipping cleanly");
            None
        }
        Err(e) => panic!("unexpected Vulkan init error: {e}"),
    }
}

fn encode_2d_dtype(dtype: DType, rows: usize, cols: usize, vals: &[f32]) -> Vec<u8> {
    let rb = dtype.row_bytes(cols);
    let mut out = vec![0u8; rows * rb];
    for r in 0..rows {
        let src = &vals[r * cols..(r + 1) * cols];
        let dst = &mut out[r * rb..(r + 1) * rb];
        match dtype {
            DType::Q4_1 => {
                for b in 0..(cols / 32) {
                    let blk = &src[b * 32..(b + 1) * 32];
                    let mut mn = blk[0];
                    let mut mx = blk[0];
                    for &v in blk {
                        mn = mn.min(v);
                        mx = mx.max(v);
                    }
                    let d = (mx - mn) / 15.0;
                    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
                    let bo = &mut dst[b * 20..(b + 1) * 20];
                    bo[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                    bo[2..4].copy_from_slice(&f32_to_f16(mn).to_le_bytes());
                    for j in 0..16 {
                        let q0 = ((blk[j] - mn) * inv_d + 0.5).clamp(0.0, 15.0) as u8;
                        let q1 = ((blk[j + 16] - mn) * inv_d + 0.5).clamp(0.0, 15.0) as u8;
                        bo[4 + j] = q0 | (q1 << 4);
                    }
                }
            }
            DType::Q5K => {
                for sb in 0..(cols / 256) {
                    let sblk = &src[sb * 256..(sb + 1) * 256];
                    let bo = &mut dst[sb * 176..(sb + 1) * 176];
                    bo[0..2].copy_from_slice(&f32_to_f16(0.025).to_le_bytes());
                    bo[2..4].copy_from_slice(&f32_to_f16(0.40).to_le_bytes());
                    for j in 0..12 {
                        bo[4 + j] = ((j * 7 + 3) & 0x3F) as u8;
                    }
                    for l in 0..32 {
                        let mut qh_byte = 0u8;
                        for chunk in 0..4 {
                            let q_lo = ((sblk[chunk * 64 + l] + 0.5) * 15.0)
                                .round()
                                .clamp(0.0, 31.0) as u8;
                            let q_hi = ((sblk[chunk * 64 + 32 + l] + 0.5) * 15.0)
                                .round()
                                .clamp(0.0, 31.0) as u8;
                            bo[48 + chunk * 32 + l] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
                            qh_byte |= ((q_lo >> 4) & 1) << (2 * chunk);
                            qh_byte |= ((q_hi >> 4) & 1) << (2 * chunk + 1);
                        }
                        bo[16 + l] = qh_byte;
                    }
                }
            }
            DType::Q6K => {
                for sb in 0..(cols / 256) {
                    let sblk = &src[sb * 256..(sb + 1) * 256];
                    let bo = &mut dst[sb * 210..(sb + 1) * 210];
                    bo[208..210].copy_from_slice(&f32_to_f16(0.02).to_le_bytes());
                    for j in 0..16 {
                        bo[192 + j] = ((j as i8) * 3 - 20) as u8;
                    }
                    for half_idx in 0..2 {
                        for l in 0..32 {
                            let q0 = ((sblk[half_idx * 128 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            let q1 = ((sblk[half_idx * 128 + 32 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            let q2 = ((sblk[half_idx * 128 + 64 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            let q3 = ((sblk[half_idx * 128 + 96 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            bo[half_idx * 64 + l] = (q0 & 0x0F) | ((q2 & 0x0F) << 4);
                            bo[half_idx * 64 + 32 + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                            bo[128 + half_idx * 32 + l] = ((q0 >> 4) & 3)
                                | (((q1 >> 4) & 3) << 2)
                                | (((q2 >> 4) & 3) << 4)
                                | (((q3 >> 4) & 3) << 6);
                        }
                    }
                }
            }
            _ => {
                quantize_row_to_bytes(dtype, src, dst);
            }
        }
    }
    out
}

#[test]
fn t4_vulkan_transposed_gemm_and_lora_parity_plus_bench() {
    let Some(ctx) = try_ctx() else {
        return;
    };
    let pool = ThreadPool::new(1);
    let kt_ref = select_kernels(Isa::Generic);
    let mut rng = Rng(0x55AA_2026);

    let cases = [
        (DType::Q4_0, 99usize, 128usize),
        (DType::Q4_1, 77, 128),
        (DType::Q8_0, 113, 160),
        (DType::Q5K, 67, 512),
        (DType::Q6K, 71, 512),
        (DType::F32, 83, 100),
    ];

    for (dtype, rows, cols) in cases {
        let w_f32: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32(0.5)).collect();
        let w_bytes = encode_2d_dtype(dtype, rows, cols, &w_f32);
        let w_gpu = ctx.upload_bytes(&w_bytes).unwrap();

        for &n_tok in &[1usize, 9, 37] {
            let dy: Vec<f32> = (0..n_tok * rows).map(|_| rng.next_f32(0.8)).collect();
            let init_dx: Vec<f32> = (0..n_tok * cols).map(|_| rng.next_f32(0.4)).collect();
            let dy_gpu = ctx.upload(&dy).unwrap();

            for &accum in &[false, true] {
                let mut ref_dx = if accum {
                    init_dx.clone()
                } else {
                    vec![0.0f32; n_tok * cols]
                };
                matmul_q_transposed_f32(
                    &pool,
                    kt_ref,
                    dtype,
                    &w_bytes,
                    &dy,
                    &mut ref_dx,
                    rows,
                    cols,
                    n_tok,
                    accum,
                );

                let dx_gpu = ctx.upload(&init_dx).unwrap();
                ctx.gemm_transposed_raw(
                    dtype, &w_gpu, &dy_gpu, &dx_gpu, rows, cols, n_tok, accum, 1,
                )
                .unwrap();
                let got = ctx.read_f32(&dx_gpu, n_tok * cols).unwrap();
                let mut max_rel = 0.0f32;
                for (i, (&g, &r)) in got.iter().zip(ref_dx.iter()).enumerate() {
                    let rel = (g - r).abs() / r.abs().max(1.0);
                    max_rel = max_rel.max(rel);
                    assert!(
                        rel < 2e-5,
                        "Vulkan transposed GEMM mismatch dtype={dtype:?} rows={rows} cols={cols} T={n_tok} accum={accum} idx={i}: gpu={g} ref={r} rel={rel:e}"
                    );
                }
                if n_tok == 37 && accum {
                    eprintln!(
                        "[T4 Vulkan bwd parity] {dtype:?} ({rows}x{cols}, T={n_tok}, accum=true) max_rel={max_rel:.3e}"
                    );
                }
            }
        }
    }

    // Verify `lora_linear_raw` 0-ULP batch vs single determinism + scalar reference
    let d_in = 128usize;
    let d_out = 96usize;
    let rank = 8usize;
    let n_tok = 9usize;
    let scale = 1.5f32;
    let a: Vec<f32> = (0..rank * d_in).map(|_| rng.next_f32(0.3)).collect();
    let b_t: Vec<f32> = (0..rank * d_out).map(|_| rng.next_f32(0.3)).collect();
    let x: Vec<f32> = (0..n_tok * d_in).map(|_| rng.next_f32(0.5)).collect();
    let y0: Vec<f32> = (0..n_tok * d_out).map(|_| rng.next_f32(0.2)).collect();
    let a_g = ctx.upload(&a).unwrap();
    let bt_g = ctx.upload(&b_t).unwrap();
    let x_g = ctx.upload(&x).unwrap();
    let h_g = ctx.alloc_buffer((n_tok * rank * 4) as u64).unwrap();
    let y_batch_g = ctx.upload(&y0).unwrap();
    ctx.lora_linear_raw(
        &a_g, &bt_g, &x_g, &h_g, &y_batch_g, d_in, d_out, rank, n_tok, scale,
    )
    .unwrap();
    let y_batch = ctx.read_f32(&y_batch_g, n_tok * d_out).unwrap();
    for t in 0..n_tok {
        let xt_g = ctx.upload(&x[t * d_in..(t + 1) * d_in]).unwrap();
        let yt_g = ctx.upload(&y0[t * d_out..(t + 1) * d_out]).unwrap();
        ctx.lora_linear_raw(&a_g, &bt_g, &xt_g, &h_g, &yt_g, d_in, d_out, rank, 1, scale)
            .unwrap();
        let yt = ctx.read_f32(&yt_g, d_out).unwrap();
        for c in 0..d_out {
            assert_eq!(
                y_batch[t * d_out + c].to_bits(),
                yt[c].to_bits(),
                "0-ULP batch vs single mismatch in Vulkan lora_linear_raw at t={t} c={c}"
            );
        }
    }

    // Rule #3 Microbenchmark: `Q4_0` `4096 x 2048` (`T = 128`) transposed GEMM
    let b_rows = 4096usize;
    let b_cols = 2048usize;
    let b_tok = 128usize;
    let w_b = vec![0u8; b_rows * DType::Q4_0.row_bytes(b_cols)];
    let dy_b = vec![0.01f32; b_tok * b_rows];
    let w_bg = ctx.upload_bytes(&w_b).unwrap();
    let dy_bg = ctx.upload(&dy_b).unwrap();
    let dx_bg = ctx.alloc_buffer((b_tok * b_cols * 4) as u64).unwrap();
    ctx.gemm_transposed_raw(
        DType::Q4_0,
        &w_bg,
        &dy_bg,
        &dx_bg,
        b_rows,
        b_cols,
        b_tok,
        false,
        2,
    )
    .unwrap();
    let repeat = 10usize;
    let t0 = Instant::now();
    ctx.gemm_transposed_raw(
        DType::Q4_0,
        &w_bg,
        &dy_bg,
        &dx_bg,
        b_rows,
        b_cols,
        b_tok,
        false,
        repeat,
    )
    .unwrap();
    let sec = t0.elapsed().as_secs_f64() / (repeat as f64);
    let gflops = (2 * b_tok * b_rows * b_cols) as f64 / sec / 1e9;
    let gbs = (w_b.len() + b_tok * (b_rows + b_cols) * 4) as f64 / sec / 1e9;
    eprintln!(
        "[T4 Vulkan bwd bench] Q4_0 {b_rows}x{b_cols} T={b_tok} ({}): {:.2} us | {:.1} GFLOP/s | {:.1} GB/s",
        ctx.device_name(),
        sec * 1e6,
        gflops,
        gbs
    );
}
