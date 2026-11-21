//! Micro-benchmark for `tm-cpu` `Q4_0 x Q8_0` GEMV and batched GEMM.
//!
//! Run:
//!   cargo run --release -p tm-cpu --example bench_gemv_q4_0

use std::time::Instant;
use tm_core::{
    BlockIQ2XXS, BlockIQ3XXS, BlockIQ4NL, BlockQ2_0, BlockQ4_0, BlockQ4_0x4, BlockQ4_1, BlockQ5K,
    BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, BlockTQ2_0, QK_K, QK2_0, QK4_0, f32_to_f16,
    quantize_row_q8_1, quantize_row_q8_k, repack_q4_0_to_q4_0x4,
};
use tm_cpu::isa::supported_isas;
use tm_cpu::{ThreadPool, select_kernels};

fn make_data(rows: usize, cols_blocks: usize, n_tokens: usize) -> (Vec<BlockQ4_0>, Vec<BlockQ8_0>) {
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    let d_w = f32_to_f16(0.03125);
    let d_x = f32_to_f16(0.0625);

    let w: Vec<BlockQ4_0> = (0..rows * cols_blocks)
        .map(|_| {
            let mut qs = [0u8; 16];
            for b in &mut qs {
                *b = (next() & 0xFF) as u8;
            }
            BlockQ4_0 { d: d_w, qs }
        })
        .collect();

    let x: Vec<BlockQ8_0> = (0..n_tokens * cols_blocks)
        .map(|_| {
            let mut qs = [0i8; 32];
            for q in &mut qs {
                *q = (next() & 0xFF) as i8;
            }
            BlockQ8_0 { d: d_x, qs }
        })
        .collect();

    (w, x)
}

fn main() {
    // Representative LLM projection shapes:
    // - 4096 x 4096 (8B / 4B FFN/attn, 9.44 MB weights — fits in M4 SLC/L2? 9.44MB < 16MB L2!)
    // - 8192 x 4096 (18.87 MB weights)
    // - 16384 x 4096 (37.75 MB weights — guarantees DRAM-bound streaming beyond M4's 16MB L2!)
    let shapes = [
        (4096usize, 4096usize, 1usize),
        (16384, 4096, 1),
        (4096, 4096, 4),
    ];
    let isas = supported_isas();

    println!(
        "Supported ISAs: {:?}",
        isas.iter().map(|i| i.name()).collect::<Vec<_>>()
    );
    println!(
        "{:<10} {:>14} {:>4} {:>8} {:>10} {:>10} {:>10}",
        "ISA", "Shape (MxKxN)", "Thr", "ms/iter", "GB/s", "GOPS", "checksum"
    );
    println!("{}", "-".repeat(74));

    for &(rows, cols, n_tokens) in &shapes {
        let cols_blocks = cols / QK4_0;
        let (w, x) = make_data(rows, cols_blocks, n_tokens);
        let weight_bytes = (rows * cols_blocks * core::mem::size_of::<BlockQ4_0>()) as f64;
        let ops = 2.0 * (rows as f64) * (cols as f64) * (n_tokens as f64);

        for &threads in &[1usize, 4usize] {
            let pool = ThreadPool::new(threads);

            for &isa in &isas {
                // Skip slow scalar on huge shape with 1 thread to keep output snappy
                if isa == tm_cpu::Isa::Generic && rows >= 16384 && threads == 1 {
                    continue;
                }
                let kt = select_kernels(isa);
                let mut out = vec![0.0f32; n_tokens * rows];
                let iters = if isa == tm_cpu::Isa::Generic { 3 } else { 30 };

                let run_once = |out_buf: &mut [f32]| {
                    if n_tokens == 1 {
                        let out_ptr = out_buf.as_mut_ptr() as usize;
                        let chunk = (rows / (threads * 4)).max(16);
                        pool.parallel_for(rows, chunk, |r0, r1| {
                            let len = r1 - r0;
                            let w_slice = &w[r0 * cols_blocks..r1 * cols_blocks];
                            // SAFETY: Disjoint row ranges `[r0..r1)` per chunk.
                            let o_slice = unsafe {
                                core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len)
                            };
                            (kt.gemv_q4_0_q8_0)(w_slice, &x, o_slice, len, cols_blocks);
                        });
                    } else {
                        let out_ptr = out_buf.as_mut_ptr() as usize;
                        let chunk = (rows / (threads * 4)).max(16);
                        pool.parallel_for(rows, chunk, |r0, r1| {
                            let len = r1 - r0;
                            let w_slice = &w[r0 * cols_blocks..r1 * cols_blocks];
                            let mut local = vec![0.0f32; n_tokens * len];
                            (kt.gemm_q4_0_q8_0)(
                                w_slice,
                                &x,
                                &mut local,
                                len,
                                cols_blocks,
                                n_tokens,
                            );
                            for t in 0..n_tokens {
                                // SAFETY: Disjoint `[r0..r1)` per token slice.
                                let dst = unsafe {
                                    core::slice::from_raw_parts_mut(
                                        (out_ptr as *mut f32).add(t * rows + r0),
                                        len,
                                    )
                                };
                                dst.copy_from_slice(&local[t * len..(t + 1) * len]);
                            }
                        });
                    }
                };

                // Warmup
                run_once(&mut out);

                let t0 = Instant::now();
                for _ in 0..iters {
                    run_once(&mut out);
                }
                let elapsed = t0.elapsed().as_secs_f64() / (iters as f64);
                let gbs = (weight_bytes / elapsed) / 1e9;
                let gops = (ops / elapsed) / 1e9;
                let checksum: f32 = out.iter().take(16).sum();

                println!(
                    "{:<10} {:>14} {:>4} {:>8.3} {:>10.2} {:>10.2} {:>10.2}",
                    isa.name(),
                    format!("{rows}x{cols}x{n_tokens}"),
                    threads,
                    elapsed * 1e3,
                    gbs,
                    gops,
                    checksum
                );
            }
        }
        println!();
    }

    // Also benchmark F16 KV dot + Q8_0 activation quantization + Q4_1 GEMV across ISAs
    let src_f32: Vec<f32> = (0..8192).map(|i| ((i as f32) - 4096.0) * 0.001).collect();
    let mut dst_q80 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; 256];
    let mut kv_f16 = vec![f32_to_f16(0.5); 64 * 1024];
    println!("--- Auxiliary CPU Kernel Microbench (1 thread) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        (kt.f32_to_f16_slice)(&src_f32[..1024], &mut kv_f16[..1024]);
        let t_q = Instant::now();
        for _ in 0..1000 {
            (kt.quantize_q8_0)(&src_f32, &mut dst_q80);
        }
        let q_us = t_q.elapsed().as_secs_f64() * 1e6 / 1000.0;

        let t_kv = Instant::now();
        let mut acc = 0.0f32;
        for step in 0..1024 {
            acc += (kt.dot_f32_f16)(&src_f32[..64], &kv_f16[step * 64..(step + 1) * 64]);
        }
        let kv_us = t_kv.elapsed().as_secs_f64() * 1e6;
        println!(
            "{:<10} quantize_q8_0(8192)={:.2} us | dot_f32_f16(1024x64)={:.2} us (acc={:.1})",
            isa.name(),
            q_us,
            kv_us,
            acc
        );
    }

    // Benchmark Q5_K x Q8_K GEMV on Qwen3.5 `ssm_out.weight` shape (2048 x 2048)
    let q5_rows = 2048usize;
    let q5_cols = 2048usize;
    let q5_nb = q5_cols / QK_K;
    let w_q5: Vec<BlockQ5K> = (0..q5_rows * q5_nb)
        .map(|i| BlockQ5K {
            d: f32_to_f16(0.03125),
            dmin: f32_to_f16(0.015625),
            scales: [(i & 0x3F) as u8; 12],
            qh: [0x55; 32],
            qs: [0x96; 128],
        })
        .collect();
    let mut x_q8k = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16]
        };
        q5_nb
    ];
    quantize_row_q8_k(&src_f32[..q5_cols], &mut x_q8k);
    let q5_bytes = (q5_rows * q5_nb * core::mem::size_of::<BlockQ5K>()) as f64;
    let pool4 = ThreadPool::new(4);
    println!("\n--- Q5_K x Q8_K GEMV (2048x2048, 4 threads) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut out_q5 = vec![0.0f32; q5_rows];
        let out_ptr = out_q5.as_mut_ptr() as usize;
        let run_q5 = || {
            pool4.parallel_for(q5_rows, 512, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` per worker.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q5_k_q8_k)(&w_q5[r0 * q5_nb..r1 * q5_nb], &x_q8k, o, len, q5_nb);
            });
        };
        run_q5();
        let iters = if isa == tm_cpu::Isa::Generic { 10 } else { 50 };
        let t0 = Instant::now();
        for _ in 0..iters {
            run_q5();
        }
        let sec = t0.elapsed().as_secs_f64() / (iters as f64);
        println!(
            "{:<10} {:>7.3} ms | {:>6.2} GB/s | checksum={:.2}",
            isa.name(),
            sec * 1e3,
            (q5_bytes / sec) / 1e9,
            out_q5.iter().take(8).sum::<f32>()
        );
    }

    // Benchmark Q8_0 x Q8_0 GEMV (SmolLM2-360M lm_head shape: 49152 x 960)
    let q8_rows = 49152usize;
    let q8_cols = 960usize;
    let q8_nb = q8_cols / QK4_0;
    let w_q80: Vec<BlockQ8_0> = (0..q8_rows * q8_nb)
        .map(|i| BlockQ8_0 {
            d: f32_to_f16(0.03125),
            qs: [((i & 0x7F) as i8) - 64; 32],
        })
        .collect();
    let x_q80 = dst_q80[..q8_nb].to_vec();
    let q8_bytes = (q8_rows * q8_nb * core::mem::size_of::<BlockQ8_0>()) as f64;
    println!("\n--- Q8_0 x Q8_0 GEMV (49152x960, 4 threads) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut out_q8 = vec![0.0f32; q8_rows];
        let out_ptr = out_q8.as_mut_ptr() as usize;
        let run_q8 = || {
            pool4.parallel_for(q8_rows, 3072, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` per worker.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q8_0_q8_0)(&w_q80[r0 * q8_nb..r1 * q8_nb], &x_q80, o, len, q8_nb);
            });
        };
        run_q8();
        let iters = if isa == tm_cpu::Isa::Generic { 5 } else { 30 };
        let t0 = Instant::now();
        for _ in 0..iters {
            run_q8();
        }
        let sec = t0.elapsed().as_secs_f64() / (iters as f64);
        println!(
            "{:<10} {:>7.3} ms | {:>6.2} GB/s | checksum={:.2}",
            isa.name(),
            sec * 1e3,
            (q8_bytes / sec) / 1e9,
            out_q8.iter().take(8).sum::<f32>()
        );
    }

    // Benchmark Q4_1 x Q8_1 GEMV (Qwen3.5-0.8B ffn_down shape: 1024 x 3584)
    let q41_rows = 1024usize;
    let q41_cols = 3584usize;
    let q41_nb = q41_cols / QK4_0;
    let w_q41: Vec<BlockQ4_1> = (0..q41_rows * q41_nb)
        .map(|i| BlockQ4_1 {
            d: f32_to_f16(0.03125),
            m: f32_to_f16(-0.125),
            qs: [(i & 0xFF) as u8; 16],
        })
        .collect();
    let mut x_q81 = vec![
        BlockQ8_1 {
            d: 0,
            s: 0,
            qs: [0; 32]
        };
        q41_nb
    ];
    quantize_row_q8_1(&src_f32[..q41_cols], &mut x_q81);
    let q41_bytes = (q41_rows * q41_nb * core::mem::size_of::<BlockQ4_1>()) as f64;
    println!("\n--- Q4_1 x Q8_1 GEMV (1024x3584, 4 threads) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut out_q41 = vec![0.0f32; q41_rows];
        let out_ptr = out_q41.as_mut_ptr() as usize;
        let run_q41 = || {
            pool4.parallel_for(q41_rows, 256, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` per worker.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_1_q8_1)(&w_q41[r0 * q41_nb..r1 * q41_nb], &x_q81, o, len, q41_nb);
            });
        };
        run_q41();
        let iters = if isa == tm_cpu::Isa::Generic { 20 } else { 100 };
        let t0 = Instant::now();
        for _ in 0..iters {
            run_q41();
        }
        let sec = t0.elapsed().as_secs_f64() / (iters as f64);
        println!(
            "{:<10} {:>7.3} ms | {:>6.2} GB/s | checksum={:.2}",
            isa.name(),
            sec * 1e3,
            (q41_bytes / sec) / 1e9,
            out_q41.iter().take(8).sum::<f32>()
        );
    }

    // Benchmark Q6_K x Q8_K GEMV (2048 x 2048, 4 threads)
    let w_q6: Vec<BlockQ6K> = (0..q5_rows * q5_nb)
        .map(|i| BlockQ6K {
            ql: [(i & 0xFF) as u8; 128],
            qh: [0x55; 64],
            scales: [((i & 0x0F) as i8) - 7; 16],
            d: f32_to_f16(0.03125),
        })
        .collect();
    let q6_bytes = (q5_rows * q5_nb * core::mem::size_of::<BlockQ6K>()) as f64;
    println!("\n--- Q6_K x Q8_K GEMV (2048x2048, 4 threads) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut out_q6 = vec![0.0f32; q5_rows];
        let out_ptr = out_q6.as_mut_ptr() as usize;
        let run_q6 = || {
            pool4.parallel_for(q5_rows, 512, |r0, r1| {
                let len = r1 - r0;
                // SAFETY: Disjoint `[r0..r1)` per worker.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q6_k_q8_k)(&w_q6[r0 * q5_nb..r1 * q5_nb], &x_q8k, o, len, q5_nb);
            });
        };
        run_q6();
        let iters = if isa == tm_cpu::Isa::Generic { 10 } else { 50 };
        let t0 = Instant::now();
        for _ in 0..iters {
            run_q6();
        }
        let sec = t0.elapsed().as_secs_f64() / (iters as f64);
        println!(
            "{:<10} {:>7.3} ms | {:>6.2} GB/s | checksum={:.2}",
            isa.name(),
            sec * 1e3,
            (q6_bytes / sec) / 1e9,
            out_q6.iter().take(8).sum::<f32>()
        );
    }

    // Benchmark Qwen3.5 GDN Recurrence (16 heads x 128x128 state) + RMSNorm across ISAs
    let hv = 16usize;
    let hd = 128usize;
    let q_gdn: Vec<f32> = (0..hv * hd)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.002)
        .collect();
    let k_gdn: Vec<f32> = (0..hv * hd)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.002)
        .collect();
    let v_gdn: Vec<f32> = (0..hv * hd)
        .map(|i| ((i % 83) as f32 - 41.0) * 0.005)
        .collect();
    let norm_w: Vec<f32> = vec![1.0; 2048];
    println!("\n--- GDN Recurrence (16 heads x 128x128, 4 threads) & RMSNorm(2048) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut s_all = vec![0.01f32; hv * hd * hd];
        let mut o_all = vec![0.0f32; hv * hd];
        let s_ptr = s_all.as_mut_ptr() as usize;
        let o_ptr = o_all.as_mut_ptr() as usize;
        let scale = 1.0 / (hd as f32).sqrt();
        let run_gdn = || {
            pool4.parallel_for(hv, 4, |h0, h1| {
                for h in h0..h1 {
                    // SAFETY: Disjoint state and output slices per head `h`.
                    let (sh, oh) = unsafe {
                        (
                            core::slice::from_raw_parts_mut(
                                (s_ptr as *mut f32).add(h * hd * hd),
                                hd * hd,
                            ),
                            core::slice::from_raw_parts_mut((o_ptr as *mut f32).add(h * hd), hd),
                        )
                    };
                    (kt.gdn_recurrence_head)(
                        &q_gdn[h * hd..(h + 1) * hd],
                        &k_gdn[h * hd..(h + 1) * hd],
                        &v_gdn[h * hd..(h + 1) * hd],
                        0.92,
                        0.35,
                        scale,
                        sh,
                        oh,
                        hd,
                    );
                }
            });
        };
        run_gdn();
        let iters = 500;
        let t0 = Instant::now();
        for _ in 0..iters {
            run_gdn();
        }
        let gdn_us = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);

        let mut rms_out = vec![0.0f32; 2048];
        let t_r = Instant::now();
        for _ in 0..2000 {
            (kt.rms_norm_f32)(&mut rms_out, &src_f32[..2048], &norm_w, 1e-6);
        }
        let rms_us = t_r.elapsed().as_secs_f64() * 1e6 / 2000.0;

        let bf16_row: Vec<u16> = (0..1536)
            .map(|i| 0x3F00u16.wrapping_add(i as u16 & 0x7F))
            .collect();
        let t_bf = Instant::now();
        let mut bf_acc = 0.0f32;
        for _ in 0..2000 {
            bf_acc += (kt.dot_f32_bf16)(&src_f32[..1536], &bf16_row);
        }
        let bf_us = t_bf.elapsed().as_secs_f64() * 1e6 / 2000.0;

        println!(
            "{:<10} gdn_recur_16h={:>6.2} us/layer | rms_norm(2048)={:>5.3} us | dot_bf16(1536)={:>5.3} us | chk={:.3}",
            isa.name(),
            gdn_us,
            rms_us,
            bf_us,
            o_all[0] + rms_out[0] + bf_acc * 1e-6
        );
    }

    // M3 Benchmark: Repacked `Q4_0x4` (`Q4_0_4x8`) GEMV (1 tok) & Prefill GEMM (32 tok) + Q8_0 KV
    let m3_rows = 4096usize;
    let m3_cols = 4096usize;
    let m3_nb = m3_cols / QK4_0;
    let m3_ntok = 32usize;
    let (w_raw, x_batch) = make_data(m3_rows, m3_nb, m3_ntok);
    let mut w_rep = vec![
        BlockQ4_0x4 {
            d: [0; 4],
            qs: [0; 64]
        };
        (m3_rows / 4) * m3_nb
    ];
    repack_q4_0_to_q4_0x4(&w_raw, &mut w_rep, m3_rows, m3_nb);
    let m3_bytes = (w_rep.len() * core::mem::size_of::<BlockQ4_0x4>()) as f64;
    let m3_ops_gemm = 2.0 * (m3_rows as f64) * (m3_cols as f64) * (m3_ntok as f64);

    println!(
        "\n--- M3 Repacked Q4_0x4 (4096x4096, 4 threads): GEMV (N=1) & Prefill GEMM (N=32) ---"
    );
    for &isa in &isas {
        let kt = select_kernels(isa);
        let mut out_v = vec![0.0f32; m3_rows];
        let ov_ptr = out_v.as_mut_ptr() as usize;
        let n_groups = m3_rows / 4;
        let run_v = || {
            pool4.parallel_for(n_groups, n_groups / 4, |g0, g1| {
                let r0 = g0 * 4;
                let len = (g1 - g0) * 4;
                // SAFETY: Disjoint `[r0..r0+len)` slice per worker.
                let o =
                    unsafe { core::slice::from_raw_parts_mut((ov_ptr as *mut f32).add(r0), len) };
                (kt.gemv_q4_0x4_q8_0)(
                    &w_rep[g0 * m3_nb..g1 * m3_nb],
                    &x_batch[..m3_nb],
                    o,
                    len,
                    m3_nb,
                );
            });
        };
        run_v();
        let iters = if isa == tm_cpu::Isa::Generic { 5 } else { 40 };
        let t0 = Instant::now();
        for _ in 0..iters {
            run_v();
        }
        let sec_v = t0.elapsed().as_secs_f64() / (iters as f64);

        let mut out_m = vec![0.0f32; m3_ntok * m3_rows];
        let om_ptr = out_m.as_mut_ptr() as usize;
        let run_m = || {
            pool4.parallel_for(n_groups, n_groups / 4, |g0, g1| {
                let mut g = g0;
                let mut tile = [0.0f32; 64 * 32];
                while g < g1 {
                    let bg = (g1 - g).min(16);
                    let b_rows = bg * 4;
                    let r = g * 4;
                    (kt.gemm_q4_0x4_q8_0)(
                        &w_rep[g * m3_nb..(g + bg) * m3_nb],
                        &x_batch,
                        &mut tile[..b_rows * m3_ntok],
                        b_rows,
                        m3_nb,
                        m3_ntok,
                    );
                    for t in 0..m3_ntok {
                        // SAFETY: Disjoint `[t * m3_rows + r .. + b_rows]` per worker.
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(
                                (om_ptr as *mut f32).add(t * m3_rows + r),
                                b_rows,
                            )
                        };
                        dst.copy_from_slice(&tile[t * b_rows..(t + 1) * b_rows]);
                    }
                    g += bg;
                }
            });
        };
        run_m();
        let iters_m = if isa == tm_cpu::Isa::Generic { 2 } else { 20 };
        let t1 = Instant::now();
        for _ in 0..iters_m {
            run_m();
        }
        let sec_m = t1.elapsed().as_secs_f64() / (iters_m as f64);

        let mut kv_acc = 0.0f32;
        let t_kv8 = Instant::now();
        for step in 0..512 {
            kv_acc += (kt.dot_f32_q8_0)(
                &src_f32[..64],
                &dst_q80[(step % 128) * 2..(step % 128) * 2 + 2],
            );
        }
        let kv8_us = t_kv8.elapsed().as_secs_f64() * 1e6;

        println!(
            "{:<10} GEMV={:>6.3} ms ({:>6.2} GB/s) | GEMM32={:>6.3} ms ({:>6.1} GOPS) | KV_Q8(512x64)={:>5.2} us | chk={:.1}",
            isa.name(),
            sec_v * 1e3,
            (m3_bytes / sec_v) / 1e9,
            sec_m * 1e3,
            (m3_ops_gemm / sec_m) / 1e9,
            kv8_us,
            out_v[0] + out_m[0] + kv_acc * 1e-6
        );
    }

    // M4 Benchmark: Low-Bit & T-MAC 16-row `TBL` Lookup Kernels (`Q2_0`, `TQ2_0`, `IQ2_XXS`, `IQ3_XXS`, `IQ4_NL`)
    let m4_rows = 4096usize;
    let m4_cols = 4096usize;
    let nb_q2 = m4_cols / QK2_0;
    let nb_qk = m4_cols / QK_K;
    let nb_q4 = m4_cols / QK4_0;
    let w_q20: Vec<BlockQ2_0> = (0..m4_rows * nb_q2)
        .map(|i| BlockQ2_0 {
            d: f32_to_f16(0.03125),
            qs: [(i & 0xFF) as u8; 16],
        })
        .collect();
    let w_tq20: Vec<BlockTQ2_0> = (0..m4_rows * nb_qk)
        .map(|i| BlockTQ2_0 {
            qs: [((i * 3) % 85) as u8; 64],
            d: f32_to_f16(0.03125),
        })
        .collect();
    let w_iq2: Vec<BlockIQ2XXS> = (0..m4_rows * nb_qk)
        .map(|i| BlockIQ2XXS {
            d: f32_to_f16(0.03125),
            qs: [(i & 0x7FFF) as u16; 32],
        })
        .collect();
    let w_iq3: Vec<BlockIQ3XXS> = (0..m4_rows * nb_qk)
        .map(|i| BlockIQ3XXS {
            d: f32_to_f16(0.03125),
            qs: [(i & 0xFF) as u8; 96],
        })
        .collect();
    let w_iq4: Vec<BlockIQ4NL> = (0..m4_rows * nb_q4)
        .map(|i| BlockIQ4NL {
            d: f32_to_f16(0.03125),
            qs: [(i & 0xFF) as u8; 16],
        })
        .collect();
    let mut x_q8k_4096 = vec![
        BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16]
        };
        nb_qk
    ];
    quantize_row_q8_k(&src_f32[..m4_cols], &mut x_q8k_4096);
    let x_q80_4096 = dst_q80[..nb_q4].to_vec();

    println!("\n--- M4 Low-Bit & T-MAC TBL Lookup GEMV (4096x4096, 4 threads) ---");
    for &isa in &isas {
        let kt = select_kernels(isa);
        let iters = if isa == tm_cpu::Isa::Generic { 3 } else { 30 };
        let mut out = vec![0.0f32; m4_rows];
        let out_ptr = out.as_mut_ptr() as usize;

        macro_rules! bench_fmt {
            ($w_vec:expr, $nb:expr, $call:expr) => {{
                let run = || {
                    pool4.parallel_for(m4_rows, 1024, |r0, r1| {
                        let len = r1 - r0;
                        // SAFETY: Disjoint `[r0..r1)` per worker.
                        let o = unsafe {
                            core::slice::from_raw_parts_mut((out_ptr as *mut f32).add(r0), len)
                        };
                        $call(&$w_vec[r0 * $nb..r1 * $nb], o, len, $nb);
                    });
                };
                run();
                let t0 = Instant::now();
                for _ in 0..iters {
                    run();
                }
                let sec = t0.elapsed().as_secs_f64() / (iters as f64);
                let bytes = ($w_vec.len() * core::mem::size_of_val(&$w_vec[0])) as f64;
                (sec * 1e3, (bytes / sec) / 1e9, out[0])
            }};
        }

        let (ms_q2, bw_q2, c0) = bench_fmt!(w_q20, nb_q2, |ws, o, len, nb| {
            (kt.gemv_q2_0_q8_0)(ws, &x_q80_4096, o, len, nb)
        });
        let (ms_tq2, bw_tq2, c1) = bench_fmt!(w_tq20, nb_qk, |ws, o, len, nb| {
            (kt.gemv_tq2_0_q8_k)(ws, &x_q8k_4096, o, len, nb)
        });
        let (ms_iq2, bw_iq2, c2) = bench_fmt!(w_iq2, nb_qk, |ws, o, len, nb| {
            (kt.gemv_iq2_xxs_q8_k)(ws, &x_q8k_4096, o, len, nb)
        });
        let (ms_iq3, bw_iq3, c3) = bench_fmt!(w_iq3, nb_qk, |ws, o, len, nb| {
            (kt.gemv_iq3_xxs_q8_k)(ws, &x_q8k_4096, o, len, nb)
        });
        let (ms_iq4, bw_iq4, c4) = bench_fmt!(w_iq4, nb_q4, |ws, o, len, nb| {
            (kt.gemv_iq4_nl_q8_0)(ws, &x_q80_4096, o, len, nb)
        });

        println!(
            "{:<10} Q2_0={:>5.2}ms({:>5.1}GB/s) | TQ2_0={:>5.2}ms({:>5.1}GB/s) | IQ2={:>5.2}ms({:>5.1}GB/s) | IQ3={:>5.2}ms({:>5.1}GB/s) | IQ4_NL={:>5.2}ms({:>5.1}GB/s) | chk={:.1}",
            isa.name(),
            ms_q2,
            bw_q2,
            ms_tq2,
            bw_tq2,
            ms_iq2,
            bw_iq2,
            ms_iq3,
            bw_iq3,
            ms_iq4,
            bw_iq4,
            c0 + c1 + c2 + c3 + c4
        );
    }
}
