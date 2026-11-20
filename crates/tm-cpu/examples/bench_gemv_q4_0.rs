//! Micro-benchmark for `tm-cpu` `Q4_0 x Q8_0` GEMV and batched GEMM.
//!
//! Run:
//!   cargo run --release -p tm-cpu --example bench_gemv_q4_0

use std::time::Instant;
use tm_core::{
    BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, QK_K, QK4_0,
    f32_to_f16, quantize_row_q8_1, quantize_row_q8_k,
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
}
