//! Micro-benchmark for `tm-cpu` `Q4_0 x Q8_0` GEMV and batched GEMM.
//!
//! Run:
//!   cargo run --release -p tm-cpu --example bench_gemv_q4_0

use std::time::Instant;
use tm_core::{BlockQ4_0, BlockQ8_0, QK4_0, f32_to_f16};
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
}
