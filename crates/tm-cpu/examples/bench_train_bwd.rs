//! Microbenchmark for `matmul_q_transposed_f32` (`dX = dY * W_q`) backward kernel (`AGENTS.md` Rule #3).

use std::time::Instant;
use tm_core::{DType, quantize_row_to_bytes};
use tm_cpu::{ThreadPool, active_kernels, matmul_q_transposed_f32, recommended_threads};

fn main() {
    let rows = 3584usize;
    let cols = 1024usize;
    let n_tok = 64usize;
    let threads = recommended_threads();
    let pool = ThreadPool::new(threads);
    let kt = active_kernels();

    let src_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (((i * 17 + 5) % 97) as f32 - 48.0) * 0.01)
        .collect();
    let rb = DType::Q4_0.row_bytes(cols);
    let mut w_q4 = vec![0u8; rows * rb];
    for r in 0..rows {
        quantize_row_to_bytes(
            DType::Q4_0,
            &src_f32[r * cols..(r + 1) * cols],
            &mut w_q4[r * rb..(r + 1) * rb],
        );
    }

    let dy: Vec<f32> = (0..n_tok * rows)
        .map(|i| (((i * 13 + 3) % 89) as f32 - 44.0) * 0.02)
        .collect();
    let mut dx = vec![0.0f32; n_tok * cols];

    for _ in 0..5 {
        matmul_q_transposed_f32(
            &pool,
            kt,
            DType::Q4_0,
            &w_q4,
            &dy,
            &mut dx,
            rows,
            cols,
            n_tok,
            false,
        );
    }

    let iters = 30usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        matmul_q_transposed_f32(
            &pool,
            kt,
            DType::Q4_0,
            &w_q4,
            &dy,
            &mut dx,
            rows,
            cols,
            n_tok,
            false,
        );
    }
    let sec = t0.elapsed().as_secs_f64() / (iters as f64);
    let flops = 2.0 * (n_tok as f64) * (rows as f64) * (cols as f64);
    let gflops = flops / (sec * 1e9);
    let gbs = (w_q4.len() as f64) / (sec * 1e9);
    println!(
        "bench_train_bwd: Q4_0 transposed [{n_tok}x{rows}] * [{rows}x{cols}] isa={} threads={threads} -> {:.2} ms ({:.2} GFLOP/s, {:.2} GB/s, sample={:.4})",
        kt.isa.name(),
        sec * 1e3,
        gflops,
        gbs,
        dx[0]
    );
}
