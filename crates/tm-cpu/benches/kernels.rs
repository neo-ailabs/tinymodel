//! Zero-dependency microbenchmarks for `tm-cpu` kernels (`AGENTS.md` Rule #3: `cargo bench -p tm-cpu`).

use std::hint::black_box;
use std::time::Instant;

use tm_core::{BlockNvFp4, BlockQ4_0, BlockQ8_0, f32_to_f16};
use tm_cpu::{Isa, active_kernels, select_kernels};

fn bench_mamba2_scan() {
    let n_heads = 32usize;
    let head_dim = 64usize;
    let d_state = 128usize;
    let iters = 2_000usize;

    let x_h: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.13).sin()).collect();
    let b_g: Vec<f32> = (0..d_state).map(|i| ((i as f32) * 0.07).cos()).collect();
    let c_g: Vec<f32> = (0..d_state).map(|i| ((i as f32) * 0.11).sin()).collect();

    for (label, kt) in [
        ("generic", select_kernels(Isa::Generic)),
        ("active", active_kernels()),
    ] {
        let mut s_h = vec![0.01f32; head_dim * d_state];
        let mut y_h = vec![0.0f32; head_dim];
        // Warmup
        for _ in 0..64 {
            (kt.mamba2_scan_head)(
                black_box(&x_h),
                black_box(&b_g),
                black_box(&c_g),
                0.95,
                0.25,
                0.5,
                &mut s_h,
                &mut y_h,
                head_dim,
                d_state,
            );
        }
        let t0 = Instant::now();
        for _ in 0..(iters * n_heads) {
            (kt.mamba2_scan_head)(
                black_box(&x_h),
                black_box(&b_g),
                black_box(&c_g),
                0.95,
                0.25,
                0.5,
                &mut s_h,
                &mut y_h,
                head_dim,
                d_state,
            );
        }
        let secs = t0.elapsed().as_secs_f64();
        let state_bytes_rw = (iters * n_heads * head_dim * d_state * 8) as f64;
        let gbps = state_bytes_rw / secs / 1e9;
        println!(
            "mamba2_scan_head ({label}, isa={}): {:.2} us/layer ({n_heads}x{head_dim}x{d_state}), state_bw={gbps:.1} GB/s (sink={:.4})",
            kt.isa.name(),
            (secs * 1e6) / (iters as f64),
            y_h[0]
        );
    }
}

fn bench_q4_0_gemv() {
    let rows = 1024usize;
    let cols_blocks = 32usize;
    let iters = 500usize;
    let w = vec![
        BlockQ4_0 {
            d: f32_to_f16(0.125),
            qs: [0x5A; 16],
        };
        rows * cols_blocks
    ];
    let x = vec![
        BlockQ8_0 {
            d: f32_to_f16(0.25),
            qs: [3; 32],
        };
        cols_blocks
    ];
    let mut out = vec![0.0f32; rows];
    let kt = active_kernels();
    for _ in 0..32 {
        (kt.gemv_q4_0_q8_0)(black_box(&w), black_box(&x), &mut out, rows, cols_blocks);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        (kt.gemv_q4_0_q8_0)(black_box(&w), black_box(&x), &mut out, rows, cols_blocks);
    }
    let secs = t0.elapsed().as_secs_f64();
    let gbps = ((iters * rows * cols_blocks * 18) as f64) / secs / 1e9;
    println!(
        "gemv_q4_0_q8_0 (isa={}): {gbps:.1} GB/s (sink={:.2})",
        kt.isa.name(),
        out[0]
    );
}

fn bench_nvfp4_gemv() {
    let rows = 2304usize;
    let cols_blocks = 2304usize / 64;
    let iters = 500usize;
    let w = vec![
        BlockNvFp4 {
            d: [0x38, 0x3A, 0x36, 0x3C],
            qs: [0x5A; 32],
        };
        rows * cols_blocks
    ];
    let x = vec![
        BlockQ8_0 {
            d: f32_to_f16(0.25),
            qs: [3; 32],
        };
        cols_blocks * 2
    ];
    let mut out = vec![0.0f32; rows];
    for (label, kt) in [
        ("generic", select_kernels(Isa::Generic)),
        ("active", active_kernels()),
    ] {
        for _ in 0..32 {
            (kt.gemv_nvfp4_q8_0)(black_box(&w), black_box(&x), &mut out, rows, cols_blocks);
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            (kt.gemv_nvfp4_q8_0)(black_box(&w), black_box(&x), &mut out, rows, cols_blocks);
        }
        let secs = t0.elapsed().as_secs_f64();
        let gbps = ((iters * rows * cols_blocks * 36) as f64) / secs / 1e9;
        println!(
            "gemv_nvfp4_q8_0 ({label}, isa={}): {gbps:.1} GB/s (sink={:.2})",
            kt.isa.name(),
            out[0]
        );
    }
}

fn main() {
    bench_mamba2_scan();
    bench_q4_0_gemv();
    bench_nvfp4_gemv();
}
