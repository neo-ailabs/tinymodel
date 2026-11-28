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

fn bench_f32_q8_0_gemv_gemm() {
    let rows = 1024usize;
    let cols_blocks = 32usize;
    let cols = cols_blocks * 32;
    let n_tok = 8usize;
    let iters = 200usize;
    let w = vec![
        BlockQ8_0 {
            d: f32_to_f16(0.125),
            qs: [5; 32],
        };
        rows * cols_blocks
    ];
    let x: Vec<f32> = (0..n_tok * cols)
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();
    let mut out_v = vec![0.0f32; rows];
    let mut out_m = vec![0.0f32; n_tok * rows];
    let kt = active_kernels();

    for _ in 0..16 {
        (kt.gemv_f32_q8_0)(
            black_box(&w),
            black_box(&x[..cols]),
            &mut out_v,
            rows,
            cols_blocks,
        );
        (kt.gemm_f32_q8_0)(
            black_box(&w),
            black_box(&x),
            &mut out_m,
            rows,
            cols_blocks,
            n_tok,
        );
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        (kt.gemv_f32_q8_0)(
            black_box(&w),
            black_box(&x[..cols]),
            &mut out_v,
            rows,
            cols_blocks,
        );
    }
    let secs_v = t0.elapsed().as_secs_f64();
    let gbps_v = ((iters * rows * cols_blocks * 34) as f64) / secs_v / 1e9;

    let t1 = Instant::now();
    for _ in 0..iters {
        (kt.gemm_f32_q8_0)(
            black_box(&w),
            black_box(&x),
            &mut out_m,
            rows,
            cols_blocks,
            n_tok,
        );
    }
    let secs_m = t1.elapsed().as_secs_f64();
    let gflops_m = ((iters * 2 * n_tok * rows * cols) as f64) / secs_m / 1e9;
    println!(
        "gemv_f32_q8_0 (isa={}): {gbps_v:.1} GB/s | gemm_f32_q8_0 (m={n_tok}): {gflops_m:.1} GFLOP/s (sink={:.2})",
        kt.isa.name(),
        out_v[0] + out_m[0]
    );
}

fn bench_f16_kv_ops() {
    let hd = 128usize;
    let n_kv = 512usize;
    let iters = 2_000usize;
    let q: Vec<f32> = (0..hd).map(|i| ((i as f32) * 0.05).sin()).collect();
    let kv_f32: Vec<f32> = (0..n_kv * hd).map(|i| ((i as f32) * 0.02).cos()).collect();
    let mut kv_f16 = vec![0u16; n_kv * hd];
    let mut out = vec![0.0f32; hd];
    let kt = active_kernels();
    (kt.f32_to_f16_slice)(&kv_f32, &mut kv_f16);

    let mut sink = 0.0f32;
    for _ in 0..16 {
        for p in 0..n_kv {
            let s = (kt.dot_f32_f16)(black_box(&q), black_box(&kv_f16[p * hd..(p + 1) * hd]));
            (kt.axpy_f32_f16)(
                &mut out,
                s * 0.001,
                black_box(&kv_f16[p * hd..(p + 1) * hd]),
            );
            sink += s;
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        for p in 0..n_kv {
            let s = (kt.dot_f32_f16)(black_box(&q), black_box(&kv_f16[p * hd..(p + 1) * hd]));
            (kt.axpy_f32_f16)(
                &mut out,
                s * 0.001,
                black_box(&kv_f16[p * hd..(p + 1) * hd]),
            );
            sink += s;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let gbps = ((iters * n_kv * hd * 2 * 2) as f64) / secs / 1e9;
    println!(
        "dot+axpy_f32_f16 (isa={}): {gbps:.1} GB/s (sink={:.2})",
        kt.isa.name(),
        sink + out[0]
    );
}

fn bench_q2_0_gemv() {
    let rows = 2048usize;
    let cols = 2048usize;
    let cols_blocks = cols / 64;
    let iters = 200usize;

    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let mut w = vec![tm_core::BlockQ2_0 { d: 0, qs: [0; 16] }; rows * cols_blocks];
    tm_core::quantize_row_q2_0(&w_f32, &mut w);

    let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.019).cos()).collect();
    let mut x_q8 = vec![tm_core::BlockQ8_0 { d: 0, qs: [0; 32] }; cols / 32];
    tm_core::quantize_row_q8_0(&x_f32, &mut x_q8);
    let mut out = vec![0.0f32; rows];
    let kt = active_kernels();

    for _ in 0..10 {
        (kt.gemv_q2_0_q8_0)(black_box(&w), black_box(&x_q8), &mut out, rows, cols_blocks);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        (kt.gemv_q2_0_q8_0)(black_box(&w), black_box(&x_q8), &mut out, rows, cols_blocks);
    }
    let secs = t0.elapsed().as_secs_f64();
    let gflops = (2.0 * (rows * cols * iters) as f64) / secs / 1e9;
    println!(
        "gemv_q2_0_q8_0 2048x2048 (isa={}): {gflops:.1} GFLOP/s (out[0]={:.3})",
        kt.isa.name(),
        out[0]
    );
}

fn bench_q5_k_gemv() {
    let rows = 2048usize;
    let cols = 2048usize;
    let cols_blocks = cols / 256;
    let iters = 200usize;

    let w: Vec<tm_core::BlockQ5K> = (0..rows * cols_blocks)
        .map(|i| tm_core::BlockQ5K {
            d: tm_core::f32_to_f16(0.025),
            dmin: tm_core::f32_to_f16(0.01),
            scales: [((i % 15) as u8) | (((i * 3 % 15) as u8) << 4); 12],
            qh: [(i * 7 % 256) as u8; 32],
            qs: [(i * 13 % 256) as u8; 128],
        })
        .collect();

    let x_f32: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.019).cos()).collect();
    let mut x_q8k = vec![
        tm_core::BlockQ8K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        };
        cols_blocks
    ];
    tm_core::quantize_row_q8_k(&x_f32, &mut x_q8k);
    let mut out = vec![0.0f32; rows];
    let kt = active_kernels();

    for _ in 0..10 {
        (kt.gemv_q5_k_q8_k)(
            black_box(&w),
            black_box(&x_q8k),
            &mut out,
            rows,
            cols_blocks,
        );
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        (kt.gemv_q5_k_q8_k)(
            black_box(&w),
            black_box(&x_q8k),
            &mut out,
            rows,
            cols_blocks,
        );
    }
    let secs = t0.elapsed().as_secs_f64();
    let gflops = (2.0 * (rows * cols * iters) as f64) / secs / 1e9;
    println!(
        "gemv_q5_k_q8_k 2048x2048 (isa={}): {gflops:.1} GFLOP/s (out[0]={:.3})",
        kt.isa.name(),
        out[0]
    );
}

fn main() {
    bench_mamba2_scan();
    bench_q4_0_gemv();
    bench_q2_0_gemv();
    bench_nvfp4_gemv();
    bench_q5_k_gemv();
    bench_f32_q8_0_gemv_gemm();
    bench_f16_kv_ops();
}
