//! Micro-benchmark for `tm-metal` `Q4_0 x F32` GEMV and batched GEMM on Apple Silicon.
//!
//! Run:
//!   cargo run --release -p tm-metal --example bench_metal_gemv_q4_0

#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("bench_metal_gemv_q4_0 requires an Apple platform with Metal");
}

#[cfg(target_vendor = "apple")]
fn main() {
    use std::time::Instant;
    use tm_core::{BlockQ4_0, QK4_0, f32_to_f16};
    use tm_metal::MetalContext;

    let ctx = MetalContext::new().expect("failed to initialize Metal");
    println!(
        "Metal GPU: {} (max working set: {:.1} MiB)",
        ctx.device_name(),
        ctx.recommended_max_working_set_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!(
        "{:<14} {:>14} {:>8} {:>10} {:>10} {:>10}",
        "Kernel", "Shape (MxKxN)", "ms/iter", "GB/s", "GOPS", "checksum"
    );
    println!("{}", "-".repeat(72));

    let shapes = [
        (4096usize, 4096usize, 1usize),
        (16384, 4096, 1),
        (32768, 4096, 1),
        (4096, 4096, 4),
        (16384, 4096, 4),
    ];

    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    for &(rows, cols, n_tok) in &shapes {
        let cols_blocks = cols / QK4_0;
        let d_w = f32_to_f16(0.03125);
        let w: Vec<BlockQ4_0> = (0..rows * cols_blocks)
            .map(|_| {
                let mut qs = [0u8; 16];
                for b in &mut qs {
                    *b = (next() & 0xFF) as u8;
                }
                BlockQ4_0 { d: d_w, qs }
            })
            .collect();
        let x: Vec<f32> = (0..n_tok * cols)
            .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
            .collect();

        let w_buf = ctx.buffer_from_slice(&w).unwrap();
        let x_buf = ctx.buffer_from_slice(&x).unwrap();
        let mut y_buf = ctx
            .alloc_buffer(n_tok * rows * core::mem::size_of::<f32>())
            .unwrap();

        let weight_bytes = (rows * cols_blocks * core::mem::size_of::<BlockQ4_0>()) as f64;
        let ops = 2.0 * (rows as f64) * (cols as f64) * (n_tok as f64);
        let repeat = 50usize;

        // Warmup
        if n_tok == 1 {
            ctx.gemv_q4_0_f32_repeat(&w_buf, &x_buf, &mut y_buf, rows, cols, 5)
                .unwrap();
        } else {
            ctx.gemm_q4_0_f32_repeat(&w_buf, &x_buf, &mut y_buf, rows, cols, n_tok, 5)
                .unwrap();
        }

        let t0 = Instant::now();
        if n_tok == 1 {
            ctx.gemv_q4_0_f32_repeat(&w_buf, &x_buf, &mut y_buf, rows, cols, repeat)
                .unwrap();
        } else {
            ctx.gemm_q4_0_f32_repeat(&w_buf, &x_buf, &mut y_buf, rows, cols, n_tok, repeat)
                .unwrap();
        }
        let elapsed = t0.elapsed().as_secs_f64() / (repeat as f64);
        let gbs = (weight_bytes / elapsed) / 1e9;
        let gops = (ops / elapsed) / 1e9;
        let checksum: f32 = y_buf.as_f32_slice().iter().take(16).sum();

        println!(
            "{:<14} {:>14} {:>8.3} {:>10.2} {:>10.2} {:>10.2}",
            if n_tok == 1 {
                "gemv_lpr8_nr2"
            } else {
                "gemm_lpr8"
            },
            format!("{rows}x{cols}x{n_tok}"),
            elapsed * 1e3,
            gbs,
            gops,
            checksum
        );
    }
}
