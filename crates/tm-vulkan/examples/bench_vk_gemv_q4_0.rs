//! Micro-benchmark for `tm-vulkan` Q4_0 x F32 GEMV (SoA vs AoS layouts).
//!
//! Run: `cargo run --release -p tm-vulkan --example bench_vk_gemv_q4_0`
//! (select a device with `TM_VK_DEVICE=<index|name>`).

use std::time::Instant;

use tm_core::{BlockQ4_0, QK4_0, f32_to_f16};
use tm_vulkan::{Q4Layout, VulkanContext};

fn main() {
    // Optional geometry override: bench_vk_gemv_q4_0 <wg> <nr>
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let ctx = VulkanContext::new().expect("failed to initialize Vulkan");
    let i = ctx.info();
    println!(
        "Vulkan GPU: {} [{}] driver={} subgroup={} VRAM={:.1} GiB",
        i.name,
        i.device_type,
        i.driver_name,
        i.subgroup_size,
        i.device_local_bytes as f64 / (1u64 << 30) as f64
    );
    if let [wg, nr] = args[..] {
        ctx.set_q4_0_gemv_config(wg, nr).expect("bad gemv config");
        println!("gemv config: wg={wg} nr={nr}");
    }
    println!(
        "{:<8} {:>18} {:>9} {:>9} {:>9}",
        "layout", "shape (MxKxN)", "us/iter", "GB/s", "GOPS"
    );
    println!("{}", "-".repeat(58));

    let shapes = [
        (2048usize, 2048usize, 1usize),
        (4096, 4096, 1),
        (14336, 4096, 1),
        (4096, 14336, 1),
        (151936, 1024, 1),
        (65536, 4096, 1),
        (4096, 4096, 4),
        (4096, 4096, 8),
        (4096, 14336, 4),
        (4096, 14336, 8),
    ];
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    for &(rows, cols, n_tok) in &shapes {
        let w: Vec<BlockQ4_0> = (0..rows * cols / QK4_0)
            .map(|_| {
                let mut qs = [0u8; 16];
                for b in &mut qs {
                    *b = (next() & 0xFF) as u8;
                }
                BlockQ4_0 {
                    d: f32_to_f16(0.03125),
                    qs,
                }
            })
            .collect();
        let x: Vec<f32> = (0..n_tok * cols)
            .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
            .collect();
        let xb = ctx.upload(&x).unwrap();
        let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();
        for layout in [Q4Layout::Soa, Q4Layout::Aos] {
            let wm = ctx.upload_q4_0(&w, rows, cols, layout).unwrap();
            ctx.gemm_q4_0_f32(&wm, &xb, &yb, n_tok, 10).unwrap();
            let repeat = 200;
            let t0 = Instant::now();
            ctx.gemm_q4_0_f32(&wm, &xb, &yb, n_tok, repeat).unwrap();
            let dt = t0.elapsed().as_secs_f64() / repeat as f64;
            let gbs = wm.weight_bytes() as f64 / dt / 1e9;
            let gops = 2.0 * (rows * cols * n_tok) as f64 / dt / 1e9;
            println!(
                "{:<8} {:>18} {:>9.1} {:>9.1} {:>9.1}",
                format!("{layout:?}"),
                format!("{rows}x{cols}x{n_tok}"),
                dt * 1e6,
                gbs,
                gops
            );
        }
    }
}
