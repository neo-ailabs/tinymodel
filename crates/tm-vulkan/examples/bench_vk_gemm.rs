//! V3 Prefill Kernel Microbenchmark (`AGENTS.md` §5 Rule 3):
//! Compares `gemm_tiled` (fallback), `gemm_coopmat_f16` (`BN=64` & `BN=128`),
//! fused `gemm_pair_silu_coopmat_f16`, `gemm_coopmat_i8` (`Q8_1` + `16x16x16 i8*i8->i32`),
//! and `fa_gqa_tiled` vs `fa_coopmat` on real `Llama-3.2-1B` and `Qwen3-8B` `pp512` geometries.

use std::time::Instant;

use tm_core::{BlockQ4_0, DType, f32_to_f16};
use tm_vulkan::VulkanContext;

fn make_q4_0(rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 32;
    let mut out = Vec::with_capacity(nb * 18);
    let b = BlockQ4_0 {
        d: f32_to_f16(0.03125),
        qs: [0x95; 16],
    };
    for _ in 0..nb {
        out.extend_from_slice(&b.d.to_le_bytes());
        out.extend_from_slice(&b.qs);
    }
    out
}

fn bench_op<F: FnMut()>(warmup: usize, repeat_submit: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..repeat_submit {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e6 / (repeat_submit as f64)
}

fn main() {
    let ctx = VulkanContext::new().expect("VulkanContext::new");
    let info = ctx.info();
    println!(
        "Device: {} ({} · {}) | subgroup={} | coopmat_f16={} | coopmat_i8={}",
        info.name,
        info.driver_name,
        info.driver_info,
        info.subgroup_size,
        info.coopmat_f16_f32acc,
        info.coopmat_i8_i32acc
    );
    println!();

    let repeat_in_cb = 8usize;
    let shapes = [
        (
            "Llama1B wq   (2048x512x2048)",
            2048usize,
            2048usize,
            512usize,
        ),
        ("Llama1B gate (8192x512x2048)", 8192, 2048, 512),
        ("Qwen8B  wq   (4096x512x4096)", 4096, 4096, 512),
        ("Qwen8B  gate (12288x512x4096)", 12288, 4096, 512),
        ("Baseline     (4096x512x14336)", 4096, 14336, 512),
    ];

    println!(
        "{:<30} {:>12} {:>14} {:>15} {:>14}",
        "GEMM Shape (Q4_0, M=512)",
        "Tiled FP32",
        "CoopMat 64x64",
        "CoopMat 128x128",
        "CoopMat INT8"
    );
    println!("{}", "-".repeat(88));

    for &(label, rows, cols, n_tok) in &shapes {
        let w_bytes = make_q4_0(rows, cols);
        let x = vec![0.125f32; n_tok * cols];
        let wb = ctx.upload_bytes(&w_bytes).unwrap();
        let xb = ctx.upload(&x).unwrap();
        let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();
        let flops = 2.0 * (rows as f64) * (cols as f64) * (n_tok as f64);

        let us_tiled = bench_op(2, 3, || {
            ctx.gemm_fallback_raw(
                DType::Q4_0,
                &wb,
                &xb,
                &yb,
                rows,
                cols,
                n_tok,
                false,
                repeat_in_cb,
            )
            .unwrap();
        }) / (repeat_in_cb as f64);
        let tf_tiled = (flops / 1e12) / (us_tiled * 1e-6);

        let s_f16_64 = if info.coopmat_f16_f32acc {
            let us = bench_op(2, 4, || {
                ctx.gemm_coopmat_f16_raw(
                    DType::Q4_0,
                    &wb,
                    &xb,
                    &yb,
                    rows,
                    cols,
                    n_tok,
                    64,
                    false,
                    repeat_in_cb,
                )
                .unwrap();
            }) / (repeat_in_cb as f64);
            format!("{:>6.1}us/{:>4.1}T", us, (flops / 1e12) / (us * 1e-6))
        } else {
            "n/a".to_string()
        };

        let s_f16_128 = if info.coopmat_f16_f32acc {
            let us = bench_op(2, 4, || {
                ctx.gemm_coopmat_f16_raw(
                    DType::Q4_0,
                    &wb,
                    &xb,
                    &yb,
                    rows,
                    cols,
                    n_tok,
                    128,
                    false,
                    repeat_in_cb,
                )
                .unwrap();
            }) / (repeat_in_cb as f64);
            format!("{:>6.1}us/{:>4.1}T", us, (flops / 1e12) / (us * 1e-6))
        } else {
            "n/a".to_string()
        };

        let s_i8 = if info.coopmat_i8_i32acc {
            let nb_x = n_tok * cols / 32;
            let xqb = ctx.alloc_buffer((nb_x * 32) as u64).unwrap();
            let xdb = ctx.alloc_buffer((nb_x * 8) as u64).unwrap();
            let us = bench_op(2, 4, || {
                ctx.gemm_coopmat_i8_raw(
                    DType::Q4_0,
                    &wb,
                    &xb,
                    &xqb,
                    &xdb,
                    &yb,
                    rows,
                    cols,
                    n_tok,
                    false,
                    repeat_in_cb,
                )
                .unwrap();
            }) / (repeat_in_cb as f64);
            format!("{:>6.1}us/{:>4.1}T", us, (flops / 1e12) / (us * 1e-6))
        } else {
            "n/a".to_string()
        };

        println!(
            "{:<30} {:>6.0}us/{:>4.1}T {:>14} {:>14} {:>14}",
            label, us_tiled, tf_tiled, s_f16_64, s_f16_128, s_i8
        );
    }

    if info.coopmat_f16_f32acc {
        println!();
        for &(label, rows, cols, n_tok) in &[
            ("Llama1B Fused Gate+Up+SiLU", 8192usize, 2048usize, 512usize),
            ("Qwen8B  Fused Gate+Up+SiLU", 12288, 4096, 512),
        ] {
            let wg = ctx.upload_bytes(&make_q4_0(rows, cols)).unwrap();
            let wu = ctx.upload_bytes(&make_q4_0(rows, cols)).unwrap();
            let xb = ctx.upload(&vec![0.125f32; n_tok * cols]).unwrap();
            let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();
            let flops = 4.0 * (rows as f64) * (cols as f64) * (n_tok as f64);
            let us = bench_op(2, 4, || {
                ctx.gemm_pair_silu_coopmat_f16_raw(
                    DType::Q4_0,
                    &wg,
                    &wu,
                    &xb,
                    &yb,
                    rows,
                    cols,
                    n_tok,
                    repeat_in_cb,
                )
                .unwrap();
            }) / (repeat_in_cb as f64);
            println!(
                "{:<30}: {:>7.1} us ({:>5.2} TFLOP/s effective incl. SwiGLU)",
                label,
                us,
                (flops / 1e12) / (us * 1e-6)
            );
        }
    }

    println!();
    println!(
        "{:<34} {:>16} {:>18}",
        "Causal FlashAttention (pp512)", "Tiled Subgroup", "CoopMat 16x16x16"
    );
    println!("{}", "-".repeat(72));
    for &(label, n_head, n_head_kv, hd) in &[
        ("Llama-3.2-1B (32h/8kv, hd=64)", 32usize, 8usize, 64usize),
        ("Qwen3-8B     (32h/8kv, hd=128)", 32, 8, 128),
    ] {
        let n_tok = 512usize;
        let q = ctx.upload(&vec![0.1f32; n_tok * n_head * hd]).unwrap();
        let k = ctx
            .upload(&vec![f32_to_f16(0.1); n_tok * n_head_kv * hd])
            .unwrap();
        let v = ctx
            .upload(&vec![f32_to_f16(0.1); n_tok * n_head_kv * hd])
            .unwrap();
        let out = ctx.alloc_buffer((n_tok * n_head * hd * 4) as u64).unwrap();
        let causal_flops = 4.0 * ((n_tok * (n_tok + 1) / 2) as f64) * ((n_head * hd) as f64);

        let us_tiled = bench_op(2, 4, || {
            ctx.flash_attn_batch_raw(&q, &k, &v, &out, n_head, n_head_kv, hd, 0, n_tok, false, 8)
                .unwrap();
        }) / 8.0;
        let s_cm = if info.coopmat_f16_f32acc && info.subgroup_size == 32 {
            let us_cm = bench_op(2, 4, || {
                ctx.flash_attn_batch_raw(
                    &q, &k, &v, &out, n_head, n_head_kv, hd, 0, n_tok, true, 8,
                )
                .unwrap();
            }) / 8.0;
            format!(
                "{:>6.1}us ({:>5.2}T)",
                us_cm,
                (causal_flops / 1e12) / (us_cm * 1e-6)
            )
        } else {
            "n/a".to_string()
        };
        println!(
            "{:<34} {:>6.1}us ({:>5.2}T) {:>18}",
            label,
            us_tiled,
            (causal_flops / 1e12) / (us_tiled * 1e-6),
            s_cm
        );
    }
}
