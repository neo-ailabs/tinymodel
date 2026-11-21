//! L1 GPU parity tests: `tm-vulkan` Q4_0 x F32 GEMV / multi-token GEMV vs the scalar
//! `tm-core` reference, for both weight layouts.
//!
//! Run on a specific device with `TM_VK_DEVICE=<index|name>` (e.g. `llvmpipe` for CI).
//! Tests print a notice and pass when no Vulkan device is available.

use tm_core::{BlockQ4_0, QK4_0, dequantize_row_q4_0, f32_to_f16};
use tm_vulkan::{Q4Layout, VulkanContext};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
    fn next_f32(&mut self) -> f32 {
        (((self.next_u64() % 2001) as f32) - 1000.0) / 500.0
    }
}

fn ctx() -> Option<VulkanContext> {
    match VulkanContext::new() {
        Ok(c) => {
            let i = c.info();
            eprintln!(
                "[tm-vulkan] device={} driver={} ({}) subgroup={} (control={}) coopmat={}",
                i.name,
                i.driver_name,
                i.driver_info,
                i.subgroup_size,
                i.subgroup_size_control,
                i.cooperative_matrix
            );
            Some(c)
        }
        Err(tm_vulkan::VulkanError::DeviceNotFound(sel)) => {
            panic!("TM_VK_DEVICE='{sel}' did not match any Vulkan device");
        }
        Err(e) => {
            eprintln!("[tm-vulkan] skipping: no usable Vulkan device ({e})");
            None
        }
    }
}

fn random_q4_0(rng: &mut Rng, n: usize) -> Vec<BlockQ4_0> {
    (0..n)
        .map(|i| {
            let mut qs = [0u8; 16];
            if i == 0 {
                for (j, b) in qs.iter_mut().enumerate() {
                    *b = (j as u8) | (((15 - j) as u8) << 4);
                }
            } else {
                for b in &mut qs {
                    *b = rng.next_u8();
                }
            }
            BlockQ4_0 {
                d: f32_to_f16(rng.next_f32() * 0.25),
                qs,
            }
        })
        .collect()
}

fn reference(
    w: &[BlockQ4_0],
    x: &[f32],
    rows: usize,
    cols: usize,
    n_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    let nb = cols / QK4_0;
    let mut out = vec![0.0f32; n_tok * rows];
    let mut l1 = vec![0.0f32; n_tok * rows];
    let mut deq = vec![0.0f32; cols];
    for r in 0..rows {
        dequantize_row_q4_0(&w[r * nb..(r + 1) * nb], &mut deq);
        for t in 0..n_tok {
            let xt = &x[t * cols..(t + 1) * cols];
            let (mut s, mut n) = (0.0f32, 0.0f32);
            for k in 0..cols {
                let p = deq[k] * xt[k];
                s += p;
                n += p.abs();
            }
            out[t * rows + r] = s;
            l1[t * rows + r] = n.max(1.0);
        }
    }
    (out, l1)
}

fn check(
    ctx: &VulkanContext,
    layout: Q4Layout,
    rows: usize,
    cols: usize,
    n_tok: usize,
    rng: &mut Rng,
) {
    let w = random_q4_0(rng, rows * cols / QK4_0);
    let x: Vec<f32> = (0..n_tok * cols).map(|_| rng.next_f32()).collect();
    let (expected, l1) = reference(&w, &x, rows, cols, n_tok);

    let wm = ctx.upload_q4_0(&w, rows, cols, layout).unwrap();
    let xb = ctx.upload(&x).unwrap();
    let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();
    ctx.gemm_q4_0_f32(&wm, &xb, &yb, n_tok, 1).unwrap();
    let got = ctx.read_f32(&yb, n_tok * rows).unwrap();
    for i in 0..n_tok * rows {
        let rel = (got[i] - expected[i]).abs() / l1[i];
        assert!(
            rel <= 1e-5,
            "{layout:?} mismatch rows={rows} cols={cols} n_tok={n_tok} i={i}: got={} expected={} rel={rel}",
            got[i],
            expected[i]
        );
    }
}

#[test]
fn vk_gemv_q4_0_parity() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(0xA11C_E500_1234_5678);
    for layout in [Q4Layout::Soa, Q4Layout::Aos] {
        for &(rows, cols) in &[
            (1usize, 32usize),
            (31, 64),
            (33, 256),
            (129, 1024),
            (515, 4096),
        ] {
            check(&ctx, layout, rows, cols, 1, &mut rng);
        }
    }
}

#[test]
fn vk_gemm_q4_0_multi_token_parity() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(0xBEEF_F00D_8765_4321);
    for layout in [Q4Layout::Soa, Q4Layout::Aos] {
        for n_tok in 1..=8 {
            check(&ctx, layout, 67, 512, n_tok, &mut rng);
        }
    }
}

#[test]
fn vk_rejects_bad_ntok() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(7);
    let w = random_q4_0(&mut rng, 4);
    let wm = ctx.upload_q4_0(&w, 4, 32, Q4Layout::Soa).unwrap();
    let xb = ctx.alloc_buffer(9 * 32 * 4).unwrap();
    let yb = ctx.alloc_buffer(9 * 4 * 4).unwrap();
    assert!(ctx.gemm_q4_0_f32(&wm, &xb, &yb, 9, 1).is_err());
    assert!(ctx.gemm_q4_0_f32(&wm, &xb, &yb, 0, 1).is_err());
}
