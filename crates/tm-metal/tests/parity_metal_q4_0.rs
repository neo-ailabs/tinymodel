//! L1 GPU parity tests: compare `tm-metal` `Q4_0 x F32` GEMV and batched GEMM
//! against the scalar `tm-core` reference.

#![cfg(target_vendor = "apple")]

use tm_core::{BlockQ4_0, QK4_0, dequantize_row_q4_0, f32_to_f16};
use tm_metal::MetalContext;

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

fn ref_gemm_q4_0_f32(
    w: &[BlockQ4_0],
    x: &[f32],
    rows: usize,
    cols: usize,
    n_tokens: usize,
) -> (Vec<f32>, Vec<f32>) {
    let cols_blocks = cols / QK4_0;
    let mut out = vec![0.0f32; n_tokens * rows];
    let mut l1 = vec![0.0f32; n_tokens * rows];
    let mut deq = vec![0.0f32; cols];

    for r in 0..rows {
        dequantize_row_q4_0(&w[r * cols_blocks..(r + 1) * cols_blocks], &mut deq);
        for t in 0..n_tokens {
            let xt = &x[t * cols..(t + 1) * cols];
            let mut sum = 0.0f32;
            let mut norm = 0.0f32;
            for k in 0..cols {
                let p = deq[k] * xt[k];
                sum += p;
                norm += p.abs();
            }
            out[t * rows + r] = sum;
            l1[t * rows + r] = norm.max(1.0);
        }
    }
    (out, l1)
}

#[test]
fn metal_gemv_q4_0_f32_parity() {
    let ctx = MetalContext::new().expect("MetalContext::new failed");
    let mut rng = Rng(0xA11C_E500_1234_5678);

    for &(rows, cols) in &[(1usize, 32usize), (31, 64), (33, 256), (129, 1024)] {
        let cols_blocks = cols / QK4_0;
        let w: Vec<BlockQ4_0> = (0..rows * cols_blocks)
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
            .collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.next_f32()).collect();

        let (expected, l1) = ref_gemm_q4_0_f32(&w, &x, rows, cols, 1);

        let w_buf = ctx.buffer_from_slice(&w).unwrap();
        let x_buf = ctx.buffer_from_slice(&x).unwrap();
        let mut y_buf = ctx
            .alloc_buffer(rows * core::mem::size_of::<f32>())
            .unwrap();

        ctx.gemv_q4_0_f32(&w_buf, &x_buf, &mut y_buf, rows, cols)
            .unwrap();
        let got = y_buf.as_f32_slice();

        for r in 0..rows {
            let rel = (got[r] - expected[r]).abs() / l1[r];
            assert!(
                rel <= 1e-5,
                "Metal GEMV mismatch rows={rows} cols={cols} r={r}: got={} expected={} rel={rel}",
                got[r],
                expected[r]
            );
        }
    }
}

#[test]
fn metal_gemm_q4_0_f32_batched_parity() {
    let ctx = MetalContext::new().expect("MetalContext::new failed");
    let mut rng = Rng(0xBEEF_F00D_8765_4321);

    let rows = 67;
    let cols = 512;
    let cols_blocks = cols / QK4_0;
    let w: Vec<BlockQ4_0> = (0..rows * cols_blocks)
        .map(|_| {
            let mut qs = [0u8; 16];
            for b in &mut qs {
                *b = rng.next_u8();
            }
            BlockQ4_0 {
                d: f32_to_f16(rng.next_f32() * 0.25),
                qs,
            }
        })
        .collect();

    for n_tok in [1usize, 2, 4, 8] {
        let x: Vec<f32> = (0..n_tok * cols).map(|_| rng.next_f32()).collect();
        let (expected, l1) = ref_gemm_q4_0_f32(&w, &x, rows, cols, n_tok);

        let w_buf = ctx.buffer_from_slice(&w).unwrap();
        let x_buf = ctx.buffer_from_slice(&x).unwrap();
        let mut y_buf = ctx
            .alloc_buffer(n_tok * rows * core::mem::size_of::<f32>())
            .unwrap();

        ctx.gemm_q4_0_f32_repeat(&w_buf, &x_buf, &mut y_buf, rows, cols, n_tok, 1)
            .unwrap();
        let got = y_buf.as_f32_slice();

        for i in 0..n_tok * rows {
            let rel = (got[i] - expected[i]).abs() / l1[i];
            assert!(
                rel <= 1e-5,
                "Metal batched GEMM mismatch n_tok={n_tok} i={i}: got={} expected={} rel={rel}",
                got[i],
                expected[i]
            );
        }
    }
}
