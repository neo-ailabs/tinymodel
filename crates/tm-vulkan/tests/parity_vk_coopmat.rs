//! L1 / L1b GPU parity tests for V3 prefill kernels (`AGENTS.md` §5 Rule 3 & `TESTING.md` §5):
//! - `fa_gqa_tiled` (portable LDS-tiled causal GQA FlashAttention) across all drivers (`radv`, `UHD`, `llvmpipe`).
//! - `gemm_coopmat_f16` (`16x16x16 f16*f16+f32->f32`) across `Q4_0`, `Q4_1`, `Q8_0`, `Q5_K`, `Q6_K`, `F32`,
//!   both `BN = 64` and `BN = 128`, non-multiple edge shapes, and fused `add_residual`.
//! - `gemm_pair_silu_coopmat_f16` (`Gate + Up + SwiGLU` fused `16x16x16` CoopMat GEMM).
//! - `gemm_coopmat_i8` (`16x16x16 i8*i8+i32->i32`): both **bit-exact** integer-scaled parity (`0.0` error)
//!   and dynamic `Q8_1` activation-quantized parity against a scalar `Q8_1` reference.
//! - `fa_coopmat` (`16x16x16` CoopMat causal FlashAttention) across `head_dim = 64, 128, 256`.

use tm_core::{
    BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, DType, dequantize_row_bytes, f16_to_f32,
    f32_to_f16,
};
use tm_vulkan::VulkanContext;

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
    fn u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
    fn f32(&mut self) -> f32 {
        (((self.next_u64() % 2001) as f32) - 1000.0) / 500.0
    }
    fn f16_exact(&mut self) -> f32 {
        f16_to_f32(f32_to_f16(self.f32()))
    }
}

fn get_ctx_or_skip() -> Option<VulkanContext> {
    match VulkanContext::new() {
        Ok(c) => Some(c),
        Err(tm_vulkan::VulkanError::DeviceNotFound(sel)) => {
            panic!("TM_VK_DEVICE='{sel}' did not match any Vulkan device");
        }
        Err(e) => {
            eprintln!("[tm-vulkan] skipping V3 kernel parity ({e})");
            None
        }
    }
}

fn make_weight_bytes(rng: &mut Rng, dt: DType, rows: usize, cols: usize) -> Vec<u8> {
    match dt {
        DType::Q4_0 => {
            let nb = rows * cols / 32;
            let mut out = Vec::with_capacity(nb * 18);
            for _ in 0..nb {
                let b = BlockQ4_0 {
                    d: f32_to_f16(rng.f32() * 0.125),
                    qs: core::array::from_fn(|_| rng.u8()),
                };
                out.extend_from_slice(&b.d.to_le_bytes());
                out.extend_from_slice(&b.qs);
            }
            out
        }
        DType::Q4_1 => {
            let nb = rows * cols / 32;
            let mut out = Vec::with_capacity(nb * 20);
            for _ in 0..nb {
                let b = BlockQ4_1 {
                    d: f32_to_f16(rng.f32() * 0.125),
                    m: f32_to_f16(rng.f32() * 0.05),
                    qs: core::array::from_fn(|_| rng.u8()),
                };
                out.extend_from_slice(&b.d.to_le_bytes());
                out.extend_from_slice(&b.m.to_le_bytes());
                out.extend_from_slice(&b.qs);
            }
            out
        }
        DType::Q8_0 => {
            let nb = rows * cols / 32;
            let mut out = Vec::with_capacity(nb * 34);
            for _ in 0..nb {
                let d_pow2 = 0.03125 * ((1 + (rng.u8() & 3)) as f32);
                let b = BlockQ8_0 {
                    d: f32_to_f16(d_pow2),
                    qs: core::array::from_fn(|_| rng.u8() as i8),
                };
                out.extend_from_slice(&b.d.to_le_bytes());
                for &q in &b.qs {
                    out.push(q as u8);
                }
            }
            out
        }
        DType::Q5K => {
            let nb = rows * cols / 256;
            let mut out = Vec::with_capacity(nb * 176);
            for _ in 0..nb {
                let b = BlockQ5K {
                    d: f32_to_f16(0.015625),
                    dmin: f32_to_f16(0.0078125),
                    scales: core::array::from_fn(|_| rng.u8()),
                    qh: core::array::from_fn(|_| rng.u8()),
                    qs: core::array::from_fn(|_| rng.u8()),
                };
                out.extend_from_slice(&b.d.to_le_bytes());
                out.extend_from_slice(&b.dmin.to_le_bytes());
                out.extend_from_slice(&b.scales);
                out.extend_from_slice(&b.qh);
                out.extend_from_slice(&b.qs);
            }
            out
        }
        DType::Q6K => {
            let nb = rows * cols / 256;
            let mut out = Vec::with_capacity(nb * 210);
            for _ in 0..nb {
                let b = BlockQ6K {
                    ql: core::array::from_fn(|_| rng.u8()),
                    qh: core::array::from_fn(|_| rng.u8()),
                    scales: core::array::from_fn(|_| ((rng.u8() % 15) as i8) - 7),
                    d: f32_to_f16(0.015625),
                };
                out.extend_from_slice(&b.ql);
                out.extend_from_slice(&b.qh);
                for &sc in &b.scales {
                    out.push(sc as u8);
                }
                out.extend_from_slice(&b.d.to_le_bytes());
            }
            out
        }
        DType::F32 => {
            let mut out = Vec::with_capacity(rows * cols * 4);
            for _ in 0..rows * cols {
                let v = rng.f16_exact();
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        other => panic!("unexpected dtype {other:?}"),
    }
}

fn ref_gemm_f16w(
    dt: DType,
    w_bytes: &[u8],
    x: &[f32],
    init_y: Option<&[f32]>,
    rows: usize,
    cols: usize,
    n_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    let rb = dt.row_bytes(cols);
    let mut deq = vec![0.0f32; cols];
    let mut out = vec![0.0f32; n_tok * rows];
    let mut l1 = vec![0.0f32; n_tok * rows];
    for r in 0..rows {
        dequantize_row_bytes(dt, &w_bytes[r * rb..(r + 1) * rb], &mut deq);
        for v in &mut deq {
            *v = f16_to_f32(f32_to_f16(*v));
        }
        for t in 0..n_tok {
            let xt = &x[t * cols..(t + 1) * cols];
            let mut s = 0.0f32;
            let mut n = 0.0f32;
            for k in 0..cols {
                let xk = f16_to_f32(f32_to_f16(xt[k]));
                let p = deq[k] * xk;
                s += p;
                n += p.abs();
            }
            if let Some(res) = init_y {
                s += res[t * rows + r];
                n += res[t * rows + r].abs();
            }
            out[t * rows + r] = s;
            l1[t * rows + r] = n.max(1.0);
        }
    }
    (out, l1)
}

#[allow(clippy::too_many_arguments)]
fn ref_flash_attn(
    q: &[f32],
    k_u16: &[u16],
    v_u16: &[u16],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    base_pos: usize,
    n_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let group = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_tok * q_dim];
    let mut l1 = vec![1.0f32; n_tok * q_dim];

    for tok in 0..n_tok {
        let seq_len = base_pos + tok + 1;
        for h in 0..n_head {
            let kv_h = h / group;
            let qv = &q[tok * q_dim + h * head_dim..tok * q_dim + (h + 1) * head_dim];
            let mut scores = vec![0.0f32; seq_len];
            let mut m = f32::NEG_INFINITY;
            for (t, sc) in scores.iter_mut().enumerate() {
                let kb = t * kv_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += (qv[d] * scale) * f16_to_f32(k_u16[kb + d]);
                }
                *sc = dot;
                m = m.max(dot);
            }
            let mut denom = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            let ob = tok * q_dim + h * head_dim;
            let mut head_norm = 0.0f32;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (t, &w) in scores.iter().enumerate() {
                    acc += (w / denom) * f16_to_f32(v_u16[t * kv_dim + kv_h * head_dim + d]);
                }
                out[ob + d] = acc;
                head_norm += acc.abs();
            }
            for d in 0..head_dim {
                l1[ob + d] = head_norm.max(1.0);
            }
        }
    }
    (out, l1)
}

#[test]
fn vk_fa_gqa_tiled_fallback_parity() {
    let Some(ctx) = get_ctx_or_skip() else { return };
    let mut rng = Rng(0xFA11_BAC0_1234_5678);
    for &(n_head, n_head_kv, head_dim, base_pos, n_tok) in &[
        (4usize, 2usize, 64usize, 0usize, 1usize),
        (8, 2, 64, 19, 35),
        (4, 1, 128, 0, 67),
        (4, 2, 256, 7, 19),
    ] {
        let max_kv = base_pos + n_tok + 4;
        let q: Vec<f32> = (0..n_tok * n_head * head_dim)
            .map(|_| rng.f16_exact())
            .collect();
        let k_u16: Vec<u16> = (0..max_kv * n_head_kv * head_dim)
            .map(|_| f32_to_f16(rng.f32() * 0.5))
            .collect();
        let v_u16: Vec<u16> = (0..max_kv * n_head_kv * head_dim)
            .map(|_| f32_to_f16(rng.f32() * 0.5))
            .collect();
        let (expected, l1) = ref_flash_attn(
            &q, &k_u16, &v_u16, n_head, n_head_kv, head_dim, base_pos, n_tok,
        );

        let qb = ctx.upload(&q).unwrap();
        let kb = ctx.upload(&k_u16).unwrap();
        let vb = ctx.upload(&v_u16).unwrap();
        let ob = ctx
            .alloc_buffer((n_tok * n_head * head_dim * 4) as u64)
            .unwrap();

        ctx.flash_attn_batch_raw(
            &qb, &kb, &vb, &ob, n_head, n_head_kv, head_dim, base_pos, n_tok, false, 1,
        )
        .unwrap();
        let got = ctx.read_f32(&ob, expected.len()).unwrap();
        for i in 0..expected.len() {
            let rel = (got[i] - expected[i]).abs() / l1[i];
            assert!(
                rel <= 1e-5,
                "fa_gqa_tiled mismatch hd={head_dim} n_tok={n_tok} i={i}: got={} expected={} rel={rel}",
                got[i],
                expected[i]
            );
        }
    }
}

#[test]
fn vk_coopmat_f16_gemm_and_fused_swiglu_and_fa_parity() {
    let Some(ctx) = get_ctx_or_skip() else { return };
    if !ctx.info().coopmat_f16_f32acc {
        eprintln!(
            "[tm-vulkan] skipping CoopMat F16 parity on {} (coopmat_f16_f32acc=false)",
            ctx.device_name()
        );
        return;
    }

    let mut rng = Rng(0xC009_FA16_9060_1200);
    // 1. Test `gemm_coopmat_f16` across all 6 dtypes, BN={64,128}, edge shapes, and residual add.
    for &(dt, rows, cols, n_tok, bn, add_res) in &[
        (DType::Q4_0, 65usize, 256usize, 17usize, 64u32, false),
        (DType::Q4_0, 256, 512, 67, 128, true),
        (DType::Q4_1, 131, 256, 35, 64, true),
        (DType::Q8_0, 192, 256, 64, 128, false),
        (DType::Q5K, 97, 256, 33, 64, false),
        (DType::Q6K, 128, 512, 65, 128, true),
        (DType::F32, 71, 128, 19, 64, false),
    ] {
        let w_bytes = make_weight_bytes(&mut rng, dt, rows, cols);
        let x: Vec<f32> = (0..n_tok * cols).map(|_| rng.f16_exact()).collect();
        let res_init: Vec<f32> = (0..n_tok * rows).map(|_| rng.f16_exact()).collect();
        let (expected, l1) = ref_gemm_f16w(
            dt,
            &w_bytes,
            &x,
            add_res.then_some(res_init.as_slice()),
            rows,
            cols,
            n_tok,
        );

        let wb = ctx.upload_bytes(&w_bytes).unwrap();
        let xb = ctx.upload(&x).unwrap();
        let yb = if add_res {
            ctx.upload(&res_init).unwrap()
        } else {
            ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap()
        };

        ctx.gemm_coopmat_f16_raw(dt, &wb, &xb, &yb, rows, cols, n_tok, bn, add_res, 1)
            .unwrap();
        let got = ctx.read_f32(&yb, expected.len()).unwrap();
        for i in 0..expected.len() {
            let rel = (got[i] - expected[i]).abs() / l1[i];
            assert!(
                rel <= 1e-5,
                "gemm_coopmat_f16 {dt:?} bn={bn} res={add_res} {rows}x{cols}x{n_tok} i={i}: got={} expected={} rel={rel}",
                got[i],
                expected[i]
            );
        }
    }

    // 2. Test `gemm_pair_silu_coopmat_f16` across Q4_0, Q4_1, Q8_0.
    for dt in [DType::Q4_0, DType::Q4_1, DType::Q8_0] {
        let (rows, cols, n_tok) = (99usize, 256usize, 67usize);
        let wg_bytes = make_weight_bytes(&mut rng, dt, rows, cols);
        let wu_bytes = make_weight_bytes(&mut rng, dt, rows, cols);
        let x: Vec<f32> = (0..n_tok * cols).map(|_| rng.f16_exact()).collect();
        let (g_ref, _) = ref_gemm_f16w(dt, &wg_bytes, &x, None, rows, cols, n_tok);
        let (u_ref, _) = ref_gemm_f16w(dt, &wu_bytes, &x, None, rows, cols, n_tok);
        let expected: Vec<f32> = g_ref
            .iter()
            .zip(&u_ref)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let norm = expected.iter().map(|v| v.abs()).fold(1.0f32, f32::max);

        let wgb = ctx.upload_bytes(&wg_bytes).unwrap();
        let wub = ctx.upload_bytes(&wu_bytes).unwrap();
        let xb = ctx.upload(&x).unwrap();
        let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();
        ctx.gemm_pair_silu_coopmat_f16_raw(dt, &wgb, &wub, &xb, &yb, rows, cols, n_tok, 1)
            .unwrap();
        let got = ctx.read_f32(&yb, expected.len()).unwrap();
        for i in 0..expected.len() {
            let rel = (got[i] - expected[i]).abs() / norm;
            assert!(
                rel <= 2e-5,
                "gemm_pair_silu_coopmat_f16 {dt:?} i={i}: got={} expected={} rel={rel}",
                got[i],
                expected[i]
            );
        }
    }

    // 3. Test `fa_coopmat` across head_dim = 64, 128, 256 (wave32).
    if ctx.info().subgroup_size == 32 {
        for &(n_head, n_head_kv, head_dim, base_pos, n_tok) in &[
            (4usize, 2usize, 64usize, 0usize, 1usize),
            (8, 2, 64, 23, 65),
            (4, 1, 128, 0, 99),
            (4, 2, 256, 11, 35),
        ] {
            let max_kv = base_pos + n_tok + 4;
            let q: Vec<f32> = (0..n_tok * n_head * head_dim)
                .map(|_| rng.f16_exact() * 0.5)
                .collect();
            let k_u16: Vec<u16> = (0..max_kv * n_head_kv * head_dim)
                .map(|_| f32_to_f16(rng.f32() * 0.5))
                .collect();
            let v_u16: Vec<u16> = (0..max_kv * n_head_kv * head_dim)
                .map(|_| f32_to_f16(rng.f32() * 0.5))
                .collect();
            let (expected, l1) = ref_flash_attn(
                &q, &k_u16, &v_u16, n_head, n_head_kv, head_dim, base_pos, n_tok,
            );

            let qb = ctx.upload(&q).unwrap();
            let kb = ctx.upload(&k_u16).unwrap();
            let vb = ctx.upload(&v_u16).unwrap();
            let ob = ctx
                .alloc_buffer((n_tok * n_head * head_dim * 4) as u64)
                .unwrap();

            ctx.flash_attn_batch_raw(
                &qb, &kb, &vb, &ob, n_head, n_head_kv, head_dim, base_pos, n_tok, true, 1,
            )
            .unwrap();
            let got = ctx.read_f32(&ob, expected.len()).unwrap();
            for i in 0..expected.len() {
                let rel = (got[i] - expected[i]).abs() / l1[i];
                assert!(
                    rel <= 5e-4,
                    "fa_coopmat mismatch hd={head_dim} n_tok={n_tok} i={i}: got={} expected={} rel={rel}",
                    got[i],
                    expected[i]
                );
            }
        }
    }
}

#[test]
fn vk_coopmat_i8_bit_exact_and_q8_1_parity() {
    let Some(ctx) = get_ctx_or_skip() else { return };
    if !ctx.info().coopmat_i8_i32acc {
        eprintln!(
            "[tm-vulkan] skipping CoopMat INT8 parity on {} (coopmat_i8_i32acc=false)",
            ctx.device_name()
        );
        return;
    }

    let mut rng = Rng(0x1818_3232_9060_BEEF);
    // 1. Bit-exact (`0.0` error) integer WMMA verification (`TESTING.md` §5):
    //    Use integer-valued X with block max = 127.0 (`d_x == 1.0` exact!) and `d_w == 0.5`!
    let (rows, cols, n_tok) = (67usize, 128usize, 65usize);
    let nb_x = n_tok * cols / 32;
    let mut x_exact = vec![0.0f32; n_tok * cols];
    for blk in 0..nb_x {
        x_exact[blk * 32] = 127.0;
        for j in 1..32 {
            x_exact[blk * 32 + j] = ((rng.u8() % 61) as i32 - 30) as f32;
        }
    }
    let nb_w = rows * cols / 32;
    let mut w_q8_bytes = Vec::with_capacity(nb_w * 34);
    let mut w_i8 = vec![0i32; rows * cols];
    for b_idx in 0..nb_w {
        let d_f16 = f32_to_f16(0.5);
        w_q8_bytes.extend_from_slice(&d_f16.to_le_bytes());
        for j in 0..32 {
            let q = ((rng.u8() % 61) as i8) - 30;
            w_q8_bytes.push(q as u8);
            w_i8[b_idx * 32 + j] = q as i32;
        }
    }

    let wb = ctx.upload_bytes(&w_q8_bytes).unwrap();
    let xb = ctx.upload(&x_exact).unwrap();
    let xqb = ctx.alloc_buffer((nb_x * 32) as u64).unwrap();
    let xdb = ctx.alloc_buffer((nb_x * 8) as u64).unwrap();
    let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();

    ctx.gemm_coopmat_i8_raw(
        DType::Q8_0,
        &wb,
        &xb,
        &xqb,
        &xdb,
        &yb,
        rows,
        cols,
        n_tok,
        false,
        1,
    )
    .unwrap();
    let got = ctx.read_f32(&yb, n_tok * rows).unwrap();
    for t in 0..n_tok {
        for r in 0..rows {
            let mut exact_isum = 0i64;
            for k in 0..cols {
                exact_isum += (x_exact[t * cols + k] as i64) * (w_i8[r * cols + k] as i64);
            }
            let expected = (exact_isum as f32) * 0.5;
            assert_eq!(
                got[t * rows + r],
                expected,
                "INT8 coopmat bit-exact mismatch at t={t}, r={r}"
            );
        }
    }

    // 2. Arbitrary random Q4_0 / Q4_1 / Q8_0 weights + random FP32 X vs scalar Q8_1 reference (`rel <= 1e-5`).
    for dt in [DType::Q4_0, DType::Q4_1, DType::Q8_0] {
        let w_bytes = make_weight_bytes(&mut rng, dt, rows, cols);
        let x: Vec<f32> = (0..n_tok * cols).map(|_| rng.f32()).collect();

        // Scalar Q8_1 reference (`127.0 * inv_amax` matches GPU `v * (127.0 / amax)` within <= 1 LSB on .5 ties)
        let wb = ctx.upload_bytes(&w_bytes).unwrap();
        let xb = ctx.upload(&x).unwrap();
        let yb = ctx.alloc_buffer((n_tok * rows * 4) as u64).unwrap();
        ctx.gemm_coopmat_i8_raw(dt, &wb, &xb, &xqb, &xdb, &yb, rows, cols, n_tok, false, 1)
            .unwrap();
        let got = ctx.read_f32(&yb, n_tok * rows).unwrap();

        // Read back GPU-quantized Q8_1 (`xqb`, `xdb`) to verify `quantize_q8_1` (<= 1 LSB on rcp tie-breaks)
        // AND verify `gemm_coopmat_i8` itself to `rel <= 1e-5`!
        let xq_words_f32 = ctx.read_f32(&xqb, nb_x * 8).unwrap();
        let xd_f32 = ctx.read_f32(&xdb, nb_x * 2).unwrap();
        let mut x_q8_gpu = vec![0i8; n_tok * cols];
        for blk in 0..nb_x {
            let sl = &x[blk * 32..(blk + 1) * 32];
            let amax = sl.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let sum_x: f32 = sl.iter().sum();
            let d_cpu = amax / 127.0;
            let d_gpu = xd_f32[blk * 2];
            let sum_gpu = xd_f32[blk * 2 + 1];
            assert!(
                (d_gpu - d_cpu).abs() <= 1e-6,
                "d mismatch: {d_gpu} vs {d_cpu}"
            );
            assert!(
                (sum_gpu - sum_x).abs() <= 1e-4,
                "sum mismatch: {sum_gpu} vs {sum_x}"
            );
            let inv_d = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            for w_idx in 0..8 {
                let packed = xq_words_f32[blk * 8 + w_idx].to_bits().to_le_bytes();
                for (b_sub, &pb) in packed.iter().enumerate() {
                    let j = w_idx * 4 + b_sub;
                    let q_gpu = pb as i8;
                    let q_cpu = (sl[j] * inv_d).round_ties_even().clamp(-127.0, 127.0) as i8;
                    assert!(
                        (q_gpu as i32 - q_cpu as i32).abs() <= 1,
                        "q8_1 mismatch > 1 LSB: gpu={q_gpu} cpu={q_cpu}"
                    );
                    x_q8_gpu[blk * 32 + j] = q_gpu;
                }
            }
        }

        let rb = dt.row_bytes(cols);
        for r in 0..rows {
            let row_raw = &w_bytes[r * rb..(r + 1) * rb];
            for t in 0..n_tok {
                let mut ref_acc = 0.0f32;
                let mut ref_l1 = 0.0f32;
                let nb_row = cols / 32;
                for ib in 0..nb_row {
                    let xblk = t * nb_row + ib;
                    let dx = xd_f32[xblk * 2];
                    let sx = xd_f32[xblk * 2 + 1];
                    let xq = &x_q8_gpu[xblk * 32..(xblk + 1) * 32];
                    let term = match dt {
                        DType::Q4_0 => {
                            let bo = ib * 18;
                            let dw = f16_to_f32(u16::from_le_bytes([row_raw[bo], row_raw[bo + 1]]));
                            let mut qsum = 0i32;
                            for j in 0..16 {
                                let byte = row_raw[bo + 2 + j];
                                qsum += ((byte & 15) as i32 - 8) * (xq[j] as i32);
                                qsum += ((byte >> 4) as i32 - 8) * (xq[j + 16] as i32);
                            }
                            (qsum as f32) * (dx * dw)
                        }
                        DType::Q4_1 => {
                            let bo = ib * 20;
                            let dw = f16_to_f32(u16::from_le_bytes([row_raw[bo], row_raw[bo + 1]]));
                            let mw =
                                f16_to_f32(u16::from_le_bytes([row_raw[bo + 2], row_raw[bo + 3]]));
                            let mut qsum = 0i32;
                            for j in 0..16 {
                                let byte = row_raw[bo + 4 + j];
                                qsum += ((byte & 15) as i32) * (xq[j] as i32);
                                qsum += ((byte >> 4) as i32) * (xq[j + 16] as i32);
                            }
                            (qsum as f32) * (dx * dw) + sx * mw
                        }
                        DType::Q8_0 => {
                            let bo = ib * 34;
                            let dw = f16_to_f32(u16::from_le_bytes([row_raw[bo], row_raw[bo + 1]]));
                            let mut qsum = 0i32;
                            for j in 0..32 {
                                qsum += (row_raw[bo + 2 + j] as i8 as i32) * (xq[j] as i32);
                            }
                            (qsum as f32) * (dx * dw)
                        }
                        _ => unreachable!(),
                    };
                    ref_acc += term;
                    ref_l1 += term.abs();
                }
                let got_v = got[t * rows + r];
                let rel = (got_v - ref_acc).abs() / ref_l1.max(1.0);
                assert!(
                    rel <= 1e-5,
                    "gemm_coopmat_i8 {dt:?} t={t} r={r}: got={got_v} ref={ref_acc} rel={rel}"
                );
            }
        }
    }
}
