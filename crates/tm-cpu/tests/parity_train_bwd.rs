//! L1 & L1b parity + FP64 central finite-difference Jacobian tests for `tm-cpu` backward kernels
//! (`AGENTS.md` Hard Rule #3 & `ADR-015`).

use tm_core::{
    DType, dequantize_q4_0x4_single_row, dequantize_row_bytes, f16_to_f32, f32_to_f16,
    quantize_row_to_bytes, repack_q4_0_bytes_to_q4_0x4,
};
use tm_cpu::{
    Isa, RopeType, ThreadPool, detect_caps, generic, gqa_attention_f16_kv_batch_parallel,
    gqa_attention_f16_kv_bwd_f32, matmul_q_transposed_f32, rms_norm_bwd_f32,
    rope_bwd_inplace_tables, rope_compute_cos_sin, rope_inplace_tables, select_kernels,
    swiglu_bwd_f32, swiglu_inplace,
};

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
    fn f32(&mut self, scale: f32) -> f32 {
        ((((self.next_u64() % 20001) as f32) - 10000.0) / 10000.0) * scale
    }
}

fn supported_isas() -> Vec<Isa> {
    let caps = detect_caps();
    let mut out = vec![Isa::Generic];
    if caps.neon {
        out.push(Isa::NeonV80);
    }
    if caps.dotprod {
        out.push(Isa::NeonDotprod);
    }
    if caps.i8mm {
        out.push(Isa::NeonI8mm);
    }
    out
}

#[test]
fn l1_and_l1b_transposed_matmul_bit_exact_all_isas_dtypes_threads_and_batches() {
    let mut rng = Rng(0x2026_1122_ABCD_EF01);
    let rows = 36usize; // multiple of 4 for Q4_0x4
    let cols = 256usize; // multiple of 256 for K-quants & low-bit
    let max_tok = 9usize; // odd > 8 to exercise 8-token tile + 2-token & 1-token NEON tails!

    let mut w_f32 = vec![0.0f32; rows * cols];
    for v in &mut w_f32 {
        *v = rng.f32(0.5);
    }
    let mut dy = vec![0.0f32; max_tok * rows];
    for v in &mut dy {
        *v = rng.f32(1.2);
    }
    let mut init_dx = vec![0.0f32; max_tok * cols];
    for v in &mut init_dx {
        *v = rng.f32(0.3);
    }

    let dtypes = [
        DType::Q4_0,
        DType::Q4_0x4,
        DType::Q4_1,
        DType::Q8_0,
        DType::Q5K,
        DType::Q6K,
        DType::Q2_0,
        DType::TQ2_0,
        DType::Iq2Xxs,
        DType::Iq3Xxs,
        DType::Iq4Nl,
        DType::F16,
        DType::BF16,
        DType::F32,
    ];

    let pool1 = ThreadPool::new(1);
    let pool4 = ThreadPool::new(4);
    let kt_gen = select_kernels(Isa::Generic);
    let isas = supported_isas();

    for &dt in &dtypes {
        // Encode `w_f32` (or construct valid raw blocks for read-only GGUF formats) into `dt`
        let w_bytes = match dt {
            DType::Q4_0x4 => {
                let rb = DType::Q4_0.row_bytes(cols);
                let mut raw = vec![0u8; rows * rb];
                for r in 0..rows {
                    quantize_row_to_bytes(
                        DType::Q4_0,
                        &w_f32[r * cols..(r + 1) * cols],
                        &mut raw[r * rb..(r + 1) * rb],
                    );
                }
                let mut rep = vec![0u8; rows * rb];
                repack_q4_0_bytes_to_q4_0x4(&raw, &mut rep, rows, cols);
                rep
            }
            DType::Q4_1 | DType::Q5K | DType::Q6K | DType::BF16 => {
                let rb = dt.row_bytes(cols);
                let mut buf = vec![0u8; rows * rb];
                // Fill with deterministic pseudo-random bytes and clamp f16 header scales to finite values
                for b in &mut buf {
                    *b = (rng.next_u64() & 0xFF) as u8;
                }
                let blk_elems = dt.block_elements();
                let blk_bytes = dt.block_bytes();
                let n_blks = (rows * cols) / blk_elems;
                for ib in 0..n_blks {
                    let off = ib * blk_bytes;
                    match dt {
                        DType::Q4_1 => {
                            let d = f32_to_f16(rng.f32(0.05)).to_le_bytes();
                            let m = f32_to_f16(rng.f32(0.05)).to_le_bytes();
                            buf[off..off + 2].copy_from_slice(&d);
                            buf[off + 2..off + 4].copy_from_slice(&m);
                        }
                        DType::Q5K => {
                            let d = f32_to_f16(rng.f32(0.05).abs()).to_le_bytes();
                            let dmin = f32_to_f16(rng.f32(0.05).abs()).to_le_bytes();
                            buf[off..off + 2].copy_from_slice(&d);
                            buf[off + 2..off + 4].copy_from_slice(&dmin);
                        }
                        DType::Q6K => {
                            let d = f32_to_f16(rng.f32(0.05)).to_le_bytes();
                            buf[off + 208..off + 210].copy_from_slice(&d);
                        }
                        DType::BF16 => {
                            let bits = rng.f32(0.5).to_bits();
                            let bf = ((bits >> 16) as u16).to_le_bytes();
                            buf[off..off + 2].copy_from_slice(&bf);
                        }
                        _ => {}
                    }
                }
                buf
            }
            _ => {
                let rb = dt.row_bytes(cols);
                let mut buf = vec![0u8; rows * rb];
                for r in 0..rows {
                    quantize_row_to_bytes(
                        dt,
                        &w_f32[r * cols..(r + 1) * cols],
                        &mut buf[r * rb..(r + 1) * rb],
                    );
                }
                buf
            }
        };

        // 1. Scalar `generic` kernel baseline (`1` thread) + explicit dequantized reference
        let mut gen_dx = init_dx.clone();
        matmul_q_transposed_f32(
            &pool1,
            kt_gen,
            dt,
            &w_bytes,
            &dy,
            &mut gen_dx,
            rows,
            cols,
            max_tok,
            true,
        );

        let mut explicit_dx = init_dx.clone();
        let mut row_dec = vec![0.0f32; cols];
        for r in 0..rows {
            if dt == DType::Q4_0x4 {
                dequantize_q4_0x4_single_row(&w_bytes, cols, r, &mut row_dec);
            } else {
                let rb = dt.row_bytes(cols);
                dequantize_row_bytes(dt, &w_bytes[r * rb..(r + 1) * rb], &mut row_dec);
            }
            for t in 0..max_tok {
                generic::axpy_f32(
                    &mut explicit_dx[t * cols..(t + 1) * cols],
                    dy[t * rows + r],
                    &row_dec,
                );
            }
        }
        for (&a, &b) in gen_dx.iter().zip(explicit_dx.iter()) {
            assert!(
                (a - b).abs() <= 1e-6,
                "explicit dequant vs generic drift for {dt:?}: {a} vs {b}"
            );
            if dt != DType::Q8_0 {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "expected 0-ULP vs explicit dequant on {dt:?}"
                );
            }
        }

        // 2. Verify every ISA on both 1 thread and 4 threads matches `gen_dx` BIT-FOR-BIT (`to_bits()`)!
        for &isa in &isas {
            let kt = select_kernels(isa);
            for pool in [&pool1, &pool4] {
                let mut got = init_dx.clone();
                matmul_q_transposed_f32(
                    pool, kt, dt, &w_bytes, &dy, &mut got, rows, cols, max_tok, true,
                );
                for (idx, (&a, &b)) in got.iter().zip(gen_dx.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "0-ULP mismatch for dtype={dt:?} isa={isa:?} threads={} at idx={idx}: got={a} ref={b}",
                        pool.num_threads()
                    );
                }
            }
        }

        // 3. L1b batch-width invariance: single-token `n_tok=1` vs batched `n_tok=max_tok` (`accum=false`)
        let kt_active = select_kernels(*isas.last().unwrap());
        let mut batched = vec![0.0f32; max_tok * cols];
        matmul_q_transposed_f32(
            &pool4,
            kt_active,
            dt,
            &w_bytes,
            &dy,
            &mut batched,
            rows,
            cols,
            max_tok,
            false,
        );
        for t in 0..max_tok {
            let mut single = vec![0.0f32; cols];
            matmul_q_transposed_f32(
                &pool1,
                kt_gen,
                dt,
                &w_bytes,
                &dy[t * rows..(t + 1) * rows],
                &mut single,
                rows,
                cols,
                1,
                false,
            );
            for c in 0..cols {
                assert_eq!(
                    single[c].to_bits(),
                    batched[t * cols + c].to_bits(),
                    "L1b batch-width mismatch for dtype={dt:?} token={t} col={c}"
                );
            }
        }
    }
}

#[test]
fn l1_backward_ops_match_fp64_central_finite_differences() {
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);
    let eps_fd = 1e-3f64;

    // 1. RMSNorm backward (`dx` and `dweight`) vs FP64 central finite difference
    let n = 64usize;
    let rms_eps = 1e-5f32;
    let x: Vec<f32> = (0..n).map(|_| rng.f32(1.1)).collect();
    let w: Vec<f32> = (0..n).map(|_| 1.0 + rng.f32(0.3)).collect();
    let dy: Vec<f32> = (0..n).map(|_| rng.f32(0.8)).collect();

    let mut dx = vec![0.0f32; n];
    let mut dw = vec![0.0f32; n];
    rms_norm_bwd_f32(&mut dx, Some(&mut dw), &x, &w, &dy, rms_eps);

    let eval_rms_loss = |xv: &[f64], wv: &[f64]| -> f64 {
        let ss: f64 = xv.iter().map(|&v| v * v).sum::<f64>() / (n as f64);
        let r = 1.0 / (ss + (rms_eps as f64)).sqrt();
        xv.iter()
            .zip(wv.iter())
            .zip(dy.iter())
            .map(|((&xi, &wi), &dyi)| xi * r * wi * (dyi as f64))
            .sum()
    };

    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let w64: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    for i in 0..n {
        let mut xp = x64.clone();
        let mut xm = x64.clone();
        xp[i] += eps_fd;
        xm[i] -= eps_fd;
        let num_dx = (eval_rms_loss(&xp, &w64) - eval_rms_loss(&xm, &w64)) / (2.0 * eps_fd);
        let err = ((dx[i] as f64) - num_dx).abs() / num_dx.abs().max(1.0);
        assert!(err < 1e-5, "rms_norm dx[{i}] err={err:e}");

        let mut wp = w64.clone();
        let mut wm = w64.clone();
        wp[i] += eps_fd;
        wm[i] -= eps_fd;
        let num_dw = (eval_rms_loss(&x64, &wp) - eval_rms_loss(&x64, &wm)) / (2.0 * eps_fd);
        let err_w = ((dw[i] as f64) - num_dw).abs() / num_dw.abs().max(1.0);
        assert!(err_w < 1e-5, "rms_norm dw[{i}] err={err_w:e}");
    }

    // 2. RoPE backward (`Norm` and `Neox`) exact transpose `<RoPE(x), dy> == <x, RoPE_bwd(dy)>`
    let n_heads = 4usize;
    let head_dim = 16usize;
    let rope_dim = 12usize; // partial RoPE!
    let freqs: Vec<f64> = (0..rope_dim / 2)
        .map(|i| 10000.0f64.powf(-2.0 * (i as f64) / (rope_dim as f64)))
        .collect();
    let mut cos = vec![0.0f32; rope_dim / 2];
    let mut sin = vec![0.0f32; rope_dim / 2];
    rope_compute_cos_sin(&freqs, 7, &mut cos, &mut sin);

    for rtype in [RopeType::Norm, RopeType::Neox] {
        let rx: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.f32(1.0)).collect();
        let rdy: Vec<f32> = (0..n_heads * head_dim).map(|_| rng.f32(1.0)).collect();
        let mut y = rx.clone();
        rope_inplace_tables(&mut y, n_heads, head_dim, rope_dim, &cos, &sin, rtype);
        let mut dx_rope = rdy.clone();
        rope_bwd_inplace_tables(&mut dx_rope, n_heads, head_dim, rope_dim, &cos, &sin, rtype);

        let lhs: f64 = y
            .iter()
            .zip(rdy.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let rhs: f64 = rx
            .iter()
            .zip(dx_rope.iter())
            .map(|(&a, &b)| (a as f64) * (b as f64))
            .sum();
        let rel = (lhs - rhs).abs() / lhs.abs().max(1.0);
        assert!(rel < 1e-6, "RoPE {rtype:?} adjoint mismatch rel={rel:e}");
    }

    // 3. SwiGLU backward vs FP64 central finite difference
    let gate: Vec<f32> = (0..n).map(|_| rng.f32(1.5)).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.f32(1.5)).collect();
    let d_out: Vec<f32> = (0..n).map(|_| rng.f32(1.0)).collect();
    let mut d_gate = vec![0.0f32; n];
    let mut d_up = vec![0.0f32; n];
    swiglu_bwd_f32(&mut d_gate, &mut d_up, &gate, &up, &d_out);

    let mut fwd_hb = gate.clone();
    swiglu_inplace(&mut fwd_hb, &up);
    for i in 0..n {
        let swiglu64 = |g: f64, u: f64| (g / (1.0 + (-g).exp())) * u;
        let g0 = gate[i] as f64;
        let u0 = up[i] as f64;
        let dy0 = d_out[i] as f64;
        let num_dg = (swiglu64(g0 + eps_fd, u0) - swiglu64(g0 - eps_fd, u0)) / (2.0 * eps_fd) * dy0;
        let num_du = (swiglu64(g0, u0 + eps_fd) - swiglu64(g0, u0 - eps_fd)) / (2.0 * eps_fd) * dy0;
        assert!(
            ((d_gate[i] as f64) - num_dg).abs() < 1e-5,
            "swiglu d_gate[{i}] mismatch"
        );
        assert!(
            ((d_up[i] as f64) - num_du).abs() < 1e-5,
            "swiglu d_up[{i}] mismatch"
        );
    }

    // 4. Causal GQA F16-KV attention backward (`n_heads=4, n_kv_heads=2` -> 2:1 GQA sharing!) vs FP64 finite diff
    let n_tok = 5usize;
    let nh = 4usize;
    let nkv = 2usize;
    let hd = 16usize;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let q_in: Vec<f32> = (0..n_tok * q_dim).map(|_| rng.f32(0.7)).collect();
    // Quantize K/V to exact F16 values first so finite difference differentiates wrt the F16-representable K/V!
    let k_f16: Vec<u16> = (0..n_tok * kv_dim)
        .map(|_| f32_to_f16(rng.f32(0.7)))
        .collect();
    let v_f16: Vec<u16> = (0..n_tok * kv_dim)
        .map(|_| f32_to_f16(rng.f32(0.7)))
        .collect();
    let do_in: Vec<f32> = (0..n_tok * q_dim).map(|_| rng.f32(0.7)).collect();

    let kt = select_kernels(*supported_isas().last().unwrap());
    let pool = ThreadPool::new(2);
    let mut scores = vec![0.0f32; n_tok];
    let mut out_check = vec![0.0f32; n_tok * q_dim];
    gqa_attention_f16_kv_batch_parallel(
        &pool,
        kt,
        &mut out_check,
        &q_in,
        &k_f16,
        &v_f16,
        nh,
        nkv,
        hd,
        0,
        n_tok,
        n_tok,
        &mut scores,
    );

    let mut dq = vec![0.0f32; n_tok * q_dim];
    let mut dk = vec![0.0f32; n_tok * kv_dim];
    let mut dv = vec![0.0f32; n_tok * kv_dim];
    gqa_attention_f16_kv_bwd_f32(
        kt, &mut dq, &mut dk, &mut dv, &q_in, &k_f16, &v_f16, &do_in, nh, nkv, hd, n_tok, None,
    );

    let attn_loss64 = |qv: &[f64], kv: &[f64], vv: &[f64]| -> f64 {
        let scale = 1.0 / (hd as f64).sqrt();
        let group = nh / nkv;
        let mut loss = 0.0f64;
        for t in 0..n_tok {
            for h in 0..nh {
                let kvh = h / group;
                let mut sc = vec![0.0f64; t + 1];
                for s in 0..=t {
                    let mut dot = 0.0f64;
                    for d in 0..hd {
                        dot += qv[t * q_dim + h * hd + d] * kv[s * kv_dim + kvh * hd + d];
                    }
                    sc[s] = dot * scale;
                }
                let mx = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = sc.iter().map(|&v| (v - mx).exp()).collect();
                let z: f64 = exps.iter().sum();
                for d in 0..hd {
                    let mut o = 0.0f64;
                    for s in 0..=t {
                        o += (exps[s] / z) * vv[s * kv_dim + kvh * hd + d];
                    }
                    loss += o * (do_in[t * q_dim + h * hd + d] as f64);
                }
            }
        }
        loss
    };

    let q64: Vec<f64> = q_in.iter().map(|&v| v as f64).collect();
    let k64: Vec<f64> = k_f16.iter().map(|&v| f16_to_f32(v) as f64).collect();
    let v64: Vec<f64> = v_f16.iter().map(|&v| f16_to_f32(v) as f64).collect();

    for i in 0..q64.len() {
        let mut qp = q64.clone();
        let mut qm = q64.clone();
        qp[i] += eps_fd;
        qm[i] -= eps_fd;
        let num = (attn_loss64(&qp, &k64, &v64) - attn_loss64(&qm, &k64, &v64)) / (2.0 * eps_fd);
        let err = ((dq[i] as f64) - num).abs() / num.abs().max(1.0);
        assert!(
            err < 1e-5,
            "gqa dq[{i}] got={} num={num} err={err:e}",
            dq[i]
        );
    }
    for i in 0..k64.len() {
        let mut kp = k64.clone();
        let mut km = k64.clone();
        kp[i] += eps_fd;
        km[i] -= eps_fd;
        let num = (attn_loss64(&q64, &kp, &v64) - attn_loss64(&q64, &km, &v64)) / (2.0 * eps_fd);
        let err = ((dk[i] as f64) - num).abs() / num.abs().max(1.0);
        assert!(
            err < 1e-5,
            "gqa dk[{i}] got={} num={num} err={err:e}",
            dk[i]
        );

        let mut vp = v64.clone();
        let mut vm = v64.clone();
        vp[i] += eps_fd;
        vm[i] -= eps_fd;
        let num_v = (attn_loss64(&q64, &k64, &vp) - attn_loss64(&q64, &k64, &vm)) / (2.0 * eps_fd);
        let err_v = ((dv[i] as f64) - num_v).abs() / num_v.abs().max(1.0);
        assert!(
            err_v < 1e-5,
            "gqa dv[{i}] got={} num={num_v} err={err_v:e}",
            dv[i]
        );
    }
}
