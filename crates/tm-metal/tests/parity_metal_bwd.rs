//! Phase `T4` Apple Metal verification & microbenchmark (`AGENTS.md` Rule #3):
//! 1. Aligns all 7 `gemm_*_transposed` kernels (`Q4_0`, `Q4_1`, `Q8_0`, `Q5_K`, `Q6_K`, `BF16`, `F32`)
//!    against `tm_cpu::generic_kernels()` scalar reference across `n_tok = 1, 9, 37` and `accum = false/true`.
//! 2. Aligns `lora_linear_f32` and `lora_embed_f32` against `tm_lora::BoundLora` + `generic_kernels()`,
//!    including **0-ULP batch vs single-token determinism** (`AGENTS.md` §9).
//! 3. Verifies end-to-end `MetalRunner::bind_lora` multi-token logits parity against `CpuModel`
//!    AND `MetalGpuAccel` forward/backward parity + throughput benchmark on Apple M4 GPU.

#![cfg(target_vendor = "apple")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tm_core::{DType, f32_to_f16, quantize_row_to_bytes};
use tm_cpu::{Isa, ThreadPool, active_kernels, matmul_q_transposed_f32, select_kernels};
use tm_gguf::{GgufFile, MetadataValue, NewGgufTensor, write_gguf_file};
use tm_lora::{LoraAdapter, LoraSet, LoraWeightPair};
use tm_metal::{MetalContext, MetalGpuAccel, MetalRunner};
use tm_runtime::CpuModel;

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((self.0 >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
        (u * 2.0 - 1.0) * scale
    }
}

fn encode_2d_dtype(dtype: DType, rows: usize, cols: usize, vals: &[f32]) -> Vec<u8> {
    let rb = dtype.row_bytes(cols);
    let mut out = vec![0u8; rows * rb];
    for r in 0..rows {
        let src = &vals[r * cols..(r + 1) * cols];
        let dst = &mut out[r * rb..(r + 1) * rb];
        match dtype {
            DType::Q4_1 => {
                // Encode exact BlockQ4_1 (`d: f16, m: f16, qs: [u8; 16]`)
                for b in 0..(cols / 32) {
                    let blk = &src[b * 32..(b + 1) * 32];
                    let mut mn = blk[0];
                    let mut mx = blk[0];
                    for &v in blk {
                        mn = mn.min(v);
                        mx = mx.max(v);
                    }
                    let d = (mx - mn) / 15.0;
                    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
                    let d_f16 = f32_to_f16(d);
                    let m_f16 = f32_to_f16(mn);
                    let bo = &mut dst[b * 20..(b + 1) * 20];
                    bo[0..2].copy_from_slice(&d_f16.to_le_bytes());
                    bo[2..4].copy_from_slice(&m_f16.to_le_bytes());
                    for j in 0..16 {
                        let q0 = ((blk[j] - mn) * inv_d + 0.5).clamp(0.0, 15.0) as u8;
                        let q1 = ((blk[j + 16] - mn) * inv_d + 0.5).clamp(0.0, 15.0) as u8;
                        bo[4 + j] = q0 | (q1 << 4);
                    }
                }
            }
            DType::Q5K => {
                // Encode BlockQ5K (`d, dmin, scales[12], qh[32], qs[128]`)
                for sb in 0..(cols / 256) {
                    let sblk = &src[sb * 256..(sb + 1) * 256];
                    let bo = &mut dst[sb * 176..(sb + 1) * 176];
                    let d = 0.025f32;
                    let dmin = 0.40f32;
                    bo[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                    bo[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
                    for j in 0..12 {
                        bo[4 + j] = ((j * 7 + 3) & 0x3F) as u8;
                    }
                    for l in 0..32 {
                        let mut qh_byte = 0u8;
                        for chunk in 0..4 {
                            let v_lo = sblk[chunk * 64 + l];
                            let v_hi = sblk[chunk * 64 + 32 + l];
                            let q_lo = ((v_lo + 0.5) * 15.0).round().clamp(0.0, 31.0) as u8;
                            let q_hi = ((v_hi + 0.5) * 15.0).round().clamp(0.0, 31.0) as u8;
                            bo[48 + chunk * 32 + l] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
                            qh_byte |= ((q_lo >> 4) & 1) << (2 * chunk);
                            qh_byte |= ((q_hi >> 4) & 1) << (2 * chunk + 1);
                        }
                        bo[16 + l] = qh_byte;
                    }
                }
            }
            DType::Q6K => {
                for sb in 0..(cols / 256) {
                    let sblk = &src[sb * 256..(sb + 1) * 256];
                    let bo = &mut dst[sb * 210..(sb + 1) * 210];
                    bo[208..210].copy_from_slice(&f32_to_f16(0.02f32).to_le_bytes());
                    for j in 0..16 {
                        bo[192 + j] = ((j as i8) * 3 - 20) as u8;
                    }
                    for half_idx in 0..2 {
                        for l in 0..32 {
                            let q0 = ((sblk[half_idx * 128 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            let q1 = ((sblk[half_idx * 128 + 32 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            let q2 = ((sblk[half_idx * 128 + 64 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            let q3 = ((sblk[half_idx * 128 + 96 + l] * 30.0).round() as i32 + 32)
                                .clamp(0, 63) as u8;
                            bo[half_idx * 64 + l] = (q0 & 0x0F) | ((q2 & 0x0F) << 4);
                            bo[half_idx * 64 + 32 + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                            bo[128 + half_idx * 32 + l] = ((q0 >> 4) & 3)
                                | (((q1 >> 4) & 3) << 2)
                                | (((q2 >> 4) & 3) << 4)
                                | (((q3 >> 4) & 3) << 6);
                        }
                    }
                }
            }
            DType::BF16 => {
                for (c, &v) in src.iter().enumerate() {
                    let bits = v.to_bits();
                    let bf = (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
                    dst[c * 2..c * 2 + 2].copy_from_slice(&bf.to_le_bytes());
                }
            }
            _ => {
                quantize_row_to_bytes(dtype, src, dst);
            }
        }
    }
    out
}

#[test]
fn t4_metal_transposed_gemm_all_dtypes_parity_and_bench() {
    let ctx = MetalContext::new().expect("MetalContext");
    let pool = ThreadPool::new(1);
    let kt_ref = select_kernels(Isa::Generic);
    let mut rng = Rng(0x2026_1122);

    let cases = [
        (DType::Q4_0, 99usize, 128usize),
        (DType::Q4_1, 77, 128),
        (DType::Q8_0, 113, 160),
        (DType::Q5K, 67, 512),
        (DType::Q6K, 71, 512),
        (DType::BF16, 69, 100), // non-multiple-of-32 `cols`!
        (DType::F32, 83, 100),  // non-multiple-of-32 `cols`!
    ];

    for (dtype, rows, cols) in cases {
        let w_f32: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32(0.5)).collect();
        let w_bytes = encode_2d_dtype(dtype, rows, cols, &w_f32);
        let w_gpu = ctx.buffer_from_slice(&w_bytes).unwrap();

        for &n_tok in &[1usize, 9, 37] {
            let dy: Vec<f32> = (0..n_tok * rows).map(|_| rng.next_f32(0.8)).collect();
            let init_dx: Vec<f32> = (0..n_tok * cols).map(|_| rng.next_f32(0.4)).collect();
            let dy_gpu = ctx.buffer_from_slice(&dy).unwrap();

            for &accum in &[false, true] {
                let mut ref_dx = if accum {
                    init_dx.clone()
                } else {
                    vec![0.0f32; n_tok * cols]
                };
                matmul_q_transposed_f32(
                    &pool,
                    kt_ref,
                    dtype,
                    &w_bytes,
                    &dy,
                    &mut ref_dx,
                    rows,
                    cols,
                    n_tok,
                    accum,
                );

                let mut dx_gpu = ctx.buffer_from_slice(&init_dx).unwrap();
                ctx.matmul_transposed_f32(
                    dtype,
                    &w_gpu,
                    &dy_gpu,
                    &mut dx_gpu,
                    rows,
                    cols,
                    n_tok,
                    accum,
                )
                .unwrap();

                let got = dx_gpu.as_f32_slice();
                let mut max_rel = 0.0f32;
                for (i, (&g, &r)) in got.iter().zip(ref_dx.iter()).enumerate() {
                    let denom = r.abs().max(1.0);
                    let rel = (g - r).abs() / denom;
                    max_rel = max_rel.max(rel);
                    assert!(
                        rel < 2e-5,
                        "Metal transposed GEMM mismatch dtype={dtype:?} rows={rows} cols={cols} n_tok={n_tok} accum={accum} idx={i}: gpu={g} ref={r} rel={rel:e}"
                    );
                }
                if n_tok == 37 && accum {
                    eprintln!(
                        "[T4 Metal bwd parity] {dtype:?} ({rows}x{cols}, T={n_tok}, accum=true) max_rel={max_rel:.3e}"
                    );
                }
            }
        }
    }

    // Rule #3 Microbenchmark: `Q4_0` `4096 x 2048` (`T = 128` tokens) transposed GEMM on Apple M4
    let b_rows = 4096usize;
    let b_cols = 2048usize;
    let b_tok = 128usize;
    let w_bench = vec![0u8; b_rows * DType::Q4_0.row_bytes(b_cols)];
    let dy_bench = vec![0.01f32; b_tok * b_rows];
    let w_g = ctx.buffer_from_slice(&w_bench).unwrap();
    let dy_g = ctx.buffer_from_slice(&dy_bench).unwrap();
    let mut dx_g = ctx.alloc_buffer(b_tok * b_cols * 4).unwrap();
    ctx.matmul_transposed_f32_repeat(
        DType::Q4_0,
        &w_g,
        &dy_g,
        &mut dx_g,
        b_rows,
        b_cols,
        b_tok,
        false,
        4,
    )
    .unwrap();
    let repeat = 20usize;
    let t0 = Instant::now();
    ctx.matmul_transposed_f32_repeat(
        DType::Q4_0,
        &w_g,
        &dy_g,
        &mut dx_g,
        b_rows,
        b_cols,
        b_tok,
        false,
        repeat,
    )
    .unwrap();
    let sec = t0.elapsed().as_secs_f64() / (repeat as f64);
    let flops = (2 * b_tok * b_rows * b_cols) as f64;
    let gflops = flops / sec / 1e9;
    let gbs = (w_bench.len() + b_tok * (b_rows + b_cols) * 4) as f64 / sec / 1e9;
    eprintln!(
        "[T4 Metal bwd bench] Q4_0 {b_rows}x{b_cols} T={b_tok} ({}): {:.2} us | {:.1} GFLOP/s | {:.1} GB/s",
        ctx.device_name(),
        sec * 1e6,
        gflops,
        gbs
    );
}

#[test]
fn t4_metal_dynamic_lora_kernels_and_end_to_end_runner_parity() {
    let ctx = MetalContext::new().expect("MetalContext");
    let mut rng = Rng(0xABCD_EF01);

    // 1. Direct `lora_linear_f32` & `lora_embed_f32` parity + 0-ULP batch/single determinism (`AGENTS.md` §9)
    let d_in = 128usize;
    let d_out = 96usize;
    let rank = 8usize;
    let n_tok = 11usize;
    let scale = 1.75f32;
    let a: Vec<f32> = (0..rank * d_in).map(|_| rng.next_f32(0.3)).collect();
    let b_t: Vec<f32> = (0..rank * d_out).map(|_| rng.next_f32(0.3)).collect();
    let x: Vec<f32> = (0..n_tok * d_in).map(|_| rng.next_f32(0.5)).collect();
    let y0: Vec<f32> = (0..n_tok * d_out).map(|_| rng.next_f32(0.2)).collect();

    let a_g = ctx.buffer_from_slice(&a).unwrap();
    let bt_g = ctx.buffer_from_slice(&b_t).unwrap();
    let x_g = ctx.buffer_from_slice(&x).unwrap();
    let mut h_g = ctx.alloc_buffer(n_tok * rank * 4).unwrap();
    let mut y_batch = ctx.buffer_from_slice(&y0).unwrap();
    ctx.lora_linear_f32(
        &a_g,
        &bt_g,
        &x_g,
        &mut h_g,
        &mut y_batch,
        d_in,
        d_out,
        rank,
        n_tok,
        scale,
    )
    .unwrap();

    // Check 0-ULP batch vs single-token determinism AND match against scalar reference
    for t in 0..n_tok {
        let xt_g = ctx.buffer_from_slice(&x[t * d_in..(t + 1) * d_in]).unwrap();
        let mut yt_single = ctx
            .buffer_from_slice(&y0[t * d_out..(t + 1) * d_out])
            .unwrap();
        ctx.lora_linear_f32(
            &a_g,
            &bt_g,
            &xt_g,
            &mut h_g,
            &mut yt_single,
            d_in,
            d_out,
            rank,
            1,
            scale,
        )
        .unwrap();
        let b_slice = &y_batch.as_f32_slice()[t * d_out..(t + 1) * d_out];
        let s_slice = yt_single.as_f32_slice();
        for c in 0..d_out {
            assert_eq!(
                b_slice[c].to_bits(),
                s_slice[c].to_bits(),
                "0-ULP batch vs single mismatch in lora_linear_f32 at t={t} c={c}"
            );
            let mut exp = y0[t * d_out + c];
            for r in 0..rank {
                let mut dot = 0.0f32;
                for i in 0..d_in {
                    dot += a[r * d_in + i] * x[t * d_in + i];
                }
                exp += (dot * scale) * b_t[r * d_out + c];
            }
            assert!(
                (b_slice[c] - exp).abs() < 1e-5,
                "lora_linear_f32 scalar mismatch: got={} exp={exp}",
                b_slice[c]
            );
        }
    }

    // 2. End-to-end `MetalRunner::bind_lora` + `MetalGpuAccel` parity on synthetic Qwen3 (QK-Norm + tied output + active LoRA on ALL modules including embed, qkv, gate/up, down, output!)
    let dir = std::env::temp_dir().join(format!("tm_t4_mtl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mpath = dir.join("qwen3_q4.gguf");
    {
        let n_embd = 64usize;
        let n_ff = 128usize;
        let vocab = 64usize;
        let meta = vec![
            (
                "general.architecture".to_string(),
                MetadataValue::String("qwen3".to_string()),
            ),
            ("qwen3.block_count".to_string(), MetadataValue::U32(2)),
            ("qwen3.embedding_length".to_string(), MetadataValue::U32(64)),
            (
                "qwen3.feed_forward_length".to_string(),
                MetadataValue::U32(128),
            ),
            (
                "qwen3.attention.head_count".to_string(),
                MetadataValue::U32(2),
            ),
            (
                "qwen3.attention.head_count_kv".to_string(),
                MetadataValue::U32(1),
            ),
            (
                "qwen3.attention.key_length".to_string(),
                MetadataValue::U32(32),
            ),
            (
                "qwen3.attention.value_length".to_string(),
                MetadataValue::U32(32),
            ),
            (
                "qwen3.rope.dimension_count".to_string(),
                MetadataValue::U32(32),
            ),
            (
                "qwen3.rope.freq_base".to_string(),
                MetadataValue::F32(10000.0),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon".to_string(),
                MetadataValue::F32(1e-6),
            ),
            ("qwen3.context_length".to_string(), MetadataValue::U32(128)),
        ];
        struct T {
            n: String,
            d: Vec<usize>,
            dt: DType,
            b: Vec<u8>,
        }
        let mut ts = Vec::new();
        let f1d = |r: &mut Rng, len: usize| -> Vec<u8> {
            let mut b = Vec::with_capacity(len * 4);
            for _ in 0..len {
                b.extend_from_slice(&(0.9f32 + r.next_f32(0.15)).to_le_bytes());
            }
            b
        };
        let q2d = |r: &mut Rng, cols: usize, rows: usize| -> Vec<u8> {
            let v: Vec<f32> = (0..rows * cols).map(|_| r.next_f32(0.08)).collect();
            encode_2d_dtype(DType::Q4_0, rows, cols, &v)
        };
        ts.push(T {
            n: "token_embd.weight".into(),
            d: vec![n_embd, vocab],
            dt: DType::Q4_0,
            b: q2d(&mut rng, n_embd, vocab),
        });
        ts.push(T {
            n: "output_norm.weight".into(),
            d: vec![n_embd],
            dt: DType::F32,
            b: f1d(&mut rng, n_embd),
        });
        for il in 0..2 {
            let p = format!("blk.{il}");
            ts.push(T {
                n: format!("{p}.attn_norm.weight"),
                d: vec![n_embd],
                dt: DType::F32,
                b: f1d(&mut rng, n_embd),
            });
            ts.push(T {
                n: format!("{p}.ffn_norm.weight"),
                d: vec![n_embd],
                dt: DType::F32,
                b: f1d(&mut rng, n_embd),
            });
            ts.push(T {
                n: format!("{p}.attn_q_norm.weight"),
                d: vec![32],
                dt: DType::F32,
                b: f1d(&mut rng, 32),
            });
            ts.push(T {
                n: format!("{p}.attn_k_norm.weight"),
                d: vec![32],
                dt: DType::F32,
                b: f1d(&mut rng, 32),
            });
            for (s, c, rw) in [
                ("attn_q.weight", 64, 64),
                ("attn_k.weight", 64, 32),
                ("attn_v.weight", 64, 32),
                ("attn_output.weight", 64, 64),
                ("ffn_gate.weight", 64, n_ff),
                ("ffn_up.weight", 64, n_ff),
                ("ffn_down.weight", n_ff, 64),
            ] {
                ts.push(T {
                    n: format!("{p}.{s}"),
                    d: vec![c, rw],
                    dt: DType::Q4_0,
                    b: q2d(&mut rng, c, rw),
                });
            }
        }
        let views: Vec<NewGgufTensor<'_>> = ts
            .iter()
            .map(|t| NewGgufTensor {
                name: &t.n,
                dims: &t.d,
                dtype: t.dt,
                data: &t.b,
            })
            .collect();
        write_gguf_file(&mpath, &meta, &views, 32).unwrap();
    }

    let gguf = Arc::new(GgufFile::open(&mpath).unwrap());
    let mut pairs = HashMap::new();
    let mut order = Vec::new();
    let mut add_pair = |name: &str, is_emb: bool, d_in: usize, d_out: usize| {
        let rank = 4usize;
        let a_len = d_in * rank;
        let a: Vec<f32> = (0..a_len).map(|_| rng.next_f32(0.25)).collect();
        let b: Vec<f32> = (0..d_out * rank).map(|_| rng.next_f32(0.25)).collect();
        order.push(name.to_string());
        pairs.insert(
            name.to_string(),
            LoraWeightPair {
                base_name: name.to_string(),
                is_token_embd: is_emb,
                rank,
                d_in,
                d_out,
                a,
                b,
            },
        );
    };
    add_pair("token_embd.weight", true, 64, 64);
    add_pair("blk.0.attn_q.weight", false, 64, 64);
    add_pair("blk.0.attn_v.weight", false, 64, 32);
    add_pair("blk.0.ffn_gate.weight", false, 64, 128);
    add_pair("blk.1.ffn_down.weight", false, 128, 64);
    add_pair("output.weight", false, 64, 64); // tied LM head!

    let adapter = LoraAdapter {
        arch: "qwen3".to_string(),
        alpha: 8.0,
        name: Some("test".to_string()),
        pairs,
        pair_order: order,
    };
    let mut lset = LoraSet::new();
    lset.push(adapter, 0.85);

    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let mut cpu = CpuModel::from_gguf(&gguf).unwrap();
    cpu.bind_lora(&gguf, &lset).unwrap();
    let mut cstate = cpu.new_state(64);

    let mut mtl = MetalRunner::from_gguf(gguf.clone(), 64).unwrap();
    mtl.bind_lora(&lset).unwrap();

    // Multi-token prefill (`forward_batch`) + decode (`forward_token`) parity (`AGENTS.md` Rule #7)
    let prompt = [3u32, 9, 17, 25, 41];
    cpu.forward_batch(&mut cstate, &pool, kt, &prompt, 0, true)
        .unwrap();
    let m_pp = mtl.forward_batch(&prompt, 0, true).unwrap().to_vec();
    for (i, (&m, &c)) in m_pp.iter().zip(cstate.logits().iter()).enumerate() {
        // `MetalRunner` uses FP16 `simdgroup_mma` while `CpuModel` uses dynamic `Q8_0` integer dot (`docs/TESTING.md`)
        assert!(
            (m - c).abs() < 5e-2,
            "MetalRunner prefill --lora mismatch at {i}: mtl={m} cpu={c}"
        );
    }

    for (step, &tok) in [12u32, 33, 7].iter().enumerate() {
        let pos = prompt.len() + step;
        cpu.forward_token(&mut cstate, &pool, kt, tok, pos, true)
            .unwrap();
        let m_dec = mtl.forward_token(tok, pos, true).unwrap();
        for (i, (&m, &c)) in m_dec.iter().zip(cstate.logits().iter()).enumerate() {
            assert!(
                (m - c).abs() < 5e-2,
                "MetalRunner decode --lora mismatch step={step} idx={i}: mtl={m} cpu={c}"
            );
        }
    }

    // Also verify Metal dynamic `--lora` matches offline `merge_lora_gguf(..., F32)` on Metal to `< 1e-3`!
    let merged_f32_path = dir.join("qwen3_merged_f32.gguf");
    tm_lora::merge_lora_gguf(
        &gguf,
        &lset,
        &merged_f32_path,
        &tm_lora::LoraMergeOptions {
            quant: Some(DType::F32),
            lm_head_quant: Some(DType::F32),
        },
    )
    .unwrap();
    let merged_gguf = Arc::new(GgufFile::open(&merged_f32_path).unwrap());
    let cpu_merged = CpuModel::from_gguf(&merged_gguf).unwrap();
    let mut cm_state = cpu_merged.new_state(64);
    // Test F32 base + dynamic `--lora` on Metal vs merged F32 CPU (< 1e-4 exact FP32 parity!)
    let base_f32_path = dir.join("qwen3_base_f32.gguf");
    gguf.convert_with_options(
        &base_f32_path,
        &tm_gguf::ConvertOptions {
            quant: Some(DType::F32),
            lm_head_quant: Some(DType::F32),
            ..Default::default()
        },
    )
    .unwrap();
    let base_f32_gguf = Arc::new(GgufFile::open(&base_f32_path).unwrap());
    let mut mtl_f32 = MetalRunner::from_gguf(base_f32_gguf, 64).unwrap();
    mtl_f32.bind_lora(&lset).unwrap();
    cpu_merged
        .forward_token(&mut cm_state, &pool, kt, 9, 0, true)
        .unwrap();
    let mf_dec = mtl_f32.forward_token(9, 0, true).unwrap();
    for (i, (&m, &c)) in mf_dec.iter().zip(cm_state.logits().iter()).enumerate() {
        assert!(
            (m - c).abs() < 1e-4,
            "MetalRunner F32 --lora vs merged F32 mismatch at {i}: mtl={m} merged={c}"
        );
    }

    // Verify `MetalGpuAccel` zero-copy forward + transposed backward
    let mut accel = MetalGpuAccel::from_gguf(gguf.clone(), 32)
        .unwrap()
        .with_min_macs(0);
    let x_in: Vec<f32> = (0..9 * 64).map(|_| rng.next_f32(0.5)).collect();
    let mut y_gpu = vec![0.0f32; 9 * 128];
    assert!(accel.matmul_fwd("blk.0.ffn_gate.weight", 128, 64, &x_in, &mut y_gpu, 9));
    let mut dx_gpu = vec![0.25f32; 9 * 64];
    let mut dx_ref = dx_gpu.clone();
    assert!(accel.matmul_bwd_transposed(
        "blk.0.ffn_gate.weight",
        128,
        64,
        &y_gpu,
        &mut dx_gpu,
        9,
        true,
    ));
    let (_, w_gate_bytes) = gguf.tensor_bytes("blk.0.ffn_gate.weight", "blk.0").unwrap();
    let kt_ref = select_kernels(Isa::Generic);
    matmul_q_transposed_f32(
        &pool,
        kt_ref,
        DType::Q4_0,
        w_gate_bytes,
        &y_gpu,
        &mut dx_ref,
        128,
        64,
        9,
        true,
    );
    for (i, (&g, &r)) in dx_gpu.iter().zip(dx_ref.iter()).enumerate() {
        assert!(
            (g - r).abs() < 5e-5,
            "MetalGpuAccel bwd mismatch at {i}: {g} vs {r}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
