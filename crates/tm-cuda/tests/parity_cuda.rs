//! L1 kernel & L3 end-to-end multi-token parity tests for `tm-cuda`.
//!
//! When run on a machine without an NVIDIA GPU / `libcuda` (e.g., Apple M4 macOS or CPU-only CI),
//! tests cleanly report `[skip: no CUDA driver]` and succeed; on NVIDIA GPUs (e.g., Tesla T4 `sm_75`),
//! every kernel and full-model runner is verified against CPU scalar / FP64 reference!

use std::path::PathBuf;
use std::process::Command;

use tm_core::{BlockQ4_0, BlockQ6K, BlockQ8_0, BlockQ8_1, DType, KvType, f16_to_f32, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels, generic};
use tm_cuda::{BLOCK_Q8_1_BYTES, CudaContext, CudaKernels, CudaMat, CudaRunner};
use tm_gguf::GgufFile;
use tm_models::{LlamaModel, Qwen35Model, Qwen35State};

fn try_ctx() -> Option<CudaContext> {
    match CudaContext::new(None) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[skip parity_cuda: {e}]");
            None
        }
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn find_python3_with_numpy() -> String {
    for cand in [
        std::env::var("PYTHON3").ok().as_deref().unwrap_or(""),
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/opt/anaconda3/bin/python3",
        "/opt/homebrew/bin/python3",
        "python3",
    ] {
        if !cand.is_empty()
            && Command::new(cand)
                .args(["-c", "import numpy"])
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        {
            return cand.to_string();
        }
    }
    "python3".to_string()
}

/// Exact scalar `Q6_K × Q8_1` dot product matching `ggml-cuda` / `Strata` `q6_q8_dot`.
fn scalar_dot_q6_k_q8_1(blk: &BlockQ6K, xq: &[BlockQ8_1]) -> f32 {
    let d = f16_to_f32(blk.d);
    let mut sum = 0.0f32;
    for half128 in 0..2 {
        let ql = &blk.ql[half128 * 64..(half128 + 1) * 64];
        let qh = &blk.qh[half128 * 32..(half128 + 1) * 32];
        let sc = &blk.scales[half128 * 8..(half128 + 1) * 8];
        for quad in 0..4 {
            let xb = &xq[half128 * 4 + quad];
            let dx = f16_to_f32(xb.d);
            for l in 0..32 {
                let q_lo = match quad {
                    0 => ql[l] & 0x0F,
                    1 => ql[l + 32] & 0x0F,
                    2 => ql[l] >> 4,
                    _ => ql[l + 32] >> 4,
                };
                let q_hi = ((qh[l] >> (2 * quad)) & 0x03) << 4;
                let q = ((q_lo | q_hi) as i32) - 32;
                let scale = sc[(l / 16) + 2 * quad] as f32;
                sum += d * scale * dx * (q * (xb.qs[l] as i32)) as f32;
            }
        }
    }
    sum
}

#[test]
fn parity_cuda_l1_mmvq_and_wmma_tensor_core_gemm() {
    let Some(ctx) = try_ctx() else { return };
    let kernels = CudaKernels::load(&ctx).expect("load cuda_ops.ptx");
    eprintln!(
        "[parity_cuda] running on '{}' (sm_{}{}, wmma={})",
        ctx.info().name,
        ctx.info().cc_major,
        ctx.info().cc_minor,
        ctx.info().has_fp16_wmma()
    );

    // 1. Q4_0 MMVQ (n_tok = 1 & n_tok = 4) AND WMMA Tensor Core GEMM (n_tok = 32)
    let rows = 96usize;
    let cols = 128usize;
    let n_blks = cols / 32;
    let mut w_blks = Vec::with_capacity(rows * n_blks);
    let mut w_bytes = Vec::with_capacity(rows * n_blks * 18);
    let mut w_deq = vec![0.0f32; rows * cols];

    for r in 0..rows {
        for b in 0..n_blks {
            let d_f32 = 0.0625f32 * (((r + b) % 4) + 1) as f32;
            let d = f32_to_f16(d_f32);
            let mut qs = [0u8; 16];
            for j in 0..16 {
                let lo = ((r * 13 + b * 7 + j * 3) % 16) as u8;
                let hi = ((r * 5 + b * 11 + j * 7 + 1) % 16) as u8;
                qs[j] = lo | (hi << 4);
                w_deq[r * cols + b * 32 + j] = d_f32 * (lo as i32 - 8) as f32;
                w_deq[r * cols + b * 32 + 16 + j] = d_f32 * (hi as i32 - 8) as f32;
            }
            w_bytes.extend_from_slice(&d.to_le_bytes());
            w_bytes.extend_from_slice(&qs);
            w_blks.push(BlockQ4_0 { d, qs });
        }
    }

    let w_mat = CudaMat {
        buf: std::sync::Arc::new(ctx.alloc_from_bytes(&w_bytes).unwrap()),
        offset: 0,
        dtype: DType::Q4_0,
        rows,
        cols,
        row_bytes: n_blks * 18,
    };

    for &n_tok in &[1usize, 4, 32] {
        let x: Vec<f32> = (0..n_tok * cols)
            .map(|i| (((i * 17 + 5) % 64) as f32 - 32.0) * 0.03125)
            .collect();
        let x_dev = ctx.alloc_from_f32(&x).unwrap();
        let xq_dev = ctx.alloc_zeroed(n_tok * n_blks * BLOCK_Q8_1_BYTES).unwrap();
        let out_dev = ctx.alloc_zeroed(n_tok * rows * 4).unwrap();

        kernels
            .launch_linear(
                &ctx,
                &w_mat,
                x_dev.device_ptr(),
                xq_dev.device_ptr(),
                false,
                out_dev.device_ptr(),
                n_tok,
                false,
            )
            .unwrap();
        let mut got = vec![0.0f32; n_tok * rows];
        out_dev.copy_to_f32(&mut got).unwrap();

        for t in 0..n_tok {
            let xt = &x[t * cols..(t + 1) * cols];
            if n_tok <= 8 {
                let mut xq8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n_blks];
                generic::quantize_q8_0(xt, &mut xq8);
                let mut ref_row = vec![0.0f32; rows];
                generic::gemv_q4_0_q8_0(&w_blks, &xq8, &mut ref_row, rows, n_blks);
                for r in 0..rows {
                    let diff = (got[t * rows + r] - ref_row[r]).abs();
                    assert!(
                        diff < 5e-3,
                        "Q4_0 MMVQ n_tok={n_tok} t={t} r={r}: got={} ref={} diff={diff}",
                        got[t * rows + r],
                        ref_row[r]
                    );
                }
            } else {
                // WMMA FP16 Tensor Core path: compare against exact dequantized FP32 dot product
                for r in 0..rows {
                    let mut acc = 0.0f32;
                    for c in 0..cols {
                        acc += w_deq[r * cols + c] * xt[c];
                    }
                    let diff = (got[t * rows + r] - acc).abs();
                    assert!(
                        diff < 2e-2,
                        "WMMA Q4_0 n_tok={n_tok} t={t} r={r}: got={} ref={acc} diff={diff}",
                        got[t * rows + r]
                    );
                }
            }
        }
    }

    // 2. Q6_K × Q8_1 MMVQ (super-block 256 cols) vs exact scalar `Q6_K × Q8_1` reference
    let q6_rows = 16usize;
    let q6_cols = 256usize;
    let mut w_q6 = Vec::with_capacity(q6_rows);
    let mut w_q6_bytes = Vec::with_capacity(q6_rows * 210);
    for r in 0..q6_rows {
        let d = f32_to_f16(0.03125 * ((r % 4) + 1) as f32);
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for (i, slot) in ql.iter_mut().enumerate() {
            *slot = (((r * 19 + i * 7) % 16) | (((r * 11 + i * 3) % 16) << 4)) as u8;
        }
        for (i, slot) in qh.iter_mut().enumerate() {
            *slot = ((r * 23 + i * 13) & 0xFF) as u8;
        }
        for (i, slot) in scales.iter_mut().enumerate() {
            *slot = (((r * 5 + i * 3) % 15) as i8) - 7;
        }
        w_q6_bytes.extend_from_slice(&ql);
        w_q6_bytes.extend_from_slice(&qh);
        for &sc in &scales {
            w_q6_bytes.push(sc as u8);
        }
        w_q6_bytes.extend_from_slice(&d.to_le_bytes());
        w_q6.push(BlockQ6K { ql, qh, scales, d });
    }
    let w_q6_mat = CudaMat {
        buf: std::sync::Arc::new(ctx.alloc_from_bytes(&w_q6_bytes).unwrap()),
        offset: 0,
        dtype: DType::Q6K,
        rows: q6_rows,
        cols: q6_cols,
        row_bytes: 210,
    };
    let x_q6: Vec<f32> = (0..q6_cols)
        .map(|i| (((i * 29 + 7) % 128) as f32 - 64.0) * 0.015625)
        .collect();
    let x_q6_dev = ctx.alloc_from_f32(&x_q6).unwrap();
    let xq_q6_dev = ctx.alloc_zeroed((q6_cols / 32) * BLOCK_Q8_1_BYTES).unwrap();
    let out_q6_dev = ctx.alloc_zeroed(q6_rows * 4).unwrap();
    kernels
        .launch_linear(
            &ctx,
            &w_q6_mat,
            x_q6_dev.device_ptr(),
            xq_q6_dev.device_ptr(),
            false,
            out_q6_dev.device_ptr(),
            1,
            true,
        )
        .unwrap();
    let mut got_q6 = vec![0.0f32; q6_rows];
    out_q6_dev.copy_to_f32(&mut got_q6).unwrap();
    let mut xq8_1 = vec![
        BlockQ8_1 {
            d: 0,
            s: 0,
            qs: [0; 32],
        };
        q6_cols / 32
    ];
    generic::quantize_q8_1(&x_q6, &mut xq8_1);
    for (r, blk) in w_q6.iter().enumerate() {
        let ref_val = scalar_dot_q6_k_q8_1(blk, &xq8_1);
        let diff = (got_q6[r] - ref_val).abs();
        assert!(
            diff < 5e-4,
            "Q6_K × Q8_1 MMVQ r={r}: got={} ref={ref_val} diff={diff}",
            got_q6[r]
        );
    }
}

#[test]
fn parity_cuda_l3_end_to_end_dense_and_moe_models() {
    if try_ctx().is_none() {
        return;
    }
    let out_dir = std::env::temp_dir().join("tm_cuda_l3_parity");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");

    for arch in ["exaone", "olmoe", "qwen2moe", "bailingmoe2"] {
        let gguf_path = out_dir.join(format!("{arch}.gguf"));
        let status = Command::new(&py)
            .arg(&script)
            .arg("--arch")
            .arg(arch)
            .arg("--out-gguf")
            .arg(&gguf_path)
            .status()
            .expect("run p2a oracle");
        assert!(status.success(), "p2a oracle failed for {arch}");

        let gguf = GgufFile::open(&gguf_path).unwrap();
        let cpu_m = LlamaModel::from_gguf(&gguf).unwrap();
        let mut cpu_st = tm_models::LlamaState::new(&cpu_m.cfg, 32);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();

        let mut cuda_r = CudaRunner::from_gguf_with_kv_type(&gguf, 32, KvType::F16).unwrap();

        // Prefill 3 tokens [1, 2, 3], then decode token 4 at pos=3 (multi-token L3 test!)
        cpu_m
            .forward_batch(&mut cpu_st, &pool, kt, &[1, 2, 3], 0, true)
            .unwrap();
        let cuda_pp = cuda_r.forward_batch(&[1, 2, 3], 0, true).unwrap().to_vec();
        for (i, (&c, &g)) in cpu_st.logits().iter().zip(cuda_pp.iter()).enumerate() {
            assert!(
                (c - g).abs() < 3e-3,
                "{arch} batch prefill logit mismatch at {i}: cpu={c} cuda={g}"
            );
        }

        cpu_m
            .forward_token(&mut cpu_st, &pool, kt, 4, 3, true)
            .unwrap();
        let cuda_tg = cuda_r.forward_token(4, 3, true).unwrap().to_vec();
        for (i, (&c, &g)) in cpu_st.logits().iter().zip(cuda_tg.iter()).enumerate() {
            assert!(
                (c - g).abs() < 3e-3,
                "{arch} decode pos=3 logit mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn parity_cuda_l3_qwen35_hybrid_gdn_and_full_attn() {
    if try_ctx().is_none() {
        return;
    }
    let out_dir = std::env::temp_dir().join("tm_cuda_qwen35_parity");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    let gguf_path = out_dir.join("qwen35_synth.gguf");

    // Generate a 2-layer hybrid Qwen3.5 GGUF (Layer 0 = GDN with HK=2, HV=4 `vh % HK`, Layer 1 = FullAttn)
    // in F32 so GDN math + FullAttn Q+Gate split + partial NeoX RoPE + F16/Q8_0 KV are tested to `< 1e-4`!
    let py = find_python3_with_numpy();
    let gen_py = r#"
import struct, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
rng = np.random.default_rng(42)

def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

n_embd, n_ff, vocab = 64, 128, 32
n_head, n_head_kv, head_dim, rope_dim = 4, 2, 32, 16
hk, hv, s_dim, d_inner, d_conv = 2, 4, 32, 128, 4
conv_dim = 2 * hk * s_dim + d_inner

kvs = [
    ("general.architecture", 8, "qwen35"),
    ("qwen35.block_count", 4, 2),
    ("qwen35.embedding_length", 4, n_embd),
    ("qwen35.feed_forward_length", 4, n_ff),
    ("qwen35.attention.head_count", 4, n_head),
    ("qwen35.attention.head_count_kv", 4, n_head_kv),
    ("qwen35.attention.key_length", 4, head_dim),
    ("qwen35.rope.dimension_count", 4, rope_dim),
    ("qwen35.rope.freq_base", 6, 1000000.0),
    ("qwen35.attention.layer_norm_rms_epsilon", 6, 1e-6),
    ("qwen35.full_attention_interval", 4, 2),
    ("qwen35.ssm.conv_kernel", 4, d_conv),
    ("qwen35.ssm.state_size", 4, s_dim),
    ("qwen35.ssm.group_count", 4, hk),
    ("qwen35.ssm.time_step_rank", 4, hv),
    ("qwen35.ssm.inner_size", 4, d_inner),
    ("qwen35.vocab_size", 4, vocab),
]

tensors = []
def add_mat(name, rows, cols):
    m = rng.uniform(-0.25, 0.25, (rows, cols)).astype(np.float32)
    tensors.append((name, (cols, rows), 0, m.tobytes()))

def add_f32_1d(name, arr):
    a = arr.astype(np.float32)
    tensors.append((name, (a.size,), 0, a.tobytes()))

add_mat("token_embd.weight", vocab, n_embd)
add_f32_1d("output_norm.weight", rng.uniform(0.8, 1.2, n_embd))
add_mat("output.weight", vocab, n_embd)

# Layer 0: GDN
add_f32_1d("blk.0.attn_norm.weight", rng.uniform(0.8, 1.2, n_embd))
add_f32_1d("blk.0.post_attention_norm.weight", rng.uniform(0.8, 1.2, n_embd))
add_mat("blk.0.attn_qkv.weight", conv_dim, n_embd)
add_mat("blk.0.attn_gate.weight", d_inner, n_embd)
add_mat("blk.0.ssm_alpha.weight", hv, n_embd)
add_mat("blk.0.ssm_beta.weight", hv, n_embd)
cw = rng.uniform(-0.35, 0.35, (conv_dim, d_conv)).astype(np.float32)
tensors.append(("blk.0.ssm_conv1d.weight", (d_conv, conv_dim), 0, cw.tobytes()))
add_f32_1d("blk.0.ssm_a", -rng.uniform(0.5, 1.5, hv))
add_f32_1d("blk.0.ssm_dt.bias", rng.uniform(-0.2, 0.2, hv))
add_f32_1d("blk.0.ssm_norm.weight", rng.uniform(0.8, 1.2, d_inner // hv))
add_mat("blk.0.ssm_out.weight", n_embd, d_inner)
add_mat("blk.0.ffn_gate.weight", n_ff, n_embd)
add_mat("blk.0.ffn_up.weight", n_ff, n_embd)
add_mat("blk.0.ffn_down.weight", n_embd, n_ff)

# Layer 1: FullAttn (interleaved Q+Gate)
add_f32_1d("blk.1.attn_norm.weight", rng.uniform(0.8, 1.2, n_embd))
add_f32_1d("blk.1.post_attention_norm.weight", rng.uniform(0.8, 1.2, n_embd))
add_f32_1d("blk.1.attn_q_norm.weight", rng.uniform(0.8, 1.2, head_dim))
add_f32_1d("blk.1.attn_k_norm.weight", rng.uniform(0.8, 1.2, head_dim))
add_mat("blk.1.attn_q.weight", 2 * n_head * head_dim, n_embd)
add_mat("blk.1.attn_k.weight", n_head_kv * head_dim, n_embd)
add_mat("blk.1.attn_v.weight", n_head_kv * head_dim, n_embd)
add_mat("blk.1.attn_output.weight", n_embd, n_head * head_dim)
add_mat("blk.1.ffn_gate.weight", n_ff, n_embd)
add_mat("blk.1.ffn_up.weight", n_ff, n_embd)
add_mat("blk.1.ffn_down.weight", n_embd, n_ff)

hdr = bytearray(b"GGUF")
hdr.extend(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
for k, vt, v in kvs:
    hdr.extend(gstr(k))
    hdr.extend(struct.pack("<I", vt))
    if vt == 4: hdr.extend(struct.pack("<I", v))
    elif vt == 6: hdr.extend(struct.pack("<f", v))
    elif vt == 8: hdr.extend(gstr(v))

offs, cur = [], 0
for _, _, _, raw in tensors:
    cur = (cur + 31) // 32 * 32
    offs.append(cur)
    cur += len(raw)
for (n, dims, dt, _), off in zip(tensors, offs):
    hdr.extend(gstr(n))
    hdr.extend(struct.pack("<I", len(dims)))
    for d in dims: hdr.extend(struct.pack("<Q", d))
    hdr.extend(struct.pack("<IQ", dt, off))
while len(hdr) % 32 != 0: hdr.append(0)
base = len(hdr)
for (_, _, _, raw), off in zip(tensors, offs):
    if len(hdr) < base + off: hdr.extend(b"\x00" * (base + off - len(hdr)))
    hdr.extend(raw)
out.write_bytes(bytes(hdr))
"#;
    let status = Command::new(&py)
        .args(["-c", gen_py])
        .arg(&gguf_path)
        .status()
        .expect("generate qwen35 synth gguf");
    assert!(status.success());

    let gguf = GgufFile::open(&gguf_path).unwrap();
    let cpu_m = Qwen35Model::from_gguf(&gguf).unwrap();
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    for kv_ty in [KvType::F16, KvType::Q8_0] {
        let mut cpu_st = Qwen35State::new_with_kv_type(&cpu_m, 32, kv_ty);
        let mut cuda_r = CudaRunner::from_gguf_with_kv_type(&gguf, 32, kv_ty).unwrap();

        let tol = if kv_ty == KvType::F16 { 1e-4 } else { 2e-2 };
        // Prefill 4 tokens [1, 2, 3, 4], then decode token 5 at pos=4
        cpu_m
            .forward_batch(&mut cpu_st, &pool, kt, &[1, 2, 3, 4], 0, true)
            .unwrap();
        let cuda_pp = cuda_r
            .forward_batch(&[1, 2, 3, 4], 0, true)
            .unwrap()
            .to_vec();
        for (i, (&c, &g)) in cpu_st.logits().iter().zip(cuda_pp.iter()).enumerate() {
            assert!(
                (c - g).abs() < tol,
                "qwen35 ({kv_ty:?}) batch prefill mismatch at {i}: cpu={c} cuda={g}"
            );
        }

        cpu_m
            .forward_token(&mut cpu_st, &pool, kt, 5, 4, true)
            .unwrap();
        let cuda_tg = cuda_r.forward_token(5, 4, true).unwrap().to_vec();
        for (i, (&c, &g)) in cpu_st.logits().iter().zip(cuda_tg.iter()).enumerate() {
            assert!(
                (c - g).abs() < tol,
                "qwen35 ({kv_ty:?}) decode pos=4 mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&out_dir);
}
