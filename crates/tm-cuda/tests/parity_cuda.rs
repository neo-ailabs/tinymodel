//! L1 kernel & L3 end-to-end multi-token parity tests for `tm-cuda`.
//!
//! When run on a machine without an NVIDIA GPU / `libcuda` (e.g., Apple M4 macOS or CPU-only CI),
//! tests cleanly report `[skip: no CUDA driver]` and succeed; on NVIDIA GPUs (e.g., Tesla T4 `sm_75`),
//! every kernel and full-model runner is verified against CPU scalar / FP64 reference!

use std::path::PathBuf;
use std::process::Command;

use tm_core::{BlockQ4_0, BlockQ6K, BlockQ8_0, BlockQ8_1, DType, KvType, f16_to_f32, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels, generic};
use tm_cuda::{
    BLOCK_Q8_1_BYTES, CudaContext, CudaKernels, CudaMat, CudaRunner, DFlashCudaRunner,
    Inkling2CudaRunner,
};
use tm_gguf::GgufFile;
use tm_models::{DFlashModel, DFlashState, LlamaModel, Qwen35Model, Qwen35State, argmax};

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
    let home = std::env::var("HOME").unwrap_or_default();
    let home_anaconda = format!("{home}/anaconda3/bin/python3");
    let home_miniconda = format!("{home}/miniconda3/bin/python3");
    let home_venv = format!("{home}/.venv/bin/python3");
    for cand in [
        std::env::var("PYTHON3").ok().as_deref().unwrap_or(""),
        home_anaconda.as_str(),
        home_miniconda.as_str(),
        home_venv.as_str(),
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

#[test]
fn parity_cuda_l3_qwen35moe_hybrid_and_inkling2_dflash() {
    if try_ctx().is_none() {
        return;
    }
    let out_dir = std::env::temp_dir().join(format!("tm_cuda_f1_f2_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    // 1. Qwen3.5-MoE (`qwen35moe`): 2-layer GDN + FullAttn with 8 experts (`top_k = 2`) + gated shared expert!
    let q35m_path = out_dir.join("qwen35moe_synth.gguf");
    let py = find_python3_with_numpy();
    let gen_q35m = r#"
import struct, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
rng = np.random.default_rng(123)
def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

n_embd, n_ff_exp, n_ff_sh, vocab = 64, 64, 64, 32
n_exp, top_k = 8, 2
n_head, n_head_kv, head_dim, rope_dim = 4, 2, 32, 16
hk, hv, s_dim, d_inner, d_conv = 2, 4, 32, 128, 4
conv_dim = 2 * hk * s_dim + d_inner

kvs = [
    ("general.architecture", 8, "qwen35moe"),
    ("qwen35moe.block_count", 4, 2),
    ("qwen35moe.embedding_length", 4, n_embd),
    ("qwen35moe.feed_forward_length", 4, n_ff_exp),
    ("qwen35moe.expert_feed_forward_length", 4, n_ff_exp),
    ("qwen35moe.expert_shared_feed_forward_length", 4, n_ff_sh),
    ("qwen35moe.expert_count", 4, n_exp),
    ("qwen35moe.expert_used_count", 4, top_k),
    ("qwen35moe.attention.head_count", 4, n_head),
    ("qwen35moe.attention.head_count_kv", 4, n_head_kv),
    ("qwen35moe.attention.key_length", 4, head_dim),
    ("qwen35moe.rope.dimension_count", 4, rope_dim),
    ("qwen35moe.rope.freq_base", 6, 1000000.0),
    ("qwen35moe.attention.layer_norm_rms_epsilon", 6, 1e-6),
    ("qwen35moe.full_attention_interval", 4, 2),
    ("qwen35moe.ssm.conv_kernel", 4, d_conv),
    ("qwen35moe.ssm.state_size", 4, s_dim),
    ("qwen35moe.ssm.group_count", 4, hk),
    ("qwen35moe.ssm.time_step_rank", 4, hv),
    ("qwen35moe.ssm.inner_size", 4, d_inner),
    ("qwen35moe.vocab_size", 4, vocab),
]

tensors = []
def add_mat(name, rows, cols):
    m = rng.uniform(-0.25, 0.25, (rows, cols)).astype(np.float32)
    tensors.append((name, (cols, rows), 0, m.tobytes()))
def add_3d(name, n_e, rows, cols):
    m = rng.uniform(-0.2, 0.2, (n_e, rows, cols)).astype(np.float32)
    tensors.append((name, (cols, rows, n_e), 0, m.tobytes()))
def add_f32_1d(name, arr):
    a = arr.astype(np.float32)
    tensors.append((name, (a.size,), 0, a.tobytes()))

add_mat("token_embd.weight", vocab, n_embd)
add_f32_1d("output_norm.weight", rng.uniform(0.8, 1.2, n_embd))
add_mat("output.weight", vocab, n_embd)

for il in range(2):
    add_f32_1d(f"blk.{il}.attn_norm.weight", rng.uniform(0.8, 1.2, n_embd))
    add_f32_1d(f"blk.{il}.post_attention_norm.weight", rng.uniform(0.8, 1.2, n_embd))
    if il == 0:
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
    else:
        add_f32_1d("blk.1.attn_q_norm.weight", rng.uniform(0.8, 1.2, head_dim))
        add_f32_1d("blk.1.attn_k_norm.weight", rng.uniform(0.8, 1.2, head_dim))
        add_mat("blk.1.attn_q.weight", 2 * n_head * head_dim, n_embd)
        add_mat("blk.1.attn_k.weight", n_head_kv * head_dim, n_embd)
        add_mat("blk.1.attn_v.weight", n_head_kv * head_dim, n_embd)
        add_mat("blk.1.attn_output.weight", n_embd, n_head * head_dim)
    add_mat(f"blk.{il}.ffn_gate_inp.weight", n_exp, n_embd)
    add_3d(f"blk.{il}.ffn_gate_exps.weight", n_exp, n_ff_exp, n_embd)
    add_3d(f"blk.{il}.ffn_up_exps.weight", n_exp, n_ff_exp, n_embd)
    add_3d(f"blk.{il}.ffn_down_exps.weight", n_exp, n_embd, n_ff_exp)
    add_mat(f"blk.{il}.ffn_gate_shexp.weight", n_ff_sh, n_embd)
    add_mat(f"blk.{il}.ffn_up_shexp.weight", n_ff_sh, n_embd)
    add_mat(f"blk.{il}.ffn_down_shexp.weight", n_embd, n_ff_sh)
    add_mat(f"blk.{il}.ffn_gate_inp_shexp.weight", 1, n_embd)

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
    assert!(
        Command::new(&py)
            .args(["-c", gen_q35m])
            .arg(&q35m_path)
            .status()
            .unwrap()
            .success()
    );
    let q35m_gguf = GgufFile::open(&q35m_path).unwrap();
    let cpu_q35m = Qwen35Model::from_gguf(&q35m_gguf).unwrap();
    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let mut cpu_q35m_st = Qwen35State::new_with_kv_type(&cpu_q35m, 32, KvType::F16);
    let mut cuda_q35m = CudaRunner::from_gguf_with_kv_type(&q35m_gguf, 32, KvType::F16).unwrap();

    cpu_q35m
        .forward_batch(&mut cpu_q35m_st, &pool, kt, &[1, 2, 3], 0, true)
        .unwrap();
    let cuda_pp = cuda_q35m
        .forward_batch(&[1, 2, 3], 0, true)
        .unwrap()
        .to_vec();
    for (i, (&c, &g)) in cpu_q35m_st.logits().iter().zip(cuda_pp.iter()).enumerate() {
        assert!(
            (c - g).abs() < 2e-4,
            "qwen35moe prefill mismatch at {i}: cpu={c} cuda={g}"
        );
    }
    // Single-token decode x2 (captures and replays CudaGraphExec on MoE + GDN + gated shexp!)
    for (step, &tid) in [4u32, 5].iter().enumerate() {
        let pos = 3 + step;
        cpu_q35m
            .forward_token(&mut cpu_q35m_st, &pool, kt, tid, pos, true)
            .unwrap();
        let cuda_tg = cuda_q35m.forward_token(tid, pos, true).unwrap().to_vec();
        for (i, (&c, &g)) in cpu_q35m_st.logits().iter().zip(cuda_tg.iter()).enumerate() {
            assert!(
                (c - g).abs() < 2e-4,
                "qwen35moe decode pos={pos} mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    // 2. Inkling-2-Mini (`inkling2_loop`) + DSpark (`dflash`):
    //    Compare against `tools/oracle/inkling2_oracle.py` FP64 float-act reference across 7 tokens,
    //    speculative window rollback (`commit_speculative_window`), AND 100%-GPU `DFlashCudaRunner` vs CPU `DFlashModel`!
    let ik_gguf_path = out_dir.join("synthetic_inkling2_loop_q8_0.gguf");
    let ik_ref_path = out_dir.join("ref_q8_act.tmlg");
    assert!(
        Command::new(&py)
            .arg(repo_root().join("tools/oracle/inkling2_oracle.py"))
            .arg("--gen-gguf")
            .arg(&ik_gguf_path)
            .arg("--tokens")
            .arg("5,12,29,41,7,63,18")
            .arg("--out")
            .arg(&ik_ref_path)
            .status()
            .unwrap()
            .success()
    );
    let raw_ref = std::fs::read(&ik_ref_path).unwrap();
    let n_steps = u32::from_le_bytes(raw_ref[8..12].try_into().unwrap()) as usize;
    let n_vocab = u32::from_le_bytes(raw_ref[12..16].try_into().unwrap()) as usize;
    let mut off = 16;
    let mut ik_toks = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        ik_toks.push(u32::from_le_bytes(
            raw_ref[off..off + 4].try_into().unwrap(),
        ));
        off += 4;
    }
    let mut ik_ref = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let mut row = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            row.push(f32::from_le_bytes(
                raw_ref[off..off + 4].try_into().unwrap(),
            ));
            off += 4;
        }
        ik_ref.push(row);
    }

    let ik_gguf = GgufFile::open(&ik_gguf_path).unwrap();
    let mut ik_cuda =
        Inkling2CudaRunner::from_gguf_with_kv_type(&ik_gguf, 64, KvType::F16).unwrap();
    for (step, &tid) in ik_toks.iter().enumerate() {
        let got = ik_cuda.forward_token(tid, step, true).unwrap();
        let want = &ik_ref[step];
        assert_eq!(
            argmax(got),
            argmax(want),
            "Inkling2CudaRunner Top-1 mismatch at step {step}"
        );
        let mut max_diff = 0.0f32;
        for (&a, &b) in got.iter().zip(want.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 5e-3,
            "Inkling2CudaRunner step {step} max_diff={max_diff}"
        );
    }

    // Speculative window (`W = 4`) + ShortConv snapshot rollback (`committed = 2`) -> continue `[2..5]`!
    ik_cuda.reset_state().unwrap();
    let tap_visits = [1usize, 5, 11, 16];
    let mut win_am = [0u32; 4];
    let _ = ik_cuda
        .forward_window_with_taps(
            &ik_toks[0..4],
            0,
            false,
            true,
            &tap_visits,
            Some(&mut win_am),
        )
        .unwrap();
    for s in 0..4 {
        assert_eq!(win_am[s], argmax(&ik_ref[s]));
    }
    ik_cuda.commit_speculative_window(2).unwrap();
    let mut win_am2 = [0u32; 3];
    let _ = ik_cuda
        .forward_window_with_taps(&ik_toks[2..5], 2, false, false, &[], Some(&mut win_am2))
        .unwrap();
    for s in 0..3 {
        assert_eq!(
            win_am2[s],
            argmax(&ik_ref[2 + s]),
            "post-rollback argmax mismatch at {}",
            2 + s
        );
    }

    // Verify P1-2 3-Tier `Inkling2CudaRunner` (`VRAM + DEVICEMAP Host RAM + NvmeDirect` + `NvmeLayerRamCache`)!
    {
        // SAFETY: Single-threaded test (`--test-threads=1`) temporarily sets tiering env vars.
        unsafe {
            std::env::set_var("TM_CUDA_INKLING2_VRAM_MIB", "0");
            std::env::set_var("TM_INKLING2_CACHE_EXPERTS", "2");
        }
        let mut ik_tiered =
            Inkling2CudaRunner::from_gguf_with_kv_type(&ik_gguf, 64, KvType::F16).unwrap();
        assert!(ik_tiered.has_host_offload_layers());
        let hm = tm_models::Inkling2Model::from_gguf_with_options(
            &ik_gguf,
            &tm_models::Inkling2LoadOptions {
                vram_expert_layers: ik_tiered.vram_expert_mask().to_vec(),
                ram_expert_budget_bytes: Some(0), // Force all offloaded layers onto `NvmeDirect` + `NvmeLayerRamCache`!
                host_moe_only: true,
            },
        )
        .unwrap();
        assert!(hm.stats.nvme_direct_expert_layers > 0);
        assert_eq!(hm.stats.nvme_cache_slots_per_layer, 2);
        let mut hs = tm_models::Inkling2State::new_host_moe_only(&hm);
        let pool_h = ThreadPool::new(2);
        let kt_h = active_kernels();

        for (step, &tid) in ik_toks.iter().enumerate() {
            let mut cb =
                |ip: usize, cur_n: &[f32], t_ids: &[usize], t_w: &[f32], m_out: &mut [f32]| {
                    let mut hctx = hs.host_moe_ctx_mut();
                    hm.eval_moe_experts_ctx(
                        ip, cur_n, t_ids, t_w, &pool_h, kt_h, &mut hctx, m_out, false,
                    );
                };
            let got = ik_tiered
                .forward_token_hybrid(tid, step, true, Some(&mut cb))
                .unwrap();
            assert_eq!(
                argmax(got),
                argmax(&ik_ref[step]),
                "3-tier Inkling2CudaRunner Top-1 mismatch at step {step}"
            );
        }
        ik_tiered.reset_state().unwrap();
        let mut win_am_t = [0u32; 4];
        {
            let mut cb_w = |ip: usize,
                            w_len: usize,
                            cur_w: &[f32],
                            ids_w: &[usize],
                            ws_w: &[f32],
                            out_w: &mut [f32]| {
                let mut hctx = hs.host_moe_ctx_mut();
                hm.eval_moe_experts_window_ctx(
                    ip, w_len, cur_w, ids_w, ws_w, &pool_h, kt_h, &mut hctx, out_w, false,
                );
            };
            let _ = ik_tiered
                .forward_window_hybrid(
                    &ik_toks[0..4],
                    0,
                    false,
                    false,
                    &[],
                    Some(&mut cb_w),
                    Some(&mut win_am_t),
                )
                .unwrap();
        }
        for s in 0..4 {
            assert_eq!(
                win_am_t[s],
                argmax(&ik_ref[s]),
                "3-tier Inkling2CudaRunner window mismatch at {s}"
            );
        }
        // SAFETY: Restore environment variables before continuing.
        unsafe {
            std::env::remove_var("TM_CUDA_INKLING2_VRAM_MIB");
            std::env::remove_var("TM_INKLING2_CACHE_EXPERTS");
        }
    }

    // Also verify DFlashCudaRunner (`DSpark`) builds and runs deterministic `inject + draft` on GPU!
    let dflash_py = r#"
import struct, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
n_embd, vocab_pad, vocab_sz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(777)
def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

n_layers, block_size, n_ff = 2, 4, 128
n_head, n_head_kv = 4, 2
head_dim = n_embd // n_head
q_dim, kv_dim = n_head * head_dim, n_head_kv * head_dim
markov_rank, n_taps = 32, 4

kvs = [
    ("general.architecture", 8, "dflash"),
    ("dflash.block_count", 4, n_layers),
    ("dflash.block_size", 4, block_size),
    ("dflash.embedding_length", 4, n_embd),
    ("dflash.feed_forward_length", 4, n_ff),
    ("dflash.attention.head_count", 4, n_head),
    ("dflash.attention.head_count_kv", 4, n_head_kv),
    ("dflash.attention.key_length", 4, head_dim),
    ("dflash.rope.dimension_count", 4, head_dim),
    ("dflash.rope.freq_base", 6, 1000000.0),
    ("dflash.attention.layer_norm_rms_epsilon", 6, 1e-6),
    ("dflash.attention.sliding_window", 4, 4),
    ("dflash.attention.sliding_window_pattern", 97, [True, False]),
    ("dflash.target_layers", 94, [1, 5, 11, 16]),
    ("dflash.dspark.markov_rank", 4, markov_rank),
    ("dflash.vocab_size", 4, vocab_pad),
    ("dflash.vocab_size_unpadded", 4, vocab_sz),
    ("dflash.final_logit_softcapping", 6, 30.0),
]
tensors = []
def add_f32(name, dims, lo=-0.2, hi=0.2):
    a = rng.uniform(lo, hi, dims[::-1]).astype(np.float32)
    tensors.append((name, dims, 0, a.tobytes()))

add_f32("dspark_fuse.weight", (n_taps, n_layers + 1), -0.5, 0.5)
add_f32("dspark_norm.weight", (n_embd,), 0.85, 1.15)
add_f32("mask_embd.weight", (n_embd,), -0.2, 0.2)
add_f32("output_norm.weight", (n_embd,), 0.85, 1.15)
add_f32("markov_w1.weight", (markov_rank, vocab_pad), -0.2, 0.2)
add_f32("joint_proj.weight", (2 * n_embd + 2 * markov_rank, 3 * markov_rank), -0.05, 0.05)
add_f32("joint_proj.bias", (3 * markov_rank,), -0.05, 0.05)
add_f32("dspark_hidden_out.weight", (markov_rank, n_embd), -0.1, 0.1)
add_f32("dspark_hidden_out.bias", (n_embd,), -0.05, 0.05)
for il in range(n_layers):
    p = f"blk.{il}"
    add_f32(f"{p}.dflare_norm.weight", (n_embd,), 0.85, 1.15)
    add_f32(f"{p}.attn_k_ctx.weight", (n_embd, kv_dim), -0.1, 0.1)
    add_f32(f"{p}.attn_v_ctx.weight", (n_embd, kv_dim), -0.1, 0.1)
    add_f32(f"{p}.attn_k_ctx_norm.weight", (head_dim,), 0.9, 1.1)
    add_f32(f"{p}.attn_norm.weight", (n_embd,), 0.85, 1.15)
    add_f32(f"{p}.attn_q.weight", (n_embd, q_dim), -0.1, 0.1)
    add_f32(f"{p}.attn_k.weight", (n_embd, kv_dim), -0.1, 0.1)
    add_f32(f"{p}.attn_v.weight", (n_embd, kv_dim), -0.1, 0.1)
    add_f32(f"{p}.attn_output.weight", (q_dim, n_embd), -0.1, 0.1)
    add_f32(f"{p}.attn_q_norm.weight", (head_dim,), 0.9, 1.1)
    add_f32(f"{p}.attn_k_norm.weight", (head_dim,), 0.9, 1.1)
    add_f32(f"{p}.ffn_norm.weight", (n_embd,), 0.85, 1.15)
    add_f32(f"{p}.ffn_gate.weight", (n_embd, n_ff), -0.1, 0.1)
    add_f32(f"{p}.ffn_up.weight", (n_embd, n_ff), -0.1, 0.1)
    add_f32(f"{p}.ffn_down.weight", (n_ff, n_embd), -0.1, 0.1)

hdr = bytearray(b"GGUF")
hdr.extend(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
for k, vt, v in kvs:
    hdr.extend(gstr(k))
    if vt == 94:
        hdr.extend(struct.pack("<IIQ", 9, 4, len(v)))
        for x in v: hdr.extend(struct.pack("<I", x))
    elif vt == 97:
        hdr.extend(struct.pack("<IIQ", 9, 7, len(v)))
        for x in v: hdr.append(1 if x else 0)
    else:
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
    let dflash_path = out_dir.join("dflash_synth.gguf");
    assert!(
        Command::new(&py)
            .args([
                "-c",
                dflash_py,
                dflash_path.to_str().unwrap(),
                &ik_cuda.cfg.n_embd.to_string(),
                &ik_cuda.cfg.vocab_padded.to_string(),
                &ik_cuda.cfg.vocab_size.to_string(),
            ])
            .status()
            .unwrap()
            .success()
    );
    let df_gguf = GgufFile::open(&dflash_path).unwrap();
    let _df_cpu = DFlashModel::from_gguf(&df_gguf).unwrap();
    let _df_st = DFlashState::new(&_df_cpu, 64);
    let mut df_cuda = DFlashCudaRunner::from_gguf(&ik_cuda, &df_gguf, 64).unwrap();
    ik_cuda.reset_state().unwrap();
    let prompt = &ik_toks[0..5];
    let mut prompt_am = [0u32; 5];
    let _ = ik_cuda
        .forward_window_with_taps(prompt, 0, false, false, &tap_visits, Some(&mut prompt_am))
        .unwrap();
    let mut drafts_a = [0u32; 3];
    let mut probs_a = [0.0f32; 3];
    let n_drafted = df_cuda
        .inject_and_draft_greedy_with_probs(
            &mut ik_cuda,
            0,
            prompt.len(),
            prompt.len() - 1,
            prompt_am[4],
            3,
            &mut drafts_a,
            &mut probs_a,
        )
        .unwrap();
    assert_eq!(n_drafted, 3);
    for (&d_id, &p) in drafts_a.iter().zip(probs_a.iter()) {
        assert!((d_id as usize) < ik_cuda.cfg.vocab_size);
        assert!(p > 0.0 && p <= 1.0);
    }
    // Reset & repeat must be bit-for-bit identical!
    df_cuda.reset_state(&ik_cuda).unwrap();
    ik_cuda.reset_state().unwrap();
    let _ = ik_cuda
        .forward_window_with_taps(prompt, 0, false, false, &tap_visits, None)
        .unwrap();
    let mut drafts_b = [0u32; 3];
    let mut probs_b = [0.0f32; 3];
    df_cuda
        .inject_and_draft_greedy_with_probs(
            &mut ik_cuda,
            0,
            prompt.len(),
            prompt.len() - 1,
            prompt_am[4],
            3,
            &mut drafts_b,
            &mut probs_b,
        )
        .unwrap();
    assert_eq!(drafts_a, drafts_b);
    assert_eq!(probs_a, probs_b);

    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn parity_cuda_l3_qwen4exp_multishard_ple_qsa_gdn_moe() {
    if try_ctx().is_none() {
        return;
    }
    let out_dir = std::env::temp_dir().join(format!("tm_cuda_qwen4exp_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    // Build a 2-shard `qwen4exp` GGUF (`...-00001-of-00002.gguf` + `...-00002-of-00002.gguf`):
    // - 2 layers (`full_attention_interval = 2`): Layer 0 = Sigmoid-GDN, Layer 1 = QSA + PLE (`ple.layers = [1]`)
    // - 4-stream HC (`hc_count = 4, hc_low_rank = 32`), `n_embd = 256` (`hc_dim = 1024`)
    // - 16 fine-grained `Q2_0` experts (`top_k = 3`, `n_ff_exp = 64`) + gated shared expert (`Q5_0` / `Q4_K` / `Q3_K` / `IQ4_XS` / `IQ4_NL` / `BF16` mixed GSQ-RCO dtypes!)
    // - Shard 2 holds `per_layer_token_embd.weight` in `IQ4_NL` (`dt = 20`)
    // - `compress_ratios = [0, 2]`, `indexer.top_k = 4` (`top_blocks = 2`) -> at `pos >= 5` (`n_vis_blocks = 3 > 2`), QSA sparse block masking actively fires on both CPU and CUDA!
    let shard1 = out_dir.join("qwen4exp-synth-00001-of-00002.gguf");
    let shard2 = out_dir.join("qwen4exp-synth-00002-of-00002.gguf");
    let py = find_python3_with_numpy();
    let gen_py = r#"
import struct, sys
from pathlib import Path
import numpy as np

s1, s2 = Path(sys.argv[1]), Path(sys.argv[2])
use_f32 = sys.argv[3] == "f32"
rng = np.random.default_rng(2026)

def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

def write_gguf(path, kvs, tensors):
    hdr = bytearray(b"GGUF")
    hdr.extend(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for k, vt, v in kvs:
        hdr.extend(gstr(k))
        if vt == 94:
            hdr.extend(struct.pack("<IIQ", 9, 4, len(v)))
            for x in v: hdr.extend(struct.pack("<I", x))
        elif vt == 910:
            hdr.extend(struct.pack("<IIQ", 9, 10, len(v)))
            for x in v: hdr.extend(struct.pack("<Q", x))
        else:
            hdr.extend(struct.pack("<I", vt))
            if vt == 2: hdr.extend(struct.pack("<H", v))
            elif vt == 4: hdr.extend(struct.pack("<I", v))
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
    path.write_bytes(bytes(hdr))

def pack_q2_0(rows, cols):
    # 18 bytes per 64 weights: f16 d + 16 packed bytes (q in 0..3 -> q - 2)
    out = bytearray()
    for _ in range(rows * (cols // 64)):
        d = np.float16(rng.uniform(0.02, 0.08)).tobytes()
        qs = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        out.extend(d + qs)
    return bytes(out)

def pack_iq4_nl(rows, cols):
    # 18 bytes per 32 weights: f16 d + 16 nibble bytes
    out = bytearray()
    for _ in range(rows * (cols // 32)):
        d = np.float16(rng.uniform(0.005, 0.02)).tobytes()
        qs = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        out.extend(d + qs)
    return bytes(out)

def pack_q5_0(rows, cols):
    # 22 bytes per 32 weights: f16 d + 4 qh + 16 qs
    out = bytearray()
    for _ in range(rows * (cols // 32)):
        d = np.float16(rng.uniform(0.005, 0.02)).tobytes()
        qh = rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()
        qs = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        out.extend(d + qh + qs)
    return bytes(out)

def pack_q4_k(rows, cols):
    # 144 bytes per 256 weights: f16 d + f16 dmin + 12 scales + 128 qs
    out = bytearray()
    for _ in range(rows * (cols // 256)):
        d = np.float16(rng.uniform(0.002, 0.01)).tobytes()
        dm = np.float16(rng.uniform(0.001, 0.005)).tobytes()
        sc = rng.integers(1, 16, size=12, dtype=np.uint8).tobytes()
        qs = rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()
        out.extend(d + dm + sc + qs)
    return bytes(out)

def pack_q3_k(rows, cols):
    # 110 bytes per 256 weights: 32 hmask + 64 qs + 12 scales + f16 d
    out = bytearray()
    for _ in range(rows * (cols // 256)):
        hm = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        qs = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
        sc = rng.integers(0, 256, size=12, dtype=np.uint8).tobytes()
        d = np.float16(rng.uniform(0.003, 0.01)).tobytes()
        out.extend(hm + qs + sc + d)
    return bytes(out)

def pack_iq4_xs(rows, cols):
    # 136 bytes per 256 weights: f16 d + u16 scales_h + 4 scales_l + 128 qs
    out = bytearray()
    for _ in range(rows * (cols // 256)):
        d = np.float16(rng.uniform(0.0005, 0.002)).tobytes()
        sh = struct.pack("<H", int(rng.integers(0, 65536)))
        sl = rng.integers(0, 256, size=4, dtype=np.uint8).tobytes()
        qs = rng.integers(0, 256, size=128, dtype=np.uint8).tobytes()
        out.extend(d + sh + sl + qs)
    return bytes(out)

n_embd, hc, hc_lr, vocab = 256, 4, 32, 32
hc_dim = hc * n_embd # 1024
n_exp, top_k, n_ff_exp, n_ff_sh = 16, 3, 64, 256
n_head, n_head_kv, head_dim, rope_dim = 4, 2, 64, 16
hk, hv, s_dim, d_inner, d_conv = 2, 4, 32, 128, 4
conv_dim = 2 * hk * s_dim + d_inner
idx_nh, idx_d, idx_topk = 2, 32, 4
ple_ng, ple_hpg, ple_hd, ple_ck = 3, 4, 32, 3 # ple_nh = 2 * 4 = 8, 8 * 32 = 256 == n_embd!

kvs1 = [
    ("general.architecture", 8, "qwen4exp"),
    ("split.no", 2, 0),
    ("split.count", 2, 2),
    ("qwen4exp.block_count", 4, 2),
    ("qwen4exp.embedding_length", 4, n_embd),
    ("qwen4exp.hyper_connection.count", 4, hc),
    ("qwen4exp.hyper_connection.low_rank", 4, hc_lr),
    ("qwen4exp.expert_count", 4, n_exp),
    ("qwen4exp.expert_used_count", 4, top_k),
    ("qwen4exp.expert_feed_forward_length", 4, n_ff_exp),
    ("qwen4exp.expert_shared_feed_forward_length", 4, n_ff_sh),
    ("qwen4exp.attention.head_count", 4, n_head),
    ("qwen4exp.attention.head_count_kv", 4, n_head_kv),
    ("qwen4exp.attention.key_length", 4, head_dim),
    ("qwen4exp.rope.dimension_count", 4, rope_dim),
    ("qwen4exp.rope.freq_base", 6, 10000000.0),
    ("qwen4exp.attention.layer_norm_rms_epsilon", 6, 1e-6),
    ("qwen4exp.full_attention_interval", 4, 2),
    ("qwen4exp.ssm.conv_kernel", 4, d_conv),
    ("qwen4exp.ssm.state_size", 4, s_dim),
    ("qwen4exp.ssm.group_count", 4, hk),
    ("qwen4exp.ssm.time_step_rank", 4, hv),
    ("qwen4exp.ssm.inner_size", 4, d_inner),
    ("qwen4exp.attention.indexer.head_count", 4, idx_nh),
    ("qwen4exp.attention.indexer.key_length", 4, idx_d),
    ("qwen4exp.attention.indexer.top_k", 4, idx_topk),
    ("qwen4exp.attention.compress_ratios", 94, [0, 2]),
    ("qwen4exp.ple.layers", 94, [1]),
    ("qwen4exp.ple.ngram_size", 4, ple_ng),
    ("qwen4exp.ple.heads_per_ngram", 4, ple_hpg),
    ("qwen4exp.embedding_length_per_layer_input", 4, ple_hd),
    ("qwen4exp.ple.conv_kernel", 4, ple_ck),
    ("qwen4exp.ple.eos_token_id", 4, 31),
    ("qwen4exp.ple.image_token_id", 4, 30),
    ("qwen4exp.ple.layer_multipliers", 910, [1315423911, 2654435761, 2246822519]),
    ("qwen4exp.ple.head_offsets", 910, [i * 16 for i in range(8)]),
    ("qwen4exp.ple.head_vocab_sizes", 910, [16 for _ in range(8)]),
]

t1 = []
def add_f32_1d(name, n, lo=0.85, hi=1.15):
    a = rng.uniform(lo, hi, n).astype(np.float32)
    t1.append((name, (n,), 0, a.tobytes()))
def add_bf16_1d(name, n, lo=0.85, hi=1.15):
    a = rng.uniform(lo, hi, n).astype(np.float32)
    bf = (a.view(np.uint32) >> 16).astype(np.uint16)
    t1.append((name, (n,), 30, bf.tobytes()))
def add_f32_mat(name, rows, cols, s=0.12):
    a = rng.uniform(-s, s, (rows, cols)).astype(np.float32)
    t1.append((name, (cols, rows), 0, a.tobytes()))

def add_qmat(t_list, name, rows, cols, dt, packed_bytes):
    if use_f32:
        a = rng.uniform(-0.12, 0.12, (rows, cols)).astype(np.float32)
        t_list.append((name, (cols, rows), 0, a.tobytes()))
    else:
        t_list.append((name, (cols, rows), dt, packed_bytes))

def add_q3d(t_list, name, n_e, rows, cols, dt, packed_bytes):
    if use_f32:
        a = rng.uniform(-0.12, 0.12, (n_e, rows, cols)).astype(np.float32)
        t_list.append((name, (cols, rows, n_e), 0, a.tobytes()))
    else:
        t_list.append((name, (cols, rows, n_e), dt, packed_bytes))

add_qmat(t1, "token_embd.weight", vocab, n_embd, 20, pack_iq4_nl(vocab, n_embd))
add_qmat(t1, "output.weight", vocab, n_embd, 12, pack_q4_k(vocab, n_embd))
add_bf16_1d("output_hc_norm.weight", hc_dim)
add_qmat(t1, "output_hc_down.weight", hc_lr, hc_dim, 6, pack_q5_0(hc_lr, hc_dim))
add_f32_mat("output_hc_up.weight", hc_dim, hc_lr, 0.08)

for il in range(2):
    p = f"blk.{il}"
    for kind in ["attn", "ffn"]:
        add_bf16_1d(f"{p}.hc_{kind}_norm.weight", hc_dim)
        add_qmat(t1, f"{p}.hc_{kind}_down.weight", hc_lr, hc_dim, 23, pack_iq4_xs(hc_lr, hc_dim))
        add_f32_mat(f"{p}.hc_{kind}_up.weight", hc_dim, hc_lr, 0.08)
        add_f32_mat(f"{p}.hc_{kind}_inject.weight", hc, hc_dim, 0.08)

    if il == 0:
        add_qmat(t1, f"{p}.attn_qkv.weight", conv_dim, n_embd, 11, pack_q3_k(conv_dim, n_embd))
        add_qmat(t1, f"{p}.attn_gate.weight", d_inner, n_embd, 12, pack_q4_k(d_inner, n_embd))
        add_f32_mat(f"{p}.ssm_alpha.weight", hv, n_embd, 0.1)
        add_f32_mat(f"{p}.ssm_beta.weight", hv, n_embd, 0.1)
        cw = rng.uniform(-0.25, 0.25, (conv_dim, d_conv)).astype(np.float32)
        t1.append((f"{p}.ssm_conv1d.weight", (d_conv, conv_dim), 0, cw.tobytes()))
        add_f32_1d(f"{p}.ssm_a", hv, -1.4, -0.6)
        add_f32_1d(f"{p}.ssm_dt.bias", hv, -0.1, 0.1)
        add_f32_1d(f"{p}.ssm_norm.weight", s_dim)
        add_f32_mat(f"{p}.ssm_out.weight", n_embd, d_inner, 0.1)
    else:
        add_bf16_1d(f"{p}.attn_q_norm.weight", head_dim)
        add_bf16_1d(f"{p}.attn_k_norm.weight", head_dim)
        add_qmat(t1, f"{p}.attn_q.weight", 2 * n_head * head_dim, n_embd, 12, pack_q4_k(2 * n_head * head_dim, n_embd))
        add_qmat(t1, f"{p}.attn_k.weight", n_head_kv * head_dim, n_embd, 6, pack_q5_0(n_head_kv * head_dim, n_embd))
        add_qmat(t1, f"{p}.attn_v.weight", n_head_kv * head_dim, n_embd, 20, pack_iq4_nl(n_head_kv * head_dim, n_embd))
        add_qmat(t1, f"{p}.attn_output.weight", n_embd, n_head * head_dim, 23, pack_iq4_xs(n_embd, n_head * head_dim))
        add_f32_mat(f"{p}.indexer.q_proj.weight", idx_nh * idx_d, n_embd, 0.35)
        add_f32_mat(f"{p}.indexer.k_proj.weight", idx_d, n_embd, 0.35)
        add_bf16_1d(f"{p}.indexer.q_norm.weight", idx_d)
        add_bf16_1d(f"{p}.indexer.k_norm.weight", idx_d)
        # PLE weights on layer 1
        add_qmat(t1, f"{p}.ple_key.weight", hc_dim, n_embd, 12, pack_q4_k(hc_dim, n_embd))
        add_qmat(t1, f"{p}.ple_value.weight", n_embd, n_embd, 6, pack_q5_0(n_embd, n_embd))
        add_bf16_1d(f"{p}.ple_norm_key.weight", hc_dim)
        add_bf16_1d(f"{p}.ple_norm_query.weight", hc_dim)
        add_bf16_1d(f"{p}.ple_norm_conv.weight", hc_dim)
        pcw = rng.uniform(-0.2, 0.2, (hc_dim, ple_ck)).astype(np.float32)
        t1.append((f"{p}.ple_conv1d.weight", (ple_ck, hc_dim), 0, pcw.tobytes()))

    add_f32_mat(f"{p}.ffn_gate_inp.weight", n_exp, n_embd, 0.45)
    # Give experts well-separated router bias so F32-act vs Q8_1-act (~1e-3) never flips top-3 discrete expert IDs!
    eb = (np.linspace(-1.5, 1.5, n_exp) * (1.0 if il == 0 else -1.0)).astype(np.float32)
    t1.append((f"{p}.exp_probs_b.bias", (n_exp,), 0, eb.tobytes()))
    add_q3d(t1, f"{p}.ffn_gate_exps.weight", n_exp, n_ff_exp, n_embd, 42 if il == 0 else 6, pack_q2_0(n_exp * n_ff_exp, n_embd) if il == 0 else pack_q5_0(n_exp * n_ff_exp, n_embd))
    add_q3d(t1, f"{p}.ffn_up_exps.weight", n_exp, n_ff_exp, n_embd, 11 if il == 0 else 23, pack_q3_k(n_exp * n_ff_exp, n_embd) if il == 0 else pack_iq4_xs(n_exp * n_ff_exp, n_embd))
    add_q3d(t1, f"{p}.ffn_down_exps.weight", n_exp, n_embd, n_ff_exp, 42 if il == 0 else 20, pack_q2_0(n_exp * n_embd, n_ff_exp) if il == 0 else pack_iq4_nl(n_exp * n_embd, n_ff_exp))
    add_qmat(t1, f"{p}.ffn_gate_shexp.weight", n_ff_sh, n_embd, 12, pack_q4_k(n_ff_sh, n_embd))
    add_qmat(t1, f"{p}.ffn_up_shexp.weight", n_ff_sh, n_embd, 11, pack_q3_k(n_ff_sh, n_embd))
    add_qmat(t1, f"{p}.ffn_down_shexp.weight", n_embd, n_ff_sh, 6, pack_q5_0(n_embd, n_ff_sh))
    add_f32_mat(f"{p}.ffn_gate_inp_shexp.weight", 1, n_embd, 0.2)

write_gguf(s1, kvs1, t1)

kvs2 = [("split.no", 2, 1), ("split.count", 2, 2)]
t2 = []
add_qmat(t2, "per_layer_token_embd.weight", 128, ple_hd, 20, pack_iq4_nl(128, ple_hd))
write_gguf(s2, kvs2, t2)
"#;
    for mode in ["f32", "gsq"] {
        assert!(
            Command::new(&py)
                .args([
                    "-c",
                    gen_py,
                    shard1.to_str().unwrap(),
                    shard2.to_str().unwrap(),
                    mode,
                ])
                .status()
                .unwrap()
                .success()
        );

        let gguf = GgufFile::open(&shard1).unwrap();
        let cpu_m = tm_models::Qwen4ExpModel::from_gguf(&gguf).unwrap();
        let mut cpu_st = tm_models::Qwen4ExpState::new(&cpu_m, 32);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();

        let mut cuda_r = {
            let tmp_gguf = GgufFile::open(&shard1).unwrap();
            CudaRunner::from_gguf_with_kv_type(&tmp_gguf, 32, KvType::F16).unwrap()
        };

        let pp = [2u32, 5, 31, 7, 9];
        cpu_m
            .forward_batch(&mut cpu_st, &pool, kt, &pp, 0, true)
            .unwrap();
        let cuda_pp = cuda_r.forward_batch(&pp, 0, true).unwrap().to_vec();
        let cos_sim = |a: &[f32], b: &[f32]| -> f32 {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / (na.sqrt() * nb.sqrt()).max(1e-12)
        };
        let mut max_diff_pp = 0.0f32;
        for (&c, &g) in cpu_st.logits().iter().zip(cuda_pp.iter()) {
            max_diff_pp = max_diff_pp.max((c - g).abs());
        }
        if mode == "f32" {
            assert!(
                max_diff_pp < 2e-4,
                "qwen4exp F32 prefill max_diff={max_diff_pp} (cpu={:?}, cuda={:?})",
                &cpu_st.logits()[..6],
                &cuda_pp[..6]
            );
        } else {
            let cs = cos_sim(cpu_st.logits(), &cuda_pp);
            assert!(
                cs > 0.996,
                "qwen4exp GSQ-RCO prefill cos_sim={cs} max_diff={max_diff_pp}"
            );
        }
        assert_eq!(argmax(cpu_st.logits()), argmax(&cuda_pp));

        for (step, &tok) in [11u32, 14, 3].iter().enumerate() {
            let pos = pp.len() + step;
            cpu_m
                .forward_token(&mut cpu_st, &pool, kt, tok, pos, true)
                .unwrap();
            let cuda_tg = cuda_r.forward_token(tok, pos, true).unwrap().to_vec();
            let mut max_diff_tg = 0.0f32;
            for (&c, &g) in cpu_st.logits().iter().zip(cuda_tg.iter()) {
                max_diff_tg = max_diff_tg.max((c - g).abs());
            }
            if mode == "f32" {
                assert!(
                    max_diff_tg < 2e-4,
                    "qwen4exp F32 decode pos={pos} max_diff={max_diff_tg} (cpu={:?}, cuda={:?})",
                    &cpu_st.logits()[..6],
                    &cuda_tg[..6]
                );
                assert_eq!(
                    argmax(cpu_st.logits()),
                    argmax(&cuda_tg),
                    "qwen4exp (f32) decode argmax mismatch at pos={pos}"
                );
            } else {
                let cs = cos_sim(cpu_st.logits(), &cuda_tg);
                assert!(
                    cs > 0.998,
                    "qwen4exp GSQ-RCO decode pos={pos} cos_sim={cs} max_diff={max_diff_tg}"
                );
            }
        }

        // Verify `Qwen4ExpCudaRunner::verify_window_greedy` PLE + GDN rollback determinism!
        cuda_r.reset_state().unwrap();
        let _ = cuda_r.forward_batch(&pp, 0, true).unwrap();
        let mut ref_q4 = Vec::with_capacity(6);
        let mut cur_q4 = 11u32;
        for s in 0..6 {
            cur_q4 = cuda_r.forward_token_greedy(cur_q4, pp.len() + s).unwrap();
            ref_q4.push(cur_q4);
        }
        cuda_r.reset_state().unwrap();
        let _ = cuda_r.forward_batch(&pp, 0, true).unwrap();
        let mut spec_q4 = Vec::with_capacity(6);
        let mut comm_q4 = Vec::new();
        let mut pos_q4 = pp.len();
        let mut tok_q4 = 11u32;
        // Step 1: pass 1 matching draft (`ref_q4[0]`) + 2 intentionally wrong drafts (`31, 30`) to force partial rollback!
        let fake_drafts = [ref_q4[0], (ref_q4[1] + 7) % 30, 29];
        let (_d, acc_q4) = cuda_r
            .verify_window_greedy(tok_q4, &fake_drafts, pos_q4, &mut comm_q4)
            .unwrap();
        assert_eq!(acc_q4, 1);
        for &ct in &comm_q4 {
            tok_q4 = ct;
            spec_q4.push(ct);
            pos_q4 += 1;
        }
        while spec_q4.len() < 6 {
            tok_q4 = cuda_r.forward_token_greedy(tok_q4, pos_q4).unwrap();
            spec_q4.push(tok_q4);
            pos_q4 += 1;
        }
        assert_eq!(
            ref_q4, spec_q4,
            "qwen4exp ({mode}) verify_window_greedy rollback diverged from greedy baseline!"
        );
    }

    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn parity_cuda_l3_hy4_strata_engram_mla_latmoe() {
    if try_ctx().is_none() {
        return;
    }
    let out_dir = std::env::temp_dir().join(format!("tm_cuda_hy4_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    // Build a synthetic `.strata` v1.0 (`Hy4Flash`, `profile = 1`) + sibling `EngramShard` (`profile = 2`):
    // - 2 layers:
    //   * Layer 0: Engram + 4-stream iHC + SWA MLA (`heads_swa = 2`, `sliding_window = 4` ring buffer + sinks) + Dense SwiGLU MLP
    //   * Layer 1: Engram + 4-stream iHC + Sparse MLA (`heads_sparse = 4` + sinks) + `LatentMoE` (`4` experts in Sec 4, `top_k = 2`, SmoothQuant `sq[d]`, clamped SwiGLU + Shared Expert)
    // - Exercises BOTH `F32` (`max_diff < 2e-4`, exact `argmax`) AND mixed quantized `S8_K` / `S6_K` / `S4_K` / `S3_K` / `S2_K` / `Q8_ROW` (`cos_sim > 0.998`)!
    // - Exercises BOTH Section 2 `ENGR` hot-seed hits AND external `EngramShard` Section 4 `4096 B` page-LRU misses, plus `Tokenizer::from_hf_tokenizer_json`!
    let model_strata = out_dir.join("Hy4-Synth-SQ-64.strata");
    let engram_strata = out_dir.join("Hy4-Synth-Engram-Q8.strata");
    let py = find_python3_with_numpy();
    let gen_py = r#"
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np

m_path, e_path = Path(sys.argv[1]), Path(sys.argv[2])
use_f32 = sys.argv[3] == "f32"
rng = np.random.default_rng(4242)

def pack_strata_file(path, profile_id, secs, eng_sha=b"\x00" * 32):
    # secs: list of 6 tuples `(raw_bytes, sha32_or_None, flags)`
    frame = bytearray(4096)
    frame[0:8] = b"STRATA\x00\x01"
    struct.pack_into("<IHH", frame, 8, 0, profile_id, 6)
    cur = 4096
    payloads = []
    for idx in range(6):
        raw, sha_ovr, flg = secs[idx]
        cur = (cur + 4095) // 4096 * 4096
        off = cur if len(raw) > 0 else 0
        sha = sha_ovr if sha_ovr is not None else (hashlib.sha256(raw).digest() if len(raw) > 0 else b"\x00" * 32)
        struct.pack_into("<QQ32sIIQ", frame, 16 + idx * 64, off, len(raw), sha, idx, flg, 0)
        payloads.append((off, raw))
        cur += len(raw)
    frame[0x190:0x1B0] = eng_sha
    out = bytearray(frame)
    for off, raw in payloads:
        if len(raw) == 0:
            continue
        if len(out) < off:
            out.extend(b"\x00" * (off - len(out)))
        out.extend(raw)
    path.write_bytes(bytes(out))

def pack_q8_row(mat):
    # mat: [rows, cols] float32 -> f32 scales[rows] + int8 qs[rows, cols]
    rows, cols = mat.shape
    mx = np.maximum(np.max(np.abs(mat), axis=1), 1e-8)
    sc = (mx / 127.0).astype(np.float32)
    q = np.clip(np.round(mat / sc[:, None]), -127, 127).astype(np.int8)
    return sc.tobytes() + q.tobytes()

def pack_sk_matrix(mat, code):
    # mat: [rows, cols] float32 (`cols % 256 == 0`) -> S_K superblocks (`code` in {2,3,4,6,8})
    rows, cols = mat.shape
    assert cols % 256 == 0
    out = bytearray()
    for r in range(rows):
        for sb in range(cols // 256):
            blk = mat[r, sb * 256:(sb + 1) * 256]
            d_sb = np.float16(0.025)
            d_f = float(d_sb)
            sub_u8 = np.array([200, 180, 220, 240], dtype=np.uint8)
            hdr = d_sb.tobytes() + sub_u8.tobytes()
            if code == 8:
                pl = bytearray(256)
                for lane in range(32):
                    eff = d_f * (float(sub_u8[lane >> 3]) / 255.0)
                    for j in range(8):
                        q = int(np.clip(np.round(blk[lane * 8 + j] / eff), -127, 127))
                        pl[lane * 8 + j] = q & 0xFF
                out.extend(hdr + bytes(pl))
            elif code == 4:
                pl = bytearray(128)
                for lane in range(32):
                    eff = d_f * (float(sub_u8[lane >> 3]) / 255.0)
                    for j in range(4):
                        q0 = int(np.clip(np.round(blk[lane * 8 + 2 * j] / eff + 7.5), 0, 15))
                        q1 = int(np.clip(np.round(blk[lane * 8 + 2 * j + 1] / eff + 7.5), 0, 15))
                        pl[lane * 4 + j] = q0 | (q1 << 4)
                out.extend(hdr + bytes(pl))
            elif code == 2:
                pl = bytearray(64)
                for lane in range(32):
                    eff = d_f * (float(sub_u8[lane >> 3]) / 255.0)
                    for j in range(2):
                        b = 0
                        for m in range(4):
                            q = int(np.clip(np.round(blk[lane * 8 + 4 * j + m] / eff + 1.5), 0, 3))
                            b |= (q << (2 * m))
                        pl[lane * 2 + j] = b
                out.extend(hdr + bytes(pl))
            elif code == 3:
                pl = bytearray(96)
                for lane in range(32):
                    eff = d_f * (float(sub_u8[lane >> 3]) / 255.0)
                    hb = 0
                    for j in range(2):
                        lo_b = 0
                        for m in range(4):
                            q = int(np.clip(np.round(blk[lane * 8 + 4 * j + m] / eff + 3.5), 0, 7))
                            lo_b |= ((q & 3) << (2 * m))
                            hb |= (((q >> 2) & 1) << (4 * j + m))
                        pl[lane * 2 + j] = lo_b
                    pl[64 + lane] = hb
                out.extend(hdr + bytes(pl))
            elif code == 6:
                pl = bytearray(192)
                for lane in range(32):
                    eff = d_f * (float(sub_u8[lane >> 3]) / 255.0)
                    for j in range(2):
                        lo0 = int(np.clip(np.round(blk[lane * 8 + 4 * j + 0] / eff + 31.5), 0, 63))
                        lo1 = int(np.clip(np.round(blk[lane * 8 + 4 * j + 1] / eff + 31.5), 0, 63))
                        lo2 = int(np.clip(np.round(blk[lane * 8 + 4 * j + 2] / eff + 31.5), 0, 63))
                        lo3 = int(np.clip(np.round(blk[lane * 8 + 4 * j + 3] / eff + 31.5), 0, 63))
                        pl[lane * 4 + 2 * j + 0] = (lo0 & 0xF) | ((lo1 & 0xF) << 4)
                        pl[lane * 4 + 2 * j + 1] = (lo2 & 0xF) | ((lo3 & 0xF) << 4)
                        pl[128 + lane * 2 + j] = ((lo0 >> 4) & 3) | (((lo1 >> 4) & 3) << 2) | (((lo2 >> 4) & 3) << 4) | (((lo3 >> 4) & 3) << 6)
                out.extend(hdr + bytes(pl))
            else:
                raise ValueError(code)
    return bytes(out)

h, hc, vocab = 256, 4, 32
hc_dim = hc * h # 1024
q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head = 256, 256, 48, 16, 64, 64
heads_swa, heads_sp = 2, 4
dense_ff, lat_dim, sh_ff = 256, 256, 256
n_exp, top_k = 4, 2
n_mod, ng, nh_eng, row_dim = 2, 3, 4, 32
n_cols = (ng - 1) * nh_eng # 8
eng_dim = n_cols * row_dim # 256

# 1. Build external EngramShard (Section 4 = 2 x 4096 B pages, `rows_per_page = 4096 // 36 = 113`, total 200 rows)
row_bytes = 4 + row_dim # 36
rows_per_page = 4096 // row_bytes # 113
total_eng_rows = 160
eng_pages = bytearray(2 * 4096)
all_eng_raw = []
for rid in range(total_eng_rows):
    sc = np.float32(rng.uniform(0.01, 0.05))
    qs = rng.integers(-60, 60, size=row_dim, dtype=np.int8)
    rb = sc.tobytes() + qs.tobytes()
    all_eng_raw.append(rb)
    pg, idx_in_pg = rid // rows_per_page, rid % rows_per_page
    off = pg * 4096 + idx_in_pg * row_bytes
    eng_pages[off:off + row_bytes] = rb
eng_sec4 = bytes(eng_pages)
eng_sha = hashlib.sha256(eng_sec4).digest()
empty_secs = [(b"", None, 0) for _ in range(6)]
eng_secs = list(empty_secs)
eng_secs[4] = (eng_sec4, eng_sha, 0)
pack_strata_file(e_path, 2, eng_secs, eng_sha)

# Build Section 2 hot seed with ONLY even `row_id`s so odd `row_id`s miss into `e_path` Section 4!
seed_rids = [rid for rid in range(0, total_eng_rows, 2)]
seed_hdr = bytearray(64)
seed_hdr[0:4] = b"ENGR"
struct.pack_into("<III", seed_hdr, 4, 8, len(seed_rids), row_bytes)
seed_hdr[16:48] = eng_sha
seed_idx = bytearray()
seed_dat = bytearray()
for slot, rid in enumerate(seed_rids):
    seed_idx.extend(struct.pack("<QI", rid, slot))
    seed_dat.extend(all_eng_raw[rid])
while (64 + len(seed_idx)) % 64 != 0:
    seed_idx.append(0)
sec2_bytes = bytes(seed_hdr + seed_idx + seed_dat)

# 2. Trunk tensors (Section 1)
trunk_buf = bytearray()
trunk_list = []

def add_trunk(name, shape, fmt_name, raw):
    while len(trunk_buf) % 64 != 0:
        trunk_buf.append(0)
    off = len(trunk_buf)
    trunk_buf.extend(raw)
    trunk_list.append({
        "name": name,
        "format": fmt_name,
        "shape": list(shape),
        "offset": off,
        "bytes": len(raw),
    })

def add_vec(name, n, lo=0.85, hi=1.15):
    a = rng.uniform(lo, hi, size=(n,)).astype(np.float32)
    add_trunk(name, [n], "F32", a.tobytes())

def add_mat(name, rows, cols, sk_fmt="S8_K", s=0.08):
    a = rng.uniform(-s, s, size=(rows, cols)).astype(np.float32)
    if use_f32 or sk_fmt == "F32":
        add_trunk(name, [rows, cols], "F32", a.tobytes())
    elif sk_fmt == "Q8_ROW":
        add_trunk(name, [rows, cols], "Q8_ROW", pack_q8_row(a))
    else:
        cmap = {"S2_K": 2, "S3_K": 3, "S4_K": 4, "S6_K": 6, "S8_K": 8}
        add_trunk(name, [rows, cols], sk_fmt, pack_sk_matrix(a, cmap[sk_fmt]))

add_mat("model.embed_tokens.weight", vocab, h, "S6_K", 0.12)
add_mat("lm_head.weight", vocab, h, "S8_K", 0.10)
add_vec("model.norm.weight", h)
add_mat("model.hc_head.hc_head_fn", hc, hc_dim, "F32", 0.05)
add_vec("model.hc_head.hc_head_scale", 1, 0.5, 1.0)
add_vec("model.hc_head.hc_head_base", hc, -0.2, 0.2)

for il in range(2):
    lp = f"model.layers.{il}"
    # Engram on both layers (`ple_layers = [0, 1]`)
    ep = f"{lp}.engram_layer"
    for ch in range(n_cols):
        add_vec(f"{ep}.lookup_per_head_norms.{ch}.weight", row_dim)
    for j in range(hc):
        add_mat(f"{ep}.key_projs.{j}.weight", h, eng_dim, "S4_K", 0.06)
        add_vec(f"{ep}.key_projs.{j}.bias", h, -0.02, 0.02)
        add_vec(f"{ep}.norm_key.{j}.weight", h)
        add_vec(f"{ep}.norm_query.{j}.weight", h)
        add_vec(f"{ep}.unified_per_hc_scales.{j}", h, -0.05, 0.05)
    add_mat(f"{ep}.value_proj.weight", h, eng_dim, "Q8_ROW", 0.06)
    add_vec(f"{ep}.value_proj.bias", h, -0.02, 0.02)
    add_vec(f"{ep}.unified_pre_gamma", h, 0.1, 0.3)

    for sub_k in ["hc_attn_layer", "hc_mlp_layer"]:
        add_mat(f"{lp}.{sub_k}.hc_pre.hc_fn", 2 * hc, hc_dim, "F32", 0.05)
        add_vec(f"{lp}.{sub_k}.hc_pre.hc_scale", 2, 0.5, 1.0)
        add_vec(f"{lp}.{sub_k}.hc_pre.hc_base", 2 * hc, -0.2, 0.2)

    add_vec(f"{lp}.input_layernorm.weight", h)
    add_vec(f"{lp}.post_attention_layernorm.weight", h)

    nh = heads_swa if il == 0 else heads_sp
    ap = f"{lp}.self_attn"
    add_mat(f"{ap}.q_a_proj.weight", q_lora, h, "S8_K", 0.08)
    add_vec(f"{ap}.q_a_layernorm.weight", q_lora)
    add_mat(f"{ap}.q_b_proj.weight", nh * qk_head, q_lora, "S6_K", 0.08)
    add_mat(f"{ap}.kv_a_proj_with_mqa.weight", kv_lora + qk_rope, h, "Q8_ROW", 0.08)
    add_vec(f"{ap}.kv_a_layernorm.weight", kv_lora)
    add_mat(f"{ap}.kv_b_proj.weight", nh * (qk_nope + v_head), kv_lora, "S4_K", 0.08)
    add_vec(f"{ap}.learnable_sink_param", nh, -1.0, 0.5)
    add_mat(f"{ap}.linear_gate.weight", nh * v_head, h, "S3_K", 0.08)
    add_mat(f"{ap}.o_proj.weight", h, nh * v_head, "F32" if nh * v_head < 256 else "S8_K", 0.08)
    add_vec(f"{ap}.post_proj_norm.weight", h)

    mp = f"{lp}.mlp"
    if il == 0:
        add_mat(f"{mp}.gate_proj.weight", dense_ff, h, "S4_K", 0.08)
        add_mat(f"{mp}.up_proj.weight", dense_ff, h, "S6_K", 0.08)
        add_mat(f"{mp}.down_proj.weight", h, dense_ff, "S8_K", 0.08)
        add_vec(f"{mp}.post_fc2_norm.weight", h)
    else:
        add_mat(f"{mp}.gate.weight", n_exp, h, "Q8_ROW", 0.15)
        add_vec(f"{mp}.gate.e_score_correction_bias", n_exp, -0.05, 0.05)
        add_mat(f"{mp}.routed_expert_down_proj.weight", lat_dim, h, "S8_K", 0.08)
        add_mat(f"{mp}.routed_expert_up_proj.weight", h, lat_dim, "Q8_ROW", 0.08)
        add_mat(f"{mp}.shared_experts.gate_proj.weight", sh_ff, h, "S4_K", 0.08)
        add_mat(f"{mp}.shared_experts.up_proj.weight", sh_ff, h, "S4_K", 0.08)
        add_mat(f"{mp}.shared_experts.down_proj.weight", h, sh_ff, "S6_K", 0.08)
        add_vec(f"{mp}.combine_output_norm.weight", h)

# 3. Section 3 (expert directory) + Section 4 (expert bodies for Layer 1)
sec3_buf = bytearray()
sec4_buf = bytearray()
# Store experts in shuffled rank order `[2, 0, 3, 1]` to test `rank_of` indirection!
gu_codes = [2, 3, 4, 6]
dn_codes = [4, 6, 8, 2]
for rank, eid in enumerate([2, 0, 3, 1]):
    while len(sec4_buf) % 64 != 0:
        sec4_buf.append(0)
    body_off = len(sec4_buf)
    sq = np.float16(rng.uniform(0.7, 1.4, size=(lat_dim,))).tobytes()
    gu_mat = rng.uniform(-0.08, 0.08, size=(2 * lat_dim, lat_dim)).astype(np.float32)
    dn_mat = rng.uniform(-0.08, 0.08, size=(lat_dim, lat_dim)).astype(np.float32)
    if use_f32:
        gu_code, dn_code = 101, 101
        gu_b = gu_mat.tobytes()
        dn_b = dn_mat.tobytes()
    else:
        gu_code = gu_codes[eid]
        dn_code = dn_codes[eid]
        gu_b = pack_sk_matrix(gu_mat, gu_code)
        dn_b = pack_sk_matrix(dn_mat, dn_code)
    dn_off = len(sq) + len(gu_b)
    body = sq + gu_b + dn_b
    sec4_buf.extend(body)
    sec3_buf.extend(struct.pack("<HBBHHQIIII", eid, gu_code, dn_code, 0, 0, body_off, len(body), dn_off, 0, 0))

tok_obj = {
    "model": {
        "type": "BPE",
        "vocab": {
            "<pad>": 0, "<bos>": 1, "<eos>": 2,
            "h": 3, "e": 4, "l": 5, "o": 6, "Ġ": 7, "w": 8,
            "he": 9, "ll": 10, "llo": 11, "hello": 12, "Ġw": 13,
            **{f"tok{i}": i for i in range(14, vocab)}
        },
        "merges": ["h e", ["l", "l"], "ll o", ["he", "llo"], "Ġ w"],
    },
    "added_tokens": [
        {"id": 0, "content": "<pad>", "special": True},
        {"id": 1, "content": "<bos>", "special": True},
        {"id": 2, "content": "<eos>", "special": True},
    ],
    "pre_tokenizer": {"type": "Sequence", "pretokenizers": []},
}

manifest = {
    "config": {
        "text_config": {
            "num_hidden_layers": 2,
            "hidden_size": h,
            "hc_mult": hc,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "hc_eps": 0.01,
            "hc_magnitude": 2.0,
            "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "qk_head_dim": qk_head,
            "v_head_dim": v_head,
            "sliding_window": 4,
            "compressed_indexer_compress_ratio": 4,
            "index_topk": 64,
            "compressed_indexer_num_attention_heads": 2,
            "compressed_indexer_head_dim": 32,
            "swiglu_limit": 7.0,
            "routed_scaling_factor": 2.5,
            "n_routed_experts": n_exp,
            "num_experts_per_tok": top_k,
            "num_attention_heads": heads_sp,
            "swa_num_attention_heads": heads_swa,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rope_parameters": {
                "deepseek_sparse_attention": {"rope_theta": 1000000.0},
                "sliding_attention": {"rope_theta": 10000.0},
            },
            "layer_types": ["sliding_attention", "deepseek_sparse_attention"],
        }
    },
    "engram_meta": {
        "ple_ngram_size": ng,
        "ple_heads_per_ngram": nh_eng,
        "engram_pad_id": 0,
        "ple_layers": [0, 1],
        "engram_token_map": [i if i < 28 else -1 for i in range(vocab)],
        "ple_layer_multipliers": [1315423911, 2654435761, 2246822519, 3266489917, 668265263, 374761393],
        "ple_head_vocab_sizes": [17] * (n_mod * n_cols),
        "ple_head_offsets": [i * 10 for i in range(n_mod * n_cols)],
    },
    "trunk_tensors": trunk_list,
    "tokenizer_json": json.dumps(tok_obj),
    "chat_template": "",
}

sec0_bytes = json.dumps(manifest).encode("utf-8")
m_secs = list(empty_secs)
m_secs[0] = (sec0_bytes, None, 0)
m_secs[1] = (bytes(trunk_buf), None, 0)
m_secs[2] = (sec2_bytes, None, 0)
m_secs[3] = (bytes(sec3_buf), None, 0)
m_secs[4] = (bytes(sec4_buf), None, 0)
pack_strata_file(m_path, 1, m_secs, eng_sha)
"#;
    let script_path = out_dir.join("gen_hy4.py");
    std::fs::write(&script_path, gen_py).unwrap();

    for mode in ["f32", "sk"] {
        let st = std::process::Command::new(&py)
            .arg(&script_path)
            .arg(&model_strata)
            .arg(&engram_strata)
            .arg(mode)
            .status()
            .unwrap();
        assert!(st.success(), "gen_hy4.py ({mode}) failed");

        assert!(tm_models::is_strata_file(&model_strata));
        let (cpu_m, cpu_eng) = tm_models::Hy4Model::open(&model_strata, None).unwrap();
        if mode == "f32" {
            let tok = tm_tokenizer::Tokenizer::from_hf_tokenizer_json(
                &cpu_m.man.tokenizer_json,
                Some(cpu_m.man.cfg.bos_id),
                Some(cpu_m.man.cfg.eos_id),
            )
            .unwrap();
            let enc = tok.encode("hello w", false);
            assert_eq!(enc, vec![12, 13]);
            assert_eq!(tok.decode(&enc), "hello w");
            assert!(tok.is_stop_token(2));
        }

        let mut cpu_st = cpu_m.new_state(cpu_eng, 64);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();

        // Sibling `Hy4-Synth-Engram-Q8.strata` is auto-discovered when `engram_path` is `None`!
        let (mut cuda_r, _) = CudaRunner::from_strata(&model_strata, None, 64).unwrap();

        // Include token `29` (`engram_token_map[29] == -1`, dead token) + 5 tokens so Layer 0 SWA (`window=4`) wraps!
        let pp = [3u32, 9, 29, 12, 5];
        cpu_m
            .forward_tokens(&mut cpu_st, &pool, kt, &pp, 0, true)
            .unwrap();
        let cuda_pp = cuda_r.forward_batch(&pp, 0, true).unwrap().to_vec();

        let cos_sim = |a: &[f32], b: &[f32]| -> f32 {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / (na.sqrt() * nb.sqrt()).max(1e-12)
        };
        let mut max_diff_pp = 0.0f32;
        for (&c, &g) in cpu_st.logits().iter().zip(cuda_pp.iter()) {
            max_diff_pp = max_diff_pp.max((c - g).abs());
        }
        if mode == "f32" {
            assert!(
                max_diff_pp < 2e-4,
                "hy4 F32 prefill max_diff={max_diff_pp} (cpu={:?}, cuda={:?})",
                &cpu_st.logits()[..6],
                &cuda_pp[..6]
            );
            assert_eq!(argmax(cpu_st.logits()), argmax(&cuda_pp));
        } else {
            let cs = cos_sim(cpu_st.logits(), &cuda_pp);
            assert!(
                cs > 0.998,
                "hy4 S_K+Q8_ROW prefill cos_sim={cs} max_diff={max_diff_pp}"
            );
        }

        // Decode 3 tokens (captures & replays `CudaGraphExec` across SWA wrap + LatentMoE + Engram!)
        for (step, &tok_id) in [11u32, 6, 14].iter().enumerate() {
            let pos = pp.len() + step;
            cpu_m
                .forward_tokens(&mut cpu_st, &pool, kt, &[tok_id], pos, true)
                .unwrap();
            let cuda_tg = cuda_r.forward_token(tok_id, pos, true).unwrap().to_vec();
            let mut max_diff_tg = 0.0f32;
            for (&c, &g) in cpu_st.logits().iter().zip(cuda_tg.iter()) {
                max_diff_tg = max_diff_tg.max((c - g).abs());
            }
            if mode == "f32" {
                assert!(
                    max_diff_tg < 2e-4,
                    "hy4 F32 decode pos={pos} max_diff={max_diff_tg} (cpu={:?}, cuda={:?})",
                    &cpu_st.logits()[..6],
                    &cuda_tg[..6]
                );
                assert_eq!(argmax(cpu_st.logits()), argmax(&cuda_tg));
            } else {
                let cs = cos_sim(cpu_st.logits(), &cuda_tg);
                assert!(
                    cs > 0.998,
                    "hy4 S_K+Q8_ROW decode pos={pos} cos_sim={cs} max_diff={max_diff_tg}"
                );
            }
        }

        // Also verify `n_tok = 24 >= 16` Tensor Core `.strata` prefill (`wmma_hy4_sk_gemm_f32` +
        // `wmma_hy4_latmoe_grouped_f32` + multi-token absorbed MLA) against CPU reference!
        cpu_st.reset();
        cuda_r.reset_state().unwrap();
        let pp24: Vec<u32> = (0..24).map(|i| ((i * 5 + 3) % 31) as u32).collect();
        cpu_m
            .forward_tokens(&mut cpu_st, &pool, kt, &pp24, 0, true)
            .unwrap();
        let cuda_pp24 = cuda_r.forward_batch(&pp24, 0, true).unwrap().to_vec();
        if mode == "f32" {
            let max_diff24 = cpu_st
                .logits()
                .iter()
                .zip(cuda_pp24.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff24 < 5e-3,
                "hy4 F32 WMMA pp24 max_diff={max_diff24} (cpu={:?}, cuda={:?})",
                &cpu_st.logits()[..6],
                &cuda_pp24[..6]
            );
            assert_eq!(argmax(cpu_st.logits()), argmax(&cuda_pp24));
        } else {
            let cs24 = cos_sim(cpu_st.logits(), &cuda_pp24);
            assert!(cs24 > 0.997, "hy4 S_K WMMA pp24 cos_sim={cs24}");
        }
    }

    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn parity_cuda_p0_p1_p2_mtp_dfly2_moe_wmma_and_ddr_offload() {
    if try_ctx().is_none() {
        return;
    }
    let out_dir = std::env::temp_dir().join(format!("tm_cuda_opt_suite_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    let py = find_python3_with_numpy();

    // Part A: `Qwen35` with built-in `blk.{L}.nextn.*` MTP (`P0-1`), `n_tok = 20`
    //         MoE grouped WMMA prefill (`P0-2`), fused dual gate+up (`P0-3`), AND
    //         `TM_CUDA_MOE_HOST_OFFLOAD=1` zero-copy DDR MoE (`P2`)!
    let q35_mtp_path = out_dir.join("qwen35moe_nextn_mtp.gguf");
    let gen_q35_mtp = r#"
import struct, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
rng = np.random.default_rng(20261127)
def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

n_layers = 2
d, vocab = 128, 32
n_head, n_head_kv, head_dim, rope_dim = 4, 2, 32, 16
hk, hv, s_dim, d_inner, d_conv = 2, 4, 32, 128, 4
conv_dim = 2 * hk * s_dim + d_inner
n_exp, top_k, n_ff_exp, n_ff_sh = 4, 2, 64, 128

kvs = [
    ("general.architecture", 8, "qwen35moe"),
    ("qwen35moe.block_count", 4, n_layers),
    ("qwen35moe.nextn_predict_layers", 4, 1),
    ("qwen35moe.embedding_length", 4, d),
    ("qwen35moe.expert_count", 4, n_exp),
    ("qwen35moe.expert_used_count", 4, top_k),
    ("qwen35moe.expert_feed_forward_length", 4, n_ff_exp),
    ("qwen35moe.expert_shared_feed_forward_length", 4, n_ff_sh),
    ("qwen35moe.attention.head_count", 4, n_head),
    ("qwen35moe.attention.head_count_kv", 4, n_head_kv),
    ("qwen35moe.attention.key_length", 4, head_dim),
    ("qwen35moe.rope.dimension_count", 4, rope_dim),
    ("qwen35moe.rope.freq_base", 6, 1000000.0),
    ("qwen35moe.attention.layer_norm_rms_epsilon", 6, 1e-6),
    ("qwen35moe.full_attention_interval", 4, 2),
    ("qwen35moe.ssm.conv_kernel", 4, d_conv),
    ("qwen35moe.ssm.state_size", 4, s_dim),
    ("qwen35moe.ssm.group_count", 4, hk),
    ("qwen35moe.ssm.time_step_rank", 4, hv),
    ("qwen35moe.ssm.inner_size", 4, d_inner),
    ("qwen35moe.vocab_size", 4, vocab),
]

def pack_q8_0(a):
    flat = a.reshape(-1, 32)
    mx = np.maximum(np.max(np.abs(flat), axis=1), 1e-8)
    sc = np.float16(mx / 127.0)
    sf = sc.astype(np.float32)[:, None]
    q = np.clip(np.round(flat / sf), -127, 127).astype(np.int8)
    buf = bytearray()
    for i in range(flat.shape[0]):
        buf.extend(sc[i].tobytes() + q[i].tobytes())
    return bytes(buf)

tensors = []
def add_1d(name, n, lo=0.85, hi=1.15):
    a = rng.uniform(lo, hi, n).astype(np.float32)
    tensors.append((name, (n,), 0, a.tobytes()))
def add_f32_2d(name, rows, cols, s=0.12):
    a = rng.uniform(-s, s, (rows, cols)).astype(np.float32)
    tensors.append((name, (cols, rows), 0, a.tobytes()))
def add_q8_2d(name, rows, cols, s=0.12):
    a = rng.uniform(-s, s, (rows, cols)).astype(np.float32)
    tensors.append((name, (cols, rows), 8, pack_q8_0(a)))
def add_q8_3d(name, ne, rows, cols, s=0.12):
    a = rng.uniform(-s, s, (ne, rows, cols)).astype(np.float32)
    tensors.append((name, (cols, rows, ne), 8, pack_q8_0(a)))

add_q8_2d("token_embd.weight", vocab, d, 0.2)
add_1d("output_norm.weight", d)
add_q8_2d("output.weight", vocab, d, 0.2)

for il in [0, 1, 2]:
    p = f"blk.{il}"
    add_1d(f"{p}.attn_norm.weight", d)
    add_1d(f"{p}.post_attention_norm.weight", d)
    if il == 0:
        add_q8_2d(f"{p}.attn_qkv.weight", conv_dim, d)
        add_q8_2d(f"{p}.attn_gate.weight", d_inner, d)
        add_f32_2d(f"{p}.ssm_alpha.weight", hv, d)
        add_f32_2d(f"{p}.ssm_beta.weight", hv, d)
        cw = rng.uniform(-0.2, 0.2, (conv_dim, d_conv)).astype(np.float32)
        tensors.append((f"{p}.ssm_conv1d.weight", (d_conv, conv_dim), 0, cw.tobytes()))
        add_1d(f"{p}.ssm_a", hv, -1.2, -0.5)
        add_1d(f"{p}.ssm_dt.bias", hv, -0.1, 0.1)
        add_1d(f"{p}.ssm_norm.weight", s_dim)
        add_q8_2d(f"{p}.ssm_out.weight", d, d_inner)
    else:
        add_1d(f"{p}.attn_q_norm.weight", head_dim)
        add_1d(f"{p}.attn_k_norm.weight", head_dim)
        add_q8_2d(f"{p}.attn_q.weight", 2 * n_head * head_dim, d)
        add_q8_2d(f"{p}.attn_k.weight", n_head_kv * head_dim, d)
        add_q8_2d(f"{p}.attn_v.weight", n_head_kv * head_dim, d)
        add_q8_2d(f"{p}.attn_output.weight", d, n_head * head_dim)
    add_f32_2d(f"{p}.ffn_gate_inp.weight", n_exp, d, 0.4)
    eb = np.linspace(-1.2, 1.2, n_exp).astype(np.float32)
    tensors.append((f"{p}.exp_probs_b.bias", (n_exp,), 0, eb.tobytes()))
    add_q8_3d(f"{p}.ffn_gate_exps.weight", n_exp, n_ff_exp, d)
    add_q8_3d(f"{p}.ffn_up_exps.weight", n_exp, n_ff_exp, d)
    add_q8_3d(f"{p}.ffn_down_exps.weight", n_exp, d, n_ff_exp)
    add_q8_2d(f"{p}.ffn_gate_shexp.weight", n_ff_sh, d)
    add_q8_2d(f"{p}.ffn_up_shexp.weight", n_ff_sh, d)
    add_q8_2d(f"{p}.ffn_down_shexp.weight", d, n_ff_sh)
    add_f32_2d(f"{p}.ffn_gate_inp_shexp.weight", 1, d)

add_1d("blk.2.nextn.enorm.weight", d)
add_1d("blk.2.nextn.hnorm.weight", d)
add_q8_2d("blk.2.nextn.eh_proj.weight", d, 2 * d)
add_1d("blk.2.nextn.shared_head_norm.weight", d)

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
    for dm in dims: hdr.extend(struct.pack("<Q", dm))
    hdr.extend(struct.pack("<IQ", dt, off))
while len(hdr) % 32 != 0: hdr.append(0)
base = len(hdr)
for (_, _, _, raw), off in zip(tensors, offs):
    if len(hdr) < base + off: hdr.extend(b"\x00" * (base + off - len(hdr)))
    hdr.extend(raw)
out.write_bytes(bytes(hdr))
"#;
    assert!(
        Command::new(&py)
            .args(["-c", gen_q35_mtp, q35_mtp_path.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );

    let gguf_q35 = GgufFile::open(&q35_mtp_path).unwrap();
    let cpu_q35 = Qwen35Model::from_gguf(&gguf_q35).unwrap();
    assert!(cpu_q35.mtp.is_some());
    let mut cpu_st = Qwen35State::new(&cpu_q35, 64);
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    // SAFETY: `--test-threads=1` serializes env var changes across tests.
    unsafe { std::env::set_var("TM_CUDA_MOE_HOST_OFFLOAD", "1") };
    let mut cuda_q35 =
        tm_cuda::Qwen35CudaRunner::from_gguf_with_kv_type(&gguf_q35, 64, KvType::F16).unwrap();
    unsafe { std::env::remove_var("TM_CUDA_MOE_HOST_OFFLOAD") };
    assert!(cuda_q35.has_builtin_mtp());
    assert_eq!(cuda_q35.host_offloaded_moe_layers, 1);

    let pp20: Vec<u32> = (0..20).map(|i| ((i * 3 + 1) % 31) as u32).collect();
    cpu_q35
        .forward_batch(&mut cpu_st, &pool, kt, &pp20, 0, true)
        .unwrap();
    let cuda_pp20 = cuda_q35.forward_batch(&pp20, 0, true).unwrap().to_vec();
    assert_eq!(argmax(cpu_st.logits()), argmax(&cuda_pp20));

    let first_tok = argmax(&cuda_pp20);
    let mut ref_seq = Vec::with_capacity(16);
    let mut cur = first_tok;
    for s in 0..16 {
        cur = cuda_q35.forward_token_greedy(cur, pp20.len() + s).unwrap();
        ref_seq.push(cur);
    }

    cuda_q35.reset_state().unwrap();
    let _ = cuda_q35.forward_batch(&pp20, 0, true).unwrap();
    let mut spec_seq = Vec::with_capacity(16);
    let mut committed_buf = Vec::new();
    let mut pos = pp20.len();
    let mut cur_s = first_tok;
    while spec_seq.len() < 16 {
        let rem = 16 - spec_seq.len();
        let dk = 3usize.min(rem.saturating_sub(1));
        let (_d_cnt, _a_cnt) = cuda_q35
            .spec_step_mtp_greedy(cur_s, pos, dk, &mut committed_buf)
            .unwrap();
        for &ct in &committed_buf {
            cur_s = ct;
            spec_seq.push(ct);
            pos += 1;
            if spec_seq.len() >= 16 {
                break;
            }
        }
    }
    assert_eq!(
        ref_seq, spec_seq,
        "Qwen35 built-in nextn MTP speculative sequence diverged from greedy baseline!"
    );

    // Part B: `Hy4-Flash` `.strata` with built-in `DFly2` (`P0-1`) + Matrix-Absorbed MLA (`P1-1`)!
    let hy4_dfly_strata = out_dir.join("Hy4-DFly2-Synth.strata");
    let gen_hy4_dfly = r#"
import hashlib, json, struct, sys
from pathlib import Path
import numpy as np

m_path = Path(sys.argv[1])
rng = np.random.default_rng(9090)

def pack_strata_file(path, profile_id, secs):
    frame = bytearray(4096)
    frame[0:8] = b"STRATA\x00\x01"
    struct.pack_into("<IHH", frame, 8, 0, profile_id, 6)
    cur = 4096
    payloads = []
    for idx in range(6):
        raw, flg = secs[idx]
        cur = (cur + 4095) // 4096 * 4096
        off = cur if len(raw) > 0 else 0
        sha = hashlib.sha256(raw).digest() if len(raw) > 0 else b"\x00" * 32
        struct.pack_into("<QQ32sIIQ", frame, 16 + idx * 64, off, len(raw), sha, idx, flg, 0)
        payloads.append((off, raw))
        cur += len(raw)
    out = bytearray(frame)
    for off, raw in payloads:
        if len(raw) == 0: continue
        if len(out) < off: out.extend(b"\x00" * (off - len(out)))
        out.extend(raw)
    path.write_bytes(bytes(out))

h, hc, vocab = 256, 4, 32
hc_dim = hc * h
q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head = 256, 256, 48, 16, 64, 64
heads_swa, heads_sp = 2, 4
dense_ff, lat_dim, sh_ff = 256, 256, 256
n_exp, top_k = 4, 2

trunk_buf = bytearray()
trunk_list = []
def add_trunk(name, shape, fmt_name, raw):
    while len(trunk_buf) % 64 != 0: trunk_buf.append(0)
    off = len(trunk_buf)
    trunk_buf.extend(raw)
    trunk_list.append({"name": name, "format": fmt_name, "shape": list(shape), "offset": off, "bytes": len(raw)})
def add_vec(name, n, lo=0.85, hi=1.15):
    a = rng.uniform(lo, hi, size=(n,)).astype(np.float32)
    add_trunk(name, [n], "F32", a.tobytes())
def add_mat(name, rows, cols, s=0.08):
    a = rng.uniform(-s, s, size=(rows, cols)).astype(np.float32)
    add_trunk(name, [rows, cols], "F32", a.tobytes())

add_mat("model.embed_tokens.weight", vocab, h, 0.12)
add_mat("lm_head.weight", vocab, h, 0.10)
add_vec("model.norm.weight", h)
add_mat("model.hc_head.hc_head_fn", hc, hc_dim, 0.05)
add_vec("model.hc_head.hc_head_scale", 1, 0.5, 1.0)
add_vec("model.hc_head.hc_head_base", hc, -0.2, 0.2)

for il in range(2):
    lp = f"model.layers.{il}"
    for sub_k in ["hc_attn_layer", "hc_mlp_layer"]:
        add_mat(f"{lp}.{sub_k}.hc_pre.hc_fn", 2 * hc, hc_dim, 0.05)
        add_vec(f"{lp}.{sub_k}.hc_pre.hc_scale", 2, 0.5, 1.0)
        add_vec(f"{lp}.{sub_k}.hc_pre.hc_base", 2 * hc, -0.2, 0.2)
    add_vec(f"{lp}.input_layernorm.weight", h)
    add_vec(f"{lp}.post_attention_layernorm.weight", h)
    nh = heads_swa if il == 0 else heads_sp
    ap = f"{lp}.self_attn"
    add_mat(f"{ap}.q_a_proj.weight", q_lora, h)
    add_vec(f"{ap}.q_a_layernorm.weight", q_lora)
    add_mat(f"{ap}.q_b_proj.weight", nh * qk_head, q_lora)
    add_mat(f"{ap}.kv_a_proj_with_mqa.weight", kv_lora + qk_rope, h)
    add_vec(f"{ap}.kv_a_layernorm.weight", kv_lora)
    add_mat(f"{ap}.kv_b_proj.weight", nh * (qk_nope + v_head), kv_lora)
    add_vec(f"{ap}.learnable_sink_param", nh, -1.0, 0.5)
    add_mat(f"{ap}.linear_gate.weight", nh * v_head, h)
    add_mat(f"{ap}.o_proj.weight", h, nh * v_head)
    add_vec(f"{ap}.post_proj_norm.weight", h)
    mp = f"{lp}.mlp"
    if il == 0:
        add_mat(f"{mp}.gate_proj.weight", dense_ff, h)
        add_mat(f"{mp}.up_proj.weight", dense_ff, h)
        add_mat(f"{mp}.down_proj.weight", h, dense_ff)
        add_vec(f"{mp}.post_fc2_norm.weight", h)
    else:
        add_mat(f"{mp}.gate.weight", n_exp, h, 0.15)
        add_vec(f"{mp}.gate.e_score_correction_bias", n_exp, -0.05, 0.05)
        add_mat(f"{mp}.routed_expert_down_proj.weight", lat_dim, h)
        add_mat(f"{mp}.routed_expert_up_proj.weight", h, lat_dim)
        add_mat(f"{mp}.shared_experts.gate_proj.weight", sh_ff, h)
        add_mat(f"{mp}.shared_experts.up_proj.weight", sh_ff, h)
        add_mat(f"{mp}.shared_experts.down_proj.weight", h, sh_ff)
        add_vec(f"{mp}.combine_output_norm.weight", h)

dp = "model.dfly_layers"
add_vec(f"{dp}.mtp_fusion.enorm.weight", h)
add_vec(f"{dp}.mtp_fusion.hnorm.weight", h)
add_mat(f"{dp}.mtp_fusion.eh_proj.weight", h, 2 * h)
add_vec(f"{dp}.mtp_fusion.gate", h, 0.1, 0.5)
for dl in range(2):
    dlp = f"{dp}.{dl}"
    add_vec(f"{dlp}.input_layernorm.weight", h)
    add_vec(f"{dlp}.post_attention_layernorm.weight", h)
    ap = f"{dlp}.self_attn"
    add_mat(f"{ap}.q_a_proj.weight", q_lora, h)
    add_vec(f"{ap}.q_a_layernorm.weight", q_lora)
    add_mat(f"{ap}.q_b_proj.weight", heads_sp * qk_head, q_lora)
    add_mat(f"{ap}.kv_a_proj_with_mqa.weight", kv_lora + qk_rope, h)
    add_vec(f"{ap}.kv_a_layernorm.weight", kv_lora)
    add_mat(f"{ap}.kv_b_proj.weight", heads_sp * (qk_nope + v_head), kv_lora)
    add_vec(f"{ap}.learnable_sink_param", heads_sp, -1.0, 0.5)
    add_mat(f"{ap}.linear_gate.weight", heads_sp * v_head, h)
    add_mat(f"{ap}.o_proj.weight", h, heads_sp * v_head)
    add_vec(f"{ap}.post_proj_norm.weight", h)
    add_mat(f"{dlp}.mlp.gate_proj.weight", dense_ff, h)
    add_mat(f"{dlp}.mlp.up_proj.weight", dense_ff, h)
    add_mat(f"{dlp}.mlp.down_proj.weight", h, dense_ff)
add_vec(f"{dp}.final_norm.weight", h)
add_vec(f"{dp}.hidden_correction.hidden_norm.weight", h)
add_vec(f"{dp}.hidden_correction.embed_norm.weight", h)
add_mat(f"{dp}.hidden_correction.gate_proj.weight", dense_ff, 2 * h)
add_mat(f"{dp}.hidden_correction.up_proj.weight", dense_ff, 2 * h)
add_mat(f"{dp}.hidden_correction.down_proj.weight", h, dense_ff)
add_mat(f"{dp}.confidence_head.proj.weight", 1, h, 0.2)
add_vec(f"{dp}.confidence_head.proj.bias", 1, 1.5, 2.5)

sec3_buf, sec4_buf = bytearray(), bytearray()
for eid in range(n_exp):
    while len(sec4_buf) % 64 != 0: sec4_buf.append(0)
    body_off = len(sec4_buf)
    sq = np.float16(rng.uniform(0.8, 1.2, size=(lat_dim,))).tobytes()
    gu_b = rng.uniform(-0.08, 0.08, size=(2 * lat_dim, lat_dim)).astype(np.float32).tobytes()
    dn_b = rng.uniform(-0.08, 0.08, size=(lat_dim, lat_dim)).astype(np.float32).tobytes()
    dn_off = len(sq) + len(gu_b)
    body = sq + gu_b + dn_b
    sec4_buf.extend(body)
    sec3_buf.extend(struct.pack("<HBBHHQIIII", eid, 101, 101, 0, 0, body_off, len(body), dn_off, 0, 0))

manifest = {
    "config": {
        "text_config": {
            "dfly_mask_id": 30,
            "dfly_layers": [1, 1],
            "num_hidden_layers": 2,
            "hidden_size": h,
            "hc_mult": hc,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "hc_eps": 0.01,
            "hc_magnitude": 2.0,
            "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope,
            "qk_rope_head_dim": qk_rope,
            "qk_head_dim": qk_head,
            "v_head_dim": v_head,
            "sliding_window": 4,
            "compressed_indexer_compress_ratio": 4,
            "index_topk": 64,
            "compressed_indexer_num_attention_heads": 2,
            "compressed_indexer_head_dim": 32,
            "swiglu_limit": 7.0,
            "routed_scaling_factor": 2.5,
            "n_routed_experts": n_exp,
            "num_experts_per_tok": top_k,
            "num_attention_heads": heads_sp,
            "swa_num_attention_heads": heads_swa,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rope_parameters": {
                "deepseek_sparse_attention": {"rope_theta": 1000000.0},
                "sliding_attention": {"rope_theta": 10000.0},
            },
            "layer_types": ["sliding_attention", "deepseek_sparse_attention"],
        },
    },
    "engram_meta": {
        "ple_ngram_size": 3,
        "ple_heads_per_ngram": 4,
        "engram_pad_id": 0,
        "ple_layers": [],
        "engram_token_map": list(range(vocab)),
        "ple_layer_multipliers": [],
        "ple_head_vocab_sizes": [],
        "ple_head_offsets": [],
    },
    "trunk_tensors": trunk_list,
    "tokenizer_json": "{}",
    "chat_template": "",
}
secs = [(b"", 0) for _ in range(6)]
secs[0] = (json.dumps(manifest).encode("utf-8"), 0)
secs[1] = (bytes(trunk_buf), 0)
secs[3] = (bytes(sec3_buf), 0)
secs[4] = (bytes(sec4_buf), 0)
pack_strata_file(m_path, 1, secs)
"#;
    assert!(
        Command::new(&py)
            .args(["-c", gen_hy4_dfly, hy4_dfly_strata.to_str().unwrap()])
            .status()
            .unwrap()
            .success()
    );

    let (mut hy4_cuda, _) =
        tm_cuda::Hy4CudaRunner::from_strata(&hy4_dfly_strata, None, 64).unwrap();
    assert!(hy4_cuda.has_builtin_dfly2());
    let pp_hy = [1u32, 4, 7, 11, 15];
    let lg_hy = hy4_cuda.forward_batch(&pp_hy, 0, true).unwrap().to_vec();
    let hy_first = argmax(&lg_hy);

    let mut hy_ref_seq = Vec::with_capacity(12);
    let mut hy_cur = hy_first;
    for s in 0..12 {
        hy_cur = hy4_cuda
            .forward_token_greedy(hy_cur, pp_hy.len() + s)
            .unwrap();
        hy_ref_seq.push(hy_cur);
    }

    hy4_cuda.reset_state().unwrap();
    let _ = hy4_cuda.forward_batch(&pp_hy, 0, true).unwrap();
    let mut hy_spec_seq = Vec::with_capacity(12);
    let mut hy_committed = Vec::new();
    let mut hy_pos = pp_hy.len();
    let mut hy_tok = hy_first;
    while hy_spec_seq.len() < 12 {
        let rem = 12 - hy_spec_seq.len();
        let dk = 3usize.min(rem.saturating_sub(1));
        let (_d_cnt, _a_cnt) = hy4_cuda
            .spec_step_dfly2_greedy(hy_tok, hy_pos, dk, 0.25, &mut hy_committed)
            .unwrap();
        for &ct in &hy_committed {
            hy_tok = ct;
            hy_spec_seq.push(ct);
            hy_pos += 1;
            if hy_spec_seq.len() >= 12 {
                break;
            }
        }
    }
    assert_eq!(
        hy_ref_seq, hy_spec_seq,
        "Hy4 built-in DFly2 speculative sequence diverged from greedy baseline!"
    );

    // Part C: `n_tok = 68 >= 64` `wmma_gemm_mixed_64x64_f32` + `launch_gdn_multi_step` +
    //         captured `verify_graphs` (`P0-A/B`) + `TM_CUDA_MOE_CACHE_EXPERTS=2` (`P2`) 0-ULP!
    {
        // SAFETY: `--test-threads=1` serializes env var changes across tests.
        unsafe {
            std::env::set_var("TM_CUDA_MOE_HOST_OFFLOAD", "1");
            std::env::set_var("TM_CUDA_MOE_CACHE_EXPERTS", "0");
        }
        let mut q35_nocache =
            tm_cuda::Qwen35CudaRunner::from_gguf_with_kv_type(&gguf_q35, 128, KvType::F16).unwrap();
        unsafe {
            std::env::set_var("TM_CUDA_MOE_CACHE_EXPERTS", "2");
        }
        let mut q35_cached =
            tm_cuda::Qwen35CudaRunner::from_gguf_with_kv_type(&gguf_q35, 128, KvType::F16).unwrap();
        unsafe {
            std::env::remove_var("TM_CUDA_MOE_HOST_OFFLOAD");
            std::env::remove_var("TM_CUDA_MOE_CACHE_EXPERTS");
        }

        let pp68: Vec<u32> = (0..68).map(|i| ((i * 7 + 2) % 31) as u32).collect();
        let mut cpu_st68 = Qwen35State::new(&cpu_q35, 128);
        cpu_q35
            .forward_batch(&mut cpu_st68, &pool, kt, &pp68, 0, true)
            .unwrap();
        let lg_nocache = q35_nocache.forward_batch(&pp68, 0, true).unwrap().to_vec();
        let lg_cached = q35_cached.forward_batch(&pp68, 0, true).unwrap().to_vec();
        assert_eq!(argmax(cpu_st68.logits()), argmax(&lg_nocache));
        for (i, (&a, &b)) in lg_nocache.iter().zip(lg_cached.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "P2 VRAM expert cache prefill diverged at logit[{i}]: {a} vs {b}"
            );
        }

        // Run 8 decode steps so `moe_cache_tick_promote` promotes hot experts into VRAM slots
        // and verify 0-ULP bit-exactness at every step!
        let mut t_cur = argmax(&lg_nocache);
        for s in 0..8 {
            let l0 = q35_nocache
                .forward_token(t_cur, pp68.len() + s, true)
                .unwrap()
                .to_vec();
            let l1 = q35_cached
                .forward_token(t_cur, pp68.len() + s, true)
                .unwrap()
                .to_vec();
            for (i, (&a, &b)) in l0.iter().zip(l1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "P2 VRAM expert cache decode step {s} diverged at logit[{i}]: {a} vs {b}"
                );
            }
            t_cur = argmax(&l0);
        }

        // Verify `verify_graphs` capture (call 1) + replay (call 2) on `Qwen35CudaRunner`!
        let mut ref_vw = Vec::new();
        let mut c_vw = t_cur;
        for s in 0..6 {
            c_vw = q35_cached
                .forward_token_greedy(c_vw, pp68.len() + 8 + s)
                .unwrap();
            ref_vw.push(c_vw);
        }
        q35_cached.reset_state().unwrap();
        let _ = q35_cached.forward_batch(&pp68, 0, true).unwrap();
        let mut c2 = argmax(&lg_cached);
        for s in 0..8 {
            c2 = q35_cached.forward_token_greedy(c2, pp68.len() + s).unwrap();
        }
        assert_eq!(c2, t_cur);
        let mut vw_comm = Vec::new();
        let mut vw_seq = Vec::new();
        let mut vw_pos = pp68.len() + 8;
        let mut vw_tok = t_cur;
        // Call 1 captures `verify_graphs[1]` (`d_len = 2`), Call 2 replays `verify_graphs[1]`!
        let (_, acc1) = q35_cached
            .verify_window_greedy(
                vw_tok,
                &[ref_vw[0], (ref_vw[1] + 5) % 30],
                vw_pos,
                &mut vw_comm,
            )
            .unwrap();
        assert_eq!(acc1, 1);
        for &ct in &vw_comm {
            vw_tok = ct;
            vw_seq.push(ct);
            vw_pos += 1;
        }
        let (_, acc2) = q35_cached
            .verify_window_greedy(vw_tok, &[ref_vw[2], ref_vw[3]], vw_pos, &mut vw_comm)
            .unwrap();
        assert_eq!(acc2, 2);
        for &ct in &vw_comm {
            vw_tok = ct;
            vw_seq.push(ct);
            vw_pos += 1;
        }
        while vw_seq.len() < 6 {
            vw_tok = q35_cached.forward_token_greedy(vw_tok, vw_pos).unwrap();
            vw_seq.push(vw_tok);
            vw_pos += 1;
        }
        assert_eq!(
            ref_vw, vw_seq,
            "Qwen35 verify_graphs capture/replay diverged!"
        );
    }

    // Part D: `Qwen4Exp` + companion `qwen4exp-mtp` GGUF (`mtp-q2_0.gguf`, `P0-B` + `P1`)!
    let q4_trunk_path = out_dir.join("qwen4exp_trunk.gguf");
    let q4_mtp_path = out_dir.join("mtp-q2_0.gguf");
    let gen_q4_mtp = r#"
import struct, sys
from pathlib import Path
import numpy as np

trunk_out, mtp_out = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(20261128)

def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

def write_gguf(path, kvs, tensors):
    hdr = bytearray(b"GGUF")
    hdr.extend(struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for k, vt, v in kvs:
        hdr.extend(gstr(k))
        if vt == 94:
            hdr.extend(struct.pack("<IIQ", 9, 4, len(v)))
            for x in v: hdr.extend(struct.pack("<I", x))
        else:
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
        for dm in dims: hdr.extend(struct.pack("<Q", dm))
        hdr.extend(struct.pack("<IQ", dt, off))
    while len(hdr) % 32 != 0: hdr.append(0)
    base = len(hdr)
    for (_, _, _, raw), off in zip(tensors, offs):
        if len(hdr) < base + off: hdr.extend(b"\x00" * (base + off - len(hdr)))
        hdr.extend(raw)
    path.write_bytes(bytes(hdr))

def pack_q2_0(rows, cols):
    out = bytearray()
    for _ in range(rows * (cols // 64)):
        d = np.float16(rng.uniform(0.02, 0.08)).tobytes()
        qs = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        out.extend(d + qs)
    return bytes(out)

def pack_q3_k(rows, cols):
    out = bytearray()
    for _ in range(rows * (cols // 256)):
        hm = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        qs = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()
        sc = rng.integers(0, 256, size=12, dtype=np.uint8).tobytes()
        d = np.float16(rng.uniform(0.003, 0.01)).tobytes()
        out.extend(hm + qs + sc + d)
    return bytes(out)

n_embd, hc, hc_lr, vocab = 256, 4, 32, 32
hc_dim = hc * n_embd
n_exp, top_k, n_ff_exp, n_ff_sh = 8, 2, 64, 128
n_head, n_head_kv, head_dim, rope_dim = 4, 2, 64, 16
hk, hv, s_dim, d_inner, d_conv = 2, 4, 32, 128, 4
conv_dim = 2 * hk * s_dim + d_inner
idx_nh, idx_d, idx_topk = 2, 32, 8

kvs_t = [
    ("general.architecture", 8, "qwen4exp"),
    ("qwen4exp.block_count", 4, 2),
    ("qwen4exp.embedding_length", 4, n_embd),
    ("qwen4exp.hyper_connection.count", 4, hc),
    ("qwen4exp.hyper_connection.low_rank", 4, hc_lr),
    ("qwen4exp.expert_count", 4, n_exp),
    ("qwen4exp.expert_used_count", 4, top_k),
    ("qwen4exp.expert_feed_forward_length", 4, n_ff_exp),
    ("qwen4exp.expert_shared_feed_forward_length", 4, n_ff_sh),
    ("qwen4exp.attention.head_count", 4, n_head),
    ("qwen4exp.attention.head_count_kv", 4, n_head_kv),
    ("qwen4exp.attention.key_length", 4, head_dim),
    ("qwen4exp.rope.dimension_count", 4, rope_dim),
    ("qwen4exp.rope.freq_base", 6, 10000000.0),
    ("qwen4exp.attention.layer_norm_rms_epsilon", 6, 1e-6),
    ("qwen4exp.full_attention_interval", 4, 2),
    ("qwen4exp.ssm.conv_kernel", 4, d_conv),
    ("qwen4exp.ssm.state_size", 4, s_dim),
    ("qwen4exp.ssm.group_count", 4, hk),
    ("qwen4exp.ssm.time_step_rank", 4, hv),
    ("qwen4exp.ssm.inner_size", 4, d_inner),
    ("qwen4exp.attention.indexer.head_count", 4, idx_nh),
    ("qwen4exp.attention.indexer.key_length", 4, idx_d),
    ("qwen4exp.attention.indexer.top_k", 4, idx_topk),
    ("qwen4exp.attention.compress_ratios", 94, [0, 4]),
    ("qwen4exp.ple.layers", 94, []),
]

tt = []
def add_1d(tl, name, n, lo=0.85, hi=1.15):
    a = rng.uniform(lo, hi, n).astype(np.float32)
    tl.append((name, (n,), 0, a.tobytes()))
def add_mat(tl, name, rows, cols, s=0.12):
    a = rng.uniform(-s, s, (rows, cols)).astype(np.float32)
    tl.append((name, (cols, rows), 0, a.tobytes()))

add_mat(tt, "token_embd.weight", vocab, n_embd, 0.2)
add_mat(tt, "output.weight", vocab, n_embd, 0.2)
add_1d(tt, "output_hc_norm.weight", hc_dim)
add_mat(tt, "output_hc_down.weight", hc_lr, hc_dim, 0.08)
add_mat(tt, "output_hc_up.weight", hc_dim, hc_lr, 0.08)

for il in range(2):
    p = f"blk.{il}"
    for kind in ["attn", "ffn"]:
        add_1d(tt, f"{p}.hc_{kind}_norm.weight", hc_dim)
        add_mat(tt, f"{p}.hc_{kind}_down.weight", hc_lr, hc_dim, 0.08)
        add_mat(tt, f"{p}.hc_{kind}_up.weight", hc_dim, hc_lr, 0.08)
        add_mat(tt, f"{p}.hc_{kind}_inject.weight", hc, hc_dim, 0.08)
    if il == 0:
        add_mat(tt, f"{p}.attn_qkv.weight", conv_dim, n_embd)
        add_mat(tt, f"{p}.attn_gate.weight", d_inner, n_embd)
        add_mat(tt, f"{p}.ssm_alpha.weight", hv, n_embd)
        add_mat(tt, f"{p}.ssm_beta.weight", hv, n_embd)
        cw = rng.uniform(-0.2, 0.2, (conv_dim, d_conv)).astype(np.float32)
        tt.append((f"{p}.ssm_conv1d.weight", (d_conv, conv_dim), 0, cw.tobytes()))
        add_1d(tt, f"{p}.ssm_a", hv, -1.4, -0.6)
        add_1d(tt, f"{p}.ssm_dt.bias", hv, -0.1, 0.1)
        add_1d(tt, f"{p}.ssm_norm.weight", s_dim)
        add_mat(tt, f"{p}.ssm_out.weight", n_embd, d_inner)
    else:
        add_1d(tt, f"{p}.attn_q_norm.weight", head_dim)
        add_1d(tt, f"{p}.attn_k_norm.weight", head_dim)
        add_mat(tt, f"{p}.attn_q.weight", 2 * n_head * head_dim, n_embd)
        add_mat(tt, f"{p}.attn_k.weight", n_head_kv * head_dim, n_embd)
        add_mat(tt, f"{p}.attn_v.weight", n_head_kv * head_dim, n_embd)
        add_mat(tt, f"{p}.attn_output.weight", n_embd, n_head * head_dim)
        add_mat(tt, f"{p}.indexer.q_proj.weight", idx_nh * idx_d, n_embd)
        add_mat(tt, f"{p}.indexer.k_proj.weight", idx_d, n_embd)
        add_1d(tt, f"{p}.indexer.q_norm.weight", idx_d)
        add_1d(tt, f"{p}.indexer.k_norm.weight", idx_d)
    add_mat(tt, f"{p}.ffn_gate_inp.weight", n_exp, n_embd, 0.4)
    eb = np.linspace(-1.2, 1.2, n_exp).astype(np.float32)
    tt.append((f"{p}.exp_probs_b.bias", (n_exp,), 0, eb.tobytes()))
    tt.append((f"{p}.ffn_gate_exps.weight", (n_embd, n_ff_exp, n_exp), 42, pack_q2_0(n_exp * n_ff_exp, n_embd)))
    tt.append((f"{p}.ffn_up_exps.weight", (n_embd, n_ff_exp, n_exp), 11, pack_q3_k(n_exp * n_ff_exp, n_embd)))
    tt.append((f"{p}.ffn_down_exps.weight", (n_ff_exp, n_embd, n_exp), 42, pack_q2_0(n_exp * n_embd, n_ff_exp)))
    add_mat(tt, f"{p}.ffn_gate_shexp.weight", n_ff_sh, n_embd)
    add_mat(tt, f"{p}.ffn_up_shexp.weight", n_ff_sh, n_embd)
    add_mat(tt, f"{p}.ffn_down_shexp.weight", n_embd, n_ff_sh)
    add_mat(tt, f"{p}.ffn_gate_inp_shexp.weight", 1, n_embd)
write_gguf(trunk_out, kvs_t, tt)

# Build companion `qwen4exp-mtp` GGUF (`31` tensors, `GemmaRMSNorm` stored centered at 0.0!)
tm = []
def add_gnorm(name, n):
    a = rng.uniform(-0.1, 0.1, n).astype(np.float32)
    tm.append((name, (n,), 0, a.tobytes()))
def add_hc_mtp(pfx, with_inj):
    add_gnorm(f"{pfx}.hc_norm.weight", hc_dim)
    add_mat(tm, f"{pfx}.input_mix_weight_down.weight", hc_lr, hc_dim, 0.08)
    add_mat(tm, f"{pfx}.input_mix_weight_up.weight", hc_dim, hc_lr, 0.08)
    if with_inj:
        add_mat(tm, f"{pfx}.block_inject_weight.weight", hc, hc_dim, 0.08)

add_gnorm("mtp.pre_fc_norm_embedding.weight", n_embd)
add_gnorm("mtp.pre_fc_norm_hidden.weight", hc_dim)
add_mat(tm, "mtp.fc_embedding.weight", n_embd, n_embd, 0.1)
add_mat(tm, "mtp.fc_hidden.weight", n_embd, n_embd, 0.1)
add_hc_mtp("mtp.hyper_connection_mixer", False)
add_hc_mtp("mtp.layers.0.attn_hyper_connection", True)
add_hc_mtp("mtp.layers.0.mlp_hyper_connection", True)
add_gnorm("mtp.layers.0.self_attn.q_norm.weight", head_dim)
add_gnorm("mtp.layers.0.self_attn.k_norm.weight", head_dim)
add_mat(tm, "mtp.layers.0.self_attn.q_proj.weight", 2 * n_head * head_dim, n_embd)
add_mat(tm, "mtp.layers.0.self_attn.k_proj.weight", n_head_kv * head_dim, n_embd)
add_mat(tm, "mtp.layers.0.self_attn.v_proj.weight", n_head_kv * head_dim, n_embd)
add_mat(tm, "mtp.layers.0.self_attn.o_proj.weight", n_embd, n_head * head_dim)
add_mat(tm, "mtp.layers.0.self_attn.indexer.index_qk_proj.weight", (idx_nh + 1) * idx_d, n_embd)
add_gnorm("mtp.layers.0.self_attn.indexer.q_layernorm.weight", idx_d)
add_gnorm("mtp.layers.0.self_attn.indexer.k_layernorm.weight", idx_d)
add_mat(tm, "mtp.layers.0.mlp.gate.weight", n_exp, n_embd, 0.4)
tm.append(("mtp.layers.0.mlp.experts.gate_up_proj", (n_embd, 2 * n_ff_exp, n_exp), 42, pack_q2_0(n_exp * 2 * n_ff_exp, n_embd)))
tm.append(("mtp.layers.0.mlp.experts.down_proj", (n_ff_exp, n_embd, n_exp), 42, pack_q2_0(n_exp * n_embd, n_ff_exp)))
add_mat(tm, "mtp.layers.0.mlp.shared_expert.gate_proj.weight", n_ff_sh, n_embd)
add_mat(tm, "mtp.layers.0.mlp.shared_expert.up_proj.weight", n_ff_sh, n_embd)
add_mat(tm, "mtp.layers.0.mlp.shared_expert.down_proj.weight", n_embd, n_ff_sh)
add_mat(tm, "mtp.layers.0.mlp.shared_expert_gate.weight", 1, n_embd)
assert len(tm) == 31, f"expected 31 MTP tensors, got {len(tm)}"
write_gguf(mtp_out, [("general.architecture", 8, "qwen4exp-mtp")], tm)
"#;
    assert!(
        Command::new(&py)
            .args([
                "-c",
                gen_q4_mtp,
                q4_trunk_path.to_str().unwrap(),
                q4_mtp_path.to_str().unwrap()
            ])
            .status()
            .unwrap()
            .success()
    );

    let q4_gguf = GgufFile::open(&q4_trunk_path).unwrap();
    let mut q4_cuda = CudaRunner::from_gguf_with_kv_type(&q4_gguf, 64, KvType::F16).unwrap();
    // Sibling `mtp-q2_0.gguf` in the same directory is auto-discovered!
    assert_eq!(q4_cuda.builtin_spec_name(), Some("mtp"));

    let pp_q4 = [1u32, 3, 5, 7, 9, 11, 13, 15];
    let lg_q4 = q4_cuda.forward_batch(&pp_q4, 0, true).unwrap().to_vec();
    let q4_first = argmax(&lg_q4);

    let mut q4_ref_seq = Vec::with_capacity(14);
    let mut q4_cur = q4_first;
    for s in 0..14 {
        q4_cur = q4_cuda
            .forward_token_greedy(q4_cur, pp_q4.len() + s)
            .unwrap();
        q4_ref_seq.push(q4_cur);
    }

    q4_cuda.reset_state().unwrap();
    let _ = q4_cuda.forward_batch(&pp_q4, 0, true).unwrap();
    let mut q4_mtp_seq = Vec::with_capacity(14);
    let mut q4_comm = Vec::new();
    let mut q4_pos = pp_q4.len();
    let mut q4_tok = q4_first;
    while q4_mtp_seq.len() < 14 {
        let rem = 14 - q4_mtp_seq.len();
        let dk = 3usize.min(rem.saturating_sub(1));
        let (_d, _a) = q4_cuda
            .spec_step_builtin_greedy(q4_tok, q4_pos, dk, &mut q4_comm)
            .unwrap();
        for &ct in &q4_comm {
            q4_tok = ct;
            q4_mtp_seq.push(ct);
            q4_pos += 1;
            if q4_mtp_seq.len() >= 14 {
                break;
            }
        }
    }
    assert_eq!(
        q4_ref_seq, q4_mtp_seq,
        "Qwen4Exp neural MTP (`mtp-q2_0.gguf`) speculative sequence diverged from greedy baseline!"
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}
