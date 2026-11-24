//! End-to-end L1/L3 parity tests for `LlamaVkRunner`:
//! 1. **Exact FP32 model parity (`rel <= 1e-5`)**: synthetic `F32` Llama & Qwen3 (QK-Norm + NeoX RoPE)
//!    models compared against `tm_models::LlamaModel` across multi-token decode and `forward_batch(35)`.
//! 2. **Mixed-quantization decode vs batch invariance (`rel <= 1e-5`) & CPU Q8 KL parity**:
//!    synthetic model mixing `Q8_0` (`token_embd`), `Q6_K` (`output.weight`), `Q4_0` (`QKV`, `Wo`,
//!    fused `Gate+Up+SwiGLU`), `Q5_K` (`blk.0.ffn_down`), and `Q4_1` (`blk.1.ffn_down`) — verifies
//!    that 35-token batched prefill (`gemm_tiled` + `gemm_kquant`) matches 35 sequential pre-recorded
//!    decode steps (`gemv_res` + `gemv_qkv_q4_0` + `gemv_pair_silu_q4_0`) to `rel <= 1e-5`, and
//!    matches CPU Q8-activation `LlamaModel` with `KL < 1e-3` and 100% Top-1 agreement.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tm_core::{BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{LlamaModel, LlamaState, argmax};
use tm_vulkan::{LlamaVkRunner, VkPrefillMode, VulkanContext};

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
        (((self.next_u64() % 2001) as f32) - 1000.0) / 1000.0
    }
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn kv_str(buf: &mut Vec<u8>, k: &str, v: &str) {
    push_str(buf, k);
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_str(buf, v);
}

fn kv_u32(buf: &mut Vec<u8>, k: &str, v: u32) {
    push_str(buf, k);
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn kv_f32(buf: &mut Vec<u8>, k: &str, v: f32) {
    push_str(buf, k);
    buf.extend_from_slice(&6u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

fn make_q4_0_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 32;
    let mut out = Vec::with_capacity(nb * 18);
    for _ in 0..nb {
        let b = BlockQ4_0 {
            d: f32_to_f16(rng.f32() * 0.05),
            qs: core::array::from_fn(|_| rng.u8()),
        };
        out.extend_from_slice(&b.d.to_le_bytes());
        out.extend_from_slice(&b.qs);
    }
    out
}

fn make_q4_1_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 32;
    let mut out = Vec::with_capacity(nb * 20);
    for _ in 0..nb {
        let b = BlockQ4_1 {
            d: f32_to_f16(rng.f32() * 0.04),
            m: f32_to_f16(rng.f32() * 0.02),
            qs: core::array::from_fn(|_| rng.u8()),
        };
        out.extend_from_slice(&b.d.to_le_bytes());
        out.extend_from_slice(&b.m.to_le_bytes());
        out.extend_from_slice(&b.qs);
    }
    out
}

fn make_q8_0_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 32;
    let mut out = Vec::with_capacity(nb * 34);
    for _ in 0..nb {
        let b = BlockQ8_0 {
            d: f32_to_f16(rng.f32() * 0.02),
            qs: core::array::from_fn(|_| rng.u8() as i8),
        };
        out.extend_from_slice(&b.d.to_le_bytes());
        for &q in &b.qs {
            out.push(q as u8);
        }
    }
    out
}

fn make_q5_k_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 256;
    let mut out = Vec::with_capacity(nb * 176);
    for _ in 0..nb {
        let b = BlockQ5K {
            d: f32_to_f16(0.01 + rng.f32().abs() * 0.02),
            dmin: f32_to_f16(0.005 + rng.f32().abs() * 0.01),
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

fn make_q6_k_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 256;
    let mut out = Vec::with_capacity(nb * 210);
    for _ in 0..nb {
        let b = BlockQ6K {
            ql: core::array::from_fn(|_| rng.u8()),
            qh: core::array::from_fn(|_| rng.u8()),
            scales: core::array::from_fn(|_| ((rng.u8() % 15) as i8) - 7),
            d: f32_to_f16(0.0005 + rng.f32().abs() * 0.0005),
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

fn make_f32_bytes(rng: &mut Rng, len: usize, center: f32, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for _ in 0..len {
        let v = center + rng.f32() * scale;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn serialize_gguf(
    arch: &str,
    n_layers: usize,
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    vocab: usize,
    tensors: Vec<TensorEntry>,
) -> Vec<u8> {
    let mut kv = Vec::new();
    kv_str(&mut kv, "general.architecture", arch);
    kv_u32(&mut kv, &format!("{arch}.block_count"), n_layers as u32);
    kv_u32(&mut kv, &format!("{arch}.embedding_length"), n_embd as u32);
    kv_u32(&mut kv, &format!("{arch}.feed_forward_length"), n_ff as u32);
    kv_u32(
        &mut kv,
        &format!("{arch}.attention.head_count"),
        n_head as u32,
    );
    kv_u32(
        &mut kv,
        &format!("{arch}.attention.head_count_kv"),
        n_head_kv as u32,
    );
    kv_u32(
        &mut kv,
        &format!("{arch}.attention.key_length"),
        head_dim as u32,
    );
    kv_u32(&mut kv, &format!("{arch}.vocab_size"), vocab as u32);
    kv_f32(
        &mut kv,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );
    kv_f32(&mut kv, &format!("{arch}.rope.freq_base"), 10_000.0);

    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"GGUF");
    hdr.extend_from_slice(&3u32.to_le_bytes());
    hdr.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&10u64.to_le_bytes());
    hdr.extend_from_slice(&kv);

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur_off = 0u64;
    for t in &tensors {
        offsets.push(cur_off);
        cur_off = (cur_off + t.data.len() as u64).div_ceil(32) * 32;
    }

    for (t, &off) in tensors.iter().zip(&offsets) {
        push_str(&mut hdr, &t.name);
        hdr.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            hdr.extend_from_slice(&d.to_le_bytes());
        }
        hdr.extend_from_slice(&t.ggml_type.to_le_bytes());
        hdr.extend_from_slice(&off.to_le_bytes());
    }
    while !hdr.len().is_multiple_of(32) {
        hdr.push(0);
    }
    let data_start = hdr.len();
    for (t, &off) in tensors.iter().zip(&offsets) {
        let target = data_start + off as usize;
        hdr.resize(target, 0);
        hdr.extend_from_slice(&t.data);
    }
    hdr
}

fn build_f32_gguf(arch: &str) -> Vec<u8> {
    let mut rng = Rng(match arch {
        "qwen3" => 0x1122_3344_5566_7788,
        "qwen2" => 0x55AA_33CC_77EE_9911,
        _ => 0x8877_6655_4433_2211,
    });
    let (n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, vocab) = (2, 128, 256, 4, 2, 32, 40);
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    let mut tensors = vec![
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, vocab * n_embd, 0.0, 0.2),
        },
        TensorEntry {
            name: "output_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.2),
        },
    ];
    if arch == "qwen2" {
        tensors.push(TensorEntry {
            name: "output.bias".into(),
            dims: vec![vocab as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, vocab, 0.0, 0.3),
        });
    }
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        tensors.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.2),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.2),
        });
        if arch == "qwen3" {
            tensors.push(TensorEntry {
                name: format!("{p}.attn_q_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.2),
            });
            tensors.push(TensorEntry {
                name: format!("{p}.attn_k_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.2),
            });
        }
        if arch == "qwen2" {
            for (bsuf, blen) in [
                ("attn_q.bias", q_dim),
                ("attn_k.bias", kv_dim),
                ("attn_v.bias", kv_dim),
                ("attn_output.bias", n_embd),
            ] {
                tensors.push(TensorEntry {
                    name: format!("{p}.{bsuf}"),
                    dims: vec![blen as u64],
                    ggml_type: 0,
                    data: make_f32_bytes(&mut rng, blen, 0.0, 0.45),
                });
            }
        }
        for (suf, cols, rows) in [
            ("attn_q.weight", n_embd, q_dim),
            ("attn_k.weight", n_embd, kv_dim),
            ("attn_v.weight", n_embd, kv_dim),
            ("attn_output.weight", q_dim, n_embd),
            ("ffn_gate.weight", n_embd, n_ff),
            ("ffn_up.weight", n_embd, n_ff),
            ("ffn_down.weight", n_ff, n_embd),
        ] {
            tensors.push(TensorEntry {
                name: format!("{p}.{suf}"),
                dims: vec![cols as u64, rows as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, rows * cols, 0.0, 0.12),
            });
        }
    }
    serialize_gguf(
        arch, n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, vocab, tensors,
    )
}

fn build_mixed_quant_gguf(arch: &str) -> Vec<u8> {
    let mut rng = Rng(match arch {
        "qwen3" => 0xCAFE_BABE_9060_1200,
        "qwen2" => 0xBADC_0FFE_2026_0500,
        _ => 0xDEAD_BEEF_1234_5678,
    });
    let (n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, vocab) = (2, 256, 512, 4, 2, 64, 48);
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    let mut tensors = vec![
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 8, // Q8_0
            data: make_q8_0_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 14, // Q6_K
            data: make_q6_k_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.25),
        },
    ];

    for l in 0..n_layers {
        let p = format!("blk.{l}");
        tensors.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.25),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.25),
        });
        if arch == "qwen3" {
            tensors.push(TensorEntry {
                name: format!("{p}.attn_q_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.25),
            });
            tensors.push(TensorEntry {
                name: format!("{p}.attn_k_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.25),
            });
        }
        if arch == "qwen2" {
            for (bsuf, blen) in [
                ("attn_q.bias", q_dim),
                ("attn_k.bias", kv_dim),
                ("attn_v.bias", kv_dim),
            ] {
                tensors.push(TensorEntry {
                    name: format!("{p}.{bsuf}"),
                    dims: vec![blen as u64],
                    ggml_type: 0,
                    data: make_f32_bytes(&mut rng, blen, 0.0, 0.5),
                });
            }
        }
        for (suf, cols, rows) in [
            ("attn_q.weight", n_embd, q_dim),
            ("attn_k.weight", n_embd, kv_dim),
            ("attn_v.weight", n_embd, kv_dim),
            ("attn_output.weight", q_dim, n_embd),
            ("ffn_gate.weight", n_embd, n_ff),
            ("ffn_up.weight", n_embd, n_ff),
        ] {
            tensors.push(TensorEntry {
                name: format!("{p}.{suf}"),
                dims: vec![cols as u64, rows as u64],
                ggml_type: 2, // Q4_0
                data: make_q4_0_bytes(&mut rng, rows, cols),
            });
        }
        // Layer 0 uses Q5_K, Layer 1 uses Q4_1 -> exercises both K-quant and Q4_1 GEMV + GEMM!
        let (dt, bytes) = if l == 0 {
            (13u32, make_q5_k_bytes(&mut rng, n_embd, n_ff))
        } else {
            (3u32, make_q4_1_bytes(&mut rng, n_embd, n_ff))
        };
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: dt,
            data: bytes,
        });
    }
    serialize_gguf(
        arch, n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, vocab, tensors,
    )
}

fn open_bytes_as_gguf(tag: &str, bytes: &[u8]) -> Arc<GgufFile> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = std::env::temp_dir().join(format!("tm_vk_{tag}_{}_{stamp}.gguf", std::process::id()));
    std::fs::write(&tmp, bytes).expect("write temp gguf");
    let gguf = Arc::new(GgufFile::open(&tmp).expect("open temp gguf"));
    let _ = std::fs::remove_file(&tmp);
    gguf
}

fn kl_div(ref_l: &[f32], cand_l: &[f32]) -> f64 {
    let mr = ref_l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mc = cand_l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sr: f64 = ref_l.iter().map(|&x| ((x as f64) - mr).exp()).sum();
    let sc: f64 = cand_l.iter().map(|&x| ((x as f64) - mc).exp()).sum();
    let lzr = sr.ln();
    let lzc = sc.ln();
    let mut kl = 0.0f64;
    for (&r, &c) in ref_l.iter().zip(cand_l.iter()) {
        let lp = (r as f64) - mr - lzr;
        let lq = (c as f64) - mc - lzc;
        kl += lp.exp() * (lp - lq);
    }
    kl.max(0.0)
}

fn get_ctx_or_skip() -> Option<VulkanContext> {
    match VulkanContext::new() {
        Ok(c) => Some(c),
        Err(tm_vulkan::VulkanError::DeviceNotFound(sel)) => {
            panic!("TM_VK_DEVICE='{sel}' not found");
        }
        Err(e) => {
            eprintln!("[tm-vulkan] skipping runner parity ({e})");
            None
        }
    }
}

#[test]
fn vk_f32_exact_cpu_parity_llama_and_qwen3() {
    for arch in ["llama", "qwen2", "qwen3"] {
        let Some(ctx) = get_ctx_or_skip() else { return };
        let gguf = open_bytes_as_gguf(&format!("f32_{arch}"), &build_f32_gguf(arch));
        let cpu_model = LlamaModel::from_gguf(&gguf).unwrap();
        let mut cpu_state = LlamaState::new(&cpu_model.cfg, 64);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();

        let mut vk_runner = LlamaVkRunner::from_gguf_with_context(ctx, gguf.clone(), 64).unwrap();

        let seq = [2u32, 11, 7, 33, 19, 5, 27];
        for (pos, &tok) in seq.iter().enumerate() {
            cpu_model
                .forward_token(&mut cpu_state, &pool, kt, tok, pos, true)
                .unwrap();
            let cpu_l = cpu_state.logits();
            let vk_l = vk_runner.forward_token(tok, pos, true).unwrap();
            let norm: f32 = cpu_l.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
            for (i, (&g, &e)) in vk_l.iter().zip(cpu_l.iter()).enumerate() {
                let rel = (g - e).abs() / norm;
                assert!(
                    rel <= 2e-4,
                    "F32 {arch} decode step={pos} i={i}: got={g} expected={e} rel={rel}"
                );
            }
            let g_top = argmax(vk_l);
            let c_top = argmax(cpu_l);
            assert!(
                g_top == c_top || (cpu_l[g_top as usize] - cpu_l[c_top as usize]).abs() <= 1e-3,
                "F32 {arch} step={pos} top-1 mismatch: vk={g_top} ({}) cpu={c_top} ({})",
                vk_l[g_top as usize],
                cpu_l[c_top as usize]
            );
        }

        // Batched prefill of 35 tokens (> 32 tile boundary) vs CPU F32
        let mut cpu_b = LlamaState::new(&cpu_model.cfg, 64);
        let batch_ids: Vec<u32> = (0..35).map(|i| ((i * 5 + 1) % 40) as u32).collect();
        cpu_model
            .forward_batch(&mut cpu_b, &pool, kt, &batch_ids, 0, true)
            .unwrap();
        let vk_b = vk_runner.forward_batch(&batch_ids, 0, true).unwrap();
        let norm: f32 = cpu_b.logits().iter().map(|v| v.abs()).sum::<f32>().max(1.0);
        for (i, (&g, &e)) in vk_b.iter().zip(cpu_b.logits().iter()).enumerate() {
            let rel = (g - e).abs() / norm;
            assert!(
                rel <= 2e-4,
                "F32 {arch} batch(35) i={i}: got={g} expected={e} rel={rel}"
            );
        }
    }
}

#[test]
fn vk_mixed_quant_decode_vs_batch_and_cpu_kl_parity() {
    for arch in ["llama", "qwen2", "qwen3"] {
        let Some(ctx1) = get_ctx_or_skip() else {
            return;
        };
        let Some(ctx2) = get_ctx_or_skip() else {
            return;
        };
        ctx2.set_prefill_mode(VkPrefillMode::Fallback);
        let has_cm_f16 = ctx1.info().coopmat_f16_f32acc;
        let has_cm_i8 = ctx1.info().coopmat_i8_i32acc;
        let gguf = open_bytes_as_gguf(&format!("quant_{arch}"), &build_mixed_quant_gguf(arch));

        let cpu_model = LlamaModel::from_gguf(&gguf).unwrap();
        let mut cpu_state = LlamaState::new(&cpu_model.cfg, 64);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();

        let mut vk_dec = LlamaVkRunner::from_gguf_with_context(ctx1, gguf.clone(), 64).unwrap();
        let mut vk_bat = LlamaVkRunner::from_gguf_with_context(ctx2, gguf.clone(), 64).unwrap();

        let batch_ids: Vec<u32> = (0..35).map(|i| ((i * 7 + 3) % 48) as u32).collect();
        let mut kl_sum = 0.0f64;
        for (pos, &tok) in batch_ids.iter().enumerate() {
            cpu_model
                .forward_token(&mut cpu_state, &pool, kt, tok, pos, true)
                .unwrap();
            let vk_l = vk_dec.forward_token(tok, pos, true).unwrap();
            let kl = kl_div(cpu_state.logits(), vk_l);
            assert!(
                kl < 5e-3,
                "quant {arch} step={pos} KL vs CPU Q8 too high: {kl:.3e}"
            );
            kl_sum += kl;
            let g_top = argmax(vk_l);
            let c_top = argmax(cpu_state.logits());
            let cl = cpu_state.logits();
            assert!(
                g_top == c_top || (cl[g_top as usize] - cl[c_top as usize]).abs() <= 5e-2,
                "quant {arch} step={pos} top-1 mismatch: vk={g_top} cpu={c_top}"
            );
        }
        let mean_kl = kl_sum / (batch_ids.len() as f64);
        assert!(
            mean_kl < 1e-3,
            "quant {arch} mean KL vs CPU Q8 too high: {mean_kl:.3e}"
        );

        // Batched 35-token prefill (`gemm_tiled` + `gemm_kquant`) MUST match 35 sequential
        // pre-recorded decode steps (`gemv_res` + `gemv_qkv_q4_0` + `gemv_pair_silu_q4_0`) to 1e-5!
        let dec_final = vk_dec.logits().to_vec();
        let bat_final = vk_bat.forward_batch(&batch_ids, 0, true).unwrap();
        let norm: f32 = dec_final.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
        for (i, (&b, &d)) in bat_final.iter().zip(dec_final.iter()).enumerate() {
            let rel = (b - d).abs() / norm;
            assert!(
                rel <= 1e-5,
                "quant {arch} batch(35) vs decode(35) i={i}: batch={b} dec={d} rel={rel}"
            );
        }

        // Verify GPU-side `forward_token_greedy`
        let next_tok = 19u32;
        let greedy_id = vk_dec.forward_token_greedy(next_tok, 35).unwrap();
        let full_logits = vk_bat.forward_token(next_tok, 35, true).unwrap();
        assert_eq!(greedy_id, argmax(full_logits));

        // When Cooperative Matrix is available (`radv`), verify end-to-end batched prefill for
        // BOTH `CoopMatF16` (including fused `gemm_pair_silu_coopmat_f16` + `fa_coopmat`) and `CoopMatI8`!
        if has_cm_f16 {
            let ctx_cm16 = get_ctx_or_skip().unwrap();
            ctx_cm16.set_prefill_mode(VkPrefillMode::CoopMatF16);
            let mut vk_cm16 =
                LlamaVkRunner::from_gguf_with_context(ctx_cm16, gguf.clone(), 64).unwrap();
            assert_eq!(vk_cm16.active_prefill_backend(), "coopmat-f16");
            let cm16_final = vk_cm16.forward_batch(&batch_ids, 0, true).unwrap();
            let kl16 = kl_div(&dec_final, cm16_final);
            assert!(
                kl16 < 2e-4,
                "quant {arch} CoopMatF16 batch(35) KL vs decode too high: {kl16:.3e}"
            );
            for (i, (&b, &d)) in cm16_final.iter().zip(dec_final.iter()).enumerate() {
                let rel = (b - d).abs() / norm;
                assert!(
                    rel <= 6e-4,
                    "quant {arch} CoopMatF16 batch(35) i={i}: got={b} dec={d} rel={rel}"
                );
            }
        }
        if has_cm_i8 {
            let ctx_cmi8 = get_ctx_or_skip().unwrap();
            ctx_cmi8.set_prefill_mode(VkPrefillMode::CoopMatI8);
            let mut vk_cmi8 =
                LlamaVkRunner::from_gguf_with_context(ctx_cmi8, gguf.clone(), 64).unwrap();
            assert_eq!(vk_cmi8.active_prefill_backend(), "coopmat-i8");
            let cmi8_final = vk_cmi8.forward_batch(&batch_ids, 0, true).unwrap();
            let kli8 = kl_div(&dec_final, cmi8_final);
            assert!(
                kli8 < 1.5e-3,
                "quant {arch} CoopMatI8 batch(35) KL vs decode too high: {kli8:.3e}"
            );
        }
    }
}

#[test]
fn vk_p0_smollm3_nope_phi3_sliced_longrope_and_mistral_swa_guard() {
    let Some(ctx1) = get_ctx_or_skip() else {
        return;
    };
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap();
    let script = repo_root.join("tools/oracle/p0_architectures_oracle.py");
    let py = [
        "/opt/anaconda3/bin/python3",
        "/opt/homebrew/bin/python3",
        "python3",
    ]
    .into_iter()
    .find(|c| {
        std::process::Command::new(c)
            .args(["-c", "import numpy"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
    .unwrap_or("python3");

    let dir = std::env::temp_dir().join(format!("tm_vk_p0_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 1. `smollm3` (NoPE every 4th layer) & `phi3` (fused `attn_qkv` + fused `ffn_up` + LongRoPE)
    for (arch, ctx_cap, c_in) in [("smollm3", 64usize, Some(ctx1)), ("phi3", 64usize, None)] {
        let ctx = c_in.or_else(get_ctx_or_skip).unwrap();
        let p = dir.join(format!("{arch}_q4.gguf"));
        assert!(
            std::process::Command::new(py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&p)
                .arg("--quant")
                .arg("q4_0")
                .arg("--eval-ctx")
                .arg(ctx_cap.to_string())
                .status()
                .unwrap()
                .success()
        );
        let gguf = std::sync::Arc::new(GgufFile::open(&p).unwrap());
        let cpu = LlamaModel::from_gguf(&gguf).unwrap();
        let mut cpu_st = LlamaState::new(&cpu.cfg, ctx_cap);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();
        let mut vk = LlamaVkRunner::from_gguf_with_context(ctx, gguf.clone(), ctx_cap).unwrap();
        let seq = [1u32, 5, 12, 7, 19, 3];
        for (pos, &t) in seq.iter().enumerate() {
            cpu.forward_token(&mut cpu_st, &pool, kt, t, pos, true)
                .unwrap();
            let vk_l = vk.forward_token(t, pos, true).unwrap();
            assert!(
                kl_div(cpu_st.logits(), vk_l) < 5e-3,
                "Vulkan {arch} step {pos} KL too high"
            );
        }
    }

    // 2. `mistral` with `sliding_window = 3 < max_ctx = 64` cleanly returns `InvalidDimensions` (`AGENTS.md` Rule #6)
    let p_mis = dir.join("mistral_q4.gguf");
    assert!(
        std::process::Command::new(py)
            .arg(&script)
            .arg("--arch")
            .arg("mistral")
            .arg("--out-gguf")
            .arg(&p_mis)
            .arg("--quant")
            .arg("q4_0")
            .status()
            .unwrap()
            .success()
    );
    let gguf_mis = std::sync::Arc::new(GgufFile::open(&p_mis).unwrap());
    let ctx_mis = get_ctx_or_skip().unwrap();
    let res = LlamaVkRunner::from_gguf_with_context(ctx_mis, gguf_mis, 64);
    assert!(matches!(
        res,
        Err(tm_vulkan::VulkanError::InvalidDimensions(_))
    ));

    let _ = std::fs::remove_dir_all(&dir);
}
