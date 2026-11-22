//! End-to-end L1/L3 parity tests for `Qwen35VkRunner` (`AGENTS.md` §5 #3, #7 & §9):
//! 1. **Exact FP32 hybrid parity (`rel <= 1e-5`)**: synthetic `qwen35` model (`4` layers: `3` GDN
//!    linear-attention + `1` gated full-attention, with `HK = 2, HV = 4` so `vh % HK` tiled head
//!    grouping is strictly exercised — catching FLA contiguous grouping bugs immediately!)
//!    verified against `tm_models::Qwen35Model` across multi-token decode and `forward_batch(19)`.
//! 2. **Mixed-quantization decode vs batch invariance + CoopMat/Fallback parity + CPU Q8 KL**:
//!    synthetic `qwen35` model matching real Qwen3.5 per-tensor quantization (`Q6_K` tied
//!    `token_embd`, `Q4_0` `wqkv`/`wz`/`wqg`/`wk`/`wv`/`w_gate`/`w_up`, `Q8_0` `w_alpha`/`w_beta`,
//!    `Q5_K` `ssm_out`, `Q4_1` `ffn_down`) across 35 tokens (`Fallback` on all drivers +
//!    `CoopMatF16` / `CoopMatI8` when supported), plus `reset_state()` bit-exact repeatability.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tm_core::{BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{Qwen35Model, Qwen35State, argmax};
use tm_vulkan::{Qwen35VkRunner, VkPrefillMode, VulkanContext, VulkanError};

fn ctx() -> Option<VulkanContext> {
    match VulkanContext::new() {
        Ok(c) => Some(c),
        Err(VulkanError::DeviceNotFound(sel)) => {
            panic!("TM_VK_DEVICE='{sel}' did not match any Vulkan device");
        }
        Err(e) => {
            eprintln!("[tm-vulkan] skipping: no usable Vulkan device ({e})");
            None
        }
    }
}

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
            d: f32_to_f16(rng.f32() * 0.04),
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
            d: f32_to_f16(rng.f32() * 0.03),
            m: f32_to_f16(rng.f32() * 0.015),
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
            d: f32_to_f16(rng.f32() * 0.015),
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
            d: f32_to_f16(0.008 + rng.f32().abs() * 0.01),
            dmin: f32_to_f16(0.004 + rng.f32().abs() * 0.005),
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
fn serialize_qwen35_gguf(
    n_layers: usize,
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    rope_dim: usize,
    d_conv: usize,
    s_dim: usize,
    hk: usize,
    hv: usize,
    vocab: usize,
    tensors: Vec<TensorEntry>,
) -> Vec<u8> {
    let mut kv = Vec::new();
    kv_str(&mut kv, "general.architecture", "qwen35");
    kv_u32(&mut kv, "qwen35.block_count", n_layers as u32);
    kv_u32(&mut kv, "qwen35.embedding_length", n_embd as u32);
    kv_u32(&mut kv, "qwen35.feed_forward_length", n_ff as u32);
    kv_u32(&mut kv, "qwen35.attention.head_count", n_head as u32);
    kv_u32(&mut kv, "qwen35.attention.head_count_kv", n_head_kv as u32);
    kv_u32(&mut kv, "qwen35.attention.key_length", head_dim as u32);
    kv_u32(&mut kv, "qwen35.rope.dimension_count", rope_dim as u32);
    kv_f32(&mut kv, "qwen35.rope.freq_base", 10_000_000.0);
    kv_f32(&mut kv, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    kv_u32(&mut kv, "qwen35.full_attention_interval", 4);
    kv_u32(&mut kv, "qwen35.ssm.conv_kernel", d_conv as u32);
    kv_u32(&mut kv, "qwen35.ssm.state_size", s_dim as u32);
    kv_u32(&mut kv, "qwen35.ssm.group_count", hk as u32);
    kv_u32(&mut kv, "qwen35.ssm.time_step_rank", hv as u32);
    kv_u32(&mut kv, "qwen35.ssm.inner_size", (hv * s_dim) as u32);
    kv_u32(&mut kv, "qwen35.vocab_size", vocab as u32);
    kv_u32(&mut kv, "qwen35.context_length", 256);

    let mut tinfos = Vec::new();
    let mut offset = 0u64;
    for t in &tensors {
        push_str(&mut tinfos, &t.name);
        tinfos.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            tinfos.extend_from_slice(&d.to_le_bytes());
        }
        tinfos.extend_from_slice(&t.ggml_type.to_le_bytes());
        tinfos.extend_from_slice(&offset.to_le_bytes());
        let padded = (t.data.len() as u64).div_ceil(32) * 32;
        offset += padded;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&18u64.to_le_bytes());
    out.extend_from_slice(&kv);
    out.extend_from_slice(&tinfos);
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for t in &tensors {
        out.extend_from_slice(&t.data);
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
    }
    out
}

fn open_temp_gguf(tag: &str, bytes: &[u8]) -> Arc<GgufFile> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "tm_vk_qwen35_{tag}_{}_{stamp}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write temp gguf");
    let f = GgufFile::open(&path).expect("open temp gguf");
    let _ = std::fs::remove_file(&path);
    Arc::new(f)
}

fn max_rel_err(got: &[f32], r: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), r.len());
    let mut max_abs = 0.0f32;
    let norm: f32 = r.iter().map(|v| v.abs()).sum::<f32>().max(1.0);
    for (&a, &b) in got.iter().zip(r.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    (max_abs / norm, max_abs)
}

fn kl_divergence(ref_logits: &[f32], cand_logits: &[f32]) -> f64 {
    let mr = ref_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mc = cand_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let sr: f64 = ref_logits.iter().map(|&x| ((x as f64) - mr).exp()).sum();
    let sc: f64 = cand_logits.iter().map(|&x| ((x as f64) - mc).exp()).sum();
    let lzr = sr.ln();
    let lzc = sc.ln();
    let mut kl = 0.0f64;
    for (&r, &c) in ref_logits.iter().zip(cand_logits.iter()) {
        let lp = ((r as f64) - mr) - lzr;
        let lq = ((c as f64) - mc) - lzc;
        kl += lp.exp() * (lp - lq);
    }
    kl.max(0.0)
}

#[test]
fn vk_qwen35_f32_exact_parity_and_hk_grouping() {
    // HK = 2, HV = 4 (HK != HV!) strictly exercises `kh = vh % HK` tiled grouping (`AGENTS.md` §9)!
    let mut rng = Rng(0x5177_3501_A9B8_C7D6);
    let n_layers = 4usize; // layers 0,1,2 = GDN; layer 3 = FullAttn
    let n_embd = 64usize;
    let n_ff = 128usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = 32usize;
    let rope_dim = 16usize;
    let d_conv = 4usize;
    let s_dim = 32usize;
    let hk = 2usize;
    let hv = 4usize;
    let d_inner = hv * s_dim; // 128
    let conv_dim = 2 * hk * s_dim + d_inner; // 256
    let q_dim = n_head * head_dim; // 128
    let qg_dim = q_dim * 2; // 256
    let kv_dim = n_head_kv * head_dim; // 64
    let vocab = 96usize;

    let mut ts = vec![
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
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        },
    ];

    for l in 0..n_layers {
        let p = format!("blk.{l}");
        ts.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        });
        ts.push(TensorEntry {
            name: format!("{p}.post_attention_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        });
        if l < 3 {
            ts.push(TensorEntry {
                name: format!("{p}.attn_qkv.weight"),
                dims: vec![n_embd as u64, conv_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, conv_dim * n_embd, 0.0, 0.08),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_gate.weight"),
                dims: vec![n_embd as u64, d_inner as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, d_inner * n_embd, 0.0, 0.08),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_alpha.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv * n_embd, 0.0, 0.05),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_beta.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv * n_embd, 0.0, 0.05),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_conv1d.weight"),
                dims: vec![d_conv as u64, conv_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, conv_dim * d_conv, 0.0, 0.25),
            });
            // `ssm_a` is negative decay rate in Qwen3.5 (`-exp(A_log)`)
            ts.push(TensorEntry {
                name: format!("{p}.ssm_a"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, -0.5, 0.2),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_dt.bias"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, 0.1, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_norm.weight"),
                dims: vec![s_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, s_dim, 1.0, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_out.weight"),
                dims: vec![d_inner as u64, n_embd as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, n_embd * d_inner, 0.0, 0.06),
            });
        } else {
            ts.push(TensorEntry {
                name: format!("{p}.attn_q_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_k_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_q.weight"),
                dims: vec![n_embd as u64, qg_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, qg_dim * n_embd, 0.0, 0.08),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_k.weight"),
                dims: vec![n_embd as u64, kv_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, kv_dim * n_embd, 0.0, 0.08),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_v.weight"),
                dims: vec![n_embd as u64, kv_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, kv_dim * n_embd, 0.0, 0.08),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_output.weight"),
                dims: vec![q_dim as u64, n_embd as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, n_embd * q_dim, 0.0, 0.08),
            });
        }
        ts.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_ff * n_embd, 0.0, 0.07),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_ff * n_embd, 0.0, 0.07),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd * n_ff, 0.0, 0.07),
        });
    }

    let bytes = serialize_qwen35_gguf(
        n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, rope_dim, d_conv, s_dim, hk, hv,
        vocab, ts,
    );
    let gguf = open_temp_gguf("f32", &bytes);
    let cpu_model = Qwen35Model::from_gguf(&gguf).expect("cpu qwen35");
    let mut cpu_state = Qwen35State::new(&cpu_model, 64);
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    let Some(ctx) = ctx() else {
        return;
    };
    ctx.set_prefill_mode(VkPrefillMode::Fallback);
    let mut vk = Qwen35VkRunner::from_gguf_with_context(ctx, gguf.clone(), 64).expect("vk qwen35");

    let prompt: Vec<u32> = (0..19).map(|i| ((i * 7 + 3) % vocab) as u32).collect();
    for (pos, &tok) in prompt.iter().enumerate() {
        cpu_model
            .forward_token(&mut cpu_state, &pool, kt, tok, pos, true)
            .expect("cpu tok");
        let vk_logits = vk.forward_token(tok, pos, true).expect("vk tok");
        let (rel, abs) = max_rel_err(vk_logits, cpu_state.logits());
        assert!(
            rel <= 1e-5,
            "F32 Qwen3.5 decode step {pos} mismatch: rel={rel:.3e} abs={abs:.3e}"
        );
    }

    // Verify `reset_state()` + `forward_batch(19)` matches CPU multi-token reference!
    vk.reset_state().expect("vk reset");
    let batch_logits = vk.forward_batch(&prompt, 0, true).expect("vk batch");
    let (rel_b, abs_b) = max_rel_err(batch_logits, cpu_state.logits());
    assert!(
        rel_b <= 1e-5,
        "F32 Qwen3.5 forward_batch(19) mismatch: rel={rel_b:.3e} abs={abs_b:.3e}"
    );
}

#[test]
fn vk_qwen35_mixed_quant_coopmat_fallback_and_reset_parity() {
    let mut rng = Rng(0x9060_2528_0404_ABCD);
    // Dimensions divisible by 256 where Q5_K / Q6_K superblocks apply, matching real Qwen3.5!
    let n_layers = 4usize; // 3 GDN + 1 FullAttn
    let n_embd = 256usize;
    let n_ff = 512usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = 64usize;
    let rope_dim = 32usize;
    let d_conv = 4usize;
    let s_dim = 64usize;
    let hk = 4usize;
    let hv = 4usize;
    let d_inner = hv * s_dim; // 256 (multiple of 256 for Q5_K ssm_out!)
    let conv_dim = 2 * hk * s_dim + d_inner; // 768
    let q_dim = n_head * head_dim; // 256
    let qg_dim = q_dim * 2; // 512
    let kv_dim = n_head_kv * head_dim; // 128
    let vocab = 128usize;

    let mut ts = vec![
        // Tied Q6_K token_embd (no separate output.weight, matching Qwen3.5-0.8B/2B!)
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 14, // Q6_K
            data: make_q6_k_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        },
    ];

    for l in 0..n_layers {
        let p = format!("blk.{l}");
        ts.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        });
        ts.push(TensorEntry {
            name: format!("{p}.post_attention_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        });
        if l < 3 {
            ts.push(TensorEntry {
                name: format!("{p}.attn_qkv.weight"),
                dims: vec![n_embd as u64, conv_dim as u64],
                ggml_type: 2, // Q4_0
                data: make_q4_0_bytes(&mut rng, conv_dim, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_gate.weight"),
                dims: vec![n_embd as u64, d_inner as u64],
                ggml_type: 2, // Q4_0
                data: make_q4_0_bytes(&mut rng, d_inner, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_alpha.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: 8, // Q8_0
                data: make_q8_0_bytes(&mut rng, hv, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_beta.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: 8, // Q8_0
                data: make_q8_0_bytes(&mut rng, hv, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_conv1d.weight"),
                dims: vec![d_conv as u64, conv_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, conv_dim * d_conv, 0.0, 0.2),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_a"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, -0.4, 0.15),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_dt.bias"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, 0.05, 0.05),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_norm.weight"),
                dims: vec![s_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, s_dim, 1.0, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_out.weight"),
                dims: vec![d_inner as u64, n_embd as u64],
                ggml_type: 13, // Q5_K
                data: make_q5_k_bytes(&mut rng, n_embd, d_inner),
            });
        } else {
            ts.push(TensorEntry {
                name: format!("{p}.attn_q_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_k_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_q.weight"),
                dims: vec![n_embd as u64, qg_dim as u64],
                ggml_type: 2, // Q4_0
                data: make_q4_0_bytes(&mut rng, qg_dim, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_k.weight"),
                dims: vec![n_embd as u64, kv_dim as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, kv_dim, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_v.weight"),
                dims: vec![n_embd as u64, kv_dim as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, kv_dim, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_output.weight"),
                dims: vec![q_dim as u64, n_embd as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, n_embd, q_dim),
            });
        }
        ts.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2, // Q4_0
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2, // Q4_0
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 3, // Q4_1 (matching Qwen3.5 ffn_down!)
            data: make_q4_1_bytes(&mut rng, n_embd, n_ff),
        });
    }

    let bytes = serialize_qwen35_gguf(
        n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, rope_dim, d_conv, s_dim, hk, hv,
        vocab, ts,
    );
    let gguf = open_temp_gguf("mixed", &bytes);
    let cpu_model = Qwen35Model::from_gguf(&gguf).expect("cpu qwen35");
    let mut cpu_state = Qwen35State::new(&cpu_model, 64);
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    let prompt: Vec<u32> = (0..35).map(|i| ((i * 11 + 5) % vocab) as u32).collect();
    cpu_model
        .forward_batch(&mut cpu_state, &pool, kt, &prompt, 0, true)
        .expect("cpu batch");

    let Some(base_ctx) = ctx() else {
        return;
    };
    let info = base_ctx.info().clone();

    // 1. Sequential pre-recorded decode + Fallback batch invariance
    base_ctx.set_prefill_mode(VkPrefillMode::Fallback);
    let mut vk_fb =
        Qwen35VkRunner::from_gguf_with_context(base_ctx, gguf.clone(), 64).expect("vk fb");
    let mut dec_logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let is_last = pos + 1 == prompt.len();
        let l = vk_fb.forward_token(tok, pos, is_last).expect("vk dec");
        if is_last {
            dec_logits = l.to_vec();
        }
    }
    let greedy_tok = vk_fb
        .forward_token_greedy(prompt[0], prompt.len())
        .expect("vk greedy");
    assert!((greedy_tok as usize) < vocab);

    vk_fb.reset_state().expect("reset");
    let fb_batch = vk_fb
        .forward_batch(&prompt, 0, true)
        .expect("fb batch")
        .to_vec();
    let (rel_fb, _) = max_rel_err(&fb_batch, &dec_logits);
    assert!(
        rel_fb <= 1e-5,
        "Mixed-quant Fallback batch(35) vs decode mismatch: rel={rel_fb:.3e}"
    );

    // Repeat after reset_state() -> must be bit-identical (0.0 diff)!
    vk_fb.reset_state().expect("reset 2");
    let fb_batch_2 = vk_fb.forward_batch(&prompt, 0, true).expect("fb batch 2");
    let (_, abs_rep) = max_rel_err(fb_batch_2, &fb_batch);
    assert_eq!(
        abs_rep, 0.0,
        "reset_state() must produce bit-identical output"
    );

    // Compare against CPU Q8-activation reference (`Q8_0`/`Q8_1`/`Q8_K` quantized activations vs Vulkan FP32)
    let kl_cpu = kl_divergence(cpu_state.logits(), &dec_logits);
    assert!(
        kl_cpu < 2.5e-3 && argmax(cpu_state.logits()) == argmax(&dec_logits),
        "Vulkan vs CPU Qwen3.5 mixed-quant KL={kl_cpu:.3e}"
    );

    // 2. Test CoopMatF16 + CoopMatI8 prefill modes if hardware supports them!
    let sg = info.subgroup_size;
    if info.coopmat_f16_f32acc && (sg == 32 || sg == 64) {
        let ctx_f16 = VulkanContext::new().expect("ctx f16");
        ctx_f16.set_prefill_mode(VkPrefillMode::CoopMatF16);
        let mut vk_f16 =
            Qwen35VkRunner::from_gguf_with_context(ctx_f16, gguf.clone(), 64).expect("vk f16");
        let f16_logits = vk_f16.forward_batch(&prompt, 0, true).expect("f16 batch");
        let kl_f16 = kl_divergence(&dec_logits, f16_logits);
        let (rel_f16, _) = max_rel_err(f16_logits, &dec_logits);
        assert!(
            kl_f16 < 2e-4 && rel_f16 <= 6e-4 && argmax(f16_logits) == argmax(&dec_logits),
            "CoopMatF16 Qwen3.5 batch mismatch: kl={kl_f16:.3e} rel={rel_f16:.3e}"
        );
    }

    if info.coopmat_i8_i32acc && (sg == 32 || sg == 64) {
        let ctx_i8 = VulkanContext::new().expect("ctx i8");
        ctx_i8.set_prefill_mode(VkPrefillMode::CoopMatI8);
        let mut vk_i8 = Qwen35VkRunner::from_gguf_with_context(ctx_i8, gguf, 64).expect("vk i8");
        let i8_logits = vk_i8.forward_batch(&prompt, 0, true).expect("i8 batch");
        let kl_i8 = kl_divergence(&dec_logits, i8_logits);
        assert!(
            kl_i8 < 1.5e-3 && argmax(i8_logits) == argmax(&dec_logits),
            "CoopMatI8 Qwen3.5 batch mismatch: kl={kl_i8:.3e}"
        );
    }
}

/// Independent FP64 (`f64`) GDN oracle (`TESTING.md` §1: GDN operators compared against FP64 reference)
/// verifying causal Conv1d + pre-split SiLU + Q/K-only L2 norm + `vh % HK` tiled recurrence +
/// `1/sqrt(S)` query scaling + gated RMSNorm with SiLU on `z` across 12 recurrent steps.
#[test]
fn vk_qwen35_gdn_layer_vs_fp64_oracle() {
    let mut rng = Rng(0x6464_6464_1234_5678);
    let n_embd = 64usize;
    let n_ff = 128usize;
    let d_conv = 4usize;
    let s_dim = 32usize;
    let hk = 2usize;
    let hv = 4usize; // HK != HV -> exercises `kh = vh % HK`!
    let d_inner = hv * s_dim;
    let conv_dim = 2 * hk * s_dim + d_inner;
    let vocab = 64usize;
    let eps = 1e-6f64;

    let make_vec = |rng: &mut Rng, len: usize, center: f32, scale: f32| -> (Vec<f32>, Vec<u8>) {
        let mut vf = Vec::with_capacity(len);
        let mut vb = Vec::with_capacity(len * 4);
        for _ in 0..len {
            let v = center + rng.f32() * scale;
            vf.push(v);
            vb.extend_from_slice(&v.to_le_bytes());
        }
        (vf, vb)
    };

    let (emb, emb_b) = make_vec(&mut rng, vocab * n_embd, 0.0, 0.25);
    let (out_norm, out_norm_b) = make_vec(&mut rng, n_embd, 1.0, 0.1);
    let (attn_norm, attn_norm_b) = make_vec(&mut rng, n_embd, 1.0, 0.1);
    let (post_norm, post_norm_b) = make_vec(&mut rng, n_embd, 1.0, 0.1);
    let (wqkv, wqkv_b) = make_vec(&mut rng, conv_dim * n_embd, 0.0, 0.12);
    let (wz, wz_b) = make_vec(&mut rng, d_inner * n_embd, 0.0, 0.12);
    let (w_alpha, w_alpha_b) = make_vec(&mut rng, hv * n_embd, 0.0, 0.08);
    let (w_beta, w_beta_b) = make_vec(&mut rng, hv * n_embd, 0.0, 0.08);
    let (conv1d, conv1d_b) = make_vec(&mut rng, conv_dim * d_conv, 0.0, 0.3);
    let (ssm_a, ssm_a_b) = make_vec(&mut rng, hv, -0.45, 0.2);
    let (dt_bias, dt_bias_b) = make_vec(&mut rng, hv, 0.1, 0.1);
    let (ssm_norm, ssm_norm_b) = make_vec(&mut rng, s_dim, 1.0, 0.1);
    let (w_out, w_out_b) = make_vec(&mut rng, n_embd * d_inner, 0.0, 0.1);
    let (w_gate, w_gate_b) = make_vec(&mut rng, n_ff * n_embd, 0.0, 0.09);
    let (w_up, w_up_b) = make_vec(&mut rng, n_ff * n_embd, 0.0, 0.09);
    let (w_down, w_down_b) = make_vec(&mut rng, n_embd * n_ff, 0.0, 0.09);

    let ts = vec![
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 0,
            data: emb_b,
        },
        TensorEntry {
            name: "output_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: out_norm_b,
        },
        TensorEntry {
            name: "blk.0.attn_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: attn_norm_b,
        },
        TensorEntry {
            name: "blk.0.post_attention_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: post_norm_b,
        },
        TensorEntry {
            name: "blk.0.attn_qkv.weight".into(),
            dims: vec![n_embd as u64, conv_dim as u64],
            ggml_type: 0,
            data: wqkv_b,
        },
        TensorEntry {
            name: "blk.0.attn_gate.weight".into(),
            dims: vec![n_embd as u64, d_inner as u64],
            ggml_type: 0,
            data: wz_b,
        },
        TensorEntry {
            name: "blk.0.ssm_alpha.weight".into(),
            dims: vec![n_embd as u64, hv as u64],
            ggml_type: 0,
            data: w_alpha_b,
        },
        TensorEntry {
            name: "blk.0.ssm_beta.weight".into(),
            dims: vec![n_embd as u64, hv as u64],
            ggml_type: 0,
            data: w_beta_b,
        },
        TensorEntry {
            name: "blk.0.ssm_conv1d.weight".into(),
            dims: vec![d_conv as u64, conv_dim as u64],
            ggml_type: 0,
            data: conv1d_b,
        },
        TensorEntry {
            name: "blk.0.ssm_a".into(),
            dims: vec![hv as u64],
            ggml_type: 0,
            data: ssm_a_b,
        },
        TensorEntry {
            name: "blk.0.ssm_dt.bias".into(),
            dims: vec![hv as u64],
            ggml_type: 0,
            data: dt_bias_b,
        },
        TensorEntry {
            name: "blk.0.ssm_norm.weight".into(),
            dims: vec![s_dim as u64],
            ggml_type: 0,
            data: ssm_norm_b,
        },
        TensorEntry {
            name: "blk.0.ssm_out.weight".into(),
            dims: vec![d_inner as u64, n_embd as u64],
            ggml_type: 0,
            data: w_out_b,
        },
        TensorEntry {
            name: "blk.0.ffn_gate.weight".into(),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 0,
            data: w_gate_b,
        },
        TensorEntry {
            name: "blk.0.ffn_up.weight".into(),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 0,
            data: w_up_b,
        },
        TensorEntry {
            name: "blk.0.ffn_down.weight".into(),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 0,
            data: w_down_b,
        },
    ];

    let bytes = serialize_qwen35_gguf(
        1, n_embd, n_ff, 2, 2, 32, 16, d_conv, s_dim, hk, hv, vocab, ts,
    );
    let gguf = open_temp_gguf("fp64_oracle", &bytes);
    let Some(ctx) = ctx() else {
        return;
    };
    ctx.set_prefill_mode(VkPrefillMode::Fallback);
    let mut vk = Qwen35VkRunner::from_gguf_with_context(ctx, gguf, 32).expect("vk runner");

    let rms_norm_64 = |x: &[f64], w: &[f32]| -> Vec<f64> {
        let ss: f64 = x.iter().map(|&v| v * v).sum::<f64>() / (x.len() as f64);
        let sc = 1.0 / (ss + eps).sqrt();
        x.iter()
            .zip(w.iter())
            .map(|(&xi, &wi)| xi * sc * (wi as f64))
            .collect()
    };
    let matvec_64 = |w: &[f32], x: &[f64], rows: usize, cols: usize| -> Vec<f64> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| (w[r * cols + c] as f64) * x[c])
                    .sum::<f64>()
            })
            .collect()
    };
    let silu_64 = |v: f64| v / (1.0 + (-v).exp());

    let mut conv_hist = vec![0.0f64; conv_dim * 3];
    let mut state = vec![0.0f64; hv * s_dim * s_dim];
    let toks: Vec<u32> = (0..12).map(|i| ((i * 5 + 2) % vocab) as u32).collect();

    for (pos, &tok) in toks.iter().enumerate() {
        let tid = tok as usize;
        let mut x: Vec<f64> = emb[tid * n_embd..(tid + 1) * n_embd]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let xb = rms_norm_64(&x, &attn_norm);

        let qkv = matvec_64(&wqkv, &xb, conv_dim, n_embd);
        let z = matvec_64(&wz, &xb, d_inner, n_embd);
        let alpha = matvec_64(&w_alpha, &xb, hv, n_embd);
        let beta_raw = matvec_64(&w_beta, &xb, hv, n_embd);

        let mut act_qkv = vec![0.0f64; conv_dim];
        for ch in 0..conv_dim {
            let h = &mut conv_hist[ch * 3..(ch + 1) * 3];
            let w = &conv1d[ch * 4..(ch + 1) * 4];
            let raw = h[0] * (w[0] as f64)
                + h[1] * (w[1] as f64)
                + h[2] * (w[2] as f64)
                + qkv[ch] * (w[3] as f64);
            h[0] = h[1];
            h[1] = h[2];
            h[2] = qkv[ch];
            act_qkv[ch] = silu_64(raw);
        }

        let k_dim = hk * s_dim;
        let mut q = act_qkv[0..k_dim].to_vec();
        let mut k = act_qkv[k_dim..2 * k_dim].to_vec();
        let v = act_qkv[2 * k_dim..conv_dim].to_vec();
        for h_idx in 0..hk {
            let qs = &mut q[h_idx * s_dim..(h_idx + 1) * s_dim];
            let q_norm = (qs.iter().map(|&x| x * x).sum::<f64>() + eps).sqrt();
            for x in qs.iter_mut() {
                *x /= q_norm;
            }
            let ks = &mut k[h_idx * s_dim..(h_idx + 1) * s_dim];
            let k_norm = (ks.iter().map(|&x| x * x).sum::<f64>() + eps).sqrt();
            for x in ks.iter_mut() {
                *x /= k_norm;
            }
        }

        let mut gdn_out = vec![0.0f64; d_inner];
        let q_scale = 1.0 / (s_dim as f64).sqrt();
        for vh_i in 0..hv {
            let kh_i = vh_i % hk;
            let x_sp = alpha[vh_i] + (dt_bias[vh_i] as f64);
            let sp = if x_sp > 20.0 {
                x_sp
            } else {
                (1.0 + x_sp.exp()).ln()
            };
            let g = (sp * (ssm_a[vh_i] as f64)).exp();
            let b = 1.0 / (1.0 + (-beta_raw[vh_i]).exp());

            let q_h = &q[kh_i * s_dim..(kh_i + 1) * s_dim];
            let k_h = &k[kh_i * s_dim..(kh_i + 1) * s_dim];
            let v_h = &v[vh_i * s_dim..(vh_i + 1) * s_dim];
            let mut o_h = vec![0.0f64; s_dim];

            for j in 0..s_dim {
                let s_row = &mut state[(vh_i * s_dim + j) * s_dim..(vh_i * s_dim + j + 1) * s_dim];
                let mut sum_kv = 0.0f64;
                for i in 0..s_dim {
                    s_row[i] *= g;
                    sum_kv += s_row[i] * k_h[i];
                }
                let delta = (v_h[j] - sum_kv) * b;
                let mut sum_oq = 0.0f64;
                for i in 0..s_dim {
                    s_row[i] += delta * k_h[i];
                    sum_oq += s_row[i] * q_h[i];
                }
                o_h[j] = sum_oq * q_scale;
            }

            let normed = rms_norm_64(&o_h, &ssm_norm);
            let z_h = &z[vh_i * s_dim..(vh_i + 1) * s_dim];
            for i in 0..s_dim {
                gdn_out[vh_i * s_dim + i] = normed[i] * silu_64(z_h[i]);
            }
        }

        let xb2 = matvec_64(&w_out, &gdn_out, n_embd, d_inner);
        for i in 0..n_embd {
            x[i] += xb2[i];
        }
        let ffn_in = rms_norm_64(&x, &post_norm);
        let gate = matvec_64(&w_gate, &ffn_in, n_ff, n_embd);
        let up = matvec_64(&w_up, &ffn_in, n_ff, n_embd);
        let hb: Vec<f64> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu_64(g) * u)
            .collect();
        let down = matvec_64(&w_down, &hb, n_embd, n_ff);
        for i in 0..n_embd {
            x[i] += down[i];
        }
        let final_in = rms_norm_64(&x, &out_norm);
        let ref_logits: Vec<f32> = matvec_64(&emb, &final_in, vocab, n_embd)
            .into_iter()
            .map(|v| v as f32)
            .collect();

        let vk_logits = vk.forward_token(tok, pos, true).expect("vk decode");
        let (rel, abs) = max_rel_err(vk_logits, &ref_logits);
        assert!(
            rel <= 1e-5 && abs <= 1e-3,
            "FP64 oracle step {pos} mismatch: rel={rel:.3e} abs={abs:.3e}"
        );
    }
}
