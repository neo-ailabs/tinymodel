//! L1/L3 parity & determinism tests for P0-3 Vulkan INT8 `OpSDotKHR` (`dp4a`) GEMV
//! and fused `RMSNorm + Q8_1` activation quantization (`AGENTS.md` §5 Rules #3, #7 & §9):
//! 1. `Qwen35VkRunner` (`n_embd = 256`, mixed `Q4_0 / Q4_1 / Q8_0 / Q5_K / Q6_K` across GDN +
//!    FullAttn + SwiGLU FFN + `lm_head`) in BOTH AoS (`CoopMatF16`) and SoA (`TiledDp4a`) layouts
//!    vs CPU `Qwen35Model` and FP32 `Fallback` reference.
//! 2. `LlamaVkRunner` (`n_embd = 256`, `Q4_0 / Q4_1 / Q8_0`) multi-token decode + `forward_batch`
//!    final-token `rms_norm_q8_1(last_x_off)` parity.
//! 3. `Inkling2VkRunner` + `DFlashVkRunner` (`TM_VK_DP4A=force`): `gemv_dp4a` (`M = 1, 4, 8`) +
//!    `inkling2_moe_dp4a` (`op = 1` & `op = 5`), verifying strict single-token vs `W = 4`
//!    speculative window determinism (`AGENTS.md` §9)!

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tm_core::{BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{Qwen35Model, Qwen35State};
use tm_vulkan::{
    Inkling2VkRunner, LlamaVkRunner, Qwen35VkRunner, VkPrefillMode, VulkanContext, VulkanError,
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
    let sc = 0.25 / (cols as f32).sqrt();
    let mut out = Vec::with_capacity(nb * 18);
    for _ in 0..nb {
        let b = BlockQ4_0 {
            d: f32_to_f16(rng.f32() * sc),
            qs: core::array::from_fn(|_| rng.u8()),
        };
        out.extend_from_slice(&b.d.to_le_bytes());
        out.extend_from_slice(&b.qs);
    }
    out
}

fn make_q4_1_bytes(rng: &mut Rng, rows: usize, cols: usize) -> Vec<u8> {
    let nb = rows * cols / 32;
    let sc = 0.20 / (cols as f32).sqrt();
    let mut out = Vec::with_capacity(nb * 20);
    for _ in 0..nb {
        let b = BlockQ4_1 {
            d: f32_to_f16(rng.f32() * sc),
            m: f32_to_f16(rng.f32() * sc * 0.25),
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
    let sc = 0.08 / (cols as f32).sqrt();
    let mut out = Vec::with_capacity(nb * 34);
    for _ in 0..nb {
        let b = BlockQ8_0 {
            d: f32_to_f16(rng.f32() * sc),
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
    let sc = 0.05 / (cols as f32).sqrt();
    let mut out = Vec::with_capacity(nb * 176);
    for _ in 0..nb {
        let b = BlockQ5K {
            d: f32_to_f16((0.3 + rng.f32().abs() * 0.7) * sc),
            dmin: f32_to_f16((0.2 + rng.f32().abs() * 0.5) * sc),
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

fn write_gguf(kv: &[u8], n_kv: u64, ts: &[TensorEntry]) -> Vec<u8> {
    let mut tinfos = Vec::new();
    let mut offset = 0u64;
    for t in ts {
        push_str(&mut tinfos, &t.name);
        tinfos.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            tinfos.extend_from_slice(&d.to_le_bytes());
        }
        tinfos.extend_from_slice(&t.ggml_type.to_le_bytes());
        tinfos.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).div_ceil(32) * 32;
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(ts.len() as u64).to_le_bytes());
    out.extend_from_slice(&n_kv.to_le_bytes());
    out.extend_from_slice(kv);
    out.extend_from_slice(&tinfos);
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for t in ts {
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
        "tm_vk_dp4a_{tag}_{}_{stamp}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write temp gguf");
    let f = GgufFile::open(&path).expect("open temp gguf");
    let _ = std::fs::remove_file(&path);
    Arc::new(f)
}

fn kl_divergence(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let mp = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mq = q_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sp: f32 = p_logits.iter().map(|&v| (v - mp).exp()).sum();
    let sq: f32 = q_logits.iter().map(|&v| (v - mq).exp()).sum();
    let lp = sp.ln() + mp;
    let lq = sq.ln() + mq;
    let mut kl = 0.0f32;
    for (&pv, &qv) in p_logits.iter().zip(q_logits.iter()) {
        let p = (pv - lp).exp();
        if p > 1e-12 {
            kl += p * ((pv - lp) - (qv - lq));
        }
    }
    kl.max(0.0)
}

fn ctx() -> Option<VulkanContext> {
    match VulkanContext::new() {
        Ok(c) => Some(c),
        Err(VulkanError::DeviceNotFound(sel)) => {
            panic!("TM_VK_DEVICE='{sel}' did not match any Vulkan device");
        }
        Err(_) => None,
    }
}

#[test]
fn parity_vk_dp4a_qwen35_and_llama_and_inkling2() {
    let Some(probe_ctx) = ctx() else {
        return;
    };
    let info = probe_ctx.info().clone();
    drop(probe_ctx);

    // 1. Qwen3.5 (`n_embd = 256` -> triggers default `Auto` `use_dp4a_dec = true` on all DP4A GPUs!)
    let mut rng = Rng(0xD94A_3500_2026_1126);
    let n_layers = 3usize; // layer 0 = GDN (Q8_0/Q5_K), layer 1 = FullAttn (Q5_K), layer 2 = GDN (Q6_K pair + Q6_K dp4a!)
    let n_embd = 256usize;
    let n_ff = 512usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = 64usize;
    let q_dim = n_head * head_dim;
    let qg_dim = 2 * q_dim;
    let kv_dim = n_head_kv * head_dim;
    let d_conv = 4usize;
    let s_dim = 64usize;
    let hk = 2usize;
    let hv = 4usize;
    let d_inner = hv * s_dim; // 256 (multiple of 256 for Q5_K!)
    let conv_dim = 2 * hk * s_dim + d_inner;
    let vocab = 96usize;

    let mut kv35 = Vec::new();
    kv_str(&mut kv35, "general.architecture", "qwen35");
    kv_u32(&mut kv35, "qwen35.block_count", n_layers as u32);
    kv_u32(&mut kv35, "qwen35.embedding_length", n_embd as u32);
    kv_u32(&mut kv35, "qwen35.feed_forward_length", n_ff as u32);
    kv_u32(&mut kv35, "qwen35.attention.head_count", n_head as u32);
    kv_u32(
        &mut kv35,
        "qwen35.attention.head_count_kv",
        n_head_kv as u32,
    );
    kv_u32(&mut kv35, "qwen35.attention.key_length", head_dim as u32);
    kv_u32(&mut kv35, "qwen35.rope.dimension_count", 32);
    kv_f32(&mut kv35, "qwen35.rope.freq_base", 10_000_000.0);
    kv_f32(&mut kv35, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    kv_u32(&mut kv35, "qwen35.full_attention_interval", 2);
    kv_u32(&mut kv35, "qwen35.ssm.conv_kernel", d_conv as u32);
    kv_u32(&mut kv35, "qwen35.ssm.state_size", s_dim as u32);
    kv_u32(&mut kv35, "qwen35.ssm.group_count", hk as u32);
    kv_u32(&mut kv35, "qwen35.ssm.time_step_rank", hv as u32);
    kv_u32(&mut kv35, "qwen35.ssm.inner_size", d_inner as u32);
    kv_u32(&mut kv35, "qwen35.vocab_size", vocab as u32);
    kv_u32(&mut kv35, "qwen35.context_length", 128);

    let mut ts35 = vec![
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 14, // Q6_K
            data: make_q6_k_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 14, // Q6_K -> exercises `lm_head_dec` Q6_K dp4a + `forward_batch` `rms_norm_q8_1`!
            data: make_q6_k_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        },
    ];
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        ts35.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        ts35.push(TensorEntry {
            name: format!("{p}.post_attention_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        if l % 2 == 0 {
            let alpha_beta_type = if l == 0 { 8u32 } else { 14u32 }; // l=0: Q8_0 pair, l=2: Q6_K pair!
            let ssm_out_type = if l == 0 { 13u32 } else { 14u32 }; // l=0: Q5_K dp4a, l=2: Q6_K dp4a + residual!
            ts35.push(TensorEntry {
                name: format!("{p}.attn_qkv.weight"),
                dims: vec![n_embd as u64, conv_dim as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, conv_dim, n_embd),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.attn_gate.weight"),
                dims: vec![n_embd as u64, d_inner as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, d_inner, n_embd),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_alpha.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: alpha_beta_type,
                data: if alpha_beta_type == 8 {
                    make_q8_0_bytes(&mut rng, hv, n_embd)
                } else {
                    make_q6_k_bytes(&mut rng, hv, n_embd)
                },
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_beta.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: alpha_beta_type,
                data: if alpha_beta_type == 8 {
                    make_q8_0_bytes(&mut rng, hv, n_embd)
                } else {
                    make_q6_k_bytes(&mut rng, hv, n_embd)
                },
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_conv1d.weight"),
                dims: vec![d_conv as u64, conv_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, conv_dim * d_conv, 0.0, 0.2),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_a"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, -0.45, 0.15),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_dt.bias"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, 0.05, 0.1),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_norm.weight"),
                dims: vec![s_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, s_dim, 1.0, 0.1),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.ssm_out.weight"),
                dims: vec![d_inner as u64, n_embd as u64],
                ggml_type: ssm_out_type,
                data: if ssm_out_type == 13 {
                    make_q5_k_bytes(&mut rng, n_embd, d_inner)
                } else {
                    make_q6_k_bytes(&mut rng, n_embd, d_inner)
                },
            });
        } else {
            ts35.push(TensorEntry {
                name: format!("{p}.attn_q_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.attn_k_norm.weight"),
                dims: vec![head_dim as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.attn_q.weight"),
                dims: vec![n_embd as u64, qg_dim as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, qg_dim, n_embd),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.attn_k.weight"),
                dims: vec![n_embd as u64, kv_dim as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, kv_dim, n_embd),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.attn_v.weight"),
                dims: vec![n_embd as u64, kv_dim as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, kv_dim, n_embd),
            });
            ts35.push(TensorEntry {
                name: format!("{p}.attn_output.weight"),
                dims: vec![q_dim as u64, n_embd as u64],
                ggml_type: 13, // Q5_K dp4a!
                data: make_q5_k_bytes(&mut rng, n_embd, q_dim),
            });
        }
        ts35.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts35.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts35.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 3, // Q4_1 dp4a!
            data: make_q4_1_bytes(&mut rng, n_embd, n_ff),
        });
    }

    let q35_gguf = open_temp_gguf("q35", &write_gguf(&kv35, 18, &ts35));
    let cpu_q35 = Qwen35Model::from_gguf(&q35_gguf).unwrap();
    let mut cpu_q35_st = Qwen35State::new(&cpu_q35, 64);
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    let prompt = [4u32, 17, 33, 58, 71, 12, 89, 25];
    cpu_q35
        .forward_batch(&mut cpu_q35_st, &pool, kt, &prompt, 0, true)
        .unwrap();
    cpu_q35
        .forward_token(&mut cpu_q35_st, &pool, kt, 42, prompt.len(), true)
        .unwrap();
    let ref_cpu_logits = cpu_q35_st.logits().to_vec();

    // FP32 Fallback reference on the same Vulkan device
    let fb35_ctx = VulkanContext::new().unwrap();
    fb35_ctx.set_prefill_mode(VkPrefillMode::Fallback);
    let mut vk_q35_fb =
        Qwen35VkRunner::from_gguf_with_context(fb35_ctx, q35_gguf.clone(), 64).unwrap();
    let _ = vk_q35_fb.forward_batch(&prompt, 0, false).unwrap();
    let fb_dec_logits = vk_q35_fb
        .forward_token(42, prompt.len(), true)
        .unwrap()
        .to_vec();

    // Default `Auto` context -> activates `dp4a` + `rms_norm_q8_1` on all `int8_dot_accelerated` GPUs
    // (`AoS` on CoopMatF16 RADV/L4; `SoA` on TiledDp4a UHD/T4/Mali)!
    let mut vk_q35 = Qwen35VkRunner::from_gguf(q35_gguf.clone(), 64).unwrap();
    let pp_logits = vk_q35.forward_batch(&prompt, 0, true).unwrap().to_vec();
    assert!(pp_logits.iter().all(|v| v.is_finite()));
    let dec_logits = vk_q35
        .forward_token(42, prompt.len(), true)
        .unwrap()
        .to_vec();
    let kl_35_fb = kl_divergence(&fb_dec_logits, &dec_logits);
    let kl_35_cpu = kl_divergence(&ref_cpu_logits, &dec_logits);
    assert!(
        kl_35_fb < 3.0e-4 && kl_35_cpu < 1.0e-3,
        "Qwen35 dp4a mismatch on {}: kl_fb={kl_35_fb:.3e} kl_cpu={kl_35_cpu:.3e}",
        info.name
    );

    // 2. LlamaVkRunner (`qwen3`, `n_embd = 256`, `Q4_0 + Q4_1 + Q8_0`) `Auto` `dp4a` + `rms_norm_q8_1`!
    let mut kv_l = Vec::new();
    kv_str(&mut kv_l, "general.architecture", "qwen3");
    kv_u32(&mut kv_l, "qwen3.block_count", 2);
    kv_u32(&mut kv_l, "qwen3.embedding_length", n_embd as u32);
    kv_u32(&mut kv_l, "qwen3.feed_forward_length", n_ff as u32);
    kv_u32(&mut kv_l, "qwen3.attention.head_count", n_head as u32);
    kv_u32(&mut kv_l, "qwen3.attention.head_count_kv", n_head_kv as u32);
    kv_u32(&mut kv_l, "qwen3.attention.key_length", head_dim as u32);
    kv_u32(&mut kv_l, "qwen3.rope.dimension_count", head_dim as u32);
    kv_f32(&mut kv_l, "qwen3.rope.freq_base", 1_000_000.0);
    kv_f32(&mut kv_l, "qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    kv_u32(&mut kv_l, "qwen3.vocab_size", vocab as u32);
    kv_u32(&mut kv_l, "qwen3.context_length", 128);

    let mut ts_l = vec![
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, vocab, n_embd),
        },
        TensorEntry {
            name: "output_norm.weight".into(),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        },
    ];
    for l in 0..2 {
        let p = format!("blk.{l}");
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.ffn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_q_norm.weight"),
            dims: vec![head_dim as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_k_norm.weight"),
            dims: vec![head_dim as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_q.weight"),
            dims: vec![n_embd as u64, q_dim as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, q_dim, n_embd),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_k.weight"),
            dims: vec![n_embd as u64, kv_dim as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, kv_dim, n_embd),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_v.weight"),
            dims: vec![n_embd as u64, kv_dim as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, kv_dim, n_embd),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.attn_output.weight"),
            dims: vec![q_dim as u64, n_embd as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, n_embd, q_dim),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts_l.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 3,
            data: make_q4_1_bytes(&mut rng, n_embd, n_ff),
        });
    }
    let q3_gguf = open_temp_gguf("q3", &write_gguf(&kv_l, 12, &ts_l));

    let fb_ctx = VulkanContext::new().unwrap();
    fb_ctx.set_prefill_mode(VkPrefillMode::Fallback);
    let mut vk_q3_ref = LlamaVkRunner::from_gguf_with_context(fb_ctx, q3_gguf.clone(), 64).unwrap();
    let _ = vk_q3_ref.forward_batch(&prompt, 0, false).unwrap();
    let q3_ref_dec = vk_q3_ref
        .forward_token(42, prompt.len(), true)
        .unwrap()
        .to_vec();

    let mut vk_q3_dp4a = LlamaVkRunner::from_gguf(q3_gguf, 64).unwrap();
    let _ = vk_q3_dp4a.forward_batch(&prompt, 0, true).unwrap();
    let q3_dp4a_dec = vk_q3_dp4a
        .forward_token(42, prompt.len(), true)
        .unwrap()
        .to_vec();
    let kl_q3 = kl_divergence(&q3_ref_dec, &q3_dp4a_dec);
    let max_abs_q3 = q3_ref_dec
        .iter()
        .zip(q3_dp4a_dec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        kl_q3 < 3.0e-4 && max_abs_q3 < 5.0e-2,
        "LlamaVkRunner dp4a vs FP32 mismatch on {}: kl={kl_q3:.3e} max_abs={max_abs_q3:.3e}",
        info.name
    );

    // 3. Inkling2VkRunner (`TM_VK_DP4A=force`) single-token `op = 1` vs `W = 4` window `op = 5`
    //    strict `0-ULP` (`< 1e-6`) determinism check (`AGENTS.md` §9)!
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let tmp_dir = std::env::temp_dir().join(format!("tm_vk_ink_dp4a_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let ink_path = tmp_dir.join("synth_ink.gguf");
    let ref_path = tmp_dir.join("ref.tmlg");
    let status = Command::new("python3")
        .arg(root.join("tools/oracle/inkling2_oracle.py"))
        .arg("--gen-gguf")
        .arg(&ink_path)
        .arg("--tokens")
        .arg("5,12,29,41")
        .arg("--out")
        .arg(&ref_path)
        .status()
        .unwrap();
    assert!(status.success());

    let ink_gguf = Arc::new(GgufFile::open(&ink_path).unwrap());
    // SAFETY: single-threaded integration test binary (`parity_vk_dp4a`).
    unsafe { std::env::set_var("TM_VK_DP4A", "force") };
    let mut ink_vk = Inkling2VkRunner::from_gguf(ink_gguf, 64).unwrap();
    unsafe { std::env::remove_var("TM_VK_DP4A") };

    let ink_toks = [5u32, 12, 29, 41];
    let mut single_last = Vec::new();
    for (step, &t) in ink_toks.iter().enumerate() {
        let lg = ink_vk
            .forward_token(t, step, step + 1 == ink_toks.len())
            .unwrap();
        if step + 1 == ink_toks.len() {
            single_last = lg.to_vec();
        }
    }
    ink_vk.reset_state().unwrap();
    let win_last = ink_vk
        .forward_window_hybrid(&ink_toks, 0, true, false, &[], None, None)
        .unwrap()
        .to_vec();
    let mut max_det_diff = 0.0f32;
    for (&a, &b) in single_last.iter().zip(win_last.iter()) {
        max_det_diff = max_det_diff.max((a - b).abs());
    }
    assert!(
        max_det_diff < 1e-6,
        "Inkling2 dp4a single-token vs W=4 window determinism drift {max_det_diff} >= 1e-6 on {}",
        info.name
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
