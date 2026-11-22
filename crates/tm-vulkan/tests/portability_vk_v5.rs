//! V5 cross-device Vulkan portability, subgroup-width (`8 / 16 / 32 / 64`), no-shuffle LDS fallback,
//! UMA zero-copy (`VK_EXT_external_memory_host`), and trap/bounds safety tests (`docs/research/11-amdgpu.md` §4.1 & §5).
//!
//! Running across `TM_VK_DEVICE=radv`, `TM_VK_DEVICE=UHD`, and `TM_VK_DEVICE=llvmpipe` exercises
//! real hardware execution on subgroup widths `8`, `16` (exact Mali-G715 / Mali-G610 / Pi 5 V3DV
//! subgroup width!), `32`, and `64` (AMD wave64).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tm_core::{BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, f32_to_f16};
use tm_gguf::GgufFile;
use tm_models::argmax;
use tm_vulkan::{
    LlamaVkRunner, Q4Layout, Qwen35VkRunner, VkPrefillMode, VulkanContext, VulkanError,
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

fn open_temp_gguf(tag: &str, bytes: &[u8]) -> Arc<GgufFile> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!(
        "tm_vk_v5_{tag}_{}_{stamp}.gguf",
        std::process::id()
    ));
    std::fs::write(&path, bytes).expect("write temp gguf");
    let f = GgufFile::open(&path).expect("open temp gguf");
    let _ = std::fs::remove_file(&path);
    Arc::new(f)
}

fn build_synthetic_qwen3_gguf() -> Arc<GgufFile> {
    let mut rng = Rng(0x7501_CA5E_99AA_5511);
    let n_layers = 2usize;
    let n_embd = 256usize;
    let n_ff = 512usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = 64usize;
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let vocab = 128usize;

    let mut kv = Vec::new();
    kv_str(&mut kv, "general.architecture", "qwen3");
    kv_u32(&mut kv, "qwen3.block_count", n_layers as u32);
    kv_u32(&mut kv, "qwen3.embedding_length", n_embd as u32);
    kv_u32(&mut kv, "qwen3.feed_forward_length", n_ff as u32);
    kv_u32(&mut kv, "qwen3.attention.head_count", n_head as u32);
    kv_u32(&mut kv, "qwen3.attention.head_count_kv", n_head_kv as u32);
    kv_u32(&mut kv, "qwen3.attention.key_length", head_dim as u32);
    kv_u32(&mut kv, "qwen3.rope.dimension_count", head_dim as u32);
    kv_f32(&mut kv, "qwen3.rope.freq_base", 1_000_000.0);
    kv_f32(&mut kv, "qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    kv_u32(&mut kv, "qwen3.vocab_size", vocab as u32);
    kv_u32(&mut kv, "qwen3.context_length", 256);

    let mut ts = vec![
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
            name: format!("{p}.ffn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.1),
        });
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
            dims: vec![n_embd as u64, q_dim as u64],
            ggml_type: 2, // Q4_0
            data: make_q4_0_bytes(&mut rng, q_dim, n_embd),
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
            ggml_type: 13, // Q5_K
            data: make_q5_k_bytes(&mut rng, n_embd, q_dim),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 3, // Q4_1
            data: make_q4_1_bytes(&mut rng, n_embd, n_ff),
        });
    }

    let mut tinfos = Vec::new();
    let mut offset = 0u64;
    for t in &ts {
        push_str(&mut tinfos, &t.name);
        tinfos.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            tinfos.extend_from_slice(&d.to_le_bytes());
        }
        tinfos.extend_from_slice(&t.ggml_type.to_le_bytes());
        tinfos.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).div_ceil(64) * 64;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(ts.len() as u64).to_le_bytes());
    out.extend_from_slice(&13u64.to_le_bytes());
    kv_u32(&mut kv, "general.alignment", 64);
    out.extend_from_slice(&kv);
    out.extend_from_slice(&tinfos);
    while !out.len().is_multiple_of(64) {
        out.push(0);
    }
    for t in &ts {
        out.extend_from_slice(&t.data);
        while !out.len().is_multiple_of(64) {
            out.push(0);
        }
    }
    while !out.len().is_multiple_of(4096) {
        out.push(0);
    }
    open_temp_gguf("qwen3_v5", &out)
}

fn build_synthetic_qwen35_gguf() -> Arc<GgufFile> {
    let mut rng = Rng(0x35_7501_BEEF_CAFE);
    let n_layers = 4usize;
    let n_embd = 256usize;
    let n_ff = 512usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = 128usize;
    let rope_dim = 64usize;
    let d_conv = 4usize;
    let s_dim = 64usize;
    let hk = 2usize;
    let hv = 4usize;
    let d_inner = hv * s_dim; // 256
    let conv_dim = 2 * hk * s_dim + d_inner; // 512
    let q_dim = n_head * head_dim; // 512
    let qg_dim = q_dim * 2; // 1024
    let kv_dim = n_head_kv * head_dim; // 256
    let vocab = 128usize;

    let mut kv = Vec::new();
    kv_str(&mut kv, "general.architecture", "qwen35");
    kv_u32(&mut kv, "general.alignment", 64);
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
    kv_u32(&mut kv, "qwen35.ssm.inner_size", d_inner as u32);
    kv_u32(&mut kv, "qwen35.vocab_size", vocab as u32);
    kv_u32(&mut kv, "qwen35.context_length", 256);

    let mut ts = vec![
        TensorEntry {
            name: "token_embd.weight".into(),
            dims: vec![n_embd as u64, vocab as u64],
            ggml_type: 14,
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
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, conv_dim, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.attn_gate.weight"),
                dims: vec![n_embd as u64, d_inner as u64],
                ggml_type: 2,
                data: make_q4_0_bytes(&mut rng, d_inner, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_alpha.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: 8,
                data: make_q8_0_bytes(&mut rng, hv, n_embd),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_beta.weight"),
                dims: vec![n_embd as u64, hv as u64],
                ggml_type: 8,
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
                data: make_f32_bytes(&mut rng, hv, -0.45, 0.15),
            });
            ts.push(TensorEntry {
                name: format!("{p}.ssm_dt.bias"),
                dims: vec![hv as u64],
                ggml_type: 0,
                data: make_f32_bytes(&mut rng, hv, 0.05, 0.1),
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
                ggml_type: 13,
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
                ggml_type: 2,
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
                ggml_type: 13,
                data: make_q5_k_bytes(&mut rng, n_embd, q_dim),
            });
        }
        ts.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 2,
            data: make_q4_0_bytes(&mut rng, n_ff, n_embd),
        });
        ts.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 3,
            data: make_q4_1_bytes(&mut rng, n_embd, n_ff),
        });
    }

    let mut tinfos = Vec::new();
    let mut offset = 0u64;
    for t in &ts {
        push_str(&mut tinfos, &t.name);
        tinfos.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            tinfos.extend_from_slice(&d.to_le_bytes());
        }
        tinfos.extend_from_slice(&t.ggml_type.to_le_bytes());
        tinfos.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).div_ceil(64) * 64;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(ts.len() as u64).to_le_bytes());
    out.extend_from_slice(&19u64.to_le_bytes());
    out.extend_from_slice(&kv);
    out.extend_from_slice(&tinfos);
    while !out.len().is_multiple_of(64) {
        out.push(0);
    }
    for t in &ts {
        out.extend_from_slice(&t.data);
        while !out.len().is_multiple_of(64) {
            out.push(0);
        }
    }
    while !out.len().is_multiple_of(4096) {
        out.push(0);
    }
    open_temp_gguf("qwen35_v5", &out)
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

#[test]
fn vk_v5_trap_checks_never_silently_drop_work() {
    let Some(ctx) = ctx() else {
        return;
    };

    // 1. Bogus device selector must return `DeviceNotFound`.
    let bad = VulkanContext::with_selector(Some("nonexistent-vulkan-device-xyz-999"));
    assert!(
        matches!(bad, Err(VulkanError::DeviceNotFound(_))),
        "expected DeviceNotFound, got {:?}",
        bad.err()
    );

    // 2. Invalid non-power-of-two subgroup size must fail cleanly.
    let sel = std::env::var("TM_VK_DEVICE").ok();
    let bad_sg = VulkanContext::with_selector_and_subgroup(sel.as_deref(), Some(24));
    assert!(
        matches!(bad_sg, Err(VulkanError::InvalidDimensions(_))),
        "expected InvalidDimensions on sg=24, got {:?}",
        bad_sg.err()
    );

    let max_wg = ctx.info().max_workgroup_invocations;
    let sg = ctx.info().subgroup_size;

    // 3. Workgroup exceeding `maxComputeWorkGroupInvocations` MUST return `WorkgroupTooLarge`
    //    instead of silently skipping dispatch (`AGENTS.md` §9 & `11-amdgpu.md` §5).
    let blocks = vec![
        BlockQ4_0 {
            d: f32_to_f16(0.1),
            qs: [0x88; 16],
        };
        4
    ];
    let w = ctx
        .upload_q4_0(&blocks, 4, 32, Q4Layout::Aos)
        .expect("upload q4_0");
    let x = ctx.upload(&[1.0f32; 32]).expect("upload x");
    let y = ctx.alloc_buffer(16).expect("alloc y");

    let over_wg = (max_wg / sg + 1) * sg;
    ctx.set_q4_0_gemv_config(over_wg, 2)
        .expect("aligned oversize wg accepted before pipeline creation");
    let err_large = ctx.gemm_q4_0_f32(&w, &x, &y, 1, 1);
    assert!(
        matches!(err_large, Err(VulkanError::WorkgroupTooLarge { .. })),
        "expected WorkgroupTooLarge for wg={over_wg}, got {err_large:?}",
    );

    // 4. Workgroup not a multiple of `subgroup_size` MUST return `InvalidDimensions` before mutating config.
    if sg >= 8 {
        let err_unaligned = ctx.set_q4_0_gemv_config(sg + 3, 2);
        assert!(
            matches!(err_unaligned, Err(VulkanError::InvalidDimensions(_))),
            "expected InvalidDimensions for wg={}, got {err_unaligned:?}",
            sg + 3
        );
    }

    // Restore valid config and verify normal dispatch still succeeds cleanly (`MUST #6`).
    ctx.set_q4_0_gemv_config(128, 2)
        .expect("restore valid gemv config");
    ctx.gemm_q4_0_f32(&w, &x, &y, 1, 1)
        .expect("valid gemv after rejected trap");

    // 5. Oversized `max_ctx` must fail upfront with `OutOfDeviceMemory` or `BufferTooLarge`
    //    before allocating/mutating device state (`MUST #6`).
    let gguf = build_synthetic_qwen3_gguf();
    let huge = LlamaVkRunner::from_gguf_with_context(ctx, gguf, 250_000_000);
    assert!(
        matches!(
            huge,
            Err(VulkanError::OutOfDeviceMemory { .. } | VulkanError::BufferTooLarge { .. })
        ),
        "expected upfront memory error on max_ctx=250M, got {:?}",
        huge.err()
    );
}

#[test]
fn vk_v5_subgroup_width_matrix_and_zero_copy_parity() {
    let sel = std::env::var("TM_VK_DEVICE").ok();
    let Some(base_ctx) = ctx() else {
        return;
    };
    let info = base_ctx.info().clone();

    let mut supported_sgs = Vec::new();
    if info.subgroup_size_control {
        for &cand in &[8u32, 16, 32, 64] {
            if cand >= info.min_subgroup_size && cand <= info.max_subgroup_size {
                supported_sgs.push(cand);
            }
        }
    } else {
        supported_sgs.push(info.subgroup_size);
    }
    assert!(!supported_sgs.is_empty());

    let qwen3_gguf = build_synthetic_qwen3_gguf();
    let qwen35_gguf = build_synthetic_qwen35_gguf();
    let prompt = [
        3u32, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123, 7, 15, 23,
    ];

    // Baseline outputs on the device's default subgroup size (`Fallback` mode).
    base_ctx.set_prefill_mode(VkPrefillMode::Fallback);
    let mut ref_q3 =
        LlamaVkRunner::from_gguf_with_context(base_ctx, qwen3_gguf.clone(), 64).expect("ref q3");
    let ref_q3_pp = ref_q3
        .forward_batch(&prompt, 0, true)
        .expect("ref q3 pp")
        .to_vec();
    let ref_q3_dec = ref_q3
        .forward_token(29, prompt.len(), true)
        .expect("ref q3 dec")
        .to_vec();

    let base_ctx35 = VulkanContext::with_selector(sel.as_deref()).expect("base ctx35");
    base_ctx35.set_prefill_mode(VkPrefillMode::Fallback);
    let mut ref_q35 = Qwen35VkRunner::from_gguf_with_context(base_ctx35, qwen35_gguf.clone(), 64)
        .expect("ref q35");
    let ref_q35_pp = ref_q35
        .forward_batch(&prompt, 0, true)
        .expect("ref q35 pp")
        .to_vec();
    let ref_q35_dec = ref_q35
        .forward_token(29, prompt.len(), true)
        .expect("ref q35 dec")
        .to_vec();

    eprintln!(
        "[portability_vk_v5] device='{}' subgroups={supported_sgs:?} q3_weights={} q35_weights={}",
        info.name,
        ref_q3.weights_mode(),
        ref_q35.weights_mode(),
    );

    // 1. Sweep every hardware-supported subgroup width (`8 / 16 / 32 / 64`).
    for &sg in &supported_sgs {
        let ctx_q3 = VulkanContext::with_selector_and_subgroup(sel.as_deref(), Some(sg))
            .unwrap_or_else(|e| panic!("create context with sg={sg}: {e}"));
        assert_eq!(ctx_q3.info().subgroup_size, sg);
        ctx_q3.set_prefill_mode(VkPrefillMode::Fallback);

        let mut r_q3 = LlamaVkRunner::from_gguf_with_context(ctx_q3, qwen3_gguf.clone(), 64)
            .unwrap_or_else(|e| panic!("LlamaVkRunner sg={sg}: {e}"));
        let got_q3_pp = r_q3.forward_batch(&prompt, 0, true).expect("q3 pp");
        let (rel_pp, abs_pp) = max_rel_err(got_q3_pp, &ref_q3_pp);
        assert!(
            rel_pp <= 1e-5 && abs_pp <= 1e-3,
            "Qwen3 Fallback sg={sg} pp drift: rel={rel_pp:.3e}, abs={abs_pp:.3e}"
        );
        let got_q3_dec = r_q3.forward_token(29, prompt.len(), true).expect("q3 dec");
        let (rel_dec, abs_dec) = max_rel_err(got_q3_dec, &ref_q3_dec);
        assert!(
            rel_dec <= 1e-5 && abs_dec <= 1e-3,
            "Qwen3 Fallback sg={sg} dec drift: rel={rel_dec:.3e}, abs={abs_dec:.3e}"
        );
        assert_eq!(argmax(got_q3_dec), argmax(&ref_q3_dec));

        let ctx_q35 = VulkanContext::with_selector_and_subgroup(sel.as_deref(), Some(sg))
            .unwrap_or_else(|e| panic!("create ctx35 with sg={sg}: {e}"));
        ctx_q35.set_prefill_mode(VkPrefillMode::Fallback);
        let mut r_q35 = Qwen35VkRunner::from_gguf_with_context(ctx_q35, qwen35_gguf.clone(), 64)
            .unwrap_or_else(|e| panic!("Qwen35VkRunner sg={sg}: {e}"));
        let got_q35_pp = r_q35.forward_batch(&prompt, 0, true).expect("q35 pp");
        let (rel35_pp, abs35_pp) = max_rel_err(got_q35_pp, &ref_q35_pp);
        assert!(
            rel35_pp <= 1e-5 && abs35_pp <= 1e-3,
            "Qwen3.5 Fallback sg={sg} pp drift: rel={rel35_pp:.3e}, abs={abs35_pp:.3e}"
        );
        let got_q35_dec = r_q35
            .forward_token(29, prompt.len(), true)
            .expect("q35 dec");
        let (rel35_dec, abs35_dec) = max_rel_err(got_q35_dec, &ref_q35_dec);
        assert!(
            rel35_dec <= 1e-5 && abs35_dec <= 1e-3,
            "Qwen3.5 Fallback sg={sg} dec drift: rel={rel35_dec:.3e}, abs={abs35_dec:.3e}"
        );
        assert_eq!(argmax(got_q35_dec), argmax(&ref_q35_dec));

        // Also verify `CoopMatF16` on wave64 (`sg == 64`) when supported (e.g. RDNA4 RX 9060 XT)!
        if info.coopmat_f16_f32acc && (sg == 32 || sg == 64) {
            let ctx_cm = VulkanContext::with_selector_and_subgroup(sel.as_deref(), Some(sg))
                .expect("ctx coopmat");
            ctx_cm.set_prefill_mode(VkPrefillMode::CoopMatF16);
            let mut cm_q35 =
                Qwen35VkRunner::from_gguf_with_context(ctx_cm, qwen35_gguf.clone(), 64)
                    .unwrap_or_else(|e| panic!("Qwen35 CoopMatF16 sg={sg}: {e}"));
            let cm_logits = cm_q35.forward_batch(&prompt, 0, true).expect("cm pp");
            let (rel_cm, _) = max_rel_err(cm_logits, &ref_q35_pp);
            assert!(
                rel_cm <= 5e-4,
                "Qwen3.5 CoopMatF16 sg={sg} drift: rel={rel_cm:.3e}"
            );
            assert_eq!(argmax(cm_logits), argmax(&ref_q35_pp));
        }
    }

    // 2. Verify Pi 5 `V3DV` subgroup-shuffle-free LDS recurrence (`gdn_recurrence_lds.comp`)
    //    is BIT-FOR-BIT IDENTICAL (`0.0` diff!) to `gdn_recurrence.comp`!
    let ctx_lds = VulkanContext::with_selector(sel.as_deref()).expect("ctx lds");
    ctx_lds.set_prefill_mode(VkPrefillMode::Fallback);
    ctx_lds.set_force_lds_recurrence(true);
    let mut r_lds =
        Qwen35VkRunner::from_gguf_with_context(ctx_lds, qwen35_gguf.clone(), 64).expect("r_lds");
    assert!(r_lds.uses_lds_recurrence());
    let lds_pp = r_lds
        .forward_batch(&prompt, 0, true)
        .expect("lds pp")
        .to_vec();
    let lds_dec = r_lds
        .forward_token(29, prompt.len(), true)
        .expect("lds dec")
        .to_vec();
    let (_, abs_lds_pp) = max_rel_err(&lds_pp, &ref_q35_pp);
    let (_, abs_lds_dec) = max_rel_err(&lds_dec, &ref_q35_dec);
    assert_eq!(
        abs_lds_pp, 0.0,
        "gdn_recurrence_lds batch must be bit-identical to shuffle recurrence (got {abs_lds_pp:e})"
    );
    assert_eq!(
        abs_lds_dec, 0.0,
        "gdn_recurrence_lds decode must be bit-identical to shuffle recurrence (got {abs_lds_dec:e})"
    );

    // 3. On UMA devices (`Intel UHD 770`, `lavapipe`), verify `force_slab_copy = true` vs default
    //    (`VK_EXT_external_memory_host` zero-copy when accepted by kernel, else `uma-mapped`) is
    //    0.0 bit-identical!
    if info.unified_memory {
        let ctx_copy = VulkanContext::with_selector(sel.as_deref()).expect("ctx slab copy");
        ctx_copy.set_prefill_mode(VkPrefillMode::Fallback);
        ctx_copy.set_force_slab_copy(true);
        let mut r_copy =
            Qwen35VkRunner::from_gguf_with_context(ctx_copy, qwen35_gguf, 64).expect("r_copy");
        assert!(!r_copy.is_zero_copy());
        assert_eq!(r_copy.weights_mode(), "uma-mapped");
        let copy_pp = r_copy
            .forward_batch(&prompt, 0, true)
            .expect("copy pp")
            .to_vec();
        let copy_dec = r_copy
            .forward_token(29, prompt.len(), true)
            .expect("copy dec")
            .to_vec();
        assert_eq!(copy_pp, ref_q35_pp);
        assert_eq!(copy_dec, ref_q35_dec);
    }
}
