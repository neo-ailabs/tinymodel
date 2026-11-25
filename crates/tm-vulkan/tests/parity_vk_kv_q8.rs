//! End-to-end Vulkan `KvType::Q8_0` KV cache parity tests (`P1-A5`).
//!
//! Verifies `rope_kv_q8`, `gqa_attn_q8`, `gqa_attn_splitk_q8`, and `fa_gqa_tiled_q8`
//! against `KvType::F16` on:
//! 1. Standard Llama/Qwen3 GQA (`head_dim=64`, non-split-K decode + tiled prefill + greedy)
//! 2. Split-K FlashDecoding (`head_dim=128`, `gqa_attn_splitk_q8` + `gqa_attn_reduce` + prefill)
//! 3. Hybrid Qwen3.5 (GDN + full-attention with partial NeoX RoPE + `Q8_0` KV cache + `reset_state`)

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use tm_core::KvType;
use tm_gguf::GgufFile;
use tm_vulkan::{LlamaVkRunner, Qwen35VkRunner, VulkanContext};

const GGUF_MAGIC: u32 = 0x4655_4747;

struct SynthGguf {
    meta: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl SynthGguf {
    fn new() -> Self {
        Self {
            meta: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn meta_str(&mut self, k: &str, v: &str) {
        let mut b = Vec::new();
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
        self.meta.push((k.to_string(), 8, b));
    }

    fn meta_u32(&mut self, k: &str, v: u32) {
        self.meta.push((k.to_string(), 4, v.to_le_bytes().to_vec()));
    }

    fn meta_f32(&mut self, k: &str, v: f32) {
        self.meta.push((k.to_string(), 6, v.to_le_bytes().to_vec()));
    }

    fn add_f32(&mut self, name: &str, dims: &[usize], data: &[f32]) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.tensors.push((
            name.to_string(),
            dims.iter().map(|&d| d as u64).collect(),
            0,
            bytes,
        ));
    }

    fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.meta.len() as u64).to_le_bytes());
        for (k, ty, val) in &self.meta {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(val);
        }
        let mut off = 0u64;
        let mut offs = Vec::with_capacity(self.tensors.len());
        for (name, dims, dt, bytes) in &self.tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&dt.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
            offs.push(off);
            let aligned = (bytes.len() as u64 + 31) & !31;
            off += aligned;
        }
        let pad = (32 - (out.len() % 32)) % 32;
        out.resize(out.len() + pad, 0);
        let data_start = out.len();
        for ((_, _, _, bytes), &t_off) in self.tensors.iter().zip(&offs) {
            out.resize(data_start + t_off as usize, 0);
            out.extend_from_slice(bytes);
        }
        out
    }
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self, scale: f32) -> f32 {
        let u = (self.next_u32() as f32) / (u32::MAX as f32);
        (u * 2.0 - 1.0) * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32(scale)).collect()
    }
}

fn open_bytes_as_gguf(tag: &str, bytes: &[u8]) -> Arc<GgufFile> {
    let dir = std::env::temp_dir().join(format!("tm_vk_kv_q8_{}", std::process::id()));
    let _ = fs::create_dir_all(&dir);
    let path: PathBuf = dir.join(format!("{tag}.gguf"));
    fs::write(&path, bytes).expect("write synth gguf");
    let gguf = Arc::new(GgufFile::open(&path).expect("open synth gguf"));
    let _ = fs::remove_file(&path);
    gguf
}

fn build_llama_gguf(head_dim: usize) -> Vec<u8> {
    let n_layers = 2usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let n_embd = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let n_ff = n_embd * 2;
    let vocab = 64usize;

    let mut g = SynthGguf::new();
    g.meta_str("general.architecture", "llama");
    g.meta_u32("llama.block_count", n_layers as u32);
    g.meta_u32("llama.embedding_length", n_embd as u32);
    g.meta_u32("llama.feed_forward_length", n_ff as u32);
    g.meta_u32("llama.attention.head_count", n_head as u32);
    g.meta_u32("llama.attention.head_count_kv", n_head_kv as u32);
    g.meta_u32("llama.attention.key_length", head_dim as u32);
    g.meta_u32("llama.rope.dimension_count", head_dim as u32);
    g.meta_u32("llama.context_length", 256);
    g.meta_f32("llama.rope.freq_base", 10000.0);
    g.meta_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);

    let mut rng = Rng(0x1234_5678 ^ (head_dim as u64));
    let inv_e = 0.25 / (n_embd as f32).sqrt();
    let inv_f = 0.25 / (n_ff as f32).sqrt();
    g.add_f32(
        "token_embd.weight",
        &[n_embd, vocab],
        &rng.vec(n_embd * vocab, 0.3),
    );
    g.add_f32("output_norm.weight", &[n_embd], &vec![1.0; n_embd]);
    g.add_f32(
        "output.weight",
        &[n_embd, vocab],
        &rng.vec(n_embd * vocab, inv_e),
    );

    for l in 0..n_layers {
        let p = format!("blk.{l}");
        g.add_f32(
            &format!("{p}.attn_norm.weight"),
            &[n_embd],
            &vec![1.0; n_embd],
        );
        g.add_f32(
            &format!("{p}.attn_q.weight"),
            &[n_embd, n_embd],
            &rng.vec(n_embd * n_embd, inv_e),
        );
        g.add_f32(
            &format!("{p}.attn_k.weight"),
            &[n_embd, kv_dim],
            &rng.vec(n_embd * kv_dim, inv_e),
        );
        g.add_f32(
            &format!("{p}.attn_v.weight"),
            &[n_embd, kv_dim],
            &rng.vec(n_embd * kv_dim, inv_e),
        );
        g.add_f32(
            &format!("{p}.attn_output.weight"),
            &[n_embd, n_embd],
            &rng.vec(n_embd * n_embd, inv_e),
        );
        g.add_f32(
            &format!("{p}.ffn_norm.weight"),
            &[n_embd],
            &vec![1.0; n_embd],
        );
        g.add_f32(
            &format!("{p}.ffn_gate.weight"),
            &[n_embd, n_ff],
            &rng.vec(n_embd * n_ff, inv_e),
        );
        g.add_f32(
            &format!("{p}.ffn_up.weight"),
            &[n_embd, n_ff],
            &rng.vec(n_embd * n_ff, inv_e),
        );
        g.add_f32(
            &format!("{p}.ffn_down.weight"),
            &[n_ff, n_embd],
            &rng.vec(n_ff * n_embd, inv_f),
        );
    }
    g.build()
}

fn build_qwen35_gguf() -> Vec<u8> {
    let n_layers = 2usize; // layer 0 = GDN, layer 1 = FullAttn (interval=2)
    let n_embd = 64usize;
    let n_ff = 128usize;
    let n_head = 2usize;
    let n_head_kv = 1usize;
    let head_dim = 64usize;
    let rope_dim = 32usize; // partial NeoX RoPE
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let ssm_k = 4usize;
    let ssm_s = 16usize;
    let ssm_g = 2usize;
    let ssm_dt = 4usize;
    let ssm_inner = ssm_dt * ssm_s;
    let conv_dim = 2 * ssm_g * ssm_s + ssm_inner;
    let vocab = 64usize;

    let mut g = SynthGguf::new();
    g.meta_str("general.architecture", "qwen35");
    g.meta_u32("qwen35.block_count", n_layers as u32);
    g.meta_u32("qwen35.embedding_length", n_embd as u32);
    g.meta_u32("qwen35.feed_forward_length", n_ff as u32);
    g.meta_u32("qwen35.attention.head_count", n_head as u32);
    g.meta_u32("qwen35.attention.head_count_kv", n_head_kv as u32);
    g.meta_u32("qwen35.attention.key_length", head_dim as u32);
    g.meta_u32("qwen35.rope.dimension_count", rope_dim as u32);
    g.meta_u32("qwen35.context_length", 256);
    g.meta_f32("qwen35.rope.freq_base", 10_000_000.0);
    g.meta_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6);
    g.meta_u32("qwen35.full_attention_interval", 2);
    g.meta_u32("qwen35.ssm.conv_kernel", ssm_k as u32);
    g.meta_u32("qwen35.ssm.state_size", ssm_s as u32);
    g.meta_u32("qwen35.ssm.group_count", ssm_g as u32);
    g.meta_u32("qwen35.ssm.time_step_rank", ssm_dt as u32);
    g.meta_u32("qwen35.ssm.inner_size", ssm_inner as u32);

    let mut rng = Rng(0xDEAD_BEEF);
    let inv_e = 0.25 / (n_embd as f32).sqrt();
    let inv_f = 0.25 / (n_ff as f32).sqrt();
    g.add_f32(
        "token_embd.weight",
        &[n_embd, vocab],
        &rng.vec(n_embd * vocab, 0.3),
    );
    g.add_f32("output_norm.weight", &[n_embd], &vec![1.0; n_embd]);
    g.add_f32(
        "output.weight",
        &[n_embd, vocab],
        &rng.vec(n_embd * vocab, inv_e),
    );

    // Layer 0: GDN
    g.add_f32("blk.0.attn_norm.weight", &[n_embd], &vec![1.0; n_embd]);
    g.add_f32(
        "blk.0.post_attention_norm.weight",
        &[n_embd],
        &vec![1.0; n_embd],
    );
    g.add_f32(
        "blk.0.attn_qkv.weight",
        &[n_embd, conv_dim],
        &rng.vec(n_embd * conv_dim, inv_e),
    );
    g.add_f32(
        "blk.0.attn_gate.weight",
        &[n_embd, ssm_inner],
        &rng.vec(n_embd * ssm_inner, inv_e),
    );
    g.add_f32(
        "blk.0.ssm_alpha.weight",
        &[n_embd, ssm_dt],
        &rng.vec(n_embd * ssm_dt, inv_e),
    );
    g.add_f32(
        "blk.0.ssm_beta.weight",
        &[n_embd, ssm_dt],
        &rng.vec(n_embd * ssm_dt, inv_e),
    );
    g.add_f32(
        "blk.0.ssm_conv1d.weight",
        &[ssm_k, conv_dim],
        &rng.vec(ssm_k * conv_dim, 0.2),
    );
    g.add_f32("blk.0.ssm_a", &[ssm_dt], &vec![-0.5; ssm_dt]);
    g.add_f32("blk.0.ssm_dt.bias", &[ssm_dt], &vec![0.1; ssm_dt]);
    g.add_f32("blk.0.ssm_norm.weight", &[ssm_s], &vec![1.0; ssm_s]);
    g.add_f32(
        "blk.0.ssm_out.weight",
        &[ssm_inner, n_embd],
        &rng.vec(ssm_inner * n_embd, inv_e),
    );
    g.add_f32(
        "blk.0.ffn_gate.weight",
        &[n_embd, n_ff],
        &rng.vec(n_embd * n_ff, inv_e),
    );
    g.add_f32(
        "blk.0.ffn_up.weight",
        &[n_embd, n_ff],
        &rng.vec(n_embd * n_ff, inv_e),
    );
    g.add_f32(
        "blk.0.ffn_down.weight",
        &[n_ff, n_embd],
        &rng.vec(n_ff * n_embd, inv_f),
    );

    // Layer 1: FullAttn
    g.add_f32("blk.1.attn_norm.weight", &[n_embd], &vec![1.0; n_embd]);
    g.add_f32(
        "blk.1.post_attention_norm.weight",
        &[n_embd],
        &vec![1.0; n_embd],
    );
    g.add_f32(
        "blk.1.attn_q.weight",
        &[n_embd, q_dim * 2],
        &rng.vec(n_embd * q_dim * 2, inv_e),
    );
    g.add_f32(
        "blk.1.attn_k.weight",
        &[n_embd, kv_dim],
        &rng.vec(n_embd * kv_dim, inv_e),
    );
    g.add_f32(
        "blk.1.attn_v.weight",
        &[n_embd, kv_dim],
        &rng.vec(n_embd * kv_dim, inv_e),
    );
    g.add_f32(
        "blk.1.attn_q_norm.weight",
        &[head_dim],
        &vec![1.0; head_dim],
    );
    g.add_f32(
        "blk.1.attn_k_norm.weight",
        &[head_dim],
        &vec![1.0; head_dim],
    );
    g.add_f32(
        "blk.1.attn_output.weight",
        &[q_dim, n_embd],
        &rng.vec(q_dim * n_embd, inv_e),
    );
    g.add_f32(
        "blk.1.ffn_gate.weight",
        &[n_embd, n_ff],
        &rng.vec(n_embd * n_ff, inv_e),
    );
    g.add_f32(
        "blk.1.ffn_up.weight",
        &[n_embd, n_ff],
        &rng.vec(n_embd * n_ff, inv_e),
    );
    g.add_f32(
        "blk.1.ffn_down.weight",
        &[n_ff, n_embd],
        &rng.vec(n_ff * n_embd, inv_f),
    );
    g.build()
}

fn kl_div(p_logits: &[f32], q_logits: &[f32]) -> (f64, f32) {
    let mp = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mq = q_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sp: f64 = p_logits.iter().map(|&x| ((x - mp) as f64).exp()).sum();
    let sq: f64 = q_logits.iter().map(|&x| ((x - mq) as f64).exp()).sum();
    let mut kl = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&lp, &lq) in p_logits.iter().zip(q_logits.iter()) {
        max_abs = max_abs.max((lp - lq).abs());
        let p = ((lp - mp) as f64).exp() / sp;
        let q = (((lq - mq) as f64).exp() / sq).max(1e-30);
        if p > 1e-30 {
            kl += p * (p / q).ln();
        }
    }
    (kl.max(0.0), max_abs)
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[test]
fn vk_kv_q8_0_matches_f16_across_gqa_splitk_prefill_and_qwen35() {
    let Ok(_ctx) = VulkanContext::new() else {
        return;
    };

    let toks: Vec<u32> = (0..24).map(|i| ((i * 7 + 3) % 64) as u32).collect();

    // 1. Llama: head_dim=64 (standard gqa_attn_q8) and head_dim=128 (Split-K gqa_attn_splitk_q8)
    for head_dim in [64usize, 128usize] {
        let gguf = open_bytes_as_gguf(&format!("llama_hd{head_dim}"), &build_llama_gguf(head_dim));
        let mut r_f16 =
            LlamaVkRunner::from_gguf_with_kv(gguf.clone(), 128, KvType::F16).expect("f16 runner");
        let mut r_q8 =
            LlamaVkRunner::from_gguf_with_kv(gguf, 128, KvType::Q8_0).expect("q8_0 runner");

        let mut f16_last = Vec::new();
        let mut q8_last = Vec::new();
        for (pos, &tok) in toks.iter().enumerate() {
            let l_f16 = r_f16
                .forward_token(tok, pos, true)
                .expect("f16 dec")
                .to_vec();
            let l_q8 = r_q8.forward_token(tok, pos, true).expect("q8 dec").to_vec();
            let (kl, max_abs) = kl_div(&l_f16, &l_q8);
            assert!(
                kl < 5e-4,
                "llama hd={head_dim} step={pos}: Q8_0 vs F16 kl={kl:.3e} >= 5e-4 (max_abs={max_abs:.3e})"
            );
            if pos + 1 == toks.len() {
                f16_last = l_f16;
                q8_last = l_q8;
            }
        }

        // Batched prefill with Q8_0 KV cache must match Q8_0 decode almost bit-exact (F32 weights)
        let q8_batch = r_q8
            .forward_batch(&toks, 0, true)
            .expect("q8 batch")
            .to_vec();
        let (kl_b, max_b) = kl_div(&q8_last, &q8_batch);
        assert_eq!(
            argmax(&q8_last),
            argmax(&q8_batch),
            "llama hd={head_dim} Q8_0 batch vs decode top1 mismatch"
        );
        assert!(
            max_b < 2e-3 && kl_b < 1e-6,
            "llama hd={head_dim} Q8_0 batch vs decode mismatch: kl={kl_b:.3e}, max_abs={max_b:.3e}"
        );

        // Greedy parity
        let greedy = r_q8
            .forward_token_greedy(*toks.last().unwrap(), toks.len() - 1)
            .expect("q8 greedy");
        assert_eq!(greedy, argmax(&q8_last));
        let (kl_f, max_f) = kl_div(&f16_last, &q8_last);
        eprintln!(
            "[parity_vk_kv_q8] llama hd={head_dim}: PASS (q8_vs_f16 kl={kl_f:.3e}, max_abs={max_f:.3e}; batch_vs_dec max_abs={max_b:.3e})"
        );
    }

    // 2. Qwen3.5 hybrid GDN + FullAttn with Q8_0 KV cache
    let q35_gguf = open_bytes_as_gguf("qwen35_kv_q8", &build_qwen35_gguf());
    let mut q35_f16 = Qwen35VkRunner::from_gguf_with_kv(q35_gguf.clone(), 128, KvType::F16)
        .expect("q35 f16 runner");
    let mut q35_q8 =
        Qwen35VkRunner::from_gguf_with_kv(q35_gguf, 128, KvType::Q8_0).expect("q35 q8 runner");

    let mut q35_f16_last = Vec::new();
    let mut q35_q8_last = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        let l_f16 = q35_f16
            .forward_token(tok, pos, true)
            .expect("q35 f16")
            .to_vec();
        let l_q8 = q35_q8
            .forward_token(tok, pos, true)
            .expect("q35 q8")
            .to_vec();
        let (kl, max_abs) = kl_div(&l_f16, &l_q8);
        assert!(
            kl < 5e-4,
            "qwen35 step={pos}: Q8_0 vs F16 kl={kl:.3e} >= 5e-4 (max_abs={max_abs:.3e})"
        );
        if pos + 1 == toks.len() {
            q35_f16_last = l_f16;
            q35_q8_last = l_q8;
        }
    }

    q35_q8.reset_state().expect("q35 reset");
    let q35_batch = q35_q8
        .forward_batch(&toks, 0, true)
        .expect("q35 q8 batch")
        .to_vec();
    let (kl_b, max_b) = kl_div(&q35_q8_last, &q35_batch);
    assert_eq!(argmax(&q35_q8_last), argmax(&q35_batch));
    assert!(
        max_b < 2e-3 && kl_b < 1e-6,
        "qwen35 Q8_0 batch vs decode mismatch: kl={kl_b:.3e}, max_abs={max_b:.3e}"
    );
    let (kl_f, max_f) = kl_div(&q35_f16_last, &q35_q8_last);
    eprintln!(
        "[parity_vk_kv_q8] qwen35: PASS (q8_vs_f16 kl={kl_f:.3e}, max_abs={max_f:.3e}; batch_vs_dec max_abs={max_b:.3e})"
    );
}
