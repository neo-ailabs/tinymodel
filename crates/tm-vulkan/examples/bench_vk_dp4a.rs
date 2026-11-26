//! Hardware-timestamp micro-benchmark (`AGENTS.md` §5 Rule #3) comparing Vulkan decode
//! GEMV + `RMSNorm` stage throughput (`GB/s` & `us/tok`) with `TM_VK_DP4A=0` (FP32 unpack)
//! vs `TM_VK_DP4A=1` (fused `rms_norm_q8_1` + `OpSDotKHR` `sdot4x8` `dp4a`).
//!
//! Run: `cargo run --release -p tm-vulkan --example bench_vk_dp4a`

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tm_core::{BlockQ4_0, BlockQ8_0, f32_to_f16};
use tm_gguf::GgufFile;
use tm_vulkan::{LlamaVkRunner, VulkanContext};

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

fn build_bench_gguf(n_embd: usize, n_ff: usize, vocab: usize) -> Arc<GgufFile> {
    let n_layers = 4usize;
    let n_head = n_embd / 128;
    let n_head_kv = (n_head / 4).max(1);
    let head_dim = 128usize;
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;

    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_u8 = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s & 0xFF) as u8
    };
    let q4 = |rows: usize, cols: usize, next: &mut dyn FnMut() -> u8| -> Vec<u8> {
        let nb = rows * cols / 32;
        let mut v = Vec::with_capacity(nb * 18);
        let d = f32_to_f16(0.02).to_le_bytes();
        for _ in 0..nb {
            v.extend_from_slice(&d);
            for _ in 0..16 {
                v.push(next());
            }
        }
        let _ = core::mem::size_of::<BlockQ4_0>();
        v
    };
    let q8 = |rows: usize, cols: usize, next: &mut dyn FnMut() -> u8| -> Vec<u8> {
        let nb = rows * cols / 32;
        let mut v = Vec::with_capacity(nb * 34);
        let d = f32_to_f16(0.01).to_le_bytes();
        for _ in 0..nb {
            v.extend_from_slice(&d);
            for _ in 0..32 {
                v.push(next());
            }
        }
        let _ = core::mem::size_of::<BlockQ8_0>();
        v
    };
    let f32_ones = |n: usize| -> Vec<u8> {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.extend_from_slice(&1.0f32.to_le_bytes());
        }
        v
    };

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

    let mut ts: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![
        (
            "token_embd.weight".into(),
            vec![n_embd as u64, vocab as u64],
            8,
            q8(vocab, n_embd, &mut next_u8),
        ),
        (
            "output.weight".into(),
            vec![n_embd as u64, vocab as u64],
            2,
            q4(vocab, n_embd, &mut next_u8),
        ),
        (
            "output_norm.weight".into(),
            vec![n_embd as u64],
            0,
            f32_ones(n_embd),
        ),
    ];
    for l in 0..n_layers {
        let p = format!("blk.{l}");
        ts.push((
            format!("{p}.attn_norm.weight"),
            vec![n_embd as u64],
            0,
            f32_ones(n_embd),
        ));
        ts.push((
            format!("{p}.ffn_norm.weight"),
            vec![n_embd as u64],
            0,
            f32_ones(n_embd),
        ));
        ts.push((
            format!("{p}.attn_q_norm.weight"),
            vec![head_dim as u64],
            0,
            f32_ones(head_dim),
        ));
        ts.push((
            format!("{p}.attn_k_norm.weight"),
            vec![head_dim as u64],
            0,
            f32_ones(head_dim),
        ));
        ts.push((
            format!("{p}.attn_q.weight"),
            vec![n_embd as u64, q_dim as u64],
            2,
            q4(q_dim, n_embd, &mut next_u8),
        ));
        ts.push((
            format!("{p}.attn_k.weight"),
            vec![n_embd as u64, kv_dim as u64],
            2,
            q4(kv_dim, n_embd, &mut next_u8),
        ));
        ts.push((
            format!("{p}.attn_v.weight"),
            vec![n_embd as u64, kv_dim as u64],
            2,
            q4(kv_dim, n_embd, &mut next_u8),
        ));
        ts.push((
            format!("{p}.attn_output.weight"),
            vec![q_dim as u64, n_embd as u64],
            8,
            q8(n_embd, q_dim, &mut next_u8),
        ));
        ts.push((
            format!("{p}.ffn_gate.weight"),
            vec![n_embd as u64, n_ff as u64],
            2,
            q4(n_ff, n_embd, &mut next_u8),
        ));
        ts.push((
            format!("{p}.ffn_up.weight"),
            vec![n_embd as u64, n_ff as u64],
            2,
            q4(n_ff, n_embd, &mut next_u8),
        ));
        ts.push((
            format!("{p}.ffn_down.weight"),
            vec![n_ff as u64, n_embd as u64],
            2,
            q4(n_embd, n_ff, &mut next_u8),
        ));
    }

    let mut tinfos = Vec::new();
    let mut off = 0u64;
    for (name, dims, dt, data) in &ts {
        push_str(&mut tinfos, name);
        tinfos.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            tinfos.extend_from_slice(&d.to_le_bytes());
        }
        tinfos.extend_from_slice(&dt.to_le_bytes());
        tinfos.extend_from_slice(&off.to_le_bytes());
        off += (data.len() as u64).div_ceil(32) * 32;
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(ts.len() as u64).to_le_bytes());
    out.extend_from_slice(&12u64.to_le_bytes());
    out.extend_from_slice(&kv);
    out.extend_from_slice(&tinfos);
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for (_, _, _, data) in &ts {
        out.extend_from_slice(data);
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("tm_bench_dp4a_{stamp}.gguf"));
    std::fs::write(&path, &out).unwrap();
    let f = GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    Arc::new(f)
}

fn main() {
    let ctx = VulkanContext::new().expect("failed to initialize Vulkan");
    let info = ctx.info().clone();
    println!(
        "Vulkan GPU: {} [{}] int8_dot_accel={}",
        info.name, info.driver_name, info.int8_dot_accelerated
    );
    drop(ctx);

    let gguf = build_bench_gguf(2048, 6144, 16384);
    for (label, flag) in [
        ("FP32-unpack (TM_VK_DP4A=0)", "0"),
        ("DP4A+FusedNorm (TM_VK_DP4A=1)", "1"),
    ] {
        // SAFETY: single-threaded benchmark binary.
        unsafe { std::env::set_var("TM_VK_DP4A", flag) };
        let mut runner = LlamaVkRunner::from_gguf(gguf.clone(), 128).unwrap();
        let rep = runner.profile(40, 32).unwrap();
        println!(
            "\n=== {label}: 1-CB decode wall={:.1} us/tok (gpu={:.1} us/tok) ===",
            rep.decode_single_cb_wall_us, rep.decode_single_cb_gpu_us
        );
        for st in &rep.decode_stages {
            if st.bytes > 0 && st.gpu_us > 0.0 {
                let gbs = (st.bytes as f64 / 1e9) / (st.gpu_us / 1e6);
                println!("  {:<18} {:>8.1} us  {:>7.1} GB/s", st.name, st.gpu_us, gbs);
            }
        }
    }
}
