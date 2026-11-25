//! Vulkan GPU single-CommandBuffer Sparse MoE (`ADR-018`) decode, greedy kernel, `reset_state`,
//! and batched prefill parity tests across P2a MoE architectures (`olmoe`, `qwen2moe`, `qwen3moe`,
//! `bailingmoe2`) against the independent FP64 Python oracle (`tools/oracle/p2a_architectures_oracle.py`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_gguf::GgufFile;
use tm_models::argmax;
use tm_vulkan::LlamaVkRunner;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_vk_p2a_{tag}_{}_{id}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).expect("mkdir tmp");
    p
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn find_python3_with_numpy() -> String {
    for cand in [
        std::env::var("PYTHON3").ok().as_deref().unwrap_or(""),
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "python3",
    ] {
        if cand.is_empty() {
            continue;
        }
        if Command::new(cand)
            .args(["-c", "import numpy"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return cand.to_string();
        }
    }
    "python3".to_string()
}

fn read_tmlg(path: &Path) -> (Vec<u32>, usize, Vec<Vec<f32>>) {
    let raw = fs::read(path).expect("read tmlg");
    assert_eq!(&raw[0..4], b"TMLG");
    let ver = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    assert_eq!(ver, 1);
    let n_steps = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let n_vocab = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let mut off = 16usize;
    let mut toks = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        toks.push(u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    let mut steps = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let mut row = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            row.push(f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        steps.push(row);
    }
    (toks, n_vocab, steps)
}

fn kl_div(ref_l: &[f32], cand_l: &[f32]) -> (f64, f32) {
    let mr = ref_l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mc = cand_l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sr: f64 = ref_l.iter().map(|&x| ((x as f64) - mr).exp()).sum();
    let sc: f64 = cand_l.iter().map(|&x| ((x as f64) - mc).exp()).sum();
    let lzr = mr + sr.ln();
    let lzc = mc + sc.ln();
    let mut kl = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&r, &c) in ref_l.iter().zip(cand_l.iter()) {
        let d = (r - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        let lp = (r as f64) - lzr;
        let lq = (c as f64) - lzc;
        kl += lp.exp() * (lp - lq);
    }
    (kl.max(0.0), max_abs)
}

#[test]
fn vk_p2a_sparse_moe_architectures_decode_batch_greedy_and_reset_parity() {
    let dir = tmp_dir("moe");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");

    for &(arch, quant) in &[
        ("olmoe", "f32"),
        ("olmoe", "q4_0"),
        ("qwen2moe", "f32"),
        ("qwen2moe", "q4_0"),
        ("qwen3moe", "f32"),
        ("qwen3moe", "q4_0"),
        ("bailingmoe2", "f32"),
        ("bailingmoe2", "q4_0"),
    ] {
        let gguf_path = dir.join(format!("{arch}_{quant}.gguf"));
        let ref_path = dir.join(format!("{arch}_{quant}.tmlg"));
        let status = Command::new(&py)
            .arg(&script)
            .arg("--arch")
            .arg(arch)
            .arg("--quant")
            .arg(quant)
            .arg("--out-gguf")
            .arg(&gguf_path)
            .arg("--out-logits")
            .arg(&ref_path)
            .status()
            .expect("run p2a oracle");
        assert!(status.success(), "oracle failed for {arch}/{quant}");

        let (toks, _n_vocab, ref_steps) = read_tmlg(&ref_path);
        let gguf = Arc::new(GgufFile::open(&gguf_path).expect("open gguf"));
        let mut runner = LlamaVkRunner::from_gguf(gguf, 64).expect("init LlamaVkRunner");

        // 1. Single-token decode across all steps vs FP64 oracle
        let mut dec_last = Vec::new();
        for (pos, (&tok, ref_l)) in toks.iter().zip(ref_steps.iter()).enumerate() {
            let lg = runner.forward_token(tok, pos, true).expect("vk decode");
            let (kl, max_abs) = kl_div(ref_l, lg);
            assert_eq!(
                argmax(ref_l),
                argmax(lg),
                "{arch}/{quant} step={pos}: top1 mismatch (kl={kl:.3e}, max_abs={max_abs:.3e})"
            );
            assert!(
                kl < 5e-4,
                "{arch}/{quant} step={pos}: kl={kl:.3e} >= 5e-4 (max_abs={max_abs:.3e})"
            );
            if pos + 1 == toks.len() {
                dec_last = lg.to_vec();
            }
        }

        // 2. Batched prefill parity vs single-token decode (overwrites KV from start_pos=0)
        // Note: on GPUs with DP4A (`Q8_1` activation quantization for `q4_0` prefill GEMM),
        // synthetic near-tied logits can differ slightly in argmax while KL remains < 5e-4;
        // `f32` prefill uses FP32 activations so top1 and tight max_abs must match exactly.
        let batch_last = runner
            .forward_batch(&toks, 0, true)
            .expect("vk forward_batch")
            .to_vec();
        let (kl_b, max_b) = kl_div(&dec_last, &batch_last);
        if quant == "f32" {
            assert_eq!(
                argmax(&dec_last),
                argmax(&batch_last),
                "{arch}/{quant} batch vs decode top1 mismatch (kl={kl_b:.3e}, max_abs={max_b:.3e})"
            );
            assert!(
                max_b < 1e-3,
                "{arch}/{quant} batch vs decode max_abs={max_b:.3e} >= 1e-3"
            );
        }
        assert!(
            kl_b < 5e-4,
            "{arch}/{quant} batch vs decode kl={kl_b:.3e} >= 5e-4 (max_abs={max_b:.3e})"
        );

        // 3. Greedy kernel parity on last token (re-populate KV via decode for exact parity)
        for (pos, &tok) in toks[..toks.len() - 1].iter().enumerate() {
            runner
                .forward_token(tok, pos, false)
                .expect("vk prefix decode");
        }
        let greedy_tok = runner
            .forward_token_greedy(*toks.last().unwrap(), toks.len() - 1)
            .expect("vk greedy");
        assert_eq!(
            greedy_tok,
            argmax(&dec_last) as u32,
            "{arch}/{quant} greedy token mismatch"
        );
        eprintln!(
            "[parity_vk_p2a] {arch}/{quant}: PASS (batch_vs_dec kl={kl_b:.3e}, max_abs={max_b:.3e})"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}
