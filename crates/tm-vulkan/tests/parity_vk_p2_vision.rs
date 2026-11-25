//! Vulkan GPU parity tests for P2 Multimodal `forward_embed` (`ADR-019`).
//!
//! Verifies that `LlamaVkRunner::forward_embed` produces logits matching `forward_token`
//! and CPU `LlamaModel::forward_embed` on Vulkan GPUs.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::LlamaModel;
use tm_vulkan::LlamaVkRunner;

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
        "python3",
    ] {
        if cand.is_empty() {
            continue;
        }
        if Command::new(cand)
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

#[test]
fn parity_vk_p2_forward_embed_matches_forward_token_and_cpu() {
    let out_dir = std::env::temp_dir().join("tm_p2_vk_embed_parity");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let gguf_path = out_dir.join("exaone_vk_mm.gguf");
    let ref_path = out_dir.join("exaone_vk_mm_ref.bin");
    let py = find_python3_with_numpy();
    let status = Command::new(&py)
        .arg(&script)
        .arg("--arch")
        .arg("exaone")
        .arg("--out-gguf")
        .arg(&gguf_path)
        .arg("--out-logits")
        .arg(&ref_path)
        .status()
        .expect("run p2a oracle");
    assert!(status.success());

    let gguf = Arc::new(GgufFile::open(&gguf_path).unwrap());
    let cpu_model = LlamaModel::from_gguf(&gguf).unwrap();
    let d = cpu_model.cfg.n_embd;

    // 1. Vulkan token-only sequence [1, 2, 3]
    let mut runner_tok = LlamaVkRunner::from_gguf(gguf.clone(), 32).unwrap();
    runner_tok.forward_token(1, 0, false).unwrap();
    runner_tok.forward_token(2, 1, false).unwrap();
    let logits_tok = runner_tok.forward_token(3, 2, true).unwrap().to_vec();

    // 2. Vulkan multimodal sequence: token 1, then precomputed embed(2) via forward_embed, then token 3
    let mut emb2 = vec![0.0f32; d];
    cpu_model.embed_token(2, &mut emb2);
    let mut runner_mm = LlamaVkRunner::from_gguf(gguf.clone(), 32).unwrap();
    runner_mm.forward_token(1, 0, false).unwrap();
    runner_mm.forward_embed(&emb2, 1, false).unwrap();
    let logits_mm = runner_mm.forward_token(3, 2, true).unwrap().to_vec();

    let mut max_abs = 0.0f32;
    for (&a, &b) in logits_tok.iter().zip(logits_mm.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-5,
        "Vulkan forward_embed vs forward_token max_abs = {max_abs:.6e}"
    );

    // 3. Also verify against CPU forward_embed
    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let mut cpu_st = tm_models::LlamaState::new(&cpu_model.cfg, 32);
    cpu_model
        .forward_token(&mut cpu_st, &pool, kt, 1, 0, false)
        .unwrap();
    cpu_model
        .forward_embed(&mut cpu_st, &pool, kt, &emb2, 1, false)
        .unwrap();
    cpu_model
        .forward_token(&mut cpu_st, &pool, kt, 3, 2, true)
        .unwrap();

    let mut max_cpu_gpu = 0.0f32;
    for (&c, &g) in cpu_st.logits().iter().zip(logits_mm.iter()) {
        max_cpu_gpu = max_cpu_gpu.max((c - g).abs());
    }
    // Vulkan stores KV cache in FP16 while CPU uses F32, so ~1e-3 max_abs is expected.
    assert!(
        max_cpu_gpu < 2e-3,
        "Vulkan forward_embed vs CPU forward_embed max_abs = {max_cpu_gpu:.6e}"
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}
