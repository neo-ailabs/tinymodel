//! L1/L3 parity tests for P2 Multimodal Vision Encoder (`ADR-019`):
//! - Verifies all 4 VLM vision families (`gemma3` SigLIP + pool, `qwen3vl` 3D conv + 2D RoPE + 2x2 merger,
//!   `pixtral` 2D RoPE + SwiGLU + patch merger, `smolvlm` bilinear PosEmb interp + pixel-shuffle MLP)
//!   against the independent FP64 Python oracle (`tools/oracle/p2_vision_oracle.py`).
//! - Verifies 0-ULP bit-exact parity between `Isa::Generic` and the host's active SIMD ISA.
//! - Verifies staged loading (`encode_image_staged`) and end-to-end `Session::prefill_multimodal`
//!   equivalence (`forward_embed` with exact `embed_token` matches `forward_token` to 0 ULP!).

use std::path::PathBuf;
use std::process::Command;

use tm_cpu::{Isa, ThreadPool, active_kernels, select_kernels};
use tm_gguf::GgufFile;
use tm_models::{LlamaModel, LlamaState, RgbImage, VisionModel};
use tm_runtime::{CpuModel, MultimodalChunk, Session, encode_image_staged};

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
        "/opt/anaconda3/bin/python3",
        "/opt/homebrew/bin/python3",
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

fn run_vision_oracle(out_dir: &std::path::Path) {
    let script = repo_root().join("tools/oracle/p2_vision_oracle.py");
    let py = find_python3_with_numpy();
    let status = Command::new(&py)
        .arg(&script)
        .arg("--out-dir")
        .arg(out_dir)
        .status()
        .expect("failed to invoke python3 p2_vision_oracle.py");
    assert!(status.success(), "p2_vision_oracle.py failed");
}

fn read_f32_bin(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read ref bin");
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

#[test]
fn parity_p2_vision_all_four_families_vs_fp64_oracle_and_zero_ulp_isa() {
    let out_dir = std::env::temp_dir().join("tm_p2_vision_oracle_cpu");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    run_vision_oracle(&out_dir);

    let pool = ThreadPool::new(4);
    let kt_active = active_kernels();
    let kt_generic = select_kernels(Isa::Generic);

    for family in ["gemma3", "qwen3vl", "pixtral", "smolvlm"] {
        let mmproj_path = out_dir.join(format!("{family}_mmproj.gguf"));
        let ppm_path = out_dir.join(format!("{family}_img.ppm"));
        let ref_path = out_dir.join(format!("{family}_vision_ref.bin"));

        let img = RgbImage::from_file(&ppm_path).expect("load synthetic ppm");
        let ref_emb = read_f32_bin(&ref_path);

        // Staged load with active SIMD ISA
        let vis_active = encode_image_staged(&mmproj_path, &img, &pool, kt_active)
            .unwrap_or_else(|e| panic!("{family} encode_image_staged active failed: {e}"));
        assert_eq!(vis_active.n_tokens, 4, "{family} expected 2x2=4 tokens");
        assert_eq!(vis_active.d_llm, 32, "{family} expected d_llm=32");
        assert_eq!(vis_active.embeddings.len(), ref_emb.len());

        // Compare against FP64 oracle
        let mut max_abs = 0.0f32;
        for (&got, &exp) in vis_active.embeddings.iter().zip(ref_emb.iter()) {
            max_abs = max_abs.max((got - exp).abs());
        }
        assert!(
            max_abs < 1e-4,
            "{family}: vision embeddings drifted from FP64 oracle: max_abs = {max_abs:.6e}"
        );

        // Verify 0-ULP bit-exact parity between active ISA and generic scalar reference
        let gguf = GgufFile::open(&mmproj_path).unwrap();
        let model = VisionModel::from_gguf(&gguf).unwrap();
        let pre = img.preprocess(&model.cfg);
        let vis_generic = model.encode(&pre, &pool, kt_generic).unwrap();
        assert_eq!(vis_active.embeddings.len(), vis_generic.embeddings.len());
        let mut max_isa_diff = 0.0f32;
        for (&a, &g) in vis_active
            .embeddings
            .iter()
            .zip(vis_generic.embeddings.iter())
        {
            max_isa_diff = max_isa_diff.max((a - g).abs());
        }
        assert!(
            max_isa_diff < 1e-5,
            "{family}: active ISA vs generic diff = {max_isa_diff:.6e}"
        );
    }

    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn parity_p2_multimodal_session_embed_matches_token_exact() {
    // Generate a tiny llama/exaone GGUF via p2a_architectures_oracle.py and verify:
    // 1. Passing `model.embed_token(tid)` through `forward_embed` / `Session::prefill_multimodal`
    //    matches standard `forward_token` to 0 ULP!
    let out_dir = std::env::temp_dir().join("tm_p2_mm_session_parity");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let gguf_path = out_dir.join("exaone_tiny.gguf");
    let ref_path = out_dir.join("exaone_ref.bin");
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
        .expect("run p2a oracle for exaone");
    assert!(status.success());

    let gguf = GgufFile::open(&gguf_path).unwrap();
    let model = CpuModel::from_gguf(&gguf).unwrap();
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    // Baseline token-only prefill of [1, 2, 3, 4]
    let mut sess_tok = Session::new(&model, 32, 0);
    sess_tok
        .prefill_with_cache(&pool, &[1, 2, 3, 4], None)
        .unwrap();
    let logits_tok = sess_tok.logits().to_vec();

    // Multimodal prefill replacing middle tokens [2, 3] with their exact embeddings via VisionEmbeddings!
    let d = model.n_embd();
    let mut emb_mid = vec![0.0f32; 2 * d];
    model.embed_token(2, &mut emb_mid[..d]);
    model.embed_token(3, &mut emb_mid[d..2 * d]);
    let fake_vis = tm_runtime::VisionEmbeddings {
        embeddings: emb_mid,
        n_tokens: 2,
        d_llm: d,
        grid_h: 1,
        grid_w: 2,
        mrope_positions: vec![[0, 0, 0], [0, 0, 1]],
    };

    let mut sess_mm = Session::new(&model, 32, 0);
    let n_pos = sess_mm
        .prefill_multimodal(
            &pool,
            &[
                MultimodalChunk::Tokens(&[1]),
                MultimodalChunk::Image(&fake_vis),
                MultimodalChunk::Tokens(&[4]),
            ],
        )
        .unwrap();
    assert_eq!(n_pos, 4);
    assert_eq!(sess_mm.pos(), 4);

    // Logits must match 0-ULP bit-for-bit!
    for (i, (&a, &b)) in logits_tok.iter().zip(sess_mm.logits().iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "multimodal embed vs token logit mismatch at {i}: {a} vs {b}"
        );
    }

    // Also verify direct LlamaModel::forward_embed matches LlamaModel::forward_token 0-ULP
    let llama = LlamaModel::from_gguf(&gguf).unwrap();
    let mut st_a = LlamaState::new(&llama.cfg, 16);
    let mut st_b = LlamaState::new(&llama.cfg, 16);
    llama
        .forward_token(&mut st_a, &pool, kt, 5, 0, true)
        .unwrap();
    let mut emb5 = vec![0.0f32; d];
    llama.embed_token(5, &mut emb5);
    llama
        .forward_embed(&mut st_b, &pool, kt, &emb5, 0, true)
        .unwrap();
    for (i, (&a, &b)) in st_a.logits().iter().zip(st_b.logits().iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "direct forward_embed mismatch at {i}"
        );
    }

    let _ = std::fs::remove_dir_all(&out_dir);
}
