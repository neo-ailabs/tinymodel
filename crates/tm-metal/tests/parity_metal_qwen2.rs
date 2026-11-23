//! Apple Metal GPU multi-token parity tests for `qwen2` (Qwen2 & Qwen2.5) against
//! `LlamaModel` CPU and `tools/oracle/qwen2_oracle.py` (`AGENTS.md` Rules #3, #7, #9).

#![cfg(target_vendor = "apple")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_metal::MetalRunner;
use tm_models::{LlamaModel, LlamaState, argmax};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_mtl_qwen2_{tag}_{}_{id}", std::process::id()));
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
fn metal_qwen2_multi_token_decode_batch_and_greedy_parity() {
    let dir = tmp_dir("parity");
    let py = find_python3_with_numpy();
    let oracle_script = repo_root().join("tools/oracle/qwen2_oracle.py");

    for (tag, quant, extra_args, kl_tol, diff_tol, tie_tol) in [
        (
            "f32_tied_all_bias",
            "f32",
            vec![],
            1e-7f64,
            5e-4f32,
            1e-3f32,
        ),
        (
            "q4_0_untied_qkv_bias",
            "q4_0",
            vec!["--untied", "--no-wo-bias", "--no-output-bias"],
            5e-3f64,
            1.5e-1f32,
            8e-2f32,
        ),
        (
            "q4_0_tied_full_bias",
            "q4_0",
            vec![],
            5e-3f64,
            1.5e-1f32,
            8e-2f32,
        ),
    ] {
        let gguf_path = dir.join(format!("{tag}.gguf"));
        let mut cmd = Command::new(&py);
        cmd.arg(&oracle_script)
            .arg("--out-gguf")
            .arg(&gguf_path)
            .arg("--quant")
            .arg(quant);
        for a in &extra_args {
            cmd.arg(a);
        }
        assert!(cmd.status().expect("run oracle").success());

        let gguf = Arc::new(GgufFile::open(&gguf_path).expect("open gguf"));
        let cpu = LlamaModel::from_gguf(&gguf).expect("cpu qwen2");
        let mut cpu_st = LlamaState::new(&cpu.cfg, 64);
        let pool = ThreadPool::new(2);
        let kt = active_kernels();

        let mut mtl_dec = MetalRunner::from_gguf(gguf.clone(), 64).expect("MetalRunner(qwen2)");
        assert_eq!(mtl_dec.arch(), "qwen2");

        let seq = [1u32, 7, 14, 23, 5, 31, 9, 18];
        for (pos, &tok) in seq.iter().enumerate() {
            cpu.forward_token(&mut cpu_st, &pool, kt, tok, pos, true)
                .expect("cpu forward_token");
            let mtl_l = mtl_dec
                .forward_token(tok, pos, true)
                .expect("metal forward_token");
            let (kl, max_d) = kl_div(cpu_st.logits(), mtl_l);
            let c_top = argmax(cpu_st.logits());
            let m_top = argmax(mtl_l);
            assert!(
                c_top == m_top
                    || (cpu_st.logits()[c_top as usize] - cpu_st.logits()[m_top as usize]).abs()
                        <= tie_tol,
                "Metal qwen2 {tag} step {pos} Top-1 mismatch: cpu={c_top} mtl={m_top}"
            );
            assert!(
                kl <= kl_tol && max_d <= diff_tol,
                "Metal qwen2 {tag} step {pos}: kl={kl:.3e} max_d={max_d:.3e}"
            );
        }

        // Greedy kernel (`forward_token_greedy`) must pick the exact argmax of Metal's own logits (including `output.bias`)!
        let greedy_tok = mtl_dec
            .forward_token_greedy(11, seq.len())
            .expect("metal greedy");
        let mut mtl_ref = MetalRunner::from_gguf(gguf.clone(), 64).expect("MetalRunner greedy ref");
        for (pos, &tok) in seq.iter().enumerate() {
            let _ = mtl_ref.forward_token(tok, pos, false).expect("prefill");
        }
        let mtl_step_l = mtl_ref
            .forward_token(11, seq.len(), true)
            .expect("ref logits");
        assert_eq!(greedy_tok, argmax(mtl_step_l));

        // Batched prefill (> 32 tile boundary) vs CPU batched prefill
        let batch_toks: Vec<u32> = (0..37).map(|i| ((i * 5 + 3) % 320) as u32).collect();
        let mut cpu_b = LlamaState::new(&cpu.cfg, 64);
        cpu.forward_batch(&mut cpu_b, &pool, kt, &batch_toks, 0, true)
            .expect("cpu batch");
        let mut mtl_bat = MetalRunner::from_gguf(gguf.clone(), 64).expect("MetalRunner batch");
        let mtl_bl = mtl_bat
            .forward_batch(&batch_toks, 0, true)
            .expect("metal batch");
        let (kl_b, max_db) = kl_div(cpu_b.logits(), mtl_bl);
        let cb_top = argmax(cpu_b.logits());
        let mb_top = argmax(mtl_bl);
        assert!(
            cb_top == mb_top
                || (cpu_b.logits()[cb_top as usize] - cpu_b.logits()[mb_top as usize]).abs()
                    <= tie_tol
        );
        assert!(
            kl_b <= kl_tol && max_db <= diff_tol,
            "Metal qwen2 {tag} batch(37): kl={kl_b:.3e} max_d={max_db:.3e}"
        );
    }

    let _ = fs::remove_dir_all(&dir);
}
