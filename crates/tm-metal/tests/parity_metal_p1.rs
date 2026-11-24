//! Apple Metal GPU multi-token decode, greedy kernel, batched prefill (`> 32` tile boundary),
//! `reset_state`, and dynamic GPU LoRA (`bind_lora`) parity tests across all P1 sub-8B
//! architectures (`olmo2`, `olmo3`, `granite`, `mamba2`, `nemotron_h`, `falcon-h1`,
//! `granitehybrid`) per `AGENTS.md` Rules #3, #7, #9.

#![cfg(target_vendor = "apple")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_lora::LoraSet;
use tm_metal::MetalRunner;
use tm_models::argmax;
use tm_runtime::CpuModel;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_mtl_p1_{tag}_{}_{id}", std::process::id()));
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
fn metal_p1_architectures_decode_greedy_batch_reset_and_lora_parity() {
    let dir = tmp_dir("parity");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p1_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    // (tag, arch, max_ctx, quant, kl_tol, diff_tol, tie_tol)
    let cases: &[(&str, &str, usize, &str, f64, f32, f32)] = &[
        ("olmo2_f32", "olmo2", 64, "f32", 5e-6, 3e-3, 3e-3),
        ("olmo3_q4", "olmo3", 64, "q4_0", 5e-3, 1.5e-1, 8e-2),
        ("granite_short_f32", "granite", 32, "f32", 5e-6, 3e-3, 3e-3),
        ("granite_long_q4", "granite", 64, "q4_0", 5e-3, 1.5e-1, 8e-2),
        ("mamba2_f32", "mamba2", 64, "f32", 5e-6, 3e-3, 3e-3),
        ("mamba2_q4", "mamba2", 64, "q4_0", 5e-3, 1.5e-1, 8e-2),
        ("nemotron_h_f32", "nemotron_h", 64, "f32", 5e-6, 3e-3, 3e-3),
        (
            "nemotron_h_q4",
            "nemotron_h",
            64,
            "q4_0",
            5e-3,
            1.5e-1,
            8e-2,
        ),
        ("falcon_h1_f32", "falcon-h1", 64, "f32", 5e-6, 3e-3, 3e-3),
        ("falcon_h1_q4", "falcon-h1", 64, "q4_0", 5e-3, 1.5e-1, 8e-2),
        (
            "granitehybrid_f32",
            "granitehybrid",
            64,
            "f32",
            5e-6,
            3e-3,
            3e-3,
        ),
        (
            "granitehybrid_q4",
            "granitehybrid",
            64,
            "q4_0",
            5e-3,
            1.5e-1,
            8e-2,
        ),
    ];

    let seq = [1u32, 7, 14, 23, 5, 31, 9, 18];

    for &(tag, arch, max_ctx, quant, kl_tol, diff_tol, tie_tol) in cases {
        let gguf_path = dir.join(format!("{tag}.gguf"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&gguf_path)
                .arg("--quant")
                .arg(quant)
                .arg("--eval-ctx")
                .arg(max_ctx.to_string())
                .status()
                .expect("run oracle")
                .success()
        );

        let gguf = Arc::new(GgufFile::open(&gguf_path).expect("open gguf"));
        let cpu = CpuModel::from_gguf(&gguf).expect("cpu model");
        let mut cpu_st = cpu.new_state(max_ctx);

        let mut mtl_dec = MetalRunner::from_gguf(gguf.clone(), max_ctx).expect("MetalRunner");
        assert_eq!(mtl_dec.arch(), arch);

        let mut first_pass_last = Vec::new();
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
                "Metal {tag} step {pos} Top-1 mismatch: cpu={c_top} mtl={m_top}"
            );
            assert!(
                kl <= kl_tol && max_d <= diff_tol,
                "Metal {tag} step {pos}: kl={kl:.3e} (tol {kl_tol:.1e}) max_d={max_d:.3e} (tol {diff_tol:.1e})"
            );
            if pos + 1 == seq.len() {
                first_pass_last = mtl_l.to_vec();
            }
        }

        // Greedy kernel must match argmax of Metal's own logits
        let greedy_tok = mtl_dec
            .forward_token_greedy(11, seq.len())
            .expect("metal greedy");
        let mut mtl_ref = MetalRunner::from_gguf(gguf.clone(), max_ctx).expect("MetalRunner ref");
        for (pos, &tok) in seq.iter().enumerate() {
            let _ = mtl_ref.forward_token(tok, pos, false).expect("prefill");
        }
        let mtl_step_l = mtl_ref
            .forward_token(11, seq.len(), true)
            .expect("ref logits");
        assert_eq!(
            greedy_tok,
            argmax(mtl_step_l),
            "Metal {tag} greedy mismatch"
        );

        // `reset_state()` must zero recurrent Mamba-2 `S` & `conv_hist` + KV cache cleanly!
        mtl_dec.reset_state();
        for (pos, &tok) in seq.iter().enumerate() {
            let mtl_l = mtl_dec
                .forward_token(tok, pos, pos + 1 == seq.len())
                .expect("post-reset forward_token");
            if pos + 1 == seq.len() {
                for (i, (&a, &b)) in first_pass_last.iter().zip(mtl_l.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "Metal {tag} reset_state drift at logit[{i}]"
                    );
                }
            }
        }

        // Batched prefill (`n_batch` crossing 32-token tile boundary when `max_ctx >= 64`)
        let n_batch = if max_ctx >= 64 { 37 } else { 21 };
        let batch_toks: Vec<u32> = (0..n_batch).map(|i| ((i * 5 + 3) % 120) as u32).collect();
        let mut cpu_b = cpu.new_state(max_ctx);
        cpu.forward_batch(&mut cpu_b, &pool, kt, &batch_toks, 0, true)
            .expect("cpu batch");
        let mut mtl_bat = MetalRunner::from_gguf(gguf.clone(), max_ctx).expect("MetalRunner batch");
        let mtl_bl = mtl_bat
            .forward_batch(&batch_toks, 0, true)
            .expect("metal batch");
        let (kl_b, max_db) = kl_div(cpu_b.logits(), mtl_bl);
        let cb_top = argmax(cpu_b.logits());
        let mb_top = argmax(mtl_bl);
        assert!(
            cb_top == mb_top
                || (cpu_b.logits()[cb_top as usize] - cpu_b.logits()[mb_top as usize]).abs()
                    <= tie_tol,
            "Metal {tag} batch({n_batch}) Top-1 mismatch: cpu={cb_top} mtl={mb_top}"
        );
        assert!(
            kl_b <= kl_tol && max_db <= diff_tol,
            "Metal {tag} batch({n_batch}): kl={kl_b:.3e} max_d={max_db:.3e}"
        );

        // Dynamic GPU LoRA (`bind_lora`) parity on `granitehybrid_f32` (`token_embd` + `ssm_in` + `attn_q` + `output`)!
        if tag == "granitehybrid_f32" {
            let mut pairs = std::collections::HashMap::new();
            let mut order = Vec::new();
            for (tname, is_emb, cols, rows) in [
                ("token_embd.weight", true, 128usize, 128usize),
                ("blk.0.ssm_in.weight", false, 128usize, 388usize),
                ("blk.1.attn_q.weight", false, 128usize, 128usize),
                ("output.weight", false, 128usize, 128usize),
            ] {
                order.push(tname.to_string());
                let (a_len, b_len) = if is_emb {
                    (rows * 4, cols * 4)
                } else {
                    (4 * cols, rows * 4)
                };
                pairs.insert(
                    tname.to_string(),
                    tm_lora::LoraWeightPair {
                        base_name: tname.to_string(),
                        is_token_embd: is_emb,
                        rank: 4,
                        d_in: cols,
                        d_out: rows,
                        a: (0..a_len)
                            .map(|i| ((i as f32) * 0.019).sin() * 0.15)
                            .collect(),
                        b: (0..b_len)
                            .map(|i| ((i as f32) * 0.023).cos() * 0.15)
                            .collect(),
                    },
                );
            }
            let adapter = tm_lora::LoraAdapter {
                arch: "granitehybrid".to_string(),
                alpha: 8.0,
                name: Some("granitehybrid-gpu-lora".to_string()),
                pairs,
                pair_order: order,
            };
            let mut set = LoraSet::new();
            set.push(adapter, 1.0);
            set.validate_against_gguf(&gguf).unwrap();

            let mut cpu_lora = CpuModel::from_gguf(&gguf).unwrap();
            cpu_lora.bind_lora(&gguf, &set).unwrap();
            let mut cpu_lst = cpu_lora.new_state(max_ctx);
            cpu_lora
                .forward_batch(&mut cpu_lst, &pool, kt, &seq, 0, true)
                .unwrap();

            let mut mtl_lora = MetalRunner::from_gguf(gguf.clone(), max_ctx).unwrap();
            mtl_lora.bind_lora(&set).unwrap();
            let mtl_ll = mtl_lora.forward_batch(&seq, 0, true).unwrap();
            let (kl_l, max_dl) = kl_div(cpu_lst.logits(), mtl_ll);
            assert!(
                kl_l <= 5e-6 && max_dl <= 3e-3,
                "Metal granitehybrid GPU LoRA mismatch: kl={kl_l:.3e} max_d={max_dl:.3e}"
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
