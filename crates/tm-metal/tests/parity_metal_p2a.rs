//! Apple Metal GPU single-CommandBuffer Sparse MoE (`GpuMoeEngine` + `GpuSparseMoe`) decode,
//! greedy kernel, `reset_state`, batched prefill (`> 32` tile boundary), `profile`, and dynamic
//! GPU LoRA (`bind_lora` on router + shared expert + 1D `ffn_gate_inp_shexp`!) parity tests across
//! all 7 P2a architectures (`olmoe`, `granitemoe`, `granitemoehybrid` / `granitehybrid`,
//! `qwen2moe`, `qwen3moe`, `bailingmoe2`, `exaone`) per `AGENTS.md` Rules #3, #7, #9.
//!
//! Note on `Q4_0` reference: Metal `gemv_q4_0_res` / `moe_indexed_gemv_q4_0` multiply `Q4_0`
//! weights directly by unquantized `F32` activations (whereas CPU `Q4_0` quantizes activations to
//! `Q8_0`). Comparing Metal `Q4_0` directly against the independent `Q4_0 x FP64` Python oracle
//! (`--quant q4_0` without `--act-q8`) verifies `moe_indexed_gemv_q4_0` to `kl <= 5e-5` (100x
//! tighter than `Q8_0`-activation comparison) without spurious `Q8_0` router tie flips!

#![cfg(target_vendor = "apple")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_lora::{LoraAdapter, LoraSet, LoraWeightPair};
use tm_metal::MetalRunner;
use tm_models::argmax;
use tm_runtime::CpuModel;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_mtl_p2a_{tag}_{}_{id}", std::process::id()));
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
fn metal_p2a_architectures_decode_greedy_batch_reset_profile_and_lora_parity() {
    let dir = tmp_dir("parity");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let max_ctx = 64usize;

    // (tag, arch, quant, kl_tol, diff_tol)
    let cases: &[(&str, &str, &str, f64, f32)] = &[
        ("olmoe_f32", "olmoe", "f32", 5e-6, 3e-3),
        ("olmoe_q4", "olmoe", "q4_0", 5e-5, 5e-3),
        ("granitemoe_f32", "granitemoe", "f32", 5e-6, 3e-3),
        ("granitemoe_q4", "granitemoe", "q4_0", 5e-5, 5e-3),
        (
            "granitemoehybrid_f32",
            "granitemoehybrid",
            "f32",
            5e-6,
            3e-3,
        ),
        ("granitehybrid_moe_q4", "granitehybrid", "q4_0", 5e-5, 5e-3),
        ("qwen2moe_f32", "qwen2moe", "f32", 5e-6, 3e-3),
        ("qwen2moe_q4", "qwen2moe", "q4_0", 5e-5, 5e-3),
        ("qwen3moe_f32", "qwen3moe", "f32", 5e-6, 3e-3),
        ("qwen3moe_q4", "qwen3moe", "q4_0", 5e-5, 5e-3),
        ("bailingmoe2_f32", "bailingmoe2", "f32", 5e-6, 3e-3),
        ("bailingmoe2_q4", "bailingmoe2", "q4_0", 5e-5, 8e-3),
        ("exaone_f32", "exaone", "f32", 5e-6, 3e-3),
        ("exaone_q4", "exaone", "q4_0", 5e-5, 5e-3),
    ];

    let seq_csv = "1,7,14,23,5,31,9,18";
    let seq = [1u32, 7, 14, 23, 5, 31, 9, 18];

    for &(tag, arch, quant, kl_tol, diff_tol) in cases {
        let gguf_path = dir.join(format!("{tag}.gguf"));
        let tmlg_path = dir.join(format!("{tag}.tmlg"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&gguf_path)
                .arg("--out-logits")
                .arg(&tmlg_path)
                .arg("--tokens")
                .arg(seq_csv)
                .arg("--quant")
                .arg(quant)
                .status()
                .expect("run p2a oracle")
                .success()
        );
        let (_, _, oracle_steps) = read_tmlg(&tmlg_path);

        let gguf = Arc::new(GgufFile::open(&gguf_path).expect("open gguf"));
        let cpu = CpuModel::from_gguf(&gguf).expect("cpu model");
        let mut cpu_st = cpu.new_state(max_ctx);

        let mut mtl_dec = MetalRunner::from_gguf(gguf.clone(), max_ctx).expect("MetalRunner");
        assert_eq!(mtl_dec.arch(), arch);

        let mut first_pass_last = Vec::new();
        for (pos, &tok) in seq.iter().enumerate() {
            let mtl_l = mtl_dec
                .forward_token(tok, pos, true)
                .expect("metal forward_token");
            let ref_l: &[f32] = if quant == "f32" {
                cpu.forward_token(&mut cpu_st, &pool, kt, tok, pos, true)
                    .expect("cpu forward_token");
                cpu_st.logits()
            } else {
                &oracle_steps[pos]
            };
            let (kl, max_d) = kl_div(ref_l, mtl_l);
            assert_eq!(
                argmax(ref_l),
                argmax(mtl_l),
                "Metal {tag} step {pos} Top-1 mismatch"
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

        // `reset_state()` must reproduce bit-identical logits
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

        // Batched prefill (`n_batch = 37` crossing 32-token tile boundary):
        // - For `f32`, compare against `CpuModel::forward_batch`.
        // - For `q4_0`, compare `MetalRunner::forward_batch(37)` against `MetalRunner::forward_token` x 37 (`kl <= 5e-6`)!
        let n_batch = 37usize;
        let batch_toks: Vec<u32> = (0..n_batch).map(|i| ((i * 5 + 3) % 90) as u32).collect();
        let mut mtl_bat = MetalRunner::from_gguf(gguf.clone(), max_ctx).expect("MetalRunner batch");
        let mtl_bl = mtl_bat
            .forward_batch(&batch_toks, 0, true)
            .expect("metal batch")
            .to_vec();

        let ref_bl: Vec<f32> = if quant == "f32" {
            let mut cpu_b = cpu.new_state(max_ctx);
            cpu.forward_batch(&mut cpu_b, &pool, kt, &batch_toks, 0, true)
                .expect("cpu batch");
            cpu_b.logits().to_vec()
        } else {
            let mut mtl_seq =
                MetalRunner::from_gguf(gguf.clone(), max_ctx).expect("MetalRunner seq");
            let mut last = Vec::new();
            for (pos, &tok) in batch_toks.iter().enumerate() {
                let l = mtl_seq
                    .forward_token(tok, pos, pos + 1 == n_batch)
                    .expect("seq tok");
                if pos + 1 == n_batch {
                    last = l.to_vec();
                }
            }
            last
        };
        let (kl_b, max_db) = kl_div(&ref_bl, &mtl_bl);
        assert_eq!(
            argmax(&ref_bl),
            argmax(&mtl_bl),
            "Metal {tag} batch({n_batch}) Top-1 mismatch"
        );
        // Note: `LlamaMetalRunner::forward_batch` uses `simdgroup_half8x8` (`FP16` Q) FlashAttention,
        // whereas `forward_token` / `CpuModel` use `FP32` Q (`kl <= 5e-5 << 1e-3` L2/L3 GPU gate).
        assert!(
            kl_b <= 5e-5 && max_db <= 2.5e-2,
            "Metal {tag} batch({n_batch}): kl={kl_b:.3e} max_d={max_db:.3e}"
        );

        // Profile smoke test on `olmoe_q4` (LlamaMetalRunner) & `granitehybrid_moe_q4` (Mamba2MetalRunner)
        if tag == "olmoe_q4" || tag == "granitehybrid_moe_q4" {
            let prof = mtl_bat.profile(2, 8).expect("MetalRunner profile");
            assert!(prof.decode_single_cb_wall_us > 0.0);
        }

        // Dynamic GPU LoRA (`bind_lora`) on `qwen2moe_f32` (router + shared expert + 1D `ffn_gate_inp_shexp`!)
        // and `granitemoehybrid_f32` (`ssm_in` + router + shared expert)!
        if tag == "qwen2moe_f32" {
            let mut pairs = HashMap::new();
            let mut order = Vec::new();
            for (tname, is_emb, cols, rows) in [
                ("token_embd.weight", true, 96usize, 64usize),
                ("blk.0.ffn_gate_inp.weight", false, 64usize, 8usize),
                ("blk.0.ffn_gate_shexp.weight", false, 64usize, 64usize),
                ("blk.0.ffn_gate_inp_shexp.weight", false, 64usize, 1usize),
                ("output.weight", false, 64usize, 96usize),
            ] {
                order.push(tname.to_string());
                let a_len = cols * 4;
                let b_len = rows * 4;
                pairs.insert(
                    tname.to_string(),
                    LoraWeightPair {
                        base_name: tname.to_string(),
                        is_token_embd: is_emb,
                        rank: 4,
                        d_in: cols,
                        d_out: rows,
                        a: (0..a_len)
                            .map(|i| ((i as f32) * 0.017).sin() * 0.12)
                            .collect(),
                        b: (0..b_len)
                            .map(|i| ((i as f32) * 0.023).cos() * 0.12)
                            .collect(),
                    },
                );
            }
            let adapter = LoraAdapter {
                arch: "qwen2moe".to_string(),
                alpha: 8.0,
                name: Some("qwen2moe-gpu-lora".to_string()),
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
                "Metal qwen2moe GPU LoRA mismatch: kl={kl_l:.3e} max_d={max_dl:.3e}"
            );
        }

        if tag == "granitemoehybrid_f32" {
            let mut pairs = HashMap::new();
            let mut order = Vec::new();
            for (tname, cols, rows) in [
                ("blk.0.ssm_in.weight", 64usize, 388usize),
                ("blk.0.ffn_gate_inp.weight", 64usize, 8usize),
                ("blk.1.ffn_up_shexp.weight", 64usize, 64usize),
            ] {
                order.push(tname.to_string());
                pairs.insert(
                    tname.to_string(),
                    LoraWeightPair {
                        base_name: tname.to_string(),
                        is_token_embd: false,
                        rank: 4,
                        d_in: cols,
                        d_out: rows,
                        a: (0..4 * cols)
                            .map(|i| ((i as f32) * 0.019).sin() * 0.12)
                            .collect(),
                        b: (0..rows * 4)
                            .map(|i| ((i as f32) * 0.029).cos() * 0.12)
                            .collect(),
                    },
                );
            }
            let adapter = LoraAdapter {
                arch: "granitemoehybrid".to_string(),
                alpha: 8.0,
                name: Some("gmh-gpu-lora".to_string()),
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
                "Metal granitemoehybrid GPU LoRA mismatch: kl={kl_l:.3e} max_d={max_dl:.3e}"
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
