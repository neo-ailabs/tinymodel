//! End-to-end CPU multi-token FP64 oracle parity, multi-ISA determinism, `1T == 4T` &
//! `forward_batch == forward_token` **0-ULP** invariance, `Session` + `PrefixCache` + `Q8_0` KV
//! (including recurrent `LayerState::Ssm` on `granitemoehybrid`), `SuffixDrafter` speculative
//! decoding, and dynamic vs `merge_lora_gguf` LoRA parity (including 1D `ffn_gate_inp_shexp.weight`!)
//! across all 7 P2a architectures (`olmoe`, `granitemoe`, `granitemoehybrid` / `granitehybrid`,
//! `qwen2moe`, `qwen3moe`, `bailingmoe2`, `exaone`) per `AGENTS.md` Rules #3, #5, #7, #9.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_core::KvType;
use tm_cpu::{Isa, ThreadPool, detect_caps, select_kernels};
use tm_gguf::GgufFile;
use tm_lora::{LoraAdapter, LoraMergeOptions, LoraSet, LoraWeightPair, merge_lora_gguf};
use tm_models::argmax;
use tm_runtime::{CpuModel, Session, SpecController, SpecMode, SuffixDrafter};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_p2a_{tag}_{}_{id}", std::process::id()));
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

fn kl_div(ref_logits: &[f32], cand_logits: &[f32]) -> (f64, f32) {
    let max_r = ref_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let max_c = cand_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let zr: f64 = ref_logits.iter().map(|&v| ((v as f64) - max_r).exp()).sum();
    let zc: f64 = cand_logits
        .iter()
        .map(|&v| ((v as f64) - max_c).exp())
        .sum();
    let log_zr = max_r + zr.ln();
    let log_zc = max_c + zc.ln();
    let mut kl = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&r, &c) in ref_logits.iter().zip(cand_logits.iter()) {
        let d = (r - c).abs();
        if d > max_abs {
            max_abs = d;
        }
        let lp = (r as f64) - log_zr;
        let lq = (c as f64) - log_zc;
        kl += lp.exp() * (lp - lq);
    }
    (kl.max(0.0), max_abs)
}

#[test]
fn p2a_architectures_multi_token_oracle_parity_isa_and_batch_0ulp() {
    let dir = tmp_dir("oracle");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let tokens_csv = "1,5,12,7,19,3,28,14";

    // Note: on `q4_0` MoE models, `ffn_gate_inp.weight` stays `F32` (matching `llama-quantize`),
    // so SIMD (`vfmaq_f32`) vs scalar (`Generic`) router accumulation differs by ~1 ULP, which can
    // straddle 1 discontinuous `quantize_row_q8_0` half-integer bin (`kl ~ 2.5e-5 <= 1e-4`).
    // Within every ISA, `1T == 4T` and `forward_batch == forward_token` are strictly 0-ULP (`to_bits()`).
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, &str, &str, f64, f32, f64, f32)] = &[
        ("olmoe_f32", "olmoe", "f32", 1e-8, 5e-4, 1e-8, 5e-4),
        ("olmoe_q4", "olmoe", "q4_0", 1e-4, 3e-2, 1e-4, 3e-2),
        (
            "granitemoe_f32",
            "granitemoe",
            "f32",
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "granitemoe_q4",
            "granitemoe",
            "q4_0",
            1e-4,
            3e-2,
            1e-4,
            3e-2,
        ),
        (
            "granitemoehybrid_f32",
            "granitemoehybrid",
            "f32",
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "granitehybrid_moe_q4",
            "granitehybrid",
            "q4_0",
            1e-4,
            3e-2,
            1e-4,
            3e-2,
        ),
        ("qwen2moe_f32", "qwen2moe", "f32", 1e-8, 5e-4, 1e-8, 5e-4),
        ("qwen2moe_q4", "qwen2moe", "q4_0", 1e-4, 3e-2, 1e-4, 3e-2),
        ("qwen3moe_f32", "qwen3moe", "f32", 1e-8, 5e-4, 1e-8, 5e-4),
        ("qwen3moe_q4", "qwen3moe", "q4_0", 1e-4, 3e-2, 1e-4, 3e-2),
        (
            "bailingmoe2_f32",
            "bailingmoe2",
            "f32",
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "bailingmoe2_q4",
            "bailingmoe2",
            "q4_0",
            1e-4,
            5e-2,
            1e-4,
            5e-2,
        ),
        ("exaone_f32", "exaone", "f32", 1e-8, 5e-4, 1e-8, 5e-4),
        ("exaone_q4", "exaone", "q4_0", 1e-4, 3e-2, 1e-4, 3e-2),
    ];

    let caps = detect_caps();
    let mut isa_cands = vec![Isa::Generic];
    if caps.neon {
        isa_cands.push(Isa::NeonV80);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if caps.dotprod {
            isa_cands.push(Isa::NeonDotprod);
        }
        if caps.i8mm {
            isa_cands.push(Isa::NeonI8mm);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if caps.avx2 {
            isa_cands.push(Isa::Avx2);
        }
    }

    let pool_1t = ThreadPool::new(1);
    let pool_4t = ThreadPool::new(4);
    let kt_gen = select_kernels(Isa::Generic);

    for &(tag, arch, quant, oracle_kl_tol, oracle_diff_tol, isa_kl_tol, isa_diff_tol) in cases {
        let gguf_path = dir.join(format!("{tag}.gguf"));
        let tmlg_path = dir.join(format!("{tag}.tmlg"));
        let mut cmd = Command::new(&py);
        cmd.arg(&script)
            .arg("--arch")
            .arg(arch)
            .arg("--out-gguf")
            .arg(&gguf_path)
            .arg("--out-logits")
            .arg(&tmlg_path)
            .arg("--tokens")
            .arg(tokens_csv)
            .arg("--quant")
            .arg(quant);
        if quant == "q4_0" {
            cmd.arg("--act-q8");
        }
        assert!(cmd.status().expect("run p2a oracle").success());

        let (toks, vocab, ref_steps) = read_tmlg(&tmlg_path);
        assert_eq!(vocab, 96);

        let gguf = GgufFile::open(&gguf_path).expect("open gguf");
        let model = CpuModel::from_gguf(&gguf).expect("load CpuModel");
        assert_eq!(model.arch(), arch);
        assert!(model.estimate_dirty_bytes(64) > 0);

        // 1. Multi-token FP64 oracle parity (`Isa::Generic` scalar reference)
        let mut st_gen = model.new_state(64);
        let mut gen_steps = Vec::with_capacity(toks.len());
        for (pos, &tok) in toks.iter().enumerate() {
            model
                .forward_token(&mut st_gen, &pool_1t, kt_gen, tok, pos, true)
                .expect("generic forward_token");
            let l = st_gen.logits().to_vec();
            let (kl, max_d) = kl_div(&ref_steps[pos], &l);
            assert!(
                kl <= oracle_kl_tol && max_d <= oracle_diff_tol,
                "{tag} step {pos}: kl={kl:.3e} (tol {oracle_kl_tol:.1e}), max_diff={max_d:.3e} (tol {oracle_diff_tol:.1e})"
            );
            assert_eq!(
                argmax(&ref_steps[pos]),
                argmax(&l),
                "{tag} step {pos} Top-1 mismatch vs FP64 oracle"
            );
            gen_steps.push(l);
        }

        // 2. Multi-ISA parity + `1T == 4T` 0-ULP + `batch == single` 0-ULP!
        for &isa in &isa_cands {
            let kt = select_kernels(isa);
            let mut st_1t = model.new_state(64);
            let mut st_4t = model.new_state(64);
            for (pos, &tok) in toks.iter().enumerate() {
                model
                    .forward_token(&mut st_1t, &pool_1t, kt, tok, pos, true)
                    .unwrap();
                model
                    .forward_token(&mut st_4t, &pool_4t, kt, tok, pos, true)
                    .unwrap();
                let (kl_isa, diff_isa) = kl_div(&gen_steps[pos], st_1t.logits());
                assert!(
                    kl_isa <= isa_kl_tol && diff_isa <= isa_diff_tol,
                    "{tag} ISA {:?} step {pos}: kl={kl_isa:.3e}, diff={diff_isa:.3e}",
                    isa
                );
                for (i, (&a, &b)) in st_1t.logits().iter().zip(st_4t.logits().iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{tag} {:?} 1T vs 4T 0-ULP drift at step {pos} idx {i}",
                        isa
                    );
                }
            }

            let mut st_batch = model.new_state(64);
            model
                .forward_batch(&mut st_batch, &pool_4t, kt, &toks, 0, true)
                .unwrap();
            for (i, (&a, &b)) in st_1t
                .logits()
                .iter()
                .zip(st_batch.logits().iter())
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tag} {:?} batch vs single-token 0-ULP drift at idx {i}",
                    isa
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p2a_session_prefix_cache_speculative_and_1d_shexp_lora_merge() {
    let dir = tmp_dir("session_lora");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let kt = tm_cpu::active_kernels();
    let prompt = [1u32, 4, 9, 12, 1, 4, 9, 12, 7];

    // 1. Session + `PrefixCache` + `SuffixDrafter` speculative + `Q8_0` KV across P2a representatives
    for arch in ["olmoe", "granitemoehybrid", "bailingmoe2", "exaone"] {
        let gguf_path = dir.join(format!("{arch}_q4.gguf"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&gguf_path)
                .arg("--quant")
                .arg("q4_0")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&gguf_path).unwrap();
        let cpu_model = CpuModel::from_gguf(&gguf).unwrap();

        let mut sess = Session::new(&cpu_model, 64, 16 * 1024 * 1024);
        let rep1 = sess
            .prefill_with_cache(&pool, &prompt, None)
            .expect("prefill 1");
        assert_eq!(rep1.cached_tokens, 0);
        let first_logits = sess.logits().to_vec();
        let mut greedy_ref = vec![argmax(&first_logits)];
        for _ in 0..7 {
            let last = *greedy_ref.last().unwrap();
            sess.decode_step(&pool, last, true).expect("decode_step");
            greedy_ref.push(argmax(sess.logits()));
        }

        let rep2 = sess
            .prefill_with_cache(&pool, &prompt, None)
            .expect("prefill 2");
        assert_eq!(
            rep2.cached_tokens,
            prompt.len() - 1,
            "{arch}: PrefixCache did not restore exact recurrent/KV prefix"
        );
        for (i, (&a, &b)) in first_logits.iter().zip(sess.logits().iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{arch}: PrefixCache restore 0-ULP mismatch at idx {i}"
            );
        }

        let mut drafter = SuffixDrafter::default();
        let mut spec_ctrl = SpecController::new(4);
        let (greedy_spec, _stats) = sess
            .generate_greedy_speculative(
                &pool,
                &prompt,
                greedy_ref.len(),
                SpecMode::Suffix,
                &mut drafter,
                None,
                &mut spec_ctrl,
                |_, _| false,
            )
            .expect("speculative greedy");
        assert_eq!(
            greedy_ref, greedy_spec,
            "{arch}: PrefixCache + Suffix speculative greedy mismatch"
        );

        let mut sess_q8 = Session::new_with_options(&cpu_model, 64, 0, KvType::Q8_0);
        sess_q8
            .prefill_with_cache(&pool, &prompt, None)
            .expect("prefill q8");
        sess.prefill_with_cache(&pool, &prompt, None)
            .expect("prefill f16");
        let (kl_q8, _) = kl_div(sess.logits(), sess_q8.logits());
        assert!(kl_q8 < 5e-2, "{arch}: Q8_0 KV KL too high: {kl_q8:.3e}");
    }

    // 2. `qwen2moe` F32 dynamic `bind_lora` vs offline `merge_lora_gguf` (including router `ffn_gate_inp`,
    //    shared expert `ffn_gate_shexp`, AND 1D `ffn_gate_inp_shexp.weight [64]`!)
    let q2m_path = dir.join("qwen2moe_f32.gguf");
    assert!(
        Command::new(&py)
            .arg(&script)
            .arg("--arch")
            .arg("qwen2moe")
            .arg("--out-gguf")
            .arg(&q2m_path)
            .arg("--quant")
            .arg("f32")
            .status()
            .unwrap()
            .success()
    );
    let q2m_gguf = GgufFile::open(&q2m_path).unwrap();
    let mut pairs = HashMap::new();
    let mut order = Vec::new();
    for (tname, cols, rows) in [
        ("blk.0.ffn_gate_inp.weight", 64usize, 8usize),
        ("blk.0.ffn_gate_shexp.weight", 64usize, 64usize),
        ("blk.0.ffn_gate_inp_shexp.weight", 64usize, 1usize),
        ("output.weight", 64usize, 96usize),
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
                    .map(|i| ((i as f32) * 0.031).sin() * 0.2)
                    .collect(),
                b: (0..rows * 4)
                    .map(|i| ((i as f32) * 0.027).cos() * 0.2)
                    .collect(),
            },
        );
    }
    let adapter = LoraAdapter {
        arch: "qwen2moe".to_string(),
        alpha: 8.0,
        name: Some("qwen2moe-lora".to_string()),
        pairs,
        pair_order: order,
    };
    let adapter_path = dir.join("qwen2moe_lora.gguf");
    adapter.save_gguf(&adapter_path, false).unwrap();
    let reloaded = LoraAdapter::open(&adapter_path).unwrap();

    let mut set = LoraSet::new();
    set.push(reloaded, 1.0);
    set.validate_against_gguf(&q2m_gguf).unwrap();

    let mut dyn_model = CpuModel::from_gguf(&q2m_gguf).unwrap();
    dyn_model.bind_lora(&q2m_gguf, &set).unwrap();
    let mut dyn_st = dyn_model.new_state(64);
    dyn_model
        .forward_batch(&mut dyn_st, &pool, kt, &[2, 9, 15, 7, 21], 0, true)
        .unwrap();

    let merged_path = dir.join("qwen2moe_merged.gguf");
    let rep = merge_lora_gguf(&q2m_gguf, &set, &merged_path, &LoraMergeOptions::default()).unwrap();
    assert_eq!(rep.merged_tensors, 4);
    let merged_gguf = GgufFile::open(&merged_path).unwrap();
    let merged_model = CpuModel::from_gguf(&merged_gguf).unwrap();
    let mut merged_st = merged_model.new_state(64);
    merged_model
        .forward_batch(&mut merged_st, &pool, kt, &[2, 9, 15, 7, 21], 0, true)
        .unwrap();

    let (kl_m, max_dm) = kl_div(dyn_st.logits(), merged_st.logits());
    assert!(
        kl_m <= 1e-9 && max_dm <= 1e-4,
        "qwen2moe dynamic vs merged LoRA mismatch: kl={kl_m:.3e} max_d={max_dm:.3e}"
    );

    let _ = fs::remove_dir_all(&dir);
}
