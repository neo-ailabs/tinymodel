//! End-to-end CPU multi-token FP64 oracle parity, multi-ISA determinism, batch invariance,
//! `PrefixCache` recurrent `LayerState::Ssm` exact-checkpoint restore, `SuffixDrafter`
//! speculative decoding (`verify_speculative` + `commit_speculative`), and dynamic/merged LoRA
//! parity across all P1 sub-8B architectures (`olmo2`, `olmo3`, `granite`, `mamba2`,
//! `nemotron_h`, `falcon-h1`, `granitehybrid`) per `AGENTS.md` Rules #3, #5, #7, #9.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_core::KvType;
use tm_cpu::{Isa, ThreadPool, detect_caps, select_kernels};
use tm_gguf::GgufFile;
use tm_lora::{LoraMergeOptions, LoraSet, merge_lora_gguf};
use tm_models::argmax;
use tm_runtime::{CpuModel, Session, SpecController, SpecMode, SuffixDrafter};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_p1_{tag}_{}_{id}", std::process::id()));
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
fn p1_architectures_multi_token_oracle_parity_isa_and_batch() {
    let dir = tmp_dir("oracle");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p1_architectures_oracle.py");
    let tokens_csv = "1,5,12,7,19,3,28,14";

    // (tag, arch, eval_ctx, quant, no_rope_finetuned, oracle_kl_tol, oracle_diff_tol, isa_kl_tol, isa_diff_tol)
    // Note: for `q4_0` + `--act-q8`, `quantize_row_q8_0` is a discontinuous `round()` step function,
    // so a 1-ULP FP32-vs-FP64 trajectory difference straddling a half-integer threshold flips 1 `i8`
    // bin (`kl ~ 3e-5 << 1e-3` L2 gate), whereas all CPU ISAs share the exact FP32 `quantize_q8_0`
    // input and stay within `kl <= 5e-7`!
    #[allow(clippy::type_complexity)]
    let cases: &[(&str, &str, usize, &str, bool, f64, f32, f64, f32)] = &[
        (
            "olmo2_f32",
            "olmo2",
            64,
            "f32",
            false,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "olmo3_q4", "olmo3", 64, "q4_0", false, 1e-4, 3e-2, 5e-7, 2e-3,
        ),
        (
            "granite_short_f32",
            "granite",
            32,
            "f32",
            false,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "granite_long_q4",
            "granite",
            64,
            "q4_0",
            false,
            1e-4,
            3e-2,
            5e-7,
            2e-3,
        ),
        (
            "granite_norope_f32",
            "granite",
            64,
            "f32",
            true,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "mamba2_f32",
            "mamba2",
            64,
            "f32",
            false,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "mamba2_q4",
            "mamba2",
            64,
            "q4_0",
            false,
            1e-4,
            3e-2,
            5e-7,
            2e-3,
        ),
        (
            "nemotron_h_f32",
            "nemotron_h",
            64,
            "f32",
            false,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "nemotron_h_q4",
            "nemotron_h",
            64,
            "q4_0",
            false,
            1e-4,
            3e-2,
            5e-7,
            2e-3,
        ),
        (
            "falcon_h1_f32",
            "falcon-h1",
            64,
            "f32",
            false,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "falcon_h1_q4",
            "falcon-h1",
            64,
            "q4_0",
            false,
            1e-4,
            3e-2,
            5e-7,
            2e-3,
        ),
        (
            "granitehybrid_f32",
            "granitehybrid",
            64,
            "f32",
            false,
            1e-8,
            5e-4,
            1e-8,
            5e-4,
        ),
        (
            "granitehybrid_q4",
            "granitehybrid",
            64,
            "q4_0",
            false,
            1e-4,
            3e-2,
            5e-7,
            2e-3,
        ),
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

    let pool1 = ThreadPool::new(1);
    let pool4 = ThreadPool::new(4);
    let kt_gen = select_kernels(Isa::Generic);

    for &(tag, arch, eval_ctx, quant, no_rope_ft, kl_tol, diff_tol, isa_kl_tol, isa_diff_tol) in
        cases
    {
        let gguf_path = dir.join(format!("{tag}.gguf"));
        let ref_tmlg = dir.join(format!("{tag}.tmlg"));
        let mut cmd = Command::new(&py);
        cmd.arg(&script)
            .arg("--arch")
            .arg(arch)
            .arg("--out-gguf")
            .arg(&gguf_path)
            .arg("--out-tmlg")
            .arg(&ref_tmlg)
            .arg("--tokens")
            .arg(tokens_csv)
            .arg("--quant")
            .arg(quant)
            .arg("--eval-ctx")
            .arg(eval_ctx.to_string());
        if quant == "q4_0" {
            cmd.arg("--act-q8");
        }
        if no_rope_ft {
            cmd.arg("--no-rope-finetuned");
        }
        let st = cmd.status().expect("run p1_architectures_oracle.py");
        assert!(st.success(), "oracle failed for {tag}");

        let (toks, vocab, ref_steps) = read_tmlg(&ref_tmlg);
        let gguf = GgufFile::open(&gguf_path).expect("open gguf");
        let model = CpuModel::from_gguf(&gguf).expect("load CpuModel");
        assert_eq!(model.arch(), arch);
        assert_eq!(model.vocab_size(), vocab);

        // 1. Multi-token FP64 oracle parity (`AGENTS.md` Rule #7)
        let mut st_gen = model.new_state(eval_ctx);
        let mut gen_steps = Vec::with_capacity(toks.len());
        for (pos, &tid) in toks.iter().enumerate() {
            model
                .forward_token(&mut st_gen, &pool1, kt_gen, tid, pos, true)
                .expect("forward_token");
            let cand = st_gen.logits().to_vec();
            let (kl, max_d) = kl_div(&ref_steps[pos], &cand);
            assert_eq!(
                argmax(&ref_steps[pos]),
                argmax(&cand),
                "{tag} step {pos}: Top-1 mismatch vs Python FP64 oracle"
            );
            assert!(
                kl <= kl_tol && max_d <= diff_tol,
                "{tag} step {pos}: kl={kl:.3e} (tol {kl_tol:.1e}), max_d={max_d:.3e} (tol {diff_tol:.1e})"
            );
            gen_steps.push(cand);
        }

        // 2. Host ISA parity + 1T vs 4T 0-ULP determinism + batch vs single-token 0-ULP invariance
        for &isa in &isa_cands {
            let kt = select_kernels(isa);
            let mut st1 = model.new_state(eval_ctx);
            let mut st4 = model.new_state(eval_ctx);
            for (pos, &tid) in toks.iter().enumerate() {
                model
                    .forward_token(&mut st1, &pool1, kt, tid, pos, true)
                    .expect("st1");
                model
                    .forward_token(&mut st4, &pool4, kt, tid, pos, true)
                    .expect("st4");
                for (a, b) in st1.logits().iter().zip(st4.logits().iter()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{tag} {isa:?} 1T vs 4T drift");
                }
                let (kl, max_d) = kl_div(&gen_steps[pos], st4.logits());
                assert!(
                    kl <= isa_kl_tol && max_d <= isa_diff_tol,
                    "{tag} {isa:?} step {pos}: kl={kl:.3e} max_d={max_d:.3e}"
                );
            }

            let mut st_batch = model.new_state(eval_ctx);
            model
                .forward_batch(&mut st_batch, &pool4, kt, &toks, 0, true)
                .expect("forward_batch");
            for (i, (&a, &b)) in st4
                .logits()
                .iter()
                .zip(st_batch.logits().iter())
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tag} {isa:?} batch vs single logit[{i}] mismatch: {a} vs {b}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p1_session_prefix_cache_q8_kv_speculative_and_mamba2_lora_merge() {
    let dir = tmp_dir("session");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p1_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    // Repeated n-grams so `SuffixDrafter` actively drafts & exercises partial/full rollback!
    let prompt = [1u32, 5, 12, 7, 1, 5, 12, 7, 19];

    for arch in [
        "olmo2",
        "granite",
        "mamba2",
        "nemotron_h",
        "falcon-h1",
        "granitehybrid",
    ] {
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
                .expect("run oracle")
                .success()
        );
        let gguf = GgufFile::open(&gguf_path).expect("open gguf");
        let cpu_model = CpuModel::from_gguf(&gguf).expect("CpuModel");

        let mut sess = Session::new(&cpu_model, 64, 16 * 1024 * 1024);
        let rep1 = sess
            .prefill_with_cache(&pool, &prompt, None)
            .expect("prefill 1");
        assert_eq!(rep1.cached_tokens, 0);
        let mut greedy_ref = vec![argmax(sess.logits())];
        for _ in 0..7 {
            let last = *greedy_ref.last().unwrap();
            sess.decode_step(&pool, last, true).expect("decode_step");
            greedy_ref.push(argmax(sess.logits()));
        }

        // Re-prefill hits `PrefixCache` (including exact checkpoint restore for recurrent `LayerState::Ssm`!)
        let rep2 = sess
            .prefill_with_cache(&pool, &prompt, None)
            .expect("prefill 2");
        assert_eq!(
            rep2.cached_tokens,
            prompt.len() - 1,
            "{arch}: PrefixCache did not restore exact recurrent/KV prefix"
        );

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

        // `Q8_0` KV cache execution smoke + KL sanity
        let mut sess_q8 = Session::new_with_options(&cpu_model, 64, 0, KvType::Q8_0);
        sess_q8
            .prefill_with_cache(&pool, &prompt, None)
            .expect("prefill q8");
        sess.prefill_with_cache(&pool, &prompt, None)
            .expect("prefill f16");
        let (kl_q8, _) = kl_div(sess.logits(), sess_q8.logits());
        assert!(kl_q8 < 5e-2, "{arch}: Q8_0 KV KL too high: {kl_q8:.3e}");

        // Verify dynamic LoRA == offline `merge_lora_gguf` on `falcon-h1` (`ssm_in` + `attn_q`)!
        if arch == "falcon-h1" {
            let f32_path = dir.join("falcon_h1_f32.gguf");
            assert!(
                Command::new(&py)
                    .arg(&script)
                    .arg("--arch")
                    .arg("falcon-h1")
                    .arg("--out-gguf")
                    .arg(&f32_path)
                    .arg("--quant")
                    .arg("f32")
                    .status()
                    .expect("falcon-h1 f32")
                    .success()
            );
            let f32_gguf = GgufFile::open(&f32_path).unwrap();
            let mut pairs = std::collections::HashMap::new();
            let mut order = Vec::new();
            for (tname, cols, rows) in [
                ("blk.0.ssm_in.weight", 128usize, 388usize),
                ("blk.0.ssm_out.weight", 128usize, 128usize),
                ("blk.0.attn_q.weight", 128usize, 128usize),
            ] {
                order.push(tname.to_string());
                pairs.insert(
                    tname.to_string(),
                    tm_lora::LoraWeightPair {
                        base_name: tname.to_string(),
                        is_token_embd: false,
                        rank: 4,
                        d_in: cols,
                        d_out: rows,
                        a: (0..4 * cols)
                            .map(|i| ((i as f32) * 0.013).sin() * 0.2)
                            .collect(),
                        b: (0..rows * 4)
                            .map(|i| ((i as f32) * 0.017).cos() * 0.15)
                            .collect(),
                    },
                );
            }
            let adapter = tm_lora::LoraAdapter {
                arch: "falcon-h1".to_string(),
                alpha: 8.0,
                name: Some("falcon-h1-lora".to_string()),
                pairs,
                pair_order: order,
            };
            let mut set = LoraSet::new();
            set.push(adapter, 1.0);
            set.validate_against_gguf(&f32_gguf)
                .expect("validate falcon-h1 lora");

            let mut dyn_model = CpuModel::from_gguf(&f32_gguf).unwrap();
            dyn_model.bind_lora(&f32_gguf, &set).unwrap();
            let kt = select_kernels(Isa::Generic);
            let mut dyn_st = dyn_model.new_state(64);
            dyn_model
                .forward_batch(&mut dyn_st, &pool, kt, &prompt, 0, true)
                .unwrap();

            let merged_path = dir.join("falcon_h1_merged.gguf");
            merge_lora_gguf(&f32_gguf, &set, &merged_path, &LoraMergeOptions::default())
                .expect("merge falcon-h1 lora");
            let merged_gguf = GgufFile::open(&merged_path).unwrap();
            let merged_model = CpuModel::from_gguf(&merged_gguf).unwrap();
            let mut merged_st = merged_model.new_state(64);
            merged_model
                .forward_batch(&mut merged_st, &pool, kt, &prompt, 0, true)
                .unwrap();
            let (kl_lora, max_d_lora) = kl_div(merged_st.logits(), dyn_st.logits());
            assert!(
                kl_lora < 1e-10 && max_d_lora < 1e-4,
                "falcon-h1 dynamic vs merged LoRA mismatch: kl={kl_lora:.3e} max_d={max_d_lora:.3e}"
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
