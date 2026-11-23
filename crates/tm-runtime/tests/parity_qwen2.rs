//! End-to-end `qwen2` (Qwen2 & Qwen2.5) CPU multi-token parity, ISA determinism,
//! speculative decoding, `PrefixCache`, `Q8_0` KV, `Q4_0x4` repack, `--prune-vocab`,
//! and `qwen2` BPE digit-splitting tests (`AGENTS.md` Rules #3, #5, #7, #9).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_core::{DType, KvType};
use tm_cpu::{Isa, ThreadPool, active_kernels, detect_caps, select_kernels};
use tm_gguf::{ConvertOptions, GgufFile};
use tm_models::{LlamaModel, LlamaState, argmax};
use tm_runtime::{CpuModel, Session, SpecController, SpecMode, SuffixDrafter};
use tm_tokenizer::Tokenizer;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_qwen2_{tag}_{}_{id}", std::process::id()));
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
fn qwen2_multi_token_oracle_parity_isa_batch_and_convert() {
    let dir = tmp_dir("oracle");
    let py = find_python3_with_numpy();
    let oracle_script = repo_root().join("tools/oracle/qwen2_oracle.py");
    let tokens_csv = "1,5,12,7,19,3,28,14";

    for (case_tag, quant, extra_args, kl_tol, diff_tol) in [
        ("f32_tied_all_bias", "f32", vec![], 1e-11f64, 2e-5f32),
        (
            "q4_0_untied_qkv_bias",
            "q4_0",
            vec!["--act-q8", "--untied", "--no-wo-bias", "--no-output-bias"],
            1e-9f64,
            2e-4f32,
        ),
        (
            "q4_0_tied_full_bias",
            "q4_0",
            vec!["--act-q8"],
            1e-9f64,
            2e-4f32,
        ),
    ] {
        let gguf_path = dir.join(format!("{case_tag}.gguf"));
        let ref_tmlg = dir.join(format!("{case_tag}_ref.tmlg"));
        let mut cmd = Command::new(&py);
        cmd.arg(&oracle_script)
            .arg("--out-gguf")
            .arg(&gguf_path)
            .arg("--out-tmlg")
            .arg(&ref_tmlg)
            .arg("--tokens")
            .arg(tokens_csv)
            .arg("--quant")
            .arg(quant);
        for a in &extra_args {
            cmd.arg(a);
        }
        let st = cmd.status().expect("run qwen2_oracle.py");
        assert!(st.success(), "qwen2_oracle.py failed for {case_tag}");

        let (toks, vocab, ref_steps) = read_tmlg(&ref_tmlg);
        let gguf = GgufFile::open(&gguf_path).expect("open synthetic qwen2 gguf");
        let model = LlamaModel::from_gguf(&gguf).expect("load LlamaModel(qwen2)");
        assert_eq!(model.cfg.arch, "qwen2");
        assert_eq!(model.cfg.vocab_size, vocab);

        let pool1 = ThreadPool::new(1);
        let pool4 = ThreadPool::new(4);
        let kt_gen = select_kernels(Isa::Generic);

        // 1. Multi-token parity vs independent FP64 oracle (`AGENTS.md` Rule #7)
        let mut st_gen = LlamaState::new(&model.cfg, 64);
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
                "{case_tag} step {pos}: Top-1 mismatch"
            );
            assert!(
                kl <= kl_tol && max_d <= diff_tol,
                "{case_tag} step {pos}: kl={kl:.3e} (tol {kl_tol:.1e}), max_d={max_d:.3e} (tol {diff_tol:.1e})"
            );
            gen_steps.push(cand);
        }

        // 2. All supported host ISAs (`neon`, `dotprod`, `i8mm`, `avx2`) + 4T pool match `generic`!
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
        for &isa in &isa_cands {
            let kt = select_kernels(isa);
            let mut st1 = LlamaState::new(&model.cfg, 64);
            let mut st4 = LlamaState::new(&model.cfg, 64);
            for (pos, &tid) in toks.iter().enumerate() {
                model
                    .forward_token(&mut st1, &pool1, kt, tid, pos, true)
                    .expect("st1");
                model
                    .forward_token(&mut st4, &pool4, kt, tid, pos, true)
                    .expect("st4");
                // 1T vs 4T must be 0-ULP bit-exact (`AGENTS.md` §9)
                for (a, b) in st1.logits().iter().zip(st4.logits().iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{case_tag} {isa:?} 1T vs 4T drift"
                    );
                }
                let (kl, max_d) = kl_div(&gen_steps[pos], st4.logits());
                assert!(
                    kl <= 1e-11 && max_d <= 5e-5,
                    "{case_tag} {isa:?} step {pos}: kl={kl:.3e} max_d={max_d:.3e}"
                );
            }

            // 3. `forward_batch` vs `forward_token` 0-ULP invariance (`AGENTS.md` §9)
            let mut st_batch = LlamaState::new(&model.cfg, 64);
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
                    "{case_tag} {isa:?} batch vs single logit[{i}] mismatch: {a} vs {b}"
                );
            }
        }

        // 4. `Q4_0 -> Q4_0x4` lossless repack + `--prune-vocab` slicing `output.bias`
        if quant == "q4_0" {
            let q4x4_path = dir.join(format!("{case_tag}_q4x4.gguf"));
            gguf.convert_to_q4_0x4(&q4x4_path).expect("repack q4_0x4");
            let gguf_x4 = GgufFile::open(&q4x4_path).expect("open q4x4");
            let model_x4 = LlamaModel::from_gguf(&gguf_x4).expect("load q4x4 qwen2");
            let kt = active_kernels();
            let mut st_ref = LlamaState::new(&model.cfg, 64);
            let mut st_x4 = LlamaState::new(&model_x4.cfg, 64);
            model
                .forward_batch(&mut st_ref, &pool4, kt, &toks, 0, true)
                .expect("ref batch");
            model_x4
                .forward_batch(&mut st_x4, &pool4, kt, &toks, 0, true)
                .expect("x4 batch");
            for (a, b) in st_ref.logits().iter().zip(st_x4.logits().iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "Q4_0x4 repack must be 0-ULP lossless"
                );
            }

            if case_tag == "q4_0_tied_full_bias" {
                let pruned_path = dir.join("qwen2_pruned.gguf");
                let rep = gguf
                    .convert_with_options(
                        &pruned_path,
                        &ConvertOptions {
                            quant: Some(DType::Q4_0),
                            lm_head_quant: None,
                            prune_vocab: Some(272),
                            vocab_corpus: None,
                        },
                    )
                    .expect("prune vocab on qwen2 with output.bias");
                assert!(rep.new_vocab_size <= 276 && rep.new_vocab_size < rep.old_vocab_size);
                let gguf_p = GgufFile::open(&pruned_path).expect("open pruned");
                let model_p = LlamaModel::from_gguf(&gguf_p).expect("load pruned qwen2");
                assert_eq!(
                    model_p.output_bias.as_ref().map(|v| v.len()),
                    Some(rep.new_vocab_size)
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn qwen2_session_prefix_cache_q8_kv_spec_and_tokenizer() {
    let dir = tmp_dir("session");
    let py = find_python3_with_numpy();
    let oracle_script = repo_root().join("tools/oracle/qwen2_oracle.py");
    let gguf_path = dir.join("qwen2_q4.gguf");
    let st = Command::new(&py)
        .arg(&oracle_script)
        .arg("--out-gguf")
        .arg(&gguf_path)
        .arg("--quant")
        .arg("q4_0")
        .status()
        .expect("run oracle");
    assert!(st.success());

    let gguf = GgufFile::open(&gguf_path).expect("open gguf");

    // 1. Tokenizer: `qwen2` pre-tokenizer splits multi-digit numbers into individual digits (`max_digits = 1`)
    //    and auto-detects `<|im_end|>` (id 2) as `eot_id`.
    let tok = Tokenizer::from_gguf(&gguf).expect("load qwen2 tokenizer");
    assert_eq!(tok.eot_id, Some(2));
    assert_eq!(tok.eos_id, Some(2));
    let digit_ids = tok.encode("2026", false);
    // '2'->5, '0'->3, '2'->5, '6'->9
    assert_eq!(digit_ids, vec![5, 3, 5, 9]);

    // 2. `CpuModel::from_gguf` + `Session` + `PrefixCache` + `SuffixDrafter` speculative decoding + `Q8_0` KV
    let cpu_model = CpuModel::from_gguf(&gguf).expect("CpuModel(qwen2)");
    assert_eq!(cpu_model.arch(), "qwen2");
    let pool = ThreadPool::new(2);
    let prompt = [1u32, 5, 12, 7, 1, 5, 12, 7, 19];

    let mut sess = Session::new(&cpu_model, 64, 16 * 1024 * 1024);
    let rep1 = sess
        .prefill_with_cache(&pool, &prompt, None)
        .expect("prefill 1");
    assert_eq!(rep1.cached_tokens, 0);
    let mut greedy_off = Vec::new();
    let mut cur = argmax(sess.logits());
    greedy_off.push(cur);
    for _ in 0..8 {
        sess.decode_step(&pool, cur, true).expect("step off");
        cur = argmax(sess.logits());
        greedy_off.push(cur);
    }

    // Re-prefill on the SAME `Session` hits `PrefixCache` (`cached_tokens == prompt.len() - 1`)
    // and runs `Suffix` speculative decoding: greedy output MUST match 100%!
    let rep2 = sess
        .prefill_with_cache(&pool, &prompt, None)
        .expect("prefill 2");
    assert_eq!(
        rep2.cached_tokens,
        prompt.len() - 1,
        "PrefixCache should restore full qwen2 prefix"
    );
    let mut suffix = SuffixDrafter::default();
    let mut ctrl = SpecController::new(4);
    let (greedy_spec, _stats) = sess
        .generate_greedy_speculative(
            &pool,
            &prompt,
            greedy_off.len(),
            SpecMode::Suffix,
            &mut suffix,
            None,
            &mut ctrl,
            |_, _| false,
        )
        .expect("speculative greedy on qwen2");
    assert_eq!(
        greedy_off, greedy_spec,
        "speculative greedy mismatch on qwen2"
    );

    // `Q8_0` KV cache runs cleanly
    let mut sess_q8 = Session::new_with_options(&cpu_model, 64, 0, KvType::Q8_0);
    sess_q8
        .prefill_with_cache(&pool, &prompt, None)
        .expect("q8 kv prefill");
    let next_q8 = argmax(sess_q8.logits());
    sess_q8
        .decode_step(&pool, next_q8, true)
        .expect("q8 decode step");

    let _ = fs::remove_dir_all(&dir);
}
