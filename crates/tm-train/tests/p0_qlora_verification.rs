//! QLoRA 0-ULP forward parity (`B == 0` and `B != 0`) and `f64` finite-difference Jacobian
//! verification across all P0 sub-8B architectures (`smollm3`, `phi3` fused QKV & FFN + LongRoPE,
//! `mistral` SWA, `gemma2` `attn_logit_softcapping`, `gemma3`, and `lfm3` `attn_gate` + sandwich norms).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_lora::LoraSet;
use tm_models::{Gemma4Model, Gemma4State, Lfm2Model, Lfm2State, LlamaModel, LlamaState};
use tm_train::{
    CheckpointMode, Gemma4Trainer, Lfm2Trainer, LlamaTrainer, LoraGradStore, LoraTargets,
    SplitMix64, init_lora_adapter, init_lora_adapter_gemma4, init_lora_adapter_lfm2,
};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_train_p0_{tag}_{}_{id}", std::process::id()));
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

fn randomize_lora_b(rng: &mut SplitMix64, adapter: &mut tm_lora::LoraAdapter) {
    for name in &adapter.pair_order {
        let p = adapter.pairs.get_mut(name).unwrap();
        for b in &mut p.b {
            *b = rng.uniform_sym(0.18);
        }
    }
}

#[test]
fn p0_zero_ulp_forward_parity_across_all_architectures_and_lora() {
    let dir = tmp_dir("fwd0ulp");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p0_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let targets = LoraTargets::parse_csv("all").unwrap();

    let seqs: [Vec<u32>; 3] = [
        vec![3],
        vec![1, 5, 2, 9, 14, 7, 3, 11, 6],
        (0..35).map(|i| ((i * 7 + 3) % 90) as u32).collect(),
    ];

    // 1. Llama-family P0 variants (`smollm3`, `phi3` LongRoPE + fused QKV/FFN, `mistral` SWA)
    for (arch, max_ctx) in [
        ("smollm3", 64usize),
        ("phi3", 64usize),
        ("mistral", 64usize),
    ] {
        let path = dir.join(format!("{arch}_q4.gguf"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&path)
                .arg("--quant")
                .arg("q4_0")
                .arg("--eval-ctx")
                .arg(max_ctx.to_string())
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let mut model = LlamaModel::from_gguf(&gguf).unwrap();
        let mut state = LlamaState::new(&model.cfg, max_ctx);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];

        {
            let mut trainer = LlamaTrainer::new(&model, max_ctx, 4, CheckpointMode::Fp16);
            for seq in &seqs {
                state.reset();
                model
                    .forward_batch(&mut state, &pool, kt, seq, 0, true)
                    .unwrap();
                trainer.forward_last_logits(&pool, None, seq, &mut tr_logits);
                for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{arch} 0-ULP mismatch (no LoRA) len={} idx={i}",
                        seq.len()
                    );
                }
            }
        }

        let mut ad = init_lora_adapter(&model, 4, 8.0, &targets, 42).unwrap();
        randomize_lora_b(&mut SplitMix64::new(43), &mut ad);
        let mut set = LoraSet::new();
        set.push(ad.clone(), 1.0);
        set.validate_against_gguf(&gguf)
            .expect("validate LoRA against base GGUF");
        model.bind_lora(&set);
        state.lora_scratch.ensure_rank(4);
        let mut trainer = LlamaTrainer::new(&model, max_ctx, 4, CheckpointMode::Fp16);
        for seq in &seqs {
            state.reset();
            model
                .forward_batch(&mut state, &pool, kt, seq, 0, true)
                .unwrap();
            trainer.forward_last_logits(&pool, Some(&ad), seq, &mut tr_logits);
            for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{arch} 0-ULP mismatch (with LoRA) len={} idx={i}",
                    seq.len()
                );
            }
        }
    }

    // 2. Gemma-family P0 variants (`gemma2` `attn_logit_softcapping`, `gemma3` dual RoPE + no V-Norm)
    for arch in ["gemma2", "gemma3"] {
        let path = dir.join(format!("{arch}_q4.gguf"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&path)
                .arg("--quant")
                .arg("q4_0")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let mut model = Gemma4Model::from_gguf(&gguf).unwrap();
        let mut state = Gemma4State::new(&model, 64);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];

        {
            let mut trainer = Gemma4Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
            for seq in &seqs {
                state.reset();
                model
                    .forward_batch(&mut state, &pool, kt, seq, 0, true)
                    .unwrap();
                trainer.forward_last_logits(&pool, None, seq, &mut tr_logits);
                for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{arch} 0-ULP mismatch (no LoRA) len={} idx={i}",
                        seq.len()
                    );
                }
            }
        }

        let mut ad = init_lora_adapter_gemma4(&model, 4, 8.0, &targets, 55).unwrap();
        randomize_lora_b(&mut SplitMix64::new(56), &mut ad);
        let mut set = LoraSet::new();
        set.push(ad.clone(), 1.0);
        model.bind_lora(&set);
        state.lora_scratch.ensure_rank(4);
        let mut trainer = Gemma4Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
        for seq in &seqs {
            state.reset();
            model
                .forward_batch(&mut state, &pool, kt, seq, 0, true)
                .unwrap();
            trainer.forward_last_logits(&pool, Some(&ad), seq, &mut tr_logits);
            for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{arch} 0-ULP mismatch (with LoRA) len={} idx={i}",
                    seq.len()
                );
            }
        }
    }

    // 3. `lfm3` (`ShortConv` + SWA + Full + `attn_gate` + sandwich norms)
    {
        let path = dir.join("lfm3_q4.gguf");
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg("lfm3")
                .arg("--out-gguf")
                .arg(&path)
                .arg("--quant")
                .arg("q4_0")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let mut model = Lfm2Model::from_gguf(&gguf).unwrap();
        let mut state = Lfm2State::new(&model, 64);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];

        {
            let mut trainer = Lfm2Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
            for seq in &seqs {
                state.reset();
                model
                    .forward_batch(&mut state, &pool, kt, seq, 0, true)
                    .unwrap();
                trainer.forward_last_logits(&pool, None, seq, &mut tr_logits);
                for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "lfm3 0-ULP mismatch (no LoRA) len={} idx={i}",
                        seq.len()
                    );
                }
            }
        }

        let mut ad = init_lora_adapter_lfm2(&model, 4, 8.0, &targets, 77).unwrap();
        randomize_lora_b(&mut SplitMix64::new(78), &mut ad);
        let mut set = LoraSet::new();
        set.push(ad.clone(), 1.0);
        model.bind_lora(&set);
        state.lora_scratch.ensure_rank(4);
        let mut trainer = Lfm2Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
        for seq in &seqs {
            state.reset();
            model
                .forward_batch(&mut state, &pool, kt, seq, 0, true)
                .unwrap();
            trainer.forward_last_logits(&pool, Some(&ad), seq, &mut tr_logits);
            for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "lfm3 0-ULP mismatch (with LoRA) len={} idx={i}",
                    seq.len()
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p0_finite_difference_jacobian_all_architectures() {
    let dir = tmp_dir("fd");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p0_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let targets = LoraTargets::parse_csv("all").unwrap();
    // Sequence length 10 (> sliding_window=3) with masked prompt tokens
    let tokens = [2u32, 7, 13, 4, 19, 8, 25, 11, 6, 17, 3];
    let mask = [0u8, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1];
    let eps = 2e-3f32;

    // 1. `smollm3`, `phi3`, `mistral`
    for arch in ["smollm3", "phi3", "mistral"] {
        let path = dir.join(format!("{arch}_f32.gguf"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&path)
                .arg("--quant")
                .arg("f32")
                .arg("--eval-ctx")
                .arg("64")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let model = LlamaModel::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter(&model, 2, 4.0, &targets, 101).unwrap();
        randomize_lora_b(&mut SplitMix64::new(102), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = LlamaTrainer::new(&model, 64, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(&pool, &ad, &mut grads, &tokens, &mask, 1.0);

        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let a_indices: Vec<usize> = if is_emb {
                vec![7 * 2, 19 * 2 + 1]
            } else {
                vec![0, ad.pairs[&name].a.len() - 1]
            };
            for idx in a_indices {
                let orig = ad.pairs[&name].a[idx];
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig + eps;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig - eps;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig;
                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = grads.grads[&name].da[idx] as f64;
                let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1.0);
                assert!(
                    rel < 5e-4,
                    "{arch} dA mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
            let blen = ad.pairs[&name].b.len();
            for idx in [0usize, blen / 2, blen - 1] {
                let orig = ad.pairs[&name].b[idx];
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig + eps;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig - eps;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig;
                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = grads.grads[&name].db[idx] as f64;
                let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1.0);
                assert!(
                    rel < 5e-4,
                    "{arch} dB mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
        }
    }

    // 2. `gemma2` (`sech^2` chain rule for `attn_logit_softcapping`) & `gemma3`
    for arch in ["gemma2", "gemma3"] {
        let path = dir.join(format!("{arch}_f32.gguf"));
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg(arch)
                .arg("--out-gguf")
                .arg(&path)
                .arg("--quant")
                .arg("f32")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let model = Gemma4Model::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter_gemma4(&model, 2, 4.0, &targets, 201).unwrap();
        randomize_lora_b(&mut SplitMix64::new(202), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Gemma4Trainer::new(&model, 16, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            // Account for Gemma's `sqrt(n_embd) = 11.31x` embedding scale so effective perturbation matches `~2e-3`.
            let step = if is_emb { 1e-3f32 } else { eps };
            let a_indices: Vec<usize> = if is_emb {
                vec![7 * 2, 19 * 2 + 1]
            } else {
                vec![0, ad.pairs[&name].a.len() - 1]
            };
            for idx in a_indices {
                let orig = ad.pairs[&name].a[idx];
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig + step;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig - step;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig;
                let num = (lp - lm) / (2.0 * step as f64);
                let ana = grads.grads[&name].da[idx] as f64;
                let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1.0);
                assert!(
                    rel < 5e-4,
                    "{arch} dA mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
            for idx in [0usize, ad.pairs[&name].b.len() - 1] {
                let orig = ad.pairs[&name].b[idx];
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig + step;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig - step;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig;
                let num = (lp - lm) / (2.0 * step as f64);
                let ana = grads.grads[&name].db[idx] as f64;
                let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1.0);
                assert!(
                    rel < 5e-4,
                    "{arch} dB mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
        }
    }

    // 3. `lfm3` (`w_attn_gate` sigmoid backward + `post_attn_norm` / `post_ffw_norm` + dual RoPE SWA)
    {
        let path = dir.join("lfm3_f32.gguf");
        assert!(
            Command::new(&py)
                .arg(&script)
                .arg("--arch")
                .arg("lfm3")
                .arg("--out-gguf")
                .arg(&path)
                .arg("--quant")
                .arg("f32")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let model = Lfm2Model::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter_lfm2(&model, 2, 4.0, &targets, 301).unwrap();
        randomize_lora_b(&mut SplitMix64::new(302), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Lfm2Trainer::new(&model, 16, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let a_indices: Vec<usize> = if is_emb {
                vec![7 * 2, 19 * 2 + 1]
            } else {
                vec![0, ad.pairs[&name].a.len() - 1]
            };
            for idx in a_indices {
                let orig = ad.pairs[&name].a[idx];
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig + eps;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig - eps;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().a[idx] = orig;
                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = grads.grads[&name].da[idx] as f64;
                let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1.0);
                assert!(
                    rel < 5e-4,
                    "lfm3 dA mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
            for idx in [0usize, ad.pairs[&name].b.len() - 1] {
                let orig = ad.pairs[&name].b[idx];
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig + eps;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig - eps;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig;
                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = grads.grads[&name].db[idx] as f64;
                let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1.0);
                assert!(
                    rel < 5e-4,
                    "lfm3 dB mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
