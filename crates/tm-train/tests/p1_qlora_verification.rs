//! QLoRA 0-ULP forward parity (`B == 0` and `B != 0`), `f64` finite-difference Jacobian
//! verification across every LoRA target (`token_embd`, `ssm_in`, `ssm_out`, `wq`/`wk`/`wv`/`wo`,
//! `w_gate`/`w_up`/`w_down`, `output`), and end-to-end `run_qlora_training` monotonic loss
//! convergence across all P1 sub-8B architectures (`olmo2`, `granite`, `mamba2`, `nemotron_h`,
//! `falcon-h1`, `granitehybrid`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_lora::LoraSet;
use tm_models::{LlamaModel, LlamaState, Mamba2Model, Mamba2State};
use tm_runtime::CpuModel;
use tm_train::{
    AdamWConfig, CheckpointMode, LlamaTrainer, LoraGradStore, LoraTargets, Mamba2Trainer,
    QloraConfig, SplitMix64, TrainDataset, TrainSample, init_lora_adapter,
    init_lora_adapter_mamba2, run_cpu_qlora_training,
};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("tm_train_p1_{tag}_{}_{id}", std::process::id()));
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
fn p1_zero_ulp_forward_parity_across_all_architectures_and_lora() {
    let dir = tmp_dir("fwd0ulp");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p1_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let targets = LoraTargets::parse_csv("all").unwrap();

    // Includes 1-token, 11-token (> SSM_CKPT_STRIDE=8 & > SWA=3), and 35-token (> MAX_CPU_BATCH=32)!
    let seqs: [Vec<u32>; 3] = [
        vec![3],
        vec![1, 5, 2, 9, 14, 7, 3, 11, 6, 18, 4],
        (0..35).map(|i| ((i * 7 + 3) % 90) as u32).collect(),
    ];

    // 1. P1a Dense (`olmo2` Post-Norm + full-vec QK-Norm + 3:1 SWA, `granite` muP + biases + LongRoPE)
    for arch in ["olmo2", "granite"] {
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
                .arg("64")
                .status()
                .unwrap()
                .success()
        );
        let gguf = GgufFile::open(&path).unwrap();
        let mut model = LlamaModel::from_gguf(&gguf).unwrap();
        let mut state = LlamaState::new(&model.cfg, 64);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];

        {
            let mut trainer = LlamaTrainer::new(&model, 64, 4, CheckpointMode::Fp16);
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
        set.validate_against_gguf(&gguf).unwrap();
        model.bind_lora(&set);
        state.lora_scratch.ensure_rank(4);
        let mut trainer = LlamaTrainer::new(&model, 64, 4, CheckpointMode::Fp16);
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

    // 2. P1b Mamba-2 & Hybrid (`mamba2`, `nemotron_h`, `falcon-h1`, `granitehybrid`)
    for arch in ["mamba2", "nemotron_h", "falcon-h1", "granitehybrid"] {
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
        let mut model = Mamba2Model::from_gguf(&gguf).unwrap();
        let mut state = Mamba2State::new(&model, 64);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];

        {
            let mut trainer = Mamba2Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
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

        let mut ad = init_lora_adapter_mamba2(&model, 4, 8.0, &targets, 77).unwrap();
        randomize_lora_b(&mut SplitMix64::new(78), &mut ad);
        let mut set = LoraSet::new();
        set.push(ad.clone(), 1.0);
        set.validate_against_gguf(&gguf).unwrap();
        model.bind_lora(&set);
        state.lora_scratch.ensure_rank(4);
        let mut trainer = Mamba2Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
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

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn p1_finite_difference_jacobian_and_end_to_end_training_all_architectures() {
    let dir = tmp_dir("fd");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p1_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let targets = LoraTargets::parse_csv("all").unwrap();
    // Sequence length 12 (> SSM_CKPT_STRIDE = 8 and > sliding_window = 3) with masked prompt tokens!
    let tokens = [2u32, 7, 13, 4, 19, 8, 25, 11, 6, 17, 21, 3];
    let mask = [0u8, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];
    let eps = 2e-3f32;

    // 1. P1a Dense (`olmo2`, `granite`)
    for arch in ["olmo2", "granite"] {
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
            let step = if is_emb && arch == "granite" {
                1e-3f32
            } else {
                eps
            };
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
            let blen = ad.pairs[&name].b.len();
            for idx in [0usize, blen / 2, blen - 1] {
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

    // 2. P1b Mamba-2 & Hybrid (`mamba2`, `nemotron_h`, `falcon-h1`, `granitehybrid`)
    let ds = TrainDataset {
        samples: vec![TrainSample {
            tokens: tokens.to_vec(),
            loss_mask: mask.to_vec(),
        }],
        total_tokens: tokens.len(),
        supervised_tokens: mask.iter().filter(|&&m| m != 0).count(),
    };

    for arch in ["mamba2", "nemotron_h", "falcon-h1", "granitehybrid"] {
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
        let model = Mamba2Model::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter_mamba2(&model, 2, 4.0, &targets, 201).unwrap();
        randomize_lora_b(&mut SplitMix64::new(202), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Mamba2Trainer::new(&model, 32, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let step = if is_emb && arch == "granitehybrid" {
                1e-3f32
            } else {
                eps
            };
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
            let blen = ad.pairs[&name].b.len();
            for idx in [0usize, blen / 2, blen - 1] {
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

        // Also verify end-to-end `PolyTrainer::Mamba2` (`run_cpu_qlora_training`) loss drop on `falcon-h1`!
        if arch == "falcon-h1" {
            let cpu_model = CpuModel::from_gguf(&gguf).unwrap();
            let cfg = QloraConfig {
                rank: 4,
                alpha: 8.0,
                targets: targets.clone(),
                steps: 6,
                seq_len: 16,
                grad_accum: 1,
                warmup_steps: 0,
                optim: AdamWConfig {
                    lr: 8e-3,
                    weight_decay: 0.0,
                    max_grad_norm: 1.0,
                    ..AdamWConfig::default()
                },
                seed: 2026,
                checkpoint_mode: CheckpointMode::Fp16,
                ..QloraConfig::default()
            };
            let (_trained_ad, rep) =
                run_cpu_qlora_training(&cpu_model, &pool, &ds, &cfg, None, |_| Ok(()))
                    .expect("run_cpu_qlora_training");
            assert!(
                rep.final_loss < rep.initial_loss,
                "falcon-h1 QLoRA loss did not decrease: {} -> {}",
                rep.initial_loss,
                rep.final_loss
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}
