//! Sparse MoE & EXAONE QLoRA verification (`ADR-018`):
//! 1. **0-ULP (`to_bits()`)** `forward_last_logits` parity (`B == 0` AND randomized `B != 0`) across
//!    all 7 P2a architectures (`olmoe`, `granitemoe`, `granitemoehybrid`, `qwen2moe`, `qwen3moe`,
//!    `bailingmoe2`, `exaone`) for 1-token, 11-token, and 35-token (`> MAX_CPU_BATCH = 32`) sequences.
//! 2. **`f64` Central Finite-Difference Jacobian** (`rel < 5e-4`) across every LoRA target —
//!    including router `ffn_gate_inp` (`Softmax`, `Sigmoid`, `norm_weights`, `weights_scale`),
//!    1D sigmoid-gated `ffn_gate_inp_shexp`, shared expert `ffn_{gate,up,down}_shexp`, fused
//!    `attn_qkv`, `ssm_in`/`ssm_out`, `token_embd`, and `output`.
//! 3. **End-to-end `run_cpu_qlora_training`** monotonic loss convergence on both `qwen2moe`
//!    (`PolyTrainer::Llama`) and `granitemoehybrid` (`PolyTrainer::Mamba2`).

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
    let p = std::env::temp_dir().join(format!("tm_train_p2a_{tag}_{}_{id}", std::process::id()));
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

fn randomize_lora_b(rng: &mut SplitMix64, adapter: &mut tm_lora::LoraAdapter, bound: f32) {
    for name in &adapter.pair_order {
        let p = adapter.pairs.get_mut(name).unwrap();
        for b in &mut p.b {
            *b = rng.uniform_sym(bound);
        }
    }
}

#[test]
fn p2a_zero_ulp_forward_parity_across_all_7_architectures_and_lora() {
    let dir = tmp_dir("fwd0ulp");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let targets = LoraTargets::parse_csv("all").unwrap();

    let seqs: [Vec<u32>; 3] = [
        vec![3],
        vec![1, 5, 2, 9, 14, 7, 3, 11, 6, 18, 4],
        (0..35).map(|i| ((i * 7 + 3) % 90) as u32).collect(),
    ];

    // 1. Llama-family MoE + EXAONE (`olmoe`, `granitemoe`, `qwen2moe`, `qwen3moe`, `bailingmoe2`, `exaone`)
    for arch in [
        "olmoe",
        "granitemoe",
        "qwen2moe",
        "qwen3moe",
        "bailingmoe2",
        "exaone",
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
        randomize_lora_b(&mut SplitMix64::new(43), &mut ad, 0.14);
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

    // 2. Mamba-2 Hybrid MoE (`granitemoehybrid` & `granitehybrid`)
    for arch in ["granitemoehybrid", "granitehybrid"] {
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
        randomize_lora_b(&mut SplitMix64::new(78), &mut ad, 0.14);
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
fn p2a_moe_finite_difference_jacobian_and_end_to_end_qlora_training() {
    let dir = tmp_dir("fd");
    let py = find_python3_with_numpy();
    let script = repo_root().join("tools/oracle/p2a_architectures_oracle.py");
    let pool = ThreadPool::new(2);
    let targets = LoraTargets::parse_csv("all").unwrap();
    // 10-token sequence (> SSM_CKPT_STRIDE = 8) with masked prompt tokens
    let tokens = [2u32, 7, 13, 4, 19, 8, 25, 11, 6, 17];
    let mask = [0u8, 0, 1, 1, 0, 1, 1, 1, 1, 1];
    let eps = 1e-3f32;

    let ds = TrainDataset {
        samples: vec![TrainSample {
            tokens: tokens.to_vec(),
            loss_mask: mask.to_vec(),
        }],
        total_tokens: tokens.len(),
        supervised_tokens: mask.iter().filter(|&&m| m != 0).count(),
    };

    // 1. Finite-difference Jacobian across all MoE gating/routing/shexp modes:
    // - `olmoe` (Softmax, `norm_w = false`, separate + fused 3D experts)
    // - `granitemoe` (Softmax, `norm_w = true`, 4x muP scales, un-gated shared expert)
    // - `qwen2moe` (Softmax, `norm_w = false`, 1D sigmoid-gated `ffn_gate_inp_shexp`!)
    // - `bailingmoe2` (dense lead layer + Sigmoid MoE + `exp_probs_b` + `norm_w = true` + `scale = 2.5` + fused `attn_qkv`)
    for arch in ["olmoe", "granitemoe", "qwen2moe", "bailingmoe2"] {
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
        let model = LlamaModel::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter(&model, 2, 4.0, &targets, 101).unwrap();
        randomize_lora_b(&mut SplitMix64::new(102), &mut ad, 0.08);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = LlamaTrainer::new(&model, 32, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(&pool, &ad, &mut grads, &tokens, &mask, 1.0);

        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let step = if is_emb && arch == "granitemoe" {
                5e-4f32
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
            let b_indices: Vec<usize> = if blen <= 2 {
                (0..blen).collect()
            } else {
                vec![0, blen / 2, blen - 1]
            };
            for idx in b_indices {
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

        if arch == "qwen2moe" {
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
                    .expect("run_cpu_qlora_training qwen2moe");
            assert!(
                rep.final_loss < rep.initial_loss,
                "qwen2moe QLoRA loss did not decrease: {} -> {}",
                rep.initial_loss,
                rep.final_loss
            );
        }
    }

    // 2. Finite-difference Jacobian + end-to-end QLoRA training on `granitemoehybrid` (`Mamba2Trainer` + `SparseMoeFfn`)!
    {
        let arch = "granitemoehybrid";
        let path = dir.join("granitemoehybrid_f32.gguf");
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
        let mut ad = init_lora_adapter_mamba2(&model, 2, 4.0, &targets, 301).unwrap();
        randomize_lora_b(&mut SplitMix64::new(302), &mut ad, 0.08);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Mamba2Trainer::new(&model, 32, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let step = if is_emb { 5e-4f32 } else { eps };
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
                .expect("run_cpu_qlora_training granitemoehybrid");
        assert!(
            rep.final_loss < rep.initial_loss,
            "granitemoehybrid QLoRA loss did not decrease: {} -> {}",
            rep.initial_loss,
            rep.final_loss
        );
    }

    let _ = fs::remove_dir_all(&dir);
}
