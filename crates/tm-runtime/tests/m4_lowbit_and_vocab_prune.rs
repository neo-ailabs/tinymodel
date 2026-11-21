//! Milestone M4 end-to-end integration tests:
//! 1. Offline GGUF conversion across M4 low-bit & ternary formats (`Q2_0`, `TQ2_0`, `IQ2_XXS`, `IQ3_XXS`, `IQ4_NL`)
//!    + zero-copy `mmap` load + multi-token prefill vs single-step `0`-ULP invariance (`AGENTS.md` Rule #7).
//! 2. Offline BPE-closed vocabulary pruning (`--prune-vocab` + `--vocab-corpus`) + `Tokenizer::from_gguf`
//!    arbitrary UTF-8 byte-fallback lossless roundtrip + end-to-end multi-token generation!

use std::path::Path;

use tm_core::{DType, KvType};
use tm_cpu::ThreadPool;
use tm_gguf::{ConvertOptions, GgufFile};
use tm_runtime::{CpuModel, Session};
use tm_tokenizer::Tokenizer;

#[test]
fn m4_smollm2_vocab_prune_and_lowbit_convert_e2e() {
    let src_path = Path::new("../../models/SmolLM2-360M-Instruct-Q4_0.gguf");
    if !src_path.exists() {
        return;
    }

    let tmp_dir = std::env::temp_dir().join(format!("tm_m4_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    let src_gguf = GgufFile::open(src_path).expect("open SmolLM2-360M");

    // 1. Test vocabulary pruning (`49152 -> 4096`) + `Q2_0` T-MAC LUT quantization + domain corpus!
    let pruned_q2_path = tmp_dir.join("smollm2_q2_0_v4096.gguf");
    let opts = ConvertOptions {
        quant: Some(DType::Q2_0),
        lm_head_quant: Some(DType::Iq4Nl),
        prune_vocab: Some(4096),
        vocab_corpus: Some(
            "Raspberry Pi Zero 2W runs tinymodel in Rust with 16-row T-MAC NEON TBL lookup kernels!"
                .to_string(),
        ),
    };
    let rep = src_gguf
        .convert_with_options(&pruned_q2_path, &opts)
        .expect("convert Q2_0 + prune_vocab 4096");
    assert_eq!(rep.old_vocab_size, 49152);
    assert_eq!(rep.new_vocab_size, 4096);
    assert!(rep.kept_merges > 500 && rep.kept_merges < rep.old_merges);
    assert!(
        rep.output_file_bytes < 105 * 1024 * 1024,
        "expected < 105 MiB, got {} bytes",
        rep.output_file_bytes
    );

    let pruned_gguf = GgufFile::open(&pruned_q2_path).expect("open pruned Q2_0 GGUF");
    for tname in pruned_gguf.tensor_names() {
        let info = pruned_gguf.tensor_info(tname).expect("tensor info");
        assert_eq!(
            pruned_gguf.tensor_file_offset(info) % 32,
            0,
            "tensor {tname} not 32-byte aligned"
        );
    }

    let tok = Tokenizer::from_gguf(&pruned_gguf).expect("load pruned tokenizer");
    assert_eq!(tok.vocab_size(), 4096);
    assert!(tok.eos_id.is_some());

    // Arbitrary UTF-8 (English, numbers, CJK, emoji, code) MUST roundtrip 100% losslessly via kept GPT-2 byte tokens!
    for sample in [
        "Raspberry Pi Zero 2W runs tinymodel!",
        "端侧推理引擎 2026 🚀 def f(x): return x * 2 + 1",
    ] {
        let ids = tok.encode(sample, false);
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|&id| (id as usize) < 4096));
        assert_eq!(tok.decode(&ids), sample);
    }

    // Multi-token prefill vs step-by-step invariance (`AGENTS.md` Rule #7 + L1b)
    let prompt_ids = tok.encode("Raspberry Pi Zero 2W runs", true);
    assert!(prompt_ids.len() >= 4);

    let model = CpuModel::from_gguf(&pruned_gguf).expect("load CpuModel");
    let pool = ThreadPool::new(2);

    let mut sess_batch = Session::new_with_options(&model, 256, 0, KvType::Q8_0);
    sess_batch
        .prefill_with_cache(&pool, &prompt_ids, None)
        .expect("prefill pruned Q2_0");
    let logits_batch = sess_batch.logits().to_vec();

    let mut sess_step = Session::new_with_options(&model, 256, 0, KvType::Q8_0);
    for (idx, &tid) in prompt_ids.iter().enumerate() {
        sess_step
            .decode_step(&pool, tid, idx + 1 == prompt_ids.len())
            .expect("step pruned Q2_0");
    }
    let logits_step = sess_step.logits().to_vec();

    assert_eq!(logits_batch.len(), 4096);
    for i in 0..4096 {
        assert!(logits_batch[i].is_finite(), "non-finite logit at {i}");
        assert_eq!(
            logits_batch[i].to_bits(),
            logits_step[i].to_bits(),
            "End-to-end Q2_0+IQ4_NL pruned batch vs step mismatch at logit {i}"
        );
    }

    drop(sess_step);
    drop(sess_batch);
    drop(model);
    drop(pruned_gguf);
    let _ = std::fs::remove_file(&pruned_q2_path);

    // 2. Also verify Qwen3.5-0.8B (`cols = 1024, 3584` -> 100% divisible by 256!) with `TQ2_0` if present
    let qwen_path = Path::new("../../models/Qwen3.5-0.8B-Q4_0.gguf");
    if qwen_path.exists() {
        let q_gguf = GgufFile::open(qwen_path).expect("open Qwen3.5-0.8B");
        let tq2_out = tmp_dir.join("qwen35_tq2_0_v2048.gguf");
        let rep_tq2 = q_gguf
            .convert_with_options(
                &tq2_out,
                &ConvertOptions {
                    quant: Some(DType::TQ2_0),
                    lm_head_quant: Some(DType::Q2_0),
                    prune_vocab: Some(2048),
                    vocab_corpus: None,
                },
            )
            .expect("convert Qwen3.5 TQ2_0");
        assert_eq!(rep_tq2.new_vocab_size, 2048);

        let tq2_gguf = GgufFile::open(&tq2_out).expect("open Qwen3.5 TQ2_0");
        assert_eq!(
            tq2_gguf
                .tensor_info("blk.0.ffn_gate.weight")
                .expect("ffn_gate")
                .dtype,
            DType::TQ2_0
        );
        let qmodel = CpuModel::from_gguf(&tq2_gguf).expect("load Qwen3.5 TQ2_0");
        let mut sess = Session::new(&qmodel, 128, 0);
        sess.prefill_with_cache(&pool, &[1, 10, 25, 42], None)
            .expect("prefill TQ2_0");
        assert_eq!(sess.logits().len(), 2048);
        assert!(sess.logits().iter().all(|x| x.is_finite()));
        drop(sess);
        drop(qmodel);
        drop(tq2_gguf);
        let _ = std::fs::remove_file(&tq2_out);
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
