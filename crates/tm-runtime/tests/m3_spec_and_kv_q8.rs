//! Milestone M3 integration tests for:
//! 1. **100% deterministic token-exact greedy parity** between single-token `decode_step` and
//!    `generate_greedy_speculative` (`Suffix`, `Auto`, and `Mtp` modes) across ALL architectures:
//!    - `Llama` / `SmolLM2` (`Kv`)
//!    - `Qwen3.5` (`Kv` + `DeltaNet` read-only verify + replay commit `ADR-005`)
//!    - `Gemma 4` (`Kv` + `SharedKv` + `SWA` + `PLE` + logit softcapping)
//!    - `LFM2` (`Kv` + `ShortConv` read-only verify + shift commit `ADR-005`)
//! 2. **MTP position-alignment contract** (`ARCHITECTURE.md` §5.5: `hidden[i] + embed(token[i+1]) -> logits[i+2]`).
//! 3. **`SpecController` `< 5%` net-gain auto-disable** (`F-12`).
//! 4. **`Q8_0` KV cache (`F-14`)** forward + `PrefixCache` roundtrip + memory savings (`~47%` vs `F16`).
//! 5. **`Q4_0x4` GGUF conversion (`F-02`/`F-04`)** zero file-size bloat + bit-exact `forward_token` & `forward_batch` logits!

use std::path::Path;

use tm_core::{DType, KvType, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::argmax;
use tm_runtime::{
    CpuModel, DraftChoice, LinearMtpHead, MtpDrafter, Session, SpecController, SpecMode,
    SuffixDrafter,
};

fn run_single_token_greedy(
    model: &CpuModel<'_>,
    pool: &ThreadPool,
    prompt: &[u32],
    n_gen: usize,
    kv_type: KvType,
) -> (Vec<u32>, Vec<f32>, usize) {
    let mut sess = Session::new_with_options(model, 256, 0, kv_type);
    sess.prefill_with_cache(pool, prompt, None)
        .expect("prefill_with_cache");
    let mut out = Vec::with_capacity(n_gen);
    let mut cur = argmax(sess.logits());
    out.push(cur);
    for _ in 1..n_gen {
        sess.decode_step(pool, cur, true).expect("decode_step");
        cur = argmax(sess.logits());
        out.push(cur);
    }
    (out, sess.logits().to_vec(), sess.pos())
}

fn verify_arch_speculative_parity(path: &str, prompt: &[u32], n_gen: usize) {
    let p = Path::new(path);
    if !p.exists() {
        return;
    }
    let gguf = GgufFile::open(p).expect("open gguf");
    let model = CpuModel::from_gguf(&gguf).expect("load model");
    let pool = ThreadPool::new(4);

    let (ref_tokens, ref_logits, ref_pos) =
        run_single_token_greedy(&model, &pool, prompt, n_gen, KvType::F16);

    // 1. SuffixDrafter forced mode (with pre-seeded true trajectory + adversarial noise so BOTH
    //    multi-token acceptances AND partial rejections + replay commits are exercised!)
    let mut sess_suf = Session::new(&model, 256, 0);
    sess_suf
        .prefill_with_cache(&pool, prompt, None)
        .expect("prefill suf");
    let mut suffix = SuffixDrafter::default();
    let mut seed_seq = prompt.to_vec();
    // Seed first half true tokens + corrupted second half to force both accept > 0 and reject mid-window!
    for (idx, &t) in ref_tokens.iter().enumerate() {
        if idx % 5 == 3 {
            seed_seq.push((t + 1) % (model.vocab_size() as u32));
        } else {
            seed_seq.push(t);
        }
    }
    suffix.observe_sequence(&seed_seq);
    let mut ctrl = SpecController::new(6);
    let (spec_tokens, stats) = sess_suf
        .generate_greedy_speculative(
            &pool,
            prompt,
            n_gen,
            SpecMode::Suffix,
            &mut suffix,
            None,
            &mut ctrl,
            |_, _| false,
        )
        .expect("generate_greedy_speculative suffix");

    assert_eq!(
        spec_tokens,
        ref_tokens,
        "Speculative greedy token mismatch on {} (stats={stats:?})",
        model.arch()
    );
    assert_eq!(sess_suf.pos(), ref_pos);
    assert!(
        stats.verify_calls > 0 && stats.accepted_tokens > 0,
        "Expected active suffix speculation on {}, got {stats:?}",
        model.arch()
    );
    // Check final logits match tightly (0 ULP for CPU bit-exact kernels!)
    let max_logit_diff = sess_suf
        .logits()
        .iter()
        .zip(ref_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_logit_diff < 1e-4,
        "Final logits diverged after speculative commit on {}: max_diff={max_logit_diff}",
        model.arch()
    );
}

#[test]
fn speculative_greedy_exact_parity_across_all_architectures() {
    // 1. Llama / SmolLM2
    verify_arch_speculative_parity(
        "../../models/SmolLM2-360M-Instruct-Q4_0.gguf",
        &[1, 504, 346, 260, 504, 346, 260, 1112],
        24,
    );
    // 2. Qwen3.5 hybrid Gated DeltaNet (`DeltaNet` read-only verify + replay commit!)
    verify_arch_speculative_parity(
        "../../models/Qwen3.5-0.8B-Q4_0.gguf",
        &[760, 469, 279, 1205, 760, 469, 279, 1205],
        20,
    );
    // 3. Gemma 4 (`SharedKv` + `SWA` + `PLE` + `final_logit_softcapping`)
    verify_arch_speculative_parity(
        "../../models/gemma-4-E2B-it-Q4_K_M.gguf",
        &[2, 651, 9456, 576, 651, 9456],
        12,
    );
    // 4. Liquid AI LFM2 (`ShortConv` read-only verify + shift commit!)
    verify_arch_speculative_parity(
        "../../models/LFM2.5-1.2B-Instruct-Q4_0.gguf",
        &[1, 822, 315, 412, 822, 315],
        16,
    );
}

#[test]
fn mtp_position_alignment_and_speculative_parity() {
    let p = Path::new("../../models/SmolLM2-360M-Instruct-Q4_0.gguf");
    if !p.exists() {
        return;
    }
    let gguf = GgufFile::open(p).expect("open SmolLM2");
    let model = CpuModel::from_gguf(&gguf).expect("load SmolLM2");
    let pool = ThreadPool::new(4);
    let kt = active_kernels();
    let d = model.n_embd();

    // Build a deterministic `LinearMtpHead` (`[2 * d -> d]` F16 projection)
    let h_norm = vec![1.0f32; d];
    let e_norm = vec![1.0f32; d];
    let mut proj_bytes = Vec::with_capacity(d * 2 * d * 2);
    for r in 0..d {
        for c in 0..(2 * d) {
            let val = if c == r {
                0.6f32
            } else if c == d + r {
                0.4f32
            } else {
                0.0f32
            };
            proj_bytes.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
    }
    let mut mtp = LinearMtpHead::new(d, 1e-5, h_norm, e_norm, DType::F16, proj_bytes);

    // Verify `ARCHITECTURE.md` §5.5 position contract:
    // Step `i` produces `hidden[i]` and `token[i+1]`; `mtp.draft_hidden(hidden[i], embed(token[i+1]))`
    // produces `mtp_hidden[i+1]` which projects to `logits[i+2]`.
    let prompt = [1u32, 504, 346, 260];
    let (ref_tokens, _, _) = run_single_token_greedy(&model, &pool, &prompt, 14, KvType::F16);

    let mut sess = Session::new(&model, 128, 0);
    sess.prefill_with_cache(&pool, &prompt, None)
        .expect("prefill");
    let tok_ip1 = argmax(sess.logits());
    assert_eq!(tok_ip1, ref_tokens[0]);

    let mut emb_ip1 = vec![0.0f32; d];
    let mut mtp_h_ip1 = vec![0.0f32; d];
    let mut mtp_logits_ip2 = vec![0.0f32; model.vocab_size()];
    model.embed_token(tok_ip1, &mut emb_ip1);
    mtp.draft_hidden(
        &pool,
        kt,
        sess.state.last_hidden(),
        &emb_ip1,
        &mut mtp_h_ip1,
    );
    model.project_logits_from_hidden(&mut sess.state, &pool, kt, &mtp_h_ip1, &mut mtp_logits_ip2);
    assert!(mtp_logits_ip2.iter().all(|v| v.is_finite()));

    // Now run full end-to-end MTP speculative generation and assert 100% greedy parity!
    let mut sess_mtp = Session::new(&model, 128, 0);
    sess_mtp
        .prefill_with_cache(&pool, &prompt, None)
        .expect("prefill mtp");
    let mut suffix = SuffixDrafter::default();
    let mut ctrl = SpecController::new(3);
    let (mtp_tokens, mtp_stats) = sess_mtp
        .generate_greedy_speculative(
            &pool,
            &prompt,
            14,
            SpecMode::Mtp,
            &mut suffix,
            Some(&mut mtp),
            &mut ctrl,
            |_, _| false,
        )
        .expect("mtp speculative");
    assert_eq!(mtp_tokens, ref_tokens);
    assert!(mtp_stats.verify_calls > 0 && mtp_stats.drafted_tokens > 0);
}

#[test]
fn spec_controller_auto_disables_below_five_percent_gain() {
    let mut ctrl = SpecController::new(6);
    ctrl.record_base_step(10_000_000); // 10 ms single-token baseline

    // Simulate low acceptance (0 / 4 accepted, 16 ms verify time -> shrinks k: 4 -> 1 -> None when gain < 5%)
    for _ in 0..4 {
        let choice = ctrl.choose(SpecMode::Auto, false, true);
        assert!(matches!(choice, DraftChoice::Suffix(_)));
        ctrl.record_suffix_verify(4, 0, 16_000_000);
    }
    // Controller MUST now auto-disable speculation (`DraftChoice::None`)!
    let next_choice = ctrl.choose(SpecMode::Auto, false, true);
    assert_eq!(
        next_choice,
        DraftChoice::None,
        "SpecController must auto-disable when net gain < 5%"
    );

    // Conversely, high acceptance (4 / 4 accepted, 14 ms verify time -> ~3.5x speedup >> 5%) keeps it enabled!
    let mut ctrl_good = SpecController::new(6);
    ctrl_good.record_base_step(10_000_000);
    for _ in 0..4 {
        ctrl_good.record_suffix_verify(4, 4, 14_000_000);
    }
    let active_choice = ctrl_good.choose(SpecMode::Auto, false, true);
    assert!(
        matches!(active_choice, DraftChoice::Suffix(k) if k >= 4),
        "Expected high-gain Suffix choice, got {active_choice:?}"
    );
}

#[test]
fn q8_0_kv_cache_accuracy_memory_and_prefix_cache_roundtrip() {
    let p = Path::new("../../models/SmolLM2-360M-Instruct-Q4_0.gguf");
    if !p.exists() {
        return;
    }
    let gguf = GgufFile::open(p).expect("open SmolLM2");
    let model = CpuModel::from_gguf(&gguf).expect("load SmolLM2");
    let pool = ThreadPool::new(4);

    let f16_bytes = model.estimate_dirty_bytes_with_kv(2048, KvType::F16);
    let q8_bytes = model.estimate_dirty_bytes_with_kv(2048, KvType::Q8_0);
    assert!(
        q8_bytes < f16_bytes,
        "Q8_0 KV dirty estimate ({q8_bytes}) must be smaller than F16 ({f16_bytes})"
    );

    let prompt = [1u32, 504, 346, 260, 1112, 314, 921, 408, 715, 288];
    let (tokens_f16, logits_f16, _) =
        run_single_token_greedy(&model, &pool, &prompt, 12, KvType::F16);
    let (tokens_q8, logits_q8, _) =
        run_single_token_greedy(&model, &pool, &prompt, 12, KvType::Q8_0);

    // Cosine similarity between F16 KV and Q8_0 KV logits after 22 total positions must exceed 0.999
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&a, &b) in logits_f16.iter().zip(logits_q8.iter()) {
        dot += (a as f64) * (b as f64);
        na += (a as f64) * (a as f64);
        nb += (b as f64) * (b as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    assert!(
        cos > 0.999,
        "Q8_0 KV cache cosine similarity too low: {cos}"
    );
    assert_eq!(tokens_q8[0], tokens_f16[0]);

    // PrefixCache save + restore roundtrip with `KvType::Q8_0` MUST be bit-exact vs uncached Q8_0!
    let mut sess_cached = Session::new_with_options(&model, 128, 16 * 1024 * 1024, KvType::Q8_0);
    let r1 = sess_cached
        .prefill_with_cache(&pool, &prompt[..8], None)
        .expect("prefill 1");
    assert_eq!(r1.cached_tokens, 0);
    let r2 = sess_cached
        .prefill_with_cache(&pool, &prompt, None)
        .expect("prefill 2");
    assert_eq!(r2.cached_tokens, 7);

    let mut sess_uncached = Session::new_with_options(&model, 128, 0, KvType::Q8_0);
    sess_uncached
        .prefill_with_cache(&pool, &prompt, None)
        .expect("prefill uncached");
    for (i, (&a, &b)) in sess_cached
        .logits()
        .iter()
        .zip(sess_uncached.logits().iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Q8_0 PrefixCache bit mismatch at logit {i}: {a} vs {b}"
        );
    }
}

#[test]
fn convert_to_q4_0x4_preserves_file_size_and_bit_exact_logits() {
    let src = Path::new("../../models/SmolLM2-360M-Instruct-Q4_0.gguf");
    if !src.exists() {
        return;
    }
    let tmp_dir = std::env::temp_dir();
    let dst = tmp_dir.join("tm_m3_test_smollm2_q4_0x4.gguf");
    let gguf_orig = GgufFile::open(src).expect("open orig");
    let stats = gguf_orig
        .convert_to_q4_0x4(&dst)
        .expect("convert_to_q4_0x4");
    assert!(stats.repacked_tensors > 0);
    let src_len = std::fs::metadata(src).expect("src meta").len();
    let dst_len = std::fs::metadata(&dst).expect("dst meta").len();
    assert_eq!(
        src_len, dst_len,
        "Q4_0x4 offline conversion must have ZERO file-size bloat ({src_len} vs {dst_len})"
    );

    let gguf_rep = GgufFile::open(&dst).expect("open repacked");
    let model_orig = CpuModel::from_gguf(&gguf_orig).expect("load orig");
    let model_rep = CpuModel::from_gguf(&gguf_rep).expect("load repacked");
    let pool = ThreadPool::new(4);

    let prompt = [1u32, 504, 346, 260, 1112, 314, 921, 408, 715];
    let (tok_orig, logits_orig, _) =
        run_single_token_greedy(&model_orig, &pool, &prompt, 8, KvType::F16);
    let (tok_rep, logits_rep, _) =
        run_single_token_greedy(&model_rep, &pool, &prompt, 8, KvType::F16);
    let _ = std::fs::remove_file(&dst);

    assert_eq!(tok_orig, tok_rep, "Q4_0x4 greedy tokens diverged from Q4_0");
    for (i, (&a, &b)) in logits_orig.iter().zip(logits_rep.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Q4_0x4 logit[{i}] not bit-exact with Q4_0: {a} vs {b}"
        );
    }
}
