//! End-to-end multi-token parity, 3-tier `O_DIRECT` placement, `PrefixCache`, and speculative
//! decoding tests for `Inkling-2-Mini` (`general.architecture == "inkling2_loop"`).

use std::path::PathBuf;
use std::process::Command;

use tm_core::KvType;
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{Inkling2LoadOptions, Inkling2Model, Inkling2State, argmax};
use tm_runtime::{CpuModel, Session, SpecController, SpecMode, SuffixDrafter};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn read_tmlg_f32(path: &std::path::Path) -> (Vec<u32>, usize, Vec<Vec<f32>>) {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(&raw[0..4], b"TMLG");
    let ver = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    assert_eq!(ver, 1);
    let n_steps = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let n_vocab = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let mut off = 16;
    let mut toks = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        toks.push(u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    let mut mat = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let mut row = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            row.push(f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        mat.push(row);
    }
    assert_eq!(off, raw.len());
    (toks, n_vocab, mat)
}

fn find_python3_with_numpy() -> String {
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("PYTHON3") {
        candidates.push(p);
    }
    candidates.extend([
        "python3".to_string(),
        "/opt/anaconda3/bin/python3".to_string(),
        "/opt/homebrew/bin/python3".to_string(),
        "/usr/local/bin/python3".to_string(),
    ]);
    for bin in candidates {
        if let Ok(st) = Command::new(&bin)
            .args(["-c", "import numpy"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            && st.success()
        {
            return bin;
        }
    }
    "python3".to_string()
}

#[test]
fn inkling2_loop_matches_fp64_python_oracle_and_tiered_nvme_streaming_bit_exact() {
    let root = repo_root();
    let tmp_dir = std::env::temp_dir().join(format!("tm_inkling2_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let gguf_path = tmp_dir.join("synthetic_inkling2_loop_q8_0.gguf");
    let ref_path = tmp_dir.join("ref.tmlg");

    let status = Command::new(find_python3_with_numpy())
        .arg(root.join("tools/oracle/inkling2_oracle.py"))
        .arg("--gen-gguf")
        .arg(&gguf_path)
        .arg("--tokens")
        .arg("5,12,29,41,7,63,18")
        .arg("--out")
        .arg(&ref_path)
        .status()
        .expect("failed to invoke inkling2_oracle.py");
    assert!(status.success(), "inkling2_oracle.py failed");

    let (toks, ref_vocab, ref_logits) = read_tmlg_f32(&ref_path);
    assert_eq!(toks.len(), 7);
    assert_eq!(ref_vocab, 70);

    let gguf = GgufFile::open(&gguf_path).unwrap();
    let pool = ThreadPool::new(4);
    let pool_1t = ThreadPool::new(1);
    let kt = active_kernels();

    // 1. Zero-copy mmap mode: compare multi-token logits against FP64 Python Oracle (`AGENTS.md` Rule #7)
    let model_mmap = Inkling2Model::from_gguf(&gguf).unwrap();
    assert_eq!(model_mmap.cfg.vocab_size, 70);
    assert_eq!(model_mmap.cfg.vocab_padded, 96);
    assert_eq!(model_mmap.cfg.n_layers, 16);
    assert_eq!(model_mmap.stats.ram_mmap_expert_layers, 8);
    assert_eq!(model_mmap.stats.nvme_direct_expert_layers, 0);

    let mut state_mmap = Inkling2State::new(&model_mmap, 64);
    let mut mmap_logits_steps = Vec::with_capacity(toks.len());
    for (step, &tid) in toks.iter().enumerate() {
        model_mmap
            .forward_token(&mut state_mmap, &pool, kt, tid, step, true)
            .unwrap();
        let got = state_mmap.logits().to_vec();
        let want = &ref_logits[step];
        assert_eq!(
            argmax(&got),
            argmax(want),
            "Top-1 mismatch vs FP64 oracle at step {step}"
        );
        let mut max_diff = 0.0f32;
        for (&a, &b) in got.iter().zip(want.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 5e-4,
            "step {step}: max_abs_diff {max_diff} >= 5e-4 vs FP64 oracle"
        );
        mmap_logits_steps.push(got);
    }

    // 2. Single-thread vs 4-thread MUST be 0-ULP bit-exact (`AGENTS.md` §9 determinism)
    let mut state_1t = Inkling2State::new(&model_mmap, 64);
    for (step, &tid) in toks.iter().enumerate() {
        model_mmap
            .forward_token(&mut state_1t, &pool_1t, kt, tid, step, true)
            .unwrap();
        assert_eq!(
            state_1t.logits(),
            mmap_logits_steps[step].as_slice(),
            "1-thread vs 4-thread ULP drift at step {step}"
        );
    }

    // Repeat 4-thread forward_token 16 times to guard against any multi-thread non-determinism (`AGENTS.md` §9).
    for iter in 0..16 {
        state_mmap.reset();
        for (step, &tid) in toks.iter().enumerate() {
            model_mmap
                .forward_token(&mut state_mmap, &pool, kt, tid, step, true)
                .unwrap();
            assert_eq!(
                state_mmap.logits(),
                mmap_logits_steps[step].as_slice(),
                "4-thread non-determinism at iter={iter} step={step}"
            );
        }
    }

    // 3. Loop-Aware 3-Tier Placement (`O_DIRECT`): budget enough for ONLY the 4 core layers (`ip=2..6`)
    //    so `core_layers_in_fast_tiers == 4` (`ram_owned_expert_layers == 4`) and all 4 outer layers
    //    (`ip=0,1,6,7`) stream via parallel `O_DIRECT` (`nvme_direct_expert_layers == 4`)!
    //    Each layer's experts = 8 * (64*64/32*34 + 64*32/32*34) = 8 * (4352 + 2176) = 52,224 bytes.
    let four_core_budget = 4 * 52_224u64;
    let model_tiered = Inkling2Model::from_gguf_with_options(
        &gguf,
        &Inkling2LoadOptions {
            vram_expert_layers: Vec::new(),
            ram_expert_budget_bytes: Some(four_core_budget),
            host_moe_only: false,
        },
    )
    .unwrap();
    assert_eq!(model_tiered.stats.ram_owned_expert_layers, 4);
    assert_eq!(model_tiered.stats.nvme_direct_expert_layers, 4);
    assert_eq!(model_tiered.stats.core_layers_in_fast_tiers, 4);

    let mut state_tiered = Inkling2State::new(&model_tiered, 64);
    for (step, &tid) in toks.iter().enumerate() {
        model_tiered
            .forward_token(&mut state_tiered, &pool, kt, tid, step, true)
            .unwrap();
        assert_eq!(
            state_tiered.logits(),
            mmap_logits_steps[step].as_slice(),
            "Tiered RAM+NVMe O_DIRECT logits diverged from mmap at step {step}"
        );
    }

    // 4. Unified `Session`: `PrefixCache` checkpoint save/restore over 16 KV + 16 ShortConv slots
    //    AND `SuffixDrafter` speculative decoding (`verify_speculative` + `commit_speculative`)!
    let cpu_model = CpuModel::from_gguf(&gguf).unwrap();
    let mut sess_ref = Session::new_with_options(&cpu_model, 64, 16 * 1024 * 1024, KvType::F16);
    sess_ref.prefill_with_cache(&pool, &toks, Some(4)).unwrap();
    assert_eq!(
        sess_ref.logits(),
        mmap_logits_steps.last().unwrap().as_slice()
    );

    // Extending the prompt `[toks..., 21, 33]` MUST hit the cached checkpoint at len=6 (`toks[..6]`)
    // and produce bit-exact logits matching a fresh uncached session!
    let ext_prompt = [5u32, 12, 29, 41, 7, 63, 18, 21, 33];
    let rep_hit = sess_ref
        .prefill_with_cache(&pool, &ext_prompt, None)
        .unwrap();
    assert_eq!(rep_hit.cached_tokens, 6);

    let mut sess_fresh = Session::new_with_options(&cpu_model, 64, 0, KvType::F16);
    sess_fresh
        .prefill_with_cache(&pool, &ext_prompt, None)
        .unwrap();
    assert_eq!(sess_ref.logits(), sess_fresh.logits());

    // Greedy speculative decoding vs pure single-token greedy decoding MUST produce identical tokens!
    let rep_prompt = [5u32, 12, 29, 5, 12, 29, 5, 12];
    let mut sess_greedy = Session::new_with_options(&cpu_model, 64, 0, KvType::F16);
    sess_greedy
        .prefill_with_cache(&pool, &rep_prompt, None)
        .unwrap();
    let mut expected_gen = Vec::new();
    let mut next_t = argmax(sess_greedy.logits());
    expected_gen.push(next_t);
    for _ in 1..6 {
        sess_greedy.decode_step(&pool, next_t, true).unwrap();
        next_t = argmax(sess_greedy.logits());
        expected_gen.push(next_t);
    }

    let mut sess_spec = Session::new_with_options(&cpu_model, 64, 0, KvType::F16);
    sess_spec
        .prefill_with_cache(&pool, &rep_prompt, None)
        .unwrap();
    let mut suffix = SuffixDrafter::default();
    let mut ctrl = SpecController::default();
    let (spec_gen, _stats) = sess_spec
        .generate_greedy_speculative(
            &pool,
            &rep_prompt,
            6,
            SpecMode::Suffix,
            &mut suffix,
            None,
            &mut ctrl,
            |_, _| false,
        )
        .unwrap();
    assert_eq!(
        spec_gen, expected_gen,
        "Speculative verify/commit diverged from single-token greedy on inkling2_loop"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
