//! Multi-token Vulkan parity tests for `Inkling-2-Mini` (`inkling2_loop`) across:
//! 1. 100% pure-GPU execution (`vram-full`), AND
//! 2. Loop-aware 3-Tier `GPU VRAM + Host RAM + NVMe O_DIRECT` hybrid execution!

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{Inkling2LoadOptions, Inkling2Model, Inkling2State, argmax};
use tm_vulkan::{Inkling2VkRunner, VulkanError};

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
fn parity_vk_inkling2_pure_gpu_and_3tier_hybrid() {
    let root = repo_root();
    let tmp_dir = std::env::temp_dir().join(format!("tm_vk_inkling2_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let gguf_path = tmp_dir.join("synthetic_inkling2_loop_q8_0.gguf");
    let ref_path = tmp_dir.join("ref_float_act.tmlg");

    let status = Command::new(find_python3_with_numpy())
        .arg(root.join("tools/oracle/inkling2_oracle.py"))
        .arg("--gen-gguf")
        .arg(&gguf_path)
        .arg("--tokens")
        .arg("5,12,29,41,7,63,18")
        .arg("--float-act")
        .arg("--out")
        .arg(&ref_path)
        .status()
        .expect("failed to invoke inkling2_oracle.py");
    assert!(status.success());

    let (toks, _, ref_logits) = read_tmlg_f32(&ref_path);
    let gguf = Arc::new(GgufFile::open(&gguf_path).unwrap());

    // 1. 100% pure-GPU mode (`vram-full`) vs FP64 float-activation oracle
    let mut runner = match Inkling2VkRunner::from_gguf(gguf.clone(), 64) {
        Ok(r) => r,
        Err(VulkanError::Loader(_) | VulkanError::NoDevice) => {
            eprintln!("skipping parity_vk_inkling2: no Vulkan loader/device in environment");
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return;
        }
        Err(e) => panic!("Inkling2VkRunner::from_gguf failed: {e}"),
    };
    assert!(!runner.has_host_offload_layers());
    assert_eq!(runner.weights_mode(), "vram-full");

    for (step, &tid) in toks.iter().enumerate() {
        let got = runner.forward_token(tid, step, true).unwrap();
        let want = &ref_logits[step];
        assert_eq!(
            argmax(got),
            argmax(want),
            "pure-GPU Top-1 mismatch at step {step} on {}",
            runner.device_name()
        );
        let mut max_diff = 0.0f32;
        for (&a, &b) in got.iter().zip(want.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 5e-3,
            "pure-GPU step {step} max_diff {max_diff} >= 5e-3 on {}",
            runner.device_name()
        );
    }

    // Verify reset_state() + forward_token_greedy() matches step 0 argmax!
    runner.reset_state().unwrap();
    let greedy0 = runner.forward_token_greedy(toks[0], 0).unwrap();
    assert_eq!(greedy0, argmax(&ref_logits[0]));

    // 2. Forced 3-Tier Hybrid (`GPU VRAM + Host Anonymous RAM + NVMe O_DIRECT`)!
    //    Give VRAM enough budget for dense + state + 2 of the 4 core layers (`ip=2,3`),
    //    Host RAM enough for the other 2 core layers (`ip=4,5`), and stream the 4 outer layers
    //    (`ip=0,1,6,7`) via parallel `O_DIRECT` NVMe!
    let one_layer_exp_bytes = 52_224u64;
    let two_core_bytes = 2 * one_layer_exp_bytes;
    let scratch_est = ((24 * 64 + 4 * 2 * 32 + 2 * 96 + 65536) * 4) as u64;
    let base_plus_2core = runner.vram_total_bytes() + scratch_est - 6 * one_layer_exp_bytes + 8192;
    let mut hybrid_runner =
        Inkling2VkRunner::from_gguf_with_vram_budget(gguf.clone(), 64, Some(base_plus_2core))
            .unwrap();
    assert!(hybrid_runner.has_host_offload_layers());

    let host_model = Inkling2Model::from_gguf_with_options(
        &gguf,
        &Inkling2LoadOptions {
            vram_expert_layers: hybrid_runner.vram_expert_mask().to_vec(),
            ram_expert_budget_bytes: Some(two_core_bytes),
            host_moe_only: true,
        },
    )
    .unwrap();
    assert!(host_model.stats.nvme_direct_expert_layers >= 4);
    let mut host_state = Inkling2State::new(&host_model, 64);
    let pool = ThreadPool::new(4);
    let kt = active_kernels();

    for (step, &tid) in toks.iter().enumerate() {
        let mut cb = |ip: usize,
                      cur_normed: &[f32],
                      top_ids: &[usize],
                      top_weights: &[f32],
                      moe_out: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch) = host_state.host_moe_parts_mut();
            host_model.eval_moe_experts_f32(
                ip,
                cur_normed,
                top_ids,
                top_weights,
                &pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                moe_out,
            );
        };
        let got = hybrid_runner
            .forward_token_hybrid(tid, step, true, Some(&mut cb))
            .unwrap();
        let top_got = argmax(got);
        let want = &ref_logits[step];
        let mut max_diff_h = 0.0f32;
        for (&a, &b) in got.iter().zip(want.iter()) {
            max_diff_h = max_diff_h.max((a - b).abs());
        }
        assert_eq!(
            top_got,
            argmax(want),
            "3-tier hybrid Top-1 mismatch at step {step} ({})",
            hybrid_runner.weights_mode()
        );
        assert!(
            max_diff_h < 5e-3,
            "3-tier hybrid step {step} max_diff_h {max_diff_h} >= 5e-3 ({})",
            hybrid_runner.weights_mode()
        );
    }

    // 3. Lockstep multi-token window (`W = 4`) + `commit_speculative_window(2)` rollback +
    //    `DSpark` target visit tap capture (`[1, 5, 11, 16]`):
    //    Run `toks[0..4]` in a single lockstep window with `save_conv_snapshots = true`,
    //    verify all 4 GPU argmax IDs match `ref_logits[0..4]`, roll back to `committed = 2`,
    //    then run `toks[2..5]` in a 3-token window and verify `argmax` & final logits match `ref_logits[4]`!
    hybrid_runner.reset_state().unwrap();
    let tap_visits = [1usize, 5, 11, 16];
    let mut win_argmax = [0u32; 4];
    {
        let mut cb_win = |ip: usize,
                          w_len: usize,
                          cur_win: &[f32],
                          ids_win: &[usize],
                          w_win: &[f32],
                          out_win: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch, win_slot_outs) =
                host_state.host_moe_window_parts_mut();
            host_model.eval_moe_experts_window_f32(
                ip,
                w_len,
                cur_win,
                ids_win,
                w_win,
                &pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                win_slot_outs,
                out_win,
            );
        };
        let _ = hybrid_runner
            .forward_window_hybrid(
                &toks[0..4],
                0,
                false,
                true,
                &tap_visits,
                Some(&mut cb_win),
                Some(&mut win_argmax),
            )
            .unwrap();
    }
    for s in 0..4 {
        assert_eq!(
            win_argmax[s],
            argmax(&ref_logits[s]),
            "window argmax mismatch at slot {s}"
        );
    }
    let mut taps_read = vec![0.0f32; 4 * tap_visits.len() * 64];
    hybrid_runner
        .read_dspark_taps(4, tap_visits.len(), &mut taps_read)
        .unwrap();
    assert!(taps_read.iter().any(|&v| v.abs() > 1e-3));

    // Roll back from 4 tokens (`pos=0..4`) to `committed = 2` (`pos=0..2`), then evaluate `toks[2..5]` at `start_pos = 2`!
    hybrid_runner.commit_speculative_window(2).unwrap();
    let mut win_argmax_after = [0u32; 3];
    let got_step4 = {
        let mut cb_win = |ip: usize,
                          w_len: usize,
                          cur_win: &[f32],
                          ids_win: &[usize],
                          w_win: &[f32],
                          out_win: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch, win_slot_outs) =
                host_state.host_moe_window_parts_mut();
            host_model.eval_moe_experts_window_f32(
                ip,
                w_len,
                cur_win,
                ids_win,
                w_win,
                &pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                win_slot_outs,
                out_win,
            );
        };
        hybrid_runner
            .forward_window_hybrid(
                &toks[2..5],
                2,
                true,
                false,
                &[],
                Some(&mut cb_win),
                Some(&mut win_argmax_after),
            )
            .unwrap()
    };
    for s in 0..3 {
        assert_eq!(
            win_argmax_after[s],
            argmax(&ref_logits[2 + s]),
            "post-rollback window argmax mismatch at step {}",
            2 + s
        );
    }
    let mut max_diff_rb = 0.0f32;
    for (&a, &b) in got_step4.iter().zip(ref_logits[4].iter()) {
        max_diff_rb = max_diff_rb.max((a - b).abs());
    }
    assert!(
        max_diff_rb < 5e-3,
        "post-rollback step 4 max_diff {max_diff_rb} >= 5e-3"
    );

    // 4. Batched-GEMM prefill window (`W = 10 >= 8`, `save_conv_snapshots = false`):
    //    Extend the 7-token sequence to 10 tokens (`[5,12,29,41,7,63,18,15,2,3]`, `min_router_margin > 6e-3`),
    //    verifying `forward_window_hybrid` (2D batched prefill + grouped MoE) matches sequential
    //    single-token `forward_token_hybrid` across all 10 tokens on both pure-GPU and 3-tier hybrid runners!
    let toks10 = [5u32, 12, 29, 41, 7, 63, 18, 15, 2, 3];
    runner.reset_state().unwrap();
    for (s, &tid) in toks10.iter().enumerate() {
        let _ = runner.forward_token(tid, s, s + 1 == toks10.len()).unwrap();
    }
    let seq_pure_last = runner
        .forward_batch_hybrid(&[], toks10.len(), false, None)
        .unwrap()
        .to_vec();

    runner.reset_state().unwrap();
    let mut win10_am = [0u32; 10];
    let win_pure_last = runner
        .forward_window_hybrid(
            &toks10,
            0,
            true,
            false,
            &tap_visits,
            None,
            Some(&mut win10_am),
        )
        .unwrap()
        .to_vec();
    let mut taps10 = vec![0.0f32; 10 * tap_visits.len() * 64];
    runner
        .read_dspark_taps(10, tap_visits.len(), &mut taps10)
        .unwrap();
    for s in 0..4 {
        let base_a = s * tap_visits.len() * 64;
        let base_b = s * tap_visits.len() * 64;
        for d in 0..(tap_visits.len() * 64) {
            assert!(
                (taps10[base_a + d] - taps_read[base_b + d]).abs() < 5e-3,
                "W=10 batched tap mismatch vs W=4 tap at tok {s} dim {d}"
            );
        }
    }
    assert_eq!(argmax(&win_pure_last), argmax(&seq_pure_last));
    let mut max_diff_w10_pure = 0.0f32;
    for (&a, &b) in win_pure_last.iter().zip(seq_pure_last.iter()) {
        max_diff_w10_pure = max_diff_w10_pure.max((a - b).abs());
    }
    assert!(
        max_diff_w10_pure < 5e-3,
        "pure-GPU W=10 batched GEMM max_diff {max_diff_w10_pure} >= 5e-3"
    );
    for s in 0..7 {
        assert_eq!(win10_am[s], argmax(&ref_logits[s]));
    }

    hybrid_runner.reset_state().unwrap();
    let mut win10_hyb_am = [0u32; 10];
    let win_hyb_last = {
        let mut cb_win = |ip: usize,
                          w_len: usize,
                          cur_win: &[f32],
                          ids_win: &[usize],
                          w_win: &[f32],
                          out_win: &mut [f32]| {
            let (serial_pool, expert_slots, mv_scratch, win_slot_outs) =
                host_state.host_moe_window_parts_mut();
            host_model.eval_moe_experts_window_f32(
                ip,
                w_len,
                cur_win,
                ids_win,
                w_win,
                &pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                win_slot_outs,
                out_win,
            );
        };
        hybrid_runner
            .forward_window_hybrid(
                &toks10,
                0,
                true,
                false,
                &[],
                Some(&mut cb_win),
                Some(&mut win10_hyb_am),
            )
            .unwrap()
            .to_vec()
    };
    assert_eq!(win10_hyb_am, win10_am);
    assert_eq!(argmax(&win_hyb_last), argmax(&seq_pure_last));
    let mut max_diff_w10_hyb = 0.0f32;
    for (&a, &b) in win_hyb_last.iter().zip(seq_pure_last.iter()) {
        max_diff_w10_hyb = max_diff_w10_hyb.max((a - b).abs());
    }
    assert!(
        max_diff_w10_hyb < 5e-3,
        "3-tier hybrid W=10 batched GEMM max_diff {max_diff_w10_hyb} >= 5e-3"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
