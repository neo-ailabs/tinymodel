//! V5 portability & subgroup-width ablation benchmark on real GGUF models:
//! - Sweeps every hardware-supported subgroup width (`8 / 16 / 32 / 64`) on the active device.
//! - Compares `gdn_recurrence` (`subgroupShuffleXor`) vs `gdn_recurrence_lds` (Pi 5 `V3DV`
//!   no-shuffle LDS fallback) on `Qwen3.5`.
//! - Verifies 100% deterministic greedy token sequence match across all modes!

use std::sync::Arc;
use std::time::Instant;

use tm_gguf::GgufFile;
use tm_vulkan::{Qwen35VkRunner, VulkanContext};

fn bench_config(
    selector: Option<&str>,
    sg: Option<u32>,
    force_lds: bool,
    gguf: &Arc<GgufFile>,
    n_pp: usize,
    n_tg: usize,
) -> (f64, f64, f64, u64, String, u32, bool) {
    let t_load = Instant::now();
    let ctx = VulkanContext::with_selector_and_subgroup(selector, sg).expect("VulkanContext");
    ctx.set_force_lds_recurrence(force_lds);
    let actual_sg = ctx.info().subgroup_size;
    let mut runner =
        Qwen35VkRunner::from_gguf_with_context(ctx, gguf.clone(), (n_pp + n_tg + 64).max(1024))
            .expect("Qwen35VkRunner");
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let wmode = runner.weights_mode().to_string();
    let used_lds = runner.uses_lds_recurrence();

    let pp_ids: Vec<u32> = (0..n_pp)
        .map(|i| ((i * 7 + 13) % runner.cfg.vocab_size.max(2)).max(1) as u32)
        .collect();

    // Warmup
    let _ = runner.forward_token_greedy(1, 0).expect("warm dec");
    runner.reset_state().expect("reset");
    let _ = runner.forward_batch(&pp_ids, 0, false).expect("warm pp");

    // Timed prefill
    runner.reset_state().expect("reset");
    let t_pp = Instant::now();
    let _ = runner.forward_batch(&pp_ids, 0, false).expect("pp");
    let pp_tps = (n_pp as f64) / t_pp.elapsed().as_secs_f64().max(1e-9);

    // Timed greedy decode
    runner.reset_state().expect("reset");
    let mut tok = runner.forward_token_greedy(1, 0).expect("tok0");
    let mut h = 0xcbf29ce484222325u64 ^ (tok as u64);
    h = h.wrapping_mul(0x100000001b3);
    let t_tg = Instant::now();
    for step in 0..n_tg {
        tok = runner.forward_token_greedy(tok, 1 + step).expect("tg");
        h ^= tok as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let tg_tps = (n_tg as f64) / t_tg.elapsed().as_secs_f64().max(1e-9);

    (load_ms, pp_tps, tg_tps, h, wmode, actual_sg, used_lds)
}

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3.5-0.8B-Q4_0.gguf".to_string());
    let gguf = Arc::new(GgufFile::open(&model_path).expect("open gguf"));
    let sel = std::env::var("TM_VK_DEVICE").ok();
    let base = VulkanContext::with_selector(sel.as_deref()).expect("base ctx");
    let info = base.info().clone();
    drop(base);

    let (n_pp, n_tg) = if info.unified_memory {
        (256usize, 32usize)
    } else {
        (512usize, 128usize)
    };

    println!(
        "=== V5 Subgroup & LDS Portability Ablation: {} on {} ({}) ===",
        model_path, info.name, info.driver_info
    );
    let mut sgs = Vec::new();
    if info.subgroup_size_control {
        for &c in &[8u32, 16, 32, 64] {
            if c >= info.min_subgroup_size && c <= info.max_subgroup_size {
                sgs.push(c);
            }
        }
    } else {
        sgs.push(info.subgroup_size);
    }

    let mut ref_hash = None;
    for &sg in &sgs {
        let (load_ms, pp_tps, tg_tps, h, wmode, actual_sg, used_lds) =
            bench_config(sel.as_deref(), Some(sg), false, &gguf, n_pp, n_tg);
        let h0 = *ref_hash.get_or_insert(h);
        println!(
            "  sg={:<2} recurrence={:<7} weights={:<14} load={:>6.1}ms pp{}={:>8.2} tok/s tg{}={:>7.2} tok/s hash={:016x} match={}",
            actual_sg,
            if used_lds { "lds" } else { "shuffle" },
            wmode,
            load_ms,
            n_pp,
            pp_tps,
            n_tg,
            tg_tps,
            h,
            h == h0
        );
    }

    let (load_ms, pp_tps, tg_tps, h, wmode, actual_sg, used_lds) =
        bench_config(sel.as_deref(), None, true, &gguf, n_pp, n_tg);
    let h0 = *ref_hash.get_or_insert(h);
    println!(
        "  sg={:<2} recurrence={:<7} weights={:<14} load={:>6.1}ms pp{}={:>8.2} tok/s tg{}={:>7.2} tok/s hash={:016x} match={}",
        actual_sg,
        if used_lds { "lds" } else { "shuffle" },
        wmode,
        load_ms,
        n_pp,
        pp_tps,
        n_tg,
        tg_tps,
        h,
        h == h0
    );
}
