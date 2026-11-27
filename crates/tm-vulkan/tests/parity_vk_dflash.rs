//! Multi-step L1/L3 Vulkan parity tests for `DFlashVkRunner` (`DSpark` / `dflash`)
//! against CPU `tm_models::DFlashModel` (`AGENTS.md` §5 Rules #3 & #7):
//! 1. Multi-token batch tap injection (`inject_batch` across prompt window `W = 5`, including
//!    both SWA and full-attention layers) + single-CB `draft_block_greedy(k = 3)`.
//! 2. Multi-round speculative `inject_and_draft_greedy` (`1` Vulkan CB for `inject + draft`)
//!    after `commit_speculative_window`, plus `TM_VK_DRAFT_VOCAB` pruned vocabulary parity
//!    and `reset_state()` bit-exact repeatability!

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use tm_core::{BlockQ8_0, f32_to_f16};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::{DFlashModel, DFlashState};
use tm_vulkan::{DFlashVkRunner, Inkling2VkRunner, VulkanError};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
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

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn i8_sym(&mut self) -> i8 {
        ((self.next_u64() % 201) as i32 - 100) as i8
    }
    fn f32(&mut self) -> f32 {
        (((self.next_u64() % 2001) as f32) - 1000.0) / 1000.0
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn kv_str(buf: &mut Vec<u8>, k: &str, v: &str) {
    push_str(buf, k);
    buf.extend_from_slice(&8u32.to_le_bytes());
    push_str(buf, v);
}

fn kv_u32(buf: &mut Vec<u8>, k: &str, v: u32) {
    push_str(buf, k);
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn kv_f32(buf: &mut Vec<u8>, k: &str, v: f32) {
    push_str(buf, k);
    buf.extend_from_slice(&6u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn kv_u32_arr(buf: &mut Vec<u8>, k: &str, vals: &[u32]) {
    push_str(buf, k);
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for &v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn kv_bool_arr(buf: &mut Vec<u8>, k: &str, vals: &[bool]) {
    push_str(buf, k);
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&7u32.to_le_bytes());
    buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for &b in vals {
        buf.push(u8::from(b));
    }
}

struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

fn make_f32_bytes(rng: &mut Rng, len: usize, center: f32, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for _ in 0..len {
        let v = center + rng.f32() * scale;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn make_bf16_bytes(rng: &mut Rng, len: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        let u = f32_to_bf16_bits(rng.f32() * scale);
        out.extend_from_slice(&u.to_le_bytes());
    }
    out
}

fn make_q8_0_bytes(rng: &mut Rng, rows: usize, cols: usize, scale: f32) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let nb = rows * cols / 32;
    let mut out = Vec::with_capacity(nb * 34);
    for _ in 0..nb {
        let b = BlockQ8_0 {
            d: f32_to_f16((0.2 + rng.f32().abs() * 0.8) * scale),
            qs: core::array::from_fn(|_| rng.i8_sym()),
        };
        out.extend_from_slice(&b.d.to_le_bytes());
        for &q in &b.qs {
            out.push(q as u8);
        }
    }
    out
}

fn build_synthetic_dflash_gguf(
    n_embd: usize,
    vocab_padded: usize,
    vocab_size: usize,
    target_layers: &[u32],
) -> Vec<u8> {
    let n_layers = 2usize;
    let block_size = 4usize;
    let n_ff = 128usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = n_embd / n_head;
    let q_dim = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let markov_rank = 32usize;
    let n_taps = target_layers.len();

    let mut rng = Rng(0xD5AA_2C00_2026_1126);
    let mut tensors = Vec::with_capacity(9 + n_layers * 15);

    tensors.push(TensorEntry {
        name: "dspark_fuse.weight".to_string(),
        dims: vec![n_taps as u64, (n_layers + 1) as u64],
        ggml_type: 0,
        data: make_f32_bytes(&mut rng, n_taps * (n_layers + 1), 0.0, 0.8),
    });
    tensors.push(TensorEntry {
        name: "dspark_norm.weight".to_string(),
        dims: vec![n_embd as u64],
        ggml_type: 0,
        data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
    });
    tensors.push(TensorEntry {
        name: "mask_embd.weight".to_string(),
        dims: vec![n_embd as u64],
        ggml_type: 0,
        data: make_f32_bytes(&mut rng, n_embd, 0.0, 0.25),
    });
    tensors.push(TensorEntry {
        name: "output_norm.weight".to_string(),
        dims: vec![n_embd as u64],
        ggml_type: 0,
        data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
    });
    // Real DSpark checkpoints store `markov_w1.weight` and `dspark_hidden_out.weight` as BF16 (`30`)!
    tensors.push(TensorEntry {
        name: "markov_w1.weight".to_string(),
        dims: vec![markov_rank as u64, vocab_padded as u64],
        ggml_type: 30,
        data: make_bf16_bytes(&mut rng, markov_rank * vocab_padded, 0.25),
    });
    tensors.push(TensorEntry {
        name: "joint_proj.weight".to_string(),
        dims: vec![
            (2 * n_embd + 2 * markov_rank) as u64,
            (3 * markov_rank) as u64,
        ],
        ggml_type: 8,
        data: make_q8_0_bytes(
            &mut rng,
            3 * markov_rank,
            2 * n_embd + 2 * markov_rank,
            0.0015,
        ),
    });
    tensors.push(TensorEntry {
        name: "joint_proj.bias".to_string(),
        dims: vec![(3 * markov_rank) as u64],
        ggml_type: 0,
        data: make_f32_bytes(&mut rng, 3 * markov_rank, 0.0, 0.05),
    });
    tensors.push(TensorEntry {
        name: "dspark_hidden_out.weight".to_string(),
        dims: vec![markov_rank as u64, n_embd as u64],
        ggml_type: 30,
        data: make_bf16_bytes(&mut rng, markov_rank * n_embd, 0.15),
    });
    tensors.push(TensorEntry {
        name: "dspark_hidden_out.bias".to_string(),
        dims: vec![n_embd as u64],
        ggml_type: 0,
        data: make_f32_bytes(&mut rng, n_embd, 0.0, 0.05),
    });

    for il in 0..n_layers {
        let p = format!("blk.{il}");
        tensors.push(TensorEntry {
            name: format!("{p}.dflare_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_k_ctx.weight"),
            dims: vec![n_embd as u64, kv_dim as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, kv_dim, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_v_ctx.weight"),
            dims: vec![n_embd as u64, kv_dim as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, kv_dim, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_k_ctx_norm.weight"),
            dims: vec![head_dim as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_q.weight"),
            dims: vec![n_embd as u64, q_dim as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, q_dim, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_k.weight"),
            dims: vec![n_embd as u64, kv_dim as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, kv_dim, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_v.weight"),
            dims: vec![n_embd as u64, kv_dim as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, kv_dim, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_output.weight"),
            dims: vec![q_dim as u64, n_embd as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, n_embd, q_dim, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_q_norm.weight"),
            dims: vec![head_dim as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.attn_k_norm.weight"),
            dims: vec![head_dim as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, head_dim, 1.0, 0.1),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_norm.weight"),
            dims: vec![n_embd as u64],
            ggml_type: 0,
            data: make_f32_bytes(&mut rng, n_embd, 1.0, 0.15),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_gate.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, n_ff, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_up.weight"),
            dims: vec![n_embd as u64, n_ff as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, n_ff, n_embd, 0.0015),
        });
        tensors.push(TensorEntry {
            name: format!("{p}.ffn_down.weight"),
            dims: vec![n_ff as u64, n_embd as u64],
            ggml_type: 8,
            data: make_q8_0_bytes(&mut rng, n_embd, n_ff, 0.0012),
        });
    }

    let mut kv = Vec::new();
    let mut n_kv = 0u64;
    kv_str(&mut kv, "general.architecture", "dflash");
    n_kv += 1;
    kv_u32(&mut kv, "dflash.block_count", n_layers as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.block_size", block_size as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.embedding_length", n_embd as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.feed_forward_length", n_ff as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.attention.head_count", n_head as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.attention.head_count_kv", n_head_kv as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.attention.key_length", head_dim as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.rope.dimension_count", head_dim as u32);
    n_kv += 1;
    kv_f32(&mut kv, "dflash.rope.freq_base", 1_000_000.0);
    n_kv += 1;
    kv_f32(&mut kv, "dflash.attention.layer_norm_rms_epsilon", 1e-6);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.attention.sliding_window", 4);
    n_kv += 1;
    kv_bool_arr(
        &mut kv,
        "dflash.attention.sliding_window_pattern",
        &[true, false],
    );
    n_kv += 1;
    kv_u32_arr(&mut kv, "dflash.target_layers", target_layers);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.dspark.markov_rank", markov_rank as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.vocab_size", vocab_padded as u32);
    n_kv += 1;
    kv_u32(&mut kv, "dflash.vocab_size_unpadded", vocab_size as u32);
    n_kv += 1;
    kv_f32(&mut kv, "dflash.final_logit_softcapping", 30.0);
    n_kv += 1;

    let mut tinfos = Vec::new();
    let mut offset = 0u64;
    for t in &tensors {
        push_str(&mut tinfos, &t.name);
        tinfos.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for &d in &t.dims {
            tinfos.extend_from_slice(&d.to_le_bytes());
        }
        tinfos.extend_from_slice(&t.ggml_type.to_le_bytes());
        tinfos.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).div_ceil(32) * 32;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&n_kv.to_le_bytes());
    out.extend_from_slice(&kv);
    out.extend_from_slice(&tinfos);
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for t in &tensors {
        out.extend_from_slice(&t.data);
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
    }
    out
}

#[test]
fn parity_vk_dflash_multi_step_and_speculative_loop() {
    let root = repo_root();
    let tmp_dir = std::env::temp_dir().join(format!("tm_vk_dflash_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let target_gguf_path = tmp_dir.join("synthetic_inkling2_loop_q8_0.gguf");
    let ref_path = tmp_dir.join("ref_float_act.tmlg");
    let dflash_gguf_path = tmp_dir.join("synthetic_dflash_q8_0.gguf");

    let status = Command::new(find_python3_with_numpy())
        .arg(root.join("tools/oracle/inkling2_oracle.py"))
        .arg("--gen-gguf")
        .arg(&target_gguf_path)
        .arg("--tokens")
        .arg("5,12,29,41,7")
        .arg("--float-act")
        .arg("--out")
        .arg(&ref_path)
        .status()
        .expect("failed to invoke inkling2_oracle.py");
    assert!(status.success());

    let target_gguf = Arc::new(GgufFile::open(&target_gguf_path).unwrap());
    let mut target_vk = match Inkling2VkRunner::from_gguf(target_gguf.clone(), 64) {
        Ok(r) => r,
        Err(VulkanError::Loader(_) | VulkanError::NoDevice) => {
            eprintln!("skipping parity_vk_dflash: no Vulkan loader/device in environment");
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return;
        }
        Err(e) => panic!("Inkling2VkRunner::from_gguf failed: {e}"),
    };

    let tap_layers_u32 = [1u32, 5, 11, 16];
    let tap_visits = [1usize, 5, 11, 16];
    let dflash_bytes = build_synthetic_dflash_gguf(
        target_vk.cfg.n_embd,
        target_vk.cfg.vocab_padded,
        target_vk.cfg.vocab_size,
        &tap_layers_u32,
    );
    std::fs::write(&dflash_gguf_path, &dflash_bytes).unwrap();
    let dflash_gguf = GgufFile::open(&dflash_gguf_path).unwrap();

    let dflash_cpu = DFlashModel::from_gguf(&dflash_gguf).unwrap();
    let mut dflash_cpu_state = DFlashState::new(&dflash_cpu, 64);
    let mut dflash_vk = DFlashVkRunner::from_gguf(&target_vk, &dflash_gguf, 64).unwrap();

    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let n_embd = target_vk.cfg.n_embd;
    let n_taps = tap_visits.len();

    // 1. Prompt prefill (`W = 5`, exercises SWA cutoff `sliding_window = 4` -> `n_swa_eff = 5` on layer 0
    //    and full attention on layer 1) + `inject_batch` + `draft_block_greedy(k = 3)`.
    let prompt = [5u32, 12, 29, 41, 7];
    let mut prompt_argmax = [0u32; 5];
    let _ = target_vk
        .forward_window_hybrid(
            &prompt,
            0,
            false,
            false,
            &tap_visits,
            None,
            Some(&mut prompt_argmax),
        )
        .unwrap();

    let mut host_taps = vec![0.0f32; prompt.len() * n_taps * n_embd];
    target_vk
        .read_dspark_taps(prompt.len(), n_taps, &mut host_taps)
        .unwrap();

    for t in 0..prompt.len() {
        let slice = &host_taps[t * n_taps * n_embd..(t + 1) * n_taps * n_embd];
        dflash_cpu
            .inject_token(&mut dflash_cpu_state, &pool, kt, t, slice)
            .unwrap();
    }
    dflash_vk.inject_batch(&target_vk, 0, prompt.len()).unwrap();

    let anchor0 = prompt_argmax[prompt.len() - 1];
    let mut drafts_cpu = [0u32; 3];
    let mut drafts_vk = [0u32; 3];
    {
        let tvk = std::cell::RefCell::new(&mut target_vk);
        dflash_cpu
            .draft_block_greedy(
                &mut dflash_cpu_state,
                &pool,
                kt,
                prompt.len() - 1,
                anchor0,
                3,
                |tid, out| tvk.borrow_mut().embed_token_normed(tid, out).unwrap(),
                |hid, cap| tvk.borrow_mut().project_dspark_greedy(hid, cap).unwrap(),
                &mut drafts_cpu,
            )
            .unwrap();
    }
    dflash_vk
        .draft_block_greedy(&mut target_vk, prompt.len() - 1, anchor0, 3, &mut drafts_vk)
        .unwrap();
    assert_eq!(
        drafts_vk,
        drafts_cpu,
        "DFlashVkRunner step 0 draft mismatch vs CPU DFlashModel on {}",
        target_vk.device_name()
    );

    // 2. Speculative verification window (`W = 4`: `[anchor0, d0, d1, d2]`), partial commit (`committed = 2`),
    //    and single-CB `inject_and_draft_greedy(inj_start_pos = 5, inj_count = 2, pos0 = 6, k = 3)`!
    let verify_win = [anchor0, drafts_vk[0], drafts_vk[1], drafts_vk[2]];
    let mut verify_argmax = [0u32; 4];
    let _ = target_vk
        .forward_window_hybrid(
            &verify_win,
            prompt.len(),
            false,
            true,
            &tap_visits,
            None,
            Some(&mut verify_argmax),
        )
        .unwrap();
    let committed = 2usize; // accept 1 draft token -> commit positions 5 and 6
    target_vk.commit_speculative_window(committed).unwrap();

    let mut step1_taps = vec![0.0f32; committed * n_taps * n_embd];
    target_vk
        .read_dspark_taps(committed, n_taps, &mut step1_taps)
        .unwrap();
    for c in 0..committed {
        let slice = &step1_taps[c * n_taps * n_embd..(c + 1) * n_taps * n_embd];
        dflash_cpu
            .inject_token(&mut dflash_cpu_state, &pool, kt, prompt.len() + c, slice)
            .unwrap();
    }

    let anchor1 = verify_argmax[committed - 1];
    let pos1 = prompt.len() + committed - 1;
    let mut drafts_cpu_1 = [0u32; 3];
    let mut drafts_vk_1 = [0u32; 3];
    {
        let tvk = std::cell::RefCell::new(&mut target_vk);
        dflash_cpu
            .draft_block_greedy(
                &mut dflash_cpu_state,
                &pool,
                kt,
                pos1,
                anchor1,
                3,
                |tid, out| tvk.borrow_mut().embed_token_normed(tid, out).unwrap(),
                |hid, cap| tvk.borrow_mut().project_dspark_greedy(hid, cap).unwrap(),
                &mut drafts_cpu_1,
            )
            .unwrap();
    }
    dflash_vk
        .inject_and_draft_greedy(
            &mut target_vk,
            prompt.len(),
            committed,
            pos1,
            anchor1,
            3,
            &mut drafts_vk_1,
        )
        .unwrap();
    assert_eq!(
        drafts_vk_1,
        drafts_cpu_1,
        "DFlashVkRunner fused inject_and_draft_greedy mismatch at pos={pos1} on {}",
        target_vk.device_name()
    );

    // 3. `reset_state()` repeatability: re-injecting `prompt` reproduces `drafts_vk` bit-for-bit!
    dflash_vk.reset_state(&target_vk).unwrap();
    target_vk.reset_state().unwrap();
    let _ = target_vk
        .forward_window_hybrid(&prompt, 0, false, false, &tap_visits, None, None)
        .unwrap();
    let mut drafts_vk_repeat = [0u32; 3];
    dflash_vk
        .inject_and_draft_greedy(
            &mut target_vk,
            0,
            prompt.len(),
            prompt.len() - 1,
            anchor0,
            3,
            &mut drafts_vk_repeat,
        )
        .unwrap();
    assert_eq!(drafts_vk_repeat, drafts_vk);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
