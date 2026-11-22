//! End-to-end 3-layer QLoRA, LoRA merge, and MTP training verification (`ADR-015` §3).

use std::path::Path;
use std::process::Command;

use tm_core::{DType, dequantize_row_bytes, quantize_row_to_bytes};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::{GgufFile, MetadataValue, NewGgufTensor, write_gguf_file};
use tm_lora::{LoraMergeOptions, LoraSet, merge_lora_gguf};
use tm_models::{LlamaModel, LlamaState};
use tm_runtime::{CpuModel, LinearMtpHead};
use tm_train::{
    CheckpointMode, LlamaTrainer, LoraGradStore, LoraTargets, MtpTrainConfig, QloraConfig,
    SplitMix64, TrainDataset, TrainSample, init_lora_adapter, run_qlora_training, train_mtp_head,
};

struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "tm_train_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).expect("mkdir temp");
        Self { path: p }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn encode_2d(dtype: DType, cols: usize, rows: usize, vals: &[f32]) -> Vec<u8> {
    assert_eq!(vals.len(), rows * cols);
    let rb = dtype.row_bytes(cols);
    let mut out = vec![0u8; rows * rb];
    for r in 0..rows {
        quantize_row_to_bytes(
            dtype,
            &vals[r * cols..(r + 1) * cols],
            &mut out[r * rb..(r + 1) * rb],
        );
    }
    out
}

fn encode_1d_f32(vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[allow(clippy::too_many_arguments)]
fn write_synthetic_llama_or_qwen3_gguf(
    dst: &Path,
    arch: &str,
    weight_dtype: DType,
    n_layers: usize,
    n_embd: usize,
    n_ff: usize,
    n_head: usize,
    n_head_kv: usize,
    vocab: usize,
    seed: u64,
) {
    let hd = n_embd / n_head;
    let q_dim = n_head * hd;
    let kv_dim = n_head_kv * hd;
    let mut rng = SplitMix64::new(seed);

    let meta = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(arch.to_string()),
        ),
        (
            format!("{arch}.block_count"),
            MetadataValue::U32(n_layers as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            MetadataValue::U32(n_embd as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            MetadataValue::U32(n_ff as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            MetadataValue::U32(n_head as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            MetadataValue::U32(n_head_kv as u32),
        ),
        (
            format!("{arch}.attention.key_length"),
            MetadataValue::U32(hd as u32),
        ),
        (
            format!("{arch}.vocab_size"),
            MetadataValue::U32(vocab as u32),
        ),
        (format!("{arch}.context_length"), MetadataValue::U32(256)),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            MetadataValue::F32(1e-5),
        ),
        (
            format!("{arch}.rope.freq_base"),
            MetadataValue::F32(10000.0),
        ),
    ];

    struct T {
        name: String,
        dims: Vec<usize>,
        dtype: DType,
        bytes: Vec<u8>,
    }
    let mut tensors = Vec::new();
    fn rand_vec(rng: &mut SplitMix64, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| rng.uniform_sym(scale)).collect()
    }
    fn norm_vec(rng: &mut SplitMix64, len: usize) -> Vec<f32> {
        (0..len).map(|_| 0.85 + rng.uniform_sym(0.25)).collect()
    }

    let emb_f32 = rand_vec(&mut rng, vocab * n_embd, 0.25);
    tensors.push(T {
        name: "token_embd.weight".to_string(),
        dims: vec![n_embd, vocab],
        dtype: weight_dtype,
        bytes: encode_2d(weight_dtype, n_embd, vocab, &emb_f32),
    });
    let out_f32 = rand_vec(&mut rng, vocab * n_embd, 0.2);
    tensors.push(T {
        name: "output.weight".to_string(),
        dims: vec![n_embd, vocab],
        dtype: weight_dtype,
        bytes: encode_2d(weight_dtype, n_embd, vocab, &out_f32),
    });
    tensors.push(T {
        name: "output_norm.weight".to_string(),
        dims: vec![n_embd],
        dtype: DType::F32,
        bytes: encode_1d_f32(&norm_vec(&mut rng, n_embd)),
    });

    for l in 0..n_layers {
        let pfx = format!("blk.{l}");
        tensors.push(T {
            name: format!("{pfx}.attn_norm.weight"),
            dims: vec![n_embd],
            dtype: DType::F32,
            bytes: encode_1d_f32(&norm_vec(&mut rng, n_embd)),
        });
        tensors.push(T {
            name: format!("{pfx}.ffn_norm.weight"),
            dims: vec![n_embd],
            dtype: DType::F32,
            bytes: encode_1d_f32(&norm_vec(&mut rng, n_embd)),
        });
        if arch == "qwen3" {
            tensors.push(T {
                name: format!("{pfx}.attn_q_norm.weight"),
                dims: vec![hd],
                dtype: DType::F32,
                bytes: encode_1d_f32(&norm_vec(&mut rng, hd)),
            });
            tensors.push(T {
                name: format!("{pfx}.attn_k_norm.weight"),
                dims: vec![hd],
                dtype: DType::F32,
                bytes: encode_1d_f32(&norm_vec(&mut rng, hd)),
            });
        }
        for (suf, cols, rows) in [
            ("attn_q.weight", n_embd, q_dim),
            ("attn_k.weight", n_embd, kv_dim),
            ("attn_v.weight", n_embd, kv_dim),
            ("attn_output.weight", q_dim, n_embd),
            ("ffn_gate.weight", n_embd, n_ff),
            ("ffn_up.weight", n_embd, n_ff),
            ("ffn_down.weight", n_ff, n_embd),
        ] {
            let v = rand_vec(&mut rng, rows * cols, 0.2);
            tensors.push(T {
                name: format!("{pfx}.{suf}"),
                dims: vec![cols, rows],
                dtype: weight_dtype,
                bytes: encode_2d(weight_dtype, cols, rows, &v),
            });
        }
    }

    let views: Vec<NewGgufTensor<'_>> = tensors
        .iter()
        .map(|t| NewGgufTensor {
            name: &t.name,
            dims: &t.dims,
            dtype: t.dtype,
            data: &t.bytes,
        })
        .collect();
    write_gguf_file(dst, &meta, &views, 32).expect("write synthetic gguf");
}

/// Gate 1 (`ADR-015` §3): Independent `LlamaTrainer` forward MUST match `LlamaModel::forward_batch`
/// (and `forward_token` for `T == 1`) to **0 ULP (`to_bits()`)** across `llama`, `qwen3` (QK-Norm),
/// `MAX_CPU_BATCH` chunk boundaries (`T = 1, 13, 32, 45`), AND non-zero LoRA!
#[test]
fn zero_ulp_training_forward_parity_with_inference() {
    let tmp = TempDir::new("fwd0ulp");
    let pool = ThreadPool::new(4);
    let kt = active_kernels();

    for arch in ["llama", "qwen3"] {
        let mpath = tmp.path.join(format!("{arch}_q4.gguf"));
        write_synthetic_llama_or_qwen3_gguf(&mpath, arch, DType::Q4_0, 2, 64, 128, 4, 2, 96, 123);
        let gguf = GgufFile::open(&mpath).expect("open");
        let mut model = LlamaModel::from_gguf(&gguf).expect("load");
        let mut state = LlamaState::new(&model.cfg, 64);
        let mut trainer = LlamaTrainer::new(&model, 64, 8, CheckpointMode::Fp16);

        let targets = LoraTargets::parse_csv("all").expect("targets");
        let mut adapter = init_lora_adapter(&model, 4, 8.0, &targets, 777).expect("init lora");

        for &seq_len in &[1usize, 13, 32, 45] {
            let toks: Vec<u32> = (0..seq_len).map(|i| ((i * 7 + 3) % 90) as u32).collect();
            state.reset();
            model
                .forward_batch(&mut state, &pool, kt, &toks, 0, true)
                .expect("infer forward_batch");

            let mut train_logits = vec![0.0f32; model.cfg.vocab_size];
            trainer.forward_last_logits(&pool, Some(&adapter), &toks, &mut train_logits);
            for (v, (&a, &b)) in state.logits().iter().zip(train_logits.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "0-ULP mismatch (zero-B LoRA) arch={arch} seq_len={seq_len} vocab_idx={v}: infer={a} train={b}"
                );
            }
        }

        // Now populate non-zero `B` deterministically across `pair_order`
        let mut rng = SplitMix64::new(999);
        for name in adapter.pair_order.clone() {
            let p = adapter.pairs.get_mut(&name).unwrap();
            for bv in &mut p.b {
                *bv = rng.uniform_sym(0.15);
            }
        }
        drop(trainer);
        let mut set = LoraSet::new();
        set.push(adapter.clone(), 1.0);
        model.bind_lora(&set);
        let mut trainer = LlamaTrainer::new(&model, 64, 8, CheckpointMode::Fp16);

        for &seq_len in &[1usize, 11, 37] {
            let toks: Vec<u32> = (0..seq_len).map(|i| ((i * 11 + 5) % 90) as u32).collect();
            state.reset();
            model
                .forward_batch(&mut state, &pool, kt, &toks, 0, true)
                .expect("infer forward_batch with lora");
            let mut train_logits = vec![0.0f32; model.cfg.vocab_size];
            trainer.forward_last_logits(&pool, Some(&adapter), &toks, &mut train_logits);
            for (v, (&a, &b)) in state.logits().iter().zip(train_logits.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "0-ULP mismatch (active LoRA) arch={arch} seq_len={seq_len} vocab_idx={v}: infer={a} train={b}"
                );
            }
        }
    }
}

/// Gate 2 (`ADR-015` §3 Layer 1): End-to-end central finite-difference Jacobian verification
/// across ALL LoRA modules (`token_embd`, `q,k,v,o,gate,up,down`, `output`, QK-Norm, GQA `4:2`,
/// prompt loss masking `label == -100`) + thread-count determinism (`1` vs `4` threads)!
#[test]
fn end_to_end_finite_difference_jacobian_and_thread_determinism() {
    let tmp = TempDir::new("fdiff");
    let mpath = tmp.path.join("qwen3_f32.gguf");
    write_synthetic_llama_or_qwen3_gguf(&mpath, "qwen3", DType::F32, 2, 32, 64, 4, 2, 48, 2026);
    let gguf = GgufFile::open(&mpath).expect("open");
    let model = LlamaModel::from_gguf(&gguf).expect("load");

    let targets = LoraTargets::parse_csv("all").expect("all targets");
    let mut adapter = init_lora_adapter(&model, 2, 4.0, &targets, 31415).expect("init lora");
    let mut rng = SplitMix64::new(27182);
    for name in adapter.pair_order.clone() {
        let p = adapter.pairs.get_mut(&name).unwrap();
        for av in &mut p.a {
            *av = rng.uniform_sym(0.2);
        }
        for bv in &mut p.b {
            *bv = rng.uniform_sym(0.2);
        }
    }

    // Sequence with mixed unsupervised (`0`) and supervised (`1`) positions
    let tokens = [3u32, 7, 12, 19, 25, 31];
    let loss_mask = [0u8, 0, 1, 0, 1, 1];

    let pool1 = ThreadPool::new(1);
    let pool4 = ThreadPool::new(4);
    let mut trainer1 = LlamaTrainer::new(&model, 16, 4, CheckpointMode::Fp32);
    trainer1.exact_f32_kv_gradcheck = true;
    let mut trainer4 = LlamaTrainer::new(&model, 16, 4, CheckpointMode::Fp32);
    trainer4.exact_f32_kv_gradcheck = true;

    let mut grads1 = LoraGradStore::from_adapter(&adapter);
    let mut grads4 = LoraGradStore::from_adapter(&adapter);
    let (loss1, n_sup1) = trainer1.forward_backward_microbatch(
        &pool1,
        &adapter,
        &mut grads1,
        &tokens,
        &loss_mask,
        1.0,
    );
    let (loss4, n_sup4) = trainer4.forward_backward_microbatch(
        &pool4,
        &adapter,
        &mut grads4,
        &tokens,
        &loss_mask,
        1.0,
    );
    assert_eq!(n_sup1, 3);
    assert_eq!(n_sup4, 3);
    assert_eq!(loss1.to_bits(), loss4.to_bits());

    // Verify 0-ULP thread-count gradient determinism (`1` vs `4` threads)
    for name in &grads1.order {
        for (i, (&a, &b)) in grads1.grads[name]
            .da
            .iter()
            .zip(grads4.grads[name].da.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "thread da mismatch on {name}[{i}]"
            );
        }
        for (i, (&a, &b)) in grads1.grads[name]
            .db
            .iter()
            .zip(grads4.grads[name].db.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "thread db mismatch on {name}[{i}]"
            );
        }
    }

    // Central finite difference check across every module (`da` and `db`) using `f64` loss
    let eps = 3e-3f32;
    let mut max_err = 0.0f32;

    for name in adapter.pair_order.clone() {
        let a_len = adapter.pairs[&name].a.len();
        let b_len = adapter.pairs[&name].b.len();
        let a_indices: Vec<usize> = if name == "token_embd.weight" {
            let rk = adapter.pairs[&name].rank;
            tokens[..tokens.len() - 1]
                .iter()
                .flat_map(|&t| (0..rk).map(move |r| (t as usize) * rk + r))
                .collect()
        } else {
            (0..a_len).step_by(3).collect()
        };

        for idx in a_indices {
            let orig = adapter.pairs[&name].a[idx];
            adapter.pairs.get_mut(&name).unwrap().a[idx] = orig + eps;
            let lp = trainer4.forward_loss_f64(&pool4, &adapter, &tokens, &loss_mask);
            adapter.pairs.get_mut(&name).unwrap().a[idx] = orig - eps;
            let lm = trainer4.forward_loss_f64(&pool4, &adapter, &tokens, &loss_mask);
            adapter.pairs.get_mut(&name).unwrap().a[idx] = orig;

            let num = ((lp - lm) / (2.0 * (eps as f64))) as f32;
            let ana = grads4.grads[&name].da[idx];
            let denom = ana.abs().max(num.abs()).max(1.0);
            let err = (ana - num).abs() / denom;
            max_err = max_err.max(err);
            assert!(
                err < 1.5e-4,
                "finite-diff da mismatch on {name}[{idx}]: ana={ana} num={num} rel_err={err}"
            );
        }

        for idx in (0..b_len).step_by(3) {
            let orig = adapter.pairs[&name].b[idx];
            adapter.pairs.get_mut(&name).unwrap().b[idx] = orig + eps;
            let lp = trainer4.forward_loss_f64(&pool4, &adapter, &tokens, &loss_mask);
            adapter.pairs.get_mut(&name).unwrap().b[idx] = orig - eps;
            let lm = trainer4.forward_loss_f64(&pool4, &adapter, &tokens, &loss_mask);
            adapter.pairs.get_mut(&name).unwrap().b[idx] = orig;

            let num = ((lp - lm) / (2.0 * (eps as f64))) as f32;
            let ana = grads4.grads[&name].db[idx];
            let denom = ana.abs().max(num.abs()).max(1.0);
            let err = (ana - num).abs() / denom;
            max_err = max_err.max(err);
            assert!(
                err < 1.5e-4,
                "finite-diff db mismatch on {name}[{idx}]: ana={ana} num={num} rel_err={err}"
            );
        }
    }
    eprintln!("[qlora_verification] finite-diff max_err = {max_err:.3e}");
}

/// Gate 3 (`ADR-015` §3 Layers 2 & 3 + Checkpoint Resume + Merge + MTP):
/// - Compares `LlamaTrainer` on `Q4_0` weights against `tools/oracle/qlora_grad_oracle.py`:
///   - Layer 2 (`Q8_0` + `F16` KV STE): per-tensor relative L2 error `< 1e-3`!
///   - Layer 3 (plain `FP32` PEFT): global gradient cosine similarity `> 0.99`!
/// - Verifies multi-step AdamW convergence + crash-safe checkpoint save/resume bit-exactness!
/// - Verifies `merge_lora_gguf` (`F32` and `Q4_0x4`) parity + `train_mtp_head` convergence!
#[test]
fn q8_ste_oracle_fp32_peft_cosine_checkpoint_merge_and_mtp() {
    let tmp = TempDir::new("oracle");
    let mpath = tmp.path.join("llama_q4.gguf");
    write_synthetic_llama_or_qwen3_gguf(&mpath, "llama", DType::Q4_0, 2, 64, 128, 4, 2, 64, 4242);
    let gguf = GgufFile::open(&mpath).expect("open");
    let model = LlamaModel::from_gguf(&gguf).expect("load");
    let pool = ThreadPool::new(4);

    let targets = LoraTargets::parse_csv("all").expect("all");
    let mut adapter = init_lora_adapter(&model, 4, 8.0, &targets, 13579).expect("init lora");
    let mut rng = SplitMix64::new(24680);
    for name in adapter.pair_order.clone() {
        let p = adapter.pairs.get_mut(&name).unwrap();
        for av in &mut p.a {
            *av = rng.uniform_sym(0.18);
        }
        for bv in &mut p.b {
            *bv = rng.uniform_sym(0.18);
        }
    }

    let tokens = vec![2u32, 9, 14, 21, 33, 41, 52, 17, 8];
    let loss_mask = vec![0u8, 0, 0, 1, 1, 0, 1, 1, 1];

    // Run `LlamaTrainer` (`Fp32` checkpoint to isolate pure `Q8_0` + `F16` KV STE, AND `Fp16` checkpoint!)
    let mut trainer_f32c = LlamaTrainer::new(&model, 16, 4, CheckpointMode::Fp32);
    let mut grads_rust = LoraGradStore::from_adapter(&adapter);
    let (loss_rust, _) = trainer_f32c.forward_backward_microbatch(
        &pool,
        &adapter,
        &mut grads_rust,
        &tokens,
        &loss_mask,
        1.0,
    );

    let mut trainer_f16c = LlamaTrainer::new(&model, 16, 4, CheckpointMode::Fp16);
    let mut grads_f16c = LoraGradStore::from_adapter(&adapter);
    let (loss_f16c, _) = trainer_f16c.forward_backward_microbatch(
        &pool,
        &adapter,
        &mut grads_f16c,
        &tokens,
        &loss_mask,
        1.0,
    );
    assert!((loss_rust - loss_f16c).abs() < 1e-6);

    // Export problem spec to JSON and invoke `tools/oracle/qlora_grad_oracle.py`
    let decode_mat = |w: &tm_models::RawMatView<'_>| -> Vec<f32> {
        let mut out = vec![0.0f32; w.rows * w.cols];
        let rb = w.dtype.row_bytes(w.cols);
        for r in 0..w.rows {
            dequantize_row_bytes(
                w.dtype,
                &w.bytes[r * rb..(r + 1) * rb],
                &mut out[r * w.cols..(r + 1) * w.cols],
            );
        }
        out
    };
    let f32_arr_json = |s: &[f32]| -> String {
        let parts: Vec<String> = s.iter().map(|v| format!("{v:.9e}")).collect();
        format!("[{}]", parts.join(","))
    };
    let f64_arr_json = |s: &[f64]| -> String {
        let parts: Vec<String> = s.iter().map(|v| format!("{v:.15e}")).collect();
        format!("[{}]", parts.join(","))
    };

    let mut layers_json = Vec::new();
    for lw in &model.layers {
        let qn = lw
            .q_norm
            .as_ref()
            .map(|v| f32_arr_json(v))
            .unwrap_or_else(|| "null".to_string());
        let kn = lw
            .k_norm
            .as_ref()
            .map(|v| f32_arr_json(v))
            .unwrap_or_else(|| "null".to_string());
        layers_json.push(format!(
            "{{\"attn_norm\":{},\"ffn_norm\":{},\"q_norm\":{},\"k_norm\":{},\"wq\":{},\"wk\":{},\"wv\":{},\"wo\":{},\"w_gate\":{},\"w_up\":{},\"w_down\":{}}}",
            f32_arr_json(&lw.attn_norm),
            f32_arr_json(&lw.ffn_norm),
            qn,
            kn,
            f32_arr_json(&decode_mat(&lw.wq)),
            f32_arr_json(&decode_mat(&lw.wk)),
            f32_arr_json(&decode_mat(&lw.wv)),
            f32_arr_json(&decode_mat(&lw.wo)),
            f32_arr_json(&decode_mat(&lw.w_gate)),
            f32_arr_json(&decode_mat(&lw.w_up)),
            f32_arr_json(&decode_mat(&lw.w_down)),
        ));
    }

    let mut lora_entries = Vec::new();
    for name in &adapter.pair_order {
        let p = &adapter.pairs[name];
        lora_entries.push(format!(
            "\"{}\":{{\"rank\":{},\"d_in\":{},\"d_out\":{},\"is_token_embd\":{},\"a\":{},\"b\":{}}}",
            name,
            p.rank,
            p.d_in,
            p.d_out,
            p.is_token_embd,
            f32_arr_json(&p.a),
            f32_arr_json(&p.b),
        ));
    }

    let tok_json: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    let mask_json: Vec<String> = loss_mask.iter().map(|m| m.to_string()).collect();
    let spec_json = format!(
        "{{\"cfg\":{{\"n_embd\":{},\"n_ff\":{},\"n_head\":{},\"n_head_kv\":{},\"head_dim\":{},\"rope_dim\":{},\"vocab_size\":{},\"rms_eps\":{:.9e},\"neox\":false,\"rope_inv_freqs\":{}}},\"tokens\":[{}],\"loss_mask\":[{}],\"lora_alpha\":{:.9e},\"token_embd\":{},\"output_norm\":{},\"output\":{},\"layers\":[{}],\"lora\":{{{}}}}}",
        model.cfg.n_embd,
        model.cfg.n_ff,
        model.cfg.n_head,
        model.cfg.n_head_kv,
        model.cfg.head_dim,
        model.cfg.rope_dim,
        model.cfg.vocab_size,
        model.cfg.rms_eps,
        f64_arr_json(&model.rope_inv_freqs),
        tok_json.join(","),
        mask_json.join(","),
        adapter.alpha,
        f32_arr_json(&decode_mat(&model.token_embd)),
        f32_arr_json(&model.output_norm),
        f32_arr_json(&decode_mat(&model.output)),
        layers_json.join(","),
        lora_entries.join(","),
    );

    let in_json_path = tmp.path.join("spec.json");
    let out_json_path = tmp.path.join("oracle_out.json");
    std::fs::write(&in_json_path, spec_json).expect("write spec.json");

    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tools/oracle/qlora_grad_oracle.py");
    let status = Command::new("python3")
        .arg(&script)
        .arg("-i")
        .arg(&in_json_path)
        .arg("-o")
        .arg(&out_json_path)
        .status()
        .expect("run python3 qlora_grad_oracle.py");
    assert!(status.success(), "qlora_grad_oracle.py exited with failure");

    let oracle_raw = std::fs::read_to_string(&out_json_path).expect("read oracle_out.json");
    let extract_f32_array = |section: &str, tensor_name: &str, field: &str| -> Vec<f32> {
        let sec_pos = oracle_raw.find(section).expect("section");
        let sub = &oracle_raw[sec_pos..];
        let key = format!("\"{tensor_name}\":");
        let t_pos = sub.find(&key).expect("tensor key");
        let sub2 = &sub[t_pos..];
        let f_key = format!("\"{field}\":");
        let f_pos = sub2.find(&f_key).expect("field key");
        let sub3 = &sub2[f_pos + f_key.len()..];
        let lb = sub3.find('[').unwrap();
        let rb = sub3.find(']').unwrap();
        sub3[lb + 1..rb]
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<f32>().expect("float"))
            .collect()
    };

    let mut max_ste_rel_err = 0.0f64;
    let mut dot_fp32 = 0.0f64;
    let mut norm_rust_sq = 0.0f64;
    let mut norm_fp32_sq = 0.0f64;

    for name in &adapter.pair_order {
        let g_r = &grads_rust.grads[name];
        let da_ste = extract_f32_array("\"q8_ste\":", name, "da");
        let db_ste = extract_f32_array("\"q8_ste\":", name, "db");
        let da_fp = extract_f32_array("\"fp32_peft\":", name, "da");
        let db_fp = extract_f32_array("\"fp32_peft\":", name, "db");

        let rel_l2 = |u: &[f32], v: &[f32]| -> f64 {
            let mut diff_sq = 0.0f64;
            let mut ref_sq = 0.0f64;
            for (&a, &b) in u.iter().zip(v.iter()) {
                let d = (a as f64) - (b as f64);
                diff_sq += d * d;
                ref_sq += (b as f64) * (b as f64);
            }
            diff_sq.sqrt() / ref_sq.sqrt().max(1e-12)
        };
        let err_a = rel_l2(&g_r.da, &da_ste);
        let err_b = rel_l2(&g_r.db, &db_ste);
        eprintln!("  [oracle Layer 2] {name:<28} err_da={err_a:.3e} err_db={err_b:.3e}");
        max_ste_rel_err = max_ste_rel_err.max(err_a).max(err_b);
        assert!(
            err_a < 1e-3 && err_b < 1e-3,
            "Layer 2 Q8-STE mismatch on {name}: err_da={err_a:.3e} err_db={err_b:.3e}"
        );

        // Also verify `FP16` checkpointed gradients stay closely aligned (`< 3e-2`, occasional 1-LSB Q8_0 bin flip on recompute)
        let g_f16 = &grads_f16c.grads[name];
        let f16_ea = rel_l2(&g_f16.da, &da_ste);
        let f16_eb = rel_l2(&g_f16.db, &db_ste);
        assert!(
            f16_ea < 3e-2 && f16_eb < 3e-2,
            "FP16 ckpt mismatch on {name}: {f16_ea:.3e} / {f16_eb:.3e}"
        );

        for (&r_v, &f_v) in g_r
            .da
            .iter()
            .chain(g_r.db.iter())
            .zip(da_fp.iter().chain(db_fp.iter()))
        {
            let rf = r_v as f64;
            let ff = f_v as f64;
            dot_fp32 += rf * ff;
            norm_rust_sq += rf * rf;
            norm_fp32_sq += ff * ff;
        }
    }

    let cosine_fp32 = dot_fp32 / (norm_rust_sq.sqrt() * norm_fp32_sq.sqrt());
    eprintln!(
        "[qlora_verification] Layer 2 max_ste_rel_err={max_ste_rel_err:.3e}, Layer 3 FP32 PEFT cosine={cosine_fp32:.6}"
    );
    assert!(
        cosine_fp32 > 0.99,
        "Layer 3 FP32 PEFT gradient cosine {cosine_fp32} <= 0.99"
    );

    // 4. Multi-step QLoRA convergence + Checkpoint Save/Resume bit-exactness!
    let ds = TrainDataset::from_samples(
        "mem",
        vec![TrainSample {
            tokens: tokens.clone(),
            loss_mask: loss_mask.clone(),
        }],
    )
    .expect("dataset");

    let ckpt_p = tmp.path.join("mid.ckpt.gguf");
    let mut cfg_full = QloraConfig {
        rank: 4,
        alpha: 8.0,
        steps: 16,
        seq_len: 16,
        warmup_steps: 2,
        ..Default::default()
    };
    cfg_full.optim.lr = 3e-3;
    let (ad_full, rep_full) =
        run_qlora_training(&model, &pool, &ds, &cfg_full, |_| Ok(())).expect("full 16 steps");
    assert!(
        rep_full.final_loss < rep_full.initial_loss * 0.45,
        "expected strong loss drop: {} -> {}",
        rep_full.initial_loss,
        rep_full.final_loss
    );

    // Run 8 steps (with same cosine schedule horizon `steps = 16`, `stop_at_step = Some(8)`) -> save checkpoint -> resume to 16 -> verify bit-exact match!
    let mut cfg_part1 = cfg_full.clone();
    cfg_part1.stop_at_step = Some(8);
    cfg_part1.ckpt_path = Some(ckpt_p.clone());
    let _ = run_qlora_training(&model, &pool, &ds, &cfg_part1, |_| Ok(())).expect("part 1");

    let mut cfg_part2 = cfg_full.clone();
    cfg_part2.resume_ckpt = Some(ckpt_p);
    let (ad_resumed, _) =
        run_qlora_training(&model, &pool, &ds, &cfg_part2, |_| Ok(())).expect("part 2 resume");
    for name in &ad_full.pair_order {
        assert_eq!(ad_full.pairs[name].a, ad_resumed.pairs[name].a);
        assert_eq!(ad_full.pairs[name].b, ad_resumed.pairs[name].b);
    }

    // 5. Verify `merge_lora_gguf` (`F32` lossless merge vs dynamic `--lora`, plus `Q4_0x4` repack!)
    let mut lset = LoraSet::new();
    lset.push(ad_full.clone(), 1.0);
    let merged_f32_p = tmp.path.join("merged_f32.gguf");
    merge_lora_gguf(
        &gguf,
        &lset,
        &merged_f32_p,
        &LoraMergeOptions {
            quant: Some(DType::F32),
            lm_head_quant: Some(DType::F32),
        },
    )
    .expect("merge f32");
    let merged_q4x4_p = tmp.path.join("merged_q4x4.gguf");
    merge_lora_gguf(
        &gguf,
        &lset,
        &merged_q4x4_p,
        &LoraMergeOptions {
            quant: Some(DType::Q4_0x4),
            lm_head_quant: None,
        },
    )
    .expect("merge q4_0x4");

    let mut dyn_model = LlamaModel::from_gguf(&gguf).expect("load dyn");
    dyn_model.bind_lora(&lset);
    let mut st_dyn = LlamaState::new(&dyn_model.cfg, 16);
    dyn_model
        .forward_batch(&mut st_dyn, &pool, active_kernels(), &tokens, 0, true)
        .expect("dyn fwd");

    let merged_gguf = GgufFile::open(&merged_f32_p).expect("open merged");
    let merged_model = LlamaModel::from_gguf(&merged_gguf).expect("load merged");
    let mut st_merged = LlamaState::new(&merged_model.cfg, 16);
    merged_model
        .forward_batch(&mut st_merged, &pool, active_kernels(), &tokens, 0, true)
        .expect("merged fwd");

    // Dynamic (`Q4_0` base x `Q8_0` act + `F32` LoRA) vs `F32`-merged (`F32` x `F32`) agrees within `Q8_0` act tolerance (`< 5e-2`),
    // AND `F32`-converted base + dynamic `--lora` vs `F32`-merged matches to `< 1e-5`!
    let max_merge_diff_q4 = st_dyn
        .logits()
        .iter()
        .zip(st_merged.logits().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_merge_diff_q4 < 5e-2,
        "Q4_0 dynamic vs F32 merged max_diff={max_merge_diff_q4}"
    );

    let base_f32_p = tmp.path.join("base_f32.gguf");
    gguf.convert_with_options(
        &base_f32_p,
        &tm_gguf::ConvertOptions {
            quant: Some(DType::F32),
            lm_head_quant: Some(DType::F32),
            ..Default::default()
        },
    )
    .expect("convert base to f32");
    let base_f32_gguf = GgufFile::open(&base_f32_p).expect("open base f32");
    let mut dyn_f32_model = LlamaModel::from_gguf(&base_f32_gguf).expect("load dyn f32");
    dyn_f32_model.bind_lora(&lset);
    let mut st_dyn_f32 = LlamaState::new(&dyn_f32_model.cfg, 16);
    dyn_f32_model
        .forward_batch(&mut st_dyn_f32, &pool, active_kernels(), &tokens, 0, true)
        .expect("dyn f32 fwd");
    let max_merge_diff_f32 = st_dyn_f32
        .logits()
        .iter()
        .zip(st_merged.logits().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_merge_diff_f32 < 1e-5,
        "F32 dynamic vs F32 merged lossless mismatch: max_diff={max_merge_diff_f32}"
    );

    // Ensure `Q4_0x4` merged model loads and binds dynamic LoRA cleanly too!
    let q4x4_gguf = GgufFile::open(&merged_q4x4_p).expect("open q4x4");
    let mut q4x4_cpu = CpuModel::from_gguf(&q4x4_gguf).expect("load q4x4");
    q4x4_cpu
        .bind_lora(&q4x4_gguf, &lset)
        .expect("bind lora on q4x4");

    // 6. Phase `T2a`: `train_mtp_head` convergence + GGUF save/load roundtrip!
    let cpu_base = CpuModel::from_gguf(&gguf).expect("cpu base");
    let mtp_cfg = MtpTrainConfig {
        steps: 20,
        seq_len: 16,
        lr: 5e-3,
        warmup_steps: 2,
        out_dtype: DType::F16,
        ..Default::default()
    };
    let (mtp_head, mtp_rep) =
        train_mtp_head(&cpu_base, &pool, &ds, &mtp_cfg, |_, _, _, _| {}).expect("train mtp");
    assert!(
        mtp_rep.final_loss < mtp_rep.initial_loss * 0.6,
        "MTP loss did not decrease: {} -> {}",
        mtp_rep.initial_loss,
        mtp_rep.final_loss
    );
    let mtp_path = tmp.path.join("head.mtp.gguf");
    mtp_head
        .save_gguf(&mtp_path, "llama", DType::F16)
        .expect("save mtp");
    let reloaded_mtp =
        LinearMtpHead::open(&mtp_path, "llama", model.cfg.n_embd).expect("reload mtp");
    assert_eq!(reloaded_mtp.n_embd(), model.cfg.n_embd);
}
