#![allow(clippy::vec_init_then_push)]

//! Phase `T3` Verification Suite (`ADR-015`):
//! 1. **0-ULP Training Forward Parity Gate** (`to_bits()`) for `Qwen3.5`, `Gemma 4`, and `LFM2`
//!    (both synthetic `Q4_0`/`F32` and real `Qwen3.5-0.8B-Q4_0.gguf` / `gemma-4-E2B-it-Q4_0.gguf`
//!    when present in `../../models/`), across `T = 1`, `T = 9`, and `T = 35` (`> MAX_CPU_BATCH`),
//!    with BOTH `None` (zero LoRA) AND random non-zero `LoraAdapter` (`--targets all`).
//! 2. **End-to-End `f64` Central Finite-Difference Jacobian Verification (`< 3e-4`)** on synthetic
//!    `F32` models exercising EVERY hybrid trap (`GDN` tiled `hv > hk` + `d_conv=4` + `SiLU` before
//!    split + `L2` on `q/k` only + `SiLU(z)` gate; `Gemma 4` cross-layer shared KV fan-in + `wv=None`
//!    KV-tying + SWA + `attn_scale=1.0` + unit V norm + global & per-layer PLE + `out_scale` +
//!    `final_logit_softcapping`; `LFM2` `ShortConv` + SWA `GQA`).
//! 3. **Multi-Step QLoRA Convergence (`run_cpu_qlora_training`)** on quantized hybrid GGUF models.

use std::path::{Path, PathBuf};

use tm_core::{DType, quantize_row_to_bytes};
use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::{GgufFile, MetadataValue, NewGgufTensor, write_gguf_file};
use tm_lora::LoraSet;
use tm_models::{Gemma4Model, Gemma4State, Lfm2Model, Lfm2State, Qwen35Model, Qwen35State};
use tm_runtime::CpuModel;
use tm_train::{
    CheckpointMode, Gemma4Trainer, Lfm2Trainer, LoraGradStore, LoraTargets, QloraConfig,
    Qwen35Trainer, SplitMix64, TrainDataset, TrainSample, init_lora_adapter_gemma4,
    init_lora_adapter_lfm2, init_lora_adapter_qwen35, run_cpu_qlora_training,
};

fn encode_tensor(
    rng: &mut SplitMix64,
    cols: usize,
    rows: usize,
    dtype: DType,
    scale: f32,
) -> Vec<u8> {
    let vals: Vec<f32> = (0..rows * cols).map(|_| rng.uniform_sym(scale)).collect();
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

fn encode_norm_f32(rng: &mut SplitMix64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for _ in 0..len {
        let v = 1.0f32 + rng.uniform_sym(0.15);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn write_synthetic_qwen35_gguf(path: &Path, mat_dtype: DType, seed: u64) {
    // Choose dimensions >= 32 when `Q4_0` so block alignment holds (`s_dim=32`, `d=32`, `ff=64`).
    // Also exercise `hv = 4 > hk = 2` (`kh = vh % hk` tiled grouping!) and partial RoPE (`rope_dim = 8 < hd = 16`).
    let mut rng = SplitMix64::new(seed);
    let d = 32usize;
    let ff = 64usize;
    let n_layers = 2usize; // blk.0 = GDN, blk.1 = FullAttn
    let n_head = 2usize;
    let n_head_kv = 1usize;
    let hd = 16usize;
    let rope_dim = 8usize;
    let q_dim = n_head * hd;
    let qg_dim = q_dim * 2;
    let kv_dim = n_head_kv * hd;
    let d_conv = 4usize;
    let hk = 2usize;
    let hv = 4usize;
    let s_dim = 32usize;
    let d_inner = hv * s_dim; // 128
    let conv_dim = 2 * hk * s_dim + d_inner; // 256
    let vocab = 32usize;

    let meta = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String("qwen35".to_string()),
        ),
        (
            "qwen35.block_count".to_string(),
            MetadataValue::U32(n_layers as u32),
        ),
        (
            "qwen35.embedding_length".to_string(),
            MetadataValue::U32(d as u32),
        ),
        (
            "qwen35.feed_forward_length".to_string(),
            MetadataValue::U32(ff as u32),
        ),
        (
            "qwen35.attention.head_count".to_string(),
            MetadataValue::U32(n_head as u32),
        ),
        (
            "qwen35.attention.head_count_kv".to_string(),
            MetadataValue::U32(n_head_kv as u32),
        ),
        (
            "qwen35.attention.key_length".to_string(),
            MetadataValue::U32(hd as u32),
        ),
        (
            "qwen35.rope.dimension_count".to_string(),
            MetadataValue::U32(rope_dim as u32),
        ),
        (
            "qwen35.rope.freq_base".to_string(),
            MetadataValue::F32(10000.0),
        ),
        (
            "qwen35.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValue::F32(1e-5),
        ),
        (
            "qwen35.full_attention_interval".to_string(),
            MetadataValue::U32(2),
        ),
        (
            "qwen35.ssm.conv_kernel".to_string(),
            MetadataValue::U32(d_conv as u32),
        ),
        (
            "qwen35.ssm.state_size".to_string(),
            MetadataValue::U32(s_dim as u32),
        ),
        (
            "qwen35.ssm.group_count".to_string(),
            MetadataValue::U32(hk as u32),
        ),
        (
            "qwen35.ssm.time_step_rank".to_string(),
            MetadataValue::U32(hv as u32),
        ),
        (
            "qwen35.ssm.inner_size".to_string(),
            MetadataValue::U32(d_inner as u32),
        ),
        (
            "qwen35.vocab_size".to_string(),
            MetadataValue::U32(vocab as u32),
        ),
        ("qwen35.context_length".to_string(), MetadataValue::U32(128)),
    ];

    struct T {
        name: String,
        dims: Vec<usize>,
        dt: DType,
        bytes: Vec<u8>,
    }
    let mut ts = Vec::new();
    ts.push(T {
        name: "token_embd.weight".into(),
        dims: vec![d, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, vocab, mat_dtype, 0.35),
    });
    ts.push(T {
        name: "output_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "output.weight".into(),
        dims: vec![d, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, vocab, mat_dtype, 0.35),
    });

    // Layer 0: GDN
    ts.push(T {
        name: "blk.0.attn_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.0.post_attention_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.0.attn_qkv.weight".into(),
        dims: vec![d, conv_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, conv_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.0.attn_gate.weight".into(),
        dims: vec![d, d_inner],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, d_inner, mat_dtype, 0.3),
    });
    // `w_alpha` and `w_beta` have cols=d=32, rows=hv=4
    ts.push(T {
        name: "blk.0.ssm_alpha.weight".into(),
        dims: vec![d, hv],
        dt: DType::F32,
        bytes: encode_tensor(&mut rng, d, hv, DType::F32, 0.3),
    });
    ts.push(T {
        name: "blk.0.ssm_beta.weight".into(),
        dims: vec![d, hv],
        dt: DType::F32,
        bytes: encode_tensor(&mut rng, d, hv, DType::F32, 0.3),
    });
    ts.push(T {
        name: "blk.0.ssm_conv1d.weight".into(),
        dims: vec![d_conv, conv_dim],
        dt: DType::F32,
        bytes: encode_tensor(&mut rng, d_conv, conv_dim, DType::F32, 0.4),
    });
    // Negative `ssm_a` (`-exp(...)`) so `g_exp = exp(softplus * ssm_a)` is in `(0, 1)`
    let mut ssm_a_bytes = Vec::new();
    for _ in 0..hv {
        let v = -0.35f32 - rng.uniform_sym(0.15).abs();
        ssm_a_bytes.extend_from_slice(&v.to_le_bytes());
    }
    ts.push(T {
        name: "blk.0.ssm_a".into(),
        dims: vec![hv],
        dt: DType::F32,
        bytes: ssm_a_bytes,
    });
    ts.push(T {
        name: "blk.0.ssm_dt.bias".into(),
        dims: vec![hv],
        dt: DType::F32,
        bytes: encode_tensor(&mut rng, hv, 1, DType::F32, 0.4),
    });
    ts.push(T {
        name: "blk.0.ssm_norm.weight".into(),
        dims: vec![s_dim],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, s_dim),
    });
    ts.push(T {
        name: "blk.0.ssm_out.weight".into(),
        dims: vec![d_inner, d],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d_inner, d, mat_dtype, 0.3),
    });
    for (suf, c_in, r_out) in [
        ("ffn_gate.weight", d, ff),
        ("ffn_up.weight", d, ff),
        ("ffn_down.weight", ff, d),
    ] {
        ts.push(T {
            name: format!("blk.0.{suf}"),
            dims: vec![c_in, r_out],
            dt: mat_dtype,
            bytes: encode_tensor(&mut rng, c_in, r_out, mat_dtype, 0.3),
        });
    }

    // Layer 1: FullAttn
    ts.push(T {
        name: "blk.1.attn_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.1.post_attention_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.1.attn_q_norm.weight".into(),
        dims: vec![hd],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, hd),
    });
    ts.push(T {
        name: "blk.1.attn_k_norm.weight".into(),
        dims: vec![hd],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, hd),
    });
    ts.push(T {
        name: "blk.1.attn_q.weight".into(),
        dims: vec![d, qg_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, qg_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.1.attn_k.weight".into(),
        dims: vec![d, kv_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, kv_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.1.attn_v.weight".into(),
        dims: vec![d, kv_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, kv_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.1.attn_output.weight".into(),
        dims: vec![q_dim, d],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, q_dim, d, mat_dtype, 0.3),
    });
    for (suf, c_in, r_out) in [
        ("ffn_gate.weight", d, ff),
        ("ffn_up.weight", d, ff),
        ("ffn_down.weight", ff, d),
    ] {
        ts.push(T {
            name: format!("blk.1.{suf}"),
            dims: vec![c_in, r_out],
            dt: mat_dtype,
            bytes: encode_tensor(&mut rng, c_in, r_out, mat_dtype, 0.3),
        });
    }

    let views: Vec<NewGgufTensor<'_>> = ts
        .iter()
        .map(|t| NewGgufTensor {
            name: &t.name,
            dims: &t.dims,
            dtype: t.dt,
            data: &t.bytes,
        })
        .collect();
    write_gguf_file(path, &meta, &views, 32).expect("write synthetic qwen35 gguf");
}

fn write_synthetic_gemma4_gguf(path: &Path, mat_dtype: DType, seed: u64) {
    // 3 layers:
    // - blk.0: SWA own-KV (`wv = Some`)
    // - blk.1: Full-attn own-KV (`wv = None`, exercising KV-tied `v = k`!)
    // - blk.2: SWA shared-KV (`kv_source[2] = 0`, exercising cross-layer KV gradient fan-in!)
    // Also exercises global PLE + per-layer PLE + `layer_output_scale` + `final_logit_softcapping`.
    let mut rng = SplitMix64::new(seed);
    let d = 32usize;
    let n_layers = 3usize;
    let n_head = 2usize;
    let n_head_kv = 1usize;
    let hd_full = 32usize;
    let hd_swa = 32usize;
    let ple_dim = 32usize;
    let ple_total = n_layers * ple_dim;
    let vocab = 32usize;

    let meta = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String("gemma4".to_string()),
        ),
        (
            "gemma4.block_count".to_string(),
            MetadataValue::U32(n_layers as u32),
        ),
        (
            "gemma4.embedding_length".to_string(),
            MetadataValue::U32(d as u32),
        ),
        (
            "gemma4.feed_forward_length".to_string(),
            MetadataValue::Array(vec![
                MetadataValue::U32(64),
                MetadataValue::U32(96),
                MetadataValue::U32(64),
            ]),
        ),
        (
            "gemma4.attention.head_count".to_string(),
            MetadataValue::U32(n_head as u32),
        ),
        (
            "gemma4.attention.head_count_kv".to_string(),
            MetadataValue::U32(n_head_kv as u32),
        ),
        (
            "gemma4.attention.key_length".to_string(),
            MetadataValue::U32(hd_full as u32),
        ),
        (
            "gemma4.attention.key_length_swa".to_string(),
            MetadataValue::U32(hd_swa as u32),
        ),
        (
            "gemma4.rope.dimension_count".to_string(),
            MetadataValue::U32(16),
        ),
        (
            "gemma4.rope.dimension_count_swa".to_string(),
            MetadataValue::U32(16),
        ),
        (
            "gemma4.rope.freq_base".to_string(),
            MetadataValue::F32(100_000.0),
        ),
        (
            "gemma4.rope.freq_base_swa".to_string(),
            MetadataValue::F32(10_000.0),
        ),
        (
            "gemma4.attention.sliding_window".to_string(),
            MetadataValue::U32(3),
        ),
        (
            "gemma4.attention.sliding_window_pattern".to_string(),
            MetadataValue::Array(vec![
                MetadataValue::Bool(true),
                MetadataValue::Bool(false),
                MetadataValue::Bool(true),
            ]),
        ),
        (
            "gemma4.attention.shared_kv_layers".to_string(),
            MetadataValue::U32(1),
        ),
        (
            "gemma4.embedding_length_per_layer_input".to_string(),
            MetadataValue::U32(ple_dim as u32),
        ),
        (
            "gemma4.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValue::F32(1e-5),
        ),
        (
            "gemma4.final_logit_softcapping".to_string(),
            MetadataValue::F32(15.0),
        ),
        (
            "gemma4.vocab_size".to_string(),
            MetadataValue::U32(vocab as u32),
        ),
        ("gemma4.context_length".to_string(), MetadataValue::U32(128)),
    ];

    struct T {
        name: String,
        dims: Vec<usize>,
        dt: DType,
        bytes: Vec<u8>,
    }
    let mut ts = Vec::new();
    ts.push(T {
        name: "token_embd.weight".into(),
        dims: vec![d, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, vocab, mat_dtype, 0.25),
    });
    ts.push(T {
        name: "output_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "output.weight".into(),
        dims: vec![d, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, vocab, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "per_layer_token_embd.weight".into(),
        dims: vec![ple_total, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, ple_total, vocab, mat_dtype, 0.25),
    });
    ts.push(T {
        name: "per_layer_model_proj.weight".into(),
        dims: vec![d, ple_total],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, ple_total, mat_dtype, 0.25),
    });
    ts.push(T {
        name: "per_layer_proj_norm.weight".into(),
        dims: vec![ple_dim],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, ple_dim),
    });

    let ffs = [64usize, 96, 64];
    for (il, &ff) in ffs.iter().enumerate().take(n_layers) {
        let p = format!("blk.{il}");
        let q_dim = n_head * hd_full;
        let kv_dim = n_head_kv * hd_full;
        for nname in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
            "post_norm.weight",
        ] {
            ts.push(T {
                name: format!("{p}.{nname}"),
                dims: vec![d],
                dt: DType::F32,
                bytes: encode_norm_f32(&mut rng, d),
            });
        }
        ts.push(T {
            name: format!("{p}.attn_q_norm.weight"),
            dims: vec![hd_full],
            dt: DType::F32,
            bytes: encode_norm_f32(&mut rng, hd_full),
        });
        ts.push(T {
            name: format!("{p}.attn_q.weight"),
            dims: vec![d, q_dim],
            dt: mat_dtype,
            bytes: encode_tensor(&mut rng, d, q_dim, mat_dtype, 0.3),
        });
        if il < 2 {
            ts.push(T {
                name: format!("{p}.attn_k_norm.weight"),
                dims: vec![hd_full],
                dt: DType::F32,
                bytes: encode_norm_f32(&mut rng, hd_full),
            });
            ts.push(T {
                name: format!("{p}.attn_k.weight"),
                dims: vec![d, kv_dim],
                dt: mat_dtype,
                bytes: encode_tensor(&mut rng, d, kv_dim, mat_dtype, 0.3),
            });
            if il == 0 {
                // `il == 0` has `wv`; `il == 1` omits `wv` (`wv == None` KV-tied)!
                ts.push(T {
                    name: format!("{p}.attn_v.weight"),
                    dims: vec![d, kv_dim],
                    dt: mat_dtype,
                    bytes: encode_tensor(&mut rng, d, kv_dim, mat_dtype, 0.3),
                });
            }
        }
        ts.push(T {
            name: format!("{p}.attn_output.weight"),
            dims: vec![q_dim, d],
            dt: mat_dtype,
            bytes: encode_tensor(&mut rng, q_dim, d, mat_dtype, 0.3),
        });
        for (suf, c_in, r_out) in [
            ("ffn_gate.weight", d, ff),
            ("ffn_up.weight", d, ff),
            ("ffn_down.weight", ff, d),
            ("inp_gate.weight", d, ple_dim),
            ("proj.weight", ple_dim, d),
        ] {
            ts.push(T {
                name: format!("{p}.{suf}"),
                dims: vec![c_in, r_out],
                dt: mat_dtype,
                bytes: encode_tensor(&mut rng, c_in, r_out, mat_dtype, 0.25),
            });
        }
        let scale_v = (0.85f32 + 0.08 * (il as f32)).to_le_bytes().to_vec();
        ts.push(T {
            name: format!("{p}.layer_output_scale.weight"),
            dims: vec![1],
            dt: DType::F32,
            bytes: scale_v,
        });
    }

    let views: Vec<NewGgufTensor<'_>> = ts
        .iter()
        .map(|t| NewGgufTensor {
            name: &t.name,
            dims: &t.dims,
            dtype: t.dt,
            data: &t.bytes,
        })
        .collect();
    write_gguf_file(path, &meta, &views, 32).expect("write synthetic gemma4 gguf");
}

fn write_synthetic_lfm2_gguf(path: &Path, mat_dtype: DType, seed: u64) {
    // 2 layers: blk.0 = ShortConv (`head_count_kv = 0`, `l_cache = 4`), blk.1 = GQA (`head_count_kv = 1`, `n_swa = 3`)
    let mut rng = SplitMix64::new(seed);
    let d = 32usize;
    let ff = 64usize;
    let n_head = 2usize;
    let hd = 16usize;
    let q_dim = n_head * hd;
    let kv_dim = hd;
    let l_cache = 4usize;
    let vocab = 32usize;

    let meta = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String("lfm2".to_string()),
        ),
        ("lfm2.block_count".to_string(), MetadataValue::U32(2)),
        (
            "lfm2.embedding_length".to_string(),
            MetadataValue::U32(d as u32),
        ),
        (
            "lfm2.feed_forward_length".to_string(),
            MetadataValue::U32(ff as u32),
        ),
        (
            "lfm2.attention.head_count".to_string(),
            MetadataValue::U32(n_head as u32),
        ),
        (
            "lfm2.attention.head_count_kv".to_string(),
            MetadataValue::Array(vec![MetadataValue::U32(0), MetadataValue::U32(1)]),
        ),
        (
            "lfm2.attention.key_length".to_string(),
            MetadataValue::U32(hd as u32),
        ),
        (
            "lfm2.rope.dimension_count".to_string(),
            MetadataValue::U32(hd as u32),
        ),
        (
            "lfm2.shortconv.l_cache".to_string(),
            MetadataValue::U32(l_cache as u32),
        ),
        (
            "lfm2.attention.sliding_window".to_string(),
            MetadataValue::U32(3),
        ),
        (
            "lfm2.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValue::F32(1e-5),
        ),
        (
            "lfm2.vocab_size".to_string(),
            MetadataValue::U32(vocab as u32),
        ),
        ("lfm2.context_length".to_string(), MetadataValue::U32(128)),
    ];

    struct T {
        name: String,
        dims: Vec<usize>,
        dt: DType,
        bytes: Vec<u8>,
    }
    let mut ts = Vec::new();
    ts.push(T {
        name: "token_embd.weight".into(),
        dims: vec![d, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, vocab, mat_dtype, 0.35),
    });
    ts.push(T {
        name: "token_embd_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "output.weight".into(),
        dims: vec![d, vocab],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, vocab, mat_dtype, 0.35),
    });

    // blk.0: ShortConv
    ts.push(T {
        name: "blk.0.attn_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.0.ffn_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.0.shortconv.in_proj.weight".into(),
        dims: vec![d, 3 * d],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, 3 * d, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.0.shortconv.conv.weight".into(),
        dims: vec![l_cache, d],
        dt: DType::F32,
        bytes: encode_tensor(&mut rng, l_cache, d, DType::F32, 0.4),
    });
    ts.push(T {
        name: "blk.0.shortconv.out_proj.weight".into(),
        dims: vec![d, d],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, d, mat_dtype, 0.3),
    });
    for (suf, c_in, r_out) in [
        ("ffn_gate.weight", d, ff),
        ("ffn_up.weight", d, ff),
        ("ffn_down.weight", ff, d),
    ] {
        ts.push(T {
            name: format!("blk.0.{suf}"),
            dims: vec![c_in, r_out],
            dt: mat_dtype,
            bytes: encode_tensor(&mut rng, c_in, r_out, mat_dtype, 0.3),
        });
    }

    // blk.1: GQA
    ts.push(T {
        name: "blk.1.attn_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.1.ffn_norm.weight".into(),
        dims: vec![d],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, d),
    });
    ts.push(T {
        name: "blk.1.attn_q_norm.weight".into(),
        dims: vec![hd],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, hd),
    });
    ts.push(T {
        name: "blk.1.attn_k_norm.weight".into(),
        dims: vec![hd],
        dt: DType::F32,
        bytes: encode_norm_f32(&mut rng, hd),
    });
    ts.push(T {
        name: "blk.1.attn_q.weight".into(),
        dims: vec![d, q_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, q_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.1.attn_k.weight".into(),
        dims: vec![d, kv_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, kv_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.1.attn_v.weight".into(),
        dims: vec![d, kv_dim],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, d, kv_dim, mat_dtype, 0.3),
    });
    ts.push(T {
        name: "blk.1.attn_output.weight".into(),
        dims: vec![q_dim, d],
        dt: mat_dtype,
        bytes: encode_tensor(&mut rng, q_dim, d, mat_dtype, 0.3),
    });
    for (suf, c_in, r_out) in [
        ("ffn_gate.weight", d, ff),
        ("ffn_up.weight", d, ff),
        ("ffn_down.weight", ff, d),
    ] {
        ts.push(T {
            name: format!("blk.1.{suf}"),
            dims: vec![c_in, r_out],
            dt: mat_dtype,
            bytes: encode_tensor(&mut rng, c_in, r_out, mat_dtype, 0.3),
        });
    }

    let views: Vec<NewGgufTensor<'_>> = ts
        .iter()
        .map(|t| NewGgufTensor {
            name: &t.name,
            dims: &t.dims,
            dtype: t.dt,
            data: &t.bytes,
        })
        .collect();
    write_gguf_file(path, &meta, &views, 32).expect("write synthetic lfm2 gguf");
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
fn t3_zero_ulp_forward_parity_qwen35_gemma4_and_lfm2() {
    let dir = std::env::temp_dir().join(format!("tm_t3_fwd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let p_qwen = dir.join("qwen35_q4.gguf");
    let p_gemma = dir.join("gemma4_q4.gguf");
    let p_lfm = dir.join("lfm2_q4.gguf");
    write_synthetic_qwen35_gguf(&p_qwen, DType::Q4_0, 0x1111);
    write_synthetic_gemma4_gguf(&p_gemma, DType::Q4_0, 0x2222);
    write_synthetic_lfm2_gguf(&p_lfm, DType::Q4_0, 0x3333);

    let pool = ThreadPool::new(2);
    let kt = active_kernels();
    let targets = LoraTargets::parse_csv("all").unwrap();

    // Test T=1, T=9, and T=35 (> MAX_CPU_BATCH=32)
    let seqs: [Vec<u32>; 3] = [
        vec![3],
        vec![1, 5, 2, 9, 14, 7, 3, 11, 6],
        (0..35).map(|i| ((i * 7 + 3) % 31) as u32).collect(),
    ];

    // 1. Qwen3.5 synthetic Q4_0
    {
        let gguf = GgufFile::open(&p_qwen).unwrap();
        let mut model = Qwen35Model::from_gguf(&gguf).unwrap();
        let mut state = Qwen35State::new(&model, 64);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];

        {
            let mut trainer = Qwen35Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
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
                        "Qwen3.5 0-ULP mismatch (no LoRA) len={} idx={i}: {a} vs {b}",
                        seq.len()
                    );
                }
            }
        }

        let mut ad = init_lora_adapter_qwen35(&model, 4, 8.0, &targets, 77).unwrap();
        randomize_lora_b(&mut SplitMix64::new(88), &mut ad);
        let mut set = LoraSet::new();
        set.push(ad.clone(), 1.0);
        model.bind_lora(&set);
        state.lora_scratch.ensure_rank(4);
        let mut trainer = Qwen35Trainer::new(&model, 64, 4, CheckpointMode::Fp16);
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
                    "Qwen3.5 0-ULP mismatch (with LoRA) len={} idx={i}: {a} vs {b}",
                    seq.len()
                );
            }
        }
    }

    // 2. Gemma 4 synthetic Q4_0
    {
        let gguf = GgufFile::open(&p_gemma).unwrap();
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
                        "Gemma 4 0-ULP mismatch (no LoRA) len={} idx={i}: {a} vs {b}",
                        seq.len()
                    );
                }
            }
        }

        let mut ad = init_lora_adapter_gemma4(&model, 4, 8.0, &targets, 99).unwrap();
        randomize_lora_b(&mut SplitMix64::new(101), &mut ad);
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
                    "Gemma 4 0-ULP mismatch (with LoRA) len={} idx={i}: {a} vs {b}",
                    seq.len()
                );
            }
        }
    }

    // 3. LFM2 synthetic Q4_0
    {
        let gguf = GgufFile::open(&p_lfm).unwrap();
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
                        "LFM2 0-ULP mismatch (no LoRA) len={} idx={i}: {a} vs {b}",
                        seq.len()
                    );
                }
            }
        }

        let mut ad = init_lora_adapter_lfm2(&model, 4, 8.0, &targets, 123).unwrap();
        randomize_lora_b(&mut SplitMix64::new(124), &mut ad);
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
                    "LFM2 0-ULP mismatch (with LoRA) len={} idx={i}: {a} vs {b}",
                    seq.len()
                );
            }
        }
    }

    // 4. Real local models 0-ULP gate if present in workspace `models/`
    let ws_models = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let real_qwen = ws_models.join("Qwen3.5-0.8B-Q4_0.gguf");
    if real_qwen.exists() {
        let gguf = GgufFile::open(&real_qwen).unwrap();
        let model = Qwen35Model::from_gguf(&gguf).unwrap();
        let mut state = Qwen35State::new(&model, 32);
        let mut trainer = Qwen35Trainer::new(&model, 32, 4, CheckpointMode::Fp16);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];
        let prompt = [151644u32, 872, 198, 785, 6722, 315, 9625, 374, 220];
        model
            .forward_batch(&mut state, &pool, kt, &prompt, 0, true)
            .unwrap();
        trainer.forward_last_logits(&pool, None, &prompt, &mut tr_logits);
        for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Real Qwen3.5-0.8B 0-ULP mismatch at {i}"
            );
        }
    }

    let real_gemma = ws_models.join("gemma-4-E2B-it-Q4_0.gguf");
    if real_gemma.exists() {
        let gguf = GgufFile::open(&real_gemma).unwrap();
        let model = Gemma4Model::from_gguf(&gguf).unwrap();
        let mut state = Gemma4State::new(&model, 32);
        let mut trainer = Gemma4Trainer::new(&model, 32, 4, CheckpointMode::Fp16);
        let mut tr_logits = vec![0.0f32; model.cfg.vocab_size];
        let prompt = [2u32, 105, 2364, 107, 651, 9456, 576, 886];
        model
            .forward_batch(&mut state, &pool, kt, &prompt, 0, true)
            .unwrap();
        trainer.forward_last_logits(&pool, None, &prompt, &mut tr_logits);
        for (i, (&a, &b)) in tr_logits.iter().zip(state.logits().iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Real Gemma-4-E2B 0-ULP mismatch at {i}"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn t3_end_to_end_finite_difference_jacobian_qwen35_gemma4_and_lfm2() {
    let dir = std::env::temp_dir().join(format!("tm_t3_fd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let p_qwen = dir.join("qwen35_f32.gguf");
    let p_gemma = dir.join("gemma4_f32.gguf");
    let p_lfm = dir.join("lfm2_f32.gguf");
    write_synthetic_qwen35_gguf(&p_qwen, DType::F32, 0xAAAA_0001);
    write_synthetic_gemma4_gguf(&p_gemma, DType::F32, 0xBBBB_0002);
    write_synthetic_lfm2_gguf(&p_lfm, DType::F32, 0xCCCC_0003);

    let pool = ThreadPool::new(2);
    let targets = LoraTargets::parse_csv("all").unwrap();
    // Sequence length 11 (> GDN_STATE_CKPT_STRIDE=8 AND > sliding_window=3!) with masked prompt tokens
    let tokens = [2u32, 7, 13, 4, 19, 8, 25, 11, 6, 17, 29, 3];
    let mask = [0u8, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1];
    let eps = 2e-3f32;

    // 1. Qwen3.5 GDN + FullAttn finite-difference Jacobian across ALL LoRA modules
    {
        let gguf = GgufFile::open(&p_qwen).unwrap();
        let model = Qwen35Model::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter_qwen35(&model, 2, 4.0, &targets, 301).unwrap();
        randomize_lora_b(&mut SplitMix64::new(302), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Qwen35Trainer::new(&model, 16, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        let mut max_rel = 0.0f64;
        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let a_indices: Vec<usize> = if is_emb {
                vec![7 * 2, 13 * 2 + 1, 25 * 2]
            } else {
                vec![0, 5, ad.pairs[&name].a.len() - 1]
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
                let denom = ana.abs().max(num.abs()).max(1.0);
                let rel = (ana - num).abs() / denom;
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 5e-4,
                    "Qwen3.5 dA mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
            for idx in [
                0usize,
                ad.pairs[&name].b.len() / 2,
                ad.pairs[&name].b.len() - 1,
            ] {
                let orig = ad.pairs[&name].b[idx];
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig + eps;
                let lp = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig - eps;
                let lm = trainer.forward_loss_f64(&pool, &ad, &tokens, &mask);
                ad.pairs.get_mut(&name).unwrap().b[idx] = orig;
                let num = (lp - lm) / (2.0 * eps as f64);
                let ana = grads.grads[&name].db[idx] as f64;
                let denom = ana.abs().max(num.abs()).max(1.0);
                let rel = (ana - num).abs() / denom;
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 5e-4,
                    "Qwen3.5 dB mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
        }
        eprintln!("[T3 Qwen3.5 FD] max_rel = {max_rel:.3e}");
    }

    // 2. Gemma 4 PLE + SWA + Shared-KV + KV-tying + Softcap finite-difference Jacobian
    {
        let gguf = GgufFile::open(&p_gemma).unwrap();
        let model = Gemma4Model::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter_gemma4(&model, 2, 4.0, &targets, 401).unwrap();
        randomize_lora_b(&mut SplitMix64::new(402), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Gemma4Trainer::new(&model, 16, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        let mut max_rel = 0.0f64;
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
                let denom = ana.abs().max(num.abs()).max(1.0);
                let rel = (ana - num).abs() / denom;
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 5e-4,
                    "Gemma 4 dA mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
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
                let denom = ana.abs().max(num.abs()).max(1.0);
                let rel = (ana - num).abs() / denom;
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 5e-4,
                    "Gemma 4 dB mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
        }
        eprintln!("[T3 Gemma 4 FD] max_rel = {max_rel:.3e}");
    }

    // 3. LFM2 ShortConv + SWA GQA finite-difference Jacobian
    {
        let gguf = GgufFile::open(&p_lfm).unwrap();
        let model = Lfm2Model::from_gguf(&gguf).unwrap();
        let mut ad = init_lora_adapter_lfm2(&model, 2, 4.0, &targets, 501).unwrap();
        randomize_lora_b(&mut SplitMix64::new(502), &mut ad);
        let mut grads = LoraGradStore::from_adapter(&ad);
        let mut trainer = Lfm2Trainer::new(&model, 16, 2, CheckpointMode::Fp32);
        trainer.exact_f32_kv_gradcheck = true;
        trainer.forward_backward_microbatch(None, &pool, &ad, &mut grads, &tokens, &mask, 1.0);

        let mut max_rel = 0.0f64;
        for name in ad.pair_order.clone() {
            let is_emb = ad.pairs[&name].is_token_embd;
            let a_indices: Vec<usize> = if is_emb {
                vec![7 * 2, 25 * 2 + 1]
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
                let denom = ana.abs().max(num.abs()).max(1.0);
                let rel = (ana - num).abs() / denom;
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 5e-4,
                    "LFM2 dA mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
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
                let denom = ana.abs().max(num.abs()).max(1.0);
                let rel = (ana - num).abs() / denom;
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 5e-4,
                    "LFM2 dB mismatch on {name}[{idx}]: ana={ana} num={num} rel={rel:e}"
                );
            }
        }
        eprintln!("[T3 LFM2 FD] max_rel = {max_rel:.3e}");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn t3_polymorphic_qlora_convergence_across_all_hybrid_archs() {
    let dir = std::env::temp_dir().join(format!("tm_t3_conv_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let p_qwen = dir.join("qwen35_q4.gguf");
    let p_gemma = dir.join("gemma4_q4.gguf");
    let p_lfm = dir.join("lfm2_q4.gguf");
    write_synthetic_qwen35_gguf(&p_qwen, DType::Q4_0, 0x7777);
    write_synthetic_gemma4_gguf(&p_gemma, DType::Q4_0, 0x8888);
    write_synthetic_lfm2_gguf(&p_lfm, DType::Q4_0, 0x9999);

    let ds = TrainDataset {
        samples: vec![
            TrainSample {
                tokens: vec![1, 4, 9, 12, 5, 18, 21, 7, 14, 3],
                loss_mask: vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            TrainSample {
                tokens: vec![2, 6, 11, 15, 8, 19, 23, 10, 16, 4],
                loss_mask: vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            },
        ],
        total_tokens: 20,
        supervised_tokens: 16,
    };

    let pool = ThreadPool::new(2);
    let mut cfg = QloraConfig {
        rank: 4,
        alpha: 8.0,
        steps: 24,
        seq_len: 16,
        grad_accum: 1,
        warmup_steps: 2,
        ..QloraConfig::default()
    };
    cfg.optim.lr = 8e-3;

    for (label, path) in [("qwen35", &p_qwen), ("gemma4", &p_gemma), ("lfm2", &p_lfm)] {
        let gguf = GgufFile::open(path).unwrap();
        let cpu_model = CpuModel::from_gguf(&gguf).unwrap();
        let (adapter, rep) =
            run_cpu_qlora_training(&cpu_model, &pool, &ds, &cfg, None, |_| Ok(())).unwrap();
        eprintln!(
            "[T3 {label} QLoRA] params={} loss {:.4} -> {:.4} (min {:.4})",
            rep.trainable_params, rep.initial_loss, rep.final_loss, rep.min_loss
        );
        assert!(
            rep.final_loss < rep.initial_loss * 0.55,
            "{label} QLoRA did not converge significantly: {:.4} -> {:.4}",
            rep.initial_loss,
            rep.final_loss
        );

        // Verify saved adapter validates cleanly against base GGUF and binds into `CpuModel`
        let out_lora = dir.join(format!("{label}_lora.gguf"));
        adapter.save_gguf(&out_lora, true).unwrap();
        let reloaded = tm_lora::LoraAdapter::open(&out_lora).unwrap();
        let mut set = LoraSet::new();
        set.push(reloaded, 1.0);
        let mut m2 = CpuModel::from_gguf(&gguf).unwrap();
        m2.bind_lora(&gguf, &set).unwrap();
    }

    let _ = std::fs::remove_dir_all(&dir);
}
