use std::fs;
use std::path::{Path, PathBuf};

use tm_cpu::{ThreadPool, active_kernels};
use tm_gguf::GgufFile;
use tm_models::ModelError;
use tm_runtime::{CpuModel, Session};

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_BF16: u32 = 30;

const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

struct MiniGgufBuilder {
    kvs: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl MiniGgufBuilder {
    fn new() -> Self {
        Self {
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn enc_str(s: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + s.len());
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        out
    }

    fn add_str(&mut self, k: &str, v: &str) {
        self.kvs
            .push((k.to_string(), GGUF_TYPE_STRING, Self::enc_str(v)));
    }

    fn add_u32(&mut self, k: &str, v: u32) {
        self.kvs
            .push((k.to_string(), GGUF_TYPE_UINT32, v.to_le_bytes().to_vec()));
    }

    fn add_f32(&mut self, k: &str, v: f32) {
        self.kvs
            .push((k.to_string(), GGUF_TYPE_FLOAT32, v.to_le_bytes().to_vec()));
    }

    fn add_u32_arr(&mut self, k: &str, vals: &[u32]) {
        let mut p = Vec::with_capacity(12 + vals.len() * 4);
        p.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        p.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            p.extend_from_slice(&v.to_le_bytes());
        }
        self.kvs.push((k.to_string(), GGUF_TYPE_ARRAY, p));
    }

    fn add_bool_arr(&mut self, k: &str, vals: &[bool]) {
        let mut p = Vec::with_capacity(12 + vals.len());
        p.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
        p.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            p.push(u8::from(v));
        }
        self.kvs.push((k.to_string(), GGUF_TYPE_ARRAY, p));
    }

    fn add_tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, raw: Vec<u8>) {
        self.tensors
            .push((name.to_string(), dims.to_vec(), ggml_type, raw));
    }

    fn write_to(&self, path: &Path) {
        let align = 32usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());

        for (k, vt, vb) in &self.kvs {
            buf.extend_from_slice(&Self::enc_str(k));
            buf.extend_from_slice(&vt.to_le_bytes());
            buf.extend_from_slice(vb);
        }

        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut cur = 0usize;
        for (_, _, _, raw) in &self.tensors {
            cur = cur.div_ceil(align) * align;
            offsets.push(cur as u64);
            cur += raw.len();
        }

        for ((name, dims, gtype, _), &off) in self.tensors.iter().zip(&offsets) {
            buf.extend_from_slice(&Self::enc_str(name));
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&gtype.to_le_bytes());
            buf.extend_from_slice(&off.to_le_bytes());
        }

        let pad = (align - (buf.len() % align)) % align;
        buf.resize(buf.len() + pad, 0);
        let data_base = buf.len();
        for ((_, _, _, raw), &off) in self.tensors.iter().zip(&offsets) {
            let cur_rel = buf.len() - data_base;
            if cur_rel < off as usize {
                buf.resize(buf.len() + (off as usize - cur_rel), 0);
            }
            buf.extend_from_slice(raw);
        }
        fs::write(path, buf).expect("write synthetic gguf");
    }
}

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let u = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    }
}

fn pack_f32_vec(rng: &mut Rng, len: usize, base: f32, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for _ in 0..len {
        let v = base + rng.next_f32(scale);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn pack_bf16_vec(rng: &mut Rng, len: usize, scale: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        let v = rng.next_f32(scale);
        let b = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&b.to_le_bytes());
    }
    out
}

fn pack_q4_0_mat(rng: &mut Rng, in_dim: usize, out_dim: usize, scale: f32) -> Vec<u8> {
    assert_eq!(in_dim % 32, 0);
    let n = in_dim * out_dim;
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        vals.push(rng.next_f32(scale));
    }
    let mut out = Vec::with_capacity((n / 32) * 18);
    for blk in vals.as_chunks::<32>().0 {
        let amax = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let d = if amax > 0.0 { amax / 7.0 } else { 0.0 };
        let d_f16 = tm_core::f32_to_f16(d);
        let d_rt = tm_core::f16_to_f32(d_f16);
        let inv = if d_rt != 0.0 { 1.0 / d_rt } else { 0.0 };
        out.extend_from_slice(&d_f16.to_le_bytes());
        for j in 0..16 {
            let q0 = ((blk[j] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let q1 = ((blk[j + 16] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
    }
    out
}

fn write_synthetic_gemma4_gguf(path: &Path) {
    let mut rng = Rng(0x6E44_A400_2026_1121);
    let n_vocab = 64usize;
    let n_embd = 64usize;
    let n_layers = 4usize; // [SWA(0), Full(1), SWA-shared(2->0), Full-shared(3->1)]
    let shared_kv_layers = 2usize;
    let n_head = 2usize;
    let n_head_kv = 1usize;
    let hd_swa = 32usize;
    let hd_full = 64usize;
    let ple_dim = 32usize;
    let ple_total = n_layers * ple_dim; // 128

    let mut b = MiniGgufBuilder::new();
    b.add_str("general.architecture", "gemma4");
    b.add_u32("gemma4.context_length", 128);
    b.add_u32("gemma4.embedding_length", n_embd as u32);
    b.add_u32("gemma4.embedding_length_per_layer_input", ple_dim as u32);
    b.add_u32("gemma4.block_count", n_layers as u32);
    b.add_u32("gemma4.attention.shared_kv_layers", shared_kv_layers as u32);
    b.add_u32_arr("gemma4.feed_forward_length", &[128, 128, 256, 256]);
    b.add_u32("gemma4.attention.head_count", n_head as u32);
    b.add_u32("gemma4.attention.head_count_kv", n_head_kv as u32);
    b.add_u32("gemma4.attention.key_length", hd_full as u32);
    b.add_u32("gemma4.attention.value_length", hd_full as u32);
    b.add_u32("gemma4.attention.key_length_swa", hd_swa as u32);
    b.add_u32("gemma4.attention.value_length_swa", hd_swa as u32);
    b.add_u32("gemma4.attention.sliding_window", 5);
    b.add_bool_arr(
        "gemma4.attention.sliding_window_pattern",
        &[true, false, true, false],
    );
    b.add_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6);
    b.add_u32("gemma4.rope.dimension_count", hd_full as u32);
    b.add_u32("gemma4.rope.dimension_count_swa", hd_swa as u32);
    b.add_f32("gemma4.rope.freq_base", 1_000_000.0);
    b.add_f32("gemma4.rope.freq_base_swa", 10_000.0);
    b.add_f32("gemma4.final_logit_softcapping", 30.0);

    b.add_tensor(
        "token_embd.weight",
        &[n_embd as u64, n_vocab as u64],
        GGML_TYPE_Q4_0,
        pack_q4_0_mat(&mut rng, n_embd, n_vocab, 0.15),
    );
    b.add_tensor(
        "output_norm.weight",
        &[n_embd as u64],
        GGML_TYPE_F32,
        pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
    );
    b.add_tensor(
        "rope_freqs.weight",
        &[(hd_full / 2) as u64],
        GGML_TYPE_F32,
        pack_f32_vec(&mut rng, hd_full / 2, 1.5, 0.25),
    );
    b.add_tensor(
        "per_layer_token_embd.weight",
        &[ple_total as u64, n_vocab as u64],
        GGML_TYPE_Q4_0,
        pack_q4_0_mat(&mut rng, ple_total, n_vocab, 0.15),
    );
    b.add_tensor(
        "per_layer_model_proj.weight",
        &[n_embd as u64, ple_total as u64],
        GGML_TYPE_BF16,
        pack_bf16_vec(&mut rng, n_embd * ple_total, 0.12),
    );
    b.add_tensor(
        "per_layer_proj_norm.weight",
        &[ple_dim as u64],
        GGML_TYPE_F32,
        pack_f32_vec(&mut rng, ple_dim, 1.0, 0.1),
    );

    let is_swa = [true, false, true, false];
    let n_ffs = [128usize, 128, 256, 256];
    for il in 0..n_layers {
        let hd = if is_swa[il] { hd_swa } else { hd_full };
        let q_dim = n_head * hd;
        let kv_dim = n_head_kv * hd;
        let n_ff = n_ffs[il];

        b.add_tensor(
            &format!("blk.{il}.attn_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.attn_q_norm.weight"),
            &[hd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, hd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.attn_q.weight"),
            &[n_embd as u64, q_dim as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_embd, q_dim, 0.12),
        );
        if il < n_layers - shared_kv_layers {
            b.add_tensor(
                &format!("blk.{il}.attn_k_norm.weight"),
                &[hd as u64],
                GGML_TYPE_F32,
                pack_f32_vec(&mut rng, hd, 1.0, 0.1),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_k.weight"),
                &[n_embd as u64, kv_dim as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, kv_dim, 0.12),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_v.weight"),
                &[n_embd as u64, kv_dim as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, kv_dim, 0.12),
            );
        }
        b.add_tensor(
            &format!("blk.{il}.attn_output.weight"),
            &[q_dim as u64, n_embd as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, q_dim, n_embd, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.post_attention_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_gate.weight"),
            &[n_embd as u64, n_ff as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_embd, n_ff, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_up.weight"),
            &[n_embd as u64, n_ff as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_embd, n_ff, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_down.weight"),
            &[n_ff as u64, n_embd as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_ff, n_embd, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.post_ffw_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.inp_gate.weight"),
            &[n_embd as u64, ple_dim as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_embd, ple_dim, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.proj.weight"),
            &[ple_dim as u64, n_embd as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, ple_dim, n_embd, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.post_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.layer_output_scale.weight"),
            &[1],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, 1, 0.9, 0.02),
        );
    }

    b.write_to(path);
}

fn write_synthetic_lfm2_gguf(path: &Path) {
    let mut rng = Rng(0x1F42_0000_2026_1121);
    let n_vocab = 64usize;
    let n_embd = 64usize;
    let n_ff = 128usize;
    let n_layers = 4usize;
    let n_head = 2usize;
    let hd = 32usize;
    let l_cache = 4usize;
    let head_count_kv = [0u32, 1, 0, 1];

    let mut b = MiniGgufBuilder::new();
    b.add_str("general.architecture", "lfm2");
    b.add_u32("lfm2.context_length", 128);
    b.add_u32("lfm2.embedding_length", n_embd as u32);
    b.add_u32("lfm2.feed_forward_length", n_ff as u32);
    b.add_u32("lfm2.block_count", n_layers as u32);
    b.add_u32("lfm2.attention.head_count", n_head as u32);
    b.add_u32_arr("lfm2.attention.head_count_kv", &head_count_kv);
    b.add_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5);
    b.add_u32("lfm2.attention.sliding_window", 5);
    b.add_u32("lfm2.rope.dimension_count", hd as u32);
    b.add_f32("lfm2.rope.freq_base", 1_000_000.0);
    b.add_u32("lfm2.shortconv.l_cache", l_cache as u32);

    b.add_tensor(
        "token_embd.weight",
        &[n_embd as u64, n_vocab as u64],
        GGML_TYPE_Q4_0,
        pack_q4_0_mat(&mut rng, n_embd, n_vocab, 0.15),
    );
    b.add_tensor(
        "token_embd_norm.weight",
        &[n_embd as u64],
        GGML_TYPE_F32,
        pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
    );

    for (il, &hkv) in head_count_kv.iter().enumerate() {
        b.add_tensor(
            &format!("blk.{il}.attn_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_norm.weight"),
            &[n_embd as u64],
            GGML_TYPE_F32,
            pack_f32_vec(&mut rng, n_embd, 1.0, 0.1),
        );
        if hkv == 0 {
            b.add_tensor(
                &format!("blk.{il}.shortconv.in_proj.weight"),
                &[n_embd as u64, (3 * n_embd) as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, 3 * n_embd, 0.12),
            );
            b.add_tensor(
                &format!("blk.{il}.shortconv.conv.weight"),
                &[l_cache as u64, n_embd as u64],
                GGML_TYPE_F32,
                pack_f32_vec(&mut rng, l_cache * n_embd, 0.0, 0.25),
            );
            b.add_tensor(
                &format!("blk.{il}.shortconv.out_proj.weight"),
                &[n_embd as u64, n_embd as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, n_embd, 0.12),
            );
        } else {
            let kv_dim = (hkv as usize) * hd;
            b.add_tensor(
                &format!("blk.{il}.attn_q_norm.weight"),
                &[hd as u64],
                GGML_TYPE_F32,
                pack_f32_vec(&mut rng, hd, 1.0, 0.1),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_k_norm.weight"),
                &[hd as u64],
                GGML_TYPE_F32,
                pack_f32_vec(&mut rng, hd, 1.0, 0.1),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_q.weight"),
                &[n_embd as u64, n_embd as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, n_embd, 0.12),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_k.weight"),
                &[n_embd as u64, kv_dim as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, kv_dim, 0.12),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_v.weight"),
                &[n_embd as u64, kv_dim as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, kv_dim, 0.12),
            );
            b.add_tensor(
                &format!("blk.{il}.attn_output.weight"),
                &[n_embd as u64, n_embd as u64],
                GGML_TYPE_Q4_0,
                pack_q4_0_mat(&mut rng, n_embd, n_embd, 0.12),
            );
        }
        b.add_tensor(
            &format!("blk.{il}.ffn_gate.weight"),
            &[n_embd as u64, n_ff as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_embd, n_ff, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_up.weight"),
            &[n_embd as u64, n_ff as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_embd, n_ff, 0.12),
        );
        b.add_tensor(
            &format!("blk.{il}.ffn_down.weight"),
            &[n_ff as u64, n_embd as u64],
            GGML_TYPE_Q4_0,
            pack_q4_0_mat(&mut rng, n_ff, n_embd, 0.12),
        );
    }

    b.write_to(path);
}

fn assert_bit_exact_logits(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: len mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: non-bit-exact logit at idx={i}: {x} ({:#010x}) vs {y} ({:#010x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

fn assert_batch_logits_parity(
    a: &[f32],
    b: &[f32],
    _isa: tm_cpu::Isa,
    _large_batch: bool,
    label: &str,
) {
    assert_bit_exact_logits(a, b, label);
}

fn temp_gguf_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tm_m2_test_{}_{}.gguf", std::process::id(), name))
}

#[test]
fn gemma4_batch_invariance_state_atomicity_and_prefix_cache() {
    let path = temp_gguf_path("gemma4");
    write_synthetic_gemma4_gguf(&path);
    let gguf = GgufFile::open(&path).expect("open synthetic gemma4");
    let model = CpuModel::from_gguf(&gguf).expect("load gemma4");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    // 1. Batch-width invariance (`AGENTS.md` §9): sequential `forward_token` vs `forward_batch`
    let seq: Vec<u32> = (0..18).map(|i| ((i * 7 + 3) % 64) as u32).collect();
    let mut st_seq = model.new_state(32);
    for (pos, &tid) in seq.iter().enumerate() {
        model
            .forward_token(&mut st_seq, &pool, kt, tid, pos, pos + 1 == seq.len())
            .unwrap();
    }
    let mut st_bat = model.new_state(32);
    model
        .forward_batch(&mut st_bat, &pool, kt, &seq, 0, true)
        .unwrap();
    assert_bit_exact_logits(st_seq.logits(), st_bat.logits(), "gemma4 batch vs seq");

    // Next decode token after batch vs seq must ALSO match bit-for-bit
    model
        .forward_token(&mut st_seq, &pool, kt, 19, seq.len(), true)
        .unwrap();
    model
        .forward_token(&mut st_bat, &pool, kt, 19, seq.len(), true)
        .unwrap();
    assert_bit_exact_logits(
        st_seq.logits(),
        st_bat.logits(),
        "gemma4 post-batch decode step",
    );

    // 2. State atomicity (`AGENTS.md` Rule #6): errors leave state untouched
    let snap_logits: Vec<u32> = st_bat.logits().iter().map(|x| x.to_bits()).collect();
    assert!(matches!(
        model.forward_token(&mut st_bat, &pool, kt, 999, 19, true),
        Err(ModelError::TokenOutOfRange { .. })
    ));
    assert!(matches!(
        model.forward_token(&mut st_bat, &pool, kt, 1, 32, true),
        Err(ModelError::ContextOverflow { .. })
    ));
    assert!(matches!(
        model.forward_batch(&mut st_bat, &pool, kt, &[1, 2, 999], 19, true),
        Err(ModelError::TokenOutOfRange { .. })
    ));
    let after_logits: Vec<u32> = st_bat.logits().iter().map(|x| x.to_bits()).collect();
    assert_eq!(snap_logits, after_logits, "gemma4 state mutated on error");

    // 3. `Session::prefill_with_cache` exact repeat, prefix truncation, & multi-turn extension
    let mut session = Session::new(&model, 64, 4 * 1024 * 1024);
    let turn1: Vec<u32> = vec![2, 5, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53];
    let rep1 = session.prefill_with_cache(&pool, &turn1, None).unwrap();
    assert_eq!(rep1.cached_tokens, 0);
    let mut uncached = model.new_state(64);
    model
        .forward_batch(&mut uncached, &pool, kt, &turn1, 0, true)
        .unwrap();
    assert_bit_exact_logits(session.logits(), uncached.logits(), "gemma4 cache turn1");

    // Exact repeat: reuses 11/12 tokens
    let rep_repeat = session.prefill_with_cache(&pool, &turn1, None).unwrap();
    assert_eq!(rep_repeat.cached_tokens, 11);
    assert_eq!(rep_repeat.evaluated_tokens, 1);
    assert_bit_exact_logits(
        session.logits(),
        uncached.logits(),
        "gemma4 cache exact repeat",
    );

    // Partial shared prefix (first 7 tokens match `turn1`, then diverges!)
    let branch: Vec<u32> = vec![2, 5, 11, 17, 23, 29, 31, 9, 14, 18];
    let rep_branch = session.prefill_with_cache(&pool, &branch, None).unwrap();
    assert_eq!(rep_branch.cached_tokens, 7);
    uncached.reset();
    model
        .forward_batch(&mut uncached, &pool, kt, &branch, 0, true)
        .unwrap();
    assert_bit_exact_logits(
        session.logits(),
        uncached.logits(),
        "gemma4 cache partial prefix truncation",
    );

    // Multi-turn extension (`turn1 ++ extra`, crossing SWA window = 5!)
    let mut turn2 = turn1.clone();
    turn2.extend_from_slice(&[6, 12, 24, 30, 36, 42]);
    let rep_ext = session.prefill_with_cache(&pool, &turn2, None).unwrap();
    assert_eq!(rep_ext.cached_tokens, 11);
    uncached.reset();
    model
        .forward_batch(&mut uncached, &pool, kt, &turn2, 0, true)
        .unwrap();
    assert_bit_exact_logits(
        session.logits(),
        uncached.logits(),
        "gemma4 cache multi-turn extension",
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn lfm2_batch_invariance_state_atomicity_and_recurrent_prefix_cache() {
    let path = temp_gguf_path("lfm2");
    write_synthetic_lfm2_gguf(&path);
    let gguf = GgufFile::open(&path).expect("open synthetic lfm2");
    let model = CpuModel::from_gguf(&gguf).expect("load lfm2");
    let pool = ThreadPool::new(2);
    let kt = active_kernels();

    // 1a. Speculative-verify batch width (`7 < 8`): MUST be 100% bit-exact (`to_bits()`) on ALL ISAs including `i8mm`!
    let spec7: Vec<u32> = vec![3, 11, 19, 27, 35, 43, 51];
    let mut st_s7 = model.new_state(48);
    for (pos, &tid) in spec7.iter().enumerate() {
        model
            .forward_token(&mut st_s7, &pool, kt, tid, pos, pos + 1 == spec7.len())
            .unwrap();
    }
    let mut st_b7 = model.new_state(48);
    model
        .forward_batch(&mut st_b7, &pool, kt, &spec7, 0, true)
        .unwrap();
    assert_bit_exact_logits(st_s7.logits(), st_b7.logits(), "lfm2 spec-7 bit-exact");

    // 1b. Large prefill (`20 >= 8`) across ShortConv (`l_cache=4`) + GQA (`sliding_window=5`)
    let seq: Vec<u32> = (0..20).map(|i| ((i * 5 + 1) % 64) as u32).collect();
    let mut st_seq = model.new_state(48);
    for (pos, &tid) in seq.iter().enumerate() {
        model
            .forward_token(&mut st_seq, &pool, kt, tid, pos, pos + 1 == seq.len())
            .unwrap();
    }
    let mut st_bat = model.new_state(48);
    model
        .forward_batch(&mut st_bat, &pool, kt, &seq, 0, true)
        .unwrap();
    assert_batch_logits_parity(
        st_seq.logits(),
        st_bat.logits(),
        kt.isa,
        true,
        "lfm2 batch vs seq",
    );

    model
        .forward_token(&mut st_seq, &pool, kt, 7, seq.len(), true)
        .unwrap();
    model
        .forward_token(&mut st_bat, &pool, kt, 7, seq.len(), true)
        .unwrap();
    assert_batch_logits_parity(
        st_seq.logits(),
        st_bat.logits(),
        kt.isa,
        true,
        "lfm2 post-batch decode step",
    );

    // 2. State atomicity (`Rule #6`)
    let snap_logits: Vec<u32> = st_bat.logits().iter().map(|x| x.to_bits()).collect();
    assert!(matches!(
        model.forward_token(&mut st_bat, &pool, kt, 500, 21, true),
        Err(ModelError::TokenOutOfRange { .. })
    ));
    assert!(matches!(
        model.forward_batch(&mut st_bat, &pool, kt, &[1; 40], 20, true),
        Err(ModelError::ContextOverflow { .. })
    ));
    let after_logits: Vec<u32> = st_bat.logits().iter().map(|x| x.to_bits()).collect();
    assert_eq!(snap_logits, after_logits, "lfm2 state mutated on error");

    // 3. Recurrent `PrefixCache` with periodic `checkpoint_every = Some(4)` + LRU budget check
    let mut session = Session::new(&model, 64, 64 * 1024);
    let turn1: Vec<u32> = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
    let rep1 = session.prefill_with_cache(&pool, &turn1, Some(4)).unwrap();
    assert_eq!(rep1.cached_tokens, 0);
    assert!(session.cache.used_bytes() <= session.cache.max_bytes());

    let mut uncached = model.new_state(64);
    model
        .forward_batch(&mut uncached, &pool, kt, &turn1, 0, true)
        .unwrap();
    assert_batch_logits_parity(
        session.logits(),
        uncached.logits(),
        kt.isa,
        true,
        "lfm2 cache turn1",
    );

    // Exact repeat: MUST be 100% bit-exact (`to_bits()`) to `session` turn1 on ALL ISAs!
    let turn1_snap = session.logits().to_vec();
    let rep_repeat = session.prefill_with_cache(&pool, &turn1, Some(4)).unwrap();
    assert_eq!(rep_repeat.cached_tokens, 11);
    assert_eq!(rep_repeat.evaluated_tokens, 1);
    assert_bit_exact_logits(session.logits(), &turn1_snap, "lfm2 cache exact repeat");

    // Diverging after 9 tokens: recurrent checkpoint at 8 (`checkpoint_every = 4`) must be reused!
    let branch: Vec<u32> = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 40, 41, 42];
    let rep_branch = session.prefill_with_cache(&pool, &branch, Some(4)).unwrap();
    assert_eq!(rep_branch.cached_tokens, 8);
    uncached.reset();
    model
        .forward_batch(&mut uncached, &pool, kt, &branch, 0, true)
        .unwrap();
    assert_batch_logits_parity(
        session.logits(),
        uncached.logits(),
        kt.isa,
        true,
        "lfm2 recurrent checkpoint_every=4 branch",
    );

    // Multi-turn extension from `branch`
    let mut turn2 = branch.clone();
    turn2.extend_from_slice(&[44, 45, 46, 47, 48]);
    let rep_ext = session.prefill_with_cache(&pool, &turn2, Some(4)).unwrap();
    assert_eq!(rep_ext.cached_tokens, 11);
    uncached.reset();
    model
        .forward_batch(&mut uncached, &pool, kt, &turn2, 0, true)
        .unwrap();
    assert_batch_logits_parity(
        session.logits(),
        uncached.logits(),
        kt.isa,
        true,
        "lfm2 recurrent multi-turn extension",
    );

    // LRU memory ceiling check under tight 8 KiB cap
    let mut tight = Session::new(&model, 64, 8 * 1024);
    for seed in 0..10u32 {
        let p: Vec<u32> = (0..10).map(|i| (seed * 3 + i) % 64).collect();
        let _ = tight.prefill_with_cache(&pool, &p, None).unwrap();
        assert!(
            tight.cache.used_bytes() <= tight.cache.max_bytes(),
            "prefix cache exceeded max_bytes: {} > {}",
            tight.cache.used_bytes(),
            tight.cache.max_bytes()
        );
    }

    let _ = fs::remove_file(&path);
}

#[test]
fn qwen35_real_gdn_recurrent_prefix_cache_if_present() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let qwen_path = repo_root.join("models/Qwen3.5-0.8B-Q4_0.gguf");
    if !qwen_path.exists() {
        return;
    }
    let gguf = GgufFile::open(&qwen_path).unwrap();
    let model = CpuModel::from_gguf(&gguf).unwrap();
    let pool = ThreadPool::new(4);
    let kt = active_kernels();

    let mut session = Session::new(&model, 128, 64 * 1024 * 1024);
    let turn1 = vec![760u32, 6722, 315, 9625, 374, 12095];
    session.prefill_with_cache(&pool, &turn1, None).unwrap();

    let mut turn2 = turn1.clone();
    turn2.extend_from_slice(&[13, 576, 6722, 315]);
    let rep = session.prefill_with_cache(&pool, &turn2, None).unwrap();
    assert_eq!(rep.cached_tokens, turn1.len() - 1);

    let mut uncached = model.new_state(128);
    model
        .forward_batch(&mut uncached, &pool, kt, &turn2, 0, true)
        .unwrap();
    assert_bit_exact_logits(
        session.logits(),
        uncached.logits(),
        "qwen35 real GDN multi-turn prefix cache",
    );
}
