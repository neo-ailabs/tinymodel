//! Deterministic LoRA initialization, gradient storage, AdamW / LoRA+ optimizer,
//! cosine warmup LR schedule, and crash-safe atomic checkpointing (`ADR-015`).

use std::collections::HashMap;
use std::path::Path;

use tm_core::{DType, dequantize_row_bytes};
use tm_gguf::{GgufFile, MetadataValue, NewGgufTensor, write_gguf_file_atomic};
use tm_lora::{LoraAdapter, LoraWeightPair};
use tm_models::{
    Gemma4Model, Lfm2Model, Lfm2OpWeights, LlamaModel, Mamba2AttnWeights, Mamba2Block,
    Mamba2FfnWeights, Mamba2Model, Mamba2SsmWeights, Qwen35AttnWeights, Qwen35Model,
};
use tm_runtime::CpuModel;

use crate::TrainError;

/// Deterministic platform-independent `SplitMix64` PRNG (zero external dependencies).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[-bound, +bound)`.
    #[inline]
    pub fn uniform_sym(&mut self, bound: f32) -> f32 {
        let u01 = ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        ((2.0 * u01 - 1.0) as f32) * bound
    }
}

/// Configurable projection targets for LoRA injection (`--targets q,k,v,o,gate,up,down[,ssm,ple,embed,lm_head]`).
#[derive(Debug, Clone)]
pub struct LoraTargets {
    pub q: bool,
    pub k: bool,
    pub v: bool,
    pub o: bool,
    pub gate: bool,
    pub up: bool,
    pub down: bool,
    /// Qwen3.5 GDN `attn_gate` (`wz`), `ssm_alpha`, and `ssm_beta` projections (enabled by default).
    pub ssm_gates: bool,
    /// Gemma 4 Per-Layer Embedding (`per_layer_model_proj`, `inp_gate`, `proj`) projections.
    pub ple: bool,
    pub embed: bool,
    pub lm_head: bool,
}

impl Default for LoraTargets {
    fn default() -> Self {
        Self {
            q: true,
            k: true,
            v: true,
            o: true,
            gate: true,
            up: true,
            down: true,
            ssm_gates: true,
            ple: false,
            embed: false,
            lm_head: false,
        }
    }
}

impl LoraTargets {
    pub fn parse_csv(spec: &str) -> Result<Self, TrainError> {
        if spec.trim().eq_ignore_ascii_case("all") {
            return Ok(Self {
                q: true,
                k: true,
                v: true,
                o: true,
                gate: true,
                up: true,
                down: true,
                ssm_gates: true,
                ple: true,
                embed: true,
                lm_head: true,
            });
        }
        let mut t = Self {
            q: false,
            k: false,
            v: false,
            o: false,
            gate: false,
            up: false,
            down: false,
            ssm_gates: false,
            ple: false,
            embed: false,
            lm_head: false,
        };
        let mut any = false;
        for part in spec.split(',') {
            let p = part.trim().to_ascii_lowercase();
            if p.is_empty() {
                continue;
            }
            any = true;
            match p.as_str() {
                "q" | "attn_q" => t.q = true,
                "k" | "attn_k" => t.k = true,
                "v" | "attn_v" => t.v = true,
                "qkv" | "attn_qkv" | "shortconv" => {
                    t.q = true;
                    t.k = true;
                    t.v = true;
                }
                "o" | "wo" | "attn_output" | "ssm_out" | "out_proj" => t.o = true,
                "gate" | "ffn_gate" => t.gate = true,
                "up" | "ffn_up" => t.up = true,
                "down" | "ffn_down" => t.down = true,
                "ssm" | "ssm_gates" | "attn_gate" | "z" => t.ssm_gates = true,
                "ple" => t.ple = true,
                "embed" | "token_embd" => t.embed = true,
                "lm_head" | "output" => t.lm_head = true,
                other => {
                    return Err(TrainError::InvalidConfig(format!(
                        "unknown --targets module '{other}' (expected q,k,v,o,gate,up,down,ssm,ple,embed,lm_head or 'all')"
                    )));
                }
            }
        }
        if !any {
            return Err(TrainError::InvalidConfig(
                "--targets must select at least one module".to_string(),
            ));
        }
        Ok(t)
    }
}

fn push_embed_pair(
    rng: &mut SplitMix64,
    pairs: &mut HashMap<String, LoraWeightPair>,
    order: &mut Vec<String>,
    vocab: usize,
    d_out: usize,
    rank: usize,
) {
    let bound = 1.0f32 / (rank as f32).sqrt();
    let a: Vec<f32> = (0..vocab * rank).map(|_| rng.uniform_sym(bound)).collect();
    let b = vec![0.0f32; d_out * rank];
    let name = "token_embd.weight".to_string();
    order.push(name.clone());
    pairs.insert(
        name.clone(),
        LoraWeightPair {
            base_name: name,
            is_token_embd: true,
            rank,
            d_in: vocab,
            d_out,
            a,
            b,
        },
    );
}

fn push_linear_pair(
    rng: &mut SplitMix64,
    pairs: &mut HashMap<String, LoraWeightPair>,
    order: &mut Vec<String>,
    base_name: &str,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    let bound = 1.0f32 / (d_in as f32).sqrt();
    let a: Vec<f32> = (0..rank * d_in).map(|_| rng.uniform_sym(bound)).collect();
    let b = vec![0.0f32; d_out * rank];
    order.push(base_name.to_string());
    pairs.insert(
        base_name.to_string(),
        LoraWeightPair {
            base_name: base_name.to_string(),
            is_token_embd: false,
            rank,
            d_in,
            d_out,
            a,
            b,
        },
    );
}

/// Initializes a new [`LoraAdapter`] for any supported [`CpuModel`] (`llama`, `qwen3`, `qwen35`,
/// `gemma4`, `lfm2`) using Kaiming-uniform `A` and exact zero `B`.
pub fn init_lora_adapter_for_cpu_model(
    model: &CpuModel<'_>,
    rank: usize,
    alpha: f32,
    targets: &LoraTargets,
    seed: u64,
) -> Result<LoraAdapter, TrainError> {
    match model {
        CpuModel::Llama(m) => init_lora_adapter(m, rank, alpha, targets, seed),
        CpuModel::Qwen35(m) => init_lora_adapter_qwen35(m, rank, alpha, targets, seed),
        CpuModel::Gemma4(m) => init_lora_adapter_gemma4(m, rank, alpha, targets, seed),
        CpuModel::Lfm2(m) => init_lora_adapter_lfm2(m, rank, alpha, targets, seed),
        CpuModel::Mamba2(m) => init_lora_adapter_mamba2(m, rank, alpha, targets, seed),
        CpuModel::Inkling2(m) => Err(TrainError::InvalidConfig(format!(
            "QLoRA fine-tuning is not yet supported on architecture '{}'",
            m.cfg.arch
        ))),
    }
}

/// Initializes a [`LoraAdapter`] for hybrid `Qwen35Model` (GDN + FullAttn + SwiGLU).
pub fn init_lora_adapter_qwen35(
    model: &Qwen35Model<'_>,
    rank: usize,
    alpha: f32,
    targets: &LoraTargets,
    seed: u64,
) -> Result<LoraAdapter, TrainError> {
    if rank == 0 {
        return Err(TrainError::InvalidConfig(
            "LoRA rank must be >= 1".to_string(),
        ));
    }
    let mut rng = SplitMix64::new(seed);
    let mut pairs = HashMap::new();
    let mut order = Vec::new();

    if targets.embed {
        push_embed_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.cfg.vocab_size,
            model.cfg.n_embd,
            rank,
        );
    }

    for lw in &model.layers {
        match &lw.attn {
            Qwen35AttnWeights::Gdn {
                wqkv,
                wz,
                w_alpha,
                w_beta,
                w_out,
                ..
            } => {
                if targets.q || targets.k || targets.v {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wqkv.name, wqkv.cols, wqkv.rows, rank,
                    );
                }
                if targets.ssm_gates {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wz.name, wz.cols, wz.rows, rank,
                    );
                    let r_small = rank.min(w_alpha.rows).max(1);
                    push_linear_pair(
                        &mut rng,
                        &mut pairs,
                        &mut order,
                        w_alpha.name,
                        w_alpha.cols,
                        w_alpha.rows,
                        r_small,
                    );
                    push_linear_pair(
                        &mut rng,
                        &mut pairs,
                        &mut order,
                        w_beta.name,
                        w_beta.cols,
                        w_beta.rows,
                        r_small,
                    );
                }
                if targets.o {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, w_out.name, w_out.cols, w_out.rows, rank,
                    );
                }
            }
            Qwen35AttnWeights::FullAttn {
                wqg, wk, wv, wo, ..
            } => {
                if targets.q {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wqg.name, wqg.cols, wqg.rows, rank,
                    );
                }
                if targets.k {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wk.name, wk.cols, wk.rows, rank,
                    );
                }
                if targets.v {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wv.name, wv.cols, wv.rows, rank,
                    );
                }
                if targets.o {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wo.name, wo.cols, wo.rows, rank,
                    );
                }
            }
        }
        match &lw.ffn {
            tm_models::Qwen35FfnWeights::Dense {
                w_gate,
                w_up,
                w_down,
            } => {
                if targets.gate {
                    push_linear_pair(
                        &mut rng,
                        &mut pairs,
                        &mut order,
                        w_gate.name,
                        w_gate.cols,
                        w_gate.rows,
                        rank,
                    );
                }
                if targets.up {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, w_up.name, w_up.cols, w_up.rows, rank,
                    );
                }
                if targets.down {
                    push_linear_pair(
                        &mut rng,
                        &mut pairs,
                        &mut order,
                        w_down.name,
                        w_down.cols,
                        w_down.rows,
                        rank,
                    );
                }
            }
            tm_models::Qwen35FfnWeights::Moe(moe) => {
                if let Some(ref sh) = moe.shared_expert {
                    if targets.gate {
                        push_linear_pair(
                            &mut rng,
                            &mut pairs,
                            &mut order,
                            sh.w_gate.name,
                            sh.w_gate.cols,
                            sh.w_gate.rows,
                            rank,
                        );
                    }
                    if targets.up {
                        push_linear_pair(
                            &mut rng,
                            &mut pairs,
                            &mut order,
                            sh.w_up.name,
                            sh.w_up.cols,
                            sh.w_up.rows,
                            rank,
                        );
                    }
                    if targets.down {
                        push_linear_pair(
                            &mut rng,
                            &mut pairs,
                            &mut order,
                            sh.w_down.name,
                            sh.w_down.cols,
                            sh.w_down.rows,
                            rank,
                        );
                    }
                }
            }
        }
    }

    if targets.lm_head {
        push_linear_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.output.name,
            model.output.cols,
            model.output.rows,
            rank,
        );
    }

    Ok(LoraAdapter {
        arch: model.cfg.arch.clone(),
        alpha,
        name: Some("tinymodel-qlora".to_string()),
        pairs,
        pair_order: order,
    })
}

/// Initializes a [`LoraAdapter`] for `Mamba2Model` (`mamba2`, `nemotron_h`, `falcon-h1`, `granitehybrid`).
pub fn init_lora_adapter_mamba2(
    model: &Mamba2Model<'_>,
    rank: usize,
    alpha: f32,
    targets: &LoraTargets,
    seed: u64,
) -> Result<LoraAdapter, TrainError> {
    if rank == 0 {
        return Err(TrainError::InvalidConfig(
            "LoRA rank must be >= 1".to_string(),
        ));
    }
    let mut rng = SplitMix64::new(seed);
    let mut pairs = HashMap::new();
    let mut order = Vec::new();

    if targets.embed {
        push_embed_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.cfg.vocab_size,
            model.cfg.n_embd,
            rank,
        );
    }

    let add_ssm = |ssm: &Mamba2SsmWeights<'_>,
                   rng: &mut SplitMix64,
                   pairs: &mut HashMap<String, LoraWeightPair>,
                   order: &mut Vec<String>| {
        if targets.q || targets.k || targets.v || targets.ssm_gates {
            push_linear_pair(
                rng,
                pairs,
                order,
                ssm.ssm_in.name,
                ssm.ssm_in.cols,
                ssm.ssm_in.rows,
                rank,
            );
        }
        if targets.o {
            push_linear_pair(
                rng,
                pairs,
                order,
                ssm.ssm_out.name,
                ssm.ssm_out.cols,
                ssm.ssm_out.rows,
                rank,
            );
        }
    };
    let add_attn = |attn: &Mamba2AttnWeights<'_>,
                    rng: &mut SplitMix64,
                    pairs: &mut HashMap<String, LoraWeightPair>,
                    order: &mut Vec<String>| {
        if attn.wq.is_sliced {
            if targets.q || targets.k || targets.v {
                let d_out = attn.wq.rows + attn.wk.rows + attn.wv.rows;
                push_linear_pair(rng, pairs, order, attn.wq.name, attn.wq.cols, d_out, rank);
            }
        } else {
            if targets.q {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    attn.wq.name,
                    attn.wq.cols,
                    attn.wq.rows,
                    rank,
                );
            }
            if targets.k {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    attn.wk.name,
                    attn.wk.cols,
                    attn.wk.rows,
                    rank,
                );
            }
            if targets.v {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    attn.wv.name,
                    attn.wv.cols,
                    attn.wv.rows,
                    rank,
                );
            }
        }
        if targets.o {
            push_linear_pair(
                rng,
                pairs,
                order,
                attn.wo.name,
                attn.wo.cols,
                attn.wo.rows,
                rank,
            );
        }
    };
    let add_ffn = |ffn: &Mamba2FfnWeights<'_>,
                   rng: &mut SplitMix64,
                   pairs: &mut HashMap<String, LoraWeightPair>,
                   order: &mut Vec<String>| match ffn {
        Mamba2FfnWeights::SwiGlu {
            w_gate,
            w_up,
            w_down,
            ..
        } => {
            if targets.gate {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    w_gate.name,
                    w_gate.cols,
                    w_gate.rows,
                    rank,
                );
            }
            if targets.up {
                push_linear_pair(rng, pairs, order, w_up.name, w_up.cols, w_up.rows, rank);
            }
            if targets.down {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    w_down.name,
                    w_down.cols,
                    w_down.rows,
                    rank,
                );
            }
        }
        Mamba2FfnWeights::Relu2 { w_up, w_down, .. } => {
            if targets.up || targets.gate {
                push_linear_pair(rng, pairs, order, w_up.name, w_up.cols, w_up.rows, rank);
            }
            if targets.down {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    w_down.name,
                    w_down.cols,
                    w_down.rows,
                    rank,
                );
            }
        }
        Mamba2FfnWeights::SparseMoe(moe) => {
            if targets.gate {
                push_linear_pair(
                    rng,
                    pairs,
                    order,
                    moe.gate_inp.name,
                    moe.gate_inp.cols,
                    moe.gate_inp.rows,
                    rank,
                );
            }
            if let Some(ref sh) = moe.shared_expert {
                if targets.gate {
                    push_linear_pair(
                        rng,
                        pairs,
                        order,
                        sh.w_gate.name,
                        sh.w_gate.cols,
                        sh.w_gate.rows,
                        rank,
                    );
                    if let Some(ref gi) = sh.w_gate_inp_shexp {
                        push_linear_pair(rng, pairs, order, gi.name, gi.cols, gi.rows, rank);
                    }
                }
                if targets.up {
                    push_linear_pair(
                        rng,
                        pairs,
                        order,
                        sh.w_up.name,
                        sh.w_up.cols,
                        sh.w_up.rows,
                        rank,
                    );
                }
                if targets.down {
                    push_linear_pair(
                        rng,
                        pairs,
                        order,
                        sh.w_down.name,
                        sh.w_down.cols,
                        sh.w_down.rows,
                        rank,
                    );
                }
            }
        }
    };

    for lw in &model.layers {
        match &lw.block {
            Mamba2Block::SsmOnly { ssm } => add_ssm(ssm, &mut rng, &mut pairs, &mut order),
            Mamba2Block::AttnOnly { attn } => add_attn(attn, &mut rng, &mut pairs, &mut order),
            Mamba2Block::FfnOnly { ffn } => add_ffn(ffn, &mut rng, &mut pairs, &mut order),
            Mamba2Block::SsmAndFfn { ssm, ffn, .. } => {
                add_ssm(ssm, &mut rng, &mut pairs, &mut order);
                add_ffn(ffn, &mut rng, &mut pairs, &mut order);
            }
            Mamba2Block::AttnAndFfn { attn, ffn, .. } => {
                add_attn(attn, &mut rng, &mut pairs, &mut order);
                add_ffn(ffn, &mut rng, &mut pairs, &mut order);
            }
            Mamba2Block::ParallelAttnSsmAndFfn { attn, ssm, ffn, .. } => {
                add_attn(attn, &mut rng, &mut pairs, &mut order);
                add_ssm(ssm, &mut rng, &mut pairs, &mut order);
                add_ffn(ffn, &mut rng, &mut pairs, &mut order);
            }
        }
    }

    if targets.lm_head {
        push_linear_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.output.name,
            model.output.cols,
            model.output.rows,
            rank,
        );
    }

    Ok(LoraAdapter {
        arch: model.cfg.arch.clone(),
        alpha,
        name: Some("tinymodel-qlora".to_string()),
        pairs,
        pair_order: order,
    })
}

/// Initializes a [`LoraAdapter`] for `Gemma4Model` (SWA + KV-sharing + GeGLU + PLE).
pub fn init_lora_adapter_gemma4(
    model: &Gemma4Model<'_>,
    rank: usize,
    alpha: f32,
    targets: &LoraTargets,
    seed: u64,
) -> Result<LoraAdapter, TrainError> {
    if rank == 0 {
        return Err(TrainError::InvalidConfig(
            "LoRA rank must be >= 1".to_string(),
        ));
    }
    let mut rng = SplitMix64::new(seed);
    let mut pairs = HashMap::new();
    let mut order = Vec::new();

    if targets.embed {
        push_embed_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.cfg.vocab_size,
            model.cfg.n_embd,
            rank,
        );
    }

    if targets.ple
        && let Some(ref pg) = model.ple_global
    {
        push_linear_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            pg.model_proj.name,
            pg.model_proj.cols,
            pg.model_proj.rows,
            rank,
        );
    }

    for lw in &model.layers {
        if targets.q {
            push_linear_pair(
                &mut rng, &mut pairs, &mut order, lw.wq.name, lw.wq.cols, lw.wq.rows, rank,
            );
        }
        if let Some(ref kv) = lw.kv {
            if targets.k {
                push_linear_pair(
                    &mut rng, &mut pairs, &mut order, kv.wk.name, kv.wk.cols, kv.wk.rows, rank,
                );
            }
            if targets.v
                && let Some(ref wv) = kv.wv
            {
                push_linear_pair(
                    &mut rng, &mut pairs, &mut order, wv.name, wv.cols, wv.rows, rank,
                );
            }
        }
        if targets.o {
            push_linear_pair(
                &mut rng, &mut pairs, &mut order, lw.wo.name, lw.wo.cols, lw.wo.rows, rank,
            );
        }
        if targets.gate {
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                lw.w_gate.name,
                lw.w_gate.cols,
                lw.w_gate.rows,
                rank,
            );
        }
        if targets.up {
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                lw.w_up.name,
                lw.w_up.cols,
                lw.w_up.rows,
                rank,
            );
        }
        if targets.down {
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                lw.w_down.name,
                lw.w_down.cols,
                lw.w_down.rows,
                rank,
            );
        }
        if targets.ple
            && let Some(ref ple) = lw.ple
        {
            let r_ple = rank.min(ple.inp_gate.rows).max(1);
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                ple.inp_gate.name,
                ple.inp_gate.cols,
                ple.inp_gate.rows,
                r_ple,
            );
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                ple.proj.name,
                ple.proj.cols,
                ple.proj.rows,
                r_ple,
            );
        }
    }

    if targets.lm_head {
        push_linear_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.output.name,
            model.output.cols,
            model.output.rows,
            rank,
        );
    }

    Ok(LoraAdapter {
        arch: model.cfg.arch.clone(),
        alpha,
        name: Some("tinymodel-qlora".to_string()),
        pairs,
        pair_order: order,
    })
}

/// Initializes a [`LoraAdapter`] for `Lfm2Model` (`ShortConv` + `GQA` + SwiGLU).
pub fn init_lora_adapter_lfm2(
    model: &Lfm2Model<'_>,
    rank: usize,
    alpha: f32,
    targets: &LoraTargets,
    seed: u64,
) -> Result<LoraAdapter, TrainError> {
    if rank == 0 {
        return Err(TrainError::InvalidConfig(
            "LoRA rank must be >= 1".to_string(),
        ));
    }
    let mut rng = SplitMix64::new(seed);
    let mut pairs = HashMap::new();
    let mut order = Vec::new();

    if targets.embed {
        push_embed_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.cfg.vocab_size,
            model.cfg.n_embd,
            rank,
        );
    }

    for lw in &model.layers {
        match &lw.op {
            Lfm2OpWeights::ShortConv {
                in_proj, out_proj, ..
            } => {
                if targets.q || targets.k || targets.v {
                    push_linear_pair(
                        &mut rng,
                        &mut pairs,
                        &mut order,
                        in_proj.name,
                        in_proj.cols,
                        in_proj.rows,
                        rank,
                    );
                }
                if targets.o {
                    push_linear_pair(
                        &mut rng,
                        &mut pairs,
                        &mut order,
                        out_proj.name,
                        out_proj.cols,
                        out_proj.rows,
                        rank,
                    );
                }
            }
            Lfm2OpWeights::Gqa {
                wq,
                wk,
                wv,
                wo,
                w_attn_gate,
                ..
            } => {
                if targets.q {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wq.name, wq.cols, wq.rows, rank,
                    );
                }
                if targets.k {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wk.name, wk.cols, wk.rows, rank,
                    );
                }
                if targets.v {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wv.name, wv.cols, wv.rows, rank,
                    );
                }
                if targets.ssm_gates
                    && let Some(wg) = w_attn_gate
                {
                    let r_small = rank.min(wg.rows).max(1);
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wg.name, wg.cols, wg.rows, r_small,
                    );
                }
                if targets.o {
                    push_linear_pair(
                        &mut rng, &mut pairs, &mut order, wo.name, wo.cols, wo.rows, rank,
                    );
                }
            }
        }
        if targets.gate {
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                lw.w_gate.name,
                lw.w_gate.cols,
                lw.w_gate.rows,
                rank,
            );
        }
        if targets.up {
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                lw.w_up.name,
                lw.w_up.cols,
                lw.w_up.rows,
                rank,
            );
        }
        if targets.down {
            push_linear_pair(
                &mut rng,
                &mut pairs,
                &mut order,
                lw.w_down.name,
                lw.w_down.cols,
                lw.w_down.rows,
                rank,
            );
        }
    }

    if targets.lm_head {
        push_linear_pair(
            &mut rng,
            &mut pairs,
            &mut order,
            model.output.name,
            model.output.cols,
            model.output.rows,
            rank,
        );
    }

    Ok(LoraAdapter {
        arch: model.cfg.arch.clone(),
        alpha,
        name: Some("tinymodel-qlora".to_string()),
        pairs,
        pair_order: order,
    })
}

/// Initializes a new [`LoraAdapter`] for `LlamaModel` using Kaiming-uniform `A` and exact zero `B`
/// (`B == 0` guarantees step-0 forward is bit-identical to the frozen base model).
pub fn init_lora_adapter(
    model: &LlamaModel<'_>,
    rank: usize,
    alpha: f32,
    targets: &LoraTargets,
    seed: u64,
) -> Result<LoraAdapter, TrainError> {
    if rank == 0 {
        return Err(TrainError::InvalidConfig(
            "LoRA rank must be >= 1".to_string(),
        ));
    }
    let mut rng = SplitMix64::new(seed);
    let mut pairs = HashMap::new();
    let mut order = Vec::new();

    if targets.embed {
        let vocab = model.cfg.vocab_size;
        let d_out = model.cfg.n_embd;
        let bound = 1.0f32 / (rank as f32).sqrt();
        let a: Vec<f32> = (0..vocab * rank).map(|_| rng.uniform_sym(bound)).collect();
        let b = vec![0.0f32; d_out * rank];
        let name = "token_embd.weight".to_string();
        order.push(name.clone());
        pairs.insert(
            name.clone(),
            LoraWeightPair {
                base_name: name,
                is_token_embd: true,
                rank,
                d_in: vocab,
                d_out,
                a,
                b,
            },
        );
    }

    let mut add_linear = |base_name: &str, d_in: usize, d_out: usize| {
        let bound = 1.0f32 / (d_in as f32).sqrt();
        let a: Vec<f32> = (0..rank * d_in).map(|_| rng.uniform_sym(bound)).collect();
        let b = vec![0.0f32; d_out * rank];
        order.push(base_name.to_string());
        pairs.insert(
            base_name.to_string(),
            LoraWeightPair {
                base_name: base_name.to_string(),
                is_token_embd: false,
                rank,
                d_in,
                d_out,
                a,
                b,
            },
        );
    };

    for lw in &model.layers {
        if lw.wq.is_sliced {
            if targets.q || targets.k || targets.v {
                add_linear(lw.wq.name, lw.wq.cols, lw.wq.rows + lw.wk.rows + lw.wv.rows);
            }
        } else {
            if targets.q {
                add_linear(lw.wq.name, lw.wq.cols, lw.wq.rows);
            }
            if targets.k {
                add_linear(lw.wk.name, lw.wk.cols, lw.wk.rows);
            }
            if targets.v {
                add_linear(lw.wv.name, lw.wv.cols, lw.wv.rows);
            }
        }
        if targets.o {
            add_linear(lw.wo.name, lw.wo.cols, lw.wo.rows);
        }
        if let Some(ref moe) = lw.moe {
            if targets.gate {
                add_linear(moe.gate_inp.name, moe.gate_inp.cols, moe.gate_inp.rows);
            }
            if let Some(ref sh) = moe.shared_expert {
                if targets.gate {
                    add_linear(sh.w_gate.name, sh.w_gate.cols, sh.w_gate.rows);
                    if let Some(ref gi) = sh.w_gate_inp_shexp {
                        add_linear(gi.name, gi.cols, gi.rows);
                    }
                }
                if targets.up {
                    add_linear(sh.w_up.name, sh.w_up.cols, sh.w_up.rows);
                }
                if targets.down {
                    add_linear(sh.w_down.name, sh.w_down.cols, sh.w_down.rows);
                }
            }
        } else {
            if lw.w_gate.is_sliced {
                if targets.gate || targets.up {
                    add_linear(
                        lw.w_gate.name,
                        lw.w_gate.cols,
                        lw.w_gate.rows + lw.w_up.rows,
                    );
                }
            } else {
                if targets.gate {
                    add_linear(lw.w_gate.name, lw.w_gate.cols, lw.w_gate.rows);
                }
                if targets.up {
                    add_linear(lw.w_up.name, lw.w_up.cols, lw.w_up.rows);
                }
            }
            if targets.down {
                add_linear(lw.w_down.name, lw.w_down.cols, lw.w_down.rows);
            }
        }
    }

    if targets.lm_head {
        add_linear("output.weight", model.output.cols, model.output.rows);
    }

    Ok(LoraAdapter {
        arch: model.cfg.arch.clone(),
        alpha,
        name: Some("tinymodel-qlora".to_string()),
        pairs,
        pair_order: order,
    })
}

/// FP32 gradient accumulators matching `LoraAdapter::pair_order`.
#[derive(Debug, Clone)]
pub struct LoraGradPair {
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LoraGradStore {
    pub grads: HashMap<String, LoraGradPair>,
    pub order: Vec<String>,
}

impl LoraGradStore {
    pub fn from_adapter(adapter: &LoraAdapter) -> Self {
        let mut grads = HashMap::with_capacity(adapter.pair_order.len());
        for name in &adapter.pair_order {
            let p = &adapter.pairs[name];
            grads.insert(
                name.clone(),
                LoraGradPair {
                    da: vec![0.0; p.a.len()],
                    db: vec![0.0; p.b.len()],
                },
            );
        }
        Self {
            grads,
            order: adapter.pair_order.clone(),
        }
    }

    pub fn zero_grad(&mut self) {
        for g in self.grads.values_mut() {
            g.da.fill(0.0);
            g.db.fill(0.0);
        }
    }

    pub fn scale_grads(&mut self, scale: f32) {
        if (scale - 1.0).abs() < 1e-12 {
            return;
        }
        for name in &self.order {
            let g = self.grads.get_mut(name).expect("grad slot exists");
            for v in &mut g.da {
                *v *= scale;
            }
            for v in &mut g.db {
                *v *= scale;
            }
        }
    }

    /// Deterministic global L2 gradient norm across `order`.
    pub fn global_grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f64;
        for name in &self.order {
            let g = &self.grads[name];
            for &v in &g.da {
                let vf = v as f64;
                sum_sq += vf * vf;
            }
            for &v in &g.db {
                let vf = v as f64;
                sum_sq += vf * vf;
            }
        }
        sum_sq.sqrt() as f32
    }

    /// Clips gradients in-place to `max_norm` (`> 0.0`), returning the pre-clip global L2 norm.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.global_grad_norm();
        if max_norm > 0.0 && norm > max_norm {
            let scale = max_norm / (norm + 1e-12);
            self.scale_grads(scale);
        }
        norm
    }

    pub fn total_trainable_params(&self) -> usize {
        self.grads.values().map(|g| g.da.len() + g.db.len()).sum()
    }
}

/// AdamW + LoRA+ (`lr_b = lr * lora_plus_ratio`) configuration.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    pub lr: f32,
    pub lora_plus_ratio: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            lora_plus_ratio: 1.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

/// Cosine learning rate schedule with linear warmup (`step` is 0-indexed).
pub fn lr_at_step(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    base_lr: f32,
    min_lr_ratio: f32,
) -> f32 {
    if total_steps <= 1 {
        return base_lr;
    }
    if warmup_steps > 0 && step < warmup_steps {
        return base_lr * ((step + 1) as f32) / (warmup_steps as f32);
    }
    let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
    let progress =
        ((step.saturating_sub(warmup_steps)) as f64 / (decay_steps as f64)).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (core::f64::consts::PI * progress).cos());
    let min_lr = (base_lr as f64) * (min_lr_ratio as f64);
    (min_lr + ((base_lr as f64) - min_lr) * cosine) as f32
}

#[derive(Debug, Clone)]
pub struct AdamWMoments {
    pub m_a: Vec<f32>,
    pub v_a: Vec<f32>,
    pub m_b: Vec<f32>,
    pub v_b: Vec<f32>,
}

/// Decoupled AdamW optimizer with optional LoRA+ asymmetric `B` learning rate.
#[derive(Debug, Clone)]
pub struct AdamW {
    pub cfg: AdamWConfig,
    pub step: u64,
    pub moments: HashMap<String, AdamWMoments>,
}

impl AdamW {
    pub fn new(adapter: &LoraAdapter, cfg: AdamWConfig) -> Self {
        let mut moments = HashMap::with_capacity(adapter.pair_order.len());
        for name in &adapter.pair_order {
            let p = &adapter.pairs[name];
            moments.insert(
                name.clone(),
                AdamWMoments {
                    m_a: vec![0.0; p.a.len()],
                    v_a: vec![0.0; p.a.len()],
                    m_b: vec![0.0; p.b.len()],
                    v_b: vec![0.0; p.b.len()],
                },
            );
        }
        Self {
            cfg,
            step: 0,
            moments,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.moments
            .values()
            .map(|m| (m.m_a.len() + m.v_a.len() + m.m_b.len() + m.v_b.len()) * 4)
            .sum()
    }

    /// Applies one AdamW update step to `adapter` at learning rate `cur_lr`.
    pub fn step_update(&mut self, adapter: &mut LoraAdapter, grads: &LoraGradStore, cur_lr: f32) {
        self.step += 1;
        let t = self.step as i32;
        let b1 = self.cfg.beta1 as f64;
        let b2 = self.cfg.beta2 as f64;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        let eps = self.cfg.eps as f64;
        let wd = self.cfg.weight_decay as f64;
        let lr_a = cur_lr as f64;
        let lr_b = (cur_lr * self.cfg.lora_plus_ratio) as f64;

        let update_vec = |w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f64| {
            for i in 0..w.len() {
                let gi = g[i] as f64;
                if !gi.is_finite() {
                    continue;
                }
                let mi = b1 * (m[i] as f64) + (1.0 - b1) * gi;
                let vi = b2 * (v[i] as f64) + (1.0 - b2) * gi * gi;
                m[i] = mi as f32;
                v[i] = vi as f32;
                let m_hat = mi / bc1;
                let v_hat = vi / bc2;
                let wi = (w[i] as f64) * (1.0 - lr * wd) - lr * m_hat / (v_hat.sqrt() + eps);
                w[i] = wi as f32;
            }
        };

        for name in &adapter.pair_order {
            let pair = adapter.pairs.get_mut(name).expect("pair exists");
            let grad = &grads.grads[name];
            let mom = self.moments.get_mut(name).expect("moments exist");
            update_vec(&mut pair.a, &grad.da, &mut mom.m_a, &mut mom.v_a, lr_a);
            update_vec(&mut pair.b, &grad.db, &mut mom.m_b, &mut mom.v_b, lr_b);
        }
    }
}

fn encode_f32_le(vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(vals.len() * 4);
    for &x in vals {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Saves a full crash-safe training checkpoint (`adapter` + `AdamW` moments + `step`) as an atomic GGUF v3 file.
pub fn save_train_checkpoint<P: AsRef<Path>>(
    dst_path: P,
    adapter: &LoraAdapter,
    opt: &AdamW,
) -> Result<usize, TrainError> {
    let meta = vec![
        (
            "general.type".to_string(),
            MetadataValue::String("train_ckpt".to_string()),
        ),
        (
            "general.architecture".to_string(),
            MetadataValue::String(adapter.arch.clone()),
        ),
        (
            "adapter.lora.alpha".to_string(),
            MetadataValue::F32(adapter.alpha),
        ),
        ("train.step".to_string(), MetadataValue::U64(opt.step)),
        ("train.lr".to_string(), MetadataValue::F32(opt.cfg.lr)),
        (
            "train.lora_plus_ratio".to_string(),
            MetadataValue::F32(opt.cfg.lora_plus_ratio),
        ),
    ];

    struct Entry {
        name: String,
        dims: [usize; 2],
        bytes: Vec<u8>,
    }
    let mut entries = Vec::with_capacity(adapter.pair_order.len() * 6);
    for base in &adapter.pair_order {
        let p = &adapter.pairs[base];
        let m = &opt.moments[base];
        let a_dims = if p.is_token_embd {
            [p.rank, p.d_in]
        } else {
            [p.d_in, p.rank]
        };
        let b_dims = [p.rank, p.d_out];
        for (suffix, dims, slice) in [
            ("lora_a", a_dims, &p.a[..]),
            ("lora_b", b_dims, &p.b[..]),
            ("m_a", a_dims, &m.m_a[..]),
            ("v_a", a_dims, &m.v_a[..]),
            ("m_b", b_dims, &m.m_b[..]),
            ("v_b", b_dims, &m.v_b[..]),
        ] {
            entries.push(Entry {
                name: format!("{base}.{suffix}"),
                dims,
                bytes: encode_f32_le(slice),
            });
        }
    }
    let views: Vec<NewGgufTensor<'_>> = entries
        .iter()
        .map(|e| NewGgufTensor {
            name: &e.name,
            dims: &e.dims,
            dtype: DType::F32,
            data: &e.bytes,
        })
        .collect();
    Ok(write_gguf_file_atomic(dst_path, &meta, &views, 32)?)
}

/// Resumes `(LoraAdapter, AdamW)` from a GGUF checkpoint written by [`save_train_checkpoint`].
pub fn load_train_checkpoint<P: AsRef<Path>>(
    ckpt_path: P,
    cfg: AdamWConfig,
) -> Result<(LoraAdapter, AdamW), TrainError> {
    let gguf = GgufFile::open(&ckpt_path)?;
    let path_str = gguf.path().display().to_string();
    let gtype = gguf
        .get_meta("general.type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if gtype != "train_ckpt" {
        return Err(TrainError::Checkpoint {
            path: path_str,
            detail: format!("expected general.type='train_ckpt', got '{gtype}'"),
        });
    }
    let arch = gguf.req_str("general.architecture")?.to_string();
    let alpha = gguf
        .get_meta("adapter.lora.alpha")
        .and_then(|v| v.as_f32())
        .unwrap_or(0.0);
    let step = gguf
        .get_meta("train.step")
        .and_then(|v| v.as_usize())
        .unwrap_or(0) as u64;

    let mut order = Vec::new();
    for tname in gguf.tensor_names() {
        if let Some(base) = tname.strip_suffix(".lora_a") {
            order.push(base.to_string());
        }
    }
    if order.is_empty() {
        return Err(TrainError::Checkpoint {
            path: path_str,
            detail: "checkpoint contains 0 .lora_a tensors".to_string(),
        });
    }

    let read_vec = |name: &str, expected_len: usize| -> Result<Vec<f32>, TrainError> {
        let (info, bytes) = gguf.tensor_bytes(name, "ckpt")?;
        let mut out = vec![0.0f32; expected_len];
        dequantize_row_bytes(info.dtype, bytes, &mut out);
        Ok(out)
    };

    let mut pairs = HashMap::with_capacity(order.len());
    let mut moments = HashMap::with_capacity(order.len());

    for base in &order {
        let a_info =
            gguf.tensor_info(&format!("{base}.lora_a"))
                .ok_or_else(|| TrainError::Checkpoint {
                    path: path_str.clone(),
                    detail: format!("missing {base}.lora_a"),
                })?;
        let b_info =
            gguf.tensor_info(&format!("{base}.lora_b"))
                .ok_or_else(|| TrainError::Checkpoint {
                    path: path_str.clone(),
                    detail: format!("missing {base}.lora_b"),
                })?;
        let is_token_embd = base.ends_with("token_embd.weight");
        let (rank, d_in, d_out) = if is_token_embd {
            (a_info.dims[0], a_info.dims[1], b_info.dims[1])
        } else {
            (a_info.dims[1], a_info.dims[0], b_info.dims[1])
        };
        let a_len = a_info.dims[0] * a_info.dims[1];
        let b_len = b_info.dims[0] * b_info.dims[1];

        let a = read_vec(&format!("{base}.lora_a"), a_len)?;
        let b = read_vec(&format!("{base}.lora_b"), b_len)?;
        let m_a = read_vec(&format!("{base}.m_a"), a_len)?;
        let v_a = read_vec(&format!("{base}.v_a"), a_len)?;
        let m_b = read_vec(&format!("{base}.m_b"), b_len)?;
        let v_b = read_vec(&format!("{base}.v_b"), b_len)?;

        pairs.insert(
            base.clone(),
            LoraWeightPair {
                base_name: base.clone(),
                is_token_embd,
                rank,
                d_in,
                d_out,
                a,
                b,
            },
        );
        moments.insert(base.clone(), AdamWMoments { m_a, v_a, m_b, v_b });
    }

    let adapter = LoraAdapter {
        arch,
        alpha,
        name: Some("tinymodel-qlora".to_string()),
        pairs,
        pair_order: order,
    };
    let opt = AdamW { cfg, step, moments };
    Ok((adapter, opt))
}
