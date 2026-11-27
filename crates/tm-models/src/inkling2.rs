//! Thinking Machines Lab `Inkling-2-Mini` (`general.architecture == "inkling2_loop"` / `"inkling2"`)
//! looped Mixture-of-Experts forward pass + Loop-Aware 3-Tier Expert Placement (`ADR-016`).
//!
//! Upstream algorithmic reference: `ggml-org/llama.cpp` `src/models/inkling2.cpp` and
//! `src/models/inkling2-loop.cpp` (MIT License, Copyright (c) 2023-2026 Georgi Gerganov and
//! contributors).
//!
//! Strictly obeys `AGENTS.md`:
//! - `#![forbid(unsafe_code)]` (inherited from `tm-models` root — 0 `unsafe` blocks);
//! - Full dtype dispatch (`Rule #5`): every dense & expert matrix carries its own `DType`;
//! - Zero heap allocations on `forward_token` / `forward_batch` / `verify_speculative` (`Rule #4`);
//! - State safety (`Rule #6`): validates context bounds and token IDs BEFORE mutating `LayerState`;
//! - Determinism (`§9`): multi-token batch and parallel NVMe `O_DIRECT` expert evaluation reduce
//!   experts in fixed `0..n_expert_used` order with identical per-row dot products (`0-ULP` parity);
//! - Loop-Aware 3-Tier Memory Management (`ADR-016`): prioritizes `3×` looped core layers
//!   (`blk.core_start..core_end`) in anonymous RAM via uncached `DirectTensorReader` (`O_DIRECT`)
//!   so `25.91 GiB` `Inkling-2-Mini-Q8_0.gguf` runs at full precision on 16 GB RAM hosts without
//!   Linux page-cache duplication or swap thrashing.

use std::sync::Arc;

use tm_core::{DType, KvType, QK8_0, dequantize_row_bytes, probe_system_memory};
use tm_cpu::{
    HugePageAlignedPin, HugePageBytes, KernelTable, MAX_CPU_BATCH, MatvecScratch, ThreadPool,
    matmul_f32, matvec_batch_f32_unquant_act, matvec_f32, matvec_f32_unquant_act,
    release_mmap_page_cache, rms_norm, rms_norm_unit_inplace, softcap_inplace_parallel,
};
use tm_gguf::{DirectTensorReader, GgufError, GgufFile, PageAlignedSlot};

use crate::llama::ModelError;
use crate::state::{LayerState, MAX_SPEC_WINDOW, SpecVerifyBuf, decode_embedding_row};

/// Maximum multi-token lockstep window for `eval_moe_experts_window_f32` (`DSpark` verify + 256-tok prefill).
pub const MAX_INKLING2_WINDOW: usize = 256;

/// Controller indices for `inkling2_loop` cross-pass messages (`inkling2-loop.cpp:48-50`).
pub const LOOP_CTRL_Q: usize = 0;
pub const LOOP_CTRL_K: usize = 1;
pub const LOOP_CTRL_V: usize = 2;
pub const LOOP_CTRL_FFN: usize = 3;
pub const LOOP_CTRL_COUNT: usize = 4;
const LOOP_CTRL_NAMES: [&str; LOOP_CTRL_COUNT] = ["q", "k", "v", "ffn"];

/// Hyperparameters for `inkling2` and `inkling2_loop`.
#[derive(Debug, Clone)]
pub struct Inkling2Config {
    pub arch: String,
    pub n_layers_phys: usize,
    /// Logical KV + ShortConv slot count (`n_phys + (n_passes - 1) * (core_end - core_start)`).
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff_exp: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub expert_weights_scale: f32,
    pub n_head_per_layer: Vec<usize>,
    pub n_head_kv_per_layer: Vec<usize>,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub n_swa: usize,
    pub is_swa: Vec<bool>,
    pub shortconv_l_cache: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub attn_res_block_size: usize,
    pub final_logit_softcapping: Option<f32>,
    pub n_relative: usize,
    pub n_relative_extent: usize,
    pub qk_clip: f32,
    pub relative_clip: f32,
    pub log_scale_floor: usize,
    pub log_scale_alpha: f32,
    pub vocab_padded: usize,
    /// Unpadded active vocabulary size (`200058` for `Inkling-2-Mini`).
    pub vocab_size: usize,
    pub max_ctx_train: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub n_passes: usize,
    pub gate_rank: usize,
    pub loop_varying: bool,
    pub sconv_channels_max: usize,
}

impl Inkling2Config {
    #[inline]
    pub fn is_looped(&self) -> bool {
        self.n_passes > 1 && self.core_end > self.core_start
    }

    #[inline]
    pub fn layer_phys(&self, il: usize) -> usize {
        if il < self.n_layers_phys {
            il
        } else {
            self.core_start + (il - self.n_layers_phys) % (self.core_end - self.core_start)
        }
    }

    #[inline]
    pub fn n_head(&self, ip: usize) -> usize {
        self.n_head_per_layer[ip]
    }

    #[inline]
    pub fn n_head_kv(&self, ip: usize) -> usize {
        self.n_head_kv_per_layer[ip]
    }

    #[inline]
    pub fn q_dim(&self, ip: usize) -> usize {
        self.n_head(ip) * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self, ip: usize) -> usize {
        self.n_head_kv(ip) * self.head_dim
    }

    #[inline]
    pub fn rel_extent_for(&self, ip: usize) -> usize {
        if self.is_swa[ip] {
            self.n_swa
        } else {
            self.n_relative_extent
        }
    }

    #[inline]
    pub fn max_n_head(&self) -> usize {
        self.n_head_per_layer.iter().copied().max().unwrap_or(1)
    }

    #[inline]
    pub fn max_n_head_kv(&self) -> usize {
        self.n_head_kv_per_layer.iter().copied().max().unwrap_or(1)
    }

    #[inline]
    pub fn max_rel_extent(&self) -> usize {
        self.n_swa.max(self.n_relative_extent)
    }

    #[inline]
    pub fn max_depth_blocks(&self) -> usize {
        self.n_layers_phys.div_ceil(self.attn_res_block_size) + 2
    }

    /// Total floats across all `n_layers` logical `ShortConv` histories.
    #[inline]
    pub fn total_shortconv_floats(&self) -> usize {
        let hist_len = self.shortconv_l_cache.saturating_sub(1);
        let mut sum = 0usize;
        for il in 0..self.n_layers {
            let ip = self.layer_phys(il);
            sum += (2 * self.kv_dim(ip) + 2 * self.n_embd) * hist_len;
        }
        sum
    }
}

/// Options controlling how `Inkling2Model` places dense & MoE expert weights across RAM, VRAM, and NVMe.
#[derive(Debug, Clone, Default)]
pub struct Inkling2LoadOptions {
    /// Physical layers `[0..n_layers_phys)` whose MoE experts are already resident in GPU VRAM
    /// (`Inkling2VkRunner`), so host RAM skips allocating them (`0` duplicate bytes).
    pub vram_expert_layers: Vec<bool>,
    /// Optional explicit host anonymous-RAM budget (in bytes) for MoE expert layers.
    /// When `None`, checks `TM_INKLING2_RAM_MIB`, then `probe_system_memory()`.
    pub ram_expert_budget_bytes: Option<u64>,
    /// When `true` (`Inkling2VkRunner` already owns all dense/KV weights in VRAM), skips reading
    /// `~1.70 GiB` of dense tensors into host RAM, freeing `~1.70 GiB` for 2 additional MoE layers!
    pub host_moe_only: bool,
}

/// Summary of loop-aware MoE expert tier placement (`ADR-016`).
#[derive(Debug, Clone, Default)]
pub struct Inkling2PlacementStats {
    pub dense_bytes: u64,
    pub vram_expert_layers: usize,
    pub ram_mmap_expert_layers: usize,
    pub ram_owned_expert_layers: usize,
    pub nvme_direct_expert_layers: usize,
    pub core_layers_in_fast_tiers: usize,
    pub ram_owned_bytes: u64,
    pub nvme_bytes_per_decode_tok: u64,
    pub direct_io_active: bool,
}

/// Stage-by-stage wall-clock & active-byte profile for `Inkling2Model` decode (`tinymodel profile`).
#[derive(Debug, Clone, Default)]
pub struct Inkling2DecodeProfile {
    pub n_steps: usize,
    pub embed_mix_ns: u64,
    pub loop_ctrl_ns: u64,
    pub dense_attn_proj_ns: u64,
    pub gqa_kv_ns: u64,
    pub moe_route_ns: u64,
    pub moe_ram_ns: u64,
    pub moe_ssd_ns: u64,
    pub lm_head_ns: u64,
    pub dense_attn_bytes: u64,
    pub moe_ram_bytes: u64,
    pub moe_ssd_bytes: u64,
    pub lm_head_bytes: u64,
}

/// Backing bytes for a matrix (zero-copy `GgufFile` slice, `Vec<u8>`, or `2 MiB` `MADV_HUGEPAGE` RAM).
pub enum BytesBacking<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
    HugePage(Arc<HugePageBytes>),
}

impl<'a> BytesBacking<'a> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            BytesBacking::Borrowed(b) => b,
            BytesBacking::Owned(v) => v.as_slice(),
            BytesBacking::HugePage(hp) => hp.as_slice(),
        }
    }
}

/// Matrix tensor carrying its own `DType` (`AGENTS.md` Rule #5).
pub struct OwnedOrBorrowedMat<'a> {
    pub backing: BytesBacking<'a>,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> OwnedOrBorrowedMat<'a> {
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.backing.as_slice()
    }
}

/// Storage tier for one physical layer's `[n_expert]` MoE expert weights (`ffn_gate_up_exps` + `ffn_down_exps`).
pub enum ExpertLayerStore<'a> {
    /// Resident in host memory (`mmap` zero-copy when model fits in RAM, or anonymous `Vec<u8>`
    /// populated via `O_DIRECT` when tiered placement is active).
    InMemory {
        gate_up: BytesBacking<'a>,
        gate_up_dtype: DType,
        gate_up_expert_bytes: usize,
        down: BytesBacking<'a>,
        down_dtype: DType,
        down_expert_bytes: usize,
    },
    /// Streamed on demand per active expert via `DirectTensorReader` (`O_DIRECT` + `pread64`)
    /// into pre-allocated `PageAlignedSlot`s (`0` Linux page-cache footprint, `0` token-path allocs).
    NvmeDirect {
        gate_up_offset: u64,
        gate_up_dtype: DType,
        gate_up_expert_bytes: usize,
        down_offset: u64,
        down_dtype: DType,
        down_expert_bytes: usize,
    },
    /// Owned by GPU VRAM (`Inkling2VkHybridRunner`); host stores 0 bytes for this layer.
    DeviceVram,
}

/// Loop controller weights for one core physical layer `ip in core_start..core_end`.
pub struct Inkling2LoopWeights {
    /// `[4, n_embd]` row-major (`ne = [n_embd, 4]`).
    pub msg_query: Vec<f32>,
    /// `[LOOP_CTRL_COUNT]` controllers (`Q=0, K=1, V=2, FFN=3`), each `[gate_rank, 2 * n_embd]`.
    pub ctrl_down: [Vec<f32>; LOOP_CTRL_COUNT],
    /// `[LOOP_CTRL_COUNT]`, each `[n_embd, gate_rank]`.
    pub ctrl_gain: [Vec<f32>; LOOP_CTRL_COUNT],
    /// `[LOOP_CTRL_COUNT]`, each `[n_embd, gate_rank]`.
    pub ctrl_forget: [Vec<f32>; LOOP_CTRL_COUNT],
}

/// Dense + MoE weights for one physical layer `ip in 0..n_layers_phys`.
pub struct Inkling2LayerWeights<'a> {
    pub wq: OwnedOrBorrowedMat<'a>,
    pub wk: OwnedOrBorrowedMat<'a>,
    pub wv: OwnedOrBorrowedMat<'a>,
    pub wo: OwnedOrBorrowedMat<'a>,
    pub wqkv_gate: OwnedOrBorrowedMat<'a>,
    pub attn_q_clip: Vec<f32>,
    pub attn_sink_proj: OwnedOrBorrowedMat<'a>,
    pub attn_rel: OwnedOrBorrowedMat<'a>,
    /// Row-major `[extent, n_rel]` (`ne = [n_rel, extent]`).
    pub attn_rel_bias: Vec<f32>,
    pub attn_rel_clip: Vec<f32>,
    /// Row-major `[kv_dim, n_conv]` (`ne = [n_conv, kv_dim]`).
    pub attn_k_conv: Vec<f32>,
    pub attn_v_conv: Vec<f32>,
    /// Row-major `[n_embd, n_conv]` (`ne = [n_conv, n_embd]`).
    pub attn_post_conv: Vec<f32>,
    pub ffn_post_conv: Vec<f32>,
    /// Present on SWA layers (`is_swa[ip] == true`).
    pub attn_norm: Option<Vec<f32>>,
    /// Present on full-attention layers (`is_swa[ip] == false`).
    pub attn_q_inp_norm: Option<Vec<f32>>,
    pub attn_k_inp_norm: Option<Vec<f32>>,
    pub attn_v_inp_norm: Option<Vec<f32>>,
    pub attn_k_conv_proj: Option<OwnedOrBorrowedMat<'a>>,
    pub attn_v_conv_proj: Option<OwnedOrBorrowedMat<'a>>,
    /// Present on `ip > 0`: `[n_embd]` if `is_swa[ip]`, `[3 * n_embd]` if `!is_swa[ip]`.
    pub attn_res_query: Option<Vec<f32>>,
    pub ffn_res_query: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate_inp: OwnedOrBorrowedMat<'a>,
    pub exp_probs_b: Option<Vec<f32>>,
    /// Optional per-expert global FP32 output scale (`blk.{ip}.ffn_gate_up_exps.scale`, `[n_expert]`) for `NVFP4`.
    pub ffn_gate_up_exps_scale: Option<Vec<f32>>,
    /// Optional per-expert global FP32 output scale (`blk.{ip}.ffn_down_exps.scale`, `[n_expert]`) for `NVFP4`.
    pub ffn_down_exps_scale: Option<Vec<f32>>,
    pub experts: ExpertLayerStore<'a>,
}

/// Loaded `Inkling-2` / `Inkling-2-Mini` (`inkling2_loop`) model.
pub struct Inkling2Model<'a> {
    pub cfg: Inkling2Config,
    pub stats: Inkling2PlacementStats,
    pub token_embd: OwnedOrBorrowedMat<'a>,
    pub token_embd_norm: Vec<f32>,
    pub output_res_query: Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output: OwnedOrBorrowedMat<'a>,
    pub layers: Vec<Inkling2LayerWeights<'a>>,
    pub loop_layers: Vec<Inkling2LoopWeights>,
    direct_reader: Option<DirectTensorReader>,
    max_expert_chunk_bytes: usize,
}

/// Per-expert worker item used for parallel zero-allocation `O_DIRECT` + AVX2 MoE execution.
pub struct ExpertWorkSlot {
    pub slot: PageAlignedSlot,
    pub mv_scratch: MatvecScratch,
    pub gate_up_buf: Vec<f32>,
    pub hidden_buf: Vec<f32>,
    /// Multi-token window gathered input `[MAX_INKLING2_WINDOW * n_embd.max(n_ff_exp)]`.
    pub win_in_buf: Vec<f32>,
    /// Multi-token window gate_up buffer `[MAX_INKLING2_WINDOW * 2 * n_ff_exp]`.
    pub win_gate_up_buf: Vec<f32>,
    /// Multi-token window hidden buffer `[MAX_INKLING2_WINDOW * n_ff_exp]` for grouped MoE.
    pub win_hidden_buf: Vec<f32>,
    pub out_buf: Vec<f32>,
}

/// Pre-allocated per-session inference state for `Inkling2Model` (`0` heap allocations on token path).
pub struct Inkling2State {
    pub max_ctx: usize,
    /// `[0..n_slots)` = `LayerState::Kv`, `[n_slots..2 * n_slots)` = `LayerState::ShortConv`.
    pub layers: Vec<LayerState>,
    cur: Vec<f32>,
    inp_sa: Vec<f32>,
    ffn_inp: Vec<f32>,
    prelude: Vec<f32>,
    bank_out: Vec<f32>,
    blocks_buf: Vec<f32>,
    blocks_len: usize,
    prelude_blocks_buf: Vec<f32>,
    prelude_blocks_len: usize,
    bank_buf: Vec<f32>,
    bank_len: usize,
    mix_states_hat: Vec<f32>,
    mix_scores: Vec<f32>,
    inputs: Vec<f32>,
    messages: Vec<f32>,
    ctrl_in: Vec<f32>,
    ctrl_g: Vec<f32>,
    ctrl_gain: Vec<f32>,
    ctrl_forget: Vec<f32>,
    q_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    r_buf: Vec<f32>,
    k_dyn: Vec<f32>,
    v_dyn: Vec<f32>,
    conv_tmp: Vec<f32>,
    sinks: Vec<f32>,
    rel_table: Vec<f32>,
    scores: Vec<f32>,
    attn_vec: Vec<f32>,
    gate_buf: Vec<f32>,
    wo_out: Vec<f32>,
    router_logits: Vec<f32>,
    router_probs: Vec<f32>,
    router_sel: Vec<f32>,
    top_ids: Vec<usize>,
    top_weights: Vec<f32>,
    moe_out: Vec<f32>,
    logits_padded: Vec<f32>,
    logits: Vec<f32>,
    pub last_hidden: Vec<f32>,
    /// Saved visit inputs for `DSpark` (`dflash`) feature extraction (`inkling2-loop.cpp` PR #32218):
    /// `[0..n_slots)` = `cur` entering visit `v`, slot `n_slots` = `result_norm` (`last_hidden`).
    pub slot_outputs: Vec<f32>,
    mv_scratch: MatvecScratch,
    /// Single-thread pool handle (`0` worker threads spawned) passed into per-expert workers.
    serial_pool: ThreadPool,
    /// Dedicated concurrent I/O pool (`top_k <= 8` threads) for `NvmeDirect` SSD expert streaming
    /// so `-t 4` P-core compute pools still issue full queue-depth `8` `pread`s on outer layers.
    io_pool: ThreadPool,
    pub expert_slots: Vec<ExpertWorkSlot>,
    /// Pre-allocated `[MAX_INKLING2_WINDOW * n_expert_used * n_embd]` per-slot outputs for `0-ULP`
    /// deterministic multi-token grouped MoE evaluation.
    pub win_slot_outs: Vec<f32>,
    win_cur: Vec<f32>,
    win_inp_sa: Vec<f32>,
    win_ffn_inp: Vec<f32>,
    win_prelude: Vec<f32>,
    win_bank_out: Vec<f32>,
    win_blocks_buf: Vec<f32>,
    win_prelude_blocks_buf: Vec<f32>,
    win_bank_buf: Vec<f32>,
    win_messages: Vec<f32>,
    win_q_in: Vec<f32>,
    win_k_in: Vec<f32>,
    win_v_in: Vec<f32>,
    win_q_buf: Vec<f32>,
    win_k_buf: Vec<f32>,
    win_v_buf: Vec<f32>,
    win_r_buf: Vec<f32>,
    win_sinks: Vec<f32>,
    win_gate_buf: Vec<f32>,
    win_k_dyn: Vec<f32>,
    win_v_dyn: Vec<f32>,
    win_attn_vec: Vec<f32>,
    win_wo_out: Vec<f32>,
    win_moe_out: Vec<f32>,
    win_top_ids: Vec<usize>,
    win_top_weights: Vec<f32>,
}

fn read_u_or_arr(
    gguf: &GgufFile,
    key: &str,
    expected_len: usize,
) -> Result<Vec<usize>, ModelError> {
    let mv = gguf
        .get_meta(key)
        .ok_or_else(|| GgufError::MissingMetadata {
            path: gguf.path().display().to_string(),
            key: key.to_string(),
        })?;
    if let Some(arr) = mv.as_usize_array() {
        if arr.len() != expected_len {
            return Err(GgufError::MetadataTypeMismatch {
                path: gguf.path().display().to_string(),
                key: key.to_string(),
                expected: "array matching block_count",
            }
            .into());
        }
        Ok(arr)
    } else if let Some(single) = mv.as_usize() {
        Ok(vec![single; expected_len])
    } else {
        Err(GgufError::MetadataTypeMismatch {
            path: gguf.path().display().to_string(),
            key: key.to_string(),
            expected: "integer or integer[]",
        }
        .into())
    }
}

fn load_f32_vec(
    gguf: &GgufFile,
    direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    expected_dims: &[usize],
) -> Result<Vec<f32>, ModelError> {
    load_f32_vec_opt(gguf, direct, name, layer, expected_dims, false)
}

fn load_f32_vec_opt(
    gguf: &GgufFile,
    direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    expected_dims: &[usize],
    skip_data: bool,
) -> Result<Vec<f32>, ModelError> {
    if skip_data {
        let _ = gguf.req_tensor_file_range(name, layer, expected_dims)?;
        return Ok(Vec::new());
    }
    let n_elem: usize = expected_dims.iter().product();
    let mut out = vec![0.0f32; n_elem];
    if let Some((reader, slot)) = direct {
        let (info, f_off, b_len) = gguf.req_tensor_file_range(name, layer, expected_dims)?;
        let mut raw = vec![0u8; b_len];
        reader.read_exact_uncached(f_off, &mut raw, slot)?;
        dequantize_row_bytes(info.dtype, &raw, &mut out);
    } else {
        let (info, bytes) = gguf.req_tensor(name, layer, expected_dims)?;
        dequantize_row_bytes(info.dtype, bytes, &mut out);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn load_dense_mat_opt<'a>(
    gguf: &'a GgufFile,
    mut direct: Option<(&DirectTensorReader, &mut PageAlignedSlot)>,
    name: &str,
    layer: &str,
    cols: usize,
    rows: usize,
    dense_bytes_acc: &mut u64,
    skip_data: bool,
) -> Result<OwnedOrBorrowedMat<'a>, ModelError> {
    if skip_data {
        let (info, _, _) = gguf.req_tensor_file_range(name, layer, &[cols, rows])?;
        return Ok(OwnedOrBorrowedMat {
            backing: BytesBacking::Borrowed(&[]),
            dtype: info.dtype,
            rows,
            cols,
        });
    }
    if let Some((reader, ref mut slot)) = direct {
        let (info, f_off, b_len) = gguf.req_tensor_file_range(name, layer, &[cols, rows])?;
        let dtype = info.dtype;
        let mut owned = vec![0u8; b_len];
        reader.read_exact_uncached(f_off, &mut owned, slot)?;
        let _ = tm_cpu::mlock_slice(&owned);
        *dense_bytes_acc += b_len as u64;
        Ok(OwnedOrBorrowedMat {
            backing: BytesBacking::Owned(owned),
            dtype,
            rows,
            cols,
        })
    } else {
        let (info, bytes) = gguf.req_tensor(name, layer, &[cols, rows])?;
        *dense_bytes_acc += bytes.len() as u64;
        Ok(OwnedOrBorrowedMat {
            backing: BytesBacking::Borrowed(bytes),
            dtype: info.dtype,
            rows,
            cols,
        })
    }
}

impl Inkling2Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, ModelError> {
        let arch = gguf.req_str("general.architecture")?.to_string();
        if arch != "inkling2" && arch != "inkling2_loop" {
            return Err(ModelError::UnsupportedArch(arch));
        }
        let pfx = arch.as_str();
        let n_layers_phys = gguf.req_usize(&format!("{pfx}.block_count"))?;
        let max_ctx_train = gguf
            .get_meta(&format!("{pfx}.context_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);
        let n_embd = gguf.req_usize(&format!("{pfx}.embedding_length"))?;
        let n_ff_arr = read_u_or_arr(
            gguf,
            &format!("{pfx}.expert_feed_forward_length"),
            n_layers_phys,
        )?;
        let n_ff_exp = n_ff_arr[0];
        let n_expert = gguf.req_usize(&format!("{pfx}.expert_count"))?;
        let eu_arr = read_u_or_arr(gguf, &format!("{pfx}.expert_used_count"), n_layers_phys)?;
        let n_expert_used = eu_arr[0];
        let expert_weights_scale = gguf
            .get_meta(&format!("{pfx}.expert_weights_scale"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1.0);

        let n_head_per_layer =
            read_u_or_arr(gguf, &format!("{pfx}.attention.head_count"), n_layers_phys)?;
        let n_head_kv_per_layer = read_u_or_arr(
            gguf,
            &format!("{pfx}.attention.head_count_kv"),
            n_layers_phys,
        )?;
        let head_dim = gguf.req_usize(&format!("{pfx}.attention.key_length"))?;
        let val_dim = gguf
            .get_meta(&format!("{pfx}.attention.value_length"))
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);
        if val_dim != head_dim {
            return Err(ModelError::InvalidConfig(format!(
                "inkling2 key_length ({head_dim}) != value_length ({val_dim})"
            )));
        }

        let rms_eps = gguf.req_f32(&format!("{pfx}.attention.layer_norm_rms_epsilon"))?;
        let n_swa = gguf.req_usize(&format!("{pfx}.attention.sliding_window"))?;
        let swa_key = format!("{pfx}.attention.sliding_window_pattern");
        let is_swa = gguf
            .get_meta(&swa_key)
            .and_then(|v| v.as_bool_array())
            .ok_or_else(|| GgufError::MissingMetadata {
                path: gguf.path().display().to_string(),
                key: swa_key.clone(),
            })?;
        if is_swa.len() != n_layers_phys {
            return Err(ModelError::InvalidConfig(format!(
                "{swa_key} length {} != block_count {n_layers_phys}",
                is_swa.len()
            )));
        }

        let shortconv_l_cache = gguf.req_usize(&format!("{pfx}.shortconv.l_cache"))?;
        let situ_beta = gguf.req_f32(&format!("{pfx}.activation.situ_beta"))?;
        let situ_linear_beta = gguf.req_f32(&format!("{pfx}.activation.situ_linear_beta"))?;
        let attn_res_block_size = gguf.req_usize(&format!("{pfx}.attn_res.block_size"))?;
        let final_logit_softcapping = gguf
            .get_meta(&format!("{pfx}.final_logit_softcapping"))
            .and_then(|v| v.as_f32())
            .filter(|&c| c > 0.0);

        let n_relative = gguf.req_usize(&format!("{pfx}.attention.relative_length"))?;
        let n_relative_extent = gguf.req_usize(&format!("{pfx}.attention.relative_extent"))?;
        let qk_clip = gguf.req_f32(&format!("{pfx}.attention.qk_clip"))?;
        let relative_clip = gguf.req_f32(&format!("{pfx}.attention.relative_clip"))?;
        let log_scale_floor = gguf.req_usize(&format!("{pfx}.attention.log_scale_floor"))?;
        let log_scale_alpha = gguf.req_f32(&format!("{pfx}.attention.log_scale_alpha"))?;
        let vocab_padded = gguf.req_usize(&format!("{pfx}.vocab_size"))?;
        let vocab_size = gguf.req_usize(&format!("{pfx}.vocab_size_unpadded"))?;

        if n_swa == 0
            || n_relative == 0
            || n_relative_extent == 0
            || attn_res_block_size == 0
            || log_scale_floor == 0
            || !(2..=64).contains(&shortconv_l_cache)
            || vocab_size == 0
            || vocab_size > vocab_padded
            || n_expert_used == 0
            || n_expert_used > n_expert
        {
            return Err(ModelError::InvalidConfig(
                "inkling2: invalid dimensions, vocabulary, or layer pattern".to_string(),
            ));
        }

        let (core_start, core_end, n_passes, gate_rank, loop_varying, n_layers) = if arch
            == "inkling2_loop"
        {
            let cs = gguf.req_usize(&format!("{pfx}.loop.start"))?;
            let ce = gguf.req_usize(&format!("{pfx}.loop.end"))?;
            let np = gguf.req_usize(&format!("{pfx}.loop.count"))?;
            let gr = gguf.req_usize(&format!("{pfx}.loop.gate_rank"))?;
            let lv = gguf
                .get_meta(&format!("{pfx}.loop.varying"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let min_passes = if lv { 1 } else { 2 };
            if np < min_passes
                || gr == 0
                || cs == 0
                || cs >= ce
                || ce > n_layers_phys
                || !cs.is_multiple_of(attn_res_block_size)
                || !ce.is_multiple_of(attn_res_block_size)
            {
                return Err(ModelError::InvalidConfig(format!(
                    "inkling2_loop: invalid loop core [{cs}, {ce}) block={attn_res_block_size} passes={np} rank={gr}"
                )));
            }
            let n_slot = n_layers_phys + (np - 1) * (ce - cs);
            (cs, ce, np, gr, lv, n_slot)
        } else {
            (0, 0, 1, 0, false, n_layers_phys)
        };

        let mut sconv_channels_max = 0usize;
        for ip in 0..n_layers_phys {
            let nh = n_head_per_layer[ip];
            let nh_kv = n_head_kv_per_layer[ip];
            if nh_kv == 0 || !nh.is_multiple_of(nh_kv) {
                return Err(ModelError::InvalidConfig(format!(
                    "inkling2 layer {ip}: invalid head counts ({nh} / {nh_kv})"
                )));
            }
            sconv_channels_max = sconv_channels_max.max(2 * nh_kv * head_dim + 2 * n_embd);
        }

        Ok(Self {
            arch,
            n_layers_phys,
            n_layers,
            n_embd,
            n_ff_exp,
            n_expert,
            n_expert_used,
            expert_weights_scale,
            n_head_per_layer,
            n_head_kv_per_layer,
            head_dim,
            rms_eps,
            n_swa,
            is_swa,
            shortconv_l_cache,
            situ_beta,
            situ_linear_beta,
            attn_res_block_size,
            final_logit_softcapping,
            n_relative,
            n_relative_extent,
            qk_clip,
            relative_clip,
            log_scale_floor,
            log_scale_alpha,
            vocab_padded,
            vocab_size,
            max_ctx_train,
            core_start,
            core_end,
            n_passes,
            gate_rank,
            loop_varying,
            sconv_channels_max,
        })
    }
}

impl<'a> Inkling2Model<'a> {
    /// Loads `Inkling2Model` with automatic loop-aware RAM/NVMe tier planning (`ADR-016`).
    pub fn from_gguf(gguf: &'a GgufFile) -> Result<Self, ModelError> {
        Self::from_gguf_with_options(gguf, &Inkling2LoadOptions::default())
    }

    /// Loads `Inkling2Model` with explicit VRAM skip mask and/or RAM expert budget.
    pub fn from_gguf_with_options(
        gguf: &'a GgufFile,
        opts: &Inkling2LoadOptions,
    ) -> Result<Self, ModelError> {
        let cfg = Inkling2Config::from_gguf(gguf)?;
        let n_phys = cfg.n_layers_phys;
        let n_embd = cfg.n_embd;
        let n_ff = cfg.n_ff_exp;
        let n_exp = cfg.n_expert;

        // Inspect total file bytes and expert layer byte sizes without touching mmap pages!
        let mut total_model_bytes = 0u64;
        for tname in gguf.tensor_names() {
            if let Some(ti) = gguf.tensor_info(tname) {
                total_model_bytes += ti.byte_len as u64;
            }
        }
        let mut expert_layer_bytes = vec![0u64; n_phys];
        let mut max_expert_chunk_bytes = 4096usize;
        for (ip, layer_b) in expert_layer_bytes.iter_mut().enumerate() {
            let ltag = format!("blk.{ip}");
            let (_, _, gu_b) = gguf.req_tensor_file_range(
                &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                &ltag,
                &[n_embd, 2 * n_ff, n_exp],
            )?;
            let (_, _, dn_b) = gguf.req_tensor_file_range(
                &format!("blk.{ip}.ffn_down_exps.weight"),
                &ltag,
                &[n_ff, n_embd, n_exp],
            )?;
            *layer_b = (gu_b + dn_b) as u64;
            max_expert_chunk_bytes = max_expert_chunk_bytes.max(gu_b / n_exp).max(dn_b / n_exp);
        }

        let env_ram_mib: Option<u64> = std::env::var("TM_INKLING2_RAM_MIB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        let explicit_budget = opts
            .ram_expert_budget_bytes
            .or_else(|| env_ram_mib.map(|m| m * 1024 * 1024));

        let has_vram_skip = opts.vram_expert_layers.iter().any(|&v| v);
        let sys_avail = probe_system_memory()
            .map(|b| b.available_bytes)
            .unwrap_or(16 * 1024 * 1024 * 1024);

        // Use zero-copy mmap ONLY when no explicit cap/VRAM split is requested AND the entire GGUF
        // fits comfortably inside `available_bytes` with >= 2 GiB OS + KV headroom.
        let can_zero_copy_mmap = explicit_budget.is_none()
            && !has_vram_skip
            && total_model_bytes.saturating_add(2 * 1024 * 1024 * 1024) <= sys_avail;

        let file_exists = gguf.path().exists();
        let use_direct = !can_zero_copy_mmap && file_exists;

        let mut direct_reader = if use_direct {
            Some(DirectTensorReader::open(gguf.path())?)
        } else {
            None
        };
        let macos_nocache = direct_reader
            .as_ref()
            .is_some_and(|r| tm_cpu::set_file_nocache(r.fallback_file()));
        let mut io_slot = PageAlignedSlot::new(16 * 1024 * 1024);
        let mut stats = Inkling2PlacementStats {
            direct_io_active: direct_reader.as_ref().is_some_and(|r| r.is_o_direct())
                || macos_nocache,
            ..Default::default()
        };

        let skip_dense = opts.host_moe_only;
        // `token_embd.weight` (`883 MiB` BF16) only reads a single `4.5 KiB` row per token, so keep it
        // zero-copy borrowed from `GgufFile`'s `mmap` even when `use_direct` is active! That frees
        // ~883 MiB of host anonymous RAM for nearly 2 extra `NVFP4` MoE layers on 16 GB unified memory.
        let token_embd = load_dense_mat_opt(
            gguf,
            None,
            "token_embd.weight",
            "global",
            n_embd,
            cfg.vocab_padded,
            &mut stats.dense_bytes,
            skip_dense,
        )?;
        let token_embd_norm = load_f32_vec_opt(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "token_embd_norm.weight",
            "global",
            &[n_embd],
            skip_dense,
        )?;
        let output_res_query = load_f32_vec_opt(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output_res_query.weight",
            "global",
            &[n_embd],
            skip_dense,
        )?;
        let output_norm = load_f32_vec_opt(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output_norm.weight",
            "global",
            &[n_embd],
            skip_dense,
        )?;
        let output = load_dense_mat_opt(
            gguf,
            direct_reader.as_ref().map(|r| (r, &mut io_slot)),
            "output.weight",
            "global",
            n_embd,
            cfg.vocab_padded,
            &mut stats.dense_bytes,
            skip_dense,
        )?;

        // Compute exact remaining per-layer dense bytes (`wq/wk/wv/wo/gate/rel/loop`) before allocating MoE RAM.
        let mut rem_dense_layer_bytes = 0u64;
        if !skip_dense {
            for tname in gguf.tensor_names() {
                if tname.starts_with("blk.")
                    && !tname.ends_with("_exps.weight")
                    && let Some(ti) = gguf.tensor_info(tname)
                {
                    rem_dense_layer_bytes += ti.byte_len as u64;
                }
            }
        }

        // Determine loop-prioritized placement for each physical layer's MoE experts:
        // Priority 1: Core layers `core_start..core_end` (executed `n_passes` times per token!)
        // Priority 2: Outer layers `0..core_start` and `core_end..n_phys` (executed `1` time per token!)
        let mut ram_expert_plan = vec![false; n_phys];
        if !use_direct {
            ram_expert_plan.fill(true);
        } else {
            let auto_budget = {
                let (sys_tot, fresh_avail) = probe_system_memory()
                    .map(|b| (b.total_bytes, b.available_bytes))
                    .unwrap_or((16 * 1024 * 1024 * 1024, sys_avail));
                // When `skip_dense` is true (`host_moe_only`), dense weights are 0 bytes on host;
                // otherwise reserve `rem_dense_layer_bytes` + 1.85 GiB for OS, KV state & scratch.
                let reserve = if skip_dense {
                    2300 * 1024 * 1024u64
                } else {
                    rem_dense_layer_bytes + 1850 * 1024 * 1024u64
                };
                let by_avail = fresh_avail.saturating_sub(reserve);
                #[cfg(target_os = "macos")]
                {
                    // On macOS unified memory, XNU `vm_compressor` begins compressing anonymous
                    // pages once a single process exceeds ~52% of physical RAM (`~8.5 GiB` on 16 GB).
                    // Cap owned MoE + owned dense (`output` + `rem_dense_layer_bytes`) below 52% of RAM
                    // so `dense` BF16 + `core` NVFP4 layers remain 100% uncompressed with 0 swapouts.
                    let owned_dense = output.bytes().len() as u64 + rem_dense_layer_bytes;
                    let macos_moe_cap = ((sys_tot * 52) / 100).saturating_sub(owned_dense);
                    by_avail.min(macos_moe_cap)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = sys_tot;
                    by_avail
                }
            };
            let mut rem_budget = explicit_budget.unwrap_or(auto_budget);

            let mut priority_order: Vec<usize> = (cfg.core_start..cfg.core_end).collect();
            priority_order.extend(0..cfg.core_start);
            priority_order.extend(cfg.core_end..n_phys);

            for ip in priority_order {
                let in_vram = opts.vram_expert_layers.get(ip).copied().unwrap_or(false);
                if in_vram {
                    continue;
                }
                let need = expert_layer_bytes[ip];
                if need <= rem_budget {
                    ram_expert_plan[ip] = true;
                    rem_budget -= need;
                }
            }
        }

        let mut layers = Vec::with_capacity(n_phys);
        for (ip, &in_ram_plan) in ram_expert_plan.iter().enumerate() {
            let ltag = format!("blk.{ip}");
            let swa = cfg.is_swa[ip];
            let q_dim = cfg.q_dim(ip);
            let kv_dim = cfg.kv_dim(ip);
            let nh = cfg.n_head(ip);
            let nh_kv = cfg.n_head_kv(ip);
            let hd = cfg.head_dim;
            let n_rel = cfg.n_relative;
            let n_conv = cfg.shortconv_l_cache;
            let extent = cfg.rel_extent_for(ip);

            let wq = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_q.weight"),
                &ltag,
                n_embd,
                q_dim,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let wk = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_k.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let wv = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_v.weight"),
                &ltag,
                n_embd,
                kv_dim,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let wo = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_output.weight"),
                &ltag,
                q_dim,
                n_embd,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let wqkv_gate = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_gate.weight"),
                &ltag,
                n_embd,
                q_dim,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let attn_q_clip = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_q_clip.weight"),
                &ltag,
                &[hd, nh],
                skip_dense,
            )?;
            let attn_sink_proj = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_sink_proj.weight"),
                &ltag,
                n_embd,
                nh,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let attn_rel = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel.weight"),
                &ltag,
                n_embd,
                n_rel * nh,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let attn_rel_bias = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel_bias.weight"),
                &ltag,
                &[n_rel, extent],
                skip_dense,
            )?;
            let attn_rel_clip = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_rel_clip.weight"),
                &ltag,
                &[n_rel, nh],
                skip_dense,
            )?;

            let attn_k_conv = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_k_conv.weight"),
                &ltag,
                &[n_conv, kv_dim],
                skip_dense,
            )?;
            let attn_v_conv = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_v_conv.weight"),
                &ltag,
                &[n_conv, kv_dim],
                skip_dense,
            )?;
            let attn_post_conv = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.attn_post_conv.weight"),
                &ltag,
                &[n_conv, n_embd],
                skip_dense,
            )?;
            let ffn_post_conv = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_post_conv.weight"),
                &ltag,
                &[n_conv, n_embd],
                skip_dense,
            )?;

            let (
                attn_norm,
                attn_q_inp_norm,
                attn_k_inp_norm,
                attn_v_inp_norm,
                attn_k_conv_proj,
                attn_v_conv_proj,
            ) = if swa {
                let an = load_f32_vec_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_norm.weight"),
                    &ltag,
                    &[n_embd],
                    skip_dense,
                )?;
                (Some(an), None, None, None, None, None)
            } else {
                let qn = load_f32_vec_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_q_inp_norm.weight"),
                    &ltag,
                    &[n_embd],
                    skip_dense,
                )?;
                let kn = load_f32_vec_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_k_inp_norm.weight"),
                    &ltag,
                    &[n_embd],
                    skip_dense,
                )?;
                let vn = load_f32_vec_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_v_inp_norm.weight"),
                    &ltag,
                    &[n_embd],
                    skip_dense,
                )?;
                let kp = load_dense_mat_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_k_conv_proj.weight"),
                    &ltag,
                    n_embd,
                    n_conv * nh_kv,
                    &mut stats.dense_bytes,
                    skip_dense,
                )?;
                let vp = load_dense_mat_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_v_conv_proj.weight"),
                    &ltag,
                    n_embd,
                    n_conv * nh_kv,
                    &mut stats.dense_bytes,
                    skip_dense,
                )?;
                (
                    Some(qn.clone()),
                    Some(qn),
                    Some(kn),
                    Some(vn),
                    Some(kp),
                    Some(vp),
                )
            };
            let attn_norm = if swa { attn_norm } else { None };

            let attn_res_query = if ip > 0 {
                let dims: &[usize] = if swa { &[n_embd] } else { &[n_embd, 3] };
                Some(load_f32_vec_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.attn_res_query.weight"),
                    &ltag,
                    dims,
                    skip_dense,
                )?)
            } else {
                None
            };
            let ffn_res_query = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_res_query.weight"),
                &ltag,
                &[n_embd],
                skip_dense,
            )?;
            let ffn_norm = load_f32_vec_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_norm.weight"),
                &ltag,
                &[n_embd],
                skip_dense,
            )?;
            let ffn_gate_inp = load_dense_mat_opt(
                gguf,
                direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                &format!("blk.{ip}.ffn_gate_inp.weight"),
                &ltag,
                n_embd,
                n_exp,
                &mut stats.dense_bytes,
                skip_dense,
            )?;
            let exp_probs_b = if gguf
                .tensor_info(&format!("blk.{ip}.exp_probs_b.bias"))
                .is_some()
            {
                Some(load_f32_vec_opt(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.exp_probs_b.bias"),
                    &ltag,
                    &[n_exp],
                    skip_dense,
                )?)
            } else {
                None
            };
            let ffn_gate_up_exps_scale = if gguf
                .tensor_info(&format!("blk.{ip}.ffn_gate_up_exps.scale"))
                .is_some()
            {
                Some(load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.ffn_gate_up_exps.scale"),
                    &ltag,
                    &[n_exp],
                )?)
            } else {
                None
            };
            let ffn_down_exps_scale = if gguf
                .tensor_info(&format!("blk.{ip}.ffn_down_exps.scale"))
                .is_some()
            {
                Some(load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.ffn_down_exps.scale"),
                    &ltag,
                    &[n_exp],
                )?)
            } else {
                None
            };

            let is_core = (cfg.core_start..cfg.core_end).contains(&ip);
            let in_vram = opts.vram_expert_layers.get(ip).copied().unwrap_or(false);
            let experts = if in_vram {
                stats.vram_expert_layers += 1;
                if is_core {
                    stats.core_layers_in_fast_tiers += 1;
                }
                ExpertLayerStore::DeviceVram
            } else if !use_direct {
                let (gu_info, gu_bytes) = gguf.req_tensor(
                    &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                    &ltag,
                    &[n_embd, 2 * n_ff, n_exp],
                )?;
                let (dn_info, dn_bytes) = gguf.req_tensor(
                    &format!("blk.{ip}.ffn_down_exps.weight"),
                    &ltag,
                    &[n_ff, n_embd, n_exp],
                )?;
                let (gu_backing, dn_backing) = if skip_dense {
                    // Hybrid GPU+CPU mode: stage host experts into `2 MiB` `MADV_HUGEPAGE`
                    // anonymous memory (enabling both zero-TLB-miss CPU `AVX2` decode AND zero-copy
                    // `VK_EXT_external_memory_host` PCIe TensorCore prefill!) and immediately
                    // release the `4 KiB` file-backed page cache (`MADV_DONTNEED`).
                    let mut gu_hp = HugePageBytes::new_zeroed(gu_bytes.len());
                    gu_hp.as_mut_slice().copy_from_slice(gu_bytes);
                    release_mmap_page_cache(gu_bytes);
                    let mut dn_hp = HugePageBytes::new_zeroed(dn_bytes.len());
                    dn_hp.as_mut_slice().copy_from_slice(dn_bytes);
                    release_mmap_page_cache(dn_bytes);
                    stats.ram_owned_expert_layers += 1;
                    stats.ram_owned_bytes += (gu_bytes.len() + dn_bytes.len()) as u64;
                    (
                        BytesBacking::HugePage(Arc::new(gu_hp)),
                        BytesBacking::HugePage(Arc::new(dn_hp)),
                    )
                } else {
                    stats.ram_mmap_expert_layers += 1;
                    (
                        BytesBacking::Borrowed(gu_bytes),
                        BytesBacking::Borrowed(dn_bytes),
                    )
                };
                if is_core {
                    stats.core_layers_in_fast_tiers += 1;
                }
                ExpertLayerStore::InMemory {
                    gate_up: gu_backing,
                    gate_up_dtype: gu_info.dtype,
                    gate_up_expert_bytes: gu_bytes.len() / n_exp,
                    down: dn_backing,
                    down_dtype: dn_info.dtype,
                    down_expert_bytes: dn_bytes.len() / n_exp,
                }
            } else if in_ram_plan {
                let rdr = direct_reader.as_ref().expect("direct_reader active");
                let (gu_info, gu_off, gu_len) = gguf.req_tensor_file_range(
                    &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                    &ltag,
                    &[n_embd, 2 * n_ff, n_exp],
                )?;
                let gu_dt = gu_info.dtype;
                let (dn_info, dn_off, dn_len) = gguf.req_tensor_file_range(
                    &format!("blk.{ip}.ffn_down_exps.weight"),
                    &ltag,
                    &[n_ff, n_embd, n_exp],
                )?;
                let dn_dt = dn_info.dtype;
                let (gu_backing, dn_backing) = if skip_dense || gu_len >= 2 * 1024 * 1024 {
                    let mut gu_hp = HugePageBytes::new_zeroed(gu_len);
                    rdr.read_exact_uncached(gu_off, gu_hp.as_mut_slice(), &mut io_slot)?;
                    let mut dn_hp = HugePageBytes::new_zeroed(dn_len);
                    rdr.read_exact_uncached(dn_off, dn_hp.as_mut_slice(), &mut io_slot)?;
                    (
                        BytesBacking::HugePage(Arc::new(gu_hp)),
                        BytesBacking::HugePage(Arc::new(dn_hp)),
                    )
                } else {
                    let mut gu_owned = vec![0u8; gu_len];
                    rdr.read_exact_uncached(gu_off, &mut gu_owned, &mut io_slot)?;
                    let mut dn_owned = vec![0u8; dn_len];
                    rdr.read_exact_uncached(dn_off, &mut dn_owned, &mut io_slot)?;
                    (BytesBacking::Owned(gu_owned), BytesBacking::Owned(dn_owned))
                };
                stats.ram_owned_expert_layers += 1;
                stats.ram_owned_bytes += (gu_len + dn_len) as u64;
                if is_core {
                    stats.core_layers_in_fast_tiers += 1;
                }
                ExpertLayerStore::InMemory {
                    gate_up: gu_backing,
                    gate_up_dtype: gu_dt,
                    gate_up_expert_bytes: gu_len / n_exp,
                    down: dn_backing,
                    down_dtype: dn_dt,
                    down_expert_bytes: dn_len / n_exp,
                }
            } else {
                let (gu_info, gu_off, gu_len) = gguf.req_tensor_file_range(
                    &format!("blk.{ip}.ffn_gate_up_exps.weight"),
                    &ltag,
                    &[n_embd, 2 * n_ff, n_exp],
                )?;
                let gu_dt = gu_info.dtype;
                let (dn_info, dn_off, dn_len) = gguf.req_tensor_file_range(
                    &format!("blk.{ip}.ffn_down_exps.weight"),
                    &ltag,
                    &[n_ff, n_embd, n_exp],
                )?;
                let dn_dt = dn_info.dtype;
                let passes = if is_core { cfg.n_passes } else { 1 };
                let per_exp = (gu_len / n_exp + dn_len / n_exp) as u64;
                stats.nvme_direct_expert_layers += 1;
                stats.nvme_bytes_per_decode_tok +=
                    (passes as u64) * (cfg.n_expert_used as u64) * per_exp;
                ExpertLayerStore::NvmeDirect {
                    gate_up_offset: gu_off,
                    gate_up_dtype: gu_dt,
                    gate_up_expert_bytes: gu_len / n_exp,
                    down_offset: dn_off,
                    down_dtype: dn_dt,
                    down_expert_bytes: dn_len / n_exp,
                }
            };

            layers.push(Inkling2LayerWeights {
                wq,
                wk,
                wv,
                wo,
                wqkv_gate,
                attn_q_clip,
                attn_sink_proj,
                attn_rel,
                attn_rel_bias,
                attn_rel_clip,
                attn_k_conv,
                attn_v_conv,
                attn_post_conv,
                ffn_post_conv,
                attn_norm,
                attn_q_inp_norm,
                attn_k_inp_norm,
                attn_v_inp_norm,
                attn_k_conv_proj,
                attn_v_conv_proj,
                attn_res_query,
                ffn_res_query,
                ffn_norm,
                ffn_gate_inp,
                exp_probs_b,
                ffn_gate_up_exps_scale,
                ffn_down_exps_scale,
                experts,
            });
        }

        // If every layer fit into RAM/VRAM, drop `direct_reader` so no file descriptors are kept open unnecessarily.
        if stats.nvme_direct_expert_layers == 0 {
            direct_reader = None;
        }

        let mut loop_layers = Vec::new();
        if cfg.is_looped() && !skip_dense {
            loop_layers.reserve(cfg.core_end - cfg.core_start);
            let rank = cfg.gate_rank;
            for ip in cfg.core_start..cfg.core_end {
                let ltag = format!("blk.{ip}");
                let msg_query = load_f32_vec(
                    gguf,
                    direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                    &format!("blk.{ip}.loop_msg_query.weight"),
                    &ltag,
                    &[n_embd, LOOP_CTRL_COUNT],
                )?;
                let mut ctrl_down: [Vec<f32>; LOOP_CTRL_COUNT] = Default::default();
                let mut ctrl_gain: [Vec<f32>; LOOP_CTRL_COUNT] = Default::default();
                let mut ctrl_forget: [Vec<f32>; LOOP_CTRL_COUNT] = Default::default();
                for c in 0..LOOP_CTRL_COUNT {
                    let cname = LOOP_CTRL_NAMES[c];
                    ctrl_down[c] = load_f32_vec(
                        gguf,
                        direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_down.weight"),
                        &ltag,
                        &[2 * n_embd, rank],
                    )?;
                    ctrl_gain[c] = load_f32_vec(
                        gguf,
                        direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_gain.weight"),
                        &ltag,
                        &[rank, n_embd],
                    )?;
                    ctrl_forget[c] = load_f32_vec(
                        gguf,
                        direct_reader.as_ref().map(|r| (r, &mut io_slot)),
                        &format!("blk.{ip}.loop_{cname}_ctrl_forget.weight"),
                        &ltag,
                        &[rank, n_embd],
                    )?;
                }
                loop_layers.push(Inkling2LoopWeights {
                    msg_query,
                    ctrl_down,
                    ctrl_gain,
                    ctrl_forget,
                });
            }
        }

        Ok(Self {
            cfg,
            stats,
            token_embd,
            token_embd_norm,
            output_res_query,
            output_norm,
            output,
            layers,
            loop_layers,
            direct_reader,
            max_expert_chunk_bytes,
        })
    }

    #[inline]
    pub fn embed_token(&self, token_id: u32, out_emb: &mut [f32]) {
        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes(),
            self.cfg.n_embd,
            token_id as usize,
            out_emb,
        );
    }

    /// Computes `rms_norm(token_embd[token_id], token_embd_norm, rms_eps)` (`dspark.cpp:990`).
    #[inline]
    pub fn embed_token_normed(&self, token_id: u32, tmp_emb: &mut [f32], out_normed: &mut [f32]) {
        let n_embd = self.cfg.n_embd;
        self.embed_token(token_id, &mut tmp_emb[..n_embd]);
        rms_norm(
            &mut out_normed[..n_embd],
            &tmp_emb[..n_embd],
            &self.token_embd_norm,
            self.cfg.rms_eps,
        );
    }

    /// Projects a hidden vector `[n_embd]` directly through `output.weight` (`[vocab_size]`, unpadded,
    /// without `output_norm` or softcapping — used by `DFlashModel` / `DSpark` Markov head).
    pub fn project_lm_head_raw(
        &self,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        hidden: &[f32],
        out_logits: &mut [f32],
    ) {
        let vp = self.cfg.vocab_padded;
        let vu = self.cfg.vocab_size;
        matvec_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes(),
            hidden,
            &mut state.mv_scratch,
            &mut state.logits_padded[..vp],
            vp,
            self.cfg.n_embd,
        );
        out_logits[..vu].copy_from_slice(&state.logits_padded[..vu]);
    }

    /// Copies the requested `DSpark` target visit taps (`e.g. [5, 23, 41, 48]`) from `state.slot_outputs`
    /// into `out_taps` (`target_layers.len() * n_embd` floats).
    pub fn extract_dspark_taps(
        &self,
        state: &Inkling2State,
        target_layers: &[usize],
        out_taps: &mut [f32],
    ) {
        let n_embd = self.cfg.n_embd;
        assert_eq!(out_taps.len(), target_layers.len() * n_embd);
        for (i, &visit) in target_layers.iter().enumerate() {
            assert!(visit <= self.cfg.n_layers);
            let s = visit * n_embd;
            out_taps[i * n_embd..(i + 1) * n_embd]
                .copy_from_slice(&state.slot_outputs[s..s + n_embd]);
        }
    }

    /// Returns `(gate_up_pin, gate_up_logical_bytes, down_pin, down_logical_bytes)` if physical
    /// layer `ip` holds `Q8_0` experts in `2 MiB`-aligned anonymous `HugePageBytes`, enabling
    /// zero-copy `VK_EXT_external_memory_host` GPU TensorCore prefill without extra RAM/VRAM!
    #[allow(clippy::type_complexity)]
    pub fn host_importable_expert_pins(
        &self,
        ip: usize,
    ) -> Option<(
        Arc<dyn AsRef<[u8]> + Send + Sync>,
        usize,
        Arc<dyn AsRef<[u8]> + Send + Sync>,
        usize,
    )> {
        match self.layers.get(ip)?.experts {
            ExpertLayerStore::InMemory {
                gate_up: BytesBacking::HugePage(ref gu_hp),
                gate_up_dtype: DType::Q8_0,
                down: BytesBacking::HugePage(ref dn_hp),
                down_dtype: DType::Q8_0,
                ..
            } => Some((
                Arc::new(HugePageAlignedPin(Arc::clone(gu_hp))),
                gu_hp.as_slice().len(),
                Arc::new(HugePageAlignedPin(Arc::clone(dn_hp))),
                dn_hp.as_slice().len(),
            )),
            _ => None,
        }
    }

    /// Evaluates selected MoE experts across a window of `window_len` (`1 <= W <= MAX_SPEC_WINDOW`)
    /// tokens simultaneously by **grouping unique expert IDs** (`0..n_expert`), so every requested
    /// NVMe `O_DIRECT` or Host RAM expert is read **at most once** across all `W` tokens while
    /// maintaining `0-ULP` bit-exact equivalence with sequential `eval_moe_experts_f32`!
    #[allow(clippy::too_many_arguments)]
    pub fn eval_moe_experts_window_f32(
        &self,
        ip: usize,
        window_len: usize,
        cur_normed_win: &[f32],
        top_ids_win: &[usize],
        top_weights_win: &[f32],
        pool: &ThreadPool,
        serial_pool: &ThreadPool,
        kt: &KernelTable,
        expert_slots: &mut [ExpertWorkSlot],
        mv_scratch: &mut MatvecScratch,
        win_slot_outs: &mut [f32],
        moe_out_win: &mut [f32],
    ) {
        self.eval_moe_experts_window_inner(
            ip,
            window_len,
            cur_normed_win,
            top_ids_win,
            top_weights_win,
            pool,
            None,
            serial_pool,
            kt,
            expert_slots,
            mv_scratch,
            win_slot_outs,
            moe_out_win,
            true,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_moe_experts_window_inner(
        &self,
        ip: usize,
        window_len: usize,
        cur_normed_win: &[f32],
        top_ids_win: &[usize],
        top_weights_win: &[f32],
        pool: &ThreadPool,
        io_pool: Option<&ThreadPool>,
        serial_pool: &ThreadPool,
        kt: &KernelTable,
        expert_slots: &mut [ExpertWorkSlot],
        mv_scratch: &mut MatvecScratch,
        win_slot_outs: &mut [f32],
        moe_out_win: &mut [f32],
        float_act: bool,
    ) {
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        assert!((1..=MAX_INKLING2_WINDOW).contains(&window_len));
        let top_k = top_ids_win.len() / window_len;
        assert!(top_k <= self.cfg.n_expert_used);
        let beta = self.cfg.situ_beta;
        let lb = self.cfg.situ_linear_beta;
        assert_eq!(cur_normed_win.len(), window_len * n_embd);
        assert_eq!(top_ids_win.len(), window_len * top_k);
        assert_eq!(top_weights_win.len(), window_len * top_k);
        assert_eq!(moe_out_win.len(), window_len * n_embd);

        if window_len == 1 {
            self.eval_moe_experts_inner(
                ip,
                cur_normed_win,
                top_ids_win,
                top_weights_win,
                pool,
                io_pool,
                serial_pool,
                kt,
                expert_slots,
                mv_scratch,
                moe_out_win,
                float_act,
            );
            return;
        }

        // Build deduplicated expert table (`n_uniq <= n_expert`, 256-bit token mask per unique expert)
        // with zero heap allocations (`AGENTS.md` Rule #4).
        const MAX_UNIQ_EXPS: usize = 512;
        let n_expert = self.cfg.n_expert;
        assert!(n_expert <= MAX_UNIQ_EXPS);
        let mut uniq_exps = [0usize; MAX_UNIQ_EXPS];
        let mut uniq_mask = [[0u64; 4]; MAX_UNIQ_EXPS];
        let mut exp_to_u = [u16::MAX; MAX_UNIQ_EXPS];
        let mut uniq_kslot = [[0u8; MAX_INKLING2_WINDOW]; MAX_UNIQ_EXPS];
        let mut n_uniq = 0usize;

        for tok_i in 0..window_len {
            let word = tok_i >> 6;
            let bit = 1u64 << (tok_i & 63);
            for k_idx in 0..top_k {
                if top_weights_win[tok_i * top_k + k_idx] == 0.0 {
                    continue;
                }
                let e_id = top_ids_win[tok_i * top_k + k_idx];
                let u = if exp_to_u[e_id] != u16::MAX {
                    exp_to_u[e_id] as usize
                } else {
                    let u = n_uniq;
                    exp_to_u[e_id] = u as u16;
                    uniq_exps[u] = e_id;
                    n_uniq += 1;
                    u
                };
                uniq_mask[u][word] |= bit;
                uniq_kslot[u][tok_i] = k_idx as u8;
            }
        }

        let gu_scales = self.layers[ip].ffn_gate_up_exps_scale.as_deref();
        let dn_scales = self.layers[ip].ffn_down_exps_scale.as_deref();
        // Longest-Processing-Time-First (LPT) sort by descending token count `m_cnt` so parallel
        // expert waves have homogeneous batch sizes (eliminates Zipf hot-expert barrier stragglers).
        let mut uniq_mcnt = [0usize; MAX_UNIQ_EXPS];
        let mut order = [0usize; MAX_UNIQ_EXPS];
        for u in 0..n_uniq {
            uniq_mcnt[u] = (uniq_mask[u][0].count_ones()
                + uniq_mask[u][1].count_ones()
                + uniq_mask[u][2].count_ones()
                + uniq_mask[u][3].count_ones()) as usize;
            order[u] = u;
        }
        order[..n_uniq]
            .sort_unstable_by(|&a, &b| uniq_mcnt[b].cmp(&uniq_mcnt[a]).then_with(|| a.cmp(&b)));

        match &self.layers[ip].experts {
            ExpertLayerStore::InMemory {
                gate_up,
                gate_up_dtype,
                gate_up_expert_bytes,
                down,
                down_dtype,
                down_expert_bytes,
            } => {
                let gu_all = gate_up.as_slice();
                let dn_all = down.as_slice();
                let n_workers = expert_slots.len().max(1);
                let mut base_ord = 0usize;

                // Hot experts (`m_cnt >= 12` on multi-threaded pool): split `gate_up` & `down` rows
                // across ALL worker threads so popular experts never bottleneck a single core.
                while base_ord < n_uniq
                    && pool.num_threads() > 1
                    && uniq_mcnt[order[base_ord]] >= 12
                {
                    let u = order[base_ord];
                    let e_id = uniq_exps[u];
                    let mask = uniq_mask[u];
                    let gu_slice =
                        &gu_all[e_id * gate_up_expert_bytes..(e_id + 1) * gate_up_expert_bytes];
                    let dn_slice =
                        &dn_all[e_id * down_expert_bytes..(e_id + 1) * down_expert_bytes];
                    let ws = &mut expert_slots[0];

                    let mut m_cnt = 0usize;
                    for tok_i in 0..window_len {
                        if (mask[tok_i >> 6] & (1u64 << (tok_i & 63))) != 0 {
                            ws.win_in_buf[m_cnt * n_embd..(m_cnt + 1) * n_embd].copy_from_slice(
                                &cur_normed_win[tok_i * n_embd..(tok_i + 1) * n_embd],
                            );
                            m_cnt += 1;
                        }
                    }
                    if float_act {
                        matvec_batch_f32_unquant_act(
                            pool,
                            kt,
                            *gate_up_dtype,
                            gu_slice,
                            &ws.win_in_buf[..m_cnt * n_embd],
                            m_cnt,
                            &mut ws.mv_scratch,
                            &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff],
                            2 * n_ff,
                            n_embd,
                        );
                    } else {
                        matmul_f32(
                            pool,
                            kt,
                            *gate_up_dtype,
                            gu_slice,
                            &ws.win_in_buf[..m_cnt * n_embd],
                            &mut ws.mv_scratch,
                            &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff],
                            2 * n_ff,
                            n_embd,
                            m_cnt,
                        );
                    }
                    if let Some(sc) = gu_scales {
                        let s_e = sc[e_id];
                        for v in &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff] {
                            *v *= s_e;
                        }
                    }
                    for m in 0..m_cnt {
                        let gu_m = &ws.win_gate_up_buf[m * 2 * n_ff..(m + 1) * 2 * n_ff];
                        apply_situ_glu(
                            &gu_m[..n_ff],
                            &gu_m[n_ff..2 * n_ff],
                            &mut ws.win_hidden_buf[m * n_ff..(m + 1) * n_ff],
                            beta,
                            lb,
                        );
                    }
                    if float_act {
                        matvec_batch_f32_unquant_act(
                            pool,
                            kt,
                            *down_dtype,
                            dn_slice,
                            &ws.win_hidden_buf[..m_cnt * n_ff],
                            m_cnt,
                            &mut ws.mv_scratch,
                            &mut ws.out_buf[..m_cnt * n_embd],
                            n_embd,
                            n_ff,
                        );
                    } else {
                        matmul_f32(
                            pool,
                            kt,
                            *down_dtype,
                            dn_slice,
                            &ws.win_hidden_buf[..m_cnt * n_ff],
                            &mut ws.mv_scratch,
                            &mut ws.out_buf[..m_cnt * n_embd],
                            n_embd,
                            n_ff,
                            m_cnt,
                        );
                    }
                    if let Some(sc) = dn_scales {
                        let s_e = sc[e_id];
                        for v in &mut ws.out_buf[..m_cnt * n_embd] {
                            *v *= s_e;
                        }
                    }
                    let kslots = &uniq_kslot[u];
                    let mut m_idx = 0usize;
                    for (tok_i, &ks) in kslots.iter().enumerate().take(window_len) {
                        if (mask[tok_i >> 6] & (1u64 << (tok_i & 63))) == 0 {
                            continue;
                        }
                        let k_idx = ks as usize;
                        let dst_off = (tok_i * top_k + k_idx) * n_embd;
                        win_slot_outs[dst_off..dst_off + n_embd]
                            .copy_from_slice(&ws.out_buf[m_idx * n_embd..(m_idx + 1) * n_embd]);
                        m_idx += 1;
                    }
                    base_ord += 1;
                }

                // Remaining experts in LPT order: homogeneous `m_cnt` within each parallel wave.
                while base_ord < n_uniq {
                    let batch_u = (n_uniq - base_ord).min(n_workers);
                    pool.parallel_for_each_mut(&mut expert_slots[..batch_u], |rel_u, ws| {
                        let u = order[base_ord + rel_u];
                        let e_id = uniq_exps[u];
                        let mask = uniq_mask[u];
                        let gu_slice =
                            &gu_all[e_id * gate_up_expert_bytes..(e_id + 1) * gate_up_expert_bytes];
                        let dn_slice =
                            &dn_all[e_id * down_expert_bytes..(e_id + 1) * down_expert_bytes];

                        let mut m_cnt = 0usize;
                        for tok_i in 0..window_len {
                            if (mask[tok_i >> 6] & (1u64 << (tok_i & 63))) != 0 {
                                ws.win_in_buf[m_cnt * n_embd..(m_cnt + 1) * n_embd]
                                    .copy_from_slice(
                                        &cur_normed_win[tok_i * n_embd..(tok_i + 1) * n_embd],
                                    );
                                m_cnt += 1;
                            }
                        }
                        if float_act {
                            matvec_batch_f32_unquant_act(
                                serial_pool,
                                kt,
                                *gate_up_dtype,
                                gu_slice,
                                &ws.win_in_buf[..m_cnt * n_embd],
                                m_cnt,
                                &mut ws.mv_scratch,
                                &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff],
                                2 * n_ff,
                                n_embd,
                            );
                        } else {
                            matmul_f32(
                                serial_pool,
                                kt,
                                *gate_up_dtype,
                                gu_slice,
                                &ws.win_in_buf[..m_cnt * n_embd],
                                &mut ws.mv_scratch,
                                &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff],
                                2 * n_ff,
                                n_embd,
                                m_cnt,
                            );
                        }
                        if let Some(sc) = gu_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff] {
                                *v *= s_e;
                            }
                        }
                        for m in 0..m_cnt {
                            let gu_m = &ws.win_gate_up_buf[m * 2 * n_ff..(m + 1) * 2 * n_ff];
                            apply_situ_glu(
                                &gu_m[..n_ff],
                                &gu_m[n_ff..2 * n_ff],
                                &mut ws.win_hidden_buf[m * n_ff..(m + 1) * n_ff],
                                beta,
                                lb,
                            );
                        }
                        if float_act {
                            matvec_batch_f32_unquant_act(
                                serial_pool,
                                kt,
                                *down_dtype,
                                dn_slice,
                                &ws.win_hidden_buf[..m_cnt * n_ff],
                                m_cnt,
                                &mut ws.mv_scratch,
                                &mut ws.out_buf[..m_cnt * n_embd],
                                n_embd,
                                n_ff,
                            );
                        } else {
                            matmul_f32(
                                serial_pool,
                                kt,
                                *down_dtype,
                                dn_slice,
                                &ws.win_hidden_buf[..m_cnt * n_ff],
                                &mut ws.mv_scratch,
                                &mut ws.out_buf[..m_cnt * n_embd],
                                n_embd,
                                n_ff,
                                m_cnt,
                            );
                        }
                        if let Some(sc) = dn_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.out_buf[..m_cnt * n_embd] {
                                *v *= s_e;
                            }
                        }
                    });
                    for (rel_u, ws) in expert_slots.iter().enumerate().take(batch_u) {
                        let u = order[base_ord + rel_u];
                        let mask = uniq_mask[u];
                        let kslots = &uniq_kslot[u];
                        let mut m_idx = 0usize;
                        for (tok_i, &ks) in kslots.iter().enumerate().take(window_len) {
                            if (mask[tok_i >> 6] & (1u64 << (tok_i & 63))) == 0 {
                                continue;
                            }
                            let k_idx = ks as usize;
                            let dst_off = (tok_i * top_k + k_idx) * n_embd;
                            win_slot_outs[dst_off..dst_off + n_embd]
                                .copy_from_slice(&ws.out_buf[m_idx * n_embd..(m_idx + 1) * n_embd]);
                            m_idx += 1;
                        }
                    }
                    base_ord += batch_u;
                }
            }
            ExpertLayerStore::NvmeDirect {
                gate_up_offset,
                gate_up_dtype,
                gate_up_expert_bytes,
                down_offset,
                down_dtype,
                down_expert_bytes,
            } => {
                let rdr = self
                    .direct_reader
                    .as_ref()
                    .expect("DirectTensorReader required for NvmeDirect expert layer");
                let nvme_pool = match io_pool {
                    Some(ip) if ip.num_threads() > pool.num_threads() => ip,
                    _ => pool,
                };
                let n_workers = expert_slots.len().max(1);
                let mut base_ord = 0usize;
                while base_ord < n_uniq {
                    let batch_u = (n_uniq - base_ord).min(n_workers);
                    nvme_pool.parallel_for_each_mut(&mut expert_slots[..batch_u], |rel_u, ws| {
                        let u = order[base_ord + rel_u];
                        let e_id = uniq_exps[u];
                        let mask = uniq_mask[u];
                        let mut m_cnt = 0usize;
                        for tok_i in 0..window_len {
                            if (mask[tok_i >> 6] & (1u64 << (tok_i & 63))) != 0 {
                                ws.win_in_buf[m_cnt * n_embd..(m_cnt + 1) * n_embd]
                                    .copy_from_slice(
                                        &cur_normed_win[tok_i * n_embd..(tok_i + 1) * n_embd],
                                    );
                                m_cnt += 1;
                            }
                        }
                        let gu_off = *gate_up_offset + (e_id * *gate_up_expert_bytes) as u64;
                        let gu_bytes = rdr
                            .read_into_slot(gu_off, *gate_up_expert_bytes, &mut ws.slot)
                            .expect("O_DIRECT gate_up expert read failed");
                        if float_act {
                            matvec_batch_f32_unquant_act(
                                serial_pool,
                                kt,
                                *gate_up_dtype,
                                gu_bytes,
                                &ws.win_in_buf[..m_cnt * n_embd],
                                m_cnt,
                                &mut ws.mv_scratch,
                                &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff],
                                2 * n_ff,
                                n_embd,
                            );
                        } else {
                            matmul_f32(
                                serial_pool,
                                kt,
                                *gate_up_dtype,
                                gu_bytes,
                                &ws.win_in_buf[..m_cnt * n_embd],
                                &mut ws.mv_scratch,
                                &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff],
                                2 * n_ff,
                                n_embd,
                                m_cnt,
                            );
                        }
                        if let Some(sc) = gu_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.win_gate_up_buf[..m_cnt * 2 * n_ff] {
                                *v *= s_e;
                            }
                        }
                        for m in 0..m_cnt {
                            let gu_m = &ws.win_gate_up_buf[m * 2 * n_ff..(m + 1) * 2 * n_ff];
                            apply_situ_glu(
                                &gu_m[..n_ff],
                                &gu_m[n_ff..2 * n_ff],
                                &mut ws.win_hidden_buf[m * n_ff..(m + 1) * n_ff],
                                beta,
                                lb,
                            );
                        }
                        let dn_off = *down_offset + (e_id * *down_expert_bytes) as u64;
                        let dn_bytes = rdr
                            .read_into_slot(dn_off, *down_expert_bytes, &mut ws.slot)
                            .expect("O_DIRECT down expert read failed");
                        if float_act {
                            matvec_batch_f32_unquant_act(
                                serial_pool,
                                kt,
                                *down_dtype,
                                dn_bytes,
                                &ws.win_hidden_buf[..m_cnt * n_ff],
                                m_cnt,
                                &mut ws.mv_scratch,
                                &mut ws.out_buf[..m_cnt * n_embd],
                                n_embd,
                                n_ff,
                            );
                        } else {
                            matmul_f32(
                                serial_pool,
                                kt,
                                *down_dtype,
                                dn_bytes,
                                &ws.win_hidden_buf[..m_cnt * n_ff],
                                &mut ws.mv_scratch,
                                &mut ws.out_buf[..m_cnt * n_embd],
                                n_embd,
                                n_ff,
                                m_cnt,
                            );
                        }
                        if let Some(sc) = dn_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.out_buf[..m_cnt * n_embd] {
                                *v *= s_e;
                            }
                        }
                    });
                    for (rel_u, ws) in expert_slots.iter().enumerate().take(batch_u) {
                        let u = order[base_ord + rel_u];
                        let mask = uniq_mask[u];
                        let kslots = &uniq_kslot[u];
                        let mut m_idx = 0usize;
                        for (tok_i, &ks) in kslots.iter().enumerate().take(window_len) {
                            if (mask[tok_i >> 6] & (1u64 << (tok_i & 63))) == 0 {
                                continue;
                            }
                            let k_idx = ks as usize;
                            let dst_off = (tok_i * top_k + k_idx) * n_embd;
                            win_slot_outs[dst_off..dst_off + n_embd]
                                .copy_from_slice(&ws.out_buf[m_idx * n_embd..(m_idx + 1) * n_embd]);
                            m_idx += 1;
                        }
                    }
                    base_ord += batch_u;
                }
            }
            ExpertLayerStore::DeviceVram => {
                panic!(
                    "Physical layer {ip} experts reside in GPU VRAM; host fallback cannot run DeviceVram layer"
                );
            }
        }

        // Deterministic `0..top_k` reduction per token (`0-ULP` identical to single-token!)
        moe_out_win.fill(0.0);
        for tok_i in 0..window_len {
            let out_t = &mut moe_out_win[tok_i * n_embd..(tok_i + 1) * n_embd];
            for k_idx in 0..top_k {
                let w_e = top_weights_win[tok_i * top_k + k_idx];
                if w_e == 0.0 {
                    continue;
                }
                let slot_o = &win_slot_outs
                    [(tok_i * top_k + k_idx) * n_embd..(tok_i * top_k + k_idx + 1) * n_embd];
                for (m, &o) in out_t.iter_mut().zip(slot_o.iter()) {
                    *m += w_e * o;
                }
            }
        }
    }

    /// Evaluates selected MoE experts `(top_ids, top_weights)` on a single normed input vector `cur_normed`
    /// (`[n_embd]`) for physical layer `ip` using `Q8_0 × Q8_0` CPU kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_moe_experts_single(
        &self,
        ip: usize,
        cur_normed: &[f32],
        top_ids: &[usize],
        top_weights: &[f32],
        pool: &ThreadPool,
        serial_pool: &ThreadPool,
        kt: &KernelTable,
        expert_slots: &mut [ExpertWorkSlot],
        mv_scratch: &mut MatvecScratch,
        moe_out: &mut [f32],
    ) {
        self.eval_moe_experts_inner(
            ip,
            cur_normed,
            top_ids,
            top_weights,
            pool,
            None,
            serial_pool,
            kt,
            expert_slots,
            mv_scratch,
            moe_out,
            false,
        );
    }

    /// Evaluates selected MoE experts `(top_ids, top_weights)` using exact `Q8_0 × F32` float
    /// activations (`kt.dot_f32_q8_0`), matching GPU `moe_gemv_q8_0_id` shaders across 3-tier
    /// `VRAM + Host RAM + NVMe O_DIRECT` hybrid inference with zero activation-quantization loss.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_moe_experts_f32(
        &self,
        ip: usize,
        cur_normed: &[f32],
        top_ids: &[usize],
        top_weights: &[f32],
        pool: &ThreadPool,
        serial_pool: &ThreadPool,
        kt: &KernelTable,
        expert_slots: &mut [ExpertWorkSlot],
        mv_scratch: &mut MatvecScratch,
        moe_out: &mut [f32],
    ) {
        self.eval_moe_experts_inner(
            ip,
            cur_normed,
            top_ids,
            top_weights,
            pool,
            None,
            serial_pool,
            kt,
            expert_slots,
            mv_scratch,
            moe_out,
            true,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_moe_experts_inner(
        &self,
        ip: usize,
        cur_normed: &[f32],
        top_ids: &[usize],
        top_weights: &[f32],
        pool: &ThreadPool,
        io_pool: Option<&ThreadPool>,
        serial_pool: &ThreadPool,
        kt: &KernelTable,
        expert_slots: &mut [ExpertWorkSlot],
        mv_scratch: &mut MatvecScratch,
        moe_out: &mut [f32],
        float_act: bool,
    ) {
        let mv_fn = if float_act {
            matvec_f32_unquant_act
        } else {
            matvec_f32
        };
        let n_embd = self.cfg.n_embd;
        let n_ff = self.cfg.n_ff_exp;
        let top_k = top_ids.len();
        let beta = self.cfg.situ_beta;
        let lb = self.cfg.situ_linear_beta;
        let gu_scales = self.layers[ip].ffn_gate_up_exps_scale.as_deref();
        let dn_scales = self.layers[ip].ffn_down_exps_scale.as_deref();
        moe_out.fill(0.0);

        match &self.layers[ip].experts {
            ExpertLayerStore::InMemory {
                gate_up,
                gate_up_dtype,
                gate_up_expert_bytes,
                down,
                down_dtype,
                down_expert_bytes,
            } => {
                let gu_all = gate_up.as_slice();
                let dn_all = down.as_slice();
                if expert_slots.len() >= top_k && pool.num_threads() > 1 {
                    pool.parallel_for_each_mut(&mut expert_slots[..top_k], |j, ws| {
                        let e_id = top_ids[j];
                        let gu_slice =
                            &gu_all[e_id * gate_up_expert_bytes..(e_id + 1) * gate_up_expert_bytes];
                        mv_fn(
                            serial_pool,
                            kt,
                            *gate_up_dtype,
                            gu_slice,
                            cur_normed,
                            &mut ws.mv_scratch,
                            &mut ws.gate_up_buf[..2 * n_ff],
                            2 * n_ff,
                            n_embd,
                        );
                        if let Some(sc) = gu_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.gate_up_buf[..2 * n_ff] {
                                *v *= s_e;
                            }
                        }
                        apply_situ_glu(
                            &ws.gate_up_buf[..n_ff],
                            &ws.gate_up_buf[n_ff..2 * n_ff],
                            &mut ws.hidden_buf[..n_ff],
                            beta,
                            lb,
                        );
                        let dn_slice =
                            &dn_all[e_id * down_expert_bytes..(e_id + 1) * down_expert_bytes];
                        mv_fn(
                            serial_pool,
                            kt,
                            *down_dtype,
                            dn_slice,
                            &ws.hidden_buf[..n_ff],
                            &mut ws.mv_scratch,
                            &mut ws.out_buf[..n_embd],
                            n_embd,
                            n_ff,
                        );
                        if let Some(sc) = dn_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.out_buf[..n_embd] {
                                *v *= s_e;
                            }
                        }
                    });
                    for (j, &w_e) in top_weights.iter().enumerate().take(top_k) {
                        let out_j = &expert_slots[j].out_buf[..n_embd];
                        for (m, &o) in moe_out[..n_embd].iter_mut().zip(out_j) {
                            *m += w_e * o;
                        }
                    }
                } else {
                    let ws = &mut expert_slots[0];
                    for j in 0..top_k {
                        let e_id = top_ids[j];
                        let w_e = top_weights[j];
                        let gu_slice =
                            &gu_all[e_id * gate_up_expert_bytes..(e_id + 1) * gate_up_expert_bytes];
                        mv_fn(
                            pool,
                            kt,
                            *gate_up_dtype,
                            gu_slice,
                            cur_normed,
                            mv_scratch,
                            &mut ws.gate_up_buf[..2 * n_ff],
                            2 * n_ff,
                            n_embd,
                        );
                        if let Some(sc) = gu_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.gate_up_buf[..2 * n_ff] {
                                *v *= s_e;
                            }
                        }
                        apply_situ_glu(
                            &ws.gate_up_buf[..n_ff],
                            &ws.gate_up_buf[n_ff..2 * n_ff],
                            &mut ws.hidden_buf[..n_ff],
                            beta,
                            lb,
                        );
                        let dn_slice =
                            &dn_all[e_id * down_expert_bytes..(e_id + 1) * down_expert_bytes];
                        mv_fn(
                            pool,
                            kt,
                            *down_dtype,
                            dn_slice,
                            &ws.hidden_buf[..n_ff],
                            mv_scratch,
                            &mut ws.out_buf[..n_embd],
                            n_embd,
                            n_ff,
                        );
                        if let Some(sc) = dn_scales {
                            let s_e = sc[e_id];
                            for v in &mut ws.out_buf[..n_embd] {
                                *v *= s_e;
                            }
                        }
                        for (m, &o) in moe_out[..n_embd].iter_mut().zip(&ws.out_buf[..n_embd]) {
                            *m += w_e * o;
                        }
                    }
                }
            }
            ExpertLayerStore::NvmeDirect {
                gate_up_offset,
                gate_up_dtype,
                gate_up_expert_bytes,
                down_offset,
                down_dtype,
                down_expert_bytes,
            } => {
                let rdr = self
                    .direct_reader
                    .as_ref()
                    .expect("DirectTensorReader required for NvmeDirect expert layer");
                // Parallelize across all `top_k` active experts simultaneously so NVMe sees
                // queue-depth `top_k` (`8` concurrent `O_DIRECT`/`F_NOCACHE` `pread` requests).
                let nvme_pool = match io_pool {
                    Some(ip) if ip.num_threads() > pool.num_threads() => ip,
                    _ => pool,
                };
                nvme_pool.parallel_for_each_mut(&mut expert_slots[..top_k], |j, ws| {
                    let e_id = top_ids[j];
                    let gu_off = *gate_up_offset + (e_id * *gate_up_expert_bytes) as u64;
                    let gu_bytes = rdr
                        .read_into_slot(gu_off, *gate_up_expert_bytes, &mut ws.slot)
                        .expect("O_DIRECT gate_up expert read failed");
                    mv_fn(
                        serial_pool,
                        kt,
                        *gate_up_dtype,
                        gu_bytes,
                        cur_normed,
                        &mut ws.mv_scratch,
                        &mut ws.gate_up_buf[..2 * n_ff],
                        2 * n_ff,
                        n_embd,
                    );
                    if let Some(sc) = gu_scales {
                        let s_e = sc[e_id];
                        for v in &mut ws.gate_up_buf[..2 * n_ff] {
                            *v *= s_e;
                        }
                    }
                    apply_situ_glu(
                        &ws.gate_up_buf[..n_ff],
                        &ws.gate_up_buf[n_ff..2 * n_ff],
                        &mut ws.hidden_buf[..n_ff],
                        beta,
                        lb,
                    );
                    let dn_off = *down_offset + (e_id * *down_expert_bytes) as u64;
                    let dn_bytes = rdr
                        .read_into_slot(dn_off, *down_expert_bytes, &mut ws.slot)
                        .expect("O_DIRECT down expert read failed");
                    mv_fn(
                        serial_pool,
                        kt,
                        *down_dtype,
                        dn_bytes,
                        &ws.hidden_buf[..n_ff],
                        &mut ws.mv_scratch,
                        &mut ws.out_buf[..n_embd],
                        n_embd,
                        n_ff,
                    );
                    if let Some(sc) = dn_scales {
                        let s_e = sc[e_id];
                        for v in &mut ws.out_buf[..n_embd] {
                            *v *= s_e;
                        }
                    }
                });
                // Deterministic reduction in fixed `0..top_k` order (`AGENTS.md` §9).
                for j in 0..top_k {
                    let w_e = top_weights[j];
                    let out_j = &expert_slots[j].out_buf[..n_embd];
                    for (m, &o) in moe_out[..n_embd].iter_mut().zip(out_j) {
                        *m += w_e * o;
                    }
                }
            }
            ExpertLayerStore::DeviceVram => {
                panic!(
                    "Physical layer {ip} experts reside in GPU VRAM; host fallback cannot run DeviceVram layer"
                );
            }
        }
    }

    /// Computes MoE router selection `(top_ids, top_weights)` from `cur_normed` (`[n_embd]`) for physical layer `ip`.
    #[allow(clippy::too_many_arguments)]
    pub fn route_moe_single(
        &self,
        ip: usize,
        cur_normed: &[f32],
        pool: &ThreadPool,
        kt: &KernelTable,
        mv_scratch: &mut MatvecScratch,
        router_logits: &mut [f32],
        router_probs: &mut [f32],
        router_sel: &mut [f32],
        top_ids: &mut [usize],
        top_weights: &mut [f32],
    ) {
        let n_exp = self.cfg.n_expert;
        let top_k = self.cfg.n_expert_used;
        let layer = &self.layers[ip];
        matvec_f32(
            pool,
            kt,
            layer.ffn_gate_inp.dtype,
            layer.ffn_gate_inp.bytes(),
            cur_normed,
            mv_scratch,
            &mut router_logits[..n_exp],
            n_exp,
            self.cfg.n_embd,
        );
        for e in 0..n_exp {
            let p = 1.0 / (1.0 + (-router_logits[e]).exp());
            router_probs[e] = p;
            router_sel[e] = p + layer.exp_probs_b.as_ref().map_or(0.0, |b| b[e]);
        }

        // Exact stable top-k selection (`ggml_argsort_top_k`: descending score, ties broken by smaller index).
        let mut used = [false; 512];
        debug_assert!(n_exp <= used.len());
        let mut sum_w = 0.0f32;
        for slot in 0..top_k {
            let mut best_e = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for e in 0..n_exp {
                if !used[e] && router_sel[e] > best_v {
                    best_v = router_sel[e];
                    best_e = e;
                }
            }
            used[best_e] = true;
            top_ids[slot] = best_e;
            top_weights[slot] = router_probs[best_e];
            sum_w += router_probs[best_e];
        }
        let inv_sum = self.cfg.expert_weights_scale / (sum_w + 1e-20);
        for w in &mut top_weights[..top_k] {
            *w *= inv_sum;
        }
    }

    pub fn forward_token(
        &self,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), ModelError> {
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }

        let n_embd = self.cfg.n_embd;
        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes(),
            n_embd,
            token_id as usize,
            &mut state.inp_sa,
        );
        rms_norm(
            &mut state.cur,
            &state.inp_sa,
            &self.token_embd_norm,
            self.cfg.rms_eps,
        );

        let ratio = ((pos + 1) as f32) / (self.cfg.log_scale_floor as f32);
        let tau = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();

        state.blocks_len = 0;
        let mut visit = 0usize;
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                    .copy_from_slice(&state.cur);
                visit += 1;
                self.run_layer_single(il, il, pos, tau, false, state, pool, kt);
            }
        } else {
            let cs = self.cfg.core_start;
            let ce = self.cfg.core_end;
            for il in 0..cs {
                state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                    .copy_from_slice(&state.cur);
                visit += 1;
                self.run_layer_single(il, il, pos, tau, false, state, pool, kt);
            }

            state.prelude.copy_from_slice(&state.cur);
            state.prelude_blocks_len = state.blocks_len;
            let pb_floats = state.prelude_blocks_len * n_embd;
            state.prelude_blocks_buf[..pb_floats].copy_from_slice(&state.blocks_buf[..pb_floats]);

            for pass in 0..self.cfg.n_passes {
                if pass > 0 {
                    let bank_start_blk = state.prelude_blocks_len + 1;
                    state.bank_len = state.blocks_len.saturating_sub(bank_start_blk);
                    let b_floats = state.bank_len * n_embd;
                    if b_floats > 0 {
                        let s0 = bank_start_blk * n_embd;
                        state.bank_buf[..b_floats]
                            .copy_from_slice(&state.blocks_buf[s0..s0 + b_floats]);
                    }
                    state.bank_out.copy_from_slice(&state.cur);
                    state.cur.copy_from_slice(&state.prelude);
                    state.blocks_len = state.prelude_blocks_len;
                    state.blocks_buf[..pb_floats]
                        .copy_from_slice(&state.prelude_blocks_buf[..pb_floats]);
                }

                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                        .copy_from_slice(&state.cur);
                    visit += 1;
                    if pass == 0 {
                        self.run_layer_single(il, ip, pos, tau, false, state, pool, kt);
                    } else {
                        let loop_w = &self.loop_layers[ip - cs];
                        mix_depth_into(
                            &state.bank_buf[..state.bank_len * n_embd],
                            state.bank_len,
                            &state.bank_out,
                            &loop_w.msg_query,
                            LOOP_CTRL_COUNT,
                            n_embd,
                            &mut state.mix_states_hat,
                            &mut state.mix_scores,
                            &mut state.messages,
                        );
                        self.run_layer_single(il, ip, pos, tau, true, state, pool, kt);
                    }
                }
            }

            for il in ce..self.cfg.n_layers_phys {
                state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                    .copy_from_slice(&state.cur);
                visit += 1;
                self.run_layer_single(il, il, pos, tau, false, state, pool, kt);
            }
        }
        debug_assert_eq!(visit, self.cfg.n_layers);

        // Output depth-attention mix + output RMSNorm (`inkling2.cpp:421-430`)
        mix_depth_into(
            &state.blocks_buf[..state.blocks_len * n_embd],
            state.blocks_len,
            &state.cur,
            &self.output_res_query,
            1,
            n_embd,
            &mut state.mix_states_hat,
            &mut state.mix_scores,
            &mut state.inp_sa,
        );
        rms_norm(
            &mut state.last_hidden,
            &state.inp_sa,
            &self.output_norm,
            self.cfg.rms_eps,
        );
        let final_slot_off = self.cfg.n_layers * n_embd;
        state.slot_outputs[final_slot_off..final_slot_off + n_embd]
            .copy_from_slice(&state.last_hidden);

        if compute_logits {
            let vp = self.cfg.vocab_padded;
            let vu = self.cfg.vocab_size;
            matvec_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes(),
                &state.last_hidden,
                &mut state.mv_scratch,
                &mut state.logits_padded[..vp],
                vp,
                n_embd,
            );
            state.logits.copy_from_slice(&state.logits_padded[..vu]);
            if let Some(cap) = self.cfg.final_logit_softcapping {
                softcap_inplace_parallel(pool, &mut state.logits, cap);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_single(
        &self,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        has_messages: bool,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
    ) {
        let n_embd = self.cfg.n_embd;
        let swa = self.cfg.is_swa[ip];
        let nh = self.cfg.n_head(ip);
        let nh_kv = self.cfg.n_head_kv(ip);
        let hd = self.cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nh_kv * hd;
        let n_rel = self.cfg.n_relative;
        let n_conv = self.cfg.shortconv_l_cache;
        let extent = self.cfg.rel_extent_for(ip);
        let n_slots = self.cfg.n_layers;
        let layer = &self.layers[ip];

        state.inp_sa.copy_from_slice(&state.cur);

        // 1. Depth-residual mix for attention inputs (`inputs[0..3]`)
        if let Some(ref rq) = layer.attn_res_query {
            if swa {
                mix_depth_into(
                    &state.blocks_buf[..state.blocks_len * n_embd],
                    state.blocks_len,
                    &state.cur,
                    rq,
                    1,
                    n_embd,
                    &mut state.mix_states_hat,
                    &mut state.mix_scores,
                    &mut state.inputs[..n_embd],
                );
                let (first, rest) = state.inputs.split_at_mut(n_embd);
                rest[..n_embd].copy_from_slice(first);
                rest[n_embd..2 * n_embd].copy_from_slice(first);
            } else {
                mix_depth_into(
                    &state.blocks_buf[..state.blocks_len * n_embd],
                    state.blocks_len,
                    &state.cur,
                    rq,
                    3,
                    n_embd,
                    &mut state.mix_states_hat,
                    &mut state.mix_scores,
                    &mut state.inputs[..3 * n_embd],
                );
            }
        } else {
            for j in 0..3 {
                state.inputs[j * n_embd..(j + 1) * n_embd].copy_from_slice(&state.cur);
            }
        }

        // 2. Cross-pass loop controller (`Q, K, V`) + input RMSNorm (`inkling2.cpp:306-324`)
        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            for j in 0..3 {
                control_inplace(
                    &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                    &state.messages[j * n_embd..(j + 1) * n_embd],
                    loop_w,
                    j,
                    n_embd,
                    self.cfg.gate_rank,
                    &mut state.ctrl_in,
                    &mut state.ctrl_g,
                    &mut state.ctrl_gain,
                    &mut state.ctrl_forget,
                );
            }
        }

        if swa && has_messages {
            let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
            for j in 0..3 {
                state.conv_tmp[..n_embd]
                    .copy_from_slice(&state.inputs[j * n_embd..(j + 1) * n_embd]);
                rms_norm(
                    &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                    &state.conv_tmp[..n_embd],
                    an,
                    self.cfg.rms_eps,
                );
            }
        } else if swa {
            let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
            state.conv_tmp[..n_embd].copy_from_slice(&state.inputs[..n_embd]);
            rms_norm(
                &mut state.inputs[..n_embd],
                &state.conv_tmp[..n_embd],
                an,
                self.cfg.rms_eps,
            );
            let (first, rest) = state.inputs.split_at_mut(n_embd);
            rest[..n_embd].copy_from_slice(first);
            rest[n_embd..2 * n_embd].copy_from_slice(first);
        } else {
            let norms = [
                layer.attn_q_inp_norm.as_ref().expect("q_inp_norm"),
                layer.attn_k_inp_norm.as_ref().expect("k_inp_norm"),
                layer.attn_v_inp_norm.as_ref().expect("v_inp_norm"),
            ];
            for (j, nw) in norms.iter().enumerate() {
                state.conv_tmp[..n_embd]
                    .copy_from_slice(&state.inputs[j * n_embd..(j + 1) * n_embd]);
                rms_norm(
                    &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                    &state.conv_tmp[..n_embd],
                    nw,
                    self.cfg.rms_eps,
                );
            }
        }

        // 3. Self-Attention projections, dynamic/static ShortConvs, Q/K/R clipping, V unit-RMSNorm
        matvec_f32(
            pool,
            kt,
            layer.wq.dtype,
            layer.wq.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.q_buf[..q_dim],
            q_dim,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.wk.dtype,
            layer.wk.bytes(),
            &state.inputs[n_embd..2 * n_embd],
            &mut state.mv_scratch,
            &mut state.k_buf[..kv_dim],
            kv_dim,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.wv.dtype,
            layer.wv.bytes(),
            &state.inputs[2 * n_embd..3 * n_embd],
            &mut state.mv_scratch,
            &mut state.v_buf[..kv_dim],
            kv_dim,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.attn_rel.dtype,
            layer.attn_rel.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.r_buf[..nh * n_rel],
            nh * n_rel,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.attn_sink_proj.dtype,
            layer.attn_sink_proj.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.sinks[..nh],
            nh,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.wqkv_gate.dtype,
            layer.wqkv_gate.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.gate_buf[..q_dim],
            q_dim,
            n_embd,
        );

        if !swa {
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            matvec_f32(
                pool,
                kt,
                kp.dtype,
                kp.bytes(),
                &state.inputs[n_embd..2 * n_embd],
                &mut state.mv_scratch,
                &mut state.k_dyn[..n_conv * nh_kv],
                n_conv * nh_kv,
                n_embd,
            );
            matvec_f32(
                pool,
                kt,
                vp.dtype,
                vp.bytes(),
                &state.inputs[2 * n_embd..3 * n_embd],
                &mut state.mv_scratch,
                &mut state.v_dyn[..n_conv * nh_kv],
                n_conv * nh_kv,
                n_embd,
            );
        }

        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            convolve_inplace(
                &mut state.k_buf[..kv_dim],
                &layer.attn_k_conv,
                if swa {
                    None
                } else {
                    Some(&state.k_dyn[..n_conv * nh_kv])
                },
                conv_hist,
                0,
                n_conv,
                &mut state.conv_tmp[..kv_dim],
            );
            convolve_inplace(
                &mut state.v_buf[..kv_dim],
                &layer.attn_v_conv,
                if swa {
                    None
                } else {
                    Some(&state.v_dyn[..n_conv * nh_kv])
                },
                conv_hist,
                kv_dim,
                n_conv,
                &mut state.conv_tmp[..kv_dim],
            );
        }

        clip_heads_inplace(
            &mut state.q_buf[..q_dim],
            nh,
            hd,
            Some(&layer.attn_q_clip),
            self.cfg.qk_clip,
        );
        clip_heads_inplace(
            &mut state.k_buf[..kv_dim],
            nh_kv,
            hd,
            None,
            self.cfg.qk_clip,
        );
        for h in 0..nh_kv {
            rms_norm_unit_inplace(&mut state.v_buf[h * hd..(h + 1) * hd], self.cfg.rms_eps);
        }
        clip_heads_inplace(
            &mut state.r_buf[..nh * n_rel],
            nh,
            n_rel,
            Some(&layer.attn_rel_clip),
            self.cfg.relative_clip,
        );

        if !swa {
            for v in &mut state.q_buf[..q_dim] {
                *v *= tau;
            }
            for v in &mut state.r_buf[..nh * n_rel] {
                *v *= tau;
            }
        }

        // Store K, V in slot `il` (`F16` or `Q8_0`)
        state.layers[il].store_kv_at(kt, pos, &state.k_buf[..kv_dim], &state.v_buf[..kv_dim]);

        // Precompute `rel_table` `[nh, extent + 1]`: `dot(attn_rel_bias[d], Rcur[h]) - sinks[h]`
        let tbl_cols = extent + 1;
        for h in 0..nh {
            let r_h = &state.r_buf[h * n_rel..(h + 1) * n_rel];
            let sink_h = state.sinks[h];
            let row_out = &mut state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
            for (d, dst_d) in row_out[..extent].iter_mut().enumerate() {
                let b_row = &layer.attn_rel_bias[d * n_rel..(d + 1) * n_rel];
                let mut acc = 0.0f32;
                for r in 0..n_rel {
                    acc = r_h[r].mul_add(b_row[r], acc);
                }
                *dst_d = acc - sink_h;
            }
            row_out[extent] = -sink_h;
        }

        // GQA attention with relative bias + zero-sink softmax (`inkling2.cpp:360-374`)
        let kv_start = if swa {
            (pos + 1).saturating_sub(self.cfg.n_swa)
        } else {
            0
        };
        let kv_len = pos + 1 - kv_start;
        let gqa_ratio = nh / nh_kv;
        let attn_scale = 1.0 / (hd as f32).sqrt();
        state.attn_vec[..q_dim].fill(0.0);

        match &state.layers[il] {
            LayerState::Kv {
                kv_type: KvType::F16,
                k_cache,
                v_cache,
                ..
            } => {
                for h in 0..nh {
                    let kvh = h / gqa_ratio;
                    let q_h = &state.q_buf[h * hd..(h + 1) * hd];
                    let tbl_h = &state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                    let sc_h = &mut state.scores[..kv_len];
                    let mut max_s = 0.0f32; // zero sink logit!
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let k_off = kv_pos * kv_dim + kvh * hd;
                        let dot = (kt.dot_f32_f16)(q_h, &k_cache[k_off..k_off + hd]);
                        let dist = pos - kv_pos;
                        let rel_b = tbl_h[dist.min(extent)];
                        let s = dot.mul_add(attn_scale, rel_b);
                        sc_h[idx] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut denom = (-max_s).exp();
                    for s in sc_h.iter_mut() {
                        let e = (*s - max_s).exp();
                        *s = e;
                        denom += e;
                    }
                    let inv_denom = 1.0 / denom;
                    let out_h = &mut state.attn_vec[h * hd..(h + 1) * hd];
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let p = sc_h[idx] * inv_denom;
                        let v_off = kv_pos * kv_dim + kvh * hd;
                        (kt.axpy_f32_f16)(out_h, p, &v_cache[v_off..v_off + hd]);
                    }
                }
            }
            LayerState::Kv {
                kv_type: KvType::Q8_0,
                k_q8,
                v_q8,
                ..
            } => {
                let kv_blks = kv_dim / QK8_0;
                let hd_blks = hd / QK8_0;
                for h in 0..nh {
                    let kvh = h / gqa_ratio;
                    let q_h = &state.q_buf[h * hd..(h + 1) * hd];
                    let tbl_h = &state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                    let sc_h = &mut state.scores[..kv_len];
                    let mut max_s = 0.0f32;
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let k_off = kv_pos * kv_blks + kvh * hd_blks;
                        let dot = (kt.dot_f32_q8_0)(q_h, &k_q8[k_off..k_off + hd_blks]);
                        let dist = pos - kv_pos;
                        let rel_b = tbl_h[dist.min(extent)];
                        let s = dot.mul_add(attn_scale, rel_b);
                        sc_h[idx] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut denom = (-max_s).exp();
                    for s in sc_h.iter_mut() {
                        let e = (*s - max_s).exp();
                        *s = e;
                        denom += e;
                    }
                    let inv_denom = 1.0 / denom;
                    let out_h = &mut state.attn_vec[h * hd..(h + 1) * hd];
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let p = sc_h[idx] * inv_denom;
                        let v_off = kv_pos * kv_blks + kvh * hd_blks;
                        (kt.axpy_f32_q8_0)(out_h, p, &v_q8[v_off..v_off + hd_blks]);
                    }
                }
            }
            _ => unreachable!("slot 0..n_slots is always LayerState::Kv"),
        }

        // Sigmoid output gate + `wo` + `attn_post_conv` (`inkling2.cpp:375-376`)
        for i in 0..q_dim {
            let sig = 1.0 / (1.0 + (-state.gate_buf[i]).exp());
            state.attn_vec[i] *= sig;
        }
        matvec_f32(
            pool,
            kt,
            layer.wo.dtype,
            layer.wo.bytes(),
            &state.attn_vec[..q_dim],
            &mut state.mv_scratch,
            &mut state.wo_out[..n_embd],
            n_embd,
            q_dim,
        );
        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            convolve_inplace(
                &mut state.wo_out[..n_embd],
                &layer.attn_post_conv,
                None,
                conv_hist,
                2 * kv_dim,
                n_conv,
                &mut state.conv_tmp[..n_embd],
            );
        }

        // Block residual stream boundary (`inkling2.cpp:381-385`)
        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            let b_off = state.blocks_len * n_embd;
            state.blocks_buf[b_off..b_off + n_embd].copy_from_slice(&state.inp_sa);
            state.blocks_len += 1;
            state.cur.copy_from_slice(&state.wo_out[..n_embd]);
        } else {
            for c in 0..n_embd {
                state.cur[c] = state.inp_sa[c] + state.wo_out[c];
            }
        }

        state.ffn_inp.copy_from_slice(&state.cur);

        // 4. MoE FFN (`mix` -> optional `control(FFN)` -> `ffn_norm` -> `MoE SiTU` -> `ffn_post_conv`)
        mix_depth_into(
            &state.blocks_buf[..state.blocks_len * n_embd],
            state.blocks_len,
            &state.cur,
            &layer.ffn_res_query,
            1,
            n_embd,
            &mut state.mix_states_hat,
            &mut state.mix_scores,
            &mut state.inp_sa,
        );
        if has_messages {
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            control_inplace(
                &mut state.inp_sa,
                &state.messages[LOOP_CTRL_FFN * n_embd..(LOOP_CTRL_FFN + 1) * n_embd],
                loop_w,
                LOOP_CTRL_FFN,
                n_embd,
                self.cfg.gate_rank,
                &mut state.ctrl_in,
                &mut state.ctrl_g,
                &mut state.ctrl_gain,
                &mut state.ctrl_forget,
            );
        }
        rms_norm(
            &mut state.wo_out[..n_embd],
            &state.inp_sa,
            &layer.ffn_norm,
            self.cfg.rms_eps,
        );

        self.route_moe_single(
            ip,
            &state.wo_out[..n_embd],
            pool,
            kt,
            &mut state.mv_scratch,
            &mut state.router_logits,
            &mut state.router_probs,
            &mut state.router_sel,
            &mut state.top_ids,
            &mut state.top_weights,
        );
        self.eval_moe_experts_inner(
            ip,
            &state.wo_out[..n_embd],
            &state.top_ids,
            &state.top_weights,
            pool,
            Some(&state.io_pool),
            &state.serial_pool,
            kt,
            &mut state.expert_slots,
            &mut state.mv_scratch,
            &mut state.moe_out,
            false,
        );

        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            convolve_inplace(
                &mut state.moe_out[..n_embd],
                &layer.ffn_post_conv,
                None,
                conv_hist,
                2 * kv_dim + n_embd,
                n_conv,
                &mut state.conv_tmp[..n_embd],
            );
        }

        for c in 0..n_embd {
            state.cur[c] = state.ffn_inp[c] + state.moe_out[c];
        }
    }

    /// Evaluates a prompt/prefill batch `token_ids` starting at `start_pos`.
    /// Validates bounds BEFORE mutating any state (`AGENTS.md` Rule #6), and guarantees
    /// `0-ULP` bit-exact equivalence with sequential `forward_token` (`AGENTS.md` §9).
    pub fn forward_batch(
        &self,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_ids: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
    ) -> Result<(), ModelError> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let end_pos = start_pos.saturating_add(token_ids.len());
        if end_pos > state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: end_pos - 1,
                max_ctx: state.max_ctx,
            });
        }
        for &tid in token_ids {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }
        let total = token_ids.len();
        if total == 1 {
            return self.forward_token(
                state,
                pool,
                kt,
                token_ids[0],
                start_pos,
                compute_last_logits,
            );
        }
        let mut off = 0usize;
        while off < total {
            let w = (total - off).min(MAX_INKLING2_WINDOW);
            let is_last = off + w == total;
            let want_logits = compute_last_logits && is_last;
            if w == 1 {
                self.forward_token(
                    state,
                    pool,
                    kt,
                    token_ids[off],
                    start_pos + off,
                    want_logits,
                )?;
            } else {
                self.forward_window(
                    state,
                    pool,
                    kt,
                    &token_ids[off..off + w],
                    start_pos + off,
                    want_logits,
                    None,
                )?;
            }
            off += w;
        }
        Ok(())
    }

    /// Evaluates speculative verification window `window_tokens` (`1 <= W <= MAX_SPEC_WINDOW`)
    /// while leaving `ShortConv` recurrent state restorable on partial acceptance (`ADR-005`).
    pub fn verify_speculative(
        &self,
        state: &mut Inkling2State,
        spec_buf: &mut SpecVerifyBuf,
        pool: &ThreadPool,
        kt: &KernelTable,
        window_tokens: &[u32],
        start_pos: usize,
    ) -> Result<(), ModelError> {
        let w = window_tokens.len();
        assert!((1..=MAX_SPEC_WINDOW).contains(&w));
        let end_pos = start_pos.saturating_add(w);
        if end_pos > state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos: end_pos - 1,
                max_ctx: state.max_ctx,
            });
        }
        for &tid in window_tokens {
            if (tid as usize) >= self.cfg.vocab_size {
                return Err(ModelError::TokenOutOfRange {
                    token_id: tid,
                    vocab_size: self.cfg.vocab_size,
                });
            }
        }

        let vocab = self.cfg.vocab_size;
        let n_embd = self.cfg.n_embd;
        let conv_total = spec_buf.tmp_conv_hist.len();

        // Save pre-verify ShortConv state (`tmp_conv_hist`) so `ShortConv` is untouched until `commit_speculative`.
        state.pack_shortconv_into(&mut spec_buf.tmp_conv_hist);

        if w == 1 {
            self.forward_token(state, pool, kt, window_tokens[0], start_pos, true)?;
            spec_buf.logits_win[..vocab].copy_from_slice(&state.logits);
            spec_buf.hidden_win[..n_embd].copy_from_slice(&state.last_hidden);
            state.pack_shortconv_into(&mut spec_buf.sc_bx_win[..conv_total]);
        } else {
            self.forward_window(
                state,
                pool,
                kt,
                window_tokens,
                start_pos,
                false,
                Some((
                    &mut spec_buf.logits_win[..w * vocab],
                    &mut spec_buf.hidden_win[..w * n_embd],
                    &mut spec_buf.sc_bx_win[..w * conv_total],
                    conv_total,
                )),
            )?;
        }

        // Restore pre-verify ShortConv until `commit_speculative` commits `1..=w` accepted tokens.
        state.unpack_shortconv_from(&spec_buf.tmp_conv_hist);
        spec_buf.last_window_len = w;
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        clippy::needless_range_loop
    )]
    fn forward_window(
        &self,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        window_tokens: &[u32],
        start_pos: usize,
        compute_last_logits: bool,
        mut spec_out: Option<(&mut [f32], &mut [f32], &mut [f32], usize)>,
    ) -> Result<(), ModelError> {
        let w = window_tokens.len();
        debug_assert!((2..=MAX_INKLING2_WINDOW).contains(&w));
        let n_embd = self.cfg.n_embd;
        let max_blocks = self.cfg.max_depth_blocks();
        let hist_len = self.cfg.shortconv_l_cache.saturating_sub(1);

        let mut taus = [0.0f32; MAX_INKLING2_WINDOW];
        for (t, &tid) in window_tokens.iter().enumerate() {
            decode_embedding_row(
                self.token_embd.dtype,
                self.token_embd.bytes(),
                n_embd,
                tid as usize,
                &mut state.inp_sa,
            );
            rms_norm(
                &mut state.win_cur[t * n_embd..(t + 1) * n_embd],
                &state.inp_sa,
                &self.token_embd_norm,
                self.cfg.rms_eps,
            );
            let ratio = ((start_pos + t + 1) as f32) / (self.cfg.log_scale_floor as f32);
            taus[t] = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();
        }

        // Precompute cumulative ShortConv channel float offsets per logical slot `il`
        let mut sc_slot_offs = [0usize; 128];
        if spec_out.is_some() {
            let mut acc = 0usize;
            for il in 0..self.cfg.n_layers {
                sc_slot_offs[il] = acc;
                let ip = self.cfg.layer_phys(il);
                acc += (2 * self.cfg.kv_dim(ip) + 2 * n_embd) * hist_len;
            }
        }

        state.blocks_len = 0;
        let mut visit = 0usize;
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                    .copy_from_slice(&state.win_cur[(w - 1) * n_embd..w * n_embd]);
                visit += 1;
                let sc_arg = spec_out
                    .as_mut()
                    .map(|(_, _, sc_w, ct)| (&mut **sc_w, *ct, sc_slot_offs[il]));
                self.run_layer_window(
                    il,
                    il,
                    start_pos,
                    w,
                    &taus[..w],
                    false,
                    state,
                    pool,
                    kt,
                    sc_arg,
                );
            }
        } else {
            let cs = self.cfg.core_start;
            let ce = self.cfg.core_end;
            for il in 0..cs {
                state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                    .copy_from_slice(&state.win_cur[(w - 1) * n_embd..w * n_embd]);
                visit += 1;
                let sc_arg = spec_out
                    .as_mut()
                    .map(|(_, _, sc_w, ct)| (&mut **sc_w, *ct, sc_slot_offs[il]));
                self.run_layer_window(
                    il,
                    il,
                    start_pos,
                    w,
                    &taus[..w],
                    false,
                    state,
                    pool,
                    kt,
                    sc_arg,
                );
            }

            state.win_prelude[..w * n_embd].copy_from_slice(&state.win_cur[..w * n_embd]);
            state.prelude_blocks_len = state.blocks_len;
            let pb_floats = state.prelude_blocks_len * n_embd;
            for t in 0..w {
                let off = t * max_blocks * n_embd;
                state.win_prelude_blocks_buf[off..off + pb_floats]
                    .copy_from_slice(&state.win_blocks_buf[off..off + pb_floats]);
            }

            for pass in 0..self.cfg.n_passes {
                if pass > 0 {
                    let bank_start_blk = state.prelude_blocks_len + 1;
                    state.bank_len = state.blocks_len.saturating_sub(bank_start_blk);
                    let b_floats = state.bank_len * n_embd;
                    for t in 0..w {
                        let base = t * max_blocks * n_embd;
                        if b_floats > 0 {
                            let s0 = base + bank_start_blk * n_embd;
                            state.win_bank_buf[base..base + b_floats]
                                .copy_from_slice(&state.win_blocks_buf[s0..s0 + b_floats]);
                        }
                        state.win_blocks_buf[base..base + pb_floats]
                            .copy_from_slice(&state.win_prelude_blocks_buf[base..base + pb_floats]);
                    }
                    state.win_bank_out[..w * n_embd].copy_from_slice(&state.win_cur[..w * n_embd]);
                    state.win_cur[..w * n_embd].copy_from_slice(&state.win_prelude[..w * n_embd]);
                    state.blocks_len = state.prelude_blocks_len;
                }

                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                        .copy_from_slice(&state.win_cur[(w - 1) * n_embd..w * n_embd]);
                    visit += 1;
                    if pass > 0 {
                        let loop_w = &self.loop_layers[ip - cs];
                        let b_floats = state.bank_len * n_embd;
                        for t in 0..w {
                            let b_off = t * max_blocks * n_embd;
                            mix_depth_into(
                                &state.win_bank_buf[b_off..b_off + b_floats],
                                state.bank_len,
                                &state.win_bank_out[t * n_embd..(t + 1) * n_embd],
                                &loop_w.msg_query,
                                LOOP_CTRL_COUNT,
                                n_embd,
                                &mut state.mix_states_hat,
                                &mut state.mix_scores,
                                &mut state.win_messages[t * LOOP_CTRL_COUNT * n_embd
                                    ..(t + 1) * LOOP_CTRL_COUNT * n_embd],
                            );
                        }
                    }
                    let sc_arg = spec_out
                        .as_mut()
                        .map(|(_, _, sc_w, ct)| (&mut **sc_w, *ct, sc_slot_offs[il]));
                    self.run_layer_window(
                        il,
                        ip,
                        start_pos,
                        w,
                        &taus[..w],
                        pass > 0,
                        state,
                        pool,
                        kt,
                        sc_arg,
                    );
                }
            }

            for il in ce..self.cfg.n_layers_phys {
                state.slot_outputs[visit * n_embd..(visit + 1) * n_embd]
                    .copy_from_slice(&state.win_cur[(w - 1) * n_embd..w * n_embd]);
                visit += 1;
                let sc_arg = spec_out
                    .as_mut()
                    .map(|(_, _, sc_w, ct)| (&mut **sc_w, *ct, sc_slot_offs[il]));
                self.run_layer_window(
                    il,
                    il,
                    start_pos,
                    w,
                    &taus[..w],
                    false,
                    state,
                    pool,
                    kt,
                    sc_arg,
                );
            }
        }
        debug_assert_eq!(visit, self.cfg.n_layers);

        // Output depth-attention mix + output RMSNorm per token
        let blk_floats = state.blocks_len * n_embd;
        for t in 0..w {
            let b_off = t * max_blocks * n_embd;
            mix_depth_into(
                &state.win_blocks_buf[b_off..b_off + blk_floats],
                state.blocks_len,
                &state.win_cur[t * n_embd..(t + 1) * n_embd],
                &self.output_res_query,
                1,
                n_embd,
                &mut state.mix_states_hat,
                &mut state.mix_scores,
                &mut state.inp_sa,
            );
            rms_norm(
                &mut state.win_wo_out[t * n_embd..(t + 1) * n_embd],
                &state.inp_sa,
                &self.output_norm,
                self.cfg.rms_eps,
            );
        }

        state
            .cur
            .copy_from_slice(&state.win_cur[(w - 1) * n_embd..w * n_embd]);
        state
            .last_hidden
            .copy_from_slice(&state.win_wo_out[(w - 1) * n_embd..w * n_embd]);
        let final_slot_off = self.cfg.n_layers * n_embd;
        state.slot_outputs[final_slot_off..final_slot_off + n_embd]
            .copy_from_slice(&state.last_hidden);

        let vp = self.cfg.vocab_padded;
        let vu = self.cfg.vocab_size;
        if let Some((logits_win, hidden_win, _, _)) = spec_out {
            hidden_win.copy_from_slice(&state.win_wo_out[..w * n_embd]);
            matmul_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes(),
                &hidden_win[..w * n_embd],
                &mut state.mv_scratch,
                &mut state.logits_padded[..w * vp],
                vp,
                n_embd,
                w,
            );
            for t in 0..w {
                let dst = &mut logits_win[t * vu..(t + 1) * vu];
                dst.copy_from_slice(&state.logits_padded[t * vp..t * vp + vu]);
                if let Some(cap) = self.cfg.final_logit_softcapping {
                    softcap_inplace_parallel(pool, dst, cap);
                }
            }
            state
                .logits
                .copy_from_slice(&logits_win[(w - 1) * vu..w * vu]);
        } else if compute_last_logits {
            matvec_f32(
                pool,
                kt,
                self.output.dtype,
                self.output.bytes(),
                &state.last_hidden,
                &mut state.mv_scratch,
                &mut state.logits_padded[..vp],
                vp,
                n_embd,
            );
            state.logits.copy_from_slice(&state.logits_padded[..vu]);
            if let Some(cap) = self.cfg.final_logit_softcapping {
                softcap_inplace_parallel(pool, &mut state.logits, cap);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_window(
        &self,
        il: usize,
        ip: usize,
        start_pos: usize,
        w: usize,
        taus: &[f32],
        has_messages: bool,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        mut sc_snap: Option<(&mut [f32], usize, usize)>,
    ) {
        let n_embd = self.cfg.n_embd;
        let swa = self.cfg.is_swa[ip];
        let nh = self.cfg.n_head(ip);
        let nh_kv = self.cfg.n_head_kv(ip);
        let hd = self.cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nh_kv * hd;
        let n_rel = self.cfg.n_relative;
        let n_conv = self.cfg.shortconv_l_cache;
        let hist_len = n_conv.saturating_sub(1);
        let extent = self.cfg.rel_extent_for(ip);
        let n_slots = self.cfg.n_layers;
        let max_blocks = self.cfg.max_depth_blocks();
        let top_k = self.cfg.n_expert_used;
        let layer = &self.layers[ip];

        state.win_inp_sa[..w * n_embd].copy_from_slice(&state.win_cur[..w * n_embd]);

        // 1 & 2. Depth-residual mix + optional cross-pass controller + input RMSNorm per token
        let blk_floats = state.blocks_len * n_embd;
        for t in 0..w {
            let cur_t = &state.win_cur[t * n_embd..(t + 1) * n_embd];
            let b_off = t * max_blocks * n_embd;
            if let Some(ref rq) = layer.attn_res_query {
                if swa {
                    mix_depth_into(
                        &state.win_blocks_buf[b_off..b_off + blk_floats],
                        state.blocks_len,
                        cur_t,
                        rq,
                        1,
                        n_embd,
                        &mut state.mix_states_hat,
                        &mut state.mix_scores,
                        &mut state.inputs[..n_embd],
                    );
                    let (first, rest) = state.inputs.split_at_mut(n_embd);
                    rest[..n_embd].copy_from_slice(first);
                    rest[n_embd..2 * n_embd].copy_from_slice(first);
                } else {
                    mix_depth_into(
                        &state.win_blocks_buf[b_off..b_off + blk_floats],
                        state.blocks_len,
                        cur_t,
                        rq,
                        3,
                        n_embd,
                        &mut state.mix_states_hat,
                        &mut state.mix_scores,
                        &mut state.inputs[..3 * n_embd],
                    );
                }
            } else {
                for j in 0..3 {
                    state.inputs[j * n_embd..(j + 1) * n_embd].copy_from_slice(cur_t);
                }
            }

            if has_messages {
                let loop_w = &self.loop_layers[ip - self.cfg.core_start];
                let msg_t = &state.win_messages
                    [t * LOOP_CTRL_COUNT * n_embd..(t + 1) * LOOP_CTRL_COUNT * n_embd];
                for j in 0..3 {
                    control_inplace(
                        &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                        &msg_t[j * n_embd..(j + 1) * n_embd],
                        loop_w,
                        j,
                        n_embd,
                        self.cfg.gate_rank,
                        &mut state.ctrl_in,
                        &mut state.ctrl_g,
                        &mut state.ctrl_gain,
                        &mut state.ctrl_forget,
                    );
                }
            }

            if swa && has_messages {
                let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
                for j in 0..3 {
                    state.conv_tmp[..n_embd]
                        .copy_from_slice(&state.inputs[j * n_embd..(j + 1) * n_embd]);
                    rms_norm(
                        &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                        &state.conv_tmp[..n_embd],
                        an,
                        self.cfg.rms_eps,
                    );
                }
            } else if swa {
                let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
                state.conv_tmp[..n_embd].copy_from_slice(&state.inputs[..n_embd]);
                rms_norm(
                    &mut state.inputs[..n_embd],
                    &state.conv_tmp[..n_embd],
                    an,
                    self.cfg.rms_eps,
                );
                let (first, rest) = state.inputs.split_at_mut(n_embd);
                rest[..n_embd].copy_from_slice(first);
                rest[n_embd..2 * n_embd].copy_from_slice(first);
            } else {
                let norms = [
                    layer.attn_q_inp_norm.as_ref().expect("q_inp_norm"),
                    layer.attn_k_inp_norm.as_ref().expect("k_inp_norm"),
                    layer.attn_v_inp_norm.as_ref().expect("v_inp_norm"),
                ];
                for (j, nw) in norms.iter().enumerate() {
                    state.conv_tmp[..n_embd]
                        .copy_from_slice(&state.inputs[j * n_embd..(j + 1) * n_embd]);
                    rms_norm(
                        &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                        &state.conv_tmp[..n_embd],
                        nw,
                        self.cfg.rms_eps,
                    );
                }
            }

            state.win_q_in[t * n_embd..(t + 1) * n_embd].copy_from_slice(&state.inputs[..n_embd]);
            state.win_k_in[t * n_embd..(t + 1) * n_embd]
                .copy_from_slice(&state.inputs[n_embd..2 * n_embd]);
            state.win_v_in[t * n_embd..(t + 1) * n_embd]
                .copy_from_slice(&state.inputs[2 * n_embd..3 * n_embd]);
        }

        // 3. Batched dense Self-Attention projections (`0-ULP` identical to `matvec_f32`!)
        matmul_f32(
            pool,
            kt,
            layer.wq.dtype,
            layer.wq.bytes(),
            &state.win_q_in[..w * n_embd],
            &mut state.mv_scratch,
            &mut state.win_q_buf[..w * q_dim],
            q_dim,
            n_embd,
            w,
        );
        matmul_f32(
            pool,
            kt,
            layer.wk.dtype,
            layer.wk.bytes(),
            &state.win_k_in[..w * n_embd],
            &mut state.mv_scratch,
            &mut state.win_k_buf[..w * kv_dim],
            kv_dim,
            n_embd,
            w,
        );
        matmul_f32(
            pool,
            kt,
            layer.wv.dtype,
            layer.wv.bytes(),
            &state.win_v_in[..w * n_embd],
            &mut state.mv_scratch,
            &mut state.win_v_buf[..w * kv_dim],
            kv_dim,
            n_embd,
            w,
        );
        matmul_f32(
            pool,
            kt,
            layer.attn_rel.dtype,
            layer.attn_rel.bytes(),
            &state.win_q_in[..w * n_embd],
            &mut state.mv_scratch,
            &mut state.win_r_buf[..w * nh * n_rel],
            nh * n_rel,
            n_embd,
            w,
        );
        matmul_f32(
            pool,
            kt,
            layer.attn_sink_proj.dtype,
            layer.attn_sink_proj.bytes(),
            &state.win_q_in[..w * n_embd],
            &mut state.mv_scratch,
            &mut state.win_sinks[..w * nh],
            nh,
            n_embd,
            w,
        );
        matmul_f32(
            pool,
            kt,
            layer.wqkv_gate.dtype,
            layer.wqkv_gate.bytes(),
            &state.win_q_in[..w * n_embd],
            &mut state.mv_scratch,
            &mut state.win_gate_buf[..w * q_dim],
            q_dim,
            n_embd,
            w,
        );
        if !swa {
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            let dyn_dim = n_conv * nh_kv;
            matmul_f32(
                pool,
                kt,
                kp.dtype,
                kp.bytes(),
                &state.win_k_in[..w * n_embd],
                &mut state.mv_scratch,
                &mut state.win_k_dyn[..w * dyn_dim],
                dyn_dim,
                n_embd,
                w,
            );
            matmul_f32(
                pool,
                kt,
                vp.dtype,
                vp.bytes(),
                &state.win_v_in[..w * n_embd],
                &mut state.mv_scratch,
                &mut state.win_v_dyn[..w * dyn_dim],
                dyn_dim,
                n_embd,
                w,
            );
        }

        // Causal ShortConv(K/V), Q/K/R clip, KV store, GQA attention, and sigmoid gate per token `t`
        let tbl_cols = extent + 1;
        let gqa_ratio = nh / nh_kv;
        let attn_scale = 1.0 / (hd as f32).sqrt();
        let dyn_dim = n_conv * nh_kv;
        for (t, &tau) in taus.iter().enumerate().take(w) {
            let pos = start_pos + t;
            {
                let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
                convolve_inplace(
                    &mut state.win_k_buf[t * kv_dim..(t + 1) * kv_dim],
                    &layer.attn_k_conv,
                    if swa {
                        None
                    } else {
                        Some(&state.win_k_dyn[t * dyn_dim..(t + 1) * dyn_dim])
                    },
                    conv_hist,
                    0,
                    n_conv,
                    &mut state.conv_tmp[..kv_dim],
                );
                convolve_inplace(
                    &mut state.win_v_buf[t * kv_dim..(t + 1) * kv_dim],
                    &layer.attn_v_conv,
                    if swa {
                        None
                    } else {
                        Some(&state.win_v_dyn[t * dyn_dim..(t + 1) * dyn_dim])
                    },
                    conv_hist,
                    kv_dim,
                    n_conv,
                    &mut state.conv_tmp[..kv_dim],
                );
                if let Some((ref mut sc_w, ct, il_off)) = sc_snap {
                    let len = 2 * kv_dim * hist_len;
                    let dst = t * ct + il_off;
                    sc_w[dst..dst + len].copy_from_slice(&conv_hist[..len]);
                }
            }

            let q_t = &mut state.win_q_buf[t * q_dim..(t + 1) * q_dim];
            let k_t = &mut state.win_k_buf[t * kv_dim..(t + 1) * kv_dim];
            let v_t = &mut state.win_v_buf[t * kv_dim..(t + 1) * kv_dim];
            let r_t = &mut state.win_r_buf[t * nh * n_rel..(t + 1) * nh * n_rel];

            clip_heads_inplace(q_t, nh, hd, Some(&layer.attn_q_clip), self.cfg.qk_clip);
            clip_heads_inplace(k_t, nh_kv, hd, None, self.cfg.qk_clip);
            for h in 0..nh_kv {
                rms_norm_unit_inplace(&mut v_t[h * hd..(h + 1) * hd], self.cfg.rms_eps);
            }
            clip_heads_inplace(
                r_t,
                nh,
                n_rel,
                Some(&layer.attn_rel_clip),
                self.cfg.relative_clip,
            );
            if !swa {
                for v in q_t.iter_mut() {
                    *v *= tau;
                }
                for v in r_t.iter_mut() {
                    *v *= tau;
                }
            }

            state.layers[il].store_kv_at(kt, pos, k_t, v_t);

            for h in 0..nh {
                let r_h = &r_t[h * n_rel..(h + 1) * n_rel];
                let sink_h = state.win_sinks[t * nh + h];
                let row_out = &mut state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                for (d, dst_d) in row_out[..extent].iter_mut().enumerate() {
                    let b_row = &layer.attn_rel_bias[d * n_rel..(d + 1) * n_rel];
                    let mut acc = 0.0f32;
                    for r in 0..n_rel {
                        acc = r_h[r].mul_add(b_row[r], acc);
                    }
                    *dst_d = acc - sink_h;
                }
                row_out[extent] = -sink_h;
            }

            let kv_start = if swa {
                (pos + 1).saturating_sub(self.cfg.n_swa)
            } else {
                0
            };
            let kv_len = pos + 1 - kv_start;
            let attn_t = &mut state.win_attn_vec[t * q_dim..(t + 1) * q_dim];
            attn_t.fill(0.0);

            match &state.layers[il] {
                LayerState::Kv {
                    kv_type: KvType::F16,
                    k_cache,
                    v_cache,
                    ..
                } => {
                    for h in 0..nh {
                        let kvh = h / gqa_ratio;
                        let q_h = &q_t[h * hd..(h + 1) * hd];
                        let tbl_h = &state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                        let sc_h = &mut state.scores[..kv_len];
                        let mut max_s = 0.0f32;
                        for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                            let k_off = kv_pos * kv_dim + kvh * hd;
                            let dot = (kt.dot_f32_f16)(q_h, &k_cache[k_off..k_off + hd]);
                            let dist = pos - kv_pos;
                            let rel_b = tbl_h[dist.min(extent)];
                            let s = dot.mul_add(attn_scale, rel_b);
                            sc_h[idx] = s;
                            if s > max_s {
                                max_s = s;
                            }
                        }
                        let mut denom = (-max_s).exp();
                        for s in sc_h.iter_mut() {
                            let e = (*s - max_s).exp();
                            *s = e;
                            denom += e;
                        }
                        let inv_denom = 1.0 / denom;
                        let out_h = &mut attn_t[h * hd..(h + 1) * hd];
                        for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                            let p = sc_h[idx] * inv_denom;
                            let v_off = kv_pos * kv_dim + kvh * hd;
                            (kt.axpy_f32_f16)(out_h, p, &v_cache[v_off..v_off + hd]);
                        }
                    }
                }
                LayerState::Kv {
                    kv_type: KvType::Q8_0,
                    k_q8,
                    v_q8,
                    ..
                } => {
                    let kv_blks = kv_dim / QK8_0;
                    let hd_blks = hd / QK8_0;
                    for h in 0..nh {
                        let kvh = h / gqa_ratio;
                        let q_h = &q_t[h * hd..(h + 1) * hd];
                        let tbl_h = &state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                        let sc_h = &mut state.scores[..kv_len];
                        let mut max_s = 0.0f32;
                        for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                            let k_off = kv_pos * kv_blks + kvh * hd_blks;
                            let dot = (kt.dot_f32_q8_0)(q_h, &k_q8[k_off..k_off + hd_blks]);
                            let dist = pos - kv_pos;
                            let rel_b = tbl_h[dist.min(extent)];
                            let s = dot.mul_add(attn_scale, rel_b);
                            sc_h[idx] = s;
                            if s > max_s {
                                max_s = s;
                            }
                        }
                        let mut denom = (-max_s).exp();
                        for s in sc_h.iter_mut() {
                            let e = (*s - max_s).exp();
                            *s = e;
                            denom += e;
                        }
                        let inv_denom = 1.0 / denom;
                        let out_h = &mut attn_t[h * hd..(h + 1) * hd];
                        for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                            let p = sc_h[idx] * inv_denom;
                            let v_off = kv_pos * kv_blks + kvh * hd_blks;
                            (kt.axpy_f32_q8_0)(out_h, p, &v_q8[v_off..v_off + hd_blks]);
                        }
                    }
                }
                _ => unreachable!("slot 0..n_slots is always LayerState::Kv"),
            }

            let gate_t = &state.win_gate_buf[t * q_dim..(t + 1) * q_dim];
            for i in 0..q_dim {
                let sig = 1.0 / (1.0 + (-gate_t[i]).exp());
                attn_t[i] *= sig;
            }
        }

        // Batched `wo` projection + causal `attn_post_conv` + block residual update
        matmul_f32(
            pool,
            kt,
            layer.wo.dtype,
            layer.wo.bytes(),
            &state.win_attn_vec[..w * q_dim],
            &mut state.mv_scratch,
            &mut state.win_wo_out[..w * n_embd],
            n_embd,
            q_dim,
            w,
        );
        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            let c0 = 2 * kv_dim * hist_len;
            let clen = n_embd * hist_len;
            for t in 0..w {
                convolve_inplace(
                    &mut state.win_wo_out[t * n_embd..(t + 1) * n_embd],
                    &layer.attn_post_conv,
                    None,
                    conv_hist,
                    2 * kv_dim,
                    n_conv,
                    &mut state.conv_tmp[..n_embd],
                );
                if let Some((ref mut sc_w, ct, il_off)) = sc_snap {
                    let dst = t * ct + il_off + c0;
                    sc_w[dst..dst + clen].copy_from_slice(&conv_hist[c0..c0 + clen]);
                }
            }
        }

        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            let blk_idx = state.blocks_len;
            for t in 0..w {
                let b_off = t * max_blocks * n_embd + blk_idx * n_embd;
                state.win_blocks_buf[b_off..b_off + n_embd]
                    .copy_from_slice(&state.win_inp_sa[t * n_embd..(t + 1) * n_embd]);
            }
            state.blocks_len += 1;
            state.win_cur[..w * n_embd].copy_from_slice(&state.win_wo_out[..w * n_embd]);
        } else {
            for i in 0..w * n_embd {
                state.win_cur[i] = state.win_inp_sa[i] + state.win_wo_out[i];
            }
        }

        state.win_ffn_inp[..w * n_embd].copy_from_slice(&state.win_cur[..w * n_embd]);

        // 4. MoE FFN: depth mix + optional FFN controller + `ffn_norm` + routing -> grouped MoE!
        let ffn_blk_floats = state.blocks_len * n_embd;
        for t in 0..w {
            let b_off = t * max_blocks * n_embd;
            mix_depth_into(
                &state.win_blocks_buf[b_off..b_off + ffn_blk_floats],
                state.blocks_len,
                &state.win_cur[t * n_embd..(t + 1) * n_embd],
                &layer.ffn_res_query,
                1,
                n_embd,
                &mut state.mix_states_hat,
                &mut state.mix_scores,
                &mut state.inp_sa,
            );
            if has_messages {
                let loop_w = &self.loop_layers[ip - self.cfg.core_start];
                let msg_ffn = &state.win_messages[(t * LOOP_CTRL_COUNT + LOOP_CTRL_FFN) * n_embd
                    ..(t * LOOP_CTRL_COUNT + LOOP_CTRL_FFN + 1) * n_embd];
                control_inplace(
                    &mut state.inp_sa,
                    msg_ffn,
                    loop_w,
                    LOOP_CTRL_FFN,
                    n_embd,
                    self.cfg.gate_rank,
                    &mut state.ctrl_in,
                    &mut state.ctrl_g,
                    &mut state.ctrl_gain,
                    &mut state.ctrl_forget,
                );
            }
            rms_norm(
                &mut state.win_wo_out[t * n_embd..(t + 1) * n_embd],
                &state.inp_sa,
                &layer.ffn_norm,
                self.cfg.rms_eps,
            );
            self.route_moe_single(
                ip,
                &state.win_wo_out[t * n_embd..(t + 1) * n_embd],
                pool,
                kt,
                &mut state.mv_scratch,
                &mut state.router_logits,
                &mut state.router_probs,
                &mut state.router_sel,
                &mut state.win_top_ids[t * top_k..(t + 1) * top_k],
                &mut state.win_top_weights[t * top_k..(t + 1) * top_k],
            );
        }

        self.eval_moe_experts_window_inner(
            ip,
            w,
            &state.win_wo_out[..w * n_embd],
            &state.win_top_ids[..w * top_k],
            &state.win_top_weights[..w * top_k],
            pool,
            Some(&state.io_pool),
            &state.serial_pool,
            kt,
            &mut state.expert_slots,
            &mut state.mv_scratch,
            &mut state.win_slot_outs[..w * top_k * n_embd],
            &mut state.win_moe_out[..w * n_embd],
            false,
        );

        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            let c0 = (2 * kv_dim + n_embd) * hist_len;
            let clen = n_embd * hist_len;
            for t in 0..w {
                convolve_inplace(
                    &mut state.win_moe_out[t * n_embd..(t + 1) * n_embd],
                    &layer.ffn_post_conv,
                    None,
                    conv_hist,
                    2 * kv_dim + n_embd,
                    n_conv,
                    &mut state.conv_tmp[..n_embd],
                );
                if let Some((ref mut sc_w, ct, il_off)) = sc_snap {
                    let dst = t * ct + il_off + c0;
                    sc_w[dst..dst + clen].copy_from_slice(&conv_hist[c0..c0 + clen]);
                }
            }
        }

        for i in 0..w * n_embd {
            state.win_cur[i] = state.win_ffn_inp[i] + state.win_moe_out[i];
        }
    }

    /// Profiled single-token forward pass recording wall-clock nanoseconds & active bytes per stage.
    pub fn forward_token_profiled(
        &self,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        token_id: u32,
        pos: usize,
        prof: &mut Inkling2DecodeProfile,
    ) -> Result<(), ModelError> {
        use std::time::Instant;
        if pos >= state.max_ctx {
            return Err(ModelError::ContextOverflow {
                pos,
                max_ctx: state.max_ctx,
            });
        }
        if (token_id as usize) >= self.cfg.vocab_size {
            return Err(ModelError::TokenOutOfRange {
                token_id,
                vocab_size: self.cfg.vocab_size,
            });
        }

        let t0 = Instant::now();
        let n_embd = self.cfg.n_embd;
        decode_embedding_row(
            self.token_embd.dtype,
            self.token_embd.bytes(),
            n_embd,
            token_id as usize,
            &mut state.inp_sa,
        );
        rms_norm(
            &mut state.cur,
            &state.inp_sa,
            &self.token_embd_norm,
            self.cfg.rms_eps,
        );
        let ratio = ((pos + 1) as f32) / (self.cfg.log_scale_floor as f32);
        let tau = 1.0 + self.cfg.log_scale_alpha * ratio.max(1.0).ln();
        prof.embed_mix_ns += t0.elapsed().as_nanos() as u64;

        state.blocks_len = 0;
        let cs = self.cfg.core_start;
        let ce = self.cfg.core_end;
        if !self.cfg.is_looped() {
            for il in 0..self.cfg.n_layers_phys {
                self.run_layer_single_profiled(il, il, pos, tau, false, state, pool, kt, prof);
            }
        } else {
            for il in 0..cs {
                self.run_layer_single_profiled(il, il, pos, tau, false, state, pool, kt, prof);
            }
            state.prelude.copy_from_slice(&state.cur);
            state.prelude_blocks_len = state.blocks_len;
            let pb_floats = state.prelude_blocks_len * n_embd;
            state.prelude_blocks_buf[..pb_floats].copy_from_slice(&state.blocks_buf[..pb_floats]);

            for pass in 0..self.cfg.n_passes {
                if pass > 0 {
                    let bank_start_blk = state.prelude_blocks_len + 1;
                    state.bank_len = state.blocks_len.saturating_sub(bank_start_blk);
                    let b_floats = state.bank_len * n_embd;
                    if b_floats > 0 {
                        let s0 = bank_start_blk * n_embd;
                        state.bank_buf[..b_floats]
                            .copy_from_slice(&state.blocks_buf[s0..s0 + b_floats]);
                    }
                    state.bank_out.copy_from_slice(&state.cur);
                    state.cur.copy_from_slice(&state.prelude);
                    state.blocks_len = state.prelude_blocks_len;
                    state.blocks_buf[..pb_floats]
                        .copy_from_slice(&state.prelude_blocks_buf[..pb_floats]);
                }
                for ip in cs..ce {
                    let il = if pass == 0 {
                        ip
                    } else {
                        self.cfg.n_layers_phys + (pass - 1) * (ce - cs) + (ip - cs)
                    };
                    if pass > 0 {
                        let tc = Instant::now();
                        let loop_w = &self.loop_layers[ip - cs];
                        mix_depth_into(
                            &state.bank_buf[..state.bank_len * n_embd],
                            state.bank_len,
                            &state.bank_out,
                            &loop_w.msg_query,
                            LOOP_CTRL_COUNT,
                            n_embd,
                            &mut state.mix_states_hat,
                            &mut state.mix_scores,
                            &mut state.messages,
                        );
                        prof.loop_ctrl_ns += tc.elapsed().as_nanos() as u64;
                    }
                    self.run_layer_single_profiled(
                        il,
                        ip,
                        pos,
                        tau,
                        pass > 0,
                        state,
                        pool,
                        kt,
                        prof,
                    );
                }
            }
            for il in ce..self.cfg.n_layers_phys {
                self.run_layer_single_profiled(il, il, pos, tau, false, state, pool, kt, prof);
            }
        }

        let t_out = Instant::now();
        mix_depth_into(
            &state.blocks_buf[..state.blocks_len * n_embd],
            state.blocks_len,
            &state.cur,
            &self.output_res_query,
            1,
            n_embd,
            &mut state.mix_states_hat,
            &mut state.mix_scores,
            &mut state.inp_sa,
        );
        rms_norm(
            &mut state.last_hidden,
            &state.inp_sa,
            &self.output_norm,
            self.cfg.rms_eps,
        );
        let vp = self.cfg.vocab_padded;
        let vu = self.cfg.vocab_size;
        matvec_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes(),
            &state.last_hidden,
            &mut state.mv_scratch,
            &mut state.logits_padded[..vp],
            vp,
            n_embd,
        );
        state.logits.copy_from_slice(&state.logits_padded[..vu]);
        if let Some(cap) = self.cfg.final_logit_softcapping {
            softcap_inplace_parallel(pool, &mut state.logits, cap);
        }
        prof.lm_head_ns += t_out.elapsed().as_nanos() as u64;
        prof.lm_head_bytes += self.output.bytes().len() as u64;
        prof.n_steps += 1;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer_single_profiled(
        &self,
        il: usize,
        ip: usize,
        pos: usize,
        tau: f32,
        has_messages: bool,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        prof: &mut Inkling2DecodeProfile,
    ) {
        use std::time::Instant;
        let n_embd = self.cfg.n_embd;
        let swa = self.cfg.is_swa[ip];
        let nh = self.cfg.n_head(ip);
        let nh_kv = self.cfg.n_head_kv(ip);
        let hd = self.cfg.head_dim;
        let q_dim = nh * hd;
        let kv_dim = nh_kv * hd;
        let n_rel = self.cfg.n_relative;
        let n_conv = self.cfg.shortconv_l_cache;
        let extent = self.cfg.rel_extent_for(ip);
        let n_slots = self.cfg.n_layers;
        let layer = &self.layers[ip];

        let t_mix = Instant::now();
        state.inp_sa.copy_from_slice(&state.cur);
        if let Some(ref rq) = layer.attn_res_query {
            if swa {
                mix_depth_into(
                    &state.blocks_buf[..state.blocks_len * n_embd],
                    state.blocks_len,
                    &state.cur,
                    rq,
                    1,
                    n_embd,
                    &mut state.mix_states_hat,
                    &mut state.mix_scores,
                    &mut state.inputs[..n_embd],
                );
                let (first, rest) = state.inputs.split_at_mut(n_embd);
                rest[..n_embd].copy_from_slice(first);
                rest[n_embd..2 * n_embd].copy_from_slice(first);
            } else {
                mix_depth_into(
                    &state.blocks_buf[..state.blocks_len * n_embd],
                    state.blocks_len,
                    &state.cur,
                    rq,
                    3,
                    n_embd,
                    &mut state.mix_states_hat,
                    &mut state.mix_scores,
                    &mut state.inputs[..3 * n_embd],
                );
            }
        } else {
            for j in 0..3 {
                state.inputs[j * n_embd..(j + 1) * n_embd].copy_from_slice(&state.cur);
            }
        }
        prof.embed_mix_ns += t_mix.elapsed().as_nanos() as u64;

        if has_messages {
            let tc = Instant::now();
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            for j in 0..3 {
                control_inplace(
                    &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                    &state.messages[j * n_embd..(j + 1) * n_embd],
                    loop_w,
                    j,
                    n_embd,
                    self.cfg.gate_rank,
                    &mut state.ctrl_in,
                    &mut state.ctrl_g,
                    &mut state.ctrl_gain,
                    &mut state.ctrl_forget,
                );
            }
            prof.loop_ctrl_ns += tc.elapsed().as_nanos() as u64;
        }

        let t_proj = Instant::now();
        if swa && has_messages {
            let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
            for j in 0..3 {
                state.conv_tmp[..n_embd]
                    .copy_from_slice(&state.inputs[j * n_embd..(j + 1) * n_embd]);
                rms_norm(
                    &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                    &state.conv_tmp[..n_embd],
                    an,
                    self.cfg.rms_eps,
                );
            }
        } else if swa {
            let an = layer.attn_norm.as_ref().expect("SWA attn_norm");
            state.conv_tmp[..n_embd].copy_from_slice(&state.inputs[..n_embd]);
            rms_norm(
                &mut state.inputs[..n_embd],
                &state.conv_tmp[..n_embd],
                an,
                self.cfg.rms_eps,
            );
            let (first, rest) = state.inputs.split_at_mut(n_embd);
            rest[..n_embd].copy_from_slice(first);
            rest[n_embd..2 * n_embd].copy_from_slice(first);
        } else {
            let norms = [
                layer.attn_q_inp_norm.as_ref().expect("q_inp_norm"),
                layer.attn_k_inp_norm.as_ref().expect("k_inp_norm"),
                layer.attn_v_inp_norm.as_ref().expect("v_inp_norm"),
            ];
            for (j, nw) in norms.iter().enumerate() {
                state.conv_tmp[..n_embd]
                    .copy_from_slice(&state.inputs[j * n_embd..(j + 1) * n_embd]);
                rms_norm(
                    &mut state.inputs[j * n_embd..(j + 1) * n_embd],
                    &state.conv_tmp[..n_embd],
                    nw,
                    self.cfg.rms_eps,
                );
            }
        }

        matvec_f32(
            pool,
            kt,
            layer.wq.dtype,
            layer.wq.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.q_buf[..q_dim],
            q_dim,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.wk.dtype,
            layer.wk.bytes(),
            &state.inputs[n_embd..2 * n_embd],
            &mut state.mv_scratch,
            &mut state.k_buf[..kv_dim],
            kv_dim,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.wv.dtype,
            layer.wv.bytes(),
            &state.inputs[2 * n_embd..3 * n_embd],
            &mut state.mv_scratch,
            &mut state.v_buf[..kv_dim],
            kv_dim,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.attn_rel.dtype,
            layer.attn_rel.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.r_buf[..nh * n_rel],
            nh * n_rel,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.attn_sink_proj.dtype,
            layer.attn_sink_proj.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.sinks[..nh],
            nh,
            n_embd,
        );
        matvec_f32(
            pool,
            kt,
            layer.wqkv_gate.dtype,
            layer.wqkv_gate.bytes(),
            &state.inputs[..n_embd],
            &mut state.mv_scratch,
            &mut state.gate_buf[..q_dim],
            q_dim,
            n_embd,
        );
        let mut attn_b = (layer.wq.bytes().len()
            + layer.wk.bytes().len()
            + layer.wv.bytes().len()
            + layer.attn_rel.bytes().len()
            + layer.attn_sink_proj.bytes().len()
            + layer.wqkv_gate.bytes().len()
            + layer.wo.bytes().len()) as u64;
        if !swa {
            let kp = layer.attn_k_conv_proj.as_ref().expect("k_conv_proj");
            let vp = layer.attn_v_conv_proj.as_ref().expect("v_conv_proj");
            attn_b += (kp.bytes().len() + vp.bytes().len()) as u64;
            matvec_f32(
                pool,
                kt,
                kp.dtype,
                kp.bytes(),
                &state.inputs[n_embd..2 * n_embd],
                &mut state.mv_scratch,
                &mut state.k_dyn[..n_conv * nh_kv],
                n_conv * nh_kv,
                n_embd,
            );
            matvec_f32(
                pool,
                kt,
                vp.dtype,
                vp.bytes(),
                &state.inputs[2 * n_embd..3 * n_embd],
                &mut state.mv_scratch,
                &mut state.v_dyn[..n_conv * nh_kv],
                n_conv * nh_kv,
                n_embd,
            );
        }
        prof.dense_attn_bytes += attn_b;
        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            convolve_inplace(
                &mut state.k_buf[..kv_dim],
                &layer.attn_k_conv,
                if swa {
                    None
                } else {
                    Some(&state.k_dyn[..n_conv * nh_kv])
                },
                conv_hist,
                0,
                n_conv,
                &mut state.conv_tmp[..kv_dim],
            );
            convolve_inplace(
                &mut state.v_buf[..kv_dim],
                &layer.attn_v_conv,
                if swa {
                    None
                } else {
                    Some(&state.v_dyn[..n_conv * nh_kv])
                },
                conv_hist,
                kv_dim,
                n_conv,
                &mut state.conv_tmp[..kv_dim],
            );
        }
        clip_heads_inplace(
            &mut state.q_buf[..q_dim],
            nh,
            hd,
            Some(&layer.attn_q_clip),
            self.cfg.qk_clip,
        );
        clip_heads_inplace(
            &mut state.k_buf[..kv_dim],
            nh_kv,
            hd,
            None,
            self.cfg.qk_clip,
        );
        for h in 0..nh_kv {
            rms_norm_unit_inplace(&mut state.v_buf[h * hd..(h + 1) * hd], self.cfg.rms_eps);
        }
        clip_heads_inplace(
            &mut state.r_buf[..nh * n_rel],
            nh,
            n_rel,
            Some(&layer.attn_rel_clip),
            self.cfg.relative_clip,
        );
        if !swa {
            for v in &mut state.q_buf[..q_dim] {
                *v *= tau;
            }
            for v in &mut state.r_buf[..nh * n_rel] {
                *v *= tau;
            }
        }
        prof.dense_attn_proj_ns += t_proj.elapsed().as_nanos() as u64;

        let t_gqa = Instant::now();
        state.layers[il].store_kv_at(kt, pos, &state.k_buf[..kv_dim], &state.v_buf[..kv_dim]);
        let tbl_cols = extent + 1;
        for h in 0..nh {
            let r_h = &state.r_buf[h * n_rel..(h + 1) * n_rel];
            let sink_h = state.sinks[h];
            let row_out = &mut state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
            for (d, dst_d) in row_out[..extent].iter_mut().enumerate() {
                let b_row = &layer.attn_rel_bias[d * n_rel..(d + 1) * n_rel];
                let mut acc = 0.0f32;
                for r in 0..n_rel {
                    acc = r_h[r].mul_add(b_row[r], acc);
                }
                *dst_d = acc - sink_h;
            }
            row_out[extent] = -sink_h;
        }
        let kv_start = if swa {
            (pos + 1).saturating_sub(self.cfg.n_swa)
        } else {
            0
        };
        let kv_len = pos + 1 - kv_start;
        let gqa_ratio = nh / nh_kv;
        let attn_scale = 1.0 / (hd as f32).sqrt();
        state.attn_vec[..q_dim].fill(0.0);
        match &state.layers[il] {
            LayerState::Kv {
                kv_type: KvType::F16,
                k_cache,
                v_cache,
                ..
            } => {
                for h in 0..nh {
                    let kvh = h / gqa_ratio;
                    let q_h = &state.q_buf[h * hd..(h + 1) * hd];
                    let tbl_h = &state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                    let sc_h = &mut state.scores[..kv_len];
                    let mut max_s = 0.0f32;
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let k_off = kv_pos * kv_dim + kvh * hd;
                        let dot = (kt.dot_f32_f16)(q_h, &k_cache[k_off..k_off + hd]);
                        let dist = pos - kv_pos;
                        let rel_b = tbl_h[dist.min(extent)];
                        let s = dot.mul_add(attn_scale, rel_b);
                        sc_h[idx] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut denom = (-max_s).exp();
                    for s in sc_h.iter_mut() {
                        let e = (*s - max_s).exp();
                        *s = e;
                        denom += e;
                    }
                    let inv_denom = 1.0 / denom;
                    let out_h = &mut state.attn_vec[h * hd..(h + 1) * hd];
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let p = sc_h[idx] * inv_denom;
                        let v_off = kv_pos * kv_dim + kvh * hd;
                        (kt.axpy_f32_f16)(out_h, p, &v_cache[v_off..v_off + hd]);
                    }
                }
            }
            LayerState::Kv {
                kv_type: KvType::Q8_0,
                k_q8,
                v_q8,
                ..
            } => {
                let kv_blks = kv_dim / QK8_0;
                let hd_blks = hd / QK8_0;
                for h in 0..nh {
                    let kvh = h / gqa_ratio;
                    let q_h = &state.q_buf[h * hd..(h + 1) * hd];
                    let tbl_h = &state.rel_table[h * tbl_cols..(h + 1) * tbl_cols];
                    let sc_h = &mut state.scores[..kv_len];
                    let mut max_s = 0.0f32;
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let k_off = kv_pos * kv_blks + kvh * hd_blks;
                        let dot = (kt.dot_f32_q8_0)(q_h, &k_q8[k_off..k_off + hd_blks]);
                        let dist = pos - kv_pos;
                        let rel_b = tbl_h[dist.min(extent)];
                        let s = dot.mul_add(attn_scale, rel_b);
                        sc_h[idx] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut denom = (-max_s).exp();
                    for s in sc_h.iter_mut() {
                        let e = (*s - max_s).exp();
                        *s = e;
                        denom += e;
                    }
                    let inv_denom = 1.0 / denom;
                    let out_h = &mut state.attn_vec[h * hd..(h + 1) * hd];
                    for (idx, kv_pos) in (kv_start..=pos).enumerate() {
                        let p = sc_h[idx] * inv_denom;
                        let v_off = kv_pos * kv_blks + kvh * hd_blks;
                        (kt.axpy_f32_q8_0)(out_h, p, &v_q8[v_off..v_off + hd_blks]);
                    }
                }
            }
            _ => unreachable!(),
        }
        prof.gqa_kv_ns += t_gqa.elapsed().as_nanos() as u64;

        let t_wo = Instant::now();
        for i in 0..q_dim {
            let sig = 1.0 / (1.0 + (-state.gate_buf[i]).exp());
            state.attn_vec[i] *= sig;
        }
        matvec_f32(
            pool,
            kt,
            layer.wo.dtype,
            layer.wo.bytes(),
            &state.attn_vec[..q_dim],
            &mut state.mv_scratch,
            &mut state.wo_out[..n_embd],
            n_embd,
            q_dim,
        );
        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            convolve_inplace(
                &mut state.wo_out[..n_embd],
                &layer.attn_post_conv,
                None,
                conv_hist,
                2 * kv_dim,
                n_conv,
                &mut state.conv_tmp[..n_embd],
            );
        }
        if ip.is_multiple_of(self.cfg.attn_res_block_size) {
            let b_off = state.blocks_len * n_embd;
            state.blocks_buf[b_off..b_off + n_embd].copy_from_slice(&state.inp_sa);
            state.blocks_len += 1;
            state.cur.copy_from_slice(&state.wo_out[..n_embd]);
        } else {
            for c in 0..n_embd {
                state.cur[c] = state.inp_sa[c] + state.wo_out[c];
            }
        }
        state.ffn_inp.copy_from_slice(&state.cur);
        prof.dense_attn_proj_ns += t_wo.elapsed().as_nanos() as u64;

        let t_fmix = Instant::now();
        mix_depth_into(
            &state.blocks_buf[..state.blocks_len * n_embd],
            state.blocks_len,
            &state.cur,
            &layer.ffn_res_query,
            1,
            n_embd,
            &mut state.mix_states_hat,
            &mut state.mix_scores,
            &mut state.inp_sa,
        );
        prof.embed_mix_ns += t_fmix.elapsed().as_nanos() as u64;

        if has_messages {
            let tc = Instant::now();
            let loop_w = &self.loop_layers[ip - self.cfg.core_start];
            control_inplace(
                &mut state.inp_sa,
                &state.messages[LOOP_CTRL_FFN * n_embd..(LOOP_CTRL_FFN + 1) * n_embd],
                loop_w,
                LOOP_CTRL_FFN,
                n_embd,
                self.cfg.gate_rank,
                &mut state.ctrl_in,
                &mut state.ctrl_g,
                &mut state.ctrl_gain,
                &mut state.ctrl_forget,
            );
            prof.loop_ctrl_ns += tc.elapsed().as_nanos() as u64;
        }

        let t_rt = Instant::now();
        rms_norm(
            &mut state.wo_out[..n_embd],
            &state.inp_sa,
            &layer.ffn_norm,
            self.cfg.rms_eps,
        );
        self.route_moe_single(
            ip,
            &state.wo_out[..n_embd],
            pool,
            kt,
            &mut state.mv_scratch,
            &mut state.router_logits,
            &mut state.router_probs,
            &mut state.router_sel,
            &mut state.top_ids,
            &mut state.top_weights,
        );
        prof.moe_route_ns += t_rt.elapsed().as_nanos() as u64;

        let t_moe = Instant::now();
        self.eval_moe_experts_inner(
            ip,
            &state.wo_out[..n_embd],
            &state.top_ids,
            &state.top_weights,
            pool,
            Some(&state.io_pool),
            &state.serial_pool,
            kt,
            &mut state.expert_slots,
            &mut state.mv_scratch,
            &mut state.moe_out,
            false,
        );
        let moe_ns = t_moe.elapsed().as_nanos() as u64;
        let k_used = self.cfg.n_expert_used as u64;
        match &layer.experts {
            ExpertLayerStore::InMemory {
                gate_up_expert_bytes,
                down_expert_bytes,
                ..
            } => {
                prof.moe_ram_ns += moe_ns;
                prof.moe_ram_bytes += k_used * (*gate_up_expert_bytes + *down_expert_bytes) as u64;
            }
            ExpertLayerStore::NvmeDirect {
                gate_up_expert_bytes,
                down_expert_bytes,
                ..
            } => {
                prof.moe_ssd_ns += moe_ns;
                prof.moe_ssd_bytes += k_used * (*gate_up_expert_bytes + *down_expert_bytes) as u64;
            }
            ExpertLayerStore::DeviceVram => {}
        }

        let t_pc = Instant::now();
        {
            let conv_hist = shortconv_slice_mut(&mut state.layers[n_slots + il]);
            convolve_inplace(
                &mut state.moe_out[..n_embd],
                &layer.ffn_post_conv,
                None,
                conv_hist,
                2 * kv_dim + n_embd,
                n_conv,
                &mut state.conv_tmp[..n_embd],
            );
        }
        for c in 0..n_embd {
            state.cur[c] = state.ffn_inp[c] + state.moe_out[c];
        }
        prof.moe_route_ns += t_pc.elapsed().as_nanos() as u64;
    }

    /// Commits the first `committed` (`1 <= committed <= last_window_len`) verified tokens in O(1) memcpy.
    pub fn commit_speculative(
        &self,
        state: &mut Inkling2State,
        spec_buf: &SpecVerifyBuf,
        committed: usize,
    ) {
        assert!(committed >= 1 && committed <= spec_buf.last_window_len);
        let vocab = self.cfg.vocab_size;
        let n_embd = self.cfg.n_embd;
        let conv_total = spec_buf.tmp_conv_hist.len();
        let idx = committed - 1;
        state.unpack_shortconv_from(&spec_buf.sc_bx_win[idx * conv_total..(idx + 1) * conv_total]);
        state
            .logits
            .copy_from_slice(&spec_buf.logits_win[idx * vocab..(idx + 1) * vocab]);
        state
            .last_hidden
            .copy_from_slice(&spec_buf.hidden_win[idx * n_embd..(idx + 1) * n_embd]);
    }

    /// Projects `hidden` (`[n_embd]` post-`output_norm`) through `output.weight` + optional soft-cap,
    /// dropping padded rows `[vocab_size..vocab_padded)`.
    pub fn project_logits_from_hidden(
        &self,
        state: &mut Inkling2State,
        pool: &ThreadPool,
        kt: &KernelTable,
        hidden: &[f32],
        out_logits: &mut [f32],
    ) {
        let vp = self.cfg.vocab_padded;
        let vu = self.cfg.vocab_size;
        let _ = matmul_f32;
        matvec_f32(
            pool,
            kt,
            self.output.dtype,
            self.output.bytes(),
            hidden,
            &mut state.mv_scratch,
            &mut state.logits_padded[..vp],
            vp,
            self.cfg.n_embd,
        );
        state.logits.copy_from_slice(&state.logits_padded[..vu]);
        if let Some(cap) = self.cfg.final_logit_softcapping {
            softcap_inplace_parallel(pool, &mut state.logits, cap);
        }
        if !out_logits.is_empty() {
            out_logits[..vu].copy_from_slice(&state.logits);
        }
    }
}

impl Inkling2State {
    pub fn new(model: &Inkling2Model<'_>, max_ctx: usize) -> Self {
        Self::new_with_kv_type(model, max_ctx, KvType::F16)
    }

    pub fn new_with_kv_type(model: &Inkling2Model<'_>, max_ctx: usize, kv_type: KvType) -> Self {
        let cfg = &model.cfg;
        let n_slots = cfg.n_layers;
        let hist_len = cfg.shortconv_l_cache.saturating_sub(1);

        let mut layers = Vec::with_capacity(2 * n_slots);
        for il in 0..n_slots {
            let ip = cfg.layer_phys(il);
            let kv_dim = cfg.kv_dim(ip);
            let win = if cfg.is_swa[ip] {
                Some(cfg.n_swa)
            } else {
                None
            };
            layers.push(LayerState::new_kv(max_ctx, kv_dim, win, kv_type));
        }
        for il in 0..n_slots {
            let ip = cfg.layer_phys(il);
            let ch = 2 * cfg.kv_dim(ip) + 2 * cfg.n_embd;
            layers.push(LayerState::ShortConv {
                conv_hist: vec![0.0f32; ch * hist_len],
            });
        }

        let max_q_dim = cfg.max_n_head() * cfg.head_dim;
        let max_kv_dim = cfg.max_n_head_kv() * cfg.head_dim;
        let max_blocks = cfg.max_depth_blocks();
        let max_ext = cfg.max_rel_extent();
        let max_cols = cfg
            .n_embd
            .max(max_q_dim)
            .max(2 * cfg.n_ff_exp)
            .max(2 * cfg.n_embd);

        let need_direct_slots = model.stats.nvme_direct_expert_layers > 0;
        let slot_cap = if need_direct_slots {
            model.max_expert_chunk_bytes + 8192
        } else {
            4096
        };
        let n_exp_slots = cfg.n_expert_used.max(8);
        let mut expert_slots = Vec::with_capacity(n_exp_slots);
        for _ in 0..n_exp_slots {
            expert_slots.push(ExpertWorkSlot {
                slot: PageAlignedSlot::new(slot_cap),
                mv_scratch: MatvecScratch::new(cfg.n_embd.max(cfg.n_ff_exp)),
                gate_up_buf: vec![0.0; 2 * cfg.n_ff_exp],
                hidden_buf: vec![0.0; cfg.n_ff_exp],
                win_in_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
                win_gate_up_buf: vec![0.0; MAX_INKLING2_WINDOW * 2 * cfg.n_ff_exp],
                win_hidden_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_ff_exp],
                out_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            });
        }

        Self {
            max_ctx,
            layers,
            cur: vec![0.0; cfg.n_embd],
            inp_sa: vec![0.0; cfg.n_embd],
            ffn_inp: vec![0.0; cfg.n_embd],
            prelude: vec![0.0; cfg.n_embd],
            bank_out: vec![0.0; cfg.n_embd],
            blocks_buf: vec![0.0; max_blocks * cfg.n_embd],
            blocks_len: 0,
            prelude_blocks_buf: vec![0.0; max_blocks * cfg.n_embd],
            prelude_blocks_len: 0,
            bank_buf: vec![0.0; max_blocks * cfg.n_embd],
            bank_len: 0,
            mix_states_hat: vec![0.0; (max_blocks + 1) * cfg.n_embd],
            mix_scores: vec![0.0; max_blocks + 1],
            inputs: vec![0.0; 3 * cfg.n_embd],
            messages: vec![0.0; LOOP_CTRL_COUNT * cfg.n_embd],
            ctrl_in: vec![0.0; 2 * cfg.n_embd],
            ctrl_g: vec![0.0; cfg.gate_rank.max(1)],
            ctrl_gain: vec![0.0; cfg.n_embd],
            ctrl_forget: vec![0.0; cfg.n_embd],
            q_buf: vec![0.0; max_q_dim],
            k_buf: vec![0.0; max_kv_dim],
            v_buf: vec![0.0; max_kv_dim],
            r_buf: vec![0.0; cfg.max_n_head() * cfg.n_relative],
            k_dyn: vec![0.0; cfg.shortconv_l_cache * cfg.max_n_head_kv()],
            v_dyn: vec![0.0; cfg.shortconv_l_cache * cfg.max_n_head_kv()],
            conv_tmp: vec![0.0; cfg.n_embd.max(max_kv_dim)],
            sinks: vec![0.0; cfg.max_n_head()],
            rel_table: vec![0.0; cfg.max_n_head() * (max_ext + 1)],
            scores: vec![0.0; max_ctx.max(1)],
            attn_vec: vec![0.0; max_q_dim],
            gate_buf: vec![0.0; max_q_dim],
            wo_out: vec![0.0; cfg.n_embd],
            router_logits: vec![0.0; cfg.n_expert],
            router_probs: vec![0.0; cfg.n_expert],
            router_sel: vec![0.0; cfg.n_expert],
            top_ids: vec![0; cfg.n_expert_used],
            top_weights: vec![0.0; cfg.n_expert_used],
            moe_out: vec![0.0; cfg.n_embd],
            logits_padded: vec![0.0; MAX_SPEC_WINDOW * cfg.vocab_padded],
            logits: vec![0.0; cfg.vocab_size],
            last_hidden: vec![0.0; cfg.n_embd],
            slot_outputs: vec![0.0; (n_slots + 1) * cfg.n_embd],
            mv_scratch: MatvecScratch::new(max_cols / MAX_CPU_BATCH.max(1) + max_cols),
            serial_pool: ThreadPool::new(1),
            io_pool: ThreadPool::new(if need_direct_slots {
                cfg.n_expert_used.clamp(1, 8)
            } else {
                1
            }),
            expert_slots,
            win_slot_outs: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_expert_used * cfg.n_embd],
            win_cur: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_inp_sa: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_ffn_inp: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_prelude: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_bank_out: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_blocks_buf: vec![0.0; MAX_INKLING2_WINDOW * max_blocks * cfg.n_embd],
            win_prelude_blocks_buf: vec![0.0; MAX_INKLING2_WINDOW * max_blocks * cfg.n_embd],
            win_bank_buf: vec![0.0; MAX_INKLING2_WINDOW * max_blocks * cfg.n_embd],
            win_messages: vec![0.0; MAX_INKLING2_WINDOW * LOOP_CTRL_COUNT * cfg.n_embd],
            win_q_in: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_k_in: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_v_in: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_q_buf: vec![0.0; MAX_INKLING2_WINDOW * max_q_dim],
            win_k_buf: vec![0.0; MAX_INKLING2_WINDOW * max_kv_dim],
            win_v_buf: vec![0.0; MAX_INKLING2_WINDOW * max_kv_dim],
            win_r_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.max_n_head() * cfg.n_relative],
            win_sinks: vec![0.0; MAX_INKLING2_WINDOW * cfg.max_n_head()],
            win_gate_buf: vec![0.0; MAX_INKLING2_WINDOW * max_q_dim],
            win_k_dyn: vec![0.0; MAX_INKLING2_WINDOW * cfg.shortconv_l_cache * cfg.max_n_head_kv()],
            win_v_dyn: vec![0.0; MAX_INKLING2_WINDOW * cfg.shortconv_l_cache * cfg.max_n_head_kv()],
            win_attn_vec: vec![0.0; MAX_INKLING2_WINDOW * max_q_dim],
            win_wo_out: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_moe_out: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            win_top_ids: vec![0; MAX_INKLING2_WINDOW * cfg.n_expert_used],
            win_top_weights: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_expert_used],
        }
    }

    /// Lightweight host MoE scratch state (`0` CPU KV/ShortConv caches allocated, saving `~220 MiB`
    /// when `Inkling2VkRunner` owns KV/ShortConv in GPU VRAM).
    pub fn new_host_moe_only(model: &Inkling2Model<'_>) -> Self {
        let cfg = &model.cfg;
        let need_direct_slots = model.stats.nvme_direct_expert_layers > 0;
        let slot_cap = if need_direct_slots {
            model.max_expert_chunk_bytes + 8192
        } else {
            4096
        };
        let n_exp_slots = cfg.n_expert_used.max(8);
        let mut expert_slots = Vec::with_capacity(n_exp_slots);
        for _ in 0..n_exp_slots {
            expert_slots.push(ExpertWorkSlot {
                slot: PageAlignedSlot::new(slot_cap),
                mv_scratch: MatvecScratch::new(cfg.n_embd.max(cfg.n_ff_exp)),
                gate_up_buf: vec![0.0; 2 * cfg.n_ff_exp],
                hidden_buf: vec![0.0; cfg.n_ff_exp],
                win_in_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
                win_gate_up_buf: vec![0.0; MAX_INKLING2_WINDOW * 2 * cfg.n_ff_exp],
                win_hidden_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_ff_exp],
                out_buf: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_embd],
            });
        }
        Self {
            max_ctx: 0,
            layers: Vec::new(),
            cur: Vec::new(),
            inp_sa: Vec::new(),
            ffn_inp: Vec::new(),
            prelude: Vec::new(),
            bank_out: Vec::new(),
            blocks_buf: Vec::new(),
            blocks_len: 0,
            prelude_blocks_buf: Vec::new(),
            prelude_blocks_len: 0,
            bank_buf: Vec::new(),
            bank_len: 0,
            mix_states_hat: Vec::new(),
            mix_scores: Vec::new(),
            inputs: Vec::new(),
            messages: Vec::new(),
            ctrl_in: Vec::new(),
            ctrl_g: Vec::new(),
            ctrl_gain: Vec::new(),
            ctrl_forget: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            r_buf: Vec::new(),
            k_dyn: Vec::new(),
            v_dyn: Vec::new(),
            conv_tmp: Vec::new(),
            sinks: Vec::new(),
            rel_table: Vec::new(),
            scores: Vec::new(),
            attn_vec: Vec::new(),
            gate_buf: Vec::new(),
            wo_out: Vec::new(),
            router_logits: Vec::new(),
            router_probs: Vec::new(),
            router_sel: Vec::new(),
            top_ids: Vec::new(),
            top_weights: Vec::new(),
            moe_out: Vec::new(),
            logits_padded: Vec::new(),
            logits: Vec::new(),
            last_hidden: Vec::new(),
            slot_outputs: Vec::new(),
            mv_scratch: MatvecScratch::new(cfg.n_embd.max(cfg.n_ff_exp)),
            serial_pool: ThreadPool::new(1),
            io_pool: ThreadPool::new(if need_direct_slots {
                cfg.n_expert_used.clamp(1, 8)
            } else {
                1
            }),
            expert_slots,
            win_slot_outs: vec![0.0; MAX_INKLING2_WINDOW * cfg.n_expert_used * cfg.n_embd],
            win_cur: Vec::new(),
            win_inp_sa: Vec::new(),
            win_ffn_inp: Vec::new(),
            win_prelude: Vec::new(),
            win_bank_out: Vec::new(),
            win_blocks_buf: Vec::new(),
            win_prelude_blocks_buf: Vec::new(),
            win_bank_buf: Vec::new(),
            win_messages: Vec::new(),
            win_q_in: Vec::new(),
            win_k_in: Vec::new(),
            win_v_in: Vec::new(),
            win_q_buf: Vec::new(),
            win_k_buf: Vec::new(),
            win_v_buf: Vec::new(),
            win_r_buf: Vec::new(),
            win_sinks: Vec::new(),
            win_gate_buf: Vec::new(),
            win_k_dyn: Vec::new(),
            win_v_dyn: Vec::new(),
            win_attn_vec: Vec::new(),
            win_wo_out: Vec::new(),
            win_moe_out: Vec::new(),
            win_top_ids: Vec::new(),
            win_top_weights: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        for ls in &mut self.layers {
            ls.reset();
        }
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn mv_scratch_mut(&mut self) -> &mut MatvecScratch {
        &mut self.mv_scratch
    }

    pub fn host_moe_parts_mut(
        &mut self,
    ) -> (&ThreadPool, &mut [ExpertWorkSlot], &mut MatvecScratch) {
        (
            &self.serial_pool,
            &mut self.expert_slots,
            &mut self.mv_scratch,
        )
    }

    pub fn host_moe_window_parts_mut(
        &mut self,
    ) -> (
        &ThreadPool,
        &mut [ExpertWorkSlot],
        &mut MatvecScratch,
        &mut [f32],
    ) {
        (
            &self.serial_pool,
            &mut self.expert_slots,
            &mut self.mv_scratch,
            &mut self.win_slot_outs,
        )
    }

    fn pack_shortconv_into(&self, dst: &mut [f32]) {
        let n_slots = self.layers.len() / 2;
        let mut cur = 0usize;
        for ls in &self.layers[n_slots..] {
            if let LayerState::ShortConv { conv_hist } = ls {
                dst[cur..cur + conv_hist.len()].copy_from_slice(conv_hist);
                cur += conv_hist.len();
            }
        }
        debug_assert_eq!(cur, dst.len());
    }

    fn unpack_shortconv_from(&mut self, src: &[f32]) {
        let n_slots = self.layers.len() / 2;
        let mut cur = 0usize;
        for ls in &mut self.layers[n_slots..] {
            if let LayerState::ShortConv { conv_hist } = ls {
                let len = conv_hist.len();
                conv_hist.copy_from_slice(&src[cur..cur + len]);
                cur += len;
            }
        }
        debug_assert_eq!(cur, src.len());
    }
}

#[inline]
fn shortconv_slice_mut(ls: &mut LayerState) -> &mut [f32] {
    match ls {
        LayerState::ShortConv { conv_hist } => conv_hist.as_mut_slice(),
        _ => unreachable!("expected LayerState::ShortConv"),
    }
}

#[inline]
pub fn apply_situ_glu(gate: &[f32], up: &[f32], out: &mut [f32], beta: f32, lb: f32) {
    let inv_beta = 1.0 / beta;
    let inv_lb = if lb > 0.0 { 1.0 / lb } else { 0.0 };
    for i in 0..gate.len() {
        let g = gate[i];
        let u = up[i];
        let act_g = beta * (g * inv_beta).tanh() * (1.0 / (1.0 + (-g).exp()));
        let act_u = if lb > 0.0 {
            lb * (u * inv_lb).tanh()
        } else {
            u
        };
        out[i] = act_g * act_u;
    }
}

/// Depth-residual attention (`inkling2.cpp:179-192`):
/// Normalizes `[blocks.rev()..., cur]` with unit RMSNorm (`eps = 1e-6`), scores against `n_queries`
/// query vectors (`[n_queries, n_embd]`) scaled by `1 / sqrt(n_embd)`, applies softmax, and returns
/// the weighted sum of normalized states in `out` (`[n_queries, n_embd]`).
#[allow(clippy::too_many_arguments)]
pub fn mix_depth_into(
    blocks_flat: &[f32],
    n_blocks: usize,
    cur: &[f32],
    queries: &[f32],
    n_queries: usize,
    n_embd: usize,
    states_hat: &mut [f32],
    scores: &mut [f32],
    out: &mut [f32],
) {
    let m = n_blocks + 1;
    // Match `inkling2.cpp:180-183` row order: `[blocks[n_blocks-1], ..., blocks[0], cur]`
    for r in 0..n_blocks {
        let src_blk = n_blocks - 1 - r;
        let dst = &mut states_hat[r * n_embd..(r + 1) * n_embd];
        dst.copy_from_slice(&blocks_flat[src_blk * n_embd..(src_blk + 1) * n_embd]);
        rms_norm_unit_inplace(dst, 1e-6);
    }
    {
        let dst = &mut states_hat[n_blocks * n_embd..m * n_embd];
        dst.copy_from_slice(cur);
        rms_norm_unit_inplace(dst, 1e-6);
    }

    let scale = 1.0 / (n_embd as f32).sqrt();
    for q_idx in 0..n_queries {
        let q = &queries[q_idx * n_embd..(q_idx + 1) * n_embd];
        let sc = &mut scores[..m];
        let mut max_s = f32::NEG_INFINITY;
        for (r, sc_r) in sc.iter_mut().enumerate() {
            let st = &states_hat[r * n_embd..(r + 1) * n_embd];
            let mut dot = 0.0f32;
            for c in 0..n_embd {
                dot = st[c].mul_add(q[c], dot);
            }
            let s = dot * scale;
            *sc_r = s;
            if s > max_s {
                max_s = s;
            }
        }
        let mut denom = 0.0f32;
        for sc_r in sc.iter_mut() {
            let e = (*sc_r - max_s).exp();
            *sc_r = e;
            denom += e;
        }
        let inv_d = 1.0 / denom;
        let out_q = &mut out[q_idx * n_embd..(q_idx + 1) * n_embd];
        out_q.fill(0.0);
        for (r, &e) in sc.iter().enumerate() {
            let p = e * inv_d;
            let st = &states_hat[r * n_embd..(r + 1) * n_embd];
            for c in 0..n_embd {
                out_q[c] = p.mul_add(st[c], out_q[c]);
            }
        }
    }
}

/// Cross-pass gated controller (`inkling2.cpp:241-249`):
/// `(cur - tanh(W_forget * g) * cur) + tanh(W_gain * g) * msg`, where `g = W_down * [rms(cur); rms(msg)]`.
#[allow(clippy::too_many_arguments)]
pub fn control_inplace(
    cur: &mut [f32],
    msg: &[f32],
    loop_w: &Inkling2LoopWeights,
    ctrl: usize,
    n_embd: usize,
    rank: usize,
    ctrl_in: &mut [f32],
    ctrl_g: &mut [f32],
    ctrl_gain: &mut [f32],
    ctrl_forget: &mut [f32],
) {
    ctrl_in[..n_embd].copy_from_slice(cur);
    rms_norm_unit_inplace(&mut ctrl_in[..n_embd], 1e-6);
    ctrl_in[n_embd..2 * n_embd].copy_from_slice(msg);
    rms_norm_unit_inplace(&mut ctrl_in[n_embd..2 * n_embd], 1e-6);

    let w_down = &loop_w.ctrl_down[ctrl];
    let in_dim = 2 * n_embd;
    for r in 0..rank {
        let row = &w_down[r * in_dim..(r + 1) * in_dim];
        let mut acc = 0.0f32;
        for c in 0..in_dim {
            acc = row[c].mul_add(ctrl_in[c], acc);
        }
        ctrl_g[r] = acc;
    }

    let w_gain = &loop_w.ctrl_gain[ctrl];
    let w_forget = &loop_w.ctrl_forget[ctrl];
    for c in 0..n_embd {
        let rg = &w_gain[c * rank..(c + 1) * rank];
        let rf = &w_forget[c * rank..(c + 1) * rank];
        let mut g_acc = 0.0f32;
        let mut f_acc = 0.0f32;
        for r in 0..rank {
            g_acc = rg[r].mul_add(ctrl_g[r], g_acc);
            f_acc = rf[r].mul_add(ctrl_g[r], f_acc);
        }
        ctrl_gain[c] = g_acc.tanh();
        ctrl_forget[c] = f_acc.tanh();
    }

    for c in 0..n_embd {
        let x = cur[c];
        cur[c] = (x - ctrl_forget[c] * x) + ctrl_gain[c] * msg[c];
    }
}

/// Per-head L2 norm clip (`inkling2.cpp:165-176`): optional elementwise `scale`, then clamp L2 norm to `threshold`.
pub fn clip_heads_inplace(
    vec: &mut [f32],
    n_heads: usize,
    dim: usize,
    opt_scale: Option<&[f32]>,
    threshold: f32,
) {
    if let Some(sc) = opt_scale {
        for (v, &s) in vec.iter_mut().zip(sc.iter()) {
            *v *= s;
        }
    }
    for h in 0..n_heads {
        let head = &mut vec[h * dim..(h + 1) * dim];
        let mut sq = 0.0f32;
        for &v in head.iter() {
            sq = v.mul_add(v, sq);
        }
        let norm = sq.sqrt() + 1e-6;
        let factor = (threshold / norm).clamp(0.0, 1.0);
        if factor < 1.0 {
            for v in head.iter_mut() {
                *v *= factor;
            }
        }
    }
}

/// Causal depthwise ShortConv with residual & optional dynamic per-head taps (`inkling2.cpp:196-238`).
///
/// `conv_hist` stores `[hist_len = n_conv - 1]` floats per channel contiguously starting at channel
/// `chan_offset` (`conv_hist[(chan_offset + c) * hist_len + k]` for `k in 0..n_conv-1`, oldest to newest).
pub fn convolve_inplace(
    cur: &mut [f32],
    weight: &[f32],
    dynamic: Option<&[f32]>,
    conv_hist: &mut [f32],
    chan_offset: usize,
    n_conv: usize,
    out_tmp: &mut [f32],
) {
    let n_channels = cur.len();
    let hist_len = n_conv - 1;
    let base = chan_offset * hist_len;

    for c in 0..n_channels {
        let h_off = base + c * hist_len;
        let w_off = c * n_conv;
        let mut acc = 0.0f32;
        for k in 0..hist_len {
            acc = conv_hist[h_off + k].mul_add(weight[w_off + k], acc);
        }
        acc = cur[c].mul_add(weight[w_off + hist_len], acc);
        out_tmp[c] = acc;
    }

    if let Some(dyn_taps) = dynamic {
        let n_heads = dyn_taps.len() / n_conv;
        let ch_per_head = n_channels / n_heads;
        for h in 0..n_heads {
            let tap_h = &dyn_taps[h * n_conv..(h + 1) * n_conv];
            let c0 = h * ch_per_head;
            for c in c0..c0 + ch_per_head {
                let h_off = base + c * hist_len;
                // lag = 0 -> cur[c] (time t); lag in 1..n_conv -> conv_hist[h_off + (hist_len - lag)] (time t - lag)
                let mut dyn_acc = cur[c] * tap_h[0];
                for lag in 1..n_conv {
                    dyn_acc = conv_hist[h_off + (hist_len - lag)].mul_add(tap_h[lag], dyn_acc);
                }
                out_tmp[c] += dyn_acc;
            }
        }
    }

    // Shift history window left and insert un-convolved `cur[c]` at `hist_len - 1`
    for (c, &cur_c) in cur.iter().enumerate() {
        let h_off = base + c * hist_len;
        for k in 0..hist_len.saturating_sub(1) {
            conv_hist[h_off + k] = conv_hist[h_off + k + 1];
        }
        if hist_len > 0 {
            conv_hist[h_off + hist_len - 1] = cur_c;
        }
    }

    let mult = if dynamic.is_some() { 2 } else { 1 };
    let inv_scale = 1.0 / ((mult * n_conv + 1) as f32).sqrt();
    for (cur_c, &ot) in cur.iter_mut().zip(&out_tmp[..n_channels]) {
        *cur_c = (*cur_c + ot) * inv_scale;
    }
}
