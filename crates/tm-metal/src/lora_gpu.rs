//! Shared Apple Metal dynamic LoRA side-branch engine (`lora_down_f32`, `lora_up_axpy_f32`,
//! `lora_embed_f32`) used by all Metal runners (`llama`, `qwen3`, `qwen35`, `gemma4`, `lfm2`).
//!
//! Enforces `AGENTS.md` Rules #4, #6, and #9:
//! - Validates `LoraSet` against the base `GgufFile` **before** mutating runner state.
//! - Pre-allocates all GPU `A`, `B_T`, and `h_scratch` buffers at `bind_lora` time — zero heap
//!   or GPU allocations occur on the token generation path.
//! - Checks `maxTotalThreadsPerThreadgroup` on every pipeline at construction time.

use core::ffi::c_void;
use core::ptr::NonNull;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use tm_gguf::GgufFile;
use tm_lora::{BoundLora, LoraSet};

use crate::MetalError;

type MtlBuf = Retained<ProtocolObject<dyn MTLBuffer>>;
type MtlPipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct LoraLinearArgs {
    pub d_in: u32,
    pub d_out: u32,
    pub rank: u32,
    pub n_tok: u32,
    pub scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct LoraEmbedArgs {
    pub d_out: u32,
    pub rank: u32,
    pub n_tok: u32,
    pub token_id: u32,
    pub use_tok_buf: u32,
    pub scale: f32,
}

struct GpuLoraEntry {
    a_buf: MtlBuf,
    bt_buf: MtlBuf,
    rank: usize,
    d_in: usize,
    d_out: usize,
    scale: f32,
    is_token_embd: bool,
}

/// Reusable GPU LoRA side-branch encoder bound to a Metal runner.
pub struct MetalLoraEngine {
    pipe_down: MtlPipe,
    pipe_up: MtlPipe,
    pipe_embed: MtlPipe,
    chains: HashMap<String, Vec<GpuLoraEntry>>,
    h_scratch: Option<MtlBuf>,
}

fn build_lora_pipe(
    device: &ProtocolObject<dyn MTLDevice>,
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &'static str,
    max_tg: usize,
) -> Result<MtlPipe, MetalError> {
    let ns = NSString::from_str(name);
    let func = lib
        .newFunctionWithName(&ns)
        .ok_or_else(|| MetalError::FunctionNotFound(name.to_string()))?;
    let pipe = device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| MetalError::PipelineCreation {
            name: name.to_string(),
            reason: e.localizedDescription().to_string(),
        })?;
    let limit = pipe.maxTotalThreadsPerThreadgroup();
    if max_tg > limit {
        return Err(MetalError::ThreadgroupTooLarge {
            kernel: name,
            requested: max_tg,
            limit,
        });
    }
    Ok(pipe)
}

fn upload_f32_slice(
    device: &ProtocolObject<dyn MTLDevice>,
    vals: &[f32],
) -> Result<MtlBuf, MetalError> {
    let byte_len = core::mem::size_of_val(vals).max(16);
    let buf = device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
        .ok_or(MetalError::BufferAlloc { bytes: byte_len })?;
    if !vals.is_empty() {
        // SAFETY: `buf` is newly allocated with `byte_len >= vals.len() * 4` in `StorageModeShared`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                buf.contents().as_ptr() as *mut u8,
                core::mem::size_of_val(vals),
            );
        }
    }
    Ok(buf)
}

impl MetalLoraEngine {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        lib: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            pipe_down: build_lora_pipe(device, lib, "lora_down_f32", 128)?,
            pipe_up: build_lora_pipe(device, lib, "lora_up_axpy_f32", 256)?,
            pipe_embed: build_lora_pipe(device, lib, "lora_embed_f32", 256)?,
            chains: HashMap::new(),
            h_scratch: None,
        })
    }

    /// Validates `loras` against `gguf` and uploads all `(A, B_T)` buffers + `h_scratch` atomically.
    /// If validation or any allocation fails, existing state on `self` is untouched (`Rule #6`).
    pub fn bind(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        gguf: &GgufFile,
        loras: &LoraSet,
        max_batch: usize,
    ) -> Result<(), MetalError> {
        loras.validate_against_gguf(gguf)?;
        if loras.is_empty() {
            self.chains.clear();
            self.h_scratch = None;
            return Ok(());
        }

        let bound = BoundLora::from_lora_set(loras);
        if bound.is_empty() {
            self.chains.clear();
            self.h_scratch = None;
            return Ok(());
        }

        let scratch_bytes = (max_batch.max(1) * bound.max_rank.max(1) * 4).max(16);
        let new_scratch = device
            .newBufferWithLength_options(scratch_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAlloc {
                bytes: scratch_bytes,
            })?;

        let mut new_chains: HashMap<String, Vec<GpuLoraEntry>> =
            HashMap::with_capacity(bound.name_to_slot.len());
        for (name, &slot) in &bound.name_to_slot {
            let src_chain = &bound.chains[slot];
            let mut gpu_chain = Vec::with_capacity(src_chain.len());
            for entry in src_chain {
                let a_buf = upload_f32_slice(device, &entry.a)?;
                let bt_buf = upload_f32_slice(device, &entry.b_t)?;
                gpu_chain.push(GpuLoraEntry {
                    a_buf,
                    bt_buf,
                    rank: entry.rank,
                    d_in: entry.d_in,
                    d_out: entry.d_out,
                    scale: entry.scale,
                    is_token_embd: entry.is_token_embd,
                });
            }
            new_chains.insert(name.clone(), gpu_chain);
        }

        self.chains = new_chains;
        self.h_scratch = Some(new_scratch);
        Ok(())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    #[inline]
    pub fn has(&self, tensor_name: &str) -> bool {
        !self.chains.is_empty() && self.chains.contains_key(tensor_name)
    }

    /// Encodes `y += scale * (x @ A^T) @ B^T` into `enc` for all adapters targeting `tensor_name`.
    /// Zero heap or GPU allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_linear(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        tensor_name: &str,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        x_byte_off: usize,
        y_buf: &ProtocolObject<dyn MTLBuffer>,
        y_byte_off: usize,
        n_tok: usize,
    ) {
        if self.chains.is_empty() || n_tok == 0 {
            return;
        }
        let Some(chain) = self.chains.get(tensor_name) else {
            return;
        };
        let Some(ref h_buf) = self.h_scratch else {
            return;
        };

        for entry in chain {
            if entry.is_token_embd {
                continue;
            }
            let args = LoraLinearArgs {
                d_in: entry.d_in as u32,
                d_out: entry.d_out as u32,
                rank: entry.rank as u32,
                n_tok: n_tok as u32,
                scale: entry.scale,
            };

            // 1. `lora_down_f32`: 4 `(t, r)` dot products per 128-thread TG
            let total_tr = n_tok * entry.rank;
            enc.setComputePipelineState(&self.pipe_down);
            // SAFETY: Buffers live on `self` and caller runner until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&entry.a_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(x_buf), x_byte_off, 1);
                enc.setBuffer_offset_atIndex(Some(h_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<LoraLinearArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: total_tr.div_ceil(4),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );

            // 2. `lora_up_axpy_f32`: coalesced `[d_out, n_tok]` AXPY accumulation
            let tg_x = entry.d_out.clamp(1, 256);
            enc.setComputePipelineState(&self.pipe_up);
            // SAFETY: Buffers live on `self` and caller runner until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&entry.bt_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(h_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(y_buf), y_byte_off, 2);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<LoraLinearArgs>(),
                    3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: entry.d_out.div_ceil(tg_x),
                    height: n_tok,
                    depth: 1,
                },
                MTLSize {
                    width: tg_x,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }

    /// Encodes single-token `token_embd.weight` LoRA delta into `x_buf`.
    pub fn encode_embed_single(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        tensor_name: &str,
        dummy_tok_buf: &ProtocolObject<dyn MTLBuffer>,
        token_id: u32,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
    ) {
        if self.chains.is_empty() {
            return;
        }
        let Some(chain) = self.chains.get(tensor_name) else {
            return;
        };
        for entry in chain {
            if !entry.is_token_embd {
                continue;
            }
            let args = LoraEmbedArgs {
                d_out: entry.d_out as u32,
                rank: entry.rank as u32,
                n_tok: 1,
                token_id,
                use_tok_buf: 0,
                scale: entry.scale,
            };
            let tg_x = entry.d_out.clamp(1, 256);
            enc.setComputePipelineState(&self.pipe_embed);
            // SAFETY: Buffers live on `self` and caller runner until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&entry.a_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&entry.bt_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(dummy_tok_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<LoraEmbedArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: entry.d_out.div_ceil(tg_x),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_x,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }

    /// Encodes batched `token_embd.weight` LoRA delta (`n_tok` tokens in `token_ids_buf`) into `x_buf`.
    pub fn encode_embed_batch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        tensor_name: &str,
        token_ids_buf: &ProtocolObject<dyn MTLBuffer>,
        x_buf: &ProtocolObject<dyn MTLBuffer>,
        n_tok: usize,
    ) {
        if self.chains.is_empty() || n_tok == 0 {
            return;
        }
        let Some(chain) = self.chains.get(tensor_name) else {
            return;
        };
        for entry in chain {
            if !entry.is_token_embd {
                continue;
            }
            let args = LoraEmbedArgs {
                d_out: entry.d_out as u32,
                rank: entry.rank as u32,
                n_tok: n_tok as u32,
                token_id: 0,
                use_tok_buf: 1,
                scale: entry.scale,
            };
            let tg_x = entry.d_out.clamp(1, 256);
            enc.setComputePipelineState(&self.pipe_embed);
            // SAFETY: Buffers live on `self` and caller runner until command buffer completion.
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&entry.a_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&entry.bt_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(token_ids_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 3);
                enc.setBytes_length_atIndex(
                    NonNull::from(&args).cast::<c_void>(),
                    core::mem::size_of::<LoraEmbedArgs>(),
                    4,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: entry.d_out.div_ceil(tg_x),
                    height: n_tok,
                    depth: 1,
                },
                MTLSize {
                    width: tg_x,
                    height: 1,
                    depth: 1,
                },
            );
        }
    }
}
