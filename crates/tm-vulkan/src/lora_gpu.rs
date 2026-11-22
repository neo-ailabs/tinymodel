//! Shared Vulkan dynamic GPU LoRA side-branch engine (`lora_down`, `lora_up_axpy`, `lora_embed`)
//! used by `LlamaVkRunner` and `Qwen35VkRunner` (`Phase T4`).
//!
//! Enforces `AGENTS.md` Rules #4 & #6:
//! - Validates `LoraSet` against `GgufFile` **before** mutating runner state.
//! - Pre-allocates all `VkBuffer`s and static `vk::DescriptorSet`s at `bind_lora` time so
//!   both **pre-recorded single-CB decode** and **batched prefill** execute with zero heap
//!   allocations and zero descriptor updates per token!

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use tm_gguf::GgufFile;
use tm_lora::{BoundLora, LoraSet};

use crate::context::{Inner, Pipeline, VkBuffer, VulkanContext};
use crate::kernels::{SPV_LORA_DOWN, SPV_LORA_EMBED, SPV_LORA_UP_AXPY};
use crate::{VkResultExt, VulkanError};

/// Target activation slice `(x_in -> y_out)` for a linear weight matrix inside a Vulkan runner.
#[derive(Clone)]
pub(crate) struct VkLinearSlot {
    pub name: String,
    pub x_buf: vk::Buffer,
    pub x_off: u64,
    pub x_len: u64,
    pub y_buf: vk::Buffer,
    pub y_off: u64,
    pub y_len: u64,
}

/// Target activation slice for an embedding table (`token_embd.weight`) inside a Vulkan runner.
#[derive(Clone)]
pub(crate) struct VkEmbedSlot {
    pub name: String,
    pub dec_tok_buf: vk::Buffer,
    pub dec_tok_len: u64,
    pub batch_tok_buf: vk::Buffer,
    pub batch_tok_len: u64,
    pub x_buf: vk::Buffer,
    pub x_off: u64,
    pub x_len: u64,
}

struct VkLinearLoraCmd {
    down_set: vk::DescriptorSet,
    up_set: vk::DescriptorSet,
    d_in: u32,
    d_out: u32,
    rank: u32,
    scale_bits: u32,
    sgs_per_wg: u32,
    wg_up: u32,
}

struct VkEmbedLoraCmd {
    dec_set: vk::DescriptorSet,
    batch_set: vk::DescriptorSet,
    d_out: u32,
    rank: u32,
    scale_bits: u32,
    wg_embed: u32,
}

/// Pre-allocated Vulkan GPU LoRA engine bound to a runner's static arena slices.
pub struct VkLoraEngine {
    inner: Arc<Inner>,
    desc_pool: Option<vk::DescriptorPool>,
    p_down: Arc<Pipeline>,
    p_up: Arc<Pipeline>,
    p_embed: Arc<Pipeline>,
    _buffers: Vec<VkBuffer>,
    linear_chains: HashMap<String, Vec<VkLinearLoraCmd>>,
    embed_chains: HashMap<String, Vec<VkEmbedLoraCmd>>,
}

impl Drop for VkLoraEngine {
    fn drop(&mut self) {
        if let Some(dp) = self.desc_pool.take() {
            // SAFETY: Runner guarantees all command buffers using these descriptor sets are idle
            // before dropping or replacing `VkLoraEngine`.
            unsafe {
                let _ = self.inner.device.device_wait_idle();
                self.inner.device.destroy_descriptor_pool(dp, None);
            }
        }
    }
}

fn alloc_static_set(
    d: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    infos: &[vk::DescriptorBufferInfo],
) -> Result<vk::DescriptorSet, VulkanError> {
    // SAFETY: `pool`, `layout`, and buffer handles in `infos` outlive all uses of the set.
    unsafe {
        let layouts = [layout];
        let set = d
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .ctx("vkAllocateDescriptorSets")?[0];
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(core::slice::from_ref(info))
            })
            .collect();
        d.update_descriptor_sets(&writes, &[]);
        Ok(set)
    }
}

fn cmd_barrier(d: &ash::Device, cb: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    // SAFETY: `cb` is in the recording state.
    unsafe {
        d.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

impl VkLoraEngine {
    pub fn new(ctx: &VulkanContext) -> Result<Self, VulkanError> {
        let sg = ctx.info().subgroup_size;
        let wg_down = 128u32.max(sg);
        let wg_up = 256u32.max(sg).min(ctx.info().max_workgroup_invocations);
        let p_down = ctx.pipeline("lora_down", wg_down, SPV_LORA_DOWN, 3, 20, &[wg_down])?;
        let p_up = ctx.pipeline("lora_up_axpy", wg_up, SPV_LORA_UP_AXPY, 3, 20, &[wg_up])?;
        let p_embed = ctx.pipeline("lora_embed", wg_up, SPV_LORA_EMBED, 4, 16, &[wg_up])?;
        Ok(Self {
            inner: ctx.inner.clone(),
            desc_pool: None,
            p_down,
            p_up,
            p_embed,
            _buffers: Vec::new(),
            linear_chains: HashMap::new(),
            embed_chains: HashMap::new(),
        })
    }

    /// Validates `loras` against `gguf`, uploads all `(A, B_T)` buffers, and builds static
    /// descriptor sets for every matching slot. Leaves `self` untouched on any error (`Rule #6`).
    pub(crate) fn bind(
        &mut self,
        ctx: &VulkanContext,
        gguf: &GgufFile,
        loras: &LoraSet,
        max_batch: usize,
        linear_slots: &[VkLinearSlot],
        embed_slots: &[VkEmbedSlot],
    ) -> Result<(), VulkanError> {
        loras.validate_against_gguf(gguf)?;
        if loras.is_empty() {
            self.clear();
            return Ok(());
        }
        let bound = BoundLora::from_lora_set(loras);
        if bound.is_empty() {
            self.clear();
            return Ok(());
        }

        let mut total_entries = 0u32;
        for chain in &bound.chains {
            total_entries += chain.len() as u32;
        }
        let max_sets = (total_entries * 4 + 16).max(16);
        let max_desc = max_sets * 4;
        let d = &ctx.inner.device;
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_desc,
        }];
        // SAFETY: `d` is a valid logical device.
        let new_pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(max_sets)
                    .pool_sizes(&pool_sizes),
                None,
            )
            .ctx("vkCreateDescriptorPool")?
        };

        // Guard pool on early error before committing into `self`.
        struct PoolGuard<'a> {
            d: &'a ash::Device,
            pool: Option<vk::DescriptorPool>,
        }
        impl Drop for PoolGuard<'_> {
            fn drop(&mut self) {
                if let Some(p) = self.pool.take() {
                    // SAFETY: `p` is newly created and not referenced by any in-flight CB.
                    unsafe { self.d.destroy_descriptor_pool(p, None) };
                }
            }
        }
        let mut guard = PoolGuard {
            d,
            pool: Some(new_pool),
        };

        let mut new_bufs = Vec::with_capacity(total_entries as usize * 2 + 1);
        let h_bytes = ((max_batch.max(1) * bound.max_rank.max(1) * 4) as u64).max(16);
        let h_buf = ctx.alloc_buffer(h_bytes)?;
        let h_vk = h_buf.buf;
        new_bufs.push(h_buf);

        let sg = ctx.info().subgroup_size;
        let sgs_per_wg = (self.p_down.wg_size / sg).max(1);
        let wg_up = self.p_up.wg_size;
        let wg_embed = self.p_embed.wg_size;

        let lin_map: HashMap<&str, &VkLinearSlot> =
            linear_slots.iter().map(|s| (s.name.as_str(), s)).collect();
        let emb_map: HashMap<&str, &VkEmbedSlot> =
            embed_slots.iter().map(|s| (s.name.as_str(), s)).collect();

        let mut new_lin: HashMap<String, Vec<VkLinearLoraCmd>> = HashMap::new();
        let mut new_emb: HashMap<String, Vec<VkEmbedLoraCmd>> = HashMap::new();

        for (name, &slot_idx) in &bound.name_to_slot {
            let src_chain = &bound.chains[slot_idx];
            for entry in src_chain {
                let a_buf = ctx.upload(&entry.a)?;
                let bt_buf = ctx.upload(&entry.b_t)?;
                let a_vk = a_buf.buf;
                let a_len = a_buf.byte_len();
                let bt_vk = bt_buf.buf;
                let bt_len = bt_buf.byte_len();
                new_bufs.push(a_buf);
                new_bufs.push(bt_buf);

                if entry.is_token_embd {
                    if let Some(es) = emb_map.get(name.as_str()) {
                        let dec_set = alloc_static_set(
                            d,
                            new_pool,
                            self.p_embed.set_layout,
                            &[
                                vk::DescriptorBufferInfo {
                                    buffer: a_vk,
                                    offset: 0,
                                    range: a_len,
                                },
                                vk::DescriptorBufferInfo {
                                    buffer: bt_vk,
                                    offset: 0,
                                    range: bt_len,
                                },
                                vk::DescriptorBufferInfo {
                                    buffer: es.dec_tok_buf,
                                    offset: 0,
                                    range: es.dec_tok_len,
                                },
                                vk::DescriptorBufferInfo {
                                    buffer: es.x_buf,
                                    offset: es.x_off,
                                    range: es.x_len,
                                },
                            ],
                        )?;
                        let batch_set = alloc_static_set(
                            d,
                            new_pool,
                            self.p_embed.set_layout,
                            &[
                                vk::DescriptorBufferInfo {
                                    buffer: a_vk,
                                    offset: 0,
                                    range: a_len,
                                },
                                vk::DescriptorBufferInfo {
                                    buffer: bt_vk,
                                    offset: 0,
                                    range: bt_len,
                                },
                                vk::DescriptorBufferInfo {
                                    buffer: es.batch_tok_buf,
                                    offset: 0,
                                    range: es.batch_tok_len,
                                },
                                vk::DescriptorBufferInfo {
                                    buffer: es.x_buf,
                                    offset: es.x_off,
                                    range: es.x_len,
                                },
                            ],
                        )?;
                        new_emb
                            .entry(name.clone())
                            .or_default()
                            .push(VkEmbedLoraCmd {
                                dec_set,
                                batch_set,
                                d_out: entry.d_out as u32,
                                rank: entry.rank as u32,
                                scale_bits: entry.scale.to_bits(),
                                wg_embed,
                            });
                    }
                } else if let Some(ls) = lin_map.get(name.as_str()) {
                    let down_set = alloc_static_set(
                        d,
                        new_pool,
                        self.p_down.set_layout,
                        &[
                            vk::DescriptorBufferInfo {
                                buffer: a_vk,
                                offset: 0,
                                range: a_len,
                            },
                            vk::DescriptorBufferInfo {
                                buffer: ls.x_buf,
                                offset: ls.x_off,
                                range: ls.x_len,
                            },
                            vk::DescriptorBufferInfo {
                                buffer: h_vk,
                                offset: 0,
                                range: h_bytes,
                            },
                        ],
                    )?;
                    let up_set = alloc_static_set(
                        d,
                        new_pool,
                        self.p_up.set_layout,
                        &[
                            vk::DescriptorBufferInfo {
                                buffer: bt_vk,
                                offset: 0,
                                range: bt_len,
                            },
                            vk::DescriptorBufferInfo {
                                buffer: h_vk,
                                offset: 0,
                                range: h_bytes,
                            },
                            vk::DescriptorBufferInfo {
                                buffer: ls.y_buf,
                                offset: ls.y_off,
                                range: ls.y_len,
                            },
                        ],
                    )?;
                    new_lin
                        .entry(name.clone())
                        .or_default()
                        .push(VkLinearLoraCmd {
                            down_set,
                            up_set,
                            d_in: entry.d_in as u32,
                            d_out: entry.d_out as u32,
                            rank: entry.rank as u32,
                            scale_bits: entry.scale.to_bits(),
                            sgs_per_wg,
                            wg_up,
                        });
                }
            }
        }

        let committed_pool = guard.pool.take();
        if let Some(old_dp) = self.desc_pool.take() {
            // SAFETY: Device is idle before swapping descriptor pools.
            unsafe {
                let _ = d.device_wait_idle();
                d.destroy_descriptor_pool(old_dp, None);
            }
        }
        self.desc_pool = committed_pool;
        self._buffers = new_bufs;
        self.linear_chains = new_lin;
        self.embed_chains = new_emb;
        Ok(())
    }

    pub fn clear(&mut self) {
        if let Some(dp) = self.desc_pool.take() {
            // SAFETY: Device idle before destroying descriptor pool.
            unsafe {
                let _ = self.inner.device.device_wait_idle();
                self.inner.device.destroy_descriptor_pool(dp, None);
            }
        }
        self._buffers.clear();
        self.linear_chains.clear();
        self.embed_chains.clear();
    }

    #[inline]
    pub fn has_linear(&self, tensor_name: &str) -> bool {
        !self.linear_chains.is_empty() && self.linear_chains.contains_key(tensor_name)
    }

    /// Records `lora_embed` dispatches into `cb` (followed by a compute barrier when active).
    /// Zero heap allocations.
    pub(crate) fn cmd_record_embed(
        &self,
        ctx: &VulkanContext,
        cb: vk::CommandBuffer,
        tensor_name: &str,
        n_tok: u32,
        is_batch: bool,
    ) -> Result<(), VulkanError> {
        if self.embed_chains.is_empty() || n_tok == 0 {
            return Ok(());
        }
        let Some(chain) = self.embed_chains.get(tensor_name) else {
            return Ok(());
        };
        let d = &ctx.inner.device;
        let lim = ctx.info().max_workgroup_count;
        for cmd in chain {
            let gx = cmd.d_out.div_ceil(cmd.wg_embed);
            if gx > lim[0] || n_tok > lim[1] {
                return Err(VulkanError::InvalidDimensions(format!(
                    "lora_embed dispatch ({gx}, {n_tok}, 1) exceeds device limits {:?}",
                    lim
                )));
            }
            let set = if is_batch { cmd.batch_set } else { cmd.dec_set };
            let words = [cmd.d_out, cmd.rank, n_tok, cmd.scale_bits];
            let mut push = [0u8; 16];
            for (i, &w) in words.iter().enumerate() {
                push[i * 4..(i + 1) * 4].copy_from_slice(&w.to_ne_bytes());
            }
            // SAFETY: `cb` is recording; pipeline and static descriptor set outlive execution.
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.p_embed.pipeline);
                d.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.p_embed.layout,
                    0,
                    &[set],
                    &[],
                );
                d.cmd_push_constants(
                    cb,
                    self.p_embed.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                d.cmd_dispatch(cb, gx, n_tok, 1);
            }
            cmd_barrier(d, cb);
        }
        Ok(())
    }

    /// Records `lora_down` + barrier + `lora_up_axpy` + barrier into `cb`.
    /// Zero heap allocations.
    pub(crate) fn cmd_record_linear(
        &self,
        ctx: &VulkanContext,
        cb: vk::CommandBuffer,
        tensor_name: &str,
        n_tok: u32,
    ) -> Result<(), VulkanError> {
        if self.linear_chains.is_empty() || n_tok == 0 {
            return Ok(());
        }
        let Some(chain) = self.linear_chains.get(tensor_name) else {
            return Ok(());
        };
        let d = &ctx.inner.device;
        let lim = ctx.info().max_workgroup_count;
        // Ensure prior base GEMV/GEMM writes into `y` are visible before `lora_up_axpy` `+=`!
        cmd_barrier(d, cb);

        for cmd in chain {
            let gx_down = (n_tok * cmd.rank).div_ceil(cmd.sgs_per_wg);
            let gx_up = cmd.d_out.div_ceil(cmd.wg_up);
            if gx_down > lim[0] || gx_up > lim[0] || n_tok > lim[1] {
                return Err(VulkanError::InvalidDimensions(format!(
                    "lora_linear dispatch exceeds device limits {:?}",
                    lim
                )));
            }
            let words = [cmd.d_in, cmd.d_out, cmd.rank, n_tok, cmd.scale_bits];
            let mut push = [0u8; 20];
            for (i, &w) in words.iter().enumerate() {
                push[i * 4..(i + 1) * 4].copy_from_slice(&w.to_ne_bytes());
            }
            // SAFETY: `cb` is recording; pipelines and static descriptor sets outlive execution.
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.p_down.pipeline);
                d.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.p_down.layout,
                    0,
                    &[cmd.down_set],
                    &[],
                );
                d.cmd_push_constants(
                    cb,
                    self.p_down.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                d.cmd_dispatch(cb, gx_down, 1, 1);
            }
            cmd_barrier(d, cb);

            // SAFETY: as above.
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.p_up.pipeline);
                d.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.p_up.layout,
                    0,
                    &[cmd.up_set],
                    &[],
                );
                d.cmd_push_constants(
                    cb,
                    self.p_up.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push,
                );
                d.cmd_dispatch(cb, gx_up, n_tok, 1);
            }
            cmd_barrier(d, cb);
        }
        Ok(())
    }
}
