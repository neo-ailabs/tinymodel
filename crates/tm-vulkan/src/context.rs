//! Runtime-loaded Vulkan 1.2/1.3 device context, capability discovery (`SubgroupProperties`,
//! `VK_KHR_cooperative_matrix`, UMA detection, `maxStorageBufferRange`, `maxComputeWorkGroupInvocations`),
//! zero-staging UMA buffer allocator (+ discrete GPU staging fallback), and descriptor/pipeline helpers.

use core::ffi::{CStr, c_void};
use std::sync::Arc;

use ash::vk;
use thiserror::Error;
use tm_core::DType;
use tm_gguf::GgufError;

/// Errors returned by the `tm-vulkan` GPU backend.
#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Vulkan loader unavailable (`libvulkan` could not be dynamically loaded): {0}")]
    LoaderUnavailable(String),
    #[error("no Vulkan physical device with a compute queue found")]
    NoComputeDevice,
    #[error("Vulkan API error in {context}: {source}")]
    Vk {
        context: &'static str,
        #[source]
        source: vk::Result,
    },
    #[error("no suitable Vulkan memory type for filter=0x{type_bits:x} flags=0x{flags_raw:x}")]
    NoSuitableMemoryType { type_bits: u32, flags_raw: u32 },
    #[error(
        "Vulkan dispatch workgroup ({lx}x{ly}x{lz} = {total}) exceeds device maxComputeWorkGroupInvocations ({limit})"
    )]
    WorkgroupLimitExceeded {
        lx: u32,
        ly: u32,
        lz: u32,
        total: u32,
        limit: u32,
    },
    #[error(
        "Vulkan buffer size ({size} bytes) exceeds device maxStorageBufferRange ({limit} bytes)"
    )]
    StorageBufferRangeExceeded { size: u64, limit: u64 },
    #[error("unsupported GGUF architecture '{0}' on Vulkan backend")]
    UnsupportedArch(String),
    #[error("unsupported tensor dtype {dtype:?} for '{tensor}' on Vulkan backend")]
    UnsupportedDType { tensor: String, dtype: DType },
    #[error("context position {pos} exceeds max_ctx {max_ctx}")]
    ContextOverflow { pos: usize, max_ctx: usize },
    #[error("token id {token_id} out of vocabulary range 0..{vocab_size}")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    #[error(transparent)]
    Gguf(#[from] GgufError),
}

/// Discovered Vulkan physical device capabilities (`docs/ARCHITECTURE.md` §5.4).
#[derive(Debug, Clone)]
pub struct VulkanDeviceCaps {
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version_major: u32,
    pub api_version_minor: u32,
    pub api_version_patch: u32,
    pub driver_version: u32,
    pub device_type: &'static str,
    pub is_uma: bool,
    pub subgroup_size: u32,
    pub subgroup_arithmetic: bool,
    pub has_cooperative_matrix: bool,
    pub has_shader_float16_int8: bool,
    pub has_16bit_storage: bool,
    pub has_8bit_storage: bool,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_count: [u32; 3],
    pub max_shared_memory_bytes: u32,
    pub max_storage_buffer_range: u64,
    pub min_storage_buffer_alignment: u64,
    pub device_local_bytes: u64,
}

impl VulkanDeviceCaps {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"device_name\":\"{}\",\"vendor_id\":\"0x{:04x}\",\"device_id\":\"0x{:04x}\",\
            \"api_version\":\"{}.{}.{}\",\"device_type\":\"{}\",\"is_uma\":{},\
            \"subgroup_size\":{},\"subgroup_arithmetic\":{},\"has_cooperative_matrix\":{},\
            \"has_shader_float16_int8\":{},\"has_16bit_storage\":{},\"has_8bit_storage\":{},\
            \"max_workgroup_invocations\":{},\"max_shared_memory_bytes\":{},\
            \"max_storage_buffer_range\":{},\"device_local_bytes\":{}}}",
            self.device_name.replace('\\', "\\\\").replace('"', "\\\""),
            self.vendor_id,
            self.device_id,
            self.api_version_major,
            self.api_version_minor,
            self.api_version_patch,
            self.device_type,
            self.is_uma,
            self.subgroup_size,
            self.subgroup_arithmetic,
            self.has_cooperative_matrix,
            self.has_shader_float16_int8,
            self.has_16bit_storage,
            self.has_8bit_storage,
            self.max_workgroup_invocations,
            self.max_shared_memory_bytes,
            self.max_storage_buffer_range,
            self.device_local_bytes,
        )
    }
}

/// Shared Vulkan instance, logical device, compute queue, command pool, and fence.
pub struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buf: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub caps: VulkanDeviceCaps,
}

impl VulkanContext {
    /// Dynamically loads `libvulkan` and initializes a compute `VkDevice` + command buffer.
    pub fn new() -> Result<Arc<Self>, VulkanError> {
        // SAFETY: `ash::Entry::load` dynamically `dlopen`s the platform Vulkan loader (`libvulkan.so` / `.dylib`).
        let entry = unsafe {
            ash::Entry::load().map_err(|e| VulkanError::LoaderUnavailable(e.to_string()))?
        };

        let instance_ver = unsafe {
            entry
                .try_enumerate_instance_version()
                .ok()
                .flatten()
                .unwrap_or(vk::API_VERSION_1_1)
        };
        let target_api = instance_ver.min(vk::API_VERSION_1_3);

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"tinymodel")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"tinymodel-vulkan")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(target_api);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        // SAFETY: Valid `InstanceCreateInfo` with static CStr literals.
        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateInstance",
                    source,
                })?
        };

        // SAFETY: `instance` was just created.
        let pdevs = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|source| VulkanError::Vk {
                    context: "vkEnumeratePhysicalDevices",
                    source,
                })?
        };

        let mut chosen: Option<(vk::PhysicalDevice, u32, i32)> = None;
        for pdev in pdevs {
            // SAFETY: Valid `pdev` from `enumerate_physical_devices`.
            let props = unsafe { instance.get_physical_device_properties(pdev) };
            let qfamilies = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
            let Some((q_idx, _)) = qfamilies
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 40,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 30,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 20,
                vk::PhysicalDeviceType::CPU => 10,
                _ => 5,
            };
            if chosen.is_none_or(|(_, _, best_r)| rank > best_r) {
                chosen = Some((pdev, q_idx as u32, rank));
            }
        }

        let Some((pdev, queue_family_index, _)) = chosen else {
            // SAFETY: Destroy instance before returning error.
            unsafe { instance.destroy_instance(None) };
            return Err(VulkanError::NoComputeDevice);
        };

        // SAFETY: Query physical device properties, subgroup properties, memory heaps, and extensions.
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
        let exts = unsafe {
            instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default()
        };

        let has_ext = |target: &CStr| -> bool {
            exts.iter().any(|e| {
                // SAFETY: `extension_name` is a NUL-terminated C char array from the Vulkan driver.
                let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                name == target
            })
        };
        let has_cooperative_matrix = has_ext(c"VK_KHR_cooperative_matrix");
        let has_shader_float16_int8 = has_ext(c"VK_KHR_shader_float16_int8");
        let has_16bit_storage = has_ext(c"VK_KHR_16bit_storage");
        let has_8bit_storage = has_ext(c"VK_KHR_8bit_storage");

        let mut subgroup_size = 32u32;
        let mut subgroup_arithmetic = false;
        if props.api_version >= vk::API_VERSION_1_1 {
            let mut sg_props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg_props);
            // SAFETY: `props.api_version >= 1.1` guarantees `vkGetPhysicalDeviceProperties2`.
            unsafe { instance.get_physical_device_properties2(pdev, &mut props2) };
            if sg_props.subgroup_size > 0 {
                subgroup_size = sg_props.subgroup_size;
            }
            subgroup_arithmetic = sg_props
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::ARITHMETIC);
        }

        let mut device_local_bytes = 0u64;
        for h in &mem_props.memory_heaps[..mem_props.memory_heap_count as usize] {
            if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                device_local_bytes = device_local_bytes.max(h.size);
            }
        }

        let is_uma = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            || mem_props.memory_types[..mem_props.memory_type_count as usize]
                .iter()
                .any(|mt| {
                    let heap = mem_props.memory_heaps[mt.heap_index as usize];
                    heap.size >= device_local_bytes / 2
                        && mt.property_flags.contains(
                            vk::MemoryPropertyFlags::DEVICE_LOCAL
                                | vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                });

        // SAFETY: `props.device_name` is a NUL-terminated C string from Vulkan driver.
        let dev_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let dev_type_str = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
            vk::PhysicalDeviceType::CPU => "cpu",
            _ => "other",
        };

        let caps = VulkanDeviceCaps {
            device_name: dev_name,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            api_version_major: vk::api_version_major(props.api_version),
            api_version_minor: vk::api_version_minor(props.api_version),
            api_version_patch: vk::api_version_patch(props.api_version),
            driver_version: props.driver_version,
            device_type: dev_type_str,
            is_uma,
            subgroup_size,
            subgroup_arithmetic,
            has_cooperative_matrix,
            has_shader_float16_int8,
            has_16bit_storage,
            has_8bit_storage,
            max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
            max_workgroup_size: props.limits.max_compute_work_group_size,
            max_workgroup_count: props.limits.max_compute_work_group_count,
            max_shared_memory_bytes: props.limits.max_compute_shared_memory_size,
            max_storage_buffer_range: props.limits.max_storage_buffer_range as u64,
            min_storage_buffer_alignment: props.limits.min_storage_buffer_offset_alignment,
            device_local_bytes,
        };

        let priorities = [1.0f32];
        let q_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        let dev_info = vk::DeviceCreateInfo::default().queue_create_infos(&q_info);
        // SAFETY: Valid `pdev` and `q_info`.
        let device = unsafe {
            instance
                .create_device(pdev, &dev_info, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateDevice",
                    source,
                })?
        };
        // SAFETY: Queue 0 of `queue_family_index` was requested in `dev_info`.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: Valid `device`.
        let cmd_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateCommandPool",
                    source,
                })?
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: Valid `cmd_pool`.
        let cmd_buf = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|source| VulkanError::Vk {
                    context: "vkAllocateCommandBuffers",
                    source,
                })?[0]
        };

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: Valid `device`.
        let fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateFence",
                    source,
                })?
        };

        Ok(Arc::new(Self {
            _entry: entry,
            instance,
            pdev,
            device,
            queue,
            queue_family_index,
            cmd_pool,
            cmd_buf,
            fence,
            mem_props,
            caps,
        }))
    }

    pub fn find_memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            (type_bits & (1u32 << i)) != 0
                && self.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(required)
        })
    }

    /// Validates compute workgroup dimensions BEFORE dispatching (`AGENTS.md` §9).
    #[inline]
    pub fn check_workgroup(&self, lx: u32, ly: u32, lz: u32) -> Result<(), VulkanError> {
        let total = lx.saturating_mul(ly).saturating_mul(lz);
        if total > self.caps.max_workgroup_invocations
            || lx > self.caps.max_workgroup_size[0]
            || ly > self.caps.max_workgroup_size[1]
            || lz > self.caps.max_workgroup_size[2]
        {
            return Err(VulkanError::WorkgroupLimitExceeded {
                lx,
                ly,
                lz,
                total,
                limit: self.caps.max_workgroup_invocations,
            });
        }
        Ok(())
    }

    /// Inserts a compute-to-compute SSBO memory barrier inside `cmd_buf`.
    #[inline]
    pub fn cmd_compute_barrier(&self, cmd: vk::CommandBuffer) {
        let mem_barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        // SAFETY: `cmd` is in recording state.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &mem_barrier,
                &[],
                &[],
            );
        }
    }

    /// Records `record_fn(cmd_buf)`, submits once to `self.queue`, and waits on `self.fence`.
    pub fn submit_and_wait<F>(&self, record_fn: F) -> Result<(), VulkanError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), VulkanError>,
    {
        // SAFETY: Reset single-flight primary command buffer and record compute commands.
        unsafe {
            self.device
                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())
                .map_err(|source| VulkanError::Vk {
                    context: "vkResetCommandBuffer",
                    source,
                })?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(self.cmd_buf, &begin_info)
                .map_err(|source| VulkanError::Vk {
                    context: "vkBeginCommandBuffer",
                    source,
                })?;
        }

        record_fn(self.cmd_buf)?;

        // SAFETY: End command buffer, reset fence, submit to compute queue, and wait for completion.
        unsafe {
            self.device
                .end_command_buffer(self.cmd_buf)
                .map_err(|source| VulkanError::Vk {
                    context: "vkEndCommandBuffer",
                    source,
                })?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|source| VulkanError::Vk {
                    context: "vkResetFences",
                    source,
                })?;
            let cmd_bufs = [self.cmd_buf];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
            self.device
                .queue_submit(self.queue, &submit, self.fence)
                .map_err(|source| VulkanError::Vk {
                    context: "vkQueueSubmit",
                    source,
                })?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|source| VulkanError::Vk {
                    context: "vkWaitForFences",
                    source,
                })?;
        }
        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: Wait idle before destroying owned Vulkan handles in reverse creation order.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Owned Vulkan storage buffer (`VkBuffer` + `VkDeviceMemory`) with optional persistent host pointer.
/// Pads allocation by +16 bytes (rounded up to 64B) so unaligned 32-bit tail loads are always safe.
pub struct GpuBuffer {
    ctx: Arc<VulkanContext>,
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
    pub alloc_size: u64,
    mapped: *mut c_void,
}

// SAFETY: `GpuBuffer` synchronizes GPU access via `VulkanContext::submit_and_wait` fences.
unsafe impl Send for GpuBuffer {}
// SAFETY: Shared references only expose immutable buffer metadata / descriptors.
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Allocates a host-coherent SSBO (prefers `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` on UMA
    /// such as Pixel 8 Mali-G715, RK3588, V3DV, APU; falls back to `HOST_VISIBLE | HOST_COHERENT`).
    pub fn new_host_visible(
        ctx: &Arc<VulkanContext>,
        byte_len: usize,
    ) -> Result<Self, VulkanError> {
        let logical = (byte_len.max(16)) as u64;
        if logical > ctx.caps.max_storage_buffer_range {
            return Err(VulkanError::StorageBufferRangeExceeded {
                size: logical,
                limit: ctx.caps.max_storage_buffer_range,
            });
        }
        let alloc_size = (logical + 16 + 63) & !63;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let binfo = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Valid `BufferCreateInfo`.
        let buf = unsafe {
            ctx.device
                .create_buffer(&binfo, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateBuffer",
                    source,
                })?
        };
        // SAFETY: Valid `buf`.
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buf) };

        // CRITICAL on Arm Mali / Android SoCs (e.g., Tensor G3 Mali-G715, RK3588 Mali-G610):
        // Memory type 0 is often `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` WITHOUT `HOST_CACHED`
        // (`MT_NORMAL_NC` uncached write-combine), whereas another memory type exposes
        // `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` (hardware-coherent SLC + GPU L2)!
        // Always prefer `HOST_CACHED` first when available!
        let cached_uma_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let uma_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached_host_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let host_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let mem_idx = ctx
            .find_memory_type(req.memory_type_bits, cached_uma_flags)
            .or_else(|| ctx.find_memory_type(req.memory_type_bits, uma_flags))
            .or_else(|| ctx.find_memory_type(req.memory_type_bits, cached_host_flags))
            .or_else(|| ctx.find_memory_type(req.memory_type_bits, host_flags))
            .ok_or(VulkanError::NoSuitableMemoryType {
                type_bits: req.memory_type_bits,
                flags_raw: host_flags.as_raw(),
            })?;

        let minfo = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_idx);

        // SAFETY: Valid `MemoryAllocateInfo`, bind at offset 0, and persistently map zero-initialized memory.
        let (mem, mapped) = unsafe {
            let mem = ctx.device.allocate_memory(&minfo, None).map_err(|source| {
                ctx.device.destroy_buffer(buf, None);
                VulkanError::Vk {
                    context: "vkAllocateMemory",
                    source,
                }
            })?;
            if let Err(source) = ctx.device.bind_buffer_memory(buf, mem, 0) {
                ctx.device.free_memory(mem, None);
                ctx.device.destroy_buffer(buf, None);
                return Err(VulkanError::Vk {
                    context: "vkBindBufferMemory",
                    source,
                });
            }
            let ptr = ctx
                .device
                .map_memory(mem, 0, alloc_size, vk::MemoryMapFlags::empty())
                .map_err(|source| {
                    ctx.device.free_memory(mem, None);
                    ctx.device.destroy_buffer(buf, None);
                    VulkanError::Vk {
                        context: "vkMapMemory",
                        source,
                    }
                })?;
            core::ptr::write_bytes(ptr as *mut u8, 0, alloc_size as usize);
            (mem, ptr)
        };

        Ok(Self {
            ctx: Arc::clone(ctx),
            buf,
            mem,
            size: logical,
            alloc_size,
            mapped,
        })
    }

    /// Uploads `data` into a GPU SSBO:
    /// - On UMA (`is_uma`), writes directly into `DEVICE_LOCAL | HOST_VISIBLE` memory (zero staging buffer!).
    /// - On discrete GPUs (`!is_uma`), uploads via a temporary host staging buffer into `DEVICE_LOCAL` VRAM!
    pub fn from_bytes_device_optimal(
        ctx: &Arc<VulkanContext>,
        data: &[u8],
    ) -> Result<Self, VulkanError> {
        if ctx.caps.is_uma {
            let mut gb = Self::new_host_visible(ctx, data.len())?;
            gb.write_bytes(0, data);
            return Ok(gb);
        }

        let logical = (data.len().max(16)) as u64;
        if logical > ctx.caps.max_storage_buffer_range {
            return Err(VulkanError::StorageBufferRangeExceeded {
                size: logical,
                limit: ctx.caps.max_storage_buffer_range,
            });
        }
        let alloc_size = (logical + 16 + 63) & !63;
        let binfo = vk::BufferCreateInfo::default()
            .size(alloc_size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: Create `DEVICE_LOCAL` destination buffer and copy from staging buffer.
        let buf = unsafe {
            ctx.device
                .create_buffer(&binfo, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateBuffer(device)",
                    source,
                })?
        };
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buf) };
        let Some(dev_idx) =
            ctx.find_memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        else {
            unsafe { ctx.device.destroy_buffer(buf, None) };
            let mut gb = Self::new_host_visible(ctx, data.len())?;
            gb.write_bytes(0, data);
            return Ok(gb);
        };

        let minfo = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(dev_idx);
        // SAFETY: Allocate and bind `DEVICE_LOCAL` memory.
        let mem = unsafe {
            let mem = ctx.device.allocate_memory(&minfo, None).map_err(|source| {
                ctx.device.destroy_buffer(buf, None);
                VulkanError::Vk {
                    context: "vkAllocateMemory(device)",
                    source,
                }
            })?;
            ctx.device
                .bind_buffer_memory(buf, mem, 0)
                .map_err(|source| {
                    ctx.device.free_memory(mem, None);
                    ctx.device.destroy_buffer(buf, None);
                    VulkanError::Vk {
                        context: "vkBindBufferMemory(device)",
                        source,
                    }
                })?;
            mem
        };

        let mut staging = Self::new_host_visible(ctx, data.len())?;
        staging.write_bytes(0, data);
        ctx.submit_and_wait(|cmd| {
            let regions = [vk::BufferCopy::default().size(alloc_size)];
            // SAFETY: Both `staging.buf` and `buf` have at least `alloc_size` bytes.
            unsafe {
                ctx.device.cmd_copy_buffer(cmd, staging.buf, buf, &regions);
            }
            Ok(())
        })?;

        Ok(Self {
            ctx: Arc::clone(ctx),
            buf,
            mem,
            size: logical,
            alloc_size,
            mapped: core::ptr::null_mut(),
        })
    }

    pub fn from_f32_slice(ctx: &Arc<VulkanContext>, slice: &[f32]) -> Result<Self, VulkanError> {
        let byte_len = slice.len() * 4;
        // SAFETY: `f32` is POD and 4-byte aligned.
        let bytes = unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        Self::from_bytes_device_optimal(ctx, bytes)
    }

    pub fn write_bytes(&mut self, offset: usize, src: &[u8]) {
        assert!(!self.mapped.is_null(), "GpuBuffer is not host-mapped");
        assert!(offset + src.len() <= self.alloc_size as usize);
        if src.is_empty() {
            return;
        }
        // SAFETY: Bounds checked above; `mapped` is `HOST_COHERENT`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (self.mapped as *mut u8).add(offset),
                src.len(),
            );
        }
    }

    pub fn write_u32_slice(&mut self, offset_elems: usize, src: &[u32]) {
        // SAFETY: `u32` is POD.
        let bytes =
            unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        self.write_bytes(offset_elems * 4, bytes);
    }

    pub fn write_f32_slice(&mut self, offset_elems: usize, src: &[f32]) {
        // SAFETY: `f32` is POD.
        let bytes =
            unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        self.write_bytes(offset_elems * 4, bytes);
    }

    pub fn read_f32_slice(&self, offset_elems: usize, dst: &mut [f32]) {
        assert!(!self.mapped.is_null(), "GpuBuffer is not host-mapped");
        let byte_off = offset_elems * 4;
        let byte_len = dst.len() * 4;
        assert!(byte_off + byte_len <= self.alloc_size as usize);
        if dst.is_empty() {
            return;
        }
        // SAFETY: Bounds checked above; `mapped` is `HOST_COHERENT`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                (self.mapped as *const u8).add(byte_off),
                dst.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    pub fn read_u32(&self, index: usize) -> u32 {
        assert!(!self.mapped.is_null(), "GpuBuffer is not host-mapped");
        assert!((index + 1) * 4 <= self.alloc_size as usize);
        // SAFETY: Bounds checked and 4-byte aligned.
        unsafe { *((self.mapped as *const u32).add(index)) }
    }

    pub fn zero_fill(&mut self) {
        if !self.mapped.is_null() {
            // SAFETY: `mapped` covers `alloc_size` bytes.
            unsafe {
                core::ptr::write_bytes(self.mapped as *mut u8, 0, self.alloc_size as usize);
            }
        }
    }

    #[inline]
    pub fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buf)
            .offset(0)
            .range(vk::WHOLE_SIZE)
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: Unmap (if mapped), destroy buffer, and free backing memory.
        unsafe {
            if !self.mapped.is_null() {
                self.ctx.device.unmap_memory(self.mem);
            }
            self.ctx.device.destroy_buffer(self.buf, None);
            self.ctx.device.free_memory(self.mem, None);
        }
    }
}

/// Compiled compute pipeline (`VkDescriptorSetLayout` + `VkPipelineLayout` + `VkPipeline`).
pub struct ComputeKernel {
    ctx: Arc<VulkanContext>,
    pub set_layout: vk::DescriptorSetLayout,
    pub pipe_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub local_size: [u32; 3],
    pub num_bindings: u32,
}

impl ComputeKernel {
    pub fn from_spirv(
        ctx: &Arc<VulkanContext>,
        spv_bytes: &[u8],
        num_bindings: u32,
        push_constant_bytes: u32,
        local_size: [u32; 3],
    ) -> Result<Self, VulkanError> {
        ctx.check_workgroup(local_size[0], local_size[1], local_size[2])?;
        let (chunks, rem) = spv_bytes.as_chunks::<4>();
        assert!(rem.is_empty());
        let words: Vec<u32> = chunks.iter().copied().map(u32::from_le_bytes).collect();

        let mod_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: `words` is a validated SPIR-V 1.0 binary emitted by `naga`.
        let shader_mod = unsafe {
            ctx.device
                .create_shader_module(&mod_info, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateShaderModule",
                    source,
                })?
        };

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..num_bindings)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ds_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: Valid `DescriptorSetLayoutCreateInfo`.
        let set_layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(&ds_info, None)
                .map_err(|source| {
                    ctx.device.destroy_shader_module(shader_mod, None);
                    VulkanError::Vk {
                        context: "vkCreateDescriptorSetLayout",
                        source,
                    }
                })?
        };

        let pc_ranges = if push_constant_bytes > 0 {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_bytes),
            ]
        } else {
            Vec::new()
        };
        let set_layouts = [set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_ranges);
        // SAFETY: Valid `PipelineLayoutCreateInfo`.
        let pipe_layout = unsafe {
            ctx.device
                .create_pipeline_layout(&pl_info, None)
                .map_err(|source| {
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    ctx.device.destroy_shader_module(shader_mod, None);
                    VulkanError::Vk {
                        context: "vkCreatePipelineLayout",
                        source,
                    }
                })?
        };

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_mod)
            .name(c"main");
        let cp_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipe_layout)];
        // SAFETY: Valid `ComputePipelineCreateInfo`; shader module can be destroyed right after pipeline creation.
        let pipeline = unsafe {
            let res =
                ctx.device
                    .create_compute_pipelines(vk::PipelineCache::null(), &cp_info, None);
            ctx.device.destroy_shader_module(shader_mod, None);
            match res {
                Ok(pipes) => pipes[0],
                Err((_, source)) => {
                    ctx.device.destroy_pipeline_layout(pipe_layout, None);
                    ctx.device.destroy_descriptor_set_layout(set_layout, None);
                    return Err(VulkanError::Vk {
                        context: "vkCreateComputePipelines",
                        source,
                    });
                }
            }
        };

        Ok(Self {
            ctx: Arc::clone(ctx),
            set_layout,
            pipe_layout,
            pipeline,
            local_size,
            num_bindings,
        })
    }

    /// Binds pipeline, descriptor set, optional push constants, validates workgroup limits, and dispatches!
    #[inline]
    pub fn cmd_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        dset: vk::DescriptorSet,
        push_bytes: &[u8],
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), VulkanError> {
        if groups_x == 0 || groups_y == 0 || groups_z == 0 {
            return Ok(());
        }
        self.ctx
            .check_workgroup(self.local_size[0], self.local_size[1], self.local_size[2])?;
        // SAFETY: `cmd` is recording, `dset` matches `self.set_layout`, and `push_bytes` matches `pipe_layout`.
        unsafe {
            self.ctx
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipe_layout,
                0,
                &[dset],
                &[],
            );
            if !push_bytes.is_empty() {
                self.ctx.device.cmd_push_constants(
                    cmd,
                    self.pipe_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
            }
            self.ctx
                .device
                .cmd_dispatch(cmd, groups_x, groups_y, groups_z);
        }
        Ok(())
    }
}

impl Drop for ComputeKernel {
    fn drop(&mut self) {
        // SAFETY: Destroy pipeline and layout handles owned by `self`.
        unsafe {
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx
                .device
                .destroy_pipeline_layout(self.pipe_layout, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Resettable descriptor pool pre-allocated per runner / step (avoids `VK_KHR_push_descriptor` extension dependency).
pub struct DescriptorArena {
    ctx: Arc<VulkanContext>,
    pool: vk::DescriptorPool,
}

impl DescriptorArena {
    pub fn new(
        ctx: &Arc<VulkanContext>,
        max_sets: u32,
        max_storage_buffers: u32,
    ) -> Result<Self, VulkanError> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(max_storage_buffers)];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(&sizes);
        // SAFETY: Valid `DescriptorPoolCreateInfo`.
        let pool = unsafe {
            ctx.device
                .create_descriptor_pool(&info, None)
                .map_err(|source| VulkanError::Vk {
                    context: "vkCreateDescriptorPool",
                    source,
                })?
        };
        Ok(Self {
            ctx: Arc::clone(ctx),
            pool,
        })
    }

    pub fn reset(&mut self) -> Result<(), VulkanError> {
        // SAFETY: Must only be called when no command buffer referencing sets from `self.pool` is in flight.
        unsafe {
            self.ctx
                .device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|source| VulkanError::Vk {
                    context: "vkResetDescriptorPool",
                    source,
                })
        }
    }

    pub fn alloc_and_write(
        &self,
        kernel: &ComputeKernel,
        buffers: &[&GpuBuffer],
    ) -> Result<vk::DescriptorSet, VulkanError> {
        assert_eq!(buffers.len(), kernel.num_bindings as usize);
        let layouts = [kernel.set_layout];
        let ainfo = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        // SAFETY: Valid `DescriptorPool` with sufficient remaining capacity.
        let dset = unsafe {
            self.ctx
                .device
                .allocate_descriptor_sets(&ainfo)
                .map_err(|source| VulkanError::Vk {
                    context: "vkAllocateDescriptorSets",
                    source,
                })?[0]
        };

        let buf_infos: Vec<[vk::DescriptorBufferInfo; 1]> =
            buffers.iter().map(|b| [b.descriptor_info()]).collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = buf_infos
            .iter()
            .enumerate()
            .map(|(idx, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(idx as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(bi)
            })
            .collect();
        // SAFETY: `dset` and `buf_infos` remain valid across `update_descriptor_sets`.
        unsafe {
            self.ctx.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(dset)
    }
}

impl Drop for DescriptorArena {
    fn drop(&mut self) {
        // SAFETY: Destroy owned `VkDescriptorPool`.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

#[inline]
pub fn as_bytes<T: Copy>(val: &T) -> &[u8] {
    // SAFETY: Caller passes `#[repr(C)]` POD push-constant structs containing only `u32`/`f32`.
    unsafe {
        core::slice::from_raw_parts((val as *const T) as *const u8, core::mem::size_of::<T>())
    }
}
