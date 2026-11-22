//! Instance / device selection, memory, pipelines and command submission.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::{VkResultExt, VulkanError};

const VENDOR_AMD: u32 = 0x1002;
const VENDOR_INTEL: u32 = 0x8086;
const STAGING_BYTES: u64 = 64 << 20;
const DESC_POOL_SETS: u32 = 4096;
const DESC_POOL_BUFFERS: u32 = 4096 * 8;

/// Batched prefill kernel selection policy for [`VulkanContext`] / [`crate::LlamaVkRunner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VkPrefillMode {
    /// Automatically pick Cooperative Matrix (`16x16x16`) when supported by the device, else
    /// fall back to portable subgroup/LDS tiled kernels.
    Auto,
    /// Force portable subgroup/LDS tiled fallback (`gemm_tiled` + `gemm_kquant` + `fa_gqa_tiled`).
    Fallback,
    /// Prefer `16x16x16 f16*f16+f32->f32` Cooperative Matrix GEMM + FlashAttention.
    CoopMatF16,
    /// Prefer `16x16x16 i8*i8+i32->i32` Cooperative Matrix GEMM (with dynamic `Q8_1` activation
    /// quantization) for `Q4_0`/`Q4_1`/`Q8_0`, and `CoopMatF16` for remaining ops.
    CoopMatI8,
}

impl VkPrefillMode {
    pub fn from_env() -> Self {
        match std::env::var("TM_VK_PREFILL")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some(s) if s.eq_ignore_ascii_case("fallback") || s == "0" => Self::Fallback,
            Some(s) if s.eq_ignore_ascii_case("f16") || s.eq_ignore_ascii_case("coopmat_f16") => {
                Self::CoopMatF16
            }
            Some(s) if s.eq_ignore_ascii_case("i8") || s.eq_ignore_ascii_case("coopmat_i8") => {
                Self::CoopMatI8
            }
            _ => Self::Auto,
        }
    }
}

/// Capabilities of the selected Vulkan device (also used by `probe --json`).
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: &'static str,
    pub driver_name: String,
    pub driver_info: String,
    pub api_version: (u32, u32, u32),
    /// Subgroup size every pipeline is created with (pinned when size control is available).
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub subgroup_size_control: bool,
    pub subgroup_shuffle: bool,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_count: [u32; 3],
    pub max_shared_memory: u32,
    pub max_alloc_bytes: u64,
    pub max_storage_buffer_range: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub timestamp_period_ns: f32,
    pub timestamp_compute: bool,
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub storage_16bit: bool,
    pub storage_8bit: bool,
    pub int8_dot_accelerated: bool,
    pub cooperative_matrix: bool,
    pub coopmat_f16_f32acc: bool,
    pub coopmat_i8_i32acc: bool,
    pub coopmat_shapes: Vec<String>,
    pub shader_bfloat16: bool,
    pub shader_float8: bool,
    pub device_local_bytes: u64,
    /// Size of the largest DEVICE_LOCAL | HOST_VISIBLE heap (≈ full VRAM when ReBAR is on).
    pub host_visible_device_local_bytes: u64,
    pub unified_memory: bool,
    pub external_memory_host: bool,
    pub external_memory_host_align: u64,
}

pub(crate) struct Inner {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    ext_host: Option<ash::ext::external_memory_host::Device>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: `Inner` is dropped only after every buffer / pipeline / pool holding an
        // `Arc<Inner>` or created from this device is gone; waiting idle first guarantees
        // no submitted work still references them.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A device buffer (storage + transfer usage). Device-local unless created as host-visible.
pub struct VkBuffer {
    inner: Arc<Inner>,
    pub(crate) buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
    mapped: Option<*mut u8>,
}

// SAFETY: Vulkan buffer/memory handles are plain identifiers usable from any thread; the
// mapped pointer is only dereferenced through `&self`/`&mut self` methods after GPU work
// on the buffer has been waited for (all submissions in this crate are synchronous).
unsafe impl Send for VkBuffer {}

impl VkBuffer {
    pub fn byte_len(&self) -> u64 {
        self.size
    }

    pub(crate) fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        // SAFETY: the buffer and memory were created from `inner.device`; all submissions
        // are waited on before returning, so the GPU no longer uses them.
        unsafe {
            if self.mapped.is_some() {
                self.inner.device.unmap_memory(self.mem);
            }
            self.inner.device.destroy_buffer(self.buf, None);
            self.inner.device.free_memory(self.mem, None);
        }
    }
}

pub(crate) struct Pipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) n_bindings: u32,
    pub(crate) wg_size: u32,
    pub(crate) name: &'static str,
}

struct Exec {
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    staging: Option<VkBuffer>,
}

/// One recorded dispatch-batch; created by [`VulkanContext::run`].
pub(crate) struct Recorder<'a> {
    pub(crate) ctx: &'a VulkanContext,
    pub(crate) cb: vk::CommandBuffer,
    desc_pool: vk::DescriptorPool,
}

/// The selected Vulkan device, its compute queue, and a pipeline cache.
pub struct VulkanContext {
    pub(crate) inner: Arc<Inner>,
    info: DeviceInfo,
    exec: Mutex<Exec>,
    pub(crate) pipelines: Mutex<HashMap<(&'static str, u32), Arc<Pipeline>>>,
    pub(crate) gemv_cfg: Mutex<(u32, u32)>,
    prefill_mode: Mutex<VkPrefillMode>,
    force_lds_recurrence: Mutex<bool>,
    force_slab_copy: Mutex<bool>,
}

fn cstr_to_string(s: Result<&CStr, std::ffi::FromBytesUntilNulError>) -> String {
    s.map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn type_name(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn type_score(t: vk::PhysicalDeviceType) -> u32 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

struct Probe {
    info: DeviceInfo,
    queue_family: u32,
    f11_storage16: bool,
    f12_int8: bool,
    f12_fp16: bool,
    f12_storage8: bool,
    f12_vmm: bool,
    f12_vmm_dev_scope: bool,
    f13_size_control: bool,
    f13_full_subgroups: bool,
    core_int16: bool,
    enable_coopmat: bool,
}

fn comp_type_short(ty: vk::ComponentTypeKHR) -> &'static str {
    match ty {
        vk::ComponentTypeKHR::FLOAT16 => "f16",
        vk::ComponentTypeKHR::FLOAT32 => "f32",
        vk::ComponentTypeKHR::FLOAT64 => "f64",
        vk::ComponentTypeKHR::SINT8 => "i8",
        vk::ComponentTypeKHR::SINT16 => "i16",
        vk::ComponentTypeKHR::SINT32 => "i32",
        vk::ComponentTypeKHR::UINT8 => "u8",
        vk::ComponentTypeKHR::UINT16 => "u16",
        vk::ComponentTypeKHR::UINT32 => "u32",
        other => match other.as_raw() {
            1000141000 => "bf16",
            1000491002 => "fp8_e4m3",
            1000491003 => "fp8_e5m2",
            _ => "other",
        },
    }
}

fn probe_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
    index: usize,
) -> Option<Probe> {
    // SAFETY: `pdev` was returned by `enumerate_physical_devices` on `instance`; every
    // query below only writes into the locally owned output structs.
    unsafe {
        let props = instance.get_physical_device_properties(pdev);
        let api = props.api_version;
        let api_ge_13 = vk::api_version_minor(api) >= 3 || vk::api_version_major(api) > 1;

        let qf_props = instance.get_physical_device_queue_family_properties(pdev);
        let (qf, qf_ts_bits) = qf_props
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            // Prefer a compute queue that also does graphics (the "universal" queue):
            // it has the highest priority/throughput on most drivers.
            .max_by_key(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS) as u32)
            .map(|(i, q)| (i as u32, q.timestamp_valid_bits))?;

        let exts = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default();
        let has_ext = |name: &CStr| {
            exts.iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
        };

        let has_cm_ext = has_ext(c"VK_KHR_cooperative_matrix");
        let has_ext_host = has_ext(c"VK_EXT_external_memory_host");

        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut m3 = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        let mut sgc = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut idp = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
        let mut ext_host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut sg)
                .push_next(&mut m3)
                .push_next(&mut drv);
            if api_ge_13 {
                p2 = p2.push_next(&mut sgc).push_next(&mut idp);
            }
            if has_ext_host {
                p2 = p2.push_next(&mut ext_host_props);
            }
            instance.get_physical_device_properties2(pdev, &mut p2);
        }

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f_cm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let core_int16;
        {
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12);
            if api_ge_13 {
                f2 = f2.push_next(&mut f13);
            }
            if has_cm_ext {
                f2 = f2.push_next(&mut f_cm);
            }
            instance.get_physical_device_features2(pdev, &mut f2);
            core_int16 = f2.features.shader_int16 == vk::TRUE;
        }

        let mem = instance.get_physical_device_memory_properties(pdev);
        let heaps = mem.memory_heaps_as_slice();
        let mut device_local = 0u64;
        let mut hv_device_local = 0u64;
        for h in heaps {
            if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                device_local = device_local.max(h.size);
            }
        }
        for t in mem.memory_types_as_slice() {
            let f = t.property_flags;
            if f.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ) {
                hv_device_local = hv_device_local.max(heaps[t.heap_index as usize].size);
            }
        }

        let size_control = api_ge_13
            && f13.subgroup_size_control == vk::TRUE
            && sgc
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE);
        let (min_sg, max_sg) = if api_ge_13 && sgc.min_subgroup_size > 0 {
            (sgc.min_subgroup_size, sgc.max_subgroup_size)
        } else {
            (sg.subgroup_size, sg.subgroup_size)
        };
        // AMD RDNA: wave32 is the native fit for GEMV and 16x16x16 WMMA cooperative matrix.
        // Intel Gen12 / Xe: SIMD16 is the native EU execution width (avoids SIMD32 GRF spills
        // and preserves full EU thread occupancy; measured +75% pp / +46% tg on UHD 770).
        let chosen_sg = if size_control
            && props.vendor_id == VENDOR_AMD
            && (min_sg..=max_sg).contains(&32)
        {
            32
        } else if size_control && props.vendor_id == VENDOR_INTEL && (min_sg..=max_sg).contains(&16)
        {
            16
        } else {
            sg.subgroup_size
        };

        let f12_vmm = f12.vulkan_memory_model == vk::TRUE;
        let f12_vmm_dev_scope = f12.vulkan_memory_model_device_scope == vk::TRUE;
        let enable_coopmat =
            has_cm_ext && f_cm.cooperative_matrix == vk::TRUE && f12_vmm && f12_vmm_dev_scope;
        let mut coopmat_shapes = Vec::new();
        let mut coopmat_f16_f32acc = false;
        let mut coopmat_i8_i32acc = false;
        if enable_coopmat {
            let cm = ash::khr::cooperative_matrix::Instance::new(entry, instance);
            if let Ok(list) = cm.get_physical_device_cooperative_matrix_properties(pdev) {
                for p in list {
                    if p.scope == vk::ScopeKHR::SUBGROUP {
                        if p.m_size == 16
                            && p.n_size == 16
                            && p.k_size == 16
                            && p.a_type == vk::ComponentTypeKHR::FLOAT16
                            && p.b_type == vk::ComponentTypeKHR::FLOAT16
                            && p.c_type == vk::ComponentTypeKHR::FLOAT32
                            && p.result_type == vk::ComponentTypeKHR::FLOAT32
                            && f12.shader_float16 == vk::TRUE
                        {
                            coopmat_f16_f32acc = true;
                        }
                        if p.m_size == 16
                            && p.n_size == 16
                            && p.k_size == 16
                            && p.a_type == vk::ComponentTypeKHR::SINT8
                            && p.b_type == vk::ComponentTypeKHR::SINT8
                            && p.c_type == vk::ComponentTypeKHR::SINT32
                            && p.result_type == vk::ComponentTypeKHR::SINT32
                            && f12.shader_int8 == vk::TRUE
                        {
                            coopmat_i8_i32acc = true;
                        }
                        coopmat_shapes.push(format!(
                            "{}x{}x{} {}*{}+{}->{}",
                            p.m_size,
                            p.n_size,
                            p.k_size,
                            comp_type_short(p.a_type),
                            comp_type_short(p.b_type),
                            comp_type_short(p.c_type),
                            comp_type_short(p.result_type),
                        ));
                    }
                }
            }
        }

        let info = DeviceInfo {
            index,
            name: cstr_to_string(props.device_name_as_c_str()),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_type: type_name(props.device_type),
            driver_name: cstr_to_string(drv.driver_name_as_c_str()),
            driver_info: cstr_to_string(drv.driver_info_as_c_str()),
            api_version: (
                vk::api_version_major(api),
                vk::api_version_minor(api),
                vk::api_version_patch(api),
            ),
            subgroup_size: chosen_sg,
            min_subgroup_size: min_sg,
            max_subgroup_size: max_sg,
            subgroup_size_control: size_control,
            subgroup_shuffle: sg
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::SHUFFLE),
            max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
            max_workgroup_count: props.limits.max_compute_work_group_count,
            max_shared_memory: props.limits.max_compute_shared_memory_size,
            max_alloc_bytes: m3.max_memory_allocation_size,
            max_storage_buffer_range: props.limits.max_storage_buffer_range as u64,
            min_storage_buffer_offset_alignment: props.limits.min_storage_buffer_offset_alignment,
            timestamp_period_ns: props.limits.timestamp_period,
            timestamp_compute: qf_ts_bits > 0 && props.limits.timestamp_period > 0.0,
            shader_float16: f12.shader_float16 == vk::TRUE,
            shader_int8: f12.shader_int8 == vk::TRUE,
            storage_16bit: f11.storage_buffer16_bit_access == vk::TRUE,
            storage_8bit: f12.storage_buffer8_bit_access == vk::TRUE,
            int8_dot_accelerated: api_ge_13
                && idp.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE,
            cooperative_matrix: enable_coopmat,
            coopmat_f16_f32acc,
            coopmat_i8_i32acc,
            coopmat_shapes,
            shader_bfloat16: has_ext(c"VK_KHR_shader_bfloat16"),
            shader_float8: has_ext(c"VK_EXT_shader_float8"),
            device_local_bytes: device_local,
            host_visible_device_local_bytes: hv_device_local,
            unified_memory: props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU,
            external_memory_host: has_ext_host
                && ext_host_props.min_imported_host_pointer_alignment > 0,
            external_memory_host_align: if has_ext_host {
                ext_host_props.min_imported_host_pointer_alignment
            } else {
                0
            },
        };
        Some(Probe {
            info,
            queue_family: qf,
            f11_storage16: f11.storage_buffer16_bit_access == vk::TRUE,
            f12_int8: f12.shader_int8 == vk::TRUE,
            f12_fp16: f12.shader_float16 == vk::TRUE,
            f12_storage8: f12.storage_buffer8_bit_access == vk::TRUE,
            f12_vmm,
            f12_vmm_dev_scope,
            f13_size_control: size_control,
            f13_full_subgroups: api_ge_13 && f13.compute_full_subgroups == vk::TRUE,
            core_int16,
            enable_coopmat,
        })
    }
}

fn load_instance() -> Result<(ash::Entry, ash::Instance), VulkanError> {
    // SAFETY: loading libvulkan via dlopen; the returned entry keeps the library alive.
    let entry = unsafe { ash::Entry::load() }.map_err(|e| VulkanError::Loader(e.to_string()))?;
    // SAFETY: plain query on a valid entry.
    let loader_ver = unsafe { entry.try_enumerate_instance_version() }
        .ctx("vkEnumerateInstanceVersion")?
        .unwrap_or(vk::API_VERSION_1_0);
    let api = if loader_ver >= vk::API_VERSION_1_3 {
        vk::API_VERSION_1_3
    } else {
        vk::API_VERSION_1_2
    };
    let app = vk::ApplicationInfo::default()
        .application_name(c"tinymodel")
        .engine_name(c"tinymodel")
        .api_version(api);
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: `ci` and `app` outlive the call.
    let instance = unsafe { entry.create_instance(&ci, None) }.ctx("vkCreateInstance")?;
    Ok((entry, instance))
}

impl VulkanContext {
    /// Lists every Vulkan device visible to the loader (for `probe`).
    pub fn list_devices() -> Result<Vec<DeviceInfo>, VulkanError> {
        let (entry, instance) = load_instance()?;
        // SAFETY: valid instance.
        let pdevs = unsafe { instance.enumerate_physical_devices() };
        let out = pdevs
            .map(|v| {
                v.into_iter()
                    .enumerate()
                    .filter_map(|(i, p)| probe_device(&entry, &instance, p, i).map(|p| p.info))
                    .collect()
            })
            .ctx("vkEnumeratePhysicalDevices");
        // SAFETY: no objects were created from the instance.
        unsafe { instance.destroy_instance(None) };
        out
    }

    /// Creates a context on the device selected by `TM_VK_DEVICE` (index or name
    /// substring), or the best device (discrete > integrated > virtual > cpu).
    pub fn new() -> Result<Self, VulkanError> {
        Self::with_selector(std::env::var("TM_VK_DEVICE").ok().as_deref())
    }

    /// Like [`VulkanContext::new`] with an explicit selector (index or name substring).
    pub fn with_selector(selector: Option<&str>) -> Result<Self, VulkanError> {
        Self::with_selector_and_subgroup(selector, None)
    }

    /// Creates a context on `selector` while optionally pinning compute pipelines to an explicit
    /// power-of-two `required_subgroup` (`8`, `16`, `32`, or `64`) via `VK_EXT_subgroup_size_control`.
    pub fn with_selector_and_subgroup(
        selector: Option<&str>,
        required_subgroup: Option<u32>,
    ) -> Result<Self, VulkanError> {
        let (entry, instance) = load_instance()?;
        match Self::create(&entry, &instance, selector, required_subgroup) {
            Ok((pdev_probe, pdev, device, queue)) => {
                // SAFETY: valid instance / physical device.
                let mem_props = unsafe { instance.get_physical_device_memory_properties(pdev) };
                let queue_family = pdev_probe.queue_family;
                let ext_host = pdev_probe
                    .info
                    .external_memory_host
                    .then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
                let inner = Arc::new(Inner {
                    _entry: entry,
                    instance,
                    device,
                    queue,
                    queue_family,
                    mem_props,
                    ext_host,
                });
                let exec = Self::create_exec(&inner, queue_family)?;
                Ok(Self {
                    inner,
                    info: pdev_probe.info,
                    exec: Mutex::new(exec),
                    pipelines: Mutex::new(HashMap::new()),
                    gemv_cfg: Mutex::new((
                        crate::kernels::DEFAULT_GEMV_WG,
                        crate::kernels::DEFAULT_GEMV_NR,
                    )),
                    prefill_mode: Mutex::new(VkPrefillMode::from_env()),
                    force_lds_recurrence: Mutex::new(false),
                    force_slab_copy: Mutex::new(false),
                })
            }
            Err(e) => {
                // SAFETY: nothing else was created from the instance on this path.
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    /// Returns the active batched-prefill execution policy.
    pub fn prefill_mode(&self) -> VkPrefillMode {
        self.prefill_mode
            .lock()
            .map(|m| *m)
            .unwrap_or(VkPrefillMode::Auto)
    }

    /// Overrides the batched-prefill execution policy for subsequent runner creations.
    pub fn set_prefill_mode(&self, mode: VkPrefillMode) {
        if let Ok(mut m) = self.prefill_mode.lock() {
            *m = mode;
        }
    }

    /// Forces `Qwen35VkRunner` to use the subgroup-shuffle-free LDS recurrence fallback
    /// (`gdn_recurrence_lds.comp`, designed for Pi 5 `V3DV`).
    pub fn set_force_lds_recurrence(&self, force: bool) {
        if let Ok(mut f) = self.force_lds_recurrence.lock() {
            *f = force;
        }
    }

    pub fn force_lds_recurrence(&self) -> bool {
        self.force_lds_recurrence
            .lock()
            .map(|f| *f)
            .unwrap_or(false)
    }

    /// Forces UMA runners to allocate mapped slabs instead of importing GGUF host pointers via
    /// `VK_EXT_external_memory_host` (used for A/B testing).
    pub fn set_force_slab_copy(&self, force: bool) {
        if let Ok(mut f) = self.force_slab_copy.lock() {
            *f = force;
        }
    }

    pub fn force_slab_copy(&self) -> bool {
        self.force_slab_copy.lock().map(|f| *f).unwrap_or(false)
    }

    fn create(
        entry: &ash::Entry,
        instance: &ash::Instance,
        selector: Option<&str>,
        required_subgroup: Option<u32>,
    ) -> Result<(Probe, vk::PhysicalDevice, ash::Device, vk::Queue), VulkanError> {
        // SAFETY: valid instance.
        let pdevs =
            unsafe { instance.enumerate_physical_devices() }.ctx("vkEnumeratePhysicalDevices")?;
        let mut probes: Vec<(Probe, vk::PhysicalDevice)> = pdevs
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| probe_device(entry, instance, p, i).map(|pr| (pr, p)))
            .collect();
        if probes.is_empty() {
            return Err(VulkanError::NoDevice);
        }
        let pos = match selector.map(str::trim).filter(|s| !s.is_empty()) {
            Some(sel) => if let Ok(idx) = sel.parse::<usize>() {
                probes.iter().position(|(p, _)| p.info.index == idx)
            } else {
                let s = sel.to_lowercase();
                probes.iter().position(|(p, _)| {
                    p.info.name.to_lowercase().contains(&s)
                        || p.info.driver_name.to_lowercase().contains(&s)
                })
            }
            .ok_or_else(|| VulkanError::DeviceNotFound(sel.to_string()))?,
            None => {
                let mut best = 0;
                let mut best_score = 0;
                for (i, &p) in pdevs.iter().enumerate() {
                    // SAFETY: `p` comes from `enumerate_physical_devices` on `instance`.
                    let t = unsafe { instance.get_physical_device_properties(p) }.device_type;
                    if let Some(pos) = probes.iter().position(|(pr, _)| pr.info.index == i)
                        && type_score(t) > best_score
                    {
                        best_score = type_score(t);
                        best = pos;
                    }
                }
                best
            }
        };
        let (mut probe, pdev) = probes.swap_remove(pos);
        if let Some(req_sg) = required_subgroup {
            if !req_sg.is_power_of_two() || !(8..=64).contains(&req_sg) {
                return Err(VulkanError::InvalidDimensions(format!(
                    "requested subgroup size {req_sg} must be a power of two in 8..=64"
                )));
            }
            if req_sg < probe.info.min_subgroup_size || req_sg > probe.info.max_subgroup_size {
                return Err(VulkanError::MissingFeature {
                    device: probe.info.name.clone(),
                    feature: "requested subgroup size outside [minSubgroupSize, maxSubgroupSize]",
                });
            }
            if req_sg != probe.info.subgroup_size && !probe.f13_size_control {
                return Err(VulkanError::MissingFeature {
                    device: probe.info.name.clone(),
                    feature: "subgroupSizeControl",
                });
            }
            probe.info.subgroup_size = req_sg;
        }

        let (maj, min, _) = probe.info.api_version;
        if maj == 1 && min < 2 {
            return Err(VulkanError::ApiTooOld {
                major: maj,
                minor: min,
            });
        }
        if !probe.f11_storage16 {
            return Err(VulkanError::MissingFeature {
                device: probe.info.name.clone(),
                feature: "storageBuffer16BitAccess",
            });
        }

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(probe.queue_family)
            .queue_priorities(&prio)];
        let core = vk::PhysicalDeviceFeatures::default().shader_int16(probe.core_int16);
        let mut f11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_int8(probe.f12_int8)
            .shader_float16(probe.f12_fp16)
            .storage_buffer8_bit_access(probe.f12_storage8)
            .vulkan_memory_model(probe.f12_vmm && probe.enable_coopmat)
            .vulkan_memory_model_device_scope(probe.f12_vmm_dev_scope && probe.enable_coopmat);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(probe.f13_size_control)
            .compute_full_subgroups(probe.f13_full_subgroups && probe.f13_size_control);
        let mut f_cm =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut ext_names: Vec<*const core::ffi::c_char> = Vec::new();
        if probe.enable_coopmat {
            ext_names.push(c"VK_KHR_cooperative_matrix".as_ptr());
        }
        if probe.info.external_memory_host {
            ext_names.push(c"VK_EXT_external_memory_host".as_ptr());
        }
        let api_ge_13 = min >= 3 || maj > 1;
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&ext_names)
            .enabled_features(&core)
            .push_next(&mut f11)
            .push_next(&mut f12);
        if api_ge_13 {
            dci = dci.push_next(&mut f13);
        }
        if probe.enable_coopmat {
            dci = dci.push_next(&mut f_cm);
        }
        // SAFETY: all create-info structs live until the call returns; enabled features and
        // extensions were all reported as supported by `probe_device`.
        let device = unsafe { instance.create_device(pdev, &dci, None) }.ctx("vkCreateDevice")?;
        // SAFETY: queue family / index 0 were requested in `qci`.
        let queue = unsafe { device.get_device_queue(probe.queue_family, 0) };
        Ok((probe, pdev, device, queue))
    }

    fn create_exec(inner: &Arc<Inner>, queue_family: u32) -> Result<Exec, VulkanError> {
        let d = &inner.device;
        // SAFETY: valid device; created objects are destroyed in `VulkanContext::drop`.
        unsafe {
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family),
                    None,
                )
                .ctx("vkCreateCommandPool")?;
            let cb = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .ctx("vkAllocateCommandBuffers")?[0];
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .ctx("vkCreateFence")?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: DESC_POOL_BUFFERS,
            }];
            let desc_pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(DESC_POOL_SETS)
                        .pool_sizes(&sizes),
                    None,
                )
                .ctx("vkCreateDescriptorPool")?;
            Ok(Exec {
                pool,
                cb,
                fence,
                desc_pool,
                staging: None,
            })
        }
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn device_name(&self) -> &str {
        &self.info.name
    }

    fn find_memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
        what: &'static str,
    ) -> Result<u32, VulkanError> {
        let types = self.inner.mem_props.memory_types_as_slice();
        let ok = |i: usize, f: vk::MemoryPropertyFlags| {
            type_bits & (1 << i) != 0 && types[i].property_flags.contains(f)
        };
        (0..types.len())
            .find(|&i| ok(i, required | preferred))
            .or_else(|| (0..types.len()).find(|&i| ok(i, required)))
            .map(|i| i as u32)
            .ok_or(VulkanError::NoMemoryType(what))
    }

    pub(crate) fn create_host_buffer(&self, bytes: u64) -> Result<VkBuffer, VulkanError> {
        self.create_buffer(bytes, true)
    }

    pub(crate) fn create_device_slab(&self, bytes: u64) -> Result<VkBuffer, VulkanError> {
        self.create_buffer(bytes, false)
    }

    fn create_buffer(&self, bytes: u64, host_visible: bool) -> Result<VkBuffer, VulkanError> {
        let size = bytes.max(16);
        let limit = self.info.max_alloc_bytes;
        if size > limit {
            return Err(VulkanError::BufferTooLarge { bytes: size, limit });
        }
        let d = &self.inner.device;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        // SAFETY: valid device; on any error path the partially created objects are freed.
        unsafe {
            let buf = d
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ctx("vkCreateBuffer")?;
            let req = d.get_buffer_memory_requirements(buf);
            let map_uma_slab = !host_visible && self.info.unified_memory;
            let (required, preferred, what) = if host_visible {
                (
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::HOST_CACHED,
                    "host-visible buffer",
                )
            } else if map_uma_slab {
                (
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED,
                    "unified-memory device slab",
                )
            } else {
                (
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                    "device-local buffer",
                )
            };
            let ty = if map_uma_slab {
                // On UMA GPUs (Intel ANV, Arm Mali, Pi 5 V3DV, lavapipe), prefer DEVICE_LOCAL +
                // HOST_VISIBLE + HOST_COHERENT (+ HOST_CACHED when offered) so weights and KV
                // slabs avoid allocating the 64 MiB staging buffer!
                self.find_memory_type(
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::HOST_CACHED,
                    what,
                )
                .or_else(|_| self.find_memory_type(req.memory_type_bits, required, preferred, what))
            } else {
                self.find_memory_type(req.memory_type_bits, required, preferred, what)
            };
            let ty = match ty {
                Ok(t) => t,
                Err(e) => {
                    d.destroy_buffer(buf, None);
                    return Err(e);
                }
            };
            let ty_flags = self.inner.mem_props.memory_types_as_slice()[ty as usize].property_flags;
            let is_coherent_mappable = ty_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mem = match d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            ) {
                Ok(m) => m,
                Err(code) => {
                    d.destroy_buffer(buf, None);
                    return Err(VulkanError::Vk {
                        call: "vkAllocateMemory",
                        code,
                    });
                }
            };
            let mut out = VkBuffer {
                inner: self.inner.clone(),
                buf,
                mem,
                size,
                mapped: None,
            };
            d.bind_buffer_memory(buf, mem, 0)
                .ctx("vkBindBufferMemory")?;
            if host_visible {
                let p = d
                    .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .ctx("vkMapMemory")?;
                out.mapped = Some(p as *mut u8);
            } else if map_uma_slab
                && is_coherent_mappable
                && let Ok(p) = d.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            {
                out.mapped = Some(p as *mut u8);
            }
            Ok(out)
        }
    }

    /// Attempts to import a read-only GGUF host memory region directly into a Vulkan buffer
    /// via `VK_EXT_external_memory_host`.
    ///
    /// Hardware DRM drivers on Linux (`i915` ANV, `amdgpu` RADV, `panthor` PanVK) pin imported
    /// `HOST_ALLOCATION_EXT` (`USERPTR`) pages with `FOLL_WRITE` because Vulkan has no read-only
    /// external-host handle flag; on `i915` ANV, `vkAllocateMemory` succeeds lazily and the first
    /// `vkQueueSubmit` (`execbuf2`) faults with `-EFAULT` -> permanent `VK_ERROR_DEVICE_LOST` on a
    /// `PROT_READ` file `mmap`! Hardware UMA GPUs therefore use direct-mapped `HOST_VISIBLE |
    /// HOST_COHERENT` slabs (`uma-mapped`, zero 64 MiB staging buffer), while CPU Vulkan drivers
    /// (`lavapipe` / `llvmpipe`, `device_type == "cpu"`) safely import `PROT_READ` `mmap` zero-copy.
    pub(crate) fn try_import_host_readonly(
        &self,
        data: &[u8],
        is_os_mmap: bool,
    ) -> Option<VkBuffer> {
        if self.info.device_type != "cpu"
            || !self.info.external_memory_host
            || self.force_slab_copy()
        {
            return None;
        }
        let ext_host = self.inner.ext_host.as_ref()?;
        let align = self.info.external_memory_host_align as usize;
        if align == 0 || !align.is_power_of_two() || data.is_empty() {
            return None;
        }
        let ptr_addr = data.as_ptr() as usize;
        if !ptr_addr.is_multiple_of(align) {
            return None;
        }
        // On Unix/Linux, `mmap` maps whole `>= 4096`-byte OS pages; only round up within that
        // guaranteed OS page boundary, else round down so we never touch unmapped virtual pages.
        let vma_safe_ceil = if is_os_mmap {
            (data.len() + 4095) & !4095
        } else {
            data.len()
        };
        let import_len = if is_os_mmap && align <= 4096 {
            (data.len() + (align - 1)) & !(align - 1)
        } else {
            data.len() & !(align - 1)
        };
        if import_len == 0 || (import_len as u64) > self.info.max_alloc_bytes {
            return None;
        }

        let d = &self.inner.device;
        // SAFETY: `data` is a valid host mapping of at least `import_len` bytes (kept alive by
        // `Arc<GgufFile>` inside the caller runner until after the returned `VkBuffer` drops);
        // on any error path all partially created Vulkan handles are destroyed before returning.
        unsafe {
            let mut ptr_props = vk::MemoryHostPointerPropertiesEXT::default();
            if (ext_host.fp().get_memory_host_pointer_properties_ext)(
                ext_host.device(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                data.as_ptr() as *const core::ffi::c_void,
                &mut ptr_props,
            ) != vk::Result::SUCCESS
            {
                return None;
            }

            let mut ext_buf_ci = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
            let buf_ci = vk::BufferCreateInfo::default()
                .size(import_len as u64)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut ext_buf_ci);
            let buf = d.create_buffer(&buf_ci, None).ok()?;
            let req = d.get_buffer_memory_requirements(buf);
            if req.size > vma_safe_ceil as u64 || req.size > self.info.max_alloc_bytes {
                d.destroy_buffer(buf, None);
                return None;
            }
            let allowed = req.memory_type_bits & ptr_props.memory_type_bits;
            let ty = match self
                .find_memory_type(
                    allowed,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_CACHED,
                    "imported host buffer",
                )
                .or_else(|_| {
                    self.find_memory_type(
                        allowed,
                        vk::MemoryPropertyFlags::empty(),
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        "imported host buffer",
                    )
                }) {
                Ok(t) => t,
                Err(_) => {
                    d.destroy_buffer(buf, None);
                    return None;
                }
            };

            let alloc_size = req.size.max(import_len as u64);
            let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(data.as_ptr() as *mut core::ffi::c_void);
            let alloc_ci = vk::MemoryAllocateInfo::default()
                .allocation_size(alloc_size)
                .memory_type_index(ty)
                .push_next(&mut import_info);
            let mem = match d.allocate_memory(&alloc_ci, None) {
                Ok(m) => m,
                Err(_) => {
                    d.destroy_buffer(buf, None);
                    return None;
                }
            };
            if d.bind_buffer_memory(buf, mem, 0).is_err() {
                d.destroy_buffer(buf, None);
                d.free_memory(mem, None);
                return None;
            }
            Some(VkBuffer {
                inner: self.inner.clone(),
                buf,
                mem,
                size: import_len as u64,
                mapped: None,
            })
        }
    }

    /// Allocates a zero-initialized device-local buffer.
    pub fn alloc_buffer(&self, bytes: u64) -> Result<VkBuffer, VulkanError> {
        let b = self.create_buffer(bytes, false)?;
        self.run(|rec| {
            // SAFETY: `b` is a valid transfer-dst buffer of `b.size` bytes.
            unsafe {
                rec.ctx
                    .inner
                    .device
                    .cmd_fill_buffer(rec.cb, b.buf, 0, vk::WHOLE_SIZE, 0)
            };
            Ok(())
        })?;
        Ok(b)
    }

    /// Uploads raw bytes into a new device-local buffer through the staging ring.
    pub fn upload_bytes(&self, data: &[u8]) -> Result<VkBuffer, VulkanError> {
        let dst = self.create_buffer(data.len() as u64, false)?;
        self.write_bytes(&dst, 0, data)?;
        Ok(dst)
    }

    /// Uploads a POD slice (e.g. `f32`, `BlockQ4_0`) into a new device-local buffer.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<VkBuffer, VulkanError> {
        // SAFETY: `T: Copy` POD data viewed as bytes for the duration of the copy.
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
        };
        self.upload_bytes(bytes)
    }

    /// Copies `data` into `dst` at byte `offset` (direct `memcpy` on coherent mapped UMA slabs,
    /// or chunked through the 64 MiB staging buffer on discrete GPUs).
    pub fn write_bytes(&self, dst: &VkBuffer, offset: u64, data: &[u8]) -> Result<(), VulkanError> {
        if offset + data.len() as u64 > dst.size {
            return Err(VulkanError::InvalidDimensions(format!(
                "write of {} bytes at {offset} overflows buffer of {} bytes",
                data.len(),
                dst.size
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        if let Some(dst_ptr) = dst.mapped {
            // SAFETY: `dst` is persistently mapped `HOST_VISIBLE | HOST_COHERENT`, idle on the GPU
            // (all submissions in `VulkanContext` wait on their fence), and `offset + data.len() <= dst.size`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    dst_ptr.add(offset as usize),
                    data.len(),
                );
            }
            return Ok(());
        }
        let chunk = STAGING_BYTES as usize;
        for (i, part) in data.chunks(chunk).enumerate() {
            let staging_ptr = self.staging_ptr()?;
            // SAFETY: the staging buffer is host-visible, coherent, `STAGING_BYTES` long and
            // idle (previous copy was waited for); `part.len() <= STAGING_BYTES`.
            unsafe { core::ptr::copy_nonoverlapping(part.as_ptr(), staging_ptr, part.len()) };
            let dst_off = offset + (i * chunk) as u64;
            self.run(|rec| {
                let exec = rec.ctx.exec.lock().map_err(|_| VulkanError::NoDevice)?;
                let src = exec
                    .staging
                    .as_ref()
                    .map(|s| s.buf)
                    .ok_or(VulkanError::NoDevice)?;
                drop(exec);
                // SAFETY: both buffers are valid with sufficient size (checked above).
                unsafe {
                    rec.ctx.inner.device.cmd_copy_buffer(
                        rec.cb,
                        src,
                        dst.buf,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: dst_off,
                            size: part.len() as u64,
                        }],
                    )
                };
                Ok(())
            })?;
        }
        Ok(())
    }

    fn staging_ptr(&self) -> Result<*mut u8, VulkanError> {
        let mut exec = self.exec.lock().map_err(|_| VulkanError::NoDevice)?;
        if exec.staging.is_none() {
            exec.staging = Some(self.create_buffer(STAGING_BYTES, true)?);
        }
        exec.staging
            .as_ref()
            .and_then(|s| s.mapped)
            .ok_or(VulkanError::NoMemoryType("staging buffer"))
    }

    /// Reads back `n` floats from the start of `src`.
    pub fn read_f32(&self, src: &VkBuffer, n: usize) -> Result<Vec<f32>, VulkanError> {
        let bytes = (n * 4) as u64;
        if bytes > src.size {
            return Err(VulkanError::InvalidDimensions(format!(
                "read of {bytes} bytes from buffer of {} bytes",
                src.size
            )));
        }
        let mut out = vec![0f32; n];
        if n == 0 {
            return Ok(out);
        }
        if let Some(src_ptr) = src.mapped {
            // SAFETY: `src` is persistently mapped `HOST_VISIBLE | HOST_COHERENT`, idle after the
            // caller's fence-synchronized dispatch, and `bytes <= src.size`.
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr as *const f32, out.as_mut_ptr(), n);
            }
            return Ok(out);
        }
        let chunk = STAGING_BYTES;
        let mut off = 0u64;
        while off < bytes {
            let len = chunk.min(bytes - off);
            let staging_ptr = self.staging_ptr()?;
            self.run(|rec| {
                let exec = rec.ctx.exec.lock().map_err(|_| VulkanError::NoDevice)?;
                let dst = exec
                    .staging
                    .as_ref()
                    .map(|s| s.buf)
                    .ok_or(VulkanError::NoDevice)?;
                drop(exec);
                // SAFETY: valid buffers; range checked above.
                unsafe {
                    rec.ctx.inner.device.cmd_copy_buffer(
                        rec.cb,
                        src.buf,
                        dst,
                        &[vk::BufferCopy {
                            src_offset: off,
                            dst_offset: 0,
                            size: len,
                        }],
                    )
                };
                Ok(())
            })?;
            // SAFETY: the copy above completed (run waits on the fence); staging memory is
            // host-coherent; destination range lies within `out`.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    staging_ptr,
                    (out.as_mut_ptr() as *mut u8).add(off as usize),
                    len as usize,
                )
            };
            off += len;
        }
        Ok(out)
    }

    /// Creates (or fetches from cache) a compute pipeline.
    ///
    /// `spec` values are bound to `constant_id = 0..spec.len()`; `spec[0]` must be the
    /// workgroup size (`local_size_x_id = 0`).
    pub(crate) fn pipeline(
        &self,
        name: &'static str,
        key: u32,
        spv: &[u8],
        n_bindings: u32,
        push_bytes: u32,
        spec: &[u32],
    ) -> Result<Arc<Pipeline>, VulkanError> {
        if let Some(p) = self
            .pipelines
            .lock()
            .map_err(|_| VulkanError::NoDevice)?
            .get(&(name, key))
        {
            return Ok(p.clone());
        }
        let wg = spec.first().copied().unwrap_or(1);
        if wg > self.info.max_workgroup_invocations {
            return Err(VulkanError::WorkgroupTooLarge {
                kernel: name,
                requested: wg,
                limit: self.info.max_workgroup_invocations,
            });
        }
        if !wg.is_multiple_of(self.info.subgroup_size) {
            return Err(VulkanError::InvalidDimensions(format!(
                "workgroup {wg} of '{name}' is not a multiple of subgroup size {}",
                self.info.subgroup_size
            )));
        }
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spv))
            .map_err(|e| VulkanError::Loader(format!("bad SPIR-V for {name}: {e}")))?;
        let d = &self.inner.device;
        let entries: Vec<vk::SpecializationMapEntry> = (0..spec.len() as u32)
            .map(|i| vk::SpecializationMapEntry {
                constant_id: i,
                offset: i * 4,
                size: 4,
            })
            .collect();
        let spec_bytes: Vec<u8> = spec.iter().flat_map(|v| v.to_ne_bytes()).collect();
        // SAFETY: valid device; SPIR-V words are validated by the driver; all create-info
        // structs outlive the calls; objects are destroyed in `VulkanContext::drop`.
        let pipe = unsafe {
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .ctx("vkCreateShaderModule")?;
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .ctx("vkCreateDescriptorSetLayout")?;
            let ranges = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_bytes,
            }];
            let set_layouts = [set_layout];
            let mut lci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if push_bytes > 0 {
                lci = lci.push_constant_ranges(&ranges);
            }
            let layout = d
                .create_pipeline_layout(&lci, None)
                .ctx("vkCreatePipelineLayout")?;
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&spec_bytes);
            let mut req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(self.info.subgroup_size);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&spec_info);
            if self.info.subgroup_size_control {
                stage = stage
                    .push_next(&mut req)
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
            }
            let ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let res = d.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None);
            d.destroy_shader_module(module, None);
            let pipeline = res
                .map_err(|(_, code)| VulkanError::Vk {
                    call: "vkCreateComputePipelines",
                    code,
                })?
                .remove(0);
            Pipeline {
                pipeline,
                layout,
                set_layout,
                n_bindings,
                wg_size: wg,
                name,
            }
        };
        let p = Arc::new(pipe);
        self.pipelines
            .lock()
            .map_err(|_| VulkanError::NoDevice)?
            .insert((name, key), p.clone());
        Ok(p)
    }

    /// Records commands via `f` into the shared command buffer, submits and waits.
    pub(crate) fn run<F>(&self, f: F) -> Result<(), VulkanError>
    where
        F: FnOnce(&mut Recorder<'_>) -> Result<(), VulkanError>,
    {
        let (cb, fence, desc_pool) = {
            let e = self.exec.lock().map_err(|_| VulkanError::NoDevice)?;
            (e.cb, e.fence, e.desc_pool)
        };
        let d = &self.inner.device;
        // SAFETY: single-threaded use of the command buffer is enforced by taking `&self`
        // in synchronous calls only (each call waits for completion before returning).
        unsafe {
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .ctx("vkResetCommandBuffer")?;
            d.reset_descriptor_pool(desc_pool, vk::DescriptorPoolResetFlags::empty())
                .ctx("vkResetDescriptorPool")?;
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .ctx("vkBeginCommandBuffer")?;
            // Ensure prior host writes to persistently mapped `HOST_VISIBLE | HOST_COHERENT`
            // buffers invalidate any stale GPU L2/Mall cache lines before shaders/transfers read.
            let host_in = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)];
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &host_in,
                &[],
                &[],
            );
        }
        let mut rec = Recorder {
            ctx: self,
            cb,
            desc_pool,
        };
        let r = f(&mut rec);
        // SAFETY: as above; the fence is unsignaled (reset after every wait).
        unsafe {
            // Vulkan 1.2 §7.9: `vkWaitForFences` alone does not flush discrete GPU L2/Mall caches
            // (`SHADER_WRITE | TRANSFER_WRITE`) to `HOST_READ`; record an explicit epilogue barrier.
            let host_out = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &host_out,
                &[],
                &[],
            );
            d.end_command_buffer(cb).ctx("vkEndCommandBuffer")?;
            r?;
            let cbs = [cb];
            let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
            d.queue_submit(self.inner.queue, &submit, fence)
                .ctx("vkQueueSubmit")?;
            d.wait_for_fences(&[fence], true, u64::MAX)
                .ctx("vkWaitForFences")?;
            d.reset_fences(&[fence]).ctx("vkResetFences")?;
        }
        Ok(())
    }
}

impl Recorder<'_> {
    /// Records one 1D dispatch followed by a compute→compute/transfer memory barrier.
    pub(crate) fn dispatch(
        &mut self,
        p: &Pipeline,
        bufs: &[&VkBuffer],
        push: &[u32],
        groups: u32,
    ) -> Result<(), VulkanError> {
        self.dispatch_xy(p, bufs, push, groups, 1)
    }

    /// Records one 2D dispatch `(gx, gy, 1)` followed by a compute→compute/transfer barrier.
    pub(crate) fn dispatch_xy(
        &mut self,
        p: &Pipeline,
        bufs: &[&VkBuffer],
        push: &[u32],
        gx: u32,
        gy: u32,
    ) -> Result<(), VulkanError> {
        if bufs.len() as u32 != p.n_bindings {
            return Err(VulkanError::InvalidDimensions(format!(
                "'{}' expects {} buffers, got {}",
                p.name,
                p.n_bindings,
                bufs.len()
            )));
        }
        if p.wg_size > self.ctx.info.max_workgroup_invocations {
            return Err(VulkanError::WorkgroupTooLarge {
                kernel: p.name,
                requested: p.wg_size,
                limit: self.ctx.info.max_workgroup_invocations,
            });
        }
        let lim = self.ctx.info.max_workgroup_count;
        if gx > lim[0] || gy > lim[1] {
            return Err(VulkanError::InvalidDimensions(format!(
                "dispatch ({gx}, {gy}, 1) for '{}' exceeds maxComputeWorkGroupCount {:?}",
                p.name, lim
            )));
        }
        let d = &self.ctx.inner.device;
        let infos: Vec<vk::DescriptorBufferInfo> = bufs
            .iter()
            .map(|b| vk::DescriptorBufferInfo {
                buffer: b.buf,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let push_bytes: Vec<u8> = push.iter().flat_map(|v| v.to_ne_bytes()).collect();
        // SAFETY: the pipeline, layout and buffers are valid for the lifetime of this
        // recording; descriptor set is freshly allocated from the per-run pool.
        unsafe {
            let layouts = [p.set_layout];
            let set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.desc_pool)
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
            d.cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            d.cmd_bind_descriptor_sets(
                self.cb,
                vk::PipelineBindPoint::COMPUTE,
                p.layout,
                0,
                &[set],
                &[],
            );
            if !push_bytes.is_empty() {
                d.cmd_push_constants(
                    self.cb,
                    p.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push_bytes,
                );
            }
            d.cmd_dispatch(self.cb, gx, gy, 1);
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ,
                );
            d.cmd_pipeline_barrier(
                self.cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        let d = &self.inner.device;
        // SAFETY: wait idle, then destroy every object created from this device by the
        // context. Buffers still alive keep `Inner` (and thus the device) alive.
        unsafe {
            let _ = d.device_wait_idle();
            if let Ok(mut map) = self.pipelines.lock() {
                for (_, p) in map.drain() {
                    d.destroy_pipeline(p.pipeline, None);
                    d.destroy_pipeline_layout(p.layout, None);
                    d.destroy_descriptor_set_layout(p.set_layout, None);
                }
            }
            if let Ok(mut e) = self.exec.lock() {
                e.staging = None;
                d.destroy_descriptor_pool(e.desc_pool, None);
                d.destroy_fence(e.fence, None);
                d.destroy_command_pool(e.pool, None);
            }
        }
    }
}
