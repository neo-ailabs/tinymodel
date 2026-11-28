//! Minimal, zero-dependency dynamic loader & safe RAII wrapper for the NVIDIA CUDA Driver API (`libcuda`).
//!
//! Automatically searches Google Colab's `/usr/lib64-nvidia/libcuda.so.1` alongside standard Linux/Windows
//! driver paths, validates launch block dimensions and shared-memory budgets before every `cuLaunchKernel`,
//! and encapsulates all `unsafe` FFI calls with explicit `// SAFETY:` invariants (`AGENTS.md` Rule #2).

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

use libloading::Library;

use crate::CudaError;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

// CUdevice_attribute constants from <cuda.h>
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
const CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE: c_int = 108;

const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 0x02;
const CU_MEMHOSTREGISTER_DEVICEMAP: c_uint = 0x02;
const CU_LIMIT_PERSISTING_L2_CACHE_SIZE: c_int = 0x06;
const CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW: c_int = 1;
const CU_ACCESS_PROPERTY_STREAMING: c_int = 1;
const CU_ACCESS_PROPERTY_PERSISTING: c_int = 2;

#[repr(C)]
pub(crate) struct CuAccessPolicyWindowAttr {
    base_ptr: *mut c_void,
    num_bytes: usize,
    hit_ratio: f32,
    hit_prop: c_int,
    miss_prop: c_int,
    _pad: [u8; 40],
}

type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnCuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnCuGetErrorName = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type FnCuGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRelease = unsafe extern "C" fn(CUdevice) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxSetLimit = unsafe extern "C" fn(c_int, usize) -> CUresult;
type FnCuMemGetInfoV2 = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
type FnCuMemAllocV2 = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnCuMemFreeV2 = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type FnCuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult;
type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
type FnCuMemHostRegister = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CUresult;
type FnCuMemHostUnregister = unsafe extern "C" fn(*mut c_void) -> CUresult;
type FnCuMemHostGetDevicePointerV2 =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, c_uint) -> CUresult;
type FnCuMemsetD8V2 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
type FnCuMemsetD8Async = unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult;
type FnCuMemcpyHtoDV2 = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnCuMemcpyDtoHV2 = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnCuMemcpyDtoDV2 = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
type FnCuMemcpyDtoDAsyncV2 =
    unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuMemcpyHtoDAsyncV2 =
    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoHAsyncV2 =
    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type FnCuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
type FnCuStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
type FnCuStreamDestroyV2 = unsafe extern "C" fn(CUstream) -> CUresult;
type FnCuStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
type FnCuStreamSetAttribute =
    unsafe extern "C" fn(CUstream, c_int, *const CuAccessPolicyWindowAttr) -> CUresult;
type FnCuEventCreate = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
type FnCuEventDestroyV2 = unsafe extern "C" fn(CUevent) -> CUresult;
type FnCuEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type FnCuEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
type FnCuEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type FnCuStreamBeginCaptureV2 = unsafe extern "C" fn(CUstream, c_int) -> CUresult;
type FnCuStreamEndCapture = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
type FnCuGraphInstantiateWithFlags =
    unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type FnCuGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
type FnCuGraphExecDestroy = unsafe extern "C" fn(CUgraphExec) -> CUresult;
type FnCuGraphDestroy = unsafe extern "C" fn(CUgraph) -> CUresult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

pub(crate) struct CudaDriver {
    _lib: Library,
    pub cu_init: FnCuInit,
    pub cu_driver_get_version: FnCuDriverGetVersion,
    pub cu_get_error_name: FnCuGetErrorName,
    pub cu_get_error_string: FnCuGetErrorString,
    pub cu_device_get_count: FnCuDeviceGetCount,
    pub cu_device_get: FnCuDeviceGet,
    pub cu_device_get_name: FnCuDeviceGetName,
    pub cu_device_get_attribute: FnCuDeviceGetAttribute,
    pub cu_device_primary_ctx_retain: FnCuDevicePrimaryCtxRetain,
    pub cu_device_primary_ctx_release: FnCuDevicePrimaryCtxRelease,
    pub cu_ctx_set_current: FnCuCtxSetCurrent,
    pub cu_ctx_set_limit: Option<FnCuCtxSetLimit>,
    pub cu_mem_get_info: FnCuMemGetInfoV2,
    pub cu_mem_alloc: FnCuMemAllocV2,
    pub cu_mem_free: FnCuMemFreeV2,
    pub cu_mem_host_alloc: Option<FnCuMemHostAlloc>,
    pub cu_mem_free_host: Option<FnCuMemFreeHost>,
    pub cu_mem_host_register: Option<FnCuMemHostRegister>,
    pub cu_mem_host_unregister: Option<FnCuMemHostUnregister>,
    pub cu_mem_host_get_device_pointer: Option<FnCuMemHostGetDevicePointerV2>,
    pub cu_memset_d8: FnCuMemsetD8V2,
    pub cu_memset_d8_async: Option<FnCuMemsetD8Async>,
    #[allow(dead_code)]
    pub cu_memcpy_htod: FnCuMemcpyHtoDV2,
    #[allow(dead_code)]
    pub cu_memcpy_dtoh: FnCuMemcpyDtoHV2,
    pub cu_memcpy_dtod: FnCuMemcpyDtoDV2,
    pub cu_memcpy_dtod_async: Option<FnCuMemcpyDtoDAsyncV2>,
    pub cu_memcpy_htod_async: FnCuMemcpyHtoDAsyncV2,
    pub cu_memcpy_dtoh_async: FnCuMemcpyDtoHAsyncV2,
    pub cu_module_load_data: FnCuModuleLoadData,
    pub cu_module_unload: FnCuModuleUnload,
    pub cu_module_get_function: FnCuModuleGetFunction,
    pub cu_func_set_attribute: FnCuFuncSetAttribute,
    pub cu_stream_create: FnCuStreamCreate,
    pub cu_stream_destroy: FnCuStreamDestroyV2,
    pub cu_stream_synchronize: FnCuStreamSynchronize,
    pub cu_stream_set_attribute: Option<FnCuStreamSetAttribute>,
    pub cu_event_create: FnCuEventCreate,
    pub cu_event_destroy: FnCuEventDestroyV2,
    pub cu_event_record: FnCuEventRecord,
    pub cu_event_synchronize: FnCuEventSynchronize,
    pub cu_event_elapsed_time: FnCuEventElapsedTime,
    pub cu_stream_begin_capture: FnCuStreamBeginCaptureV2,
    pub cu_stream_end_capture: FnCuStreamEndCapture,
    pub cu_graph_instantiate: FnCuGraphInstantiateWithFlags,
    pub cu_graph_launch: FnCuGraphLaunch,
    pub cu_graph_exec_destroy: FnCuGraphExecDestroy,
    pub cu_graph_destroy: FnCuGraphDestroy,
    pub cu_launch_kernel: FnCuLaunchKernel,
}

unsafe fn load_sym<T: Copy>(lib: &Library, name: &[u8]) -> Result<T, CudaError> {
    // SAFETY: Caller specifies the exact C ABI function pointer signature from `<cuda.h>`.
    unsafe {
        lib.get::<T>(name).map(|s| *s).map_err(|e| {
            CudaError::DriverLoad(format!(
                "missing symbol '{}': {e}",
                String::from_utf8_lossy(name)
            ))
        })
    }
}

impl CudaDriver {
    pub fn load() -> Result<Self, CudaError> {
        let candidates = [
            "libcuda.so.1",
            "/usr/lib64-nvidia/libcuda.so.1",
            "/usr/local/nvidia/lib64/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            "libcuda.so",
            "nvcuda.dll",
        ];
        let mut last_err = String::new();
        let mut loaded: Option<Library> = None;
        for cand in candidates {
            // SAFETY: Loading the system NVIDIA CUDA user-mode driver shared library.
            match unsafe { Library::new(cand) } {
                Ok(lib) => {
                    loaded = Some(lib);
                    break;
                }
                Err(e) => last_err = format!("{cand}: {e}"),
            }
        }
        let lib = loaded.ok_or(CudaError::DriverLoad(last_err))?;

        // SAFETY: All symbols match NVIDIA's stable CUDA Driver API v2 ABI (`cuda.h`).
        unsafe {
            Ok(Self {
                cu_init: load_sym(&lib, b"cuInit\0")?,
                cu_driver_get_version: load_sym(&lib, b"cuDriverGetVersion\0")?,
                cu_get_error_name: load_sym(&lib, b"cuGetErrorName\0")?,
                cu_get_error_string: load_sym(&lib, b"cuGetErrorString\0")?,
                cu_device_get_count: load_sym(&lib, b"cuDeviceGetCount\0")?,
                cu_device_get: load_sym(&lib, b"cuDeviceGet\0")?,
                cu_device_get_name: load_sym(&lib, b"cuDeviceGetName\0")?,
                cu_device_get_attribute: load_sym(&lib, b"cuDeviceGetAttribute\0")?,
                cu_device_primary_ctx_retain: load_sym(&lib, b"cuDevicePrimaryCtxRetain\0")?,
                cu_device_primary_ctx_release: load_sym(&lib, b"cuDevicePrimaryCtxRelease_v2\0")
                    .or_else(|_| load_sym(&lib, b"cuDevicePrimaryCtxRelease\0"))?,
                cu_ctx_set_current: load_sym(&lib, b"cuCtxSetCurrent\0")?,
                cu_ctx_set_limit: load_sym(&lib, b"cuCtxSetLimit\0").ok(),
                cu_mem_get_info: load_sym(&lib, b"cuMemGetInfo_v2\0")?,
                cu_mem_alloc: load_sym(&lib, b"cuMemAlloc_v2\0")?,
                cu_mem_free: load_sym(&lib, b"cuMemFree_v2\0")?,
                cu_mem_host_alloc: load_sym(&lib, b"cuMemHostAlloc\0").ok(),
                cu_mem_free_host: load_sym(&lib, b"cuMemFreeHost\0").ok(),
                cu_mem_host_register: load_sym(&lib, b"cuMemHostRegister_v2\0")
                    .or_else(|_| load_sym(&lib, b"cuMemHostRegister\0"))
                    .ok(),
                cu_mem_host_unregister: load_sym(&lib, b"cuMemHostUnregister\0").ok(),
                cu_mem_host_get_device_pointer: load_sym(&lib, b"cuMemHostGetDevicePointer_v2\0")
                    .or_else(|_| load_sym(&lib, b"cuMemHostGetDevicePointer\0"))
                    .ok(),
                cu_memset_d8: load_sym(&lib, b"cuMemsetD8_v2\0")?,
                cu_memset_d8_async: load_sym(&lib, b"cuMemsetD8Async\0").ok(),
                cu_memcpy_htod: load_sym(&lib, b"cuMemcpyHtoD_v2\0")?,
                cu_memcpy_dtoh: load_sym(&lib, b"cuMemcpyDtoH_v2\0")?,
                cu_memcpy_dtod: load_sym(&lib, b"cuMemcpyDtoD_v2\0")?,
                cu_memcpy_dtod_async: load_sym(&lib, b"cuMemcpyDtoDAsync_v2\0").ok(),
                cu_memcpy_htod_async: load_sym(&lib, b"cuMemcpyHtoDAsync_v2\0")?,
                cu_memcpy_dtoh_async: load_sym(&lib, b"cuMemcpyDtoHAsync_v2\0")?,
                cu_module_load_data: load_sym(&lib, b"cuModuleLoadData\0")?,
                cu_module_unload: load_sym(&lib, b"cuModuleUnload\0")?,
                cu_module_get_function: load_sym(&lib, b"cuModuleGetFunction\0")?,
                cu_func_set_attribute: load_sym(&lib, b"cuFuncSetAttribute\0")?,
                cu_stream_create: load_sym(&lib, b"cuStreamCreate\0")?,
                cu_stream_destroy: load_sym(&lib, b"cuStreamDestroy_v2\0")?,
                cu_stream_synchronize: load_sym(&lib, b"cuStreamSynchronize\0")?,
                cu_stream_set_attribute: load_sym(&lib, b"cuStreamSetAttribute\0").ok(),
                cu_event_create: load_sym(&lib, b"cuEventCreate\0")?,
                cu_event_destroy: load_sym(&lib, b"cuEventDestroy_v2\0")?,
                cu_event_record: load_sym(&lib, b"cuEventRecord\0")?,
                cu_event_synchronize: load_sym(&lib, b"cuEventSynchronize\0")?,
                cu_event_elapsed_time: load_sym(&lib, b"cuEventElapsedTime\0")?,
                cu_stream_begin_capture: load_sym(&lib, b"cuStreamBeginCapture_v2\0")?,
                cu_stream_end_capture: load_sym(&lib, b"cuStreamEndCapture\0")?,
                cu_graph_instantiate: load_sym(&lib, b"cuGraphInstantiateWithFlags\0")?,
                cu_graph_launch: load_sym(&lib, b"cuGraphLaunch\0")?,
                cu_graph_exec_destroy: load_sym(&lib, b"cuGraphExecDestroy\0")?,
                cu_graph_destroy: load_sym(&lib, b"cuGraphDestroy\0")?,
                cu_launch_kernel: load_sym(&lib, b"cuLaunchKernel\0")?,
                _lib: lib,
            })
        }
    }

    #[inline]
    pub fn check(&self, call: &'static str, code: CUresult) -> Result<(), CudaError> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut name_ptr: *const c_char = ptr::null();
        let mut desc_ptr: *const c_char = ptr::null();
        // SAFETY: `cuGetErrorName` and `cuGetErrorString` write static C strings for any error code.
        let detail = unsafe {
            let _ = (self.cu_get_error_name)(code, &mut name_ptr);
            let _ = (self.cu_get_error_string)(code, &mut desc_ptr);
            let n = if !name_ptr.is_null() {
                CStr::from_ptr(name_ptr).to_string_lossy()
            } else {
                "CUDA_ERROR".into()
            };
            let d = if !desc_ptr.is_null() {
                CStr::from_ptr(desc_ptr).to_string_lossy()
            } else {
                "".into()
            };
            format!("{n} ({code}): {d}")
        };
        Err(CudaError::DriverCall { call, detail })
    }
}

/// Hardware capabilities queried from a CUDA GPU.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub ordinal: i32,
    pub name: String,
    pub driver_version: i32,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub warp_size: i32,
    pub max_threads_per_block: u32,
    pub max_shared_mem_per_block: u32,
    pub max_shared_mem_optin: u32,
    pub max_persisting_l2_bytes: usize,
    pub total_vram_bytes: usize,
    pub free_vram_bytes: usize,
    pub peak_mem_bw_gbps: f64,
}

impl CudaDeviceInfo {
    #[inline]
    pub fn has_dp4a(&self) -> bool {
        self.cc_major > 6 || (self.cc_major == 6 && self.cc_minor >= 1)
    }

    #[inline]
    pub fn has_fp16_wmma(&self) -> bool {
        self.cc_major >= 7
    }
}

struct CudaInner {
    drv: CudaDriver,
    dev: CUdevice,
    ctx: CUcontext,
    module: CUmodule,
    stream: CUstream,
    info: CudaDeviceInfo,
}

// SAFETY: CUDA primary contexts and `CUdeviceptr` allocations are thread-safe when bound via
// `cuCtxSetCurrent` before driver operations (enforced in `CudaContext::bind_current`).
unsafe impl Send for CudaInner {}
// SAFETY: Shared read-only metadata/handles; mutable GPU launches are serialized on `stream`.
unsafe impl Sync for CudaInner {}

impl Drop for CudaInner {
    fn drop(&mut self) {
        // SAFETY: Releases stream, PTX module, and primary context refcount in reverse order.
        unsafe {
            let _ = (self.drv.cu_ctx_set_current)(self.ctx);
            if !self.stream.is_null() {
                let _ = (self.drv.cu_stream_synchronize)(self.stream);
                let _ = (self.drv.cu_stream_destroy)(self.stream);
            }
            if !self.module.is_null() {
                let _ = (self.drv.cu_module_unload)(self.module);
            }
            let _ = (self.drv.cu_device_primary_ctx_release)(self.dev);
        }
    }
}

/// Shared CUDA execution context holding the primary `CUcontext`, default `CUstream`, and loaded PTX module.
#[derive(Clone)]
pub struct CudaContext {
    inner: Arc<CudaInner>,
}

const EMBEDDED_PTX_SM75: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda_ops.ptx"));
const EMBEDDED_PTX_SM80: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cuda_ops_sm80.ptx"));

impl CudaContext {
    /// Initializes CUDA, selects device `ordinal` (or `TM_CUDA_DEVICE`), retains its primary context,
    /// and JIT-loads the embedded `cuda_ops.ptx` module.
    pub fn new(ordinal_override: Option<usize>) -> Result<Self, CudaError> {
        let drv = CudaDriver::load()?;
        // SAFETY: `cuInit(0)` initializes the CUDA driver globally and is idempotent.
        unsafe { drv.check("cuInit", (drv.cu_init)(0))? };

        let mut count: c_int = 0;
        // SAFETY: Writes valid device count into `count`.
        unsafe { drv.check("cuDeviceGetCount", (drv.cu_device_get_count)(&mut count))? };
        if count <= 0 {
            return Err(CudaError::NoDevice);
        }

        let ordinal = if let Some(ord) = ordinal_override {
            ord as i32
        } else if let Ok(s) = std::env::var("TM_CUDA_DEVICE") {
            s.parse::<i32>().unwrap_or(0)
        } else {
            0
        };
        if ordinal < 0 || ordinal >= count {
            return Err(CudaError::DeviceNotFound(format!(
                "ordinal {ordinal} out of range 0..{count}"
            )));
        }

        let mut dev: CUdevice = 0;
        // SAFETY: `ordinal` validated in `0..count`.
        unsafe { drv.check("cuDeviceGet", (drv.cu_device_get)(&mut dev, ordinal))? };

        let mut name_buf = [0 as c_char; 256];
        // SAFETY: `name_buf` is 256 bytes writable.
        unsafe {
            drv.check(
                "cuDeviceGetName",
                (drv.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev),
            )?;
        }
        // SAFETY: `cuDeviceGetName` null-terminates `name_buf`.
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let attr = |a: c_int| -> Result<i32, CudaError> {
            let mut val: c_int = 0;
            // SAFETY: Valid `dev` and attribute enum.
            unsafe {
                drv.check(
                    "cuDeviceGetAttribute",
                    (drv.cu_device_get_attribute)(&mut val, a, dev),
                )?
            };
            Ok(val)
        };
        let mut driver_version: c_int = 0;
        // SAFETY: Valid output pointer.
        unsafe {
            drv.check(
                "cuDriverGetVersion",
                (drv.cu_driver_get_version)(&mut driver_version),
            )?;
        }
        let cc_major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let cc_minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let sm_count = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let warp_size = attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
        let max_threads_per_block = attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? as u32;
        let max_shared_mem_per_block =
            attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? as u32;
        let max_shared_mem_optin = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .unwrap_or(max_shared_mem_per_block as i32) as u32;
        let max_persisting_l2_bytes = attr(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)
            .unwrap_or(0)
            .max(0) as usize;
        let _clock_khz = attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE).unwrap_or(0);
        let mem_clock_khz = attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE).unwrap_or(0) as f64;
        let bus_bits = attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH).unwrap_or(0) as f64;
        // DDR double data rate: 2 * mem_clock_Hz * (bus_bits / 8) / 1e9
        let peak_mem_bw_gbps = 2.0 * (mem_clock_khz * 1e3) * (bus_bits / 8.0) / 1e9;

        let mut ctx: CUcontext = ptr::null_mut();
        // SAFETY: Retains primary context on `dev` and binds it to the calling thread.
        unsafe {
            drv.check(
                "cuDevicePrimaryCtxRetain",
                (drv.cu_device_primary_ctx_retain)(&mut ctx, dev),
            )?;
            drv.check("cuCtxSetCurrent", (drv.cu_ctx_set_current)(ctx))?;
        }

        let mut free_vram: usize = 0;
        let mut total_vram: usize = 0;
        // SAFETY: Context is current; valid output pointers.
        unsafe {
            drv.check(
                "cuMemGetInfo_v2",
                (drv.cu_mem_get_info)(&mut free_vram, &mut total_vram),
            )?;
        }

        // Dispatch `compute_80` PTX on Ampere/Ada/Hopper/Blackwell (`cc_major >= 8`), falling back to `compute_75`.
        let make_ptx_c = |raw: &[u8]| -> Vec<u8> {
            let mut v = Vec::with_capacity(raw.len() + 1);
            v.extend_from_slice(raw);
            if v.last().copied() != Some(0) {
                v.push(0);
            }
            v
        };
        let mut module: CUmodule = ptr::null_mut();
        // SAFETY: Null-terminated PTX strings; context is current.
        unsafe {
            let loaded_sm80 = if cc_major >= 8 {
                let ptx80 = make_ptx_c(EMBEDDED_PTX_SM80);
                (drv.cu_module_load_data)(&mut module, ptx80.as_ptr().cast::<c_void>())
                    == CUDA_SUCCESS
                    && !module.is_null()
            } else {
                false
            };
            if !loaded_sm80 {
                let ptx75 = make_ptx_c(EMBEDDED_PTX_SM75);
                if let Err(e) = drv.check(
                    "cuModuleLoadData",
                    (drv.cu_module_load_data)(&mut module, ptx75.as_ptr().cast::<c_void>()),
                ) {
                    let _ = (drv.cu_device_primary_ctx_release)(dev);
                    return Err(e);
                }
            }
        }

        let mut stream: CUstream = ptr::null_mut();
        // SAFETY: Context is current; `CU_STREAM_DEFAULT = 0` guarantees stream ordering with `cuMemsetD8_v2`.
        unsafe {
            if let Err(e) = drv.check("cuStreamCreate", (drv.cu_stream_create)(&mut stream, 0)) {
                let _ = (drv.cu_module_unload)(module);
                let _ = (drv.cu_device_primary_ctx_release)(dev);
                return Err(e);
            }
        }

        let info = CudaDeviceInfo {
            ordinal,
            name,
            driver_version,
            cc_major,
            cc_minor,
            sm_count,
            warp_size,
            max_threads_per_block,
            max_shared_mem_per_block,
            max_shared_mem_optin: max_shared_mem_optin.max(max_shared_mem_per_block),
            max_persisting_l2_bytes,
            total_vram_bytes: total_vram,
            free_vram_bytes: free_vram,
            peak_mem_bw_gbps,
        };

        Ok(Self {
            inner: Arc::new(CudaInner {
                drv,
                dev,
                ctx,
                module,
                stream,
                info,
            }),
        })
    }

    #[inline]
    pub fn info(&self) -> &CudaDeviceInfo {
        &self.inner.info
    }

    #[inline]
    pub fn bind_current(&self) -> Result<(), CudaError> {
        // SAFETY: `self.inner.ctx` is valid for the lifetime of `CudaInner`.
        unsafe {
            self.inner.drv.check(
                "cuCtxSetCurrent",
                (self.inner.drv.cu_ctx_set_current)(self.inner.ctx),
            )
        }
    }

    pub fn mem_info(&self) -> Result<(usize, usize), CudaError> {
        self.bind_current()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: Context bound; valid output pointers.
        unsafe {
            self.inner.drv.check(
                "cuMemGetInfo_v2",
                (self.inner.drv.cu_mem_get_info)(&mut free, &mut total),
            )?;
        }
        Ok((free, total))
    }

    pub fn get_function(&self, name: &'static str) -> Result<CUfunction, CudaError> {
        self.bind_current()?;
        let cname = CString::new(name).expect("valid C kernel name");
        let mut func: CUfunction = ptr::null_mut();
        // SAFETY: `module` is valid and `cname` is null-terminated.
        unsafe {
            self.inner.drv.check(
                "cuModuleGetFunction",
                (self.inner.drv.cu_module_get_function)(
                    &mut func,
                    self.inner.module,
                    cname.as_ptr(),
                ),
            )?;
        }
        Ok(func)
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn set_func_max_smem(
        &self,
        func: CUfunction,
        bytes: u32,
    ) -> Result<(), CudaError> {
        self.bind_current()?;
        // SAFETY: Valid `func` handle from `self.inner.module`.
        unsafe {
            self.inner.drv.check(
                "cuFuncSetAttribute",
                (self.inner.drv.cu_func_set_attribute)(
                    func,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as c_int,
                ),
            )
        }
    }

    pub fn alloc_uninit(&self, bytes: usize) -> Result<CudaBuffer, CudaError> {
        let alloc_bytes = bytes.max(16);
        self.bind_current()?;
        let mut ptr_dev: CUdeviceptr = 0;
        // SAFETY: Context is current; allocates `alloc_bytes` in GPU global memory.
        unsafe {
            self.inner.drv.check(
                "cuMemAlloc_v2",
                (self.inner.drv.cu_mem_alloc)(&mut ptr_dev, alloc_bytes),
            )?;
        }
        Ok(CudaBuffer {
            ctx: self.clone(),
            ptr: ptr_dev,
            host_mapped: ptr::null_mut(),
            bytes: alloc_bytes,
        })
    }

    pub fn alloc_zeroed(&self, bytes: usize) -> Result<CudaBuffer, CudaError> {
        let alloc_bytes = bytes.max(16);
        self.bind_current()?;
        let mut ptr_dev: CUdeviceptr = 0;
        // SAFETY: Context is current; allocates `alloc_bytes` in GPU global memory and zeroes it.
        unsafe {
            self.inner.drv.check(
                "cuMemAlloc_v2",
                (self.inner.drv.cu_mem_alloc)(&mut ptr_dev, alloc_bytes),
            )?;
            if let Err(e) = self.inner.drv.check(
                "cuMemsetD8_v2",
                (self.inner.drv.cu_memset_d8)(ptr_dev, 0, alloc_bytes),
            ) {
                let _ = (self.inner.drv.cu_mem_free)(ptr_dev);
                return Err(e);
            }
        }
        Ok(CudaBuffer {
            ctx: self.clone(),
            ptr: ptr_dev,
            host_mapped: ptr::null_mut(),
            bytes: alloc_bytes,
        })
    }

    /// Allocates page-locked host memory (`cuMemHostAlloc`) for true async DMA H2D/D2H transfers (`P1-2`).
    pub fn alloc_pinned(&self, bytes: usize) -> Result<CudaPinnedBuffer, CudaError> {
        let alloc_bytes = bytes.max(16);
        self.bind_current()?;
        let (Some(alloc_fn), Some(_free_fn)) = (
            self.inner.drv.cu_mem_host_alloc,
            self.inner.drv.cu_mem_free_host,
        ) else {
            return Err(CudaError::DriverLoad(
                "cuMemHostAlloc/cuMemFreeHost unavailable".to_string(),
            ));
        };
        let mut host_ptr: *mut c_void = ptr::null_mut();
        // SAFETY: Context is bound; `flags = 0` allocates page-locked host memory.
        unsafe {
            self.inner
                .drv
                .check("cuMemHostAlloc", alloc_fn(&mut host_ptr, alloc_bytes, 0))?;
            ptr::write_bytes(host_ptr.cast::<u8>(), 0, alloc_bytes);
        }
        Ok(CudaPinnedBuffer {
            ctx: self.clone(),
            ptr: host_ptr.cast::<u8>(),
            bytes: alloc_bytes,
        })
    }

    /// Allocates zero-copy device-mapped pinned host DDR memory (`CU_MEMHOSTALLOC_DEVICEMAP`) and copies `data`
    /// into it (`P2`), returning a `CudaBuffer` whose `device_ptr()` is directly dereferenceable by all CUDA kernels
    /// over PCIe without consuming GDDR VRAM!
    pub fn alloc_host_mapped_from_bytes(&self, data: &[u8]) -> Result<CudaBuffer, CudaError> {
        let alloc_bytes = data.len().max(16);
        self.bind_current()?;
        let (Some(alloc_fn), Some(free_fn), Some(dev_ptr_fn)) = (
            self.inner.drv.cu_mem_host_alloc,
            self.inner.drv.cu_mem_free_host,
            self.inner.drv.cu_mem_host_get_device_pointer,
        ) else {
            return Err(CudaError::DriverLoad(
                "cuMemHostAlloc/cuMemHostGetDevicePointer unavailable".to_string(),
            ));
        };
        let mut host_ptr: *mut c_void = ptr::null_mut();
        let mut ptr_dev: CUdeviceptr = 0;
        // SAFETY: Allocates pinned host memory with `CU_MEMHOSTALLOC_DEVICEMAP`, copies `data`, and queries GPU VA.
        unsafe {
            self.inner.drv.check(
                "cuMemHostAlloc(DEVICEMAP)",
                alloc_fn(&mut host_ptr, alloc_bytes, CU_MEMHOSTALLOC_DEVICEMAP),
            )?;
            if !data.is_empty() {
                ptr::copy_nonoverlapping(data.as_ptr(), host_ptr.cast::<u8>(), data.len());
            } else {
                ptr::write_bytes(host_ptr.cast::<u8>(), 0, alloc_bytes);
            }
            if let Err(e) = self.inner.drv.check(
                "cuMemHostGetDevicePointer_v2",
                dev_ptr_fn(&mut ptr_dev, host_ptr, 0),
            ) {
                let _ = free_fn(host_ptr);
                return Err(e);
            }
        }
        Ok(CudaBuffer {
            ctx: self.clone(),
            ptr: ptr_dev,
            host_mapped: host_ptr,
            bytes: alloc_bytes,
        })
    }

    /// Registers an existing page-aligned host memory region (`HugePageBytes`) with
    /// `CU_MEMHOSTREGISTER_DEVICEMAP` (`P1-2`), returning a zero-copy GPU `CUdeviceptr`
    /// that shares the exact same physical DDR pages with CPU `AVX2` workers (`0` duplicate RAM!).
    ///
    /// # Safety
    /// Caller must ensure `host_ptr .. host_ptr + bytes` remains valid and unmoved until the
    /// returned [`CudaHostRegistered`] guard is dropped.
    pub unsafe fn register_host_devicemap(
        &self,
        host_ptr: *mut u8,
        bytes: usize,
    ) -> Result<CudaHostRegistered, CudaError> {
        if bytes == 0 || host_ptr.is_null() {
            return Err(CudaError::InvalidDimensions(
                "empty host buffer for cuMemHostRegister".into(),
            ));
        }
        self.bind_current()?;
        let (Some(reg_fn), Some(unreg_fn), Some(dev_ptr_fn)) = (
            self.inner.drv.cu_mem_host_register,
            self.inner.drv.cu_mem_host_unregister,
            self.inner.drv.cu_mem_host_get_device_pointer,
        ) else {
            return Err(CudaError::DriverLoad(
                "cuMemHostRegister/cuMemHostGetDevicePointer unavailable".into(),
            ));
        };
        let raw_ptr = host_ptr.cast::<c_void>();
        let mut ptr_dev: CUdeviceptr = 0;
        // SAFETY: Caller guarantees `[host_ptr .. host_ptr + bytes)` is page-aligned and outlives `CudaHostRegistered`.
        unsafe {
            self.inner.drv.check(
                "cuMemHostRegister_v2(DEVICEMAP)",
                reg_fn(raw_ptr, bytes, CU_MEMHOSTREGISTER_DEVICEMAP),
            )?;
            if let Err(e) = self.inner.drv.check(
                "cuMemHostGetDevicePointer_v2",
                dev_ptr_fn(&mut ptr_dev, raw_ptr, 0),
            ) {
                let _ = unreg_fn(raw_ptr);
                return Err(e);
            }
        }
        Ok(CudaHostRegistered {
            ctx: self.clone(),
            host_ptr: raw_ptr,
            dev_ptr: ptr_dev,
            bytes,
        })
    }

    /// Best-effort Ampere/Ada/Hopper/Blackwell L2 persisting cache hint (`P1-3`) for hot weight ranges
    /// (e.g., MTP / DFly2 / DSpark draft trunk or shared experts). Safely no-ops on `sm_75` or restricted drivers.
    pub fn pin_l2_persisting(&self, base_ptr: CUdeviceptr, num_bytes: usize) {
        let max_l2 = self.inner.info.max_persisting_l2_bytes;
        if self.inner.info.cc_major < 8 || max_l2 == 0 || num_bytes == 0 || base_ptr == 0 {
            return;
        }
        let (Some(set_limit), Some(set_stream_attr)) = (
            self.inner.drv.cu_ctx_set_limit,
            self.inner.drv.cu_stream_set_attribute,
        ) else {
            return;
        };
        // Cap window to hardware `max_persisting_l2_bytes` and CUDA's 128 MiB per-window ceiling.
        let win_bytes = num_bytes.min(max_l2).min(128 * 1024 * 1024);
        if self.bind_current().is_err() {
            return;
        }
        let attr = CuAccessPolicyWindowAttr {
            base_ptr: base_ptr as usize as *mut c_void,
            num_bytes: win_bytes,
            hit_ratio: 1.0,
            hit_prop: CU_ACCESS_PROPERTY_PERSISTING,
            miss_prop: CU_ACCESS_PROPERTY_STREAMING,
            _pad: [0u8; 40],
        };
        // SAFETY: Best-effort hint on valid bound `ctx` and `stream`; return codes ignored on unsupported SKUs.
        unsafe {
            let _ = set_limit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, win_bytes);
            let _ = set_stream_attr(
                self.inner.stream,
                CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &attr,
            );
        }
    }

    pub fn alloc_from_bytes(&self, data: &[u8]) -> Result<CudaBuffer, CudaError> {
        if data.is_empty() {
            return self.alloc_zeroed(16);
        }
        let buf = self.alloc_uninit(data.len())?;
        buf.copy_from_bytes(data)?;
        Ok(buf)
    }

    pub fn alloc_from_f32(&self, data: &[f32]) -> Result<CudaBuffer, CudaError> {
        // SAFETY: `[f32]` is plain old data with contiguous byte layout.
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data))
        };
        self.alloc_from_bytes(bytes)
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.bind_current()?;
        // SAFETY: Valid `stream` owned by `self.inner`.
        unsafe {
            self.inner.drv.check(
                "cuStreamSynchronize",
                (self.inner.drv.cu_stream_synchronize)(self.inner.stream),
            )
        }
    }

    #[allow(dead_code)]
    pub(crate) fn copy_dtoh_raw(
        &self,
        src_ptr: CUdeviceptr,
        dst: &mut [u8],
    ) -> Result<(), CudaError> {
        if dst.is_empty() {
            return Ok(());
        }
        self.bind_current()?;
        // SAFETY: Caller guarantees `src_ptr` points to at least `dst.len()` readable device bytes.
        unsafe {
            self.inner.drv.check(
                "cuMemcpyDtoH_v2",
                (self.inner.drv.cu_memcpy_dtoh)(
                    dst.as_mut_ptr().cast::<c_void>(),
                    src_ptr,
                    dst.len(),
                ),
            )
        }
    }

    #[inline]
    pub fn stream(&self) -> CUstream {
        self.inner.stream
    }

    /// Validates block size & shared memory limits and launches `func` on the context's stream.
    ///
    /// # Safety
    /// `func` must belong to `self`'s loaded PTX module and `params` must point to live argument
    /// values matching `func`'s C parameter types and count.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        kernel_name: &'static str,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        let threads = block.0.saturating_mul(block.1).saturating_mul(block.2);
        if threads == 0 || threads > self.inner.info.max_threads_per_block {
            return Err(CudaError::BlockTooLarge {
                kernel: kernel_name,
                requested: threads,
                limit: self.inner.info.max_threads_per_block,
            });
        }
        if shared_mem_bytes > self.inner.info.max_shared_mem_optin {
            return Err(CudaError::SharedMemTooLarge {
                kernel: kernel_name,
                requested: shared_mem_bytes,
                limit: self.inner.info.max_shared_mem_optin,
            });
        }
        self.bind_current()?;
        // SAFETY: Caller guarantees `params` points to valid kernel argument storage matching `func`'s PTX signature.
        unsafe {
            self.inner.drv.check(
                kernel_name,
                (self.inner.drv.cu_launch_kernel)(
                    func,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_mem_bytes,
                    self.inner.stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
            )
        }
    }
}

/// RAII device (or zero-copy device-mapped host DDR) memory buffer (`CUdeviceptr`).
pub struct CudaBuffer {
    ctx: CudaContext,
    ptr: CUdeviceptr,
    host_mapped: *mut c_void,
    bytes: usize,
}

// SAFETY: `CUdeviceptr` and pinned `host_mapped` backing belong to `CudaContext` (thread-safe via `bind_current`).
unsafe impl Send for CudaBuffer {}
// SAFETY: Shared references only read metadata or enqueue stream-serialized operations.
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    #[inline]
    pub fn is_host_mapped(&self) -> bool {
        !self.host_mapped.is_null()
    }
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    #[inline]
    pub fn device_ptr_at(&self, byte_offset: usize) -> CUdeviceptr {
        debug_assert!(byte_offset <= self.bytes);
        self.ptr + byte_offset as u64
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    pub fn zero(&self) -> Result<(), CudaError> {
        self.ctx.bind_current()?;
        // SAFETY: `self.ptr` owns `self.bytes` bytes in device memory.
        unsafe {
            self.ctx.inner.drv.check(
                "cuMemsetD8_v2",
                (self.ctx.inner.drv.cu_memset_d8)(self.ptr, 0, self.bytes),
            )
        }
    }

    pub fn copy_from_bytes(&self, src: &[u8]) -> Result<(), CudaError> {
        if src.is_empty() {
            return Ok(());
        }
        if src.len() > self.bytes {
            return Err(CudaError::InvalidDimensions(format!(
                "H2D copy {} bytes exceeds buffer size {}",
                src.len(),
                self.bytes
            )));
        }
        self.ctx.bind_current()?;
        // SAFETY: `src` and `self.ptr` are valid for `src.len()` bytes; `cuMemcpyHtoDAsync_v2`
        // is stream-ordered with `self.ctx.stream()` even for pageable host memory in CUDA.
        unsafe {
            self.ctx.inner.drv.check(
                "cuMemcpyHtoDAsync_v2",
                (self.ctx.inner.drv.cu_memcpy_htod_async)(
                    self.ptr,
                    src.as_ptr().cast::<c_void>(),
                    src.len(),
                    self.ctx.inner.stream,
                ),
            )
        }
    }

    /// Synchronously copies `src` to `self.ptr + dst_offset` (ideal for chunked streaming from disk).
    pub fn copy_from_bytes_at(&self, dst_offset: usize, src: &[u8]) -> Result<(), CudaError> {
        if src.is_empty() {
            return Ok(());
        }
        if dst_offset.saturating_add(src.len()) > self.bytes {
            return Err(CudaError::InvalidDimensions(format!(
                "H2D copy {} + {} bytes exceeds buffer size {}",
                dst_offset,
                src.len(),
                self.bytes
            )));
        }
        self.ctx.bind_current()?;
        // SAFETY: `src` and `self.ptr + dst_offset` are valid for `src.len()` bytes; synchronous `cuMemcpyHtoD_v2`
        // completes before returning so the caller can immediately reuse its host staging buffer.
        unsafe {
            self.ctx.inner.drv.check(
                "cuMemcpyHtoD_v2",
                (self.ctx.inner.drv.cu_memcpy_htod)(
                    self.ptr + dst_offset as u64,
                    src.as_ptr().cast::<c_void>(),
                    src.len(),
                ),
            )
        }
    }

    pub fn copy_from_f32(&self, src: &[f32]) -> Result<(), CudaError> {
        // SAFETY: `[f32]` is POD.
        let bytes = unsafe {
            core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), core::mem::size_of_val(src))
        };
        self.copy_from_bytes(bytes)
    }

    pub fn copy_from_u32(&self, src: &[u32]) -> Result<(), CudaError> {
        // SAFETY: `[u32]` is POD.
        let bytes = unsafe {
            core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), core::mem::size_of_val(src))
        };
        self.copy_from_bytes(bytes)
    }

    pub fn copy_to_bytes(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        if dst.is_empty() {
            return Ok(());
        }
        if dst.len() > self.bytes {
            return Err(CudaError::InvalidDimensions(format!(
                "D2H copy {} bytes exceeds buffer size {}",
                dst.len(),
                self.bytes
            )));
        }
        self.ctx.bind_current()?;
        // SAFETY: Synchronizes `stream` via stream-ordered `cuMemcpyDtoHAsync_v2` + `cuStreamSynchronize`.
        unsafe {
            self.ctx.inner.drv.check(
                "cuMemcpyDtoHAsync_v2",
                (self.ctx.inner.drv.cu_memcpy_dtoh_async)(
                    dst.as_mut_ptr().cast::<c_void>(),
                    self.ptr,
                    dst.len(),
                    self.ctx.inner.stream,
                ),
            )?;
            self.ctx.inner.drv.check(
                "cuStreamSynchronize",
                (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream),
            )
        }
    }

    pub fn copy_to_f32(&self, dst: &mut [f32]) -> Result<(), CudaError> {
        // SAFETY: `[f32]` is POD.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                dst.as_mut_ptr().cast::<u8>(),
                core::mem::size_of_val(dst),
            )
        };
        self.copy_to_bytes(bytes)
    }

    pub fn read_u32(&self) -> Result<u32, CudaError> {
        let mut val = [0u8; 4];
        self.copy_to_bytes(&mut val)?;
        Ok(u32::from_le_bytes(val))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bytes
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    #[inline]
    pub fn copy_from_device(&self, src: &CudaBuffer, bytes: usize) -> Result<(), CudaError> {
        self.copy_from_device_at(0, src, 0, bytes)
    }

    pub fn copy_from_device_at(
        &self,
        dst_off: usize,
        src: &CudaBuffer,
        src_off: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        if bytes == 0 {
            return Ok(());
        }
        if dst_off + bytes > self.bytes || src_off + bytes > src.bytes {
            return Err(CudaError::InvalidDimensions(
                "D2D copy out of bounds".into(),
            ));
        }
        self.ctx.bind_current()?;
        // SAFETY: Stream-ordered D2D copy when `cuMemcpyDtoDAsync_v2` is available, else synchronous fallback.
        unsafe {
            if let Some(dtod_async) = self.ctx.inner.drv.cu_memcpy_dtod_async {
                return self.ctx.inner.drv.check(
                    "cuMemcpyDtoDAsync_v2",
                    dtod_async(
                        self.ptr + dst_off as u64,
                        src.ptr + src_off as u64,
                        bytes,
                        self.ctx.inner.stream,
                    ),
                );
            }
            self.ctx.inner.drv.check(
                "cuStreamSynchronize",
                (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream),
            )?;
            self.ctx.inner.drv.check(
                "cuMemcpyDtoD_v2",
                (self.ctx.inner.drv.cu_memcpy_dtod)(
                    self.ptr + dst_off as u64,
                    src.ptr + src_off as u64,
                    bytes,
                ),
            )
        }
    }

    /// True asynchronous H2D DMA copy from page-locked `CudaPinnedBuffer` (`P1-2`, zero host stall!).
    pub fn copy_from_pinned_async(
        &self,
        dst_offset: usize,
        pinned: &CudaPinnedBuffer,
        src_offset: usize,
        byte_len: usize,
    ) -> Result<(), CudaError> {
        if byte_len == 0 {
            return Ok(());
        }
        if src_offset.saturating_add(byte_len) > pinned.bytes
            || dst_offset.saturating_add(byte_len) > self.bytes
        {
            return Err(CudaError::InvalidDimensions(
                "pinned H2D async copy out of bounds".into(),
            ));
        }
        self.ctx.bind_current()?;
        // SAFETY: `pinned.ptr + src_offset` is page-locked (`cuMemHostAlloc`) and valid for `byte_len` bytes.
        unsafe {
            self.ctx.inner.drv.check(
                "cuMemcpyHtoDAsync_v2",
                (self.ctx.inner.drv.cu_memcpy_htod_async)(
                    self.ptr + dst_offset as u64,
                    pinned.ptr.add(src_offset).cast::<c_void>(),
                    byte_len,
                    self.ctx.inner.stream,
                ),
            )
        }
    }

    /// True asynchronous D2H DMA copy into page-locked `CudaPinnedBuffer` (`P1-2`).
    pub fn copy_to_pinned_async(
        &self,
        src_offset: usize,
        pinned: &mut CudaPinnedBuffer,
        dst_offset: usize,
        byte_len: usize,
    ) -> Result<(), CudaError> {
        if byte_len == 0 {
            return Ok(());
        }
        if src_offset.saturating_add(byte_len) > self.bytes
            || dst_offset.saturating_add(byte_len) > pinned.bytes
        {
            return Err(CudaError::InvalidDimensions(
                "pinned D2H async copy out of bounds".into(),
            ));
        }
        self.ctx.bind_current()?;
        // SAFETY: `pinned.ptr + dst_offset` is page-locked (`cuMemHostAlloc`) and valid for `byte_len` bytes.
        unsafe {
            self.ctx.inner.drv.check(
                "cuMemcpyDtoHAsync_v2",
                (self.ctx.inner.drv.cu_memcpy_dtoh_async)(
                    pinned.ptr.add(dst_offset).cast::<c_void>(),
                    self.ptr + src_offset as u64,
                    byte_len,
                    self.ctx.inner.stream,
                ),
            )
        }
    }

    /// Stream-ordered zero-fill of `[0 .. byte_len)` inside `self`.
    pub fn zero_prefix(&self, byte_len: usize) -> Result<(), CudaError> {
        if byte_len == 0 {
            return Ok(());
        }
        if byte_len > self.bytes {
            return Err(CudaError::InvalidDimensions(
                "zero_prefix exceeds buffer size".into(),
            ));
        }
        self.ctx.bind_current()?;
        // SAFETY: Valid `self.ptr` and `byte_len <= self.bytes`; stream-ordered on `self.ctx.inner.stream`.
        unsafe {
            if let Some(ms_async) = self.ctx.inner.drv.cu_memset_d8_async {
                self.ctx.inner.drv.check(
                    "cuMemsetD8Async",
                    ms_async(self.ptr, 0, byte_len, self.ctx.inner.stream),
                )
            } else {
                self.ctx.inner.drv.check(
                    "cuStreamSynchronize",
                    (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream),
                )?;
                self.ctx.inner.drv.check(
                    "cuMemsetD8_v2",
                    (self.ctx.inner.drv.cu_memset_d8)(self.ptr, 0, byte_len),
                )
            }
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.host_mapped.is_null() {
            // SAFETY: Releases device-mapped pinned host memory (`cuMemFreeHost`).
            unsafe {
                let _ = (self.ctx.inner.drv.cu_ctx_set_current)(self.ctx.inner.ctx);
                let _ = (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream);
                if let Some(free_host) = self.ctx.inner.drv.cu_mem_free_host {
                    let _ = free_host(self.host_mapped);
                }
            }
        } else if self.ptr != 0 {
            // SAFETY: Bind context and release `self.ptr` allocated from `self.ctx`.
            unsafe {
                let _ = (self.ctx.inner.drv.cu_ctx_set_current)(self.ctx.inner.ctx);
                let _ = (self.ctx.inner.drv.cu_mem_free)(self.ptr);
            }
        }
    }
}

/// Page-locked (`cuMemHostAlloc`) host buffer for zero-stall async PCIe DMA (`P1-2`).
pub struct CudaPinnedBuffer {
    ctx: CudaContext,
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: Page-locked host memory is process-wide and safe to move across threads.
unsafe impl Send for CudaPinnedBuffer {}
// SAFETY: Shared access only reads `[u8]` or passes pointer to stream-ordered H2D DMA.
unsafe impl Sync for CudaPinnedBuffer {}

impl CudaPinnedBuffer {
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bytes
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` is valid, initialized, and owns `self.bytes` bytes.
        unsafe { core::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: Exclusive `&mut self` guarantees unique access to `self.bytes` bytes.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }

    #[inline]
    pub fn as_f32_slice(&self, len_f32: usize) -> &[f32] {
        assert!(len_f32 * 4 <= self.bytes && (self.ptr as usize).is_multiple_of(4));
        // SAFETY: `cuMemHostAlloc` returns >= 256B-aligned page-locked memory; `len_f32 * 4 <= self.bytes`.
        unsafe { core::slice::from_raw_parts(self.ptr.cast::<f32>(), len_f32) }
    }

    #[inline]
    pub fn as_u32_slice(&self, len_u32: usize) -> &[u32] {
        assert!(len_u32 * 4 <= self.bytes && (self.ptr as usize).is_multiple_of(4));
        // SAFETY: `cuMemHostAlloc` returns >= 256B-aligned page-locked memory; `len_u32 * 4 <= self.bytes`.
        unsafe { core::slice::from_raw_parts(self.ptr.cast::<u32>(), len_u32) }
    }

    #[inline]
    pub fn as_f32_slice_mut(&mut self, len_f32: usize) -> &mut [f32] {
        assert!(len_f32 * 4 <= self.bytes && (self.ptr as usize).is_multiple_of(4));
        // SAFETY: `cuMemHostAlloc` returns >= 256B-aligned page-locked memory; `len_f32 * 4 <= self.bytes`.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.cast::<f32>(), len_f32) }
    }

    #[inline]
    pub fn write_u32_at(&mut self, byte_offset: usize, src: &[u32]) {
        let byte_len = core::mem::size_of_val(src);
        assert!(byte_offset.saturating_add(byte_len) <= self.bytes);
        // SAFETY: `[u32]` is POD and `byte_offset + byte_len <= self.bytes`.
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr().cast::<u8>(),
                self.ptr.add(byte_offset),
                byte_len,
            );
        }
    }

    #[inline]
    pub fn write_u32_slice(&mut self, src: &[u32]) {
        let byte_len = core::mem::size_of_val(src);
        assert!(byte_len <= self.bytes);
        // SAFETY: `[u32]` is POD and `byte_len <= self.bytes`.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), self.ptr, byte_len);
        }
    }

    #[inline]
    pub fn write_f32_slice(&mut self, src: &[f32]) {
        let byte_len = core::mem::size_of_val(src);
        assert!(byte_len <= self.bytes);
        // SAFETY: `[f32]` is POD and `byte_len <= self.bytes`.
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), self.ptr, byte_len);
        }
    }
}

impl Drop for CudaPinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: Synchronize stream so any in-flight async DMA from/to `self.ptr` completes before `cuMemFreeHost`.
            unsafe {
                let _ = (self.ctx.inner.drv.cu_ctx_set_current)(self.ctx.inner.ctx);
                let _ = (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream);
                if let Some(free_fn) = self.ctx.inner.drv.cu_mem_free_host {
                    let _ = free_fn(self.ptr.cast::<c_void>());
                }
            }
        }
    }
}

/// RAII guard for external host memory registered with `cuMemHostRegister_v2(CU_MEMHOSTREGISTER_DEVICEMAP)`.
pub struct CudaHostRegistered {
    ctx: CudaContext,
    host_ptr: *mut c_void,
    dev_ptr: CUdeviceptr,
    bytes: usize,
}

// SAFETY: Device-mapped host registration is valid across threads within the CUDA primary context.
unsafe impl Send for CudaHostRegistered {}
// SAFETY: Shared references only read `dev_ptr` / `bytes`.
unsafe impl Sync for CudaHostRegistered {}

impl CudaHostRegistered {
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.dev_ptr
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaHostRegistered {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            // SAFETY: Synchronize stream before unregistering mapped host pages.
            unsafe {
                let _ = (self.ctx.inner.drv.cu_ctx_set_current)(self.ctx.inner.ctx);
                let _ = (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream);
                if let Some(unreg) = self.ctx.inner.drv.cu_mem_host_unregister {
                    let _ = unreg(self.host_ptr);
                }
            }
        }
    }
}

/// RAII CUDA event for sub-microsecond hardware kernel timing.
pub struct CudaEvent {
    ctx: CudaContext,
    ev: CUevent,
}

impl CudaEvent {
    pub fn new(ctx: &CudaContext) -> Result<Self, CudaError> {
        ctx.bind_current()?;
        let mut ev: CUevent = ptr::null_mut();
        // SAFETY: `flags = 0` enables hardware timing (`CU_EVENT_DEFAULT`).
        unsafe {
            ctx.inner
                .drv
                .check("cuEventCreate", (ctx.inner.drv.cu_event_create)(&mut ev, 0))?;
        }
        Ok(Self {
            ctx: ctx.clone(),
            ev,
        })
    }

    pub fn record(&self) -> Result<(), CudaError> {
        // SAFETY: Valid event and stream on `self.ctx`.
        unsafe {
            self.ctx.inner.drv.check(
                "cuEventRecord",
                (self.ctx.inner.drv.cu_event_record)(self.ev, self.ctx.inner.stream),
            )
        }
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: Valid event handle.
        unsafe {
            self.ctx.inner.drv.check(
                "cuEventSynchronize",
                (self.ctx.inner.drv.cu_event_synchronize)(self.ev),
            )
        }
    }

    pub fn elapsed_ms(&self, start: &CudaEvent) -> Result<f32, CudaError> {
        let mut ms = 0.0f32;
        // SAFETY: Both `start` and `self` were recorded and completed on `self.ctx`.
        unsafe {
            self.ctx.inner.drv.check(
                "cuEventElapsedTime",
                (self.ctx.inner.drv.cu_event_elapsed_time)(&mut ms, start.ev, self.ev),
            )?;
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.ev.is_null() {
            // SAFETY: Destroys valid event handle.
            unsafe {
                let _ = (self.ctx.inner.drv.cu_ctx_set_current)(self.ctx.inner.ctx);
                let _ = (self.ctx.inner.drv.cu_event_destroy)(self.ev);
            }
        }
    }
}

/// RAII instantiated CUDA Graph (`CUgraphExec`) for zero-host-overhead single-token decode replay.
pub struct CudaGraphExec {
    ctx: CudaContext,
    exec: CUgraphExec,
}

// SAFETY: `CUgraphExec` replay is serialized on `ctx`'s stream after `bind_current()`.
unsafe impl Send for CudaGraphExec {}
// SAFETY: Shared access only invokes `launch` on `ctx`'s stream.
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    /// Captures all stream-ordered kernel launches emitted by `record_fn` into an executable CUDA Graph.
    pub fn capture<F>(ctx: &CudaContext, record_fn: F) -> Result<Self, CudaError>
    where
        F: FnOnce() -> Result<(), CudaError>,
    {
        ctx.bind_current()?;
        ctx.synchronize()?;
        // `CU_STREAM_CAPTURE_MODE_RELAXED = 2`
        // SAFETY: Context is bound and stream is idle before beginning capture.
        unsafe {
            ctx.inner.drv.check(
                "cuStreamBeginCapture_v2",
                (ctx.inner.drv.cu_stream_begin_capture)(ctx.inner.stream, 2),
            )?;
        }
        let rec_res = record_fn();
        let mut graph: CUgraph = ptr::null_mut();
        // SAFETY: Always end stream capture even if `record_fn` returned an error.
        let end_res = unsafe {
            ctx.inner.drv.check(
                "cuStreamEndCapture",
                (ctx.inner.drv.cu_stream_end_capture)(ctx.inner.stream, &mut graph),
            )
        };
        if let Err(e) = rec_res {
            if !graph.is_null() {
                // SAFETY: Clean up partially captured graph.
                unsafe {
                    let _ = (ctx.inner.drv.cu_graph_destroy)(graph);
                }
            }
            return Err(e);
        }
        end_res?;

        let mut exec: CUgraphExec = ptr::null_mut();
        // SAFETY: `graph` is a valid captured `CUgraph`; `flags = 0`.
        let inst_res = unsafe {
            let r = (ctx.inner.drv.cu_graph_instantiate)(&mut exec, graph, 0);
            let _ = (ctx.inner.drv.cu_graph_destroy)(graph);
            ctx.inner.drv.check("cuGraphInstantiateWithFlags", r)
        };
        inst_res?;
        Ok(Self {
            ctx: ctx.clone(),
            exec,
        })
    }

    /// Replays the entire captured multi-layer kernel graph in a single ~3-5 µs driver call.
    #[inline]
    pub fn launch(&self) -> Result<(), CudaError> {
        self.ctx.bind_current()?;
        // SAFETY: `self.exec` and `self.ctx.inner.stream` are valid handles in the same context.
        unsafe {
            self.ctx.inner.drv.check(
                "cuGraphLaunch",
                (self.ctx.inner.drv.cu_graph_launch)(self.exec, self.ctx.inner.stream),
            )
        }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            // SAFETY: Synchronize stream and destroy `CUgraphExec` before context teardown.
            unsafe {
                let _ = (self.ctx.inner.drv.cu_ctx_set_current)(self.ctx.inner.ctx);
                let _ = (self.ctx.inner.drv.cu_stream_synchronize)(self.ctx.inner.stream);
                let _ = (self.ctx.inner.drv.cu_graph_exec_destroy)(self.exec);
            }
        }
    }
}
