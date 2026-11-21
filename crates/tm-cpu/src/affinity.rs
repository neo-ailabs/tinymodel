//! Heterogeneous CPU topology detection (big.LITTLE / DynamIQ + SMT) and
//! automatic performance-core thread affinity for Linux/Android and macOS.
//!
//! Motivation:
//! - On heterogeneous mobile/SBC SoCs (Google Pixel 8 Tensor G3 `1×X3 + 4×A715 + 4×A510`,
//!   Rockchip RK3588 `4×A76 + 4×A55`), leaving worker threads unpinned allows the OS
//!   scheduler to place barrier-synchronized workers onto in-order LITTLE cores (`A510`/`A55`),
//!   stalling every layer barrier (`3.3×` decode slowdown measured on Pixel 8).
//! - On homogeneous LITTLE boards (Raspberry Pi Zero 2W `4×A53`), all cores are kept.
//! - If the user already restricted affinity externally (e.g. `taskset f0`), we strictly
//!   respect that mask and size `recommended_threads()` to the allowed set.

use std::sync::OnceLock;

/// Parsed information for a single logical CPU on Linux/Android.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalCpuInfo {
    pub id: usize,
    pub cpu_part: Option<u32>,
    pub max_freq_khz: u64,
    /// Lowest CPU ID in the same SMT physical core (`thread_siblings_list`), or `id` if non-SMT.
    pub primary_sibling: usize,
}

/// Summary of the host/device CPU topology relevant to thread pool sizing and pinning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTopology {
    /// CPUs permitted when the process started (`sched_getaffinity`), sorted.
    pub initial_allowed_cpus: Vec<usize>,
    /// Whether the caller externally pinned the process to a subset of online CPUs.
    pub externally_pinned: bool,
    /// Selected performance CPUs (non-LITTLE primary SMT threads), sorted.
    pub perf_cpus: Vec<usize>,
    /// Whether heterogeneous LITTLE cores were excluded to form `perf_cpus`.
    pub excluded_little_cores: bool,
}

/// Returns true if Arm `CPU part` ID corresponds to an efficiency/LITTLE core.
#[inline]
pub fn is_arm_little_part(part: u32) -> bool {
    matches!(
        part,
        0xd03 // Cortex-A53
        | 0xd04 // Cortex-A35
        | 0xd05 // Cortex-A55
        | 0xd46 // Cortex-A510
        | 0xd80 // Cortex-A520
    )
}

/// Pure selection logic (unit-tested across Pixel 8, RK3588, Pi Zero 2W, taskset, and x86 SMT).
pub fn select_perf_topology(
    online_cpus: &[LogicalCpuInfo],
    initial_allowed: &[usize],
) -> CpuTopology {
    if online_cpus.is_empty() {
        let fallback: Vec<usize> = if initial_allowed.is_empty() {
            vec![0]
        } else {
            initial_allowed.to_vec()
        };
        return CpuTopology {
            initial_allowed_cpus: fallback.clone(),
            externally_pinned: false,
            perf_cpus: fallback,
            excluded_little_cores: false,
        };
    }

    let mut allowed_sorted: Vec<usize> = if initial_allowed.is_empty() {
        online_cpus.iter().map(|c| c.id).collect()
    } else {
        initial_allowed.to_vec()
    };
    allowed_sorted.sort_unstable();
    allowed_sorted.dedup();

    let all_ids: Vec<usize> = {
        let mut v: Vec<usize> = online_cpus.iter().map(|c| c.id).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let externally_pinned = allowed_sorted.len() < all_ids.len();

    // Filter `online_cpus` down to CPUs allowed at process startup, keeping only 1 thread per SMT core.
    let allowed_infos: Vec<&LogicalCpuInfo> = online_cpus
        .iter()
        .filter(|c| allowed_sorted.binary_search(&c.id).is_ok())
        .filter(|c| {
            c.primary_sibling == c.id || allowed_sorted.binary_search(&c.primary_sibling).is_err()
        })
        .collect();

    if allowed_infos.is_empty() {
        return CpuTopology {
            initial_allowed_cpus: allowed_sorted.clone(),
            externally_pinned,
            perf_cpus: allowed_sorted,
            excluded_little_cores: false,
        };
    }

    if externally_pinned {
        // Respect explicit caller `taskset` mask (only deduplicating SMT siblings if any).
        let cpus: Vec<usize> = allowed_infos.iter().map(|c| c.id).collect();
        return CpuTopology {
            initial_allowed_cpus: allowed_sorted,
            externally_pinned: true,
            perf_cpus: cpus,
            excluded_little_cores: false,
        };
    }

    let max_soc_freq = allowed_infos
        .iter()
        .map(|c| c.max_freq_khz)
        .max()
        .unwrap_or(0);

    let is_little = |c: &LogicalCpuInfo| -> bool {
        if let Some(part) = c.cpu_part {
            return is_arm_little_part(part);
        }
        // Fallback when `/proc/cpuinfo` hides `CPU part`: treat cores < 75% of max SoC clock as LITTLE.
        max_soc_freq > 0 && c.max_freq_khz > 0 && c.max_freq_khz * 100 < max_soc_freq * 75
    };

    let non_little: Vec<usize> = allowed_infos
        .iter()
        .filter(|c| !is_little(c))
        .map(|c| c.id)
        .collect();

    // Only exclude LITTLE cores when faster non-LITTLE cores exist on the same SoC!
    // Homogeneous boards like Pi Zero 2W (4×A53) keep all 4 cores.
    if !non_little.is_empty() && non_little.len() < allowed_infos.len() {
        CpuTopology {
            initial_allowed_cpus: allowed_sorted,
            externally_pinned: false,
            perf_cpus: non_little,
            excluded_little_cores: true,
        }
    } else {
        let all_primary: Vec<usize> = allowed_infos.iter().map(|c| c.id).collect();
        CpuTopology {
            initial_allowed_cpus: allowed_sorted,
            externally_pinned: false,
            perf_cpus: all_primary,
            excluded_little_cores: false,
        }
    }
}

/// Returns the cached host/device CPU topology.
pub fn detected_topology() -> &'static CpuTopology {
    static TOPO: OnceLock<CpuTopology> = OnceLock::new();
    TOPO.get_or_init(probe_os_topology)
}

/// Returns the recommended default thread count for `tinymodel` on this machine:
/// - Google Pixel 8 (`1×X3 + 4×A715 + 4×A510`): `5` (or `4` under `taskset f0`)
/// - Rockchip RK3588 (`4×A76 + 4×A55`): `4`
/// - Apple M4 (`4P + 6E`): `4` (`hw.perflevel0.physicalcpu`)
/// - Raspberry Pi 5 / Pi 4B / Zero 2W: `4`
pub fn recommended_threads() -> usize {
    #[cfg(target_os = "macos")]
    {
        if let Some(p_cores) = macos_perf_physical_cpus()
            && p_cores >= 1
        {
            return p_cores;
        }
    }
    let topo = detected_topology();
    topo.perf_cpus.len().max(1)
}

/// Applies performance-core CPU affinity to the calling thread when appropriate:
/// - No-op on non-Linux/Android platforms or if the user already pinned via `taskset`.
/// - When the SoC has heterogeneous LITTLE + Big/Mid cores and `n_threads <= perf_cpus.len()`,
///   pins the calling thread to `perf_cpus` so the OS scheduler never migrates barrier workers
///   onto LITTLE cores.
/// - If `n_threads > perf_cpus.len()`, restores `initial_allowed_cpus` so extra threads do not
///   oversubscribe `perf_cpus`.
pub fn apply_pool_thread_affinity(n_threads: usize) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let topo = detected_topology();
        if topo.externally_pinned || n_threads <= 1 {
            return;
        }
        if topo.excluded_little_cores && n_threads <= topo.perf_cpus.len() {
            let _ = linux_sched_setaffinity(&topo.perf_cpus);
        } else if topo.excluded_little_cores && n_threads > topo.perf_cpus.len() {
            let _ = linux_sched_setaffinity(&topo.initial_allowed_cpus);
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = n_threads;
    }
}

#[cfg(target_os = "macos")]
fn macos_perf_physical_cpus() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const u8,
            oldp: *mut core::ffi::c_void,
            oldlenp: *mut usize,
            newp: *const core::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }
    for key in [
        b"hw.perflevel0.physicalcpu\0".as_slice(),
        b"hw.physicalcpu\0".as_slice(),
    ] {
        let mut val: i32 = 0;
        let mut size = core::mem::size_of::<i32>();
        // SAFETY: `key` is NUL-terminated; `val` and `size` point to valid stack storage
        // of exact `i32` width; `newp` is null (read-only sysctl query).
        let rc = unsafe {
            sysctlbyname(
                key.as_ptr(),
                (&mut val as *mut i32).cast(),
                &mut size,
                core::ptr::null(),
                0,
            )
        };
        if rc == 0 && val > 0 {
            return Some(val as usize);
        }
    }
    None
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct LinuxCpuSet {
    bits: [u64; 16], // 1024 bits = 128 bytes (`sizeof(cpu_set_t)` on glibc/musl/bionic)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_sched_getaffinity() -> Option<Vec<usize>> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut LinuxCpuSet) -> i32;
    }
    let mut set = LinuxCpuSet { bits: [0; 16] };
    // SAFETY: `pid = 0` queries the calling thread; `set` is 128-byte aligned valid storage.
    let rc = unsafe { sched_getaffinity(0, core::mem::size_of::<LinuxCpuSet>(), &mut set) };
    if rc != 0 {
        return None;
    }
    let mut cpus = Vec::new();
    for (word_idx, &word) in set.bits.iter().enumerate() {
        if word == 0 {
            continue;
        }
        for bit in 0..64usize {
            if (word & (1u64 << bit)) != 0 {
                cpus.push(word_idx * 64 + bit);
            }
        }
    }
    if cpus.is_empty() { None } else { Some(cpus) }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_sched_setaffinity(cpus: &[usize]) -> bool {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const LinuxCpuSet) -> i32;
    }
    let mut set = LinuxCpuSet { bits: [0; 16] };
    for &cpu in cpus {
        if cpu < 1024 {
            set.bits[cpu / 64] |= 1u64 << (cpu % 64);
        }
    }
    // SAFETY: `pid = 0` updates the calling thread's CPU affinity mask using valid 128-byte `cpu_set_t`.
    unsafe { sched_setaffinity(0, core::mem::size_of::<LinuxCpuSet>(), &set) == 0 }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn parse_first_cpu_in_list(s: &str) -> Option<usize> {
    let first_tok = s.trim().split([',', '-']).next()?.trim();
    first_tok.parse::<usize>().ok()
}

/// Parses `/proc/cpuinfo` text + optional sysfs callbacks into `LogicalCpuInfo` entries.
pub fn parse_linux_cpuinfo_with<F, G>(
    cpuinfo: &str,
    mut read_max_freq_khz: F,
    mut read_primary_sibling: G,
) -> Vec<LogicalCpuInfo>
where
    F: FnMut(usize) -> u64,
    G: FnMut(usize) -> Option<usize>,
{
    let mut out: Vec<LogicalCpuInfo> = Vec::new();
    let mut cur_id: Option<usize> = None;
    let mut cur_part: Option<u32> = None;

    let flush = |out: &mut Vec<LogicalCpuInfo>,
                 id_opt: Option<usize>,
                 part_opt: Option<u32>,
                 freq_fn: &mut F,
                 sib_fn: &mut G| {
        if let Some(id) = id_opt
            && !out.iter().any(|c| c.id == id)
        {
            let max_freq_khz = freq_fn(id);
            let primary_sibling = sib_fn(id).unwrap_or(id);
            out.push(LogicalCpuInfo {
                id,
                cpu_part: part_opt,
                max_freq_khz,
                primary_sibling,
            });
        }
    };

    for line in cpuinfo.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            flush(
                &mut out,
                cur_id.take(),
                cur_part.take(),
                &mut read_max_freq_khz,
                &mut read_primary_sibling,
            );
            continue;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim();
            let val = v.trim();
            if key.eq_ignore_ascii_case("processor") {
                if cur_id.is_some() {
                    flush(
                        &mut out,
                        cur_id.take(),
                        cur_part.take(),
                        &mut read_max_freq_khz,
                        &mut read_primary_sibling,
                    );
                }
                cur_id = val.parse::<usize>().ok();
            } else if key.eq_ignore_ascii_case("CPU part") {
                let hex = val
                    .strip_prefix("0x")
                    .or_else(|| val.strip_prefix("0X"))
                    .unwrap_or(val);
                cur_part = u32::from_str_radix(hex, 16).ok();
            }
        }
    }
    flush(
        &mut out,
        cur_id,
        cur_part,
        &mut read_max_freq_khz,
        &mut read_primary_sibling,
    );
    out.sort_by_key(|c| c.id);
    out
}

fn probe_os_topology() -> CpuTopology {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpus = parse_linux_cpuinfo_with(
            &cpuinfo,
            |id| {
                for suffix in ["cpuinfo_max_freq", "scaling_max_freq"] {
                    let p = format!("/sys/devices/system/cpu/cpu{id}/cpufreq/{suffix}");
                    if let Ok(s) = std::fs::read_to_string(p)
                        && let Ok(khz) = s.trim().parse::<u64>()
                        && khz > 0
                    {
                        return khz;
                    }
                }
                0
            },
            |id| {
                let p = format!("/sys/devices/system/cpu/cpu{id}/topology/thread_siblings_list");
                std::fs::read_to_string(p)
                    .ok()
                    .and_then(|s| parse_first_cpu_in_list(&s))
            },
        );
        let initial_allowed =
            linux_sched_getaffinity().unwrap_or_else(|| cpus.iter().map(|c| c.id).collect());
        select_perf_topology(&cpus, &initial_allowed)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let ids: Vec<usize> = (0..n).collect();
        CpuTopology {
            initial_allowed_cpus: ids.clone(),
            externally_pinned: false,
            perf_cpus: ids,
            excluded_little_cores: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixel8_cpus() -> Vec<LogicalCpuInfo> {
        let mut v = Vec::new();
        for id in 0..4 {
            v.push(LogicalCpuInfo {
                id,
                cpu_part: Some(0xd46), // Cortex-A510
                max_freq_khz: 1_704_000,
                primary_sibling: id,
            });
        }
        for id in 4..8 {
            v.push(LogicalCpuInfo {
                id,
                cpu_part: Some(0xd4d), // Cortex-A715
                max_freq_khz: 2_367_000,
                primary_sibling: id,
            });
        }
        v.push(LogicalCpuInfo {
            id: 8,
            cpu_part: Some(0xd4e), // Cortex-X3
            max_freq_khz: 2_914_000,
            primary_sibling: 8,
        });
        v
    }

    #[test]
    fn pixel8_excludes_a510_little_and_recommends_5_threads() {
        let cpus = make_pixel8_cpus();
        let all: Vec<usize> = (0..9).collect();
        let topo = select_perf_topology(&cpus, &all);
        assert!(!topo.externally_pinned);
        assert!(topo.excluded_little_cores);
        assert_eq!(topo.perf_cpus, vec![4, 5, 6, 7, 8]);
    }

    #[test]
    fn pixel8_respects_explicit_taskset_mid_only() {
        let cpus = make_pixel8_cpus();
        let mid_only = vec![4, 5, 6, 7];
        let topo = select_perf_topology(&cpus, &mid_only);
        assert!(topo.externally_pinned);
        assert!(!topo.excluded_little_cores);
        assert_eq!(topo.perf_cpus, vec![4, 5, 6, 7]);
    }

    #[test]
    fn rk3588_excludes_a55_and_selects_4_a76_cores() {
        let mut cpus = Vec::new();
        for id in 0..4 {
            cpus.push(LogicalCpuInfo {
                id,
                cpu_part: Some(0xd05), // Cortex-A55
                max_freq_khz: 1_800_000,
                primary_sibling: id,
            });
        }
        for id in 4..8 {
            cpus.push(LogicalCpuInfo {
                id,
                cpu_part: Some(0xd0b), // Cortex-A76
                max_freq_khz: 2_400_000,
                primary_sibling: id,
            });
        }
        let all: Vec<usize> = (0..8).collect();
        let topo = select_perf_topology(&cpus, &all);
        assert!(topo.excluded_little_cores);
        assert_eq!(topo.perf_cpus, vec![4, 5, 6, 7]);
    }

    #[test]
    fn homogeneous_pi_zero_2w_keeps_all_4_a53_cores() {
        let cpus: Vec<LogicalCpuInfo> = (0..4)
            .map(|id| LogicalCpuInfo {
                id,
                cpu_part: Some(0xd03), // Cortex-A53
                max_freq_khz: 1_000_000,
                primary_sibling: id,
            })
            .collect();
        let all = vec![0, 1, 2, 3];
        let topo = select_perf_topology(&cpus, &all);
        assert!(!topo.excluded_little_cores);
        assert_eq!(topo.perf_cpus, vec![0, 1, 2, 3]);
    }

    #[test]
    fn x86_smt_deduplicates_hyperthreads() {
        let mut cpus = Vec::new();
        for id in 0..8 {
            cpus.push(LogicalCpuInfo {
                id,
                cpu_part: None,
                max_freq_khz: 4_500_000,
                primary_sibling: id % 4,
            });
        }
        let all: Vec<usize> = (0..8).collect();
        let topo = select_perf_topology(&cpus, &all);
        assert!(!topo.excluded_little_cores);
        assert_eq!(topo.perf_cpus, vec![0, 1, 2, 3]);
    }
}
