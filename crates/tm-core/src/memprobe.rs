//! Host and container (cgroup v2) memory probing for edge / Raspberry Pi deployments.
//!
//! Enforces `AGENTS.md` §9:
//! - Reads both `/proc/meminfo` (`MemTotal`, `MemAvailable`) and cgroup v2 limits
//!   (`memory.max`, `memory.current`) for Docker / Home Assistant / k3s on Raspberry Pi;
//! - Never silently defaults planner inputs to `0` when probing fails.

use std::fmt;

/// Validated non-zero memory budget probed from the host OS and active cgroup v2 hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Physical or container total memory ceiling in bytes (strictly > 0).
    pub total_bytes: u64,
    /// Conservative currently available memory in bytes (strictly > 0).
    pub available_bytes: u64,
    /// Optional cgroup v2 hard limit (`memory.max`), if active and finite.
    pub cgroup_max_bytes: Option<u64>,
}

/// Errors returned when host memory cannot be reliably determined (never defaults to 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemProbeError {
    UnsupportedPlatform,
    Io(String),
    MissingOrZeroField(&'static str),
}

impl fmt::Display for MemProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemProbeError::UnsupportedPlatform => {
                write!(f, "memory probe not implemented for this platform")
            }
            MemProbeError::Io(msg) => write!(f, "failed to read system memory info: {msg}"),
            MemProbeError::MissingOrZeroField(field) => {
                write!(
                    f,
                    "system memory field '{field}' is missing or zero (refusing silent 0 default)"
                )
            }
        }
    }
}

impl std::error::Error for MemProbeError {}

/// Parses `/proc/meminfo` text and extracts `(MemTotal, MemAvailable)` in bytes.
/// Falls back to `MemFree + Buffers + Cached` only on very old kernels (< 3.14) lacking `MemAvailable`.
pub fn parse_proc_meminfo(meminfo: &str) -> Result<(u64, u64), MemProbeError> {
    let mut mem_total_kb: Option<u64> = None;
    let mut mem_avail_kb: Option<u64> = None;
    let mut mem_free_kb: u64 = 0;
    let mut buffers_kb: u64 = 0;
    let mut cached_kb: u64 = 0;
    let mut saw_fallback = false;

    for line in meminfo.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(num_tok) = rest.split_whitespace().next() else {
            continue;
        };
        let Ok(val_kb) = num_tok.parse::<u64>() else {
            continue;
        };
        match key.trim() {
            "MemTotal" => mem_total_kb = Some(val_kb),
            "MemAvailable" => mem_avail_kb = Some(val_kb),
            "MemFree" => {
                mem_free_kb = val_kb;
                saw_fallback = true;
            }
            "Buffers" => buffers_kb = val_kb,
            "Cached" => cached_kb = val_kb,
            _ => {}
        }
    }

    let total_kb = mem_total_kb
        .filter(|&v| v > 0)
        .ok_or(MemProbeError::MissingOrZeroField("MemTotal"))?;
    let avail_kb = mem_avail_kb
        .or_else(|| saw_fallback.then_some(mem_free_kb + buffers_kb + cached_kb))
        .filter(|&v| v > 0)
        .ok_or(MemProbeError::MissingOrZeroField("MemAvailable"))?;

    Ok((total_kb.saturating_mul(1024), avail_kb.saturating_mul(1024)))
}

/// Parses cgroup v2 `memory.max` (`"max"` -> `None`, finite positive integer -> `Some(bytes)`).
/// Rejects `"0"` or unparseable contents rather than silently returning `0`.
pub fn parse_cgroup_v2_max(raw: &str) -> Result<Option<u64>, MemProbeError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MemProbeError::MissingOrZeroField("cgroup memory.max"));
    }
    if trimmed == "max" {
        return Ok(None);
    }
    let val = trimmed
        .parse::<u64>()
        .map_err(|_| MemProbeError::MissingOrZeroField("cgroup memory.max"))?;
    if val == 0 {
        return Err(MemProbeError::MissingOrZeroField("cgroup memory.max"));
    }
    // Treat `i64::MAX` / page-aligned near-`u64::MAX` sentinels as unlimited.
    if val >= (1u64 << 60) {
        return Ok(None);
    }
    Ok(Some(val))
}

/// Extracts the unified cgroup v2 relative path (`0::/path`) from `/proc/self/cgroup`.
pub fn parse_self_cgroup_v2_path(proc_self_cgroup: &str) -> Option<&str> {
    for line in proc_self_cgroup.lines() {
        if let Some(rel) = line.strip_prefix("0::") {
            let trimmed = rel.trim();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    None
}

/// Combines `/proc/meminfo` and optional cgroup v2 (`memory.max`, `memory.current`) into a `MemoryBudget`.
pub fn compute_budget(
    meminfo_total: u64,
    meminfo_avail: u64,
    cgroup_max: Option<u64>,
    cgroup_current: Option<u64>,
) -> Result<MemoryBudget, MemProbeError> {
    if meminfo_total == 0 {
        return Err(MemProbeError::MissingOrZeroField("MemTotal"));
    }
    if meminfo_avail == 0 {
        return Err(MemProbeError::MissingOrZeroField("MemAvailable"));
    }

    let (total_bytes, available_bytes, effective_cg) = match cgroup_max {
        Some(cg_max) if cg_max < meminfo_total => {
            let cg_rem = match cgroup_current {
                Some(cur) => cg_max.saturating_sub(cur).max(1),
                None => cg_max,
            };
            (cg_max, meminfo_avail.min(cg_rem), Some(cg_max))
        }
        _ => (meminfo_total, meminfo_avail, None),
    };

    Ok(MemoryBudget {
        total_bytes,
        available_bytes,
        cgroup_max_bytes: effective_cg,
    })
}

/// Probes host + cgroup v2 memory on Linux and Android (Raspberry Pi, Docker, Pixel / Android).
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn probe_system_memory() -> Result<MemoryBudget, MemProbeError> {
    use std::fs;
    use std::path::PathBuf;

    let meminfo =
        fs::read_to_string("/proc/meminfo").map_err(|e| MemProbeError::Io(e.to_string()))?;
    let (mem_total, mem_avail) = parse_proc_meminfo(&meminfo)?;

    let mut cg_dirs: Vec<PathBuf> = Vec::new();
    if let Ok(cg_self) = fs::read_to_string("/proc/self/cgroup")
        && let Some(rel) = parse_self_cgroup_v2_path(&cg_self)
    {
        let trimmed = rel.trim_start_matches('/');
        if !trimmed.is_empty() {
            cg_dirs.push(PathBuf::from("/sys/fs/cgroup").join(trimmed));
        }
    }
    cg_dirs.push(PathBuf::from("/sys/fs/cgroup"));

    let mut min_cg_max: Option<u64> = None;
    let mut matched_cur: Option<u64> = None;

    for dir in cg_dirs {
        if let Ok(raw_max) = fs::read_to_string(dir.join("memory.max"))
            && let Ok(Some(limit)) = parse_cgroup_v2_max(&raw_max)
            && min_cg_max.is_none_or(|prev| limit < prev)
        {
            min_cg_max = Some(limit);
            matched_cur = fs::read_to_string(dir.join("memory.current"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
        }
    }

    compute_budget(mem_total, mem_avail, min_cg_max, matched_cur)
}

/// Parses macOS `vm_stat` output + `hw.memsize` into `(total_bytes, available_bytes)`.
pub fn parse_macos_vm_stat(
    total_bytes: u64,
    vm_stat_text: &str,
) -> Result<MemoryBudget, MemProbeError> {
    if total_bytes == 0 {
        return Err(MemProbeError::MissingOrZeroField("hw.memsize"));
    }
    let mut page_size: u64 = 16384;
    let mut free_pages: u64 = 0;
    let mut active_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut speculative_pages: u64 = 0;
    let mut purgeable_pages: u64 = 0;
    let mut file_backed_pages: u64 = 0;
    let mut wired_pages: u64 = 0;
    let mut compressor_pages: u64 = 0;

    for line in vm_stat_text.lines() {
        if let Some(idx) = line.find("page size of ") {
            let after = &line[idx + "page size of ".len()..];
            if let Some(num_str) = after.split_whitespace().next()
                && let Ok(ps) = num_str.parse::<u64>()
                && ps > 0
            {
                page_size = ps;
            }
            continue;
        }
        let Some((k, rest)) = line.split_once(':') else {
            continue;
        };
        let val_str = rest.trim().trim_end_matches('.');
        let Ok(val) = val_str.parse::<u64>() else {
            continue;
        };
        match k.trim() {
            "Pages free" => free_pages = val,
            "Pages active" => active_pages = val,
            "Pages inactive" => inactive_pages = val,
            "Pages speculative" => speculative_pages = val,
            "Pages purgeable" => purgeable_pages = val,
            "File-backed pages" => file_backed_pages = val,
            "Pages wired down" => wired_pages = val,
            "Pages occupied by compressor" => compressor_pages = val,
            _ => {}
        }
    }

    // Reclaimable without compressing/swapping anonymous pages: free + speculative + max(inactive + purgeable, file_backed).
    let reclaimable = free_pages
        .saturating_add(speculative_pages)
        .saturating_add(
            inactive_pages
                .saturating_add(purgeable_pages)
                .max(file_backed_pages),
        )
        .saturating_mul(page_size);
    // Estimate live uncompressed anonymous pages currently in RAM.
    let total_paged = active_pages
        .saturating_add(inactive_pages)
        .saturating_add(speculative_pages);
    let live_anon_pages = total_paged.saturating_sub(file_backed_pages);
    let os_headroom = (total_bytes / 16).clamp(384 * 1024 * 1024, 1024 * 1024 * 1024);
    let uncompressed_cap = total_bytes
        .saturating_sub(wired_pages.saturating_mul(page_size))
        .saturating_sub(compressor_pages.saturating_mul(page_size))
        .saturating_sub(live_anon_pages.saturating_mul(page_size))
        .saturating_sub(os_headroom);
    let available_bytes = if reclaimable > 0 && live_anon_pages > 0 {
        reclaimable.min(uncompressed_cap.max(free_pages.saturating_mul(page_size)))
    } else if reclaimable > 0 {
        reclaimable.min(uncompressed_cap)
    } else {
        uncompressed_cap
    }
    .min(total_bytes)
    .max(1);

    Ok(MemoryBudget {
        total_bytes,
        available_bytes,
        cgroup_max_bytes: None,
    })
}

#[cfg(target_os = "macos")]
pub fn probe_system_memory() -> Result<MemoryBudget, MemProbeError> {
    use std::process::Command;

    let sysctl_out = Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .map_err(|e| MemProbeError::Io(e.to_string()))?;
    let total_bytes = String::from_utf8_lossy(&sysctl_out.stdout)
        .trim()
        .parse::<u64>()
        .map_err(|_| MemProbeError::MissingOrZeroField("hw.memsize"))?;

    let vm_out = Command::new("/usr/bin/vm_stat")
        .output()
        .map_err(|e| MemProbeError::Io(e.to_string()))?;
    let vm_text = String::from_utf8_lossy(&vm_out.stdout);
    parse_macos_vm_stat(total_bytes, &vm_text)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
pub fn probe_system_memory() -> Result<MemoryBudget, MemProbeError> {
    Err(MemProbeError::UnsupportedPlatform)
}

/// Detailed process memory counters distinguishing clean file-backed `mmap` pages from private dirty RAM.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessMemoryStats {
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub private_dirty_bytes: Option<u64>,
    pub pss_anon_bytes: Option<u64>,
}

/// Parses `(VmRSS, VmHWM)` in bytes from `/proc/self/status` text (current & peak RSS).
pub fn parse_proc_status_rss(status_text: &str) -> (Option<u64>, Option<u64>) {
    let mut rss_kb: Option<u64> = None;
    let mut hwm_kb: Option<u64> = None;
    for line in status_text.lines() {
        let Some((k, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(tok) = rest.split_whitespace().next() else {
            continue;
        };
        let Ok(val) = tok.parse::<u64>() else {
            continue;
        };
        match k.trim() {
            "VmRSS" => rss_kb = Some(val),
            "VmHWM" => hwm_kb = Some(val),
            _ => {}
        }
    }
    (rss_kb.map(|v| v * 1024), hwm_kb.map(|v| v * 1024))
}

/// Parses `(Private_Dirty, Pss_Anon)` in bytes from `/proc/self/smaps_rollup` text.
pub fn parse_smaps_rollup(rollup_text: &str) -> (Option<u64>, Option<u64>) {
    let mut priv_dirty_kb: Option<u64> = None;
    let mut pss_anon_kb: Option<u64> = None;
    for line in rollup_text.lines() {
        let Some((k, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(tok) = rest.split_whitespace().next() else {
            continue;
        };
        let Ok(val) = tok.parse::<u64>() else {
            continue;
        };
        match k.trim() {
            "Private_Dirty" => priv_dirty_kb = Some(val),
            "Pss_Anon" => pss_anon_kb = Some(val),
            _ => {}
        }
    }
    (
        priv_dirty_kb.map(|v| v * 1024),
        pss_anon_kb.map(|v| v * 1024),
    )
}

/// Reads current `(rss_bytes, peak_rss_bytes)` of the running process when `/proc/self/status` is available.
pub fn probe_process_rss_bytes() -> (Option<u64>, Option<u64>) {
    let st = probe_process_memory_stats();
    (st.rss_bytes, st.peak_rss_bytes)
}

/// Reads RSS, peak RSS (`VmHWM`), `Private_Dirty`, and `Pss_Anon` from `/proc/self/{status,smaps_rollup}`.
pub fn probe_process_memory_stats() -> ProcessMemoryStats {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let (rss_bytes, peak_rss_bytes) = std::fs::read_to_string("/proc/self/status")
            .ok()
            .map(|t| parse_proc_status_rss(&t))
            .unwrap_or((None, None));
        let (private_dirty_bytes, pss_anon_bytes) =
            std::fs::read_to_string("/proc/self/smaps_rollup")
                .ok()
                .map(|t| parse_smaps_rollup(&t))
                .unwrap_or((None, None));
        ProcessMemoryStats {
            rss_bytes,
            peak_rss_bytes,
            private_dirty_bytes,
            pss_anon_bytes,
        }
    }
    #[cfg(target_os = "macos")]
    {
        let pid_str = std::process::id().to_string();
        let rss_bytes = std::process::Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", &pid_str])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .map(|kb| kb * 1024);
        ProcessMemoryStats {
            rss_bytes,
            peak_rss_bytes: rss_bytes,
            private_dirty_bytes: None,
            pss_anon_bytes: None,
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        ProcessMemoryStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rpi4b_meminfo_and_docker_cgroup_v2() {
        // Representative Raspberry Pi 4B 2GB `/proc/meminfo`
        let meminfo = "\
MemTotal:        1892352 kB
MemFree:         1120400 kB
MemAvailable:    1684200 kB
Buffers:           45120 kB
Cached:           480112 kB
";
        let (total, avail) = parse_proc_meminfo(meminfo).unwrap();
        assert_eq!(total, 1_892_352 * 1024);
        assert_eq!(avail, 1_684_200 * 1024);

        // Unlimited cgroup v2 (`"max\n"`) -> uses `/proc/meminfo` directly
        assert_eq!(parse_cgroup_v2_max("max\n").unwrap(), None);
        let host_budget = compute_budget(total, avail, None, None).unwrap();
        assert_eq!(host_budget.available_bytes, avail);
        assert_eq!(host_budget.cgroup_max_bytes, None);

        // Docker `--memory=512m` on a 2GB Pi 4B (`memory.max = 536870912`, `memory.current = 104857600`)
        let cg_max = parse_cgroup_v2_max("536870912\n").unwrap();
        assert_eq!(cg_max, Some(536_870_912));
        let cg_budget = compute_budget(total, avail, cg_max, Some(104_857_600)).unwrap();
        assert_eq!(cg_budget.total_bytes, 536_870_912);
        assert_eq!(cg_budget.available_bytes, 432_013_312);
        assert_eq!(cg_budget.cgroup_max_bytes, Some(536_870_912));

        // `/proc/self/cgroup` v2 path extraction
        let proc_cg = "0::/system.slice/docker-1234abcd.scope\n";
        assert_eq!(
            parse_self_cgroup_v2_path(proc_cg),
            Some("/system.slice/docker-1234abcd.scope")
        );
    }

    #[test]
    fn refuses_silent_zero_defaults() {
        assert!(parse_proc_meminfo("").is_err());
        assert!(parse_proc_meminfo("MemTotal: 0 kB\nMemAvailable: 1024 kB\n").is_err());
        assert!(parse_cgroup_v2_max("0\n").is_err());
        assert!(parse_cgroup_v2_max("").is_err());
        assert!(compute_budget(0, 1024, None, None).is_err());
    }

    #[test]
    fn parses_proc_status_vmrss_and_vmhwm() {
        let status = "\
Name:\ttinymodel
VmPeak:\t  612340 kB
VmSize:\t  608120 kB
VmHWM:\t  548000 kB
VmRSS:\t  544128 kB
";
        let (rss, hwm) = parse_proc_status_rss(status);
        assert_eq!(rss, Some(544_128 * 1024));
        assert_eq!(hwm, Some(548_000 * 1024));

        let rollup = "\
62eebf4000-7fffcc0000 ---p 00000000 00:00 0 [rollup]
Rss:              544128 kB
Pss_Anon:          39120 kB
Private_Clean:    504000 kB
Private_Dirty:     40128 kB
";
        let (pdirty, panon) = parse_smaps_rollup(rollup);
        assert_eq!(pdirty, Some(40_128 * 1024));
        assert_eq!(panon, Some(39_120 * 1024));
    }
}
