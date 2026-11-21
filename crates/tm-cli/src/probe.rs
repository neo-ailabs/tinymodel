//! On-device hardware probe, thermal/power telemetry, L1 kernel self-check,
//! and memory-safe multi-threaded STREAM bandwidth benchmark for external device testing.
//!
//! Designed to run identically on Apple M4, Raspberry Pi (Zero 2W / 4B / 5 via SSH),
//! and Android phones (Google Pixel 8 via `adb shell` or JNI) with zero external tools.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use tm_core::{
    BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8_1, BlockQ8K, f32_to_f16,
    probe_process_memory_stats, probe_system_memory,
};
use tm_cpu::isa::{detect_caps, effective_isa, supported_isas};
use tm_cpu::{ThreadPool, generic, select_kernels};

/// Escapes an arbitrary string for RFC-8259 JSON output without external dependencies.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

pub fn opt_u64_json(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".to_string(),
    }
}

pub fn opt_f64_json(v: Option<f64>, decimals: usize) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.decimals$}"),
        _ => "null".to_string(),
    }
}

/// Lightweight thermal + frequency + battery power snapshot taken before/after benchmarks.
#[derive(Debug, Clone, Default)]
pub struct TelemetrySnapshot {
    pub max_temp_c: Option<f64>,
    pub skin_temp_c: Option<f64>,
    pub thermal_status: Option<i64>,
    pub zones: Vec<(String, f64)>,
    pub cur_freq_mhz_max: Option<f64>,
    pub governor: Option<String>,
    pub battery_power_mw: Option<f64>,
    pub battery_status: Option<String>,
}

impl TelemetrySnapshot {
    pub fn to_json(&self) -> String {
        let zones_json: Vec<String> = self
            .zones
            .iter()
            .take(10)
            .map(|(k, v)| format!("{{\"zone\":\"{}\",\"temp_c\":{v:.2}}}", json_escape(k)))
            .collect();
        let gov = match &self.governor {
            Some(g) => format!("\"{}\"", json_escape(g)),
            None => "null".to_string(),
        };
        let bat_stat = match &self.battery_status {
            Some(s) => format!("\"{}\"", json_escape(s)),
            None => "null".to_string(),
        };
        let t_stat = match self.thermal_status {
            Some(s) => s.to_string(),
            None => "null".to_string(),
        };
        format!(
            "{{\"max_temp_c\":{},\"skin_temp_c\":{},\"thermal_status\":{},\"cur_freq_mhz_max\":{},\"governor\":{},\"battery_power_mw\":{},\"battery_status\":{},\"zones\":[{}]}}",
            opt_f64_json(self.max_temp_c, 2),
            opt_f64_json(self.skin_temp_c, 2),
            t_stat,
            opt_f64_json(self.cur_freq_mhz_max, 1),
            gov,
            opt_f64_json(self.battery_power_mw, 1),
            bat_stat,
            zones_json.join(",")
        )
    }
}

/// Parses Android `dumpsys thermalservice` output (`Cached temperatures:` + `Thermal Status:`).
pub fn parse_android_dumpsys_thermal(raw: &str) -> (Option<i64>, Option<f64>, Vec<(String, f64)>) {
    let mut status: Option<i64> = None;
    let mut skin_c: Option<f64> = None;
    let mut zones: Vec<(String, f64)> = Vec::new();
    let mut in_cached = false;

    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Thermal Status:") {
            status = rest.trim().parse::<i64>().ok();
            continue;
        }
        if trimmed.starts_with("Cached temperatures:") {
            in_cached = true;
            continue;
        }
        if in_cached && !trimmed.starts_with("Temperature{") {
            in_cached = false;
        }
        if !in_cached {
            continue;
        }
        // Format: `Temperature{mValue=60.000004, mType=0, mName=BIG, mStatus=0}`
        let Some(inner) = trimmed
            .strip_prefix("Temperature{")
            .and_then(|s| s.strip_suffix('}'))
        else {
            continue;
        };
        let mut val_opt: Option<f64> = None;
        let mut type_opt: Option<i64> = None;
        let mut name_opt: Option<&str> = None;
        for part in inner.split(',') {
            let Some((k, v)) = part.trim().split_once('=') else {
                continue;
            };
            match k.trim() {
                "mValue" => val_opt = v.trim().parse::<f64>().ok(),
                "mType" => type_opt = v.trim().parse::<i64>().ok(),
                "mName" => name_opt = Some(v.trim()),
                _ => {}
            }
        }
        if let (Some(temp_c), Some(name)) = (val_opt, name_opt)
            && (5.0..=130.0).contains(&temp_c)
        {
            if name == "VIRTUAL-SKIN" || type_opt == Some(3) {
                skin_c = Some(temp_c);
            }
            // Keep primary SoC/skin/battery sensors (`mType >= 0` or `soc_therm`)
            if type_opt.unwrap_or(-1) >= 0 || name == "soc_therm" {
                zones.push((name.to_string(), temp_c));
            }
        }
    }

    zones.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    (status, skin_c, zones)
}

/// Collects non-privileged thermal, cpufreq, and battery power telemetry from sysfs when present.
pub fn collect_telemetry() -> TelemetrySnapshot {
    let mut snap = TelemetrySnapshot::default();

    // 1. `/sys/class/thermal/thermal_zone*` (Raspberry Pi, Linux, Android / Pixel)
    if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
        let mut zones = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if !name_str.starts_with("thermal_zone") {
                continue;
            }
            let p = entry.path();
            let Ok(raw_temp) = fs::read_to_string(p.join("temp")) else {
                continue;
            };
            let Ok(milli_c) = raw_temp.trim().parse::<i64>() else {
                continue;
            };
            // Linux/Android thermal zones report millidegrees C (occasionally decikelvin/decidegrees; filter sane 1..130 C)
            let temp_c = if milli_c > 1000 {
                (milli_c as f64) / 1000.0
            } else {
                milli_c as f64
            };
            if (1.0..=130.0).contains(&temp_c) {
                let ztype = fs::read_to_string(p.join("type"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| name_str.to_string());
                zones.push((ztype, temp_c));
            }
        }
        zones.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        snap.max_temp_c = zones.first().map(|z| z.1);
        snap.zones = zones;
    }

    // 1b. Android fallback: `/system/bin/dumpsys thermalservice` when SELinux hides `/sys/class/thermal`
    if snap.zones.is_empty()
        && Path::new("/system/bin/dumpsys").exists()
        && let Ok(out) = std::process::Command::new("/system/bin/dumpsys")
            .arg("thermalservice")
            .output()
        && out.status.success()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        let (st, skin, zones) = parse_android_dumpsys_thermal(&text);
        snap.thermal_status = st;
        snap.skin_temp_c = skin;
        snap.max_temp_c = zones.first().map(|z| z.1);
        snap.zones = zones;
    }

    // 2. `/sys/devices/system/cpu/cpu*/cpufreq`
    let mut max_cur_khz: Option<u64> = None;
    for cpu_idx in 0..32 {
        let base = format!("/sys/devices/system/cpu/cpu{cpu_idx}/cpufreq");
        let bp = Path::new(&base);
        if !bp.exists() {
            continue;
        }
        if snap.governor.is_none()
            && let Ok(g) = fs::read_to_string(bp.join("scaling_governor"))
        {
            snap.governor = Some(g.trim().to_string());
        }
        if let Ok(cf) = fs::read_to_string(bp.join("scaling_cur_freq"))
            && let Ok(khz) = cf.trim().parse::<u64>()
        {
            max_cur_khz = Some(max_cur_khz.map_or(khz, |prev| prev.max(khz)));
        }
    }
    snap.cur_freq_mhz_max = max_cur_khz.map(|k| (k as f64) / 1000.0);

    // 3. Android / Linux battery power supply (`/sys/class/power_supply/battery`)
    for ps_name in &["battery", "BAT0", "bms"] {
        let ps = Path::new("/sys/class/power_supply").join(ps_name);
        if !ps.exists() {
            continue;
        }
        if snap.battery_status.is_none()
            && let Ok(st) = fs::read_to_string(ps.join("status"))
        {
            snap.battery_status = Some(st.trim().to_string());
        }
        if let Ok(pw_raw) = fs::read_to_string(ps.join("power_now"))
            && let Ok(uw) = pw_raw.trim().parse::<i64>()
            && uw != 0
        {
            snap.battery_power_mw = Some((uw.unsigned_abs() as f64) / 1000.0);
            break;
        }
        if let (Ok(cur_raw), Ok(vol_raw)) = (
            fs::read_to_string(ps.join("current_now")),
            fs::read_to_string(ps.join("voltage_now")),
        ) && let (Ok(ua), Ok(uv)) =
            (cur_raw.trim().parse::<i64>(), vol_raw.trim().parse::<i64>())
            && ua != 0
            && uv > 0
        {
            // Android devices typically report µA and µV; if current < 50_000 on a phone under load, some kernels report mA.
            let amps = if ua.unsigned_abs() < 20_000 {
                (ua.unsigned_abs() as f64) / 1e3
            } else {
                (ua.unsigned_abs() as f64) / 1e6
            };
            let volts = (uv as f64) / 1e6;
            snap.battery_power_mw = Some(amps * volts * 1000.0);
            break;
        }
    }

    snap
}

/// Maps ARM `CPU implementer` + `CPU part` hex codes to human-readable microarchitecture names.
pub fn arm_part_name(implementer: &str, part: &str) -> &'static str {
    match (implementer, part) {
        ("0x41", "0xd03") => "Cortex-A53",
        ("0x41", "0xd05") => "Cortex-A55",
        ("0x41", "0xd07") => "Cortex-A57",
        ("0x41", "0xd08") => "Cortex-A72",
        ("0x41", "0xd09") => "Cortex-A73",
        ("0x41", "0xd0a") => "Cortex-A75",
        ("0x41", "0xd0b") => "Cortex-A76",
        ("0x41", "0xd0d") => "Cortex-A77",
        ("0x41", "0xd41") => "Cortex-A78",
        ("0x41", "0xd44") => "Cortex-X1",
        ("0x41", "0xd46") => "Cortex-A510",
        ("0x41", "0xd47") => "Cortex-A710",
        ("0x41", "0xd48") => "Cortex-X2",
        ("0x41", "0xd4d") => "Cortex-A715",
        ("0x41", "0xd4e") => "Cortex-X3",
        ("0x41", "0xd80") => "Cortex-A520",
        ("0x41", "0xd81") => "Cortex-A720",
        ("0x41", "0xd82") => "Cortex-X4",
        ("0x61", _) => "Apple-Silicon",
        _ => "Unknown-Core",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuCluster {
    pub uarch: String,
    pub max_freq_mhz: u64,
    pub cpus: Vec<usize>,
    pub cpu_mask_hex: String,
}

/// Parses Linux/Android `/proc/cpuinfo` + per-CPU `cpuinfo_max_freq` to discover heterogeneous clusters
/// (e.g., Pixel 8's `4x Cortex-A510`, `4x Cortex-A715`, `1x Cortex-X3`) and compute `taskset` bitmasks!
pub fn parse_cpu_clusters(cpuinfo: &str, max_freqs_khz: &BTreeMap<usize, u64>) -> Vec<CpuCluster> {
    let mut cur_proc: Option<usize> = None;
    let mut cur_impl = String::new();
    let mut cur_part = String::new();
    let mut cur_model = String::new();
    let mut per_cpu: BTreeMap<usize, (String, u64)> = BTreeMap::new();

    let flush = |proc_opt: Option<usize>,
                 imp: &str,
                 part: &str,
                 model: &str,
                 out: &mut BTreeMap<usize, (String, u64)>| {
        if let Some(p) = proc_opt {
            let uarch = if !part.is_empty() {
                let mapped = arm_part_name(imp, part);
                if mapped == "Unknown-Core" {
                    format!("part-{part}")
                } else {
                    mapped.to_string()
                }
            } else if !model.is_empty() {
                model.to_string()
            } else {
                "cpu".to_string()
            };
            let mhz = max_freqs_khz.get(&p).copied().unwrap_or(0) / 1000;
            out.insert(p, (uarch, mhz));
        }
    };

    for line in cpuinfo.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            flush(cur_proc, &cur_impl, &cur_part, &cur_model, &mut per_cpu);
            cur_proc = None;
            cur_impl.clear();
            cur_part.clear();
            cur_model.clear();
            continue;
        }
        let Some((k, v)) = trimmed.split_once(':') else {
            continue;
        };
        let key = k.trim();
        let val = v.trim();
        if key == "processor" {
            if cur_proc.is_some() {
                flush(cur_proc, &cur_impl, &cur_part, &cur_model, &mut per_cpu);
                cur_impl.clear();
                cur_part.clear();
                cur_model.clear();
            }
            cur_proc = val.parse::<usize>().ok();
        } else if key.eq_ignore_ascii_case("CPU implementer") {
            cur_impl = val.to_ascii_lowercase();
        } else if key.eq_ignore_ascii_case("CPU part") {
            cur_part = val.to_ascii_lowercase();
        } else if key.eq_ignore_ascii_case("model name") || key.eq_ignore_ascii_case("Hardware") {
            cur_model = val.to_string();
        }
    }
    flush(cur_proc, &cur_impl, &cur_part, &cur_model, &mut per_cpu);

    let mut grouped: BTreeMap<(u64, String), Vec<usize>> = BTreeMap::new();
    for (cpu, (uarch, mhz)) in per_cpu {
        grouped.entry((mhz, uarch)).or_default().push(cpu);
    }

    grouped
        .into_iter()
        .map(|((max_freq_mhz, uarch), cpus)| {
            let mut mask: u128 = 0;
            for &c in &cpus {
                if c < 128 {
                    mask |= 1u128 << c;
                }
            }
            CpuCluster {
                uarch,
                max_freq_mhz,
                cpus,
                cpu_mask_hex: format!("0x{mask:x}"),
            }
        })
        .collect()
}

pub fn detect_host_clusters() -> Vec<CpuCluster> {
    let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") else {
        return Vec::new();
    };
    let mut freqs = BTreeMap::new();
    for cpu in 0..64usize {
        let p = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
        if let Ok(s) = fs::read_to_string(p)
            && let Ok(khz) = s.trim().parse::<u64>()
        {
            freqs.insert(cpu, khz);
        }
    }
    parse_cpu_clusters(&cpuinfo, &freqs)
}

pub fn detect_device_model_name() -> String {
    if let Ok(m) = fs::read_to_string("/proc/device-tree/model") {
        let clean = m.trim_matches(|c: char| c == '\0' || c.is_whitespace());
        if !clean.is_empty() {
            return clean.to_string();
        }
    }
    if Path::new("/system/bin/getprop").exists()
        && let Ok(out) = std::process::Command::new("/system/bin/getprop")
            .arg("ro.product.model")
            .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            && out.status.success()
        {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Runs an on-device L1 bit-exact self-check across all supported ISAs (`Q4_0`, `Q4_1`, `Q8_0`, `Q5_K`, `Q6_K`).
pub fn run_kernel_self_check() -> (bool, Vec<&'static str>, Option<String>) {
    let isas = supported_isas();
    let names: Vec<&'static str> = isas.iter().map(|i| i.name()).collect();
    let d_pos = f32_to_f16(0.03125);
    let d_neg = f32_to_f16(-0.0625);

    let w_q40 = [
        BlockQ4_0 {
            d: d_pos,
            qs: [0x1E; 16],
        },
        BlockQ4_0 {
            d: d_neg,
            qs: [0xA5; 16],
        },
    ];
    let x_q80 = [
        BlockQ8_0 {
            d: d_pos,
            qs: [i8::MIN; 32],
        },
        BlockQ8_0 {
            d: d_pos,
            qs: [37; 32],
        },
    ];
    let w_q41 = [
        BlockQ4_1 {
            d: d_pos,
            m: d_neg,
            qs: [0x73; 16],
        },
        BlockQ4_1 {
            d: d_pos,
            m: d_pos,
            qs: [0xC4; 16],
        },
    ];
    let x_q81 = [
        BlockQ8_1 {
            d: d_pos,
            s: f32_to_f16(1.0),
            qs: [-19; 32],
        },
        BlockQ8_1 {
            d: d_pos,
            s: f32_to_f16(-0.5),
            qs: [23; 32],
        },
    ];
    let w_q5k = [BlockQ5K {
        d: d_pos,
        dmin: d_pos,
        scales: [0x12; 12],
        qh: [0x5A; 32],
        qs: [0x96; 128],
    }];
    let w_q6k = [BlockQ6K {
        ql: [0x6B; 128],
        qh: [0x95; 64],
        scales: [-3, 5, -2, 4, 1, -6, 3, 2, -4, 7, -1, 3, 2, -5, 6, -2],
        d: d_pos,
    }];
    let x_q8k = [BlockQ8K {
        d: 0.03125,
        qs: [5; 256],
        bsums: [80; 16],
    }];

    let ref_q40 = generic::vec_dot_q4_0_q8_0(&w_q40, &x_q80);
    let ref_q80 = generic::vec_dot_q8_0_q8_0(&x_q80, &x_q80);
    let ref_q41 = generic::vec_dot_q4_1_q8_1(&w_q41, &x_q81);
    let ref_q5k = generic::vec_dot_q5_k_q8_k(&w_q5k, &x_q8k);
    let ref_q6k = generic::vec_dot_q6_k_q8_k(&w_q6k, &x_q8k);

    for &isa in &isas {
        let kt = select_kernels(isa);
        let checks = [
            ("Q4_0", (kt.vec_dot_q4_0_q8_0)(&w_q40, &x_q80), ref_q40),
            ("Q8_0", (kt.vec_dot_q8_0_q8_0)(&x_q80, &x_q80), ref_q80),
            ("Q4_1", (kt.vec_dot_q4_1_q8_1)(&w_q41, &x_q81), ref_q41),
            ("Q5_K", (kt.vec_dot_q5_k_q8_k)(&w_q5k, &x_q8k), ref_q5k),
            ("Q6_K", (kt.vec_dot_q6_k_q8_k)(&w_q6k, &x_q8k), ref_q6k),
        ];
        for (tag, got, exp) in checks {
            if got.to_bits() != exp.to_bits() {
                return (
                    false,
                    names,
                    Some(format!(
                        "ISA '{}' failed {} self-check: got={} (0x{:08x}) != ref={} (0x{:08x})",
                        isa.name(),
                        tag,
                        got,
                        got.to_bits(),
                        exp,
                        exp.to_bits()
                    )),
                );
            }
        }
    }

    (true, names, None)
}

/// Runs a memory-safe multi-threaded STREAM Read & Copy bandwidth test using `ThreadPool`.
/// Returns `(per_array_mib, best_read_gbs, best_copy_gbs)`.
pub fn run_stream_bandwidth(
    threads: usize,
    requested_mib: Option<usize>,
    avail_bytes: Option<u64>,
) -> (usize, f64, f64) {
    let default_mib = match avail_bytes {
        Some(b) => ((b / (8 * 1024 * 1024)) as usize).clamp(16, 256),
        None => 128,
    };
    let mib = requested_mib.unwrap_or(default_mib).clamp(8, 1024);
    let n_u64 = (mib * 1024 * 1024) / core::mem::size_of::<u64>();
    let bytes = (n_u64 * core::mem::size_of::<u64>()) as f64;

    let src: Vec<u64> = (0..n_u64)
        .map(|i| {
            (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(1)
        })
        .collect();
    let mut dst: Vec<u64> = vec![0u64; n_u64];

    let n_thr = threads.max(1);
    let pool = ThreadPool::new(n_thr);
    let chunk = n_u64.div_ceil(n_thr).max(4096);

    // Compute per-chunk partial sums in a tiny scratch array (safe Rust, zero `unsafe`!).
    let n_chunks = n_u64.div_ceil(chunk);
    let partials: Vec<std::sync::atomic::AtomicU64> = (0..n_chunks)
        .map(|_| std::sync::atomic::AtomicU64::new(0))
        .collect();

    let run_read = || {
        pool.parallel_for(n_u64, chunk, |s0, s1| {
            let slice = &src[s0..s1];
            let mut a0 = 0u64;
            let mut a1 = 0u64;
            let mut a2 = 0u64;
            let mut a3 = 0u64;
            let mut i = 0usize;
            while i + 4 <= slice.len() {
                a0 = a0.wrapping_add(slice[i]);
                a1 = a1.wrapping_add(slice[i + 1]);
                a2 = a2.wrapping_add(slice[i + 2]);
                a3 = a3.wrapping_add(slice[i + 3]);
                i += 4;
            }
            while i < slice.len() {
                a0 = a0.wrapping_add(slice[i]);
                i += 1;
            }
            partials[s0 / chunk].store(
                a0.wrapping_add(a1).wrapping_add(a2).wrapping_add(a3),
                std::sync::atomic::Ordering::Relaxed,
            );
        });
    };

    // Ramp DVFS governor (`sched_pixel` / `schedutil`) for >= 350 ms BEFORE measuring,
    // otherwise mobile cores start at 500 MHz and under-report DRAM bandwidth!
    let t_warm = Instant::now();
    while t_warm.elapsed().as_secs_f64() < 0.35 {
        run_read();
    }
    dst.copy_from_slice(&src);

    // Measure sustained Read for >= 1.0 s (and >= 8 sweeps), taking the best steady-state sweep
    // and also keeping the sustained mean if needed.
    let mut best_read_sec = f64::MAX;
    let mut read_iters = 0usize;
    let t_read_total = Instant::now();
    while read_iters < 8 || t_read_total.elapsed().as_secs_f64() < 1.0 {
        let t0 = Instant::now();
        run_read();
        let sec = t0.elapsed().as_secs_f64();
        if sec < best_read_sec {
            best_read_sec = sec;
        }
        read_iters += 1;
    }

    let mut best_copy_sec = f64::MAX;
    let mut copy_iters = 0usize;
    let t_copy_total = Instant::now();
    while copy_iters < 8 || t_copy_total.elapsed().as_secs_f64() < 1.0 {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for (d_chunk, s_chunk) in dst.chunks_mut(chunk).zip(src.chunks(chunk)) {
                s.spawn(move || {
                    d_chunk.copy_from_slice(s_chunk);
                });
            }
        });
        let sec = t0.elapsed().as_secs_f64();
        if sec < best_copy_sec {
            best_copy_sec = sec;
        }
        copy_iters += 1;
    }

    let sink: u64 = partials
        .iter()
        .map(|p| p.load(std::sync::atomic::Ordering::Relaxed))
        .fold(dst[0], u64::wrapping_add);
    std::hint::black_box(sink);

    let read_gbs = (bytes / best_read_sec.max(1e-9)) / 1e9;
    let copy_gbs = (2.0 * bytes / best_copy_sec.max(1e-9)) / 1e9;
    (mib, read_gbs, copy_gbs)
}

/// Deterministic 64-bit FNV-1a digest over generated token IDs for fast L3 parity verification.
pub fn hash_token_sequence(tokens: &[u32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

pub struct ProbeOpts {
    pub threads: usize,
    pub stream_mib: Option<usize>,
    pub skip_stream: bool,
    pub json: bool,
}

pub fn parse_probe_opts(args: &[String]) -> Result<ProbeOpts> {
    let mut threads = tm_cpu::recommended_threads();
    let mut stream_mib: Option<usize> = None;
    let mut skip_stream = false;
    let mut json = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-t" | "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .context("missing value for -t/--threads")?
                    .parse()
                    .context("invalid -t/--threads")?;
            }
            "--stream-mib" => {
                i += 1;
                stream_mib = Some(
                    args.get(i)
                        .context("missing value for --stream-mib")?
                        .parse()
                        .context("invalid --stream-mib")?,
                );
            }
            "--no-stream" => {
                skip_stream = true;
            }
            "--json" => {
                json = true;
            }
            other => bail!("unknown probe argument: {other}"),
        }
        i += 1;
    }

    Ok(ProbeOpts {
        threads: threads.max(1),
        stream_mib,
        skip_stream,
        json,
    })
}

pub fn cmd_probe(opts: ProbeOpts) -> Result<()> {
    let model_name = detect_device_model_name();
    let logical_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let caps = detect_caps();
    let active = effective_isa();
    let clusters = detect_host_clusters();
    let mem_budget = probe_system_memory().ok();
    let (self_ok, tested_isas, self_err) = run_kernel_self_check();

    let stream = if opts.skip_stream {
        None
    } else {
        Some(run_stream_bandwidth(
            opts.threads,
            opts.stream_mib,
            mem_budget.map(|b| b.available_bytes),
        ))
    };
    let mem_stats = probe_process_memory_stats();
    let telemetry = collect_telemetry();

    if opts.json {
        let isas_json: Vec<String> = tested_isas.iter().map(|s| format!("\"{s}\"")).collect();
        let clusters_json: Vec<String> = clusters
            .iter()
            .map(|c| {
                let cpus_str: Vec<String> = c.cpus.iter().map(|i| i.to_string()).collect();
                format!(
                    "{{\"uarch\":\"{}\",\"max_freq_mhz\":{},\"count\":{},\"cpus\":[{}],\"cpu_mask_hex\":\"{}\"}}",
                    json_escape(&c.uarch),
                    c.max_freq_mhz,
                    c.cpus.len(),
                    cpus_str.join(","),
                    c.cpu_mask_hex
                )
            })
            .collect();

        let mem_json = match mem_budget {
            Some(b) => format!(
                "{{\"total_bytes\":{},\"available_bytes\":{},\"cgroup_max_bytes\":{}}}",
                b.total_bytes,
                b.available_bytes,
                opt_u64_json(b.cgroup_max_bytes)
            ),
            None => "null".to_string(),
        };

        let stream_json = match stream {
            Some((mib, rgbs, cgbs)) => format!(
                "{{\"threads\":{},\"array_mib\":{mib},\"read_gbs\":{rgbs:.2},\"copy_gbs\":{cgbs:.2}}}",
                opts.threads
            ),
            None => "null".to_string(),
        };

        let err_json = match &self_err {
            Some(e) => format!("\"{}\"", json_escape(e)),
            None => "null".to_string(),
        };

        println!(
            "{{\"schema_version\":1,\"device_model\":\"{}\",\"os\":\"{}\",\"arch\":\"{}\",\
            \"logical_cpus\":{},\"active_isa\":\"{}\",\
            \"caps\":{{\"neon\":{},\"dotprod\":{},\"i8mm\":{},\"avx2\":{},\"avx512_vnni\":{}}},\
            \"supported_isas\":[{}],\"clusters\":[{}],\"memory\":{},\
            \"process_rss_bytes\":{},\"process_peak_rss_bytes\":{},\
            \"process_private_dirty_bytes\":{},\"process_pss_anon_bytes\":{},\
            \"kernel_self_check\":{{\"passed\":{},\"error\":{}}},\
            \"stream_bw\":{},\"telemetry\":{}}}",
            json_escape(&model_name),
            std::env::consts::OS,
            std::env::consts::ARCH,
            logical_cpus,
            active.name(),
            caps.neon,
            caps.dotprod,
            caps.i8mm,
            caps.avx2,
            caps.avx512_vnni,
            isas_json.join(","),
            clusters_json.join(","),
            mem_json,
            opt_u64_json(mem_stats.rss_bytes),
            opt_u64_json(mem_stats.peak_rss_bytes),
            opt_u64_json(mem_stats.private_dirty_bytes),
            opt_u64_json(mem_stats.pss_anon_bytes),
            self_ok,
            err_json,
            stream_json,
            telemetry.to_json()
        );
    } else {
        println!("=== tinymodel Device Probe ===");
        println!(
            "Device: {} ({}/{}, logical_cpus={})",
            model_name,
            std::env::consts::OS,
            std::env::consts::ARCH,
            logical_cpus
        );
        println!(
            "ISA: active={} | supported={:?}",
            active.name(),
            tested_isas
        );
        if !clusters.is_empty() {
            println!("CPU Clusters:");
            for c in &clusters {
                println!(
                    "  {}x {:<14} max={} MHz cpus={:?} mask={}",
                    c.cpus.len(),
                    c.uarch,
                    c.max_freq_mhz,
                    c.cpus,
                    c.cpu_mask_hex
                );
            }
        }
        if let Some(b) = mem_budget {
            println!(
                "Memory: total={:.1} MiB | available={:.1} MiB | cgroup_max={:?}",
                b.total_bytes as f64 / 1048576.0,
                b.available_bytes as f64 / 1048576.0,
                b.cgroup_max_bytes
            );
        }
        println!(
            "Kernel L1 Self-Check: {}",
            if self_ok { "PASSED" } else { "FAILED" }
        );
        if let Some((mib, rgbs, cgbs)) = stream {
            println!(
                "STREAM Bandwidth ({} threads, {} MiB): Read={:.2} GB/s | Copy={:.2} GB/s",
                opts.threads, mib, rgbs, cgbs
            );
        }
        if let Some(tc) = telemetry.max_temp_c {
            println!(
                "Telemetry: max_temp={:.1} C | cur_freq_max={:?} MHz | power_mw={:?}",
                tc, telemetry.cur_freq_mhz_max, telemetry.battery_power_mw
            );
        }
    }

    if !self_ok {
        bail!(
            "on-device L1 kernel self-check failed: {}",
            self_err.unwrap_or_default()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pixel8_tensor_g3_clusters_and_masks() {
        // Synthetic Tensor G3 `/proc/cpuinfo`: 4x A510 (0..3), 4x A715 (4..7), 1x X3 (8)
        let mut cpuinfo = String::new();
        for cpu in 0..4 {
            cpuinfo.push_str(&format!(
                "processor\t: {cpu}\nCPU implementer\t: 0x41\nCPU part\t: 0xd46\n\n"
            ));
        }
        for cpu in 4..8 {
            cpuinfo.push_str(&format!(
                "processor\t: {cpu}\nCPU implementer\t: 0x41\nCPU part\t: 0xd4d\n\n"
            ));
        }
        cpuinfo.push_str("processor\t: 8\nCPU implementer\t: 0x41\nCPU part\t: 0xd4e\n\n");

        let mut freqs = BTreeMap::new();
        for cpu in 0..4 {
            freqs.insert(cpu, 1_704_000);
        }
        for cpu in 4..8 {
            freqs.insert(cpu, 2_367_000);
        }
        freqs.insert(8, 2_910_000);

        let clusters = parse_cpu_clusters(&cpuinfo, &freqs);
        assert_eq!(clusters.len(), 3);
        assert_eq!(clusters[0].uarch, "Cortex-A510");
        assert_eq!(clusters[0].cpu_mask_hex, "0xf");
        assert_eq!(clusters[1].uarch, "Cortex-A715");
        assert_eq!(clusters[1].cpu_mask_hex, "0xf0");
        assert_eq!(clusters[2].uarch, "Cortex-X3");
        assert_eq!(clusters[2].cpu_mask_hex, "0x100");
    }

    #[test]
    fn kernel_self_check_passes_on_host() {
        let (ok, isas, err) = run_kernel_self_check();
        assert!(ok, "self-check failed: {err:?}");
        assert!(!isas.is_empty());
    }

    #[test]
    fn parses_android_dumpsys_thermal_output() {
        let sample = "\
Thermal Status: 1
Cached temperatures:
\tTemperature{mValue=57.000004, mType=0, mName=LITTLE, mStatus=0}
\tTemperature{mValue=27.765001, mType=-1, mName=neutral_therm, mStatus=0}
\tTemperature{mValue=60.000004, mType=0, mName=BIG, mStatus=0}
\tTemperature{mValue=37.05332, mType=3, mName=VIRTUAL-SKIN, mStatus=0}
HAL Ready: true
";
        let (st, skin, zones) = parse_android_dumpsys_thermal(sample);
        assert_eq!(st, Some(1));
        assert!((skin.unwrap() - 37.05332).abs() < 1e-3);
        assert_eq!(zones[0].0, "BIG");
        assert!((zones[0].1 - 60.0).abs() < 1e-3);
    }
}
