//! Runtime CPU ISA capability probe and `TM_FORCE_ISA` override.
//!
//! Design inspired by Strata `src/quant/isa.cpp` (MIT License,
//! Copyright (c) 2026 Niko1221 and the Strata contributors).

/// Supported CPU ISA tiers, ordered roughly from narrowest to widest per architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    Generic,
    NeonV80,
    NeonDotprod,
    NeonI8mm,
    Avx2,
    Avx512,
}

impl Isa {
    pub const fn name(self) -> &'static str {
        match self {
            Isa::Generic => "generic",
            Isa::NeonV80 => "neon",
            Isa::NeonDotprod => "dotprod",
            Isa::NeonI8mm => "i8mm",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "generic" | "scalar" => Some(Isa::Generic),
            "neon" | "neon_v80" | "armv8.0" => Some(Isa::NeonV80),
            "dotprod" | "neon_dotprod" => Some(Isa::NeonDotprod),
            "i8mm" | "neon_i8mm" => Some(Isa::NeonI8mm),
            "avx2" => Some(Isa::Avx2),
            "avx512" | "avx-512" => Some(Isa::Avx512),
            _ => None,
        }
    }
}

/// Detected hardware ISA capabilities.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Caps {
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub avx2: bool,
    /// `AVX-512 F + BW + VL + FMA` (Skylake-X and newer).
    pub avx512f: bool,
    /// `AVX-512 VNNI` (Cascade Lake / Ice Lake / Sapphire Rapids / Zen 4+, which have dual 512-bit
    /// execution without Skylake-X's single-port FMA512 + Heavy License frequency throttling).
    pub avx512_vnni: bool,
}

/// Probes the current CPU's runtime ISA features.
pub fn detect_caps() -> Caps {
    #[cfg(target_arch = "aarch64")]
    {
        Caps {
            neon: true,
            dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
            avx2: false,
            avx512f: false,
            avx512_vnni: false,
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c");
        let avx512f = avx2
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl");
        Caps {
            neon: false,
            dotprod: false,
            i8mm: false,
            avx2,
            avx512f,
            avx512_vnni: avx512f && std::arch::is_x86_feature_detected!("avx512vnni"),
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Caps::default()
    }
}

/// Returns whether `isa` can safely execute on the current CPU.
pub fn isa_supported(isa: Isa, caps: Caps) -> bool {
    match isa {
        Isa::Generic => true,
        Isa::NeonV80 => caps.neon,
        Isa::NeonDotprod => caps.neon && caps.dotprod,
        Isa::NeonI8mm => caps.neon && caps.dotprod && caps.i8mm,
        Isa::Avx2 => caps.avx2,
        Isa::Avx512 => caps.avx512f,
    }
}

/// Returns the fastest ISA supported by `caps` (auto-selects `Avx512` only when `avx512_vnni` is
/// present so Skylake-X cloud CPUs avoid single-port FMA512 + Heavy License downclocking).
pub fn best_isa(caps: Caps) -> Isa {
    if caps.i8mm && caps.dotprod && caps.neon {
        Isa::NeonI8mm
    } else if caps.dotprod && caps.neon {
        Isa::NeonDotprod
    } else if caps.neon {
        Isa::NeonV80
    } else if caps.avx512_vnni {
        Isa::Avx512
    } else if caps.avx2 {
        Isa::Avx2
    } else {
        Isa::Generic
    }
}

/// Returns the effective ISA after applying `TM_FORCE_ISA` (if set and supported).
pub fn effective_isa() -> Isa {
    let caps = detect_caps();
    if let Ok(val) = std::env::var("TM_FORCE_ISA")
        && !val.is_empty()
        && let Some(forced) = Isa::parse(&val)
        && isa_supported(forced, caps)
    {
        return forced;
    }
    best_isa(caps)
}

/// Returns all ISAs supported by the current CPU, ordered from narrowest to widest.
pub fn supported_isas() -> Vec<Isa> {
    let caps = detect_caps();
    [
        Isa::Generic,
        Isa::NeonV80,
        Isa::NeonDotprod,
        Isa::NeonI8mm,
        Isa::Avx2,
        Isa::Avx512,
    ]
    .into_iter()
    .filter(|&isa| isa_supported(isa, caps))
    .collect()
}
