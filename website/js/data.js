export const I18N = {
  zh: {
    "nav.profiles": "硬件分档",
    "nav.benchmarks": "实测基准",
    "nav.models": "模型与微调",
    "nav.architecture": "架构设计",
    "nav.quickstart": "快速开始",
    "nav.roadmap": "路线图",
    "hero.eyebrow": "Rust Stable · Metal / Vulkan / CUDA · MIT / Apache-2.0",
    "hero.title_1": "从 512MB 树莓派到 Blackwell 独显的",
    "hero.title_highlight": "纯 Rust 端侧与全场景 LLM 引擎",
    "hero.sub": "tinymodel 用同一份 Rust stable 代码库覆盖 Raspberry Pi Zero 2W（≤0.5B）、手机与 ARM 开发板（≤5B）及 Apple / AMD / NVIDIA 主机 GPU（稠密 ≤9B 与 26B–120B 稀疏 MoE）。零拷贝加载单/多分片 GGUF 与 .strata 容器，一等支持 Qwen3.5/3.6 GDN+MTP、Gemma 4 PLE、Mamba-2、多模态视觉及端侧 QLoRA 微调。",
    "hero.cta_start": "快速生成命令",
    "hero.cta_bench": "查看真机实测",
    "hero.cta_repo": "GitHub 仓库",
    "kpi.1.label": "Qwen3.6-35B Q8 + MTP",
    "kpi.1.desc": "Blackwell CUDA 图捕获投机解码；30.6G DDR 卸载+64 槽缓存亦达 109.6 tok/s",
    "kpi.2.label": "RX 9060 XT Vulkan 加速",
    "kpi.2.desc": "Decode 反超 ROCm 7.1.1 HIP 最高 2.26×，零 SDK 依赖预编译 SPIR-V",
    "kpi.3.label": "数值与投机确定性门禁",
    "kpi.3.desc": "SIMD/批 GDN/热专家缓存逐位一致，开启投机贪心输出 100% 不变",
    "kpi.4.label": "micro 极简静态二进制",
    "kpi.4.desc": "512MB Zero 2W 纯 NEON 查表；Pixel 8 跑 Gemma 4 E2B 总 RSS 仅 1.47 GiB",
    "profiles.tag": "Hardware Profiles",
    "profiles.title": "一份代码，按三档硬件精准裁剪",
    "profiles.desc": "通过 Cargo feature 与运行时 ISA / GPU 自动分派（NEON / dotprod / i8mm / AVX2 + Metal / Vulkan / CUDA），兼顾百兆级微设备、移动终端与旗舰独显。",
    "bench.tag": "Verified Benchmarks",
    "bench.title": "真机实测性能与内存基线",
    "bench.desc": "所有数据均来自仓库内可复现基准协议（与同一台机器上的 Strata CUDA / ROCm HIP / llama.cpp 对照实测）。",
    "models.tag": "First-Class Architectures",
    "models.title": "全谱系小模型、四大前沿 MoE、多模态与端侧微调",
    "models.desc": "不走动态计算图，每个模型结构手写零堆分配 forward、多步投机验证图捕获与独立激活重算 QLoRA 反向。",
    "arch.tag": "System Architecture",
    "arch.title": "为端侧带宽红线与 GPU 极致吞吐而生的分层设计",
    "arch.desc": "严格隔离 unsafe 边界（仅限内核、GPU 后端、FFI 与 mmap），在生成 token 热路径上实现零堆分配。",
    "cli.tag": "Interactive Quickstart",
    "cli.title": "交互式 CLI 命令生成器",
    "cli.desc": "按你的任务场景、硬件后端（含 CUDA / Vulkan / Metal / CPU）与卸载缓存策略，即时生成可直接复制执行的命令。",
    "cli.mode": "子命令场景",
    "cli.backend": "计算后端",
    "cli.model": "示例模型",
    "cli.opt": "优化特性",
    "roadmap.tag": "Project Roadmap",
    "roadmap.title": "开发进度与里程碑",
    "roadmap.desc": "以可量化的同机吞吐、多步 KL 散度、0-ULP 对齐与二进制体积作为每阶段验收门禁。",
    "copy": "复制",
    "copied": "已复制到剪贴板",
    "footer.license": "采用 MIT OR Apache-2.0 双许可开源",
    "footer.docs": "架构文档",
    "footer.decisions": "ADR 决策记录",
    "footer.blackwell": "Blackwell 实测报告",
    "footer.roadmap": "路线图",
    "footer.deploy": "低成本云部署指南"
  },
  en: {
    "nav.profiles": "Profiles",
    "nav.benchmarks": "Benchmarks",
    "nav.models": "Models & QLoRA",
    "nav.architecture": "Architecture",
    "nav.quickstart": "Quickstart",
    "nav.roadmap": "Roadmap",
    "hero.eyebrow": "Rust Stable · Metal / Vulkan / CUDA · MIT / Apache-2.0",
    "hero.title_1": "From 512MB Pi Zero 2W to Blackwell GPUs:",
    "hero.title_highlight": "Pure Rust LLM Inference Engine",
    "hero.sub": "One Rust stable codebase spanning 512MB Raspberry Pi Zero 2W (≤0.5B), phones & ARM SBCs (≤5B), and Apple / AMD / NVIDIA GPUs (dense ≤9B & 26B–120B sparse MoEs). Zero-copy multi-shard GGUF & .strata loading, first-class Qwen3.5/3.6 GDN+MTP, Gemma 4 PLE, Mamba-2, Multimodal ViT, and on-device QLoRA.",
    "hero.cta_start": "Command Builder",
    "hero.cta_bench": "View Benchmarks",
    "hero.cta_repo": "GitHub Repo",
    "kpi.1.label": "Qwen3.6-35B Q8 + MTP",
    "kpi.1.desc": "CUDA Graph speculative decode on Blackwell; 109.6 tok/s at 30.6G via DDR offload + 64-slot cache",
    "kpi.2.label": "RX 9060 XT Vulkan",
    "kpi.2.desc": "Up to 2.26× faster decode than ROCm 7.1.1 HIP via precompiled SPIR-V",
    "kpi.3.label": "Bit-Exact Determinism",
    "kpi.3.desc": "0-ULP SIMD, batched GDN & hot-expert VRAM cache; deterministic speculative decode",
    "kpi.4.label": "micro Static Binary",
    "kpi.4.desc": "ARMv8.0 LUT on 512MB Zero 2W; Gemma 4 E2B runs in 1.47 GiB RSS on Pixel 8",
    "profiles.tag": "Hardware Profiles",
    "profiles.title": "One Codebase, Three Hardware Profiles",
    "profiles.desc": "Cargo features plus runtime ISA / GPU dispatch (NEON / dotprod / i8mm / AVX2 + Metal / Vulkan / CUDA) tailored from ultra-low-power SBCs to flagship GPUs.",
    "bench.tag": "Verified Benchmarks",
    "bench.title": "Real-Device Performance & Memory Baselines",
    "bench.desc": "Measured on real hardware against Strata CUDA, ROCm HIP, and llama.cpp on the exact same machine under our benchmark protocol.",
    "models.tag": "First-Class Architectures",
    "models.title": "Hybrid Small Models, Frontier MoEs, Vision & QLoRA",
    "models.desc": "Hand-written zero-allocation forward passes, CUDA-graphed speculative verification, and safe-Rust recomputed QLoRA backward passes.",
    "arch.tag": "System Architecture",
    "arch.title": "Engineered for Edge Memory Redlines & Peak GPU Throughput",
    "arch.desc": "Unsafe code strictly isolated to kernel, GPU, FFI, and mmap crates. Zero heap allocations on the token generation hot path.",
    "cli.tag": "Interactive Quickstart",
    "cli.title": "Interactive CLI Command Builder",
    "cli.desc": "Pick your workload, backend (CUDA / Vulkan / Metal / CPU), model, and offloading strategy to generate copy-ready commands.",
    "cli.mode": "Subcommand",
    "cli.backend": "Backend",
    "cli.model": "Sample Model",
    "cli.opt": "Optimization",
    "roadmap.tag": "Project Roadmap",
    "roadmap.title": "Milestones & Delivery Status",
    "roadmap.desc": "Every milestone is gated by same-machine throughput, multi-token KL divergence, 0-ULP parity, and binary-size checks.",
    "copy": "Copy",
    "copied": "Copied to clipboard",
    "footer.license": "Dual-licensed under MIT OR Apache-2.0",
    "footer.docs": "Architecture",
    "footer.decisions": "ADRs",
    "footer.blackwell": "Blackwell Report",
    "footer.roadmap": "Roadmap",
    "footer.deploy": "Cloud Deploy Guide"
  }
};

export const PROFILES = [
  {
    id: "micro",
    badge: "profile: micro",
    scale: "≤ 0.5B Params",
    featured: false,
    zh: {
      title: "极低功耗微型设备",
      specs: [
        ["目标内存", "512 MB（可用 ~350 MB）"],
        ["CPU 特性", "ARMv8.0 NEON（无 dotprod / FP16）"],
        ["核心技术", "16 行 T-MAC vqtbl1q 查表 + BPE 词表裁剪"],
        ["静态体积", "1.18 – 1.25 MiB stripped ELF"]
      ],
      devices: ["Pi Zero 2W", "Pi 3 / 4B", "Cortex-A53", "4×A510 实测 20.9 tok/s"]
    },
    en: {
      title: "Ultra-Low-Power Micro Devices",
      specs: [
        ["Memory Budget", "512 MB (~350 MB usable)"],
        ["CPU ISA", "ARMv8.0 NEON (no dotprod / FP16)"],
        ["Key Tech", "16-row T-MAC vqtbl1q LUT + Vocab Pruning"],
        ["Static Binary", "1.18 – 1.25 MiB stripped ELF"]
      ],
      devices: ["Pi Zero 2W", "Pi 3 / 4B", "Cortex-A53", "4×A510: 20.9 tok/s"]
    },
    cmd: "cargo zigbuild --release --target aarch64-unknown-linux-musl --no-default-features --features micro"
  },
  {
    id: "edge",
    badge: "profile: edge",
    scale: "≤ 5B Params",
    featured: false,
    zh: {
      title: "手机与高性能 ARM 开发板",
      specs: [
        ["目标内存", "4 – 16 GB（RK3588 可达 32 GB）"],
        ["CPU / GPU", "NEON dotprod / i8mm + Mali / V3DV Vulkan"],
        ["核心技术", "零膨胀 Q4_0x4 + 双门控 Vulkan MoE + 原生 Q2_0"],
        ["真机实测", "Pixel 8 带宽利用率 90.9%，脏内存 <56 MiB"]
      ],
      devices: ["Google Pixel 8 (Tensor G3)", "Rockchip RK3588", "Raspberry Pi 5", "Mali-G715 / G610"]
    },
    en: {
      title: "Smartphones & ARM Edge SBCs",
      specs: [
        ["Memory Budget", "4 – 16 GB (up to 32 GB on RK3588)"],
        ["CPU / GPU", "NEON dotprod / i8mm + Mali / V3DV Vulkan"],
        ["Key Tech", "Zero-bloat Q4_0x4 + Dual-Gate Vulkan MoE + Q2_0"],
        ["Real Device", "Pixel 8 90.9% BW util, <56 MiB dirty RAM"]
      ],
      devices: ["Google Pixel 8 (Tensor G3)", "Rockchip RK3588", "Raspberry Pi 5", "Mali-G715 / G610"]
    },
    cmd: "cargo zigbuild --release --target aarch64-unknown-linux-musl --features edge"
  },
  {
    id: "host",
    badge: "profile: host",
    scale: "≤ 9B & 26B–120B MoE",
    featured: true,
    zh: {
      title: "Apple Silicon 与桌面/服务器独显主机",
      specs: [
        ["目标内存", "16 GB+（支持零拷贝 DDR 卸载 + 64 槽热专家缓存）"],
        ["计算后端", "CUDA (sm_75–120) / Vulkan CoopMat / Metal / AVX2"],
        ["核心技术", "WMMA64 + CudaGraph 投机验证 + 28× 吸收式 MLA"],
        ["扩展能力", "内置 MTP / DFly2 / DSpark、多模态 ViT、GPU QLoRA"]
      ],
      devices: ["RTX PRO 6000 Blackwell", "RTX 30/40/50 (32GB)", "AMD RX 9060 XT", "Apple M4 (Metal)"]
    },
    en: {
      title: "Apple Silicon & Desktop / Server GPUs",
      specs: [
        ["Memory Budget", "16 GB+ (Zero-copy DDR offload + 64-slot VRAM cache)"],
        ["Backends", "CUDA (sm_75–120) / Vulkan CoopMat / Metal / AVX2"],
        ["Key Tech", "WMMA64 + CudaGraph Spec Verify + 28× Absorbed MLA"],
        ["Capabilities", "Built-in MTP / DFly2 / DSpark, Vision ViT, GPU QLoRA"]
      ],
      devices: ["RTX PRO 6000 Blackwell", "RTX 30/40/50 (32GB)", "AMD RX 9060 XT", "Apple M4 (Metal)"]
    },
    cmd: "cargo build --release --features \"vulkan,cuda,train\""
  }
];

export const BENCHMARKS = {
  cuda: {
    zh: {
      label: "NVIDIA RTX PRO 6000 Blackwell (CUDA)",
      note: "同机 RTX PRO 6000 Blackwell (98 GiB, sm_120) 实测：完成 ADR-025 / ADR-026 两轮优化后，对比初版基线与闭源 Strata CUDA（单位 tok/s）。",
      baseName: "基线 / Strata CUDA"
    },
    en: {
      label: "NVIDIA RTX PRO 6000 Blackwell (CUDA)",
      note: "Measured on the same RTX PRO 6000 Blackwell (98 GiB, sm_120) after ADR-025 & ADR-026 optimizations vs initial baseline & Strata CUDA (tok/s).",
      baseName: "Baseline / Strata CUDA"
    },
    rows: [
      { model: "Qwen3.6-35B-A3B Q8_0 (+内置 MTP)", sub: "Blackwell CUDA · p_code tg128 (省 6.6G 显存)", tm: 239.39, base: 123.95, baseLabel: "初版 CUDA", ratio: "+93.1% (1.93×)" },
      { model: "Qwen3.6-35B-A3B Q8_0 (Prefill WMMA64)", sub: "Blackwell CUDA · pp512 (寄存器批 GDN)", tm: 2579.79, base: 413.48, baseLabel: "Wave-1 CUDA", ratio: "6.24×" },
      { model: "Qwen3.8-Flash-Next Q2_0 (双分片零重排)", sub: "Blackwell CUDA · pp512 (省 44.9G 磁盘)", tm: 1133.54, base: 1115.6, baseLabel: "Strata CUDA", ratio: "101.6%" },
      { model: "Inkling-2-Mini 26B Q8_0 (+DSpark k=7)", sub: "Blackwell CUDA · p_code tg128 (省 5.5G 显存)", tm: 114.56, base: 94.5, baseLabel: "Strata CUDA", ratio: "121.2%" },
      { model: "Qwen3.6-35B Q8_0 (10层卸载 + 64槽缓存)", sub: "显存仅 30.6 GiB (<32GB) · tg64 (0-ULP)", tm: 109.61, base: 71.2, baseLabel: "无缓存 DDR 卸载", ratio: "+54.0%" },
      { model: "Tencent Hy4-Flash 120B (.strata + DFly2)", sub: "59.5 GiB 显存 (-4.5G vs Strata) · p64 tg128", tm: 44.58, base: 36.75, baseLabel: "Wave-1 吸收式 MLA", ratio: "+21.3%" }
    ]
  },
  vulkan: {
    zh: {
      label: "AMD RX 9060 XT & Blackwell (Vulkan)",
      note: "预编译 SPIR-V 1.2 + 16×16×16 CoopMat + 双门控融合 MoE，对比同机 ROCm 7.1.1 HIP 与 Vulkan 基线（单位 tok/s）。",
      baseName: "ROCm HIP / 基线"
    },
    en: {
      label: "AMD RX 9060 XT & Blackwell (Vulkan)",
      note: "Precompiled SPIR-V 1.2 + 16×16×16 CoopMat + Dual-Gate MoE vs same-machine ROCm 7.1.1 HIP & baselines (tok/s).",
      baseName: "ROCm HIP / Base"
    },
    rows: [
      { model: "SmolLM2-360M Q4_0", sub: "RX 9060 XT · tg128", tm: 608.8, base: 269.5, baseLabel: "ROCm HIP", ratio: "225.9%" },
      { model: "Qwen3.5-0.8B Q4_0 (GDN)", sub: "RX 9060 XT · tg128", tm: 366.5, base: 168.3, baseLabel: "ROCm HIP", ratio: "217.8%" },
      { model: "Qwen3.5-2B Q4_0 (GDN)", sub: "RX 9060 XT · tg128", tm: 184.5, base: 134.3, baseLabel: "ROCm HIP", ratio: "137.4%" },
      { model: "Qwen3.6-35B-A3B Q8_0 (256-Exp MoE)", sub: "Blackwell Vulkan · tg128 (moe_dual_gu_act)", tm: 107.7, base: 105.8, baseLabel: "分立 MoE", ratio: "pp512 2943" },
      { model: "Qwen3.5-0.8B Q4_0 (Prefill)", sub: "RX 9060 XT · pp512", tm: 13566.6, base: 11665.2, baseLabel: "ROCm HIP", ratio: "116.3%" }
    ]
  },
  m4: {
    zh: {
      label: "Apple M4 (Metal & CPU)",
      note: "Apple M4 16GB macOS 实测：Metal Decode 全线达 llama.cpp Metal 93.7%–99.5%，CPU Decode 达 llama.cpp 102.6%–106.3%（单位 tok/s）。",
      baseName: "llama.cpp"
    },
    en: {
      label: "Apple M4 (Metal & CPU)",
      note: "Measured on Apple M4 16GB macOS: Metal decode hits 93.7%–99.5% of llama.cpp Metal; CPU decode hits 102.6%–106.3% (tok/s).",
      baseName: "llama.cpp"
    },
    rows: [
      { model: "SmolLM2-360M Q4_0", sub: "M4 CPU · tg128", tm: 270.5, base: 254.5, baseLabel: "llama.cpp", ratio: "106.3%" },
      { model: "Qwen2.5-0.5B Q4_0", sub: "M4 CPU · tg128", tm: 193.0, base: 188.1, baseLabel: "llama.cpp", ratio: "102.6%" },
      { model: "Qwen2.5-0.5B Q4_0", sub: "M4 Metal · tg128", tm: 185.6, base: 198.1, baseLabel: "llama.cpp Metal", ratio: "93.7%" },
      { model: "Qwen3.5-0.8B Q4_0 (GDN)", sub: "M4 Metal · tg128", tm: 129.0, base: 129.6, baseLabel: "llama.cpp Metal", ratio: "99.5%" },
      { model: "Llama-3.2-1B Q4_0", sub: "M4 Metal · tg128", tm: 108.8, base: 115.1, baseLabel: "llama.cpp Metal", ratio: "94.5%" },
      { model: "Gemma 4 E2B (PLE)", sub: "M4 Metal · tg128", tm: 52.3, base: 55.1, baseLabel: "llama.cpp Metal", ratio: "95.0%" }
    ]
  },
  pixel8: {
    zh: {
      label: "Google Pixel 8 (Tensor G3 / Android 16)",
      note: "Tensor G3 真机实测（1×X3 + 4×A715，i8mm + Mali-G715 Vulkan）：零拷贝 Q4_0x4 Prefill 达 llama.cpp --repack 0 的 155.5% 且省 810 MiB 内存（单位 tok/s）。",
      baseName: "llama.cpp"
    },
    en: {
      label: "Google Pixel 8 (Tensor G3 / Android 16)",
      note: "Real device (1×X3 + 4×A715, i8mm + Mali-G715 Vulkan): zero-copy Q4_0x4 prefill reaches 155.5% of llama.cpp --repack 0 while saving 810 MiB RAM (tok/s).",
      baseName: "llama.cpp"
    },
    rows: [
      { model: "Qwen2.5-0.5B Q4_0", sub: "Tensor G3 CPU · pp512", tm: 278.0, base: 245.0, baseLabel: "llama.cpp", ratio: "113.5%" },
      { model: "Gemma 4 E2B (PLE)", sub: "Tensor G3 CPU · pp512 (RSS 1.47G)", tm: 121.9, base: 103.3, baseLabel: "llama.cpp --repack 0", ratio: "118.0%" },
      { model: "SmolLM2-360M Q4_0x4", sub: "Tensor G3 CPU · tg64", tm: 103.95, base: 98.2, baseLabel: "llama.cpp KleidiAI", ratio: "105.9%" },
      { model: "Qwen3.5-0.8B Q4_0x4 (GDN)", sub: "Tensor G3 CPU · tg64", tm: 38.96, base: 36.8, baseLabel: "llama.cpp KleidiAI", ratio: "105.9%" },
      { model: "Llama-3.2-1B Q4_0x4 (Prefill)", sub: "Tensor G3 CPU · pp512 (-810 MiB)", tm: 93.3, base: 60.0, baseLabel: "llama.cpp --repack 0", ratio: "155.5%" }
    ]
  },
  micro: {
    zh: {
      label: "micro 档 & RK3588 边缘参考",
      note: "Pixel 8 绑 4×A510 顺序小核实测（纯 ARMv8.0 NEON，0 条 dotprod）与 Zero 2W / 32GB RK3588 带宽折算参考（Decode tok/s）。",
      baseName: "验收目标 / 参考"
    },
    en: {
      label: "micro Profile & RK3588 Edge Reference",
      note: "Measured on 4×A510 in-order cores (pure ARMv8.0 NEON, 0 dotprod) plus Zero 2W & 32GB RK3588 bandwidth projections (Decode tok/s).",
      baseName: "Target / Ref"
    },
    rows: [
      { model: "SmolLM2-360M (4×A510 实测)", sub: "RSS 96–200 MiB · 脏内存 9.5 MiB", tm: 20.92, base: 5.0, baseLabel: "micro 目标红线", ratio: "418%" },
      { model: "SmolLM2-360M (Zero 2W 折算)", sub: "512MB RAM · ARMv8.0 T-MAC LUT", tm: 9.8, base: 5.0, baseLabel: "Zero 2W 目标", ratio: "196%" },
      { model: "SmolLM2-360M (RK3588 预估)", sub: "4×A76 dotprod + Mali-G610 Vulkan", tm: 82.0, base: 62.0, baseLabel: "4×A76 CPU 预估", ratio: "132%" },
      { model: "Qwen3.6-35B-A3B Q4 (32GB RK3588)", sub: "内置 nextn MTP · 6–8.5W 边缘网关", tm: 15.0, base: 10.0, baseLabel: "纯单步解码", ratio: "~1.5× MTP" }
    ]
  }
};

export const MODELS = [
  {
    kicker: "10+ Dense Families",
    status: "CPU · Metal · Vulkan · CUDA · QLoRA",
    zh: {
      title: "Llama / Qwen2–3 / SmolLM3 / Phi-4 / Mistral / OLMo / Granite / EXAONE",
      desc: "全覆盖 HF 热门 ≤9B 稠密架构；支持 LongRoPE/YaRN、NoPE、双频 SWA、muP 缩放与 Q8_0 KV 量化（脏内存降 41%–45%）。",
      tags: ["Top-1=100%", "Q4_0x4", "F16/Q8 KV"]
    },
    en: {
      title: "Llama / Qwen2–3 / SmolLM3 / Phi-4 / Mistral / OLMo / Granite / EXAONE",
      desc: "Covers 10+ popular ≤9B dense families with LongRoPE/YaRN, NoPE, dual-theta SWA, muP scaling, and Q8_0 KV quantization (-45% dirty RAM).",
      tags: ["Top-1=100%", "Q4_0x4", "F16/Q8 KV"]
    }
  },
  {
    kicker: "GDN + 256-Exp MoE + MTP",
    status: "CPU · Metal · Vulkan · CUDA",
    zh: {
      title: "Qwen3.5 & Qwen3.6-35B-A3B",
      desc: "严守 GDN_CONTRACT 实现 0-ULP 寄存器批扫描与内置 nextn MTP 图捕获；Blackwell 达 239.4 tok/s，30.6G 卸载+64 槽缓存达 109.6 tok/s。",
      tags: ["239.4 tok/s", "verify_graphs", "64-Slot Cache"]
    },
    en: {
      title: "Qwen3.5 & Qwen3.6-35B-A3B",
      desc: "0-ULP register-resident GDN scan + graphed built-in nextn MTP: 239.4 tok/s on Blackwell, or 109.6 tok/s in 30.6 GiB via 64-slot VRAM cache.",
      tags: ["239.4 tok/s", "verify_graphs", "64-Slot Cache"]
    }
  },
  {
    kicker: "PLE + Shared KV + SWA",
    status: "CPU · Metal · Vulkan · QLoRA",
    zh: {
      title: "Gemma 2 / 3 / 4 (E2B / E4B)",
      desc: "262k 词表 per-layer embedding 保持 mmap 原位按需单行读取，立省 ~1.2GB 内存；完整支持滑动窗口、20 层共享 KV、softcap 与 GeGLU。",
      tags: ["PLE mmap", "Shared KV", "RSS 1.47 GiB"]
    },
    en: {
      title: "Gemma 2 / 3 / 4 (E2B / E4B)",
      desc: "Reads 262k per-layer embeddings row-by-row from mmap (~1.2GB saved); full sliding window, 20-layer shared KV, softcap, and GeGLU.",
      tags: ["PLE mmap", "Shared KV", "RSS 1.47 GiB"]
    }
  },
  {
    kicker: "ShortConv & Mamba-2 SSM",
    status: "CPU · Metal · Vulkan · QLoRA",
    zh: {
      title: "LFM2 / LFM3 & Mamba-2 混合家族",
      desc: "完整支持 LFM2/3 短卷积混合及 Mamba-2 / Nemotron-H / Falcon-H1 / GraniteHybrid 选择性状态空间扫描、前缀检查点与 BPTT 微调。",
      tags: ["ShortConv", "Mamba-2", "KL 8.2e-10"]
    },
    en: {
      title: "LFM2 / LFM3 & Mamba-2 Hybrid Families",
      desc: "Full support for LFM2/3 ShortConv and Mamba-2 / Nemotron-H / Falcon-H1 / GraniteHybrid selective state-space scan, prefix checkpoints, and QLoRA.",
      tags: ["ShortConv", "Mamba-2", "KL 8.2e-10"]
    }
  },
  {
    kicker: "26B Looped Sparse MoE",
    status: "CUDA · Vulkan · 3-Tier O_DIRECT",
    zh: {
      title: "Inkling-2-Mini + DSpark 分块投机",
      desc: "24 物理层展开 48 逻辑槽 3× 循环 MoE；Blackwell +DSpark 代码达 114.6 tok/s（反超 Strata +21.2%），亦支持 16G RAM + 8G GPU 零 Swap 运行。",
      tags: ["114.6 tok/s", "DSpark k=7", "Zero Swap"]
    },
    en: {
      title: "Inkling-2-Mini + DSpark Block Spec",
      desc: "24 physical layers unrolled to 48 logical slots; hits 114.6 tok/s on Blackwell (+21.2% over Strata) and runs zero-swap on 16G RAM + 8G GPU.",
      tags: ["114.6 tok/s", "DSpark k=7", "Zero Swap"]
    }
  },
  {
    kicker: "4-Stream HC + QSA + 512-Exp",
    status: "Zero-Pack 2-Shard · CUDA · CPU",
    zh: {
      title: "Qwen3.8-Flash-Next (qwen4exp + MTP)",
      desc: "免 44.9 GiB .pack 转换直读双分片 GSQ-RCO Q2_0 GGUF；支持 4 流超连接、QSA 块稀疏注意力、异构双门控 MoE 与外部 mtp-q2_0.gguf。",
      tags: ["pp512 1133.5", "Zero .pack", "qwen4exp-mtp"]
    },
    en: {
      title: "Qwen3.8-Flash-Next (qwen4exp + MTP)",
      desc: "Direct 2-shard GSQ-RCO Q2_0 GGUF loading saving 44.9 GiB disk; 4-stream HC, QSA block-sparse attention, heterogeneous dual-gate MoE & MTP.",
      tags: ["pp512 1133.5", "Zero .pack", "qwen4exp-mtp"]
    }
  },
  {
    kicker: "120B .strata + Absorbed MLA",
    status: "28.4× KV Compression · DFly2",
    zh: {
      title: "Tencent Hy4-Flash + 5 层 DFly2",
      desc: "原生流式加载 .strata 容器与 62G Engram 记忆表；512-CTA 矩阵吸收式 MLA 将 KV 显存压缩 28.4×（总显存比 Strata 省 4.5 GiB），配 S_K Tensor Core。",
      tags: ["Absorbed MLA", "DFly2 72%", "-4.5G VRAM"]
    },
    en: {
      title: "Tencent Hy4-Flash + 5-Layer DFly2",
      desc: "Native .strata & 62G Engram streaming; 512-CTA matrix-absorbed MLA shrinks KV cache 28.4× (-4.5 GiB vs Strata) with S_K Tensor Core prefill.",
      tags: ["Absorbed MLA", "DFly2 72%", "-4.5G VRAM"]
    }
  },
  {
    kicker: "Multimodal Vision",
    status: "CPU · Metal · Vulkan · Staged mmap",
    zh: {
      title: "通用多模态视觉编码器 (--mmproj)",
      desc: "零外部依赖支持 Gemma 3/4 SigLIP、Qwen2.5/3-VL 3D 时序卷积、Pixtral 与 SmolVLM；支持阶段式加载，编码完即释放 mmproj 内存。",
      tags: ["SigLIP", "Qwen-VL 3D", "Staged Free"]
    },
    en: {
      title: "Multimodal Vision Encoder (--mmproj)",
      desc: "Zero-dep GGUF mmproj ViT supporting Gemma 3/4 SigLIP, Qwen2.5/3-VL 3D conv + 2D RoPE, Pixtral & SmolVLM with staged memory release.",
      tags: ["SigLIP", "Qwen-VL 3D", "Staged Free"]
    }
  },
  {
    kicker: "On-Device Fine-Tuning",
    status: "tm-lora · tm-train (#![forbid(unsafe)])",
    zh: {
      title: "GGUF LoRA · 全架构 QLoRA · MTP 训练",
      desc: "兼容 llama.cpp --lora 动态挂载与无损合并；100% safe Rust 逐层激活重算 QLoRA（含 GDN / PLE / Mamba-2 / 稀疏 MoE）与 Metal/Vulkan 加速。",
      tags: ["Safe QLoRA", "MoE & GDN Bwd", "0-ULP Gate"]
    },
    en: {
      title: "GGUF LoRA · Full-Arch QLoRA · MTP Train",
      desc: "llama.cpp-compatible --lora & lossless merge; 100% safe-Rust recomputed QLoRA across Dense, GDN, PLE, Mamba-2 & MoE with GPU acceleration.",
      tags: ["Safe QLoRA", "MoE & GDN Bwd", "0-ULP Gate"]
    }
  }
];

export const ARCH_HIGHLIGHTS = {
  zh: [
    {
      title: "CUDA WMMA64 + CudaGraph 投机验证 + 热专家缓存",
      desc: "动态加载 libcuda 双 PTX，无 bank 冲突 64×64 WMMA 令预填充飙升 2.34×–7.40×；verify_graphs 消除多步验证发射开销，64 槽频率缓存令 32GB 卡百 tok/s 畅跑 35B Q8_0。"
    },
    {
      title: "28.4× 矩阵吸收式 MLA 与双分片零重排直读",
      desc: "Hy4 潜变量缓存 [cap, 576] 配合 8-token 共享权重 512-CTA 核立省 8.87 GiB KV 显存；Qwen3.8 直接映射原始双分片 GGUF，立省 44.9 GiB 磁盘与数十分钟打包。"
    },
    {
      title: "零膨胀离线重排 Q4_0x4 与 16 行 T-MAC 查表",
      desc: "tinymodel convert 离线写出 72B 交错超块（0 字节膨胀直连 mmap）；ARMv8.0 二补数拆分 vqtbl1q 内层零乘法，全四档 ISA 0-ULP 一致，TQ2_0 提速 11.3×。"
    },
    {
      title: "零 SDK 预编译 SPIR-V 1.2 与跨驱动双门控 MoE",
      desc: "tm-vulkan 内置校验 SPIR-V 覆盖 RDNA4 CoopMat、Intel SIMD16、Mali Subgroup 16 与 Pi 5 V3DV 无 Shuffle 退回，新增 moe_dual_gu_act 单遍读输入并省去中间缓冲。"
    }
  ],
  en: [
    {
      title: "CUDA WMMA64 + Graphed Spec Verify + Hot-Expert Cache",
      desc: "Conflict-free 64×64 WMMA boosts prefill 2.34×–7.40×; verify_graphs captures multi-token MTP verification, and 64-slot GPU aging cache hits 109.6 tok/s on 35B Q8_0 in <32GB VRAM."
    },
    {
      title: "28.4× Matrix-Absorbed MLA & Zero-Pack Multi-Shard GGUF",
      desc: "Latent [cap, 576] MLA KV with 512-CTA multi-token weight reuse saves 8.87 GiB VRAM on Hy4-Flash; direct 2-shard GGUF mmap saves 44.9 GiB disk on Qwen3.8."
    },
    {
      title: "Zero-Bloat Q4_0x4 & 16-Row T-MAC vqtbl1q LUT",
      desc: "Offline 72B superblocks avoid runtime RSS doubling; ARMv8.0 split-byte LUTs run with zero inner-loop multiplies and 0-ULP parity across all ISAs (11.3× faster TQ2_0)."
    },
    {
      title: "Zero-SDK Precompiled SPIR-V 1.2 & Dual-Gate Vulkan MoE",
      desc: "Ships verified SPIR-V spanning RDNA4 CoopMat, Intel SIMD16, Mali Subgroup 16, and Pi 5 V3DV shared-mem fallback, plus fused moe_dual_gu_act shaders."
    }
  ]
};

export const ROADMAP = [
  {
    stage: "M0 · 基线建设",
    stageEn: "M0 · Baselines",
    status: "wip",
    badgeZh: "M4 / RDNA4 / Blackwell 完成",
    badgeEn: "M4 / RDNA4 / Blackwell Done",
    zh: "建立 Apple M4、RX 9060 XT、Pixel 8 与 RTX PRO 6000 Blackwell 可复现同机基线协议，Pi 5 待复核。",
    en: "Reproducible same-machine baselines across M4, RX 9060 XT, Pixel 8, and RTX PRO 6000 Blackwell; Pi 5 pending."
  },
  {
    stage: "PoC · Rust 内核与真机验证",
    stageEn: "PoC · Rust Kernels & Device Bringup",
    status: "wip",
    badgeZh: "M4 + Pixel 8 通过",
    badgeEn: "M4 + Pixel 8 Passed",
    zh: "Rust stable Q4_0×Q8_0 dotprod/i8mm asm! GEMV、877KB 静态 musl 真机 L1 自检（90.9% 带宽利用率）、objc2-metal GEMV。",
    en: "Rust stable dotprod/i8mm asm! GEMV, 877KB static musl on Pixel 8 (90.9% BW util), and objc2-metal kernels."
  },
  {
    stage: "M1 · 稠密全谱系引擎",
    stageEn: "M1 · Dense Model Suite",
    status: "wip",
    badgeZh: "10+ 架构已完成",
    badgeEn: "10+ Families Done",
    zh: "全量支持 Llama / Qwen2–3 / SmolLM3 / Phi-3–4 / Mistral / OLMo 2–3 / Granite / EXAONE、BPE/SPM 分词与采样器。",
    en: "Full stack for Llama / Qwen2–3 / SmolLM3 / Phi-3–4 / Mistral / OLMo 2–3 / Granite / EXAONE, tokenizers & samplers."
  },
  {
    stage: "M2 · 新一代混合结构 + Metal",
    stageEn: "M2 · Hybrid Models + Metal",
    status: "done",
    badgeZh: "已完成",
    badgeEn: "Completed",
    zh: "全量支持 Qwen3.5 (GDN)、Gemma 2/3/4 (PLE/SWA/Shared KV)、LFM2/3、LRU 前缀缓存与 Metal 后端（≥94.5% llama.cpp）。",
    en: "Full Qwen3.5 (GDN), Gemma 2/3/4 (PLE/SWA/Shared KV), LFM2/3, LRU prefix cache, and Metal backend (≥94.5% llama.cpp)."
  },
  {
    stage: "M3 · 极致性能 + Vulkan 后端",
    stageEn: "M3 · Performance + Vulkan",
    status: "done",
    badgeZh: "已完成",
    badgeEn: "Completed",
    zh: "离线 Q4_0x4、Q8_0 KV、自适应投机解码，以及 Vulkan CoopMat + 双门控 MoE（RX 9060 XT Decode 达 ROCm HIP 226%）。",
    en: "Offline Q4_0x4, Q8_0 KV, adaptive speculation, and Vulkan CoopMat + Dual-Gate MoE (up to 226% of ROCm HIP)."
  },
  {
    stage: "M4 · micro 极低功耗档",
    stageEn: "M4 · micro Profile",
    status: "done",
    badgeZh: "已完成",
    badgeEn: "Completed",
    zh: "Q2_0/TQ2_0/IQ2/IQ3/IQ4_NL 低比特格式、16 行 T-MAC 查表内核与 BPE 闭包词表裁剪；静态 ELF 仅 1.18 MiB。",
    en: "Q2_0/TQ2_0/IQ2/IQ3/IQ4_NL low-bit formats, 16-row T-MAC LUT kernels, BPE vocab pruning; 1.18 MiB static ELF."
  },
  {
    stage: "T1–T4 · 端侧微调与 LoRA 闭环",
    stageEn: "T1–T4 · On-Device LoRA & QLoRA",
    status: "done",
    badgeZh: "已完成",
    badgeEn: "Completed",
    zh: "GGUF LoRA 动态挂载/合并、MTP 草稿头训练、覆盖稠密/GDN/PLE/Mamba-2/MoE 的 safe Rust QLoRA 与 GPU 反向加速。",
    en: "GGUF LoRA runtime/merge, MTP training, safe-Rust QLoRA across Dense/GDN/PLE/Mamba-2/MoE, Metal/Vulkan backward accel."
  },
  {
    stage: "M5 · CUDA 后端与服务生态",
    stageEn: "M5 · CUDA Backend & Server",
    status: "wip",
    badgeZh: "CUDA + P0–P2 已完成 ✅",
    badgeEn: "CUDA + P0–P2 Done ✅",
    zh: "已完成 tm-cuda（WMMA64、verify_graphs 图捕获、异构双门控、DDR 卸载+64 槽缓存）；下一步推进 OpenAI HTTP 服务与 FFI/Python 绑定。",
    en: "Completed tm-cuda (WMMA64, graphed MTP verify, dual-gate MoE, DDR offload + 64-slot cache); HTTP server & FFI/Py bindings next."
  },
  {
    stage: "M6 · 多模态、Mamba-2 与前沿 MoE",
    stageEn: "M6 · Vision, Mamba-2 & Frontier MoE",
    status: "wip",
    badgeZh: "视觉 + 四大 MoE 已完成 ✅",
    badgeEn: "Vision + 4 MoEs Done ✅",
    zh: "已落地多模态 ViT (mmproj)、Mamba-2、9 种稀疏 MoE 及 Strata 四大家族（Inkling-2 / Qwen3.6 / Qwen3.8 / Hy4-Flash）；后续探索 NPU。",
    en: "Shipped multimodal ViT, Mamba-2, 9 sparse MoEs & all 4 Strata families (Inkling-2 / Qwen3.6 / Qwen3.8 / Hy4-Flash); NPU next."
  }
];
