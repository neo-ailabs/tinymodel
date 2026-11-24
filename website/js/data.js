export const I18N = {
  zh: {
    "nav.profiles": "硬件分档",
    "nav.benchmarks": "实测基准",
    "nav.models": "模型与微调",
    "nav.architecture": "架构设计",
    "nav.quickstart": "快速开始",
    "nav.roadmap": "路线图",
    "hero.eyebrow": "Rust Stable · Zero-Alloc Decode · MIT / Apache-2.0",
    "hero.title_1": "从 512MB 树莓派到桌面 GPU 的",
    "hero.title_highlight": "纯 Rust 端侧 LLM 引擎",
    "hero.sub": "tinymodel 用同一份 Rust stable 代码库覆盖 Raspberry Pi Zero 2W（≤0.5B）、手机与 ARM 开发板（≤5B）及主机 GPU（≤8B 及 26B 循环 MoE）。直接 mmap 加载 GGUF，一等支持 Qwen3.5 GDN、Gemma 4 PLE、LFM2 与端侧 QLoRA 微调。",
    "hero.cta_start": "快速生成命令",
    "hero.cta_bench": "查看真机实测",
    "hero.cta_repo": "GitHub 仓库",
    "kpi.1.label": "micro 纯静态二进制",
    "kpi.1.desc": "aarch64-musl stripped ELF，0 条 sdot/smmla，512MB Zero 2W 稳跑",
    "kpi.2.label": "数值对齐门禁",
    "kpi.2.desc": "SIMD 与标量参考实现逐位一致，投机解码与前缀缓存 100% 确定性",
    "kpi.3.label": "RX 9060 XT Vulkan 加速",
    "kpi.3.desc": "Decode 反超 ROCm 7.1.1 HIP 最高 2.26×，零 SDK 依赖预编译 SPIR-V",
    "kpi.4.label": "Pixel 8 跑 Gemma 4 E2B",
    "kpi.4.desc": "PLE 按需单行 mmap，总 RSS 仅 1.47 GiB，私有脏内存低至 55.6 MiB",
    "profiles.tag": "Hardware Profiles",
    "profiles.title": "一份代码，按三档硬件精准裁剪",
    "profiles.desc": "通过 Cargo feature 与运行时 ISA 自动选择（NEON / dotprod / i8mm / AVX2 + Metal / Vulkan），兼顾极低功耗设备、移动终端与桌面独显。",
    "bench.tag": "Verified Benchmarks",
    "bench.title": "真机实测性能与内存基线",
    "bench.desc": "所有数据均来自仓库内可复现基准协议（与同一台机器上的 llama.cpp / ROCm HIP 对照实测）。",
    "models.tag": "First-Class Architectures",
    "models.title": "一等支持新一代混合小模型与端侧微调",
    "models.desc": "不走动态计算图，每个模型结构手写零分配 forward 与独立激活重算 QLoRA 反向。",
    "arch.tag": "System Architecture",
    "arch.title": "为端侧带宽与内存红线而生的分层设计",
    "arch.desc": "严格隔离 unsafe 边界（仅限内核、GPU 后端、FFI 与 GGUF mmap），在生成 token 热路径上实现零堆分配。",
    "cli.tag": "Interactive Quickstart",
    "cli.title": "交互式 CLI 命令生成器",
    "cli.desc": "按你的任务场景、硬件后端与量化策略，即时生成可直接复制执行的 tinymodel 命令。",
    "cli.mode": "子命令场景",
    "cli.backend": "计算后端",
    "cli.model": "示例模型",
    "cli.opt": "优化特性",
    "roadmap.tag": "Project Roadmap",
    "roadmap.title": "开发进度与里程碑",
    "roadmap.desc": "以可量化的吞吐、KL 散度与二进制体积作为每阶段验收门禁。",
    "copy": "复制",
    "copied": "已复制到剪贴板",
    "footer.license": "采用 MIT OR Apache-2.0 双许可开源",
    "footer.docs": "架构文档",
    "footer.decisions": "ADR 决策记录",
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
    "hero.eyebrow": "Rust Stable · Zero-Alloc Decode · MIT / Apache-2.0",
    "hero.title_1": "Edge-First LLM Inference Engine in",
    "hero.title_highlight": "Pure Rust Stable",
    "hero.sub": "One codebase spanning 512MB Raspberry Pi Zero 2W (≤0.5B), phones & ARM SBCs (≤5B), and desktop GPUs (≤8B & 26B looped MoE). Zero-copy GGUF mmap, first-class Qwen3.5 GDN, Gemma 4 PLE, LFM2, and on-device QLoRA.",
    "hero.cta_start": "Command Builder",
    "hero.cta_bench": "View Benchmarks",
    "hero.cta_repo": "GitHub Repo",
    "kpi.1.label": "micro Static Binary",
    "kpi.1.desc": "aarch64-musl stripped ELF, 0 sdot/smmla insns, fits 512MB Pi Zero 2W",
    "kpi.2.label": "Bit-Exact Parity",
    "kpi.2.desc": "SIMD kernels match scalar reference bit-for-bit; deterministic decode",
    "kpi.3.label": "RX 9060 XT Vulkan",
    "kpi.3.desc": "Up to 2.26× faster decode than ROCm 7.1.1 HIP via precompiled SPIR-V",
    "kpi.4.label": "Gemma 4 E2B on Pixel 8",
    "kpi.4.desc": "On-demand PLE row mmap: 1.47 GiB total RSS, only 55.6 MiB dirty RAM",
    "profiles.tag": "Hardware Profiles",
    "profiles.title": "One Codebase, Three Hardware Profiles",
    "profiles.desc": "Cargo features plus runtime ISA dispatch (NEON / dotprod / i8mm / AVX2 + Metal / Vulkan) tailored from ultra-low-power SBCs to desktop GPUs.",
    "bench.tag": "Verified Benchmarks",
    "bench.title": "Real-Device Performance & Memory Baselines",
    "bench.desc": "Measured on real hardware against llama.cpp and ROCm HIP on the exact same machine under the repository benchmark protocol.",
    "models.tag": "First-Class Architectures",
    "models.title": "Next-Gen Hybrid Models & On-Device QLoRA",
    "models.desc": "Hand-written zero-allocation forward passes and recomputed QLoRA backward passes for every supported architecture.",
    "arch.tag": "System Architecture",
    "arch.title": "Engineered for Bandwidth & Strict Memory Budgets",
    "arch.desc": "Unsafe code strictly isolated to kernel, GPU, FFI, and mmap crates. Zero heap allocations on the token generation hot path.",
    "cli.tag": "Interactive Quickstart",
    "cli.title": "Interactive CLI Command Builder",
    "cli.desc": "Pick your workload, backend, model, and runtime optimizations to generate copy-ready tinymodel commands.",
    "cli.mode": "Subcommand",
    "cli.backend": "Backend",
    "cli.model": "Sample Model",
    "cli.opt": "Optimization",
    "roadmap.tag": "Project Roadmap",
    "roadmap.title": "Milestones & Delivery Status",
    "roadmap.desc": "Every milestone is gated by quantitative throughput, multi-token KL divergence, and binary-size checks.",
    "copy": "Copy",
    "copied": "Copied to clipboard",
    "footer.license": "Dual-licensed under MIT OR Apache-2.0",
    "footer.docs": "Architecture",
    "footer.decisions": "ADRs",
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
        ["静态体积", "1.18 – 1.47 MiB stripped ELF"]
      ],
      devices: ["Pi Zero 2W", "Pi 3 / 4B", "Cortex-A53", "4×A510 实测 20.9 tok/s"]
    },
    en: {
      title: "Ultra-Low-Power Micro Devices",
      specs: [
        ["Memory Budget", "512 MB (~350 MB usable)"],
        ["CPU ISA", "ARMv8.0 NEON (no dotprod / FP16)"],
        ["Key Tech", "16-row T-MAC vqtbl1q LUT + Vocab Pruning"],
        ["Static Binary", "1.18 – 1.47 MiB stripped ELF"]
      ],
      devices: ["Pi Zero 2W", "Pi 3 / 4B", "Cortex-A53", "4×A510: 20.9 tok/s"]
    },
    cmd: "cargo zigbuild --release --target aarch64-unknown-linux-musl --no-default-features --features micro"
  },
  {
    id: "edge",
    badge: "profile: edge",
    scale: "≤ 5B Params",
    featured: true,
    zh: {
      title: "手机与高性能 ARM 开发板",
      specs: [
        ["目标内存", "4 – 16 GB（RK3588 可达 32 GB）"],
        ["CPU / GPU", "NEON dotprod / i8mm + Mali / V3DV Vulkan"],
        ["核心技术", "零膨胀 Q4_0x4 重排 + 异构大核绑核 + Q8 KV"],
        ["真机实测", "Pixel 8 带宽利用率 90.9%，脏内存 <56 MiB"]
      ],
      devices: ["Google Pixel 8 (Tensor G3)", "Raspberry Pi 5", "Rockchip RK3588", "Mali-G715 / G610"]
    },
    en: {
      title: "Smartphones & ARM Edge SBCs",
      specs: [
        ["Memory Budget", "4 – 16 GB (up to 32 GB on RK3588)"],
        ["CPU / GPU", "NEON dotprod / i8mm + Mali / V3DV Vulkan"],
        ["Key Tech", "Zero-bloat Q4_0x4 + Big-core affinity + Q8 KV"],
        ["Real Device", "Pixel 8 90.9% BW util, <56 MiB dirty RAM"]
      ],
      devices: ["Google Pixel 8 (Tensor G3)", "Raspberry Pi 5", "Rockchip RK3588", "Mali-G715 / G610"]
    },
    cmd: "cargo zigbuild --release --target aarch64-unknown-linux-musl --features edge"
  },
  {
    id: "host",
    badge: "profile: host",
    scale: "≤ 8B & 26B MoE",
    featured: false,
    zh: {
      title: "Apple Silicon 与桌面独显主机",
      specs: [
        ["目标内存", "16 GB+（支持 3 层 O_DIRECT 异构调度）"],
        ["计算后端", "Metal / Vulkan CoopMat / AVX2 / i8mm"],
        ["核心技术", "simdgroup_float8x8 + 预编译 SPIR-V 1.2"],
        ["扩展能力", "GPU 动态 LoRA、QLoRA 训练、DSpark 投机"]
      ],
      devices: ["Apple M4 (Metal)", "AMD RX 9060 XT", "Intel UHD 770", "x86_64 AVX2"]
    },
    en: {
      title: "Apple Silicon & Desktop GPUs",
      specs: [
        ["Memory Budget", "16 GB+ (3-tier O_DIRECT MoE offloading)"],
        ["Backends", "Metal / Vulkan CoopMat / AVX2 / i8mm"],
        ["Key Tech", "simdgroup_float8x8 + Precompiled SPIR-V 1.2"],
        ["Capabilities", "GPU dynamic LoRA, QLoRA training, DSpark"]
      ],
      devices: ["Apple M4 (Metal)", "AMD RX 9060 XT", "Intel UHD 770", "x86_64 AVX2"]
    },
    cmd: "cargo build --release --features metal,train"
  }
];

export const BENCHMARKS = {
  vulkan: {
    zh: {
      label: "AMD RX 9060 XT & Intel UHD 770 (Vulkan)",
      note: "Mesa 25.2.8 Vulkan 1.2 预编译 SPIR-V + 16×16×16 CoopMat，对比同机 ROCm 7.1.1 HIP 与 llama.cpp Vulkan（Decode tg128，单位 tok/s）。",
      baseName: "ROCm HIP / llama.cpp"
    },
    en: {
      label: "AMD RX 9060 XT & Intel UHD 770 (Vulkan)",
      note: "Precompiled SPIR-V 1.2 + 16×16×16 CoopMat on Mesa 25.2.8 vs ROCm 7.1.1 HIP & llama.cpp Vulkan (Decode tg128, tok/s).",
      baseName: "ROCm HIP / llama.cpp"
    },
    rows: [
      { model: "SmolLM2-360M Q4_0", sub: "RX 9060 XT · tg128", tm: 608.8, base: 269.5, baseLabel: "ROCm HIP", ratio: "225.9%" },
      { model: "Qwen3.5-0.8B Q4_0 (GDN)", sub: "RX 9060 XT · tg128", tm: 366.5, base: 168.3, baseLabel: "ROCm HIP", ratio: "217.8%" },
      { model: "Qwen3.5-2B Q4_0 (GDN)", sub: "RX 9060 XT · tg128", tm: 184.5, base: 134.3, baseLabel: "ROCm HIP", ratio: "137.4%" },
      { model: "Qwen3-8B Q4_0", sub: "RX 9060 XT · tg128", tm: 64.0, base: 59.3, baseLabel: "ROCm HIP", ratio: "108.0%" },
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
      note: "Tensor G3 真机实测（1×X3 + 4×A715，i8mm）：零拷贝 Q4_0x4 Prefill 达 llama.cpp --repack 0 的 155.5% 且省 810 MiB 内存（单位 tok/s）。",
      baseName: "llama.cpp"
    },
    en: {
      label: "Google Pixel 8 (Tensor G3 / Android 16)",
      note: "Real device (1×X3 + 4×A715, i8mm): zero-copy Q4_0x4 prefill reaches 155.5% of llama.cpp --repack 0 while saving 810 MiB RAM (tok/s).",
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
      note: "Pixel 8 绑 4×A510 顺序小核实测（纯 ARMv8.0 NEON，0 条 dotprod）与 Zero 2W / RK3588 带宽折算参考（Decode tok/s）。",
      baseName: "验收目标 / 参考"
    },
    en: {
      label: "micro Profile & RK3588 Edge Reference",
      note: "Measured on 4×A510 in-order cores (pure ARMv8.0 NEON, 0 dotprod) plus Zero 2W & RK3588 bandwidth projections (Decode tok/s).",
      baseName: "Target / Ref"
    },
    rows: [
      { model: "SmolLM2-360M (4×A510 实测)", sub: "RSS 96–200 MiB · 脏内存 9.5 MiB", tm: 20.92, base: 5.0, baseLabel: "micro 目标红线", ratio: "418%" },
      { model: "SmolLM2-360M (Zero 2W 折算)", sub: "512MB RAM · ARMv8.0 T-MAC LUT", tm: 9.8, base: 5.0, baseLabel: "Zero 2W 目标", ratio: "196%" },
      { model: "SmolLM2-360M (RK3588 预估)", sub: "4×A76 dotprod + Mali Vulkan", tm: 82.0, base: 62.0, baseLabel: "4×A76 CPU 预估", ratio: "132%" },
      { model: "Qwen3.5-0.8B (RK3588 预估)", sub: "4×A76 dotprod + Mali Vulkan", tm: 35.0, base: 27.5, baseLabel: "4×A76 CPU 预估", ratio: "127%" }
    ]
  }
};

export const MODELS = [
  {
    kicker: "GQA / Transformer",
    status: "CPU · Metal · Vulkan · QLoRA",
    zh: {
      title: "Llama / Qwen2 / Qwen2.5 / Qwen3 / SmolLM2",
      desc: "覆盖主流稠密模型与 DeepSeek-R1-Distill-Qwen；支持 YaRN 长上下文缩放、attention bias 与 Q8_0 KV 量化（脏内存降 41%–45%）。",
      tags: ["Top-1=100%", "Q4_0x4", "F16/Q8 KV"]
    },
    en: {
      title: "Llama / Qwen2 / Qwen2.5 / Qwen3 / SmolLM2",
      desc: "Covers mainstream dense models & DeepSeek-R1-Distill-Qwen with YaRN RoPE scaling, attention bias, and Q8_0 KV quantization (-41% to -45% dirty RAM).",
      tags: ["Top-1=100%", "Q4_0x4", "F16/Q8 KV"]
    }
  },
  {
    kicker: "3:1 Hybrid Linear Attn",
    status: "CPU · Metal · Vulkan · QLoRA",
    zh: {
      title: "Qwen3.5 (Gated DeltaNet)",
      desc: "严格遵循 GDN_CONTRACT 实现分块扫描、单步递推与短卷积；支持递归检查点前缀缓存与 O(1) 边界重算 QLoRA 反向。",
      tags: ["GDN", "Prefix Checkpoint", "0.8B–9B"]
    },
    en: {
      title: "Qwen3.5 (Gated DeltaNet)",
      desc: "Chunked scan, recurrent decode, and short-conv adhering strictly to GDN_CONTRACT; supports checkpointed prefix cache and O(1) recomputed QLoRA.",
      tags: ["GDN", "Prefix Checkpoint", "0.8B–9B"]
    }
  },
  {
    kicker: "PLE + Shared KV + SWA",
    status: "CPU · Metal · Vulkan · QLoRA",
    zh: {
      title: "Gemma 4 (E2B / E4B)",
      desc: "262k 词表 per-layer embedding 保持 mmap 原位按需单行读取，立省 ~1.2GB 内存；完整支持 4:1 滑动窗口、20 层共享 KV 与 GeGLU。",
      tags: ["PLE mmap", "Shared KV", "RSS 1.47 GiB"]
    },
    en: {
      title: "Gemma 4 (E2B / E4B)",
      desc: "Reads 262k per-layer embeddings row-by-row from mmap (~1.2GB saved); full 4:1 sliding window, 20-layer shared KV, softcap, and GeGLU.",
      tags: ["PLE mmap", "Shared KV", "RSS 1.47 GiB"]
    }
  },
  {
    kicker: "ShortConv Hybrid",
    status: "CPU · Metal · Vulkan · QLoRA",
    zh: {
      title: "LFM2 / LFM2.5",
      desc: "门控因果短卷积与 QK-Norm GQA 混合架构，61 步多 token 对齐 Metal mean_KL 低至 8.23e-10，非常适合 micro/edge 档。",
      tags: ["ShortConv", "KL 8.2e-10", "350M–2.6B"]
    },
    en: {
      title: "LFM2 / LFM2.5",
      desc: "Gated causal ShortConv + QK-Norm GQA hybrid achieving 8.23e-10 mean KL over 61 steps on Metal; ideal for micro and edge tiers.",
      tags: ["ShortConv", "KL 8.2e-10", "350M–2.6B"]
    }
  },
  {
    kicker: "26B Looped Sparse MoE",
    status: "ADR-016 · 3-Tier O_DIRECT",
    zh: {
      title: "Inkling-2-Mini + DSpark",
      desc: "24 物理层展开 48 逻辑槽 3× 循环 128 选 8 MoE；通过 VRAM + 匿名 RAM + NVMe O_DIRECT 在 16GB RAM + 8GB GPU 零 Swap 全量运行 Q8_0。",
      tags: ["25.91 GiB Q8_0", "DSpark 83%", "Zero Swap"]
    },
    en: {
      title: "Inkling-2-Mini + DSpark",
      desc: "24 physical layers unrolled to 48 logical slots (3× looped 8-of-128 MoE); runs 25.91 GiB Q8_0 with zero swap on 16GB RAM + 8GB GPU via O_DIRECT.",
      tags: ["25.91 GiB Q8_0", "DSpark 83%", "Zero Swap"]
    }
  },
  {
    kicker: "On-Device Fine-Tuning",
    status: "tm-lora · tm-train",
    zh: {
      title: "GGUF LoRA · QLoRA · MTP 草稿头",
      desc: "兼容 llama.cpp --lora 动态挂载与无损合并；100% safe Rust 逐层激活重算 QLoRA，支持 Metal/Vulkan 反向加速（M4 提速 1.43×–2.19×）。",
      tags: ["QLoRA", "MTP Spec", "0-ULP Gate"]
    },
    en: {
      title: "GGUF LoRA · QLoRA · MTP Draft Heads",
      desc: "Compatible with llama.cpp --lora adapters and lossless merge; safe-Rust recomputed QLoRA with Metal/Vulkan backward accel (1.43×–2.19× on M4).",
      tags: ["QLoRA", "MTP Spec", "0-ULP Gate"]
    }
  }
];

export const ARCH_HIGHLIGHTS = {
  zh: [
    {
      title: "零膨胀离线重排 Q4_0x4",
      desc: "与 llama.cpp 运行时重排导致 RSS 翻倍不同，tinymodel convert 离线生成 72B 超块交错布局，文件体积 0 膨胀且直接 mmap 进 i8mm / dotprod / Vulkan。"
    },
    {
      title: "16 行 T-MAC vqtbl1q 查表内核",
      desc: "面向 ARMv8.0 的二补数高低字节拆分激活查表，内层权重循环 0 条乘法指令，全四档 ISA 0-ULP 逐位一致，TQ2_0 三值内核提速 11.3×。"
    },
    {
      title: "统一 LayerState 与只读投机验证",
      desc: "单枚举覆盖 KV、SharedKv、DeltaNet 与 ShortConv；验证窗口只读不污染状态，接受后精确重放提交，保证开启投机解码贪心输出逐位不变。"
    },
    {
      title: "零 SDK 依赖的预编译 SPIR-V 1.2",
      desc: "tm-vulkan 直接打包经 spirv-val 校验的着色器，交叉编译至 Linux musl / Android 无需安装 Vulkan SDK 或 shaderc。"
    }
  ],
  en: [
    {
      title: "Zero-Bloat Offline Q4_0x4 Repack",
      desc: "Unlike runtime repacking that doubles RSS, tinymodel convert writes 72B interleaved superblocks offline with zero file growth for direct mmap into i8mm/dotprod/Vulkan."
    },
    {
      title: "16-Row T-MAC vqtbl1q LUT Kernels",
      desc: "Two's-complement split-byte activation LUTs for ARMv8.0 with zero multiply instructions in the inner loop, 0-ULP parity across all ISAs, and 11.3× speedup on TQ2_0."
    },
    {
      title: "Unified LayerState & Read-Only Spec Verify",
      desc: "One enum for KV, SharedKv, DeltaNet, and ShortConv. Speculative verification is read-only with exact commit replay, preserving bit-exact greedy outputs."
    },
    {
      title: "Zero-SDK Precompiled SPIR-V 1.2",
      desc: "tm-vulkan ships spirv-val verified shaders so cross-compiling to Linux musl or Android requires no Vulkan SDK or shaderc."
    }
  ]
};

export const ROADMAP = [
  {
    stage: "M0 · 基线建设",
    stageEn: "M0 · Baselines",
    status: "wip",
    badgeZh: "M4 完成 / Pi 5 待测",
    badgeEn: "M4 Done / Pi 5 WIP",
    zh: "建立 Apple M4 与端侧设备可复现的 llama.cpp 基准协议（pp/tg/RSS/温度/频率），存入 bench/baselines/。",
    en: "Reproducible llama.cpp baseline tables (pp/tg/RSS/thermal/freq) archived under bench/baselines/."
  },
  {
    stage: "PoC · Rust 内核与真机验证",
    stageEn: "PoC · Rust Kernels & Device Bringup",
    status: "wip",
    badgeZh: "M4 + Pixel 8 通过",
    badgeEn: "M4 + Pixel 8 Passed",
    zh: "Rust stable Q4_0×Q8_0 dotprod/i8mm asm! GEMV、877KB 静态 musl 真机 L1 自检、objc2-metal 量化 GEMV。",
    en: "Rust stable dotprod/i8mm asm! GEMV, 877KB static musl on Pixel 8 (90.9% BW util), and objc2-metal kernels."
  },
  {
    stage: "M1 · 最小可用引擎",
    stageEn: "M1 · MVP Engine",
    status: "wip",
    badgeZh: "M4 + Pixel 8 完成",
    badgeEn: "M4 + Pixel 8 Done",
    zh: "Cargo workspace、GGUF 校验、CPU 线程池、Llama / Qwen2 / Qwen2.5 / Qwen3 全栈支持与采样器，贪心逐 token 对齐 llama.cpp。",
    en: "Workspace, GGUF validator, thread pool, Llama / Qwen2 / Qwen2.5 / Qwen3 full stack, bit-aligned greedy decode."
  },
  {
    stage: "M2 · 新一代混合结构 + Metal",
    stageEn: "M2 · Hybrid Models + Metal",
    status: "done",
    badgeZh: "已完成",
    badgeEn: "Completed",
    zh: "全量支持 Qwen3.5 (GDN)、Gemma 4 (PLE/SWA/Shared KV)、LFM2、LRU 前缀缓存与 Metal 后端（全面 ≥94.5% llama.cpp Metal）。",
    en: "Full Qwen3.5 (GDN), Gemma 4 (PLE/SWA/Shared KV), LFM2, LRU prefix cache, and Metal backend (≥94.5% llama.cpp)."
  },
  {
    stage: "M3 · 极致性能 + Vulkan 后端",
    stageEn: "M3 · Performance + Vulkan",
    status: "done",
    badgeZh: "已完成",
    badgeEn: "Completed",
    zh: "离线 Q4_0x4、Q8_0 KV、Suffix/MTP 自适应投机解码，以及 Vulkan CoopMat 后端（RX 9060 XT Decode 达 ROCm HIP 108%–226%）。",
    en: "Offline Q4_0x4, Q8_0 KV, Suffix/MTP adaptive speculation, and Vulkan CoopMat (108%–226% of ROCm HIP on RX 9060 XT)."
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
    zh: "GGUF LoRA 动态挂载/合并、MTP 草稿头训练、全 5 架构手写激活重算 QLoRA 及 Metal/Vulkan GPU 反向加速。",
    en: "GGUF LoRA runtime/merge, MTP head training, recomputed QLoRA across all 5 architectures, Metal/Vulkan backward accel."
  },
  {
    stage: "M5 · 服务层、绑定与 CUDA",
    stageEn: "M5 · Server, Bindings & CUDA",
    status: "todo",
    badgeZh: "下一阶段",
    badgeEn: "Up Next",
    zh: "OpenAI 兼容 HTTP 服务（流式、Tool Calling、模板）、C ABI、Python 绑定、分页 KV 与 CUDA 后端。",
    en: "OpenAI-compatible HTTP server (streaming, tool calling, templates), C ABI, Python bindings, paged KV, and CUDA."
  },
  {
    stage: "M6 · 扩展与前沿架构",
    stageEn: "M6 · Extensions & Frontier MoE",
    status: "wip",
    badgeZh: "Inkling-2 26B 已合入",
    badgeEn: "Inkling-2 26B Merged",
    zh: "已合入 Inkling-2-Mini 26B 循环 MoE + 三层 O_DIRECT + DSpark 与 LEGO DRC 微调管线；后续探索多模态与 NPU。",
    en: "Merged Inkling-2-Mini 26B looped MoE + 3-tier O_DIRECT + DSpark and LEGO DRC pipeline; multimodal & NPU planned."
  }
];
