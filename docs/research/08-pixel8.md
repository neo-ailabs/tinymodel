# Google Pixel 8（Tensor G3）设备档案与真机实测报告

> 更新日期：2026-11-20（**已完成真机实测**，序列号 `3B131FDJH00216`，Android 16）  
> 固定参考设备之一（见 [AGENTS.md](../../AGENTS.md) §8）。原始基准结果存档于 [2026-11-20-a80228f.json](../../bench/results/pixel-8/2026-11-20-a80228f.json)。

---

## 1. 真机探测硬件规格（`tinymodel probe` 实测确认）

| 项 | Pixel 8 真机实测值 | 对比 Pi 5 / RK3588 |
|---|---|---|
| **SoC / OS** | **Google Tensor G3**（三星 4nm），Android 16（`sched_pixel` 调频器） | Pi 5：BCM2712 (16nm)；RK3588：8nm |
| **三丛集异构 CPU** | - `cpu0..3`：**4× Cortex-A510 @ 1704 MHz**（掩码 `0x0f`）<br>- `cpu4..7`：**4× Cortex-A715 @ 2367 MHz**（掩码 `0xf0`）<br>- `cpu8`：**1× Cortex-X3 @ 2914 MHz**（掩码 `0x100`） | Pi 5：4×A76 同构；RK3588：4×A76 + 4×A55 |
| **ISA 与 L1 自检** | 自动命中 **`Isa::NeonI8mm`**（`neon`、`dotprod`、`i8mm` 全支持；真机 L1 逐位自检 **PASSED**） | Pi 5 / RK3588 仅到 `dotprod`（无 `i8mm`） |
| **物理内存与可用水位** | 总计 `7.39 GiB`（`7,941,091,328 B`），空闲可用 **`3.81 GiB`** | 足够无压力常驻 `Qwen3.5-2B`（峰值 RSS `1.17 GiB`） |
| **实测 STREAM 持续带宽** | - **仅绑 `4×A715`（`0xf0`, 4T）**：**Read `31.17 GB/s` / Copy `40.34 GB/s`**<br>- 绑 `X3 + 4×A715`（`0x1f0`, 5T）：Read `21.84 GB/s` / Copy `25.37 GB/s`<br>- 单核 `1×X3`（`0x100`, 1T）：Read `19.19 GB/s`<br>- 全核混跑（8T，含 A510）：Read `15.61 GB/s`（**被小核拖慢一半！**） | Pi 5 约 `11.5 GB/s`；RK3588 约 `17–23 GB/s` |
| **GPU / NPU** | Arm Mali-G715 MC7（Vulkan 1.3，M3 接入）；TPU 仅限系统 AICore 使用 | — |

---

## 2. 异构大小核调度实测与已落地的动态分块优化

在真机上以 `Qwen3.5-0.8B-Q4_0`（`-n 64 -c 1024`）实测四种绑核策略及 [ops.rs](../../crates/tm-cpu/src/ops.rs) 动态细粒度分块（`dynamic_row_chunk`）改造前后的对比：

| 绑核掩码与线程配置 | 物理核心组合 | Prefill `pp512` | Decode `tg64`（优化前均分） | **Decode `tg64`（动态分块后）** | 结论 |
|---|---|---:|---:|---:|---|
| **`0x1f0` (`-t 5`, `no-little`)** | **1×X3 + 4×A715** | **70.9 – 71.3 tok/s** | 36.37 tok/s | **`37.89 tok/s`（冷机峰值）** | **最高综合吞吐（默认推荐）** |
| **`0xf0` (`-t 4`, `mid-only`)** | **4×A715 同构中核** | **71.4 – 71.7 tok/s** | 28.75 tok/s | **`28.75 tok/s`** | **最高能效档**（不开高功耗 X3 仍保住 76% 速度） |
| `0x100` (`-t 1`) | 仅 1×X3 超大核 | 36.4 tok/s | 13.49 tok/s | 13.49 tok/s | 单核带宽达 `~6.7 GB/s` |
| `0x1ff` (`-t 9`, `all`) | 全 9 核（含 4×A510） | 38.3 → **45.5 tok/s** | 8.76 tok/s | **`13.34 tok/s`（+52.3%）** | **严禁把 A510 放进推理线程池**（即使动态分块仍受 DSU 争抢拖累） |

### 已完成的代码适配清单
- ✅ **异构自适应分块**（[ops.rs](../../crates/tm-cpu/src/ops.rs)）：`n_thr <= 4` 保持每核单段连续预取（M4 / Pi 5 / `4×A715` 零回退）；`n_thr > 4` 自动切分为 `4×` 细粒度块 + 单头 GQA/GDN 原子抢占，Pixel 8 5 线程再提速 **+4.2%**、未绑核 9 线程提速 **+52.3%**。
- ✅ **Android 内存探测与峰值 RSS**（[memprobe.rs](../../crates/tm-core/src/memprobe.rs)）：支持 `target_os = "android"` 与 `/proc/self/status` `VmHWM` 读取。
- ✅ **Android `toybox taskset` 兼容**（[device_harness.py](../../bench/device_harness.py)）：自动剥离 `0x` 前缀（传 `1f0` / `f0`），同时兼容 Android `toybox` 与 Linux `util-linux`。

---

## 3. 四款基准模型真机实测结果（`0x1f0`：1×X3 + 4×A715，`Isa::NeonI8mm`）

连续跑满 4 款模型 × 多轮测试（覆盖冷机峰值与被动散热温控稳态）：

| 模型（Q4_0） | 文件大小 | Prefill `pp512`（中位数 / 峰值） | **Decode `tg64`（热稳态 / 冷峰值）** | 冻结语料 TTFT (~300 tok) | 峰值 RSS (`VmHWM`) | 额外脏内存 | 跨机/自洽确定性 |
|---|---:|---:|---:|---:|---:|---:|---|
| **`SmolLM2-360M-Instruct`** | 219.1 MiB | `180.0` / **`186.5 tok/s`** | **`83.6`** / **`90.0 tok/s`** | **1.62 – 1.75 s** | **256.5 MiB** | **+37.4 MiB** | ✅ 与 M4 **100% 逐 token 一致** |
| **`Qwen3.5-0.8B` (GDN)** | 483.7 MiB | `67.6` / **`71.4 tok/s`** | **`30.8`** / **`37.9 tok/s`** | **4.19 – 4.52 s** | **522.9 MiB** | **+39.2 MiB** | ✅ 与 M4 **100% 逐 token 一致** |
| **`Llama-3.2-1B-Instruct`** | 737.2 MiB | `60.0` / **`68.2 tok/s`** | **`23.1`** / **`23.8 tok/s`** | **5.14 s** | **769.4 MiB** | **+32.2 MiB** | ✅ 本机多轮 100% 一致¹ |
| **`Qwen3.5-2B` (GDN)** | 1158.6 MiB | `31.8` / **`33.2 tok/s`** | **`11.0`** / **`13.1 tok/s`** | **9.22 s** | **1200.0 MiB** | **+41.4 MiB** | ✅ 本机多轮 100% 一致¹ |

> ¹ 注：Pixel 8 本机多轮贪心输出 `100%` 逐位确定一致（符合 N-10）；1B/2B 模型在 350+ token 长序列末尾与 macOS M4 出现单个近距词分叉，源于 `musl libm` 与 Apple `libSystem libm` 的超越函数（`expf`/`sin`/`cos`）存在 1 ULP 舍入差异（短句如 `"The capital of France is"` 24 token 两端输出逐字完全相同）。

### 实测关键发现
1. **极低运行时脏内存开销**：得益于零拷贝 `mmap`、扁平词表与 `FP16` KV cache，哪怕跑 **248k 词表的 `Qwen3.5-2B`**，全进程 `VmHWM` 峰值内存（`1200.0 MiB`）仅比模型文件本身多出 **`41.4 MiB`**！在 8GB Pixel 8（可用 `3.81 GiB`）上甚至具备跑 **3B–4B Q4_0** 模型的充裕空间。
2. **被动散热温控幅度**：连续高负载运行十余轮后，Tensor G3 的 `Qwen3.5-0.8B` Decode 从冷机 **`37.9 tok/s`** 平滑回落至热稳态 **`30.8 tok/s`（仅降约 18.7%）**，远好于早期预估的 50% 腰斩。
3. **Qwen3.5 CPU 热点分布（Pixel 8 真机 `profile`）**：
   - `lm_head (Q6_K, 248k 词表)` 占 **27.6%**（`10.18 ms`），`ffn_gate_up+silu` 占 **20.4%**，`gdn_proj` 占 **14.9%**，`gdn_recur+gnorm` 占 **8.5%**——这验证了 M3/M4 阶段对大词表 `lm_head` 做低比特量化或 MTP 投机解码在手机端同样有接近 **25%–30%** 的提速空间。
4. **USB 供电测量提示**：USB 连线测试期间手机处于 `Charging` 状态，`/sys/class/power_supply/battery/current_now` 反映的是净充电电流而非纯放电功耗；后续测纯电池 J/token 时需切换为无线 `adb connect <ip:port>` 拔线放电测量，或使用 Perfetto ODPM 轨道。

---

## 4. 首字延迟（TTFT）与微调评估更新

- **TTFT 实测与压缩空间**：`Qwen3.5-0.8B` 约 300 token 输入实测 TTFT 为 **`4.19 – 4.52 s`**（`SmolLM2-360M` 仅 **`1.62 s`**），而模型冷/热加载本身仅 **`128 – 228 ms`**。因此手机端压 TTFT 的头号抓手依然是 **F-13 前缀缓存 + 状态快照**（可将多轮对话 TTFT 从 4.5s 压至 `< 50 ms`），其次是 M3 接入 **Mali-G715 Vulkan Prefill**。
- **端侧微调（QLoRA）修正**：实测 `Qwen3.5-0.8B` 5 核 CPU `i8mm` Prefill 为 `71.4 tok/s`，折算纯 CPU FP32/FP16 QLoRA 稳态吞吐约为 **`12 – 20 tok/s`**（接入 Mali-G715 Vulkan FP16 后有望翻倍至 **`35 – 60 tok/s`**），且 `0.8B` 训练总 RSS 仅 `~610 MiB`，在 Pixel 8 的 `3.81 GiB` 可用内存中极为宽裕。

---

## 5. 多引擎同机交替对比实测（`tinymodel` vs `llama.cpp` vs `HuggingFace candle`）

严格按照 [TESTING.md](../TESTING.md) §3.2 基准协议，使用 [device_harness.py](../../bench/device_harness.py) 的 `compare` 子命令在同一台 Google Pixel 8 上**逐轮交替运行**（保证热包络完全一致），对比了 5 组配置（原始结果存档于 [2026-11-20-8efe61d-multi-engine.json](../../bench/results/pixel-8/2026-11-20-8efe61d-multi-engine.json)）：
- **`tinymodel`**（纯 Rust，`880 KB` 静态 ELF，绑 `0x1f0` 5T，零拷贝 `mmap`）
- **`llama.cpp (--repack 0)`**（`b001d7b`，绑 `0x1f0` 5T，零拷贝 `mmap` 同口径对比）
- **`llama.cpp (KleidiAI --repack 1)`**（绑 `0x1f0` 5T，Arm 官方手写汇编微内核 + 加载时内存重排）
- **`llama.cpp (默认未绑核)`**（开箱默认 `-t 9 --repack 1`）
- **HuggingFace `candle` v0.9.2**（纯 Rust 对照基线，`2.8 MB` 静态 ELF，绑 `0x1f0` 5T；不支持 `Qwen3.5` GDN 结构）

### 5.1 实测对比总表（3 轮交替取中位数，热稳态）

| 模型（Q4_0） | 引擎配置 | Prefill `pp512` (tok/s) | **Decode `tg64` (tok/s)** | 相对 `llama.cpp (mmap)` | **峰值内存 RSS (MiB)** |
|---|---|---:|---:|---:|---:|
| **`SmolLM2-360M`**<br>（文件 `219.1 MiB`） | **`tinymodel` (`0x1f0`)** | 187.74 | **`89.23`** | **93.3%**（超默认未绑核 **+15.4%**） | **`259.0`**（最低！） |
| | `llama.cpp (--repack 0)` | 239.59 | **95.62** | 100.0% | 286.2 |
| | `llama.cpp (KleidiAI repack=1)` | 470.29 | **100.12** | 104.7% | **494.7（2.26× 文件大小！）** |
| | `llama.cpp (默认 9T 未绑核)` | 400.44 | **77.29** | 80.8% | 494.9 |
| | `HuggingFace candle` | 12.04 | **1.34** | 1.4% | **904.9（3.49× `tinymodel`！）** |
| **`Qwen3.5-0.8B`**<br>（文件 `483.7 MiB`） | **`tinymodel` (`0x1f0`)** | 59.32 | **`24.16`** | **84.5%** | **`524.6`**（省 `100–612 MiB`！） |
| | `llama.cpp (--repack 0)` | 91.72 | **28.58** | 100.0% | 625.1 |
| | `llama.cpp (KleidiAI repack=1)` | 178.07 | **30.85** | 107.9% | **1136.5（2.17× `tinymodel`！）** |
| | `llama.cpp (默认 9T 未绑核)` | 180.66 | **30.86** | 108.0% | 1137.4 |
| **`Llama-3.2-1B`**<br>（文件 `737.2 MiB`） | **`tinymodel` (`0x1f0`)** | 53.45 | **`18.35`** | **87.5%**（Prefill **89.3%**） | **`773.8`**（省 `104–2868 MiB`！） |
| | `llama.cpp (--repack 0)` | 59.87 | **20.98** | 100.0% | 878.2 |
| | `llama.cpp (KleidiAI repack=1)` | 123.25 | **25.40** | 121.1% | **1625.3（2.10× `tinymodel`！）** |
| | `llama.cpp (默认 9T 未绑核)` | 158.75 | **20.66** | 98.5% | 1625.2 |
| | `HuggingFace candle` | 7.09 | **1.00** | 4.8% | **3641.9（4.71× `tinymodel`，逼近 OOM！）** |
| **`Qwen3.5-2B`**<br>（文件 `1158.6 MiB`） | **`tinymodel` (`0x1f0`)** | **29.94** | **`9.15`** | Decode **75.2%** / **Prefill `107.8%`！** | **`1199.6`**（省 `118–1332 MiB`！） |
| | `llama.cpp (--repack 0)` | 27.77 | **12.16** | 100.0% | 1318.1 |
| | `llama.cpp (KleidiAI repack=1)` | 77.00 | **13.89** | 114.2% | **2531.9（净多吃 `1.33 GiB` 内存！）** |
| | `llama.cpp (默认 9T 未绑核)` | 83.02 | **15.33** | 126.1% | 2531.9 |

### 5.2 核心结论与下一阶段（M3）精准发力点

1. **对比纯 Rust 生态（`HuggingFace candle`）：代际级领先**
   - `tinymodel` 的 Decode 速度快 **`18.4× – 66.6×`**，Prefill 快 **`7.5× – 15.6×`**，内存占用仅为 `candle` 的 **`1/3.5 – 1/4.7`**（`Llama-1B` 上 `candle` 飙至 `3.64 GiB` 濒临触发 Android LMK 杀进程，而 `tinymodel` 仅 `773.8 MiB`），且原生支持 `Qwen3.5` GDN 混合结构。
2. **对比 `llama.cpp` 的巨大内存优势（手机端生死线）**
   - `tinymodel` 在全部四款模型上内存占用均为全场最低：比同为零拷贝的 `llama.cpp (--repack 0)` 省 **`27 – 118 MiB`**；
   - **揭开了 `llama.cpp` 默认开启 `KleidiAI repack=1` 在手机端的隐蔽代价**：由于在加载时把权重重排进匿名堆内存同时保留 `mmap` 页，`llama.cpp (KleidiAI)` 的峰值 RSS 直接**翻了 2.1 倍**（`Qwen3.5-2B` 从 `1.20 GiB` 暴涨到 **`2.53 GiB`**；若跑 4B 模型将突破 `5 GiB` 直接在 8GB 手机上 OOM 崩溃！）。
   - **对 `tinymodel` 的直接启示（验证 `F-04` 设计前瞻性！）**：在 M3 实现 **`Q4_0_4x8` 离线文件重排 + 直接零拷贝 `mmap`**，就能**同时白拿 KleidiAI 的 `2.5×` Prefill 提速与 `+15%` Decode 提速，且保持 `1.20 GiB` 的极低零拷贝内存，不花 1 字节额外脏内存**！
3. **定位出的 2 个具体 CPU 内核提速点（M3 优先攻克）**
   - **① `Qwen3.5` GDN 递推与卷积的 NEON 向量化**：目前 [ops.rs](../../crates/tm-cpu/src/ops.rs) 中的 `gdn_recurrence_parallel` 与 `l2_norm_inplace` 仍是标量循环（占 Pixel 8 单步耗时的 `15.2%`），改为手写 NEON `vfmaq_f32` 后预计直接追平 `Qwen3.5` 与 `llama.cpp (--repack 0)` 的大半差距；
   - **② `Q4_0_4x8` 重排微内核 + `Q6_K` 4 行展开 `i8mm` 内核**：攻克 Prefill `pp512` 的 `2×` 差距以及 248k 词表 `lm_head`（占 `27.6%` 时间）的带宽效率。
