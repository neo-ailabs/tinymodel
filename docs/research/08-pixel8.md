# Google Pixel 8（Tensor G3）设备档案与真机实测报告

> 更新日期：2026-11-21（**已完成 P0 测量校准 + P1 异构自动绑核 + P2 GDN/Attention/SwiGLU 向量化与并行化真机复测**，序列号 `3B131FDJH00216`，Android 16）  
> 固定参考设备之一（见 [AGENTS.md](../../AGENTS.md) §8）。原始基准与多引擎对比存档于 [2026-11-20-a80228f.json](../../bench/results/pixel-8/2026-11-20-a80228f.json)、[2026-11-20-8efe61d-multi-engine.json](../../bench/results/pixel-8/2026-11-20-8efe61d-multi-engine.json) 与最新 5 轮随机交错结果 [2026-11-21-47861b7-multi-engine.json](../../bench/results/pixel-8/2026-11-21-47861b7-multi-engine.json)。

---

## 1. 真机探测硬件规格（`tinymodel probe` 校准后实测确认）

| 项 | Pixel 8 真机实测值 | 对比 Pi 5 / RK3588 |
|---|---|---|
| **SoC / OS** | **Google Tensor G3**（三星 4nm），Android 16（`sched_pixel` 调频器） | Pi 5：BCM2712 (16nm)；RK3588：8nm |
| **三丛集异构 CPU** | - `cpu0..3`：**4× Cortex-A510 @ 1704 MHz**（掩码 `0x0f`）<br>- `cpu4..7`：**4× Cortex-A715 @ 2367 MHz**（掩码 `0xf0`）<br>- `cpu8`：**1× Cortex-X3 @ 2914 MHz**（掩码 `0x100`） | Pi 5：4×A76 同构；RK3588：4×A76 + 4×A55 |
| **ISA 与 L1 自检** | 自动命中 **`Isa::NeonI8mm`**（`neon`、`dotprod`、`i8mm` 全支持；真机 L1 逐位自检 **PASSED**） | Pi 5 / RK3588 仅到 `dotprod`（无 `i8mm`） |
| **物理内存与可用水位** | 总计 `7.39 GiB`（`7,941,091,328 B`），空闲可用 **`3.81 – 4.11 GiB`** | 足够无压力常驻 `Qwen3.5-2B`（峰值 RSS `1.17 GiB`，真实脏内存仅 **`44.9 MiB`**） |
| **DVFS 预热后 STREAM 持续带宽** | - **绑 `X3 + 4×A715`（`0x1f0`, 5T，350ms 预热 + 1s 持续测量）**：**Read `31.32 GB/s` / Copy `40.15 GB/s`**（修复前冷跑 5 轮仅测到升频前的 `21.84 GB/s`）<br>- 仅绑 `4×A715`（`0xf0`, 4T）：**Read `31.17 GB/s` / Copy `40.34 GB/s`**<br>- 全核混跑（含 A510）：Read `15.61 GB/s`（被顺序小核同步屏障拖慢一半） | Pi 5 约 `11.5 GB/s`；RK3588 约 `17–23 GB/s`；M4 约 `102 GB/s` |
| **免 Root 温度与脏内存遥测** | `/system/bin/dumpsys thermalservice` 直接读取 `BIG` / `MID` / `LITTLE` / `G3D` / `TPU` / `VIRTUAL-SKIN` 温度与 `Thermal Status`；`/proc/self/smaps_rollup` 直接分离 `Private_Dirty` 与 `Pss_Anon` | Linux SBC 走 `/sys/class/thermal` + `smaps_rollup` |

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

## 5. 多引擎同机随机交错对比实测（P0+P1+P2 优化前后）

严格按照 [TESTING.md](../TESTING.md) §3.2 基准协议，使用校准后的 [device_harness.py](../../bench/device_harness.py) `compare` 子命令在同一台 Google Pixel 8 上执行 **5 轮随机打乱引擎顺序 + 轮间冷却 + 逐轮配对比率中位数 `median(tm_i / llama_i)`**（结果存档于 [2026-11-21-47861b7-multi-engine.json](../../bench/results/pixel-8/2026-11-21-47861b7-multi-engine.json)）。

### 5.1 P1（开箱自动异构绑核）与 P2（GDN/SwiGLU/GQA 向量化与并行化）前后对比

1. **P1 开箱裸跑（不传 `taskset`、不传 `-t`）**：
   - [affinity.rs](../../crates/tm-cpu/src/affinity.rs) 自动识别异构 big.LITTLE（仅当同 SoC 存在非小核时排除 `A53/A55/A510/A520`，因此 Pi Zero 2W 同构 `4×A53` 保留全部 4 核；Pixel 8 自动选 `5` 核 `0x1f0`；RK3588 自动选 `4` 核 `0xf0`；Apple M4 自动通过 `hw.perflevel0.physicalcpu` 选 `4` 颗 P 核；外部已设 `taskset` 时严格尊重用户掩码）。
   - Pixel 8 直接执行 `./tinymodel bench -m .../Qwen3.5-0.8B-Q4_0.gguf`（零参数调优）：**`pp512 = 143.77 tok/s`、`tg64 = 43.49 tok/s`**（对比优化前未绑核的 `8.76–13.34 tok/s`，**开箱速度提升 `3.3× – 5.0×`**）。
2. **P2 算子级剖析（`tinymodel profile` `pp256` + `tg32` 真机同机对比）**：
   - **Pixel 8 (`5T`, `Qwen3.5-0.8B`)**：Prefill `gdn_recur+gnorm` 从 `778.63 ms` 降至 **`179.95 ms`（`4.33×`）**，`gdn_conv_l2` 从 `423.76 ms` 降至 **`172.57 ms`（`2.46×`）**；Decode `gdn_recur+gnorm` 从 `2193.9 us` 降至 **`907.7 us`（`2.42×`）**，`gdn_conv_l2` 从 `1507.4 us` 降至 **`729.6 us`（`2.07×`）**。
   - **Pixel 8 (`5T`, `SmolLM2-360M`)**：Prefill `swiglu` 从 `228.19 ms` 降至 **`70.06 ms`（`3.26×`）**，`pp256` 总吞吐 **`+20.5%`**（`220.2 → 265.4 tok/s`）。
   - **Apple M4 (`4T`, `Qwen3.5-0.8B`)**：`pp512` 跃升至 **`406.77 tok/s`（`+52.3%`）**，`tg64` 跃升至 **`127.81 tok/s`（`+13.3%`，达 M4 CPU `llama.cpp` 的 `118.5%`）**。

### 5.2 校准后 5 轮随机交错多引擎总表（Pixel 8，`0x1f0`，`5T`）

| 模型（Q4_0） | 引擎配置 | Prefill `pp512` (tok/s) | **配对 `pp` vs mmap** | **Decode `tg64` (tok/s)** | **配对 `tg` vs mmap** | **峰值 RSS / 真实脏内存 (MiB)** |
|---|---|---:|---:|---:|---:|---:|
| **`SmolLM2-360M`**<br>（文件 `219.1 MiB`） | **`tinymodel`** | **214.33**（前 187.7） | **89.7%**（前 78.4%） | **`88.26`** | **91.6%** | **`259.8`（dirty `43.6`）** |
| | `llama.cpp (--repack 0)` | 238.38 | 100.0% | **95.20** | 100.0% | 286.0 |
| | `llama.cpp (KleidiAI repack=1)` | 488.53 | 203.8% | **105.92** | 107.7% | **494.7（2.26× 文件！）** |
| | `HuggingFace candle`¹ | 12.04 | 5.0% | **1.34** | 1.4% | **904.9（3.48× `tinymodel`！）** |
| **`Qwen3.5-0.8B`**<br>（文件 `483.7 MiB`） | **`tinymodel`** | **126.40**（前 59.3！） | **`101.2%`（反超！）** | **`37.00`**（前 24.2！） | **`101.5%`（反超！）** | **`522.4`（dirty `43.5`）** |
| | `llama.cpp (--repack 0)` | 122.61 | 100.0% | **36.57** | 100.0% | 624.8（多占 102 MiB） |
| | `llama.cpp (KleidiAI repack=1)` | 236.42 | 192.8% | **40.15** | 108.4% | **1135.9（多占 `613.5 MiB`！）** |
| **`Llama-3.2-1B`**<br>（文件 `737.2 MiB`） | **`tinymodel`** | **73.75**（前 53.5！） | **`106.9%`（反超！）** | **`21.69`** | **90.1%**（前 87.5%） | **`775.4`（dirty `41.1`）** |
| | `llama.cpp (--repack 0)` | 62.45 | 100.0% | **23.07** | 100.0% | 876.8（多占 101 MiB） |
| | `llama.cpp (KleidiAI repack=1)` | 160.99 | 237.1% | **27.96** | 118.3% | **1623.8（多占 `848.4 MiB`！）** |
| | `HuggingFace candle`¹ | 7.09 | 11.4% | **1.00** | 4.3% | **3641.9（4.70× `tinymodel`！）** |
| **`Qwen3.5-2B`**<br>（文件 `1158.6 MiB`） | **`tinymodel`** | **44.55**（前 29.9！） | **`108.5%`（反超！）** | **`13.53`**（前 9.15！） | **96.5%**（前 75.2%！） | **`1201.2`（dirty `44.9`）** |
| | `llama.cpp (--repack 0)` | 41.84 | 100.0% | **14.52** | 100.0% | 1316.7（多占 116 MiB） |
| | `llama.cpp (KleidiAI repack=1)` | 93.77 | 223.0% | **15.26** | 106.2% | **2530.4（多占 `1.30 GiB`！）** |

> ¹ `candle` v0.9.2 数据取自首轮多引擎基线（不支持 `Qwen3.5` GDN 结构）。

### 5.3 六大固定参考设备收益汇总与下一阶段（M3）焦点

| 参考设备 | 本轮（P0 + P1 + P2）直接收益 |
|---|---|
| **Apple M4（macOS Host）** | 自动按 `hw.perflevel0.physicalcpu` 默认选 `4` 颗 P 核；`Qwen3.5-0.8B` CPU `pp512` **`+52.3%`**（`406.8 tok/s`）、`tg64` **`+13.3%`**（`127.8 tok/s`，追平本机 Metal 解码） |
| **Google Pixel 8（Tensor G3）** | 开箱免 `taskset` 提速 **`3.3×–5.0×`**；同口径零拷贝下 **3 款模型 Prefill 反超 `llama.cpp (--repack 0)`（`101.2%–108.5%`）**，`Qwen3.5-0.8B` **Prefill 与 Decode 双双反超**；真实脏内存恒定 **`41.1–44.9 MiB`** |
| **Rockchip RK3588（4×A76 + 4×A55）** | `ThreadPool` 自动排除 `4×A55` 小核绑定 `0xf0`（`4×A76`）；由于无 `i8mm`，纯 ARMv8.0 NEON 实现的 `gdn_recurrence_head` / `rms_norm_f32` / 并行 `conv1d` 与 `swiglu` 在 `Isa::NeonDotprod` 上同样获得完整 `2.5×–4×` 非矩阵算子提速 |
| **Raspberry Pi 5（4×A76 同构）** | 自动保持 `4` 线程且 `n_thr <= 4` 维持单段连续权重预取；`Qwen3.5` GDN 非矩阵阶段从占 Prefill `~40%` 压至 `< 17%` |
| **Pi 4B / Pi Zero 2W（`micro` 同构）** | 同构 `4×A53`（Zero 2W）不被误排核，保留全部 4 核；`NEON_V80_TABLE` 直接挂载新增 NEON `gdn_recurrence_head` / `rms_norm` / `l2_norm`，`micro` 纯静态二进制（`887 KB`）实测 **`0` 条 `sdot`/`smmla` 指令** |

- **下一阶段（M3）最值得投入的两项硬核工作**：
  1. **离线 `Q4_0_4x8` 交错权重重排（`F-04`）+ 零拷贝 `mmap` 微内核**：把 `llama.cpp (KleidiAI --repack 1)` 剩下的 **`2×` Prefill 优势与 `6%–18%` Decode 优势**拿到手，同时完全避免其 **`2.1×` 内存翻倍（省 `613 MiB – 1.30 GiB`）** 的致命缺陷；
  2. **`lm_head (Q6_K)` `i8mm` / 多行展开与 MTP/Suffix 投机解码 + Mali-G715 Vulkan 后端**：当前 `Qwen3.5-0.8B` Decode 中 `248k` 词表 `lm_head` 已占 **31.4%**（`7.65 ms/tok`），是单步解码最大单一耗时项。
