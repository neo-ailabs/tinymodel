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
