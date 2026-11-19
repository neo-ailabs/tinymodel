# M0 基线报告：Apple M4 (16GB) · llama.cpp

> 生成日期：2026-11-19  
> 原始结果 JSON：[apple-m4-16gb-llama-cpp-2026-11-19.json](apple-m4-16gb-llama-cpp-2026-11-19.json)  
> 复现脚本：[run_llama_baseline.py](../run_llama_baseline.py) · 冻结语料清单：[manifest.json](../data/manifest.json)（`protocol_id: tinymodel-m0-v1`）

---

## 1. 测试环境与方法

- **设备**：Apple M4，16 GB 统一内存（4 Performance 核 + 6 Efficiency 核），macOS 27.2 (`26B123`)
- **CPU 指令集实测**：NEON、`FEAT_DotProd=1`、`FEAT_I8MM=1`、`FEAT_FP16=1`、`FEAT_BF16=1`、`FEAT_SME=1`、`FEAT_SME2=1`
- **STREAM 持续内存带宽实测**（[stream_bw.c](../stream_bw.c)，单数组 512 MiB，三数组共 1.5 GiB，穿透 L2/SLC）：
  - **1 线程（单 P 核）**：Copy `83.55 GB/s` · Triad `89.30 GB/s` · 纯读 Read `53.32 GB/s`
  - **4 线程（全 P 核）**：Copy **`100.87 GB/s`** · Triad **`93.41 GB/s`** · 纯读 Read `78.96 GB/s`
  - **10 线程（4P + 6E 全核）**：Copy `93.23 GB/s` · Triad `91.93 GB/s` · 纯读 Read **`106.97 GB/s`**
- **基线引擎**：`llama.cpp` 主线 `b001d7b`（`b12530`），AppleClang 21.0.0 Release 编译：
  1. `cpu_norepack`：纯 CPU 构建（`GGML_METAL=OFF GGML_BLAS=OFF`），`-t 4 --repack 0`（零拷贝 mmap，不重排权重）
  2. `cpu_kleidiai`：纯 CPU 构建（`GGML_CPU_KLEIDIAI=ON`），`-t 4 --repack 1`（加载时重排权重，启用 Arm KleidiAI SME2 / I8MM 微内核）
  3. `metal`：Metal 构建（`GGML_METAL=ON`），`-ngl 99 -fa 1`（全部层卸载到 Apple9 / Metal4 GPU，开启 Flash Attention）
- **统计口径**：
  - 吞吐量：`llama-bench -p 512 -n 128 -r 5`，报告 **5 轮中位数**（`pp512` 预填充 tok/s，`tg128` 解码 tok/s）。
  - 端到端与内存：`/usr/bin/time -l llama-completion` 在冻结语料 `prompt_short.txt`（约 308 tokens）上贪心生成 128 tokens（`--temp 0 -s 42 -c 2048`），记录 `max_rss_mib`（包含 mmap 文件页）、`peak_footprint_mib`（进程私有脏页内存）以及生成文本 SHA-256（供 M1/M2 逐 token 对齐校验）。
  - 热状态：全程 `NSProcessInfo.thermalState = nominal`（无过热降频）。

---

## 2. 基准汇总表（中位数 tok/s 与内存峰值）

| 档位 | 模型文件 | 体积 (MiB) | 参数量 | 模式 | `pp512` (tok/s) | `tg128` (tok/s) | 等效带宽 (GB/s)¹ | Max RSS (MiB) | Dirty Footprint (MiB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| **micro** | `SmolLM2-360M-Instruct-Q4_0` | 219.1 | 362M | `cpu_norepack` | 603.7 | 254.4 | 58.0 (54.2%) | 342.4 | **116.5** |
| | | | | `cpu_kleidiai` | 1298.8 | **257.1** | 58.6 (54.8%) | 807.2 | 581.5 |
| | | | | `metal` | **4366.4** | 231.6 | 52.8 (49.4%) | 408.8 | 186.7 |
| **edge** | `Qwen3.5-0.8B-Q4_0` (GDN 3:1) | 483.7 | 752M | `cpu_norepack` | 345.1 | 108.0 | 53.6 (50.1%) | 632.5 | **132.6** |
| | | | | `cpu_kleidiai` | 741.0 | 105.2 | 52.2 (48.8%) | 1877.0 | 1377.6 |
| | | | | `metal` | **2287.7** | **128.0** | 63.5 (59.4%) | 687.0 | 190.1 |
| **edge** | `Qwen3.5-0.8B-Q4_K_M` (GDN 3:1) | 507.9 | 752M | `cpu_norepack` | 309.3 | 106.2 | 55.4 (51.8%) | 656.8 | **132.7** |
| | | | | `cpu_kleidiai` | 492.3 | 93.0 | 48.5 (45.3%) | 1764.0 | 1240.4 |
| | | | | `metal` | **2180.6** | **122.9** | 64.1 (59.9%) | 712.4 | 189.8 |
| **edge** | `Llama-3.2-1B-Instruct-Q4_0` | 737.2 | 1236M | `cpu_norepack` | 223.6 | 88.1 | 67.4 (63.0%) | 901.7 | **145.8** |
| | | | | `cpu_kleidiai` | 597.5 | 84.2 | 64.4 (60.2%) | 2662.9 | 1916.9 |
| | | | | `metal` | **1676.7** | **112.4** | 86.0 (80.4%) | 949.7 | 191.3 |
| **edge** | `Llama-3.2-1B-Instruct-Q4_K_M` | 770.3 | 1236M | `cpu_norepack` | 215.2 | 85.2 | 68.1 (63.7%) | 934.6 | **145.7** |
| | | | | `cpu_kleidiai` | 479.2 | 86.4 | 69.1 (64.6%) | 2464.4 | 1685.2 |
| | | | | `metal` | **1596.7** | **107.5** | 86.0 (80.4%) | 981.5 | 190.9 |
| **edge** | `Qwen3.5-2B-Q4_0` (GDN 3:1) | 1158.6 | 1882M | `cpu_norepack` | 142.9 | 52.4 | 63.1 (58.9%) | 1315.3 | **140.8** |
| | | | | `cpu_kleidiai` | 362.5 | 49.5 | 59.6 (55.7%) | 4185.4 | 3012.2 |
| | | | | `metal` | **1053.1** | **65.1** | 78.4 (73.3%) | 1364.8 | 192.2 |
| **edge** | `gemma-4-E2B-it-Q4_0` (PLE) | 2709.9 | 4629M | `cpu_norepack` | 210.8 | 39.8 | *112.5*² | 2902.0 | **146.1** |
| | | | | `cpu_kleidiai` | 335.0 | 40.9 | *115.5*² | 4862.5 | 3101.4 |
| | | | | `metal` | **2319.6** | **54.0** | *152.7*² | 2949.8 | 190.2 |

> ¹ **等效带宽** = `model_size_bytes × tg128 / 1e9`，括号内为除以 10 核 STREAM 纯读峰值 `106.97 GB/s` 的百分比（若按 4 个 P 核纯读上限 `78.96 GB/s` 计算，Llama-3.2-1B CPU 解码已达 **86.3%**）。  
> ² **Gemma 4 E2B 说明**：总文件 `2.71 GiB`（4.63B 参数），但其中超过一半是 **Per-Layer Embeddings (PLE)** 查表参数，每个 token 只按 token id 查一行而不参与矩阵乘。因此按总文件大小折算的“表观带宽”超过物理内存带宽（152.7 GB/s）；按实际活跃的 ~2.3B 参数（约 1.35 GB）计算，Metal 实际有效带宽约为 **73 GB/s**。

---

## 3. 实测关键发现（直接指导 `tinymodel` 架构与内核实现）

### 发现 1：在线权重重排（KleidiAI `--repack 1`）导致私有脏页内存暴涨 2–3 GiB
- `cpu_norepack` 通过只读 `mmap` 零拷贝访问权重，无论模型从 219 MiB 增至 2.71 GiB，**进程私有脏页（`peak_footprint`）始终稳定在 116–146 MiB**（仅 KV cache + 计算缓冲）。
- 开启 `cpu_kleidiai` 在线重排后，`Qwen3.5-2B` 的私有脏页暴涨到 **3012 MiB**（Max RSS 达 4185 MiB），`SmolLM2-360M` 的 RSS 也从 **342 MiB 翻倍至 807 MiB**——这在 512 MiB 内存的 Pi Zero 2W 上会**立即触发 OOM**，在 4 GB 的树莓派 5 上也会挤占大半可用内存。
- **对 `tinymodel` 的启示**：验证了我们在 `REQUIREMENTS.md` F-04 中的设计——**权重重排必须在转换/离线阶段写回文件并直接 `mmap`**（或将重排缓存落盘后 mmap），绝不能在端侧内存受限设备上做堆内存拷贝重排。

### 发现 2：CPU Prefill 提速 2.2x–2.7x，但 Decode 几乎不变（甚至略慢）
- `cpu_kleidiai` 相比 `cpu_norepack` 在 `pp512` 上取得 **2.15x–2.67x** 加速（例如 Llama-3.2-1B Q4_0 从 `223.6` 提升至 `597.5 tok/s`），但 `tg128` 解码几乎持平甚至下降 3%–12%（Qwen3.5-0.8B Q4_K_M 从 `106.2` 掉到 `93.0 tok/s`）。
- 原因有二：① 单 token 解码完全卡在 4 个 P 核的 ~79 GB/s 读取带宽墙上；② Apple M4 的 4 个 P 核**共享同一个 SME 矩阵协处理器（`sme_thread_cap = 1`）**，KleidiAI 在单线程时把 `SmolLM2-360M` 解码从 `113.6` 拉高到 `198.2 tok/s`，但 4 线程时退回 I8MM/NEON 路径。
- **对 `tinymodel` 的启示**：在 M4 上同时要写好 **I8MM (`smmla`) 多核并行 GEMV** 与 **SME2 单流 GEMM**，并根据线程数和批大小切换。

### 发现 3：超小模型（≤360M）在 M4 上 CPU 解码比 Metal GPU 更快
- `SmolLM2-360M` 的 CPU 4 核解码达到 **257.1 tok/s**，反超 Metal GPU 的 **231.6 tok/s**（快 11%）。
- 原因：模型仅 219 MiB，每步解码耗时不到 4 ms，Metal command buffer 提交与多层 kernel launch 同步开销占比显著，而 M4 的 16 MiB P-core L2 + 系统级缓存（SLC）让 CPU 访存延迟极低。到 ≥0.8B 模型后，Metal 凭借更高的全芯片带宽利用率（86 GB/s vs P 核 68 GB/s）反超 CPU 18%–35%。
- **对 `tinymodel` 的启示**：`tm-metal` 必须重视 kernel 融合（如 `gemv_pair` 合并 gate+up+SiLU、`gemv_cat` 合并 QKV）与单 command buffer 低开销提交。

### 发现 4：Qwen3.5（Gated DeltaNet 3:1）对比传统 Transformer 的吞吐特征
- `Qwen3.5-0.8B`（484 MiB）体积仅为 `Llama-3.2-1B`（737 MiB）的 65%，但有效解码带宽利用率低约 13 个百分点（CPU `53.6 GB/s` vs `67.4 GB/s`，Metal `63.5 GB/s` vs `86.0 GB/s`）。
- 原因：75% 的层是 Gated DeltaNet，每步解码除了权重 GEMV 外，还要读写固定大小的 `S×h_v×S` 递归状态矩阵并执行 conv1d + SiLU + 门控递推更新，算子更碎。将 GDN 解码全链路融合进单个寄存器驻留 kernel（参考 `Strata/metal/kernels/gdn.metal`）在 `tinymodel` M2 阶段会有直接收益。

---

## 4. 下一步（树莓派 5 待补测说明）

同一套脚本 [run_llama_baseline.py](../run_llama_baseline.py) 和 [stream_bw.c](../stream_bw.c) 已兼容 Linux AArch64（自动读取 `/proc/device-tree/model`、`/sys/class/thermal/thermal_zone0/temp` 和 `cpufreq`）。拿到 Raspberry Pi 5 环境后只需运行：

```sh
python3 bench/run_llama_baseline.py --reps 5 --cpu-threads 4 --tag rpi5-8gb
```

即可产出同构 JSON 并补全 M0 的树莓派基线列。
