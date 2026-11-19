# Apple M4 算子级 Profiling 与对比 `llama.cpp` 优化分析

> 测试环境：Apple M4 16GB（4P+6E，STREAM 峰值读带宽 `106.97 GB/s`），macOS，`tinymodel` commit `d19e433` + `tinymodel profile` vs `llama.cpp b001d7b`（Metal `-ngl 99 -fa 1` / CPU `-t 4`）。

---

## 1. 端到端现状总览

| 后端 | 模型 | 指标 | `tinymodel` | `llama.cpp` | 当前占比 |
|---|---|---|---|---|---|
| **Metal GPU** | `SmolLM2-360M-Q4_0` | `pp512` (tok/s) | 2,133.2 | 4,366.4 | **48.9%** |
| | | `tg128` (tok/s) | 226.4 | 231.6 | **97.7%** |
| | | Dirty Footprint | **137.8 MiB** | 186.7 MiB | **优于 26.2%** |
| **Metal GPU** | `Llama-3.2-1B-Q4_0` | `pp512` (tok/s) | 1,072.6 | 1,676.7 | **64.0%** |
| | | `tg128` (tok/s) | 108.2 | 112.4 | **96.3%** |
| | | Dirty Footprint | **146.8 MiB** | 191.3 MiB | **优于 23.3%** |
| **CPU (4T `i8mm`)** | `SmolLM2-360M-Q4_0` | `pp` (tok/s) | 287.7 (`pp64`) | 1,790.2 (`pp512`) | **16.1%** |
| | | `tg` (tok/s) | 224.8 | 254.4 (`--no-repack`) | **88.4%** |
| | | Dirty Footprint | **14.7 MiB** | 116.5 / 582.1 MiB | **优于 87%** |
| **CPU (4T `i8mm`)** | `Llama-3.2-1B-Q4_0` | `pp` (tok/s) | 102.2 (`pp64`) | 853.9 (`pp512`) | **12.0%** |
| | | `tg` (tok/s) | 79.9 | 88.1 (`--no-repack`) | **90.7%** |
| | | Dirty Footprint | **31.2 MiB** | 145.8 / 1217 MiB | **优于 79%** |

---

## 2. Metal GPU 硬件时间戳（`GPUStartTime`/`GPUEndTime`）拆解

### 2.1 Metal Decode（`B = 1`）算子热点

- **`Llama-3.2-1B-Q4_0`**（单 CB 端到端 `wall = 9242.9 us/tok` = `108.2 tok/s`；纯 GPU `8907.5 us/tok`；`host+sync = 335.4 us/tok [3.6%]`）：
  - **已接近硬件极限的大矩阵**：
    - `lm_head+norm`（`Q6_K`，205.5 MiB）：`2199.2 us` -> **97.98 GB/s**（达 M4 STREAM 峰值 91.6%）
    - `ffn_gate_up_silu`（`Q4_0` 融合，288.0 MiB）：`3316.1 us` -> **91.07 GB/s**
    - `ffn_down+res`（`Q4_0/Q4_1`，146.0 MiB）：`1767.9 us` -> **86.59 GB/s**
  - **明显掉速点**：
    1. **未融合的 `attn_wq` + `attn_wk` + `attn_wv`**：合计 `535.9 + 423.9 + 213.5 = 1173.3 us/tok`（占 12.5%），总权重仅 `54.0 MiB`，其中 **`attn_wk`（`n_out = 512`）仅 `22.26 GB/s`**、`attn_wv` 仅 `44.21 GB/s`！原因是 `n_out = 512` 仅启动 16 个 threadgroup，无法填满 M4 10 核 GPU 波次，且三次独立 dispatch 产生两次隐式内存屏障并重复读取 `xb` 3 次。
    2. **CPU 编码与同步开销（`host+sync = 335–385 us/tok`，在 `SmolLM2` 上占 `8.7%`）**：每个 token 结束后主机才开始编码下一轮 ~165–320 个 dispatch（且每次热路径调用 ObjC `maxTotalThreadsPerThreadgroup`），加上贪心解码把 128k `f32` logits（512 KB）传回 CPU 扫描 argmax，造成每步 ~0.35 ms 的 GPU 空泡。
    3. **细粒度算子屏障（`SmolLM2` 32 层尤为明显）**：`attn_norm (117.8 us) + ffn_norm (109.6 us) + rope_kv (93.8 us) + gqa_attn (357.3 us)` 合计 `678.5 us/tok`（**占 16.3%**）。对比 `llama.cpp` 的 `ggml-metal-fusion.h`：
       - `llama.cpp` 实现了 `GGML_METAL_FUSION_NORM_MUL_ADD`（把残差 `ADD` 与下一层 `RMS_NORM + MUL` 融合），每层少 2 次全屏障 dispatch。
       - 短序列时我们的 `gqa_attn_f16_kv` 固定开 4 个 simdgroup（128 线程）做 threadgroup 归约，而 `llama.cpp` `fa_vec_f16.metal` 在短上下文用单 simdgroup，规避了共享内存归约开销。

### 2.2 Metal Prefill（`pp512`，分块 `B = 128`）算子热点

| 算子阶段 | `SmolLM2-360M` GPU ms (`Share%`) | `Llama-3.2-1B` GPU ms (`Share%`) | 有效算力 |
|---|---|---|---|
| **`gqa_attn_batch`** | **47.60 ms (57.2%)** | **49.45 ms (34.2%)** | **148–152 GFLOP/s** ⚠️ |
| `w_gate_gemm` (`Q4_0` MMA) | 7.55 ms (9.1%) | 23.65 ms (16.4%) | **2.67–2.91 TFLOP/s** ✅ |
| `w_up_gemm+silu` (`Q4_0` MMA) | 8.20 ms (9.9%) | 25.19 ms (17.4%) | **2.46–2.73 TFLOP/s** ✅ |
| `w_down_gemm+res` (`Q4_0/Q4_1`) | 8.15 ms (9.8%) | 25.17 ms (17.4%) | **2.47–2.73 TFLOP/s** ✅ |
| `wq/wk/wv/wo_gemm` | 10.74 ms (13.0%) | 20.17 ms (13.8%) | 1.17–2.67 TFLOP/s |

- **头号瓶颈一目了然**：
  - 我们的 `gemm_q4_0_res`（`simdgroup_half8x8 -> float8x8`）已跑到 **2.67–2.91 TFLOP/s**；
  - 但 **`gqa_attn_batch` 占掉了 `SmolLM2` 57.2%、`Llama-1B` 34.2% 的全部 Prefill 时间**！因为当前 `gqa_attn_f16_kv_batch` 对每个 query token 独立开一个 threadgroup，导致 128 个 token 把整个历史 KV cache **重复从内存读了 128 遍**（`SmolLM2` 等效读了 `2.24 GiB`），算力仅 `150 GFLOP/s`。
  - **对比 `llama.cpp`**：
    1. **`fa_tensor.metal`（Tiled FlashAttention MMA）**：`16` 或 `32` 个 query token 加上同一 GQA 组的多个 query head（`group = 3` 或 `4`）**共享一次 K/V tile 加载**并用 `simdgroup_mma` 计算 $Q K^T$ 与 $P V$，可将 `48 ms` 压到 **`< 3 ms`**（直接让 `SmolLM2` `pp512` 翻倍至 `~4,100+ tok/s`，`Llama-1B` 提升至 `~1,550+ tok/s`）。
    2. **Metal 4 `mpp::tensor_ops::matmul2d` + 64×32 大分块**（`mul_mm.metal`）：`llama.cpp` 在 M4 上对 `srcB`（激活 `X`）直接用 cooperative tensor 从 device memory 读取，省掉 `X_tile` 的 threadgroup 搬运与一半 barrier，且 `NRA = 64` 进一步把 `wk`/`wv` 与 FFN GEMM 推高约 10–15%。

---

## 3. CPU（4 核 `i8mm`）算子拆解与对比 `llama.cpp`

| 算子阶段 | `SmolLM2-360M` us/tok (`Share%`, BW) | `Llama-3.2-1B` us/tok (`Share%`, BW) | 瓶颈根因 |
|---|---|---|---|
| **`gqa_attn`** | **1,059.6 us (22.7%, 3.73 GB/s)** ⚠️ | **978.3 us (7.7%, 3.23 GB/s)** ⚠️ | 标量 `f16_to_f32` 逐元素转换 + 未按 GQA KV 头分组复用 |
| **`ffn_down+res`** | **741.3 us (15.8%, 60.5 GB/s)** ⚠️ | **2,851.5 us (22.5%, 53.7 GB/s)** ⚠️ | `Q4_1` 走单行 `vec_dot` + 8192 维标量单线程 `quantize_row_q8_*` |
| `ffn_gate_up` | 1,270.7 us (27.2%, 69.6 GB/s) | 4,197.4 us (33.2%, 72.0 GB/s) | 标量 `quantize_row_q8_0` 占去约 150 us |
| `lm_head` | 686.0 us (14.7%, 73.1 GB/s) | 2,970.6 us (23.5%, 72.5 GB/s) | `Q6_K` 2 行展开可升级为 4 行 + `i8mm`/NEON `quantize_row_q8_k` |
| `swiglu` | 91.4 us (2.0%) | 145.4 us (1.1%) | 标量 `expf` 循环 |
| **Prefill (`pp64`)** | `287.7 tok/s` (vs `1,790`) ⚠️ | `102.2 tok/s` (vs `853.9`) ⚠️ | 尚未实现 CPU `forward_batch` 与 `Q4_0×Q8_0` 多 token GEMM 微内核 |

---

## 4. 按 ROI 排序的优化路线清单（可直接落地）

| 优先级 | 优化项 | 目标后端 | 预计收益（Apple M4 实测推算） |
|---|---|---|---|
| **P0** | **Metal Tiled FlashAttention (`simdgroup_mma`)**（多 query + GQA 同组共享 K/V tile） | Metal Prefill | `SmolLM2` `pp512` `2,133 -> ~4,100+ tok/s`（**~2.0×**）；`Llama-1B` `1,073 -> ~1,550+ tok/s`（**+45%**） |
| **P1** | **CPU NEON `fcvt` F16 KV 注意力 + 按 KV 头分组共享 K/V** | CPU Decode | `gqa_attn` `1,060 us -> ~80 us`：`SmolLM2` CPU `214 -> ~275 tok/s`（**反超 `llama.cpp` CPU 8%+**）；`Llama-1B` `+7%` |
| **P2** | **CPU 4 行展开 `gemv_q4_1_q8_1` + NEON 向量化 `quantize_row_q8_{0,1,k}`** | CPU Decode | `ffn_down` 带宽 `53.7 -> ~72 GB/s`：`Llama-1B` CPU `79.9 -> ~91 tok/s`（**反超 `llama.cpp` `--no-repack` 88.1 tok/s**，且零重排内存膨胀） |
| **P3** | **CPU 批量 Prefill (`forward_batch` + `smmla`/`sdot` M×N GEMM 微内核)** | CPU Prefill | CPU `pp512` 提升 **5×–7×**（`Llama-1B` `102 -> ~650+ tok/s`），同时为投机解码多 token 校验打底 |
| **P4** | **Metal Decode 融合 QKV 单次 Dispatch + GPU 侧 Greedy Argmax + 预检 `require_fit`** | Metal Decode | 消除 `wk`/`wv` 低带宽尾波（`22 GB/s -> 80+ GB/s`）及 ~250 us 主机同步开销：`Llama-1B` `108.2 -> ~116+ tok/s`、`SmolLM2` `226.4 -> ~250+ tok/s`（**全面反超 `llama.cpp` Metal**） |
| **P5** | **Metal 残差加 + `RMSNorm` 融合（`NORM_MUL_ADD`）+ 短序列 1-simdgroup GQA** | Metal Decode | 32 层小模型每 token 减少 64 次屏障，再省 `~250 us/tok` |


---

## 5. 优化落地与前后实测对比（2026-11-20）

本轮按上表落地了 **P0–P4**：
1. **Metal Tiled FlashAttention (`fa_gqa_f16_kv_batch`, `simdgroup_mma`, `BQ=32, BK=16`, 12 KB TG 内存)**：`gqa_attn_batch` 从 `47.6–49.5 ms` 降至 **`6.46–6.62 ms`**（**7.2×–7.7× 提速**，算力从 `150 GFLOP/s` 升至 `1.16 TFLOP/s`）。
2. **Metal 融合单次 Dispatch `gemv_qkv_q4_0` + 预检 `require_fit` + GPU `argmax_f32`**：`Llama-1B` 的 QKV 耗时从 `1,173.3 us`（48 GB/s）降至 **`720.5 us`（78.6 GB/s，1.63×）**，`SmolLM2` 从 `791.9 us` 降至 **`451.5 us`（1.75×）**。
3. **CPU 基础 ARMv8.0 NEON `fcvtl`/`fcvtn` F16 注意力 + NEON `quantize_q8_{0,1}` + 4 行 `gemv_q4_1_q8_1`**：微基准单线程 `dot_f32_f16` 提速 **11.7×**（`48.12 -> 4.12 us`）、`quantize_q8_0` 提速 **3.8×**（`3.93 -> 1.04 us`）；端到端 CPU `gqa_attn` 从 `1,059.6 / 978.3 us` 降至 **`261.3 / 229.7 us`（4.1×–4.3×）**。
4. **CPU `forward_batch` + 4×8 `smmla` (`i8mm`) 寄存器分块 GEMM**：CPU `pp512` 提速 **2.05×–2.41×**。

| 后端 | 模型 | 指标 | 优化前 | 优化后 | 对比 `llama.cpp` |
|---|---|---|---|---|---|
| **Metal GPU** | `SmolLM2-360M-Q4_0` | `pp512` | 2,133.2 tok/s | **3,500.0 tok/s** (**+64.1%**) | 80.2% |
| | | `tg128` | 223.4 tok/s | **229.8 tok/s** | **99.2%** |
| **Metal GPU** | `Llama-3.2-1B-Q4_0` | `pp512` | 1,072.6 tok/s | **1,351.3 tok/s** (**+26.0%**) | 80.6% |
| | | `tg128` | 106.0 tok/s | **107.5 tok/s** | **95.6%** |
| **CPU (4T)** | `SmolLM2-360M-Q4_0` | `pp512` | 287.7 tok/s | **588.8 tok/s** (**2.05×**) | 32.9% |
| | | `tg128` | 224.8 tok/s | **270.5 tok/s** (**+20.3%**) | **106.3% (`--no-repack`) / 116.1% (默认)** ✅ |
| **CPU (4T)** | `Llama-3.2-1B-Q4_0` | `pp512` | 102.2 tok/s | **246.6 tok/s** (**2.41×**) | 28.9% |
| | | `tg128` | 79.9 tok/s | **84.9 tok/s** (**+6.3%**) | **96.4% (`--no-repack`)** ✅ |
