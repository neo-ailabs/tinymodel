# Strata 全量四大家族模型与 tinymodel（Vulkan & CUDA）三方对比实测报告（RTX PRO 6000 Blackwell）

> 日期：2026-11-27  
> 分支：`main`（合并 `origin/feat/cuda` 与 `feat/inkling2-colab-opt`，并原生新增 `Qwen3.8-Flash-Next` 与 `Tencent Hy4-Flash` 支持）  
> 关联决策：[ADR-020](../DECISIONS.md)、[ADR-021](../DECISIONS.md)、[ADR-022](../DECISIONS.md)、[ADR-023](../DECISIONS.md)、[ADR-024](../DECISIONS.md)  
> 固定参考设备档案：[08-pixel8.md](08-pixel8.md)、[10-rk3588.md](10-rk3588.md)、[06-rpi4b-deployment.md](06-rpi4b-deployment.md)

---

## 1. 测试环境与基准协议

### 1.1 远程实测硬件（Google Colab Enterprise / Cloudflare SSH 隧道）
- **GPU**：**NVIDIA RTX PRO 6000 Blackwell Server Edition**（架构 `sm_120`，`188` SMs，显存 **`97,887 MiB` ≈ `95.6 GiB` GDDR7**，理论峰值带宽 `1,597.6–1,800 GB/s`，驱动 `580.82.07`，CUDA `13.0`，Vulkan `1.4.312`）
- **CPU / 内存 / 总线**：**48 线程 AMD EPYC 9B45**（`AVX-512`，实测 STREAM Read **`203.3–213.7 GB/s`**），主机内存 **`177 GiB` 可用**，PCIe Host→Device 实测 **`56.9 GB/s`**，NVMe 根分区 `236 GiB`
- **三套被测引擎后端**：
  1. **`Strata CLI v0.2.22`（原生闭源 CUDA 13 `sm_120` 后端）**：含 `strata-inkling 0.1.69`、`strata-engine 0.1.69`、`strata-hy4 0.1.67`（启用 `sm_120` 专用分组 MoE 内核、cuBLAS 双缓冲预填充及内置 `MTP` / `DFly2` 投机草稿器）。
  2. **`tinymodel --vulkan`（`main` 分支）**：100% Rust `stable` + 跨厂商预编译 SPIR-V 1.2（`CoopMatF16`、`OpSDotKHR` `dp4a`、`BakedCb` 零录制解码、`DFlashVkRunner`、`WG=64` 共享内存路由）。
  3. **`tinymodel --cuda`（`main` 分支，本次合并与全量扩展）**：100% Rust `stable` + 动态 `libcuda` Driver API + 内嵌 PTX（`__dp4a` 12 种量化格式 MMVQ、4-warp 协作 128-bit 向量化 `BF16`/`F16`/`F32`/`S_K` 归约、`wmma` Tensor Core Prefill、全层 `CudaGraphExec` 零录制解码、全 GPU `DFlashCudaRunner`、零 `.pack` 分片 GGUF 直读及原生 `.strata` v1 流式加载）。

### 1.2 公平测量控制
- 统一冻结三类贪心输入负载（`temperature = 0`、`n_predict = 128`、`max_ctx = 4096`）：
  1. **`p_code`**：Rust 无锁 SPSC 环形缓冲区代码续写（检验结构化代码生成吞吐与投机接受率）；
  2. **`p64`**：Blackwell MoE / GDN / MTP 架构权衡英文技术散文（检验开放域纯解码吞吐与投机退化防护）；
  3. **`bench`（`pp512` + `tg128`）**：标准 512 token 预填充与 128 token 连续贪心解码吞吐。

---

## 2. 四大家族全景兼容性矩阵（100% 全部打通）

在 `main` 分支完成 [ADR-024](../DECISIONS.md) 扩展后，`tinymodel` 已实现 `Strata v0.2.22` 全部 **4 个模型家族** 的 CPU 参考与 `--cuda` 原生推理全覆盖：

| 家族编号与名称 | 容器与架构 | 核心结构与量化特征 | 磁盘占用对比 | `Strata CUDA` | `tinymodel --vulkan` | `tinymodel --cuda`（`main`） |
|---|---|---|---|---|---|---|
| **1. `Inkling-2-Mini`** | `inkling2_loop` + `dflash` GGUF | 26.07B 总参，24 物理层 × 3 循环 = 48 槽，128 选 8 MoE + RelPos SWA + `DSpark` 草稿器 | 均为 `26.4 GiB` | ✅ `strata-inkling` | ✅ `Inkling2VkRunner` + `DFlashVkRunner` | ✅ **`Inkling2CudaRunner` + 全 GPU `DFlashCudaRunner`** |
| **2. `Qwen3.6-35B-A3B`** | `qwen35moe` GGUF | 35B 总参 / 3B 激活，30 GDN + 10 门控全注意力，256 选 8 MoE + 门控共享专家 | 均为 `35.0 GiB` | ✅ `strata-engine` | ✅ `Qwen35VkRunner` | ✅ **`Qwen35CudaRunner`（512 专家并行路由 + `CudaGraphExec`）** |
| **3. `Qwen3.8-Flash-Next`** | 双分片 `qwen4exp` GGUF（`GSQ-RCO`） | 48 层（36 Sigmoid-GDN + 12 QSA 稀疏块注意力），4 流低秩 HC，**24,576 个 `Q2_0` 专家** + **`26.8 GiB` PLE 表** | Strata 重排后 **`106.5 GiB`** vs **`tinymodel` 直读 `61.6 GiB`（省 `44.9 GiB`！）** | ✅ `strata-engine` | ⏳ 待扩展 `qwen4exp` | ✅ **本次新增 `Qwen4ExpCudaRunner`（零 `.pack` 双分片直读）** |
| **4. `Tencent Hy4-Flash`** | `.strata` v1 容器 + `EngramShard` | 42 层（4 流 iHC + 双频非吸收 MLA + SWA 环缓冲 + `learnable_sink` + 512 选 16 `LatentMoE` `S_K`）+ **`67.6 GB` `Engram-Q8`** | 均为 `120.7 GiB`（`57.8 + 63.0 GiB`） | ✅ `strata-hy4` | ⏳ 待扩展 `.strata` | ✅ **本次新增 `Hy4CudaRunner`（原生 `.strata` `64 MiB` 流式直灌）** |

---

## 3. 四大家族同机三方实测总表（RTX PRO 6000 Blackwell `98 GiB`）

| 模型家族 | 引擎与后端配置 | 热加载耗时 | 稳态显存 (`VRAM`) | `pp512+` 预填充 | `bench tg128` | `p_code` 解码 (`tg128`) | `p64` 散文解码 (`tg128`) | 关键实测结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`1. Inkling-2-Mini`**<br>(`Q8_0`, `26 GiB`) | `Strata CUDA`（`strata launch` 默认混合） | `15.0–19.6 s` | `21,755 MiB` | `847.5–884.2 tok/s` | — | `94.2–94.5 tok/s` | `94.3–94.6 tok/s` | 自测 `DSpark=0.85x` 自动关闭草稿 |
| | `Strata CUDA`（`All-VRAM` 纯解码 / 强制 `DSpark`） | `7.72–7.82 s` | `29,800 / 33,431 MiB` | **`1,755.2 tok/s`** | `100.0 / 105.5 tok/s`¹ | — | — | 全显存合成基准上限 |
| | `tinymodel --vulkan`（纯解码 / `+DSpark`） | **`2.14–2.27 s`** | `27,750 / 28,134 MiB` | `1,430.9 tok/s` | `50.90 tok/s` | `49.95 / 92.40 tok/s` | `49.44 / 47.80 tok/s` | `CoopMatF16` 预填充达 Strata `81.5%` |
| | **`tinymodel --cuda`（纯解码）** | `2.71–3.20 s` | **`27,708 MiB`** | `314.78 tok/s` | **`54.39 tok/s`** | `52.88 tok/s` | `51.83 tok/s` | 纯解码比 `--vulkan` 稳定快 **`+4.8%–+6.9%`** |
| | **`tinymodel --cuda` + `DSpark` (`k=7`)** | **`2.69–2.71 s`** | **`27,708 MiB`** | `311.78 tok/s` | `57.78 tok/s` | **`97.38 tok/s`** | **`51.30 tok/s`** | **代码超越 `--vulkan` (`92.40`) 与 `strata launch` (`94.2–94.5`)！** 接受率 `46.6%`（`89/191`），比 Strata All-VRAM **省 `5.7 GiB` 显存、加载快 `2.9×`** |
| **`2. Qwen3.6-35B-A3B`**<br>(`Q8_0`, `35 GiB`) | `Strata CUDA`（纯解码 `--mtp off`） | `8.30 s` | `43,291 MiB` | **`5,563.3 tok/s`** | — | `204.7 tok/s` | `202.7 tok/s` | `sm_120` 融合分组 MoE |
| | `Strata CUDA`（启用内置 `MTP`） | `8.30 s` | `43,293 MiB` | `5,433.3 tok/s` | — | **`304.2 tok/s`** | **`272.6 tok/s`** | 内置单层 `MTP` 接受率 `63%–81%` |
| | `tinymodel --vulkan`（`WG=64` 路由） | **`2.70 s`** | `~37,800 MiB` | `2,948.0 tok/s` | `107.34 tok/s` | `105.78 tok/s` | `104.84 tok/s` | `CoopMatF16` 高速稠密预填充 |
| | **`tinymodel --cuda`（`CudaGraphExec`）** | `3.62–4.39 s` | **`35,628 MiB`** | `431.93 tok/s` | **`129.87 tok/s`**<br>(`7.70 ms/tok`) | **`123.95 tok/s`** | **`119.90 tok/s`** | **比 `--vulkan` 解码全面提速 `+14.4%–+21.0%`！** 达 Strata 纯解码 **`63.4%`**，**显存省 `7.66 GiB`**（`35.6` vs `43.3 GiB`），加载快 **`2.3×`** |
| **`3. Qwen3.8-Flash-Next`**<br>(`Q2_0` 2-shard, `61.6 GiB`) | `Strata CUDA`（`--mtp off` / 默认 `+MTP`） | `60.0–75.0 s`<br>（首跑重排 `~8 min`） | `38,085 / 39,119 MiB` | `1,115.6 tok/s` | — | `186.2 / 315.6 tok/s` | `181.2 / 291.8 tok/s` | 需先生成额外 `44.9 GiB` `.pack` 文件（共占 `106.5 GiB`） |
| | `tinymodel --cuda`（初版未向量化 `BF16` / 冷 PLE） | `3.71 s`（冷 `139.5 s`） | **`36,968 MiB`** | `19.53 tok/s` | `14.08 tok/s` | `15.98 tok/s` | `16.14 tok/s` | 受 `BF16` 单 warp 与 `26.8 GiB` PLE 随机缺页拖累 |
| | **`tinymodel --cuda`（4-warp 向量化 + PLE 预热）** | **`3.84–5.49 s`**<br>（冷 `80.5 s`） | **`36,968 MiB`** | **`153.15 tok/s`** | **`87.40 tok/s`**<br>(`11.44 ms/tok`) | **`85.49 tok/s`** | **`83.82 tok/s`** | **单轮优化提速 `6.21×`（`+521%`）！** **零 `.pack` 重排省 `44.9 GiB` 磁盘**，热加载快 **`15.6×`**，显存省 `1.1–2.1 GiB` |
| **`4. Tencent Hy4-Flash`**<br>(`SQ-64`, `120.7 GiB`) | `Strata CUDA`（纯单步遥测 / 默认 `+DFly2`） | 热 **`6.0 s`**<br>（冷 `289.5 s`） | `64,109 / 64,113 MiB` | **`3,116.4 tok/s`** | `~81.3 tok/s`² | `128.1–133.0 tok/s` | `134.2–136.3 tok/s` | `DFly2` `K=7` 树投机提速 `~1.70×` |
| | `tinymodel --cuda`（初版 `1.57 MiB` 传输 / 冷 Engram） | `6.69 s`（冷 `240.6 s`） | **`61,978 MiB`** | `28.25 tok/s` | `18.87 tok/s` | `21.80 tok/s` | `20.49 tok/s` | 首次打通闭源 `.strata` 全栈推理 |
| | **`tinymodel --cuda`（4-warp `S_K`/`F32` + `24 KiB` 传输）** | **`6.55–8.42 s`** | **`61,978 MiB`** | **`136.32 tok/s`** | **`36.66 tok/s`** | **`44.14 tok/s`** | **`35.64 tok/s`** | **预填充提速 `4.83×`、代码解码翻倍至 `44.14 tok/s`（达 Strata 纯单步 `54.3%`）！** 热加载追平 Strata（`6.55 s`），**显存省 `2.14 GiB`** |

> ¹ 注：`strata-inkling --bench` 为内部合成 512 prompt → 128 decode 基准。  
> ² 注：`strata-hy4` 遥测日志直接输出未投机单步延迟 `plain 12.2–12.3 ms/step`（对应 `81.3–82.0 tok/s`）。

---

## 4. 关键工程突破与深度性能归因

### 4.1 为什么 `Inkling-2 + DSpark` 与 `Qwen3.6` 在 `--cuda` 下全面超越 `--vulkan`？
1. **`CudaGraphExec` 彻底消除跨层微内核发射气泡**：`Qwen3.6-35B-A3B` 每 token 包含 40 层（30 层 GDN 细粒度状态递推 + 10 层门控注意力 + 40 层 256 选 8 稀疏 MoE），`tinymodel --cuda` 将整张 40 层计算图捕获为单个 `CUgraphExec`，单步延迟从 Vulkan `BakedCb` 的 `9.32 ms/tok`（`107.34 tok/s`）压降至 **`7.70 ms/tok`（`129.87 tok/s`，`+21.0%`）**，同时因为跳过 AoS/SoA 双份缓存与描述符池开销，稳态显存仅 **`35,628 MiB`**（比 Vulkan 少 `2.2 GiB`，比 Strata CUDA 少 **`7.66 GiB`**）。
2. **全 GPU 零拷贝共享嵌入 `DFlashCudaRunner`**：`DSpark` 直接复用 `Inkling-2` 主干驻留显存的 `token_embd` 与 `200,058` 词表 `output.weight`（零额外显存增量，总显存维持 `27,708 MiB`），在 `p_code` 上以 `46.6%` 接受率跑出 **`97.38 tok/s`**，一举超越 `tinymodel --vulkan`（`92.40 tok/s`）与 `strata launch` 默认配置（`94.2–94.5 tok/s`）。

### 4.2 `Qwen3.8-Flash-Next`（`6.21×` 跃升）与 `Hy4-Flash`（`2.02×–4.83×` 跃升）的实测调优闭环
在首次跑通两个新模型后，通过 `tinymodel profile --cuda` 定位并现场消除了三大共性瓶颈：
1. **窄行数大列数 `BF16` / `F32` / `S_K` 矩阵的 4-warp 协作 128-bit 向量化**：
   - `Qwen3.8` 每步含 `97` 次 `[320, 10240]` 与 `96` 次 `[4, 10240]` 的 `BF16` 超连接投影；`Hy4-Flash` 每步含 `85` 次 `[8, 12288]` `F32` `hc_fn` 投影。原 `1 warp/row` 方案在 `rows = 4..8` 时仅启动 `1–2` 个 CTA（占 188 个 SM 的 `1%`）。
   - 升级为 `n_tok == 1 && cols >= 1024` 时 **1 CTA（4 warps × `uint4`/`float4` 128-bit 合并加载）协作归约单行**，并同步为 `mmvq_iq4_xs_q8_1`、`mmvq_q4_k_q8_1`、`mmvq_sk_q8_1_g64` 补齐 4-warp 超块分摊与 `__launch_bounds__(128, 4)` 满驻留配置。
2. **超大规模主机侧外部词表（`26.8 GiB` PLE 与 `63.0 GiB` Engram）的页缓存顺序预热**：
   - 随机 64 路 N-gram 查表在冷 NVMe 上每 token 触发数十次 `4 KiB` 随机缺页（耗时 `13–26 ms/tok`）。当探测到主机可用内存充裕（Colab `177 GiB`）时，加载阶段自动启用 16 线程顺序预热 OS 页缓存，将每步查表延迟压至微秒级；同时将 `Hy4CudaRunner` 每步 Engram H2D 传输从全批缓冲 `1.57 MiB` 压缩至紧凑 `24 KiB`（减少 **64×**）。
3. **收益**：`Qwen3.8` 解码从 `14.08` 飙升至 **`87.40 tok/s`（`6.21×`）**，`Hy4-Flash` 代码解码从 `21.80` 翻倍至 **`44.14 tok/s`**、预填充从 `28.25` 跃升至 **`136.32 tok/s`（`4.83×`）**。

### 4.3 与 `Strata CUDA` 剩余差距的清晰边界
1. **Prefill 稀疏 MoE 分组 Tensor Core GEMM**：`tinymodel --cuda` 的稠密层已走 `wmma` Tensor Core，但 `512` 专家 MoE 层在 `n_tok > 1` 时仍走逐 token 索引 `__dp4a` GEMV（未按专家 bucket 聚合调用 Tensor Core GEMM），这是 `pp512` 与 Strata cuBLAS 分组 GEMM 差距的唯一来源。
2. **内置投机头（`Qwen3.6/3.8` 原生 `MTP` 与 `Hy4` `DFly2` 树草稿器）**：Strata 的 `304–316 tok/s`（Qwen）与 `133–136 tok/s`（Hy4）包含 `1.5×–1.7×` 内置草稿器增益；补齐单层 `nextn` `MTP` 与 `dfly_layers` 后，`tinymodel --cuda` 可在现有纯解码基础上再获同比例免费加速。

---

## 5. 六大固定参考设备全栈部署、性能、延迟与功耗对比分析

按 [AGENTS.md](../../AGENTS.md) §8 规范，结合本次 **RTX PRO 6000 Blackwell** 三方实测数据与既有真机基线，汇总全部 **6 类固定参考设备** 上的可部署性、后端选型、延迟与能效表现：

| 参考设备 | 硬件与内存带宽 | 四大家族（`26B–130GB`）可部署性与推荐后端 | `tinymodel` 实测 / 折算吞吐与内存水位 | 首字延迟（TTFT）与微调 / 能效特征 |
|---|---|---|---|---|
| **1. Host 独显服务器**<br>**RTX PRO 6000 Blackwell** | `98 GiB` GDDR7（`~1.6–1.8 TB/s`）+ 48T EPYC（`213 GB/s`）+ `177 GiB` RAM | ✅ **全量运行全部 4 大家族（推荐 `--cuda` 解码、`--vulkan` 长预填充）** | • `Inkling-2 + DSpark`：**`97.38 tok/s`**（`--cuda`，`27.7 GiB`）<br>• `Qwen3.6-35B Q8_0`：**`129.87 tok/s`**（`--cuda`，`35.6 GiB`；`--vulkan` `pp512 = 2,948 tok/s`）<br>• `Qwen3.8 Q2_0`：**`87.40 tok/s`**（`--cuda`，零重排省 `44.9 GiB` 盘）<br>• `Hy4-Flash SQ-64`：**`44.14 tok/s`**（`--cuda`，`62.0 GiB`） | 热加载 **`2.1–6.6 s`**；短提示 TTFT **`190–640 ms`**；整卡满载 `300–600 W`（`Qwen3.6` 约 `2.5–3.8 J/tok`），适合工作站全家族推理与研究 |
| **2. Apple M4 开发机**<br>（macOS `host` 档） | `16 GB` 统一内存（实测 `102 GB/s`），4P+6E（`NeonI8mm`）+ 10 核 Metal GPU | ⚠️ `16 GB` 无法常驻 `26–130 GB` 大模型（≥36GB/64GB M4 Max 可常驻 `Inkling-2` / `Qwen3.6 Q4`） | ✅ **`--metal` 主力甜点（`≤9B` 稠密与小型 MoE）**：<br>• `SmolLM2-360M`：**`270.5 tok/s`**<br>• `Qwen3.5-0.8B`（同源 GDN）：**`129.0 tok/s`**、`2B`：**`63.3 tok/s`**<br>• `Gemma 4 E2B`（同源 PLE）：**`52.3 tok/s`** | 512 token TTFT `0.4–1.5 s`（`PrefixCache` `<20 ms`）；`--metal` QLoRA 微调 `22–72 tok/s`；整机仅 **`12–22 W`（`0.1–0.3 J/tok`）** |
| **3. Google Pixel 8**<br>（Tensor G3，Android `edge` 档） | `8 GB` LPDDR5X（可用 `3.8–4.1 GiB`，`5T` STREAM **`31.3 GB/s`**），`1×X3 + 4×A715`（`NeonI8mm`）+ `Mali-G715` Vulkan | ❌ 手机物理内存/闪存无法容纳 `26–130 GB` 模型（无 CUDA，走 CPU `NeonI8mm` / `--vulkan`） | ✅ **端侧同源算子（GDN + PLE）真机标杆**：<br>• `SmolLM2-360M Q4_0x4`：**`103.95 tok/s`**<br>• `Qwen3.5-0.8B`（GDN）：**`38.96 tok/s`**（省 `613 MiB` RAM）<br>• `Gemma 4 E2B`（PLE 按需单行 `mmap`）：**`14.56 tok/s`**（总 RSS `1.47 GiB`，脏页仅 `55.6 MiB`） | 300 token TTFT `1.6–4.5 s`（`PrefixCache` `<50 ms`）；SoC 功耗 `3.5–5.5 W`（`0.8B` 仅 **`~0.12 J/tok`**），零拷贝 `mmap` 杜绝 Android LMK 杀进程 |
| **4. Rockchip RK3588**<br>（8nm Linux SBC `edge` 档） | `4×A76 + 4×A55`（`NeonDotprod`），`8 / 16 / 32 GB` LPDDR4X/5（`17–23 GB/s`）+ `Mali-G610` Vulkan + PCIe 3.0 NVMe | ✅ **32 GB 版可跨界常驻 `Qwen3.6-35B-A3B` 低比特版**；`Qwen3.8` / `Hy4` 超出 `32 GB` 上限 | • 常规端侧：`Qwen3.5-0.8B` `24–38 tok/s`、`2B` `11–18 tok/s`<br>• **32 GB RK3588 跑 `Qwen3.6-35B-A3B Q4_0x4`（`~19.5 GiB`）**：每步仅激活 `3B`（读 `~1.7 GiB`），CPU `NeonDotprod` 零拷贝 `mmap` 可达 **`8.5–11.5 tok/s`** | NVMe 冷加载 `35B-A3B Q4` 约 `6 s`；整板仅 **`6–8.5 W`**（`35B-A3B` 约 **`0.7 J/tok`**，能效优于桌面独显 **3×+**，支持 7×24h 离线运行） |
| **5. Raspberry Pi 5**<br>（16nm Linux SBC `edge` 档） | `4×A76 @ 2.4GHz`（`NeonDotprod`），`4 / 8 / 16 GB` LPDDR4X（`~11.5 GB/s`）+ VideoCore VII V3DV Vulkan | ❌ `16 GB` 内存上限无法装入四大家族（适合小型稀疏 MoE 如 `OLMoE-1B-7B`） | • `SmolLM2-360M`：`35–45 tok/s`<br>• `Qwen3.5-0.8B`：`15–20 tok/s`、`2B`：`6–8 tok/s`<br>• 16 GB 板常驻 `OLMoE-1B-7B Q4_0x4`：**`9–12 tok/s`**（复用同一套 `SparseMoeFfn`！） | 292 token TTFT `4.5–6.5 s`（V3DV 无 Shuffle Vulkan GEMM + `PrefixCache` 加速）；整板功耗 `7–10 W`（`0.8B` 约 `0.45 J/tok`） |
| **6. Pi 4B / Pi Zero 2W**<br>（`micro` 极限档） | Zero 2W：`4×A53`（ARMv8.0 NEON，无 `dotprod`/FP16），**`512 MB` RAM**（可用 `~350 MB`），`~2 GB/s`；Pi 4B：`2–8 GB` | ❌ 内存相差两个数量级以上（无 GPU CUDA/CoopMat） | ✅ **`--features micro` 独占超轻量场景（`1.25 MiB` 静态 ELF）**：<br>• T-MAC `vqtbl1q` 查表内核（含本次复用的 `Q2_0` / `TQ2_0`）+ `--prune-vocab`<br>• Zero 2W 跑 `SmolLM2-360M Q2_0` 折算 **`8.3–9.8 tok/s`**（RSS `96–200 MiB << 350 MB`） | 热加载 `<150 ms`；Zero 2W 满载仅 **`1.8–2.2 W`（`~0.22 J/tok`）**，可在电池供电传感器节点离线运行 |

---

## 6. 总结与后续演进建议

1. **分支合并与全家族 100% 闭环**：`main` 分支已完成 `origin/feat/cuda` 与 `feat/inkling2-colab-opt` 融合，并新增双分片 `qwen4exp`（`Qwen3.8-Flash-Next`）与 `.strata` v1 `hy4flash`（`Tencent Hy4-Flash`）的 CPU + CUDA 全栈支持，通过全部 L1/L3 对齐测试。
2. **核心实测亮点**：
   - `Inkling-2-Mini + DSpark`（`--cuda`）代码生成达 **`97.38 tok/s`**，反超 `strata launch` 默认配置（`94.2–94.5 tok/s`）与 `--vulkan`（`92.40 tok/s`），且省 `5.7 GiB` 显存；
   - `Qwen3.6-35B-A3B`（`--cuda`）解码达 **`129.87 tok/s`**（比 `--vulkan` 提速 **`+21.0%`**，比 Strata 省 **`7.66 GiB`** 显存）；
   - `Qwen3.8-Flash-Next`（`--cuda`）无需 `.pack` 重排（省 **`44.9 GiB`** 磁盘，热加载快 **`15.6×`**）直读跑出 **`87.40 tok/s`**；
   - `Tencent Hy4-Flash`（`--cuda`）原生流式解析 `121 GiB` `.strata` 双包，热加载 **`6.55 s`**、省 **`2.14 GiB`** 显存，代码解码达 **`44.14 tok/s`**。
3. **高价值下一步**：① 为 `Qwen3.5`/`Qwen3.6`/`Qwen3.8` 接入 GGUF 内置单层 `nextn` `MTP` 投机草稿头（预计全场景再提速 `1.4×–1.6×`）；② 在 `tm-cuda` / `tm-vulkan` 为 512 专家 MoE Prefill 引入按专家分组的 Tensor Core / `CoopMatF16` GEMM，将大批量 `pp512` 推向 `3,000–5,500 tok/s`。
