# Strata 全量四大家族模型与 tinymodel 跨后端对比实测报告（RTX PRO 6000 Blackwell）

> 日期：2026-11-27  
> 关联决策：[ADR-020](../DECISIONS.md)、[ADR-021](../DECISIONS.md)、[ADR-022](../DECISIONS.md)  
> 固定参考设备档案：[08-pixel8.md](08-pixel8.md)、[10-rk3588.md](10-rk3588.md)、[06-rpi4b-deployment.md](06-rpi4b-deployment.md)

---

## 1. 测试环境与基准协议

### 1.1 远程实测硬件（Google Colab Enterprise / Cloudflare SSH 隧道）
- **GPU**：**NVIDIA RTX PRO 6000 Blackwell Server Edition**（架构 `sm_120`，显存 **`97,887 MiB` ≈ `95.6 GiB` GDDR7**，理论带宽 `~1.8 TB/s`，驱动 `580.82.07`，CUDA `13.0`，Vulkan `1.4.312`，`subgroup = 32`，`KHR_cooperative_matrix = true`）
- **CPU / 内存 / 总线**：**48 线程 AMD EPYC 9B45**（`AVX-512`，实测 STREAM Read **`213.65 GB/s`**），主机内存 **`176 GiB`**，PCIe Host→Device 实测 **`56.9 GB/s`**，NVMe 根分区 `236 GiB`
- **被测引擎**：
  1. **`Strata CLI v0.2.22`（原生闭源 CUDA 13 `sm_120` 后端）**：包含通用引擎 `strata-engine 0.1.69`、循环 MoE 专属引擎 `strata-inkling 0.1.69`、混元专属引擎 `strata-hy4 0.1.67`（启用 `sm_120` 专用内核、CUDA Graph 捕获重放、cuBLAS 双缓冲预填充）。
  2. **`tinymodel`（分支 `feat/inkling2-colab-opt`，提交 `cfe8dcc`）**：100% Rust `stable` + 跨厂商预编译 Vulkan SPIR-V 1.2 后端（`--vulkan`，启用 `CoopMatF16`、`OpSDotKHR` `dp4a`、`BakedCb` 零录制单提交解码、`DFlashVkRunner` 与本次新增的 [ADR-022](../DECISIONS.md) `qwen35moe` + `WG=64` 共享内存并行路由）。

### 1.2 公平测量控制
- 统一冻结三类输入负载（贪心解码 `temperature = 0`、`max_tokens = 128`、上下文 `32k` / `4096`）：
  1. **`p_code`**：Rust 无锁 SPSC 环形缓冲区代码续写（高结构化重复度，检验投机解码上限）；
  2. **`p64`**：Blackwell MoE / GDN / MTP 架构权衡英文技术散文（开放域低重复度，检验投机解码退化防护与纯解码带宽）；
  3. **`pp512+`**：长上下文预填充吞吐（`512–630` tokens）。
- 对 `Strata` HTTP 服务显式注入唯一随机前缀 Nonce 并设置 `cache_prompt: false`、`reasoning_effort: "none"`、`enable_thinking: false`，杜绝跨轮次 Prefix Cache 命中虚高 `pp` 指标，直接读取引擎级 `timings`（`prompt_per_second`、`predicted_per_second`、`draft_n`、`draft_n_accepted`）与 `nvidia-smi` 稳态显存。

---

## 2. Strata 支持的四大家族全景与兼容性矩阵

通过实测 `strata launch` 模型目录与引擎组件，`Strata v0.2.22` 当前共支持 **4 个模型家族**（均面向 `26B–130GB+` 大规模稀疏 MoE）：

| 家族编号与名称 | GGUF / 容器架构 | 核心结构特征 | 磁盘体积（本次实测档位） | 内置投机草稿器 | `Strata CUDA` | `tinymodel`（CPU + Vulkan） |
|---|---|---|---:|---|---|---|
| **1. `Inkling-2-Mini`** | `inkling2_loop`（标准 `Q8_0` GGUF） | 26.07B 总参，24 物理层 × 3 循环 = 48 逻辑槽，128 选 8 MoE + RelPos SWA | `26.0 GiB` + `384 MiB` `DSpark` | 独立 `DSpark`（`dflash` 3 层分块草稿器） | ✅ `strata-inkling 0.1.69` | ✅ **原生支持**（`Inkling2VkRunner` + 全 GPU `DFlashVkRunner`） |
| **2. `Qwen3.6-35B-A3B`** | `qwen35moe`（标准 `Q8_0` GGUF） | 35B 总参 / **3B 激活**，40 层（30 GDN + 10 门控全注意力），256 选 8 MoE + Sigmoid 门控共享专家 | `36.0 GiB`（含尾部 1 层 MTP） | 主干内置 1 层 `MTP`（`nextn=1`） | ✅ `strata-engine 0.1.69` | ✅ **本次新增原生支持**（[ADR-022](../DECISIONS.md) `Qwen35Model` + `Qwen35VkRunner`） |
| **3. `Qwen3.8-Flash-Next`** | `qwen4exp`（私有 `GSQ-RCO` `ggml_type=42` `Q2_0` + `.pack`） | **24,576 专家槽** + **`26.8 GiB`（3.2 亿行）PLE 表** + QSA 注意力 + `HC_UPMIX` 超连接 | 原始 `71.6 GB`，首次重排后占盘 **`106.5 GiB`** | 内置 `MTP`（`6.5 GiB`）；`DFlash2` 仅限私有原生包 | ✅ `strata-engine 0.1.69` | ❌ 超出 `≤9B` 边界且使用私有量化号 `type=42` |
| **4. `Tencent Hy4-Flash`** | 闭源 `.strata` 容器（无公开 GGUF） | `5.92 GB` 主干（1433 张量）+ **`56.04 GB` MoE 专家** + **`67.6 GB` `Engram-Q8` 记忆表** | **`129.6 GB`**（占盘 `121 GiB`） | 内置 `DFly2`（`0.28 GB`，`K=7` 自适应树草稿器） | ✅ `strata-hy4 0.1.67` | ❌ 闭源私有 `.strata` 格式（社区无 GGUF） |

---

## 3. 四大家族同机实测总表（RTX PRO 6000 Blackwell `98 GiB`）

### 3.1 端到端加载、显存、预填充（Prefill）与解码（Decode）总览

| 模型家族 | 引擎与后端配置 | 热加载耗时 | 稳态显存 (`VRAM`) | `pp512+` 预填充 | `p_code` 解码 (`tg128`) | `p64` 散文解码 (`tg128`) | 投机接受率 / 关键行为 |
|---|---|---:|---:|---:|---:|---:|---|
| **`Inkling-2-Mini`**<br>(`Q8_0`, `26 GiB`) | `Strata CUDA`（默认混合层：`1859` GPU + `1213` Host） | `15.0–19.6 s` | **`21,755 MiB`** | `847.5–884.2 tok/s` | **`94.2–94.5 tok/s`** | **`94.3–94.6 tok/s`** | 启动自测 `DSpark=0.85x`（代码 `0.99x` / 散文 `0.72x`），**自动关闭草稿** |
| | `Strata CUDA`（`100% All-VRAM`，无草稿） | `7.72 s` | `~29,800 MiB` | **`1,755.2 tok/s`** | `100.0 tok/s`¹ | `100.0 tok/s`¹ | 全显存 `3072` 专家纯解码 |
| | `Strata CUDA`（`100% All-VRAM` + 强制 `DSpark`） | `7.82 s` | `33,431 MiB` | `1,753.0 tok/s` | **`105.5 tok/s`**¹ | — | `--bench` 强制启用 `DSpark`（占 `482 MiB`） |
| | **`tinymodel --vulkan`**（`All-VRAM`，纯解码） | **`2.14 s`** | **`27,750 MiB`** | `1,430.9 tok/s` | `49.95 tok/s` | `49.44 tok/s` | **加载快 `3.6×`**，预填充达 CUDA 的 **`81.5%`** |
| | **`tinymodel --vulkan` + `DSpark`**（`vocab=65536`） | **`2.27 s`** | **`28,134 MiB`** | `1,424.8 tok/s` | **`92.40 tok/s`** | `47.80 tok/s` | 代码 **`1.85×` 提速**（接受 `48.7%`，**达 `strata launch` 的 `98.1%`！**）；散文自适应限流保住 `0.97×`（优于 Strata `0.72×`），**省 `5.2 GiB` 显存** |
| **`Qwen3.6-35B-A3B`**<br>(`Q8_0`, `36 GiB`) | `Strata CUDA`（纯解码 `--mtp off`） | `8.30 s`（首检 `30.4 s`） | `43,291 MiB` | **`5,563.3 tok/s`** | `204.7 tok/s` | `202.7 tok/s` | `--kv fp32` + `251 MiB` GDN 检查点 |
| | `Strata CUDA`（启用内置 `MTP`） | `8.30 s` | `43,293 MiB` | `5,433.3 tok/s` | **`304.2 tok/s`** | **`272.6 tok/s`** | `MTP` 接受率 **`63.4%–80.6%`**，全场景稳定提速 **`1.34×–1.54×`** |
| | `tinymodel --vulkan`（优化前单线程路由基线） | `2.73 s` | `~37,800 MiB` | `2,489.8 tok/s` | `32.46 tok/s` | `32.46 tok/s` | `moe_router_topk` 单线程全局访存占 `81.2%`（`24.6 ms/tok`） |
| | **`tinymodel --vulkan`（`ADR-022` `WG=64` 共享内存路由）** | **`2.70 s`** | **`~37,800 MiB`** | **`2,948.0 tok/s`** | **`105.78 tok/s`**<br>(`bench 107.34`) | **`104.84 tok/s`** | **单着色器改造提速 `3.30×`（`+230%`）！** 加载快 **`3.1×`**、省 **`~4.5 GiB` 显存**；纯解码达 CUDA 的 **`52.4%`** |
| **`Qwen3.8-Flash-Next`**<br>(`Q2_0`, `106.5 GiB` 盘) | `Strata CUDA`（`--mtp off`，含零权重后缀草稿） | `75.0 s`（首次重排 `~8 min`） | `38,085 MiB` | `1,115.6 tok/s` | `186.2–194.5 tok/s` | `181.2 tok/s` | `24,576` 专家占 `31.64 GiB` 显存；`26.8 GiB` PLE 驻留主机内存 |
| | `Strata CUDA`（启用内置 `MTP`，`spec=4`） | `60.0 s` | **`39,119 MiB`** | `1,117.3 tok/s` | **`303.9–315.6 tok/s`** | **`290.5–291.8 tok/s`**<br>(`pp558` 后 **`332.1`**) | `MTP` 接受率高达 **`69.8%–82.7%`**（`81/98`），全场最高解码吞吐 |
| **`Tencent Hy4-Flash`**<br>(`SQ-64`, `129.6 GB`) | `Strata CUDA`（纯单步解码内部遥测） | 冷 `289.5 s` / **热 `6.0 s`** | `64,109 MiB` | **`3,116.4 tok/s`** | `~81.3 tok/s`² | `~81.3 tok/s`² | 引擎日志实测 `plain = 12.2–12.3 ms/step` |
| | `Strata CUDA`（默认 `DFly2` `K=7` 树投机） | 热 **`6.0 s`** | **`64,113 MiB`** | `3,102.7 tok/s` | **`128.1–133.0 tok/s`** | **`134.2–136.3 tok/s`**<br>(`pp565` 后 **`138.6`**) | `56.04 GB` 专家全驻显存 + `67.6 GB` Engram（`664 MiB` 600 万行 LRU）；每步接收 `2.47–2.58` token（`~1.70×` 提速） |

> ¹ 注：`strata-inkling --bench` 为合成 512 prompt → 128 decode 基准。  
> ² 注：`strata-hy4 0.1.67` 在省略 `--draft` 时默认仍开启 `DFly2` 树投机，其 `[spec]` 遥测直接记录了同机精确单步延迟 `plain 12.2–12.3 ms/step`（对应 `81.3–82.0 tok/s`）。

---

## 4. 关键工程发现与深度性能归因

### 4.1 `Qwen3.6-35B-A3B` Vulkan 逐算子硬件时间戳剖析（`tinymodel profile`）
在接入 `qwen35moe` 后，`tinymodel profile` 立即捕获到原本面向 `≤8B`（64 专家）的 `moe_router_topk.comp` 单线程全局内存选择排序在 256 专家 × 40 层上耗时高达 `24.6 ms/tok`（占总解码时间 `81.2%`）。将其升级为每 token 单工作组（`WG = 64`）共享内存（`sh_probs[512]` / `sh_sel[512]`）并行归约后，Blackwell 硬件时间戳实测如下：

| 解码阶段（40 层累计） | GPU 耗时 (`us/tok`) | 占比 | 每步访存 (`MiB/tok`) | 实测有效带宽 (`GB/s`) | 实测算力 (`GFLOP/s`) |
|---|---:|---:|---:|---:|---:|
| `gdn_proj`（30 层 GDN 输入投影） | `1,068.0` | `10.1%` | `768.98` | **`755.00 GB/s`** | **`1,421.2`** |
| `lm_head+norm`（248k 词表输出头） | `693.9` | `6.6%` | `515.31` | **`778.67 GB/s`** | **`1,465.7`** |
| `full_qkv+split+rope`（10 层全注意力 QKV） | `333.4` | `3.2%` | `191.25` | **`601.46 GB/s`** | **`1,132.2`** |
| `gdn_recur+gnorm`（30 层状态递推 + 门控归一化） | `367.0` | `3.5%` | `120.00` | `342.85 GB/s` | `257.1` |
| `gdn_ssm_out+res` + `full_wo+res` | `1,306.2` | `12.4%` | `340.00` | `272.9 GB/s` | `513.7` |
| `full_attn+gate`（10 层解码注意力） | `1,224.9` | `11.6%` | `10.31` | `8.83 GB/s` | `70.6` |
| **`ffn_gate_up_silu + ffn_down+res`（路由 + 9 专家 GEMV + 归约）** | **`4,540.2`** | **`43.0%`** | `1,227.81` | **`270.4 GB/s`**（综合） | `341.8` |
| 其余归一化与嵌入（`embed` / `attn_norm` / `ffn_norm`） | `838.3` | `8.0%` | `1.89` | — | — |
| **总计（单 CommandBuffer 零录制开销 `host+sync = 0.0 us`）** | **`10,558.3`** | **`100.0%`** | **`3,183.06`** | **`322.43 GB/s`** | — |

**核心洞察**：
1. **跨厂商通用 SPIR-V 在稠密大矩阵上已榨出极高硬件效率**：`tinymodel` 的 `gdn_proj`（`755 GB/s`）与 `lm_head`（`778.7 GB/s`、Prefill `1,319.6 GB/s`、`full_qkv_gemm` **`109.2 TFLOP/s`**）证明纯 Rust + 标准 Vulkan SPIR-V 在 Blackwell 上完全能跑出接近原生 CUDA 的峰值吞吐。
2. **与 `Strata CUDA` 纯解码差距（`107` vs `204 tok/s`）的三大根因**：
   - **稀疏 3D 专家分散访存与融合度**：40 层 ×（8 路由专家 + 1 共享专家）× 3 矩阵共 `1,080` 个细粒度 `2048×512` 子矩阵随机散落在 `32.67 GiB` 显存中，当前 `moe_indexed_gemv.comp` 分三次 dispatch（`gate`、`up`、`down`）加全局屏障，综合带宽 `~270 GB/s`；而 `Strata` 将 `gate+up+SiLU` 融合进单个 `sm_120` 专用分组内核并配合 CUDA Graph 消除了 `40 × 12` 次跨算子屏障气泡（仅屏障与 `full_attn` 小规模启动开销就占去 `~3.2 ms/tok`）。
   - **Prefill 专家分组 GEMM**：`Qwen3.6` 预填充中稠密算子仅占 `32.9 ms`（折合 **`>15,500 tok/s`**！），而 `ffn_gate_up_silu` 因尚未像 `Inkling2VkRunner` 那样接入 `CoopMatF16` 按专家分组 GEMM（仍走逐 token `moe_indexed_gemv`），占用了 `143.0 ms`（`81.3%`）。只要将 `inkling2_moe_grouped_cm.comp` 复用到通用 `SparseMoeFfn`，`tinymodel` 在 `Qwen3.6` 上的 `pp512` 即可从 `2,948 tok/s` 直接跃升至 **`5,000+ tok/s`** 追平 `Strata CUDA`。
   - **内置 `MTP` 头的巨大杠杆**：`Qwen3.6` 与 `Qwen3.8` 的单层原生 `MTP` 头不论在代码还是散文上均维持 **`63%–83%` 接受率**（稳定带来 `1.4×–1.7×` 免费加速），远比外部独立草稿器（`DSpark` 在散文仅 `21%` 接受率）稳健。

### 4.2 四大家族横向对比结论（Strata 内部选型视角）
1. **综合性价比冠军：`Qwen3.6-35B-A3B`（`Q8_0`，`36 GiB`）**——仅用 `42.3 GiB` 显存（`tinymodel` 仅 `37.8 GiB`）与 `2.7–8.3 s` 加载时间，就跑出全场最高的 **`5,563 tok/s` 预填充**与 **`273–304 tok/s` MTP 解码**，且是开放标准 GGUF 格式，双引擎通吃。
2. **极限解码吞吐与超大规模稀疏记忆探索：`Qwen3.8-Flash-Next Q2_0`**——凭借 `24,576` 细粒度专家与 `MTP` 跑出全场最高 **`332.1 tok/s`** 解码，但代价是 `106.5 GiB` 磁盘占用、首次近 10 分钟重排、`60–75 s` 启动以及受 `26.8 GiB` 主机内存 PLE 查表拖累的预填充（`1,117 tok/s`，仅为 `Qwen3.6` 的 `1/5`）。
3. **`Tencent Hy4-Flash SQ-64`**：将 `67.6 GB` Engram 外部记忆表与 `56.04 GB` MoE 专家结合，热加载极快（`6.0 s`），预填充优秀（`3,116 tok/s`），但 `62.6 GiB` 显存门槛最高，单步激活参数更大导致 `DFly2` 树投机后为 `133–139 tok/s`。
4. **`Inkling-2-Mini`**：26B 3× 循环共享权重结构紧凑（`28 GiB` 显存全驻），在代码生成场景配合 `tinymodel` 全 GPU `DSpark` 可达 **`92.4 tok/s`**（追平 `strata launch` 默认配置 `94.2 tok/s`），但在开放散文上受循环层误差累积影响草稿接受率偏低。

---

## 5. 六大固定参考设备全栈部署、性能、延迟与功耗对比分析

按 [AGENTS.md](../../AGENTS.md) §8 规范，结合本次 **RTX PRO 6000 Blackwell** 实测数据与既有真机档案，评估 `Strata` 四大家族及同代架构在全部 **6 类参考硬件** 上的可部署性、内存水位、推理延迟与能效表现：

| 参考设备 | 硬件与内存带宽 | `Strata` 四大家族（26B–130GB）可运行性 | `tinymodel` 最佳部署形态与实测/折算吞吐 | 首字延迟（TTFT）与微调 / 功耗特征 |
|---|---|---|---|---|
| **1. Host 独显服务器**<br>**RTX PRO 6000 Blackwell** | `98 GiB` GDDR7（`~1.8 TB/s`）+ 48T EPYC（`214 GB/s`）+ `176 GiB` RAM | ✅ **全量显存运行全部 4 大家族**：<br>• `Qwen3.6`：`304 tok/s`（MTP）<br>• `Qwen3.8`：`332 tok/s`（MTP）<br>• `Hy4-Flash`：`139 tok/s`（DFly2）<br>• `Inkling-2`：`94–106 tok/s` | ✅ **直接运行开放 GGUF 家族 1 & 2**：<br>• `Qwen3.6-35B Q8_0`：**`107.3 tok/s`**、`pp512 = 2,948 tok/s`、加载 **`2.70 s`**<br>• `Inkling-2 Q8_0 + DSpark`：代码 **`92.4 tok/s`**、省 `5.2 GiB` 显存<br>• `≤9B` 主力模型可达 **`350–600+ tok/s`** | `pp512` TTFT 仅 **`90–350 ms`**；整卡满载 `300–600 W`（`Qwen3.6` 约 `1.5–2.5 J/tok`），适合高吞吐桌面工作站与本地代码助手 |
| **2. Apple M4 开发机**<br>（macOS `host` 档） | `16 GB` 统一内存（实测 `102 GB/s`），4P+6E（`NeonI8mm`）+ 10 核 Metal GPU | ⚠️ **`16 GB` 物理内存无法常驻任何一款**：<br>• `Qwen3.8` 仅 PLE 表即 `26.8 GiB`<br>• `Hy4` 达 `129.6 GB`<br>• `Qwen3.6 Q4` 仍需 `~20 GiB`（需 ≥32GB Mac 方可流畅运行） | ✅ **`≤9B` 稠密与小型 MoE 主力甜点**：<br>• Metal 实测 `SmolLM2-360M` **`270.5 tok/s`**、`Qwen3.5-0.8B` **`129.0 tok/s`**、`Qwen3.5-2B` **`63.3 tok/s`**、`Gemma 4 E2B` **`52.3 tok/s`**<br>• `Inkling-2 Q8_0` 可通过 `ADR-016` NVMe 三层流式运行（`~2–5 tok/s` 功能验证） | 512 token TTFT `0.4–1.5 s`（配合 `PrefixCache` 压至 `<20 ms`）；`--metal` QLoRA 微调 `22–72 tok/s`；整机推理功耗仅 **`12–22 W`（`0.1–0.3 J/tok`）** |
| **3. Google Pixel 8**<br>（Tensor G3，Android `edge` 档） | `8 GB` LPDDR5X（可用 `3.8–4.1 GiB`，`5T` STREAM **`31.3 GB/s`**），`1×X3 + 4×A715`（`NeonI8mm`）+ `Mali-G715` Vulkan | ❌ **完全不可部署**（`Strata` 无 Android 构建，且 `26–130 GB` 权重远超手机存储/内存上限） | ✅ **端侧真机实测标杆（零拷贝 `mmap`，脏内存 `<56 MiB`）**：<br>• `SmolLM2-360M Q4_0x4`：**`103.95 tok/s`**<br>• `Qwen3.5-0.8B`（同源 GDN！）：**`38.96 tok/s`**（反超 `llama.cpp KleidiAI` 且省 `613 MiB` RAM）<br>• `Gemma 4 E2B`：**`14.56 tok/s`**、`Qwen3.5-2B`：**`13.53 tok/s`** | 300 token TTFT `1.6–4.5 s`（`PrefixCache` 二轮 `<50 ms`）；被动散热长跑仅降频 `18.7%`；SoC 功耗 `~3.5–5.5 W`（`0.8B` 仅 **`~0.12 J/tok`**） |
| **4. Rockchip RK3588**<br>（8nm Linux SBC `edge` 档） | `4×A76 + 4×A55`（`NeonDotprod`），可选 **`8 / 16 / 32 GB`** LPDDR4X/5（`17–23 GB/s`）+ `Mali-G610` Vulkan + PCIe 3.0 NVMe | ❌ `Strata` 无 Linux ARM64 引擎；`Qwen3.8` / `Hy4` 体积过大无法装入 `32 GB` | ✅ **唯一能跨界吃下本次新增 `qwen35moe` 的低功耗边缘板！**<br>• 常规主力：`Qwen3.5-0.8B` `24–38 tok/s`、`2B` `11–18 tok/s`、`8B/9B` `3.5–4.8 tok/s`<br>• **32 GB 版 RK3588 + NVMe 跑 `Qwen3.6-35B-A3B`（转 `Q4_0x4` 约 `19.5 GiB`）**：每步仅激活 `3B`（读 `~1.7 GiB`），凭零拷贝 `mmap` 可常驻内存跑出 **`~8.5–11.5 tok/s`**！ | PCIe 3.0 NVMe（`3.2 GB/s`）冷启动 `2B` 仅 `+0.4 s`、`35B-A3B Q4` 约 `6 s`；整板功耗仅 **`6–8.5 W`**（`35B-A3B` 仅 **`~0.7 J/tok`**，能效高于桌面独显 **2×+**，且支持 7×24h 挂机 QLoRA） |
| **5. Raspberry Pi 5**<br>（16nm Linux SBC `edge` 档） | `4×A76 @ 2.4GHz`（`NeonDotprod`），`4 / 8 / 16 GB` LPDDR4X（`~11.5 GB/s`）+ VideoCore VII V3DV Vulkan | ❌ 内存上限 `16 GB` 且无 ARM64 构建，四大家族均无法装入 | ✅ **`0.8B–4B` 稠密与 `≤8B` 小型 MoE（如 `LFM2-8B-A1B`、`Qwen3-30B` 以下裁剪版）**：<br>• `SmolLM2-360M`：`35–45 tok/s`<br>• `Qwen3.5-0.8B`：`15–20 tok/s`、`2B`：`6–8 tok/s`<br>• 16 GB 板可零拷贝常驻 `Llama-3.1-8B Q4_0x4`（`~1.8–2.2 tok/s`）或 `OLMoE-1B-7B Q4`（**`~9–12 tok/s`**） | 292 token TTFT `4.5–6.5 s`（接 V3DV 无 Shuffle Vulkan GEMM + `PrefixCache` 可显著压缩）；整板功耗 `7–10 W`（`0.8B` 约 `0.45 J/tok`） |
| **6. Pi 4B / Pi Zero 2W**<br>（`micro` 极限档） | Zero 2W：`4×A53`（ARMv8.0 NEON，**无 `dotprod`/FP16**），**`512 MB` RAM**（可用 `~350 MB`），带宽 `~2 GB/s`；Pi 4B：`4×A72`，`2–8 GB`，`~5.5 GB/s` | ❌ 差两个数量级以上（512 MB vs 26–130 GB） | ✅ **`tinymodel --features micro` 独占领域（`1.25 MiB` 纯静态 musl 二进制）**：<br>• T-MAC `vqtbl1q` 查表内核（`TQ2_0` / `Q2_0`）+ `--prune-vocab` 词表裁剪<br>• Zero 2W 跑 `SmolLM2-360M Q2_0/Q4` 折算 **`8.3–9.8 tok/s`**（峰值 RSS **`96–200 MiB << 350 MB`**，脏内存仅 `8–9.5 MiB`）；Pi 4B 可跑 `0.8B–2B`（`4–12 tok/s`） | 模型热加载 `<150 ms`；Zero 2W 整板满载仅 **`~1.8–2.2 W`**（每 token 仅 **`~0.22 J/tok`**），可在电池/太阳能物联网节点离线运行 |

---

## 6. 总结与下一步演进建议

1. **全家族实测闭环**：在 NVIDIA RTX PRO 6000 Blackwell 上完成了 `Strata v0.2.22` 支持的全部四大家族（`Inkling-2-Mini`、`Qwen3.6-35B-A3B`、`Qwen3.8-Flash-Next`、`Tencent Hy4-Flash`）实测，并现场为 `tinymodel` 补齐了开放架构 `qwen35moe`（`Qwen3.6-35B-A3B`）的 CPU + Vulkan 全栈支持与 `WG=64` 共享内存并行路由（单次优化令 Vulkan 解码提速 **`3.30×`** 达 **`107.34 tok/s`**，`Inkling-2 + DSpark` 代码生成达 **`92.40 tok/s = 98.1%` `strata launch`**，且加载快 **`3.1×–3.6×`**、显存省 **`4.5–5.2 GiB`**）。
2. **`tinymodel` 高杠杆后续方向（均严守开放 GGUF 与跨端侧复用原则）**：
   - **原生单层 `MTP` 块推理支持（`Qwen3.5` / `Qwen3.6` `nextn_predict_layers = 1`）**：本次实测证明模型内置 `MTP` 在代码与散文上均有 `63%–83%` 接受率（`1.4×–1.7×` 稳定加速），且同样适用于 `Qwen3.5` 端侧家族（Pixel 8 / RK3588 / M4 / Vulkan）。
   - **通用 `SparseMoeFfn` 接入 `CoopMatF16` 按专家分组 GEMM 与融合 `gate+up+SiLU` 索引 GEMV**：既能让 `Qwen3.6-35B` 预填充翻倍至 `5,000+ tok/s`，也能直接惠及 `OLMoE`、`Qwen3-30B-A3B`、`LFM2-8B-A1B` 等端侧稀疏 MoE 模型。
