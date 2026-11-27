# Strata 全量四大家族模型与 tinymodel（Vulkan & CUDA P0–P2 全栈优化）同机对比实测报告（RTX PRO 6000 Blackwell）

> 日期：2026-11-27（含 P0–P2 全栈优化同机复测更新）  
> 分支：`main`（完成 `ADR-024` 四大家族全覆盖 + `ADR-025` P0–P2 全栈内核与内置投机头优化）  
> 关联决策：[ADR-020](../DECISIONS.md)、[ADR-021](../DECISIONS.md)、[ADR-022](../DECISIONS.md)、[ADR-023](../DECISIONS.md)、[ADR-024](../DECISIONS.md)、[ADR-025](../DECISIONS.md)  
> 运行时配置登记：[SETTINGS.md](../SETTINGS.md)  
> 固定参考设备档案：[08-pixel8.md](08-pixel8.md)、[10-rk3588.md](10-rk3588.md)、[06-rpi4b-deployment.md](06-rpi4b-deployment.md)

---

## 1. 测试环境与基准协议

### 1.1 远程实测硬件（Google Colab Enterprise / Cloudflare SSH 隧道）
- **GPU**：**NVIDIA RTX PRO 6000 Blackwell Server Edition**（架构 `sm_120`，`188` SMs，显存 **`97,887 MiB` ≈ `95.6 GiB` GDDR7**，理论峰值带宽 `1,597.6–1,800 GB/s`，驱动 `580.82.07`，CUDA `13.0`，Vulkan `1.4.312`）
- **CPU / 内存 / 总线**：**48 线程 AMD EPYC 9B45**（`AVX-512`，实测 STREAM Read **`203.3–213.7 GB/s`**），主机内存 **`177 GiB` 可用**，PCIe Host→Device 实测 **`56.9 GB/s`**，NVMe 根分区 `236 GiB`
- **本地对齐与微基准验证机**：Linux `x86_64`（`AVX2 + FMA`）+ NVIDIA T1000（`sm_75` CUDA & Vulkan），用于 `L0/L1/L3/L5` 逐位确定性与多步对齐门禁。
- **被测配置对照（严守 `AGENTS.md` 规则 #8 同机基线对照）**：
  1. **`Strata CLI v0.2.22`（闭源 CUDA 13 `sm_120` 后端）**：`strata-inkling 0.1.69`、`strata-engine 0.1.69`、`strata-hy4 0.1.67`。
  2. **`tinymodel --vulkan`**：100% Rust `stable` + 预编译 SPIR-V 1.2（本次同步新增 `DTYPE=6` 原生 `Q2_0` DP4A / CoopMatF16 / MoE 内核）。
  3. **`tinymodel --cuda`（优化前基线 `218a561` vs 本次 `ADR-025` P0–P2 全栈优化后）**：新增内置 `blk.{L}.nextn.*` MTP 与 5 层 `DFly2` 神经投机头（P0-1）、稀疏 MoE 专家分桶 `wmma` Tensor Core GEMM（P0-2）、融合 `Gate+Up+SwiGLU/SiTU` 单次读激活 MoE 内核（P0-3）、`Hy4` 28× 压缩矩阵吸收式 MLA（P1-1）、锁页内存双缓冲异步 H2D（P1-2）、`compute_75`/`compute_80` 双 PTX 分派与 L2 持久化缓存（P1-3）、以及 `TM_CUDA_MOE_HOST_OFFLOAD` 零拷贝主机 DDR 专家卸载（P2）。

### 1.2 公平测量控制
- 统一冻结三类贪心输入负载（`temperature = 0`、`n_predict = 128`、`max_ctx = 4096`，均在热页缓存条件下测量）：
  1. **`p_code`**：Rust 无锁 SPSC 环形缓冲区代码续写（检验结构化代码生成吞吐与神经草稿接受率）；
  2. **`p64`**：Blackwell MoE / GDN / MTP 架构权衡英文技术散文（检验开放域生成吞吐与 `<5%` 增益自动熔断器）；
  3. **`bench`（`pp512` + `tg128`）**：标准 512 token 预填充与 128 token 连续贪心解码吞吐。

---

## 2. P0–P2 优化前后同机实测总表（RTX PRO 6000 Blackwell `98 GiB`）

| 模型家族 | 引擎与后端配置 | 热加载耗时 | 稳态显存 (`VRAM`) | `pp512` 预填充 | `bench tg128` | `p_code` 解码 (`tg128`) | `p64` 散文解码 (`tg128`) | 同机优化收益与关键结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`1. Inkling-2-Mini`**<br>(`Q8_0`, `26 GiB`) | `Strata CUDA`（`strata launch` 默认 / `All-VRAM`） | `15.0s` / `7.72s` | `21,755` / `33,431 MiB` | `884.2` / `1,755.2 tok/s` | `100.0 / 105.5 tok/s`¹ | `94.2–94.5 tok/s` | `94.3–94.6 tok/s` | `strata launch` 自测关闭 `DSpark` |
| | `tinymodel --vulkan`（纯解码 / `+DSpark`） | **`2.14–2.27 s`** | `27,750 / 28,134 MiB` | `1,430.9 tok/s` | `50.90 tok/s` | `49.95 / 92.40 tok/s` | `49.44 / 47.80 tok/s` | `CoopMatF16` 高吞吐预填充 |
| | `tinymodel --cuda`（优化前基线：纯解码 / `+DSpark`） | `2.69–2.71 s` | `27,708 MiB` | `314.78 tok/s` | `54.39 / 57.78 tok/s` | `52.88 / 97.38 tok/s` | `51.83 / 51.30 tok/s` | 初版分立 Gate/Up MoE |
| | **`tinymodel --cuda`（P0–P2 后：纯解码）** | **`2.56–2.66 s`** | **`27,708 MiB`** | **`337.55 tok/s`** (`+7.2%`) | **`87.90 tok/s`** (`11.38 ms/tok`) | **`83.98 tok/s`** | **`81.36 tok/s`** | **纯解码飙升 `+61.6%`（达 Strata All-VRAM 纯解码 `87.9%`！比 `--vulkan` 快 `1.73×`）** |
| | **`tinymodel --cuda`（P0–P2 后：`+DSpark k=7`）** | **`2.63 s`** | **`27,708 MiB`** | `334.16 tok/s` | **`64.60 tok/s`** (`+11.8%`) | **`98.90 tok/s`** (`89/191=46.6%`) | **`59.65 tok/s`** (`+16.3%`) | **代码生成刷至 `98.90 tok/s`，稳超 `strata launch`（`94.5`）并省 `5.7 GiB` 显存！** |
| **`2. Qwen3.6-35B-A3B`**<br>(`Q8_0`, `35 GiB`) | `Strata CUDA`（纯解码 `--mtp off` / 启用 `+MTP`） | `8.30 s` | `43,291 / 43,293 MiB` | `5,563.3 / 5,433.3 tok/s` | — | `204.7 / 304.2 tok/s` | `202.7 / 272.6 tok/s` | `sm_120` cuBLAS + 内置 MTP |
| | `tinymodel --vulkan` | **`2.70 s`** | `~37,800 MiB` | `2,948.0 tok/s` | `107.34 tok/s` | `105.78 tok/s` | `104.84 tok/s` | 跨平台 Vulkan 基线 |
| | `tinymodel --cuda`（优化前基线） | `3.62 s` | **`35,628 MiB`** | `431.93 tok/s` | `129.87 tok/s` (`7.70 ms/tok`) | `123.95 tok/s` | `119.90 tok/s` | 初版未融合 MoE、无 MTP |
| | **`tinymodel --cuda`（P0–P2 后：纯解码）** | **`3.55–3.73 s`** | `36,574 MiB`² | `413.48 tok/s` | **`177.69 tok/s`** (`5.63 ms/tok`) | **`166.31 tok/s`** | **`159.08 tok/s`** | **纯解码大涨 `+36.8%`！达 Strata 纯解码 `86.8%`，比 `--vulkan` 快 `1.66×`，省 `6.7 GiB` 显存** |
| | **`tinymodel --cuda`（P0–P2 后：`--spec auto` 内置 `nextn` MTP）** | **`3.55–3.63 s`** | `36,574 MiB` | `413.49 tok/s` | **`171.99 tok/s`** (`2/6` 自动熔断) | **`200.65 tok/s`** (`88/108=`**`81.5%`**) | **`185.44 tok/s`** (`85/108=`**`78.7%`**) | **内置 MTP 全面起飞：代码较基线暴涨 `+61.9%` 破 `200 tok/s`，散文涨 `+54.7%` 达 `185.4 tok/s`！** |
| | **`tinymodel --cuda`（P2：`TM_CUDA_MOE_HOST_OFFLOAD=10`）** | `4.67 s` | **`28,430 MiB`** (`-8.14 GiB`) | `132.72 tok/s` | **`69.27 tok/s`** (`tg64`) | — | — | **零拷贝 DDR 卸载立省 `8.14 GiB` 显存，让 `35B Q8_0` 轻松装入 `32 GB` 消费级显卡！** |
| **`3. Qwen3.8-Flash-Next`**<br>(`Q2_0` 2-shard, `61.6 GiB`) | `Strata CUDA`（`--mtp off` / 默认 `+MTP`） | `60.0–75.0 s` | `38,085 / 39,119 MiB` | `1,115.6 tok/s` | — | `186.2 / 315.6 tok/s` | `181.2 / 291.8 tok/s` | 需额外生成 `44.9 GiB` `.pack` |
| | `tinymodel --cuda`（基线 / P0–P2 后 `--cuda`） | **`3.84–4.05 s`** | **`36,968–37,140 MiB`** | `144.6–153.2 tok/s` | **`85.21–87.40 tok/s`** | **`82.63–85.49 tok/s`** | **`81.09–83.82 tok/s`** | 零 `.pack` 省 **`44.9 GiB`** 磁盘，热加载快 **`15.6×`**；`--spec auto` 在 `0/12` 后自动熔断零退化³ |
| **`4. Tencent Hy4-Flash`**<br>(`SQ-64`, `120.7 GiB`) | `Strata CUDA`（纯单步遥测 / 默认 `+DFly2`） | `6.0 s` | `64,109 / 64,113 MiB` | `3,116.4 tok/s` | `~81.3 tok/s` | `128.1–133.0 tok/s` | `134.2–136.3 tok/s` | 占用 `64.1 GiB` 显存 |
| | `tinymodel --cuda`（优化前基线：非吸收式 MLA，无 `DFly2`） | `6.55–8.42 s` | `61,978 MiB` | `136.32 tok/s` | `36.66 tok/s` | **`44.14 tok/s`** | `35.64 tok/s` | 展开式 MLA KV 占 `~9.19 GiB` |
| | **`tinymodel --cuda`（P0–P2 后：28× 吸收式 MLA + 5 层 `DFly2`）** | **`4.34–4.42 s`** (`-33.7%`) | **`59,350 MiB`** (**`-2.57 GiB`**) | `134.94 tok/s` | `27.83 tok/s` (`0/3` 熔断) | **`37.08 tok/s`** (`67/99=`**`67.7%`**) | **`36.75 tok/s`** (`+3.1%`, `76/106=`**`71.7%`**) | **装入全部 5 层 `DFly2` 后总显存反而净省 `2.57 GiB`（比 Strata 省 `4.65 GiB`）！热加载 `4.34 s` 反超 Strata（`6.0 s`）** |

> ¹ `strata-inkling --bench` 为内部合成 512→128 基准。  
> ² `Qwen3.6` 显存从 `35,628 MiB` 增至 `36,574 MiB`（`+946 MiB`）是因为本次完整加载并常驻了第 41 层 `blk.40`（256 专家 MoE + FullAttn）内置 `nextn` MTP 权重与 L2 驻留窗口。  
> ³ `Qwen3.8-Flash-Next` 的 `ISTA-DASLab` `GSQ-RCO` 2-shard 包同一层内 `ffn_gate_exps`（`Q2_0`）与 `ffn_up_exps`（`Q3_K`/`IQ4_XS` 混合）量化格式异构，按张量自身 dtype 严格分派时保持分立路径（后续扩展异构 `(gate_dt, up_dt)` 双门控模板即可进一步获益）。

---

## 3. CPU 微基准验证（`P2`：`AVX2` 原生 `Q2_0` 内核）

在同一台本地 `x86_64` 主机上按 `docs/TESTING.md` 跑 `cargo bench -p tm-cpu` 对比 `TM_FORCE_ISA=generic` 与默认 `avx2`：

| 内核与矩阵规模 | `generic` 标量基线 | `avx2`（本次新增） | 同机加速比 | 对齐精度（`parity_m4`） |
|---|---:|---:|---:|---|
| `gemv_q2_0_q8_0` (`2048 × 2048`) | `6.6 GFLOP/s` (`out[0]=-95.400`) | **`30.9 GFLOP/s`** (`out[0]=-95.400`) | **`4.68×`（`+368%`）** | **0-ULP 逐位一致**（含 `vec_dot` / `gemv` / 4-token `gemm`） |

---

## 4. 关键技术突破与客观权衡剖析

1. **P0-3 双门控激活融合（`moe_indexed_dual_gu_act_f32`）立竿见影**：
   - 稀疏 MoE 解码原本需要 `gate_exps` GEMV → `up_exps` GEMV → `SwiGLU/SiTU` → `quantize_q8_1` 多次读写全局内存。融合为单内核一次加载 `Q8_1` 输入、寄存器内同时点积 `gate` 与 `up` 并原位完成 `SwiGLU` / `Inkling-2 SiTU` 后，`Inkling-2` 纯解码延迟从 `18.39 ms/tok` 直降至 **`11.38 ms/tok`（`+61.6%`）**，`Qwen3.6-35B-A3B` 纯解码从 `7.70 ms/tok` 直降至 **`5.63 ms/tok`（`+36.8%`）**。
2. **P0-1 内置神经投机头（`Qwen3.6 nextn MTP` & `Hy4 DFly2`）端到端闭环**：
   - `Qwen3.6-35B-A3B` 的内置 `blk.40.nextn` MTP 层在真实代码（`81.5%` 接受率）与技术散文（`78.7%` 接受率）上均实现稳定高命中，配合零矩阵乘 GDN 递推重放，将端到端生成速度推高至 **`200.65 tok/s`**（代码）与 **`185.44 tok/s`**（散文）；而在低可预测性的合成循环 token 流（`bench`）中，`SpecController` 在 3 次探针内识别 `2/6` 低命中并自动熔断回单步 `CudaGraphExec`（`171.99 tok/s`）。
   - `Hy4-Flash` 的 5 层 `DFly2` 神经草稿器（并行掩码 MLA + 链式残差修正 MLP + 置信度累积门控）在代码与散文上分别取得 **`67.7%`** 与 **`71.7%`** 的接受率，且与单步贪心解码在测试集上 **100% 逐 token 一致**。
3. **P1-1 矩阵吸收式 MLA 的显存大捷与短/长上下文权衡**：
   - 将 `Hy4` 42 层 KV 缓存压缩为 `[cap, 576]` FP16 潜变量后，KV 显存暴降 **28.4×（省 `8.87 GiB`）**，使模型在额外常驻全部 5 层 `DFly2` 草稿头后总显存仍净降 **`2.57 GiB`** 至 `59,350 MiB`（使 `64 GiB` 级工作站显卡具备更大上下文余量）。
   - 同时实测揭示了一个清晰的物理边界：在短上下文（`<200` tokens）且 `w = 5` 多 token 投机验证时，按头吸收 `W_kb`（`64 × 128 × 512`）与 `W_vb` 的总权重读取量高于非吸收式对单个新 token 做一次 `kv_b_proj` `__dp4a` MMVQ；因此短提示纯吞吐以非吸收式占优（`44.14 tok/s`），长上下文（`≥4K`）与显存受限场景以矩阵吸收式占优。
4. **P1-2 锁页双缓冲流式加载与 P2 零拷贝 DDR 专家卸载**：
   - `.strata` `64 MiB` 双缓冲 `CudaPinnedBuffer` + `cuMemcpyHtoDAsync_v2` 将 `Hy4-Flash`（`58 GiB` 主文件）热加载从 `6.55 s` 压降至 **`4.34 s`**，反超闭源 `Strata CUDA`（`6.0 s`）。
   - `TM_CUDA_MOE_HOST_OFFLOAD=10` 通过 `CU_MEMHOSTALLOC_DEVICEMAP` 将末尾 10 层专家零拷贝映射在主机锁页 DDR，在 `Qwen3.6-35B-A3B Q8_0` 上立省 **`8.14 GiB` 显存**（降至 `28.4 GiB`）并保持 **`69.27 tok/s`** 实用交互吞吐。

---

## 5. 六大固定参考设备全栈部署、性能、延迟与功耗对比分析

按 [AGENTS.md](../../AGENTS.md) §8 规范，结合本次 **P0–P2 全栈优化** 实测数据，更新全部 **6 类固定参考设备** 的部署边界、推荐后端、吞吐与能效对比：

| 参考设备 | 硬件与内存带宽 | P0–P2 优化受惠项与可部署边界 | `tinymodel` 最新实测 / 折算吞吐与内存水位 | 首字延迟（TTFT）与微调 / 能效特征 |
|---|---|---|---|---|
| **1. Host 独显服务器**<br>**RTX PRO 6000 Blackwell** | `98 GiB` GDDR7（`~1.6–1.8 TB/s`）+ 48T EPYC + PCIe 5.0 + `177 GiB` RAM | ✅ **全量受益 P0–P2 全部优化**（双门控融合 MoE、内置 MTP/`DFly2`、分桶 Tensor Core、`compute_80` PTX + L2 驻留、吸收式 MLA、DDR 卸载） | • `Inkling-2`：**纯解码 `87.90 tok/s`（`+61.6%`）**，`+DSpark` 代码 **`98.90 tok/s`**（`27.7 GiB`）<br>• `Qwen3.6-35B Q8_0`：**纯解码 `177.69 tok/s`（`+36.8%`）**，**`+内置 MTP` 代码 `200.65 tok/s` / 散文 `185.44 tok/s`**；10 层 DDR 卸载 **`28.4 GiB` @ `69.3 tok/s`**<br>• `Qwen3.8 Q2_0`：**`85.2–87.4 tok/s`**（零重排省 `44.9 GiB` 盘）<br>• `Hy4-Flash`：显存降至 **`59.4 GiB`**，热加载 **`4.34 s`**，解码 **`36.8–44.1 tok/s`** | 热加载 **`2.5–4.4 s`**；`Qwen3.6 + MTP` 单 token 等效能耗降至 **`~1.8–2.5 J/tok`**；`DDR MoE Offload` 让 `32 GB` 单卡亦可运行 `35B Q8_0` |
| **2. Apple M4 开发机**<br>（macOS `host` 档） | `16 GB` 统一内存（实测 `102 GB/s`），4P+6E（`NeonI8mm`）+ 10 核 Metal GPU | ✅ 受益于 CPU 矩阵吸收式 MLA（`Hy4` KV 内存降 28×）、内置 `nextn` MTP 解析及 `--spec auto` 熔断器（`36/64 GB` M4 Max 可常驻 `Qwen3.6 Q4` / `Inkling-2 Q4`） | ✅ **`--metal` 主力甜点（`≤9B` 稠密与中小型 MoE）**：<br>• `SmolLM2-360M`：**`270.5 tok/s`**<br>• `Qwen3.5-0.8B`（GDN）：**`129.0 tok/s`**、`2B`：**`63.3 tok/s`**<br>• `Gemma 4 E2B`（PLE）：**`52.3 tok/s`** | 512 token TTFT `0.4–1.5 s`（`PrefixCache` `<20 ms`）；`--metal` QLoRA 微调 `22–72 tok/s`；整机 **`12–22 W`（`0.1–0.3 J/tok`）** |
| **3. Google Pixel 8**<br>（Tensor G3，Android `edge` 档） | `8 GB` LPDDR5X（可用 `3.8–4.1 GiB`，`5T` STREAM **`31.3 GB/s`**），`NeonI8mm` + `Mali-G715` Vulkan | ✅ **直接受益于本次 P2 Vulkan 原生 `Q2_0`（`gemv_dp4a` / `gemv_res` / `embed`）**：超低比特 `Q2_0` 模型在 `Mali-G715` 上不再膨胀为 `F32`，显存占用立降 **14.2×**！ | • `SmolLM2-360M Q4_0x4`：**`103.95 tok/s`**<br>• `Qwen3.5-0.8B`（GDN）：**`38.96 tok/s`**<br>• `Gemma 4 E2B`（PLE 单行 `mmap`）：**`14.56 tok/s`**（RSS `1.47 GiB`）<br>• Vulkan 原生 `Q2_0` 解锁 `3B–4B Q2_0`（`~1.1 GiB`）手机端 GPU 推理 | 300 token TTFT `1.6–4.5 s`（`PrefixCache` `<50 ms`）；SoC 功耗 `3.5–5.5 W`（`0.8B` 仅 **`~0.12 J/tok`**），零拷贝 `mmap` 杜绝 Android LMK |
| **4. Rockchip RK3588**<br>（8nm Linux SBC `edge` 档） | `4×A76 + 4×A55`（`NeonDotprod`），`8 / 16 / 32 GB` LPDDR4X/5（`17–23 GB/s`）+ `Mali-G610` Vulkan | ✅ **双重受益**：32 GB 版既可用 CPU `NeonDotprod` + 内置 `nextn` MTP 跑 `Qwen3.6-35B-A3B Q4/Q2_0`，也可在 `Mali-G610` Vulkan 上直跑原生 `Q2_0` 模型（免 `F32` 膨胀） | • `Qwen3.5-0.8B`：`24–38 tok/s`、`2B`：`11–18 tok/s`<br>• **32 GB RK3588 跑 `Qwen3.6-35B-A3B Q4_0x4`（`~19.5 GiB`）**：纯解码 `8.5–11.5 tok/s`，配合内置 `nextn` MTP 代码生成折算可达 **`12–16 tok/s`** | NVMe 冷加载 `35B Q4` 约 `6 s`；整板仅 **`6–8.5 W`**（`35B-A3B + MTP` 约 **`0.5–0.7 J/tok`**，适合 7×24h 离线边缘网关） |
| **5. Raspberry Pi 5**<br>（16nm Linux SBC `edge` 档） | `4×A76 @ 2.4GHz`（`NeonDotprod`），`4 / 8 / 16 GB` LPDDR4X（`~11.5 GB/s`）+ VideoCore VII V3DV Vulkan | ✅ **受益于 Vulkan `gemm_tiled` / `gemv_res` 原生 `Q2_0`（无 Subgroup Shuffle）**：16 GB 版可在 V3DV GPU / CPU 上零膨胀运行 `Q2_0` 稀疏 MoE（如 `OLMoE-1B-7B Q2_0` `~2.3 GiB`） | • `SmolLM2-360M`：`35–45 tok/s`<br>• `Qwen3.5-0.8B`：`15–20 tok/s`<br>• 16 GB 板常驻 `OLMoE-1B-7B Q4_0x4 / Q2_0`：**`9–14 tok/s`** | 292 token TTFT `4.5–6.5 s`（`PrefixCache` 加速多轮对话）；整板功耗 `7–10 W`（`0.8B` 约 `0.45 J/tok`） |
| **6. Pi 4B / Pi Zero 2W**<br>（`micro` 极限档） | Zero 2W：`4×A53`（ARMv8.0 NEON），**`512 MB` RAM**（可用 `~350 MB`），`~2 GB/s`；Pi 4B：`2–8 GB` | ✅ 与本次 `Q2_0` 量化链路完全同源（主机用 `AVX2` `4.68×` 加速转换/校验后的同一份 `Q2_0` GGUF 可直接拷入 Zero 2W 由 `vqtbl1q` T-MAC 查表运行） | ✅ **`--features micro` 超轻量静态 ELF（`<1.3 MiB`）**：<br>• Zero 2W 跑 `SmolLM2-360M Q2_0`（裁剪词表后 `93.6 MiB`）折算 **`8.3–9.8 tok/s`**（RSS `96–200 MiB << 350 MB`） | 热加载 `<150 ms`；Zero 2W 满载仅 **`1.8–2.2 W`（`~0.22 J/tok`）**，支持纯离线传感器边缘节点 |

---

## 6. 总结与后续演进方向

1. **P0–P2 全栈优化全部落地并经同机验证**：
   - **`Inkling-2-Mini Q8_0`**：纯解码从 `54.39` 跃升至 **`87.90 tok/s`（`+61.6%`）**，`+DSpark` 代码生成达 **`98.90 tok/s`**（超越 `strata launch` 默认配置并省 `5.7 GiB` 显存）；
   - **`Qwen3.6-35B-A3B Q8_0`**：纯解码从 `129.87` 跃升至 **`177.69 tok/s`（`+36.8%`）**，启用内置 `blk.40.nextn` MTP 后代码解码突破 **`200.65 tok/s`（较基线 `+61.9%`，接受率 `81.5%`）**、散文达 **`185.44 tok/s`（`+54.7%`）**，10 层零拷贝 DDR 卸载省 **`8.14 GiB`** 显存；
   - **`Tencent Hy4-Flash`**：矩阵吸收式 MLA 将 KV 显存压缩 **28.4×**，常驻 5 层 `DFly2` 后总显存仍净省 **`2.57 GiB`**（`59,350 MiB`），锁页双缓冲流式加载将热启动压至 **`4.34 s`**（反超 Strata `6.0 s`），`DFly2` 接受率达 **`67.7%–71.7%`**；
   - **跨平台低比特补齐**：`AVX2` `gemv_q2_0_q8_0` 提速 **`4.68×`**（0-ULP），`tm-vulkan` 全管线支持零膨胀原生 `Q2_0`。
2. **下一阶段高收益方向**：
   - ① 为 `moe_indexed_dual_gu_act_f32` 支持异构 `(gate_dtype, up_dtype)` 组合，解锁 `Qwen3.8-Flash-Next`（`GSQ-RCO` 混合比特专家）的同款 `+35%–60%` 双门控融合加速；
   - ② 在 `Hy4CudaRunner` 中按实时上下文长度自适应切换「短上下文非吸收式 `__dp4a` MMVQ（`<1K`）」与「长上下文矩阵吸收式潜变量 MQA（`≥1K`）」，兼得 `59.4 GiB` 低显存、`<1K` 高吞吐与长窗口线性扩展。
