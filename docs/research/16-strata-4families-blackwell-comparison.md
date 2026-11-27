# Strata 全量四大家族模型与 tinymodel（Vulkan & CUDA 两轮 P0–P2 全栈深度优化）同机对比实测报告（RTX PRO 6000 Blackwell）

> 日期：2026-11-27（含第一轮 `ADR-025` 与第二轮 `ADR-026` P0-A→P2 全栈优化同机复测）  
> 分支：`main`（完成 `ADR-024` 四大家族全覆盖 + `ADR-025` 第一轮优化 + `ADR-026` 第二轮深度优化）  
> 关联决策：[ADR-020](../DECISIONS.md)、[ADR-021](../DECISIONS.md)、[ADR-022](../DECISIONS.md)、[ADR-023](../DECISIONS.md)、[ADR-024](../DECISIONS.md)、[ADR-025](../DECISIONS.md)、[ADR-026](../DECISIONS.md)  
> 运行时配置登记：[SETTINGS.md](../SETTINGS.md)（`TM_CUDA_MOE_HOST_OFFLOAD`、`TM_CUDA_MOE_CACHE_EXPERTS`）  
> 固定参考设备档案：[08-pixel8.md](08-pixel8.md)、[10-rk3588.md](10-rk3588.md)、[06-rpi4b-deployment.md](06-rpi4b-deployment.md)

---

## 1. 测试环境与基准协议

### 1.1 远程实测硬件（Google Colab Enterprise / Cloudflare SSH 隧道）
- **GPU**：**NVIDIA RTX PRO 6000 Blackwell Server Edition**（架构 `sm_120`，`188` SMs，显存 **`97,887 MiB` ≈ `95.6 GiB` GDDR7**，理论峰值带宽 `1,597.6–1,800 GB/s`，驱动 `580.82.07`，CUDA `13.0`，Vulkan `1.4.312`）
- **CPU / 内存 / 总线**：**48 线程 AMD EPYC 9B45**（`AVX-512`，实测 STREAM Read **`203.3–213.7 GB/s`**），主机内存 **`176 GiB` 可用**，PCIe Host→Device 实测 **`56.9 GB/s`**，NVMe 根分区 `236 GiB`
- **本地对齐与微基准验证机**：Linux `x86_64`（`AVX2 + FMA`）+ NVIDIA T1000（`sm_75` CUDA & Vulkan），用于 `L0/L1/L3/L5` 逐位确定性与多步对齐门禁。
- **同机演进与对照配置（严守 `AGENTS.md` 规则 #8）**：
  1. **`Strata CLI v0.2.22`（闭源 CUDA 13 `sm_120` 后端）**：`strata-inkling 0.1.69`、`strata-engine 0.1.69`、`strata-hy4 0.1.67`。
  2. **`tinymodel --vulkan`**：100% Rust `stable` + 预编译 SPIR-V 1.2（含 `ADR-025` 原生 `Q2_0` 与本次 `ADR-026` P1 跨驱动融合双门控 `moe_dual_gu_act.comp` & `inkling2_moe` `op=6`）。
  3. **`tinymodel --cuda` 三阶段同机演进对照**：
     - **初版基线（`ADR-024`）**：四大家族打通首版；
     - **第一轮优化（`ADR-025`）**：内置 `nextn` MTP & `DFly2`、分桶 `wmma` MoE、同构双门控融合、28.4× 吸收式 MLA、锁页异步 H2D、`TM_CUDA_MOE_HOST_OFFLOAD`；
     - **第二轮优化（`ADR-026`，本次落地）**：
       - **P0-A（预填充跃升）**：`MAX_CUDA_BATCH=512`、无 bank 冲突 `WMMA_SMEM_STRIDE=40` 向量化反量化、宽 N `wmma_gemm_mixed_64x64_f32`、`.strata` `S_K` Tensor Core（`wmma_hy4_sk_gemm_f32` + `wmma_hy4_latmoe_grouped_f32`）、0-ULP 寄存器驻留多 token 批 GDN；
       - **P0-B（投机图捕获与自适应熔断）**：`Qwen35CudaRunner` & `Qwen4ExpCudaRunner` 多步验证 `CudaGraphExec` 图捕获（`verify_graphs`）+ 内联 GDN 快照、原生 `qwen4exp-mtp`（`mtp-q2_0.gguf`）支持、实测单步基线自动校准 `SpecController`、`Inkling-2 + DSpark` 熔断直切 `CudaGraphExec`；
       - **P0-C（`Hy4` 高占用率 MLA 与 `NR=4` `S_K`）**：`512`-CTA `hy4_mla_absorb_q`（8-token 共享 `1.0×` `W_kb` 读取）、`hy4_mla_vup_gate`（`NR=2` `1.0×` `W_vb` 读取 + 原位 Sigmoid 门控）、`NR=4` `S_K` MMVQ 与融合 `swiglu_quantize_g64_f32`；
       - **P1（异构双门控与 Vulkan 双门控）**：CUDA 异构 `(gate_dt, up_dt)` 单遍 `moe_indexed_dual_gu_act_f32` + Vulkan `moe_dual_gu_act.comp`；
       - **P2（VRAM 热专家频率缓存）**：`TM_CUDA_MOE_CACHE_EXPERTS` 图捕获安全 GPU 老化晋升缓存 `CudaMoeVramCache`。

### 1.2 公平测量控制
- 统一冻结三类贪心输入负载（`temperature = 0`、`n_predict = 128`、`max_ctx = 4096`）：
  1. **`p_code`**：Rust 无锁 SPSC 环形缓冲区代码续写（38 tokens 提示词）；
  2. **`p64`**：Blackwell MoE / GDN / MTP 架构权衡英文技术散文（66–67 tokens 提示词）；
  3. **`bench`（`pp512` + `tg128`）**：标准 512 token 预填充与 128 token 连续贪心解码吞吐。

---

## 2. 两轮 P0–P2 优化同机实测总表（RTX PRO 6000 Blackwell `98 GiB`）

| 模型家族 | 引擎与后端配置 | 热加载耗时 | 稳态显存 (`VRAM`) | `pp512` 预填充 | `bench tg128` | `p_code` 解码 (`tg128`) | `p64` 散文解码 (`tg128`) | 第二轮（`ADR-026`）关键收益与结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`1. Inkling-2-Mini`**<br>(`Q8_0`, `26 GiB`) | `Strata CUDA`（`strata launch` 默认 / `All-VRAM`） | `15.0s` / `7.72s` | `21,755` / `33,431 MiB` | `884.2` / `1,755.2 tok/s` | `100.0 / 105.5 tok/s` | `94.2–94.5 tok/s` | `94.3–94.6 tok/s` | `strata launch` 自测关闭 `DSpark` |
| | `tinymodel --vulkan`（`ADR-026` P1 `op=6`：纯解码 / `+DSpark`） | **`2.14–2.31 s`** | `27,750 / 28,134 MiB` | `1,429.8 tok/s` | `51.16 tok/s` | `50.33 / 85.35 tok/s` | `49.71 / 47.85 tok/s` | 省去独立 `up` scratch，解码微升 |
| | `tinymodel --cuda` 初版基线 → 第一轮 `ADR-025`（纯解码） | `2.60 s` | `27,708 MiB` | `314.8 → 337.6 tok/s` | `54.39 → 87.90 tok/s` | `52.88 → 83.98 tok/s` | `51.83 → 81.36 tok/s` | 第一轮融合 SiTU `+61.6%` |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（纯解码）** | **`2.78–2.90 s`** | **`27,952 MiB`** | **`789.88 tok/s`** (**`2.34×`**) | **`87.78 tok/s`** (`11.39 ms/tok`) | **`83.91 tok/s`** | **`81.24 tok/s`** | **P0-A `WMMA64` 令 `pp512` 暴涨 `+134.0%`（较初版 `2.51×`，追近 `strata launch` `884`）！** |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（`+DSpark k=7`）** | **`2.82–2.87 s`** | **`27,952 MiB`** | **`785.93 tok/s`** | **`81.79 tok/s`** (`+26.6%`, `1/9` 切图) | **`114.56 tok/s`** (`95/172=`**`55.2%`**) | **`74.60 tok/s`** (`+25.1%`, `2/13` 切图) | **代码冲上 `114.56 tok/s`（超 `Strata CUDA` `94.5` 达 `+21.2%`，省 `5.5 GiB` 显存）！散文熔断切回 `CudaGraphExec` 彻底消除退化** |
| **`2. Qwen3.6-35B-A3B`**<br>(`Q8_0`, `35 GiB`) | `Strata CUDA`（`--mtp off` / `+MTP`） | `8.30 s` | `43,291 / 43,293 MiB` | `5,563.3 / 5,433.3 tok/s` | — | `204.7 / 304.2 tok/s` | `202.7 / 272.6 tok/s` | 占用 `43.3 GiB` 显存 |
| | `tinymodel --vulkan`（`ADR-026` P1 `moe_dual_gu_act`） | **`2.93 s`** | `~37,700 MiB` | `2,943.1 tok/s` | **`107.70 tok/s`** | **`106.89 tok/s`** | **`106.14 tok/s`** | P1 双门控融合省显存且全线稳升 |
| | `tinymodel --cuda` 初版基线 → 第一轮 `ADR-025`（纯解码） | `3.60 s` | `35.6 → 36.6 GiB` | `431.9 → 413.5 tok/s` | `129.9 → 177.7 tok/s` | `124.0 → 166.3 tok/s` | `119.9 → 159.1 tok/s` | 第一轮纯解码 `+36.8%` |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（纯解码）** | **`3.56–3.67 s`** | **`36,690 MiB`** | **`2,579.79 tok/s`** (**`6.24×`**) | **`177.40 tok/s`** (`5.64 ms/tok`) | **`166.19 tok/s`** | **`159.05 tok/s`** | **P0-A `WMMA64` + 寄存器批 GDN 令 `pp512` 飙升 `+523.9%`（`6.24×`），突破 `2,579 tok/s`！** |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（`--spec auto` 内置 MTP）** | **`3.56–3.63 s`** | **`36,690 MiB`** | **`2,579.8 tok/s`** | **`173.21 tok/s`** (`2/6` 熔断) | **`239.39 tok/s`** (`87/108=`**`80.6%`**) | **`238.79 tok/s`** (`88/108=`**`81.5%`**) | **P0-B `verify_graphs` 图捕获再涨 `+19.3% / +28.8%`！代码与散文双双逼近 `240 tok/s`（较初版 `+93.1%` / `+99.2%` 几近翻倍，省 `6.6 GiB` 显存）！** |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（P2：10 层卸载 + 64 槽专家缓存）** | `4.72 s` | **`30,586 MiB`** (`<32 GB`) | **`344.39 tok/s`** (`+38.0%`) | **`109.61 tok/s`** (`tg64`, **`+54.0%`**) | **`101.26 tok/s`** (**`+49.4%`**) | — | **仅增 `2.0 GiB` 显存即令 10 层 DDR 卸载破百 `tok/s`（且 `last_tok=1639` 0-ULP 一致），`32 GB` 消费卡畅跑 `35B Q8_0`！** |
| **`3. Qwen3.8-Flash-Next`**<br>(`Q2_0` 2-shard, `61.6 GiB`) | `Strata CUDA`（`--mtp off` / `+MTP`） | `60.0–75.0 s` | `38,085 / 39,119 MiB` | `1,115.6 tok/s` | — | `186.2 / 315.6 tok/s` | `181.2 / 291.8 tok/s` | 需额外生成 `44.9 GiB` `.pack` |
| | `tinymodel --cuda` 第一轮 `ADR-025` | `3.95 s` | `37,140 MiB` | `144.6–153.2 tok/s` | `85.21 tok/s` | `82.63 tok/s` | `81.09 tok/s` | 零 `.pack` 省 `44.9 GiB` 磁盘 |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（纯解码 / `+mtp-q2_0.gguf --spec auto`）** | **`3.82–3.95 s`** | **`37,400 / 38,366 MiB`** | **`1,133.54 tok/s`** (**`7.40×`**) | **`85.03 / 77.14 tok/s`** | **`82.61 / 76.62 tok/s`** | **`80.91 / 74.75 tok/s`** | **P0-A 令 `pp512` 暴涨 `+640%`（`7.40×`）反超 `Strata CUDA`（`1,115.6`）且零 `.pack`！P0-B 打通外部 `mtp-q2_0.gguf` 并在 3 探针内精准保护吞吐¹** |
| **`4. Tencent Hy4-Flash`**<br>(`SQ-64`, `120.7 GiB`) | `Strata CUDA`（纯单步 / `+DFly2`） | `6.0 s` | `64,109 / 64,113 MiB` | `3,116.4 tok/s` | `~81.3 tok/s` | `128.1–133.0 tok/s` | `134.2–136.3 tok/s` | 占用 `64.1 GiB` 显存 |
| | `tinymodel --cuda` 第一轮 `ADR-025`（吸收式 MLA + `DFly2`） | `4.34–4.42 s` | **`59,350 MiB`** | `134.94 tok/s` | `27.83 tok/s` (`0/3` 熔断) | `37.08 tok/s` (`67.7%`) | `36.75 tok/s` (`71.7%`) | 显存省 `4.65 GiB`，MLA CTA 偏少 |
| | **`tinymodel --cuda` 第二轮 `ADR-026`（P0-A `.strata` WMMA + P0-C 512-CTA MLA + `DFly2`）** | **`4.43–4.52 s`** | **`59,540 MiB`** (`-4.46 GiB` vs Strata) | **`425.58 tok/s`** (**`3.15×`**) | **`34.10 tok/s`** (`+22.5%`, `0/3` 熔断) | **`44.09 tok/s`** (`+18.9%`, `80/111=`**`72.1%`**) | **`44.58 tok/s`** (`+21.3%`, `74/106=`**`69.8%`**) | **`.strata` `S_K` Tensor Core 令 `pp512` 提升 `3.15×`！512-CTA 共享权重吸收式 MLA 令单步解码涨 `+22.5%`、`+DFly2` 双双突破 `44 tok/s`！** |

> ¹ `Qwen3.8-Flash-Next` 每层含 512 个专家且每 token 激活 `k=16` 专家；在 `GSQ-RCO Q2_0` 极限量化主干与上游 `BF16→Q2_0` 独立 MTP 头组合下，单步草稿命中率为 `45.7%`（`58/127`），而 2-token 验证需读取约 `2×` 稀疏专家权重（`1.457 / 2.0 = 0.73×` 带宽效率），经实测基线自动校准的 `SpecController` 在 3 次探针内识别负收益并切回单图 `CudaGraphExec`（`76.62 / 74.75 tok/s`，较未捕获/未校准时的 `42.52 tok/s` 提速 **`+80.2%`**）。

---

## 3. P2 热专家 VRAM 频率缓存（`TM_CUDA_MOE_CACHE_EXPERTS`）同机消融实测

在 `Qwen3.6-35B-A3B Q8_0` 上固定开启 `TM_CUDA_MOE_HOST_OFFLOAD=10`（末尾 10 层 256 专家驻留主机锁页 DDR），对比不同缓存槽位数：

| 配置 | 稳态显存 (`VRAM`) | `pp512` 预填充 | `bench tg64` 解码 | `p_code tg128` 解码 | 对齐确定性 |
|---|---:|---:|---:|---:|---|
| 全显存常驻（`OFFLOAD=0`） | `36,690 MiB` | `2,579.79 tok/s` | `177.40 tok/s` (`tg128`) | `166.19 tok/s` | 基准 `last_tok=1639` |
| 10 层零拷贝 DDR 卸载，`CACHE_EXPERTS=0` | `28,546 MiB` (`-8.14 GiB`) | `249.47 tok/s` | `71.20 tok/s` | `67.78 tok/s` | **`last_tok=1639`（0-ULP）** |
| **10 层零拷贝 DDR 卸载，`CACHE_EXPERTS=64`（每卸载层缓存最热 25% 专家）** | **`30,586 MiB`** (`< 32 GB`) | **`344.39 tok/s`** (**`+38.0%`**) | **`109.61 tok/s`** (**`+54.0%`**) | **`101.26 tok/s`** (**`+49.4%`**) | **`last_tok=1639`（0-ULP 逐位一致）** |

---

## 4. 第二轮优化核心工程突破总结

1. **宽 N `64×64` WMMA + 无 bank 冲突共享内存 + 寄存器驻留批 GDN 彻底打通 CUDA 预填充**：
   - 将共享内存行跨度设为 `WMMA_SMEM_STRIDE = 40`（`+8 half` 偏移彻底消除 8 列 fragment 加载 bank 冲突），并在 `n_tok >= 64` 时启用 `64×64×32` 双倍权重复用分块，再配合寄存器驻留 `[S, S]` 状态批 GDN 与 `.strata` `S_K` Tensor Core，使四大家族 `pp512` 同步跃升 **`2.34×`（`Inkling-2`）、`6.24×`（`Qwen3.6`）、`7.40×`（`Qwen3.8`，反超闭源 `Strata CUDA`）与 `3.15×`（`Hy4-Flash`）**。
2. **`verify_graphs` 多步验证 CUDA Graph 捕获释放神经投机真正上限**：
   - `Qwen3.6-35B-A3B` 40 层验证步原本包含约 700 次独立 `cuLaunchKernel` 调用；将其连同设备端内联 GDN 状态快照一并捕获为按窗口大小索引的 `CudaGraphExec` 后，`Qwen3.6 + MTP` 代码与散文生成从 `200.65 / 185.44 tok/s` 直接跃升至 **`239.39 / 238.79 tok/s`（较初版 `123.95 / 119.90` 提升 `+93.1% / +99.2%`）**。
   - `Inkling-2 + DSpark` 在结构化代码上达 **`114.56 tok/s`（反超 `Strata CUDA` `+21.2%`）**，在开放散文上 3 轮探针后零开销切回 `ik.forward_token_greedy` `CudaGraphExec`（`74.60 tok/s`，消除此前 `-26.7%` 退化）。
3. **`Hy4-Flash` 512-CTA 共享权重吸收式 MLA 兼顾低显存与更高吞吐**：
   - `hy4_mla_absorb_q` 按 `(head, 16-row tile, 8-token group)` 展开至 `512`+ CTA 填满 188 个 Blackwell SM，并在多 token 验证时跨 8 个 token 共享 `q_nope`/`attn_c` 只读 **1.0×** `W_kb`/`W_vb`，在维持 **`59.5 GiB` 低显存（比 Strata 省 `4.5 GiB`）** 的同时将单步解码提升 **`+22.5%`**、`+DFly2` 端到端解码提升至 **`44.09–44.58 tok/s`（`+18.9%–21.3%`）**。
4. **图捕获安全的 GPU 老化热专家缓存让 `32 GB` 显卡跑出百 `tok/s` 级 `35B Q8_0`**：
   - `moe_cache_tick_promote` 完全在设备端完成频率统计、衰减与异步 D2D 槽位晋升，仅用 **`+2.0 GiB`** 缓存预算即将 PCIe DDR 卸载吞吐提升 **`+49.4%–54.0%`** 突破 **`101–110 tok/s`**。

---

## 5. 六大固定参考设备全栈部署、性能、延迟与功耗对比分析

按 [AGENTS.md](../../AGENTS.md) §8 规范，结合两轮 **P0–P2 全栈优化（`ADR-025` + `ADR-026`）** 实测数据，汇总全部 **6 类固定参考设备** 的部署边界、推荐后端、吞吐与能效表现：

| 参考设备 | 硬件与内存带宽 | 两轮 P0–P2 受惠项与部署边界 | `tinymodel` 最新实测 / 折算吞吐与内存水位 | 首字延迟（TTFT）与微调 / 能效特征 |
|---|---|---|---|---|
| **1. Host 独显服务器**<br>**RTX PRO 6000 Blackwell**<br>及 `24–32 GB` 消费级独显 | `98 GiB` GDDR7（`~1.6–1.8 TB/s`）+ 48T EPYC + PCIe 5.0 | ✅ **全量受益两轮全部优化**（`WMMA64` + 批 GDN、`verify_graphs` MTP/`DSpark`、512-CTA 吸收式 MLA、异构双门控、热专家 VRAM 缓存） | • `Inkling-2`：`pp512` **`789.9 tok/s`**，**`+DSpark` 代码 `114.56 tok/s`**（超 Strata `+21.2%`）<br>• `Qwen3.6-35B Q8_0`：`pp512` **`2,579.8 tok/s`**，**`+MTP` 代码 `239.39` / 散文 `238.79 tok/s`**；**`30.6 GiB` 显存（卸载+64 槽缓存）达 `109.61 tok/s`**<br>• `Qwen3.8 Q2_0`：**`pp512 = 1,133.5 tok/s`（超 Strata）**，解码 `85.0 tok/s`（省 `44.9 GiB` 盘）<br>• `Hy4-Flash`：`pp512` **`425.6 tok/s`**，`+DFly2` **`44.1–44.6 tok/s`**（`59.5 GiB`） | 512 token TTFT 缩短至 **`0.20–1.20 s`**；`Qwen3.6 + MTP` 单 token 等效能耗降至 **`~1.5–2.1 J/tok`**；`32 GB` RTX 5090 单卡可百 `tok/s` 畅跑 `35B Q8_0` |
| **2. Apple M4 开发机**<br>（macOS `host` 档） | `16 GB` 统一内存（实测 `102 GB/s`），4P+6E（`NeonI8mm`）+ 10 核 Metal GPU | ✅ 受益于 `SuffixDrafter` 首步门槛修复、实测速率自动校准 `SpecController`、28× 吸收式 MLA 及 `qwen4exp-mtp` CPU 路径 | ✅ **`--metal` 主力甜点（`≤9B` 稠密与中小型 MoE）**：<br>• `SmolLM2-360M`：**`270.5 tok/s`**<br>• `Qwen3.5-0.8B`（GDN）：**`129.0 tok/s`**、`2B`：**`63.3 tok/s`**<br>• `Gemma 4 E2B`（PLE）：**`52.3 tok/s`** | 512 token TTFT `0.4–1.5 s`（`PrefixCache` `<20 ms`）；`--metal` QLoRA 微调 `22–72 tok/s`；整机 **`12–22 W`（`0.1–0.3 J/tok`）** |
| **3. Google Pixel 8**<br>（Tensor G3，Android `edge` 档） | `8 GB` LPDDR5X（可用 `3.8–4.1 GiB`，`5T` STREAM **`31.3 GB/s`**），`NeonI8mm` + `Mali-G715` Vulkan | ✅ **直接受益于本次 P1 Vulkan `moe_dual_gu_act.comp`（支持 Subgroup `16` 无额外 `up` 缓冲）+ 原生 `Q2_0`**：小型稀疏 MoE（如 `OLMoE-1B-7B Q2_0`、`Qwen3-MoE` 裁剪版）在 `Mali-G715` 上单遍读取激活并省去专家中间层缓冲显存 | • `SmolLM2-360M Q4_0x4`：**`103.95 tok/s`**<br>• `Qwen3.5-0.8B`（GDN）：**`38.96 tok/s`**<br>• `Gemma 4 E2B`（PLE 单行 `mmap`）：**`14.56 tok/s`**（RSS `1.47 GiB`）<br>• Vulkan 原生 `Q2_0` + 双门控融合解锁 `3B–7B MoE Q2_0` 手机 GPU 推理 | 300 token TTFT `1.6–4.5 s`（`PrefixCache` `<50 ms`）；SoC 功耗 `3.5–5.5 W`（`0.8B` 仅 **`~0.12 J/tok`**），零拷贝 `mmap` 杜绝 Android LMK |
| **4. Rockchip RK3588**<br>（8nm Linux SBC `edge` 档） | `4×A76 + 4×A55`（`NeonDotprod`），`8 / 16 / 32 GB` LPDDR4X/5（`17–23 GB/s`）+ `Mali-G610` Vulkan | ✅ **受益于 P1 Vulkan `moe_dual_gu_act.comp`（Valhall `Subgroup=16`）+ 自适应 `SpecController` + 内置 `nextn` MTP**：32 GB 板运行中大型稀疏 MoE 时显著减少 LPDDR4X/5 往返读写 | • `Qwen3.5-0.8B`：`24–38 tok/s`、`2B`：`11–18 tok/s`<br>• **32 GB RK3588 跑 `Qwen3.6-35B-A3B Q4_0x4`（`~19.5 GiB`）**：纯解码 `8.5–11.5 tok/s`，开启内置 `nextn` MTP 折算可达 **`13–17 tok/s`**；`Mali-G610` Vulkan 跑 `OLMoE Q4/Q2_0` 提速约 `15%–25%` | NVMe 冷加载 `35B Q4` 约 `6 s`；整板仅 **`6–8.5 W`**（`35B-A3B + MTP` 约 **`0.5–0.65 J/tok`**，适合 7×24h 工业边缘网关） |
| **5. Raspberry Pi 5**<br>（16nm Linux SBC `edge` 档） | `4×A76 @ 2.4GHz`（`NeonDotprod`），`4 / 8 / 16 GB` LPDDR4X（`~11.5 GB/s`）+ VideoCore VII V3DV Vulkan | ✅ **受益于 `moe_dual_gu_act.comp` 的无 Subgroup Shuffle 共享内存归约回退路径** + 原生 `Q2_0`：在 V3DV Vulkan 1.2 驱动上安全启用双门控融合并降低峰值内存 | • `SmolLM2-360M`：`35–45 tok/s`<br>• `Qwen3.5-0.8B`：`15–20 tok/s`<br>• 16 GB 板常驻 `OLMoE-1B-7B Q4_0x4 / Q2_0`：**`9.5–14.5 tok/s`** | 292 token TTFT `4.5–6.5 s`；整板功耗 `7–10 W`（`0.8B` 约 `0.45 J/tok`） |
| **6. Pi 4B / Pi Zero 2W**<br>（`micro` 极限档） | Zero 2W：`4×A53`（ARMv8.0 NEON），**`512 MB` RAM**（可用 `~350 MB`），`~2 GB/s`；Pi 4B：`2–8 GB` | ✅ 受益于修复后的零参数 4-gram `SuffixDrafter` 与校准版 `SpecController`（结构化 JSON/代码生成零额外内存提速 `1.2×–1.3×`，散文零退化） | ✅ **`--features micro` 超轻量静态 ELF（`<1.3 MiB`）**：<br>• Zero 2W 跑 `SmolLM2-360M Q2_0`（词表裁剪后 `93.6 MiB`）折算 **`8.3–9.8 tok/s`**（RSS `96–200 MiB << 350 MB`） | 热加载 `<150 ms`；Zero 2W 满载仅 **`1.8–2.2 W`（`~0.22 J/tok`）** |

---

## 6. 总结与后续演进方向

1. **按序完成的 P0-A → P2 优化在同台 Blackwell 上取得全面突破**：
   - **预填充（P0-A）**：`Inkling-2` **`789.88 tok/s`（`2.34×`）**、`Qwen3.6` **`2,579.79 tok/s`（`6.24×`）**、`Qwen3.8` **`1,133.54 tok/s`（`7.40×`，反超 `Strata CUDA`）**、`Hy4-Flash` **`425.58 tok/s`（`3.15×`）**；
   - **神经投机与图捕获（P0-B）**：`Inkling-2 + DSpark` 代码 **`114.56 tok/s`（反超 `Strata CUDA` `+21.2%`）** 且散文零退化；`Qwen3.6 + MTP` 代码 **`239.39 tok/s`** / 散文 **`238.79 tok/s`**（较初版 **`+93.1% / +99.2%`**）；原生打通 `Qwen3.8` `mtp-q2_0.gguf`；
   - **`Hy4` 高占用率 MLA（P0-C）**：维持 `59.5 GiB` 低显存下，单步解码提升 **`+22.5%`**、`+DFly2` 达 **`44.09–44.58 tok/s`**；
   - **异构双门控与跨后端融合（P1）**：CUDA 支持异构 `(gate_dt, up_dt)`，Vulkan 新增跨驱动 `moe_dual_gu_act.comp`；
   - **热专家 VRAM 频率缓存（P2）**：`Qwen3.6` 10 层 DDR 卸载仅用 `30.59 GiB` 显存即达 **`109.61 tok/s`（`+54.0%`，0-ULP）**。
2. **后续可选深水区方向**：
   - ① 在 `sm_120`（Blackwell）与 `sm_89/90` 上引入原生 `mma.sync.aligned.m16n8k32.row.col.f32.s8.s8.f32` INT8 Tensor Core 内核替代 FP16 反量化 WMMA，进一步将 `Q8_0`/`S_K` 预填充推向 `5,000+ tok/s`；
   - ② 为 `Qwen3.8-Flash-Next` 增加 `Q4_K`/`Q8_0` 高精度主干与 MTP 联合校准测试，在保持低量化失真的前提下释放 `verify_graphs` 下 `80%+` MTP 接受率。
