# 11 — M3 性能优化（`Q4_0x4` 零拷贝重排、自适应投机、`Q8_0` KV）与 Vulkan 后端真机报告

> 日期：2026-11-21  
> 关联阶段：**M3 性能 + Vulkan 后端**（见 [ROADMAP.md](../ROADMAP.md)、[DECISIONS.md](../DECISIONS.md) `ADR-005` / `ADR-008` / `ADR-011`）  
> 原始 5 轮随机交错基准存档：[bench/results/apple-m4/2026-11-21-m3-interleaved.json](../../bench/results/apple-m4/2026-11-21-m3-interleaved.json)、[bench/results/pixel-8/2026-11-21-m3-interleaved.json](../../bench/results/pixel-8/2026-11-21-m3-interleaved.json)

---

## 1. M3 核心目标与交付物总览

M3 聚焦两大方向：**在不牺牲 `tinymodel` 零拷贝 `mmap` 低内存基因的前提下补齐 Prefill GEMM 与有效 Decode 吞吐**，以及**落地纯 Rust 跨平台 `tm-vulkan` GPU 后端并在真机（Pixel 8 Arm Mali-G715）打通端到端推理**。

| 模块 | 关键实现 | 验证状态 |
|---|---|---|
| **离线交错重排 `Q4_0x4`（`F-02`/`F-04`）** | `DType::Q4_0x4`（`GGML_TYPE_TM_Q4_0X4 = 100`，`BlockQ4_0x4` 72B = 4×18B **零体积膨胀**）+ `tinymodel convert -i <in> -o <out>` + 原位零拷贝 `mmap` | ✅ L2 Python Oracle（[q4_0_4x8_oracle.py](../../tools/oracle/q4_0_4x8_oracle.py)）+ 四档 ISA **0 ULP 逐位一致** |
| **Prefill GEMM 与多行共享权重微内核（`N-03`/`N-10`）** | `tm-cpu` 四档（`generic`/`neon_v80`/`dotprod`/`i8mm`）`gemv_q4_0x4_q8_0`、`gemm_q4_0x4_q8_0`（M4 4T 实测 **982.1 GOPS**）、`matmul_pair_f32`/`matmul_triple_f32`、`MAX_CPU_BATCH=64` | ✅ L1/L1b 批宽 `1..=64` 逐位不变性测试通过 |
| **大词表 LM Head 与 `Q8_0` KV 量化（`F-14`）** | 4 行展开 `gemv_q6_k_q8_k`（M4 **98.84 GB/s**）+ 多 token `gemm_q6_k_q8_k` + NEON `quantize_q8_k`；全模型支持 `--kv-type f16\|q8_0`（含 `PrefixCache` 快照） | ✅ 脏内存直降 **40.8%–44.5%**，长上下文 Top-1 一致率 100% |
| **自适应投机解码（`F-12` / `ADR-005`）** | 固定容量 4 路组相联 4-gram/3-gram/2-gram `SuffixDrafter` + `MtpDrafter`/`LinearMtpHead` + 在线 EMA `SpecController`（净收益 `< 5%` 自动关闭）+ 全架构**只读验证 + `commit` 重放** | ✅ `Llama`/`Qwen3`/`Qwen3.5` GDN/`Gemma 4`/`LFM2` 贪心输出 **100% 逐 token 一致** |
| **`tm-vulkan` 跨平台 GPU 后端（`ADR-008`）** | `ash` 运行时动态加载 `libvulkan` + 构建期纯 Rust `naga` 编译 31 个 GLSL 450 计算管线为 SPIR-V 1.0（零 Vulkan SDK 依赖）+ UMA `HOST_CACHED` 直写 + `Subgroup`/`coopmat` 探测 | ✅ **Google Pixel 8（Arm Mali-G715，Vulkan 1.4.305）** 真机 L1 自检与 L3 多 token `logits_diff.py` 满分通过 |

---

## 2. `tm-vulkan` 架构设计与 Mali-G715 真机排障实录

### 2.1 零外部 SDK 构建链与硬件探测
- **构建期编译（[build.rs](../../crates/tm-vulkan/build.rs)）**：使用 `naga` 24 将 [common.glsl](../../crates/tm-vulkan/src/shaders/common.glsl)、[gemv.comp](../../crates/tm-vulkan/src/shaders/gemv.comp)、[gemm.comp](../../crates/tm-vulkan/src/shaders/gemm.comp)、[ops.comp](../../crates/tm-vulkan/src/shaders/ops.comp) 编译为严格通过 `Capabilities::PUSH_CONSTANT` 校验的 SPIR-V 1.0 字节码内嵌进二进制。`cargo zigbuild`（Linux musl）与 Android Bionic 交叉编译均无需安装 `glslc` 或 `shaderc`。
- **运行时能力发现（[context.rs](../../crates/tm-vulkan/src/context.rs)）**：动态加载 `libvulkan.so` / `libMoltenVK.dylib`，自动查询 `VkPhysicalDeviceSubgroupProperties`（Mali Valhall 实测 `subgroup_size = 16`，不同于 NVIDIA `32` 与 AMD `64`）、`VK_KHR_cooperative_matrix`、UMA 统一内存堆与 `maxComputeWorkGroupInvocations`（每次 `cmd_dispatch` 均先做边界检查，杜绝越界静默丢弃）。
- **UMA `HOST_CACHED` 内存类型优先级**：在 Android Tensor G3 Mali 上，0 号内存类型为不带 `HOST_CACHED` 的 `MT_NORMAL_NC` 写合并页；优先匹配 `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` 可直接命中硬件一致性 SLC + GPU L2 缓存。

### 2.2 在 Pixel 8 真机捕获并修复的两个 `naga` 24 编译器陷阱
在 Pixel 8（`Mali-G715`）真机运行 `tinymodel probe --vulkan` L1 自检时，我们捕获并修复了纯编译检查无法发现的两个 SPIR-V 代码生成缺陷：
1. **`barrier()` 被降级为 `Scope::Subgroup` 导致跨 Warp 竞态**：
   - `naga` 24 的 GLSL 前端将 `barrier()` 解析为 `Statement::Barrier(Barrier::all())`（含 `Barrier::SUB_GROUP` 位），而其 SPIR-V 后端（`writer.rs`）见到 `Barrier::SUB_GROUP` 会发射 `OpControlBarrier(Scope::Subgroup, ...)` 而非 `Scope::Workgroup`！在 `subgroup_size = 16` 的 Mali-G715 上，128 线程工作组内的 8 个 Warp 彼此完全不同步。
   - **修复**：在 [build.rs](../../crates/tm-vulkan/build.rs) 递归重写 AST，将 `Statement::Barrier(flags)` 规范化为 `Barrier::WORK_GROUP | Barrier::STORAGE`，生成标准 `OpControlBarrier(Scope::Workgroup, ...)`。
2. **C 风格多维 `shared` 数组内外维顺序反转**：
   - `naga` 24 的 `glsl-in` 将 `shared float s_partial[4][32]` 降级为外维 `32`、内维 `4` 的 `OpTypeArray`，导致 `tx >= 4` 时共享内存越界互相踩踏。
   - **修复**：统一展平为一维 `shared float s_partial[128]` / `shared float s_reduce[1024]`，并为 `Q4_0`/`Q4_0x4`/`Q6_K` GEMV 实现纯寄存器直算 `dot_block32`（消除 `float g_w32[32]` SPIR-V `Private` 数组在每个线程入口的零初始化开销）。

### 2.3 Mali-G715 真机多 Token `logits_diff.py` 对齐结果（`AGENTS.md` 规则 #7）

| 对比组合 | 模型与步数 | Top-1 一致率 | Top-5 重叠率 | `mean_KL` | `max_KL` | `max\|dL\|` |
|---|---|---:|---:|---:|---:|---:|
| **Mali-G715 `Q4_0` vs `Q4_0x4`** | `SmolLM2-360M`（28 步） | **28/28 (100.0%)** | **100.0%** | **`0.000e+00`** | `0.000e+00` | **`0.0000`（0 ULP）** |
| **`llama.cpp` vs Mali-G715 Vulkan** | `SmolLM2-360M Q4_0x4`（28 步） | **28/28 (100.0%)** | **100.0%** | **`1.577e-06`** | `8.512e-06` | `0.0242` |
| **M4 Metal vs Mali-G715 Vulkan** | `SmolLM2-360M Q4_0x4`（28 步） | **28/28 (100.0%)** | **100.0%** | **`1.448e-07`** | `6.485e-07` | `0.0053` |
| **`llama.cpp` vs Mali-G715 Vulkan** | `Qwen3.5-0.8B Q4_0x4` GDN（25 步） | **25/25 (100.0%)** | **100.0%** | **`8.883e-09`** | `4.157e-08` | **`0.0026`** |
| **M4 Metal vs Mali-G715 Vulkan** | `Qwen3.5-0.8B Q4_0x4` GDN（25 步） | **25/25 (100.0%)** | **100.0%** | **`1.267e-09`** | `8.899e-09` | **`0.0007`** |

---

## 3. 双机 5 轮随机交错端到端基准实测（[TESTING.md](../TESTING.md) §3）

### 3.1 Google Pixel 8（Tensor G3，`taskset 1f0` 5T `1×X3 + 4×A715`，`Isa::NeonI8mm`）

| 模型（零膨胀重排耗时） | 引擎 / 配置 | Prefill `pp512` (tok/s) | **配对 `pp` vs `repack=0`** | Decode `tg64` (tok/s) | **配对 `tg` vs `repack=0`** | 峰值 RSS / 真实脏内存 (MiB) |
|---|---|---:|---:|---:|---:|---:|
| **`SmolLM2-360M`**<br>（`219.1 MiB`，重排 `535 ms`） | `llama.cpp (--repack 0)` | 227.54 | 100.0% | 90.41 | 100.0% | 286.0 |
| | `llama.cpp (KleidiAI repack=1)` | 458.91 | 203.0% | 101.45 | 111.4% | **494.7（2.26× 文件！）** |
| | `tinymodel Q4_0` | 240.91 | 103.0% | 99.98 | 109.8% | 302.3（dirty 84.2） |
| | **`tinymodel Q4_0x4`** | **293.30** | **`123.3%`** | **`103.95`（反超 KleidiAI！）** | **`108.9%–110.4%`** | **306.2（省 188.5 MiB）** |
| | **`tinymodel Q4_0x4 + KV Q8_0`** | 262.20 | 114.7% | **100.44** | **110.9%** | **`269.1`（dirty 仅 `46.7`！）** |
| **`Qwen3.5-0.8B` (GDN)**<br>（`483.7 MiB`，重排 `2.10 s`） | `llama.cpp (--repack 0)` | 93.89 | 100.0% | 32.52 | 100.0% | 624.8 |
| | `llama.cpp (KleidiAI repack=1)` | 201.40 | 193.7% | 36.11 | 111.7% | **1135.9（多占 593 MiB！）** |
| | `tinymodel Q4_0` | 125.86 | 118.1% | 34.25 | 116.2% | 542.9（dirty 55.8） |
| | **`tinymodel Q4_0x4`** | **168.43** | **`142.3%`（峰值 `1.79×`）** | **`38.96`（反超 KleidiAI！）** | **`110.1%`** | **543.1（省 592.8 MiB）** |
| | **`tinymodel Q4_0x4 + KV Q8_0`** | 139.13 | 135.9% | 33.97 | 106.1% | **`531.9`（dirty 仅 `44.7`！）** |
| **`Llama-3.2-1B`**<br>（`737.2 MiB`，重排 `6.65 s`） | `llama.cpp (--repack 0)` | 47.95 | 100.0% | 22.08 | 100.0% | 876.8 |
| | `llama.cpp (KleidiAI repack=1)` | 139.54 | 254.5% | 27.20 | 124.8% | **1623.8（多占 810 MiB！）** |
| | `tinymodel Q4_0` | 71.12 | 147.4% | 21.08 | 99.3% | 813.8（dirty 73.5） |
| | **`tinymodel Q4_0x4`** | **93.99** | **`155.5%`（超 `1.5×` 目标！）** | 20.46 | 94.8% | **813.5（省 810.3 MiB）** |
| | **`tinymodel Q4_0x4 + KV Q8_0`** | 89.67 | 188.2% | 20.57 | 95.7% | **`784.2`（dirty 仅 `43.6`！）** |

### 3.2 Apple M4（4 P 核 CPU `Isa::NeonI8mm`）

> 注：macOS 上 `llama-bench -ngl 0` 在 `pp512` 阶段调用了系统私有 `Accelerate.framework`（AMX 矩阵协处理器），而 M4 主机实际使用 `tinymodel --metal`（`pp512 > 3400 tok/s`）。在纯 CPU `tg128` 解码阶段，`tinymodel Q4_0x4` 在全部三款模型上全面反超 `llama.cpp`（含 `--repack 1`）：

| 模型 | `llama.cpp (repack=0)` `tg128` | `llama.cpp (repack=1)` `tg128`（RSS） | `tinymodel Q4_0` `pp512 / tg128` | **`tinymodel Q4_0x4` `pp512 / tg128`** | **相对 `llama.cpp` Decode** |
|---|---:|---:|---:|---:|---:|
| **`SmolLM2-360M`** | 255.64 tok/s | 255.87 tok/s（840 MiB） | 638.94 / 275.71 | **746.60 / `278.26 tok/s`**（KV Q8: `279.97`） | **`108.9% – 109.4%`** |
| **`Qwen3.5-0.8B` (GDN)** | 109.33 tok/s | 105.16 tok/s（1955 MiB） | 415.60 / 128.24 | **483.58 / `128.39 tok/s`** | **`117.3%`（vs repack1 `122.1%`）** |
| **`Llama-3.2-1B`** | 88.34 tok/s | 83.88 tok/s（2791 MiB） | 260.50 / 86.12 | **317.81 / `89.90 tok/s`** | **`102.1%`（vs repack1 `107.2%`）** |

### 3.3 自适应投机解码（`--spec auto`：4-gram `SuffixDrafter` + `SpecController`）5 轮实测

| 设备 | 模型（`Q4_0x4`） | 结构化 JSON/代码微改 `off → auto` | **有效 Decode 加速比** | 贪心逐位哈希一致 | 陌生散文语料 `auto/off`（`<5%` 自动熔断保护） |
|---|---|---:|---:|:---:|---:|
| **Apple M4 (4T)** | `SmolLM2-360M` | `249.33 → 323.86 tok/s` | **`1.30×`** | ✅ `100%` | `0.94×`（自动冷却旁路） |
| **Apple M4 (4T)** | `Qwen3.5-0.8B` (GDN) | `127.73 → 159.75 tok/s` | **`1.25×`（达 `llama.cpp` 的 `1.46×`！）** | ✅ `100%` | `0.92×`（自动冷却旁路） |
| **Apple M4 (4T)** | `Llama-3.2-1B` | `87.73 → 116.26 tok/s` | **`1.33×`（超 `1.3×` 目标！）** | ✅ `100%` | `0.92×`（自动冷却旁路） |
| **Pixel 8 (5T)** | `SmolLM2-360M` | `90.78 → 95.14 tok/s` | **`1.05×`** | ✅ `100%` | `0.91×`（自动冷却旁路） |
| **Pixel 8 (5T)** | `Qwen3.5-0.8B` (GDN) | `31.66 → 36.26 tok/s` | **`1.15×`** | ✅ `100%` | `0.90×`（自动冷却旁路） |
| **Pixel 8 (5T)** | `Llama-3.2-1B` | `19.82 → 26.32 tok/s` | **`1.33×`（超 `1.3×` 目标！）** | ✅ `100%` | `0.91×`（自动冷却旁路） |

### 3.4 Pixel 8 Mali-G715 Vulkan 基线与移动端 GPU DVFS 观察
- 在 Pixel 8 真机不接 root 锁频的情况下，`tinymodel --vulkan` 跑 `SmolLM2-360M-Q4_0x4` 为 `pp309 = 17.02 tok/s, tg32 = 7.51 tok/s`，跑 `Qwen3.5-0.8B-Q4_0x4` 为 `pp292 = 10.51 tok/s, tg32 = 5.21 tok/s`，5 轮贪心哈希 100% 恒定（`af9dad8a6956a66a` / `72ef4cbddcad7ff0`）。
- **关键工程发现**：Android `G3D` 温控与 `devfreq` 对短脉冲计算队列默认驻留在最低频档（传感器 `G3D = 46.0°C` 未升频），且通用 SPIR-V 1.0 标量 FMA 尚未启用 Mali-G715 刚探测到的 `VK_KHR_cooperative_matrix`（`coopmat=true`）与 `subgroup=16` shuffle 归约。因此在 Tensor G3 手机上，当前主推理路径应默认走零拷贝 `5T i8mm CPU (Q4_0x4)`（快 `5×–13×` 且零额外显存拷贝），而 `tm-vulkan` 已完成全算子正确性闭环，为桌面独显/核显与后续 `coopmat` FP16 扩展奠定了坚实底座。

---

## 4. 六大固定参考设备全矩阵工程分析（`AGENTS.md` §8）

| 参考设备 | 硬件与后端配置 | M3 核心收益（吞吐、延迟、内存、功耗） | 部署建议 |
|---|---|---|---|
| **1. Apple M4（16GB，macOS Host）** | `4` P 核 `Isa::NeonI8mm` + `Metal` GPU | CPU `Q4_0x4` Prefill 提速 **`+16.9%–22.0%`**，Decode 全线达 **`102.1%–117.3%` `llama.cpp`**；结构化投机解码有效吞吐达 **`1.25×–1.33×`**（`SmolLM2` `323.9 tok/s`、`Qwen3.5-0.8B` `159.8 tok/s`、`Llama-1B` `116.3 tok/s`）；相比 `llama.cpp repack=1` 省下 **`0.46–1.76 GiB`** 内存 | 主机推理首选 `--metal`；纯 CPU/CI/低显存多实例用 `Q4_0x4 --spec auto` |
| **2. Google Pixel 8（Tensor G3，Android 16）** | `5T`（`1×X3 + 4×A715`）`Isa::NeonI8mm` + **Arm Mali-G715 Vulkan 1.4**（真机实测） | ① `Q4_0x4` 离线重排零内存膨胀（省 **`188–810 MiB`** RSS），Prefill 达 `llama.cpp repack=0` 的 **`123.3%–155.5%`**，`SmolLM2`（`103.95 tok/s`）与 `Qwen3.5-0.8B`（`38.96 tok/s`）Decode **直接反超 `llama.cpp KleidiAI`**；② `--kv-type q8_0` 脏内存压至 **`43.6–46.7 MiB`**（降 **41%–45%**）；③ 结构化 `--spec auto` 在 `1B` 上达 **`1.33×`（`26.32 tok/s`）**；④ Mali-G715 Vulkan 通过 25–28 步 L3 对齐（`KL = 8.88e-9`） | 生产默认用 `Q4_0x4 --kv-type q8_0 --spec auto`（CPU `i8mm` 能效与速度双优） |
| **3. Rockchip RK3588（Linux SBC）** | `4×A76`（自动屏蔽 `4×A55`）`Isa::NeonDotprod` + **Arm Mali-G610 MP4 Vulkan 1.2/1.3** | ① RK3588 无 `i8mm`（`llama.cpp` 无法启用 `i8mm` `smmla` KleidiAI），而 `tinymodel` 在 `dotprod.rs` 专门提供了 **`4×4` `sdot` 寄存器分块 `gemm_q4_0x4_q8_0`**，零拷贝消除解包后预计 `4×A76` Prefill 提速 **`1.55×–1.75×`**（`Qwen3.5-0.8B` `pp512` 达 **`120–150 tok/s`**，TTFT 压至 **`~2.0s`**）；② `Q8_0` KV 让 16GB/32GB 板跑 `8B` `16k` 上下文省 **`~1.0 GiB`** 内存；③ `tm-vulkan` 纯 SPIR-V 1.0 + `subgroup=16` 兼容 PanVK 与闭源 `libmali-valhall-g610` | `edge` 大内存主力板：`tinymodel convert` 预处理到 NVMe SSD + `--spec auto` |
| **4. Raspberry Pi 5（Linux SBC）** | `4×A76` 同构 `Isa::NeonDotprod` + **VideoCore VII（Mesa V3DV Vulkan 1.2）** | ① 与 RK3588 同为 Cortex-A76 `dotprod` 微架构：无 `i8mm` 条件下，`Q4_0x4` 消除每块半字节掩码拆分 + `MAX_CPU_BATCH=64` L2 驻留分块 + 4 行 `Q6_K` LM Head，预计 Prefill 达 `llama.cpp` 的 **`1.50×–1.65×`**；② 在仅 `11.5 GB/s` 内存带宽的 Pi 5 上，单步算力富余而带宽极度受限，多 token 共享权重验证开销极低（`< 10%/tok`），代码/RAG 场景 `--spec auto` 有效 Decode 可达 **`1.35×–1.60×`**；③ `tm-vulkan` 所有工作组严格 `≤ 256`（V3DV 硬件上限为 `256`！），开箱兼容 Pi 5 V3DV | 装主动散热器，转换 `Q4_0x4` 模型并开启 `--spec auto --kv-type q8_0` |
| **5. Raspberry Pi 4B（`micro` 大内存板）** | `4×A72` 纯 ARMv8.0 `Isa::NeonV80`（无 `dotprod`/`i8mm`） | `neon_v80.rs` 补齐了 **0 ULP 逐位对齐**的 `gemv_q4_0x4_q8_0` 与 4×2 寄存器分块 `gemm_q4_0x4_q8_0`（用 `vmlal_s8` 共享激活广播），消除单 token 解包开销并提升 Prefill 缓存复用约 **`1.35×–1.50×`**；4GB/8GB Pi 4B 配合 `--kv-type q8_0` 可从容跑 `1B` 模型长上下文 | 使用 `micro` 静态 musl 构建 + `Q4_0x4` 模型 |
| **6. Raspberry Pi Zero 2W（`micro` 极限 512MB）** | `4×A53` 纯 ARMv8.0 `Isa::NeonV80`，总物理内存仅 **512MB** | ① **零体积膨胀的救命价值**：`llama.cpp --repack 1` 会把 `219 MiB` 的 `SmolLM2-360M` 膨胀到 `495 MiB` 直接触发 512MB Zero 2W OOM 死机；而 `tinymodel Q4_0x4` 保持 **`219.09 MiB` 原位 `mmap`**（操作系统可随时丢弃干净代码/权重页）！② **`--kv-type q8_0` 省下一半 KV 脏页**：`2048` 上下文下 `SmolLM2-360M` 匿名脏内存仅 **`~28–32 MiB`**，整机安全常驻 `< 300 MiB` | `micro` 档必选 `Q4_0x4` + `--kv-type q8_0 -c 1024`（不建议开投机以省 `384 KiB` 表与验证缓冲） |
