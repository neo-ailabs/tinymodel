# Raspberry Pi 4B（BCM2711）部署要求与优化分析

> 日期：2026-11-20  
> 关联代码：`crates/tm-cpu/src/{isa.rs,registry.rs,aarch64/neon_v80.rs}`、`crates/tm-cli`

---

## 1. 硬件能力边界（BCM2711 vs Zero 2W / Pi 5）

Raspberry Pi 4 Model B 在 `tinymodel` 的分档体系中处于 **“指令集同属 `micro`（ARMv8.0-A），内存容量跨越 `micro` 到 `edge`（1GB–8GB）”** 的交叉位置：

| 维度 | Pi Zero 2W (`micro`) | **Raspberry Pi 4B** | Raspberry Pi 5 (`edge`) |
|---|---|---|---|
| **SoC / CPU** | RP3A0：4× Cortex-A53 @ 1.0 GHz（顺序双发射） | **BCM2711：4× Cortex-A72 @ 1.5 / 1.8 GHz（乱序三发射）** | BCM2712：4× Cortex-A76 @ 2.4 GHz（乱序四发射） |
| **缓存层级** | 32 KB L1D + 512 KB L2 | **32 KB L1D/核 + 1 MB 共享 L2**（无 L3） | 64 KB L1D/核 + 512 KB L2/核 + 2 MB L3 |
| **内存规格** | 512 MB LPDDR2 | **1 GB / 2 GB / 4 GB / 8 GB LPDDR4-3200** | 4 GB / 8 GB / 16 GB LPDDR4X-4267 |
| **实测持续读带宽** | ~2.0 GB/s | **~4.5 – 5.5 GB/s**（4 核 STREAM） | ~11.5 GB/s |
| **ISA 指令集** | ARMv8.0-A NEON | **ARMv8.0-A NEON**（含基础 FP16 转换 `fcvtl`/`fcvtn`） | ARMv8.2-A：NEON + **`dotprod` (`sdot`)** + FP16 算术 + LSE |
| **缺失的关键指令** | 无 `dotprod` / `i8mm` / LSE | ❌ **无 `dotprod`**、❌ **无 `i8mm`**、❌ **无 FP16 算术**、❌ **无 ARMv8.1 LSE 原子指令**、❌ 无硬件 AES/SHA | ❌ 无 `i8mm` / SVE |
| **GPU 可用性** | VideoCore IV（不可用） | **VideoCore VI（V3D 4.2，Vulkan 1.2）——不建议用于 LLM**¹ | VideoCore VII（V3D 7.1，Vulkan 1.2/1.3） |

> ¹ **为什么 Pi 4B 不用 VideoCore VI Vulkan 跑 LLM**：① 与 CPU 共享同一条 ~5.5 GB/s LPDDR4 总线，无独立显存带宽增益；② V3D 4.2 QPU 缺乏 8-bit 整数点积、Subgroup shuffle 及 Cooperative Matrix 硬件，且单次 SSBO 绑定上限与驱动调度开销高于 4 核 Cortex-A72 NEON 直接从统一内存流式读取。因此 Pi 4B 应**固定走纯 CPU 4 线程 `Isa::NeonV80` 路径**。

---

## 2. 系统、内核与运行环境硬性要求

### 2.1 操作系统与 64 位 ABI（现场部署头号陷阱）
1. **必须运行 64 位内核（`uname -m` 必须为 `aarch64`）**：
   - 树莓派 4B 早期存量设备大量安装了 32 位 `armhf`（`armv7l`）系统。
   - **32 位模式的两个致命限制**：① 单进程虚拟地址空间上限仅 2–3 GB，无法直接 `mmap` ≥1.5B 的 GGUF 模型；② ARM32 NEON 只有 **16 个** 128-bit 向量寄存器（AArch64 有 **32 个** `v0–v31`），GEMV/GEMM 累加器严重寄存器溢出。
2. **推荐系统**：Raspberry Pi OS 64-bit（Bookworm / Trixie Lite，无桌面环境可省下 ~250 MB 内存）、Ubuntu Server 24.04 arm64 或 DietPi arm64。
3. **存量 32 位系统的免重装兼容技巧**：
   - 若用户 SD 卡是旧版 32 位 Raspberry Pi OS 用户态（`armhf`），只需在 `/boot/firmware/config.txt`（或 `/boot/config.txt`）加入 `arm_64bit=1` 重启进入 64 位内核；
   - 由于 `tinymodel` 编译产物为 **`aarch64-unknown-linux-musl` 纯静态链接 ELF**（零 glibc / `.so` 依赖，体积仅 **~760 KB**），**64 位内核可以直接运行该静态二进制，无需重装 64 位用户态系统**。

### 2.2 编译目标与二进制分发要求
- **交叉编译三元组**：`aarch64-unknown-linux-musl`（推荐，零依赖单文件拷贝即跑）或 `aarch64-unknown-linux-gnu`（需宿主机 `glibc >= 2.31`）。
- **严禁全局开启 `target-feature=+lse` 或 `+dotprod`**：
  - BCM2711（ARMv8.0）没有 ARMv8.1 LSE 原子指令（`casal`、`ldadd`、`swp`）。若直接带 `-C target-cpu=cortex-a76` 或 `+lse` 编译非受控代码，在 `Mutex` / `ThreadPool` 处会立即触发 **`SIGILL`（非法指令崩溃）**。
  - **两种合法发布形态**：
    1. **通用 ARM64 单二进制**（符合 N-07）：默认基础目标为 ARMv8.0，`dotprod`/`i8mm` 仅存在于带 `#[target_feature]` 的函数内，由 [isa.rs](../../crates/tm-cpu/src/isa.rs) 通过 Linux `getauxval(AT_HWCAP)` 运行时探测——在 Pi 4B 上自动落入 `Isa::NeonV80`，同一个二进制同时通吃 Zero 2W、Pi 4B、Pi 5 与 RK3588。
    2. **Pi 4B 专属裁剪二进制**：`RUSTFLAGS="-C target-cpu=cortex-a72" cargo zigbuild --release -p tm-cli --target aarch64-unknown-linux-musl --no-default-features --features micro`，针对 Cortex-A72 乱序三发射流水线做指令调度，且剔除更高阶 ISA 代码。

### 2.3 存储介质、Swap 与内存配置
- **存储 I/O（影响冷启动 `mmap` 缺页耗时）**：
  - 普通 MicroSD 卡顺序读仅 `40–85 MB/s`，首次加载 `Qwen3.5-2B-Q4_0`（1.16 GiB）缺页读盘需 `14–28 秒`；
  - 强烈建议使用 **USB 3.0 SSD（支持 UASP，顺序读 `~350 MB/s`）**，冷启动可压至 `~3.5 秒`。
- **关闭 SD 卡文件 Swap**：
  - Raspberry Pi OS 默认开启 `dphys-swapfile`（SD 卡上的 `/var/swap`）。一旦模型 + KV cache 逼近物理内存上限，Linux 回收只读 `mmap` 模型页与写入 SD swap 会导致吞吐从 `8 tok/s` 暴跌至 `< 0.1 tok/s` 并加速烧毁 SD 卡。
  - 部署要求：`sudo dphys-swapfile swapoff && sudo systemctl disable dphys-swapfile`（如需兜底仅允许使用内存压缩 `zram`）。
- **显存切分（1GB / 2GB 板型必调）**：
  - Headless 场景在 `config.txt` 设置 `gpu_mem=16`，释放默认占用的 64–76 MB 连续物理内存。
- **容器 / cgroup v2 限制**（对应 `AGENTS.md` §9）：
  - 在 Docker / Home Assistant OS / k3s 中部署于 Pi 4B 时，默认需挂载 `/sys/fs/cgroup` 只读；引擎内存规划器必须优先读取 `/sys/fs/cgroup/memory.max` 并与 `/proc/meminfo` 的 `MemAvailable` 取较小值，避免被容器 OOM Killer 强杀。

### 2.4 供电、散热与频率策略（直接决定持续吞吐）
- **供电**：必须使用官方 **5.1V / 3A（15W）USB-C** 电源；部署自检需确认 `vcgencmd get_throttled` 返回 `0x0`（欠压会强制锁频至 600 MHz）。
- **主频解锁（`1.5 GHz -> 1.8 GHz`，免费 +20% 算力）**：
  - Pi 4B Rev 1.4/1.5（及部分更新固件板卡）在 `config.txt` 中设置 `arm_boost=1` 可将 4 核满载频率从 `1500 MHz` 提升至 **`1800 MHz`**，对受算力限制的 Prefill 和 NEON 解包有直接线性收益。
- **主动散热**：
  - 4 核 Cortex-A72 持续跑满 NEON GEMV 功耗约 `6–7.5 W`；裸板无风扇约 **45–90 秒**即触及 **80°C 软降频阈值**（85°C 硬降频至 1.0 GHz），吞吐损失 30%–45%。**必须配备金属散热壳或主动风扇**，维持核心温度 `< 70°C`。
- **CPU 调频器（Governor）**：
  - Raspberry Pi OS 默认 `ondemand` 调频器在多轮对话间隔会有升频延迟，建议设置为 `schedutil` 或 `performance`；线程数固定 `-t 4`（物理 4 核，无超线程）。

---

## 3. 各内存版本（1GB–8GB）可跑模型与吞吐预期

按 Pi 4B 持续读带宽 `5.0 GB/s`、有效带宽利用率 `80%`（即 **`4.0–4.4 GB/s`** 有效模型流式读取带宽）推算：

| Pi 4B 规格 | Headless 可用内存 | 推荐模型（Q4_0） | 权重体积 | 实测运行时常驻脏页（含 248k 词表 + KV/GDN + 缓冲） | 预期 Decode (`tg`) | 预期 Prefill (`pp`, 1.8GHz) |
|---|---:|---|---:|---:|---:|---:|
| **1 GB** | ~720 MiB | `SmolLM2-135M / 360M`<br>**`Qwen3.5-0.8B`** (`-c 1024`，扁平词表后可跑！) | 90 / 219 MiB<br>484 MiB | **~26–30 MiB** (`-c 1024`)<br>**51.8 MiB** (`-c 1024`) | **18 – 38 tok/s**<br>**7.5 – 9.0 tok/s** | ~45 – 90 tok/s<br>~28 – 35 tok/s |
| **2 GB** | ~1.65 GiB | **`Qwen3.5-0.8B`** (`-c 4096`)<br>`Llama-3.2-1B` / `Qwen3-0.6B` | 484 MiB<br>737 MiB | **~65 MiB** (`-c 4096`)<br>~95 MiB (`-c 2048`) | **7.5 – 9.0 tok/s**<br>**5.5 – 6.2 tok/s** | ~28 – 35 tok/s<br>~22 – 28 tok/s |
| **4 GB** | ~3.55 GiB | **`Qwen3.5-2B`** (GDN)<br>`Llama-3.2-3B`<br>`Gemma 4 E2B`（需 USB3 SSD 按需读 PLE） | 1159 MiB<br>~1850 MiB<br>2710 MiB¹ | **~95 MiB** (`-c 2048`)<br>~150 MiB (`-c 2048`)<br>~140 MiB | **3.5 – 3.9 tok/s**<br>**2.2 – 2.5 tok/s**<br>**~2.8 – 3.2 tok/s** | ~12 – 15 tok/s<br>~8 – 10 tok/s<br>~12 tok/s |
| **8 GB** | ~7.45 GiB | `Qwen3.5-4B` / `Phi-4-mini`<br>`Gemma 4 E2B`（全驻留页缓存） | ~2.3–2.7 GiB | ~150–250 MiB | **1.6 – 3.3 tok/s** | ~6 – 12 tok/s |

> ¹ `Gemma 4 E2B Q4_0` 总文件 `2.71 GiB`，在 4GB 板型上与 OS 页缓存较紧；但由于其中约 `1.35 GiB` 为 PLE 词表嵌入（每步仅按 token ID 查单行），只要配合 USB3 SSD + 零拷贝 `mmap` 即可在 4GB 板型稳定运行，8GB 板型则可整文件缓存在内存中。

---

## 4. `tinymodel` 针对 Pi 4B / Zero 2W 的代码审计与已落地优化（2026-11-20 ✅）

通过本机反汇编检查 `aarch64-unknown-linux-musl` 产物及运行 `TM_FORCE_ISA=neon`（对应 Pi 4B 的 `Isa::NeonV80` 路径）剖析，定位并**已全部实现验证**以下 4 项改进：

### ✅ P0：补齐 `Isa::NeonV80` 下的 `Q8_0` / `Q4_1` / `Q5_K` / `Q6_K` 纯 ARMv8.0 NEON 内核
- **根因**：此前 [registry.rs](../../crates/tm-cpu/src/registry.rs) 的 `NEON_V80_TABLE` 中，`gemv_q8_0_q8_0`、`gemv_q4_1_q8_1`、`gemv_q5_k_q8_k`、`gemv_q6_k_q8_k` 均回退至标量 `generic`。
- **实现**：在 [neon_v80.rs](../../crates/tm-cpu/src/aarch64/neon_v80.rs) 中基于纯 ARMv8.0 `vmull_s8` / `vmlal_s8` / `vpaddlq_s16` / `vpadalq_s16` 实现了 4 行展开 `Q8_0`（防 `-128 × -128` 溢出的 `dot32_s8_full_v80`）、4 行展开 `Q4_1`、2 行展开 `Q5_K` 与 2 行展开 `Q6_K`，全部通过 L1 逐位一致测试（[parity_q4_0.rs](../../crates/tm-cpu/tests/parity_q4_0.rs)）。
- **同机实测收益（`TM_FORCE_ISA=neon`, 4 线程）**：
  - 微基准：`Q8_0` GEMV `21.77 -> 76.15 GB/s`（**3.50×**，达 `dotprod` 的 98.7%）；`Q4_1` GEMV `13.52 -> 56.66 GB/s`（**4.19×**）；`Q6_K` GEMV `50.06 -> 79.69 GB/s`（**1.59×**）。
  - `SmolLM2-360M-Q4_0`：`lm_head` (`Q8_0`) 耗时 `2193.3 µs -> 687.6 µs`（**3.19×**），端到端 Decode 从 **`172.01 tok/s` 跃升至 `248.24–257.78 tok/s`（+44.3% 提速，已达 `i8mm` 的 95.3%）**。
  - `Qwen3.5-0.8B-Q4_0`：`ffn_down` 耗时 `1559.6 µs -> 906.2 µs`（**-41.9%**），端到端 Decode 从 **`88.93 tok/s` 提升至 `106.17–107.87 tok/s`（+19.4% 提速，已达 `i8mm` 的 95.6%）**。

### ✅ P1：修复 `tm-models/Cargo.toml` 的 `micro` feature 隔离
- **根因**：此前 [crates/tm-models/Cargo.toml](../../crates/tm-models/Cargo.toml) 未对 `tm-cpu` 声明 `default-features = false`，导致 `--no-default-features --features micro` 仍合并进 `edge = ["dotprod", "i8mm"]`。
- **实现与验证**：补上 `default-features = false` 后，`RUSTFLAGS="-C target-cpu=cortex-a72" cargo zigbuild --release -p tm-cli --target aarch64-unknown-linux-musl --no-default-features --features micro` 产出的纯静态 ELF 仅 **`745 KB`**，`llvm-objdump -d` 检出 `sdot`/`udot`/`smmla`/`ummla` 指令数为 **`0`**（编译器自带 outline atomics 亦通过 `__aarch64_have_lse_atomics` 运行时回退 `ldaxr`/`ldxr`，在 ARMv8.0 上 100% 安全）。

### ✅ P2：GGUF 零拷贝 `StringSpans` + 扁平连续池 `Tokenizer`（消灭 ~100 万次堆分配）
- **根因**：`Qwen3.5` 词表高达 `248,320`（加 `247,574` 条 BPE merges），旧实现中 `tm-gguf` 与 `tm-tokenizer` 合计分配约 **99.2 万个独立 `String`**，吃掉 **~95 MiB** 不可回收脏页内存。
- **实现**：
  1. [tm-gguf/src/lib.rs](../../crates/tm-gguf/src/lib.rs) 新增零拷贝 `MetadataValue::StringSpans`（直接记录只读 `mmap` 内的 `(offset, len)` 切片跨度）及紧凑定长 `U32Array` / `I32Array` / `F32Array`；
  2. [tm-tokenizer/src/lib.rs](../../crates/tm-tokenizer/src/lib.rs) 重写为单块连续 `vocab_pool: String` + `vocab_offsets: Vec<u32>` + 开放寻址 `u32` 哈希表 + `u64 -> (u32, u32)` 整数对 BPE 合并表（全程 `#![forbid(unsafe_code)]`）。
- **同机实测收益（`/usr/bin/time -l`，`Qwen3.5-0.8B`）**：
  - Metal 路径：Max RSS 从 `193.9 MiB` 降至 **`106.9 MiB`**，Dirty Footprint 从 `258.8 MiB` 骤降至 **`171.7 MiB`（净省 `87.2 MiB` = -33.7% 脏内存）**，启动阶段退休指令数减少 **4.21 亿条（-30.9%）**；
  - CPU 路径（`TM_FORCE_ISA=neon`, `-c 1024`，含完整 248k 词表 + KV + GDN 状态）：全进程 `peak memory footprint`（纯脏页内存）仅 **`51.8 MiB`**（总 RSS `539.4 MiB`），使 **`Qwen3.5-0.8B-Q4_0` 在 1 GB 版树莓派 4B 上也能无 Swap 稳定运行**！

### ✅ P3：Linux 内存探测（`/proc/meminfo` + `cgroup v2`）与 `madvise(MADV_SEQUENTIAL)`
- [tm-core/src/memprobe.rs](../../crates/tm-core/src/memprobe.rs) 实现了符合 `AGENTS.md` §9 的 `probe_system_memory()`：同时读取 `/proc/meminfo` (`MemAvailable`) 与 `/proc/self/cgroup` + `/sys/fs/cgroup/.../memory.{max,current}`（Docker / Home Assistant / k3s 场景），严禁在缺失或异常时静默默认为 `0`；
- [tm-cli/src/main.rs](../../crates/tm-cli/src/main.rs) 在分配 KV / GDN 状态前执行脏页水位校验，超限时提前报错提示缩减 `-c <ctx>`；
- [tm-gguf/src/lib.rs](../../crates/tm-gguf/src/lib.rs) 在 Unix `mmap` 打开后自动调用 `advise(Advice::Sequential)` 提升树莓派 SD 卡 / USB3 SSD 冷启动顺序预读吞吐。
