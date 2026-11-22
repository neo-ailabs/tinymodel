# 架构决策记录（ADR）

格式：编号 / 状态（Proposed、Accepted、Superseded）/ 背景 / 决策 / 后果。新增决策追加在文末，不修改已接受的条目；需要推翻时新增一条，并把旧条目标记为 Superseded。

---

### ADR-001 硬件分三档：micro / edge / host
- **状态**：Accepted（2026-11-19）
- **背景**：需求从“≤5B、主机到树莓派”逐步扩展为“Zero 2W 也要支持，主机要跑 8B”。三档设备在带宽、内存和指令集上相差一到两个数量级。
- **决策**：一份代码，通过 cargo feature 和 profile 编出 `micro`（≤0.5B）、`edge`（≤5B）、`host`（≤8B）三种配置。模型层共用，后端和功能开关不同。
- **后果**：内核需要额外维护 ARMv8.0 这一档；功能必须能按 feature 裁剪。

### ADR-002 实现语言：Rust（stable）
- **状态**：Accepted（2026-11-19）
- **背景**：见 `research/03-rust-feasibility.md`。本机实测：dotprod intrinsics 已经稳定；i8mm 和 FP16 还不稳定，但用 `asm!` 可以在 stable 上使用。
- **决策**：全部使用 Rust stable。还不稳定的指令集中放在 `tm-cpu::asm` 模块；GPU 内核使用各平台原生的着色器语言。
- **后果**：要自己移植 ggml、KleidiAI 的内核；换来交叉编译简单、内存安全，以及能用 feature 裁剪。

### ADR-003 自研运行时 + 兼容 GGUF + 移植内核（方案 A，保留方案 B 作为后路）
- **状态**：Accepted（2026-11-19，用户确认）
- **决策**：运行时和模型层自研；直接读 GGUF；CPU 内核以 Strata `src/quant/`（ggml quants.c 的 MIT 移植）为蓝本移植成 Rust。如果 PoC 中某个内核明显落后，允许暂时通过 `cc` crate 接入 C 内核（ggml MIT、KleidiAI Apache-2.0）。
- **后果**：不依赖 candle 这类动态张量框架；必须保留上游的版权声明。

### ADR-004 每个模型手写 forward，不做通用计算图
- **状态**：Accepted
- **背景**：≤8B 模型的结构种类有限；Strata 的实践也证明手写 forward 可行。
- **决策**：`tm-models` 里每个模型结构一个模块，配合静态内存规划。
- **后果**：新模型结构需要写代码，靠模板和 logits 比对工具降低成本。

### ADR-005 统一 LayerState，投机验证采用“只读状态 + 重放提交”
- **状态**：Accepted
- **背景**：混合模型的递归状态不能像 KV 那样截断；做快照代价大。参考 Strata `verify.hpp`。
- **决策**：见 `ARCHITECTURE.md` 5.2 节。
- **后果**：验证窗口期间需要保存每行的输入；内核需要支持 `write=false` 模式。

### ADR-006 正确性优先：标量参考实现 + 逐位对齐
- **状态**：Accepted
- **决策**：每种格式先写标量参考实现；SIMD 和 GPU 版本必须和它逐位一致（GPU 允许按文档规定的容差）；端到端测试比对多 token logits（KL 散度、top-1 一致率）。
- **后果**：新增一个内核时，必须同时提供对齐测试和 bench。

### ADR-007 GPU 后端顺序：Metal → CUDA → Vulkan
- **状态**：Superseded by ADR-008
- **背景**：主力开发机是 M4；Strata 的 Metal 内核可以直接参考。
- **决策**：M2 阶段做 Metal，M3 阶段做 CUDA，M5 阶段做 Vulkan。wgpu 和 CubeCL 只作为对比实验，不作为主路径。

### ADR-008 GPU 后端顺序调整为：Metal → Vulkan → CUDA
- **状态**：Accepted（2026-11-19，用户确认）；替代 ADR-007
- **背景**：项目的重点是端侧和跨平台。Vulkan 一套实现就能覆盖 AMD、Intel、NVIDIA 显卡，以及 Mali（手机、RK3588）和 VideoCore VII（Pi 5），对 edge 档也有价值；CUDA 只服务于 NVIDIA 显卡。
- **决策**：M2 阶段做 Metal（主力开发机是 M4），M3 阶段做 Vulkan，M5 阶段做 CUDA。CUDA 定位为 NVIDIA 显卡上的极致性能路径；在它实现之前，NVIDIA 显卡先用 Vulkan 后端。
- **后果**：Vulkan 的量化 GEMV 和 matmul 需要较早投入（可以参考 llama.cpp 的 ggml-vulkan，MIT）；需要处理不同驱动对 subgroup 和 cooperative matrix 扩展的支持差异。

### ADR-009 许可证：MIT OR Apache-2.0 双许可
- **状态**：Accepted（2026-11-19，用户确认）
- **背景**：这是 Rust 生态的惯例；它和要移植的上游许可证都兼容：ggml 和 Strata 是 MIT，KleidiAI 是 Apache-2.0。
- **决策**：项目以 `MIT OR Apache-2.0` 双许可发布，许可证文本放在仓库根目录的 `LICENSE-MIT` 和 `LICENSE-APACHE`；每个 crate 的 `Cargo.toml` 都写 `license = "MIT OR Apache-2.0"`。
- **后果**：移植的文件要保留上游的版权声明；第三方声明汇总在 `THIRD_PARTY_NOTICES.md` 中（第一次移植代码时创建）。

### ADR-010 M2 混合模型状态管理、混合前缀缓存与 Gemma 4 PLE 按需查表
- **状态**：Accepted（2026-11-21）
- **背景**：M2 引入三类异构层结构——`Qwen3.5`（GDN 线性注意力递归状态 + Conv1d 历史）、`Gemma 4`（262k 词表 × 35 层 = 1.27GB 的 `per_layer_token_embd` PLE 词表、4:1 滑动窗口 SWA/比例 RoPE 全注意力、20 层跨层共享 KV 缓存 `shared_kv_layers`、`BF16` 模型投影）以及 `LFM2`（门控因果 `ShortConv` + QK-Norm GQA）。在 4GB–8GB 手机/SBC 上，全量解压 PLE 词表或为共享层重复分配 KV 缓存会直接突破内存上限；而递归层无法像标准 KV 那样任意截断回退。
- **决策**：
  1. 在 `tm-models::LayerState` 统一建模 `Kv { window }`、`SharedKv { source_layer, window }`、`DeltaNet { state, conv_hist }`、`ShortConv { conv_hist }` 四种变体，跨层共享层零重复 KV 分配，且在 Prefill 进入 `n_layer_kv_from_start` 时自动裁剪非输出 token 行。
  2. 在 `tm-runtime` 实现带内存字节硬上限的 LRU `PrefixCache`：对纯注意力/SWA/共享 KV 模型支持**任意最长公共前缀截断复用**；对包含 `DeltaNet`/`ShortConv` 递归层的混合模型支持**精确检查点（含定期 `checkpoint_every` 边界快照）恢复**，保证多轮会话与未缓存全量 Prefill **逐位一致（bit-exact）**。
  3. `Gemma 4` 的 `per_layer_token_embd` 保持 GGUF mmap 原位不解压，CPU 与 Metal 均仅按当前输入 token ID 按需读取单行（约 5KB/token），省下 ~1.2GB 常驻内存。
- **后果**：`Gemma 4 E2B` 在 Pixel 8 真机总 RSS 仅 1.47GiB、匿名堆内存仅 55.6MiB 即可流畅运行；`tm-runtime` 作为上层 CLI/Server/FFI 的统一会话入口。

### ADR-011 M3 零膨胀离线权重重排 `Q4_0x4`、自适应投机解码与纯 Rust 编译期 SPIR-V Vulkan 后端
- **状态**：Accepted（2026-11-21）
- **背景**：M3 需要同时提升端侧 CPU 的 Prefill GEMM 与有效 Decode 吞吐，并落地跨平台 Vulkan 后端。`llama.cpp` 默认的运行时 KleidiAI 重排（`--repack 1`）虽能加速 GEMM，但会将全部权重复制进匿名堆内存导致 RSS 翻倍（在 Pixel 8 上多占 `613 MiB – 1.30 GiB`，在 512MB Zero 2W 上直接 OOM）；同时 Vulkan 若在构建期依赖外部 C++ `shaderc`/`glslc`，会破坏 `cargo zigbuild` 单命令交叉编译体验。
- **决策**：
  1. **零膨胀离线交错布局 `DType::Q4_0x4`（`GGML_TYPE_TM_Q4_0X4 = 100`）**：通过 `tinymodel convert` 离线将 4 行乘数对齐的 `Q4_0` 二维权重重排为 72 字节 `BlockQ4_0x4`（`4×f16` 缩放因子 + `2×2` 半块交错的 64B 半字节负载，文件字节数与原 GGUF 完全相同），推理时直接 0 拷贝 `mmap` 映射进 `i8mm` `smmla` / `dotprod` `sdot` / `neon_v80` / Vulkan 内核，且保证与原始 `Q4_0` **0 ULP 逐位一致**。
  2. **在线收益门控投机解码（`--spec auto`）与 `Q8_0` KV 缓存（`--kv-type q8_0`）**：在 `tm-runtime` 实现固定容量（384 KiB）4 路组相联 4-gram/3-gram/2-gram `SuffixDrafter`、`MtpDrafter`/`LinearMtpHead` 与在线 EMA `SpecController`（预期净收益 `< 5%` 自动进入冷却关闭）；底层全部五类模型结构严格遵循 `ADR-005` 只读验证 + `commit_speculative` 重放提交，确保投机开关不影响贪心输出的任何一个 token。
  3. **纯 Rust 编译期 SPIR-V 1.0 Vulkan 后端（`tm-vulkan`）**：使用 `ash` 运行时动态 `dlopen` `libvulkan`，构建期使用纯 Rust `naga` 将 GLSL 450 着色器编译为不依赖可选 8/16-bit SSBO 扩展的 SPIR-V 1.0（工作组统一 `≤ 256` 以兼容 Pi 5 V3DV），并在 `build.rs` 修正 `naga` 24 的 `Barrier::SUB_GROUP` 作用域降级问题、展平多维共享数组，UMA 设备优先选择 `HOST_CACHED` 内存类型。
- **后果**：Pixel 8 上 `Llama-3.2-1B` 零拷贝 Prefill 达 `1.56×` `llama.cpp (--repack 0)` 且省 `810 MiB` 内存，`Qwen3.5-0.8B` 与 `SmolLM2-360M` Decode 反超 `llama.cpp KleidiAI`；结构化任务有效 Decode 提速 `1.30×–1.33×`；`tm-vulkan` 在无 Vulkan SDK 环境下可直接交叉编译至 Linux musl 与 Android Bionic，并在 Mali-G715 真机通过 `Qwen3.5` GDN 多 token 端到端验证（`mean_KL = 8.88e-9`）。

### ADR-012 M4 `micro` 档：16 行 T-MAC 二补数有符号 `TBL` 激活查表、低比特码本与 BPE 闭包离线词表裁剪
- **状态**：Accepted（2026-11-21）
- **背景**：M4 面向 `micro` 档极限硬件（Raspberry Pi Zero 2W：`512 MB` LPDDR2 可用约 `350 MB`、持续带宽 `~2 GB/s`、四核顺序执行 Cortex-A53 仅有纯 ARMv8.0 NEON，**无 `dotprod`、无 `i8mm`、无硬件 FP16 运算**）。在该档位上：① 传统逐元素解包乘加（`SMULL`/`SMLAL`）对 2-bit / BitNet b1.58 三值权重的指令开销高；② 小模型（如 `SmolLM2-360M` 49k 词表、`Qwen3.5-0.8B` 248k 词表）的嵌入与 LM Head 占总权重体积的 `20%–50%`；③ 纯静态二进制体积必须 `≤ 2 MB`（`N-05`），且全内核仍须满足 `ADR-006` 与标量参考 `0-ULP` 逐位一致。
- **决策**：
  1. **16 行 T-MAC 二补数有符号 `vqtbl1q` 激活查表内核（`tm-cpu::aarch64::lut`）**：针对 `Q2_0`（2.25 bpw）与 BitNet b1.58 三值 `TQ2_0`（2.06 bpw），每个 K-tile 在栈上（`8–16 KiB`，零堆分配，完全常驻 Cortex-A53 的 `32 KiB` L1D）将量化激活对 `(y0, y1)` 预计算为 16 项二补数拆分的 `(lo: u8, hi: i8)` 表（满足恒等式 `(hi as i16) * 256 + (lo as i16) == val`，无需零点偏移补偿）；内层对 16 行权重做 `vld1q_u8` 连续读取 + 寄存器内 `16×16` 字节转置后，仅靠 `vqtbl1q_u8`/`vqtbl1q_s8` 与 `vaddl_u8`/`vaddl_s8` 宽加完成 16 行累加（**内层权重循环 0 条乘法指令**，且与 `generic` **0-ULP 逐位一致**）。
  2. **低比特码本内核（`IQ2_XXS` / `IQ3_XXS` / `IQ4_NL`）**：在 `tm-core::iq_tables` 移植 `IQ2XXS_GRID`、`IQ3XXS_GRID`、`KSIGNS64` 奇偶校验符号表与 `KVALUES_IQ4NL` 非线性码本；ARMv8.0 端用无分支二补数异或 `(g ^ sm) - sm` 与单指令 `vqtbl1q_s8` 16 项非线性查表实现，同时提供 `dotprod` `sdot` 加速路径；`tinymodel convert` 在张量列数非 256 整数倍时（如 `SmolLM2-360M` `n_embd=960`）自动按张量回退至 `Q2_0`（64 块）或 `IQ4_NL`（32 块）。
  3. **BPE 闭包离线词表裁剪（`tinymodel convert --prune-vocab <N> [--vocab-corpus <file>]`）**：强制保留特殊控制 token 与全部 256 个 GPT-2 字节级回退单字符 / `<0xXX>` token（占用 `< 350` 槽位，数学保证任意语言 UTF-8 输入 **0% UNK 无损编码**），对语料高频词与剩余预算按 BPE merge 排名递归补齐祖先闭包，同步裁剪并重映射 `tokenizer.ggml.{tokens,merges,scores,token_type}` 与 `token_embd`/`output`/`per_layer_token_embd` 行，保持 32 字节对齐零拷贝 `mmap`。
- **后果**：`micro` 档 `aarch64-unknown-linux-musl` 纯静态 stripped ELF 仅 `1.2 MB` 且经机器码全量扫描确认含 `0` 条 `sdot`/`udot`/`smmla`/`ummla` 指令；`TQ2_0` 在纯 ARMv8.0 NEON 下较标量提速 `11.3×`；`SmolLM2-360M` 在 `micro` 配置下峰值 RSS 控制在 `190–248 MiB`（远低于 Zero 2W 的 `350 MB` 红线）。

### ADR-013 AMD GPU 支持走 Vulkan 1.2/1.3 预编译 SPIR-V + Cooperative Matrix 架构（升级 ADR-011 第 3 条）
- **状态**：Accepted（2026-11-21，用户确认）
- **背景**：x86 开发主机配备 AMD Radeon RX 9060 XT（RDNA4/`gfx1200`，8 GiB）与 Intel UHD 770（ANV），可选 Vulkan 或 HIP/ROCm 7.1.1（分析见 `research/11-amdgpu.md`）。`ADR-011` 第 3 条的 `naga` SPIR-V 1.0 过渡方案无法生成 `VK_KHR_cooperative_matrix`、`subgroupAdd`、`subgroupShuffleXor` 与 16/8-bit 算术指令，且引入 239 个构建期依赖 crate。
- **决策**：AMD 显卡通过 `tm-vulkan`（`ash` + GLSL 450 → SPIR-V 1.2）一等公民支持；仓库直接提交经 `spirv-val` 校验的 28 个预编译 `spv/*.spv`（本机有 `glslc` 时自动重编，无 Vulkan SDK 时零外部依赖直接构建，同时彻底移除 `naga` 构建依赖）；实现预录制 Decode 命令缓冲、`16×16×16` FP16/INT8 Cooperative Matrix Prefill、融合 SwiGLU、`fa_coopmat`、LDS 预取 GDN 递推、Subgroup `8/16/32/64` 全适配（AMD 默认 `wave32`、Intel 默认 `SIMD16`）、Pi 5 `V3DV` 无 Shuffle `gdn_recurrence_lds` 退回，以及 `Q4_0x4` / 低比特权重的加载期无损解交错兼容。
- **后果**：保持普通 `cargo build` 与 `aarch64` 交叉编译零 SDK 依赖，同时解锁硬件矩阵与 Subgroup 原语；CI 通过 `lavapipe` 跑全套对齐测试。

### ADR-014 V6 决策门结论：维持 `tm-vulkan` 为 AMD 与跨平台主力后端，HIP 留至 M5 与 `tm-cuda` 共源评估
- **状态**：Accepted（2026-11-21）
- **背景**：完成 V0–V5（见 `research/11-amdgpu.md`）后，在 AMD Radeon RX 9060 XT（Mesa 25.2.8 vs ROCm 7.1.1）和 Intel UHD 770（ANV）上的 5 轮实测显示：
  1. **Decode（`tg128`）全面超越 HIP**：全部 5 个基准模型上 `tinymodel --vulkan` 同时击败 `llama.cpp` Vulkan（`104%–183%`）和 **ROCm 7.1.1 HIP（`108%–226%`）**（`SmolLM2-360M` `608.8 tok/s = 225.9%` HIP、`Qwen3.5-0.8B` `366.5 tok/s = 217.8%` HIP、`Qwen3.5-2B` `184.5 tok/s = 137.4%` HIP、`Qwen3-8B` `64.0 tok/s = 108.0%` HIP）。
  2. **Prefill（`pp512`）追平乃至反超 HIP**：达 `llama.cpp` Vulkan 的 **`96.5%–474.9%`**、ROCm HIP 的 **`66.9%–116.3%`**（GPU 硬件时间戳热态达 HIP 的 **`82.7%–116.3%`**，其中 `Qwen3.5-0.8B` `13566.6 tok/s` 直接反超 HIP `16.3%`）。
  3. **单后端跨厂商复用**：同一套 `tm-vulkan` 在 **Intel UHD 770** 核显上 Decode 超越 `llama.cpp` Vulkan **`18.2%–50.0%`**，且通过 `aarch64-linux-android` 与 `aarch64-unknown-linux-gnu` 交叉编译检查。
- **决策**：**不单独提前开辟 HIP 后端**，`tm-vulkan` 作为 AMD、Intel 及 Arm 端侧 GPU 的正式主力后端；待 M5 建设 `tm-cuda` 时，再按需评估 `.cu` 内核对 `hipcc` 的共源复用。
- **后果**：省去维护第三套 GPU 后端及 `~2 GiB` ROCm 运行时依赖，直接衔接 M5 服务层与 `tm-cuda`。

### ADR-015 `host` 档支持 `Inkling-2-Mini` (`inkling2_loop`) 全量 `Q8_0` 三层循环感知存储与 `DSpark` (`dflash`) 分块投机
- **状态**：Accepted（2026-11-22，用户确认在新分支 `feat/inkling2-q8` 实施）
- **背景**：用户希望利用 `tinymodel` 在本机（i7-12700 `15.4 GiB` DDR4 + RX 9060 XT `8 GiB` GDDR6 + Predator GM7000 PCIe 4.0 NVMe）及 32GB+ 主机上全量运行 26.07B 物理参数的循环稀疏 MoE 模型 `Inkling-2-Mini-Q8_0.gguf`（`25.91 GiB`）及原生草稿器 `dspark-Inkling-2-Mini-Q8_0.gguf`（`383.6 MiB`）。全量 `Q8_0` 仅含 `Q8_0`/`BF16`/`F32` 三种已有 `DType`，且单 token 实际读取仅 `5.15 GiB`（其中 85.3% 流量集中在 `14.13 GiB` 的稠密层与 `3×` 循环核专家上），但 `26.3 GiB` 总权重超出本机 `21 GiB` 易失存储上限，常规 `mmap` 会触发全层 LRU 随机缺页风暴并造成页缓存双份驻留。
- **决策**：
  1. **三层循环感知存储规划（Tier 0 VRAM + Tier 1 匿名 RAM + Tier 2 `O_DIRECT` NVMe 双缓冲槽）**：加载期按“每 token 逻辑访问频次 ÷ 字节体积”静态划分权重驻留层级——优先将全部 24 层稠密注意力/卷积/Loop 控制器、LM Head、`DSpark` 及高频 `blk.6..17`（`3×` 循环核）专家锁入 VRAM 与主机匿名 RAM（通过不触碰 `mmap` 页的无缓存 `DirectTensorReader` 加载，彻底杜绝 Linux 页缓存双份占用）；当 `VRAM + 可用 RAM < 总权重` 时，仅将单遍外层专家留在 NVMe，运行期通过预分配的 4096 字节对齐槽位并发 `O_DIRECT` 直读 Top-8 激活专家（生成路径零堆分配、零 `unsafe` 越界）。
  2. **严格遵守 48 逻辑层独立状态与填充词表截断契约**：24 层物理权重展开为 48 个独立逻辑层状态槽（40 层 `SWA=1024` 环形 KV + 8 层全局 KV + 48 份 4 路 `ShortConv` 历史）；采样前严格将 `201,024` 行填充 logits 截断至 `vocab_size_unpadded = 200,058`。
  3. **补齐 `tm-cpu::x86_64::avx2` 0-ULP 内核并隔离 `micro` 边界**：在 `tm-cpu` 实装 `x86_64` `AVX2+FMA` 向量化内核（与 `generic` 逐位一致），同时保持 `micro` 档 `aarch64-musl` 静态构建零体积膨胀。
- **后果**：在不破坏 `micro`/`edge` 零拷贝精简架构的前提下，使 `tinymodel` 能够在 16GB RAM + 8GB VRAM 主机上无 OOM、无 Swap 抖动地全量运行 `25.91 GiB` `Inkling-2-Mini-Q8_0` + `DSpark`，并在 32GB+ 主机上自动退化为纯内存+显存零读盘运行。
