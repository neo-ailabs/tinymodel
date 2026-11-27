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

### ADR-015 支持 GGUF LoRA 推理与量化基座上的手写逐层微调（QLoRA + MTP 草稿头）
- **状态**：Accepted（2026-11-22，用户确认）
- **背景**：端侧与单机场景存在强烈的本地指令对齐、工具调用微调与 MTP 投机草稿头自举需求。对比上游 `llama.cpp` 当前重新引入的 `llama-finetune`（仅支持 FP32 全参数微调，1B 模型约需 `24 GB` 内存且关闭 FlashAttention），`tinymodel` 需要一条能在几百 MiB 脏内存内直接基于冻结量化权重（`Q4_0 / Q4_0x4 / Q5_K / Q6_K / 低比特`）训练 LoRA 的路径，同时不能污染 `micro` 档 `≤ 2 MB` 的纯静态二进制（`N-05`），也不能破坏推理热路径的零堆分配契约（`MUST #4`）。此外，推理前向采用了 `Q8_0` 动态激活量化，不能直接套用纯 FP32 框架的 `1e-4` 梯度阈值。
- **决策**：
  1. **范围界定**：调整 `REQUIREMENTS.md` 第 5 节——正式支持 **GGUF LoRA 动态挂载/离线合并、MTP 投机草稿头训练，以及冻结量化基座上的 LoRA 微调（QLoRA）**；**不做全参数预训练/全参数微调**。
  2. **模块与 Feature 隔离**：新增两个 `#![forbid(unsafe_code)]` crate——`crates/tm-lora`（兼容 `llama.cpp` `general.type = "adapter"` / `adapter.type = "lora"` 规范的零拷贝/轻量适配器加载、多适配器缩放叠加与 `lora-merge` 离线融合）和 `crates/tm-train`（通过 `tm-cli` 的 `train` feature 开关挂载，默认不编入 `micro` 档）；所有 `unsafe` 反向算子（如 `matmul_q_transposed_f32`）严格留在 `tm-cpu / tm-metal / tm-vulkan` 内并配齐三件套（`MUST #2, #3`）。
  3. **独立训练前向 + 零 LoRA 逐位一致门禁**：不把激活保存逻辑塞进零分配推理路径，而是在 `tm-train` 编写调用同一套底层算子的独立训练前向（含逐层 `FP16` 入口检查点、单层静态 `TrainArena` 与跳过 `label = -100` 的分块交叉熵），并用硬性测试强制约束 **“LoRA 权重全零时，训练前向输出的 logits 与推理 `forward_batch` 逐位一致（0 ULP）”**，防止训练与部署数值漂移。
  4. **Q8 激活直通估计（STE）与三层梯度验收**：前向保持与部署一致的量化点积，反向采用 STE + FP32 累加（固定归约顺序以保证确定性 `N-10`）；按 ① 算子级 `f64` 中心差分（`< 1e-5`）、② 含同款 `Q8` fake-quant 的模拟量化 Oracle（`< 1e-3`）、③ 纯 FP32 PEFT 梯度方向余弦（`> 0.99`）与多步 loss 收敛三层验收。
  5. **渐进实施顺序**：按 **`T1`（LoRA 加载与合并）→ `T2a`（MTP 投机头训练与 CLI `--mtp` 接入）→ `T2b`（标准 Transformer CPU QLoRA）→ `T3`（Qwen3.5 GDN / Gemma 4 / LFM2 混合反向）→ `T4`（Metal / Vulkan GPU 反向加速）** 推进。
- **后果**：`micro` 推理产物保持 `< 1.2 MB` 零体积膨胀；M3 已实现的 `--spec mtp` 获得真实模型可用的训练闭环；端侧与主机可在数百 MiB 额外内存下直接微调量化 GGUF 模型并导出与 `llama.cpp --lora` 互通的适配器。

### ADR-016 `host` 档支持 `Inkling-2-Mini` (`inkling2_loop`) 全量 `Q8_0` 三层循环感知存储与 `DSpark` (`dflash`) 分块投机
- **状态**：Accepted（2026-11-22，用户确认在新分支 `feat/inkling2-q8` 实施）
- **背景**：用户希望利用 `tinymodel` 在本机（i7-12700 `15.4 GiB` DDR4 + RX 9060 XT `8 GiB` GDDR6 + Predator GM7000 PCIe 4.0 NVMe）及 32GB+ 主机上全量运行 26.07B 物理参数的循环稀疏 MoE 模型 `Inkling-2-Mini-Q8_0.gguf`（`25.91 GiB`）及原生草稿器 `dspark-Inkling-2-Mini-Q8_0.gguf`（`383.6 MiB`）。全量 `Q8_0` 仅含 `Q8_0`/`BF16`/`F32` 三种已有 `DType`，且单 token 实际读取仅 `5.15 GiB`（其中 85.3% 流量集中在 `14.13 GiB` 的稠密层与 `3×` 循环核专家上），但 `26.3 GiB` 总权重超出本机 `21 GiB` 易失存储上限，常规 `mmap` 会触发全层 LRU 随机缺页风暴并造成页缓存双份驻留。
- **决策**：
  1. **三层循环感知存储规划（Tier 0 VRAM + Tier 1 匿名 RAM + Tier 2 `O_DIRECT` NVMe 双缓冲槽）**：加载期按“每 token 逻辑访问频次 ÷ 字节体积”静态划分权重驻留层级——优先将全部 24 层稠密注意力/卷积/Loop 控制器、LM Head、`DSpark` 及高频 `blk.6..17`（`3×` 循环核）专家锁入 VRAM 与主机匿名 RAM（通过不触碰 `mmap` 页的无缓存 `DirectTensorReader` 加载，彻底杜绝 Linux 页缓存双份占用）；当 `VRAM + 可用 RAM < 总权重` 时，仅将单遍外层专家留在 NVMe，运行期通过预分配的 4096 字节对齐槽位并发 `O_DIRECT` 直读 Top-8 激活专家（生成路径零堆分配、零 `unsafe` 越界）。
  2. **严格遵守 48 逻辑层独立状态与填充词表截断契约**：24 层物理权重展开为 48 个独立逻辑层状态槽（40 层 `SWA=1024` 环形 KV + 8 层全局 KV + 48 份 4 路 `ShortConv` 历史）；采样前严格将 `201,024` 行填充 logits 截断至 `vocab_size_unpadded = 200,058`。
  3. **补齐 `tm-cpu::x86_64::avx2` 0-ULP 内核并隔离 `micro` 边界**：在 `tm-cpu` 实装 `x86_64` `AVX2+FMA` 向量化内核（与 `generic` 逐位一致），同时保持 `micro` 档 `aarch64-musl` 静态构建零体积膨胀。
- **后果**：在不破坏 `micro`/`edge` 零拷贝精简架构的前提下，使 `tinymodel` 能够在 16GB RAM + 8GB VRAM 主机上无 OOM、无 Swap 抖动地全量运行 `25.91 GiB` `Inkling-2-Mini-Q8_0` + `DSpark`，并在 32GB+ 主机上自动退化为纯内存+显存零读盘运行。

### ADR-017 P1 架构扩展：`olmo2`/`granite` 稠密路径增强与统一 Mamba-2 SSM 混合架构（`mamba2` / `nemotron_h` / `falcon-h1` / `granitehybrid`）
- **状态**：Accepted（2026-11-24，用户确认）
- **背景**：完成 Hugging Face ≤8B 模型全量普查与 P0 架构合入（见 `research/15-hf-sub8b-architectures.md`）后，剩余高价值 ≤8B 生态集中在两类 P1 需求：① AI2 全开源基线 `OLMo 2 / OLMo 3`（纯 Post-Norm、无 Pre-Norm、全向量 `[q_dim]`/`[kv_dim]` QK-Norm 及 3:1 SWA）与 IBM 企业级 `Granite 3.x / 4.0`（四重 muP 缩放系数：`embedding_scale`、`attention.scale`、`residual_scale`、`1 / logit_scale`）；② 新一代 **Mamba-2 选择性状态空间模型（SSM）** 及其三大端侧混合变体——NVIDIA `Nemotron-H`（按层分离的 Mamba-2 / 无 RoPE GQA / ReLU² FFN）、TII `Falcon-H1`（同层并行的 GQA + Mamba-2 + 串行 SwiGLU）和 IBM `Granite-4.0-Hybrid`（按层交替的 Mamba-2 / Norm-RoPE GQA + muP 缩放 SwiGLU）。上游 `llama.cpp` 对这四类 SSM 模型共享完全一致的 `build_mamba2_layer` 张量契约（`ssm_in` / `ssm_conv1d` / `ssm_dt.bias` / `ssm_a` / `ssm_d` / 分组 `ssm_norm` / `ssm_out`）。
- **决策**：
  1. **稠密路径零分叉复用**：在 `LlamaModel` / `LlamaMetalRunner` / `LlamaTrainer` 中支持可选 Pre-Norm（`olmo2` 为 `None`）、Post-Norm（`post_attention_norm` / `post_ffw_norm`）、按张量维度自动判别逐头 `[head_dim]` 或全向量 `[q_dim]`/`[kv_dim]` QK-Norm、逐层 `is_swa` 掩码，以及 Granite 的四重 muP 缩放系数与可选 MLP 偏置，统一承载 `olmo2`（含 OLMo 3）与 `granite`。
  2. **落地 `LayerState::Ssm { state, conv_hist }` 与 0-ULP `mamba2_scan_head` 内核三件套**：按 `ARCHITECTURE.md` §5.2 补齐 `LayerState::Ssm`（支持 `PrefixCache` 检查点恢复与 `ADR-005` 投机验证只读扫描 + 重放提交）；在 `tm-cpu` 实现 `generic`、`aarch64::neon_v80` 与 `x86_64::avx2` 三档 `mamba2_scan_head`（统一 8 路成对树形 FMA 归约，达成全 CPU ISA **0-ULP** 逐位一致），并在 `tm-metal` 实现 `mamba2_conv_scan_f32` GPU 内核。
  3. **单一 `Mamba2Model` 统一驱动四类 SSM/混合架构（`mamba2`、`nemotron_h`、`falcon-h1`、`granitehybrid`）**：将每层建模为可组合的算子序列（纯 `Mamba2Mixer`、按层异构 `Ssm | Attn | Relu2Ffn`、或同层并行 `Attn + Ssm -> SwiGLU`），严格遵守分组 RMSNorm（`n_group` 组独立归一化）与 `ssm_d` 按头广播契约，并配套 `Mamba2Trainer` 支持 `O(1)` 块边界状态检查点的精确 QLoRA 反向传播。
- **后果**：新增一套 Mamba-2 核心算子即可一举打通 `mamba2`、`nemotron_h`、`falcon-h1`、`granitehybrid` 四大架构（连同 `olmo2`/`granite` 共 6 类架构标识），使 `tinymodel` 对 Hugging Face 现代 ≤8B 文本与混合架构实现近乎全量覆盖，且保持推理热路径零堆分配与 `micro` 档静态体积预算。

## ADR-018：通用细粒度稀疏 MoE 子模块（`SparseMoeFfn`）与 P2a 架构群支持（2026-11-24）

- **状态**：已接受
- **背景**：按 `docs/research/15-hf-sub8b-architectures.md` P2a 路线，端侧 ≤8B 稀疏混合专家模型（`olmoe`、`granitemoe`、`granitehybrid`/`granitemoehybrid`、`qwen2moe`、`qwen3moe`、`bailingmoe2`）可将每 token 访存压至 0.2B–1.0B，显著缓解 Pixel 8、RK3588 与 Pi 5 的内存带宽瓶颈；此前 MoE 路由与专家求值仅内嵌在 `inkling2.rs` 中，尚未复用于标准 Attention 与 Mamba-2 混合主干。
- **决策**：
  1. 在 `crates/tm-models/src/moe.rs`（`#![forbid(unsafe_code)]`）新增可复用的 `SparseMoeFfn` 与预分配 `MoeScratch`，支持三维打包专家张量（分离 `ffn_{gate,up,down}_exps.weight` 与融合 `ffn_gate_up_exps.weight` 零拷贝切片，各带自身 `DType`）、`Softmax` / `Sigmoid` / `SoftmaxWeight` / `SqrtSoftplus` 门控、可选 `exp_probs_b.bias` 路由偏置、`expert_weights_norm`（`6.103515625e-5` F16 下限保护）与 `expert_weights_scale`，以及可选共享专家（`ffn_{gate,up,down}_shexp.weight` 及 `qwen2moe` 的 `sigmoid(ffn_gate_inp_shexp)` 门控）；单步与批处理严格按确定性 `0..n_expert_used` 顺序归约以保持 `0-ULP` 批不变性。
  2. 将 `SparseMoeFfn` 同时挂载到 `LlamaModel`（解锁 `olmoe`、`granitemoe`、`qwen2moe`、`qwen3moe`、`bailingmoe2`，并顺带接入同构稠密架构 `exaone`）与 `Mamba2Model`（解锁 `expert_count > 0` 的 `granitehybrid` / `granitemoehybrid`）。
  3. 在 `tm-metal` 新增单 `MTLCommandBuffer` 零 CPU 往返的 GPU 路由与间接专家偏移内核（`moe_router_topk_f32` + 索引化专家投影），并在 `tm-train`（`LlamaTrainer`、`Mamba2Trainer`）补齐通过激活专家、共享专家与门控路由器的 **0-ULP** 前向门禁及精确解析反向传播。
- **后果**：无需新增底层 CPU SIMD 指令即可在稠密 Attention 与 Mamba-2 混合两条主干上同时解锁全部主流端侧细粒度 MoE 架构，保持 token 生成零堆分配与全栈 `CPU + Metal + QLoRA` 一致性。

## ADR-019：通用多模态视觉编码器（`VisionModel`）与阶段加载 VLM 运行时（2026-11-25）

- **状态**：已接受
- **背景**：按 `docs/research/15-hf-sub8b-architectures.md` §4.2 P2 与 `ROADMAP.md` M6，2025–2026 年主流端侧 ≤8B 模型（`Qwen3.5`、`Qwen3-VL`、`Qwen2.5-VL`、`Gemma 3/4`、`Ministral 3`、`SmolVLM`、`LFM2-VL`）普遍以原生 VLM 形态发布，在 `llama.cpp` 生态中统一通过独立 `mmproj` GGUF 承载视觉编码器与多模态投影器。端侧设备（Apple M4 / Pixel 8 / RK3588 / Pi 5）内存受限，不能让视觉编码器权重在纯文本解码阶段常驻内存，也不能引入重型第三方图像依赖膨胀 `micro` 档体积。
- **决策**：
  1. 在 `crates/tm-models/src/vision.rs`（`#![forbid(unsafe_code)]`，零新增外部 crate）实现兼容 `llama.cpp` / `mtmd` `mmproj` GGUF 规范的零拷贝 `VisionModel`，统一覆盖四大视觉编码器与投影族：**SigLIP / Gemma 3 & 4**（2D Patch Conv + 可学习位置编码 + Pre-LN + 空间平均池化 + `mm.soft_emb_norm` + 线性投影）、**Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5**（3D 时序-2 卷积 `v.patch_embd.weight` + `.weight.1` + 2D Vision RoPE + 融合 QKV + SwiGLU + 2×2 空间合并 MLP + 3D M-RoPE 坐标）、**Pixtral / Mistral-3**（2D Vision RoPE + Pre-RMSNorm + SwiGLU + `mm.patch_merger`）及 **SmolVLM / LFM2-VL**（双线性位置编码插值 + Pixel-Shuffle MLP），并内置纯 Rust PPM/PGM/BMP 解码、Keys 双三次缩放与 Qwen-VL `smart_resize_dims`。
  2. 在全部五大主干（`LlamaModel`、`Qwen35Model`、`Gemma4Model`、`Lfm2Model`、`Mamba2Model`）及对应的 Apple Metal GPU Runner（`MetalRunner::forward_embed`）补齐零堆分配 `forward_embed` 视觉嵌入注入接口，在 `tm-runtime` 提供 `Session::prefill_multimodal` 与**阶段加载（Staged Load）**`encode_image_staged`（编码完成后立即释放 `mmproj` mmap，稳态解码零额外内存占用），并在 `tm-cli` `run` 接入 `--mmproj <file> --image <file>`。
- **后果**：零外部依赖一次打通全栈原生多模态推理（CPU + Apple Metal），全部 4 大视觉族通过独立 FP64 Python Oracle（`tools/oracle/p2_vision_oracle.py`）对齐与 CPU/Metal `forward_embed` 0-ULP/精确一致验收，且不影响 `micro` 档纯文本静态二进制体积。

## ADR-020：NVIDIA CUDA 后端（`tm-cuda` 动态 Driver API + CUDA Graph）与原生 `NVFP4` (`ggml_type = 40`) 支持（2026-11-27）

- **状态**：已接受
- **背景**：按 `ROADMAP.md` M5 GPU 后端顺序（Metal → Vulkan → CUDA）及 Unsloth Dynamic 混合精度 GGUF（如 `Inkling-2-Mini-UD-Q8_K_XL.gguf` 采用 `GGML_TYPE_NVFP4 = 40` + 逐专家 `F32` `.scale`），需要为 `tinymodel` 落地零编译期 CUDA SDK 硬依赖的 `tm-cuda` 后端，并在全栈补齐 `DType::NvFp4` 内核三件套及 macOS 16GB 统一内存零 Swap 调度。
- **决策**：
  1. **动态 `libcuda` Driver API + 双路径 PTX 构建（`crates/tm-cuda`）**：通过 `libloading` 在运行时动态加载 `libcuda.so.1`（macOS / 无 GPU 宿主机可无痛编译检查）；`build.rs` 自动检测 `nvcc` 编译 `kernels/cuda_ops.cu`（无 `nvcc` 时回退到随仓预编译 `ptx/cuda_ops.ptx`）。每次 `cuLaunchKernel` 前显式校验线程块与动态共享内存上限。
  2. **双模态 `__dp4a` MMVQ + `wmma` Tensor Core GEMM + 单次 `cuGraphLaunch` 解码**：移植并扩展 `Strata`（MIT）风格 4-warp 协同 `__dp4a` MMVQ（覆盖 `Q4_0`/`Q4_1`/`Q8_0`/`Q5_K`/`Q6_K`/`F16`/`BF16`/`F32`/`NVFP4`）与 `16×16×16` FP16 Tensor Core 预填充 GEMM；`LlamaCudaRunner`（覆盖稠密与 `SparseMoe` 架构群）与 `Qwen35CudaRunner`（严格遵循 `GDN_CONTRACT` 五大不变量）共享绑定词表 `Arc<CudaBuffer>`，并将整步 Decode 捕获为可参数化更新 `(token_id, pos)` 的 `CUDA Graph`，消除百余次内核发射开销。
  3. **原生 `DType::NvFp4` (`ggml_type = 40`, `4.50 bpw`) 与 macOS 统一内存分层增强**：在 `tm-core` / `tm-cpu` 实现 `BlockNvFp4`（4×`UE4M3` 子块缩放 + 64×`E2M1` 码本）及 `generic` / `neon` / `dotprod` 4 行 `vqtbl1q_s8 + vdotq_s32` 内核（全四档 ISA **0-ULP** 逐位一致）；在 `Inkling2Model` 支持逐专家 `.scale`、macOS `F_NOCACHE` 无缓存 8 路并发 SSD 流式直读、稠密 `BF16` 权重 `mlock` 防压缩与 32-token 锁步批处理 Prefill。
- **后果**：`tinymodel --cuda` 在 NVIDIA Tesla T4 (`sm_75`) 上通过 L1/L3 对齐校验并在核心 MMVQ/GDN 微基准上达到 `Strata` 的 `100%–112%`；Apple M4 (16 GB) 可无 Swap 流畅运行 21.30 GiB `Inkling-2-Mini-UD-Q8_K_XL`（RAM MoE 有效带宽 `63.2 GB/s`，`lm_head` `66.2 GB/s`）。

## ADR-021：Vulkan P0 性能优化三件套——预录制解码命令缓冲 + CPU/GPU 异步分拆 MoE、全 GPU 单 CB `DSpark` (`DFlashVkRunner`) 与 `OpSDotKHR` (`dp4a`) 融合解码内核（2026-11-26）

- **状态**：已接受
- **背景**：在远程 NVIDIA L4（`sm_89`，`23 GiB` GDDR6）与本机多显卡对比分析 `tinymodel` Vulkan 后端与 `Strata v0.1.69` CUDA/Metal 后端的性能差距时，定位到三大 P0 瓶颈：① `Inkling2VkRunner` 单步解码每 token 动态录制 24 段 Vulkan Command Buffer（`3.37 ms/tok` CPU 开销），且显存放不下的外层 MoE 专家（`14.03 ms/tok`）纯靠主机 DDR 串行求值，闲置了 PCIe 零拷贝带宽与 GPU 算力；② `DSpark`（`dflash`）投机草稿器此前在主机 CPU 运行并通过 PCIe 逐步回传/投影，导致 `k=3` 投机解码（`20.26 tok/s`）未能显著超越纯解码（`20.49 tok/s`）；③ 在启用 `CoopMatF16` 预填充的独立显卡（NVIDIA L4、AMD RDNA4）上，权重保持 AoS `BlockQ4_0`/`BlockQ8_0` 布局以兼容 `gemm_coopmat_f16`，导致单步解码退回逐半字 `uint16` 读取与 FP32 解包乘加（`Qwen3.5-9B-Q4_0` 仅发挥约 `65%` 显存带宽）。
- **决策**：
  1. **零录制开销 `BakedCb` 解码图 + CPU DDR 与 GPU 零拷贝 PCIe 异步分拆 MoE（P0-1）**：在 `VulkanContext` 扩容持久描述符集缓存（`32,768` 组）并引入可重放的 `BakedCb`，将动态 `pos`/`tau` 下沉至持久映射 `dyn_pos_buf`，使稳态解码 CPU 录制开销降至 `0.00 ms/tok`；对主机内存层 MoE，将 Top-8 激活专家按 `TM_VK_HYBRID_GPU_EXPS`（独立显卡默认分担 `2` 个专家）拆分为主机多线程 AVX2/NEON 与 GPU `VK_EXT_external_memory_host` 零拷贝子命令缓冲并发执行，并在统一归约阶段按原始 `0..top_k` 顺序精确累加。
  2. **100% GPU 单 Command Buffer `DFlashVkRunner` + `0-ULP` 去重分组验证（P0-2）**：新增 `crates/tm-vulkan/src/dflash_runner.rs`（`dflash_ops.comp`、`dflash_attn.comp`），直接借用目标模型驻留显存的 `tok_embd` 与 `output`（省去 `938 MiB` 重复显存），将“提交窗口特征注入（`inject_batch`）+ `bs=8` 非因果分块注意力三层 Transformer + `k` 步链式循环 Markov 头 + 软上限贪心 Argmax”合并为**单次 Vulkan 提交**（零中间 CPU-GPU 同步，仅回读 `4 × k` 字节草稿 token ID），支持可选 `TM_VK_DRAFT_VOCAB` 高频+特殊词表裁剪加速；目标模型验证窗口（`w <= 8`）采用单次读权重的寄存器多步 GEMV（`M = 4/8`）与专家去重寄存器 MoE（`op = 5`），严格保持与单步解码 `0-ULP` 逐位一致（`AGENTS.md` §9）。
  3. **全布局 `OpSDotKHR` (`dp4a`) 整数点积 GEMV + 单工作组融合 `RMSNorm + Q8_1`（P0-3）**：基于 SPIR-V `SPV_KHR_integer_dot_product`（`OpSDotKHR` `4×i8` 硬件加速指令，无需 `#![feature]`）补齐 AoS 与 SoA 全套解码内核（`gemv_dp4a`、`gemv_pair_dp4a`、`gemv_qkv_q4_0_dp4a`、`gemv_pair_silu_q4_0_dp4a` 及其 SoA 变体与 `inkling2_moe_dp4a`），解决 AoS 双块 `36B = 9 × uint32` 对齐加载与 `Q8_0` 8 通道合并读取问题；配套单工作组零全局屏障（`sg == 32` 快路径）融合 `rms_norm_q8_1.comp`，一次全局访存同时输出 FP32 归一化向量（供残差/路由使用）与 `Q8_1` 量化块，并统一接入 `Qwen35VkRunner`、`LlamaVkRunner`、`Inkling2VkRunner` 与 `DFlashVkRunner`（支持 `TM_VK_DP4A=0` 降档对照）。
- **后果**：在不引入任何外部依赖或破坏 `Fallback`/`llvmpipe` 兼容性的前提下，全面消除 Vulkan 解码录制开销、CPU-GPU 草稿往返同步与 FP32 解包算术瓶颈，并通过 NVIDIA、Intel ANV 与 `llvmpipe` 多驱动对齐门禁与 `0-ULP` 投机窗口确定性验证。

## ADR-022：Vulkan P1/P2 深度优化六件套——自适应置信度投机控制、词表范数排序与验证图预烘焙、隔离式 `Q8_1` 融合算子、短窗口 PCIe/DDR 分工、`Q6_K dp4a` + GDN 融合解码及双缓冲流水上传（2026-11-27）

- **状态**：已接受
- **背景**：完成 Vulkan P0 优化后，在 NVIDIA L4（`23 GiB` GDDR6 + PCIe Gen4）上的逐阶段硬件时间戳剖析揭示了剩余六项次生瓶颈：① `DSpark` 在开放域散文等低接受率（`~36%`）文本上固定生成 `k=3` 草稿，导致验证窗口成本高于收益（`19.38 tok/s` 低于纯解码 `24.27 tok/s`）；② 朴素 `0..N` 词表截断会漏掉 `o200k_base` 高 ID 编程符号，且 `w=2..8` 多 token 投机验证窗口仍动态录制 Command Buffer；③ 将 `Q8_1` 量化算子塞入主干大着色器（`inkling2_ops.comp` / `inkling2_moe.comp`）会抬高共享内存占用并拉低 SM 驻留度，而未融合 `quantize_q8_1` 会引入额外全局屏障；④ 短 prompt（`w <= 64`）混合预填充仅走单侧 PCIe 或单侧 CPU 未能并行吃满双通道带宽；⑤ `Q6_K`（如 `Qwen3.5` 24.8 万词表 `output.weight`）尚未接入 `OpSDotKHR`（`dp4a`），且 GDN/FullAttn 解码仍存在可融合的门控与激活量化步骤；⑥ 单缓冲 `64 MiB` staging 上传使磁盘/页缓存拷贝与 PCIe DMA 严格串行。
- **决策**：
  1. **Markov 在线置信度门控与收益守护控制器（P1-1）**：在 `dflash_ops.comp`（`op=9`）零额外共享内存复用 `wg_reduce_sum` 在线计算精确软上限 Softmax Top-1 概率 `p_top1`，并在 `tm-cli` 引入零堆分配 `DSparkAdaptiveCtrl`——既在提交 26B 目标验证前即时截断低置信度尾部草稿，又按 EMA 接受率与实测单步吞吐守护动态调节 `active_k ∈ 0..=max_k`（`k=0` 冷却期通过动态槽位累积特征并在恢复探测时批量注入）。
  2. **`||markov_w1[id]||_2` 词表重要性排序 + 全窗口 `BakedSpecPlan` 预烘焙（P1-2）**：`DFlashVkRunner` 在 `TM_VK_DRAFT_VOCAB` 裁剪时保留 ASCII 头尾特殊 token 并按 `markov_w1` 行 L2 范数降序选出高活跃子集（升序排列保确定性平局规则）；`Inkling2VkRunner` 为 `w ∈ 1..=8` 缓存完整预录制 `BakedSpecPlan`（含动态 `dspark_taps_buf` 写入偏移与按窗口规模自适应缩放的 PCIe 并发专家数），将多步验证的 CPU 录制开销降至 `0.00 ms`。
  3. **隔离式零屏障 `Q8_1` 融合着色器与并行投影波次（P1-3）**：将 `Inkling-2` 的 `rmsnorm_q8_1`、`sigmoid_mul_q8_1` 与 `moe_situ_glu_q8_1` 隔离至独立 `inkling2_q8_ops.comp`，确保主干 FP32/MoE 着色器零额外共享内存与满驻留度；同时引入 `dispatch_gemv_prequant` 恢复注意力多分支单波次并发发射并裁掉 SWA 层冗余 KV 输入拷贝，在 `DFlashVkRunner` 融合 `op=4/8` `Q8_1` 打包驱动 `lm_head` `dp4a`。
  4. **短窗口预填充批量输出与 PCIe/DDR4 专家并发切分（P1-4）**：放开 `w >= 2` 批量 2D 输出阶段与 `final_tap` 提取；在 `w ∈ 9..=64` 混合卸载预填充中按专家 ID 50/50 切分（GPU 零拷贝 PCIe 负责前半专家，`moe_reduce` `max_e_excl` 过滤；主机多线程 DDR 并发负责后半专家），而 `w > 64`（如 `pp512`）保持全 GPU `CoopMatF16` 分组 MoE 复用专家权重。
  5. **SWAR 符号扩展 `Q6_K × Q8_1` `dp4a` GEMV + GDN/FullAttn 融合解码（P2-1）**：在 `gemv_dp4a.comp`（含 `M=1/4/8`）与 `gemv_pair_dp4a.comp` 新增 `DTYPE=4`（`Q6_K`），利用无借位 SWAR 位运算将 4×6-bit `(q - 32)` 直接扩展为有符号 `int8x4` 送入 `OpSDotKHR`（零额外 `dm.y` 修正项）；在 `Qwen35VkRunner` 新增 `gdn_prep_decode.comp`（合并 `gdn_gates` + `gdn_conv_silu_l2`）、`gdn_norm_gated_silu_q8_1.comp` 与 `sigmoid_mul_q8_1.comp` 消除中间激活回写重读。
  6. **双缓冲 `2 × 8 MiB` 固定内存流水线 DMA 上传（P2-2）**：在 `VulkanContext` 将单块 `64 MiB` staging 替换为双槽 `2 × 8 MiB` 环形缓冲（主机峰值常驻减半至 `16 MiB`），通过 `write_bytes_from_chunks` 让槽 `i & 1` 的 CPU `pread`/`memcpy` 与槽 `1 - (i & 1)` 的 GPU `vkCmdCopyBuffer` PCIe DMA 全程流水重叠。
- **后果**：在 NVIDIA L4 实测中，`Inkling-2-Mini + DSpark` 开放域散文从负加速 `19.38 tok/s` 提升至 `22.08 tok/s`（`+13.9%`，被拒草稿减少 `52.5%`），Rust 代码在 `TM_VK_DRAFT_VOCAB=65536` 下保持 `88` 接受数与逐位一致 `last_tok` 达 `29.22 tok/s`（短 prompt `pp39` 提速 `+12.6%`），`Qwen3.5-9B` 与 `Inkling-2-Mini`（`26 GiB`）加载耗时缩短 `12.2%–20.4%`，且全量通过本地 NVIDIA、Intel ANV 与 `llvmpipe` 三驱动对齐门禁。

## ADR-023：`qwen35moe`（Qwen3.6-35B-A3B）混合 GDN + 稀疏 MoE 双后端接入与 `Q8_0` 3D 专家 8-lane 合并访存（2026-11-27）

- **状态**：已接受
- **背景**：在与 `Strata` 支持的全部模型家族对比实测中，`Qwen3.6-35B-A3B` 采用 `general.architecture = "qwen35moe"`：主干注意力与状态空间层沿用 3:1 交错的 `Gated DeltaNet`（30 层）与门控全注意力（10 层），但每层 FFN 替换为 **256 路由专家 Top-8（Softmax 重归一化，`norm_weights = true`）+ Sigmoid 标量门控共享专家（`ffn_gate_inp_shexp.weight`）**，且 GGUF `block_count = 41` 包含末尾 1 层 `nextn_predict_layers = 1` MTP 预测块、同时省略顶层 `feed_forward_length` 元数据。此前 `Qwen35Model` 与 `Qwen35VkRunner` 仅接受稠密 `"qwen35"`，且 `moe_indexed_gemv.comp` 的 `Q8_0`（`DTYPE = 2`）分支在 `sg_size <= 32` 下逐块单字节串行读取，未能发挥高带宽显存潜力。
- **决策**：
  1. **统一 `qwen35` / `qwen35moe` 模型与状态契约**：在 `tm-models::qwen35`、`tm-vulkan::qwen35_runner`、`tm-runtime` 与 `tm-cli` 同时识别 `"qwen35"` 与 `"qwen35moe"`；自动按 `block_count - nextn_predict_layers` 裁剪尾部 MTP 块，兼容缺失顶层 `feed_forward_length` 的 MoE GGUF，并将 `"qwen35moe"` 纳入 `parse_moe_arch_params` 的 `default_norm_w = true` 默认规则。
  2. **零拷贝复用 `SparseMoeFfn` 与单 CommandBuffer Vulkan MoE 管线**：CPU 端将 `Qwen35LayerWeights::ffn` 泛化为 `Dense | Moe(Box<SparseMoeFfn>)`（保持单 token 与批处理 `0-ULP` 逐位一致及热路径零堆分配）；Vulkan 端在 `Qwen35VkRunner` 接入 `PendingMat3D`、`moe_router_topk`、`moe_indexed_gemv`（按 `gemv_nr` 启用 `NR=4` 四行复用）与 `moe_reduce`（含 `mode=1` Sigmoid 共享专家门控归约，且 `ffn_gate_inp_shexp` 固定读取 `s_xb` 避免被共享专家 `s_hb` 量化覆盖）。
  3. **工作组级共享内存并行路由（`moe_router_topk.comp` `WG=64`）与 `moe_indexed_gemv.comp` `Q8_0` 8-lane 合并向量化**：将 `moe_router_topk.comp` 从单线程全局内存选择排序升级为每 token 单工作组（`WG = 64`）共享内存（`sh_probs[512]` / `sh_sel[512]`）并行归约与精确索引平局裁决（`e_a < e_b`）；同时将 `moe_indexed_gemv.comp` 的 `DTYPE == 2u` 升级为跨 Subgroup 宽度（`8/16/32/64`）可移植的 8-lane 协作块步进（每轮并行解包 `sg_size / 8` 个 `Q8_0` 块，使用 32-bit `w32` 读取与 `vec4` 激活点积，零 `subgroupShuffle`）。
- **后果**：`tinymodel` 在 CPU 与 Vulkan GPU 双后端原生支持 `Qwen3.6-35B-A3B` 等 `qwen35moe` 模型；合成 `F32` 与 `Q8_0` 多步解码/预填充对齐测试在 NVIDIA、Intel ANV 与 `llvmpipe` 三套驱动全部通过（`F32 rel <= 1e-5`、`Q8_0` 批处理与逐步解码 `rel <= 1e-5`）。
