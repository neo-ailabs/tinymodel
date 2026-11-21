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
