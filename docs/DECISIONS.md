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

### ADR-010 AMD GPU 支持走 Vulkan 后端，HIP 推迟到 M5
- **状态**：Accepted（2026-11-21，用户确认）
- **背景**：开发主机有 AMD Radeon RX 9060 XT（RDNA4/gfx1200，8 GiB），可选 Vulkan（RADV）或 HIP/ROCm 7.1.1。分析见 `research/11-amdgpu.md`。
- **决策**：AMD 显卡通过 `tm-vulkan`（ash + GLSL→SPIR-V）支持，本机作为 Vulkan 主力开发机；HIP 不单独立项，等 V3 拿到 prefill 差距数据后，再评估是否在 M5 和 `tm-cuda` 共用 kernel 源码。
- **后果**：Vulkan 后端从 M3 提前开始；kernel 需要兼容 subgroup 宽度 16/32/64；CI 用 lavapipe 跑对齐测试；普通构建不能强制依赖 `glslc`（需要提交预编译的 SPIR-V）。
