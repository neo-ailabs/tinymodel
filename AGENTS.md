# AGENTS.md

本文件给 AI 编码 agent 和人类开发者看。动手之前先读完本文件，再按任务读相关文档。

## 1. 项目一句话

**tinymodel**：用 Rust stable 编写的通用端侧与全场景 LLM 推理及微调引擎，分三档——`micro`（Pi Zero 2W，≤0.5B）、`edge`（Pi 5、Pixel 8、RK3588，≤5B）、`host`（主机，稠密 ≤9B 及 26B–120B 稀疏 MoE，Metal / Vulkan / CUDA）。

## 2. 当前阶段

- **已落地阶段**：**M1–M4、T1–T4（LoRA / QLoRA / MTP）、M5 CUDA 后端及两轮 P0–P2 深度优化（`ADR-025` / `ADR-026`）、M6 多模态视觉 / Mamba-2 / 稀疏 MoE 与 `Strata` 四大家族支持均已完成**；下一步推进 **M5 HTTP 服务层（OpenAI 兼容、Tool Calling、分页 KV）与 C ABI / Python 绑定**，以及 Pi 5 / Zero 2W 物理板复核，见 `docs/ROADMAP.md`。
- 开始任何任务前，先查看 `docs/ROADMAP.md` 的状态列；完成后把状态更新为 ✅，并写上日期。

## 3. 必读文档索引

| 任务类型 | 先读 |
|---|---|
| 任何任务 | 本文件、`docs/REQUIREMENTS.md`、`docs/DECISIONS.md`、`docs/SETTINGS.md` |
| 架构和 crate 改动 | `docs/ARCHITECTURE.md` |
| CPU 和 GPU 内核 | `docs/ARCHITECTURE.md` 5.3、5.4 节；`docs/research/04-strata-lessons.md` 第 2、3 节；`docs/research/03-rust-feasibility.md` 第 2 节；`docs/research/16-strata-4families-blackwell-comparison.md` |
| 新模型结构 | `docs/ARCHITECTURE.md` 5.2 节；Qwen3.5/3.6 还要读 `Strata/docs/GDN_CONTRACT.md` |
| 测试和基准 | `docs/TESTING.md` |
| 服务层 | `docs/ARCHITECTURE.md` 第 6 节；`docs/research/04-strata-lessons.md` 第 4、5 节 |

## 4. 仓库布局（crate 以 `tm-` 为前缀）

```text
crates/tm-core       dtype、量化块、TensorView、Arena、错误类型（无外部依赖）
crates/tm-gguf       GGUF（含多分片）解析、mmap、ModelSpec 校验、离线重排与词表裁剪
crates/tm-cpu        CPU 内核：generic（参考实现）/ aarch64 / x86_64 / asm / isa / registry / pool
crates/tm-metal      Apple Metal 后端（objc2-metal + MSL，feature 开关）
crates/tm-vulkan     跨平台 Vulkan 后端（ash + 预编译 SPIR-V 1.2，CoopMat / Subgroup / 原生 Q2_0）
crates/tm-cuda       NVIDIA CUDA 后端（动态 libcuda Driver API + compute_75/80 双 PTX + CudaGraph）
crates/tm-lora       兼容 llama.cpp 的动态 GGUF LoRA 挂载与无损 lora-merge（100% safe Rust）
crates/tm-models     各模型结构的零分配 forward、.strata 解析、多模态视觉 ViT 与神经投机头
crates/tm-train      手写逐层激活重算 QLoRA 与 train-mtp 草稿头训练（100% safe Rust + GPU 加速）
crates/tm-runtime    Session、LayerState、Sampler、spec（Suffix/MTP/DFly2/DSpark）、PrefixCache
crates/tm-tokenizer  内置 BPE / SentencePiece / HF tokenizer.json（无外部重依赖）
crates/tm-cli        CLI（run / bench / convert / finetune / train-mtp / lora-merge / probe）
tools/               Python FP64 oracle、logits_diff.py、quant_quality.py、LEGO 微调流水线
bench/               基准协议、设备统一调度器 device_harness.py、基线与实测结果
docs/                正式设计文档与 SETTINGS.md；docs/research/ 存放硬件档案与实测报告
website/             零构建依赖纯静态项目官网（HTML + CSS + ES Module JS）
Strata/              第三方参考项目（MIT），只读，不要修改，也不要让它参与构建
```

## 5. 硬性规则（MUST）

1. **只用 Rust stable**。不要用 `#![feature]`。还不稳定的指令（i8mm、FP16 运算、SVE/SME、AMX）只能在 `tm-cpu/src/*/asm` 里用 `asm!` 实现。
2. **`unsafe` 只能出现在** `tm-cpu`、`tm-metal`、`tm-cuda`、`tm-vulkan`、`tm-ffi`、`tm-gguf`（仅限 mmap 部分）。每个 `unsafe` 块都要写 `// SAFETY:` 注释。
3. **新内核三件套**：
   - `generic` 标量参考实现；
   - 和参考实现**逐位一致**的对齐测试（GPU 允许按文档规定的容差）；
   - bench。

   三样缺一不可，否则不能合并。
4. **生成 token 的路径上不做堆分配**：所有缓冲在加载时从 Arena 切好。
5. **按张量自身的 dtype 分派内核**，不要假设同一种张量在每一层都用同一种量化格式；激活量化格式跟随权重类型。
6. **会改状态的操作**：如果走不了，必须在修改状态之前返回错误；同一层状态上的调用严格顺序执行。
7. **新模型结构**必须有端到端的**多 token** logits 比对（`tools/logits_diff.py`），不能只测单 token。
8. **声称有性能提升**，必须附上同一台机器、按 `docs/TESTING.md` 基准协议测得的基线数据。
9. **移植代码**（ggml、Strata、KleidiAI）时，在文件头保留原版权声明和许可证，并注明来源路径和上游 commit。
10. **不要修改 `Strata/`**，也不要把它加进 workspace。
11. 架构层面的取舍写进 `docs/DECISIONS.md`（追加新的 ADR，不修改已接受的条目）。

## 6. 约定（SHOULD）

- 错误处理：库 crate 用 `thiserror` 定义具体的错误类型，二进制 crate 可以用 `anyhow`；GGUF 解析报错时要带上文件名、张量名和层号。
- 不要用 `unwrap()`，测试和不可能失败的常量除外。
- SIMD 函数写成 `#[target_feature(enable = "...")] unsafe fn`，在 `registry.rs` 里注册，运行时按 ISA 从宽到窄选择。可以用 `TM_FORCE_ISA` 环境变量强制降档。
- 新增环境变量要登记到 `docs/SETTINGS.md`，写明默认值、用途和状态。尽量少加开关。
- 代码注释用英文；文档可以用中文。
- 提交信息格式：`<crate>: <动词开头的简述>`，例如 `tm-cpu: add q4_0 x q8_0 dotprod gemv`。

## 7. 常用命令

```sh
cargo build --release --features "vulkan,cuda"         # 本机 host 全特性构建（macOS 用 --features metal）
cargo test --workspace --features "vulkan,cuda" -- --test-threads=1   # 全栈 L0/L1/L3 测试门禁
TM_FORCE_ISA=generic cargo test -p tm-cpu              # 降档跑标量对齐测试
cargo bench -p tm-cpu                                  # CPU 内核微基准
cargo zigbuild --release --target aarch64-unknown-linux-musl --no-default-features --features micro   # Zero 2W
cargo zigbuild --release --target aarch64-unknown-linux-musl --features edge                         # Pi 5 / RK3588
cargo clippy --workspace --all-targets --features "vulkan,cuda" -- -D warnings
cargo fmt --all
```

## 8. 环境

- 开发机：Apple M4，16GB，macOS；工具链 `stable`（1.99.1）和 `nightly`。**项目固定用 stable**（`rust-toolchain.toml`）。
- M4 支持 NEON、dotprod、i8mm、SME2；本机实测见 `docs/research/poc/`。
- 交叉编译用 `cargo-zigbuild` 或 `cross`。
- **分析报告的固定参考设备**：凡是涉及性能、内存、部署、微调、延迟、功耗的分析，除了 M4、Pi 5、Pi 4B / Zero 2W 之外，**都要附上 Google Pixel 8（Tensor G3，Android）和 Rockchip RK3588（4×A76 + 4×A55，Linux SBC）的分析**。设备档案见 `docs/research/08-pixel8.md` 和 `docs/research/10-rk3588.md`。

## 9. 已知陷阱（务必注意）

- **GDN（Qwen3.5）**：SiLU 作用在拆分 q/k/v **之前**；v 不做 L2 归一化；q 额外乘 1/√S；value head 对应的 key head 是 `vh % HK`（**和 FLA 的分组方向相反**）；输出门 Qwen3.5 用 SiLU。**这些地方写错了，输出看起来仍然正常**。
- **Metal**：threadgroup 超过 `maxTotalThreadsPerThreadgroup` 时 dispatch 会**静默不执行**，每次都要检查；Apple GPU 上发散地访问 constant 内存会串行化，码本要放进 threadgroup 内存。
- **线程池**：认领任务要带 epoch，防止晚醒的 worker 抢到下一批任务；worker 不要放在 SMT 兄弟线程上。
- **确定性**：多 token 内核里每个 token 的运算顺序必须和单 token 内核一致，否则输出会因为是否投机而不同。
- **stop 串**要在文本层判断，并带回扣机制，不能用分词后的子 token 当 stop id。
- **内存探测**：要读 cgroup v2 的限制（Docker 和树莓派场景）；规划器的输入不允许默默默认为 0。
- **macOS**：大 arena 要 mlock，否则可能被内存压缩。
