# tinymodel

用 Rust 编写的通用端侧 LLM 推理引擎，覆盖 **Raspberry Pi Zero 2W 到高性能主机**，支持 **≤8B** 参数的模型。

> 状态：**设计阶段**（M0 / PoC 尚未开始）。目前仓库里只有文档。

## 目标

| Profile | 设备 | 模型 | 后端 |
|---|---|---|---|
| `micro` | Pi Zero 2W / Pi 3 / Pi 4 | ≤0.5B | ARMv8.0 NEON、查表内核 |
| `edge` | Pi 5、手机、ARM 开发板 | ≤5B | NEON dotprod / i8mm，可选 Vulkan |
| `host` | x86 主机、Apple Silicon、独显主机 | ≤8B | AVX2 / AVX-512、Metal、Vulkan、CUDA |

- 模型结构：Llama / Qwen3、**Qwen3.5（Gated DeltaNet 混合）**、**Gemma 4（PLE）**、LFM2，后续支持 Mamba2、小型 MoE 和多模态。
- 格式：直接加载 GGUF；主力量化格式是 Q4_0（重排布局）和 Q8_0，兼容 K-quant 和 IQ 系列。
- 性能抓手：降低每个 token 读取的字节数、投机解码和 MTP、prefill GEMM 微内核。

## 文档

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | **开发者和 AI agent 必读**：规则、约定、命令 |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | 需求规格（分档、模型、功能、非功能、不做的事） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计（crate 结构、LayerState、内核、后端） |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 路线图和验收标准 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 架构决策记录（ADR） |
| [docs/TESTING.md](docs/TESTING.md) | 正确性分层和基准协议 |
| [docs/research/](docs/research/) | 调研报告：模型现状、引擎设计、Rust 可行性、Strata 借鉴 |

## 参考代码

`Strata/` 是第三方项目（MIT，github.com/Niko1221/Strata），**只作为参考**，不属于本项目的源码。借鉴要点见 [docs/research/04-strata-lessons.md](docs/research/04-strata-lessons.md)。

## 许可

采用以下两种许可证之一，由使用者任选：

- Apache License 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT License（[LICENSE-MIT](LICENSE-MIT)）

除非另有明确声明，你有意提交给本项目的任何贡献，都按上述双许可发布，不附加任何其他条款或条件。

移植自 ggml、Strata（MIT）或 KleidiAI（Apache-2.0）的代码，在文件头保留原有的版权声明，并汇总到 `THIRD_PARTY_NOTICES.md`。
