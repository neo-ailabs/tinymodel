# 调研报告索引

以下是 2026-11-18 至 11-19 的调研快照，**内容保持原样不再更新**。结论已经整理进 `docs/REQUIREMENTS.md`、`ARCHITECTURE.md` 和 `DECISIONS.md`；如果有冲突，以这些正式文档为准。

| 文件 | 内容 |
|---|---|
| [01-models-landscape.md](01-models-landscape.md) | 2026 年端侧小模型现状：Qwen3.5、Gemma 4、Phi-4-mini、SmolLM3、LFM2.5、MiniCPM 等，以及推理框架 |
| [02-engine-design-research.md](02-engine-design-research.md) | 引擎设计调研：带宽和算力上限的推算、现有引擎对比、算子清单、优化项；附 Zero 2W 档（8.5 节）和主机 ≤8B 档（8.6 节） |
| [03-rust-feasibility.md](03-rust-feasibility.md) | Rust 可行性：本机实测 intrinsics 的稳定性、GPU 库选型、实施策略、PoC 计划 |
| [04-strata-lessons.md](04-strata-lessons.md) | Strata（MIT）借鉴清单：状态管理、量化层、Metal GEMV、测试实践，以及踩过的坑 |
| [poc/](poc/) | Rust NEON intrinsics 实测源码（`dot.rs` 在 stable 上可用；`mm.rs`、`f16.rs` 不能编译；`asm.rs`、`mmasm.rs` 用 asm! 绕过） |
