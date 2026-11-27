# 调研与实测报告索引

01–10 为项目立项期（2026-11-18 至 11-19）的基础调研与固定参考设备档案（保持原样归档）；11–16 为各里程碑阶段的架构扩展设计与真机实测验证报告。若早期调研与后续演进结论存在差异，以 `docs/DECISIONS.md`、`docs/ARCHITECTURE.md` 和 `docs/ROADMAP.md` 为准。

| 文件 | 内容 |
|---|---|
| [01-models-landscape.md](01-models-landscape.md) | 2026 年端侧小模型现状：Qwen3.5、Gemma 4、Phi-4-mini、SmolLM3、LFM2.5、MiniCPM 等，以及推理框架 |
| [02-engine-design-research.md](02-engine-design-research.md) | 引擎设计调研：带宽和算力上限推算、现有引擎对比、算子清单、Zero 2W 档（8.5 节）与主机档（8.6 节） |
| [03-rust-feasibility.md](03-rust-feasibility.md) | Rust 可行性：intrinsics 稳定性实测、GPU 库选型、实施策略与 PoC 计划 |
| [04-strata-lessons.md](04-strata-lessons.md) | Strata（MIT）借鉴清单：状态管理、量化层、Metal GEMV、测试实践与常见陷阱 |
| [05-profile-vs-llamacpp.md](05-profile-vs-llamacpp.md) | 性能剖析与 `llama.cpp` 基线对比 |
| [06-rpi4b-deployment.md](06-rpi4b-deployment.md) | Raspberry Pi 4B 部署要求与优化 |
| [07-on-device-finetuning.md](07-on-device-finetuning.md) | 端侧小模型微调路线（LoRA / QLoRA / MTP）与设备内存需求 |
| [08-pixel8.md](08-pixel8.md) | **Google Pixel 8（Tensor G3）设备档案**：固定参考设备，所有性能与部署报告必覆盖 |
| [09-power-efficiency.md](09-power-efficiency.md) | 降低推理功耗：能耗 / token 指标、线程亲和实测与测量方法 |
| [10-rk3588.md](10-rk3588.md) | **Rockchip RK3588 设备档案**：固定参考设备（4×A76 + 4×A55、Mali-G610、RKNPU2），所有报告必覆盖 |
| [11-amdgpu.md](11-amdgpu.md) | **AMD RX 9060 XT（RDNA4）与 Intel UHD 770 Vulkan 剖析**：预编译 SPIR-V、CoopMat 与对比 ROCm HIP 实测 |
| [11-m3-performance-and-vulkan.md](11-m3-performance-and-vulkan.md) | **M3 性能与跨平台 Vulkan 总结**：`Q4_0x4`、`Q8_0` KV、自适应投机解码及 Pixel 8 / M4 / RDNA4 实测 |
| [12-m4-micro-lowbit-lut-and-vocab-pruning.md](12-m4-micro-lowbit-lut-and-vocab-pruning.md) | **M4 micro 档总结**：`Q2_0`/`TQ2_0`/`IQ` 低比特、16 行 T-MAC `vqtbl1q` 查表内核、BPE 词表裁剪与 4×A510 实测 |
| [13-inkling2-mini-evaluation.md](13-inkling2-mini-evaluation.md) | **Inkling-2-Mini（26B 循环稀疏 MoE）评估**：架构特性与多层循环专家存储剖析 |
| [14-inkling2-mini-q8-design.md](14-inkling2-mini-q8-design.md) | **Inkling-2-Mini Q8_0 三层异构调度设计**：VRAM + RAM + NVMe `O_DIRECT` 零 Swap 与 `DSpark` 分块投机 |
| [15-hf-sub8b-architectures.md](15-hf-sub8b-architectures.md) | **Hugging Face ≤8B 模型架构深度调研**：六大架构谱系、`config.json` 核验与扩展优先级 |
| [16-strata-4families-blackwell-comparison.md](16-strata-4families-blackwell-comparison.md) | **Strata 四大家族与 tinymodel（Vulkan & CUDA 两轮 P0–P2 优化）Blackwell 同机实测报告**：`RTX PRO 6000` 全量吞吐、64 槽热专家缓存消融与六大固定设备对比 |
| [poc/](poc/) | Rust NEON intrinsics 早期实测源码（`dot.rs`、`asm.rs`、`mmasm.rs` 等） |
