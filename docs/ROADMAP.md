# 路线图（Roadmap）

> 每个阶段都有可量化的验收标准。基线以同一台机器上的 llama.cpp 为准（见 `TESTING.md` 的基准协议）。

| 阶段 | 内容 | 验收标准 | 状态 |
|---|---|---|---|
| **M0 基线** | 在 M4 和 Pi 5 上用 llama.cpp 测量基准模型集的 pp、tg、内存峰值、温度和频率 | 有可复现的基线表格，存入 `bench/baselines/` | 🟡 M4 已完成（2026-11-19），Pi 5 待测 |
| **PoC** | ① 用 Rust 写 Q4_0×Q8_0 GEMV（NEON dotprod + i8mm `asm!`），以 `Strata/src/quant/arm/arm_dots.inc` 为蓝本；② 交叉编译 aarch64-musl，在 Pi 5 和 Zero 2W 上运行；③ objc2-metal 量化 GEMV；④ 内核对齐和 bench 测试框架 | ① 在 M4 上 ≥95% ggml；② 二进制 <2MB，Pi 5 带宽利用率 ≥80%；③ ≥90% llama.cpp Metal | 🟡 M4 全部达标（2026-11-19：① 120% ggml；② 404KB/407KB 静态 ELF；③ 90.3 GB/s ≈ 105% llama.cpp Metal；④ L1/L1b 齐备），Pi 5 真机待跑 |
| **M1 最小可用** | Cargo workspace 骨架；tm-core / tm-gguf（张量表校验）/ tm-cpu（generic + NEON + dotprod + AVX2）/ 线程池；Llama + Qwen3 的 forward；tokenizer；采样；CLI `run` / `bench` | 贪心解码输出和 llama.cpp 逐 token 一致；Pi 5 decode ≥95% llama.cpp | ⬜ |
| **M2 新结构 + Metal** | Qwen3.5（GDN，按 `GDN_CONTRACT` 实现）、Gemma 4（PLE、滑动窗口）、LFM2；LayerState；前缀缓存；**Metal 后端** | 多 token logits 的 KL 低于阈值；M4 Metal decode ≥90% MLX | ⬜ |
| **M3 性能** | prefill GEMM 微内核；suffix 和 MTP 投机解码 + 按收益选择；多 token 共享权重；LM head 优化；KV 量化；**Vulkan 后端**（覆盖 AMD、Intel 显卡、Mali、Pi 5 VideoCore） | Pi 5 prefill ≥1.5× llama.cpp，有效 decode ≥1.3× | ⬜ |
| **M4 micro 档** | ARMv8.0 NEON 和 TBL 查表内核；低比特格式（IQ2/3、Q2_0、三值）；词表裁剪；零拷贝 | Zero 2W 上 SmolLM2-360M Q4 ≥5 tok/s，内存 ≤350MB | ⬜ |
| **M5 服务与生态** | OpenAI 兼容服务（tool calling、模板）；C ABI；Python 绑定；**CUDA 后端**；分页 KV；并发 | 工具调用语料切片测试全部通过 | ⬜ |
| **M6 扩展** | 多模态编码器；小型 MoE；Mamba2；NPU 实验（Hailo / RKNPU） | — | ⬜ |
