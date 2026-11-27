# 需求规格（Requirements）

> 状态：v0.2（初版 2026-11-19；2026-11-27 同步 M1–M4、T1–T4 及 `ADR-016`/`ADR-018`/`ADR-019`/`ADR-024`–`ADR-026` 扩展）。调研依据见 `docs/research/`。

## 1. 项目定位

**tinymodel** 是用 Rust 编写的通用端侧 LLM 推理引擎，目标硬件覆盖 **Raspberry Pi Zero 2W 到高性能主机**，支持**最高 8B 参数**的模型。

差异化目标：
1. **低功耗设备上做到最好**。现有引擎（llama.cpp、candle、mistral.rs）都没有针对树莓派这一档深度优化，Zero 2W 级别的设备更是没有专门的内核。
2. **同一份代码覆盖三档硬件**，按编译配置裁剪。
3. **一等支持新一代小模型结构**：混合线性注意力（Qwen3.5）、PLE（Gemma 4）、短卷积（LFM2）。

## 2. 硬件分档（Profiles）

| Profile | 目标设备 | 模型规模 | 内存 | 带宽 | CPU 特性 | 加速器 |
|---|---|---|---|---|---|---|
| `micro` | Raspberry Pi Zero 2W（以及 Pi 3 / Pi 4） | **≤0.5B**（实际主要是 100–350M） | 512MB（实际可用约 350MB） | 约 2 GB/s | Cortex-A53，ARMv8.0 NEON（**没有 dotprod，不支持 FP16 运算**） | 无 |
| `edge` | Raspberry Pi 5、手机（参考机型 **Google Pixel 8 / Tensor G3**）、RK3588、Jetson | **≤5B** | 4–16GB | 约 11.5 GB/s（Pi 5） | A76+：NEON + dotprod（i8mm 视芯片而定） | 可选 Vulkan |
| `host` | x86 台式机、Apple Silicon、独显主机 | **≤8B**（可延伸到 9B，例如 Qwen3.5-9B） | 16GB 以上 | 70 GB/s – 1.8 TB/s | AVX2 / AVX-512 / VNNI / AMX；NEON / i8mm / SME2 | **Metal、Vulkan、CUDA**（按此顺序实现） |

主力开发机：Apple M4，16GB（属于 host 档，同时用来开发 ARM CPU 内核）。

## 3. 功能需求

### 3.1 模型结构（按优先级）

| 优先级 | 结构 | 代表模型 | 关键算子和状态 |
|---|---|---|---|
| P0 | 标准 Transformer（GQA） | Llama 3.2 1B/3B、Llama 3.1 8B、Qwen3 0.6B–8B、SmolLM2/3、Phi-4-mini | RMSNorm、RoPE（含 YaRN 等长上下文缩放）、GQA、SwiGLU、KV cache |
| P0 | 混合线性注意力（3:1） | **Qwen3.5** 0.8B / 2B / 4B / 9B | Gated DeltaNet（分块扫描 + 单步递推）、短卷积、门控全注意力、循环状态 |
| P1 | 滑动窗口注意力 + PLE + MatFormer | **Gemma 4 E2B / E4B**、Gemma 3 / 3n | 滑动窗口、soft-capping、QK-norm、PLE 按行读取、KV 层间共享 |
| P1 | 短卷积混合 | LFM2 / LFM2.5（350M–2.6B） | 门控短卷积 + GQA |
| P2 | Mamba2 混合 | Granite 4 等 | selective scan |
| P2 | 小型 MoE（总参数 ≤8B） | LFM2-8B-A1B 等 | top-k 路由、专家 GEMV |
| P2 | 多模态编码器 | Qwen3.5-VL、Gemma 4、MiniCPM-V | ViT、Conformer 音频编码器、投影层 |
| P3 | 三值 / BitNet | BitNet b1.58 | 查表矩阵乘 |

### 3.2 模型格式与量化
- **F-01** 直接加载 **GGUF**（mmap、零拷贝），按每个模型的张量表逐层校验（名字、形状、dtype）。
- **F-02** 量化格式：
  - 主力：**Q4_0（重排布局）**、Q8_0；
  - 兼容：Q4_K / Q5_K / Q6_K、IQ4_NL / IQ4_XS；
  - 极低比特（micro 档）：IQ2 / IQ3、Q2_0、三值；
  - 原始权重：BF16 / F16 / F32。
- **F-03** 按张量自身的 dtype 分派内核（支持混合量化）；激活量化格式跟随权重类型。
- **F-04** 离线工具：重排权重、裁剪词表（micro 档）、导出内部布局缓存。

### 3.3 运行时
- **F-10** 统一的 `LayerState`，同时覆盖 KV cache、DeltaNet/SSM 状态、卷积历史。
- **F-11** 采样：greedy、temperature、top-k、top-p、min-p、重复惩罚、logit bias、固定种子。
- **F-12** 投机解码：
  - n-gram / suffix 查找（不需要额外权重）；
  - MTP 头；
  - 小草稿模型（host 档）；
  - 按收益自动开关。
- **F-13** 前缀缓存：KV 按前缀复用；递归状态按检查点复用（检查点放在 system prompt 末尾、每轮开头、每 N 个 token），内存有上限，按 LRU 淘汰。
- **F-14** KV cache 量化（Q8 / Q4）；host 档支持分页 KV。
- **F-15** 会话状态可以保存到磁盘和恢复。
- **F-16** host 档：同时处理 2–8 条请求、连续批处理、CPU 与 GPU 混合执行（按层卸载）。

### 3.4 接口
- **F-20** CLI：`run`、`chat`、`bench`、`inspect`（查看 GGUF 内容）、`convert`。
- **F-21** OpenAI 兼容的 HTTP 服务（host 和 edge 档）：流式输出、tool calling、聊天模板（minijinja）。
- **F-22** C ABI（cbindgen），以及 Python 绑定（pyo3）。移动端绑定（uniffi）是远期目标。
- **F-23** Tokenizer：内置轻量的 BPE 和 SentencePiece 实现（不依赖 HF tokenizers）。

## 4. 非功能需求

| ID | 需求 | 指标 |
|---|---|---|
| N-01 | **正确性** | 贪心解码的输出和 llama.cpp / HF 参考实现逐 token 一致；量化内核和标量参考实现**逐位一致** |
| N-02 | decode 性能 | 达到理论带宽上限的 ≥80%；edge 档有效 decode 速度 ≥1.3× llama.cpp（含投机解码） |
| N-03 | prefill 性能 | edge 档 ≥1.5–2× llama.cpp |
| N-04 | Metal 性能 | M4 上 decode 速度 ≥90% MLX / llama.cpp Metal |
| N-05 | 体积 | micro 档的静态二进制 **≤2MB**（目标 1MB） |
| N-06 | 内存 | micro 档在可用内存约 350MB 的条件下能跑 SmolLM2-360M Q4；生成 token 的路径上**零堆分配** |
| N-07 | 可移植性 | 一个二进制通过运行时 ISA 选择，覆盖同一架构下的所有 CPU |
| N-08 | 工具链 | 固定使用 **Rust stable**，不依赖 nightly |
| N-09 | 安全 | `unsafe` 只能出现在内核 crate 和 FFI crate 里；GGUF 解析能扛住损坏的输入（fuzz 测试） |
| N-10 | 确定性 | 同样的输入、种子和硬件，输出必须一致；多 token 内核里每个 token 的运算顺序和单 token 内核一致 |

## 5. 不做的事（Non-goals）

- 全参数预训练与全参数微调（支持 GGUF LoRA 推理/合并、MTP 草稿头微调与冻结量化基座上的 LoRA 微调，见 `ADR-015`）。
- 多卡分布式张量并行（单机单卡下的 26B–120B 稀疏 MoE 推理、三层 `O_DIRECT` 调度与零拷贝 DDR 卸载 + 热专家 VRAM 缓存已按 `ADR-016`/`ADR-024`–`ADR-026` 支持）。
- 服务器级别的高并发（vLLM 那种）。
- 早期不做 NPU（Hailo、QNN、RKNPU），放到最后阶段做实验性支持。
- 不做通用计算图编译器（不走 TVM / MLC 路线）。

## 6. 验收用的基准模型集

| Profile | 模型 |
|---|---|
| micro | SmolLM2-135M、SmolLM2-360M、LFM2-350M、Gemma 3 270M |
| edge | Llama 3.2 1B/3B、Qwen3.5-0.8B/2B/4B、Gemma 4 E2B、Phi-4-mini（测试设备：Pi 5、**Google Pixel 8**） |
| host | Llama 3.1 8B、Qwen3-8B、Qwen3.5-9B、Gemma 4 E4B |
