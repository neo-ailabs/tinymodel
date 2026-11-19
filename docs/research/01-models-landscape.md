# 端侧小模型调研报告（截至 2026-11）

> [!NOTE]
> 信息来自网络检索结果的汇总，经过交叉比对。标 ⚠️ 的条目只有单一来源或细节存疑，选型前请到官方模型卡确认。

## 1. 主流端侧小模型一览

| 模型系列 | 厂商 | 端侧常用尺寸 | 模态 | 许可证 | 亮点 |
|---|---|---|---|---|---|
| **Qwen3.5 小模型** | 阿里 | 0.8B / 2B / 4B / 9B（2026-03 发布） | 文本 + 视觉 | Apache 2.0 | 262K 上下文，可切换思考/非思考模式，多语言和工具调用强，中文首选 |
| **Qwen3** | 阿里 | 0.6B / 1.7B / 4B / 8B | 文本 | Apache 2.0 | 生态成熟，GGUF/MLX 量化版本最全 |
| **Gemma 4 E2B / E4B** | Google | 有效参数 2.3B / 4.5B（总参数 5.1B / 8B） | 文本 + 图像/视频 + 音频 | Apache 2.0 | 用 PLE（逐层嵌入）降低内存占用，128K 上下文，原生函数调用，支持 MTP 加速 |
| **Gemma 3n E2B / E4B** | Google | 同上一代 | 文本 + 图像 + 音频 | Gemma 条款 | 采用 MatFormer，可裁剪子模型 |
| **Phi-4-mini / Phi-4-multimodal** | 微软 | 3.8B | 文本（multimodal 版另含图像和音频） | MIT | 数学、推理、代码在同尺寸中最强，主要面向英文 |
| **SmolLM3** | Hugging Face | 3B（另有 135M/360M/1.7B 的 SmolLM2） | 文本 | Apache 2.0 | 训练数据和流程完全公开，支持双模式推理，适合研究和二次训练 |
| **Llama 3.2** | Meta | 1B / 3B | 文本 | Llama 社区许可 | 1B Q4 量化约 0.7GB，常被用作最低基线 |
| **MobileLLM** | Meta | 125M ~ 1B | 文本 | 研究许可 | 网络深而窄、共享嵌入、GQA，主要供研究参考 |
| **LFM2 / LFM2.5** | Liquid AI | 350M ~ 2.6B，另有 24B-A2B MoE | 文本 / 视觉 / 音频 | LFM 开放许可 | 非纯 Transformer 的混合架构，解码速度很快，适合微调 |
| **MiniCPM / MiniCPM-V / MiniCPM-o** | 面壁智能 | 1B ~ 8B；⚠️ MiniCPM5-2B、MiniCPM-o 4.5 | 全模态（看、听、说） | 开源（需商用登记） | 国产端侧代表，已在多家车企座舱量产，推进 1.58-bit 三值量化 |
| **Apple Foundation Model** | Apple | 约 3B 稠密（⚠️ 新一代据称另有稀疏大版本） | 文本 + 图像 | 不开放权重 | 只能在 Apple 设备上通过 Swift `FoundationModels` 框架调用 |
| **Gemini Nano** | Google | 不公开 | 多模态 | 不开放权重 | 通过 Android AICore 在 Pixel、三星等设备上调用 |

## 2. 按参数规模划分

| 档位 | 代表模型 | Q4 内存 | 适用设备与任务 |
|---|---|---|---|
| **< 1B** | Qwen3.5-0.8B、Qwen3-0.6B、SmolLM2-360M、LFM2-350M、MobileLLM | < 0.6 GB | IoT、浏览器（WASM/WebGPU）；分类、补全、意图路由 |
| **1–2B** | Llama 3.2 1B、Qwen3.5-2B、Gemma 4 E2B、LFM2-1.2B、MiniCPM 2B | 0.7–1.5 GB | 中低端手机、树莓派；摘要、改写、简单对话 |
| **3–4B** | Qwen3.5-4B、Gemma 4 E4B、Phi-4-mini、SmolLM3、Llama 3.2 3B | 2–3 GB | 8GB 内存的旗舰手机和笔记本；RAG、工具调用、多模态 |
| **7–9B** | Qwen3.5-9B、Qwen3-8B、MiniCPM-V 8B | 5–6 GB | 16GB 以上的 Mac 或 AI PC；代码、复杂推理 |

## 3. 按需求选型

- **中文 + 通用 + 工具调用**：Qwen3.5（2B/4B）
- **手机端多模态（图像、语音）**：Gemma 4 E2B/E4B、MiniCPM-o
- **数学和推理（英文）**：Phi-4-mini
- **完全开源可复现、自己训练**：SmolLM3
- **极致速度或针对垂直领域微调**：LFM2.5
- **iOS 原生应用**：Apple Foundation Models 框架
- **Android 系统级能力**：Gemini Nano（AICore）或 Gemma + LiteRT-LM

## 4. 端侧推理框架

| 框架 | 格式 | 适用场景 |
|---|---|---|
| **llama.cpp**（含 Ollama、LM Studio） | GGUF | 全平台通用，CPU/Metal/Vulkan/CUDA 都支持 |
| **MLX / mlx-lm** | safetensors | Apple Silicon 上最快，自带 LoRA 训练 |
| **ExecuTorch** | PyTorch 导出 | iOS/Android 生产部署，支持高通 QNN 等 NPU |
| **LiteRT-LM / MediaPipe**（Google AI Edge） | .litertlm / .task | Gemma 系列在 Android/iOS/Web 上部署 |
| **MLC-LLM / WebLLM** | MLC | 浏览器端 WebGPU 推理 |
| **MNN / MNN-LLM**（阿里）、**mllm** | 自定义 | 国产手机和 NPU 适配 |

## 5. 2026 年趋势

1. **用“有效参数”控制内存**：PLE、MatFormer、MoE（如 LFM2.5-24B-A2B）让模型总参数更大，但实际运行时只占小模型的内存。
2. **原生全模态**：视觉、音频编码器内置到 2–4B 模型里，可以在设备上完成实时语音交互（还能配合 Kokoro-82M 等轻量 TTS）。
3. **思考模式可切换**：小模型也支持长链推理，按需开启。
4. **极低比特量化**：从 4-bit 普遍可用，推进到 1.58-bit 三值量化。
5. **NPU 普及**：手机、AI PC 和车机都配备 NPU，ExecuTorch、FlagOS 等统一软件栈减少了碎片化问题。
