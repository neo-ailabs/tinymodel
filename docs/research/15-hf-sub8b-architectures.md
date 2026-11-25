# Hugging Face ≤8B 模型架构深度调研报告（2026-11-24）

> [!NOTE]
> 本报告基于 **2026-11-24** 对 Hugging Face Hub API（`https://huggingface.co/api/models`）的实时采样与代表仓库 `config.json` 原始字段核验，并结合架构文献交叉整理。主要服务于 `tinymodel` 后续模型覆盖与算子规划。

## 1. 采样方法与口径

1. **纯文本生成（`text-generation`）**：按月下载量（`downloads`）抓取 Top-1,000 与按热度（`trendingScore`）抓取 Top-300，筛选包含 `safetensors.total` 元数据且参数量落在 **`50M – 8.5B`** 的模型（涵盖标称 8B 实际 8.03B–8.5B 的常见变体）。
   - Top-1,000 中命中 **292 个 ≤8.5B 模型**，合计月下载量 **152,776,326 次**。
   - 热度 Top-300 中命中 **80 个 ≤8.5B 模型**，用于捕捉最新发布的架构趋势。
2. **多模态（`image-text-to-text`）**：按月下载量抓取 Top-200，筛选 `50M – 8.5B`，命中 **44 个模型**。由于 2025–2026 年大量新一代小模型（如 Qwen3.5、Gemma 3/4、Ministral 3）原生合一了视觉编码器，必须结合该类目才能还原真实生态体量。
3. **配置核验**：直接拉取各代表架构官方仓库的 `config.json`，核对 `model_type`、`architectures`、`layer_types`、`hybrid_override_pattern`、`no_rope_layers`、`sliding_window`、MoE 与 Mamba-2 参数。

---

## 2. Hugging Face Hub 实测分布

### 2.1 `text-generation` 月下载 Top-1,000（≤8.5B，共 292 个模型）

| 排名 | `model_type` | 模型数（占比） | 月下载量（占比） | 代表模型（参数量） | `tinymodel` 现状 |
|---:|---|---:|---:|---|---|
| 1 | **`llama`** | 64 (21.9%) | 19,563,726 (12.8%) | `meta-llama/Llama-3.1-8B-Instruct` (8.03B) | ✅ 已支持 |
| 2 | **`qwen2`** | 55 (18.8%) | 29,878,413 (19.6%) | `Qwen/Qwen2.5-7B-Instruct` (7.62B) | ✅ 已支持 |
| 3 | **`qwen3`** | 51 (17.5%) | 45,980,115 (30.1%) | `Qwen/Qwen3-0.6B` (0.75B) | ✅ 已支持 |
| 4 | `gpt_neox` | 9 (3.1%) | 4,081,207 (2.7%) | `EleutherAI/pythia-160m` (0.21B) | —（历史基线） |
| 5 | **`mistral`** | 8 (2.7%) | 2,374,918 (1.6%) | `mistralai/Mistral-7B-Instruct-v0.2` (7.24B) | ✅ 已支持（P0） |
| 6 | `gpt2` | 6 (2.1%) | 36,242,551 (23.7%) | `openai-community/gpt2` (0.14B) | —（历史/CI 基线） |
| 7 | **`gemma3_text`** | 6 (2.1%) | 454,525 (0.3%) | `google/gemma-3-1b-it` (1.00B) | ✅ 已支持（P0） |
| 8 | **`lfm2`** | 6 (2.1%) | 522,345 (0.3%) | `LiquidAI/LFM2.5-230M-Base` (0.23B) | ✅ 已支持 |
| 9 | **`granite`** | 5 (1.7%) | 586,190 (0.4%) | `ibm-granite/granite-4.1-3b` (3.40B) | ✅ 已支持（P1） |
| 10 | **`phi3`** | 5 (1.7%) | 1,168,582 (0.8%) | `microsoft/Phi-3-mini` / `Phi-4-mini` (3.82B) | ✅ 已支持（P0） |
| 11 | **`qwen3_5`** | 4 (1.4%) | 2,267,361 (1.5%) | `Qwen3.5-0.8B/2B/4B/9B`（主力在 VLM 类目） | ✅ 已支持 |
| 12 | **`olmo3`** | 4 (1.4%) | 359,402 (0.2%) | `allenai/Olmo-3-7B-Instruct` (7.30B) | ✅ 已支持（P1） |
| 13 | **`exaone`** | 3 (1.0%) | 421,979 (0.3%) | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` (7.82B) | ✅ 已支持（P2a） |
| 14 | **`smollm3`** | 3 (1.0%) | 483,866 (0.3%) | `HuggingFaceTB/SmolLM3-3B` (3.08B) | ✅ 已支持（P0） |
| 15 | **`nemotron_h`** | 3 (1.0%) | 327,774 (0.2%) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (3.97B) | ✅ 已支持（P1） |
| 16 | **`gemma2`** | 3 (1.0%) | 384,068 (0.3%) | `google/gemma-2-2b-it` (2.61B) | ✅ 已支持（P0） |
| 17 | **`olmo2`** | 3 (1.0%) | 281,785 (0.2%) | `allenai/OLMo-2-0425-1B` (1.48B) | ✅ 已支持（P1） |
| 18 | **`gemma4`** | 3 (1.0%) | 258,626 (0.2%) | `Gemma-4-E2B` / `E4B`（主力在 VLM 类目） | ✅ 已支持 |
| 19 | `nemotron_labs_diffusion` | 2 (0.7%) | 300,104 (0.2%) | `nvidia/Nemotron-Labs-Diffusion-3B` (3.83B) | ⬜ 扩散范式 |
| 20 | **`olmoe` / `granitemoe`** | 4 (1.4%) | 317,391 (0.2%) | `OLMoE-1B-7B` (6.92B) / `granite-3.1-3b-a800m` | ✅ 已支持（P2a） |
| 21 | `minicpm_sala` | 1 (0.3%) | 631,783 (0.4%) | `openbmb/MiniCPM-SALA` | ⬜ 需稀疏/线性混合 |
| 22 | **`bailing_hybrid` (`bailingmoe2`)** | 1 (0.3%) | 447,716 (0.3%) | `inclusionAI/Ling-3.0-tiny` (7.89B) | ✅ 已支持（P2a） |
| 23 | **`lfm3`** | 1 (0.3%) | 146,872 (0.1%) | `LiquidAI/LFM3-1.7B` (1.73B) | ✅ 已支持（P0） |

**核心统计结论**：
- **`llama + qwen2 + qwen3`** 三者合计占 Top-1,000 ≤8.5B 模型数的 **58.2%**、月下载量的 **62.5%**；剔除 `gpt2`（23.7%，主要为教程与 CI 测试调用）和 `gpt_neox`（2.7%）后，在现代纯文本小模型下载量中占比高达 **~85%**。
- 热度榜（`trendingScore` Top-300）则清晰展现了 2026 年的新架构增量：`qwen3_5`、`lfm3`、`smollm3`、`olala`（6.77B-A1B MoE）、`aurora2`（400M-A20M MoE）、`nemotron_h`、`granitemoehybrid`、`bailing_hybrid`、`bitnet`（`bitnet-b1.58-2B-4T`）与扩散语言模型 `nemotron_labs_diffusion`。

### 2.2 `image-text-to-text` 月下载 Top-200（≤8.5B，共 44 个模型）

| `model_type` | 模型数 | 月下载量 | 代表模型（参数量） |
|---|---:|---:|---|
| **`qwen3_5`** | 12 | 16,607,471 | `Qwen/Qwen3.5-2B` (2.27B, 511 万 dl) |
| `qwen2_5_vl` | 6 | 9,427,411 | `Qwen/Qwen2.5-VL-7B-Instruct` (8.29B) |
| `qwen3_vl` | 3 | 4,060,635 | `Qwen/Qwen3-VL-4B-Instruct` (4.44B) |
| `gemma3` | 3 | 2,445,434 | `google/gemma-3-4b-it` (4.30B) |
| `florence2` | 2 | 23,159,408 | `microsoft/Florence-2-base` (0.23B，视觉/OCR) |
| `qwen2_vl` | 2 | 1,430,459 | `Qwen/Qwen2-VL-2B-Instruct` (2.21B) |
| 专用 OCR / 文档 VLM | 8+ | >12,000,000 | `Unlimited-OCR`、`DeepSeek-OCR`、`LightOnOCR-2-1B`、`HunyuanOCR`、`granite-docling-258M` |
| `smolvlm` / `moondream1` / `internvl_chat` / `minicpmv4_7` / `gemma4` | 各 1 | 30 万 – 239 万 | `SmolVLM2-500M`、`moondream2`、`MiniCPM-V-4.7-1B`、`gemma-4-E4B-it` |

---

## 3. 六大架构谱系深度拆解

### 3.1 第一类：标准稠密 GQA Transformer 家族
核心骨架统一为 **Pre-RMSNorm + GQA + RoPE + SwiGLU**，各家差异集中在少量投影与缩放细节：

| `model_type` | 代表模型 | 关键结构差异（相对标准 `llama`） |
|---|---|---|
| `llama` | Llama 3.1 8B / 3.2 1B & 3B、SmolLM2 | 基线实现；3.2 1B/3B 绑定词嵌入（tied embeddings） |
| `qwen2` | Qwen2 / 2.5、DeepSeek-R1-Distill-Qwen | `attn_{q,k,v}.bias`（及部分模型 `output.bias`） |
| `qwen3` | Qwen3 0.6B / 1.7B / 4B / 8B | 去除 QKV bias，新增 **QK-Norm**（逐 head RMSNorm） |
| `phi3` | Phi-3 / 3.5-mini、**Phi-4-mini (3.8B)** | 融合 `qkv_proj` 与 `gate_up_proj`；Phi-4-mini 实测 `H=3072, L=32, 24Q/8KV, FFN=8192, V=200064`，LongRoPE |
| `mistral` / `ministral3` | Mistral 7B、Ministral 3 3B / 8B | 滑动窗口 SWA；Ministral 3 原生多模态 + 256K 上下文 |
| `granite` | `ibm-granite/granite-4.1-3b` (3.4B) | 实测 `H=2560, L=40, 40Q/8KV, FFN=8192, V=100352`；muP 缩放系数（embedding / residual / attention / logits） |
| `exaone` | `EXAONE-3.5-7.8B-Instruct` | 与 `llama` 高度同构，层归一化/RoPE 细节微调 |

### 3.2 第二类：稠密 Transformer 的长上下文与稳定性变体
为缓解长上下文 KV Cache 膨胀与训练发散，衍生出 NoPE、SWA、Post-Norm、PLE 等结构：

- **`smollm3`（`HuggingFaceTB/SmolLM3-3B`）**
  - 实测配置：`H=2048, L=36, 16Q/4KV, FFN=11008, V=128256`，`layer_types` 全为 `full_attention`。
  - 核心特征：**3:1 RoPE / NoPE 交替**（`no_rope_layers` 中 27 层启用 RoPE、9 层禁用位置编码），配合 YaRN 扩展至 128K。
- **`olmo2` / `olmo3`（`allenai/Olmo-3-7B-Instruct`）**
  - 实测配置：`H=4096, L=32, 32Q/32KV (MHA), FFN=11008, V=100278`。
  - 核心特征：**Post-Norm + QK-Norm**；`olmo3` 新增 **3:1 滑动窗口与全局注意力混合**（24 层 `sliding_attention, w=4096` + 8 层 `full_attention`）。
- **`gemma2` / `gemma3` / `gemma3n` / `gemma4`**
  - `gemma2`：1:1 SWA/Global 交替、GeGLU、Attention 与 Final Logit Softcap、4×RMSNorm（Pre + Post）。
  - `gemma3` / `gemma3_text`（1B / 4B）：5:1 局部/全局注意力、QK-Norm、局部与全局使用不同 RoPE base。
  - `gemma3n` / `gemma4`（E2B / E4B）：**PLE（Per-Layer Embeddings 逐层嵌入）+ 跨层共享 KV + MatFormer 弹性子网**，将 5B–8B 总参数压缩到 2B–4B 的运行时内存占用。

### 3.3 第三类：混合注意力 / SSM / 短卷积架构（2025–2026 核心增量）
核心思想是用固定大小循环状态（$O(1)$ 显存）替代大部分 Softmax Attention，显著压缩长上下文 KV Cache：

| `model_type` | 代表模型（实测配置） | 混合方式与状态算子 |
|---|---|---|
| **`qwen3_5`** | Qwen3.5 0.8B / 2B / 4B / 9B | **顺序混合（3:1）**：Gated DeltaNet（GDN 线性注意力 + Conv1d）+ Full Attention |
| **`lfm2`** | LiquidAI LFM2 / LFM2.5 (230M–2.6B) | **顺序混合**：门控 `ShortConv` + Full Attention（QK-Norm） |
| **`lfm3`** | `LiquidAI/LFM3-1.7B` (`H=2048, L=24, 16Q/4KV, FFN=7680`) | **三混合**：`13` 层 `conv` + `5` 层 `sliding_attention (w=256)` + `6` 层 `full_attention` |
| **`nemotron_h`** | `NVIDIA-Nemotron-3-Nano-4B` (`H=3136, L=42, 40Q/8KV`) / `Nemotron-H-8B` | **顺序混合**：`hybrid_override_pattern` 显式编排 Mamba-2 (`M`)、MLP (`-`) 与极少数 Attention (`*`)，KV Cache 削减 ~90% |
| **`falcon_h1`** | `tiiuae/Falcon-H1-3B-Instruct` (`H=2560, L=32, 10Q/2KV, mamba_n_heads=32`) | **并行混合**：同一层内并行执行 Mamba-2 SSM 头与 GQA Attention 头，输出融合 |
| **`granitemoehybrid`** | IBM Granite-4.0-H-Tiny (7B-A1B) / `antares-1b` | **顺序混合 + MoE**：9:1 Mamba-2 + Attention，FFN 替换为细粒度稀疏 MoE |
| **`minicpm_sala`** | `openbmb/MiniCPM-SALA` (`H=4096, L=32, 32Q/2KV, FFN=16384`) | **稀疏 + 线性混合（SALA）**：结合稀疏注意力与线性注意力 |
| `zamba2` / `rwkv7` | Zamba2 1.2B / 2.7B、RWKV-7 | Zamba2 为 Mamba-2 + 跨层共享 Attention；RWKV-7 为纯广义线性状态 RNN |

### 3.4 第四类：小型稀疏 MoE（总参 ≤8B，激活参 0.8B–2.2B）
- **`olmoe`（`allenai/OLMoE-1B-7B`）**：64 专家选 8，总参 6.92B / 激活 1B，全开源训练配方。
- **`bailing_hybrid`（`inclusionAI/Ling-3.0-tiny`，7.89B）**：实测 `H=1536, L=24, 16Q/16KV, FFN=4608, V=157184`，**128 专家选 8** 细粒度 MoE + 混合注意力。
- **`granitemoe`（`ibm-granite/granite-3.1-3b-a800m`）**：3.3B 总参 / 0.8B 激活。
- **`jetmoe`（`jetmoe/jetmoe-8b`）**：8B 总参 / 2.2B 激活，结合 MoA（注意力头专家）+ MoE。
- 其他社区新秀：`olala`（`Olala-v1.0-mini-7A1B`）、`aurora2`（`400M-A20M`）。

### 3.5 第五类：原生极低比特与非自回归范式
- **`bitnet`（`microsoft/bitnet-b1.58-2B-4T`）**：原生三值权重 $\{-1, 0, 1\}$ `BitLinear` + SubLN + ReLU²，2B 参数在 4T token 上从头训练。
- **`nemotron_labs_diffusion`（`nvidia/Nemotron-Labs-Diffusion-3B`）**：离散扩散语言模型（dLLM），通过多步去噪实现块级并行生成，2026 年首次进入 HF ≤8B 热门榜。

### 3.6 第六类：历史基线与专用视觉/OCR 架构
- 历史纯文本基线：`gpt2`、`gpt_neox`（Pythia）、`bloom`、`t5` / `t5gemma`（下载量高但多用于教学、评测和扩散模型文本编码器，不作为新端侧选型）。
- 端侧 OCR / 文档理解小模型：`florence2`、`deepseek_vl_v2`（DeepSeek-OCR）、`unlimited_ocr`、`mistral3`（LightOnOCR-2-1B）、`hunyuan_vl`（HunyuanOCR）、`idefics3`（granite-docling-258M），参数集中在 **0.25B–3.3B**。

---

## 4. `tinymodel` 覆盖度评估与后续扩展路线

### 4.1 当前覆盖度
`crates/tm-models/src/` 已实现 `llama`、`qwen2`、`qwen3`、`smollm3`、`phi3`（含 Phi-3.5 / Phi-4-mini）、`mistral` / `mistral3`、`olmo2` / `olmo3`、`granite`、`exaone`、`olmoe`、`granitemoe`、`qwen2moe`、`qwen3moe`、`bailingmoe2`、`qwen35`、`gemma2`、`gemma3`、`gemma4`、`lfm2`、`lfm3`、`mamba2`、`nemotron_h`、`falcon-h1`、`granitehybrid` / `granitemoehybrid`、`inkling2`。按 Hugging Face Top-1,000 实测数据：
- **存量纯文本覆盖率**：在完成 P0 + P1 + P2a 扩展后，已覆盖现代 ≤8B 文本与稀疏 MoE 模型下载量的 **~99%+**（剔除 `gpt2`/`gpt_neox` 历史教学模型）。
- **新一代混合与稀疏范式全覆盖**：已完整覆盖 **通用细粒度稀疏 MoE（`SparseMoeFfn`：`olmoe`、`granitemoe`、`granitemoehybrid`、`qwen2moe`、`qwen3moe`、`bailingmoe2`）**、**GDN 线性注意力（Qwen3.5）**、**Mamba-2 SSM 及三大混合拓扑（`mamba2` 纯 SSM / `nemotron_h` 异构顺序混合 + ReLU² / `falcon-h1` 并行 Attention+SSM / `granitehybrid` & `granitemoehybrid` 交错混合 + MoE + muP）**、**PLE + 共享 KV + SWA + 注意力软上限（Gemma 2 / 3 / 4）**、**门控 ShortConv + SWA/Full 混合注意力 + 逐头门控（LFM2 / LFM3）** 及 **大型循环稀疏 MoE（Inkling-2）**。

### 4.2 算子复用与扩展优先级建议

| 优先级 | 目标架构 | 所需新增机制 | 预估改动量 | 状态 / 收益 |
|---|---|---|---|---|
| **P0（零新算子，高 ROI）** | `smollm3`、`mistral`/`mistral3`、`phi3`（含 Phi-4-mini） | 在 `llama.rs` 增加 `no_rope_layers` 掩码、SWA 窗口、融合 QKV/GateUp 零拷贝切分与 LongRoPE | 低（复用现有 CPU/Metal/Vulkan 内核） | ✅ **已完成（2026-11-24）**：补齐 SmolLM3、Phi-3/Phi-4-mini、Mistral/Ministral 全栈支持（含融合张量切片 LoRA 与 `LlamaSpm` 分词器） |
| **P0（零新算子）** | `gemma3` / `gemma2` | 复用 `gemma4.rs` 的 SWA、QK-Norm、GeGLU、softcap 向下兼容（补充无 V-Norm 与注意力内部 `attn_logit_softcapping`） | 低 | ✅ **已完成（2026-11-24）**：覆盖 Gemma 2/3 1B–4B 社区生态（含 `sech²` 软上限反向梯度） |
| **P0（零新算子）** | `lfm3` | 在 `lfm2.rs` 补齐三路 `ShortConv` + SWA + Full 分发、双频/+inf 部分 RoPE、逐头 `attn_gate` 与三明治归一化 | 低 | ✅ **已完成（2026-11-24）**：支持 Liquid AI 最新 `LFM3-1.7B`（CPU + Metal + QLoRA 全栈） |
| **P1（小改动）** | `olmo2` / `olmo3`、`granite` | 纯 Post-Norm + 全向量 `[q_dim]`/`[kv_dim]` QK-Norm + 3:1 双频 SWA（OLMo 2/3）；4 项 muP 缩放系数 + MLP bias + LongRoPE（Granite） | 低 | ✅ **已完成（2026-11-24）**：覆盖 AI2 OLMo 2/3 与 IBM Granite 3.x/4.0 稠密全栈（CPU + Metal + QLoRA） |
| **P1（战略新算子）** | **Mamba-2 SSM**（`mamba2`、`nemotron_h`、`falcon-h1`、`granitehybrid`） | 新增 `tm-cpu` Mamba-2 离散状态扫描三件套 + 因果 Conv1d + 分组 `ssm_norm` + `LayerState::Ssm` 前缀缓存/投机回滚 + Metal `Mamba2MetalRunner` + `Mamba2Trainer` 8 步边界重算 BPTT | 中（对应 `ROADMAP.md` M6） | ✅ **已完成（2026-11-24）**：**一箭四雕**全栈解锁 `mamba2`、`nemotron_h`、`falcon-h1`、`granitehybrid`（见 `ADR-017`） |
| **P2a（通用化）** | **通用细粒度稀疏 MoE + `exaone`**（`olmoe`、`granitemoe`、`granitemoehybrid`、`qwen2moe`、`qwen3moe`、`bailingmoe2`、`exaone`） | 公共 `SparseMoeFfn`（零拷贝 3D 分离/融合专家、Softmax/Sigmoid/`exp_probs_b`/`norm_weights`/共享专家/1D `ffn_gate_inp_shexp`）+ Apple Metal 单 CB `GpuMoeEngine` + `moe_qlora` 解析反向 | 中（对应 `ROADMAP.md` M6） | ✅ **已完成（2026-11-24）**：一次贯通 Attention 与 Mamba-2 双骨干 6 种稀疏 MoE 架构及 `exaone`（见 `ADR-018`） |
| **P2（多模态，M6）** | 视觉编码器（SigLIP / Qwen-VL ViT / Pixtral / SmolVLM） | 图像预处理 + ViT 编码器 + Projector + 阶段加载 `prefill_multimodal` | 中高 | ✅ **已完成（2026-11-25）**：零外部依赖全栈落地 `VisionModel`（CPU + Apple Metal `forward_embed` + CLI `--mmproj`/`--image`），释放 `Qwen3.5`、`Gemma 3/4`、`Qwen2.5/3-VL`、`Ministral 3`、`SmolVLM`、`LFM2-VL` 原生多模态能力（见 `ADR-019`） |
