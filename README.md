# tinymodel

用 **Rust stable** 编写的通用端侧与全场景 LLM 推理及微调引擎，一套代码覆盖 **512 MB Raspberry Pi Zero 2W**、**手机与 ARM 开发板** 到 **Apple Silicon / AMD / Intel / NVIDIA Blackwell GPU 主机**。

> 当前状态：**M1–M4、T1–T4 已完成 ✅；M5 CUDA 后端及两轮 P0–P2 深度优化已完成 ✅；M6 多模态视觉、Mamba-2 与全部四大前沿稀疏 MoE 家族已落地 ✅**（详见 [docs/ROADMAP.md](docs/ROADMAP.md) 与 [docs/DECISIONS.md](docs/DECISIONS.md)）。

## 1. 三档硬件裁剪（Profiles）

| Profile | 目标设备 | 模型规模 | 计算后端与核心技术 |
|---|---|---|---|
| `micro` | Pi Zero 2W / Pi 3 / Pi 4B | ≤0.5B | 纯 ARMv8.0 NEON（`0` 条 `sdot`/`smmla`）、16 行 T-MAC `vqtbl1q` 查表内核、离线 BPE 词表裁剪、**`1.18 MiB`** 纯静态 `musl` ELF |
| `edge` | Google Pixel 8、Raspberry Pi 5、Rockchip RK3588 | ≤5B（32GB 板支持 35B MoE） | NEON `dotprod` / `i8mm` `asm!`、异构大核亲和调度、预编译 SPIR-V 1.2 Vulkan（Mali / VideoCore VII V3DV）、零膨胀 `Q4_0x4` 与 `Q8_0` KV |
| `host` | Apple Silicon、AMD / Intel / NVIDIA 独显主机 | ≤9B 稠密 & **26B–120B 稀疏 MoE** | `AVX2`、**Metal**（`simdgroup_float8x8`）、**Vulkan CoopMat**、**CUDA**（动态 `libcuda`、双 PTX `compute_75`/`80`、`WMMA64` Tensor Core、`CudaGraphExec` 图捕获、零拷贝 DDR 专家卸载 + 64 槽 VRAM 热专家缓存） |

## 2. 一等支持的模型架构与微调能力

- **主流稠密架构**：`Llama`、`Qwen2 / Qwen2.5 / Qwen3`、`SmolLM2 / SmolLM3`、`Phi-3 / Phi-4-mini`、`Mistral / Ministral`、`OLMo 2 / 3`、`Granite`、`EXAONE`。
- **新一代混合线性/状态空间/边缘架构**：
  - **`Qwen3.5`（Gated DeltaNet 3:1 混合）**：严格遵循 `GDN_CONTRACT`，支持递归检查点前缀缓存与 0-ULP 寄存器驻留批扫描；
  - **`Gemma 2 / 3 / 4`（PLE + 共享 KV + SWA）**：262k 词表 PLE 原位按需单行 `mmap`（Pixel 8 跑 `E2B` 总 RSS 仅 `1.47 GiB`）；
  - **`LFM2 / LFM3`（门控短卷积混合）** 与 **`Mamba-2`（`mamba2` / `nemotron_h` / `falcon-h1` / `granitehybrid`）**。
- **细粒度与超大规模稀疏 MoE（含 `Strata` 全部四大家族）**：
  - 中小型 MoE：`OLMoE`、`GraniteMoE`、`Qwen2MoE`、`Qwen3MoE`、`BailingMoE2`；
  - **`Inkling-2-Mini`（26B 循环 MoE + `DSpark` 分块投机）**：支持全显存 CUDA/Vulkan 及 16GB RAM + 8GB GPU 三层 `O_DIRECT` 零 Swap 调度；
  - **`Qwen3.6-35B-A3B`（`qwen35moe` + 内置 `nextn` MTP）**：支持 `verify_graphs` 多步图捕获与 `TM_CUDA_MOE_HOST_OFFLOAD` + `TM_CUDA_MOE_CACHE_EXPERTS` 热专家缓存；
  - **`Qwen3.8-Flash-Next`（`qwen4exp` 双分片直读 + 4 流 HC + QSA + 外部 `qwen4exp-mtp`）**：零 `.pack` 磁盘膨胀；
  - **`Tencent Hy4-Flash`（`.strata` v1 容器直读 + 28.4× 矩阵吸收式 MLA + 5 层 `DFly2`）**。
- **多模态视觉（`--mmproj` / `--image`）**：兼容 `llama.cpp` `mmproj` GGUF（`SigLIP`、`Qwen2.5/3-VL` 3D 卷积、`Pixtral`、`SmolVLM`），阶段加载编码完即释放视觉权重。
- **端侧微调（`tm-lora` & `tm-train`）**：兼容 `llama.cpp --lora` 动态挂载与无损 `lora-merge`；100% safe Rust 手写逐层激活重算 **QLoRA**（覆盖稠密、GDN、PLE、ShortConv、Mamba-2 与稀疏 MoE）及 `train-mtp` 草稿头训练，支持 Metal / Vulkan 反向加速。

## 3. 真机实测亮点

| 硬件平台 | 代表模型与配置 | 实测指标与同机基线对比 |
|---|---|---|
| **NVIDIA RTX PRO 6000 Blackwell** (`--cuda`) | `Qwen3.6-35B-A3B Q8_0` (`+内置 MTP`) | `pp512 = 2,579.8 tok/s`，**代码 `239.39 tok/s` / 散文 `238.79 tok/s`**；10 层卸载+64 槽缓存仅占 **`30.6 GiB` 显存达 `109.61 tok/s`** |
| **NVIDIA RTX PRO 6000 Blackwell** (`--cuda`) | `Inkling-2-Mini Q8_0` (`+DSpark`) / `Qwen3.8 Q2_0` | `Inkling-2` 代码 **`114.56 tok/s`**（反超 `Strata CUDA` `+21.2%`）；`Qwen3.8` **`pp512 = 1,133.5 tok/s`**（反超 `Strata` 且省 `44.9 GiB` 磁盘） |
| **AMD RX 9060 XT** (`--vulkan`) | `SmolLM2-360M` / `Qwen3.5-0.8B` (`Q4_0`) | Decode **`608.8` / `366.5 tok/s`**（反超同机 ROCm 7.1.1 HIP **`225.9%` / `217.8%`**） |
| **Apple M4 16GB** (`--metal` & CPU) | `SmolLM2-360M` / `Qwen3.5-0.8B` / `Gemma 4 E2B` | CPU `270.5 tok/s`（`106.3%` `llama.cpp`）；Metal `129.0` / `52.3 tok/s`（`99.5%` / `95.0%` `llama.cpp Metal`） |
| **Google Pixel 8** (`Tensor G3`) | `Llama-3.2-1B` / `SmolLM2` / `Gemma 4 E2B` | `Q4_0x4` Prefill **`155.5%`** `llama.cpp --repack 0`（省 `810 MiB`）；`E2B` 总 RSS **`1.47 GiB`**、脏内存仅 `55.6 MiB` |
| **4× Cortex-A510** (`micro` 档真机) | `SmolLM2-360M Q2_0`（词表裁剪后 `93.6 MiB`） | **`20.92 tok/s`**，峰值 RSS `96–200 MiB << 350 MB`（折算 Pi Zero 2W 稳达 **`8.3–9.8 tok/s`**） |

## 4. 快速开始

```sh
# 1. 构建（按硬件选择后端：cuda / vulkan / metal / 默认 CPU）
cargo build --release --features "vulkan,cuda"

# 2. 推理生成（开启自适应神经投机解码）
./target/release/tinymodel run -m model.gguf -p "Explain Rust zero-cost abstractions:" -n 128 --cuda --spec auto

# 3. 32GB 级消费显卡运行 35B Q8_0（开启 10 层零拷贝卸载 + 64 槽 VRAM 热专家缓存）
TM_CUDA_MOE_HOST_OFFLOAD=10 TM_CUDA_MOE_CACHE_EXPERTS=64 \
  ./target/release/tinymodel bench -m Qwen3.6-35B-A3B-Q8_0.gguf -c 4096 -n 128 --cuda

# 4. 交叉编译 Pi Zero 2W 纯静态 micro 二进制（< 1.3 MiB）
cargo zigbuild --release --target aarch64-unknown-linux-musl --no-default-features --features micro

# 5. 本地预览项目官网（零依赖静态站）
python3 -m http.server 8000 --directory website
```

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | **开发者和 AI agent 必读**：硬性规则、约定、测试门禁与已知陷阱 |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | 需求规格（硬件三档、模型结构、功能与非功能指标） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计（Crate 分层、`LayerState`、CPU/Metal/Vulkan/CUDA 内核与投机验证） |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 里程碑路线图、验收标准与实测进度 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 架构决策记录（`ADR-001` 至 `ADR-026`） |
| [docs/SETTINGS.md](docs/SETTINGS.md) | 运行时环境变量登记表与默认值说明 |
| [docs/TESTING.md](docs/TESTING.md) | L0–L5 正确性分层、多驱动 Vulkan / CUDA 验证与基准协议 |
| [docs/research/](docs/research/) | 硬件档案、模型谱系与 Blackwell / RDNA4 / Pixel 8 / M4 实测报告 |
| [website/README.md](website/README.md) | 零依赖静态项目官网预览与云端部署指南 |

## 6. 参考代码与许可

`Strata/` 是只读第三方参考目录（MIT，`github.com/Niko1221/Strata`），不参与构建。移植自 `ggml`、`Strata` 或 `KleidiAI` 的代码在文件头保留原版权声明并汇总于 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本项目采用 **Apache License 2.0**（[LICENSE-APACHE](LICENSE-APACHE)）或 **MIT License**（[LICENSE-MIT](LICENSE-MIT)）双许可发布，由使用者任选。
