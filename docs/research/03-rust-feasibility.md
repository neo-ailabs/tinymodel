# 用 Rust 编写端侧推理引擎的可行性

> [!NOTE]
> 第 2 节的指令支持情况是 **2026-11-19 在本机（M4，rustc 1.99.1 stable）实际编译运行的结果**，测试源码在 `docs/research/poc/`。其余内容来自网络检索。

## 0. 结论

**可行，推荐用 Rust。** 有两个前提：
1. CPU 内核层要接受 `unsafe` + `std::arch` intrinsics + 少量 `asm!` 内联汇编；
2. GPU 内核本来就要用 MSL、CUDA、GLSL 编写，和宿主语言用 Rust 还是 C++ 没有关系。

| 维度 | 评估 | 说明 |
|---|---|---|
| CPU 性能上限 | ✅ 与 C++ 相当 | 两者都用 LLVM 生成代码；热点循环写成 `unsafe` 后，生成的机器码基本一样 |
| ARM 指令支持 | 🟡 基本可用 | NEON 和 dotprod 已稳定；i8mm、FP16、SVE/SME 还不稳定，**用 `asm!` 在 stable 上可以绕过**（已实测） |
| x86 指令支持 | ✅ | AVX2、FMA、AVX-512（1.89 起稳定）；AMX 需要用内联汇编 |
| GPU | ✅ 可用 | Metal 用 `objc2-metal`，CUDA 用 `cudarc`，Vulkan 用 `ash`；只是周边生态比 C++ 薄一些 |
| 交叉编译到树莓派 | ✅ 优势明显 | 一条 `cargo zigbuild` 或 `cross` 命令就能编译出 aarch64 musl 的静态二进制，比 C++ 配交叉工具链简单很多 |
| 体积（Zero 2W） | ✅ | 带 dotprod 测试代码的二进制约 470KB；核心引擎预计 1–2MB |
| 工程层面 | ✅ 优势明显 | 内存安全；mmap 的权重可以用生命周期借用，编译器保证不会悬空；`cargo` 的 feature 正好对应三档配置；服务、Python 绑定、C ABI 都有现成的库 |
| 参考资料 | 🟡 | 一流的 CPU 内核（ggml、KleidiAI、XNNPACK）都是 C 或汇编，需要自己移植 |

---

## 1. 现有 Rust 推理项目（说明这条路走得通）

| 项目 | 定位 | 参考价值 |
|---|---|---|
| **candle**（Hugging Face） | 张量框架，支持 CPU、CUDA、Metal | 可以参考 GGUF 解析和 k-quant 内核的移植方式；动态张量框架不适合 Zero 2W |
| **mistral.rs** | 基于 candle 的推理服务 | 有分页注意力、连续批处理、投机解码；据称 CPU 和 CUDA 的生成速度已经接近 llama.cpp |
| **burn + CubeCL** | 用 Rust 写 GPU 内核，可以编译到 CUDA、ROCm、Metal、Vulkan、WebGPU | 能用一份代码覆盖多种 GPU，但量化矩阵乘的性能还需要验证 |
| ratchet、kalosm 等 | WebGPU 或应用层 | 参考价值有限 |

**结论**：这些项目证明 Rust 在主机档能做到接近 llama.cpp。但**没有一个是针对树莓派和 Zero 2W 深度优化的**，这正好是可以做出差异化的地方。

---

## 2. SIMD 支持情况（本机 stable 1.99.1 实测）

| 能力 | 用到它的设备 | 在 stable 上的状态 | 做法 |
|---|---|---|---|
| NEON 基础指令 | 所有 ARM 设备 | ✅ 稳定 | `std::arch::aarch64` |
| **dotprod**（`vdotq_s32`） | 树莓派 5、手机、M 系列 | ✅ **稳定，实测通过** | 直接用 intrinsics |
| **i8mm**（`vmmlaq_s32`） | M4、新款手机、Graviton | ❌ 不稳定（需要 `stdarch_neon_i8mm`） | **用 `asm!("smmla ...")` 在 stable 上实测通过** |
| FP16 运算（`vfmaq_f16`） | A76+、M 系列 | ❌ 不稳定（需要 `stdarch_neon_f16`） | 用 `asm!`，或者先转成 FP32 计算；数据类型用 `half` crate |
| 运行时检测 dotprod、i8mm | 全部 | ✅ 稳定（`is_aarch64_feature_detected!`） | 启动时选出最优的函数表 |
| 运行时检测 SME | M4 | ❌ 不稳定 | 在 macOS 上读 `sysctl hw.optional.arm.FEAT_SME` |
| SVE / SME intrinsics | Graviton、M4 | ❌ 只能用 nightly | 用 `asm!`，或者通过 FFI 调用 KleidiAI（C 库，Apache-2.0）的 SME2 内核 |
| AVX2 / FMA / AVX-512 / VNNI | x86 | ✅ 稳定 | `std::arch::x86_64` |
| AMX | 至强、新款酷睿 | ⚠️ 未稳定 | 用 `asm!` |
| `#[target_feature]` 加在安全函数上 | 全部 | ✅（1.86 起） | 一份代码编译成多个版本 |
| `std::simd`（可移植 SIMD） | — | ❌ 只能用 nightly | **不依赖它**；自己写一层很薄的抽象，或者用 `pulp` / `multiversion` |

> [!TIP]
> 建议**整个项目固定使用 stable**，不稳定的指令统一放进 `kernels::asm` 模块用 `asm!` 实现。数量不多（smmla、fp16 fma、SME 的若干条指令），也方便以后 intrinsics 稳定了再一次性替换。

---

## 3. GPU 后端选型

| 后端 | Rust 库 | 内核用什么写 | 备注 |
|---|---|---|---|
| **Metal** | `objc2-metal`（`metal-rs` 已经弃用） | MSL，编译时生成 metallib | 和 llama.cpp 的做法一样；M4 上优先做 |
| **CUDA** | `cudarc` | CUDA C，通过 `build.rs` 调用 nvcc 编译成 PTX 或 cubin | candle 和 mistral.rs 都在用，很成熟 |
| **Vulkan** | `ash` | GLSL 编译成 SPIR-V（构建时用 glslc 或 shaderc） | 覆盖 AMD、Intel 显卡、Mali 和树莓派 5 的 VideoCore；要用上 subgroup 和 cooperative matrix 扩展 |
| 备选：CubeCL | `cubecl` | 用 Rust 宏编写内核 | 一份代码覆盖多种 GPU；建议先用它的矩阵乘做对比实验，**不要一开始就绑定它** |
| 不推荐：wgpu | — | WGSL | 调度开销大，量化内核的性能受限；只在需要支持浏览器时考虑 |

---

## 4. 和 C++ 对比，对这个项目的影响

| 方面 | Rust | C++ |
|---|---|---|
| 内核性能 | 一样（都是 LLVM）；热点循环需要 `get_unchecked` 或指针来去掉边界检查 | 一样 |
| 新指令扩展的支持 | 落后几个版本，需要用 `asm!` 补 | 一般第一时间可用 |
| 交叉编译和发布 | **`cargo` + zig 或 musl 一条命令搞定** | CMake + 工具链文件，比较繁琐 |
| 内存和并发安全 | **编译器保证**（会话、缓存、多线程服务都受益） | 靠代码规范和 sanitizer |
| 可选功能裁剪 | **cargo feature 天然就能做三档配置** | CMake 的 option 加宏 |
| 生态 | `memmap2`、`half`、`minijinja`（聊天模板）、`axum`、`pyo3`、`uniffi`、`cbindgen` | 更多 C 库可以直接用 |
| 编译时间 | 偏长（泛型展开） | 偏长 |

---

## 5. 实施策略

| 方案 | 做法 | 评估 |
|---|---|---|
| **A. 纯 Rust（推荐）** | 运行时、模型层、CPU 内核都用 Rust 写（intrinsics + 少量 `asm!`）；GPU 内核用原生着色器语言 | 可控性最好，交叉编译最简单，Zero 2W 上体积最小 |
| B. Rust + C 内核 | Rust 写运行时；CPU 内核用 `cc` crate 编译 ggml（MIT）和 KleidiAI（Apache-2.0）的 C 代码 | 起步最快，可以作为 A 的**逃生通道**（比如 M4 的 SME2 先接 KleidiAI） |
| C. 基于 candle | 在 candle 的张量框架上写模型 | 上手快，但内存会动态分配，层也被框架绑死，不适合 micro 和 edge 档，**不推荐** |

---

## 6. Cargo workspace 结构草案

```text
tinymodel/
├─ crates/
│  ├─ tm-core        # dtype、量化块格式、张量视图、内存池（无依赖）
│  ├─ tm-gguf        # GGUF 解析、mmap、离线或加载时重排权重
│  ├─ tm-cpu         # aarch64/{neon,dotprod,i8mm(asm),sme(asm|ffi)}、x86_64/{avx2,avx512}、generic；运行时选择
│  ├─ tm-metal       # objc2-metal + .metal 着色器      [feature = "metal"]
│  ├─ tm-cuda        # cudarc + .cu 内核                [feature = "cuda"]
│  ├─ tm-vulkan      # ash + .comp 着色器               [feature = "vulkan"]
│  ├─ tm-models      # llama、qwen3、qwen35（DeltaNet）、gemma4（PLE）、lfm2
│  ├─ tm-runtime     # LayerState、会话、采样、投机解码与 MTP、前缀缓存
│  ├─ tm-tokenizer   # 轻量 BPE 和 SentencePiece（不依赖 HF tokenizers，减小 micro 档体积）
│  ├─ tm-cli / tm-server (axum, OpenAI 兼容) / tm-ffi (C ABI) / tm-py (pyo3)
└─ Cargo.toml        # 配置：micro = [cpu-armv8]、edge = [cpu, vulkan?]、host = [cpu, metal, cuda, vulkan, server]
```

**发布配置**：`panic = "abort"`、`lto = "fat"`、`codegen-units = 1`、`strip = true`。不同目标用不同的 `-C target-cpu`：Zero 2W 用 `cortex-a53`，树莓派 5 用 `cortex-a76`，主机用通用基线再加运行时选择。

---

## 7. 技术验证（PoC）计划（1–2 周，验证决定成败的几个点）

| # | 验证内容 | 通过标准 |
|---|---|---|
| 1 | 用 Rust 写 **Q4_0 × Q8_0 的矩阵向量乘**（NEON dotprod 版本 + i8mm `asm!` 版本），和 ggml 对比 | 在 M4 上单线程和多线程都达到 ggml 的 95% 以上 |
| 2 | 交叉编译到 `aarch64-unknown-linux-musl`，在树莓派 5 和 Zero 2W 上运行 | 静态二进制小于 2MB；树莓派 5 上的 GEMV 带宽利用率达到 80% 以上 |
| 3 | 用 `objc2-metal` 写量化矩阵向量乘，和 llama.cpp 的 Metal 后端对比 | 达到 90% 以上 |
| 4 | Llama 3.2 1B 端到端贪心解码 | 输出和 llama.cpp 逐 token 一致 |

如果第 1、2 项通过，就可以确定走纯 Rust 路线；如果第 1 项明显落后，就先用方案 B 接入 C 内核，后面再逐步替换。

## 8. 风险

- **ARM 新指令扩展的 intrinsics 不稳定**：用 `asm!` 解决，影响范围小。
- **要自己移植内核**：ggml 和 KleidiAI 的许可证（MIT、Apache-2.0）允许参考和移植，记得保留版权声明。
- **CubeCL 等新生态还不成熟**：只作为备选，不绑定。
- **团队对 Rust 和 `unsafe` 的熟悉程度**：把 `unsafe` 集中在 `tm-cpu` 和各 GPU crate 里，上层保持安全代码。
