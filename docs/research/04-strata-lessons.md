# Strata 借鉴清单（面向 Rust 端侧引擎）

> [!NOTE]
> Strata 位于 `tinymodel/Strata`，**MIT 许可**，约 30 万行代码，面向“把 35B 以上的 MoE 模型塞进桌面显卡”。本文由三个只读调研助手并行分析后汇总，下文路径都相对于 `Strata/`。
>
> **总体判断**：可借鉴的地方很多，但要挑着拿。最有价值的是 **状态管理和投机解码**、**CPU 量化层和 Metal GEMV**、**测试与对齐方法**。专家缓存、多卡、NVFP4 等约一半以上的代码对 ≤8B 稠密模型没有用处。

## 结论速览

| 优先级 | 借鉴项 | 位置 | 对我们的价值 |
|---|---|---|---|
| ⭐⭐⭐ | GDN 投机验证“只读状态、提交时重放” | `src/qwen35/q36_engine.cu`、`src/core/verify.hpp` | 直接决定 `LayerState` 的设计，不用快照，省内存 |
| ⭐⭐⭐ | CPU 量化层（ggml quants.c 的 MIT 移植）+ 按 ISA 分档的内核表 | `src/quant/`、`registry.hpp`、`dispatch.cpp`、`isa.cpp` | 可以直接作为 Rust CPU 内核的移植蓝本 |
| ⭐⭐⭐ | 逐位对齐测试体系（scalar 参考实现 + oracle + 启动自检） | `tests/quant/run_oracle.sh`、`*_parity.cpp`、`tools/canonical_xcheck.py` | 保证正确性的基础设施 |
| ⭐⭐⭐ | Metal 量化 GEMV（unit 抽象，一份代码覆盖所有格式） | `metal/kernels/gemv.metal`、`quant.metal`、`units.metal` | M4 后端几乎可以照着移植 |
| ⭐⭐ | 混合模型前缀缓存 = token 序列 + 定点状态检查点 | `conversation_cache.hpp`、`docs/MESSAGE_BOUNDARY_CACHE.md` | 多轮对话和 Agent 场景必须有 |
| ⭐⭐ | 按收益选择投机策略 + suffix drafter | `src/spec/controller.hpp`、`draft_policy.hpp`、`suffix_drafter.hpp` | 树莓派上不需要额外权重就能投机 |
| ⭐⭐ | 线程池设计 | `src/kernels/cpu/pool.cpp` | 避开死锁和 SMT 两个坑 |
| ⭐⭐ | 工具调用解析的语料加切片测试 | `serve/test_tool_corpus.py`、`tool_parser_golden.json` | 服务层最大的 bug 来源 |
| ⭐ | PLE 异步按行读取 | `include/strata/ngram/ple_reader.hpp` | Gemma 4 在小内存设备上能跑起来的关键 |
| ⭐ | 纯函数内存规划器、cgroup 内存读取、分包签名 | `strata_control/vram_fit.py`、`cgroup_memory.py`、`packaging/ENGINE-COMPONENT.md` | 部署和分发 |

---

## 1. 运行时与状态管理

1. **每个模型一个独立驱动，没有通用计算图**：`q36_engine.cu` 用扁平的 `Layer`/`Model` 结构，`attn` 布尔值区分 GDN 层和注意力层，MTP 块当作第 n_layers 层，复用同一个 `layer()` 函数。**这和我们“每个模型手写 forward”的方向一致，可以确认可行。**
2. **加载时按“每个模型的张量表”逐层校验**（`include/strata/core/family_layout.hpp`、`src/core/layout.cpp`）：核对张量名、形状、dtype，遇到不认识的张量直接拒绝，报错写明层号。→ Rust 里可以用 `ModelSpec` 表配合 `Result` 错误类型来做。
3. **按张量自身的 dtype 分派内核，不按它的角色**（`layer.cpp:141-161`）：同一种张量在不同层可能用不同量化格式（混合量化）；**激活量化格式也要跟着权重类型走，选错了误差约 1%，而且输出看起来完全正常**。→ 每个 `Mat` 都带 dtype，matmul 按 dtype 选内核。
4. **内存预先规划，生成 token 时零分配**（`layer.hpp:57-95`）：scratch 各层共用，只有状态每层一份。教训：每个 token 都分配一次 pinned 内存，会导致整个设备同步。
5. ⭐ **GDN 状态在投机解码中的回滚**（`verify.hpp:13-18`）：
   - 验证窗口里只读递归状态、不写回，同时保存每行的 qkv/α/β 输入；
   - 接受 a 个 token 后，用 `commit(a+1)` **只对保留的行重放 conv 和递归**，不重算任何投影；
   - KV cache 不回滚：被拒绝的位置下次直接覆盖；
   - 窗口大小 T=1 时原地推进状态。
   → **`LayerState` 采用“只读验证 + 重放提交”，不做快照恢复**，在树莓派上省内存。
6. **混合模型的前缀缓存**：KV 可以按前缀截断复用；递归状态**只能在保存过检查点的位置恢复**。检查点放在这些位置：
   - system prompt 末尾（常驻不淘汰）；
   - 每个 assistant 轮次开始时；
   - 每 N 个 token；
   - 上一轮边界。

   另外两条规则：只在 prefill chunk 本来就走到的边界保存；内存有上限，按 LRU 淘汰。
7. **投机策略按收益选择**（`spec/controller.hpp`）：目标是最大化“期望提交的 token 数 ÷ 每步耗时”，在不投机、suffix 查找、MTP 三者之间选，接受率用 EMA 在线学习，收益不到约 5% 就关闭。`suffix_drafter` 用 trigram 哈希表，每个 token 约 20B 内存，不需要任何权重。
8. **PLE 异步读取**（`ple_reader.hpp`）：token id 一确定就发出读请求，和 embedding、第 0 层的计算并行，外加一个有上限的行缓存。

## 2. CPU 内核与量化

1. **`src/quant/` 是 ggml-cpu quants.c 用生成器做的 MIT 移植**（锁定版本，见 `UPSTREAM.md`），分为 generic / AVX / AVX2 / AVX-512 / NEON(dotprod) 几档，入口是 `x86/x86_dots.inc` 和 `arm/arm_dots.inc`。
   - 每档一张内核表（`registry.hpp`），`dispatch.cpp` 按 ISA 从宽到窄排序，选出 CPU 能跑的最好一档。
   - → **Rust 版直接照这个结构写**：`trait`/函数指针表 + 按 ISA 从宽到窄选择。
2. **运行时探测**（`isa.cpp`）可以直接照抄：
   - x86：cpuid + xgetbv（还要检查操作系统是否开启了 YMM/ZMM 寄存器）；
   - Linux ARM：getauxval（HWCAP_ASIMDDP、HWCAP2_I8MM）；
   - macOS：sysctl `hw.optional.arm.FEAT_*`；
   - 支持 `FORCE_ISA` 环境变量强制降档测试。
3. ⚠️ **空缺**：Strata 在没有 dotprod 时直接退回标量实现，**没有纯 ARMv8.0 NEON 这一档**。所以 Zero 2W（A53）和树莓派 4 的内核要我们自己写（用 `vmull`/`vpadal`，或者查表）。
4. **算术契约 + 逐位对齐**：每档 SIMD 都和标量版本逐位一致，标量版本和 ggml 对齐；启动时用 2 的幂作 scale 的随机块做自检，保证浮点运算精确。
5. **多 token 共享权重解码**（`kq_avx2.cpp`）：把 token 循环放进 block 循环里，权重的解包和 scale 每块只做一次，结果仍然和 ggml 逐位相同。→ 投机验证和小 batch 解码可以直接复用这个思路。
6. **线程池**（`pool.cpp`）：
   - 每层天然是一个 barrier，所以用扁平 batch；
   - 用 **epoch|njobs|index 的 64 位 CAS** 认领任务，防止晚醒的 worker 抢到下一批任务（死锁或重复执行，issue #29）；
   - 先自旋再睡眠；
   - **每个物理核一个 worker，避开 SMT 兄弟线程**（否则带宽减半）；
   - 识别 P/E 核，读取 cgroup 配额。
7. **小设备的低比特候选**：Q2_0（2.25 bpw，值为 (code−1)·d，接近三值），定义在 `strata-quant-common.h:191`。
8. **补齐 NEON 的 BF16→F32 向量化路径**：ggml 在 ARM 上缺这一项，导致 CPU 视觉编码器慢 100 倍（`docs/MAC.md:217`）。

## 3. Metal 后端（用于 M4）

1. **unit 抽象的 GEMV**：每种格式只需要描述 `fmt<F>{QK,BB,UNIT,TAB}` 和解码方式，GEMV 本体只写一次，覆盖 F32、BF16、Q8_0、Q4_0、K-quants、IQ 系列。
   - 模板参数：LPR（几个 lane 共享一行）、NR（每 lane 处理几行）、KS（K 维切分，按固定顺序归约，结果确定）；
   - 用 `function_constant NTOK=1..8` 针对 token 数特化，一个窗口内权重只读一次。
2. **技巧**：
   - 取字段时只用 AND 掩码、不移位，再把 x 预乘 2^-k 补偿；
   - 带 min 的格式预先算好每 16 个 x 的和；
   - IQ 码本先拷进 threadgroup 内存，因为 Apple GPU 上发散地访问 constant 内存会串行化。
3. **融合内核**：`gemv_pair`（gate+up+SiLU 一次完成）、`gemv_cat`（两个小矩阵合并成一次 dispatch）、`gemv_x`（x 暂存到 threadgroup 内存，**在 M4 上有收益，在 CUDA 上反而更慢**）。
4. **prefill**：`mma.metal` 用 simdgroup 矩阵乘；`dense_mm.metal` 先反量化成 half，再用 MetalPerformancePrimitives 做矩阵乘。
5. **运行时**（`mtl_rt.mm`、`gpu/runtime.cpp`）：
   - **mmap 的模型文件通过 `newBufferWithBytesNoCopy` 零拷贝映射**，超过 maxBufferLength 时切成重叠的块；
   - 所有 buffer 放进 `MTLResidencySet`；
   - 一个验证窗口用一个 command buffer，只在需要的地方加 barrier。

   → 用 `objc2-metal` 可以照搬这套。
6. **GDN 的 Metal 实现**：decode 是全程融合的单个内核，每个 head 用 512 线程，状态放在寄存器里；prefill 拆成 pre / scan / post 三遍。

## 4. 工程实践

1. **逐内核对齐程序**（`src/kernels/cpu/*_parity.cpp`）：不需要模型也不需要 GPU，用随机量化块覆盖所有码本索引；同一个程序加上 `--bench` 就是微基准。**另外要求批宽不变性：同一个 token 单独算和放进批里算，结果必须逐位相同。**
2. **先写 oracle 再写引擎**：每种新格式或重排布局，先在 Python 里实现参考版（`ref/`、`tools/canonical_xcheck.py`）。
3. **量化质量门槛**：`tools/quant_quality.py` 以 Q8_0 为基准，测 top-1 一致率和 KL；token id 文件和代码一起提交，不依赖 tokenizer 也能复现。这比只看困惑度更适合评估小模型。
4. ⭐ **工具调用解析的测试方法**：同一份语料分别整段读、逐字节读、按 3/7 字节分块读、在每个字节处切成两半读，所有读法必须产生相同的事件序列。**这是最值得照抄的一项。**另外，用 golden 文件对比聊天模板的渲染结果和 Hugging Face 的结果。
5. **stop 串必须在文本层判断，并带回扣（holdback）机制**：用分词后的子 token 当 stop id，会把 “Nerazim” 截断成 “Neraz”。
6. **基准协议**（`docs/BENCHMARK_PROTOCOL.md`）：
   - prompt 语料冻结并记录 SHA-256；
   - 同一台机器上两个引擎交替运行，取中位数；
   - 输出 llama.cpp 同名的 timings 字段，方便直接对比。
7. **内存规划器写成纯函数**，并提供独立的 CLI；规划的输入**不允许默默默认为 0**（曾因此漏算 117MB）；**读 cgroup v2 内存限制**（树莓派和 Docker 场景必须做）。
8. **环境变量登记表**：自动扫描所有 getenv 生成文档，新增开关没有登记状态时测试失败（Strata 已经有 272 个开关，这是失控之后的补救）。
9. **分包签名**：manifest 包含平台、后端、架构、每个文件的 SHA-256 和 Ed25519 签名，包目录不可变。

---

## 5. 来自文档和提交历史的坑

| 类别 | 教训 |
|---|---|
| GDN 数学 | SiLU 作用在拆分 q/k/v 之前；v 不做 L2 归一化；q 额外乘 1/√S；value head 对应的 key head 是 `vh % HK`，**和 FLA 的分组方向相反**；Qwen3.5 的输出门是 SiLU。**这些地方算错了，输出看起来仍然正常** |
| 状态契约 | 会改状态的调用如果走不了，必须在动状态之前拒绝；同一层状态上的调用必须严格顺序执行 |
| 确定性 | chunk 切法和单 token、多 token 内核的差异都会改变浮点结果。→ 要求多 token 内核的逐 token 运算和单 token 内核一致 |
| MTP | 第 i 个 cell 的输入是位置 i 的 hidden 加上位置 i+1 的 token，预测位置 i+2 |
| 缓存安全 | 出现 NaN/Inf 后清空全部缓存；复用 KV 只能限定在共享前缀内；只看 token 相等不够，图片也要一致 |
| 测试盲区 | 只测单 token 发现不了位置不前进的 bug；PLE 模块单独测过，但从没被调用，导致整个模型算错。→ **必须做端到端的多 token logits 比对，并有 FP64 参考实现** |
| Metal | threadgroup 超过 `maxTotalThreadsPerThreadgroup` 时 dispatch **静默不执行**，每次都要检查；寄存器压力因芯片而异；macOS 只让约 3/4 的内存常驻，大 arena 要 mlock |
| 服务层 | 工具调用参数里的 True/False、没有 `</think>` 就结束、Anthropic 的 tool_result 顺序、非 UTF-8 输出、CRLF 破坏 jinja 模板 |
| 并发 | 设计阶段就把请求生命周期写成状态机（取消、中止、排队） |

## 6. 不适用的部分

- MoE 专家缓存、SSD 流式加载专家、PCIe 环形缓冲、多 GPU 切分、显存弹性、KV 流式驻留、KVA 近似计算。
- NVFP4、Gyro trellis 量化、按显卡测出来的 MMVQ 行数表、QSA 稀疏注意力、HIP/SYCL 各种包。
- 和 Qwen3.8-Next 深度耦合的 `prefill.cpp`（610KB）和 `kernels.cu`（200KB）。
- Python 控制面 + 子进程管道协议：我们用进程内的 C ABI，这一整类问题都不会出现。

## 7. 对路线图的调整

| 阶段 | 新增或调整 |
|---|---|
| PoC | GEMV 的 Rust 移植**以 `src/quant/arm/arm_dots.inc` 为蓝本**；照抄 `isa.cpp` 的探测逻辑；建立 scalar 参考实现 + 逐位对齐测试 + `--bench` 的测试框架 |
| M1 | 张量表校验、按 dtype 分派、arena 零分配；用 epoch-CAS 线程池；端到端多 token logits 比对 |
| M2 | 按 `GDN_CONTRACT.md` 实现 Qwen3.5；`LayerState` 采用“只读验证 + 重放提交”；检查点式前缀缓存；PLE 异步读取 |
| M3 | suffix drafter + 按收益选择投机策略；多 token 共享权重解码 |
| M4 | Metal 后端移植 unit 抽象 GEMV 和 objc2-metal 零拷贝映射 |
| 服务 | 工具调用语料 + 切片测试、stop 回扣、聊天模板 golden 测试 |
| 自研 | **ARMv8.0 NEON 和查表内核**（Zero 2W、树莓派 4），这是 Strata 没有覆盖的部分 |

> [!IMPORTANT]
> 移植代码时保留 MIT 版权声明。CPU 量化层本身也来自 ggml（MIT），要同时保留 ggml 的版权声明。
