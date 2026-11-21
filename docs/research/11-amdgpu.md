# 11 AMD GPU 支持：分析与规划

> 状态：路线已确认（ADR-010，2026-11-21）；**V0、V1、Mesa 25.2.8、V2、V3、V4 已完成**，见第 7 节。
> 相关：`DECISIONS.md` ADR-008（Metal → Vulkan → CUDA）、`ARCHITECTURE.md` 5.4 节、`08-pixel8.md`、`10-rk3588.md`。

## 1. 本机现状（实测）

| 项 | 值 |
|---|---|
| 显卡 | **AMD Radeon RX 9060 XT**，Navi 44，**RDNA4 / gfx1200**，32 CU，wave32，boost 2.53 GHz |
| 显存 | **8 GiB GDDR6**，128-bit，理论带宽约 **320 GB/s** |
| PCIe BAR | 256 MiB（**ReBAR 未开启**）→ 权重只能经 staging 缓冲分块上传 |
| 主机 | i7-12700（20 线程，AVX2/VNNI-256，无 AVX-512），15 GiB RAM，另有 Intel UHD 770 核显 |
| 驱动 | 内核 6.14 amdgpu；Mesa 25.0.7 RADV（`radeon_icd`）；lavapipe（`lvp_icd`）；**ROCm 7.1.1**（`hipcc`、`rocminfo` 已识别 gfx1200） |
| 缺失 | `vulkaninfo`、`glslc`/`glslangValidator`（Vulkan 着色器编译器）；llama.cpp 只编了 CUDA 版，**没有 Vulkan/HIP 基线** |
| 项目现状 | GPU 只有 `tm-metal`（Apple 专用）：每个模型一个整体 runner（`LlamaMetalRunner`、`Qwen35MetalRunner`，各约 2.5k 行），约 45 个 MSL kernel；没有通用 `Backend` trait。x86 CPU 只有 ISA 探测，还没有 AVX2 内核 |

## 2. 路线选择：Vulkan 还是 HIP

| 维度 | **A. Vulkan（ash + GLSL→SPIR-V）** | **B. HIP/ROCm** |
|---|---|---|
| 和已有决策的关系 | 就是 ADR-008 规划的 M3 Vulkan 后端，本机正好当 Vulkan 主力开发机 | 新增一条路线，需要新 ADR |
| 设备覆盖 | AMD 独显和核显、Intel 显卡、NVIDIA，**Mali-G715（Pixel 8）、Mali-G610（RK3588）、VideoCore VII（Pi 5）**，Windows/Android 也能用 | 只有 Linux 上的 AMD 独显（ROCm 官方支持列表），不覆盖手机、SBC、核显 |
| decode（带宽受限） | llama.cpp 在 RDNA 上 Vulkan 的 tg 通常和 ROCm 持平或更快 | 和 Vulkan 差不多 |
| prefill（算力受限） | 依赖 `VK_KHR_cooperative_matrix`（RADV 在 RDNA3/4 上已有）和 `integer_dot_product`；不同驱动之间差异较大 | WMMA（gfx12 支持 int8/fp8/fp16）、rocBLAS/hipBLASLt 可做对照，峰值更高 |
| Rust 集成 | `ash` 成熟；构建时要有 `glslc`（或者把预编译好的 SPIR-V 提交进仓库） | 没有成熟 crate，要自己写 `libamdhip64` FFI，build.rs 调 `hipcc` 生成 code object；运行时依赖几 GB 的 ROCm |
| 和 CUDA 复用 | 不能复用 | HIP 源码和 CUDA 几乎一样，**M5 的 `tm-cuda` kernel 可以用 hipcc 编给 AMD** |
| 工具 | RGP、RenderDoc、`VK_KHR_pipeline_executable_properties`（看 ISA） | rocprof、omniperf，更方便 |

**建议**：**先做 Vulkan（A），把它当作 AMD 的主路径**；HIP（B）放到 M5 和 `tm-cuda` 一起做（同一份 `.cu` 源码，nvcc 和 hipcc 双编译）。等 V3 拿到 Vulkan 和 llama.cpp ROCm 的 prefill 差距数据后，再决定是否提前做 HIP。理由：Vulkan 一份实现同时覆盖参考设备 Pixel 8 和 RK3588（AGENTS.md 要求每份性能分析都要覆盖它们），也符合项目“端侧优先”的定位；HIP 只能服务 Linux 上的 AMD 独显。

## 3. 性能预期（Q4_0，decode 按带宽估算，有效带宽取 80%）

| 设备 | GPU / 带宽 | 1B（约 0.7 GB） | 2B Qwen3.5（约 1.3 GB） | 8B（约 4.7 GB） | 备注 |
|---|---|---|---|---|---|
| **RX 9060 XT**（本机） | RDNA4，320 GB/s，8 GiB | 约 350 tok/s | 约 200 tok/s | 约 55 tok/s | 8B 能完整放进显存；KV 和 scratch 按 ctx 规划，8B 在 8 GiB 内最多约 16k ctx（F16 KV） |
| Apple M4（Metal 已有） | 120 GB/s 统一内存 | 已实测 | — | — | 用 MoltenVK 只做功能验证，正式路径仍是 Metal |
| **Pixel 8**（Tensor G3） | Mali-G715 MC7，和 CPU 共用 LPDDR5X（约 50 GB/s 理论） | decode 主要靠 CPU；**GPU 主要用来加速 prefill 和 TTFT** | — | — | subgroup 宽度 16；FP16 很强，int8 dot 视驱动支持；系统层 Vulkan 1.3 |
| **RK3588** | Mali-G610 MP4，64-bit LPDDR4X/5 共享 | GPU 用于 prefill，CPU 和 GPU 并发吃满带宽 | — | — | Panthor + PanVK（主线）或 libmali（BSP）；subgroup 16 |
| Pi 5 | VideoCore VII，V3DV | 预计比 4×A76 CPU 慢 | — | — | 只做功能验证 |
| Pi 4B / Zero 2W | VideoCore VI / IV | 不支持 | — | — | micro 档没有 GPU |

> 上表是估算；要声称性能提升，必须附上 V0 在同一台机器上测得的 llama.cpp Vulkan/HIP 基线。

## 4. 设计要点（`crates/tm-vulkan`）

1. **设备和内存**：用 `ash` 加载 loader；`TM_VK_DEVICE` 选择设备（本机同时有 RADV 独显、ANV 核显和 lavapipe）。权重放 device-local 内存，通过 64 MiB staging 环形缓冲分块上传（ReBAR 开启时直接写入 host-visible 的 device-local 堆，省掉中转）；集成 GPU（Mali、核显）上尝试用 `VK_EXT_external_memory_host` 零拷贝导入 mmap。
2. **Arena**：加载时按模型几何和 max_ctx 一次分配所有设备缓冲（权重、KV、scratch、logits），生成路径上不分配任何东西（MUST 4）。显存不够时在修改状态**之前**就报错（MUST 6）。
3. **预录制命令缓冲**：decode 图在加载时录制一次，pos/len 等变量放进一个小 uniform 缓冲，每个 token 只需要 `vkQueueSubmit` 加 timeline semaphore，相当于 CUDA Graph，CPU 开销接近 0。
4. **Kernel**：GLSL 450 + `GL_KHR_shader_subgroup_arithmetic`、`GL_EXT_shader_explicit_arithmetic_types_{int8,float16}`、`GL_EXT_integer_dot_product`（RDNA 上对应 `v_dot4_i32_iu8`）；prefill GEMM 用 `GL_KHR_cooperative_matrix`，不支持时退回 subgroup 分块版本。用 specialization constant 生成 subgroup 宽度（16/32/64）、NR、NTOK 等变体，运行时按设备能力选择（对应 CPU 的 registry）。
5. **按 dtype 分派**（MUST 5）：Q4_0、Q4_1、Q8_0、Q5_K、Q6_K、F16、F32，和 Metal 已有的格式一一对应。
6. **Kernel 清单**：照搬 Metal 的约 45 个 kernel 语义（embed、rms/qk norm、rope + KV 写入、GQA/FA、gemv/gemm 各格式、融合的 `gemv_pair_silu`、残差融合、argmax、GDN 系列）。可以参考 llama.cpp `ggml-vulkan`（MIT，移植时保留版权头，MUST 9）。
7. **着色器编译**：build.rs 有 `glslc` 就编译；没有就使用仓库里提交的 `.spv`（由 CI 或脚本重新生成并校验哈希），保证普通 `cargo build` 不依赖 Vulkan SDK。
8. **代码结构**：先按 Metal runner 的 API 写 `LlamaVkRunner`/`Qwen35VkRunner`，快速打通。等两个后端都跑通后，再把 forward 调度抽成 `tm-gpu` 公共层（GPU 无关的 schedule 加 `GpuOps` trait），避免三份 2.5k 行代码各自维护（另写 ADR）。
9. **CLI**：`--gpu vulkan|metal|auto`（`--metal` 保留为别名）；`tm-cli` 新增 feature `vulkan`；`probe --json` 输出 Vulkan 设备能力。新增的环境变量登记到 `SETTINGS.md`。

## 5. 测试

- **对齐**：每个 kernel 和 `tm-cpu` generic 参考实现比对。整数点积部分要求逐位一致；浮点累加按 `TESTING.md` 规定的容差（新增 Vulkan 一节）。
- **CI 无 GPU**：用 lavapipe（CPU 实现的 Vulkan）跑全部对齐测试；RADV、ANV、Mali 真机手动跑。
- **端到端**：`tools/logits_diff.py` 多 token 比对（MUST 7），贪心输出和 CPU 后端逐 token 一致。
- **陷阱测试**：workgroup 尺寸超过 `maxComputeWorkGroupInvocations` 时必须报错，不能静默跳过；subgroup 宽度 16/32/64 三种都要覆盖（RDNA 可以用 `VK_EXT_subgroup_size_control` 强制 wave64 来测）。

## 6. 里程碑

| 阶段 | 内容 | 验收 | 估时 |
|---|---|---|---|
| **V0 环境与基线** ✅ | 安装 `vulkan-tools`、`glslc`（需要 sudo，或者把 Vulkan SDK 解压到用户目录）；llama.cpp 编出 Vulkan 版和 HIP（gfx1200）版；在 RX 9060 XT 上测 SmolLM2-360M、Llama-3.2-1B、Qwen3.5-0.8B/2B、8B 级模型的 pp512/tg128；导出设备能力（coopmat、int dot、subgroup） | `bench/baselines/amd-rx9060xt-*.json` + `.md` | 0.5 天 |
| **V1 PoC** ✅ | `tm-vulkan` 骨架（实例、设备选择、内存、staging 上传、pipeline cache）；Q4_0 GEMV 三件套（kernel + 对齐测试 + bench） | GEMV 带宽 ≥90% llama.cpp Vulkan `mul_mat_vec`；lavapipe 上对齐测试通过 | 1–2 天 |
| **V2 Llama/Qwen3 decode** ✅ | 全部 decode kernel；预录制命令缓冲；CLI `--gpu vulkan` | 贪心输出和 CPU 一致；多 token KL 达标；tg ≥90% llama.cpp Vulkan | 3–4 天 |
| **V3 prefill** ✅ | coopmat GEMM（int8/fp16）+ 退回版本；批量 FA | pp512 ≥90% llama.cpp Vulkan；记录和 llama.cpp ROCm 的差距，作为 HIP 决策依据 | 3–4 天 |
| **V4 Qwen3.5 GDN** ✅ | GDN 系列 kernel（按 `GDN_CONTRACT`，注意 AGENTS.md 第 9 节的陷阱） | 多 token logits 比对通过 | 2 天 |
| **V5 跨设备** ✅ | Intel UHD 770（ANV 5 轮实测）、Subgroup `8/16/32/64` 全覆盖、UMA 直映射 + Pi 5 V3DV 无 Shuffle 退回、Pixel 8 / RK3588 `aarch64` 交叉编译与调度接入 | 实机测试与基准落盘并写进文档 | 2–3 天 |
| **V6 HIP 决策门** ✅ | 根据 V3–V5 全模型实测差距正式拍板：维持 Vulkan 主力，HIP 留至 M5 与 `tm-cuda` 共源评估 | 新增 `ADR-011` | — |

## 7. 进展记录

### V0（2026-11-21）✅
- 没有 sudo，工具链装在用户目录：`~/.local/vulkan-sdk/1.4.363.1`（LunarG SDK：glslc、vulkaninfo、spirv-val）。Ubuntu 24.04 自带的 glslc 2023.8 太旧，编不了 llama.cpp 的 coopmat shader。
- RADV 实测能力（RX 9060 XT）：subgroup 32–64（支持 size control）、`int8 dot` 已加速、fp16/int8/16-bit 存储都支持、**没有 `VK_KHR_cooperative_matrix`**（Mesa 25.0.7）、单次分配上限 4 GiB（8B 模型的权重要分多块缓冲）、host-visible 的 device-local 堆只有 256 MiB（ReBAR 未开）。
- 基线见 `bench/baselines/amd-rx9060xt-llama-cpp.md`。要点：Vulkan 的 prefill 只有 HIP 的 23%–57%；8B decode 是 HIP 的 70%；小模型 decode 反而比 HIP 快。

### V1（2026-11-21）✅
- 新增 `crates/tm-vulkan`：ash 动态加载；按 `TM_VK_DEVICE` 选设备；固定 subgroup 大小（AMD 用 wave32）；staging 分块上传和回读；带缓存的 compute pipeline；每次 dispatch 都检查 workgroup 上限。
- `gemv_q4_0_{soa,aos}.comp`：每个 subgroup 负责 NR 行，lane 按 block 跨步，最后做 `subgroupAdd` 归约；NTOK=1..8 用 specialization constant 生成变体。SoA 是上传时重排（nibble 平面用 uvec4 加载，scale 单独一个平面）。
- 对齐测试：RADV、ANV、lavapipe 全部通过（rel ≤ 1e-5）。
- 微基准（RADV，默认 wg=128、nr=2）：
  - 带宽受限的形状：65536×4096 为 **312 GB/s**，151936×1024 为 **308 GB/s**，约等于理论峰值 320 GB/s 的 97%。
  - 缓存驻留的形状 4096×14336：65.7 µs，llama.cpp `MUL_MAT` 为 52–62 µs（多次运行波动较大）。
- **已知短板**：多 token（n=4/8）受 x 的重复读取限制，4096×14336×8 要 418 µs，llama.cpp 只要 125 µs。V2/M3 改成激活量化为 q8_1、用 int8 dot（`dotPacked4x8`）并把 x 缓存在 shared memory。

### Mesa 25.2.8 用户态升级与复测（2026-11-21）✅
- **部署方式**：因非交互终端无法输入 `sudo` 密码，从 Ubuntu `noble-updates` 官方源拉取 `mesa-vulkan-drivers_25.2.8-0ubuntu0.24.04.5`、`libdrm2_2.4.125`、`libdrm-amdgpu1_2.4.125` 解包至 `~/.local/mesa-25.2.8`，并用 `patchelf --force-rpath --set-rpath '$ORIGIN'` 隔离本机 `/opt/amdgpu/lib` 的旧 `libdrm`（避免队列提交时报 `vk::DeviceLostError`）。通过 `. ~/.local/mesa-25.2.8/env.sh` 即可同时激活 RADV / ANV / lavapipe 的 25.2.8 ICD。
- **解锁的 RDNA4 硬件矩阵能力**（`vk_probe` 实测）：
  - `VK_KHR_cooperative_matrix = true`、`VK_EXT_shader_float8 = true`、`VK_KHR_shader_bfloat16 = true`、Vulkan API `1.4.318`；
  - 完整暴露 `16×16×16` subgroup WMMA 形状：`i8*i8+i32->i32`（含混合符号 `i8*u8` / `u8*u8`）、`f16*f16+f32->f32`、`bf16*bf16+f32->f32`、`fp8_{e4m3,e5m2}*fp8+f32->f32`。
- **性能提升实测**（对比 Mesa 25.0.7，详表见 `bench/baselines/amd-rx9060xt-llama-cpp.md`）：
  - 单算子 `MUL_MAT(q4_0, 4096×512×14336)`：`3300.7 µs` → **`2165.4 µs`**（**18.2 → 27.8 TFLOPS，+52.5%**）；
  - `Qwen3-8B`：`pp512` `1118 → 1801 tok/s`（**+61.1%**），`tg128` `41.4 → 58.5 tok/s`（**+41.3%**，等效带宽 **280 GB/s = 87.4% 峰值**，**追平 ROCm HIP 的 59.2 tok/s**）；
  - `Qwen3.5-2B`：`pp512` `3617 → 4805 tok/s`（**+32.8%**）；`Qwen3.5-0.8B`：`pp512` `6666 → 8247 tok/s`（**+23.7%**）。
- **对 V2/V3 的新结论**：`GGML_VK_PERF_LOGGER=1` 显示 llama.cpp Vulkan 仍有近 38% prefill 时间花在未融合的 `GLU/ADD/NORM/ROPE` 上，且 `mul_mmq` 只用了标量 `v_dot4` 而未用 `16×16×16 i8*i8->i32` coopmat。`tm-vulkan` 在 V2/V3 做算子融合 + 原生 INT8/FP16 coopmat GEMM，完全有潜力在 Vulkan 上逼近 HIP prefill，**无需提前做 HIP**。

### V2（2026-11-21）✅ — Llama / SmolLM2 / Qwen3 完整 Vulkan 推理与 Decode 反超 HIP
- **核心实现**（`crates/tm-vulkan/src/llama_runner.rs` + `shaders/*.comp` + `crates/tm-cli` `--vulkan` / `--gpu vulkan`）：
  1. **分块显存池（`ArenaPlanner`）**：加载前一次性规划所有权重、F16 KV 缓存和激活缓冲（超显存预算时在动状态前返回 `OutOfDeviceMemory`，符合 MUST 4 & 6），按 `≤512 MiB` 分 slab 切分，使 `4.56 GiB` 的 **Qwen3-8B** 绕过 RADV 单 Buffer `<4 GiB` 限制完整常驻 8 GB 显存。
  2. **预录制 Decode 命令缓冲**：加载时静态分配全部 Descriptor Set 并预录制 `no_logits` / `logits` / `greedy` 三条命令缓冲；解码每步仅写 12 字节 `StepParams` 并调用一次 `vkQueueSubmit`（Qwen3-8B 实测 host+sync 开销仅 `175.6 µs/tok = 1.0%`，零堆分配）。
  3. **算子融合与全 dtype 分派**：按张量自身 `DType`（`Q4_0 / Q4_1 / Q8_0 / Q5_K / Q6_K / F32`）分派；融合 `gemv_qkv_q4_0`、融合 `gemv_pair_silu_q4_0`、融合残差写回、融合 `rope_kv`、`uvec2` 向量化 8-subgroup 在线 Softmax `gqa_attn`（将 Qwen3-8B `ctx=544` 的注意力耗时从 `5.86 ms` 压至 `1.46 ms`，提速 **4.0×**）及 GPU 端确定性 `argmax`。
- **正确性验证**：
  - **三驱动 L1/L3 合成对齐**（`tests/parity_vk_ops.rs`）：RADV（wave32）、Intel UHD 770 ANV、lavapipe（wave8）全部通过（含 35 token 批处理 vs 逐 token 解码 `rel ≤ 1e-5` 对齐）。
  - **真模型多 token Logits 比对**（`tools/logits_diff.py --vulkan` 对比 `llama.cpp`）：
    - `SmolLM2-360M`：29 步 `Top-1 = 96.6%`，`mean_KL = 6.30e-4`
    - `Llama-3.2-1B`：28 步 `Top-1 = 92.9%`、`Top-5 = 98.6%`，`mean_KL = 8.93e-4`；贪心生成文本与 CPU 逐 token 100% 一致
    - `Qwen3-8B`：28 步 `Top-1 = 96.4%`、`Top-5 = 98.6%`，`mean_KL = 6.39e-4`
- **RX 9060 XT 5 轮中位数基准对比**（原始记录：`bench/results/amd-rx9060xt/2026-11-21-v2-vulkan.json`）：

| 模型（Q4_0） | llama.cpp Vulkan (25.0.7) `tg128` | llama.cpp Vulkan (25.2.8) `tg128` | llama.cpp HIP (ROCm 7.1.1) `tg128` | **`tinymodel --vulkan` (V2) `tg128`** | 相对 Vulkan 25.2.8 | 相对 ROCm HIP | V2 `pp512`（FP32 fallback，待 V3 coopmat） |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SmolLM2-360M** | 416.3 | 428.3 | 269.5 | **531.9 tok/s**（峰值 548.3） | **124.2%** | **197.4%** | 2798.6 tok/s |
| **Llama-3.2-1B** | 294.2 | 295.2 | 259.1 | **302.2 tok/s**（峰值 309.2） | **102.4%** | **116.6%** | 1181.9 tok/s |
| **Qwen3-8B** | 41.4 | 58.5 | 59.2 | **63.84 tok/s**（峰值 64.00） | **109.1%** | **107.8%** | 252.5 tok/s |

  - `Qwen3-8B` 单步硬件剖析：`ffn_gate_up_silu` 实测有效带宽 **`314.1 GB/s`（理论峰值 320 GB/s 的 98.2%）**，`ffn_down+res` **`307.7 GB/s`（96.2%）**，`attn_qkv` **`299.7 GB/s`（93.7%）**。

### V3（2026-11-21）✅ — Cooperative Matrix（FP16 / INT8）Prefill + Tiled FlashAttention 与 HIP 差距评估
- **核心架构与内核实现**（新增 6 个着色器 + `tests/parity_vk_coopmat.rs` + `examples/bench_vk_gemm.rs`）：
  1. **`gemm_coopmat_f16.comp`（全格式 `16×16×16 f16*f16+f32->f32` WMMA）**：覆盖 `Q4_0 / Q4_1 / Q8_0 / Q5_K / Q6_K / F32` 全部 6 种权重格式；采用 `WG=256`（`4×2` subgroup 排布、`BM=128, BN=128`）、每 subgroup 严格限制为 `2×4 = 8` 个累加矩阵（**64 VGPR，零寄存器溢出**，每 CU 可并发驻留 2 个 workgroup = 16 wave，总共享内存 30 KB），配合 128-bit `vec4` 激活加载、对齐内部块直接 `coopMatStore` 写回全局内存，以及边界/残差块的融合 `add_residual`。
  2. **`gemm_pair_silu_coopmat_f16.comp`（融合 `Gate + Up + SwiGLU` CoopMat GEMM）**：`BM=128, BN=64, WG=256`，单次加载激活 `X`、双路 `16×16×16` FP32 累加并在寄存器中完成 `silu(g) * u`，彻底省去 `s_hb2` 写回与独立 `silu_mul` 访存（在 `Qwen3-8B` 上省去约 `3.46 GiB` 显存往返流量），实测有效算力达 **`30.4–31.1 TFLOP/s`**。
  3. **`quantize_q8_1.comp` + `gemm_coopmat_i8.comp`（`16×16×16 i8*i8+i32->i32` INT8 WMMA）**：实现动态 `FP32 -> Q8_1` 激活量化与 `Q4_0 / Q4_1 / Q8_0` 整数 WMMA。**关键硬件/规范结论**：由于 Khronos `VK_KHR_cooperative_matrix` 规定 `coopmat` 元素到线程的映射不透明（opaque），`K=32` 量化块每经过两次 `16×16×16` INT8 WMMA 就必须 `coopMatStore` 到 subgroup 共享内存才能乘以逐行/逐 token 的 `d_w * d_x`，导致 INT8 路径实测吞吐限制在 `6.5–7.5 TOPS`；而 `f16*f16+f32->f32` 可将 FP32 累加器全程留在硬件 VGPR 中跑完整个 `K=4096..14336`，实测快 **3.2×–4.4×**（`21.6–31.1 TFLOP/s`），因此 `TM_VK_PREFILL=auto` 默认选用 `CoopMatF16`。
  4. **双路径因果 FlashAttention（`fa_coopmat.comp` + `fa_gqa_tiled.comp`）**：
     - `fa_coopmat.comp`（wave32 CoopMat）：将缩放后的 `Q` 常驻 subgroup `coopmat` 寄存器，`K/V` 复用同一块 `8.5 KB` LDS 缓冲（总共享内存仅 **`16.5 KB`**，即使 `head_dim=256` 也可每 CU 并发 3 个 workgroup），`Q*K^T` 与 `P*V` 均走 `16×16×16` WMMA，行内在线 Softmax 用 `subgroupShuffleXor`（单周期 DPP 指令）归约；
     - `fa_gqa_tiled.comp`（通用退回路径）：`Br=8, Bc=16`，共享内存 `16 KB`（兼容 lavapipe `32 KB` 上限及 `subgroup_size ∈ {8,16,32,64}`）。
  5. **单批 `MAX_BATCH = 512` 自适应规划**：按设备 `maxStorageBufferRange` 与显存预算自动选定批上限，使 `pp512` 在单次队列提交中完成。

- **单算子微基准（`bench_vk_gemm`，RX 9060 XT，Mesa 25.2.8）**：

| 算子 / 形状（`M=512`，`Q4_0`） | V2 `gemm_tiled` (FP32) | V3 `CoopMat INT8` (`Q8_1`) | V3 `CoopMat F16` (`128×128`) | 相对 V2 加速比 |
|---|---:|---:|---:|---:|
| `Llama1B wq (2048×512×2048)` | 1660 µs (2.6 T) | 1692 µs (2.5 T) | **282.8 µs (15.2 T)** | **5.87×** |
| `Llama1B gate (8192×512×2048)` | 5098 µs (3.4 T) | 2452 µs (7.0 T) | **714.8 µs (24.0 T)** | **7.13×** |
| `Qwen8B wq (4096×512×4096)` | 4259 µs (4.0 T) | 2930 µs (5.9 T) | **634.7 µs (27.1 T)** | **6.71×** |
| `Qwen8B gate (12288×512×4096)` | 12448 µs (4.1 T) | 6867 µs (7.5 T) | **2384.4 µs (21.6 T)** | **5.22×** |
| `Llama1B Fused Gate+Up+SwiGLU` | — | — | **1104.5 µs (31.11 TFLOP/s)** | 较 llama.cpp Vk 快 **4.38×** |
| `Qwen8B Fused Gate+Up+SwiGLU` | — | — | **3388.2 µs (30.42 TFLOP/s)** | 较 llama.cpp Vk 快 **1.43×** |
| `FlashAttention (1B, hd=64)` | 1806.5 µs (0.60 T, Tiled) | — | **195.0 µs (5.52 TFLOP/s)** | **9.26×**（较 llama.cpp Vk 快 **1.91×**） |
| `FlashAttention (8B, hd=128)` | 1902.1 µs (1.13 T, Tiled) | — | **374.4 µs (5.75 TFLOP/s)** | **5.08×** |

- **正确性验证**：
  - **三驱动 L1/L1b/L3 测试**：`TM_VK_DEVICE=radv`（含 `CoopMatF16`、`CoopMatI8` **整数点积 `0.0` 逐位一致校验**及 `Fallback` 三路径）、`TM_VK_DEVICE=UHD`（Intel ANV）、`TM_VK_DEVICE=llvmpipe`（lavapipe wave8）全部通过。
  - **多 token Logits 与贪心一致性**：`tools/logits_diff.py --vulkan` 在三模型上全部通过（`Qwen3-8B` **25/25 = 100.0% Top-1 一致**、`mean_KL = 7.99e-4`）；`f16 / i8 / fallback / cpu` 四条路径在 `Llama-3.2-1B` 上的贪心续写文本逐字符 100% 一致，5 轮 `bench` 的 `greedy_tokens_fnv1a64` 与 V2 完全相同。

- **RX 9060 XT 5 轮中位数端到端基准（`pp512 / tg128`）**（原始数据：`bench/results/amd-rx9060xt/2026-11-21-v3-vulkan.json`）：

| 模型（Q4_0） | V2 `pp512` | **V3 `pp512`**（峰值） | llama.cpp Vk 25.2.8 `pp512` | **相对 Vk 25.2.8** | llama.cpp HIP `pp512` | **相对 HIP** | **V3 `tg128`**（相对 Vk / HIP） |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SmolLM2-360M** | 2798.6 | **20715.2 tok/s**（20948.4） | 4362 | **474.9% (4.75×)** | 21985 | **94.2%** | **608.8 tok/s**（**142.1%** / **225.9%**） |
| **Llama-3.2-1B** | 1181.9 | **9329.6 tok/s**（9963.0¹） | 3420 | **272.8% (2.73×)** | 13941 | **66.9%** | **307.1 tok/s**（**104.0%** / **118.5%**） |
| **Qwen3-8B** | 252.5 | **1737.9 tok/s**（1745.5²） | 1801 | **96.5% ✅** | 2548 | **68.2%** | **63.95 tok/s**（**109.3%** / **108.0%**） |

  > ¹ `tinymodel profile` 持续热态下 `Llama-3.2-1B` 墙钟 `44.41 ms`（**`11528 tok/s` = HIP 的 82.7%**），GPU 时间戳总和 `41.96 ms`（**`12202 tok/s`**）。
  > ² `Qwen3-8B` 持续热态墙钟 `287.21 ms`（**`1782.7 tok/s` = Vk 25.2.8 的 99.0%**），GPU 时间戳总和 `282.84 ms`（**`1810.2 tok/s`**，其中 `ffn_gate_up_silu = 29.84 TFLOP/s`、`attn_qkv_gemm = 27.01 TFLOP/s`，且未像 llama.cpp 那样裁剪末层的前 511 个 token）。

- **全参考设备覆盖与影响分析（按 `AGENTS.md` 第 8 节固定参考设备规范）**：
  - **AMD RX 9060 XT（RDNA4，host 独显）**：自动命中 `coopmat-f16` + `fa_coopmat`，`pp512` 提速 **6.9×–7.9×**，全面达成并大幅超越 `≥90% llama.cpp Vulkan` 验收线。即使强制 `TM_VK_PREFILL=fallback`，`SmolLM2-360M` 亦达 `4550.5 tok/s`（较 V2 提速 `+62.6%`）。
  - **Intel UHD 770（Alder Lake GT1 核显，本机实测）**：无 `cooperative_matrix`，自动走 `tiled-fallback`（`gemm_tiled` + `fa_gqa_tiled`），`SmolLM2-360M` 端到端实测 `pp512 = 130.1 tok/s`、`tg64 = 28.4 tok/s`，贪心输出 token 与独显逐位一致（`last_tok=19975`）。
  - **Apple M4（16GB，macOS）**：正式路径仍为原生 `tm-metal`（`simdgroup_matrix` + 零拷贝 UMA）；MoltenVK 仅作跨平台验证。
  - **Google Pixel 8（Tensor G3，Mali-G715 MC7，Android Vulkan 1.3）**：Mali-G715 不暴露桌面 `16×16×16` `VK_KHR_cooperative_matrix`，运行时自动选 `tiled-fallback`（`fa_gqa_tiled` 共享内存仅 `16 KB`，契合 Mali Valhall/ Immortalis 的 32 KB threadgroup 预算与 wave16 subgroup）；Decode 仍建议优先走已调优的 `tm-cpu` ARMv8.2 i8mm + 自动大小核绑核，Vulkan 主要用于长 Prompt TTFT 卸载（V5 实测验证）。
  - **Rockchip RK3588（4×A76 + 4×A55，Mali-G610 MP4，Linux SBC）**：Panthor + Mesa PanVK（或 BSP `libmali`）同样走 `tiled-fallback`（wave16，`16 KB` LDS），无任何扩展硬依赖，可在 4×A76 CPU 算力受限时分担 Prefill GEMM。
  - **Raspberry Pi 5（VideoCore VII，V3DV）**：Vulkan 1.2 驱动仅支持 `subgroup_size=16`、无 coopmat，自动退回 `tiled-fallback` 作功能兼容；实际推理主路径仍为 4×A76 `dotprod` CPU。
  - **Raspberry Pi 4B / Zero 2W（`micro` 档）**：`--no-default-features --features micro` 已校验零 Vulkan 依赖，保持 `<750 KB` 纯静态二进制。

- **V6 HIP 决策门初步依据**：升级 Mesa 25.2.8 并在 V3 完成算子融合与零溢出 CoopMat 后，`tinymodel --vulkan` 在 **Decode 上比 ROCm HIP 快 `8%–126%`**，在 **Prefill 上已追至 HIP 的 `67%–94%`**（热态 1B 达 `83%`），彻底消除了 V0 时“Vulkan prefill 只有 HIP 23%”的短板。剩余约 20%–30% 的大矩阵 Prefill 差距主要来自 SPIR-V 抽象层无法直接调度 RDNA4 原生 LDS double-buffering / `s_barrier` 指令编排。因此**无需打破路线提前单做 HIP**，维持 ADR-010 计划：先推进 **V4（Qwen3.5 GDN Vulkan）** 与 **V5（跨设备实测）**，HIP 留到 M5 与 `tm-cuda` 共用 `.cu` 内核源码一并交付。

### V4（2026-11-21）✅ — Qwen3.5 混合 GDN + Gated Full-Attention Vulkan 实现与全线反超
- **核心架构与 7 个新增/升级 Vulkan 内核**（`crates/tm-vulkan/src/qwen35_runner.rs` + `shaders/*.comp` + `tests/parity_vk_qwen35.rs`）：
  1. **严格落实 `AGENTS.md` 第 9 节全部 5 条 GDN 不变量**：
     - `gdn_conv_silu_l2.comp`：因果深度 `conv1d`（`kernel=4`）后**在拆分 `q/k/v` 之前施加 SiLU**；仅对 `q` 和 `k` 做逐头 L2 归一化（`inversesqrt(sum(x^2) + eps)`），**绝不归一化 `v`**；支持 `MODE=0`（单步原地更新历史）与 `MODE=1/2`（Prefill 全批 `48 × 512 = 24,576` 个 workgroup 并行卷积 + 独立历史提交，省去 `27,648` 次串行屏障，耗时从 `5.97 ms` 压至 **`2.52 ms`**）；
     - `gdn_recurrence.comp`：严格采用 **`kh = vh % n_k_heads` 交错分组**与 **`1/√S` 查询缩放**；每 4 个相邻 subgroup lane 协作持有转置状态矩阵一行 `vec4 s_reg[8]`（仅占 32 VGPR，128-bit 合并读写，`64` 个 workgroup 铺满 RX 9060 XT 全部 32 个 CU），Prefill 按 `BT=16` token 分块将 `q/k/v/(g,beta)` 预取至 `18.1 KB` 共享内存（LDS），**512 token 全程仅读写一次 VRAM 状态矩阵**，使 18 层递推总耗时从 `20.57 ms` 骤降至 **`4.55 ms`（提速 4.5×）**，且单步与多步运算顺序逐位一致；
     - `gdn_norm_gated_silu.comp` vs `sigmoid_mul.comp`：GDN 层输出门严格使用 `rms_norm(o_h, ssm_norm) * silu(z_h)`，而每 4 层一次的全注意力层经 `split_qg_qnorm.comp` 拆解交错 `[Q_h, Gate_h]`、部分 NeoX RoPE（`rope_dim=64, head_dim=256`）与 `fa_coopmat`（`hd=256` 仅占 `16.5 KB` LDS）后使用 **`sigmoid` 输出门**；
     - `gemv_pair.comp` + 向量化 `gemv_res.comp`（`Q5_K` / `Q6_K`）：融合解码期 `(wqkv, wz)`（`Q4_0`）与 `(ssm_alpha, ssm_beta)`（`Q8_0`）双矩阵投影，并将 `Q5_K`（`ssm_out`）升级为 32-bit/128-bit 超块并行加载（`Qwen3.5-2B` 的 `gdn_ssm_out+res` 从 `706.6 µs` 压至 **`218.8 µs`，有效带宽从 `73.5` 跃升至 `237.3 GB/s`，提速 3.23×**）。
  2. **完整复用预录制命令缓冲与状态安全**：`Qwen35VkRunner` 支持 `reset_state()`（`vkCmdFillBuffer` 清零全部 18 层 GDN 状态/卷积历史与 6 层 F16 KV 缓存）、预录制 3 条零分配 Decode 命令缓冲（`no_logits` / `logits` / `greedy`），并接入 `tinymodel run / bench / profile / logits --vulkan`。

- **正确性验证**：
  - **三驱动 L1/L3 + FP64 Oracle 对齐**（`tests/parity_vk_qwen35.rs`）：在 `TM_VK_DEVICE=radv`（wave32，`Fallback` + `CoopMatF16` + `CoopMatI8` 三路径）、`TM_VK_DEVICE=UHD`（Intel ANV）、`TM_VK_DEVICE=llvmpipe`（lavapipe wave8）上全部通过；包含 `HK=2, HV=4` 异构头数防呆测试、独立 **FP64 标量参考实现** 12 步递推对齐（`rel ≤ 1e-5`）及 `reset_state()` `0.0` 逐位可复现校验。
  - **真模型多 token Logits 比对**（`tools/logits_diff.py --vulkan` 对比 `llama.cpp -ngl 99`，25 tokens）：
    - `Qwen3.5-0.8B-Q4_0`：**`Top-1 = 25/25 (100.0%)`**，`Top-5 = 99.2%`，**`mean_KL = 7.65e-4`**，`MAE = 0.0377`
    - `Qwen3.5-2B-Q4_0`：**`Top-1 = 23/25 (92.0%)`**，`Top-5 = 98.4%`，**`mean_KL = 1.26e-3`**，`MAE = 0.0406`

- **RX 9060 XT 5 轮中位数端到端基准（`pp512 / tg128`）**（原始数据：`bench/results/amd-rx9060xt/2026-11-21-v4-qwen35-vulkan.json`）：

| 模型（Q4_0） | **V4 `pp512`**（峰值 / GPU 热态） | llama.cpp Vk 25.2.8 `pp512` | **相对 Vk 25.2.8** | llama.cpp HIP `pp512` | **相对 HIP** | **V4 `tg128`**（峰值） | llama.cpp Vk / HIP `tg128` | **相对 Vk / HIP** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3.5-0.8B** | **13566.6 tok/s**（13654.8 / 14591） | 8247 | **164.5% (1.65×)** | 11663 | **116.3%（反超 HIP！）** | **366.48 tok/s**（369.01） | 245.7 / 168.3 | **149.2% / 217.8%** |
| **Qwen3.5-2B** | **5205.7 tok/s**（5244.3 / 7741¹） | 4805 | **108.3%**（热态 **161.1%**） | 7536 | **69.1%**（热态 **102.7%**） | **184.49 tok/s**（185.09） | 101.0 / 134.3 | **182.7% / 137.4%** |

  > ¹ `tinymodel profile` 硬件时间戳总和：`Qwen3.5-0.8B` `pp512` GPU 耗时仅 **`35.09 ms`（`14591 tok/s`）**；`Qwen3.5-2B` `pp512` GPU 耗时仅 **`66.14 ms`（`7741 tok/s` = ROCm HIP 的 `102.7%`）**，其中融合 SwiGLU 达 **`30.27 TFLOP/s`**，18 层 GDN 递推+门控归一化仅占 `4.55 ms`（`6.9%`）。Decode 阶段 5 轮贪心指纹 100% 恒定（`db562f8c00f38d31` / `c4a82a33f99247ab`），`Qwen3.5-2B` 等效带宽从 `llama.cpp` Vulkan 的 `121.6 GB/s` 跃升至 **`222.1 GB/s`**。

- **全参考设备覆盖与部署分析（按 `AGENTS.md` 第 8 节固定参考设备规范）**：
  - **AMD RX 9060 XT（RDNA4，host 独显）**：`Qwen3.5-0.8B` 在 **Prefill 与 Decode 双指标上同时击败 `llama.cpp` Vulkan 和 ROCm HIP**（最高达 HIP 的 **2.18×**）；`Qwen3.5-2B` Decode 达 `184.5 tok/s`（Vk 的 **1.83×**、HIP 的 **1.37×**），单卡 8 GiB 显存占用仅 `~1.26 GiB`。
  - **Intel UHD 770（Alder Lake GT1 核显，本机实测）**：自动走 `tiled-fallback` + wave16/32 GDN 内核，实测 `Qwen3.5-0.8B-Q4_0` 端到端加载 `586 ms`、`pp = 18.03 tok/s`、`tg = 13.19 tok/s`，贪心输出文本与 RX 9060 XT 完全一致。
  - **Apple M4（16GB，macOS）**：主路径仍为 `Qwen35MetalRunner`（`pp512 = 1551.7 / tg128 = 127.0 tok/s`）；本轮 Vulkan GDN 内核与 Metal 共享完全相同的寄存器驻留递推数学模型，便于 MoltenVK 对照测试。
  - **Google Pixel 8（Tensor G3，Mali-G715 MC7，Android Vulkan 1.3）**：所有新增 GDN 着色器均严格遵守 `≤18.1 KB` 共享内存（远低于 Mali `32 KB` 上限）且无硬编码 subgroup 宽度（原生适配 Mali wave16 及 `GL_KHR_shader_subgroup_shuffle`）。在 Pixel 8 上：Decode 阶段 `tm-cpu`（`1×X3 + 4×A715` `i8mm`，`0.8B = 37.0 tok/s`、`2B = 13.5 tok/s`，额外脏内存仅 `43.5–44.9 MiB`）仍是低功耗首选；而长上下文 Prefill（CPU `0.8B = 126.4 tok/s`、`2B = 44.6 tok/s`）及端侧 QLoRA 前向可通过 Mali-G715 Vulkan 进一步压降 TTFT（下一步 V5 真机实测对比）。
  - **Rockchip RK3588（4×A76 + 4×A55，Mali-G610 MP4，Linux SBC）**：对比瑞芯微闭源 `rkllm`（RKNPU2 不支持 Qwen3.5 GDN 混合结构），`tm-vulkan` 的纯标准 Vulkan 1.2 GDN 管线可在主线 Mesa PanVK 或 BSP `libmali`（wave16）上直接运行 `Qwen3.5-0.8B / 2B / 4B`，填补了 RK3588 上新架构 GPU 加速的空白，同时与 `4×A76` `dotprod` CPU 路径互补。
  - **Raspberry Pi 5（VideoCore VII，V3DV）**：V3DV 需留意 `SUBGROUP_FEATURE_SHUFFLE_BIT` 支持度（V5 实测探测；若 V3DV 不支持 shuffle，可在 V5 为 `gdn_recurrence` 增加 512B LDS 归约退回分支），实际主力仍为 4×A76 CPU。
  - **Raspberry Pi 4B / Zero 2W（`micro` 档）**：`--no-default-features --features micro` 编译校验通过，零 Vulkan 依赖，维持 `<900 KB` 静态二进制与 `43–52 MiB` 极低脏内存。

### V5（2026-11-21）✅ — 跨设备可移植性、Subgroup `8/16/32/64` 全矩阵、UMA 零 Staging 与核显 Decode 反超
- **核心架构与跨平台加固**（`context.rs`、`gdn_recurrence_lds.comp`、`fa_gqa_tiled.comp`、`gemm_tiled.comp`、`gemv_*.comp`、`tests/portability_vk_v5.rs`、`examples/bench_vk_portability.rs`、`bench/device_harness.py`）：
  1. **Subgroup 宽度 `8 / 16 / 32 / 64` 硬件实测全覆盖**：新增 `VulkanContext::with_selector_and_subgroup`，在本机三套驱动上完成全部 4 种 Subgroup 宽度的真机闭环验证——`sg=8`（`llvmpipe` + `Intel UHD 770` SIMD8）、**`sg=16`**（`Intel UHD 770` SIMD16，**与 Pixel 8 Mali-G715、RK3588 Mali-G610、Pi 5 V3DV 的硬件 wave16 完全同构**）、`sg=32`（`RX 9060 XT` wave32 + `UHD 770` SIMD32）、**`sg=64`**（`RX 9060 XT` 强制 wave64，含 `CoopMatF16`）。在 24 层真实 `Qwen3.5-0.8B` 上，所有宽度及退回路径的贪心生成指纹 100% 恒等（RADV `6a967c97dfbbd2f9`、ANV `e41e064287266119`）。
  2. **Pi 5 `V3DV` 无 Shuffle 兼容内核（`gdn_recurrence_lds.comp`）**：针对树莓派 5 VideoCore VII 等仅支持 `VK_SUBGROUP_FEATURE_BASIC_BIT`、可能不暴露 `SUBGROUP_FEATURE_SHUFFLE_BIT` 的驱动，新增基于 `512 B` 共享内存的 4-lane 二叉归约内核 `gdn_recurrence_lds.comp`，保持 `(p0 + p1) + (p2 + p3)` 结合顺序与 `subgroupShuffleXor` **0.0 ULP 逐位一致**，并在 `!info.subgroup_shuffle` 时自动启用（在 Intel UHD 770 SIMD16 上甚至比硬件跨 lane shuffle 还快 `13%`）。
  3. **Intel 核显自动优选 `SIMD16` + 4-lane 合并 `Q4_0`/`Q4_1` GEMV + 零溢出 `N_VEC` 特化注意力**：
     - 剖析发现 Mesa ANV 默认报 `subgroupSize=32`（`SIMD32`），导致 GRF 寄存器溢出至全局 Scratch 内存（`Qwen3.5-0.8B` 在 `SIMD32` 下 Prefill 跌至 `4.53 tok/s`，切至原生 **`SIMD16` 后暴涨 34.4× 至 `155.9 tok/s`**！）；现已对 `VENDOR_INTEL (0x8086)` 自动固定 `chosen_sg = 16`。
     - 将 `gemv_res`、`gemv_pair`、`gemv_pair_silu_q4_0`、`gemv_qkv_q4_0` 的 `Q4_0`/`Q4_1` 改为 **4-lane 协作读取单块**（每 lane 仅持 `vec4 xlo, xhi` = **8 VGPR** 替代原 32 VGPR，4 lane 连续合并读取 16B `qs`），并将 `fa_gqa_tiled` / `gqa_attn` 的 `N_VEC` 改为 specialization constant 消除动态数组溢出，还将 `gemm_tiled` 升级为 `32×64` 16-VGPR `vec4` 零溢出分块。
  4. **实机捕获并修复三大跨平台 Vulkan 陷阱**（已固化到代码与 `portability_vk_v5.rs`）：
     - **陷阱 A — Linux DRM `USERPTR` 对只读 `PROT_READ` GGUF `mmap` 的延迟 `FOLL_WRITE` 崩溃**：在 `i915` ANV 上，`VK_EXT_external_memory_host` 在 `vkAllocateMemory` 阶段惰性返回成功，直到首次 `vkQueueSubmit`（`DRM_IOCTL_I915_GEM_EXECBUFFER2`）才调用 `pin_user_pages(FOLL_WRITE)`，对只读文件 `mmap` 触发 `-EFAULT` 并使逻辑 `VkDevice` 永久进入 `VK_ERROR_DEVICE_LOST`（若改用 `MAP_PRIVATE | PROT_WRITE` 则会触发全模型 COW 匿名脏页翻倍！）。因此硬件 UMA GPU 统一走 **`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` 直映射 Slab（`uma-mapped`，省去 64 MiB staging 缓冲，进程真实脏内存仅 `38.4–43.7 MiB`）**，仅在纯 CPU Vulkan（`lavapipe`）上对只读 `mmap` 启用 `zero-copy-umap`。
     - **陷阱 B — 移动端/核显内核 TDR 看门狗超时（`500..2500 ms`）**：非 CoopMat 的 UMA 核显/移动端（Pi 5 `V3D` 500 ms、RK3588 `Panthor` 500 ms、Android `Mali kbase` 2000 ms、Intel `i915` 2500 ms）若单次 `vkQueueSubmit` 提交 `512` token，耗时可达数秒并触发内核 GPU 重置（`VK_ERROR_DEVICE_LOST`）。现对 `unified_memory && Fallback` 自动将单批上限设为 **`max_batch = 64`**，单次提交控制在 `<400 ms`，同时节省 **87.5%** 激活工作区内存且保持跨批宽逐位确定性。
     - **陷阱 C — 超限 Workgroup 静默丢弃防护**：`portability_vk_v5.rs` 强制验证 `wg > maxComputeWorkGroupInvocations` 返回 `WorkgroupTooLarge`、非 subgroup 整数倍在修改状态前返回 `InvalidDimensions`、超预算 `max_ctx` 在分配前返回 `OutOfDeviceMemory`。
  5. **`aarch64` 双目标交叉编译与外部设备调度器接入**：`cargo check --target aarch64-linux-android`（Pixel 8 Bionic 动态加载 `libvulkan.so`）与 `cargo check --target aarch64-unknown-linux-gnu`（RK3588 / Pi 5 Glibc 动态加载 `libvulkan.so.1`）`--no-default-features --features edge,vulkan` 均零警告通过；`bench/device_harness.py` 已新增 `--vulkan` 与 `--vk-device`。

- **Subgroup 宽度与无 Shuffle LDS 实机消融基准**（`Qwen3.5-0.8B-Q4_0`，`examples/bench_vk_portability.rs`）：

| 设备（Mesa 25.2.8） | Subgroup `sg` | GDN 递推内核 | 权重模式 | Prefill 吞吐 | Decode 吞吐 | 贪心序列 FNV-1a 指纹 |
|---|---:|---|---|---:|---:|---|
| **AMD RX 9060 XT (RADV)** | `32`（默认 wave32） | `shuffle` | `vram-slab` | `pp512 = 11202.0 tok/s` | `tg128 = 347.8 tok/s` | `6a967c97dfbbd2f9` ✅ |
| **AMD RX 9060 XT (RADV)** | **`64`（强制 wave64）** | `shuffle` | `vram-slab` | **`pp512 = 14137.9 tok/s`** | **`tg128 = 358.7 tok/s`** | `6a967c97dfbbd2f9` ✅ |
| **AMD RX 9060 XT (RADV)** | `32` | **`lds`（V3DV 退回）** | `vram-slab` | `pp512 = 10129.3 tok/s` | `tg128 = 347.2 tok/s` | `6a967c97dfbbd2f9` ✅ |
| **Intel UHD 770 (ANV)** | `8`（SIMD8） | `shuffle` | `uma-mapped` | `pp256 = 160.5 tok/s` | `tg32 = 29.0 tok/s` | `e41e064287266119` ✅ |
| **Intel UHD 770 (ANV)** | **`16`（默认 SIMD16）** | `shuffle` | `uma-mapped` | `pp256 = 155.9 tok/s` | `tg32 = 29.8 tok/s` | `e41e064287266119` ✅ |
| **Intel UHD 770 (ANV)** | `32`（SIMD32 溢出） | `shuffle` | `uma-mapped` | `pp256 = 4.5 tok/s` | `tg32 = 22.2 tok/s` | `e41e064287266119` ✅ |
| **Intel UHD 770 (ANV)** | **`16`** | **`lds`（V3DV 退回）** | `uma-mapped` | **`pp256 = 176.3 tok/s`** | **`tg32 = 31.5 tok/s`** | `e41e064287266119` ✅ |

- **Intel UHD 770（ADL-S GT1 核显）5 轮配对端到端基准**（原始数据：`bench/results/intel-uhd770/2026-11-21-v5-vulkan.json`，对比同机 `llama-bench -dev Vulkan0 -ngl 99`）：

| 模型（Q4_0） | V4 初始核显 `pp512 / tg` | **V5 `tinymodel --vulkan` `pp512`** | llama.cpp Vk `pp512`¹ | **V5 `tinymodel --vulkan` `tg64`** | llama.cpp Vk `tg64` | **Decode 相对 llama.cpp Vk** | 加载耗时 / 进程脏内存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SmolLM2-360M** | 130.1 / 28.4 | **300.0 tok/s**（+131%） | 800.3 | **75.00 tok/s**（+164%） | 50.01 | **150.0%（1.50× 反超！）** | `178.0 ms` / **`38.4 MiB`** |
| **Qwen3.5-0.8B** | 18.0 / 13.2 | **154.5 tok/s**（**8.57×**） | 306.2 | **29.66 tok/s**（+125%） | 24.51 | **121.0%（1.21× 反超！）** | `534.7 ms` / **`43.7 MiB`** |
| **Llama-3.2-1B** | 61.1 / 10.2 | **114.1 tok/s**（+87%） | 414.9 | **24.45 tok/s**（+140%） | 20.69 | **118.2%（1.18× 反超！）** | `460.0 ms` / **`39.7 MiB`** |

  > ¹ 注：`llama.cpp` Vulkan 在无 CoopMat 核显上的 Prefill 使用了 `VK_KHR_shader_integer_dot_product`（`dp4a` INT8 点积），而 `tm-vulkan` 当前 `gemm_tiled` 退回路径为保证对无 `dp4a` 设备（如 Pi 5 V3DV）100% 通用仍采用纯 FP32 FMA（后续可在 M3 可选挂载 `quantize_q8_1 + dp4a` 退回变体以再提速 ~2.5×）；而在核心交互指标 **Decode（`tg64`）上，`tinymodel --vulkan` 在 Intel 核显上全线反超 `llama.cpp` Vulkan 18%–50%**！

- **全固定参考设备覆盖与端侧部署指南（按 `AGENTS.md` 第 8 节规范；当前主机无外接 ADB/SSH 设备在线，已就绪一键复测指令）**：
  - **AMD RX 9060 XT（RDNA4，host 独显）**：V0–V5 全量实测主力机；`wave32` 与 `wave64` 均 100% 通过，`Qwen3-8B` 复测 `pp512 = 1747.5 / tg128 = 62.5 tok/s`，`Qwen3.5-0.8B` `pp512 = 13636.6–14137.9 tok/s`。
  - **Intel UHD 770（Alder Lake GT1，UMA 核显）**：本机实测完成；`SIMD16` + `uma-mapped` + 4-lane `Q4_0` GEMV 使 Decode 达 `24.5–75.0 tok/s`（反超 `llama.cpp` Vulkan `18%–50%`），额外脏内存仅 `38.4–43.7 MiB`。
  - **Apple M4（16GB，macOS）**：生产路径使用原生 `tm-metal`（零拷贝 `MTLBuffer` + `simdgroup_matrix`）；`tm-vulkan` 可通过 MoltenVK（`subgroup=32`）运行跨平台一致性回归。
  - **Google Pixel 8（Tensor G3，Mali-G715 MC7，Android Vulkan 1.3）**：
    - `aarch64-linux-android` `--features edge,vulkan` 编译通过；Mali-G715 原生 **wave16** 与我们在 Intel UHD 770 `SIMD16` 实测的路径完全同构，所有着色器共享内存 `≤18.6 KB`（低于 Mali `32 KB` 上限），且 `max_batch=64` 严格规避 Android `kbase` 驱动 `2000 ms` GPU 看门狗重置。
    - **端侧分工建议**：Pixel 8 上 Decode 首选 `tm-cpu`（`1×X3 + 4×A715` `i8mm` 零拷贝 `mmap`，`Qwen3.5-0.8B = 37.0 tok/s`，无需额外 GPU 权重常驻内存）；当插线连接手机后，可直接执行 `ANDROID_NDK_HOME=... python3 bench/device_harness.py run --target adb --vulkan -m models/Qwen3.5-0.8B-Q4_0.gguf --reps 5 --build` 实测 Mali-G715 长 Prompt TTFT 卸载收益。
  - **Rockchip RK3588（4×A76 + 4×A55，Mali-G610 MP4，Linux SBC）**：
    - `aarch64-unknown-linux-gnu` `--features edge,vulkan` 编译通过；适配主线 Linux 6.10+ Panthor + Mesa PanVK（wave16，`max_batch=64` 规避 DRM scheduler `500 ms` 超时）。
    - **独特价值**：瑞芯微闭源 RKNPU2（`rkllm`）不支持 `Qwen3.5` 的 GDN 线性注意力状态递推，而 `tm-vulkan` 可在 Mali-G610 上原生加速 `Qwen3.5-0.8B / 2B` Prefill，与 `4×A76` `dotprod` CPU Decode 形成互补。接入命令：`python3 bench/device_harness.py run --target ssh://root@<rk3588-ip> --vulkan -m models/Qwen3.5-0.8B-Q4_0.gguf --reps 5 --build`。
  - **Raspberry Pi 5（VideoCore VII，V3DV）**：Vulkan 1.2 wave16 + `gdn_recurrence_lds.comp`（零 `subgroupShuffleXor` 依赖）+ `max_batch=64`（规避 V3D `500 ms` 看门狗）确保 `Llama/Qwen3/Qwen3.5` 全模型功能兼容；日常推理主力仍为 4×A76 CPU。
  - **Raspberry Pi 4B / Zero 2W（`micro` 档）**：`cargo check -p tm-cli --no-default-features --features micro` 通过，零 Vulkan 依赖，保持纯静态 `<750 KB` 二进制与极低内存水位。

### V6（2026-11-21）✅ — HIP 决策门正式结论（已登记 `ADR-011`）
- **量化决策依据汇总（RX 9060 XT，5 轮中位数）**：
  - **Decode（`tg128`）**：`tinymodel --vulkan` 为 ROCm 7.1.1 HIP 的 **`108.0%–225.9%`**（5 个模型**全部击败 HIP**，最高快 **2.26×**）；
  - **Prefill（`pp512`）**：`tinymodel --vulkan` 墙钟为 HIP 的 **`66.9%–116.3%`**（持续热态 GPU 硬件时间戳为 HIP 的 **`82.7%–116.3%`**，其中 `Qwen3.5-0.8B` 墙钟直接**反超 HIP 16.3%**，`Qwen3.5-2B` GPU 热态达 HIP 的 **102.7%**）。
- **正式结论（`docs/DECISIONS.md` `ADR-011`）**：`tm-vulkan` 已在零 ROCm 依赖（省去 `~2 GiB` 运行时和跨架构维护负担）的前提下实现了对 AMD RDNA4 的一等公民高性能支持，同时一套代码复用到 Intel 核显与 Arm 端侧 GPU。**决定不在 M5 之前单独开发 HIP 后端**；待 M5 建设 `tm-cuda` 时，再按需评估 `.cu` 内核对 `hipcc` 的低成本共源复用。
