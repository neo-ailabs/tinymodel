# 11 AMD GPU 支持：分析与规划

> 状态：路线已确认（ADR-010，2026-11-21）；**V0、V1、Mesa 25.2.8、V2 已完成**，见第 7 节。
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
| **V0 环境与基线** | 安装 `vulkan-tools`、`glslc`（需要 sudo，或者把 Vulkan SDK 解压到用户目录）；llama.cpp 编出 Vulkan 版和 HIP（gfx1200）版；在 RX 9060 XT 上测 SmolLM2-360M、Llama-3.2-1B、Qwen3.5-0.8B/2B、8B 级模型的 pp512/tg128；导出设备能力（coopmat、int dot、subgroup） | `bench/baselines/amd-rx9060xt-*.json` + `.md` | 0.5 天 |
| **V1 PoC** | `tm-vulkan` 骨架（实例、设备选择、内存、staging 上传、pipeline cache）；Q4_0 GEMV 三件套（kernel + 对齐测试 + bench） | GEMV 带宽 ≥90% llama.cpp Vulkan `mul_mat_vec`；lavapipe 上对齐测试通过 | 1–2 天 |
| **V2 Llama/Qwen3 decode** | 全部 decode kernel；预录制命令缓冲；CLI `--gpu vulkan` | 贪心输出和 CPU 一致；多 token KL 达标；tg ≥90% llama.cpp Vulkan | 3–4 天 |
| **V3 prefill** | coopmat GEMM（int8/fp16）+ 退回版本；批量 FA | pp512 ≥90% llama.cpp Vulkan；记录和 llama.cpp ROCm 的差距，作为 HIP 决策依据 | 3–4 天 |
| **V4 Qwen3.5 GDN** | GDN 系列 kernel（按 `GDN_CONTRACT`，注意 AGENTS.md 第 9 节的陷阱） | 多 token logits 比对通过 | 2 天 |
| **V5 跨设备** | Intel UHD 770（ANV）、Pixel 8 Mali-G715、RK3588 Mali-G610、Pi 5 V3DV | 各设备功能通过，测出性能数据并写进文档 | 2–3 天 |
| **V6 HIP 决策门** | 根据 V3 的差距决定：放到 M5 和 tm-cuda 共用源码，还是提前做 | 新增 ADR | — |

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
  - 下一步 **V3**：把 `gemm_tiled` 与批量注意力接入 Mesa 25.2.8 解锁的 **`16×16×16` INT8/FP16 `VK_KHR_cooperative_matrix`（WMMA）**，把 `pp512` 推高 5–8×。
