# 运行时环境变量登记表（`SETTINGS.md`）

本文件按 `AGENTS.md` §6 登记 `tinymodel` 中所有可配置环境变量、默认值、用途与状态。原则上尽量少加开关，所有开关必须保持确定性与向后兼容默认行为。

| 环境变量 | 默认值 | 适用后端 / Crate | 用途说明 | 状态 |
|---|---|---|---|---|
| `TM_FORCE_ISA` | 未设置（自动按 CPU 硬件能力选择最宽 ISA） | `tm-cpu` | 强制降档 CPU 内核分派表（可选值：`generic`、`neon`、`dotprod`、`i8mm`、`avx2`、`avx512vnni`），用于对齐测试与跨档位基准对比 | Active |
| `TM_CUDA_DEVICE` | 未设置（默认设备 `0`） | `tm-cuda` | 按设备序号或 GPU 名称子串选择 CUDA 设备 | Active |
| `TM_CUDA_MOE_HOST_OFFLOAD` | 未设置（VRAM 充足时 `0` 层；VRAM 不足时自动计算最少卸载层数） | `tm-cuda` | 控制稀疏 MoE 专家权重（`ffn_{gate,up,down}_exps`）通过 `cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP)` 驻留主机锁页 DDR 并由 GPU PCIe/C2C 零拷贝按需读取的层数：`0`/`off` 禁用，`all` 卸载全部 MoE 层，或填正整数 `N` 卸载末尾 `N` 层 | Active |
| `TM_CUDA_MOE_CACHE_EXPERTS` | 未设置（默认 `0` 槽位，禁用） | `tm-cuda` | 当启用 `TM_CUDA_MOE_HOST_OFFLOAD` 零拷贝主机 DDR 专家卸载时，为每个被卸载的 MoE 层在 VRAM 中分配 `K` 个热点专家槽位（`0..=n_expert`），由 GPU 捕获安全的频率衰减老化内核 `moe_cache_tick_promote` 自动将高频专家异步拉入 VRAM 槽位，保持 `0-ULP` 逐位无损并降低 PCIe 往返带宽 | Active |
| `TM_VK_DEVICE` | 未设置（优先独立 GPU，其次集成 GPU） | `tm-vulkan` | 按设备序号或 Vulkan 设备名子串选择 GPU | Active |
| `TM_VK_DP4A` | `auto` | `tm-vulkan` | 控制 Vulkan INT8 DP4A 解码内核开关（`auto` / `force` / `off`） | Active |
| `TM_INKLING2_RAM_MIB` | 未设置（按系统可用内存与安全水位自动规划） | `tm-models` | 显式指定 `Inkling-2-Mini` 循环感知主机 RAM 专家层预算（单位 MiB），其余外层自动走 NVMe `O_DIRECT` 流式按需读取 | Active |
| `TM_INKLING2_CACHE_EXPERTS` | 未设置（自动利用剩余分段 RAM 预算分配 `0..=32` 槽位/层） | `tm-models` | 为每个 `NvmeDirect` 外层 MoE 分配频率衰减老化（LFU-aging）热点专家主机 RAM 缓存槽位数（`0` 禁用），保持 `0-ULP` 逐位一致并消除高频专家重复 SSD 读取 | Active |
| `TM_CUDA_INKLING2_VRAM_MIB` | 未设置（自动按可用 VRAM 减去安全余量） | `tm-cuda` | 显式覆盖 `Inkling2CudaRunner` 三级（VRAM + `DEVICEMAP` 零拷贝主机 RAM + NVMe `O_DIRECT`）循环感知专家放置的 VRAM 预算（单位 MiB） | Active |
| `TM_CUDA_HYBRID_GPU_EXPS` | 未设置（独立 PCIe GPU 默认 `2`） | `tm-cuda` | 在 `Inkling2CudaRunner` 混合分级模式下，控制每个零拷贝 `DEVICEMAP` 主机 RAM 专家层由 GPU PCIe 并行分担的 top-k 专家槽位数（其余 `k - k_gpu` 个专家同时在 CPU DDR 上运行） | Active |

