# 运行时环境变量登记表（`SETTINGS.md`）

本文件按 `AGENTS.md` §6 登记 `tinymodel` 中所有可配置环境变量、默认值、用途与状态。原则上尽量少加开关，所有开关必须保持确定性与向后兼容默认行为。

| 环境变量 | 默认值 | 适用后端 / Crate | 用途说明 | 状态 |
|---|---|---|---|---|
| `TM_FORCE_ISA` | 未设置（自动按 CPU 硬件能力选择最宽 ISA） | `tm-cpu` | 强制降档 CPU 内核分派表（可选值：`generic`、`neon`、`dotprod`、`i8mm`、`avx2`、`avx512vnni`），用于对齐测试与跨档位基准对比 | Active |
| `TM_CUDA_DEVICE` | 未设置（默认设备 `0`） | `tm-cuda` | 按设备序号或 GPU 名称子串选择 CUDA 设备 | Active |
| `TM_CUDA_MOE_HOST_OFFLOAD` | 未设置（VRAM 充足时 `0` 层；VRAM 不足时自动计算最少卸载层数） | `tm-cuda` | 控制稀疏 MoE 专家权重（`ffn_{gate,up,down}_exps`）通过 `cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP)` 驻留主机锁页 DDR 并由 GPU PCIe/C2C 零拷贝按需读取的层数：`0`/`off` 禁用，`all` 卸载全部 MoE 层，或填正整数 `N` 卸载末尾 `N` 层 | Active |
| `TM_VK_DEVICE` | 未设置（优先独立 GPU，其次集成 GPU） | `tm-vulkan` | 按设备序号或 Vulkan 设备名子串选择 GPU | Active |
| `TM_VK_DP4A` | `auto` | `tm-vulkan` | 控制 Vulkan INT8 DP4A 解码内核开关（`auto` / `force` / `off`） | Active |
