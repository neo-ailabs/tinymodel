# 环境变量与开关登记表（Settings）

> 按 `AGENTS.md` 第 6 节约定，仓库内支持的所有环境变量统一登记在本表，尽量少加开关。

| 环境变量 | 默认值 | 用途 | 状态 |
|---|---|---|---|
| `TM_FORCE_ISA` | 自动探测最宽可用 ISA | 强制 `tm-cpu` 降档到指定 ISA（如 `generic`、`neon`、`dotprod`、`avx2`），用于对齐测试与消融基准 | 稳定 |
| `TM_VK_DEVICE` | 自动选最优设备（独显 > 核显 > 虚拟 > CPU） | 按设备序号或名称/驱动子串选择 Vulkan 设备（如 `radv`、`UHD`、`llvmpipe`） | 稳定 |
| `TM_VK_PREFILL` | `auto` | Vulkan 批处理预填充策略：`auto`、`f16`（`CoopMatF16`）、`i8`（`CoopMatI8`）、`fallback` | 稳定 |
| `TM_VK_VRAM_MIB` | 自动（`device_local - headroom`） | 覆盖 `Inkling2VkRunner` 的显存预算（单位 MiB），用于混合卸载分级测试 | 稳定 |
| `TM_VK_HYBRID_GPU_EXPS` | 独显 `clamp(top_k/4, 1, 3)`，UMA `0` | 混合卸载 MoE 层中，通过零拷贝 PCIe 与主机 CPU 并发求值的每 token 专家数（`0..top_k`） | 稳定 |
| `TM_VK_DRAFT_VOCAB` | `0`（全词表） | `DFlashVkRunner` 草稿投影裁剪词表大小（`0` 扫描完整词表；设为正整数如 `65536` 时按 `\|\|markov_w1[id]\|\|_2` 重要性范数 + ASCII 头尾特殊 token 排序筛选高频子集） | 实验 |
| `TM_VK_DP4A` | `auto`（硬件支持 `int8_dot_accelerated` 时自动为 4/5/6-bit 权重及融合 `Q8_1` 路径启用） | 控制 Vulkan 解码 GEMV 是否启用 `OpSDotKHR`（`dp4a`）整数点积路径（覆盖 `Q4_0`/`Q4_1`/`Q8_0`/`Q5_K`/`Q6_K`；`0` 强制关闭回退 FP32；`1` 额外对纯 `Q8_0` 循环模型启用独立 `inkling2_q8_ops` 融合激活量化） | 稳定 |
| `TM_VK_PROF` | `0`（关闭） | `Inkling2VkRunner` 性能剖析：`1` 记录零开销静态 CB 下的分段墙钟耗时；`2` 额外注入 `vkCmdWriteTimestamp` 采集逐阶段硬件耗时 | 调试 |
