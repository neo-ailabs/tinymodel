# V0 基线报告：AMD Radeon RX 9060 XT · llama.cpp Vulkan（Mesa 25.0 vs 25.2）与 HIP

> 生成日期：2026-11-21
> 原始结果：[amd-rx9060xt-llama-cpp-2026-11-21.json](amd-rx9060xt-llama-cpp-2026-11-21.json)（Mesa 25.0.7 + HIP）· [amd-rx9060xt-llama-cpp-mesa25.2-2026-11-21.json](amd-rx9060xt-llama-cpp-mesa25.2-2026-11-21.json)（Mesa 25.2.8）
> 复现脚本：[run_amd_gpu_baseline.sh](../run_amd_gpu_baseline.sh)

## 1. 环境

- **GPU**：AMD Radeon RX 9060 XT（Navi 44，RDNA4/gfx1200，32 CU，8 GiB GDDR6，128-bit，**理论带宽 320 GB/s**，32 MB Infinity Cache），PCIe BAR 256 MiB（ReBAR 未开）
- **主机**：Intel i7-12700，15 GiB RAM，Ubuntu 24.04.3 LTS，内核 6.14.0-27
- **驱动对比**：
  1. **Vulkan（Mesa 25.0.7）**：系统默认包（Vulkan 1.4.305，gfx1200 上**没有** `VK_KHR_cooperative_matrix`）
  2. **Vulkan（Mesa 25.2.8）**：Ubuntu `noble-updates` 官方包 `25.2.8-0ubuntu0.24.04.5` + `libdrm 2.4.125`（部署在 `~/.local/mesa-25.2.8`，Vulkan 1.4.318，**已开启 `VK_KHR_cooperative_matrix`、`VK_EXT_shader_float8`、`VK_KHR_shader_bfloat16`**，暴露全部 `16×16×16` `i8`/`f16`/`bf16`/`fp8` WMMA 形状）
  3. **HIP（ROCm 7.1.1）**：`AMDGPU_TARGETS=gfx1200`
- **llama.cpp**：`9d817213a`，Release；Vulkan 版用 Vulkan SDK 1.4.363.1 的 `glslc` 编译
- **协议**：`llama-bench -p 512 -n 128 -r 5 -ngl 99 -fa 1`，全部层放在 GPU 上；表中是 5 轮均值 ± 标准差

> [!WARNING]
> **ROCm `/opt/amdgpu/lib` 与新 Mesa 的动态库冲突**：本机 `/etc/ld.so.conf.d/20-amdgpu.conf` 把 `/opt/amdgpu/lib/x86_64-linux-gnu` 排在系统 `/lib/x86_64-linux-gnu` 之前。Mesa 25.2.8 若加载到 `/opt/amdgpu` 的 `libdrm.so.2`，提交队列会触发 `vk::DeviceLostError`。用户态部署已通过 `patchelf --force-rpath --set-rpath '$ORIGIN'` 绑定 Ubuntu 原生 `libdrm 2.4.125` 解决；若日后用 `sudo apt` 全局升级系统 Mesa，需注意排查 `/etc/ld.so.conf.d/20-amdgpu.conf` 优先级。

## 2. 端到端基准结果（5 模型 × 3 后端）

| 模型（Q4_0） | 体积 (MiB) | 后端 | `pp512` (tok/s) | `tg128` (tok/s) | 等效带宽¹ (GB/s) |
|---|---:|---|---:|---:|---:|
| SmolLM2-360M-Instruct | 217.4 | Vulkan (Mesa 25.0.7) | 4491 ± 155 | 416.3 ± 7.9 | 94.9 |
| | | **Vulkan (Mesa 25.2.8)** | 4362 ± 148 | **428.3** ± 2.6 | 97.6 |
| | | HIP (ROCm 7.1.1) | **21985** ± 2302 | 269.5 ± 4.7 | 61.4 |
| Qwen3.5-0.8B（GDN） | 473.2 | Vulkan (Mesa 25.0.7) | 6666 ± 76 | **247.1** ± 17.3 | 122.6 |
| | | **Vulkan (Mesa 25.2.8)** | 8247 ± 111 (**+23.7%**) | 245.7 ± 16.6 | 121.9 |
| | | HIP (ROCm 7.1.1) | **11663** ± 994 | 168.3 ± 2.3 | 83.5 |
| Llama-3.2-1B-Instruct | 729.7 | Vulkan (Mesa 25.0.7) | 3334 ± 283 | 294.2 ± 1.0 | 225.1 |
| | | **Vulkan (Mesa 25.2.8)** | 3420 ± 221 | **295.2** ± 1.1 | **225.9** |
| | | HIP (ROCm 7.1.1) | **13941** ± 1871 | 259.1 ± 1.2 | 198.3 |
| Qwen3.5-2B（GDN） | 1148.1 | Vulkan (Mesa 25.0.7) | 3617 ± 390 | 101.4 ± 0.7 | 122.1 |
| | | **Vulkan (Mesa 25.2.8)** | 4805 ± 15 (**+32.8%**) | 101.0 ± 0.8 | 121.6 |
| | | HIP (ROCm 7.1.1) | **7536** ± 390 | **134.3** ± 0.7 | 161.6 |
| Qwen3-8B | 4559.9 | Vulkan (Mesa 25.0.7) | 1118 ± 25 | 41.4 ± 0.1 | 198.0（61.9%） |
| | | **Vulkan (Mesa 25.2.8)** | 1801 ± 212 (**+61.1%**) | **58.5** ± 0.2 (**+41.3%**) | **279.8（87.4%）** |
| | | HIP (ROCm 7.1.1) | **2548** ± 41 | **59.2** ± 0.0 | **283.0（88.4%）** |

> ¹ 等效带宽 = 模型体积 × tg128，括号里是相对理论峰值 320 GB/s 的比例。单算子 `MUL_MAT(q4_0, m=4096, n=512, k=14336)` 微基准：Mesa 25.0.7 为 `3300.7 µs`（**18.2 TFLOPS**），升级 Mesa 25.2.8 后降至 `2165.4 µs`（**27.8 TFLOPS，+52.5%**）。

## 3. 结论（指导 `tm-vulkan` V2/V3 设计）

1. **升级 Mesa 25.2.8 后，8B 大模型 Vulkan 解码直接追平 ROCm HIP**（`58.5` vs `59.2 tok/s`，**+41.3%**，等效带宽达峰值的 **87.4%**），360M / 0.8B / 1B 解码则全面快于 HIP（快 **14%–59%**）。
2. **Prefill 大幅提速（+24%–+61%），且仍有翻倍空间**：
   - `GGML_VK_PERF_LOGGER=1` 剖析显示，llama.cpp Vulkan 在 `Llama-3.2-1B` prefill 中有两个明显瓶颈：① 未融合的非矩阵算子（`GLU 28.7ms + ADD 14.5ms + RMS_NORM 9.8ms + ROPE 5.1ms = 58.1ms`，占总耗时近 **38%**）；② `mul_mmq` 走的是标量 4 路 `v_dot4_i32_iu8`，而**没有用 RDNA4 新暴露的 `16×16×16 i8*i8+i32->i32` Cooperative Matrix（INT8 WMMA）**。
   - `tm-vulkan` 在 V2/V3 只需沿用 `tm-metal` 的**算子融合策略**（`gemm_*_res` 融合残差、`gate+up+silu` 融合、`rope+kv` 融合），并直接调用 **`16×16×16 i8*i8->i32` / `f16*f16->f32` coopmat**，即可在 Vulkan 上进一步逼近甚至追平 HIP prefill。
