# Google Pixel 8（Tensor G3）设备档案与分析

> 日期：2026-11-20  
> 这是固定参考设备之一（见 `AGENTS.md` §8）。今后凡是涉及性能、内存、部署、微调、延迟的分析，都要附上 Pixel 8 这一列。  
> ⚠️ **本文所有吞吐和带宽数字都是估算**，还没有在真机上测过（当前没有连接设备）。规格里标“待核实”的项，要用真机上的 `/proc/cpuinfo`、`getauxval` 和 STREAM 结果确认。

---

## 1. 硬件规格

| 项 | Pixel 8 | 对比 Pi 5 |
|---|---|---|
| SoC | Google Tensor G3（三星 4nm） | BCM2712（16nm） |
| CPU | **1× Cortex-X3 @ 2.91 GHz + 4× Cortex-A715 @ 2.37 GHz + 4× Cortex-A510 @ 1.70 GHz**（9 核，三种核） | 4× Cortex-A76 @ 2.4 GHz（同构） |
| ISA | **ARMv9.0-A**：NEON、dotprod、**i8mm**、BF16、**FP16 运算**、SVE2（向量长度 128 位，相对 NEON 没有宽度优势） | ARMv8.2-A：NEON、dotprod、FP16；无 i8mm |
| 内存 | **8 GB LPDDR5X**（Pixel 8 Pro 为 12 GB） | 4/8/16 GB LPDDR4X |
| 带宽 | 理论约 50–68 GB/s（待核实）；CPU 实际可持续读取**估计 25–40 GB/s** | 约 11.5 GB/s |
| GPU | Arm **Mali-G715 MC7**，Vulkan 1.3 | VideoCore VII |
| NPU | Tensor G3 TPU：**只能通过 Android AICore / Gemini Nano 使用，第三方原生代码用不了**；NNAPI 已废弃 | 无 |
| 散热 | 被动散热，**Tensor G3 持续满载会明显降频**（估计几分钟后掉到峰值的 50–70%） | 可以加主动风扇 |

**结论**：Pixel 8 的带宽是 Pi 5 的 2–3 倍，还有 i8mm 和 FP16 运算，峰值性能远高于 Pi 5。但它有两个难点：**三种核混合调度**和**温控降频**，必须专门处理。

---

## 2. `tinymodel` 在 Pixel 8 上的适配情况（代码审计）

| 项 | 现状 | 问题与建议 |
|---|---|---|
| ISA 探测 | [isa.rs](../../crates/tm-cpu/src/isa.rs) 用的是 `std::arch::is_aarch64_feature_detected!`，在 Android 上也能用 | ✅ 应该会自动选中 `Isa::NeonI8mm`，A510 也支持 i8mm |
| **线程池分块** | [ops.rs](../../crates/tm-cpu/src/ops.rs) 的 `matvec_f32` 按“每个线程一块”切分（`rows.div_ceil(n_thr)`），这是为同构核设计的 | ❌ **在 Pixel 8 上最慢的 A510 会拖住整体**。建议：① 默认只用 5 个线程（X3 + 4×A715）；② 异构机器上把块切细（每线程 4–8 块），靠现有的原子认领做动态负载均衡；③ 可选按核绑定（`sched_setaffinity`）。这一点也适用于大多数手机 |
| 内存探测 | [memprobe.rs](../../crates/tm-core/src/memprobe.rs) 已开启 `#[cfg(any(target_os = "linux", target_os = "android"))]` 并支持从 `/proc/self/status` 读取 `VmHWM` 峰值内存 | ✅ 已修复 Android 目标条件，并在 SELinux 拦截 cgroup 文件时自动回退只读 `/proc/meminfo`；注意 Android LMK 会在 `MemAvailable` 耗尽前杀进程，仍需留足余量 |
| `mlock` | ARCHITECTURE 建议锁住大 arena | 普通 App 的 `RLIMIT_MEMLOCK` 一般只有 64 KB，**基本不能用**；只能靠 mmap 加页缓存 |
| 16 KB 页 | — | Android 15 起支持 16 KB 页内核（Pixel 8 可以在开发者选项里切换）。**ELF 要用 `-z max-page-size=16384` 链接**，否则在 16 KB 内核上无法加载。GGUF 是整个文件做 mmap，不受影响 |
| GPU | 还没有 Vulkan 后端（M3） | Mali-G715 支持 Vulkan 1.3 和整数点积扩展；是否支持 cooperative matrix 待核实 |

### 一键自动化验证（已实现 `bench/device_harness.py`，不需要 NDK）
插上 Pixel 8 并开启 USB 调试（或 `adb connect <ip:port>`）后，直接运行统一外部设备测试接口即可自动完成交叉编译、按需推送、异构核识别与绑核、STREAM 带宽、温度/电池功耗采集和贪心对齐校验：
```sh
python3 bench/device_harness.py list
python3 bench/device_harness.py probe --target adb --build
# 排除 A510 小核（4×A715 + 1×X3）跑最高性能，或用 --cluster-policy mid-only 只跑 4×A715 省电模式：
python3 bench/device_harness.py run --target adb -m models/Qwen3.5-0.8B-Q4_0.gguf \
    --cluster-policy no-little --reps 5 --compare-local --build
```

---

## 3. 推理性能估算（Qwen3.5-0.8B-Q4_0 和 2B）

| 指标 | Pixel 8（估算） | Pi 5（估算） | Pi 4B（估算） | M4（实测） |
|---|---:|---:|---:|---:|
| decode 0.8B（受带宽限制） | **30–55 tok/s**，降频后更低 | 15–20 tok/s | 7.5–9 tok/s | CPU 112.8 / Metal 127.0 tok/s |
| decode 2B | 15–25 tok/s | 6–8 tok/s | 3.5–3.9 tok/s | CPU 51.2 / Metal 63.0 tok/s |
| prefill 0.8B（受算力限制，i8mm） | 100–250 tok/s | 40–80 tok/s | 28–35 tok/s | Metal 1536 tok/s |
| 可以放心跑的模型上限 | **2B–4B**（App 实际可用内存估计 3–4 GB） | 2B–4B | 0.8B–2B | 8B |

---

## 4. 首字延迟（TTFT）

- 292 token 的 prompt：prefill 估计 **1.2–3 s**。比 Pi 4B（8–10 s）快很多，但离 M4 Metal（214 ms）差得远。
- 收益最大的手段和其他设备一样：**前缀缓存加状态快照**。Qwen3.5 的快照约 18 MiB 加上每个 token 12 KB，恢复只要几毫秒。
- 手机特有的问题：
  - **冷启动**：App 被系统杀掉后，模型页可能被换出，重新打开时要从 UFS 闪存读。UFS 4.0 顺序读超过 2 GB/s，0.8B 模型的冷启动在 0.5 s 以内（估算）。
  - **首轮计算前先把线程放到大核上**，可以用 ADPF（Android Dynamic Performance Framework）的 hint 申请提频；否则调度器可能从小核开始慢慢升频，首字延迟会多出几百毫秒。

---

## 5. 微调

| 项 | Pixel 8 |
|---|---|
| 浮点算力 | FP32 峰值估计约 245 GFLOPS（X3 加 4×A715，不算 A510）。**有 FP16 运算**，反向可以用 FP16 计算、FP32 累加，但要校验精度 |
| 降频后的有效算力 | 约 70–100 GFLOPS FP32 |
| 可以跑的模型 | 0.8B 宽裕，2B 可以（脏内存约 250–400 MiB，加上模型文件） |
| 估算训练吞吐 | 0.8B **约 15–25 tok/s**，2B 约 6–10 tok/s |
| 适合做什么 | 充电且温度正常时，过夜做几百到几千条样本的个性化微调；10 万 token 约 1–2 小时 |
| 限制 | 后台训练会被 Android 限制，要用前台服务或 WorkManager 的充电约束；要能随时中断并恢复 |

---

## 6. 待办（真机到手后）
1. 用 musl 二进制在 adb shell 下跑 `bench`、`profile`，并测 STREAM 带宽，替换本文所有估算值。
2. 分别测 1/5/9 线程，以及“绑大核”和“不绑”的情况，验证三种核的调度问题。
3. 测持续 10 分钟满载下的降频曲线（读 `/sys/class/thermal`）。
4. 修复 `memprobe` 的 `target_os = "android"` 条件，以及 `matvec_f32` 在异构核上的分块策略（各自配测试）。
