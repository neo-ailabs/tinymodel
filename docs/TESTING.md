# 测试与基准规范（Testing & Benchmarking）

## 1. 正确性分层

| 层级 | 内容 | 位置 | 何时运行 |
|---|---|---|---|
| L0 单元 | GGUF 解析（含损坏输入的 fuzz 测试）、量化块的打包和解包、tokenizer | 各 crate 的 `tests/` | 每次提交 |
| **L1 内核对齐** | 每个（格式 × ISA × 内核）组合都和 `generic` 标量参考实现**逐位比对**；随机量化块覆盖所有码本索引和符号组合；用 2 的幂作 scale，保证浮点运算精确 | `crates/tm-cpu/tests/parity_*.rs` | 每次提交（只跑本机 ISA；可以用 `TM_FORCE_ISA` 降档跑） |
| L1b 批宽不变性 | 同一个 token 单独算，和放进 1..8 个 token 的批里算，结果必须逐位相同 | 同上 | 每次提交 |
| L2 oracle | 新格式或重排布局先在 Python 里实现参考版，并验证 `dequant(ours) == dequant(ggml)` | `tools/oracle/` | 新增格式时 |
| **L3 端到端** | 贪心解码输出和 llama.cpp 逐 token 一致；**多 token** logits 的 KL 散度和 top-1 一致率；GDN 等复杂算子要和 FP64 参考实现比对 | `tools/logits_diff.py` | 新增模型和发布前 |
| L4 量化质量 | 以 Q8_0 为基准测 top-1 一致率和 KL；token id 文件和代码一起提交 | `tools/quant_quality.py`、`bench/data/quality/` | 新增量化格式时 |
| L5 服务 | 工具调用语料的切片测试（整段、逐字节、3/7 字节分块、每个字节处切成两半）；聊天模板 golden（和 HF 的渲染结果对比）；stop 回扣 | `crates/tm-server/tests/` | 每次提交 |

> [!IMPORTANT]
> 只测单 token 发现不了位置不前进的 bug；某个模块单独测过、却从未被接入调用，也会导致整个模型算错。L3 必须是**多 token 的端到端**测试。

## 2. 微基准
- 每个内核的对齐测试同时提供 bench 模式（`cargo bench -p tm-cpu` 或 `--bench`）。
- GEMV 输出**带宽利用率**（实际 GB/s ÷ 在同一台机器上实测的 STREAM 带宽）。

## 3. 端到端基准协议
1. **冻结 prompt 语料**并记录 SHA-256，manifest 对不上就拒绝运行。
2. **同一台机器**上和 llama.cpp 交替运行，至少 5 轮，取中位数；两个引擎都通过 CLI 或 OpenAI HTTP 接口调用，输出同名的 timings 字段（pp tok/s、tg tok/s、TTFT）。
3. 分别记录：加载时间（冷启动和热启动）、内存峰值（RSS）、温度和频率（树莓派必须装主动散热）。
4. 结果 JSON 带 schema 版本，存入 `bench/results/<device>/<date>-<commit>.json`。
5. 声称有性能提升，必须附上同一台机器的基线。

## 4. 设备矩阵与外部设备测试接口

| 设备 | 角色 | 接入方式 | CI |
|---|---|---|---|
| Apple M4 16GB | 主力开发；aarch64 内核、Metal | `--target local` | 本地 |
| **Google Pixel 8（Tensor G3）** | edge 档手机基准（异构 1×X3 + 4×A715 + 4×A510、i8mm、功耗） | `--target adb`（USB / 无线 ADB） | 手动（已实测） |
| **Rockchip RK3588** | edge 档大内存 SBC 基准（4×A76 + 4×A55、64-bit LPDDR4X/5、Mali-G610） | `--target ssh://root@<host>` | 手动 |
| Raspberry Pi 5 | edge 档 SBC 基准（4×A76、dotprod） | `--target ssh://pi@<host>` | 手动或自托管 runner |
| Raspberry Pi 4B | micro ISA 大内存基准（4×A72、纯 ARMv8.0 NEON） | `--target ssh://pi@<host> --profile micro` | 手动 |
| Raspberry Pi Zero 2W | micro 档极限内存基准（4×A53、512MB） | `--target ssh://pi@<host> --profile micro` | 手动 |
| **x86 + AMD RX 9060 XT**（RDNA4，8 GiB） | Vulkan 后端主力开发机（RADV）；同机还有 Intel UHD 770（ANV）和 lavapipe | `--target local` | 本地 |
| x86 + NVIDIA | AVX2 / AVX-512、CUDA | `--target ssh://...` | 远期 |
| CI（GitHub Actions） | x86_64 和 aarch64 Linux 的 L0/L1/L5 测试、交叉编译检查、GPU 后端只做编译检查 | — | 每次提交 |

### 外部设备一键测试命令（`bench/device_harness.py` + `tinymodel probe`）
1. **端侧单文件自包含探针**（无需目标机装 Python / 编译器）：
   - `tinymodel probe [-t <threads>] [--stream-mib <mib>] [--json]`：输出 ISA 能力、大小核拓扑及 `taskset` 掩码、`MemAvailable`/cgroup v2 水位、L1 内核逐位自检、内存安全的多线程 STREAM 持续读写带宽、温度与电池功率。
   - `tinymodel bench -m <model.gguf> [-f bench/data/prompt_short.txt] --json`：输出 `load_ms`、`ttft_ms`、`pp_tps`、`tg_tps`、有效带宽、`VmHWM` 峰值内存、前后温度/功率及贪心 token 序列指纹。
2. **宿主机统一调度器**：
   ```sh
   python3 bench/device_harness.py list
   python3 bench/device_harness.py probe --target adb --build
   python3 bench/device_harness.py run --target adb -m models/Qwen3.5-0.8B-Q4_0.gguf --cluster-policy no-little --reps 5 --compare-local --build
   python3 bench/device_harness.py run --target adb --vulkan -m models/Qwen3.5-0.8B-Q4_0.gguf --reps 5 --build
   python3 bench/device_harness.py run --target ssh://root@rk3588.local --vulkan -m models/Qwen3.5-0.8B-Q4_0.gguf --reps 5 --build
   python3 bench/device_harness.py run --target ssh://pi@raspberrypi.local --profile micro -m models/SmolLM2-360M-Instruct-Q4_0.gguf --reps 5 --build
   ```

## 5. Vulkan 后端（`tm-vulkan`）

- **对齐容差**：GPU 浮点累加顺序和标量参考实现不同，按 `|got − ref| / max(Σ|w·x|, 1) ≤ 1e-5` 判定（和 Metal 一致）；整数点积路径要求逐位一致。
- **设备与 Subgroup 全覆盖**：本机用户态已部署 **Mesa 25.2.8**（`~/.local/mesa-25.2.8/env.sh`，开启 RDNA4 `VK_KHR_cooperative_matrix`）。每次改 shader，都要在三种驱动上跑完整测试套件（含 `parity_vk_q4_0`、`parity_vk_ops`、`parity_vk_coopmat`、`parity_vk_qwen35` FP64 GDN oracle 及 **`portability_vk_v5`**）：`TM_VK_DEVICE=radv`（RADV wave32 + 强制 wave64 + coopmat）、`TM_VK_DEVICE=UHD`（ANV SIMD8 / **SIMD16** / SIMD32 + `uma-mapped`）、`TM_VK_DEVICE=llvmpipe`（lavapipe wave8 + `zero-copy-umap`，CI 用），并校验 `gdn_recurrence_lds`（Pi 5 `V3DV` 无 Shuffle 退回）`0.0` 逐位一致及超限 Workgroup / 超预算显存前置报错。
  ```sh
  . ~/.local/mesa-25.2.8/env.sh
  for d in radv UHD llvmpipe; do TM_VK_DEVICE=$d cargo test -p tm-vulkan; done
  ```
- **微基准与可移植性消融**：`cargo run --release -p tm-vulkan --example bench_vk_gemv_q4_0 [wg nr]`（Decode GEMV）、`cargo run --release -p tm-vulkan --example bench_vk_gemm`（V3 Prefill）及 `cargo run --release -p tm-vulkan --example bench_vk_portability [model.gguf]`（V5 Subgroup `8/16/32/64` 与 `shuffle`/`lds` 消融）。可通过 `TM_VK_PREFILL=auto|f16|i8|fallback` 切换端到端 Prefill 路径。RX 9060 XT 有 32 MB Infinity Cache，**小于 32 MB 的形状测的是缓存带宽**；带宽利用率要看 ≥64 MB 的形状（例如 `65536x4096`、`151936x1024`）。
- **基线**：`bench/run_amd_gpu_baseline.sh <out.json>`（llama.cpp Vulkan + HIP，`-p 512 -n 128 -r 5 -ngl 99 -fa 1`）。
- **Shader 编译**：改了 `shaders/*.comp` 之后运行 `crates/tm-vulkan/gen_spv.sh`，把重新生成的 `spv/` 一起提交。
