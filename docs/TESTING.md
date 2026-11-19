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

## 4. 设备矩阵

| 设备 | 角色 | CI |
|---|---|---|
| Apple M4 16GB | 主力开发；aarch64 内核、Metal | 本地 |
| Raspberry Pi 5 | edge 档基准 | 手动或自托管 runner |
| Raspberry Pi Zero 2W | micro 档基准 | 手动 |
| x86 + NVIDIA | AVX2 / AVX-512、CUDA | 远期 |
| CI（GitHub Actions） | x86_64 和 aarch64 Linux 的 L0/L1/L5 测试、交叉编译检查、GPU 后端只做编译检查 | 每次提交 |
