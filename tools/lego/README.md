# LEGO 积木拼搭垂域模型原型 (`tools/lego/`)

纯 Python 标准库实现的乐高拼搭**数据预处理 + 物理/可拼搭性 DRC 校验 + SFT/DPO 数据集生成**原型，可直接对接 `tools/prepare_sft_dataset.py` 和 `tinymodel finetune`。

## 模块结构

| 文件 | 作用 |
|---|---|
| `parts.py` | 高频标准零件库（砖/板/光滑砖/坡面件/拐角砖），包含精确 LDU 尺寸、整数体素占位、顶部凸点（Stud）与底部管孔（Tube）连接点、4 方向离散旋转 |
| `ldraw.py` | 标准 LDraw (`.ldr`) 与模型友好的 **Compact Grid DSL**（Y 轴朝上、整数凸点/薄板栅格、`rot ∈ {0,1,2,3}`）双向无损编解码与原点归一化 |
| `validator.py` | 乐高版 **DRC 物理校验器**：三维体素穿模检测、凸点-管孔咬合连通图、悬空/接地与重心稳定性检查、分步可拼搭性校验、BOM 零件预算约束 |
| `dataset.py` | 6 类通过 DRC 校验的多步拼搭模板（金字塔、小屋、拱桥、小卡车、瞭望塔、楼梯）+ 自动生成穿模/悬空/孤岛负样本，导出 `sft_messages.jsonl`、`sft_prompt_completion.jsonl`、`dpo_pairs.jsonl` |
| `cli.py` | 命令行工具（`convert` / `validate` / `generate` / `repair-demo`） |
| `test_lego.py` | `pytest` 单元测试与端到端回归测试 |

## 快速使用

```bash
# 1. 运行测试
pytest tools/lego/test_lego.py -v

# 2. 生成验证通过的 .ldr / .dsl 样例、SFT 与 DPO 训练集
python3 -m tools.lego.cli generate --out-dir /tmp/lego_data --variants 12

# 3. 对单个模型跑物理 DRC 校验（支持 .ldr 和 .dsl）
python3 -m tools.lego.cli validate /tmp/lego_data/ldr/mini_house.ldr
python3 -m tools.lego.cli validate /tmp/lego_data/dsl/mini_house.dsl --json

# 4. LDraw 与 Compact DSL 互转（导出的 .ldr 可直接在 BrickLink Studio / LDView 打开）
python3 -m tools.lego.cli convert /tmp/lego_data/ldr/mini_house.ldr --normalize
python3 -m tools.lego.cli convert /tmp/lego_data/dsl/mini_house.dsl --to-ldraw -o /tmp/out.ldr

# 5. 查看 DRC 报错与修复闭环演示
python3 -m tools.lego.cli repair-demo
```

## 对接 `tinymodel` QLoRA 微调

生成的数据集可直接送入仓库已有的训练管线：

```bash
# 方式 A：直接喂给 tinymodel finetune（prompt/completion JSONL）
tinymodel finetune --model <base.gguf> --data /tmp/lego_data/sft_prompt_completion.jsonl ...

# 方式 B：用 HuggingFace 分词器转成 TMSFT v1 二进制（支持多轮 DRC 自修复 assistant mask）
python3 tools/prepare_sft_dataset.py --hf-model <hf_model_id> \
    -i /tmp/lego_data/sft_messages.jsonl -o /tmp/lego_data/train.tmsft --max-seq-len 512
```
