#!/usr/bin/env python3
"""QLoRA fine-tuning script for Qwen2.5-Coder-7B-Instruct on LEGO Compact Grid DSL.

Designed for a single 24GB GPU (RTX 4090 / A10G / L40S):
  - Base model: Qwen/Qwen2.5-Coder-7B-Instruct (4-bit NF4 double quant)
  - LoRA: r=32, alpha=64 on all linear projections (q/k/v/o/gate/up/down_proj)
  - Loss masking: assistant turns only (supports multi-turn DRC self-repair SFT)
  - Export: saves LoRA adapter (can be merged & converted to GGUF for `tinymodel`)

Requirements on GPU machine:
  pip install torch transformers peft trl datasets bitsandbytes accelerate

Example:
  python3 tools/lego/train_qwen25_coder_7b.py \
      --train-jsonl data/lego/train.jsonl \
      --val-jsonl data/lego/val.jsonl \
      --output-dir runs/qwen25-coder-7b-lego-lora \
      --epochs 4 --max-seq-len 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QLoRA SFT for Qwen2.5-Coder-7B-Instruct on LEGO DSL")
    ap.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID or local path",
    )
    ap.add_argument("--train-jsonl", type=Path, required=True, help="Training messages JSONL")
    ap.add_argument("--val-jsonl", type=Path, default=None, help="Optional validation messages JSONL")
    ap.add_argument("--output-dir", type=Path, required=True, help="Checkpoint & LoRA output directory")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch-size", type=int, default=2, help="Per-device micro batch size")
    ap.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA (use bf16 LoRA on >=48GB GPU)")
    ap.add_argument("--merge-and-save", action="store_true", help="Merge LoRA into base weights after training")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            f"Missing GPU training dependency: {exc}\n"
            "Install on your GPU host with:\n"
            "  pip install torch transformers peft datasets bitsandbytes accelerate"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_and_tokenize(jsonl_path: Path) -> Dataset:
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj["messages"]
                full_ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False
                )
                labels = [-100] * len(full_ids)

                # Mask only assistant completions for loss (handles multi-turn DRC repair cleanly)
                for idx, m in enumerate(msgs):
                    if m.get("role") != "assistant":
                        continue
                    s = len(
                        tokenizer.apply_chat_template(
                            msgs[:idx], tokenize=True, add_generation_prompt=True
                        )
                    )
                    e = len(
                        tokenizer.apply_chat_template(
                            msgs[: idx + 1], tokenize=True, add_generation_prompt=False
                        )
                    )
                    for pos in range(min(s, len(full_ids)), min(e, len(full_ids))):
                        labels[pos] = full_ids[pos]

                full_ids = full_ids[: args.max_seq_len]
                labels = labels[: args.max_seq_len]
                if any(lbl != -100 for lbl in labels):
                    rows.append(
                        {
                            "input_ids": full_ids,
                            "attention_mask": [1] * len(full_ids),
                            "labels": labels,
                        }
                    )
        return Dataset.from_list(rows)

    train_ds = load_and_tokenize(args.train_jsonl)
    val_ds = load_and_tokenize(args.val_jsonl) if (args.val_jsonl and args.val_jsonl.exists()) else None
    print(f"[train] loaded {len(train_ds)} train samples" + (f", {len(val_ds)} val samples" if val_ds else ""))

    quant_cfg = None
    if not args.no_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    pad_id = tokenizer.pad_token_id

    def collate_fn(batch: list[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn_mask, labels = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attn_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=args.bf16,
        fp16=not args.bf16,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        save_total_limit=2,
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()

    adapter_dir = args.output_dir / "lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[train] saved LoRA adapter -> {adapter_dir}")

    if args.merge_and_save:
        merged_dir = args.output_dir / "merged_hf"
        base = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        from peft import PeftModel

        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[train] saved merged HF weights -> {merged_dir}")
        print("[train] convert to GGUF for tinymodel via llama.cpp convert_hf_to_gguf.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
