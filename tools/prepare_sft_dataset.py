#!/usr/bin/env python3
"""Prepare chat / instruction SFT datasets into `TMSFT` v1 binary (`ADR-015`).

Supports:
  1. HuggingFace `transformers` tokenizer (`--hf-model <id_or_dir>`) for exact
     Jinja `apply_chat_template` assistant-turn loss masking on `{"messages": [...]}` JSONL.
  2. Pure-stdlib conversion of pre-tokenized `{"tokens": [...], "loss_mask": [...]}` JSONL
     into compact `TMSFT` v1 binary (`b"TMSF"` magic) for zero-overhead on-device training.

Usage examples:
  # With HuggingFace chat template (masks system + user + template headers with 0, assistant with 1):
  python3 tools/prepare_sft_dataset.py --hf-model HuggingFaceTB/SmolLM2-360M-Instruct \
      -i sft_messages.jsonl -o train.tmsft --max-seq-len 256

  # From pre-tokenized JSONL (pure Python stdlib):
  python3 tools/prepare_sft_dataset.py -i pretok.jsonl -o train.tmsft --max-seq-len 256
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

TMSFT_MAGIC = b"TMSF"
TMSFT_VERSION = 1


def write_tmsft(out_path: Path, records: list[tuple[list[int], list[int]]]) -> int:
    with open(out_path, "wb") as f:
        f.write(TMSFT_MAGIC)
        f.write(struct.pack("<II", TMSFT_VERSION, len(records)))
        for toks, mask in records:
            assert len(toks) == len(mask)
            n = len(toks)
            f.write(struct.pack("<I", n))
            f.write(struct.pack(f"<{n}I", *toks))
            f.write(bytes(1 if m else 0 for m in mask))
        return f.tell()


def encode_messages_with_hf(tokenizer, messages: list[dict], max_seq_len: int) -> tuple[list[int], list[int]]:
    full_ids: list[int] = list(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    mask = [0] * len(full_ids)

    # Compute exact prefix boundaries around assistant messages so only assistant completions get mask=1
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_msgs = messages[:idx]
        start_ids = list(
            tokenizer.apply_chat_template(prefix_msgs, tokenize=True, add_generation_prompt=True)
        )
        end_ids = list(
            tokenizer.apply_chat_template(messages[: idx + 1], tokenize=True, add_generation_prompt=False)
        )
        s = min(len(start_ids), len(full_ids))
        e = min(len(end_ids), len(full_ids))
        for p in range(s, e):
            mask[p] = 1

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        mask = mask[:max_seq_len]
    if mask:
        mask[0] = 0
    return full_ids, mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert SFT JSONL into TMSFT v1 binary")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input JSONL file")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output .tmsft binary")
    ap.add_argument("--hf-model", help="Optional HuggingFace tokenizer ID or local directory")
    ap.add_argument("--max-seq-len", type=int, default=256, help="Max tokens per sample")
    args = ap.parse_args()

    hf_tok = None
    if args.hf_model:
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(args.hf_model)

    records: list[tuple[list[int], list[int]]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "tokens" in obj:
                toks = [int(x) for x in obj["tokens"]][: args.max_seq_len]
                mask = [int(x) for x in obj.get("loss_mask", [1] * len(toks))][: len(toks)]
                if mask:
                    mask[0] = 0
            elif "messages" in obj:
                if hf_tok is None:
                    raise SystemExit(
                        f"Line {line_no}: 'messages' format requires --hf-model <tokenizer>"
                    )
                toks, mask = encode_messages_with_hf(hf_tok, obj["messages"], args.max_seq_len)
            elif "prompt" in obj and "completion" in obj:
                if hf_tok is None:
                    raise SystemExit(
                        f"Line {line_no}: raw text JSONL without --hf-model can be passed directly to `tinymodel finetune --data {args.input}`!"
                    )
                p_ids = list(hf_tok.encode(obj["prompt"], add_special_tokens=True))
                c_ids = list(hf_tok.encode(obj["completion"], add_special_tokens=False))
                if hf_tok.eos_token_id is not None and (not c_ids or c_ids[-1] != hf_tok.eos_token_id):
                    c_ids.append(int(hf_tok.eos_token_id))
                toks = (p_ids + c_ids)[: args.max_seq_len]
                mask = ([0] * len(p_ids) + [1] * len(c_ids))[: len(toks)]
                if mask:
                    mask[0] = 0
            else:
                raise SystemExit(f"Line {line_no}: unsupported JSONL schema")

            if len(toks) >= 2 and sum(mask[1:]) > 0:
                records.append((toks, mask))

    n_bytes = write_tmsft(args.output, records)
    total_toks = sum(len(t) for t, _ in records)
    sup_toks = sum(sum(m[1:]) for _, m in records)
    print(
        f"[prepare_sft_dataset] wrote {len(records)} samples ({total_toks} tokens, "
        f"{sup_toks} supervised) -> {args.output} ({n_bytes} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
