#!/usr/bin/env python3
"""Blind DRC benchmark evaluator & iterative DRC self-repair loop for LEGO models.

Evaluates any OpenAI-compatible inference endpoint (vLLM, llama-server, Ollama, etc.)
or pre-generated predictions JSONL on a held-out LEGO validation set, reporting:
  - Syntax parse rate
  - Zero-collision rate
  - 100% stud-to-tube connectivity rate (analogous to PCB 100% net connectivity)
  - Full DRC pass rate (collision-free + connected + grounded + COM stable + buildable + BOM ok)
  - Optional multi-round DRC error-feedback repair gain (`--repair-rounds`)
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ldraw import parse_dsl
from .validator import ValidationReport, validate_model


def extract_dsl_block(raw_text: str) -> str:
    """Extract Compact Grid DSL from raw LLM completion (strips markdown fences if present)."""
    m = re.search(r"```(?:dsl|lego|text)?\s*\n(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    text = m.group(1) if m else raw_text
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith(("MODEL ", "DESC ", "STEP ", "BRICK ")):
            lines.append(s)
    return "\n".join(lines) + ("\n" if lines else "")


@dataclass
class EvalMetrics:
    total: int = 0
    parsed_ok: int = 0
    zero_collision: int = 0
    full_connectivity: int = 0
    bom_ok: int = 0
    drc_passed: int = 0

    def summary(self) -> Dict[str, float | int]:
        denom = max(self.total, 1)
        return {
            "total": self.total,
            "parse_rate": round(self.parsed_ok / denom, 4),
            "zero_collision_rate": round(self.zero_collision / denom, 4),
            "full_connectivity_rate": round(self.full_connectivity / denom, 4),
            "bom_ok_rate": round(self.bom_ok / denom, 4),
            "drc_pass_rate": round(self.drc_passed / denom, 4),
        }


def evaluate_prediction(
    completion: str,
    name: str,
    bom_limit: Optional[Dict[str, int]] = None,
) -> Tuple[bool, Optional[ValidationReport], str]:
    """Parse and DRC-validate a single LLM completion."""
    dsl = extract_dsl_block(completion)
    if not dsl.strip():
        return False, None, "Empty output: no MODEL/STEP/BRICK lines found."
    try:
        model = parse_dsl(dsl, default_name=name)
    except Exception as exc:
        return False, None, f"DSL syntax error: {exc}"
    rep = validate_model(model, bom_limit=bom_limit)
    return True, rep, rep.format_feedback()


def evaluate_predictions_file(pred_jsonl: Path) -> Dict[str, float | int]:
    """Score offline predictions JSONL: each line has {name, completion, optional bom}."""
    m = EvalMetrics()
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m.total += 1
            ok, rep, _ = evaluate_prediction(
                obj.get("completion", ""),
                name=str(obj.get("name", f"sample_{idx}")),
                bom_limit=obj.get("bom"),
            )
            if ok and rep is not None:
                m.parsed_ok += 1
                if rep.collision_count == 0:
                    m.zero_collision += 1
                if rep.connected_components == 1 and rep.largest_component_ratio == 1.0:
                    m.full_connectivity += 1
                if rep.bom_ok:
                    m.bom_ok += 1
                if rep.passed:
                    m.drc_passed += 1
    return m.summary()


def call_openai_chat(
    base_url: str,
    model: str,
    messages: List[dict],
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 1536,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate LEGO model completions with DRC checks")
    ap.add_argument("--predictions", type=Path, help="Offline predictions JSONL to score")
    ap.add_argument("--val-jsonl", type=Path, help="Validation messages JSONL for live API eval")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--repair-rounds", type=int, default=0, help="Iterative DRC self-repair rounds (0=zero-shot)")
    args = ap.parse_args()

    if args.predictions:
        print(json.dumps(evaluate_predictions_file(args.predictions), indent=2))
        return 0

    if not args.val_jsonl:
        raise SystemExit("Provide either --predictions <file.jsonl> or --val-jsonl <val.jsonl>")

    m = EvalMetrics()
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta", {})
            name = str(meta.get("name", f"val_{idx}"))
            bom = meta.get("bom")
            # Keep system + first user message
            msgs = [x for x in obj["messages"][:2]]

            m.total += 1
            ok, rep, feedback = False, None, ""
            for rnd in range(args.repair_rounds + 1):
                completion = call_openai_chat(args.base_url, args.model, msgs, api_key=args.api_key)
                ok, rep, feedback = evaluate_prediction(completion, name=name, bom_limit=bom)
                if ok and rep is not None and rep.passed:
                    break
                if rnd < args.repair_rounds:
                    msgs.append({"role": "assistant", "content": completion})
                    msgs.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous draft failed LEGO DRC validation:\n{feedback}\n"
                                "Fix all errors and output the corrected Compact Grid DSL."
                            ),
                        }
                    )

            if ok and rep is not None:
                m.parsed_ok += 1
                if rep.collision_count == 0:
                    m.zero_collision += 1
                if rep.connected_components == 1 and rep.largest_component_ratio == 1.0:
                    m.full_connectivity += 1
                if rep.bom_ok:
                    m.bom_ok += 1
                if rep.passed:
                    m.drc_passed += 1

    print(json.dumps(m.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
