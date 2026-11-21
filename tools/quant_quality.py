#!/usr/bin/env python3
"""L4 Quantization Quality Evaluation Tool (`docs/TESTING.md` §1 L4).

Measures multi-token KL divergence (`mean_KL`, `max_KL`), Top-1 agreement, Top-5 overlap,
and model footprint across low-bit quantization formats (`IQ4_NL`, `IQ3_XXS`, `Q2_0`,
`IQ2_XXS`, `TQ2_0`, `Q4_0x4`) against a `Q8_0` (or `F16`/`Q4_0`) reference GGUF using
frozen committed token IDs in `bench/data/quality/`.

Usage:
  python3 tools/quant_quality.py -m models/SmolLM2-360M-Instruct-Q4_0.gguf \
      --tokens-file bench/data/quality/eval_tokens_smollm2.txt \
      --formats q4_0x4,iq4_nl,iq3_xxs,q2_0,iq2_xxs,tq2_0
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from logits_diff import compare_dumps, read_tmlg


def load_frozen_tokens(path: Path) -> list[int]:
    tokens: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for part in line.split(","):
            p = part.strip()
            if p:
                tokens.append(int(p))
    if len(tokens) < 4:
        raise ValueError(f"Expected >= 4 tokens in {path}, got {len(tokens)}")
    return tokens


def ensure_tinymodel_bin(repo_root: Path) -> Path:
    bin_path = repo_root / "target" / "release" / "tinymodel"
    if not bin_path.exists():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "tm-cli"],
            cwd=repo_root,
            check=True,
        )
    return bin_path


def run_logits(tm_bin: Path, model_path: Path, tok_csv: str, out_bin: Path, threads: int = 4) -> None:
    subprocess.run(
        [
            str(tm_bin),
            "logits",
            "-m",
            str(model_path),
            "-o",
            str(out_bin),
            "--tokens",
            tok_csv,
            "-t",
            str(threads),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="L4 quantization quality evaluator (KL & Top-1 vs reference)")
    ap.add_argument("-m", "--model", required=True, help="Source GGUF model (Q8_0/F16/Q4_0)")
    ap.add_argument(
        "--ref-model",
        default=None,
        help="Optional higher-precision reference GGUF (defaults to --model converted to q8_0 or --model directly)",
    )
    ap.add_argument(
        "--tokens-file",
        default="bench/data/quality/eval_tokens_smollm2.txt",
        help="Committed frozen token ID file",
    )
    ap.add_argument(
        "--formats",
        default="q4_0x4,iq4_nl,iq3_xxs,q2_0,iq2_xxs,tq2_0",
        help="Comma-separated target --quant formats to evaluate",
    )
    ap.add_argument(
        "--lm-head-quant",
        default=None,
        help="Optional --lm-head-quant override (e.g. q8_0 or iq4_nl to preserve LM head precision)",
    )
    ap.add_argument("-t", "--threads", type=int, default=4)
    ap.add_argument("--json-out", default=None, help="Optional JSON summary output path")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    src_model = Path(args.model).resolve()
    tokens_file = (repo_root / args.tokens_file).resolve() if not Path(args.tokens_file).is_absolute() else Path(args.tokens_file)
    tokens = load_frozen_tokens(tokens_file)
    tok_csv = ",".join(str(t) for t in tokens)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    tm_bin = ensure_tinymodel_bin(repo_root)

    with tempfile.TemporaryDirectory(prefix="tm_l4_qual_") as tmpdir:
        tmp = Path(tmpdir)
        if args.ref_model:
            ref_gguf = Path(args.ref_model).resolve()
            ref_label = ref_gguf.name
        else:
            # Per `docs/TESTING.md` L4: use Q8_0 reference baseline
            ref_gguf = tmp / "ref_q8_0.gguf"
            subprocess.run(
                [str(tm_bin), "convert", "-i", str(src_model), "-o", str(ref_gguf), "--quant", "q8_0"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            ref_label = f"{src_model.stem} (Q8_0 baseline)"

        ref_logits_bin = tmp / "ref.tmlg"
        run_logits(tm_bin, ref_gguf, tok_csv, ref_logits_bin, args.threads)
        ref_toks, n_vocab, ref_mat = read_tmlg(ref_logits_bin)

        print(f"=== L4 Quantization Quality Report: {src_model.name} ===")
        print(f"Reference: {ref_label} | Tokens: {len(tokens)} ({tokens_file.name}) | Vocab: {n_vocab}")
        if args.lm_head_quant:
            print(f"LM Head Override: --lm-head-quant {args.lm_head_quant}")
        print(
            f"{'Format':<12} {'Size(MiB)':>10} {'Ratio':>8} {'Top-1%':>8} {'Top-5%':>8} {'Mean KL':>11} {'Max KL':>11} {'MAE':>9}"
        )
        print("-" * 84)

        src_mib = src_model.stat().st_size / (1024.0 * 1024.0)
        results = []

        for fmt in formats:
            cand_gguf = tmp / f"cand_{fmt}.gguf"
            cmd = [str(tm_bin), "convert", "-i", str(src_model), "-o", str(cand_gguf), "--quant", fmt]
            if args.lm_head_quant:
                cmd.extend(["--lm-head-quant", args.lm_head_quant])
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

            cand_mib = cand_gguf.stat().st_size / (1024.0 * 1024.0)
            cand_logits = tmp / f"cand_{fmt}.tmlg"
            run_logits(tm_bin, cand_gguf, tok_csv, cand_logits, args.threads)
            cand_toks, _, cand_mat = read_tmlg(cand_logits)

            # Reuse `compare_dumps` silently to get exact KL & Top-1/Top-5 metrics
            stats = compare_dumps(fmt, ref_toks, cand_toks, ref_mat, cand_mat, verbose=False)
            cand_gguf.unlink(missing_ok=True)
            cand_logits.unlink(missing_ok=True)

            row = {
                "format": fmt,
                "lm_head_quant": args.lm_head_quant,
                "size_mib": round(cand_mib, 2),
                "size_ratio_vs_src": round(cand_mib / max(src_mib, 1e-9), 4),
                "top1_agreement": round(stats["top1_rate"], 4),
                "top5_overlap": round(stats["top5_overlap"], 4),
                "mean_kl": round(stats["mean_kl"], 6),
                "max_kl": round(stats["max_kl"], 6),
                "mean_mae": round(stats["mean_mae"], 6),
            }
            results.append(row)
            print(
                f"{fmt:<12} {cand_mib:>10.2f} {100.0 * cand_mib / src_mib:>7.1f}% "
                f"{100.0 * stats['top1_rate']:>7.2f}% {100.0 * stats['top5_overlap']:>7.2f}% "
                f"{stats['mean_kl']:>11.5f} {stats['max_kl']:>11.5f} {stats['mean_mae']:>9.5f}"
            )

        if args.json_out:
            out_p = Path(args.json_out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_model": src_model.name,
                        "reference": ref_label,
                        "tokens_file": str(args.tokens_file),
                        "num_tokens": len(tokens),
                        "lm_head_quant": args.lm_head_quant,
                        "results": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
