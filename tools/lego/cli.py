#!/usr/bin/env python3
"""CLI entrypoint for the LEGO domain model data & DRC validation prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import build_mini_house, generate_dataset, make_corrupted_variant
from .ldraw import load_model, to_dsl, to_ldraw
from .validator import validate_model


def cmd_convert(args: argparse.Namespace) -> int:
    model = load_model(args.input)
    if args.normalize:
        model = model.normalize_origin()
    if args.to_ldraw:
        out = to_ldraw(model)
    else:
        out = to_dsl(model)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding="utf-8")
        print(f"Wrote {len(model.bricks)} bricks -> {args.output}")
    else:
        sys.stdout.write(out)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    model = load_model(args.input)
    bom_limit = None
    if args.bom:
        bom_limit = json.loads(args.bom.read_text(encoding="utf-8"))
    report = validate_model(model, bom_limit=bom_limit)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.format_feedback())
    return 0 if report.passed else 1


def cmd_generate(args: argparse.Namespace) -> int:
    stats = generate_dataset(
        out_dir=args.out_dir,
        seed=args.seed,
        extra_variants=args.variants,
    )
    print(
        f"[lego-dataset] generated {stats['models']} verified builds, "
        f"{stats['sft_samples']} SFT samples, {stats['dpo_pairs']} DPO pairs -> {args.out_dir}"
    )
    return 0


def cmd_repair_demo(_args: argparse.Namespace) -> int:
    good = build_mini_house()
    broken = make_corrupted_variant(good, "collision")
    broken_report = validate_model(broken)
    good_report = validate_model(good)

    print("=== 1. Broken Draft DRC Check ===")
    print(broken_report.format_feedback())
    print("\n=== 2. After Applying Repair (Ground Truth) ===")
    print(good_report.format_feedback())
    return 0 if (not broken_report.passed and good_report.passed) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LEGO domain model preprocessing & DRC validation CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_conv = sub.add_parser("convert", help="Convert between LDraw (.ldr) and Compact Grid DSL (.dsl)")
    p_conv.add_argument("input", type=Path, help="Input .ldr or .dsl file")
    p_conv.add_argument("-o", "--output", type=Path, help="Output file (stdout if omitted)")
    p_conv.add_argument("--to-ldraw", action="store_true", help="Export to LDraw .ldr instead of Compact DSL")
    p_conv.add_argument("--normalize", action="store_true", help="Normalize origin to (0,0,0) and sort")
    p_conv.set_defaults(func=cmd_convert)

    p_val = sub.add_parser("validate", help="Run physical & buildability DRC checks on a LEGO model")
    p_val.add_argument("input", type=Path, help="Input .ldr or .dsl file")
    p_val.add_argument("--bom", type=Path, help="Optional JSON file mapping part_id -> max count")
    p_val.add_argument("--json", action="store_true", help="Output structured JSON report")
    p_val.set_defaults(func=cmd_validate)

    p_gen = sub.add_parser("generate", help="Generate verified builds, SFT JSONL, and DPO pairs")
    p_gen.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    p_gen.add_argument("--variants", type=int, default=12, help="Number of extra color-swapped variants")
    p_gen.set_defaults(func=cmd_generate)

    p_demo = sub.add_parser("repair-demo", help="Run a quick DRC error-feedback and repair demo")
    p_demo.set_defaults(func=cmd_repair_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
