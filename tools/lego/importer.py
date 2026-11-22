"""Batch importer for real-world LEGO model archives (.ldr, .mpd, BrickLink .io, OMR JSONL).

Filters models by part-catalog coverage and physical DRC validation, converts them
into Compact Grid DSL, and splits into leakage-free train/val SFT JSONL files
formatted for Qwen2.5-Coder-7B-Instruct chat fine-tuning.
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .dataset import SYSTEM_PROMPT, make_corrupted_variant
from .ldraw import LegoModel, parse_ldraw, to_dsl
from .parts import PART_CATALOG
from .validator import validate_model


@dataclass
class ImportStats:
    total_seen: int = 0
    parsed_ok: int = 0
    skipped_unsupported_parts: int = 0
    skipped_drc_failed: int = 0
    skipped_brick_bounds: int = 0
    accepted: int = 0


def _extract_ldraw_from_io(io_path: Path) -> Optional[str]:
    """Extract the embedded model.ldr text from a BrickLink Studio .io (zip) archive."""
    with zipfile.ZipFile(io_path, "r") as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith((".ldr", ".mpd"))]
        if not candidates:
            return None
        # Prefer model.ldr if present
        candidates.sort(key=lambda n: (0 if n.lower().endswith("model.ldr") else 1, len(n)))
        return zf.read(candidates[0]).decode("utf-8", errors="replace")


def _extract_primary_mpd_submodel(mpd_text: str) -> str:
    """For multi-part .mpd files, extract the first (main) `0 FILE ...` block."""
    lines = mpd_text.splitlines()
    file_indices = [i for i, ln in enumerate(lines) if ln.strip().upper().startswith("0 FILE ")]
    if len(file_indices) <= 1:
        return mpd_text
    return "\n".join(lines[file_indices[0] : file_indices[1]])


def iter_raw_ldraw_sources(input_path: Path) -> Iterable[Tuple[str, str, str]]:
    """Yield (name, description, ldraw_text) tuples from a file, directory, or OMR JSONL."""
    if input_path.is_file():
        suf = input_path.suffix.lower()
        if suf in (".jsonl", ".json"):
            with open(input_path, "r", encoding="utf-8") as f:
                for idx, raw in enumerate(f, 1):
                    line = raw.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    # Compatible with DylanRiden/sets_lego_omr_full and custom OMR JSONL
                    ldr = (
                        obj.get("mpd")
                        or obj.get("ldr")
                        or obj.get("ldraw")
                        or obj.get("content")
                        or ""
                    )
                    if not ldr:
                        continue
                    name = str(obj.get("set_num") or obj.get("name") or f"omr_{idx}")
                    desc = str(
                        obj.get("description")
                        or obj.get("set_name")
                        or obj.get("name")
                        or name
                    )
                    if obj.get("theme"):
                        desc = f"{desc} (Theme: {obj['theme']})"
                    yield name, desc, _extract_primary_mpd_submodel(ldr)
            return
        if suf == ".io":
            txt = _extract_ldraw_from_io(input_path)
            if txt:
                yield input_path.stem, input_path.stem, _extract_primary_mpd_submodel(txt)
            return
        if suf in (".ldr", ".mpd"):
            txt = input_path.read_text(encoding="utf-8", errors="replace")
            yield input_path.stem, input_path.stem, _extract_primary_mpd_submodel(txt)
            return

    if input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in (".ldr", ".mpd"):
                txt = p.read_text(encoding="utf-8", errors="replace")
                yield p.stem, p.stem, _extract_primary_mpd_submodel(txt)
            elif suf == ".io":
                txt = _extract_ldraw_from_io(p)
                if txt:
                    yield p.stem, p.stem, _extract_primary_mpd_submodel(txt)


def _check_ldraw_part_coverage(ldraw_text: str) -> Tuple[int, int]:
    """Return (supported_count, total_type1_lines) before full parsing."""
    supported = 0
    total = 0
    for raw in ldraw_text.splitlines():
        line = raw.strip()
        if not line.startswith("1 "):
            continue
        toks = line.split()
        if len(toks) < 15:
            continue
        total += 1
        part_id = " ".join(toks[14:]).lower().removesuffix(".dat")
        if part_id in PART_CATALOG:
            supported += 1
    return supported, total


def import_to_sft(
    input_path: Path,
    out_dir: Path,
    min_bricks: int = 4,
    max_bricks: int = 250,
    require_drc_pass: bool = True,
    val_ratio: float = 0.1,
    include_repair_turns: bool = True,
    seed: int = 42,
) -> ImportStats:
    """Import real LDraw/.io/OMR sources into Qwen2.5-Coder-7B train/val SFT JSONL."""
    stats = ImportStats()
    accepted_models: List[LegoModel] = []

    for name, desc, ldraw_text in iter_raw_ldraw_sources(input_path):
        stats.total_seen += 1
        sup, tot = _check_ldraw_part_coverage(ldraw_text)
        if tot == 0 or sup < tot:
            stats.skipped_unsupported_parts += 1
            continue
        try:
            model = parse_ldraw(ldraw_text, default_name=name).normalize_origin()
            if desc and not model.description:
                model.description = desc
            stats.parsed_ok += 1
        except Exception:
            stats.skipped_unsupported_parts += 1
            continue

        if not (min_bricks <= len(model.bricks) <= max_bricks):
            stats.skipped_brick_bounds += 1
            continue

        if require_drc_pass:
            rep = validate_model(model)
            if not rep.passed:
                stats.skipped_drc_failed += 1
                continue

        accepted_models.append(model)
        stats.accepted += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(accepted_models)

    n_val = max(1, int(len(accepted_models) * val_ratio)) if len(accepted_models) >= 2 else 0
    val_models = accepted_models[:n_val]
    train_models = accepted_models[n_val:]

    def _write_split(models: List[LegoModel], path: Path, add_repairs: bool) -> int:
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for m in models:
                dsl_text = to_dsl(m)
                bom_str = ", ".join(f"{k}x{v}" for k, v in m.bom().items())
                user_prompt = (
                    f"Design a LEGO build: {m.description or m.name}\n"
                    f"Available BOM: {bom_str}\n"
                    f"Output Compact Grid DSL."
                )
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": dsl_text},
                            ],
                            "meta": {"name": m.name, "bricks": len(m.bricks), "bom": m.bom()},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1

                if add_repairs:
                    mode = rng.choice(("collision", "floating", "disconnected"))
                    broken = make_corrupted_variant(m, mode)
                    b_rep = validate_model(broken)
                    if not b_rep.passed:
                        repair_prompt = (
                            f"Your previous draft failed LEGO DRC validation:\n"
                            f"{b_rep.format_feedback()}\n"
                            f"Fix all errors and output the corrected Compact Grid DSL."
                        )
                        f.write(
                            json.dumps(
                                {
                                    "messages": [
                                        {"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": user_prompt},
                                        {"role": "assistant", "content": to_dsl(broken)},
                                        {"role": "user", "content": repair_prompt},
                                        {"role": "assistant", "content": dsl_text},
                                    ],
                                    "meta": {"name": f"{m.name}_repair_{mode}", "bricks": len(m.bricks)},
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
        return count

    _write_split(train_models, out_dir / "train.jsonl", add_repairs=include_repair_turns)
    _write_split(val_models, out_dir / "val.jsonl", add_repairs=False)

    summary = {
        "total_seen": stats.total_seen,
        "parsed_ok": stats.parsed_ok,
        "skipped_unsupported_parts": stats.skipped_unsupported_parts,
        "skipped_brick_bounds": stats.skipped_brick_bounds,
        "skipped_drc_failed": stats.skipped_drc_failed,
        "accepted": stats.accepted,
        "train_models": len(train_models),
        "val_models": len(val_models),
    }
    (out_dir / "import_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats
