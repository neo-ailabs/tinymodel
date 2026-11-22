"""Unit and integration tests for the LEGO domain prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.lego.dataset import (
    build_mini_house,
    default_templates,
    generate_dataset,
    make_corrupted_variant,
)
from tools.lego.ldraw import LegoModel, PlacedBrick, parse_dsl, parse_ldraw, to_dsl, to_ldraw
from tools.lego.validator import validate_model


def test_all_default_templates_pass_drc_and_roundtrip() -> None:
    for model in default_templates():
        rep = validate_model(model)
        assert rep.passed, f"{model.name} failed DRC:\n{rep.format_feedback()}"
        assert rep.collision_count == 0
        assert rep.connected_components == 1
        assert rep.largest_component_ratio == 1.0
        assert rep.grounded
        assert rep.com_stable
        assert rep.step_buildable

        # Round-trip DSL
        dsl_text = to_dsl(model)
        reparsed_dsl = parse_dsl(dsl_text)
        assert reparsed_dsl.bricks == model.bricks

        # Round-trip LDraw (.ldr)
        ldr_text = to_ldraw(model)
        reparsed_ldr = parse_ldraw(ldr_text)
        assert reparsed_ldr.bricks == model.bricks
        assert validate_model(reparsed_ldr).passed


def test_collision_detection() -> None:
    m = LegoModel(
        name="overlap",
        bricks=[
            PlacedBrick("3001", 4, 0, 0, 0, rot=0, step=1),
            PlacedBrick("3003", 1, 1, 0, 0, rot=0, step=1),  # overlaps at x=1..2
        ],
    )
    rep = validate_model(m)
    assert not rep.passed
    assert rep.collision_count == 1
    assert any(i.code == "COLLISION" for i in rep.issues)


def test_tile_blocks_top_stud_connection() -> None:
    """A brick resting on top of a smooth tile (no studs) is NOT connected."""
    m = LegoModel(
        name="on_tile",
        bricks=[
            PlacedBrick("3068b", 15, 0, 0, 0, rot=0, step=1),  # 2x2 smooth tile (no top studs)
            PlacedBrick("3003", 4, 0, 1, 0, rot=0, step=2),  # 2x2 brick sitting on tile
        ],
    )
    rep = validate_model(m)
    assert not rep.passed
    assert rep.connected_components == 2
    assert any(i.code == "DISCONNECTED" for i in rep.issues)
    assert any(i.code == "FLOATING_STEP" for i in rep.issues)


def test_corrupted_variants_caught_by_drc() -> None:
    house = build_mini_house()
    for mode, expected_code in [
        ("collision", "COLLISION"),
        ("floating", "FLOATING"),
        ("disconnected", "DISCONNECTED"),
    ]:
        broken = make_corrupted_variant(house, mode)
        rep = validate_model(broken)
        assert not rep.passed
        assert any(i.code == expected_code for i in rep.issues), (
            f"Expected {expected_code} for mode={mode}, got {rep.issues}"
        )


def test_bom_limit_enforcement() -> None:
    house = build_mini_house()
    exact_bom = house.bom()
    assert validate_model(house, bom_limit=exact_bom).passed

    tight_bom = dict(exact_bom)
    first_part = next(iter(tight_bom))
    tight_bom[first_part] -= 1
    rep = validate_model(house, bom_limit=tight_bom)
    assert not rep.passed
    assert not rep.bom_ok
    assert any(i.code == "BOM_EXCEEDED" for i in rep.issues)


def test_corner_brick_rotation_and_l_shape_fit() -> None:
    """Two 2x2 corner bricks (2357) interlocking in a 2x2 square + 1x1 fill without collision."""
    m = LegoModel(
        name="corner_fit",
        bricks=[
            PlacedBrick("3022", 71, 0, 0, 0, rot=0, step=1),  # 2x2 plate base
            PlacedBrick("2357", 4, 0, 1, 0, rot=0, step=2),  # occupies (0,0),(1,0),(0,1)
            PlacedBrick("3005", 14, 1, 1, 1, rot=0, step=2),  # fits cleanly in empty (1,1)
        ],
    )
    rep = validate_model(m)
    assert rep.passed, rep.format_feedback()


def test_dataset_generation_outputs_valid_jsonl(tmp_path: Path) -> None:
    stats = generate_dataset(tmp_path, seed=7, extra_variants=4)
    assert stats["models"] == 10
    assert stats["sft_samples"] == 40  # 10 base + 30 DRC repair multi-turn samples
    assert stats["dpo_pairs"] == 30

    for fname in ("sft_messages.jsonl", "sft_prompt_completion.jsonl", "dpo_pairs.jsonl"):
        p = tmp_path / fname
        assert p.exists()
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) > 0

    # Verify generated .ldr files reload and pass DRC
    ldr_files = list((tmp_path / "ldr").glob("*.ldr"))
    assert len(ldr_files) == 10
    for lf in ldr_files:
        m = parse_ldraw(lf.read_text(encoding="utf-8"), default_name=lf.stem)
        assert validate_model(m).passed


def test_importer_and_eval_drc_pipeline(tmp_path: Path) -> None:
    import zipfile
    from tools.lego.eval_drc import evaluate_predictions_file
    from tools.lego.importer import import_to_sft

    src_dir = tmp_path / "raw"
    src_dir.mkdir()
    templates = default_templates()
    for m in templates:
        (src_dir / f"{m.name}.ldr").write_text(to_ldraw(m), encoding="utf-8")

    # Also pack one as BrickLink Studio .io (zip)
    with zipfile.ZipFile(src_dir / "house_studio.io", "w") as zf:
        zf.writestr("model.ldr", to_ldraw(build_mini_house()))

    out_sft = tmp_path / "sft"
    stats = import_to_sft(src_dir, out_sft, val_ratio=0.2, seed=42)
    assert stats.total_seen == len(templates) + 1
    assert stats.accepted == len(templates) + 1
    assert (out_sft / "train.jsonl").exists()
    assert (out_sft / "val.jsonl").exists()

    # Score offline predictions (1 good wrapped in markdown fence + 1 broken collision)
    good = build_mini_house()
    broken = make_corrupted_variant(good, "collision")
    pred_path = tmp_path / "preds.jsonl"
    pred_path.write_text(
        "\n".join(
            [
                json.dumps({"name": "good", "completion": f"```dsl\n{to_dsl(good)}```", "bom": good.bom()}),
                json.dumps({"name": "bad", "completion": to_dsl(broken), "bom": good.bom()}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = evaluate_predictions_file(pred_path)
    assert summary["total"] == 2
    assert summary["parse_rate"] == 1.0
    assert summary["drc_pass_rate"] == 0.5
    assert summary["zero_collision_rate"] == 0.5

