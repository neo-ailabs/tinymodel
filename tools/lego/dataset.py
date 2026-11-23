"""Synthetic LEGO model generator and SFT / DPO dataset exporter.

Generates physically verified, multi-step LEGO models (with semantic sub-module step labels)
plus deliberate negative/corrupted variants for DPO / DRC self-repair training.
Exports JSONL directly compatible with `tools/prepare_sft_dataset.py` and `tinymodel finetune`.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .ldraw import LegoModel, PlacedBrick, to_dsl, to_ldraw
from .validator import validate_model

SYSTEM_PROMPT = (
    "You are a LEGO brick assembly designer. Output a physically valid, step-by-step "
    "assembly in Compact Grid DSL (Y-up: x,z in studs, y in plates, rot in 0..3). "
    "Ensure zero collisions, 100% stud-to-tube connectivity, and step-by-step stability."
)


def build_stepped_pyramid(base_color: int = 14, trim_color: int = 19) -> LegoModel:
    """3-tier interlocking stepped pyramid on a 4x4 plate."""
    bricks = [
        # Step 1: 4x4 base plate
        PlacedBrick("3031", trim_color, 0, 0, 0, rot=0, step=1),
        # Step 2: 4x4 first brick layer (two 2x4 bricks)
        PlacedBrick("3001", base_color, 0, 1, 0, rot=0, step=2),
        PlacedBrick("3001", base_color, 0, 1, 2, rot=0, step=2),
        # Step 3: 2x2 second layer centered (interlocks across both 2x4 bricks)
        PlacedBrick("3003", base_color, 1, 4, 1, rot=0, step=3),
        # Step 4: capstone 2x2 tile
        PlacedBrick("3068b", trim_color, 1, 7, 1, rot=0, step=4),
    ]
    return LegoModel(
        name="stepped_pyramid",
        description="A 4x4 interlocking stepped pyramid with a smooth 2x2 capstone tile.",
        bricks=bricks,
        step_labels={1: "base_plate", 2: "lower_tier", 3: "upper_tier", 4: "capstone"},
    ).normalize_origin()


def build_mini_house(wall_color: int = 15, roof_color: int = 4, base_color: int = 2) -> LegoModel:
    """Mini 4x6 cottage with interlocking corner walls and 45-degree sloped roof."""
    bricks = [
        # Step 1: 6x4 base plate
        PlacedBrick("3032", base_color, 0, 0, 0, rot=0, step=1),
        # Step 2: Lower wall ring (y=1..3), leaving a 2-stud front doorway at z=3, x=2..3
        PlacedBrick("3009", wall_color, 0, 1, 0, rot=0, step=2),  # back wall 1x6
        PlacedBrick("3622", wall_color, 0, 1, 1, rot=1, step=2),  # left wall 1x3 along Z
        PlacedBrick("3622", wall_color, 5, 1, 1, rot=1, step=2),  # right wall 1x3 along Z
        PlacedBrick("3005", wall_color, 1, 1, 3, rot=0, step=2),  # front-left pillar
        PlacedBrick("3005", wall_color, 4, 1, 3, rot=0, step=2),  # front-right pillar
        # Step 3: Upper wall ring (y=4..6) with staggered joints bridging doorway
        PlacedBrick("3010", wall_color, 1, 4, 3, rot=0, step=3),  # front lintel 1x4
        PlacedBrick("3010", wall_color, 1, 4, 0, rot=0, step=3),  # back 1x4
        PlacedBrick("3010", wall_color, 0, 4, 0, rot=1, step=3),  # left 1x4 along Z
        PlacedBrick("3010", wall_color, 5, 4, 0, rot=1, step=3),  # right 1x4 along Z
        # Step 4: Roof tie plate layer (y=7)
        PlacedBrick("3032", roof_color, 0, 7, 0, rot=0, step=4),
        # Step 5: Pitched roof using 2x4 45-degree slopes
        PlacedBrick("3037", roof_color, 1, 8, 0, rot=2, step=5),  # back slope facing -Z
        PlacedBrick("3037", roof_color, 1, 8, 2, rot=0, step=5),  # front slope facing +Z
        # Step 6: Roof ridge tiles
        PlacedBrick("3069b", roof_color, 1, 11, 1, rot=0, step=6),
        PlacedBrick("3069b", roof_color, 3, 11, 1, rot=0, step=6),
    ]
    return LegoModel(
        name="mini_house",
        description="A 6x4 mini cottage with a front doorway, staggered walls, and a sloped red roof.",
        bricks=bricks,
        step_labels={
            1: "grass_base",
            2: "lower_walls_with_doorway",
            3: "upper_walls_and_lintel",
            4: "roof_deck",
            5: "pitched_roof_slopes",
            6: "roof_ridge",
        },
    ).normalize_origin()


def build_arch_bridge(pillar_color: int = 71, deck_color: int = 72, rail_color: int = 14) -> LegoModel:
    """Span bridge with two 2x2 stone pillars, a 2x8 spanning deck, and safety railings."""
    bricks = [
        # Step 1: Left & right pillar bases
        PlacedBrick("3003", pillar_color, 0, 0, 0, rot=0, step=1),
        PlacedBrick("3003", pillar_color, 6, 0, 0, rot=0, step=1),
        # Step 2: Second course of pillars
        PlacedBrick("3003", pillar_color, 0, 3, 0, rot=0, step=2),
        PlacedBrick("3003", pillar_color, 6, 3, 0, rot=0, step=2),
        # Step 3: 2x8 spanning bridge deck connecting both pillars
        PlacedBrick("3034", deck_color, 0, 6, 0, rot=0, step=3),
        # Step 4: Deck road tiles and end guard rails
        PlacedBrick("3068b", deck_color, 3, 7, 0, rot=0, step=4),
        PlacedBrick("3023", rail_color, 0, 7, 0, rot=0, step=4),
        PlacedBrick("3023", rail_color, 6, 7, 0, rot=0, step=4),
        PlacedBrick("3023", rail_color, 0, 7, 1, rot=0, step=4),
        PlacedBrick("3023", rail_color, 6, 7, 1, rot=0, step=4),
    ]
    return LegoModel(
        name="arch_bridge",
        description="An 8-stud span bridge supported by two 2x2 pillars with a 2x8 deck and yellow railings.",
        bricks=bricks,
        step_labels={
            1: "pillar_footings",
            2: "pillar_columns",
            3: "bridge_deck",
            4: "road_surface_and_railings",
        },
    ).normalize_origin()


def build_mini_truck(body_color: int = 1, cab_color: int = 15, chassis_color: int = 0) -> LegoModel:
    """Compact 2x8 mini pickup truck with chassis, cab, windshield slope, and cargo bed."""
    bricks = [
        # Step 1: Wheel-axle blocks (front and rear)
        PlacedBrick("3004", chassis_color, 1, 0, 0, rot=1, step=1),  # 1x2 along Z at x=1
        PlacedBrick("3004", chassis_color, 6, 0, 0, rot=1, step=1),  # 1x2 along Z at x=6
        # Step 2: 2x8 chassis frame spanning both axles
        PlacedBrick("3034", chassis_color, 0, 3, 0, rot=0, step=2),
        # Step 3: Lower body (hood + cab base + truck bed sides)
        PlacedBrick("3001", body_color, 0, 4, 0, rot=0, step=3),  # front 2x4 body
        PlacedBrick("3010", body_color, 4, 4, 0, rot=0, step=3),  # bed left wall
        PlacedBrick("3010", body_color, 4, 4, 1, rot=0, step=3),  # bed right wall
        # Step 4: Cab roof & windshield
        PlacedBrick("3039", 43, 1, 7, 0, rot=3, step=4),  # trans-blue 2x2 slope windshield
        PlacedBrick("3003", cab_color, 3, 7, 0, rot=0, step=4),  # cab rear 2x2
        PlacedBrick("3068b", cab_color, 3, 10, 0, rot=0, step=4),  # smooth cab roof tile
    ]
    return LegoModel(
        name="mini_truck",
        description="A 2x8 mini pickup truck with black axles, blue cargo bed, trans-blue windshield, and white cab.",
        bricks=bricks,
        step_labels={
            1: "axles",
            2: "chassis_frame",
            3: "lower_body_and_bed",
            4: "cab_and_windshield",
        },
    ).normalize_origin()


def build_watchtower(stone_color: int = 71, accent_color: int = 72, flag_color: int = 4) -> LegoModel:
    """4x4 medieval watchtower with corner battlements and a red beacon."""
    bricks = [
        # Step 1: 4x4 foundation plate
        PlacedBrick("3031", accent_color, 0, 0, 0, rot=0, step=1),
        # Step 2: Layer 1 (2x4 x 2 along X)
        PlacedBrick("3001", stone_color, 0, 1, 0, rot=0, step=2),
        PlacedBrick("3001", stone_color, 0, 1, 2, rot=0, step=2),
        # Step 3: Layer 2 (rotated 90 deg for strong cross-interlocking)
        PlacedBrick("3001", stone_color, 0, 4, 0, rot=1, step=3),
        PlacedBrick("3001", stone_color, 2, 4, 0, rot=1, step=3),
        # Step 4: Layer 3 (back to X orientation)
        PlacedBrick("3001", stone_color, 0, 7, 0, rot=0, step=4),
        PlacedBrick("3001", stone_color, 0, 7, 2, rot=0, step=4),
        # Step 5: Battlement platform
        PlacedBrick("3031", accent_color, 0, 10, 0, rot=0, step=5),
        # Step 6: 4 corner crenels + central beacon
        PlacedBrick("3005", stone_color, 0, 11, 0, rot=0, step=6),
        PlacedBrick("3005", stone_color, 3, 11, 0, rot=0, step=6),
        PlacedBrick("3005", stone_color, 0, 11, 3, rot=0, step=6),
        PlacedBrick("3005", stone_color, 3, 11, 3, rot=0, step=6),
        PlacedBrick("3062b", flag_color, 1, 11, 1, rot=0, step=6),
    ]
    return LegoModel(
        name="watchtower",
        description="A 4x4 interlocking stone watchtower with four corner battlements and a red round beacon.",
        bricks=bricks,
        step_labels={
            1: "foundation",
            2: "tower_tier_1",
            3: "tower_tier_2_cross",
            4: "tower_tier_3",
            5: "battlement_deck",
            6: "crenels_and_beacon",
        },
    ).normalize_origin()


def build_staircase(color_a: int = 19, color_b: int = 25) -> LegoModel:
    """4-step ascending interlocking staircase on a 6x4 base."""
    bricks = [
        # Step 1: 6x4 base plate
        PlacedBrick("3032", color_a, 0, 0, 0, rot=0, step=1),
        # Step 2: Full-width bottom tier (y=1..3, x=0..5, z=0..3)
        PlacedBrick("2456", color_b, 0, 1, 0, rot=0, step=2),
        PlacedBrick("2456", color_b, 0, 1, 2, rot=0, step=2),
        # Step 3: Second stair tier (y=4..6, x=2..5, z=0..3)
        PlacedBrick("3001", color_a, 2, 4, 0, rot=0, step=3),
        PlacedBrick("3001", color_a, 2, 4, 2, rot=0, step=3),
        # Step 4: Top stair tier (y=7..9, x=4..5, z=0..3)
        PlacedBrick("3001", color_b, 4, 7, 0, rot=1, step=4),
        # Step 5: Smooth top landing tiles
        PlacedBrick("3068b", color_a, 4, 10, 0, rot=0, step=5),
        PlacedBrick("3068b", color_a, 4, 10, 2, rot=0, step=5),
    ]
    return LegoModel(
        name="staircase",
        description="A 3-tier interlocking staircase with smooth landing tiles on a 6x4 base.",
        bricks=bricks,
        step_labels={
            1: "base_plate",
            2: "step_tier_1",
            3: "step_tier_2",
            4: "step_tier_3",
            5: "top_landing",
        },
    ).normalize_origin()


def make_corrupted_variant(model: LegoModel, mode: str) -> LegoModel:
    """Create a deliberately broken model variant for DPO / DRC self-repair training."""
    broken = copy.deepcopy(model)
    broken.name = f"{model.name}_broken_{mode}"
    if not broken.bricks:
        return broken

    if mode == "collision":
        # Duplicate the last brick at the exact same coordinates
        dup = copy.deepcopy(broken.bricks[-1])
        broken.bricks.append(dup)
    elif mode == "floating":
        # Shift the last brick upward into mid-air
        broken.bricks[-1].y += 6
    elif mode == "disconnected":
        # Add an isolated brick far away on the ground
        max_x = max(b.x for b in broken.bricks) + 10
        broken.bricks.append(PlacedBrick("3005", 4, max_x, 0, 0, rot=0, step=broken.bricks[-1].step))
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")
    return broken


def default_templates() -> List[LegoModel]:
    """Return the built-in suite of verified template builds."""
    return [
        build_stepped_pyramid(),
        build_mini_house(),
        build_arch_bridge(),
        build_mini_truck(),
        build_watchtower(),
        build_staircase(),
    ]


def generate_dataset(out_dir: Path, seed: int = 42, extra_variants: int = 12) -> Dict[str, int]:
    """Generate verified .ldr/.dsl files, SFT JSONL (messages & prompt/completion), and DPO pairs."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ldr_dir = out_dir / "ldr"
    dsl_dir = out_dir / "dsl"
    ldr_dir.mkdir(exist_ok=True)
    dsl_dir.mkdir(exist_ok=True)

    models: List[LegoModel] = default_templates()

    # Create color-swapped variants to broaden SFT diversity
    palette = [0, 1, 2, 4, 14, 15, 19, 25, 71, 72]
    builders = [
        lambda: build_stepped_pyramid(rng.choice(palette), rng.choice(palette)),
        lambda: build_mini_house(rng.choice(palette), rng.choice(palette), rng.choice(palette)),
        lambda: build_arch_bridge(rng.choice(palette), rng.choice(palette), rng.choice(palette)),
        lambda: build_mini_truck(rng.choice(palette), rng.choice(palette), 0),
        lambda: build_watchtower(rng.choice(palette), rng.choice(palette), rng.choice(palette)),
        lambda: build_staircase(rng.choice(palette), rng.choice(palette)),
    ]
    for i in range(extra_variants):
        m = builders[i % len(builders)]()
        m.name = f"{m.name}_v{i + 1}"
        models.append(m)

    sft_messages_path = out_dir / "sft_messages.jsonl"
    sft_pc_path = out_dir / "sft_prompt_completion.jsonl"
    dpo_path = out_dir / "dpo_pairs.jsonl"

    sft_count = 0
    dpo_count = 0

    with (
        open(sft_messages_path, "w", encoding="utf-8") as f_msg,
        open(sft_pc_path, "w", encoding="utf-8") as f_pc,
        open(dpo_path, "w", encoding="utf-8") as f_dpo,
    ):
        for m in models:
            report = validate_model(m)
            if not report.passed:
                raise RuntimeError(
                    f"Template model '{m.name}' failed validation:\n{report.format_feedback()}"
                )

            # Save .ldr and .dsl
            (ldr_dir / f"{m.name}.ldr").write_text(to_ldraw(m), encoding="utf-8")
            dsl_text = to_dsl(m)
            (dsl_dir / f"{m.name}.dsl").write_text(dsl_text, encoding="utf-8")

            bom_str = ", ".join(f"{k}x{v}" for k, v in m.bom().items())
            user_prompt = (
                f"Design a LEGO build: {m.description}\n"
                f"Available BOM: {bom_str}\n"
                f"Output Compact Grid DSL."
            )

            # 1. Chat messages JSONL (for `prepare_sft_dataset.py --hf-model ...`)
            f_msg.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": dsl_text},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            # 2. Prompt/completion JSONL (directly consumable by `tinymodel finetune --data`)
            f_pc.write(
                json.dumps(
                    {
                        "prompt": f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:\n",
                        "completion": dsl_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sft_count += 1

            # 3. Generate DPO preference + DRC repair pairs for every corruption mode
            for mode in ("collision", "floating", "disconnected"):
                broken = make_corrupted_variant(m, mode)
                b_report = validate_model(broken)
                assert not b_report.passed, f"Expected broken variant {broken.name} to fail DRC"
                broken_dsl = to_dsl(broken)

                # DPO pair (chosen=valid, rejected=broken)
                f_dpo.write(
                    json.dumps(
                        {
                            "prompt": user_prompt,
                            "chosen": dsl_text,
                            "rejected": broken_dsl,
                            "corruption_mode": mode,
                            "drc_feedback": b_report.format_feedback(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                dpo_count += 1

                # Also add a multi-turn DRC self-repair SFT sample
                repair_user = (
                    f"Your previous draft failed LEGO DRC validation:\n"
                    f"{b_report.format_feedback()}\n"
                    f"Fix all errors and output the corrected Compact Grid DSL."
                )
                f_msg.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": broken_dsl},
                                {"role": "user", "content": repair_user},
                                {"role": "assistant", "content": dsl_text},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                sft_count += 1

    return {
        "models": len(models),
        "sft_samples": sft_count,
        "dpo_pairs": dpo_count,
    }
