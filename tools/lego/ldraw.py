"""LDraw (.ldr) <-> Compact Grid DSL codec and model normalization.

Compact Grid DSL example:
    MODEL mini_house
    STEP 1 # foundation
    BRICK 3032 color=71 x=0 y=0 z=0 rot=0
    STEP 2 # walls
    BRICK 3001 color=15 x=0 y=1 z=0 rot=0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .parts import (
    LDU_PER_PLATE,
    LDU_PER_STUD,
    ROT_TO_MATRIX,
    get_part,
    matrix_to_rot,
)


@dataclass
class PlacedBrick:
    part_id: str
    color: int
    x: int  # stud grid X
    y: int  # plate grid Y (Y-up, ground = 0)
    z: int  # stud grid Z
    rot: int = 0  # 0, 1, 2, 3 (90 deg CW around Y)
    step: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        self.part_id = self.part_id.lower().removesuffix(".dat")
        self.rot = int(self.rot) % 4
        self.color = int(self.color)
        self.x = int(self.x)
        self.y = int(self.y)
        self.z = int(self.z)
        self.step = int(self.step)


@dataclass
class LegoModel:
    name: str
    description: str = ""
    bricks: List[PlacedBrick] = field(default_factory=list)
    step_labels: Dict[int, str] = field(default_factory=dict)

    def normalize_origin(self) -> "LegoModel":
        """Shift all bricks so min(x)=0, min(y)=0, min(z)=0 and sort deterministically."""
        if not self.bricks:
            return LegoModel(name=self.name, description=self.description)
        min_x = min(b.x for b in self.bricks)
        min_y = min(b.y for b in self.bricks)
        min_z = min(b.z for b in self.bricks)
        shifted = [
            PlacedBrick(
                part_id=b.part_id,
                color=b.color,
                x=b.x - min_x,
                y=b.y - min_y,
                z=b.z - min_z,
                rot=b.rot,
                step=b.step,
                note=b.note,
            )
            for b in self.bricks
        ]
        shifted.sort(key=lambda b: (b.step, b.y, b.z, b.x, b.part_id, b.color, b.rot))
        return LegoModel(
            name=self.name,
            description=self.description,
            bricks=shifted,
            step_labels=dict(self.step_labels),
        )

    def bom(self) -> Dict[str, int]:
        """Return Bill of Materials: {part_id: count}."""
        counts: Dict[str, int] = {}
        for b in self.bricks:
            counts[b.part_id] = counts.get(b.part_id, 0) + 1
        return dict(sorted(counts.items()))

    def colored_bom(self) -> Dict[str, int]:
        """Return Bill of Materials keyed by '{part_id}:{color}'."""
        counts: Dict[str, int] = {}
        for b in self.bricks:
            key = f"{b.part_id}:{b.color}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


def to_dsl(model: LegoModel, include_header: bool = True) -> str:
    """Serialize a LegoModel to compact, token-efficient grid DSL."""
    lines: List[str] = []
    if include_header:
        lines.append(f"MODEL {model.name}")
        if model.description:
            lines.append(f"DESC {model.description}")

    by_step: Dict[int, List[PlacedBrick]] = {}
    for b in model.bricks:
        by_step.setdefault(b.step, []).append(b)

    for step in sorted(by_step.keys()):
        label = model.step_labels.get(step, "")
        step_hdr = f"STEP {step} # {label}" if label else f"STEP {step}"
        lines.append(step_hdr)
        for b in by_step[step]:
            line = f"BRICK {b.part_id} color={b.color} x={b.x} y={b.y} z={b.z} rot={b.rot}"
            lines.append(line)
    return "\n".join(lines) + "\n"


_BRICK_RE = re.compile(
    r"^BRICK\s+(\S+)\s+color=(-?\d+)\s+x=(-?\d+)\s+y=(-?\d+)\s+z=(-?\d+)\s+rot=(-?\d+)\s*$"
)
_STEP_RE = re.compile(r"^STEP\s+(\d+)(?:\s*#\s*(.*))?$")


def parse_dsl(text: str, default_name: str = "unnamed") -> LegoModel:
    """Parse compact grid DSL back into a LegoModel."""
    name = default_name
    description = ""
    bricks: List[PlacedBrick] = []
    step_labels: Dict[int, str] = {}
    current_step = 1

    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("MODEL "):
            name = line[len("MODEL ") :].strip()
            continue
        if line.startswith("DESC "):
            description = line[len("DESC ") :].strip()
            continue
        m_step = _STEP_RE.match(line)
        if m_step:
            current_step = int(m_step.group(1))
            lbl = (m_step.group(2) or "").strip()
            if lbl:
                step_labels[current_step] = lbl
            continue
        m_brick = _BRICK_RE.match(line)
        if m_brick:
            part_id, color, x, y, z, rot = m_brick.groups()
            bricks.append(
                PlacedBrick(
                    part_id=part_id,
                    color=int(color),
                    x=int(x),
                    y=int(y),
                    z=int(z),
                    rot=int(rot),
                    step=current_step,
                )
            )
            continue
        raise ValueError(f"DSL parse error at line {line_no}: {line!r}")

    return LegoModel(
        name=name,
        description=description,
        bricks=bricks,
        step_labels=step_labels,
    )


def to_ldraw(model: LegoModel) -> str:
    """Export LegoModel to standard LDraw (.ldr) format (Y-down, LDU coordinates)."""
    lines: List[str] = [
        f"0 {model.description or model.name}",
        f"0 Name: {model.name}.ldr",
        "0 Author: tinymodel-lego",
    ]
    by_step: Dict[int, List[PlacedBrick]] = {}
    for b in model.bricks:
        by_step.setdefault(b.step, []).append(b)

    steps = sorted(by_step.keys())
    for idx, step in enumerate(steps):
        label = model.step_labels.get(step)
        if label:
            lines.append(f"0 // STEP {step}: {label}")
        for b in by_step[step]:
            spec = get_part(b.part_id)
            rw, rd = spec.rotated_dims(b.rot)
            # LDraw origin: top-center of bounding box in X/Z, top surface in -Y
            cx = (b.x + rw / 2.0) * LDU_PER_STUD
            cz = (b.z + rd / 2.0) * LDU_PER_STUD
            cy = -(b.y + spec.height) * LDU_PER_PLATE
            mat = ROT_TO_MATRIX[b.rot]
            mat_str = " ".join(str(v) for v in mat)
            lines.append(
                f"1 {b.color} {cx:g} {cy:g} {cz:g} {mat_str} {b.part_id}.dat"
            )
        if idx < len(steps) - 1:
            lines.append("0 STEP")
    return "\n".join(lines) + "\n"


def parse_ldraw(text: str, default_name: str = "unnamed") -> LegoModel:
    """Parse an LDraw (.ldr) file into a discrete-grid LegoModel."""
    name = default_name
    description = ""
    bricks: List[PlacedBrick] = []
    step_labels: Dict[int, str] = {}
    current_step = 1
    first_comment_seen = False

    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("0 "):
            content = line[2:].strip()
            if content.upper() == "STEP":
                current_step += 1
            elif content.startswith("Name:"):
                name = content[len("Name:") :].strip().removesuffix(".ldr")
            elif content.startswith("// STEP "):
                m = re.match(r"//\s*STEP\s+(\d+)\s*:\s*(.*)", content)
                if m:
                    step_labels[int(m.group(1))] = m.group(2).strip()
            elif not first_comment_seen and not content.startswith("Author:"):
                description = content
                first_comment_seen = True
            continue
        if line.startswith("1 "):
            toks = line.split()
            if len(toks) < 15:
                raise ValueError(f"Malformed LDraw type-1 line {line_no}: {line!r}")
            color = int(toks[1])
            cx, cy, cz = float(toks[2]), float(toks[3]), float(toks[4])
            mat = tuple(float(toks[i]) for i in range(5, 14))
            part_file = " ".join(toks[14:])
            part_id = part_file.lower().removesuffix(".dat")
            spec = get_part(part_id)
            rot = matrix_to_rot(mat)
            rw, rd = spec.rotated_dims(rot)

            fx = cx / LDU_PER_STUD - rw / 2.0
            fz = cz / LDU_PER_STUD - rd / 2.0
            fy = -cy / LDU_PER_PLATE - spec.height

            ix, iy, iz = round(fx), round(fy), round(fz)
            if abs(fx - ix) > 1e-3 or abs(fy - iy) > 1e-3 or abs(fz - iz) > 1e-3:
                raise ValueError(
                    f"Off-grid LDraw coordinate at line {line_no}: "
                    f"({fx:.3f}, {fy:.3f}, {fz:.3f}) for part {part_id}"
                )
            bricks.append(
                PlacedBrick(
                    part_id=part_id,
                    color=color,
                    x=ix,
                    y=iy,
                    z=iz,
                    rot=rot,
                    step=current_step,
                )
            )

    return LegoModel(
        name=name,
        description=description,
        bricks=bricks,
        step_labels=step_labels,
    )


def load_model(path: Path) -> LegoModel:
    """Auto-detect .ldr vs .dsl by extension (or content) and load a LegoModel."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".ldr", ".mpd"):
        return parse_ldraw(text, default_name=path.stem)
    if path.suffix.lower() in (".dsl", ".lego", ".txt"):
        return parse_dsl(text, default_name=path.stem)
    # Fallback by content inspection
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("MODEL ", "STEP ", "BRICK ")):
            return parse_dsl(text, default_name=path.stem)
        if s.startswith(("0 ", "1 ")):
            return parse_ldraw(text, default_name=path.stem)
    return parse_dsl(text, default_name=path.stem)
