"""Physical & buildability DRC validator for LEGO models.

Checks performed (analogous to PCB DRC + Netlist Connectivity):
  1. Catalog & bounds validity (unknown parts, negative Y below ground)
  2. Voxel collision detection (no two bricks overlap in 3D space)
  3. Stud-to-tube connectivity graph (100% single connected component, no floating islands)
  4. Ground contact & static stability (center of mass projects inside ground support box)
  5. Step-by-step buildability (each step's bricks attach to ground or previously placed bricks)
  6. Optional BOM inventory constraint check
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ldraw import LegoModel
from .parts import PART_CATALOG, get_part


@dataclass
class ValidationIssue:
    code: str  # e.g. "COLLISION", "DISCONNECTED", "FLOATING_STEP", "UNSTABLE_COM", "BOM_EXCEEDED"
    severity: str  # "error" or "warning"
    message: str
    brick_indices: List[int] = field(default_factory=list)


@dataclass
class ValidationReport:
    model_name: str
    brick_count: int
    step_count: int
    collision_count: int
    connected_components: int
    largest_component_ratio: float
    grounded: bool
    com_stable: bool
    step_buildable: bool
    bom_ok: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "passed": self.passed,
            "brick_count": self.brick_count,
            "step_count": self.step_count,
            "collision_count": self.collision_count,
            "connected_components": self.connected_components,
            "largest_component_ratio": round(self.largest_component_ratio, 4),
            "grounded": self.grounded,
            "com_stable": self.com_stable,
            "step_buildable": self.step_buildable,
            "bom_ok": self.bom_ok,
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "message": i.message,
                    "brick_indices": i.brick_indices,
                }
                for i in self.issues
            ],
        }

    def format_feedback(self) -> str:
        """Format concise DRC diagnostics suitable as LLM repair feedback."""
        if self.passed:
            return (
                f"[PASS] {self.model_name}: {self.brick_count} bricks, "
                f"{self.step_count} steps, 0 collisions, 100% connected, stable."
            )
        lines = [
            f"[FAIL] {self.model_name}: {len([i for i in self.issues if i.severity == 'error'])} "
            f"DRC error(s) across {self.brick_count} bricks:"
        ]
        for issue in self.issues:
            lines.append(f"  - [{issue.code}] {issue.message}")
        return "\n".join(lines)


def validate_model(
    model: LegoModel,
    bom_limit: Optional[Dict[str, int]] = None,
) -> ValidationReport:
    """Run full physical & buildability DRC checks on a LegoModel."""
    issues: List[ValidationIssue] = []
    n = len(model.bricks)
    steps = sorted({b.step for b in model.bricks})

    if n == 0:
        issues.append(ValidationIssue("EMPTY_MODEL", "error", "Model contains 0 bricks."))
        return ValidationReport(
            model_name=model.name,
            brick_count=0,
            step_count=0,
            collision_count=0,
            connected_components=0,
            largest_component_ratio=0.0,
            grounded=False,
            com_stable=False,
            step_buildable=False,
            bom_ok=True,
            issues=issues,
        )

    # 1. Part validity & ground plane check
    valid_indices: List[int] = []
    for idx, b in enumerate(model.bricks):
        if b.part_id not in PART_CATALOG:
            issues.append(
                ValidationIssue(
                    "UNKNOWN_PART",
                    "error",
                    f"Brick #{idx} uses unknown part '{b.part_id}'.",
                    [idx],
                )
            )
            continue
        if b.y < 0:
            issues.append(
                ValidationIssue(
                    "BELOW_GROUND",
                    "error",
                    f"Brick #{idx} ({b.part_id}) is below ground plane at y={b.y}.",
                    [idx],
                )
            )
        valid_indices.append(idx)

    # 2. Voxel collision check
    voxel_owner: Dict[Tuple[int, int, int], int] = {}
    collision_pairs: Set[Tuple[int, int]] = set()
    brick_voxels: Dict[int, List[Tuple[int, int, int]]] = {}
    ground_xs: List[float] = []
    ground_zs: List[float] = []

    for idx in valid_indices:
        b = model.bricks[idx]
        spec = get_part(b.part_id)
        voxels = spec.world_voxels(b.x, b.y, b.z, b.rot)
        brick_voxels[idx] = voxels
        for v in voxels:
            if v[1] == 0:
                ground_xs.append(v[0] + 0.5)
                ground_zs.append(v[2] + 0.5)
            if v in voxel_owner:
                other = voxel_owner[v]
                pair = (min(other, idx), max(other, idx))
                if pair not in collision_pairs:
                    collision_pairs.add(pair)
                    ob = model.bricks[other]
                    issues.append(
                        ValidationIssue(
                            "COLLISION",
                            "error",
                            f"Brick #{idx} ({b.part_id} at {b.x},{b.y},{b.z}) collides with "
                            f"Brick #{other} ({ob.part_id} at {ob.x},{ob.y},{ob.z}) at voxel {v}.",
                            [other, idx],
                        )
                    )
            else:
                voxel_owner[v] = idx

    # 3. Stud-to-tube connectivity graph
    top_stud_map: Dict[Tuple[int, int, int], List[int]] = {}
    bottom_tube_map: Dict[Tuple[int, int, int], List[int]] = {}
    for idx in valid_indices:
        b = model.bricks[idx]
        spec = get_part(b.part_id)
        for pos in spec.world_top_studs(b.x, b.y, b.z, b.rot):
            top_stud_map.setdefault(pos, []).append(idx)
        for pos in spec.world_bottom_tubes(b.x, b.y, b.z, b.rot):
            bottom_tube_map.setdefault(pos, []).append(idx)

    adj: Dict[int, Set[int]] = {idx: set() for idx in valid_indices}
    for pos, bottom_bricks in bottom_tube_map.items():
        top_bricks = top_stud_map.get(pos, [])
        for u in bottom_bricks:
            for v in top_bricks:
                if u != v:
                    adj[u].add(v)
                    adj[v].add(u)

    visited: Set[int] = set()
    components: List[List[int]] = []
    for idx in valid_indices:
        if idx in visited:
            continue
        comp: List[int] = []
        q = deque([idx])
        visited.add(idx)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        components.append(comp)

    largest = max((len(c) for c in components), default=0)
    largest_ratio = largest / len(valid_indices) if valid_indices else 0.0

    if len(components) > 1:
        components_sorted = sorted(components, key=len, reverse=True)
        for comp in components_sorted[1:]:
            sample = ", ".join(f"#{i}({model.bricks[i].part_id})" for i in comp[:4])
            issues.append(
                ValidationIssue(
                    "DISCONNECTED",
                    "error",
                    f"Disconnected island of {len(comp)} brick(s): {sample} has no stud/tube connection to main body.",
                    comp,
                )
            )

    # 4. Ground contact & static center-of-mass stability
    grounded = True
    for comp in components:
        if not any(model.bricks[i].y == 0 for i in comp):
            grounded = False
            sample = ", ".join(f"#{i}({model.bricks[i].part_id})" for i in comp[:4])
            issues.append(
                ValidationIssue(
                    "FLOATING",
                    "error",
                    f"Component [{sample}] does not touch ground (y=0).",
                    comp,
                )
            )

    com_stable = False
    if ground_xs and ground_zs:
        total_voxels = 0
        sum_x = 0.0
        sum_z = 0.0
        for voxels in brick_voxels.values():
            for vx, _, vz in voxels:
                sum_x += vx + 0.5
                sum_z += vz + 0.5
                total_voxels += 1
        com_x = sum_x / total_voxels
        com_z = sum_z / total_voxels
        min_gx, max_gx = min(ground_xs) - 0.5, max(ground_xs) + 0.5
        min_gz, max_gz = min(ground_zs) - 0.5, max(ground_zs) + 0.5
        com_stable = (min_gx <= com_x <= max_gx) and (min_gz <= com_z <= max_gz)
        if not com_stable:
            issues.append(
                ValidationIssue(
                    "UNSTABLE_COM",
                    "error",
                    f"Center of mass ({com_x:.2f}, {com_z:.2f}) falls outside ground support box "
                    f"X=[{min_gx:.1f}, {max_gx:.1f}], Z=[{min_gz:.1f}, {max_gz:.1f}].",
                )
            )

    # 5. Step-by-step buildability: each step's bricks must connect to ground or earlier/same-step anchored bricks
    step_buildable = True
    anchored: Set[int] = set()
    by_step: Dict[int, List[int]] = {}
    for idx in valid_indices:
        by_step.setdefault(model.bricks[idx].step, []).append(idx)

    for step in sorted(by_step.keys()):
        step_set = set(by_step[step])
        # BFS from already anchored bricks + ground bricks in this step
        step_anchored: Set[int] = set()
        q = deque()
        for idx in by_step[step]:
            if model.bricks[idx].y == 0 or any(nb in anchored for nb in adj[idx]):
                step_anchored.add(idx)
                q.append(idx)
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in step_set and nb not in step_anchored:
                    step_anchored.add(nb)
                    q.append(nb)
        floating_in_step = sorted(step_set - step_anchored)
        if floating_in_step:
            step_buildable = False
            sample = ", ".join(
                f"#{i}({model.bricks[i].part_id} at {model.bricks[i].x},{model.bricks[i].y},{model.bricks[i].z})"
                for i in floating_in_step[:4]
            )
            issues.append(
                ValidationIssue(
                    "FLOATING_STEP",
                    "error",
                    f"Step {step} places unanchored brick(s) before their support exists: {sample}.",
                    floating_in_step,
                )
            )
        anchored.update(step_anchored)

    # 6. Optional BOM inventory check
    bom_ok = True
    if bom_limit is not None:
        used = model.bom()
        for pid, count in used.items():
            allowed = bom_limit.get(pid, 0)
            if count > allowed:
                bom_ok = False
                issues.append(
                    ValidationIssue(
                        "BOM_EXCEEDED",
                        "error",
                        f"Part '{pid}' used {count} time(s), exceeding BOM limit of {allowed}.",
                    )
                )

    return ValidationReport(
        model_name=model.name,
        brick_count=n,
        step_count=len(steps),
        collision_count=len(collision_pairs),
        connected_components=len(components),
        largest_component_ratio=largest_ratio,
        grounded=grounded,
        com_stable=com_stable,
        step_buildable=step_buildable,
        bom_ok=bom_ok,
        issues=issues,
    )
