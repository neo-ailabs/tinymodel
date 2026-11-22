"""Curated LEGO part catalog with exact LDraw geometry, voxel occupancy, and stud/tube connectors.

Coordinate conventions:
  - Compact DSL grid (Y-up):
      x, z: integer stud units (1 stud = 20 LDU)
      y:    integer plate units (1 plate = 8 LDU; 1 standard brick = 3 plates = 24 LDU)
      (x, y, z) is the minimum corner of the part's axis-aligned bounding box in grid coordinates
      after applying horizontal rotation `rot` in {0, 1, 2, 3} (0, 90, 180, 270 deg CW around +Y).
  - LDraw file format (Y-down):
      1 stud = 20 LDU along X and Z; 1 plate = 8 LDU along -Y.
      LDraw part origins sit at the top-center of the part (y_ldraw = top surface, extending +height down).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

LDU_PER_STUD = 20
LDU_PER_PLATE = 8

# Standard LDraw color IDs
COLORS: Dict[int, str] = {
    0: "Black",
    1: "Blue",
    2: "Green",
    4: "Red",
    7: "Light Gray",
    14: "Yellow",
    15: "White",
    19: "Tan",
    25: "Orange",
    47: "Trans-Clear",
    43: "Trans-Blue",
    71: "Light Bluish Gray",
    72: "Dark Bluish Gray",
}

# LDraw 3x3 rotation matrices for horizontal CW rotations around Y (0, 90, 180, 270 deg)
ROT_TO_MATRIX: Dict[int, Tuple[int, ...]] = {
    0: (1, 0, 0, 0, 1, 0, 0, 0, 1),
    1: (0, 0, 1, 0, 1, 0, -1, 0, 0),
    2: (-1, 0, 0, 0, 1, 0, 0, 0, -1),
    3: (0, 0, -1, 0, 1, 0, 1, 0, 0),
}


@dataclass(frozen=True)
class PartSpec:
    part_id: str
    name: str
    category: str  # "brick", "plate", "tile", "slope"
    width: int  # studs along local X (rot=0)
    depth: int  # studs along local Z (rot=0)
    height: int  # plates along Y (brick=3, plate=1, tile=1)
    # Local (dx, dz) occupied cells in [0..width-1] x [0..depth-1]; None means full rectangle
    footprint: Tuple[Tuple[int, int], ...] | None = None
    # Local (dx, dz) cells that have a top stud (tiles have none; slopes have partial studs)
    top_studs: Tuple[Tuple[int, int], ...] | None = None
    # Local (dx, dz) cells that have bottom anti-stud clutch
    bottom_tubes: Tuple[Tuple[int, int], ...] | None = None

    def local_cells(self) -> Tuple[Tuple[int, int], ...]:
        if self.footprint is not None:
            return self.footprint
        return tuple((dx, dz) for dx in range(self.width) for dz in range(self.depth))

    def local_top_studs(self) -> Tuple[Tuple[int, int], ...]:
        if self.top_studs is not None:
            return self.top_studs
        if self.category == "tile":
            return ()
        return self.local_cells()

    def local_bottom_tubes(self) -> Tuple[Tuple[int, int], ...]:
        if self.bottom_tubes is not None:
            return self.bottom_tubes
        return self.local_cells()

    def rotated_dims(self, rot: int) -> Tuple[int, int]:
        rot = rot % 4
        if rot in (0, 2):
            return self.width, self.depth
        return self.depth, self.width

    def rotate_cell(self, dx: int, dz: int, rot: int) -> Tuple[int, int]:
        """Rotate a local cell (dx, dz) into the rotated part's [0..rw-1] x [0..rd-1] box."""
        rot = rot % 4
        w, d = self.width, self.depth
        if rot == 0:
            return dx, dz
        if rot == 1:
            return dz, w - 1 - dx
        if rot == 2:
            return w - 1 - dx, d - 1 - dz
        return d - 1 - dz, dx

    def world_voxels(self, x: int, y: int, z: int, rot: int) -> List[Tuple[int, int, int]]:
        """Return all occupied integer voxels (vx, vy, vz) in Y-up grid space."""
        voxels: List[Tuple[int, int, int]] = []
        for dx, dz in self.local_cells():
            rx, rz = self.rotate_cell(dx, dz, rot)
            for dy in range(self.height):
                voxels.append((x + rx, y + dy, z + rz))
        return voxels

    def world_top_studs(self, x: int, y: int, z: int, rot: int) -> List[Tuple[int, int, int]]:
        """Return world coordinates (sx, y_top, sz) of top studs.

        A top stud at (sx, y_top, sz) connects to a bottom tube at (sx, y_top, sz).
        """
        y_top = y + self.height
        out: List[Tuple[int, int, int]] = []
        for dx, dz in self.local_top_studs():
            rx, rz = self.rotate_cell(dx, dz, rot)
            out.append((x + rx, y_top, z + rz))
        return out

    def world_bottom_tubes(self, x: int, y: int, z: int, rot: int) -> List[Tuple[int, int, int]]:
        """Return world coordinates (sx, y, sz) of bottom anti-stud connectors."""
        out: List[Tuple[int, int, int]] = []
        for dx, dz in self.local_bottom_tubes():
            rx, rz = self.rotate_cell(dx, dz, rot)
            out.append((x + rx, y, z + rz))
        return out


# Curated high-frequency LEGO catalog (standard bricks, plates, tiles, slopes)
PART_CATALOG: Dict[str, PartSpec] = {
    spec.part_id: spec
    for spec in [
        # Standard bricks (height = 3 plates)
        PartSpec("3005", "Brick 1x1", "brick", 1, 1, 3),
        PartSpec("3004", "Brick 1x2", "brick", 2, 1, 3),
        PartSpec("3622", "Brick 1x3", "brick", 3, 1, 3),
        PartSpec("3010", "Brick 1x4", "brick", 4, 1, 3),
        PartSpec("3009", "Brick 1x6", "brick", 6, 1, 3),
        PartSpec("3003", "Brick 2x2", "brick", 2, 2, 3),
        PartSpec("3002", "Brick 2x3", "brick", 3, 2, 3),
        PartSpec("3001", "Brick 2x4", "brick", 4, 2, 3),
        PartSpec("2456", "Brick 2x6", "brick", 6, 2, 3),
        PartSpec("3062b", "Brick 1x1 Round", "brick", 1, 1, 3),
        PartSpec(
            "2357",
            "Brick 2x2 Corner",
            "brick",
            2,
            2,
            3,
            footprint=((0, 0), (1, 0), (0, 1)),
            top_studs=((0, 0), (1, 0), (0, 1)),
            bottom_tubes=((0, 0), (1, 0), (0, 1)),
        ),
        # Standard plates (height = 1 plate)
        PartSpec("3024", "Plate 1x1", "plate", 1, 1, 1),
        PartSpec("3023", "Plate 1x2", "plate", 2, 1, 1),
        PartSpec("3623", "Plate 1x3", "plate", 3, 1, 1),
        PartSpec("3710", "Plate 1x4", "plate", 4, 1, 1),
        PartSpec("3666", "Plate 1x6", "plate", 6, 1, 1),
        PartSpec("3022", "Plate 2x2", "plate", 2, 2, 1),
        PartSpec("3021", "Plate 2x3", "plate", 3, 2, 1),
        PartSpec("3020", "Plate 2x4", "plate", 4, 2, 1),
        PartSpec("3795", "Plate 2x6", "plate", 6, 2, 1),
        PartSpec("3034", "Plate 2x8", "plate", 8, 2, 1),
        PartSpec("3031", "Plate 4x4", "plate", 4, 4, 1),
        PartSpec("3032", "Plate 4x6", "plate", 6, 4, 1),
        # Smooth tiles (height = 1 plate, NO top studs)
        PartSpec("3070b", "Tile 1x1", "tile", 1, 1, 1, top_studs=()),
        PartSpec("3069b", "Tile 1x2", "tile", 2, 1, 1, top_studs=()),
        PartSpec("3068b", "Tile 2x2", "tile", 2, 2, 1, top_studs=()),
        # Slopes (partial top studs on the flat back row)
        PartSpec("3040", "Slope 45 2x1", "slope", 1, 2, 3, top_studs=((0, 0),)),
        PartSpec("3039", "Slope 45 2x2", "slope", 2, 2, 3, top_studs=((0, 0), (1, 0))),
        PartSpec(
            "3037",
            "Slope 45 2x4",
            "slope",
            4,
            2,
            3,
            top_studs=((0, 0), (1, 0), (2, 0), (3, 0)),
        ),
        PartSpec("54200", "Slope 30 1x1 Cheese", "slope", 1, 1, 2, top_studs=()),
    ]
}


def get_part(part_id: str) -> PartSpec:
    norm = part_id.lower().removesuffix(".dat")
    if norm not in PART_CATALOG:
        raise KeyError(f"Unknown part_id '{part_id}' (not in curated catalog)")
    return PART_CATALOG[norm]


def matrix_to_rot(mat: Tuple[float, ...], tol: float = 1e-3) -> int:
    """Match an LDraw 3x3 rotation matrix to discrete horizontal rotation {0, 1, 2, 3}."""
    for rot, target in ROT_TO_MATRIX.items():
        if all(abs(a - b) <= tol for a, b in zip(mat, target)):
            return rot
    raise ValueError(f"Unsupported non-horizontal or non-orthogonal LDraw matrix: {mat}")
