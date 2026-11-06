"""Fixed-pose, footprint-aware, balanced grid candidate-pool generation.

This module builds complete staging layouts on a 2-D table grid without
enumerating the combinatorial joint layout space.  It is designed for the
Tower task used by ``find_optimal_initial_layout_tower_neural.py``:

* the first part (normally ``base_plate``) is preassembled at one of the
  existing 3x3 assembly-region centers;
* every other part uses exactly one pose read from a saved debug JSON;
* the staging table is discretized into fixed-size cells (default 0.03 m);
* a large part occupies several adjacent cells according to its rotated
  footprint;
* candidate counts are balanced across assembly regions;
* within a region, low-usage placement blocks/zones are preferred so a finite
  pool covers the table more evenly;
* table bounds, arm-base keepout, base-plate overlap, fixed-obstacle AABBs,
  and footprint-inflated grid overlap are used as fast deterministic prefilters;
* exact L2/L3 evaluation remains the final authority.

The generator returns XY-only pool entries to preserve compatibility with the
existing DynEdge scorer and exact evaluator.  ``lock_fixed_poses_from_debug``
must be called before generation/evaluation so that the same pose is used by
feature construction, prefiltering, and L2 evaluation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Region = Tuple[str, Tuple[int, int], np.ndarray]
XYLayout = Dict[str, np.ndarray]
PoolItem = Tuple[Region, XYLayout]


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cell_size: float
    origin_x: float
    origin_y: float
    cols: int
    rows: int


@dataclass(frozen=True)
class PartGridShape:
    part_id: str
    width: float
    length: float
    cells_x: int
    cells_y: int

    @property
    def area(self) -> float:
        return float(self.width * self.length)


@dataclass(frozen=True)
class Placement:
    placement_id: int
    row: int
    col: int
    xy: Tuple[float, float]
    occupied_mask: int
    zone_id: int


# ---------------------------------------------------------------------------
# Fixed-pose handling
# ---------------------------------------------------------------------------


def _matrix_from_debug(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid fixed rotation matrix with shape={matrix.shape}")
    return matrix


def lock_fixed_poses_from_debug(
    searcher: object,
    debug_json_path: str,
    *,
    matrix_atol: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Restrict each non-preassembled part to one saved pose.

    The function first matches by ``rot_name`` and then checks the stored
    rotation matrix.  If the name is unavailable, it matches by matrix.  A
    mismatch is treated as an error rather than silently using another pose.
    """

    path = Path(debug_json_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixed-pose debug JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rot_names = payload.get("rot_name", {})
    rotmats = payload.get("chosen_rotmat", {})
    pose_tags = payload.get("pose_tag", {})
    if not isinstance(rot_names, Mapping) or not isinstance(rotmats, Mapping):
        raise ValueError("debug JSON must contain rot_name and chosen_rotmat maps")

    first_pid: Optional[str] = None
    if bool(getattr(searcher, "preassemble_first_part", False)):
        first_method = getattr(searcher, "_first_part_id", None)
        if callable(first_method):
            first_pid = first_method()

    locked: Dict[str, Dict[str, object]] = {}
    for pid in list(getattr(searcher, "part_order", [])):
        if pid == first_pid:
            continue
        if pid not in rotmats:
            raise KeyError(f"fixed pose missing chosen_rotmat for part={pid}")

        target_matrix = _matrix_from_debug(rotmats[pid])
        target_name = str(rot_names.get(pid, ""))
        candidates = list(getattr(searcher, "rot_cands", {}).get(pid, []) or [])
        if not candidates:
            raise RuntimeError(f"searcher has no rotation candidates for part={pid}")

        matched = None
        if target_name:
            same_name = [
                candidate
                for candidate in candidates
                if str(getattr(candidate, "rot_name", "")) == target_name
            ]
            for candidate in same_name:
                matrix = np.asarray(getattr(candidate, "rotmat", np.eye(3)), dtype=float)
                if np.allclose(matrix, target_matrix, atol=matrix_atol):
                    matched = candidate
                    break
            if matched is None and same_name:
                max_error = min(
                    float(np.max(np.abs(np.asarray(c.rotmat, dtype=float) - target_matrix)))
                    for c in same_name
                )
                raise RuntimeError(
                    f"rot_name matched but matrix differs for part={pid}, "
                    f"rot_name={target_name}, max_abs_error={max_error:.3e}"
                )

        if matched is None:
            for candidate in candidates:
                matrix = np.asarray(getattr(candidate, "rotmat", np.eye(3)), dtype=float)
                if np.allclose(matrix, target_matrix, atol=matrix_atol):
                    matched = candidate
                    break

        if matched is None:
            available = [str(getattr(c, "rot_name", "")) for c in candidates]
            raise RuntimeError(
                f"cannot match fixed pose for part={pid}, requested={target_name!r}, "
                f"available={available}"
            )

        # The exact evaluator iterates searcher.rot_cands[pid].  Keeping only
        # one candidate guarantees that L2 uses the same pose as the grid pool.
        searcher.rot_cands[pid] = [matched]
        locked[pid] = {
            "rot_name": str(getattr(matched, "rot_name", target_name)),
            "pose_tag": str(getattr(matched, "tag", pose_tags.get(pid, "unknown"))),
            "rotmat": np.asarray(matched.rotmat, dtype=float).tolist(),
            "footprint": np.asarray(matched.footprint, dtype=float)[:2].tolist(),
            "z_offset": float(getattr(matched, "z_offset", 0.0)),
        }

    if verbose:
        print(f"[grid] fixed poses loaded from: {path}")
        for pid, item in locked.items():
            print(
                f"[grid]   {pid:16s} rot={item['rot_name']} "
                f"footprint={np.round(item['footprint'], 4).tolist()}"
            )
    return locked


def decorate_candidate_with_fixed_poses(searcher: object, candidate: object) -> None:
    """Attach fixed-pose metadata to a LayoutCandidate used for NN features."""

    first_pid: Optional[str] = None
    if bool(getattr(searcher, "preassemble_first_part", False)):
        first_method = getattr(searcher, "_first_part_id", None)
        if callable(first_method):
            first_pid = first_method()

    for pid in list(getattr(searcher, "part_order", [])):
        if pid not in getattr(candidate, "xy", {}):
            continue
        if pid == first_pid:
            goal = getattr(searcher, "world_poses", {}).get(pid)
            if goal is not None:
                gp, gr = goal
                candidate.chosen_rotmat[pid] = np.asarray(gr, dtype=float).copy()
                candidate.z_offset[pid] = float(np.asarray(gp, dtype=float)[2])
                candidate.pose_tag[pid] = "preassembled_at_assembly_region"
                candidate.rot_name[pid] = "goal_pose"
            continue
        pose_list = list(getattr(searcher, "rot_cands", {}).get(pid, []) or [])
        if not pose_list:
            continue
        pose = pose_list[0]
        candidate.chosen_rotmat[pid] = np.asarray(pose.rotmat, dtype=float).copy()
        candidate.z_offset[pid] = float(getattr(pose, "z_offset", 0.0))
        candidate.pose_tag[pid] = str(getattr(pose, "tag", "unknown"))
        candidate.rot_name[pid] = str(getattr(pose, "rot_name", "unknown"))


# ---------------------------------------------------------------------------
# Grid geometry and placements
# ---------------------------------------------------------------------------


def _build_grid(searcher: object, cell_size: float) -> GridSpec:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    x_min, x_max = map(float, getattr(searcher, "table_x_range"))
    y_min, y_max = map(float, getattr(searcher, "table_y_range"))
    eps = 1e-9
    cols = max(
        1,
        int(math.floor((x_max - x_min + eps) / cell_size)),
    )
    rows = max(
        1,
        int(math.floor((y_max - y_min + eps) / cell_size)),
    )
    used_width = cols * cell_size
    used_height = rows * cell_size
    origin_x = x_min + 0.5 * ((x_max - x_min) - used_width)
    origin_y = y_min + 0.5 * ((y_max - y_min) - used_height)
    return GridSpec(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cell_size=float(cell_size),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        cols=cols,
        rows=rows,
    )


def _shape_for_pose(pid: str, pose: object, grid: GridSpec, pair_clearance: float) -> PartGridShape:
    footprint = np.asarray(getattr(pose, "footprint", [0.05, 0.05]), dtype=float)[:2]
    width = max(float(footprint[0]), 1e-6)
    length = max(float(footprint[1]), 1e-6)
    # Inflate each full footprint by one pairwise-clearance distance.  Two
    # adjacent occupied blocks then reserve approximately half the clearance
    # on each side before the exact mesh-clearance check.
    cells_x = max(1, int(math.ceil((width + pair_clearance) / grid.cell_size)))
    cells_y = max(1, int(math.ceil((length + pair_clearance) / grid.cell_size)))
    return PartGridShape(pid, width, length, cells_x, cells_y)


def _shape_from_world_pose(
    searcher: object,
    pid: str,
    rotmat: np.ndarray,
    grid: GridSpec,
    pair_clearance: float,
) -> PartGridShape:
    vertices = np.asarray(getattr(searcher, "mesh_vertices")[pid], dtype=float)
    world = vertices @ np.asarray(rotmat, dtype=float).T
    extent = world.max(axis=0) - world.min(axis=0)
    width = max(float(extent[0]), 1e-6)
    length = max(float(extent[1]), 1e-6)
    cells_x = max(1, int(math.ceil((width + pair_clearance) / grid.cell_size)))
    cells_y = max(1, int(math.ceil((length + pair_clearance) / grid.cell_size)))
    return PartGridShape(pid, width, length, cells_x, cells_y)


def _block_center(grid: GridSpec, shape: PartGridShape, row: int, col: int) -> np.ndarray:
    x = grid.origin_x + (float(col) + 0.5 * shape.cells_x) * grid.cell_size
    y = grid.origin_y + (float(row) + 0.5 * shape.cells_y) * grid.cell_size
    return np.array([x, y], dtype=float)


def _block_mask(grid: GridSpec, shape: PartGridShape, row: int, col: int) -> int:
    row_bits = ((1 << shape.cells_x) - 1) << col
    mask = 0
    for rr in range(row, row + shape.cells_y):
        mask |= row_bits << (rr * grid.cols)
    return mask


def _mask_for_rectangle(
    grid: GridSpec,
    center_xy: np.ndarray,
    width: float,
    length: float,
    pair_clearance: float,
) -> int:
    half_x = 0.5 * (float(width) + pair_clearance)
    half_y = 0.5 * (float(length) + pair_clearance)
    x0, x1 = float(center_xy[0]) - half_x, float(center_xy[0]) + half_x
    y0, y1 = float(center_xy[1]) - half_y, float(center_xy[1]) + half_y
    c0 = max(0, int(math.floor((x0 - grid.origin_x) / grid.cell_size)))
    c1 = min(grid.cols - 1, int(math.ceil((x1 - grid.origin_x) / grid.cell_size)) - 1)
    r0 = max(0, int(math.floor((y0 - grid.origin_y) / grid.cell_size)))
    r1 = min(grid.rows - 1, int(math.ceil((y1 - grid.origin_y) / grid.cell_size)) - 1)
    if c0 > c1 or r0 > r1:
        return 0
    row_bits = ((1 << (c1 - c0 + 1)) - 1) << c0
    mask = 0
    for rr in range(r0, r1 + 1):
        mask |= row_bits << (rr * grid.cols)
    return mask


def _zone_id(
    grid: GridSpec,
    row: int,
    col: int,
    shape: PartGridShape,
    macro_rows: int,
    macro_cols: int,
) -> int:
    center_row = float(row) + 0.5 * shape.cells_y
    center_col = float(col) + 0.5 * shape.cells_x
    zr = min(macro_rows - 1, int(center_row / max(grid.rows, 1) * macro_rows))
    zc = min(macro_cols - 1, int(center_col / max(grid.cols, 1) * macro_cols))
    return int(zr * macro_cols + zc)


def _within_table(center: np.ndarray, shape: PartGridShape, grid: GridSpec) -> bool:
    return (
        float(center[0]) - 0.5 * shape.width >= grid.x_min - 1e-9
        and float(center[0]) + 0.5 * shape.width <= grid.x_max + 1e-9
        and float(center[1]) - 0.5 * shape.length >= grid.y_min - 1e-9
        and float(center[1]) + 0.5 * shape.length <= grid.y_max + 1e-9
    )


def _is_support_table_obstacle(searcher: object, obstacle: object) -> bool:
    """Best-effort identification of the work table so it is not rejected."""
    table_name = str(getattr(searcher, "table_name", "work_table") or "work_table").lower()
    for attr in ("name", "_name", "model_name", "_sealp_name", "_sealp_part_id"):
        value = str(getattr(obstacle, attr, "") or "").lower()
        if value and table_name in value:
            return True
    extractor = getattr(searcher, "_extract_vertices_from_cmodel", None)
    if callable(extractor):
        try:
            vertices = extractor(obstacle)
            if vertices is not None and len(vertices) > 0:
                vertices = np.asarray(vertices, dtype=float)
                bmin, bmax = vertices.min(axis=0), vertices.max(axis=0)
                xlo, xhi = map(float, searcher.table_x_range)
                ylo, yhi = map(float, searcher.table_y_range)
                covers_xy = (
                    bmin[0] <= xlo + 0.03
                    and bmax[0] >= xhi - 0.03
                    and bmin[1] <= ylo + 0.03
                    and bmax[1] >= yhi - 0.03
                )
                top_matches = abs(float(bmax[2]) - float(searcher.table_top_z)) <= 0.03
                if covers_xy and top_matches:
                    return True
        except Exception:
            pass
    return False


def _fixed_environment_obstacles(searcher: object) -> List[object]:
    return [
        obstacle
        for obstacle in list(getattr(searcher, "env_obs", []) or [])
        if not _is_support_table_obstacle(searcher, obstacle)
    ]


def _fixed_obstacle_grid_mask(
    searcher: object,
    grid: GridSpec,
    obstacles: Sequence[object],
    clearance: float,
) -> int:
    """Convert fixed environment obstacles to one coarse XY-AABB grid mask.

    Candidate-pool generation deliberately avoids WRS mesh collision here.
    Exact collision/IK checks remain in the later L2 evaluator.
    """
    extractor = getattr(searcher, "_extract_vertices_from_cmodel", None)
    if not callable(extractor):
        return 0

    mask = 0
    for obstacle in obstacles:
        try:
            vertices = extractor(obstacle)
            if vertices is None or len(vertices) == 0:
                continue
            vertices = np.asarray(vertices, dtype=float)
            bmin = vertices.min(axis=0)
            bmax = vertices.max(axis=0)

            table_z = float(getattr(searcher, "table_top_z", 0.0))
            if float(bmax[2]) < table_z - 0.05:
                continue

            center = np.array(
                [
                    0.5 * (float(bmin[0]) + float(bmax[0])),
                    0.5 * (float(bmin[1]) + float(bmax[1])),
                ],
                dtype=float,
            )
            width = max(0.0, float(bmax[0]) - float(bmin[0]))
            length = max(0.0, float(bmax[1]) - float(bmin[1]))
            mask |= _mask_for_rectangle(
                grid,
                center,
                width,
                length,
                float(clearance),
            )
        except Exception:
            continue
    return mask


def _enumerate_part_placements(
    searcher: object,
    pid: str,
    pose: object,
    shape: PartGridShape,
    grid: GridSpec,
    static_mask: int,
    macro_rows: int,
    macro_cols: int,
    first_pid: Optional[str],
    rejection_counts: Dict[str, int],
    fixed_obstacles: Sequence[object],
) -> List[Placement]:
    """Enumerate legal placements using only grid/AABB approximations.

    No ``is_mcdwith``, mesh-clearance, or robot collision query is called here.
    Those expensive checks remain the responsibility of exact L2 evaluation.
    """
    del first_pid, fixed_obstacles

    placements: List[Placement] = []
    placement_id = 0
    max_row = grid.rows - shape.cells_y
    max_col = grid.cols - shape.cells_x
    if max_row < 0 or max_col < 0:
        return placements

    for row in range(max_row + 1):
        for col in range(max_col + 1):
            center = _block_center(grid, shape, row, col)
            mask = _block_mask(grid, shape, row, col)

            if mask & static_mask:
                rejection_counts["static_mask"] = (
                    rejection_counts.get("static_mask", 0) + 1
                )
                continue

            if not _within_table(center, shape, grid):
                rejection_counts["table_bounds"] = (
                    rejection_counts.get("table_bounds", 0) + 1
                )
                continue

            keepout = searcher._staging_arm_keepout_reason(pid, center, pose)
            if keepout:
                rejection_counts["arm_keepout"] = (
                    rejection_counts.get("arm_keepout", 0) + 1
                )
                continue

            placements.append(
                Placement(
                    placement_id=placement_id,
                    row=row,
                    col=col,
                    xy=(float(center[0]), float(center[1])),
                    occupied_mask=mask,
                    zone_id=_zone_id(
                        grid,
                        row,
                        col,
                        shape,
                        macro_rows=macro_rows,
                        macro_cols=macro_cols,
                    ),
                )
            )
            placement_id += 1

    return placements


# ---------------------------------------------------------------------------
# Balanced complete-layout sampling
# ---------------------------------------------------------------------------


def _region_targets(regions: Sequence[Region], pool_size: int) -> Dict[str, int]:
    base, remainder = divmod(int(pool_size), len(regions))
    return {
        str(region[0]): base + (1 if index < remainder else 0)
        for index, region in enumerate(regions)
    }


def _layout_signature(
    region: Region,
    chosen: Mapping[str, Placement],
) -> Tuple[object, ...]:
    values: List[object] = [str(region[0]), int(region[1][0]), int(region[1][1])]
    for pid in sorted(chosen):
        placement = chosen[pid]
        values.extend((pid, int(placement.row), int(placement.col)))
    return tuple(values)


def _apply_full_layout(
    searcher: object,
    region: Region,
    xy: Mapping[str, np.ndarray],
    free_parts: Sequence[str],
) -> object:
    from sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm import LayoutCandidate

    # The outer region loop has already set this assembly region.
    # Repeating it here would rebuild all goal collision models per proposal.
    candidate = LayoutCandidate(
        xy={pid: np.asarray(value, dtype=float).copy() for pid, value in xy.items()}
    )
    candidate.assembly_region_id = str(region[0])
    candidate.assembly_region_rc = tuple(region[1])
    candidate.assembly_station_pos = np.asarray(region[2], dtype=float).copy()
    searcher._apply_first_part_as_assembled(candidate)
    for pid in free_parts:
        pose = searcher.rot_cands[pid][0]
        searcher._apply_staging_pose(pid, candidate.xy[pid], pose)
    decorate_candidate_with_fixed_poses(searcher, candidate)
    return candidate


def _full_layout_prefilter_reason(
    searcher: object,
    candidate: object,
    free_parts: Sequence[str],
) -> Optional[str]:
    """Fast coarse prefilter using only existing grid/AABB constraints.

    Exact WRS collision, home-collision, and mesh-clearance checks are deferred
    to L2, which remains the final authority.
    """
    del free_parts
    order_reason = searcher._order_x_constraint_reason(candidate)
    if order_reason:
        return "order_x_constraint"
    return None


def _construct_one_layout(
    part_order: Sequence[str],
    placements_by_part: Mapping[str, Sequence[Placement]],
    static_mask: int,
    placement_usage: Mapping[str, np.ndarray],
    zone_usage: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    max_branches: int,
) -> Optional[Dict[str, Placement]]:
    chosen: Dict[str, Placement] = {}

    def backtrack(index: int, occupied: int) -> bool:
        if index >= len(part_order):
            return True
        pid = part_order[index]
        available = [
            placement
            for placement in placements_by_part[pid]
            if not (placement.occupied_mask & occupied)
        ]
        if not available:
            return False

        p_usage = placement_usage[pid]
        z_usage = zone_usage[pid]
        scored = [
            (
                int(z_usage[placement.zone_id]),
                int(p_usage[placement.placement_id]),
                float(rng.random()),
                placement,
            )
            for placement in available
        ]
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        branch_count = min(max(1, int(max_branches)), len(scored))
        for _, _, _, placement in scored[:branch_count]:
            chosen[pid] = placement
            if backtrack(index + 1, occupied | placement.occupied_mask):
                return True
            chosen.pop(pid, None)
        return False

    if backtrack(0, static_mask):
        return dict(chosen)
    return None


def build_footprint_grid_candidate_pool(
    searcher: object,
    regions: Sequence[Region],
    *,
    pool_size: int = 400,
    cell_size: float = 0.03,
    pair_clearance: Optional[float] = None,
    seed: int = 0,
    macro_rows: int = 6,
    macro_cols: int = 4,
    max_attempts_per_layout: int = 120,
    max_branches: int = 24,
    verbose: bool = True,
) -> Tuple[List[PoolItem], Dict[str, object]]:
    """Build a fixed-budget balanced pool of complete grid layouts."""

    regions = list(regions)
    if not regions:
        raise ValueError("no assembly regions supplied")
    if pool_size <= 0:
        return [], {"candidate_count": 0}
    if str(getattr(searcher, "preassemble_first_part", False)).lower() not in ("true", "1"):
        raise ValueError("footprint_grid_balanced requires preassemble_first_part=True")

    macro_rows = max(1, int(macro_rows))
    macro_cols = max(1, int(macro_cols))
    pair_clearance = (
        float(getattr(searcher, "min_staging_mesh_clearance", 0.01))
        if pair_clearance is None
        else max(0.0, float(pair_clearance))
    )
    grid = _build_grid(searcher, float(cell_size))
    fixed_obstacles = _fixed_environment_obstacles(searcher)
    rng = np.random.default_rng(int(seed))
    first_pid = searcher._first_part_id()
    free_parts = [pid for pid in searcher.part_order if pid != first_pid]

    shapes: Dict[str, PartGridShape] = {
        pid: _shape_for_pose(pid, searcher.rot_cands[pid][0], grid, pair_clearance)
        for pid in free_parts
    }
    free_parts.sort(key=lambda pid: shapes[pid].area, reverse=True)

    targets = _region_targets(regions, int(pool_size))
    region_counts = {str(region[0]): 0 for region in regions}
    rejection_counts: Dict[str, int] = {}
    valid_placement_counts: Dict[str, Dict[str, int]] = {}
    zone_coverage: Dict[str, Dict[str, int]] = {}
    discrete_upper_bounds: Dict[str, str] = {}
    pool: List[PoolItem] = []
    seen = set()

    for region in regions:
        rid = str(region[0])
        target = int(targets[rid])
        searcher._set_region_from_tuple(region)
        base_xy: Optional[np.ndarray] = None
        static_mask = 0
        if first_pid is not None and first_pid in searcher.world_poses:
            gp, gr = searcher.world_poses[first_pid]
            base_xy = np.asarray(gp, dtype=float)[:2].copy()
            base_shape = _shape_from_world_pose(
                searcher, first_pid, np.asarray(gr, dtype=float), grid, pair_clearance
            )
            static_mask = _mask_for_rectangle(
                grid,
                base_xy,
                base_shape.width,
                base_shape.length,
                pair_clearance,
            )
            searcher._apply_first_part_as_assembled()

        static_mask |= _fixed_obstacle_grid_mask(
            searcher,
            grid,
            fixed_obstacles,
            pair_clearance,
        )

        placements_by_part: Dict[str, List[Placement]] = {}
        for pid in free_parts:
            placements = _enumerate_part_placements(
                searcher=searcher,
                pid=pid,
                pose=searcher.rot_cands[pid][0],
                shape=shapes[pid],
                grid=grid,
                static_mask=static_mask,
                macro_rows=macro_rows,
                macro_cols=macro_cols,
                first_pid=first_pid,
                rejection_counts=rejection_counts,
                fixed_obstacles=fixed_obstacles,
            )
            if not placements:
                raise RuntimeError(f"region={rid} has no legal placements for part={pid}")
            placements_by_part[pid] = placements

        valid_placement_counts[rid] = {
            pid: len(placements_by_part[pid]) for pid in free_parts
        }
        naive_upper = 1
        for pid in free_parts:
            naive_upper *= len(placements_by_part[pid])
        discrete_upper_bounds[rid] = f"{float(naive_upper):.6e}"

        placement_usage: Dict[str, np.ndarray] = {
            pid: np.zeros(len(placements_by_part[pid]), dtype=np.int64)
            for pid in free_parts
        }
        zone_usage: Dict[str, np.ndarray] = {
            pid: np.zeros(macro_rows * macro_cols, dtype=np.int64)
            for pid in free_parts
        }

        attempts = 0
        max_attempts = max(target * max(1, int(max_attempts_per_layout)), target + 20)
        while region_counts[rid] < target and attempts < max_attempts:
            attempts += 1
            chosen = _construct_one_layout(
                part_order=free_parts,
                placements_by_part=placements_by_part,
                static_mask=static_mask,
                placement_usage=placement_usage,
                zone_usage=zone_usage,
                rng=rng,
                max_branches=max_branches,
            )
            if not chosen:
                rejection_counts["backtrack_no_solution"] = (
                    rejection_counts.get("backtrack_no_solution", 0) + 1
                )
                continue
            signature = _layout_signature(region, chosen)
            if signature in seen:
                rejection_counts["duplicate"] = rejection_counts.get("duplicate", 0) + 1
                # Slightly discourage the same placement combination on retry.
                for pid, placement in chosen.items():
                    placement_usage[pid][placement.placement_id] += 1
                    zone_usage[pid][placement.zone_id] += 1
                continue

            xy: XYLayout = {
                pid: np.asarray(placement.xy, dtype=float)
                for pid, placement in chosen.items()
            }
            if first_pid is not None and base_xy is not None:
                xy[first_pid] = base_xy.copy()
            candidate = _apply_full_layout(searcher, region, xy, free_parts)
            reason = _full_layout_prefilter_reason(searcher, candidate, free_parts)
            if reason is not None:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                for pid, placement in chosen.items():
                    # Avoid repeatedly proposing the same bad local combination.
                    placement_usage[pid][placement.placement_id] += 1
                continue

            seen.add(signature)
            pool.append((region, {pid: value.copy() for pid, value in candidate.xy.items()}))
            region_counts[rid] += 1
            for pid, placement in chosen.items():
                placement_usage[pid][placement.placement_id] += 1
                zone_usage[pid][placement.zone_id] += 1

        if region_counts[rid] != target:
            print(
                f"[grid][ERROR] region={rid} generated={region_counts[rid]}/"
                f"{target}, attempts={attempts}"
            )
            print(f"[grid][ERROR] rejection_counts={rejection_counts}")
            print(
                f"[grid][ERROR] valid_placements="
                f"{valid_placement_counts.get(rid, {})}"
            )
            raise RuntimeError(
                f"failed to fill region quota: region={rid}, "
                f"generated={region_counts[rid]}, target={target}, attempts={attempts}"
            )

        zone_coverage[rid] = {
            pid: int(np.count_nonzero(zone_usage[pid])) for pid in free_parts
        }

    if len(pool) != int(pool_size):
        raise RuntimeError(f"generated {len(pool)} layouts, expected {pool_size}")

    fixed_poses = {
        pid: {
            "rot_name": str(searcher.rot_cands[pid][0].rot_name),
            "pose_tag": str(searcher.rot_cands[pid][0].tag),
            "footprint": np.asarray(searcher.rot_cands[pid][0].footprint, dtype=float)[:2].tolist(),
            "grid_cells": [shapes[pid].cells_x, shapes[pid].cells_y],
        }
        for pid in free_parts
    }
    meta: Dict[str, object] = {
        "pool_build_mode": "footprint_grid_balanced_v1",
        "candidate_count": len(pool),
        "requested_candidate_count": int(pool_size),
        "cell_size": float(cell_size),
        "pair_clearance": float(pair_clearance),
        "grid_shape": [grid.rows, grid.cols],
        "grid_origin": [grid.origin_x, grid.origin_y],
        "macro_grid_shape": [macro_rows, macro_cols],
        "region_targets": targets,
        "region_counts": region_counts,
        "valid_placement_counts": valid_placement_counts,
        "zone_coverage": zone_coverage,
        "naive_discrete_upper_bounds": discrete_upper_bounds,
        "fixed_poses": fixed_poses,
        "prefilter_rejections": rejection_counts,
        "unique_layout_count": len(seen),
        "fixed_environment_obstacle_count": len(fixed_obstacles),
    }

    if verbose:
        print("\n========== Footprint Grid Candidate Pool ==========")
        print(f"grid             = {grid.rows} x {grid.cols}, cell={cell_size:.3f} m")
        print(f"pool             = {len(pool)} / {pool_size}")
        print(f"region targets   = {targets}")
        print(f"region counts    = {region_counts}")
        print(f"fixed poses      = { {pid: item['rot_name'] for pid, item in fixed_poses.items()} }")
        print(f"grid cells/part  = { {pid: item['grid_cells'] for pid, item in fixed_poses.items()} }")
        print(f"rejections       = {rejection_counts}")
    return pool, meta
