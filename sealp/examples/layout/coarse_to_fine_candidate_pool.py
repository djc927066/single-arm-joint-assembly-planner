"""Footprint-aware coarse-to-fine candidate-pool generation.

Only the candidate-pool generation is changed. The existing neural ranking,
exact L2/L3 validation, IK, grasp matching, collision checking, layout score,
and output format remain unchanged.

Both stages apply deterministic, inexpensive pre-filters before a candidate is
sent to DynEdge or the exact evaluator:

1. part footprint must stay inside the table;
2. occupied grid cells must not overlap;
3. staging poses must stay outside the arm-base keepout rectangles;
4. at least one pose must satisfy the upright hard constraint;
5. part AABBs must keep the configured coarse safety clearance.

A candidate is retained when at least one joint assignment of admissible pose
footprints satisfies all five pre-filters. Exact mesh/robot/IK checks are still
performed by the original evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


Region = Tuple[str, Tuple[int, int], np.ndarray]
XYLayout = Dict[str, np.ndarray]
PoolItem = Tuple[Region, XYLayout]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class GridGeometry:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spacing: float
    origin_x: float
    origin_y: float
    cols: int
    rows: int


@dataclass(frozen=True)
class PartFootprint:
    part_id: str
    width: float
    length: float
    cells_x: int
    cells_y: int

    @property
    def area(self) -> float:
        return float(self.width * self.length)


@dataclass(frozen=True)
class PoseFootprint:
    """Cheap XY representation of one existing rotation candidate."""

    part_id: str
    pose_index: int
    width: float
    length: float
    cells_x: int
    cells_y: int
    candidate: object

    @property
    def area(self) -> float:
        return float(self.width * self.length)


ConditionBuilder = Callable[[object, Region], Mapping[str, object]]


_REASON_KEYS = (
    "table_bounds",
    "grid_overlap",
    "staging_arm_keepout",
    "upright_constraint",
    "aabb_clearance",
    "no_pose_candidate",
    "no_legal_pose_position",
    "duplicate_layout",
)


def _new_stats() -> Dict[str, int]:
    return {key: 0 for key in _REASON_KEYS}


def _effective_clearance(searcher: object, requested_margin: float) -> float:
    """Use no less than the exact pipeline's configured staging clearance."""
    exact_margin = float(getattr(searcher, "min_staging_mesh_clearance", 0.0) or 0.0)
    return max(0.0, float(requested_margin), exact_margin)


def _condition_geometry(
    searcher: object,
    region: Region,
    spacing: float,
    margin: float,
    condition_builder: ConditionBuilder,
) -> Tuple[GridGeometry, Dict[str, PartFootprint]]:
    cond = condition_builder(searcher, region)
    table_x_range = cond.get("table_x_range")
    table_y_range = cond.get("table_y_range")
    parts = cond.get("parts")
    if not isinstance(table_x_range, (list, tuple)) or len(table_x_range) != 2:
        raise ValueError("condition sample is missing table_x_range")
    if not isinstance(table_y_range, (list, tuple)) or len(table_y_range) != 2:
        raise ValueError("condition sample is missing table_y_range")
    if not isinstance(parts, list):
        raise ValueError("condition sample is missing parts")

    x_min, x_max = map(float, table_x_range)
    y_min, y_max = map(float, table_y_range)
    if spacing <= 0:
        raise ValueError("grid spacing must be positive")

    cols = max(1, int(math.floor((x_max - x_min) / spacing)))
    rows = max(1, int(math.floor((y_max - y_min) / spacing)))
    grid_width = cols * spacing
    grid_height = rows * spacing
    origin_x = x_min + 0.5 * ((x_max - x_min) - grid_width)
    origin_y = y_min + 0.5 * ((y_max - y_min) - grid_height)
    grid = GridGeometry(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        spacing=float(spacing),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        cols=cols,
        rows=rows,
    )

    footprints: Dict[str, PartFootprint] = {}
    for item in parts:
        if not isinstance(item, Mapping):
            continue
        pid = str(item.get("part_id", ""))
        fp = item.get("footprint")
        if not pid or not isinstance(fp, (list, tuple)) or len(fp) < 2:
            continue
        width = max(float(fp[0]), 1e-6)
        length = max(float(fp[1]), 1e-6)
        cells_x = max(1, int(math.ceil((width + 2.0 * margin) / spacing)))
        cells_y = max(1, int(math.ceil((length + 2.0 * margin) / spacing)))
        footprints[pid] = PartFootprint(
            part_id=pid,
            width=width,
            length=length,
            cells_x=cells_x,
            cells_y=cells_y,
        )
    return grid, footprints


def _first_part_id(searcher: object) -> Optional[str]:
    if not bool(getattr(searcher, "preassemble_first_part", False)):
        return None
    method = getattr(searcher, "_first_part_id", None)
    if not callable(method):
        return None
    value = method()
    return str(value) if value is not None else None


def _fallback_pose(fp: PartFootprint) -> PoseFootprint:
    return PoseFootprint(
        part_id=fp.part_id,
        pose_index=0,
        width=fp.width,
        length=fp.length,
        cells_x=fp.cells_x,
        cells_y=fp.cells_y,
        candidate=None,
    )


def _pose_profiles(
    searcher: object,
    pid: str,
    spacing: float,
    clearance: float,
    fallback: Optional[PartFootprint],
    stats: Optional[Dict[str, int]] = None,
    apply_upright: bool = True,
) -> List[PoseFootprint]:
    """Convert existing rotation candidates to cheap footprint profiles."""
    rot_map = getattr(searcher, "rot_cands", {})
    candidates = list(rot_map.get(pid, [])) if isinstance(rot_map, Mapping) else []
    profiles: List[PoseFootprint] = []
    upright_reason = getattr(searcher, "_upright_hard_constraint_reason", None)

    for pose_index, cand in enumerate(candidates):
        if apply_upright and callable(upright_reason):
            try:
                if upright_reason(pid, cand):
                    if stats is not None:
                        stats["upright_constraint"] += 1
                    continue
            except Exception:
                # The exact evaluator will remain the final authority. A broken
                # cheap check must not crash pool construction.
                pass
        try:
            fp = np.asarray(getattr(cand, "footprint"), dtype=float).reshape(-1)
            width = float(fp[0])
            length = float(fp[1])
        except Exception:
            continue
        if not np.isfinite(width) or not np.isfinite(length) or width <= 0 or length <= 0:
            continue
        profiles.append(PoseFootprint(
            part_id=pid,
            pose_index=pose_index,
            width=width,
            length=length,
            cells_x=max(1, int(math.ceil((width + 2.0 * clearance) / spacing))),
            cells_y=max(1, int(math.ceil((length + 2.0 * clearance) / spacing))),
            candidate=cand,
        ))

    if not profiles and fallback is not None and not candidates:
        # Compatibility fallback for simple tests or tasks without rot_cands.
        profiles = [_fallback_pose(fallback)]

    if not profiles and stats is not None:
        stats["no_pose_candidate"] += 1
    return profiles


def _inside_table(
    center: np.ndarray,
    profile: PoseFootprint,
    grid: GridGeometry,
) -> bool:
    """Physical footprint must be inside the table; no extra edge margin."""
    c = np.asarray(center, dtype=float).reshape(-1)
    if c.size < 2 or not np.all(np.isfinite(c[:2])):
        return False
    half_x = 0.5 * profile.width
    half_y = 0.5 * profile.length
    return (
        float(c[0]) - half_x >= grid.x_min - 1e-9
        and float(c[0]) + half_x <= grid.x_max + 1e-9
        and float(c[1]) - half_y >= grid.y_min - 1e-9
        and float(c[1]) + half_y <= grid.y_max + 1e-9
    )


def _staging_keepout_reason(
    searcher: object,
    pid: str,
    center: np.ndarray,
    profile: PoseFootprint,
) -> Optional[str]:
    method = getattr(searcher, "_staging_arm_keepout_reason", None)
    if callable(method) and profile.candidate is not None:
        try:
            return method(pid, np.asarray(center, dtype=float), profile.candidate)
        except Exception:
            pass

    if not bool(getattr(searcher, "filter_staging_near_arms", False)):
        return None
    clear_x = float(getattr(searcher, "staging_arm_x_clearance", 0.0) or 0.0)
    clear_y = float(getattr(searcher, "staging_arm_y_clearance", 0.0) or 0.0)
    arm_map_method = getattr(searcher, "_arm_base_xy_map", None)
    if not callable(arm_map_method):
        return None
    x, y = map(float, np.asarray(center, dtype=float)[:2])
    for name, arm_xy in arm_map_method().items():
        arm_x, arm_y = map(float, arm_xy)
        if (
            abs(x - arm_x) < clear_x + 0.5 * profile.width
            and abs(y - arm_y) < clear_y + 0.5 * profile.length
        ):
            return f"{pid} staging overlaps {name} keepout"
    return None


def _aabb_too_close(
    center_a: np.ndarray,
    profile_a: PoseFootprint,
    center_b: np.ndarray,
    profile_b: PoseFootprint,
    clearance: float,
) -> bool:
    dx = abs(float(center_a[0]) - float(center_b[0]))
    dy = abs(float(center_a[1]) - float(center_b[1]))
    limit_x = 0.5 * (profile_a.width + profile_b.width) + clearance
    limit_y = 0.5 * (profile_a.length + profile_b.length) + clearance
    return dx < limit_x and dy < limit_y


def _occupied_cells(
    grid: GridGeometry,
    center: np.ndarray,
    profile: PoseFootprint,
    clearance: float,
) -> Set[Cell]:
    """Conservative grid cells covered by footprint plus safety clearance."""
    x0 = float(center[0]) - 0.5 * profile.width - clearance
    x1 = float(center[0]) + 0.5 * profile.width + clearance
    y0 = float(center[1]) - 0.5 * profile.length - clearance
    y1 = float(center[1]) + 0.5 * profile.length + clearance
    c0 = int(math.floor((x0 - grid.origin_x) / grid.spacing))
    c1 = int(math.ceil((x1 - grid.origin_x) / grid.spacing)) - 1
    r0 = int(math.floor((y0 - grid.origin_y) / grid.spacing))
    r1 = int(math.ceil((y1 - grid.origin_y) / grid.spacing)) - 1
    c0, c1 = max(0, c0), min(grid.cols - 1, c1)
    r0, r1 = max(0, r0), min(grid.rows - 1, r1)
    if c0 > c1 or r0 > r1:
        return set()
    return {(row, col) for row in range(r0, r1 + 1) for col in range(c0, c1 + 1)}


def _block_center(grid: GridGeometry, profile: PoseFootprint, row: int, col: int) -> np.ndarray:
    x = grid.origin_x + (float(col) + 0.5 * profile.cells_x) * grid.spacing
    y = grid.origin_y + (float(row) + 0.5 * profile.cells_y) * grid.spacing
    return np.array([x, y], dtype=float)


def _legal_blocks(
    grid: GridGeometry,
    profile: PoseFootprint,
) -> List[Tuple[int, int]]:
    max_r = grid.rows - profile.cells_y
    max_c = grid.cols - profile.cells_x
    if max_r < 0 or max_c < 0:
        return []
    return [
        (row, col)
        for row in range(max_r + 1)
        for col in range(max_c + 1)
    ]


def _placement_reject_reason(
    searcher: object,
    pid: str,
    center: np.ndarray,
    profile: PoseFootprint,
    grid: GridGeometry,
    clearance: float,
    occupied: Set[Cell],
    assigned: Sequence[Tuple[str, np.ndarray, PoseFootprint]],
) -> Tuple[Optional[str], Set[Cell]]:
    if not _inside_table(center, profile, grid):
        return "table_bounds", set()

    if _staging_keepout_reason(searcher, pid, center, profile):
        return "staging_arm_keepout", set()

    cells = _occupied_cells(grid, center, profile, clearance)
    if cells and not occupied.isdisjoint(cells):
        return "grid_overlap", cells

    for _, other_center, other_profile in assigned:
        if _aabb_too_close(center, profile, other_center, other_profile, clearance):
            return "aabb_clearance", cells

    return None, cells


def _layout_signature(
    region: Region,
    xy: Mapping[str, np.ndarray],
    quantum: float,
) -> Tuple[object, ...]:
    q = max(float(quantum), 1e-6)
    values: List[object] = [str(region[0]), int(region[1][0]), int(region[1][1])]
    for pid in sorted(xy):
        p = np.asarray(xy[pid], dtype=float)
        values.extend((pid, int(round(float(p[0]) / q)), int(round(float(p[1]) / q))))
    return tuple(values)


def _find_pose_assignment(
    searcher: object,
    region: Region,
    xy: Mapping[str, np.ndarray],
    grid: GridGeometry,
    fallback_footprints: Mapping[str, PartFootprint],
    clearance: float,
    stats: Dict[str, int],
) -> Tuple[bool, Optional[str]]:
    """Find one joint pose-footprint assignment satisfying all cheap filters."""
    first_pid = _first_part_id(searcher)
    part_order = [str(pid) for pid in getattr(searcher, "part_order", [])]
    assigned: List[Tuple[str, np.ndarray, PoseFootprint]] = []
    occupied: Set[Cell] = set()

    if first_pid and first_pid in xy and first_pid in fallback_footprints:
        # The first part is already assembled. Its assembly-region validity is
        # checked by the original region generator, so upright/arm keepout do
        # not apply here. It still occupies physical space for other parts.
        first_profile = _fallback_pose(fallback_footprints[first_pid])
        first_center = np.asarray(xy[first_pid], dtype=float)[:2]
        if not _inside_table(first_center, first_profile, grid):
            stats["table_bounds"] += 1
            return False, "table_bounds"
        assigned.append((first_pid, first_center, first_profile))
        occupied.update(_occupied_cells(grid, first_center, first_profile, clearance))

    option_map: Dict[str, List[PoseFootprint]] = {}
    for pid in part_order:
        if pid == first_pid or pid not in xy:
            continue
        profiles = _pose_profiles(
            searcher,
            pid,
            spacing=grid.spacing,
            clearance=clearance,
            fallback=fallback_footprints.get(pid),
            stats=stats,
            apply_upright=True,
        )
        center = np.asarray(xy[pid], dtype=float)[:2]
        position_valid: List[PoseFootprint] = []
        local_reason: Optional[str] = None
        for profile in profiles:
            if not _inside_table(center, profile, grid):
                stats["table_bounds"] += 1
                local_reason = local_reason or "table_bounds"
                continue
            if _staging_keepout_reason(searcher, pid, center, profile):
                stats["staging_arm_keepout"] += 1
                local_reason = local_reason or "staging_arm_keepout"
                continue
            position_valid.append(profile)
        if not position_valid:
            stats["no_legal_pose_position"] += 1
            return False, local_reason or "no_legal_pose_position"
        # Small profiles first usually find a valid joint assignment quickly.
        option_map[pid] = sorted(position_valid, key=lambda p: (p.area, p.pose_index))

    # Fail-fast ordering: most constrained part first, then larger footprint.
    free_parts = sorted(
        option_map,
        key=lambda pid: (
            len(option_map[pid]),
            -max(profile.area for profile in option_map[pid]),
            part_order.index(pid) if pid in part_order else 10**6,
        ),
    )

    last_reason: Optional[str] = None

    def backtrack(index: int, cur_occupied: Set[Cell]) -> bool:
        nonlocal last_reason
        if index >= len(free_parts):
            return True
        pid = free_parts[index]
        center = np.asarray(xy[pid], dtype=float)[:2]
        for profile in option_map[pid]:
            reason, cells = _placement_reject_reason(
                searcher,
                pid,
                center,
                profile,
                grid,
                clearance,
                cur_occupied,
                assigned,
            )
            if reason is not None:
                stats[reason] += 1
                last_reason = reason
                continue
            assigned.append((pid, center, profile))
            if backtrack(index + 1, cur_occupied | cells):
                return True
            assigned.pop()
        return False

    ok = backtrack(0, occupied)
    if not ok:
        stats["no_legal_pose_position"] += 1
    return ok, last_reason or (None if ok else "no_legal_pose_position")



def _allocate_region_targets(
    regions: Sequence[Region],
    pool_size: int,
    region_weights: Optional[Mapping[str, float]],
) -> Dict[str, int]:
    """Allocate an exact integer candidate quota to each assembly region.

    The input region order is preserved when fractional remainders tie, so a
    center-first region list remains center-first.
    """
    region_ids = [str(region[0]) for region in regions]
    if not region_ids or pool_size <= 0:
        return {rid: 0 for rid in region_ids}

    if region_weights:
        raw_weights = [max(0.0, float(region_weights.get(rid, 0.0))) for rid in region_ids]
    else:
        raw_weights = [1.0 for _ in region_ids]
    total = float(sum(raw_weights))
    if total <= 0.0:
        raw_weights = [1.0 for _ in region_ids]
        total = float(len(region_ids))

    exact = [float(pool_size) * weight / total for weight in raw_weights]
    targets = [int(math.floor(value)) for value in exact]
    remainder = int(pool_size - sum(targets))
    order = sorted(
        range(len(region_ids)),
        key=lambda i: (-(exact[i] - targets[i]), i),
    )
    for i in order[:remainder]:
        targets[i] += 1
    return {rid: int(targets[i]) for i, rid in enumerate(region_ids)}

def build_footprint_coarse_pool(
    searcher: object,
    regions: Sequence[Region],
    pool_size: int,
    spacing: float,
    margin: float,
    seed: int,
    condition_builder: ConditionBuilder,
    jitter_ratio: float = 0.0,
    max_attempts_per_layout: int = 80,
    region_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[List[PoolItem], Dict[str, object]]:
    """Build a quota-controlled coarse pool with all five hard pre-filters."""
    regions = list(regions)
    if not regions or pool_size <= 0:
        return [], {"candidate_count": 0}

    clearance = _effective_clearance(searcher, margin)
    rng = np.random.default_rng(seed)
    geometry_cache: Dict[str, Tuple[GridGeometry, Dict[str, PartFootprint]]] = {}
    pose_cache: Dict[Tuple[str, str], List[PoseFootprint]] = {}
    usage_cache: Dict[Tuple[str, str], np.ndarray] = {}
    result: List[PoolItem] = []
    seen = set()
    stats = _new_stats()
    first_pid = _first_part_id(searcher)
    part_order = [str(pid) for pid in getattr(searcher, "part_order", [])]

    targets = _allocate_region_targets(regions, pool_size, region_weights)
    accepted_by_region: Dict[str, int] = {str(region[0]): 0 for region in regions}
    attempts_by_region: Dict[str, int] = {str(region[0]): 0 for region in regions}
    max_attempts_by_region: Dict[str, int] = {
        str(region[0]): max(20, int(targets.get(str(region[0]), 0)) * max_attempts_per_layout)
        for region in regions
    }

    attempts = 0
    max_attempts = max(pool_size * max_attempts_per_layout, pool_size + 20)
    while len(result) < pool_size and attempts < max_attempts:
        region = None
        for candidate_region in regions:
            rid = str(candidate_region[0])
            if accepted_by_region[rid] >= targets.get(rid, 0):
                continue
            if attempts_by_region[rid] >= max_attempts_by_region[rid]:
                continue
            region = candidate_region
            break
        if region is None:
            break

        region_key = str(region[0])
        attempts_by_region[region_key] += 1
        attempts += 1
        if region_key not in geometry_cache:
            geometry_cache[region_key] = _condition_geometry(
                searcher, region, spacing, clearance, condition_builder)
        grid, fallback_footprints = geometry_cache[region_key]

        xy: XYLayout = {}
        occupied: Set[Cell] = set()
        assigned: List[Tuple[str, np.ndarray, PoseFootprint]] = []
        chosen_cells: Dict[str, Set[Cell]] = {}

        if first_pid and first_pid in fallback_footprints:
            first_center = np.asarray(region[2], dtype=float)[:2]
            first_profile = _fallback_pose(fallback_footprints[first_pid])
            # Do not apply staging keepout/upright to the preassembled first part.
            if not _inside_table(first_center, first_profile, grid):
                stats["table_bounds"] += 1
                continue
            xy[first_pid] = first_center.copy()
            assigned.append((first_pid, first_center, first_profile))
            first_cells = _occupied_cells(grid, first_center, first_profile, clearance)
            occupied.update(first_cells)
            chosen_cells[first_pid] = first_cells

        free_parts = [
            pid for pid in part_order
            if pid != first_pid and pid in fallback_footprints
        ]
        # Use the largest admissible pose footprint for fail-safe placement order.
        def largest_area(pid: str) -> float:
            key = (region_key, pid)
            if key not in pose_cache:
                pose_cache[key] = _pose_profiles(
                    searcher,
                    pid,
                    spacing=spacing,
                    clearance=clearance,
                    fallback=fallback_footprints.get(pid),
                    stats=stats,
                    apply_upright=True,
                )
            profiles = pose_cache[key]
            return max((p.area for p in profiles), default=0.0)

        free_parts.sort(key=largest_area, reverse=True)
        success = True

        for pid in free_parts:
            profiles = pose_cache.get((region_key, pid), [])
            if not profiles:
                success = False
                break

            usage_key = (region_key, pid)
            usage = usage_cache.setdefault(
                usage_key, np.zeros((grid.rows, grid.cols), dtype=np.int64))
            placements: List[Tuple[float, float, int, int, PoseFootprint, np.ndarray, Set[Cell]]] = []

            for profile in profiles:
                for row, col in _legal_blocks(grid, profile):
                    center = _block_center(grid, profile, row, col)
                    if jitter_ratio > 0:
                        jitter = min(0.45, max(0.0, float(jitter_ratio))) * spacing
                        center = center + rng.uniform(-jitter, jitter, size=2)
                    clip = getattr(searcher, "_clip_xy_for_part", None)
                    if callable(clip):
                        center = np.asarray(clip(pid, center), dtype=float)[:2]

                    reason, cells = _placement_reject_reason(
                        searcher,
                        pid,
                        center,
                        profile,
                        grid,
                        clearance,
                        occupied,
                        assigned,
                    )
                    if reason is not None:
                        stats[reason] += 1
                        continue
                    if cells:
                        coverage_cost = float(np.mean([usage[r, c] for r, c in cells]))
                    else:
                        coverage_cost = float("inf")
                    placements.append((
                        coverage_cost,
                        float(rng.random()),
                        row,
                        col,
                        profile,
                        center,
                        cells,
                    ))

            if not placements:
                stats["no_legal_pose_position"] += 1
                success = False
                break

            placements.sort(key=lambda item: (item[0], item[1], item[4].area))
            _, _, _, _, profile, center, cells = placements[0]
            xy[pid] = center.copy()
            assigned.append((pid, center, profile))
            occupied.update(cells)
            chosen_cells[pid] = cells

        if not success:
            continue

        signature = _layout_signature(region, xy, spacing * 0.25)
        if signature in seen:
            stats["duplicate_layout"] += 1
            continue
        seen.add(signature)
        result.append((region, xy))
        accepted_by_region[region_key] += 1

        for pid, cells in chosen_cells.items():
            if pid == first_pid:
                continue
            usage = usage_cache.get((region_key, pid))
            if usage is None:
                continue
            for row, col in cells:
                usage[row, col] += 1

    region_counts: Dict[str, int] = {}
    for region, _ in result:
        region_counts[str(region[0])] = region_counts.get(str(region[0]), 0) + 1
    representative_grid = next(iter(geometry_cache.values()))[0] if geometry_cache else None
    meta: Dict[str, object] = {
        "pool_build_mode": "footprint_coarse_grid_prefilter_v2",
        "candidate_count": len(result),
        "requested_candidate_count": int(pool_size),
        "coarse_grid_spacing": float(spacing),
        "footprint_margin": float(margin),
        "effective_aabb_clearance": float(clearance),
        "coarse_jitter_ratio": float(jitter_ratio),
        "generation_attempts": int(attempts),
        "region_counts": region_counts,
        "region_targets": targets,
        "region_weights": ({str(k): float(v) for k, v in region_weights.items()}
                           if region_weights else None),
        "region_attempts": attempts_by_region,
        "prefilter_enabled": [
            "table_bounds",
            "grid_overlap",
            "staging_arm_keepout",
            "upright_constraint",
            "aabb_clearance",
        ],
        "prefilter_rejections": stats,
    }
    if representative_grid is not None:
        meta["grid_shape"] = [representative_grid.rows, representative_grid.cols]
    return result, meta


def build_local_fine_pool(
    searcher: object,
    selected: Sequence[Tuple[int, Region, Mapping[str, np.ndarray]]],
    fine_spacing: float,
    xy_radius: float,
    variants_per_layout: int,
    margin: float,
    seed: int,
    condition_builder: ConditionBuilder,
) -> Tuple[List[PoolItem], List[Dict[str, int]], Dict[str, object]]:
    """Generate local variants and discard impossible ones before rescoring."""
    if fine_spacing <= 0 or xy_radius < 0 or variants_per_layout <= 0:
        return [], [], {"candidate_count": 0}

    clearance = _effective_clearance(searcher, margin)
    rng = np.random.default_rng(seed)
    result: List[PoolItem] = []
    provenance: List[Dict[str, int]] = []
    seen = set()
    stats = _new_stats()
    first_pid = _first_part_id(searcher)
    part_order = [str(pid) for pid in getattr(searcher, "part_order", [])]

    radius_steps = max(1, int(round(xy_radius / fine_spacing))) if xy_radius > 0 else 0
    offsets: List[np.ndarray] = []
    if radius_steps > 0:
        for step_index in range(1, radius_steps + 1):
            d = step_index * fine_spacing
            offsets.extend([
                np.array([d, 0.0]),
                np.array([-d, 0.0]),
                np.array([0.0, d]),
                np.array([0.0, -d]),
                np.array([d, d]),
                np.array([d, -d]),
                np.array([-d, d]),
                np.array([-d, -d]),
            ])

    attempted_variants = 0
    for selected_rank, (coarse_index, region, base_xy_raw) in enumerate(selected):
        grid, fallback_footprints = _condition_geometry(
            searcher, region, fine_spacing, clearance, condition_builder)
        base_xy = {
            str(pid): np.asarray(value, dtype=float)[:2].copy()
            for pid, value in base_xy_raw.items()
        }
        free_parts = [pid for pid in part_order if pid != first_pid and pid in base_xy]
        raw_variants: List[XYLayout] = [base_xy]
        operations = [(pid, offset) for pid in free_parts for offset in offsets]
        if operations:
            start = int(rng.integers(0, len(operations)))
            operations = operations[start:] + operations[:start]

        # Generate more raw trials than the requested quota because pre-filtering
        # may reject some of them.
        raw_target = max(variants_per_layout * 6, variants_per_layout + len(operations))
        for pid, offset in operations:
            if len(raw_variants) >= raw_target:
                break
            trial = {key: value.copy() for key, value in base_xy.items()}
            moved = trial[pid] + offset
            clip = getattr(searcher, "_clip_xy_for_part", None)
            if callable(clip):
                moved = np.asarray(clip(pid, moved), dtype=float)
            trial[pid] = moved[:2].copy()
            raw_variants.append(trial)

        fill_attempts = 0
        while len(raw_variants) < raw_target and fill_attempts < raw_target * 20:
            fill_attempts += 1
            trial = {key: value.copy() for key, value in base_xy.items()}
            changed = False
            for pid in free_parts:
                if not offsets or rng.random() > 0.35:
                    continue
                moved = trial[pid] + offsets[int(rng.integers(0, len(offsets)))]
                clip = getattr(searcher, "_clip_xy_for_part", None)
                if callable(clip):
                    moved = np.asarray(clip(pid, moved), dtype=float)
                trial[pid] = moved[:2].copy()
                changed = True
            if changed:
                raw_variants.append(trial)

        accepted_for_source = 0
        for raw_index, xy in enumerate(raw_variants):
            if accepted_for_source >= variants_per_layout:
                break
            attempted_variants += 1
            signature = _layout_signature(region, xy, fine_spacing * 0.25)
            if signature in seen:
                stats["duplicate_layout"] += 1
                continue
            valid, _ = _find_pose_assignment(
                searcher,
                region,
                xy,
                grid,
                fallback_footprints,
                clearance,
                stats,
            )
            if not valid:
                continue
            seen.add(signature)
            result.append((region, xy))
            provenance.append({
                "source_coarse_index": int(coarse_index),
                "source_coarse_rank": int(selected_rank + 1),
                "local_variant_index": int(accepted_for_source),
                "raw_variant_index": int(raw_index),
            })
            accepted_for_source += 1

    meta = {
        "pool_build_mode": "local_fine_grid_prefilter_v2",
        "candidate_count": len(result),
        "attempted_variant_count": int(attempted_variants),
        "selected_coarse_count": len(selected),
        "fine_grid_spacing": float(fine_spacing),
        "fine_xy_radius": float(xy_radius),
        "variants_per_layout": int(variants_per_layout),
        "effective_aabb_clearance": float(clearance),
        "prefilter_enabled": [
            "table_bounds",
            "grid_overlap",
            "staging_arm_keepout",
            "upright_constraint",
            "aabb_clearance",
        ],
        "prefilter_rejections": stats,
    }
    return result, provenance, meta
