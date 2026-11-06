"""Uniform table-covering candidate pool for DynEdge / scorer search.

Each grid anchor on the table defines one **layout cluster**: free parts are
placed at footprint-aware offsets around the anchor (not one huge part per
distant cell).  Coarse filter drops obviously bad layouts; flatsurface poses
are tried per part.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def _max_footprint(searcher) -> Tuple[float, float]:
    fx, fy = 0.05, 0.05
    for pid in searcher.part_order:
        for cand in searcher.rot_cands.get(pid, []) or []:
            fp = np.asarray(getattr(cand, "footprint", [0.05, 0.05]), dtype=float)[:2]
            fx = max(fx, float(fp[0]))
            fy = max(fy, float(fp[1]))
    return fx, fy


def _auto_anchor_spacing(searcher, user_spacing: float) -> float:
    """Anchor grid step: at least largest footprint + clearance."""
    fx, fy = _max_footprint(searcher)
    clearance = float(getattr(searcher, "min_staging_mesh_clearance", 0.01))
    auto = max(fx, fy) + clearance + 0.04
    if user_spacing <= 0:
        return auto
    return max(float(user_spacing), auto * 0.85)


def table_anchor_grid(searcher, spacing: float) -> List[np.ndarray]:
    """Uniform anchor points covering the staging table."""
    xlo, xhi = float(searcher.table_x_range[0]), float(searcher.table_x_range[1])
    ylo, yhi = float(searcher.table_y_range[0]), float(searcher.table_y_range[1])
    fx, fy = _max_footprint(searcher)
    pad_x, pad_y = fx / 2.0 + 0.02, fy / 2.0 + 0.02
    xlo, xhi = xlo + pad_x, xhi - pad_x
    ylo, yhi = ylo + pad_y, yhi - pad_y
    if xlo >= xhi or ylo >= yhi:
        return [np.array([0.5 * (xlo + xhi), 0.5 * (ylo + yhi)], dtype=float)]

    step = _auto_anchor_spacing(searcher, spacing)
    cols = max(1, int(math.floor((xhi - xlo) / step)) + 1)
    rows = max(1, int(math.floor((yhi - ylo) / step)) + 1)
    xs = np.linspace(xlo, xhi, cols, dtype=float)
    ys = np.linspace(ylo, yhi, rows, dtype=float)
    return [np.array([float(x), float(y)], dtype=float) for y in ys for x in xs]


def _footprint_of(searcher, pid: str) -> np.ndarray:
    cands = searcher.rot_cands.get(pid, []) or []
    if not cands:
        return np.array([0.05, 0.05], dtype=float)
    return np.asarray(cands[0].footprint, dtype=float)[:2]


def _cluster_offset_templates(n: int, gap: float = 0.02) -> List[List[np.ndarray]]:
    """Relative offsets from anchor for n parts (index 0 = largest at origin)."""
    if n <= 0:
        return [[]]
    templates: List[List[np.ndarray]] = []

    ring: List[np.ndarray] = [np.zeros(2, dtype=float)]
    if n > 1:
        r0 = 0.14
        for i in range(1, n):
            ang = 2.0 * math.pi * (i - 1) / max(n - 1, 1)
            ring.append(np.array([r0 * math.cos(ang), r0 * math.sin(ang)], dtype=float))
    templates.append(ring)

    line: List[np.ndarray] = []
    x = 0.0
    for i in range(n):
        line.append(np.array([x, 0.0], dtype=float))
        x += 0.14 + gap
    templates.append(line)

    grid: List[np.ndarray] = []
    cols = max(1, int(math.ceil(math.sqrt(n))))
    for i in range(n):
        r, c = divmod(i, cols)
        grid.append(np.array([c * (0.13 + gap), r * (0.13 + gap)], dtype=float))
    templates.append(grid)

    return templates


def _grasp_available(searcher, pid: str) -> bool:
    try:
        gc = searcher._grasp_collection(pid)
        return gc is not None and len(gc) > 0
    except Exception:
        return False


def _feasible_poses_at_xy(
    searcher,
    pid: str,
    xy: np.ndarray,
    placed: List[str],
    check_home: bool = False,
) -> List[object]:
    ok: List[object] = []
    for cand in searcher.rot_cands.get(pid, []) or []:
        if searcher._upright_hard_constraint_reason(pid, cand):
            continue
        if searcher._staging_arm_keepout_reason(pid, xy, cand):
            continue
        searcher._apply_staging_pose(pid, xy, cand)
        if searcher._pairwise_collision(active_pids=[pid] + placed):
            continue
        if searcher._mesh_clearance_reason(active_pids=[pid] + placed):
            continue
        if check_home and searcher._robot_home_collision_reason(active_pids=[pid]):
            continue
        ok.append(cand)
    return ok


def _clip_xy_for_part(searcher, pid: str, xy: np.ndarray, cand) -> np.ndarray:
    if hasattr(searcher, "_clip_xy_for_part"):
        return np.asarray(searcher._clip_xy_for_part(pid, xy), dtype=float)
    fp = np.asarray(getattr(cand, "footprint", [0.05, 0.05]), dtype=float)[:2]
    xlo, xhi = searcher.table_x_range
    ylo, yhi = searcher.table_y_range
    x = float(np.clip(xy[0], xlo + fp[0] / 2, xhi - fp[0] / 2))
    y = float(np.clip(xy[1], ylo + fp[1] / 2, yhi - fp[1] / 2))
    return np.array([x, y], dtype=float)


def _build_layout_with_poses(
    searcher,
    region: Tuple[str, Tuple[int, int], np.ndarray],
    xy: Dict[str, np.ndarray],
    pose_pick: Dict[str, object],
):
    from sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm import LayoutCandidate

    cand = LayoutCandidate(xy={
        pid: np.asarray(val, dtype=float).copy() for pid, val in xy.items()
    })
    cand.assembly_region_id = region[0]
    cand.assembly_region_rc = region[1]
    cand.assembly_station_pos = np.asarray(region[2], dtype=float)
    for pid, rc in pose_pick.items():
        cand.pose_tag[pid] = str(getattr(rc, "tag", "unknown"))
        cand.rot_name[pid] = str(getattr(rc, "rot_name", "unknown"))
        cand.z_offset[pid] = float(getattr(rc, "z_offset", 0.0))
        cand.chosen_rotmat[pid] = np.asarray(getattr(rc, "rotmat", np.eye(3)), dtype=float)
    return cand


def _layout_at_anchor(
    searcher,
    region: Tuple[str, Tuple[int, int], np.ndarray],
    anchor: np.ndarray,
    rel_offsets: List[np.ndarray],
    pose_variant: int = 0,
):
    """Place a part cluster around one table anchor."""
    first_pid = searcher._first_part_id() if searcher.preassemble_first_part else None
    free = [
        pid for pid in searcher.part_order
        if pid != first_pid and _grasp_available(searcher, pid)
    ]
    if not free or len(rel_offsets) < len(free):
        return None

    free.sort(
        key=lambda p: float(np.prod(_footprint_of(searcher, p))),
        reverse=True,
    )
    gap = float(getattr(searcher, "min_staging_mesh_clearance", 0.01))

    xy: Dict[str, np.ndarray] = {}
    pose_pick: Dict[str, object] = {}
    placed: List[str] = []

    if first_pid is not None and first_pid in searcher.world_poses:
        gp, _ = searcher.world_poses[first_pid]
        xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
        if first_pid in searcher.rot_cands:
            pose_pick[first_pid] = searcher.rot_cands[first_pid][0]
            searcher._apply_staging_pose(first_pid, xy[first_pid], pose_pick[first_pid])
        placed.append(first_pid)

    for i, pid in enumerate(free):
        off = np.asarray(rel_offsets[i], dtype=float)
        fp = _footprint_of(searcher, pid)
        scale = 1.0 + 0.15 * (float(fp[0]) / 0.12)
        pos = anchor + off * scale
        feasible = _feasible_poses_at_xy(searcher, pid, pos, placed, check_home=False)
        if not feasible:
            return None
        pick_idx = min(int(pose_variant), len(feasible) - 1)
        rc = feasible[pick_idx]
        pos = _clip_xy_for_part(searcher, pid, pos, rc)
        feasible2 = _feasible_poses_at_xy(searcher, pid, pos, placed, check_home=False)
        if not feasible2:
            return None
        rc = feasible2[min(pick_idx, len(feasible2) - 1)]
        xy[pid] = pos
        pose_pick[pid] = rc
        searcher._apply_staging_pose(pid, pos, rc)
        placed.append(pid)

    if searcher._pairwise_collision(active_pids=placed):
        return None
    if searcher._mesh_clearance_reason(active_pids=placed):
        return None
    return _build_layout_with_poses(searcher, region, xy, pose_pick)


def _random_layout_at_anchor(
    searcher,
    region: Tuple[str, Tuple[int, int], np.ndarray],
    anchor: np.ndarray,
    rng: np.random.Generator,
    pose_variant: int = 0,
):
    """Fallback: bias collision-free random sample toward anchor."""
    first_pid = searcher._first_part_id() if searcher.preassemble_first_part else None
    free = [p for p in searcher.part_order if p != first_pid]
    if not free:
        return None
    free.sort(key=lambda p: float(np.prod(_footprint_of(searcher, p))), reverse=True)
    for _ in range(12):
        xy = searcher.sample_collision_free_xy(rng)
        if xy is None:
            continue
        if free[0] not in xy:
            continue
        delta = anchor - xy[free[0]]
        shifted = {
            pid: np.asarray(xy[pid], dtype=float) + delta
            for pid in xy if pid != first_pid
        }
        if first_pid and first_pid in xy:
            shifted[first_pid] = np.asarray(xy[first_pid], dtype=float)
        pose_pick: Dict[str, object] = {}
        placed: List[str] = []
        if first_pid and first_pid in shifted:
            placed.append(first_pid)
            if first_pid in searcher.rot_cands:
                pose_pick[first_pid] = searcher.rot_cands[first_pid][0]
                searcher._apply_staging_pose(
                    first_pid, shifted[first_pid], pose_pick[first_pid])
        ok = True
        for pid in searcher.part_order:
            if pid == first_pid or pid not in shifted:
                continue
            pos = shifted[pid]
            feas = _feasible_poses_at_xy(searcher, pid, pos, placed, check_home=False)
            if not feas:
                ok = False
                break
            rc = feas[min(pose_variant, len(feas) - 1)]
            pos = _clip_xy_for_part(searcher, pid, pos, rc)
            feas2 = _feasible_poses_at_xy(searcher, pid, pos, placed, check_home=False)
            if not feas2:
                ok = False
                break
            rc = feas2[min(pose_variant, len(feas2) - 1)]
            pose_pick[pid] = rc
            shifted[pid] = pos
            searcher._apply_staging_pose(pid, pos, rc)
            placed.append(pid)
        if not ok:
            continue
        if searcher._pairwise_collision(active_pids=placed):
            continue
        if searcher._mesh_clearance_reason(active_pids=placed):
            continue
        return _build_layout_with_poses(searcher, region, shifted, pose_pick)
    return None


def build_uniform_candidate_pool(
    searcher,
    region: Tuple[str, Tuple[int, int], np.ndarray],
    spacing: float = 0.11,
    max_candidates: int = 800,
    pose_variants_per_layout: int = 2,
) -> List:
    searcher._set_region_from_tuple(region)
    step = _auto_anchor_spacing(searcher, spacing)
    anchors = table_anchor_grid(searcher, spacing)
    if not anchors:
        return []

    first_pid = searcher._first_part_id() if searcher.preassemble_first_part else None
    n_free = sum(
        1 for pid in searcher.part_order
        if pid != first_pid and _grasp_available(searcher, pid))
    templates = _cluster_offset_templates(n_free)
    pool: List = []
    seen: set = set()

    def _sig(cand) -> str:
        parts = []
        for pid in searcher.part_order:
            if pid not in cand.xy:
                continue
            p = cand.xy[pid]
            parts.append(f"{pid}:{p[0]:.3f},{p[1]:.3f}:{cand.rot_name.get(pid, '')}")
        return "|".join(parts)

    def _try_add(cand) -> bool:
        if cand is None:
            return False
        key = _sig(cand)
        if key in seen:
            return False
        seen.add(key)
        pool.append(cand)
        return True

    for ai, anchor in enumerate(anchors):
        for tmpl in templates:
            for variant in range(max(1, int(pose_variants_per_layout))):
                if _try_add(_layout_at_anchor(
                        searcher, region, anchor, tmpl, pose_variant=variant)):
                    if len(pool) >= int(max_candidates):
                        return pool
        rng = np.random.default_rng(10007 + ai)
        for variant in range(max(1, int(pose_variants_per_layout))):
            if _try_add(_random_layout_at_anchor(
                    searcher, region, anchor, rng, pose_variant=variant)):
                if len(pool) >= int(max_candidates):
                    return pool
    return pool


def estimate_pool_size(searcher, spacing: float = 0.11) -> int:
    anchors = table_anchor_grid(searcher, spacing)
    n_free = max(1, len(searcher.part_order) - 1)
    n_tmpl = len(_cluster_offset_templates(n_free))
    return len(anchors) * n_tmpl * 2
