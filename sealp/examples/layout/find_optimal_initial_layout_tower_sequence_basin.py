#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence-aware feasibility-basin search for Tower initial layouts.

This is a traditional, training-free search method built on the existing
strict/global pipeline.

Pipeline
--------
1. Reuse arm-conditioned IK/common-grasp hint NPZ files.
2. Convert positive left/right feasibility maps into basin margins with a
   distance transform.
3. Extract diverse high-score/high-margin cells for each assembly step.
4. Construct complete layouts in assembly order with beam search and cheap
   geometry/robot-home pruning.
5. Send only top proposals to the original exact L2 evaluator.
6. Reuse the existing multiscale pattern refinement and optional L3 check.
7. Optionally test the final layout under bounded XY perturbations.

The hint maps only propose/rank cells. Final feasibility and score always come
from the original strict evaluate_layout implementation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod

from sealp.examples.layout.assembly_rl.assembly_layout_env import (
    AssemblyGridState,
    build_assembly_validator,
)

LayoutCandidate = fol.LayoutCandidate


DEFAULT_MERGED_HINT = os.path.join(
    fol.PROJECT_ROOT,
    "hint_cache",
    "tower_ik_hint_r1row.npz",
)
DEFAULT_SINGLE_HINT = os.path.join(
    fol.PROJECT_ROOT,
    "hint_cache",
    "tower_ik_hint_r1c1.npz",
)

BCFG: Dict[str, object] = {
    "ik_hint": None,
    "regions": None,
    "beam_width": 24,
    "part_candidates": 14,
    "exact_proposals": 36,
    "exact_per_region": 0,
    "elite": 3,
    "certify_topk": 0,
    "certify_grasp_cap": 0,
    "min_hint_score": 1e-6,
    "margin_cap": 0.08,
    "margin_weight": 0.45,
    "hint_weight": 0.40,
    "distance_weight": 0.15,
    "nms_cells": 2,
    "fallback_explore": 20,
    "robust_evals": 0,
    "robust_sigma": 0.01,
    "report_json": None,
}


@dataclass(frozen=True)
class BasinPoint:
    part_id: str
    arm_id: int
    pose_id: int
    row: int
    col: int
    xy: Tuple[float, float]
    hint_score: float
    margin_m: float
    utility: float


@dataclass
class BeamState:
    xy: Dict[str, np.ndarray]
    pose_ids: Dict[str, int] = field(default_factory=dict)
    arm_ids: Dict[str, int] = field(default_factory=dict)
    hint_scores: List[float] = field(default_factory=list)
    margins: List[float] = field(default_factory=list)
    distance_sum: float = 0.0
    rank_score: float = -np.inf

    def fingerprint(self, resolution: float) -> Tuple:
        scale = max(float(resolution), 1e-9)
        return tuple(
            (pid, int(round(float(pos[0]) / scale)), int(round(float(pos[1]) / scale)))
            for pid, pos in sorted(self.xy.items())
        )


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """Distance to the nearest infeasible/outside cell, in grid-cell units."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
        out = distance_transform_edt(padded)[1:-1, 1:-1]
        return np.asarray(out, dtype=np.float32)
    except Exception:
        # Dependency-free Manhattan fallback. It is conservative and sufficient
        # for ranking robust interior cells.
        h, w = padded.shape
        inf = h + w + 10
        d = np.where(padded, inf, 0).astype(np.int32)
        for r in range(h):
            for c in range(w):
                if d[r, c] == 0:
                    continue
                if r > 0:
                    d[r, c] = min(d[r, c], d[r - 1, c] + 1)
                if c > 0:
                    d[r, c] = min(d[r, c], d[r, c - 1] + 1)
        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                if r + 1 < h:
                    d[r, c] = min(d[r, c], d[r + 1, c] + 1)
                if c + 1 < w:
                    d[r, c] = min(d[r, c], d[r, c + 1] + 1)
        return d[1:-1, 1:-1].astype(np.float32)


class ArmFeasibilityBasin:
    """Loads IK hints and provides exact cell centers and basin peaks."""

    REQUIRED_KEYS = (
        "region_ids",
        "part_ids",
        "left_scores",
        "right_scores",
        "scores",
        "geometry_valid",
        "computed_mask",
        "resolution",
    )

    def __init__(self, searcher, path: str):
        self.searcher = searcher
        self.path = str(Path(path).expanduser().resolve())
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        with np.load(self.path, allow_pickle=True) as npz:
            missing = [k for k in self.REQUIRED_KEYS if k not in npz.files]
            if missing:
                raise ValueError(f"IK hint missing keys: {missing}")
            self.region_ids = [str(v) for v in npz["region_ids"].tolist()]
            self.part_ids = [str(v) for v in npz["part_ids"].tolist()]
            self.left_scores = np.asarray(npz["left_scores"], dtype=np.float32)
            self.right_scores = np.asarray(npz["right_scores"], dtype=np.float32)
            self.scores = np.asarray(npz["scores"], dtype=np.float32)
            self.geometry_valid = np.asarray(npz["geometry_valid"], dtype=bool)
            self.computed_mask = np.asarray(npz["computed_mask"], dtype=bool)
            self.resolution = float(np.asarray(npz["resolution"]).item())

        expected = self.left_scores.shape
        for name, arr in (
            ("right_scores", self.right_scores),
            ("scores", self.scores),
            ("geometry_valid", self.geometry_valid),
            ("computed_mask", self.computed_mask),
        ):
            if arr.shape != expected:
                raise ValueError(f"{name} shape={arr.shape}, expected={expected}")
        if expected[0] != len(self.region_ids) or expected[1] != len(self.part_ids):
            raise ValueError("IK hint region/part metadata do not match tensor shape")

        self.region_to_index = {rid: i for i, rid in enumerate(self.region_ids)}
        self.part_to_index = {pid: i for i, pid in enumerate(self.part_ids)}
        self.max_poses = int(expected[2])
        self.height = int(expected[3])
        self.width = int(expected[4])
        self._cell_xy: Dict[str, np.ndarray] = {}
        self._margin_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}

        validator, task, prepared = build_assembly_validator(
            asmdef_path=searcher.asmdef_path,
            config_yaml=searcher.config_yaml,
            grasp_dir=searcher.grasp_dir,
            cdprim_type=searcher.cdprim_type,
            planner_obstacle_mode=searcher.planner_obstacle_mode,
            max_poses=self.max_poses,
            force_first_at_region_center=True,
        )
        self._prepared = prepared
        for rid in self.region_ids:
            state = AssemblyGridState(
                validator,
                task,
                region_id=rid,
                resolution=self.resolution,
                max_poses=self.max_poses,
            )
            region_index = self.region_to_index[rid]
            row_offset, col_offset, mismatch_rate = self._align_current_grid(
                state=state,
                task=task,
                region_index=region_index,
                region_id=rid,
            )
            centers = np.zeros((self.height, self.width, 2), dtype=np.float32)
            for row in range(self.height):
                for col in range(self.width):
                    centers[row, col] = np.asarray(
                        state.workspace.spec.cell_center(
                            row + row_offset,
                            col + col_offset,
                        ),
                        dtype=np.float32,
                    )
            self._cell_xy[rid] = centers
            print(
                f"[basin-grid] region={rid} "
                f"current={int(state.workspace.spec.height)}x"
                f"{int(state.workspace.spec.width)} "
                f"cache={self.height}x{self.width} "
                f"offset=({row_offset},{col_offset}) "
                f"geometry_mismatch={mismatch_rate:.6f}"
            )

    def _align_current_grid(
        self,
        state,
        task,
        region_index: int,
        region_id: str,
    ) -> Tuple[int, int, float]:
        """Align a regenerated workspace grid to the cached IK tensor.

        A table interval whose length is very close to an integer multiple of
        ``resolution`` can become 54 or 55 cells after a small floating-point
        or boundary-policy change. The IK values remain valid; only the index
        origin may be shifted by one boundary cell.

        We therefore compare regenerated ``flat_action_mask`` geometry against
        the cached ``geometry_valid`` tensor and select the contiguous crop with
        the lowest XOR mismatch. No IK score is recomputed or interpolated.
        """
        current_h = int(state.workspace.spec.height)
        current_w = int(state.workspace.spec.width)
        if current_h < self.height or current_w < self.width:
            raise ValueError(
                f"{region_id}: current grid {current_h}x{current_w} is smaller "
                f"than cached grid {self.height}x{self.width}; "
                "the IK cache was generated with incompatible workspace bounds"
            )

        row_offsets = range(current_h - self.height + 1)
        col_offsets = range(current_w - self.width + 1)

        current_masks: Dict[str, np.ndarray] = {}
        old_step = int(getattr(state, "current_step", 0))
        try:
            for part_id in self.part_ids:
                if part_id not in task.decision_parts:
                    continue
                state.current_step = int(task.decision_parts.index(part_id))
                flat = np.asarray(state.flat_action_mask(), dtype=bool)
                expected_size = self.max_poses * current_h * current_w
                if flat.size != expected_size:
                    raise ValueError(
                        f"{region_id}/{part_id}: current action mask size "
                        f"{flat.size}, expected {expected_size}"
                    )
                current_masks[part_id] = flat.reshape(
                    self.max_poses,
                    current_h,
                    current_w,
                )
        finally:
            state.current_step = old_step

        best_key: Optional[Tuple[int, int, int]] = None
        best_compared = 0
        for row_offset in row_offsets:
            for col_offset in col_offsets:
                mismatch = 0
                compared = 0
                for part_id, current in current_masks.items():
                    part_index = self.part_to_index[part_id]
                    cropped = current[
                        :,
                        row_offset : row_offset + self.height,
                        col_offset : col_offset + self.width,
                    ]
                    cached = self.geometry_valid[region_index, part_index]
                    mismatch += int(np.count_nonzero(cropped ^ cached))
                    compared += int(cached.size)

                # Stable tie break: prefer the smallest crop offset.
                key = (mismatch, int(row_offset), int(col_offset))
                if best_key is None or key < best_key:
                    best_key = key
                    best_compared = compared

        if best_key is None:
            raise RuntimeError(f"{region_id}: failed to align workspace grid")

        mismatch, row_offset, col_offset = best_key
        mismatch_rate = (
            float(mismatch) / float(best_compared)
            if best_compared > 0
            else 0.0
        )
        if mismatch_rate > 0.05:
            print(
                f"[basin-grid][WARN] region={region_id} best geometry "
                f"mismatch={mismatch_rate:.3%}; cache and current geometry "
                "may differ beyond a one-cell boundary shift"
            )
        return int(row_offset), int(col_offset), float(mismatch_rate)

    def has_region(self, region_id: str) -> bool:
        return str(region_id) in self.region_to_index

    def _score_array(
        self,
        region_index: int,
        part_index: int,
        arm_id: int,
        pose_id: int,
    ) -> np.ndarray:
        source = self.left_scores if int(arm_id) == 0 else self.right_scores
        return source[region_index, part_index, pose_id]

    def _margin_array(
        self,
        region_index: int,
        part_index: int,
        arm_id: int,
        pose_id: int,
        min_score: float,
    ) -> np.ndarray:
        key = (region_index, part_index, arm_id, pose_id)
        cached = self._margin_cache.get(key)
        if cached is not None:
            return cached
        score = self._score_array(region_index, part_index, arm_id, pose_id)
        valid = (
            self.geometry_valid[region_index, part_index, pose_id]
            & self.computed_mask[region_index, part_index, pose_id]
            & np.isfinite(score)
            & (score > float(min_score))
        )
        margin = _distance_transform(valid) * self.resolution
        self._margin_cache[key] = margin.astype(np.float32)
        return self._margin_cache[key]

    def peaks(
        self,
        region_id: str,
        part_id: str,
        allowed_pose_count: int,
        limit: int,
        min_score: float,
        margin_cap: float,
        margin_weight: float,
        hint_weight: float,
        nms_cells: int,
    ) -> List[BasinPoint]:
        if region_id not in self.region_to_index:
            return []
        if part_id not in self.part_to_index:
            return []
        r = self.region_to_index[region_id]
        p = self.part_to_index[part_id]
        candidates: List[BasinPoint] = []
        max_pose = min(int(allowed_pose_count), self.max_poses)

        for arm_id in (0, 1):
            for pose_id in range(max_pose):
                score = self._score_array(r, p, arm_id, pose_id)
                margin = self._margin_array(
                    r,
                    p,
                    arm_id,
                    pose_id,
                    min_score,
                )
                valid = (
                    self.geometry_valid[r, p, pose_id]
                    & self.computed_mask[r, p, pose_id]
                    & np.isfinite(score)
                    & (score > float(min_score))
                )
                rows, cols = np.nonzero(valid)
                if rows.size == 0:
                    continue
                norm_margin = np.clip(
                    margin[rows, cols] / max(float(margin_cap), 1e-9),
                    0.0,
                    1.0,
                )
                utility = (
                    float(hint_weight) * score[rows, cols]
                    + float(margin_weight) * norm_margin
                )
                order = np.argsort(-utility)
                per_map_cap = max(int(limit) * 3, int(limit))
                for local_i in order[:per_map_cap]:
                    row = int(rows[local_i])
                    col = int(cols[local_i])
                    xy = self._cell_xy[region_id][row, col]
                    candidates.append(
                        BasinPoint(
                            part_id=part_id,
                            arm_id=arm_id,
                            pose_id=pose_id,
                            row=row,
                            col=col,
                            xy=(float(xy[0]), float(xy[1])),
                            hint_score=float(score[row, col]),
                            margin_m=float(margin[row, col]),
                            utility=float(utility[local_i]),
                        )
                    )

        candidates.sort(
            key=lambda q: (q.utility, q.margin_m, q.hint_score),
            reverse=True,
        )
        selected: List[BasinPoint] = []
        radius2 = int(nms_cells) ** 2
        for point in candidates:
            too_close = False
            for old in selected:
                if (
                    point.arm_id == old.arm_id
                    and point.pose_id == old.pose_id
                    and (point.row - old.row) ** 2
                    + (point.col - old.col) ** 2
                    <= radius2
                ):
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(point)
            if len(selected) >= int(limit):
                break
        return selected

    def nearest_metrics(
        self,
        region_id: str,
        part_id: str,
        xy: Sequence[float],
        arm_tag: Optional[str] = None,
    ) -> Dict[str, float]:
        if region_id not in self.region_to_index or part_id not in self.part_to_index:
            return {"hint_score": 0.0, "margin_m": 0.0}
        centers = self._cell_xy[region_id]
        delta = centers - np.asarray(xy, dtype=np.float32).reshape(1, 1, 2)
        flat_index = int(np.argmin(np.sum(delta * delta, axis=2)))
        row, col = np.unravel_index(flat_index, (self.height, self.width))
        r = self.region_to_index[region_id]
        p = self.part_to_index[part_id]

        if str(arm_tag) == "lft":
            arms = (0,)
        elif str(arm_tag) == "rgt":
            arms = (1,)
        else:
            arms = (0, 1)

        best_score = 0.0
        best_margin = 0.0
        best_pose = -1
        best_arm = -1
        for arm_id in arms:
            for pose_id in range(self.max_poses):
                score = float(
                    self._score_array(r, p, arm_id, pose_id)[row, col]
                )
                margin = float(
                    self._margin_array(
                        r,
                        p,
                        arm_id,
                        pose_id,
                        float(BCFG["min_hint_score"]),
                    )[row, col]
                )
                key = (margin, score)
                if key > (best_margin, best_score):
                    best_score = score
                    best_margin = margin
                    best_pose = pose_id
                    best_arm = arm_id
        return {
            "hint_score": best_score,
            "margin_m": best_margin,
            "nearest_row": int(row),
            "nearest_col": int(col),
            "best_pose_id": int(best_pose),
            "best_arm_id": int(best_arm),
        }


class SequenceBasinSearcher(gmod.GlobalLayoutSearcher):
    """Sequence beam search over arm-conditioned feasibility basins."""

    def _resolve_hint_path(self) -> str:
        explicit = BCFG.get("ik_hint")
        if explicit:
            return str(explicit)
        if os.path.isfile(DEFAULT_MERGED_HINT):
            return DEFAULT_MERGED_HINT
        return DEFAULT_SINGLE_HINT

    def _selected_regions(
        self,
        all_regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
        basin: ArmFeasibilityBasin,
    ) -> List[Tuple[str, Tuple[int, int], np.ndarray]]:
        requested = BCFG.get("regions")
        requested_set = None
        if requested:
            requested_set = {
                token.strip()
                for token in str(requested).replace(";", ",").split(",")
                if token.strip()
            }
        out = []
        for region in all_regions:
            rid = str(region[0])
            if not basin.has_region(rid):
                continue
            if requested_set is not None and rid not in requested_set:
                continue
            out.append(region)
        return out

    def _initialize_partial_geometry(
        self,
        region: Tuple[str, Tuple[int, int], np.ndarray],
        state: BeamState,
    ) -> List[str]:
        self._set_region_from_tuple(region)
        self._apply_first_part_as_assembled()
        active: List[str] = []
        first = self._first_part_id() if self.preassemble_first_part else None
        if first is not None and first in self.staging_models:
            active.append(first)

        for pid in self.part_order:
            if pid == first or pid not in state.xy:
                continue
            pose_id = int(state.pose_ids.get(pid, 0))
            cands = self.rot_cands.get(pid, [])
            if not cands:
                continue
            pose_id = int(np.clip(pose_id, 0, len(cands) - 1))
            self._apply_staging_pose(pid, state.xy[pid], cands[pose_id])
            active.append(pid)
        return active

    def _partial_accept(
        self,
        region: Tuple[str, Tuple[int, int], np.ndarray],
        state: BeamState,
        point: BasinPoint,
        previous_part: Optional[str],
    ) -> bool:
        pid = point.part_id
        cands = self.rot_cands.get(pid, [])
        if not cands or point.pose_id >= len(cands):
            return False
        xy = np.asarray(point.xy, dtype=float)
        cand = cands[point.pose_id]

        if (
            previous_part is not None
            and bool(getattr(self, "enforce_order_x_constraint", False))
            and previous_part in state.xy
        ):
            prev_x = float(state.xy[previous_part][0])
            tolerance = float(getattr(self, "order_x_tolerance", 0.0))
            if float(xy[0]) > prev_x + tolerance:
                return False

        active = self._initialize_partial_geometry(region, state)
        if self._staging_arm_keepout_reason(pid, xy, cand):
            return False
        self._apply_staging_pose(pid, xy, cand)

        for other in active:
            if other in self.staging_models and self.staging_models[pid].is_mcdwith(
                self.staging_models[other]
            ):
                return False

        active_with_current = active + [pid]
        if self._mesh_clearance_reason(active_pids=active_with_current):
            return False
        if self._robot_home_collision_reason(active_pids=[pid]):
            return False
        if self._robot_home_clearance_reason(active_pids=[pid]):
            return False
        return True

    def _rank_state(self, state: BeamState, decision_count: int) -> float:
        if not state.hint_scores:
            return -np.inf
        min_hint = float(min(state.hint_scores))
        mean_hint = float(np.mean(state.hint_scores))
        min_margin = float(min(state.margins))
        norm_margin = float(
            np.clip(
                min_margin / max(float(BCFG["margin_cap"]), 1e-9),
                0.0,
                1.0,
            )
        )
        distance_score = math.exp(
            -state.distance_sum / max(0.55 * max(decision_count, 1), 1e-9)
        )
        return float(
            float(BCFG["margin_weight"]) * norm_margin
            + float(BCFG["hint_weight"]) * (0.65 * min_hint + 0.35 * mean_hint)
            + float(BCFG["distance_weight"]) * distance_score
        )

    def _beam_region(
        self,
        basin: ArmFeasibilityBasin,
        region: Tuple[str, Tuple[int, int], np.ndarray],
        verbose: bool,
    ) -> List[BeamState]:
        rid = str(region[0])
        self._set_region_from_tuple(region)
        first = self._first_part_id() if self.preassemble_first_part else None
        initial_xy: Dict[str, np.ndarray] = {}
        if first is not None and first in self.world_poses:
            initial_xy[first] = np.asarray(
                self.world_poses[first][0][:2],
                dtype=float,
            )
        beam = [BeamState(xy=initial_xy)]
        decision_parts = [pid for pid in self.part_order if pid != first]
        previous: Optional[str] = None

        for depth, pid in enumerate(decision_parts, start=1):
            points = basin.peaks(
                region_id=rid,
                part_id=pid,
                allowed_pose_count=len(self.rot_cands.get(pid, [])),
                limit=int(BCFG["part_candidates"]),
                min_score=float(BCFG["min_hint_score"]),
                margin_cap=float(BCFG["margin_cap"]),
                margin_weight=float(BCFG["margin_weight"]),
                hint_weight=float(BCFG["hint_weight"]),
                nms_cells=int(BCFG["nms_cells"]),
            )
            if not points:
                if verbose:
                    print(f"[basin/{rid}] no candidate peaks for {pid}")
                return []

            expanded: List[BeamState] = []
            for state in beam:
                for point in points:
                    if not self._partial_accept(region, state, point, previous):
                        continue
                    child = BeamState(
                        xy={k: np.asarray(v, dtype=float).copy() for k, v in state.xy.items()},
                        pose_ids=dict(state.pose_ids),
                        arm_ids=dict(state.arm_ids),
                        hint_scores=list(state.hint_scores),
                        margins=list(state.margins),
                        distance_sum=float(state.distance_sum),
                    )
                    child.xy[pid] = np.asarray(point.xy, dtype=float)
                    child.pose_ids[pid] = int(point.pose_id)
                    child.arm_ids[pid] = int(point.arm_id)
                    child.hint_scores.append(float(point.hint_score))
                    child.margins.append(float(point.margin_m))
                    goal_xy = np.asarray(self.world_poses[pid][0][:2], dtype=float)
                    child.distance_sum += float(
                        np.linalg.norm(child.xy[pid] - goal_xy)
                    )
                    child.rank_score = self._rank_state(
                        child,
                        len(decision_parts),
                    )
                    expanded.append(child)

            if not expanded:
                if verbose:
                    print(
                        f"[basin/{rid}] beam collapsed at "
                        f"step={depth} part={pid}"
                    )
                return []

            expanded.sort(key=lambda s: s.rank_score, reverse=True)
            unique: List[BeamState] = []
            seen = set()
            for state in expanded:
                fp = state.fingerprint(basin.resolution)
                if fp in seen:
                    continue
                seen.add(fp)
                unique.append(state)
                if len(unique) >= int(BCFG["beam_width"]):
                    break
            beam = unique
            previous = pid

            if verbose:
                best = beam[0]
                print(
                    f"[basin/{rid}] step={depth}/{len(decision_parts)} "
                    f"part={pid:16s} peaks={len(points):3d} "
                    f"expanded={len(expanded):4d} beam={len(beam):3d} "
                    f"rank={best.rank_score:.4f} "
                    f"min_margin={min(best.margins):.3f}m "
                    f"min_hint={min(best.hint_scores):.3f}"
                )
        return beam

    def _candidate_report(
        self,
        basin: ArmFeasibilityBasin,
        cand: LayoutCandidate,
    ) -> Dict:
        rows = {}
        first = self._first_part_id() if self.preassemble_first_part else None
        for pid in self.part_order:
            if pid == first or pid not in cand.xy:
                continue
            rows[pid] = basin.nearest_metrics(
                cand.assembly_region_id,
                pid,
                cand.xy[pid],
                cand.arm_choice.get(pid),
            )
        margins = [
            float(row["margin_m"])
            for row in rows.values()
        ]
        hints = [
            float(row["hint_score"])
            for row in rows.values()
        ]
        return {
            "assembly_region_id": str(cand.assembly_region_id),
            "layout_score": float(cand.layout_score),
            "l2_pass": bool(cand.l2_pass),
            "l3_pass": bool(cand.l3_pass),
            "min_basin_margin_m": min(margins) if margins else 0.0,
            "mean_basin_margin_m": float(np.mean(margins)) if margins else 0.0,
            "min_hint_score": min(hints) if hints else 0.0,
            "mean_hint_score": float(np.mean(hints)) if hints else 0.0,
            "arm_choice": dict(cand.arm_choice),
            "grasp_counts": dict(cand.grasp_counts),
            "parts": rows,
        }

    def _robust_validate(
        self,
        basin: ArmFeasibilityBasin,
        cand: LayoutCandidate,
        region: Tuple[str, Tuple[int, int], np.ndarray],
        seed: int,
    ) -> Dict:
        n = int(BCFG["robust_evals"])
        if n <= 0:
            return {
                "enabled": False,
                "evaluations": 0,
                "passes": 0,
                "pass_rate": None,
            }
        rng = np.random.default_rng(seed + 91073)
        first = self._first_part_id() if self.preassemble_first_part else None
        passes = 0
        records = []
        # Robustness trials are post-search analysis. They are intentionally
        # evaluated outside the optimization budget and reported separately.
        old_cap = fast.MAX_GRASPS_PER_POSE
        for i in range(n):
            trial_xy = nsga2._copy_xy(cand.xy)
            for pid in self.part_order:
                if pid == first or pid not in trial_xy:
                    continue
                jitter = rng.normal(
                    0.0,
                    float(BCFG["robust_sigma"]),
                    size=2,
                )
                trial_xy[pid] = self._clip_xy_for_part(
                    pid,
                    np.asarray(trial_xy[pid], dtype=float) + jitter,
                )
            self._set_region_from_tuple(region)
            try:
                fast._pose_cache_reset_for_layout()
            except Exception:
                pass
            child = LayoutCandidate(xy=trial_xy)
            t0 = time.time()
            ok = bool(self.evaluate_layout(child))
            elapsed = float(time.time() - t0)
            passes += int(ok)
            records.append(
                {
                    "index": i,
                    "pass": ok,
                    "score": float(getattr(child, "layout_score", -1.0)),
                    "fail_part": getattr(child, "fail_part", None),
                    "fail_reason": getattr(child, "fail_reason", ""),
                    "elapsed_seconds": elapsed,
                }
            )
        fast.MAX_GRASPS_PER_POSE = old_cap
        done = len(records)
        return {
            "enabled": True,
            "sigma_m": float(BCFG["robust_sigma"]),
            "evaluations": done,
            "passes": passes,
            "pass_rate": float(passes / done) if done else None,
            "records": records,
        }


    def _select_exact_proposals(
        self,
        proposals: Sequence[Tuple[float, BeamState, Tuple]],
        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
    ) -> List[Tuple[float, BeamState, Tuple]]:
        """Select exact-L2 proposals with optional per-region balancing.

        exact_per_region=0 keeps the original global top-K behavior.
        A positive value performs rank-wise round-robin selection across
        regions first, then fills any remaining slots using global ranking.
        """
        limit = min(len(proposals), int(BCFG["exact_proposals"]))
        if limit <= 0:
            return []

        per_region = int(BCFG.get("exact_per_region", 0))
        ranked = sorted(proposals, key=lambda row: row[0], reverse=True)
        if per_region <= 0:
            return ranked[:limit]

        grouped: Dict[str, List[Tuple[float, BeamState, Tuple]]] = {
            str(region[0]): [] for region in regions
        }
        for item in ranked:
            grouped.setdefault(str(item[2][0]), []).append(item)

        selected: List[Tuple[float, BeamState, Tuple]] = []
        selected_ids = set()

        # Rank-wise round robin gives each region equal opportunity.
        for local_rank in range(per_region):
            for region in regions:
                rid = str(region[0])
                group = grouped.get(rid, [])
                if local_rank >= len(group):
                    continue
                item = group[local_rank]
                key = id(item[1])
                if key in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(key)
                if len(selected) >= limit:
                    return selected

        # Fill spare capacity by global prior rank.
        for item in ranked:
            key = id(item[1])
            if key in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(key)
            if len(selected) >= limit:
                break
        return selected

    def _full_grasp_certify(
        self,
        ranked: Sequence[LayoutCandidate],
        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
    ) -> Tuple[List[LayoutCandidate], List[Dict]]:
        """Re-evaluate final Top-K with a full or larger grasp set.

        This bypasses the NSGA evaluation cache because the earlier score may
        have been computed using the fast grasp cap. The certification stage
        is intentionally outside the search budget and is reported separately.
        """
        topk = min(int(BCFG.get("certify_topk", 0)), len(ranked))
        if topk <= 0:
            return list(ranked), []

        cap = int(BCFG.get("certify_grasp_cap", 0))
        rows: List[Dict] = []
        certified: List[LayoutCandidate] = []

        print(
            "\n---------- Full-grasp final certification "
            f"(top_k={topk}, cap={cap}; 0=all grasps) ----------"
        )

        old_cap = fast.MAX_GRASPS_PER_POSE
        try:
            fast.MAX_GRASPS_PER_POSE = cap
            for rank, src in enumerate(ranked[:topk], start=1):
                region = getattr(src, "_nsga_region_tuple", None)
                if region is None:
                    region = self._region_by_id(
                        regions,
                        src.assembly_region_id,
                    )
                self._set_region_from_tuple(region)
                try:
                    fast._pose_cache_reset_for_layout()
                except Exception:
                    pass

                fresh = LayoutCandidate(xy=nsga2._copy_xy(src.xy))
                t0 = time.time()
                ok = bool(self.evaluate_layout(fresh))
                elapsed = float(time.time() - t0)
                if ok:
                    setattr(fresh, "_nsga_region_tuple", region)
                    certified.append(fresh)

                rows.append(
                    {
                        "rank": rank,
                        "region_id": str(src.assembly_region_id),
                        "fast_score": float(src.layout_score),
                        "certified": ok,
                        "certified_score": (
                            float(fresh.layout_score) if ok else None
                        ),
                        "elapsed_seconds": elapsed,
                        "fail_part": getattr(fresh, "fail_part", None),
                        "fail_reason": getattr(fresh, "fail_reason", ""),
                        "grasp_cap": cap,
                    }
                )
                print(
                    f"[certify] {rank:02d}/{topk} "
                    f"region={src.assembly_region_id} "
                    f"fast={float(src.layout_score):.4f} "
                    f"{'PASS' if ok else 'FAIL'} "
                    f"exact={float(fresh.layout_score) if ok else float('nan'):.4f} "
                    f"time={elapsed:.1f}s"
                )
        finally:
            fast.MAX_GRASPS_PER_POSE = old_cap

        certified.sort(
            key=lambda cand: float(cand.layout_score),
            reverse=True,
        )
        if not certified:
            print(
                "[certify][WARN] no Top-K candidate passed certification; "
                "keeping the fast-ranked candidates."
            )
            return list(ranked), rows

        # Certified candidates come first; keep remaining fast candidates only
        # as fallback choices for optional L3.
        certified_ids = {
            (
                cand.assembly_region_id,
                tuple(
                    (pid, tuple(np.round(xy, 6)))
                    for pid, xy in sorted(cand.xy.items())
                ),
            )
            for cand in certified
        }
        tail = []
        for cand in ranked:
            key = (
                cand.assembly_region_id,
                tuple(
                    (pid, tuple(np.round(xy, 6)))
                    for pid, xy in sorted(cand.xy.items())
                ),
            )
            if key not in certified_ids:
                tail.append(cand)
        return certified + tail, rows

    def _write_basin_report(self, report: Mapping) -> Optional[str]:
        explicit = BCFG.get("report_json")
        if explicit:
            output = Path(str(explicit)).expanduser().resolve()
        else:
            output_dir = _argv_value(
                "--output-dir",
                fol.DEFAULT_OUTPUT_DIR,
            )
            output = Path(output_dir).expanduser().resolve() / (
                f"{self.output_name}_basin_report.json"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] basin report -> {output}")
        return str(output)

    def random_search(
        self,
        n_samples: int,
        seed: int,
        max_resample_layout: int = 80,
        verbose: bool = True,
        enable_l3: bool = False,
        l3_top_k: int = 3,
        l3_obstacle_mode: str = "staging_aware",
        require_l3: bool = True,
    ) -> Optional[LayoutCandidate]:
        del max_resample_layout
        t0 = time.time()
        self._reset_eval_progress_stats()
        hint_path = self._resolve_hint_path()
        basin = ArmFeasibilityBasin(self, hint_path)

        all_regions = self._order_regions_center_first(
            self._assembly_region_candidates(),
            verbose=verbose,
        )
        regions = self._selected_regions(all_regions, basin)
        if not regions:
            raise RuntimeError(
                "No overlap between strict-search assembly regions and "
                f"IK-hint regions={basin.region_ids}"
            )

        print("\n========== Sequence-Aware Feasibility-Basin Search ==========")
        print(f"IK hint            = {hint_path}")
        print(f"hint regions       = {basin.region_ids}")
        print(f"searched regions   = {[r[0] for r in regions]}")
        print(f"beam width         = {BCFG['beam_width']}")
        print(f"part candidates    = {BCFG['part_candidates']}")
        print(f"exact proposals    = {BCFG['exact_proposals']}")
        print(f"margin rank cap     = {BCFG['margin_cap']} m  # ranking saturation, not a hard constraint")
        print(f"max evaluations    = {nsga2._CFG.get('max_evals')}")

        proposals: List[Tuple[float, BeamState, Tuple]] = []
        region_stats: Dict[str, Dict] = {}
        for region in regions:
            rid = str(region[0])
            beam = self._beam_region(basin, region, verbose)
            region_stats[rid] = {
                "complete_beam_count": len(beam),
                "best_beam_rank": float(beam[0].rank_score) if beam else None,
            }
            for state in beam:
                proposals.append((state.rank_score, state, region))

        proposals.sort(key=lambda row: row[0], reverse=True)
        selected_proposals = self._select_exact_proposals(
            proposals,
            regions,
        )
        proposal_limit = len(selected_proposals)
        feasible: List[LayoutCandidate] = []
        exact_rows = []

        print("\n---------- Exact L2 evaluation of basin proposals ----------")
        print(
            f"selection mode      = "
            f"{'balanced' if int(BCFG.get('exact_per_region', 0)) > 0 else 'global-top-k'}"
        )
        print(f"exact per region    = {BCFG.get('exact_per_region', 0)}")
        for rank, (prior_rank, state, region) in enumerate(
            selected_proposals,
            start=1,
        ):
            if self._eval_budget_exhausted():
                break
            cand = self._evaluate_gene(state.xy, region)
            ok = bool(getattr(cand, "l2_pass", False))
            exact_rows.append(
                {
                    "rank": rank,
                    "region_id": str(region[0]),
                    "prior_rank_score": float(prior_rank),
                    "l2_pass": ok,
                    "layout_score": float(getattr(cand, "layout_score", -1.0)),
                    "fail_part": getattr(cand, "fail_part", None),
                    "fail_reason": getattr(cand, "fail_reason", ""),
                    "beam_min_margin_m": float(min(state.margins)) if state.margins else 0.0,
                    "beam_min_hint_score": float(min(state.hint_scores)) if state.hint_scores else 0.0,
                }
            )
            rid = str(region[0])
            region_stats.setdefault(rid, {})
            region_stats[rid]["exact_attempted"] = int(
                region_stats[rid].get("exact_attempted", 0)
            ) + 1
            region_stats[rid]["exact_passed"] = int(
                region_stats[rid].get("exact_passed", 0)
            ) + int(ok)

            print(
                f"[exact] {rank:03d}/{proposal_limit} "
                f"region={region[0]} prior={prior_rank:.4f} "
                f"{'L2_OK' if ok else 'FAIL ':5s} "
                f"score={float(getattr(cand, 'layout_score', -1.0)):.4f}"
            )
            if ok:
                feasible.append(cand)

        if not feasible and int(BCFG["fallback_explore"]) > 0:
            print(
                "\n[basin] no exact feasible proposal; "
                "falling back to the existing global explorer."
            )
            feasible = self._global_explore(
                np.random.default_rng(seed),
                regions,
                int(BCFG["fallback_explore"]),
                verbose,
            )

        if not feasible:
            report = {
                "method": "sequence_aware_feasibility_basin_minimal",
                "success": False,
                "ik_hint": hint_path,
                "regions": region_stats,
                "exact_proposals": exact_rows,
                "wall_seconds": time.time() - t0,
            }
            self._write_basin_report(report)
            return None

        feasible.sort(key=lambda c: float(c.layout_score), reverse=True)
        elites = self._unique_elites(
            feasible,
            limit=max(int(BCFG["elite"]), int(l3_top_k)),
        )
        refined: List[LayoutCandidate] = []

        print("\n---------- Existing multiscale pattern refinement ----------")
        for rank, cand in enumerate(elites[: int(BCFG["elite"])], start=1):
            if self._eval_budget_exhausted():
                refined.append(cand)
                continue
            print(
                f"[refine] elite#{rank} "
                f"start={float(cand.layout_score):.4f}"
            )
            refined.append(
                self._pattern_refine(
                    cand,
                    steps=[float(v) for v in gmod.GCFG["refine_steps"]],
                    rounds=int(gmod.GCFG["refine_rounds"]),
                    diagonal=bool(gmod.GCFG["refine_diagonal"]),
                    verbose=verbose,
                )
            )

        pool = [
            cand
            for cand in feasible + refined
            if bool(getattr(cand, "l2_pass", False))
        ]
        ranked = self._unique_elites(
            pool,
            limit=max(int(BCFG["elite"]), int(l3_top_k)),
        )
        ranked.sort(key=lambda c: float(c.layout_score), reverse=True)
        if not ranked:
            return None

        ranked, certification_rows = self._full_grasp_certify(
            ranked,
            regions,
        )
        best = ranked[0]
        region = getattr(best, "_nsga_region_tuple", None)
        if region is None:
            region = self._region_by_id(regions, best.assembly_region_id)

        if enable_l3:
            print("\n---------- Optional L3 validation ----------")
            accepted = None
            for rank, cand in enumerate(
                ranked[: min(int(l3_top_k), len(ranked))],
                start=1,
            ):
                print(
                    f"[L3] rank={rank} "
                    f"score={float(cand.layout_score):.4f}"
                )
                if self.validate_full_sequence_l3(
                    cand,
                    obstacle_mode=l3_obstacle_mode,
                    verbose=True,
                ):
                    accepted = cand
                    break
            if accepted is not None:
                best = accepted
            elif require_l3:
                print("[FAIL] all requested L3 candidates failed.")
                return None

        robust = self._robust_validate(basin, best, region, seed)
        report = {
            "method": "sequence_aware_feasibility_basin_minimal",
            "success": True,
            "ik_hint": hint_path,
            "searched_regions": [str(r[0]) for r in regions],
            "configuration": dict(BCFG),
            "region_stats": region_stats,
            "exact_proposals": exact_rows,
            "full_grasp_certification": certification_rows,
            "best": self._candidate_report(basin, best),
            "robust_validation": robust,
            "real_evaluations": int(getattr(self, "_nsga_eval_count", 0)),
            "eval_cache_hits": int(getattr(self, "_nsga_cache_hits", 0)),
            "wall_seconds": float(time.time() - t0),
        }
        self._write_basin_report(report)

        print("\n========== Basin Search Summary ==========")
        print(
            f"best score          = {float(best.layout_score):.4f}"
        )
        print(f"best region         = {best.assembly_region_id}")
        best_metrics = report["best"]
        print(
            f"min basin margin    = "
            f"{best_metrics['min_basin_margin_m']:.4f} m"
        )
        print(
            f"mean basin margin   = "
            f"{best_metrics['mean_basin_margin_m']:.4f} m"
        )
        print(
            f"real evaluations    = "
            f"{getattr(self, '_nsga_eval_count', 0)}"
        )
        print(f"wall time           = {time.time() - t0:.1f}s")
        return best


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _argv_value(name: str, default: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return str(default)
    if index + 1 >= len(sys.argv):
        return str(default)
    return str(sys.argv[index + 1])


def _consume_basin_args() -> None:
    scalar_args = {
        "--basin-ik-hint": ("ik_hint", str),
        "--basin-regions": ("regions", str),
        "--basin-beam-width": ("beam_width", int),
        "--basin-part-candidates": ("part_candidates", int),
        "--basin-exact-proposals": ("exact_proposals", int),
        "--basin-exact-per-region": ("exact_per_region", int),
        "--basin-elite": ("elite", int),
        "--basin-certify-topk": ("certify_topk", int),
        "--basin-certify-grasp-cap": ("certify_grasp_cap", int),
        "--basin-min-hint-score": ("min_hint_score", float),
        "--basin-margin-cap": ("margin_cap", float),
        "--basin-margin-weight": ("margin_weight", float),
        "--basin-hint-weight": ("hint_weight", float),
        "--basin-distance-weight": ("distance_weight", float),
        "--basin-nms-cells": ("nms_cells", int),
        "--basin-fallback-explore": ("fallback_explore", int),
        "--basin-robust-evals": ("robust_evals", int),
        "--basin-robust-sigma": ("robust_sigma", float),
        "--basin-report-json": ("report_json", str),
    }
    for cli_name, (key, cast) in scalar_args.items():
        value = fast._consume_extra_value(cli_name)
        if value is not None:
            BCFG[key] = cast(value)

    if int(BCFG["beam_width"]) <= 0:
        raise ValueError("--basin-beam-width must be positive")
    if int(BCFG["part_candidates"]) <= 0:
        raise ValueError("--basin-part-candidates must be positive")
    if int(BCFG["exact_proposals"]) <= 0:
        raise ValueError("--basin-exact-proposals must be positive")
    if int(BCFG.get("exact_per_region", 0)) < 0:
        raise ValueError("--basin-exact-per-region must be non-negative")
    if int(BCFG.get("certify_topk", 0)) < 0:
        raise ValueError("--basin-certify-topk must be non-negative")
    if int(BCFG.get("certify_grasp_cap", 0)) < 0:
        raise ValueError("--basin-certify-grasp-cap must be non-negative")
    if float(BCFG["margin_cap"]) <= 0:
        raise ValueError("--basin-margin-cap must be positive")


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = SequenceBasinSearcher


def main() -> None:
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()

    # Reuse existing global flags for refinement and evaluation budget.
    gmod._consume_global_args()
    _consume_basin_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    print("[sequence-basin] config:")
    for key, value in BCFG.items():
        print(f"    {key:24s} = {value}")
    print(f"    {'refine_steps':24s} = {gmod.GCFG['refine_steps']}")
    print(f"    {'max_evals':24s} = {nsga2._CFG.get('max_evals')}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(
            f"[sequence-basin] wall-clock total = "
            f"{time.perf_counter() - wall_t0:.3f}s"
        )
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
