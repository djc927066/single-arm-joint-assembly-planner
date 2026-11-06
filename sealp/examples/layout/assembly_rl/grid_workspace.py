#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discrete workspace and action masking for the Tower RL layout task.

This is step 2 of the RL/neural-combinatorial layout pipeline.
It does *not* train a network and it does *not* run L2 by itself.  It converts
continuous table coordinates into a 2 cm grid, tracks the footprint of already
placed parts, and produces legal pose-position action masks.

Coordinate convention
---------------------
* ``col`` / ``grid_x`` increases with world X.
* ``row`` / ``grid_y`` increases with world Y.
* A discrete action is ``(pose_id, row, col)``.
* The flattened action index is::

      action = pose_id * (H * W) + row * W + col

The selected cell denotes the *center* of the part footprint.  Z is not a
decision variable; it remains the selected rotation candidate's ``z_offset``.

The module intentionally uses an axis-aligned footprint approximation for fast
masking.  The exact mesh, IK, grasp and robot collision checks remain in
``fixed_l2_validator.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core geometry: independent of WRS/SEALP, so it can be unit-tested alone.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float
    width: int
    height: int

    @classmethod
    def from_ranges(
        cls,
        x_range: Sequence[float],
        y_range: Sequence[float],
        resolution: float = 0.02,
    ) -> "GridSpec":
        if resolution <= 0.0 or not math.isfinite(float(resolution)):
            raise ValueError(f"resolution must be positive, got {resolution!r}")
        x_min, x_max = float(x_range[0]), float(x_range[1])
        y_min, y_max = float(y_range[0]), float(y_range[1])
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(
                f"invalid workspace ranges: x={x_range!r}, y={y_range!r}"
            )

        # ceil keeps the complete usable range even if the range is not an
        # exact multiple of the requested resolution.
        width = int(math.ceil((x_max - x_min) / resolution - 1e-12))
        height = int(math.ceil((y_max - y_min) / resolution - 1e-12))
        return cls(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            resolution=float(resolution),
            width=width,
            height=height,
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def n_cells(self) -> int:
        return self.height * self.width

    def cell_center(self, row: int, col: int) -> np.ndarray:
        self.validate_cell(row, col)
        x = self.x_min + (int(col) + 0.5) * self.resolution
        y = self.y_min + (int(row) + 0.5) * self.resolution
        return np.asarray([x, y], dtype=float)

    def world_to_cell(self, xy: Sequence[float], *, clip: bool = False) -> Tuple[int, int]:
        p = np.asarray(xy, dtype=float)
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            raise ValueError(f"xy must contain two finite values, got {xy!r}")
        col = int(math.floor((float(p[0]) - self.x_min) / self.resolution))
        row = int(math.floor((float(p[1]) - self.y_min) / self.resolution))
        if clip:
            col = int(np.clip(col, 0, self.width - 1))
            row = int(np.clip(row, 0, self.height - 1))
        else:
            self.validate_cell(row, col)
        return row, col

    def validate_cell(self, row: int, col: int) -> None:
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            raise ValueError(
                f"grid cell (row={row}, col={col}) is out of range; "
                f"valid row=0..{self.height - 1}, col=0..{self.width - 1}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FootprintRecord:
    part_id: str
    pose_id: int
    xy: np.ndarray
    footprint: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    z_offset: float
    kind: str = "decision"  # decision | preassembled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_id": self.part_id,
            "pose_id": int(self.pose_id),
            "xy": np.asarray(self.xy, dtype=float).tolist(),
            "footprint": np.asarray(self.footprint, dtype=float).tolist(),
            "aabb_min": np.asarray(self.aabb_min, dtype=float).tolist(),
            "aabb_max": np.asarray(self.aabb_max, dtype=float).tolist(),
            "z_offset": float(self.z_offset),
            "kind": str(self.kind),
        }


@dataclass(frozen=True)
class PlacementCheck:
    valid: bool
    reason: str = ""
    blocker: Optional[str] = None


@dataclass(frozen=True)
class StaticKeepout:
    name: str
    aabb_min: np.ndarray
    aabb_max: np.ndarray


class GridWorkspace:
    """2-D table grid and conservative footprint-based collision mask."""

    def __init__(
        self,
        spec: GridSpec,
        *,
        min_clearance: float = 0.01,
    ) -> None:
        self.spec = spec
        self.min_clearance = max(0.0, float(min_clearance))
        self.placements: Dict[str, FootprintRecord] = {}
        self.static_keepouts: List[StaticKeepout] = []

    # -------------------------- coordinate helpers -----------------------

    @staticmethod
    def footprint_aabb(
        xy: Sequence[float], footprint: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        center = np.asarray(xy, dtype=float)
        fp = np.asarray(footprint, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError(f"xy must contain two finite values, got {xy!r}")
        if fp.shape != (2,) or np.any(fp <= 0.0) or not np.all(np.isfinite(fp)):
            raise ValueError(f"footprint must contain two positive values, got {footprint!r}")
        half = fp / 2.0
        return center - half, center + half

    @staticmethod
    def rectangle_distance(
        amin: np.ndarray,
        amax: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> float:
        """Euclidean separation between two closed axis-aligned rectangles."""
        sep = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
        return float(np.linalg.norm(sep))

    def inside_table(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        eps = 1e-9
        return bool(
            aabb_min[0] >= self.spec.x_min - eps
            and aabb_max[0] <= self.spec.x_max + eps
            and aabb_min[1] >= self.spec.y_min - eps
            and aabb_max[1] <= self.spec.y_max + eps
        )

    def cells_intersecting_aabb(
        self, aabb_min: Sequence[float], aabb_max: Sequence[float]
    ) -> Tuple[slice, slice]:
        """Return rows/cols whose cells intersect the given world AABB."""
        amin = np.asarray(aabb_min, dtype=float)
        amax = np.asarray(aabb_max, dtype=float)
        if amin.shape != (2,) or amax.shape != (2,) or np.any(amax < amin):
            raise ValueError("invalid AABB")

        col0 = int(math.floor((float(amin[0]) - self.spec.x_min) / self.spec.resolution))
        col1 = int(math.floor((float(amax[0]) - self.spec.x_min) / self.spec.resolution))
        row0 = int(math.floor((float(amin[1]) - self.spec.y_min) / self.spec.resolution))
        row1 = int(math.floor((float(amax[1]) - self.spec.y_min) / self.spec.resolution))

        col0 = int(np.clip(col0, 0, self.spec.width - 1))
        col1 = int(np.clip(col1, 0, self.spec.width - 1))
        row0 = int(np.clip(row0, 0, self.spec.height - 1))
        row1 = int(np.clip(row1, 0, self.spec.height - 1))
        return slice(row0, row1 + 1), slice(col0, col1 + 1)

    # ---------------------------- state changes -------------------------

    def reset(self) -> None:
        self.placements.clear()
        self.static_keepouts.clear()

    def add_static_keepout(
        self,
        name: str,
        center_xy: Sequence[float],
        half_extents_xy: Sequence[float],
    ) -> None:
        center = np.asarray(center_xy, dtype=float)
        half = np.asarray(half_extents_xy, dtype=float)
        if center.shape != (2,) or half.shape != (2,) or np.any(half < 0.0):
            raise ValueError("invalid keepout center or half extents")
        self.static_keepouts.append(
            StaticKeepout(
                name=str(name),
                aabb_min=center - half,
                aabb_max=center + half,
            )
        )

    def check_candidate(
        self,
        *,
        part_id: str,
        xy: Sequence[float],
        footprint: Sequence[float],
        clearance: Optional[float] = None,
        ignore_static_keepout: bool = False,
    ) -> PlacementCheck:
        amin, amax = self.footprint_aabb(xy, footprint)
        if not self.inside_table(amin, amax):
            return PlacementCheck(False, "outside_table")

        required = self.min_clearance if clearance is None else max(0.0, float(clearance))
        for other_id, other in self.placements.items():
            if other_id == part_id:
                continue
            distance = self.rectangle_distance(amin, amax, other.aabb_min, other.aabb_max)
            if distance < required - 1e-12:
                reason = "footprint_overlap" if distance <= 1e-12 else "clearance_too_small"
                return PlacementCheck(False, reason, blocker=other_id)

        if not ignore_static_keepout:
            for keepout in self.static_keepouts:
                distance = self.rectangle_distance(amin, amax, keepout.aabb_min, keepout.aabb_max)
                if distance <= 1e-12:
                    return PlacementCheck(False, "static_keepout", blocker=keepout.name)

        return PlacementCheck(True)

    def place(
        self,
        *,
        part_id: str,
        pose_id: int,
        xy: Sequence[float],
        footprint: Sequence[float],
        z_offset: float,
        kind: str = "decision",
        clearance: Optional[float] = None,
        ignore_static_keepout: bool = False,
        allow_replace: bool = False,
    ) -> FootprintRecord:
        if part_id in self.placements and not allow_replace:
            raise ValueError(f"part {part_id!r} has already been placed")
        check = self.check_candidate(
            part_id=part_id,
            xy=xy,
            footprint=footprint,
            clearance=clearance,
            ignore_static_keepout=ignore_static_keepout,
        )
        if not check.valid:
            extra = f", blocker={check.blocker}" if check.blocker else ""
            raise ValueError(f"cannot place {part_id}: {check.reason}{extra}")

        amin, amax = self.footprint_aabb(xy, footprint)
        record = FootprintRecord(
            part_id=str(part_id),
            pose_id=int(pose_id),
            xy=np.asarray(xy, dtype=float),
            footprint=np.asarray(footprint, dtype=float),
            aabb_min=amin,
            aabb_max=amax,
            z_offset=float(z_offset),
            kind=str(kind),
        )
        self.placements[str(part_id)] = record
        return record

    def remove(self, part_id: str) -> None:
        self.placements.pop(str(part_id), None)

    # ------------------------- observations / masks ---------------------

    def occupancy_channels(self) -> np.ndarray:
        """Return four numeric channels shaped ``[4, H, W]``.

        Channel 0: all placed footprints.
        Channel 1: preassembled footprints only.
        Channel 2: static robot-base keepout rectangles.
        Channel 3: clearance halo around placed footprints.
        """
        h, w = self.spec.shape
        channels = np.zeros((4, h, w), dtype=np.float32)

        for record in self.placements.values():
            rows, cols = self.cells_intersecting_aabb(record.aabb_min, record.aabb_max)
            channels[0, rows, cols] = 1.0
            if record.kind == "preassembled":
                channels[1, rows, cols] = 1.0

            halo_min = record.aabb_min - self.min_clearance
            halo_max = record.aabb_max + self.min_clearance
            rows_h, cols_h = self.cells_intersecting_aabb(halo_min, halo_max)
            channels[3, rows_h, cols_h] = 1.0

        for keepout in self.static_keepouts:
            rows, cols = self.cells_intersecting_aabb(keepout.aabb_min, keepout.aabb_max)
            channels[2, rows, cols] = 1.0

        return channels

    def valid_center_mask(
        self,
        *,
        part_id: str,
        footprint: Sequence[float],
        extra_check: Optional[Callable[[np.ndarray], PlacementCheck]] = None,
        order_x_max: Optional[float] = None,
    ) -> np.ndarray:
        """Return a Boolean ``[H, W]`` mask for one fixed pose footprint."""
        mask = np.zeros(self.spec.shape, dtype=bool)
        for row in range(self.spec.height):
            for col in range(self.spec.width):
                xy = self.spec.cell_center(row, col)
                if order_x_max is not None and float(xy[0]) > float(order_x_max) + 1e-12:
                    continue
                base = self.check_candidate(
                    part_id=part_id,
                    xy=xy,
                    footprint=footprint,
                    # The project-specific arm keepout check below already
                    # accounts for the candidate footprint.  The static map is
                    # an observation channel, not the final authority.
                    ignore_static_keepout=True,
                )
                if not base.valid:
                    continue
                if extra_check is not None:
                    check = extra_check(xy)
                    if not check.valid:
                        continue
                mask[row, col] = True
        return mask

    @staticmethod
    def flatten_action(pose_id: int, row: int, col: int, *, height: int, width: int) -> int:
        if pose_id < 0 or row < 0 or row >= height or col < 0 or col >= width:
            raise ValueError("invalid action components")
        return int(pose_id) * (height * width) + int(row) * width + int(col)

    @staticmethod
    def unflatten_action(action: int, *, height: int, width: int) -> Tuple[int, int, int]:
        if action < 0:
            raise ValueError("action must be non-negative")
        cells = height * width
        pose_id = int(action) // cells
        grid_id = int(action) % cells
        row = grid_id // width
        col = grid_id % width
        return pose_id, row, col


# ---------------------------------------------------------------------------
# Project adapter: uses FixedL2Validator/searcher but still does not run L2.
# ---------------------------------------------------------------------------


class TowerGridState:
    """Sequential Tower placement state used later by ``tower_layout_env.py``."""

    def __init__(
        self,
        validator: Any,
        *,
        region_id: str,
        resolution: float = 0.02,
    ) -> None:
        self.validator = validator
        self.searcher = validator.searcher
        self.region_id = str(region_id)
        self.region_name, self.region_rc, self.region_center = validator._resolve_region(self.region_id)

        self.searcher._set_assembly_station(
            np.asarray(self.region_center, dtype=float),
            region_id=self.region_name,
            rc=tuple(self.region_rc),
        )

        spec = GridSpec.from_ranges(
            self.searcher.table_x_range,
            self.searcher.table_y_range,
            resolution=resolution,
        )
        self.workspace = GridWorkspace(
            spec,
            min_clearance=float(self.searcher.min_staging_mesh_clearance),
        )
        self.decision_parts = validator._decision_parts()
        self.current_step = 0
        self.selected_pose_ids: Dict[str, int] = {}
        self.selected_xy: Dict[str, np.ndarray] = {}

        self._add_robot_keepout_observation()
        self._add_preassembled_first_part()

    @property
    def done(self) -> bool:
        return self.current_step >= len(self.decision_parts)

    @property
    def current_part(self) -> Optional[str]:
        return None if self.done else self.decision_parts[self.current_step]

    @property
    def max_pose_count(self) -> int:
        return max(len(self.searcher.rot_cands[p]) for p in self.decision_parts)

    @property
    def action_space_size(self) -> int:
        return self.max_pose_count * self.workspace.spec.n_cells

    def _add_robot_keepout_observation(self) -> None:
        # This channel displays the base-centered rectangles.  Exact candidate
        # validity still calls _staging_arm_keepout_reason, which expands the
        # rectangle by half of the selected footprint.
        half = np.asarray(
            [
                float(self.searcher.staging_arm_x_clearance),
                float(self.searcher.staging_arm_y_clearance),
            ],
            dtype=float,
        )
        for name, center in self.searcher._arm_base_xy_map().items():
            self.workspace.add_static_keepout(name, center, half)

    def _add_preassembled_first_part(self) -> None:
        first_pid = self.searcher._first_part_id()
        if not self.searcher.preassemble_first_part or first_pid is None:
            return
        if first_pid not in self.searcher.world_poses:
            raise ValueError(f"preassembled part {first_pid!r} has no world pose")

        goal_pos, goal_rot = self.searcher.world_poses[first_pid]
        verts = np.asarray(self.searcher.mesh_vertices[first_pid], dtype=float)
        world_local = verts.dot(np.asarray(goal_rot, dtype=float).T)
        extent = world_local.max(axis=0) - world_local.min(axis=0)
        footprint = extent[:2]

        # The assembly-region filter has already verified base_plate vs robot.
        self.workspace.place(
            part_id=first_pid,
            pose_id=-1,
            xy=np.asarray(goal_pos, dtype=float)[:2],
            footprint=footprint,
            z_offset=float(np.asarray(goal_pos, dtype=float)[2]),
            kind="preassembled",
            clearance=0.0,
            ignore_static_keepout=True,
        )

    def _candidate(self, part_id: str, pose_id: int) -> Any:
        candidates = self.searcher.rot_cands.get(part_id, [])
        if pose_id < 0 or pose_id >= len(candidates):
            raise ValueError(
                f"pose_id={pose_id} is invalid for {part_id}; valid 0..{len(candidates) - 1}"
            )
        return candidates[pose_id]

    def _order_x_max_for_current(self, part_id: str) -> Optional[float]:
        if not bool(self.searcher.enforce_order_x_constraint):
            return None
        idx = self.decision_parts.index(part_id)
        if idx <= 0:
            return None
        prev_pid = self.decision_parts[idx - 1]
        prev_xy = self.selected_xy.get(prev_pid)
        if prev_xy is None:
            return None
        return float(prev_xy[0]) + float(self.searcher.order_x_tolerance)

    def _project_specific_check(self, part_id: str, cand: Any, xy: np.ndarray) -> PlacementCheck:
        # Exact footprint-aware arm-base keepout from the current strict code.
        hit = self.searcher._staging_arm_keepout_reason(part_id, xy, cand)
        if hit:
            return PlacementCheck(False, "robot_arm_keepout", blocker=str(hit))
        return PlacementCheck(True)

    def pose_position_mask(self, part_id: Optional[str] = None) -> np.ndarray:
        """Return Boolean mask shaped ``[max_pose_count, H, W]``."""
        pid = str(part_id or self.current_part)
        if self.done and part_id is None:
            return np.zeros(
                (self.max_pose_count, *self.workspace.spec.shape), dtype=bool
            )
        if pid not in self.decision_parts:
            raise ValueError(f"unknown decision part {pid!r}")

        out = np.zeros((self.max_pose_count, *self.workspace.spec.shape), dtype=bool)
        order_x_max = self._order_x_max_for_current(pid)
        for pose_id, cand in enumerate(self.searcher.rot_cands[pid]):
            out[pose_id] = self.workspace.valid_center_mask(
                part_id=pid,
                footprint=np.asarray(cand.footprint, dtype=float),
                order_x_max=order_x_max,
                extra_check=lambda xy, _pid=pid, _cand=cand: self._project_specific_check(
                    _pid, _cand, xy
                ),
            )
        return out

    def flat_action_mask(self) -> np.ndarray:
        return self.pose_position_mask().reshape(-1)

    def check_action(self, action: int) -> PlacementCheck:
        if self.done:
            return PlacementCheck(False, "episode_already_complete")
        pose_id, row, col = GridWorkspace.unflatten_action(
            int(action),
            height=self.workspace.spec.height,
            width=self.workspace.spec.width,
        )
        if pose_id >= self.max_pose_count:
            return PlacementCheck(False, "pose_padding")
        pid = str(self.current_part)
        candidates = self.searcher.rot_cands[pid]
        if pose_id >= len(candidates):
            return PlacementCheck(False, "pose_padding")
        if not self.pose_position_mask(pid)[pose_id, row, col]:
            return PlacementCheck(False, "masked_action")
        return PlacementCheck(True)

    def apply_action(self, action: int) -> FootprintRecord:
        check = self.check_action(action)
        if not check.valid:
            raise ValueError(f"invalid action {action}: {check.reason}")

        pose_id, row, col = GridWorkspace.unflatten_action(
            int(action),
            height=self.workspace.spec.height,
            width=self.workspace.spec.width,
        )
        pid = str(self.current_part)
        cand = self._candidate(pid, pose_id)
        xy = self.workspace.spec.cell_center(row, col)

        record = self.workspace.place(
            part_id=pid,
            pose_id=pose_id,
            xy=xy,
            footprint=np.asarray(cand.footprint, dtype=float),
            z_offset=float(cand.z_offset),
            kind="decision",
            # Project arm keepout was checked separately and more accurately.
            ignore_static_keepout=True,
        )
        self.selected_pose_ids[pid] = int(pose_id)
        self.selected_xy[pid] = xy.copy()
        self.current_step += 1
        return record

    def observation(self) -> Dict[str, Any]:
        """Numeric state; no camera or image sensor is used."""
        return {
            "current_step": int(self.current_step),
            "current_part": self.current_part,
            "assembly_center": np.asarray(self.region_center, dtype=float)[:2].astype(np.float32),
            "occupancy": self.workspace.occupancy_channels(),
            "action_mask": self.flat_action_mask() if not self.done else np.zeros(self.action_space_size, dtype=bool),
        }

    def export_fixed_l2_request(self) -> Dict[str, Any]:
        if not self.done:
            raise ValueError(
                f"layout is incomplete: placed {self.current_step}/{len(self.decision_parts)} parts"
            )
        return {
            "region_id": self.region_id,
            "parts": {
                pid: {
                    "xy": np.asarray(self.selected_xy[pid], dtype=float).tolist(),
                    "pose_id": int(self.selected_pose_ids[pid]),
                }
                for pid in self.decision_parts
            },
        }

    def summary(self) -> Dict[str, Any]:
        mask = self.pose_position_mask() if not self.done else None
        return {
            "region_id": self.region_id,
            "region_rc": [int(self.region_rc[0]), int(self.region_rc[1])],
            "assembly_center": np.asarray(self.region_center, dtype=float).tolist(),
            "grid": self.workspace.spec.to_dict(),
            "decision_parts": list(self.decision_parts),
            "current_step": int(self.current_step),
            "current_part": self.current_part,
            "max_pose_count": int(self.max_pose_count),
            "action_space_size": int(self.action_space_size),
            "placed": {k: v.to_dict() for k, v in self.workspace.placements.items()},
            "valid_actions_current": int(mask.sum()) if mask is not None else 0,
            "valid_actions_per_pose": (
                [int(mask[i].sum()) for i in range(mask.shape[0])] if mask is not None else []
            ),
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 2 cm Tower RL grid and inspect legal actions."
    )
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--region-id", default="r1_c1")
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--max-rot-candidates", type=int, default=12)
    parser.add_argument(
        "--random-complete",
        action="store_true",
        help="Choose one random legal action per part and export a complete request.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--request-json",
        help="When --random-complete is used, save a pure fixed-L2 request JSON.",
    )
    parser.add_argument("--output-npz", help="Optional occupancy/action-mask NPZ output.")
    return parser.parse_args()


def main() -> None:
    # Deferred import keeps the core grid class testable without WRS.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    from .fixed_l2_validator import FixedL2Validator, build_default_searcher

    args = _parse_args()
    searcher = build_default_searcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_rot_candidates=args.max_rot_candidates,
    )
    validator = FixedL2Validator(searcher)
    state = TowerGridState(
        validator,
        region_id=args.region_id,
        resolution=args.resolution,
    )

    rng = np.random.default_rng(args.seed)
    if args.random_complete:
        while not state.done:
            mask = state.flat_action_mask()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                raise RuntimeError(
                    f"no legal action for step={state.current_step}, part={state.current_part}"
                )
            action = int(rng.choice(valid))
            record = state.apply_action(action)
            print(
                f"[place] step={state.current_step:02d} part={record.part_id:14s} "
                f"pose={record.pose_id:2d} xy={np.round(record.xy, 4).tolist()}"
            )
        payload = {
            "summary": state.summary(),
            "fixed_l2_request": state.export_fixed_l2_request(),
        }
    else:
        payload = state.summary()

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON saved to: {path}")

    if args.request_json:
        if not args.random_complete:
            raise ValueError("--request-json requires --random-complete")
        request_path = Path(args.request_json)
        request_path.parent.mkdir(parents=True, exist_ok=True)
        with request_path.open("w", encoding="utf-8") as f:
            json.dump(state.export_fixed_l2_request(), f, ensure_ascii=False, indent=2)
        print(f"[OK] fixed-L2 request saved to: {request_path}")

    if args.output_npz:
        obs = state.observation()
        path = Path(args.output_npz)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            occupancy=np.asarray(obs["occupancy"], dtype=np.float32),
            action_mask=np.asarray(obs["action_mask"], dtype=np.uint8),
            assembly_center=np.asarray(obs["assembly_center"], dtype=np.float32),
            current_step=np.asarray([obs["current_step"]], dtype=np.int32),
        )
        print(f"[OK] NPZ saved to: {path}")


if __name__ == "__main__":
    main()
