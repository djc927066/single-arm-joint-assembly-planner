#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic assembly-conditioned layout environment.

This module generalises ``TowerLayoutEnv`` to arbitrary SEALP ASMDEF tasks.
It intentionally does not contain a policy network yet.  It provides the
stable environment/observation interface required by the later
PointNet + GAT + spatial policy + MaskablePPO implementation.

Core rules
----------
1. Parts are read strictly in ASMDEF ``assembly.step`` order.
2. The first ASMDEF part is always preassembled at the selected assembly
   region and is never an Action target or a grasp/IK planning target.
3. All remaining parts are placed sequentially with one discrete action:
   ``(pose_id, grid_row, grid_col)``.
4. The number of real parts and poses may vary.  Fixed-size tensors use
   ``part_mask`` and ``pose_mask`` padding.
5. Legal assembly regions are computed dynamically by the existing searcher:
   the current task's preassembled first part is tested against robot collision
   boxes at every 3x3 center.  No region ID is hard-coded as forbidden.
6. The first ASMDEF step can optionally be normalised to ``rel_pos=[0,0,0]``
   before the searcher is built.  This makes "place the first part directly at
   the assembly region" explicit and consistent across tasks.

No camera is used.  Every observation entry is a numeric tensor computed from
ASMDEF, STL geometry, current placements and candidate poses.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import gymnasium as gym
    from gymnasium import spaces

    _EnvBase = gym.Env
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _EnvBase = object

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout.assembly_rl.fixed_l2_validator import (  # noqa: E402
    FixedL2Result,
    FixedL2Validator,
    PREFERRED_REGION_IDS,
    build_default_searcher,
)
from sealp.examples.layout.assembly_rl.grid_workspace import (  # noqa: E402
    FootprintRecord,
    GridSpec,
    GridWorkspace,
    PlacementCheck,
)


REGION_MODE_FIXED = "fixed"
REGION_MODE_PREFERRED = "preferred"
REGION_MODE_ALL = "all"
REGION_MODES = (REGION_MODE_FIXED, REGION_MODE_PREFERRED, REGION_MODE_ALL)


# ---------------------------------------------------------------------------
# Generic ASMDEF metadata
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _rotation_6d(rotmat: Sequence[Sequence[float]]) -> np.ndarray:
    rot = np.asarray(rotmat, dtype=float)
    if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
        return np.zeros(6, dtype=np.float32)
    return np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class AssemblyStepSpec:
    step: int
    part: str
    parent: str
    rel_pos: np.ndarray
    rel_rotmat: np.ndarray
    deps: Tuple[int, ...]


@dataclass
class AssemblyTaskSpec:
    asmdef_path: str
    name: str
    part_order: List[str]
    steps: List[AssemblyStepSpec]
    masses: Dict[str, float]
    model_ids: Dict[str, str]
    model_paths: Dict[str, str]
    symmetry_group_for_part: Dict[str, str]
    symmetry_group_sizes: Dict[str, int]

    @property
    def first_part(self) -> str:
        if not self.part_order:
            raise ValueError("ASMDEF contains no assembly steps")
        return self.part_order[0]

    @property
    def decision_parts(self) -> List[str]:
        return self.part_order[1:]

    @property
    def part_to_index(self) -> Dict[str, int]:
        return {part: index for index, part in enumerate(self.part_order)}

    @classmethod
    def load(cls, asmdef_path: str) -> "AssemblyTaskSpec":
        path = Path(asmdef_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, Mapping):
            raise ValueError(f"invalid ASMDEF root in {path}")

        raw_models = dict(data.get("models") or {})
        raw_parts = dict(data.get("parts") or {})
        raw_steps = list(data.get("assembly") or [])
        if not raw_steps:
            raise ValueError(f"ASMDEF {path} has no assembly steps")

        steps: List[AssemblyStepSpec] = []
        for raw in raw_steps:
            step = int(raw["step"])
            part = str(raw["part"])
            parent = str(raw.get("parent", "fixture"))
            rel_pos = np.asarray(raw.get("rel_pos", [0.0, 0.0, 0.0]), dtype=float)
            rel_rot = np.asarray(raw.get("rel_rotmat", np.eye(3)), dtype=float)
            deps = tuple(int(v) for v in (raw.get("deps") or []))
            if rel_pos.shape != (3,):
                raise ValueError(f"assembly step {step} rel_pos must be length 3")
            if rel_rot.shape != (3, 3):
                raise ValueError(f"assembly step {step} rel_rotmat must be 3x3")
            steps.append(
                AssemblyStepSpec(
                    step=step,
                    part=part,
                    parent=parent,
                    rel_pos=rel_pos,
                    rel_rotmat=rel_rot,
                    deps=deps,
                )
            )
        steps.sort(key=lambda row: row.step)
        expected = list(range(len(steps)))
        actual = [row.step for row in steps]
        if actual != expected:
            raise ValueError(
                f"ASMDEF steps must be continuous from 0; expected={expected}, actual={actual}"
            )

        part_order = [row.part for row in steps]
        if len(set(part_order)) != len(part_order):
            raise ValueError("each part may appear only once in assembly steps")

        model_ids: Dict[str, str] = {}
        model_paths: Dict[str, str] = {}
        masses: Dict[str, float] = {}
        for part in part_order:
            if part not in raw_parts:
                raise ValueError(f"assembly part {part!r} is missing from parts section")
            part_spec = dict(raw_parts[part] or {})
            model_id = str(part_spec.get("model", ""))
            if not model_id or model_id not in raw_models:
                raise ValueError(f"part {part!r} has invalid model id {model_id!r}")
            model_spec = dict(raw_models[model_id] or {})
            model_path = str(model_spec.get("path", ""))
            if not model_path:
                raise ValueError(f"model {model_id!r} has no path")
            model_ids[part] = model_id
            model_paths[part] = model_path
            masses[part] = max(0.0, _safe_float(part_spec.get("mass"), 0.0))

        group_for_part: Dict[str, str] = {}
        group_sizes: Dict[str, int] = {}
        for group_name, members in dict(data.get("symmetry_groups") or {}).items():
            valid_members = [str(p) for p in (members or []) if str(p) in part_order]
            size = len(valid_members)
            for part in valid_members:
                group_for_part[part] = str(group_name)
                group_sizes[part] = size

        return cls(
            asmdef_path=str(path),
            name=str(data.get("name", path.stem)),
            part_order=part_order,
            steps=steps,
            masses=masses,
            model_ids=model_ids,
            model_paths=model_paths,
            symmetry_group_for_part=group_for_part,
            symmetry_group_sizes=group_sizes,
        )


@dataclass(frozen=True)
class PreparedAsmdef:
    original_path: str
    effective_path: str
    first_part: str
    original_first_rel_pos: Tuple[float, float, float]
    forced_first_at_region_center: bool
    temporary: bool


def prepare_asmdef_for_preassembled_first_part(
    asmdef_path: str,
    *,
    force_first_at_region_center: bool = True,
) -> PreparedAsmdef:
    """Create an effective ASMDEF with step-0 translation set to zero.

    Child target poses remain consistent because their transforms are composed
    from the same first-part frame.  The first-part rotation is preserved.
    Relative model paths are converted to absolute paths before writing the
    temporary file, so moving the effective ASMDEF does not break them.
    """
    original = Path(asmdef_path).expanduser().resolve()
    with original.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw_steps = list((data or {}).get("assembly") or [])
    if not raw_steps:
        raise ValueError(f"ASMDEF {original} has no assembly steps")
    raw_steps.sort(key=lambda row: int(row["step"]))
    first = raw_steps[0]
    first_part = str(first["part"])
    original_rel = np.asarray(first.get("rel_pos", [0.0, 0.0, 0.0]), dtype=float)
    if original_rel.shape != (3,):
        raise ValueError("first assembly step rel_pos must be length 3")

    if not force_first_at_region_center:
        return PreparedAsmdef(
            original_path=str(original),
            effective_path=str(original),
            first_part=first_part,
            original_first_rel_pos=tuple(float(v) for v in original_rel),
            forced_first_at_region_center=False,
            temporary=False,
        )

    # Convert relative model paths before moving the YAML into a temp folder.
    for model_spec in dict((data or {}).get("models") or {}).values():
        if not isinstance(model_spec, Mapping) or "path" not in model_spec:
            continue
        model_path = Path(str(model_spec["path"]))
        if not model_path.is_absolute():
            model_spec["path"] = str((original.parent / model_path).resolve())

    first["rel_pos"] = [0.0, 0.0, 0.0]
    # Replace the original order-preserving list in data.
    data["assembly"] = raw_steps

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{original.stem}_preassembled_",
        suffix=".asmdef",
        text=True,
    )
    os.close(fd)
    with open(temp_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    return PreparedAsmdef(
        original_path=str(original),
        effective_path=str(Path(temp_name).resolve()),
        first_part=first_part,
        original_first_rel_pos=tuple(float(v) for v in original_rel),
        forced_first_at_region_center=True,
        temporary=True,
    )


def build_assembly_validator(
    *,
    asmdef_path: str,
    config_yaml: str,
    grasp_dir: str,
    cdprim_type: Optional[str] = "box",
    planner_obstacle_mode: str = "staging_aware",
    max_poses: int = 16,
    force_first_at_region_center: bool = True,
) -> Tuple[FixedL2Validator, AssemblyTaskSpec, PreparedAsmdef]:
    """Build the generic validator and remove Tower-only evaluator biases."""
    prepared = prepare_asmdef_for_preassembled_first_part(
        asmdef_path,
        force_first_at_region_center=force_first_at_region_center,
    )
    task = AssemblyTaskSpec.load(prepared.effective_path)
    searcher = build_default_searcher(
        asmdef_path=prepared.effective_path,
        config_yaml=config_yaml,
        grasp_dir=grasp_dir,
        cdprim_type=cdprim_type,
        planner_obstacle_mode=planner_obstacle_mode,
        max_rot_candidates=max_poses,
    )

    # Generic environment: remove staging preferences that were introduced for
    # the original Tower search.  Feasibility remains governed by geometry and
    # fixed L2.  Grasp/IK preferences will later enter through explicit hints.
    searcher.preassemble_first_part = True
    if hasattr(searcher, "enforce_order_x_constraint"):
        searcher.enforce_order_x_constraint = False
    if hasattr(searcher, "enable_y_side_distribution_score"):
        searcher.enable_y_side_distribution_score = False
    if hasattr(searcher, "prefer_upright_when_topdown_low"):
        searcher.prefer_upright_when_topdown_low = False

    # No hard-coded forbidden IDs.  ``_assembly_region_candidates`` dynamically
    # tests the current task's first preassembled object against robot boxes.
    validator = FixedL2Validator(
        searcher,
        forbidden_region_ids=(),
        preferred_region_ids=PREFERRED_REGION_IDS,
    )

    if validator._decision_parts() != task.decision_parts:
        raise RuntimeError(
            "searcher/ASMDEF order mismatch: "
            f"searcher={validator._decision_parts()}, asmdef={task.decision_parts}"
        )
    return validator, task, prepared


# ---------------------------------------------------------------------------
# Generic sequential grid state
# ---------------------------------------------------------------------------


class AssemblyGridState:
    """Grid state for an arbitrary ASMDEF task.

    The first part is inserted into ``workspace.placements`` as
    ``kind='preassembled'`` and never appears in ``decision_parts``.
    """

    def __init__(
        self,
        validator: FixedL2Validator,
        task: AssemblyTaskSpec,
        *,
        region_id: str,
        resolution: float = 0.02,
        max_poses: int = 16,
    ) -> None:
        self.validator = validator
        self.searcher = validator.searcher
        self.task = task
        self.region_id = str(region_id)
        self.max_poses = int(max_poses)
        if self.max_poses <= 0:
            raise ValueError("max_poses must be positive")

        self.region_name, self.region_rc, self.region_center = validator._resolve_region(
            self.region_id
        )
        self.searcher._set_assembly_station(
            np.asarray(self.region_center, dtype=float),
            region_id=self.region_name,
            rc=tuple(self.region_rc),
        )

        self.workspace = GridWorkspace(
            GridSpec.from_ranges(
                self.searcher.table_x_range,
                self.searcher.table_y_range,
                resolution=resolution,
            ),
            min_clearance=float(self.searcher.min_staging_mesh_clearance),
        )
        self.part_order = list(task.part_order)
        self.decision_parts = list(task.decision_parts)
        self.current_step = 0
        self.selected_pose_ids: Dict[str, int] = {}
        self.selected_xy: Dict[str, np.ndarray] = {}

        for part in self.decision_parts:
            count = len(self.searcher.rot_cands.get(part, []))
            if count <= 0:
                raise ValueError(f"decision part {part!r} has no pose candidates")
            if count > self.max_poses:
                raise ValueError(
                    f"part {part!r} has {count} poses > max_poses={self.max_poses}; "
                    "increase --max-poses or reduce searcher candidates"
                )

        self._add_robot_keepout_observation()
        self._add_preassembled_first_part()

    @property
    def done(self) -> bool:
        return self.current_step >= len(self.decision_parts)

    @property
    def current_part(self) -> Optional[str]:
        return None if self.done else self.decision_parts[self.current_step]

    @property
    def action_space_size(self) -> int:
        return self.max_poses * self.workspace.spec.n_cells

    def pose_count(self, part_id: str) -> int:
        return len(self.searcher.rot_cands.get(part_id, []))

    def pose_mask(self, part_id: Optional[str] = None) -> np.ndarray:
        out = np.zeros(self.max_poses, dtype=bool)
        pid = part_id if part_id is not None else self.current_part
        if pid is not None:
            out[: min(self.max_poses, self.pose_count(str(pid)))] = True
        return out

    def _add_robot_keepout_observation(self) -> None:
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
        first = self.task.first_part
        if first not in self.searcher.world_poses:
            raise ValueError(f"preassembled first part {first!r} has no world pose")
        goal_pos, goal_rot = self.searcher.world_poses[first]
        verts = np.asarray(self.searcher.mesh_vertices[first], dtype=float)
        rotated = verts.dot(np.asarray(goal_rot, dtype=float).T)
        extent = rotated.max(axis=0) - rotated.min(axis=0)
        self.workspace.place(
            part_id=first,
            pose_id=-1,
            xy=np.asarray(goal_pos, dtype=float)[:2],
            footprint=extent[:2],
            z_offset=float(np.asarray(goal_pos, dtype=float)[2]),
            kind="preassembled",
            clearance=0.0,
            # The dynamic assembly-region filter already checked this object.
            ignore_static_keepout=True,
        )

    def _candidate(self, part_id: str, pose_id: int) -> Any:
        candidates = self.searcher.rot_cands.get(part_id, [])
        if pose_id < 0 or pose_id >= len(candidates):
            raise ValueError(
                f"pose_id={pose_id} invalid for {part_id}; valid=0..{len(candidates)-1}"
            )
        return candidates[pose_id]

    def _robot_keepout_check(self, part_id: str, cand: Any, xy: np.ndarray) -> PlacementCheck:
        hit = self.searcher._staging_arm_keepout_reason(part_id, xy, cand)
        if hit:
            return PlacementCheck(False, "robot_arm_keepout", blocker=str(hit))
        return PlacementCheck(True)

    def pose_position_mask(self, part_id: Optional[str] = None) -> np.ndarray:
        pid = str(part_id or self.current_part) if (part_id or self.current_part) else None
        out = np.zeros(
            (self.max_poses, *self.workspace.spec.shape),
            dtype=bool,
        )
        if pid is None:
            return out
        if pid not in self.decision_parts:
            raise ValueError(f"unknown decision part {pid!r}")

        for pose_id, cand in enumerate(self.searcher.rot_cands[pid]):
            if pose_id >= self.max_poses:
                break
            out[pose_id] = self.workspace.valid_center_mask(
                part_id=pid,
                footprint=np.asarray(cand.footprint, dtype=float),
                order_x_max=None,  # Tower-only order-X constraint is removed.
                extra_check=lambda xy, _pid=pid, _cand=cand: self._robot_keepout_check(
                    _pid, _cand, xy
                ),
            )
        return out

    def flat_action_mask(self) -> np.ndarray:
        return self.pose_position_mask().reshape(-1)

    def check_action(self, action: int) -> PlacementCheck:
        if self.done:
            return PlacementCheck(False, "episode_already_complete")
        if action < 0 or action >= self.action_space_size:
            return PlacementCheck(False, "action_out_of_range")
        pose_id, row, col = GridWorkspace.unflatten_action(
            int(action),
            height=self.workspace.spec.height,
            width=self.workspace.spec.width,
        )
        pid = str(self.current_part)
        if pose_id >= self.pose_count(pid):
            return PlacementCheck(False, "pose_padding")
        if not self.pose_position_mask(pid)[pose_id, row, col]:
            return PlacementCheck(False, "masked_action")
        return PlacementCheck(True)

    def apply_action(self, action: int) -> FootprintRecord:
        check = self.check_action(int(action))
        if not check.valid:
            raise ValueError(f"invalid action {action}: {check.reason}")
        pose_id, row, col = GridWorkspace.unflatten_action(
            int(action),
            height=self.workspace.spec.height,
            width=self.workspace.spec.width,
        )
        part = str(self.current_part)
        cand = self._candidate(part, pose_id)
        xy = self.workspace.spec.cell_center(row, col)
        record = self.workspace.place(
            part_id=part,
            pose_id=pose_id,
            xy=xy,
            footprint=np.asarray(cand.footprint, dtype=float),
            z_offset=float(cand.z_offset),
            kind="decision",
            ignore_static_keepout=True,
        )
        self.selected_pose_ids[part] = int(pose_id)
        self.selected_xy[part] = np.asarray(xy, dtype=float).copy()
        self.current_step += 1
        return record

    def export_fixed_l2_request(self) -> Dict[str, Any]:
        if not self.done:
            raise ValueError(
                f"layout incomplete: {self.current_step}/{len(self.decision_parts)} decisions"
            )
        return {
            "region_id": self.region_id,
            # The first preassembled part is deliberately absent.
            "parts": {
                part: {
                    "xy": np.asarray(self.selected_xy[part], dtype=float).tolist(),
                    "pose_id": int(self.selected_pose_ids[part]),
                }
                for part in self.decision_parts
            },
        }


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------


class AssemblyLayoutEnv(_EnvBase):
    """Generic sequential layout environment prepared for PointNet + GAT."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    PART_FEATURE_DIM = 26
    POSE_FEATURE_DIM = 17

    def __init__(
        self,
        validator: FixedL2Validator,
        task: AssemblyTaskSpec,
        *,
        resolution: float = 0.02,
        max_parts: int = 12,
        max_poses: int = 16,
        region_mode: str = REGION_MODE_PREFERRED,
        fixed_region_id: Optional[str] = None,
        preferred_mass: float = 0.75,
        run_l2: bool = True,
        seed: Optional[int] = None,
        prepared_asmdef: Optional[PreparedAsmdef] = None,
    ) -> None:
        if region_mode not in REGION_MODES:
            raise ValueError(f"region_mode must be one of {REGION_MODES}")
        if max_parts <= 1:
            raise ValueError("max_parts must be at least 2")
        if max_poses <= 0:
            raise ValueError("max_poses must be positive")
        if len(task.part_order) > max_parts:
            raise ValueError(
                f"ASMDEF has {len(task.part_order)} parts > max_parts={max_parts}"
            )
        if not 0.0 <= preferred_mass <= 1.0:
            raise ValueError("preferred_mass must be in [0,1]")

        self.validator = validator
        self.searcher = validator.searcher
        self.task = task
        self.prepared_asmdef = prepared_asmdef
        self.resolution = float(resolution)
        self.max_parts = int(max_parts)
        self.max_poses = int(max_poses)
        self.region_mode = str(region_mode)
        self.preferred_mass = float(preferred_mass)
        self.run_l2 = bool(run_l2)
        self._rng = np.random.default_rng(seed)

        self.region_rows = validator.available_regions()
        self.region_ids = [str(row["region_id"]) for row in self.region_rows]
        if not self.region_ids:
            raise RuntimeError(
                "no legal assembly region for this task's first preassembled part"
            )
        self.preferred_region_ids = [
            rid for rid in PREFERRED_REGION_IDS if rid in self.region_ids
        ]
        self.other_region_ids = [
            rid for rid in self.region_ids if rid not in self.preferred_region_ids
        ]
        self.fixed_region_id = str(fixed_region_id or self._default_region_id())
        if self.fixed_region_id not in self.region_ids:
            raise ValueError(
                f"fixed region {self.fixed_region_id!r} invalid; valid={self.region_ids}"
            )

        template = AssemblyGridState(
            validator,
            task,
            region_id=self.fixed_region_id,
            resolution=self.resolution,
            max_poses=self.max_poses,
        )
        self.grid_height = template.workspace.spec.height
        self.grid_width = template.workspace.spec.width
        self.action_n = template.action_space_size
        self.n_parts = len(task.part_order)
        self.n_decision_parts = len(task.decision_parts)
        if self.n_decision_parts <= 0:
            raise ValueError("ASMDEF needs at least one non-preassembled decision part")

        self.state: Optional[AssemblyGridState] = None
        self.current_region_id: Optional[str] = None
        self._terminated = False
        self._truncated = False
        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_l2_result: Optional[FixedL2Result] = None

        self._parent_adjacency = self._build_parent_adjacency()
        self._dependency_adjacency = self._build_dependency_adjacency()
        self._symmetry_adjacency = self._build_symmetry_adjacency()

        if spaces is not None:
            self.action_space = spaces.Discrete(self.action_n)
            self.observation_space = spaces.Dict(
                {
                    "occupancy": spaces.Box(
                        0.0,
                        1.0,
                        shape=(4, self.grid_height, self.grid_width),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.MultiBinary(self.action_n),
                    "assembly_center": spaces.Box(
                        0.0, 1.0, shape=(2,), dtype=np.float32
                    ),
                    "step_normalized": spaces.Box(
                        0.0, 1.0, shape=(1,), dtype=np.float32
                    ),
                    "part_features": spaces.Box(
                        -1.0,
                        1.0,
                        shape=(self.max_parts, self.PART_FEATURE_DIM),
                        dtype=np.float32,
                    ),
                    "part_mask": spaces.MultiBinary(self.max_parts),
                    "decision_mask": spaces.MultiBinary(self.max_parts),
                    "current_part_mask": spaces.MultiBinary(self.max_parts),
                    "parent_adjacency": spaces.MultiBinary(
                        (self.max_parts, self.max_parts)
                    ),
                    "dependency_adjacency": spaces.MultiBinary(
                        (self.max_parts, self.max_parts)
                    ),
                    "symmetry_adjacency": spaces.MultiBinary(
                        (self.max_parts, self.max_parts)
                    ),
                    "pose_features": spaces.Box(
                        -1.0,
                        1.0,
                        shape=(self.max_poses, self.POSE_FEATURE_DIM),
                        dtype=np.float32,
                    ),
                    "pose_mask": spaces.MultiBinary(self.max_poses),
                    "grasp_hint": spaces.Box(
                        0.0, 1.0, shape=(self.max_poses,), dtype=np.float32
                    ),
                    "ik_hint": spaces.Box(
                        0.0,
                        1.0,
                        shape=(self.max_poses, self.grid_height, self.grid_width),
                        dtype=np.float32,
                    ),
                    "hint_available": spaces.MultiBinary(2),
                }
            )

    # ----------------------------- Gym API -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        region_id = str(options.get("region_id") or self._sample_region_id())
        if region_id not in self.region_ids:
            raise ValueError(f"invalid reset region {region_id!r}; valid={self.region_ids}")

        self.state = AssemblyGridState(
            self.validator,
            self.task,
            region_id=region_id,
            resolution=self.resolution,
            max_poses=self.max_poses,
        )
        self.current_region_id = region_id
        self._terminated = False
        self._truncated = False
        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_l2_result = None

        info = self._base_info()
        info.update(
            {
                "event": "reset",
                "valid_action_count": int(self.action_masks().sum()),
            }
        )
        return self._observation(), info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        state = self._require_state()
        if self._terminated or self._truncated:
            raise RuntimeError("episode ended; call reset()")

        action = int(action)
        check = state.check_action(action)
        if not check.valid:
            self._terminated = True
            reward = -1.0
            self._episode_return += reward
            info = self._base_info()
            info.update(
                {
                    "event": "invalid_action",
                    "action": action,
                    "failure_reason": check.reason,
                    "episode_steps": self._episode_steps,
                    "episode_return": self._episode_return,
                    "valid_action_count": 0,
                }
            )
            return self._observation(), reward, True, False, info

        part = str(state.current_part)
        pose_id, row, col = GridWorkspace.unflatten_action(
            action,
            height=self.grid_height,
            width=self.grid_width,
        )
        record = state.apply_action(action)
        self._episode_steps += 1
        reward = self._placement_shaping_reward(part, record.xy)
        info = self._base_info()
        info.update(
            {
                "event": "placement",
                "action": action,
                "part_id": part,
                "pose_id": int(pose_id),
                "grid_row": int(row),
                "grid_col": int(col),
                "xy": np.asarray(record.xy, dtype=float).tolist(),
                "z_offset": float(record.z_offset),
                "placement_reward": float(reward),
            }
        )

        if state.done:
            request = state.export_fixed_l2_request()
            info["fixed_l2_request"] = request
            if self.run_l2:
                result = self.validator.evaluate_fixed_layout(
                    region_id=request["region_id"],
                    part_xy={p: row["xy"] for p, row in request["parts"].items()},
                    part_pose_id={
                        p: int(row["pose_id"]) for p, row in request["parts"].items()
                    },
                )
                self._last_l2_result = result
                terminal_reward = self._l2_terminal_reward(result)
                reward += terminal_reward
                info.update(
                    {
                        "event": "l2_complete",
                        "l2_pass": bool(result.l2_pass),
                        "layout_score": float(result.layout_score),
                        "fail_part": result.fail_part,
                        "fail_reason": result.fail_reason,
                        "fail_detail": dict(result.fail_detail),
                        "terminal_reward": float(terminal_reward),
                        "l2_result": result.to_dict(),
                    }
                )
            else:
                info.update(
                    {
                        "event": "layout_complete_without_l2",
                        "l2_pass": None,
                        "layout_score": None,
                        "terminal_reward": 0.0,
                    }
                )
            self._terminated = True
        elif not np.any(state.flat_action_mask()):
            reward -= 2.0
            self._terminated = True
            info.update(
                {
                    "event": "dead_end",
                    "dead_end_part": state.current_part,
                    "failure_reason": "no_valid_action_for_next_part",
                    "terminal_reward": -2.0,
                }
            )

        self._episode_return += float(reward)
        info.update(
            {
                "episode_steps": int(self._episode_steps),
                "episode_return": float(self._episode_return),
                "next_part": None if self._terminated else state.current_part,
                "valid_action_count": int(self.action_masks().sum()),
            }
        )
        return self._observation(), float(reward), self._terminated, self._truncated, info

    def close(self) -> None:
        # The searcher has already loaded all ASMDEF contents, so the temporary
        # normalised file can safely be removed when the environment closes.
        prepared = self.prepared_asmdef
        if prepared and prepared.temporary:
            try:
                os.remove(prepared.effective_path)
            except FileNotFoundError:
                pass

    def render(self) -> None:
        state = self._require_state()
        payload = {
            "task": self.task.name,
            "region_id": self.current_region_id,
            "preassembled_part": self.task.first_part,
            "current_step": state.current_step,
            "current_part": state.current_part,
            "placements": {
                p: row.to_dict() for p, row in state.workspace.placements.items()
            },
            "episode_return": self._episode_return,
            "terminated": self._terminated,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # -------------------------- action masks ---------------------------

    def action_masks(self) -> np.ndarray:
        if self.state is None or self._terminated or self._truncated:
            return np.zeros(self.action_n, dtype=bool)
        return self.state.flat_action_mask()

    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def sample_valid_action(self) -> int:
        valid = np.flatnonzero(self.action_masks())
        if valid.size == 0:
            raise RuntimeError("current state has no valid action")
        return int(self._rng.choice(valid))

    # ------------------------- observations ----------------------------

    def _observation(self) -> Dict[str, np.ndarray]:
        state = self._require_state()
        current_part_mask = np.zeros(self.max_parts, dtype=np.int8)
        if not self._terminated and state.current_part is not None:
            current_part_mask[self.task.part_to_index[str(state.current_part)]] = 1

        part_mask = np.zeros(self.max_parts, dtype=np.int8)
        part_mask[: self.n_parts] = 1
        decision_mask = np.zeros(self.max_parts, dtype=np.int8)
        decision_mask[1 : self.n_parts] = 1  # index 0 is always preassembled.

        pose_mask = state.pose_mask().astype(np.int8)
        return {
            "occupancy": np.asarray(
                state.workspace.occupancy_channels(), dtype=np.float32
            ),
            "action_mask": self.action_masks().astype(np.int8),
            "assembly_center": self._normalized_assembly_center(state),
            "step_normalized": np.asarray(
                [state.current_step / max(1, self.n_decision_parts)],
                dtype=np.float32,
            ),
            "part_features": self._part_features(state),
            "part_mask": part_mask,
            "decision_mask": decision_mask,
            "current_part_mask": current_part_mask,
            "parent_adjacency": self._parent_adjacency.copy(),
            "dependency_adjacency": self._dependency_adjacency.copy(),
            "symmetry_adjacency": self._symmetry_adjacency.copy(),
            "pose_features": self._current_pose_features(state),
            "pose_mask": pose_mask,
            # Real hints are plugged in during step 3 of the project plan.
            "grasp_hint": np.zeros(self.max_poses, dtype=np.float32),
            "ik_hint": np.zeros(
                (self.max_poses, self.grid_height, self.grid_width),
                dtype=np.float32,
            ),
            "hint_available": np.zeros(2, dtype=np.int8),
        }

    def _normalized_assembly_center(self, state: AssemblyGridState) -> np.ndarray:
        spec = state.workspace.spec
        center = np.asarray(state.region_center, dtype=float)[:2]
        return np.asarray(
            [
                (center[0] - spec.x_min) / max(1e-9, spec.x_max - spec.x_min),
                (center[1] - spec.y_min) / max(1e-9, spec.y_max - spec.y_min),
            ],
            dtype=np.float32,
        )

    def _part_features(self, state: AssemblyGridState) -> np.ndarray:
        out = np.zeros((self.max_parts, self.PART_FEATURE_DIM), dtype=np.float32)
        spec = state.workspace.spec
        span_x = max(1e-9, spec.x_max - spec.x_min)
        span_y = max(1e-9, spec.y_max - spec.y_min)
        span_z = max(span_x, span_y)
        center = np.asarray(state.region_center, dtype=float)
        index_of = self.task.part_to_index

        for index, part in enumerate(self.task.part_order):
            verts = np.asarray(self.searcher.mesh_vertices[part], dtype=float)
            extent = verts.max(axis=0) - verts.min(axis=0)
            goal_pos, goal_rot = self.searcher.world_poses[part]
            goal_pos = np.asarray(goal_pos, dtype=float)
            step_spec = self.task.steps[index]
            placed_record = state.workspace.placements.get(part)
            placed = placed_record is not None
            is_current = bool(state.current_part == part and not self._terminated)

            row = np.zeros(self.PART_FEATURE_DIM, dtype=np.float32)
            # 0..2: explicit STL extent.
            row[0:3] = [extent[0] / span_x, extent[1] / span_y, extent[2] / span_z]
            # 3: bounded mass feature.
            mass = self.task.masses.get(part, 0.0)
            row[3] = mass / (1.0 + mass)
            # 4: bounding-box volume relative to workspace cube.
            row[4] = float(np.prod(extent)) / max(1e-9, span_x * span_y * span_z)
            # 5: ASMDEF step, normalised across task length.
            row[5] = index / max(1, self.n_parts - 1)
            # 6..8: target position relative to selected assembly center.
            delta = goal_pos - center
            row[6:9] = [delta[0] / span_x, delta[1] / span_y, delta[2] / span_z]
            # 9..14: target rotation 6-D.
            row[9:15] = _rotation_6d(goal_rot)
            # 15..16: parent relation.
            parent_index = index_of.get(step_spec.parent, -1)
            row[15] = parent_index / max(1, self.n_parts - 1) if parent_index >= 0 else 0.0
            row[16] = 1.0 if parent_index >= 0 else 0.0
            # 17: dependency count.
            row[17] = len(step_spec.deps) / max(1, self.n_parts - 1)
            # 18: symmetry-group size.
            row[18] = self.task.symmetry_group_sizes.get(part, 1) / max(1, self.n_parts)
            # 19..20: dynamic placement/current flags.
            row[19] = 1.0 if placed else 0.0
            row[20] = 1.0 if is_current else 0.0
            # 21..23: selected staging pose and XY, if already placed.
            if placed_record is not None:
                if placed_record.pose_id >= 0:
                    row[21] = placed_record.pose_id / max(1, self.max_poses - 1)
                row[22] = (placed_record.xy[0] - spec.x_min) / span_x
                row[23] = (placed_record.xy[1] - spec.y_min) / span_y
            # 24: first/preassembled flag; 25: valid/non-padding flag.
            row[24] = 1.0 if index == 0 else 0.0
            row[25] = 1.0
            out[index] = np.clip(row, -1.0, 1.0)
        return out

    def _current_pose_features(self, state: AssemblyGridState) -> np.ndarray:
        out = np.zeros((self.max_poses, self.POSE_FEATURE_DIM), dtype=np.float32)
        if self._terminated or state.current_part is None:
            return out
        part = str(state.current_part)
        spec = state.workspace.spec
        span_x = max(1e-9, spec.x_max - spec.x_min)
        span_y = max(1e-9, spec.y_max - spec.y_min)
        span_z = max(span_x, span_y)
        for pose_id, cand in enumerate(self.searcher.rot_cands[part]):
            if pose_id >= self.max_poses:
                break
            rot = np.asarray(cand.rotmat, dtype=float)
            extent = np.asarray(cand.extent, dtype=float)
            footprint = np.asarray(cand.footprint, dtype=float)
            tag = str(cand.tag).lower()
            row = np.zeros(self.POSE_FEATURE_DIM, dtype=np.float32)
            row[0:6] = _rotation_6d(rot)
            row[6:9] = [extent[0] / span_x, extent[1] / span_y, extent[2] / span_z]
            row[9:11] = [footprint[0] / span_x, footprint[1] / span_y]
            row[11] = _safe_float(cand.z_offset) / span_z
            row[12] = 1.0 if "identity" in tag else 0.0
            row[13] = 1.0 if "stable" in tag else 0.0
            row[14] = 1.0 if "upright" in tag else 0.0
            row[15] = 1.0 if ("rot90" in tag or "auto" in tag) else 0.0
            row[16] = 1.0
            out[pose_id] = np.clip(row, -1.0, 1.0)
        return out

    # ------------------------- graph tensors ---------------------------

    def _build_parent_adjacency(self) -> np.ndarray:
        out = np.zeros((self.max_parts, self.max_parts), dtype=np.int8)
        index = self.task.part_to_index
        for step in self.task.steps:
            if step.parent in index:
                out[index[step.parent], index[step.part]] = 1
        return out

    def _build_dependency_adjacency(self) -> np.ndarray:
        out = np.zeros((self.max_parts, self.max_parts), dtype=np.int8)
        index = self.task.part_to_index
        for step in self.task.steps:
            for dependency_step in step.deps:
                if 0 <= dependency_step < len(self.task.steps):
                    source = self.task.steps[dependency_step].part
                    out[index[source], index[step.part]] = 1
        return out

    def _build_symmetry_adjacency(self) -> np.ndarray:
        out = np.zeros((self.max_parts, self.max_parts), dtype=np.int8)
        index = self.task.part_to_index
        groups: Dict[str, List[str]] = {}
        for part, group in self.task.symmetry_group_for_part.items():
            groups.setdefault(group, []).append(part)
        for members in groups.values():
            for a in members:
                for b in members:
                    if a != b:
                        out[index[a], index[b]] = 1
        return out

    # ----------------------- rewards/regions/info ----------------------

    def _placement_shaping_reward(self, part: str, xy: Sequence[float]) -> float:
        state = self._require_state()
        target = np.asarray(self.searcher.world_poses[part][0], dtype=float)[:2]
        distance = float(np.linalg.norm(np.asarray(xy, dtype=float) - target))
        spec = state.workspace.spec
        diagonal = math.hypot(spec.x_max - spec.x_min, spec.y_max - spec.y_min)
        return float(0.05 * (1.0 - min(1.0, distance / max(1e-9, diagonal))))

    @staticmethod
    def _l2_terminal_reward(result: FixedL2Result) -> float:
        if not result.l2_pass:
            return -2.0
        score = float(np.clip(_safe_float(result.layout_score), 0.0, 1.0))
        return 5.0 + 2.0 * score

    def _default_region_id(self) -> str:
        return self.preferred_region_ids[0] if self.preferred_region_ids else self.region_ids[0]

    def _sample_region_id(self) -> str:
        if self.region_mode == REGION_MODE_FIXED:
            return self.fixed_region_id
        if self.region_mode == REGION_MODE_PREFERRED:
            pool = self.preferred_region_ids or self.region_ids
            return str(self._rng.choice(pool))
        if not self.preferred_region_ids or not self.other_region_ids:
            return str(self._rng.choice(self.region_ids))
        choose_preferred = bool(self._rng.random() < self.preferred_mass)
        pool = self.preferred_region_ids if choose_preferred else self.other_region_ids
        return str(self._rng.choice(pool))

    def _require_state(self) -> AssemblyGridState:
        if self.state is None:
            raise RuntimeError("call reset() before using the environment")
        return self.state

    def _base_info(self) -> Dict[str, Any]:
        state = self._require_state()
        return {
            "task_name": self.task.name,
            "region_id": self.current_region_id,
            "assembly_center": np.asarray(state.region_center, dtype=float).tolist(),
            "preassembled_part": self.task.first_part,
            "decision_parts": list(self.task.decision_parts),
            "current_part": state.current_part,
        }

    def compact_description(self) -> Dict[str, Any]:
        prepared = self.prepared_asmdef
        return {
            "task_name": self.task.name,
            "asmdef": self.task.asmdef_path,
            "preassembled_part": self.task.first_part,
            "first_part_is_action_target": False,
            "first_part_is_grasp_target": False,
            "first_part_original_rel_pos": (
                list(prepared.original_first_rel_pos) if prepared else None
            ),
            "force_first_at_region_center": (
                bool(prepared.forced_first_at_region_center) if prepared else None
            ),
            "part_order": list(self.task.part_order),
            "decision_parts": list(self.task.decision_parts),
            "real_part_count": self.n_parts,
            "decision_count": self.n_decision_parts,
            "max_parts": self.max_parts,
            "max_poses": self.max_poses,
            "valid_regions": list(self.region_ids),
            "preferred_regions": list(self.preferred_region_ids),
            "region_mode": self.region_mode,
            "grid_shape": [self.grid_height, self.grid_width],
            "episode_horizon": self.n_decision_parts,
            "action_space_size": self.action_n,
            "observation_shapes": {
                "occupancy": [4, self.grid_height, self.grid_width],
                "action_mask": [self.action_n],
                "part_features": [self.max_parts, self.PART_FEATURE_DIM],
                "part_mask": [self.max_parts],
                "decision_mask": [self.max_parts],
                "current_part_mask": [self.max_parts],
                "parent_adjacency": [self.max_parts, self.max_parts],
                "dependency_adjacency": [self.max_parts, self.max_parts],
                "symmetry_adjacency": [self.max_parts, self.max_parts],
                "pose_features": [self.max_poses, self.POSE_FEATURE_DIM],
                "pose_mask": [self.max_poses],
                "grasp_hint": [self.max_poses],
                "ik_hint": [self.max_poses, self.grid_height, self.grid_width],
            },
            "gymnasium_available": gym is not None,
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generic masked-random ASMDEF layout episodes."
    )
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--max-parts", type=int, default=12)
    parser.add_argument("--max-poses", type=int, default=16)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--region-mode", choices=REGION_MODES, default=REGION_MODE_PREFERRED)
    parser.add_argument("--region-id", default=None)
    parser.add_argument("--preferred-mass", type=float, default=0.75)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-l2", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument(
        "--keep-first-rel-pos",
        action="store_true",
        help="Do not force ASMDEF step-0 rel_pos to [0,0,0].",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validator, task, prepared = build_assembly_validator(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_poses=args.max_poses,
        force_first_at_region_center=not args.keep_first_rel_pos,
    )
    env = AssemblyLayoutEnv(
        validator,
        task,
        resolution=args.resolution,
        max_parts=args.max_parts,
        max_poses=args.max_poses,
        region_mode=args.region_mode,
        fixed_region_id=args.region_id,
        preferred_mass=args.preferred_mass,
        run_l2=not args.skip_l2,
        seed=args.seed,
        prepared_asmdef=prepared,
    )

    print("========== AssemblyLayoutEnv ==========")
    print(json.dumps(env.compact_description(), ensure_ascii=False, indent=2))
    if args.describe_only:
        env.close()
        return

    episodes: List[Dict[str, Any]] = []
    pass_count = 0
    try:
        for episode in range(args.episodes):
            options = None
            if args.region_mode == REGION_MODE_FIXED and args.region_id:
                options = {"region_id": args.region_id}
            _, info = env.reset(seed=args.seed + episode, options=options)
            print(
                f"[reset] episode={episode:03d} task={task.name} "
                f"region={info['region_id']} preassembled={task.first_part} "
                f"part={info['current_part']} valid={info['valid_action_count']}"
            )
            terminated = truncated = False
            final_info = info
            while not (terminated or truncated):
                action = env.sample_valid_action()
                _, reward, terminated, truncated, final_info = env.step(action)
                print(
                    f"[step] episode={episode:03d} "
                    f"step={final_info['episode_steps']:02d}/{env.n_decision_parts:02d} "
                    f"event={final_info['event']:22s} "
                    f"part={str(final_info.get('part_id', '-')):16s} "
                    f"pose={str(final_info.get('pose_id', '-')):>2s} "
                    f"reward={reward:+.4f} next={final_info.get('next_part')}"
                )
            if final_info.get("l2_pass") is True:
                pass_count += 1
            episodes.append(
                {
                    "episode": episode,
                    "region_id": final_info.get("region_id"),
                    "event": final_info.get("event"),
                    "episode_steps": final_info.get("episode_steps"),
                    "episode_return": final_info.get("episode_return"),
                    "l2_pass": final_info.get("l2_pass"),
                    "layout_score": final_info.get("layout_score"),
                    "fail_part": final_info.get("fail_part"),
                    "fail_reason": final_info.get("fail_reason"),
                    "fixed_l2_request": final_info.get("fixed_l2_request"),
                    "l2_result": final_info.get("l2_result"),
                }
            )
    finally:
        env.close()

    payload = {
        "environment": env.compact_description(),
        "episodes": episodes,
        "statistics": {
            "episode_count": len(episodes),
            "l2_pass_count": pass_count,
            "l2_pass_rate": pass_count / max(1, len(episodes)),
        },
    }
    print(json.dumps(_jsonable(payload["statistics"]), ensure_ascii=False, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        print(f"[OK] saved to: {path}")


if __name__ == "__main__":
    main()
