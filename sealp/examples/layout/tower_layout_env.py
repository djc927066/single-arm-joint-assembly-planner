#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gymnasium-compatible Tower layout environment.

This is step 3 of the reinforcement-learning / neural-combinatorial layout
pipeline.  It wraps the already tested components:

* ``FixedL2Validator`` -- validates the exact policy-selected XY and pose IDs;
* ``TowerGridState`` -- 2 cm table grid, footprint occupancy and action masks.

No camera is used.  The observation is a collection of numeric arrays computed
from ASMDEF, STL geometry, the selected assembly region and already placed
parts.

Episode definition
------------------
One episode is one complete layout attempt.  ``base_plate`` is preassembled at
the fixed assembly region.  The policy then makes exactly one action for each
remaining part in ASMDEF order:

    post_br -> post_fr -> post_bl -> post_fl -> middle_plate -> top_cross

Action definition
-----------------
A flattened discrete action encodes ``(pose_id, grid_row, grid_col)``::

    action = pose_id * (H * W) + grid_row * W + grid_col

With the current 2 cm grid, H=54, W=24 and K=12, so the fixed action space has
15552 actions.  Invalid geometric actions are excluded by ``action_mask``.

Reward used in this first environment version
---------------------------------------------
* legal intermediate placement: small distance-based shaping reward;
* completed layout that passes L2: ``5 + 2 * layout_score``;
* completed layout that fails L2: ``-2``;
* invalid/masked action: ``-1`` and terminate;
* geometric dead end before all parts are placed: ``-2`` and terminate.

Grasp Hint and IK Hint arrays are already present in the observation, but are
filled with zeros and marked unavailable in this step.  Later scripts can plug
real precomputed hints into the same stable observation interface.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # The smoke test can still run without Gymnasium being installed.
    import gymnasium as gym
    from gymnasium import spaces

    _EnvBase = gym.Env
except ImportError:  # pragma: no cover - depends on the user's environment
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
from sealp.examples.layout.assembly_rl.grid_workspace import GridWorkspace, TowerGridState  # noqa: E402


REGION_MODE_FIXED = "fixed"
REGION_MODE_PREFERRED = "preferred"
REGION_MODE_ALL = "all"
REGION_MODES = (REGION_MODE_FIXED, REGION_MODE_PREFERRED, REGION_MODE_ALL)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy values to JSON-compatible Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class TowerLayoutEnv(_EnvBase):
    """Sequential layout environment with exact fixed-layout L2 termination.

    Parameters
    ----------
    validator:
        Initialized :class:`FixedL2Validator` containing the existing searcher.
    resolution:
        Table grid resolution in metres.  The project default is 0.02 m.
    region_mode:
        ``preferred`` samples only r1_c0/r1_c1/r1_c2;
        ``all`` samples all seven valid regions, with ``preferred_mass`` total
        probability assigned to the three middle regions;
        ``fixed`` always uses ``fixed_region_id``.
    fixed_region_id:
        Region used when ``region_mode='fixed'``.
    preferred_mass:
        Total probability mass of the three preferred middle regions when
        sampling all seven regions.
    run_l2:
        If true, the sixth action automatically invokes fixed-layout L2.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    PART_FEATURE_DIM = 13
    POSE_FEATURE_DIM = 17

    def __init__(
        self,
        validator: FixedL2Validator,
        *,
        resolution: float = 0.02,
        region_mode: str = REGION_MODE_PREFERRED,
        fixed_region_id: str = "r1_c1",
        preferred_mass: float = 0.75,
        run_l2: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if region_mode not in REGION_MODES:
            raise ValueError(f"region_mode must be one of {REGION_MODES}, got {region_mode!r}")
        if not 0.0 <= float(preferred_mass) <= 1.0:
            raise ValueError("preferred_mass must be in [0, 1]")

        self.validator = validator
        self.searcher = validator.searcher
        self.resolution = float(resolution)
        self.region_mode = str(region_mode)
        self.fixed_region_id = str(fixed_region_id)
        self.preferred_mass = float(preferred_mass)
        self.run_l2 = bool(run_l2)
        self._rng = np.random.default_rng(seed)

        self.region_rows = validator.available_regions()
        self.region_ids = [str(row["region_id"]) for row in self.region_rows]
        self.preferred_region_ids = [
            rid for rid in PREFERRED_REGION_IDS if rid in self.region_ids
        ]
        self.other_region_ids = [
            rid for rid in self.region_ids if rid not in self.preferred_region_ids
        ]
        if not self.region_ids:
            raise RuntimeError("no valid assembly regions are available")
        if self.fixed_region_id not in self.region_ids:
            raise ValueError(
                f"fixed_region_id={self.fixed_region_id!r} is invalid; valid={self.region_ids}"
            )

        # Build one temporary state to determine all fixed space dimensions.
        template_region = self._template_region_id()
        template = TowerGridState(
            validator,
            region_id=template_region,
            resolution=self.resolution,
        )
        self.decision_parts = list(template.decision_parts)
        self.n_parts = len(self.decision_parts)
        self.max_pose_count = int(template.max_pose_count)
        self.grid_height = int(template.workspace.spec.height)
        self.grid_width = int(template.workspace.spec.width)
        self.action_n = int(template.action_space_size)

        self.state: Optional[TowerGridState] = None
        self.current_region_id: Optional[str] = None
        self._terminated = False
        self._truncated = False
        self._episode_return = 0.0
        self._episode_steps = 0
        self._last_l2_result: Optional[FixedL2Result] = None
        self._last_info: Dict[str, Any] = {}

        if spaces is not None:
            self.action_space = spaces.Discrete(self.action_n)
            self.observation_space = spaces.Dict(
                {
                    "occupancy": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(4, self.grid_height, self.grid_width),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.MultiBinary(self.action_n),
                    "assembly_center": spaces.Box(
                        low=0.0, high=1.0, shape=(2,), dtype=np.float32
                    ),
                    "current_step": spaces.Box(
                        low=0.0, high=1.0, shape=(1,), dtype=np.float32
                    ),
                    "current_part_one_hot": spaces.MultiBinary(self.n_parts),
                    "all_part_features": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(self.n_parts, self.PART_FEATURE_DIM),
                        dtype=np.float32,
                    ),
                    "pose_features": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(self.max_pose_count, self.POSE_FEATURE_DIM),
                        dtype=np.float32,
                    ),
                    "grasp_hint": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.max_pose_count,),
                        dtype=np.float32,
                    ),
                    "ik_hint": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.max_pose_count, self.grid_height, self.grid_width),
                        dtype=np.float32,
                    ),
                    "hint_available": spaces.MultiBinary(2),
                }
            )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

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
        requested_region = options.get("region_id")
        if requested_region is not None:
            region_id = str(requested_region)
            if region_id not in self.region_ids:
                raise ValueError(f"invalid reset region {region_id!r}; valid={self.region_ids}")
        else:
            region_id = self._sample_region_id()

        self.state = TowerGridState(
            self.validator,
            region_id=region_id,
            resolution=self.resolution,
        )
        self.current_region_id = region_id
        self._terminated = False
        self._truncated = False
        self._episode_return = 0.0
        self._episode_steps = 0
        self._last_l2_result = None

        obs = self._observation()
        info = self._base_info()
        info.update(
            {
                "event": "reset",
                "valid_action_count": int(self.action_masks().sum()),
            }
        )
        self._last_info = info
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        state = self._require_state()
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after the episode ended; call reset() first")

        action_int = int(action)
        if action_int < 0 or action_int >= self.action_n:
            return self._terminate_invalid_action(action_int, "action_out_of_range")

        check = state.check_action(action_int)
        if not check.valid:
            return self._terminate_invalid_action(action_int, check.reason)

        part_id = str(state.current_part)
        pose_id, row, col = GridWorkspace.unflatten_action(
            action_int,
            height=self.grid_height,
            width=self.grid_width,
        )
        record = state.apply_action(action_int)
        self._episode_steps += 1

        step_reward = self._placement_shaping_reward(part_id, record.xy)
        reward = float(step_reward)
        info = self._base_info()
        info.update(
            {
                "event": "placement",
                "action": action_int,
                "part_id": part_id,
                "pose_id": int(pose_id),
                "grid_row": int(row),
                "grid_col": int(col),
                "xy": np.asarray(record.xy, dtype=float).tolist(),
                "z_offset": float(record.z_offset),
                "placement_reward": float(step_reward),
            }
        )

        # The sixth valid placement creates a complete layout and triggers L2.
        if state.done:
            request = state.export_fixed_l2_request()
            info["fixed_l2_request"] = request
            if self.run_l2:
                result = self.validator.evaluate_fixed_layout(
                    region_id=request["region_id"],
                    part_xy={
                        pid: spec["xy"] for pid, spec in request["parts"].items()
                    },
                    part_pose_id={
                        pid: int(spec["pose_id"])
                        for pid, spec in request["parts"].items()
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

        # A partial layout can geometrically block every action for the next
        # ASMDEF part.  End immediately rather than asking the policy to select
        # an impossible action.
        elif not np.any(state.flat_action_mask()):
            reward += -2.0
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
        self._last_info = info
        obs = self._observation()
        return obs, float(reward), bool(self._terminated), bool(self._truncated), info

    def render(self) -> None:
        state = self._require_state()
        print(
            json.dumps(
                {
                    "region_id": self.current_region_id,
                    "current_step": state.current_step,
                    "current_part": state.current_part,
                    "placed": {
                        pid: record.to_dict()
                        for pid, record in state.workspace.placements.items()
                    },
                    "episode_return": self._episode_return,
                    "terminated": self._terminated,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Action mask interface
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return a Boolean mask compatible with sb3-contrib MaskablePPO."""
        if self.state is None or self._terminated or self._truncated:
            return np.zeros(self.action_n, dtype=bool)
        return np.asarray(self.state.flat_action_mask(), dtype=bool)

    def get_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def sample_valid_action(self) -> int:
        valid = np.flatnonzero(self.action_masks())
        if len(valid) == 0:
            raise RuntimeError("the current state has no valid action")
        return int(self._rng.choice(valid))

    # ------------------------------------------------------------------
    # Numeric observation construction
    # ------------------------------------------------------------------

    def _observation(self) -> Dict[str, np.ndarray]:
        state = self._require_state()
        current_one_hot = np.zeros(self.n_parts, dtype=np.int8)
        if not self._terminated and not state.done and state.current_part is not None:
            current_one_hot[state.current_step] = 1

        hint_available = np.zeros(2, dtype=np.int8)
        grasp_hint = np.zeros(self.max_pose_count, dtype=np.float32)
        ik_hint = np.zeros(
            (self.max_pose_count, self.grid_height, self.grid_width),
            dtype=np.float32,
        )

        return {
            "occupancy": np.asarray(
                state.workspace.occupancy_channels(), dtype=np.float32
            ),
            "action_mask": self.action_masks().astype(np.int8),
            "assembly_center": self._normalized_assembly_center(state),
            "current_step": np.asarray(
                [min(1.0, state.current_step / max(1, self.n_parts))],
                dtype=np.float32,
            ),
            "current_part_one_hot": current_one_hot,
            "all_part_features": self._all_part_features(state),
            "pose_features": self._current_pose_features(state),
            "grasp_hint": grasp_hint,
            "ik_hint": ik_hint,
            "hint_available": hint_available,
        }

    def _normalized_assembly_center(self, state: TowerGridState) -> np.ndarray:
        spec = state.workspace.spec
        center = np.asarray(state.region_center, dtype=float)[:2]
        return np.asarray(
            [
                (center[0] - spec.x_min) / max(1e-9, spec.x_max - spec.x_min),
                (center[1] - spec.y_min) / max(1e-9, spec.y_max - spec.y_min),
            ],
            dtype=np.float32,
        )

    def _all_part_features(self, state: TowerGridState) -> np.ndarray:
        spec = state.workspace.spec
        span_x = max(1e-9, spec.x_max - spec.x_min)
        span_y = max(1e-9, spec.y_max - spec.y_min)
        span_z = max(span_x, span_y)
        features = np.zeros(
            (self.n_parts, self.PART_FEATURE_DIM), dtype=np.float32
        )

        for index, pid in enumerate(self.decision_parts):
            verts = np.asarray(self.searcher.mesh_vertices[pid], dtype=float)
            extent = verts.max(axis=0) - verts.min(axis=0)
            goal_pos = np.asarray(self.searcher.world_poses[pid][0], dtype=float)
            placed = pid in state.selected_xy
            is_current = bool(
                not self._terminated and state.current_part is not None and pid == state.current_part
            )

            row = np.zeros(self.PART_FEATURE_DIM, dtype=np.float32)
            # 0..2: local STL extent.
            row[0:3] = np.asarray(
                [extent[0] / span_x, extent[1] / span_y, extent[2] / span_z],
                dtype=np.float32,
            )
            # 3..5: final assembly target position for the current region.
            row[3:6] = np.asarray(
                [
                    (goal_pos[0] - spec.x_min) / span_x,
                    (goal_pos[1] - spec.y_min) / span_y,
                    goal_pos[2] / span_z,
                ],
                dtype=np.float32,
            )
            # 6..8: ASMDEF order and dynamic state.
            row[6] = index / max(1, self.n_parts - 1)
            row[7] = 1.0 if placed else 0.0
            row[8] = 1.0 if is_current else 0.0
            # 9..10: selected pose state.
            row[9] = 1.0 if placed else 0.0
            if placed:
                row[10] = state.selected_pose_ids[pid] / max(1, self.max_pose_count - 1)
                xy = np.asarray(state.selected_xy[pid], dtype=float)
                row[11] = (xy[0] - spec.x_min) / span_x
                row[12] = (xy[1] - spec.y_min) / span_y
            features[index] = np.clip(row, -1.0, 1.0)
        return features

    def _current_pose_features(self, state: TowerGridState) -> np.ndarray:
        out = np.zeros(
            (self.max_pose_count, self.POSE_FEATURE_DIM), dtype=np.float32
        )
        if self._terminated or state.done or state.current_part is None:
            return out

        pid = str(state.current_part)
        spec = state.workspace.spec
        span_x = max(1e-9, spec.x_max - spec.x_min)
        span_y = max(1e-9, spec.y_max - spec.y_min)
        span_z = max(span_x, span_y)

        for pose_id, cand in enumerate(self.searcher.rot_cands[pid]):
            if pose_id >= self.max_pose_count:
                break
            rot = np.asarray(cand.rotmat, dtype=float)
            extent = np.asarray(cand.extent, dtype=float)
            footprint = np.asarray(cand.footprint, dtype=float)
            tag = str(cand.tag).lower()
            row = np.zeros(self.POSE_FEATURE_DIM, dtype=np.float32)
            # Standard continuous 6-D rotation representation: first 2 columns.
            row[0:6] = np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)
            row[6:9] = np.asarray(
                [extent[0] / span_x, extent[1] / span_y, extent[2] / span_z],
                dtype=np.float32,
            )
            row[9:11] = np.asarray(
                [footprint[0] / span_x, footprint[1] / span_y],
                dtype=np.float32,
            )
            row[11] = _safe_float(cand.z_offset) / span_z
            # Four simple, non-exclusive pose categories.
            row[12] = 1.0 if "identity" in tag else 0.0
            row[13] = 1.0 if "stable" in tag else 0.0
            row[14] = 1.0 if "upright" in tag else 0.0
            row[15] = 1.0 if ("rot90" in tag or "auto" in tag) else 0.0
            row[16] = 1.0  # valid/non-padding pose
            out[pose_id] = np.clip(row, -1.0, 1.0)
        return out

    # ------------------------------------------------------------------
    # Rewards and termination helpers
    # ------------------------------------------------------------------

    def _placement_shaping_reward(self, part_id: str, xy: Sequence[float]) -> float:
        """Small dense reward; L2 remains the authoritative objective."""
        state = self._require_state()
        target_xy = np.asarray(self.searcher.world_poses[part_id][0], dtype=float)[:2]
        distance = float(np.linalg.norm(np.asarray(xy, dtype=float) - target_xy))
        spec = state.workspace.spec
        diagonal = math.hypot(spec.x_max - spec.x_min, spec.y_max - spec.y_min)
        distance_norm = min(1.0, distance / max(1e-9, diagonal))
        # Range approximately [0, 0.05].  This is intentionally much smaller
        # than the terminal L2 reward so it cannot dominate feasibility.
        return float(0.05 * (1.0 - distance_norm))

    @staticmethod
    def _l2_terminal_reward(result: FixedL2Result) -> float:
        if result.l2_pass:
            score = float(np.clip(_safe_float(result.layout_score), 0.0, 1.0))
            return float(5.0 + 2.0 * score)
        return -2.0

    def _terminate_invalid_action(
        self, action: int, reason: str
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._terminated = True
        reward = -1.0
        self._episode_return += reward
        info = self._base_info()
        info.update(
            {
                "event": "invalid_action",
                "action": int(action),
                "failure_reason": str(reason),
                "episode_steps": int(self._episode_steps),
                "episode_return": float(self._episode_return),
                "valid_action_count": 0,
            }
        )
        self._last_info = info
        return self._observation(), reward, True, False, info

    # ------------------------------------------------------------------
    # Region selection and diagnostics
    # ------------------------------------------------------------------

    def _template_region_id(self) -> str:
        if self.region_mode == REGION_MODE_FIXED:
            return self.fixed_region_id
        if self.preferred_region_ids:
            return self.preferred_region_ids[0]
        return self.region_ids[0]

    def _sample_region_id(self) -> str:
        if self.region_mode == REGION_MODE_FIXED:
            return self.fixed_region_id
        if self.region_mode == REGION_MODE_PREFERRED:
            pool = self.preferred_region_ids or self.region_ids
            return str(self._rng.choice(pool))

        # All seven valid regions.  The three middle regions retain priority.
        if not self.preferred_region_ids or not self.other_region_ids:
            return str(self._rng.choice(self.region_ids))
        choose_preferred = bool(self._rng.random() < self.preferred_mass)
        pool = self.preferred_region_ids if choose_preferred else self.other_region_ids
        return str(self._rng.choice(pool))

    def _require_state(self) -> TowerGridState:
        if self.state is None:
            raise RuntimeError("environment has not been reset")
        return self.state

    def _base_info(self) -> Dict[str, Any]:
        state = self._require_state()
        return {
            "region_id": self.current_region_id,
            "region_rc": [int(state.region_rc[0]), int(state.region_rc[1])],
            "assembly_center": np.asarray(state.region_center, dtype=float).tolist(),
            "current_step": int(state.current_step),
            "current_part": state.current_part,
            "decision_parts": list(self.decision_parts),
            "grid_shape": [self.grid_height, self.grid_width],
            "action_space_size": int(self.action_n),
        }

    def compact_description(self) -> Dict[str, Any]:
        return {
            "valid_regions": list(self.region_ids),
            "preferred_regions": list(self.preferred_region_ids),
            "region_mode": self.region_mode,
            "fixed_region_id": self.fixed_region_id,
            "resolution": self.resolution,
            "grid_shape": [self.grid_height, self.grid_width],
            "decision_parts": list(self.decision_parts),
            "episode_horizon": self.n_parts,
            "max_pose_count": self.max_pose_count,
            "action_space_size": self.action_n,
            "observation_shapes": {
                "occupancy": [4, self.grid_height, self.grid_width],
                "action_mask": [self.action_n],
                "assembly_center": [2],
                "current_step": [1],
                "current_part_one_hot": [self.n_parts],
                "all_part_features": [self.n_parts, self.PART_FEATURE_DIM],
                "pose_features": [self.max_pose_count, self.POSE_FEATURE_DIM],
                "grasp_hint": [self.max_pose_count],
                "ik_hint": [self.max_pose_count, self.grid_height, self.grid_width],
                "hint_available": [2],
            },
            "run_l2": self.run_l2,
            "gymnasium_available": gym is not None,
        }


# ---------------------------------------------------------------------------
# Masked-random smoke test
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete masked-random Tower layout episodes."
    )
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--max-rot-candidates", type=int, default=12)
    parser.add_argument(
        "--region-mode",
        choices=REGION_MODES,
        default=REGION_MODE_PREFERRED,
    )
    parser.add_argument("--region-id", default="r1_c1")
    parser.add_argument("--preferred-mass", type=float, default=0.75)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        help="Complete the six geometric actions without invoking L2.",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    searcher = build_default_searcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_rot_candidates=args.max_rot_candidates,
    )
    validator = FixedL2Validator(searcher)
    env = TowerLayoutEnv(
        validator,
        resolution=args.resolution,
        region_mode=args.region_mode,
        fixed_region_id=args.region_id,
        preferred_mass=args.preferred_mass,
        run_l2=not args.skip_l2,
        seed=args.seed,
    )

    print("========== TowerLayoutEnv ==========")
    print(json.dumps(env.compact_description(), ensure_ascii=False, indent=2))

    episode_rows: List[Dict[str, Any]] = []
    pass_count = 0
    for episode_index in range(args.episodes):
        reset_options = None
        if args.region_mode == REGION_MODE_FIXED:
            reset_options = {"region_id": args.region_id}
        obs, info = env.reset(seed=args.seed + episode_index, options=reset_options)
        print(
            f"[reset] episode={episode_index:03d} region={info['region_id']} "
            f"part={info['current_part']} valid={info['valid_action_count']}"
        )

        terminated = False
        truncated = False
        final_info: Dict[str, Any] = info
        while not (terminated or truncated):
            action = env.sample_valid_action()
            obs, reward, terminated, truncated, final_info = env.step(action)
            print(
                f"[step] episode={episode_index:03d} "
                f"step={final_info['episode_steps']:02d} "
                f"event={final_info['event']:22s} "
                f"part={str(final_info.get('part_id', '-')):14s} "
                f"pose={str(final_info.get('pose_id', '-')):>2s} "
                f"reward={reward:+.4f} "
                f"next={str(final_info.get('next_part'))}"
            )

        l2_pass = final_info.get("l2_pass")
        if l2_pass is True:
            pass_count += 1
        row = {
            "episode": episode_index,
            "region_id": final_info.get("region_id"),
            "event": final_info.get("event"),
            "episode_steps": final_info.get("episode_steps"),
            "episode_return": final_info.get("episode_return"),
            "l2_pass": l2_pass,
            "layout_score": final_info.get("layout_score"),
            "fail_part": final_info.get("fail_part"),
            "fail_reason": final_info.get("fail_reason"),
            "fixed_l2_request": final_info.get("fixed_l2_request"),
            "l2_result": final_info.get("l2_result"),
        }
        episode_rows.append(row)
        print(
            f"[done] episode={episode_index:03d} event={row['event']} "
            f"return={_safe_float(row['episode_return']):+.4f} "
            f"l2_pass={row['l2_pass']} fail_part={row['fail_part']}"
        )

    payload = {
        "environment": env.compact_description(),
        "episodes": episode_rows,
        "statistics": {
            "episode_count": int(args.episodes),
            "l2_pass_count": int(pass_count),
            "l2_pass_rate": float(pass_count / args.episodes),
        },
    }
    print(json.dumps(_jsonable(payload["statistics"]), ensure_ascii=False, indent=2))

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        print(f"[OK] episode results saved to: {path}")


if __name__ == "__main__":
    main()
