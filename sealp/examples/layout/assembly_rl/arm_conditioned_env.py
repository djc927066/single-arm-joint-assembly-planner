#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage-B arm-conditioned wrapper with strict L2 arm enforcement.

This wrapper upgrades the policy-facing action from

    (pose_id, row, col)

to

    (arm_id, pose_id, row, col)

without changing the already-tested geometry state machine.

Stage-B rule
------------
The final L2 evaluator must use exactly the arm selected by the joint action.
Before the terminal action triggers L2, this wrapper temporarily replaces the
searcher's ``_arm_order(part_id)`` with a strict one-arm order.  The original
method is restored immediately after L2 returns.

This reuses the already-tested FixedL2Validator and its exact collision,
common-grasp, IK, manipulability and score calculations.  The only changed
semantic is: no automatic fallback to the other arm.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    spaces = None

from .arm_action_codec import ARM_NAMES, ArmActionCodec


class ArmHintStore:
    """Load and index left/right IK-hint arrays from the existing NPZ."""

    def __init__(
        self,
        npz_path: str,
        *,
        decision_parts,
        max_poses: int,
        grid_shape: Tuple[int, int],
    ) -> None:
        path = Path(npz_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"IK hint cache not found: {path}")

        with np.load(path, allow_pickle=True) as npz:
            required = ("region_ids", "part_ids", "left_scores", "right_scores")
            missing = [key for key in required if key not in npz.files]
            if missing:
                raise ValueError(
                    f"IK cache is missing keys {missing}; available={npz.files}"
                )
            self.region_ids = [str(v) for v in npz["region_ids"].tolist()]
            self.part_ids = [str(v) for v in npz["part_ids"].tolist()]
            self.left_scores = np.asarray(npz["left_scores"], dtype=np.float32)
            self.right_scores = np.asarray(npz["right_scores"], dtype=np.float32)
            self.scalar_scores = (
                np.asarray(npz["scores"], dtype=np.float32)
                if "scores" in npz.files
                else None
            )

        expected = (
            len(self.region_ids),
            len(self.part_ids),
            int(max_poses),
            int(grid_shape[0]),
            int(grid_shape[1]),
        )
        if self.left_scores.shape != expected:
            raise ValueError(
                f"left_scores shape={self.left_scores.shape}, expected={expected}"
            )
        if self.right_scores.shape != expected:
            raise ValueError(
                f"right_scores shape={self.right_scores.shape}, expected={expected}"
            )

        missing_parts = [p for p in decision_parts if p not in self.part_ids]
        if missing_parts:
            raise ValueError(
                f"IK cache does not cover decision parts {missing_parts}; "
                f"cache parts={self.part_ids}"
            )

        if self.scalar_scores is not None:
            expected_scalar = np.maximum(self.left_scores, self.right_scores)
            if not np.allclose(
                self.scalar_scores,
                expected_scalar,
                atol=1e-6,
                equal_nan=True,
            ):
                max_error = float(
                    np.nanmax(np.abs(self.scalar_scores - expected_scalar))
                )
                raise ValueError(
                    "cache consistency check failed: scores != max(left,right), "
                    f"max_abs_error={max_error:.6g}"
                )

        self._region_to_index = {
            region_id: i for i, region_id in enumerate(self.region_ids)
        }
        self._part_to_index = {
            part_id: i for i, part_id in enumerate(self.part_ids)
        }
        self.source_path = str(path)

    def has_region(self, region_id: str) -> bool:
        return str(region_id) in self._region_to_index

    def scores_for(self, region_id: str, part_id: str) -> np.ndarray:
        """Return [2, max_poses, H, W] in left/right order."""
        rid = str(region_id)
        pid = str(part_id)
        if rid not in self._region_to_index:
            raise KeyError(
                f"region {rid!r} not in IK cache {self.region_ids}"
            )
        if pid not in self._part_to_index:
            raise KeyError(
                f"part {pid!r} not in IK cache {self.part_ids}"
            )
        ri = self._region_to_index[rid]
        pi = self._part_to_index[pid]
        return np.stack(
            [self.left_scores[ri, pi], self.right_scores[ri, pi]],
            axis=0,
        ).astype(np.float32, copy=False)


class ArmConditionedAssemblyEnv:
    """Policy-facing wrapper with joint arm/pose/row/column actions."""

    def __init__(
        self,
        base_env,
        *,
        ik_hint_path: str,
        hard_mask_zero_arm_ik: bool = True,
        arm_ik_threshold: float = 0.0,
    ) -> None:
        self.base_env = base_env
        self.n_arms = 2
        self.max_poses = int(base_env.max_poses)
        self.grid_height = int(base_env.grid_height)
        self.grid_width = int(base_env.grid_width)
        self.codec = ArmActionCodec(
            n_arms=self.n_arms,
            n_poses=self.max_poses,
            height=self.grid_height,
            width=self.grid_width,
        )
        self.action_n = self.codec.action_n
        self.hard_mask_zero_arm_ik = bool(hard_mask_zero_arm_ik)
        self.arm_ik_threshold = float(arm_ik_threshold)
        self.arm_hints = ArmHintStore(
            ik_hint_path,
            decision_parts=base_env.task.decision_parts,
            max_poses=self.max_poses,
            grid_shape=(self.grid_height, self.grid_width),
        )
        self.selected_arms: Dict[str, str] = {}

        if spaces is not None:
            self.action_space = spaces.Discrete(self.action_n)
        else:
            self.action_space = None

    @property
    def task(self):
        return self.base_env.task

    @property
    def state(self):
        return self.base_env.state

    @property
    def n_decision_parts(self) -> int:
        return int(self.base_env.n_decision_parts)

    def _current_context(self) -> Tuple[Optional[str], Optional[str]]:
        state = self.base_env.state
        if state is None:
            return None, None
        region_id = str(state.region_id)
        part_id = state.current_part
        return region_id, None if part_id is None else str(part_id)

    def current_arm_hints(self) -> np.ndarray:
        """Return [2,K,H,W]; zeros after the episode is complete."""
        region_id, part_id = self._current_context()
        if region_id is None or part_id is None:
            return np.zeros(
                (
                    self.n_arms,
                    self.max_poses,
                    self.grid_height,
                    self.grid_width,
                ),
                dtype=np.float32,
            )
        return self.arm_hints.scores_for(region_id, part_id)

    def _augment_observation(
        self, observation: Mapping[str, Any]
    ) -> Dict[str, Any]:
        out = dict(observation)
        spatial_mask = np.asarray(
            observation["action_mask"], dtype=bool
        ).reshape(
            self.max_poses,
            self.grid_height,
            self.grid_width,
        )
        hints = self.current_arm_hints()
        joint_mask = np.broadcast_to(
            spatial_mask[None, ...],
            (
                self.n_arms,
                self.max_poses,
                self.grid_height,
                self.grid_width,
            ),
        ).copy()

        if self.hard_mask_zero_arm_ik:
            joint_mask &= hints > self.arm_ik_threshold

        # Keep the old mask under an explicit name for debugging.
        out["spatial_action_mask"] = spatial_mask.reshape(-1).astype(np.int8)
        out["ik_hint_by_arm"] = hints.astype(np.float32, copy=False)
        out["action_mask"] = joint_mask.reshape(-1).astype(np.int8)
        return out


    def _l2_searcher_objects(self) -> List[Any]:
        """Return every distinct searcher object that may execute L2."""
        candidates = [
            getattr(self.base_env.validator, "searcher", None),
            getattr(self.base_env.validator, "_searcher", None),
            getattr(self.base_env.state, "searcher", None)
            if self.base_env.state is not None
            else None,
        ]
        out: List[Any] = []
        seen = set()
        for obj in candidates:
            if obj is None or id(obj) in seen:
                continue
            if not hasattr(obj, "_arm_order"):
                continue
            out.append(obj)
            seen.add(id(obj))
        if not out:
            raise RuntimeError(
                "Cannot enforce selected arms: no L2 searcher with _arm_order "
                "was found on validator/state."
            )
        return out

    @contextmanager
    def _strict_arm_order_context(
        self,
        selected_arms: Mapping[str, str],
    ) -> Iterator[List[Dict[str, Any]]]:
        """Temporarily force each decision part to use exactly one arm.

        The underlying searcher evaluates arms with:

            for arm_tag in self._arm_order(pid):

        Returning a one-element tuple disables automatic fallback to the other
        arm while preserving all existing L2 calculations.
        """
        normalized = {
            str(part): str(arm)
            for part, arm in selected_arms.items()
        }
        invalid = {
            part: arm
            for part, arm in normalized.items()
            if arm not in ARM_NAMES
        }
        if invalid:
            raise ValueError(f"invalid selected arm map: {invalid}")

        targets = self._l2_searcher_objects()
        restore_rows = []
        trace: List[Dict[str, Any]] = []

        try:
            for searcher in targets:
                instance_dict = getattr(searcher, "__dict__", {})
                had_instance_attr = "_arm_order" in instance_dict
                old_instance_value = instance_dict.get("_arm_order")
                old_bound_method = getattr(searcher, "_arm_order")

                def strict_arm_order(
                    _self,
                    part_id,
                    _old=old_bound_method,
                    _selected=normalized,
                    _trace=trace,
                ):
                    pid = str(part_id)
                    if pid in _selected:
                        result = (str(_selected[pid]),)
                        forced = True
                    else:
                        result = tuple(_old(pid))
                        forced = False
                    _trace.append(
                        {
                            "part_id": pid,
                            "arm_order": list(result),
                            "forced": bool(forced),
                        }
                    )
                    return result

                searcher._arm_order = MethodType(strict_arm_order, searcher)
                restore_rows.append(
                    (searcher, had_instance_attr, old_instance_value)
                )

            yield trace
        finally:
            for searcher, had_instance_attr, old_instance_value in reversed(
                restore_rows
            ):
                if had_instance_attr:
                    searcher._arm_order = old_instance_value
                else:
                    try:
                        delattr(searcher, "_arm_order")
                    except AttributeError:
                        pass

    @staticmethod
    def _l2_arm_choice_from_info(info: Mapping[str, Any]) -> Dict[str, str]:
        result = info.get("l2_result")
        if not isinstance(result, Mapping):
            return {}
        raw = result.get("arm_choice")
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(part): str(arm)
            for part, arm in raw.items()
            if str(arm) in ARM_NAMES
        }

    def _arm_choice_mismatches(
        self,
        l2_arm_choice: Mapping[str, str],
        *,
        require_all: bool,
    ) -> Dict[str, Dict[str, Optional[str]]]:
        mismatches: Dict[str, Dict[str, Optional[str]]] = {}
        for part, selected in self.selected_arms.items():
            actual = l2_arm_choice.get(part)
            if actual is None and not require_all:
                continue
            if actual != selected:
                mismatches[part] = {
                    "selected": selected,
                    "l2": actual,
                }
        return mismatches

    def reset(self, *args, **kwargs):
        self.selected_arms.clear()
        observation, info = self.base_env.reset(*args, **kwargs)
        info = dict(info)
        info.update(
            {
                "arm_conditioned": True,
                "l2_selected_arm_enforced": True,
                "joint_action_space_size": self.action_n,
                "selected_arms": {},
            }
        )
        return self._augment_observation(observation), info

    def step(self, joint_action: int):
        arm_id, pose_id, row, col = self.codec.unflatten(int(joint_action))
        spatial_action = self.codec.spatial_action(int(joint_action))

        state_before = self.base_env.state
        step_before = -1 if state_before is None else int(state_before.current_step)
        part_before = (
            None
            if state_before is None or state_before.current_part is None
            else str(state_before.current_part)
        )
        selected_arm = ARM_NAMES[arm_id]
        arm_hint_value = float(
            self.current_arm_hints()[arm_id, pose_id, row, col]
        )

        # The terminal action triggers L2 inside base_env.step().  The current
        # part's arm must therefore be added before calling base_env.step().
        will_trigger_l2 = (
            part_before is not None
            and step_before == self.n_decision_parts - 1
        )
        proposed_arm_map = dict(self.selected_arms)
        if part_before is not None:
            proposed_arm_map[part_before] = selected_arm

        strict_trace: List[Dict[str, Any]] = []
        if will_trigger_l2:
            with self._strict_arm_order_context(proposed_arm_map) as trace:
                observation, reward, terminated, truncated, info = (
                    self.base_env.step(spatial_action)
                )
                strict_trace = list(trace)
        else:
            observation, reward, terminated, truncated, info = (
                self.base_env.step(spatial_action)
            )

        state_after = self.base_env.state
        step_after = -1 if state_after is None else int(state_after.current_step)
        accepted = step_after > step_before

        if accepted and part_before is not None:
            self.selected_arms[part_before] = selected_arm

        info = dict(info)
        l2_completed = info.get("event") == "l2_complete"
        strict_enforced = bool(will_trigger_l2 and l2_completed)
        l2_arm_choice = self._l2_arm_choice_from_info(info)

        # On a successful L2 result, every decision part must be returned with
        # exactly the selected arm.  On a failed result, only compare the parts
        # that the evaluator reached before failing.
        require_all = bool(info.get("l2_pass") is True)
        mismatches = self._arm_choice_mismatches(
            l2_arm_choice,
            require_all=require_all,
        )
        arm_choice_match = len(mismatches) == 0

        if strict_enforced and not arm_choice_match:
            raise RuntimeError(
                "Strict L2 arm enforcement mismatch: "
                f"selected={self.selected_arms}, "
                f"l2={l2_arm_choice}, mismatches={mismatches}"
            )

        info.update(
            {
                "joint_action": int(joint_action),
                "spatial_action": int(spatial_action),
                "arm_id": int(arm_id),
                "arm": selected_arm,
                "arm_ik_hint": arm_hint_value,
                "arm_action_accepted": bool(accepted),
                "selected_arms": dict(self.selected_arms),
                "l2_selected_arm_enforced": strict_enforced,
                "strict_arm_order_trace": strict_trace,
                "l2_arm_choice": l2_arm_choice,
                "l2_arm_choice_match": arm_choice_match
                if l2_completed
                else None,
                "l2_arm_choice_mismatches": mismatches,
            }
        )

        if "fixed_l2_request" in info:
            request = deepcopy(info["fixed_l2_request"])
            for part_id, row_data in request.get("parts", {}).items():
                row_data["arm"] = self.selected_arms.get(part_id)
            info["arm_annotated_l2_request"] = request

        return (
            self._augment_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self.base_env.close()

    def compact_description(self) -> Dict[str, Any]:
        base = dict(self.base_env.compact_description())
        base.update(
            {
                "arm_conditioned_stage": "B-strict-l2-arm",
                "joint_action": ["arm", "pose", "row", "col"],
                "joint_action_space_size": self.action_n,
                "ik_hint_by_arm_shape": [
                    self.n_arms,
                    self.max_poses,
                    self.grid_height,
                    self.grid_width,
                ],
                "hard_mask_zero_arm_ik": self.hard_mask_zero_arm_ik,
                "arm_ik_threshold": self.arm_ik_threshold,
                "arm_hint_path": self.arm_hints.source_path,
                "l2_selected_arm_enforced": True,
            }
        )
        return base
