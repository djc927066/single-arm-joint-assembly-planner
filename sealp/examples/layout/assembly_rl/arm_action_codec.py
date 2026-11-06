#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Arm-conditioned action codec.

Joint action:
    (arm_id, pose_id, row, col)

Flattening order:
    arm -> pose -> row -> col

For 2 arms, 16 poses and a 54x24 grid:
    action_n = 2 * 16 * 54 * 24 = 41472
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


ARM_LEFT = 0
ARM_RIGHT = 1
ARM_NAMES = ("lft", "rgt")


@dataclass(frozen=True)
class ArmActionCodec:
    n_arms: int
    n_poses: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.n_arms, self.n_poses, self.height, self.width) <= 0:
            raise ValueError("all dimensions must be positive")

    @property
    def cells(self) -> int:
        return self.height * self.width

    @property
    def spatial_action_n(self) -> int:
        return self.n_poses * self.cells

    @property
    def action_n(self) -> int:
        return self.n_arms * self.spatial_action_n

    def flatten(self, arm_id: int, pose_id: int, row: int, col: int) -> int:
        arm_id = int(arm_id)
        pose_id = int(pose_id)
        row = int(row)
        col = int(col)
        if not 0 <= arm_id < self.n_arms:
            raise ValueError(f"arm_id out of range: {arm_id}")
        if not 0 <= pose_id < self.n_poses:
            raise ValueError(f"pose_id out of range: {pose_id}")
        if not 0 <= row < self.height:
            raise ValueError(f"row out of range: {row}")
        if not 0 <= col < self.width:
            raise ValueError(f"col out of range: {col}")
        return (((arm_id * self.n_poses + pose_id) * self.height + row) * self.width + col)

    def unflatten(self, action: int) -> Tuple[int, int, int, int]:
        action = int(action)
        if not 0 <= action < self.action_n:
            raise ValueError(f"action out of range: {action}")
        col = action % self.width
        q = action // self.width
        row = q % self.height
        q //= self.height
        pose_id = q % self.n_poses
        arm_id = q // self.n_poses
        return arm_id, pose_id, row, col

    def spatial_action(self, action: int) -> int:
        """Drop arm_id and return the old (pose,row,col) flat action."""
        _, pose_id, row, col = self.unflatten(action)
        return (pose_id * self.height + row) * self.width + col

    def expand_spatial_mask(
        self,
        spatial_mask: np.ndarray,
        ik_hint_by_arm: np.ndarray | None = None,
        hard_ik_threshold: float | None = None,
    ) -> np.ndarray:
        """Expand old [pose,H,W] geometry mask to [arm,pose,H,W].

        IK remains a soft hint by default. Set hard_ik_threshold only for
        diagnostics, not as the permanent training default.
        """
        mask = np.asarray(spatial_mask, dtype=bool)
        expected = (self.n_poses, self.height, self.width)
        if mask.shape != expected:
            raise ValueError(f"spatial_mask shape {mask.shape}, expected {expected}")
        out = np.broadcast_to(mask[None, ...], (self.n_arms, *expected)).copy()

        if hard_ik_threshold is not None:
            if ik_hint_by_arm is None:
                raise ValueError("ik_hint_by_arm is required for hard IK masking")
            hints = np.asarray(ik_hint_by_arm, dtype=np.float32)
            expected_hints = (self.n_arms, *expected)
            if hints.shape != expected_hints:
                raise ValueError(
                    f"ik_hint_by_arm shape {hints.shape}, expected {expected_hints}"
                )
            out &= hints > float(hard_ik_threshold)

        return out.reshape(-1)


def self_test() -> None:
    codec = ArmActionCodec(n_arms=2, n_poses=16, height=54, width=24)
    assert codec.action_n == 41472

    probes = [
        (0, 0, 0, 0),
        (0, 15, 53, 23),
        (1, 0, 0, 0),
        (1, 15, 53, 23),
        (1, 7, 18, 11),
    ]
    for item in probes:
        action = codec.flatten(*item)
        decoded = codec.unflatten(action)
        assert decoded == item, (item, action, decoded)

    spatial = np.ones((16, 54, 24), dtype=bool)
    expanded = codec.expand_spatial_mask(spatial)
    assert expanded.shape == (41472,)
    assert int(expanded.sum()) == 41472

    print("[OK] ArmActionCodec self-test passed")
    print(f"action_n={codec.action_n}")
    print("flattening order=arm->pose->row->col")


if __name__ == "__main__":
    self_test()
