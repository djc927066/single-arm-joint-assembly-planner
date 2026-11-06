#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime loader for precomputed pose-wise grasp hints.

The first ASMDEF part is intentionally absent from the cache because it is
preassembled at the assembly region and is never a grasp/action target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class GraspHintCache:
    part_ids: tuple[str, ...]
    scores: np.ndarray
    pose_mask: np.ndarray
    pose_valid: np.ndarray
    topdown_ratio: np.ndarray
    table_clear_ratio: np.ndarray
    source_path: str

    @classmethod
    def load(
        cls,
        path: str,
        *,
        decision_parts: Sequence[str],
        max_poses: int,
    ) -> "GraspHintCache":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"grasp hint cache not found: {source}")
        with np.load(source, allow_pickle=False) as data:
            required = {
                "part_ids",
                "scores",
                "pose_mask",
                "pose_valid",
                "topdown_ratio",
                "table_clear_ratio",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"grasp hint cache missing arrays: {sorted(missing)}")
            part_ids = tuple(str(v) for v in data["part_ids"].tolist())
            scores = np.asarray(data["scores"], dtype=np.float32)
            pose_mask = np.asarray(data["pose_mask"], dtype=bool)
            pose_valid = np.asarray(data["pose_valid"], dtype=bool)
            topdown_ratio = np.asarray(data["topdown_ratio"], dtype=np.float32)
            table_clear_ratio = np.asarray(data["table_clear_ratio"], dtype=np.float32)

        expected_shape = (len(part_ids), int(max_poses))
        for name, value in {
            "scores": scores,
            "pose_mask": pose_mask,
            "pose_valid": pose_valid,
            "topdown_ratio": topdown_ratio,
            "table_clear_ratio": table_clear_ratio,
        }.items():
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {value.shape} != expected {expected_shape}; "
                    f"regenerate cache with --max-poses {max_poses}"
                )

        decision_parts = tuple(str(v) for v in decision_parts)
        if set(part_ids) != set(decision_parts):
            raise ValueError(
                "grasp hint parts do not match ASMDEF decision parts: "
                f"cache={part_ids}, asmdef={decision_parts}"
            )
        return cls(
            part_ids=part_ids,
            scores=np.clip(scores, 0.0, 1.0),
            pose_mask=pose_mask,
            pose_valid=pose_valid,
            topdown_ratio=np.clip(topdown_ratio, 0.0, 1.0),
            table_clear_ratio=np.clip(table_clear_ratio, 0.0, 1.0),
            source_path=str(source),
        )

    @property
    def part_to_row(self) -> Dict[str, int]:
        return {part: i for i, part in enumerate(self.part_ids)}

    def vector(self, part_id: str, max_poses: int) -> np.ndarray:
        out = np.zeros(int(max_poses), dtype=np.float32)
        row = self.part_to_row.get(str(part_id))
        if row is None:
            return out
        count = min(out.size, self.scores.shape[1])
        out[:count] = self.scores[row, :count]
        return out

    def valid_pose_mask(self, part_id: str, max_poses: int) -> np.ndarray:
        out = np.zeros(int(max_poses), dtype=bool)
        row = self.part_to_row.get(str(part_id))
        if row is None:
            return out
        count = min(out.size, self.pose_valid.shape[1])
        out[:count] = self.pose_valid[row, :count] & self.pose_mask[row, :count]
        return out
