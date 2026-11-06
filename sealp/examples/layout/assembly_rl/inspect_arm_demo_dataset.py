#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate and summarize an arm demonstration dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    index_path = dataset_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    arm_counts = Counter()
    pose_counts = Counter()
    part_counts = Counter()
    invalid_expert_actions = []
    transition_total = 0
    scores = []

    for row in index.get("episodes", []):
        episode_path = dataset_dir / row["file"]
        with np.load(episode_path, allow_pickle=False) as npz:
            actions = np.asarray(npz["joint_action"], dtype=np.int64)
            packed = np.asarray(npz["action_mask_packed"], dtype=np.uint8)
            action_n = int(np.asarray(npz["action_mask_n"]).item())
            arms = np.asarray(npz["arm_id"], dtype=np.int64)
            poses = np.asarray(npz["pose_id"], dtype=np.int64)
            parts = np.asarray(npz["part_id"]).astype(str)

            if packed.ndim != 2:
                raise ValueError(
                    f"{episode_path}: packed mask must be [T,N], "
                    f"got {packed.shape}"
                )
            if actions.shape[0] != packed.shape[0]:
                raise ValueError(
                    f"{episode_path}: action/mask time mismatch"
                )

            masks = np.unpackbits(
                packed,
                axis=1,
                count=action_n,
                bitorder="little",
            ).astype(bool)

            for step, action in enumerate(actions):
                if not 0 <= int(action) < action_n:
                    invalid_expert_actions.append(
                        (str(episode_path), step, int(action), "out_of_range")
                    )
                elif not bool(masks[step, int(action)]):
                    invalid_expert_actions.append(
                        (str(episode_path), step, int(action), "masked")
                    )

            arm_counts.update(int(v) for v in arms.tolist())
            pose_counts.update(int(v) for v in poses.tolist())
            part_counts.update(parts.tolist())
            transition_total += int(actions.size)
            scores.append(float(np.asarray(npz["layout_score"]).item()))

    print("=" * 72)
    print(f"dataset             : {dataset_dir}")
    print(f"task                : {index.get('task_name')}")
    print(f"region              : {index.get('region_id')}")
    print(f"successful episodes : {len(index.get('episodes', []))}")
    print(f"attempted episodes  : {index.get('attempted_episode_count')}")
    print(f"discarded failures  : {len(index.get('failures', []))}")
    print(f"transitions         : {transition_total}")
    print(f"arm counts          : left={arm_counts[0]} right={arm_counts[1]}")
    print(f"pose counts         : {dict(sorted(pose_counts.items()))}")
    print(f"part counts         : {dict(part_counts)}")
    print(
        "mean layout score   : "
        + (
            f"{float(np.mean(scores)):.6f}"
            if scores
            else "N/A"
        )
    )
    print(f"invalid actions     : {len(invalid_expert_actions)}")
    print("=" * 72)

    if invalid_expert_actions:
        for item in invalid_expert_actions[:20]:
            print("[INVALID]", item)
        raise RuntimeError(
            "Expert actions failed action-mask validation"
        )

    expected_transitions = (
        len(index.get("episodes", []))
        * len(index.get("decision_parts", []))
    )
    if transition_total != expected_transitions:
        raise RuntimeError(
            f"transition total {transition_total}, "
            f"expected {expected_transitions}"
        )

    print("[OK] dataset validation passed")


if __name__ == "__main__":
    main()
