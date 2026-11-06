#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect a multi-region strict-arm demonstration dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    index = json.loads(
        (dataset_dir / "index.json").read_text(encoding="utf-8")
    )

    region_episodes = Counter()
    region_transitions = Counter()
    region_scores = defaultdict(list)
    arm_counts = Counter()
    pose_counts = Counter()
    signatures = set()
    duplicate_signatures = []
    invalid_actions = []
    total_transitions = 0

    for row in index.get("episodes", []):
        region = str(row.get("region_id", "unknown"))
        signature = str(row.get("trajectory_signature", ""))
        if signature:
            if signature in signatures:
                duplicate_signatures.append(signature)
            signatures.add(signature)

        path = dataset_dir / row["file"]
        with np.load(path, allow_pickle=False) as npz:
            actions = np.asarray(npz["joint_action"], dtype=np.int64)
            packed = np.asarray(
                npz["action_mask_packed"],
                dtype=np.uint8,
            )
            action_n = int(np.asarray(npz["action_mask_n"]).item())
            masks = np.unpackbits(
                packed,
                axis=1,
                count=action_n,
                bitorder="little",
            ).astype(bool)

            for step, action in enumerate(actions):
                if not 0 <= int(action) < action_n:
                    invalid_actions.append(
                        (row["file"], step, int(action), "range")
                    )
                elif not bool(masks[step, int(action)]):
                    invalid_actions.append(
                        (row["file"], step, int(action), "masked")
                    )

            transition_count = int(actions.size)
            region_episodes[region] += 1
            region_transitions[region] += transition_count
            region_scores[region].append(
                float(np.asarray(npz["layout_score"]).item())
            )
            arm_counts.update(
                int(v)
                for v in np.asarray(npz["arm_id"]).tolist()
            )
            pose_counts.update(
                int(v)
                for v in np.asarray(npz["pose_id"]).tolist()
            )
            total_transitions += transition_count

    print("=" * 76)
    print(f"dataset            : {dataset_dir}")
    print(f"episodes           : {len(index.get('episodes', []))}")
    print(f"transitions        : {total_transitions}")
    print(f"regions            : {dict(region_episodes)}")
    print(f"region transitions : {dict(region_transitions)}")
    print(f"arms               : left={arm_counts[0]} right={arm_counts[1]}")
    print(f"poses              : {dict(sorted(pose_counts.items()))}")
    print(f"failures           : {len(index.get('failures', []))}")
    print(
        f"score-filtered     : "
        f"{len(index.get('filtered_successes', []))}"
    )
    print(f"duplicates skipped : {len(index.get('duplicates', []))}")
    print(f"duplicate saved    : {len(duplicate_signatures)}")
    print(f"invalid actions    : {len(invalid_actions)}")
    for region in sorted(region_scores):
        scores = np.asarray(region_scores[region], dtype=float)
        print(
            f"score[{region}]       : "
            f"n={scores.size} mean={scores.mean():.4f} "
            f"min={scores.min():.4f} max={scores.max():.4f}"
        )
    print("=" * 76)

    if invalid_actions:
        for row in invalid_actions[:20]:
            print("[INVALID]", row)
        raise RuntimeError("invalid expert actions found")
    if duplicate_signatures:
        raise RuntimeError("duplicate saved trajectories found")

    expected = (
        len(index.get("episodes", []))
        * len(index.get("decision_parts", []))
    )
    if total_transitions != expected:
        raise RuntimeError(
            f"transition count {total_transitions}, expected {expected}"
        )
    print("[OK] multi-region dataset validation passed")


if __name__ == "__main__":
    main()
