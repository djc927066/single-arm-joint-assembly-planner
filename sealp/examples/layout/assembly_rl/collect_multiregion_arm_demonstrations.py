#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect strict-arm demonstrations from multiple assembly regions.

The collector can seed a new dataset from an existing single-region dataset,
then add unique successful episodes from new regions. Successful trajectories
may be filtered by a minimum final layout score.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from .arm_conditioned_env import ArmConditionedAssemblyEnv
from .assembly_layout_env import (
    AssemblyLayoutEnv,
    REGION_MODE_FIXED,
    build_assembly_validator,
)
from .collect_arm_demonstrations import (
    _distance_map,
    _jsonable,
    _score_actions,
    _select_action,
    _snapshot_observation,
    _stack_transitions,
    _write_index,
)


def _trajectory_signature(
    region_id: str,
    actions: List[int],
) -> str:
    return (
        str(region_id)
        + "|"
        + ",".join(str(int(v)) for v in actions)
    )


def _copy_base_dataset(
    base_dataset: Path,
    output_dir: Path,
    index: Dict[str, Any],
    signatures: Set[str],
) -> None:
    base_index_path = base_dataset / "index.json"
    if not base_index_path.is_file():
        raise FileNotFoundError(base_index_path)
    base_index = json.loads(
        base_index_path.read_text(encoding="utf-8")
    )

    if base_index.get("task_name") != index["task_name"]:
        raise ValueError("base dataset task does not match")
    if list(base_index.get("decision_parts", [])) != index["decision_parts"]:
        raise ValueError("base dataset decision parts do not match")

    episodes_dir = output_dir / "episodes"
    for row in base_index.get("episodes", []):
        source_path = base_dataset / row["file"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        with np.load(source_path, allow_pickle=False) as npz:
            actions = [
                int(v)
                for v in np.asarray(npz["joint_action"]).tolist()
            ]
        episode_id = len(index["episodes"])
        region_id = str(
            row.get("region_id")
            or base_index.get("region_id")
            or "unknown"
        )
        signature = _trajectory_signature(region_id, actions)
        if signature in signatures:
            continue

        filename = (
            f"episode_{episode_id:05d}_{region_id}_base.npz"
        )
        destination = episodes_dir / filename
        shutil.copy2(source_path, destination)

        new_row = dict(row)
        new_row.update(
            {
                "episode_id": episode_id,
                "file": f"episodes/{filename}",
                "region_id": region_id,
                "source_dataset": str(base_dataset),
                "trajectory_signature": signature,
            }
        )
        index["episodes"].append(new_row)
        index["saved_success_count"] += 1
        index["saved_transition_count"] += int(
            row.get("transition_count", len(actions))
        )
        signatures.add(signature)


def _build_env(args: argparse.Namespace, region_id: str, seed: int):
    validator, task, prepared = build_assembly_validator(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_poses=args.max_poses,
        force_first_at_region_center=True,
    )
    base_env = AssemblyLayoutEnv(
        validator,
        task,
        resolution=args.resolution,
        max_parts=args.max_parts,
        max_poses=args.max_poses,
        region_mode=REGION_MODE_FIXED,
        fixed_region_id=region_id,
        run_l2=True,
        seed=seed,
        prepared_asmdef=prepared,
        grasp_hint_path=args.grasp_hint,
        grasp_reward_weight=args.grasp_reward_weight,
        mask_invalid_grasp_poses=False,
        ik_hint_path=args.ik_hint,
        ik_reward_weight=args.ik_reward_weight,
        mask_zero_ik_actions=False,
        ik_mask_threshold=0.0,
    )
    env = ArmConditionedAssemblyEnv(
        base_env,
        ik_hint_path=args.ik_hint,
        hard_mask_zero_arm_ik=True,
        arm_ik_threshold=0.0,
    )
    description = env.compact_description()
    if description.get("arm_conditioned_stage") != "B-strict-l2-arm":
        raise RuntimeError(
            "multi-region collection requires Stage-B strict-arm env"
        )
    if not env.arm_hints.has_region(region_id):
        raise RuntimeError(
            f"IK cache does not contain requested region {region_id}"
        )
    return env, base_env, task, description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--grasp-hint", required=True)
    parser.add_argument("--ik-hint", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-dataset")
    parser.add_argument(
        "--target-successes-per-region",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-attempted-per-region",
        type=int,
        default=7,
    )
    parser.add_argument("--min-layout-score", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--proposal-top-k", type=int, default=128)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--w-grasp", type=float, default=0.05)
    parser.add_argument("--w-ik", type=float, default=0.10)
    parser.add_argument("--w-distance", type=float, default=0.05)
    parser.add_argument("--grasp-reward-weight", type=float, default=0.05)
    parser.add_argument("--ik-reward-weight", type=float, default=0.10)
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--max-parts", type=int, default=12)
    parser.add_argument("--max-poses", type=int, default=16)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=[
            "mesh",
            "env_only",
            "none",
            "staging_aware",
            "executor_match",
        ],
    )
    parser.add_argument("--max-attempts-per-part", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_successes_per_region <= 0:
        raise ValueError(
            "--target-successes-per-region must be positive"
        )
    if (
        args.max_attempted_per_region
        < args.target_successes_per_region
    ):
        raise ValueError(
            "--max-attempted-per-region must be >= target"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory must be empty or absent: {output_dir}"
        )
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"

    probe_env, probe_base, task, probe_description = _build_env(
        args,
        args.regions[0],
        args.seed,
    )
    probe_env.close()

    index: Dict[str, Any] = {
        "format_version": "2026-07-17-arm-bc-multiregion-v1",
        "dataset_type": (
            "successful_strict_arm_multiregion_demonstrations"
        ),
        "task_name": task.name,
        "asmdef": str(Path(args.asmdef).resolve()),
        "config": str(Path(args.config).resolve()),
        "grasp_hint": str(Path(args.grasp_hint).resolve()),
        "ik_hint": str(Path(args.ik_hint).resolve()),
        "region_id": "multi",
        "regions": list(args.regions),
        "part_order": list(task.part_order),
        "decision_parts": list(task.decision_parts),
        "environment_template": probe_description,
        "collection": {
            "target_successes_per_region": (
                args.target_successes_per_region
            ),
            "max_attempted_per_region": (
                args.max_attempted_per_region
            ),
            "min_layout_score": args.min_layout_score,
            "seed": args.seed,
            "proposal_top_k": args.proposal_top_k,
            "sample_top_k": args.sample_top_k,
            "temperature": args.temperature,
            "weights": {
                "grasp": args.w_grasp,
                "ik": args.w_ik,
                "distance": args.w_distance,
            },
        },
        "attempted_episode_count": 0,
        "attempted_by_region": {},
        "saved_success_count": 0,
        "saved_transition_count": 0,
        "saved_by_region": {},
        "episodes": [],
        "failures": [],
        "filtered_successes": [],
        "duplicates": [],
    }

    signatures: Set[str] = set()
    if args.base_dataset:
        _copy_base_dataset(
            Path(args.base_dataset).expanduser().resolve(),
            output_dir,
            index,
            signatures,
        )

    _write_index(index_path, index)
    start_time = time.perf_counter()
    incomplete_regions: List[str] = []

    for region_index, region_id in enumerate(args.regions):
        region_seed = args.seed + region_index * 10000
        rng = np.random.default_rng(region_seed)
        env, base_env, _, description = _build_env(
            args,
            region_id,
            region_seed,
        )
        saved_for_region = 0
        attempted_for_region = 0

        try:
            for attempt in range(args.max_attempted_per_region):
                if saved_for_region >= args.target_successes_per_region:
                    break

                episode_seed = region_seed + attempt
                observation, reset_info = env.reset(
                    seed=episode_seed,
                    options={"region_id": region_id},
                )
                distance_map = _distance_map(base_env)
                rejected_spatial_actions: Set[int] = set()
                attempts_for_part = 0
                terminated = truncated = False
                transitions: List[Dict[str, Any]] = []
                final_info = dict(reset_info)

                print(
                    f"[multi-reset] region={region_id} "
                    f"attempt={attempt:03d} "
                    f"saved={saved_for_region}/"
                    f"{args.target_successes_per_region} "
                    f"valid={int(np.asarray(observation['action_mask']).sum())}"
                )

                while not (terminated or truncated):
                    state = base_env.state
                    if state is None or state.current_part is None:
                        raise RuntimeError("missing current part")
                    part_before = str(state.current_part)
                    observation_before = observation

                    scores = _score_actions(
                        observation_before,
                        distance_map,
                        w_grasp=args.w_grasp,
                        w_ik=args.w_ik,
                        w_distance=args.w_distance,
                    )
                    action = _select_action(
                        rng,
                        observation_before,
                        scores,
                        env,
                        proposal_top_k=args.proposal_top_k,
                        sample_top_k=args.sample_top_k,
                        temperature=args.temperature,
                        rejected_spatial_actions=rejected_spatial_actions,
                    )
                    arm_id, pose_id, row, col = env.codec.unflatten(
                        action
                    )
                    observation, reward, terminated, truncated, final_info = (
                        env.step(action)
                    )
                    accepted = bool(
                        final_info["arm_action_accepted"]
                    )

                    if accepted:
                        transitions.append(
                            {
                                "observation": _snapshot_observation(
                                    observation_before
                                ),
                                "joint_action": int(action),
                                "arm_id": int(arm_id),
                                "pose_id": int(pose_id),
                                "row": int(row),
                                "col": int(col),
                                "reward": float(reward),
                                "arm_ik_hint": float(
                                    final_info["arm_ik_hint"]
                                ),
                                "part_id": part_before,
                            }
                        )
                        rejected_spatial_actions.clear()
                        attempts_for_part = 0
                        if not (terminated or truncated):
                            distance_map = _distance_map(base_env)
                    else:
                        rejected_spatial_actions.add(
                            env.codec.spatial_action(action)
                        )
                        attempts_for_part += 1
                        if (
                            attempts_for_part
                            >= args.max_attempts_per_part
                        ):
                            raise RuntimeError(
                                f"{part_before} exceeded exact attempts"
                            )

                    print(
                        f"[multi-step] region={region_id} "
                        f"attempt={attempt:03d} "
                        f"part={part_before:16s} "
                        f"arm={final_info['arm']:3s} "
                        f"pose={pose_id:02d} row={row:02d} col={col:02d} "
                        f"accepted={accepted}"
                    )

                attempted_for_region += 1
                index["attempted_episode_count"] += 1

                l2_pass = bool(final_info.get("l2_pass", False))
                strict = bool(
                    final_info.get(
                        "l2_selected_arm_enforced",
                        False,
                    )
                )
                arm_match = final_info.get("l2_arm_choice_match")
                score = float(
                    final_info.get("layout_score", -1.0)
                )
                actions = [
                    int(row["joint_action"])
                    for row in transitions
                ]
                signature = _trajectory_signature(
                    region_id,
                    actions,
                )

                if not (l2_pass and strict and arm_match is True):
                    index["failures"].append(
                        {
                            "region_id": region_id,
                            "attempt": attempt,
                            "seed": episode_seed,
                            "fail_part": final_info.get("fail_part"),
                            "fail_reason": final_info.get(
                                "fail_reason"
                            ),
                            "selected_arms": final_info.get(
                                "selected_arms"
                            ),
                            "l2_arm_choice": final_info.get(
                                "l2_arm_choice"
                            ),
                        }
                    )
                    print(
                        f"[multi-discard] region={region_id} "
                        f"attempt={attempt:03d} L2 failed"
                    )
                elif score < args.min_layout_score:
                    index["filtered_successes"].append(
                        {
                            "region_id": region_id,
                            "attempt": attempt,
                            "seed": episode_seed,
                            "layout_score": score,
                            "reason": "below_min_layout_score",
                        }
                    )
                    print(
                        f"[multi-filter] region={region_id} "
                        f"attempt={attempt:03d} score={score:.4f}"
                    )
                elif signature in signatures:
                    index["duplicates"].append(
                        {
                            "region_id": region_id,
                            "attempt": attempt,
                            "seed": episode_seed,
                            "layout_score": score,
                            "trajectory_signature": signature,
                        }
                    )
                    print(
                        f"[multi-duplicate] region={region_id} "
                        f"attempt={attempt:03d}"
                    )
                else:
                    if len(transitions) != env.n_decision_parts:
                        raise RuntimeError(
                            "successful episode has wrong transition count"
                        )
                    episode_id = len(index["episodes"])
                    filename = (
                        f"episode_{episode_id:05d}_{region_id}.npz"
                    )
                    arrays = _stack_transitions(
                        transitions,
                        layout_score=score,
                        selected_arms=final_info["selected_arms"],
                    )
                    arrays["region_id"] = np.asarray(
                        region_id,
                        dtype="<U64",
                    )
                    np.savez_compressed(
                        episodes_dir / filename,
                        **arrays,
                    )
                    index["episodes"].append(
                        {
                            "episode_id": episode_id,
                            "region_id": region_id,
                            "source_attempt": attempt,
                            "seed": episode_seed,
                            "file": f"episodes/{filename}",
                            "transition_count": len(transitions),
                            "layout_score": score,
                            "selected_arms": dict(
                                final_info["selected_arms"]
                            ),
                            "l2_arm_choice": dict(
                                final_info.get(
                                    "l2_arm_choice",
                                    {},
                                )
                            ),
                            "trajectory_signature": signature,
                        }
                    )
                    index["saved_success_count"] += 1
                    index["saved_transition_count"] += len(
                        transitions
                    )
                    saved_for_region += 1
                    signatures.add(signature)
                    print(
                        f"[multi-save] region={region_id} "
                        f"file={filename} score={score:.4f} "
                        f"saved={saved_for_region}/"
                        f"{args.target_successes_per_region}"
                    )

                index["attempted_by_region"][region_id] = (
                    attempted_for_region
                )
                index["saved_by_region"][region_id] = (
                    saved_for_region
                )
                index["elapsed_seconds"] = (
                    time.perf_counter() - start_time
                )
                _write_index(index_path, index)
        finally:
            env.close()

        if saved_for_region < args.target_successes_per_region:
            incomplete_regions.append(region_id)

    index["completed_target"] = not incomplete_regions
    index["incomplete_regions"] = incomplete_regions
    index["elapsed_seconds"] = time.perf_counter() - start_time
    _write_index(index_path, index)

    print("=" * 76)
    print(f"dataset             : {output_dir}")
    print(f"total successes     : {index['saved_success_count']}")
    print(f"total transitions   : {index['saved_transition_count']}")
    print(f"saved by region     : {index['saved_by_region']}")
    print(f"incomplete regions  : {incomplete_regions}")
    print("=" * 76)

    if incomplete_regions:
        raise RuntimeError(
            "Some regions did not reach the requested unique, "
            "score-filtered success count. Inspect index.json before "
            "raising max attempts."
        )


if __name__ == "__main__":
    main()
