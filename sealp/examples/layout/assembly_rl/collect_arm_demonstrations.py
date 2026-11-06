#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect successful strict-arm demonstrations for behavior cloning.

Each successful episode is stored as one compressed NPZ file. Failed terminal
episodes are recorded in index.json but are not used as expert positives.

The dataset keeps task-agnostic numeric observations:
- occupancy;
- part features and graph adjacencies;
- pose features;
- grasp hint;
- per-arm IK hint;
- packed 41,472-dimensional action mask;
- joint expert action (arm, pose, row, col).

PointNet mesh embeddings can be attached later using the saved task/part
metadata; no part-name one-hot feature is introduced here.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

import numpy as np

from .arm_conditioned_env import ArmConditionedAssemblyEnv
from .assembly_layout_env import (
    AssemblyLayoutEnv,
    REGION_MODE_FIXED,
    build_assembly_validator,
)


OBS_FLOAT32 = (
    "part_features",
    "pose_features",
)
OBS_FLOAT16 = (
    "occupancy",
    "grasp_hint",
    "ik_hint_by_arm",
)
OBS_INT8 = (
    "part_mask",
    "decision_mask",
    "current_part_mask",
    "parent_adjacency",
    "dependency_adjacency",
    "symmetry_adjacency",
    "pose_mask",
)


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


def _softmax_sample(
    rng: np.random.Generator,
    indices: np.ndarray,
    scores: np.ndarray,
    temperature: float,
) -> int:
    if indices.size == 0:
        raise RuntimeError("cannot sample from an empty candidate set")
    if temperature <= 1e-9 or indices.size == 1:
        return int(indices[int(np.argmax(scores))])
    scaled = (scores - float(np.max(scores))) / float(temperature)
    scaled = np.clip(scaled, -60.0, 0.0)
    probs = np.exp(scaled)
    total = float(probs.sum())
    if not math.isfinite(total) or total <= 0.0:
        return int(indices[int(np.argmax(scores))])
    probs /= total
    return int(rng.choice(indices, p=probs))


def _distance_map(base_env) -> np.ndarray:
    state = base_env.state
    if state is None:
        raise RuntimeError("environment has not been reset")
    center = np.asarray(state.region_center, dtype=float)[:2]
    result = np.zeros(
        (base_env.grid_height, base_env.grid_width),
        dtype=np.float32,
    )
    for row in range(base_env.grid_height):
        for col in range(base_env.grid_width):
            xy = np.asarray(
                state.workspace.spec.cell_center(row, col),
                dtype=float,
            )
            distance = float(np.linalg.norm(xy - center))
            result[row, col] = float(np.exp(-distance / 0.55))
    return result


def _score_actions(
    observation: Mapping[str, Any],
    distance_map: np.ndarray,
    *,
    w_grasp: float,
    w_ik: float,
    w_distance: float,
) -> np.ndarray:
    grasp = np.asarray(observation["grasp_hint"], dtype=np.float32)
    ik = np.asarray(observation["ik_hint_by_arm"], dtype=np.float32)
    return (
        float(w_grasp) * grasp[None, :, None, None]
        + float(w_ik) * ik
        + float(w_distance) * distance_map[None, None, :, :]
    ).reshape(-1)


def _select_action(
    rng: np.random.Generator,
    observation: Mapping[str, Any],
    scores: np.ndarray,
    env: ArmConditionedAssemblyEnv,
    *,
    proposal_top_k: int,
    sample_top_k: int,
    temperature: float,
    rejected_spatial_actions: Set[int],
) -> int:
    mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
    valid = np.flatnonzero(mask)

    if rejected_spatial_actions:
        valid = np.asarray(
            [
                int(action)
                for action in valid
                if env.codec.spatial_action(int(action))
                not in rejected_spatial_actions
            ],
            dtype=np.int64,
        )
    if valid.size == 0:
        raise RuntimeError("no valid joint action remains")

    top_k = min(max(1, int(proposal_top_k)), valid.size)
    valid_scores = scores[valid]
    if top_k < valid.size:
        local = np.argpartition(valid_scores, -top_k)[-top_k:]
        candidates = valid[local]
    else:
        candidates = valid

    candidate_scores = scores[candidates]
    order = np.argsort(candidate_scores)[::-1]
    candidates = candidates[order]
    candidate_scores = candidate_scores[order]

    sample_k = min(max(1, int(sample_top_k)), candidates.size)
    return _softmax_sample(
        rng,
        candidates[:sample_k],
        candidate_scores[:sample_k],
        float(temperature),
    )


def _snapshot_observation(observation: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    snapshot: Dict[str, np.ndarray] = {}

    for key in OBS_FLOAT32:
        snapshot[key] = np.asarray(observation[key], dtype=np.float32).copy()
    for key in OBS_FLOAT16:
        snapshot[key] = np.asarray(observation[key], dtype=np.float16).copy()
    for key in OBS_INT8:
        snapshot[key] = np.asarray(observation[key], dtype=np.int8).copy()

    action_mask = np.asarray(
        observation["action_mask"],
        dtype=np.uint8,
    ).reshape(-1)
    snapshot["action_mask_packed"] = np.packbits(
        action_mask,
        bitorder="little",
    )
    snapshot["action_mask_n"] = np.asarray(
        action_mask.size,
        dtype=np.int32,
    )
    return snapshot


def _stack_transitions(
    transitions: List[Dict[str, Any]],
    *,
    layout_score: float,
    selected_arms: Mapping[str, str],
) -> Dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("cannot save an empty successful episode")

    result: Dict[str, np.ndarray] = {}
    observation_keys = list(transitions[0]["observation"].keys())

    for key in observation_keys:
        if key == "action_mask_n":
            result[key] = np.asarray(
                transitions[0]["observation"][key],
                dtype=np.int32,
            )
        else:
            result[key] = np.stack(
                [row["observation"][key] for row in transitions],
                axis=0,
            )

    result["joint_action"] = np.asarray(
        [row["joint_action"] for row in transitions],
        dtype=np.int32,
    )
    result["arm_id"] = np.asarray(
        [row["arm_id"] for row in transitions],
        dtype=np.int8,
    )
    result["pose_id"] = np.asarray(
        [row["pose_id"] for row in transitions],
        dtype=np.int16,
    )
    result["row"] = np.asarray(
        [row["row"] for row in transitions],
        dtype=np.int16,
    )
    result["col"] = np.asarray(
        [row["col"] for row in transitions],
        dtype=np.int16,
    )
    result["reward"] = np.asarray(
        [row["reward"] for row in transitions],
        dtype=np.float32,
    )
    result["arm_ik_hint"] = np.asarray(
        [row["arm_ik_hint"] for row in transitions],
        dtype=np.float32,
    )
    result["part_id"] = np.asarray(
        [row["part_id"] for row in transitions],
        dtype="<U128",
    )
    result["layout_score"] = np.asarray(layout_score, dtype=np.float32)
    result["selected_arms_json"] = np.asarray(
        json.dumps(dict(selected_arms), ensure_ascii=False),
        dtype="<U2048",
    )
    return result


def _write_index(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--grasp-hint", required=True)
    parser.add_argument("--ik-hint", required=True)
    parser.add_argument("--region-id", default="r1_c1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-successes", type=int, default=5)
    parser.add_argument("--max-attempted-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--proposal-top-k", type=int, default=128)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.02)
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
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--max-attempts-per-part", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_successes <= 0:
        raise ValueError("--target-successes must be positive")
    if args.max_attempted_episodes < args.target_successes:
        raise ValueError(
            "--max-attempted-episodes must be >= --target-successes"
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"

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
        fixed_region_id=args.region_id,
        run_l2=True,
        seed=args.seed,
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
            "Demonstrations require the Stage-B strict-arm environment"
        )
    if not description.get("l2_selected_arm_enforced", False):
        raise RuntimeError("strict L2 arm enforcement is not enabled")

    index: Dict[str, Any] = {
        "format_version": "2026-07-17-arm-bc-v1",
        "dataset_type": "successful_strict_arm_demonstrations",
        "task_name": task.name,
        "asmdef": str(Path(args.asmdef).resolve()),
        "config": str(Path(args.config).resolve()),
        "grasp_hint": str(Path(args.grasp_hint).resolve()),
        "ik_hint": str(Path(args.ik_hint).resolve()),
        "region_id": args.region_id,
        "part_order": list(task.part_order),
        "decision_parts": list(task.decision_parts),
        "environment": description,
        "collection": {
            "target_successes": args.target_successes,
            "max_attempted_episodes": args.max_attempted_episodes,
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
        "saved_success_count": 0,
        "saved_transition_count": 0,
        "episodes": [],
        "failures": [],
    }
    _write_index(index_path, index)

    rng = np.random.default_rng(args.seed)
    start_time = time.perf_counter()

    try:
        for attempt in range(args.max_attempted_episodes):
            if index["saved_success_count"] >= args.target_successes:
                break

            observation, reset_info = env.reset(
                seed=args.seed + attempt,
                options={"region_id": args.region_id},
            )
            terminated = truncated = False
            rejected_spatial_actions: Set[int] = set()
            attempts_for_current_part = 0
            distance_map = _distance_map(base_env)
            transitions: List[Dict[str, Any]] = []
            final_info = dict(reset_info)

            print(
                f"[demo-reset] attempt={attempt:03d} "
                f"saved={index['saved_success_count']}/{args.target_successes} "
                f"part={reset_info['current_part']} "
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
                arm_id, pose_id, row, col = env.codec.unflatten(action)

                observation, reward, terminated, truncated, final_info = env.step(
                    action
                )
                accepted = bool(final_info["arm_action_accepted"])

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
                    attempts_for_current_part = 0
                    if not (terminated or truncated):
                        distance_map = _distance_map(base_env)
                else:
                    rejected_spatial_actions.add(
                        env.codec.spatial_action(action)
                    )
                    attempts_for_current_part += 1
                    if attempts_for_current_part >= args.max_attempts_per_part:
                        raise RuntimeError(
                            f"{part_before} exceeded "
                            f"{args.max_attempts_per_part} exact attempts"
                        )

                print(
                    f"[demo-step] attempt={attempt:03d} "
                    f"part={part_before:16s} "
                    f"arm={final_info['arm']:3s} pose={pose_id:02d} "
                    f"row={row:02d} col={col:02d} "
                    f"accepted={accepted} reward={float(reward):+.4f}"
                )

            index["attempted_episode_count"] += 1
            l2_pass = bool(final_info.get("l2_pass", False))
            strict = bool(final_info.get("l2_selected_arm_enforced", False))
            arm_match = final_info.get("l2_arm_choice_match")

            if l2_pass and strict and arm_match is True:
                if len(transitions) != env.n_decision_parts:
                    raise RuntimeError(
                        f"successful episode has {len(transitions)} "
                        f"transitions, expected {env.n_decision_parts}"
                    )

                saved_id = int(index["saved_success_count"])
                filename = f"episode_{saved_id:05d}.npz"
                episode_path = episodes_dir / filename
                arrays = _stack_transitions(
                    transitions,
                    layout_score=float(final_info["layout_score"]),
                    selected_arms=final_info["selected_arms"],
                )
                np.savez_compressed(episode_path, **arrays)

                index["saved_success_count"] += 1
                index["saved_transition_count"] += len(transitions)
                index["episodes"].append(
                    {
                        "episode_id": saved_id,
                        "source_attempt": attempt,
                        "seed": args.seed + attempt,
                        "file": f"episodes/{filename}",
                        "transition_count": len(transitions),
                        "layout_score": float(final_info["layout_score"]),
                        "selected_arms": dict(
                            final_info["selected_arms"]
                        ),
                        "l2_arm_choice": dict(
                            final_info.get("l2_arm_choice", {})
                        ),
                    }
                )
                print(
                    f"[demo-save] {filename} "
                    f"score={float(final_info['layout_score']):.4f} "
                    f"saved={index['saved_success_count']}/"
                    f"{args.target_successes}"
                )
            else:
                failure = {
                    "source_attempt": attempt,
                    "seed": args.seed + attempt,
                    "l2_pass": l2_pass,
                    "strict": strict,
                    "arm_match": arm_match,
                    "fail_part": final_info.get("fail_part"),
                    "fail_reason": final_info.get("fail_reason"),
                    "l2_arm_choice": final_info.get("l2_arm_choice"),
                    "selected_arms": final_info.get("selected_arms"),
                }
                index["failures"].append(_jsonable(failure))
                print(
                    f"[demo-discard] attempt={attempt:03d} "
                    f"fail_part={failure['fail_part']} "
                    f"reason={failure['fail_reason']}"
                )

            index["elapsed_seconds"] = time.perf_counter() - start_time
            _write_index(index_path, index)
    finally:
        env.close()

    index["completed_target"] = (
        index["saved_success_count"] >= args.target_successes
    )
    index["elapsed_seconds"] = time.perf_counter() - start_time
    _write_index(index_path, index)

    print("=" * 72)
    print(
        f"saved_successes={index['saved_success_count']}/"
        f"{args.target_successes}"
    )
    print(f"attempted={index['attempted_episode_count']}")
    print(f"transitions={index['saved_transition_count']}")
    print(f"dataset={output_dir}")
    print("=" * 72)

    if not index["completed_target"]:
        raise RuntimeError(
            "Target success count was not reached. Inspect index.json "
            "failures before increasing max attempted episodes."
        )


if __name__ == "__main__":
    main()
