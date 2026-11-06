#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke-test strict arm-conditioned L2 evaluation.

Use one episode first, then three.  The selected joint-action arm is
strictly enforced inside the existing L2 evaluator; automatic fallback to the
other arm is disabled.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from .arm_conditioned_env import ArmConditionedAssemblyEnv
from .assembly_layout_env import (
    AssemblyLayoutEnv,
    REGION_MODE_FIXED,
    build_assembly_validator,
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
    if indices.size <= 0:
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


def _score_joint_actions(
    observation: Mapping[str, Any],
    *,
    w_grasp: float,
    w_ik: float,
    w_distance: float,
    distance_map: np.ndarray,
) -> np.ndarray:
    hints = np.asarray(observation["ik_hint_by_arm"], dtype=np.float32)
    grasp = np.asarray(observation["grasp_hint"], dtype=np.float32)
    if grasp.ndim != 1:
        raise ValueError(f"grasp_hint must be 1-D, got {grasp.shape}")
    return (
        float(w_grasp) * grasp[None, :, None, None]
        + float(w_ik) * hints
        + float(w_distance) * distance_map[None, None, :, :]
    ).reshape(-1)


def _select_action(
    rng: np.random.Generator,
    observation: Mapping[str, Any],
    scores: np.ndarray,
    *,
    proposal_top_k: int,
    sample_top_k: int,
    temperature: float,
    rejected_spatial_actions: Set[int],
    env: ArmConditionedAssemblyEnv,
) -> int:
    mask = np.asarray(observation["action_mask"], dtype=bool).reshape(-1)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise RuntimeError("joint action mask contains no valid action")

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
        raise RuntimeError("all valid spatial candidates were exactly rejected")

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
        temperature,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--grasp-hint", required=True)
    parser.add_argument("--ik-hint", required=True)
    parser.add_argument("--region-id", default="r1_c1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--allow-zero-arm-ik",
        action="store_true",
        help="Do not hard-mask arm-specific IK hint <= threshold.",
    )
    parser.add_argument("--arm-ik-threshold", type=float, default=0.0)
    parser.add_argument("--max-attempts-per-part", type=int, default=128)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        hard_mask_zero_arm_ik=not args.allow_zero_arm_ik,
        arm_ik_threshold=args.arm_ik_threshold,
    )

    print("=" * 76)
    print("Arm-conditioned Stage-B strict-L2 test")
    print("=" * 76)
    print(json.dumps(env.compact_description(), ensure_ascii=False, indent=2))
    description = env.compact_description()
    if not description.get("l2_selected_arm_enforced", False):
        raise RuntimeError("Stage-B environment did not enable strict L2 arms")
    print(
        "[STRICT] L2 will use exactly the arm selected by each joint action; "
        "no automatic arm fallback is allowed."
    )

    rng = np.random.default_rng(args.seed)
    episodes: List[Dict[str, Any]] = []
    pass_count = 0

    try:
        for episode in range(args.episodes):
            observation, info = env.reset(
                seed=args.seed + episode,
                options={"region_id": args.region_id},
            )
            terminated = truncated = False
            rejected_spatial_actions: Set[int] = set()
            attempts_for_current_part = 0
            distance_map = _distance_map(base_env)
            final_info = dict(info)

            print(
                f"[reset-arm] episode={episode:03d} "
                f"region={info['region_id']} "
                f"part={info['current_part']} "
                f"valid_joint={int(np.asarray(observation['action_mask']).sum())}"
            )

            while not (terminated or truncated):
                state = base_env.state
                if state is None or state.current_part is None:
                    raise RuntimeError("missing current part before terminal state")
                part_before = str(state.current_part)
                step_before = int(state.current_step)

                scores = _score_joint_actions(
                    observation,
                    w_grasp=args.w_grasp,
                    w_ik=args.w_ik,
                    w_distance=args.w_distance,
                    distance_map=distance_map,
                )
                joint_action = _select_action(
                    rng,
                    observation,
                    scores,
                    proposal_top_k=args.proposal_top_k,
                    sample_top_k=args.sample_top_k,
                    temperature=args.temperature,
                    rejected_spatial_actions=rejected_spatial_actions,
                    env=env,
                )
                arm_id, pose_id, row, col = env.codec.unflatten(joint_action)
                observation, reward, terminated, truncated, final_info = env.step(
                    joint_action
                )

                accepted = bool(final_info["arm_action_accepted"])
                if accepted:
                    rejected_spatial_actions.clear()
                    attempts_for_current_part = 0
                    distance_map = (
                        _distance_map(base_env)
                        if not (terminated or truncated)
                        else distance_map
                    )
                else:
                    rejected_spatial_actions.add(
                        env.codec.spatial_action(joint_action)
                    )
                    attempts_for_current_part += 1
                    if attempts_for_current_part >= args.max_attempts_per_part:
                        raise RuntimeError(
                            f"part {part_before} exceeded "
                            f"{args.max_attempts_per_part} exact attempts"
                        )

                print(
                    f"[select-arm] episode={episode:03d} "
                    f"step={step_before + 1:02d}/{env.n_decision_parts:02d} "
                    f"part={part_before:16s} "
                    f"arm={final_info['arm']:3s} "
                    f"pose={pose_id:02d} row={row:02d} col={col:02d} "
                    f"ik_arm={float(final_info['arm_ik_hint']):.3f} "
                    f"accepted={accepted} "
                    f"event={final_info.get('event')} "
                    f"reward={float(reward):+.4f}"
                )

            l2_pass = bool(final_info.get("l2_pass", False))
            pass_count += int(l2_pass)
            episodes.append(
                {
                    "episode": episode,
                    "l2_pass": l2_pass,
                    "layout_score": final_info.get("layout_score"),
                    "fail_part": final_info.get("fail_part"),
                    "fail_reason": final_info.get("fail_reason"),
                    "selected_arms": dict(final_info.get("selected_arms", {})),
                    "arm_annotated_l2_request": final_info.get(
                        "arm_annotated_l2_request"
                    ),
                    "l2_selected_arm_enforced": final_info.get(
                        "l2_selected_arm_enforced"
                    ),
                    "l2_arm_choice": final_info.get("l2_arm_choice"),
                    "l2_arm_choice_match": final_info.get(
                        "l2_arm_choice_match"
                    ),
                    "strict_arm_order_trace": final_info.get(
                        "strict_arm_order_trace"
                    ),
                }
            )
            strict_flag = bool(
                final_info.get("l2_selected_arm_enforced", False)
            )
            arm_match = final_info.get("l2_arm_choice_match")
            if not strict_flag:
                raise RuntimeError(
                    "terminal L2 ran without strict selected-arm enforcement"
                )
            if l2_pass and arm_match is not True:
                raise RuntimeError(
                    "successful L2 did not return the selected arms exactly"
                )
            print(
                f"[episode-arm] episode={episode:03d} "
                f"pass={l2_pass} "
                f"score={final_info.get('layout_score')} "
                f"strict={strict_flag} "
                f"arm_match={arm_match} "
                f"selected={final_info.get('selected_arms')} "
                f"l2={final_info.get('l2_arm_choice')}"
            )
    finally:
        env.close()

    payload = {
        "stage": "B-strict-l2-arm",
        "l2_selected_arm_enforced": True,
        "environment": env.compact_description(),
        "episodes": episodes,
        "statistics": {
            "episode_count": len(episodes),
            "l2_pass_count": pass_count,
            "l2_pass_rate": pass_count / max(1, len(episodes)),
            "strict_enforced_count": sum(
                1
                for row in episodes
                if row.get("l2_selected_arm_enforced") is True
            ),
            "arm_choice_match_count": sum(
                1
                for row in episodes
                if row.get("l2_arm_choice_match") is True
            ),
        },
    }
    print(json.dumps(payload["statistics"], ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2)
        print(f"[OK] saved to: {output_path}")


if __name__ == "__main__":
    main()
