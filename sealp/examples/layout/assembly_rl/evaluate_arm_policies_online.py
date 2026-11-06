#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online strict-L2 comparison of BC, heuristic and random arm policies.

This script evaluates the learned checkpoint in the real sequential
ArmConditionedAssemblyEnv rather than only replaying saved observations.

Policies:
- bc: masked ArmBCPolicy checkpoint;
- grasp_ik_greedy: existing grasp + per-arm IK + distance heuristic;
- random: uniformly random valid joint action.

All policies use:
- the same episode seeds;
- the 41,472-dimensional joint action space;
- arm-specific IK masking;
- exact geometry rejection handling;
- strict Stage-B L2 arm enforcement.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

from .arm_bc_model import ArmBCConfig, ArmBCPolicy
from .arm_conditioned_env import ArmConditionedAssemblyEnv
from .assembly_layout_env import (
    AssemblyLayoutEnv,
    REGION_MODE_FIXED,
    build_assembly_validator,
)


MODEL_FLOAT_KEYS = (
    "occupancy",
    "part_features",
    "part_mask",
    "current_part_mask",
    "parent_adjacency",
    "dependency_adjacency",
    "symmetry_adjacency",
    "pose_features",
    "grasp_hint",
    "ik_hint_by_arm",
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


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(name)


def _distance_map(base_env) -> np.ndarray:
    state = base_env.state
    if state is None:
        raise RuntimeError("environment is not reset")
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


def _mask_rejected_spatial_actions(
    action_mask: np.ndarray,
    rejected_spatial_actions: Set[int],
    *,
    spatial_action_n: int,
) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1).copy()
    if not rejected_spatial_actions:
        return mask

    for spatial_action in rejected_spatial_actions:
        if not 0 <= int(spatial_action) < spatial_action_n:
            raise ValueError(
                f"rejected spatial action out of range: {spatial_action}"
            )
        mask[int(spatial_action)] = False
        second_arm_action = int(spatial_action) + spatial_action_n
        if second_arm_action < mask.size:
            mask[second_arm_action] = False
    return mask


def _observation_to_batch(
    observation: Mapping[str, Any],
    action_mask: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for key in MODEL_FLOAT_KEYS:
        batch[key] = torch.from_numpy(
            np.asarray(observation[key], dtype=np.float32)
        ).unsqueeze(0).to(device)
    batch["action_mask"] = torch.from_numpy(
        np.asarray(action_mask, dtype=np.bool_)
    ).unsqueeze(0).to(device)
    return batch


def _sample_from_logits(
    logits: torch.Tensor,
    rng: np.random.Generator,
    *,
    temperature: float,
    sample_top_k: int,
) -> Tuple[int, float, float]:
    flat = logits.detach().float().cpu().numpy().reshape(-1)
    finite = np.isfinite(flat)
    valid = np.flatnonzero(finite)
    if valid.size == 0:
        raise RuntimeError("BC logits contain no finite valid action")

    order = valid[np.argsort(flat[valid])[::-1]]
    best = int(order[0])
    best_logit = float(flat[best])
    second_logit = (
        float(flat[int(order[1])])
        if order.size > 1
        else best_logit
    )
    margin = best_logit - second_logit

    if temperature <= 1e-9 or order.size == 1:
        return best, best_logit, margin

    top_k = min(max(1, int(sample_top_k)), order.size)
    candidates = order[:top_k]
    candidate_logits = flat[candidates]
    scaled = (
        candidate_logits - float(candidate_logits.max())
    ) / float(temperature)
    scaled = np.clip(scaled, -60.0, 0.0)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    selected = int(rng.choice(candidates, p=probabilities))
    return selected, float(flat[selected]), margin


def _select_bc_action(
    model: ArmBCPolicy,
    observation: Mapping[str, Any],
    action_mask: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    *,
    temperature: float,
    sample_top_k: int,
) -> Tuple[int, Dict[str, float]]:
    batch = _observation_to_batch(
        observation,
        action_mask,
        device,
    )
    model.eval()
    with torch.no_grad():
        logits = model.masked_flat_logits(batch)[0]
    action, selected_logit, margin = _sample_from_logits(
        logits,
        rng,
        temperature=temperature,
        sample_top_k=sample_top_k,
    )
    return action, {
        "selected_logit": selected_logit,
        "top1_top2_margin": margin,
    }


def _softmax_sample(
    rng: np.random.Generator,
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    temperature: float,
) -> int:
    if candidates.size == 0:
        raise RuntimeError("cannot sample an empty candidate set")
    if temperature <= 1e-9 or candidates.size == 1:
        return int(candidates[int(np.argmax(scores))])
    scaled = (scores - float(np.max(scores))) / float(temperature)
    scaled = np.clip(scaled, -60.0, 0.0)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


def _select_heuristic_action(
    observation: Mapping[str, Any],
    action_mask: np.ndarray,
    distance_map: np.ndarray,
    rng: np.random.Generator,
    *,
    w_grasp: float,
    w_ik: float,
    w_distance: float,
    proposal_top_k: int,
    sample_top_k: int,
    temperature: float,
) -> Tuple[int, Dict[str, float]]:
    grasp = np.asarray(
        observation["grasp_hint"],
        dtype=np.float32,
    )
    ik = np.asarray(
        observation["ik_hint_by_arm"],
        dtype=np.float32,
    )
    scores = (
        float(w_grasp) * grasp[None, :, None, None]
        + float(w_ik) * ik
        + float(w_distance) * distance_map[None, None, :, :]
    ).reshape(-1)

    valid = np.flatnonzero(action_mask)
    if valid.size == 0:
        raise RuntimeError("heuristic has no valid action")

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
    action = _softmax_sample(
        rng,
        candidates[:sample_k],
        candidate_scores[:sample_k],
        temperature=temperature,
    )
    return action, {
        "heuristic_score": float(scores[action]),
    }


def _select_random_action(
    action_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, Dict[str, float]]:
    valid = np.flatnonzero(action_mask)
    if valid.size == 0:
        raise RuntimeError("random policy has no valid action")
    return int(rng.choice(valid)), {}


def _load_bc_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ArmBCPolicy, Dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    config = ArmBCConfig(**checkpoint["model_config"])
    model = ArmBCPolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _build_env(args: argparse.Namespace, seed: int):
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
            "online BC requires Stage-B strict-arm environment"
        )
    if not description.get("l2_selected_arm_enforced", False):
        raise RuntimeError("strict L2 arm enforcement is disabled")
    return env, base_env, description


def _run_policy(
    policy_name: str,
    args: argparse.Namespace,
    model: ArmBCPolicy | None,
    device: torch.device,
) -> Dict[str, Any]:
    env, base_env, description = _build_env(args, args.seed)
    spatial_action_n = (
        env.max_poses * env.grid_height * env.grid_width
    )
    episodes: List[Dict[str, Any]] = []
    pass_scores: List[float] = []
    fail_parts = Counter()
    fail_reasons = Counter()
    arm_counts = Counter()
    pose_counts = Counter()
    exact_rejection_total = 0

    try:
        for episode in range(args.episodes_per_policy):
            episode_seed = args.seed + episode
            rng = np.random.default_rng(
                episode_seed
                + {
                    "bc": 100000,
                    "grasp_ik_greedy": 200000,
                    "random": 300000,
                }[policy_name]
            )
            observation, reset_info = env.reset(
                seed=episode_seed,
                options={"region_id": args.region_id},
            )
            distance_map = _distance_map(base_env)
            rejected_spatial_actions: Set[int] = set()
            attempts_for_current_part = 0
            terminated = truncated = False
            steps: List[Dict[str, Any]] = []
            final_info = dict(reset_info)

            print(
                f"[online-reset] policy={policy_name:18s} "
                f"episode={episode:03d} seed={episode_seed} "
                f"part={reset_info['current_part']} "
                f"valid={int(np.asarray(observation['action_mask']).sum())}"
            )

            while not (terminated or truncated):
                state = base_env.state
                if state is None or state.current_part is None:
                    raise RuntimeError("missing current part")
                part_before = str(state.current_part)
                step_before = int(state.current_step)

                effective_mask = _mask_rejected_spatial_actions(
                    np.asarray(observation["action_mask"], dtype=np.bool_),
                    rejected_spatial_actions,
                    spatial_action_n=spatial_action_n,
                )
                if not effective_mask.any():
                    raise RuntimeError(
                        f"no valid action remains for {part_before}"
                    )

                if policy_name == "bc":
                    if model is None:
                        raise RuntimeError("BC model is not loaded")
                    action, diagnostics = _select_bc_action(
                        model,
                        observation,
                        effective_mask,
                        device,
                        rng,
                        temperature=args.bc_temperature,
                        sample_top_k=args.bc_sample_top_k,
                    )
                elif policy_name == "grasp_ik_greedy":
                    action, diagnostics = _select_heuristic_action(
                        observation,
                        effective_mask,
                        distance_map,
                        rng,
                        w_grasp=args.w_grasp,
                        w_ik=args.w_ik,
                        w_distance=args.w_distance,
                        proposal_top_k=args.proposal_top_k,
                        sample_top_k=args.sample_top_k,
                        temperature=args.heuristic_temperature,
                    )
                elif policy_name == "random":
                    action, diagnostics = _select_random_action(
                        effective_mask,
                        rng,
                    )
                else:
                    raise ValueError(policy_name)

                arm_id, pose_id, row, col = env.codec.unflatten(action)
                observation, reward, terminated, truncated, final_info = (
                    env.step(action)
                )
                accepted = bool(final_info["arm_action_accepted"])

                if accepted:
                    rejected_spatial_actions.clear()
                    attempts_for_current_part = 0
                    arm_counts.update([str(final_info["arm"])])
                    pose_counts.update([int(pose_id)])
                else:
                    exact_rejection_total += 1
                    rejected_spatial_actions.add(
                        env.codec.spatial_action(action)
                    )
                    attempts_for_current_part += 1
                    if attempts_for_current_part >= args.max_attempts_per_part:
                        raise RuntimeError(
                            f"{part_before} exceeded "
                            f"{args.max_attempts_per_part} exact attempts"
                        )

                step_row = {
                    "step": step_before,
                    "part_id": part_before,
                    "joint_action": int(action),
                    "arm_id": int(arm_id),
                    "arm": str(final_info["arm"]),
                    "pose_id": int(pose_id),
                    "row": int(row),
                    "col": int(col),
                    "arm_ik_hint": float(final_info["arm_ik_hint"]),
                    "accepted": accepted,
                    "reward": float(reward),
                    "event": final_info.get("event"),
                    **diagnostics,
                }
                steps.append(step_row)

                diagnostic_text = ""
                if policy_name == "bc":
                    diagnostic_text = (
                        f" margin={diagnostics['top1_top2_margin']:.3f}"
                    )
                print(
                    f"[online-step] policy={policy_name:18s} "
                    f"episode={episode:03d} "
                    f"step={step_before + 1:02d}/{env.n_decision_parts:02d} "
                    f"part={part_before:16s} "
                    f"arm={final_info['arm']:3s} "
                    f"pose={pose_id:02d} row={row:02d} col={col:02d} "
                    f"accepted={accepted} "
                    f"event={final_info.get('event')} "
                    f"reward={float(reward):+.4f}"
                    f"{diagnostic_text}"
                )

            strict = bool(
                final_info.get("l2_selected_arm_enforced", False)
            )
            arm_match = final_info.get("l2_arm_choice_match")
            l2_pass = bool(final_info.get("l2_pass", False))
            if not strict:
                raise RuntimeError(
                    "terminal L2 did not enforce selected arms"
                )
            if l2_pass and arm_match is not True:
                raise RuntimeError(
                    "successful L2 arm choice differs from policy"
                )

            layout_score = final_info.get("layout_score")
            if l2_pass and layout_score is not None:
                pass_scores.append(float(layout_score))
            if not l2_pass:
                if final_info.get("fail_part"):
                    fail_parts.update([str(final_info["fail_part"])])
                if final_info.get("fail_reason"):
                    fail_reasons.update([str(final_info["fail_reason"])])

            episode_row = {
                "episode": episode,
                "seed": episode_seed,
                "l2_pass": l2_pass,
                "layout_score": layout_score,
                "strict": strict,
                "arm_match": arm_match,
                "selected_arms": dict(
                    final_info.get("selected_arms", {})
                ),
                "l2_arm_choice": dict(
                    final_info.get("l2_arm_choice", {})
                ),
                "fail_part": final_info.get("fail_part"),
                "fail_reason": final_info.get("fail_reason"),
                "steps": steps,
            }
            episodes.append(episode_row)

            print(
                f"[online-episode] policy={policy_name:18s} "
                f"episode={episode:03d} "
                f"pass={l2_pass} score={layout_score} "
                f"strict={strict} arm_match={arm_match}"
            )
    finally:
        env.close()

    pass_count = sum(int(row["l2_pass"]) for row in episodes)
    return {
        "policy": policy_name,
        "environment": description,
        "episodes": episodes,
        "statistics": {
            "episode_count": len(episodes),
            "l2_pass_count": pass_count,
            "l2_pass_rate": pass_count / max(1, len(episodes)),
            "mean_layout_score_on_pass": (
                float(np.mean(pass_scores))
                if pass_scores
                else None
            ),
            "exact_rejection_total": exact_rejection_total,
            "arm_counts": dict(arm_counts),
            "pose_counts": {
                str(key): int(value)
                for key, value in sorted(pose_counts.items())
            },
            "fail_part_counts": dict(fail_parts),
            "fail_reason_counts": dict(fail_reasons),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--grasp-hint", required=True)
    parser.add_argument("--ik-hint", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--region-id", default="r1_c1")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["bc"],
        choices=["bc", "grasp_ik_greedy", "random"],
    )
    parser.add_argument("--episodes-per-policy", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--bc-temperature", type=float, default=0.0)
    parser.add_argument("--bc-sample-top-k", type=int, default=5)

    parser.add_argument("--proposal-top-k", type=int, default=128)
    parser.add_argument("--sample-top-k", type=int, default=16)
    parser.add_argument(
        "--heuristic-temperature",
        type=float,
        default=0.02,
    )
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
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_policy <= 0:
        raise ValueError("--episodes-per-policy must be positive")
    if "bc" in args.policies and not args.checkpoint:
        raise ValueError("--checkpoint is required for policy bc")

    device = resolve_device(args.device)
    model = None
    checkpoint_metadata = None
    if "bc" in args.policies:
        model, checkpoint_metadata = _load_bc_checkpoint(
            args.checkpoint,
            device,
        )
        expected_action_n = 2 * args.max_poses * 54 * 24
        if model.config.action_n != expected_action_n:
            raise ValueError(
                f"checkpoint action_n={model.config.action_n}, "
                f"expected={expected_action_n}"
            )

    print("=" * 80)
    print("Online strict-arm policy evaluation")
    print(f"policies   : {args.policies}")
    print(f"episodes   : {args.episodes_per_policy} each")
    print(f"seed range : {args.seed}.."
          f"{args.seed + args.episodes_per_policy - 1}")
    print(f"device     : {device}")
    print("=" * 80)

    results: Dict[str, Any] = {}
    for policy_name in args.policies:
        results[policy_name] = _run_policy(
            policy_name,
            args,
            model,
            device,
        )

    payload = {
        "format_version": "2026-07-17-arm-online-v1",
        "checkpoint": (
            str(Path(args.checkpoint).expanduser().resolve())
            if args.checkpoint
            else None
        ),
        "checkpoint_epoch": (
            checkpoint_metadata.get("epoch")
            if checkpoint_metadata is not None
            else None
        ),
        "seed": args.seed,
        "episodes_per_policy": args.episodes_per_policy,
        "policies": results,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    for policy_name, result in results.items():
        stats = result["statistics"]
        print(
            f"{policy_name:18s} "
            f"pass={stats['l2_pass_count']}/"
            f"{stats['episode_count']} "
            f"rate={stats['l2_pass_rate']:.3f} "
            f"mean_score={stats['mean_layout_score_on_pass']} "
            f"exact_reject={stats['exact_rejection_total']}"
        )
    print(f"[OK] saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
