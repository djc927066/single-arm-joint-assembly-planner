#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare heuristic policies before PPO training.

Policies
--------
random
    Uniform over geometrically valid actions, with optional L1.5 exact
    geometry proposal filtering.

grasp_greedy
    Score = w_grasp * GraspHint + w_distance * DistanceHint.

grasp_ik_greedy
    Score = w_grasp * GraspHint + w_ik * IKHint
            + w_distance * DistanceHint.

For greedy policies, the evaluator:
1. ranks all currently masked actions;
2. keeps the top ``proposal_top_k`` candidates;
3. runs the L1.5 exact geometry check on candidates in rank order;
4. collects up to ``sample_top_k`` exact-valid actions;
5. samples from them with a softmax temperature.

This module does not train a neural policy.  It is a diagnostic baseline used
to verify that Grasp/IK hints improve strict-L2 success before Maskable PPO.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .assembly_layout_env import (
    AssemblyLayoutEnv,
    GridWorkspace,
    REGION_MODE_FIXED,
    _jsonable,
    build_assembly_validator,
)


POLICY_RANDOM = "random"
POLICY_GRASP = "grasp_greedy"
POLICY_GRASP_IK = "grasp_ik_greedy"
SUPPORTED_POLICIES = (POLICY_RANDOM, POLICY_GRASP, POLICY_GRASP_IK)


def _parse_csv(value: str) -> List[str]:
    out = [token.strip() for token in str(value).split(",") if token.strip()]
    if not out:
        raise argparse.ArgumentTypeError("at least one policy is required")
    invalid = [token for token in out if token not in SUPPORTED_POLICIES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown policies={invalid}; supported={SUPPORTED_POLICIES}"
        )
    return out


def _softmax_sample(
    actions: np.ndarray,
    scores: np.ndarray,
    *,
    rng: np.random.Generator,
    temperature: float,
) -> int:
    if actions.size == 0:
        raise ValueError("cannot sample from empty action set")
    if actions.size == 1 or temperature <= 1e-9:
        return int(actions[int(np.argmax(scores))])

    logits = np.asarray(scores, dtype=float) / float(temperature)
    logits -= float(np.max(logits))
    probs = np.exp(np.clip(logits, -60.0, 0.0))
    total = float(np.sum(probs))
    if not math.isfinite(total) or total <= 0.0:
        return int(actions[int(np.argmax(scores))])
    probs /= total
    return int(rng.choice(actions, p=probs))


def _distance_hint_map(env: AssemblyLayoutEnv, part_id: str) -> np.ndarray:
    """Return [H,W] closeness to the final assembly target, in [0,1]."""
    state = env._require_state()
    spec = state.workspace.spec
    target = np.asarray(env.searcher.world_poses[part_id][0], dtype=float)[:2]
    diagonal = math.hypot(spec.x_max - spec.x_min, spec.y_max - spec.y_min)

    out = np.zeros((env.grid_height, env.grid_width), dtype=np.float32)
    for row in range(env.grid_height):
        for col in range(env.grid_width):
            xy = spec.cell_center(row, col)
            distance = float(np.linalg.norm(xy - target))
            out[row, col] = 1.0 - min(
                1.0,
                distance / max(1e-9, diagonal),
            )
    return out


def _flat_policy_scores(
    env: AssemblyLayoutEnv,
    policy: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    state = env._require_state()
    if state.current_part is None:
        raise RuntimeError("no current part")
    part_id = str(state.current_part)

    grasp = env._current_grasp_hint(state).astype(np.float32, copy=False)
    ik = env._current_ik_hint(state).astype(np.float32, copy=False)
    distance = _distance_hint_map(env, part_id)

    components = {
        "grasp": np.broadcast_to(
            grasp[:, None, None],
            (env.max_poses, env.grid_height, env.grid_width),
        ),
        "ik": ik,
        "distance": np.broadcast_to(
            distance[None, :, :],
            (env.max_poses, env.grid_height, env.grid_width),
        ),
    }

    # Raw component maps are returned; caller applies CLI weights.
    return np.zeros(env.action_n, dtype=np.float32), {
        key: np.asarray(value, dtype=np.float32).reshape(-1)
        for key, value in components.items()
    }


def _candidate_order(
    *,
    policy: str,
    valid_actions: np.ndarray,
    weighted_scores: np.ndarray,
    rng: np.random.Generator,
    proposal_top_k: int,
) -> np.ndarray:
    if valid_actions.size == 0:
        return np.zeros(0, dtype=np.int64)

    limit = min(int(proposal_top_k), int(valid_actions.size))
    if policy == POLICY_RANDOM:
        shuffled = valid_actions.copy()
        rng.shuffle(shuffled)
        return shuffled[:limit]

    valid_scores = weighted_scores[valid_actions]
    if limit >= valid_actions.size:
        order = np.argsort(valid_scores)[::-1]
        return valid_actions[order]

    # argpartition avoids sorting all 20k actions.
    top_local = np.argpartition(valid_scores, -limit)[-limit:]
    top_local = top_local[np.argsort(valid_scores[top_local])[::-1]]
    return valid_actions[top_local]


def select_action(
    env: AssemblyLayoutEnv,
    *,
    policy: str,
    rng: np.random.Generator,
    proposal_top_k: int,
    sample_top_k: int,
    exact_prefilter: bool,
    temperature: float,
    w_grasp: float,
    w_ik: float,
    w_distance: float,
) -> Tuple[Optional[int], Dict[str, Any]]:
    state = env._require_state()
    part_id = str(state.current_part)
    mask = env.action_masks().astype(bool, copy=False)
    valid_actions = np.flatnonzero(mask)

    _, components = _flat_policy_scores(env, policy)
    if policy == POLICY_RANDOM:
        weighted = np.zeros(env.action_n, dtype=np.float32)
    elif policy == POLICY_GRASP:
        weighted = (
            float(w_grasp) * components["grasp"]
            + float(w_distance) * components["distance"]
        )
    elif policy == POLICY_GRASP_IK:
        weighted = (
            float(w_grasp) * components["grasp"]
            + float(w_ik) * components["ik"]
            + float(w_distance) * components["distance"]
        )
    else:
        raise ValueError(f"unsupported policy {policy!r}")

    ordered = _candidate_order(
        policy=policy,
        valid_actions=valid_actions,
        weighted_scores=weighted,
        rng=rng,
        proposal_top_k=proposal_top_k,
    )

    exact_valid: List[int] = []
    rejection_reasons: Counter[str] = Counter()
    checked = 0

    for action in ordered:
        checked += 1
        if exact_prefilter:
            exact = env.check_exact_action(int(action))
            if not exact.valid:
                rejection_reasons[str(exact.reason)] += 1
                continue
        exact_valid.append(int(action))
        if len(exact_valid) >= max(1, int(sample_top_k)):
            break

    if not exact_valid:
        return None, {
            "part_id": part_id,
            "valid_action_count": int(valid_actions.size),
            "proposal_count": int(ordered.size),
            "exact_checked": int(checked),
            "exact_valid_count": 0,
            "exact_rejections": dict(rejection_reasons),
            "failure_reason": "no_exact_valid_proposal",
        }

    actions = np.asarray(exact_valid, dtype=np.int64)
    candidate_scores = weighted[actions]
    if policy == POLICY_RANDOM:
        chosen = int(rng.choice(actions))
    else:
        chosen = _softmax_sample(
            actions,
            candidate_scores,
            rng=rng,
            temperature=float(temperature),
        )

    pose_id, row, col = GridWorkspace.unflatten_action(
        chosen,
        height=env.grid_height,
        width=env.grid_width,
    )

    detail = {
        "part_id": part_id,
        "valid_action_count": int(valid_actions.size),
        "proposal_count": int(ordered.size),
        "exact_checked": int(checked),
        "exact_valid_count": int(actions.size),
        "exact_rejections": dict(rejection_reasons),
        "action": int(chosen),
        "pose_id": int(pose_id),
        "grid_row": int(row),
        "grid_col": int(col),
        "score": float(weighted[chosen]),
        "grasp_hint": float(components["grasp"][chosen]),
        "ik_hint": float(components["ik"][chosen]),
        "distance_hint": float(components["distance"][chosen]),
    }
    return chosen, detail


def _build_env(args: argparse.Namespace) -> AssemblyLayoutEnv:
    validator, task, prepared = build_assembly_validator(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_poses=args.max_poses,
        force_first_at_region_center=not args.keep_first_rel_pos,
    )
    return AssemblyLayoutEnv(
        validator,
        task,
        resolution=args.resolution,
        max_parts=args.max_parts,
        max_poses=args.max_poses,
        region_mode=REGION_MODE_FIXED,
        fixed_region_id=args.region_id,
        run_l2=not args.skip_l2,
        seed=args.seed,
        prepared_asmdef=prepared,
        grasp_hint_path=args.grasp_hint,
        grasp_reward_weight=args.grasp_reward_weight,
        mask_invalid_grasp_poses=args.mask_invalid_grasp_poses,
        ik_hint_path=args.ik_hint,
        ik_reward_weight=args.ik_reward_weight,
        mask_zero_ik_actions=args.mask_zero_ik_actions,
        ik_mask_threshold=args.ik_mask_threshold,
        exact_geometry_check=True,
        exact_geometry_reject_penalty=args.exact_geometry_reject_penalty,
    )


def evaluate_policy(
    env: AssemblyLayoutEnv,
    *,
    policy: str,
    episodes: int,
    base_seed: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    pass_count = 0
    event_counter: Counter[str] = Counter()
    fail_reason_counter: Counter[str] = Counter()
    fail_part_counter: Counter[str] = Counter()
    fail_type_episode_counter: Counter[str] = Counter()
    fail_type_attempt_counter: Counter[str] = Counter()
    exact_rejection_counter: Counter[str] = Counter()
    layout_scores: List[float] = []

    for episode in range(int(episodes)):
        episode_seed = int(base_seed + episode)
        rng = np.random.default_rng(episode_seed)
        _, reset_info = env.reset(
            seed=episode_seed,
            options={"region_id": args.region_id},
        )
        print(
            f"[reset] policy={policy:16s} episode={episode:03d} "
            f"region={reset_info['region_id']} "
            f"part={reset_info['current_part']} "
            f"valid={reset_info['valid_action_count']}"
        )

        terminated = truncated = False
        selections: List[Dict[str, Any]] = []
        final_info: Dict[str, Any] = dict(reset_info)

        while not (terminated or truncated):
            action, selection = select_action(
                env,
                policy=policy,
                rng=rng,
                proposal_top_k=args.proposal_top_k,
                sample_top_k=args.sample_top_k,
                exact_prefilter=not args.no_exact_prefilter,
                temperature=args.temperature,
                w_grasp=args.w_grasp,
                w_ik=args.w_ik,
                w_distance=args.w_distance,
            )
            selections.append(selection)
            for reason, count in dict(selection.get("exact_rejections") or {}).items():
                exact_rejection_counter[str(reason)] += int(count)

            if action is None:
                # No mutation has happened. Record a diagnostic dead end.
                final_info = {
                    **env._base_info(),
                    "event": "proposal_dead_end",
                    "episode_steps": int(env._episode_steps),
                    "episode_return": float(env._episode_return),
                    "l2_pass": False,
                    "layout_score": -1.0,
                    "fail_part": selection.get("part_id"),
                    "fail_reason": selection.get("failure_reason"),
                }
                env._terminated = True
                terminated = True
                print(
                    f"[select] policy={policy:16s} episode={episode:03d} "
                    f"part={selection.get('part_id')} FAILED "
                    f"rejections={selection.get('exact_rejections')}"
                )
                break

            _, reward, terminated, truncated, final_info = env.step(action)
            print(
                f"[select] policy={policy:16s} episode={episode:03d} "
                f"step={final_info.get('episode_steps', 0):02d}/"
                f"{env.n_decision_parts:02d} "
                f"part={selection['part_id']:16s} "
                f"pose={selection['pose_id']:02d} "
                f"score={selection['score']:.4f} "
                f"g={selection['grasp_hint']:.3f} "
                f"ik={selection['ik_hint']:.3f} "
                f"event={final_info.get('event')} reward={reward:+.4f}"
            )

        event = str(final_info.get("event"))
        event_counter[event] += 1
        fail_reason = str(
            final_info.get("fail_reason")
            or final_info.get("failure_reason")
            or ""
        )
        if fail_reason:
            # Retained for backward compatibility. For strict-L2 messages the
            # prefix before ':' is usually the part id, not the reason type.
            category = fail_reason.split(":", 1)[0]
            fail_reason_counter[category] += 1

        fail_part = final_info.get("fail_part")
        if fail_part:
            fail_part_counter[str(fail_part)] += 1

        fail_detail = dict(final_info.get("fail_detail") or {})
        if not fail_detail:
            l2_result = final_info.get("l2_result") or {}
            fail_detail = dict(l2_result.get("fail_detail") or {})
        active_types = []
        for reason, count in fail_detail.items():
            count_i = int(count)
            if count_i > 0:
                active_types.append(str(reason))
                fail_type_attempt_counter[str(reason)] += count_i
        for reason in set(active_types):
            fail_type_episode_counter[reason] += 1
        if not active_types and fail_reason and not final_info.get("l2_pass"):
            fallback = str(final_info.get("failure_reason") or "unclassified_failure")
            fail_type_episode_counter[fallback] += 1
            fail_type_attempt_counter[fallback] += 1

        l2_pass = final_info.get("l2_pass") is True
        if l2_pass:
            pass_count += 1
        score = final_info.get("layout_score")
        if l2_pass and score is not None:
            layout_scores.append(float(score))

        rows.append(
            {
                "policy": policy,
                "episode": int(episode),
                "seed": episode_seed,
                "region_id": final_info.get("region_id", args.region_id),
                "event": event,
                "episode_steps": final_info.get("episode_steps"),
                "episode_return": final_info.get("episode_return"),
                "l2_pass": bool(l2_pass),
                "layout_score": score,
                "fail_part": final_info.get("fail_part"),
                "fail_reason": final_info.get("fail_reason"),
                "failure_reason": final_info.get("failure_reason"),
                "failure_blocker": final_info.get("failure_blocker"),
                "fixed_l2_request": final_info.get("fixed_l2_request"),
                "l2_result": final_info.get("l2_result"),
                "selections": selections,
            }
        )

    return {
        "policy": policy,
        "episodes": rows,
        "statistics": {
            "episode_count": int(len(rows)),
            "l2_pass_count": int(pass_count),
            "l2_pass_rate": float(pass_count / max(1, len(rows))),
            "mean_layout_score_on_pass": (
                float(np.mean(layout_scores)) if layout_scores else None
            ),
            "event_counts": dict(event_counter),
            "fail_reason_counts": dict(fail_reason_counter),
            "fail_part_counts": dict(fail_part_counter),
            "fail_type_episode_counts": dict(fail_type_episode_counter),
            "fail_type_attempt_counts": dict(fail_type_attempt_counter),
            "exact_rejection_counts": dict(exact_rejection_counter),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random, Grasp-greedy and Grasp+IK-greedy policies."
    )
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--grasp-hint", required=True)
    parser.add_argument("--ik-hint", required=True)
    parser.add_argument("--region-id", default="r1_c1")
    parser.add_argument(
        "--policies",
        type=_parse_csv,
        default=list(SUPPORTED_POLICIES),
    )
    parser.add_argument("--episodes-per-policy", type=int, default=20)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-l2", action="store_true")
    parser.add_argument("--mask-invalid-grasp-poses", action="store_true")
    parser.add_argument("--mask-zero-ik-actions", action="store_true")
    parser.add_argument("--ik-mask-threshold", type=float, default=0.0)
    parser.add_argument("--no-exact-prefilter", action="store_true")
    parser.add_argument("--exact-geometry-reject-penalty", type=float, default=-1.0)
    parser.add_argument("--keep-first-rel-pos", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if args.episodes_per_policy <= 0:
        parser.error("--episodes-per-policy must be positive")
    if args.proposal_top_k <= 0:
        parser.error("--proposal-top-k must be positive")
    if args.sample_top_k <= 0:
        parser.error("--sample-top-k must be positive")
    if args.sample_top_k > args.proposal_top_k:
        parser.error("--sample-top-k cannot exceed --proposal-top-k")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    for name in ("w_grasp", "w_ik", "w_distance"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_','-')} must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    print("[evaluate_hint_policy] policies=", args.policies)
    print(
        "[evaluate_hint_policy] score weights:",
        {
            "grasp": args.w_grasp,
            "ik": args.w_ik,
            "distance": args.w_distance,
        },
    )
    print(
        "[evaluate_hint_policy] proposal:",
        {
            "top_k": args.proposal_top_k,
            "sample_top_k": args.sample_top_k,
            "temperature": args.temperature,
            "exact_prefilter": not args.no_exact_prefilter,
        },
    )

    env = _build_env(args)
    results: List[Dict[str, Any]] = []
    try:
        print(
            json.dumps(
                env.compact_description(),
                ensure_ascii=False,
                indent=2,
            )
        )
        for policy_index, policy in enumerate(args.policies):
            result = evaluate_policy(
                env,
                policy=policy,
                episodes=args.episodes_per_policy,
                base_seed=args.seed,
                args=args,
            )
            results.append(result)
            print(
                f"[summary] policy={policy:16s} "
                f"pass={result['statistics']['l2_pass_count']}/"
                f"{result['statistics']['episode_count']} "
                f"rate={result['statistics']['l2_pass_rate']:.3f} "
                f"mean_score={result['statistics']['mean_layout_score_on_pass']}"
            )
    finally:
        env.close()

    payload = {
        "environment": env.compact_description(),
        "configuration": {
            "policies": list(args.policies),
            "episodes_per_policy": args.episodes_per_policy,
            "proposal_top_k": args.proposal_top_k,
            "sample_top_k": args.sample_top_k,
            "temperature": args.temperature,
            "exact_prefilter": not args.no_exact_prefilter,
            "weights": {
                "grasp": args.w_grasp,
                "ik": args.w_ik,
                "distance": args.w_distance,
            },
        },
        "results": results,
        "comparison": {
            row["policy"]: row["statistics"] for row in results
        },
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"[OK] saved to: {output}")


if __name__ == "__main__":
    main()
