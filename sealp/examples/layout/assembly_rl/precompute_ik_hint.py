#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute coarse-to-dense IK/common-grasp hints for AssemblyLayoutEnv.

The first ASMDEF part is always preassembled and is never evaluated as a pick
or IK target.  For every remaining decision part, candidate stable pose and
staging grid cell, this module estimates whether the left or right arm has at
least one *common grasp id* that is feasible both at the staging pose and at
that part's final ASMDEF target pose.

The computation is intentionally configurable:

* ``--max-grasps`` deterministically caps the grasp ids evaluated per part;
* ``--grid-stride`` evaluates a coarse grid and nearest-fills the 2 cm map;
* ``--parts`` / ``--poses`` / ``--max-cells-per-pose`` support smoke tests;
* the full dense arrays keep a ``computed_mask`` so partial runs are explicit.

The resulting NPZ is a soft prior, not a replacement for fixed-layout L2.
"""
from __future__ import annotations

IK_HINT_VERSION = "2026-07-16-v1-common-grasp-coarse-dense"

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from .assembly_layout_env import (
    AssemblyGridState,
    AssemblyTaskSpec,
    PreparedAsmdef,
    build_assembly_validator,
)


ARM_TAGS: Tuple[str, str] = ("lft", "rgt")


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items or None


def _parse_pose_ids(value: Optional[str]) -> Optional[List[int]]:
    values = _parse_csv(value)
    if values is None:
        return None
    out: List[int] = []
    for token in values:
        if "-" in token:
            left, right = token.split("-", 1)
            a, b = int(left), int(right)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def _deterministic_gid_subset(grasp_count: int, cap: int) -> List[int]:
    n = int(grasp_count)
    if n <= 0:
        return []
    if cap <= 0 or n <= cap:
        return list(range(n))
    ids = np.linspace(0, n - 1, num=int(cap), dtype=np.int64)
    return sorted(set(int(x) for x in ids.tolist()))


def _arm_for_tag(searcher: Any, arm_tag: str) -> Any:
    if arm_tag == "lft":
        return searcher.robot.lft_arm
    if arm_tag == "rgt":
        return searcher.robot.rgt_arm
    raise KeyError(arm_tag)


def _feasible_gids(
    arm: Any,
    grasp_collection: Any,
    gid_iter: Sequence[int],
    pos: Sequence[float],
    rotmat: Sequence[Sequence[float]],
    obstacle_list: Sequence[Any],
) -> List[int]:
    """Replicate the per-pose filter used by reason_common_gids."""
    pos_arr = np.asarray(pos, dtype=float)
    rot_arr = np.asarray(rotmat, dtype=float)
    obstacles = list(obstacle_list)
    out: List[int] = []
    ee = arm.end_effector

    for gid in gid_iter:
        try:
            grasp = grasp_collection[int(gid)]
            jaw_center_pos = pos_arr + rot_arr.dot(np.asarray(grasp.ac_pos, dtype=float))
            jaw_center_rotmat = rot_arr.dot(np.asarray(grasp.ac_rotmat, dtype=float))
            jnt_values = arm.ik(
                tgt_pos=jaw_center_pos,
                tgt_rotmat=jaw_center_rotmat,
            )
            if jnt_values is None:
                continue
            arm.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
            if arm.is_collided(obstacle_list=obstacles):
                continue
            if ee.is_mesh_collided(cmodel_list=obstacles):
                continue
            out.append(int(gid))
        except Exception:
            # A single malformed grasp or numerical IK failure must not abort
            # the complete cache generation.
            continue
    return out


def _common_score(common_count: int, target_count: int) -> float:
    """Map common-grasp count to [0,1] without letting huge libraries dominate."""
    n = int(common_count)
    denom = int(target_count)
    if n <= 0 or denom <= 0:
        return 0.0
    ratio = math.log1p(n) / max(1e-9, math.log1p(denom))
    return float(np.clip(0.5 + 0.5 * ratio, 0.0, 1.0))


def _previously_assembled_parts(task: AssemblyTaskSpec, part_id: str) -> set:
    index = task.part_to_index[str(part_id)]
    return set(task.part_order[:index])


def _planner_obstacles_for_part(searcher: Any, task: AssemblyTaskSpec, part_id: str) -> List[Any]:
    """Build obstacles that are independent of unknown staging placements.

    We include environment objects according to ``planner_obstacle_mode`` and
    goal models of parts assembled before the current part.  Future staging
    objects are intentionally omitted because their policy-selected positions
    are unknown during offline hint precomputation; geometry/action masks and
    terminal L2 handle those interactions later.
    """
    placed = _previously_assembled_parts(task, part_id)
    raw: List[Any] = list(searcher.env_obs)
    for previous in placed:
        model = searcher.goal_models.get(previous)
        if model is not None:
            raw.append(model)
    return list(
        searcher._planner_obstacles(
            raw,
            current_pid=str(part_id),
            placed=placed,
        )
    )


def _sample_axis(size: int, stride: int) -> np.ndarray:
    indices = list(range(0, int(size), max(1, int(stride))))
    if not indices or indices[-1] != size - 1:
        indices.append(size - 1)
    return np.asarray(sorted(set(indices)), dtype=np.int32)


def _nearest_fill(
    values: np.ndarray,
    sampled_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Nearest-fill sampled scores only onto geometrically valid cells."""
    out = np.zeros_like(values, dtype=np.float32)
    sample_coords = np.argwhere(sampled_mask)
    if sample_coords.size == 0:
        return out
    sample_values = values[sampled_mask]
    valid_coords = np.argwhere(valid_mask)
    for row, col in valid_coords:
        delta = sample_coords - np.asarray([row, col], dtype=np.int32)
        nearest = int(np.argmin(np.sum(delta * delta, axis=1)))
        out[int(row), int(col)] = float(sample_values[nearest])
    return out


def _resolve_regions(validator: Any, selector: str) -> List[str]:
    rows = validator.available_regions()
    available = [str(row["region_id"]) for row in rows]
    preferred = [str(row["region_id"]) for row in rows if bool(row.get("preferred"))]
    token = str(selector).strip()
    if token == "all":
        return available
    if token == "preferred":
        return preferred or available
    requested = _parse_csv(token) or []
    unknown = [rid for rid in requested if rid not in available]
    if unknown:
        raise ValueError(f"unknown/illegal regions={unknown}; available={available}")
    return requested


def _resolve_parts(task: AssemblyTaskSpec, selector: Optional[str]) -> List[str]:
    requested = _parse_csv(selector)
    if requested is None:
        return list(task.decision_parts)
    unknown = [part for part in requested if part not in task.decision_parts]
    if unknown:
        raise ValueError(
            f"parts are not decision targets: {unknown}; decision_parts={task.decision_parts}"
        )
    return requested


def _save_npz(path: str, arrays: Mapping[str, np.ndarray]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **arrays)
    print(f"[checkpoint] NPZ saved to: {output}")


def _summary_payload(
    *,
    asmdef_path: str,
    task: AssemblyTaskSpec,
    prepared: PreparedAsmdef,
    region_ids: Sequence[str],
    selected_parts: Sequence[str],
    selected_pose_ids: Optional[Sequence[int]],
    max_poses: int,
    height: int,
    width: int,
    resolution: float,
    max_grasps: int,
    grid_stride: int,
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    scores = np.asarray(arrays["scores"], dtype=np.float32)
    computed = np.asarray(arrays["computed_mask"], dtype=bool)
    geometry = np.asarray(arrays["geometry_valid"], dtype=bool)
    target_counts = np.asarray(arrays["target_feasible_count"], dtype=np.int32)

    region_summaries: Dict[str, Any] = {}
    all_regions = [str(x) for x in arrays["region_ids"].tolist()]
    all_parts = [str(x) for x in arrays["part_ids"].tolist()]
    for region_id in region_ids:
        r = all_regions.index(region_id)
        part_rows: Dict[str, Any] = {}
        for part_id in selected_parts:
            p = all_parts.index(part_id)
            mask = computed[r, p]
            vals = scores[r, p][mask]
            geom_count = int(geometry[r, p].sum())
            part_rows[part_id] = {
                "target_feasible_lft": int(target_counts[r, p, 0]),
                "target_feasible_rgt": int(target_counts[r, p, 1]),
                "geometry_valid_cells": geom_count,
                "computed_cells": int(mask.sum()),
                "positive_cells": int(np.count_nonzero(vals > 0.0)),
                "score_min": float(vals.min()) if vals.size else 0.0,
                "score_mean": float(vals.mean()) if vals.size else 0.0,
                "score_max": float(vals.max()) if vals.size else 0.0,
            }
        region_summaries[region_id] = part_rows

    return {
        "version": IK_HINT_VERSION,
        "asmdef": str(Path(asmdef_path).resolve()),
        "effective_asmdef": task.asmdef_path,
        "task_name": task.name,
        "preassembled_part": task.first_part,
        "preassembled_part_skipped": True,
        "first_part_forced_to_region_center": bool(prepared.forced_first_at_region_center),
        "decision_parts": list(task.decision_parts),
        "selected_parts": list(selected_parts),
        "selected_pose_ids": None if selected_pose_ids is None else list(selected_pose_ids),
        "region_ids": list(region_ids),
        "max_poses": int(max_poses),
        "grid": {
            "height": int(height),
            "width": int(width),
            "resolution": float(resolution),
            "grid_stride": int(grid_stride),
        },
        "max_grasps": int(max_grasps),
        "score_definition": (
            "0 if no common grasp; otherwise 0.5 + 0.5*"
            "log(1+n_common)/log(1+n_target), max over left/right arm"
        ),
        "obstacle_scope": (
            "environment per planner mode + already-assembled goal parts; "
            "future staging parts omitted because their positions are policy decisions"
        ),
        "region_summary": region_summaries,
    }


def precompute(args: argparse.Namespace) -> Dict[str, Any]:
    validator, task, prepared = build_assembly_validator(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_poses=args.max_poses,
        force_first_at_region_center=not args.keep_first_rel_pos,
    )
    searcher = validator.searcher
    region_ids = _resolve_regions(validator, args.regions)
    selected_parts = _resolve_parts(task, args.parts)
    selected_pose_ids = _parse_pose_ids(args.poses)

    if not region_ids:
        raise RuntimeError("no region selected")

    template = AssemblyGridState(
        validator,
        task,
        region_id=region_ids[0],
        resolution=args.resolution,
        max_poses=args.max_poses,
    )
    height = int(template.workspace.spec.height)
    width = int(template.workspace.spec.width)
    part_ids = list(task.decision_parts)
    part_to_index = {part: i for i, part in enumerate(part_ids)}
    region_to_index = {region: i for i, region in enumerate(region_ids)}

    shape = (len(region_ids), len(part_ids), args.max_poses, height, width)
    arrays: Dict[str, np.ndarray] = {
        "version": np.asarray(IK_HINT_VERSION),
        "task_name": np.asarray(task.name),
        "preassembled_part": np.asarray(task.first_part),
        "region_ids": np.asarray(region_ids, dtype="U64"),
        "part_ids": np.asarray(part_ids, dtype="U128"),
        "scores": np.zeros(shape, dtype=np.float16),
        "left_scores": np.zeros(shape, dtype=np.float16),
        "right_scores": np.zeros(shape, dtype=np.float16),
        "left_common_count": np.zeros(shape, dtype=np.uint16),
        "right_common_count": np.zeros(shape, dtype=np.uint16),
        "arm_preference": np.zeros(shape, dtype=np.uint8),
        "geometry_valid": np.zeros(shape, dtype=np.bool_),
        "computed_mask": np.zeros(shape, dtype=np.bool_),
        "sampled_mask": np.zeros(shape, dtype=np.bool_),
        "pose_mask": np.zeros((len(part_ids), args.max_poses), dtype=np.bool_),
        "target_feasible_count": np.zeros(
            (len(region_ids), len(part_ids), len(ARM_TAGS)), dtype=np.uint16
        ),
        "grid_height": np.asarray(height, dtype=np.int32),
        "grid_width": np.asarray(width, dtype=np.int32),
        "resolution": np.asarray(float(args.resolution), dtype=np.float32),
        "grid_stride": np.asarray(int(args.grid_stride), dtype=np.int32),
        "max_grasps": np.asarray(int(args.max_grasps), dtype=np.int32),
    }

    rows = _sample_axis(height, args.grid_stride)
    cols = _sample_axis(width, args.grid_stride)
    rng = np.random.default_rng(args.seed)

    print("========== IK Hint Precomputation ==========")
    print(f"version               = {IK_HINT_VERSION}")
    print(f"task                  = {task.name}")
    print(f"preassembled skipped  = {task.first_part}")
    print(f"regions               = {region_ids}")
    print(f"parts                 = {selected_parts}")
    print(f"poses filter          = {selected_pose_ids}")
    print(f"grid                  = {height} x {width} @ {args.resolution:.3f} m")
    print(f"grid stride           = {args.grid_stride}")
    print(f"max grasps            = {args.max_grasps}")

    for region_id in region_ids:
        r_index = region_to_index[region_id]
        state = AssemblyGridState(
            validator,
            task,
            region_id=region_id,
            resolution=args.resolution,
            max_poses=args.max_poses,
        )
        print(f"\n[region] {region_id} center={np.round(state.region_center, 4).tolist()}")

        for part_id in selected_parts:
            p_index = part_to_index[part_id]
            decision_step = task.decision_parts.index(part_id)
            state.current_step = int(decision_step)
            geometry = state.flat_action_mask().reshape(
                args.max_poses, height, width
            ).astype(bool)
            arrays["geometry_valid"][r_index, p_index] = geometry

            candidates = list(searcher.rot_cands.get(part_id, []))
            pose_count = min(len(candidates), args.max_poses)
            arrays["pose_mask"][p_index, :pose_count] = True
            pose_ids = list(range(pose_count))
            if selected_pose_ids is not None:
                pose_ids = [pid for pid in pose_ids if pid in selected_pose_ids]
            if not pose_ids:
                print(f"  [skip] {part_id}: no selected pose ids")
                continue

            grasp_collection = searcher._grasp_collection(part_id)
            if grasp_collection is None or len(grasp_collection) == 0:
                print(f"  [skip] {part_id}: missing/empty grasp collection")
                continue
            base_gids = _deterministic_gid_subset(
                len(grasp_collection), args.max_grasps
            )
            goal_pos, goal_rot = searcher.world_poses[part_id]
            obstacles = _planner_obstacles_for_part(searcher, task, part_id)
            print(
                f"  [part] {part_id:18s} grasps={len(grasp_collection):4d} "
                f"sampled={len(base_gids):3d} poses={len(pose_ids):2d} "
                f"obs={len(obstacles):2d}"
            )

            target_feasible: Dict[str, List[int]] = {}
            for arm_i, arm_tag in enumerate(ARM_TAGS):
                arm = _arm_for_tag(searcher, arm_tag)
                arm.backup_state()
                try:
                    feasible = _feasible_gids(
                        arm,
                        grasp_collection,
                        base_gids,
                        goal_pos,
                        goal_rot,
                        obstacles,
                    )
                finally:
                    arm.restore_state()
                target_feasible[arm_tag] = feasible
                arrays["target_feasible_count"][r_index, p_index, arm_i] = int(
                    len(feasible)
                )
                print(
                    f"         target arm={arm_tag} feasible={len(feasible):3d}/"
                    f"{len(base_gids):3d}"
                )

            for pose_id in pose_ids:
                cand = candidates[pose_id]
                valid = geometry[pose_id]
                sample_coords = [
                    (int(row), int(col))
                    for row in rows
                    for col in cols
                    if bool(valid[int(row), int(col)])
                ]
                if args.max_cells_per_pose > 0 and len(sample_coords) > args.max_cells_per_pose:
                    chosen = rng.choice(
                        len(sample_coords),
                        size=int(args.max_cells_per_pose),
                        replace=False,
                    )
                    sample_coords = [sample_coords[int(i)] for i in sorted(chosen.tolist())]

                left_values = np.zeros((height, width), dtype=np.float32)
                right_values = np.zeros((height, width), dtype=np.float32)
                left_counts = np.zeros((height, width), dtype=np.uint16)
                right_counts = np.zeros((height, width), dtype=np.uint16)
                sampled = np.zeros((height, width), dtype=bool)

                for row, col in sample_coords:
                    xy = state.workspace.spec.cell_center(row, col)
                    stage_pos = np.asarray(
                        [float(xy[0]), float(xy[1]), float(cand.z_offset)],
                        dtype=float,
                    )
                    stage_rot = np.asarray(cand.rotmat, dtype=float)
                    sampled[row, col] = True

                    for arm_tag, values, counts in (
                        ("lft", left_values, left_counts),
                        ("rgt", right_values, right_counts),
                    ):
                        survivors = target_feasible[arm_tag]
                        if not survivors:
                            continue
                        arm = _arm_for_tag(searcher, arm_tag)
                        arm.backup_state()
                        try:
                            common = _feasible_gids(
                                arm,
                                grasp_collection,
                                survivors,
                                stage_pos,
                                stage_rot,
                                obstacles,
                            )
                        finally:
                            arm.restore_state()
                        counts[row, col] = int(len(common))
                        values[row, col] = _common_score(
                            len(common), len(survivors)
                        )

                left_dense = _nearest_fill(left_values, sampled, valid)
                right_dense = _nearest_fill(right_values, sampled, valid)
                left_count_dense = np.rint(
                    _nearest_fill(left_counts.astype(np.float32), sampled, valid)
                ).astype(np.uint16)
                right_count_dense = np.rint(
                    _nearest_fill(right_counts.astype(np.float32), sampled, valid)
                ).astype(np.uint16)
                combined = np.maximum(left_dense, right_dense)
                preference = np.zeros((height, width), dtype=np.uint8)
                preference[(left_dense > right_dense) & (left_dense > 0)] = 1
                preference[(right_dense > left_dense) & (right_dense > 0)] = 2
                preference[(left_dense == right_dense) & (left_dense > 0)] = 3

                arrays["left_scores"][r_index, p_index, pose_id] = left_dense.astype(np.float16)
                arrays["right_scores"][r_index, p_index, pose_id] = right_dense.astype(np.float16)
                arrays["scores"][r_index, p_index, pose_id] = combined.astype(np.float16)
                arrays["left_common_count"][r_index, p_index, pose_id] = left_count_dense
                arrays["right_common_count"][r_index, p_index, pose_id] = right_count_dense
                arrays["arm_preference"][r_index, p_index, pose_id] = preference
                arrays["sampled_mask"][r_index, p_index, pose_id] = sampled
                arrays["computed_mask"][r_index, p_index, pose_id] = valid

                vals = combined[valid]
                print(
                    f"         pose={pose_id:02d} {str(cand.rot_name):12s} "
                    f"sampled={int(sampled.sum()):4d} valid={int(valid.sum()):4d} "
                    f"positive={int(np.count_nonzero(vals > 0)):4d} "
                    f"mean={float(vals.mean()) if vals.size else 0.0:.3f} "
                    f"max={float(vals.max()) if vals.size else 0.0:.3f}"
                )

            if args.output_npz:
                _save_npz(args.output_npz, arrays)

    summary = _summary_payload(
        asmdef_path=args.asmdef,
        task=task,
        prepared=prepared,
        region_ids=region_ids,
        selected_parts=selected_parts,
        selected_pose_ids=selected_pose_ids,
        max_poses=args.max_poses,
        height=height,
        width=width,
        resolution=args.resolution,
        max_grasps=args.max_grasps,
        grid_stride=args.grid_stride,
        arrays=arrays,
    )

    if args.output_npz:
        _save_npz(args.output_npz, arrays)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] JSON saved to: {output_json}")

    if prepared.temporary:
        try:
            os.remove(prepared.effective_path)
        except FileNotFoundError:
            pass
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute common-grasp IK hints for generic ASMDEF layouts."
    )
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--max-poses", type=int, default=16)
    parser.add_argument("--max-grasps", type=int, default=32)
    parser.add_argument("--grid-stride", type=int, default=2)
    parser.add_argument(
        "--max-cells-per-pose",
        type=int,
        default=0,
        help="0 means all coarse-grid valid cells; positive values are for smoke tests.",
    )
    parser.add_argument(
        "--regions",
        default="r1_c1",
        help="Comma-separated region IDs, or 'preferred', or 'all'.",
    )
    parser.add_argument(
        "--parts",
        default=None,
        help="Optional comma-separated decision part IDs.",
    )
    parser.add_argument(
        "--poses",
        default=None,
        help="Optional pose IDs, e.g. '0,5,8-10'.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--keep-first-rel-pos", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(f"[precompute_ik_hint] version={IK_HINT_VERSION}")
    args = _parse_args()
    summary = precompute(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
