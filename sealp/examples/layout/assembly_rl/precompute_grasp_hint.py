#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute pose-wise grasp hints for a generic ASMDEF layout task.

The output is a *soft prior*, not a replacement for fixed L2 validation.
For every non-preassembled part and every stable pose, the script evaluates:

- top-down grasp ratio and continuous downward alignment;
- action-center-above-table ratio (a cheap table-clearance proxy);
- logarithmic support from the number of usable grasps;
- orientation diversity of grasp frames.

ASMDEF step 0 is always skipped: it is directly preassembled at the selected
assembly region and never becomes a grasp/action target.
"""
from __future__ import annotations

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

from .assembly_layout_env import build_assembly_validator  # noqa: E402


SCRIPT_VERSION = "2026-07-16-v2-field-map-fix"

DEFAULT_WEIGHTS = {
    "topdown_ratio": 0.40,
    "down_alignment_mean": 0.20,
    "table_clear_ratio": 0.25,
    "usable_count_support": 0.10,
    "orientation_diversity": 0.05,
}


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


def _safe_unit(vector: Sequence[float]) -> Optional[np.ndarray]:
    value = np.asarray(vector, dtype=float).reshape(-1)
    if value.size != 3 or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm <= 1e-10:
        return None
    return value / norm


def _best_down_alignment(world_rotmat: np.ndarray) -> float:
    """Match the project's existing top-down convention (±X and ±Z axes)."""
    down = np.asarray([0.0, 0.0, -1.0], dtype=float)
    axes = (
        world_rotmat[:, 0],
        -world_rotmat[:, 0],
        world_rotmat[:, 2],
        -world_rotmat[:, 2],
    )
    values: List[float] = []
    for axis in axes:
        unit = _safe_unit(axis)
        if unit is not None:
            values.append(float(np.dot(unit, down)))
    return float(np.clip(max(values, default=0.0), 0.0, 1.0))


def _orientation_diversity(world_rotmats: Sequence[np.ndarray]) -> float:
    """Entropy of the dominant unsigned axis of the grasp-frame Z axis.

    This is sign invariant and therefore remains useful for symmetric parts
    such as cylindrical legs.  It is only a small term in the final hint.
    """
    counts = np.zeros(3, dtype=float)
    for rot in world_rotmats:
        axis = _safe_unit(np.asarray(rot, dtype=float)[:, 2])
        if axis is None:
            continue
        counts[int(np.argmax(np.abs(axis)))] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0.0] / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / math.log(3.0)) if probs.size > 1 else 0.0


def _extract_grasps(collection: Any) -> List[Any]:
    if collection is None:
        return []
    try:
        return list(collection)
    except TypeError:
        values = getattr(collection, "_grasp_list", None)
        return list(values or [])


def _pose_stats(
    grasps: Sequence[Any],
    *,
    object_rotmat: np.ndarray,
    z_offset: float,
    table_top_z: float,
    align_cos: float,
    min_ac_height: float,
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    total = len(grasps)
    if total == 0:
        return {
            "score": 0.0,
            "pose_valid": False,
            "total_count": 0,
            "evaluated_count": 0,
            "topdown_count": 0,
            "topdown_ratio": 0.0,
            "down_alignment_mean": 0.0,
            "table_clear_count": 0,
            "table_clear_ratio": 0.0,
            "usable_topdown_count": 0,
            "usable_count_support": 0.0,
            "orientation_diversity": 0.0,
        }

    alignments: List[float] = []
    world_rotmats: List[np.ndarray] = []
    topdown_count = 0
    table_clear_count = 0
    usable_topdown_count = 0

    object_rotmat = np.asarray(object_rotmat, dtype=float)
    object_origin_z = float(table_top_z) + float(z_offset)
    threshold_z = float(table_top_z) + float(min_ac_height)

    for grasp in grasps:
        try:
            grasp_rot = np.asarray(grasp.ac_rotmat, dtype=float)
            grasp_pos = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
            if grasp_rot.shape != (3, 3):
                continue
            world_rot = object_rotmat @ grasp_rot
            world_pos = object_rotmat @ grasp_pos
            center_z = object_origin_z + float(world_pos[2])
            alignment = _best_down_alignment(world_rot)
            topdown = alignment >= float(align_cos)
            table_clear = center_z >= threshold_z
            alignments.append(alignment)
            world_rotmats.append(world_rot)
            topdown_count += int(topdown)
            table_clear_count += int(table_clear)
            usable_topdown_count += int(topdown and table_clear)
        except Exception:
            continue

    evaluated = max(1, len(alignments))
    topdown_ratio = usable_topdown_count / evaluated
    clear_ratio = table_clear_count / evaluated
    alignment_mean = float(np.mean(alignments)) if alignments else 0.0
    count_support = math.log1p(table_clear_count) / math.log1p(max(1, evaluated))
    diversity = _orientation_diversity(world_rotmats)

    components = {
        "topdown_ratio": float(topdown_ratio),
        "down_alignment_mean": float(alignment_mean),
        "table_clear_ratio": float(clear_ratio),
        "usable_count_support": float(count_support),
        "orientation_diversity": float(diversity),
    }
    score = sum(float(weights[key]) * components[key] for key in DEFAULT_WEIGHTS)
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        # Only an empty grasp set or every action centre below the table makes
        # a pose invalid.  Lack of top-down grasps remains a soft penalty.
        "pose_valid": bool(table_clear_count > 0),
        "total_count": int(total),
        "evaluated_count": int(len(alignments)),
        "topdown_count": int(topdown_count),
        "topdown_ratio": float(topdown_ratio),
        "down_alignment_mean": float(alignment_mean),
        "table_clear_count": int(table_clear_count),
        "table_clear_ratio": float(clear_ratio),
        "usable_topdown_count": int(usable_topdown_count),
        "usable_count_support": float(count_support),
        "orientation_diversity": float(diversity),
    }


def precompute_grasp_hints(
    *,
    asmdef_path: str,
    config_yaml: str,
    grasp_dir: str,
    max_poses: int = 16,
    cdprim_type: str = "box",
    planner_obstacle_mode: str = "staging_aware",
    align_cos: float = 0.82,
    min_ac_height: float = 0.003,
    force_first_at_region_center: bool = True,
) -> Dict[str, Any]:
    validator, task, prepared = build_assembly_validator(
        asmdef_path=asmdef_path,
        config_yaml=config_yaml,
        grasp_dir=grasp_dir,
        cdprim_type=cdprim_type,
        planner_obstacle_mode=planner_obstacle_mode,
        max_poses=max_poses,
        force_first_at_region_center=force_first_at_region_center,
    )
    searcher = validator.searcher
    decision_parts = list(task.decision_parts)
    n_parts = len(decision_parts)
    shape = (n_parts, int(max_poses))

    arrays: Dict[str, np.ndarray] = {
        "scores": np.zeros(shape, dtype=np.float32),
        "pose_mask": np.zeros(shape, dtype=bool),
        "pose_valid": np.zeros(shape, dtype=bool),
        "total_count": np.zeros(shape, dtype=np.int32),
        "evaluated_count": np.zeros(shape, dtype=np.int32),
        "topdown_count": np.zeros(shape, dtype=np.int32),
        "topdown_ratio": np.zeros(shape, dtype=np.float32),
        "down_alignment_mean": np.zeros(shape, dtype=np.float32),
        "table_clear_count": np.zeros(shape, dtype=np.int32),
        "table_clear_ratio": np.zeros(shape, dtype=np.float32),
        "usable_topdown_count": np.zeros(shape, dtype=np.int32),
        "usable_count_support": np.zeros(shape, dtype=np.float32),
        "orientation_diversity": np.zeros(shape, dtype=np.float32),
    }
    part_rows: List[Dict[str, Any]] = []
    # Shared models/grasp files and identical pose matrices are computed once.
    stat_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    print("========== Pose-wise Grasp Hint ==========")
    print(
        f"[skip] preassembled step-0 part: {task.first_part} "
        "(not an action/grasp/IK target)"
    )
    for part_index, part in enumerate(decision_parts):
        pkl_path = searcher.grasp_file_for_part.get(part)
        collection = searcher.grasp_cache.get(pkl_path) if pkl_path else None
        grasps = _extract_grasps(collection)
        candidates = list(searcher.rot_cands.get(part, []))[: int(max_poses)]
        arrays["pose_mask"][part_index, : len(candidates)] = True
        pose_rows: List[Dict[str, Any]] = []

        if not pkl_path or not grasps:
            print(f"[WARN] {part:18s}: missing/empty grasp collection")
        else:
            print(
                f"[part] {part:18s}: grasps={len(grasps):4d} "
                f"poses={len(candidates):2d} file={Path(pkl_path).name}"
            )

        for pose_id, candidate in enumerate(candidates):
            rot = np.asarray(candidate.rotmat, dtype=float)
            signature = (
                str(Path(pkl_path).resolve()) if pkl_path else "",
                tuple(np.round(rot.reshape(-1), 8)),
                round(float(candidate.z_offset), 8),
                round(float(align_cos), 6),
                round(float(min_ac_height), 6),
            )
            stats = stat_cache.get(signature)
            if stats is None:
                stats = _pose_stats(
                    grasps,
                    object_rotmat=rot,
                    z_offset=float(candidate.z_offset),
                    table_top_z=float(searcher.table_top_z),
                    align_cos=float(align_cos),
                    min_ac_height=float(min_ac_height),
                    weights=DEFAULT_WEIGHTS,
                )
                stat_cache[signature] = stats

            # The persisted NPZ field is named ``scores`` for compatibility
            # with GraspHintCache, while the per-pose statistic is ``score``.
            # Use an explicit schema instead of assuming identical key names.
            array_to_stat = {
                "scores": "score",
                "pose_valid": "pose_valid",
                "total_count": "total_count",
                "evaluated_count": "evaluated_count",
                "topdown_count": "topdown_count",
                "topdown_ratio": "topdown_ratio",
                "down_alignment_mean": "down_alignment_mean",
                "table_clear_count": "table_clear_count",
                "table_clear_ratio": "table_clear_ratio",
                "usable_topdown_count": "usable_topdown_count",
                "usable_count_support": "usable_count_support",
                "orientation_diversity": "orientation_diversity",
            }
            for array_key, stat_key in array_to_stat.items():
                arrays[array_key][part_index, pose_id] = stats[stat_key]
            pose_rows.append(
                {
                    "pose_id": int(pose_id),
                    "rot_name": str(candidate.rot_name),
                    "pose_tag": str(candidate.tag),
                    **stats,
                }
            )
            print(
                f"       pose={pose_id:02d} {str(candidate.rot_name):12s} "
                f"score={stats['score']:.3f} "
                f"top={stats['topdown_ratio']:.3f} "
                f"clear={stats['table_clear_ratio']:.3f} "
                f"valid={int(stats['pose_valid'])}"
            )

        part_rows.append(
            {
                "part_id": part,
                "model_id": task.model_ids.get(part),
                "grasp_file": pkl_path,
                "grasp_count": len(grasps),
                "pose_count": len(candidates),
                "poses": pose_rows,
            }
        )

    payload = {
        "task_name": task.name,
        "asmdef": str(Path(asmdef_path).expanduser().resolve()),
        "effective_asmdef": task.asmdef_path,
        "preassembled_part": task.first_part,
        "preassembled_part_skipped": True,
        "decision_parts": decision_parts,
        "max_poses": int(max_poses),
        "align_cos": float(align_cos),
        "min_ac_height": float(min_ac_height),
        "score_weights": dict(DEFAULT_WEIGHTS),
        "score_note": (
            "soft pose prior; pose_valid only checks non-empty grasp set and "
            "action-center-above-table proxy; fixed L2 remains authoritative"
        ),
        "parts": part_rows,
        "arrays": arrays,
        "prepared_asmdef": prepared,
    }
    return payload


def save_outputs(payload: Dict[str, Any], *, output_npz: str, output_json: str) -> None:
    npz_path = Path(output_npz).expanduser().resolve()
    json_path = Path(output_json).expanduser().resolve()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = dict(payload["arrays"])
    np.savez_compressed(
        npz_path,
        part_ids=np.asarray(payload["decision_parts"], dtype="U128"),
        **arrays,
    )
    json_payload = {k: v for k, v in payload.items() if k != "arrays"}
    prepared = json_payload.get("prepared_asmdef")
    if prepared is not None:
        json_payload["prepared_asmdef"] = {
            "original_path": prepared.original_path,
            "effective_path": prepared.effective_path,
            "first_part": prepared.first_part,
            "original_first_rel_pos": list(prepared.original_first_rel_pos),
            "forced_first_at_region_center": prepared.forced_first_at_region_center,
            "temporary": prepared.temporary,
        }
    json_payload["output_npz"] = str(npz_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(json_payload), f, ensure_ascii=False, indent=2)
    print(f"[OK] NPZ saved to: {npz_path}")
    print(f"[OK] JSON saved to: {json_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute pose-wise grasp hints.")
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--max-poses", type=int, default=16)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--align-cos", type=float, default=0.82)
    parser.add_argument(
        "--min-ac-height",
        type=float,
        default=0.003,
        help="Minimum action-center height above table used by the cheap proxy.",
    )
    parser.add_argument("--keep-first-rel-pos", action="store_true")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    print(f"[precompute_grasp_hint] version={SCRIPT_VERSION}")
    args = _parse_args()
    payload = precompute_grasp_hints(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        max_poses=args.max_poses,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        align_cos=args.align_cos,
        min_ac_height=args.min_ac_height,
        force_first_at_region_center=not args.keep_first_rel_pos,
    )
    try:
        save_outputs(payload, output_npz=args.output_npz, output_json=args.output_json)
    finally:
        prepared = payload.get("prepared_asmdef")
        if prepared is not None and getattr(prepared, "temporary", False):
            try:
                os.remove(prepared.effective_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
