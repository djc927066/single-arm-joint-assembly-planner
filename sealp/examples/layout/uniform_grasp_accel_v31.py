#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic grasp coverage and exact same-grasp continuity checks.

The planner interface reason_common_gids means:
    one unchanged grasp ID must remain valid from the initial staging/pick pose
    through all intermediate approach/depart poses to the final assembly/place
    pose. The object is not released and regrasped in between.

Modes
-----
same_grasp_progressive
    Exact incremental set intersection. After each pose, only grasp IDs that
    are feasible at every previous pose remain candidates.

same_grasp_full
    Reference mode. Evaluate the full selected subset at every pose and
    intersect the feasible sets afterwards. It should return the same IDs as
    same_grasp_progressive, but normally uses more IK/collision evaluations.

FPS seed modes
--------------
medoid (default)
    Start from the actual grasp closest to the feature centroid. This avoids
    making a rare outlier the first representative. Subsequent points are the
    farthest from the selected set, so boundary/extreme grasps are still added.

extreme
    Start from the grasp farthest from the centroid.

first
    Start from original grasp ID 0.

No random-number generator is used.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import numpy as np

import find_optimal_initial_layout_tower_strict_pycharm_fast as fast


DEFAULT_SEARCH_GRASP_CAP = int(os.environ.get("SEALP_SEARCH_GRASP_CAP", "160"))
MAX_POSE_CACHE_RECORDS = int(os.environ.get("SEALP_POSE_CACHE_MAX", "40000"))

_MODE_ALIASES = {
    "same_grasp_progressive": "same_grasp_progressive",
    "common_progressive": "same_grasp_progressive",
    "same_grasp_full": "same_grasp_full",
    "common_full": "same_grasp_full",
}
VALID_SEED_MODES = {"medoid", "extreme", "first"}

_PROGRESSIVE_CACHE: "OrderedDict[Tuple, Dict[str, set]]" = OrderedDict()
_FPS_CACHE: Dict[Tuple[int, int, str], List[int]] = {}
_MODE = "same_grasp_progressive"
_FPS_SEED_MODE = "medoid"


def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _grasp_feature(grasp) -> np.ndarray:
    """10-D descriptor: position, approach, jaw axis, jaw width."""
    try:
        p = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
    except Exception:
        p = np.zeros(3, dtype=float)

    try:
        R = np.asarray(grasp.ac_rotmat, dtype=float).reshape(3, 3)
        approach = _safe_unit(R[:, 2])
        jaw_axis = _safe_unit(R[:, 0])
    except Exception:
        approach = np.zeros(3, dtype=float)
        jaw_axis = np.zeros(3, dtype=float)

    try:
        ee = np.asarray(grasp.ee_values, dtype=float).reshape(-1)
        jaw_width = float(ee[0]) if ee.size else 0.0
    except Exception:
        jaw_width = 0.0

    return np.concatenate(
        [p, approach, jaw_axis, np.array([jaw_width], dtype=float)]
    )


def _stable_argmax(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    target = float(np.max(values))
    tied = np.flatnonzero(
        np.isclose(values, target, rtol=0.0, atol=1e-14)
    )
    return int(tied[0])


def _stable_argmin(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float).reshape(-1)
    target = float(np.min(values))
    tied = np.flatnonzero(
        np.isclose(values, target, rtol=0.0, atol=1e-14)
    )
    return int(tied[0])


def _deterministic_fps_gids(
    grasp_collection,
    cap: int,
    seed_mode: str | None = None,
) -> List[int]:
    """Deterministic medoid-seeded farthest-point coverage."""
    n = len(grasp_collection)
    if cap <= 0 or n <= cap:
        return list(range(n))

    mode = str(seed_mode or _FPS_SEED_MODE).strip().lower()
    if mode not in VALID_SEED_MODES:
        raise ValueError(
            f"seed_mode must be one of {sorted(VALID_SEED_MODES)}, got {mode}"
        )

    key = (id(grasp_collection), int(cap), mode)
    cached = _FPS_CACHE.get(key)
    if cached is not None:
        return list(cached)

    X = np.vstack(
        [_grasp_feature(grasp_collection[i]) for i in range(n)]
    )

    # Normalize translation and jaw width so one unit does not dominate.
    for lo, hi in ((0, 3), (9, 10)):
        block = X[:, lo:hi]
        mean = np.mean(block, axis=0)
        scale = np.std(block, axis=0)
        scale[scale < 1e-9] = 1.0
        X[:, lo:hi] = (block - mean) / scale

    center = np.mean(X, axis=0)
    d_center = np.sum((X - center) ** 2, axis=1)

    if mode == "medoid":
        first = _stable_argmin(d_center)
    elif mode == "extreme":
        first = _stable_argmax(d_center)
    else:
        first = 0

    selected = [int(first)]
    min_d2 = np.sum((X - X[first]) ** 2, axis=1)
    min_d2[first] = -np.inf

    while len(selected) < int(cap):
        nxt = _stable_argmax(min_d2)
        selected.append(int(nxt))
        d2 = np.sum((X - X[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[np.asarray(selected, dtype=int)] = -np.inf

    result = sorted(selected)
    _FPS_CACHE[key] = list(result)
    return result


def _gc_subset_gids(grasp_collection) -> List[int]:
    return _deterministic_fps_gids(
        grasp_collection,
        int(fast.MAX_GRASPS_PER_POSE),
        seed_mode=_FPS_SEED_MODE,
    )


def _cache_entry(pose_key: Tuple) -> Dict[str, set]:
    entry = _PROGRESSIVE_CACHE.get(pose_key)
    if entry is None:
        entry = {"tested": set(), "feasible": set()}
        _PROGRESSIVE_CACHE[pose_key] = entry
    else:
        _PROGRESSIVE_CACHE.move_to_end(pose_key)

    while len(_PROGRESSIVE_CACHE) > MAX_POSE_CACHE_RECORDS:
        _PROGRESSIVE_CACHE.popitem(last=False)
    return entry


def _evaluate_missing(
    robot,
    grasp_collection,
    gids: Iterable[int],
    pos: np.ndarray,
    rotmat: np.ndarray,
    obstacle_list,
    entry: Dict[str, set],
) -> None:
    ee = robot.end_effector

    for raw_gid in gids:
        gid = int(raw_gid)
        if gid in entry["tested"]:
            continue

        entry["tested"].add(gid)
        grasp = grasp_collection[gid]

        jaw_center_pos = (
            np.asarray(pos, dtype=float)
            + np.asarray(rotmat, dtype=float).dot(
                np.asarray(grasp.ac_pos, dtype=float)
            )
        )
        jaw_center_rotmat = np.asarray(rotmat, dtype=float).dot(
            np.asarray(grasp.ac_rotmat, dtype=float)
        )

        fast._POSE_CACHE_STATS["ik_evals"] += 1
        jnt_values = robot.ik(
            tgt_pos=jaw_center_pos,
            tgt_rotmat=jaw_center_rotmat,
        )
        if jnt_values is None:
            continue

        robot.goto_given_conf(
            jnt_values=jnt_values,
            ee_values=grasp.ee_values,
        )
        if robot.is_collided(obstacle_list=obstacle_list):
            continue
        if ee.is_mesh_collided(cmodel_list=obstacle_list):
            continue

        entry["feasible"].add(gid)


def _pose_rows(
    robot,
    grasp_collection,
    goal_pose_list,
    obstacle_list,
):
    obs = list(obstacle_list) if obstacle_list else []
    obs_key = fast._obstacle_pose_key(obs)
    gc_key = id(grasp_collection)
    rb_key = id(robot)

    rows = []
    for original_i, goal_pose in enumerate(goal_pose_list):
        pos = np.asarray(goal_pose[0], dtype=float)
        rotmat = np.asarray(goal_pose[1], dtype=float)
        pose_key = (
            rb_key,
            gc_key,
            pos.tobytes(),
            rotmat.tobytes(),
            obs_key,
        )

        existing = _PROGRESSIVE_CACHE.get(pose_key)
        cached = existing is not None and bool(existing["tested"])
        feasible_n = len(existing["feasible"]) if cached else 10**9

        # Cached/restrictive poses first. This changes only evaluation order,
        # never the final set intersection.
        order_key = (
            0 if cached else 1,
            feasible_n,
            -original_i,
        )
        rows.append((order_key, pose_key, pos, rotmat))

    rows.sort(key=lambda row: row[0])
    return rows, obs


def _same_grasp_progressive(
    self,
    grasp_collection,
    goal_pose_list,
    obstacle_list=None,
    toggle_dbg=False,
):
    """Exact incremental intersection F1 ∩ F2 ∩ ... ∩ Fm."""
    if toggle_dbg and fast._ORIG_REASON_COMMON_GIDS is not None:
        return fast._ORIG_REASON_COMMON_GIDS(
            self,
            grasp_collection,
            goal_pose_list,
            obstacle_list=obstacle_list,
            toggle_dbg=toggle_dbg,
        )

    robot = self.robot
    pose_rows, obs = _pose_rows(
        robot,
        grasp_collection,
        goal_pose_list,
        obstacle_list,
    )
    survivors = list(_gc_subset_gids(grasp_collection))

    for _, pose_key, pos, rotmat in pose_rows:
        entry = _cache_entry(pose_key)

        # A grasp that failed an earlier pose cannot belong to the final
        # same-grasp intersection, even if it works at later poses.
        missing = [
            gid for gid in survivors
            if gid not in entry["tested"]
        ]
        if missing:
            fast._POSE_CACHE_STATS["miss"] += 1
            _evaluate_missing(
                robot,
                grasp_collection,
                missing,
                pos,
                rotmat,
                obs,
                entry,
            )
        else:
            fast._POSE_CACHE_STATS["hit"] += 1

        survivors = [
            gid for gid in survivors
            if gid in entry["feasible"]
        ]
        if not survivors:
            return []

    return survivors


def _same_grasp_full(
    self,
    grasp_collection,
    goal_pose_list,
    obstacle_list=None,
    toggle_dbg=False,
):
    """Reference mode: full subset at every pose, then exact intersection."""
    if toggle_dbg and fast._ORIG_REASON_COMMON_GIDS is not None:
        return fast._ORIG_REASON_COMMON_GIDS(
            self,
            grasp_collection,
            goal_pose_list,
            obstacle_list=obstacle_list,
            toggle_dbg=toggle_dbg,
        )

    robot = self.robot
    pose_rows, obs = _pose_rows(
        robot,
        grasp_collection,
        goal_pose_list,
        obstacle_list,
    )
    subset = list(_gc_subset_gids(grasp_collection))
    intersection = set(subset)

    for _, pose_key, pos, rotmat in pose_rows:
        entry = _cache_entry(pose_key)
        missing = [
            gid for gid in subset
            if gid not in entry["tested"]
        ]
        if missing:
            fast._POSE_CACHE_STATS["miss"] += 1
            _evaluate_missing(
                robot,
                grasp_collection,
                missing,
                pos,
                rotmat,
                obs,
                entry,
            )
        else:
            fast._POSE_CACHE_STATS["hit"] += 1

        intersection.intersection_update(entry["feasible"])
        if not intersection:
            return []

    return sorted(map(int, intersection))


def _bounded_reset_for_layout() -> None:
    fast._POSE_FEASIBLE_CACHE.clear()
    while len(_PROGRESSIVE_CACHE) > MAX_POSE_CACHE_RECORDS:
        _PROGRESSIVE_CACHE.popitem(last=False)


def reset_all() -> None:
    _PROGRESSIVE_CACHE.clear()
    _FPS_CACHE.clear()
    fast._pose_cache_reset_stats()


def install(
    search_cap: int = DEFAULT_SEARCH_GRASP_CAP,
    mode: str = "same_grasp_progressive",
    fps_seed_mode: str = "medoid",
) -> None:
    global _MODE, _FPS_SEED_MODE

    normalized_mode = _MODE_ALIASES.get(str(mode).strip().lower())
    if normalized_mode is None:
        raise ValueError(
            "mode must be same_grasp_progressive or same_grasp_full"
        )

    seed_mode = str(fps_seed_mode).strip().lower()
    if seed_mode not in VALID_SEED_MODES:
        raise ValueError(
            f"fps_seed_mode must be one of {sorted(VALID_SEED_MODES)}"
        )

    _MODE = normalized_mode
    _FPS_SEED_MODE = seed_mode

    from wrs.manipulation.pick_place import PickPlacePlanner

    fast.MAX_GRASPS_PER_POSE = int(search_cap)
    fast._gc_subset_gids = _gc_subset_gids
    fast._pose_cache_reset_for_layout = _bounded_reset_for_layout

    fn = (
        _same_grasp_progressive
        if normalized_mode == "same_grasp_progressive"
        else _same_grasp_full
    )
    fast._cached_reason_common_gids = fn
    fn._is_ik_cached = True
    PickPlacePlanner.reason_common_gids = fn

    print(
        "[uniform-grasp-v3] deterministic FPS installed: "
        f"cap={fast.MAX_GRASPS_PER_POSE}, seed={_FPS_SEED_MODE}, "
        f"pose_cache_max={MAX_POSE_CACHE_RECORDS}"
    )
    print(
        "[uniform-grasp-v3] continuity semantics: the same grasp ID is "
        "maintained from initial staging/pick to final assembly/place; "
        "release/regrasp is not allowed"
    )
    print(f"[uniform-grasp-v3] mode={_MODE}")
