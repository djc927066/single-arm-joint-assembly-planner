#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic grasp-coverage sampling and progressive common-grasp filtering.

This module patches the existing fast backend without changing the strict
verification criteria. Search uses a deterministic, approximately uniform
coverage subset of the grasp library; final certification can still set
MAX_GRASPS_PER_POSE=0 to use all grasps.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import numpy as np

import find_optimal_initial_layout_tower_strict_pycharm_fast as fast


DEFAULT_SEARCH_GRASP_CAP = int(os.environ.get("SEALP_SEARCH_GRASP_CAP", "128"))
MAX_POSE_CACHE_RECORDS = int(os.environ.get("SEALP_POSE_CACHE_MAX", "40000"))

# pose key -> {"tested": set[int], "feasible": set[int]}
_PROGRESSIVE_CACHE: "OrderedDict[Tuple, Dict[str, set]]" = OrderedDict()
_FPS_CACHE: Dict[Tuple[int, int], List[int]] = {}


def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _grasp_feature(grasp) -> np.ndarray:
    """Pose-space descriptor used only for deterministic coverage selection."""
    try:
        p = np.asarray(grasp.ac_pos, dtype=float).reshape(3)
    except Exception:
        p = np.zeros(3, dtype=float)
    try:
        R = np.asarray(grasp.ac_rotmat, dtype=float).reshape(3, 3)
        # Two orthogonal tool axes represent orientation without Euler singularity.
        a = _safe_unit(R[:, 2])
        b = _safe_unit(R[:, 0])
    except Exception:
        a = np.zeros(3, dtype=float)
        b = np.zeros(3, dtype=float)
    try:
        ee = np.asarray(grasp.ee_values, dtype=float).reshape(-1)
        jaw = float(ee[0]) if ee.size else 0.0
    except Exception:
        jaw = 0.0
    return np.concatenate([p, a, b, np.array([jaw], dtype=float)])


def _deterministic_fps_gids(grasp_collection, cap: int) -> List[int]:
    """Deterministic farthest-point sampling in grasp pose space.

    No RNG or seed is used. The first sample is the point farthest from the
    feature centroid; each next sample maximizes distance to the selected set.
    """
    n = len(grasp_collection)
    if cap <= 0 or n <= cap:
        return list(range(n))

    key = (id(grasp_collection), int(cap))
    cached = _FPS_CACHE.get(key)
    if cached is not None:
        return cached

    X = np.vstack([_grasp_feature(grasp_collection[i]) for i in range(n)])

    # Normalize translation and jaw width so orientation is not overwhelmed.
    for lo, hi in ((0, 3), (9, 10)):
        block = X[:, lo:hi]
        scale = np.std(block, axis=0)
        scale[scale < 1e-9] = 1.0
        X[:, lo:hi] = (block - np.mean(block, axis=0)) / scale

    center = np.mean(X, axis=0)
    d_center = np.sum((X - center) ** 2, axis=1)
    first = int(np.argmax(d_center))

    selected = [first]
    min_d2 = np.sum((X - X[first]) ** 2, axis=1)
    min_d2[first] = -1.0

    while len(selected) < cap:
        nxt = int(np.argmax(min_d2))
        selected.append(nxt)
        d2 = np.sum((X - X[nxt]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected] = -1.0

    out = sorted(selected)
    _FPS_CACHE[key] = out
    return out


def _gc_subset_gids(grasp_collection) -> List[int]:
    cap = int(fast.MAX_GRASPS_PER_POSE)
    return _deterministic_fps_gids(grasp_collection, cap)


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
    for gid in gids:
        gid = int(gid)
        if gid in entry["tested"]:
            continue
        entry["tested"].add(gid)
        grasp = grasp_collection[gid]
        jaw_center_pos = pos + rotmat.dot(np.asarray(grasp.ac_pos, dtype=float))
        jaw_center_rotmat = rotmat.dot(np.asarray(grasp.ac_rotmat, dtype=float))
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


def _progressive_reason_common_gids(
    self,
    grasp_collection,
    goal_pose_list,
    obstacle_list=None,
    toggle_dbg=False,
):
    """Exact intersection on the selected deterministic grasp subset.

    Unlike the old implementation, each subsequent pose tests only grasps that
    survived previous poses. Pose order is deterministic and favors an already
    cached/restrictive pose; with no cache, the last pose is checked first
    because it is commonly the repeated assembly-goal pose.
    """
    if toggle_dbg and fast._ORIG_REASON_COMMON_GIDS is not None:
        return fast._ORIG_REASON_COMMON_GIDS(
            self,
            grasp_collection,
            goal_pose_list,
            obstacle_list=obstacle_list,
            toggle_dbg=toggle_dbg,
        )

    robot = self.robot
    obs = list(obstacle_list) if obstacle_list else []
    obs_key = fast._obstacle_pose_key(obs)
    gc_key = id(grasp_collection)
    rb_key = id(robot)
    survivors = list(_gc_subset_gids(grasp_collection))

    pose_rows = []
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
        # cached first; then smaller cached feasible set; otherwise last pose first.
        order_key = (
            0 if cached else 1,
            feasible_n,
            -original_i,
        )
        pose_rows.append((order_key, pose_key, pos, rotmat))

    pose_rows.sort(key=lambda row: row[0])

    for _, pose_key, pos, rotmat in pose_rows:
        entry = _cache_entry(pose_key)
        missing = [g for g in survivors if g not in entry["tested"]]
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

        feasible = entry["feasible"]
        survivors = [g for g in survivors if g in feasible]
        if not survivors:
            return []

    return survivors


def _bounded_reset_for_layout() -> None:
    """Keep reusable pose/grasp results; LRU bounds memory instead of clearing."""
    fast._POSE_FEASIBLE_CACHE.clear()
    while len(_PROGRESSIVE_CACHE) > MAX_POSE_CACHE_RECORDS:
        _PROGRESSIVE_CACHE.popitem(last=False)


def reset_all() -> None:
    _PROGRESSIVE_CACHE.clear()
    _FPS_CACHE.clear()
    fast._pose_cache_reset_stats()


def install(search_cap: int = DEFAULT_SEARCH_GRASP_CAP) -> None:
    from wrs.manipulation.pick_place import PickPlacePlanner

    fast.MAX_GRASPS_PER_POSE = int(search_cap)
    fast._gc_subset_gids = _gc_subset_gids
    fast._cached_reason_common_gids = _progressive_reason_common_gids
    fast._pose_cache_reset_for_layout = _bounded_reset_for_layout

    _progressive_reason_common_gids._is_ik_cached = True
    PickPlacePlanner.reason_common_gids = _progressive_reason_common_gids

    print(
        "[uniform-grasp] deterministic pose-space FPS installed: "
        f"cap={fast.MAX_GRASPS_PER_POSE}, "
        f"pose_cache_max={MAX_POSE_CACHE_RECORDS}"
    )
    print(
        "[uniform-grasp] progressive intersection enabled: "
        "later poses test only surviving grasp IDs"
    )
