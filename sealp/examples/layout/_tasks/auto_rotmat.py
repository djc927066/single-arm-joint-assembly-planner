"""Auto staging-rotmat inference from grasp library + mesh geometry
====================================================================

This module is used by task adapters such as ``shelf_unit.py`` to infer
generic staging orientation candidates.

Core idea
---------
A part should not be assumed to lie flat on the table. For thin / plate-like
parts, flat placement can make the gripper collide with the table even though
the grasp library contains valid grasps in the object frame.

Therefore this module:

1. reads the mesh AABB;
2. enumerates axis-aligned support orientations;
3. for thin parts, filters out low-height "flat" placements and keeps
   graspable upright / side-standing placements;
4. samples yaw rotations around world Z for every kept orientation;
5. computes the z offset that places the rotated mesh on the table;
6. ranks candidates by shared grasp feasibility between staging and goal.

The returned format remains compatible with ``find_optimal_layout.py``:

    {part_id: [(rotmat, z_offset), ...]}

where ``z_offset`` is the z coordinate that makes the rotated mesh just touch
the table top at z=0.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm


# ── Cache directory (created on first miss) ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(_HERE, "_auto_rotmat_cache")


# ══════════════════════════════════════════════════════════════════════════
#  Candidate rotmat pool
# ══════════════════════════════════════════════════════════════════════════
def _axis_aligned_base_rotmats() -> List[np.ndarray]:
    """Return 6 rotations mapping the mesh local +Z axis to ±world axes.

    For an axis-aligned mesh, this is enough to express all principal
    support orientations. Extra yaw around world Z is added later.
    """
    return [
        np.eye(3),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(180.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(-90.0)),
    ]


def _dedup_rotmats(rotmats: Sequence[np.ndarray], ndigits: int = 4) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    seen = set()
    for R in rotmats:
        key = tuple(np.round(np.asarray(R, dtype=float), ndigits).reshape(-1).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(np.asarray(R, dtype=float))
    return out


def _candidate_rotmats(n_yaw: int = 8) -> List[np.ndarray]:
    """Generate axis-aligned support orientations × yaw around world Z.

    ``n_yaw=8`` gives 0/45/90/.../315 degrees. This is more general than
    only using 0/90/180/270 and is useful for non-square plates or parts
    whose best arm reachability occurs at a diagonal yaw.
    """
    base = _axis_aligned_base_rotmats()
    yaw_angles = np.linspace(0.0, 360.0, int(n_yaw), endpoint=False)
    yaws = [rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(a))
            for a in yaw_angles]
    return _dedup_rotmats([Y @ B for B in base for Y in yaws])


# ══════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════════
def _mesh_aabb_corners(mesh_path: str) -> np.ndarray:
    """Return the 8 AABB corner points in mesh local frame as (8, 3)."""
    cm = mcm.CollisionModel(initor=mesh_path)
    cm.pos = np.zeros(3)
    cm.rotmat = np.eye(3)
    lo, hi = np.asarray(cm.trm_mesh.bounds, dtype=float)
    return np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
    ], dtype=float)


def _local_aabb_size(aabb_corners: np.ndarray) -> np.ndarray:
    lo = np.asarray(aabb_corners, dtype=float).min(axis=0)
    hi = np.asarray(aabb_corners, dtype=float).max(axis=0)
    return hi - lo


def _rotated_aabb_size(rotmat: np.ndarray, aabb_corners: np.ndarray) -> np.ndarray:
    pts = (np.asarray(rotmat, dtype=float) @ aabb_corners.T).T
    return pts.max(axis=0) - pts.min(axis=0)


def _z_offset_for_flat_on_table(rotmat: np.ndarray,
                                aabb_corners: np.ndarray) -> float:
    """Return z_off such that min_z(rotmat @ mesh + [0,0,z_off]) = 0."""
    rotated = (np.asarray(rotmat, dtype=float) @ aabb_corners.T).T
    return float(-rotated[:, 2].min())


def _is_thin_or_plate_like(aabb_corners: np.ndarray,
                           thin_aspect_ratio: float = 0.35) -> bool:
    """Detect whether an object is plate-like from its AABB dimensions.

    The rule is intentionally generic:
        min_dim / max_dim <= thin_aspect_ratio

    For shelf boards, 0.015 / 0.150 = 0.10, so it is detected as thin.
    For a near-cube, the ratio is close to 1 and no upright-only filtering
    is applied.
    """
    dims = np.sort(_local_aabb_size(aabb_corners))
    max_dim = float(dims[-1])
    if max_dim <= 1e-9:
        return False
    return float(dims[0] / max_dim) <= float(thin_aspect_ratio)


def _filter_graspable_upright_candidates(
    candidates: Sequence[np.ndarray],
    aabb_corners: np.ndarray,
    *,
    filter_flat_for_thin: bool = True,
    thin_aspect_ratio: float = 0.35,
    upright_height_ratio: float = 0.35,
) -> List[np.ndarray]:
    """Filter flat placements for thin parts.

    This is the key generic rule requested by the user:

    * Do NOT hard-code "shelf must stand upright".
    * Instead, if a part is thin / plate-like, reject orientations where the
      rotated AABB height is too small. Those are the flat-on-table poses that
      often make the gripper collide with the table.
    * After this filtering, yaw around world Z is still preserved, so the
      layout search can choose the best yaw.

    For non-thin parts, keep all candidate orientations.
    """
    candidates = list(candidates)
    if not filter_flat_for_thin:
        return candidates

    if not _is_thin_or_plate_like(aabb_corners, thin_aspect_ratio):
        return candidates

    dims = _local_aabb_size(aabb_corners)
    max_dim = float(np.max(dims))
    min_height = max_dim * float(upright_height_ratio)

    kept: List[np.ndarray] = []
    for R in candidates:
        height = float(_rotated_aabb_size(R, aabb_corners)[2])
        if height + 1e-9 >= min_height:
            kept.append(np.asarray(R, dtype=float))

    # Safety fallback: do not return empty merely because the threshold was
    # too strict for an odd geometry.
    return kept if kept else candidates


# ══════════════════════════════════════════════════════════════════════════
#  IK probing
# ══════════════════════════════════════════════════════════════════════════
def _probe_indices(grasp_collection, max_grasps: Optional[int]) -> List[int]:
    """Uniformly subsample grasp ids for cheap auto-ranking."""
    n = len(grasp_collection)
    if max_grasps is None or max_grasps <= 0 or max_grasps >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, int(max_grasps), dtype=int).tolist()))


def _feasible_grasp_ids(arm, grasp_collection,
                         base_pos: np.ndarray, base_rot: np.ndarray,
                         max_grasps: Optional[int] = None,
                         obstacle_list: Optional[Sequence[object]] = None,
                         ) -> set:
    """Return grasp ids whose world TCP pose is IK-solvable and collision-free.

    If ``obstacle_list`` is supplied, the arm is moved to the IK solution and
    checked against those obstacles. Passing the work table here is what makes
    flat plate poses automatically fail when the gripper collides with the
    tabletop.
    """
    feasible: set = set()
    for i in _probe_indices(grasp_collection, max_grasps):
        g = grasp_collection[i]
        tcp_pos = base_rot @ g.ac_pos + base_pos
        tcp_rot = base_rot @ g.ac_rotmat
        jv = arm.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
        if jv is None:
            continue
        if obstacle_list:
            arm.backup_state()
            try:
                arm.goto_given_conf(jnt_values=jv)
                hit = arm.is_collided(obstacle_list=list(obstacle_list))
                collided = hit[0] if isinstance(hit, tuple) else hit
                if collided:
                    continue
            finally:
                arm.restore_state()
        feasible.add(i)
    return feasible


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class _RotmatScore:
    R: np.ndarray
    z_off: float
    height: float
    footprint_area: float
    per_arm_common: dict
    score_max: int
    score_sum: int

    def __repr__(self):  # pragma: no cover - debug only
        return (f"R(z_ax→world:{tuple(np.round(self.R[:, 2], 2))}) "
                f"z_off={self.z_off:.3f} h={self.height:.3f} "
                f"foot={self.footprint_area:.4f} "
                f"common={self.per_arm_common} max={self.score_max}")


def _cache_key(*, mesh_path: str, goal_pos: np.ndarray, goal_rotmat: np.ndarray,
               staging_xy: np.ndarray, arm_tags: Sequence[str],
               n_yaw: int, max_grasps_per_probe: Optional[int],
               obstacle_count: int, policy_sig: str) -> str:
    try:
        st = os.stat(mesh_path)
        mesh_sig = f"{int(st.st_mtime)}:{int(st.st_size)}"
    except OSError:
        mesh_sig = "missing"
    parts = [
        "auto_rotmat_v3_graspable_upright",
        os.path.basename(mesh_path),
        mesh_sig,
        ",".join(f"{v:.4f}" for v in np.asarray(goal_pos).reshape(-1)),
        ",".join(f"{v:.4f}" for v in np.asarray(goal_rotmat).reshape(-1)),
        ",".join(f"{v:.4f}" for v in np.asarray(staging_xy).reshape(-1)),
        "|".join(arm_tags),
        f"yaw{n_yaw}",
        f"probe{max_grasps_per_probe}",
        f"obs{obstacle_count}",
        policy_sig,
    ]
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def auto_staging_rotmat_candidates(
    *,
    mesh_path: str,
    grasp_collection,
    goal_pos: np.ndarray,
    goal_rotmat: np.ndarray,
    staging_xy: np.ndarray,
    arms: Sequence[Tuple[str, object]],
    n_yaw: int = 8,
    top_k: int = 6,
    min_common_gids: int = 1,
    max_grasps_per_probe: Optional[int] = 128,
    obstacle_list: Optional[Sequence[object]] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    cache_tag: str = "",
    verbose: bool = False,
    # New generic policy knobs
    filter_flat_for_thin: bool = True,
    thin_aspect_ratio: float = 0.35,
    upright_height_ratio: float = 0.35,
) -> List[Tuple[np.ndarray, float]]:
    """Rank staging rotmat candidates and return top-k ``(R, z_off)``.

    The function remains task-agnostic. It never checks part names such as
    ``shelf``. Upright behavior is inferred from mesh geometry:

    * If the mesh is thin / plate-like, flat orientations are filtered out.
    * For every remaining orientation, yaw around world Z is enumerated.
    * If ``obstacle_list`` contains the work table, IK candidates whose gripper
      collides with the table are automatically rejected.
    """
    arm_tags = tuple(t for t, _ in arms)
    policy_sig = (
        f"flatfilter={int(filter_flat_for_thin)};"
        f"thin={thin_aspect_ratio:.3f};"
        f"upright={upright_height_ratio:.3f};"
        f"goalalign=0.120"
    )
    key = _cache_key(mesh_path=mesh_path, goal_pos=goal_pos,
                     goal_rotmat=goal_rotmat, staging_xy=staging_xy,
                     arm_tags=arm_tags, n_yaw=n_yaw,
                     max_grasps_per_probe=max_grasps_per_probe,
                     obstacle_count=0 if not obstacle_list else len(obstacle_list),
                     policy_sig=policy_sig)
    cache_path: Optional[str] = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_tag}__{key}.pkl")
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    cached = pickle.load(fh)
                if verbose:
                    print(f"  [auto_rotmat] cache hit ({cache_tag}): "
                          f"{len(cached)} candidates")
                return [(np.asarray(R, dtype=float), float(z))
                        for R, z in cached]
            except Exception:
                pass

    corners = _mesh_aabb_corners(mesh_path)
    raw_candidates = _candidate_rotmats(n_yaw=n_yaw)
    candidates = _filter_graspable_upright_candidates(
        raw_candidates,
        corners,
        filter_flat_for_thin=filter_flat_for_thin,
        thin_aspect_ratio=thin_aspect_ratio,
        upright_height_ratio=upright_height_ratio,
    )

    if verbose:
        dims = _local_aabb_size(corners)
        print(f"  [auto_rotmat:{cache_tag}] mesh_dims={np.round(dims, 4).tolist()} "
              f"thin={_is_thin_or_plate_like(corners, thin_aspect_ratio)} "
              f"raw={len(raw_candidates)} kept={len(candidates)} "
              f"n_yaw={n_yaw}")

    # Goal-side feasible set is independent of staging rotmat.
    G_goal_per_arm: dict = {}
    for tag, arm in arms:
        G_goal_per_arm[tag] = _feasible_grasp_ids(
            arm, grasp_collection,
            np.asarray(goal_pos, dtype=float),
            np.asarray(goal_rotmat, dtype=float),
            max_grasps=max_grasps_per_probe,
            obstacle_list=obstacle_list,
        )
    if verbose:
        print(f"  [auto_rotmat:{cache_tag}] G_goal per arm: " +
              ", ".join(f"{t}={len(G_goal_per_arm[t])}" for t in arm_tags))

    scored: List[_RotmatScore] = []
    for R_stg in candidates:
        z_off = _z_offset_for_flat_on_table(R_stg, corners)
        base_pos = np.array([float(staging_xy[0]),
                             float(staging_xy[1]),
                             float(z_off)])
        rot_size = _rotated_aabb_size(R_stg, corners)
        height = float(rot_size[2])
        footprint_area = float(max(rot_size[0], 1e-6) * max(rot_size[1], 1e-6))

        per_arm: dict = {}
        for tag, arm in arms:
            G_stg = _feasible_grasp_ids(
                arm, grasp_collection, base_pos, R_stg,
                max_grasps=max_grasps_per_probe,
                obstacle_list=obstacle_list,
            )
            per_arm[tag] = len(G_stg & G_goal_per_arm[tag])
        sm = max(per_arm.values()) if per_arm else 0
        ss = sum(per_arm.values())
        scored.append(_RotmatScore(
            R=R_stg.copy(),
            z_off=z_off,
            height=height,
            footprint_area=footprint_area,
            per_arm_common=per_arm,
            score_max=sm,
            score_sum=ss,
        ))

    # Thin plates: goal-aligned staging (same rot as assembly goal, lifted above
    # table) yields pure translation transport — critical for shelf pick→place.
    if _is_thin_or_plate_like(corners, thin_aspect_ratio):
        gr = np.asarray(goal_rotmat, dtype=float)
        yaw_angles = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
        goal_z_bump = 0.12
        seen = {tuple(np.round(s.R, 4).reshape(-1).tolist()) for s in scored}
        for ang in yaw_angles:
            R_yaw = rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(float(ang)))
            R_stg = R_yaw @ gr
            key = tuple(np.round(R_stg, 4).reshape(-1).tolist())
            if key in seen:
                continue
            seen.add(key)
            z_off = _z_offset_for_flat_on_table(R_stg, corners) + goal_z_bump
            base_pos = np.array([float(staging_xy[0]),
                                 float(staging_xy[1]),
                                 float(z_off)])
            rot_size = _rotated_aabb_size(R_stg, corners)
            height = float(rot_size[2])
            footprint_area = float(max(rot_size[0], 1e-6) * max(rot_size[1], 1e-6))
            per_arm: dict = {}
            for tag, arm in arms:
                G_stg = _feasible_grasp_ids(
                    arm, grasp_collection, base_pos, R_stg,
                    max_grasps=max_grasps_per_probe,
                    obstacle_list=obstacle_list,
                )
                per_arm[tag] = len(G_stg & G_goal_per_arm[tag])
            sm = max(per_arm.values()) if per_arm else 0
            ss = sum(per_arm.values())
            scored.append(_RotmatScore(
                R=R_stg.copy(),
                z_off=z_off,
                height=height,
                footprint_area=footprint_area,
                per_arm_common=per_arm,
                score_max=sm,
                score_sum=ss,
            ))

    # Main score: common grasp count. Tie-breakers:
    #   1. higher total common count,
    #   2. larger height (keeps plate-like parts upright),
    #   3. larger footprint area for more stable staging.
    scored.sort(key=lambda s: (-s.score_max, -s.score_sum,
                               -s.height, -s.footprint_area))

    if verbose:
        print(f"  [auto_rotmat:{cache_tag}] top {min(top_k, len(scored))}:")
        for s in scored[:top_k]:
            print(f"    {s}")

    out: List[Tuple[np.ndarray, float]] = []
    for s in scored:
        if s.score_max < min_common_gids:
            continue
        out.append((s.R.copy(), float(s.z_off)))
        if len(out) >= top_k:
            break

    # Last-resort fallback: if the collision/table filter is too strict, keep
    # top geometric upright candidates with zero common-gid score. This prevents
    # the caller from falling back to a flat pose.
    if not out and scored:
        for s in scored[:top_k]:
            out.append((s.R.copy(), float(s.z_off)))

    if cache_path is not None:
        with open(cache_path, "wb") as fh:
            pickle.dump([(R.tolist(), z) for R, z in out], fh)
    return out
