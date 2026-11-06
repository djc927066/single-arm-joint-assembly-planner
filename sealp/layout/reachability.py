"""
IK Reachability Checking
==========================

Utility functions to check whether a robot can reach target poses
(TCP position + orientation) and whether the resulting configuration
is collision-free.

These functions wrap the WRS robot's ``.ik()`` and ``.is_collided()``
API into simple, layout-evaluation-friendly interfaces.

Usage::

    from sealp.layout.reachability import check_ik_reachability

    result = check_ik_reachability(
        robot, tgt_pos, tgt_rotmat, obstacle_list=[ground])
    if result.reachable:
        print(f"IK solution: {result.jnt_values}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════
#  Result data classes
# ══════════════════════════════════════════════════════════════
@dataclass
class IKResult:
    """Result of a single IK reachability check.

    Attributes
    ----------
    reachable : bool
        Whether a valid IK solution exists.
    collision_free : bool
        Whether the IK solution is collision-free.
    jnt_values : np.ndarray or None
        Joint values of the IK solution (if found).
    manipulability : float
        Manipulability score at the IK solution (0 if not reachable).
    """
    reachable: bool = False
    collision_free: bool = False
    jnt_values: Optional[np.ndarray] = None
    manipulability: float = 0.0


@dataclass
class PoseReachabilityResult:
    """Result of checking reachability to a pose with multiple grasps.

    Attributes
    ----------
    reachable : bool
        Whether at least one grasp + IK solution reaches the pose.
    n_reachable_grasps : int
        Number of grasps with valid IK solutions.
    n_collision_free : int
        Number of grasps with collision-free IK solutions.
    n_total_grasps : int
        Total number of grasps evaluated.
    best_jnt_values : np.ndarray or None
        Joint values of the best (highest manipulability) solution.
    best_manipulability : float
        Manipulability of the best solution.
    best_grasp_id : int or None
        Index of the best grasp in the collection.
    """
    reachable: bool = False
    n_reachable_grasps: int = 0
    n_collision_free: int = 0
    n_total_grasps: int = 0
    best_jnt_values: Optional[np.ndarray] = None
    best_manipulability: float = 0.0
    best_grasp_id: Optional[int] = None


# ══════════════════════════════════════════════════════════════
#  IK reachability (single pose, no grasps)
# ══════════════════════════════════════════════════════════════
def check_ik_reachability(
    robot,
    tgt_pos: np.ndarray,
    tgt_rotmat: np.ndarray,
    seed_jnt_values: Optional[np.ndarray] = None,
    obstacle_list: Optional[List] = None,
    check_collision: bool = True,
) -> IKResult:
    """Check if a robot arm can reach a target TCP pose.

    Parameters
    ----------
    robot : SglArmRobotInterface
        Robot arm with ``.ik()``, ``.goto_given_conf()``,
        ``.is_collided()``, and ``.manipulability_val()`` methods.
    tgt_pos : np.ndarray
        Target TCP position ``[x, y, z]``.
    tgt_rotmat : np.ndarray
        Target TCP orientation (3×3).
    seed_jnt_values : np.ndarray or None
        Seed for the IK solver.
    obstacle_list : list or None
        Collision obstacles.
    check_collision : bool
        Whether to check collision after IK.

    Returns
    -------
    IKResult
    """
    if obstacle_list is None:
        obstacle_list = []

    # Try IK
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat,
                          seed_jnt_values=seed_jnt_values)
    if jnt_values is None:
        return IKResult(reachable=False)

    # Move robot to IK solution and check collision
    robot.backup_state()
    try:
        robot.goto_given_conf(jnt_values=jnt_values)

        collision_free = True
        if check_collision and obstacle_list:
            collision_free = not robot.is_collided(
                obstacle_list=obstacle_list)

        # Compute manipulability at this configuration
        manip = robot.manipulability_val()

        return IKResult(
            reachable=True,
            collision_free=collision_free,
            jnt_values=jnt_values.copy(),
            manipulability=manip,
        )
    finally:
        robot.restore_state()


# ══════════════════════════════════════════════════════════════
#  Pose reachability with grasp collection
# ══════════════════════════════════════════════════════════════
def check_pose_reachability(
    robot,
    obj_pos: np.ndarray,
    obj_rotmat: np.ndarray,
    grasp_collection,
    obstacle_list: Optional[List] = None,
    max_grasps: int = 50,
) -> PoseReachabilityResult:
    """Check how many grasps can reach an object at a given pose.

    For each grasp in the collection, computes the TCP pose from the
    grasp's ``ac_pos`` / ``ac_rotmat`` and the object's world pose,
    then checks IK reachability.

    Parameters
    ----------
    robot : SglArmRobotInterface
        Robot arm.
    obj_pos : np.ndarray
        Object world position.
    obj_rotmat : np.ndarray
        Object world orientation (3×3).
    grasp_collection
        Iterable of grasps with ``.ac_pos`` and ``.ac_rotmat``.
    obstacle_list : list or None
        Collision obstacles.
    max_grasps : int
        Maximum number of grasps to evaluate (for speed).

    Returns
    -------
    PoseReachabilityResult
    """
    if obstacle_list is None:
        obstacle_list = []

    best_manip = -1.0
    best_jnts = None
    best_gid = None
    n_reachable = 0
    n_cfree = 0
    n_total = min(len(grasp_collection), max_grasps)

    for gid in range(n_total):
        grasp = grasp_collection[gid]

        # Compute TCP pose from grasp + object pose
        # tcp_pos = obj_rotmat @ ac_pos + obj_pos
        # tcp_rotmat = obj_rotmat @ ac_rotmat
        tcp_pos = obj_rotmat @ grasp.ac_pos + obj_pos
        tcp_rotmat = obj_rotmat @ grasp.ac_rotmat

        result = check_ik_reachability(
            robot=robot,
            tgt_pos=tcp_pos,
            tgt_rotmat=tcp_rotmat,
            obstacle_list=obstacle_list,
            check_collision=True,
        )

        if result.reachable:
            n_reachable += 1
            if result.collision_free:
                n_cfree += 1
                if result.manipulability > best_manip:
                    best_manip = result.manipulability
                    best_jnts = result.jnt_values
                    best_gid = gid

    return PoseReachabilityResult(
        reachable=n_reachable > 0,
        n_reachable_grasps=n_reachable,
        n_collision_free=n_cfree,
        n_total_grasps=n_total,
        best_jnt_values=best_jnts,
        best_manipulability=max(best_manip, 0.0),
        best_grasp_id=best_gid,
    )
