"""
Manipulability Scoring
========================

Utility functions for computing Yoshikawa manipulability at
given joint configurations or target TCP poses.

These wrap the WRS robot's built-in ``manipulability_val()`` and
``manipulability_mat()`` methods into layout-evaluation helpers.

The Yoshikawa manipulability measure is::

    w = sqrt(det(J @ J^T))

where ``J`` is the 6×n Jacobian.  Higher values indicate the robot
is further from singularity and has better dexterity.

Usage::

    from sealp.layout.manipulability import compute_manipulability_at_pose

    score = compute_manipulability_at_pose(robot, tgt_pos, tgt_rotmat)
    if score is not None:
        print(f"Manipulability: {score:.4f}")
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_manipulability(robot, jnt_values: np.ndarray) -> float:
    """Compute Yoshikawa manipulability at a given joint configuration.

    Parameters
    ----------
    robot : SglArmRobotInterface
        Robot arm with ``goto_given_conf()`` and ``manipulability_val()``.
    jnt_values : np.ndarray
        Joint configuration to evaluate.

    Returns
    -------
    float
        Manipulability value (≥ 0).  Returns 0 on error.
    """
    robot.backup_state()
    try:
        robot.goto_given_conf(jnt_values=jnt_values)
        return robot.manipulability_val()
    except Exception:
        return 0.0
    finally:
        robot.restore_state()


def compute_manipulability_at_pose(
    robot,
    tgt_pos: np.ndarray,
    tgt_rotmat: np.ndarray,
    seed_jnt_values: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Compute manipulability at a target TCP pose.

    Solves IK first, then evaluates manipulability at the solution.
    Returns ``None`` if IK has no solution.

    Parameters
    ----------
    robot : SglArmRobotInterface
    tgt_pos : np.ndarray
        Target TCP position.
    tgt_rotmat : np.ndarray
        Target TCP orientation.
    seed_jnt_values : np.ndarray or None
        Seed for IK solver.

    Returns
    -------
    float or None
        Manipulability value, or ``None`` if the pose is unreachable.
    """
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat,
                          seed_jnt_values=seed_jnt_values)
    if jnt_values is None:
        return None
    return compute_manipulability(robot, jnt_values)


def compute_manipulability_ellipsoid(
    robot,
    jnt_values: np.ndarray,
) -> Optional[tuple]:
    """Compute the manipulability ellipsoid at a configuration.

    Returns
    -------
    tuple of (linear_ellipsoid_mat, angular_ellipsoid_mat) or None
        Each is a 3×3 matrix whose columns are the scaled eigenvectors
        of the corresponding J·Jᵀ sub-block.
    """
    robot.backup_state()
    try:
        robot.goto_given_conf(jnt_values=jnt_values)
        return robot.manipulability_mat()
    except Exception:
        return None
    finally:
        robot.restore_state()
