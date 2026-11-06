"""
Robot Factory
==============

Extensible factory for creating robot instances by type name.
Currently supports Piper; add new robots by registering them
in ``ROBOT_REGISTRY``.

Usage::

    from sealp.editor.robot_factory import create_robot, ROBOT_REGISTRY
    robot = create_robot("piper", pos=np.zeros(3), rotmat=np.eye(3))
"""

from __future__ import annotations

from typing import Callable, Dict, Optional
import numpy as np


# Registry: robot_type → factory function
# Each factory: (pos, rotmat, **kwargs) → SglArmRobotInterface
ROBOT_REGISTRY: Dict[str, Callable] = {}


def register_robot(robot_type: str):
    """Decorator to register a robot factory function."""
    def decorator(fn):
        ROBOT_REGISTRY[robot_type] = fn
        return fn
    return decorator


@register_robot("piper")
def _create_piper(pos=np.zeros(3), rotmat=np.eye(3), **kwargs):
    from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
    return PiperSglArm(pos=pos, rotmat=rotmat, enable_cc=True)


def create_robot(robot_type: str, pos=None, rotmat=None, **kwargs):
    """Create a robot instance by type name.

    Parameters
    ----------
    robot_type : str
        Key in ``ROBOT_REGISTRY`` (e.g. ``"piper"``).
    pos : np.ndarray, optional
        Base world position (default: origin).
    rotmat : np.ndarray, optional
        Base world orientation (default: identity).

    Returns
    -------
    SglArmRobotInterface
    """
    if pos is None:
        pos = np.zeros(3)
    if rotmat is None:
        rotmat = np.eye(3)
    if robot_type not in ROBOT_REGISTRY:
        available = ", ".join(sorted(ROBOT_REGISTRY.keys()))
        raise ValueError(
            f"Unknown robot type {robot_type!r}. "
            f"Available: {available}")
    return ROBOT_REGISTRY[robot_type](pos=pos, rotmat=rotmat, **kwargs)


def available_robots():
    """Return list of registered robot type names."""
    return sorted(ROBOT_REGISTRY.keys())
