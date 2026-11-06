"""Panthera-HT 单臂地基适配器。

WRS/Panda3D 在首次调用 :meth:`arm` 时才加载，纯数学代码可以继续独立使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def _vec3(value: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


def _rot3(value: ArrayLike) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3, 3)
        or not np.all(np.isfinite(result))
        or not np.allclose(result.T @ result, np.eye(3), atol=1e-8, rtol=0)
        or not np.isclose(np.linalg.det(result), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("base_rotmat must be a right-handed rotation matrix")
    return result.copy()


@dataclass
class PantheraArmProvider:
    """Create and own exactly one collision-enabled Panthera-HT single arm.

    ``home`` defaults to the WRS Panthera model's canonical six-zero pose.
    The returned TCP transform is in the WRS world frame.
    """

    base_pos: ArrayLike = field(default_factory=lambda: np.zeros(3))
    base_rotmat: ArrayLike = field(default_factory=lambda: np.eye(3))
    home: ArrayLike = field(default_factory=lambda: np.zeros(6))
    enable_cc: bool = True
    name: str = "single_arm_joint_planner_panthera"
    _arm_instance: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_pos = _vec3(self.base_pos, "base_pos")
        self.base_rotmat = _rot3(self.base_rotmat)
        home = np.asarray(self.home, dtype=float)
        if home.shape != (6,) or not np.all(np.isfinite(home)):
            raise ValueError("home must be a finite six-joint vector")
        self.home = home.copy()
        if not isinstance(self.enable_cc, bool):
            raise TypeError("enable_cc must be bool")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be nonempty")

    def arm(self) -> Any:
        """Return the owned Panthera arm, creating it on first use."""
        if self._arm_instance is None:
            from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import (
                PantheraHTSglArm,
            )

            self._arm_instance = PantheraHTSglArm(
                pos=self.base_pos,
                rotmat=self.base_rotmat,
                name=self.name,
                enable_cc=self.enable_cc,
            )
            # PantheraHTSglArm.__init__ has already called setup_cc().
            self._arm_instance.goto_given_conf(self.home)
        return self._arm_instance

    def home_jnt_values(self) -> np.ndarray:
        """Return an independent copy of the canonical home joint vector."""
        return np.asarray(self.home, dtype=float).copy()

    def home_tcp_world(self) -> np.ndarray:
        """Return the home TCP pose as a 4x4 local-to-world transform."""
        pos, rotmat = self.arm().fk(self.home_jnt_values())
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(rotmat, dtype=float)
        transform[:3, 3] = np.asarray(pos, dtype=float)
        return transform
