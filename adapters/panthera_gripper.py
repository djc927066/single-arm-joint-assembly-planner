"""Panthera 夹爪的 `asmcore.robotics.GripperPort` 实现。

这是论文二 G 指标在本项目里唯一需要改的东西：算法、判据、容差全部照搬，
夹爪从 WRS3 的 MechBase/JawView 换成 WRS1 的 `PantheraGripper`。

WRS1 侧的事实（读自 `wrs/robot_sim/end_effectors/grippers/panthera_gripper/`）：

* `jaw_range = [0.0, 0.085]` 米
* 两指沿 **±Y** 平动（`jnts[0].loc_motion_ax = +y`，`jnts[1] = -y`），
  不是 X —— 所以 `open_dir` 取 +Y，并在构造时按实测指间向量复核一次
* TCP（acting center）在夹爪基座 +Z 方向 0.17 m 处，`loc_acting_center_rotmat`
  是单位阵
* 三个连杆：`jlc.anchor.lnk_list[0]` 掌部，`jlc.jnts[0].lnk` 与 `jlc.jnts[1].lnk`
  两指。`link_parts()` 按「掌部在前、手指在后」返回，与 WRS3 的
  `runtime_lnks` 次序一致（graspability 靠这个次序豁免指垫贴合面）

WRS/Panda3D 只在首次调用时加载，纯数学代码可以继续独立使用。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from asmcore.model import MeshData, Part


def _link_mesh(lnk: Any) -> tuple[np.ndarray, np.ndarray]:
    """取连杆的**局部**三角网格。"""
    cmodel = getattr(lnk, "cmodel", None)
    if cmodel is None:
        raise ValueError("连杆没有 cmodel，无法取网格")
    mesh = getattr(cmodel, "trm_mesh", None)
    if mesh is None:
        raise ValueError("cmodel 没有 trm_mesh")
    return (np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64))


def _link_pose(lnk: Any) -> np.ndarray:
    """连杆的世界位姿 4×4。"""
    pos = np.asarray(getattr(lnk, "gl_pos"), dtype=float).reshape(3)
    rot = np.asarray(getattr(lnk, "gl_rotmat"), dtype=float).reshape(3, 3)
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = pos
    return tf


class PantheraGripperPort:
    """满足 `asmcore.robotics.GripperPort` 的 Panthera 夹爪。

    `grip_at` 收到的是 **TCP 位姿**；内部换算成夹爪基座位姿再 `fix_to`。
    传入什么坐标系，`link_parts()` 就在什么坐标系返回 —— graspability 在零件
    局部系里调用它，因此返回的也是零件局部系下的连杆位姿。
    """

    def __init__(self, *, jaw_range=None, name: str = "panthera_graspability") -> None:
        self._jaw_range_override = None if jaw_range is None else (
            float(jaw_range[0]), float(jaw_range[1]))
        self._name = name
        self._gripper = None
        self._open_dir = None

    # Panthera 的 STL 指垫带倒角，而且关节原点到实际接触面的标定不是理想解析
    # 关系。实测 Bridge 方柱四个侧夹在宽度校准后的最大点残差为 0.239 mm；
    # 要从带倒角的三角网格中取到完整平面片，并覆盖同一指垫被 SDF 报告的最大
    # 0.995 mm 局部干涉，需要 1.0 mm 的模型带宽。它们是夹爪 CAD/运动学模型
    # 的就位容差，不是环境碰撞容差，也不
    # 替代后面的 Force Closure 检查。
    pad_seating_tolerance_m = 0.5e-3
    pad_patch_tolerance_m = 1.0e-3

    # ---- 惰性构造 ----
    def _ee(self):
        if self._gripper is None:
            from wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper import (
                PantheraGripper,
            )

            kw = {"name": self._name}
            if self._jaw_range_override is not None:
                kw["jaw_range"] = np.asarray(self._jaw_range_override, dtype=float)
            self._gripper = PantheraGripper(**kw)
        return self._gripper

    def _links(self):
        ee = self._ee()
        return [ee.jlc.anchor.lnk_list[0], ee.jlc.jnts[0].lnk, ee.jlc.jnts[1].lnk]

    # ---- GripperPort ----
    @property
    def open_dir(self) -> np.ndarray:
        """TCP 系下的张开方向。首次访问时按实测指间向量标定并缓存。"""
        if self._open_dir is None:
            self._open_dir = self._calibrate_open_dir()
        return self._open_dir

    def _calibrate_open_dir(self) -> np.ndarray:
        """量两个开口下两指原点的连线，换算到 TCP 系，取单位向量。

        Panthera 的两指沿 ±Y 平动，预期结果就是 (0, 1, 0)；这里实测一次是为了
        在夹爪模型被改动时立刻暴露问题，而不是让 G 指标悄悄算错。
        """
        ee = self._ee()
        lo, hi = float(ee.jaw_range[0]), float(ee.jaw_range[1])
        wide, narrow = hi * 0.9, hi * 0.3
        saved = ee.get_jaw_width()
        try:
            ee.change_jaw_width(wide)
            a = _link_pose(self._links()[1])[:3, 3] - _link_pose(self._links()[2])[:3, 3]
            ee.change_jaw_width(narrow)
            b = _link_pose(self._links()[1])[:3, 3] - _link_pose(self._links()[2])[:3, 3]
        finally:
            ee.change_jaw_width(saved)
        delta = a - b                      # 张大时指间向量的增量方向
        size = float(np.linalg.norm(delta))
        if size < 1e-9:
            raise ValueError("无法标定张开方向：改变开口时两指原点没有相对移动")
        world = delta / size
        rot = np.asarray(ee.rotmat, dtype=float) @ np.asarray(
            ee.loc_acting_center_rotmat, dtype=float)
        local = rot.T @ world
        local = local / np.linalg.norm(local)
        # 与 ±Y 对齐时消掉数值噪声，保证是干净的单位轴
        axis = int(np.argmax(np.abs(local)))
        if abs(local[axis]) > 0.999:
            clean = np.zeros(3)
            clean[axis] = np.sign(local[axis])
            return clean
        return local

    @property
    def jaw_range(self) -> tuple[float, float]:
        r = np.asarray(self._ee().jaw_range, dtype=float)
        return (float(r[0]), float(r[1]))

    def clone(self) -> "PantheraGripperPort":
        """独立实例。WRS1 的 EE 对象没有深拷贝，这里重新构造一个。"""
        other = PantheraGripperPort(jaw_range=self._jaw_range_override, name=self._name)
        other._open_dir = None if self._open_dir is None else self._open_dir.copy()
        return other

    def grip_at(self, pos, rotmat, jaw_width) -> None:
        ee = self._ee()
        pos = np.asarray(pos, dtype=float).reshape(3)
        rotmat = np.asarray(rotmat, dtype=float).reshape(3, 3)
        # TCP 位姿 -> 基座位姿
        acting_rot = np.asarray(ee.loc_acting_center_rotmat, dtype=float)
        acting_pos = np.asarray(ee.loc_acting_center_pos, dtype=float)
        base_rot = rotmat @ acting_rot.T
        base_pos = pos - base_rot @ acting_pos
        ee.fix_to(base_pos, base_rot)
        lo, hi = self.jaw_range
        ee.change_jaw_width(float(np.clip(jaw_width, lo, hi)))

    def link_parts(self) -> tuple[Part, ...]:
        names = ("palm", "finger_0", "finger_1")
        out = []
        for name, lnk in zip(names, self._links()):
            vertices, faces = _link_mesh(lnk)
            out.append(Part(name, MeshData(vertices, faces), _link_pose(lnk)))
        return tuple(out)
