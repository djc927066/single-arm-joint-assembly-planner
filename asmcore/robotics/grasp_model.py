"""抓取记录与夹爪端口。

WRS3 原版里 `robotics/graspability.py` 直接吃 `wrs.grasp.grasp.Grasp` 和
`wrs.robots` 的 `MechBase`/`JawView`，还要靠 `adapters/wrs_scene.py` 把场景对象
转成 `Part`。本模块把这三样收敛成**一个**结构化协议，算法层因此不再触碰任何
`wrs.*`，换夹爪（Panthera / OR2FG7 / 任何两指夹爪）只需另写一个实现。

指标与算法一概未变：`GraspRecord` 字段与原 `Grasp.to_dict()` 中被用到的部分
一一对应（`pose`、`pre_pose`、`jaw_width`、`provenance`），`GripperPort` 暴露的
也正是原代码用到的那几个成员。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..model import FloatArray, Part, checked_tf, freeze, readonly


@dataclass(frozen=True, eq=False)
class GraspRecord:
    """一个候选抓取，位姿表达在**零件局部系**。

    对应 WRS3 `Grasp` 中被 graspability 用到的全部信息：
    `pose` 是夹爪 TCP 的目标位姿，`pre_pose` 是接近位姿，`jaw_width` 是标定后
    的开口指令，`provenance` 保留生成来源与接触点（`contacts_local_m`、
    `normals_on_object_local`），力闭合判定要用。
    """

    pose: FloatArray
    pre_pose: FloatArray
    jaw_width: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", checked_tf(self.pose))
        object.__setattr__(self, "pre_pose", checked_tf(self.pre_pose))
        if not np.isfinite(self.jaw_width) or self.jaw_width < 0:
            raise ValueError("jaw_width 必须是有限非负数")
        object.__setattr__(self, "jaw_width", float(self.jaw_width))
        object.__setattr__(self, "provenance", freeze(dict(self.provenance)))

    def to_dict(self) -> dict:
        return dict(
            pose=self.pose.tolist(),
            pre_pose=self.pre_pose.tolist(),
            jaw_width=self.jaw_width,
            provenance=_jsonable(self.provenance),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraspRecord":
        return cls(
            np.asarray(data["pose"], dtype=float),
            np.asarray(data["pre_pose"], dtype=float),
            float(data["jaw_width"]),
            data.get("provenance", {}),
        )

    def replace(self, **kw) -> "GraspRecord":
        base = dict(pose=self.pose, pre_pose=self.pre_pose,
                    jaw_width=self.jaw_width, provenance=dict(self.provenance))
        prov = kw.pop("provenance", None)
        if prov is not None:
            base["provenance"] = dict(base["provenance"], **dict(prov))
        base.update(kw)
        return GraspRecord(**base)


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


@runtime_checkable
class GripperPort(Protocol):
    """两指平行夹爪。实现方由 `adapters/` 提供，本包不依赖任何机器人框架。

    原 WRS3 代码用到的成员一一对应：
    `open_dir` ← `gripper.open_dir`，`jaw_range` ← `gripper.jaw_range`，
    `clone()` ← `gripper.clone()`，`grip_at()` ← `gripper.grip_at()`，
    `link_parts()` ← `part_from_scene_object(link) for link in runtime_lnks`。
    """

    @property
    def open_dir(self) -> FloatArray:
        """夹爪局部系下两指张开的单位方向。"""
        ...

    @property
    def jaw_range(self) -> tuple[float, float]:
        """(最小开口, 最大开口)，米。"""
        ...

    def clone(self) -> "GripperPort":
        """独立副本；`grip_at` 不得影响原对象。"""
        ...

    def grip_at(self, pos: FloatArray, rotmat: FloatArray, jaw_width: float) -> None:
        """把夹爪摆到该位姿并张到该开口。"""
        ...

    def link_parts(self) -> tuple[Part, ...]:
        """当前位姿下各连杆的网格，`assembled_tf` 为世界位姿。

        约定与 WRS3 的 `runtime_lnks` 一致：**第 0 个是掌部，其后是手指**。
        graspability 依赖这个次序：掌部不豁免贴合面、手指豁免。
        """
        ...


def finger_parts(gripper: GripperPort) -> tuple[Part, ...]:
    """手指连杆（跳过掌部），对应原代码的 `runtime_lnks[1:]`。"""
    return tuple(gripper.link_parts()[1:])


def gripper_signature(gripper: GripperPort) -> tuple:
    """进入分析器 binding 的夹爪指纹，等价于原版对连杆几何与开口的摘要。"""
    parts = gripper.link_parts()
    return (
        tuple((p.geometry.geometry_id, readonly(p.assembled_tf).tobytes().hex()) for p in parts),
        tuple(np.asarray(gripper.open_dir, dtype=float).tolist()),
        tuple(float(x) for x in gripper.jaw_range),
    )
