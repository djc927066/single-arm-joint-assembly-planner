"""供算法层单测使用的两指平行夹爪桩件。

只是一个满足 `asmcore.robotics.GripperPort` 的最小几何模型：掌部 + 两片指垫。
真机实现见 `adapters/panthera_gripper.py`；两者走同一个端口，所以本桩件验证的
是迁移后的 G 指标算法本身，与具体夹爪无关。

局部系约定与 WRS 一致：+Z 是接近方向，`open_dir` 沿 +X 张开，TCP 在原点。
"""
from __future__ import annotations

import numpy as np

from asmcore.model import MeshData, Part


def _box(sx, sy, sz):
    h = np.array([sx, sy, sz], dtype=float) / 2.0
    v = np.array([[x, y, z] for x in (-h[0], h[0])
                  for y in (-h[1], h[1]) for z in (-h[2], h[2])], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
    return MeshData(v, f)


def _tf(pos, rotmat=None):
    m = np.eye(4)
    if rotmat is not None:
        m[:3, :3] = rotmat
    m[:3, 3] = pos
    return m


class StubParallelGripper:
    """掌部 + 两片指垫，指垫内表面恰好落在 ±jaw_width/2。"""

    PALM = (0.090, 0.050, 0.030)
    PALM_Z = -0.080          # 掌部中心相对 TCP 的位置（沿 -Z 后退）
    PAD = (0.008, 0.030, 0.040)

    def __init__(self, jaw_range=(0.0, 0.090)):
        self._jaw_range = (float(jaw_range[0]), float(jaw_range[1]))
        self._pos = np.zeros(3)
        self._rot = np.eye(3)
        self._width = float(jaw_range[1])

    # ---- GripperPort ----
    @property
    def open_dir(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    @property
    def jaw_range(self) -> tuple[float, float]:
        return self._jaw_range

    def clone(self) -> "StubParallelGripper":
        other = StubParallelGripper(self._jaw_range)
        other._pos = self._pos.copy()
        other._rot = self._rot.copy()
        other._width = self._width
        return other

    def grip_at(self, pos, rotmat, jaw_width) -> None:
        self._pos = np.asarray(pos, dtype=float).reshape(3).copy()
        self._rot = np.asarray(rotmat, dtype=float).reshape(3, 3).copy()
        self._width = float(jaw_width)

    def link_parts(self) -> tuple[Part, ...]:
        base = _tf(self._pos, self._rot)
        half = self._width / 2.0
        out = [Part("palm", _box(*self.PALM), base @ _tf([0.0, 0.0, self.PALM_Z]))]
        for sign, name in ((-1.0, "finger_0"), (1.0, "finger_1")):
            # 指垫内表面正好在 ±half：中心再外移半个垫厚
            cx = sign * (half + self.PAD[0] / 2.0)
            out.append(Part(name, _box(*self.PAD), base @ _tf([cx, 0.0, 0.0])))
        return tuple(out)
