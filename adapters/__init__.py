"""与机器人地基耦合的唯一一层。

`asmcore` 不 import 任何 `wrs.*`，`jbls` 只依赖下面这三个协议。所以换地基
（WRS1 -> WRS3）时只需重写本目录，算法层不动。该边界由
`tests/test_asmcore_standalone.py` 锁死。

当前地基：仓库根目录中的 WRS1 `wrs/`。它与 `asmcore/`、`jbls/` 平级，
因此从本仓库根目录运行或以 editable 方式安装本项目时可以直接 `import wrs`。

三个适配边界及其 WRS1 侧来源（`ArmProvider` 已实现，后两项仍缺正式实现）：

| 协议 | 来源 |
|---|---|
| `ArmProvider`      | `sealp/layout/layout_robot_factory.py`（已含单臂 `PantheraHTSglArm` + `_SingleArmRobotShell`） |
| `GraspFeasibility` | `sealp/examples/layout/bsfs/oracle.py` 的 L1（`PickPlacePlanner.reason_common_gids`）；数学部分见 `asmcore.mechanics.force_closure` |
| `MotionWitness`    | `sealp/primitives/direct_transport.py`（pick → transfer → insert → release → retreat） |
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class ArmProvider(Protocol):
    """提供单臂 Panthera-HT 实例及其 home 位形。"""

    def arm(self) -> Any:
        """返回带碰撞检查的机械臂对象。注意 `PantheraHTSglArm.__init__` 内部
        已调用 `setup_cc()`，重复调用会导致 `is_collided()` 误判全部抓取失败。"""
        ...

    def home_jnt_values(self) -> np.ndarray: ...

    def home_tcp_world(self) -> np.ndarray | None:
        """home 位形下的世界 TCP，用于空臂往返的代价项。"""
        ...


class GraspFeasibility(Protocol):
    """论文一 L1 层：两端 IK + 共同抓取推理。"""

    def common_grasp_ids(
        self,
        part_id: str,
        pick_pose: tuple[np.ndarray, np.ndarray],
        place_pose: tuple[np.ndarray, np.ndarray],
        obstacles: Sequence[Any],
    ) -> list[int]:
        """返回在取放两端都可达、且保持同一物体相对抓取的抓取 id。
        空列表 = 确定性失败，按论文一假设 A7 可以硬剪枝。"""
        ...


class MotionWitness(Protocol):
    """论文一 L3 层：完整运动证据。"""

    def transport(
        self,
        part_id: str,
        pick_pose: tuple[np.ndarray, np.ndarray],
        place_pose: tuple[np.ndarray, np.ndarray],
        grasp_ids: Sequence[int],
        obstacles: Sequence[Any],
    ) -> Any | None:
        """成功返回轨迹，失败返回 None。

        重要：RRT 超时**不是**不可行性的证明（论文一假设 A7）。调用方必须把
        超时与确定性几何/IK 失败区分开，前者不得用于硬剪枝。"""
        ...


# 具体实现放在独立模块中，避免只导入协议时就加载 Panda3D/WRS。
from .panthera import PantheraArmProvider

__all__ = [
    "ArmProvider",
    "GraspFeasibility",
    "MotionWitness",
    "PantheraArmProvider",
]
