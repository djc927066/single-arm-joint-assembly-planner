"""面向机器人的一层：抓取质量 G。

从 WRS3 的 `wrs/assembly/robotics/` 迁移而来，算法与指标未改，夹爪改走
`grasp_model.GripperPort` 端口，因此本包不 import 任何 `wrs.*`。
Panthera 的端口实现在仓库根的 `adapters/`。

原 `execution.py`（WRS3 运行时的抓取—搬运—插入—交接回放，63 KB）没有迁移：
WRS1 下的等价物是 `sealp/primitives/direct_transport.py`。它里面被 G 指标用到
的几何helpers 已经抽到 `grasp_geometry.py`。完整原件见 `E:/Projects/WRS2-assembly-planner/wrs/assembly/robotics/`。

    from asmcore.robotics import GraspabilityAnalyzer, GraspabilityConfig
"""

from .grasp_model import GraspRecord, GripperPort, finger_parts, gripper_signature
from .grasp_geometry import (
    box_grasps,
    finger_force_contacts,
    generate_grasps,
    grasp_contacts,
)
from .graspability import GraspabilityAnalyzer, GraspabilityConfig, GraspabilityResult

__all__ = [
    "GraspRecord", "GripperPort", "finger_parts", "gripper_signature",
    "box_grasps", "generate_grasps", "grasp_contacts", "finger_force_contacts",
    "GraspabilityAnalyzer", "GraspabilityConfig", "GraspabilityResult",
]
