"""asmcore 需要、但自己不实现的能力（端口）。

asmcore 不 import 任何 `wrs.*`，所以凡是需要真实机器人与抓取数据的地方都在这里
声明成结构化协议（`typing.Protocol`），由仓库根的 `adapters/` 用 WRS1 实现。
实现方无需继承本模块的任何类，只要成员对得上即可。

`GraspabilityPort` 是论文二 Algorithm 1（`plan_quality_sequence`）唯一的外部依赖。
原 WRS3 实现绑定 WRS3 的 Grasp / MechBase / JawView 与 execution.py，因此没有
vendored，完整副本见 `E:/Projects/WRS2-assembly-planner/wrs/assembly/robotics/graspability.py`，可作移植参考。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..model import Assembly, AssemblyState


class GraspabilityResultLike(Protocol):
    """`GraspabilityPort.analyze` 的返回值。

    `count` 是当前装配状态下该零件的合格抓取数，即论文二的 G 质量：
    `None` 表示未能判定（调用方记入 unresolved，不等同于 0）；
    `0` 表示确无合格抓取（剪枝）。
    """

    count: int | None


@runtime_checkable
class GraspabilityPort(Protocol):
    """统计某零件在当前已装配障碍环境下的合格抓取数。"""

    assembly: Assembly
    """所绑定的装配体。`plan_quality_sequence` 会用 digest 校验是否同一个。"""

    binding: str
    """标识本分析器配置的摘要，进入搜索的缓存键，配置变了必须随之改变。"""

    def analyze(self, state: AssemblyState, part_id: str) -> GraspabilityResultLike:
        """`state` 是该零件装入后的已安装状态（障碍环境由它决定）。"""
        ...
