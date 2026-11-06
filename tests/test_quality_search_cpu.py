"""论文二 Algorithm 1（`plan_quality_sequence`）可用性与纯 CPU 配置。

背景：WRS3 原包里这个入口依赖 `robotics/graspability.py`，而那个文件绑定 WRS3
运行时、没有 vendored。asmcore 改为在 `planning/ports.py` 声明 `GraspabilityPort`
结构化协议，由 `adapters/` 用 WRS1 实现；本文件锁死这条契约。

另一个坑：`StabilitySweepConfig.backend` 默认是 `"cuda"`（需要 torch+CUDA），
`QualitySearchConfig.direction.method` 默认是 `"socp"`（需要 clarabel）。两者都
不在上游的基础依赖里。本文件同时锁死"纯 CPU 也能跑"的配置组合。
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from asmcore.mechanics.stability_sweep import StabilitySweepConfig
from asmcore.model import Assembly, MeshData, Part, digest
from asmcore.motion.directions import DirectionConfig
from asmcore.planning.ports import GraspabilityPort
from asmcore.planning.quality_search import QualitySearchConfig, plan_quality_sequence


def box(sx, sy, sz):
    h = np.array([sx, sy, sz], dtype=float) / 2.0
    v = np.array([[x, y, z] for x in (-h[0], h[0])
                  for y in (-h[1], h[1]) for z in (-h[2], h[2])], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int64)
    return MeshData(v, f)


def tf(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


class StubGraspability:
    """满足 `GraspabilityPort` 的最小实现，供算法层单测使用。

    真实实现在 `adapters/`，用 WRS1 的抓取集合 + `reason_common_gids` 统计。
    """

    def __init__(self, assembly: Assembly, count: int = 8) -> None:
        self.assembly = assembly
        self.binding = digest(("stub_graspability/1", count))
        self._count = count

    def analyze(self, state, part_id):
        return SimpleNamespace(count=self._count)


@pytest.fixture(scope="module")
def assembly():
    return Assembly((
        Part("table", box(1.0, 1.0, 0.02), tf(0, 0, -0.01),
             mass_kg=50.0, com_local_m=np.zeros(3), friction=0.5, fixed=True),
        Part("post_a", box(0.06, 0.06, 0.20), tf(-0.10, 0, 0.10),
             mass_kg=0.5, com_local_m=np.zeros(3), friction=0.5),
        Part("post_b", box(0.06, 0.06, 0.20), tf(+0.10, 0, 0.10),
             mass_kg=0.5, com_local_m=np.zeros(3), friction=0.5),
        Part("lintel", box(0.32, 0.06, 0.04), tf(0.0, 0, 0.22),
             mass_kg=0.4, com_local_m=np.zeros(3), friction=0.5),
    ))


def cpu_config(**kw):
    """纯 CPU 配置：numpy 单纯形 + fibonacci 方向采样，不需要 torch/clarabel。"""
    return QualitySearchConfig(
        direction=DirectionConfig(method="fibonacci"),
        sweep=StabilitySweepConfig(backend="numpy"),
        time_limit_s=kw.pop("time_limit_s", 600.0),
        **kw,
    )


def test_stub_satisfies_graspability_port(assembly):
    """端口是结构化协议，实现方不需要继承任何东西。"""
    assert isinstance(StubGraspability(assembly), GraspabilityPort)


def test_default_config_still_requests_gpu_and_socp():
    """锁死上游默认值，避免有人误以为开箱即用就是纯 CPU。"""
    assert StabilitySweepConfig().backend == "cuda"
    assert QualitySearchConfig().direction.method == "socp"


def test_quality_search_runs_on_cpu(assembly):
    """论文二的剪枝搜索应在纯 CPU 下跑通，并把横梁排到最后。"""
    result = plan_quality_sequence(assembly, StubGraspability(assembly), config=cpu_config())
    assert result.status == "success", dict(result.diagnostics)
    plan = getattr(result, "plan", None)
    assert plan is not None
    order = [step.part_id for step in plan.assembly_steps]
    assert order[-1] == "lintel", f"横梁必须最后装, 实得 {order}"
    assert result.diagnostics["search_complete"] is True
    assert result.diagnostics["pruned"] >= 1, "不稳定分支应当被剪掉"
    assert "torch" not in sys.modules and "clarabel" not in sys.modules


def test_quality_search_rejects_foreign_analyzer(assembly):
    """分析器必须绑定同一个装配体，否则缓存键会失真。"""
    other = Assembly((
        Part("solo", box(0.05, 0.05, 0.05), tf(0, 0, 0.025),
             mass_kg=0.2, com_local_m=np.zeros(3), friction=0.5),
    ))
    with pytest.raises(ValueError, match="another assembly"):
        plan_quality_sequence(assembly, StubGraspability(other), config=cpu_config())
