"""论文二 G 指标（Graspability Analyzer）迁移后的行为锁定。

用桩夹爪在 Bridge 基准场景上跑，验证的是**迁移后的算法本身**没有走样；
换成 Panthera 只是换一个 `GripperPort` 实现，算法路径完全相同。
Panthera 侧的实机校验见 `test_panthera_graspability.py`（需要 WRS1）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from asmcore.model import AssemblyState
from asmcore.planning.ports import GraspabilityPort
from asmcore.robotics import (
    GraspabilityAnalyzer,
    GraspabilityConfig,
    GraspRecord,
    GripperPort,
    box_grasps,
    generate_grasps,
)
from stub_gripper import StubParallelGripper
from tasks.bridge import build_bridge, movable_ids

MOVABLE = movable_ids()


@pytest.fixture(scope="module")
def assembly():
    return build_bridge()


@pytest.fixture(scope="module")
def gripper():
    return StubParallelGripper()


@pytest.fixture(scope="module")
def analyzer(assembly, gripper):
    return GraspabilityAnalyzer(assembly, gripper, config=GraspabilityConfig(max_grasps=12))


def state_of(assembly, ids):
    fixed = {p.part_id for p in assembly.parts if p.fixed}
    full = assembly.initial_state()
    return AssemblyState({k: v for k, v in full.poses.items() if k in set(ids) | fixed})


def test_stub_satisfies_gripper_port(gripper):
    assert isinstance(gripper, GripperPort)


def test_analyzer_satisfies_quality_search_port(analyzer):
    """G 分析器可以直接喂给 plan_quality_sequence。"""
    assert isinstance(analyzer, GraspabilityPort)


def test_box_grasps_respect_jaw_range(assembly, gripper):
    """长方体侧夹：开口超过夹爪量程的朝向必须被丢掉。"""
    parts = {p.part_id: p for p in assembly.parts}
    assert len(box_grasps(parts["post_l"], gripper)) == 4      # 50 mm 见方，四个朝向都行
    # 横梁 340 mm 长，沿长轴的两个朝向超出 90 mm 开口
    assert len(box_grasps(parts["beam"], gripper)) == 2


def test_calibrated_width_matches_part_width(assembly, gripper):
    """垫片标定后的开口应等于零件在张开方向上的实际宽度。"""
    parts = {p.part_id: p for p in assembly.parts}
    for pid, expected in (("post_l", 0.050), ("beam", 0.060), ("load", 0.060)):
        widths = [g.jaw_width for g in generate_grasps(parts[pid], gripper, max_grasps=12)]
        assert widths, pid
        assert np.allclose(widths, expected, atol=1e-6), (pid, widths)


def test_force_closure_is_certified(analyzer, assembly):
    """每个被接受的抓取都必须有 rank 6 的力闭合证书，而不是仅仅对踵。"""
    result = analyzer.analyze(state_of(assembly, ["post_l"]), "post_l")
    assert result.status == "complete"
    assert result.count == 4
    for c in result.diagnostics["candidates"]:
        assert c["status"] == "force_closed"
        assert c["closure"]["rank"] == 6
        assert c["closure"]["margin"] > 0


def test_graspability_depends_on_assembled_environment(analyzer, assembly):
    """G 是**当前已装配环境**下的合格抓取数，这是论文二 G 指标的定义。"""
    alone = analyzer.analyze(state_of(assembly, ["post_l"]), "post_l")
    crowded = analyzer.analyze(state_of(assembly, MOVABLE), "post_l")
    assert alone.count == 4
    assert crowded.count == 0, "横梁装上之后夹爪够不到支柱，G 必须归零"
    assert crowded.catalogue_size == alone.catalogue_size, "目录是零件固有的，不随环境变"
    reasons = {c["status"] for c in crowded.diagnostics["candidates"]}
    assert reasons == {"environment_collision"}


def test_top_part_stays_graspable_when_assembled(analyzer, assembly):
    """载荷在最上层，装满之后仍然可抓。"""
    assert analyzer.analyze(state_of(assembly, MOVABLE), "load").count == 2


def test_results_are_cached_and_bound(analyzer, assembly):
    state = state_of(assembly, ["post_l"])
    first = analyzer.analyze(state, "post_l")
    assert analyzer.analyze(state, "post_l") is first
    grasps = analyzer.accepted_grasps(first)
    assert len(grasps) == first.count
    assert all(isinstance(g, GraspRecord) for g in grasps)


def test_rejects_absent_part(analyzer, assembly):
    with pytest.raises(ValueError, match="absent/unknown"):
        analyzer.analyze(state_of(assembly, ["post_l"]), "beam")


def test_no_wrs_and_no_heavy_deps_pulled_in():
    """在隔离解释器中验证算法层不会拉入 wrs / torch / clarabel。

    整套测试还会主动测试 Panthera，因此不能用当前 pytest 进程的
    ``sys.modules`` 判断；测试文件顺序不应改变这个不变量的结果。
    """
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; import asmcore.robotics; "
        "bad=[m for m in sys.modules if m.split('.')[0] in {'wrs','torch','clarabel'}]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
