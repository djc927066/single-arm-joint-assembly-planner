"""Panthera 夹爪接入 G 指标的实机校验（需要 WRS1 + Panda3D）。

算法本身由 `test_graspability_migration.py` 用桩夹爪锁定；本文件只验证
`adapters/panthera_gripper.PantheraGripperPort` 这个端口实现是否接对了：
张开方向、开口量程、TCP 换算、连杆网格与位姿。

在没有 WRS1 的环境（例如纯算法 CI）里整体跳过。
"""
from __future__ import annotations

import numpy as np
import pytest

from asmcore.robotics import GraspabilityAnalyzer, GraspabilityConfig, GripperPort, box_grasps
from tasks.bridge import build_bridge

wrs = pytest.importorskip("wrs", reason="需要仓库根目录下的 WRS1 地基")

from adapters.panthera_gripper import PantheraGripperPort  # noqa: E402


@pytest.fixture(scope="module")
def gripper():
    return PantheraGripperPort()


def test_satisfies_gripper_port(gripper):
    assert isinstance(gripper, GripperPort)


def test_jaw_range_matches_hardware(gripper):
    lo, hi = gripper.jaw_range
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.085, abs=1e-6), "Panthera 开口量程应为 0–85 mm"


def test_open_direction_is_the_y_axis(gripper):
    """两指沿 ±Y 平动，实测标定应得到干净的 Y 轴单位向量。"""
    d = np.asarray(gripper.open_dir, dtype=float)
    assert d.shape == (3,)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert np.isclose(abs(d[1]), 1.0, atol=1e-6), f"预期沿 Y 张开，实得 {d}"


def test_link_parts_are_palm_then_fingers(gripper):
    """次序必须是掌部在前、手指在后：G 指标靠它豁免指垫贴合面。"""
    gripper.grip_at(np.zeros(3), np.eye(3), 0.05)
    parts = gripper.link_parts()
    assert len(parts) == 3
    assert [p.part_id for p in parts] == ["palm", "finger_0", "finger_1"]
    for p in parts:
        assert len(p.geometry.vertices) > 0 and len(p.geometry.faces) > 0


def test_fingers_separate_by_the_commanded_width(gripper):
    """开口指令变化时，两指沿张开方向的间距应随之变化。"""
    def gap(width):
        gripper.grip_at(np.zeros(3), np.eye(3), width)
        a, b = gripper.link_parts()[1:]
        delta = a.assembled_tf[:3, 3] - b.assembled_tf[:3, 3]
        return float(abs(delta @ np.asarray(gripper.open_dir, dtype=float)))

    assert gap(0.070) > gap(0.030) + 0.030


def test_clone_is_independent(gripper):
    other = gripper.clone()
    gripper.grip_at(np.zeros(3), np.eye(3), 0.080)
    other.grip_at(np.array([1.0, 0.0, 0.0]), np.eye(3), 0.020)
    assert not np.allclose(
        gripper.link_parts()[0].assembled_tf, other.link_parts()[0].assembled_tf
    )


def test_real_mesh_tolerances_are_separate_from_numerical_roundoff(gripper):
    """真实 STL 就位容差不能再冒充微米级浮点舍入误差。"""
    assert gripper.pad_seating_tolerance_m == pytest.approx(0.5e-3)
    assert gripper.pad_patch_tolerance_m == pytest.approx(1.0e-3)
    assert gripper.pad_seating_tolerance_m > GraspabilityConfig().contact_roundoff_m


def test_bridge_posts_have_qualified_grasps(gripper):
    """真夹爪在 Bridge 支柱上应当能给出力闭合的侧夹。

    支柱截面 50 mm，落在 Panthera 的 0–85 mm 量程内。
    """
    assembly = build_bridge()
    parts = {p.part_id: p for p in assembly.parts}
    assert box_grasps(parts["post_l"], gripper), "50 mm 见方的支柱应当有可行侧夹"

    analyzer = GraspabilityAnalyzer(assembly, gripper, config=GraspabilityConfig(max_grasps=8))
    fixed = {p.part_id for p in assembly.parts if p.fixed}
    full = assembly.initial_state()
    from asmcore.model import AssemblyState

    alone = AssemblyState(
        {k: v for k, v in full.poses.items() if k in {"post_l"} | fixed}
    )
    result = analyzer.analyze(alone, "post_l")
    assert result.catalogue_size > 0
    assert result.status in ("complete", "partial")
    if result.count is not None:
        assert result.count > 0, "单独装入时支柱应当至少有一个合格抓取"
