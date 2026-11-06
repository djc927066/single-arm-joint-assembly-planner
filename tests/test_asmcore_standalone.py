"""锁死 asmcore 的三条不变量, 以及单臂 S=0 硬剪枝闸门的行为.

不变量
  1. asmcore 不导入任何 wrs.*        —— 它必须能活在 WRS1 环境里
  2. asmcore 只需要 numpy + scipy     —— clarabel/open3d/mujoco 都不是必需
  3. 显式给定 mating axis 时不触发 SOCP —— 联合搜索的方向来自 asmdef, 不需要解方向锥

场景: table(固定) + post_a + post_b + lintel 横梁跨在两根柱子上。
{table, post_a, lintel} 是典型的不稳定中间状态: 横梁悬臂在单根柱子上,
质心落在支撑多边形之外 -> 静力平衡无解 -> S=0 -> 按既定方案硬剪枝。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from asmcore.contact.analysis import analyze_contacts
from asmcore.contact.graph import build_contact_graph
from asmcore.geometry.proximity import MeshProximity
from asmcore.mechanics.equilibrium import StabilityConfig, check_equilibrium
from asmcore.model import Assembly, AssemblyState, MeshData, Part

BACKEND = MeshProximity(numerical_tol_m=1e-10)
UP = np.array([[0.0, 0.0, 1.0]])


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


def state_of(assembly, ids):
    full = assembly.initial_state()
    return AssemblyState({k: v for k, v in full.poses.items() if k in ids})


def is_stable(assembly, ids) -> bool:
    """S > 0 判据: 该中间状态存在静力平衡解 (论文二 Eq.5/6)."""
    st = state_of(assembly, ids)
    graph = build_contact_graph(assembly, st, analyze_contacts(assembly, st, backend=BACKEND))
    return check_equilibrium(assembly, st, graph, config=StabilityConfig()).status == "feasible"


# ---- 不变量 -----------------------------------------------------------------

def test_no_wrs_import_anywhere_in_asmcore():
    """整包 import 一遍, 确认没有任何 wrs.* 被拉进来."""
    code = (
        "import importlib, pkgutil, sys, asmcore\n"
        "for m in pkgutil.walk_packages(asmcore.__path__, 'asmcore.'):\n"
        "    importlib.import_module(m.name)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'wrs'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"asmcore 拉入了 wrs 模块: {out.stdout}"


def test_only_numpy_scipy_required():
    code = (
        "import importlib, pkgutil, sys, asmcore\n"
        "for m in pkgutil.walk_packages(asmcore.__path__, 'asmcore.'):\n"
        "    importlib.import_module(m.name)\n"
        "heavy = [x for x in ('clarabel','open3d','mujoco','panda3d','trimesh')\n"
        "         if x in sys.modules]\n"
        "print(heavy)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"导入期拉入重依赖: {out.stdout}"


# ---- 稳定性判据 --------------------------------------------------------------

def test_full_assembly_is_stable(assembly):
    assert is_stable(assembly, {"table", "post_a", "post_b", "lintel"})


def test_both_posts_alone_are_stable(assembly):
    assert is_stable(assembly, {"table", "post_a", "post_b"})


def test_cantilevered_lintel_is_unstable(assembly):
    """论文二的 S=0: 横梁只搭在一根柱子上, 质心悬空."""
    assert not is_stable(assembly, {"table", "post_a", "lintel"})


# ---- 单臂硬剪枝闸门 ----------------------------------------------------------

@pytest.mark.parametrize("order,feasible", [
    (("post_a", "post_b", "lintel"), True),
    (("post_b", "post_a", "lintel"), True),
    (("post_a", "lintel", "post_b"), False),
    (("post_b", "lintel", "post_a"), False),
    (("lintel", "post_a", "post_b"), False),
])
def test_single_arm_order_gate(assembly, order, feasible):
    """单臂方案: 任一中间状态 S=0 即判该顺序不可行 (无第二只手可托)."""
    ids = {"table"}
    ok = True
    for pid in order:
        ids = ids | {pid}
        if not is_stable(assembly, ids):
            ok = False
            break
    assert ok is feasible


# ---- 显式方向绕开 SOCP -------------------------------------------------------

def test_given_mating_axis_needs_no_socp(assembly):
    """联合搜索里方向由 asmdef 的 mating axis 给定 -> 不必解方向锥, 不需要 clarabel."""
    from asmcore.motion.part_motion import MotionConfig, plan_removal
    st = assembly.initial_state()
    graph = build_contact_graph(assembly, st, analyze_contacts(assembly, st, backend=BACKEND))
    result = plan_removal(assembly, st, "lintel", graph=graph, backend=BACKEND,
                          config=MotionConfig(), directions=UP)
    assert result.status == "success"
    assert "clarabel" not in sys.modules


def test_sequence_search_single_arm_no_auxiliary(assembly):
    """0 辅助资源 (单臂) 下应能搜到一条全程稳定的装配顺序, 且横梁最后装."""
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator, plan_sequence

    class AxisEvaluator(SequenceEvaluator):
        def evaluate(self, state, active, part_id, *,
                     remainder_support_ids=None, directions=None):
            return super().evaluate(state, active, part_id,
                                    remainder_support_ids=remainder_support_ids,
                                    directions=UP)

    cfg = SequenceConfig(method="dfs", max_auxiliary_resources=0,
                         max_expansions=256, time_limit_s=180.0)
    result = plan_sequence(assembly, config=cfg, evaluator=AxisEvaluator(assembly, cfg))
    assert result.status == "success", result.diagnostics
    order = [s.part_id for s in result.assembly_steps]
    assert order[-1] == "lintel", f"横梁必须最后装, 实得 {order}"
