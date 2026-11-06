"""
Fast Dual-Arm Layout Search (CEM + 三层可执行性评分)
=====================================================

目的
----
在 *不修改* ``sealp/examples/motion/dual_sequence_execution.py`` 的前
提下，用 CEM (Cross-Entropy Method) 而不是 DFS 快速找出一组 staging
位置，使得每件零件的 ``staging -> goal`` 整条路径上 **每一帧 IK 可解
+ 实时无碰撞**。

为什么不再用 DFS
~~~~~~~~~~~~~~~~~
原 DFS 沿装配步逐件枚举 + 回溯，几何约束稠密时分支爆炸；CEM 把决策
变成一次 10 维 (5 件 x (x, y)) 连续优化，每代并行评估 N 个候选，elite
更新高斯均值 + sigma 收缩，几代就能收敛。

三层评分（与 dual_sequence_execution 几何流对齐）
-------------------------------------------------
**L1 (毫秒)** 纯几何：
    * 件间不重叠 (``cmodel.is_mcdwith``)
    * staging 不侵入 ``parent==fixture`` 件的 goal 体积
    * 双臂 home 姿态下 staging 与任一只臂不穿模

**L2 (~1s/件)** ``reason_common_gids`` + 中间段 IK：
    * 按真实装配顺序模拟每一步 obstacle list（已 placed 件用 goal cm，
      未 placed 件用 staging cm）。
    * 对每件双臂之一调 ``reason_common_gids`` 拿 staging+goal 双端
      IK + 共同抓取 + 持物碰撞过滤；通过的 grasp 再过 ``pick_depart``
      (+Z 抬升 0.05m) 与 ``place_approach`` (-Z 下放 0.05m) 中间段每
      个采样点的 IK 检查。
    * 等价于"两端 + 直线段每帧 IK 可解 + 抓取无碰撞"，足以判断 demo
      里 5/5 能跑通的布局，且比完整 RRT 评分便宜 ~10x。

**L3 (~30s/layout, 可选)** 完整 RRT motion 校验：
    * 用 ``--with-l3`` 启用。逐件按装配顺序跑 ``TransportPrimitive.plan``
      (内部走 ``PickPlacePlanner.gen_pick_and_place``)，等价对**整条
      RRT + IK 轨迹每一帧**做无碰撞 + IK 校验。L3 通过 ⇒ 原话意义上
      的"完美摆放"。
    * 该路径需要 panda3d 全局 ``base``，启用时脚本会创建一个图标化的
      ``wd.World`` 让 builtins.base 生效。L1+L2 不需要 base。

可视化（自动）
---------------
``_output/fast_diagnostics/<task>/`` 下输出：

* ``fig_workspace_topdown.png`` —— 桌面 top-down，最终 staging 实心
  彩点、goal 虚线圈、机械臂 base + 双臂 home 位置、staging->goal 直
  线、每件采样盒淡色填充。
* ``fig_grasp_field.png``       —— 五件 grasp 数热力图小图组：每张
  在该件采样盒内 5x5 扫描，其它件用最终选定位置；红 X 是 CEM 选定。
* ``fig_convergence.png``       —— CEM 每代 best / elite mean L2
  分；L3 试过的代次单独标注。
* ``fig_arm_assignment.png``    —— 哪只臂搬哪件（饼图）。
* ``fig_pruning_funnel.png``    —— 累计 L1 / L2 / L3 通过-淘汰条形图。

Usage::

    python -m sealp.examples.layout.fast_layout_search                   # L1+L2, 通常 < 30s
    python -m sealp.examples.layout.fast_layout_search --pop 32 --gens 8 # 加大搜索预算
    python -m sealp.examples.layout.fast_layout_search --with-l3         # 含完整 RRT 校验
    python -m sealp.examples.layout.fast_layout_search --no-vis          # 关闭画图
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import wrs.modeling.collision_model as mcm
from wrs.manipulation.pick_place import PickPlacePlanner

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda

# Panthera-HT 双臂右臂相对左臂 base 的 y 偏移（与
# ``DualPantheraHTNoBody.__init__`` 中 ``arm_y_offset`` 默认值对齐）。
# Piper 时代这里隐含为 0.597；切换到 Panthera-HT 后必须同步采样窗口。
_DUAL_ARM_Y_OFFSET = 0.62

from sealp.assembly_sequence import AssemblyDef
from sealp.colliders import StaticEnvironment
from sealp.config import load_config
from sealp.layout import WorkspaceLayout
from sealp.layout.dual_staging_search import find_obstacle_def
from sealp.examples.layout.eval_dual_layout import (
    STAGING_SEEDS,
    FIXTURE_POS, FIXTURE_ROTMAT,
    ROBOT_BASE_POS, ROBOT_BASE_ROTMAT,
    load_grasp_cache, model_alias_for_part,
)


# ══════════════════════════════════════════════════════════════
#  几何流常量（与 dual_sequence_execution.RELAXED_PLANNING 对齐）
# ══════════════════════════════════════════════════════════════
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0])      # +Z 抬升
PICK_DEPART_DIST = 0.05
PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0])  # -Z 下放
PLACE_APPROACH_DIST = 0.05
LEG_PLACE_DEPART_DIR = np.array([-1.0, 0.0, 0.0])
SEAT_PLACE_DEPART_DIR = np.array([0.0, 0.0, 1.0])
LEG_PLACE_DEPART_DIST = 0.05
SEAT_PLACE_DEPART_DIST = 0.05
APPROACH_DIST = 0.0       # pick_approach 距离=0 -> 不做 approach
LINEAR_GRANULARITY = 0.04
HOME_JV = np.zeros(6)


def _here(*parts) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))


@dataclass
class FastSearchTask:
    name: str
    asmdef_path: str
    grasp_pickles: Dict[str, str]
    part_ids: Tuple[str, ...]
    staging_seeds: Dict[str, np.ndarray]
    fixture_pos: np.ndarray
    fixture_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    robot_base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    robot_base_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    config_yaml_path: str = ""
    table_obstacle_name: str = "work_table"
    table_margin: float = 0.06
    # per-part 采样盒：{pid: ((xlo, xhi), (ylo, yhi))}；None ⇒ 自动
    part_xy_bounds: Optional[Dict[str, Tuple[Tuple[float, float],
                                              Tuple[float, float]]]] = None
    model_alias_fn: Optional[Callable[[str], str]] = None
    output_layout_name: str = "dual_fastsearched"


YUANCHAIR_FAST_TASK = FastSearchTask(
    name="yuanchair",
    asmdef_path=_here("..", "..", "assembly_sequence",
                      "_demo_output", "yuanchair.asmdef"),
    grasp_pickles={
        "leg_model":  _here("..", "grasp", "_output",
                            "demo_yuanchair-part2_grasps.pickle"),
        "seat_model": _here("..", "grasp", "_output",
                            "demo_yuanchair-part1_grasps.pickle"),
    },
    part_ids=("seat", "leg_bl", "leg_br", "leg_fl", "leg_fr"),
    staging_seeds=STAGING_SEEDS,
    fixture_pos=FIXTURE_POS,
    fixture_rotmat=FIXTURE_ROTMAT,
    robot_base_pos=ROBOT_BASE_POS,
    robot_base_rotmat=ROBOT_BASE_ROTMAT,
    config_yaml_path=_here("..", "..", "config", "sample_config.yaml"),
    model_alias_fn=model_alias_for_part,
    output_layout_name="dual_yuanchair_fastsearched",
)


# ══════════════════════════════════════════════════════════════
#  几何 helper
# ══════════════════════════════════════════════════════════════
def _step_parent_id(asm, part_id):
    for s in asm.steps:
        if s.part_id == part_id:
            return s.parent_id
    return None


def _arms_collide_at_home(robot, obstacle_list) -> bool:
    """双臂在 home 姿态下是否与任一障碍穿模。"""
    if not obstacle_list:
        return False
    for arm in (robot.lft_arm, robot.rgt_arm):
        hit = arm.is_collided(obstacle_list=list(obstacle_list))
        collided = hit[0] if isinstance(hit, tuple) else hit
        if collided:
            return True
    return False


def _segment_reachable(arm, grasp, base_pos, base_rot, direction,
                       distance, n_samples=4) -> bool:
    """staging / goal 抓取沿 direction 直线段中间各采样点 IK 是否一直可解。"""
    if distance is None or distance <= 1e-6 or n_samples < 1:
        return True
    direction = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        return True
    dir_unit = direction / n
    tcp_pos = base_rot.dot(grasp.ac_pos) + base_pos
    tcp_rot = base_rot.dot(grasp.ac_rotmat)
    seed = arm.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
    if seed is None:
        return False
    for i in range(1, n_samples + 1):
        d = distance * i / n_samples
        end_pos = tcp_pos + dir_unit * d
        jv = arm.ik(tgt_pos=end_pos, tgt_rotmat=tcp_rot, seed_jnt_values=seed)
        if jv is None:
            return False
        seed = jv
    return True


def _reason_common_ok(arm, gc, sp, sr, gp, gr, obstacle_list) -> Tuple[bool, int]:
    """staging->goal 双端 IK + 共同抓取 + pick_depart / place_approach 中间段。

    Returns (ok, n_grasps)。
    """
    if gc is None or len(gc) == 0:
        return False, 0
    planner = PickPlacePlanner(robot=arm)
    gids = planner.reason_common_gids(
        grasp_collection=gc,
        goal_pose_list=[(sp, sr), (gp, gr)],
        obstacle_list=obstacle_list,
    )
    if not gids:
        return False, 0
    gids = [g for g in gids
            if _segment_reachable(arm, gc[g], sp, sr,
                                  PICK_DEPART_DIR, PICK_DEPART_DIST)]
    if not gids:
        return False, 0
    rev = -np.asarray(PLACE_APPROACH_DIR, dtype=float)
    gids = [g for g in gids
            if _segment_reachable(arm, gc[g], gp, gr,
                                  rev, PLACE_APPROACH_DIST)]
    if not gids:
        return False, 0
    return True, len(gids)


def _arm_priority_for_part(part_id: str) -> Tuple[str, str]:
    """命名约定：尾字符 'r' -> 右臂优先；其余 -> 左臂优先。"""
    pid = str(part_id)
    return ("rgt", "lft") if pid and pid[-1] == 'r' else ("lft", "rgt")


# ══════════════════════════════════════════════════════════════
#  layout + 评分容器
# ══════════════════════════════════════════════════════════════
@dataclass
class _ScoredLayout:
    xy: Dict[str, np.ndarray]
    layout_score: float = -np.inf
    grasp_counts: Dict[str, int] = field(default_factory=dict)
    arm_choice: Dict[str, str] = field(default_factory=dict)
    l1_pass: bool = False
    l2_pass: bool = False
    l3_pass: bool = False
    fail_reason: str = ""


# ══════════════════════════════════════════════════════════════
#  核心搜索器
# ══════════════════════════════════════════════════════════════
class FastLayoutSearcher:
    """三层评分（L1/L2 默认；L3 可选）+ 统一状态。"""

    def __init__(self, task: FastSearchTask, *, enable_l3: bool = False,
                 ik_retry_n: int = 0):
        self.task = task
        self.enable_l3 = enable_l3
        self.model_alias_fn = (task.model_alias_fn
                               or (lambda pid: f"{pid}_model"))

        # ── 仅 L3 启用时才创建 ShowBase（让 builtins.base 生效）──
        # PickPlacePlanner / approach_depart_planner 内部某些 toggle_dbg
        # 守卫的代码路径会引用全局 ``base``，部分 mc 操作也依赖。L1+L2
        # 不调到这些路径，无需 ShowBase。
        self._world = None
        if enable_l3:
            try:
                _ = base                        # 已存在直接复用
            except NameError:
                try:
                    import wrs.visualization.panda.world as wd
                    from panda3d.core import WindowProperties
                    self._world = wd.World(cam_pos=[1.5, -0.3, 1.2],
                                            lookat_pos=[0.3, -0.3, 0.1])
                    wp = WindowProperties()
                    wp.setIconified(True)
                    base.win.requestProperties(wp)
                    print("[fast_layout] 已创建 ShowBase（图标化），L3 校验可用。")
                except Exception as _err:
                    print(f"[fast_layout][WARN] L3 启用失败 (ShowBase): "
                          f"{_err!r}；继续按 L1+L2 运行。")
                    self.enable_l3 = False

        # ── asm + world poses ────────────────────────────
        self.asm = AssemblyDef.load(task.asmdef_path)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=task.fixture_pos,
            fixture_rotmat=task.fixture_rotmat)

        # ── 环境障碍 + 桌面定义 ──────────────────────────
        self.env_obs: List = []
        self.table_def: Optional[dict] = None
        if task.config_yaml_path and os.path.isfile(task.config_yaml_path):
            cfg = load_config(task.config_yaml_path)
            env = StaticEnvironment(
                obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
            self.env_obs = list(env.obstacle_list)
            self.table_def = find_obstacle_def(
                cfg.obstacle_defs, task.table_obstacle_name)

        # ── 双臂 + 抓取库 ────────────────────────────────
        self.robot = pda.DualPantheraHTNoBody(
            pos=task.robot_base_pos,
            rotmat=task.robot_base_rotmat,
            arm_y_offset=_DUAL_ARM_Y_OFFSET,
            enable_cc=True)
        self.robot.lft_arm.goto_given_conf(HOME_JV)
        self.robot.rgt_arm.goto_given_conf(HOME_JV)
        # —— 让 manipulator 在 trac_ik 抽风时按 N 个随机 seed 重试 ——
        # 每次失败的 ik 多跑 N 次 (~N*2ms)，但能挽救边缘 grasp，
        # 让 reason_common_gids 的 0 通过率明显下降。
        if ik_retry_n > 0:
            for arm_robot in (self.robot.lft_arm.manipulator,
                               self.robot.rgt_arm.manipulator):
                if hasattr(arm_robot, "_ik_retry_n"):
                    arm_robot._ik_retry_n = ik_retry_n
            print(f"  [TracIK] 启用多 seed 重试: "
                  f"_ik_retry_n={ik_retry_n} （仅在主调用 None 时触发）")
        self.grasp_cache = load_grasp_cache(task.grasp_pickles)

        # ── staging 探针 + goal cm ───────────────────────
        self.search_part_ids = [p for p in task.part_ids
                                if p in task.staging_seeds]
        self.staging_obs: Dict[str, "mcm.CollisionModel"] = {}
        for pid in self.search_part_ids:
            mp = self.asm.model_path(pid)
            if not os.path.isfile(mp):
                continue
            cm = mcm.CollisionModel(initor=mp)
            cm.pos = np.zeros(3)
            cm.rotmat = np.eye(3)
            cm._sealp_part_id = pid
            cm._sealp_role = "fast_staging"
            self.staging_obs[pid] = cm
        self.goal_obs: Dict[str, "mcm.CollisionModel"] = {}
        for pid, (gp, gr) in self.world_poses.items():
            if pid not in self.asm.part_ids:
                continue
            mp = self.asm.model_path(pid)
            if not os.path.isfile(mp):
                continue
            cm = mcm.CollisionModel(initor=mp)
            cm.pos = gp
            cm.rotmat = gr
            cm._sealp_part_id = pid
            cm._sealp_role = "fast_goal"
            self.goal_obs[pid] = cm

        # ── per-part 采样盒 ──────────────────────────────
        self.bounds = self._resolve_xy_bounds()
        print("\n[xy 采样盒]")
        for pid in self.search_part_ids:
            (xlo, xhi), (ylo, yhi) = self.bounds[pid]
            print(f"  {pid:<8s}  x in [{xlo:+.3f}, {xhi:+.3f}]  "
                  f"y in [{ylo:+.3f}, {yhi:+.3f}]   "
                  f"seed={np.round(task.staging_seeds[pid], 3).tolist()}")

        # 统计 / 收敛轨迹
        self.funnel = {"sampled": 0, "l1_pass": 0,
                       "l2_pass": 0, "l3_attempts": 0, "l3_pass": 0}
        self.history: List[Tuple[int, float, float]] = []
        # 收集本次跑过 L3 的代次（用于在收敛图上标记）
        self.l3_marked_gens: List[Tuple[int, bool]] = []

    # ── per-part 采样盒：基于桌面 + 双臂 y 区 ──────────
    def _resolve_xy_bounds(
        self) -> Dict[str, Tuple[Tuple[float, float],
                                  Tuple[float, float]]]:
        if self.task.part_xy_bounds is not None:
            return dict(self.task.part_xy_bounds)
        if self.table_def is not None and self.table_def.get("type") == "box":
            tp = self.table_def["pos"]
            te = self.table_def["extent"]
            tx_lo = float(tp[0]) - float(te[0]) / 2 + self.task.table_margin
            tx_hi = float(tp[0]) + float(te[0]) / 2 - self.task.table_margin
            ty_lo = float(tp[1]) - float(te[1]) / 2 + self.task.table_margin
            ty_hi = float(tp[1]) + float(te[1]) / 2 - self.task.table_margin
        else:
            tx_lo, tx_hi, ty_lo, ty_hi = 0.0, 0.7, -1.0, 0.4
        # 双臂 y 中心（与 ``DualPantheraHTNoBody`` 的 arm_y_offset 一致）。
        # Panthera-HT 桌腿 cdprim 较粗，右半区下扩到 rgt_y-0.42 才能容下
        # 实测可行的 leg_fr/leg_br y≈-0.85（见 eval_dual_layout.STAGING_SEEDS）。
        lft_y, rgt_y = 0.0, -_DUAL_ARM_Y_OFFSET
        out: Dict[str, Tuple[Tuple[float, float],
                              Tuple[float, float]]] = {}
        for pid in self.search_part_ids:
            x_lo = max(tx_lo, 0.18)
            x_hi = min(tx_hi, 0.55)
            if pid == "seat":
                # seat 比腿大 + 离 fixture 较近 → 偏前 + 偏左
                out[pid] = ((max(x_lo, 0.22), min(x_hi, 0.40)),
                            (-0.22, -0.02))
            elif pid.endswith("r"):
                # 右半区（命名尾字 'r'）→ 围绕 rgt_y
                out[pid] = ((x_lo, x_hi),
                            (max(ty_lo, rgt_y - 0.42),
                             min(ty_hi, rgt_y + 0.20)))
            else:
                # 左半区 → 围绕 lft_y
                out[pid] = ((x_lo, x_hi),
                            (max(ty_lo, lft_y + 0.06),
                             min(ty_hi, lft_y + 0.32)))
        return out

    # ── 同步所有 staging probe 到一个 layout ─────────────
    def _apply_xy(self, xy: Dict[str, np.ndarray]):
        for pid in self.search_part_ids:
            cm = self.staging_obs.get(pid)
            if cm is None:
                continue
            p = xy.get(pid)
            if p is None:
                p = self.task.staging_seeds[pid]
            cm.pos = np.array([float(p[0]), float(p[1]), 0.0])
            cm.rotmat = np.eye(3)

    # ── L1：纯几何 ──────────────────────────────────────
    def l1(self, sl: _ScoredLayout) -> bool:
        self._apply_xy(sl.xy)
        ids = list(self.staging_obs.keys())
        for i in range(len(ids)):
            cm_i = self.staging_obs[ids[i]]
            for j in range(i + 1, len(ids)):
                cm_j = self.staging_obs[ids[j]]
                if cm_i.is_mcdwith(cm_j):
                    sl.fail_reason = f"L1: {ids[i]} vs {ids[j]} overlap"
                    return False
        for pid in ids:
            cm = self.staging_obs[pid]
            for gid, gcm in self.goal_obs.items():
                if gid == pid:
                    continue
                if _step_parent_id(self.asm, gid) != "fixture":
                    continue
                if cm.is_mcdwith(gcm):
                    sl.fail_reason = f"L1: {pid} 侵入 fixture goal {gid}"
                    return False
        if _arms_collide_at_home(
                self.robot, [self.staging_obs[p] for p in ids]):
            sl.fail_reason = "L1: home 姿态下 staging 与机械臂穿模"
            return False
        sl.l1_pass = True
        return True

    # ── 步骤感知的 obstacle list（与 demo 的 _global_verify 一致）──
    def _step_aware_obs(self, current_pid: str, placed: set) -> List:
        obs = list(self.env_obs)
        for other in placed:
            if other in self.goal_obs:
                obs.append(self.goal_obs[other])
        for other in self.search_part_ids:
            if other == current_pid or other in placed:
                continue
            if other in self.staging_obs:
                obs.append(self.staging_obs[other])
        return obs

    # ── L2：reason_common_gids + 中间段 IK，按装配步序 ─
    def l2(self, sl: _ScoredLayout, *, retry: int = 0) -> bool:
        """L2 评估。retry > 0 时对每一步双臂均失败时重 reason ``retry``
        次，专治 trac_ik 偶尔抽风。每次重试代价 = 一次完整 reason
        (~1-3s/件)，所以 retry=1~2 已足够。
        """
        self._apply_xy(sl.xy)
        placed: set = set()
        counts: Dict[str, int] = {}
        arms: Dict[str, str] = {}
        for s in self.asm.steps:
            pid = s.part_id
            if pid not in self.search_part_ids or pid not in self.world_poses:
                continue
            sp = self.staging_obs[pid].pos.copy()
            sr = self.staging_obs[pid].rotmat.copy()
            gp, gr = self.world_poses[pid]
            gc = self.grasp_cache.get(self.model_alias_fn(pid))
            if gc is None:
                sl.fail_reason = f"L2: 抓取库缺 {pid}"
                sl.grasp_counts = counts
                return False
            obs = self._step_aware_obs(pid, placed)
            pref = _arm_priority_for_part(pid)
            arm0 = (self.robot.lft_arm if pref[0] == "lft"
                    else self.robot.rgt_arm)
            arm1 = (self.robot.lft_arm if pref[1] == "lft"
                    else self.robot.rgt_arm)
            tag, n = None, 0
            for attempt in range(retry + 1):
                ok0, n0 = _reason_common_ok(arm0, gc, sp, sr, gp, gr, obs)
                if ok0:
                    tag, n = pref[0], n0
                    break
                ok1, n1 = _reason_common_ok(arm1, gc, sp, sr, gp, gr, obs)
                if ok1:
                    tag, n = pref[1], n1
                    break
            if tag is None:
                sl.fail_reason = (f"L2: step={s.step_id} {pid} "
                                   f"双臂均无可行 reason "
                                   f"(placed={list(placed)}, "
                                   f"tried {retry + 1} attempts)")
                sl.grasp_counts = counts
                return False
            counts[pid] = n
            arms[pid] = tag
            placed.add(pid)
        sl.grasp_counts = counts
        sl.arm_choice = arms
        if counts:
            vs = list(counts.values())
            sl.layout_score = float(min(vs) + 0.1 * (sum(vs) / len(vs)))
        sl.l2_pass = True
        return True

    # ── L3：完整 motion 校验（可选）────────────────────
    def l3(self, sl: _ScoredLayout) -> bool:
        """对每件按装配顺序跑 PickPlacePlanner.gen_pick_and_place。

        等价对整条 RRT + IK 轨迹每一帧做无碰撞 + IK 校验。任一件失败
        ⇒ False。
        """
        if not self.enable_l3:
            return False
        from sealp.primitives.transport import TransportPrimitive
        self._apply_xy(sl.xy)
        lft_t = TransportPrimitive(self.robot.lft_arm)
        rgt_t = TransportPrimitive(self.robot.rgt_arm)
        placed: set = set()
        for s in self.asm.steps:
            pid = s.part_id
            if pid not in self.search_part_ids or pid not in self.world_poses:
                continue
            tag = sl.arm_choice.get(pid, "lft")
            transport = lft_t if tag == "lft" else rgt_t
            sp = self.staging_obs[pid].pos.copy()
            sr = self.staging_obs[pid].rotmat.copy()
            gp, gr = self.world_poses[pid]
            mp = self.asm.model_path(pid)
            obj_cm = mcm.CollisionModel(initor=mp)
            obj_cm.pos = sp
            obj_cm.rotmat = sr
            gc = self.grasp_cache.get(self.model_alias_fn(pid))
            obs = self._step_aware_obs(pid, placed)
            place_dep_dir = (LEG_PLACE_DEPART_DIR if pid.startswith("leg_")
                             else SEAT_PLACE_DEPART_DIR)
            place_dep_dist = (LEG_PLACE_DEPART_DIST if pid.startswith("leg_")
                              else SEAT_PLACE_DEPART_DIST)
            try:
                res = transport.plan(
                    obj_cmodel=obj_cm,
                    grasp_collection=gc,
                    goal_pose_list=[(gp, gr)],
                    obstacle_list=obs,
                    approach_distance=APPROACH_DIST,
                    depart_distance=PICK_DEPART_DIST,
                    pick_depart_direction=PICK_DEPART_DIR,
                    pick_depart_distance=PICK_DEPART_DIST,
                    place_approach_direction_list=[PLACE_APPROACH_DIR],
                    place_approach_distance_list=[PLACE_APPROACH_DIST],
                    place_depart_direction_list=[place_dep_dir],
                    place_depart_distance_list=[place_dep_dist],
                    linear_granularity=LINEAR_GRANULARITY,
                )
            except Exception as e:
                sl.fail_reason = (f"L3: step {s.step_id} {pid} {tag} "
                                   f"抛错 {type(e).__name__}: {e!r}")
                return False
            if not res.success:
                sl.fail_reason = (f"L3: step {s.step_id} {pid} {tag} 失败: "
                                   f"{res.error_msg or 'no plan'}")
                return False
            placed.add(pid)
        sl.l3_pass = True
        return True


# ══════════════════════════════════════════════════════════════
#  CEM 优化器
# ══════════════════════════════════════════════════════════════
def _clip_xy(p, bounds):
    (xlo, xhi), (ylo, yhi) = bounds
    return np.array([float(np.clip(p[0], xlo, xhi)),
                     float(np.clip(p[1], ylo, yhi))])


def _seed_with_noise(seeds, pids, bounds, rng, scale):
    return {pid: _clip_xy(np.array([seeds[pid][0], seeds[pid][1]])
                          + rng.normal(0, scale, 2), bounds[pid])
            for pid in pids}


def cem_search(searcher: FastLayoutSearcher,
               *,
               pop_size: int = 24,
               elite_n: int = 6,
               generations: int = 8,
               sigma_init: float = 0.08,
               sigma_decay: float = 0.7,
               sigma_min: float = 0.015,
               try_l3_top_k: int = 0,
               rng_seed: int = 0,
               l2_retry: int = 0,
               verbose: bool = True,
               ) -> Optional[_ScoredLayout]:
    """CEM 主循环。

    第 0 代：种子 + 种子+小扰动 + 全盒高斯采样混合
    第 i 代：以上代精英每件 (x, y) 经验均值/方差为高斯，采 pop_size
    每代评分 L1->L2，取 top-K 精英；try_l3_top_k>0 时对当前 best 前
    K 个跑 L3，任一通过 ⇒ 立即返回。

    Returns L3 通过 layout（如果 enable_l3=False，永远不会 L3 通过；
    则返回 L2 最佳 layout）。
    """
    rng = np.random.default_rng(rng_seed)
    pids = searcher.search_part_ids
    bounds = searcher.bounds
    seeds = searcher.task.staging_seeds

    mu = {pid: np.array([float(seeds[pid][0]),
                         float(seeds[pid][1])]) for pid in pids}
    sigma = {pid: np.array([sigma_init, sigma_init]) for pid in pids}

    # 第 0 代：种子 + 小扬动 + 中等扬动 + 全盒高斯，三档分散覆盖。
    # anchor 提到 1/2：小规模 (pop=12) 时 6 个候选都贴近种子，
    # trac_ik 单次抽风也很难全军覆没。
    candidates: List[_ScoredLayout] = []
    candidates.append(_ScoredLayout(
        xy={pid: np.array([float(seeds[pid][0]),
                            float(seeds[pid][1])]) for pid in pids}))
    n_anchor = max(2, pop_size // 2)
    n_small_noise = (n_anchor - 1) // 2
    for _ in range(n_small_noise):
        candidates.append(_ScoredLayout(
            xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init / 2)))
    for _ in range((n_anchor - 1) - n_small_noise):
        candidates.append(_ScoredLayout(
            xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init)))
    while len(candidates) < pop_size:
        candidates.append(_ScoredLayout(
            xy={pid: _clip_xy(rng.normal(mu[pid], sigma_init * 1.4),
                              bounds[pid]) for pid in pids}))

    best_overall: Optional[_ScoredLayout] = None

    for gen in range(generations):
        scored: List[_ScoredLayout] = []
        for sl in candidates:
            searcher.funnel["sampled"] += 1
            if not searcher.l1(sl):
                continue
            searcher.funnel["l1_pass"] += 1
            if searcher.l2(sl, retry=l2_retry):
                searcher.funnel["l2_pass"] += 1
                scored.append(sl)
        scored.sort(key=lambda s: s.layout_score, reverse=True)

        if scored:
            best = scored[0]
            elite = scored[:elite_n]
            elite_mean = float(np.mean([e.layout_score for e in elite]))
            if (best_overall is None
                    or best.layout_score > best_overall.layout_score):
                best_overall = best
        else:
            best, elite, elite_mean = None, [], -np.inf
        searcher.history.append((gen,
                                  best.layout_score if best else -np.inf,
                                  elite_mean))

        if verbose:
            n_l1 = sum(1 for c in candidates if c.l1_pass)
            best_msg = (f"best={best.layout_score:.2f}  "
                        f"counts={best.grasp_counts}"
                        if best else "-")
            print(f"\n[CEM gen {gen}/{generations - 1}]  "
                  f"sampled={pop_size}  L1ok={n_l1}  L2ok={len(scored)}  "
                  f"elite_mean={elite_mean:.2f}  {best_msg}")

        # 没有任何 layout 通过 L1+L2 ⇒ 兜底重启
        # 不再围绕种子打转：sigma×2.0、保留 1 个种子锚点、剩余样本
        # 一半在种子大噪声、另一半全盒均匀重采 (mu 退回种子)，
        # 跳出 trac_ik "本轮全军覆没" 的运气坑。
        if not scored:
            if verbose:
                print("  -> 本代无可行解，sigma 放大 2.0x，"
                      "半数样本全盒重采。")
            for pid in pids:
                sigma[pid] = np.minimum(sigma[pid] * 2.0,
                                         np.array([0.25, 0.25]))
                mu[pid] = np.array([float(seeds[pid][0]),
                                     float(seeds[pid][1])])
            candidates = [_ScoredLayout(
                xy={pid: np.array([float(seeds[pid][0]),
                                    float(seeds[pid][1])]) for pid in pids})]
            n_noise = (pop_size - 1) // 2
            for _ in range(n_noise):
                candidates.append(_ScoredLayout(
                    xy=_seed_with_noise(seeds, pids, bounds, rng,
                                         sigma_init * 1.5)))
            for _ in range((pop_size - 1) - n_noise):
                xy = {pid: np.array([
                    float(rng.uniform(bounds[pid][0][0], bounds[pid][0][1])),
                    float(rng.uniform(bounds[pid][1][0], bounds[pid][1][1])),
                ]) for pid in pids}
                candidates.append(_ScoredLayout(xy=xy))
            continue

        # L3：对前 K 个 best 做完整 motion 校验
        if try_l3_top_k > 0 and searcher.enable_l3:
            for sl in scored[:try_l3_top_k]:
                searcher.funnel["l3_attempts"] += 1
                t0 = time.time()
                ok = searcher.l3(sl)
                dt = time.time() - t0
                searcher.l3_marked_gens.append((gen, ok))
                tag = "[OK] L3 通过" if ok else "[NO] L3 失败"
                if verbose:
                    print(f"  | try_l3({sl.layout_score:.2f}, "
                          f"counts={sl.grasp_counts}) {tag}  ({dt:.1f}s)")
                    if not ok:
                        print(f"    reason: {sl.fail_reason}")
                if ok:
                    searcher.funnel["l3_pass"] += 1
                    return sl

        # 用精英更新 (mu, sigma)，新一代采样
        for pid in pids:
            ps = np.array([e.xy[pid] for e in elite])
            mu[pid] = ps.mean(axis=0)
            sigma[pid] = np.maximum(ps.std(axis=0) * sigma_decay, sigma_min)
        if verbose:
            sig_brief = "  ".join(
                f"{p}=σ{sigma[p][0]:.3f}/{sigma[p][1]:.3f}" for p in pids)
            print(f"  σ 更新: {sig_brief}")
        candidates = []
        # 保留当前最佳作为锚点（elitism）+ 新代高斯采样
        candidates.append(_ScoredLayout(xy=dict(best.xy)))
        for _ in range(pop_size - 1):
            xy = {pid: _clip_xy(rng.normal(mu[pid], sigma[pid]),
                                 bounds[pid]) for pid in pids}
            candidates.append(_ScoredLayout(xy=xy))

    return best_overall


# ══════════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════════
def _save_diagnostics(searcher: FastLayoutSearcher,
                       solution: Optional[_ScoredLayout],
                       out_dir: str,
                       *,
                       with_grasp_field: bool = True,
                       grasp_field_grid: int = 4):
    """画 5 张诊断图。

    画图按 "便宜的先画" 排序：topdown -> convergence -> arm -> funnel ->
    grasp_field（最慢，~1-3 min/件）。前 4 张几秒钟内就能落盘，即便
    后面被 Ctrl+C 也能拿到主结论；grasp_field 网格扫描默认 4x4 (16 点
    /件)，比早期 5x5 (25 点/件) 快 36%。设 ``with_grasp_field=False``
    可完全跳过这张。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
    except ImportError:
        print("[diagnostics] 未安装 matplotlib，跳过可视化。")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[diagnostics] 输出目录: {os.path.relpath(out_dir)}")

    pids = searcher.search_part_ids
    cmap = plt.get_cmap("tab10")
    pid_color = {pid: cmap(i) for i, pid in enumerate(pids)}

    # ── (1) Top-down 工作区 ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    if searcher.table_def is not None:
        tp = searcher.table_def["pos"]
        te = searcher.table_def["extent"]
        ax.add_patch(Rectangle(
            (tp[0] - te[0] / 2, tp[1] - te[1] / 2), te[0], te[1],
            facecolor="#f3ead0", edgecolor="#a08750", lw=1.5,
            label="table"))
    fp = searcher.task.fixture_pos
    ax.plot(fp[0], fp[1], "kP", markersize=14, label="fixture")
    rb = searcher.task.robot_base_pos
    ax.plot(rb[0], rb[1], "ko", markersize=10, label="robot base")
    for tag, dy in [("lft", 0.0), ("rgt", -0.597)]:
        ax.plot(rb[0], rb[1] + dy, "kX", markersize=10, alpha=0.5)
        ax.annotate(f"{tag}_arm", xy=(rb[0], rb[1] + dy),
                    xytext=(8, -8), textcoords="offset points",
                    fontsize=8, color="#555")
    for pid, (gp, _gr) in searcher.world_poses.items():
        if pid not in pids:
            continue
        ax.add_patch(Circle((gp[0], gp[1]), 0.025,
                            facecolor="none", edgecolor=pid_color[pid],
                            ls="--", lw=1.2))
        ax.annotate(f"{pid} (goal)", xy=(gp[0], gp[1]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7, color=pid_color[pid])
    for pid in pids:
        (xlo, xhi), (ylo, yhi) = searcher.bounds[pid]
        ax.add_patch(Rectangle(
            (xlo, ylo), xhi - xlo, yhi - ylo,
            facecolor=pid_color[pid], alpha=0.06,
            edgecolor=pid_color[pid], ls=":", lw=1.0))
    if solution is not None:
        for pid in pids:
            p = solution.xy.get(pid)
            if p is None:
                continue
            arm = solution.arm_choice.get(pid, "?")
            n = solution.grasp_counts.get(pid, 0)
            ax.plot(p[0], p[1], marker="o", markersize=14,
                    markerfacecolor=pid_color[pid], markeredgecolor="k", lw=0)
            ax.annotate(f"{pid} [{arm}, n={n}]",
                        xy=(p[0], p[1]),
                        xytext=(10, -16), textcoords="offset points",
                        fontsize=8, fontweight="bold")
            gp, _ = searcher.world_poses[pid]
            ax.plot([p[0], gp[0]], [p[1], gp[1]], "-",
                    color=pid_color[pid], alpha=0.5, lw=1.2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"[fast_layout] top-down: staging -> goal "
                  f"({searcher.task.name})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_workspace_topdown.png"), dpi=140)
    plt.close(fig)
    print("   [1/5] fig_workspace_topdown.png")

    # ── (2) 收敛曲线 ──────────────────────────────────
    if searcher.history:
        gens = [h[0] for h in searcher.history]
        bests = [h[1] for h in searcher.history]
        means = [h[2] for h in searcher.history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gens, bests, "o-", label="best of generation",
                color="#1f77b4")
        ax.plot(gens, means, "s--", label="elite mean",
                color="#ff7f0e")
        for g, ok in searcher.l3_marked_gens:
            ax.axvline(g, color="green" if ok else "red",
                       alpha=0.25, ls="-",
                       label="L3 try" if g == searcher.l3_marked_gens[0][0]
                       else None)
        ax.set_xlabel("generation")
        ax.set_ylabel("L2 score = min + 0.1 * mean(grasp_count)")
        ax.set_title("[fast_layout] CEM convergence")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_convergence.png"), dpi=140)
        plt.close(fig)
        print("   [2/5] fig_convergence.png")

    # ── (3) 臂分配饼图 ────────────────────────────────
    if solution is not None and solution.arm_choice:
        arm_counts: Dict[str, int] = {"lft": 0, "rgt": 0}
        for tag in solution.arm_choice.values():
            arm_counts[tag] = arm_counts.get(tag, 0) + 1
        fig, ax = plt.subplots(figsize=(5, 5))
        sizes = [arm_counts.get("lft", 0), arm_counts.get("rgt", 0)]
        labels = [f"lft_arm ({sizes[0]} parts)",
                  f"rgt_arm ({sizes[1]} parts)"]
        ax.pie(sizes, labels=labels, colors=["#4c9be8", "#e8884c"],
                autopct="%1.0f%%", startangle=90)
        ax.set_title("[fast_layout] arm assignment")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_arm_assignment.png"), dpi=140)
        plt.close(fig)
        print("   [3/5] fig_arm_assignment.png")

    # ── (4) 过滤漏斗 ──────────────────────────────────
    fnl = searcher.funnel
    stages = ["sampled", "L1 pass", "L2 pass", "L3 attempt", "L3 pass"]
    counts = [fnl["sampled"], fnl["l1_pass"], fnl["l2_pass"],
              fnl["l3_attempts"], fnl["l3_pass"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(stages[::-1], counts[::-1],
                   color=["#7fb069", "#4c9be8", "#bb87b6",
                          "#e8a07c", "#e85a5a"])
    for b, c in zip(bars, counts[::-1]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" {c}", va="center", fontsize=9)
    ax.set_title("[fast_layout] candidate pruning funnel")
    ax.set_xlabel("# candidates")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_pruning_funnel.png"), dpi=140)
    plt.close(fig)
    print("   [4/5] fig_pruning_funnel.png")

    # ── (5) Grasp 数热力图（最慢，放最后）─────────────
    # 5 件 x grasp_field_grid^2 次 reason_common_gids，每件 ~10-30s。
    # 用 try / KeyboardInterrupt 包裹，被中断时保留已画好的部分图。
    if with_grasp_field and solution is not None:
        try:
            ncols = min(3, len(pids))
            nrows = int(np.ceil(len(pids) / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(5 * ncols, 4.4 * nrows))
            axes = np.atleast_1d(axes).flatten()
            n_grid = max(2, int(grasp_field_grid))
            print(f"   [5/5] fig_grasp_field.png  扫描 {n_grid}x{n_grid} 网格 "
                  f"x {len(pids)} 件 = {n_grid * n_grid * len(pids)} 次 "
                  f"reason，预计 {n_grid * n_grid * len(pids) * 0.6:.0f}-"
                  f"{n_grid * n_grid * len(pids) * 1.2:.0f}s …")
            for k, pid in enumerate(pids):
                ax = axes[k]
                (xlo, xhi), (ylo, yhi) = searcher.bounds[pid]
                xs = np.linspace(xlo, xhi, n_grid)
                ys = np.linspace(ylo, yhi, n_grid)
                field = np.full((n_grid, n_grid), np.nan)
                t_part = time.time()
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        sl = _ScoredLayout(xy=dict(solution.xy))
                        sl.xy[pid] = np.array([x, y])
                        if not searcher.l1(sl):
                            continue
                        placed = set()
                        for s in searcher.asm.steps:
                            if s.part_id == pid:
                                break
                            if s.part_id in searcher.search_part_ids:
                                placed.add(s.part_id)
                        obs = searcher._step_aware_obs(pid, placed)
                        sp = searcher.staging_obs[pid].pos.copy()
                        sr = searcher.staging_obs[pid].rotmat.copy()
                        gp, gr = searcher.world_poses[pid]
                        gc = searcher.grasp_cache.get(
                            searcher.model_alias_fn(pid))
                        pref = _arm_priority_for_part(pid)
                        arm0 = (searcher.robot.lft_arm if pref[0] == "lft"
                                else searcher.robot.rgt_arm)
                        arm1 = (searcher.robot.lft_arm if pref[1] == "lft"
                                else searcher.robot.rgt_arm)
                        ok0, n0 = _reason_common_ok(
                            arm0, gc, sp, sr, gp, gr, obs)
                        if ok0:
                            field[iy, ix] = n0
                        else:
                            ok1, n1 = _reason_common_ok(
                                arm1, gc, sp, sr, gp, gr, obs)
                            if ok1:
                                field[iy, ix] = n1
                im = ax.imshow(field, origin="lower",
                               extent=[xlo, xhi, ylo, yhi],
                               cmap="viridis", aspect="auto")
                fig.colorbar(im, ax=ax, label="#common grasps")
                ax.set_title(f"{pid}: feasible-grasps field")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                p = solution.xy.get(pid)
                if p is not None:
                    ax.plot(p[0], p[1], "rx", markersize=12, mew=2)
                    ax.annotate(
                        f"selected n={solution.grasp_counts.get(pid, 0)}",
                        xy=(p[0], p[1]),
                        xytext=(8, -10), textcoords="offset points",
                        color="red", fontsize=8)
                print(f"          {pid:<8s} ({k + 1}/{len(pids)}) "
                      f"完成 {(time.time() - t_part):.1f}s")
            for k in range(len(pids), len(axes)):
                axes[k].axis("off")
            fig.suptitle(f"[fast_layout] grasp-count fields per part - "
                          f"{searcher.task.name}",
                         y=1.02, fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "fig_grasp_field.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)
            print("   [5/5] fig_grasp_field.png  done")
        except KeyboardInterrupt:
            print("   [5/5] fig_grasp_field.png  Ctrl+C 中断，跳过此图（其它已保存）")
            try:
                plt.close("all")
            except Exception:
                pass
    elif not with_grasp_field:
        print("   [5/5] fig_grasp_field.png  --no-grasp-field 跳过")

    print(f"\n[diagnostics] 已写入 {os.path.relpath(out_dir)}/")
    for f in ("fig_workspace_topdown.png", "fig_grasp_field.png",
              "fig_convergence.png", "fig_arm_assignment.png",
              "fig_pruning_funnel.png"):
        path = os.path.join(out_dir, f)
        if os.path.isfile(path):
            print(f"   - {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════
def run_fast_search(task: FastSearchTask,
                     *,
                     pop_size: int = 24,
                     elite_n: int = 6,
                     generations: int = 8,
                     try_l3_top_k: int = 0,
                     enable_l3: bool = False,
                     sigma_init: float = 0.08,
                     sigma_decay: float = 0.7,
                     rng_seed: int = 0,
                     visualize: bool = True,
                     with_grasp_field: bool = True,
                     grasp_field_grid: int = 4,
                     l2_retry: int = 0,
                     ik_retry_n: int = 0,
                     verbose: bool = True,
                     ) -> Tuple[Optional[str], Optional[_ScoredLayout]]:
    print("=" * 60)
    print(f"  Fast Layout Search - {task.name}")
    print(f"  pop={pop_size}  elite={elite_n}  gens={generations}  "
          f"try_l3_top_k={try_l3_top_k}  enable_l3={enable_l3}  "
          f"l2_retry={l2_retry}  ik_retry_n={ik_retry_n}  seed={rng_seed}")
    print("=" * 60)

    t0 = time.time()
    searcher = FastLayoutSearcher(task, enable_l3=enable_l3,
                                   ik_retry_n=ik_retry_n)
    solution = cem_search(
        searcher,
        pop_size=pop_size, elite_n=elite_n, generations=generations,
        sigma_init=sigma_init, sigma_decay=sigma_decay,
        try_l3_top_k=try_l3_top_k if searcher.enable_l3 else 0,
        rng_seed=rng_seed, l2_retry=l2_retry, verbose=verbose,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    if solution is not None and solution.l3_pass:
        print(f"[OK] L3 通过 layout，用时 {elapsed:.1f}s")
    elif solution is not None and solution.l2_pass:
        if searcher.enable_l3:
            print(f"[WARN] 未找到 L3 通过 layout，用时 {elapsed:.1f}s")
            print(f"  当前最佳 (仅 L2 通过) score={solution.layout_score:.2f}")
            print(f"  最近 L3 失败原因: {solution.fail_reason}")
        else:
            print(f"[OK] L2 最佳 layout，用时 {elapsed:.1f}s   "
                  f"(L3 未启用，加 --with-l3 做完整 RRT 校验)")
    else:
        print(f"[FAIL] 未能产生任何 L1+L2 通过的 layout，用时 {elapsed:.1f}s")
    print("=" * 60)

    if solution is not None and solution.l2_pass:
        print("最终 staging:")
        for pid in searcher.search_part_ids:
            p = solution.xy.get(pid)
            arm = solution.arm_choice.get(pid, "?")
            n = solution.grasp_counts.get(pid, 0)
            pos_s = (np.round(p, 3).tolist() if p is not None else "-")
            print(f"  {pid:<8s}  pos={pos_s}   arm={arm}   "
                  f"feasible_grasps={n}")

    out_path: Optional[str] = None
    if solution is not None and solution.l2_pass:
        out_dir = os.path.join(os.path.dirname(__file__), "_output")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "" if solution.l3_pass else (
            "" if not searcher.enable_l3 else "_l2only")
        out_path = os.path.join(
            out_dir, f"{task.output_layout_name}{suffix}.layout")
        layout = WorkspaceLayout(
            robot_base_pos=task.robot_base_pos.copy(),
            robot_base_rotmat=task.robot_base_rotmat.copy(),
            assembly_station_pos=task.fixture_pos.copy(),
            assembly_station_rotmat=task.fixture_rotmat.copy(),
            staging_positions={
                pid: (np.array([float(solution.xy[pid][0]),
                                 float(solution.xy[pid][1]), 0.0]),
                      np.eye(3))
                for pid in searcher.search_part_ids
                if pid in solution.xy
            },
            name=f"{task.output_layout_name}{suffix}",
            metadata={
                "search_method": "cem_fast",
                "arm_choice": dict(solution.arm_choice),
                "grasp_counts": dict(solution.grasp_counts),
                "l1_pass": True,
                "l2_pass": bool(solution.l2_pass),
                "l3_pass": bool(solution.l3_pass),
                "l3_enabled_in_search": bool(searcher.enable_l3),
                "elapsed_seconds": float(elapsed),
                "pop_size": int(pop_size),
                "generations": int(generations),
            },
        )
        layout.save(out_path)
        print(f"\n[OK] 保存 layout -> {os.path.relpath(out_path)}")

    if visualize:
        diag_dir = os.path.join(os.path.dirname(__file__),
                                 "_output", "fast_diagnostics", task.name)
        _save_diagnostics(searcher, solution, diag_dir,
                           with_grasp_field=with_grasp_field,
                           grasp_field_grid=grasp_field_grid)

    return out_path, solution


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
_TASK_REGISTRY = {"yuanchair": YUANCHAIR_FAST_TASK}


def _build_argparser():
    p = argparse.ArgumentParser(
        description="Fast Dual-Arm Layout Search via CEM + 三层评分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="yuanchair",
                   choices=list(_TASK_REGISTRY.keys()))
    p.add_argument("--pop", type=int, default=24, help="每代候选数")
    p.add_argument("--elite", type=int, default=6,
                   help="精英数（更新下代高斯均值/方差）")
    p.add_argument("--gens", type=int, default=8, help="最大代数")
    p.add_argument("--with-l3", action="store_true",
                   help="启用 L3 完整 RRT motion 校验（最严格，每次 ~30s）")
    p.add_argument("--try-l3-top-k", type=int, default=2,
                   help="启用 L3 时每代对前 K 个 L2 最佳跑一次 L3")
    p.add_argument("--sigma-init", type=float, default=0.08)
    p.add_argument("--sigma-decay", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-vis", action="store_true",
                   help="关闭 matplotlib 诊断图")
    p.add_argument("--no-grasp-field", action="store_true",
                   help="跳过最慢的 grasp_field 热力图（其它 4 张图照画）")
    p.add_argument("--grasp-field-grid", type=int, default=4,
                   help="grasp_field 网格分辨率 N（每件 NxN 次 reason）")
    p.add_argument("--l2-retry", type=int, default=0,
                   help="L2 评估时单步双臂均失败时重试次数，"
                        "专治 trac_ik 偶尔抽风（每次 ~1-3s/件）")
    p.add_argument("--ik-retry-n", type=int, default=1,
                   help="manipulator TracIK 主调用失败时再用 N 个随机 seed 重试，"
                        "粒度比 --l2-retry 更细。0=完全关闭（与原行为一致）")
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = _build_argparser().parse_args()
    task = _TASK_REGISTRY[args.task]
    out_path, solution = run_fast_search(
        task,
        pop_size=args.pop, elite_n=args.elite, generations=args.gens,
        try_l3_top_k=args.try_l3_top_k if args.with_l3 else 0,
        enable_l3=args.with_l3,
        sigma_init=args.sigma_init, sigma_decay=args.sigma_decay,
        rng_seed=args.seed,
        visualize=not args.no_vis,
        with_grasp_field=not args.no_grasp_field,
        grasp_field_grid=args.grasp_field_grid,
        l2_retry=args.l2_retry,
        ik_retry_n=args.ik_retry_n,
        verbose=not args.quiet,
    )

    if out_path is not None and solution is not None and solution.l2_pass:
        print("\n下一步：")
        if solution.l3_pass:
            print("  本 layout 已通过完整 motion 校验，可直接复制为 demo "
                  "默认加载文件名 dual_yuanchair_searched.layout 后执行：")
            print("    python -m sealp.examples.motion.dual_sequence_execution")
        else:
            print("  L1+L2 已通过；想跑完整 RRT 校验：")
            print("    python -m sealp.examples.layout.fast_layout_search "
                  "--with-l3")
            print("  或直接用本 layout 执行（demo 内部会自己跑 RRT）：")
            print("    1) 复制 -> dual_yuanchair_searched.layout")
            print("    2) python -m sealp.examples.motion.dual_sequence_execution")


if __name__ == "__main__":
    main()
