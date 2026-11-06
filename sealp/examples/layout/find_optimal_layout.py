"""
Fast Optimal Dual-Arm Layout Search (Random Sampling / CEM)
============================================================
两种搜索算法可选；都使用同一套综合评分标准：

  Layout Score
      = w1 * 抓取冗余度 (Grasp Count, 含 min/mean 双项)
      + w2 * 灵巧度    (Manipulability)
      + w3 * 移动代价  (Distance Cost, 负权重)

模式
----
* ``mode="random"`` (默认):
    在每件 staging 物的 (x,y) 盒内独立均匀随机采样 ``n_samples=20`` 个
    候选布局。每个候选走 L1 (快速可行性) + L2 (rotmat 枚举 + 抓取/灵巧度
    打分 + chosen rotmat pairwise 复检)。最终按 ``layout_score`` 排序
    返回最高分。优势：~10x 于 CEM；劣势：覆盖性弱于多代演化。

* ``mode="cem"`` (慢但更彻底):
    Cross-Entropy Method 多代演化；适合零件数较多 / bounds 较宽 / 需要
    L3 RRT 校验的场景。

* 不论哪种模式，L2 内层都会在 ``STAGING_ROTMAT_CANDIDATES`` 上枚举
  每件零件的姿态（如 leg 的直立 + 8 个躺姿），按"n_grasps 最高 + 直
  立优先"挑选；非中心对称零件的 (pos, rotmat, 桌面占位) 三件套是
  *联合* 评估的，不是分开打分。

机器人：双臂 Panthera-HT（``DualPantheraHTNoBody``，右臂 base y=-0.62）。
与下游 ``dual_sequence_execution.py`` / ``eval_dual_layout.py`` /
``search_dual_layout.py`` 完全对齐，写出的
``dual_yuanchair_optimal_searched.layout`` 会被
``dual_sequence_execution.py`` 自动加载并跑出动画。

CLI 用法
--------
    # 默认 random 模式 (20 样本) ：
    python find_optimal_layout.py
    # 加大样本数 / 换种子 ：
    python find_optimal_layout.py --n-samples 50 --seed 42
    # 跑慢但更彻底的 CEM ：
    python find_optimal_layout.py --mode cem --gens 8 --pop 24
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.manipulation.pick_place import PickPlacePlanner
import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda

from sealp.assembly_sequence import AssemblyDef
from sealp.colliders import StaticEnvironment
from sealp.config import load_config
from sealp.layout import WorkspaceLayout
from sealp.layout.dual_staging_search import find_obstacle_def
from sealp.examples.layout.eval_dual_layout import (
    STAGING_SEEDS, FIXTURE_POS, FIXTURE_ROTMAT,
    ROBOT_BASE_POS, ROBOT_BASE_ROTMAT,
    load_grasp_cache, model_alias_for_part,
)
from sealp.layout.reachability import check_pose_reachability

# Panthera-HT 双臂右臂 base 相对左臂的 y 偏移（见
# ``DualPantheraHTNoBody.__init__`` 的 ``arm_y_offset`` 默认值）。Piper 时代
# 这里是 -0.597；切换到 Panthera-HT 后必须同步，否则 staging 搜索的右半
# 区采样窗口会偏。
_DUAL_ARM_Y_OFFSET = 0.62

# ══════════════════════════════════════════════════════════════
#  评分权重配置 (可根据实际需求微调)
# ══════════════════════════════════════════════════════════════
#  归一化评分权重 + 物理量纲常量
# ══════════════════════════════════════════════════════════════
#  Layout Score = WEIGHT_GRASP      * grasp_score
#               + WEIGHT_MANIP_EP   * manip_ep_score        (端点平均, 旧 manip)
#               + WEIGHT_MANIP_TRAJ * manip_traj_score      (沿轨迹下界, 新)
#               + WEIGHT_DIST       * dist_score
#  四个 *_score 都 ∈ [0, 1]；WEIGHT_* 之和 = 1.0 → layout_score ∈ [0, 1]。
#
#  论文卖点："考虑装配全流程约束"。具体落地为两个互补的可操作性分量：
#    1. manip_ep_score  端点 (pick / place) 平均可操作性  —— 保证可抓且可放；
#    2. manip_traj_score  沿 pick→place 直线轨迹的最差 (min-along-path)
#       可操作性  —— 保证中段不穿过腕奇异。
#  二者结合等价于 Yoshikawa 路径可操作性下界 (1985)，是
#  layout-optimization / kinetostatic capability map / surgical-robot path
#  planning 共同沿用的指标。**单端点指标看不见中段穿奇异**，这正是把它
#  作为独立分量、且给更高权重 (0.20 vs 0.10) 的原因。
#
#  归一化原理（每项都用平滑曲线，全程保留梯度；target 是"半饱和点"）
#  --------------------------------------------------------------------------
#  * grasp_score: Hill(min(n), T_min, k)·w + Hill(mean(n), T_mean, k)·(1-w)
#                 Hill(x, T, k) = (x/T)^k / (1 + (x/T)^k)
#                 - n=0 → 0；n=T → 0.5；n=∞ → 1，**永不硬饱和**；
#                 - 指数 k 控制门槛锐利程度（k=2 是常用的"软门槛"曲线）；
#                 - min/mean 用 7:3 偏向瓶颈件，但 mean=0 时也能拿点分。
#  * manip_ep_score:   1 - exp(-avg_manip_ep / T_manip_ep)
#  * manip_traj_score: 1 - exp(-mean(traj_min_per_part) / T_manip_traj)
#                 - 与 ep 同形归一化，但喂的是「每件零件的轨迹最差 mu」的
#                   均值；T_manip_traj 比 T_manip_ep 略低，反映"最差点"
#                   通常偏低的统计事实。
#  * dist_score:  exp(-avg_dist / D_decay)
#                 - dist=0 → 1；dist=D → 0.368；dist=2D → 0.135；
#                 - 指数衰减，没有 dist>max 那种"突然归零"截断。
#
#  调权指引
#  --------
#  * WEIGHT_GRASP      ≥ 0.35（直接关联 RRT 成功率）；
#  * WEIGHT_MANIP_EP   端点远离奇异；0.08~0.12；
#  * WEIGHT_MANIP_TRAJ 整条 transport 远离奇异（论文主卖点）；0.15~0.25；
#  * WEIGHT_DIST       Panthera-HT 双臂建议 0.2~0.3。
WEIGHT_GRASP      = 0.30   # 抓取冗余度权重 ∈ [0, 1]
WEIGHT_MANIP_EP   = 0.10   # 端点灵巧度权重 ∈ [0, 1] (旧 manip)
WEIGHT_MANIP_TRAJ = 0.4   # 轨迹灵巧度权重 ∈ [0, 1] (论文主卖点：全流程约束)
WEIGHT_DIST       = 0.20   # 短运输权重    ∈ [0, 1] (正向；越短得分越高)

# 向后兼容别名：旧代码 / 诊断 JSON 引用的 WEIGHT_MANIP 等同于端点分量
WEIGHT_MANIP = WEIGHT_MANIP_EP

# Hill function 半饱和点（n=T 时 Hill = 0.5；T 之上仍有梯度趋于 1.0）
NORM_GRASP_MIN_TARGET  = 15.0   # min(n) 达到 15 视为"刚好充分"
NORM_GRASP_MEAN_TARGET = 25.0   # mean(n) 达到 25 视为"刚好充分"
NORM_GRASP_HILL_K      = 2.0    # Hill 指数 k；越大门槛越锐利 (k=2 常用)
NORM_GRASP_MIN_WEIGHT  = 0.7    # grasp_score 内 min:mean 比例 (7:3 偏向瓶颈)

# Manipulability 指数饱和参数（avg=T 时 1-exp(-1) ≈ 0.632）
NORM_MANIP_TARGET      = 0.030  # 端点 avg_manip 半饱和点
NORM_TRAJ_MANIP_TARGET = 0.020  # 轨迹 min_along_path 半饱和点（比端点低）

# 轨迹离散点数（含端点）。N=8 ⇒ 7 段 ⇒ pick + place + 6 中间点；
# 经验上对 6-DoF 直线 transport 已足够暴露中段奇异，且每件多 6 次 IK，
# 总开销约比仅端点 +30%，仍在可用预算内。
N_TRAJ_WAYPOINTS = 12

# staging→goal 旋转角超过此阈值时，在 pick→place 中叠加正弦抬升弧，避免
# 薄板（shelf 竖放→平放）SLERP 中段扫过已装零件 / 桌面障碍。
TRAJ_LIFT_ANGLE_RAD = np.deg2rad(25.0)
TRAJ_LIFT_PEAK_M = 0.19


def _traj_lift_scale(alpha: float) -> float:
    """Smooth lift profile; exponent < 1 widens the high-clearance mid section."""
    return float(np.sin(np.pi * float(alpha)) ** 0.55)


def _traj_object_pos(sp: np.ndarray, gp: np.ndarray, alpha: float,
                     *, lift_peak: float, rot_ang: float) -> np.ndarray:
    """Object position at path parameter alpha (0=pick, 1=place)."""
    a = float(alpha)
    p = (1.0 - a) * np.asarray(sp, dtype=float) + a * np.asarray(gp, dtype=float)
    p = np.asarray(p, dtype=float).copy()
    if lift_peak <= 0.0:
        return p
    xy_span = float(np.linalg.norm((np.asarray(gp) - np.asarray(sp))[:2]))
    # 水平槽插入：接近 goal 前保持高 z（仅小角度旋转；大角 shelf 在 L2 跳过 traj 探针）
    slot_insert = rot_ang <= TRAJ_LIFT_ANGLE_RAD and xy_span > 0.12
    if slot_insert:
        z_high = max(float(sp[2]), float(gp[2])) + lift_peak
        if a <= 1e-9:
            return np.asarray(sp, dtype=float)
        if a < 0.12:
            t = a / 0.12
            p[2] = (1.0 - t) * float(sp[2]) + t * z_high
            return p
        if a < 0.78:
            p[2] = z_high
            return p
        t = (a - 0.78) / 0.22
        p[2] = (1.0 - t) * z_high + t * float(gp[2])
        return p
    p[2] += lift_peak * _traj_lift_scale(a)
    return p

# 每件 part 轨迹验证最多尝试多少个 grasp（valid_gids 通常几十~几百）。
# shelf 竖放→平放时可行 grasp 分散，6 太保守易漏解；--easy 模式会临时调大。
TRAJ_GRASP_TRY_LIMIT = 6
DEBUG_L2_FAIL = False
# shelf_unit：L2 在 shelf_m / shelf_t 失败时打印逐步 grasp 漏斗对比
SHELF_FUNNEL_COMPARE = False
# Distance 指数衰减参数（dist=D 时 exp(-1) ≈ 0.368）
NORM_DIST_DECAY = 0.5           # 衰减距离 ≈ 桌面半边长 ~0.5m

# 同臂 staging 件之间的最小中心距 [m]。Panthera-HT 右臂 base 在 y=-0.62，
# 当两件 staging 都在 [y ∈ -0.71~-0.58] 这种"夹在右臂工作区中段"位置
# 时，PickPlacePlanner 的 RRT 复验经常踩到对方（实测 leg_br/leg_fr 中心距
# 0.128m 直接挂"robot_vs_obstacle"）。L2 端点 reasoning 看不出来——所以
# 这里强制每个手臂自己处理的两件 staging 中心距 ≥ 0.18m。0.18 是经验值：
# 比 leg cdprim 直径 (≈0.06m) + EE 半宽 (≈0.06m) × 2 略宽，给 RRT 留余量。
MIN_SAME_ARM_DIST = 0.05

# ══════════════════════════════════════════════════════════════
#  几何流常量
# ══════════════════════════════════════════════════════════════
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0])
PICK_DEPART_DIST = 0.05
PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0])
PLACE_APPROACH_DIST = 0.05
LEG_PLACE_DEPART_DIR = np.array([-1.0, 0.0, 0.0])
SEAT_PLACE_DEPART_DIR = np.array([0.0, 0.0, 1.0])
LEG_PLACE_DEPART_DIST = 0.05
SEAT_PLACE_DEPART_DIST = 0.05
APPROACH_DIST = 0.0
LINEAR_GRANULARITY = 0.04
HOME_JV = np.zeros(6)


def _part_motion_params(part_id: str) -> dict:
    """YuanChair 用 seat/leg 方向；ShelfUnit 用 ``shelf_motion`` 专用方向。"""
    from sealp.examples.layout._tasks.shelf_motion import (
        is_shelf_unit_part, motion_params, transport_kwargs,
    )
    if is_shelf_unit_part(part_id):
        mp = motion_params(part_id)
        tk = transport_kwargs(part_id)
        return dict(
            pick_depart_dir=mp.pick_depart_dir,
            pick_depart_dist=mp.pick_depart_dist,
            place_approach_dir=mp.place_approach_dir,
            place_approach_dist=mp.place_approach_dist,
            place_depart_dir=mp.place_depart_dir,
            place_depart_dist=mp.place_depart_dist,
            transport=tk,
        )
    pd_dir = LEG_PLACE_DEPART_DIR if part_id.startswith("leg_") else SEAT_PLACE_DEPART_DIR
    pd_dist = LEG_PLACE_DEPART_DIST if part_id.startswith("leg_") else SEAT_PLACE_DEPART_DIST
    return dict(
        pick_depart_dir=PICK_DEPART_DIR,
        pick_depart_dist=PICK_DEPART_DIST,
        place_approach_dir=PLACE_APPROACH_DIR,
        place_approach_dist=PLACE_APPROACH_DIST,
        place_depart_dir=pd_dir,
        place_depart_dist=pd_dist,
        transport=dict(
            pick_depart_direction=PICK_DEPART_DIR,
            pick_depart_distance=PICK_DEPART_DIST,
            place_approach_direction_list=[PLACE_APPROACH_DIR],
            place_approach_distance_list=[PLACE_APPROACH_DIST],
            place_depart_direction_list=[pd_dir],
            place_depart_distance_list=[pd_dist],
        ),
    )


def _patch_rrt_for_l3_relaxed() -> None:
    """L3 与 ``dual_sequence_execution_shelf`` 对齐的 RRT 稀疏/超时。"""
    from wrs.motion.probabilistic.rrt_connect import RRTConnect
    if getattr(RRTConnect.plan, "_sealp_l3_relaxed_patched", False):
        return
    _orig = RRTConnect.plan

    def _patched(self, *args, **kwargs):
        kwargs["ext_dist"] = 0.30
        kwargs["smoothing_n_iter"] = 150
        kwargs["max_time"] = 10.0
        return _orig(self, *args, **kwargs)

    _patched._sealp_l3_relaxed_patched = True
    RRTConnect.plan = _patched


# ══════════════════════════════════════════════════════════════
#  STAGING rotmat 候选：腿允许"立着 / 躺着 / 斜着躺"
# ══════════════════════════════════════════════════════════════
#  这一节与 ``dual_sequence_execution.py`` 中的同名常量逐字对齐，
#  保证 CEM 搜出来的 (xy, rotmat) 在执行端 ``_validate_loaded_layout``
#  里第一个 rot 候选就能直通过，不需要再切换姿态。
#  * 直立 = rotmat I（cylinder 长轴沿世界 +Z）
#  * 躺姿 = Ry(−90°) 把长轴放到水平，再加 yaw(Rz) 斜角
#  * 躺姿 z 抬 ``_LYING_Z_OFFSET`` 防止半埋桌面
#  * seat 只直立，不参与躺姿搜索
_LYING_Z_OFFSET = 0.02  # m，躺姿 staging 的 z 抬升


def _make_lying_leg_rotmats(
    yaw_deg_list=(0, 45, -45, 90, -90, 135, -135, 180),
):
    """生成"桌腿水平躺着 + 任意 yaw 斜角"的 rotmat 列表（不含直立）。"""
    base_lying = rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(-90.0))
    out = []
    for yaw_deg in yaw_deg_list:
        yaw_R = rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(yaw_deg))
        out.append(yaw_R @ base_lying)
    return out


# 每件零件的 rotmat 候选列表 = [(rotmat, z_offset)]。
# 顺序约定："直立"放最前；CEM L2 内层按顺序枚举，相同 n_grasps 时
# 同分先到先得，因此天然偏好"现已直立可行→保持直立"的解，只有当
# 直立全部 fail 时才回退到躺姿，与 dual demo 的兜底策略一致。
_LEG_ROTMAT_CANDS: List[Tuple[np.ndarray, float]] = (
    [(np.eye(3), 0.0)]                                            # 直立优先
    + [(R, _LYING_Z_OFFSET) for R in _make_lying_leg_rotmats()]   # 8 躺姿
)
_SEAT_ROTMAT_CANDS: List[Tuple[np.ndarray, float]] = [(np.eye(3), 0.0)]

STAGING_ROTMAT_CANDIDATES: Dict[str, List[Tuple[np.ndarray, float]]] = {
    "seat":   _SEAT_ROTMAT_CANDS,
    "leg_bl": _LEG_ROTMAT_CANDS,
    "leg_br": _LEG_ROTMAT_CANDS,
    "leg_fl": _LEG_ROTMAT_CANDS,
    "leg_fr": _LEG_ROTMAT_CANDS,
}


def _has_multi_rot(pid: str) -> bool:
    """该 pid 是否有多种 rotmat 候选（在 L1 pairwise 检查里要跳过）。"""
    return len(STAGING_ROTMAT_CANDIDATES.get(pid, _SEAT_ROTMAT_CANDS)) > 1


def _classify_pose(rotmat: np.ndarray) -> str:
    """根据 rotmat 第三列（本地 +Z 在世界中的方向）判定 staging 姿态。"""
    z_world = np.asarray(rotmat)[:, 2]
    return "直立" if abs(float(z_world[2])) > 0.85 else "躺/斜"


# ══════════════════════════════════════════════════════════════
#  任务规范
# ══════════════════════════════════════════════════════════════
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
    part_xy_bounds: Optional[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    model_alias_fn: Optional[Callable[[str], str]] = None
    output_layout_name: str = "dual_optimal_searched"


YUANCHAIR_FAST_TASK = FastSearchTask(
    name="yuanchair",
    asmdef_path=_here("..", "..", "assembly_sequence", "_demo_output", "yuanchair.asmdef"),
    grasp_pickles={
        "leg_model": _here("..", "grasp", "_output", "demo_yuanchair-part2_grasps.pickle"),
        "seat_model": _here("..", "grasp", "_output", "demo_yuanchair-part1_grasps.pickle"),
    },
    part_ids=("seat", "leg_bl", "leg_br", "leg_fl", "leg_fr"),
    staging_seeds=STAGING_SEEDS,
    fixture_pos=FIXTURE_POS,
    fixture_rotmat=FIXTURE_ROTMAT,
    robot_base_pos=ROBOT_BASE_POS,
    robot_base_rotmat=ROBOT_BASE_ROTMAT,
    config_yaml_path=_here("..", "..", "config", "sample_config.yaml"),
    model_alias_fn=model_alias_for_part,
    output_layout_name="dual_yuanchair_optimal_searched",
)


# ══════════════════════════════════════════════════════════════
#  几何 helper
# ══════════════════════════════════════════════════════════════
def _step_parent_id(asm, part_id):
    for s in asm.steps:
        if s.part_id == part_id:
            return s.parent_id
    return None


def _reset_robot_for_l3(robot) -> None:
    """PickPlace/RRT 规划会 mutate hold 状态；L3 每次尝试前后需清干净。"""
    for arm in (robot.lft_arm, robot.rgt_arm):
        ee = arm.end_effector
        if ee is not None:
            ee.oiee_list = []
            ee.oiee_list_bk.clear()
            ee.oiee_pose_list_bk.clear()
        arm.goto_given_conf(HOME_JV)


def _arms_collide_at_home(robot, obstacle_list) -> bool:
    if not obstacle_list: return False
    for arm in (robot.lft_arm, robot.rgt_arm):
        hit = arm.is_collided(obstacle_list=list(obstacle_list))
        collided = hit[0] if isinstance(hit, tuple) else hit
        if collided: return True
    return False


def _segment_reachable(arm, grasp, base_pos, base_rot, direction, distance,
                       n_samples: int = 4, obstacle_list=None) -> bool:
    """沿 ``direction``×``distance`` 做线性段中点采样的 IK + collision 验证。

    Args:
        n_samples: 在 (0, distance] 上等间隔取 ``n_samples`` 个采样点（端点 0
            不再重检——``reason_common_gids`` 已经查过）。
        obstacle_list: 与 ``reason_common_gids`` 对齐的 obstacle 列表。**只查
            IK 解存在还不够**：解出的 joint values 可能恰好让某个 link 穿过
            邻位 staging（右臂从 base 跨到 leg_br staging 时擦过 leg_fr 就是
            这种）；这里 ``goto_given_conf`` 后 ``is_collided`` 兜底，把"端
            点端 ok 但线性段中段撞"的 grasp id 在 L2 就过滤掉。
    """
    if distance is None or distance <= 1e-6 or n_samples < 1: return True
    direction = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1e-9: return True
    dir_unit = direction / n
    tcp_pos = base_rot.dot(grasp.ac_pos) + base_pos
    tcp_rot = base_rot.dot(grasp.ac_rotmat)
    seed = arm.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
    if seed is None: return False
    has_obs = bool(obstacle_list)
    for i in range(1, n_samples + 1):
        d = distance * i / n_samples
        end_pos = tcp_pos + dir_unit * d
        jv = arm.ik(tgt_pos=end_pos, tgt_rotmat=tcp_rot, seed_jnt_values=seed)
        if jv is None: return False
        if has_obs:
            arm.backup_state()
            try:
                arm.goto_given_conf(jnt_values=jv)
                if arm.is_collided(obstacle_list=obstacle_list):
                    return False
            finally:
                arm.restore_state()
        seed = jv
    return True


def _pick_depart_candidates(part_id: str) -> List[Tuple[np.ndarray, float]]:
    """part 对应的 pick 撤离方向列表（shelf 多候选 OR 通过）。"""
    from sealp.examples.layout._tasks.shelf_motion import (
        is_shelf_unit_part, pick_depart_candidates,
    )
    if is_shelf_unit_part(part_id or ""):
        return pick_depart_candidates(part_id)
    mot = _part_motion_params(part_id or "seat")
    return [(np.asarray(mot["pick_depart_dir"], dtype=float),
             float(mot["pick_depart_dist"]))]


def _place_approach_candidates(part_id: str) -> List[Tuple[np.ndarray, float]]:
    from sealp.examples.layout._tasks.shelf_motion import (
        is_shelf_unit_part, place_approach_candidates,
    )
    if is_shelf_unit_part(part_id or ""):
        return place_approach_candidates(part_id)
    mot = _part_motion_params(part_id or "seat")
    rev = -np.asarray(mot["place_approach_dir"], dtype=float)
    return [(rev, float(mot["place_approach_dist"]))]


def _gids_survive_pick_depart(arm, gc, gids, sp, sr, part_id: str,
                              obstacle_list) -> List[int]:
    """保留至少一种 pick 撤离方向可达的 grasp id。"""
    if not gids:
        return []
    surviving: List[int] = []
    for g in gids:
        for d, dist in _pick_depart_candidates(part_id):
            if _segment_reachable(
                arm, gc[g], sp, sr, d, dist, obstacle_list=obstacle_list,
            ):
                surviving.append(g)
                break
    return surviving


def _gids_survive_place_approach(arm, gc, gids, gp, gr, part_id: str,
                                obstacle_list) -> List[int]:
    if not gids:
        return []
    surviving: List[int] = []
    for g in gids:
        for d, dist in _place_approach_candidates(part_id):
            if _segment_reachable(
                arm, gc[g], gp, gr, d, dist, obstacle_list=obstacle_list,
            ):
                surviving.append(g)
                break
    return surviving


def _reason_common_ok(arm, gc, sp, sr, gp, gr, obstacle_list,
                      part_id: str = "",
                      return_reason: bool = False):
    """
    检查 staging pose -> goal pose 是否存在共同抓取。

    return_reason=True 时额外返回失败原因，方便调试：
        no_grasp_collection
        no_common_gids
        pick_depart_unreachable
        place_approach_unreachable
        ok
    """
    if gc is None or len(gc) == 0:
        if return_reason:
            return False, 0, [], "no_grasp_collection"
        return False, 0, []

    planner = PickPlacePlanner(robot=arm)

    gids_raw = planner.reason_common_gids(
        grasp_collection=gc,
        goal_pose_list=[(sp, sr), (gp, gr)],
        obstacle_list=obstacle_list,
    )

    if not gids_raw:
        if return_reason:
            # 进一步诊断：无障碍时是否有 common grasp
            gids_no_obs = planner.reason_common_gids(
                grasp_collection=gc,
                goal_pose_list=[(sp, sr), (gp, gr)],
                obstacle_list=[],
            )

            # 只保留环境障碍物，例如 work_table
            env_obs = [
                o for o in obstacle_list
                if getattr(o, "_sealp_role", None) == "environment_obstacle"
            ]
            gids_env_only = planner.reason_common_gids(
                grasp_collection=gc,
                goal_pose_list=[(sp, sr), (gp, gr)],
                obstacle_list=env_obs,
            )

            # 去掉未装 staging，只保留环境 + 已装件
            no_staging_obs = [
                o for o in obstacle_list
                if getattr(o, "_sealp_role", None) != "fast_staging"
            ]
            gids_no_staging = planner.reason_common_gids(
                grasp_collection=gc,
                goal_pose_list=[(sp, sr), (gp, gr)],
                obstacle_list=no_staging_obs,
            )

            # 打印 obstacle 详情
            obs_info = []
            for o in obstacle_list:
                obs_info.append(
                    f"{getattr(o, '_sealp_part_id', None)}:"
                    f"{getattr(o, '_sealp_role', None)}"
                )

            return (
                False,
                0,
                [],
                "no_common_gids | "
                f"no_obs={len(gids_no_obs)} | "
                f"env_only={len(gids_env_only)} | "
                f"no_staging={len(gids_no_staging)} | "
                f"full_obs=0 | "
                f"obs={obs_info}"
            )

        return False, 0, []

    gids_after_common = list(gids_raw)

    gids_after_pick_depart = _gids_survive_pick_depart(
        arm, gc, gids_after_common, sp, sr, part_id, obstacle_list)

    if not gids_after_pick_depart:
        if return_reason:
            return (
                False,
                0,
                [],
                f"pick_depart_unreachable: common={len(gids_after_common)} -> 0",
            )
        return False, 0, []

    gids_after_place_approach = _gids_survive_place_approach(
        arm, gc, gids_after_pick_depart, gp, gr, part_id, obstacle_list)

    if not gids_after_place_approach:
        if return_reason:
            return (
                False,
                0,
                [],
                f"place_approach_unreachable: "
                f"common={len(gids_after_common)}, "
                f"after_pick_depart={len(gids_after_pick_depart)} -> 0",
            )
        return False, 0, []

    if return_reason:
        return (
            True,
            len(gids_after_place_approach),
            list(gids_after_place_approach),
            f"ok: common={len(gids_after_common)}, "
            f"after_pick_depart={len(gids_after_pick_depart)}, "
            f"after_place_approach={len(gids_after_place_approach)}",
        )

    return True, len(gids_after_place_approach), list(gids_after_place_approach)


def _compute_grasp_funnel(arm, gc, sp, sr, gp, gr, obstacle_list, *,
                          part_id: str = "") -> dict:
    """staging→goal 抓取漏斗：no_obs → env → no_staging → full → pick_depart → place。"""
    out = {
        "goal_z": float(np.asarray(gp, dtype=float)[2]),
        "no_obs": 0,
        "env_only": 0,
        "no_staging": 0,
        "full_common": 0,
        "pick_depart": 0,
        "place_approach": 0,
        "n_obs": len(obstacle_list),
        "fail_at": "ok",
    }
    if gc is None or len(gc) == 0:
        out["fail_at"] = "no_grasp_collection"
        return out

    planner = PickPlacePlanner(robot=arm)
    poses = [(np.asarray(sp, dtype=float), np.asarray(sr, dtype=float)),
             (np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))]

    gids_no_obs = planner.reason_common_gids(
        grasp_collection=gc, goal_pose_list=poses, obstacle_list=[])
    out["no_obs"] = len(gids_no_obs)

    env_obs = [
        o for o in obstacle_list
        if getattr(o, "_sealp_role", None) == "environment_obstacle"
    ]
    gids_env = planner.reason_common_gids(
        grasp_collection=gc, goal_pose_list=poses, obstacle_list=env_obs)
    out["env_only"] = len(gids_env)

    no_stg_obs = [
        o for o in obstacle_list
        if getattr(o, "_sealp_role", None) != "fast_staging"
    ]
    gids_no_stg = planner.reason_common_gids(
        grasp_collection=gc, goal_pose_list=poses, obstacle_list=no_stg_obs)
    out["no_staging"] = len(gids_no_stg)

    gids_full = planner.reason_common_gids(
        grasp_collection=gc, goal_pose_list=poses, obstacle_list=obstacle_list)
    out["full_common"] = len(gids_full)
    if not gids_full:
        out["fail_at"] = "no_common_gids"
        return out

    gids_pick = _gids_survive_pick_depart(
        arm, gc, list(gids_full), sp, sr, part_id, obstacle_list)
    out["pick_depart"] = len(gids_pick)
    if not gids_pick:
        out["fail_at"] = "pick_depart"
        return out

    gids_place = _gids_survive_place_approach(
        arm, gc, gids_pick, gp, gr, part_id, obstacle_list)
    out["place_approach"] = len(gids_place)
    if not gids_place:
        out["fail_at"] = "place_approach"
        return out
    return out


def _staging_pose_for_funnel(sl, task, pid: str, rot_cands) -> Tuple[np.ndarray, np.ndarray]:
    """优先用已 commit 的 rotmat；否则用候选 [0]（mesh 竖立优先）。"""
    if pid in sl.xy:
        sp_xy = np.asarray(sl.xy[pid], dtype=float)
    elif pid in task.staging_seeds:
        sp_xy = np.asarray(task.staging_seeds[pid], dtype=float)
    else:
        raise KeyError(
            f"staging xy missing for {pid!r} (not in layout sample or task.staging_seeds)"
        )
    if pid in sl.chosen_rotmat:
        rot = np.asarray(sl.chosen_rotmat[pid], dtype=float)
        z_off = float(sl.z_offset.get(pid, 0.0))
    else:
        rot, z_off = rot_cands[0]
        rot = np.asarray(rot, dtype=float)
        z_off = float(z_off)
    sp = np.array([float(sp_xy[0]), float(sp_xy[1]), z_off], dtype=float)
    return sp, rot


def _funnel_rot_cands(pid: str) -> List[Tuple[np.ndarray, float]]:
    """shelf 漏斗诊断用的 staging rotmat 候选（skip part 时仍可读 goal 对比）。"""
    if pid in STAGING_ROTMAT_CANDIDATES:
        return STAGING_ROTMAT_CANDIDATES[pid]
    if pid.startswith("shelf_") and "shelf_m" in STAGING_ROTMAT_CANDIDATES:
        return STAGING_ROTMAT_CANDIDATES["shelf_m"]
    return [(np.eye(3), 0.0)]


def _print_shelf_m_t_funnel_compare(searcher, sl, *, fail_pid: str, placed_at_fail: set) -> None:
    """同一块 shelf 板：对比 shelf_m / shelf_t 在不同 step 上下文下的 grasp 漏斗。"""
    if not SHELF_FUNNEL_COMPARE or searcher.task.name != "shelf_unit":
        return
    if not fail_pid.startswith("shelf_"):
        return

    gc = searcher.grasp_cache.get(searcher.model_alias_fn("shelf_m"))
    if gc is None:
        return

    active = set(searcher.search_part_ids)
    placed_now = set(placed_at_fail)
    placed_step1 = {"side_l"} if "side_l" in active else set()

    scenarios: List[Tuple[str, str, set]] = []
    if fail_pid == "shelf_t" and "shelf_m" in placed_now and "shelf_t" in active:
        if "shelf_m" in active:
            scenarios.append(("shelf_m", "step1(仅 side_l)", placed_step1))
            scenarios.append(("shelf_m", "step2(+shelf_m goal)", placed_now))
        scenarios.append(("shelf_t", "step2(+shelf_m goal)", placed_now))
    elif "shelf_m" in active:
        scenarios.append(("shelf_m", "step1(仅 side_l)", placed_step1))
        if "shelf_t" in active:
            scenarios.append(("shelf_t", "step1*(无 shelf_m goal)", placed_step1))

    scenarios = [
        (pid, ctx, ps) for pid, ctx, ps in scenarios
        if pid in active and pid in searcher.world_poses
    ]
    if not scenarios:
        return

    rows: List[Tuple[str, str, str, dict]] = []
    for pid, ctx_label, placed_set in scenarios:
        gp, gr = searcher.world_poses[pid]
        rot_cands = _funnel_rot_cands(pid)
        sp, rot = _staging_pose_for_funnel(sl, searcher.task, pid, rot_cands)
        for arm_tag in _arm_priority_for_part(pid, task_name=searcher.task.name):
            arm = (searcher.robot.lft_arm if arm_tag == "lft"
                   else searcher.robot.rgt_arm)
            obs = searcher._step_aware_obs(pid, placed_set)
            funnel = _compute_grasp_funnel(
                arm, gc, sp, rot, gp, gr, obs, part_id=pid)
            rows.append((pid, ctx_label, arm_tag, funnel))

    has_t = "shelf_t" in active
    title = ("shelf_m vs shelf_t" if has_t else "shelf_m only")
    print(f"\n      [shelf-funnel] 同一块板 {title}（staging 用已 commit 或 rot[0]）")
    print(f"        fail_at={fail_pid}  placed_now={sorted(placed_at_fail)}")
    hdr = (
        f"  {'part':7s} {'context':22s} {'arm':4s} {'gz':>5s} "
        f"{'no_obs':>6s} {'env':>4s} {'no_stg':>6s} {'full':>4s} "
        f"{'pick':>4s} {'place':>5s} {'fail@':12s}"
    )
    print(hdr)
    for pid, ctx, arm_tag, f in rows:
        mark = " <<" if pid == fail_pid else ""
        print(
            f"  {pid:7s} {ctx:22s} {arm_tag:4s} {f['goal_z']:5.2f} "
            f"{f['no_obs']:6d} {f['env_only']:4d} {f['no_staging']:6d} "
            f"{f['full_common']:4d} {f['pick_depart']:4d} {f['place_approach']:5d} "
            f"{f['fail_at']:12s}{mark}"
        )
    print("        列说明: no_obs=无障碍 common | env=仅桌面 | no_stg=无未装 staging | "
          "full=全障碍 common | pick/place=撤离/接近后剩余")


def _trajectory_manipulability(
        arm,
        grasp,
        sp: np.ndarray, sr: np.ndarray,
        gp: np.ndarray, gr: np.ndarray,
        obstacle_list: Optional[List] = None,
        seed_jnt_values: Optional[np.ndarray] = None,
        n_waypoints: int = N_TRAJ_WAYPOINTS,
) -> Tuple[bool, float, float, List[float]]:
    """Yoshikawa 可操作性沿 pick→place 运输轨迹的最差/平均值。

    论文卖点 "整个运动过程中保持较高的可操作性" 的实现：固定 grasp 在
    物体坐标系内，对**物体位姿**做插值（位置线性 + 姿态 SLERP），再按
    ``tcp = R_obj @ ac_pos + p_obj`` 得到每个 waypoint 的 TCP，做 warm-start
    IK + 碰撞 + ``μ_i = sqrt(det(J J^T))``。

    注意：必须在物体位姿空间插值，而不能在 TCP 世界坐标上线性插值——当
    staging rotmat ≠ goal rotmat（shelf 竖放→平放、yuanchair 腿躺→立）时，
    后者会在 waypoint 2~3 产生物理上不可能的姿态，导致假阴性 IK fail。
    """
    if obstacle_list is None:
        obstacle_list = []
    n_waypoints = max(2, int(n_waypoints))
    sr = np.asarray(sr, dtype=float)
    gr = np.asarray(gr, dtype=float)
    sp = np.asarray(sp, dtype=float)
    gp = np.asarray(gp, dtype=float)
    alphas = np.linspace(0.0, 1.0, n_waypoints)
    _, rot_ang = rm.axangle_between_rotmat(sr, gr)
    rot_ang = float(rot_ang) if np.isfinite(rot_ang) else 0.0
    xy_span = float(np.linalg.norm((gp - sp)[:2]))
    obj_rot_seq = list(rm.rotmat_slerp(sr, gr, n_waypoints))
    lift_peak = 0.0
    if rot_ang > TRAJ_LIFT_ANGLE_RAD:
        lift_peak = min(
            TRAJ_LIFT_PEAK_M,
            0.04 + 0.10 * float(rot_ang / np.pi),
        )
    if rot_ang <= TRAJ_LIFT_ANGLE_RAD and xy_span > 0.12:
        lift_peak = max(lift_peak, min(TRAJ_LIFT_PEAK_M, 0.14 + 0.35 * xy_span))
    obj_pos_seq: List[np.ndarray] = []
    for a in alphas:
        obj_pos_seq.append(
            _traj_object_pos(sp, gp, a, lift_peak=lift_peak, rot_ang=rot_ang))
    pos_seq = [R @ grasp.ac_pos + p for R, p in zip(obj_rot_seq, obj_pos_seq)]
    rot_seq = [R @ grasp.ac_rotmat for R in obj_rot_seq]

    has_obs = bool(obstacle_list)
    curve: List[float] = []
    seed = seed_jnt_values
    for i, (p, R) in enumerate(zip(pos_seq, rot_seq)):
        # waypoint 0 是 pick 端点本身：若调用者已经传入 seed_jnt_values
        # （来自 L2 端点 check 的 best_jnt_values），直接复用，避免再调一
        # 次 TracIK（2ms timeout 偶发 warm-start fail），保证端点连续性。
        if i == 0 and seed_jnt_values is not None:
            jv = seed_jnt_values
        else:
            jv = arm.ik(tgt_pos=p, tgt_rotmat=R, seed_jnt_values=seed)
            # warm-start fail 兜底：再用 home_conf 试一次
            if jv is None:
                jv = arm.ik(tgt_pos=p, tgt_rotmat=R, seed_jnt_values=None)
        if jv is None:
            return False, 0.0, 0.0, curve
        arm.backup_state()
        try:
            arm.goto_given_conf(jnt_values=jv)
            if has_obs and arm.is_collided(obstacle_list=obstacle_list):
                return False, 0.0, 0.0, curve
            mu = float(arm.manipulability_val())
        finally:
            arm.restore_state()
        if not np.isfinite(mu) or mu < 0.0:
            mu = 0.0
        curve.append(mu)
        seed = jv  # warm-start 下一个 waypoint，保证轨迹是同一 IK 分支
    arr = np.asarray(curve, dtype=float)
    return True, float(arr.min()), float(arr.mean()), curve


def _probe_trajectory_grasps(
        arm,
        gc,
        sp: np.ndarray, sr: np.ndarray,
        gp: np.ndarray, gr: np.ndarray,
        obstacle_list: Optional[List],
        valid_gids: List[int],
        pick_gid: Optional[int],
        pick_jv: Optional[np.ndarray],
        *,
        max_tries: int,
        pick_best_tmin: bool = False,
) -> Tuple[bool, Optional[int], Optional[np.ndarray], float, float, List[float]]:
    """Try grasps for pick→place trajectory feasibility / scoring."""
    tried: List[int] = []
    if pick_gid is not None and int(pick_gid) in valid_gids:
        tried.append(int(pick_gid))
    for gid in valid_gids:
        gid_i = int(gid)
        if gid_i not in tried:
            tried.append(gid_i)
    tried = tried[:max(1, int(max_tries))]

    best: Optional[Tuple[int, Optional[np.ndarray], float, float, List[float]]] = None
    for gid_try in tried:
        if gid_try >= len(gc):
            continue
        ok, t_min, t_mean, t_curve = _trajectory_manipulability(
            arm, gc[gid_try], sp, sr, gp, gr,
            obstacle_list=obstacle_list,
            seed_jnt_values=(pick_jv if gid_try == pick_gid else None),
            n_waypoints=N_TRAJ_WAYPOINTS,
        )
        if not ok:
            continue
        seed_out = pick_jv if gid_try == pick_gid else None
        if not pick_best_tmin:
            return True, gid_try, seed_out, t_min, t_mean, t_curve
        if best is None or t_min > best[2]:
            best = (gid_try, seed_out, t_min, t_mean, t_curve)
    if best is not None:
        gid_try, seed_out, t_min, t_mean, t_curve = best
        return True, gid_try, seed_out, t_min, t_mean, t_curve
    return False, None, None, 0.0, 0.0, []


def _skip_traj_probe_for_part(task_name: str, part_id: str,
                              sr: np.ndarray, gr: np.ndarray) -> bool:
    """shelf 竖 staging→横 goal（≈120°）时线性 SLERP 轨迹探针假阴性过多，改由 L3/RRT 验证。"""
    if task_name != "shelf_unit" or not str(part_id).startswith("shelf_"):
        return False
    _, rot_ang = rm.axangle_between_rotmat(
        np.asarray(sr, dtype=float), np.asarray(gr, dtype=float))
    return float(rot_ang) > np.deg2rad(60.0)


def _arm_priority_for_part(part_id: str, *,
                           task_name: str = "") -> Tuple[str, ...]:
    """返回 L2 尝试顺序；shelf_unit 预检显示仅左臂可达。"""
    if task_name == "shelf_unit":
        return ("lft",)
    pid = str(part_id)
    return ("rgt", "lft") if pid and pid[-1] == 'r' else ("lft", "rgt")


# ══════════════════════════════════════════════════════════════
#  layout + 评分容器 (新增了灵巧度 / 距离 / 姿态指标)
# ══════════════════════════════════════════════════════════════
@dataclass
class _ScoredLayout:
    xy: Dict[str, np.ndarray]
    layout_score: float = -np.inf
    grasp_counts: Dict[str, int] = field(default_factory=dict)
    arm_choice: Dict[str, str] = field(default_factory=dict)
    # —— rotmat 维度：CEM 只采样 xy，rotmat 由 L2 内层枚举挑选并 commit ——
    chosen_rotmat: Dict[str, np.ndarray] = field(default_factory=dict)
    z_offset: Dict[str, float] = field(default_factory=dict)
    pose_tag: Dict[str, str] = field(default_factory=dict)
    manipulability_avg: float = 0.0       # 端点 avg manip（保留为对比项）
    distance_cost: float = 0.0
    # 论文主卖点：装配全流程约束 —— 轨迹可操作性下界
    traj_manip_min_avg: float = 0.0       # mean over parts of (min along path)
    # per-part 详细量（commit 后由 L2 填入；保留原始物理单位，方便日志/调试）
    per_part_manip: Dict[str, float] = field(default_factory=dict)   # 端点
    per_part_dist: Dict[str, float] = field(default_factory=dict)
    per_part_traj_min: Dict[str, float] = field(default_factory=dict)  # 轨迹 min
    per_part_traj_mean: Dict[str, float] = field(default_factory=dict) # 轨迹 mean
    per_part_traj_curve: Dict[str, List[float]] = field(default_factory=dict)
    # 归一化后的四分量（写诊断 JSON / 出图时使用，避免在外部重算）
    grasp_score_norm: float = 0.0
    manip_score_norm: float = 0.0            # 端点项归一化（旧）
    traj_manip_score_norm: float = 0.0       # 轨迹项归一化（新）
    dist_score_norm: float = 0.0
    l1_pass: bool = False
    l2_pass: bool = False
    l3_pass: bool = False
    fail_reason: str = ""
    fail_step_id: int = -1
    fail_part_id: str = ""
    l2_fail_breakdown: Dict[str, int] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  核心搜索器
# ══════════════════════════════════════════════════════════════
class FastLayoutSearcher:
    def __init__(self, task: FastSearchTask, *, enable_l3: bool = False, ik_retry_n: int = 0):
        self.task = task
        self.enable_l3 = enable_l3
        self.model_alias_fn = (task.model_alias_fn or (lambda pid: f"{pid}_model"))
        self._world = None

        self.asm = AssemblyDef.load(task.asmdef_path)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=task.fixture_pos, fixture_rotmat=task.fixture_rotmat)

        self.env_obs: List = []
        self.table_def: Optional[dict] = None
        if task.config_yaml_path and os.path.isfile(task.config_yaml_path):
            cfg = load_config(task.config_yaml_path)
            env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
            self.env_obs = list(env.obstacle_list)
            self.table_def = find_obstacle_def(cfg.obstacle_defs, task.table_obstacle_name)

        self.robot = pda.DualPantheraHTNoBody(
            pos=task.robot_base_pos, rotmat=task.robot_base_rotmat,
            arm_y_offset=_DUAL_ARM_Y_OFFSET, enable_cc=True)
        self.robot.lft_arm.goto_given_conf(HOME_JV)
        self.robot.rgt_arm.goto_given_conf(HOME_JV)

        if ik_retry_n > 0:
            for arm_robot in (self.robot.lft_arm.manipulator, self.robot.rgt_arm.manipulator):
                if hasattr(arm_robot, "_ik_retry_n"): arm_robot._ik_retry_n = ik_retry_n

        self.grasp_cache = load_grasp_cache(task.grasp_pickles)

        self.search_part_ids = [p for p in task.part_ids if p in task.staging_seeds]
        self.staging_obs: Dict[str, "mcm.CollisionModel"] = {}
        for pid in self.search_part_ids:
            mp = self.asm.model_path(pid)
            if not os.path.isfile(mp): continue
            cm = mcm.CollisionModel(initor=mp)
            cm.pos = np.zeros(3);
            cm.rotmat = np.eye(3)
            cm._sealp_part_id = pid;
            cm._sealp_role = "fast_staging"
            self.staging_obs[pid] = cm

        self.goal_obs: Dict[str, "mcm.CollisionModel"] = {}
        for pid, (gp, gr) in self.world_poses.items():
            if pid not in self.asm.part_ids: continue
            mp = self.asm.model_path(pid)
            if not os.path.isfile(mp): continue
            cm = mcm.CollisionModel(initor=mp)
            cm.pos = gp;
            cm.rotmat = gr
            cm._sealp_part_id = pid;
            cm._sealp_role = "fast_goal"
            self.goal_obs[pid] = cm

        self.bounds = self._resolve_xy_bounds()

        self.funnel = {"sampled": 0, "l1_pass": 0, "l2_pass": 0, "l3_attempts": 0, "l3_pass": 0}
        self.history: List[Tuple[int, float, float]] = []
        self.l3_marked_gens: List[Tuple[int, bool]] = []
        # 由 random_search 填充：每个样本完整诊断（含 fail_reason / per-component
        # 归一化分量）。供下游 diagnostics_plot.py 出图。
        self.all_samples: List[Dict] = []

    def _resolve_xy_bounds(self):
        if self.task.part_xy_bounds is not None: return dict(self.task.part_xy_bounds)
        if self.table_def is not None and self.table_def.get("type") == "box":
            tp, te = self.table_def["pos"], self.table_def["extent"]
            tx_lo = float(tp[0]) - float(te[0]) / 2 + self.task.table_margin
            tx_hi = float(tp[0]) + float(te[0]) / 2 - self.task.table_margin
            ty_lo = float(tp[1]) - float(te[1]) / 2 + self.task.table_margin
            ty_hi = float(tp[1]) + float(te[1]) / 2 - self.task.table_margin
        else:
            tx_lo, tx_hi, ty_lo, ty_hi = 0.0, 0.7, -1.0, 0.4
        # 与 ``DualPantheraHTNoBody.__init__`` 中 ``arm_y_offset=0.62`` 对齐。
        # 切换到 Panthera-HT 后：右臂 base 在 y=-0.62（Piper 时代是 -0.597），
        # 同时桌腿 cdprim 较粗，DFS 实测有效的 seed 右臂腿 y≈-0.85。这里
        # 把右臂区下扩到 rgt_y-0.42 / 上扩到 rgt_y+0.20 与 dual demo 对齐。
        lft_y, rgt_y = 0.0, -_DUAL_ARM_Y_OFFSET
        out = {}
        for pid in self.search_part_ids:
            x_lo, x_hi = max(tx_lo, 0.18), min(tx_hi, 0.55)
            if pid == "seat":
                out[pid] = ((max(x_lo, 0.22), min(x_hi, 0.40)), (-0.22, -0.02))
            elif pid.endswith("r"):
                out[pid] = ((x_lo, x_hi),
                            (max(ty_lo, rgt_y - 0.42), min(ty_hi, rgt_y + 0.20)))
            else:
                out[pid] = ((x_lo, x_hi),
                            (max(ty_lo, lft_y + 0.06), min(ty_hi, lft_y + 0.32)))
        return out

    def _apply_pose(self, sl: _ScoredLayout):
        """把 (xy + chosen_rotmat + z_offset) 套用到 staging cmodel。

        * L1 调用时 sl.chosen_rotmat 还是空字典 → 全部以 rotmat=I + z=0 套
          用（保守上界，作为"在直立摆放下也得通过"的初筛）。
        * L2 边跑边把 chosen_rotmat / z_offset commit 到 sl，后续 step 的
          obstacle_list 已经看到当前 step 的真实姿态。
        * L3 与"试跑真实 RRT"前调用，``sl.chosen_rotmat`` 已经填完。
        """
        for pid in self.search_part_ids:
            cm = self.staging_obs.get(pid)
            if cm is None: continue
            p = sl.xy.get(pid, self.task.staging_seeds[pid])
            rot = sl.chosen_rotmat.get(pid, np.eye(3))
            z_off = sl.z_offset.get(pid, 0.0)
            cm.pos = np.array([float(p[0]), float(p[1]), float(z_off)])
            cm.rotmat = np.asarray(rot)

    def l1(self, sl: _ScoredLayout) -> bool:
        """便宜的几何预筛。Pairwise 检查只对 rotmat 固定的 pid（如 seat）
        生效；两腿之间留给 L2 用真实选定的 rotmat 再复检 —— 否则会把
        "直立确实重叠但躺姿可解" 的 layout 误杀。
        """
        self._apply_pose(sl)
        ids = list(self.staging_obs.keys())
        for i in range(len(ids)):
            pi = ids[i]
            cm_i = self.staging_obs[pi]
            for j in range(i + 1, len(ids)):
                pj = ids[j]
                if _has_multi_rot(pi) and _has_multi_rot(pj):
                    continue  # 双方都可换 rotmat → L2 复检
                cm_j = self.staging_obs[pj]
                if cm_i.is_mcdwith(cm_j):
                    sl.fail_reason = f"L1: {pi} vs {pj} overlap"
                    return False
        for pid in ids:
            if _has_multi_rot(pid):
                continue  # rotmat 待定，留给 L2
            cm = self.staging_obs[pid]
            for gid, gcm in self.goal_obs.items():
                if gid == pid: continue
                if _step_parent_id(self.asm, gid) != "fixture": continue
                if cm.is_mcdwith(gcm):
                    sl.fail_reason = f"L1: {pid} 侵入 fixture goal {gid}"
                    return False
        # arm-vs-staging at home：用直立做下界检查；L2 内层会用真实 rotmat
        # 再硬查一次，躺姿延伸更远的真实占位由 L2 兜住。
        if _arms_collide_at_home(self.robot, [self.staging_obs[p] for p in ids]):
            sl.fail_reason = "L1: home 姿态下 staging(upright) 与机械臂穿模"
            return False
        sl.l1_pass = True
        return True

    def _step_aware_obs(self, current_pid: str, placed: set) -> List:
        obs = list(self.env_obs)
        for other in placed:
            if other in self.goal_obs: obs.append(self.goal_obs[other])
        for other in self.search_part_ids:
            if other == current_pid or other in placed: continue
            if other in self.staging_obs: obs.append(self.staging_obs[other])
        return obs

    # === 【核心修改：综合评分注入 + 多 rotmat 内层枚举】 ===
    def l2(self, sl: _ScoredLayout, *, retry: int = 0) -> bool:
        """按装配顺序逐 step 评估。每个 step 内层在 STAGING_ROTMAT_CANDIDATES
        上枚举，挑选"优先臂里 n_grasps 最高 + 同分时 manip 更优"的 rotmat。

        commit 顺序：本 step commit 后，后续 step 的 ``_step_aware_obs`` 看到
        的就是本 step 真实选定的 (rotmat, z_offset) — 与 dual demo DFS 同语义。
        """
        self._apply_pose(sl)
        placed: set = set()
        counts: Dict[str, int] = {}
        arms: Dict[str, str] = {}
        chosen_rotmats: Dict[str, np.ndarray] = {}
        z_offs: Dict[str, float] = {}
        pose_tags: Dict[str, str] = {}
        # per-part 详细分量（commit 后存入 sl，方便日志/调试）
        per_part_manip: Dict[str, float] = {}
        per_part_dist: Dict[str, float] = {}
        # 论文卖点：装配全流程约束 —— 每件 part 的轨迹可操作性下界
        per_part_traj_min: Dict[str, float] = {}
        per_part_traj_mean: Dict[str, float] = {}
        per_part_traj_curve: Dict[str, List[float]] = {}

        total_manip = 0.0
        total_dist = 0.0
        total_traj_min = 0.0

        for s in self.asm.steps:
            pid = s.part_id
            if pid not in self.search_part_ids or pid not in self.world_poses: continue

            sp_xy = sl.xy.get(pid, self.task.staging_seeds[pid])
            gp, gr = self.world_poses[pid]
            gc = self.grasp_cache.get(self.model_alias_fn(pid))
            if gc is None:
                sl.fail_reason = (
                    f"L2: 抓取库缺 {pid}, "
                    f"alias={self.model_alias_fn(pid)}, "
                    f"available_grasp_keys={list(self.grasp_cache.keys())}"
                )
                return False

            rot_cands = STAGING_ROTMAT_CANDIDATES.get(pid, [(np.eye(3), 0.0)])
            pref = _arm_priority_for_part(pid, task_name=self.task.name)

            # best = (n, arm_tag, rot, z_off, avg_manip, sp, pose_tag,
            #         pick_gid, pick_jv, valid_gids, t_min, t_mean, t_curve)
            best: Optional[Tuple[int, str, np.ndarray, float, float,
                                  np.ndarray, str, Optional[int],
                                  Optional[np.ndarray], List[int],
                                  float, float, List[float]]] = None
            traj_probe_tries = min(TRAJ_GRASP_TRY_LIMIT, 24)
            l2_fail_breakdown: Dict[str, int] = {}
            # 优先臂枚举所有 rotmat；若该臂找不到任何 feasible rotmat 才退到副臂
            for arm_tag in pref:
                arm_obj = (self.robot.lft_arm if arm_tag == "lft" else self.robot.rgt_arm)
                for rot_idx, (rot, z_off) in enumerate(rot_cands):
                    sp = np.array([float(sp_xy[0]), float(sp_xy[1]), float(z_off)])
                    self.staging_obs[pid].pos = sp
                    self.staging_obs[pid].rotmat = np.asarray(rot)
                    # 硬检查：home 姿态下双臂 vs 当前 rotmat 下所有 staging
                    home_obs = [self.staging_obs[p] for p in self.search_part_ids
                                if p in self.staging_obs]
                    if _arms_collide_at_home(self.robot, home_obs):
                        l2_fail_breakdown["home_arm_vs_staging_collision"] = (
                            l2_fail_breakdown.get("home_arm_vs_staging_collision", 0) + 1
                        )
                        if DEBUG_L2_FAIL:
                            print(
                                f"      [L2-debug] step={s.step_id} pid={pid} arm={arm_tag} "
                                f"rot_idx={rot_idx} pose={_classify_pose(np.asarray(rot))} "
                                f"z_off={float(z_off):.4f} "
                                f"sp={np.round(sp, 4).tolist()} "
                                f"fail=home_arm_vs_staging_collision "
                                f"home_obs={[p for p in self.search_part_ids if p in self.staging_obs]}"
                            )
                        continue
                    obs = self._step_aware_obs(pid, placed)
                    ok, n, valid_gids, reason_msg = _reason_common_ok(
                        arm_obj,
                        gc,
                        sp,
                        np.asarray(rot),
                        gp,
                        gr,
                        obs,
                        part_id=pid,
                        return_reason=True,
                    )

                    if not ok:
                        fail_key = str(reason_msg).split(":")[0].split("|")[0].strip()
                        l2_fail_breakdown[fail_key] = l2_fail_breakdown.get(fail_key, 0) + 1
                        if DEBUG_L2_FAIL:
                            print(
                                f"      [L2-debug] step={s.step_id} pid={pid} arm={arm_tag} "
                                f"rot_idx={rot_idx if 'rot_idx' in locals() else '?'} "
                                f"pose={_classify_pose(np.asarray(rot))} "
                                f"z_off={float(z_off):.4f} "
                                f"sp={np.round(sp, 4).tolist()} "
                                f"gp={np.round(gp, 4).tolist()} "
                                f"fail={reason_msg} "
                                f"n_obs={len(obs)}"
                            )
                        continue
                    pick_reach = check_pose_reachability(arm_obj, sp, np.asarray(rot), gc, obs, max_grasps=5)
                    place_reach = check_pose_reachability(arm_obj, gp, gr, gc, obs, max_grasps=5)
                    avg_manip = (pick_reach.best_manipulability + place_reach.best_manipulability) / 2.0
                    p_tag = _classify_pose(np.asarray(rot))
                    if _skip_traj_probe_for_part(self.task.name, pid, rot, gr):
                        traj_gid = int(valid_gids[0]) if valid_gids else pick_reach.best_grasp_id
                        traj_jv = (pick_reach.best_jnt_values.copy()
                                   if pick_reach.best_jnt_values is not None else None)
                        ok_traj = traj_gid is not None and traj_jv is not None
                        t_min = t_mean = float(avg_manip)
                        t_curve = [float(avg_manip)]
                    else:
                        ok_traj, traj_gid, traj_jv, t_min, t_mean, t_curve = _probe_trajectory_grasps(
                            arm_obj, gc, sp, np.asarray(rot), gp, gr, obs, valid_gids,
                            pick_reach.best_grasp_id, pick_reach.best_jnt_values,
                            max_tries=traj_probe_tries,
                            pick_best_tmin=False,
                        )
                    if not ok_traj:
                        l2_fail_breakdown["traj_probe"] = (
                            l2_fail_breakdown.get("traj_probe", 0) + 1
                        )
                        if DEBUG_L2_FAIL:
                            print(
                                f"      [L2-debug] step={s.step_id} pid={pid} arm={arm_tag} "
                                f"rot_idx={rot_idx} pose={p_tag} "
                                f"fail=traj_probe "
                                f"(tried up to {traj_probe_tries}/{len(valid_gids)} grasps)"
                            )
                        continue
                    cand = (n, arm_tag, np.asarray(rot).copy(), float(z_off),
                            float(avg_manip), sp.copy(), p_tag,
                            traj_gid,
                            (traj_jv.copy() if traj_jv is not None else None),
                            list(valid_gids),
                            float(t_min), float(t_mean), list(t_curve))
                    # n_grasps 大 > 轨迹 min manip 大 > 端点 manip 大 > 直立优先
                    def _key(c):
                        return (c[0], c[10], c[4], 1 if c[6] == "直立" else 0)
                    if best is None or _key(cand) > _key(best):
                        best = cand
                if best is not None:
                    break  # 优先臂已找到可行解，跳过副臂

            if best is None:
                self.staging_obs[pid].pos = np.array([float(sp_xy[0]), float(sp_xy[1]), 0.0])
                self.staging_obs[pid].rotmat = np.eye(3)

                sl.fail_step_id = int(s.step_id)
                sl.fail_part_id = str(pid)
                sl.l2_fail_breakdown = dict(l2_fail_breakdown)
                breakdown_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(
                        l2_fail_breakdown.items(), key=lambda kv: -kv[1])
                )
                n_arms = len(pref)
                arm_label = "单臂(lft)" if n_arms == 1 else f"{n_arms}臂"
                sl.fail_reason = (
                    f"L2: step={s.step_id} {pid} 所有 rotmat × {arm_label}均无可行 "
                    f"(reason+traj, tested {len(rot_cands)} rot × {n_arms} arm(s)). "
                    f"fail_breakdown=[{breakdown_str}]. "
                    f"xy={np.round(sp_xy, 4).tolist()}, "
                    f"goal_pos={np.round(gp, 4).tolist()}, "
                    f"grasp_alias={self.model_alias_fn(pid)}, "
                    f"grasp_num={len(gc) if gc is not None else 0}, "
                    f"placed={list(placed)}, "
                    f"obs_num={len(self._step_aware_obs(pid, placed))}"
                )

                if DEBUG_L2_FAIL:
                    print("      [L2-debug-summary]")
                    print(f"        step      = {s.step_id}")
                    print(f"        pid       = {pid}")
                    print(f"        xy        = {np.round(sp_xy, 4).tolist()}")
                    print(f"        goal_pos  = {np.round(gp, 4).tolist()}")
                    print(f"        grasp_key = {self.model_alias_fn(pid)}")
                    print(f"        grasp_num = {len(gc) if gc is not None else 0}")
                    print(f"        rot_num   = {len(rot_cands)}")
                    print(f"        arms      = {pref}")
                    print(f"        placed    = {list(placed)}")
                    print(f"        obs_num   = {len(self._step_aware_obs(pid, placed))}")

                _print_shelf_m_t_funnel_compare(
                    self, sl, fail_pid=pid, placed_at_fail=set(placed))

                return False

            (n, arm_tag, rot, z_off, avg_manip, sp, p_tag,
             pick_gid, pick_jv, valid_gids, t_min, t_mean, t_curve) = best
            # commit：写回 staging_obs 让后续 step 看见真实占位
            self.staging_obs[pid].pos = sp
            self.staging_obs[pid].rotmat = rot
            dist = float(np.linalg.norm(np.array(sp) - np.array(gp)))

            # 对胜出的 (rotmat, arm) 再用完整 grasp 池精化轨迹 min manip 评分
            if TRAJ_GRASP_TRY_LIMIT > traj_probe_tries:
                arm_obj_traj = (self.robot.lft_arm if arm_tag == "lft"
                                else self.robot.rgt_arm)
                obs_traj = self._step_aware_obs(pid, placed)
                ok_ref, ref_gid, ref_jv, ref_tmin, ref_tmean, ref_curve = (
                    _probe_trajectory_grasps(
                        arm_obj_traj, gc, sp, rot, gp, gr, obs_traj, valid_gids,
                        pick_gid, pick_jv,
                        max_tries=TRAJ_GRASP_TRY_LIMIT,
                        pick_best_tmin=True,
                    ))
                if ok_ref:
                    pick_gid, pick_jv = ref_gid, ref_jv
                    t_min, t_mean, t_curve = ref_tmin, ref_tmean, ref_curve

            counts[pid] = n
            arms[pid] = arm_tag
            chosen_rotmats[pid] = rot
            z_offs[pid] = z_off
            pose_tags[pid] = p_tag
            per_part_manip[pid] = float(avg_manip)
            per_part_dist[pid] = float(dist)
            per_part_traj_min[pid] = float(t_min)
            per_part_traj_mean[pid] = float(t_mean)
            per_part_traj_curve[pid] = [float(v) for v in t_curve]
            total_manip += avg_manip
            total_dist += dist
            total_traj_min += t_min
            placed.add(pid)

        sl.grasp_counts = counts
        sl.arm_choice = arms
        sl.chosen_rotmat = chosen_rotmats
        sl.z_offset = z_offs
        sl.pose_tag = pose_tags

        # ── 复检：用 L2 选定的 (chosen_rotmat, z_offset) 把所有 staging cm
        # 重新套一遍，再做 pairwise mcdwith。L1 为了让躺姿可解跳过了
        # "双方都可换 rotmat"的 pair；这里用真实占位最后兜一次，防止
        # leg_bl 直立 + leg_fl 躺姿/斜角恰好互相穿过 (用 rotmat=I 检查不
        # 出来的真实物体重叠)。
        self._apply_pose(sl)
        ids = list(self.staging_obs.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self.staging_obs[ids[i]].is_mcdwith(self.staging_obs[ids[j]]):
                    sl.fail_reason = (
                        f"L2 post-check: {ids[i]}({pose_tags.get(ids[i],'?')}) "
                        f"vs {ids[j]}({pose_tags.get(ids[j],'?')}) "
                        f"chosen-rotmat overlap")
                    return False

        # 同臂 staging 件最小距离 兜底（防止 RRT 复验擦邻位）：
        # L2 端点 reasoning 通过、_segment_reachable 也加了 collision 检查，
        # 但 RRT 中段轨迹仍可能擦过 0.13~0.17m 中心距的同臂邻位 staging。
        # 这里硬卡 MIN_SAME_ARM_DIST，把不可行 layout 直接在 L2 拒掉。
        same_arm_pairs: Dict[str, List[str]] = {"lft": [], "rgt": []}
        for pid, tag in arms.items():
            same_arm_pairs[tag].append(pid)
        for tag, plist in same_arm_pairs.items():
            for i in range(len(plist)):
                pi = plist[i]
                for j in range(i + 1, len(plist)):
                    pj = plist[j]
                    d_ij = float(np.linalg.norm(
                        np.asarray(sl.xy[pi], dtype=float)
                        - np.asarray(sl.xy[pj], dtype=float)))
                    if d_ij < MIN_SAME_ARM_DIST:
                        sl.fail_reason = (
                            f"L2 post-check: 同臂({tag}) {pi} vs {pj} 中心距"
                            f"={d_ij:.3f}m < MIN_SAME_ARM_DIST={MIN_SAME_ARM_DIST:.2f}m"
                            f"（RRT 复验会踩邻位）")
                        return False

        # per-part 详情写回 sl（即使下面 counts 为空也能 introspect）
        sl.per_part_manip = per_part_manip
        sl.per_part_dist = per_part_dist
        sl.per_part_traj_min = per_part_traj_min
        sl.per_part_traj_mean = per_part_traj_mean
        sl.per_part_traj_curve = per_part_traj_curve

        if counts:
            n_parts = len(counts)
            vs = list(counts.values())
            avg_manip = total_manip / n_parts
            avg_dist = total_dist / n_parts
            avg_traj_min = total_traj_min / n_parts  # mean over parts of min-along-path

            # === 归一化各分量到 [0, 1]，全程平滑保留梯度 ===
            # 1) Grasp: Hill function — 软门槛，无硬饱和
            #    Hill(x, T, k) = (x/T)^k / (1 + (x/T)^k)
            def _hill(x: float, T: float, k: float) -> float:
                if x <= 0.0: return 0.0
                r = (x / T) ** k
                return r / (1.0 + r)
            hill_min = _hill(float(min(vs)),
                             NORM_GRASP_MIN_TARGET, NORM_GRASP_HILL_K)
            hill_mean = _hill(float(sum(vs) / n_parts),
                              NORM_GRASP_MEAN_TARGET, NORM_GRASP_HILL_K)
            grasp_score = (
                NORM_GRASP_MIN_WEIGHT * hill_min
                + (1.0 - NORM_GRASP_MIN_WEIGHT) * hill_mean)
            # 2a) Manip-EP: 指数饱和（端点平均）
            manip_score = float(1.0 - np.exp(-max(avg_manip, 0.0)
                                              / NORM_MANIP_TARGET))
            # 2b) Manip-TRAJ: 指数饱和（沿轨迹下界，论文主卖点）
            traj_manip_score = float(1.0 - np.exp(
                -max(avg_traj_min, 0.0) / NORM_TRAJ_MANIP_TARGET))
            # 3) Dist: 指数衰减 exp(-x/D)，越短越好；无 dist>max 硬归零
            dist_score = float(np.exp(-max(avg_dist, 0.0) / NORM_DIST_DECAY))

            sl.manipulability_avg = avg_manip
            sl.distance_cost = avg_dist
            sl.traj_manip_min_avg = avg_traj_min
            sl.grasp_score_norm = float(grasp_score)
            sl.manip_score_norm = float(manip_score)
            sl.traj_manip_score_norm = float(traj_manip_score)
            sl.dist_score_norm = float(dist_score)
            sl.layout_score = (
                WEIGHT_GRASP      * grasp_score
                + WEIGHT_MANIP_EP   * manip_score
                + WEIGHT_MANIP_TRAJ * traj_manip_score
                + WEIGHT_DIST       * dist_score)

        sl.l2_pass = True
        return True

    def l3(self, sl: _ScoredLayout) -> bool:
        if not self.enable_l3: return False
        from sealp.primitives.transport import TransportPrimitive
        _patch_rrt_for_l3_relaxed()
        _reset_robot_for_l3(self.robot)
        try:
            # L3 必须用 L2 commit 后的 (xy, chosen_rotmat, z_offset)
            self._apply_pose(sl)
            lft_t = TransportPrimitive(self.robot.lft_arm)
            rgt_t = TransportPrimitive(self.robot.rgt_arm)
            placed: set = set()
            for s in self.asm.steps:
                pid = s.part_id
                if pid not in self.search_part_ids or pid not in self.world_poses: continue
                tag = sl.arm_choice.get(pid, "lft")
                transport = lft_t if tag == "lft" else rgt_t
                sp, sr = self.staging_obs[pid].pos.copy(), self.staging_obs[pid].rotmat.copy()
                gp, gr = self.world_poses[pid]
                mp = self.asm.model_path(pid)
                obj_cm = mcm.CollisionModel(initor=mp);
                obj_cm.pos = sp;
                obj_cm.rotmat = sr
                gc = self.grasp_cache.get(self.model_alias_fn(pid))
                obs = self._step_aware_obs(pid, placed)
                mot = _part_motion_params(pid)
                tk = mot["transport"]
                try:
                    res = transport.plan(
                        obj_cmodel=obj_cm, grasp_collection=gc, goal_pose_list=[(gp, gr)], obstacle_list=obs,
                        approach_distance=APPROACH_DIST, depart_distance=PICK_DEPART_DIST,
                        linear_granularity=LINEAR_GRANULARITY,
                        **tk,
                    )
                except Exception as e:
                    sl.fail_reason = f"L3: step {s.step_id} {pid} {tag} 抛错 {type(e).__name__}: {e!r}"
                    return False
                if not res.success:
                    sl.fail_reason = f"L3: step {s.step_id} {pid} {tag} 失败: {res.error_msg or 'no plan'}"
                    return False
                placed.add(pid)
            sl.l3_pass = True
            return True
        finally:
            _reset_robot_for_l3(self.robot)


# ══════════════════════════════════════════════════════════════
#  CEM 优化器
# ══════════════════════════════════════════════════════════════
def _clip_xy(p, bounds):
    (xlo, xhi), (ylo, yhi) = bounds
    return np.array([float(np.clip(p[0], xlo, xhi)), float(np.clip(p[1], ylo, yhi))])


def _seed_with_noise(seeds, pids, bounds, rng, scale):
    return {pid: _clip_xy(np.array([seeds[pid][0], seeds[pid][1]]) + rng.normal(0, scale, 2), bounds[pid]) for pid in
            pids}


def _summarize_search_failures(samples_diag: List[Dict], *,
                               asm_steps: Optional[List] = None,
                               verbose: bool = True) -> None:
    """汇总 random/CEM 全失败时的 L1/L2 漏斗，便于定位卡在哪一步。"""
    if not samples_diag or not verbose:
        return
    from collections import Counter

    n = len(samples_diag)
    l1_ok = [r for r in samples_diag if r.get("l1_pass")]
    l2_ok = [r for r in samples_diag if r.get("l2_pass")]
    l1_fail = n - len(l1_ok)
    l2_fail = len(l1_ok) - len(l2_ok)

    print("\n" + "=" * 60)
    print("[Failure Summary] 搜索失败漏斗")
    print("=" * 60)
    print(f"  总样本     = {n}")
    print(f"  L1 通过    = {len(l1_ok)}  |  L1 失败 = {l1_fail}")
    print(f"  L2 通过    = {len(l2_ok)}  |  L2 失败 = {l2_fail} (在 L1 通过样本中)")

    if l2_fail <= 0 and l1_fail <= 0:
        print("  (无失败样本可汇总)")
        return

    step_ctr: Counter = Counter()
    part_ctr: Counter = Counter()
    reason_ctr: Counter = Counter()
    breakdown_ctr: Counter = Counter()

    for rec in samples_diag:
        if rec.get("l1_pass") and not rec.get("l2_pass"):
            step = rec.get("fail_step_id", -1)
            part = rec.get("fail_part_id", "")
            if step >= 0:
                step_ctr[int(step)] += 1
            if part:
                part_ctr[str(part)] += 1
            reason = str(rec.get("fail_reason", ""))
            if "no_common_gids" in reason:
                reason_ctr["no_common_gids (staging↔goal 无共同抓取)"] += 1
            elif "pick_depart_unreachable" in reason:
                reason_ctr["pick_depart_unreachable (抓取后撤离不可达)"] += 1
            elif "home_arm_vs_staging" in reason:
                reason_ctr["home_arm_vs_staging_collision"] += 1
            elif "traj_probe" in reason:
                reason_ctr["traj_probe (轨迹探测失败)"] += 1
            else:
                reason_ctr["other"] += 1
            for k, v in rec.get("l2_fail_breakdown", {}).items():
                breakdown_ctr[str(k)] += int(v)

    if step_ctr:
        print("\n  ── L2 首次失败：装配 step ──")
        step_names = {}
        if asm_steps:
            for s in asm_steps:
                step_names[int(s.step_id)] = str(s.part_id)
        for step_id, cnt in step_ctr.most_common():
            pname = step_names.get(step_id, "?")
            print(f"    step {step_id} ({pname}): {cnt}/{l2_fail} 样本")

    if part_ctr:
        print("\n  ── L2 首次失败：零件 ──")
        for pid, cnt in part_ctr.most_common():
            print(f"    {pid}: {cnt}/{l2_fail} 样本")

    if reason_ctr:
        print("\n  ── 失败类型（按样本主因粗分）──")
        for tag, cnt in reason_ctr.most_common():
            print(f"    {tag}: {cnt}")

    if breakdown_ctr:
        print("\n  ── L2 内层 rot×arm 尝试失败计数（跨样本累加）──")
        for tag, cnt in breakdown_ctr.most_common(8):
            print(f"    {tag}: {cnt}")

    # 典型样本：第一个 L2 失败
    exemplar = next(
        (r for r in samples_diag if r.get("l1_pass") and not r.get("l2_pass")),
        None,
    )
    if exemplar:
        print("\n  ── 典型失败样本 #{} ──".format(exemplar.get("idx", "?")))
        print(f"    fail_step  = {exemplar.get('fail_step_id', '?')}")
        print(f"    fail_part  = {exemplar.get('fail_part_id', '?')}")
        bd = exemplar.get("l2_fail_breakdown", {})
        if bd:
            print(f"    breakdown  = {bd}")
        fr = str(exemplar.get("fail_reason", ""))
        if len(fr) > 220:
            fr = fr[:220] + "…"
        print(f"    reason     = {fr}")

    print("\n  提示：当前仅搜索 staging 的 (x,y)；fixture 位置固定。")
    print("        若 goal 仅单臂可达，请调整 fixture / 抓取过滤，或启用 --l2-debug。")
    print("=" * 60)


def random_search(searcher: FastLayoutSearcher, *,
                  n_samples: int = 20,
                  rng_seed: int = 0,
                  include_seed: bool = True,
                  try_l3_top_k: int = 0,
                  verbose: bool = True) -> Optional[_ScoredLayout]:
    """在每件 staging 物的 (x_bound, y_bound) 盒内随机采样 ``n_samples`` 个
    候选布局；每个候选走 L1 + L2；L2 内层会在
    ``STAGING_ROTMAT_CANDIDATES`` 上自动挑选 (n_grasps 最高 + 直立优先) 的
    rotmat 与 z_offset，并在末尾用 chosen rotmat 复检 pairwise 占位重叠。
    最终按 ``layout_score`` 排序，返回最高分作为 layout。

    设计动机：CEM 多代搜索对桌椅这种 5 件零件的小问题来说太重；实测 20 个
    均匀随机点 + L2 内层 rotmat 枚举已能覆盖足够多样本，单次跑只要
    ~20 × (L1 + L2)，是 CEM (8 代 × 24 = 192 评估) 的 ~1/10 时间。

    注意零件 *非中心对称* (例如 leg 的躺姿)：
        - L2 会把 ``(rotmat, z_offset)`` 套到 staging cm 上重算 footprint，
          所以"pos+rotmat+桌面占位"三件套是 *联合* 评估，不是分开打分；
        - L2 末尾的 pairwise mcdwith 复检确保两个躺/斜腿不会在桌面上
          实际穿模 (L1 为了让躺姿可解跳过了 multi_rot pair)。

    Args:
        n_samples: 总样本数 (含 seed anchor)。
        rng_seed: 随机种子，用于复现实验。
        include_seed: True 时第 0 个候选 = ``task.staging_seeds``
            (已知可行的 baseline；纯随机失败时仍能拿到结果)。
        verbose: True 时打印每个样本的 score / pose_tag / fail_reason。
    """
    rng = np.random.default_rng(rng_seed)
    pids = searcher.search_part_ids
    bounds = searcher.bounds
    seeds = searcher.task.staging_seeds

    candidates: List[_ScoredLayout] = []
    if include_seed:
        candidates.append(_ScoredLayout(xy={
            pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])])
            for pid in pids
        }))
    while len(candidates) < n_samples:
        xy = {}
        for pid in pids:
            (xlo, xhi), (ylo, yhi) = bounds[pid]
            xy[pid] = np.array([
                float(rng.uniform(xlo, xhi)),
                float(rng.uniform(ylo, yhi)),
            ])
        candidates.append(_ScoredLayout(xy=xy))

    n_rot_total = sum(
        len(STAGING_ROTMAT_CANDIDATES.get(p, [(np.eye(3), 0.0)]))
        for p in pids
    )
    if verbose:
        print(f"\n[Random] 评估 {n_samples} 个样本"
              f" (L2 内层每样本枚举 {n_rot_total} 个 (pid,rotmat) 组合,"
              f" seed={rng_seed})")

    scored: List[_ScoredLayout] = []
    # ── 诊断：每个样本完整记录（用于后续画图 / 论文证据） ──
    # 字段：idx, is_seed, xy, l1_pass, l2_pass, fail_reason；
    # 若 l2 通过，还有 score / grasp_counts / arm_choice / per_part_* / norm 三分量
    samples_diag: List[Dict] = []
    t_start = time.time()
    for idx, sl in enumerate(candidates):
        searcher.funnel["sampled"] += 1
        tag = " (seed anchor)" if (include_seed and idx == 0) else ""
        is_seed_anchor = bool(include_seed and idx == 0)
        rec: Dict = {
            "idx": int(idx),
            "is_seed_anchor": is_seed_anchor,
            "xy": {pid: [float(sl.xy[pid][0]), float(sl.xy[pid][1])]
                   for pid in pids},
            "l1_pass": False,
            "l2_pass": False,
            "fail_reason": "",
        }
        if not searcher.l1(sl):
            rec["fail_reason"] = str(sl.fail_reason)
            samples_diag.append(rec)
            if verbose:
                print(f"  #{idx:>2d}{tag}  L1 fail: {sl.fail_reason}")
            continue
        rec["l1_pass"] = True
        searcher.funnel["l1_pass"] += 1
        if not searcher.l2(sl):
            rec["fail_reason"] = str(sl.fail_reason)
            rec["fail_step_id"] = int(getattr(sl, "fail_step_id", -1))
            rec["fail_part_id"] = str(getattr(sl, "fail_part_id", ""))
            rec["l2_fail_breakdown"] = dict(getattr(sl, "l2_fail_breakdown", {}))
            samples_diag.append(rec)
            if verbose:
                print(f"  #{idx:>2d}{tag}  L1 ok, L2 fail: {sl.fail_reason}")
            continue
        rec["l2_pass"] = True
        searcher.funnel["l2_pass"] += 1
        scored.append(sl)
        # 写完整诊断
        rec.update({
            "score": float(sl.layout_score),
            "grasp_counts": {p: int(n) for p, n in sl.grasp_counts.items()},
            "arm_choice": dict(sl.arm_choice),
            "pose_tag": dict(sl.pose_tag),
            "per_part_manip": dict(sl.per_part_manip),
            "per_part_dist": dict(sl.per_part_dist),
            "avg_manip": float(sl.manipulability_avg),
            "avg_dist": float(sl.distance_cost),
            "grasp_score_norm": float(sl.grasp_score_norm),
            "manip_score_norm": float(sl.manip_score_norm),
            "traj_manip_score_norm": float(sl.traj_manip_score_norm),
            "dist_score_norm": float(sl.dist_score_norm),
            "traj_manip_min_avg": float(sl.traj_manip_min_avg),
            "per_part_traj_min": dict(sl.per_part_traj_min),
            "per_part_traj_mean": dict(sl.per_part_traj_mean),
            "per_part_traj_curve": {p: list(v) for p, v in
                                    sl.per_part_traj_curve.items()},
        })
        samples_diag.append(rec)
        if verbose:
            leg_pids = [p for p in sl.pose_tag if p != "seat"]
            n_lying = sum(
                1 for p in leg_pids
                if sl.pose_tag.get(p, "直立") == "躺/斜"
            )
            print(f"  #{idx:>2d}{tag}  score={sl.layout_score:.2f}  "
                  f"counts={sl.grasp_counts}  "
                  f"lying={n_lying}/{len(leg_pids)}")
    # 挂到 searcher 供 run_fast_search 写诊断 JSON
    searcher.all_samples = samples_diag

    elapsed = time.time() - t_start
    if not scored:
        if verbose:
            print(f"\n[Random] {n_samples} 个样本均不可行 ({elapsed:.1f}s)；"
                  f"建议放大 bounds / 换 seed / 增大 n_samples 重跑。")
            _summarize_search_failures(
                samples_diag,
                asm_steps=list(searcher.asm.steps),
                verbose=True,
            )
        return None

    scored.sort(key=lambda s: s.layout_score, reverse=True)
    best = scored[0]
    searcher.history.append(
        (0, best.layout_score,
         float(np.mean([s.layout_score for s in scored]))))

    if verbose:
        leg_pids = [p for p in best.pose_tag if p != "seat"]
        n_lying = sum(
            1 for p in leg_pids
            if best.pose_tag.get(p, "直立") == "躺/斜"
        )
        pose_brief = "/".join(
            f"{p}={best.pose_tag.get(p,'?')}" for p in leg_pids
        )
        print(f"\n[Random] L2 通过 {len(scored)}/{n_samples} ({elapsed:.1f}s)")
        print(f"  best_score   = {best.layout_score:.4f}  "
              f"(avg_manip_ep={best.manipulability_avg:.4f}, "
              f"avg_traj_min={best.traj_manip_min_avg:.4f}, "
              f"avg_dist={best.distance_cost:.3f}m, "
              f"lying={n_lying}/{len(leg_pids)})")
        print(f"  grasp_counts = {best.grasp_counts}")
        print(f"  arm_choice   = {best.arm_choice}")
        print(f"  pose_tag     = {pose_brief}")
        # per-part manipulability / distance / 轨迹下界
        if best.per_part_manip:
            manip_str = " | ".join(
                f"{p}={best.per_part_manip.get(p, 0.0):.4f}"
                for p in best.grasp_counts)
            print(f"  manip per part EP    = {manip_str}")
        if best.per_part_traj_min:
            traj_str = " | ".join(
                f"{p}={best.per_part_traj_min.get(p, 0.0):.4f}"
                for p in best.grasp_counts)
            print(f"  traj_min per part    = {traj_str}  "
                  f"(min along path, lower = closer to singularity)")
        if best.per_part_dist:
            dist_str = " | ".join(
                f"{p}={best.per_part_dist.get(p, 0.0):.3f}m"
                for p in best.grasp_counts)
            print(f"  distance per part    = {dist_str}")
        print(f"  z_offset     = {{{', '.join(f'{p}={z:.3f}' for p,z in best.z_offset.items())}}}")

    # random 模式也支持 L3：对 L2 通过的 top-k 做真实 RRT 复验（与 CEM 一致）
    if try_l3_top_k > 0 and searcher.enable_l3 and scored:
        scored.sort(key=lambda s: s.layout_score, reverse=True)
        if verbose:
            print(f"\n[Random] L3 复验 top-{min(try_l3_top_k, len(scored))} "
                  f"(TransportPrimitive + PickPlace/RRT)…")
        for sl in scored[:try_l3_top_k]:
            searcher.funnel["l3_attempts"] += 1
            t0 = time.time()
            ok = searcher.l3(sl)
            dt = time.time() - t0
            tag = "[OK] L3 通过" if ok else "[NO] L3 失败"
            if verbose:
                print(f"  | try_l3 score={sl.layout_score:.2f} {tag}  ({dt:.1f}s)")
                if not ok:
                    print(f"    reason: {sl.fail_reason}")
            if ok:
                searcher.funnel["l3_pass"] += 1
                return sl
        if verbose:
            print("[Random] top-k 均未通过 L3；回退为 L2 最高分 layout（"
                  "执行端可能仍失败，建议增大 n_samples 或换 seed）。")
        return scored[0]

    return best


def cem_search(searcher: FastLayoutSearcher, *, pop_size: int = 24, elite_n: int = 6, generations: int = 8,
               sigma_init: float = 0.08, sigma_decay: float = 0.7, sigma_min: float = 0.015,
               try_l3_top_k: int = 0, rng_seed: int = 0, l2_retry: int = 0, verbose: bool = True) -> Optional[
    _ScoredLayout]:
    rng = np.random.default_rng(rng_seed)
    pids = searcher.search_part_ids
    bounds = searcher.bounds
    seeds = searcher.task.staging_seeds

    mu = {pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])]) for pid in pids}
    sigma = {pid: np.array([sigma_init, sigma_init]) for pid in pids}

    candidates: List[_ScoredLayout] = []
    candidates.append(_ScoredLayout(xy={pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])]) for pid in pids}))
    n_anchor = max(2, pop_size // 2)
    n_small_noise = (n_anchor - 1) // 2
    for _ in range(n_small_noise): candidates.append(
        _ScoredLayout(xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init / 2)))
    for _ in range((n_anchor - 1) - n_small_noise): candidates.append(
        _ScoredLayout(xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init)))
    while len(candidates) < pop_size:
        candidates.append(
            _ScoredLayout(xy={pid: _clip_xy(rng.normal(mu[pid], sigma_init * 1.4), bounds[pid]) for pid in pids}))

    best_overall: Optional[_ScoredLayout] = None

    for gen in range(generations):
        scored: List[_ScoredLayout] = []
        for sl in candidates:
            searcher.funnel["sampled"] += 1
            if not searcher.l1(sl): continue
            searcher.funnel["l1_pass"] += 1
            if searcher.l2(sl, retry=l2_retry):
                searcher.funnel["l2_pass"] += 1
                scored.append(sl)

        scored.sort(key=lambda s: s.layout_score, reverse=True)

        if scored:
            best = scored[0]
            elite = scored[:elite_n]
            elite_mean = float(np.mean([e.layout_score for e in elite]))
            if (best_overall is None or best.layout_score > best_overall.layout_score):
                best_overall = best
        else:
            best, elite, elite_mean = None, [], -np.inf

        searcher.history.append((gen, best.layout_score if best else -np.inf, elite_mean))

        if verbose:
            n_l1 = sum(1 for c in candidates if c.l1_pass)
            if best is not None:
                # 计本代 best 的姿态摘要：n_lying / n_total（seat 不算腿）
                leg_pids = [p for p in best.pose_tag if p != "seat"]
                n_lying = sum(1 for p in leg_pids if best.pose_tag.get(p, "直立") == "躺/斜")
                pose_brief = "/".join(
                    f"{p}={best.pose_tag.get(p, '?')}" for p in leg_pids
                )
                best_msg = (
                    f"best_score={best.layout_score:.4f} "
                    f"(avg_manip={best.manipulability_avg:.4f}, "
                    f"avg_dist={best.distance_cost:.3f}m, "
                    f"lying={n_lying}/{len(leg_pids)}; {pose_brief})"
                )
            else:
                best_msg = "-"
            print(f"\n[CEM gen {gen}/{generations - 1}] sampled={pop_size} L1ok={n_l1} L2ok={len(scored)} "
                  f"elite_mean={elite_mean:.2f}  {best_msg}")

        if not scored:
            if verbose: print("  -> 本代无可行解，sigma 放大，半数样本全盒重采。")
            for pid in pids:
                sigma[pid] = np.minimum(sigma[pid] * 2.0, np.array([0.25, 0.25]))
                mu[pid] = np.array([float(seeds[pid][0]), float(seeds[pid][1])])
            candidates = [
                _ScoredLayout(xy={pid: np.array([float(seeds[pid][0]), float(seeds[pid][1])]) for pid in pids})]
            n_noise = (pop_size - 1) // 2
            for _ in range(n_noise): candidates.append(
                _ScoredLayout(xy=_seed_with_noise(seeds, pids, bounds, rng, sigma_init * 1.5)))
            for _ in range((pop_size - 1) - n_noise):
                xy = {pid: np.array([float(rng.uniform(bounds[pid][0][0], bounds[pid][0][1])),
                                     float(rng.uniform(bounds[pid][1][0], bounds[pid][1][1]))]) for pid in pids}
                candidates.append(_ScoredLayout(xy=xy))
            continue

        if try_l3_top_k > 0 and searcher.enable_l3:
            for sl in scored[:try_l3_top_k]:
                searcher.funnel["l3_attempts"] += 1
                t0 = time.time()
                ok = searcher.l3(sl)
                dt = time.time() - t0
                searcher.l3_marked_gens.append((gen, ok))
                tag = "[OK] L3 通过" if ok else "[NO] L3 失败"
                if verbose:
                    print(f"  | try_l3({sl.layout_score:.2f}) {tag}  ({dt:.1f}s)")
                    if not ok: print(f"    reason: {sl.fail_reason}")
                if ok:
                    searcher.funnel["l3_pass"] += 1
                    return sl

        for pid in pids:
            ps = np.array([e.xy[pid] for e in elite])
            mu[pid] = ps.mean(axis=0)
            sigma[pid] = np.maximum(ps.std(axis=0) * sigma_decay, sigma_min)

        candidates = [_ScoredLayout(xy=dict(best.xy))]
        for _ in range(pop_size - 1):
            xy = {pid: _clip_xy(rng.normal(mu[pid], sigma[pid]), bounds[pid]) for pid in pids}
            candidates.append(_ScoredLayout(xy=xy))

    return best_overall


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════
def run_fast_search(task: FastSearchTask, *,
                    mode: str = "random",
                    n_samples: int = 20,
                    pop_size: int = 24, elite_n: int = 6, generations: int = 8,
                    try_l3_top_k: int = 0, enable_l3: bool = False, sigma_init: float = 0.08,
                    sigma_decay: float = 0.7, rng_seed: int = 0, l2_retry: int = 0, ik_retry_n: int = 0,
                    verbose: bool = True) -> Tuple[Optional[str], Optional[_ScoredLayout]]:
    """统一搜索入口。

    Args:
        mode: ``"random"`` (默认) — 在 staging 范围里均匀采样 ``n_samples`` 个
            候选，每个走 L1+L2，选最高分；速度~10x 于 CEM，适合 5 件零件
            的小问题。``"cem"`` — Cross-Entropy Method 多代搜索，更彻底但
            慢；适合零件数较多 / bounds 较宽 / 需要 L3 RRT 校验的场景。
        n_samples: ``mode='random'`` 时的随机样本数 (含 seed anchor)。
        pop_size/elite_n/generations/sigma_*: ``mode='cem'`` 时的 CEM 超参。
    """
    if mode not in ("random", "cem"):
        raise ValueError(f"mode must be 'random' or 'cem', got {mode!r}")

    print("=" * 60)
    print(f"  Fast Optimal Layout Search - {task.name}")
    print(f"  Scoring (normalized ∈ [0,1]):"
          f" Grasp*{WEIGHT_GRASP} + Manip_EP*{WEIGHT_MANIP_EP}"
          f" + Manip_TRAJ*{WEIGHT_MANIP_TRAJ} + Dist*{WEIGHT_DIST}"
          f"  ({WEIGHT_GRASP + WEIGHT_MANIP_EP + WEIGHT_MANIP_TRAJ + WEIGHT_DIST:.2f} total)"
          f"  | N_TRAJ_WAYPOINTS={N_TRAJ_WAYPOINTS}")
    if mode == "random":
        print(f"  mode=random  n_samples={n_samples}  seed={rng_seed}")
    else:
        print(f"  mode=cem  pop={pop_size}  elite={elite_n}  gens={generations}  seed={rng_seed}")
    print("=" * 60)

    t0 = time.time()
    searcher = FastLayoutSearcher(task, enable_l3=enable_l3, ik_retry_n=ik_retry_n)
    if mode == "random":
        solution = random_search(
            searcher, n_samples=n_samples, rng_seed=rng_seed, verbose=verbose,
            try_l3_top_k=try_l3_top_k if searcher.enable_l3 else 0,
        )
        search_method = "random_optimal"
    else:
        solution = cem_search(
            searcher, pop_size=pop_size, elite_n=elite_n, generations=generations,
            sigma_init=sigma_init, sigma_decay=sigma_decay,
            try_l3_top_k=try_l3_top_k if searcher.enable_l3 else 0,
            rng_seed=rng_seed, l2_retry=l2_retry, verbose=verbose,
        )
        search_method = "cem_optimal"
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    if solution is not None and solution.l2_pass:
        print(f"[OK] 找到最佳布局，用时 {elapsed:.1f}s")
        print(f"  最终综合得分 score = {solution.layout_score:.4f}  ∈ [0, 1]")
        # 把当前最佳布局的"原始物理量"按零件列出来——方便人肉判断
        # 哪个零件 manip 偏低 / 哪个距离过长。
        if solution.per_part_manip:
            print(f"  per-part manipulability (Yoshikawa, 越大越好):")
            for pid in solution.grasp_counts:
                m = solution.per_part_manip.get(pid, 0.0)
                # 给个粗略的"健康度"标签，方便扫读
                if m >= NORM_MANIP_TARGET:
                    tag = "好"
                elif m >= NORM_MANIP_TARGET * 0.5:
                    tag = "中"
                else:
                    tag = "低(临近奇异)"
                print(f"     {pid:>8s}: manip={m:.4f}  [{tag}]")
        if solution.per_part_dist:
            print(f"  per-part transport distance (越短越好):")
            for pid in solution.grasp_counts:
                d = solution.per_part_dist.get(pid, 0.0)
                print(f"     {pid:>8s}: dist={d:.3f}m")
    else:
        print(f"[FAIL] 搜索失败，用时 {elapsed:.1f}s")
    print("=" * 60)

    out_path: Optional[str] = None
    if solution is not None and solution.l2_pass:
        out_dir = os.path.join(os.path.dirname(__file__), "_output")
        os.makedirs(out_dir, exist_ok=True)
        # 执行端固定读 ``{output_layout_name}.layout``；始终写入该主文件，
        # 避免 enable-l3 时只更新 *_l2only.layout 而主文件仍是旧 run 的残留。
        out_path = os.path.join(out_dir, f"{task.output_layout_name}.layout")
        # 把 L2 选定的 (chosen_rotmat, z_offset) 一并落到 staging_positions
        # 里：pos.z = z_offset、rotmat = 选定姿态。下游
        # ``dual_sequence_execution._validate_loaded_layout`` 的 rot_cands
        # 第一项 = ``(st.rotmat, 0.0)``，因此第一次试就能直接通过 reason。
        staging_positions = {}
        for pid in searcher.search_part_ids:
            if pid not in solution.xy: continue
            rot = solution.chosen_rotmat.get(pid, np.eye(3))
            z_off = solution.z_offset.get(pid, 0.0)
            staging_positions[pid] = (
                np.array([
                    float(solution.xy[pid][0]),
                    float(solution.xy[pid][1]),
                    float(z_off),
                ]),
                np.asarray(rot, dtype=float),
            )

        layout = WorkspaceLayout(
            robot_base_pos=task.robot_base_pos.copy(),
            robot_base_rotmat=task.robot_base_rotmat.copy(),
            assembly_station_pos=task.fixture_pos.copy(),
            assembly_station_rotmat=task.fixture_rotmat.copy(),
            staging_positions=staging_positions,
            name=task.output_layout_name,
            metadata={
                "search_method": search_method,
                "n_samples": int(n_samples) if mode == "random" else None,
                "l3_enabled": bool(searcher.enable_l3),
                "l3_pass": bool(solution.l3_pass),
                # 关键：标注本布局是为哪种机器人搜出来的。下游
                # ``dual_sequence_execution.py`` 加载时会校验该字段，
                # 不匹配则丢弃缓存重新走 DFS / 重搜。
                "robot_type": "panthera_ht",
                "arm_y_offset": float(_DUAL_ARM_Y_OFFSET),
                "weights": {
                    "grasp": float(WEIGHT_GRASP),
                    "manip": float(WEIGHT_MANIP_EP),       # 端点（向后兼容字段名）
                    "manip_ep": float(WEIGHT_MANIP_EP),
                    "manip_traj": float(WEIGHT_MANIP_TRAJ),
                    "dist": float(WEIGHT_DIST),
                },
                "norm_targets": {
                    "grasp_min": float(NORM_GRASP_MIN_TARGET),
                    "grasp_mean": float(NORM_GRASP_MEAN_TARGET),
                    "grasp_hill_k": float(NORM_GRASP_HILL_K),
                    "grasp_min_weight": float(NORM_GRASP_MIN_WEIGHT),
                    "manip": float(NORM_MANIP_TARGET),
                    "manip_traj": float(NORM_TRAJ_MANIP_TARGET),
                    "dist_decay": float(NORM_DIST_DECAY),
                    "n_traj_waypoints": int(N_TRAJ_WAYPOINTS),
                },
                "score_components": {
                    "per_part_manip": dict(solution.per_part_manip),
                    "per_part_dist": dict(solution.per_part_dist),
                    "per_part_traj_min": dict(solution.per_part_traj_min),
                    "per_part_traj_mean": dict(solution.per_part_traj_mean),
                    "per_part_traj_curve": {p: list(v) for p, v in
                                            solution.per_part_traj_curve.items()},
                    "avg_manip": float(solution.manipulability_avg),
                    "avg_dist": float(solution.distance_cost),
                    "traj_manip_min_avg": float(solution.traj_manip_min_avg),
                },
                "filters": {
                    "min_same_arm_dist": float(MIN_SAME_ARM_DIST),
                },
                "arm_choice": dict(solution.arm_choice),
                "grasp_counts": dict(solution.grasp_counts),
                "pose_tags": dict(solution.pose_tag),
                "z_offsets": {pid: float(z) for pid, z in solution.z_offset.items()},
                "manipulability_avg": float(solution.manipulability_avg),
                "distance_cost": float(solution.distance_cost),
                "total_score": float(solution.layout_score),
            },
        )
        layout.save(out_path)
        print(f"\n[OK] 保存 optimal layout -> {os.path.relpath(out_path)}")

        # ── 同时写一份诊断 JSON，供 diagnostics_plot.py 出论文图用 ──
        # 包含：本次搜索常量、漏斗计数、每个样本完整记录（含 fail 原因
        # 和 per-component 归一化分量）、bounds、最优解 xy/rotmat 等。
        try:
            import json
            diag_path = os.path.splitext(out_path)[0] + "_diagnostics.json"
            diag_payload = {
                "task_name": task.name,
                "search_method": search_method,
                "n_samples": int(n_samples) if mode == "random" else None,
                "robot_type": "panthera_ht",
                "arm_y_offset": float(_DUAL_ARM_Y_OFFSET),
                "weights": {"grasp": float(WEIGHT_GRASP),
                            "manip": float(WEIGHT_MANIP_EP),
                            "manip_ep": float(WEIGHT_MANIP_EP),
                            "manip_traj": float(WEIGHT_MANIP_TRAJ),
                            "dist": float(WEIGHT_DIST)},
                "norm_targets": {
                    "grasp_min": float(NORM_GRASP_MIN_TARGET),
                    "grasp_mean": float(NORM_GRASP_MEAN_TARGET),
                    "grasp_hill_k": float(NORM_GRASP_HILL_K),
                    "grasp_min_weight": float(NORM_GRASP_MIN_WEIGHT),
                    "manip": float(NORM_MANIP_TARGET),
                    "manip_traj": float(NORM_TRAJ_MANIP_TARGET),
                    "dist_decay": float(NORM_DIST_DECAY),
                    "n_traj_waypoints": int(N_TRAJ_WAYPOINTS),
                },
                "filters": {
                    "min_same_arm_dist": float(MIN_SAME_ARM_DIST),
                },
                "funnel": dict(searcher.funnel),
                "bounds": {
                    pid: [list(searcher.bounds[pid][0]),
                          list(searcher.bounds[pid][1])]
                    for pid in searcher.search_part_ids
                },
                "robot_base_pos": list(map(float, task.robot_base_pos)),
                "fixture_pos": list(map(float, task.fixture_pos)),
                "search_part_ids": list(searcher.search_part_ids),
                "world_poses": {
                    pid: {"pos": list(map(float, p)),
                          "rotmat": [list(map(float, r)) for r in R]}
                    for pid, (p, R) in searcher.world_poses.items()
                },
                "best": {
                    "xy": {pid: [float(solution.xy[pid][0]),
                                 float(solution.xy[pid][1])]
                           for pid in solution.xy},
                    "score": float(solution.layout_score),
                    "grasp_counts": dict(solution.grasp_counts),
                    "arm_choice": dict(solution.arm_choice),
                    "pose_tag": dict(solution.pose_tag),
                    "z_offset": {pid: float(z)
                                 for pid, z in solution.z_offset.items()},
                    "chosen_rotmat": {
                        pid: [list(map(float, row)) for row in rot]
                        for pid, rot in solution.chosen_rotmat.items()},
                    "per_part_manip": dict(solution.per_part_manip),
                    "per_part_dist": dict(solution.per_part_dist),
                    "avg_manip": float(solution.manipulability_avg),
                    "avg_dist": float(solution.distance_cost),
                    "grasp_score_norm": float(solution.grasp_score_norm),
                    "manip_score_norm": float(solution.manip_score_norm),
                    "traj_manip_score_norm": float(solution.traj_manip_score_norm),
                    "dist_score_norm": float(solution.dist_score_norm),
                    "traj_manip_min_avg": float(solution.traj_manip_min_avg),
                    "per_part_traj_min": dict(solution.per_part_traj_min),
                    "per_part_traj_mean": dict(solution.per_part_traj_mean),
                    "per_part_traj_curve": {p: list(v) for p, v in
                                            solution.per_part_traj_curve.items()},
                    "n_traj_waypoints": int(N_TRAJ_WAYPOINTS),
                },
                "all_samples": list(getattr(searcher, "all_samples", [])),
            }
            with open(diag_path, "w", encoding="utf-8") as fp:
                json.dump(diag_payload, fp, ensure_ascii=False, indent=2)
            print(f"[OK] 保存搜索诊断 -> {os.path.relpath(diag_path)}")
            print(f"     可用 `python -m sealp.examples.layout.diagnostics_plot "
                  f"--layout {os.path.relpath(out_path)}` 出图。")
        except Exception as _diag_err:
            print(f"[WARN] 写诊断 JSON 失败: {_diag_err!r}")

    return out_path, solution


def main():
    parser = argparse.ArgumentParser(description="Fast Optimal Layout Search")
    parser.add_argument(
        "--mode", choices=["random", "cem"], default="random",
        help="random=随机采样 N 个候选取最高分(默认,快); cem=多代演化(慢,更彻底)"
    )
    parser.add_argument(
        "--n-samples", "--n", dest="n_samples", type=int, default=20,
        help="[random] 随机采样数 (默认 20，含 seed anchor)"
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子；同种子可复现")
    parser.add_argument("--pop", type=int, default=24, help="[cem] 每代候选数")
    parser.add_argument("--gens", type=int, default=8, help="[cem] 最大代数")
    parser.add_argument(
        "--enable-l3", action="store_true",
        help="开启 L3 RRT 真实校验 (random 模式下默认关闭，cem 模式下传该 flag 才开)"
    )
    args = parser.parse_args()

    task = YUANCHAIR_FAST_TASK
    # 默认 random 模式不跑 L3；L3 比较贵且 L2 末尾已加 pairwise 复检，
    # 对 5 件桌椅来说 random + L2 通过率足以直接落盘下游执行。
    run_fast_search(
        task,
        mode=args.mode,
        n_samples=args.n_samples,
        pop_size=args.pop,
        generations=args.gens,
        rng_seed=args.seed,
        enable_l3=args.enable_l3,
        try_l3_top_k=3 if args.enable_l3 else 0,
    )

if __name__ == "__main__":
    main()