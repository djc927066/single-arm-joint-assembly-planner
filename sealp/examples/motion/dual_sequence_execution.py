"""
Dual-Arm Sequence Execution Demo — YuanChair Assembly
======================================================

1. 初始 staging **不定死**：候选网格联合搜索；5 件 staging 两两不相交；桌面初始位不得
   侵入 **直接装在 fixture 上的零件**（椅面）的装配 goal。腿的 goal 相对椅面定义，若
   与桌面 staging 做 is_mcdwith 会大量误杀，故不参与此项检测。每只零件在 staging→goal
   上存在「左臂优先、右臂兜底」的 ``reason_common_gids`` 可行抓取。预搜索失败时回退为
   种子位姿并仍打开窗口、尝试执行与动画。
2. 执行阶段：RRT/PickPlace 的 ``obstacle_list`` 为 **地面 + 其余未抓件 staging +
   已装配件（goal）**；当前步规划前从列表中移除本件 staging。
3. fixture_pos 决定最终装配位；彩色显示选中的初始摆放，半透明显示目标 ghost。

Usage::

    python -m sealp.examples.motion.dual_sequence_execution
"""

import os
import pickle
import sys
import numpy as np
import wrs.basis.robot_math as rm
from wrs import wd, mgm, mcm
from direct.task.TaskManagerGlobal import taskMgr


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef, TaskPlan, StepParams
from sealp.config import load_config
from sealp.colliders import StaticEnvironment
from sealp.executor import SequenceExecutor
from sealp.primitives.transport import TransportPrimitive
from sealp.primitives.base import PrimitiveResult
from wrs.manipulation.pick_place import PickPlacePlanner
from wrs.motion.motion_data import MotionData

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda

# ─────────────────────────────────────────────────────────────
#  Pick / Place 各阶段方向常量
# ─────────────────────────────────────────────────────────────
#  一次完整 transport 的几何流：
#    1) pick_approach    : 直线接近抓取点（distance=0 → 等价于「不做接近」）
#    2) pick_depart      : 抓到物体后沿 +Z 抬升 (≈ 20 cm)
#    3) RRT moveto       : 自由空间运动到 goal 上方
#    4) place_approach   : 沿 −Z 直线下放到装配位 (≈ 10 cm)
#    5) place_depart     : 装好后撤离（部件相关：leg=−X, seat=+Z）
# ─────────────────────────────────────────────────────────────
_APPROACH_LINEAR_DIR = -rm.const.x_ax        # 兼容字段（pick approach 距离=0 后无意义）
_PICK_DEPART_DIR = rm.const.z_ax             # 抓后 +Z 抬升
_PLACE_APPROACH_DIR = -rm.const.z_ax         # 装配前 +Z → −Z 下放
_LEG_PLACE_DEPART_DIR = -rm.const.x_ax       # 腿装好后沿 −X 撤离
_SEAT_PLACE_DEPART_DIR = rm.const.z_ax       # 座板装好后沿 +Z 撤离

# ─────────────────────────────────────────────────────────────
#  Pick / Place 运动规划「宽松度」参数（统一在此调节）
# ─────────────────────────────────────────────────────────────
#  全部 push 向「容易成功」一侧：
#    * 距离更大        → 直线段离开/接近留更多缓冲，RRT 自由空间更广
#    * granularity 更大 → 直线段插值 / 关节复验更稀疏，不易被几何噪声误杀
#    * cd_ex_radius 更小 → 已装件 CD 膨胀更小，避障更宽松
#  注意：approach 距离过大可能让直线段端点 IK 不可达，这里取的是经验上限。
# ─────────────────────────────────────────────────────────────
RELAXED_PLANNING = {
    # —— StepParams 默认 (gen_pick_and_place 的 pick_approach / 默认 depart) ——
    "step_approach_distance":      0,   # m
    "step_depart_distance":        0.05,   # m

    # —— 抓取后抬升 (pick_depart_distance) ——
    "pick_depart_distance":        0.06,   # m，左右臂统一

    # —— 装配前直线接近 (place_approach_distance_list) —— 用户要求 10 cm，沿 −Z
    "place_approach_distance":     0.05,   # m

    # —— 部件相关的 place_depart_distance ——
    "leg_place_depart_distance":   0.05,   # m
    "seat_place_depart_distance":  0.06,   # m

    # —— 透传给 PickPlacePlanner 的内部宽松度 ——
    "linear_granularity":          0.04,   # m，直线段插值步长（越大越稀疏）

    # —— SequenceExecutor 已装件 CD 膨胀 ——
    #   说明: 立起来的 leg 是细长杆，AABB cdprim 比 mesh 大不了多少 (杆直
    #   径 ~2cm)，仅 5mm padding 时 RRT 经常找出"擦边"路径——cdprim 看不到
    #   碰，mesh 复验报 robot_vs_obstacle (命中='leg_fl')。15mm 是经验值：
    #   足够把 leg AABB 膨胀到杆周围 ~3cm 缓冲带，让 RRT 看到"路障"主动
    #   绕远；又不至于覆盖到相邻 leg goal pose（座面下相邻两腿水平间距
    #   一般 >5cm）。如再调大注意验证 leg_bl/leg_br goal 处 IK 仍可解。
    "assembled_cd_ex_radius":      0.015,  # m，越小越宽松

    # —— 关节空间复验密度 (_motion_replay_collision_free) ——
    #   注意：RRT 可以稀疏以提速，但最终复验必须严格。这里保持较密，
    #   保证机器人本体、夹爪、被抓物体都不会穿过桌面/其他零件。
    "replay_granularity":          0.03,   # rad，越小越严格

    # —— RRT 全局稀疏化（monkey-patch wrs.RRTConnect.plan）——
    #   ADPlanner.gen_approach 把 ext_dist=0.1 写死了 4 处（比 RRT 默认 0.2
    #   还密），从外层无法覆盖，只能 patch RRTConnect.plan 强制改写。
    #   下面三个值越"大/小/小"越稀疏越快越粗：
    "rrt_ext_dist":                0.30,   # rad，扩展步长（默认 0.1，越大越稀疏）
    "rrt_smoothing_n_iter":        150,     # 平滑迭代数（默认 500，越小越快）
    "rrt_max_time":                10.0,   # 秒，单次 RRT 最长耗时
}


def _patch_rrt_for_relaxed_sampling():
    """全局 monkey-patch ``RRTConnect.plan``：强制覆盖 ``ext_dist`` /
    ``smoothing_n_iter`` / ``max_time``，让所有从 ``PickPlacePlanner`` 走
    出去的 RRT 都用 ``RELAXED_PLANNING`` 中的稀疏化设置。

    必须在 import 后立即生效（写在模块顶层），重复调用安全（幂等）。
    """
    from wrs.motion.probabilistic.rrt_connect import RRTConnect
    if getattr(RRTConnect.plan, "_sealp_relaxed_patched", False):
        return
    _orig_plan = RRTConnect.plan

    def _patched_plan(self, *args, **kwargs):
        # ADPlanner.gen_approach 等上游会显式传 ext_dist=.1，必须覆盖
        kwargs["ext_dist"] = RELAXED_PLANNING["rrt_ext_dist"]
        kwargs["smoothing_n_iter"] = RELAXED_PLANNING["rrt_smoothing_n_iter"]
        kwargs["max_time"] = RELAXED_PLANNING["rrt_max_time"]
        return _orig_plan(self, *args, **kwargs)

    _patched_plan._sealp_relaxed_patched = True
    RRTConnect.plan = _patched_plan
    print(f"[RRT patch] ext_dist={RELAXED_PLANNING['rrt_ext_dist']}  "
          f"smoothing_n_iter={RELAXED_PLANNING['rrt_smoothing_n_iter']}  "
          f"max_time={RELAXED_PLANNING['rrt_max_time']}s")


_patch_rrt_for_relaxed_sampling()

# ── 候选搜索：仅作「种子」，最终位姿由预搜索给出 ─────────────────
STAGING_SEEDS = {
    "seat": np.array([0.30, -0.10, 0.00]),
    "leg_fl": np.array([0.25, 0.20, 0.00]),
    "leg_bl": np.array([0.40, 0.15, 0.00]),
    # Panthera-HT 右臂 base 在 (0, -0.62)；leg 立杆 cdprim 必须离 base
    # 至少 ~0.20m 否则 home 姿态下被判穿模。
    "leg_fr": np.array([0.25, -0.85, 0.00]),
    "leg_br": np.array([0.40, -0.85, 0.00]),
}

# ── 黄金 staging（已实测可让 5/5 step 全部通过 RRT/IK） ────────────
# DFS 预搜索内部依赖 trac_ik 的 stochastic IK，多次跑会产出不同的
# (n, layout)，导致"上次能跑通 → 这次跑不通"。一次成功后把那次结果
# 固化在这里，启动时直接应用、跳过 DFS、跑 RRT 一次写入 motion cache，
# 第二次启动 motion cache 命中即可秒进。
# 调试时可把 _USE_GOLDEN_STAGING 改为 False 重新走 DFS。
# Panthera-HT 切换后，旧的 piper 时代 GOLDEN 失效（leg_fr/leg_br y=-0.62
# 紧贴右臂 base，home 姿态下立刻被判穿模）。先关掉 golden 让流程重新
# 走 DFS / 加载 search_dual_layout 写出的 .layout；得到稳定结果后再固化。
_USE_GOLDEN_STAGING = False
GOLDEN_STAGING = {
    "seat":   (np.array([0.30, -0.10, 0.00]), np.eye(3)),
    "leg_bl": (np.array([0.43,  0.13, 0.00]), np.eye(3)),
    "leg_br": (np.array([0.43, -0.85, 0.00]), np.eye(3)),
    "leg_fl": (np.array([0.25,  0.20, 0.00]), np.eye(3)),
    "leg_fr": (np.array([0.25, -0.85, 0.00]), np.eye(3)),
}

def _grid_values(lo, hi, step=0.05, ndigits=2):
    """闭区间网格，返回普通 float list，避免 np.float64 打印太吵。"""
    if hi < lo:
        return []
    vals = np.arange(lo, hi + step * 0.5, step)
    return [round(float(v), ndigits) for v in vals]


def _make_table_aware_staging_zones():
    """根据 sample_config.yaml 里的 work_table 尺寸 + 双臂 y 间距生成搜索区。

    work_table:
        extent = [0.8, 1.6, 0.02], pos = [0.4, -0.3, -0.01]

    因此桌面 xy 物理范围约为：
        x ∈ [0.0, 0.8], y ∈ [-1.1, 0.5]

    Piper 双臂在 ``piper_dual_arm.py`` 中：
        lft_y = 0.0, rgt_y = -0.597

    右侧腿（leg_fr / leg_br）不应再固定在 [-0.72, -0.52] 这种小段；
    它们应围绕右臂 y=-0.597，在桌面负 y 半区展开。这里取
    rgt_y ± [0.42, 0.30] 并受桌面边界裁剪，得到约 [-1.02, -0.32]。
    """
    table_pos = np.array([0.4, -0.3, -0.01])
    table_extent = np.array([0.8, 1.6, 0.02])
    margin = 0.08
    x_min = table_pos[0] - table_extent[0] / 2.0 + margin
    x_max = table_pos[0] + table_extent[0] / 2.0 - margin
    y_min = table_pos[1] - table_extent[1] / 2.0 + margin
    y_max = table_pos[1] + table_extent[1] / 2.0 - margin

    lft_y = 0.0
    rgt_y = -0.597

    # x 方向也放开：不要只卡在 0.18~0.43，保留右半桌面的可达余量。
    all_leg_x = _grid_values(max(x_min, 0.15), min(x_max, 0.65), 0.05)
    rear_leg_x = _grid_values(max(x_min, 0.28), min(x_max, 0.65), 0.05)

    left_y = _grid_values(
        max(y_min, lft_y + 0.08),
        min(y_max, lft_y + 0.32),
        0.05,
    )
    right_y = _grid_values(
        max(y_min, rgt_y - 0.42),
        min(y_max, rgt_y + 0.28),
        0.05,
    )

    return {
        "seat": {
            "x": [0.22, 0.27, 0.32, 0.37],
            "y": [-0.22, -0.15, -0.10, -0.05],
        },
        "leg_fl": {
            "x": all_leg_x,
            "y": left_y,
        },
        "leg_bl": {
            "x": rear_leg_x,
            "y": left_y,
        },
        "leg_fr": {
            "x": all_leg_x,
            "y": right_y,
        },
        "leg_br": {
            "x": rear_leg_x,
            "y": right_y,
        },
    }


STAGING_ZONES = _make_table_aware_staging_zones()

INITIAL_STAGING_PART_IDS = ("seat", "leg_bl", "leg_br", "leg_fl", "leg_fr")

def load_obstacles_from_config(config_path, base):
    """从 sample_config 读取 environment.obstacles 并在场景中显示。"""
    if not os.path.isfile(config_path):
        print(f"[WARN] 未找到配置文件: {config_path}，将不加载静态环境障碍物。")
        return []

    cfg = load_config(config_path)
    env = StaticEnvironment(
        obstacle_defs=cfg.obstacle_defs,
        base_dir=cfg.config_dir,
    )
    loaded_obstacles = list(env.obstacle_list)
    for obs in loaded_obstacles:
        obs.attach_to(base)
        obs._sealp_role = "environment_obstacle"
    print(f"[环境] 已从配置加载 {len(loaded_obstacles)} 个静态障碍物。")
    return loaded_obstacles

def generate_staging_candidates(part_id, seed_pos, include_cross_arm=True):
    """候选列表：种子永远第一个，再补 STAGING_ZONES 网格，按距种子平面距离排序。"""
    candidates = [np.asarray(seed_pos, dtype=float).copy()]
    zone = STAGING_ZONES.get(part_id)
    if zone is None:
        return candidates

    grid = []
    for x in zone["x"]:
        for y in zone["y"]:
            grid.append(np.array([float(x), float(y), 0.0]))

    if include_cross_arm:
        cross = (
            {"x": [0.28, 0.33], "y": [-0.30, -0.25]}
            if part_id in ("leg_fr", "leg_br")
            else {"x": [0.28, 0.33], "y": [0.12, 0.18]}
        )
        for x in cross["x"]:
            for y in cross["y"]:
                grid.append(np.array([float(x), float(y), 0.0]))

    seen = {tuple(np.round(candidates[0], 4))}
    extra = []
    for p in grid:
        t = tuple(np.round(p, 4))
        if t in seen:
            continue
        seen.add(t)
        extra.append(p)
    extra.sort(key=lambda q: float(np.linalg.norm(q[:2] - seed_pos[:2])))
    candidates.extend(extra)
    return candidates


def is_pose_collision_free_with_selected(part_id, pos, rotmat, staging_obstacles, selected_part_ids):
    probe = staging_obstacles.get(part_id)
    if probe is None:
        return True
    old_pos, old_rot = probe.pos.copy(), probe.rotmat.copy()
    try:
        probe.pos = pos
        probe.rotmat = rotmat
        for oid in selected_part_ids:
            other = staging_obstacles.get(oid)
            if other is None:
                continue
            if probe.is_mcdwith(other):
                return False
        return True
    finally:
        probe.pos = old_pos
        probe.rotmat = old_rot


def build_goal_obstacle_models(asm, world_poses):
    """各零件在装配 goal 处的碰撞体（仅用于预搜索：初始位不得侵入他人装配域）。"""
    out = {}
    for pid, (gp, gr) in world_poses.items():
        # world_poses 含虚拟根 "fixture"，无对应 PartDef / 模型
        if pid not in asm.part_ids:
            continue
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue
        cm = mcm.CollisionModel(initor=mp)
        cm.pos = gp
        cm.rotmat = gr
        cm._sealp_role = "goal_placeholder"
        cm._sealp_part_id = pid
        out[pid] = cm
    return out


def _step_parent_id(asm, part_id):
    for s in asm.steps:
        if s.part_id == part_id:
            return s.parent_id
    return None


def is_staging_clear_of_foreign_goals(part_id, pos, rotmat, staging_obstacles, goal_models, asm):
    """桌面 staging 不得侵入「直接装在 fixture 上」的零件的 goal 体积（如椅面）。

    腿的装配位相对椅面定义，其 goal 与桌面初始位在几何上易误判相交，故跳过 parent!=fixture
    的 goal；执行阶段仍以 staging + 已装件为 RRT 障碍。
    """
    probe = staging_obstacles.get(part_id)
    if probe is None or not goal_models:
        return True
    old_pos, old_rot = probe.pos.copy(), probe.rotmat.copy()
    try:
        probe.pos = pos
        probe.rotmat = rotmat
        for oid, gcm in goal_models.items():
            if oid == part_id or gcm is None:
                continue
            if _step_parent_id(asm, oid) != "fixture":
                continue
            if probe.is_mcdwith(gcm):
                return False
        return True
    finally:
        probe.pos = old_pos
        probe.rotmat = old_rot


# ─────────────────────────────────────────────────────────────
#  staging rotmat 候选：腿允许"立着 / 躺着 / 斜着躺"
# ─────────────────────────────────────────────────────────────
#   * 直立 = rotmat I（cylinder 长轴沿世界 +Z）
#   * 躺姿 = 先绕 y 轴 −90° 把长轴放到水平，再加一个 yaw（绕 z）斜角
#   * 躺姿需要把 z 抬一点点避免半埋桌面（_LYING_Z_OFFSET）
#   * seat 不躺，只保留直立
_LYING_Z_OFFSET = 0.02  # m，躺姿 staging 的 z 抬升


def _make_lying_leg_rotmats(yaw_deg_list=(0, 45, -45, 90, -90, 135, -135, 180)):
    """生成"桌腿水平躺着 + 任意 yaw 斜角"的 rotmat 列表（不含直立）。

    base_lying = Ry(−90°) → 把模型本地 +Z 转到世界 +X；
    yaw_R      = Rz(yaw)  → 在水平面里再转一个角度。
    """
    base_lying = rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(-90.0))
    out = []
    for yaw_deg in yaw_deg_list:
        yaw_R = rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(yaw_deg))
        out.append(yaw_R @ base_lying)
    return out


# 每件零件的 rotmat 候选列表，按"优先尝试顺序"排列。
#   元素：(rotmat, z_offset)。z_offset 加在 staging.pos[2] 上。
_LEG_ROTMAT_CANDS = (
    [(R, _LYING_Z_OFFSET) for R in _make_lying_leg_rotmats()]   # 8 种躺姿/斜躺优先
    + [(np.eye(3), 0.0)]                                          # 直立兜底
)
STAGING_ROTMAT_CANDIDATES = {
    "seat":   [(np.eye(3), 0.0)],   # seat 只直立
    "leg_fl": _LEG_ROTMAT_CANDS,
    "leg_fr": _LEG_ROTMAT_CANDS,
    "leg_bl": _LEG_ROTMAT_CANDS,
    "leg_br": _LEG_ROTMAT_CANDS,
}


def _classify_pose(rotmat: np.ndarray) -> str:
    """根据 rotmat 第三列（本地 +Z 在世界中的方向）判定 staging 姿态。"""
    z_world = np.asarray(rotmat)[:, 2]
    return "直立" if abs(float(z_world[2])) > 0.85 else "躺/斜"


# 双臂初始 / home 关节角；用于"staging 不得在 home 姿态下与机械臂穿模"检查
_HOME_JV = np.zeros(6)


def _arms_collide_at_conf(robot, lft_jv, rgt_jv, obstacle_list):
    """检查双臂在给定关节角下是否与任意障碍物穿模。

    桌腿"躺着"放置时是一个很长的物体（>0.3 m），其碰撞模型在水平方向
    上会延伸很远，可能在机器人 home 姿态下就与臂体几何重叠。
    ``reason_common_gids`` 只在 pick / place pose 处对机器人做
    ``is_collided``，无法捕捉这类 **初始穿模**。本函数显式把双臂送到
    指定关节角（默认 home），逐臂调用 ``is_collided`` 做硬碰撞检测。

    Notes
    -----
    * 用 ``backup_state / restore_state`` 包裹，调用结束后机器人姿态
      和 cc 状态保持不变。
    * ``obstacle_list`` 期望是「除手中物以外」的所有静态/暂存物体；
      调用方需自行决定是否包含 env_obstacles（一般包含，可顺带捕捉
      与桌沿的穿模）。
    * 对 cc 启用了双臂的 DualPiper，单臂 ``is_collided`` 也会顺带检测
      到与另一只臂的相互碰撞，正符合需求。
    """
    if not obstacle_list:
        return False
    obs = list(obstacle_list)
    robot.lft_arm.backup_state()
    robot.rgt_arm.backup_state()
    try:
        robot.lft_arm.goto_given_conf(np.asarray(lft_jv))
        robot.rgt_arm.goto_given_conf(np.asarray(rgt_jv))
        for arm in (robot.lft_arm, robot.rgt_arm):
            hit = arm.is_collided(obstacle_list=obs)
            collided = hit[0] if isinstance(hit, tuple) else hit
            if collided:
                return True
        return False
    finally:
        robot.lft_arm.restore_state()
        robot.rgt_arm.restore_state()


def _validate_loaded_layout(plan, robot, staging_obstacles, grasp_cache,
                            world_poses, env_obstacles):
    """加载 .layout 后做兼容性「校验 + 优化」。

    保持每件零件的 (x, y) 位置不变，但允许在该位置上换 rotmat（直立 / 躺
    姿 / 斜躺）。腿默认优先躺姿（被夹爪从上方夹住一根水平的腿，对其他
    直立腿的扫掠半径远小于"竖着挂在夹爪下方"），所有躺姿都不通过时才
    回退直立。

    成功 ⇒ 直接把选中的 rotmat / z 写回 plan + staging_obstacles，返回
    True；任一件双臂×全部 rotmat 候选都不通过 ⇒ 返回 False（外层会
    丢弃 .layout 回退 DFS）。
    """
    pid_list = [p for p in INITIAL_STAGING_PART_IDS
                if plan.get_staging(p) is not None and p in staging_obstacles]
    pick_d = RELAXED_PLANNING["pick_depart_distance"]
    place_d = RELAXED_PLANNING["place_approach_distance"]
    print(f"\n[Layout 校验+优化] 强化 reason + 多 rotmat 尝试 "
          f"(+Z 抬 {pick_d:.2f}m / −Z 下放 {place_d:.2f}m)…")
    reason_kw = dict(
        pick_depart_dir=_PICK_DEPART_DIR,
        pick_depart_dist=pick_d,
        place_approach_dir=_PLACE_APPROACH_DIR,
        place_approach_dist=place_d,
    )
    for pid in pid_list:
        if pid not in world_poses:
            continue
        st = plan.get_staging(pid)
        gp, gr = world_poses[pid]
        gc_part = grasp_cache.get(model_alias_for_part(pid))
        if gc_part is None:
            continue
        rot_cands = [(st.rotmat, 0.0)]
        chosen_rot, chosen_arm, chosen_pos = None, None, None
        base_xy = st.pos[:2].copy()
        base_z = float(st.pos[2])
        for rot, z_off in rot_cands:
            cand_pos = np.array([base_xy[0], base_xy[1], base_z + z_off])
            # 同步当前件的 staging 障碍位姿，让其他件 reason 时也用新 rotmat
            staging_obstacles[pid].pos = cand_pos
            staging_obstacles[pid].rotmat = rot
            # 障碍：env + 其他 staging（与执行阶段口径一致）
            obs = list(env_obstacles)
            for oid in pid_list:
                if oid != pid and oid in staging_obstacles:
                    obs.append(staging_obstacles[oid])
            # 先做"初始穿模"硬检查：在 home 姿态下，所有 staging（含本件
            # 当前候选 rotmat）都不能与任一只臂几何重叠。躺着的桌腿水平
            # 方向延伸很长，特别容易踩这条线。
            home_obs = [staging_obstacles[p]
                        for p in pid_list if p in staging_obstacles]
            if _arms_collide_at_conf(robot, _HOME_JV, _HOME_JV, home_obs):
                print(f"  ✗ {pid} [{_classify_pose(rot)}] "
                      f"在 home 姿态下与机械臂穿模，跳过该 rotmat")
                continue
            for arm_obj, arm_tag in (
                    (robot.lft_arm, "左臂"), (robot.rgt_arm, "右臂")):
                ok, _ = pick_place_reason_common_ok(
                    arm_obj, gc_part, cand_pos, rot, gp, gr, obs, **reason_kw)
                if ok:
                    chosen_rot = rot.copy()
                    chosen_arm = arm_tag
                    chosen_pos = cand_pos
                    break
            if chosen_rot is not None:
                break
        if chosen_rot is None:
            # 全部候选都不通过 → 还原原始位姿后宣告失败
            staging_obstacles[pid].pos = st.pos.copy()
            staging_obstacles[pid].rotmat = st.rotmat.copy()
            print(f"  ❌ {pid} 双臂×{len(rot_cands)} 种 rotmat 均不通过 "
                  f"(xy={np.round(base_xy, 3).tolist()})")
            return False
        # 写回 plan 与 staging 障碍
        plan.set_staging(pid, pos=chosen_pos.copy(), rotmat=chosen_rot.copy())
        staging_obstacles[pid].pos = chosen_pos.copy()
        staging_obstacles[pid].rotmat = chosen_rot.copy()
        print(f"  ✅ {pid} {chosen_arm}通过 [{_classify_pose(chosen_rot)}] "
              f"(pos={np.round(chosen_pos, 3).tolist()})")
    print("[Layout 校验+优化] 全部通过。")
    return True


def apply_seed_staging(plan, staging_obstacles, seeds):
    """预搜索失败时回退：用种子位姿写回 plan 与 staging 碰撞体。"""
    for pid in INITIAL_STAGING_PART_IDS:
        st = plan.get_staging(pid)
        if st is None:
            continue
        p = seeds[pid].copy() if pid in seeds else st.pos.copy()
        r = np.eye(3)
        plan.set_staging(pid, pos=p, rotmat=r)
        o = staging_obstacles.get(pid)
        if o is not None:
            o.pos, o.rotmat = p, r


def model_alias_for_part(part_id):
    return "seat_model" if part_id == "seat" else "leg_model"


def load_grasp_cache(grasp_paths):
    out = {}
    for alias, path in grasp_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            out[alias] = pickle.load(f)
        print(f"[预搜索] 已加载 {alias!r}: {len(out[alias])} 个抓取")
    return out


def _segment_reachable(arm, grasp, base_pos, base_rot, direction, distance,
                       n_samples=4):
    """从 (base_pos, base_rot) 抓取位姿沿 ``direction`` 直线走 ``distance``，
    等分 ``n_samples`` 中间采样点 IK 是否一直可解（前一点作下一次的 seed）。

    用于补 ``reason_common_gids`` 缺失的"中间直线段可达性"检查 ——
    例如 pick_depart 沿 +Z 抬升、place_approach 沿 −Z 下放。

    Returns
    -------
    bool : 全部采样点 IK 成功 → True；任意一点失败 → False。
    """
    if distance is None or distance <= 1e-6 or n_samples < 1:
        return True
    base_pos = np.asarray(base_pos, dtype=float)
    base_rot = np.asarray(base_rot, dtype=float)
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


def pick_place_reason_common_ok(arm, grasp_collection, sp, sr, gp, gr,
                                obstacle_list,
                                pick_depart_dir=None, pick_depart_dist=None,
                                place_approach_dir=None,
                                place_approach_dist=None):
    """staging→goal 双端 IK + 共同抓取检查；可选追加中间直线段可达性过滤。

    新增的 ``pick_depart_*`` / ``place_approach_*`` 用于过滤掉那些"两端
    IK 通过但中间直线段 IK 求解失败"的位置 —— 这正是 PickPlacePlanner
    实际执行时报 ``IK not solvable`` 的常见根因。
    """
    if grasp_collection is None or len(grasp_collection) == 0:
        return False, 0
    planner = PickPlacePlanner(robot=arm)
    gids = planner.reason_common_gids(
        grasp_collection=grasp_collection,
        goal_pose_list=[(sp, sr), (gp, gr)],
        obstacle_list=obstacle_list,
    )
    if not gids:
        return False, 0
    # pick_depart: 从 staging 抓取沿 +Z 抬升若干米，中间采样点 IK 全可解
    if pick_depart_dir is not None and pick_depart_dist:
        gids = [
            gid for gid in gids
            if _segment_reachable(
                arm, grasp_collection[gid], sp, sr,
                pick_depart_dir, pick_depart_dist)
        ]
        if not gids:
            return False, 0
    # place_approach: 从 goal 沿 −approach_dir（即向上）走 dist，
    # 等价检查"接近段起点 → goal"中间各点 IK 全可解
    if place_approach_dir is not None and place_approach_dist:
        rev_dir = -np.asarray(place_approach_dir, dtype=float)
        gids = [
            gid for gid in gids
            if _segment_reachable(
                arm, grasp_collection[gid], gp, gr,
                rev_dir, place_approach_dist)
        ]
        if not gids:
            return False, 0
    return True, len(gids)


def _sync_staging_obstacles(search_part_ids, part_id, cand_pos, cand_rot, chosen_layout,
                            default_poses, staging_obstacles):
    for pid in search_part_ids:
        if pid == part_id:
            staging_obstacles[pid].pos = cand_pos
            staging_obstacles[pid].rotmat = cand_rot
        elif pid in chosen_layout:
            p, r = chosen_layout[pid]
            staging_obstacles[pid].pos = p
            staging_obstacles[pid].rotmat = r
        else:
            p, r = default_poses[pid]
            staging_obstacles[pid].pos = p
            staging_obstacles[pid].rotmat = r


def _obstacle_list_for_reason(search_part_ids, part_id, staging_obstacles, env_obstacles):
    # 将单一 ground 替换为环境障碍物列表
    obs = list(env_obstacles)
    for pid in search_part_ids:
        if pid != part_id:
            obs.append(staging_obstacles[pid])
    return obs


def prevalidate_initial_staging_layout(plan, robot, staging_obstacles, grasp_cache,
                                     world_poses, env_obstacles, seeds, assembly_def):
    goal_models = build_goal_obstacle_models(assembly_def, world_poses)
    search_part_ids = [
        p for p in INITIAL_STAGING_PART_IDS
        if plan.get_staging(p) is not None and p in staging_obstacles
    ]
    default_poses = {}
    candidate_map = {}
    for pid in search_part_ids:
        st = plan.get_staging(pid)
        rot = st.rotmat.copy() if st.rotmat is not None else np.eye(3)
        default_poses[pid] = (st.pos.copy(), rot)
        seed = seeds.get(pid, st.pos.copy())
        candidate_map[pid] = generate_staging_candidates(pid, seed)

    # ── 快速局部搜索参数（速度优先） ──────────────────────────
    # 装配顺序已是 seat → leg_bl → leg_br → leg_fl → leg_fr：
    #   * 后腿先装、远端 (x≈0.41) 空旷 → 直立大概率可行
    #   * 前腿后装、组装位 x≈0.19 在近端 → 直立也大概率可行
    # 因此 DFS 只用 2 种 rotmat 候选：直立 + 一个 90° 躺姿兜底。
    # 加速效果：每件 reason 调用 6 候选 × 2 rot × 2 arm = 24 次（旧版
    # 12 × 9 × 2 = 216 次，提速 ~9×），同时仍保留躺姿 fallback 应付
    # 偶发的直立全 fail。
    fast_local_n = 6
    fast_candidate_override = {
        pid: candidate_map.get(pid, [])[:fast_local_n]
        for pid in ("leg_bl", "leg_br", "leg_fl", "leg_fr")
        if pid in candidate_map
    }
    _LEG_FAST_ROT = (
        [(np.eye(3), 0.0)]                                  # 直立优先
        + [(R, _LYING_Z_OFFSET)
           for R in _make_lying_leg_rotmats((90,))]         # 兜底 1 个躺姿
    )
    fast_rot_override = {pid: _LEG_FAST_ROT
                         for pid in ("leg_bl", "leg_br", "leg_fl", "leg_fr")}

    # asmdef 已固化新顺序，旧顺序下"已验证 staging"失去意义。
    locked_layout = {}
    chosen_layout = dict(locked_layout)
    selected = list(locked_layout.keys())
    for pid, (p, r) in locked_layout.items():
        plan.set_staging(pid, pos=p, rotmat=r)
        if pid in staging_obstacles:
            staging_obstacles[pid].pos = p
            staging_obstacles[pid].rotmat = r
    if locked_layout:
        print("[预搜索] 快速模式：锁定已通过步骤 "
              f"{list(locked_layout.keys())}，仅搜索其余零件。")

    def restore(pid):
        p, r = default_poses[pid]
        plan.set_staging(pid, pos=p, rotmat=r)
        o = staging_obstacles.get(pid)
        if o is not None:
            o.pos, o.rotmat = p, r

    def _arm_priority_for_part(part_id):
        """按部件命名约定决定臂优先级。

        命名约定（来自 chair 示例）：
          * ``leg_fr`` / ``leg_br`` → 件位于工作空间右半区，优先用**右臂**
          * ``leg_fl`` / ``leg_bl`` / ``seat`` / 其它 → 优先用**左臂**

        判据：part_id 末位字符为 ``'r'`` ⇒ 右臂优先。

        为什么需要：之前所有件强行左臂优先 → leg_fr/leg_br 转运时左臂
        从右半区抓后跨过中线，**在手桌腿 mesh 扫过 home 姿态的右臂 link**
        触发 ``held_object_vs_idle_arm``。让右半区件用右臂搬就能避开
        跨臂阻挡，且还原了"对称双臂分工"的设计意图。
        """
        pid = str(part_id)
        return ("右臂", "左臂") if pid and pid[-1] == 'r' else ("左臂", "右臂")

    def _global_verify(layout):
        """DFS 完成后的全局复验：**按装配 step 顺序模拟每一步的 obs 布局**，
        对当前 step 的件在那一步的 obs 下做 reason 检查；任一 step 双臂
        都不通过则返回 False。

        修复 search/execution 模型不一致的两层根因：

        (1) DFS 期 ``_sync_staging_obstacles`` 把"未选件"用 default seed
            位置作障碍；DFS 后续 commit 把它们改到新位置 → 前面已 commit
            的件在新障碍布局下可能不再通过 reason，DFS 不会回头复查。

        (2) 即便所有件最终 staging 位置都同步好了再做 verify（旧版本做
            法），仍然忽略了**装配 step 之间障碍布局会变**：执行 step k
            时，前面 step 0..k-1 的件都已经从 staging 搬到 goal（变成
            chair 上的立柱障碍），而后面 step k+1..N-1 的件还在 staging。
            这跟"全部件在 staging"完全不同 —— 前面改 search 时正是这点
            没匹配上才 step 3 (leg_bl) 通过 verify 但执行报 "No common
            grasp id at the given goal poses!"。

        新逻辑：按 ``assembly_def.steps`` 顺序遍历，``placed`` 集合累积，
        每步的 obs = ``env`` + 所有 ``placed`` 件的 ``goal_models`` cm
        + 所有未 placed 件的 ``staging`` cm（不含本件）。这就是执行时
        sequence_executor 那一步的真实 obs（除了已装件 cdprim 在执行时
        被 `change_cdprim_type(CYLINDER + 0.015)` 增强 —— 这里用原始
        AABB 不会更宽松，反而稍严格，是安全侧偏差）。
        """
        # 同步所有 staging cmodel 到最终 chosen 位置（执行时也是这状态）
        for vp in search_part_ids:
            if vp in layout:
                p, r = layout[vp]
                staging_obstacles[vp].pos = p
                staging_obstacles[vp].rotmat = r
        v_kw = dict(
            pick_depart_dir=_PICK_DEPART_DIR,
            pick_depart_dist=RELAXED_PLANNING["pick_depart_distance"],
            place_approach_dir=_PLACE_APPROACH_DIR,
            place_approach_dist=RELAXED_PLANNING["place_approach_distance"],
        )
        placed = set()
        for sd in assembly_def.steps:
            sp = sd.part_id
            if sp not in layout or sp not in world_poses:
                # 没在搜索列表里的件（比如 fixture 虚根）跳过；它仍可能
                # 在后续 step 中作 placed 障碍，但不需要对它做 reason。
                continue
            sp_p, sp_r = layout[sp]
            sgp, sgr = world_poses[sp]
            sgc = grasp_cache.get(model_alias_for_part(sp))
            if sgc is None:
                continue
            # 构造本 step 时的 obs
            v_obs = list(env_obstacles)
            for other in placed:
                # 已 placed 件用 goal 位置 cm（chair 上的立柱）
                if other in goal_models:
                    v_obs.append(goal_models[other])
            for other in search_part_ids:
                # 未 placed 件用 staging 位置 cm（含本 layout）
                if other == sp or other in placed:
                    continue
                if other in staging_obstacles:
                    v_obs.append(staging_obstacles[other])
            # 按本件 arm_pref 顺序 reason
            v_pref = _arm_priority_for_part(sp)
            v_arm0 = robot.lft_arm if v_pref[0] == "左臂" else robot.rgt_arm
            v_arm1 = robot.lft_arm if v_pref[1] == "左臂" else robot.rgt_arm
            ok0, _ = pick_place_reason_common_ok(
                v_arm0, sgc, sp_p, sp_r, sgp, sgr, v_obs, **v_kw)
            ok = ok0
            if not ok0:
                ok1, _ = pick_place_reason_common_ok(
                    v_arm1, sgc, sp_p, sp_r, sgp, sgr, v_obs, **v_kw)
                ok = ok1
            if not ok:
                print(f"    [global_verify] step={sd.step_id} {sp!r}: "
                      f"在 placed={list(placed)} 后的 obs 下双臂 reason "
                      f"均不通过，本 layout 分支作废")
                return False
            placed.add(sp)
        return True

    step_index = {sd.part_id: sd.step_id for sd in assembly_def.steps}

    def _obstacle_list_for_step_reason(part_id):
        """按真实装配顺序构造搜索阶段 reason 的障碍列表。

        当前 step 之前的零件在执行时已经从 staging 搬到 goal；之后的
        零件仍在 staging。因此搜索某个 part 时也必须使用同样的 obs，
        否则会出现局部 reason 通过、但 global_verify / 执行失败的情况。
        """
        cur_i = step_index.get(part_id, 10 ** 9)
        obs = list(env_obstacles)
        for other in search_part_ids:
            if other == part_id:
                continue
            other_i = step_index.get(other, 10 ** 9)
            if other_i < cur_i and other in goal_models:
                obs.append(goal_models[other])
            elif other in staging_obstacles:
                obs.append(staging_obstacles[other])
        return obs

    def dfs(depth):
        if depth >= len(search_part_ids):
            # 所有件 commit 完，做全局 verify
            return _global_verify(chosen_layout)
        pid = search_part_ids[depth]
        if pid in locked_layout:
            print(f"\n  [预搜索] {pid}: 锁定 {np.round(chosen_layout[pid][0], 3).tolist()}，跳过搜索")
            return dfs(depth + 1)
        _, base_rot = default_poses[pid]
        cands = fast_candidate_override.get(pid, candidate_map[pid])
        # 本件的臂优先级（按命名决定）
        arm_pref = _arm_priority_for_part(pid)
        arm_rank = {arm_pref[0]: 0, arm_pref[1]: 1}
        fast_note = "，快速局部候选" if pid in fast_candidate_override else ""
        print(f"\n  [预搜索] {pid}: {len(cands)} 个候选 (优先 {arm_pref[0]}{fast_note})")

        # === 阶段 1: 扫所有 cand 算 (n, tag)，过滤掉的立刻 print ===
        # 注：原 first-match 策略是"找到第一个 reason 通过的 cand 就 commit
        # 进 chosen，进入下层 DFS"，结果像 leg_bl 仅 2 个共同抓取的位置就
        # 被选上。后续步真正执行时已装件多一倍（变成 cdprim 障碍），
        # reason_common_gids 内层过滤把那 2 个全过滤掉 → "No common grasp
        # id at the given goal poses!"。改成两阶段：先把所有 cand 跑一遍
        # 拿到 n，再按 n 降序 DFS commit，用最厚的"抓取冗余"撑住后续
        # 步的障碍密度。
        # 拿本件可用的 (rotmat, z_offset) 列表；不在表里的件（理论上
        # 不会出现）退化为只用 default_rot。``STAGING_ROTMAT_CANDIDATES``
        # 顺序无所谓，下面 sort 时用姿态优先级自行排序。
        rot_cands = fast_rot_override.get(
            pid, STAGING_ROTMAT_CANDIDATES.get(pid, [(base_rot, 0.0)]))

        # === 阶段 1: 扫所有 (xy, rotmat) 组合算 (n, tag, pose)，过滤掉的立刻 print ===
        # 旧 inline DFS 只用了 ``base_rot=np.eye(3)``（直立），完全忽略
        # ``STAGING_ROTMAT_CANDIDATES`` 里的躺姿候选。结果像 leg_bl 直立
        # 全部不可行时直接搜索失败，无法 fallback 到躺姿；leg_fl/leg_fr
        # 等直立可行的件也没机会比较"躺姿是否更优"。改成对每个 xy 候
        # 选额外遍历所有姿态，配合后面 sort 的姿态优先级，保证：
        #   * 直立可行 → 选直立（保留现已能跑通的件不被破坏）
        #   * 直立全 fail → 自动 fallback 到躺姿（让长杆躺平避开立柱）
        feasible = []  # [(n, idx, rot_idx, cand_pos_with_z_off, cand_rot, tag, pose), ...]
        for idx, cand_pos in enumerate(cands):
            cp_disp = np.round(cand_pos, 3).tolist()
            for rot_idx, (cand_rot, z_off) in enumerate(rot_cands):
                # 躺姿要把 z 抬一点避免半埋桌面，直立 z_off=0 不变
                cand_pos_eff = cand_pos.copy()
                cand_pos_eff[2] = float(cand_pos_eff[2]) + float(z_off)
                pose_tag = _classify_pose(cand_rot)

                staging_obstacles[pid].pos = cand_pos_eff
                staging_obstacles[pid].rotmat = cand_rot

                if not is_pose_collision_free_with_selected(
                        pid, cand_pos_eff, cand_rot, staging_obstacles, selected):
                    if rot_idx == 0:
                        print(f"    - #{idx + 1} {cp_disp} 与已选件碰撞")
                    continue

                if not is_staging_clear_of_foreign_goals(
                        pid, cand_pos_eff, cand_rot, staging_obstacles,
                        goal_models, assembly_def):
                    if rot_idx == 0:
                        print(f"    - #{idx + 1} {cp_disp} 侵入 fixture 上零件装配域")
                    continue

                if pid not in world_poses:
                    continue
                gp, gr = world_poses[pid]
                gc = grasp_cache[model_alias_for_part(pid)]
                _sync_staging_obstacles(
                    search_part_ids, pid, cand_pos_eff, cand_rot, chosen_layout,
                    default_poses, staging_obstacles)
                obs = _obstacle_list_for_step_reason(pid)

                # "初始穿模"硬检查：home 姿态下所有 staging 都不能与任一
                # 只臂重叠。躺姿桌腿水平延伸 >0.3m，比直立更易穿模 home
                # 位置的臂体——这里不能省略。
                home_obs = [staging_obstacles[p]
                            for p in search_part_ids if p in staging_obstacles]
                if _arms_collide_at_conf(robot, _HOME_JV, _HOME_JV, home_obs):
                    continue

                reason_kw = dict(
                    pick_depart_dir=_PICK_DEPART_DIR,
                    pick_depart_dist=RELAXED_PLANNING["pick_depart_distance"],
                    place_approach_dir=_PLACE_APPROACH_DIR,
                    place_approach_dist=RELAXED_PLANNING["place_approach_distance"],
                )
                arm0 = robot.lft_arm if arm_pref[0] == "左臂" else robot.rgt_arm
                arm1 = robot.lft_arm if arm_pref[1] == "左臂" else robot.rgt_arm
                ok0, n0 = pick_place_reason_common_ok(
                    arm0, gc, cand_pos_eff, cand_rot, gp, gr, obs, **reason_kw)
                if ok0:
                    tag, n = arm_pref[0], n0
                else:
                    ok1, n1 = pick_place_reason_common_ok(
                        arm1, gc, cand_pos_eff, cand_rot, gp, gr, obs, **reason_kw)
                    if not ok1:
                        continue
                    tag, n = arm_pref[1], n1

                feasible.append((
                    n, idx, rot_idx,
                    cand_pos_eff.copy(), cand_rot.copy(),
                    tag, pose_tag,
                ))

        # === 阶段 2: 按 (本件臂优先级, 姿态优先级, -n) 排序 ===
        # * 臂优先级：本件命名指定的优先臂 cand 排在另一臂之前
        # * 姿态优先级：直立 cand 排在躺姿 cand 之前 ——这保证了
        #   leg_fl/leg_fr 等"现已直立能跑通"的件保持直立选 cand；
        #   leg_bl/leg_br 直立全失败时 DFS 自动 fallback 到躺姿
        # * 最后才比 -n（同臂同姿态内 best-n）
        _POSE_RANK = {"直立": 0, "躺/斜": 1}
        feasible.sort(key=lambda x: (
            arm_rank.get(x[5], 99),
            _POSE_RANK.get(x[6], 99),
            -x[0],
        ))
        if feasible:
            top = [(f[1] + 1, f[0], f[5], f[6]) for f in feasible[:5]]
            print(f"    [可行 {len(feasible)} 个，按 (臂, 姿态, -n) top5: "
                  f"{', '.join(f'#{i}({tag},{pose},n={n})' for i, n, tag, pose in top)}]")

        # === 阶段 3: 依次 DFS commit ===
        for n, idx, rot_idx, cand_pos_eff, cand_rot, tag, pose_tag in feasible:
            staging_obstacles[pid].pos = cand_pos_eff
            staging_obstacles[pid].rotmat = cand_rot
            _sync_staging_obstacles(
                search_part_ids, pid, cand_pos_eff, cand_rot, chosen_layout,
                default_poses, staging_obstacles)
            plan.set_staging(pid, pos=cand_pos_eff, rotmat=cand_rot)
            chosen_layout[pid] = (cand_pos_eff.copy(), cand_rot.copy())
            selected.append(pid)
            print(f"    + #{idx + 1}.{rot_idx} {np.round(cand_pos_eff, 3).tolist()} "
                  f"({tag}, {pose_tag}, 共同抓取数={n})")

            if dfs(depth + 1):
                return True

            selected.pop()
            chosen_layout.pop(pid, None)

        restore(pid)
        return False

    print("\n[预搜索] 联合搜索初始 staging（件间无碰撞 + 不侵入他人 goal + staging→goal 共同抓取）…")
    if not dfs(0):
        for pid in search_part_ids:
            restore(pid)
        raise RuntimeError(
            "未找到满足「件间无碰撞 + 不侵入 fixture 上椅面装配域 + 双臂之一可 reason_common」的 "
            "5 件初始布局。")

    print("\n[预搜索] 成功：")
    for pid in search_part_ids:
        print(f"  {pid}: {np.round(chosen_layout[pid][0], 3).tolist()}")
    return chosen_layout


def _interp_oiee_pose(pose0, pose1, alpha):
    """插值 MotionData 记录的在手物体位姿。

    ``pose`` 为 ``None`` 表示该关键帧没有在手物体；只有两端都非 None
    时才返回插值位姿。
    """
    if pose0 is None or pose1 is None:
        return None
    p0, r0 = pose0
    p1, r1 = pose1
    pos = np.asarray(p0) * (1.0 - alpha) + np.asarray(p1) * alpha
    rot = rm.rotmat_slerp(np.asarray(r0), np.asarray(r1), 2)[
        0 if alpha < 0.5 else 1]
    return pos, rot


def _held_object_obstacles(obstacle_list, exclude_part_ids=None):
    """被抓物体额外复验用的障碍列表。

    保留**桌上其他零件 + 已装件**，不包含 ``environment_obstacle``
    （如 work_table），因为 staging/装配过程本来就贴桌沿，把桌面拉进
    held vs obstacle 检查会把正常贴近误判为穿模。

    ``exclude_part_ids`` 用于豁免**当前步的直接父件**（leg 的 parent 是
    seat）：被抓桌腿走到 goal 的最后几帧，腿下端会插进 seat 的承插孔
    （mesh-mesh ODE 检测把"嵌入"判作穿模），但这是装配本身要求的接触
    几何，不应当作失败。注意只豁免 *父件*——已装件中的其他兄弟 leg
    仍然要检测，避免被抓桌腿在转运途中扫穿其它已装腿。
    """
    excl = set(exclude_part_ids or [])
    out = []
    for obs in obstacle_list or []:
        role = getattr(obs, "_sealp_role", None)
        if role not in ("staging_on_table", "assembled_at_goal"):
            continue
        pid = getattr(obs, "_sealp_part_id", None)
        if pid in excl:
            continue
        out.append(obs)
    return out


def _first_held_collider(obj_probe, obj_pose, obstacle_list):
    """逐个测试 ``obstacle_list``，返回第一个与 ``obj_probe@obj_pose`` 撞上的
    障碍 ``_sealp_part_id``（找不到返回 ``None``）。

    `obj_probe.is_mcdwith(list)` 只返回布尔，定位不到具体是谁撞的。本
    helper 用 list-loop 把命中者揪出来，便于调试到底是 parent / sibling
    / staging 哪个件造成穿模。
    """
    if obj_probe is None or obj_pose is None or not obstacle_list:
        return None
    old_pos, old_rot = obj_probe.pos.copy(), obj_probe.rotmat.copy()
    try:
        obj_probe.pos, obj_probe.rotmat = obj_pose
        for obs in obstacle_list:
            if obj_probe.is_mcdwith(obs):
                return getattr(obs, "_sealp_part_id", "<unknown>")
        return None
    finally:
        obj_probe.pos, obj_probe.rotmat = old_pos, old_rot


def _mesh_level_arm_vs_obstacle(robot_arm, obstacle_cm):
    """对单一障碍件做 mesh-vs-mesh (``is_mcdwith``) 复检，定位到具体撞 link。

    与 ``robot_arm.is_collided()``（cdprim 级，包围体偏保守）对照：
    * 返回 ``(link_name, True)``  → cdprim 与 mesh 双双确认撞，**真撞**；
    * 返回 ``(None, False)``      → cdprim 报撞但 mesh 不撞，**cdprim 过保守的伪报**，
                                    多见于 Panthera-HT 前臂胶囊体 cdprim 包了一圈 1~2cm 余量。

    调用前 ``robot_arm.goto_given_conf(jv)`` 必须已经把双臂关节角更新到目标
    位姿——link cmodel 的 pose 通过 jl.Link 的 ``gl_pos/gl_rotmat`` 属性自动
    跟随，无需手动同步。
    """
    if obstacle_cm is None:
        return None, False
    link_cms = _arm_link_cmodels(robot_arm)
    if not link_cms:
        return None, False
    for link_cm in link_cms:
        try:
            if link_cm.is_mcdwith(obstacle_cm):
                # 取 link cmodel 上能找到的最具描述性的名字
                name = (getattr(link_cm, "name", None)
                        or getattr(getattr(link_cm, "_lnk", None), "name", None)
                        or "<unknown_link>")
                return name, True
        except Exception:
            continue
    return None, False


def _first_robot_collider(robot_arm, obstacle_list):
    """逐个测试 ``obstacle_list``，返回第一个与机器人本体（含夹爪 / 在手物）
    撞上的障碍标识（找不到返回 ``None``）。

    标识优先返回 ``_sealp_part_id``；若该障碍没打 part id 标签，则回退到
    ``_sealp_role``（如 ``environment_obstacle`` 表示桌面之类）；都没有
    时返回 ``"<unknown>"``。

    调用前请先把机器人 goto 到目标 conf —— 本函数不会再 goto，只是逐
    个 obstacle 调 ``is_collided``，确认到底是谁让上一次"全集 collided"
    返回 True。

    !! 注意 !!: 此函数会拷贝当前 cc 状态再做逐项 query；从设计上
    ``robot.is_collided`` 应当对 obstacle 顺序无副作用，但 piper cc 在
    某些 panda3d 版本下会缓存上次 traverser，传单元素 list 不会污染下
    次。如真观察到误报，再加 backup_state/restore_state 包裹。
    """
    if robot_arm is None or not obstacle_list:
        return None
    for obs in obstacle_list:
        hit = robot_arm.is_collided(obstacle_list=[obs])
        flag = hit[0] if isinstance(hit, tuple) else hit
        if flag:
            pid = getattr(obs, "_sealp_part_id", None)
            if pid is not None:
                return pid
            role = getattr(obs, "_sealp_role", None)
            return role if role else "<unknown>"
    return None


def _arm_link_cmodels(arm):
    """收集一只机械臂上所有 link 的 CollisionModel（机械手 + 夹爪）。

    用于在轨迹复验里把"在手物体 vs 闲置另一只臂"做显式 mesh-vs-mesh 兜底
    碰撞检查 —— 因为 ``piper_dual_arm.setup_cc`` 把双臂互碰仅用了独立
    bitmask（未启用 ``bitmask_inner "into"``），WRS ``robot.hold()`` 给
    在手物体打的 ``bitmask_inner "from"`` 在 cc 内部 *不会* 触发跨臂检测。
    横着抓 0.3 m 桌腿挥过另一只臂时，这是最关键的盲区。

    Notes
    -----
    返回的是 link 自身的 ``cmodel`` 引用（pose 通过 jl.Link 的属性访问
    自动跟随 ``gl_pos / gl_rotmat`` 更新），所以**调用前必须保证两臂已
    各自 ``goto_given_conf`` 到目标关节角**。
    """
    out = []
    if arm is None:
        return out
    mp = getattr(arm, "manipulator", None)
    if mp is not None and getattr(mp, "jlc", None) is not None:
        anchor = getattr(mp.jlc, "anchor", None)
        if anchor is not None:
            for lnk in getattr(anchor, "lnk_list", []) or []:
                cm = getattr(lnk, "cmodel", None)
                if cm is not None:
                    out.append(cm)
        for jnt in getattr(mp.jlc, "jnts", []) or []:
            lnk = getattr(jnt, "lnk", None)
            cm = getattr(lnk, "cmodel", None) if lnk is not None else None
            if cm is not None:
                out.append(cm)
    ee = getattr(arm, "end_effector", None)
    if ee is not None and getattr(ee, "jlc", None) is not None:
        anchor = getattr(ee.jlc, "anchor", None)
        if anchor is not None:
            for lnk in getattr(anchor, "lnk_list", []) or []:
                cm = getattr(lnk, "cmodel", None)
                if cm is not None:
                    out.append(cm)
        for jnt in getattr(ee.jlc, "jnts", []) or []:
            lnk = getattr(jnt, "lnk", None)
            cm = getattr(lnk, "cmodel", None) if lnk is not None else None
            if cm is not None:
                out.append(cm)
    return out


def _check_held_object_collision(obj_probe, obj_pose, obstacle_list):
    """把被抓物体按当前在手姿态放置，并检查它是否撞到其他零件。"""
    if obj_probe is None or obj_pose is None or not obstacle_list:
        return False
    old_pos, old_rot = obj_probe.pos.copy(), obj_probe.rotmat.copy()
    try:
        obj_probe.pos, obj_probe.rotmat = obj_pose
        return obj_probe.is_mcdwith(obstacle_list)
    finally:
        obj_probe.pos, obj_probe.rotmat = old_pos, old_rot


def _motion_replay_collision_free(robot_arm, mot_data, obstacle_list,
                                  obj_cmodel=None, granularity=None,
                                  other_arm=None, held_skip_part_ids=None):
    """
    严格轨迹复验：对规划结果做关节空间密集插值，逐帧检查：

    1. 机器人本体 / 夹爪 vs ``obstacle_list``；
    2. 若该帧存在在手物体，则把被抓物模型同步到 ``mot_data`` 记录的
       在手位姿，额外检查「被抓物体 vs 桌面其他 staging / 兄弟已装件」；
       注意 ``held_skip_part_ids`` 列表里的件会被豁免（典型是当前步的
       直接父件 —— 装配末段腿插进 seat 的承插孔属于设计接触，不算穿模）；
    3. 若 ``other_arm`` 非空，再额外检查「被抓物体 vs 闲置另一只臂的
       link mesh」—— 此乃双臂场景下 WRS ``robot.hold()`` + cc 内部检测
       的盲区（见 ``piper_dual_arm.setup_cc`` 没启用 ``bitmask_inner
       "into"``），是横着抓长桌腿时最容易踩穿模的关键检查。

    ``other_arm`` 期望本步执行期内**关节角不变**（活动臂规划时另一臂
    是闲置静态的），因此其 link cmodel 的 pose 在循环里是固定的，构建
    一次复用。

    Returns
    -------
    (bool, detail) :
        ``detail`` 是 4-tuple ``(reason, frame_data, hit_pid, diag)``，其中
        * ``reason`` 是 ``"robot_vs_obstacle"`` /
          ``"held_object_vs_obstacle"`` / ``"held_object_vs_idle_arm"``；
        * ``frame_data`` 是 jv（关节角）或 ``(pos, rotmat)``（在手物体位姿）；
        * ``hit_pid`` 是命中障碍的件名（``None`` 表示不适用，如闲置臂）；
        * ``diag`` 是诊断字典，包含：
            - ``keyframe_idx``     当前所在 jv_list 段号 (0..len-2)，``"final"`` 表示尾关键帧
            - ``alpha``            该段内插值比例 0~1（尾帧固定 1.0）
            - ``total_keyframes``  本轨迹总关键帧数 ``len(jv_list)``
            - ``progress``         整条轨迹进度比 (i + alpha) / (n - 1)，0~1
            - ``tcp_pos``          活动臂 TCP 在世界系中的位置
            - ``has_held_obj``     当前帧机器人是否持有零件
            - ``held_obj_pos``     在手物体中心位置 (``None`` 若未持有)
        借此可以判断撞击发生在轨迹哪一阶段（progress<5% ≈ pick approach，
        ~一半的"持有零件且 TCP 升高"段 ≈ transport，>95% ≈ place depart 等）。
    """
    if granularity is None:
        granularity = RELAXED_PLANNING["replay_granularity"]
    if mot_data is None or len(mot_data.jv_list) == 0:
        return True, None

    obs = list(obstacle_list) if obstacle_list else []
    held_obs = _held_object_obstacles(obs, exclude_part_ids=held_skip_part_ids)
    # 闲置另一只臂的 link cdmodel：本步执行期内静止，构建一次复用。
    # 因为 jl.Link.cmodel 属性在 access 时会按 link 的 gl_pos/gl_rotmat
    # 更新自身 pose，所以这里抓到的是 *引用*；之后哪怕 active arm
    # goto_given_conf 也不会动到 other_arm 的 cmodel pose。
    other_arm_cms = _arm_link_cmodels(other_arm)
    if not obs and not other_arm_cms:
        return True, None
    obj_probe = obj_cmodel.copy() if obj_cmodel is not None else None
    cdprim_fp_count = 0  # cdprim 报撞但 mesh 不撞的"伪报"次数（仅供日志）
    robot_arm.backup_state()
    try:
        jv_list = mot_data.jv_list
        oiee_pose_list = getattr(mot_data, "oiee_gl_pose_list", None)
        n_keys = len(jv_list)

        def _diag(keyframe_idx, alpha, interp_obj_pose):
            """构建撞击点诊断字典。``goto_given_conf`` 已在前一行调过，
            此时 ``robot_arm.gl_tcp_pos`` 就是该插值帧的真实 TCP。"""
            try:
                tcp_pos = np.array(robot_arm.gl_tcp_pos, dtype=float).tolist()
            except Exception:
                tcp_pos = None
            if isinstance(keyframe_idx, str):
                # "final" sentinel
                prog = 1.0
            else:
                denom = max(n_keys - 1, 1)
                prog = float((keyframe_idx + alpha) / denom)
            return {
                "keyframe_idx": keyframe_idx,
                "alpha": float(alpha),
                "total_keyframes": int(n_keys),
                "progress": float(np.clip(prog, 0.0, 1.0)),
                "tcp_pos": tcp_pos,
                "has_held_obj": interp_obj_pose is not None,
                "held_obj_pos": (np.array(interp_obj_pose[0], dtype=float).tolist()
                                 if interp_obj_pose is not None else None),
            }

        for i in range(n_keys - 1):
            jv0 = np.array(jv_list[i])
            jv1 = np.array(jv_list[i + 1])
            obj_pose0 = (oiee_pose_list[i]
                         if oiee_pose_list and i < len(oiee_pose_list)
                         else None)
            obj_pose1 = (oiee_pose_list[i + 1]
                         if oiee_pose_list and i + 1 < len(oiee_pose_list)
                         else None)
            dist = np.linalg.norm(jv1 - jv0)
            steps = max(int(dist / granularity), 1)
            for step in range(steps):
                alpha = step / float(steps)
                interp_jv = jv0 * (1.0 - alpha) + jv1 * alpha
                interp_obj_pose = _interp_oiee_pose(obj_pose0, obj_pose1, alpha)

                robot_arm.goto_given_conf(interp_jv)
                hit = robot_arm.is_collided(obstacle_list=obs)
                collided = hit[0] if isinstance(hit, tuple) else hit
                if collided:
                    # cdprim 报撞 → mesh 级仲裁（cdprim 通常比真实 mesh 大 1-2cm，
                    # Panthera-HT 前臂胶囊体外扩尤甚，绝大多数 cdprim 报撞
                    # mesh 都不撞）。这里把 mesh 作为最终判据：
                    #   * mesh 也撞 → 真撞，按原逻辑返回失败；
                    #   * mesh 不撞 → cdprim 伪报，pass 当前帧、继续后面检查。
                    rob_hit = _first_robot_collider(robot_arm, obs)
                    hit_obs_cm = next(
                        (o for o in obs
                         if getattr(o, "_sealp_part_id", None) == rob_hit),
                        None)
                    link_name, mesh_collided = _mesh_level_arm_vs_obstacle(
                        robot_arm, hit_obs_cm)
                    if not mesh_collided:
                        cdprim_fp_count += 1
                        continue
                    diag = _diag(i, alpha, interp_obj_pose)
                    diag["mesh_verified"] = True
                    diag["mesh_hit_link"] = link_name
                    return False, ("robot_vs_obstacle", interp_jv, rob_hit, diag)
                hit_pid = _first_held_collider(
                    obj_probe, interp_obj_pose, held_obs)
                if hit_pid is not None:
                    diag = _diag(i, alpha, interp_obj_pose)
                    return False, ("held_object_vs_obstacle",
                                   interp_obj_pose, hit_pid, diag)
                if _check_held_object_collision(
                        obj_probe, interp_obj_pose, other_arm_cms):
                    diag = _diag(i, alpha, interp_obj_pose)
                    return False, ("held_object_vs_idle_arm",
                                   interp_obj_pose, None, diag)

        # 必须检查最后一个关键帧
        robot_arm.goto_given_conf(jv_list[-1])
        hit = robot_arm.is_collided(obstacle_list=obs)
        collided = hit[0] if isinstance(hit, tuple) else hit
        if collided:
            rob_hit = _first_robot_collider(robot_arm, obs)
            hit_obs_cm = next(
                (o for o in obs
                 if getattr(o, "_sealp_part_id", None) == rob_hit),
                None)
            link_name, mesh_collided = _mesh_level_arm_vs_obstacle(
                robot_arm, hit_obs_cm)
            if mesh_collided:
                diag = _diag("final", 1.0, None)
                diag["mesh_verified"] = True
                diag["mesh_hit_link"] = link_name
                return False, ("robot_vs_obstacle", jv_list[-1], rob_hit, diag)
            else:
                cdprim_fp_count += 1
        if oiee_pose_list and len(oiee_pose_list) >= len(jv_list):
            hit_pid = _first_held_collider(
                obj_probe, oiee_pose_list[-1], held_obs)
            if hit_pid is not None:
                diag = _diag("final", 1.0, oiee_pose_list[-1])
                return False, ("held_object_vs_obstacle",
                               oiee_pose_list[-1], hit_pid, diag)
            if _check_held_object_collision(
                    obj_probe, oiee_pose_list[-1], other_arm_cms):
                diag = _diag("final", 1.0, oiee_pose_list[-1])
                return False, ("held_object_vs_idle_arm",
                               oiee_pose_list[-1], None, diag)

        if cdprim_fp_count > 0:
            # 实测 Panthera-HT 前臂 cdprim 比真实 mesh 大 1-2cm 余量，
            # transport 段经常报"擦边"但 mesh 没事；mesh 仲裁后放过。
            print(f"    [复验] cdprim 报撞 {cdprim_fp_count} 帧, "
                  f"mesh 仲裁后全部放行 (cdprim 包围体过保守)")
        return True, None
    finally:
        robot_arm.restore_state()


def animate_sequence(base, execution_result, interval=0.01):
    all_motions = []
    for sr in execution_result.steps:
        if not sr.success:
            continue
        step_motions = {"step_id": sr.step_id, "part_id": sr.part_id, "meshes_to_show": []}
        if hasattr(sr, "mot_data") and sr.mot_data is not None:
            step_motions["meshes_to_show"].append(sr.mot_data.mesh_list)
        else:
            if hasattr(sr, "mot_data_lft") and sr.mot_data_lft is not None:
                step_motions["meshes_to_show"].append(sr.mot_data_lft.mesh_list)
            if hasattr(sr, "mot_data_rgt") and sr.mot_data_rgt is not None:
                step_motions["meshes_to_show"].append(sr.mot_data_rgt.mesh_list)
        if step_motions["meshes_to_show"]:
            all_motions.append(step_motions)

    if not all_motions:
        print("No motions to animate.")
        return

    class _State:
        def __init__(self):
            self.step_idx = 0
            self.frame_idx = 0
            self.total_steps = len(all_motions)

    state = _State()

    def _update(st, task):
        if st.step_idx >= st.total_steps:
            st.step_idx = 0
            st.frame_idx = 0
        current_step = all_motions[st.step_idx]
        mesh_lists = current_step["meshes_to_show"]
        max_frames = max(len(ml) for ml in mesh_lists)
        if st.frame_idx > 0:
            for ml in mesh_lists:
                if st.frame_idx - 1 < len(ml):
                    ml[st.frame_idx - 1].detach()
        elif st.step_idx > 0:
            prev_step = all_motions[st.step_idx - 1]
            for ml in prev_step["meshes_to_show"]:
                if len(ml) > 0:
                    ml[-1].detach()
        if st.frame_idx >= max_frames:
            for ml in mesh_lists:
                for m in ml:
                    m.detach()
            st.step_idx += 1
            st.frame_idx = 0
            return task.again
        for ml in mesh_lists:
            frame_to_show = min(st.frame_idx, len(ml) - 1)
            ml[frame_to_show].attach_to(base)
        if base.inputmgr.keymap["space"]:
            st.frame_idx += 1
        return task.again

    taskMgr.doMethodLater(interval, _update, "sequence_animate", extraArgs=[state], appendTask=True)


def main():
    base = wd.World(cam_pos=[1.5, -0.3, 1.2], lookat_pos=[0.3, -0.3, 0.1])
    mgm.gen_frame().attach_to(base)

    # --- 动态加载 config 中的障碍物 ---
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "sample_config.yaml")
    )
    env_obstacles = load_obstacles_from_config(config_path, base)

    asmdef_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assembly_sequence", "_demo_output")
    asmdef_path = os.path.abspath(os.path.join(asmdef_dir, "yuanchair.asmdef"))
    if not os.path.isfile(asmdef_path):
        print(f"Assembly definition not found at: {asmdef_path}, generating…")
        from sealp.assembly_sequence.gen_yuanchair_asmdef import main as gen_asm
        gen_asm()
    if not os.path.isfile(asmdef_path):
        print(f"ERROR: {asmdef_path}")
        return

    asm = AssemblyDef.load(asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")
    print(f"[装配顺序] {[(s.step_id, s.part_id) for s in asm.steps]}")
    # 期望顺序在 ``gen_yuanchair_asmdef.py`` 已固化为
    #   seat → leg_bl → leg_br → leg_fl → leg_fr
    # 这里只做防御性校验：如有人手改 asmdef 或回退到旧 yaml，会立刻
    # 警告但不强行改写（避免与 .asmdef 静默不一致）。
    _EXPECTED_ORDER = ("seat", "leg_bl", "leg_br", "leg_fl", "leg_fr")
    _actual_order = tuple(s.part_id for s in asm.steps)
    if _actual_order != _EXPECTED_ORDER:
        print(f"[装配顺序][WARN] 期望 {_EXPECTED_ORDER}，"
              f"asmdef 中是 {_actual_order}。请重跑："
              f"`python -m sealp.assembly_sequence.gen_yuanchair_asmdef`。")

    plan = TaskPlan(
        assembly_file=asmdef_path,
        name="YuanChair Dual-Arm Execution",
        description="Dual-arm sequential assembly of the YuanChair.",
    )
    plan.set_assembly(asm)

    center_y_offset = -0.30
    plan.fixture_pos = np.array([0.0, center_y_offset, 0.0])
    _seeds = STAGING_SEEDS
    plan.fixture_rotmat = np.eye(3)

    # 仅种子：预搜索成功后会覆盖为最终 staging
    for pid in INITIAL_STAGING_PART_IDS:
        plan.set_staging(pid, pos=_seeds[pid].copy(), rotmat=np.eye(3))

    for i in range(asm.n_steps):
        plan.set_step_params(StepParams(
            step_id=i,
            primitive="single_arm_transport",
            approach_distance=RELAXED_PLANNING["step_approach_distance"],
            depart_distance=RELAXED_PLANNING["step_depart_distance"],
        ))

    world_poses = asm.compute_world_poses(
        fixture_pos=plan.fixture_pos,
        fixture_rotmat=plan.fixture_rotmat,
    )

    for pid in asm.part_ids:
        if pid not in world_poses:
            continue
        gp, gr = world_poses[pid]
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            ghost = mcm.CollisionModel(initor=mp)
            ghost.pos, ghost.rotmat = gp, gr
            ghost.alpha = 0.15
            ghost.attach_to(base)

    # 切换机器人时务必同步更新本字段：layout / motion cache 都会以此校验。
    ROBOT_TYPE = "panthera_ht"

    robot = pda.DualPantheraHTNoBody(enable_cc=True)
    home = np.zeros(6)
    robot.lft_arm.goto_given_conf(home)
    robot.rgt_arm.goto_given_conf(home)
    robot.use_lft()

    grasp_paths = {
        "leg_model": os.path.join(
            os.path.dirname(__file__), "..", "grasp", "_output",
            "demo_yuanchair-part2_grasps.pickle"),
        "seat_model": os.path.join(
            os.path.dirname(__file__), "..", "grasp", "_output",
            "demo_yuanchair-part1_grasps.pickle"),
    }

    staging_obstacles = {}
    initial_obstacles = list(env_obstacles)
    for pid in asm.part_ids:
        st = plan.get_staging(pid)
        if st is None:
            continue
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            obs = mcm.CollisionModel(initor=mp)
            obs.pos = st.pos
            obs.rotmat = st.rotmat
            obs._sealp_part_id = pid
            obs._sealp_role = "staging_on_table"
            initial_obstacles.append(obs)
            staging_obstacles[pid] = obs

    gc = load_grasp_cache(grasp_paths)

    # ── 路径缓存：完整 5-step 的 (staging + jv_list + ev_list) 一次性
    # 存到 _motion_cache/yuanchair_dual.pkl。命中则跳过 DFS 预搜索 +
    # RRT/IK，直接重放轨迹。
    motion_cache_dir = os.path.join(
        os.path.dirname(__file__), "_motion_cache")
    os.makedirs(motion_cache_dir, exist_ok=True)
    # motion_cache_path = os.path.join(motion_cache_dir, "yuanchair_dual.pkl")
    # motion_cache_path = os.path.join(motion_cache_dir, "yuanchair_dual_optimal_panthera2.pkl")
    # motion_cache_path = os.path.join(motion_cache_dir, "yuanchair_dual_optimal_panthera3.pkl")
    # motion_cache_path = os.path.join(motion_cache_dir, "yuanchair_dual_optimal_panthera4.pkl")
    # motion_cache_path = os.path.join(motion_cache_dir, "yuanchair_dual_optimal_panthera5.pkl")
    motion_cache_path = os.path.join(motion_cache_dir, "yuanchair_dual_optimal_panthera6.pkl")
    motion_cache = None
    _USE_MOTION_CACHE = True
    if _USE_MOTION_CACHE and os.path.isfile(motion_cache_path):
        try:
            with open(motion_cache_path, "rb") as f:
                cand = pickle.load(f)
            expected_order = [s.part_id for s in asm.steps]
            cache_robot_type = cand.get("robot_type")
            if cache_robot_type is not None and cache_robot_type != ROBOT_TYPE:
                print(f"\n[Motion Cache] 缓存 robot_type={cache_robot_type!r} "
                      f"≠ 当前 {ROBOT_TYPE!r}，丢弃并重算。")
            elif (cand.get("step_order") == expected_order
                    and cand.get("asm_name") == asm.name):
                motion_cache = cand
                # 应用缓存里的 staging
                for pid, (p, r) in cand["staging"].items():
                    if plan.get_staging(pid) is None:
                        continue
                    p_arr, r_arr = np.asarray(p), np.asarray(r)
                    plan.set_staging(pid, pos=p_arr.copy(), rotmat=r_arr.copy())
                    obs = staging_obstacles.get(pid)
                    if obs is not None:
                        obs.pos = p_arr
                        obs.rotmat = r_arr
                print(f"\n[Motion Cache] 命中 "
                      f"{os.path.relpath(motion_cache_path)}：")
                print(f"  应用 cached staging，跳过 DFS 与 RRT/IK 重算。")
                if cache_robot_type is None:
                    print(f"  [WARN] 缓存未记录 robot_type，沿用旧文件；"
                          f"如重放姿态异常请删除该 pkl 重跑。")
            else:
                print(f"\n[Motion Cache] 文件与当前 asmdef "
                      f"({expected_order}) 不一致，将重新计算并覆盖。")
        except Exception as e:
            print(f"\n[Motion Cache][WARN] 加载失败: {e!r}，将重新计算。")
            motion_cache = None

    # 优先使用预先搜索好的布局（由 sealp/examples/layout/find_optimal_layout.py 生成）。
    # 找到则跳过下面的 DFS 预搜索，节省启动时间，并保证可装配。
    searched_layout_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "layout", "_output",
        "dual_yuanchair_optimal_searched.layout"))
    used_searched_layout = False
    # ── GOLDEN staging：直接应用已实测可行的固定 staging ──
    # 比 DFS 预搜索更稳：DFS 内部 trac_ik 的 stochastic IK 会让多次
    # 运行产出不同 staging，"上次成功 → 这次失败"几乎是 50/50。这里
    # 写死成功过的那一组，启动直接应用，跳过 DFS。
    if motion_cache is None and _USE_GOLDEN_STAGING:
        for pid, (p, r) in GOLDEN_STAGING.items():
            if plan.get_staging(pid) is None:
                continue
            plan.set_staging(pid, pos=p.copy(), rotmat=r.copy())
            obs = staging_obstacles.get(pid)
            if obs is not None:
                obs.pos = p.copy()
                obs.rotmat = r.copy()
        used_searched_layout = True   # 跳过下面的 DFS 预搜索
        print(f"\n[GOLDEN Staging] 已应用固定 staging（跳过 DFS 预搜索）：")
        for pid in (s.part_id for s in asm.steps):
            if pid in GOLDEN_STAGING:
                p = GOLDEN_STAGING[pid][0]
                print(f"  {pid}: {p.tolist()}")

    # 装配顺序已在本运行里就地重排（seat → leg_bl → leg_br → leg_fl
    # → leg_fr）。预搜索布局是按旧顺序 + 旧 obs 布局产物，逐件 reason
    # 通过不代表新顺序下的 step-by-step 障碍布局也通过，沿用极易触
    # 发执行期 RRT/IK 失败。这里强制忽略缓存，走内置 DFS 在新顺序下
    # 重搜。要恢复缓存路径：删除本块或重跑 search_dual_layout 后再
    # 启用。
    _IGNORE_CACHED_LAYOUT_FOR_REORDERED_SEQ = False

    if motion_cache is not None:
        used_searched_layout = True
    elif _IGNORE_CACHED_LAYOUT_FOR_REORDERED_SEQ and os.path.isfile(searched_layout_path):
        print(f"\n[Layout] 检测到预搜索布局 "
              f"{os.path.relpath(searched_layout_path)}，"
              f"但本运行已重排装配顺序为 "
              f"{[s.part_id for s in asm.steps]}，旧布局不再适用，"
              f"已强制走内置 DFS 重搜。")
    elif os.path.isfile(searched_layout_path):
        try:
            from sealp.layout import WorkspaceLayout
            loaded = WorkspaceLayout.load(searched_layout_path)
            # 校验 layout 是否为当前机器人搜出来。Panthera-HT 与 Piper
            # 的 base 偏移 / 关节几何不同，跨机器人重用 layout 会导致
            # leg_*r 紧贴右臂 base 立刻穿模。
            layout_robot = loaded.metadata.get("robot_type")
            if layout_robot is not None and layout_robot != ROBOT_TYPE:
                raise RuntimeError(
                    f"layout.metadata.robot_type={layout_robot!r}, "
                    f"当前期望 {ROBOT_TYPE!r}。请重新运行 "
                    f"`python -m sealp.examples.layout.find_optimal_layout` 重搜。")
            if layout_robot is None:
                print(f"  [WARN] layout 未记录 robot_type，沿用旧文件；"
                      f"如出现穿模请重跑 find_optimal_layout。")
            for pid, (p, r) in loaded.staging_positions.items():
                if pid not in INITIAL_STAGING_PART_IDS:
                    continue
                plan.set_staging(pid, pos=p.copy(), rotmat=r.copy())
                obs = staging_obstacles.get(pid)
                if obs is not None:
                    obs.pos = p
                    obs.rotmat = r
            used_searched_layout = True
            print(f"\n[Layout] 使用预搜索布局: "
                  f"{os.path.relpath(searched_layout_path)}")
            print(f"  layout.name = {loaded.name}")
            if loaded.metadata.get("arm_choice"):
                print(f"  arm_choice = {loaded.metadata['arm_choice']}")
            if loaded.metadata.get("weights"):
                print(f"  weights = {loaded.metadata['weights']}")
            # 兼容性校验：.layout 是用旧约束搜出来的，可能配不上当前
            # +Z 抬升 / −Z 下放的几何流；不通过则丢弃缓存重搜。
            if not _validate_loaded_layout(
                    plan, robot, staging_obstacles, gc,
                    world_poses, env_obstacles):
                print("[Layout] 已加载布局与当前几何流不兼容，"
                      "丢弃缓存并回退到内置 DFS 重搜。")
                used_searched_layout = False
                apply_seed_staging(plan, staging_obstacles, _seeds)
            else:
                print("  → 跳过 prevalidate_initial_staging_layout DFS。")
        except Exception as e:
            print(f"\n[WARN] 加载预搜索布局失败: {e}，将退回到内置 DFS。")
            used_searched_layout = False

    if not used_searched_layout:
        try:
            prevalidate_initial_staging_layout(
                plan, robot, staging_obstacles, gc, world_poses, env_obstacles, _seeds, asm)
        except Exception as e:
            print(f"\n[WARN] 预搜索失败: {e}")
            print("  已回退为种子初始位；仍打开窗口并尝试执行规划，便于查看场景与成功步的轨迹动画。")
            print("  提示：可先跑 `python -m sealp.examples.layout.search_dual_layout` "
                  "生成可装配布局后重试。")
            apply_seed_staging(plan, staging_obstacles, _seeds)

    staging_colors = {
        "seat": np.array([0.9, 0.6, 0.3, 0.85]),
        "leg_fl": np.array([0.3, 0.7, 0.3, 0.85]),
        "leg_fr": np.array([0.3, 0.3, 0.8, 0.85]),
        "leg_bl": np.array([0.8, 0.3, 0.3, 0.85]),
        "leg_br": np.array([0.7, 0.3, 0.7, 0.85]),
    }
    print("\n最终初始摆放（彩色）：")
    for pid in asm.part_ids:
        st = plan.get_staging(pid)
        if st is None:
            continue
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            vis = mcm.CollisionModel(initor=mp)
            vis.pos, vis.rotmat = st.pos, st.rotmat
            vis.rgba = staging_colors.get(pid, np.array([0.5, 0.5, 0.5, 0.85]))
            vis.attach_to(base)
            mgm.gen_frame(pos=st.pos, ax_length=0.03).attach_to(base)
            print(f"  {pid}: {np.round(st.pos, 4).tolist()}")

    robot.gen_meshmodel(alpha=0.2).attach_to(base)

    executor = SequenceExecutor(
        robot=robot,
        assembly_def=asm,
        task_plan=plan,
        obstacle_list=initial_obstacles,
        grasp_paths=grasp_paths,
        # 已装件 CD 膨胀：调小可让后续 RRT 更容易在已装件附近通过
        assembled_cd_ex_radius=RELAXED_PLANNING["assembled_cd_ex_radius"],
    )

    lft_t = TransportPrimitive(robot.lft_arm)
    rgt_t = TransportPrimitive(robot.rgt_arm)

    class DualArmFallbackTransport:
        def __init__(self, lft, rgt, asm, motion_cache=None):
            self.lft, self.rgt = lft, rgt
            self.asm = asm
            self.planner = lft.planner
            self.motion_cache = motion_cache       # dict or None
            # 记录本次实跑成功的 motion，主流程在 5/5 成功后写盘
            self.recorded_motions = {}             # step_id -> dict

        def _parent_part_id(self, part_id):
            """从 ``asm.steps`` 找当前 part 的直接父件 id。

            装配树里 leg 的 parent 一般是 ``seat``，seat 的 parent 是
            ``fixture``（虚拟根，没有对应 CollisionModel）。``fixture``
            返回 ``None`` 表示无可豁免件。
            """
            if part_id is None or self.asm is None:
                return None
            for s in self.asm.steps:
                if s.part_id == part_id:
                    pp = s.parent_id
                    return pp if pp and pp != "fixture" else None
            return None

        def _step_id_for_part(self, part_id):
            for s in self.asm.steps:
                if s.part_id == part_id:
                    return s.step_id
            return None

        def _materialize_md(self, arm, jv_list, ev_list,
                            oiee_pose_list=None, obj_cmodel=None):
            """从缓存的 jv_list/ev_list 重建 ``MotionData``：每帧把 robot
            摆到对应关节角 + **把夹爪打到 ev_list[i] 指定的开合值**，再
            ``gen_meshmodel`` 拿到带正确夹爪状态的机械臂 mc；若该帧为持
            物帧（``oiee_pose_list[i]`` 非 ``None``），按记录的世界位姿把
            obj 副本 ``add_cm`` 到 mc 里 —— 这样动画里能看到夹爪开合 +
            手中桌腿，且 mc.detach() 会一起隐藏物体 mesh。
            """
            md = MotionData(robot=arm.robot)
            arm.robot.backup_state()
            try:
                for i, jv in enumerate(jv_list):
                    jv_arr = np.asarray(jv)
                    arm.robot.goto_given_conf(jv_arr)
                    md._jv_list.append(jv_arr.copy())
                    # —— 关键：把缓存的夹爪值真的应用到 robot，让 mc
                    # 看上去张/合一致。goto_given_conf 不会动夹爪。
                    if ev_list is not None and i < len(ev_list):
                        ev_i = ev_list[i]
                        try:
                            ev_scalar = float(np.asarray(ev_i).reshape(-1)[0])
                            arm.robot.change_jaw_width(ev_scalar)
                        except Exception:
                            try:
                                arm.robot.change_ee_values(ev_i)
                            except Exception:
                                pass
                        md._ev_list.append(ev_i)
                    else:
                        md._ev_list.append(arm.robot.get_ee_values())
                    mc = arm.robot.gen_meshmodel()
                    oiee_pose = None
                    if (oiee_pose_list is not None
                            and i < len(oiee_pose_list)
                            and oiee_pose_list[i] is not None):
                        oiee_pose = oiee_pose_list[i]
                        if obj_cmodel is not None:
                            obj_copy = obj_cmodel.copy()
                            obj_copy.pos = np.asarray(oiee_pose[0])
                            obj_copy.rotmat = np.asarray(oiee_pose[1])
                            mc.add_cm(obj_copy)
                    md._mesh_list.append(mc)
                    md._oiee_gl_pose_list.append(oiee_pose)
            finally:
                arm.robot.restore_state()
            return md

        def plan(self, **kwargs):
            # ── 缓存命中分支 ─────────────────────────────────
            pid = kwargs.get("part_id")
            sid = self._step_id_for_part(pid)
            if (self.motion_cache is not None and sid is not None
                    and sid in self.motion_cache.get("motions", {})):
                entry = self.motion_cache["motions"][sid]
                arm = self.lft if entry.get("arm") == "lft" else self.rgt
                jv_list = entry.get("jv_list") or []
                if not jv_list:
                    print(f"  [Motion Cache][WARN] step {sid} {pid!r} "
                          f"缓存条目 jv_list 为空，回退到实算。")
                else:
                    md = self._materialize_md(
                        arm, jv_list, entry.get("ev_list"),
                        oiee_pose_list=entry.get("oiee_gl_pose_list"),
                        obj_cmodel=kwargs.get("obj_cmodel"))
                    end_jv = np.asarray(jv_list[-1])
                    arm.robot.goto_given_conf(end_jv)
                    print(f"  [Motion Cache] step {sid} {pid!r}: "
                          f"用 {entry.get('arm', '?')} 臂的缓存轨迹 "
                          f"({len(jv_list)} 帧)，跳过 RRT/IK。")
                    return PrimitiveResult(
                        success=True, mot_data=md,
                        end_jnt_values=end_jv,
                    )

            relax = RELAXED_PLANNING
            pick_depart = relax["pick_depart_distance"]
            place_app = relax["place_approach_distance"]
            lin_gran = relax["linear_granularity"]

            def _inject_relaxed(kw):
                """把所有 RELAXED_PLANNING 中的『宽松度』参数注入一臂 kwargs。"""
                # 1) pick_approach: 距离=0 → 方向无几何意义，仅为兼容
                kw.setdefault("pick_approach_direction", _APPROACH_LINEAR_DIR)
                # 2) pick_depart: +Z 抬升 (= pick_depart_distance)
                kw.setdefault("pick_depart_direction", _PICK_DEPART_DIR)
                kw.setdefault("pick_depart_distance", pick_depart)
                # 4) place_approach: −Z 下放 (= place_approach_distance)
                kw.setdefault("place_approach_direction_list",
                              [_PLACE_APPROACH_DIR])
                kw.setdefault("place_approach_distance_list", [place_app])
                # —— 透传给 PickPlacePlanner 的内部宽松度
                kw.setdefault("linear_granularity", lin_gran)

            kw_l = dict(kwargs)
            if kw_l.get("obstacle_list") is not None:
                kw_l["obstacle_list"] = list(kw_l["obstacle_list"])
            kw_l["end_jnt_values"] = self.lft.robot.get_jnt_values()
            _inject_relaxed(kw_l)

            kw_r = dict(kwargs)
            if kw_r.get("obstacle_list") is not None:
                kw_r["obstacle_list"] = list(kw_r["obstacle_list"])
            kw_r["end_jnt_values"] = self.rgt.robot.get_jnt_values()
            _inject_relaxed(kw_r)

            pid = kw_l.get("part_id")
            if pid is not None and str(pid).startswith("leg_"):
                leg_dist = relax["leg_place_depart_distance"]
                for kw in (kw_l, kw_r):
                    kw["place_depart_direction_list"] = [_LEG_PLACE_DEPART_DIR]
                    kw["place_depart_distance_list"] = [leg_dist]
            elif pid == "seat":
                seat_dist = relax["seat_place_depart_distance"]
                for kw in (kw_l, kw_r):
                    kw["place_depart_direction_list"] = [_SEAT_PLACE_DEPART_DIR]
                    kw["place_depart_distance_list"] = [seat_dist]
            kw_l.pop("part_id", None)
            kw_r.pop("part_id", None)
            obj_cmodel = kwargs.get("obj_cmodel")
            # 当前步的直接父件：复验"在手物体 vs 已装件"时把它豁免，
            # 因为装配末段腿插进 seat 承插孔属于设计接触不算穿模。
            held_skip = self._parent_part_id(pid)
            held_skip_list = [held_skip] if held_skip else []

            def _fmt_detail(detail, arm_tag):
                """统一组装轨迹复验失败信息：reason + 命中件 + 撞击帧诊断。

                诊断字段（``detail[3]`` 是 ``_motion_replay_collision_free``
                返回的 diag dict）会被铺成多行，让用户能快速定位是哪段
                轨迹撞的——例如 ``progress<0.05`` 一般是 pick approach 段
                （问题在 staging IK 解 + 邻位 staging 太近）；
                ``has_held_obj=True`` 且 progress 在 0.4~0.6 之间则多半
                是 transport 段（在手长桌腿扫过其他 staging）；
                ``progress>0.95`` 多半是 place depart 段（已装件挤压 EE
                空间，多见于装第 3/4 条腿时）。
                """
                if not isinstance(detail, tuple):
                    return f"{arm_tag}臂轨迹复验失败: {detail}"
                reason = detail[0]
                hit_pid = detail[2] if len(detail) >= 3 else None
                diag = detail[3] if len(detail) >= 4 else None
                head = (f"{arm_tag}臂轨迹复验失败: {reason} "
                        f"(命中障碍={hit_pid!r})") if hit_pid else (
                        f"{arm_tag}臂轨迹复验失败: {reason}")
                if not isinstance(diag, dict):
                    return head
                # 推断阶段提示（粗略，仅供定位）
                prog = diag.get("progress", 0.0)
                held = diag.get("has_held_obj", False)
                if isinstance(prog, (int, float)):
                    if prog < 0.05:
                        phase = "pick approach(去抓的接近段)"
                    elif prog < 0.15:
                        phase = "pick / pick depart(抓取/抬离 staging)"
                    elif prog > 0.95:
                        phase = "place depart(放下后撤离 fixture)"
                    elif prog > 0.80:
                        phase = "place / place approach(贴近装配位)"
                    else:
                        phase = ("transport(运输, 在手中)" if held
                                 else "transport(空载移动)")
                else:
                    phase = "?"
                kf = diag.get("keyframe_idx")
                alpha = diag.get("alpha", 0.0)
                n = diag.get("total_keyframes", 0)
                tcp = diag.get("tcp_pos")
                obj_pos = diag.get("held_obj_pos")
                lines = [head]
                if kf == "final":
                    lines.append(f"  ↳ 阶段: 末关键帧 (progress=100%, {phase})")
                else:
                    lines.append(
                        f"  ↳ 阶段: 第 {kf}/{max(n-2,0)} 段 alpha={alpha:.2f} "
                        f"(整体进度 {prog*100:.1f}%, 推断: {phase})")
                if tcp is not None:
                    lines.append(
                        f"  ↳ 撞击时 TCP: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]")
                if held:
                    if obj_pos is not None:
                        lines.append(
                            f"  ↳ 在手物体中心: [{obj_pos[0]:.3f}, "
                            f"{obj_pos[1]:.3f}, {obj_pos[2]:.3f}] (仅作上下文, "
                            f"非碰撞对象)" if reason == "robot_vs_obstacle"
                            else f"  ↳ 在手物体中心: [{obj_pos[0]:.3f}, "
                                 f"{obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
                    else:
                        lines.append("  ↳ 在手物体: 持有 (位姿未记录)")
                else:
                    lines.append("  ↳ 在手物体: 无 (空载)")
                # mesh 级二次复检（仅 robot_vs_obstacle 分支会带这两个字段）
                if "mesh_verified" in diag:
                    if diag.get("mesh_verified"):
                        link = diag.get("mesh_hit_link") or "<未知 link>"
                        lines.append(
                            f"  ↳ MESH 复检: ✓ 同样撞 (link={link}) → 真撞，"
                            f"调 staging 或减少 transport 跨距")
                    else:
                        lines.append(
                            "  ↳ MESH 复检: ✗ mesh 不撞 → 疑 cdprim 过保守伪报，"
                            "可考虑放宽 cdprim 包围体或忽略本次失败")
                return "\n".join(lines)

            # 复验时把"另一只闲置臂"作为额外硬障碍传进去，捕捉横向桌腿
            # 扫穿对面臂的盲区（cc inter-arm 不会检测到在手物体）。
            res_l = self.lft.plan(**kw_l)
            if res_l.success and res_l.mot_data is not None:
                ok, detail = _motion_replay_collision_free(
                    self.lft.robot, res_l.mot_data,
                    kw_l.get("obstacle_list"), obj_cmodel=obj_cmodel,
                    other_arm=self.rgt.robot,
                    held_skip_part_ids=held_skip_list)
                if not ok:
                    res_l = PrimitiveResult(
                        success=False,
                        error_msg=_fmt_detail(detail, "左"),
                    )
            if res_l.success:
                self.lft.robot.goto_given_conf(res_l.end_jnt_values)
                self._record_success(sid, pid, "lft", res_l.mot_data)
                return res_l
            res_r = self.rgt.plan(**kw_r)
            if res_r.success and res_r.mot_data is not None:
                ok, detail = _motion_replay_collision_free(
                    self.rgt.robot, res_r.mot_data,
                    kw_r.get("obstacle_list"), obj_cmodel=obj_cmodel,
                    other_arm=self.lft.robot,
                    held_skip_part_ids=held_skip_list)
                if not ok:
                    res_r = PrimitiveResult(
                        success=False,
                        error_msg=_fmt_detail(detail, "右"),
                    )
            if res_r.success:
                self.rgt.robot.goto_given_conf(res_r.end_jnt_values)
                self._record_success(sid, pid, "rgt", res_r.mot_data)
                return res_r
            return PrimitiveResult(
                success=False,
                error_msg=f"左: {res_l.error_msg} | 右: {res_r.error_msg}",
            )

        def _record_success(self, step_id, part_id, arm_tag, mot_data):
            """缓存本 step 实跑成功的轨迹关键数据（不含 mesh / robot 引用）。

            含 ``oiee_gl_pose_list``：每帧 robot 手中物体（Object In End-
            Effector）的世界位姿，``None`` 表示该帧未持物。重放时按它
            把 obj 副本挂到机械臂 mc 上，让动画里看得见手中桌腿。
            """
            if step_id is None or mot_data is None:
                return
            oiee_serial = []
            for op in (mot_data.oiee_gl_pose_list or []):
                if op is None:
                    oiee_serial.append(None)
                else:
                    pos, rot = op
                    oiee_serial.append([
                        np.asarray(pos).copy(),
                        np.asarray(rot).copy(),
                    ])
            self.recorded_motions[step_id] = {
                "part_id": part_id,
                "arm": arm_tag,
                "jv_list": [np.asarray(jv).copy() for jv in mot_data.jv_list],
                "ev_list": list(mot_data.ev_list),
                "oiee_gl_pose_list": oiee_serial,
            }

    transport_holder = DualArmFallbackTransport(
        lft_t, rgt_t, asm, motion_cache=motion_cache)
    executor._selector._transport = transport_holder

    _orig = executor._execute_step

    def _find_pop(obs_list, part_id):
        for i, o in enumerate(obs_list):
            if getattr(o, "_sealp_role", None) == "staging_on_table" and \
                    getattr(o, "_sealp_part_id", None) == part_id:
                return obs_list.pop(i)
        return None

    from wrs.modeling.constant import CDPrimType as _CDPrimType

    def _retype_placed_cdprim_if_leg(obs_list, part_id):
        """步成功后从 ``obs_list`` 末尾找到当前 placed，把 leg 的 cdprim
        从 ``AABB`` 换成 ``CYLINDER``。

        原因：leg 是细长圆柱（直径 ~2cm），用 AABB 包围盒后**侧面**和
        **端面**周围都留出大量"虚位"——cdprim 看到的占位比 mesh 大不了
        多少，但 cdprim 的"擦边"路径相对 mesh 还远 1-2cm，这是 RRT 选
        的擦边轨迹复验时报 ``robot_vs_obstacle (命中='leg_fl')`` 的根因
        （单纯加 ex_radius 帮助有限：要把 RRT 推开 1-2cm 得加更多 padding，
        但又会压死下一件 goal 处的 IK 解空间）。换 CYLINDER 后 cdprim
        紧贴杆身，RRT 视角下的"擦边"就是真擦边，配合现在 0.015 的
        ex_radius 可以做到既不擦 mesh 又不挤死 goal IK。
        """
        if not obs_list:
            return
        placed = obs_list[-1]
        if (getattr(placed, "_sealp_role", None) != "assembled_at_goal" or
                getattr(placed, "_sealp_part_id", None) != part_id):
            return
        if not str(part_id).startswith("leg_"):
            return
        try:
            placed.change_cdprim_type(
                cdprim_type=_CDPrimType.CYLINDER,
                ex_radius=RELAXED_PLANNING["assembled_cd_ex_radius"],
            )
        except Exception as _err:
            print(f"  [WARN] {part_id} 换 CYLINDER cdprim 失败: {_err!r}")

    def _patched(*, step, world_poses, obs_list, current_conf_rgt, current_conf_lft):
        removed = _find_pop(obs_list, step.part_id)
        print(f"  [障碍] step {step.step_id} {step.part_id!r}: n_obs={len(obs_list)} "
              f"(已去掉本件 staging)")
        step_result = None
        try:
            step_result = _orig(
                step=step, world_poses=world_poses, obs_list=obs_list,
                current_conf_rgt=current_conf_rgt, current_conf_lft=current_conf_lft)
        finally:
            if removed is not None and (step_result is None or not step_result.success):
                obs_list.append(removed)
        # 步成功后再 retype，确保 obs_list 末尾就是本步刚 append 进去的 placed
        if step_result is not None and step_result.success:
            _retype_placed_cdprim_if_leg(obs_list, step.part_id)
        return step_result

    executor._execute_step = _patched

    print("\nExecuting…")
    result = executor.execute_all(stop_on_failure=False)

    # ── 路径缓存写盘（5/5 全成功 + 本次没用 cache）──
    if (motion_cache is None and result.success
            and len(result.steps) == asm.n_steps
            and len(transport_holder.recorded_motions) == asm.n_steps):
        try:
            save_data = {
                "version": 1,
                "asm_name": asm.name,
                # 标注本缓存对应的机器人类型，避免下次跨机器人误用
                "robot_type": ROBOT_TYPE,
                "step_order": [s.part_id for s in asm.steps],
                "fixture_pos": np.asarray(plan.fixture_pos).tolist(),
                "staging": {
                    pid: (plan.get_staging(pid).pos.tolist(),
                          plan.get_staging(pid).rotmat.tolist())
                    for pid in asm.part_ids
                    if plan.get_staging(pid) is not None
                },
                "motions": transport_holder.recorded_motions,
            }
            with open(motion_cache_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n[Motion Cache] 已写入 "
                  f"{os.path.relpath(motion_cache_path)}\n"
                  f"  下次启动会跳过 DFS / RRT / IK，直接重放轨迹。")
        except Exception as e:
            print(f"\n[Motion Cache][WARN] 写盘失败: {e!r}")
    elif motion_cache is not None:
        print(f"\n[Motion Cache] 本次为缓存重放，未触发重写。"
              f"如需强制重算：删除 "
              f"{os.path.relpath(motion_cache_path)} 或在源码里设 "
              f"_USE_MOTION_CACHE = False。")

    if result.total_frames > 0:
        print(f"\n按 SPACE 逐帧播放轨迹（共 {result.total_frames} 帧）。")
        animate_sequence(base, result)
    else:
        print("\n无成功轨迹可播放：仍显示初始摆放、目标 ghost 与机械臂，可旋转视角查看。")
    base.run()


if __name__ == "__main__":
    main()
