"""
Dual-Arm Layout Feasibility Evaluation — YuanChair
====================================================

按 ``sealp/examples/motion/dual_sequence_execution.py`` 的设定，对**双臂 Panthera-HT**
(``DualPantheraHTNoBody``) + YuanChair 五件套的初始物品摆放（staging）做布局可行性
评估。与 ``eval_layout.py`` 的区别：

- 机器人由单臂 ``PantheraHTSglArm`` 换为双臂 ``DualPantheraHTNoBody``。
- staging 种子、fixture 偏移、抓取缓存、静态障碍均与 dual demo 对齐：
  * STAGING_SEEDS、FIXTURE_POS_OFFSET 复用 demo 的常量。
  * grasp_paths 直接复用 ``examples/grasp/_output`` 下的 pickle。
  * 静态环境从 ``sealp/config/sample_config.yaml`` 加载。
- 可行性评估对**左臂**与**右臂**分别调用 ``check_pose_reachability``，按
  「任一臂可行 → 该步可行」聚合，并记录每只臂各自的成功抓取数与最佳操控度。
- 3D 可视化：对每个 staging / 装配位绘制并列两颗小球：
  * 左侧球 = 左臂能否完成（绿/红）
  * 右侧球 = 右臂能否完成（绿/红）

Usage::

    python -m sealp.examples.layout.eval_dual_layout

Prerequisites:
    - ``yuanchair.asmdef`` 已生成（否则会自动调用
      ``gen_yuanchair_asmdef.py`` 生成）。
    - 抓取 pickle 已存在于 ``sealp/examples/grasp/_output/``。
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wrs import wd, rm, mgm, mcm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef
from sealp.colliders import StaticEnvironment
from sealp.config import load_config
from sealp.layout import WorkspaceLayout
from sealp.layout.reachability import check_pose_reachability

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda



STAGING_SEEDS: Dict[str, np.ndarray] = {
    "seat":   np.array([0.30, -0.10, 0.00]),
    "leg_fl": np.array([0.25,  0.20, 0.00]),
    "leg_bl": np.array([0.40,  0.15, 0.00]),
    # 右臂 base 在 (0, -0.62)；leg 是 33cm 立杆，cdprim 在 y 方向膨胀
    # 较多。seed y 必须离右臂 base 至少 ~0.20m，否则 home 姿态下
    # _arms_collide_at_home 会判穿模，DFS 第一层就会被全部剔掉。
    "leg_fr": np.array([0.25, -0.85, 0.00]),
    "leg_br": np.array([0.40, -0.85, 0.00]),
}

FIXTURE_POS = np.array([0.0, -0.30, 0.0])
FIXTURE_ROTMAT = np.eye(3)

ROBOT_BASE_POS = np.zeros(3)
ROBOT_BASE_ROTMAT = np.eye(3)

MAX_GRASPS_PER_STEP = 20

# 若存在该 .layout 文件，将自动用其覆盖 STAGING_SEEDS（推荐先跑
# `python -m sealp.examples.layout.search_dual_layout` 生成）。
SEARCHED_LAYOUT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "_output", "dual_yuanchair_searched.layout"))


def model_alias_for_part(part_id: str) -> str:
    """与 dual_sequence_execution.py 一致的模型别名映射。"""
    return "seat_model" if part_id == "seat" else "leg_model"


def apply_searched_layout(layout: WorkspaceLayout,
                          searched_path: str = SEARCHED_LAYOUT_PATH) -> bool:
    """若 ``searched_path`` 存在，将其 fixture/staging 写入 ``layout``。

    Returns
    -------
    bool
        是否成功覆盖（文件不存在返回 False）。
    """
    if not os.path.isfile(searched_path):
        return False
    loaded = WorkspaceLayout.load(searched_path)
    layout.assembly_station_pos = loaded.assembly_station_pos.copy()
    layout.assembly_station_rotmat = loaded.assembly_station_rotmat.copy()
    for pid, (p, r) in loaded.staging_positions.items():
        layout.set_staging(pid, p.copy(), r.copy())
    layout.name = loaded.name
    layout.metadata = dict(loaded.metadata)
    return True


# ══════════════════════════════════════════════════════════════
#  双臂可行性数据结构
# ══════════════════════════════════════════════════════════════
@dataclass
class DualStepFeasibility:
    """单个装配步骤的双臂可行性结果。"""
    step_id: int = -1
    part_id: str = ""

    pick_lft_ok: bool = False
    pick_rgt_ok: bool = False
    place_lft_ok: bool = False
    place_rgt_ok: bool = False

    n_pick_lft: int = 0
    n_pick_rgt: int = 0
    n_place_lft: int = 0
    n_place_rgt: int = 0

    manip_pick_lft: float = 0.0
    manip_pick_rgt: float = 0.0
    manip_place_lft: float = 0.0
    manip_place_rgt: float = 0.0

    error_msg: str = ""

    @property
    def feasible_lft(self) -> bool:
        return self.pick_lft_ok and self.place_lft_ok

    @property
    def feasible_rgt(self) -> bool:
        return self.pick_rgt_ok and self.place_rgt_ok

    @property
    def feasible(self) -> bool:
        return self.feasible_lft or self.feasible_rgt

    @property
    def preferred_arm(self) -> str:
        """左臂优先、右臂兜底（与 dual demo 策略一致）。"""
        if self.feasible_lft:
            return "lft"
        if self.feasible_rgt:
            return "rgt"
        return "none"

    @property
    def best_manipulability(self) -> float:
        cands = []
        if self.feasible_lft:
            cands.append(0.5 * (self.manip_pick_lft + self.manip_place_lft))
        if self.feasible_rgt:
            cands.append(0.5 * (self.manip_pick_rgt + self.manip_place_rgt))
        return max(cands) if cands else 0.0


@dataclass
class DualFeasibilityReport:
    """整局布局的双臂可行性聚合。"""
    layout: Optional[WorkspaceLayout] = None
    steps: List[DualStepFeasibility] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_feasible(self) -> int:
        return sum(1 for s in self.steps if s.feasible)

    @property
    def feasibility_rate(self) -> float:
        return self.n_feasible / self.n_steps if self.n_steps else 0.0

    @property
    def n_lft_only(self) -> int:
        return sum(1 for s in self.steps
                   if s.feasible_lft and not s.feasible_rgt)

    @property
    def n_rgt_only(self) -> int:
        return sum(1 for s in self.steps
                   if s.feasible_rgt and not s.feasible_lft)

    @property
    def n_both_arms(self) -> int:
        return sum(1 for s in self.steps
                   if s.feasible_lft and s.feasible_rgt)

    @property
    def avg_manipulability(self) -> float:
        feasible = [s for s in self.steps if s.feasible]
        if not feasible:
            return 0.0
        return float(np.mean([s.best_manipulability for s in feasible]))

    @property
    def composite_score(self) -> float:
        return self.feasibility_rate * (1.0 + self.avg_manipulability)

    def summary(self) -> str:
        lines = [
            f"Dual-Arm Feasibility Report",
            f"  Layout: {self.layout.name if self.layout else '?'}",
            f"  Steps: {self.n_feasible}/{self.n_steps} feasible "
            f"({self.feasibility_rate:.0%})",
            f"  Both arms OK: {self.n_both_arms} | "
            f"Lft only: {self.n_lft_only} | Rgt only: {self.n_rgt_only}",
            f"  Avg manipulability (best arm): {self.avg_manipulability:.4f}",
            f"  Composite score: {self.composite_score:.4f}",
            "",
        ]
        for s in self.steps:
            tag = {"lft": "L", "rgt": "R", "none": "X"}[s.preferred_arm]
            icon = "[OK]" if s.feasible else "[FAIL]"
            lft = (f"L(pick {'Y' if s.pick_lft_ok else 'N'} "
                   f"{s.n_pick_lft}g m={s.manip_pick_lft:.3f} | "
                   f"place {'Y' if s.place_lft_ok else 'N'} "
                   f"{s.n_place_lft}g m={s.manip_place_lft:.3f})")
            rgt = (f"R(pick {'Y' if s.pick_rgt_ok else 'N'} "
                   f"{s.n_pick_rgt}g m={s.manip_pick_rgt:.3f} | "
                   f"place {'Y' if s.place_rgt_ok else 'N'} "
                   f"{s.n_place_rgt}g m={s.manip_place_rgt:.3f})")
            lines.append(
                f"  {icon}[{tag}] step {s.step_id} ({s.part_id}): "
                f"{lft}  {rgt}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  双臂布局评估
# ══════════════════════════════════════════════════════════════
def evaluate_dual_layout(
    layout: WorkspaceLayout,
    assembly_def: AssemblyDef,
    robot_dual,
    grasp_cache: Dict,
    obstacle_list: Optional[List] = None,
    max_grasps_per_step: int = MAX_GRASPS_PER_STEP,
    verbose: bool = True,
) -> DualFeasibilityReport:
    """对双臂布局做逐步可达性评估。

    Parameters
    ----------
    layout : WorkspaceLayout
        待评估布局（含 staging、robot_base、fixture）。
    assembly_def : AssemblyDef
        装配定义。
    robot_dual : DualPantheraHTNoBody
        双臂机器人实例（左右臂 base 已按 ``layout.robot_base_pos`` 摆好）。
    grasp_cache : dict
        ``{model_alias: GraspCollection}``，alias 与 ``part.model`` 对应。
    obstacle_list : list or None
        静态障碍（地面、桌面、墙等）。装配过程动态加入已装件。
    max_grasps_per_step : int
        每步最多评估的抓取数。
    verbose : bool
        逐步打印日志。
    """
    if obstacle_list is None:
        obstacle_list = []

    world_poses = assembly_def.compute_world_poses(
        fixture_pos=layout.assembly_station_pos,
        fixture_rotmat=layout.assembly_station_rotmat,
    )
    steps = assembly_def.get_execution_order()

    report = DualFeasibilityReport(layout=layout)
    dynamic_obs = list(obstacle_list)

    if verbose:
        print("=" * 60)
        print(f"Dual-Arm Layout Evaluation: {layout.name}")
        print(f"  Steps: {len(steps)} | Robot base: "
              f"{layout.robot_base_pos.tolist()} | "
              f"Fixture: {layout.assembly_station_pos.tolist()}")
        print("=" * 60)

    for i, step in enumerate(steps):
        pid = step.part_id
        sf = DualStepFeasibility(step_id=step.step_id, part_id=pid)

        if verbose:
            print(f"\n-- Step {step.step_id}: {pid!r} ({i + 1}/{len(steps)}) --")

        # ── 抓取库 ──────────────────────────────────────
        try:
            part = assembly_def.get_part(pid)
        except KeyError:
            sf.error_msg = f"Part {pid!r} not in assembly_def"
            report.steps.append(sf)
            if verbose:
                print(f"  [!] {sf.error_msg}")
            continue
        alias = part.model
        gc = grasp_cache.get(alias) or grasp_cache.get(model_alias_for_part(pid))
        if gc is None or len(gc) == 0:
            sf.error_msg = f"No grasps for model {alias!r}"
            report.steps.append(sf)
            if verbose:
                print(f"  [!] {sf.error_msg}")
            continue

        staging = layout.get_staging(pid)
        if staging is None:
            sf.error_msg = f"No staging for {pid!r}"
            report.steps.append(sf)
            if verbose:
                print(f"  [!] {sf.error_msg}")
            continue
        pick_pos, pick_rot = staging

        if pid not in world_poses:
            sf.error_msg = f"No world pose for {pid!r}"
            report.steps.append(sf)
            if verbose:
                print(f"  [!] {sf.error_msg}")
            continue
        place_pos, place_rot = world_poses[pid]

        # ── 左臂 pick / place ───────────────────────────
        pl = check_pose_reachability(
            robot=robot_dual.lft_arm,
            obj_pos=pick_pos, obj_rotmat=pick_rot,
            grasp_collection=gc, obstacle_list=dynamic_obs,
            max_grasps=max_grasps_per_step,
        )
        ll = check_pose_reachability(
            robot=robot_dual.lft_arm,
            obj_pos=place_pos, obj_rotmat=place_rot,
            grasp_collection=gc, obstacle_list=dynamic_obs,
            max_grasps=max_grasps_per_step,
        )
        sf.pick_lft_ok = pl.n_collision_free > 0
        sf.place_lft_ok = ll.n_collision_free > 0
        sf.n_pick_lft = pl.n_collision_free
        sf.n_place_lft = ll.n_collision_free
        sf.manip_pick_lft = pl.best_manipulability
        sf.manip_place_lft = ll.best_manipulability

        # ── 右臂 pick / place ───────────────────────────
        pr = check_pose_reachability(
            robot=robot_dual.rgt_arm,
            obj_pos=pick_pos, obj_rotmat=pick_rot,
            grasp_collection=gc, obstacle_list=dynamic_obs,
            max_grasps=max_grasps_per_step,
        )
        lr = check_pose_reachability(
            robot=robot_dual.rgt_arm,
            obj_pos=place_pos, obj_rotmat=place_rot,
            grasp_collection=gc, obstacle_list=dynamic_obs,
            max_grasps=max_grasps_per_step,
        )
        sf.pick_rgt_ok = pr.n_collision_free > 0
        sf.place_rgt_ok = lr.n_collision_free > 0
        sf.n_pick_rgt = pr.n_collision_free
        sf.n_place_rgt = lr.n_collision_free
        sf.manip_pick_rgt = pr.best_manipulability
        sf.manip_place_rgt = lr.best_manipulability

        if verbose:
            print(f"  L: pick {sf.n_pick_lft}g m={sf.manip_pick_lft:.3f} | "
                  f"place {sf.n_place_lft}g m={sf.manip_place_lft:.3f}")
            print(f"  R: pick {sf.n_pick_rgt}g m={sf.manip_pick_rgt:.3f} | "
                  f"place {sf.n_place_rgt}g m={sf.manip_place_rgt:.3f}")
            print(f"  → preferred arm: {sf.preferred_arm}")

        if not sf.feasible:
            reasons = []
            if not (sf.pick_lft_ok or sf.pick_rgt_ok):
                reasons.append("无臂可达 pick")
            if not (sf.place_lft_ok or sf.place_rgt_ok):
                reasons.append("无臂可达 place")
            if not reasons:
                reasons.append("单臂无法同时 pick+place")
            sf.error_msg = "; ".join(reasons)

        report.steps.append(sf)

        # ── 已成功步骤 → 加入 dynamic 障碍 ──────────────
        if sf.feasible:
            mp = assembly_def.model_path(pid)
            if os.path.isfile(mp):
                placed = mcm.CollisionModel(initor=mp)
                placed.pos = place_pos
                placed.rotmat = place_rot
                dynamic_obs.append(placed)

    if verbose:
        print("\n" + "=" * 60)
        print(report.summary())
        print("=" * 60)

    return report


# ══════════════════════════════════════════════════════════════
#  辅助：加载抓取缓存
# ══════════════════════════════════════════════════════════════
def load_grasp_cache(grasp_paths: Dict[str, str]) -> Dict:
    cache = {}
    for alias, path in grasp_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Grasp pickle not found: {path}")
        with open(path, "rb") as fh:
            cache[alias] = pickle.load(fh)
        print(f"  Loaded grasps {alias!r}: {len(cache[alias])} from "
              f"{os.path.relpath(path)}")
    return cache


# ══════════════════════════════════════════════════════════════
#  辅助：加载静态障碍
# ══════════════════════════════════════════════════════════════
def load_obstacles_from_config(config_path: str, base) -> List:
    if not os.path.isfile(config_path):
        print(f"[WARN] config 不存在: {config_path}，跳过静态障碍。")
        return []
    cfg = load_config(config_path)
    env = StaticEnvironment(
        obstacle_defs=cfg.obstacle_defs,
        base_dir=cfg.config_dir,
    )
    obs = list(env.obstacle_list)
    for o in obs:
        o.attach_to(base)
    print(f"[环境] 已加载 {len(obs)} 个静态障碍。")
    return obs


# ══════════════════════════════════════════════════════════════
#  可视化：双臂可行性指示
# ══════════════════════════════════════════════════════════════
def visualize_dual_report(base, layout: WorkspaceLayout,
                          assembly_def: AssemblyDef,
                          report: DualFeasibilityReport):
    """每个 pick / place 上方画并列两颗小球：左臂(+y) / 右臂(-y)，绿=可行 红=不可行。"""
    RADIUS = 0.012
    OK = np.array([0.15, 0.85, 0.15, 0.9])
    BAD = np.array([0.90, 0.15, 0.15, 0.9])
    Y_OFFSET = 0.02
    Z_OFFSET = 0.05

    world_poses = assembly_def.compute_world_poses(
        fixture_pos=layout.assembly_station_pos,
        fixture_rotmat=layout.assembly_station_rotmat,
    )

    for sf in report.steps:
        pid = sf.part_id

        staging = layout.get_staging(pid)
        if staging is not None:
            pick_pos, _ = staging
            for ok, dy in (
                (sf.pick_lft_ok, +Y_OFFSET),
                (sf.pick_rgt_ok, -Y_OFFSET),
            ):
                s = mcm.gen_sphere(radius=RADIUS)
                s.pos = pick_pos + np.array([0.0, dy, Z_OFFSET])
                s.rgba = OK if ok else BAD
                s.attach_to(base)

        if pid in world_poses:
            place_pos, _ = world_poses[pid]
            for ok, dy in (
                (sf.place_lft_ok, +Y_OFFSET),
                (sf.place_rgt_ok, -Y_OFFSET),
            ):
                s = mcm.gen_sphere(radius=RADIUS)
                s.pos = place_pos + np.array([0.0, dy, Z_OFFSET])
                s.rgba = OK if ok else BAD
                s.attach_to(base)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    # ------------------------------------------------------------------
    # 1. 场景
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=[1.5, -0.3, 1.2], lookat_pos=[0.3, -0.3, 0.1])
    mgm.gen_frame(ax_length=0.15).attach_to(base)

    # 地面（轻量参考），真正的桌面/墙体由 sample_config.yaml 提供
    ground = mcm.gen_box(
        xyz_lengths=rm.vec(2, 2, 0.01),
        rgb=rm.vec(0.75, 0.75, 0.75), alpha=1)
    ground.pos = np.array([0.3, -0.3, -0.02])
    ground.attach_to(base)

    # ------------------------------------------------------------------
    # 2. 静态障碍（与 dual demo 一致）
    # ------------------------------------------------------------------
    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "sample_config.yaml"))
    env_obstacles = load_obstacles_from_config(config_path, base)

    # ------------------------------------------------------------------
    # 3. 装配定义
    # ------------------------------------------------------------------
    asmdef_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "assembly_sequence", "_demo_output")
    asmdef_path = os.path.abspath(os.path.join(asmdef_dir, "yuanchair.asmdef"))
    if not os.path.isfile(asmdef_path):
        print(f"Assembly definition not found, generating...")
        from sealp.assembly_sequence.gen_yuanchair_asmdef import main as gen_asm
        gen_asm()
    asm = AssemblyDef.load(asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")

    # ------------------------------------------------------------------
    # 4. 构造双臂布局（数据与 dual_sequence_execution.py 对齐）
    # ------------------------------------------------------------------
    layout = WorkspaceLayout(
        robot_base_pos=ROBOT_BASE_POS.copy(),
        robot_base_rotmat=ROBOT_BASE_ROTMAT.copy(),
        assembly_station_pos=FIXTURE_POS.copy(),
        assembly_station_rotmat=FIXTURE_ROTMAT.copy(),
        name="dual_yuanchair_seed_layout",
    )
    for pid in ("seat", "leg_fl", "leg_bl", "leg_fr", "leg_br"):
        layout.set_staging(pid, STAGING_SEEDS[pid].copy(), np.eye(3))

    # 4b. 若存在搜索结果，覆盖 staging（→ 评估时应全部绿球）
    if apply_searched_layout(layout):
        print(f"\n[Layout] 已加载预搜索结果: "
              f"{os.path.relpath(SEARCHED_LAYOUT_PATH)}")
        print(f"  layout.name = {layout.name}")
        if layout.metadata.get("arm_choice"):
            print(f"  arm_choice = {layout.metadata['arm_choice']}")
    else:
        print(f"\n[Layout] 未找到 {os.path.basename(SEARCHED_LAYOUT_PATH)}，"
              f"使用 STAGING_SEEDS 默认布局。")
        print("  提示：先跑 `python -m sealp.examples.layout.search_dual_layout` "
              "生成可装配布局。")

    # ------------------------------------------------------------------
    # 5. 可视化 staging 实体 + 装配 ghost
    # ------------------------------------------------------------------
    staging_colors = {
        "seat":   np.array([0.9, 0.6, 0.3, 0.85]),
        "leg_fl": np.array([0.3, 0.7, 0.3, 0.85]),
        "leg_fr": np.array([0.3, 0.3, 0.8, 0.85]),
        "leg_bl": np.array([0.8, 0.3, 0.3, 0.85]),
        "leg_br": np.array([0.7, 0.3, 0.7, 0.85]),
    }
    for pid in asm.part_ids:
        st = layout.get_staging(pid)
        if st is None:
            continue
        pos, rotmat = st
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            staged = mcm.CollisionModel(initor=mp)
            staged.pos = pos
            staged.rotmat = rotmat
            staged.rgba = staging_colors.get(pid, np.array([0.5, 0.5, 0.5, 0.85]))
            staged.attach_to(base)
            mgm.gen_frame(pos=pos, ax_length=0.03).attach_to(base)

    world_poses = asm.compute_world_poses(
        fixture_pos=layout.assembly_station_pos,
        fixture_rotmat=layout.assembly_station_rotmat,
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

    # ------------------------------------------------------------------
    # 6. 双臂机器人（home conf）
    # ------------------------------------------------------------------
    robot = pda.DualPantheraHTNoBody(
        pos=layout.robot_base_pos,
        rotmat=layout.robot_base_rotmat,
        enable_cc=True,
    )
    home = np.zeros(6)
    robot.lft_arm.goto_given_conf(home)
    robot.rgt_arm.goto_given_conf(home)
    robot.gen_meshmodel(alpha=0.25).attach_to(base)

    # ------------------------------------------------------------------
    # 7. 抓取缓存（路径与 dual demo 一致）
    # ------------------------------------------------------------------
    grasp_paths = {
        "leg_model": os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "grasp", "_output",
            "demo_yuanchair-part2_grasps.pickle")),
        "seat_model": os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "grasp", "_output",
            "demo_yuanchair-part1_grasps.pickle")),
    }
    print("\nLoading grasp cache...")
    grasp_cache = load_grasp_cache(grasp_paths)

    # ------------------------------------------------------------------
    # 8. 评估双臂布局
    # ------------------------------------------------------------------
    print("\nEvaluating dual-arm layout feasibility...")
    obstacle_list = list(env_obstacles) + [ground]
    report = evaluate_dual_layout(
        layout=layout,
        assembly_def=asm,
        robot_dual=robot,
        grasp_cache=grasp_cache,
        obstacle_list=obstacle_list,
        max_grasps_per_step=MAX_GRASPS_PER_STEP,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 9. 可视化每步可行性
    # ------------------------------------------------------------------
    visualize_dual_report(base, layout, asm, report)

    # ------------------------------------------------------------------
    # 10. 保存布局
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)
    layout_path = os.path.join(out_dir, "dual_yuanchair_eval.layout")
    layout.save(layout_path)
    print(f"\nLayout saved to: {layout_path}")

    print("\n图例：每个 pick/place 上方两颗球 — 左侧(+y)=左臂、右侧(-y)=右臂；"
          "绿=该臂可达且无碰撞，红=不可行。")
    base.run()


if __name__ == "__main__":
    main()
