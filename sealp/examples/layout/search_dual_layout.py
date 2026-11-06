"""
Search Dual-Arm Feasible Staging Layout
=========================================

为某个装配任务 (``AssemblyTaskSpec``) 一键生成「所有零件均可被装配」的
初始 staging 布局。整张桌面候选网格直接从 SEALP 配置 YAML 中指定的桌面
障碍 (默认 ``work_table``) 自动生成 —— 不需要再手填 ``STAGING_ZONES``。

设计要点
--------
1. **装配解耦**：用 ``AssemblyTaskSpec`` 数据类描述「一个装配任务」
   所需的全部参数（asmdef、抓取库、零件、种子、fixture、机器人 base、
   桌面来源等）。换装配 = 新建一个 spec 实例，无需改搜索代码。
2. **范围解放**：候选网格通过 ``make_grid_zone_from_box`` 直接由
   ``sample_config.yaml`` 中 ``work_table`` 这条 box 障碍的几何参数
   生成 —— 整张桌面都可以摆，分辨率 / 留白可调。
3. **统一入口**：``run_search(task_spec)`` 完成「加载 asm + env + robot
   + grasp → DFS 搜索 → 保存 .layout」的全流程。

输出文件位置::

    sealp/examples/layout/_output/<spec.output_layout_name>.layout

后续 ``eval_dual_layout.py`` / ``dual_sequence_execution.py`` 会自动加载
默认输出路径（``dual_yuanchair_searched.layout``），评估时全绿球，序列
执行直接进动画。

Usage::

    python -m sealp.examples.layout.search_dual_layout                # 跑 YuanChair
    python -m sealp.examples.layout.search_dual_layout --resolution 0.04
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef
from sealp.colliders import StaticEnvironment
from sealp.config import load_config
from sealp.layout import (
    search_dual_feasible_layout,
    make_grid_zone_from_box,
    find_obstacle_def,
)

from sealp.examples.layout.eval_dual_layout import (
    STAGING_SEEDS,
    FIXTURE_POS, FIXTURE_ROTMAT,
    ROBOT_BASE_POS, ROBOT_BASE_ROTMAT,
    load_grasp_cache, model_alias_for_part,
)

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda


# ══════════════════════════════════════════════════════════════
#  装配任务规范（一处声明 → 处处复用）
# ══════════════════════════════════════════════════════════════
@dataclass
class AssemblyTaskSpec:
    """描述一个「双臂装配任务」所需的全部参数。

    新装配品 → 新建一个 ``AssemblyTaskSpec`` 实例并传给 ``run_search``，
    不需要改搜索算法 / 流程代码。
    """
    name: str

    # ── 装配定义 ─────────────────────────────────────────────
    asmdef_path: str

    # ── 抓取库（model alias → grasp pickle 绝对路径） ───────
    grasp_pickles: Dict[str, str]

    # ── 装配品本身的零件 / 初始种子 ──────────────────────────
    part_ids: Tuple[str, ...]
    staging_seeds: Dict[str, np.ndarray]

    # ── 工作站 / 机器人位姿 ──────────────────────────────────
    fixture_pos: np.ndarray
    fixture_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    robot_base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    robot_base_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))

    # ── 候选生成：从配置文件桌面 box 自动出网格 ──────────────
    config_yaml_path: str = ""
    table_obstacle_name: str = "work_table"
    table_resolution: float = 0.01   # 5 mm
    table_margin: float = 0.05
    table_z_override: Optional[float] = 0.0  # 强制候选 z 值；None 用 box 顶面

    # ── 候选数上限（保护 DFS 性能；5mm 网格满桌≈3.4万格，DFS 必爆）──
    #   按距种子距离取最近的 N 个；典型值 300~500，覆盖种子周围 8~12 cm。
    #   设为 None 或 0 = 不限制（仅在 resolution 较大时合理，否则 DFS 卡死）。
    max_candidates_per_part: Optional[int] = 1000

    # ── 模型别名映射；默认从 asm 取 part.model ──────────────
    model_alias_fn: Optional[Callable[[str], str]] = None

    # ── 输出 .layout 文件名（自动加 .layout 后缀，写入 _output/）─
    output_layout_name: str = "dual_searched"

    # ── 双臂搜索优先级 ─────────────────────────────────────
    arm_priority: Tuple[str, str] = ("lft", "rgt")

    # ── 几何流可达性（与 dual_sequence_execution.RELAXED_PLANNING 对齐） ──
    #   * 搜索阶段除两端 IK + 共同抓取检查外，会额外检查 pick_depart 沿
    #     pick_depart_dir 抬升 pick_depart_dist 米、place_approach 沿
    #     place_approach_dir 下放 place_approach_dist 米的中间各点 IK 是否
    #     可解；都通过的位置才会进入 .layout，避免执行阶段 IK 失败。
    #   * 设为 None / 0 则跳过中间段检查（退化为旧行为）。
    pick_depart_dir: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0]))   # +Z 抬升
    pick_depart_dist: Optional[float] = 0.20
    place_approach_dir: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0]))  # −Z 下放
    place_approach_dist: Optional[float] = 0.10


# ══════════════════════════════════════════════════════════════
#  YuanChair 任务实例（内置示例）
# ══════════════════════════════════════════════════════════════
def _here(*parts) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *parts))


YUANCHAIR_TASK = AssemblyTaskSpec(
    name="yuanchair",
    asmdef_path=_here("..", "..", "assembly_sequence",
                      "_demo_output", "yuanchair.asmdef"),
    grasp_pickles={
        "leg_model":  _here("..", "grasp", "_output",
                            "demo_yuanchair-part2_grasps.pickle"),
        "seat_model": _here("..", "grasp", "_output",
                            "demo_yuanchair-part1_grasps.pickle"),
    },
    part_ids=("seat", "leg_fl", "leg_bl", "leg_fr", "leg_br"),
    staging_seeds=STAGING_SEEDS,
    fixture_pos=FIXTURE_POS,
    fixture_rotmat=FIXTURE_ROTMAT,
    robot_base_pos=ROBOT_BASE_POS,
    robot_base_rotmat=ROBOT_BASE_ROTMAT,
    config_yaml_path=_here("..", "..", "config", "sample_config.yaml"),
    table_obstacle_name="work_table",
    table_resolution=0.005,   # 5 mm
    table_margin=0.05,
    max_candidates_per_part=400,
    table_z_override=0.0,
    model_alias_fn=model_alias_for_part,
    output_layout_name="dual_yuanchair_searched",
    arm_priority=("lft", "rgt"),
)


# ══════════════════════════════════════════════════════════════
#  通用搜索流程
# ══════════════════════════════════════════════════════════════
def _resolve_table_zone(task: AssemblyTaskSpec) -> Tuple[Dict, list]:
    """从 task.config_yaml_path 提取桌面 box → 生成候选 zone + env_obstacles。"""
    env_obs = []
    if not task.config_yaml_path or not os.path.isfile(task.config_yaml_path):
        print(f"[WARN] 未提供有效 config_yaml_path，候选 zone 将仅含 seed。")
        return {}, env_obs

    cfg = load_config(task.config_yaml_path)
    env = StaticEnvironment(
        obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    env_obs = list(env.obstacle_list)
    print(f"[环境] 已加载 {len(env_obs)} 个静态障碍。")

    table_def = find_obstacle_def(cfg.obstacle_defs, task.table_obstacle_name)
    if table_def is None or table_def.get("type") != "box":
        print(f"[WARN] 配置中未找到名为 {task.table_obstacle_name!r} 的 box "
              f"障碍，候选 zone 将仅含 seed。")
        return {}, env_obs

    zone, meta = make_grid_zone_from_box(
        box_pos=table_def["pos"],
        box_extent=table_def["extent"],
        resolution=task.table_resolution,
        margin=task.table_margin,
        return_meta=True,
    )
    cx, cy = meta["box_center_xy"]
    lx, ly = meta["box_extent_xy"]
    px = meta["physical_bounds_x"]
    py = meta["physical_bounds_y"]
    sx = meta["sample_bounds_x"]
    sy = meta["sample_bounds_y"]
    print(f"[桌面] {task.table_obstacle_name!r}  "
          f"center=({cx:+.3f}, {cy:+.3f})  extent=({lx:.3f}, {ly:.3f})")
    print(f"       物理边界  X ∈ [{px[0]:+.3f}, {px[1]:+.3f}]  "
          f"Y ∈ [{py[0]:+.3f}, {py[1]:+.3f}]")
    print(f"       留白边界  X ∈ [{sx[0]:+.3f}, {sx[1]:+.3f}]  "
          f"Y ∈ [{sy[0]:+.3f}, {sy[1]:+.3f}]   "
          f"(margin={meta['margin']}m)")
    print(f"       候选网格  {meta['n_x']} × {meta['n_y']} = "
          f"{meta['n_total']}   (resolution={meta['resolution']}m)")
    return zone, env_obs


def run_search(task: AssemblyTaskSpec,
               verbose: bool = True) -> str:
    """执行一次完整的双臂可装配 staging 搜索；返回保存的 .layout 路径。"""
    # ── asm ──────────────────────────────────────────────
    if not os.path.isfile(task.asmdef_path):
        raise FileNotFoundError(
            f"asmdef not found: {task.asmdef_path}\n"
            f"提示：先生成装配定义，例如 yuanchair: "
            f"`python -m sealp.assembly_sequence.gen_yuanchair_asmdef`")
    asm = AssemblyDef.load(task.asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")

    # ── env + 全桌面候选 zone ────────────────────────────
    table_zone, env_obs = _resolve_table_zone(task)

    # 修改 zone 的 z（默认 0；用户可强制覆盖）
    if table_zone and task.table_z_override is not None:
        # 通过种子的 z 表达式由 generate_staging_candidates 控制；
        # 此处把所有 staging_seeds 的 z 也对齐到 override，避免不一致
        for pid, seed in task.staging_seeds.items():
            if seed.shape[0] >= 3:
                seed[2] = task.table_z_override

    # 所有 part 共享同一个全桌面 zone（不再每件零件定死小窗口）
    candidate_zones = {pid: table_zone for pid in task.part_ids} \
        if table_zone else None

    # ── 双臂机器人 ───────────────────────────────────────
    robot = pda.DualPantheraHTNoBody(
        pos=task.robot_base_pos,
        rotmat=task.robot_base_rotmat,
        enable_cc=True)
    home = np.zeros(6)
    robot.lft_arm.goto_given_conf(home)
    robot.rgt_arm.goto_given_conf(home)

    # ── 抓取库 ───────────────────────────────────────────
    print("\nLoading grasp cache...")
    grasp_cache = load_grasp_cache(task.grasp_pickles)

    # ── 几何流概览 ───────────────────────────────────────
    if (task.pick_depart_dist and task.pick_depart_dist > 0) or \
       (task.place_approach_dist and task.place_approach_dist > 0):
        pdv = np.asarray(task.pick_depart_dir) if task.pick_depart_dir is not None else None
        pad = np.asarray(task.place_approach_dir) if task.place_approach_dir is not None else None
        print(f"\n[几何流] 搜索时将额外检查中间段 IK 可达性：")
        if task.pick_depart_dist:
            print(f"   pick_depart : 沿 {pdv.tolist() if pdv is not None else 'None'} "
                  f"抬升 {task.pick_depart_dist:.3f} m")
        if task.place_approach_dist:
            print(f"   place_appr  : 沿 {pad.tolist() if pad is not None else 'None'} "
                  f"下放 {task.place_approach_dist:.3f} m")
    else:
        print("\n[几何流] 仅检查两端 IK + 共同抓取（未启用中间段过滤）。")

    # ── 搜索 ─────────────────────────────────────────────
    layout = search_dual_feasible_layout(
        assembly_def=asm,
        robot_dual=robot,
        grasp_cache=grasp_cache,
        fixture_pos=task.fixture_pos,
        fixture_rotmat=task.fixture_rotmat,
        staging_seeds=task.staging_seeds,
        candidate_zones=candidate_zones,
        cross_arm_zones=None,        # 全桌面候选已包含跨臂区
        part_ids=task.part_ids,
        env_obstacles=env_obs,
        model_alias_fn=task.model_alias_fn,
        robot_base_pos=task.robot_base_pos,
        robot_base_rotmat=task.robot_base_rotmat,
        layout_name=task.output_layout_name,
        arm_priority=task.arm_priority,
        pick_depart_dir=task.pick_depart_dir,
        pick_depart_dist=task.pick_depart_dist,
        place_approach_dir=task.place_approach_dir,
        place_approach_dist=task.place_approach_dist,
        max_candidates_per_part=task.max_candidates_per_part,
        verbose=verbose,
    )

    # ── 保存 ─────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task.output_layout_name}.layout")
    layout.save(out_path)
    print(f"\n[OK] Layout 已保存: {out_path}")
    print(f"     name = {layout.name}")
    print(f"     arm_choice = {layout.metadata.get('arm_choice', {})}")
    return out_path


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def _build_argparser():
    p = argparse.ArgumentParser(
        description="双臂可装配 staging 布局搜索",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", default="yuanchair",
                   choices=["yuanchair"],
                   help="选择内置任务规范")
    p.add_argument("--resolution", type=float, default=None,
                   help="覆盖桌面候选网格分辨率 (m)")
    p.add_argument("--margin", type=float, default=None,
                   help="覆盖桌面留白 (m)")
    p.add_argument("--pick-depart-dist", type=float, default=None,
                   help="覆盖 pick_depart 中间段检查距离 (m)；0 = 关闭该项检查")
    p.add_argument("--place-approach-dist", type=float, default=None,
                   help="覆盖 place_approach 中间段检查距离 (m)；0 = 关闭该项检查")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="每件零件最多保留的候选数；0 / 负数 = 不限制 (危险)")
    p.add_argument("--quiet", action="store_true", help="减少日志输出")
    return p


_TASK_REGISTRY: Dict[str, AssemblyTaskSpec] = {
    "yuanchair": YUANCHAIR_TASK,
}


def main():
    args = _build_argparser().parse_args()
    task = _TASK_REGISTRY[args.task]
    if args.resolution is not None:
        task.table_resolution = args.resolution
    if args.margin is not None:
        task.table_margin = args.margin
    if args.pick_depart_dist is not None:
        task.pick_depart_dist = args.pick_depart_dist if args.pick_depart_dist > 0 else None
    if args.place_approach_dist is not None:
        task.place_approach_dist = args.place_approach_dist if args.place_approach_dist > 0 else None
    if args.max_candidates is not None:
        task.max_candidates_per_part = args.max_candidates if args.max_candidates > 0 else None

    out_path = run_search(task, verbose=not args.quiet)

    print("\n下一步：")
    print("  python -m sealp.examples.layout.eval_dual_layout"
          "        # 评估应全部绿球")
    print("  python -m sealp.examples.motion.dual_sequence_execution"
          "  # 跳过预搜索，直接执行 + 动画")
    print(f"\n（如果想让上述两个脚本默认加载这份 layout，输出文件名需要"
          f"是 dual_yuanchair_searched.layout；当前: "
          f"{os.path.basename(out_path)}）")


if __name__ == "__main__":
    main()
