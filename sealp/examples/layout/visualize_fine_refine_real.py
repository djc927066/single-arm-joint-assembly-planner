# -*- coding: utf-8 -*-
"""
可视化 Phase B "由粗到细" 精修 —— 使用**真实权重 layout_score**
==============================================================

与 ``visualize_fine_refine.py`` 的区别:
    * 那个用的是合成 score(x,y) 曲面, 只讲算法过程, 数字是假的;
    * **本脚本用真实的 evaluate_layout / layout_score** —— 会真正构造搜索器
      (机器人 + 网格 + grasp + 碰撞检测), 从一个已有 ``.layout`` 出发, 对某个
      零件的 (x, y) 用与 ``find_optimal_initial_layout_tower_global.py`` 里
      ``_pattern_refine`` **完全相同** 的 ``_offsets`` + 贪心接受 + 缩步逻辑
      跑精修, 每个探测点记录真实加权分数, 最后画出轨迹与分数曲线。

因此本脚本需要能跑起整套仿真 (与真实 layout 搜索脚本相同的环境: config /
asmdef / grasp 目录 / 机器人模型 / 网格 / panda3d)。每个探测点 = 一次真实
``_evaluate_gene``, 会比较慢; 默认只精修**单个零件**、步长与脚本默认一致,
评估次数有限。

用法
----
# 用默认 asmdef/config/grasp-dir, 从 tower_global.layout 精修 post_bl:
python -m sealp.examples.layout.visualize_fine_refine_real \
    --layout sealp/examples/layout/_output/tower_global.layout --part post_bl --cdprim-type box

# 换零件 / 步长 / 加对角 / 顺带画一张低分辨率真实分数热力图 (较慢):
python -m sealp.examples.layout.visualize_fine_refine_real \
    --layout .../tower_global.layout --part post_fr \
    --steps 0.03,0.015,0.008 --rounds 2 --diagonal --heatmap 9

说明: 本脚本的搜索器构造刻意"镜像" find_optimal_initial_layout_tower_strict.py
的 main() 构造块, 以保证与真实搜索完全同口径。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import yaml

import matplotlib
import matplotlib.pyplot as plt

# 真实 layout 脚本之间用"裸模块名"互相 import, 所以必须把 layout 目录放进 sys.path。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import find_optimal_initial_layout_tower_strict as fol  # noqa: E402
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast  # noqa: E402
import find_optimal_initial_layout_tower_global as glob  # noqa: E402
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2  # noqa: E402


def _setup_cjk_font() -> None:
    """让 matplotlib 正常显示中文, 修复方块乱码与负号。"""
    from matplotlib import font_manager
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
        "PingFang SC", "Hiragino Sans GB", "STHeiti",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [n for n in candidates if n in installed]
    if picked:
        plt.rcParams["font.sans-serif"] = picked + list(plt.rcParams.get("font.sans-serif", []))
        print(f"[fine-real] 中文字体 = {picked[0]}")
    else:
        print("[fine-real] WARN: 未找到中文字体, 中文可能仍为方块。")
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
#  构造真实搜索器 (镜像 fol.main 的构造块)
# ============================================================================
def build_searcher(passthrough_argv: List[str]):
    """用真实脚本的 argparse 默认值 + 少量透传参数构造搜索器。

    通过临时替换 sys.argv 复用 fol._parse_args(), 保证与真实搜索同口径,
    然后照 fol.main() 的构造块创建 GlobalLayoutSearcher (patch 之后 fol 的
    WeightedInitialLayoutSearcher 就是它, 从而带上 _pattern_refine/_offsets)。
    """
    glob._patch_module()  # fol.WeightedInitialLayoutSearcher = GlobalLayoutSearcher

    saved = sys.argv
    sys.argv = [saved[0]] + list(passthrough_argv)
    try:
        # 与 global/nsga2 一致: 默认关闭 order-x 硬约束 (否则 tower 布局会被误杀)。
        fast._maybe_inject_default_flags()
        args = fol._parse_args()
    finally:
        sys.argv = saved

    part_order = fol._parse_part_order(args.part_order)
    grasp_map = fol._load_json_map(args.grasp_map_json)
    _use_l2_pick_quick = bool(args.enable_l2_pick_quick_check and not args.disable_l2_pick_quick_check)

    searcher = fol.WeightedInitialLayoutSearcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        fixture_pos=fol._parse_vec3(args.fixture_pos, (0.36, 0.0, 0.0)),
        fixture_rotmat=np.eye(3),
        robot_base_pos=fol._parse_vec3(args.robot_base_pos, (0.0, 0.0, 0.0)),
        robot_base_rotmat=np.eye(3),
        part_order=part_order,
        output_name=args.output_name,
        table_name=args.table_name,
        table_margin=args.table_margin,
        table_clearance=args.table_clearance,
        grasp_map=grasp_map,
        max_rot_candidates=args.max_rot_candidates,
        w_grasp=args.w_grasp,
        w_manip=args.w_manip,
        w_dist=args.w_dist,
        w_rot=args.w_rot,
        ignore_env=args.ignore_env,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        plan_assembly_region=not args.disable_assembly_region_search,
        assembly_grid=args.assembly_grid,
        preassemble_first_part=not args.disable_preassemble_first,
        filter_assembly_near_arms=False,
        assembly_arm_x_clearance=args.assembly_arm_x_clearance,
        assembly_arm_y_clearance=args.assembly_arm_y_clearance,
        filter_staging_near_arms=not args.disable_staging_arm_keepout,
        staging_arm_x_clearance=args.staging_arm_x_clearance,
        staging_arm_y_clearance=args.staging_arm_y_clearance,
        goal_y_side_biased_sampling=not args.disable_goal_y_side_biased_sampling,
        goal_y_side_bias_ratio=args.goal_y_side_bias_ratio,
        goal_y_side_eps=args.goal_y_side_eps,
        use_flatsurface=not args.disable_flatsurface,
        fs_stability_threshold=args.fs_stability_threshold,
        check_robot_home_collision=not args.disable_home_collision_check,
        strict_initial_robot_collision=not args.disable_strict_initial_robot_collision,
        min_staging_mesh_clearance=args.min_staging_mesh_clearance,
        enforce_order_x_constraint=not args.disable_order_x_constraint,
        order_x_tolerance=args.order_x_tolerance,
        enable_y_side_distribution_score=not args.disable_y_side_distribution_score,
        prefer_upright_when_topdown_low=not args.disable_upright_preference,
        topdown_min_count=args.topdown_min_count,
        check_l2_pick_quick_motion=_use_l2_pick_quick,
        l2_pick_check_parts=fol._parse_part_order(args.l2_pick_check_parts) or [],
        l3_skip_parts=fol._parse_part_order(args.l3_skip_parts) or [],
        l2_pick_check_lift_dist=args.l2_pick_check_lift_dist,
        l2_pick_check_directions=fol._parse_part_order(args.l2_pick_check_directions) or [],
        l2_pick_check_tilt=args.l2_pick_check_tilt,
        robot_home_clearance=args.robot_home_clearance,
        prefer_stl_upface=not args.disable_prefer_stl_upface,
        w_stl_upface=args.w_stl_upface,
        stl_upface_min_thinness=args.stl_upface_min_thinness,
    )
    return searcher


# ============================================================================
#  从 .layout 载入起点 gene + region + 搜索口径
# ============================================================================
def load_layout_doc(layout_path: str) -> Tuple[Dict[str, np.ndarray], str, dict]:
    """从 .layout 读取 gene (x,y)、assembly_region_id 与 metadata。"""
    with open(layout_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    staging = doc.get("staging", {}) or {}
    gene: Dict[str, np.ndarray] = {}
    for pid, node in staging.items():
        pos = node.get("pos")
        if pos is None:
            continue
        gene[pid] = np.array([float(pos[0]), float(pos[1])], dtype=float)
    meta = doc.get("metadata") or {}
    region_id = str(meta.get("assembly_region_id") or "")
    return gene, region_id, meta


def apply_searcher_from_layout_metadata(searcher, metadata: dict) -> float:
    """把 .layout 里记录的约束/权重写回搜索器, 保证重算分数与搜索时同口径。"""
    sc = metadata.get("spatial_constraints") or {}
    _pairs = [
        ("enforce_order_x_constraint", "enforce_order_x_constraint", bool),
        ("order_x_tolerance", "order_x_tolerance", float),
        ("min_staging_mesh_clearance", "min_staging_mesh_clearance", float),
        ("enable_y_side_distribution_score", "enable_y_side_distribution_score", bool),
        ("force_upright_when_topdown_low", "prefer_upright_when_topdown_low", bool),
        ("topdown_min_count", "topdown_min_count", int),
        ("topdown_align_cos", "topdown_align_cos", float),
        ("check_l2_pick_quick_motion", "check_l2_pick_quick_motion", bool),
        ("l2_pick_check_lift_dist", "l2_pick_check_lift_dist", float),
        ("l2_pick_check_tilt", "l2_pick_check_tilt", float),
        ("goal_y_side_biased_sampling", "goal_y_side_biased_sampling", bool),
        ("goal_y_side_bias_ratio", "goal_y_side_bias_ratio", float),
        ("goal_y_side_eps", "goal_y_side_eps", float),
        ("robot_home_clearance", "robot_home_clearance", float),
    ]
    for src, dst, cast in _pairs:
        if src in sc:
            setattr(searcher, dst, cast(sc[src]))
    if "l2_pick_check_parts" in sc:
        searcher.l2_pick_check_parts = set(sc["l2_pick_check_parts"] or [])
    if "l2_pick_check_directions" in sc:
        searcher.l2_pick_check_directions = list(sc["l2_pick_check_directions"] or [])

    weights = metadata.get("weights") or {}
    if "grasp" in weights:
        searcher.w_grasp = float(weights["grasp"])
    if "manip" in weights:
        searcher.w_manip = float(weights["manip"])
    if "dist" in weights:
        searcher.w_dist = float(weights["dist"])
    if "rot" in weights:
        searcher.w_rot = float(weights["rot"])

    if "filter_staging_near_arms" in metadata:
        searcher.filter_staging_near_arms = bool(metadata["filter_staging_near_arms"])
    if "staging_arm_x_clearance" in metadata:
        searcher.staging_arm_x_clearance = float(metadata["staging_arm_x_clearance"])
    if "staging_arm_y_clearance" in metadata:
        searcher.staging_arm_y_clearance = float(metadata["staging_arm_y_clearance"])

    return float(metadata.get("score", float("nan")))


def _offsets(step: float, diagonal: bool) -> List[Tuple[float, float]]:
    """与 global 脚本一致。"""
    base = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]
    if diagonal:
        base += [(step, step), (step, -step), (-step, step), (-step, -step)]
    return base


# ============================================================================
#  真实评分的坐标模式搜索 (复现 _pattern_refine, 带 trace)
# ============================================================================
def pattern_refine_real(searcher, gene0: Dict[str, np.ndarray], region,
                        part_id: str, steps: List[float], rounds: int,
                        diagonal: bool) -> Dict:
    eps = 1e-4
    best_xy = nsga2._copy_xy(gene0)

    base_cand = searcher._evaluate_gene(nsga2._copy_xy(best_xy), region)
    if not bool(getattr(base_cand, "l2_pass", False)):
        print(f"[fine-real] 警告: 起点布局本身 L2 不可行 "
              f"(fail={getattr(base_cand, 'fail_reason', '?')}); 仍继续演示。")
    best_score = float(getattr(base_cand, "layout_score", -1.0))

    accepted = [(best_xy[part_id].copy(), best_score, steps[0])]
    probes: List[Tuple[np.ndarray, float, float, bool]] = []
    step_marks: List[Tuple[int, float]] = []
    n_eval = 0

    for step in steps:
        step_marks.append((n_eval, step))
        for _r in range(max(1, rounds)):
            improved = False
            anchor = np.asarray(best_xy[part_id], dtype=float)
            for dx, dy in _offsets(step, diagonal):
                trial_xy = nsga2._copy_xy(best_xy)
                trial_xy[part_id] = searcher._clip_xy_for_part(
                    part_id, anchor + np.array([dx, dy], dtype=float))
                child = searcher._evaluate_gene(trial_xy, region)
                n_eval += 1
                ok = bool(getattr(child, "l2_pass", False))
                sc = float(getattr(child, "layout_score", -1.0))
                is_acc = ok and sc > best_score + eps
                probes.append((np.asarray(trial_xy[part_id]).copy(), sc if ok else np.nan,
                               step, is_acc))
                print(f"  [{part_id}] step={step:g} off=({dx:+.3f},{dy:+.3f}) "
                      f"{'L2_OK' if ok else 'FAIL '} score={sc:.4f} "
                      f"{'<= ACCEPT' if is_acc else ''}")
                if is_acc:
                    best_xy = nsga2._copy_xy(trial_xy)
                    best_score = sc
                    anchor = np.asarray(best_xy[part_id], dtype=float)
                    improved = True
                    accepted.append((best_xy[part_id].copy(), best_score, step))
            if not improved:
                break

    return {
        "accepted": accepted, "probes": probes, "step_marks": step_marks,
        "best_xy": best_xy, "best_score": best_score, "n_eval": n_eval,
        "start_score": accepted[0][1],
    }


def _real_heatmap(searcher, gene0, region, part_id, center, half, n):
    """在 part 的 (x,y) 周围采一张低分辨率真实分数网格 (慢!)。"""
    gx = np.linspace(center[0] - half, center[0] + half, n)
    gy = np.linspace(center[1] - half, center[1] + half, n)
    GZ = np.full((n, n), np.nan)
    for i, yy in enumerate(gy):
        for j, xx in enumerate(gx):
            trial = nsga2._copy_xy(gene0)
            trial[part_id] = searcher._clip_xy_for_part(part_id, np.array([xx, yy]))
            c = searcher._evaluate_gene(trial, region)
            if bool(getattr(c, "l2_pass", False)):
                GZ[i, j] = float(getattr(c, "layout_score", np.nan))
    return gx, gy, GZ


# ============================================================================
#  画图
# ============================================================================
def _plot(trace, part_id, steps, heatmap, out_path, show):
    accepted, probes = trace["accepted"], trace["probes"]
    step_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))
    color_of = {s: step_colors[i] for i, s in enumerate(steps)}

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.3, 1.0]})

    if heatmap is not None:
        gx, gy, GZ = heatmap
        cf = ax.contourf(gx, gy, GZ, levels=20, cmap="Greys", alpha=0.85)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="真实 layout_score")

    for xy, s, step, is_acc in probes:
        if is_acc:
            ax.scatter(xy[0], xy[1], s=55, color=color_of[step],
                       edgecolors="k", linewidths=0.6, zorder=4)
        else:
            ax.scatter(xy[0], xy[1], s=16, color="0.5", marker="x", alpha=0.7, zorder=3)
    for k in range(1, len(accepted)):
        p0, p1, step = accepted[k - 1][0], accepted[k][0], accepted[k][2]
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color=color_of[step], lw=2.0), zorder=5)

    ax.scatter(*accepted[0][0], s=180, marker="*", color="tab:red",
               edgecolors="k", zorder=6, label="起点 (layout 中的位置)")
    ax.scatter(*trace["best_xy"][part_id], s=90, marker="o", facecolors="none",
               edgecolors="tab:red", linewidths=2.2, zorder=6, label="精修终点")
    handles = [ax.scatter([], [], s=55, color=color_of[s], edgecolors="k",
                          label=f"接受: step={s:g} m") for s in steps]
    handles.append(ax.scatter([], [], s=16, color="0.5", marker="x", label="拒绝(未提升/不可行)"))
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(f"{part_id} X (m)")
    ax.set_ylabel(f"{part_id} Y (m)")
    ax.set_title(f"真实权重分数下的 Phase B 精修: 零件 [{part_id}] XY 轨迹\n"
                 "(箭头=被接受的移动; 颜色=当时步长, 粗->细)")
    ax.set_aspect("equal", adjustable="box")

    best = trace["start_score"]
    xs, ys = [0], [best]
    for i, (xy, s, step, is_acc) in enumerate(probes, start=1):
        if is_acc:
            best = s
        xs.append(i)
        ys.append(best)
    ax2.plot(xs, ys, color="tab:blue", lw=2.0, label="当前最优真实分数 (贪心)")
    ax2.scatter(range(1, len(probes) + 1), [pr[1] for pr in probes],
                s=12, color="0.6", alpha=0.6, label="每次试探真实分数 (NaN=不可行)")
    for eval_idx, step in trace["step_marks"]:
        ax2.axvline(eval_idx + 0.5, color=color_of[step], ls="--", lw=1.3)
        ax2.text(eval_idx + 0.5, ax2.get_ylim()[0], f" step={step:g}",
                 rotation=90, va="bottom", ha="left", fontsize=8, color=color_of[step])
    ax2.set_xlabel("真实评估次数 (eval)")
    ax2.set_ylabel("真实 layout_score")
    ax2.set_title("贪心接受: 真实分数只升不降; 虚线=切换到更小步长")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Phase B pattern refine (真实权重)  |  part={part_id}  "
        f"|  步长={[f'{s:g}' for s in steps]}  |  真实评估={trace['n_eval']}  "
        f"|  分数 {trace['start_score']:.4f} -> {trace['best_score']:.4f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"[fine-real] 图已保存 -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_floats(spec: str) -> List[float]:
    return [float(s) for s in str(spec).replace(",", " ").split() if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用真实权重 layout_score 可视化 Phase B 由粗到细精修。")
    ap.add_argument("--layout", required=True, help="起点 .layout 路径")
    ap.add_argument("--part", default="post_bl", help="要精修的零件 id (默认 post_bl)")
    ap.add_argument("--steps", default="0.03,0.015,0.008", help="步长(米), 粗->细")
    ap.add_argument("--rounds", type=int, default=2, help="每个步长最多轮数")
    ap.add_argument("--diagonal", action="store_true", help="额外尝试 4 个对角方向")
    ap.add_argument("--heatmap", type=int, default=0,
                    help="额外采 NxN 真实分数热力图 (慢, 0=不采)。建议 7~11。")
    ap.add_argument("--heatmap-half", type=float, default=0.08,
                    help="热力图半宽 (米), 以零件当前位置为中心。默认 0.08")
    ap.add_argument("--out", default=None, help="输出 PNG (默认 _output/fine_refine_real.png)")
    ap.add_argument("--no-show", action="store_true", help="只存图不弹窗")
    # 透传给真实搜索器构造的常用参数 (其余用脚本默认)。
    ap.add_argument("--cdprim-type", default="box", help="碰撞原语类型 (透传, 默认 box)")
    ap.add_argument("--config", default=None, help="覆盖 config 路径 (透传)")
    ap.add_argument("--asmdef", default=None, help="覆盖 asmdef 路径 (透传)")
    ap.add_argument("--grasp-dir", default=None, help="覆盖 grasp 目录 (透传)")
    args = ap.parse_args()

    if not os.path.isfile(args.layout):
        ap.error(f"layout 不存在: {args.layout}")
    steps = _parse_floats(args.steps)
    if not steps:
        ap.error("--steps 解析为空")

    if args.no_show:
        matplotlib.use("Agg")
    _setup_cjk_font()

    passthrough: List[str] = ["--cdprim-type", args.cdprim_type]
    if args.config:
        passthrough += ["--config", args.config]
    if args.asmdef:
        passthrough += ["--asmdef", args.asmdef]
    if args.grasp_dir:
        passthrough += ["--grasp-dir", args.grasp_dir]

    print("[fine-real] 构造真实搜索器 (需要完整仿真环境, 稍慢) ...")
    t0 = time.time()
    searcher = build_searcher(passthrough)
    print(f"[fine-real] 搜索器就绪 ({time.time() - t0:.1f}s)")

    gene0, region_id, layout_meta = load_layout_doc(args.layout)
    file_score = apply_searcher_from_layout_metadata(searcher, layout_meta)
    print(f"[fine-real] 已从 layout 恢复搜索口径 "
          f"(order-x={searcher.enforce_order_x_constraint}, "
          f"weights={searcher.w_grasp}/{searcher.w_manip}/{searcher.w_dist}/{searcher.w_rot})")
    if np.isfinite(file_score):
        print(f"[fine-real] layout 文件记录分数 = {file_score:.4f}")

    if args.part not in gene0:
        ap.error(f"零件 {args.part!r} 不在 layout 里; 可选: {sorted(gene0.keys())}")
    regions = searcher._assembly_region_candidates()
    region = searcher._region_by_id(regions, region_id) if region_id else regions[0]
    print(f"[fine-real] region={region_id or region[0]}, 精修零件={args.part}, "
          f"起点XY={gene0[args.part].tolist()}")

    trace = pattern_refine_real(searcher, gene0, region, args.part,
                                steps, args.rounds, args.diagonal)
    print(f"[fine-real] 真实评估 {trace['n_eval']} 次, "
          f"分数 {trace['start_score']:.4f} -> {trace['best_score']:.4f}, "
          f"接受 {len(trace['accepted']) - 1} 次")

    heatmap = None
    if args.heatmap and args.heatmap >= 2:
        print(f"[fine-real] 采 {args.heatmap}x{args.heatmap} 真实分数热力图 (慢) ...")
        heatmap = _real_heatmap(searcher, gene0, region, args.part,
                                gene0[args.part], args.heatmap_half, int(args.heatmap))

    out_path = args.out or os.path.join(_HERE, "_output", "fine_refine_real.png")
    _plot(trace, args.part, steps, heatmap, out_path, show=not args.no_show)
    print("[fine-real] done.")


if __name__ == "__main__":
    main()
