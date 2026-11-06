# -*- coding: utf-8 -*-
"""
论文级 Layout 诊断可视化
=========================

读取 ``find_optimal_layout`` 产出的 ``<name>.layout`` 与对应的
``<name>_diagnostics.json``，生成 5 张严谨的可视化图：

    F1  workspace_topdown        — 工作空间俯视图（base / staging / goal /
                                    transport corridor）
    F2  reachability_heatmap     — 每个零件在自身 staging 范围内 XY 网格
                                    扫描得到的 (n_grasps × manipulability)
                                    quality 场，标记所选最优位置
    F3  score_decomposition     — 上：top-K 候选的加权分量堆积柱状图；
                                    下：Hill / exp / exp-decay 三条归一化
                                    曲线，分别标出最优解的取值
    F4  sample_distribution      — 候选样本得分直方图 + L2 通过样本的
                                    (grasp,dist) 散点（颜色=manip）
    F5  constraint_funnel        — sampled → L1 pass → L2 pass → final
                                    的样本计数漏斗

输出位置：
    ``_output/diagnostics_figs/<task>/<fig_name>.{pdf,png}``  以及
    汇总的 ``dashboard.png``。

用法
----

    python -m sealp.examples.layout.diagnostics_plot \
        --layout sealp/examples/layout/_output/dual_yuanchair_optimal_searched.layout

可选参数：

    --skip-heatmap        跳过 F2（重新做 IK 扫描需 ~3-8 分钟）
    --grid-res 12         热力图 XY 分辨率（默认 14；越大越精细越慢）
    --out-dir <path>      自定义输出目录

热力图结果会缓存到 ``_output/diagnostics_cache/<task>_reach.npz``，
下次运行同 grid_res 直接读缓存。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 静默 matplotlib backend；headless 环境也能出图 ────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from sealp.layout import WorkspaceLayout


# ══════════════════════════════════════════════════════════════════════════
#  Publication-quality matplotlib style
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavuserif",
})

# 配色：左/右臂、得分分量、热力图
_ARM_COLOR = {"lft": "#2ca02c", "rgt": "#1f77b4",
              "dual": "#9467bd", "?": "#888888"}
_COMP_COLOR = {"grasp": "#1f77b4", "manip": "#2ca02c", "dist": "#d62728"}


# ══════════════════════════════════════════════════════════════════════════
#  通用 helpers
# ══════════════════════════════════════════════════════════════════════════
def _load_diag(layout_path: str) -> Tuple[dict, bool]:
    """读 ``<layout>_diagnostics.json``。如不存在，回退到 layout.metadata 凑数。

    Returns:
        ``(diag_dict, has_samples)``。``has_samples`` 为 False 表示 fallback，
        此时 F3 / F4 / F5 会自动 skip 并打印提示。
    """
    base, _ = os.path.splitext(layout_path)
    diag_path = base + "_diagnostics.json"
    if os.path.isfile(diag_path):
        with open(diag_path, "r", encoding="utf-8") as fp:
            return json.load(fp), True
    # ── Fallback：从 layout.metadata + FastLayoutSearcher 凑 F1/F2 需要的字段
    print(f"[WARN] 未找到诊断 JSON：{os.path.relpath(diag_path)}")
    print(f"       使用 layout.metadata + 重新实例化 searcher 凑 F1/F2；"
          f"F3/F4/F5 因缺 per-sample 数据将跳过。")
    print(f"       若需要全部 5 张图，重新跑一次 "
          f"`python -m sealp.examples.layout.find_optimal_layout` 即可。")
    layout = WorkspaceLayout.load(layout_path)
    md = dict(layout.metadata or {})
    # 实例化 searcher 抓 bounds/world_poses/chosen_rotmat
    from sealp.examples.layout.find_optimal_layout import (
        FastLayoutSearcher, YUANCHAIR_FAST_TASK, _DUAL_ARM_Y_OFFSET,
    )
    searcher = FastLayoutSearcher(YUANCHAIR_FAST_TASK, enable_l3=False)
    fake = {
        "task_name": YUANCHAIR_FAST_TASK.name,
        "robot_type": md.get("robot_type", "panthera_ht"),
        "arm_y_offset": float(md.get("arm_y_offset", _DUAL_ARM_Y_OFFSET)),
        "weights": md.get("weights", {"grasp": 0.5, "manip": 0.2, "dist": 0.3}),
        "norm_targets": md.get("norm_targets", {}),
        "filters": md.get("filters", {}),
        "funnel": {},
        "bounds": {
            pid: [list(searcher.bounds[pid][0]),
                  list(searcher.bounds[pid][1])]
            for pid in searcher.search_part_ids},
        "robot_base_pos": list(map(float, YUANCHAIR_FAST_TASK.robot_base_pos)),
        "fixture_pos": list(map(float, YUANCHAIR_FAST_TASK.fixture_pos)),
        "search_part_ids": list(searcher.search_part_ids),
        "world_poses": {
            pid: {"pos": list(map(float, p)),
                  "rotmat": [list(map(float, r)) for r in R]}
            for pid, (p, R) in searcher.world_poses.items()},
        "best": {
            "xy": {pid: [float(layout.staging_positions[pid][0][0]),
                         float(layout.staging_positions[pid][0][1])]
                   for pid in layout.staging_positions},
            # 取 layout staging 的 rotmat 作为 chosen_rotmat
            "chosen_rotmat": {
                pid: [list(map(float, r))
                      for r in layout.staging_positions[pid][1]]
                for pid in layout.staging_positions},
            "z_offset": md.get("z_offsets", {}),
            "arm_choice": md.get("arm_choice", {}),
            "grasp_counts": md.get("grasp_counts", {}),
            "pose_tag": md.get("pose_tags", {}),
            "per_part_manip": (md.get("score_components", {})
                               .get("per_part_manip", {})),
            "per_part_dist": (md.get("score_components", {})
                              .get("per_part_dist", {})),
            "avg_manip": float(md.get("manipulability_avg",
                                      md.get("score_components", {})
                                        .get("avg_manip", 0.0))),
            "avg_dist": float(md.get("distance_cost",
                                     md.get("score_components", {})
                                       .get("avg_dist", 0.0))),
            "grasp_score_norm": 0.0,
            "manip_score_norm": 0.0,
            "dist_score_norm": 0.0,
            "score": float(md.get("total_score", 0.0)),
        },
        "all_samples": [],
    }
    return fake, False


def _hill(x: float, T: float, k: float) -> float:
    if x <= 0.0:
        return 0.0
    r = (x / T) ** k
    return r / (1.0 + r)


def _exp_sat(x: float, T: float) -> float:
    if x <= 0.0:
        return 0.0
    return float(1.0 - np.exp(-x / T))


def _exp_decay(x: float, D: float) -> float:
    if x <= 0.0:
        return 1.0
    return float(np.exp(-x / D))


def _arm_color(arm_tag: str) -> str:
    return _ARM_COLOR.get(arm_tag, _ARM_COLOR["?"])


def _savefig(fig, out_dir: str, name: str, formats=("pdf", "png")):
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  F1: Workspace top-down
# ══════════════════════════════════════════════════════════════════════════
def plot_workspace_topdown(layout: WorkspaceLayout, diag: dict, out_dir: str):
    """画桌面俯视图：robot base / staging / goal / transport corridor / bounds。

    论证：整体几何合理性，arm 分配是否对称，transport 路径是否短。
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    arm_y_off = float(diag.get("arm_y_offset", 0.62))
    rb = np.asarray(diag.get("robot_base_pos", [0, 0, 0]), dtype=float)
    lft_base = (rb[0], rb[1])
    rgt_base = (rb[0], rb[1] - arm_y_off)

    bounds = diag.get("bounds", {})
    world_poses = diag.get("world_poses", {})
    best = diag.get("best", {})
    arm_choice = best.get("arm_choice", {}) or layout.metadata.get("arm_choice", {})

    # ── 计算画面范围（覆盖 base + bounds + goal） ──
    xs, ys = [lft_base[0], rgt_base[0]], [lft_base[1], rgt_base[1]]
    for pid, b in bounds.items():
        (xlo, xhi), (ylo, yhi) = b
        xs.extend([xlo, xhi]); ys.extend([ylo, yhi])
    for pid, wp in world_poses.items():
        xs.append(wp["pos"][0]); ys.append(wp["pos"][1])
    pad = 0.10
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")

    # ── 各 part bounds（虚线矩形），用 arm 颜色 ──
    for pid, b in bounds.items():
        (xlo, xhi), (ylo, yhi) = b
        c = _arm_color(arm_choice.get(pid, "?"))
        rect = Rectangle(
            (xlo, ylo), xhi - xlo, yhi - ylo,
            linewidth=1.0, edgecolor=c, facecolor=c, alpha=0.06,
            linestyle="--", zorder=1)
        ax.add_patch(rect)

    # ── 桌面 / fixture 大致 ──
    fx = diag.get("fixture_pos", [0.0, 0.0, 0.0])
    ax.scatter([fx[0]], [fx[1]], marker="s", s=120,
               c="none", edgecolors="#444444", linewidths=1.5, zorder=3)
    ax.annotate("fixture", (fx[0], fx[1]), xytext=(-22, -14),
                textcoords="offset points", fontsize=8, color="#444444")

    # ── 机械臂 base ──
    ax.scatter([lft_base[0]], [lft_base[1]],
               marker="P", s=160, c=_ARM_COLOR["lft"],
               edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate("lft_arm base", lft_base, xytext=(6, 6),
                textcoords="offset points", fontsize=9,
                color=_ARM_COLOR["lft"], fontweight="bold")
    ax.scatter([rgt_base[0]], [rgt_base[1]],
               marker="P", s=160, c=_ARM_COLOR["rgt"],
               edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate("rgt_arm base", rgt_base, xytext=(6, -14),
                textcoords="offset points", fontsize=9,
                color=_ARM_COLOR["rgt"], fontweight="bold")

    # ── Goal 与 staging + transport corridor ──
    for pid in layout.staging_positions:
        st_pos, _ = layout.staging_positions[pid]
        if pid in world_poses:
            gp = world_poses[pid]["pos"]
        else:
            continue
        arm = arm_choice.get(pid, "?")
        c = _arm_color(arm)

        # Goal：空心方框
        ax.scatter([gp[0]], [gp[1]], marker="s", s=100,
                   c="none", edgecolors=c, linewidths=1.6, zorder=4)
        # Staging：实心圆
        ax.scatter([st_pos[0]], [st_pos[1]], marker="o", s=150,
                   c=c, edgecolors="black", linewidths=0.8,
                   zorder=6, alpha=0.95)
        # 箭头 staging → goal
        arr = FancyArrowPatch(
            (st_pos[0], st_pos[1]), (gp[0], gp[1]),
            arrowstyle="->", mutation_scale=12,
            color=c, alpha=0.55, linewidth=1.3, zorder=3,
            linestyle="-")
        ax.add_patch(arr)
        # 标签
        ax.annotate(pid, (st_pos[0], st_pos[1]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8.5, color=c, fontweight="bold")
        ax.annotate("goal", (gp[0], gp[1]),
                    xytext=(6, -12), textcoords="offset points",
                    fontsize=7.5, color=c, alpha=0.85)

    # ── Legend ──
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="P", color="w", markerfacecolor=_ARM_COLOR["lft"],
               markersize=10, markeredgecolor="black", label="lft arm base"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor=_ARM_COLOR["rgt"],
               markersize=10, markeredgecolor="black", label="rgt arm base"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbbbbb",
               markersize=10, markeredgecolor="black", label="chosen staging"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
               markeredgecolor="#444444", markersize=10, label="goal pose"),
        Line2D([0], [0], color="#bbbbbb", linestyle="--", label="search bounds"),
        Line2D([0], [0], color="#888888", linestyle="-", label="transport corridor"),
    ]
    ax.legend(handles=legend_elems, loc="best", framealpha=0.92)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Workspace layout — task={diag.get('task_name', '?')}  "
        f"score={best.get('score', float('nan')):.4f}  "
        f"|  arm_choice: {arm_choice}",
        fontsize=10)
    _savefig(fig, out_dir, "F1_workspace_topdown")


# ══════════════════════════════════════════════════════════════════════════
#  F2: Reachability heatmap (the killer figure)
# ══════════════════════════════════════════════════════════════════════════
def _build_reachability_landscape(
        layout: WorkspaceLayout, diag: dict, *,
        grid_res: int = 14, max_grasps_per_eval: int = 30,
        cache_dir: Optional[str] = None,
        verbose: bool = True) -> Dict[str, dict]:
    """对每个 part 在 bounds 内做 XY 网格扫描，记录 n_grasps & best_manip。

    Returns:
        ``{pid: {"X": (N,) 1D, "Y": (M,) 1D, "n_grasps": (M,N), "manip": (M,N),
                  "chosen_xy": (2,), "bounds": ((xlo,xhi),(ylo,yhi))}, ...}``
    """
    task_name = diag.get("task_name", "yuanchair")
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{task_name}_reach_grid{grid_res}.npz")
        if os.path.isfile(cache_path):
            if verbose:
                print(f"[F2] 命中缓存：{os.path.relpath(cache_path)}")
            data = np.load(cache_path, allow_pickle=True)
            return {k: data[k].item() for k in data.files}

    if verbose:
        print(f"[F2] 计算 reachability landscape (grid={grid_res}x{grid_res}, "
              f"max_grasps={max_grasps_per_eval})…")
    # 用 FastLayoutSearcher 复用 reachability 检查的完整环境
    from sealp.examples.layout.find_optimal_layout import (
        FastLayoutSearcher, YUANCHAIR_FAST_TASK,
    )
    from sealp.layout.reachability import check_pose_reachability
    import wrs.modeling.collision_model as mcm

    searcher = FastLayoutSearcher(YUANCHAIR_FAST_TASK, enable_l3=False)

    bounds = diag.get("bounds", {})
    best = diag.get("best", {})
    best_xy = best.get("xy", {})
    best_arm_choice = best.get("arm_choice", {})
    chosen_rotmat_arr = best.get("chosen_rotmat", {})
    z_offset = best.get("z_offset", {})

    # 把其它部件（非当前扫描的）放到各自选定的最优 staging 上，作为 obstacle
    # 这样扫描的 "context" 接近真实使用场景。
    def _set_other_staging(this_pid):
        for pid in searcher.search_part_ids:
            if pid == this_pid:
                continue
            xy = best_xy.get(pid)
            if xy is None:
                continue
            z = float(z_offset.get(pid, 0.0))
            r = chosen_rotmat_arr.get(pid)
            R = np.eye(3) if r is None else np.asarray(r, dtype=float)
            searcher.staging_obs[pid].pos = np.array(
                [float(xy[0]), float(xy[1]), z])
            searcher.staging_obs[pid].rotmat = R

    out: Dict[str, dict] = {}
    for pid in searcher.search_part_ids:
        if pid not in bounds:
            continue
        (xlo, xhi), (ylo, yhi) = bounds[pid]
        X = np.linspace(xlo, xhi, grid_res)
        Y = np.linspace(ylo, yhi, grid_res)
        # 选定的 rotmat & arm
        rot_arr = chosen_rotmat_arr.get(pid)
        R = np.eye(3) if rot_arr is None else np.asarray(rot_arr, dtype=float)
        arm_tag = best_arm_choice.get(pid, "lft")
        arm_obj = (searcher.robot.lft_arm if arm_tag == "lft"
                   else searcher.robot.rgt_arm)
        z = float(z_offset.get(pid, 0.0))

        _set_other_staging(pid)

        n_grasps_mat = np.zeros((len(Y), len(X)), dtype=float)
        manip_mat = np.zeros((len(Y), len(X)), dtype=float)

        if pid in searcher.world_poses:
            gp, gr = searcher.world_poses[pid]
        else:
            gp, gr = None, None

        t0 = time.time()
        for j, y in enumerate(Y):
            for i, x in enumerate(X):
                sp = np.array([float(x), float(y), z])
                searcher.staging_obs[pid].pos = sp
                searcher.staging_obs[pid].rotmat = R
                # 该 step 的 obs：所有其它 staging + env_obs
                obs = list(searcher.env_obs) + [
                    searcher.staging_obs[p]
                    for p in searcher.search_part_ids
                    if p != pid and p in searcher.staging_obs]
                gc = searcher.grasp_cache.get(searcher.model_alias_fn(pid))
                if gc is None or len(gc) == 0:
                    continue
                # 仅评估 pick 端（place 端的可达性与 staging XY 无关，
                # 不影响 landscape 的形状；省一半 IK 计算量）。
                pr = check_pose_reachability(
                    arm_obj, sp, R, gc, obs,
                    max_grasps=max_grasps_per_eval)
                n_grasps_mat[j, i] = int(pr.n_collision_free)
                manip_mat[j, i] = float(pr.best_manipulability)
        dt = time.time() - t0
        if verbose:
            print(f"     · {pid:>8s}: {grid_res*grid_res} cells in {dt:.1f}s, "
                  f"max n_grasps={int(n_grasps_mat.max())}, "
                  f"max manip={manip_mat.max():.4f}")

        chosen = best_xy.get(pid, [None, None])
        out[pid] = {
            "X": X, "Y": Y,
            "n_grasps": n_grasps_mat, "manip": manip_mat,
            "chosen_xy": np.asarray(chosen, dtype=float),
            "bounds": ((xlo, xhi), (ylo, yhi)),
            "arm_tag": arm_tag,
        }

    if cache_path is not None:
        # 把每个 part 的 dict 存成单独条目
        np.savez(cache_path,
                 **{pid: np.array(v, dtype=object) for pid, v in out.items()})
        if verbose:
            print(f"[F2] 写缓存：{os.path.relpath(cache_path)}")
    return out


def plot_reachability_heatmaps(
        layout: WorkspaceLayout, diag: dict, out_dir: str, *,
        grid_res: int = 14, cache_dir: Optional[str] = None):
    """每个 part 一个 panel：n_grasps × manipulability 的 quality 场。

    论证（论文核心）：所选位置位于"高 quality"区域的局部极大，且邻域内
    的位置都不如它（视觉上一目了然）。
    """
    land = _build_reachability_landscape(
        layout, diag, grid_res=grid_res, cache_dir=cache_dir)

    pids = list(land.keys())
    n = len(pids)
    if n == 0:
        return
    # 排版：2 行 × ceil(n/2) 列
    n_col = (n + 1) // 2 if n > 1 else 1
    n_row = 2 if n > n_col else 1
    fig = plt.figure(figsize=(4.2 * n_col, 3.8 * n_row + 0.4))
    gs = GridSpec(n_row, n_col, figure=fig, wspace=0.35, hspace=0.45)

    # 为了对比公平：quality = n_grasps_normalized * manip_normalized
    # 各 part 各自归一化到 [0,1]（max 取该 part 网格内的最大）
    for k, pid in enumerate(pids):
        r = k // n_col
        c = k % n_col
        ax = fig.add_subplot(gs[r, c])
        d = land[pid]
        ng = d["n_grasps"]
        mp = d["manip"]
        denom_n = max(float(ng.max()), 1.0)
        denom_m = max(float(mp.max()), 1e-6)
        # quality \in [0, 1]
        quality = (ng / denom_n) * (mp / denom_m)
        im = ax.pcolormesh(
            d["X"], d["Y"], quality, cmap="viridis",
            vmin=0.0, vmax=max(float(quality.max()), 1e-6),
            shading="auto")
        # 等值线（突出局部极大）
        try:
            ax.contour(d["X"], d["Y"], quality,
                       levels=[0.3, 0.6, 0.85],
                       colors="white", alpha=0.5, linewidths=0.6)
        except Exception:
            pass
        # bounds 外框
        (xlo, xhi), (ylo, yhi) = d["bounds"]
        ax.add_patch(Rectangle(
            (xlo, ylo), xhi - xlo, yhi - ylo,
            edgecolor="white", facecolor="none",
            linewidth=1.2, linestyle="--"))
        # 选定位置
        cx, cy = float(d["chosen_xy"][0]), float(d["chosen_xy"][1])
        ax.scatter([cx], [cy], marker="*", s=260,
                   c="red", edgecolors="white", linewidths=1.2,
                   zorder=10)
        # 在选定格上读出 quality 值
        # （由于 chosen 可能不在网格点上，做最近邻读取）
        i_best = int(np.argmin(np.abs(d["X"] - cx)))
        j_best = int(np.argmin(np.abs(d["Y"] - cy)))
        q_best = float(quality[j_best, i_best])

        ax.set_title(
            f"{pid}  ({d['arm_tag']})\n"
            f"chosen quality≈{q_best:.2f}  "
            f"max grid quality={float(quality.max()):.2f}",
            fontsize=9.5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("normalized quality\n($n_g/n_g^{\\max}$ × $\\mu/\\mu^{\\max}$)",
                       fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Per-part reachability landscape — chosen position (★) "
        "vs. neighbourhood quality",
        fontsize=11, y=1.005)
    _savefig(fig, out_dir, "F2_reachability_heatmap")


# ══════════════════════════════════════════════════════════════════════════
#  F3: Score decomposition + normalization curves
# ══════════════════════════════════════════════════════════════════════════
def plot_score_decomposition(diag: dict, out_dir: str):
    """上：top-K 加权分量堆积；下：三条归一化曲线 + chosen 标记。"""
    weights = diag.get("weights", {"grasp": 0.5, "manip": 0.2, "dist": 0.3})
    norm = diag.get("norm_targets", {})
    T_min = norm.get("grasp_min", 15.0)
    T_mean = norm.get("grasp_mean", 25.0)
    k = norm.get("grasp_hill_k", 2.0)
    w_min = norm.get("grasp_min_weight", 0.7)
    T_manip = norm.get("manip", 0.030)
    D_dist = norm.get("dist_decay", 0.5)

    samples = diag.get("all_samples", [])
    passed = [s for s in samples if s.get("l2_pass")]
    passed.sort(key=lambda s: -float(s.get("score", 0.0)))
    top_k = passed[:min(8, len(passed))]
    best = passed[0] if passed else None

    fig = plt.figure(figsize=(11.0, 7.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0],
                  hspace=0.45, wspace=0.32)

    # ── 上：top-K 堆积柱状 ──
    ax0 = fig.add_subplot(gs[0, :])
    if top_k:
        idx = np.arange(len(top_k))
        gs_vals = np.array([s["grasp_score_norm"] for s in top_k]) * weights["grasp"]
        mp_vals = np.array([s["manip_score_norm"] for s in top_k]) * weights["manip"]
        ds_vals = np.array([s["dist_score_norm"]  for s in top_k]) * weights["dist"]
        ax0.bar(idx, gs_vals, label=f"$w_g·grasp$ (w={weights['grasp']})",
                color=_COMP_COLOR["grasp"], edgecolor="white", linewidth=0.5)
        ax0.bar(idx, mp_vals, bottom=gs_vals,
                label=f"$w_\\mu·manip$ (w={weights['manip']})",
                color=_COMP_COLOR["manip"], edgecolor="white", linewidth=0.5)
        ax0.bar(idx, ds_vals, bottom=gs_vals + mp_vals,
                label=f"$w_d·dist$ (w={weights['dist']})",
                color=_COMP_COLOR["dist"], edgecolor="white", linewidth=0.5)
        # 标 total score
        totals = gs_vals + mp_vals + ds_vals
        for i, t in enumerate(totals):
            ax0.text(i, t + 0.01, f"{t:.3f}", ha="center",
                     fontsize=8.5, color="#333")
        # 高亮 best
        ax0.axvspan(-0.5, 0.5, color="#fff3a8", alpha=0.45, zorder=0)
        ax0.text(0, -0.04, "best", ha="center", color="#7a5b00", fontsize=8.5,
                 transform=ax0.get_xaxis_transform())
        ax0.set_xticks(idx)
        ax0.set_xticklabels([f"#{int(s['idx'])}" for s in top_k])
        ax0.set_ylabel("weighted contribution\n→ layout_score ∈ [0,1]")
        ax0.set_xlabel(
            "L2-passed candidates (sorted by total score, best→worst)")
        ax0.legend(loc="upper right", ncol=3, framealpha=0.9)
        ax0.set_ylim(0, max(1.0, totals.max() * 1.12))
        ax0.set_title("Score decomposition of top-K candidates "
                      "(every component is normalised to [0,1])",
                      fontsize=10.5)
    else:
        ax0.text(0.5, 0.5, "no L2-passed samples", ha="center", va="center",
                 transform=ax0.transAxes)

    # ── 下：三条归一化曲线 + chosen 标记 ──
    # (1) Hill (grasp)
    ax1 = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0, max(60, 3 * T_min), 200)
    ax1.plot(xs, [_hill(x, T_min, k) for x in xs],
             color=_COMP_COLOR["grasp"], linewidth=2,
             label=f"$Hill(n_{{min}};T={T_min:.0f},k={k:.0f})$")
    ax1.plot(xs, [_hill(x, T_mean, k) for x in xs],
             color=_COMP_COLOR["grasp"], linewidth=2, linestyle="--",
             label=f"$Hill(\\bar n;T={T_mean:.0f},k={k:.0f})$")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
    if best is not None:
        n_min = min(best["grasp_counts"].values())
        n_mean = float(np.mean(list(best["grasp_counts"].values())))
        ax1.scatter([n_min], [_hill(n_min, T_min, k)],
                    marker="*", s=180, color="red",
                    edgecolors="white", linewidths=1.0, zorder=10,
                    label=f"chosen $n_{{min}}$={n_min}")
        ax1.scatter([n_mean], [_hill(n_mean, T_mean, k)],
                    marker="o", s=80, color="red",
                    edgecolors="white", linewidths=1.0, zorder=10,
                    label=f"chosen $\\bar n$={n_mean:.1f}")
    ax1.set_xlabel("$n_{grasps}$")
    ax1.set_ylabel("normalised grasp_score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title("Grasp redundancy (Hill function, soft saturation)",
                  fontsize=9.5)

    # (2) Manip
    ax2 = fig.add_subplot(gs[1, 1])
    xs = np.linspace(0, max(0.08, 3 * T_manip), 200)
    ax2.plot(xs, [_exp_sat(x, T_manip) for x in xs],
             color=_COMP_COLOR["manip"], linewidth=2,
             label=f"$1-e^{{-\\mu/{T_manip}}}$")
    ax2.axhline(0.632, color="gray", linestyle=":", linewidth=0.7,
                label="0.632 ≈ $1-e^{-1}$")
    if best is not None:
        m_best = float(best.get("avg_manip", 0.0))
        ax2.scatter([m_best], [_exp_sat(m_best, T_manip)],
                    marker="*", s=180, color="red",
                    edgecolors="white", linewidths=1.0, zorder=10,
                    label=f"chosen $\\bar\\mu$={m_best:.4f}")
    ax2.set_xlabel("avg manipulability $\\bar\\mu$")
    ax2.set_ylabel("normalised manip_score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("Manipulability (exp saturation)", fontsize=9.5)

    # (3) Dist
    ax3 = fig.add_subplot(gs[1, 2])
    xs = np.linspace(0, max(1.5, 3 * D_dist), 200)
    ax3.plot(xs, [_exp_decay(x, D_dist) for x in xs],
             color=_COMP_COLOR["dist"], linewidth=2,
             label=f"$e^{{-d/{D_dist}}}$")
    ax3.axhline(0.368, color="gray", linestyle=":", linewidth=0.7,
                label="0.368 ≈ $e^{-1}$")
    if best is not None:
        d_best = float(best.get("avg_dist", 0.0))
        ax3.scatter([d_best], [_exp_decay(d_best, D_dist)],
                    marker="*", s=180, color="red",
                    edgecolors="white", linewidths=1.0, zorder=10,
                    label=f"chosen $\\bar d$={d_best:.3f}m")
    ax3.set_xlabel("avg transport distance $\\bar d$ [m]")
    ax3.set_ylabel("normalised dist_score")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_title("Transport distance (exp decay)", fontsize=9.5)

    fig.suptitle(
        "Score decomposition & normalisation curves — "
        "the chosen layout sits in the high-quality regime of every component",
        fontsize=11, y=0.995)
    _savefig(fig, out_dir, "F3_score_decomposition")


# ══════════════════════════════════════════════════════════════════════════
#  F4: Sample distribution
# ══════════════════════════════════════════════════════════════════════════
def plot_sample_distribution(diag: dict, out_dir: str):
    """直方图 + 2D 散点（grasp vs dist，颜色=manip），强调 best 处于尖峰。"""
    samples = diag.get("all_samples", [])
    passed = [s for s in samples if s.get("l2_pass")]
    if not passed:
        return
    scores = np.array([float(s["score"]) for s in passed])
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    fig = plt.figure(figsize=(11.0, 4.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28, width_ratios=[1, 1.3])

    # ── 左：直方图 ──
    ax0 = fig.add_subplot(gs[0, 0])
    bins = max(6, int(np.sqrt(len(scores))) + 2)
    ax0.hist(scores, bins=bins, color="#a6cee3",
             edgecolor="white", linewidth=0.6)
    ax0.axvline(best_score, color="red", linewidth=1.8, linestyle="--",
                label=f"best = {best_score:.4f}")
    ax0.axvline(scores.mean(), color="#333", linewidth=1.0, linestyle=":",
                label=f"mean = {scores.mean():.4f}")
    ax0.set_xlabel("layout_score ∈ [0,1]")
    ax0.set_ylabel("# of L2-passed candidates")
    ax0.set_title(f"Score histogram (n={len(scores)} L2-passed)",
                  fontsize=10)
    ax0.legend(loc="upper left", fontsize=8.5)

    # ── 右：3 分量散点（颜色 = manip，size = grasp） ──
    ax1 = fig.add_subplot(gs[0, 1])
    gss = np.array([s["grasp_score_norm"] for s in passed])
    ds = np.array([s["dist_score_norm"] for s in passed])
    ms = np.array([s["manip_score_norm"] for s in passed])
    sc = ax1.scatter(gss, ds, c=ms, s=80,
                     cmap="plasma", vmin=0, vmax=max(ms.max(), 1e-6),
                     edgecolors="black", linewidths=0.5, alpha=0.85)
    # best 标星
    ax1.scatter([gss[best_idx]], [ds[best_idx]],
                marker="*", s=320, c="red",
                edgecolors="white", linewidths=1.4, zorder=10,
                label="best")
    ax1.set_xlabel("normalised grasp_score")
    ax1.set_ylabel("normalised dist_score")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    cbar = plt.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("normalised manip_score")
    ax1.set_title("Component-space projection of L2-passed candidates\n"
                  "(top-right = high reach & short transport)",
                  fontsize=10)
    ax1.legend(loc="lower left", fontsize=8.5)

    fig.suptitle(
        f"Sample distribution — best layout dominates the {len(scores)} "
        f"candidates in all three normalised components",
        fontsize=11, y=1.02)
    _savefig(fig, out_dir, "F4_sample_distribution")


# ══════════════════════════════════════════════════════════════════════════
#  F5: Constraint funnel
# ══════════════════════════════════════════════════════════════════════════
def plot_constraint_funnel(diag: dict, out_dir: str):
    """漏斗：sampled → L1 → L2 → final，附带 fail 原因 top-K。"""
    funnel = diag.get("funnel", {})
    samples = diag.get("all_samples", [])
    n_sampled = int(funnel.get("sampled", len(samples)))
    n_l1 = int(funnel.get("l1_pass", sum(1 for s in samples if s.get("l1_pass"))))
    n_l2 = int(funnel.get("l2_pass", sum(1 for s in samples if s.get("l2_pass"))))
    n_final = 1 if n_l2 > 0 else 0

    stages = ["sampled", "L1 pass", "L2 pass", "final"]
    counts = [n_sampled, n_l1, n_l2, n_final]
    colors = ["#cccccc", "#a6cee3", "#1f78b4", "#d62728"]

    fig = plt.figure(figsize=(11.0, 4.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32, width_ratios=[1, 1.1])

    # ── 左：水平条形漏斗 ──
    ax0 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(stages))[::-1]
    ax0.barh(y, counts, color=colors, edgecolor="white", height=0.7)
    for yi, (cnt, st) in enumerate(zip(counts, stages)):
        pct = (cnt / n_sampled * 100.0) if n_sampled > 0 else 0.0
        ax0.text(cnt + max(counts) * 0.015, y[yi],
                 f"{cnt}  ({pct:.0f}%)",
                 va="center", fontsize=10, color="#222")
    ax0.set_yticks(y)
    ax0.set_yticklabels(stages, fontsize=10)
    ax0.set_xlabel("# of candidates")
    ax0.set_xlim(0, max(counts) * 1.20 if max(counts) > 0 else 1)
    ax0.set_title("Constraint filtering funnel\n"
                  "(every survivor satisfies L1 home-clearance + "
                  "L2 endpoint IK & collision)",
                  fontsize=10)

    # ── 右：fail 原因分类 ──
    ax1 = fig.add_subplot(gs[0, 1])
    failed = [s for s in samples if not s.get("l2_pass")]
    cats: Dict[str, int] = {}
    for s in failed:
        reason = s.get("fail_reason", "") or "<empty>"
        # 截掉冗长信息，按"L1:" / "L2:" / 关键词归类
        if reason.startswith("L1:"):
            key = "L1: " + reason.split(":", 1)[1].strip().split("(")[0][:38]
        elif reason.startswith("L2:"):
            key = "L2: " + reason.split(":", 1)[1].strip().split("(")[0][:38]
        elif "post-check" in reason:
            key = "L2 post-check: " + reason.split("post-check:")[1].strip()[:30]
        else:
            key = reason[:50]
        cats[key] = cats.get(key, 0) + 1
    cats_sorted = sorted(cats.items(), key=lambda kv: -kv[1])
    cats_top = cats_sorted[:10]  # 最多 10 类
    if cats_top:
        keys = [k for k, _ in cats_top]
        vals = [v for _, v in cats_top]
        y = np.arange(len(keys))[::-1]
        ax1.barh(y, vals, color="#fc8d62",
                 edgecolor="white", height=0.7)
        ax1.set_yticks(y)
        ax1.set_yticklabels(keys, fontsize=7.8)
        ax1.set_xlabel("# of candidates rejected by this cause")
        ax1.set_title("Top failure causes among rejected candidates",
                      fontsize=10)
        for yi, v in enumerate(vals):
            ax1.text(v + max(vals) * 0.015, y[yi], str(v),
                     va="center", fontsize=8.5, color="#333")
    else:
        ax1.text(0.5, 0.5, "no failed samples",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    fig.suptitle(
        "Constraint funnel — the final layout is the unique survivor "
        "of a multi-stage hard-constraint filter",
        fontsize=11, y=1.04)
    _savefig(fig, out_dir, "F5_constraint_funnel")


# ══════════════════════════════════════════════════════════════════════════
#  Dashboard：把 5 张图拼成一张 PNG 方便快看
# ══════════════════════════════════════════════════════════════════════════
def make_dashboard(out_dir: str):
    """把已生成的 F1~F5 PNG 拼成一张 dashboard.png。"""
    try:
        from PIL import Image
    except ImportError:
        print("[Dashboard][SKIP] 缺少 Pillow（pip install Pillow），跳过 dashboard。")
        return
    names = ["F1_workspace_topdown", "F2_reachability_heatmap",
             "F3_score_decomposition", "F4_sample_distribution",
             "F5_constraint_funnel"]
    paths = [os.path.join(out_dir, n + ".png") for n in names]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return
    imgs = [Image.open(p) for p in paths]
    max_w = max(im.width for im in imgs)
    # 等宽缩放
    imgs_resized = []
    for im in imgs:
        if im.width != max_w:
            ratio = max_w / im.width
            im = im.resize((max_w, int(im.height * ratio)),
                           Image.LANCZOS)
        imgs_resized.append(im)
    total_h = sum(im.height for im in imgs_resized) + 8 * (len(imgs_resized) - 1)
    dash = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in imgs_resized:
        dash.paste(im, (0, y))
        y += im.height + 8
    out_path = os.path.join(out_dir, "dashboard.png")
    dash.save(out_path, optimize=True)
    print(f"[Dashboard] 写出 {os.path.relpath(out_path)}")


# ══════════════════════════════════════════════════════════════════════════
#  Entry
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-grade layout diagnostics figures.")
    parser.add_argument("--layout", required=True,
                        help="path to .layout file produced by "
                             "find_optimal_layout")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: alongside layout)")
    parser.add_argument("--grid-res", type=int, default=14,
                        help="F2 reachability grid resolution (default 14, "
                             "higher = finer / slower)")
    parser.add_argument("--skip-heatmap", action="store_true",
                        help="skip F2 (saves ~3-8 min)")
    parser.add_argument("--cache-dir", default=None,
                        help="F2 cache dir (default: _output/diagnostics_cache)")
    args = parser.parse_args()

    if not os.path.isfile(args.layout):
        print(f"[ERROR] layout 文件不存在: {args.layout}")
        sys.exit(1)

    layout = WorkspaceLayout.load(args.layout)
    diag, has_samples = _load_diag(args.layout)

    task_name = diag.get("task_name", "layout")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.layout), "diagnostics_figs", task_name)
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(args.layout), "diagnostics_cache")

    print("=" * 64)
    print(f"  Layout Diagnostics Plotter")
    print(f"  layout    : {os.path.relpath(args.layout)}")
    if has_samples:
        print(f"  diag JSON : OK (n_samples={len(diag.get('all_samples', []))})")
    else:
        print(f"  diag JSON : <missing>   F3/F4/F5 will be SKIPPED")
    print(f"  out_dir   : {os.path.relpath(out_dir)}")
    print("=" * 64)

    t_all = time.time()
    print("\n[F1] Workspace top-down …")
    plot_workspace_topdown(layout, diag, out_dir)

    if not args.skip_heatmap:
        print("\n[F2] Reachability heatmap …")
        plot_reachability_heatmaps(
            layout, diag, out_dir,
            grid_res=args.grid_res, cache_dir=cache_dir)
    else:
        print("\n[F2] skipped (--skip-heatmap)")

    if has_samples:
        print("\n[F3] Score decomposition + normalisation …")
        plot_score_decomposition(diag, out_dir)

        print("\n[F4] Sample distribution …")
        plot_sample_distribution(diag, out_dir)

        print("\n[F5] Constraint funnel …")
        plot_constraint_funnel(diag, out_dir)
    else:
        print("\n[F3/F4/F5] skipped — re-run find_optimal_layout 以生成诊断 JSON。")

    print("\n[Dashboard] Combining into dashboard.png …")
    make_dashboard(out_dir)

    print(f"\n[OK] 全部完成，用时 {time.time() - t_all:.1f}s。")
    print(f"     图片目录: {os.path.relpath(out_dir)}")


if __name__ == "__main__":
    main()
