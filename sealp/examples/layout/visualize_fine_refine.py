# -*- coding: utf-8 -*-
"""
可视化说明 Phase B "由粗到细" 局部精修 (pattern refine) 的过程
==============================================================

对应脚本:
    find_optimal_initial_layout_tower_global.py 里的 ``_pattern_refine`` /
    ``_offsets`` (Phase B)。

Phase B 到底在做什么 (与本脚本一一对应):
    1. 选出得分最高的前 K 个布局作为 elite;
    2. 对每个 elite, 逐个"自由零件"的 (x, y) 位置做小步长探测:
       方向由 ``_offsets(step, diagonal)`` 给出 —— 默认 ±x / ±y 四个方向,
       ``--diagonal`` 再加 4 个对角;
    3. 若某个探测点 L2 通过且 layout_score 变高, 就"贪心接受"(保留新位置)
       并把锚点移过去;
    4. 某个步长在一轮里没有任何改进 -> 认为该尺度收敛, 换成更小步长继续
       (coarse -> fine)。

本脚本用一个**合成的 score(x, y) 曲面**代替真实的 layout_score
(真实评分需要机器人/网格/碰撞检测), 但**搜索逻辑与真实脚本完全一致**:
同样的 ``_offsets``、同样的贪心接受、同样的"无改进即缩小步长"。这样就能在
不连硬件、不加载任何模型的情况下, 直观看清"由粗到细"的爬坡轨迹。

真实脚本里会对**每个自由零件**依次这样精修 (round-robin); 这里为看得清楚
只演示**单个零件**的 (x, y) 精修 —— 多个零件只是把同一套逻辑重复施加。

用法
----
# 生成静态图 (默认步长 0.03,0.015,0.008, 与脚本默认一致), 保存并显示:
python -m sealp.examples.layout.visualize_fine_refine

# 换步长 / 加对角方向 / 改起点 / 改每步最大轮数:
python -m sealp.examples.layout.visualize_fine_refine \
    --steps 0.04,0.02,0.01,0.005 --diagonal --rounds 3 --start 0.10,-0.30

# 只存图不弹窗 (无显示环境时):
python -m sealp.examples.layout.visualize_fine_refine --no-show
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def _setup_cjk_font() -> None:
    """让 matplotlib 能正常显示中文, 避免标题/坐标轴出现方块乱码。

    按优先级挑一个系统里真实存在的中文字体 (Windows 常见: 微软雅黑/黑体/宋体;
    也兼顾 macOS/Linux 常见字体)。同时修复负号 '-' 显示为方块的问题。
    """
    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",  # Windows
        "PingFang SC", "Hiragino Sans GB", "STHeiti",                 # macOS
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",  # Linux
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    picked = [name for name in candidates if name in installed]
    if picked:
        plt.rcParams["font.sans-serif"] = picked + list(plt.rcParams.get("font.sans-serif", []))
        print(f"[fine-viz] 中文字体 = {picked[0]}")
    else:
        print("[fine-viz] WARN: 未找到中文字体, 中文可能仍显示为方块。"
              "可安装 Microsoft YaHei / Noto Sans CJK 后重试。")
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ============================================================================
#  与真实脚本一致的算法零件
# ============================================================================
def _offsets(step: float, diagonal: bool) -> List[Tuple[float, float]]:
    """完全照搬 find_optimal_initial_layout_tower_global.py 里的 _offsets。"""
    base = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]
    if diagonal:
        base += [(step, step), (step, -step), (-step, step), (-step, -step)]
    return base


def _synthetic_score(xy: np.ndarray, optimum: np.ndarray) -> float:
    """一个平滑的合成 "layout_score(x, y)" 曲面 (越高越好)。

    用两项之和模拟真实评分的地形: 一个主峰(全局最优附近) + 轻微起伏,
    好让"由粗到细"能先大步靠近主峰、再小步微调到峰顶。
    """
    d = xy - optimum
    main = np.exp(-np.sum(d ** 2) / (2.0 * 0.09 ** 2))       # 主高斯峰
    ripple = 0.06 * np.cos(xy[0] * 40.0) * np.cos(xy[1] * 40.0)  # 细小起伏
    return float(main + ripple)


def pattern_refine_trace(
    start_xy: np.ndarray,
    optimum: np.ndarray,
    steps: List[float],
    rounds: int,
    diagonal: bool,
) -> Dict:
    """复现 _pattern_refine 的贪心坐标搜索, 并记录全过程用于画图。

    返回一个 trace 字典:
        accepted   : [(xy, score, step)]   历次被接受的点(含起点)
        probes     : [(xy, score, step, accepted_bool)]  所有试探点
        step_marks : [(eval_idx, step)]    步长切换发生在第几次评估
    """
    eps = 1e-4
    best_xy = np.asarray(start_xy, dtype=float).copy()
    best_score = _synthetic_score(best_xy, optimum)

    accepted: List[Tuple[np.ndarray, float, float]] = [(best_xy.copy(), best_score, steps[0])]
    probes: List[Tuple[np.ndarray, float, float, bool]] = []
    step_marks: List[Tuple[int, float]] = []
    n_eval = 0

    for step in steps:
        step_marks.append((n_eval, step))
        for _r in range(max(1, rounds)):
            improved = False
            anchor = best_xy.copy()
            for dx, dy in _offsets(step, diagonal):
                trial = anchor + np.array([dx, dy], dtype=float)
                s = _synthetic_score(trial, optimum)
                n_eval += 1
                is_acc = s > best_score + eps
                probes.append((trial.copy(), s, step, is_acc))
                if is_acc:
                    best_xy = trial.copy()
                    best_score = s
                    anchor = best_xy.copy()
                    improved = True
                    accepted.append((best_xy.copy(), best_score, step))
            if not improved:
                break  # 该步长收敛 -> 进入更小步长

    return {
        "accepted": accepted,
        "probes": probes,
        "step_marks": step_marks,
        "best_xy": best_xy,
        "best_score": best_score,
        "n_eval": n_eval,
    }


# ============================================================================
#  画图
# ============================================================================
def _plot(trace: Dict, start_xy: np.ndarray, optimum: np.ndarray,
          steps: List[float], out_path: str, show: bool) -> None:
    accepted = trace["accepted"]
    probes = trace["probes"]

    # 每个步长一种颜色 (粗->细: 暖->冷)。
    step_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))
    color_of = {s: step_colors[i] for i, s in enumerate(steps)}

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- 左: score 等高线 + 搜索轨迹 ----
    acc_pts = np.array([a[0] for a in accepted])
    pad = max(steps) * 4 + 0.05
    lo = np.minimum(acc_pts.min(axis=0), optimum) - pad
    hi = np.maximum(acc_pts.max(axis=0), optimum) + pad
    gx = np.linspace(lo[0], hi[0], 240)
    gy = np.linspace(lo[1], hi[1], 240)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            GZ[i, j] = _synthetic_score(np.array([GX[i, j], GY[i, j]]), optimum)

    cf = ax.contourf(GX, GY, GZ, levels=30, cmap="Greys", alpha=0.85)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="合成 layout_score (越高越好)")

    # 所有试探点: 接受=实心大点(按步长着色), 拒绝=灰色小叉
    for xy, s, step, is_acc in probes:
        if is_acc:
            ax.scatter(xy[0], xy[1], s=55, color=color_of[step],
                       edgecolors="k", linewidths=0.6, zorder=4)
        else:
            ax.scatter(xy[0], xy[1], s=16, color="0.45", marker="x",
                       alpha=0.6, zorder=3)

    # 接受轨迹连线 (按接受时的步长着色)
    for k in range(1, len(accepted)):
        p0 = accepted[k - 1][0]
        p1 = accepted[k][0]
        step = accepted[k][2]
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color=color_of[step],
                                    lw=2.0), zorder=5)

    ax.scatter(*start_xy, s=180, marker="*", color="tab:red",
               edgecolors="k", zorder=6, label="起点 (elite 初始 XY)")
    ax.scatter(*optimum, s=160, marker="X", color="tab:green",
               edgecolors="k", zorder=6, label="(合成)最优")
    ax.scatter(*trace["best_xy"], s=90, marker="o", facecolors="none",
               edgecolors="tab:red", linewidths=2.2, zorder=6, label="精修终点")

    # 步长图例
    handles = [ax.scatter([], [], s=55, color=color_of[s], edgecolors="k",
                          label=f"接受: step={s:g} m") for s in steps]
    handles.append(ax.scatter([], [], s=16, color="0.45", marker="x",
                              label="拒绝(未提升)"))
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("零件 X (m)")
    ax.set_ylabel("零件 Y (m)")
    ax.set_title("Phase B 由粗到细坐标模式搜索: 单个零件 XY 精修轨迹\n"
                 "(箭头=被接受的移动; 颜色=当时步长, 粗->细)")
    ax.set_aspect("equal", adjustable="box")

    # ---- 右: 评估序 vs 当前最好分数 (贪心单调上升 + 步长切换标注) ----
    # 按 probes 的评估顺序重建 "当前最优" 曲线, 命中接受点时抬升。
    best = _synthetic_score(start_xy, optimum)
    xs = [0]
    ys = [best]
    for i, (xy, s, step, is_acc) in enumerate(probes, start=1):
        if is_acc:
            best = s
        xs.append(i)
        ys.append(best)
    ax2.plot(xs, ys, color="tab:blue", lw=2.0, label="当前最优分数 (贪心, 单调不降)")
    ax2.scatter([p for p in range(1, len(probes) + 1)],
                [pr[1] for pr in probes],
                s=10, color="0.6", alpha=0.5, label="每次试探的分数")

    # 步长切换竖线
    for eval_idx, step in trace["step_marks"]:
        ax2.axvline(eval_idx + 0.5, color=color_of[step], ls="--", lw=1.3)
        ax2.text(eval_idx + 0.5, ax2.get_ylim()[0], f" step={step:g}",
                 rotation=90, va="bottom", ha="left", fontsize=8,
                 color=color_of[step])
    ax2.set_xlabel("评估次数 (eval)")
    ax2.set_ylabel("分数")
    ax2.set_title("贪心接受: 分数只升不降; 虚线=切换到更小步长")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Phase B pattern refine 说明  |  步长(粗->细)={[f'{s:g}' for s in steps]}  "
        f"|  总评估={trace['n_eval']}  "
        f"|  分数 {accepted[0][1]:.4f} -> {trace['best_score']:.4f}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"[fine-viz] 图已保存 -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _parse_floats(spec: str) -> List[float]:
    return [float(s) for s in str(spec).replace(",", " ").split() if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="可视化说明 global 脚本 Phase B '由粗到细' 局部精修过程。")
    ap.add_argument("--steps", default="0.03,0.015,0.008",
                    help="精修步长(米), 由粗到细, 逗号分隔。默认与脚本一致。")
    ap.add_argument("--rounds", type=int, default=2,
                    help="每个步长最多扫描轮数 (与 --global-refine-rounds 对应)。")
    ap.add_argument("--diagonal", action="store_true",
                    help="额外尝试 4 个对角方向 (与 --global-refine-diagonal 对应)。")
    ap.add_argument("--start", default="0.10,-0.28",
                    help="起点 (elite 初始 XY), 'x,y' 米。")
    ap.add_argument("--optimum", default="0.23,-0.35",
                    help="合成最优位置 'x,y' 米 (默认取桌面中心附近)。")
    ap.add_argument("--out", default=None, help="输出 PNG 路径 (默认 _output/fine_refine_demo.png)")
    ap.add_argument("--no-show", action="store_true", help="只保存不弹窗")
    args = ap.parse_args()

    steps = _parse_floats(args.steps)
    if not steps:
        ap.error("--steps 解析为空")
    start_xy = np.array(_parse_floats(args.start)[:2], dtype=float)
    optimum = np.array(_parse_floats(args.optimum)[:2], dtype=float)

    if args.no_show:
        matplotlib.use("Agg")

    _setup_cjk_font()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_output", "fine_refine_demo.png")

    print(f"[fine-viz] steps(粗->细) = {steps}, rounds={args.rounds}, diagonal={args.diagonal}")
    print(f"[fine-viz] start={start_xy.tolist()}, optimum={optimum.tolist()}")

    trace = pattern_refine_trace(start_xy, optimum, steps, args.rounds, args.diagonal)
    print(f"[fine-viz] 总评估 {trace['n_eval']} 次, "
          f"分数 {trace['accepted'][0][1]:.4f} -> {trace['best_score']:.4f}, "
          f"接受移动 {len(trace['accepted']) - 1} 次")

    _plot(trace, start_xy, optimum, steps, out_path, show=not args.no_show)


if __name__ == "__main__":
    main()
