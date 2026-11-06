"""Space-time view of a layout (2026-11-01): order = time, layout = space.

(a) exact optimum (conflict_<mode>_<model>.json): every part's footprint on the table, shaded by
    its install step (light = early, dark = late), assembly site marked
(b) the relaxed solution the prices settled on (price_star_<mode>_<model>.json), same encoding;
    footprint overlaps outlined in red
(c) price map: every candidate pose (centre) that ended with a price, shaded by price; the
    optimum's footprints outlined
(d) time chart in install order: on the table until picked (grey), then installed (blue);
    the "give-way" loss of each part from diag_gap (cheaper poses it gave up)

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_spacetime --model rack10_tall --mode chain
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .run import OUT, log

BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = ["#fde3cc", "#f9c08f", "#f29d55", "#e07b24", "#b85f12"]
INK, INK2, GRID, SURF, CRIT = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb", "#d03b3b"


def ramp(k, n):
    i = 0 if n <= 1 else round(k / (n - 1) * (len(BLUES) - 1))
    return BLUES[i]


def short(p):
    return p.replace("shelf_", "S").replace("post_", "P").replace("beam_", "B").replace("lp_", "L").replace("up_", "U")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rack10_tall")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="chain", choices=("bls", "chain"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly, Rectangle
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from joint_search.problem import Problem
    pb = Problem(a.model, center=tuple(float(v) for v in a.goal_pos.split(",")))
    sc = pb.scene
    opt = json.loads((OUT / f"conflict_{a.mode}_{a.model}_k{a.k}.json").read_text(encoding="utf-8"))
    pr = json.loads((OUT / f"price_star_{a.mode}_{a.model}_k{a.k}.json").read_text(encoding="utf-8"))
    dg = OUT / f"diag_gap_{a.mode}_{a.model}_k{a.k}.json"
    diag = json.loads(dg.read_text(encoding="utf-8"))["parts"] if dg.exists() else {}
    n = len(opt["order"])
    goal = np.mean([sc.goal[p][:3, 3] for p in opt["order"]], axis=0)
    tc, ts = np.asarray(sc.table_center), np.asarray(sc.table_size)

    def table(ax, title):
        ax.set_facecolor(SURF)
        ax.add_patch(Rectangle(tc[:2] - ts[:2] / 2, ts[0], ts[1], fill=False, ec=GRID, lw=1))
        ax.plot(*goal[:2], marker="X", ms=12, color=INK, zorder=6)
        ax.annotate("装配处", goal[:2], xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9, zorder=6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_xlabel("x (m)", color=INK2, fontsize=8)
        ax.set_ylabel("y (m)", color=INK2, fontsize=8)

    def layout(ax, order, lay, outline_overlaps=False):
        polys = {}
        for k, p in enumerate(order):
            P = sc.polygons[p][lay[p]]
            polys[p] = P
            ax.add_patch(MPoly(np.asarray(P.exterior.coords), closed=True, fc=ramp(k, n), ec=SURF, lw=2, alpha=0.95, zorder=3))
            c = P.centroid
            ax.text(c.x, c.y, f"{k + 1} {short(p)}", ha="center", va="center", fontsize=8, color=INK, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
        if outline_overlaps:
            ps = list(order)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    if polys[ps[i]].distance(polys[ps[j]]) < pb.gap:
                        for p in (ps[i], ps[j]):
                            ax.add_patch(MPoly(np.asarray(polys[p].exterior.coords), closed=True, fill=False,
                                               ec=CRIT, lw=2, zorder=5))
        return polys

    fig, axs = plt.subplots(2, 2, figsize=(14, 12), facecolor=SURF)
    # (a)
    table(axs[0, 0], f"(a) 精确最优布局  代价 {opt['cost']:.4f}\n数字 = 第几个装（浅 = 早装，深 = 晚装）")
    opt_polys = layout(axs[0, 0], opt["order"], opt["layout"])
    # (b)
    rel = pr["relaxed"]
    r_order, r_lay = [r["part"] for r in rel], {r["part"]: r["pose"] for r in rel}
    same_order = r_order == opt["order"]
    same_pose = sum(r_lay[p] == opt["layout"][p] for p in r_order)
    table(axs[0, 1], f"(b) 价格法松弛解（下界 {pr['summary']['price_bound']:.4f}）\n"
                     f"顺序与最优{'相同' if same_order else '不同'}；{same_pose}/{n} 个零件位置与最优相同；红框 = 占地重叠")
    layout(axs[0, 1], r_order, r_lay, outline_overlaps=True)
    # (c)
    ax = axs[1, 0]
    table(ax, "(c) 占用费地图：零件躺在这里会挡别人，要付的价格（越深越贵）\n黑框 = 最优布局")
    pts = [(sc.polygons[p][key].centroid, v) for p, d in pr["prices"].items() for key, v in d.items()]
    if pts:
        vmax = max(v for _, v in pts)
        for c, v in sorted(pts, key=lambda t: t[1]):
            i = min(len(ORANGE) - 1, int(v / vmax * (len(ORANGE) - 1) + 0.5))
            ax.scatter(c.x, c.y, s=60, color=ORANGE[i], edgecolors=SURF, linewidths=1.5, zorder=4)
        for i, col in enumerate(ORANGE):
            ax.scatter([], [], s=60, color=col, label=f"≈ {vmax * i / (len(ORANGE) - 1):.3f}")
        ax.legend(title="价格", fontsize=8, title_fontsize=8, frameon=False, loc="lower left")
    for p, P in opt_polys.items():
        ax.add_patch(MPoly(np.asarray(P.exterior.coords), closed=True, fill=False, ec=INK, lw=1, zorder=3))
    # (d)
    ax = axs[1, 1]
    ax.set_facecolor(SURF)
    order = opt["order"]
    for k, p in enumerate(order):
        y = n - 1 - k
        ax.barh(y, k + 0.5, left=0, height=0.62, color=GRID, edgecolor=SURF, linewidth=2)
        ax.barh(y, n - k - 0.5, left=k + 0.5, height=0.62, color=ramp(k, n), edgecolor=SURF, linewidth=2)
        loss = diag.get(p, {}).get("loss", 0.0)
        if loss > 1e-9:
            ax.text(n + 0.1, y, f"让路代价 +{loss:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{k + 1}. {p}" for k, p in reversed(list(enumerate(order)))], fontsize=8, color=INK)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([f"第{k + 1}步" for k in range(n)], fontsize=7, color=INK2, rotation=45)
    ax.set_xlim(0, n + 2.2)
    for s in ax.spines.values():
        s.set_color(GRID)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRID, label="躺在桌上（可能挡别人取件）"),
                       Patch(color=BLUES[5], label="已装在框架上（可能挡后续插入）")],
              fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2)
    ax.set_title("(d) 时间轴（最优顺序）：越晚装，在桌上躺得越久", fontsize=11, color=INK, loc="left")
    fig.suptitle(f"{a.model} · {a.mode} 代价：顺序 = 时间，布局 = 空间", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    f = OUT / f"viz_spacetime_{a.mode}_{a.model}.png"
    fig.savefig(f, dpi=130, facecolor=SURF)
    log(f"-> {f}  (relaxed order same as optimum: {same_order}; same poses {same_pose}/{n})")


if __name__ == "__main__":
    sys.exit(main())
