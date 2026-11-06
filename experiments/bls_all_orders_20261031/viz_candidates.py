"""Candidate table poses of each part in the mainline (joint_search.Problem, default settings) (2026-11-01).

One panel per part, plan view of the work table, the candidate positions stage by stage:
  grid    the fine grid shared by every part (1.5 cm linspace, scene grid="fine")
  fit     positions where some resting orientation of the part fits (table, robot rectangle, assembly hull)
  HOME    positions lost because the arm at HOME would hit the part (Problem._drop_home_colliding)
  FPS     the pool * n_cand farthest-point positions (scene._select)
  kept    the first n_cand of them with a usable grasp in the IK catalogue (Problem: cat["pick"]),
          drawn with the footprint of their first orientation in SEALP order
Last panel: the kept positions of every part on one table (where they coincide).

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_candidates [--model triangular_frame6]
"""
from __future__ import annotations

import argparse
import sys

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .run import OUT, log  # noqa: E402
from .viz_lattice import GRID, INK, INK2, SURF  # noqa: E402

COLORS = {"post_A": "#2f8f45", "post_B": "#c73d3d", "post_C": "#3b5a94", "beam_AB": "#e08a2c",
          "beam_BC": "#c9a425", "beam_CA": "#9b5ca8"}
PALETTE = ["#2a78d6", "#d0573b", "#2f8f45", "#9b5ca8", "#c9a425", "#3ba3a3", "#c73d7a", "#6b6b6b", "#8a6d3b", "#4b4bb0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    a = ap.parse_args()
    from joint_search.problem import Problem
    from lazy_cost.scene import key_xy, load_scene
    from shapely.geometry import MultiPoint

    pb = Problem(a.model, center=tuple(float(v) for v in a.goal_pos.split(",")))
    sc = pb.scene
    fresh = load_scene(a.model, center=tuple(sc.center) if getattr(sc, "center", None) is not None else None,
                       sampler=sc.sampler, placements=sc.placements, grid=sc.grid, keepout=sc.keepout,
                       grid_step=sc.grid_step)        # the same scene before the HOME check
    xy = lambda keys: np.array([[float(v) for v in k.split(",")] for k in dict.fromkeys(keys)]).reshape(-1, 2)
    lo = sc.table_center[:2] - sc.table_size[:2] / 2 + 0.01
    hi = sc.table_center[:2] + sc.table_size[:2] / 2 - 0.01
    n = np.floor((hi - lo) / sc.grid_step + 1e-3).astype(int) + 1
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n[0]), np.linspace(lo[1], hi[1], n[1]))
    plan = {q: np.asarray(sc.meshes[q].vertices) @ sc.goal[q][:3, :3].T + sc.goal[q][:3, 3] for q in ("base",) + tuple(sc.ids)}
    site = MultiPoint(np.vstack(list(plan.values()))[:, :2]).convex_hull.buffer(0.01)
    col = {p: COLORS.get(p, PALETTE[i % len(PALETTE)]) for i, p in enumerate(sc.ids)}
    st = {}
    for p in sc.ids:
        fit0 = set(fresh.fit_xy[p])
        fit1 = set(sc.fit_xy[p])
        fps = list(dict.fromkeys(key_xy(k) for k, _ in sc.candidates[p]))
        kept_keys = {}
        for k in pb.cat["pick"][p]:
            kept_keys.setdefault(key_xy(k), k)                # first orientation (SEALP order) per position
        st[p] = dict(fit=xy(fit1), home=xy(fit0 - fit1), fps=xy(fps), kept=xy(kept_keys), keys=kept_keys,
                     n_fit0=len(fit0), n_fit=len(fit1), n_fps=len(fps), n_kept=len(kept_keys),
                     n_orient=len(pb.cat["pick"][p]))
        log(f"  {p:10s} grid {gx.size}  fit {len(fit0)}  -HOME {len(fit0 - fit1)}  FPS {len(fps)}  kept {len(kept_keys)}"
            f" positions ({st[p]['n_orient']} position x orientation)")
    allk = [set(st[p]["keys"]) for p in sc.ids]
    shared_all = set.intersection(*allk)
    log(f"kept positions shared by every part: {len(shared_all)}; union {len(set.union(*allk))}")
    plot(a.model, sc, pb, st, col, gx, gy, lo, hi, plan, site, shared_all)


def plot(model, sc, pb, st, col, gx, gy, lo, hi, plan, site, shared_all):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ids = list(sc.ids)
    nc = 4
    nr = -(-(len(ids) + 1) // nc)
    tc, ts = sc.table_center, sc.table_size
    w = 4.6
    fig, axs = plt.subplots(nr, nc, figsize=(nc * w, nr * (w * ts[1] / ts[0] + 0.55) + 1.6), facecolor=SURF)
    axs = np.atleast_2d(axs)

    def frame(ax, title):
        ax.set_facecolor(SURF)
        ax.add_patch(Rectangle(tc[:2] - ts[:2] / 2, ts[0], ts[1], fc="#f3efe6", ec=INK2, lw=1.0, zorder=0))
        ax.add_patch(Rectangle((-0.12, -0.12), 0.24, 0.24, fc="#e4e2dd", ec=INK2, lw=0.8, ls="--", zorder=1))
        ax.text(0, 0, "机械臂\n禁区", ha="center", va="center", fontsize=7, color=INK2, zorder=2)
        ax.add_patch(Polygon(np.asarray(site.exterior.coords), fc="#e4e2dd", ec=INK2, lw=0.8, ls="--", zorder=1))
        for q, v in plan.items():
            from shapely.geometry import MultiPoint
            h = MultiPoint(v[:, :2]).convex_hull
            if h.geom_type == "Polygon":
                ax.add_patch(Polygon(np.asarray(h.exterior.coords), fc="#cfccc5" if q == "base" else col.get(q, "#999"),
                                     ec="none", alpha=0.55 if q == "base" else 0.35, zorder=2))
        ax.scatter(gx.ravel(), gy.ravel(), s=0.6, c=GRID, zorder=1, linewidths=0)
        ax.set_xlim(tc[0] - ts[0] / 2 - 0.01, tc[0] + ts[0] / 2 + 0.01)
        ax.set_ylim(tc[1] - ts[1] / 2 - 0.01, tc[1] + ts[1] / 2 + 0.01)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")

    for i, p in enumerate(ids):
        ax = axs.flat[i]
        s, c = st[p], col[p]
        frame(ax, f"{p}：贴合 {s['n_fit0']} 点，HOME 去掉 {s['n_fit0'] - s['n_fit']}，"
                  f"FPS {s['n_fps']}，保留 {s['n_kept']}")
        ax.scatter(*s["fit"].T, s=4, c=c, alpha=0.35, linewidths=0, zorder=3)
        if len(s["home"]):
            ax.scatter(*s["home"].T, s=5, marker="x", c="#555", alpha=0.6, linewidths=0.6, zorder=3)
        ax.scatter(*s["fps"].T, s=14, facecolors="none", edgecolors=c, linewidths=0.8, zorder=4)
        for k in s["keys"].values():
            ax.add_patch(Polygon(np.asarray(sc.polygons[p][k].exterior.coords), fc=c, ec=c, alpha=0.10, lw=0.5, zorder=4))
        ax.scatter(*s["kept"].T, s=14, c=c, linewidths=0, zorder=5)
    ax = axs.flat[len(ids)]
    frame(ax, f"所有零件保留的位置叠在一起（{len(shared_all)} 个点所有零件都有）")
    for j, p in enumerate(ids):
        ax.scatter(*st[p]["kept"].T, s=34 - 4 * j, c=col[p], alpha=0.85, linewidths=0, zorder=5 + j, label=p)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2, bbox_to_anchor=(0.0, -0.01))
    for ax in list(axs.flat)[len(ids) + 1:]:
        ax.axis("off")
    handles = [Line2D([], [], ls="", marker="o", ms=1.6, color=GRID, label="共用细网格（1.5 cm）"),
               Line2D([], [], ls="", marker="o", ms=2.6, color="#777", alpha=0.35, label="零件放得下的点（任一朝向）"),
               Line2D([], [], ls="", marker="x", ms=3.5, color="#555", label="HOME 姿态的机械臂会撞：去掉"),
               Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="#777", label="最远点采样 80 个"),
               Line2D([], [], ls="", marker="o", ms=5, color="#777", label="最终保留（有可用抓取的前 40 个）+ 轮廓"),
               Rectangle((0, 0), 1, 1, fc="#e4e2dd", ec=INK2, ls="--", label="禁区：机械臂底座 / 装配区 +1 cm"),
               Rectangle((0, 0), 1, 1, fc="#cfccc5", alpha=0.55, label="底座与零件目标位置（俯视）")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"{model}：主线（joint_search.Problem 默认参数）每个零件的候选摆放位置。"
                 f"所有零件共用同一张网格，但各自筛选出自己的候选", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=2.2)
    f = OUT / f"candidates_{model}.png"
    fig.savefig(f, dpi=140, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
