"""The cost of one arrow of the state graph, drawn: Euclidean distances (chain cost) (2026-11-01).

step(S, last, p) = min over x in D(S, p) of
    |above pick(x) - above goal(last)|  (transit, from HOME for the first part)
  + LIFT (descend, grasp, lift) + |above goal(p) - above pick(x)| (transfer) + LIFT (insert)
"above" = + LIFT in z; points = part origins (BLS cost). Panels:
  (a) the formula on a side-view sketch, with the numbers of one arrow
  (b, c) same arrow {A,B,C} --AB->, last = A vs last = C: same best x, other start, other cost
  (d) the minimum over D: every table position of AB with its cost (last = C), positions not in D crossed
  (e, f) same part BC, S = {B,C} vs S = {A,B,C,CA}: D shrinks, the best x moves away, the cost grows
(b, c, e, f) are plan views with the true coordinates: installed S, p's goal (ghost) and its best
pose x (solid footprint), D (green dots, red crosses = removed), the two Euclidean segments labelled.

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_step_cost
"""
from __future__ import annotations

import sys

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .joint import UP, Relaxed  # noqa: E402
from .lazy import setup  # noqa: E402
from .run import LIFT, OUT, log  # noqa: E402
from .viz_domain import COLORS, SHORT  # noqa: E402

EX = [(("post_A", "post_B", "post_C"), "post_A", "beam_AB"),
      (("post_A", "post_B", "post_C"), "post_C", "beam_AB"),
      (("post_B", "post_C"), "post_C", "beam_BC"),
      (("beam_CA", "post_A", "post_B", "post_C"), "beam_CA", "beam_BC")]
TRANSIT, TRANSFER, VERT = "#2978d5", "#d63a99", "#262626"
KEY = ["A", "B", "C", "AB", "BC", "CA"]
nm = lambda S: "{" + ",".join(sorted((SHORT[q] for q in S), key=KEY.index)) + "}"


def main():
    pb, O, geo, ls, orders, *_ = setup("triangular_frame6", "0.373,0.0,0.0", 8)
    rx = Relaxed(ls, orders, "chain")
    sc, cd = pb.scene, ls.cd

    def evaluate(S, last, p):
        S = tuple(sorted(S))
        xs = rx.domain(S, p)
        sp = cd["start"][p][xs]
        frm = cd["home"] if last is None else cd["goal"][last] + UP
        tr = np.linalg.norm(sp + UP - frm, axis=1)
        tf = np.linalg.norm(cd["goal"][p] + UP - (sp + UP), axis=1)
        c = tr + LIFT + tf + LIFT
        i = int(np.argmin(c))
        assert abs(c[i] - rx.step(S, last, p)) < 1e-9
        return dict(S=S, last=last, p=p, frm=frm, xs=xs, sp=sp, cost=c, pick=sp[i], key=O["poses"][p][xs[i]],
                    transit=float(tr[i]), transfer=float(tf[i]), step=float(c[i]), goal=cd["goal"][p],
                    allx=cd["start"][p], removed=np.setdiff1d(np.arange(len(cd["start"][p])), xs))
    ex = [evaluate(*e) for e in EX]
    for e in ex:
        log(f"{nm(e['S'])} --{SHORT[e['p']]}-> last {SHORT[e['last']]}: |D| {len(e['xs'])}/{len(e['allx'])},"
            f" best {e['key']}: transit {e['transit']:.4f} + {LIFT} + transfer {e['transfer']:.4f} + {LIFT} = {e['step']:.4f}")
    compose(ex, sc)


def compose(ex, sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from shapely.geometry import MultiPoint
    from .viz_lattice import INK, INK2, SURF
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    hexc = lambda c: "#%02x%02x%02x" % tuple(int(255 * v) for v in c)
    up = np.array([0, 0, LIFT])
    v3 = lambda a: f"({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})"
    bb = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    hull = {q: MultiPoint((np.asarray(sc.meshes[q].vertices) @ sc.goal[q][:3, :3].T + sc.goal[q][:3, 3])[:, :2]).convex_hull
            for q in ("base",) + tuple(sc.ids)}
    fig = plt.figure(figsize=(21, 15.5), facecolor=SURF)
    W, H = 0.31, 0.40
    pos = [(0.015, 0.495), (0.35, 0.495), (0.685, 0.495), (0.015, 0.03), (0.35, 0.03), (0.685, 0.03)]

    # (a) the formula on a side-view sketch, numbers of ex[1]
    e = ex[1]
    ax = fig.add_axes([pos[0][0], pos[0][1], W, H])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 0.5, fc="#efe8da", ec="none"))
    pF, pPu, pP, pGu, pG = (0.6, 4.3), (3.6, 3.1), (3.6, 1.0), (8.2, 3.1), (8.2, 1.6)
    ax.plot(*zip(pF, pPu), color=TRANSIT, lw=3)
    ax.plot(*zip(pPu, pP), color=VERT, lw=2.2)
    ax.plot(*zip(pPu, pGu), color=TRANSFER, lw=3)
    ax.plot(*zip(pGu, pG), color=VERT, lw=2.2)
    for (x, y), c in ((pF, TRANSIT), (pPu, VERT), (pP, VERT), (pGu, TRANSFER), (pG, TRANSFER)):
        ax.plot(x, y, "o", color=c, ms=8)
    ax.add_patch(Rectangle((3.1, 0.5), 1.0, 0.5, fc=hexc(COLORS[e["p"]]), ec="none"))
    ax.add_patch(Rectangle((7.7, 1.1), 1.0, 0.5, fc=hexc(COLORS[e["p"]]), ec="none", alpha=0.35))
    ax.text(pF[0], pF[1] + 0.3, "① 起点：上一个零件的目标点上方（第一个零件从 HOME 出发）", fontsize=9.5, color=INK)
    ax.text(pPu[0] + 0.15, pPu[1] + 0.22, "② 抓取点上方", fontsize=9.5, color=INK)
    ax.text(pP[0] + 0.2, pP[1] + 0.05, "③ 桌上抓取点（位置 x）", fontsize=9.5, color=INK)
    ax.text(pGu[0] - 0.5, pGu[1] + 0.22, "④ 目标点上方", fontsize=9.5, color=INK)
    ax.text(pG[0] + 0.2, pG[1] - 0.25, "⑤ 目标点", fontsize=9.5, color=INK)
    ax.text(1.75, 3.25, "空手 = |② − ①|", color=TRANSIT, fontsize=11, bbox=bb)
    ax.text(3.75, 1.9, "0.10", color=VERT, fontsize=11, bbox=bb)
    ax.text(5.0, 2.55, "搬运 = |④ − ②|", color=TRANSFER, fontsize=11, bbox=bb)
    ax.text(8.35, 2.3, "0.10", color=VERT, fontsize=11, bbox=bb)
    F, P, G = e["frm"], e["pick"], e["goal"]
    d1, d2 = P + up - F, G + up - (P + up)
    ax.text(0.0, 9.35, "(a) 一条箭头的代价 = 四段长度之和，在 D(S, p) 里取最小的桌面位置 x（示意，不按比例）",
            fontsize=11.5, color=INK, va="top")
    ax.text(0.0, 8.75,
            f"例 (c)：{nm(e['S'])} —装 AB→，上一个 = C，最省的 x = ({P[0]:.3f}, {P[1]:.3f})\n"
            f"① = C 的目标 + 0.10 = {v3(F)}\n② = x + 0.10 = {v3(P + up)}\n④ = AB 的目标 + 0.10 = {v3(G + up)}\n"
            f"空手 = √(({d1[0]:+.3f})² + ({d1[1]:+.3f})² + ({d1[2]:+.3f})²) = {e['transit']:.4f}\n"
            f"搬运 = √(({d2[0]:+.3f})² + ({d2[1]:+.3f})² + ({d2[2]:+.3f})²) = {e['transfer']:.4f}\n"
            f"step = {e['transit']:.4f} + 0.10 + {e['transfer']:.4f} + 0.10 = {e['step']:.4f}",
            fontsize=9.8, color=INK2, va="top", linespacing=1.55)

    X0, X1, Y0, Y1 = 0.10, 0.52, -0.265, 0.285

    def base_plan(ax, e, show_S=True):
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fc="#f3efe6", ec="none", zorder=0))
        ax.add_patch(Rectangle((-0.12, -0.12), 0.24, 0.24, fc="#e4e2dd", ec=INK2, lw=0.6, ls="--", zorder=1))
        ax.text(0.112, 0.0, "机械臂\n禁区", fontsize=8, color=INK2, va="center")
        ax.fill(*hull["base"].exterior.xy, color="#cfd6e3", lw=0, zorder=1)
        for q in sc.ids:
            if q in e["S"]:
                ax.fill(*hull[q].exterior.xy, color=hexc(COLORS[q]), alpha=0.9, lw=0, zorder=2)
            elif q == e["p"]:
                ax.fill(*hull[q].exterior.xy, color=hexc(COLORS[q]), alpha=0.35, lw=1.2, ls="--",
                        ec=hexc(COLORS[q]), zorder=2)
            else:
                ax.fill(*hull[q].exterior.xy, fc="none", ec="#b9b8b1", lw=0.8, ls=":", zorder=2)
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y0, Y1)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(0.1, 0.55, 0.1))
        ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(color="white", lw=0.8, zorder=0.5)
        for s in ax.spines.values():
            s.set_visible(False)

    def plan(slot, e, title):
        ax = fig.add_axes([pos[slot][0] + 0.01, pos[slot][1] + 0.06, W - 0.01, H - 0.095])
        base_plan(ax, e)
        dxy = {tuple(v) for v in np.round(e["sp"][:, :2], 3)}
        rem = {tuple(v) for v in np.round(e["allx"][e["removed"]][:, :2], 3)} - dxy
        ax.scatter(*np.array(sorted(dxy)).T, s=16, c="#2f8f45", zorder=3, lw=0)
        if rem:
            ax.scatter(*np.array(sorted(rem)).T, marker="x", s=26, c="#c0392b", lw=1.1, zorder=3)
        ax.add_patch(Polygon(np.asarray(sc.polygons[e["p"]][e["key"]].exterior.coords), fc=hexc(COLORS[e["p"]]),
                             ec=INK, lw=0.8, zorder=4))
        F, P, G = e["frm"], e["pick"], e["goal"]
        ax.annotate("", P[:2], F[:2], arrowprops=dict(arrowstyle="-|>", color=TRANSIT, lw=2.4, mutation_scale=14), zorder=5)
        ax.annotate("", G[:2], P[:2], arrowprops=dict(arrowstyle="-|>", color=TRANSFER, lw=2.4, mutation_scale=14), zorder=5)
        ax.plot(*F[:2], "o", ms=9, mfc="white", mec=TRANSIT, mew=2.2, zorder=6)
        ax.plot(*P[:2], "o", ms=6, color=VERT, zorder=6)
        ax.plot(*G[:2], "o", ms=7, color=TRANSFER, zorder=6)

        callouts = [(f"空手 {e['transit']:.4f}", (F[:2] + P[:2]) / 2, TRANSIT),
                    (f"搬运 {e['transfer']:.4f}", (P[:2] + G[:2]) / 2, TRANSFER),
                    (f"起点 = {SHORT[e['last']]} 的目标", F[:2], TRANSIT),
                    ("x = 最省的桌面位置", P[:2], VERT)]
        for j, (text, tgt, color) in enumerate(callouts):
            ax.annotate(text, tgt, xytext=(0.115, -0.165 - 0.026 * j), fontsize=9.5, color=color, va="center",
                        bbox=bb, zorder=7, arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.8,
                                                           shrinkA=0, shrinkB=3))
        n_d, n_a = len(dxy), len(dxy | rem)
        ax.set_title(f"{title}\n箭头 {nm(e['S'])} —装 {SHORT[e['p']]}→，上一个 = {SHORT[e['last']]}；"
                     f"D 里 {n_d} / {n_a} 个桌面位置",
                     fontsize=10.5, color=INK, loc="left")
        d1, d2 = P + up - F, G + up - (P + up)
        fig.text(pos[slot][0] + 0.012, pos[slot][1] + 0.005,
                 f"空手 = √(({d1[0]:+.3f})² + ({d1[1]:+.3f})² + ({d1[2]:+.3f})²) = {e['transit']:.4f}\n"
                 f"搬运 = √(({d2[0]:+.3f})² + ({d2[1]:+.3f})² + ({d2[2]:+.3f})²) = {e['transfer']:.4f}\n"
                 f"step = {e['transit']:.4f} + 0.10 + {e['transfer']:.4f} + 0.10 = {e['step']:.4f}",
                 fontsize=9.8, color=INK2, linespacing=1.5)
    plan(1, ex[0], "(b) 同一条箭头，上一个 = A：起点就在旁边")
    plan(2, ex[1], "(c) 同一条箭头，上一个 = C：x 不变，起点变远，空手变长")
    plan(4, ex[2], "(e) 装 BC，已装 {B,C}：D 大，可以放在目标旁边")
    plan(5, ex[3], "(f) 装 BC，已装 {A,B,C,CA}：近处位置被删，只能放远处")

    # (d) the minimum over D (example c): every position with its cost
    e = ex[1]
    ax = fig.add_axes([pos[3][0] + 0.01, pos[3][1] + 0.06, W - 0.03, H - 0.12])
    tc, ts = sc.table_center, sc.table_size
    base_plan(ax, e)
    ax.set_xlim(tc[0] - ts[0] / 2, tc[0] + ts[0] / 2)
    ax.set_ylim(tc[1] - ts[1] / 2, tc[1] + ts[1] / 2)
    best = {}
    for xy, c in zip(map(tuple, np.round(e["sp"][:, :2], 3)), e["cost"]):
        best[xy] = min(best.get(xy, np.inf), c)
    rem = {tuple(v) for v in np.round(e["allx"][e["removed"]][:, :2], 3)} - set(best)
    if rem:
        ax.scatter(*np.array(sorted(rem)).T, marker="x", s=40, c="#c0392b", lw=1.3, zorder=3)
    bxy, bc = np.array(list(best)), np.array(list(best.values()))
    s_ = ax.scatter(*bxy.T, c=bc, cmap="viridis_r", s=60, zorder=4, edgecolors="white", linewidths=0.6)
    for (x, y), c in zip(bxy, bc):
        ax.text(x, y - 0.02, f"{c:.2f}", fontsize=6.5, ha="center", va="top", color=INK2, zorder=4)
    ax.scatter(*e["pick"][:2], marker="*", s=300, c="#d03b3b", edgecolors="white", zorder=5)
    ax.plot(*e["frm"][:2], "o", ms=9, mfc="white", mec=TRANSIT, mew=2.2, zorder=6)
    cb = fig.colorbar(s_, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("这个位置的 step（m）", fontsize=9)
    ax.set_title("(d) “取最小”：例 (c) 里 D 的每个桌面位置都算一遍四段之和（整张桌子）\n"
                 "数字 = 该位置的 step，红星 = 最小 = 这条箭头的代价，红叉 = 不在 D 里",
                 fontsize=10.5, color=INK, loc="left")
    fig.text(pos[3][0] + 0.012, pos[3][1] + 0.005,
             f"D 里 {len(best)} 个位置，最小的是 ({e['pick'][0]:.3f}, {e['pick'][1]:.3f})：{e['step']:.4f}。"
             f"\n同一个 x 也是 (b) 的最小（0.4720），因为两者只差在空手那一段的起点",
             fontsize=9.8, color=INK2, linespacing=1.5)
    fig.text(0.015, 0.975, "网上一条箭头（边）的代价：欧氏距离之和，在可用桌面位置 D(S, p) 里取最小", fontsize=15, color=INK)
    fig.text(0.015, 0.957, "蓝 = 空手（上一个零件的目标上方 → 抓取点上方），品红 = 搬运（抓取点上方 → 目标点上方），"
                           "各再加 0.10 m 的抓取和插入。“上方” = z + 0.10 m，点 = 零件原点。", fontsize=10.5, color=INK2)
    fig.text(0.015, 0.941, "(b)(c)(e)(f) 是俯视图，坐标单位 m；实色 = 已装零件，虚线半透明 = 这一步零件的目标，"
                           "带黑边的 = 它在桌上最省的摆放；绿点 = D 里的桌面位置，红叉 = 被已装零件删掉的位置", fontsize=10.5, color=INK2)
    f = OUT / "step_cost_triangular_frame6.png"
    fig.savefig(f, dpi=110, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
