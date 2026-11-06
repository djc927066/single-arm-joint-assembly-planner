"""Which arrows of the frame's state graph the lower bound prunes (chain cost) (2026-11-01).

For every arrow (installed set S --p--> S + {p}) the relaxed cost of the cheapest complete order
through it: every part at its cheapest pose usable under its installed set (no table conflicts),
min over the parts that can come right before p:
    through(S, p) = min_last  F(S, last) + step(S, last, p) + H(S + {p}, p)
F = cheapest relaxed prefix from HOME, H = cheapest relaxed completion (joint.Relaxed, Held-Karp).
An arrow whose value >= the optimum U (3.0509, the exact joint search) cannot be on a better plan:
pruned before any layout is searched for it. Drawn on the same graph as viz_lattice.py.

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_prune
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .joint import Relaxed
from .lazy import setup
from .run import OUT, log
from .viz_lattice import EDGE, GRID, HI, INK, INK2, KEYORD, NODE, SHORT, SURF

CUT = "#d03b3b"


def main():
    pb, O, geo, ls, orders, *_ = setup("triangular_frame6", "0.373,0.0,0.0", 8)
    rx = Relaxed(ls, orders, "chain")
    U = json.loads((OUT / "conflict_chain_triangular_frame6_k8.json").read_text(encoding="utf-8"))["cost"]
    full = tuple(sorted(O["ids"]))
    F = {((), None): 0.0}
    for S in sorted(set(rx.nxt), key=len):
        for (T, last), v in list(F.items()):
            if T != S:
                continue
            for p in rx.nxt[S]:
                S2 = tuple(sorted(S + (p,)))
                c = v + rx.step(S, last, p)
                if c < F.get((S2, p), np.inf):
                    F[S2, p] = c
    through = {}
    for S in rx.nxt:
        for p in rx.nxt[S]:
            S2 = tuple(sorted(S + (p,)))
            through[S, p] = min((v + rx.step(S, last, p) + rx.h(S2, p) for (T, last), v in F.items() if T == S), default=np.inf)
    root = rx.h((), None)
    order_val = {o: rx.cost(o) for o in orders}
    log(f"root bound {root:.4f}, optimum U {U:.4f}; arrows below U: {sum(v < U - 1e-9 for v in through.values())} of {len(through)};"
        f" orders below U: {sum(v < U - 1e-9 for v in order_val.values())} of {len(orders)}")
    for (S, p), v in sorted(through.items(), key=lambda t: t[1]):
        log(f"  {'+'.join(SHORT[x] for x in S) or '{}':16s} --{SHORT[p]:2s}->  {v:.4f}  {'KEEP' if v < U - 1e-9 else 'pruned'}")
    plot({(frozenset(S), p): v for (S, p), v in through.items()}, root, U, min(order_val, key=order_val.get), order_val)


def plot(through, root, U, best, order_val, out="prune_triangular_frame6.png", what="chain 代价"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    edges = list(through)
    nodes = {frozenset()} | {S | {p} for S, p in edges}
    label = lambda S: "{" + ",".join(sorted((SHORT[p] for p in S), key=KEYORD.index)) + "}" if S else "{ }"
    layers = {}
    for S in nodes:
        layers.setdefault(len(S), []).append(S)
    pos = {}
    for k, Ss in layers.items():
        Ss.sort(key=lambda S: [KEYORD.index(SHORT[p]) for p in sorted(S, key=lambda x: KEYORD.index(SHORT[x]))])
        for i, S in enumerate(Ss):
            pos[S] = (k * 3.2, (len(Ss) - 1) / 2 - i)
    on_best = {(frozenset(best[:k]), best[k]) for k in range(len(best))}
    node_val = {S: min([v for (T, p), v in through.items() if T | {p} == S or T == S], default=root) for S in nodes}
    fig = plt.figure(figsize=(28, 14.5), facecolor=SURF)
    ax = fig.add_axes([0.01, 0.07, 0.98, 0.84])
    ax.set_facecolor(SURF)
    for S, p in edges:
        T = S | {p}
        (x0, y0), (x1, y1) = pos[S], pos[T]
        v = through[S, p]
        keep, hi = v < U - 1e-9, (S, p) in on_best
        col = HI if hi else (EDGE if keep else CUT)
        ax.add_patch(FancyArrowPatch((x0 + 0.78, y0), (x1 - 0.78, y1), arrowstyle="-|>", mutation_scale=16,
                                     lw=3.6 if hi else 1.6, color=col, alpha=1.0 if (hi or keep) else 0.45,
                                     linestyle="-" if (hi or keep) else (0, (4, 3)), zorder=3 if hi else 1))
        t = 0.45
        ax.text(x0 + 0.78 + t * (x1 - x0 - 1.56), y0 + t * (y1 - y0), f"{SHORT[p]} {v:.3f}", fontsize=11, fontweight="bold",
                color=HI if hi else (INK2 if keep else CUT), ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.92))
    for S, (x, y) in pos.items():
        hi = any((T, p) in on_best for (T, p) in edges if T == S or T | {p} == S) or S == frozenset()
        cut = node_val[S] >= U - 1e-9
        ax.add_patch(FancyBboxPatch((x - 0.78, y - 0.36), 1.56, 0.72, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=NODE, ec=HI if hi else (CUT if cut else GRID), lw=3.0 if hi else 1.4,
                                    alpha=0.5 if cut and not hi else 1.0, zorder=5))
        ax.text(x, y + 0.12, label(S), ha="center", va="center", fontsize=12.5 if len(S) < 6 else 11, fontweight="bold", color=INK, zorder=6,
                alpha=0.5 if cut and not hi else 1.0)
        ax.text(x, y - 0.17, f"最乐观 {node_val[S]:.3f}", ha="center", va="center", fontsize=11, fontweight="bold",
                color=CUT if cut and not hi else INK2, zorder=6)
    for k in sorted(layers):
        ax.text(k * 3.2, max(p[1] for S, p in pos.items() if len(S) == k) + 0.75, f"已装 {k} 个", ha="center",
                fontsize=14, fontweight="bold", color=INK2)
    ax.set_xlim(-1.0, 6 * 3.2 + 1.0)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.85, max(p[1] for p in pos.values()) + 1.15)
    ax.axis("off")
    n_keep = sum(v < U - 1e-9 for v in through.values())
    n_ord = sum(v < U - 1e-9 for v in order_val.values())
    ax.set_title(f"下界剪枝（{what}）：箭头上的数 = 经过这条箭头的所有顺序里，最乐观的总代价（不管零件互相挡）。"
                 f"真实最优 U = {U:.4f}\n"
                 f"数值 ≥ U 的箭头（红色虚线）不可能出现在更好的方案里，连布局都不用搜。"
                 f"30 条箭头剩 {n_keep} 条，48 个顺序剩 {n_ord} 个", fontsize=17, fontweight="bold", color=INK, loc="left")
    handles = [Line2D([], [], color=HI, lw=3.6, label=f"最优顺序 {'→'.join(SHORT[q] for q in best)}（最乐观 {root:.4f}，真实 {U:.4f}）"),
               Line2D([], [], color=EDGE, lw=1.2, label="最乐观值 < U：需要看"),
               Line2D([], [], color=CUT, lw=1.2, ls=(0, (4, 3)), alpha=0.6, label="最乐观值 ≥ U：剪掉")]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0.0, -0.07),
              prop=dict(size=13.5, weight="bold"))
    f = OUT / out
    fig.savefig(f, dpi=125, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
