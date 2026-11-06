"""State graph (lattice) of the triangular frame: installed sets = nodes, one more part = edge (2026-11-01).

Left: every reachable installed set, one column per number of installed parts; an arrow = install one
part (its label); the number in a node = how many full orders pass through it; the Chen / chain-cost
best order highlighted. Right: the 48 orders = all paths from {} to the full set.

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_lattice
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .run import OUT, log

INK, INK2, GRID, SURF, EDGE, HI, NODE = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb", "#b9b8b1", "#2a78d6", "#f0efec"
SHORT = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}
KEYORD = ["A", "B", "C", "AB", "BC", "CA"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    d = json.loads((OUT / "sga_orders_triangular_frame6.json").read_text(encoding="utf-8"))
    edges = [(frozenset(e["installed"]), e["part"]) for e in d["edges"]]
    best = tuple(d["best"][0])
    full = frozenset(SHORT)
    nodes = {frozenset()} | {S | {p} for S, p in edges}
    out = {}
    for S, p in edges:
        out.setdefault(S, []).append(p)
    n_from, n_to = {full: 1}, {frozenset(): 1}               # orders from the node to full / from {} to the node
    for S in sorted(nodes, key=len, reverse=True):
        if S != full:
            n_from[S] = sum(n_from[S | {p}] for p in out.get(S, []))
    for S in sorted(nodes, key=len):
        for p in out.get(S, []):
            n_to[S | {p}] = n_to.get(S | {p}, 0) + n_to[S]
    orders = []

    def rec(S, o):
        if S == full:
            orders.append(o)
            return
        for p in sorted(out.get(S, []), key=lambda x: KEYORD.index(SHORT[x])):
            rec(S | {p}, o + [p])
    rec(frozenset(), [])
    label = lambda S: "{" + ",".join(sorted((SHORT[p] for p in S), key=KEYORD.index)) + "}" if S else "{ }"
    layers = {}
    for S in nodes:
        layers.setdefault(len(S), []).append(S)
    pos = {}
    for k, Ss in layers.items():
        Ss.sort(key=lambda S: [KEYORD.index(SHORT[p]) for p in sorted(S, key=lambda x: KEYORD.index(SHORT[x]))])
        for i, S in enumerate(Ss):
            pos[S] = (k * 3.2, (len(Ss) - 1) / 2 - i)
    best_edges = {(frozenset(best[:k]), best[k]) for k in range(len(best))}

    fig = plt.figure(figsize=(27, 14), facecolor=SURF)
    ax = fig.add_axes([0.01, 0.05, 0.70, 0.86])
    ax.set_facecolor(SURF)
    for S, p in edges:
        T = S | {p}
        (x0, y0), (x1, y1) = pos[S], pos[T]
        hi = (S, p) in best_edges
        ax.add_patch(FancyArrowPatch((x0 + 0.78, y0), (x1 - 0.78, y1), arrowstyle="-|>", mutation_scale=16,
                                     lw=3.4 if hi else 1.5, color=HI if hi else EDGE, zorder=3 if hi else 1))
        t = 0.42
        ax.text(x0 + 0.78 + t * (x1 - x0 - 1.56), y0 + t * (y1 - y0), SHORT[p], fontsize=12, fontweight="bold",
                color=HI if hi else INK2, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.9))
    on_best = {frozenset(best[:k]) for k in range(len(best) + 1)}
    for S, (x, y) in pos.items():
        hi = S in on_best
        ax.add_patch(FancyBboxPatch((x - 0.78, y - 0.36), 1.56, 0.72, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=NODE, ec=HI if hi else GRID, lw=3.0 if hi else 1.4, zorder=5))
        ax.text(x, y + 0.12, label(S), ha="center", va="center", fontsize=12.5 if len(S) < 6 else 11, fontweight="bold", color=INK, zorder=6)
        ax.text(x, y - 0.17, f"{n_to[S]}×{n_from[S]} 条", ha="center", va="center", fontsize=11, fontweight="bold", color=INK2, zorder=6)
    for k in sorted(layers):
        ax.text(k * 3.2, max(p[1] for S, p in pos.items() if len(S) == k) + 0.75, f"已装 {k} 个", ha="center",
                fontsize=14, fontweight="bold", color=INK2)
    ax.set_xlim(-1.0, 6 * 3.2 + 1.0)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.75, max(p[1] for p in pos.values()) + 1.15)
    ax.axis("off")
    ax.set_title(f"状态网：{len(nodes)} 个状态（已装集合），{len(edges)} 条箭头（再装一个零件）。"
                 f"每条从 {{ }} 到全装好的路径 = 一个装配顺序，共 {len(orders)} 条",
                 fontsize=17, fontweight="bold", color=INK, loc="left")
    ax.text(-0.9, min(p[1] for p in pos.values()) - 0.6,
            "节点下方 a×b 条：从 { } 走到这里有 a 种走法，从这里走到全装好有 b 种走法，经过它的顺序共 a×b 个。"
            "蓝色 = 论文 SGA 与 chain 代价的最优顺序", fontsize=12.5, fontweight="bold", color=INK2)
    # the 48 orders
    ax2 = fig.add_axes([0.72, 0.05, 0.27, 0.86])
    ax2.axis("off")
    ax2.set_title(f"{len(orders)} 个顺序（= 网上的 {len(orders)} 条路径）", fontsize=17, fontweight="bold", color=INK, loc="left")
    per = (len(orders) + 1) // 2
    for i, o in enumerate(orders):
        col, row = divmod(i, per)
        hi = tuple(o) == best
        ax2.text(col * 0.5, 1 - (row + 0.6) / (per + 0.5), f"{i + 1:2d}. " + " → ".join(SHORT[p] for p in o),
                 fontsize=12.5, color=HI if hi else INK, fontweight="bold",
                 transform=ax2.transAxes, family="Microsoft YaHei")
    f = OUT / "lattice_triangular_frame6.png"
    fig.savefig(f, dpi=130, facecolor=SURF)
    log(f"-> {f}  ({len(nodes)} nodes, {len(edges)} edges, {len(orders)} orders)")


if __name__ == "__main__":
    sys.exit(main())
