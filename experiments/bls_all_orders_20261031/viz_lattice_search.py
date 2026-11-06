"""The lattice search of lattice_search.py drawn on the frame's state graph, step by step (2026-11-01).

Left: the state graph (as viz_lattice.py). Arrows: blue = walked by the search (a pose of that part
was checked and accepted; [n] = the steps), red dashed = generated but never expanded
(its best f >= U, value shown), grey dotted = never generated (its state was never expanded).
Right: every pop of the search in order, with f, what was decided, and why the search turned.

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_lattice_search
"""
from __future__ import annotations

import json
import sys

from .run import OUT, log
from .viz_lattice import EDGE, GRID, HI, INK, INK2, KEYORD, NODE, SHORT, SURF

CUT, DIM = "#d03b3b", "#c9c7c1"
CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    d = json.loads((OUT / "lattice_search_triangular_frame6.json").read_text(encoding="utf-8"))
    lat = json.loads((OUT / "sga_orders_triangular_frame6.json").read_text(encoding="utf-8"))
    edges = [(frozenset(e["installed"]), e["part"]) for e in lat["edges"]]
    U = d["U"]
    nm = lambda S: "{" + ",".join(sorted((SHORT[q] for q in S), key=KEYORD.index)) + "}" if S else "{ }"
    xy = lambda key: "(" + ",".join(key.split(",")[:2]) + ")"

    # ---- events, numbered, and the status of every arrow --------------------------------------
    steps, walked, gen_f, expanded = [], {}, {}, {}
    tr = d["trace"]
    for i, t in enumerate(tr):
        if t["kind"] == "expand":
            R = frozenset(t["R"])
            expanded[R] = expanded.get(R, 0) + 1
            g = tr[i + 1]["gens"] if i + 1 < len(tr) and tr[i + 1]["kind"] == "gens" else []
            parts = []
            for e in sorted(g, key=lambda e: e["f"]):
                arrow = (R - {e["p"]}, e["p"])
                gen_f[arrow] = min(gen_f.get(arrow, 9), e["f"])
                s = f"{SHORT[e['p']]} {e['f']:.4f}"
                if e["f"] > e["f_nofp"] + 1e-9:
                    s += f"（本来 {e['f_nofp']:.4f}，最好的位置被桌上的后缀零件占了）"
                parts.append(s)
            suf = " ".join(SHORT[q[0]] for q in t["suffix"]) or "无"
            steps.append(("expand", t["f"], f"展开 {nm(R)}（后缀已定：{suf}）\n      谁是 {nm(R)} 里最后装的？候选：" + "；".join(parts)))
        elif t["kind"] == "pose":
            R = frozenset(t["R"])
            arrow = (R - {t["p"]}, t["p"])
            k = len([s for s in steps if s[0] == "pose"])
            walked.setdefault(arrow, []).append(k)
            last = len(R) == 1
            what = f"定 {SHORT[t['p']]} 为 {nm(R)} 里最后装的，桌上放 {xy(t['key'])}（" + ("它最省的位置" if t["rank"] == 0 else f"它第 {t['rank'] + 1} 省的位置") + "）"
            steps.append(("pose", t["f"], what + ("：检查通过 → 得到完整方案" if last else "：检查通过")))
        elif t["kind"] == "U":
            steps.append(("U", t["U"], f"更新当前最好 U = {t['U']:.4f}（顺序 {' → '.join(SHORT[q] for q in t['order'])}）"))
    for e in d["left"]:
        if e["kind"] == "pose":
            arrow = (frozenset(e["R"]) - {e["p"]}, e["p"])
            gen_f[arrow] = min(gen_f.get(arrow, 9), e["f"])
    left_min = min(e["f"] for e in d["left"])

    # ---- layout as viz_lattice ---------------------------------------------------------------
    nodes = {frozenset()} | {S | {p} for S, p in edges}
    layers = {}
    for S in nodes:
        layers.setdefault(len(S), []).append(S)
    pos = {}
    for k, Ss in layers.items():
        Ss.sort(key=lambda S: [KEYORD.index(SHORT[p]) for p in sorted(S, key=lambda x: KEYORD.index(SHORT[x]))])
        for i, S in enumerate(Ss):
            pos[S] = (k * 3.2, (len(Ss) - 1) / 2 - i)
    fig = plt.figure(figsize=(33, 15), facecolor=SURF)
    ax = fig.add_axes([0.005, 0.07, 0.60, 0.83])
    ax.set_facecolor(SURF)
    pose_no = {}                                        # pose step k -> overall step number
    n = 0
    for i, s in enumerate(steps):
        if s[0] == "pose":
            pose_no[n] = i + 1
            n += 1
    for S, p in edges:
        T = S | {p}
        (x0, y0), (x1, y1) = pos[S], pos[T]
        a = (S, p)
        if a in walked:
            col, lw, ls, al, lab = HI, 3.6, "-", 1.0, f"{SHORT[p]} " + " ".join(f"[{pose_no[k]}]" for k in walked[a])
        elif a in gen_f:
            col, lw, ls, al, lab = CUT, 1.8, (0, (4, 3)), 0.8, f"{SHORT[p]} {gen_f[a]:.3f}≥U"
        else:
            col, lw, ls, al, lab = DIM, 1.2, (0, (1, 2)), 0.9, SHORT[p]
        ax.add_patch(FancyArrowPatch((x0 + 0.78, y0), (x1 - 0.78, y1), arrowstyle="-|>", mutation_scale=16,
                                     lw=lw, color=col, alpha=al, linestyle=ls, zorder=3 if a in walked else 1))
        t = 0.5 if a in walked else 0.42
        ax.text(x0 + 0.78 + t * (x1 - x0 - 1.56), y0 + t * (y1 - y0), lab, fontsize=12 if a in walked else 10.5,
                fontweight="bold", color=col if a in walked or a in gen_f else "#9a9893", ha="center", va="center",
                zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none", alpha=0.92))
    for S, (x, y) in pos.items():
        ex = expanded.get(S, 0)
        reached = ex > 0 or not S
        ax.add_patch(FancyBboxPatch((x - 0.78, y - 0.36), 1.56, 0.72, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=NODE, ec=HI if ex else GRID, lw=3.0 if ex else 1.2,
                                    alpha=1.0 if reached else 0.45, zorder=5))
        ax.text(x, y + 0.12, nm(S), ha="center", va="center", fontsize=12.5 if len(S) < 6 else 11, fontweight="bold",
                color=INK if reached else "#9a9893", zorder=6)
        ax.text(x, y - 0.17, f"展开 {ex} 次" if ex else ("起点 { }" if not S else "未展开"), ha="center", va="center",
                fontsize=11, fontweight="bold", color=HI if ex else "#9a9893", zorder=6)
    for k in sorted(layers):
        ax.text(k * 3.2, max(p[1] for S, p in pos.items() if len(S) == k) + 0.75, f"已装 {k} 个", ha="center",
                fontsize=14, fontweight="bold", color=INK2)
    ax.set_xlim(-1.0, 6 * 3.2 + 1.0)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.85, max(p[1] for p in pos.values()) + 1.15)
    ax.axis("off")
    st = d["stats"]
    ax.set_title(f"反向搜索在状态网上实际走过的部分：从右（全装好）往左。蓝 = 走过（[n] = 右边的步骤号），"
                 f"红虚线 = 生成了但乐观值 ≥ U，没展开，灰点线 = 根本没生成\n"
                 f"共弹出 {st['pops']} 次，展开 {st['expanded']} 个节点，几何检查 {st['pose_checks']} 次；"
                 f"最优 U = {U:.4f}，下界 {d['root']:.4f}", fontsize=15.5, fontweight="bold", color=INK, loc="left")
    handles = [Line2D([], [], color=HI, lw=3.6, label="走过：检查了这一步的桌面位置并通过"),
               Line2D([], [], color=CUT, lw=1.8, ls=(0, (4, 3)), label="生成了，但乐观值 ≥ U：不用检查"),
               Line2D([], [], color=DIM, lw=1.2, ls=(0, (1, 2)), label="没生成：它所在的状态从未被展开")]
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0.0, -0.08),
              prop=dict(size=13, weight="bold"))

    # ---- the steps -------------------------------------------------------------------------
    ax2 = fig.add_axes([0.615, 0.03, 0.38, 0.92])
    ax2.axis("off")
    ax2.set_title("搜索每一步（每次从堆里取出乐观值 f 最小的一项）", fontsize=16, fontweight="bold", color=INK, loc="left")
    y = 0.985
    for i, (kind, f, text) in enumerate(steps):
        col = HI if kind == "pose" else ("#2f8f45" if kind == "U" else INK)
        ax2.text(0.0, y, f"[{i + 1}] f {f:.4f}  {text}", fontsize=11.2, color=col,
                 fontweight="bold", va="top", transform=ax2.transAxes, linespacing=1.4)
        y -= 0.027 * (text.count("\n") + 1) + 0.008
    ax2.text(0.0, y - 0.005, f"停止：堆里剩下 {len(d['left'])} 项，最小的 f = {left_min:.4f} ≥ U = {U:.4f}。\n"
             f"再往下任何分支连最乐观都比 U 贵 → U 就是最优，不用再看。\n"
             f"对比：48 个顺序、30 条箭头、D 里一共 3135 个（箭头，位置）组合；这里只检查了 {st['pose_checks']} 个。",
             fontsize=12, color=CUT, fontweight="bold", va="top", transform=ax2.transAxes, linespacing=1.5)
    f = OUT / "lattice_search_triangular_frame6.png"
    fig.savefig(f, dpi=110, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
