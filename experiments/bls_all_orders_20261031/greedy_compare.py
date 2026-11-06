"""Greedy (no backtracking) vs the mainline best-first reverse search on the triangular frame (2026-11-02).

Greedy uses exactly the labels of lattice_joint.search (f = exact suffix + F-table estimate) and the
same collision checks, but at every step it commits to the child with the smallest f among the usable
(part, pose) options and never goes back. The best-first search keeps every other option in its list.

    python -X utf8 -m experiments.bls_all_orders_20261031.greedy_compare          # build + greedy + draw
    python -X utf8 -m experiments.bls_all_orders_20261031.greedy_compare --draw   # draw from _out/greedy_compare.json
Needs _out/slides_reverse.json (slides_reverse.py) for f along the optimal plan.
Output: _out/greedy_compare.json, _out/slide_reverse_5_greedy.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy (c10.dll)

import numpy as np  # noqa: E402

from .slides_reverse import BAD, COLORS, GREY, HI, INK, INK2, OK, SHORT, SURF, poly, save, setup  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"
DATA = OUT / "greedy_compare.json"


def greedy(inst, lat, srch, chunk=8):
    """reverse greedy: from the full set, commit to the usable (p, x) with the smallest f, never go back."""
    feas, geo = inst.feas, inst.geo
    R, suffix, g, steps = inst.ids, (), 0.0, []
    while R:
        best = None
        for p in R:
            R1 = tuple(q for q in R if q != p)
            if p not in lat.nxt.get(R1, ()):
                continue
            xs = lat.domain(R1, p)
            xs = xs[srch.footprint_free(p, suffix)[xs]]
            if not len(xs):
                continue
            link = lat.const(p) if not suffix else float(lat.tr[p, suffix[0][0]][suffix[0][1]])
            f = g + lat.own[p][xs] + link + lat.prefix_bound(R1, p, xs)
            o = np.argsort(f, kind="stable")
            okp, okg = feas.base_ok(R1, p)
            obs = tuple(sorted(geo.oi[(q, xq)] for q, xq in suffix))
            n_tried = 0
            for i in range(0, len(o), chunk):
                c = [int(v) for v in xs[o[i:i + chunk]]]
                ok = feas.usable(p, c, okp, okg, obs) if obs else [True] * len(c)
                j = next((j for j, v in enumerate(ok) if v), None)
                if j is None:
                    n_tried += len(c)
                    continue
                n_tried += j + 1
                cand = (float(f[o[i + j]]), p, c[j], R1, link, n_tried, float(f[o[0]]))
                if best is None or cand[0] < best[0]:
                    best = cand
                break
        if best is None:
            return dict(plan=None, dead_end=[SHORT[q] for q in R], steps=steps)
        f, p, x, R1, link, n_tried, f0 = best
        g += float(lat.own[p][x]) + link
        suffix = ((p, x),) + suffix
        steps.append(dict(part=SHORT[p], pose=x, key=inst.O["poses"][p][x], f=f, f_first=f0, tried=n_tried))
        R = R1
    p, x = suffix[0]
    cost = g + float(lat.tr[None, p][x])
    return dict(plan=[[q, int(v)] for q, v in suffix], cost=cost, steps=steps)


def compute():
    from lattice_joint.instance import Instance
    from lattice_joint.lattice import Lattice
    from lattice_joint.search import LatticeSearch, plan_cost
    inst = Instance.build("triangular_frame6")
    lat = Lattice(inst).build()
    srch = LatticeSearch(inst, lat)
    gr = greedy(inst, lat, srch)
    sc = inst.pb.scene
    if gr["plan"] is not None:
        c = plan_cost(inst, [(q, x) for q, x in gr["plan"]])
        assert abs(c - gr["cost"]) < 1e-9, (c, gr["cost"])
    best = json.loads((OUT / "slides_reverse.json").read_text(encoding="utf-8"))
    lay = lambda plan: [[q, x, inst.O["poses"][q][x], poly(sc.polygons[q][inst.O["poses"][q][x]])] for q, x in plan]
    d = dict(greedy=gr, best=dict(plan=best["plan"], cost=best["U"], bars=best["bars"], stats=best["stats"]),
             root=float(lat.root), greedy_layout=lay(gr["plan"]) if gr["plan"] else None, best_layout=lay(best["plan"]),
             table=best["table"], goal_hull=best["goal_hull"])
    DATA.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"-> {DATA}")
    print("greedy:", [(s["part"], s["pose"], round(s["f"], 4), s["tried"]) for s in gr["steps"]],
          "cost", gr.get("cost"), "dead end", gr.get("dead_end"))
    return d


def draw(d):
    plt = setup()
    from matplotlib.patches import Polygon, Rectangle
    gr, be = d["greedy"], d["best"]
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.92, "贪心（选了不回头） vs 现在的反向搜索（可以回头）", fontsize=27, color=INK)
    ax = fig.add_axes([0.06, 0.38, 0.42, 0.47])
    ax.set_facecolor(SURF)
    bf = [b["f"] for b in be["bars"]]
    gf = [d["root"]] + [s["f"] for s in gr["steps"]]
    if gr["plan"]:
        gf[-1] = gr["cost"]
    xs = np.arange(len(bf))
    ax.plot(xs, bf, "-o", color=HI, lw=3.5, ms=10, label=f"反向搜索：{be['cost']:.4f}（最优）")
    ax.plot(np.arange(len(gf)), gf, "-s", color=BAD, lw=3.5, ms=10,
            label=f"贪心：{gr['cost']:.4f}" if gr["plan"] else "贪心：走不通")
    for x, v in enumerate(bf):
        ax.text(x, v + 0.006, f"{v:.4f}", ha="center", va="bottom", fontsize=12, color=HI)
    for x, v in enumerate(gf):
        if x and abs(v - bf[min(x, len(bf) - 1)]) > 1e-4:
            ax.text(x, v - 0.008, f"{v:.4f}", ha="center", va="top", fontsize=12, color=BAD)
    lab = ["空"] + [f"第{k}步" for k in range(1, len(bf))]
    ax.set_xticks(xs, lab, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("标签 f", fontsize=15)
    ax.set_xlabel("倒着定：最后一个 → 第一个", fontsize=14, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lo, hi = min(bf + gf), max(bf + gf)
    ax.set_ylim(lo - 0.04, hi + 0.04)
    ax.legend(loc="upper left", fontsize=15, frameon=False)
    # what each did, step by step
    tx = fig.add_axes([0.06, 0.04, 0.42, 0.27])
    tx.axis("off")
    bo = [SHORT[q] for q, _ in be["plan"]]
    go = [s["part"] for s in reversed(gr["steps"])]
    rows = [("反向搜索", " → ".join(bo), HI), ("贪心", " → ".join(go) if gr["plan"] else "走不通", BAD)]
    for j, (a, b, c) in enumerate(rows):
        tx.text(0.0, 0.88 - j * 0.24, a, fontsize=18, color=c, transform=tx.transAxes, va="top")
        tx.text(0.20, 0.88 - j * 0.24, b, fontsize=18, color=INK, transform=tx.transAxes, va="top")
    xy = lambda key: key.split(",")[:2]                       # same place, other orientation = same choice here
    bkeys = [key for _, _, key, _ in reversed(d["best_layout"])]
    k = next((i for i, (s, bk) in enumerate(zip(gr["steps"], bkeys)) if xy(s["key"]) != xy(bk)), None)
    if k is not None:
        s = gr["steps"][k]
        sx = np.array(xy(s["key"]), float)
        bxy = {SHORT[q]: np.array(xy(key), float) for q, _, key, _ in d["best_layout"]}
        taken = next((q for q, v in bxy.items() if q != s["part"] and np.linalg.norm(v - sx) < 0.03), None)
        gxy = {t["part"]: np.array(xy(t["key"]), float) for t in gr["steps"]}
        tx.text(0.0, 0.30, f"第 {k + 1} 步分岔：贪心把 {s['part']} 放在 ({sx[0]:.3f}, {sx[1]:.3f})，眼前最便宜（f = {s['f']:.4f}）",
                fontsize=12.5, color=INK2, transform=tx.transAxes, va="top")
        if taken:
            t = gxy[taken]
            tx.text(0.0, 0.14, f"但那里正是 {taken} 最好的位置 → {taken} 只能换到 ({t[0]:.3f}, {t[1]:.3f})，后面变贵，贪心回不去",
                    fontsize=12.5, color=INK2, transform=tx.transAxes, va="top")
    # layouts
    tc, ts = np.array(d["table"]["center"]), np.array(d["table"]["size"])
    for j, (name, layout, cost, c) in enumerate([("贪心", d["greedy_layout"], gr.get("cost"), BAD),
                                                 ("反向搜索（最优）", d["best_layout"], be["cost"], HI)]):
        ax = fig.add_axes([0.51 + j * 0.245, 0.10, 0.235, 0.74])
        ax.add_patch(Rectangle(tc - ts / 2, ts[0], ts[1], fc="#f3efe6", ec=c, lw=3, zorder=0))
        ax.add_patch(Polygon(d["goal_hull"]["base"], fc="#cfccc5", ec=INK2, lw=0.8, zorder=1))
        P = np.array(d["goal_hull"]["base"])
        ax.text(P[:, 0].mean(), P[:, 1].mean(), "底座", ha="center", va="center", fontsize=13, color=INK2, zorder=2)
        order = [SHORT[q] for q, *_ in layout] if layout else []
        for q, x, key, Pq in layout or []:
            ax.add_patch(Polygon(Pq, fc=COLORS[q], ec=COLORS[q], alpha=0.6, lw=1.5, zorder=3))
            Pq = np.array(Pq)
            ax.text(Pq[:, 0].mean(), Pq[:, 1].mean(), SHORT[q], ha="center", va="center", fontsize=14, color=INK,
                    zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        ax.set_xlim(tc[0] - ts[0] / 2 - 0.01, tc[0] + ts[0] / 2 + 0.01)
        ax.set_ylim(tc[1] - ts[1] / 2 - 0.01, tc[1] + ts[1] / 2 + 0.01)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{name}\n代价 {cost:.4f}" if cost else name, fontsize=18, color=c)
    if gr["plan"]:
        gap = (gr["cost"] - be["cost"]) / be["cost"]
        same = [SHORT[q] for q, _ in be["plan"]] == [t["part"] for t in reversed(gr["steps"])]
        fig.text(0.53, 0.07, ("顺序一样，差在布局。" if same else "") + f"贪心比最优贵 {gap:.1%}",
                 fontsize=16, color=INK)
        fig.text(0.53, 0.03, "两者用同一套标签和碰撞检查，差别只在“能不能回头”", fontsize=16, color=INK)
    save(fig, "slide_reverse_5_greedy.png")


def main():
    d = json.loads(DATA.read_text(encoding="utf-8")) if "--draw" in sys.argv and DATA.exists() else compute()
    draw(d)


if __name__ == "__main__":
    sys.exit(main())
