"""Slide figures (16:9) that explain the reverse search of the mainline on the triangular frame (2026-11-02).

    python -X utf8 -m experiments.bls_all_orders_20261031.slides_reverse          # rebuild + search + draw
    python -X utf8 -m experiments.bls_all_orders_20261031.slides_reverse --draw   # draw from _out/slides_reverse.json

Data from lattice_joint (current screening): instance, lattice, a traced LatticeSearch run.
  slide_reverse_1_why.png        why backwards: when C is placed, BC and CA lie on the table at known poses
  slide_reverse_2_r0..r3, _end   the "boxes" list (best-first), round by round, real numbers of the run
  slide_reverse_3_f.png          f along the optimal plan: exact suffix part + F-table estimate
  slide_reverse_4_stop.png       when to stop and what was never touched
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy (c10.dll)

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"
DATA = OUT / "slides_reverse.json"
SHORT = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}
COLORS = {"post_A": "#2f8f45", "post_B": "#c73d3d", "post_C": "#3b5a94", "beam_AB": "#e08a2c",
          "beam_BC": "#c9a425", "beam_CA": "#9b5ca8"}
INK, INK2, GRID, SURF, HI, OK, BAD, GREY = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb", "#2a78d6", "#2f8f45", "#d0573b", "#a9a8a2"


def poly(P):
    return [list(map(float, c)) for c in P.exterior.coords]


def compute():
    from shapely.geometry import MultiPoint
    from lattice_joint.instance import Instance
    from lattice_joint.lattice import Lattice
    from lattice_joint.search import LatticeSearch
    inst = Instance.build("triangular_frame6")
    lat = Lattice(inst).build()
    srch = LatticeSearch(inst, lat, verbose=True, short=lambda p: SHORT[p])
    res = srch.run()
    sc, O = inst.pb.scene, inst.O
    plan = [(q, int(x)) for q, x in res.plan]
    trace = [t for t in res.trace if t["kind"] == "pose"]
    # f along the plan: depth k = the last k parts decided
    bars = [dict(k=0, f=float(lat.root), exact=0.0)]
    for k in range(1, len(plan) + 1):
        suf = plan[-k:]
        g = lat.const(suf[-1][0])
        for i, (q, x) in enumerate(suf):
            g += float(lat.own[q][x])
            if i + 1 < len(suf):
                h, xh = suf[i + 1]
                g += float(lat.tr[q, h][xh])
        if k == len(plan):
            g += float(lat.tr[None, plan[0][0]][plan[0][1]])
        p, x = suf[0]
        fs = [t["f"] for t in trace if t["p"] == p and t["x"] == x and [list(s) for s in t["suffix"]] ==
              [[q, xq] for q, xq in suf[1:]]]
        bars.append(dict(k=k, part=SHORT[p], pose=x, f=float(min(fs)), exact=float(g)))
    # C placed with BC, CA on the table (the optimal plan)
    C = "post_C"
    i = [q for q, _ in plan].index(C)
    suffix = tuple(plan[i + 1:])
    R1 = tuple(sorted(q for q, _ in plan[:i]))
    D = set(int(v) for v in lat.domain(R1, C))
    free = srch.footprint_free(C, suffix)
    tried = {}
    for t in trace:
        if t["p"] == C and [list(s) for s in t["suffix"]] == [[q, xq] for q, xq in suffix]:
            tried[int(t["x"])] = dict(ok=bool(t["ok"]), f=float(t["f"]), rank=int(t["rank"]))
    cpos = []
    for x, key in enumerate(O["poses"][C]):
        cat = "notD" if x not in D else ("overlap" if not free[x] else
                                          ("tried_ok" if tried.get(x, {}).get("ok") else
                                           "blocked" if x in tried else "untried"))
        if x == plan[i][1]:
            cat = "chosen"
        cpos.append(dict(x=x, xy=[float(v) for v in inst.start[C][x][:2]], cat=cat, f=tried.get(x, {}).get("f")))
    goal_hull = {}
    for q in ("base",) + tuple(sc.ids):
        v = np.asarray(sc.meshes[q].vertices) @ sc.goal[q][:3, :3].T + sc.goal[q][:3, 3]
        goal_hull[q] = poly(MultiPoint(v[:, :2]).convex_hull)
    d = dict(U=float(res.U), root=float(lat.root), plan=plan, bars=bars, stats=dict(res.stats),
             popped_f=[float(t["f"]) for t in res.trace if t["kind"] in ("pose", "expand")],
             left_f=[float(e["f"]) for e in res.left], n_left=int(res.stats["n_left"]),
             n_pairs=int(sum(len(lat.domain(S, p)) for S, p in lat.arrows)),
             C=dict(installed=list(R1), suffix=[[q, x, O["poses"][q][x], poly(sc.polygons[q][O["poses"][q][x]])]
                                                for q, x in suffix],
                    chosen_poly=poly(sc.polygons[C][O["poses"][C][plan[i][1]]]), pos=cpos, tried=tried),
             table=dict(center=[float(v) for v in sc.table_center[:2]], size=[float(v) for v in sc.table_size[:2]]),
             goal_hull=goal_hull)
    DATA.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"-> {DATA}")
    return d


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    return plt


def save(fig, name):
    f = OUT / name
    fig.savefig(f, dpi=120, facecolor=SURF)
    print(f"-> {f}")


# ---- slide 1: why backwards --------------------------------------------------------------------
def slide_why(d):
    plt = setup()
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.93, "为什么倒着排：轮到 C 选位置时，桌上躺着什么是确定的", fontsize=28, color=INK)
    ax = fig.add_axes([0.02, 0.50, 0.50, 0.38])
    ax.axis("off")
    ax.set_xlim(0, 6.4)
    ax.set_ylim(-1.6, 1.4)
    order = [SHORT[q] for q, _ in d["plan"]]
    ic = order.index("C")
    for j, s in enumerate(order):
        c = GREY if j < ic else (HI if j == ic else COLORS[d["plan"][j][0]])
        ax.add_patch(FancyBboxPatch((j + 0.12, 0.0), 0.8, 0.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=c if j != ic else "#ffffff", ec=c, lw=4 if j == ic else 2, alpha=1 if j >= ic else 0.55))
        ax.text(j + 0.52, 0.4, s, ha="center", va="center", fontsize=24, color="#ffffff" if j != ic else HI)
    ax.text(0.05, 1.1, "装配顺序（从左到右）", fontsize=17, color=INK2)
    ax.add_patch(FancyArrowPatch((6.0, -0.25), (0.6, -0.25), arrowstyle="-|>", mutation_scale=22, lw=2.5, color=HI))
    ax.text(3.3, -0.55, "搜索的方向：从最后一个往前定  CA → BC → C → …", ha="center", fontsize=16, color=HI)
    ax.text(ic * 0.5 + 0.05, -1.25, "已经装到底座上\n搜索前就查好（D 表）", ha="center", fontsize=15, color=INK2)
    ax.text(ic + 0.52, -1.25, "正在选\n位置", ha="center", fontsize=15, color=HI)
    ax.text((ic + 1 + 6) / 2 + 0.05, -1.25, "还躺在桌上\n位置已经定了", ha="center", fontsize=15, color=INK)
    tx = fig.add_axes([0.02, 0.05, 0.50, 0.40])
    tx.axis("off")
    lines = [("给 C 选位置要查两件事：", INK),
             ("① 和底座、已装的 A、B、AB 撞不撞", INK2),
             ("     → 和桌上放什么无关，搜索前一次算好", INK2),
             ("② 和桌上还没装的 BC、CA 撞不撞", INK),
             ("     → 倒着排时 BC、CA 的位置已经定了，当场就能查", HI),
             ("正着排的话，轮到 C 时 BC、CA 还没定，没法查", BAD)]
    for j, (s, c) in enumerate(lines):
        tx.text(0.02, 0.92 - j * 0.16, s, fontsize=19, color=c, transform=tx.transAxes, va="top")
    # plan view
    C = d["C"]
    ax = fig.add_axes([0.54, 0.17, 0.45, 0.70])
    tc, ts = np.array(d["table"]["center"]), np.array(d["table"]["size"])
    ax.add_patch(Rectangle(tc - ts / 2, ts[0], ts[1], fc="#f3efe6", ec=INK2, lw=1.2, zorder=0))
    ax.add_patch(Rectangle((-0.12, -0.12), 0.24, 0.24, fc="#e4e2dd", ec=INK2, lw=1, ls="--", zorder=1))
    ax.text(0, 0, "机械臂", ha="center", va="center", fontsize=13, color=INK2, zorder=2)
    for q, P in d["goal_hull"].items():
        if q == "base" or q in C["installed"]:
            ax.add_patch(Polygon(P, fc="#cfccc5" if q == "base" else GREY, ec=INK2, lw=0.8, alpha=0.8, zorder=2))
    gx = np.mean([np.mean(np.array(P)[:, 0]) for P in [d["goal_hull"]["base"]]])
    gy = np.mean(np.array(d["goal_hull"]["base"])[:, 1])
    ax.text(gx, gy, "底座\n已装 A B AB", ha="center", va="center", fontsize=13, color=INK, zorder=3)
    for q, x, key, P in C["suffix"]:
        ax.add_patch(Polygon(P, fc=COLORS[q], ec=COLORS[q], alpha=0.55, lw=1.5, zorder=3))
        P = np.array(P)
        ax.text(P[:, 0].mean(), P[:, 1].mean(), SHORT[q], ha="center", va="center", fontsize=17, color=INK, zorder=6)
    ax.add_patch(Polygon(C["chosen_poly"], fc="none", ec=HI, lw=2.5, zorder=5))
    sty = dict(notD=(GRID, "o", 30), untried=(GREY, "o", 45), overlap=("#e08a2c", "x", 80), blocked=(BAD, "X", 170),
               tried_ok=(OK, "o", 150), chosen=(HI, "*", 520))
    rank = ["notD", "untried", "overlap", "tried_ok", "blocked", "chosen"]
    by = {}
    for e in C["pos"]:
        k = tuple(np.round(e["xy"], 4))
        if k not in by or rank.index(e["cat"]) > rank.index(by[k]):
            by[k] = e["cat"]
    for cat in rank:
        pts = np.array([k for k, c in by.items() if c == cat])
        if len(pts):
            c, m, s = sty[cat]
            ax.scatter(pts[:, 0], pts[:, 1], c=c, marker=m, s=s, linewidths=2 if m == "x" else 0.5,
                       edgecolors="white" if m not in ("x",) else None, zorder=7)
    grp = {}                                                   # tried poses grouped by position
    for x, t in sorted(C["tried"].items(), key=lambda kv: kv[1]["rank"]):
        grp.setdefault(tuple(np.round(C["pos"][int(x)]["xy"], 4)), []).append(t)
    last = max(t["rank"] for t in C["tried"].values())
    for xy, ts_ in grp.items():
        ok = ts_[0]["ok"]
        word = "选中" if any(t["rank"] == last for t in ts_) else ("通过" if ok else "被挡")
        s = "第" + "、".join(str(t["rank"] + 1) for t in ts_) + f"个：{word}"
        right = xy[0] > tc[0] + 0.05
        off, ha = {"选中": ((-12, -42), "right"), "被挡": ((0, -40), "center")}.get(word, ((14, 14), "left"))
        ax.annotate(s, xy, xytext=off, textcoords="offset points", ha=ha,
                    fontsize=15, color=HI if word == "选中" else (OK if ok else BAD), zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    ax.set_xlim(tc[0] - ts[0] / 2 - 0.01, tc[0] + ts[0] / 2 + 0.01)
    ax.set_ylim(tc[1] - ts[1] / 2 - 0.01, tc[1] + ts[1] / 2 + 0.01)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("俯视：给 C 选位置时的桌面", fontsize=19, color=INK, loc="left")
    n = {c: sum(1 for v in by.values() if v == c) for c in rank}
    n.update(blocked=1, tried_ok=1, chosen=1)
    h = [Line2D([], [], ls="", marker=sty[c][1], ms=11, color=sty[c][0], mew=2, label=lab) for c, lab in [
        ("notD", f"和底座/已装零件撞（{n['notD']} 处）"), ("overlap", f"和 BC/CA 占地重叠（{n['overlap']} 处）"),
        ("blocked", "试了：被 BC/CA 挡住"), ("tried_ok", "试了：通过，但后面变贵"),
        ("chosen", "试了：通过，最终选中"), ("untried", f"估价太高，没轮到（{n['untried']} 处）")] if n.get(c, 1)]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.765, 0.0), ncol=2, frameon=False, fontsize=13)
    save(fig, "slide_reverse_1_why.png")


# ---- slide 2: the list of boxes, round by round --------------------------------------------------
NODES = {   # id: (column, row, text)
    "root": (0, 0.0, "什么都\n没定"),
    "CA": (1, 1.15, "最后装 CA\n（66 号）"), "CAr": (1, 2.35, "CA 其余\n21 个位置"),
    "AB": (1, -0.05, "最后装 AB"), "BC": (1, -1.25, "最后装 BC"),
    "BC_CA": (2, 1.15, "BC → CA\n（BC 35 号）"), "AB_CA": (2, -0.05, "AB → CA"),
    "C_BC_CA": (3, 1.15, "C → BC → CA"), "AB_BC_CA": (3, -0.05, "AB → BC → CA"),
    "END": (4.4, 1.15, "A→B→AB→C→BC→CA\n全部定完"),
}
EDGES = [("root", "CA"), ("root", "CAr"), ("root", "AB"), ("root", "BC"), ("CA", "BC_CA"), ("CA", "AB_CA"),
         ("BC_CA", "C_BC_CA"), ("BC_CA", "AB_BC_CA"), ("C_BC_CA", "END")]
F = {"root": 3.0925, "CA": 3.0925, "CAr": 3.0932, "AB": 3.0975, "BC": 3.2819, "BC_CA": 3.0925, "AB_CA": 3.2276,
     "C_BC_CA": 3.1297, "AB_BC_CA": 3.2807, "END": 3.1740}
ROUNDS = [
    dict(title="第 0 轮：只有一个盒子，标签 = F 表的下界", show=["root"], opened=[], nxt="root",
         say="标签 f = 这份方案最少要花多少（乐观估计）"),
    dict(title="第 1 轮：打开它 → 最后装谁？三个盒子", show=["root", "CA", "AB", "BC"], opened=["root"], nxt="CA",
         say="规则只有一条：永远打开标签最便宜的"),
    dict(title="第 2 轮：打开“最后装 CA”：取 CA 最便宜的位置 66 号，碰撞检查通过", show=["root", "CA", "CAr", "AB", "BC", "BC_CA", "AB_CA"],
         opened=["root", "CA"], nxt="BC_CA", say="CA 其余位置留在盒子里，标签换成下一个位置的估价"),
    dict(title="第 3 轮：打开“BC → CA”：BC 放 35 号，和桌上的 CA 不撞，通过", show=["root", "CA", "CAr", "AB", "BC", "BC_CA", "AB_CA",
                                                                       "C_BC_CA", "AB_BC_CA"],
         opened=["root", "CA", "BC_CA"], nxt="CAr", up="C_BC_CA",
         say="C 最便宜的位置被桌上的 BC、CA 占了/挡了 → 标签涨到 3.1297 → 最便宜的换人了，去开别的盒子"),
    dict(title="第 492 轮：一份方案全部定完，真实代价 3.1740 → 停", show=list(NODES), opened=["root", "CA", "BC_CA", "C_BC_CA", "END"],
         nxt=None, end=True,
         say="桌上剩下最便宜的标签 3.1743 ≥ 3.1740：标签只会低估，剩下的不可能更便宜。灰色虚线盒子（标签 > 3.1740）从没打开过"),
]


def slide_rounds(d):
    plt = setup()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    W, H, DX = 1.75, 0.85, 2.35
    for ri, r in enumerate(ROUNDS):
        fig = plt.figure(figsize=(16, 9), facecolor=SURF)
        fig.text(0.03, 0.92, r["title"], fontsize=25, color=INK)
        ax = fig.add_axes([0.01, 0.13, 0.64, 0.74])
        ax.axis("off")
        ax.set_xlim(-1.1, 4.4 * DX + 1.5)
        ax.set_ylim(-2.0, 3.1)
        show, opened, end = r["show"], r["opened"], r.get("end")
        pos = {k: (c * DX, y) for k, (c, y, _) in NODES.items()}
        for a, b in EDGES:
            if a in show and b in show:
                (x0, y0), (x1, y1) = pos[a], pos[b]
                dashed = b == "END"
                ax.add_patch(FancyArrowPatch((x0 + W / 2, y0), (x1 - (W + 1.1 if b == "END" else W) / 2, y1),
                                             arrowstyle="-|>", mutation_scale=18, lw=2.6 if b in opened else 1.4,
                                             ls="--" if dashed else "-", color=HI if b in opened else GREY))
        if end:
            ax.text((pos["C_BC_CA"][0] + pos["END"][0]) / 2 - 0.25, 2.0, "中间几百轮\n开开换换", ha="center",
                    fontsize=13, color=INK2)
        for k in show:
            x, y = pos[k]
            w = W + 1.1 if k == "END" else W
            never = end and k not in opened and F[k] >= d["U"]       # label above the optimum: never opened
            if k in opened:
                fc, ec, tc_ = "#e8f0fb", HI, INK
            elif never:
                fc, ec, tc_ = "#f0efec", GREY, GREY
            else:
                fc, ec, tc_ = "#ffffff", INK2, INK
            if k == r.get("nxt"):
                ec = BAD
            ax.add_patch(FancyBboxPatch((x - w / 2, y - H / 2), w, H, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        fc=fc, ec=ec, lw=3.5 if k == r.get("nxt") or k in opened else 1.6,
                                        ls="--" if never else "-"))
            if end and k not in opened and not never:
                ax.text(x, y + 0.17, NODES[k][2].split("\n")[0].replace(" 其余", " 其余位置"), ha="center", va="center",
                        fontsize=14, color=tc_)
                ax.text(x, y - 0.17, "打开过，都变贵了", ha="center", va="center", fontsize=11, color=INK2)
            else:
                ax.text(x, y + 0.08, NODES[k][2], ha="center", va="center", fontsize=14, color=tc_)
            up = k == r.get("up")
            ax.text(x, y - H / 2 - 0.2, f"f = {F[k]:.4f}" + ("  ↑涨价" if up else ""), ha="center", va="center",
                    fontsize=14, color=BAD if up else (HI if k in opened else tc_))
        # the list
        lx = fig.add_axes([0.66, 0.13, 0.32, 0.74])
        lx.axis("off")
        lx.set_xlim(0, 1)
        lx.set_ylim(0, 1)
        lx.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.97, boxstyle="round,pad=0.0,rounding_size=0.03",
                                    fc="#ffffff", ec=INK2, lw=1.5))
        lx.text(0.06, 0.92, "待开的盒子（按标签排序）", fontsize=18, color=INK)
        items = [k for k in show if k not in opened]
        if not end:
            items = sorted(items, key=lambda k: F[k])
        else:
            items = ["min"]
        y = 0.80
        for j, k in enumerate(items):
            if k == "min":
                lx.text(0.06, y, f"完整方案：3.1740", fontsize=19, color=HI)
                lx.text(0.06, y - 0.10, f"剩下 {d['n_left']} 个盒子", fontsize=17, color=INK2)
                lx.text(0.06, y - 0.18, f"最便宜的标签 {min(d['left_f']):.4f}", fontsize=17, color=INK2)
                lx.text(0.06, y - 0.30, "3.1743 ≥ 3.1740 → 停", fontsize=21, color=BAD)
                break
            name = NODES[k][2].replace("\n", " ")
            lx.text(0.06, y, f"{F[k]:.4f}", fontsize=17, color=BAD if j == 0 else INK)
            lx.text(0.30, y, name, fontsize=15, color=BAD if j == 0 else INK)
            if j == 0:
                lx.text(0.06, y - 0.055, "↑ 下一个打开它", fontsize=13, color=BAD)
                y -= 0.05
            y -= 0.095
        fig.text(0.03, 0.05, r["say"], fontsize=17, color=HI)
        if ri == 2:
            fig.text(0.66, 0.085, "注：CA 65 号与 66 号同一处、两种朝向，估价相同，图中只画一个", fontsize=10.5, color=INK2)
        save(fig, f"slide_reverse_2_r{ri}.png" if not end else "slide_reverse_2_end.png")


# ---- slide 3: f along the optimal plan -----------------------------------------------------------
def slide_f(d):
    plt = setup()
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.92, "标签 f 怎么变：越往前定，真实的部分越多、估计的部分越少", fontsize=26, color=INK)
    ax = fig.add_axes([0.07, 0.22, 0.60, 0.64])
    ax.set_facecolor(SURF)
    b = d["bars"]
    xs = np.arange(len(b))
    ex = np.array([v["exact"] for v in b])
    f = np.array([v["f"] for v in b])
    ax.bar(xs, ex, color=HI, width=0.62, label="已定部分：真实代价（坐标算出）")
    ax.bar(xs, f - ex, bottom=ex, color="#c9dcf3", width=0.62, hatch="//", edgecolor="#8fb4e6",
           label="没定部分：查 F 表的乐观估计")
    for x, v in zip(xs, f):
        ax.text(x, v + 0.03, f"{v:.4f}", ha="center", fontsize=15, color=BAD if v > d["root"] + 1e-6 else INK)
    ax.axhline(d["root"], color=INK2, ls="--", lw=1.4, label=f"F 表下界 {d['root']:.4f}")
    lab = ["什么都\n没定"] + ["+" + v["part"] + f"\n{v['pose']} 号" for v in b[1:]]
    ax.set_xticks(xs, lab, fontsize=15)
    ax.set_ylim(0, 4.35)
    ax.set_ylabel("f", fontsize=16)
    ax.tick_params(axis="y", labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=14, frameon=False)
    ax.set_xlabel("倒着定：最后一个 → 第一个", fontsize=16, color=INK2)
    k = next(i for i, v in enumerate(b) if v["f"] > d["root"] + 1e-6)
    ax.annotate("C 便宜的位置被桌上的\nBC、CA 占了/挡了 → 涨价", (k, f[k]), xytext=(k + 0.1, 3.75), fontsize=14, color=BAD,
                arrowprops=dict(arrowstyle="-|>", color=BAD, lw=2))
    tx = fig.add_axes([0.70, 0.22, 0.28, 0.64])
    tx.axis("off")
    lines = [("f = 真实部分 + 估计部分", INK), ("", INK),
             ("估计部分只会偏低", INK2), ("（F 表没算桌上零件互相挡）", INK2), ("", INK),
             ("所以 f 只会往上走", INK), ("不会往下走", INK), ("", INK),
             ("全部定完时", INK2), ("估计部分 = 0", INK2), (f"f = 真实代价 = {d['U']:.4f}", HI)]
    for j, (s, c) in enumerate(lines):
        tx.text(0.02, 0.98 - j * 0.085, s, fontsize=19, color=c, transform=tx.transAxes, va="top")
    save(fig, "slide_reverse_3_f.png")


# ---- slide 4: stop and what was saved --------------------------------------------------------------
def slide_stop(d):
    plt = setup()
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.text(0.03, 0.92, "什么时候停：手里的真实代价 ≤ 所有没开盒子的标签", fontsize=26, color=INK)
    ax = fig.add_axes([0.06, 0.40, 0.88, 0.44])
    ax.set_facecolor(SURF)
    bins = np.linspace(3.09, 3.40, 63)
    pf, lf = np.clip(d["popped_f"], bins[0], bins[-1]), np.clip(d["left_f"], bins[0], bins[-1])
    ax.hist(pf, bins=bins, color=HI, alpha=0.9, label=f"打开过的（{d['stats']['pops']} 次）")
    ax.hist(lf, bins=bins, color=GREY, alpha=0.9, label=f"没打开过的（标签最便宜的 {len(d['left_f'])} 个）")
    ax.axvline(d["U"], color=BAD, lw=3)
    ax.text(d["U"] + 0.002, ax.get_ylim()[1] * 0.92, f"最优 {d['U']:.4f}", fontsize=17, color=BAD)
    ax.axvline(d["root"], color=INK2, lw=1.5, ls="--")
    ax.text(d["root"] + 0.002, ax.get_ylim()[1] * 0.75, f"F 表下界\n{d['root']:.4f}", fontsize=14, color=INK2)
    ax.set_xlabel("标签 f", fontsize=16)
    ax.set_ylabel("个数", fontsize=16)
    ax.tick_params(labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=15, frameon=False)
    st = d["stats"]
    nums = [(f"{d['n_pairs']}", "（箭头，位置）组合"), (f"{st['pose_checks']}", "次位置碰撞检查"),
            (f"{st['pose_blocked']}", "次被桌上零件挡住，换下一个"), (f"{st['complete']}", "个完整方案 = 最优")]
    for j, (a, b) in enumerate(nums):
        x = 0.06 + j * 0.235
        fig.text(x, 0.22, a, fontsize=40, color=HI if j else INK2)
        fig.text(x, 0.16, b, fontsize=16, color=INK2)
    fig.text(0.06, 0.06, "红线右边的盒子从没打开：标签只会低估，它们的真实代价一定 ≥ 标签 > 3.1740，不用看", fontsize=18, color=INK)
    save(fig, "slide_reverse_4_stop.png")


def main():
    d = json.loads(DATA.read_text(encoding="utf-8")) if "--draw" in sys.argv and DATA.exists() else compute()
    slide_why(d)
    slide_rounds(d)
    slide_f(d)
    slide_stop(d)


if __name__ == "__main__":
    sys.exit(main())
