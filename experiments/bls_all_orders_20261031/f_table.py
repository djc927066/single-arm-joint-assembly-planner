"""The F table of the mainline (lattice_joint, current screening) on the triangular frame (2026-11-02).

F(S, last) = cheapest optimistic cost from HOME installing the set S with `last` installed last
(forward Held-Karp over the lattice, lattice_joint.lattice.Lattice.build). This script rebuilds the
instance and the lattice only (no search), redoes the DP with back-pointers and writes
  _out/f_table_triangular_frame6.json   every (S, last): F, best predecessor, step, chosen pose
  _out/f_table_triangular_frame6.png    F on the lattice + the table

    python -X utf8 -m experiments.bls_all_orders_20261031.f_table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy (c10.dll)

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"
SHORT = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}
KEYORD = ["A", "B", "C", "AB", "BC", "CA"]
INK, INK2, GRID, SURF, EDGE, HI, NODE = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb", "#b9b8b1", "#2a78d6", "#f0efec"


def sn(p):
    return SHORT[p] if p else "HOME"


def label(S):
    return "{" + ",".join(sorted((SHORT[p] for p in S), key=KEYORD.index)) + "}" if S else "{ }"


def compute():
    from lattice_joint.instance import Instance
    from lattice_joint.lattice import Lattice
    inst = Instance.build("triangular_frame6")
    lat = Lattice(inst).build()
    rows = {((), None): dict(S=[], last=None, F=0.0, prev=None)}
    for S in sorted(lat.nxt, key=len):
        for (T, last), r in [(k, v) for k, v in rows.items() if k[0] == S]:
            for p in lat.nxt[S]:
                xs = lat.domain(S, p)
                c = lat.tr[last, p][xs] + lat.own[p][xs]
                i = int(np.argmin(c))
                S2 = tuple(sorted(S + (p,)))
                v = r["F"] + float(c[i])
                if (S2, p) not in rows or v < rows[S2, p]["F"]:
                    x = int(xs[i])
                    rows[S2, p] = dict(S=list(S2), last=p, F=v, prev=[list(S), last], step=float(c[i]),
                                       transit=float(lat.tr[last, p][x]), own=float(lat.own[p][x]),
                                       pose=x, pose_key=inst.O["poses"][p][x], n_dom=int(len(xs)),
                                       n_poses=int(len(inst.O["poses"][p])))
    for k, r in rows.items():               # the DP of lattice_joint must agree
        if k[1] is not None:
            assert abs(lat.F[k] - r["F"]) < 1e-12, k
    full = inst.ids
    ret = {q: lat.const(q) for q in lat.lasts[full]}
    d = dict(root=lat.root, ret=ret, rows=[r for k, r in rows.items() if k[1] is not None],
             arrows=[[list(S), p, int(len(lat.domain(S, p))), int(len(inst.O["poses"][p]))] for S, p in lat.arrows])
    assert abs(min(r["F"] + ret[r["last"]] for r in d["rows"] if len(r["S"]) == len(full)) - lat.root) < 1e-12
    OUT.mkdir(exist_ok=True)
    f = OUT / "f_table_triangular_frame6.json"
    f.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"-> {f}")
    return d


def draw(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "bold"
    rows = d["rows"]
    F = {(frozenset(r["S"]), r["last"]): r for r in rows}
    byS = {}
    for r in rows:
        byS.setdefault(frozenset(r["S"]), []).append(r)
    nodes = [frozenset()] + list(byS)
    full = max(nodes, key=len)
    best_last = min(byS[full], key=lambda r: r["F"] + d["ret"][r["last"]])["last"]
    on_best = set()                                            # back-pointers of the root bound
    k = (full, best_last)
    while k[1] is not None:
        on_best.add(k)
        S, q = F[k]["prev"]
        k = (frozenset(S), q)
    layers = {}
    for S in nodes:
        layers.setdefault(len(S), []).append(S)
    DX, DY, W = 3.4, 1.75, 1.7
    pos, H = {}, {}
    for n, Ss in layers.items():
        Ss.sort(key=lambda S: [KEYORD.index(SHORT[p]) for p in sorted(S, key=lambda x: KEYORD.index(SHORT[x]))])
        for i, S in enumerate(Ss):
            pos[S] = (n * DX, ((len(Ss) - 1) / 2 - i) * DY)
            H[S] = 0.42 + 0.30 * len(byS.get(S, [None]))

    fig = plt.figure(figsize=(30, 16), facecolor=SURF)
    ax = fig.add_axes([0.005, 0.04, 0.66, 0.88])
    ax.set_facecolor(SURF)
    for S, p, nd, npos in d["arrows"]:
        S = frozenset(S)
        T = S | {p}
        (x0, y0), (x1, y1) = pos[S], pos[T]
        hi = (T, p) in on_best and set(F[T, p]["prev"][0]) == S
        ax.add_patch(FancyArrowPatch((x0 + W / 2, y0), (x1 - W / 2, y1), arrowstyle="-|>", mutation_scale=16,
                                     lw=3.6 if hi else 1.4, color=HI if hi else EDGE, zorder=3 if hi else 1))
        t = 0.40
        ax.text(x0 + W / 2 + t * (x1 - x0 - W), y0 + t * (y1 - y0), f"{SHORT[p]} ({nd}/{npos})", fontsize=10.5,
                color=HI if hi else INK2, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.10", fc=SURF, ec="none", alpha=0.9))
    for S, (x, y) in pos.items():
        rs = sorted(byS.get(S, []), key=lambda r: KEYORD.index(SHORT[r["last"]]))
        hiS = any((S, r["last"]) in on_best for r in rs) or not S
        h = H[S]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - h / 2), W, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=NODE, ec=HI if hiS else GRID, lw=3.0 if hiS else 1.4, zorder=5))
        top = y + h / 2 - 0.22
        ax.text(x, top, label(S), ha="center", va="center", fontsize=12 if len(S) < 6 else 10.5, color=INK, zorder=6)
        if not S:
            ax.text(x, top - 0.30, "F = 0", ha="center", va="center", fontsize=12, color=INK2, zorder=6)
        for j, r in enumerate(rs):
            hi = (S, r["last"]) in on_best
            ax.text(x, top - 0.30 * (j + 1), f"末={SHORT[r['last']]}: {r['F']:.4f}", ha="center", va="center",
                    fontsize=11.5, color=HI if hi else INK2, zorder=6)
    xf, yf = pos[full]
    ax.text(xf + 0.2, yf - H[full] / 2 - 0.75, "+ 回 HOME：\n" + "\n".join(
        f"末={SHORT[q]}: {F[full, q]['F']:.4f} + {c:.4f} = {F[full, q]['F'] + c:.4f}" for q, c in
        sorted(d["ret"].items(), key=lambda kv: KEYORD.index(SHORT[kv[0]]))),
        ha="center", va="top", fontsize=11.5, color=INK, zorder=6)
    ax.text(xf + 0.2, yf - H[full] / 2 - 1.95, f"下界 root = {d['root']:.4f}", ha="center", va="top", fontsize=14,
            color=HI, zorder=6)
    for n in sorted(layers):
        ax.text(n * DX, max(pos[S][1] + H[S] / 2 for S in layers[n]) + 0.35, f"已装 {n} 个", ha="center",
                fontsize=14, color=INK2)
    ys = [pos[S][1] for S in nodes]
    ax.set_xlim(-1.1, 6 * DX + 1.3)
    ax.set_ylim(min(ys) - 2.3, max(ys) + 1.6)
    ax.axis("off")
    ax.set_title(f"F 表（当前主线 lattice_joint 的筛选）：{len(rows)} 个 (S, 最后一件) 状态（另有起点 {{ }}）。"
                 "节点里每行 = F(S, 末)：从 HOME 装完 S、最后装“末”的最小乐观代价",
                 fontsize=16, color=INK, loc="left")
    ax.text(-1.0, min(ys) - 2.1, "箭头标注 p (|D|/候选数)：这条箭头上 p 可用的桌面位置数。"
            "蓝色 = 下界 root 的回溯路径（每一步 F 取到最小值的那条箭头）", fontsize=12.5, color=INK2)

    ax2 = fig.add_axes([0.675, 0.04, 0.32, 0.88])
    ax2.axis("off")
    ax2.set_title("F(S, 末) = F(前一状态) + step；step = 移动 + 自身(抓 + 搬 + 插)", fontsize=14.5, color=INK, loc="left")
    hdr = f"{'S':<16}{'末':<4}{'F':>8}  = {'F(前)':>7} + {'移动':>6} + {'自身':>6}  前一状态/末     位置 (|D|)"
    lines = [hdr]
    order = sorted(rows, key=lambda r: (len(r["S"]), [KEYORD.index(SHORT[p]) for p in
                   sorted(r["S"], key=lambda x: KEYORD.index(SHORT[x]))], KEYORD.index(SHORT[r["last"]])))
    for r in order:
        PS, pq = r["prev"]
        Fp = 0.0 if pq is None else F[frozenset(PS), pq]["F"]
        lines.append(f"{label(r['S']):<16}{SHORT[r['last']]:<4}{r['F']:>8.4f}  = {Fp:>7.4f} + {r['transit']:>6.4f}"
                     f" + {r['own']:>6.4f}  {label(PS) + '/' + sn(pq):<14} {r['pose']:>3} ({r['n_dom']})")
    y = 0.985
    for i, ln in enumerate(lines):
        r = order[i - 1] if i else None
        hi = r is not None and (frozenset(r["S"]), r["last"]) in on_best
        ax2.text(0.0, y, ln, fontsize=10.6, family="Microsoft YaHei" if i == 0 else "monospace",
                 color=HI if hi else (INK if i else INK2), transform=ax2.transAxes, va="top")
        y -= 0.97 / (len(lines) + 1)
    f = OUT / "f_table_triangular_frame6.png"
    fig.savefig(f, dpi=120, facecolor=SURF)
    print(f"-> {f}")


def main():
    f = OUT / "f_table_triangular_frame6.json"
    d = json.loads(f.read_text(encoding="utf-8")) if "--draw" in sys.argv and f.exists() else compute()
    draw(d)


if __name__ == "__main__":
    sys.exit(main())
