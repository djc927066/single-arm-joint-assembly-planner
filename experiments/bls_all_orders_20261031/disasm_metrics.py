"""Per-candidate metrics of the first two disassembly steps (= last two installed) (2026-11-01).

From the assembled state R = all parts, every part p that can be removed (scene.step_ok(R - {p}, p))
gets the metrics a reverse search knows exactly at that moment (installed set R - {p}):
  stable     the remaining set R - {p} is statically stable (scene, yes/no)
  G          goal grasps of p free with R - {p} installed (insertion side), of all goal grasps
  options    (pose, grasp) options of p feasible on the goal side
  poses      table poses of p with at least one such option, of all poses
  best cost  cheapest such pose (BLS Euclidean step cost, HOME -> pick -> goal -> HOME)
  median     median cost of those poses (how much room is left near the cheapest)
Then the same for the second step under every first choice.

    python -X utf8 -m experiments.bls_all_orders_20261031.disasm_metrics --model triangular_frame6
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .conflict import ConflictSearch
from .lazy import setup
from .run import log


def metrics(cs, R, p):
    S = tuple(q for q in R if q != p)
    okp, okg = cs.ls.base_ok(S, p)
    xs = cs.rx.domain(S, p)
    c = cs.tr[None, p][xs] + cs.own[p][xs] if len(xs) else np.zeros(0)
    return dict(stable=bool(cs.ls.pb.scene.stable(frozenset(S))) if hasattr(cs.ls.pb.scene, "stable") else None,
                G=int(okg.sum()), G_all=len(okg), options=int((okp & okg[cs.ls.opt_g[p]]).sum()), options_all=len(okp),
                poses=len(xs), poses_all=len(cs.O["poses"][p]),
                best=float(c.min()) if len(c) else None, median=float(np.median(c)) if len(c) else None)


def show(cs, R, indent=""):
    rows = []
    for p in R:
        S = tuple(q for q in R if q != p)
        if p not in cs.rx.nxt.get(S, ()):
            continue
        m = metrics(cs, R, p)
        rows.append((p, m))
        log(f"{indent}{p:8s} stable-after {str(m['stable']):5s}  G {m['G']:3d}/{m['G_all']:<3d}  "
            f"options {m['options']:4d}/{m['options_all']:<4d}  poses {m['poses']:3d}/{m['poses_all']:<3d}  "
            f"best {m['best']:.4f}  median {m['median']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    pb, O, geo, ls, orders, *_ = setup(a.model, a.goal_pos, a.k)
    ls.pb = pb
    cs = ConflictSearch(ls, orders, "bls")
    full = cs.ids
    log("== step 1: remove one part from the assembled frame (= installed last)")
    first = show(cs, full)
    for p, _ in first:
        R = tuple(q for q in full if q != p)
        log(f"== step 2 after removing {p}:")
        show(cs, R, "   ")


if __name__ == "__main__":
    sys.exit(main())
