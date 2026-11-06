"""Why is the optimum above the conflict-free bound? (2026-11-01)

For the optimal plan (conflict_<mode>_<model>.json), every part p with installed set S_p: each pose
of p's domain D(S_p, p) cheaper (own + transition from its actual predecessor) than its chosen pose
is classified against the actual table T_p (parts installed after p, at their optimal poses):
  footprint  overlaps a part of T_p
  single     one part of T_p alone leaves no usable option at the pose
  union      every single part of T_p leaves an option, all of T_p together leave none
  free       usable with T_p: p could have taken it; the plan avoided it because of the parts
             installed BEFORE p (p at that pose would block or overlap one of them)

    python -X utf8 -m experiments.bls_all_orders_20261031.diag_gap --model rack10_tall --mode chain
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from .conflict import ConflictSearch
from .lazy import setup
from .run import OUT, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="chain", choices=("bls", "chain"))
    a = ap.parse_args()
    pb, O, geo, ls, orders, *_ = setup(a.model, a.goal_pos, a.k)
    res = json.loads((OUT / f"conflict_{a.mode}_{a.model}_k{a.k}.json").read_text(encoding="utf-8"))
    cs = ConflictSearch(ls, orders, a.mode)
    order = res["order"]
    X = {p: O["poses"][p].index(res["layout"][p]) for p in order}
    total, gap_parts, kinds = 0.0, {}, Counter()
    for k, p in enumerate(order):
        S = tuple(sorted(order[:k]))
        prev = order[k - 1] if k else None
        T = order[k + 1:]
        suf = tuple(sorted(geo.oi[(q, X[q])] for q in T))
        xs = cs.rx.domain(S, p)
        c = cs.tr[prev, p][xs] + cs.own[p][xs]
        chosen = float(cs.tr[prev, p][X[p]] + cs.own[p][X[p]])
        cheaper = xs[c < chosen - 1e-12]
        loss = chosen - float(c.min())
        total += loss
        why = Counter()
        if len(cheaper):
            cs.flags([(S, p, suf, cheaper)] + [(S, p, (geo.oi[(q, X[q])],), cheaper) for q in T])
            for x in cheaper:
                x = int(x)
                if any(x in set(cs.ls.fp.get(((q, X[q]), p), np.zeros(0, int)).tolist()) for q in T):
                    why["footprint"] += 1
                elif cs.U[S, p, suf][x]:
                    why["free"] += 1
                elif any(not cs.U[S, p, (geo.oi[(q, X[q])],)][x] for q in T):
                    why["single"] += 1
                else:
                    why["union"] += 1
        # the cheapest cheaper pose decides the loss: its reason
        first = None
        if len(cheaper):
            x0 = int(xs[np.argmin(c)])
            first = next(r for r, test in (
                ("footprint", lambda: any(x0 in set(cs.ls.fp.get(((q, X[q]), p), np.zeros(0, int)).tolist()) for q in T)),
                ("free", lambda: cs.U[S, p, suf][x0]),
                ("single", lambda: any(not cs.U[S, p, (geo.oi[(q, X[q])],)][x0] for q in T)),
                ("union", lambda: True)) if test())
            kinds[first] += loss
        gap_parts[p] = dict(loss=loss, cheaper=len(cheaper), reasons=dict(why), cheapest_reason=first)
        log(f"{k + 1:2d} {p:8s} loss {loss:.4f}  cheaper poses {len(cheaper):3d}  {dict(why)}  cheapest: {first}")
    log(f"sum of per-step losses {total:.4f} (optimum {res['cost']:.4f}); loss by reason of the cheapest pose: "
        + ", ".join(f"{k} {v:.4f}" for k, v in kinds.most_common()))
    (OUT / f"diag_gap_{a.mode}_{a.model}_k{a.k}.json").write_text(
        json.dumps(dict(parts=gap_parts, loss_by_reason=dict(kinds), total=total), indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
