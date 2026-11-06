"""The optimistic value of every arrow of the frame's state graph, term by term (chain cost) (2026-11-01).

through(S, p) = min over last of  F(S, last) + step(S, last, p) + H(S + {p}, p)
  F(S, last)       cheapest relaxed prefix: from HOME install S, last one = last (forward Held-Karp)
  step(S, last, p) min over x in D(S, p) of  |above pick(x) - above goal(last)| + LIFT
                                              + |above goal(p) - above pick(x)| + LIFT
                   (from HOME for the first part)
  H(S', p)         cheapest relaxed completion after installing p (joint.Relaxed.h), at the end the
                   return to HOME
Logs F, the step breakdown and H for every arrow, and saves _out/worked_bound_triangular_frame6.json.

    python -X utf8 -m experiments.bls_all_orders_20261031.worked_bound
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .joint import UP, Relaxed
from .lazy import setup
from .run import LIFT, OUT, log

SHORT = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}
nm = lambda S: "{" + ",".join(sorted((SHORT[q] for q in S), key=["A", "B", "C", "AB", "BC", "CA"].index)) + "}"


def main():
    pb, O, geo, ls, orders, *_ = setup("triangular_frame6", "0.373,0.0,0.0", 8)
    rx = Relaxed(ls, orders, "chain")
    cd = ls.cd

    def breakdown(S, last, p):
        xs = rx.domain(S, p)
        sp, gp = cd["start"][p][xs], cd["goal"][p]
        frm = cd["home"] if last is None else cd["goal"][last] + UP
        tr = np.linalg.norm((sp + UP) - frm, axis=1)
        tf = np.linalg.norm((gp + UP) - (sp + UP), axis=1)
        c = tr + LIFT + tf + LIFT
        i = int(np.argmin(c))
        return dict(n_dom=len(xs), pose=O["poses"][p][xs[i]], transit=float(tr[i]), transfer=float(tf[i]),
                    lift=2 * LIFT, step=float(c[i]), pick=(sp[i]).round(3).tolist())

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
    log(f"HOME tcp {np.round(cd['home'], 3)}; goals (above = +{LIFT} m z): "
        + ", ".join(f"{SHORT[p]} {np.round(cd['goal'][p], 3)}" for p in O["ids"]))
    log(f"root bound H({{}}) = {rx.h((), None):.4f}")
    rows = []
    for S in sorted(rx.nxt, key=lambda S: (len(S), nm(S))):
        for p in sorted(rx.nxt[S], key=lambda q: list(SHORT).index(q)):
            S2 = tuple(sorted(S + (p,)))
            cands = []
            for (T, last), f in F.items():
                if T != S:
                    continue
                b = breakdown(S, last, p)
                h = rx.h(S2, p)
                cands.append(dict(last=last, F=f, H=h, total=f + b["step"] + h, **b))
            best = min(cands, key=lambda r: r["total"])
            rows.append(dict(S=list(S), p=p, through=best["total"], cands=cands))
            log(f"\n{nm(S):16s} --{SHORT[p]:2s}->  through = {best['total']:.4f}   (|D| = {best['n_dom']} poses)")
            for r in sorted(cands, key=lambda r: r["total"]):
                log(f"    last {SHORT.get(r['last'], 'HOME'):4s}: F {r['F']:.4f} + step {r['step']:.4f}"
                    f" [transit {r['transit']:.4f} + lift {LIFT:.2f} + transfer {r['transfer']:.4f} + insert {LIFT:.2f};"
                    f" pose {r['pose']}] + H {r['H']:.4f} = {r['total']:.4f}")
    (OUT / "worked_bound_triangular_frame6.json").write_text(json.dumps(dict(
        home=cd["home"].tolist(), goal={p: cd["goal"][p].tolist() for p in O["ids"]}, rows=rows), indent=1,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
