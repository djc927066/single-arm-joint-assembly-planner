"""Reverse joint search with space-time prices in the bound (2026-11-01).

conflict.py's reverse best-first tree; only the node bound changes. At node (R, suffix), the
parts of R will all be picked while p (the part just decided) and every other part of R that is
installed later lies on the table, so the price constraints whose two parts are both in R still
apply below this node: their prices are added to the Held-Karp plan of R (occupancy price at the
spot of the part lying on the table, victim price at the blocked poses of the part picked under
installed set S) and their sum is subtracted. Valid for any prices >= 0, so the search stays exact;
prices = 0 gives conflict.py.

Prices: price.py's subgradient at the root, once, with an upper bound guessed as
ub_factor x (conflict-free bound) (no known optimum used).

    python -X utf8 -m experiments.bls_all_orders_20261031.priced --model rack10_tall --mode chain
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from time import perf_counter

import numpy as np

from .conflict import CH, ConflictSearch
from .lazy import setup
from .price import PriceBound
from .run import OUT, log


class PricedSearch(ConflictSearch):
    def __init__(self, ls, orders, mode):
        self.ctx, self.lam, self.pmemo = None, {}, {}
        super().__init__(ls, orders, mode)

    def set_prices(self, lam):
        self.lam = {c: v for c, v in lam.items() if v > 1e-12 and c[0] in ("F", "B")}
        self.pmemo = {}

    def prices_for(self, R):
        """prices of the constraints with both parts in R (and the victim's installed set inside R)."""
        Rs, B, A, tot = set(R), {}, {}, 0.0
        for c, v in self.lam.items():
            if c[0] == "F":
                _, q, y, p, xs = c
                S = None
            else:
                _, q, y, p, S, xs = c
            if q not in Rs or p not in Rs or (S is not None and not set(S) <= Rs):
                continue
            tot += v
            B.setdefault(q, np.zeros(len(self.O["poses"][q])))[y] += v
            if S is None:
                B.setdefault(p, np.zeros(len(self.O["poses"][p])))[list(xs)] += v
            else:
                A.setdefault((p, S), np.zeros(len(self.O["poses"][p])))[list(xs)] += v
        return B, A, tot

    def plan(self, R, suf, head):
        if not self.lam:
            return super().plan(R, suf, head)
        self.ctx = (R,) + self.prices_for(R)
        v, pl = super().plan(R, suf, head)
        tot = self.ctx[3]
        self.ctx = None
        self.st["priced_plans"] += 1
        return (v - tot if v < np.inf else v), pl

    def cheapest(self, queries):
        if self.ctx is None:
            return super().cheapest(queries)
        R, B, A, _ = self.ctx
        out, run = {}, []
        for q in dict.fromkeys(queries):
            S, last, p, suf = q
            key = q + (R,)
            if key in self.pmemo:
                out[q] = self.pmemo[key]
                continue
            xs = self.rx.domain(S, p)
            xs = xs[self.free(p, suf)[xs]]
            c = self.tr[last, p][xs] + self.own[p][xs]
            if p in B:
                c = c + B[p][xs]
            if (p, S) in A:
                c = c + A[p, S][xs]
            o = np.argsort(c, kind="stable")
            run.append([q, key, xs[o], c[o], 0])
        while run:
            self.flags([(q[0], q[2], q[3], xs[i:i + CH]) for q, _, xs, c, i in run])
            nxt = []
            for r in run:
                q, key, xs, c, i = r
                m = self.U[q[0], q[2], q[3]]
                hit = next((j for j in range(i, min(i + CH, len(xs))) if m[int(xs[j])]), None)
                if hit is not None or i + CH >= len(xs):
                    self.pmemo[key] = out[q] = (float(c[hit]), int(xs[hit])) if hit is not None else (np.inf, None)
                else:
                    r[4] = i + CH
                    nxt.append(r)
            run = nxt
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="chain", choices=("bls", "chain"))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--ub-factor", type=float, default=1.10)
    a = ap.parse_args()
    T0 = perf_counter()
    pb, O, geo, ls, orders, t_setup, t_prep = setup(a.model, a.goal_pos, a.k)
    cs = PricedSearch(ls, orders, a.mode)
    t1 = perf_counter()
    pbd = PriceBound(cs, star=True, K=40)
    L0, _ = pbd.solve()
    pbest, _, hist, _ = pbd.run(ub=L0 * a.ub_factor, iters=a.iters)
    cs.set_prices(pbd.best_lam)
    t_price, g_price = perf_counter() - t1, dict(geo.st)
    log(f"prices: bound {L0:.4f} -> {pbest:.4f} with {len(cs.lam)} priced constraints ({t_price:.1f}s)")
    best = cs.run()
    t_search = cs.st["seconds"]
    root_priced = cs.root_bound
    path, n = [], best
    while n is not None:
        path.append(n)
        n = n["parent"]
    ref_f = OUT / f"conflict_{a.mode}_{a.model}_k{a.k}.json"
    ref = json.loads(ref_f.read_text(encoding="utf-8")) if ref_f.exists() else {}
    g = geo.st
    summary = dict(model=a.model, mode=a.mode, cost=best["g"] if best else None, unpriced_optimum=ref.get("cost"),
                   match=None if not ref or not best else abs(best["g"] - ref["cost"]) < 1e-9,
                   order=[p for p, _ in best["seq"]] if best else None,
                   root_bound_unpriced=L0, root_bound_priced=root_priced, price_phase_bound=pbest,
                   price_s=t_price, price_gpu_s=g_price["gpu_s"], search_s=t_search, total_s=perf_counter() - T0,
                   path_steps=dict(Counter(n["how"] for n in path)), **{k: v for k, v in cs.st.items() if k != "seconds"},
                   gpu_s=g["gpu_s"], pairs=g["pairs"],
                   unpriced=dict(expanded=ref.get("expanded"), seconds=ref.get("seconds"), pairs=ref.get("pairs"),
                                 gpu_s=ref.get("gpu_s"), dp=ref.get("dp"), expanded_pose=ref.get("expanded_pose"),
                                 expanded_order=ref.get("expanded_order")))
    log(json.dumps(summary, indent=1, ensure_ascii=False))
    (OUT / f"priced_{a.mode}_{a.model}_k{a.k}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                                                encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
