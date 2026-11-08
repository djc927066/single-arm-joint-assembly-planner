"""The state graph (lattice) and its optimistic bound (steps 1 and 2 of the search).

Nodes = installed sets S (feasible prefixes of the stable orders); an arrow S --p--> S + {p} =
install p next. On each arrow:
  D(S, p)          table poses x of p with some option (x, g) free under the installed set S
                   (installed_checks; the parts on the table are ignored here)
  step(S, last, p) min over x in D(S, p) of
                       |above pick(x) - above goal(last)| + LIFT + |above goal(p) - above pick(x)| + LIFT
                   (from HOME for the first part); "above" = + LIFT in z
  const(last)      |above goal(last) - HOME|, the return after the last part
F(S, last): forward Held-Karp, cheapest relaxed cost from HOME installing S with last one = last.
root = min over last of F(full, last) + const(last): a lower bound of every plan.
Admissible: Euclidean lines are the shortest moves, D(S, p) contains every pose usable in a real
plan (which also has to clear the parts on the table), and the minimum is taken over it.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .geometry import LIFT, log

UP = np.array([0.0, 0.0, LIFT])


class Lattice:
    def __init__(self, inst):
        self.inst = inst
        self.nxt = {}                               # S -> parts that can be installed next
        for o in inst.orders:
            for k in range(len(o)):
                self.nxt.setdefault(tuple(sorted(o[:k])), set()).add(o[k])
        self.arrows = [(S, p) for S in self.nxt for p in self.nxt[S]]
        self.dom = {}
        ids = inst.ids
        # exact cost pieces, per pose: own = LIFT + transfer + LIFT, tr[q, p] = transit q -> p (None = HOME)
        self.own = {p: LIFT + np.linalg.norm(inst.goal[p] + UP - (inst.start[p] + UP), axis=1) + LIFT for p in ids}
        self.tr = {(None, p): np.linalg.norm(inst.start[p] + UP - inst.home, axis=1) for p in ids}
        for p in ids:
            for q in ids:
                if q != p:
                    self.tr[q, p] = np.linalg.norm(inst.start[p] + UP - (inst.goal[q] + UP), axis=1)
        self.F, self.lasts, self.root = {}, {}, np.inf
        self.sub = {}                               # R -> table prices of the prefix R (prices.py)

    # ---- step 1 ---------------------------------------------------------------------------
    def domain(self, S, p):
        if (S, p) not in self.dom:
            feas, O = self.inst.feas, self.inst.O
            okp, okg = feas.base_ok(S, p)
            ok = okp & okg[feas.opt_g[p]]
            self.dom[S, p] = np.unique(np.array([x for x, _ in O["opt"][p]])[ok])
        return self.dom[S, p]

    # ---- step 2 ---------------------------------------------------------------------------
    def step(self, S, last, p):
        xs = self.domain(S, p)
        return float((self.tr[last, p][xs] + self.own[p][xs]).min()) if len(xs) else np.inf

    def const(self, last):
        return float(np.linalg.norm(self.inst.goal[last] + UP - self.inst.home))

    def build(self):
        """D on every arrow, then the forward DP F and the root bound."""
        t0 = perf_counter()
        for S, p in self.arrows:
            self.domain(S, p)
        F = {((), None): 0.0}
        for S in sorted(self.nxt, key=len):
            for (T, last), v in list(F.items()):
                if T != S:
                    continue
                for p in self.nxt[S]:
                    S2 = tuple(sorted(S + (p,)))
                    c = v + self.step(S, last, p)
                    if c < F.get((S2, p), np.inf):
                        F[S2, p] = c
        self.F = F
        for (T, q) in F:
            self.lasts.setdefault(T, []).append(q)
        full = self.inst.ids
        self.root = min(F[full, q] + self.const(q) for q in self.lasts.get(full, []))
        self.seconds = perf_counter() - t0
        log(f"[1] D(S, p) on {len(self.arrows)} arrows; [2] F on {len(F)} (S, last) states;"
            f" root bound {self.root:.4f}  ({self.seconds:.1f}s)")
        return self

    def prefix_bound(self, R, p, xs):
        """optimistic cost of installing R from HOME and then moving to p at poses xs:
        min over the possible lasts q of R of F(R, q) + transit(q -> p at xs); HOME if R is empty."""
        if not R:
            return self.tr[None, p][xs]
        qs = self.lasts.get(R)
        if not qs:
            return np.full(len(xs), np.inf)
        return np.min(np.stack([self.F[R, q] + self.tr[q, p][xs] for q in qs]), axis=0)

    def priced_prefix(self, R, p, xs, suffix):
        """as prefix_bound, with the parts of R competing for the table (prices.py); the points under p
        and the suffix are taken, so their prices are not given back."""
        s = self.sub.get(R)
        if s is None:
            return np.full(len(xs), -np.inf)
        b = np.min(np.stack([v + self.tr[q, p][xs] for q, v in s["F"].items()]), axis=0)
        return b + s["price"][p][xs] + sum(s["price"][q][x] for q, x in suffix) - s["total"]
