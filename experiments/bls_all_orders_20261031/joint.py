"""Joint order x layout search: best-first over orders on the conflict-free relaxation (2026-10-31).

Relaxation: drop the conflicts between the parts on the table. The cost of an order is then a sum
of per-step terms, each the cheapest pose of the part in its installed-set domain D(S, p) (and, for
the chain cost, measured from the goal of the previous part):
    step(S, last, p) = min_{x in D(S, p)} cost(p, x | last)
D(S, p) needs only the installed-set check of (S, p): no pairwise geometry. Held-Karp over
(S, last) gives h(S, last) = cheapest relaxed completion, an admissible bound for every order with
that prefix. A* on (S, last) with h pops the orders in nondecreasing relaxed cost; each popped order
gets the exact lazy layout search (lazy.py) with the incumbent as upper bound; the search stops
when the next relaxed cost >= incumbent (then no later order can be cheaper: proven optimum).

    python -X utf8 -m experiments.bls_all_orders_20261031.joint --model triangular_frame6 --mode chain
    python -X utf8 -m experiments.bls_all_orders_20261031.joint --model rack10_tall --mode chain --strategy all

--strategy all: every stable order searched on its own (ub = inf), the reference.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from time import perf_counter

import numpy as np

from .lazy import setup
from .run import LIFT, OUT, log

UP = np.array([0.0, 0.0, LIFT])


class Relaxed:
    """step costs and Held-Karp completion bound of the conflict-free relaxation."""

    def __init__(self, ls, orders, mode):
        self.ls, self.mode, self.n = ls, mode, len(orders[0])
        self.orders = set(orders)
        self.nxt = {}
        for o in orders:
            for k in range(len(o)):
                self.nxt.setdefault(tuple(sorted(o[:k])), set()).add(o[k])
        self.dom, self.h_memo = {}, {}

    def domain(self, S, p):
        if (S, p) not in self.dom:
            ls = self.ls
            okp, okg = ls.base_ok(S, p)
            ok = okp & okg[ls.opt_g[p]]
            self.dom[S, p] = np.unique(np.array([x for x, _ in ls.O["opt"][p]])[ok])
        return self.dom[S, p]

    def step(self, S, last, p):
        xs = self.domain(S, p)
        if not len(xs):
            return np.inf
        cd = self.ls.cd
        sp, gp = cd["start"][p][xs], cd["goal"][p]
        frm = cd["home"] if self.mode == "bls" or last is None else cd["goal"][last] + UP
        c = np.linalg.norm((sp + UP) - frm, axis=1) + LIFT + np.linalg.norm((gp + UP) - (sp + UP), axis=1) + LIFT
        if self.mode == "bls":
            c = c + np.linalg.norm((gp + UP) - cd["home"])
        return float(c.min())

    def const(self, last):
        return 0.0 if self.mode == "bls" else float(np.linalg.norm(self.ls.cd["goal"][last] + UP - self.ls.cd["home"]))

    def h(self, S, last):
        """cheapest relaxed completion from installed set S (last part installed: last)."""
        key = (S, last)
        if key not in self.h_memo:
            if len(S) == self.n:
                v = self.const(last)
            else:
                v = min((self.step(S, last, p) + self.h(tuple(sorted(S + (p,))), p) for p in self.nxt.get(S, ())),
                        default=np.inf)
            self.h_memo[key] = v
        return self.h_memo[key]

    def orders_by_bound(self):
        """A* on (S, last): yields (relaxed cost, order) in nondecreasing relaxed cost."""
        heap, cnt = [(self.h((), None), 0, 0.0, (), None, ())], 1
        while heap:
            f, _, g, S, last, seq = heapq.heappop(heap)
            if f == np.inf:
                return
            if len(seq) == self.n:
                if seq in self.orders:
                    yield f, seq
                continue
            for p in sorted(self.nxt.get(S, ())):
                c = self.step(S, last, p)
                S2 = tuple(sorted(S + (p,)))
                g2 = g + c
                f2 = g2 + self.h(S2, p)
                if f2 < np.inf:
                    heapq.heappush(heap, (f2, cnt, g2, S2, p, seq + (p,)))
                    cnt += 1

    def cost(self, order):
        S, last, v = (), None, 0.0
        for p in order:
            v += self.step(S, last, p)
            S, last = tuple(sorted(S + (p,))), p
        return v + self.const(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="chain", choices=("bls", "chain"))
    ap.add_argument("--strategy", default="joint", choices=("joint", "all"))
    a = ap.parse_args()
    T0 = perf_counter()
    pb, O, geo, ls, orders, t_setup, t_prep = setup(a.model, a.goal_pos, a.k)
    rx = Relaxed(ls, orders, a.mode)
    t1 = perf_counter()
    root = rx.h((), None)
    t_relax = perf_counter() - t1
    log(f"relaxation: {len(rx.dom)} installed-set domains (S, p), {len(rx.h_memo)} (S, last) states, "
        f"root bound {root:.4f}, {t_relax:.1f}s (gpu so far {geo.st['gpu_s']:.0f}s)")
    res, inc, inc_order, inc_lay = [], np.inf, None, None
    t_search = perf_counter()
    if a.strategy == "joint":
        stream = rx.orders_by_bound()
    else:
        stream = ((rx.cost(o), o) for o in orders)
    popped = 0
    for lb, o in stream:
        popped += 1
        if a.strategy == "joint" and lb >= inc - 1e-12:
            log(f"stop: next relaxed bound {lb:.4f} >= incumbent {inc:.4f} after {popped - 1} orders")
            break
        t = perf_counter()
        best, st = ls.run(tuple(o), a.mode, ub=inc if a.strategy == "joint" else np.inf)
        assert st["root_lb"] == lb or abs(st["root_lb"] - lb) < 1e-9, (st["root_lb"], lb)   # = lazy search's root
        c = None if best["lay"] is None else float(best["c"])
        if c is not None and c < inc - 1e-12:
            inc, inc_order, inc_lay = c, o, best["lay"]
        res.append(dict(order=list(o), relaxed=lb, cost=c, wall_s=perf_counter() - t,
                        **{k: v for k, v in st.items() if k != "root_lb"}))
        log(f"[{popped}] {' '.join(o)}  relaxed {lb:.4f}  exact {'-' if c is None else f'{c:.4f}'}  "
            f"incumbent {inc:.4f}  nodes {st['nodes']}  ({perf_counter() - t:.1f}s; gpu {geo.st['gpu_s']:.0f}s)")
    g = geo.st
    summary = dict(model=a.model, mode=a.mode, strategy=a.strategy, n_orders=len(orders), searched=len(res),
                   best=inc, best_order=list(inc_order) if inc_order else None,
                   best_layout={p: O["poses"][p][x] for p, x in inc_lay.items()} if inc_lay else None,
                   root_bound=root, total_s=perf_counter() - T0, setup_s=t_setup, prep_s=t_prep, relax_s=t_relax,
                   search_s=perf_counter() - t_search, nodes=sum(r["nodes"] for r in res), installed_sp=len(ls.base),
                   **g, cached_pose_queries=len(geo.bp), placed_obstacles=len({k[0] for k in geo.bp}))
    log(json.dumps(summary, indent=1, ensure_ascii=False))
    OUT.mkdir(exist_ok=True)
    f = OUT / f"joint_{a.strategy}_{a.mode}_{a.model}_k{a.k}.json"
    f.write_text(json.dumps(dict(summary=summary, results=res), indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
