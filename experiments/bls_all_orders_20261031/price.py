"""Price (Lagrangian) lower bound of the joint order x layout problem (2026-11-01).

Kept exact in the subproblem: feasible orders (subset lattice), installed-set domains D(S, p),
the cost (chain / bls). Relaxed with a price each (every feasible plan satisfies them, so any
prices >= 0 give a valid lower bound):
  footprint   (p at x) + (q at y) <= 1                      if the two poses overlap
  blocking    (p at x after installed set S) + (q at y) <= 1 if q not in S (q still on the table
              when p is picked) and q at y alone leaves no usable option of p at x under S
Subproblem = Held-Karp over the subsets with pose costs + prices: q pays the price of the spot y
it occupies; p pays at x only for the installed sets S that leave the blocker on the table, so a
price moves the part (layout) or the order (install the blocker earlier), whichever is cheaper.
Prices rise by subgradient steps on the conflicts the relaxed solution actually hits (the
geometry is checked only for those pairs).

    python -X utf8 -m experiments.bls_all_orders_20261031.price --model rack10_tall --mode chain
"""
from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

import numpy as np

from .conflict import ConflictSearch
from .lazy import setup
from .run import OUT, log


class PriceBound:
    def __init__(self, cs, star=True, K=40):
        self.cs, self.n = cs, len(cs.ids)
        self.star, self.K, self._cand = star, K, {}
        self.npose = {p: len(cs.O["poses"][p]) for p in cs.ids}
        self.lam = {}                       # constraint -> price
        self.B = {p: np.zeros(self.npose[p]) for p in cs.ids}
        self.A = {}                         # (p, S) -> price per pose of p

    def solve(self):
        """min over orders and poses of cost + prices -> (value without the -sum(lam) constant, solution)."""
        cs = self.cs
        F = {((), None): (0.0, None)}
        for S in cs.sets:
            if len(S) == self.n:
                break
            lasts = [l for (T, l) in F if T == S]
            for p in cs.rx.nxt.get(S, ()):
                xs = cs.rx.domain(S, p)
                if not len(xs):
                    continue
                base = cs.own[p][xs] + self.B[p][xs]
                a = self.A.get((p, S))
                if a is not None:
                    base = base + a[xs]
                S2 = tuple(sorted(S + (p,)))
                for l in lasts:
                    v = cs.tr[l, p][xs] + base
                    i = int(np.argmin(v))
                    c = F[S, l][0] + float(v[i])
                    if (S2, p) not in F or c < F[S2, p][0]:
                        F[S2, p] = (c, (S, l, int(xs[i])))
        full = cs.ids
        best, q = np.inf, None
        for (S, l), (c, _) in F.items():
            if S == full and c + cs.const(l) < best:
                best, q = c + cs.const(l), l
        sol, S = [], full
        while q is not None:
            prevS, l, x = F[S, q][1]
            sol.append((q, x, prevS))
            S, q = prevS, l
        return best, sol[::-1]                         # forward: (part, pose, installed set)

    def violations(self, sol):
        return self.violations_star(sol) if self.star else self.violations_pair(sol)

    def cand(self, p, S, x):
        """the K cheapest poses of p's domain under S (own cost) and x: the poses a star covers."""
        key = (p, S)
        if key not in self._cand:
            xs = self.cs.rx.domain(S, p)
            self._cand[key] = xs[np.argsort(self.cs.own[p][xs], kind="stable")[:self.K]]
        return np.union1d(self._cand[key], [x]).astype(int)

    def violations_star(self, sol):
        """stars around each part q on the table at y: q at y + p at any pose blocked by q@y <= 1
        (footprint: any p; blocking: p picked while q is on the table, for p's installed set)."""
        cs = self.cs
        out, items = [], []
        at = {p: x for p, x, _ in sol}
        for i, (p, x, S) in enumerate(sol):
            for q, y, _ in sol:
                if q == p:
                    continue
                f = cs.ls.fp.get(((q, y), p))
                if f is not None:
                    xs = np.intersect1d(f, self.cand(p, S, x))
                    if x in set(xs.tolist()):
                        out.append(("F", q, y, p, tuple(int(v) for v in xs)))
            for q, y, _ in sol[i + 1:]:
                c = self.cand(p, S, x)
                items.append((S, p, (cs.geo.oi[(q, y)],), c, q, y, x))
        cs.flags([it[:4] for it in items])
        for S, p, suf, c, q, y, x in items:
            m = cs.U[S, p, suf]
            bl = tuple(int(v) for v in c if not m[int(v)])
            if x in bl:
                out.append(("B", q, y, p, S, bl))
        return out

    def violations_pair(self, sol):
        """conflicts hit by the relaxed solution (footprint pairs, blocking pairs: lazy geometry)."""
        cs = self.cs
        out, items = [], []
        for i, (p, x, S) in enumerate(sol):
            for q, y, _ in sol[i + 1:]:
                f = cs.ls.fp.get(((p, x), q))
                if f is not None and y in set(f.tolist()):
                    out.append(("f", p, x, q, y))
                items.append((S, p, (cs.geo.oi[(q, y)],), [x], q, y))
        cs.flags([it[:4] for it in items])
        for S, p, suf, xs, q, y in items:
            if not cs.U[S, p, suf][xs[0]]:
                out.append(("b", p, xs[0], S, q, y))
        return out

    def lhs(self, c, sol):
        at = {p: (x, S) for p, x, S in sol}
        if c[0] == "F":
            _, q, y, p, xs = c
            return (at[q][0] == y) + (at[p][0] in xs)
        if c[0] == "B":
            _, q, y, p, S, xs = c
            return (at[q][0] == y) + (at[p][1] == S and at[p][0] in xs)
        if c[0] == "f":
            _, p, x, q, y = c
            return (at[p][0] == x) + (at[q][0] == y)
        _, p, x, S, q, y = c
        return (at[p] == (x, S)) + (at[q][0] == y)

    def add(self, c, d):
        """change the price of constraint c by d (pose prices follow)."""
        if c[0] == "F":
            _, q, y, p, xs = c
            self.B[q][y] += d
            self.B[p][list(xs)] += d
        elif c[0] == "B":
            _, q, y, p, S, xs = c
            self.B[q][y] += d
            self.A.setdefault((p, S), np.zeros(self.npose[p]))[list(xs)] += d
        elif c[0] == "f":
            _, p, x, q, y = c
            self.B[p][x] += d
            self.B[q][y] += d
        else:
            _, p, x, S, q, y = c
            self.A.setdefault((p, S), np.zeros(self.npose[p]))[x] += d
            self.B[q][y] += d
        self.lam[c] = self.lam.get(c, 0.0) + d

    def run(self, ub, iters=300, theta=2.0, patience=20):
        t0 = perf_counter()
        hist, best, best_it, since = [], -np.inf, 0, 0
        for it in range(iters):
            v, sol = self.solve()
            L = v - sum(self.lam.values())
            if L > best + 1e-9:
                best, best_it, since = L, it, 0
                self.best_sol = sol
                self.best_lam = dict(self.lam)
                self.best_B = {p: b.copy() for p, b in self.B.items()}
                self.best_A = {k: a.copy() for k, a in self.A.items()}
            else:
                since += 1
                if since >= patience:
                    theta, since = theta / 2, 0
            for c in self.violations(sol):
                self.lam.setdefault(c, 0.0)
            g = {c: self.lhs(c, sol) - 1 for c in self.lam}
            g = {c: v for c, v in g.items() if v > 0 or self.lam[c] > 0}
            hist.append(dict(it=it, L=L, best=best, n_viol=sum(1 for v in g.values() if v > 0),
                             n_cons=len(self.lam), theta=theta))
            if not any(v > 0 for v in g.values()):
                log(f"  it {it}: relaxed solution hits no pairwise conflict (L {L:.4f})")
                break
            if theta < 1e-4:
                break
            step = theta * max(ub - L, 1e-6) / sum(v * v for v in g.values())
            for c, gc in g.items():
                d = max(-self.lam[c], step * gc)
                if d:
                    self.add(c, d)
            if it % 20 == 0:
                log(f"  it {it}: L {L:.4f}  best {best:.4f}  conflicts hit {hist[-1]['n_viol']}  "
                    f"constraints {len(self.lam)}  theta {theta:.3g}  ({perf_counter() - t0:.1f}s)")
        return best, best_it, hist, perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="chain", choices=("bls", "chain"))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--pair", action="store_true", help="pairwise constraints instead of stars")
    ap.add_argument("--K", type=int, default=40, help="cheapest poses of a part a star covers")
    a = ap.parse_args()
    pb, O, geo, ls, orders, t_setup, t_prep = setup(a.model, a.goal_pos, a.k)
    ref = None
    for f, get in ((OUT / f"conflict_{a.mode}_{a.model}_k{a.k}.json", lambda d: d["cost"]),
                   (OUT / f"joint_all_{a.mode}_{a.model}_k{a.k}.json", lambda d: d["summary"]["best"])):
        if f.exists():
            ref = get(json.loads(f.read_text(encoding="utf-8")))
            break
    cs = ConflictSearch(ls, orders, a.mode)
    pbd = PriceBound(cs, star=not a.pair, K=a.K)
    L0, _ = pbd.solve()
    log(f"conflict-free bound {L0:.4f}; optimum {ref}")
    best, best_it, hist, secs = pbd.run(ub=ref, iters=a.iters)
    g = geo.st
    summary = dict(model=a.model, mode=a.mode, optimum=ref, conflict_free_bound=L0, price_bound=best,
                   gap_conflict_free=(ref - L0) / ref, gap_price=(ref - best) / ref,
                   gap_closed=(best - L0) / (ref - L0) if ref - L0 > 1e-12 else None,
                   best_iter=best_it, iters=len(hist), constraints=len(pbd.lam), seconds=secs,
                   gpu_s=g["gpu_s"], pairs=g["pairs"])
    log(json.dumps(summary, indent=1))
    OUT.mkdir(exist_ok=True)
    keys = O["poses"]
    rel = [dict(part=p, pose=keys[p][x], installed=list(S)) for p, x, S in pbd.best_sol]
    price, victim = {}, {}          # per pose: occupancy price (B); largest price as a blocked victim (A)
    for p in cs.ids:
        price[p] = {keys[p][i]: float(v) for i, v in enumerate(pbd.best_B[p]) if v > 1e-9}
        vic = np.zeros(len(keys[p]))
        for (q, S), arr in pbd.best_A.items():
            if q == p:
                vic = np.maximum(vic, arr)
        victim[p] = {keys[p][i]: float(v) for i, v in enumerate(vic) if v > 1e-9}
    (OUT / f"price_{'pair' if a.pair else 'star'}_{a.mode}_{a.model}_k{a.k}.json").write_text(
        json.dumps(dict(summary=summary, hist=hist, relaxed=rel, prices=price, victim_prices=victim), indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
