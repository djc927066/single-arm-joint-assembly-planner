"""Table-area prices: a tighter root bound, with the parts on the table competing for space.

A point of the table lies under at most one part, so for any prices lam_t >= 0 and any plan
    cost + sum_p price(p, x_p) - sum_t lam_t  <=  cost ,   price(p, x) = sum of lam_t under p at pose x
and the minimum of the left side over all plans is a lower bound of every plan. That minimum is the
forward DP of the lattice with the prices added to the step cost. table_prices tunes one set of prices
per undecided set R and stores it in Lattice.sub; the search reads it through Lattice.priced_prefix. Footprints are
grown by just under gap / 2: two of them sharing a point are then closer than gap, which a plan may
not be. table_prices looks for good prices by subgradient ascent; bad ones only give a weaker bound.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import scipy.sparse as sp
from matplotlib.path import Path as MPath

from .geometry import log


def cover(inst, lat, step):
    """per part: used poses, and the sparse (used pose x table point) matrix of the grown footprints."""
    ids, pb, O = inst.ids, inst.pb, inst.O
    used = {p: np.unique(np.concatenate([lat.dom[a] for a in lat.dom if a[1] == p])) for p in ids}
    r = pb.gap / 2 * 0.99
    polys = {p: [pb.scene.polygons[p][O["poses"][p][int(x)]].buffer(r) for x in used[p]] for p in ids}
    b = np.array([P.bounds for p in ids for P in polys[p]])
    gx = np.arange(b[:, 0].min(), b[:, 2].max() + step, step)
    gy = np.arange(b[:, 1].min(), b[:, 3].max() + step, step)
    C = {}
    for p in ids:
        rows, cols = [], []
        for i, P in enumerate(polys[p]):
            x0, y0, x1, y1 = P.bounds
            ix = np.nonzero((gx >= x0) & (gx <= x1))[0]
            iy = np.nonzero((gy >= y0) & (gy <= y1))[0]
            I, J = (v.ravel() for v in np.meshgrid(ix, iy, indexing="ij"))
            c = (I * len(gy) + J)[MPath(np.asarray(P.exterior.coords)).contains_points(np.c_[gx[I], gy[J]])]
            rows.append(np.full(len(c), i))
            cols.append(c)
        cols = np.concatenate(cols)
        C[p] = sp.csr_matrix((np.ones(len(cols)), (np.concatenate(rows), cols)), shape=(len(used[p]), len(gx) * len(gy)))
    live = np.unique(np.concatenate([C[p].indices for p in ids]))
    return used, {p: C[p][:, live].tocsr() for p in ids}, len(live)


def table_prices(inst, lat, U, step=0.005, iters=1500):
    """fill lat.sub: for every undecided set R, the prices of the best bound found for installing R.
    U = any upper bound of a whole plan. Returns the bound of the whole assembly."""
    t0 = perf_counter()
    ids, fp = inst.ids, inst.feas.fp
    used, C, T = cover(inst, lat, step)
    row = {p: {int(x): i for i, x in enumerate(used[p])} for p in ids}
    arcs, lasts = [], {(): [None]}                 # (S, last, p, S + p, rows of the poses, poses, step cost)
    for S in sorted(lat.nxt, key=len):
        for last in lasts.get(S, []):
            for p in lat.nxt[S]:
                xs = lat.domain(S, p)
                S2 = tuple(sorted(S + (p,)))
                if p not in lasts.setdefault(S2, []):
                    lasts[S2].append(p)
                if len(xs):
                    arcs.append((S, last, p, S2, np.array([row[p][int(x)] for x in xs]), xs, lat.tr[last, p][xs] + lat.own[p][xs]))
    const = {p: lat.const(p) for p in ids}

    def tune(R):
        Rs, whole = set(R), R == ids
        mine = [a for a in arcs if Rs.issuperset(a[3])]
        tail = (lambda q: const[q]) if whole else (lambda q: 0.0)

        def dp(lam):
            pen = {p: C[p] @ lam for p in R}
            F, pred = {((), None): 0.0}, {}
            for S, last, p, S2, ri, xs, base in mine:
                v = F.get((S, last))
                if v is None:
                    continue
                c = base + pen[p][ri]
                j = int(c.argmin())
                if v + c[j] < F.get((S2, p), np.inf):
                    F[S2, p] = v + c[j]
                    pred[S2, p] = (S, last, int(xs[j]), int(ri[j]))
            q = min(lasts[R], key=lambda q: F.get((R, q), np.inf) + tail(q))
            plan, k = [], (R, q)
            while k[0]:
                S, last, x, i = pred[k]
                plan.append((k[1], x, i))
                k = (S, last)
            return F[R, q] + tail(q) - lam.sum(), plan[::-1], {v: F[R, v] for v in lasts[R] if (R, v) in F}

        def real_cost(plan):
            c, last = 0.0, None
            for p, x, _ in plan:
                c += lat.tr[last, p][x] + lat.own[p][x]
                last = p
            return c + tail(last)

        def clear(plan):
            for a, (p, x, _) in enumerate(plan):
                for q, y, _ in plan[a + 1:]:
                    f = fp.get(((p, x), q))
                    if f is not None and y in f:
                        return False
            return True

        lam = np.zeros(T)
        L0 = dp(lam)[0]
        best, blam, bF, alpha, stall, ub = -np.inf, lam, None, 1.0, 0, U if whole else 1.1 * L0
        for _ in range(iters):
            L, plan, F = dp(lam)
            if L > best + 1e-7:
                best, blam, bF, stall = L, lam.copy(), F, 0
            else:
                stall += 1
                if stall >= 40:
                    alpha, stall, lam = max(alpha / 2, 1e-3), 0, blam.copy()
                    L, plan, F = dp(lam)
            if clear(plan):
                ub = min(ub, real_cost(plan))
            g = sum(C[p][i].toarray().ravel() for p, _, i in plan) - 1
            g[(lam <= 0) & (g < 0)] = 0
            n = float(g @ g)
            if n == 0:                             # the plan fits and every price is paid: bound = its cost
                break
            lam = np.maximum(0, lam + alpha * max(ub - L, 1e-4) / n * g)
        price = {}
        for p in ids:
            price[p] = np.zeros(len(lat.own[p]))
            price[p][used[p]] = C[p] @ blam
        return best, dict(F=bF, price=price, total=float(blam.sum()))

    root = -np.inf
    for R in [*lat.nxt, ids]:
        if len(R) >= 2:
            b, s = tune(R)
            if R == ids:
                root = b
            else:
                lat.sub[R] = s
    log(f"table prices: {len(lat.sub)} undecided sets, whole bound {root:.4f}, {T} table points at {step * 1000:.0f} mm"
        f"  ({perf_counter() - t0:.0f}s)")
    return root
