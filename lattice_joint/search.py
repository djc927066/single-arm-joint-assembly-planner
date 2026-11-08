"""Reverse best-first search on the lattice with incumbent pruning (steps 3 to 5).

Node = (R undecided, suffix = decided (part, pose) in forward order, g = exact cost of the suffix
from its head on, the return to HOME included). Child = choose p as the last of R (an arrow
R \\ {p} --p--> R of the lattice) and its pose x. Priority
    f = g + own(p, x) + link(p -> head) + min_q [ F(R \\ {p}, q) + transit(q -> p at x) ]
exact except the prefix part, which is optimistic (Lattice.prefix_bound): admissible.
With table prices (prices.py) the prefix part is the larger of that and Lattice.priced_prefix, and f is
never below the f of the parent. Both are lower bounds, so f stays admissible.
For one (node, p) the poses x in D(R \\ {p}, p) whose footprint clears the suffix are sorted by f
and handed out lazily ("pose generator"): a pose is checked only when it is popped, against the
suffix parts lying on the table (Feasibility.usable, a chunk of poses per GPU round).
Merging: the search below a node depends only on R, the head (part, pose) and the SET of suffix
(part, pose) (they all lie on the table while R is installed; the head is where the arm goes after
R), so of the nodes with the same key only the one with the smallest g is kept. Exact.
Incumbent U = best complete plan; the search stops when the smallest f in the heap is >= U, and U
is then the optimum over the discrete candidates.
"""
from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import psutil

from .geometry import LIFT, log
from .lattice import UP


@dataclass
class Result:
    U: float
    plan: tuple                          # ((part, pose index), ...) in forward order
    stats: Counter
    seconds: float
    left: list = field(default_factory=list)     # never expanded heap entries (f >= U), smallest first
    trace: list = field(default_factory=list)    # every pop (verbose runs only)


class LatticeSearch:
    def __init__(self, inst, lat, merge=True, chunk=8, cap=5_000_000, verbose=False, short=None,
                 U0=np.inf, plan0=None, min_free_gb=0.0):
        self.inst, self.lat = inst, lat
        self.merge, self.chunk, self.cap, self.verbose = merge, chunk, cap, verbose
        self.sn = short or (lambda p: p)
        self.U0, self.plan0, self.min_free_gb = U0, plan0, min_free_gb     # known plan; stop before RAM runs out

    def _name(self, S):
        return "{" + ",".join(self.sn(q) for q in S) + "}" if S else "{}"

    def footprint_free(self, p, suffix):
        m = np.ones(len(self.inst.O["poses"][p]), bool)
        for q, x in suffix:
            f = self.inst.feas.fp.get(((q, x), p))
            if f is not None:
                m[f] = False
        return m

    def score(self, R1, p, xs, suffix, base):
        """f of p at poses xs; base = the exact part."""
        lat = self.lat
        return base + np.maximum(lat.prefix_bound(R1, p, xs), lat.priced_prefix(R1, p, xs, suffix))

    def dive(self):
        """reverse greedy: commit to the usable (p, x) with the smallest f, never go back.
        -> (cost, plan), (None, None) at a dead end. Gives the search its first incumbent."""
        inst, lat, feas, geo = self.inst, self.lat, self.inst.feas, self.inst.geo
        R, suffix, g = inst.ids, (), 0.0
        while R:
            best = None
            for p in R:
                R1 = tuple(q for q in R if q != p)
                if p not in lat.nxt.get(R1, ()):
                    continue
                xs = lat.domain(R1, p)
                xs = xs[self.footprint_free(p, suffix)[xs]]
                if not len(xs):
                    continue
                link = lat.const(p) if not suffix else float(lat.tr[p, suffix[0][0]][suffix[0][1]])
                f = self.score(R1, p, xs, suffix, g + lat.own[p][xs] + link)
                o = np.argsort(f, kind="stable")
                okp, okg = feas.base_ok(R1, p)
                obs = tuple(sorted(geo.oi[(q, xq)] for q, xq in suffix))
                for i in range(0, len(o), self.chunk):
                    c = [int(v) for v in xs[o[i:i + self.chunk]]]
                    ok = feas.usable(p, c, okp, okg, obs) if obs else [True] * len(c)
                    j = next((j for j, v in enumerate(ok) if v), None)
                    if j is not None:
                        if best is None or f[o[i + j]] < best[0]:
                            best = (float(f[o[i + j]]), p, c[j], R1, link)
                        break
            if best is None:
                return None, None
            _, p, x, R, link = best
            g += float(lat.own[p][x]) + link
            suffix = ((p, x),) + suffix
        p, x = suffix[0]
        return g + float(lat.tr[None, p][x]), suffix

    def run(self):
        inst, lat, feas, geo = self.inst, self.lat, self.inst.feas, self.inst.geo
        own, tr, nm, sn, verbose = lat.own, lat.tr, self._name, self.sn, self.verbose
        st = Counter()
        nodes, heap, cnt = [], [], [0]
        U, best = self.U0, self.plan0
        trace = []
        T = trace.append if verbose else (lambda d: None)
        pf, seen = {}, {}
        key_of = lambda R, suffix: (R, suffix[0], frozenset(suffix))

        def push(f, kind, payload):
            if f >= U - 1e-12:                                # would never be popped
                st["pruned_at_push"] += 1
                return
            heapq.heappush(heap, (f, cnt[0], kind, payload))
            cnt[0] += 1

        def expand(nid, f_parent):
            R, suffix, g = nodes[nid]
            st["expanded"] += 1
            gens = []
            for p in R:
                R1 = tuple(q for q in R if q != p)
                if p not in lat.nxt.get(R1, ()):              # R1 --p--> R is not an arrow
                    continue
                xs = lat.domain(R1, p)
                xs = xs[self.footprint_free(p, suffix)[xs]]
                if not len(xs):
                    st["empty_domain"] += 1
                    continue
                link = lat.const(p) if not suffix else float(tr[p, suffix[0][0]][suffix[0][1]])
                f = np.maximum(self.score(R1, p, xs, suffix, g + own[p][xs] + link), f_parent)
                o = np.argsort(f, kind="stable")
                st["generators"] += 1
                push(float(f[o[0]]), "pose", (nid, p, xs[o], f[o], 0))
                if verbose:
                    d0 = lat.domain(R1, p)
                    gens.append(dict(p=p, f=float(f[o[0]]), n=int(len(xs)), key=inst.O["poses"][p][int(xs[o[0]])],
                                     f_nofp=float(np.min(g + own[p][d0] + link + lat.prefix_bound(R1, p, d0)))))
            if verbose:
                T(dict(kind="gens", R=list(R), suffix=[q for q, _ in suffix], gens=gens))
                for d in gens:
                    log(f"         generator {nm(tuple(q for q in R if q != d['p']))} --{sn(d['p'])}->: {d['n']} poses,"
                        f" best f {d['f']:.4f}" + (f" (without the suffix on the table {d['f_nofp']:.4f})"
                                                   if d["f"] > d["f_nofp"] + 1e-9 else ""))

        nodes.append((inst.ids, (), 0.0))
        push(lat.root, "node", 0)
        t0 = perf_counter()
        while heap:
            f, _, kind, pl = heapq.heappop(heap)
            if f >= U - 1e-12:
                st["stopped_at_f"] = f
                heapq.heappush(heap, (f, 0, kind, pl))
                break
            st["pops"] += 1
            if st["pops"] >= self.cap:
                st["capped"] = True
                heapq.heappush(heap, (f, 0, kind, pl))
                break
            if st["pops"] % 20000 == 0 and psutil.virtual_memory().available < self.min_free_gb * 2 ** 30:
                st["capped"] = st["out_of_memory"] = True
                heapq.heappush(heap, (f, 0, kind, pl))
                break
            if not verbose and st["pops"] % 20000 == 0:
                log(f"  ... pops {st['pops']}, f {f:.4f}, U {U:.4f}, heap {len(heap)}, nodes {st['expanded']},"
                    f" pose checks {st['pose_checks']}, gpu rounds {st['gpu_rounds']}  ({perf_counter() - t0:.0f}s)")
            if kind == "node":
                R, suffix, g = nodes[pl]
                if suffix and self.merge and g > seen[key_of(R, suffix)] + 1e-12:
                    st["superseded"] += 1          # a cheaper equivalent node was found after this one
                    continue
                T(dict(kind="expand", R=list(R), suffix=[[q, int(x)] for q, x in suffix], f=f))
                if verbose:
                    log(f"  pop  f {f:.4f}  expand R = {nm(R)}  suffix = {' '.join(sn(q) for q, _ in suffix) or '-'}")
                expand(pl, f)
                continue
            nid, p, xs, fs, i = pl
            R, suffix, g = nodes[nid]
            x = int(xs[i])
            R1 = tuple(q for q in R if q != p)
            m = pf.setdefault((nid, p), {})
            if x not in m:
                okp, okg = feas.base_ok(R1, p)
                obs = tuple(sorted(geo.oi[(q, xq)] for q, xq in suffix))
                c = [int(v) for v in xs[i:i + self.chunk] if int(v) not in m]
                m.update(zip(c, feas.usable(p, c, okp, okg, obs) if obs else [True] * len(c)))
                st["gpu_rounds"] += 1
            ok = m[x]
            st["pose_checks"] += 1
            if i + 1 < len(xs):
                push(float(fs[i + 1]), "pose", (nid, p, xs, fs, i + 1))
            T(dict(kind="pose", R=list(R), p=p, x=x, key=inst.O["poses"][p][x], rank=int(i), f=f, ok=bool(ok),
                   suffix=[[q, int(xq)] for q, xq in suffix]))
            if not ok:
                st["pose_blocked"] += 1
                if verbose:
                    log(f"  pop  f {f:.4f}  {nm(R1)} --{sn(p)}-> at pose {x}: blocked by the suffix on the table")
                continue
            link = lat.const(p) if not suffix else float(tr[p, suffix[0][0]][suffix[0][1]])
            g1 = g + float(own[p][x]) + link
            suffix1 = ((p, x),) + suffix
            if not R1:
                cost = g1 + float(tr[None, p][x])
                st["complete"] += 1
                if verbose:
                    log(f"  pop  f {f:.4f}  {{}} --{sn(p)}-> at pose {x}: COMPLETE plan, cost {cost:.4f}"
                        + ("  -> new U" if cost < U else ""))
                if cost < U:
                    U, best = cost, suffix1
                    trace.append(dict(kind="U", U=U, order=[q for q, _ in best]))
                    if not verbose:
                        log(f"  new U {U:.4f} at pop {st['pops']}: {' '.join(sn(q) for q, _ in best)}"
                            f"  ({perf_counter() - t0:.0f}s)")
                continue
            if self.merge:
                k = key_of(R1, suffix1)
                if k in seen and seen[k] <= g1 + 1e-12:
                    st["merged"] += 1
                    continue
                seen[k] = g1
            nodes.append((R1, suffix1, g1))
            if verbose:
                log(f"  pop  f {f:.4f}  {nm(R1)} --{sn(p)}-> at pose {x}: ok, child R = {nm(R1)}")
            push(f, "node", len(nodes) - 1)
        seconds = perf_counter() - t0
        left = []
        for f, _, kind, pl in heapq.nsmallest(300, heap, key=lambda e: e[:2]):
            if kind == "node":
                R, suffix, g = nodes[pl]
                left.append(dict(f=f, kind="node", R=list(R), suffix=[q for q, _ in suffix]))
            else:
                nid, p, xs, fs, i = pl
                R, suffix, g = nodes[nid]
                left.append(dict(f=f, kind="pose", R=list(R), p=p, i=int(i), n=len(xs), suffix=[q for q, _ in suffix]))
        st["n_left"] = len(heap)
        return Result(U, best, st, seconds, left, trace)


def plan_cost(inst, plan):
    """cost of a plan recomputed from the coordinates (check of the search's bookkeeping)."""
    c, prev = 0.0, inst.home
    for q, x in plan:
        P = inst.start[q][x] + UP
        c += np.linalg.norm(P - prev) + LIFT + np.linalg.norm(inst.goal[q] + UP - P) + LIFT
        prev = inst.goal[q] + UP
    return float(c + np.linalg.norm(inst.home - prev))
