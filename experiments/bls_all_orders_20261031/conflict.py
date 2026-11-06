"""Conflict-driven joint order x layout search, one reverse best-first tree (2026-11-01).

Node (reverse, as joint_search): R = undecided parts (installed before the suffix, order open),
suffix = decided (part, pose) in forward order, head = its first part. Deciding p (the last of R)
at pose x is checked exactly at that moment: installed set R \\ {p}, table = the suffix.

Each node carries a PLAN: the relaxed completion of R. Held-Karp over the feasible subsets of R,
every step at the cheapest pose of the part that is usable under the installed set AND the current
suffix (footprint + blocking, lazy geometry); conflicts among the parts of R are ignored. Its cost
is an admissible bound of the rest (more suffix = more obstacles = never cheaper), and its last
part p* at pose x* is the node's proposed next decision.
Children are generated lazily in bound order: following the plan is the cheapest child; another
pose of p* ("pose change") or another part as the last of R ("order change") is expanded only
when its optimistic bound becomes the smallest in the open list, i.e. when a conflict has raised
the bound of the plan above it. The first complete node popped is optimal (bounds admissible).

    python -X utf8 -m experiments.bls_all_orders_20261031.conflict --model triangular_frame6 --mode chain
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter
from time import perf_counter

import numpy as np

from .joint import UP, Relaxed
from .lazy import setup
from .run import LIFT, OUT, log

CH = 8          # poses checked per GPU round of one lazy scan


class ConflictSearch:
    def __init__(self, ls, orders, mode):
        self.ls, self.geo, self.O, self.mode = ls, ls.geo, ls.O, mode
        self.rx = Relaxed(ls, orders, mode)
        self.orders = set(orders)
        self.ids = tuple(sorted(self.O["ids"]))
        cd = ls.cd
        self.own, self.tr = {}, {}
        for p in self.ids:
            sp, gp = cd["start"][p], cd["goal"][p]
            o = LIFT + np.linalg.norm((gp + UP) - (sp + UP), axis=1) + LIFT
            self.own[p] = o + np.linalg.norm((gp + UP) - cd["home"]) if mode == "bls" else o
            self.tr[None, p] = np.linalg.norm((sp + UP) - cd["home"], axis=1)
            for q in self.ids:
                if q != p:
                    self.tr[q, p] = self.tr[None, p] if mode == "bls" else np.linalg.norm((sp + UP) - (cd["goal"][q] + UP), axis=1)
        full = self.ids
        self.sets = sorted(set(self.rx.nxt) | {full}, key=len)       # feasible prefix sets
        self.U, self.FP, self.memo = {}, {}, {}
        self.st = Counter()
        self.F0 = self.dp(full, ())[0]                               # suffix-free lattice (generator bounds)

    def const(self, p):
        return self.rx.const(p)

    def free(self, p, suf):
        key = (p, suf)
        if key not in self.FP:
            m = np.ones(len(self.O["poses"][p]), bool)
            for o in suf:
                f = self.ls.fp.get((self.geo.obs[o], p))
                if f is not None:
                    m[f] = False
            self.FP[key] = m
        return self.FP[key]

    def flags(self, items):
        """items [(S, p, suf, xs)]: usable flags under installed set S and table suf, one GPU pass."""
        pq, gq, todo = [], [], []
        for S, p, suf, xs in items:
            m = self.U.setdefault((S, p, suf), {})
            new = [int(x) for x in xs if int(x) not in m]
            if new and suf:
                pq += [(o, p, x) for o in suf for x in new]
                gq += [(o, p) for o in suf]
            todo.append((S, p, suf, new, m))
        self.geo.resolve(pq, gq)
        for S, p, suf, new, m in todo:
            if new:
                okp, okg = self.ls.base_ok(S, p)
                for x, f in zip(new, self.ls.evaluate(p, new, okp, okg, suf) if suf else [True] * len(new)):
                    m[x] = f

    def cheapest(self, queries):
        """queries [(S, last, p, suf)] -> {query: (value, pose)}: cheapest usable pose of p after last."""
        out, run = {}, []
        for q in dict.fromkeys(queries):
            if q in self.memo:
                out[q] = self.memo[q]
                continue
            S, last, p, suf = q
            xs = self.rx.domain(S, p)
            xs = xs[self.free(p, suf)[xs]]
            c = self.tr[last, p][xs] + self.own[p][xs]
            o = np.argsort(c, kind="stable")
            run.append([q, xs[o], c[o], 0])
        while run:
            self.flags([(q[0], q[2], q[3], xs[i:i + CH]) for q, xs, c, i in run])
            nxt = []
            for r in run:
                q, xs, c, i = r
                m = self.U[q[0], q[2], q[3]]
                hit = next((j for j in range(i, min(i + CH, len(xs))) if m[int(xs[j])]), None)
                if hit is not None or i + CH >= len(xs):
                    self.memo[q] = out[q] = (float(c[hit]), int(xs[hit])) if hit is not None else (np.inf, None)
                else:
                    r[3] = i + CH
                    nxt.append(r)
            run = nxt
        return out

    def dp(self, R, suf):
        """Held-Karp over the feasible subsets of R from HOME: F[(S, last)] = (cost, (prev S, prev last, pose))."""
        self.st["dp"] += 1
        Rs = set(R)
        subs = [S for S in self.sets if set(S) <= Rs]
        F = {((), None): (0.0, None)}
        for S in subs:
            if len(S) == len(R):
                break
            lasts = [l for (T, l) in F if T == S]
            edges = [(l, p) for p in self.rx.nxt.get(S, ()) if p in Rs for l in lasts]
            key = (lambda l, p: (S, None if self.mode == "bls" else l, p, suf))
            val = self.cheapest([key(l, p) for l, p in edges])
            for l, p in edges:
                v, x = val[key(l, p)]
                if v == np.inf:
                    continue
                c = F[S, l][0] + v
                S2 = tuple(sorted(S + (p,)))
                if (S2, p) not in F or c < F[S2, p][0]:
                    F[S2, p] = (c, (S, l, x))
        return F, R

    def plan(self, R, suf, head):
        """relaxed completion of R under suffix suf ending into head (p, x) -> (bound, (p*, x*))."""
        F, _ = self.dp(R, suf)
        best = (np.inf, None)
        for (S, q), (c, back) in F.items():
            if S != R:
                continue
            link = self.const(q) if head is None else float(self.tr[q, head[0]][head[1]])
            if c + link < best[0]:
                best = (c + link, (q, back[2]))
        return best

    def run(self, node_cap=200_000):
        t0 = perf_counter()
        full = self.ids
        rest, pl = self.plan(full, (), None)
        root = dict(R=full, suf=(), seq=(), g=0.0, plan=pl, how="root", parent=None)
        self.root_bound, self.root_plan = rest, pl
        heap, cnt = [(rest, 0, "node", root)], 1
        while heap:
            f, _, kind, it = heapq.heappop(heap)
            if kind == "gen":
                node, p, xs, fs, i = it
                if i + 1 < len(xs):
                    heapq.heappush(heap, (fs[i + 1], cnt, "gen", (node, p, xs, fs, i + 1)))
                    cnt += 1
                S, x = tuple(q for q in node["R"] if q != p), int(xs[i])
                self.flags([(S, p, node["suf"], xs[i:i + CH])])
                if not self.U[S, p, node["suf"]][x]:
                    self.st["conflict_prunes"] += 1         # blocked by the suffix: exact, at decision time
                    continue
                head = node["seq"][0] if node["seq"] else None
                g = node["g"] + float(self.own[p][x]) + (self.const(p) if head is None else float(self.tr[p, head[0]][head[1]]))
                if not S:
                    g += float(self.tr[None, p][x])
                o = self.geo.oi[(p, x)]
                suf2 = tuple(sorted(node["suf"] + (o,)))
                rest, pl = self.plan(S, suf2, (p, x)) if S else (0.0, None)
                how = "follow" if (p, x) == node["plan"] else "pose" if p == node["plan"][0] else "order"
                self.st["created_" + how] += 1
                if rest < np.inf:
                    child = dict(R=S, suf=suf2, seq=((p, x),) + node["seq"], g=g, plan=pl, how=how, parent=node)
                    heapq.heappush(heap, (g + rest, cnt, "node", child))
                    cnt += 1
                continue
            node = it
            if not node["R"]:
                self.st["seconds"] = perf_counter() - t0
                return node
            self.st["expanded"] += 1
            self.st["expanded_" + node["how"]] += 1
            if self.st["expanded"] > node_cap:
                break
            R, head = node["R"], (node["seq"][0] if node["seq"] else None)
            for p in R:                                         # p = installed last among R
                S = tuple(q for q in R if q != p)
                if p not in self.rx.nxt.get(S, ()):
                    continue
                xs = self.rx.domain(S, p)
                xs = xs[self.free(p, node["suf"])[xs]]
                if not len(xs):
                    continue
                link = self.const(p) if head is None else float(self.tr[p, head[0]][head[1]])
                if S:                                           # optimistic rest of S into p (suffix-free lattice)
                    lb = np.min([c + self.tr[l, p][xs] for (T, l), (c, _) in self.F0.items() if T == S], axis=0)
                else:
                    lb = self.tr[None, p][xs]
                fs = node["g"] + self.own[p][xs] + link + lb
                o = np.argsort(fs, kind="stable")
                xs, fs = xs[o], fs[o]
                heapq.heappush(heap, (fs[0], cnt, "gen", (node, p, xs, fs, 0)))
                cnt += 1
        self.st["seconds"] = perf_counter() - t0
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mode", default="chain", choices=("bls", "chain"))
    a = ap.parse_args()
    T0 = perf_counter()
    pb, O, geo, ls, orders, t_setup, t_prep = setup(a.model, a.goal_pos, a.k)
    cs = ConflictSearch(ls, orders, a.mode)
    t_init = perf_counter() - T0 - t_setup - t_prep
    best = cs.run()
    log(f"root plan: last part {cs.root_plan[0]}, bound {cs.root_bound:.4f}  (init {t_init:.1f}s, search {cs.st['seconds']:.1f}s)")
    path = []
    n = best
    while n is not None:
        path.append(n)
        n = n["parent"]
    order = [p for p, _ in best["seq"]] if best else None
    on_path = Counter(n["how"] for n in path)
    ref = None
    for f, get in ((OUT / f"joint_all_{a.mode}_{a.model}_k{a.k}.json", lambda d: d["summary"]["best"]),
                   (OUT / f"search_{a.model}_k{a.k}.json",
                    lambda d: min(r[a.mode]["cost"] for r in d["results"] if r[a.mode]["cost"] is not None))):
        if f.exists():
            ref = get(json.loads(f.read_text(encoding="utf-8")))
            break
    g = geo.st
    summary = dict(model=a.model, mode=a.mode, n_orders=len(orders), cost=best["g"] if best else None,
                   reference_optimum=ref, match=None if ref is None or best is None else abs(best["g"] - ref) < 1e-9,
                   order=order, order_is_stable=tuple(order) in cs.orders if order else None,
                   layout={p: O["poses"][p][x] for p, x in best["seq"]} if best else None,
                   root_bound=cs.root_bound, root_plan_last=cs.root_plan[0],
                   path_steps=dict(on_path), **dict(cs.st), total_s=perf_counter() - T0, setup_s=t_setup,
                   prep_s=t_prep, installed_sp=len(ls.base), **g, cached_pose_queries=len(geo.bp),
                   placed_obstacles=len({k[0] for k in geo.bp}))
    log(json.dumps(summary, indent=1, ensure_ascii=False))
    OUT.mkdir(exist_ok=True)
    f = OUT / f"conflict_{a.mode}_{a.model}_k{a.k}.json"
    f.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
