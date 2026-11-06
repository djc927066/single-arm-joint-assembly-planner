"""Backward beam search over (order, initial layout, assembly grasp).

Node (backward = the last parts are decided first):
    R      undecided parts; in the forward plan they are installed BEFORE the decided suffix,
           in an order still open (only the set matters for the obstacles of later steps)
    suffix decided steps (part, pose key, grasp), in forward order
    head   first part of the suffix with its (pose, grasp) = interface state: its transit start is
           still open and is resolved when the part before it is decided
    table  initial poses of the suffix parts (they are on the table during every earlier step)
    g_lb / g_est   resolved segments: lower bound / estimate (bound + blocked-ness correction)
    nxt    the part installed right before the head (order lookahead), decided next
    pend   lower bound of the head's open transit, over the retreat points of nxt's grasps

Level 1 decides the last part p with its (pose, grasp) and the part nxt before it (one of the parts
that can be installed right before p); its cost includes the retreat back to HOME. Every later level
expands only the node's nxt: deciding its (pose, grasp) resolves the move into the head exactly, and
it picks the part before nxt. The first part starts from HOME.
Values of a node:  f_lb  = g_lb  + pend + h(R)    never above the true plan time (pruning)
                   f_est = g_est + pend + h(R)    ordering and beam selection
Each expansion keeps the M best (x, g, nxt) by f_est (at most ``per_x`` per pose and nxt, so the
layout keeps alternatives); each level keeps the B best nodes; identical nodes (same R, suffix,
nxt) are merged. Complete nodes (R empty) are full plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from lazy_cost.scene import key_xy

from .expand import expand


@dataclass
class Node:
    R: frozenset
    suffix: tuple = ()                 # ((part, x, g), ...) forward order
    table: dict = field(default_factory=dict)
    g_lb: float = 0.0
    g_est: float = 0.0
    pend: float = 0.0
    segs: tuple = ()
    f_lb: float = 0.0
    f_est: float = 0.0
    nxt: str | None = None             # the part installed right before the head (decided next)

    @property
    def head(self):
        return self.suffix[0] if self.suffix else None

    def key(self):
        return (self.R, self.suffix, self.nxt)


def _pend_from(pb, p, kids, q):
    """Lower bound of the move into p for each child (x, g) when the part before p is q (any of q's
    grasps): min over q's retreat points of the joint-time bound to p's hover."""
    from .problem import t_acc
    cat = pb.cat
    P = np.stack([v["retreat"] for v in cat["goal"][q].values()])
    HV = np.stack([cat["pick"][p][c.x][c.g]["hover"] for c in kids])
    return t_acc(P[None], HV[:, None]).min(1)


def lattice_bound(pb):
    """Subset-lattice lower bound D[(S, q)]: the relaxed time to install the set S from HOME in an order
    that is feasible for scene.step_ok and ends with q (q's own step and the move into q included).
    Relaxation: every part's own time (fixed + transfer) and every move a -> b at their minima over all
    (pose, grasp); table blocking ignored. Never above the true time of S (admissible); missing key =
    no feasible order ends with q."""
    from itertools import combinations
    from .problem import HOME, t_acc
    cat, ids = pb.cat, list(pb.ids)
    own, HV, RET = {}, {}, {}
    for p in ids:
        goal = cat["goal"][p]
        rows = [(x, g, e) for x, gs in cat["pick"][p].items() for g, e in gs.items() if g in goal]
        own[p] = min(float(t_acc(e["lift"], goal[g]["preins"])) + pb.fix_pick[p, x, g] + pb.fix_goal[p, g]
                     for x, g, e in rows)
        HV[p] = np.stack([e["hover"] for _, _, e in rows])
        RET[p] = np.stack([v["retreat"] for v in goal.values()])
    tmin = {(a, b): float(t_acc(RET[a][:, None], HV[b][None]).min()) for a in ids for b in ids if a != b}
    home = {b: float(t_acc(HOME[None], HV[b]).min()) for b in ids}
    D = {}
    for r in range(1, len(ids) + 1):
        for S in combinations(ids, r):
            S = frozenset(S)
            for q in S:
                prev = S - {q}
                if not pb.scene.step_ok(prev, q):
                    continue
                if not prev:
                    D[S, q] = own[q] + home[q]
                    continue
                v = min((D[prev, a] + tmin[a, q] for a in prev if (prev, a) in D), default=None)
                if v is not None:
                    D[S, q] = own[q] + v
    return D


def _merge(children, pb):
    """Children with the same future: same undecided set R, same part before the head (nxt), same table
    (every suffix part at the same pose, hence the same blocking of R's poses) and the same head pose;
    the suffix ORDER may differ. Their futures differ only in the move into the head, which depends on
    the head's grasp: label 1 dominates label 2 if g_est1 + T(hover1, hover2) <= g_est2 (triangle
    inequality of the joint-time bound). -> kept children, number dropped."""
    from .problem import t_acc
    groups = {}
    for n in children:
        h = n.head
        groups.setdefault((n.R, n.nxt, frozenset(n.table.items()), h[0], h[1]), []).append(n)
    kept, dropped = [], 0
    for g in groups.values():
        g.sort(key=lambda n: n.g_est)
        live = []
        for n in g:
            hv = pb.cat["pick"][n.head[0]][n.head[1]][n.head[2]]["hover"]
            if any(m.g_est + float(t_acc(mh, hv)) <= n.g_est + 1e-12 for m, mh in live):
                dropped += 1
                continue
            live.append((n, hv))
        kept += [n for n, _ in live]
    return kept, dropped


def beam_search(pb, B=10, M=10, N=5, per_x=2, log=print, bound="sum", merge=False):
    """bound: "sum" = h(R) sum of per-part minima (default); "lattice" = D[(R, nxt)] of lattice_bound,
    orders that cannot be completed are pruned. merge: drop children dominated by a child with the same
    future (_merge) before the beam is cut."""
    D = lattice_bound(pb) if bound == "lattice" else None
    root = Node(R=frozenset(pb.ids))
    root.f_lb = root.f_est = pb.h_lb(root.R)
    level, complete = [root], []
    st = dict(expansions=0, pairs=0, screened=0, pruned=0, orient_fallback=0, children=0, levels=[])
    t0 = perf_counter()
    depth = 0
    while level:
        depth += 1
        tl = perf_counter()
        children = {}
        lv = dict(nodes=len(level), expansions=0, step_ok_fail=0, pairs=0, screened=0, feasible=0, candidates=0, kept_per_node=0)
        for node in level:
            for p in ([node.nxt] if node.nxt else sorted(node.R)):        # root: every part can be last
                kids, s = expand(pb, node, p)
                st["expansions"] += 1
                lv["expansions"] += 1
                lv["step_ok_fail"] += s.get("reason") == "unstable_or_blocked_insertion"
                lv["pairs"] += s["pairs"]
                lv["screened"] += s.get("screened", 0)
                lv["feasible"] += len(kids)
                st["pairs"] += s["pairs"]
                st["screened"] += s.get("screened", 0)
                st["pruned"] += s["pruned"]
                st["orient_fallback"] += s.get("orient_fallback", 0)
                R1 = node.R - {p}
                hR = pb.h_lb(R1)
                # the part before p: one of its possible predecessors; none when p is the first part
                qs = (pb.preds(R1, p) or sorted(R1)) if R1 and kids else [None]
                scored = []
                for q in qs:
                    if D is not None:
                        if q and (R1, q) not in D:          # no feasible order of R1 ends with q
                            lv["dead_q"] = lv.get("dead_q", 0) + 1
                            continue
                        hR = D[R1, q] if q else 0.0
                    pend = _pend_from(pb, p, kids, q) if q else [c.pend_lb for c in kids]
                    for c, pd in zip(kids, pend):
                        g_lb, g_est = node.g_lb + c.add_lb, node.g_est + c.add_est
                        scored.append((g_est + pd + hR, g_lb + pd + hR, g_lb, g_est, c, float(pd), q))
                scored.sort(key=lambda t: t[0])
                lv["candidates"] += len(scored)
                kept, per = [], {}
                for t in scored:
                    xy = (t[6], key_xy(t[4].x))               # per position (and part before), any orientation
                    if per.get(xy, 0) >= per_x:
                        continue
                    per[xy] = per.get(xy, 0) + 1
                    kept.append(t)
                    if len(kept) == M:
                        break
                lv["kept_per_node"] += len(kept)
                for f_est, f_lb, g_lb, g_est, c, pd, q in kept:
                    n = Node(R=R1, suffix=((p, c.x, c.g),) + node.suffix, table=dict(node.table, **{p: c.x}),
                             g_lb=g_lb, g_est=g_est, pend=pd, segs=tuple(c.segs) + node.segs, f_lb=f_lb, f_est=f_est,
                             nxt=q)
                    if n.key() not in children or children[n.key()].f_est > n.f_est:
                        children[n.key()] = n
        st["children"] += len(children)
        pool = list(children.values())
        if merge:
            pool, lv["merged"] = _merge(pool, pb)
        ranked = sorted(pool, key=lambda n: n.f_est)
        done = [n for n in ranked if not n.R]
        complete += done
        level = [n for n in ranked if n.R][:B]
        st["levels"].append(dict(depth=depth, **lv, children=len(children), kept=len(level), complete=len(done),
                                 best_f_est=ranked[0].f_est if ranked else None, s=perf_counter() - tl))
        log(f"  level {depth}: {len(children)} children, keep {len(level)}"
            + (f", {len(done)} complete" if done else "") + f", best f_est {ranked[0].f_est:.3f} s"
            + f"  [{perf_counter() - tl:.1f}s]" if ranked else f"  level {depth}: no feasible child")
    complete.sort(key=lambda n: n.g_est)
    st["search_s"] = perf_counter() - t0
    return complete[:N], st
