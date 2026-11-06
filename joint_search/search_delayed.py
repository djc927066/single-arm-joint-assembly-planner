"""Backward beam search with DELAYED initial poses (the default search of run_frame).

Each forward step k depends on: the installed set (parts before k, as a set), the initial poses of the
parts after k (on the table during k), k's own (pose x, grasp g), and - for the empty-hand move INTO
k - the grasp of the part before k (its retreat point is above the assembly, so it depends on its
grasp only, not on where it stood on the table). So when part p is decided (backward), everything
about p's own step is known EXCEPT where the arm comes from, and that is fixed by the grasp of the
part decided at the next level.

Hence an expansion of a node whose head (h, g_h) is decided but not yet placed decides
    the new part p (the last one installed among R) and its grasp g_p, and
    the pose x_h of the head, now that the move into h starts at retreat(g_p).
p's feasible poses and their own-step costs are computed NOW (installed R \\ {p}, table = suffix + h
at x_h) and stored in the child; p's pose is chosen one level later among them. Ranking uses the
same information as the eager search: f = g (resolved) + min over p's feasible poses of (own step +
lower bound of the move into p over every possible predecessor) + h(R \\ {p}).

The checks and costs are those of joint_search.expand, called with the head placed at a candidate
pose: it screens p (step_ok, footprint gap, insertion side with the table, pick side) and returns,
per feasible (x, g): transfer + fixed (p's own step), the connection retreat(g) -> hover of the head
(= the move into the head, exact now) and the pending bound of the move into p. Only the K_h head
poses with the smallest (own step of h + cheapest move into it from any grasp of p) are tried
(prefilter, optimistic); each of them is evaluated exactly and kept as a child of its own, like the
top-B candidates of BLS (K_h = 10 since 2026-10-25; it was 2, which missed the best head pose).
When p is the first part (R \\ {p} empty) its pose is fixed at once (it starts from HOME).
Nodes with the same (R, head part, head grasp, decided table poses) have the same future; only the
cheapest is kept. In the first ``quota_levels`` levels at least ``quota`` nodes per last-installed
part are kept, so no first choice is dropped before its costs are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .expand import expand
from .problem import HOME, t_acc
from .search import Node


@dataclass
class DNode:
    R: frozenset                        # undecided parts (installed before the head)
    head: tuple | None = None           # (part, grasp): decided, pose still open
    opts: dict = field(default_factory=dict)    # head's feasible poses: x -> (own lb, own est, segs)
    suffix: tuple = ()                  # fully decided steps after the head, forward order: (part, x, g)
    table: dict = field(default_factory=dict)   # initial poses of the suffix parts
    g_lb: float = 0.0
    g_est: float = 0.0
    f_lb: float = 0.0
    f_est: float = 0.0
    root: str | None = None             # last-installed part (the first backward decision)
    segs: tuple = ()

    def key(self):
        return (self.R, self.head, frozenset(self.table.items()))


def _split(c):
    """An expand Child -> (own lb, own est, connection lb, connection est, own segs, connection seg)."""
    own = [s for s in c.segs if s["kind"] in ("transfer", "fixed") or (s["kind"] == "transit" and s.get("after") == "HOME")]
    conn = next(s for s in c.segs if s["kind"] == "return_home" or (s["kind"] == "transit" and s.get("after") != "HOME"))
    return (sum(s["lb"] for s in own), sum(s["est"] for s in own), conn["lb"], conn["est"], own, conn)


def _children_from(pb, node, p, x_h, kids, hR, own_h):
    """Group the expand children of p (head at x_h, or the root when x_h is None) by grasp."""
    R1 = node.R - {p}
    by = {}
    for c in kids:
        o_lb, o_est, c_lb, c_est, own, conn = _split(c)
        d = by.setdefault(c.g, dict(conn=(c_lb, c_est, conn), opts={}))
        d["opts"][c.x] = (o_lb, o_est, tuple(own), c.pend_lb)
    out = []
    for g, d in by.items():
        c_lb, c_est, conn = d["conn"]
        if x_h is None:
            head_lb, head_est, suffix, table, hsegs = c_lb, c_est, node.suffix, dict(node.table), (conn,)
        else:
            h, gh = node.head
            w_lb, w_est, wsegs = own_h
            head_lb, head_est = c_lb + w_lb, c_est + w_est
            suffix, table = ((h, x_h, gh),) + node.suffix, dict(node.table, **{h: x_h})
            hsegs = (conn,) + wsegs
        g_lb, g_est = node.g_lb + head_lb, node.g_est + head_est
        if not R1:                                            # p is the first part: from HOME, pose now
            x, (o_lb, o_est, own, _) = min(d["opts"].items(), key=lambda t: t[1][1])
            out.append(DNode(R=R1, suffix=((p, x, g),) + suffix, table=dict(table, **{p: x}),
                             g_lb=g_lb + o_lb, g_est=g_est + o_est, f_lb=g_lb + o_lb, f_est=g_est + o_est,
                             root=node.root or p, segs=own + hsegs + node.segs))
            continue
        b_lb = min(o_lb + pend for o_lb, _, _, pend in d["opts"].values())
        b_est = min(o_est + pend for _, o_est, _, pend in d["opts"].values())
        opts = {x: (o_lb, o_est, own) for x, (o_lb, o_est, own, _) in d["opts"].items()}
        out.append(DNode(R=R1, head=(p, g), opts=opts, suffix=suffix, table=table, g_lb=g_lb, g_est=g_est,
                         f_lb=g_lb + b_lb + hR, f_est=g_est + b_est + hR, root=node.root or p,
                         segs=hsegs + node.segs))
    return out


def expand_delayed(pb, node, p, M=5, per_x=2, K_h=10):
    """Children of ``node`` deciding p (+ grasp) and the head's pose. -> list[DNode], stats."""
    cat = pb.cat
    R1 = node.R - {p}
    if not pb.scene.step_ok(R1, p):
        return [], dict(reason="unstable_or_blocked_insertion")
    hR = pb.h_lb(R1)
    if node.head is None:                                     # root: p is installed last
        kids, st = expand(pb, Node(R=node.R), p)
        out = _children_from(pb, node, p, None, kids, hR, None)
    else:
        h, gh = node.head
        # head poses to try: own step of h + cheapest move into it from any retreat of p
        ret = np.stack([v["retreat"] for v in cat["goal"][p].values()])
        xs = list(node.opts)
        hov = np.stack([cat["pick"][h][x][gh]["hover"] for x in xs])
        into = t_acc(ret[None], hov[:, None]).min(1)
        score = np.array([node.opts[x][1] for x in xs]) + into
        out, st = [], {}
        for j in np.argsort(score)[:K_h]:
            x_h = xs[j]
            onode = Node(R=node.R, suffix=((h, x_h, gh),) + node.suffix, table=dict(node.table, **{h: x_h}))
            kids, st = expand(pb, onode, p)
            out += _children_from(pb, node, p, x_h, kids, hR, node.opts[x_h])
    out.sort(key=lambda n: n.f_est)
    if node.head is None:                                     # root: the M best grasps of p
        return out[:M], st
    # every head pose tried gets its best child (each pose a child of its own, as BLS keeps its top
    # candidates), then up to per_x per pose, at most max(M, K_h) children in all
    xkey = lambda n: n.suffix[1][1] if (not n.R and len(n.suffix) > 1) else n.suffix[0][1]
    kept, per = [], {}
    for n in out:
        if xkey(n) not in per:
            kept.append(n)
            per[xkey(n)] = 1
    for n in out:
        if len(kept) >= max(M, K_h):
            break
        if n not in kept and per[xkey(n)] < per_x:
            kept.append(n)
            per[xkey(n)] += 1
    kept.sort(key=lambda n: n.f_est)
    return kept[:max(M, K_h)], st


def beam_search_delayed(pb, B=10, M=5, N=10 ** 9, per_x=2, K_h=10, quota=2, quota_levels=2, log=print):
    h0 = pb.h_lb(frozenset(pb.ids))
    root = DNode(R=frozenset(pb.ids), f_lb=h0, f_est=h0)
    level, complete = [root], []
    st = dict(expansions=0, children=0, levels=[])
    t0 = perf_counter()
    depth = 0
    while level:
        depth += 1
        tl = perf_counter()
        children = {}
        for node in level:
            for p in sorted(node.R):
                kids, _ = expand_delayed(pb, node, p, M=M, per_x=per_x, K_h=K_h)
                st["expansions"] += 1
                for c in kids:
                    if not c.R:
                        complete.append(c)
                        continue
                    k = c.key()
                    if k not in children or children[k].f_est > c.f_est:
                        children[k] = c
        st["children"] += len(children)
        ranked = sorted(children.values(), key=lambda n: n.f_est)
        if depth <= quota_levels and quota:
            keep, cnt = [], {}
            for n in ranked:                                     # at least `quota` per first decision
                if cnt.get(n.root, 0) < quota:
                    keep.append(n)
                    cnt[n.root] = cnt.get(n.root, 0) + 1
            kid = {id(n) for n in keep}
            level = sorted(keep + [n for n in ranked if id(n) not in kid][:max(0, B - len(keep))], key=lambda n: n.f_est)
        else:
            level = ranked[:B]
        st["levels"].append(dict(depth=depth, children=len(children), kept=len(level), complete=len(complete),
                                 best_f_est=ranked[0].f_est if ranked else None, s=perf_counter() - tl))
        log(f"  level {depth}: {len(children)} children, keep {len(level)}"
            + (f", {len(complete)} complete" if complete else "")
            + (f", best f_est {ranked[0].f_est:.3f} s" if ranked else "") + f"  [{perf_counter() - tl:.1f}s]")
    complete.sort(key=lambda n: n.g_est)
    st["search_s"] = perf_counter() - t0
    return complete[:N], st
