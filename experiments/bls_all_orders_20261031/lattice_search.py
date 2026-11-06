"""The lattice search end to end, every step logged (chain cost, Euclidean) (2026-11-01).

1. D(S, p) per arrow: table poses of p usable under its installed set S (installed-set checks).
2. F(S, last): forward Held-Karp on the lattice, each arrow at its cheapest pose in D, no
   conflicts between table parts -> optimistic cost of installing S from HOME ending with last.
3. Reverse best-first. Node = (R undecided, suffix = decided (part, pose) in forward order, g =
   exact cost of the suffix from its head on, return to HOME included). Child = choose p as the
   last of R (an arrow R \\ {p} --p--> R) and its pose x; real check at that moment: x in
   D(R \\ {p}, p), footprint clear of the suffix parts (they lie on the table) and some grasp not
   blocked by them (lazy geometry, as lazy.py). Priority
      f = g + own(p, x) + link(p -> head) + min_q [ F(R \\ {p}, q) + transit(q -> p at x) ]
   (exact except the prefix, optimistic): an admissible bound of every plan below.
   Poses of one (node, p) are generated lazily in f order: a pose is checked only when popped.
4. Incumbent U = best complete plan found so far; an entry with f >= U is never expanded.
5. Stop when the smallest f in the heap >= U: U is the optimum.
Logs every pop, saves _out/lattice_search_{model}.json (trace for viz_lattice_search.py).

    python -X utf8 -m experiments.bls_all_orders_20261031.lattice_search [--model triangular_frame6]
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

SHORT = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--quiet", action="store_true", help="log only a summary of the pops (no trace)")
    ap.add_argument("--cap", type=int, default=5_000_000, help="stop after this many pops (then report the gap)")
    ap.add_argument("--no-merge", action="store_true", help="keep nodes that differ only in the order of the suffix")
    ap.add_argument("--chunk", type=int, default=8, help="poses of one generator checked per GPU round")
    a = ap.parse_args()
    T0 = perf_counter()
    pb, O, geo, ls, orders, *_ = setup(a.model, a.goal_pos, a.k)
    sn = (lambda p: SHORT.get(p, p.replace("shelf_", "s").replace("post_", "p")))
    nm = lambda S: "{" + ",".join(sn(q) for q in S) + "}" if S else "{}"
    rx = Relaxed(ls, orders, "chain")
    cd = ls.cd
    ids = tuple(sorted(O["ids"]))
    own = {p: LIFT + np.linalg.norm(cd["goal"][p] + UP - (cd["start"][p] + UP), axis=1) + LIFT for p in ids}
    tr = {(None, p): np.linalg.norm(cd["start"][p] + UP - cd["home"], axis=1) for p in ids}
    for p in ids:
        for q in ids:
            if q != p:
                tr[q, p] = np.linalg.norm(cd["start"][p] + UP - (cd["goal"][q] + UP), axis=1)
    st = Counter()

    # ---- 1 + 2: D on every arrow, forward DP F ------------------------------------------------
    t1 = perf_counter()
    arrows = [(S, p) for S in rx.nxt for p in rx.nxt[S]]
    for S, p in arrows:
        rx.domain(S, p)
    F = {((), None): 0.0}
    for S in sorted(rx.nxt, key=len):
        for (T, last), v in list(F.items()):
            if T != S:
                continue
            for p in rx.nxt[S]:
                S2 = tuple(sorted(S + (p,)))
                c = v + rx.step(S, last, p)
                if c < F.get((S2, p), np.inf):
                    F[S2, p] = c
    root = min(F[ids, q] + rx.const(q) for (T, q) in F if T == ids)
    lasts = {}
    for (T, q) in F:
        lasts.setdefault(T, []).append(q)
    t_bound = perf_counter() - t1
    log(f"[1] D(S, p) on {len(arrows)} arrows; [2] F on {len(F)} (S, last) states; root bound {root:.4f}  ({t_bound:.1f}s)")

    def lb_prefix(R, p, xs):
        """min over the possible lasts q of R of F(R, q) + transit(q -> p at xs); HOME if R is empty."""
        if not R:
            return tr[None, p][xs]
        return np.min(np.stack([F[R, q] + tr[q, p][xs] for q in lasts.get(R, [])]), axis=0) \
            if lasts.get(R) else np.full(len(xs), np.inf)

    def footprint_free(p, suffix):
        m = np.ones(len(O["poses"][p]), bool)
        for q, x in suffix:
            f = ls.fp.get(((q, x), p))
            if f is not None:
                m[f] = False
        return m

    # ---- 3 + 4 + 5: reverse best-first with incumbent ---------------------------------------
    nodes = []                      # id -> (R, suffix, g)
    heap, cnt = [], 0
    U, best = np.inf, None
    trace = []
    T = (lambda d: None) if a.quiet else trace.append
    pf = {}                         # (node, part) -> {pose: usable}, filled a chunk at a time
    # The rest of the search below a node depends only on R, the head (part, pose) and the SET of
    # suffix (part, pose): those parts all lie on the table while R is installed, and the head is
    # where the arm goes after R. The order inside the suffix only changes g. So of the nodes with
    # the same key only the one with the smallest g is kept (merging, exact).
    seen = {}
    key_of = lambda R, suffix: (R, suffix[0], frozenset(suffix))

    def push(f, kind, payload):
        nonlocal cnt
        heapq.heappush(heap, (f, cnt, kind, payload))
        cnt += 1

    def expand(nid):
        R, suffix, g = nodes[nid]
        st["expanded"] += 1
        gens = []
        for p in R:
            R1 = tuple(q for q in R if q != p)
            if p not in rx.nxt.get(R1, ()):                     # R1 --p--> R is not an arrow
                continue
            xs = rx.domain(R1, p)
            xs = xs[footprint_free(p, suffix)[xs]]
            if not len(xs):
                st["empty_domain"] += 1
                continue
            link = rx.const(p) if not suffix else float(tr[p, suffix[0][0]][suffix[0][1]])
            f = g + own[p][xs] + link + lb_prefix(R1, p, xs)
            o = np.argsort(f, kind="stable")
            st["generators"] += 1
            push(float(f[o[0]]), "pose", (nid, p, xs[o], f[o], 0))
            if not a.quiet:
                gens.append(dict(p=p, f=float(f[o[0]]), n=int(len(xs)), key=O["poses"][p][int(xs[o[0]])],
                                 f_nofp=float(np.min(g + own[p][rx.domain(R1, p)] + link + lb_prefix(R1, p, rx.domain(R1, p))))))
        T(dict(kind="gens", R=list(R), suffix=[q for q, _ in suffix], gens=gens))
        if not a.quiet:
            for d in gens:
                log(f"         generator {nm(tuple(q for q in R if q != d['p']))} --{sn(d['p'])}->: {d['n']} poses,"
                    f" best f {d['f']:.4f}" + (f" (without the suffix on the table {d['f_nofp']:.4f})"
                                               if d["f"] > d["f_nofp"] + 1e-9 else ""))

    nodes.append((ids, (), 0.0))
    push(root, "node", 0)
    t2 = perf_counter()
    while heap:
        f, _, kind, pl = heapq.heappop(heap)
        if f >= U - 1e-12:
            st["stopped_at_f"] = f
            break
        st["pops"] += 1
        if st["pops"] >= a.cap:
            st["capped"] = True
            heapq.heappush(heap, (f, 0, kind, pl))
            break
        if a.quiet and st["pops"] % 20000 == 0:
            log(f"  ... pops {st['pops']}, f {f:.4f}, U {U:.4f}, heap {len(heap)}, nodes {st['expanded']},"
                f" pose checks {st['pose_checks']}, gpu rounds {st['gpu_rounds']}  ({perf_counter() - t2:.0f}s)")
        if kind == "node":
            R, suffix, g = nodes[pl]
            if suffix and not a.no_merge and g > seen[key_of(R, suffix)] + 1e-12:
                st["superseded"] += 1          # a cheaper equivalent node was found after this one
                continue
            T(dict(kind="expand", R=list(R), suffix=[[q, int(x)] for q, x in suffix], f=f))
            if not a.quiet:
                log(f"  pop  f {f:.4f}  expand R = {nm(R)}  suffix = {' '.join(sn(q) for q, _ in suffix) or '-'}")
            expand(pl)
            continue
        nid, p, xs, fs, i = pl
        R, suffix, g = nodes[nid]
        x = int(xs[i])
        R1 = tuple(q for q in R if q != p)
        okp, okg = ls.base_ok(R1, p)
        obs = tuple(sorted(geo.oi[(q, xq)] for q, xq in suffix))
        m = pf.setdefault((nid, p), {})
        if x not in m:
            c = [int(v) for v in xs[i:i + a.chunk] if int(v) not in m]
            m.update(zip(c, ls.usable(p, c, okp, okg, obs) if obs else [True] * len(c)))
            st["gpu_rounds"] += 1
        ok = m[x]
        st["pose_checks"] += 1
        if i + 1 < len(xs):
            push(float(fs[i + 1]), "pose", (nid, p, xs, fs, i + 1))
        T(dict(kind="pose", R=list(R), p=p, x=x, key=O["poses"][p][x], rank=int(i), f=f, ok=bool(ok), suffix=[[q, int(xq)] for q, xq in suffix]))
        if not ok:
            st["pose_blocked"] += 1
            if not a.quiet:
                log(f"  pop  f {f:.4f}  {nm(R1)} --{sn(p)}-> at pose {x}: blocked by the suffix on the table")
            continue
        link = rx.const(p) if not suffix else float(tr[p, suffix[0][0]][suffix[0][1]])
        g1 = g + float(own[p][x]) + link
        suffix1 = ((p, x),) + suffix
        if not R1:
            cost = g1 + float(tr[None, p][x])
            st["complete"] += 1
            if not a.quiet:
                log(f"  pop  f {f:.4f}  {{}} --{sn(p)}-> at pose {x}: COMPLETE plan, cost {cost:.4f}"
                    + ("  -> new U" if cost < U else ""))
            if cost < U:
                U, best = cost, suffix1
                trace.append(dict(kind="U", U=U, order=[q for q, _ in best]))
                if a.quiet:
                    log(f"  new U {U:.4f} at pop {st['pops']}: {' '.join(sn(q) for q, _ in best)}  ({perf_counter() - t2:.0f}s)")
            continue
        if not a.no_merge:
            k = key_of(R1, suffix1)
            if k in seen and seen[k] <= g1 + 1e-12:
                st["merged"] += 1
                continue
            seen[k] = g1
        nodes.append((R1, suffix1, g1))
        if not a.quiet:
            log(f"  pop  f {f:.4f}  {nm(R1)} --{sn(p)}-> at pose {x}: ok, child R = {nm(R1)}")
        push(f, "node", len(nodes) - 1)
    t_search = perf_counter() - t2
    left_detail = []                # what was never expanded: (f, arrow, which pose of its generator)
    for f, _, kind, pl in sorted(heap) + ([(st["stopped_at_f"], 0, kind, pl)] if "stopped_at_f" in st else []):
        if kind == "node":
            R, suffix, g = nodes[pl]
            left_detail.append(dict(f=f, kind="node", R=list(R), suffix=[q for q, _ in suffix]))
        else:
            nid, p, xs, fs, i = pl
            R, suffix, g = nodes[nid]
            left_detail.append(dict(f=f, kind="pose", R=list(R), p=p, i=int(i), n=len(xs), suffix=[q for q, _ in suffix]))
    left_detail.sort(key=lambda d: d["f"])
    for d in left_detail[:15]:
        log(f"  never expanded: f {d['f']:.4f}  " + (f"{nm(tuple(q for q in d['R'] if q != d['p']))} --{sn(d['p'])}->"
            f" pose #{d['i'] + 1} of {d['n']}" if d["kind"] == "pose" else f"node R = {nm(tuple(d['R']))}")
            + f"  (suffix {' '.join(sn(q) for q in d['suffix']) or '-'})")
    left = [e[0] for e in heap]
    order = [q for q, _ in best]
    # recompute the plan's cost from scratch
    c, prev = 0.0, cd["home"]
    for q, x in best:
        P = cd["start"][q][x] + UP
        c += np.linalg.norm(P - prev) + LIFT + np.linalg.norm(cd["goal"][q] + UP - P) + LIFT
        prev = cd["goal"][q] + UP
    c += np.linalg.norm(cd["home"] - prev)
    assert abs(c - U) < 1e-9, (c, U)
    log(f"\nresult: order {' -> '.join(sn(q) for q in order)}, cost U = {U:.4f} (recomputed {c:.4f})")
    log(f"  layout: " + ", ".join(f"{sn(q)} {O['poses'][q][x]}" for q, x in best))
    log(f"  search {t_search:.1f}s: pops {st['pops']}, nodes expanded {st['expanded']}, pose checks {st['pose_checks']}"
        f" (blocked {st['pose_blocked']}), complete plans {st['complete']}; merged {st['merged']},"
        f" superseded {st['superseded']}")
    if st.get("capped"):
        log(f"  CAPPED at {a.cap} pops: lower bound {min(left):.4f}, gap to U {(U - min(left)) / U:.2%}")
    log(f"  stopped: smallest f left = {st.get('stopped_at_f', np.inf):.4f} >= U; {len(left)} heap entries never"
        f" expanded (all f >= U), min {min(left) if left else np.inf:.4f}")
    space = sum(len(rx.domain(S, p)) for S, p in arrows)
    log(f"  for scale: {len(orders)} orders, {len(arrows)} arrows, {space} (arrow, pose) pairs in the D's;"
        f" total {perf_counter() - T0:.0f}s")
    (OUT / f"lattice_search_{a.model}{'_nomerge' if a.no_merge else ''}.json").write_text(json.dumps(dict(
        U=U, order=order, layout=[[q, int(x), O["poses"][q][x]] for q, x in best], root=root,
        stats=dict(st), heap_left=sorted(left)[:50], n_left=len(left), left=left_detail[:300], trace=trace), indent=1,
        default=float), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
