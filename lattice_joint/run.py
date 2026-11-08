"""Exact joint search of assembly order x initial layout on the installed-set lattice.

    python -X utf8 -m lattice_joint.run [--model triangular_frame6] [--quiet] [--no-merge] [--no-prices]
Output: lattice_joint/_output/<model>.json (plan, stats, the never expanded entries, trace).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy / Panda3D on Windows (c10.dll)

OUT = Path(__file__).resolve().parent / "_output"
FRAME = {"post_A": "A", "post_B": "B", "post_C": "C", "beam_AB": "AB", "beam_BC": "BC", "beam_CA": "CA"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default=None, help="assembly centre x,y,z (m); default: base on the table at x 0.373")
    ap.add_argument("--k", type=int, default=8, help="grasps kept per table pose (smallest fixed + transfer bound)")
    ap.add_argument("--quiet", action="store_true", help="progress every 20000 pops instead of every pop (no trace)")
    ap.add_argument("--no-merge", action="store_true", help="keep nodes that differ only in the order of the suffix")
    ap.add_argument("--no-self-mesh", action="store_true",
                    help="no mesh check of the gripper vs the part it handles (as the experiment)")
    ap.add_argument("--chunk", type=int, default=8, help="poses of one generator checked per GPU round")
    ap.add_argument("--cap", type=int, default=5_000_000, help="stop after this many pops (then report the gap)")
    ap.add_argument("--no-prices", action="store_true", help="plain F bound, without the table prices (prices.py)")
    ap.add_argument("--min-free-gb", type=float, default=2.5, help="stop when less free RAM than this is left")
    a = ap.parse_args(argv)
    from .instance import Instance
    from .lattice import Lattice
    from .prices import table_prices
    from .search import LatticeSearch, plan_cost
    from .geometry import log
    T0 = perf_counter()
    center = None if a.goal_pos is None else tuple(float(v) for v in a.goal_pos.split(","))
    inst = Instance.build(a.model, center, a.k, self_mesh=not a.no_self_mesh)
    sn = lambda p: FRAME.get(p, p.replace("shelf_", "s"))
    lat = Lattice(inst).build()
    root0, t1 = lat.root, perf_counter()
    probe = LatticeSearch(inst, lat, chunk=a.chunk)
    U0, plan0 = probe.dive()
    if not a.no_prices:
        lat.root = max(root0, table_prices(inst, lat, 1.2 * root0 if U0 is None else U0))
        U1, plan1 = probe.dive()
        if U1 is not None and (U0 is None or U1 < U0):
            U0, plan0 = U1, plan1
    bound_s = perf_counter() - t1
    log(f"root bound {root0:.4f} -> {lat.root:.4f}, first incumbent {U0}  ({bound_s:.0f}s)")
    res = LatticeSearch(inst, lat, merge=not a.no_merge, chunk=a.chunk, cap=a.cap, verbose=not a.quiet, short=sn,
                        U0=float("inf") if U0 is None else U0, plan0=plan0, min_free_gb=a.min_free_gb).run()
    st = res.stats
    if res.plan is None:
        log(f"no plan found ({'capped' if st.get('capped') else 'infeasible'}): pops {st['pops']}, search {res.seconds:.0f}s,"
            f" smallest f left {min((e['f'] for e in res.left), default=float('inf')):.4f}")
        return 1
    c = plan_cost(inst, res.plan)
    assert abs(c - res.U) < 1e-9, (c, res.U)
    O = inst.O
    log(f"\nresult: order {' -> '.join(sn(q) for q, _ in res.plan)}, cost {res.U:.4f} (recomputed {c:.4f})")
    log("  layout: " + ", ".join(f"{sn(q)} {O['poses'][q][x]}" for q, x in res.plan))
    log(f"  search {res.seconds:.1f}s: pops {st['pops']}, nodes expanded {st['expanded']}, pose checks {st['pose_checks']}"
        f" (blocked {st['pose_blocked']}), complete plans {st['complete']}; merged {st['merged']}, superseded {st['superseded']}")
    gap_lb = min((e["f"] for e in res.left), default=float("inf"))
    if st.get("capped"):
        log(f"  CAPPED at {a.cap} pops: lower bound {gap_lb:.4f}, gap {(res.U - gap_lb) / res.U:.2%}")
    else:
        log(f"  optimal: smallest f left {gap_lb:.4f} >= U; {st['n_left']} entries never expanded")
    space = sum(len(lat.domain(S, p)) for S, p in lat.arrows)
    total = perf_counter() - T0
    log(f"  for scale: {len(inst.orders)} orders, {len(lat.arrows)} arrows, {space} (arrow, pose) pairs; total {total:.0f}s")
    OUT.mkdir(exist_ok=True)
    f = OUT / f"{a.model}{'_nomerge' if a.no_merge else ''}{'_plain' if a.no_prices else ''}.json"
    f.write_text(json.dumps(dict(
        model=a.model, U=res.U, order=[q for q, _ in res.plan], layout=[[q, int(x), O["poses"][q][x]] for q, x in res.plan],
        root=lat.root, root_plain=root0, optimal=not st.get("capped"), lower=min(gap_lb, res.U), stats=dict(st),
        seconds=dict(inst.timing, lattice_s=lat.seconds, bound_s=bound_s, search_s=res.seconds, total_s=total),
        n_orders=len(inst.orders), n_arrows=len(lat.arrows), n_sets=len(lat.nxt) + 1, n_poses=sum(len(inst.O["poses"][p]) for p in inst.ids),
        left=res.left, trace=res.trace), indent=1, default=float), encoding="utf-8")
    log(f"-> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
