"""Baseline: fix the order first, then search the layout, against the joint search (lattice_joint).

The fixed-order search is the mainline restricted to one order: same instance, candidates, collision
checks and cost, so it returns the exact best layout of that order (the strongest sequential baseline).
The order is the one with the best optimistic score (relaxed cost: no conflicts between table parts).
With --sga the orders are those of the best S/G/A score (Chen Eq. 2, sga_orders.py). With --all every
stable order is searched; the smallest cost must equal the joint optimum.

    python -X utf8 -m experiments.baselines.order_then_layout [--model rack10_tall] [--sga | --all]
Output: experiments/baselines/_out/order_then_layout_<model>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy / Panda3D on Windows (c10.dll)

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out"


def relaxed_cost(lat, order):
    """optimistic cost of one order: the best pose of every step, conflicts between table parts ignored."""
    c, last = 0.0, None
    for k, p in enumerate(order):
        c += lat.step(tuple(sorted(order[:k])), last, p)
        last = p
    return c + lat.const(last)


def fixed_order(inst, order, cap):
    from lattice_joint.lattice import Lattice
    from lattice_joint.search import LatticeSearch, plan_cost
    sub = replace(inst, orders=[tuple(order)])
    lat = Lattice(sub).build()
    t0 = perf_counter()
    res = LatticeSearch(sub, lat, cap=cap).run()
    r = dict(order=list(order), seconds=perf_counter() - t0, pops=res.stats["pops"], pose_checks=res.stats["pose_checks"],
             capped=bool(res.stats.get("capped")))
    if res.plan is None:
        return dict(r, U=None, layout=None)
    assert abs(plan_cost(sub, res.plan) - res.U) < 1e-9
    return dict(r, U=float(res.U), layout=layout(inst, res.plan))


def layout(inst, plan):
    return {q: dict(pose=int(x), key=inst.O["poses"][q][x], xy=[float(v) for v in inst.start[q][x][:2]]) for q, x in plan}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default=None, help="assembly centre x,y,z (m); default: base on the table at x 0.373")
    ap.add_argument("--all", action="store_true", help="search every stable order, best optimistic score first")
    ap.add_argument("--sga", action="store_true", help="the orders with the best S/G/A score")
    ap.add_argument("--cap", type=int, default=5_000_000)
    a = ap.parse_args(argv)
    from lattice_joint.instance import Instance
    from lattice_joint.lattice import Lattice
    from lattice_joint.search import LatticeSearch
    inst = Instance.build(a.model, None if a.goal_pos is None else tuple(float(v) for v in a.goal_pos.split(",")))
    lat = Lattice(inst).build()
    t0 = perf_counter()
    res = LatticeSearch(inst, lat, cap=a.cap).run()
    joint = dict(U=float(res.U), order=[q for q, _ in res.plan], layout=layout(inst, res.plan), root=float(lat.root),
                 seconds=perf_counter() - t0, pops=res.stats["pops"], pose_checks=res.stats["pose_checks"],
                 capped=bool(res.stats.get("capped")))
    print(f"JOINT {joint['U']:.4f}  {' '.join(joint['order'])}  {joint['seconds']:.1f}s, {joint['pops']} pops", flush=True)
    ranked = sorted(((relaxed_cost(lat, o), tuple(o)) for o in inst.orders), key=lambda t: t[0])
    if a.sga:
        sga = OUT.parents[1] / "bls_all_orders_20261031" / "_out" / f"sga_orders_{a.model}.json"
        best = [tuple(o) for o in json.loads(sga.read_text(encoding="utf-8"))["best"]]
        ranked = [t for t in ranked if t[1] in best]
        assert len(ranked) == len(best), "SGA order is not a stable order here"
    rows, f = [], OUT / f"order_then_layout_{a.model}{'_sga' if a.sga else ''}.json"
    OUT.mkdir(exist_ok=True)
    for i, (score, o) in enumerate(ranked if a.all or a.sga else ranked[:1]):
        r = dict(fixed_order(inst, o, a.cap), rank=i + 1, score=float(score))
        rows.append(r)
        u = r["U"]
        gap = "infeasible" if u is None else f"{u:.4f} (+{(u - joint['U']) / joint['U']:.2%})"
        print(f"ORDER rank {i + 1}/{len(ranked)} score {score:.4f} -> {gap}  {' '.join(o)}  {r['seconds']:.1f}s", flush=True)
        f.write_text(json.dumps(dict(model=a.model, joint=joint, n_orders=len(ranked), orders=rows), indent=1), encoding="utf-8")
    ok = [r["U"] for r in rows if r["U"] is not None]
    if a.all and ok:
        print(f"SWEEP best {min(ok):.4f}, joint {joint['U']:.4f}, mean {np.mean(ok):.4f}, worst {max(ok):.4f},"
              f" infeasible {len(rows) - len(ok)}, total {sum(r['seconds'] for r in rows):.0f}s", flush=True)
    print(f"-> {f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
