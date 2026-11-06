"""viz_prune.py redrawn with the current mainline (lattice_joint, current screening) (2026-11-02).

Same figure as viz_prune.py: on every arrow S --p--> S + {p} of the lattice the optimistic cost of the
cheapest complete order through it,
    through(S, p) = min_last F(S, last) + step(S, last, p) + H(S + {p}, p)
F = lattice_joint's forward DP, H = the same DP backwards (cheapest relaxed completion + return to HOME).
U and the optimal order come from lattice_joint/_output/triangular_frame6.json (python -m lattice_joint.run).

    python -X utf8 -m experiments.bls_all_orders_20261031.viz_prune_mainline
Output: _out/prune_triangular_frame6_mainline.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy (c10.dll)

import numpy as np  # noqa: E402

from .run import log  # noqa: E402
from .viz_lattice import SHORT  # noqa: E402
from .viz_prune import plot  # noqa: E402

RES = Path(__file__).resolve().parents[2] / "lattice_joint" / "_output" / "triangular_frame6.json"


def main():
    from lattice_joint.instance import Instance
    from lattice_joint.lattice import Lattice
    res = json.loads(RES.read_text(encoding="utf-8"))
    inst = Instance.build("triangular_frame6")
    lat = Lattice(inst).build()
    full = inst.ids
    H = {(full, q): lat.const(q) for q in lat.lasts[full]}            # backward DP: completion + return
    for S in sorted(lat.nxt, key=len, reverse=True):
        for last in lat.lasts.get(S, [None]):
            v = [lat.step(S, last, p) + H.get((tuple(sorted(S + (p,))), p), np.inf) for p in lat.nxt[S]]
            if v:
                H[S, last] = min(v)
    root = H[(), None]
    assert abs(root - lat.root) < 1e-9, (root, lat.root)
    through = {}
    for S, p in lat.arrows:
        S2 = tuple(sorted(S + (p,)))
        through[S, p] = min(lat.F[S, last] + lat.step(S, last, p) + H[S2, p] if S else
                            lat.step(S, None, p) + H[S2, p] for last in (lat.lasts.get(S) or [None]))

    def order_cost(o):
        c, last = 0.0, None
        for k, p in enumerate(o):
            c += lat.step(tuple(sorted(o[:k])), last, p)
            last = p
        return c + lat.const(last)
    order_val = {tuple(o): order_cost(o) for o in inst.orders}
    U, best = res["U"], tuple(res["order"])
    log(f"root bound {root:.4f}, optimum U {U:.4f} ({' '.join(SHORT[q] for q in best)}, optimistic {order_val[best]:.4f});"
        f" arrows below U: {sum(v < U - 1e-9 for v in through.values())} of {len(through)};"
        f" orders below U: {sum(v < U - 1e-9 for v in order_val.values())} of {len(order_val)}")
    for (S, p), v in sorted(through.items(), key=lambda t: t[1]):
        log(f"  {'+'.join(SHORT[x] for x in S) or '{}':16s} --{SHORT[p]:2s}->  {v:.4f}  {'KEEP' if v < U - 1e-9 else 'pruned'}")
    for o, v in sorted(order_val.items(), key=lambda t: t[1])[:12]:
        log(f"  order {' '.join(SHORT[q] for q in o):22s} {v:.4f}  {'KEEP' if v < U - 1e-9 else 'pruned'}")
    plot({(frozenset(S), p): v for (S, p), v in through.items()}, root, U, best, order_val,
         out="prune_triangular_frame6_mainline.png", what="当前主线 lattice_joint，chain 代价")


if __name__ == "__main__":
    sys.exit(main())
