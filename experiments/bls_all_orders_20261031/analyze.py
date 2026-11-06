"""Summary of the all-orders exact search (run.py search output + precompute).

    python -X utf8 -m experiments.bls_all_orders_20261031.analyze --model triangular_frame6
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict

import numpy as np

from .run import OUT, search_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    d = pickle.load(open(OUT / f"pre_{a.model}_k{a.k}.pkl", "rb"))
    res = json.loads((OUT / f"search_{a.model}_k{a.k}.json").read_text(encoding="utf-8"))["results"]
    ids = d["ids"]
    beams = [p for p in ids if p.startswith("beam")]
    out = {}
    for mode in ("bls", "chain"):
        print(f"\n===== {mode} =====")
        rows = [(r["order"], r[mode]) for r in res]
        feas = [(o, r) for o, r in rows if r["cost"] is not None]
        c = np.array([r["cost"] for _, r in feas])
        print(f"feasible orders {len(feas)}/{len(rows)}; cost min {c.min():.4f} median {np.median(c):.4f} max {c.max():.4f} "
              f"(max/min {c.max() / c.min():.3f}); distinct values {len(set(np.round(c, 6)))}")
        best = [o for o, r in feas if r["cost"] <= c.min() + 1e-9]
        print(f"optimal orders ({len(best)}): " + " | ".join(" ".join(o) for o in best[:6]) + (" ..." if len(best) > 6 else ""))
        # cost by the order of the beams only
        by_beams = defaultdict(list)
        for o, r in feas:
            by_beams[tuple(p for p in o if p in beams)].append(r["cost"])
        print("by beam order: " + "; ".join(f"{'>'.join(b[5:] for b in k)}: {min(v):.4f}-{max(v):.4f} (n={len(v)})"
                                             for k, v in sorted(by_beams.items(), key=lambda t: min(t[1]))))
        gap = np.array([(r["cost"] - r["root_lb"]) / r["cost"] for _, r in feas])
        nodes = np.array([r["nodes"] for _, r in feas])
        print(f"gap (opt - sum of per-part minima) / opt: median {np.median(gap) * 100:.2f}% max {gap.max() * 100:.2f}%; "
              f"nodes median {np.median(nodes):.0f} max {nodes.max()}; empty-domain prunes total "
              f"{sum(r['empty_prunes'] for _, r in feas)}")
        # same pose across orders?
        print("optimal pose per part across orders (distinct positions x,y / distinct poses):")
        for p in ids:
            poses = [r["layout"][p] for _, r in feas]
            xy = [",".join(k.split(",")[:2]) for k in poses]
            top = Counter(xy).most_common(2)
            print(f"  {p:8s} {len(set(xy)):2d} positions / {len(set(poses)):2d} poses; most common {top}")
        # conflicts: parts not at the cheapest pose of their own root domain
        moved, why = Counter(), Counter()
        for _, r in feas:
            for p, v in r["parts"].items():
                if v["n_cheaper"]:
                    moved[p] += 1
                    for b in v["blocked_by"]:
                        why[b.split(":")[1]] += 1
        n_moved = [sum(1 for v in r["parts"].values() if v["n_cheaper"]) for _, r in feas]
        print(f"parts pushed off their own best pose by later parts: per order mean {np.mean(n_moved):.2f} "
              f"max {max(n_moved)}; by part {dict(moved)}; reason counts {dict(why)}")
        out[mode] = dict(min=float(c.min()), max=float(c.max()), best=best, gap_median=float(np.median(gap)))
    # order effect on feasibility: root domain size per part vs installed set
    print("\n===== domains (installed set effect) =====")
    for p in ids:
        sizes = sorted({r["bls"]["root_dom"][p] for r in res})
        print(f"  {p:8s} usable poses with installed sets: {sizes}  (of {len(d['poses'][p])})")
    # how much of the pairwise table did the searches touch?
    touched = set()
    import experiments.bls_all_orders_20261031.run as run
    orig = run.search_order

    for r in res:
        o = r["order"]
        for mode in ("bls", "chain"):
            best, st, root, cost, const = orig(d, o, mode, 2_000_000, touched)
    n_obs = len(d["obs"])
    print(f"\npairwise table: {n_obs} (part, pose) obstacles precomputed; the 96 searches placed "
          f"{len(touched)} distinct ones ({len(touched) / n_obs * 100:.1f}%)")
    (OUT / f"summary_{a.model}_k{a.k}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
