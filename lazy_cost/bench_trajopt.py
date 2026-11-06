"""Light TrajOpt settings for the lazy refinement: time per segment and path quality.

    & $py -X utf8 -m lazy_cost.bench_trajopt [--per-model 40]
Output: lazy_cost/_output/bench_trajopt.json

Segments: the BLOCKED segments (straight line collides) of the parts-only contexts used by
exp_rack10 (--boxes 0,0 --crowd 0 --gap 0.03, seed 11): transit start->hover and transfer
lift->preins (holding the part). Every variant plans the same segments through
CuroboRefiner (first pass with the variant; failures retried with the full planner),
and the path is re-timed with topp as in the cost module.

baseline: plan_cspace (graph-seeded TrajOpt, 4 seeds, 100 + 100 iterations: initial pass
          and time-optimal fine-tune pass), the setting used before 2026-10-16. The chosen
          default is now "1 pass, 50 it, 2 seeds".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"
VARIANTS = {          # name: (TrajOpt seeds the planner is built with, light first pass or None)
    "baseline": (4, None),
    "1 pass, 100 it, 4 seeds": (4, dict(finetune=0, iters=100)),
    "1 pass, 50 it, 4 seeds": (4, dict(finetune=0, iters=50)),
    "1 pass, 25 it, 4 seeds": (4, dict(finetune=0, iters=25)),
    "1 pass, 50 it, 2 seeds": (2, dict(finetune=0, iters=50)),
    "1 pass, 25 it, 2 seeds": (2, dict(finetune=0, iters=25)),
    "1 pass, 50 it, 1 seed": (1, dict(finetune=0, iters=50)),
    "1 pass, 25 it, 1 seed": (1, dict(finetune=0, iters=25)),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-model", type=int, default=40)
    ap.add_argument("--contexts", type=int, default=30)
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    import torch

    from .checker import GpuChecker
    from .cost import LazyStepCost
    from .endpoints import load_or_build
    from .exp_rack10 import make_context
    from .refine import CuroboRefiner
    from .scene import load_scene

    report = {}
    for model in ("triangular_frame6", "rack10"):
        sc = load_scene(model)
        chk = GpuChecker(sc)
        cat, _ = load_or_build(sc, chk)
        refs = {}
        for seeds in sorted({v[0] for v in VARIANTS.values()}):
            refs[seeds] = CuroboRefiner(chk, batch=1, seeds=seeds, fast=None)
            for p in sc.ids:
                refs[seeds].fit(p)
        lb = LazyStepCost(sc, cat, chk, None, k=3)
        rng = np.random.default_rng(11)

        def start_free(w, q):
            chk.set_world(chk.world_dict(*w))
            return bool(chk.configs_free(q)[0])

        items = []
        for ci in range(args.contexts):
            c = make_context(sc, cat, rng, ci, 12, start_free, boxes_range=(0, 0), crowd=0, gap=0.03)
            if c is None:
                continue
            p = c["part"]
            ranked, _ = lb.rank(p, c["start"], c["installed"], c["placed"], c["boxes"], c["cands"], refine=False)
            W = chk.world_dict(c["installed"], c["placed"], c["boxes"])
            for r in ranked:
                for g, segs in r.segments.items():
                    e = cat["pick"][p][r.key][g]
                    if segs["transit"][1] == "bound":
                        items.append(("transit", W, p, r.key, g, c["start"], e["hover"]))
                    if segs["transfer"][1] == "bound":
                        items.append(("transfer", W, p, r.key, g, e["lift"], cat["goal"][p][g]["preins"]))
        sel = np.random.default_rng(0).permutation(len(items))[:args.per_model]
        items = [items[i] for i in sorted(sel)]
        print(f"{model}: {len(items)} blocked segments ({sum(i[0] == 'transit' for i in items)} transit)", flush=True)
        res = {}
        for name, (seeds, fast) in VARIANTS.items():
            ref = refs[seeds]
            ref.fast = fast
            dur, ms, retried = [], [], 0
            try:
                for kind, W, p, x, g, qa, qb in items:          # one warm-up call per variant
                    if kind == "transit":
                        ref.transit(W, p, [(qa, qb, sc.pose[p][x])], sc.held_spheres(p))
                    else:
                        ref.transfer(W, p, sc.grasps()[p][g][0], [(qa, qb)])
                    break
                for kind, W, p, x, g, qa, qb in items:
                    r0 = ref.stats.get("retried", 0)
                    torch.cuda.synchronize()
                    t0 = perf_counter()
                    if kind == "transit":
                        v = ref.transit(W, p, [(qa, qb, sc.pose[p][x])], sc.held_spheres(p))[0]
                    else:
                        v = ref.transfer(W, p, sc.grasps()[p][g][0], [(qa, qb)])[0]
                    torch.cuda.synchronize()
                    ms.append(1e3 * (perf_counter() - t0))
                    dur.append(v)
                    retried += ref.stats.get("retried", 0) - r0
            except Exception as ex:                                # e.g. a seed count the solver was not built for
                print(f"  {name:24s} FAILED: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
                res[name] = dict(error=str(ex)[:300])
                continue
            res[name] = dict(ms=ms, dur=dur, retried=retried)
            print(f"  {name:24s} median {np.median(ms):6.0f} ms/segment (p90 {np.percentile(ms, 90):5.0f}), "
                  f"first pass failed {retried}/{len(items)}, no path {sum(not np.isfinite(d) for d in dur)}", flush=True)
        base = np.array(res["baseline"]["dur"])
        for name, r in res.items():
            if "dur" not in r:
                continue
            d = np.array(r["dur"])
            ok = np.isfinite(d) & np.isfinite(base)
            ratio = d[ok] / base[ok]
            r["ratio_vs_baseline"] = dict(p10=float(np.percentile(ratio, 10)), p50=float(np.median(ratio)),
                                          p90=float(np.percentile(ratio, 90)), max=float(ratio.max()))
            print(f"  {name:24s} path time / baseline: p10 {r['ratio_vs_baseline']['p10']:.3f} p50 {r['ratio_vs_baseline']['p50']:.3f} "
                  f"p90 {r['ratio_vs_baseline']['p90']:.3f} max {r['ratio_vs_baseline']['max']:.2f}", flush=True)
        report[model] = {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}
        del refs, ref
        torch.cuda.empty_cache()
    OUT.mkdir(exist_ok=True)
    (OUT / "bench_trajopt.json").write_text(json.dumps(report, default=float), encoding="utf-8")
    print("->", OUT / "bench_trajopt.json")


if __name__ == "__main__":
    main()
