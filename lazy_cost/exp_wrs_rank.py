"""How well does the fast cost rank the candidate poses of one step? Reference = WRS RRT.

    & $py -X utf8 -m lazy_cost.exp_wrs_rank [--contexts 20] [--cap 10] [--grasps 3] [--start 0]

Context (same generator as exp_rack10, parts are the only obstacles): a random stable order and
a step k; parts before k installed, parts after k at random table poses; the part of step k has
up to ``cap`` candidate poses; the arm starts at the previous part's retreat (or HOME).
Step cost of (candidate x, grasp g):
    transit  start -> hover(x,g) (+ vertical descent)     empty hand, part x on the table
    fixed    prepick -> pick -> lift, preins -> ins -> retreat   (same value in every method)
    transfer lift(x,g) -> preins(g)                         holding the part
Candidate cost = min over a grasp set S_x (the 3 grasps with the lowest bound; the same set for
every method and for the reference).
Fast estimates (one batched GPU pass, LazyStepCost bounds only, no planner):
    lb      joint-time lower bound of each straight segment
    old     blocked segment: lb * (1 + 0.243 + 1.093 depth)      (previous correction)
    new     blocked segment: lb + 4.7 s/m * depth                 (current correction)
Reference (WRS, CPU): each transit / transfer segment checked with WRS mesh collision on the
straight joint line; free -> its lb (exact); blocked -> WRS RRT-Connect (ext 0.05 rad, 6 seeds),
mesh re-check, shortcut, corner blend, TOPP time.
Output: lazy_cost/_output/wrs_rank_<model><tag>.jsonl (one row per context) + summary json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .exp_rack10 import OUT, make_context, metrics

C_NEW = 4.7
OLD_W, OLD_B = 1.0931, 0.2431


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--contexts", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="first context index (to split runs)")
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--grasps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    import torch
    from joint_search.wrs_check import MeshWorld
    from .checker import GpuChecker
    from .cost import LazyStepCost, t_acc
    from .detour import blend
    from .endpoints import load_or_build
    from .scene import load_scene
    from .timing import topp

    sync = torch.cuda.synchronize
    rng = np.random.default_rng(args.seed)
    sc = load_scene(args.model)
    chk = GpuChecker(sc)
    cat, _ = load_or_build(sc, chk)
    lb_only = LazyStepCost(sc, cat, chk, None, k=3)
    W = MeshWorld(SimpleNamespace(scene=sc))
    G = sc.grasps()
    OUT.mkdir(exist_ok=True)
    out = OUT / f"wrs_rank_{args.model}{args.tag}.jsonl"

    def wrs_time(a, b):
        """-> (time without descent, how, seconds spent)."""
        t0 = perf_counter()
        lb = float(t_acc(a, b))
        if W.free([a, b])[0]:
            return lb, "straight", perf_counter() - t0
        for seed in range(6):
            P, _ = W.planner._rrt(a, b, seed=seed)
            if P is None or not W.free(P)[0]:
                continue
            S = W.shortcut(P)
            t = float(topp(S)[1][-1])
            if len(S) > 2 and W.free(blend(S))[0]:
                t = min(t, float(topp(blend(S))[1][-1]))
            return t, "rrt", perf_counter() - t0
        return np.inf, "rrt_failed", perf_counter() - t0

    for ci in range(args.contexts):
        def start_free(w, q):
            chk.set_world(chk.world_dict(*w))
            return bool(chk.configs_free(q)[0])

        c = make_context(sc, cat, rng, ci, args.cap, start_free, boxes_range=(0, 0), crowd=0.0, gap=0.03)
        if ci < args.start:
            continue
        if c is None:
            continue
        p, start, cands = c["part"], c["start"], c["cands"]
        # ---- fast cost: one batched GPU pass (timed on the second call: no CUDA warm-up) ---------
        lb_only.rank(p, start, c["installed"], c["placed"], c["boxes"], cands, refine=False)
        sync(); t0 = perf_counter()
        ranked, _ = lb_only.rank(p, start, c["installed"], c["placed"], c["boxes"], cands, refine=False)
        sync(); fast_s = perf_counter() - t0
        goal = cat["goal"][p]
        sel, est = {}, {}
        for r in ranked:
            if r.status.startswith("pruned"):
                continue
            e_x = []
            for g, sg in r.segments.items():
                e = cat["pick"][p][r.key][g]
                fixed = float(t_acc(e["prepick"], e["pick"]) + t_acc(e["pick"], e["lift"])
                              + t_acc(goal[g]["preins"], goal[g]["ins"]) + t_acc(goal[g]["ins"], goal[g]["retreat"]))
                v = {"lb": fixed, "old": fixed, "new": fixed}
                for kind in ("transit", "transfer"):
                    val, how = sg[kind]
                    d = float(r.features[g][kind]["depth"]) if how == "bound" else 0.0
                    v["lb"] += val
                    v["old"] += val if how != "bound" else val * (1 + OLD_B + OLD_W * d)
                    v["new"] += val if how != "bound" else val + C_NEW * d
                e_x.append((v["lb"], int(g), v, fixed, {k: sg[k][1] for k in sg}))
            e_x.sort(key=lambda t: t[0])
            sel[r.key] = e_x[:args.grasps]
        # ---- WRS reference ----------------------------------------------------------------------
        t0 = perf_counter()
        table = {q: T for _, q, T in c["placed"]}
        seg_rows, truth = [], {}
        n_rrt = 0
        for x, items in sel.items():
            vals = []
            for _, g, v, fixed, how in items:
                e = cat["pick"][p][x][g]
                W.set(c["installed"], dict(table, **{p: sc.pose[p][x]}))
                ta, ha, sa = wrs_time(start, e["hover"])
                W.set(c["installed"], table, (p, G[p][g][0], e["lift"]))
                tb, hb, sb = wrs_time(e["lift"], goal[g]["preins"])
                n_rrt += (ha != "straight") + (hb != "straight")
                desc = float(t_acc(e["hover"], e["prepick"]))
                tot = ta + desc + tb + fixed
                vals.append(tot)
                for kind, tt, hw, ss in (("transit", ta + desc, ha, sa), ("transfer", tb, hb, sb)):
                    seg_rows.append(dict(cand=x, g=g, kind=kind, ours=how[kind], wrs=hw, wrs_time=tt, wrs_s=ss))
                for k in ("lb", "old", "new"):
                    v[k] = float(v[k])
            truth[x] = min(vals)
        W.set([], {})
        ref_s = perf_counter() - t0
        pred = {k: {x: min(v[k] for _, _, v, _, _ in items) for x, items in sel.items()} for k in ("lb", "old", "new")}
        m = {k: metrics(pred[k], truth, 3) for k in pred}
        row = dict(ctx=ci, part=p, k=c["k"], installed=len(c["installed"]), placed=len(c["placed"]),
                   candidates=len(cands), alive=len(sel), fast_s=fast_s, wrs_s=ref_s, wrs_rrt_segments=n_rrt,
                   metrics=m if all(m.values()) else None, pred=pred,
                   truth={x: (v if np.isfinite(v) else None) for x, v in truth.items()}, segments=seg_rows)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        mm = m["new"]
        print(f"ctx {ci}: {p} k={c['k']} cands {len(cands)} alive {len(sel)} | fast {1e3 * fast_s:.0f} ms | WRS {ref_s:.1f}s "
              f"({n_rrt} RRT segs) | " + (" ".join(f"{k}: top1 {m[k]['top1']:.0f} regret {m[k]['regret']:.3f} rho {m[k]['spearman']:.2f}"
                                                  for k in m) if mm else "too few for metrics"), flush=True)


if __name__ == "__main__":
    main()
