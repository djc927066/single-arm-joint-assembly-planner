"""Low-cost check: how much do grasp choice and IK branch change the joint-time lower bound?

    & $py -X utf8 -m lazy_cost.check_grasp_ik [--pool 120]
Output: lazy_cost/_output/check_grasp_ik.json (+ printed tables)

No cuRobo planning and no clutter: only the layout-independent part (table + assembly base),
the 8 posts x 40 candidate poses. Step bound LB = t(start -> prepick) + t(lift -> preins).

Configuration sets compared, per (post, candidate):
  current      the catalogue in _cache (3 goal-side-diverse grasps, first IK solution of each
               keypose solved independently)
  3 consistent the same 3 grasps; prepick and lift take the IK solution nearest to each pick
               solution (same branch), best branch kept
  two-side k   k grasps chosen farthest-apart among grasps feasible at the goal AND on >=25%
               of the candidates, branch-consistent (k = 3, 8)
  reference    ``pool`` grasps chosen farthest-apart among all goal-feasible grasps (stands in
               for "all grasps"), every consistent branch, best kept
Starts: HOME, and the retreat of another post (the step before, as in the reverse search).
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"
BRANCH_TOL = 0.5          # rad: prepick/lift within this of pick on every joint = same IK branch


def t_acc_many(A, B):
    """Rest-to-rest joint-time bound between configuration arrays (broadcast) -> [...]."""
    from curobo_bridge.assembly_demo import AMAX, VMAX
    d = np.abs(np.asarray(B) - np.asarray(A))
    t = np.where(d < VMAX ** 2 / AMAX, 2 * np.sqrt(d / AMAX), d / VMAX + VMAX / AMAX)
    return t.max(-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=int, default=120)
    ap.add_argument("--min-pick-share", type=float, default=0.25)
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    from curobo_bridge.assembly_demo import BatchIK, keyposes
    from scipy.stats import spearmanr

    from .checker import GpuChecker
    from .endpoints import diverse, load_or_build
    from .scene import AssemblyScene

    t_start = perf_counter()
    sc = AssemblyScene(importlib.import_module("scenes.benchmarks.rack10").spec())
    chk = GpuChecker(sc)
    cat, _ = load_or_build(sc, chk)
    chk.set_world(chk.world_dict())
    bik = BatchIK(batch=512, seeds=32, returns=4)
    posts = [p for p in sc.ids if p.startswith(("lp_", "up_"))]
    grasps = sc.grasps()
    report, rows = {}, {}

    # ---- current catalogue: branch consistency of its (prepick, pick, lift) triples ---------------
    jump = []
    for p in posts:
        for x, gs in cat["pick"][p].items():
            for g, e in gs.items():
                jump.append(max(np.abs(e["prepick"] - e["pick"]).max(), np.abs(e["lift"] - e["pick"]).max()))
    jump = np.array(jump)
    report["current_branch"] = dict(triples=len(jump), inconsistent_share=float(np.mean(jump > BRANCH_TOL)),
                                    max_jump_p50=float(np.median(jump)), max_jump_p90=float(np.percentile(jump, 90)),
                                    max_jump_max=float(jump.max()))
    print("current catalogue triples:", report["current_branch"], flush=True)

    # ---- per post: goal side over all grasps, pick side over a covering pool ----------------------
    goal_cfg = {}
    for p in posts:
        t0 = perf_counter()
        part = SimpleNamespace(goal=sc.goal[p])
        sph = sc.held_spheres(p)
        usable = [g for g, (G, _) in enumerate(grasps[p]) if keyposes(part, sc.goal[p], G)["ins"][2, 2] <= 0.3]
        P = np.stack([keyposes(part, np.eye(4), grasps[p][g][0])[k] for g in usable for k in ("preins", "retreat")])
        q, ok = bik.solve(P)
        q, ok = q.reshape(len(usable), 2, -1, 6), ok.reshape(len(usable), 2, -1)
        goal = {}                                   # g -> list of (preins, retreat) consistent pairs
        for i, g in enumerate(usable):
            opts = []
            for b in np.flatnonzero(ok[i, 0]):
                pre = q[i, 0, b]
                rs = [r for r in np.flatnonzero(ok[i, 1]) if np.abs(q[i, 1, r] - pre).max() < BRANCH_TOL]
                if rs:
                    opts.append((pre, q[i, 1, rs[0]]))
            if not opts:
                continue
            pre = np.stack([o[0] for o in opts])
            ret = np.stack([o[1] for o in opts])
            free = chk.configs_free(pre, (sph, grasps[p][g][0])) & chk.configs_free(ret)
            opts = [o for o, f in zip(opts, free) if f]
            if opts:
                goal[g] = opts
        goal_cfg[p] = goal
        feas = list(goal)
        pool = set(diverse([grasps[p][g][0] for g in feas], feas, args.pool)) | set(cat["goal"][p])
        pool = sorted(pool & set(feas))
        cands = sc.candidates[p]
        P = np.stack([keyposes(part, T, grasps[p][g][0])[k] for g in pool for _, T in cands for k in ("prepick", "pick", "lift")])
        q, ok = bik.solve(P)
        q, ok = q.reshape(len(pool), len(cands), 3, -1, 6), ok.reshape(len(pool), len(cands), 3, -1)
        pick = {}                                   # (g, ci) -> list of (prepick, lift) consistent options
        for gi, g in enumerate(pool):
            G = grasps[p][g][0]
            items = []
            for ci, (key, T) in enumerate(cands):
                kp = keyposes(part, T, G)
                if kp["pick"][2, 2] > 0.3 or kp["pick"][2, 3] < 0.012:
                    continue
                for b in np.flatnonzero(ok[gi, ci, 1]):
                    pk = q[gi, ci, 1, b]
                    pre = [r for r in np.flatnonzero(ok[gi, ci, 0]) if np.abs(q[gi, ci, 0, r] - pk).max() < BRANCH_TOL]
                    lif = [r for r in np.flatnonzero(ok[gi, ci, 2]) if np.abs(q[gi, ci, 2, r] - pk).max() < BRANCH_TOL]
                    if pre and lif:
                        items.append((ci, q[gi, ci, 0, pre[0]], q[gi, ci, 2, lif[0]]))
            if not items:
                continue
            pre = np.stack([it[1] for it in items])
            lif = np.stack([it[2] for it in items])
            free = chk.configs_free(pre) & chk.configs_free(lif, (sph, G))
            for (ci, a, b), f in zip(items, free):
                if f:
                    pick.setdefault((g, ci), []).append((a, b))
        rows[p] = dict(goal=goal, pick=pick, pool=pool, n_cands=len(cands))
        n_pick = {g: len({ci for (gg, ci) in pick if gg == g}) for g in pool}
        print(f"{p}: grasps {len(grasps[p])}, goal-usable {len(usable)}, goal-feasible {len(feas)}, pool {len(pool)}, "
              f"pickable cands (any pool grasp) {len({ci for _, ci in pick})}/{len(cands)}, "
              f"median cands per grasp {np.median(list(n_pick.values())):.0f}  [{perf_counter() - t0:.0f}s]", flush=True)

    # ---- bounds per configuration-set choice ---------------------------------------------------------
    def lb_set(p, start, gset, consistent=True):
        """per candidate: min over grasps in gset (and branches) of LB; inf if none."""
        r = rows[p]
        out = np.full(r["n_cands"], np.inf)
        for g in gset:
            if g not in r["goal"]:
                continue
            pre_ins = np.stack([o[0] for o in r["goal"][g]])
            for ci in range(r["n_cands"]):
                opts = r["pick"].get((g, ci))
                if not opts:
                    continue
                A = np.stack([o[0] for o in opts])
                L = np.stack([o[1] for o in opts])
                v = t_acc_many(start[None], A) + t_acc_many(L[:, None], pre_ins[None]).min(1)
                out[ci] = min(out[ci], v.min())
        return out

    def lb_current(p, start):
        out = np.full(rows[p]["n_cands"], np.inf)
        keys = [k for k, _ in sc.candidates[p]]
        for x, gs in cat["pick"][p].items():
            ci = keys.index(x)
            for g, e in gs.items():
                out[ci] = min(out[ci], t_acc_many(start, e["prepick"]) + t_acc_many(e["lift"], cat["goal"][p][g]["preins"]))
        return out

    def two_side(p, k):
        r = rows[p]
        cnt = {g: len({ci for (gg, ci) in r["pick"] if gg == g}) for g in r["pool"]}
        ok_g = [g for g in r["pool"] if cnt[g] >= args.min_pick_share * r["n_cands"]]
        return diverse([grasps[p][g][0] for g in ok_g], ok_g, k)

    methods = {"current": None, "3 consistent": lambda p: list(cat["goal"][p]),
               "two-side 3": lambda p: two_side(p, 3), "two-side 8": lambda p: two_side(p, 8)}
    results = {}
    for start_name in ("HOME", "prev retreat"):
        per = {m: dict(pickable=[], ratio=[], spearman=[], top3=[], regret=[], wrong_prune=[]) for m in methods}
        spread = []
        for i, p in enumerate(posts):
            if start_name == "HOME":
                start = np.zeros(6)
            else:
                q_prev = posts[i - 1]
                start = cat["goal"][q_prev][min(cat["goal"][q_prev])]["retreat"]
            ref = lb_set(p, start, rows[p]["pool"])
            have = np.isfinite(ref)
            # spread over grasps for the same candidate (reference pool, best branch per grasp)
            per_g = np.stack([lb_set(p, start, [g]) for g in rows[p]["pool"]])
            for ci in np.flatnonzero(have):
                v = per_g[:, ci][np.isfinite(per_g[:, ci])]
                if len(v) > 1:
                    spread.append(float(v.max() / v.min()))
            for m, f in methods.items():
                v = lb_current(p, start) if f is None else lb_set(p, start, f(p))
                per[m]["pickable"].append(int(np.isfinite(v).sum()))
                per[m]["wrong_prune"].append(int((have & ~np.isfinite(v)).sum()))
                both = have & np.isfinite(v)
                per[m]["ratio"] += list(v[both] / ref[both])
                vv = np.where(np.isfinite(v), v, 1e3)[have]
                rr = ref[have]
                per[m]["spearman"].append(float(spearmanr(vv, rr)[0]))
                top_m, top_r = set(np.argsort(vv)[:3]), set(np.argsort(rr)[:3])
                per[m]["top3"].append(len(top_m & top_r) / 3)
                per[m]["regret"].append(float(rr[np.argmin(vv)] / rr.min() - 1))
        print(f"\n[start = {start_name}]  8 posts x 40 candidates; reference pickable "
              f"{sum(per['current']['pickable']) + sum(per['current']['wrong_prune'])} (approx.)", flush=True)
        print(f"{'set':14s} {'pickable':>9s} {'wrongly lost':>13s} {'LB/ref p50':>11s} {'p90':>6s} {'max':>6s} "
              f"{'spearman':>9s} {'top3':>6s} {'regret':>8s}")
        res = {}
        for m in methods:
            d = per[m]
            ratio = np.array(d["ratio"])
            res[m] = dict(pickable=int(sum(d["pickable"])), wrongly_lost=int(sum(d["wrong_prune"])),
                          ratio_p50=float(np.median(ratio)), ratio_p90=float(np.percentile(ratio, 90)), ratio_max=float(ratio.max()),
                          spearman=float(np.mean(d["spearman"])), top3=float(np.mean(d["top3"])), regret=float(np.mean(d["regret"])),
                          regret_max=float(np.max(d["regret"])))
            v = res[m]
            print(f"{m:14s} {v['pickable']:9d} {v['wrongly_lost']:13d} {v['ratio_p50']:11.3f} {v['ratio_p90']:6.3f} "
                  f"{v['ratio_max']:6.3f} {v['spearman']:9.3f} {v['top3']:6.2f} {v['regret']:8.3f}", flush=True)
        res["grasp_spread_max_over_min"] = dict(p50=float(np.median(spread)), p90=float(np.percentile(spread, 90)),
                                                max=float(np.max(spread)))
        res["reference_pickable"] = int(sum(int(np.isfinite(lb_set(p, np.zeros(6), rows[p]["pool"])).sum()) for p in posts))
        print("same candidate, different grasps (best/worst LB):", res["grasp_spread_max_over_min"],
              "| reference pickable:", res["reference_pickable"], flush=True)
        results[start_name] = res
    report["results"] = results
    report["params"] = dict(pool=args.pool, branch_tol=BRANCH_TOL, min_pick_share=args.min_pick_share,
                            seconds=perf_counter() - t_start)
    OUT.mkdir(exist_ok=True)
    (OUT / "check_grasp_ik.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("->", OUT / "check_grasp_ik.json")


if __name__ == "__main__":
    main()
