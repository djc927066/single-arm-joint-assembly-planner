"""One part, one step: rank its candidate poses with the search's fast cost, compare with WRS RRT.

    & $py -X utf8 -m joint_search.exp_step_rank [--contexts 12] [--grasps 3] [--seed 5]

Context: a random stable order and a step k; parts before k installed at their goals, parts after
k at random table poses (footprints >= gap apart); the arm starts at the previous part's retreat
(a random usable grasp of it) or at HOME. Part p = order[k], all its candidate poses that do not
overlap the parts on the table.

Fast cost (GPU, exactly the search's rules, see expand.py):
  screening  insertion moment (per grasp), then pick moment (per pose, grasp), arm + gripper vs the
             installed parts and the parts on the table; gripper closed to the grasp's jaw
             (screen.CLOSE_FINGERS)
  cost       transit   start -> above pick        empty hand (open), p at x as an obstacle
             fixed     above pick -> pick -> above pick, above insert -> ins -> retreat
             transfer  above pick -> above insert holding p (gripper closed)
             straight joint line free -> lower bound; blocked -> lb (method 'lb') or
             lb + 4.7 s/m * max penetration depth (method 'depth')
Candidate cost = min over its grasps; for the reference the 3 grasps with the lowest estimate
(per candidate) are planned, and both methods take the min over the same grasps.
Reference (WRS, CPU): transit / transfer each: WRS mesh check of the straight joint line (same
jaw as above); free -> lb, blocked -> WRS RRT-Connect + mesh re-check + shortcut + blend, TOPP time.
Output: joint_search/_output/step_rank<tag>.jsonl + printed summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"
C_DEPTH = 4.7


def fast_rank(pb, p, start, installed, table, xs):
    """-> dict (x, g) -> dict(lb, depth_est, segs), seconds. Same screening / cost as expand.py."""
    import torch
    from .problem import t_acc
    sc, cat, chk = pb.scene, pb.cat, pb.chk
    jaws = (lambda gs: np.array([pb.jaw[p][g] for g in gs])) if pb.close_fingers else (lambda gs: None)
    torch.cuda.synchronize()
    t0 = perf_counter()
    placed = [(f"{q}@{k}", q, sc.pose[q][k]) for q, k in table.items()]
    goal = cat["goal"][p]
    rows_all = [(x, g) for x in xs for g in cat["pick"][p][x]]
    gl_all = sorted({g for _, g in rows_all})
    chk.set_world(chk.world_dict(installed, placed))
    ok_ins, = chk.batch_free([(np.stack([goal[g]["ins"] for g in gl_all]), None, -1, -1, jaws(gl_all))])
    gok = {g for g, k in zip(gl_all, ok_ins) if k}
    rows0 = [(x, g) for x, g in rows_all if g in gok]
    if not rows0:
        return {}, perf_counter() - t0, dict(grasps=len(gl_all), ins_ok=0, pairs=0)
    ok_pick, = chk.batch_free([(np.stack([cat["pick"][p][x][g]["pick"] for x, g in rows0]), None, -1, -1,
                                jaws([g for _, g in rows0]))])
    rows = [(x, g) for (x, g), k in zip(rows0, ok_pick) if k]
    if not rows:
        return {}, perf_counter() - t0, dict(grasps=len(gl_all), ins_ok=len(gok), pairs=0)
    gl = sorted({g for _, g in rows})
    gi = {g: i for i, g in enumerate(gl)}
    xi = {x: i for i, x in enumerate(xs)}
    E = lambda k: np.stack([cat["pick"][p][x][g][k] for x, g in rows])
    above, pick = E("lift"), E("pick")
    pre = np.stack([goal[g]["preins"] for _, g in rows])
    ex = [chk.obstacle_spheres(pb.held[p], sc.pose[p][x]) for x in xs]
    (tr_free, tf_free), (f_tr, f_tf) = chk.batch_free(
        [(np.broadcast_to(start, (len(rows), 6)), above, -1, np.array([xi[x] for x, _ in rows])),
         (above, pre, np.array([gi[g] for _, g in rows]), -1, jaws([g for _, g in rows]))],
        held_bank=[(pb.held[p], sc.grasps()[p][g][0]) for g in gl], extra_bank=ex, step=0.02, features=True)
    out = {}
    for i, (x, g) in enumerate(rows):
        e, v = cat["pick"][p][x][g], goal[g]
        fixed = float(t_acc(e["lift"], e["pick"]) + t_acc(e["pick"], e["lift"]) + t_acc(v["preins"], v["ins"]) + t_acc(v["ins"], v["retreat"]))
        lb_tr, lb_tf = float(t_acc(start, e["lift"])), float(t_acc(e["lift"], v["preins"]))
        d_tr = 0.0 if tr_free[i] else float(f_tr["depth"][i])
        d_tf = 0.0 if tf_free[i] else float(f_tf["depth"][i])
        out[x, g] = dict(fixed=fixed, lb_tr=lb_tr, lb_tf=lb_tf, free_tr=bool(tr_free[i]), free_tf=bool(tf_free[i]),
                         d_tr=d_tr, d_tf=d_tf, lb=fixed + lb_tr + lb_tf,
                         depth=fixed + lb_tr + lb_tf + C_DEPTH * (d_tr + d_tf))
    torch.cuda.synchronize()
    return out, perf_counter() - t0, dict(grasps=len(gl_all), ins_ok=len(gok), pairs=len(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--contexts", type=int, default=12)
    ap.add_argument("--grasps", type=int, default=2)
    ap.add_argument("--cap", type=int, default=10, help="candidates per context (best-estimated first)")
    ap.add_argument("--rrt-seeds", type=int, default=2)
    ap.add_argument("--rrt-time", type=float, default=8.0)
    ap.add_argument("--gap", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    from scipy.stats import spearmanr
    from lazy_cost.detour import blend
    from lazy_cost.timing import topp
    from .problem import HOME, Problem, t_acc
    from .wrs_check import MeshWorld

    pb = Problem(args.model, penalty=str(Path(__file__).resolve().parents[1] / "lazy_cost" / "penalty_depth.json"))
    sc, cat = pb.scene, pb.cat
    W = MeshWorld(pb)
    W.planner.rrt_kw["max_time"] = args.rrt_time
    G = sc.grasps()
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"step_rank{args.tag}.jsonl"

    def wrs_time(a, b):
        t0 = perf_counter()
        lb = float(t_acc(a, b))
        if W.free([a, b])[0]:
            return lb, "straight", perf_counter() - t0
        if not W.free([a, a])[0] or not W.free([b, b])[0]:
            return np.inf, "endpoint_in_collision", perf_counter() - t0
        for seed in range(args.rrt_seeds):
            P, _ = W.planner._rrt(a, b, seed=seed)
            if P is None or not W.free(P)[0]:
                continue
            S = W.shortcut(P)
            t = float(topp(S)[1][-1])
            if len(S) > 2 and W.free(blend(S))[0]:
                t = min(t, float(topp(blend(S))[1][-1]))
            return t, "rrt", perf_counter() - t0
        return np.inf, "rrt_failed", perf_counter() - t0

    done = 0
    for attempt in range(200):
        if done == args.contexts:
            break
        order = list(sc.orders[int(rng.integers(len(sc.orders)))])
        k = int(rng.integers(len(order)))
        p, installed, later = order[k], order[:k], order[k + 1:]
        table, polys = {}, []
        for q in rng.permutation(later):
            keys = [c for c, _ in sc.candidates[q] if c in cat["pick"][q] and all(sc.polygons[q][c].distance(o) >= args.gap for o in polys)]
            if not keys:
                break
            c = keys[int(rng.integers(len(keys)))]
            table[str(q)] = c
            polys.append(sc.polygons[q][c])
        if len(table) != len(later):
            continue
        xs = [c for c, _ in sc.candidates[p] if c in cat["pick"][p] and all(sc.polygons[p][c].distance(o) >= args.gap for o in polys)]
        prev = order[k - 1] if k else None
        start = HOME if prev is None else cat["goal"][prev][sorted(cat["goal"][prev])[int(rng.integers(len(cat["goal"][prev])))]]["retreat"]
        fast_rank(pb, p, start, installed, table, xs)                         # CUDA warm-up for this world
        est, fast_s, scr = fast_rank(pb, p, start, installed, table, xs)
        cands = sorted({x for x, _ in est}, key=lambda x: min(est[xx, g]["depth"] for xx, g in est if xx == x))[:args.cap]
        if len(cands) < 3:
            print(f"ctx {attempt}: {p} k={k} only {len(cands)} feasible candidates ({scr}), skipped", flush=True)
            continue
        # ---- WRS reference on the 3 best-estimated grasps per candidate --------------------------
        t0 = perf_counter()
        tbl = {q: sc.pose[q][c] for q, c in table.items()}
        truth, sel, segs, n_rrt = {}, {}, [], 0
        for x in cands:
            gs = sorted((g for xx, g in est if xx == x), key=lambda g: est[x, g]["depth"])[:args.grasps]
            sel[x] = gs
            vals = []
            for g in gs:
                e, v, r = cat["pick"][p][x][g], cat["goal"][p][g], est[x, g]
                W.set(installed, dict(tbl, **{p: sc.pose[p][x]}))                                      # empty hand, open
                ta, ha, sa = wrs_time(start, e["lift"])
                W.set(installed, tbl, (p, G[p][g][0], e["lift"]), jaw=pb.jaw[p][g] if pb.close_fingers else None)
                tb, hb, sb = wrs_time(e["lift"], v["preins"])
                n_rrt += (ha != "straight") + (hb != "straight")
                vals.append(r["fixed"] + ta + tb)
                segs += [dict(cand=x, g=g, kind="transit", ours="straight" if r["free_tr"] else "blocked", depth=r["d_tr"],
                              lb=r["lb_tr"], wrs=ha, wrs_time=ta, wrs_s=sa),
                         dict(cand=x, g=g, kind="transfer", ours="straight" if r["free_tf"] else "blocked", depth=r["d_tf"],
                              lb=r["lb_tf"], wrs=hb, wrs_time=tb, wrs_s=sb)]
            truth[x] = min(vals)
            print(f"    {x:22s} est {min(est[x, g]['depth'] for g in gs):.3f}  WRS {truth[x]:.3f}  "
                  f"[{perf_counter() - t0:.0f}s]", flush=True)
        W.set([], {})
        wrs_s = perf_counter() - t0
        pred = {m: {x: min(est[x, g][m] for g in sel[x]) for x in cands} for m in ("lb", "depth")}
        fin = [x for x in cands if np.isfinite(truth[x])]
        if len(fin) < 3:
            print(f"ctx {attempt}: {p} k={k} only {len(fin)} candidates with a WRS time, skipped", flush=True)
            continue
        best = min(truth[x] for x in fin)
        met = {}
        for m in pred:
            pv = np.array([pred[m][x] for x in fin]); tv = np.array([truth[x] for x in fin])
            top = fin[int(np.argmin(pv))]
            met[m] = dict(spearman=float(spearmanr(pv, tv)[0]), top1=float(truth[top] <= best + 1e-9),
                          top1_2pct=float(truth[top] <= 1.02 * best), regret=float(truth[top] / best - 1),
                          top3=float(len(set(np.argsort(pv)[:3]) & set(np.argsort(tv)[:3])) / 3),
                          err=float(np.mean(np.abs(pv / tv - 1))))
        row = dict(ctx=attempt, part=p, k=k, order=order, installed=installed, table=table, candidates=len(xs),
                   feasible=len(cands), screening=scr, fast_s=fast_s, wrs_s=wrs_s, wrs_rrt_segments=n_rrt,
                   metrics=met, pred=pred, truth={x: (v if np.isfinite(v) else None) for x, v in truth.items()}, segments=segs)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        done += 1
        print(f"ctx {attempt}: {p} k={k} | grasps {scr['grasps']} -> insertion ok {scr['ins_ok']} -> pairs {scr['pairs']}, "
              f"candidates {len(cands)}/{len(xs)} | fast {1e3 * fast_s:.0f} ms | WRS {wrs_s:.0f} s ({n_rrt} RRT segs) | "
              + " ".join(f"{m}: top1 {v['top1']:.0f} regret {100 * v['regret']:.2f}% rho {v['spearman']:.2f}" for m, v in met.items()),
              flush=True)


if __name__ == "__main__":
    main()
