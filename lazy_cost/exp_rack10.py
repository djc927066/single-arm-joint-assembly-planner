"""Validate LazyStepCost on rack10 with dense clutter against full cuRobo planning.

    & $py -X utf8 -m lazy_cost.exp_rack10 [--contexts 30] [--cap 12] [--k 3]
    & $py -X utf8 -m lazy_cost.exp_rack10 --summarize lazy_cost/_output/<run>.jsonl

Context: a statically stable order and progress k (target = a post), installed parts at
their goals, the other parts crowded near the site (weight exp(-d/8 cm)), 6-10 boxes (half
are 15-25 cm walls between the site and the target's candidates). Up to ``cap`` candidates
whose footprint is clear of everything on the table.

Reference: every (candidate, grasp) that survives the hard prune is planned with the
sequential cuRobo MotionPlanner (both segments, the part being picked in the transit world,
held part attached on transfer); cost = topp duration, best grasp. Compared rankings:
lazy (top-K exact by construction w.r.t. its own planner calls), lower bound only, and
lower bound with blocked segments x1.25. One JSON line per context is appended as soon
as it finishes.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

PKG = Path(__file__).resolve().parent
OUT = PKG / "_output"


def make_context(scene, cat, rng, ci, cap, start_free=None, pick_k=None, boxes_range=(6, 10), crowd=0.08, gap=0.01):
    """start_free(world_args, q) -> bool rejects layouts in which the arm's starting
    configuration already collides (no real layout puts obstacles there).
    boxes_range: (min, max) number of extra boxes, (0, 0) = parts are the only obstacles;
    crowd: other parts are placed with weight exp(-d / crowd) to the site (<= 0: uniform);
    gap: minimum clearance between table footprints (m)."""
    from shapely.geometry import box as sbox
    targets = [p for p in scene.ids if cat["pick"][p] and cat["goal"][p]]
    pose = scene.pose
    bb = scene.site_box
    site_fp = sbox(*(bb[0, :2] - 0.02), *(bb[1, :2] + 0.02))
    robot_fp = sbox(-0.13, -0.13, 0.13, 0.13)
    for _ in range(300):
        order = scene.sample_order(rng)
        ks = [k for k, p in enumerate(order[:-1]) if p in targets]
        if not ks:
            continue
        k = int(rng.choice(ks)) if pick_k is None else pick_k(order, ks)
        p, installed, others = order[k], list(order[:k]), list(order[k + 1:])
        placed, polys = [], []
        for q in rng.permutation(others):
            keys = [c for c, _ in scene.candidates[q] if all(scene.polygons[q][c].distance(o) >= gap for o in polys)]
            if not keys:
                break
            d = np.array([np.linalg.norm(pose[q][c][:2, 3] - scene.site_xy) for c in keys])
            w = np.exp(-d / crowd) if crowd > 0 else np.ones(len(keys))
            c = keys[int(rng.choice(len(keys), p=w / w.sum()))]
            placed.append((f"{q}@{c}", str(q), pose[q][c]))
            polys.append(scene.polygons[q][c])
        if len(placed) != len(others):
            continue
        boxes, bpolys = [], []
        n = int(rng.integers(boxes_range[0], boxes_range[1] + 1))
        ref = [pose[p][c][:2, 3] for c, _ in scene.candidates[p]]
        for _ in range(600):
            if len(boxes) == n:
                break
            if len(boxes) % 2 == 0:
                tgt = ref[int(rng.integers(len(ref)))]
                xy = scene.site_xy + rng.uniform(0.35, 0.75) * (tgt - scene.site_xy) + rng.normal(0, 0.02, 2)
                dims = np.r_[rng.uniform(0.03, 0.05, 2), rng.uniform(0.15, 0.25)]
            else:
                xy = np.r_[rng.uniform(0.02, 0.52), rng.uniform(-0.28, 0.28)]
                dims = np.r_[rng.uniform(0.03, 0.06, 2), rng.uniform(0.08, 0.20)]
            fp = sbox(xy[0] - dims[0] / 2, xy[1] - dims[1] / 2, xy[0] + dims[0] / 2, xy[1] + dims[1] / 2)
            if fp.intersects(robot_fp) or fp.intersects(site_fp) or any(fp.distance(o) < 0.015 for o in polys + bpolys):
                continue
            T = np.eye(4)
            T[:3, 3] = [xy[0], xy[1], dims[2] / 2]
            boxes.append((f"box{ci}_{len(boxes)}", dims, T))
            bpolys.append(fp)
        cands = [c for c, _ in scene.candidates[p] if c in cat["pick"][p]
                 and all(scene.polygons[p][c].distance(o) >= gap for o in polys + bpolys)]
        if len(cands) < 5:
            continue
        prev = order[k - 1] if k else None
        start = cat["goal"][prev][min(cat["goal"][prev])]["retreat"] if prev and cat["goal"][prev] else np.zeros(6)
        if start_free is not None and not start_free((installed, placed, boxes), start):
            continue
        return dict(ctx=ci, order=order, k=k, part=p, installed=installed, placed=placed, boxes=boxes,
                    cands=list(rng.permutation(cands)[:cap]), start=start)
    return None


def metrics(pred, truth, k):
    """pred / truth: dict cand -> value (inf allowed in pred). Over candidates with finite truth."""
    from scipy.stats import spearmanr
    xs = [x for x in truth if np.isfinite(truth[x])]
    if len(xs) < 3:
        return None
    t = np.array([truth[x] for x in xs])
    p = np.array([pred.get(x, np.inf) for x in xs])
    p = np.where(np.isfinite(p), p, np.nanmax(np.where(np.isfinite(p), p, np.nan)) * 10 if np.isfinite(p).any() else 1.0)
    best = t.min()
    top1 = int(np.argmin(p))
    kk = min(k, len(xs))
    return dict(n=len(xs), spearman=float(spearmanr(p, t)[0]) if np.ptp(p) > 0 else 0.0,
                top1=float(t[top1] <= best + 1e-9), top1_within2pct=float(t[top1] <= best * 1.02),
                regret=float(t[top1] / best - 1),
                topk_recall=float(len(set(np.argsort(p)[:kk]) & set(np.argsort(t)[:kk])) / kk))


def summarize(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("metrics")]
    print(f"{path}\n{len(rows)} contexts")
    names = list(rows[0]["metrics"]) if rows else []
    print(f"{'ranking':16s} {'spearman':>9s} {'top1':>6s} {'top1<=2%':>9s} {'topK':>6s} {'regret':>8s} {'max regret':>11s}")
    out = {}
    for n in names:
        m = [r["metrics"][n] for r in rows if r["metrics"].get(n)]
        f = lambda k: float(np.mean([x[k] for x in m]))
        out[n] = {k: f(k) for k in ("spearman", "top1", "top1_within2pct", "topk_recall", "regret")}
        out[n]["max_regret"] = float(max(x["regret"] for x in m))
        v = out[n]
        print(f"{n:16s} {v['spearman']:9.3f} {v['top1']:6.2f} {v['top1_within2pct']:9.2f} {v['topk_recall']:6.2f} "
              f"{v['regret']:8.3f} {v['max_regret']:11.3f}")
    T = lambda k: [r["timing"][k] for r in rows]
    cand = np.array(T("candidates"), float)
    timing = dict(
        lazy_ms_per_context=float(np.mean(T("lazy_ms"))), lazy_ms_per_candidate=float(np.mean(np.array(T("lazy_ms")) / cand)),
        bounds_only_ms_per_candidate=float(np.mean(np.array(T("bounds_ms")) / cand)),
        reference_ms_per_candidate=float(np.mean(np.array(T("reference_ms")) / cand)),
        refined_segments_per_context=float(np.mean(T("refined_segments"))),
        reference_segments_per_context=float(np.mean(T("reference_segments"))),
        pruned_fraction=float(np.mean(np.array(T("pruned")) / cand)),
        blocked_pair_fraction=float(np.mean([r["timing"]["blocked_pairs"] / max(1, r["timing"]["pairs"]) for r in rows])),
        settled_agree_with_reference_median=float(np.median([d for r in rows for d in r.get("settled_vs_ref", [])] or [np.nan])))
    print("timing:", json.dumps({k: round(v, 3) for k, v in timing.items()}))
    return dict(contexts=len(rows), rankings=out, timing=timing)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="rack10")
    ap.add_argument("--contexts", type=int, default=30)
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--boxes", default="6,10", help="min,max extra boxes; 0,0 = parts only")
    ap.add_argument("--crowd", type=float, default=0.08, help="parts near the site, weight exp(-d/crowd); 0 = uniform")
    ap.add_argument("--gap", type=float, default=0.01, help="min clearance between table footprints (m)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--refiner", default="trajopt", choices=["trajopt", "graph"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--n-cand", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--summarize", default=None)
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return

    from curobo_bridge.runtime import prepare
    prepare()
    import torch
    from lazy_cost.timing import topp
    from curobo_bridge.export_model import make_wrs_arm
    from curobo_bridge.planner import PantheraPlanner

    from .checker import GpuChecker
    from .cost import LazyStepCost, t_acc
    from .endpoints import load_or_build
    from .refine import CuroboRefiner
    from .scene import load_scene

    rng = np.random.default_rng(args.seed)
    t0 = perf_counter()
    scene = load_scene(args.model, n_cand=args.n_cand)
    chk = GpuChecker(scene)
    cat, src = load_or_build(scene, chk)
    print(f"{args.model}: {len(scene.ids)} parts, {scene.n_orders} orders; catalogue {src} "
          f"({cat['seconds']:.0f}s when built): " + json.dumps(cat["report"]), flush=True)
    ref = CuroboRefiner(chk, batch=args.batch, method=args.refiner)
    for p in scene.ids:          # per-part preprocessing, like the grasp library
        ref.fit(p)
    lazy = LazyStepCost(scene, cat, chk, ref, k=args.k)
    lb_only = LazyStepCost(scene, cat, chk, None, k=args.k)
    arm = make_wrs_arm()
    planner = PantheraPlanner(arm=arm, attach_spheres=32)
    planner.set_world(chk.world_dict(), [])
    planner._planner("curobo")
    for p in scene.ids:
        planner.prefit(scene.meshes[p])
    print(f"setup {perf_counter() - t0:.1f}s (refiner {ref.setup_s:.1f}s)", flush=True)

    def tcp(q):
        pp, r = arm.fk(np.asarray(q, float))
        h = np.eye(4)
        h[:3, :3], h[:3, 3] = r, pp
        return h

    def plan_time(d, a, b, held=None):
        planner.set_world(d, [])
        if held is not None:
            planner.hold(a, scene.meshes[held[0]], tcp(a) @ np.linalg.inv(held[1]))
        path, _ = planner._curobo("curobo", a, b)
        if held is not None:
            planner.release()
        return np.inf if path is None else float(topp(path)[1][-1])

    OUT.mkdir(exist_ok=True)
    out = OUT / f"exp_{args.model}{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print("->", out, flush=True)
    for ci in range(args.contexts):
        def start_free(w, q):
            chk.set_world(chk.world_dict(*w))
            return bool(chk.configs_free(q)[0])

        c = make_context(scene, cat, rng, ci, args.cap, start_free, boxes_range=tuple(int(v) for v in args.boxes.split(",")),
                         crowd=args.crowd, gap=args.gap)
        if c is None:
            print(f"ctx {ci}: no context", flush=True)
            continue
        p, start, cands = c["part"], c["start"], c["cands"]
        call = (p, start, c["installed"], c["placed"], c["boxes"], cands)
        # ---- bounds only (no planner) and lazy ---------------------------------------------------
        torch.cuda.synchronize()
        t1 = perf_counter()
        r_lb, s_lb = lb_only.rank(*call)
        torch.cuda.synchronize()
        bounds_ms = 1e3 * (perf_counter() - t1)
        rs0 = dict(ref.stats)
        t1 = perf_counter()
        r_lz, s_lz = lazy.rank(*call)
        torch.cuda.synchronize()
        lazy_ms = 1e3 * (perf_counter() - t1)
        alive = [r for r in r_lz if not r.status.startswith("pruned")]
        # ---- reference: plan every surviving (x, g) --------------------------------------------
        t1 = perf_counter()
        W = chk.world_dict(c["installed"], c["placed"], c["boxes"])
        truth, n_ref, settled_vs_ref, seg_data = {}, 0, [], []
        lb_by = {r.key: r for r in r_lb}
        G = scene.grasps()[p]
        for r in alive:
            x = r.key
            Wx = dict(W, mesh=dict(W["mesh"], target=chk.mesh_entry(p, scene.pose[p][x])))
            vals = []
            for g in r.segments:
                e = cat["pick"][p][x][g]
                a = plan_time(Wx, start, e["hover"]) + t_acc(e["hover"], e["prepick"])
                b = plan_time(W, e["lift"], cat["goal"][p][g]["preins"], held=(p, G[g][0]))
                n_ref += 2
                for kind, val in (("transit", a), ("transfer", b)):
                    f = lb_by[x].features.get(g, {}).get(kind)
                    if f is not None:
                        seg_data.append(dict(cand=x, g=int(g), kind=kind, ref=val, how=lb_by[x].segments[g][kind][1],
                                             **{k: (None if v != v else v) for k, v in f.items()}))
                vals.append(a + b)
                for name, qa, qb, v in (("transit", start, e["hover"], a), ("transfer", e["lift"], cat["goal"][p][g]["preins"], b)):
                    val, how = r.segments[g][name]
                    if how in ("straight", "planned") and np.isfinite(v):
                        settled_vs_ref.append(val / v)
            truth[x] = min(vals)
        ref_ms = 1e3 * (perf_counter() - t1)
        arm.goto_given_conf(np.zeros(6))
        # ---- predictions ----------------------------------------------------------------------------
        pred_lazy = {r.key: (r.cost if r.status == "settled" else 1e3 + r.cost) for r in alive}
        pred_lb = {r.key: r.lb for r in r_lb if not r.status.startswith("pruned")}
        pred_pen = {}
        for r in r_lb:
            if r.status.startswith("pruned"):
                continue
            pred_pen[r.key] = min(sum(v * (1.0 if how == "straight" else 1.25) for v, how in s.values())
                                  for s in r.segments.values())
        m = {"lazy": metrics(pred_lazy, truth, args.k), "lower_bound": metrics(pred_lb, truth, args.k),
             "bound_x1.25": metrics(pred_pen, truth, args.k)}
        row = dict(ctx=ci, part=p, k=c["k"], installed=len(c["installed"]), placed=len(c["placed"]), boxes=len(c["boxes"]),
                   metrics=m if all(m.values()) else None,
                   timing=dict(candidates=len(cands), pruned=s_lz["pruned"], pairs=s_lz["pairs"], blocked_pairs=s_lz["blocked_pairs"],
                               blocked_transit=s_lz["blocked_transit"], blocked_transfer=s_lz["blocked_transfer"],
                               bounds_ms=bounds_ms, lazy_ms=lazy_ms, refined_segments=s_lz["refined_segments"],
                               refine_batches=s_lz["refine_batches"], reference_ms=ref_ms, reference_segments=n_ref,
                               stage_ms={k: s_lz[k] for k in s_lz if k.endswith("_ms")},
                               refiner={k: ref.stats.get(k, 0) - rs0.get(k, 0) for k in ref.stats}),
                   settled_vs_ref=settled_vs_ref, seg_data=seg_data,
                   lb_pairs={r.key: {str(g): {k: s[k][1] for k in s} for g, s in r.segments.items()} for r in r_lb if not r.status.startswith("pruned")},
                   lazy=[dict(key=r.key, status=r.status, cost=r.cost, lb=r.lb) for r in r_lz],
                   truth={k: (v if np.isfinite(v) else None) for k, v in truth.items()})
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")
        mm = m["lazy"]
        print(f"ctx {ci}: {p} k={c['k']} placed {len(c['placed'])} boxes {len(c['boxes'])} | cands {len(cands)} pruned {s_lz['pruned']} "
              f"blocked {s_lz['blocked_pairs']}/{s_lz['pairs']} refined {s_lz['refined_segments']} | lazy {lazy_ms:.0f} ms "
              f"(bounds {bounds_ms:.0f}) vs reference {ref_ms:.0f} ms ({n_ref} plans) | "
              + ("lazy top1 %d regret %.3f topK %.2f" % (mm["top1"], mm["regret"], mm["topk_recall"]) if mm else "too few for metrics"),
              flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
