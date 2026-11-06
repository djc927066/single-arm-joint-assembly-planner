"""exp_rack10 + detour indicators (lazy_cost.detour) logged for every blocked segment.

    python -m lazy_cost.exp_detour --model rack10 --boxes 0,0 --crowd 0 --gap 0.03 --tag _feat3_parts

Same contexts / reference as exp_rack10 (same seed -> same contexts). Each seg_data row of a
blocked segment additionally carries qa, qb, desc (descent time included in lb of a transit)
and "detour" (DetourProbe indicators); the row gets timing.detour (ms per stage).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .exp_rack10 import OUT, make_context, metrics, summarize

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
    from .detour import DetourProbe, blend
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
    probe = DetourProbe(chk)
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

    last = {}

    def plan_time(d, a, b, held=None):
        planner.set_world(d, [])
        if held is not None:
            planner.hold(a, scene.meshes[held[0]], tcp(a) @ np.linalg.inv(held[1]))
        path, _ = planner._curobo("curobo", a, b)
        if held is not None:
            planner.release()
        last["path"] = path
        return np.inf if path is None else float(topp(path)[1][-1])

    def debias(P, raw, held=None, extras=None):
        """De-taxed reference: min(raw TOPP, TOPP(greedy shortcut of the cuRobo path),
        TOPP(corner-blended shortcut) if that curve is free) - all w.r.t. our checker
        (batch_free: same test as the screening, incl. held part vs arm)."""
        if P is None:
            return raw, None
        hb = [] if held is None else [held]
        eb = [] if extras is None else list(extras)
        hi, ei = (0 if held is not None else -1), (0 if extras is not None else -1)

        def free_lines(A, B):
            return chk.batch_free([(A, B, hi, ei)], held_bank=hb, extra_bank=eb, step=0.02)[0]

        if len(P) > 60:
            P = P[np.unique(np.r_[np.linspace(0, len(P) - 1, 60).round().astype(int)])]
        out, i = [P[0]], 0
        while i < len(P) - 1:
            js = np.arange(len(P) - 1, i, -1)
            ok = free_lines(np.repeat(P[i][None], len(js), 0), P[js])
            if not ok.any():
                return raw, None
            i = int(js[int(np.argmax(ok))])
            out.append(P[i])
        S = np.stack(out)
        best = min(raw, float(topp(S)[1][-1]))
        if len(S) > 2:
            Bc = blend(S)
            if free_lines(Bc[:-1], Bc[1:]).all():
                best = min(best, float(topp(Bc)[1][-1]))
        return best, len(S)

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
        # ---- detour indicators of every blocked segment (world = W, set by lb_only.rank) -------
        G = scene.grasps()[p]
        part_sph = scene.held_spheres(p)
        gl = sorted(cat["goal"][p])
        gidx = {g: i for i, g in enumerate(gl)}
        xs = [r.key for r in r_lb if not r.status.startswith("pruned")]
        xidx = {x: i for i, x in enumerate(xs)}
        segs, keys = [], []
        for r in r_lb:
            if r.status.startswith("pruned"):
                continue
            for g, sg in r.segments.items():
                e = cat["pick"][p][r.key][g]
                for kind in ("transit", "transfer"):
                    if sg[kind][1] != "bound":
                        continue
                    f = r.features[g][kind]
                    qa, qb = (start, e["hover"]) if kind == "transit" else (e["lift"], cat["goal"][p][g]["preins"])
                    segs.append(dict(qa=np.asarray(qa, float), qb=np.asarray(qb, float),
                                     s_in=None if f["s_in"] != f["s_in"] else f["s_in"],
                                     s_out=None if f["s_out"] != f["s_out"] else f["s_out"],
                                     held=gidx[g] if kind == "transfer" else -1, extra=xidx[r.key] if kind == "transit" else -1))
                    keys.append((r.key, int(g), kind))
        torch.cuda.synchronize()
        t1 = perf_counter()
        det, det_t = probe.run(segs, held_bank=[(part_sph, G[g][0]) for g in gl],
                               extra_bank=[chk.obstacle_spheres(part_sph, scene.pose[p][x]) for x in xs], step=0.02)
        torch.cuda.synchronize()
        det_t["wall_ms"] = 1e3 * (perf_counter() - t1)
        det_by = {k: (s, d) for k, s, d in zip(keys, segs, det)}
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
        chk.set_world(W)
        rng2 = np.random.default_rng(1000 + ci)
        debias_ms = 0.0
        for r in alive:
            x = r.key
            Wx = dict(W, mesh=dict(W["mesh"], target=chk.mesh_entry(p, scene.pose[p][x])))
            ex = [chk.obstacle_spheres(part_sph, scene.pose[p][x])]
            vals = []
            for g in r.segments:
                e = cat["pick"][p][x][g]
                dsc = t_acc(e["hover"], e["prepick"])
                hold = (part_sph, G[g][0])
                a0 = plan_time(Wx, start, e["hover"])
                Pa = last["path"]
                b0 = plan_time(W, e["lift"], cat["goal"][p][g]["preins"], held=(p, G[g][0]))
                Pb = last["path"]
                a, b = a0 + dsc, b0
                n_ref += 2
                td = perf_counter()
                a_sc, na = debias(Pa, a0, extras=ex)
                b_sc, nb = debias(Pb, b0, held=hold)
                debias_ms += 1e3 * (perf_counter() - td)
                extra_ref = {}
                if rng2.random() < 0.04:          # noise sample: re-plan both segments once more
                    a2 = plan_time(Wx, start, e["hover"])
                    a2_sc, _ = debias(last["path"], a2, extras=ex)
                    b2 = plan_time(W, e["lift"], cat["goal"][p][g]["preins"], held=(p, G[g][0]))
                    b2_sc, _ = debias(last["path"], b2, held=hold)
                    extra_ref = {"transit": dict(ref2=a2 + dsc, ref2_sc=a2_sc + dsc), "transfer": dict(ref2=b2, ref2_sc=b2_sc)}
                for kind, val, vsc, nv in (("transit", a, a_sc + dsc, na), ("transfer", b, b_sc, nb)):
                    f = lb_by[x].features.get(g, {}).get(kind)
                    if f is not None:
                        row_s = dict(cand=x, g=int(g), kind=kind, ref=val, ref_sc=vsc, sc_vertices=nv,
                                     how=lb_by[x].segments[g][kind][1], desc=dsc if kind == "transit" else 0.0,
                                     **extra_ref.get(kind, {}),
                                     **{k: (None if v != v else v) for k, v in f.items()})
                        if (x, int(g), kind) in det_by:
                            s_, d_ = det_by[x, int(g), kind]
                            row_s.update(qa=s_["qa"].tolist(), qb=s_["qb"].tolist(), detour=d_)
                        seg_data.append(row_s)
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
                               refine_batches=s_lz["refine_batches"], detour=det_t, debias_ms=debias_ms, reference_ms=ref_ms, reference_segments=n_ref,
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
              f"blocked {s_lz['blocked_pairs']}/{s_lz['pairs']} detour {len(segs)} segs {det_t.get('wall_ms', 0):.0f} ms "
              f"(esc {det_t.get('escape_ms', 0):.0f} chk {det_t.get('detour_check_ms', 0):.0f} topp {det_t.get('topp_ms', 0):.0f}) refined {s_lz['refined_segments']} | lazy {lazy_ms:.0f} ms "
              f"(bounds {bounds_ms:.0f}) vs reference {ref_ms:.0f} ms ({n_ref} plans) | "
              + ("lazy top1 %d regret %.3f topK %.2f" % (mm["top1"], mm["regret"], mm["topk_recall"]) if mm else "too few for metrics"),
              flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
