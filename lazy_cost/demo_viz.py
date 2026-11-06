"""Record a few lazy-ranking runs (20 candidates) for the HTML visualisation.

    & $py -X utf8 -m lazy_cost.demo_viz [--scenes 3] [--cap 20]
Output: lazy_cost/_output/viz_data.json (+ viz.html built by make_viz.py)

Step shown: the LAST post of the order (rack10 always ends with the top shelf, which cannot
be picked from the table, so this is the last step a pickable part is installed), i.e. one
of the first expansions of the reverse search. Every surviving candidate is also planned
with sequential cuRobo for comparison.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-alive", type=int, default=7)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    import torch
    from lazy_cost.timing import topp
    from curobo_bridge.export_model import make_wrs_arm
    from curobo_bridge.planner import PantheraPlanner
    from shapely.geometry import MultiPoint

    from .checker import GpuChecker
    from .cost import LazyStepCost, t_acc
    from .endpoints import load_or_build
    from .exp_rack10 import make_context
    from .refine import CuroboRefiner
    from .scene import AssemblyScene

    rng = np.random.default_rng(args.seed)
    scene = AssemblyScene(importlib.import_module("scenes.benchmarks.rack10").spec(), n_cand=40)
    chk = GpuChecker(scene)
    cat, _ = load_or_build(scene, chk)
    ref = CuroboRefiner(chk, batch=1)
    for p in scene.ids:
        ref.fit(p)
    lazy = LazyStepCost(scene, cat, chk, ref, k=args.k)
    arm = make_wrs_arm()
    planner = PantheraPlanner(arm=arm, attach_spheres=32)
    planner.set_world(chk.world_dict(), [])
    planner._planner("curobo")
    for p in scene.ids:
        planner.prefit(scene.meshes[p])

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

    def start_free(w, q):
        chk.set_world(chk.world_dict(*w))
        return bool(chk.configs_free(q)[0])

    def footprint(pid, T):
        v = scene.meshes[pid].vertices @ T[:3, :3].T + T[:3, 3]
        return [list(map(float, xy)) for xy in MultiPoint(v[:, :2]).convex_hull.exterior.coords], float(v[:, 2].max())

    posts = {p for p in scene.ids if p.startswith(("lp_", "up_"))}
    last_post = lambda order, ks: max(k for k, p in enumerate(order) if p in posts)
    scenes, tries = [], 0
    while len(scenes) < args.scenes and tries < 60:
        tries += 1
        c = make_context(scene, cat, rng, tries, args.cap, start_free, last_post)
        if c is None:
            continue
        p, start, cands = c["part"], c["start"], c["cands"]
        lazy.trace = []
        torch.cuda.synchronize()
        t0 = perf_counter()
        ranked, st = lazy.rank(p, start, c["installed"], c["placed"], c["boxes"], cands)
        torch.cuda.synchronize()
        lazy_ms = 1e3 * (perf_counter() - t0)
        alive = [r for r in ranked if not r.status.startswith("pruned")]
        if len(alive) < args.min_alive:
            print(f"try {tries}: {p} only {len(alive)} candidates survive, skipped", flush=True)
            continue
        W = chk.world_dict(c["installed"], c["placed"], c["boxes"])
        G = scene.grasps()[p]
        truth = {}
        t0 = perf_counter()
        for r in alive:
            Wx = dict(W, mesh=dict(W["mesh"], target=chk.mesh_entry(p, scene.pose[p][r.key])))
            vals = []
            for g in r.segments:
                e = cat["pick"][p][r.key][g]
                vals.append(plan_time(Wx, start, e["hover"]) + t_acc(e["hover"], e["prepick"])
                            + plan_time(W, e["lift"], cat["goal"][p][g]["preins"], held=(p, G[g][0])))
            truth[r.key] = min(vals)
        ref_ms = 1e3 * (perf_counter() - t0)
        arm.goto_given_conf(np.zeros(6))
        obstacles = []
        for q in c["installed"]:
            poly, h = footprint(q, scene.goal[q])
            obstacles.append(dict(kind="installed", name=q, poly=poly, h=h))
        for name, q, T in c["placed"]:
            poly, h = footprint(q, T)
            obstacles.append(dict(kind="placed", name=q, poly=poly, h=h))
        for name, dims, T in c["boxes"]:
            x, y = T[0, 3], T[1, 3]
            dx, dy = dims[0] / 2, dims[1] / 2
            obstacles.append(dict(kind="box", name=name, h=float(dims[2]),
                                  poly=[[x - dx, y - dy], [x + dx, y - dy], [x + dx, y + dy], [x - dx, y + dy], [x - dx, y - dy]]))
        poly, h = footprint("base", scene.goal["base"])
        obstacles.append(dict(kind="site", name="base", poly=poly, h=h))
        tr_order = sorted(truth, key=truth.get)
        cand = []
        for i, r in enumerate(ranked):
            poly, _ = footprint(p, scene.pose[p][r.key])
            cand.append(dict(key=r.key, poly=poly, xy=[float(v) for v in scene.pose[p][r.key][:2, 3]],
                             yaw=int(r.key.split("yaw")[1]), status=r.status,
                             lb=None if not np.isfinite(r.lb) else r.lb,
                             cost=None if not np.isfinite(r.cost) else r.cost,
                             lazy_rank=i + 1 if not r.status.startswith("pruned") else None,
                             truth=truth.get(r.key), truth_rank=tr_order.index(r.key) + 1 if r.key in truth else None,
                             segments={str(g): {k: [None if not np.isfinite(v[0]) else v[0], v[1]] for k, v in s.items()}
                                       for g, s in r.segments.items()}))
        scenes.append(dict(part=p, k=c["k"], order=c["order"], n_parts=len(scene.ids), K=args.k,
                           installed=c["installed"], lazy_ms=lazy_ms, reference_ms=ref_ms, stats={k: v for k, v in st.items()},
                           trace=lazy.trace, candidates=cand, obstacles=obstacles,
                           table=dict(center=scene.table_center[:2].tolist(), size=scene.table_size[:2].tolist())))
        print(f"scene {len(scenes)}: {p} (step {c['k'] + 1}/{len(scene.ids)}), {len(cands)} candidates, {len(alive)} survive, "
              f"lazy {lazy_ms:.0f} ms ({st['refined_segments']} cuRobo calls) vs reference {ref_ms:.0f} ms", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "viz_data.json").write_text(json.dumps(dict(scenes=scenes), default=float), encoding="utf-8")
    print("->", OUT / "viz_data.json")


if __name__ == "__main__":
    main()
