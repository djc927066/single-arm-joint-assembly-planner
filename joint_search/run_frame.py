"""Joint backward search on the triangular frame (parts are the only obstacles).

    & $py -X utf8 -m joint_search.run_frame [--B 10] [--M 10] [--N 5] [--search eager|delayed]
Output: joint_search/_output/frame_<time>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

OUT = Path(__file__).resolve().parent / "_output"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--M", type=int, default=None,
                    help="children kept per (node, part) and expansion: default 10 for eager, 5 (grasps of the root) for delayed")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--penalty", default=None, help="penalty model json (default lazy_cost/penalty_model.json); e.g. lazy_cost/penalty_depth.json")
    ap.add_argument("--detax", action="store_true", help="time cuRobo paths after shortcut + corner blending (executed path)")
    ap.add_argument("--grasps-per-pose", type=int, default=8,
                    help="grasps screened per candidate pose and expansion, smallest bound first; 0 = all")
    ap.add_argument("--sampler", default="fps", choices=("fps", "random"), help="candidate poses (lazy_cost.scene)")
    ap.add_argument("--placements", default="sealp", choices=("sealp", "goal"),
                    help="resting orientations of the candidate poses: sealp = SEALP/BLS rot_cands (default), "
                         "goal = goal orientation only")
    ap.add_argument("--grid", default="fine", choices=("fine", "shared"),
                    help="candidate grid: fine = one fine grid (--grid-step) for every part, linspace to both "
                         "table edges (default); shared = one 4 cm np.arange grid (before 2026-10-24)")
    ap.add_argument("--grid-step", type=float, default=0.015, help="spacing of the fine grid (m)")
    ap.add_argument("--keepout", default="sealp", choices=("sealp", "box"),
                    help="robot / assembly keepout of the candidate footprints: sealp = SEALP rule, arm rectangle "
                         "12 cm + 1 cm from the assembly (default); box = 13 cm square + base AABB grown by 3 cm")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0",
                    help="assembly centre x,y,z (m): where the base sits, the whole assembly moves with it "
                         "(SEALP --goal-pos, same default); 'model' = as in the model files")
    ap.add_argument("--home-clear", type=float, default=0.03,
                    help="clearance (m) the arm at HOME keeps from every part on the table and from the "
                         "finished assembly (SEALP robot_home_clearance, 3 cm); 0 = collision only")
    ap.add_argument("--orient", default="first", choices=("first", "best"),
                    help="orientation per candidate position (placements sealp): first = SEALP, the first one in "
                         "SEALP order that is feasible in the search state (default); best = every feasible one, "
                         "ranked by cost")
    ap.add_argument("--search", default="eager", choices=("eager", "delayed"),
                    help="eager (default since 2026-10-26, BLS-like) = part, pose and grasp decided together, the M "
                         "best (pose, grasp) per (node, part) go on, the move into a part is resolved one level later "
                         "(search.py); delayed = each part's initial pose is chosen one level later, when the part "
                         "before it is known (search_delayed.py, default 2026-10-25)")
    ap.add_argument("--K-h", type=int, default=10, help="delayed search: head poses evaluated exactly per expansion (prefilter keeps K_h)")
    ap.add_argument("--quota", type=int, default=2, help="delayed search: nodes kept per last-installed part in the first levels")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    if args.M is None:
        args.M = 10 if args.search == "eager" else 5

    from lazy_cost.refine import CuroboRefiner
    from .problem import Problem
    from .search import beam_search
    from .validate import select

    t0 = perf_counter()
    pb = Problem(args.model, penalty=args.penalty, grasps_per_pose=args.grasps_per_pose,
                 sampler=args.sampler, placements=args.placements,
                 orient=args.orient, grid=args.grid, grid_step=args.grid_step, keepout=args.keepout,
                 home_clear=args.home_clear,
                 center=None if args.goal_pos == "model" else tuple(float(v) for v in args.goal_pos.split(",")))
    t1 = perf_counter()
    ref = CuroboRefiner(pb.chk, detax=args.detax)
    for p in pb.ids:
        ref.fit(p)
    setup_s = perf_counter() - t0
    timing = dict(pb.timing, planners_s=perf_counter() - t1, setup_s=setup_s)
    from lazy_cost.scene import key_xy
    sc = pb.scene
    cand_stats = {p: dict(fitting_positions=len(sc.fit_xy[p]), options_after_home=len(sc.options[p]),
                          home_removed=pb.home_report[p][1], fps_positions=len({key_xy(x) for x, _ in sc.candidates[p]}),
                          fps_poses=len(sc.candidates[p]), kept_positions=len({key_xy(x) for x in pb.cat["pick"][p]}),
                          kept_poses=len(pb.cat["pick"][p]),
                          kept_pairs=sum(len(v) for v in pb.cat["pick"][p].values()), usable_grasps=len(pb.cat["goal"][p]))
                  for p in pb.ids}
    print(f"{args.model}: parts {list(pb.ids)}, {pb.scene.n_orders} stable orders, {pb.scene.n_cand} candidate poses; "
          f"usable grasps per part {[len(pb.cat['goal'][p]) for p in pb.ids]}"
          + f"; setup {setup_s:.0f}s (catalogue, planners)", flush=True)
    print(f"  resting orientations: { {p: [u[0] or 'goal' for u in r.get('used', [('', None, None)])] for p, r in pb.scene.placement_report.items()} }", flush=True)
    cr = pb.center_report
    print(f"  assembly centre {[round(v, 3) for v in cr['center']]}: "
          + (f"MOVED by {cr['shift']} m, the arm at HOME hit the finished assembly at the requested one"
             if cr["moved"] else "the arm at HOME clears the finished assembly"), flush=True)
    print(f"  candidate poses removed because the arm at HOME would hit the part: "
          f"{ {p: r[1] for p, r in pb.home_report.items()} }", flush=True)

    print(f"\n[search] beam B={args.B}, M={args.M} pairs per part and expansion", flush=True)
    if args.search == "delayed":
        from .search_delayed import beam_search_delayed
        complete, st = beam_search_delayed(pb, B=args.B, M=args.M, quota=args.quota, K_h=args.K_h)
    else:
        complete, st = beam_search(pb, B=args.B, M=args.M, N=10 ** 9)
    seen, plans = set(), []
    for n in sorted(complete, key=lambda n: n.g_est):            # distinct order + layout, best grasps
        k = tuple((p, x) for p, x, _ in n.suffix)
        if k not in seen:
            seen.add(k)
            plans.append(n)
        if len(plans) == args.N:
            break
    print(f"  {st['expansions']} expansions, {st.get('pairs', '-')} (pose, grasp) pairs, {st.get('screened', '-')} screened, "
          f"{st.get('pruned', '-')} pruned, "
          f"{len(complete)} complete plans ({len(seen)} distinct order+layout); search {st['search_s']:.1f}s"
          + (f"; orientation fallbacks {st['orient_fallback']}" if args.search == "eager" and pb.orient == "first" else ""), flush=True)
    if hasattr(pb.scene, "online_stats"):
        o = pb.scene.online_stats
        print(f"  online step checks: {o['lp_calls']} stability LPs ({o['lp_s']:.1f}s), {o['sweep_calls']} insertion sweeps ({o['sweep_s']:.1f}s)", flush=True)

    print(f"\n[validate] full-process validation (cuRobo plans every free segment + mesh collision check of the "
          f"whole execution) of the {len(plans)} best distinct plans", flush=True)
    res = select(pb, plans, ref)
    b = res["best"]
    OUT.mkdir(exist_ok=True)
    out = OUT / f"frame{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    if b is None:
        print(f"  no plan passed the full-process validation (rejected: {[(r['plan'], r['reason']) for r in res['rejected']]})", flush=True)
        out.write_text(json.dumps(dict(args=vars(args), setup_s=setup_s, search=st, tried=res["tried"], rejected=res["rejected"],
                                       plans=[dict(steps=n.suffix, est=n.g_est, lb=n.g_lb) for n in plans],
                                       chosen=None, validate_s=res["validate_s"], curobo=res["stats"]), indent=1, default=str),
                       encoding="utf-8")
        print("->", out)
        return
    print(f"  chosen plan: {res['time']:.3f} s, full-process validation OK; {res['stats']['planned']} cuRobo plans "
          f"({res['validate_s']:.1f}s validation)"
          + (f"; rejected plans {[(r['plan'], r['reason']) for r in res['rejected']]}" if res['rejected'] else "")
          + (f"; segments that needed a retry: {res['stats']['retried_segments']}" if res['stats'].get('retried_segments') else ""))
    print("\n  forward plan:")
    for k, (p, x, g) in enumerate(b.suffix):
        print(f"   {k + 1}. {p:8s} initial pose {x:22s} grasp {g}")
    print("\n  segments (s):  kind      part      executed  how       cuRobo")
    for r in res["rows"]:
        print(f"                 {r['kind']:9s} {str(r['part']):9s} {r['value']:8.3f}  {r['how']:9s} {r['planned']:8.3f}")
    out.write_text(json.dumps(dict(
        args=vars(args), timing=dict(timing, search_s=st["search_s"], validate_s=res["validate_s"], total_s=perf_counter() - t0),
        catalogue=dict(source=pb.cat_source, seconds=pb.cat.get("seconds"), report=pb.cat.get("report")), candidates=cand_stats,
        home_report=pb.home_report, center_report=pb.center_report, placement_report=pb.scene.placement_report, penalty=pb.penalty.name, setup_s=setup_s, search=st, online_step=getattr(pb.scene, "online_stats", None),
        plans=[dict(steps=n.suffix, est=n.g_est, lb=n.g_lb) for n in plans], tried=res["tried"], rejected=res["rejected"],
        chosen=dict(steps=b.suffix, est=b.g_est, lb=b.g_lb, time=res["time"], rows=res["rows"], check_rows=res["check_rows"]),
        validate_s=res["validate_s"], curobo=res["stats"]), indent=1, default=str), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
