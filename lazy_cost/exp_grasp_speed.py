"""Common-grasp screening: cuRobo GPU batch vs WRS CPU one-by-one, same rule, same order.

    & $py -X utf8 -m lazy_cost.exp_grasp_speed [--model triangular_frame6] [--parts beam_CA ...]

Rule (both): world = table + assembly base; a grasp g is goal-feasible if the insertion pose
goal_p @ G has an IK solution that is collision-free; a pair (candidate x, g) is kept if g is
goal-feasible and the pick pose T_x @ G has a collision-free IK solution.
Order (both): goal side for all grasps first, pick side only for goal-feasible grasps.
  cuRobo : BatchIK (1024 poses per batch, 32 seeds, 4 returned solutions) + one batched sphere
           collision pass per stage (GpuChecker), every returned solution may be used
  WRS    : PantheraHTSglArm.ik one pose at a time (TracIK if installed, else WRS JLC IK, one
           solution) + mesh-mesh collision of the arm links (WRS/ODE), as a WRS pick-and-place
           script / wrs.grasp.reasoner.reason_common_gids would do
WRS is too slow to run completely (~0.27 s per IK call without TracIK, ~12 000 calls for one
part), so it is MEASURED on a random sample (--sample goal-side and pick-side calls per part) and
its full time is ESTIMATED = measured time per call x number of calls of the full screening
(goal side: all grasps; pick side: candidates x WRS goal-feasible grasps, at the sampled rate).
On the sampled poses the two methods are also compared (same decision or not).
Output: lazy_cost/_output/grasp_speed_<model>.json and a printed table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--full", action="store_true", help="run WRS completely (feasible with TracIK) and compare result sets")
    ap.add_argument("--tracik-timeout", type=float, default=0.01,
                    help="TracIK time limit (s). WRS's Panthera default 0.002 is too short on this machine: it fails even "
                         "when seeded with an exact solution; 0.01 solves 120/120 insertion poses")
    ap.add_argument("--tracik-type", default="Speed", help="Speed (return the first solution) or Distance (closest to the seed)")
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    import torch
    import wrs.modeling.collision_model as mcm
    import wrs.basis.trimesh as wtrm
    from curobo_bridge.assembly_demo import BatchIK
    from curobo_bridge.export_model import make_wrs_arm, wrs_collision_parts
    from wrs.robot_sim.manipulators.panthera_ht import panthera_ht as pht
    from .checker import GpuChecker
    from .endpoints import _ik_free
    from .scene import load_scene

    sync = torch.cuda.synchronize
    sc = load_scene(args.model)
    chk = GpuChecker(sc)
    chk.set_world(chk.world_dict())
    t = perf_counter()
    bik = BatchIK(batch=1024, seeds=32, returns=4)
    bik.solve(np.repeat(np.eye(4)[None], 1024, 0))
    sync()
    curobo_setup = perf_counter() - t
    arm = make_wrs_arm()
    links = [l for _, _, l in wrs_collision_parts(arm, False)]
    T = np.eye(4)
    T[:3, 3] = sc.table_center
    obs = [mcm.gen_box(xyz_lengths=np.asarray(sc.table_size, float), pos=T[:3, 3], rotmat=T[:3, :3]),
           mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(sc.meshes["base"].vertices),
                                                  faces=np.asarray(sc.meshes["base"].faces)))]
    obs[1].pose = (sc.goal["base"][:3, 3].copy(), sc.goal["base"][:3, :3].copy())
    ik_backend = "TracIK" if getattr(pht, "_HAS_TRAC_IK", False) else "WRS JLC IK"
    if getattr(pht, "_HAS_TRAC_IK", False):
        import os
        from trac_ik.trac_ik import TracIK
        arm.manipulator._ik_solver = TracIK("base_link", "link6", os.path.join(os.path.dirname(pht.__file__),
                                            "Panthera-HT_description_with_finger.urdf"),
                                            timeout=args.tracik_timeout, solver_type=args.tracik_type)
        ik_backend = f"TracIK ({args.tracik_type}, timeout {1e3 * args.tracik_timeout:g} ms)"

    wt = dict(ik=0.0, cc=0.0, ik_calls=0, cc_calls=0)

    def wrs_ok(P):
        t = perf_counter()
        q = arm.ik(P[:3, 3], P[:3, :3])
        wt["ik"] += perf_counter() - t
        wt["ik_calls"] += 1
        if q is None:
            return 0
        t = perf_counter()
        arm.goto_given_conf(np.asarray(q, float))
        hit = any(l.cmodel.is_mcdwith(obs) for l in links)
        wt["cc"] += perf_counter() - t
        wt["cc_calls"] += 1
        return 1 if hit else 2      # 1 = IK only, 2 = IK + free

    def curobo_stage(poses):
        """cuRobo IK (batched) then one batched collision pass; -> usable [N] and (ik_s, cc_s)."""
        sync(); t = perf_counter()
        qq, ok = bik.solve(poses)
        sync(); t_ik = perf_counter() - t
        N, Rn = ok.shape
        idx = np.flatnonzero(ok.reshape(-1))
        free = np.zeros(N * Rn, bool)
        t = perf_counter()
        if len(idx):
            free[idx] = chk.batch_free([(qq.reshape(-1, 6)[idx], None, -1, -1)])[0]
        sync()
        return free.reshape(N, Rn).any(1), ok.any(1), (t_ik, perf_counter() - t)

    rows = []
    parts = args.parts or list(sc.ids)
    for pid in parts:
        Gs = np.stack([G for G, _ in sc.grasps()[pid]])
        cands = np.stack([Tc for _, Tc in sc.candidates[pid]])
        n, m = len(Gs), len(cands)
        ins = sc.goal[pid][None] @ Gs
        # ---- cuRobo ----------------------------------------------------------------------------
        sync(); t0 = perf_counter()
        q, u = _ik_free(bik, chk, ins)
        sync(); c_goal = perf_counter() - t0
        g1 = np.flatnonzero(u.any(1))
        t1 = perf_counter()
        Pk = (cands[:, None] @ Gs[g1][None]).reshape(-1, 4, 4)
        qp, up = _ik_free(bik, chk, Pk) if len(Pk) else (None, np.zeros((0, 1), bool))
        sync(); c_pick = perf_counter() - t1
        c_pairs = {(i // len(g1), int(g1[i % len(g1)])) for i in np.flatnonzero(up.any(1))} if len(g1) else set()
        if args.full:
            rows.append(full_compare(pid, Gs, cands, ins, curobo_stage, wrs_ok, wt, arm))
            continue
        # ---- WRS: measured on a sample, full time estimated ----------------------------------------
        rs = np.random.default_rng(0)
        gi = rs.choice(n, min(args.sample, n), replace=False)
        t0 = perf_counter()
        w_goal = np.array([wrs_ok(ins[g]) for g in gi])
        w_goal_call = (perf_counter() - t0) / len(gi)
        frac_goal = float((w_goal == 2).mean())
        c_goal_on = np.isin(gi, g1)
        pool = [(xi, int(g)) for xi in range(m) for g in g1]               # pick side on cuRobo goal-feasible grasps
        pi = rs.choice(len(pool), min(args.sample, len(pool)), replace=False) if pool else []
        t1 = perf_counter()
        w_pick = np.array([wrs_ok(cands[pool[i][0]] @ Gs[pool[i][1]]) for i in pi])
        w_pick_call = (perf_counter() - t1) / max(len(pi), 1)
        c_pick_on = np.array([(pool[i][0], pool[i][1]) in c_pairs for i in pi])
        arm.goto_given_conf(np.zeros(6))
        w_goal_est = w_goal_call * n
        w_pick_est = w_pick_call * m * frac_goal * n
        r = dict(part=pid, grasps=n, candidates=m,
                 curobo=dict(goal_ik_free=int(len(g1)), pick_poses=int(len(g1) * m), pairs=len(c_pairs),
                             goal_s=c_goal, pick_s=c_pick, total_s=c_goal + c_pick),
                 wrs=dict(sampled_goal=len(gi), sampled_pick=len(pi), ms_per_goal_call=1e3 * w_goal_call,
                          ms_per_pick_call=1e3 * w_pick_call, goal_feasible_rate=frac_goal,
                          est_goal_s=w_goal_est, est_pick_s=w_pick_est, total_s=w_goal_est + w_pick_est,
                          measured_s=w_goal_call * len(gi) + w_pick_call * len(pi)),
                 agree=dict(goal_same=float(np.mean((w_goal == 2) == c_goal_on)), goal_wrs_yes=int((w_goal == 2).sum()),
                            goal_curobo_yes=int(c_goal_on.sum()), goal_both_yes=int(((w_goal == 2) & c_goal_on).sum()),
                            pick_same=float(np.mean((w_pick == 2) == c_pick_on)) if len(pi) else None,
                            pick_wrs_yes=int((w_pick == 2).sum()), pick_curobo_yes=int(c_pick_on.sum()),
                            pick_both_yes=int(((w_pick == 2) & c_pick_on).sum())))
        r["speedup"] = r["wrs"]["total_s"] / max(r["curobo"]["total_s"], 1e-9)
        rows.append(r)
        print(f"{pid:8s} grasps {n:5d} | cuRobo {r['curobo']['total_s']:6.2f}s goal {len(g1):4d} pairs {len(c_pairs):5d} | "
              f"WRS est. {r['wrs']['total_s']:7.0f}s ({r['wrs']['ms_per_goal_call']:.0f}/{r['wrs']['ms_per_pick_call']:.0f} ms per call) | "
              f"x{r['speedup']:.0f} | same decision goal {r['agree']['goal_same']:.2f} pick {r['agree']['pick_same']}", flush=True)
    if args.full:
        tot = lambda who, k: sum(r[who][k] for r in rows)
        summary = dict(model=args.model, mode="full", ik_backend_wrs=ik_backend, curobo_setup_s=curobo_setup, parts=rows,
                       total={f"{w}_{k}": tot(w, k) for w in ("curobo", "wrs") for k in ("total_s", "ik_s", "cc_s", "pairs")})
        summary["total"]["speedup"] = summary["total"]["wrs_total_s"] / summary["total"]["curobo_total_s"]
        OUT.mkdir(exist_ok=True)
        (OUT / f"grasp_speed_{args.model}_full.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        T_ = summary["total"]
        print(f"TOTAL ({ik_backend}, full run): cuRobo {T_['curobo_total_s']:.1f}s (IK {T_['curobo_ik_s']:.1f}, collision {T_['curobo_cc_s']:.2f}) "
              f"vs WRS {T_['wrs_total_s']:.1f}s (IK {T_['wrs_ik_s']:.1f}, collision {T_['wrs_cc_s']:.1f}) -> x{T_['speedup']:.1f}; "
              f"pairs cuRobo {T_['curobo_pairs']}, WRS {T_['wrs_pairs']}")
        return
    tot = lambda who, k: sum(r[who][k] for r in rows)
    summary = dict(model=args.model, ik_backend_wrs=ik_backend, curobo_setup_s=curobo_setup, parts=rows,
                   total=dict(curobo_s=tot("curobo", "total_s"), wrs_s=tot("wrs", "total_s"), curobo_pairs=tot("curobo", "pairs")))
    summary["total"]["speedup"] = summary["total"]["wrs_s"] / summary["total"]["curobo_s"]
    OUT.mkdir(exist_ok=True)
    (OUT / f"grasp_speed_{args.model}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"TOTAL cuRobo {summary['total']['curobo_s']:.1f}s (+{curobo_setup:.1f}s one-time setup), WRS {summary['total']['wrs_s']:.1f}s "
          f"(estimated, {ik_backend}), speed-up x{summary['total']['speedup']:.0f}")


def full_compare(pid, Gs, cands, ins, curobo_stage, wrs_ok, wt, arm):
    """Both pipelines completely (goal side, then pick side on each method's own goal-feasible grasps)."""
    n, m = len(Gs), len(cands)
    # cuRobo
    c_goal, c_goal_ik, (cg_ik, cg_cc) = curobo_stage(ins)
    g1 = np.flatnonzero(c_goal)
    Pk = (cands[:, None] @ Gs[g1][None]).reshape(-1, 4, 4)
    c_pick, c_pick_ik, (cp_ik, cp_cc) = curobo_stage(Pk) if len(Pk) else (np.zeros(0, bool), np.zeros(0, bool), (0.0, 0.0))
    c_pairs = {(i // len(g1), int(g1[i % len(g1)])) for i in np.flatnonzero(c_pick)} if len(g1) else set()
    # WRS
    for k in wt:
        wt[k] = 0 if "calls" in k else 0.0
    w_goal = np.array([wrs_ok(P) for P in ins])
    w1 = np.flatnonzero(w_goal == 2)
    w_pairs = {(xi, int(g)) for xi in range(m) for g in w1 if wrs_ok(cands[xi] @ Gs[g]) == 2}
    arm.goto_given_conf(np.zeros(6))
    # agreement on the same questions: goal side over all grasps; pick side over the pairs both pipelines asked
    both_g = set(g1) & set(w1)
    asked = {(xi, int(g)) for xi in range(m) for g in both_g}
    r = dict(part=pid, grasps=n, candidates=m,
             curobo=dict(goal_ik=int(c_goal_ik.sum()), goal_ik_free=int(len(g1)), pairs=len(c_pairs),
                         ik_s=cg_ik + cp_ik, cc_s=cg_cc + cp_cc, total_s=cg_ik + cp_ik + cg_cc + cp_cc),
             wrs=dict(goal_ik=int((w_goal >= 1).sum()), goal_ik_free=int(len(w1)), pairs=len(w_pairs),
                      ik_s=wt["ik"], cc_s=wt["cc"], total_s=wt["ik"] + wt["cc"], ik_calls=wt["ik_calls"], cc_calls=wt["cc_calls"],
                      ms_per_ik=1e3 * wt["ik"] / max(wt["ik_calls"], 1), ms_per_cc=1e3 * wt["cc"] / max(wt["cc_calls"], 1)),
             goal=dict(both=len(both_g), only_curobo=len(set(g1) - set(w1)), only_wrs=len(set(w1) - set(g1)),
                       agree=float(np.mean(np.isin(np.arange(n), g1) == (w_goal == 2)))),
             pick_on_common_grasps=dict(asked=len(asked), both=len(asked & c_pairs & w_pairs),
                                        only_curobo=len((asked & c_pairs) - w_pairs), only_wrs=len((asked & w_pairs) - c_pairs),
                                        agree=1 - len((asked & c_pairs) ^ (asked & w_pairs)) / max(len(asked), 1)))
    r["speedup"] = r["wrs"]["total_s"] / max(r["curobo"]["total_s"], 1e-9)
    print(f"{pid:8s} grasps {n:5d} | cuRobo {r['curobo']['total_s']:5.2f}s (IK {r['curobo']['ik_s']:.2f}) goal {len(g1):4d} pairs {len(c_pairs):5d} | "
          f"WRS {r['wrs']['total_s']:6.1f}s (IK {r['wrs']['ik_s']:.1f}, {r['wrs']['ms_per_ik']:.1f} ms/IK) goal {len(w1):4d} pairs {len(w_pairs):5d} | "
          f"x{r['speedup']:.1f} | agree goal {r['goal']['agree']:.3f} pick {r['pick_on_common_grasps']['agree']:.3f}", flush=True)
    return r


if __name__ == "__main__":
    main()
