"""cuRobo validation of the plan found by lattice_joint.run (order + layout), with joint_search's Validator.

The search fixes the order and the table pose of every part; it only requires that SOME grasp is
usable at each step. Here:
1. per step k, the usable grasps: options (x_k, g) free under the installed set (parts before k)
   and not blocked by the parts still on the table (parts after k, at their poses) - the same
   checks as the search (Feasibility, sphere model);
2. one grasp per step by a small DP over the chain, minimising the joint-time lower bound of the
   whole execution (transit retreat(g_{k-1}) -> hover(x_k, g_k), fixed, transfer lift -> preins,
   return to HOME), as joint_search.problem times it;
3. joint_search.validate.Validator: every transit / transfer / return planned with cuRobo, executed
   on the straight joint line when that is free, else on the cuRobo path; then a mesh check (WRS)
   of the WHOLE executed motion, fixed motions included (descent, grasp, lift, insert, release,
   retreat);
4. if a step fails (cuRobo cannot plan a segment, or a motion collides), its grasp is excluded and
   the DP picks another one (at most --tries plans).

    python -X utf8 -m lattice_joint.validate [--model triangular_frame6] [--tries 5]
Input: lattice_joint/_output/<model>.json. Output: lattice_joint/_output/<model>_validated.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from types import SimpleNamespace

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy / Panda3D on Windows (c10.dll)

import numpy as np  # noqa: E402

from .geometry import log  # noqa: E402
from .run import OUT  # noqa: E402


def usable_grasps(inst, plan, k):
    """catalogue grasp ids usable at step k of the plan (installed = parts before k, table = parts after k)."""
    feas, geo, O = inst.feas, inst.geo, inst.O
    p, xi = plan[k]
    S = tuple(sorted(q for q, _ in plan[:k]))
    suf = tuple(sorted(geo.oi[(q, xq)] for q, xq in plan[k + 1:]))
    okp, okg = feas.base_ok(S, p)
    geo.resolve([(o, p, xi) for o in suf], [(o, p) for o in suf])
    gok = okg.copy()
    for o in suf:
        gok &= ~geo.bg[o, p]
    ids = geo.opt_at[p][xi]
    m = okp[ids] & gok[feas.opt_g[p][ids]]
    for o in suf:
        m &= ~geo.bp[o, p, xi]
    return [O["gl"][p][gi] for gi in feas.opt_g[p][ids][m]]


def choose_grasps(pb, steps, cands, banned):
    """DP over the chain: one grasp per step minimising the joint-time lower bound of the execution."""
    from joint_search.problem import HOME, t_acc
    cat = pb.cat
    prev = {None: (0.0, None)}                         # grasp of the previous step -> (cost, back)
    layers = []
    for k, (p, x) in enumerate(steps):
        cur = {}
        for g in cands[k]:
            if (k, g) in banned:
                continue
            e, v = cat["pick"][p][x][g], cat["goal"][p][g]
            own = pb.fix_pick[p, x, g] + pb.fix_goal[p, g] + float(t_acc(e["lift"], v["preins"]))
            best = (np.inf, None)
            for gq, (c, _) in prev.items():
                start = HOME if k == 0 else cat["goal"][steps[k - 1][0]][gq]["retreat"]
                c1 = c + float(t_acc(start, e["hover"])) + own
                if c1 < best[0]:
                    best = (c1, gq)
            cur[g] = best
        if not cur:
            return None, np.inf
        layers.append(cur)
        prev = cur
    p_last = steps[-1][0]
    total, g = min((c + float(t_acc(cat["goal"][p_last][g]["retreat"], HOME)), g) for g, (c, _) in prev.items())
    out = [g]
    for k in range(len(steps) - 1, 0, -1):
        out.append(layers[k][out[-1]][1])
    return out[::-1], total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default=None, help="assembly centre x,y,z (m), as run.py")
    ap.add_argument("--tries", type=int, default=5, help="plans (grasp choices) validated at most")
    ap.add_argument("--detax", action="store_true", help="time cuRobo paths after shortcut + corner blending")
    a = ap.parse_args(argv)
    from joint_search.validate import Validator
    from lazy_cost.refine import CuroboRefiner
    from .instance import Instance
    T0 = perf_counter()
    res = json.loads((OUT / f"{a.model}.json").read_text(encoding="utf-8"))
    inst = Instance.build(a.model, None if a.goal_pos is None else tuple(float(v) for v in a.goal_pos.split(",")))
    pb, O = inst.pb, inst.O
    plan = [(q, int(xi)) for q, xi, _ in res["layout"]]
    for (q, xi, key) in res["layout"]:
        assert O["poses"][q][xi] == key, (q, xi, key)
    steps = [(q, O["poses"][q][xi]) for q, xi in plan]
    cands = [usable_grasps(inst, plan, k) for k in range(len(plan))]
    log(f"plan from the search: cost {res['U']:.4f}; order {' -> '.join(q for q, _ in steps)}")
    for k, ((q, x), c) in enumerate(zip(steps, cands)):
        log(f"  step {k + 1} {q:8s} at {x}: {len(c)} usable grasps {c}")
    t1 = perf_counter()
    ref = CuroboRefiner(pb.chk, detax=a.detax)
    for p in pb.ids:
        ref.fit(p)
    log(f"cuRobo planners {perf_counter() - t1:.0f}s")
    val = Validator(pb, ref)
    banned, tries, result = set(), [], None
    for t in range(a.tries):
        gs, lb = choose_grasps(pb, steps, cands, banned)
        if gs is None:
            log("  no grasp choice left")
            break
        node = SimpleNamespace(suffix=[(q, x, g) for (q, x), g in zip(steps, gs)])
        t2 = perf_counter()
        ok, time, rows, crows = val.validate(node)
        failed = [r for r in rows if not np.isfinite(r["planned"])]
        bad = [r for r in crows if not r["ok"]]
        tries.append(dict(grasps=gs, lb=lb, ok=bool(ok), time=float(time), validate_s=perf_counter() - t2,
                          rows=rows, check_rows=crows))
        log(f"\n  try {t + 1}: grasps {gs}, joint-time lower bound {lb:.3f} s")
        for r in rows:
            log(f"    {r['kind']:8s} {str(r['part']):8s}  bound {r['lb']:.3f}  cuRobo {r['planned']:.3f}  executed {r['value']:.3f} ({r['how']})")
        if ok:
            log(f"  -> PASSED: every segment planned by cuRobo, mesh check of the whole execution collision-free;"
                f" executed time {time:.3f} s ({sum(r['how'] == 'planned' for r in rows)} segments on cuRobo paths)")
            result = tries[-1]
            break
        why = [("curobo", r["kind"], r["part"]) for r in failed] + [("collision", r["motion"], r["part"], r["collision"]) for r in bad]
        log(f"  -> FAILED: {why}")
        idx = {q: k for k, (q, _) in enumerate(steps)}
        for r in failed:
            if r["part"] is not None:
                banned.add((idx[r["part"]], gs[idx[r["part"]]]))
        for r in bad:
            if r["part"] is not None:
                banned.add((idx[r["part"]], gs[idx[r["part"]]]))
        if not failed and not any(r["part"] is not None for r in bad):
            break                                   # only the return to HOME failed: grasps cannot fix it
    log(f"\nvalidation {'PASSED' if result else 'FAILED'}; cuRobo planned {val.stats['planned']} segments"
        f" ({val.stats['plan_s']:.1f}s); total {perf_counter() - T0:.0f}s")
    f = OUT / f"{a.model}_validated.json"
    f.write_text(json.dumps(dict(model=a.model, search_cost=res["U"], order=[q for q, _ in steps],
                                 layout=[x for _, x in steps], usable_grasps=cands, passed=result is not None,
                                 tries=tries, stats=val.stats), indent=1, default=float), encoding="utf-8")
    log(f"-> {f}")
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
