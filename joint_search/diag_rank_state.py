"""Why does the WRS RRT reference rank the candidates of one search state so differently?

    & $py -X utf8 -m joint_search.diag_rank_state [--state 4]

Reads joint_search/_output/rank_rack10/state_<i>.json (the RRT reference) and, for every segment
it planned, WITHOUT running RRT:
  1. where the straight joint line collides under the WRS mesh check (first / last colliding t,
     what it hits);
  2. a simple hand-made detour: lift the TCP straight up by h at the start and/or the end
     (WRS IK seeded with the endpoint), path a -> up(a) [-> up(b)] -> b, corner-blended, mesh-checked,
     TOPP-timed. h in 3..20 cm. Its time is a real collision-free time (upper bound of the optimum).
Then truth2 = min(RRT, detour) per segment and the candidate ranking is re-scored.
Output: joint_search/_output/rank_rack10/diag_state_<i>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "_output" / "rank_rack10"
HS = (0.03, 0.05, 0.08, 0.12, 0.20)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=int, default=4)
    args = ap.parse_args(argv)
    from lazy_cost.detour import blend
    from lazy_cost.timing import topp
    from .problem import HOME, Problem, t_acc
    from .wrs_check import MeshWorld, densify

    R = json.loads((OUT / f"state_{args.state}.json").read_text(encoding="utf-8"))
    D = pickle.loads((OUT / "states.pkl").read_bytes())
    s = D["states"][args.state]
    node, p = s["node"], s["part"]
    R1 = sorted(node.R - {p})
    pb = Problem("rack10")
    sc, cat, G = pb.scene, pb.cat, pb.scene.grasps()
    W = MeshWorld(pb)
    arm = W.arm
    tbl = {q: sc.pose[q][k] for q, k in node.table.items()}
    h, hx, hg = node.head

    def where(a, b):
        P = densify([a, b])
        hits, what = [], set()
        for i, q in enumerate(P):
            ok, why = W.free([q, q])
            if not ok:
                hits.append(i / (len(P) - 1)); what.add(why)
        return (min(hits), max(hits), sorted(what)) if hits else (None, None, [])

    def up(q, dz):
        """TCP raised by dz, same orientation: damped least squares on the WRS FK / Jacobian, seeded at q."""
        q = np.asarray(q, float).copy()
        pos0, rot0 = arm.fk(jnt_values=q)[:2]
        tgt = pos0 + np.array([0, 0, dz])
        for _ in range(50):
            pos, rot = arm.fk(jnt_values=q)[:2]
            dr = 0.5 * (np.cross(rot[:, 0], rot0[:, 0]) + np.cross(rot[:, 1], rot0[:, 1]) + np.cross(rot[:, 2], rot0[:, 2]))
            e = np.r_[tgt - pos, dr]
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                return q if arm.are_jnts_in_ranges(jnt_values=q) else None
            J = np.zeros((6, 6))
            for j in range(6):
                dq = np.zeros(6); dq[j] = 1e-5
                pj, rj = arm.fk(jnt_values=q + dq)[:2]
                w = 0.5 * (np.cross(rot[:, 0], rj[:, 0]) + np.cross(rot[:, 1], rj[:, 1]) + np.cross(rot[:, 2], rj[:, 2]))
                J[:, j] = np.r_[pj - pos, w] / 1e-5
            q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        return None

    def detour(a, b):
        best, how = np.inf, None
        for dz in HS:
            ua, ub = up(a, dz), up(b, dz)
            for name, P in (("lift_start", [a, ua, b]), ("lift_end", [a, ub, b]), ("lift_both", [a, ua, ub, b])):
                if any(x is None for x in P):
                    continue
                P = np.asarray(P)
                for path in (blend(P), P):
                    if W.free(path)[0]:
                        t = float(topp(path)[1][-1])
                        if t < best:
                            best, how = t, f"{name} {int(dz * 100)}cm"
                        break
        return best, how

    out, cache = [], {}
    for sg in R["segments"]:
        x, g, kind = sg["x"], sg["g"], sg["kind"]
        if kind == "transfer":
            a, b = cat["pick"][p][x][g]["lift"], cat["goal"][p][g]["preins"]
            W.set(R1, tbl, (p, G[p][g][0], a), jaw=pb.jaw[p][g] if pb.close_fingers else None)
            key = ("transfer", x, g)
        else:
            a, b = cat["goal"][p][g]["retreat"], cat["pick"][h][hx][hg]["hover"]
            W.set(sorted(node.R), tbl)
            key = ("connection", g)
        if key not in cache:
            if sg["wrs"] == "straight":
                cache[key] = dict(t_in=None, t_out=None, hits=[], det=sg["lb"], det_how="straight")
            else:
                t_in, t_out, hits = where(a, b)
                det, how = detour(a, b)
                cache[key] = dict(t_in=t_in, t_out=t_out, hits=hits, det=det, det_how=how)
            c = cache[key]
            print(f"{kind:10s} {x:20s} g{g:<5d} lb {sg['lb']:.3f} RRT {sg['wrs_time'] or float('nan'):.3f} "
                  f"detour {c['det']:.3f} ({c['det_how']})  hit t {c['t_in']}..{c['t_out']} {c['hits']}", flush=True)
        out.append(dict(sg, **cache[key]))
    (OUT / f"diag_state_{args.state}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("->", OUT / f"diag_state_{args.state}.json")


if __name__ == "__main__":
    main()
