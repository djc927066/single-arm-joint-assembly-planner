"""Why is the held part flagged against the scene when WRS says free?  (uses diag_segments.json)

1. TCP of the same joint values: cuRobo kinematics (our checker) vs WRS forward kinematics.
2. For transfer segments we block but WRS passes: where along the line (t) and against which
   world object our held-part spheres collide, and the WRS mesh clearance there.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"


def main():
    import torch
    from .problem import Problem
    from .wrs_check import MeshWorld
    pb = Problem("triangular_frame6")
    sc, chk = pb.scene, pb.chk
    W = MeshWorld(pb)
    G = sc.grasps()
    D = json.loads((OUT / "diag_segments.json").read_text(encoding="utf-8"))["segments"]

    # ---- 1. TCP agreement -------------------------------------------------------------------------
    qs = np.array([s["a"] for s in D[:200:10]] + [s["b"] for s in D[:200:10]])
    _, R, p = chk.kinematics(torch.as_tensor(qs, dtype=torch.float32, device="cuda"))
    R, p = R.cpu().numpy(), p.cpu().numpy()
    dpos, dang = [], []
    for i, q in enumerate(qs):
        W.arm.goto_given_conf(q)
        dpos.append(np.linalg.norm(W.arm.gl_tcp_pos - p[i]))
        c = (np.trace(W.arm.gl_tcp_rotmat.T @ R[i]) - 1) / 2
        dang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    print(f"TCP cuRobo vs WRS over {len(qs)} configurations: position max {1e3 * max(dpos):.2f} mm, "
          f"rotation max {max(dang):.3f} deg")

    # ---- 2. false-blocked transfer segments ----------------------------------------------------------
    def ours_free(s, world_parts):
        p_ = s["part"]
        placed = [(f"{q}@{c}", q, sc.pose[q][c]) for q, c in s["table"].items() if q in world_parts]
        chk.set_world(chk.world_dict([q for q in s["installed"] if q in world_parts], placed))
        jw = np.array([pb.jaw[p_][s["g"]]]) if pb.close_fingers else None
        A, B = np.array([s["a"]]), np.array([s["b"]])
        ok, = chk.batch_free([(A, B, 0, -1, jw)], held_bank=[(pb.held[p_], G[p_][s["g"]][0])], step=0.02)
        return bool(ok[0])

    fb = []
    for s in D:
        if s["kind"] != "transfer" or not s["wrs_free"]:
            continue
        if ours_free(s, set(sc.ids)):
            continue
        fb.append(s)
    print(f"\n{len(fb)} transfer segments blocked by us, free for WRS")
    culprit = Counter()
    for s in fb:
        if ours_free(s, set()):          # only table + base
            hits = []
            for q in set(s["installed"]) | set(s["table"]):
                if not ours_free(s, {q}):
                    hits.append(("installed " if q in s["installed"] else "on table ") + q)
            culprit[" + ".join(sorted(hits)) or "only together"] += 1
        else:
            culprit["table / assembly base"] += 1
    for k, v in culprit.most_common():
        print(f"  {v:4d}  {k}")

    # where along the line: endpoints alone?
    ends = Counter()
    for s in fb[:150]:
        p_ = s["part"]
        placed = [(f"{q}@{c}", q, sc.pose[q][c]) for q, c in s["table"].items()]
        chk.set_world(chk.world_dict(s["installed"], placed))
        jw = np.array([pb.jaw[p_][s["g"]]] * 2) if pb.close_fingers else None
        ok, = chk.batch_free([(np.array([s["a"], s["b"]]), None, 0, -1, jw)], held_bank=[(pb.held[p_], G[p_][s["g"]][0])])
        ends[("start (above pick) blocked" if not ok[0] else "start free") + ", " +
             ("end (above insert) blocked" if not ok[1] else "end free")] += 1
    print("\nendpoints of those segments (our check):")
    for k, v in ends.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
