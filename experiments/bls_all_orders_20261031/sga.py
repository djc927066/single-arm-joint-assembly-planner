"""Chen et al. (T-ASE 2021) S, G, A on every edge of the joint search's order lattice (2026-11-01).

An edge (S, p) = install p after the set S, i.e. the state Pi = base + S + p at their goals.
The three paper qualities of that state, computed exactly as asmcore's quality search
(planning/quality_search.py `_AssemblyQualityEvaluator.expand`):
  S  stability: nominal equilibrium (gravity, friction, no support) and, if balanced, the minimum
     sampled capped radial load factor over 300 disturbance directions (DirectionalStabilityAnalyzer);
     0 if the state cannot balance (single arm: no second hand, so S = 0 is infeasible)
  G  graspability: catalogue grasps of p that are force-closed (6D LP on the finger pads) and
     whose gripper at p's goal hits neither p nor the installed parts (GraspabilityAnalyzer,
     Panthera gripper through adapters/panthera_gripper.PantheraGripperPort)
  A  assemblability: local pure-translation cone of p in the contact graph of Pi
     (assembly_directions), scored by the paper's table (score_assemblability, asp_old 10/9/3/2/1/0)
Checks against what the joint search already uses on the same edge:
  S > 0  vs  scene.stable(S + p) (the step_ok equilibrium)
  A > 0  vs  scene.insertable(p, S) (straight sweep along the removal axis)
  G      vs  the joint catalogue: goal grasps of p free with S installed (GPU sphere model: insertion,
           pre-insertion holding p, insertion line, release line, retreat), = ls.base_ok(S, p)[1]

    python -X utf8 -m experiments.bls_all_orders_20261031.sga --model triangular_frame6
"""
from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

import numpy as np

from .lazy import setup
from .run import OUT, log


def real_grasps(pb, O, asm, gripper):
    """The joint search's goal grasps of every part (O["gl"]: WRS antipodal, Panthera; TCP pose in the
    part frame + jaw width) as asmcore GraspRecords; the two nominal contacts (TCP -+ opening axis x
    width / 2) are snapped onto the part mesh. Checks first that asmcore's part frame = the scene's."""
    import trimesh
    from asmcore.robotics.grasp_model import GraspRecord
    parts = {p.part_id: p for p in asm.parts}
    od = np.asarray(gripper.open_dir, float)
    out = {}
    for p in O["ids"]:
        sm, ap = pb.scene.meshes[p], parts[p]
        if not np.allclose(np.sort(np.asarray(sm.bounds), 0), np.sort(np.c_[ap.geometry.vertices.min(0), ap.geometry.vertices.max(0)].T, 0), atol=1e-6) \
                or not np.allclose(pb.scene.goal[p][:3, :3], ap.assembled_tf[:3, :3], atol=1e-6):   # site may be shifted
            raise SystemExit(f"{p}: asmcore part frame differs from the scene's")
        recs = []
        for g in O["gl"][p]:
            G, w = pb.scene.grasps()[p][g]
            c, axis = G[:3, 3], G[:3, :3] @ od
            pts, _, _ = trimesh.proximity.closest_point(sm, np.array([c - axis * w / 2, c + axis * w / 2]))
            wc = float(np.linalg.norm(pts[1] - pts[0]))     # jaw closed on the part (catalogue w = open width)
            pre = G.copy()
            pre[:3, 3] -= G[:3, 2] * 0.06
            recs.append(GraspRecord(G, pre, wc, provenance=dict(jaw_width=wc, contacts_local_m=pts, open_width=w,
                                                                generator="joint_catalogue", catalogue_index=int(g))))
        out[p] = recs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sweep-backend", default="cuda", choices=("cuda", "numpy", "highs"))
    ap.add_argument("--real-grasps", action="store_true",
                    help="paper G on the joint search's own Panthera goal grasps instead of asmcore's box side pinches")
    a = ap.parse_args()
    if a.model != "triangular_frame6":
        raise SystemExit("only the triangular frame has an asmcore assembly model (build_model.py) so far")
    pb, O, geo, ls, orders, *_ = setup(a.model, a.goal_pos, a.k)
    import scenes.triangular_frame6.build_model as bm
    from asmcore.model import AssemblyState
    from asmcore.motion.directions import DirectionConfig, assembly_directions
    from asmcore.mechanics.stability_sweep import DirectionalStabilityAnalyzer, StabilitySweepConfig
    from asmcore.planning.quality import score_assemblability
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator, _balanced
    from asmcore.robotics import GraspabilityAnalyzer
    from adapters.panthera_gripper import PantheraGripperPort

    asm, _, _ = bm.build_assembly()
    seq = SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0))
    tf = {p.part_id: p.assembled_tf for p in asm.parts}
    fixed = {p.part_id: p.assembled_tf for p in asm.parts if p.fixed}
    port = PantheraGripperPort()
    if a.real_grasps:
        from asmcore.robotics import GraspabilityConfig
        recs = real_grasps(pb, O, asm, port)
        grasp = GraspabilityAnalyzer(asm, port, grasps=recs,
                                     config=GraspabilityConfig(max_grasps=max(len(v) for v in recs.values())))
        for p in O["ids"]:                      # catalogue screening (contacts, pad seating, force closure, self-collision)
            cat = grasp._catalogue(p)
            log(f"{p}: {len(recs[p])} joint goal grasps -> {len(cat)} seated on the pads, "
                f"{sum(r[3] == 'force_closed' for r in cat)} force-closed without self-collision")
    else:
        grasp = GraspabilityAnalyzer(asm, port)
    dcfg, scfg = DirectionConfig(), StabilitySweepConfig(backend=a.sweep_backend)
    sc = pb.scene
    edges = sorted({(tuple(sorted(o[:k])), o[k]) for o in orders for k in range(len(o))}, key=lambda e: (len(e[0]), e))
    rows, S_cache = [], {}
    for S, p in edges:
        t0 = perf_counter()
        state = AssemblyState(dict(fixed, **{q: tf[q] for q in S + (p,)}))
        graph = seq.graph(state)
        key = tuple(sorted(S + (p,)))
        if key not in S_cache:
            nominal = seq.equilibrium(state, ())
            if _balanced(nominal):
                sw = DirectionalStabilityAnalyzer(asm, state, graph, config=scfg).analyze()
                S_cache[key] = (sw.sampled_minimum_load_factor, sw.status)
            else:
                S_cache[key] = (0.0 if nominal.status == "infeasible" else None, nominal.status)
        s_val, s_status = S_cache[key]
        d = assembly_directions(graph, (p,), config=dcfg)
        A = score_assemblability(d)
        g = grasp.analyze(state, p)
        okp, okg = ls.base_ok(S, p)
        r = dict(installed=list(S), part=p, S=s_val, S_status=s_status, A=A.score, A_case=A.case, A_region=A.region,
                 A_dir=None if d.best_direction is None else np.round(-np.asarray(d.best_direction), 3).tolist(),
                 G=g.count, G_known=g.diagnostics["known_count"], G_catalogue=g.catalogue_size, G_status=g.status,
                 G_joint=int(okg.sum()), G_joint_all=len(okg),
                 stable_scene=bool(sc.stable(frozenset(S) | {p})), insertable_scene=bool(sc.insertable(p, frozenset(S))),
                 seconds=perf_counter() - t0)
        if a.real_grasps:
            cat = grasp._catalogue(p)
            P = {int(cat[i][0]["provenance"]["catalogue_index"]) for i in g.accepted_ids}
            J = {int(O["gl"][p][i]) for i in np.nonzero(okg)[0]}
            r.update(both=len(P & J), paper_only=len(P - J), joint_only=len(J - P))
        rows.append(r)
        log(f"{'+'.join(x.split('_')[1] for x in S) or '-':16s} + {p:8s}  S {('%.3f' % s_val) if s_val is not None else 'unk':>6s}"
            f" (scene stable {r['stable_scene']})  A {r['A']} [{r['A_case']}/{r['A_region']}] insert dir {r['A_dir']}"
            f" (scene insertable {r['insertable_scene']})  G {r['G']}/{r['G_catalogue']} ({r['G_status']})"
            f"  joint G {r['G_joint']}/{r['G_joint_all']}"
            + (f"  both {r['both']} paper-only {r['paper_only']} joint-only {r['joint_only']}" if a.real_grasps else "")
            + f"  ({r['seconds']:.1f}s)")
    agree_S = sum((r["S"] is not None and r["S"] > 1e-8) == r["stable_scene"] for r in rows)
    agree_A = sum((r["A"] is not None and r["A"] > 0) == r["insertable_scene"] for r in rows)
    gp = np.array([r["G"] if r["G"] is not None else np.nan for r in rows], float)
    gj = np.array([r["G_joint"] for r in rows], float)
    m = ~np.isnan(gp)
    corr = float(np.corrcoef(gp[m], gj[m])[0, 1]) if m.sum() > 2 and gp[m].std() > 0 and gj[m].std() > 0 else None
    summary = dict(edges=len(rows), S_agrees_with_scene=agree_S, A_agrees_with_insertable=agree_A,
                   G_paper_vs_joint_corr=corr)
    log(json.dumps(summary, indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / f"sga_{a.model}{'_real' if a.real_grasps else ''}.json").write_text(json.dumps(dict(summary=summary, edges=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
