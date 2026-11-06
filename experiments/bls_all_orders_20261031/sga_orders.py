"""S, G, A on every step of the triangular frame and the orders they choose (2026-11-01).

Per step (install p after the set S = remove p from S + {p}):
  S_after  stability margin of S + {p} (Chen: the state after installing p)
  S_left   stability margin of S       (what is left after removing p)
  G        pickle grasps of p at its goal: gripper (closed on p) free of the base and of S, Panthera-HT
           IK found, arm links 2-5 + palm + fingers free of S, the base and the table (triangle meshes)
  A        Chen assemblability of p in the contact graph of S + {p}
Orders:
  greedy, S_left : from the assembled frame remove the removable part with the largest S_left * G * A
  greedy, S_after: same with S_after (= the same for all candidates of one step, so G * A decides)
  Chen Eq.(2)    : over all stable orders, max xi = min S_after * min G * min A (no support: single arm)

    python -X utf8 -m experiments.bls_all_orders_20261031.sga_orders
"""
from __future__ import annotations

import json
import sys
from time import perf_counter

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .run import OUT, log  # noqa: E402


def build_asm(sc, model):
    """asmcore assembly + fixed poses for the model (frame: build_model.py; benchmarks: as
    scenes.benchmarks.common.verify_equilibrium), table and robot base pose."""
    if model == "triangular_frame6":
        import scenes.triangular_frame6.build_model as bm
        from lazy_cost.scene import FRAME_DIR
        asm, _, _ = bm.build_assembly()
        tb = json.loads((FRAME_DIR / "task.json").read_text(encoding="utf-8"))["table"]
        return asm, tb["table_center"], tb["table_size"], tb["robot_base_pos"], tb["robot_base_rotmat"]
    from asmcore.model import Assembly, Part
    from scenes.benchmarks.common import TABLE_CENTER, TABLE_SIZE
    from tasks.bridge import box_mesh, pose
    m = sc.model
    parts = [Part("table", box_mesh(*TABLE_SIZE), pose(*TABLE_CENTER), mass_kg=40.0, com_local_m=np.zeros(3),
                  friction=0.5, fixed=True)]
    for p in m.all:
        c = m.local_cells[p.pid]
        vol = np.prod(c[:, 1::2] - c[:, 0::2], axis=1)
        com = ((c[:, 0::2] + c[:, 1::2]) / 2 * vol[:, None]).sum(0) / vol.sum()
        parts.append(Part(p.pid, m.mesh(p.pid), m.goal[p.pid], mass_kg=m.mass(p.pid), com_local_m=com,
                          friction=0.5, fixed=p.pid == "base"))
    return Assembly(tuple(parts)), TABLE_CENTER.tolist(), TABLE_SIZE.tolist(), [0.0, 0.0, 0.0], np.eye(3).tolist()


def stable_orders(sc):
    if hasattr(sc, "orders"):
        return [tuple(o) for o in sc.orders]
    out = []

    def rec(R, suf):                      # backward: remove a part that can be installed last
        if not R:
            out.append(tuple(suf))
            return
        for p in sorted(R):
            if sc.step_ok(R - {p}, p):
                rec(R - {p}, [p] + suf)
    rec(frozenset(sc.ids), [])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    a = ap.parse_args()
    import trimesh
    import wrs.basis.trimesh as wtrm
    import wrs.modeling.collision_model as mcm
    import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg
    from adapters.panthera import PantheraArmProvider
    from asmcore.model import AssemblyState
    from asmcore.motion.directions import DirectionConfig, assembly_directions
    from asmcore.mechanics.stability_sweep import DirectionalStabilityAnalyzer, StabilitySweepConfig
    from asmcore.planning.quality import score_assemblability
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
    from curobo_bridge.export_model import wrs_collision_parts
    from lazy_cost.scene import load_scene

    sc = load_scene(a.model)
    asm, t_center, t_size, r_pos, r_rot = build_asm(sc, a.model)
    seq = SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0))
    tf = {p.part_id: p.assembled_tf for p in asm.parts}
    fixed = {p.part_id: p.assembled_tf for p in asm.parts if p.fixed}
    state = lambda parts: AssemblyState(dict(fixed, **{q: tf[q] for q in parts}))
    S_cache = {}

    def S_of(parts):
        key = frozenset(parts)
        if key not in S_cache:
            if not key:
                S_cache[key] = float("nan")
            else:
                st = state(key)
                S_cache[key] = DirectionalStabilityAnalyzer(asm, st, seq.graph(st), config=StabilitySweepConfig()).analyze().sampled_minimum_load_factor
        return S_cache[key]

    def cm(mesh, T):
        c = mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)))
        c.pose = (T[:3, 3].copy(), T[:3, :3].copy())
        return c

    cms = {q: cm(sc.meshes[q], sc.goal[q]) for q in ("base",) + tuple(sc.ids)}
    T_table = np.eye(4)
    T_table[:3, 3] = t_center
    cms["table"] = cm(trimesh.creation.box(extents=t_size), T_table)
    gripper = pg.PantheraGripper()
    arm = PantheraArmProvider(base_pos=r_pos, base_rotmat=r_rot).arm()
    arm_links = [lnk for _, _, lnk in wrs_collision_parts(arm, include_base=False)]
    grasps = {}
    for p in sc.ids:                                   # pickle grasps, jaw closed on the part
        gs = sc.grasps()[p]
        Gs = np.stack([G for G, _ in gs])
        W = np.array([w for _, w in gs])
        ax = Gs[:, :3, 1]
        pts, _, _ = trimesh.proximity.closest_point(sc.meshes[p], np.r_[Gs[:, :3, 3] - ax * W[:, None] / 2, Gs[:, :3, 3] + ax * W[:, None] / 2])
        grasps[p] = list(zip(Gs, np.clip(np.linalg.norm(pts[len(W):] - pts[:len(W)], axis=1), *gripper.jaw_range)))
    ik_cache = {}                                      # (p, grasp) -> joint values or None (independent of the step)

    def G_of(S, p):
        installed = [cms[q] for q in S]
        n = 0
        for i, (G, w) in enumerate(grasps[p]):
            T = sc.goal[p] @ G
            gripper.grip_at_by_pose(T[:3, 3], T[:3, :3], float(w))
            if gripper.is_mesh_collided([cms["base"]] + installed):
                continue
            if (p, i) not in ik_cache:
                ik_cache[p, i] = arm.ik(T[:3, 3], T[:3, :3])
            q = ik_cache[p, i]
            if q is None:
                continue
            arm.goto_given_conf(q)
            arm.change_jaw_width(float(w))
            if any(l.cmodel.is_mcdwith(installed + [cms["base"], cms["table"]]) for l in arm_links):
                continue
            n += 1
        return n

    orders = stable_orders(sc)
    log(f"{a.model}: {len(sc.ids)} parts, {len(orders)} stable orders")
    edges = sorted({(tuple(sorted(o[:k])), o[k]) for o in orders for k in range(len(o))}, key=lambda e: (len(e[0]), e))
    E, t0 = {}, perf_counter()
    for S, p in edges:
        d = assembly_directions(seq.graph(state(S + (p,))), (p,), config=DirectionConfig())
        E[S, p] = dict(S_after=S_of(S + (p,)), S_left=S_of(S), G=G_of(S, p), A=float(score_assemblability(d).score))
        log(f"  {'+'.join(x.split('_')[1] for x in S) or '-':14s} + {p:8s} S_after {E[S, p]['S_after']:.4f}  "
            f"S_left {E[S, p]['S_left']:.4f}  G {E[S, p]['G']:4d}  A {E[S, p]['A']:g}   ({perf_counter() - t0:.0f}s)")

    def xi(order):
        steps = [E[tuple(sorted(order[:k])), order[k]] for k in range(len(order))]
        return min(s["S_after"] for s in steps) * min(s["G"] for s in steps) * min(s["A"] for s in steps)

    def greedy(key):
        R, rev = tuple(sorted(sc.ids)), []
        while R:
            cands = [p for p in R if (tuple(q for q in R if q != p), p) in E]
            sc_ = {p: key(E[tuple(q for q in R if q != p), p]) for p in cands}
            p = max(sc_, key=sc_.get)
            log(f"    from {{{', '.join(x.split('_')[1] for x in R)}}}: " + ", ".join(f"{c} {v:.3f}" for c, v in sorted(sc_.items(), key=lambda t: -t[1]))
                + f"  -> remove {p}")
            rev.append(p)
            R = tuple(q for q in R if q != p)
        return tuple(rev[::-1])

    res = {}
    for name, key in (("greedy, S_left", lambda s: s["S_left"] * s["G"] * s["A"] if s["S_left"] == s["S_left"] else s["G"] * s["A"]),
                      ("greedy, S_after", lambda s: s["S_after"] * s["G"] * s["A"])):
        log(f"\n== {name}")
        o = greedy(key)
        res[name] = o
        log(f"  order: {' -> '.join(o)}   xi = {xi(o):.4f}")
    ranked = sorted(orders, key=xi, reverse=True)
    best = [o for o in ranked if abs(xi(o) - xi(ranked[0])) < 1e-12]
    log(f"\n== Chen Eq.(2) over all {len(orders)} orders: best xi = {xi(ranked[0]):.4f}, reached by {len(best)} order(s):")
    for o in best:
        log(f"  {' -> '.join(o)}")
    log("  top xi values: " + ", ".join(f"{xi(o):.4f}" for o in ranked[:8]))
    for name, o in res.items():
        log(f"  {name}: xi {xi(o):.4f}, rank {1 + sum(xi(x) > xi(o) + 1e-12 for x in orders)} of {len(orders)}")
    (OUT / f"sga_orders_{a.model}.json").write_text(json.dumps(dict(
        edges=[dict(installed=list(S), part=p, **v) for (S, p), v in E.items()],
        greedy={k: list(v) for k, v in res.items()}, best=[list(o) for o in best], best_xi=xi(ranked[0])), indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
