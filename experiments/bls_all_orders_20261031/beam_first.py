"""Remove one beam from the assembled triangular frame: S, G, A of that step (2026-11-01).

For p in (beam_AB, beam_BC, beam_CA), removed first (= installed last, everything else installed):
  S  stability margin of what is left (R - {p}) and of the assembled frame (asmcore sweep, Chen S)
  G  grasps of p from its WRS pickle (all of them, no IK catalogue): gripper (WRS1 Panthera, triangle
     meshes) at p's goal pose, jaw closed on p (width = p's thickness along the opening axis); counted
     if it hits none of the other installed parts nor the base. Reference: base only.
  A  local translation cone of p in the assembled frame (asmcore assembly_directions, paper table)

    python -X utf8 -m experiments.bls_all_orders_20261031.beam_first
"""
from __future__ import annotations

import sys
from time import perf_counter

from curobo_bridge.runtime import prepare

prepare()

import numpy as np  # noqa: E402

from .run import log  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gone", default="", help="parts already removed, comma separated (default: none = assembled frame)")
    ap.add_argument("--arm", action="store_true",
                    help="also the arm: IK of the Panthera-HT at the grasp's goal pose, then arm links 2-5 + palm + "
                         "fingers (triangle meshes) free of the installed parts, the base and the table")
    a = ap.parse_args()
    gone = tuple(x for x in a.gone.split(",") if x)
    import trimesh
    import wrs.basis.trimesh as wtrm
    import wrs.modeling.collision_model as mcm
    import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg
    import scenes.triangular_frame6.build_model as bm
    from asmcore.model import AssemblyState
    from asmcore.motion.directions import DirectionConfig, assembly_directions
    from asmcore.mechanics.stability_sweep import DirectionalStabilityAnalyzer, StabilitySweepConfig
    from asmcore.planning.quality import score_assemblability
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
    from lazy_cost.scene import load_scene

    sc = load_scene("triangular_frame6")
    asm, _, _ = bm.build_assembly()
    seq = SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0))
    tf = {p.part_id: p.assembled_tf for p in asm.parts}
    fixed = {p.part_id: p.assembled_tf for p in asm.parts if p.fixed}

    def state(parts):
        return AssemblyState(dict(fixed, **{q: tf[q] for q in parts}))

    def S_of(parts):
        st = state(parts)
        return DirectionalStabilityAnalyzer(asm, st, seq.graph(st), config=StabilitySweepConfig()).analyze().sampled_minimum_load_factor

    cms = {}
    for q in ("base",) + tuple(sc.ids):
        c = mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(sc.meshes[q].vertices), faces=np.asarray(sc.meshes[q].faces)))
        c.pose = (sc.goal[q][:3, 3].copy(), sc.goal[q][:3, :3].copy())
        cms[q] = c
    gripper = pg.PantheraGripper()
    if a.arm:
        import json as _json
        from adapters.panthera import PantheraArmProvider
        from curobo_bridge.export_model import wrs_collision_parts
        from lazy_cost.scene import FRAME_DIR
        tb = _json.loads((FRAME_DIR / "task.json").read_text(encoding="utf-8"))["table"]
        tc, ts = np.asarray(tb["table_center"], float), np.asarray(tb["table_size"], float)
        tm = trimesh.creation.box(extents=ts)
        cms["table"] = mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(tm.vertices), faces=np.asarray(tm.faces)))
        cms["table"].pose = (tc.copy(), np.eye(3))
        arm = PantheraArmProvider(base_pos=tb["robot_base_pos"], base_rotmat=tb["robot_base_rotmat"]).arm()
        arm_links = [lnk for _, _, lnk in wrs_collision_parts(arm, include_base=False)]   # links 2-5, palm, fingers
    full = tuple(q for q in sc.ids if q not in gone)            # parts present now
    S_full = S_of(full)
    import json
    from .run import OUT
    rf = OUT / "sga_triangular_frame6_real.json"                 # robot-version G (IK catalogue) of the same step
    robot_G = {(tuple(sorted(r["installed"])), r["part"]): (r["G_joint"], r["G_joint_all"])
               for r in json.loads(rf.read_text(encoding="utf-8"))["edges"]} if rf.exists() else {}
    cands = [p for p in full if sc.stable(frozenset(full) - {p}) and sc.insertable(p, frozenset(full) - {p})]
    log(f"present: {', '.join(full)}; removed so far: {', '.join(gone) or '-'}; removable now: {', '.join(cands)}")
    log(f"current state: S = {S_full:.4f}")
    cases = []
    for p in cands:
        t0 = perf_counter()
        rest = tuple(q for q in full if q != p)
        S_rest = S_of(rest)
        d = assembly_directions(seq.graph(state(full)), (p,), config=DirectionConfig())
        A = score_assemblability(d)
        grasps = sc.grasps()[p]
        Gs = np.stack([G for G, _ in grasps])
        W = np.array([w for _, w in grasps])
        axis = Gs[:, :3, 1]                                      # Panthera opens along the TCP's y axis
        pts, _, _ = trimesh.proximity.closest_point(sc.meshes[p], np.r_[Gs[:, :3, 3] - axis * W[:, None] / 2,
                                                                         Gs[:, :3, 3] + axis * W[:, None] / 2])
        wc = np.linalg.norm(pts[len(W):] - pts[:len(W)], axis=1)  # jaw closed on the part
        others = [cms[q] for q in rest]
        n_base = n_inst = 0
        status = []                                              # per grasp: (world TCP pose, "base" / "installed" / "free")
        for (G, _), w in zip(grasps, wc):
            T = sc.goal[p] @ G
            gripper.grip_at_by_pose(T[:3, 3], T[:3, :3], float(np.clip(w, *gripper.jaw_range)))
            if gripper.is_mesh_collided([cms["base"]]):
                status.append((T, "base"))
                continue
            n_base += 1
            if not gripper.is_mesh_collided(others):
                n_inst += 1
                status.append((T, "free"))
            else:
                status.append((T, "installed"))
        n_ik = n_arm = 0
        if a.arm:                                                # arm on the gripper-free grasps
            obst = others + [cms["base"], cms["table"]]
            for i, ((T, s), w) in enumerate(zip(status, wc)):
                if s != "free":
                    continue
                q = arm.ik(T[:3, 3], T[:3, :3])
                if q is None:
                    status[i] = (T, "no_ik")
                    continue
                n_ik += 1
                arm.goto_given_conf(q)
                arm.change_jaw_width(float(np.clip(w, *gripper.jaw_range)))
                if any(lnk.cmodel.is_mcdwith(obst) for lnk in arm_links):
                    status[i] = (T, "arm_hit")
                else:
                    n_arm += 1
        cases.append(dict(part=p, rest=rest, S=S_rest, G=n_inst, G_base=n_base, n=len(grasps), A=A.score, status=status,
                          G_ik=n_ik, G_arm=n_arm))
        log(f"remove {p} next:\n"
            f"   S of what is left = {S_rest:.4f}   (current state {S_full:.4f})\n"
            f"   G = {n_inst} of {len(grasps)} pickle grasps free at the goal with the other {len(rest)} parts installed"
            f"   (base only: {n_base})\n"
            + (f"   with the arm: IK found for {n_ik} of those {n_inst}, arm + gripper free of parts/base/table: {n_arm}\n"
               if a.arm else "")
            + f""
            f"   robot G (IK catalogue, insertion/retreat motions) = {robot_G.get((tuple(sorted(rest)), p), '?')}\n"
            f"   A = {A.score} [{A.case}: {A.region}], removal direction {np.round(np.asarray(d.best_direction), 3).tolist()}"
            f"   ({perf_counter() - t0:.0f}s)")
    if not gone and not a.arm:
        plot(sc, cases, S_full)


def plot(sc, cases, S_full, n_red=150, stub=0.05):
    """Row 1: 3-D view, row 2: top view. Removed beam orange, other installed parts blue, base grey;
    a grasp = a stub from its TCP back along the approach axis: free green, blocked by the installed
    parts red (a random sample); blocked by the base not drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from .run import OUT
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    BASE, OTHER, MOVED, FREE, BLOCK = "#c3c2b7", "#2a78d6", "#eb6834", "#0ca30c", "#d03b3b"
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(18, 11.5), facecolor=SURF)
    centre = np.mean([sc.goal[q][:3, 3] for q in sc.ids], axis=0)

    def tris(q):
        m = sc.meshes[q]
        v = np.asarray(m.vertices) @ sc.goal[q][:3, :3].T + sc.goal[q][:3, 3]
        return v[np.asarray(m.faces)]

    for col, c in enumerate(cases):
        free = [T for T, s in c["status"] if s == "free"]
        red = [T for T, s in c["status"] if s == "installed"]
        red_show = [red[i] for i in rng.choice(len(red), min(n_red, len(red)), replace=False)] if red else []
        segs = lambda Ts: [np.stack([T[:3, 3], T[:3, 3] - stub * T[:3, 2]]) for T in Ts]
        title = (f"先拆 {c['part']}（= 最后装）\nS（剩余）= {c['S']:.4f}   A = {c['A']:g}\n"
                 f"G = {c['G']} / {c['n']}（只有底座时 {c['G_base']}）")
        # 3-D
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        ax.set_facecolor(SURF)
        ax.add_collection3d(Poly3DCollection(tris("base"), facecolor=BASE, edgecolor="none", alpha=0.35))
        for q in c["rest"]:
            ax.add_collection3d(Poly3DCollection(tris(q), facecolor=OTHER, edgecolor="none", alpha=0.45))
        ax.add_collection3d(Poly3DCollection(tris(c["part"]), facecolor=MOVED, edgecolor="none", alpha=0.95))
        ax.add_collection3d(Line3DCollection(segs(red_show), colors=BLOCK, linewidths=0.8, alpha=0.5))
        ax.add_collection3d(Line3DCollection(segs(free), colors=FREE, linewidths=0.9, alpha=0.8))
        r = 0.16
        ax.set_xlim(centre[0] - r, centre[0] + r)
        ax.set_ylim(centre[1] - r, centre[1] + r)
        ax.set_zlim(centre[2] - 0.08, centre[2] + 0.24)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-60)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=INK2, labelsize=7)
        # top view
        ax2 = fig.add_subplot(2, 3, col + 4)
        ax2.set_facecolor(SURF)
        for q, colr, a in [("base", BASE, 0.5)] + [(q, OTHER, 0.55) for q in c["rest"]] + [(c["part"], MOVED, 1.0)]:
            for t in tris(q):
                ax2.fill(t[:, 0], t[:, 1], color=colr, alpha=a, lw=0)
        for Ts, colr, a in ((red_show, BLOCK, 0.45), (free, FREE, 0.8)):
            for s in segs(Ts):
                ax2.plot(s[:, 0], s[:, 1], color=colr, lw=0.8, alpha=a)
            if Ts:
                P = np.array([T[:3, 3] for T in Ts])
                ax2.scatter(P[:, 0], P[:, 1], s=6, color=colr, alpha=a, lw=0)
        ax2.set_aspect("equal")
        ax2.set_xlim(centre[0] - r, centre[0] + r)
        ax2.set_ylim(centre[1] - r, centre[1] + r)
        ax2.set_title("俯视图", fontsize=10, color=INK2, loc="left")
        ax2.tick_params(colors=INK2, labelsize=7)
        for s in ax2.spines.values():
            s.set_color("#d9d8d3")
    handles = [Line2D([], [], color=MOVED, lw=8, label="要拆的横梁"),
               Line2D([], [], color=OTHER, lw=8, alpha=0.6, label="其余已装零件"),
               Line2D([], [], color=BASE, lw=8, alpha=0.6, label="底座"),
               Line2D([], [], color=FREE, lw=2, label="可用抓取（夹爪不碰已装零件和底座）"),
               Line2D([], [], color=BLOCK, lw=2, label=f"被已装零件挡住（随机显示至多 {n_red} 个）")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.suptitle(f"三角框架：从装好状态先拆一根横梁（装好时 S = {S_full:.4f}）。"
                 f"抓取画成从抓取点沿接近方向后退 {stub * 100:.0f} cm 的短线；撞底座的抓取不画",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    f = OUT / "beam_first_sga.png"
    fig.savefig(f, dpi=120, facecolor=SURF)
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
