"""Output the chosen layout of a joint_search run and check it with WRS (no cuRobo).

    & $py -X utf8 -m joint_search.wrs_check [joint_search/_output/frame_*.json]   (default: newest)

For every transit / transfer / return segment of the chosen plan, in its forward world (installed
parts at their goals, parts not yet assembled at their table poses, the table and the assembly base):
1. the straight joint line is checked against the meshes (WRS / ODE mesh-mesh, every 0.02 rad:
   arm links vs scene, held part vs scene, held part vs arm links 2-5, arm self-collision);
   free -> executed as the straight line, time = its lower bound;
2. otherwise WRS RRT-Connect (ext_dist 0.05, 150 smoothing iterations, up to 6 seeds), the path is
   re-checked with the same mesh test (RRT checks only its nodes), greedily shortcut with it, its
   corners blended if still free, and timed with lazy_cost.timing.topp.
The short fixed motions (hover -> prepick -> pick -> lift, preins -> ins -> retreat) are taken
from the search; they touch the part being picked / inserted and are not mesh-checked here.

--rrt-all: plan EVERY free segment with RRT-Connect (as SEALP's L3), straight or not, and report
the raw RRT path time (WRS smoothing only) and the time after the greedy shortcut + corner blend.

Output: joint_search/_output/<run>_layout.json (order, initial pose of every part, grasp, per
segment result) and <run>_layout.png (top view of the table).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"
STEP = 0.02
ARM_LINKS = ("link2", "link3", "link4", "link5")
RRT_SEEDS = 6
RRT_EXT = 0.05          # RRT extension step (rad); WRS checks collisions only at the tree nodes


def wrs_cm(mesh, pose=None, name=""):
    import wrs.basis.trimesh as wtrm
    import wrs.modeling.collision_model as mcm
    cm = mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces)), name=name)
    if pose is not None:
        cm.pose = (np.asarray(pose)[:3, 3].copy(), np.asarray(pose)[:3, :3].copy())
    return cm


def densify(P, step=STEP):
    P = np.asarray(P, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        out += [a + t * (b - a) for t in np.linspace(0, 1, n + 1)[1:]]
    return np.asarray(out)


class MeshWorld:
    def __init__(self, pb):
        import wrs.modeling.collision_model as mcm
        from curobo_bridge.planner import PantheraPlanner
        self.pb, sc = pb, pb.scene
        self.planner = PantheraPlanner(rrt_ext_dist=RRT_EXT, rrt_max_time=20.0)   # WRS arm, mesh parts, self pairs, RRT
        self.arm = self.planner.arm
        self.parts = self.planner.parts
        T = np.eye(4)
        T[:3, 3] = sc.table_center
        self.table = mcm.gen_box(xyz_lengths=np.asarray(sc.table_size, float), pos=T[:3, 3], rotmat=T[:3, :3])
        self.base = wrs_cm(sc.meshes["base"], sc.goal["base"], "base")
        self.held = None
        self.obs = []
        self._cm, self._held_cm = {}, {}     # one WRS model per part, re-posed (building them per call leaks memory)

    def _pose(self, cache, p, T):
        if p not in cache:
            cache[p] = wrs_cm(self.pb.scene.meshes[p], None, p)
        cm = cache[p]
        cm.pose = (np.asarray(T)[:3, 3].copy(), np.asarray(T)[:3, :3].copy())
        return cm

    def set(self, installed, table, held=None, jaw=None, held_skip=()):
        """installed: part ids at their goals; table: {pid: 4x4}; held: (pid, G, q_grasp) or None;
        jaw: gripper opening (m) for the following checks / RRT (None = fully open, 0.085);
        held_skip: obstacles the HELD PART is not checked against ("table": it rests on the table
        when lifted, "installed": base + installed parts, it reaches its seat when inserted; the robot
        is still checked)."""
        sc = self.pb.scene
        if self.held is not None:
            self.arm.release(self.held)
            self.held = None
        self.arm.change_jaw_width(0.085 if jaw is None else float(jaw))
        inst = [self._pose(self._cm, p, sc.goal[p]) for p in installed]
        self.obs = [self.table, self.base] + inst + [self._pose(self._cm, p, T) for p, T in table.items()]
        seat = [self.base] + inst if "installed" in held_skip else []
        self.held_obs = [o for o in self.obs if not (("table" in held_skip and o is self.table) or any(o is c for c in seat))]
        self.planner._wobs = self.obs
        if held is not None:
            p, G, q = held
            self.arm.goto_given_conf(np.asarray(q, float))
            tcp = np.eye(4)
            tcp[:3, :3], tcp[:3, 3] = self.arm.gl_tcp_rotmat, self.arm.gl_tcp_pos
            self.held = self._pose(self._held_cm, p, tcp @ np.linalg.inv(G))
            self.held_Ginv = np.linalg.inv(G)
            self.arm.hold(self.held)

    def place_held(self):
        """Put the held part at TCP @ inv(G) for the arm's current configuration. The WRS Panthera
        does NOT move a held object with goto_given_conf (checked: 0.0 mm after a 0.4 m move), so
        without this the held part stays where it was grasped."""
        if self.held is not None:
            tcp = np.eye(4)
            tcp[:3, :3], tcp[:3, 3] = self.arm.gl_tcp_rotmat, self.arm.gl_tcp_pos
            T = tcp @ self.held_Ginv
            self.held.pose = (T[:3, 3].copy(), T[:3, :3].copy())

    def _which(self, cm, obs):
        """Name of the first obstacle that cm hits."""
        return next(("table" if o is self.table else getattr(o, "name", "?") for o in obs if cm.is_mcdwith([o])), "?")

    def free(self, P):
        """Mesh test of a joint path (densified) -> (ok, first colliding state description)."""
        arm, parts, obs = self.arm, self.parts, self.obs
        arm_links = [parts[n].cmodel for n in ARM_LINKS if n in parts]
        for q in densify(P):
            arm.goto_given_conf(q)
            self.place_held()
            for n, l in parts.items():
                if l.cmodel.is_mcdwith(obs):
                    return False, f"{n} vs {self._which(l.cmodel, obs)}"
            if self.held is not None:
                if self.held.is_mcdwith(self.held_obs):
                    return False, f"held part vs {self._which(self.held, self.held_obs)}"
                if any(self.held.is_mcdwith([c]) for c in arm_links):
                    return False, "held part vs arm"
            for a, b in self.planner.self_pairs:
                if a in parts and b in parts and parts[a].cmodel.is_mcdwith([parts[b].cmodel]):
                    return False, f"self {a}-{b}"
        return True, None

    def shortcut(self, P):
        P = np.asarray(P, float)
        out, i = [P[0]], 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1 and not self.free([P[i], P[j]])[0]:
                j -= 1
            out.append(P[j])
            i = j
        return np.asarray(out)


def topview(pb, steps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from scipy.spatial import ConvexHull
    sc = pb.scene
    fig, ax = plt.subplots(figsize=(7, 7))
    c, s = sc.table_center, sc.table_size
    ax.add_patch(Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], fc="#f2efe8", ec="#999"))

    def foot(pid, T, **kw):
        v = np.asarray(sc.meshes[pid].vertices) @ T[:3, :3].T + T[:3, 3]
        h = ConvexHull(v[:, :2])
        ax.add_patch(Polygon(v[h.vertices, :2], **kw))
        return v[:, :2].mean(0)

    foot("base", sc.goal["base"], fc="#ccc", ec="#666", alpha=0.6)
    for p in sc.ids:
        foot(p, sc.goal[p], fc="none", ec="#888", ls="--")
    cmap = plt.get_cmap("viridis")
    for k, (p, x, g) in enumerate(steps):
        m = foot(p, sc.pose[p][x], fc=cmap(k / max(len(steps) - 1, 1)), ec="k", alpha=0.85)
        ax.text(m[0], m[1], f"{k + 1}\n{p}", ha="center", va="center", fontsize=8, color="w", weight="bold")
    ax.plot(0, 0, "k^", ms=10)
    ax.text(0, -0.02, "robot base", ha="center", va="top", fontsize=8)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Initial layout (number = assembly order); dashed = goal")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", nargs="?", default=None)
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--rrt-all", action="store_true", help="RRT for every free segment (raw and shortcut times)")
    ap.add_argument("--seed0", type=int, default=0, help="first RRT seed (repeated runs)")
    args = ap.parse_args(argv)
    res_path = Path(args.result) if args.result else Path(max(glob.glob(str(OUT / "frame*_2*.json")), key=lambda f: Path(f).stat().st_mtime))
    R = json.loads(res_path.read_text(encoding="utf-8"))
    steps = [(p, x, int(g)) for p, x, g in R["chosen"]["steps"]]

    from lazy_cost.detour import blend
    from lazy_cost.timing import topp
    from .problem import Problem, t_acc
    from .validate import forward_steps

    pb = Problem(args.model)
    sc = pb.scene
    W = MeshWorld(pb)
    G = sc.grasps()
    print(f"{res_path.name}: chosen plan, search estimate {R['chosen']['est']:.3f} s, bound {R['chosen']['lb']:.3f} s\n")
    print("  order  part      initial pose (x, y, yaw)   grasp")
    layout = []
    for k, (p, x, g) in enumerate(steps):
        T = sc.pose[p][x]
        yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        layout.append(dict(order=k + 1, part=p, pose_key=x, xyz=T[:3, 3].tolist(), pose=T.tolist(), grasp=g,
                           grasp_tcp_in_object=np.asarray(G[p][g][0]).tolist(), goal=sc.goal[p].tolist()))
        print(f"  {k + 1:5d}  {p:8s}  {T[0, 3]:6.3f} {T[1, 3]:7.3f}  {x.split(',')[-1]:>8s}   {g}")

    print("\n  segment   part      straight?  result                      time (s)   check (s)")
    rows, total, ok_all = [], 0.0, True
    for s in forward_steps(pb, SimpleNamespace(suffix=steps)):
        if s["kind"] == "fixed":
            rows.append(dict(kind="fixed", part=s["part"], time=s["value"], how="fixed (not mesh-checked)"))
            total += s["value"]
            continue
        t0 = perf_counter()
        table = {q: sc.pose[q][xx] for q, xx in s["table"].items()}
        if s["kind"] == "transit":
            table[s["part"]] = sc.pose[s["part"]][s["x"]]
        held = (s["part"], G[s["part"]][s["g"]][0], s["a"]) if s["kind"] == "transfer" else None
        W.set(s["installed"], table, held, jaw=pb.jaw[s["part"]][s["g"]] if held and pb.close_fingers else None)
        lb = float(t_acc(s["a"], s["b"]))
        ok, why = W.free([s["a"], s["b"]])
        raw = None
        if args.rrt_all:
            t, how = np.inf, "RRT failed"
            for seed in range(args.seed0, args.seed0 + RRT_SEEDS):
                P, dt = W.planner._rrt(s["a"], s["b"], seed=seed)
                if P is None:
                    continue
                okp, whyp = W.free(P)
                if not okp:
                    how = f"RRT path rejected ({whyp})"
                    continue
                raw = float(topp(P)[1][-1])
                S = W.shortcut(P)
                t, how = float(topp(S)[1][-1]), f"RRT seed {seed}: {len(P)} -> {len(S)} waypoints"
                if len(S) > 2 and W.free(blend(S))[0]:
                    t, how = min(t, float(topp(blend(S))[1][-1])), how + ", blended"
                break
        elif ok:
            t, how = lb, "straight line free"
        else:
            t, how = np.inf, f"blocked ({why}); RRT failed"
            for seed in range(RRT_SEEDS):          # RRT only checks its nodes: re-check every path densely
                P, dt = W.planner._rrt(s["a"], s["b"], seed=seed)
                if P is None:
                    continue
                okp, whyp = W.free(P)
                if not okp:
                    how = f"blocked ({why}); RRT path rejected ({whyp})"
                    continue
                S = W.shortcut(P)
                t, how = float(topp(S)[1][-1]), f"blocked ({why}); RRT seed {seed}, {len(S)} waypoints"
                if len(S) > 2 and W.free(blend(S))[0]:     # corners of the polyline force a stop: blend them
                    t, how = min(t, float(topp(blend(S))[1][-1])), how + ", blended"
                break
        ok_all &= bool(np.isfinite(t))
        total += t
        rows.append(dict(kind=s["kind"], part=s["part"], time=t, raw_rrt=raw, lb=lb, straight=bool(ok), how=how,
                         check_s=perf_counter() - t0))
        print(f"  {s['kind']:9s} {str(s['part']):9s} {'yes' if ok else 'no':9s}  {how[:62]:62s} {t:7.3f}"
              + (f"  raw {raw:6.3f}" if raw is not None else "") + f"   {perf_counter() - t0:6.1f}", flush=True)
    W.set([], {})
    print(f"\n  WRS check {'PASSED' if ok_all else 'FAILED'}; executed time {total:.3f} s "
          f"(search estimate {R['chosen']['est']:.3f}, cuRobo-confirmed {R['chosen']['confirmed']:.3f})")
    stem = res_path.stem
    if args.rrt_all:
        raw_total = sum(r["raw_rrt"] if r.get("raw_rrt") is not None else r["time"] for r in rows)
        print(f"  RRT for every segment: raw RRT paths {raw_total:.3f} s, after shortcut + blend {total:.3f} s")
        stem += "_rrtall"
    out = OUT / f"{stem}_layout.json"
    out.write_text(json.dumps(dict(source=res_path.name, model=args.model, ok=ok_all, total_time=total, estimate=R["chosen"]["est"],
                                   bound=R["chosen"]["lb"], layout=layout, segments=rows), indent=1, default=float), encoding="utf-8")
    png = OUT / f"{stem}_layout.png"
    topview(pb, steps, png)
    print("->", out, "\n->", png)


if __name__ == "__main__":
    main()
