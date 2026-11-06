"""Assembly scenes for the cost: rack10-style box models (scenes/benchmarks) and the triangular
frame (scenes/triangular_frame6). ``load_scene(name)`` returns either.

Attributes used by the cost: name, ids, meshes, goal, table_center/size, site_xy, site_box,
candidates[pid] = [(key, 4x4)], polygons[pid][key] (table footprint), pose[pid][key],
grasps() (WRS antipodal grasp files, see gen_grasps.py), held_spheres(pid),
step_ok(installed, pid) (computed online: static stability + insertion path), sample_order(rng).

Candidates (same rule for every model): table grid (step 4 cm) x resting orientations x yaws
0/45/90/135 deg, footprint inside the table and clear of the robot base and of the assembly base.
grid="shared" (default): one grid for every part, np.arange from the table corner (the far edges
lose up to one step). grid="fine": one fine grid for every part (spacing
grid_step, default 1.5 cm, np.linspace over the table to both edges): any free slot at least
grid_step wider than the part holds a point, whatever the part's size, and all parts share one
lattice; the candidates are thinned afterwards by farthest-point sampling (pool * n_cand
positions), so the IK work does not grow with the density.
keepout="box" (default): the footprint must clear the square |x|, |y| < 13 cm around the robot
base and the assembly base's AABB grown by 3 cm. keepout="sealp" (SEALP/BLS rule): clear the
arm-base rectangle |x|, |y| < 12 cm (SEALP staging_arm_x/y_clearance, here against the true
footprint instead of an AABB centred on the point) and keep >= 1 cm (SEALP
min_staging_mesh_clearance) from the assembly: the convex hull of the base and of every part at
its goal, in plan view (SEALP checks the base mesh only; a part staged under the place of an
installed part would collide with it). The arm at HOME is checked by the consumer
(joint_search.problem._drop_home_colliding), as SEALP's _robot_home_collision_reason.
center=(x, y, z): the assembly centre (SEALP --goal-pos, the preassembled base's position): the
whole assembly (base + every goal) is translated so that the base sits there; None = as in the
model files (frame: 0.35, 0, 0).
Resting orientations: placements="goal" (default) = the part rests as at its goal orientation
(rack10 shelves stand on their long edge, see AssemblyScene);
placements="sealp" = the orientation candidates of SEALP/BLS used the SEALP way
(sealp_rot_candidates: the same functions, order and cap; they already contain the 90-degree
yaws, so no extra yaw is added): farthest-point sampling over (x, y) only, every orientation at
each position, and the consumer (joint_search.problem / expand, Problem.orient) uses per position
the first orientation in SEALP order that is feasible (SEALP: first feasible rot_cand per xy,
poses_per_xy = 1).
Keys are "x,y,yaw<deg>" for the goal orientation and "x,y,yaw<deg>,<tag>" for the others.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FRAME_DIR = REPO / "scenes" / "triangular_frame6"


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1.0]])


def model_files(name):
    """(STL directory in the part frame, WRS grasp directory) of a model."""
    if name == "triangular_frame6":
        return FRAME_DIR / "assets" / "meshes", FRAME_DIR / "assets" / "grasps_wrs"
    root = REPO / "scenes" / "benchmarks" / name
    return root / "stl_m", root / "grasps"


def load_scene(name, **kw):
    if name == "triangular_frame6":
        return FrameScene(**kw)
    import importlib
    return AssemblyScene(importlib.import_module(f"scenes.benchmarks.{name}").spec(), **kw)


YAW_M = 0.035 / 45.0      # farthest-point sampling: 45 deg of yaw counts as 3.5 cm of position


def key_yaw(key):
    """Yaw (deg) of a candidate key "x,y,yaw<deg>[,s<i>]"."""
    return float(key.split("yaw")[1].split(",")[0])


def key_xy(key):
    """Position part "x,y" of a candidate key."""
    return ",".join(key.split(",")[:2])


def key_placement(key):
    """Resting-orientation tag of a candidate key ("" = goal orientation, "s<i>" = other)."""
    parts = key.split(",")
    return parts[3] if len(parts) > 3 else ""


def sealp_rot_candidates(mesh_path, vertices):
    """Orientation candidates of a part exactly as SEALP/BLS build them
    (find_optimal_initial_layout_tower_strict_pycharm: _precompute_rot_candidates, the module the
    BSFS entry bsfs/run.py imports), using its own functions and defaults:
      1. _stable_pose_candidates_from_flatsurface(mesh, DEFAULT_FS_STABILITY_THRESHOLD = 0.10,
         max(2 * DEFAULT_MAX_ROT_CANDIDATES, 16)) -> "fs_XX" (WRS FSReferencePoses)
      2. + _fallback_rot_candidates_with_names(): identity and 90-degree rotations x 4 yaws
         ("identity", "rot90_XX"; no stability check)
      3. _dedupe_pose_rot_candidates (same rotation matrix)
      4. tags and sort: identity, then fs_*, then auto_upright, then the rest; inside a group the
         lower and the smaller footprint first
      5. the first DEFAULT_MAX_ROT_CANDIDATES (12)
    Note: FSReferencePoses alone misses faces of box-like parts (its ODE ray from the centre of
    mass hits the triangle diagonal of a face); in SEALP the 90-degree fallbacks restore them,
    which is why the whole procedure, not the stable-pose call alone, is reproduced.
    The SEALP force_rot_name override and the post-search 20-degree yaw refinement of
    bsfs/run.py are not reproduced. -> [(rot_name, tag, R)] in SEALP order."""
    import sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm as fol
    pose_rots = list(fol._stable_pose_candidates_from_flatsurface(
        str(mesh_path), stability_threshold=fol.DEFAULT_FS_STABILITY_THRESHOLD,
        max_candidates=max(fol.DEFAULT_MAX_ROT_CANDIDATES * 2, 16)))
    pose_rots.extend(fol._fallback_rot_candidates_with_names())
    pose_rots = fol._dedupe_pose_rot_candidates(pose_rots)
    verts = np.asarray(vertices, float)
    _, _, identity_extent = fol._bounds_after_rotation(verts, np.eye(3))
    is_flat = float(identity_extent.min() / max(identity_extent.max(), 1e-9)) < 0.22
    cands = []
    for rot_name, _, R in pose_rots:
        _, _, extent = fol._bounds_after_rotation(verts, R)
        from_fs = str(rot_name).startswith("fs_")
        if np.allclose(R, np.eye(3), atol=1e-6):
            tag = "identity"
        elif from_fs and is_flat and extent[2] > identity_extent[2] * 2.0:
            tag = "fs_upright_lifted"
        elif from_fs:
            tag = "fs_stable_lifted"
        elif is_flat and extent[2] > identity_extent[2] * 2.0:
            tag = "auto_upright_lifted"
        else:
            tag = "rot90_lifted"
        group = 0 if tag == "identity" else 1 if from_fs else 2 if tag == "auto_upright_lifted" else 3
        cands.append(((group, float(extent[2]), float(np.linalg.norm(extent[:2]))), str(rot_name), tag,
                      np.asarray(R, float)))
    cands.sort(key=lambda c: c[0])
    return [(name, tag, R) for _, name, tag, R in cands[:fol.DEFAULT_MAX_ROT_CANDIDATES]]


def _down(R):
    """Part-frame direction that points down when the part is rotated by R."""
    d = np.asarray(R, float).T @ np.array([0.0, 0.0, -1.0])
    return d / np.linalg.norm(d)


def fps_order(xy, yaw_deg, n):
    """Farthest-point order (first ``n``) over (x, y, yaw): yaw is circular with period 180 deg and
    weighted by YAW_M per degree. Starts at the option closest to the mean (as SEALP BSFS)."""
    th = np.radians(2.0 * np.asarray(yaw_deg, float))                      # period 180 deg
    r = YAW_M * 90.0 / np.sqrt(2.0)                                           # 45 deg -> chord YAW_M * 45
    F = np.c_[np.asarray(xy, float), r * np.cos(th), r * np.sin(th)]
    if not len(F):
        return []
    order = [int(np.argmin(np.linalg.norm(F - F.mean(0), axis=1)))]
    d = np.linalg.norm(F - F[order[0]], axis=1)
    while len(order) < min(n, len(F)):
        i = int(np.argmax(d))
        order.append(i)
        d = np.minimum(d, np.linalg.norm(F - F[i], axis=1))
    return order


class _SceneBase:
    """Candidate generation, grasp loading and the part sphere cover shared by all models.
    Subclasses set name, ids, meshes, goal, table_center, table_size, then call _place(center) and
    _finish()."""

    def _place(self, center):
        """Translate the assembly so that the base is at ``center`` (None: keep the model's)."""
        self.center = None if center is None else np.asarray(center, float).reshape(3)
        if self.center is not None:
            d = self.center - self.goal["base"][:3, 3]
            for T in self.goal.values():
                T[:3, 3] += d

    def _finish(self, n_cand, yaws, step, seed, sampler="random", pool=2, placements="goal",
                grid="shared", keepout="box", grid_step=0.015):
        """sampler "random": n_cand options drawn at random (seed). "fps": the first pool * n_cand
        options in farthest-point order over (x, y, yaw); the consumer keeps the first n_cand of
        them that have a usable grasp (joint_search.problem), so every kept candidate is usable and
        the kept set stays spread out. Candidates are listed in that order."""
        from shapely.affinity import translate
        from shapely.geometry import MultiPoint, box as sbox

        rng = np.random.default_rng(seed)
        self.n_cand, self.sampler, self.pool = n_cand, sampler, int(pool)
        self.placements = placements
        if grid not in ("shared", "fine"):
            raise ValueError(f"grid must be shared or fine, not {grid!r}")
        self.grid, self.grid_step = grid, float(grid_step)
        if keepout not in ("box", "sealp"):
            raise ValueError(f"keepout must be box or sealp, not {keepout!r}")
        self.keepout = keepout
        self.placement_report = {}
        lo = self.table_center[:2] - self.table_size[:2] / 2 + 0.01
        hi = self.table_center[:2] + self.table_size[:2] / 2 - 0.01
        table = sbox(*lo, *hi)
        bb = self.meshes["base"].bounds + self.goal["base"][:3, 3]
        if keepout == "sealp":
            robot = sbox(-0.12, -0.12, 0.12, 0.12)
            plan = [np.asarray(self.meshes[q].vertices) @ self.goal[q][:3, :3].T + self.goal[q][:3, 3]
                    for q in ("base",) + tuple(self.ids)]
            site = MultiPoint(np.vstack(plan)[:, :2]).convex_hull.buffer(0.01)
        else:
            robot = sbox(-0.13, -0.13, 0.13, 0.13)
            site = sbox(*(bb[0, :2] - 0.03), *(bb[1, :2] + 0.03))
        self.site_xy = self.goal["base"][:2, 3].copy()
        self.site_box = bb
        self.candidates, self.polygons, self.fit_xy, self.pose = {}, {}, {}, {}
        self.options, self._orients, self._rng = {}, {}, rng   # options[pid] = [(key, 4x4, footprint)] that fit
        xs, ys = np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step)
        if grid == "fine":
            n = np.floor((hi - lo) / self.grid_step + 1e-3).astype(int) + 1
            xs, ys = np.linspace(lo[0], hi[0], n[0]), np.linspace(lo[1], hi[1], n[1])
        for pid in self.ids:
            R0 = self.goal[pid][:3, :3]                       # rest as at the goal (flat face down)
            if pid in getattr(self, "stand_on_edge", ()):     # thin plate: stand it on its long edge
                R0 = R0 @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # part z -> horizontal
            options = []
            orients = self._orientations(pid, R0)
            shapes = []                                       # (tag, yaw, R, vertices, footprint hull)
            for tag, Rb in orients:
                for yaw in ((0,) if self.placements == "sealp" else yaws):
                    R = _rz(yaw) @ Rb
                    vr = np.asarray(self.meshes[pid].vertices) @ R.T
                    shapes.append((tag, yaw, R, vr, MultiPoint(vr[:, :2]).convex_hull))
            for tag, yaw, R, vr, hull in shapes:
                for x in xs:
                    for y in ys:
                        fp = translate(hull, x, y)
                        if not table.contains(fp) or fp.intersects(robot) or fp.intersects(site):
                            continue
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = [x, y, -vr[:, 2].min() + 0.0002]
                        options.append((f"{x:.3f},{y:.3f},yaw{yaw}" + (f",{tag}" if tag else ""), T, fp))
            self.options[pid], self._orients[pid] = options, orients
            self._select(pid)
        self._grasps = self._load_grasps()

    def _select(self, pid):
        """Candidates of ``pid`` from its options (every (position, orientation) that fits):
        farthest-point order (sampler "fps") or a random draw; sets fit_xy, candidates, polygons, pose."""
        options, orients, n_cand, sampler, rng = self.options[pid], self._orients[pid], self.n_cand, self.sampler, self._rng
        if sampler == "fps" and self.placements == "sealp":
            # SEALP: farthest-point order over the positions (x, y); at each position every
            # orientation that fits, in SEALP order
            at = {}
            for i, o in enumerate(options):
                at.setdefault(key_xy(o[0]), []).append(i)
            xy_keys = list(at)
            rank = {t: r for r, (t, _) in enumerate(orients)}
            order = fps_order([options[at[k][0]][1][:2, 3] for k in xy_keys], [0.0] * len(xy_keys),
                              self.pool * n_cand)
            pick = [i for j in order for i in sorted(at[xy_keys[j]], key=lambda i: rank[key_placement(options[i][0])])]
        elif sampler == "fps":                                # farthest-point order over (x, y, yaw)
            pick = fps_order([o[1][:2, 3] for o in options], [key_yaw(o[0]) for o in options], self.pool * n_cand)
        else:
            pick = sorted(rng.permutation(len(options))[:n_cand])
        self.fit_xy[pid] = list(dict.fromkeys(key_xy(o[0]) for o in options))   # every position that fits
        self.candidates[pid] = [(options[i][0], options[i][1]) for i in pick]
        self.polygons[pid] = {options[i][0]: options[i][2] for i in pick}
        self.pose[pid] = dict(self.candidates[pid])

    def drop_options(self, pid, keys):
        """Remove the options ``keys`` of ``pid`` (a check the scene cannot make itself, e.g. the arm at
        HOME, which needs the GPU checker built on this scene) and redo the selection, so the
        farthest-point sampling runs on the options that are left."""
        keys = set(keys)
        if keys:
            self.options[pid] = [o for o in self.options[pid] if o[0] not in keys]
            self._select(pid)

    def _orientations(self, pid, R0):
        """[(tag, R)] resting orientations of a part (module docstring): placements="sealp" = the
        SEALP/BLS rot_cands; otherwise (and for the stand_on_edge override) only the goal one R0."""
        if pid in getattr(self, "stand_on_edge", ()) or self.placements != "sealp":
            self.placement_report[pid] = dict(mode="goal")
            return [("", R0)]
        mesh_path = model_files(self.name)[0] / f"{pid}.stl"
        rc = sealp_rot_candidates(mesh_path, self.meshes[pid].vertices)
        self.placement_report[pid] = dict(mode="sealp", used=[(n, t, [round(float(c), 3) for c in _down(R)])
                                                              for n, t, R in rc])
        return [(n, R) for n, _, R in rc]

    def _load_grasps(self):
        """WRS antipodal grasp files written by ``lazy_cost.gen_grasps`` -> {pid: [(G, jaw
        width)]}, G = TCP (action centre) pose in the part frame."""
        from wrs.grasping.grasp import GraspCollection
        gdir = model_files(self.name)[1]
        out = {}
        for pid in self.ids:
            path = gdir / f"{pid}_grasps.pickle"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run `python -m lazy_cost.gen_grasps --model {self.name}`")
            gs = []
            for g in GraspCollection.load_from_disk(str(path)):
                G = np.eye(4)
                G[:3, :3], G[:3, 3] = g.ac_rotmat, g.ac_pos
                gs.append((G, float(np.ravel(g.ee_values)[0])))
            out[pid] = gs
        return out

    def grasps(self):
        return self._grasps

    def held_spheres(self, pid, spacing=0.006, radius=0.004):
        """Surface sphere cover of a part in its local frame: points sampled evenly on the
        mesh surface every ~``spacing`` m, each a sphere of ``radius`` (the cover reaches at
        most ``radius`` beyond the surface; robot spheres, >= 1 cm, cannot pass between
        neighbours). Used for the held part and for the part at its candidate pose."""
        if not hasattr(self, "_cover"):
            self._cover = {}
        if pid not in self._cover:
            import trimesh
            m = self.meshes[pid]
            n = int(np.ceil(m.area / spacing ** 2))
            pts, _ = trimesh.sample.sample_surface_even(m, n, seed=0)
            pts = np.r_[pts, m.vertices]
            self._cover[pid] = np.c_[pts, np.full(len(pts), radius)]
        return self._cover[pid]


class AssemblyScene(_SceneBase):
    """Box-union model from scenes/benchmarks (e.g. rack10)."""

    def __init__(self, spec, *, n_cand=40, yaws=(0, 45, 90, 135), step=0.04, seed=0, sampler="random", pool=2,
                 placements="goal", grid="shared", keepout="box", center=None,
                 grid_step=0.015):
        import trimesh
        from scenes.benchmarks.common import TABLE_CENTER, TABLE_SIZE, Model, explore

        self.model = m = Model(spec)
        self.name = spec.name
        self.ids = tuple(m.ids)
        self.table_center, self.table_size = TABLE_CENTER.copy(), TABLE_SIZE.copy()
        self.meshes = {pid: trimesh.Trimesh(m.mesh(pid).vertices, m.mesh(pid).faces, process=False)
                       for pid in ["base", *self.ids]}
        self.goal = {pid: np.array(T) for pid, T in m.goal.items()}
        # rack10 shelves (220 x 84 x 12): every WRS grasp clamps the 12 mm thickness, so lying flat the
        # lower finger is inside the table; on the table they stand on the 220 x 12 edge (84 mm tall)
        self.stand_on_edge = tuple(p for p in self.ids if p.startswith("shelf_"))
        self._place(center)
        self._finish(n_cand, yaws, step, seed, sampler, pool, placements, grid, keepout, grid_step)
        self._graph = explore(m, "insertion")
        self.n_orders = self._graph["orders"]

    def step_ok(self, installed, pid) -> bool:
        """Static stability (support rule, verified against the asmcore equilibrium LP for
        this benchmark family) and a free insertion path for adding ``pid``."""
        S = set(installed)
        return self.model.stable_rule(S | {pid}) and not self.model.insertion_blockers(pid, S)

    def sample_order(self, rng):
        from scenes.benchmarks.common import sample_orders
        return sample_orders(self.model, self._graph, rng, k=1)[0]


class FrameScene(_SceneBase):
    """Triangular frame: 3 posts + 3 beams on a base.

    step_ok is computed ONLINE (no order table): asmcore static equilibrium LP of the assembly
    state before and after the step, and a mesh sweep of the part along its insertion axis
    against the parts already installed. Results are memoised per part set, so each state is
    solved once, the first time the search reaches it. The 48 stable orders of validation.json
    are kept only for the experiments that sample a random order (sample_order, n_orders)."""

    REMOVAL_TRAVEL = 0.15     # m swept along the removal axis (+z) for the insertion check
    REMOVAL_STEP = 0.005

    def __init__(self, *, n_cand=40, yaws=(0, 45, 90, 135), step=0.04, seed=0, sampler="random", pool=2,
                 placements="goal", grid="shared", keepout="box", center=None,
                 grid_step=0.015):
        import trimesh
        task = json.loads((FRAME_DIR / "task.json").read_text(encoding="utf-8"))
        val = json.loads((FRAME_DIR / "validation.json").read_text(encoding="utf-8"))
        self.name = "triangular_frame6"
        self.ids = tuple(task["movable_ids"])
        self.table_center = np.array(task["table"]["table_center"], float)
        self.table_size = np.array(task["table"]["table_size"], float)
        self.meshes, self.goal = {}, {}
        for pid in ("base",) + self.ids:
            self.meshes[pid] = trimesh.load(FRAME_DIR / "assets" / "meshes" / f"{pid}.stl", process=False)
            self.goal[pid] = np.array(task["parts"][pid]["goal_tf"], float)
        self.removal_axis = {p: np.array(task["parts"][p].get("removal_axis_world", [0, 0, 1]), float) for p in self.ids}
        self.orders = [tuple(o) for o in val["nominal_static_orders"]]
        self.n_orders = len(self.orders)
        self._eq = None
        self._stable, self._free = {}, {}
        self.online_stats = dict(lp_calls=0, lp_s=0.0, sweep_calls=0, sweep_s=0.0)
        self._place(center)
        self._finish(n_cand, yaws, step, seed, sampler, pool, placements, grid, keepout, grid_step)

    # ---- online feasibility of a step ------------------------------------------------------------
    def stable(self, parts) -> bool:
        """asmcore static equilibrium (gravity, friction, no auxiliary support) of the base + parts."""
        key = frozenset(parts)
        if key not in self._stable:
            from time import perf_counter
            from asmcore.model import AssemblyState
            if self._eq is None:
                import scenes.triangular_frame6.build_model as bm
                from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
                asm, _, _ = bm.build_assembly()
                self._eq = (SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0)),
                            {p.part_id: p.assembled_tf for p in asm.parts},
                            {p.part_id: p.assembled_tf for p in asm.parts if p.fixed})
            ev, tf, fixed = self._eq
            t0 = perf_counter()
            r = ev.equilibrium(AssemblyState(dict(fixed, **{p: tf[p] for p in key})), ())
            self.online_stats["lp_calls"] += 1
            self.online_stats["lp_s"] += perf_counter() - t0
            self._stable[key] = r.status == "feasible"
        return self._stable[key]

    def insertable(self, pid, installed) -> bool:
        """The part, moved along its removal axis from its goal, does not hit the installed parts or
        the base (mesh-mesh, WRS/ODE) = its straight insertion path is free (object only)."""
        key = (pid, frozenset(installed))
        if key not in self._free:
            from time import perf_counter
            import wrs.basis.trimesh as wtrm
            import wrs.modeling.collision_model as mcm
            t0 = perf_counter()
            cm = lambda m: mcm.CollisionModel(initor=wtrm.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces)))
            if not hasattr(self, "_cms"):
                self._cms = {p: cm(self.meshes[p]) for p in ("base",) + self.ids}
                self._mover = {p: cm(self.meshes[p]) for p in self.ids}
            obs = []
            for q in ("base",) + tuple(installed):
                c = self._cms[q]
                c.pose = (self.goal[q][:3, 3].copy(), self.goal[q][:3, :3].copy())
                obs.append(c)
            mv, G = self._mover[pid], self.goal[pid]
            ok = True
            for d in np.arange(0.001, self.REMOVAL_TRAVEL + 1e-9, self.REMOVAL_STEP):
                mv.pose = (G[:3, 3] + d * self.removal_axis[pid], G[:3, :3].copy())
                if mv.is_mcdwith(obs):
                    ok = False
                    break
            self.online_stats["sweep_calls"] += 1
            self.online_stats["sweep_s"] += perf_counter() - t0
            self._free[key] = ok
        return self._free[key]

    def step_ok(self, installed, pid) -> bool:
        """Adding ``pid`` after the set ``installed`` (= removing it last from installed + pid):
        both states statically stable and the straight insertion path of pid free."""
        S = frozenset(installed)
        return self.stable(S) and self.stable(S | {pid}) and self.insertable(pid, S)

    def sample_order(self, rng):
        return list(self.orders[int(rng.integers(len(self.orders)))])
