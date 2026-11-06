"""Shared modelling, feasibility analysis and preview for single-arm benchmark assemblies.

Every part is a union of axis-aligned boxes (minus pockets), installed top-down at a fixed
site on the 0.6 x 0.6 m SEALP table. For every transition "installed set S -> add part p"
three conditions are evaluated:

1. static support : every installed part has its declared supports installed
                    (rule); cross-checked with asmcore static equilibrium LP on every
                    reachable state.
2. insertion path : p translated from its goal up to the 6 cm pre-insert height does not
                    penetrate any installed part (exact box cells, 0.3 mm tolerance).
3. grasp          : some Panthera pinch grasp of p (4 yaws x 7 approach rolls) has goal-side
                    IK (cuRobo batch IK: pre-insert, insert, retreat) and a gripper that does
                    not hit the table, the base, installed parts or p itself, at the goal and
                    along the 6 cm insertion. Gripper = palm + two fingers as oriented boxes
                    measured from the WRS Panthera gripper meshes in the TCP frame.

The state graph is enumerated exactly (all reachable installed sets). Orders are counted
by dynamic programming over that graph; "a always before b" is exact (no co-reachable
state contains b without a). Robot transport between table and site is NOT checked here.
"""
from __future__ import annotations

import base64
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .boxkit import box, cells, mesh

SITE = np.array([0.30, 0.0, 0.0])          # nearer the robot: more reachable grasp orientations
TABLE_CENTER = np.array([0.2335, 0.0, -0.01])
TABLE_SIZE = np.array([0.6, 0.6, 0.02])
DENSITY = 650.0                     # kg/m^3 (wood / PLA-like)
CLEAR = 0.0002                      # pocket side clearance
JAW_MAX = 0.085
ROLLS = (0, 30, -30, 60, -60, 90, -90)
APPROACH = 0.06
PART_TOL, GRIP_TOL, OWN_TOL = 3e-4, 5e-4, 1.2e-3


@dataclass
class PartSpec:
    pid: str
    name: str
    solids: list
    holes: list = field(default_factory=list)
    pos: tuple = (0.0, 0.0, 0.0)          # goal translation relative to SITE
    color: tuple = (0.6, 0.6, 0.6)
    supports: tuple = ()                   # all must be installed first (static rule)
    grasp_region: tuple | None = None      # local box used for pinch grasps
    role: str = ""


def part(pid, name, solids_mm, holes_mm=(), *, color, supports=(), grasp_mm=None, role=""):
    """Build a PartSpec from boxes given in the ASSEMBLY frame in millimetres.

    The part's local frame is re-centred on its bounding-box centre so STL files and
    staging poses are well conditioned; ``pos`` stores that centre relative to the site.
    """
    sol = np.asarray(solids_mm, float) / 1000
    hol = np.asarray(holes_mm, float).reshape(-1, 6) / 1000
    c = np.array([(sol[:, 0].min() + sol[:, 1].max()) / 2, (sol[:, 2].min() + sol[:, 3].max()) / 2,
                  (sol[:, 4].min() + sol[:, 5].max()) / 2])
    shift = np.repeat(c, 2)
    g = None if grasp_mm is None else tuple(np.asarray(grasp_mm, float) / 1000 - shift)
    return PartSpec(pid, name, [tuple(b - shift) for b in sol], [tuple(b - shift) for b in hol],
                    tuple(c), tuple(color), tuple(supports), g, role)


@dataclass
class ModelSpec:
    name: str
    title: str
    base: PartSpec
    parts: list
    description: list


# ---------------------------------------------------------------------------------------
# gripper: palm + 2 fingers in the TCP frame (metres), measured from WRS Panthera meshes
def gripper_boxes(w):
    h = w / 2
    f0 = np.array([[-0.0225, 0.0185], [h - 0.0008, h + 0.0354], [-0.0952, 0.0098]])
    f1 = np.array([[-0.0224, 0.0187], [-h - 0.0357, -h + 0.0001], [-0.0950, 0.0101]])
    palm = np.array([[-0.0250, 0.0251], [-0.0793, 0.0790], [-0.1653, -0.0842]])
    return [(n, b.mean(1), (b[:, 1] - b[:, 0]) / 2)
            for n, b in (("palm", palm), ("finger_0", f0), ("finger_1", f1))]


def obb_hits(c, R, h, boxes, tol):
    """Oriented box (centre c, axes = columns of R, half sizes h) vs AABBs [K,6].

    Separating-axis test; True where the overlap exceeds ``tol`` on every axis."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    a = np.stack([boxes[:, 0] + boxes[:, 1], boxes[:, 2] + boxes[:, 3],
                  boxes[:, 4] + boxes[:, 5]], 1) / 2
    e = np.stack([boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2],
                  boxes[:, 5] - boxes[:, 4]], 1) / 2
    d = a - c
    axes = [np.eye(3)[i] for i in range(3)] + [R[:, j] for j in range(3)]
    for i in range(3):
        for j in range(3):
            v = np.cross(np.eye(3)[i], R[:, j])
            n = np.linalg.norm(v)
            if n > 1e-9:
                axes.append(v / n)
    hit = np.ones(len(boxes), bool)
    for L in axes:
        ra = e @ np.abs(L)
        rb = float(np.sum(h * np.abs(R.T @ L)))
        hit &= np.abs(d @ L) < ra + rb - tol
    return hit


def aabb_hits(a, boxes, tol):
    """AABBs a [K,6] vs boxes [M,6] -> bool [K,M], overlap > tol on all axes."""
    if len(a) == 0 or len(boxes) == 0:
        return np.zeros((len(a), len(boxes)), bool)
    ok = np.ones((len(a), len(boxes)), bool)
    for k in range(3):
        lo = np.maximum(a[:, None, 2 * k], boxes[None, :, 2 * k])
        hi = np.minimum(a[:, None, 2 * k + 1], boxes[None, :, 2 * k + 1])
        ok &= hi - lo > tol
    return ok


def _rz(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1.0]])


def _ry(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1.0, 0], [-math.sin(t), 0, math.cos(t)]])


# ---------------------------------------------------------------------------------------
class Model:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.all = [spec.base] + list(spec.parts)
        self.ids = [p.pid for p in spec.parts]
        self.by = {p.pid: p for p in self.all}
        self.local_cells = {p.pid: cells(p.solids, p.holes) for p in self.all}
        self.goal = {}
        for p in self.all:
            T = np.eye(4)
            T[:3, 3] = SITE + np.asarray(p.pos, float)
            self.goal[p.pid] = T
        self.world_cells = {pid: c + np.repeat(self.goal[pid][:3, 3], 2)
                            for pid, c in self.local_cells.items()}
        t = np.r_[TABLE_CENTER - TABLE_SIZE / 2, TABLE_CENTER + TABLE_SIZE / 2]
        self.table_cells = np.array([[t[0], t[3], t[1], t[4], t[2], t[5]]])
        self.volume = {pid: float(np.prod(c[:, 1::2] - c[:, 0::2], axis=1).sum())
                       for pid, c in self.local_cells.items()}
        self._grasps, self._ik = {}, {}

    def mesh(self, pid):
        p = self.by[pid]
        return mesh(p.solids, p.holes)

    def mass(self, pid):
        return DENSITY * self.volume[pid]

    def bounds(self, pid):
        c = self.local_cells[pid]
        return c[:, 0::2].min(0), c[:, 1::2].max(0)

    # ---- grasps ------------------------------------------------------------------------
    def grasps(self, pid):
        """Pinch grasps on the grasp region: 4 yaws (opening along x or y) x 7 rolls."""
        if pid not in self._grasps:
            p = self.by[pid]
            if p.grasp_region is None:
                lo, hi = self.bounds(pid)
            else:
                r = p.grasp_region
                lo, hi = np.array(r[0::2]), np.array(r[1::2])
            size, centre = hi - lo, (lo + hi) / 2
            out = []
            for yaw in (0, 90, 180, 270):
                r = _rz(yaw) @ np.diag([1.0, -1.0, -1.0])          # tool z points down
                opening = r @ np.array([0, 1.0, 0])                 # Panthera opens along tool y
                width = float(np.abs(opening) @ size)
                if width > JAW_MAX:
                    continue
                tcp = centre.copy()
                tcp[2] += 0.2 * size[2]
                for roll in ROLLS:
                    G = np.eye(4)
                    G[:3, :3] = r @ _ry(roll)
                    G[:3, 3] = tcp
                    out.append(dict(G=G, width=width, yaw=yaw, roll=roll))
            self._grasps[pid] = out
        return self._grasps[pid]

    def goal_ik(self, bik):
        """Goal-side keyposes (pre-insert, insert, retreat) solvable, per part and grasp."""
        from types import SimpleNamespace
        from curobo_bridge.assembly_demo import keyposes
        for pid in self.ids:
            gs = self.grasps(pid)
            if not gs:
                self._ik[pid] = np.zeros(0, bool)
                continue
            part = SimpleNamespace(goal=self.goal[pid])
            P = np.stack([keyposes(part, np.eye(4), g["G"])[k] for g in gs
                          for k in ("preins", "ins", "retreat")])
            _, ok = bik.solve(P)
            self._ik[pid] = ok.reshape(len(gs), 3, -1).any(-1).all(-1)

    # ---- geometric checks ----------------------------------------------------------------
    def obstacles(self, installed):
        return {"table": self.table_cells, "base": self.world_cells["base"],
                **{q: self.world_cells[q] for q in installed}}

    def insertion_blockers(self, pid, installed):
        """Installed parts penetrated when p descends from +6 cm to its goal."""
        if not hasattr(self, "_ins"):
            self._ins = {}
        if pid not in self._ins:
            hit = set()
            for q in ["base", *[x for x in self.ids if x != pid]]:
                for dz in np.linspace(0, APPROACH, 13):
                    moved = self.world_cells[pid] + np.array([0, 0, 0, 0, dz, dz])
                    if aabb_hits(moved, self.world_cells[q], PART_TOL).any():
                        hit.add(q)
                        break
            self._ins[pid] = hit
        return self._ins[pid] & (set(installed) | {"base"})

    def grasp_blockers(self, pid, g, installed):
        """Obstacles hit by the gripper for grasp g (goal and along the insertion)."""
        obs = self.obstacles(installed)
        hit = set()
        for dz in (0.0, 0.02, 0.04, APPROACH):
            T = self.goal[pid].copy()
            T[2, 3] += dz
            tcp = T @ g["G"]
            R = tcp[:3, :3]
            for name, c, h in gripper_boxes(g["width"]):
                cw = R @ c + tcp[:3, 3]
                for q, cl in obs.items():
                    if obb_hits(cw, R, h, cl, GRIP_TOL).any():
                        hit.add(q)
                if dz == 0.0:
                    own = self.world_cells[pid]
                    if obb_hits(cw, R, h, own, OWN_TOL if name != "palm" else GRIP_TOL).any():
                        hit.add("self")
        return hit

    def _pair_hits(self, pid):
        """Per grasp: every obstacle it hits with ALL other parts installed.

        A gripper box hitting part q does not depend on which other parts are present,
        so the hit set for any installed set S is this set restricted to S + fixed."""
        if not hasattr(self, "_hits"):
            self._hits = {}
        if pid not in self._hits:
            others = [q for q in self.ids if q != pid]
            self._hits[pid] = [self.grasp_blockers(pid, g, others) for g in self.grasps(pid)]
        return self._hits[pid]

    def grasp_report(self, pid, installed):
        """(usable grasp count, Counter of blockers over IK-feasible grasps)."""
        ok, blockers = 0, Counter()
        ik = self._ik.get(pid)
        present = set(installed) | {"table", "base", "self"}
        for i, hits in enumerate(self._pair_hits(pid)):
            if ik is not None and len(ik) and not ik[i]:
                continue
            b = hits & present
            if b:
                blockers.update(b)
            else:
                ok += 1
        return ok, blockers

    def stable_rule(self, installed):
        s = set(installed) | {"base"}
        return all(set(self.by[p].supports) <= s for p in installed)


# ---------------------------------------------------------------------------------------
def explore(model: Model, level: str):
    """Exact state graph. level: 'static' | 'insertion' | 'full'."""
    ids = model.ids
    start = frozenset()
    trans, seen, frontier = {}, {start}, [start]
    info = {}
    while frontier:
        S = frontier.pop()
        trans[S] = []
        for p in ids:
            if p in S:
                continue
            T = S | {p}
            if not model.stable_rule(T):
                continue
            if level in ("insertion", "full") and model.insertion_blockers(p, S):
                continue
            if level == "full":
                n, blk = model.grasp_report(p, S)
                info[S, p] = (n, blk)
                if n == 0:
                    continue
            trans[S].append(p)
            if T not in seen:
                seen.add(T)
                frontier.append(T)
    full = frozenset(ids)
    paths = {}

    def count(S):
        if S == full:
            return 1
        if S not in paths:
            paths[S] = sum(count(S | {p}) for p in trans.get(S, []))
        return paths[S]

    n_orders = count(start)
    valid = {S for S in seen if (S == full or paths.get(S, 0) > 0)}
    before = {}
    for a, b in itertools.permutations(ids, 2):
        before[a, b] = not any(b in S and a not in S for S in valid)
    return dict(states=valid, trans=trans, paths=paths, orders=n_orders, before=before,
                info=info, n_states=len(valid),
                n_transitions=sum(1 for S in valid for p in trans.get(S, []) if S | {p} in valid))


def sample_orders(model, g, rng, k=6):
    full, out, tries = frozenset(model.ids), [], 0
    while len(out) < k and tries < 200 and g["orders"]:
        tries += 1
        S, order = frozenset(), []
        while S != full:
            opts = [p for p in g["trans"][S] if S | {p} == full or g["paths"].get(S | {p}, 0) > 0]
            w = np.array([1 if S | {p} == full else g["paths"][S | {p}] for p in opts], float)
            p = opts[int(rng.choice(len(opts), p=w / w.sum()))]
            order.append(p)
            S = S | {p}
        if order not in out:
            out.append(order)
    return out


def implied_precedences(model, g):
    pairs = [(a, b) for (a, b), v in g["before"].items() if v]
    # keep only the transitive reduction for readability
    red = []
    for a, b in pairs:
        if not any(g["before"].get((a, c)) and g["before"].get((c, b)) for c in model.ids
                   if c not in (a, b)):
            red.append((a, b))
    return pairs, red


# ---------------------------------------------------------------------------------------
def verify_equilibrium(model, states, limit=None):
    """asmcore static equilibrium (no auxiliary support) on reachable states."""
    from asmcore.model import Assembly, AssemblyState, MeshData, Part
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
    from tasks.bridge import box_mesh, pose

    parts = [Part("table", box_mesh(*TABLE_SIZE), pose(*TABLE_CENTER), mass_kg=40.0,
                  com_local_m=np.zeros(3), friction=0.5, fixed=True)]
    for p in model.all:
        lo, hi = model.bounds(p.pid)
        c = model.local_cells[p.pid]
        vol = np.prod(c[:, 1::2] - c[:, 0::2], axis=1)
        com = ((c[:, 0::2] + c[:, 1::2]) / 2 * vol[:, None]).sum(0) / vol.sum()
        parts.append(Part(p.pid, model.mesh(p.pid), model.goal[p.pid], mass_kg=model.mass(p.pid),
                          com_local_m=com, friction=0.5, fixed=p.pid == "base"))
    asm = Assembly(tuple(parts))
    ev = SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0))
    fixed = {"table": parts[0].assembled_tf, "base": model.goal["base"]}
    rows, t0 = [], perf_counter()
    for i, S in enumerate(sorted(states, key=len)):
        if limit and i >= limit:
            break
        st = AssemblyState(dict(fixed, **{p: model.goal[p] for p in S}))
        r = ev.equilibrium(st, ())
        rows.append((sorted(S), r.status))
    bad = [r for r in rows if r[1] != "feasible"]
    return dict(checked=len(rows), infeasible=len(bad), examples=bad[:5],
                seconds=perf_counter() - t0, total_states=len(states))


# ---------------------------------------------------------------------------------------
def pack_layout(model, rng=None, gap=0.015):
    """Greedy non-overlapping initial layout on the table (largest first), for display."""
    from shapely.geometry import box as sbox
    lo_t = TABLE_CENTER[:2] - TABLE_SIZE[:2] / 2 + 0.02
    hi_t = TABLE_CENTER[:2] + TABLE_SIZE[:2] / 2 - 0.02
    blo, bhi = model.bounds("base")
    forbidden = [sbox(-0.13, -0.13, 0.13, 0.13),                         # robot base keep-out
                 sbox(*(SITE[:2] + blo[:2] - 0.03), *(SITE[:2] + bhi[:2] + 0.03))]
    placed, layout = [], {}
    order = sorted(model.ids, key=lambda p: -np.prod(np.ptp(np.c_[model.bounds(p)], 1)[:2]))
    xs = np.arange(lo_t[0], hi_t[0], 0.005)
    ys = np.arange(lo_t[1], hi_t[1], 0.005)
    grid = sorted(((x, y) for x in xs for y in ys),
                  key=lambda q: (abs(q[1]) < 0.12, -abs(q[1]), q[0]))
    for pid in order:
        lo, hi = model.bounds(pid)
        done = False
        for yaw in (0, 90):
            sx, sy = (hi - lo)[:2] if yaw == 0 else (hi - lo)[1::-1]
            for x, y in grid:
                fp = sbox(x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2)
                if fp.bounds[0] < lo_t[0] or fp.bounds[1] < lo_t[1] or \
                        fp.bounds[2] > hi_t[0] or fp.bounds[3] > hi_t[1]:
                    continue
                if any(fp.intersects(f) for f in forbidden) or \
                        any(fp.distance(o) < gap for o in placed):
                    continue
                T = np.eye(4)
                T[:3, :3] = _rz(yaw)
                c = (lo + hi) / 2
                T[:3, 3] = [x, y, -lo[2]] - T[:3, :3] @ np.array([c[0], c[1], 0])
                layout[pid] = T
                placed.append(fp)
                done = True
                break
            if done:
                break
        if not done:
            raise RuntimeError(f"no room on the table for {pid}")
    return layout


# ---------------------------------------------------------------------------------------
def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def robot_meshes():
    """Panthera links at HOME (world frame), decimated for the browser."""
    import trimesh
    from curobo_bridge.export_model import make_wrs_arm, wrs_collision_parts
    arm = make_wrs_arm()
    arm.goto_given_conf(np.zeros(6))
    out = []
    for name, _, link in wrs_collision_parts(arm, include_base=True):
        src = link.cmodel.trm_mesh
        m = trimesh.Trimesh(src.vertices, src.faces, process=False)
        if len(m.faces) > 6000:
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                           o3d.utility.Vector3iVector(m.faces))
            om = om.simplify_quadric_decimation(4000)
            m = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
        m.apply_transform(np.asarray(link.gl_homomat))
        out.append(dict(name=name, v=_b64(m.vertices, "<f4"), f=_b64(m.faces, "<u4")))
    return out


def write_outputs(model: Model, out: Path, stats: dict, layout: dict, orders: list,
                  step_info: dict, include_robot=True):
    import trimesh
    out.mkdir(parents=True, exist_ok=True)
    (out / "stl_m").mkdir(exist_ok=True)
    meshes = {}
    for p in model.all:
        md = model.mesh(p.pid)
        tm = trimesh.Trimesh(md.vertices, md.faces, process=False)
        assert tm.is_watertight and tm.is_winding_consistent and tm.volume > 0, p.pid
        tm.export(out / "stl_m" / f"{p.pid}.stl")
        meshes[p.pid] = dict(v=_b64(md.vertices, "<f4"), f=_b64(md.faces, "<u4"))
    names = {p.pid: p.name for p in model.all}
    data = dict(
        title=model.spec.title, name=model.spec.name, description=model.spec.description,
        table=dict(center=TABLE_CENTER.tolist(), size=TABLE_SIZE.tolist()),
        site=SITE.tolist(),
        parts=[dict(pid=p.pid, name=p.name, role=p.role, color=list(p.color),
                    supports=list(p.supports), goal=model.goal[p.pid].tolist(),
                    stage=layout.get(p.pid, model.goal[p.pid]).tolist(),
                    size_mm=[round(1000 * s, 1) for s in np.ptp(np.c_[model.bounds(p.pid)], 1)],
                    mass_g=round(1000 * model.mass(p.pid), 1), **meshes[p.pid]) for p in model.all],
        robot=robot_meshes() if include_robot else [],
        stats=stats, orders=orders, steps=step_info, names=names)
    html = VIEWER.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    (out / "viewer.html").write_text(html, encoding="utf-8")
    preview_png(model, layout, out / "preview.png")


def preview_png(model, layout, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(17, 6.2), layout="constrained")
    eye = np.array([math.cos(math.radians(-58)) * math.cos(math.radians(28)),
                    math.sin(math.radians(-58)) * math.cos(math.radians(28)),
                    math.sin(math.radians(28))])
    light = np.array([-0.4, -0.6, 1.0])
    light /= np.linalg.norm(light)
    zmax = max(model.world_cells[p][:, 5].max() for p in model.ids) - SITE[2]
    for idx, exploded in enumerate((False, True), 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        tris_all, cols_all, depth = [], [], []
        heights = {p: model.world_cells[p][:, 4].min() for p in model.ids}
        rank = {p: i for i, p in enumerate(sorted(model.ids, key=lambda q: heights[q]))}
        for p in model.all:
            md = model.mesh(p.pid)
            v = md.vertices + model.goal[p.pid][:3, 3] - SITE
            if exploded and p.pid != "base":
                v = v + np.array([0, 0, 0.03 + 0.022 * rank[p.pid]])
            tri = v[md.faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1)[:, None]
            vis = n @ eye > 1e-9
            shade = 0.70 + 0.30 * np.maximum(n @ light, 0)
            tris_all.append(tri[vis] * 1000)
            cols_all.append(np.array(p.color)[None] * shade[vis, None])
            depth.append(tri[vis].mean(1) @ eye)
        tri = np.concatenate(tris_all)
        col = np.concatenate(cols_all)
        order = np.argsort(np.concatenate(depth))
        ax.add_collection3d(Poly3DCollection(tri[order], facecolor=col[order], edgecolor=(0, 0, 0, .15),
                                             linewidths=.2))
        lo, hi = model.bounds("base")
        r = max(hi[0] - lo[0], hi[1] - lo[1]) * 500 + 10
        top = zmax * 1000 + (0.03 + 0.022 * len(model.ids)) * 1000 * exploded + 15
        ax.set(xlim=(-r, r), ylim=(-r, r), zlim=(0, top), xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
        ax.set_box_aspect((1, 1, top / (2 * r)))
        ax.view_init(elev=28, azim=-58)
        ax.set_title("Exploded" if exploded else "Assembled", fontsize=14)
    ax = fig.add_subplot(1, 3, 3)
    t0 = TABLE_CENTER[:2] - TABLE_SIZE[:2] / 2
    ax.add_patch(Rectangle(t0 * 1000, *(TABLE_SIZE[:2] * 1000), fc="#efe8dc", ec="#8a7a62"))
    ax.add_patch(Rectangle((-60, -60), 120, 120, fc="#cfd6de", ec="#66727f"))
    ax.text(0, 0, "robot", ha="center", va="center", fontsize=9)
    for p in model.all:
        T = model.goal[p.pid] if p.pid == "base" else layout[p.pid]
        c = model.local_cells[p.pid]
        corners = np.array([[c[:, 0].min(), c[:, 2].min(), 0], [c[:, 1].max(), c[:, 2].min(), 0],
                            [c[:, 1].max(), c[:, 3].max(), 0], [c[:, 0].min(), c[:, 3].max(), 0]])
        w = (corners @ T[:3, :3].T + T[:3, 3])[:, :2] * 1000
        ax.fill(w[:, 0], w[:, 1], color=p.color, alpha=.9, ec="k", lw=.5)
        ax.text(*w.mean(0), p.pid, ha="center", va="center", fontsize=7)
    ax.set(xlim=(t0[0] * 1000 - 10, (t0[0] + TABLE_SIZE[0]) * 1000 + 10),
           ylim=(t0[1] * 1000 - 10, (t0[1] + TABLE_SIZE[1]) * 1000 + 10), aspect="equal",
           xlabel="X (mm)", ylabel="Y (mm)", title="Example initial layout (top view)")
    fig.legend(handles=[Patch(facecolor=p.color, label=p.pid) for p in model.spec.parts],
               loc="lower center", ncol=min(12, len(model.ids)), bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------
def analyse(spec: ModelSpec, out: Path, *, use_ik=True, seed=0, equilibrium=True):
    """Full pipeline for one model; returns the stats dict."""
    from curobo_bridge.runtime import prepare
    prepare()
    t_all = perf_counter()
    model = Model(spec)
    # final state must be free of interference
    for a, b in itertools.combinations([p.pid for p in model.all], 2):
        if aabb_hits(model.world_cells[a], model.world_cells[b], PART_TOL).any():
            raise RuntimeError(f"parts {a} and {b} interpenetrate at the goal")
    ik_s = 0.0
    if use_ik:
        from curobo_bridge.assembly_demo import BatchIK
        t0 = perf_counter()
        model.goal_ik(BatchIK(batch=512, seeds=32, returns=4))
        ik_s = perf_counter() - t0
    graphs, times = {}, {}
    for level in ("static", "insertion", "full"):
        t0 = perf_counter()
        graphs[level] = explore(model, level)
        times[level] = perf_counter() - t0
        print(level, "states", graphs[level]["n_states"], "orders", graphs[level]["orders"],
              f"{times[level]:.1f}s", flush=True)
    full = graphs["full"]
    if full["orders"] == 0:
        print("WARNING: no feasible order under the full model", flush=True)
    rng = np.random.default_rng(seed)
    orders = sample_orders(model, full, rng)
    prec_full, red_full = implied_precedences(model, full)
    prec_static, _ = implied_precedences(model, graphs["static"])
    from_occlusion = sorted(set(prec_full) - set(prec_static))
    # occlusion events on transitions that are statically and insertion-wise valid
    alone = {p: model.grasp_report(p, [])[0] for p in model.ids}
    partial = blocked = 0
    blocker_pairs = Counter()
    for (S, p), (n, blk) in full["info"].items():
        if n == 0:
            blocked += 1
        elif n < alone[p]:
            partial += 1
        for q in blk:
            if q in model.ids:
                blocker_pairs[f"{q} -> {p}"] += 1
    insertion_pairs = Counter()
    for S in graphs["static"]["states"]:
        for p in graphs["static"]["trans"].get(S, []):
            for q in model.insertion_blockers(p, S):
                if q in model.ids:
                    insertion_pairs[f"{q} covers {p}"] += 1
    step_info = {}
    for oi, order in enumerate(orders):
        rows = []
        for k, p in enumerate(order):
            n, blk = model.grasp_report(p, order[:k])
            rows.append(dict(part=p, grasps=n, grasps_alone=alone[p],
                             blockers=sorted(q for q in blk if q in model.ids)))
        step_info[str(oi)] = rows
    eq = verify_equilibrium(model, graphs["static"]["states"]) if equilibrium else None
    stats = dict(
        parts=len(model.ids),
        grasp_catalogue={p: len(model.grasps(p)) for p in model.ids},
        goal_ik_grasps={p: int(model._ik[p].sum()) if p in model._ik else None for p in model.ids},
        grasps_alone=alone,
        levels={lv: dict(states=g["n_states"], transitions=g["n_transitions"], orders=g["orders"],
                         seconds=times[lv]) for lv, g in graphs.items()},
        precedence_full=[list(x) for x in red_full],
        precedence_from_occlusion=[list(x) for x in from_occlusion],
        grasp_occlusion=dict(transitions_checked=len(full["info"]), partially_occluded=partial,
                             fully_blocked=blocked, blocker_pairs=dict(blocker_pairs.most_common(20))),
        insertion_cover=dict(insertion_pairs.most_common(20)),
        equilibrium_check=eq, ik_s=ik_s, total_s=perf_counter() - t_all,
        scope="static rule + asmcore equilibrium cross-check; box-cell insertion sweep; "
              "Panthera palm/finger boxes + cuRobo goal IK. No transport motion, no staging-side IK.")
    layout = pack_layout(model)
    write_outputs(model, out, stats, layout, orders, step_info)
    task = dict(name=spec.name, units="metre", site=SITE.tolist(), table_center=TABLE_CENTER.tolist(),
                table_size=TABLE_SIZE.tolist(), density_kg_m3=DENSITY,
                parts={p.pid: dict(name=p.name, role=p.role, fixed=p.pid == "base",
                                   goal_tf=model.goal[p.pid].tolist(), supports=list(p.supports),
                                   mass_kg=model.mass(p.pid), stl=f"stl_m/{p.pid}.stl",
                                   size_m=np.ptp(np.c_[model.bounds(p.pid)], 1).tolist())
                       for p in model.all},
                example_layout={p: T.tolist() for p, T in layout.items()}, example_orders=orders)
    (out / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2,
                                                    default=str), encoding="utf-8")
    return stats


VIEWER = (Path(__file__).with_name("viewer_template.html")).read_text(encoding="utf-8") \
    if Path(__file__).with_name("viewer_template.html").exists() else "__DATA__"
