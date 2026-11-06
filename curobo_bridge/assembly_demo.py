"""随机合法布局 + 逐件完整装配：分级 cuRobo 规划器 vs WRS RRT。

    & $py -X utf8 -m curobo_bridge.assembly_demo --seed 7

装配体：sealp ``topdown_tower``（底板预装在装配位；4 根柱子插底板孔 → 中板压到柱顶 →
顶部十字件插中板方孔），顺序取 asmdef 顺序。Panthera 基座在原点，装配位 (0.40, 0, 0)（见 ``SITE`` 注释）。

每件零件的动作链（``start`` = HOME 或上一件的撤离位形）::

    A  start ─规划─▶ 预抓取            （空手）
       预抓取 ─直线─▶ 抓取 ─闭合─▶ 抬升（沿世界 +Z）
    B  抬升 ─规划─▶ 预插入            （持件）
       预插入 ─直线─▶ 插入 ─张开─▶ 撤离（沿工具 −Z）
    最后：撤离 ─规划─▶ HOME

只有 A、B 和最后一段由规划器决定；直线段、抓取、夹爪动作对两种规划器完全相同。

公平性约定：
* 抓取与关键位形只筛一次、两种规划器共用（cuRobo 批量 IK 求解，WRS 网格真值复核）。
* 两种规划器的输出都用 WRS ODE 网格（含持件、接触豁免）复核；RRT 复核失败换随机种子
  重规划（最多 3 次），耗时累计。
* RRT 用 WRS 的 AABB 图元语义（桌面、垫块、已装件、待装件），起终点检测跳过（它们已由
  网格真值验证）。sealp 的 staging_aware 模式会把桌面排除，实测会让 RRT 路径穿桌面、全部被复核
  拒绝，所以这里保留桌面。复核失败换随机种子，最多 5 次，耗时累计。
* 执行时间两种口径：
  - ``same_rule``（主结果）：两种规划器的几何路径用同一个数值时间最优参数化 :func:`topp`
    （同一关节速度/加速度上限，无加加速度），RRT 折线在路点处必须停住；
  - ``native``：cuRobo 用它自己的轨迹时长（含 jerk 300 rad/s^3、时间步量化，偏保守）；
    RRT 用"每个路点停一下"的关节同步梯形速度。
  直线段按 ``topp`` 计时（两者相同），夹爪开/合各 0.5 s。
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, REPO_ROOT, prepare

#: Assembly site. Scanned x in 0.24..0.40: at the table height only x=0.40 leaves the
#: middle plate (side grasps only) enough goal-side IK solutions (63/520; 9 at x=0.32).
SITE = np.array([0.40, 0.0, 0.0])
#: Fixed staging riser for parts that only have side grasps (the middle plate): lying
#: flat on the table the fingers would go 7 mm into the table; on a 40 mm riser the
#: 158 mm palm and the elbow still hit the table on the horizontal approach. sealp sidesteps this by
#: skipping the middle plate in L3 validation (DEFAULT_L3_SKIP_PARTS); here it is planned.
from .scenes import TABLE_CENTRE, TABLE_SIZE   # noqa: E402  (rebound per example in main)
RISER = np.array([0.07, 0.05, 0.10])   # 100 mm pedestal: palm is 158 mm wide
HOME = np.zeros(6)
APPROACH, LIFT, PREINS, RETREAT = 0.05, 0.06, 0.05, 0.05
VMAX = np.array([4.2, 5.0, 5.0, 4.2, 3.7, 3.7])
AMAX = 10.0
GRIPPER_S = 0.5
LIN_STEP = 0.02           # rad, joint interpolation resolution for straight segments
LIN_DEVIATION = 0.005     # m, max TCP deviation from the Cartesian line
ASMDEF = REPO_ROOT / "sealp/assembly_sequence/_demo_output/topdown_tower.asmdef"
GRASP_DIR = REPO_ROOT / "sealp/examples/grasp/tower_grasp"
GRASP_FILE = {"base_plate": "tower_base_plate", "middle_plate": "tower_middle_plate",
              "top_cross": "tower_top_cross"}


def _h(pos=np.zeros(3), rot=np.eye(3)):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, pos
    return out


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def t_trap(d, v, a=AMAX):
    d = np.abs(d)
    return np.where(d < v * v / a, 2 * np.sqrt(d / a), d / v + v / a)


def waypoint_stop_time(path) -> float:
    """Execution time of a joint polyline stopping at every waypoint (synchronised trapezoid)."""
    path = np.asarray(path, float)
    return float(sum(np.max(t_trap(b - a, VMAX)) for a, b in zip(path[:-1], path[1:])))


def topp(path, stops=(), step=0.01):
    """Numerical time-optimal parameterisation of a joint path under per-joint velocity
    (``VMAX``) and acceleration (``AMAX``) limits; zero speed at both ends and at ``stops``
    (indices into ``path``, e.g. RRT polyline vertices). Curvature enters through the
    speed cap sqrt(a/|q''|). Returns (dense points [M,6], timestamps [M])."""
    path = np.asarray(path, float)
    pts, stop_mask = [path[0]], [True]
    stops = set(int(i) for i in stops)
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            pts.append(a + t * (b - a))
            stop_mask.append(False)
        stop_mask[-1] = (i + 1) in stops
    stop_mask[-1] = True
    pts = np.asarray(pts)
    d = np.diff(pts, axis=0)
    ds = np.linalg.norm(d, axis=1)
    keep = np.r_[True, ds > 1e-9]
    pts, stop_mask = pts[keep], np.asarray(stop_mask)[keep]
    if len(pts) < 2:
        return pts, np.zeros(len(pts))
    d = np.diff(pts, axis=0)
    ds = np.linalg.norm(d, axis=1)
    u = d / ds[:, None]                                   # unit direction per segment
    n = len(pts)
    vcap = np.full(n, np.inf)
    for k in range(n - 1):                                # velocity limit on both ends of segment
        c = np.min(VMAX / np.maximum(np.abs(u[k]), 1e-12))
        vcap[k] = min(vcap[k], c)
        vcap[k + 1] = min(vcap[k + 1], c)
    for i in range(1, n - 1):                             # curvature limit
        kappa = np.abs(u[i] - u[i - 1]) / (0.5 * (ds[i - 1] + ds[i]))
        vcap[i] = min(vcap[i], np.sqrt(np.min(AMAX / np.maximum(kappa, 1e-12))))
    vcap[stop_mask] = 0.0
    acc = np.array([np.min(AMAX / np.maximum(np.abs(u[k]), 1e-12)) for k in range(n - 1)])
    w = vcap.copy()
    w[0] = 0.0
    for k in range(n - 1):
        w[k + 1] = min(w[k + 1], np.sqrt(w[k] ** 2 + 2 * acc[k] * ds[k]))
    for k in range(n - 2, -1, -1):
        w[k] = min(w[k], np.sqrt(w[k + 1] ** 2 + 2 * acc[k] * ds[k]))
    dt = np.where(w[:-1] + w[1:] > 1e-9, 2 * ds / np.maximum(w[:-1] + w[1:], 1e-12),
                  2 * np.sqrt(ds / acc))
    return pts, np.r_[0.0, np.cumsum(dt)]


def lin_interp(a, b, step=LIN_STEP):
    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
    return np.array([a + t * (b - a) for t in np.linspace(0, 1, n + 1)])


# ---------------------------------------------------------------------------
@dataclass
class TowerPart:
    pid: str
    mesh_path: Path
    mesh: object          # trimesh, object frame (reference geometry)
    cu_mesh: object       # watertight version for cuRobo (envelope if needed)
    grasps: list          # [(G 4x4 TCP-in-object, jaw)]
    goal: np.ndarray


def load_tower():
    import trimesh
    from sealp.assembly_sequence.asmdef import AssemblyDef
    from wrs.grasping.grasp import GraspCollection
    from .export_model import conservative_envelope

    asm = AssemblyDef.load(str(ASMDEF))
    wp = asm.compute_world_poses(fixture_pos=SITE)
    parts = {}
    for step in asm.steps:
        pid = step.part_id
        path = Path(asm.model_path(pid))
        mesh = trimesh.load(path, process=True)
        cu = mesh if mesh.is_watertight else conservative_envelope(mesh, pitch=0.001, max_faces=3000)
        gfile = GRASP_FILE.get(pid, "tower_post")
        gc = GraspCollection.load_from_disk(str(GRASP_DIR / f"{gfile}_grasps.pickle"))
        grasps = [(_h(g.ac_pos, g.ac_rotmat), float(np.ravel(g.ee_values)[0])) for g in gc]
        p, r = wp[pid]
        parts[pid] = TowerPart(pid, path, mesh, cu, grasps, _h(p, r))
    order = [s.part_id for s in asm.steps]
    return parts, order


def mating_sets(parts, order, tol=0.002):
    """part -> earlier parts whose goal surfaces are within ``tol`` (insertion contacts)."""
    import trimesh
    out = {}
    for k, pid in enumerate(order):
        m = parts[pid].mesh.copy()
        m.apply_transform(parts[pid].goal)
        pts = m.sample(3000, seed=1)
        out[pid] = []
        for q in order[:k]:
            o = parts[q].mesh.copy()
            o.apply_transform(parts[q].goal)
            _, d, _ = trimesh.proximity.closest_point(o, pts)
            if d.min() < tol:
                out[pid].append(q)
    return out


# ---------------------------------------------------------------------------
def needs_riser(part) -> bool:
    return not any(G[2, 2] < -0.5 for G, _ in part.grasps)


def sample_layout(parts, order, rng, feasible=None):
    """Random collision-free staging on the table; ``feasible(pid, T)`` optionally
    rejects stagings with no IK-consistent common grasp (cheap GPU prefilter)."""
    from shapely.geometry import MultiPoint, Point, box

    lo = TABLE_CENTRE[:2] - TABLE_SIZE[:2] / 2 + 0.01
    hi = TABLE_CENTRE[:2] + TABLE_SIZE[:2] / 2 - 0.01
    table = box(lo[0], lo[1], hi[0], hi[1])
    base = parts[order[0]]
    bv = base.mesh.vertices @ base.goal[:3, :3].T + base.goal[:3, 3]
    forbid = [MultiPoint(bv[:, :2]).convex_hull.buffer(0.04), Point(0, 0).buffer(0.13)]
    layout, polys = {}, []
    for pid in order[1:]:
        m = parts[pid].mesh
        for _ in range(500):
            yaw = rng.uniform(-np.pi, np.pi)
            rot = _rz(yaw) @ parts[pid].goal[:3, :3]
            v = m.vertices @ rot.T
            xy = rng.uniform(lo, hi)
            poly = MultiPoint(v[:, :2] + xy).convex_hull
            if not table.contains(poly) or any(poly.intersects(f) for f in forbid):
                continue
            if any(poly.distance(p) < 0.045 for p in polys):
                continue
            z0 = RISER[2] if needs_riser(parts[pid]) else 0.0
            T = _h(np.array([xy[0], xy[1], z0 - v[:, 2].min() + 2e-4]), rot)
            if feasible is not None and not feasible(pid, T):
                continue
            layout[pid] = T
            polys.append(poly)
            break
        else:
            return None
    return layout


# ---------------------------------------------------------------------------
class World:
    """WRS collision models for every part at goal and at staging (shared by both planners)."""

    def __init__(self, parts, order, layout):
        import wrs.modeling.collision_model as mcm
        self.parts, self.order, self.layout = parts, order, layout
        self.table = mcm.gen_box(xyz_lengths=TABLE_SIZE, pos=TABLE_CENTRE)
        self.goal_cm, self.stage_cm, self.held_cm = {}, {}, {}
        self.risers = {}
        for pid, T in layout.items():
            if needs_riser(parts[pid]):
                pos = np.array([T[0, 3], T[1, 3], RISER[2] / 2])
                self.risers[pid] = (mcm.gen_box(xyz_lengths=RISER, pos=pos, rotmat=T[:3, :3]),
                                    _h(pos, T[:3, :3]))
        for pid, p in parts.items():
            for store, pose in ((self.goal_cm, p.goal), (self.stage_cm, layout.get(pid))):
                if pose is None:
                    continue
                cm = mcm.CollisionModel(initor=str(p.mesh_path), name=pid)
                cm.pose = (pose[:3, 3].copy(), pose[:3, :3].copy())
                store[pid] = cm
            self.held_cm[pid] = mcm.CollisionModel(initor=str(p.mesh_path), name=f"held_{pid}")

    def objects(self, assembled, staged, table=True):
        out = [self.table] if table else []
        out += [r[0] for r in self.risers.values()]
        out += [self.goal_cm[p] for p in assembled] + [self.stage_cm[p] for p in staged]
        return out

    def curobo_scene(self, assembled, staged):
        from .scenes import pose7
        d = {"cuboid": {"table": dict(dims=TABLE_SIZE.tolist(), pose=pose7(_h(TABLE_CENTRE)))},
             "mesh": {}}
        for pid, (_, pose) in self.risers.items():
            d["cuboid"][f"riser_{pid}"] = dict(dims=RISER.tolist(), pose=pose7(pose))
        for tag, pids, poses in (("goal", assembled, {p: self.parts[p].goal for p in assembled}),
                                 ("stage", staged, self.layout)):
            for p in pids:
                m = self.parts[p].cu_mesh
                d["mesh"][f"{tag}_{p}"] = dict(vertices=np.asarray(m.vertices).tolist(),
                                               faces=np.asarray(m.faces).reshape(-1).tolist(),
                                               pose=pose7(poses[p]))
        return d


class MeshOracle:
    """Geometric truth: WRS ODE triangle-mesh checks incl. held part and exemptions."""

    def __init__(self, arm, world):
        from .bench_self_collision import checked_pairs
        from .curobo_models import load_robot_dict
        from .export_model import wrs_collision_parts, FINGER_LINKS
        self.arm, self.world = arm, world
        self.links = {n: l for n, _, l in wrs_collision_parts(arm, False)}
        self.fingers = set(FINGER_LINKS)
        self.pairs = checked_pairs(load_robot_dict(ASSET_DIR / "panthera_wrsext", 1.0))
        self.arm_only = [self.links[n].cmodel for n in ("link2", "link3", "link4", "link5")]

    def tcp(self, q):
        p, r = self.arm.fk(q)
        return _h(p, r)

    def state_ok(self, q, obstacles, finger_exempt=(), held=None, held_exempt=()):
        """held = (pid, G). Returns None if collision-free, else a reason string."""
        self.arm.goto_given_conf(q)
        for n, l in self.links.items():
            obs = [o for o in obstacles if not (n in self.fingers and o in finger_exempt)]
            if l.cmodel.is_mcdwith(obs):
                return f"arm:{n}"
        for a, b in self.pairs:
            if self.links[a].cmodel.is_mcdwith([self.links[b].cmodel]):
                return f"self:{a}-{b}"
        if held is not None:
            pid, G = held
            cm = self.world.held_cm[pid]
            pose = self.tcp(q) @ np.linalg.inv(G)
            cm.pose = (pose[:3, 3].copy(), pose[:3, :3].copy())
            if cm.is_mcdwith([o for o in obstacles if o not in held_exempt]):
                return "held:scene"
            if cm.is_mcdwith(self.arm_only):
                return "held:arm"
        return None

    def path_ok(self, path, obstacles, step=0.01, **kw):
        from .planner import densify
        dense = densify(np.asarray(path, float), step)
        hits = [i for i, q in enumerate(dense) if self.state_ok(q, obstacles, **kw)]
        self.arm.goto_given_conf(HOME)
        return dict(ok=not hits, states=len(dense), hits=len(hits))


# ---------------------------------------------------------------------------
class BatchIK:
    def __init__(self, batch=512, seeds=32, returns=4):
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from .curobo_models import load_robot_dict, with_buffer
        robot = with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", 1.0), 0.0)
        self.batch, self.returns = batch, returns
        self.ik = InverseKinematics(InverseKinematicsCfg.create(
            robot=robot, num_seeds=seeds, max_batch_size=batch, position_tolerance=5e-4,
            orientation_tolerance=5e-3, self_collision_check=True, random_seed=0))

    def solve(self, poses):
        """poses [N,4,4] -> q [N,R,6], ok [N,R]"""
        import torch
        from curobo.types import GoalToolPose, Pose
        from scipy.spatial.transform import Rotation
        n = len(poses)
        qs, oks = [], []
        for s in range(0, n, self.batch):
            chunk = poses[s:s + self.batch]
            m = len(chunk)
            pad = np.concatenate([chunk, np.repeat(chunk[-1:], self.batch - m, 0)])
            xyzw = Rotation.from_matrix(pad[:, :3, :3]).as_quat()
            pose = Pose(position=torch.as_tensor(pad[:, :3, 3], dtype=torch.float32, device="cuda"),
                        quaternion=torch.as_tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32,
                                                   device="cuda"))
            goal = GoalToolPose.from_poses({"wrs_tcp": pose.unsqueeze(1)},
                                           ordered_tool_frames=["wrs_tcp"])
            r = self.ik.solve_pose(goal_tool_poses=goal, return_seeds=self.returns)
            qs.append(r.solution.reshape(self.batch, -1, 6)[:m, :self.returns].cpu().numpy())
            oks.append(r.success.reshape(self.batch, -1)[:m, :self.returns].cpu().numpy())
        return np.concatenate(qs), np.concatenate(oks)


@dataclass
class StepPlan:
    pid: str
    grasp_id: int
    G: np.ndarray
    jaw: float
    q: dict                      # keypose name -> q
    screening: dict = field(default_factory=dict)


KEYS = ("prepick", "pick", "lift", "preins", "ins", "retreat")


def keyposes(part, stage, G):
    pick = stage @ G
    z = pick[:3, 2]
    prepick = pick.copy(); prepick[:3, 3] -= APPROACH * z
    lift = pick.copy(); lift[2, 3] += LIFT
    ins = part.goal @ G
    preins = ins.copy(); preins[2, 3] += PREINS
    retreat = ins.copy(); retreat[:3, 3] -= RETREAT * ins[:3, 2]
    return dict(prepick=prepick, pick=pick, lift=lift, preins=preins, ins=ins, retreat=retreat)


def screen_grasps(k, parts, order, world, oracle, bik, mating, max_verify=60,
                  check_jaw_motion=False):
    """Select one grasp + keyposes for step k (shared by both planners)."""
    pid = order[k]
    part = parts[pid]
    stage = world.layout[pid]
    assembled, staged = order[:k], order[k:]
    t0 = time.perf_counter()
    cands = []
    for gi, (G, jaw) in enumerate(part.grasps):
        kp = keyposes(part, stage, G)
        if kp["pick"][2, 2] > 0.3 or kp["ins"][2, 2] > 0.3 or kp["pick"][2, 3] < 0.012:
            continue   # approach from below / fingertip under the table
        cands.append((gi, G, jaw, kp))
    stats = dict(grasps=len(part.grasps), direction_ok=len(cands))
    if not cands:
        return None, stats
    poses = np.stack([c[3][key] for c in cands for key in KEYS])
    q, ok = bik.solve(poses)
    q = q.reshape(len(cands), len(KEYS), -1, 6)
    ok = ok.reshape(len(cands), len(KEYS), -1)
    ranked = []
    for ci, (gi, G, jaw, kp) in enumerate(cands):
        if not ok[ci].any(axis=1).all():
            continue
        best = None
        for a in np.flatnonzero(ok[ci, 1]):          # pick branch
            for b in np.flatnonzero(ok[ci, 4]):      # insert branch
                sel = {"pick": q[ci, 1, a], "ins": q[ci, 4, b]}
                cost, valid = 0.0, True
                for key, ref in (("prepick", "pick"), ("lift", "pick"),
                                 ("preins", "ins"), ("retreat", "ins")):
                    idx = np.flatnonzero(ok[ci, KEYS.index(key)])
                    d = np.abs(q[ci, KEYS.index(key), idx] - sel[ref]).max(axis=1)
                    j = idx[np.argmin(d)]
                    if d.min() > 0.6:
                        valid = False
                        break
                    sel[key] = q[ci, KEYS.index(key), j]
                    cost += d.min()
                if valid:
                    cost += 0.3 * np.abs(sel["preins"] - sel["lift"]).max()
                    if best is None or cost < best[0]:
                        best = (cost, sel)
        if best is not None:
            ranked.append((best[0], gi, G, jaw, best[1], kp))
    stats["ik_consistent"] = len(ranked)
    stats["ik_s"] = time.perf_counter() - t0
    ranked.sort(key=lambda r: r[0])
    t1 = time.perf_counter()
    reasons = {}
    table = world.table
    w_a = world.objects(assembled, staged)
    w_b = world.objects(assembled, staged[1:])
    w_c = world.objects(assembled + [pid], staged[1:])
    mates = [world.goal_cm[m] for m in mating[pid]]
    for n_checked, (cost, gi, G, jaw, sel, kp) in enumerate(ranked[:max_verify], 1):
        held = (pid, G)
        checks = [
            ("prepick", [sel["prepick"]], w_a, dict()),
            ("approach", lin_interp(sel["prepick"], sel["pick"]), w_a,
             dict(finger_exempt=(world.stage_cm[pid],))),
            ("lift", lin_interp(sel["pick"], sel["lift"]), w_b,
             dict(held=held, held_exempt=(table,) + tuple(
                 r[0] for p, r in world.risers.items() if p == pid))),
            ("preins", [sel["preins"]], w_b, dict(held=held)),
            ("insert", lin_interp(sel["preins"], sel["ins"]), w_b,
             dict(held=held, held_exempt=tuple(mates))),
            ("retreat", lin_interp(sel["ins"], sel["retreat"]), w_c,
             dict(finger_exempt=(world.goal_cm[pid],))),
        ]
        fail = None
        for name, states, obs, kw in checks:
            if check_jaw_motion:
                opening = min(float(oracle.arm.end_effector.jaw_range[1]), jaw + 0.008)
                oracle.arm.change_jaw_width(opening if name in ("prepick", "approach", "retreat") else jaw)
            if name in ("approach", "lift", "insert", "retreat"):
                a_, b_ = {"approach": ("prepick", "pick"), "lift": ("pick", "lift"),
                          "insert": ("preins", "ins"), "retreat": ("ins", "retreat")}[name]
                pa, pb = kp[a_][:3, 3], kp[b_][:3, 3]
                for qq in states:
                    x = oracle.tcp(qq)[:3, 3]
                    t = np.clip(np.dot(x - pa, pb - pa) / max(np.dot(pb - pa, pb - pa), 1e-12), 0, 1)
                    if np.linalg.norm(x - (pa + t * (pb - pa))) > LIN_DEVIATION:
                        fail = f"{name}:not_straight"
                        break
                if fail:
                    break
            for qq in states:
                r = oracle.state_ok(qq, obs, **kw)
                if r:
                    fail = f"{name}:{r}"
                    break
            if fail:
                break
        if fail is None and check_jaw_motion:
            for width in np.linspace(jaw, opening, 5):
                oracle.arm.change_jaw_width(float(width))
                for qq, obs, exempt in ((sel["pick"], w_a, world.stage_cm[pid]),
                                        (sel["ins"], w_c, world.goal_cm[pid])):
                    reason = oracle.state_ok(qq, obs, finger_exempt=(exempt,))
                    if reason:
                        fail = f"jaw_motion:{reason}"
                        break
                if fail:
                    break
        oracle.arm.goto_given_conf(HOME)
        if fail is None:
            stats.update(verified=n_checked, verify_s=time.perf_counter() - t1,
                         rejections=reasons, total_s=time.perf_counter() - t0)
            return StepPlan(pid, gi, G, jaw, sel, stats), stats
        reasons[fail] = reasons.get(fail, 0) + 1
    stats.update(verified=min(len(ranked), max_verify), rejections=reasons,
                 verify_s=time.perf_counter() - t1, total_s=time.perf_counter() - t0)
    return None, stats


# ---------------------------------------------------------------------------
def run_segments(planner_name, plans, order, world, oracle, cu, arm, rrt_retries=5):
    """Plan the A/B/final segments for one planner. Returns list of segment dicts."""
    segs = []
    start = HOME
    for k, sp in enumerate(plans):
        pid = sp.pid
        asm_before = order[:order.index(pid)]     # includes the preassembled base
        stg = order[order.index(pid):]            # current part + later parts at staging
        for leg in ("A", "B"):
            if leg == "A":
                a, b, held = start, sp.q["prepick"], None
                cu_scene = world.curobo_scene(asm_before, stg)
                prim = world.objects(asm_before, stg)
                mesh_obs = world.objects(asm_before, stg)
                kw = dict()
            else:
                a, b, held = sp.q["lift"], sp.q["preins"], (pid, sp.G)
                cu_scene = world.curobo_scene(asm_before, stg[1:])
                prim = world.objects(asm_before, stg[1:])
                mesh_obs = world.objects(asm_before, stg[1:])
                kw = dict(held=held)
            seg = plan_one(planner_name, a, b, held, sp, world, oracle, cu, arm,
                           cu_scene, prim, mesh_obs, kw, rrt_retries)
            seg.update(step=k, part=pid, leg=leg)
            segs.append(seg)
        start = sp.q["retreat"]
    done = order
    seg = plan_one(planner_name, start, HOME, None, None, world, oracle, cu, arm,
                   world.curobo_scene(done, []), world.objects(done, []),
                   world.objects(done, []), {}, rrt_retries)
    seg.update(step=len(plans), part="-", leg="home")
    segs.append(seg)
    return segs


def plan_one(name, a, b, held, sp, world, oracle, cu, arm, cu_scene, prim, mesh_obs, kw,
             rrt_retries):
    verify = lambda path: oracle.path_ok(path, mesh_obs, **kw)
    held_pose = None
    if held is not None:
        held_pose = oracle.tcp(a) @ np.linalg.inv(sp.G)
    if name == "curobo":
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cu.set_world(cu_scene, prim, verify_fn=verify)
        if held is not None:
            cm = world.held_cm[sp.pid]
            cm.pose = (held_pose[:3, 3].copy(), held_pose[:3, :3].copy())
            cu.hold(a, world.parts[sp.pid].cu_mesh, held_pose, rrt_cmodel=cm)
        torch.cuda.synchronize()
        world_s = time.perf_counter() - t0      # scene update + attach (counted as planning)
        res = cu.plan(a, b)
        if held is not None:
            cu.release()
        arm.goto_given_conf(HOME)
        out = dict(success=res.success, source=res.source, plan_s=res.plan_s + world_s,
                   world_update_s=world_s, verify_s=res.verify_s,
                   tiers=[list(t) for t in res.tiers],
                   path=None if res.path is None else res.path.tolist())
        if res.success:
            out["exec_s"] = (cu.last_dt * (len(res.path) - 1) if res.source.startswith("curobo")
                             else waypoint_stop_time(res.path))
        return out
    # ---- WRS RRT baseline ----
    import wrs.motion.probabilistic.rrt_connect as rrtc
    import random
    cm = None
    if held is not None:
        cm = world.held_cm[sp.pid]
        arm.goto_given_conf(a)
        cm.pose = (held_pose[:3, 3].copy(), held_pose[:3, :3].copy())
        arm.hold(cm)
    plan_s = verify_s = 0.0
    tiers = []
    path = None
    for attempt in range(rrt_retries):
        np.random.seed(attempt)
        random.seed(attempt)
        t0 = time.perf_counter()
        md = rrtc.RRTConnect(arm).plan(start_conf=np.asarray(a, float), goal_conf=np.asarray(b, float),
                                        obstacle_list=prim, start_obstacle_list=[],
                                        goal_obstacle_list=[], ext_dist=0.30, max_time=10.0,
                                        smoothing_n_iter=150)
        dt = time.perf_counter() - t0
        plan_s += dt
        if md is None:
            tiers.append(["wrs_rrt", "no_path", dt])
            continue
        cand = np.asarray(md.jv_list, float)
        t1 = time.perf_counter()
        if cm is not None:
            arm.release(cm)
        v = verify(cand)
        if cm is not None:
            arm.goto_given_conf(a)
            arm.hold(cm)
        verify_s += time.perf_counter() - t1
        if v["ok"]:
            tiers.append(["wrs_rrt", "accepted", dt])
            path = cand
            break
        tiers.append(["wrs_rrt", f"rejected mesh_hits={v['hits']}", dt])
    if cm is not None:
        arm.release(cm)
    arm.goto_given_conf(HOME)
    out = dict(success=path is not None, source="wrs_rrt" if path is not None else None,
               plan_s=plan_s, verify_s=verify_s, tiers=tiers,
               path=None if path is None else path.tolist())
    if path is not None:
        out["exec_s"] = waypoint_stop_time(path)
    return out


# ---------------------------------------------------------------------------
def fixed_segments(sp):
    """Straight-line + gripper segments shared by both planners (execution time)."""
    q = sp.q
    lin = {name: lin_interp(q[a], q[b]) for name, (a, b) in dict(
        approach=("prepick", "pick"), lift=("pick", "lift"),
        insert=("preins", "ins"), retreat=("ins", "retreat")).items()}
    times = {n: float(np.max(t_trap(p[-1] - p[0], VMAX))) for n, p in lin.items()}
    return lin, times


def main(argv=None) -> None:
    prepare()
    import torch
    from .planner import PantheraPlanner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-layouts", type=int, default=30)
    parser.add_argument("--site-x", type=float, default=None,
                        help="assembly site x; Bridge uses 0.44 so the folded HOME gripper "
                             "(TCP x=0.252, z=0.169) is not parked over the 0.34 m beam")
    args = parser.parse_args(argv)
    global SITE
    if args.site_x is not None:
        SITE = np.array([args.site_x, 0.0, 0.0])
    rng = np.random.default_rng(args.seed)
    t_all = time.perf_counter()
    parts, order = load_tower()
    mating = mating_sets(parts, order)
    from .export_model import make_wrs_arm
    arm = make_wrs_arm()
    bik = BatchIK()

    # ---- random legal layout + shared grasp screening ---------------------------
    goal_ok = {}

    def ik_mask(pid, T, keys):
        part = parts[pid]
        P = np.stack([keyposes(part, T, G)[k] for G, _ in part.grasps for k in keys])
        _, o = bik.solve(P)
        return o.reshape(len(part.grasps), len(keys), -1).any(-1).all(1)

    def feasible(pid, T):
        """Some grasp is IK-reachable at every pick-side AND goal-side keypose."""
        if pid not in goal_ok:
            goal_ok[pid] = ik_mask(pid, np.eye(4), ("preins", "ins", "retreat"))
        return bool((ik_mask(pid, T, ("prepick", "pick", "lift")) & goal_ok[pid]).any())

    layout_log = []
    for attempt in range(1, args.max_layouts + 1):
        layout = sample_layout(parts, order, rng, feasible)
        if layout is None:
            layout_log.append(dict(attempt=attempt, result="no_room"))
            continue
        world = World(parts, order, layout)
        oracle = MeshOracle(arm, world)
        plans, stats = [], []
        for k in range(1, len(order)):
            sp, st = screen_grasps(k, parts, order, world, oracle, bik, mating)
            stats.append(dict(part=order[k], **st))
            if sp is None:
                break
            plans.append(sp)
        layout_log.append(dict(attempt=attempt, result="ok" if len(plans) == len(order) - 1
                               else f"no_grasp:{order[len(plans) + 1]}", screening=stats))
        print(f"layout {attempt}: {layout_log[-1]['result']}", flush=True)
        if len(plans) == len(order) - 1:
            break
    else:
        raise SystemExit("no legal layout found")
    screening_s = sum(s.get("total_s", 0.0) for s in stats)

    # ---- plan with both planners --------------------------------------------------
    cu = PantheraPlanner(arm=arm, attach_spheres=32)
    cu.set_world(world.curobo_scene(order[:1], order[1:]), [])
    cu._planner("curobo")
    cu._planner("curobo_relaxed")
    cu.last_dt = cu._mp["curobo"].trajopt_solver.config.interpolation_dt
    # held-part sphere fitting: once per part geometry, before timing (preprocessing)
    cu.setup_s["held_sphere_fit"] = sum(cu.prefit(parts[sp.pid].cu_mesh) for sp in plans)
    # warm the full planning path once (attach / world update) outside the timing
    results = {}
    for name in ("curobo", "wrs_rrt"):
        t0 = time.perf_counter()
        segs = run_segments(name, plans, order, world, oracle, cu, arm)
        results[name] = dict(segments=segs, wall_s=time.perf_counter() - t0)
        ok = all(s["success"] for s in segs)
        print(f"{name}: success={ok} plan={sum(s['plan_s'] for s in segs):.2f}s", flush=True)

    # ---- execution time: same rule (topp) + native ---------------------------------
    fixed, fixed_same = [], 0.0
    for sp in plans:
        lin, native = fixed_segments(sp)
        same = {n: float(topp(p)[1][-1]) for n, p in lin.items()}
        fixed.append(dict(native=native, same=same))
        fixed_same += sum(same.values()) + 2 * GRIPPER_S
    fixed_native = sum(sum(f["native"].values()) + 2 * GRIPPER_S for f in fixed)
    for r in results.values():
        for sg in r["segments"]:
            if sg["success"]:
                p = np.asarray(sg["path"])
                stops = range(len(p)) if sg["source"] == "wrs_rrt" else ()
                sg["same_s"] = float(topp(p, stops)[1][-1])
    summary = {}
    for name, r in results.items():
        segs = r["segments"]
        failed = [f"{s['part']}:{s['leg']}" for s in segs if not s["success"]]
        planned_same = sum(s.get("same_s", 0.0) for s in segs)
        planned_native = sum(s.get("exec_s", 0.0) for s in segs)
        summary[name] = dict(
            success=not failed, failed_segments=failed,
            planned_segments=len(segs),
            plan_s=sum(s["plan_s"] for s in segs),
            verify_s=sum(s["verify_s"] for s in segs),
            planned_exec_same_s=planned_same, planned_exec_native_s=planned_native,
            fixed_exec_same_s=fixed_same, fixed_exec_native_s=fixed_native,
            total_exec_same_s=planned_same + fixed_same,
            total_exec_native_s=planned_native + fixed_native,
            joint_path_rad=sum(float(np.sum(np.linalg.norm(np.diff(np.asarray(s["path"]), axis=0), axis=1)))
                               for s in segs if s["success"]),
            sources={str(src): sum(1 for s in segs if s["source"] == src)
                     for src in {s["source"] for s in segs}},
            verification_rejections=sum(sum(1 for t in s["tiers"] if "rejected" in str(t[1]))
                                        for s in segs))
    out_dir = OUTPUT_DIR / f"assembly_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(created=datetime.now().isoformat(timespec="seconds"), seed=args.seed,
                  order=order, site=SITE.tolist(), layout={k: v.tolist() for k, v in layout.items()},
                  layout_attempts=layout_log, grasp_screening_s=screening_s,
                  grasps={sp.pid: dict(grasp_id=sp.grasp_id, jaw=sp.jaw,
                                       q={k: v.tolist() for k, v in sp.q.items()},
                                       screening=sp.screening) for sp in plans},
                  fixed_exec=fixed, summary=summary, setup_s=cu.setup_s,
                  screening_by_part=stats,
                  held_spheres=getattr(cu, "held_sphere_info", None),
                  segments={n: r["segments"] for n, r in results.items()},
                  total_wall_s=time.perf_counter() - t_all)
    (out_dir / "run.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    export_viz(out_dir, parts, order, layout, plans, results, fixed, summary, report)
    print(json.dumps(summary, indent=2))
    print("->", out_dir)


# ---------------------------------------------------------------------------
def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def export_viz(out_dir, parts, order, layout, plans, results, fixed, summary, report, fps=30):
    """Timeline of robot link + part poses for the HTML viewer (both planners)."""
    import trimesh
    import open3d as o3d
    from scipy.spatial.transform import Rotation
    from .export_model import fk_frames, load_chain

    model = ASSET_DIR / "panthera_full"
    chain = load_chain(model)
    parent = {"base_link": -1, "link1": 0, "link2": 1, "link3": 2, "link4": 3, "link5": 4,
              "gripper_palm": 5, "finger_left": 5, "finger_right": 5}
    bodies = []
    for name in chain["collision_links"]:
        m = trimesh.load(model / f"meshes/{name}.stl")
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),
                                       o3d.utility.Vector3iVector(m.faces))
        om = om.simplify_quadric_decimation(target_number_of_triangles=2500)
        bodies.append(dict(name=name, kind="robot", v=_b64(np.asarray(om.vertices), "<f4"),
                           f=_b64(np.asarray(om.triangles), "<u4")))
    for pid in order:
        m = parts[pid].mesh
        bodies.append(dict(name=pid, kind="part", v=_b64(m.vertices, "<f4"), f=_b64(m.faces, "<u4")))
    base = np.asarray(chain["base"], float)

    def robot_poses(q):
        fr = fk_frames(chain, q)
        return [base if parent[n] < 0 else fr[parent[n]] for n in chain["collision_links"]]

    tcp_rel = np.asarray(chain["tcp"], float)

    def timeline(name):
        """list of (duration, kind, payload) pieces in execution order."""
        segs = results[name]["segments"]
        pieces = []
        by = {(s["step"], s["leg"]): s for s in segs}
        for k, sp in enumerate(plans):
            lin, lt = fixed_segments(sp)
            a = by[(k, "A")]
            pieces.append(("A", k, a, None))
            pieces.append(("lin", k, lin["approach"], lt["approach"], None))
            pieces.append(("grip", k, sp.q["pick"], GRIPPER_S, "close"))
            pieces.append(("lin", k, lin["lift"], lt["lift"], "held"))
            pieces.append(("B", k, by[(k, "B")], None))
            pieces.append(("lin", k, lin["insert"], lt["insert"], "held"))
            pieces.append(("grip", k, sp.q["ins"], GRIPPER_S, "open"))
            pieces.append(("lin", k, lin["retreat"], lt["retreat"], None))
        pieces.append(("home", len(plans), by[(len(plans), "home")], None))
        return pieces

    def timed(path, stops):
        pts, ts = topp(np.asarray(path, float), stops)
        return pts, ts

    def sample(pts, ts, n):
        tt = np.linspace(0, ts[-1], n)
        return np.stack([np.interp(tt, ts, pts[:, j]) for j in range(6)], 1)

    viz = dict(bodies=bodies, runs={})
    for name in results:
        frames_q, frames_state, events = [], [], []
        t = 0.0
        held_pid = None
        done = set()
        for piece in timeline(name):
            kind = piece[0]
            k = piece[1]
            sp = plans[k] if k < len(plans) else None
            failed = False
            if kind in ("A", "B", "home"):
                seg = piece[2]
                label = {"A": "去取件", "B": "搬运", "home": "回 HOME"}[kind]
                src = seg["source"]
                if seg["success"]:
                    path = seg["path"]
                    stops = range(len(path)) if src == "wrs_rrt" else ()
                else:   # show the unsolved segment as a straight joint move, flagged
                    failed, src = True, "failed"
                    a_q = plans[k].q["lift"] if kind == "B" else (
                        HOME if k == 0 and kind == "A" else plans[k - 1].q["retreat"])
                    b_q = plans[k].q["preins"] if kind == "B" else (
                        plans[k].q["prepick"] if kind == "A" else HOME)
                    path, stops = [a_q, b_q], ()
                pts, ts = timed(path, stops)
            elif kind == "lin":
                label, src = "直线段", "lin"
                pts, ts = timed(piece[2], ())
            else:
                label, src = ("夹紧" if piece[4] == "close" else "松开"), "grip"
                pts, ts = np.stack([piece[2], piece[2]]), np.array([0.0, GRIPPER_S])
            dur = float(ts[-1])
            if kind == "grip" and piece[4] == "close":
                held_pid = sp.pid
            n = max(2, int(round(dur * fps)) + 1)
            qs = sample(pts, ts, n)
            events.append(dict(t0=t, t1=t + dur, kind=kind, label=label, failed=failed,
                               part=sp.pid if sp else "-", source=src))
            for q in (qs[:-1] if kind != "home" else qs):
                frames_q.append(q)
                frames_state.append((held_pid, frozenset(done)))
            if kind == "grip" and piece[4] == "open":
                done.add(sp.pid)
                held_pid = None
            t += dur
        # body poses per frame
        n_rb = len(chain["collision_links"])
        arr = np.zeros((len(frames_q), n_rb + len(order), 7), np.float32)
        G = {sp.pid: sp.G for sp in plans}
        for i, (q, (hp, dn)) in enumerate(zip(frames_q, frames_state)):
            poses = robot_poses(q)
            fr = fk_frames(chain, q)
            tcp = fr[5] @ tcp_rel
            for pid in order:
                if pid == order[0] or pid in dn:
                    poses.append(parts[pid].goal)
                elif pid == hp:
                    poses.append(tcp @ np.linalg.inv(G[pid]))
                else:
                    poses.append(layout[pid])
            for j, h in enumerate(poses):
                arr[i, j, :3] = h[:3, 3]
                arr[i, j, 3:] = Rotation.from_matrix(h[:3, :3]).as_quat()
        viz["runs"][name] = dict(fps=fps, frames=len(frames_q), duration=t,
                                 poses=_b64(arr, "<f4"), events=events,
                                 summary=summary[name])
    viz["meta"] = dict(order=order, site=SITE.tolist(), seed=report["seed"],
                       grasp_screening_s=report["grasp_screening_s"],
                       layout_attempts=len(report["layout_attempts"]),
                       grasps={k: dict(id=v["grasp_id"], screening=v["screening"])
                               for k, v in report["grasps"].items()},
                       segments={n: [dict(step=s["step"], part=s["part"], leg=s["leg"],
                                          plan_s=s["plan_s"], verify_s=s["verify_s"],
                                          exec_s=s.get("exec_s"), same_s=s.get("same_s"),
                                          success=s["success"], source=s["source"],
                                          tiers=s["tiers"]) for s in r["segments"]]
                                 for n, r in results.items()})
    (out_dir / "viz.json").write_text(json.dumps(viz), encoding="utf-8")


if __name__ == "__main__":
    main()
