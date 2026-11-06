"""实验：让轨迹优化直接"长出"摆放位置（自由终端），与网格穷举真值对比。

    & $py -X utf8 -m curobo_bridge.exp_free_end            # 完整实验（真值约 15-20 min）
    & $py -X utf8 -m curobo_bridge.exp_free_end --smoke    # 只跑新方法 + 小网格

问题（Bridge，单臂 Panthera，装配位 x=0.44）：前一件 post_r 刚装完，求横梁 beam 的摆放
(x, y, 偏航) 与抓取 g，使整段代价最小::

    C = T_空手( q撤离(post_r) → 预抓取(x, g) ) + T_持件( 抬升(x, g) → q预插入(beam, g) )

两段都用 cuRobo 规划、同一个 TOPP 规则计时。世界：底座 + 两根柱子（已装）、load（已摆），
以及低矮禁入盒（底板+4 cm、机器人基座、load+4 cm；高 10 cm），它们表达"不能摆在这里"。

* 真值：网格位置 × 4 个偏航 × 所有可行抓取，两段都定点规划，取 min。
* 新方法（自由终端）：终端只约束 z 与倾斜（ToolPoseCriteria 世界系权重 [0,0,1,1,1,0]），
  x、y、偏航放开。持件段从 q预插入 出发、空手段从 q撤离 出发，每个抓取撒几个种子，
  各自"滑"到局部最优落点；再对两边的前 M 个提议交叉定点评估另一段，取链代价最小。
* 基线：只用事先挑定的一个抓取；只看直线关节时间（不看障碍）。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare

SITE = np.array([0.44, 0.0, 0.0])
TABLE_C = np.array([0.33, 0.0, -0.01])
TABLE_S = np.array([0.64, 0.80, 0.02])
LIFT, APPROACH, PREINS, RETREAT = 0.06, 0.05, 0.05, 0.05
LOAD_XY = np.array([0.20, 0.30])
FREE_W = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]      # world x, y, z | roll, pitch, yaw
B = 32


def _h(p=np.zeros(3), r=np.eye(3)):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, p
    return out


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------- scene
def build():
    import trimesh
    from asmcore.model import AssemblyState
    from asmcore.robotics import GraspabilityAnalyzer, GraspabilityConfig
    from adapters.panthera_gripper import PantheraGripperPort
    from tasks.bridge import build_bridge

    asm = build_bridge()
    parts = {}
    for p in asm.parts:
        if p.part_id == "table":
            continue
        m = trimesh.Trimesh(np.asarray(p.geometry.vertices), np.asarray(p.geometry.faces), process=False)
        g = np.asarray(p.assembled_tf, float).copy()
        g[:3, 3] += SITE
        parts[p.part_id] = dict(mesh=m, goal=g)
    fixed = {p.part_id for p in asm.parts if p.fixed}
    full = asm.initial_state()
    analyzer = GraspabilityAnalyzer(asm, PantheraGripperPort(), config=GraspabilityConfig(max_grasps=8))

    def family(pid, present, offsets):
        st = AssemblyState({q: tf for q, tf in full.poses.items() if q in present | {pid}})
        res = analyzer.analyze(st, pid)
        ext = parts[pid]["mesh"].extents
        out = []
        for gi, g in enumerate(analyzer.accepted_grasps(res)):
            G0 = np.asarray(g.pose, float)
            free = [i for i in range(3) if abs(G0[i, 1]) < 0.5]   # axes not along the closing axis
            ax = max(free, key=lambda i: ext[i])
            for off in offsets:
                for roll in np.deg2rad(np.arange(0, 360, 30)):
                    G = G0.copy()
                    G[:3, :3] = G0[:3, :3] @ _ry(roll)
                    G[ax, 3] += off
                    out.append(dict(G=G, pair=gi, roll=float(np.rad2deg(roll)), off=float(off)))
        return out

    grasps = dict(beam=family("beam", fixed | {"post_l", "post_r"}, (-0.08, 0.0, 0.08)),
                  post_r=family("post_r", fixed | {"post_l"}, (0.0,)))
    load_pose = _h(np.array([LOAD_XY[0], LOAD_XY[1], 0.03]))
    return parts, grasps, load_pose


def keepouts(load_pose):
    """(centre, dims) footprints where a part may NOT be staged (checked geometrically)."""
    return [(np.r_[SITE[:2], 0.05], np.array([0.42, 0.42, 0.10])),
            (np.array([0.0, 0.0, 0.05]), np.array([0.28, 0.28, 0.10])),
            (np.r_[load_pose[:2, 3], 0.05], np.array([0.18, 0.14, 0.10]))]


def slabs(load_pose):
    """Physical thin slabs (z 0.055-0.095) over the site and the load region.

    They sit exactly where a lifted part (bottom 6 cm above the table) or a pre-grasping
    gripper would be, so the free-end optimiser cannot "land" there; unlike a full-height
    box they do not block the arm above the posts. The robot-base keep-out is NOT
    physical (a box there would swallow link2) and is enforced by placement_ok only."""
    return [(np.r_[SITE[:2], 0.075], np.array([0.42, 0.42, 0.04])),
            (np.r_[load_pose[:2, 3], 0.075], np.array([0.18, 0.14, 0.04]))]


def scene_dict(parts, load_pose, kos):
    from .scenes import pose7
    d = {"cuboid": {"table": dict(dims=TABLE_S.tolist(), pose=pose7(_h(TABLE_C)))}, "mesh": {}}
    for i, (c, s) in enumerate(kos):
        d["cuboid"][f"keepout{i}"] = dict(dims=s.tolist(), pose=pose7(_h(c)))
    for pid, T in (("base", parts["base"]["goal"]), ("post_l", parts["post_l"]["goal"]),
                   ("post_r", parts["post_r"]["goal"]), ("load", load_pose)):
        m = parts[pid]["mesh"]
        d["mesh"][pid] = dict(vertices=m.vertices.tolist(), faces=m.faces.reshape(-1).tolist(),
                              pose=pose7(T))
    return d


def placement_ok(beam_mesh, T, kos):
    from shapely.geometry import MultiPoint, box
    v = beam_mesh.vertices @ T[:3, :3].T + T[:3, 3]
    poly = MultiPoint(v[:, :2]).convex_hull
    lo, hi = TABLE_C[:2] - TABLE_S[:2] / 2 + 0.01, TABLE_C[:2] + TABLE_S[:2] / 2 - 0.01
    if not box(lo[0], lo[1], hi[0], hi[1]).contains(poly):
        return False
    return not any(poly.intersects(box(c[0] - s[0] / 2, c[1] - s[1] / 2, c[0] + s[0] / 2, c[1] + s[1] / 2))
                   for c, s in kos)


def staging(x, y, yaw, beam_mesh):
    return _h(np.array([x, y, beam_mesh.extents[2] / 2 + 2e-4]), _rz(yaw))


def lift_tcp(T_obj, G):
    P = T_obj @ G
    P[2, 3] += LIFT
    return P


def prepick_tcp(T_obj, G):
    P = T_obj @ G
    P[:3, 3] -= APPROACH * P[:3, 2]
    return P


# ---------------------------------------------------------------------------- cuRobo helpers
class Planner:
    def __init__(self, sdict):
        import torch
        from curobo.batch_motion_planner import BatchMotionPlanner
        from curobo.motion_planner import MotionPlannerCfg
        from .curobo_models import load_robot_dict, with_attach_link, with_buffer
        robot = with_attach_link(with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", 1.0), 0.002), 32)
        self.names = robot["kinematics"]["cspace"]["joint_names"]
        self.bp = BatchMotionPlanner(MotionPlannerCfg.create(
            robot=robot, scene_model=sdict, collision_cache={"mesh": 8, "cuboid": 8}, max_batch_size=B))
        self.bp.warmup()
        self.dt = self.bp.trajopt_solver.config.interpolation_dt
        self.torch = torch
        self.calls = 0

    def js(self, q):
        from curobo.types import JointState
        return JointState.from_position(self.torch.as_tensor(np.atleast_2d(q), dtype=torch_f32(),
                                                             device="cuda"), joint_names=self.names)

    def goal(self, P):
        from curobo.types import GoalToolPose, Pose
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(P[:, :3, :3]).as_quat()
        pose = Pose(position=self.torch.as_tensor(P[:, :3, 3], dtype=torch_f32(), device="cuda"),
                    quaternion=self.torch.as_tensor(q[:, [3, 0, 1, 2]], dtype=torch_f32(), device="cuda"))
        return GoalToolPose.from_poses({"wrs_tcp": pose.unsqueeze(1)}, ordered_tool_frames=["wrs_tcp"])

    def criteria(self, free: bool, trajopt_only=False):
        from curobo.types import ToolPoseCriteria
        c = ToolPoseCriteria(terminal_pose_axes_weight_factor=FREE_W) if free else ToolPoseCriteria()
        (self.bp.trajopt_solver if trajopt_only else self.bp).update_tool_pose_criteria({"wrs_tcp": c})

    def hold(self, q, mesh, T_obj):
        import trimesh  # noqa: F401
        from curobo.scene import Mesh
        from curobo.types import Pose
        from scipy.spatial.transform import Rotation
        am = self.bp.attachment_manager
        key = id(mesh)
        if getattr(self, "_fit", (None, None))[0] != key:
            sph = am.fit_spheres([Mesh(name="held", vertices=mesh.vertices.tolist(),
                                       faces=mesh.faces.tolist(), pose=[0, 0, 0, 1, 0, 0, 0])],
                                 num_spheres=32)
            pts = self.torch.as_tensor(mesh.sample(3000, seed=0), dtype=sph.dtype, device=sph.device)
            gap = (self.torch.cdist(pts, sph[:, :3]) - sph[None, :, 3]).min(1).values
            sph = sph.clone()
            sph[:, 3] += float(self.torch.clamp(gap.max(), min=0)) + 0.001
            self._fit = (key, sph)
        xyzw = Rotation.from_matrix(T_obj[:3, :3]).as_quat()
        off = Pose(position=self.torch.as_tensor(T_obj[None, :3, 3], dtype=torch_f32(), device="cuda"),
                   quaternion=self.torch.as_tensor([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]],
                                                   dtype=torch_f32(), device="cuda"))
        am.update(self._fit[1], self.js(q), link_name="attached_object", world_objects_pose_offset=off)

    def release(self):
        self.bp.attachment_manager.detach(link_name="attached_object")

    def ik(self, P):
        """collision-aware IK, [n,4,4] -> (q [n,6], ok [n])"""
        n = len(P)
        qs, oks = [], []
        for s in range(0, n, B):
            chunk = P[s:s + B]
            pad = np.concatenate([chunk, np.repeat(chunk[-1:], B - len(chunk), 0)])
            r = self.bp.ik_solver.solve_pose(self.goal(pad), return_seeds=1)
            qs.append(r.solution.reshape(B, -1, 6)[: len(chunk), 0].cpu().numpy())
            oks.append(r.success.reshape(B, -1)[: len(chunk), 0].cpu().numpy())
        return np.concatenate(qs), np.concatenate(oks)

    def plan(self, starts, P, *, free=False, seeds=None):
        """Batched pose-goal planning. free=True: terminal x, y, yaw free (seeded by ``seeds``).
        Returns list of paths (or None)."""
        n = len(P)
        out = []
        for s in range(0, n, B):
            m = min(B, n - s)
            st = np.concatenate([starts[s:s + m], np.repeat(starts[s + m - 1:s + m], B - m, 0)])
            gp = np.concatenate([P[s:s + m], np.repeat(P[s + m - 1:s + m], B - m, 0)])
            self.calls += 1
            if not free:
                self.criteria(False)
                r = self.bp.plan_pose(self.goal(gp), self.js(st), max_attempts=1)
            else:
                sd = np.concatenate([seeds[s:s + m], np.repeat(seeds[s + m - 1:s + m], B - m, 0)])
                self.criteria(False)
                ns = self.bp.trajopt_solver.config.num_seeds
                seed_cfg = self.torch.as_tensor(sd, dtype=torch_f32(), device="cuda")[:, None].repeat(1, ns, 1)
                self.criteria(True, trajopt_only=True)
                r = self.bp.trajopt_solver.solve_pose(self.goal(gp), self.js(st), seed_config=seed_cfg,
                                                      use_implicit_goal=True, finetune_attempts=1)
                self.criteria(False, trajopt_only=True)
            if r is None:
                out += [None] * m
                continue
            ok = r.success.reshape(B, -1).any(-1).cpu().numpy()
            traj = r.interpolated_trajectory.position.reshape(B, -1, 6)
            last = r.interpolated_last_tstep.reshape(B, -1)[:, 0].cpu().numpy()
            out += [traj[i, : int(last[i]) + 1].cpu().numpy() if ok[i] else None for i in range(m)]
        return out


def torch_f32():
    import torch
    return torch.float32


def leg_time(path):
    from .assembly_demo import topp
    return float(topp(np.asarray(path))[1][-1])


def vel_lb(a, b):
    from .assembly_demo import VMAX
    return float(np.max(np.abs(np.asarray(b) - np.asarray(a)) / VMAX))


def spread(points, k):
    """Farthest-point subset indices of 2D points."""
    if len(points) <= k:
        return list(range(len(points)))
    idx = [0]
    d = np.linalg.norm(points - points[0], axis=1)
    while len(idx) < k:
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return idx


def seed_stagings(pl, pool, G, make_tcp, k):
    """IK over the whole placement pool, then k spread IK-feasible seeds (poses, q)."""
    P = np.stack([make_tcp(T, G) for T in pool])
    q, ok = pl.ik(P)
    good = np.flatnonzero(ok)
    if not len(good):
        return None, None
    pick = good[spread(np.stack([pool[i][:2, 3] for i in good]), k)]
    return P[pick], q[pick]


# ---------------------------------------------------------------------------- experiment
def main(argv=None):
    prepare()
    from .export_model import fk_frames, load_chain

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=float, default=0.06)
    ap.add_argument("--seeds", type=int, default=4, help="free-end seeds per grasp and leg")
    ap.add_argument("--cross", type=int, default=16, help="proposals per side for cross evaluation")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    chain = load_chain(ASSET_DIR / "panthera_wrsext")

    def tcp_of(q):
        return fk_frames(chain, q)[5] @ np.asarray(chain["tcp"])

    parts, grasps, load_pose = build()
    kos = keepouts(load_pose)
    beam = parts["beam"]["mesh"]
    pl = Planner(scene_dict(parts, load_pose, slabs(load_pose)))
    rep = dict(created=datetime.now().isoformat(timespec="seconds"), args=vars(args))

    # ---- anchors: q_retreat(post_r) and q_preins(beam, g) ------------------------------
    t0 = time.perf_counter()
    pr_goal = parts["post_r"]["goal"]
    R = np.stack([pr_goal @ g["G"] for g in grasps["post_r"]])
    R[:, :3, 3] -= RETREAT * R[:, :3, 2]
    q_r, ok_r = pl.ik(R)
    cand = [i for i in np.flatnonzero(ok_r) if R[i, 2, 2] < -0.5] or list(np.flatnonzero(ok_r))
    q_ret = q_r[min(cand, key=lambda i: np.abs(q_r[i]).sum())]
    bg = parts["beam"]["goal"]
    Gs = [g for g in grasps["beam"]]
    Pp = np.stack([bg @ g["G"] for g in Gs])
    Pp[:, 2, 3] += PREINS
    down_ok = np.array([(bg @ g["G"])[2, 2] < 0.3 for g in Gs])   # not approaching from below
    pl.hold(np.zeros(6), beam, bg)       # IK with the beam attached (pose fixed below per grasp)
    pl.release()
    q_pre, ok_pre = pl.ik(Pp)
    feas = [i for i in range(len(Gs)) if ok_pre[i] and down_ok[i]]
    rep["anchors"] = dict(grasp_variants=len(Gs), goal_side_feasible=len(feas),
                          seconds=time.perf_counter() - t0)
    print("anchors", rep["anchors"], flush=True)

    # ---- method: free-end proposals --------------------------------------------------------
    t_m = time.perf_counter()
    calls0 = pl.calls
    proposals = []      # dict(g, T_obj, carry, empty)
    # carry side: start q_preins(g) -> free lift pose
    pool = []
    while len(pool) < 96:
        T = staging(rng.uniform(0.02, 0.64), rng.uniform(-0.39, 0.39), rng.uniform(0, np.pi), beam)
        if placement_ok(beam, T, kos):
            pool.append(T)
    for gi in feas:
        G = Gs[gi]["G"]
        pl.hold(q_pre[gi], beam, tcp_of(q_pre[gi]) @ np.linalg.inv(G))
        P, q_seed = seed_stagings(pl, pool, G, lift_tcp, args.seeds)
        if P is None:
            pl.release()
            continue
        use = np.arange(len(P))
        seedsT = list(P)
        if len(use):
            paths = pl.plan(np.repeat(q_pre[gi][None], len(use), 0), P[use], free=True, seeds=q_seed[use])
            dbg = rep.setdefault("carry_debug", dict(seeds=0, ik_ok=0, planned=0, off_manifold=0,
                                                      bad_place=0, kept=0, dz=[], tilt=[], slide=[]))
            dbg["seeds"] += len(seedsT)
            dbg["ik_ok"] += len(use)
            for k_, path in enumerate(paths):
                if path is None:
                    continue
                dbg["planned"] += 1
                dbg["slide"].append(round(float(np.linalg.norm(tcp_of(path[-1])[:2, 3] - P[use[k_]][:2, 3])), 3))
                T_end = tcp_of(path[-1]) @ np.linalg.inv(G)
                T_obj = T_end.copy()
                T_obj[2, 3] -= LIFT
                dbg["dz"].append(round(float(T_obj[2, 3] - beam.extents[2] / 2), 4))
                dbg["tilt"].append(round(float(T_obj[2, 2]), 4))
                if abs(T_obj[2, 3] - beam.extents[2] / 2) > 0.01 or T_obj[2, 2] < 0.995:
                    dbg["off_manifold"] += 1
                    continue
                if not placement_ok(beam, T_obj, kos):
                    dbg["bad_place"] += 1
                if placement_ok(beam, T_obj, kos):
                    dbg["kept"] += 1
                    proposals.append(dict(side="carry", g=gi, T=T_obj, carry=leg_time(path[::-1]),
                                          lb_other=vel_lb(q_ret, path[-1])))
        pl.release()
    # empty side: start q_retreat(post_r) -> free prepick pose
    for gi in feas:
        G = Gs[gi]["G"]
        P, q_seed = seed_stagings(pl, pool, G, prepick_tcp, args.seeds)
        if P is None:
            continue
        use = np.arange(len(P))
        if len(use):
            paths = pl.plan(np.repeat(q_ret[None], len(use), 0), P[use], free=True, seeds=q_seed[use])
            for path in paths:
                if path is None:
                    continue
                T_tcp = tcp_of(path[-1])
                T_tcp[:3, 3] += APPROACH * T_tcp[:3, 2]
                T_obj = T_tcp @ np.linalg.inv(G)
                if abs(T_obj[2, 3] - beam.extents[2] / 2) > 0.01 or T_obj[2, 2] < 0.995:
                    continue
                if placement_ok(beam, T_obj, kos):
                    proposals.append(dict(side="empty", g=gi, T=T_obj, empty=leg_time(path),
                                          lb_other=vel_lb(path[-1], q_pre[gi])))
    # cross evaluation of the other leg for the best M proposals of each side
    best = None
    for side in ("carry", "empty"):
        props = sorted([p for p in proposals if p["side"] == side],
                       key=lambda p: p[side] + p["lb_other"])[: args.cross]
        print(f"  {side} proposals kept for cross evaluation: {len(props)}", flush=True)
        if not props:
            continue
        if side == "carry":
            P = np.stack([prepick_tcp(p["T"], Gs[p["g"]]["G"]) for p in props])
            paths = pl.plan(np.repeat(q_ret[None], len(props), 0), P)
            for p, path in zip(props, paths):
                if path is not None:
                    p["empty"] = leg_time(path)
        else:
            by_g = {}
            for p in props:
                by_g.setdefault(p["g"], []).append(p)
            for gi, ps in by_g.items():
                G = Gs[gi]["G"]
                pl.hold(q_pre[gi], beam, tcp_of(q_pre[gi]) @ np.linalg.inv(G))
                P = np.stack([lift_tcp(p["T"], G) for p in ps])
                paths = pl.plan(np.repeat(q_pre[gi][None], len(ps), 0), P)
                for p, path in zip(ps, paths):
                    if path is not None:
                        p["carry"] = leg_time(path[::-1])
                pl.release()
        for p in props:
            if "carry" in p and "empty" in p:
                c = p["carry"] + p["empty"]
                if best is None or c < best["cost"]:
                    best = dict(cost=c, side=side, g=int(p["g"]), T=p["T"].tolist(),
                                carry=p["carry"], empty=p["empty"])
    rep["method"] = dict(seconds=time.perf_counter() - t_m, batched_calls=pl.calls - calls0,
                         problems=(pl.calls - calls0) * B, proposals=len(proposals),
                         carry_proposals=sum(p["side"] == "carry" for p in proposals),
                         empty_proposals=sum(p["side"] == "empty" for p in proposals), best=best)
    rep["proposals"] = [dict(side=p["side"], g=int(p["g"]), xy=p["T"][:2, 3].tolist(),
                             yaw=float(np.arctan2(p["T"][1, 0], p["T"][0, 0])),
                             own=p.get(p["side"]), chain=(p["carry"] + p["empty"])
                             if "carry" in p and "empty" in p else None) for p in proposals]
    print("carry_debug", {k: (v if not isinstance(v, list) else v[:16]) for k, v in rep.get("carry_debug", {}).items()}, flush=True)
    print("method", {k: v for k, v in rep["method"].items() if k != "best"}, "best", best and round(best["cost"], 3), flush=True)

    # ---- ground truth: grid x yaw x grasp, both legs pinned -----------------------------------
    t_t = time.perf_counter()
    calls0 = pl.calls
    step = 0.12 if args.smoke else args.grid
    xs = np.arange(0.05, 0.63, step)
    ys = np.arange(-0.36, 0.37, step)
    yaws = np.deg2rad([0, 45, 90, 135])
    places = [staging(x, y, a, beam) for x in xs for y in ys for a in yaws]
    places = [T for T in places if placement_ok(beam, T, kos)]
    gset = feas[:: max(1, len(feas) // 6)] if args.smoke else feas
    empty_c = np.full((len(places), len(Gs)), np.nan)
    carry_c = np.full((len(places), len(Gs)), np.nan)
    lb = np.full((len(places), len(Gs)), np.nan)
    for gi in gset:
        G = Gs[gi]["G"]
        Pe = np.stack([prepick_tcp(T, G) for T in places])
        paths = pl.plan(np.repeat(q_ret[None], len(places), 0), Pe)
        for i, path in enumerate(paths):
            if path is not None:
                empty_c[i, gi] = leg_time(path)
                lb[i, gi] = vel_lb(q_ret, path[-1])
        pl.hold(q_pre[gi], beam, tcp_of(q_pre[gi]) @ np.linalg.inv(G))
        Pc = np.stack([lift_tcp(T, G) for T in places])
        paths = pl.plan(np.repeat(q_pre[gi][None], len(places), 0), Pc)
        for i, path in enumerate(paths):
            if path is not None:
                carry_c[i, gi] = leg_time(path[::-1])
                if np.isfinite(lb[i, gi]):
                    lb[i, gi] += vel_lb(path[-1], q_pre[gi])
        pl.release()
        print(f"  truth grasp {gi}: done ({time.perf_counter() - t_t:.0f}s)", flush=True)
    chain_c = empty_c + carry_c
    flat = np.nanmin(chain_c) if np.isfinite(chain_c).any() else None
    i_best, g_best = np.unravel_index(np.nanargmin(chain_c), chain_c.shape)
    # baseline 1: one grasp fixed a priori (most vertical, zero offset, first pair)
    pri = min(gset, key=lambda gi: ((bg @ Gs[gi]["G"])[2, 2], abs(Gs[gi]["off"]), Gs[gi]["pair"]))
    fixed_best = np.nanmin(chain_c[:, pri]) if np.isfinite(chain_c[:, pri]).any() else None
    # baseline 2: choose by straight-line joint time (ignores obstacles), pay the true cost
    j = np.nanargmin(np.where(np.isfinite(chain_c), lb, np.nan))
    lb_i, lb_g = np.unravel_index(j, lb.shape)
    finite = chain_c[np.isfinite(chain_c)]
    m_cost = best["cost"] if best else None
    rep["truth"] = dict(seconds=time.perf_counter() - t_t, batched_calls=pl.calls - calls0,
                        problems=(pl.calls - calls0) * B, placements=len(places), grasps=len(gset),
                        feasible_pairs=int(np.isfinite(chain_c).sum()), best=float(flat),
                        best_xy=places[i_best][:2, 3].tolist(), best_grasp=int(g_best),
                        median=float(np.median(finite)))
    rep["compare"] = dict(
        method_cost=m_cost, truth_best=float(flat),
        method_gap_pct=None if m_cost is None else 100 * (m_cost - flat) / flat,
        method_percentile=None if m_cost is None else float((finite < m_cost).mean() * 100),
        fixed_grasp=int(pri), fixed_grasp_best=None if fixed_best is None else float(fixed_best),
        fixed_grasp_gap_pct=None if fixed_best is None else 100 * (fixed_best - flat) / flat,
        straight_line_choice_cost=float(chain_c[lb_i, lb_g]),
        straight_line_gap_pct=100 * (chain_c[lb_i, lb_g] - flat) / flat,
        speedup_problems=rep["truth"]["problems"] / max(rep["method"]["problems"], 1),
        speedup_time=rep["truth"]["seconds"] / max(rep["method"]["seconds"], 1e-9))
    print("truth", {k: v for k, v in rep["truth"].items()}, flush=True)
    print("compare", rep["compare"], flush=True)
    per_xy = {}
    for i, T in enumerate(places):
        k = (round(T[0, 3], 3), round(T[1, 3], 3))
        v = np.nanmin(chain_c[i]) if np.isfinite(chain_c[i]).any() else np.nan
        per_xy[k] = np.nanmin([per_xy.get(k, np.nan), v])
    rep["heat"] = [dict(x=k[0], y=k[1], cost=None if not np.isfinite(v) else float(v)) for k, v in per_xy.items()]
    rep["keepouts"] = [dict(c=c.tolist(), s=s.tolist()) for c, s in kos]
    rep["site"] = SITE.tolist()
    out = OUTPUT_DIR / f"free_end_{'smoke_' if args.smoke else ''}{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
