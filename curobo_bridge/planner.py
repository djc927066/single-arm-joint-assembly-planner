"""Panthera 单臂的分级运动规划器：cuRobo 为主，WRS RRT 兜底，网格真值复核。

分级（每级的输出都要通过复核，失败才进入下一级）：

1. ``curobo``          ``MotionPlanner.plan_cspace``，MorphIt d1 + 2 mm 场景缓冲，
                       自碰撞不放宽（对网格真值零漏报，见 bench_self_collision）。
2. ``curobo_relaxed``  同上但自碰撞球 -2 mm。包络外扩使自碰撞误报约 14%，会把
                       TrajOpt 卡住；放宽后可能漏掉真实自碰撞，因此必须复核。
3. ``wrs_rrt``         WRS RRTConnect（sealp L3 参数：ext_dist 0.3, smoothing 150, 10 s）。
                       RRT 只在采样点上检测，同样要复核。

复核（``verify``）：关节路径按 ``check_step`` 加密，逐状态做
WRS ODE 网格场景碰撞 + 未忽略 link 对的网格自碰撞。这是几何真值，不是 WRS 的
AABB 图元语义（图元语义对真值无漏报但误报约 28%，会误杀 cuRobo 贴近障碍的合法路径）。

批量：``plan_batch`` 用 ``BatchMotionPlanner(multi_env=True)``，每个问题一个碰撞世界，
一次 GPU 调用求解全部问题；失败或复核不过的问题逐个走第 2、3 级。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .runtime import ASSET_DIR


@dataclass
class PlanResult:
    success: bool
    path: np.ndarray | None
    source: str | None
    plan_s: float
    verify_s: float = 0.0
    tiers: list = field(default_factory=list)   # [(tier, outcome, seconds)]


def _reject_msg(v: dict) -> str:
    if "scene_hits" in v:
        return f"rejected scene={v['scene_hits']} self={v['self_hits']}"
    return f"rejected mesh_hits={v.get('hits')}"


def densify(path: np.ndarray, step: float) -> np.ndarray:
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.asarray(out)


class PantheraPlanner:
    def __init__(self, *, density: float = 1.0, scene_buffer: float = 0.002,
                 relaxed_self_pad: float = -0.002, rrt_ext_dist: float = 0.30,
                 rrt_max_time: float = 10.0, rrt_smoothing: int = 150,
                 check_step: float = 0.01, collision_cache=None, arm=None,
                 attach_spheres: int = 0):
        from .bench_self_collision import checked_pairs, with_self_padding
        from .curobo_models import load_robot_dict, with_attach_link, with_buffer
        from .export_model import make_wrs_arm, wrs_collision_parts

        base = load_robot_dict(ASSET_DIR / "panthera_wrsext", density)
        self.robot = with_buffer(base, scene_buffer)
        self.robot_relaxed = with_self_padding(base, scene_buffer, relaxed_self_pad)
        self.attach_spheres = int(attach_spheres)
        if self.attach_spheres:
            self.robot = with_attach_link(self.robot, self.attach_spheres)
            self.robot_relaxed = with_attach_link(self.robot_relaxed, self.attach_spheres)
        self._held = None
        self._verify_fn = None
        self.joint_names = self.robot["kinematics"]["cspace"]["joint_names"]
        self.cache = collision_cache or {"mesh": 16, "cuboid": 16}
        self.arm = arm or make_wrs_arm()
        self.parts = {n: l for n, _, l in wrs_collision_parts(self.arm, False)}
        self.self_pairs = checked_pairs(base)
        self.rrt_kw = dict(ext_dist=rrt_ext_dist, max_time=rrt_max_time,
                           smoothing_n_iter=rrt_smoothing)
        self.check_step = check_step
        self._mp = {}
        self._scene = None
        self._wobs = None
        self.setup_s = {}

    # ---- scene -------------------------------------------------------------
    def set_scene(self, table_scene) -> None:
        """``table_scene``: :class:`curobo_bridge.scenes.TableScene`."""
        from curobo.scene import Scene
        self._scene = table_scene
        self._scene_dict = table_scene.curobo_scene_dict()
        self._wobs = table_scene.wrs_obstacles()
        for mp in self._mp.values():
            mp.update_world(Scene.create(self._scene_dict))

    def set_world(self, scene_dict: dict, rrt_obstacles: list, verify_fn=None) -> None:
        """Generic world: cuRobo scene dict, WRS obstacles for the RRT tier, and an
        optional ``verify_fn(path) -> dict(ok=..., ...)`` replacing :meth:`verify`
        (e.g. to include a held object and contact exemptions)."""
        from curobo.scene import Scene
        self._scene_dict = scene_dict
        self._wobs = rrt_obstacles
        self._verify_fn = verify_fn
        for mp in self._mp.values():
            mp.update_world(Scene.create(scene_dict))

    # ---- held object -----------------------------------------------------------
    def hold(self, q_grasp, obj_mesh_local, obj_world_pose, rrt_cmodel=None) -> None:
        """Attach an object for all tiers.

        ``obj_mesh_local``: trimesh in the object frame (watertight preferred);
        ``obj_world_pose``: 4x4 object pose when the arm is at ``q_grasp``;
        ``rrt_cmodel``: WRS CollisionModel posed at ``obj_world_pose`` (RRT tier)."""
        if not self.attach_spheres:
            raise RuntimeError("construct PantheraPlanner(attach_spheres=N) to hold objects")
        self._held = (np.asarray(q_grasp, float), obj_mesh_local,
                      np.asarray(obj_world_pose, float), rrt_cmodel, None)
        for mp in self._mp.values():
            self._attach(mp)
        if rrt_cmodel is not None:
            self.arm.goto_given_conf(self._held[0])
            self.arm.hold(rrt_cmodel)

    def _attach(self, mp) -> None:
        import torch
        from curobo.scene import Mesh
        from curobo.types import Pose
        from scipy.spatial.transform import Rotation
        q, mesh, pose, _, spheres = self._held
        am = mp.attachment_manager
        cache = self.__dict__.setdefault("_sphere_cache", {})
        key = hash(np.asarray(mesh.vertices, np.float32).tobytes())   # same geometry, one fit
        if spheres is None and key in cache:
            spheres = cache[key]
            self._held = self._held[:4] + (spheres,)
        if spheres is None:   # fit once per object mesh, reuse for every tier and hold
            obstacle = Mesh(name="held_object", vertices=np.asarray(mesh.vertices).tolist(),
                            faces=np.asarray(mesh.faces).reshape(-1, 3).tolist(),
                            pose=[0, 0, 0, 1, 0, 0, 0])
            spheres = am.fit_spheres([obstacle], num_spheres=self.attach_spheres)
            # MorphIt treats num_spheres as a cap and prunes, leaving ~8% of a post's
            # surface uncovered. Inflate every sphere by the worst surface deficit
            # (+1 mm) so the held part is fully enclosed (conservative).
            pts = torch.as_tensor(mesh.sample(3000, seed=0), dtype=spheres.dtype,
                                  device=spheres.device)
            gap = (torch.cdist(pts, spheres[:, :3]) - spheres[None, :, 3]).min(dim=1).values
            inflate = float(torch.clamp(gap.max(), min=0.0)) + 0.001
            spheres = spheres.clone()
            spheres[:, 3] += inflate
            self.held_sphere_info = dict(n=int(len(spheres)), inflate_m=inflate)
            cache[key] = spheres
            self._held = self._held[:4] + (spheres,)
        xyzw = Rotation.from_matrix(pose[:3, :3]).as_quat()
        offset = Pose(position=torch.as_tensor(pose[None, :3, 3], dtype=torch.float32, device="cuda"),
                      quaternion=torch.as_tensor([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]],
                                                 dtype=torch.float32, device="cuda"))
        am.update(spheres, self._js(q), link_name="attached_object",
                  world_objects_pose_offset=offset)

    def prefit(self, obj_mesh_local) -> float:
        """Fit (and cache) the held-object spheres ahead of planning; returns seconds.

        Sphere fitting depends only on the part geometry, so it is per-part preprocessing
        like the grasp library, not per-plan work."""
        t0 = time.perf_counter()
        saved = self._held
        self._held = (np.zeros(6), obj_mesh_local, np.eye(4), None, None)
        self._attach(self._planner("curobo"))
        self._mp["curobo"].attachment_manager.detach(link_name="attached_object")
        self._held = saved
        return time.perf_counter() - t0

    def release(self) -> None:
        if self._held is None:
            return
        for mp in self._mp.values():
            mp.attachment_manager.detach(link_name="attached_object")
        if self._held[3] is not None:
            self.arm.release(self._held[3])
        self._held = None

    def _planner(self, tier: str):
        if tier not in self._mp:
            import torch
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            t0 = time.perf_counter()
            robot = self.robot if tier == "curobo" else self.robot_relaxed
            mp = MotionPlanner(MotionPlannerCfg.create(robot=robot, scene_model=self._scene_dict,
                                                       collision_cache=self.cache))
            mp.warmup(enable_graph=True, num_warmup_iterations=2)
            torch.cuda.synchronize()
            self.setup_s[tier] = time.perf_counter() - t0
            self._mp[tier] = mp
            if self._held is not None:
                self._attach(mp)
        return self._mp[tier]

    def _js(self, q):
        import torch
        from curobo.types import JointState
        return JointState.from_position(
            torch.as_tensor(np.atleast_2d(q), dtype=torch.float32, device="cuda"),
            joint_names=self.joint_names)

    # ---- verification -------------------------------------------------------
    def verify(self, path) -> dict:
        if self._verify_fn is not None:
            return self._verify_fn(path)
        dense = densify(np.asarray(path, float), self.check_step)
        scene_hits = self_hits = 0
        for q in dense:
            self.arm.goto_given_conf(q)
            if any(l.cmodel.is_mcdwith(self._wobs) for l in self.parts.values()):
                scene_hits += 1
            if any(self.parts[a].cmodel.is_mcdwith([self.parts[b].cmodel])
                   for a, b in self.self_pairs):
                self_hits += 1
        self.arm.goto_given_conf(np.zeros(6))
        return dict(states=len(dense), scene_hits=scene_hits, self_hits=self_hits,
                    ok=scene_hits == 0 and self_hits == 0)

    # ---- tiers --------------------------------------------------------------
    def _curobo(self, tier, start, goal):
        import torch
        mp = self._planner(tier)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = mp.plan_cspace(self._js(goal), self._js(start))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if r is None or not bool(r.success.any()):
            return None, dt
        return r.get_interpolated_plan().position.reshape(-1, 6).cpu().numpy(), dt

    def _rrt(self, start, goal, seed: int = 0):
        import random
        import wrs.motion.probabilistic.rrt_connect as rrtc
        np.random.seed(seed)
        random.seed(seed)
        t0 = time.perf_counter()
        # Endpoints come from verified keyposes; skip WRS's AABB endpoint test, which
        # is ~28% false-positive against the mesh truth and would reject them.
        md = rrtc.RRTConnect(self.arm).plan(start_conf=np.asarray(start, float),
                                            goal_conf=np.asarray(goal, float),
                                            obstacle_list=self._wobs, start_obstacle_list=[],
                                            goal_obstacle_list=[], **self.rrt_kw)
        dt = time.perf_counter() - t0
        self.arm.goto_given_conf(np.zeros(6))
        return (None if md is None else np.asarray(md.jv_list, float)), dt

    def plan(self, start, goal, *, skip=()) -> PlanResult:
        res = PlanResult(False, None, None, 0.0)
        for tier in ("curobo", "curobo_relaxed", "wrs_rrt"):
            if tier in skip:
                continue
            path, dt = self._rrt(start, goal) if tier == "wrs_rrt" else \
                self._curobo(tier, start, goal)
            res.plan_s += dt
            if path is None:
                res.tiers.append((tier, "no_path", dt))
                continue
            t0 = time.perf_counter()
            v = self.verify(path)
            res.verify_s += time.perf_counter() - t0
            if not v["ok"]:
                res.tiers.append((tier, _reject_msg(v), dt))
                continue
            res.tiers.append((tier, "accepted", dt))
            res.success, res.path, res.source = True, path, tier
            return res
        return res

    # ---- batch --------------------------------------------------------------
    def plan_batch(self, problems) -> list:
        """``problems``: list of (table_scene, start, goal). One cuRobo call for all,
        then per-problem fallback. Returns PlanResult per problem (batch time is
        split evenly across problems in ``plan_s``)."""
        import torch
        from curobo.batch_motion_planner import BatchMotionPlanner
        from curobo.motion_planner import MotionPlannerCfg

        n = len(problems)
        t0 = time.perf_counter()
        cfg = MotionPlannerCfg.create(robot=self.robot,
                                      scene_model=[p[0].curobo_scene_dict() for p in problems],
                                      collision_cache=self.cache, max_batch_size=n,
                                      multi_env=True)
        bp = BatchMotionPlanner(cfg)
        bp.warmup()
        torch.cuda.synchronize()
        self.setup_s["curobo_multienv"] = time.perf_counter() - t0
        starts = np.stack([p[1] for p in problems])
        goals = np.stack([p[2] for p in problems])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = bp.plan_cspace(self._js(goals), self._js(starts), max_attempts=3)
        torch.cuda.synchronize()
        batch_s = time.perf_counter() - t0
        paths = [None] * n
        if r is not None:
            succ = r.success.reshape(n, -1).any(dim=-1).cpu().numpy()
            traj = r.interpolated_trajectory.position.reshape(n, -1, 6)
            last = r.interpolated_last_tstep.reshape(n, -1)[:, 0].cpu().numpy()
            for i in range(n):
                if succ[i]:
                    paths[i] = traj[i, : int(last[i]) + 1].cpu().numpy()
        del bp
        torch.cuda.empty_cache()
        out = []
        for i, (scene, start, goal) in enumerate(problems):
            self.set_scene(scene)
            res = PlanResult(False, None, None, batch_s / n)
            if paths[i] is not None:
                t1 = time.perf_counter()
                v = self.verify(paths[i])
                res.verify_s += time.perf_counter() - t1
                if v["ok"]:
                    res.success, res.path, res.source = True, paths[i], "curobo_multienv"
                    res.tiers.append(("curobo_multienv", "accepted", batch_s / n))
                    out.append(res)
                    continue
                res.tiers.append(("curobo_multienv",
                                  _reject_msg(v), 0.0))
            else:
                res.tiers.append(("curobo_multienv", "no_path", 0.0))
            rest = self.plan(start, goal, skip=("curobo",))
            rest.plan_s += res.plan_s
            rest.verify_s += res.verify_s
            rest.tiers = res.tiers + rest.tiers
            out.append(rest)
        return out
