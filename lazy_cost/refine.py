"""Lazy refinement: batched cuRobo planning of the few segments the bounds cannot settle.

Two BatchMotionPlanners share one collision world (``multi_env=False``, so the PRM graph
seeds are available). Measured on this machine, a batched TrajOpt call costs ~75 ms per
problem whatever the batch size (no amortisation) and pads to the full batch, so the
default batch is 1: segments are refined one at a time, cheapest bound first, and the
top-K threshold is updated after every segment.

  transit   empty hand, previous retreat/HOME -> prepick
  transfer  holding the part (spheres fitted once per part and attached per grasp; the
            object-to-TCP offset depends only on the grasp, so one attachment serves every
            candidate planned with that grasp)

Transit is planned in the context world WITHOUT the part being picked (its pose differs
per candidate); each returned path is then checked against that part's spheres at the
candidate pose, and the few that touch it are re-planned one by one with the part added.
Segment time = time-optimal parameterisation (``topp``: per-joint velocity + acceleration
limits, rest to rest) of the planned path, the same limits as the lower bound.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np


class CuroboRefiner:
    FINGERS = ("finger_left", "finger_right")

    def __init__(self, chk, *, batch=1, attach_spheres=32, max_attempts=3, cache=None, method="trajopt", seeds=2,
                 fast=(("finetune", 0), ("iters", 50)), detax=False):
        """method: "trajopt" (graph-seeded TrajOpt, the cuRobo default; used) or "graph" (PRM
        roadmap path only, shortcut with the batched checker, TrajOpt only as fallback). The
        roadmap variant was measured at ~110 ms per segment and over-estimates times (p90 +12-31%),
        ranking worse than the plain lower bound, so it is kept only for comparison (README)."""
        import torch
        from curobo.batch_motion_planner import BatchMotionPlanner
        from curobo.motion_planner import MotionPlannerCfg
        from curobo.scene import Scene
        from curobo_bridge.curobo_models import load_robot_dict, with_attach_link, with_buffer
        from curobo_bridge.runtime import ASSET_DIR

        self.chk, self.batch, self.max_attempts = chk, int(batch), int(max_attempts)
        self.method = method
        # detax: time a planned path as min(TOPP(path), TOPP(greedy shortcut), TOPP(corner-blended
        # shortcut)) with our checker. cuRobo's own paths cost ~1.2x the bound even on free lines
        # (lazy_cost/_design/detour_experiments.md); the robot executes the smoothed path.
        self.detax = bool(detax)
        # first pass: ONE TrajOpt pass of 50 L-BFGS iterations, 2 seeds (bench_trajopt.py: ~75 ms per
        # segment instead of 290-470 ms; topp time p90 +1-2%, max +4-8%, no failures in 80 segments).
        # The time-optimal fine-tune pass is skipped: the path is re-timed with topp anyway.
        # Failures fall back to the full planner (3 attempts, graph seeds). fast=None: full planner only.
        self.fast = None if fast is None else dict(fast)
        self.n_attach = int(attach_spheres)
        cache = cache or {"mesh": 24, "cuboid": 24}
        base = with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", getattr(chk, "density", 1.0)), chk.buffer)
        for link, rl in getattr(chk, "self_relief", {}).items():          # same self-collision relief as the checker
            pad = base["kinematics"].setdefault("self_collision_buffer", {})
            pad[link] = float(pad.get(link, 0.0)) - float(rl)
        # The planner's own fingers are rigid and fully open. While holding a part the real gripper is
        # closed to the grasp's jaw, so for the transfer planner the finger spheres are disabled
        # (negative radius) and re-added, closed, as extra spheres of the attached object (_attach).
        import copy
        held_base = copy.deepcopy(base)
        for link in self.FINGERS:
            for sp in held_base["kinematics"]["collision_spheres"][link]:
                sp["radius"] = -1.0
        self.n_finger = sum(len(base["kinematics"]["collision_spheres"][l]) for l in self.FINGERS)
        held = with_attach_link(held_base, self.n_attach + self.n_finger)
        t0 = perf_counter()
        d = chk.world_dict()
        self.mp = {}
        for name, robot in (("transit", base), ("transfer", held)):
            mp = BatchMotionPlanner(MotionPlannerCfg.create(robot=robot, scene_model=d, collision_cache=cache,
                                                            max_batch_size=self.batch, num_trajopt_seeds=seeds))
            mp.warmup(enable_graph=True, num_warmup_iterations=2)
            self.mp[name] = mp
        torch.cuda.synchronize()
        self.setup_s = perf_counter() - t0
        self.joint_names = self.mp["transit"].joint_names
        self._fit = {}
        self._fingers = None
        self.stats = dict(batches=0, segments=0, replans=0, plan_s=0.0, graph_s=0.0, shortcut_s=0.0,
                          graph_calls=0, graph_fallback=0)
        self._world, self._names = None, set()
        self._Scene = Scene

    # ---- world / held part ----------------------------------------------------------------------
    def set_world(self, d: dict) -> None:
        if d is self._world:
            return
        t0 = perf_counter()
        src = d
        d = self._canonical(d)
        names = {k for kind in ("mesh", "cuboid") for k in d.get(kind, {})}
        removed = self._names - names if self._world is not None else set()
        for mp in self.mp.values():
            # In BatchMotionPlanner an obstacle removed by update_world can keep acting (a part that
            # had been on the table at its pick pose made the later lift of that very part start
            # "in collision"; clear_scene_cache / enable_obstacle(False) did not always help).
            # Parts therefore keep ONE fixed obstacle name each and are parked far below the table
            # when absent (_canonical), so part obstacles are never removed; only other obstacles
            # (boxes) can be, and those are disabled first.
            for name in removed:
                try:
                    mp.scene_collision_checker.enable_obstacle(name, False)
                except Exception:
                    pass
            mp.update_world(self._Scene.create(d))
        self._world, self._names = src, names
        self.stats["world_s"] = self.stats.get("world_s", 0.0) + perf_counter() - t0

    PARK = np.array([[1.0, 0, 0, 0.0], [0, 1.0, 0, 0.0], [0, 0, 1.0, -5.0], [0, 0, 0, 1.0]])

    def _canonical(self, d):
        """Part meshes renamed to the part id ('post_A@goal', 'post_A@0.1,0.2,yaw0', 'post_A'
        -> 'post_A'); every part absent from the world is added, parked 5 m below the table."""
        ids = self.chk.scene.ids
        mesh = {}
        for name, entry in d.get("mesh", {}).items():
            pid = name.split("@")[0]
            mesh[pid if pid in ids else name] = entry
        for pid in ids:
            if pid not in mesh:
                mesh[pid] = self.chk.mesh_entry(pid, self.PARK)
        return dict(d, mesh=mesh)

    def fit(self, pid):
        """Held-part spheres (MorphIt fit of the mesh, inflated to enclose every surface
        sample + 1 mm), once per part."""
        if pid in self._fit:
            return self._fit[pid]
        t0 = perf_counter()
        import torch
        from curobo.scene import Mesh
        mesh = self.chk.scene.meshes[pid]
        am = self.mp["transfer"].attachment_manager
        ob = Mesh(name="held_object", vertices=np.asarray(mesh.vertices).tolist(),
                  faces=np.asarray(mesh.faces).reshape(-1, 3).tolist(), pose=[0, 0, 0, 1, 0, 0, 0])
        sph = am.fit_spheres([ob], num_spheres=self.n_attach)
        pts = torch.as_tensor(mesh.sample(3000, seed=0), dtype=sph.dtype, device=sph.device)
        gap = (torch.cdist(pts, sph[:, :3]) - sph[None, :, 3]).min(dim=1).values
        sph = sph.clone()
        sph[:, 3] += float(torch.clamp(gap.max(), min=0.0)) + 0.001
        self._fit[pid] = sph
        self.stats["fit_s"] = self.stats.get("fit_s", 0.0) + perf_counter() - t0
        return sph

    def _js(self, q):
        import torch
        from curobo.types import JointState
        return JointState.from_position(torch.as_tensor(np.atleast_2d(q), dtype=torch.float32, device="cuda"),
                                        joint_names=self.joint_names)

    def _finger_spheres(self, jaw=None):
        """Finger spheres [n_finger, 4] in the TCP frame, closed to ``jaw`` (None = fully open), the
        same model as the checker (both fingers moved towards each other along TCP y)."""
        import torch
        if self._fingers is None:
            sph, R, p = self.chk.kinematics(torch.zeros((1, 6), dtype=torch.float32, device="cuda"))
            self._fingers = [(side, torch.cat([(sph[0, idx, :3] - p[0]) @ R[0], sph[0, idx, 3:]], -1))
                             for side, idx in self.chk.finger.items()]
        out = []
        for side, loc in self._fingers:
            loc = loc.clone()
            if jaw is not None:
                loc[:, 1] -= side * max(0.0, (self.chk.JAW_OPEN - float(jaw)) / 2)
            out.append(loc)
        return torch.cat(out)

    def _attach(self, pid, G, q_ref, jaw=None):
        import torch
        from curobo.types import Pose
        from scipy.spatial.transform import Rotation
        _, R, p = self.chk.kinematics(torch.as_tensor(np.atleast_2d(q_ref), dtype=torch.float32, device="cuda"))
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[0].cpu().numpy(), p[0].cpu().numpy()
        obj = T @ np.linalg.inv(G)
        xyzw = Rotation.from_matrix(obj[:3, :3]).as_quat()
        off = Pose(position=torch.as_tensor(obj[None, :3, 3], dtype=torch.float32, device="cuda"),
                   quaternion=torch.as_tensor([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]], dtype=torch.float32, device="cuda"))
        fs = self._finger_spheres(jaw)                                     # TCP frame -> object frame: G @ p
        Gt = torch.as_tensor(G, dtype=fs.dtype, device=fs.device)
        fs = torch.cat([fs[:, :3] @ Gt[:3, :3].T + Gt[:3, 3], fs[:, 3:]], -1)
        sph = self.fit(pid)
        self.mp["transfer"].attachment_manager.update(torch.cat([sph, fs.to(sph.dtype)]), self._js(q_ref),
                                                      link_name="attached_object", world_objects_pose_offset=off)

    # ---- planning -------------------------------------------------------------------------------
    def _call(self, mp, chunk, attempts, graph_from):
        import torch
        m = len(chunk)
        A = np.stack([a for a, _ in chunk] + [chunk[-1][0]] * (self.batch - m))
        B = np.stack([b for _, b in chunk] + [chunk[-1][1]] * (self.batch - m))
        torch.cuda.synchronize()
        t0 = perf_counter()
        if self.fast is not None and attempts == 1:
            # light pass: the path SHAPE is all we need (it is re-timed with topp), so skip
            # the time-optimal fine-tune pass and cap iterations / seeds
            r = mp.trajopt_solver.solve_cspace(self._js(B), self._js(A), finetune_attempts=self.fast.get("finetune", 0),
                                               initial_iters=self.fast.get("iters"), num_seeds=self.fast.get("seeds"))
        else:
            r = mp.plan_cspace(self._js(B), self._js(A), max_attempts=attempts, enable_graph_attempt=graph_from)
        torch.cuda.synchronize()
        self.stats["plan_s"] += perf_counter() - t0
        self.stats["batches"] += 1
        if r is None:
            return [None] * m
        succ = r.success.reshape(self.batch, -1).any(-1).cpu().numpy()
        traj = r.interpolated_trajectory.position.reshape(self.batch, -1, 6)
        last = r.interpolated_last_tstep.reshape(self.batch, -1)[:, 0].cpu().numpy()
        return [traj[i, :int(last[i]) + 1].cpu().numpy() if succ[i] else None for i in range(m)]

    # ---- roadmap-only planning ---------------------------------------------------------------------
    def _shortcut(self, P, held=None, extras=None):
        """Greedy shortcut of a waypoint path with the batched checker (from each kept
        waypoint jump to the farthest one reachable by a free joint line). None if even the
        roadmap's own edges collide under our model (e.g. the part being picked)."""
        t0 = perf_counter()
        out, i = [P[0]], 0
        while i < len(P) - 1:
            js = list(range(len(P) - 1, i, -1))
            ok = self.chk.lines_free([np.stack([P[i], P[j]]) for j in js], held=held, extras=extras,
                                     owner=None if extras is None else [0] * len(js), step=0.02)
            if not ok.any():
                self.stats["shortcut_s"] += perf_counter() - t0
                return None
            i = js[int(np.argmax(ok))]
            out.append(P[i])
        self.stats["shortcut_s"] += perf_counter() - t0
        return np.stack(out)

    def _graph(self, mp, pairs, validate):
        import torch
        A = torch.as_tensor(np.stack([a for a, _ in pairs]), dtype=torch.float32, device="cuda")
        B = torch.as_tensor(np.stack([b for _, b in pairs]), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        t0 = perf_counter()
        r = mp.graph_planner.find_path(A, B, interpolate_waypoints=False)
        torch.cuda.synchronize()
        self.stats["graph_s"] += perf_counter() - t0
        self.stats["graph_calls"] += 1
        out = []
        for i in range(len(pairs)):
            P = None
            if bool(r.success[i]) and r.plan_waypoints[i] is not None:
                P = r.plan_waypoints[i].reshape(-1, 6).cpu().numpy().astype(float)
                P = np.r_[pairs[i][0][None], P, pairs[i][1][None]]          # exact endpoints
                P = validate(i, P)
            out.append(P)
        return out

    def _plan(self, mp, pairs, validate=None):
        """pairs: [(qa, qb)] -> paths (or None). "graph": roadmap path, shortcut and validated
        by ``validate(i, waypoints)``; the rest falls back to TrajOpt."""
        if self.method != "graph":
            return self._plan_trajopt(mp, pairs)
        out = self._graph(mp, pairs, validate or (lambda i, P: P))
        fail = [i for i, P in enumerate(out) if P is None]
        self.stats["graph_fallback"] += len(fail)
        if fail:
            for i, P in zip(fail, self._plan_trajopt(mp, [pairs[i] for i in fail])):
                out[i] = P
            self.stats["segments"] -= len(fail)
        self.stats["segments"] += len(pairs)
        return out

    def _plan_trajopt(self, mp, pairs):
        """pairs: [(qa, qb)] -> dense paths (or None). One TrajOpt attempt per batch of
        ``self.batch`` (a retry would re-solve the whole batch); the failures are then
        retried together with graph seeding and up to ``max_attempts`` attempts."""
        out = []
        for s in range(0, len(pairs), self.batch):
            out += self._call(mp, pairs[s:s + self.batch], 1, 1)
        fail = [i for i, P in enumerate(out) if P is None]
        for s in range(0, len(fail), self.batch):
            idx = fail[s:s + self.batch]
            for i, P in zip(idx, self._call(mp, [pairs[i] for i in idx], self.max_attempts, 0)):
                out[i] = P
        self.stats["segments"] += len(pairs)
        self.stats["retried"] = self.stats.get("retried", 0) + len(fail)
        return out

    def transit(self, world, pid, items, part_spheres):
        """items: [(qa, qb, pose of the part ``pid`` being picked)] -> durations (inf if no path)."""
        from lazy_cost.timing import topp
        self.set_world(world)
        self.chk.set_world(world)
        ex = [self.chk.obstacle_spheres(part_spheres, pose) for _, _, pose in items]
        paths = self._plan(self.mp["transit"], [(a, b) for a, b, _ in items],
                           lambda i, P: self._shortcut(P, extras=[ex[i]]))
        out, self.last_paths = [], []
        for (a, b, pose), P in zip(items, paths):
            if P is not None:
                ex = [self.chk.obstacle_spheres(part_spheres, pose)]
                self.chk.set_world(world)
                if not self.chk.lines_free([P], extras=ex, owner=[0], step=0.02)[0]:
                    self.stats["replans"] += 1      # path touches the part being picked
                    d = dict(world, mesh=dict(world["mesh"], **{pid: self.chk.mesh_entry(pid, pose)}))
                    self.set_world(d)
                    P = self._plan(self.mp["transit"], [(a, b)])[0]
                    self.set_world(world)
            self.last_paths.append(P)                 # the path timed (validate.full_check executes it)
            out.append(np.inf if P is None else self._time(P, extras=[self.chk.obstacle_spheres(part_spheres, pose)]))
        return out

    def transfer(self, world, pid, G, items, jaw=None):
        """items: [(qa, qb)] holding ``pid`` with grasp G, gripper closed to ``jaw`` (None = open)
        -> durations (inf if no path)."""
        from lazy_cost.timing import topp
        self.set_world(world)
        self._attach(pid, G, items[0][0], jaw)
        self.chk.set_world(world)
        held = (self.chk.scene.held_spheres(pid), G)
        paths = self._plan(self.mp["transfer"], items, lambda i, P: self._shortcut(P, held=held))
        self.mp["transfer"].attachment_manager.detach(link_name="attached_object")
        self.last_paths = list(paths)
        return [np.inf if P is None else self._time(P, held=held) for P in paths]

    def _time(self, P, held=None, extras=None):
        """TOPP time of a planned path; with detax the min over the path, its greedy shortcut and
        the corner-blended shortcut (if that curve is free). The checker's world must be set."""
        from lazy_cost.detour import blend
        from lazy_cost.timing import topp
        t = float(topp(P)[1][-1])
        if not self.detax or len(P) < 3:
            return t
        t0 = perf_counter()
        if len(P) > 60:
            P = P[np.unique(np.linspace(0, len(P) - 1, 60).round().astype(int))]
        S = self._shortcut(P, held=held, extras=extras)
        if S is not None:
            t = min(t, float(topp(S)[1][-1]))
            if len(S) > 2:
                Bc = blend(S)
                if self.chk.lines_free([Bc], held=held, extras=extras, owner=None if extras is None else [0], step=0.02)[0]:
                    t = min(t, float(topp(Bc)[1][-1]))
        self.stats["detax_s"] = self.stats.get("detax_s", 0.0) + perf_counter() - t0
        return t
