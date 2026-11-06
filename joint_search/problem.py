"""Problem data shared by the search: scene, collision checker, IK catalogue, correction model,
and the layout-independent lower-bound tables.

Step of part p at candidate x with assembly grasp g (all joint times rest-to-rest, s):
    transit   start -> hover(x,g)                    start = previous part's retreat or HOME
    fixed     hover -> prepick -> pick -> lift       (vertical descent + grasp + lift)
              + preins(g) -> ins(g) -> retreat(g)    (insert + release)
    transfer  lift(x,g) -> preins(g)                 holding p
Plan time = sum of all steps + retreat(last) -> HOME.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

HOME = np.zeros(6)


def t_acc(a, b):
    """Rest-to-rest joint-time lower bound, broadcast over leading axes."""
    from curobo_bridge.assembly_demo import AMAX, VMAX
    d = np.abs(np.asarray(b, float) - np.asarray(a, float))
    return np.where(d < VMAX ** 2 / AMAX, 2 * np.sqrt(d / AMAX), d / VMAX + VMAX / AMAX).max(-1)




class Problem:
    def __init__(self, model="triangular_frame6", gap=0.01, penalty=None, grasps_per_pose=8,
                 sampler="fps", placements="sealp", orient="first", grid="fine",
                 keepout="sealp", center=(0.373, 0.0, 0.0), grid_step=0.015,
                 home_clear=0.03):
        from curobo_bridge.runtime import prepare
        prepare()
        from lazy_cost.checker import GpuChecker
        from lazy_cost.penalty import Penalty
        from lazy_cost.scene import load_scene
        from .screen import CLOSE_FINGERS, load_or_build

        t0 = perf_counter()
        kw = dict(sampler=sampler, placements=placements, grid=grid, keepout=keepout,
                  grid_step=grid_step)
        self.scene = sc = load_scene(model, center=center, **kw)
        self.chk = GpuChecker(sc)
        self.timing = dict(scene_s=perf_counter() - t0)       # wall time of the setup steps
        t1 = perf_counter()
        self.home_clear = float(home_clear)                  # SEALP DEFAULT_ROBOT_HOME_CLEARANCE
        self.center_report = self._place_assembly_home_free()
        if self.center_report["moved"]:                     # candidates depend on the centre: rebuild
            self.scene = sc = load_scene(model, center=tuple(self.center_report["center"]), **kw)
            self.chk = GpuChecker(sc)
        self.home_report = self._drop_home_colliding()     # before the IK: the sampling is redone without them
        self.timing["home_checks_s"] = perf_counter() - t1
        t1 = perf_counter()
        self.cat, self.cat_source = load_or_build(sc, self.chk)   # IK of the insertion / pick moments (+ transition points)
        self.timing["ik_catalogue_s"] = perf_counter() - t1
        t1 = perf_counter()
        if orient not in ("first", "best"):
            raise ValueError(f"orient must be first or best, not {orient!r}")
        self.orient = orient
        if sampler == "fps" and getattr(sc, "placements", "goal") == "sealp":
            # SEALP: the first n_cand positions (x, y), in farthest-point order, that have an orientation
            # with a usable grasp, each with every such orientation in SEALP order; the search decides
            # the orientation per state (expand.py, orient)
            from lazy_cost.scene import key_xy
            for p in sc.ids:
                keep, pos = [], []
                for x, _ in sc.candidates[p]:
                    if not self.cat["pick"][p].get(x):
                        continue
                    if key_xy(x) not in pos:
                        if len(pos) == sc.n_cand:
                            continue
                        pos.append(key_xy(x))
                    keep.append(x)
                self.cat["pick"][p] = {x: self.cat["pick"][p][x] for x in keep}
        elif sampler == "fps":                             # keep the first n_cand usable poses in FPS order
            for p in sc.ids:
                keep = [x for x, _ in sc.candidates[p] if self.cat["pick"][p].get(x)][:sc.n_cand]
                self.cat["pick"][p] = {x: self.cat["pick"][p][x] for x in keep}
        # grasps screened per candidate pose and expansion, smallest bound first (0 = all; expand.py)
        self.grasps_per_pose = int(grasps_per_pose)
        self.jaw = self.cat["jaw"]
        self.close_fingers = CLOSE_FINGERS
        self.penalty = Penalty() if penalty is None else Penalty(penalty)
        self.gap = float(gap)
        self.ids = tuple(p for p in sc.ids if self.cat["pick"][p] and self.cat["goal"][p])
        if len(self.ids) != len(sc.ids):
            raise ValueError(f"parts without a usable grasp: {set(sc.ids) - set(self.ids)}")
        self.held = {p: sc.held_spheres(p) for p in self.ids}
        cat = self.cat
        # fixed short motions, per (p, x, g) and per (p, g)
        self.fix_pick = {(p, x, g): float(t_acc(e["hover"], e["prepick"]) + t_acc(e["prepick"], e["pick"]) + t_acc(e["pick"], e["lift"]))
                         for p in self.ids for x, gs in cat["pick"][p].items() for g, e in gs.items()}
        self.fix_goal = {(p, g): float(t_acc(v["preins"], v["ins"]) + t_acc(v["ins"], v["retreat"]))
                         for p in self.ids for g, v in cat["goal"][p].items()}
        self._bounds()
        self.timing["bounds_s"] = perf_counter() - t1
        self.setup_s = perf_counter() - t0

    # ---- the finished assembly and the arm at HOME -----------------------------------------------
    def _clear_cover(self, pid):
        """Sphere cover of a part grown by the HOME clearance: the arm at HOME must stay home_clear away
        (SEALP _robot_home_clearance_reason, 3 cm; there on AABBs, here on the sphere covers)."""
        c = self.scene.held_spheres(pid).copy()
        c[:, 3] += self.home_clear
        return c

    def _home_free(self, shift=(0.0, 0.0)):
        """Arm at HOME (gripper open) free of the table and the finished assembly (base + every part at
        its goal; meshes, and the sphere covers grown by home_clear), the assembly translated by
        ``shift`` (x, y)."""
        import torch
        sc, chk = self.scene, self.chk
        d = np.array([shift[0], shift[1], 0.0])
        for T in sc.goal.values():
            T[:3, 3] += d
        try:
            chk.set_world(chk.world_dict(tuple(sc.ids)))
            cover = torch.cat([chk.obstacle_spheres(self._clear_cover(q), sc.goal[q]) for q in ("base",) + tuple(sc.ids)])
            res = chk.batch_free([(HOME[None], None, -1, -1), (HOME[None], None, -1, np.array([0]))], extra_bank=[cover])
            return bool(res[0][0] and res[1][0])                 # meshes; grown covers
        finally:
            for T in sc.goal.values():
                T[:3, 3] -= d

    def _place_assembly_home_free(self, step=0.01, reach=0.20):
        """The plan ends at HOME with every part installed, so the arm at HOME must clear the FINISHED
        assembly (SEALP's centre screening, _assembly_region_reject_reason, checks only the base). If it
        does not, the assembly centre moves to the nearest centre on a grid (step, within reach) where
        it does, with the assembly inside the table and clear of the arm-base keepout (the scene's
        rule). -> dict(moved, center, shift)."""
        from shapely.geometry import MultiPoint, box as sbox
        sc = self.scene
        c0 = sc.goal["base"][:3, 3].copy()
        if self._home_free():
            return dict(moved=False, center=c0.tolist(), shift=[0.0, 0.0])
        plan = np.vstack([np.asarray(sc.meshes[q].vertices) @ sc.goal[q][:3, :3].T + sc.goal[q][:3, 3]
                          for q in ("base",) + tuple(sc.ids)])[:, :2]
        lo = sc.table_center[:2] - sc.table_size[:2] / 2 + 0.01
        hi = sc.table_center[:2] + sc.table_size[:2] / 2 - 0.01
        r = 0.12 if getattr(sc, "keepout", "box") == "sealp" else 0.13
        robot = sbox(-r, -r, r, r)
        k = np.arange(-reach, reach + 1e-9, step)
        offs = sorted(((a, b) for a in k for b in k if a * a + b * b <= reach * reach + 1e-12),
                      key=lambda o: (o[0] ** 2 + o[1] ** 2, abs(o[1]), -o[0]))
        for a, b in offs:
            P = plan + (a, b)
            if (P.min(0) < lo).any() or (P.max(0) > hi).any() or MultiPoint(P).convex_hull.intersects(robot):
                continue
            if self._home_free((a, b)):
                return dict(moved=True, center=np.round(c0 + (a, b, 0.0), 6).tolist(),
                            shift=[round(float(a), 6), round(float(b), 6)])
        raise ValueError(f"the arm at HOME hits the finished assembly at {c0.tolist()} and at every centre "
                         f"within {reach} m")

    # ---- candidate poses the arm at HOME would hit ---------------------------------------------
    def _drop_home_colliding(self, chunk=1024):
        """The plan starts at HOME with every part at its initial pose, so a (position, orientation)
        whose part (sphere cover) comes closer than home_clear to the arm at HOME (gripper open) is
        infeasible. Checked on
        every option of the scene BEFORE the farthest-point sampling (as SEALP's
        _robot_home_collision_reason inside _enumerate_xy_candidates); the sampling is then redone on
        the options left (scene.drop_options), so the IK catalogue is built only for usable poses.
        -> {pid: (kept, removed)} options."""
        sc, chk = self.scene, self.chk
        chk.set_world(chk.world_dict())
        if not chk.batch_free([(HOME[None], None, -1, -1)])[0][0]:
            raise ValueError("HOME collides with the table or the assembly base")
        report = {}
        for p in sc.ids:
            opts, part, bad = sc.options[p], self._clear_cover(p), []
            for i in range(0, len(opts), chunk):
                c = opts[i:i + chunk]
                free = chk.batch_free([(np.tile(HOME, (len(c), 1)), None, -1, np.arange(len(c)))],
                                      extra_bank=[chk.obstacle_spheres(part, T) for _, T, _ in c])[0]
                bad += [k for (k, _, _), ok in zip(c, free) if not ok]
            sc.drop_options(p, bad)
            report[p] = (len(opts) - len(bad), len(bad))
        sc.home_removed, sc.home_clear = sum(r[1] for r in report.values()), self.home_clear
        return report

    # ---- layout-independent lower bounds -------------------------------------------------------
    def _bounds(self):
        """h[p] = min over (x, g) of (transfer + fixed) + min over (x, g, any start) of the transit.
        Any start = HOME or the retreat of any other part with any of its grasps. The two minima
        are taken separately, so h[p] never exceeds the true cost of p's step: admissible."""
        cat = self.cat
        self.h = {}
        for p in self.ids:
            starts = [HOME] + [v["retreat"] for q in self.ids if q != p for v in cat["goal"][q].values()]
            S = np.stack(starts)
            a, b = np.inf, np.inf
            for x, gs in cat["pick"][p].items():
                for g, e in gs.items():
                    if g not in cat["goal"][p]:
                        continue
                    a = min(a, float(t_acc(e["lift"], cat["goal"][p][g]["preins"])) + self.fix_pick[p, x, g] + self.fix_goal[p, g])
                    b = min(b, float(t_acc(S, e["hover"]).min()))
            self.h[p] = a + b

    def h_lb(self, parts) -> float:
        return float(sum(self.h[p] for p in parts))

    # ---- helpers -------------------------------------------------------------------------------
    def placed(self, table, skip=()):
        """table: {pid: candidate key} -> world 'placed' entries for the collision checker."""
        return [(f"{q}@{k}", q, self.scene.pose[q][k]) for q, k in table.items() if q not in skip]

    def preds(self, R1, p):
        """Parts of R1 that can be installed right before p (R1 = everything installed before p)."""
        return [q for q in R1 if self.scene.step_ok(R1 - {q}, q)]
