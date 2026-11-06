"""Final step: full-process validation of the plans the search ranked best.

For a complete plan, every motion segment in forward order:
    transit_k   start -> hover(x_k, g_k)    start = HOME (k = 1) or retreat(part k-1)
    fixed_k     descent + pick + lift + insert + release + retreat (fixed transition points)
    transfer_k  lift(x_k, g_k) -> preins(g_k), holding part k
    return      retreat(part n) -> HOME
World of step k: parts 1..k-1 installed, parts k+1..n on the table at their poses, part k at x_k.

Validation of one plan (Validator.validate), nothing taken from the search:
1. every transit / transfer / return segment is planned with cuRobo (light TrajOpt, retried with
   the full planner) - a segment cuRobo cannot solve fails the plan. It is executed along its
   straight joint line if that line is free (time = its lower bound, exact), else along the cuRobo
   path (re-timed with topp);
2. the WHOLE executed motion is collision-checked against the meshes (full_check): every free
   segment along the path it is executed on, every fixed motion (descent, grasp, lift, insertion,
   release, retreat) with its own gripper state and the part it touches. The search screens the
   fixed motions with sphere covers and leaves the part being picked / released out of some of
   them (expand.py), so only this check sees, e.g., a finger under a shelf rising through it.
Plans are validated in order of their estimate; cuRobo results are cached and shared between
plans. The loop stops at the first plan that passes whose time is no slower than the lower bound
of every plan not yet validated (LazySP idea); a plan that fails is rejected.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .problem import HOME, t_acc


def forward_steps(pb, node):
    """Forward segments of a complete plan: list of dicts with worlds and endpoints."""
    cat, sc = pb.cat, pb.scene
    steps = list(node.suffix)
    segs = []
    for k, (p, x, g) in enumerate(steps):
        installed = [q for q, _, _ in steps[:k]]
        table = {q: xx for q, xx, _ in steps[k + 1:]}
        e, v = cat["pick"][p][x][g], cat["goal"][p][g]
        start = HOME if k == 0 else cat["goal"][steps[k - 1][0]][steps[k - 1][2]]["retreat"]
        segs.append(dict(kind="transit", part=p, installed=installed, table=table, a=start, b=e["hover"], g=g, x=x))
        segs.append(dict(kind="fixed", part=p, value=pb.fix_pick[p, x, g] + pb.fix_goal[p, g]))
        segs.append(dict(kind="transfer", part=p, installed=installed, table=table, a=e["lift"], b=v["preins"], g=g, x=x))
    last = steps[-1]
    segs.append(dict(kind="return", part=None, installed=[q for q, _, _ in steps], table={},
                     a=cat["goal"][last[0]][last[2]]["retreat"], b=HOME))
    return segs


class Validator:
    def __init__(self, pb, refiner):
        self.pb, self.ref = pb, refiner
        self.cache, self.paths = {}, {}
        self.stats = dict(planned=0, plan_s=0.0, cache_hits=0)
        self._mesh = None

    def _jaw(self, s):
        return self.pb.jaw[s["part"]][s["g"]] if self.pb.close_fingers else None

    def _key(self, s):
        return (s["kind"], s["part"], s.get("g"), tuple(sorted(s["installed"])), tuple(sorted(s["table"].items())),
                np.round(s["a"], 5).tobytes(), np.round(s["b"], 5).tobytes())

    def _straight_free(self, s):
        """Same check as during the search (target part as an extra obstacle on transits)."""
        pb, chk, sc = self.pb, self.pb.chk, self.pb.scene
        chk.set_world(chk.world_dict(s["installed"], pb.placed(s["table"])))
        line = (np.asarray(s["a"])[None], np.asarray(s["b"])[None])
        if s["kind"] == "transit":
            ex = [chk.obstacle_spheres(pb.held[s["part"]], sc.pose[s["part"]][s["x"]])]
            return bool(chk.batch_free([(line[0], line[1], -1, 0)], extra_bank=ex)[0][0])
        if s["kind"] == "transfer":
            held = [(pb.held[s["part"]], sc.grasps()[s["part"]][s["g"]][0])]
            jaw = self._jaw(s)
            return bool(chk.batch_free([(line[0], line[1], 0, -1, None if jaw is None else np.array([jaw]))],
                                       held_bank=held)[0][0])
        return bool(chk.batch_free([(line[0], line[1], -1, -1)])[0][0])

    def plan_time(self, s):
        """cuRobo (light TrajOpt) time of one segment, cached."""
        k = self._key(s)
        if k in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[k]
        pb, sc, ref = self.pb, self.pb.scene, self.ref
        W = pb.chk.world_dict(s["installed"], pb.placed(s["table"]))
        t0 = perf_counter()

        def plan():
            if s["kind"] == "transit":
                v = ref.transit(W, s["part"], [(s["a"], s["b"], sc.pose[s["part"]][s["x"]])], pb.held[s["part"]])[0]
                self.paths[k] = ref.last_paths[0]
                return v
            if s["kind"] == "transfer":
                v = ref.transfer(W, s["part"], sc.grasps()[s["part"]][s["g"]][0], [(s["a"], s["b"])], self._jaw(s))[0]
                self.paths[k] = ref.last_paths[0]
                return v
            ref.set_world(W)
            pb.chk.set_world(W)
            P = ref._plan(ref.mp["transit"], [(s["a"], s["b"])])[0]
            self.paths[k] = P
            return np.inf if P is None else ref._time(P)
        v = plan()
        retries = 0
        while not np.isfinite(v) and retries < 3:
            # cuRobo occasionally fails a segment it solves on a repeat call (seen only inside the
            # full pipeline, not reproducible in isolation). Reload the world and try again with
            # the full planner (several attempts, graph seeds).
            retries += 1
            ref._world, fast = None, ref.fast
            ref.fast = None
            v = plan()
            ref.fast = fast
        if retries:
            self.stats.setdefault("retried_segments", []).append(dict(kind=s["kind"], part=s["part"], retries=retries,
                                                                      solved=bool(np.isfinite(v))))
        self.stats["planned"] += 1
        self.stats["plan_s"] += perf_counter() - t0
        self.cache[k] = float(v)
        return float(v)

    def validate(self, node):
        """Full-process validation of one plan -> (ok, time, rows, check rows). rows: per segment,
        value = executed time, planned = cuRobo time, how = straight / planned / fixed."""
        rows, total, ok = [], 0.0, True
        for s in forward_steps(self.pb, node):
            if s["kind"] == "fixed":
                v, planned, how, lb = s["value"], s["value"], "fixed", s["value"]
            else:
                lb = float(t_acc(s["a"], s["b"]))
                planned = self.plan_time(s)
                ok &= bool(np.isfinite(planned))
                v, how = (lb, "straight") if self._straight_free(s) else (planned, "planned")
            total += v
            rows.append(dict(kind=s["kind"], part=s["part"], value=v, planned=planned, how=how, lb=lb))
        if not ok:
            return False, np.inf, rows, []
        mesh_ok, crows = self.full_check(node)
        return mesh_ok, (total if mesh_ok else np.inf), rows, crows

    # ---- full collision check of the executed motion (meshes) -----------------------------------
    def full_check(self, node):
        """Mesh test (WRS / ODE, joint_search.wrs_check.MeshWorld) of the whole forward execution:
        free segments along the path they are executed on (straight line if our checker finds it
        free, else its cuRobo path), fixed motions as straight joint lines:
            descent   hover -> pick      gripper open, part p on the table at x
            grasp     pick               gripper closed to the grasp's jaw width, p at x
            lift      pick -> lift       holding p (held part not checked against the table)
            insert    preins -> ins      holding p (held part not checked against the base and the
                                         installed parts: it reaches its seat; scene.step_ok sweeps it)
            release   ins                gripper open, p installed
            retreat   ins -> retreat     gripper open, p installed
        -> (ok, rows); rows: one per motion with the first collision found."""
        from types import SimpleNamespace
        from .wrs_check import MeshWorld
        pb, sc, cat = self.pb, self.pb.scene, self.pb.cat
        if self._mesh is None:
            self._mesh = MeshWorld(pb)
        W, G = self._mesh, sc.grasps()
        steps = list(node.suffix)
        segs = {(s["kind"], s["part"]): s for s in forward_steps(pb, SimpleNamespace(suffix=steps))}
        rows, ok_all = [], True

        def executed(s):
            if self._straight_free(s):
                return np.stack([s["a"], s["b"]]), "straight"
            P = self.paths.get(self._key(s))
            return (None, "no path") if P is None else (np.asarray(P, float), "planned")

        def run(step, part, motion, P, installed, table, held=None, jaw=None, held_skip=(), how="fixed"):
            nonlocal ok_all
            if P is None:
                ok, why = False, "no executed path"
            else:
                W.set(installed, table, held, jaw=jaw, held_skip=held_skip)
                ok, why = W.free(P)
            ok_all &= ok
            rows.append(dict(step=step, part=part, motion=motion, path=how, ok=bool(ok), collision=why))

        for k, (p, x, g) in enumerate(steps):
            inst = [q for q, _, _ in steps[:k]]
            later = {q: sc.pose[q][xx] for q, xx, _ in steps[k + 1:]}
            here = dict(later, **{p: sc.pose[p][x]})
            e, v = cat["pick"][p][x][g], cat["goal"][p][g]
            Gp = G[p][g][0]
            jaw = pb.jaw[p][g] if pb.close_fingers else None
            P, how = executed(segs["transit", p])
            run(k + 1, p, "transit", P, inst, here, how=how)
            run(k + 1, p, "descent", np.stack([e["hover"], e["pick"]]), inst, here)
            run(k + 1, p, "grasp", np.stack([e["pick"]]), inst, here, jaw=jaw)
            run(k + 1, p, "lift", np.stack([e["pick"], e["lift"]]), inst, later, (p, Gp, e["pick"]), jaw, ("table",))
            P, how = executed(segs["transfer", p])
            run(k + 1, p, "transfer", P, inst, later, (p, Gp, e["lift"]), jaw, how=how)
            run(k + 1, p, "insert", np.stack([v["preins"], v["ins"]]), inst, later, (p, Gp, v["preins"]), jaw, ("installed",))
            run(k + 1, p, "release", np.stack([v["ins"]]), inst + [p], later)
            run(k + 1, p, "retreat", np.stack([v["ins"], v["retreat"]]), inst + [p], later)
        P, how = executed(segs["return", None])
        run(len(steps) + 1, None, "return", P, [q for q, _, _ in steps], {}, how=how)
        return ok_all, rows


def select(pb, plans, refiner, log=print):
    """Choice among the search's complete plans (distinct order + layout): validated (cuRobo plans
    + mesh check of the whole execution) in order of their estimate; the fastest plan that passes,
    once it is no slower than the bound of every plan not validated yet."""
    val = Validator(pb, refiner)
    t0 = perf_counter()
    order = sorted(plans, key=lambda n: n.g_est)
    best, best_t, tried, rejected = None, np.inf, [], []
    for i, n in enumerate(order):
        if best is not None and best_t <= min(m.g_lb for m in order[i:]):
            break
        ok, t, rows, crows = val.validate(n)
        bad = [r for r in crows if not r["ok"]]
        failed = [r for r in rows if not np.isfinite(r["planned"])]
        tried.append(dict(order=[s[0] for s in n.suffix], est=n.g_est, lb=n.g_lb, time=t, ok=ok,
                          planned=sum(r["how"] == "planned" for r in rows)))
        why = ("OK" if ok else
               f"cuRobo failed on {[(r['kind'], r['part']) for r in failed]}" if failed else
               f"{len(bad)} colliding motions: " + "; ".join(f"step {r['step']} {r['part']} {r['motion']} ({r['collision']})"
                                                            for r in bad))
        log(f"  plan {i + 1}: estimate {n.g_est:.3f} s (bound {n.g_lb:.3f}) -> "
            + (f"{t:.3f} s ({tried[-1]['planned']} segments on cuRobo paths), full-process validation OK" if ok else
               f"REJECTED: {why}"))
        if not ok:
            rejected.append(dict(plan=i + 1, reason="curobo" if failed else "collision",
                                 failed=[(r["kind"], r["part"]) for r in failed], collisions=bad))
        elif t < best_t:
            best, best_t, best_rows, best_crows = n, t, rows, crows
    if best is None:
        return dict(best=None, time=np.inf, rows=[], check_rows=[], tried=tried, rejected=rejected,
                    validate_s=perf_counter() - t0, stats=val.stats)
    return dict(best=best, time=best_t, rows=best_rows, check_rows=best_crows, tried=tried, rejected=rejected,
                validate_s=perf_counter() - t0, stats=val.stats)
