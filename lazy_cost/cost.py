"""LazyStepCost: rank the candidate initial poses of the part installed in one step.

Step cost of candidate x (best grasp g):
    transit   start -> hover(x, g)            empty hand, joint motion (hover = 8 cm above prepick)
              + hover -> prepick(x, g)        short vertical descent onto the part
    transfer  lift(x, g) -> preins(g)         holding p
All times are rest-to-rest durations under per-joint velocity (VMAX) and acceleration
(AMAX) limits, in seconds. The transit ends ABOVE the part: a joint-straight line that ended
at prepick would sweep the open fingers sideways through the part itself.

  1+3. screen     two batched GPU passes, empty hand and holding p, each over ALL candidates
                  and grasps at once (holding p includes held part vs arm links, and the vertical
                  lift pick -> lift): step stability /
                  insertion; hover, prepick, pick, lift (holding p), preins (holding p)
                  configurations; the descent hover->prepick; and the straight joint lines
                  start->hover and lift->preins. A colliding configuration or descent prunes
                  the (x, g) pair; x is pruned when no grasp survives. A free straight line
                  makes its segment time exact; a blocked one leaves a lower bound to confirm.
  2. lower bound  lb = sum of max_j t_trap(|dq_j|) over the motions: every joint must travel
                  at least |dq_j| from rest to rest, so no path (detour or not) is faster.
  4. lazy refine  (optional) tau = K-th smallest settled cost. Only (x, g) with an unsettled
                  segment and lb < tau can still enter the top K; their blocked segments are
                  planned with cuRobo, cheapest lb first, tau updated after every plan.

``rank(..., refine=False)`` stops after the bounds (search mode: milliseconds, no planner);
with refinement the top-K set and its order equal those of planning every candidate (with
the same planner); unsettled candidates are provably outside the top K.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

HOME = np.zeros(6)
LIFT_SKIP = 0.25        # share of the pick -> lift motion not checked (part still on / at the table)


def t_acc(a, b) -> float:
    from curobo_bridge.assembly_demo import VMAX, t_trap
    return float(np.max(t_trap(np.asarray(b, float) - np.asarray(a, float), VMAX)))


@dataclass
class Ranked:
    key: str
    status: str                  # settled | bound | planner_failed | pruned:<reason>
    cost: float                  # settled cost (s), or the current lower bound if status == bound
    lb: float                    # lower bound before any refinement (joint-time bound)
    grasp: int = -1
    segments: dict = field(default_factory=dict)   # g -> {"transit": (value, how), "transfer": ...}
    features: dict = field(default_factory=dict)   # g -> {"transit": {lb, blocked-ness...}, "transfer": ...}


class LazyStepCost:
    def __init__(self, scene, catalogue, chk, refiner=None, *, k=3, step=0.02):
        self.scene, self.cat, self.chk, self.ref = scene, catalogue, chk, refiner
        self.k, self.step = int(k), float(step)
        self.trace = None            # set to a list to record (ms, event) for visualisation

    def _ev(self, t_start, **kw):
        if self.trace is not None:
            self.trace.append(dict(ms=1e3 * (perf_counter() - t_start), **kw))

    def world(self, installed, placed, boxes):
        return self.chk.world_dict(installed, placed, boxes)

    def rank(self, pid, start, installed, placed, boxes, candidates, refine=True):
        """pid: part installed in this step; start: arm configuration before the step
        (previous part's retreat, or HOME); installed: parts already at their goals;
        placed: (name, pid, 4x4) other parts on the table; boxes: (name, dims, 4x4);
        candidates: candidate keys of pid; refine: False = bounds only (no planner calls).
        Returns (list[Ranked] best first, stats)."""
        T = {"screen": 0.0, "bound": 0.0, "refine": 0.0}
        st = dict(candidates=len(candidates), refined_segments=0, refine_batches=0)
        t0 = perf_counter()
        t_start = t0
        sc, cat = self.scene, self.cat
        if not sc.step_ok(installed, pid):
            return [Ranked(x, "pruned:unstable_or_blocked_insertion", np.inf, np.inf) for x in candidates], st
        W = self.world(installed, placed, boxes)
        self.chk.set_world(W)
        part_sph = sc.held_spheres(pid)
        grasps = sc.grasps()[pid]
        start = np.asarray(start, float)
        out = {}
        # ---- 1+3. one batched screening pass ---------------------------------------------------
        goal = dict(cat["goal"][pid])
        gl = sorted(goal)
        gi = {g: i for i, g in enumerate(gl)}
        xi = {x: i for i, x in enumerate(candidates)}
        rows = [(x, g, e) for x in candidates for g, e in cat["pick"][pid].get(x, {}).items() if g in goal]
        reason = {x: "no_ik" for x in candidates}
        pairs, seg, feat = [], {}, {}
        if rows:
            n = len(rows)
            col = lambda k: np.stack([e[k] for _, _, e in rows])
            X = np.array([xi[x] for x, _, _ in rows])
            Gx = np.array([gi[g] for _, g, _ in rows])
            ck = np.array([e["pick_fixed_free"] for _, _, e in rows])
            none = np.full(n, -1)
            empty, held = self.chk.batch_free(
                [(col("hover"), None, none, X), (col("prepick"), None, none, X), (col("hover"), col("prepick"), none, X),
                 (col("pick")[ck], None, -1, -1), (np.broadcast_to(start, (n, 6)), col("hover"), none, X)],
                extra_bank=[self.chk.obstacle_spheres(part_sph, sc.pose[pid][x]) for x in candidates], step=self.step,
                features=True),                 self.chk.batch_free(
                [(col("lift"), None, Gx, -1), (np.stack([goal[g]["preins"] for g in gl]), None, np.arange(len(gl)), -1),
                 (col("lift"), np.stack([goal[g]["preins"] for _, g, _ in rows]), Gx, -1),
                 # vertical lift pick -> lift holding p; the first quarter (part still resting on
                 # the table, fingers at table height) is skipped, as for the pick configuration
                 (col("pick") + LIFT_SKIP * (col("lift") - col("pick")), col("lift"), Gx, -1)],
                held_bank=[(part_sph, grasps[g][0]) for g in gl], step=self.step, features=True)
            (empty, f_empty), (held, f_held) = empty, held
            ft_tr, ft_tf = f_empty[4], f_held[2]
            ok_hover, ok_pre, ok_desc, ok_pick_c, line_tr = empty
            ok_lift, ok_preins, line_tf, ok_liftpath = held
            ok_pick = np.ones(n, bool)
            ok_pick[ck] = ok_pick_c
            for i, (x, g, e) in enumerate(rows):
                why = ("prepick_collides" if not ok_pre[i] else "hover_collides" if not ok_hover[i]
                       else "descent_collides" if not ok_desc[i] else "pick_collides" if not ok_pick[i]
                       else "lift_collides" if not ok_lift[i] else "lift_path_collides" if not ok_liftpath[i]
                       else "preins_collides" if not ok_preins[gi[g]] else None)
                if why is None:
                    pairs.append((x, g, e))
                    reason[x] = None
                    seg[x, g] = {"transit": [None, "straight" if line_tr[i] else "bound"],
                                 "transfer": [None, "straight" if line_tf[i] else "bound"]}
                    feat[x, g] = {"transit": {k: float(v[i]) for k, v in ft_tr.items()},
                                  "transfer": {k: float(v[i]) for k, v in ft_tf.items()}}
                elif reason[x] is not None:
                    reason[x] = why
        T["screen"] = perf_counter() - t0
        self._ev(t_start, stage="prune", pruned={x: r for x, r in reason.items() if r and x not in {p[0] for p in pairs}})
        # ---- 2. lower bounds -------------------------------------------------------------------
        t0 = perf_counter()
        desc = {}
        for x, g, e in pairs:
            desc[x, g] = t_acc(e["hover"], e["prepick"])
            seg[x, g]["transit"][0] = t_acc(start, e["hover"]) + desc[x, g]
            seg[x, g]["transfer"][0] = t_acc(e["lift"], goal[g]["preins"])
            feat[x, g]["transit"]["lb"] = seg[x, g]["transit"][0]
            feat[x, g]["transfer"]["lb"] = seg[x, g]["transfer"][0]
        lb0 = {xg: sum(v[0] for v in s.values()) for xg, s in seg.items()}     # original lower bounds
        lb0_how = {xg: (s["transit"][1] == "bound", s["transfer"][1] == "bound") for xg, s in seg.items()}   # blocked after screening
        T["bound"] = perf_counter() - t0
        self._ev(t_start, stage="bound")
        self._ev(t_start, stage="straight", blocked=[[x, int(g), [k for k, v in seg[x, g].items() if v[1] == "bound"]] for x, g in seg
                                                     if any(v[1] == "bound" for v in seg[x, g].values())])
        # ---- 4. lazy refinement ----------------------------------------------------------------
        t0 = perf_counter()

        def lo(xg):
            return sum(v[0] for v in seg[xg].values())

        def hi(xg):
            return sum(v[0] for v in seg[xg].values()) if all(v[1] != "bound" for v in seg[xg].values()) else np.inf

        def best(x):
            gs = [xg for xg in seg if xg[0] == x]
            return min(lo(xg) for xg in gs), min(hi(xg) for xg in gs)

        alive = sorted({x for x, _ in seg})
        ref0 = (self.ref.stats["segments"], self.ref.stats["batches"]) if self.ref is not None else None
        while True:
            his = sorted(best(x)[1] for x in alive)
            tau = his[self.k - 1] if len(his) >= self.k else np.inf
            todo = sorted((xg for xg in seg if hi(xg) == np.inf and lo(xg) < tau
                           and np.isfinite(lo(xg))), key=lo)
            if not todo or self.ref is None or not refine:
                break
            batch = todo[:self.ref.batch]
            trn = [xg for xg in batch if seg[xg]["transit"][1] == "bound"]
            if trn:
                vals = self.ref.transit(W, pid, [(start, cat["pick"][pid][x][g]["hover"], sc.pose[pid][x]) for x, g in trn], part_sph)
                for xg, v in zip(trn, vals):
                    seg[xg]["transit"] = [v + desc[xg], "planned" if np.isfinite(v) else "planner_failed"]
            for g in {g for _, g in batch}:
                trf = [xg for xg in batch if xg[1] == g and seg[xg]["transfer"][1] == "bound"]
                if trf:
                    vals = self.ref.transfer(W, pid, grasps[g][0], [(cat["pick"][pid][x][g]["lift"], goal[g]["preins"]) for x, _ in trf])
                    for xg, v in zip(trf, vals):
                        seg[xg]["transfer"] = [v, "planned" if np.isfinite(v) else "planner_failed"]
            self.chk.set_world(W)
            self._ev(t_start, stage="refine", tau=float(tau),
                     items=[[x, int(g), lb0[x, g], hi((x, g))] for x, g in batch])
        if self.ref is not None:
            st["refined_segments"] = self.ref.stats["segments"] - ref0[0]
            st["refine_batches"] = self.ref.stats["batches"] - ref0[1]
        T["refine"] = perf_counter() - t0
        # ---- result ------------------------------------------------------------------------------
        for x in candidates:
            if x not in alive:
                out[x] = Ranked(x, f"pruned:{reason[x] or 'no_grasp'}", np.inf, np.inf)
                continue
            gs = [xg for xg in seg if xg[0] == x]
            l, h = best(x)
            g = min(gs, key=hi if np.isfinite(h) else lo)[1]
            status = "settled" if np.isfinite(h) else "bound" if np.isfinite(l) else "planner_failed"
            out[x] = Ranked(x, status, h if np.isfinite(h) else l, min(lb0[xg] for xg in gs), g,
                            {gg: {k: tuple(v) for k, v in seg[xx, gg].items()} for xx, gg in gs},
                            {gg: feat[xx, gg] for xx, gg in gs})
        ranked = sorted(out.values(), key=lambda r: (r.status.startswith("pruned"), r.status == "planner_failed", r.status == "bound", r.cost))
        st.update({f"{k}_ms": 1e3 * v for k, v in T.items()})
        st["total_ms"] = sum(1e3 * v for v in T.values())
        st["pruned"] = sum(r.status.startswith("pruned") for r in ranked)
        st["blocked_pairs"] = sum(any(v[1] != "straight" for v in seg[xg].values()) for xg in seg)
        st["blocked_transit"] = sum(lb0_how[xg][0] for xg in seg)
        st["blocked_transfer"] = sum(lb0_how[xg][1] for xg in seg)
        st["pairs"] = len(seg)
        st["tau"] = float(tau) if alive else None
        return ranked, st
