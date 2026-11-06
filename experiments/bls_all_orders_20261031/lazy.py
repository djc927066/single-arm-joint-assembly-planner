"""Same exact all-orders search as run.py, but the geometry is checked ON DEMAND (2026-10-31).

run.py precomputes every (part on the table at a pose) x (option of another part) pair: 7M pairs,
78 min on the frame, of which the searches touched 23 %. Here a check is made only when the branch
and bound needs it, and its result is cached for every later node and order:
  pose xi of p is usable under the suffix T (parts on the table)  <=>  footprint clear of T and
  some option (xi, g) is free with the installed set S AND not blocked by any part of T.
  The "not blocked by o" bit is computed per (o, p, xi) (pick-side rows of the k grasps at xi) and
  per (o, p) (goal-side rows of all grasps), only rows whose low AABB meets o's AABB go to the GPU.
  The installed-set checks run per (S, p) the first time an order needs them.
Search (per order, as run.py): reverse DFBnB, children by cost; a child is pruned by the parent's
bound first (no geometry), then by its own bound = sum over earlier parts of the cheapest pose still
usable, found by scanning the poses in cost order and checking them lazily (first usable stops).

    python -X utf8 -m experiments.bls_all_orders_20261031.lazy --model triangular_frame6
"""
from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

import numpy as np

from .run import OUT, build_options, check, costs, footprints, installed_checks, log, low_aabb, stable_orders


class Geo:
    def __init__(self, pb, O):
        import torch
        self.pb, self.O = pb, O
        sc, chk = pb.scene, pb.chk
        ids, poses, opt = O["ids"], O["poses"], O["opt"]
        self.obs = [(j, xi) for j in ids for xi in range(len(poses[j]))]
        self.oi = {o: i for i, o in enumerate(self.obs)}
        self.bank = [chk.obstacle_spheres(pb.held[j], sc.pose[j][poses[j][xi]]) for j, xi in self.obs]
        self.olo = np.stack([(e[:, :3] - e[:, 3:]).min(0).values.cpu().numpy() for e in self.bank])
        self.ohi = np.stack([(e[:, :3] + e[:, 3:]).max(0).values.cpu().numpy() for e in self.bank])
        zcut = float(self.ohi[:, 2].max())
        chk.set_world(chk.world_dict())
        self.rows = {}                      # p -> (pick arrays + env/lo/hi, goal arrays (A rows + B rows) ...)
        for p in ids:
            sets = []
            for arr in O["R"][p]:
                A, B, cfg, H, J, own, _ = arr
                env = check(chk, A, B, cfg, H, J, None, np.arange(len(A)), O["hb"], [])
                lo, hi = low_aabb(chk, A, B, cfg, H, J, O["hb"], zcut)
                sets.append((A, B, cfg, H, J, own, env & np.all(lo <= hi, 1), lo, hi))
            ox = np.array([x for x, _ in opt[p]])
            rows_at = {}
            for r, o in enumerate(sets[0][5]):
                rows_at.setdefault(int(ox[o]), []).append(r)
            self.rows[p] = (sets, {x: np.array(v) for x, v in rows_at.items()})
        # all rows of all parts and sets in one array, so one resolve = one GPU pass
        self.off, parts, base = {}, [], 0
        for p in ids:
            for si, s in enumerate(self.rows[p][0]):
                self.off[p, si] = base
                base += len(s[0])
                parts.append(s)
        cat = lambda i: np.concatenate([s[i] for s in parts])
        self.G = [cat(i) for i in range(9)]         # A, B, cfg, H, J, own, live, lo, hi
        self.opt_at = {p: {} for p in ids}          # p -> xi -> option ids at that pose
        for p in ids:
            for o, (x, _) in enumerate(opt[p]):
                self.opt_at[p].setdefault(x, []).append(o)
        self.opt_at = {p: {x: np.array(v) for x, v in d.items()} for p, d in self.opt_at.items()}
        self.bp, self.bg = {}, {}                   # caches: (o, p, xi) -> blocked mask over opt_at; (o, p) -> over grasps
        self.st = dict(gpu_calls=0, pairs=0, pick_queries=0, goal_queries=0, gpu_s=0.0)
        self.env_world = True             # False after an installed-set check changed the world

    def resolve(self, pick_q, goal_q):
        """Fill the caches for the (o, p, xi) / (o, p) not cached yet, in one GPU pass per part."""
        pick_q = [q for q in dict.fromkeys(pick_q) if q not in self.bp]
        goal_q = [q for q in dict.fromkeys(goal_q) if q not in self.bg]
        if not pick_q and not goal_q:
            return
        t0 = perf_counter()
        chk, O = self.pb.chk, self.O
        if not self.env_world:
            chk.set_world(chk.world_dict())
            self.env_world = True
        A, B, cfg, H, J, own, live, lo, hi = self.G
        pr, po, keys = [], [], []

        def add(r, o, key):
            r = r[live[r]]
            r = r[np.all((lo[r] <= self.ohi[o]) & (hi[r] >= self.olo[o]), 1)]
            pr.append(r)
            po.append(np.full(len(r), o))
            keys.extend([key] * len(r))
        for o, p, xi in pick_q:
            self.bp[o, p, xi] = np.zeros(len(self.opt_at[p][xi]), bool)
            add(self.off[p, 0] + self.rows[p][1][xi], o, (0, o, p, xi))
        for o, p in goal_q:
            self.bg[o, p] = np.zeros(len(O["gl"][p]), bool)
            for si in (1, 2):
                add(self.off[p, si] + np.arange(len(self.rows[p][0][si][0])), o, (1, o, p, None))
        if sum(len(x) for x in pr):
            pr, po = np.concatenate(pr), np.concatenate(po)
            used, loc = np.unique(po, return_inverse=True)     # only these covers go to the GPU
            fr = check(chk, A, B, cfg, H, J, loc, pr, O["hb"], [self.bank[u] for u in used])
            self.st["gpu_calls"] += 1
            self.st["pairs"] += len(pr)
            for i in np.nonzero(~fr)[0]:
                kind, o, p, xi = keys[i]
                if kind == 0:
                    self.bp[o, p, xi][np.searchsorted(self.opt_at[p][xi], own[pr[i]])] = True
                else:
                    self.bg[o, p][own[pr[i]]] = True
        self.st["pick_queries"] += len(pick_q)
        self.st["goal_queries"] += len(goal_q)
        self.st["gpu_s"] += perf_counter() - t0


class LazySearch:
    def __init__(self, pb, O, geo, fp, base, orders):
        self.pb, self.O, self.geo, self.fp, self.base = pb, O, geo, fp, base
        self.opt_g = {p: np.array([g for _, g in O["opt"][p]]) for p in O["ids"]}
        self.orders = orders

    def base_ok(self, S, p):
        if (S, p) not in self.base:
            bp, bg = installed_checks(self.pb, self.O, [(S, p)])
            self.geo.env_world = False
            self.base[S, p] = (bp[S, p], bg[S, p])
        return self.base[S, p]

    def usable(self, p, xis, okS, gokS, suf):
        """usable flags of poses xis of p (footprint already checked) under the suffix obstacles suf."""
        g = self.geo
        g.resolve([(o, p, xi) for o in suf for xi in xis], [(o, p) for o in suf])
        return self.evaluate(p, xis, okS, gokS, suf)

    def evaluate(self, p, xis, okS, gokS, suf):
        """as usable, caches already filled"""
        g = self.geo
        gbad = np.zeros(len(gokS), bool)
        for o in suf:
            gbad |= g.bg[o, p]
        gok = gokS & ~gbad
        out = []
        for xi in xis:
            ids = g.opt_at[p][xi]
            m = okS[ids] & gok[self.opt_g[p][ids]]
            for o in suf:
                if not m.any():
                    break
                m &= ~g.bp[o, p, xi]
            out.append(bool(m.any()))
        return out

    def run(self, order, mode, node_cap=2_000_000, chunk=8, ub=np.inf):
        """exact best layout of this order; with ub, only a layout cheaper than ub is looked for."""
        O, d = self.O, None
        n = len(order)
        S = {p: tuple(sorted(order[:k])) for k, p in enumerate(order)}
        cost, const = costs(self.cd, order, mode)
        base = {p: self.base_ok(S[p], p) for p in order}
        cand = {}                                   # p -> poses with some installed-set option, by cost
        for p in order:
            okS = base[p][0] & base[p][1][self.opt_g[p]]
            has = np.zeros(len(O["poses"][p]), bool)
            has[np.array([x for x, _ in O["opt"][p]])[okS]] = True
            xs = np.nonzero(has)[0]
            cand[p] = xs[np.argsort(cost[p][xs], kind="stable")]
        okS = {p: base[p][0] for p in order}
        gokS = {p: base[p][1] for p in order}
        st = dict(nodes=0, bound_prunes=0, empty_prunes=0, parent_prunes=0, capped=False)
        best = dict(c=ub, lay=None)
        memo = {}

        def cheapest(p, suf, free):
            """cost of the cheapest usable pose of p under suf (inf if none); lazy scan."""
            key = (p, suf)
            if key in memo:
                return memo[key]
            xs = cand[p][free[p][cand[p]]]
            v = np.inf
            for s in range(0, len(xs), chunk):
                c = list(xs[s:s + chunk])
                u = self.usable(p, c, okS[p], gokS[p], suf)
                hit = [x for x, f in zip(c, u) if f]
                if hit:
                    v = float(cost[p][hit[0]])
                    break
            memo[key] = v
            return v

        def cheapest_many(parts, suf, free):
            """cheapest() for several parts, the lazy scans of all of them resolved together."""
            out, todo, pos = {}, [], {}
            for p in parts:
                if (p, suf) in memo:
                    out[p] = memo[p, suf]
                else:
                    todo.append(p)
                    pos[p] = (cand[p][free[p][cand[p]]], 0)
            while todo:
                pq, gq, cur = [], [], {}
                for p in todo:
                    xs, s = pos[p]
                    cur[p] = list(xs[s:s + chunk])
                    pq += [(o, p, xi) for o in suf for xi in cur[p]]
                    gq += [(o, p) for o in suf]
                self.geo.resolve(pq, gq)
                nxt = []
                for p in todo:
                    xs, s = pos[p]
                    u = self.evaluate(p, cur[p], okS[p], gokS[p], suf)
                    hit = [x for x, f in zip(cur[p], u) if f]
                    if hit or s + chunk >= len(xs):
                        memo[p, suf] = out[p] = float(cost[p][hit[0]]) if hit else np.inf
                    else:
                        pos[p] = (xs, s + chunk)
                        nxt.append(p)
                todo = nxt
            return out

        def rec(k, suf, free, mins, g, lay):
            if st["nodes"] >= node_cap:
                st["capped"] = True
                return
            st["nodes"] += 1
            p, earlier = order[k], order[:k]
            rest = sum(mins[q] for q in earlier)
            xs = cand[p][free[p][cand[p]]]
            for s in range(0, len(xs), chunk):
                c = list(xs[s:s + chunk])
                if g + cost[p][c[0]] + rest >= best["c"] - 1e-12:
                    st["parent_prunes"] += 1
                    return
                for xi, ok in zip(c, self.usable(p, c, okS[p], gokS[p], suf)):
                    if not ok:
                        continue
                    c1 = g + cost[p][xi]
                    if c1 + rest >= best["c"] - 1e-12:
                        st["parent_prunes"] += 1
                        return
                    if k == 0:
                        best["c"], best["lay"] = c1, dict(lay, **{p: int(xi)})
                        continue
                    o = self.geo.oi[(p, int(xi))]
                    suf2 = tuple(sorted(suf + (o,)))
                    free2 = dict(free)
                    for q in earlier:
                        f = self.fp.get(((p, int(xi)), q))
                        if f is not None:
                            fr = free[q].copy()
                            fr[f] = False
                            free2[q] = fr
                    mins2 = cheapest_many(earlier, suf2, free2)
                    if any(v == np.inf for v in mins2.values()):
                        st["empty_prunes"] += 1
                        continue
                    if c1 + sum(mins2.values()) >= best["c"] - 1e-12:
                        st["bound_prunes"] += 1
                        continue
                    rec(k - 1, suf2, free2, mins2, c1, dict(lay, **{p: int(xi)}))

        t0 = perf_counter()
        free0 = {p: np.ones(len(O["poses"][p]), bool) for p in order}
        mins0 = cheapest_many(order, (), free0)
        if all(v < np.inf for v in mins0.values()):
            rec(n - 1, (), free0, mins0, const, {})
        st["seconds"] = perf_counter() - t0
        st["root_lb"] = float(sum(mins0.values()) + const)
        return best, st


def setup(model, goal_pos, k):
    """Problem, options, footprints, lazy geometry and the search object (cost data in ls.cd)."""
    from joint_search.problem import Problem
    import torch
    T0 = perf_counter()
    pb = Problem(model, center=tuple(float(v) for v in goal_pos.split(",")))
    t_setup = perf_counter() - T0
    orders = stable_orders(pb)
    t1 = perf_counter()
    O = build_options(pb, k)
    fp = footprints(pb, O["ids"], O["poses"])
    geo = Geo(pb, O)
    t_prep = perf_counter() - t1
    log(f"setup {t_setup:.0f}s; {len(orders)} stable orders; options + footprints + row AABBs {t_prep:.0f}s")
    _, _, tcp = pb.chk.kinematics(torch.zeros((1, 6), dtype=torch.float32, device=pb.chk.device))
    ls = LazySearch(pb, O, geo, fp, {}, orders)
    ls.cd = dict(home=tcp[0].cpu().numpy(), start={p: np.stack([pb.scene.pose[p][x][:3, 3] for x in O["poses"][p]]) for p in O["ids"]},
                 goal={p: pb.scene.goal[p][:3, 3].copy() for p in O["ids"]})
    return pb, O, geo, ls, orders, t_setup, t_prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--orders", type=int, default=0, help="first n orders only (0 = all)")
    a = ap.parse_args()
    T0 = perf_counter()
    pb, O, geo, ls, orders, t_setup, t_prep = setup(a.model, a.goal_pos, a.k)
    orders = orders[:a.orders or None]
    ref = {}
    rf = OUT / f"search_{a.model}_k{a.k}.json"
    if rf.exists():
        for r in json.loads(rf.read_text(encoding="utf-8"))["results"]:
            ref[tuple(r["order"])] = {m: r[m]["cost"] for m in ("bls", "chain")}
    res, t_search = [], perf_counter()
    for i, o in enumerate(orders):
        row = dict(order=list(o))
        for mode in ("bls", "chain"):
            t = perf_counter()
            best, st = ls.run(tuple(o), mode)
            st["wall_s"] = perf_counter() - t
            row[mode] = dict(cost=None if best["lay"] is None else float(best["c"]), **st,
                             ref=ref.get(tuple(o), {}).get(mode))
        res.append(row)
        g = geo.st
        b, c = row["bls"], row["chain"]
        same = all(row[m]["ref"] is None or (row[m]["cost"] is not None and abs(row[m]["cost"] - row[m]["ref"]) < 1e-9)
                   for m in ("bls", "chain"))
        log(f"[{i + 1}/{len(orders)}] bls {b['cost']:.4f} chain {c['cost']:.4f} {'= exhaustive' if same else 'DIFFERS'}  "
            f"nodes {b['nodes']}/{c['nodes']}  ({b['wall_s'] + c['wall_s']:.1f}s; total gpu {g['gpu_s']:.0f}s, "
            f"pairs {g['pairs']}, cached {len(geo.bp)} pose + {len(geo.bg)} goal, {len(ls.base)} (S,p))")
    total = perf_counter() - T0
    g = geo.st
    n_obs = len(geo.obs)
    n_pose_q = sum(len(O["poses"][p]) * sum(1 for j, _ in geo.obs if j != p) for p in O["ids"])
    summary = dict(total_s=total, setup_s=t_setup, prep_s=t_prep, search_s=perf_counter() - t_search, **g,
                   cached_pose_queries=len(geo.bp), all_pose_queries=n_pose_q,
                   cached_goal_queries=len(geo.bg), all_goal_queries=n_obs * (len(O["ids"]) - 1),
                   placed_obstacles=len({k[0] for k in geo.bp}), obstacles=n_obs,
                   all_match=all(r[m]["ref"] is None or (r[m]["cost"] is not None and abs(r[m]["cost"] - r[m]["ref"]) < 1e-9)
                                 for r in res for m in ("bls", "chain")))
    log(json.dumps(summary, indent=1))
    OUT.mkdir(exist_ok=True)
    f = OUT / f"lazy_{a.model}_k{a.k}.json"
    f.write_text(json.dumps(dict(summary=summary, results=res), indent=1), encoding="utf-8")
    log(f"-> {f}")


if __name__ == "__main__":
    sys.exit(main())
