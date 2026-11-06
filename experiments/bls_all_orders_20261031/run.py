"""Exact layout search for EVERY stable order, BLS nominal cost, with process data (2026-10-31).

Purpose: observation, not a method. For each stable assembly order of a benchmark, find the exactly
optimal initial layout over the discrete candidate poses, and log what the search sees, to look for
structure that a joint order x layout search could exploit.

Cost (BLS paper Eq. 12, sealp/examples/layout/bsfs/cost.py: object origin as the end-effector proxy,
no grasp, lift 0.10 m):
    bls    each step HOME -> above pick -> pick -> above goal -> goal -> HOME. Independent of the order.
    chain  no return to HOME between steps: above goal of the previous part -> above pick of this one
           (HOME for the first part; the last part returns to HOME). Depends on the order.
Feasibility (BLS local checks with the mainline checker, lazy_cost.checker.GpuChecker sphere model):
    a pose x of part p is usable at its step if some grasp g (the k with the smallest fixed + transfer
    bound) passes
      pick side  pick, hover (p at x as obstacle), descent, lift (held), pick -> lift (held)
      goal side  ins, preins (held), preins -> ins, release line, retreat (installed + p)
    against the table, the installed set S (parts before p in the order) and every part still on the
    table (parts after p, at their chosen poses); and x keeps pb.gap to the footprints of those parts.
    Transit / transfer lines are not checked (BLS checks them only in the final full validation).
    Collisions with table parts are pairwise (obstacle set = union), so they are precomputed once per
    (obstacle part, pose); collisions with the installed set are precomputed per (S, p).
Search: per order, reverse assignment (last part first), depth-first branch and bound, children by
cost, bound = assigned cost + sum over earlier parts of the cheapest pose left in their domain (BLS
Eq. 18), domains filtered after every assignment (empty domain -> prune, BLS rule).

    python -X utf8 -m experiments.bls_all_orders_20261031.run precompute --model triangular_frame6
    python -X utf8 -m experiments.bls_all_orders_20261031.run search --model triangular_frame6
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "_out"
SKIP = 0.25        # as joint_search.expand: share of a short motion next to a contact not checked
LIFT = 0.10        # BLS DEFAULT_LIFT


def log(*a):
    print(*a, flush=True)


# ================================================================================================
# precompute (GPU)
# ================================================================================================
class Rows:
    """Collision queries: A, B (None = configuration), held index, jaw, owner (option id), group."""

    def __init__(self):
        self.A, self.B, self.H, self.J, self.own, self.ex_self = [], [], [], [], [], []

    def add(self, A, B, h, j, own, ex_self=-1):
        self.A.append(np.asarray(A, float))
        self.B.append(None if B is None else np.asarray(B, float))
        self.H.append(h), self.J.append(j), self.own.append(own), self.ex_self.append(ex_self)

    def arrays(self):
        n = len(self.A)
        cfg = np.array([b is None for b in self.B])
        A = np.stack(self.A) if n else np.zeros((0, 6))
        B = np.stack([a if b is None else b for a, b in zip(self.A, self.B)]) if n else np.zeros((0, 6))
        return A, B, cfg, np.array(self.H, int), np.array(self.J, float), np.array(self.own, int), np.array(self.ex_self, int)


def check(chk, A, B, cfg, H, J, E, idx, held_bank, extra_bank, step=0.02, chunk=200_000):
    """free flags of rows ``idx`` (extra obstacle E[i], -1 = none)."""
    out = np.zeros(len(idx), bool)
    for s in range(0, len(idx), chunk):
        r = idx[s:s + chunk]
        res = np.zeros(len(r), bool)
        c = cfg[r]
        for m, line in ((c, False), (~c, True)):
            if m.any():
                rr = r[m]
                h = H[rr]                                   # only the held parts used here go to the GPU
                used = np.unique(h[h >= 0])
                hl = np.where(h >= 0, np.searchsorted(used, h), -1)
                (fr,) = chk.batch_free([(A[rr], B[rr] if line else None, hl, E[s:s + chunk][m] if E is not None else -1, J[rr])],
                                       held_bank=[held_bank[u] for u in used], extra_bank=extra_bank, step=step)
                res[m] = fr
        out[s:s + len(r)] = res
    return out


def low_aabb(chk, A, B, cfg, H, J, held_bank, zcut, step=0.02, chunk=40_000):
    """Per row AABB of the arm + held spheres that reach below zcut (only those can touch a part on
    the table); rows without such a sphere get an empty box (lo > hi)."""
    import torch
    dev = chk.device
    S, owner = [], []
    for i in range(len(A)):
        if cfg[i]:
            S.append(A[i][None]), owner.append(np.array([i]))
        else:
            k = max(1, int(np.ceil(np.abs(B[i] - A[i]).max() / step))) + 1
            t = np.linspace(0, 1, k)[:, None]
            S.append(A[i] + t * (B[i] - A[i])), owner.append(np.full(k, i))
    S, owner = np.concatenate(S), np.concatenate(owner)
    far = 1e3
    loc = []
    for sph, G in held_bank:
        Gi = np.linalg.inv(G)
        loc.append(np.c_[sph[:, :3] @ Gi[:3, :3].T + Gi[:3, 3], sph[:, 3] + chk.buffer])
    m = max(len(x) for x in loc)
    HB = torch.as_tensor(np.stack([np.r_[x, np.full((m - len(x), 4), np.nan)] for x in loc] + [np.full((m, 4), np.nan)]),
                         dtype=torch.float32, device=dev)
    chunk = max(1000, min(chunk, 4_000_000 // m))      # chunk x m held spheres in memory (shelves: many)
    Hs, Js = H[owner], J[owner]
    lo = torch.full((len(A), 3), far, device=dev)
    hi = torch.full((len(A), 3), -far, device=dev)
    qt = torch.as_tensor(S, dtype=torch.float32, device=dev)
    Ht = torch.as_tensor(np.where(Hs < 0, len(held_bank), Hs), device=dev)
    Jt = torch.as_tensor(Js, dtype=torch.float32, device=dev)
    Ot = torch.as_tensor(owner, device=dev)
    for s in range(0, len(S), chunk):
        sph, R, p = chk.kinematics(qt[s:s + chunk])
        b = sph.shape[0]
        sph = sph.clone()
        chk._close_fingers(sph, R, Jt[s:s + b])
        hb = HB[Ht[s:s + b]]
        hw = torch.cat([torch.einsum("bij,bmj->bmi", R, hb[..., :3]) + p[:, None], hb[..., 3:]], -1)
        al = torch.cat([sph, hw], 1)
        c, r = al[..., :3], al[..., 3:4]
        low = (~torch.isnan(r)) & ((c[..., 2:3] - r) <= zcut)
        a = torch.where(low, c - r, torch.full_like(c, far)).amin(1)
        z = torch.where(low, c + r, torch.full_like(c, -far)).amax(1)
        o = Ot[s:s + b]
        lo.index_reduce_(0, o, a, "amin")
        hi.index_reduce_(0, o, z, "amax")
    return lo.cpu().numpy(), hi.cpu().numpy()


def build_options(pb, k):
    """Options (pose, k grasps), held bank and collision rows per part."""
    from joint_search.problem import t_acc
    sc, cat = pb.scene, pb.cat
    ids = list(pb.ids)
    jaw = lambda p, g: pb.jaw[p][g] if pb.close_fingers else np.nan
    # ---- options ----------------------------------------------------------------------------
    poses = {p: [x for x in cat["pick"][p] if any(g in cat["goal"][p] for g in cat["pick"][p][x])] for p in ids}
    opt, gl = {}, {}                        # opt[p] = [(xi, gi)], gl[p] = goal grasps used
    hb, hix = [], {}
    for p in ids:
        goal, rows = cat["goal"][p], []
        for xi, x in enumerate(poses[p]):
            gs = cat["pick"][p][x]
            gg = sorted((g for g in gs if g in goal),
                        key=lambda g: pb.fix_pick[p, x, g] + float(t_acc(gs[g]["lift"], goal[g]["preins"])))[:k]
            rows += [(xi, g) for g in gg]
        gl[p] = sorted({g for _, g in rows})
        gi = {g: i for i, g in enumerate(gl[p])}
        opt[p] = [(xi, gi[g]) for xi, g in rows]
        for g in gl[p]:
            hix[p, g] = len(hb)
            hb.append((pb.held[p], sc.grasps()[p][g][0]))
    log("options: " + ", ".join(f"{p} {len(poses[p])} poses / {len(opt[p])} (pose, grasp) / {len(gl[p])} grasps" for p in ids))
    # pick-side rows (owner = option id) and goal-side rows (owner = goal grasp id), per part
    R = {}
    for p in ids:
        rp, rg, rb = Rows(), Rows(), Rows()
        for o, (xi, gi) in enumerate(opt[p]):
            x, g = poses[p][xi], gl[p][gi]
            e, h, j = cat["pick"][p][x][g], hix[p, g], jaw(p, g)
            rp.add(e["pick"], None, -1, j, o)
            rp.add(e["hover"], None, -1, np.nan, o, ex_self=xi)
            rp.add(e["hover"], e["pick"] + SKIP * (e["hover"] - e["pick"]), -1, np.nan, o)
            rp.add(e["lift"], None, h, j, o)
            rp.add(e["pick"] + SKIP * (e["lift"] - e["pick"]), e["lift"], h, j, o)
        for gi, g in enumerate(gl[p]):
            v, h, j = cat["goal"][p][g], hix[p, g], jaw(p, g)
            rg.add(v["ins"], None, -1, j, gi)
            rg.add(v["preins"], None, h, j, gi)
            rg.add(v["preins"], v["ins"], -1, j, gi)
            rg.add(v["ins"] + SKIP * (v["retreat"] - v["ins"]), v["retreat"], -1, np.nan, gi)
            rb.add(v["retreat"], None, -1, np.nan, gi)          # world after release: installed + p
        R[p] = (rp.arrays(), rg.arrays(), rb.arrays())
    return dict(ids=ids, poses=poses, opt=opt, gl=gl, hb=hb, hix=hix, R=R)


def installed_checks(pb, O, steps):
    """base_pick[(S, p)] / base_goal[(S, p)]: options / goal grasps free with the installed set S."""
    sc, chk = pb.scene, pb.chk
    ids, poses, opt, gl, hb, R = O["ids"], O["poses"], O["opt"], O["gl"], O["hb"], O["R"]
    base_pick, base_goal = {}, {}
    own_bank = {p: [chk.obstacle_spheres(pb.held[p], sc.pose[p][x]) for x in poses[p]] for p in ids}
    for S, p in steps:
        (A, B, cfg, H, J, own, exs), (gA, gB, gcfg, gH, gJ, gown, _), (bA, bB, bcfg, bH, bJ, bown, _) = R[p]
        if (S, p) not in base_pick:
            chk.set_world(chk.world_dict(S))
            fr = check(chk, A, B, cfg, H, J, exs, np.arange(len(A)), hb, own_bank[p])
            ok = np.ones(len(opt[p]), bool)
            np.logical_and.at(ok, own, fr)
            base_pick[S, p] = ok
            fg = check(chk, gA, gB, gcfg, gH, gJ, None, np.arange(len(gA)), hb, [])
            okg = np.ones(len(gl[p]), bool)
            np.logical_and.at(okg, gown, fg)
            chk.set_world(chk.world_dict(tuple(S) + (p,)))
            fb = check(chk, bA, bB, bcfg, bH, bJ, None, np.arange(len(bA)), hb, [])
            np.logical_and.at(okg, bown, fb)
            base_goal[S, p] = okg
    return base_pick, base_goal


def footprints(pb, ids, poses):
    """((p, xi), q) -> poses of q closer than pb.gap to p at pose xi."""
    sc = pb.scene
    fp = {}
    polys = {p: [sc.polygons[p][x] for x in poses[p]] for p in ids}
    bnd = {p: np.array([P.bounds for P in polys[p]]) for p in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            p, q = ids[a], ids[b]
            for xi in range(len(poses[p])):
                bp = bnd[p][xi]
                near = np.nonzero((bnd[q][:, 0] < bp[2] + pb.gap) & (bnd[q][:, 2] > bp[0] - pb.gap)
                                  & (bnd[q][:, 1] < bp[3] + pb.gap) & (bnd[q][:, 3] > bp[1] - pb.gap))[0]
                for yi in near:
                    if polys[p][xi].distance(polys[q][yi]) < pb.gap:
                        fp.setdefault(((p, xi), q), []).append(yi)
                        fp.setdefault(((q, yi), p), []).append(xi)
    fp = {k: np.array(v) for k, v in fp.items()}
    return fp


def precompute(args):
    from joint_search.problem import HOME, Problem, t_acc
    import torch
    t0 = perf_counter()
    pb = Problem(args.model, center=tuple(float(v) for v in args.goal_pos.split(",")))
    sc, cat, chk = pb.scene, pb.cat, pb.chk
    log(f"setup {perf_counter() - t0:.0f}s")
    ids = list(pb.ids)
    orders = stable_orders(pb)
    log(f"{len(orders)} stable orders")
    O = build_options(pb, args.k)
    poses, opt, gl, hb, hix, R = O["poses"], O["opt"], O["gl"], O["hb"], O["hix"], O["R"]
    t1 = perf_counter()
    steps = sorted({(tuple(sorted(o[:k])), o[k]) for o in orders for k in range(len(o))})
    base_pick, base_goal = installed_checks(pb, O, steps)
    log(f"installed-set checks: {len(steps)} (S, p) in {perf_counter() - t1:.0f}s")
    # ---- pairwise: rows vs one part on the table --------------------------------------------
    t1 = perf_counter()
    obs = [(j, xi) for j in ids for xi in range(len(poses[j]))]
    bank = [chk.obstacle_spheres(pb.held[j], sc.pose[j][poses[j][xi]]) for j, xi in obs]
    olo = np.stack([(e[:, :3] - e[:, 3:]).min(0).values.cpu().numpy() for e in bank])
    ohi = np.stack([(e[:, :3] + e[:, 3:]).max(0).values.cpu().numpy() for e in bank])
    zcut = float(ohi[:, 2].max())
    chk.set_world(chk.world_dict())
    block_pick, block_goal = {}, {}       # (obstacle index, p) -> blocked option ids / goal grasp ids
    n_pairs = n_hit = 0
    for p in ids:
        for kind, (A, B, cfg, H, J, own, _) in (("pick", R[p][0]), ("goal", R[p][1]), ("goal", R[p][2])):
            if not len(A):
                continue
            env = check(chk, A, B, cfg, H, J, None, np.arange(len(A)), hb, [])
            lo, hi = low_aabb(chk, A, B, cfg, H, J, hb, zcut)
            live = np.nonzero(env & np.all(lo <= hi, 1))[0]
            other = np.array([j != p for j, _ in obs])
            pr, po = [], []
            for c in range(0, len(live), 2048):
                r = live[c:c + 2048]
                ov = np.all((lo[r, None] <= ohi[None]) & (hi[r, None] >= olo[None]), -1) & other[None]
                a, b = np.nonzero(ov)
                pr.append(r[a]), po.append(b)
            pr = np.concatenate(pr) if pr else np.zeros(0, int)
            po = np.concatenate(po) if po else np.zeros(0, int)
            fr = check(chk, A, B, cfg, H, J, po, pr, hb, bank) if len(pr) else np.zeros(0, bool)
            n_pairs += len(pr)
            n_hit += int((~fr).sum())
            tgt = block_pick if kind == "pick" else block_goal
            for r_, o_ in zip(pr[~fr], po[~fr]):
                tgt.setdefault((int(o_), p), set()).add(int(own[r_]))
            log(f"  {p} {kind}: rows {len(A)}, low rows {len(live)}, pairs {len(pr)}, blocked {(~fr).sum()} "
                f"({perf_counter() - t1:.0f}s)")
    block_pick = {k: np.array(sorted(v)) for k, v in block_pick.items()}
    block_goal = {k: np.array(sorted(v)) for k, v in block_goal.items()}
    log(f"pairwise: {n_pairs} pairs, {n_hit} blocked, {perf_counter() - t1:.0f}s")
    t1 = perf_counter()
    fp = footprints(pb, ids, poses)
    log(f"footprints: {sum(len(v) for v in fp.values()) // 2} conflicting pose pairs, {perf_counter() - t1:.0f}s")
    # ---- geometry for the cost ----------------------------------------------------------------
    _, _, tcp = chk.kinematics(torch.zeros((1, 6), dtype=torch.float32, device=chk.device))
    home = tcp[0].cpu().numpy()
    data = dict(model=args.model, ids=ids, orders=[list(o) for o in orders], poses=poses, grasps=gl,
                opt=opt, base_pick=base_pick, base_goal=base_goal, obs=obs, block_pick=block_pick,
                block_goal=block_goal, fp=fp, home=home,
                start={p: np.stack([sc.pose[p][x][:3, 3] for x in poses[p]]) for p in ids},
                goal={p: sc.goal[p][:3, 3].copy() for p in ids}, gap=pb.gap, k=args.k,
                seconds=perf_counter() - t0)
    OUT.mkdir(exist_ok=True)
    f = OUT / f"pre_{args.model}_k{args.k}.pkl"
    pickle.dump(data, open(f, "wb"))
    log(f"-> {f}  ({perf_counter() - t0:.0f}s)")


def stable_orders(pb):
    sc = pb.scene
    if hasattr(sc, "orders"):
        return [tuple(o) for o in sc.orders]
    out = []

    def rec(R, suf):                      # backward: remove a part that can be installed last
        if not R:
            out.append(tuple(suf))
            return
        for p in sorted(R):
            if sc.step_ok(R - {p}, p):
                rec(R - {p}, [p] + suf)
    rec(frozenset(pb.ids), [])
    return out


# ================================================================================================
# search (CPU)
# ================================================================================================
def costs(d, order, mode):
    """cost[p] [n_poses]: BLS nominal cost of each pose of p in this order; + constant (chain end)."""
    up = np.array([0.0, 0.0, LIFT])
    out, prev = {}, d["home"]
    for k, p in enumerate(order):
        sp, gp = d["start"][p], d["goal"][p]
        a = np.linalg.norm((sp + up) - (d["home"] if mode == "bls" or k == 0 else prev + up), axis=1)
        c = a + LIFT + np.linalg.norm((gp + up) - (sp + up), axis=1) + LIFT
        if mode == "bls":
            c = c + np.linalg.norm((gp + up) - d["home"])
        out[p] = c
        prev = gp
    const = 0.0 if mode == "bls" else float(np.linalg.norm(d["goal"][order[-1]] + up - d["home"]))
    return out, const


def search_order(d, order, mode, node_cap, touched=None):
    ids = order
    n = len(order)
    oi = {o: i for i, o in enumerate(d["obs"])}
    S = {p: tuple(sorted(order[:k])) for k, p in enumerate(order)}
    opt_x = {p: np.array([x for x, _ in d["opt"][p]]) for p in ids}
    opt_g = {p: np.array([g for _, g in d["opt"][p]]) for p in ids}
    npose = {p: len(d["poses"][p]) for p in ids}
    cost, const = costs(d, order, mode)
    ok0 = {p: d["base_pick"][S[p], p] & d["base_goal"][S[p], p][opt_g[p]] for p in ids}
    st = dict(nodes=0, bound_prunes=0, empty_prunes=0, capped=False, level_dom=[[] for _ in range(n)])
    best = dict(c=np.inf, lay=None)

    def domain(p, ok, free):
        dm = np.zeros(npose[p], bool)
        dm[opt_x[p][ok]] = True
        return dm & free

    def place(q, xi, ok, free, earlier):
        """Apply q at pose xi to the masks of the earlier parts -> new masks or None (empty)."""
        o = oi[(q, xi)]
        if touched is not None:
            touched.add((q, xi))
        ok2, free2, doms = dict(ok), dict(free), {}
        for p in earlier:
            b = d["block_pick"].get((o, p))
            g = d["block_goal"].get((o, p))
            if b is not None or g is not None:
                m = ok[p].copy()
                if b is not None:
                    m[b] = False
                if g is not None:
                    m &= ~np.isin(opt_g[p], g)
                ok2[p] = m
            f = d["fp"].get(((q, xi), p))
            if f is not None:
                fr = free[p].copy()
                fr[f] = False
                free2[p] = fr
            doms[p] = domain(p, ok2[p], free2[p])
            if not doms[p].any():
                return None
        return ok2, free2, doms

    def rec(k, ok, free, doms, g, lay):
        if st["nodes"] >= node_cap:
            st["capped"] = True
            return
        p = order[k]
        earlier = order[:k]
        st["nodes"] += 1
        st["level_dom"][k].append(int(doms[p].sum()))
        xs = np.nonzero(doms[p])[0]
        xs = xs[np.argsort(cost[p][xs], kind="stable")]
        for xi in xs:
            c = g + cost[p][xi]
            lb = c + sum(cost[q][doms[q]].min() for q in earlier)
            if lb >= best["c"] - 1e-12:
                st["bound_prunes"] += 1
                break                                   # children sorted by cost: the rest are worse
            if k == 0:
                best["c"], best["lay"] = c, dict(lay, **{p: int(xi)})
                continue
            r = place(p, int(xi), ok, free, earlier)
            if r is None:
                st["empty_prunes"] += 1
                continue
            rec(k - 1, r[0], r[1], r[2], c, dict(lay, **{p: int(xi)}))

    free0 = {p: np.ones(npose[p], bool) for p in ids}
    doms0 = {p: domain(p, ok0[p], free0[p]) for p in ids}
    root_lb = sum(cost[p][doms0[p]].min() if doms0[p].any() else np.inf for p in ids) + const
    t0 = perf_counter()
    if all(doms0[p].any() for p in ids):
        rec(n - 1, ok0, free0, doms0, const, {})
    st["seconds"] = perf_counter() - t0
    st["level_dom"] = [float(np.mean(v)) if v else None for v in st["level_dom"]]
    return best, st, dict(root_lb=float(root_lb), root_dom={p: int(doms0[p].sum()) for p in ids}), cost, const


def why_not_best(d, order, lay, cost):
    """For each part: is its pose the cheapest of its ROOT domain? If not, what blocks the cheaper
    ones in the final layout (later parts on the table)."""
    oi = {o: i for i, o in enumerate(d["obs"])}
    S = {p: tuple(sorted(order[:k])) for k, p in enumerate(order)}
    out = {}
    for k, p in enumerate(order):
        og = np.array([g for _, g in d["opt"][p]])
        ox = np.array([x for x, _ in d["opt"][p]])
        ok0 = d["base_pick"][S[p], p] & d["base_goal"][S[p], p][og]
        dom0 = np.zeros(len(d["poses"][p]), bool)
        dom0[ox[ok0]] = True
        cheaper = [xi for xi in np.nonzero(dom0)[0] if cost[p][xi] < cost[p][lay[p]] - 1e-9]
        reasons = {}
        for xi in cheaper:
            by = []
            for q in order[k + 1:]:
                o = oi[(q, lay[q])]
                if xi in set(d["fp"].get(((q, lay[q]), p), [])):
                    by.append(f"{q}:footprint")
                    continue
                b = d["block_pick"].get((o, p))
                g = d["block_goal"].get((o, p))
                m = ok0.copy()
                if b is not None:
                    m[b] = False
                if g is not None:
                    m &= ~np.isin(og, g)
                if not m[ox == xi].any():
                    by.append(f"{q}:motion")
            reasons[int(xi)] = by
        out[p] = dict(pose=int(lay[p]), cost=float(cost[p][lay[p]]), root_min=float(cost[p][dom0].min()),
                      n_cheaper=len(cheaper), blocked_by=sorted({b for v in reasons.values() for b in v}))
    return out


def search(args):
    d = pickle.load(open(OUT / f"pre_{args.model}_k{args.k}.pkl", "rb"))
    res = []
    t0 = perf_counter()
    for i, o in enumerate(d["orders"]):
        row = dict(order=o)
        for mode in ("bls", "chain"):
            best, st, root, cost, const = search_order(d, o, mode, args.node_cap)
            r = dict(cost=None if best["lay"] is None else float(best["c"]), root_lb=root["root_lb"], **st)
            if best["lay"] is not None:
                r["layout"] = {p: d["poses"][p][x] for p, x in best["lay"].items()}
                r["parts"] = why_not_best(d, o, best["lay"], cost)
            r["root_dom"] = root["root_dom"]
            row[mode] = r
        res.append(row)
        b, c = row["bls"], row["chain"]
        log(f"[{i + 1}/{len(d['orders'])}] {' '.join(o)}  bls {b['cost'] if b['cost'] is None else round(b['cost'], 4)} "
            f"(lb {b['root_lb']:.4f}, {b['nodes']} nodes, {b['seconds']:.2f}s)  chain "
            f"{c['cost'] if c['cost'] is None else round(c['cost'], 4)} (lb {c['root_lb']:.4f}, {c['nodes']} nodes)")
    f = OUT / f"search_{args.model}_k{args.k}.json"
    f.write_text(json.dumps(dict(model=args.model, k=args.k, seconds=perf_counter() - t0, results=res), indent=1),
                 encoding="utf-8")
    log(f"-> {f}  ({perf_counter() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("precompute", "search"))
    ap.add_argument("--model", default="triangular_frame6")
    ap.add_argument("--goal-pos", default="0.373,0.0,0.0")
    ap.add_argument("--k", type=int, default=8, help="grasps per pose tried for feasibility")
    ap.add_argument("--node-cap", type=int, default=2_000_000)
    a = ap.parse_args()
    precompute(a) if a.cmd == "precompute" else search(a)


if __name__ == "__main__":
    sys.exit(main())
