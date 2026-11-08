"""Geometry of lattice_joint: options, installed-set checks, footprints and lazy table-part checks.

Taken unchanged from experiments/bls_all_orders_20261031 (run.py: Rows, check, low_aabb, build_options,
installed_checks, footprints, stable_orders; lazy.py: Geo and the feasibility part of LazySearch,
here ``Feasibility``). The checks use the mainline checker (lazy_cost.checker.GpuChecker, sphere
model) on the IK catalogue of joint_search (screen.py):

  option = (pose x of part p, grasp g); per pose the k grasps with the smallest fixed + transfer bound
  pick side  pick, hover (p at x as obstacle), descent, lift (held), pick -> lift (held)
  goal side  ins, preins (held), preins -> ins, release line, retreat (installed + p)
  installed_checks(S, p)   both sides against table + assembly base + installed set S
  Geo / Feasibility        "not blocked by part o lying on the table at its pose", per (o, p, x) on
                           the pick side and per (o, p) on the goal side, computed on demand and cached
  footprints               pose pairs of two parts closer than pb.gap on the table

Changes to the experiment (2026-11-02), so that the screening agrees with the final validation
(joint_search.validate.full_check, WRS meshes):
  near-contact segments  the experiment (as joint_search.expand) does not check the last SKIP of the
                         descent, the first SKIP of the lift and the first SKIP of the release line
                         AT ALL. They are now checked with the spheres against everything but the
                         table and the part being handled (base, installed parts, parts lying on the
                         table): a finger beside a part placed next to the assembly base hit the base
                         there (frame, beam_CA). Rows R[p][3] (pick side) and R[p][4] (goal side),
                         checked in a world without the table (world_no_table).
  the handled part       fingers vs the part being picked / released are checked with the meshes
                         (mesh_self_checks): descent + grasp per (part, resting orientation, grasp),
                         release + retreat per (part, grasp). The gripper's pose relative to the part
                         does not depend on where the part lies nor on what is installed, so this is
                         done once per scene (cached) and the failing grasps are dropped before the
                         k grasps per pose are chosen (build_options(keep=...)).
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

SKIP = 0.25        # as joint_search.expand: share of a short motion next to a contact not checked
LIFT = 0.10        # BLS DEFAULT_LIFT
NEAR = (3, 4)      # row sets of R[p] checked without the table (near-contact ends)


def log(*a):
    print(*a, flush=True)


def world_no_table(chk, installed=()):
    """collision world of chk.world_dict(installed) without the table (a tiny box far away keeps the
    cuboid slot used)."""
    d = chk.world_dict(installed)
    d["cuboid"] = {"far": dict(dims=[0.01, 0.01, 0.01], pose=[10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0])}
    return d


def mesh_self_checks(pb, cache_dir=None):
    """Gripper vs the part it handles, with the meshes (WRS), as the final validation checks it:
        pick side  descent hover -> pick (open) and grasp at pick (jaw closed), part at its table pose
        goal side  release at ins and retreat ins -> retreat (open), part in its seat
    The relative pose gripper / part depends only on the grasp and the part's resting orientation, so
    the pick side is checked once per (part, orientation, grasp) at one of its poses.
    -> keep(p, x, g) and stats; cached in cache_dir by scene name, centre and catalogue size."""
    import pickle
    sc, cat = pb.scene, pb.cat
    orient = lambda x: ",".join(x.split(",")[2:])
    center = getattr(sc, "center", None)
    key = (sc.name, None if center is None else tuple(np.round(center, 4).tolist()),
           sum(len(gs) for p in pb.ids for gs in cat["pick"][p].values()), sum(len(cat["goal"][p]) for p in pb.ids))
    f = None
    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=True)
        f = cache_dir / f"selfmesh_{sc.name}_{key[2]}_{key[3]}.pkl"
        if f.exists():
            d = pickle.loads(f.read_bytes())
            if d["key"] == key:
                pick_ok, goal_ok, st = d["pick"], d["goal"], dict(d["stats"], cached=True)
                return (lambda p, x, g: goal_ok.get((p, g), False) and pick_ok.get((p, orient(x), g), False)), st
    from joint_search.wrs_check import MeshWorld
    t0 = perf_counter()
    W = MeshWorld(pb)

    def free(P, p, T, jaw=None):
        W.set([], {}, None, jaw)
        W.obs = [W._pose(W._cm, p, T)]
        return W.free(np.stack(P))[0]
    pick_ok, goal_ok = {}, {}
    for p in pb.ids:
        for g, v in cat["goal"][p].items():
            goal_ok[p, g] = free([v["ins"]], p, sc.goal[p]) and free([v["ins"], v["retreat"]], p, sc.goal[p])
        for x, gs in cat["pick"][p].items():
            for g, e in gs.items():
                k = (p, orient(x), g)
                if k in pick_ok or not goal_ok.get((p, g), False):
                    continue
                jaw = pb.jaw[p][g] if pb.close_fingers else None
                pick_ok[k] = free([e["hover"], e["pick"]], p, sc.pose[p][x]) and free([e["pick"]], p, sc.pose[p][x], jaw)
    st = dict(goal_checked=len(goal_ok), goal_rejected=sum(not v for v in goal_ok.values()),
              pick_checked=len(pick_ok), pick_rejected=sum(not v for v in pick_ok.values()),
              seconds=perf_counter() - t0, cached=False)
    if f is not None:
        f.write_bytes(pickle.dumps(dict(key=key, pick=pick_ok, goal=goal_ok, stats=st)))
    return (lambda p, x, g: goal_ok.get((p, g), False) and pick_ok.get((p, orient(x), g), False)), st


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


def mesh_seat_checks(pb, orders, keep, cache_dir=None):
    """Gripper vs the base and the parts that always come before p, with the meshes, as the final
    validation checks it: insert preins -> ins (jaw closed, holding p), release at ins and retreat
    ins -> retreat (open). The finger tips pass within millimetres of the part below, where the sphere
    model is too coarse. These parts are installed in every plan, so a failing grasp is never usable.
    -> {(p, g): free} for the grasps keep allows and stats; cached like mesh_self_checks."""
    import pickle
    sc, cat = pb.scene, pb.cat
    center = getattr(sc, "center", None)
    key = (sc.name, None if center is None else tuple(np.round(center, 4).tolist()),
           sum(len(gs) for p in pb.ids for gs in cat["pick"][p].values()), sum(len(cat["goal"][p]) for p in pb.ids))
    f = None
    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=True)
        f = cache_dir / f"seatmesh_{sc.name}_{key[2]}_{key[3]}.pkl"
        if f.exists():
            d = pickle.loads(f.read_bytes())
            if d["key"] == key:
                return d["ok"], dict(d["stats"], cached=True)
    from joint_search.wrs_check import MeshWorld
    t0 = perf_counter()
    W, G, ok = MeshWorld(pb), sc.grasps(), {}
    before = {p: set(pb.ids) for p in pb.ids}
    for o in orders:
        for i, p in enumerate(o):
            before[p] &= set(o[:i])
    for p in pb.ids:
        inst = sorted(before[p])
        for g in sorted({g for x, gs in cat["pick"][p].items() for g in gs if g in cat["goal"][p] and keep(p, x, g)}):
            v = cat["goal"][p][g]
            W.set(inst, {}, (p, G[p][g][0], v["preins"]), pb.jaw[p][g] if pb.close_fingers else None, ("installed",))
            free = W.free(np.stack([v["preins"], v["ins"]]))[0]
            if free:
                W.set(inst + [p], {}, None, None)
                free = W.free(np.stack([v["ins"]]))[0] and W.free(np.stack([v["ins"], v["retreat"]]))[0]
            ok[p, g] = free
    st = dict(checked=len(ok), rejected=sum(not v for v in ok.values()), seconds=perf_counter() - t0, cached=False)
    if f is not None:
        f.write_bytes(pickle.dumps(dict(key=key, ok=ok, stats=st)))
    return ok, st


def build_options(pb, k, keep=None):
    """Options (pose, k grasps), held bank and collision rows per part. keep(p, x, g): grasps allowed
    (mesh_self_checks), applied before the k cheapest grasps of a pose are chosen."""
    from joint_search.problem import t_acc
    sc, cat = pb.scene, pb.cat
    ids = list(pb.ids)
    jaw = lambda p, g: pb.jaw[p][g] if pb.close_fingers else np.nan
    # ---- options ----------------------------------------------------------------------------
    ok = (lambda p, x, g: g in cat["goal"][p] and (keep is None or keep(p, x, g)))
    poses = {p: [x for x in cat["pick"][p] if any(ok(p, x, g) for g in cat["pick"][p][x])] for p in ids}
    opt, gl = {}, {}                        # opt[p] = [(xi, gi)], gl[p] = goal grasps used
    hb, hix = [], {}
    for p in ids:
        goal, rows = cat["goal"][p], []
        for xi, x in enumerate(poses[p]):
            gs = cat["pick"][p][x]
            gg = sorted((g for g in gs if ok(p, x, g)),
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
        rp, rg, rb, rpn, rgn = Rows(), Rows(), Rows(), Rows(), Rows()
        for o, (xi, gi) in enumerate(opt[p]):
            x, g = poses[p][xi], gl[p][gi]
            e, h, j = cat["pick"][p][x][g], hix[p, g], jaw(p, g)
            rp.add(e["pick"], None, -1, j, o)
            rp.add(e["hover"], None, -1, np.nan, o, ex_self=xi)
            rp.add(e["hover"], e["pick"] + SKIP * (e["hover"] - e["pick"]), -1, np.nan, o)
            rp.add(e["lift"], None, h, j, o)
            rp.add(e["pick"] + SKIP * (e["lift"] - e["pick"]), e["lift"], h, j, o)
            # near-contact ends, checked without the table (world_no_table) and without p itself
            rpn.add(e["pick"] + SKIP * (e["hover"] - e["pick"]), e["pick"], -1, np.nan, o)
            rpn.add(e["pick"], e["pick"] + SKIP * (e["lift"] - e["pick"]), h, j, o)
        for gi, g in enumerate(gl[p]):
            v, h, j = cat["goal"][p][g], hix[p, g], jaw(p, g)
            rg.add(v["ins"], None, -1, j, gi)
            rg.add(v["preins"], None, h, j, gi)
            rg.add(v["preins"], v["ins"], -1, j, gi)
            rg.add(v["ins"] + SKIP * (v["retreat"] - v["ins"]), v["retreat"], -1, np.nan, gi)
            rb.add(v["retreat"], None, -1, np.nan, gi)          # world after release: installed + p
            rgn.add(v["ins"], v["ins"] + SKIP * (v["retreat"] - v["ins"]), -1, np.nan, gi)   # without p
        R[p] = (rp.arrays(), rg.arrays(), rb.arrays(), rpn.arrays(), rgn.arrays())
    return dict(ids=ids, poses=poses, opt=opt, gl=gl, hb=hb, hix=hix, R=R)


def installed_checks(pb, O, steps):
    """base_pick[(S, p)] / base_goal[(S, p)]: options / goal grasps free with the installed set S."""
    sc, chk = pb.scene, pb.chk
    ids, poses, opt, gl, hb, R = O["ids"], O["poses"], O["opt"], O["gl"], O["hb"], O["R"]
    base_pick, base_goal = {}, {}
    own_bank = {p: [chk.obstacle_spheres(pb.held[p], sc.pose[p][x]) for x in poses[p]] for p in ids}
    for S, p in steps:
        (A, B, cfg, H, J, own, exs), (gA, gB, gcfg, gH, gJ, gown, _), (bA, bB, bcfg, bH, bJ, bown, _) = R[p][:3]
        (nA, nB, ncfg, nH, nJ, nown, _), (mA, mB, mcfg, mH, mJ, mown, _) = R[p][3:]
        if (S, p) not in base_pick:
            chk.set_world(chk.world_dict(S))
            fr = check(chk, A, B, cfg, H, J, exs, np.arange(len(A)), hb, own_bank[p])
            ok = np.ones(len(opt[p]), bool)
            np.logical_and.at(ok, own, fr)
            fg = check(chk, gA, gB, gcfg, gH, gJ, None, np.arange(len(gA)), hb, [])
            okg = np.ones(len(gl[p]), bool)
            np.logical_and.at(okg, gown, fg)
            chk.set_world(chk.world_dict(tuple(S) + (p,)))
            fb = check(chk, bA, bB, bcfg, bH, bJ, None, np.arange(len(bA)), hb, [])
            np.logical_and.at(okg, bown, fb)
            chk.set_world(world_no_table(chk, S))              # near-contact ends: no table, no p
            np.logical_and.at(ok, nown, check(chk, nA, nB, ncfg, nH, nJ, None, np.arange(len(nA)), hb, []))
            np.logical_and.at(okg, mown, check(chk, mA, mB, mcfg, mH, mJ, None, np.arange(len(mA)), hb, []))
            base_pick[S, p] = ok
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
            for si, arr in enumerate(O["R"][p]):
                A, B, cfg, H, J, own, _ = arr
                chk.set_world(world_no_table(chk) if si in NEAR else chk.world_dict())
                env = check(chk, A, B, cfg, H, J, None, np.arange(len(A)), O["hb"], [])
                lo, hi = low_aabb(chk, A, B, cfg, H, J, O["hb"], zcut)
                sets.append((A, B, cfg, H, J, own, env & np.all(lo <= hi, 1), lo, hi))
            ox = np.array([x for x, _ in opt[p]])
            rows_at = {}
            for si in (0, 3):                           # pick-side sets: rows per pose
                d = {}
                for r, o in enumerate(sets[si][5]):
                    d.setdefault(int(ox[o]), []).append(r)
                rows_at[si] = {x: np.array(v) for x, v in d.items()}
            self.rows[p] = (sets, rows_at)
        # all rows of all parts and sets in one array, so one resolve = one GPU pass
        self.off, parts, base = {}, [], 0
        for p in ids:
            for si, s in enumerate(self.rows[p][0]):
                self.off[p, si] = base
                base += len(s[0])
                parts.append(s)
        cat = lambda i: np.concatenate([s[i] for s in parts])
        self.G = [cat(i) for i in range(9)]         # A, B, cfg, H, J, own, live, lo, hi
        self.near = np.concatenate([np.full(len(s[0]), si in NEAR) for p in ids for si, s in enumerate(self.rows[p][0])])
        chk.set_world(chk.world_dict())
        self.opt_at = {p: {} for p in ids}          # p -> xi -> option ids at that pose
        for p in ids:
            for o, (x, _) in enumerate(opt[p]):
                self.opt_at[p].setdefault(x, []).append(o)
        self.opt_at = {p: {x: np.array(v) for x, v in d.items()} for p, d in self.opt_at.items()}
        self.bp, self.bg = {}, {}                   # caches: (o, p, xi) -> blocked mask over opt_at; (o, p) -> over grasps
        self.st = dict(gpu_calls=0, pairs=0, pick_queries=0, goal_queries=0, gpu_s=0.0)

    def resolve(self, pick_q, goal_q):
        """Fill the caches for the (o, p, xi) / (o, p) not cached yet: one GPU pass for the ordinary rows
        (table + base world), one for the near-contact rows (world without the table)."""
        pick_q = [q for q in dict.fromkeys(pick_q) if q not in self.bp]
        goal_q = [q for q in dict.fromkeys(goal_q) if q not in self.bg]
        if not pick_q and not goal_q:
            return
        t0 = perf_counter()
        chk, O = self.pb.chk, self.O
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
            for si in (0, 3):
                r = self.rows[p][1][si].get(xi)
                if r is not None:
                    add(self.off[p, si] + r, o, (0, o, p, xi))
        for o, p in goal_q:
            self.bg[o, p] = np.zeros(len(O["gl"][p]), bool)
            for si in (1, 2, 4):
                add(self.off[p, si] + np.arange(len(self.rows[p][0][si][0])), o, (1, o, p, None))
        if sum(len(x) for x in pr):
            pr, po = np.concatenate(pr), np.concatenate(po)
            for near in (False, True):
                m = self.near[pr] == near
                if not m.any():
                    continue
                chk.set_world(world_no_table(chk) if near else chk.world_dict())
                rr, oo = pr[m], po[m]
                kk = [k for k, b in zip(keys, m) if b]
                used, loc = np.unique(oo, return_inverse=True)     # only these covers go to the GPU
                fr = check(chk, A, B, cfg, H, J, loc, rr, O["hb"], [self.bank[u] for u in used])
                self.st["gpu_calls"] += 1
                self.st["pairs"] += len(rr)
                for i in np.nonzero(~fr)[0]:
                    kind, o, p, xi = kk[i]
                    if kind == 0:
                        self.bp[o, p, xi][np.searchsorted(self.opt_at[p][xi], own[rr[i]])] = True
                    else:
                        self.bg[o, p][own[rr[i]]] = True
        self.st["pick_queries"] += len(pick_q)
        self.st["goal_queries"] += len(goal_q)
        self.st["gpu_s"] += perf_counter() - t0


class Feasibility:
    """installed-set checks (cached per (S, p)) and table-part checks (lazy, Geo caches)."""

    def __init__(self, pb, O, geo, fp):
        self.pb, self.O, self.geo, self.fp, self.base = pb, O, geo, fp, {}
        self.opt_g = {p: np.array([g for _, g in O["opt"][p]]) for p in O["ids"]}

    def base_ok(self, S, p):
        if (S, p) not in self.base:
            bp, bg = installed_checks(self.pb, self.O, [(S, p)])
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
