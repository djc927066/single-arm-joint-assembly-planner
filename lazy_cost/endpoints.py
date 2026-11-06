"""IK endpoint catalogue: layout-independent preprocessing, cached per model on disk.

A (candidate pose x, grasp g) pair is kept when the SAME grasp works at both ends:
  goal side (per p, g; independent of x), checked first:
     1. IK of the insertion pose  ins = goal_p @ G
     2. an ins solution collision-free (arm + gripper; the held part mates, so it is not checked)
     3. preins (holding p, above ins along world z, 5 or 3 cm) and retreat (empty hand; back along
        the tool axis or straight up, 5 or 3 cm) reachable on the SAME IK branch as ins
        (every joint within BRANCH_TOL) and collision-free  ->  goal-feasible grasps
  pick side (per p, x, g), only for goal-feasible grasps:
     4. IK of the pick pose  pick = T_x @ G
     5. a pick solution collision-free (arm + gripper; the fingers enclose the part)
     6. prepick (back along the tool axis, 5 or 3 cm), hover (8 cm above prepick, empty) and lift
        (holding, up 6 or 4 cm) on the same branch as pick, collision-free
Every pair that passes is kept: no cap on the number of grasps (max_grasps=None). All IK is
batched on the GPU (cuRobo) and every stage's collision check is one batched pass. IK runs only
where the earlier stages passed (goal side before pick side): IK costs ~0.5 ms per pose, a
collision check ~1 us, so the order saves almost all of the time.

Checked against the fixed world only (table + assembly base): installed parts and parts on the
table depend on the order and the layout and are checked during the search (expand / cost).
``pick_fixed_free`` is always True now (the pick configuration is required collision-free); the
field is kept for the readers in cost.py / expand.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

CACHE = Path(__file__).resolve().parent / "_cache"


def diverse(Gs, ids, k):
    """Greedy farthest-point choice of k grasps (distance = TCP offset / 2 cm + rotation
    angle in rad), so the kept grasps differ (a WRS antipodal file lists many grasps that
    are 15 deg rotations of the same contact pair)."""
    if len(ids) <= k:
        return list(ids)

    def d(a, b):
        c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
        return np.linalg.norm(a[:3, 3] - b[:3, 3]) / 0.02 + np.arccos(np.clip(c, -1, 1))
    pick = [0]
    far = np.array([d(Gs[0], G) for G in Gs])
    while len(pick) < k:
        i = int(np.argmax(far))
        pick.append(i)
        far = np.minimum(far, [d(Gs[i], G) for G in Gs])
    return [ids[i] for i in pick]


HOVER = 0.08              # m above prepick (world z): the empty-hand transit ends here, then descends
BRANCH_TOL = 0.5          # rad: max joint jump between consecutive keyposes (same IK branch)
PICK_KEYS = ("hover", "prepick", "pick", "lift")
PREINS_D = (0.05, 0.03)                                                   # above ins, world z
RETREAT_OPTS = (("tool", 0.05), ("up", 0.05), ("tool", 0.03), ("up", 0.03))
APPROACH_D = (0.05, 0.03)                                                 # prepick: back along the tool axis
LIFT_D = (0.06, 0.04)                                                     # lift: world z


def pick_keyposes(part, T, G):
    from curobo_bridge.assembly_demo import keyposes
    kp = keyposes(part, T, G)
    hover = kp["prepick"].copy()
    hover[2, 3] += HOVER
    return dict(kp, hover=hover)


def same_branch(q, ok):
    """q [4, R, 6], ok [4, R] IK returns for (hover, prepick, pick, lift). Take every
    prepick solution, attach the nearest solution of the other keyposes, keep the first
    prepick whose triple stays on one branch (every jump < BRANCH_TOL). None if none does."""
    for b in np.flatnonzero(ok[1]):
        pre = q[1, b]
        out = []
        for k in (0, 2, 3):
            cand = np.flatnonzero(ok[k])
            j = cand[np.argmin([np.abs(q[k, c] - pre).max() for c in cand])]
            out.append(q[k, j])
        hover, pick, lift = out
        if (np.abs(hover - pre).max() < BRANCH_TOL and np.abs(pick - pre).max() < BRANCH_TOL
                and np.abs(lift - pick).max() < BRANCH_TOL):
            return np.stack([hover, pre, pick, lift])
    return None


def _ik_free(bik, chk, poses, held_bank=(), hidx=-1):
    """IK of poses [N,4,4] + collision check of every returned solution in ONE batched pass.
    hidx: held-part index per pose into held_bank (-1 = empty hand).
    -> q [N,R,6], usable [N,R] (solved AND collision-free)."""
    q, ok = bik.solve(poses)
    N, R = ok.shape
    usable = np.zeros(N * R, bool)
    idx = np.flatnonzero(ok.reshape(-1))
    if len(idx):
        h = np.repeat(np.broadcast_to(np.asarray(hidx), (N,)), R)[idx]
        usable[idx] = chk.batch_free([(q.reshape(-1, 6)[idx], None, h, -1)], held_bank=held_bank)[0]
    return q, usable.reshape(N, R)


def _near(q_from, qs, use):
    """Index of the usable solution in qs [R,6] closest to q_from if within BRANCH_TOL, else None."""
    if not use.any():
        return None
    d = np.abs(qs - q_from).max(1)
    d[~use] = np.inf
    j = int(np.argmin(d))
    return j if d[j] < BRANCH_TOL else None


def build_catalogue(scene, bik, chk, max_grasps=None) -> dict:
    """See the module docstring. max_grasps: optional cap (most different grasps); None = all."""
    t_all = perf_counter()
    chk.set_world(chk.world_dict())
    grasps = scene.grasps()
    goal, pick, report = {}, {}, {}
    npre, nret = len(PREINS_D), len(RETREAT_OPTS)
    na = npre + nret
    for pid in scene.ids:
        t0 = perf_counter()
        T = {}
        Gs = np.stack([G for G, _ in grasps[pid]])
        n = len(Gs)
        held_bank = [(scene.held_spheres(pid), G) for G in Gs]
        # ---- goal side (1-3) ---------------------------------------------------------------------
        ins = scene.goal[pid][None] @ Gs
        q_ins, u_ins = _ik_free(bik, chk, ins)
        g1 = np.flatnonzero(u_ins.any(1))
        T["goal_ins"] = perf_counter() - t0
        t = perf_counter()
        feasible = {}
        if len(g1):
            aux, hid = [], []
            for g in g1:
                I = ins[g]
                for d in PREINS_D:
                    P = I.copy()
                    P[2, 3] += d
                    aux.append(P)
                    hid.append(g)                                  # holding the part
                for kind, d in RETREAT_OPTS:
                    P = I.copy()
                    if kind == "tool":
                        P[:3, 3] -= d * I[:3, 2]
                    else:
                        P[2, 3] += d
                    aux.append(P)
                    hid.append(-1)                                 # released
            q_a, u_a = _ik_free(bik, chk, np.stack(aux), held_bank, np.array(hid))
            q_a, u_a = q_a.reshape(len(g1), na, *q_a.shape[1:]), u_a.reshape(len(g1), na, -1)
            for i, g in enumerate(g1):
                for r in np.flatnonzero(u_ins[g]):
                    qi = q_ins[g, r]
                    pre = next(((k, j) for k in range(npre) if (j := _near(qi, q_a[i, k], u_a[i, k])) is not None), None)
                    ret = next(((k, j) for k in range(npre, na) if (j := _near(qi, q_a[i, k], u_a[i, k])) is not None), None)
                    if pre and ret:
                        feasible[int(g)] = dict(preins=q_a[i, pre[0], pre[1]], ins=qi, retreat=q_a[i, ret[0], ret[1]])
                        break
        T["goal_aux"] = perf_counter() - t
        gl = sorted(feasible)
        if max_grasps and len(gl) > max_grasps:
            gl = sorted(diverse([Gs[g] for g in gl], gl, max_grasps))
        # ---- pick side (4-6), only goal-feasible grasps --------------------------------------------
        cands = scene.candidates[pid]
        side, n_pick_free = {}, 0
        t = perf_counter()
        if gl:
            Tx = np.stack([Tc for _, Tc in cands])
            Pk = (Tx[:, None] @ Gs[gl][None]).reshape(-1, 4, 4)            # row = x * len(gl) + k
            q_pk, u_pk = _ik_free(bik, chk, Pk)
            rows = np.flatnonzero(u_pk.any(1))
            n_pick_free = len(rows)
            T["pick"] = perf_counter() - t
            t = perf_counter()
            if len(rows):
                poses, hid = [], []
                for r in rows:
                    Pp, g = Pk[r], gl[r % len(gl)]
                    for a in APPROACH_D:
                        pre = Pp.copy()
                        pre[:3, 3] -= a * Pp[:3, 2]
                        hov = pre.copy()
                        hov[2, 3] += HOVER
                        poses += [pre, hov]
                        hid += [-1, -1]
                    for l in LIFT_D:
                        lf = Pp.copy()
                        lf[2, 3] += l
                        poses.append(lf)
                        hid.append(g)
                nk = 2 * len(APPROACH_D) + len(LIFT_D)
                q_b, u_b = _ik_free(bik, chk, np.stack(poses), held_bank, np.array(hid))
                q_b, u_b = q_b.reshape(len(rows), nk, *q_b.shape[1:]), u_b.reshape(len(rows), nk, -1)
                for i, r in enumerate(rows):
                    x, g = cands[r // len(gl)][0], gl[r % len(gl)]
                    for s in np.flatnonzero(u_pk[r]):
                        qp = q_pk[r, s]
                        app = None
                        for ai in range(len(APPROACH_D)):
                            j = _near(qp, q_b[i, 2 * ai], u_b[i, 2 * ai])
                            if j is None:
                                continue
                            h = _near(q_b[i, 2 * ai, j], q_b[i, 2 * ai + 1], u_b[i, 2 * ai + 1])
                            if h is not None:
                                app = (q_b[i, 2 * ai + 1, h], q_b[i, 2 * ai, j])
                                break
                        lift = next((q_b[i, k, j] for k in range(2 * len(APPROACH_D), nk)
                                     if (j := _near(qp, q_b[i, k], u_b[i, k])) is not None), None)
                        if app is not None and lift is not None:
                            side.setdefault(g, {})[x] = dict(hover=app[0], prepick=app[1], pick=qp, lift=lift,
                                                             pick_fixed_free=True)
                            break
            T["pick_aux"] = perf_counter() - t
        used = [g for g in gl if side.get(g)]
        goal[pid] = {g: feasible[g] for g in used}
        pick[pid] = {}
        for g in used:
            for key, e in side[g].items():
                pick[pid].setdefault(key, {})[g] = e
        report[pid] = dict(grasps=n, goal_ins_ik_free=int(len(g1)), goal_feasible=len(feasible),
                           pick_pairs_tested=len(gl) * len(cands), pick_ik_free=n_pick_free,
                           pairs=sum(len(v) for v in pick[pid].values()), grasps_used=len(used),
                           candidates_with_grasp=len(pick[pid]), seconds={k: round(v, 2) for k, v in T.items()})
    return dict(goal=goal, pick=pick, report=report, seconds=perf_counter() - t_all)


def load_or_build(scene, chk, max_grasps=None) -> tuple[dict, str]:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{scene.name}_n{scene.n_cand}_g{max_grasps or 'all'}_wrs_v4.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes()), "cache"
    import torch
    from curobo_bridge.assembly_demo import BatchIK
    bik = BatchIK(batch=1024, seeds=32, returns=4)
    cat = build_catalogue(scene, bik, chk, max_grasps)
    del bik
    torch.cuda.empty_cache()
    path.write_bytes(pickle.dumps(cat))
    return cat, "built"
