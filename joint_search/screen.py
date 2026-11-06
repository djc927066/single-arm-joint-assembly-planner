"""IK table for the search: the two moments that decide feasibility, plus transition points for timing.

For every part p and every WRS grasp g (G = TCP pose in the part frame, jaw = its jaw width):
    ins  = goal_p @ G          the moment the part reaches its assembly pose (gripper closed)
    pick = T_x @ G             the moment the gripper closes on the part at candidate pose x
IK is solved once here (cuRobo, batched, 32 seeds, 4 solutions) and one solution per pose is
kept: the first one that is collision-free against the fixed world (table + assembly base), the
gripper closed to the grasp's jaw width (CLOSE_FINGERS). Goal side first; the pick side only for
the grasps whose insertion moment has such a solution (IK costs ~0.5 ms per pose, a collision
check ~1 us). The search (expand.py) then checks these configurations against the parts that are
installed / on the table in each state - that is the only screening of a grasp.

Fixed transition points, one each (2026-10-24; no fallback heights):
    above pick    the open fingertips CLEAR (2 cm) above the top of the part at its candidate pose,
                  straight above the pick: the empty hand arrives here (hover), goes straight down,
                  closes, and goes straight back up to the same point (lift)
    above insert  PREINS (5 cm) above the insertion, against the part's insertion direction (task
                  insertion axis; the frame inserts straight down): the transfer ends here, then
                  the part is inserted straight
    retreat       the open fingertips CLEAR above the top of the installed part, straight above the
                  release: the next empty-hand move starts here
A point without an IK solution eliminates: the grasp (above insert, retreat) or the (pose, grasp)
pair (above pick); of the IK solutions the one closest to the moment's configuration is used.

Output (same layout the rest of joint_search reads):
    goal[p][g] = dict(preins=above insert, ins, retreat)
    pick[p][x][g] = dict(hover=above pick, prepick=pick (no separate pre-grasp point), pick,
                         lift=above pick)
    jaw[p][g] = jaw width
"""
from __future__ import annotations

import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

CACHE = Path(__file__).resolve().parent / "_cache"
# True: the gripper is closed to each grasp's jaw width at the insertion / pick moments and while
# holding the part (checker.batch_free jaw argument); False: modelled fully open (8.5 cm) everywhere
CLOSE_FINGERS = True
PREINS = 0.05       # above-insert point this far before the insertion (m)
CLEAR = 0.02        # open fingertips this far above the part's top at the hover / lift / retreat point (m)


def _top(scene, pid, T):
    """Height of the part's highest point at world pose T."""
    v = np.asarray(scene.meshes[pid].vertices)
    return float((v @ T[2, :3] + T[2, 3]).max())


def _rise(chk, q, top):
    """Vertical TCP rise [N] that puts the lowest point of the open fingers CLEAR above top [N]."""
    import torch
    out = np.empty(len(q))
    idx = torch.cat(list(chk.finger.values()))
    for s in range(0, len(q), 4096):
        sph, _, _ = chk.kinematics(torch.as_tensor(q[s:s + 4096], dtype=torch.float32, device="cuda"))
        out[s:s + 4096] = (sph[:, idx, 2] - sph[:, idx, 3]).min(1).values.cpu().numpy()
    return np.maximum(np.asarray(top) + CLEAR - out, 0.0)


def _solve_free(bik, chk, poses, jaw):
    """IK of poses [N,4,4]; the first returned solution collision-free (gripper closed to jaw [N])
    against the checker's current world. -> q [N,6], ok [N]."""
    q, ok = bik.solve(poses)
    N, R = ok.shape
    idx = np.flatnonzero(ok.reshape(-1))
    free = np.zeros(N * R, bool)
    if len(idx):
        jw = np.repeat(jaw, R)[idx] if CLOSE_FINGERS else None
        free[idx] = chk.batch_free([(q.reshape(-1, 6)[idx], None, -1, -1, jw)])[0]
    free = free.reshape(N, R)
    first = free.argmax(1)
    return q[np.arange(N), first], free.any(1)


def _nearest(bik, poses, q_ref):
    """IK of transition poses [N,4,4]; the solution closest to q_ref [N,6] (q_ref if none)."""
    if not len(poses):                                  # a part without any usable grasp
        return np.asarray(q_ref, float).reshape(-1, 6).copy(), np.zeros(0, bool)
    q, ok = bik.solve(poses)
    d = np.abs(q - q_ref[:, None]).max(-1)
    d[~ok] = np.inf
    j = d.argmin(1)
    has = np.isfinite(d[np.arange(len(q)), j])
    return np.where(has[:, None], q[np.arange(len(q)), j], q_ref), has


def build(scene, chk):
    from curobo_bridge.assembly_demo import BatchIK
    bik = BatchIK(batch=1024, seeds=32, returns=4)
    chk.set_world(chk.world_dict())
    t_all = perf_counter()
    goal, pick, jaw, report = {}, {}, {}, {}
    for pid in scene.ids:
        t0 = perf_counter()
        Gs = np.stack([G for G, _ in scene.grasps()[pid]])
        jw = np.array([j for _, j in scene.grasps()[pid]], float)
        # ---- insertion moment, every grasp -------------------------------------------------------
        ins = scene.goal[pid][None] @ Gs
        q_ins, ok_ins = _solve_free(bik, chk, ins, jw)
        g1 = np.flatnonzero(ok_ins)
        t_goal = perf_counter() - t0
        # ---- pick moment, goal-feasible grasps x candidates --------------------------------------
        t1 = perf_counter()
        cands = scene.candidates[pid]
        Tx = np.stack([T for _, T in cands])
        Pk = (Tx[:, None] @ Gs[g1][None]).reshape(-1, 4, 4) if len(g1) else np.zeros((0, 4, 4))
        q_pk, ok_pk = _solve_free(bik, chk, Pk, np.tile(jw[g1], len(cands))) if len(Pk) else (np.zeros((0, 6)), np.zeros(0, bool))
        t_pick = perf_counter() - t1
        # ---- transition points of the surviving poses (timing only) -------------------------------
        t2 = perf_counter()
        rows = np.flatnonzero(ok_pk)
        used = sorted({int(g1[r % len(g1)]) for r in rows})
        I = ins[used]
        qi = q_ins[used]

        def shift(P, v, d):
            P = P.copy(); P[:, :3, 3] += np.asarray(d, float)[..., None] * np.asarray(v, float); return P
        up = np.array([0.0, 0.0, 1.0])
        out = -np.asarray(getattr(scene, "removal_axis", {}).get(pid, up), float)   # insertion direction
        q_pre, h1 = _nearest(bik, shift(I, -out / np.linalg.norm(out), PREINS), qi)          # above insert
        r_ret = _rise(chk, qi, np.full(len(qi), _top(scene, pid, scene.goal[pid])))
        q_ret, h2 = _nearest(bik, shift(I, up, r_ret), qi)                                   # retreat
        keep = {g for g, a, b in zip(used, h1, h2) if a and b}
        rows = np.array([r for r in rows if int(g1[r % len(g1)]) in keep], int)
        P, qp = Pk[rows], q_pk[rows]
        tops = {x: _top(scene, pid, T) for x, T in cands}
        r_pk = _rise(chk, qp, np.array([tops[cands[r // len(g1)][0]] for r in rows]))
        q_above, h3 = _nearest(bik, shift(P, up, r_pk), qp) if len(rows) else (np.zeros((0, 6)), np.zeros(0, bool))
        t_aux = perf_counter() - t2
        pick[pid] = {}
        for i, r in enumerate(rows):
            if h3[i]:
                x, g = cands[r // len(g1)][0], int(g1[r % len(g1)])
                pick[pid].setdefault(x, {})[g] = dict(hover=q_above[i], prepick=qp[i], pick=qp[i], lift=q_above[i])
        kept = {g for d in pick[pid].values() for g in d}                  # grasps left with a pick pose
        goal[pid] = {g: dict(preins=q_pre[i], ins=qi[i], retreat=q_ret[i]) for i, g in enumerate(used) if g in kept}
        jaw[pid] = {g: float(jw[g]) for g in kept}
        report[pid] = dict(grasps=len(Gs), ins_ok=int(len(g1)), pick_ok=int(ok_pk.sum()), pick_pairs=int(h3.sum()),
                           grasps_used=len(kept), candidates_with_grasp=len(pick[pid]),
                           dropped=dict(above_insert=int((~h1).sum()), retreat=int((~h2).sum()),
                                        above_pick=int((~h3).sum())),
                           rise_m=dict(retreat=float(np.median(r_ret)) if len(r_ret) else None,
                                       above_pick=float(np.median(r_pk)) if len(r_pk) else None),
                           seconds=dict(goal=round(t_goal, 2), pick=round(t_pick, 2), transition=round(t_aux, 2)))
    del bik
    return dict(goal=goal, pick=pick, jaw=jaw, report=report, seconds=perf_counter() - t_all)


def load_or_build(scene, chk):
    CACHE.mkdir(exist_ok=True)
    smp = "" if getattr(scene, "sampler", "random") == "random" else f"_{scene.sampler}{scene.pool}"
    if getattr(scene, "placements", "goal") != "goal":
        smp += f"_pl{scene.placements}3"                                   # "3": former max_placements, keeps the names
    if getattr(scene, "grid", "shared") != "shared":
        smp += f"_g{scene.grid}" + (f"{1e3 * scene.grid_step:g}" if scene.grid == "fine" else "")
    if getattr(scene, "keepout", "box") != "box":
        smp += f"_k{scene.keepout}"
    if getattr(scene, "home_removed", 0):
        smp += f"_h{scene.home_removed}c{1e3 * getattr(scene, 'home_clear', 0):g}"                                    # options removed by the HOME check
    if getattr(scene, "center", None) is not None:
        smp += "_c" + "_".join(f"{1e3 * v:g}" for v in scene.center)        # assembly centre, mm
    path = CACHE / (f"{scene.name}_n{scene.n_cand}{smp}_ik_v5_{'closed' if CLOSE_FINGERS else 'open'}"
                   f"_d{chk.density:g}_b{1e3 * chk.buffer:g}_r{1e3 * max(chk.self_relief.values(), default=0):g}.pkl")
    if path.exists():
        return pickle.loads(path.read_bytes()), "cache"
    import gc
    import torch
    tab = build(scene, chk)
    gc.collect()                # the cuRobo IK solver sits in reference cycles: ~6 GB stays allocated until
    torch.cuda.empty_cache()    # collected, and the rest of the run then spills out of the 8 GB GPU (20-40x slower)
    path.write_bytes(pickle.dumps(tab))
    return tab, "built"
