"""Detour indicators for BLOCKED joint-straight segments (experimental; not used by LazyStepCost).

For each blocked segment qa -> qb (with its colliding stretch [s_in, s_out] from the screening
pass, an optional held part and an optional extra obstacle = the part being picked):

(A) escape  NPTS states on the colliding stretch are pushed out of collision by a projected,
            metric-weighted Newton iteration on the total penetration P(q) (sum of positive cuRobo
            collision-constraint values of robot + held spheres, sphere overlap with the picked
            part, held part vs arm, self-collision cost). dP/dq = J^T (sum of per-sphere escape
            gradients) is obtained in one shot by forward finite differences in joint space (7
            states per point, one batched FK + collision pass per iteration), which includes the
            held-part and picked-part terms that live outside cuRobo. The component along the line
            direction is removed (an escape is a sideways detour). Two metrics: plain (W = I) and
            slack-weighted (W^-1 = displacement slack of each joint w.r.t. the LB + 0.02 rad, so
            the bottleneck joint is moved last). Indicators per variant v in {p, w}:
            esc{v}_conv (share of freed points), esc{v}_dq / _norm, esc{v}_tdec (decoupled
            bound incl. reversals of the blended escape path), esc{v}_topp (TOPP time of it).
(B) detour upper bound: candidate smooth curves checked for collision in ONE batch_free call:
            raise_h    quadratic Bezier through the mid-stretch state with the TCP raised by h;
            retract_h  same, TCP moved h towards the base (+5 cm up);
            lift_h     corner-blended polyline through the stretch states each raised by h;
            escB{v}_a  Bezier through the mid-stretch state + a * its escape displacement;
            escP{v}_s  blended polyline through the stretch states + s * escape displacement.
            Every free curve gets tdec (decoupled bound, cheap) and topp (time-optimal duration
            of that collision-free path => an UPPER bound of the segment's optimum under our
            sphere model). ub = min topp over free curves (None if none).
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

RAISE = (0.05, 0.10, 0.15, 0.25)
RETRACT = (0.05, 0.10)
LIFT = (0.05, 0.10, 0.15)
ESC_ALPHA = (1.0, 1.5, 2.5)
ESC_SCALE = (1.0, 1.5)
NPTS = 5
EPS = 0.004


def _vmax():
    from curobo_bridge.assembly_demo import AMAX, VMAX, t_trap
    return np.asarray(VMAX, float), AMAX, t_trap


def travel_time(P):
    """max_j t_trap(total variation of joint j along P)."""
    V, _, t_trap = _vmax()
    tv = np.abs(np.diff(np.asarray(P, float), axis=0)).sum(0)
    return float(np.max(t_trap(tv, V)))


def t_dec(P):
    """Decoupled lower bound of a path: max_j sum over monotone pieces of q_j of t_trap(|piece|)."""
    V, _, t_trap = _vmax()
    d = np.diff(np.asarray(P, float), axis=0)
    best = 0.0
    for j in range(d.shape[1]):
        x = d[:, j]
        x = x[np.abs(x) > 1e-7]
        if len(x) == 0:
            continue
        cut = np.nonzero(np.sign(x[1:]) != np.sign(x[:-1]))[0] + 1
        pieces = np.array([abs(p.sum()) for p in np.split(x, cut)])
        best = max(best, float(np.sum(t_trap(pieces, V[j]))))
    return best


def d_trap(T, v, a):
    """Inverse of t_trap: displacement reachable rest-to-rest in time T."""
    T = np.asarray(T, float)
    return np.where(T <= 2 * v / a, a * T ** 2 / 4, v * T - v ** 2 / a)


def bezier(a, v, b, n=16):
    """Quadratic Bezier from a to b passing through v at t = 0.5, sampled at n+1 points."""
    c = 2 * v - 0.5 * (a + b)
    t = np.linspace(0, 1, n + 1)[:, None]
    return (1 - t) ** 2 * a + 2 * t * (1 - t) * c + t ** 2 * b


def blend(P, frac=0.45, n=6):
    """Corner blending of a polyline: each interior corner v is replaced by the quadratic Bezier
    from v + f (prev - v) to v + f (next - v) with control point v (C1 curve, f < 0.5)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    out = [P[0]]
    t = np.linspace(0, 1, n + 1)[:, None]
    for k in range(1, len(P) - 1):
        a = P[k] + frac * (P[k - 1] - P[k])
        b = P[k] + frac * (P[k + 1] - P[k])
        out += list((1 - t) ** 2 * a + 2 * t * (1 - t) * P[k] + t ** 2 * b)
    out.append(P[-1])
    return np.asarray(out)


class DetourProbe:
    def __init__(self, chk):
        import torch
        self.chk = chk
        self.dev = chk.device
        jl = chk.chk.kinematics.get_joint_limits().position
        self.lo, self.hi = (jl[0].cpu().numpy().astype(float), jl[1].cpu().numpy().astype(float))
        self.torch = torch

    # ---- per-state total penetration ------------------------------------------------------------
    def penetration(self, Q, held_loc, extra):
        """Q [n,6] numpy; held_loc = (HB [nh+1,m,4] padded bank, Hi [n] index); extra = (EB, Ei).
        Returns (P [n] total penetration, hit [n] bool)."""
        torch = self.torch
        chk = self.chk
        HB, Hi = held_loc
        EB, Ei = extra
        out_p, out_h = [], []
        qt = torch.as_tensor(Q, dtype=torch.float32, device=self.dev)
        for s in range(0, len(Q), chk.chunk):
            sph, R, p = chk.kinematics(qt[s:s + chk.chunk])
            b = sph.shape[0]
            nr = sph.shape[1]
            allsph = sph
            if HB is not None:
                hb = HB[Hi[s:s + b]]
                allsph = torch.cat([sph, torch.cat([torch.einsum("bij,bmj->bmi", R, hb[..., :3]) + p[:, None], hb[..., 3:]], -1)], 1)
            chk.chk.setup_batch_tensors(b, 1)
            cv = chk.chk.get_collision_constraint(allsph[:, None].contiguous())[:, 0].clamp(min=0)
            P = cv.sum(-1)
            hit = (cv > 0).any(-1)
            sc = chk.chk.get_self_collision(sph[:, None].contiguous()).reshape(b, -1).sum(-1)
            P = P + sc
            hit |= sc > 0
            if HB is not None and len(chk.arm_spheres):
                arm = sph[:, chk.arm_spheres]
                cs = max(16, int(2.5e7 / (arm.shape[1] * HB.shape[1])))
                for r0 in range(0, b, cs):
                    r = slice(r0, r0 + cs)
                    d = torch.cdist(allsph[r, nr:, :3], arm[r, :, :3]) - allsph[r, nr:, 3:4] - arm[r, None, :, 3]
                    pen = (-d.min(-1).values).clamp(min=0)
                    P[r] = P[r] + pen.sum(-1)
                    hit[r] |= pen.max(-1).values > 0
            if EB is not None:
                rows = torch.nonzero(Ei[s:s + b] < EB.shape[0] - 1).flatten()
                cs = max(16, int(2.5e7 / (allsph.shape[1] * EB.shape[1])))
                for r0 in range(0, len(rows), cs):
                    r = rows[r0:r0 + cs]
                    e = EB[Ei[s:s + b][r]]
                    d = torch.cdist(allsph[r, :, :3], e[..., :3]) - allsph[r, :, 3:4] - e[:, None, :, 3]
                    pen = (-d.min(-1).values).clamp(min=0)
                    P[r] = P[r] + pen.sum(-1)
                    hit[r] |= pen.max(-1).values > 0
            out_p.append(P.float().cpu().numpy())
            out_h.append(hit.cpu().numpy())
        return np.concatenate(out_p).astype(float), np.concatenate(out_h)

    def tcp_jac(self, Q):
        """TCP position Jacobian [n,3,6] by forward differences (one batch), and TCP positions."""
        torch = self.torch
        n = len(Q)
        X = np.concatenate([Q] + [Q + EPS * np.eye(6)[j] for j in range(6)])
        p = self.chk.kinematics(torch.as_tensor(X, dtype=torch.float32, device=self.dev))[2].cpu().numpy().astype(float)
        p0 = p[:n]
        return np.stack([(p[(j + 1) * n:(j + 2) * n] - p0) / EPS for j in range(6)], -1), p0

    def _banks(self, held_bank, extra_bank):
        torch = self.torch
        far = 1e3
        HB = EB = None
        if len(held_bank):
            loc = []
            for sph, G in held_bank:
                Gi = np.linalg.inv(G)
                loc.append(np.c_[sph[:, :3] @ Gi[:3, :3].T + Gi[:3, 3], sph[:, 3] + self.chk.buffer])
            m = max(len(x) for x in loc)
            pad = lambda x: np.r_[x, np.tile([far, far, far, 0.0], (m - len(x), 1))]
            HB = torch.as_tensor(np.stack([pad(x) for x in loc] + [pad(np.zeros((0, 4)))]), dtype=torch.float32, device=self.dev)
        if len(extra_bank):
            m = max(len(x) for x in extra_bank)
            padt = torch.tensor([far, far, far, 0.0], device=self.dev)
            EB = torch.stack([torch.cat([x, padt.expand(m - len(x), 4)]) for x in extra_bank] + [padt.expand(m, 4)])
        return HB, EB

    # ---- (A) escape --------------------------------------------------------------------------
    def escape(self, Q0, U, Winv, Hs, Es, HB, EB, iters=10):
        """Projected, metric-weighted Newton on P(q), batched over all points.
        Q0 [m,6], U [m,6] unit line directions (removed from the gradient), Winv [m,6].
        Returns (escaped states, freed flags)."""
        torch = self.torch
        Q = Q0.copy()
        m = len(Q)
        tH = lambda ix: torch.as_tensor(ix, device=self.dev)
        free = np.zeros(m, bool)
        for it in range(iters + 1):
            act = np.nonzero(~free)[0]
            if len(act) == 0:
                break
            k = len(act)
            last = it == iters
            X = Q[act] if last else np.concatenate([Q[act]] + [Q[act] + EPS * np.eye(6)[j] for j in range(6)])
            rep = len(X) // k
            P, hit = self.penetration(X, (HB, tH(np.tile(Hs[act], rep))), (EB, tH(np.tile(Es[act], rep))))
            p0, h0 = P[:k], hit[:k]
            free[act[~h0]] = True
            if last:
                break
            g = np.stack([(P[(j + 1) * k:(j + 2) * k] - p0) / EPS for j in range(6)], -1)
            g -= (g * U[act]).sum(1, keepdims=True) * U[act]
            wg = Winv[act] * g
            gn = (g * wg).sum(1)
            stepv = -np.where(gn[:, None] > 1e-12, 1.5 * p0[:, None] * wg / np.maximum(gn[:, None], 1e-12), 0.0)
            nrm = np.abs(stepv).max(1, keepdims=True)
            stepv = stepv * np.minimum(1.0, 0.3 / np.maximum(nrm, 1e-12))
            mv = h0 & (gn > 1e-12)
            Q[act[mv]] = np.clip(Q[act[mv]] + stepv[mv], self.lo, self.hi)
        return Q, free

    # ---- main --------------------------------------------------------------------------------
    def run(self, segs, held_bank=(), extra_bank=(), step=0.02, iters=10, do_topp=True):
        """segs: list of dict(qa, qb, s_in, s_out, held (idx or -1), extra (idx or -1)).
        The checker's world must be set. Returns (list of indicator dicts, timing dict ms)."""
        from lazy_cost.timing import topp
        torch = self.torch
        T = {}
        t0 = perf_counter()
        n = len(segs)
        if n == 0:
            return [], {}
        V, A_, t_trap = _vmax()
        HB, EB = self._banks(held_bank, extra_bank)
        nh, ne = len(held_bank), len(extra_bank)
        qa = np.array([s["qa"] for s in segs], float)
        qb = np.array([s["qb"] for s in segs], float)
        hidx = np.array([s["held"] if s["held"] >= 0 else nh for s in segs])
        eidx = np.array([s["extra"] if s["extra"] >= 0 else ne for s in segs])
        si = np.array([s["s_in"] if s["s_in"] is not None else 0.0 for s in segs])
        so = np.array([s["s_out"] if s["s_out"] is not None else 1.0 for s in segs])
        ts = si[:, None] + (so - si)[:, None] * np.linspace(0, 1, NPTS)[None]
        Q0 = qa[:, None] + ts[..., None] * (qb - qa)[:, None]                          # stretch states [n,NPTS,6]
        u = qb - qa
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        lb = np.max(t_trap(qb - qa, V), axis=1)
        slack = np.maximum(d_trap(lb[:, None], V[None], A_) - np.abs(qb - qa), 0.0)
        m = n * NPTS
        Winv = np.r_[np.ones((m, 6)), np.repeat(slack + 0.02, NPTS, 0)]
        Qs, fr = self.escape(np.r_[Q0.reshape(m, 6), Q0.reshape(m, 6)], np.tile(np.repeat(u, NPTS, 0), (2, 1)), Winv,
                             np.tile(np.repeat(hidx, NPTS), 2), np.tile(np.repeat(eidx, NPTS), 2), HB, EB, iters)
        torch.cuda.synchronize()
        T["escape_ms"] = 1e3 * (perf_counter() - t0)
        esc = {"p": (Qs[:m].reshape(n, NPTS, 6), fr[:m].reshape(n, NPTS)),
               "w": (Qs[m:].reshape(n, NPTS, 6), fr[m:].reshape(n, NPTS))}
        # ---- (B) candidate curves, one batched check ----------------------------------------------
        t1 = perf_counter()
        J, ptcp = self.tcp_jac(Q0.reshape(m, 6))
        JJt = np.einsum("bij,bkj->bik", J, J) + 1e-3 * np.eye(3)
        pinv = lambda dirs: np.einsum("bji,bj->bi", J, np.linalg.solve(JJt, dirs[..., None])[..., 0])
        dz = pinv(np.tile([0, 0, 1.0], (m, 1))).reshape(n, NPTS, 6)
        rad = ptcp.copy()
        rad[:, 2] = 0
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
        dr = pinv(-rad).reshape(n, NPTS, 6)
        clip = lambda q: np.clip(q, self.lo, self.hi)
        cands = []
        hm = NPTS // 2
        for i in range(n):
            a, b = qa[i], qb[i]
            mid = Q0[i, hm]
            for h in RAISE:
                cands.append((i, f"raise_{h:g}", bezier(a, clip(mid + h * dz[i, hm]), b)))
            for h in RETRACT:
                cands.append((i, f"retract_{h:g}", bezier(a, clip(mid + h * dr[i, hm] + 0.05 * dz[i, hm]), b)))
            for h in LIFT:
                cands.append((i, f"lift_{h:g}", blend(np.r_[[a], clip(Q0[i] + h * dz[i]), [b]])))
            for v, (Qe, _) in esc.items():
                D = Qe[i] - Q0[i]
                for al in ESC_ALPHA:
                    cands.append((i, f"escB{v}_{al:g}", bezier(a, clip(mid + al * D[hm]), b)))
                for sc in ESC_SCALE:
                    cands.append((i, f"escP{v}_{sc:g}", blend(np.r_[[a], clip(Q0[i] + sc * D), [b]])))
        A, B, Hl, El, own = [], [], [], [], []
        for c, (i, _, P) in enumerate(cands):
            A.append(P[:-1]), B.append(P[1:])
            k = len(P) - 1
            Hl.append(np.full(k, segs[i]["held"])), El.append(np.full(k, segs[i]["extra"])), own.append(np.full(k, c))
        A, B, Hl, El, own = map(np.concatenate, (A, B, Hl, El, own))
        ok = self.chk.batch_free([(A, B, Hl, El)], held_bank=held_bank, extra_bank=extra_bank, step=step)[0]
        cfree = np.ones(len(cands), bool)
        np.logical_and.at(cfree, own, ok)
        torch.cuda.synchronize()
        T["detour_check_ms"] = 1e3 * (perf_counter() - t1)
        # ---- cheap CPU: decoupled bounds ----------------------------------------------------------
        t2 = perf_counter()
        res = [dict(lb_line=float(lb[i]), cand={}) for i in range(n)]
        for i in range(n):
            for v, (Qe, f) in esc.items():
                D = Qe[i] - Q0[i]
                Pe = blend(np.r_[[qa[i]], Qe[i], [qb[i]]])
                res[i].update({f"esc{v}_conv": float(f[i].mean()), f"esc{v}_dq": float(np.abs(D).max()),
                               f"esc{v}_norm": float(np.linalg.norm(D, axis=1).max()),
                               f"esc{v}_tdec": t_dec(Pe), f"esc{v}_travel": travel_time(Pe),
                               f"esc{v}_jmax": int(np.argmax(np.abs(D).max(0)))})
        for c, (i, name, P) in enumerate(cands):
            res[i]["cand"][name] = dict(free=bool(cfree[c]), tdec=t_dec(P))
        T["tdec_ms"] = 1e3 * (perf_counter() - t2)
        # ---- TOPP (CPU, expensive) ---------------------------------------------------------------
        t3 = perf_counter()
        if do_topp:
            for c, (i, name, P) in enumerate(cands):
                if cfree[c]:
                    res[i]["cand"][name]["topp"] = float(topp(P)[1][-1])
        T["topp_ub_ms"] = 1e3 * (perf_counter() - t3)
        t4 = perf_counter()
        if do_topp:
            for i in range(n):
                for v, (Qe, _) in esc.items():
                    res[i][f"esc{v}_topp"] = float(topp(blend(np.r_[[qa[i]], Qe[i], [qb[i]]]))[1][-1])
        T["topp_esc_ms"] = 1e3 * (perf_counter() - t4)
        for r in res:
            f_ = {k: v for k, v in r["cand"].items() if v["free"]}
            r["any_free"] = bool(f_)
            r["n_free"] = len(f_)
            r["ub_tdec"] = min((v["tdec"] for v in f_.values()), default=None)
            r["ub"] = min((v.get("topp", np.inf) for v in f_.values()), default=None)
            r["ub_name"] = min(f_, key=lambda k: f_[k].get("topp", f_[k]["tdec"])) if f_ else None
            r["ub_pick_tdec"] = f_[min(f_, key=lambda k: f_[k]["tdec"])].get("topp") if f_ else None
        T["total_ms"] = 1e3 * (perf_counter() - t0)
        T["segments"] = n
        T["candidates"] = len(cands)
        return res, T
