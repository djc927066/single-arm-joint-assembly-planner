"""Replay the feat3 contexts (same seed -> same contexts) and log, for every blocked segment,
three kinds of "how much detour" indicators (no cuRobo planning; the de-taxed reference is
already in the feat3 jsonl and is matched by the evaluation):

  depth      max penetration of any sphere along the joint line (already logged, recomputed here)
  volume     the joint line densely interpolated (0.02 rad); per state P = SUM over all robot /
             held-part spheres of the penetration (cuRobo collision constraint) + self-collision
             + held part vs arm + picked-part overlap; pen_max = max_state P, pen_int = mean_state P
             * joint length (the "accumulated collision amount" along the line)
  bump       the deepest state is pushed out of collision by Newton steps on P using its gradient
             dP/dq (finite differences, component along the line removed); two metrics: plain and
             joint-slack weighted (joints with time slack w.r.t. the lower bound move first).
             The push dq becomes a smooth out-and-back bump on the line,
                 q(u) = qa + u (qb - qa) + alpha dq b(u),  b(u*) = 1 at the deepest state,
             timed with lazy_cost.timing.topp (same timing as the reference):
             bump_t1   alpha = 1 (just the push, not checked)
             bump_free smallest alpha in (1, 1.5, 2, 3) whose bump is collision-free (our checker)
                       -> the time of an explicitly constructed collision-free detour
             bump_tdec decoupled (per-joint, no path coupling) time of the alpha = 1 bump

    python -m lazy_cost.exp_slack --model rack10 --boxes 0,0 --crowd 0 --gap 0.03 --contexts 30 --tag _parts
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from time import perf_counter

import numpy as np

from .exp_rack10 import OUT, make_context

ALPHAS = (1.0, 1.5, 2.0, 3.0)
NB = 40          # bump samples


def bump_profile(u, us):
    """C1 out-and-back profile on [0, 1], 0 at both ends, 1 (flat) at us."""
    us = float(np.clip(us, 0.15, 0.85))
    S = lambda x: 3 * x ** 2 - 2 * x ** 3
    return np.where(u <= us, S(np.clip(u / us, 0, 1)), S(np.clip((1 - u) / (1 - us), 0, 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="rack10")
    ap.add_argument("--contexts", type=int, default=30)
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--boxes", default="6,10")
    ap.add_argument("--crowd", type=float, default=0.08)
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-cand", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    import torch

    from .checker import GpuChecker
    from .cost import LazyStepCost
    from .detour import DetourProbe, _vmax, d_trap, t_dec
    from .endpoints import load_or_build
    from .scene import load_scene
    from .timing import topp

    V, A_, t_trap = _vmax()
    rng = np.random.default_rng(args.seed)
    scene = load_scene(args.model, n_cand=args.n_cand)
    chk = GpuChecker(scene)
    cat, _ = load_or_build(scene, chk)
    probe = DetourProbe(chk)
    lb_only = LazyStepCost(scene, cat, chk, None, k=args.k)
    sync = torch.cuda.synchronize
    OUT.mkdir(exist_ok=True)
    out = OUT / f"exp_{args.model}_slack{args.tag}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print("->", out, flush=True)

    for ci in range(args.contexts):
        def start_free(w, q):
            chk.set_world(chk.world_dict(*w))
            return bool(chk.configs_free(q)[0])

        c = make_context(scene, cat, rng, ci, args.cap, start_free, boxes_range=tuple(int(v) for v in args.boxes.split(",")),
                         crowd=args.crowd, gap=args.gap)
        if c is None:
            print(f"ctx {ci}: no context", flush=True)
            continue
        p, start, cands = c["part"], c["start"], c["cands"]
        r_lb, _ = lb_only.rank(p, start, c["installed"], c["placed"], c["boxes"], cands)   # sets the world W
        G = scene.grasps()[p]
        part_sph = scene.held_spheres(p)
        gl = sorted(cat["goal"][p])
        gidx = {g: i for i, g in enumerate(gl)}
        xs = [r.key for r in r_lb if not r.status.startswith("pruned")]
        xidx = {x: i for i, x in enumerate(xs)}
        segs, keys = [], []
        for r in r_lb:
            if r.status.startswith("pruned"):
                continue
            for g, sg in r.segments.items():
                e = cat["pick"][p][r.key][g]
                for kind in ("transit", "transfer"):
                    if sg[kind][1] != "bound":
                        continue
                    qa, qb = (start, e["hover"]) if kind == "transit" else (e["lift"], cat["goal"][p][g]["preins"])
                    segs.append(dict(qa=np.asarray(qa, float), qb=np.asarray(qb, float), depth=float(r.features[g][kind]["depth"]),
                                     held=gidx[g] if kind == "transfer" else -1, extra=xidx[r.key] if kind == "transit" else -1))
                    keys.append((r.key, int(g), kind))
        n = len(segs)
        if n == 0:
            print(f"ctx {ci}: no blocked segment", flush=True)
            continue
        held_bank = [(part_sph, G[g][0]) for g in gl]
        extra_bank = [chk.obstacle_spheres(part_sph, scene.pose[p][x]) for x in xs]
        HB, EB = probe._banks(held_bank, extra_bank)
        nh, ne = len(held_bank), len(extra_bank)
        hidx = np.array([s["held"] if s["held"] >= 0 else nh for s in segs])
        eidx = np.array([s["extra"] if s["extra"] >= 0 else ne for s in segs])
        T = {}
        # ---- (volume) dense penetration sums along every line ------------------------------------
        sync()
        t0 = perf_counter()
        Qs, own, us = [], [], []
        for i, s in enumerate(segs):
            K = int(np.ceil(np.max(np.abs(s["qb"] - s["qa"])) / 0.02)) + 1
            u = np.linspace(0, 1, max(K, 2))
            Qs.append(s["qa"] + u[:, None] * (s["qb"] - s["qa"]))
            own.append(np.full(len(u), i))
            us.append(u)
        Qs, own, us = np.concatenate(Qs), np.concatenate(own), np.concatenate(us)
        dev = chk.device
        P, hit = probe.penetration(Qs, (HB, torch.as_tensor(hidx[own], device=dev)), (EB, torch.as_tensor(eidx[own], device=dev)))
        sync()
        T["volume_ms"] = 1e3 * (perf_counter() - t0)
        # ---- (bump) push the deepest state out, plain and slack-weighted metric -----------------
        t0 = perf_counter()
        qa = np.array([s["qa"] for s in segs])
        qb = np.array([s["qb"] for s in segs])
        Lj = np.linalg.norm(qb - qa, axis=1)
        lb = np.max(t_trap(qb - qa, V), axis=1)
        feats, Q0, ustar = [], [], []
        for i in range(n):
            m = own == i
            Pi, hi_, ui = P[m], hit[m], us[m]
            k = int(np.argmax(Pi))
            feats.append(dict(pen_max=float(Pi.max()), pen_int=float(Pi.mean() * Lj[i]), hit_share=float(hi_.mean()),
                              u_star=float(ui[k]), lj=float(Lj[i]), lb_line=float(lb[i])))
            Q0.append(qa[i] + ui[k] * (qb[i] - qa[i]))
            ustar.append(ui[k])
        Q0 = np.array(Q0)
        u_dir = (qb - qa) / np.maximum(Lj[:, None], 1e-9)
        slack = np.maximum(d_trap(lb[:, None], V[None], A_) - np.abs(qb - qa), 0.0)
        Winv = np.r_[np.ones((n, 6)), slack + 0.02]
        Qe, fr = probe.escape(np.r_[Q0, Q0], np.r_[u_dir, u_dir], Winv, np.r_[hidx, hidx], np.r_[eidx, eidx], HB, EB, iters=15)
        sync()
        T["escape_ms"] = 1e3 * (perf_counter() - t0)
        # ---- bump curves, one batched collision check ----------------------------------------------
        t0 = perf_counter()
        uu = np.linspace(0, 1, NB + 1)
        curves = []          # (seg, metric, alpha, path)
        for i in range(n):
            b = bump_profile(uu, ustar[i])[:, None]
            for mi, met in enumerate(("p", "w")):
                dq = Qe[mi * n + i] - Q0[i]
                for al in ALPHAS:
                    Pc = np.clip(qa[i] + uu[:, None] * (qb[i] - qa[i]) + al * dq * b, probe.lo, probe.hi)
                    curves.append((i, met, al, Pc))
        Aa = np.concatenate([c_[3][:-1] for c_ in curves])
        Bb = np.concatenate([c_[3][1:] for c_ in curves])
        Hl = np.concatenate([np.full(NB, segs[c_[0]]["held"]) for c_ in curves])
        El = np.concatenate([np.full(NB, segs[c_[0]]["extra"]) for c_ in curves])
        cown = np.repeat(np.arange(len(curves)), NB)
        ok = chk.batch_free([(Aa, Bb, Hl, El)], held_bank=held_bank, extra_bank=extra_bank, step=0.02)[0]
        cfree = np.ones(len(curves), bool)
        np.logical_and.at(cfree, cown, ok)
        sync()
        T["bump_check_ms"] = 1e3 * (perf_counter() - t0)
        # ---- timing of the bumps (CPU) -----------------------------------------------------------
        t0 = perf_counter()
        n_topp = 0
        for ci_, (i, met, al, Pc) in enumerate(curves):
            f = feats[i]
            if al == 1.0:
                f[f"dq_{met}"] = float(np.linalg.norm(Pc[NB // 2] - (qa[i] + 0.5 * (qb[i] - qa[i]))))
                f[f"esc_ok_{met}"] = bool(fr[(0 if met == "p" else n) + i])
                f[f"bump_t1_{met}"] = float(topp(Pc)[1][-1])
                f[f"bump_tdec_{met}"] = t_dec(Pc)
                n_topp += 1
            if cfree[ci_] and f"bump_free_{met}" not in f:
                f[f"bump_free_{met}"] = f[f"bump_t1_{met}"] if al == 1.0 else float(topp(Pc)[1][-1])
                f[f"bump_alpha_{met}"] = al
                n_topp += al != 1.0
        T["topp_ms"] = 1e3 * (perf_counter() - t0)
        T["segments"], T["curves"], T["topp_calls"] = n, len(curves), n_topp
        rows = [dict(cand=k_[0], g=k_[1], kind=k_[2], qa=s["qa"].tolist(), qb=s["qb"].tolist(), depth=s["depth"], **f)
                for k_, s, f in zip(keys, segs, feats)]
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(ctx=ci, part=p, k=c["k"], n_cands=len(cands), timing=T, segs=rows), default=float) + "\n")
        nf = np.mean([("bump_free_p" in f) or ("bump_free_w" in f) for f in feats])
        print(f"ctx {ci}: {p} blocked segs {n} | volume {T['volume_ms']:.0f} ms, escape {T['escape_ms']:.0f} ms, "
              f"bump check {T['bump_check_ms']:.0f} ms, topp {T['topp_ms']:.0f} ms | free bump found {nf:.0%}", flush=True)


if __name__ == "__main__":
    main()
