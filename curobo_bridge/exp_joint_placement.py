"""实验：摆放 + 抓取 + 空手/持件两段 的联合优化，对比网格穷举真值。

    & $py -X utf8 -m curobo_bridge.exp_joint_placement            # 完整（真值网格 0.06 m）
    & $py -X utf8 -m curobo_bridge.exp_joint_placement --smoke    # 小网格

场景与 ``exp_free_end`` 相同（Bridge，装配位 x=0.44，前一件 post_r 刚装完，求横梁 beam）。

* 方法：``placement_opt.ChainOptimizer`` 一次批量优化 (接触面对 × 种子) 个问题，变量为
  两段路径 + 摆放（由持件段终点决定）+ 抓取滚转/滑移；再对通过合法性检查的前 N 个结果用
  cuRobo 定点规划两段、同一 TOPP 规则计时，取最优。
* 真值：网格位置 × 4 偏航 × 离散抓取族（滚转 30°、滑移 −0.08/0/+0.08），两段定点规划。
  方法的抓取是连续的，可能落在离散族之外，所以可以比网格真值更好。
* 基线：事先固定一个抓取；按直线关节时间挑（不看障碍）。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare


def main(argv=None):
    prepare()
    import torch
    from curobo.scene import Scene
    from . import exp_free_end as E
    from .curobo_models import load_robot_dict, make_collision_checker, with_buffer
    from .export_model import fk_frames, load_chain
    from .placement_opt import ChainOptimizer, ChainProblem

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=float, default=0.06)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--verify", type=int, default=8, help="top optimised proposals verified by cuRobo")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    chain = load_chain(ASSET_DIR / "panthera_wrsext")
    tcp_of = lambda q: fk_frames(chain, q)[5] @ np.asarray(chain["tcp"])

    parts, grasps, load_pose = E.build()
    kos = E.keepouts(load_pose)
    sdict = E.scene_dict(parts, load_pose, E.slabs(load_pose))
    beam = parts["beam"]["mesh"]
    bg = parts["beam"]["goal"]
    pl = E.Planner(sdict)
    rep = dict(created=datetime.now().isoformat(timespec="seconds"), args=vars(args))

    # ---- anchors (same as exp_free_end) ------------------------------------------------------
    pr_goal = parts["post_r"]["goal"]
    R = np.stack([pr_goal @ g["G"] for g in grasps["post_r"]])
    R[:, :3, 3] -= E.RETREAT * R[:, :3, 2]
    q_r, ok_r = pl.ik(R)
    cand = [i for i in np.flatnonzero(ok_r) if R[i, 2, 2] < -0.5] or list(np.flatnonzero(ok_r))
    q_ret = q_r[min(cand, key=lambda i: np.abs(q_r[i]).sum())]
    Gs = grasps["beam"]
    Pp = np.stack([bg @ g["G"] for g in Gs])
    Pp[:, 2, 3] += E.PREINS
    q_pre, ok_pre = pl.ik(Pp)
    feas = [i for i in range(len(Gs)) if ok_pre[i] and (bg @ Gs[i]["G"])[2, 2] < 0.3]
    ext = beam.extents

    def slide_axis(G):
        free = [i for i in range(3) if abs(G[i, 1]) < 0.5]
        return max(free, key=lambda i: ext[i])

    # ---- method: joint optimisation ----------------------------------------------------------
    t_m = time.perf_counter()
    pool = []
    while len(pool) < 96:
        T = E.staging(rng.uniform(0.02, 0.64), rng.uniform(-0.39, 0.39), rng.uniform(0, np.pi), beam)
        if E.placement_ok(beam, T, kos):
            pool.append(T)
    rows = []   # (variant index, pool index, q_lift0, q_pp0)
    for gi in feas:
        G = Gs[gi]["G"]
        Pl = np.stack([E.lift_tcp(T, G) for T in pool])
        Pe = np.stack([E.prepick_tcp(T, G) for T in pool])
        ql, okl = pl.ik(Pl)
        qe, oke = pl.ik(Pe)
        good = np.flatnonzero(okl & oke)
        if not len(good):
            continue
        for i in good[E.spread(np.stack([pool[j][:2, 3] for j in good]), args.seeds)]:
            rows.append((gi, i, ql[i], qe[i]))
    t_seed = time.perf_counter() - t_m
    N = len(rows)
    # zero-roll / zero-slide contact pair frame; theta0/slide0 reproduce the seed variant
    base = []
    for gi, *_ in rows:
        g = Gs[gi]
        Gz = g["G"].copy()
        Gz[:3, :3] = g["G"][:3, :3] @ E._ry(-np.deg2rad(g["roll"]))
        Gz[slide_axis(g["G"]), 3] -= g["off"]
        base.append(Gz)
    L = ext[0] / 2
    xs = np.arange(-L + 0.017, L - 0.016, 0.034)
    r_s = float(np.sqrt(0.017 ** 2 + (ext[1] / 2) ** 2 + (ext[2] / 2) ** 2))
    pb = ChainProblem(
        q_retreat=q_ret, goal=bg, G0=np.stack(base),
        slide_axis=np.array([slide_axis(Gs[gi]["G"]) for gi, *_ in rows]), slide_max=L - 0.03,
        obj_spheres=np.stack([[x, 0, 0, r_s] for x in xs]), obj_half_height=ext[2] / 2 + 2e-4,
        obj_line=np.stack([[x, 0, 0] for x in np.linspace(-L, L, 7)]), obj_radius=ext[1] / 2,
        forbidden=[(c[0], c[1], s[0] / 2, s[1] / 2) for c, s in kos],
        table_lo=E.TABLE_C[:2] - E.TABLE_S[:2] / 2, table_hi=E.TABLE_C[:2] + E.TABLE_S[:2] / 2,
        lift=E.LIFT, approach=E.APPROACH, preins=E.PREINS)
    robot = with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", 1.0), 0.002)
    chk = make_collision_checker(robot, Scene.create(sdict), activation=0.02)
    opt = ChainOptimizer(chk, chain, iters=args.iters)
    yaw_of = lambda T: np.arctan2(T[1, 0], T[0, 0])
    place0 = np.stack([[pool[i][0, 3], pool[i][1, 3], yaw_of(pool[i])] for _, i, _, _ in rows])
    res = opt.solve(pb, place0,
                    np.array([np.deg2rad(Gs[gi]["roll"]) for gi, *_ in rows]),
                    np.array([Gs[gi]["off"] for gi, *_ in rows]),
                    np.stack([r[3] for r in rows]), np.stack([r[2] for r in rows]),
                    np.stack([q_pre[gi] for gi, *_ in rows]))
    print("optimiser", dict(problems=N, seconds=round(res["seconds"], 2)), flush=True)
    for l in res["log"]:
        print("   ", l, flush=True)
    # legality + ranking by the optimiser's own time proxy
    cands = []
    for i in range(N):
        x, y, a = res["place"][i]
        T_obj = E.staging(x, y, a, beam)
        ok = res["ik"][i] < 1e-5 and res["reg"][i] < 1e-6 and E.placement_ok(beam, T_obj, kos)
        cands.append(dict(i=i, T=T_obj, G=res["G"][i], ok=bool(ok), proxy=float(res["T"][i]),
                          col=float(res["col"][i])))
    legal = sorted([c for c in cands if c["ok"]], key=lambda c: (c["col"] > 1e-4, c["proxy"]))
    print(f"legal proposals {len(legal)}/{N}", flush=True)
    # cuRobo pinned verification of the best few (true cost, same TOPP rule)
    best = None
    t_v = time.perf_counter()
    for c in legal[: args.verify]:
        G = c["G"]
        Ppi = bg @ G
        Ppi[2, 3] += E.PREINS
        q_pi, ok_pi = pl.ik(Ppi[None])
        if not ok_pi[0]:
            c["verify"] = "preins_ik"
            continue
        pe = pl.plan(q_ret[None], E.prepick_tcp(c["T"], G)[None])[0]
        pl.hold(q_pi[0], beam, tcp_of(q_pi[0]) @ np.linalg.inv(G))
        pc = pl.plan(q_pi, E.lift_tcp(c["T"], G)[None])[0]
        pl.release()
        if pe is None or pc is None:
            c["verify"] = "plan_failed"
            continue
        c["empty"], c["carry"] = E.leg_time(pe), E.leg_time(pc[::-1])
        c["cost"] = c["empty"] + c["carry"]
        c["verify"] = "ok"
        if best is None or c["cost"] < best["cost"]:
            best = c
    moved = np.linalg.norm(res["place"][:, :2] - place0[:, :2], axis=1)
    dyaw = np.degrees(np.abs((res["place"][:, 2] - place0[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    rep["method"] = dict(
        seconds=time.perf_counter() - t_m, seed_ik_seconds=t_seed, optimiser_seconds=res["seconds"],
        verify_seconds=time.perf_counter() - t_v, problems=N, legal=len(legal),
        placement_moved_m=dict(median=float(np.median(moved)), max=float(np.max(moved))),
        yaw_moved_deg=dict(median=float(np.median(dyaw)), max=float(np.max(dyaw))),
        verified=[dict(proxy=c["proxy"], cost=c.get("cost"), verify=c.get("verify"),
                       xy=c["T"][:2, 3].tolist()) for c in legal[: args.verify]],
        best=None if best is None else dict(cost=best["cost"], empty=best["empty"], carry=best["carry"],
                                            xy=best["T"][:2, 3].tolist(),
                                            yaw=float(np.degrees(yaw_of(best["T"]))),
                                            theta=float(np.degrees(res["theta"][best["i"]])),
                                            slide=float(res["slide"][best["i"]])),
        log=res["log"])
    rep["proposals"] = [dict(xy=c["T"][:2, 3].tolist(), ok=c["ok"], proxy=c["proxy"],
                             seed_xy=place0[c["i"], :2].tolist()) for c in cands]
    print("method", {k: v for k, v in rep["method"].items() if k not in ("log", "verified")}, flush=True)
    for v in rep["method"]["verified"]:
        print("   verified", v, flush=True)

    # ---- control: the same seeds without optimisation (true cost, same protocol) ---------------
    t_c = time.perf_counter()
    seed_cost = np.full(N, np.nan)
    pe_all = pl.plan(np.repeat(q_ret[None], N, 0),
                     np.stack([E.prepick_tcp(pool[i], Gs[gi]["G"]) for gi, i, _, _ in rows]))
    for gi in sorted({r[0] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r[0] == gi]
        G = Gs[gi]["G"]
        pl.hold(q_pre[gi], beam, tcp_of(q_pre[gi]) @ np.linalg.inv(G))
        pcs = pl.plan(np.repeat(q_pre[gi][None], len(idx), 0), np.stack([E.lift_tcp(pool[rows[k][1]], G) for k in idx]))
        pl.release()
        for k, pc in zip(idx, pcs):
            if pc is not None and pe_all[k] is not None:
                seed_cost[k] = E.leg_time(pe_all[k]) + E.leg_time(pc[::-1])
    opt_cost = np.array([c.get("cost", np.nan) for c in cands])
    rep["control"] = dict(seconds=time.perf_counter() - t_c, seed_best=float(np.nanmin(seed_cost)),
                          seed_median=float(np.nanmedian(seed_cost)), seed_feasible=int(np.isfinite(seed_cost).sum()),
                          per_problem=[dict(seed=None if not np.isfinite(a) else float(a),
                                            opt=None if not np.isfinite(b) else float(b))
                                       for a, b in zip(seed_cost, opt_cost)])
    print("control (seeds, no optimisation)", {k: v for k, v in rep["control"].items() if k != "per_problem"}, flush=True)

    # ---- ground truth grid (same protocol as exp_free_end) ---------------------------------
    t_t = time.perf_counter()
    step = 0.12 if args.smoke else args.grid
    places = [E.staging(x, y, a, beam) for x in np.arange(0.05, 0.63, step)
              for y in np.arange(-0.36, 0.37, step) for a in np.deg2rad([0, 45, 90, 135])]
    places = [T for T in places if E.placement_ok(beam, T, kos)]
    empty_c = np.full((len(places), len(Gs)), np.nan)
    carry_c = np.full((len(places), len(Gs)), np.nan)
    lb = np.full((len(places), len(Gs)), np.nan)
    for gi in feas:
        G = Gs[gi]["G"]
        paths = pl.plan(np.repeat(q_ret[None], len(places), 0), np.stack([E.prepick_tcp(T, G) for T in places]))
        for i, p in enumerate(paths):
            if p is not None:
                empty_c[i, gi] = E.leg_time(p)
                lb[i, gi] = E.vel_lb(q_ret, p[-1])
        pl.hold(q_pre[gi], beam, tcp_of(q_pre[gi]) @ np.linalg.inv(G))
        paths = pl.plan(np.repeat(q_pre[gi][None], len(places), 0), np.stack([E.lift_tcp(T, G) for T in places]))
        for i, p in enumerate(paths):
            if p is not None:
                carry_c[i, gi] = E.leg_time(p[::-1])
                if np.isfinite(lb[i, gi]):
                    lb[i, gi] += E.vel_lb(p[-1], q_pre[gi])
        pl.release()
        print(f"  truth grasp {gi} done ({time.perf_counter() - t_t:.0f}s)", flush=True)
    chain_c = empty_c + carry_c
    finite = chain_c[np.isfinite(chain_c)]
    ib, gb = np.unravel_index(np.nanargmin(chain_c), chain_c.shape)
    truth_best = float(chain_c[ib, gb])
    pri = min(feas, key=lambda gi: ((bg @ Gs[gi]["G"])[2, 2], abs(Gs[gi]["off"]), Gs[gi]["pair"]))
    fixed_best = float(np.nanmin(chain_c[:, pri])) if np.isfinite(chain_c[:, pri]).any() else None
    j = np.nanargmin(np.where(np.isfinite(chain_c), lb, np.nan))
    li, lg = np.unravel_index(j, lb.shape)
    m_cost = rep["method"]["best"]["cost"] if rep["method"]["best"] else None
    rep["truth"] = dict(seconds=time.perf_counter() - t_t, placements=len(places), grasps=len(feas),
                        plans=2 * len(places) * len(feas), feasible_pairs=int(np.isfinite(chain_c).sum()),
                        best=truth_best, best_xy=places[ib][:2, 3].tolist(), median=float(np.median(finite)))
    rep["compare"] = dict(
        method_cost=m_cost, truth_best=truth_best,
        method_gap_pct=None if m_cost is None else 100 * (m_cost - truth_best) / truth_best,
        method_beats_share_of_grid=None if m_cost is None else float((finite > m_cost).mean()),
        seed_only_best=rep["control"]["seed_best"],
        seed_only_gap_pct=100 * (rep["control"]["seed_best"] - truth_best) / truth_best,
        fixed_grasp_best=fixed_best,
        fixed_grasp_gap_pct=None if fixed_best is None else 100 * (fixed_best - truth_best) / truth_best,
        straight_line_choice_cost=float(chain_c[li, lg]),
        straight_line_gap_pct=100 * (chain_c[li, lg] - truth_best) / truth_best,
        time_method_s=rep["method"]["seconds"], time_truth_s=rep["truth"]["seconds"],
        cuRobo_plans_method=2 * min(args.verify, len(legal)), cuRobo_plans_truth=rep["truth"]["plans"])
    print("truth", rep["truth"], flush=True)
    print("compare", rep["compare"], flush=True)
    per_xy = {}
    for i, T in enumerate(places):
        k = (round(T[0, 3], 3), round(T[1, 3], 3))
        v = np.nanmin(chain_c[i]) if np.isfinite(chain_c[i]).any() else np.nan
        per_xy[k] = np.nanmin([per_xy.get(k, np.nan), v])
    rep["heat"] = [dict(x=k[0], y=k[1], cost=None if not np.isfinite(v) else float(v)) for k, v in per_xy.items()]
    rep["keepouts"] = [dict(c=c.tolist(), s=s.tolist()) for c, s in kos]
    rep["load_xy"] = load_pose[:2, 3].tolist()
    rep["site"] = E.SITE.tolist()
    out = OUTPUT_DIR / f"joint_placement_{'smoke_' if args.smoke else ''}{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
