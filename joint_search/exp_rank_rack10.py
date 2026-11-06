"""Experiment 2: does the fast cost rank the candidate poses of a real search state correctly?

    & $py -X utf8 -m joint_search.exp_rank_rack10 --collect            # 1. backward search, save states
    & $py -X utf8 -m joint_search.exp_rank_rack10 --states 0           # 2. WRS reference for state 0
    & $py -X utf8 -m joint_search.exp_rank_rack10 --summary            # 3. tables

1. States: the backward beam search on rack10 (same B, M, rules as run_frame, depth correction
   lazy_cost/penalty_depth.json). The states are the expansions along the BEST complete plan:
   at backward depth k the node (R, decided suffix) and the part p it decides. These are the
   states the search really expanded to reach its answer.
2. Per state, expand.expand gives every feasible (x, g) of p with the newly resolved segments
       transfer     lift(x,g) -> preins(g), holding p (gripper closed to the grasp's jaw)
       connection   retreat(g) -> hover of the head (or -> HOME at the root), empty hand
       first part   HOME -> hover(x, g) (only when p is the first part of the plan)
       fixed        hover/pick/lift + preins/ins/retreat (straight-line bounds, identical on both sides)
   two estimates (the search's own numbers, no change):
       lb      straight joint-line lower bound of every segment
       depth   lb + c * max penetration depth of the blocked lines (c = 4.7 s/m)
   p's own transit (start not known yet) is a lower bound in both and is left out.
   Reference (WRS, CPU) for the same segments: WRS mesh check of the straight line (same jaw,
   held part moved with the TCP); free -> its lower bound; blocked -> WRS RRT-Connect, mesh
   re-check, shortcut + corner blend, TOPP time with the same joint limits.
   A candidate pose x is scored by its best grasp: per x, the GRASPS planned are the union of the
   top-2 grasps of each method; truth(x) = min over them, est_m(x) = min over p's grasps of method m.
3. Metrics per state (over the poses with a finite WRS time): Spearman / Kendall of est vs truth,
   top-1 hit, true best in the top-5 (the search keeps M = 5 per expansion), regret of the top-1
   pick, mean |est/truth - 1|.
Output: joint_search/_output/rank_rack10/ (states.pkl, state_<i>.json, summary.md).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np

OUT = Path(__file__).resolve().parent / "_output" / "rank_rack10"
PENALTY = str(Path(__file__).resolve().parents[1] / "lazy_cost" / "penalty_depth.json")


def collect(args):
    from .problem import Problem
    from .search import beam_search
    pb = Problem("rack10", penalty=PENALTY)
    print(f"rack10: parts {list(pb.ids)}; usable grasps per part {[len(pb.cat['goal'][p]) for p in pb.ids]}; "
          f"setup {pb.setup_s:.0f}s", flush=True)
    complete, st = beam_search(pb, B=args.B, M=args.M, N=10)
    if not complete:
        raise SystemExit("no complete plan")
    best = complete[0]
    print(f"best plan est {best.g_est:.3f} s (lb {best.g_lb:.3f}); search {st['search_s']:.1f}s, {st['expansions']} expansions")
    # nodes along the best plan: backward depth k decides suffix[-k]
    from .search import Node
    states = []
    for k in range(1, len(best.suffix) + 1):
        suf = best.suffix[len(best.suffix) - k + 1:]
        p = best.suffix[len(best.suffix) - k][0]
        node = Node(R=frozenset(pb.ids) - {q for q, _, _ in suf}, suffix=suf, table={q: x for q, x, _ in suf})
        states.append(dict(depth=k, part=p, step=len(best.suffix) - k + 1, node=node, chosen=best.suffix[len(best.suffix) - k]))
        print(f"  state {k - 1}: backward depth {k}, forward step {len(best.suffix) - k + 1}: decide {p}; "
              f"installed {len(node.R) - 1}, on the table {len(node.table)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "states.pkl").write_bytes(pickle.dumps(dict(states=states, plan=best.suffix, est=best.g_est, lb=best.g_lb,
                                                       search=st)))
    print("->", OUT / "states.pkl")


def reference(args):
    from lazy_cost.detour import blend
    from lazy_cost.timing import topp
    from .expand import expand
    from .problem import HOME, Problem, t_acc
    from .wrs_check import MeshWorld
    D = pickle.loads((OUT / "states.pkl").read_bytes())
    pb = Problem("rack10", penalty=PENALTY)
    sc, cat = pb.scene, pb.cat
    W = MeshWorld(pb)
    W.planner.rrt_kw["max_time"] = args.rrt_time
    G = sc.grasps()

    import random
    import wrs.motion.probabilistic.rrt_connect as rrtc

    def rrt(a, b, seed):
        """WRS RRT-Connect whose node check is the mesh test of W.free (arm + held part moved with the
        TCP + gripper at the set jaw) instead of the arm's own AABB check (which neither moves the
        held part nor uses the meshes)."""
        np.random.seed(seed); random.seed(seed)
        pl = rrtc.RRTConnect(W.arm)
        pl._is_collided = lambda conf, *_, **__: (not W.arm.are_jnts_in_ranges(jnt_values=conf)) or (not W.free([conf, conf])[0])
        md = pl.plan(start_conf=np.asarray(a, float), goal_conf=np.asarray(b, float), obstacle_list=W.obs,
                     start_obstacle_list=[], goal_obstacle_list=[],
                     **dict(W.planner.rrt_kw, smoothing_n_iter=0))   # WRS smoothing checks long edges sparsely; we shortcut densely
        return None if md is None else np.asarray(md.jv_list, float)

    globals()["_RRT"] = rrt

    def wrs_time(a, b):
        t0 = perf_counter()
        lb = float(t_acc(a, b))
        if W.free([a, b])[0]:
            return lb, "straight", perf_counter() - t0
        if not W.free([a, a])[0] or not W.free([b, b])[0]:
            return np.inf, "endpoint_in_collision", perf_counter() - t0
        for seed in range(args.rrt_seeds):
            P = rrt(a, b, seed)
            if P is None or not W.free(P)[0]:
                continue
            S = W.shortcut(P)
            t = float(topp(S)[1][-1])
            if len(S) > 2 and W.free(blend(S))[0]:
                t = min(t, float(topp(blend(S))[1][-1]))
            return t, "rrt", perf_counter() - t0
        return np.inf, "rrt_failed", perf_counter() - t0

    if args.debug is not None:
        debug(pb, W, D["states"][args.debug], wrs_time)
        return
    for i in args.states or []:
        s = D["states"][i]
        node, p = s["node"], s["part"]
        R1 = node.R - {p}
        expand(pb, node, p)                                           # CUDA warm-up for this world
        import torch
        torch.cuda.synchronize(); t0 = perf_counter()
        kids, stt = expand(pb, node, p)
        torch.cuda.synchronize(); fast_s = perf_counter() - t0
        # transition points missing (no IK at 5/4/3 cm): above insert / retreat fell back to the insertion
        # moment, above pick to the pick moment -> the part could only enter / leave along one exact line;
        # left out on both sides (the search should not use them either)
        ok_aux = lambda c: not (np.allclose(cat["goal"][p][c.g]["preins"], cat["goal"][p][c.g]["ins"])
                                or np.allclose(cat["goal"][p][c.g]["retreat"], cat["goal"][p][c.g]["ins"])
                                or np.allclose(cat["pick"][p][c.x][c.g]["lift"], cat["pick"][p][c.x][c.g]["pick"]))
        n_all = len(kids)
        kids = [c for c in kids if ok_aux(c)]
        print(f"state {i}: {n_all} feasible pairs, {n_all - len(kids)} without a transition point left out", flush=True)
        by_x = {}
        for c in kids:
            by_x.setdefault(c.x, []).append(c)
        sel = {x: sorted({c.g for c in sorted(cs, key=lambda c: c.add_lb)[:args.grasps]}
                         | {c.g for c in sorted(cs, key=lambda c: c.add_est)[:args.grasps]}) for x, cs in by_x.items()}
        print(f"state {i}: step {s['step']} decide {p} | installed {sorted(R1)} | table {node.table} | "
              f"{len(by_x)} poses, {len(kids)} pairs, fast {1e3 * fast_s:.0f} ms | WRS: {sum(map(len, sel.values()))} pairs", flush=True)
        tbl = {q: sc.pose[q][k] for q, k in node.table.items()}
        head = node.head
        conn, segs, t_all = {}, [], perf_counter()

        def connection(g):                                            # retreat(g) -> head hover / HOME, after releasing p
            if g not in conn:
                W.set(sorted(node.R), tbl)
                tgt = HOME if head is None else cat["pick"][head[0]][head[1]][head[2]]["hover"]
                conn[g] = wrs_time(cat["goal"][p][g]["retreat"], tgt)
            return conn[g]
        rows = {}
        for x, gs in sel.items():
            child = {c.g: c for c in by_x[x]}
            vals = []
            for g in gs:
                c, e, v = child[g], cat["pick"][p][x][g], cat["goal"][p][g]
                W.set(sorted(R1), tbl, (p, G[p][g][0], e["lift"]), jaw=pb.jaw[p][g] if pb.close_fingers else None)
                tf = wrs_time(e["lift"], v["preins"])
                cn = connection(g)
                t = pb.fix_pick[p, x, g] + pb.fix_goal[p, g] + tf[0] + cn[0]
                parts = [("transfer", tf), ("connection", cn)]
                if len(R1) == 0:
                    W.set([], dict(tbl, **{p: sc.pose[p][x]}))
                    tr = wrs_time(HOME, e["hover"])
                    t += tr[0]
                    parts.append(("home_transit", tr))
                vals.append(t)
                ours = dict(transfer=next(sg for sg in c.segs if sg["kind"] == "transfer"),
                            connection=next(sg for sg in c.segs if sg["kind"] in ("transit", "return_home") and sg.get("after") == p),
                            home_transit=next((sg for sg in c.segs if sg.get("after") == "HOME"), None))
                for kind, (tt, how, sec) in parts:
                    o = ours[kind]
                    segs.append(dict(x=x, g=g, kind=kind, lb=o["lb"], est=o["est"], blocked=o["blocked"], depth=o["depth"],
                                     wrs=how, wrs_time=tt if np.isfinite(tt) else None, wrs_s=sec))
            rows[x] = dict(lb=min(c.add_lb for c in by_x[x]), depth=min(c.add_est for c in by_x[x]),
                           truth=min(vals), grasps=gs)
            hows = " ".join(f"{d['kind'][:4]}:{d['wrs']}" for d in segs if d["x"] == x and d["g"] == gs[int(np.argmin(vals))])
            print(f"    {x:24s} lb {rows[x]['lb']:.3f}  depth {rows[x]['depth']:.3f}  WRS {rows[x]['truth']:.3f}  "
                  f"({hows})  [{perf_counter() - t_all:.0f}s]", flush=True)
        W.set([], {})
        res = dict(state=i, step=s["step"], part=p, installed=sorted(R1), table=node.table, head=head, chosen=s["chosen"],
                   poses=len(by_x), pairs=len(kids), fast_s=fast_s, wrs_s=perf_counter() - t_all,
                   rows={x: dict(r, truth=r["truth"] if np.isfinite(r["truth"]) else None) for x, r in rows.items()},
                   segments=segs, metrics=metrics(rows))
        (OUT / f"state_{i}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        m = res["metrics"]
        print(f"  -> state {i}: WRS {res['wrs_s']:.0f}s | " + " | ".join(
            f"{k}: rho {v['spearman']:.2f} tau {v['kendall']:.2f} top1 {v['top1']:.0f} top5 {v['best_in_top5']:.0f} "
            f"regret {100 * v['regret']:.1f}% err {100 * v['err']:.1f}%" for k, v in m.items() if k in ("lb", "depth")), flush=True)


def debug(pb, W, s, wrs_time):
    from time import perf_counter
    """Why WRS finds no time: check the endpoints of the first candidates' segments."""
    from .expand import expand
    from .problem import HOME
    sc, cat, G = pb.scene, pb.cat, pb.scene.grasps()
    node, p = s["node"], s["part"]
    R1 = node.R - {p}
    kids, _ = expand(pb, node, p)
    tbl = {q: sc.pose[q][k] for q, k in node.table.items()}
    head = node.head
    print("head", head)
    import sys; me = sys.modules["__main__"]
    c = next(c for c in kids if c.g == 92)
    e, v = cat["pick"][p][c.x][c.g], cat["goal"][p][c.g]
    W.set(sorted(R1), tbl, (p, G[p][c.g][0], e["lift"]), jaw=pb.jaw[p][c.g])
    print("straight", W.free([e["lift"], v["preins"]]), "dq", np.round(v["preins"] - e["lift"], 2))
    for seed in range(3):
        t0 = perf_counter(); P = me._RRT(e["lift"], v["preins"], seed)
        print("seed", seed, "path", None if P is None else len(P), f"{perf_counter() - t0:.1f}s", flush=True)
        if P is not None:
            bad = [i for i, q in enumerate(P) if not W.free([q, q])[0]]
            ebad = [i for i in range(len(P) - 1) if not W.free([P[i], P[i + 1]])[0]]
            print("   colliding nodes", bad[:10], "colliding edges", ebad[:10], "max edge step", np.abs(np.diff(P, axis=0)).max().round(3))
            if ebad:
                print("   edge", ebad[0], W.free([P[ebad[0]], P[ebad[0] + 1]]), P[ebad[0]].round(3), P[ebad[0] + 1].round(3))
        break
    return
    for c in kids[:3]:
        e, v = cat["pick"][p][c.x][c.g], cat["goal"][p][c.g]
        W.set(sorted(R1), tbl, (p, G[p][c.g][0], e["lift"]), jaw=pb.jaw[p][c.g])
        print(c.x, c.g, "transfer", W.free([e["lift"], v["preins"]]), wrs_time(e["lift"], v["preins"]), flush=True)
        W.set(sorted(node.R), tbl)
        tgt = HOME if head is None else cat["pick"][head[0]][head[1]][head[2]]["hover"]
        print("   connection", W.free([v["retreat"], tgt]), wrs_time(v["retreat"], tgt), flush=True)


def metrics(rows):
    from scipy.stats import kendalltau, spearmanr
    fin = [x for x, r in rows.items() if r["truth"] is not None and np.isfinite(r["truth"])]
    out = dict(n=len(fin), n_inf=len(rows) - len(fin))
    if len(fin) < 3:
        return out
    tv = np.array([rows[x]["truth"] for x in fin])
    best = tv.min()
    for m in ("lb", "depth"):
        pv = np.array([rows[x][m] for x in fin])
        o = np.argsort(pv, kind="stable")
        out[m] = dict(spearman=float(spearmanr(pv, tv)[0]), kendall=float(kendalltau(pv, tv)[0]),
                      top1=float(tv[o[0]] <= best + 1e-9), best_in_top5=float(best >= tv[o[:5]].min() - 1e-9),
                      regret=float(tv[o[0]] / best - 1), err=float(np.mean(np.abs(pv / tv - 1))),
                      bias=float(np.mean(pv / tv - 1)))
    return out


def summary(args):
    files = sorted(OUT.glob("state_*.json"), key=lambda f: int(f.stem.split("_")[1]))
    L = ["| 状态 | 步骤 | 零件 | 已安装 | 桌上 | 候选位置 | WRS 有解 | 方法 | Spearman | Kendall | 第 1 名命中 | 最优在前 5 | 选第 1 名的时间损失 | 平均误差 | 偏差 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        m = r["metrics"]
        for k, name in (("lb", "只用下界"), ("depth", "下界+修正")):
            if k not in m:
                continue
            v = m[k]
            L.append(f"| {r['state']} | {r['step']} | {r['part']} | {len(r['installed'])} | {len(r['table'])} | {r['poses']} | {m['n']} | {name} | "
                     f"{v['spearman']:.2f} | {v['kendall']:.2f} | {'是' if v['top1'] else '否'} | {'是' if v['best_in_top5'] else '否'} | "
                     f"{100 * v['regret']:.1f}% | {100 * v['err']:.1f}% | {100 * v['bias']:+.1f}% |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--states", type=int, nargs="*", default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--debug", type=int, default=None)
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--grasps", type=int, default=2, help="top grasps per pose and method planned by WRS")
    ap.add_argument("--rrt-seeds", type=int, default=2)
    ap.add_argument("--rrt-time", type=float, default=20.0)
    args = ap.parse_args(argv)
    if args.collect:
        collect(args)
    if args.states is not None or args.debug is not None:
        reference(args)
    if args.summary:
        summary(args)


if __name__ == "__main__":
    main()
