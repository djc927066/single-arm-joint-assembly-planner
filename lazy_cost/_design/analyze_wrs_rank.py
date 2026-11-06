"""Per-segment analysis of exp_wrs_rank: replay the contexts (same seed), recover each segment's
lower bound and penetration depth from the fast cost, join with the WRS reference.

    python -m lazy_cost._design.analyze_wrs_rank
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "_output"


def main():
    from curobo_bridge.runtime import prepare
    prepare()
    from scipy.stats import spearmanr
    from lazy_cost.checker import GpuChecker
    from lazy_cost.cost import LazyStepCost
    from lazy_cost.endpoints import load_or_build
    from lazy_cost.exp_rack10 import make_context, metrics
    from lazy_cost.scene import load_scene

    rows = {r["ctx"]: r for r in map(json.loads, (OUT / "wrs_rank_triangular_frame6.jsonl").read_text(encoding="utf-8").splitlines())}
    sc = load_scene("triangular_frame6")
    chk = GpuChecker(sc)
    cat, _ = load_or_build(sc, chk)
    lb_only = LazyStepCost(sc, cat, chk, None, k=3)
    rng = np.random.default_rng(11)
    segs = []
    for ci in range(max(rows) + 1):
        def start_free(w, q):
            chk.set_world(chk.world_dict(*w))
            return bool(chk.configs_free(q)[0])
        c = make_context(sc, cat, rng, ci, 10, start_free, boxes_range=(0, 0), crowd=0.0, gap=0.03)
        if c is None or ci not in rows:
            continue
        ranked, _ = lb_only.rank(c["part"], c["start"], c["installed"], c["placed"], c["boxes"], c["cands"], refine=False)
        info = {}
        for r in ranked:
            for g, sg in r.segments.items():
                for kind in ("transit", "transfer"):
                    info[r.key, int(g), kind] = (sg[kind][0], sg[kind][1], float(r.features[g][kind]["depth"]))
        from lazy_cost.cost import t_acc
        goal = cat["goal"][c["part"]]
        for s in rows[ci]["segments"]:
            lb, how, d = info[s["cand"], s["g"], s["kind"]]
            e = cat["pick"][c["part"]][s["cand"]][s["g"]]
            fixed = float(t_acc(e["prepick"], e["pick"]) + t_acc(e["pick"], e["lift"]) + t_acc(goal[s["g"]]["preins"], goal[s["g"]]["ins"])
                          + t_acc(goal[s["g"]]["ins"], goal[s["g"]]["retreat"]))
            segs.append(dict(ctx=ci, cand=s["cand"], g=s["g"], kind=s["kind"], fixed=fixed, lb=lb, ours=how, depth=d,
                             wrs=s["wrs"], t=s["wrs_time"], check_s=s["wrs_s"] if s["wrs"] == "straight" else None))
    S = segs
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]
    out = {}
    fp = [s for s in S if s["ours"] == "bound" and s["wrs"] == "straight"]
    tb = [s for s in S if s["ours"] == "bound" and s["wrs"] != "straight"]
    out["false_blocked"] = dict(n=len(fp), depth_mm=q([1e3 * s["depth"] for s in fp]), new_penalty_s=q([4.7 * s["depth"] for s in fp]),
                                old_penalty_s=q([s["lb"] * (0.2431 + 1.0931 * s["depth"]) for s in fp]))
    out["true_blocked"] = dict(n=len(tb), depth_mm=q([1e3 * s["depth"] for s in tb]),
                               wrs_over_lb=q([s["t"] / s["lb"] for s in tb]), wrs_extra_s=q([s["t"] - s["lb"] for s in tb]),
                               new_penalty_s=q([4.7 * s["depth"] for s in tb]),
                               rho_depth_extra=float(spearmanr([s["depth"] for s in tb], [s["t"] - s["lb"] for s in tb])[0]),
                               fit_c=float(np.dot([s["depth"] for s in tb], [s["t"] - s["lb"] for s in tb]) / np.dot([s["depth"] for s in tb], [s["depth"] for s in tb])))
    bl = [s for s in S if s["ours"] == "bound"]
    out["depth_separates_false_blocked"] = dict(
        auc=float(np.mean([[a["depth"] > b["depth"] for b in fp] for a in tb])) if fp and tb else None)
    for thr in (0.002, 0.004, 0.006, 0.01):
        out[f"thr_{int(thr * 1e3)}mm"] = dict(false_blocked_below=int(sum(s["depth"] < thr for s in fp)),
                                             true_blocked_below=int(sum(s["depth"] < thr for s in tb)))
    # ---- re-scored rankings: penalty only where a mesh re-check of the straight line also says blocked
    from collections import defaultdict
    est = lambda s, m: (s["lb"] if s["ours"] == "straight" or m == "lb" else
                        s["lb"] * (1 + 0.2431 + 1.0931 * s["depth"]) if m == "old" else
                        s["lb"] + 4.7 * s["depth"] if m == "new" else
                        (s["lb"] if s["wrs"] == "straight" else s["lb"] + 4.7 * s["depth"]))
    by = defaultdict(lambda: defaultdict(dict))
    for s_ in S:
        by[s_["ctx"]][s_["cand"], s_["g"]][s_["kind"]] = s_
    res = {m: [] for m in ("lb", "old", "new", "new_mesh")}
    for ci, pairs in by.items():
        truth, pred = {}, {m: {} for m in res}
        for (x, g), d in pairs.items():
            if len(d) < 2:
                continue
            fx = d["transit"]["fixed"]
            t = fx + sum(v["t"] for v in d.values())
            truth[x] = min(truth.get(x, np.inf), t)
            for m in res:
                pred[m][x] = min(pred[m].get(x, np.inf), fx + sum(est(v, m) for v in d.values()))
        for m in res:
            mm = metrics(pred[m], truth, 3)
            if mm:
                mm["err"] = float(np.mean([abs(pred[m][x] / truth[x] - 1) for x in truth]))
                res[m].append(mm)
    out["rescored"] = {m: dict(contexts=len(v), spearman=float(np.mean([x["spearman"] for x in v])), top1=float(np.mean([x["top1"] for x in v])),
                               top1_2pct=float(np.mean([x["top1_within2pct"] for x in v])), top3=float(np.mean([x["topk_recall"] for x in v])),
                               regret=float(np.mean([x["regret"] for x in v])), max_regret=float(max(x["regret"] for x in v)),
                               cost_err=float(np.mean([x["err"] for x in v]))) for m, v in res.items()}
    chk_s = [s_["check_s"] for s_ in S if s_["check_s"] is not None]
    out["mesh_recheck_s_per_segment"] = float(np.median(chk_s)) if chk_s else None
    print(json.dumps(out, indent=1))
    (OUT / "wrs_rank_segments.json").write_text(json.dumps(dict(summary=out, segments=S), default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
