"""Evaluate the exp_slack indicators against the de-taxed reference (R1) of the feat3 runs.

    python -m lazy_cost._design.eval_slack

Truth of a segment: free line -> its lower bound; blocked -> de-taxed cuRobo time (shortcut + blend,
never below the bound). Estimators of a blocked segment (free segments always = bound):
  lb          lower bound only
  shipped     lb * (1 + 0.2431 + 1.0931 depth)            (current penalty_model.json)
  depth       lb + c * max penetration depth              c fitted on the OTHER assembly
  pen_max     lb + c * max over states of the summed penetration of all spheres
  pen_int     lb + c * summed penetration integrated along the joint line ("collision volume")
  bump_tdec   decoupled per-joint time of the gradient push bump (slack only, no path coupling)
  bump_t1     TOPP time of the slack-weighted push bump (alpha = 1)           no fitted parameter
  bump_free   TOPP time of the first collision-free push bump (min over both metrics);
              if no bump is free: depth model                                   no fitted parameter
Segment level: Spearman of predicted vs true extra time (truth - lb), mean |log(pred / truth)|.
Ranking per context (candidate cost = best grasp of transit + transfer): mean / max regret,
share with the chosen candidate within 2 % of the best, Spearman over candidates.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .eval_detour import seg_truth

OUT = Path(__file__).resolve().parents[1] / "_output"
fin = lambda v: v is not None and np.isfinite(v)
ASM = ("rack10", "triangular_frame6")
LIN = ("depth", "pen_max", "pen_int")
METHODS = ("lb", "shipped", "depth", "pen_max", "pen_int", "bump_tdec", "bump_t1", "bump_free")


def load(asm, scen):
    f3 = [json.loads(l) for f in sorted(glob.glob(str(OUT / f"exp_{asm}_feat3_{scen}_*.jsonl"))) for l in open(f, encoding="utf-8")]
    sl = {r["ctx"]: r for f in sorted(glob.glob(str(OUT / f"exp_{asm}_slack_{scen}_*.jsonl"))) for r in map(json.loads, open(f, encoding="utf-8"))}
    ctxs = []
    for r in f3:
        ind = {(s["cand"], s["g"], s["kind"]): s for s in sl.get(r["ctx"], {}).get("segs", [])}
        segs = []
        for s in r["seg_data"]:
            t = seg_truth(s, "R1")
            if not fin(t):
                continue
            d = dict(cand=s["cand"], g=s["g"], kind=s["kind"], lb=s["lb"], truth=t, blocked=s["how"] != "straight",
                     desc=s.get("desc", 0.0), depth=s.get("depth", 0.0) or 0.0)
            if d["blocked"]:
                x = ind.get((s["cand"], s["g"], s["kind"]))
                if x is None:
                    continue
                d.update({k: v for k, v in x.items() if k not in ("qa", "qb", "cand", "g", "kind", "depth")})
            segs.append(d)
        ctxs.append(dict(ctx=r["ctx"], segs=segs))
    return ctxs


def fit_c(ctxs, feat):
    """Least squares through the origin: truth - lb = c * feat (blocked segments)."""
    x = np.array([s[feat] for c in ctxs for s in c["segs"] if s["blocked"]])
    y = np.array([s["truth"] - s["lb"] for c in ctxs for s in c["segs"] if s["blocked"]])
    return float(x @ y / max(x @ x, 1e-12))


def predict(s, m, C):
    lb = s["lb"]
    if not s["blocked"] or m == "lb":
        return lb
    if m == "shipped":
        return lb * (1 + 0.2431 + 1.0931 * s["depth"])
    if m in LIN:
        return lb + C[m] * s[m]
    if m == "bump_tdec":
        return max(lb, s["bump_tdec_w"] + s["desc"])
    if m == "bump_t1":
        return max(lb, s["bump_t1_w"] + s["desc"])
    if m == "bump_free":
        v = [s[k] for k in ("bump_free_p", "bump_free_w") if fin(s.get(k))]
        return max(lb, min(v) + s["desc"]) if v else lb + C["depth"] * s["depth"]


def rank_metrics(c, m, C):
    est, tru = {}, {}
    by = {}
    for s in c["segs"]:
        by.setdefault((s["cand"], s["g"]), {})[s["kind"]] = s
    for (x, g), d in by.items():
        if len(d) < 2:
            continue
        e = sum(predict(s, m, C) for s in d.values())
        t = sum(s["truth"] for s in d.values())
        est[x] = min(est.get(x, np.inf), e)
        tru[x] = min(tru.get(x, np.inf), t)
    if len(est) < 3:
        return None
    ks = list(est)
    best = min(tru.values())
    pick = min(ks, key=lambda k: est[k])
    reg = tru[pick] / best - 1
    return dict(regret=reg, top1=float(reg <= 0.02), rho=spearmanr([est[k] for k in ks], [tru[k] for k in ks])[0])


def main():
    res = {}
    for scen in ("parts", "dense"):
        data = {a: load(a, scen) for a in ASM}
        for test in ASM:
            other = [a for a in ASM if a != test][0]
            C = {f: fit_c(data[other], f) for f in LIN}
            B = [s for c in data[test] for s in c["segs"] if s["blocked"]]
            y = np.array([s["truth"] - s["lb"] for s in B])
            tr = np.array([s["truth"] for s in B])
            r = dict(n_blocked=len(B), n_ctx=len(data[test]), C=C,
                     bump_free_found=float(np.mean([fin(s.get("bump_free_p")) or fin(s.get("bump_free_w")) for s in B])),
                     bump_free_ge_truth=float(np.mean([min([s[k] for k in ("bump_free_p", "bump_free_w") if fin(s.get(k))]) + s["desc"] >= s["truth"] - 1e-6
                                                       for s in B if fin(s.get("bump_free_p")) or fin(s.get("bump_free_w"))])),
                     extra_median=float(np.median(y)), seg={}, rank={})
            for m in METHODS:
                pr = np.array([predict(s, m, C) for s in B])
                rho = spearmanr(pr - np.array([s["lb"] for s in B]), y)[0] if m != "lb" else float("nan")
                r["seg"][m] = dict(rho=float(rho), err=float(np.mean(np.abs(np.log(pr / tr)))))
                rm = [x for x in (rank_metrics(c, m, C) for c in data[test]) if x]
                r["rank"][m] = dict(n=len(rm), regret=float(np.mean([x["regret"] for x in rm])), max_regret=float(np.max([x["regret"] for x in rm])),
                                    top1=float(np.mean([x["top1"] for x in rm])), rho=float(np.nanmean([x["rho"] for x in rm])))
            res[f"{scen}/{test}"] = r
    (Path(__file__).parent / "eval_slack.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, r in res.items():
        print(f"\n== {k}: {r['n_blocked']} blocked segments, {r['n_ctx']} contexts, median true extra {1e3 * r['extra_median']:.0f} ms, "
              f"free bump found {r['bump_free_found']:.0%} (>= truth in {r['bump_free_ge_truth']:.0%}); "
              f"c fitted on other: " + ", ".join(f"{f} {v:.2f}" for f, v in r["C"].items()))
        print(f"   {'method':10s} {'seg rho':>8s} {'seg err':>8s} | {'regret %':>9s} {'max %':>7s} {'top1':>6s} {'rank rho':>9s}")
        for m in METHODS:
            a, b = r["seg"][m], r["rank"][m]
            print(f"   {m:10s} {a['rho']:8.2f} {a['err']:8.3f} | {100 * b['regret']:9.2f} {100 * b['max_regret']:7.1f} {b['top1']:6.2f} {b['rho']:9.3f}")


if __name__ == "__main__":
    main()
