"""Can "lower bound + blocked-ness correction" rank candidates as well as lazy refinement?

    & $py -X utf8 -m lazy_cost.analyze_penalty [files ...]   (default: _output/exp_*_feat_*.jsonl)
Output: printed tables + lazy_cost/_output/analyze_penalty.json

Data: exp_rack10 runs with per-segment features (seg_data): for every (candidate, grasp,
segment) its joint-time lower bound, whether the straight line is blocked, the blocked-ness
measures from the same collision pass (checker._blocked_features) and the cuRobo reference
time of that segment.

Estimate of a segment:  free -> lb;  blocked -> lb * (1 + e(features)), e >= 0.
Models for e:  none (plain bound) | const (one factor) | linear (non-negative least squares on
a few features). A candidate's estimate is the min over its surviving grasps of
transit + transfer; rankings are compared with the reference (full cuRobo planning) per
context. Cross-checks: fit on one assembly / scene, test on the others.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "_output"
FEATS = ("frac", "span", "depth", "depth_mean", "held", "part", "z_low_inv")


def load(files):
    data = []
    for f in files:
        name = Path(f).name
        model = "triangular_frame6" if "triangular" in name else "rack10"
        scene = "dense" if "dense" in name else "parts"
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("seg_data") or not r.get("metrics"):
                continue
            r["_model"], r["_scene"] = model, scene
            data.append(r)
    return data


def phi(s):
    span = (s["s_out"] - s["s_in"]) if s.get("s_in") is not None and s.get("s_out") is not None else 0.0
    z = s.get("z_low")
    return np.array([s["frac"], span, s["depth"], s["depth_mean"], float(s["by_held"] > 0), float(s["by_part"] > 0),
                     0.0 if z is None else 1.0 / max(z, 0.02)])


def segments(data):
    X, y = [], []
    for r in data:
        for s in r["seg_data"]:
            if s["how"] != "bound" or s["ref"] is None or not np.isfinite(s["ref"]) or s["lb"] <= 0:
                continue
            X.append(phi(s))
            y.append(max(s["ref"] / s["lb"] - 1.0, 0.0))
    return np.array(X), np.array(y)


def fit(data, kind):
    if kind == "none":
        return lambda f: 0.0
    X, y = segments(data)
    if kind == "const":
        c = float(np.mean(y)) if len(y) else 0.0
        return lambda f: c
    from scipy.optimize import nnls
    A = np.c_[X, np.ones(len(X))]
    w, _ = nnls(A, y)
    return lambda f: float(np.r_[f, 1.0] @ w)


def rank_eval(data, e_fn):
    from .exp_rack10 import metrics
    ms = []
    for r in data:
        by = {}
        for s in r["seg_data"]:
            est = s["lb"] if s["how"] == "straight" else s["lb"] * (1.0 + e_fn(phi(s)))
            by.setdefault((s["cand"], s["g"]), {})[s["kind"]] = est
        pred = {}
        for (x, g), v in by.items():
            if len(v) == 2:
                pred[x] = min(pred.get(x, np.inf), v["transit"] + v["transfer"])
        truth = {k: (v if v is not None else np.inf) for k, v in r["truth"].items()}
        m = metrics(pred, truth, 3)
        if m:
            ms.append(m)
    keys = ("spearman", "top1_within2pct", "topk_recall", "regret")
    return {k: float(np.mean([m[k] for m in ms])) for k in keys} | dict(max_regret=float(max(m["regret"] for m in ms)), n=len(ms))


def lazy_eval(data):
    ms = [r["metrics"]["lazy"] for r in data if r["metrics"].get("lazy")]
    keys = ("spearman", "top1_within2pct", "topk_recall", "regret")
    return {k: float(np.mean([m[k] for m in ms])) for k in keys} | dict(max_regret=float(max(m["regret"] for m in ms)), n=len(ms))


def main(argv=None):
    files = (argv if argv else sys.argv[1:]) or sorted(glob.glob(str(OUT / "exp_*_feat_*.jsonl")))
    data = load(files)
    groups = sorted({(r["_model"], r["_scene"]) for r in data})
    print("files:", [Path(f).name for f in files])
    print("contexts:", {f"{m}/{s}": sum(1 for r in data if (r["_model"], r["_scene"]) == (m, s)) for m, s in groups})

    # ---- 1. segment level: how much a blocked segment really costs, and which feature tells it --------
    from scipy.stats import spearmanr
    print("\n[blocked segments] real / lower bound, and Spearman correlation of each feature with it")
    report = {"segments": {}, "ranking": {}}
    for m, s in groups + [("all", "all")]:
        sub = [r for r in data if m == "all" or (r["_model"], r["_scene"]) == (m, s)]
        X, y = segments(sub)
        if len(y) < 10:
            continue
        cor = {n: float(spearmanr(X[:, i], y)[0]) for i, n in enumerate(FEATS) if np.ptp(X[:, i]) > 0}
        q = np.percentile(1 + y, [50, 90, 100])
        print(f"  {m:18s} {s:6s} n={len(y):4d} real/LB p50 {q[0]:.3f} p90 {q[1]:.3f} max {q[2]:.2f} | "
              + " ".join(f"{k} {v:+.2f}" for k, v in cor.items()))
        report["segments"][f"{m}/{s}"] = dict(n=int(len(y)), p50=float(q[0]), p90=float(q[1]), max=float(q[2]), spearman=cor)

    # ---- 2. ranking: in-sample and cross-assembly / cross-scene -----------------------------------
    print("\n[ranking vs full cuRobo planning]  spearman / top1<=2% / top3 recall / mean regret / max regret")
    rows = []
    for m, s in groups:
        test = [r for r in data if (r["_model"], r["_scene"]) == (m, s)]
        other = [r for r in data if r["_model"] != m]                    # other assembly, both scenes
        rows.append((f"{m}/{s}", "lower bound only", rank_eval(test, fit(test, "none"))))
        rows.append((f"{m}/{s}", "bound x const (in-sample)", rank_eval(test, fit(test, "const"))))
        rows.append((f"{m}/{s}", "bound + linear (in-sample)", rank_eval(test, fit(test, "linear"))))
        if other:
            rows.append((f"{m}/{s}", "bound + linear (fit on OTHER assembly)", rank_eval(test, fit(other, "linear"))))
        rows.append((f"{m}/{s}", "lazy refinement (reference)", lazy_eval(test)))
    for g, name, v in rows:
        print(f"  {g:26s} {name:40s} {v['spearman']:.3f} / {v['top1_within2pct']:.2f} / {v['topk_recall']:.2f} / "
              f"{v['regret']:.2%} / {v['max_regret']:.1%}  (n={v['n']})")
        report["ranking"].setdefault(g, {})[name] = v
    OUT.mkdir(exist_ok=True)
    (OUT / "analyze_penalty.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("->", OUT / "analyze_penalty.json")


if __name__ == "__main__":
    main()
