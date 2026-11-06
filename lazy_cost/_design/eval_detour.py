"""Offline evaluation of detour indicators (CPU only).

    python -m lazy_cost._design.eval_detour [--tag _feat3_parts] [--json out.json]

Data: lazy_cost/_output/exp_{rack10,triangular_frame6}{tag}_*.jsonl written by lazy_cost.exp_detour.
Segment truth (free straight line -> LB in every variant):
    R0  blocked -> raw cuRobo (MotionPlanner) path, TOPP          (the spec's reference)
    R1  blocked -> de-taxed: min(R0, TOPP(shortcut), TOPP(blended shortcut))   (no detour-probe input)
    R2  blocked -> min(R1, UB of the detour probe)                (best known; uses UB -> leakage for UB methods)
Estimators for blocked segments (free -> LB); parameters fitted on the OTHER assembly:
    lb, x1.25 (baseline only), nnls (current feature set, refit), shipped (penalty_model.json),
    A    (1+tau) * max(LB, TOPP(escape path))
    B    (1+tau) * T_geo, T_geo = max(LB, UB) if a free curve exists else A's time   (= design E1)
    E2   (1+tau) * (LB + lam (T_geo - LB))
    gbt  gradient boosted trees on all indicators (ratio target), cross-assembly
tau = median(truth/LB - 1) on FREE segments under the same reference processing (fit assembly).
Since free segments' truth is LB by definition, tau is computed from the planner times of free
segments (ref for R0, ref_sc for R1/R2): the planner overhead.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "_output"
MODELS = ("rack10", "triangular_frame6")
K = 3


def load(model, tag):
    fs = sorted(glob.glob(str(OUT / f"exp_{model}{tag}_*.jsonl")))
    if not fs:
        return []
    rows = [json.loads(l) for l in open(fs[-1], encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("seg_data")]


def fin(v):
    return v is not None and np.isfinite(v)


# ---- per-segment truth ---------------------------------------------------------------------
def seg_truth(s, ref):
    if s["how"] == "straight":
        return s["lb"]
    r0 = s["ref"] if fin(s["ref"]) else np.inf
    if ref == "R0":
        return r0
    r1 = s.get("ref_sc", r0)
    r1 = r1 if fin(r1) else np.inf
    if ref == "R1":
        return max(r1, s["lb"])
    d = s.get("detour") or {}
    ub = d.get("ub")
    if fin(ub):
        r1 = min(r1, ub + s["desc"])
    return max(r1, s["lb"])


def planner_overhead(rows, ref):
    """tau: median planner time / LB - 1 on FREE segments (raw for R0, de-taxed for R1/R2)."""
    key = "ref" if ref == "R0" else "ref_sc"
    v = [s[key] / s["lb"] - 1 for r in rows for s in r["seg_data"] if s["how"] == "straight" and fin(s.get(key)) and s["lb"] > 0]
    return float(np.median(v)) if v else 0.0


# ---- indicators ------------------------------------------------------------------------------
def span(s):
    return (s["s_out"] - s["s_in"]) if s.get("s_in") is not None and s.get("s_out") is not None else 0.0


def t_esc(s):
    d = s.get("detour") or {}
    v = [d.get("escp_topp"), d.get("escw_topp")]
    v = [x for x in v if fin(x)]
    return max(s["lb"], min(v) + s["desc"]) if v else s["lb"]


SUB = ("lift_0.1", "escBw_1.5", "lift_0.05", "escBw_1")    # greedy cover of the full UB, chosen on rack10


def t_geo(s, sub=None):
    d = s.get("detour") or {}
    ub = d.get("ub")
    if sub is not None:
        v = [c["topp"] for k, c in d.get("cand", {}).items() if k in sub and c["free"] and "topp" in c]
        ub = min(v) if v else None
    if fin(ub):
        return max(s["lb"], ub + s["desc"])
    return t_esc(s)


FEATS = ("ub_ratio", "any_free", "n_free", "esc_ratio", "escp_conv", "escw_conv", "escp_dq", "escw_dq",
         "depth", "frac", "span", "transit", "lb", "ubtdec_ratio", "esc_tdec_ratio")


def feat_vec(s):
    d = s.get("detour") or {}
    lb = s["lb"]
    ub = d.get("ub")
    return np.array([(ub + s["desc"]) / lb if fin(ub) else 3.0, float(d.get("any_free", False)), d.get("n_free", 0),
                     t_esc(s) / lb, d.get("escp_conv", 0), d.get("escw_conv", 0), d.get("escp_dq", 0), d.get("escw_dq", 0),
                     s["depth"], s["frac"], span(s), float(s["kind"] == "transit"), lb,
                     (d["ub_tdec"] + s["desc"]) / lb if fin(d.get("ub_tdec")) else 3.0,
                     (min(d.get("escp_tdec", 9), d.get("escw_tdec", 9)) + s["desc"]) / lb if d else 1.0])


def phi_cur(s):
    from lazy_cost.penalty import phi
    f = dict(s)
    for k in ("s_in", "s_out", "z_low"):
        if f.get(k) is None:
            f[k] = float("nan")
    return phi(f)


class Estimators:
    def __init__(self, fit_rows, ref):
        from scipy.optimize import nnls
        from sklearn.ensemble import GradientBoostingRegressor
        self.tau = planner_overhead(fit_rows, ref)
        B = [s for r in fit_rows for s in r["seg_data"] if s["how"] == "bound" and fin(seg_truth(s, ref)) and "detour" in s]
        y = np.array([seg_truth(s, ref) / s["lb"] for s in B])
        # current feature model, refit on the fit assembly
        X = np.array([phi_cur(s) for s in B])
        sol, _ = nnls(np.c_[X, np.ones(len(X))], np.maximum(y - 1, 0))
        self.nnls = sol
        self.shipped = json.loads((PKG / "penalty_model.json").read_text(encoding="utf-8"))
        # E2 lambda (least squares on the ratio, with tau fixed)
        g = np.array([t_geo(s) / s["lb"] - 1 for s in B])
        tgt = y / (1 + self.tau) - 1
        self.lam = float(np.clip((g @ tgt) / max(g @ g, 1e-12), 0, 3))
        # E2b: free affine fit  ratio = c0 + c1 * (T_geo/LB - 1)
        A = np.c_[np.ones(len(g)), g]
        self.aff = np.linalg.lstsq(A, y, rcond=None)[0]
        F = np.array([feat_vec(s) for s in B])
        self.gbt = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                             loss="absolute_error", random_state=0).fit(F, y)
        self.n_fit = len(B)

    def predict(self, s, name):
        lb = s["lb"]
        if s["how"] == "straight":
            return lb
        if name == "lb":
            return lb
        if name == "x1.25":
            return 1.25 * lb
        if name == "nnls":
            return lb * (1 + max(0.0, float(phi_cur(s) @ self.nnls[:-1] + self.nnls[-1])))
        if name == "shipped":
            return lb * (1 + max(0.0, float(phi_cur(s) @ np.array(self.shipped["w"]) + self.shipped["b"])))
        if name == "A":
            return (1 + self.tau) * t_esc(s)
        if name == "B":
            return (1 + self.tau) * t_geo(s)
        if name == "Blite":
            return (1 + self.tau) * t_geo(s, SUB)
        if name == "E2":
            return (1 + self.tau) * (lb + self.lam * (t_geo(s) - lb))
        if name == "E2aff":
            return lb * max(1.0, float(self.aff[0] + self.aff[1] * (t_geo(s) / lb - 1)))
        if name == "gbt":
            return lb * max(1.0, float(self.gbt.predict(feat_vec(s)[None])[0]))
        raise KeyError(name)


METHODS = ("lb", "x1.25", "shipped", "nnls", "A", "B", "Blite", "E2", "E2aff", "gbt")


# ---- ranking ---------------------------------------------------------------------------------
def pairs_of(r):
    P = {}
    for s in r["seg_data"]:
        P.setdefault((s["cand"], s["g"]), {})[s["kind"]] = s
    return {k: v for k, v in P.items() if len(v) == 2}


def cand_values(P, val):
    out = {}
    for (x, g), segs in P.items():
        v = sum(val(s) for s in segs.values())
        out[x] = min(out.get(x, np.inf), v)
    return out


def ranking_metrics(pred, truth):
    from lazy_cost.exp_rack10 import metrics
    return metrics(pred, truth, K)


# ---- lazy simulation ---------------------------------------------------------------------------
def simulate(P, truth_seg, est_seg, order):
    """Lazy certification of the top-K (as LazyStepCost.rank): refining a pair reveals the truth
    of its blocked segments. order: 'lb' (current: cheapest lower bound first) or 'est'.
    Returns dict(refined segments, flips of the estimated best, refinements to first correct best,
    wasted refinements, initial-best-ok, anytime regret by budget)."""
    keys = list(P)
    refined = set()
    blocked = {k: [s for s in P[k].values() if s["how"] == "bound"] for k in keys}
    T = {k: sum(truth_seg(s) for s in P[k].values()) for k in keys}
    cands = sorted({x for x, _ in keys})
    tc = {x: min(T[k] for k in keys if k[0] == x) for x in cands}
    finite = [v for v in tc.values() if np.isfinite(v)]
    if len(finite) < 3:
        return None
    best_true = min(finite)

    def lo(k):
        return sum(truth_seg(s) if (k in refined or s["how"] == "straight") else s["lb"] for s in P[k].values())

    def hi(k):
        return T[k] if (k in refined or not blocked[k]) else np.inf

    def est(k):
        return sum(truth_seg(s) if (k in refined or s["how"] == "straight") else est_seg(s) for s in P[k].values())

    def cur_best():
        e = {}
        for k in keys:
            e[k[0]] = min(e.get(k[0], np.inf), est(k))
        return min(e, key=e.get)

    def ok(x):
        return tc[x] <= best_true * 1.02

    n_seg, flips, first_ok, budget_regret = 0, 0, None, {}
    b = cur_best()
    init_ok = ok(b)
    if init_ok:
        first_ok = 0
    budget_regret[0] = tc[b] / best_true - 1
    while True:
        his = sorted(min(hi(k) for k in keys if k[0] == x) for x in cands)
        tau = his[K - 1] if len(his) >= K else np.inf
        todo = [k for k in keys if hi(k) == np.inf and lo(k) < tau and np.isfinite(lo(k))]
        if not todo:
            break
        k = min(todo, key=lo if order == "lb" else est)
        refined.add(k)
        n_seg += len(blocked[k])
        nb = cur_best()
        if nb != b:
            flips += 1
            b = nb
        if first_ok is None and ok(b):
            first_ok = n_seg
        budget_regret[n_seg] = tc[b] / best_true - 1
    final_top = sorted(cands, key=lambda x: min(hi(k) for k in keys if k[0] == x))[:K]
    wasted = sum(len(blocked[k]) for k in refined if k[0] not in final_top)
    return dict(refined=n_seg, flips=flips, first_ok=first_ok if first_ok is not None else n_seg, wasted=wasted,
                init_ok=float(init_ok), budget_regret=budget_regret)


def regret_at(br, budget):
    ks = [k for k in br if k <= budget]
    return br[max(ks)]


# ---- main ------------------------------------------------------------------------------------
def evaluate(tag, refs=("R0", "R1", "R2")):
    data = {m: load(m, tag) for m in MODELS}
    res = {"tag": tag, "contexts": {m: len(v) for m, v in data.items()}}
    # descriptive statistics
    desc = {}
    for m, rows in data.items():
        S = [s for r in rows for s in r["seg_data"]]
        Bk = [s for s in S if s["how"] == "bound"]
        Fr = [s for s in S if s["how"] == "straight"]
        q = lambda v: [float(x) for x in np.percentile(v, [10, 50, 90])] if len(v) else None
        d = dict(n_free=len(Fr), n_blocked=len(Bk),
                 free_ref_over_lb=q([s["ref"] / s["lb"] for s in Fr if fin(s["ref"])]),
                 free_refsc_over_lb=q([s["ref_sc"] / s["lb"] for s in Fr if fin(s.get("ref_sc"))]),
                 blocked_ref_over_lb=q([s["ref"] / s["lb"] for s in Bk if fin(s["ref"])]),
                 blocked_refsc_over_lb=q([max(s["ref_sc"], s["lb"]) / s["lb"] for s in Bk if fin(s.get("ref_sc"))]),
                 blocked_R2_over_lb=q([seg_truth(s, "R2") / s["lb"] for s in Bk if fin(seg_truth(s, "R2"))]),
                 blocked_ref_inf=sum(not fin(s["ref"]) for s in Bk),
                 ub_found=float(np.mean([bool((s.get("detour") or {}).get("any_free")) for s in Bk])) if Bk else None,
                 ub_over_lb=q([(s["detour"]["ub"] + s["desc"]) / s["lb"] for s in Bk if fin((s.get("detour") or {}).get("ub"))]),
                 ub_below_refsc=float(np.mean([(s["detour"]["ub"] + s["desc"]) < s["ref_sc"] - 1e-6 for s in Bk
                                              if fin((s.get("detour") or {}).get("ub")) and fin(s.get("ref_sc"))])) if Bk else None,
                 ub_below_ref=float(np.mean([(s["detour"]["ub"] + s["desc"]) < s["ref"] - 1e-6 for s in Bk
                                            if fin((s.get("detour") or {}).get("ub")) and fin(s["ref"])])) if Bk else None,
                 noise_pairs=[(s["ref"], s["ref2"]) for s in S if "ref2" in s][:50],
                 ub_names={})
        for s in Bk:
            n = (s.get("detour") or {}).get("ub_name")
            if n:
                fam = n.split("_")[0]
                d["ub_names"][fam] = d["ub_names"].get(fam, 0) + 1
        tim = [r["timing"].get("detour") or {} for r in rows]
        cand = [r["timing"]["candidates"] for r in rows]
        for k in ("escape_ms", "detour_check_ms", "tdec_ms", "topp_ub_ms", "topp_esc_ms", "total_ms", "segments", "candidates"):
            d[f"detour_{k}_per_ctx"] = float(np.mean([t.get(k, 0) for t in tim]))
        d["bounds_ms_per_ctx"] = float(np.mean([r["timing"]["bounds_ms"] for r in rows]))
        d["lazy_ms_per_ctx"] = float(np.mean([r["timing"]["lazy_ms"] for r in rows]))
        d["refined_segments_per_ctx"] = float(np.mean([r["timing"]["refined_segments"] for r in rows]))
        d["ms_per_refined_segment"] = float(np.sum([r["timing"]["stage_ms"]["refine_ms"] for r in rows]) /
                                            max(1, np.sum([r["timing"]["refined_segments"] for r in rows])))
        d["candidates_per_ctx"] = float(np.mean(cand))
        desc[m] = d
    res["describe"] = desc
    for ref in refs:
        R = {}
        for test in MODELS:
            fit = [m for m in MODELS if m != test][0]
            if not data[test] or not data[fit]:
                continue
            E = Estimators(data[fit], ref)
            out = dict(tau=E.tau, lam=E.lam, aff=E.aff.tolist(), n_fit=E.n_fit, nnls=E.nnls.tolist(), seg={}, rank={}, lazy={})
            Bk = [s for r in data[test] for s in r["seg_data"] if s["how"] == "bound" and fin(seg_truth(s, ref))]
            yt = np.array([seg_truth(s, ref) for s in Bk])
            for name in METHODS:
                p = np.array([E.predict(s, name) for s in Bk])
                le = np.abs(np.log(p / yt))
                out["seg"][name] = dict(n=len(Bk), mean_abs_log=float(le.mean()), median=float(np.median(le)),
                                        p90=float(np.percentile(le, 90)),
                                        spearman=float(spearmanr(p / np.array([s["lb"] for s in Bk]),
                                                                 yt / np.array([s["lb"] for s in Bk]))[0]) if np.ptp(p) > 0 else 0.0,
                                        bias=float(np.mean(np.log(p / yt))))
            # ranking per context
            per = {name: [] for name in METHODS}
            sims = {name: [] for name in METHODS + ("lb_order",)}
            for r in data[test]:
                P = pairs_of(r)
                tv = lambda s: seg_truth(s, ref)
                truth = cand_values(P, tv)
                for name in METHODS:
                    pred = cand_values(P, lambda s: E.predict(s, name))
                    mm = ranking_metrics(pred, truth)
                    if mm:
                        per[name].append(mm)
                    sm = simulate(P, tv, lambda s: E.predict(s, name), "est")
                    if sm:
                        sims[name].append(sm)
                sm = simulate(P, tv, lambda s: s["lb"], "lb")
                if sm:
                    sims["lb_order"].append(sm)
            for name, L in per.items():
                if not L:
                    continue
                f = lambda k: float(np.mean([x[k] for x in L]))
                out["rank"][name] = dict(n=len(L), spearman=f("spearman"), top1_2pct=f("top1_within2pct"), top3=f("topk_recall"),
                                         regret=f("regret"), max_regret=float(max(x["regret"] for x in L)),
                                         regrets=[x["regret"] for x in L])
            base = out["rank"].get("shipped", {}).get("regrets")
            for name in out["rank"]:
                rg = out["rank"][name]["regrets"]
                try:
                    out["rank"][name]["wilcoxon_vs_shipped_p"] = float(wilcoxon(rg, base).pvalue) if base and np.any(np.array(rg) != np.array(base)) else 1.0
                except ValueError:
                    out["rank"][name]["wilcoxon_vs_shipped_p"] = 1.0
            for name, L in sims.items():
                if not L:
                    continue
                f = lambda k: float(np.mean([x[k] for x in L]))
                out["lazy"][name] = dict(n=len(L), refined=f("refined"), flips=f("flips"), first_ok=f("first_ok"),
                                         wasted=f("wasted"), init_ok=f("init_ok"),
                                         **{f"regret_at_{b}": float(np.mean([regret_at(x["budget_regret"], b) for x in L])) for b in (0, 1, 2, 4, 8)})
            for name in out["rank"]:
                out["rank"][name].pop("regrets")
            R[test] = out
        res[ref] = R
    return res


def show(res):
    print("contexts", res["contexts"])
    for m, d in res["describe"].items():
        print(f"\n== {m}: " + json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in d.items()
                                           if k != "noise_pairs"}))
    for ref in ("R0", "R1", "R2"):
        for test, o in res.get(ref, {}).items():
            print(f"\n### truth {ref}, test {test} (fit on the other): tau {o['tau']:.3f} lam {o['lam']:.2f} aff {np.round(o['aff'], 3)}")
            print(f"{'method':8s} {'seg|log|':>8s} {'p90':>6s} {'bias':>6s} {'rho':>5s} | {'top1<2%':>7s} {'top3':>5s} {'regret':>7s} {'max':>6s} {'p':>5s} | "
                  f"{'refined':>7s} {'flips':>5s} {'1st ok':>6s} {'wasted':>6s} {'r@0':>6s} {'r@2':>6s} {'r@4':>6s}")
            for name in METHODS + ("lb_order",):
                sg, rk, lz = o["seg"].get(name), o["rank"].get(name), o["lazy"].get(name)
                a = f"{sg['mean_abs_log']:8.3f} {sg['p90']:6.3f} {sg['bias']:+6.3f} {sg['spearman']:5.2f}" if sg else " " * 29
                b = (f"{rk['top1_2pct']:7.2f} {rk['top3']:5.2f} {100 * rk['regret']:6.2f}% {100 * rk['max_regret']:5.1f}% {rk['wilcoxon_vs_shipped_p']:5.2f}"
                     if rk else " " * 40)
                c = (f"{lz['refined']:7.2f} {lz['flips']:5.2f} {lz['first_ok']:6.2f} {lz['wasted']:6.2f} {100 * lz['regret_at_0']:5.2f}% "
                     f"{100 * lz['regret_at_2']:5.2f}% {100 * lz['regret_at_4']:5.2f}%" if lz else "")
                print(f"{name:8s} {a} | {b} | {c}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_feat3_parts")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = evaluate(a.tag)
    show(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
