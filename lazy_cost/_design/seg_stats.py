"""CPU-only: ref/LB on free vs blocked segments in the feat2_parts runs (design notes)."""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
OUT = Path(__file__).resolve().parents[1] / "_output"
for fn in sorted(glob.glob(str(OUT / "exp_*_feat2_parts_*.jsonl"))):
    rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
    S = [s for r in rows for s in (r.get("seg_data") or []) if s.get("ref") and np.isfinite(s["ref"]) and s["lb"] > 0]
    print(Path(fn).name, "segments", len(S))
    for how in ("straight", "bound"):
        for kind in ("transit", "transfer"):
            q = np.array([s["ref"] / s["lb"] for s in S if s["how"] == how and s["kind"] == kind])
            lb = np.array([s["lb"] for s in S if s["how"] == how and s["kind"] == kind])
            if len(q) == 0: continue
            p = np.percentile(q, [10, 50, 75, 90, 100])
            add = np.array([s["ref"] - s["lb"] for s in S if s["how"] == how and s["kind"] == kind])
            print(f"  {how:8s} {kind:8s} n={len(q):4d} ref/LB p10 {p[0]:.3f} p50 {p[1]:.3f} p75 {p[2]:.3f} p90 {p[3]:.3f} max {p[4]:.2f}"
                  f" | ref-LB p50 {np.median(add):.3f}s | spearman(ratio, LB) {spearmanr(q, lb)[0]:+.2f} spearman(ref-LB, LB) {spearmanr(add, lb)[0]:+.2f}")
    B = [s for s in S if s["how"] == "bound"]
    y = np.array([s["ref"] / s["lb"] for s in B]); d = np.array([s["depth"] for s in B]); fr = np.array([s["frac"] for s in B])
    print("  blocked: spearman(ratio, depth) %+.2f, (ratio, frac) %+.2f" % (spearmanr(y, d)[0], spearmanr(y, fr)[0]))
    # same (cand,g) row: compare blocked transit ratio with straight transfer ratio etc. -> per-context cuRobo tax
    taxes = []
    for r in rows:
        f = [s["ref"] / s["lb"] for s in (r.get("seg_data") or []) if s["how"] == "straight" and s.get("ref") and s["lb"] > 0.3]
        b = [s["ref"] / s["lb"] for s in (r.get("seg_data") or []) if s["how"] == "bound" and s.get("ref") and s["lb"] > 0.3]
        if f and b: taxes.append((np.median(f), np.median(b)))
    t = np.array(taxes)
    if len(t): print("  per-context median ratio free vs blocked: corr %.2f, mean free %.3f mean blocked %.3f" % (np.corrcoef(t.T)[0, 1], t[:, 0].mean(), t[:, 1].mean()))
