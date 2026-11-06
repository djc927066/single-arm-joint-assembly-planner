"""Blocked-segment correction used online: estimate = lower bound * (1 + e(features)).

    & $py -X utf8 -m lazy_cost.penalty            # (re)fit from the parts-only feature runs

e = w . phi + b, w >= 0 (non-negative least squares) on the blocked segments of
_output/exp_*_feat2_parts_*.jsonl (both assemblies, parts as the only obstacles), target
real / lower bound - 1 from full cuRobo planning. phi: share of colliding states, length of
the colliding stretch, max and mean penetration (m), held part involved, picked part
involved, 1 / height of the deepest colliding sphere (see checker._blocked_features).
The estimate is NOT a lower bound: it is used to order the search, never to prune.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
MODEL = PKG / "penalty_model.json"
NAMES = ("frac", "span", "depth", "depth_mean", "held", "part", "z_low_inv")


def phi(f) -> np.ndarray:
    s_in, s_out, z = f.get("s_in"), f.get("s_out"), f.get("z_low")
    span = 0.0 if s_in is None or s_out is None or s_in != s_in or s_out != s_out else s_out - s_in
    zi = 0.0 if z is None or z != z else 1.0 / max(z, 0.02)
    return np.array([f["frac"], span, f["depth"], f["depth_mean"], float(f["by_held"] > 0), float(f["by_part"] > 0), zi])


class Penalty:
    """form "relative" (penalty_model.json): lb * (1 + w . phi + b).
    form "additive" (penalty_depth.json): lb + c * max penetration depth [m]; c = extra seconds per
    metre of clearance the arm must make (fitted on the de-taxed reference, ~4.6-4.8 s/m on both
    assemblies with parts as the only obstacles; lazy_cost/_design/eval_slack.py)."""

    def __init__(self, path=MODEL):
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        self.form = m.get("form", "relative")
        self.name = Path(path).stem
        if self.form == "additive":
            self.c = float(m["c"])
        else:
            self.w, self.b = np.array(m["w"]), float(m["b"])

    def factor(self, f) -> float:
        """1 + e for one blocked segment's feature dict (relative form)."""
        return 1.0 + max(0.0, float(phi(f) @ self.w + self.b))

    def estimate(self, lb, blocked, f) -> float:
        if not blocked:
            return lb
        if self.form == "additive":
            return lb + self.c * float(f["depth"])
        return lb * self.factor(f)


def fit(files=None):
    from scipy.optimize import nnls
    files = files or sorted(glob.glob(str(PKG / "_output" / "exp_*_feat2_parts_*.jsonl")))
    X, y = [], []
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            for s in r.get("seg_data") or []:
                if s["how"] != "bound" or not s.get("ref") or not np.isfinite(s["ref"]) or s["lb"] <= 0:
                    continue
                X.append(phi(s))
                y.append(max(s["ref"] / s["lb"] - 1.0, 0.0))
    X, y = np.array(X), np.array(y)
    sol, _ = nnls(np.c_[X, np.ones(len(X))], y)
    model = dict(w=sol[:-1].tolist(), b=float(sol[-1]), features=list(NAMES), n=int(len(y)),
                 files=[Path(f).name for f in files])
    MODEL.write_text(json.dumps(model, indent=2), encoding="utf-8")
    pred = np.c_[X, np.ones(len(X))] @ sol
    print(f"fitted on {len(y)} blocked segments; mean |error| of the excess {np.mean(np.abs(pred - y)):.3f} "
          f"(plain bound {np.mean(y):.3f}); weights:", dict(zip(NAMES, np.round(sol[:-1], 3))), "b", round(sol[-1], 4))
    return model


if __name__ == "__main__":
    fit()
