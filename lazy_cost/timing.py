"""Time-optimal parameterisation of a joint path under per-joint velocity AND acceleration limits.

Replaces curobo_bridge.assembly_demo.topp for timing planned paths. That function bounds the
tangential acceleration (along the path) and the curvature term separately, each by the joint
limit, so on curved paths a joint may get up to ~2x its acceleration limit and the time can fall
BELOW the straight-line rest-to-rest bound (observed: 0.42 s for a segment whose bound is 0.60 s).

Here, with s the arc length, q' = dq/ds and q'' = d2q/ds2 (finite differences on a dense
resampling), every joint satisfies
        |q'_j sdot|            <= VMAX_j
        |q'_j sddot + q''_j sdot^2| <= AMAX
i.e. the classic TOPP problem in x = sdot^2. Maximum-velocity curve from the velocity limits
and from the feasibility of the sddot interval (bisection), then a forward pass with the
largest admissible sddot and a backward pass with the smallest. Rest at both ends.
For a straight joint line (q'' = 0) the result equals max_j t_trap(|dq_j|), the bound used by
the cost module.
"""
from __future__ import annotations

import numpy as np


def _limits():
    from curobo_bridge.assembly_demo import AMAX, VMAX
    return np.asarray(VMAX, float), float(AMAX)


def _sddot_bounds(qp, qpp, x, A):
    """Per point: [lo, hi] of sddot with |qp_j sddot + qpp_j x| <= A for every joint."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (A - qpp * x[:, None]) / qp
        b = (-A - qpp * x[:, None]) / qp
    up = np.where(qp > 1e-9, a, np.where(qp < -1e-9, b, np.inf))
    lo = np.where(qp > 1e-9, b, np.where(qp < -1e-9, a, -np.inf))
    return lo.max(1), up.min(1)


def topp(path, step=0.01):
    """-> (dense points [M,6], timestamps [M]) of the time-optimal traversal of ``path``."""
    V, A = _limits()
    path = np.asarray(path, float)
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        pts += [a + t * (b - a) for t in np.linspace(0, 1, n + 1)[1:]]
    pts = np.asarray(pts)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.r_[True, ds > 1e-9]
    pts = pts[keep]
    if len(pts) < 2:
        return pts, np.zeros(len(pts))
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    qp = np.gradient(pts, s, axis=0)
    qpp = np.gradient(qp, s, axis=0)
    n = len(pts)
    # maximum velocity curve: velocity limits, then sddot-interval feasibility (bisection)
    with np.errstate(divide="ignore"):
        xv = np.min((V / np.maximum(np.abs(qp), 1e-12)) ** 2, axis=1)
    lo_x, hi_x = np.zeros(n), xv.copy()
    for _ in range(40):
        mid = 0.5 * (lo_x + hi_x)
        l, u = _sddot_bounds(qp, qpp, mid, A)
        ok = l <= u + 1e-12
        lo_x = np.where(ok, mid, lo_x)
        hi_x = np.where(ok, hi_x, mid)
    mvc = lo_x
    x = mvc.copy()
    x[0] = x[-1] = 0.0
    for k in range(n - 1):                                    # forward: accelerate as much as allowed
        _, u = _sddot_bounds(qp[k:k + 1], qpp[k:k + 1], x[k:k + 1], A)
        x[k + 1] = min(x[k + 1], max(0.0, x[k] + 2 * ds[k] * max(u[0], 0.0)))
    for k in range(n - 2, -1, -1):                             # backward: brake as late as allowed
        l, _ = _sddot_bounds(qp[k + 1:k + 2], qpp[k + 1:k + 2], x[k + 1:k + 2], A)
        x[k] = min(x[k], x[k + 1] - 2 * ds[k] * min(l[0], 0.0))
    v = np.sqrt(np.maximum(x, 0.0))
    dt = 2 * ds / np.maximum(v[:-1] + v[1:], 1e-9)
    return pts, np.r_[0.0, np.cumsum(dt)]
