"""Force closure for a set of unilateral hard contacts (paper2 G 质量的数学核心).

抽自 WRS3 的 ``wrs/assembly/robotics/graspability.py``。原文件其余部分绑定
WRS3 的 Grasp / MechBase / JawView 与 execution.py, 已移出 asmcore, 完整副本见
``E:/Projects/WRS2-assembly-planner/wrs/assembly/robotics/graspability.py``; WRS1 侧的等价实现放在 ``adapters/``。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from .equilibrium import _generators
from ..model import readonly


def check_force_closure(
    points: ArrayLike, normals: ArrayLike, *, friction: float = 0.5, cone_sides: int = 16
) -> dict[str, Any]:
    """6D positive-span LP for unilateral hard contacts at supplied pad sites.

    Finite pad corners supply torsional resistance without inventing a free
    moment variable. This is force closure, not a finite load-capacity test.
    """
    p = readonly(points, shape=(None, 3))
    n = readonly(normals, shape=p.shape)
    if not np.isfinite(friction) or friction < 0 or type(cone_sides) is not int or cone_sides < 4:
        raise ValueError("Invalid friction/cone_sides")
    if not len(p) or np.any(np.linalg.norm(n, axis=1) == 0):
        raise ValueError("Nonempty contacts and nonzero normals required")
    n = n / np.linalg.norm(n, axis=1)[:, None]
    rays = np.concatenate([_generators(v, friction, cone_sides) for v in n])
    center = p.mean(0)
    length = max(float(np.linalg.norm(p - center, axis=1).max()), 1e-9)
    lever = np.repeat((p - center) / length, cone_sides if friction else 1, axis=0)
    W = np.vstack((rays.T, np.cross(lever, rays).T))
    rank = int(np.linalg.matrix_rank(W, tol=1e-10))
    if rank < 6:
        return dict(status="not_force_closed", rank=rank, margin=0.0)
    m = len(rays)
    result = linprog(
        np.r_[np.zeros(m), -1.0],
        A_eq=np.vstack((np.column_stack((W, np.zeros(6))), np.r_[np.ones(m), 0.0])),
        b_eq=np.r_[np.zeros(6), 1.0],
        A_ub=np.column_stack((-np.eye(m), np.ones(m))),
        b_ub=np.zeros(m),
        bounds=(0, None),
        method="highs",
        options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
    )
    if result.status == 2:
        return dict(status="not_force_closed", rank=rank, margin=0.0)
    if not result.success:
        return dict(status="unknown", rank=rank, reason=result.message)
    weights, margin = result.x[:-1], float(result.x[-1])
    residual = max(
        float(np.max(np.abs(W @ weights))),
        abs(float(weights.sum()) - 1),
        float(np.max(margin - weights)),
    )
    if residual > 1e-8:
        return dict(status="unknown", rank=rank, residual=residual)
    return dict(
        status="force_closed" if margin > 1e-8 else "not_force_closed",
        rank=rank,
        margin=margin,
        residual=residual,
        torque_length_m=length,
    )
