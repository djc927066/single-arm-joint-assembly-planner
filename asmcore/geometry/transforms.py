"""Rigid rotation conventions shared by geometry inputs and migration tools."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..model import FloatArray


def rotation_xyz(rotation_rad: ArrayLike) -> FloatArray:
    """Extrinsic xyz rotation in radians, applied as Rz @ Ry @ Rx."""
    r = np.asarray(rotation_rad, dtype=float)
    if r.shape != (3,) or not np.all(np.isfinite(r)):
        raise ValueError("Rotation must contain three finite radians")
    cx, cy, cz = np.cos(r)
    sx, sy, sz = np.sin(r)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx
