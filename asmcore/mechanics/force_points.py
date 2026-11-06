"""Force-site preparation, planar hulls and deterministic curved subsets.

All selections retain input positions/normals and stay inside one patch/capacity
group. Numerical welding is limited to machine-roundoff positions with exactly
equal normals. Geometry is cached by immutable patch identity, not part name.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from ..model import ContactPatch, freeze, readonly


@dataclass(frozen=True)
class ForcePoints:
    points: np.ndarray
    normals: np.ndarray
    info: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", readonly(self.points))
        object.__setattr__(self, "normals", readonly(self.normals))
        object.__setattr__(self, "info", freeze(self.info))


_CACHE = WeakKeyDictionary()
_LOCK = RLock()
_MAX_PATCHES = 128
_MAX_POLICIES = 8


def _roundoff_length(points):
    return 64 * np.finfo(float).eps * max(0.001, float(np.max(np.abs(points), initial=0)))


def _clean(patch):
    normals = patch.normals_a_world
    cells = [c for r in patch.regions for c in r.cells_world_m if len(c)]
    if patch.dimension > 0 and len(normals) and np.all(normals == normals[0]) and cells:
        points = np.concatenate(cells)
        normals = np.broadcast_to(normals[0], points.shape)
    else:
        points = (patch.points_a_world_m + patch.points_b_world_m) / 2
    joined = np.unique(np.column_stack((points, normals)), axis=0)
    p, n = joined[:, :3], joined[:, 3:]
    exact_count = len(p)
    tol = _roundoff_length(p)
    if len(p) > 1:
        # Bounded, vectorized bins: two points sharing a bin differ by at most
        # tol in Euclidean distance. Choose an actual input, never average it.
        # Include exact normals in the key: coincident sharp edges stay distinct.
        bins = np.rint((p - p[0]) / (tol / np.sqrt(3)))
        _, ids = np.unique(np.column_stack((bins, n)), axis=0, return_index=True)
        ids.sort()
        p, n = p[ids], n[ids]
    return ForcePoints(
        p,
        n,
        dict(
            input_points=len(points),
            exact_unique_points=exact_count,
            clean_points=len(p),
            roundoff_weld_tolerance_m=tol,
            method="clean_only",
            curved_subset=False,
        ),
    )


def _planar_selection(p, n, dimension, tol):
    if len(p) < 2 or not np.all(n == n[0]):
        return None
    centered = p - p[0]
    if dimension == 1:
        far = int(np.argmax(np.einsum("ij,ij->i", centered, centered)))
        direction = centered[far]
        length = np.linalg.norm(direction)
        if length == 0:
            return np.array([0]), "point"
        direction = direction / length
        coordinate = centered @ direction
        if np.max(np.linalg.norm(centered - coordinate[:, None] * direction, axis=1)) > tol:
            return None  # a polyline is not a straight segment
        return np.unique([np.argmin(coordinate), np.argmax(coordinate)]), "line_endpoints"
    if dimension != 2 or np.max(np.abs(centered @ n[0])) > tol:
        return None
    if len(p) <= 3:
        return np.arange(len(p)), "small_planar"
    axis = np.eye(3)[np.argmin(np.abs(n[0]))]
    tangent = np.cross(n[0], axis)
    tangent /= np.linalg.norm(tangent)
    xy = centered @ np.array([tangent, np.cross(n[0], tangent)]).T
    try:
        hull = ConvexHull(xy)
    except QhullError:
        return np.arange(len(p)), "planar_hull_unresolved"
    # A numerical hull failure must not discard an outside point.
    if np.max(xy @ hull.equations[:, :2].T + hull.equations[:, 2]) > tol:
        return np.arange(len(p)), "planar_hull_unresolved"
    return np.sort(hull.vertices), "planar_hull"


def _curved_selection(p, n, patch, budget, spacing, normal_angle):
    count = len(p)
    if count <= budget:
        return np.arange(count), dict(method="small_curved", curved_subset=False)
    # Spatial distance plus normal chord distance. Fixed sorting and argmax
    # tie-breaking make this reproducible; no RNG and no N-by-N matrix.
    features = np.column_stack(((p - p[0]) / spacing, n / (2 * np.sin(normal_angle / 2))))
    boundary = [loop for r in patch.regions for loop in r.boundary_loops_world_m if len(loop)]
    priority = np.zeros(count, dtype=bool)
    if boundary:
        ids = cKDTree(p).query(np.concatenate(boundary))[1]
        priority[ids] = True
    extrema = np.r_[
        np.argmin(p, axis=0), np.argmax(p, axis=0), np.argmin(n, axis=0), np.argmax(n, axis=0)
    ]
    priority[extrema] = True
    best = np.full(count, np.inf)
    selected = np.zeros(count, dtype=bool)
    indices = []
    for _ in range(min(count, budget)):
        candidates = priority & ~selected & (best > 1.0)
        score = np.where(candidates, best, -1.0) if np.any(candidates) else best
        index = int(np.argmax(score))
        if len(indices) >= 4 and best[index] <= 1.0:
            break
        indices.append(index)
        selected[index] = True
        delta = features - features[index]
        np.minimum(best, np.einsum("ij,ij->i", delta, delta), out=best)
        best[selected] = -1.0
    return np.sort(indices), dict(
        method="curved_fps",
        curved_subset=len(indices) < count,
        boundary_priority_points=int(np.count_nonzero(priority)),
        max_normalized_cover_distance=float(np.sqrt(max(0.0, np.max(best)))),
        coverage_target_met=bool(np.max(best) <= 1.0),
    )


def prepare_force_points(
    patch: ContactPatch,
    *,
    reduce: bool = True,
    curved_budget: int = 64,
    spacing: float = 0.005,
    normal_angle: float = np.pi / 18,
    full_curves: bool = False,
) -> tuple[ForcePoints, bool]:
    key = (bool(reduce), curved_budget, spacing, normal_angle, bool(full_curves))
    with _LOCK:
        entry = _CACHE.get(patch)
        if entry is not None and key in entry:
            return entry[key], True
        clean = entry.get("clean") if entry is not None else None
    if clean is None:
        clean = _clean(patch)
    p, n = clean.points, clean.normals
    info = dict(clean.info)
    if reduce and len(p):
        selection = _planar_selection(p, n, patch.dimension, info["roundoff_weld_tolerance_m"])
        if selection is not None:
            ids, method = selection
            p, n = p[ids], n[ids]
            info["method"] = method
        elif not full_curves and patch.dimension > 0:
            ids, details = _curved_selection(p, n, patch, curved_budget, spacing, normal_angle)
            p, n = p[ids], n[ids]
            info.update(details)
        elif full_curves and patch.dimension > 0:
            info["method"] = "full_curved"
    info["selected_points"] = len(p)
    result = ForcePoints(p, n, info)
    with _LOCK:
        if patch not in _CACHE:
            if len(_CACHE) >= _MAX_PATCHES:
                del _CACHE[next(iter(_CACHE))]
            _CACHE[patch] = {"clean": clean}
        entry = _CACHE[patch]
        if len(entry) > _MAX_POLICIES:
            entry.clear()
            entry["clean"] = clean
        entry[key] = result
    return result, False
