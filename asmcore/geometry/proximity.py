"""Bounded CPU surface-distance queries and explicit solid-overlap diagnostics."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..model import (
    GeometryConfig,
    MeshData,
    Part,
    PreparedMesh,
    SurfacePatch,
    checked_tf,
    digest,
    readonly,
)
from ._triangle_batch import triangle_pairs
from .mesh_bvh import (
    BudgetExceeded,
    MeshBVH,
    PlacedBVH,
    QueryBudget,
    aabb_distance,
    node_pairs,
    triangle_pair,
)
from .preprocess import prepare_mesh
from .surfaces import extract_surfaces


@dataclass(frozen=True)
class DistanceQuery:
    points_local_m: np.ndarray
    distances_m: np.ndarray
    face_ids: np.ndarray
    completed_count: int
    status: str
    triangle_tests: int

    def __post_init__(self) -> None:
        for key in ("points_local_m", "distances_m", "face_ids"):
            object.__setattr__(
                self, key, readonly(getattr(self, key), np.int64 if key == "face_ids" else float)
            )


@dataclass(frozen=True)
class PairDistance:
    lower_bound_m: float
    upper_bound_m: float | None
    point_a_world_m: tuple | None
    point_b_world_m: tuple | None
    face_ids: tuple[int, int] | None
    status: str
    triangle_tests: int


@dataclass(frozen=True)
class OverlapResult:
    status: str
    reason: str
    witness_world_m: tuple | None = None
    triangle_tests: int = 0


class ProximityBackend(Protocol):
    geometry_config: GeometryConfig
    tol: float
    cache_hits: int

    def prepare(self, mesh: MeshData) -> tuple[PreparedMesh, Sequence[SurfacePatch]]: ...
    def index(self, mesh: MeshData) -> MeshBVH: ...
    def closest_points(
        self, mesh: MeshData, points_local_m: ArrayLike, *, budget: QueryBudget | int | None = None
    ) -> DistanceQuery: ...
    def pair_distance(
        self,
        part_a: Part,
        tf_a: ArrayLike,
        part_b: Part,
        tf_b: ArrayLike,
        *,
        budget: QueryBudget | int | None = None,
    ) -> PairDistance: ...
    def classify_overlap(
        self,
        part_a: Part,
        tf_a: ArrayLike,
        part_b: Part,
        tf_b: ArrayLike,
        *,
        budget: QueryBudget | int | None = None,
    ) -> OverlapResult: ...

    def point_location(self, placed: PlacedBVH, point: np.ndarray, budget: QueryBudget) -> str: ...


def _budget(value: QueryBudget | int | None) -> QueryBudget:
    return (
        value if isinstance(value, QueryBudget) else QueryBudget(300000 if value is None else value)
    )


class MeshProximity:
    """Reusable bounded caches of local BVHs/preprocessing, never of world poses."""

    def __init__(
        self,
        *,
        geometry_config: GeometryConfig | None = None,
        numerical_tol_m: float = 1e-10,
        cache_size: int = 32,
    ) -> None:
        self.geometry_config = geometry_config or GeometryConfig()
        if not np.isfinite(numerical_tol_m) or numerical_tol_m <= 0 or cache_size < 1:
            raise ValueError("Positive numerical tolerance and cache size required")
        self.tol, self.cache_size = numerical_tol_m, cache_size
        self._trees, self._prepared = OrderedDict(), OrderedDict()
        self.cache_hits = 0

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def prepare(self, mesh: MeshData) -> tuple[PreparedMesh, Sequence[SurfacePatch]]:
        """Return cached immutable preprocessing and patches for this config."""
        key = (mesh.geometry_id, digest(self.geometry_config))
        if key in self._prepared:
            self.cache_hits += 1
            self._prepared.move_to_end(key)
            return self._prepared[key]
        prep = prepare_mesh(mesh, config=self.geometry_config)
        return self._remember(
            self._prepared, key, (prep, extract_surfaces(prep, config=self.geometry_config))
        )

    def index(self, mesh: MeshData) -> MeshBVH:
        """Return a local BVH shared by all rigid instances of the same mesh."""
        if mesh.geometry_id in self._trees:
            self.cache_hits += 1
            self._trees.move_to_end(mesh.geometry_id)
            return self._trees[mesh.geometry_id]
        return self._remember(self._trees, mesh.geometry_id, MeshBVH(mesh))

    def closest_points(
        self, mesh: MeshData, points_local_m: ArrayLike, *, budget: QueryBudget | int | None = None
    ) -> DistanceQuery:
        query = readonly(points_local_m, shape=(None, 3))
        budget = _budget(budget)
        used = budget.used
        placed = self.index(mesh).placed(np.eye(4))
        points, distances, ids = [], [], []
        status = "complete"
        try:
            for point in query:
                d, q, fid = placed.closest(point, budget)
                points.append(q)
                distances.append(d)
                ids.append(fid)
        except BudgetExceeded:
            status = "budget_exhausted"
        return DistanceQuery(
            np.asarray(points).reshape(-1, 3),
            np.asarray(distances),
            np.asarray(ids, dtype=int),
            len(ids),
            status,
            budget.used - used,
        )

    def pair_distance(
        self,
        part_a: Part,
        tf_a: ArrayLike,
        part_b: Part,
        tf_b: ArrayLike,
        *,
        budget: QueryBudget | int | None = None,
    ) -> PairDistance:
        budget = _budget(budget)
        used = budget.used
        a = self.index(part_a.geometry).placed(checked_tf(tf_a))
        b = self.index(part_b.geometry).placed(checked_tf(tf_b))
        best, witness, ids, lower, status = np.inf, None, None, 0.0, "complete"
        try:
            for lower, ia, ib in node_pairs(a, b):
                if lower >= best:
                    break
                i, j = np.repeat(ia, len(ib)), np.tile(ib, len(ia))
                bounds = np.linalg.norm(
                    np.maximum(np.maximum(a.tri_lo[i] - b.tri_hi[j], b.tri_lo[j] - a.tri_hi[i]), 0),
                    axis=1,
                )
                keep = bounds < best
                i, j = i[keep], j[keep]
                if not len(i):
                    continue
                if len(i) > budget.remaining:
                    # A tiny remaining budget can still finish a separated box
                    # in one witness. Preserve scalar pruning and partial bounds.
                    for x, y, bound in zip(i, j, bounds[keep]):
                        if bound >= best:
                            continue
                        budget.consume()
                        pa, pb, _ = triangle_pair(a.triangles[x], b.triangles[y], self.tol)
                        distance = np.linalg.norm(pa - pb)
                        if distance < best:
                            best, witness, ids = distance, (pa, pb), (int(x), int(y))
                    continue
                budget.consume(len(i))
                pa, pb = triangle_pairs(a.triangles[i], b.triangles[j], self.tol)
                distances = np.linalg.norm(pa - pb, axis=1)
                k = np.argmin(distances)
                if distances[k] < best:
                    best, witness, ids = distances[k], (pa[k], pb[k]), (int(i[k]), int(j[k]))
                if best == 0:
                    break
        except BudgetExceeded:
            status = "budget_exhausted"
        upper = None if witness is None else float(best)
        lb = (upper or 0.0) if status == "complete" else min(lower, best)
        return PairDistance(
            float(lb),
            upper,
            None if witness is None else tuple(witness[0]),
            None if witness is None else tuple(witness[1]),
            ids,
            status,
            budget.used - used,
        )

    def point_location(self, placed: PlacedBVH, point: np.ndarray, budget: QueryBudget) -> str:
        """inside/outside/boundary/unknown via oriented solid angle for valid solids."""
        if aabb_distance(point, point, placed.lo[0], placed.hi[0]) > self.tol:
            return "outside"
        d, _, _ = placed.closest(point, budget)
        if d <= self.tol:
            return "boundary"
        budget.consume(len(placed.triangles))
        a, b, c = np.moveaxis(placed.triangles - point, 1, 0)
        la, lb, lc = [np.linalg.norm(p, axis=1) for p in (a, b, c)]
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        winding = abs(np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi))
        if winding < 1e-6:
            return "outside"
        if abs(winding - 1) < 1e-6:
            return "inside"
        return "unknown"

    def classify_overlap(
        self,
        part_a: Part,
        tf_a: ArrayLike,
        part_b: Part,
        tf_b: ArrayLike,
        *,
        budget: QueryBudget | int | None = None,
    ) -> OverlapResult:
        """Check crossings and containment; open or ambiguous solids stay unknown.

        Closed-manifold winding is a prerequisite, not a claim to repair all
        self-intersecting CAD. Proper crossings have explicit witnesses; tangent
        cases are checked using vertices and small inward probes of surface cells.
        """
        budget = _budget(budget)
        used = budget.used
        prep_a, _ = self.prepare(part_a.geometry)
        prep_b, _ = self.prepare(part_b.geometry)
        a = self.index(prep_a.mesh).placed(checked_tf(tf_a))
        b = self.index(prep_b.mesh).placed(checked_tf(tf_b))
        hit = None
        solid = (
            prep_a.is_closed
            and prep_b.is_closed
            and prep_a.orientation_reliable
            and prep_b.orientation_reliable
        )
        try:
            if aabb_distance(a.lo[0], a.hi[0], b.lo[0], b.hi[0]) > self.tol:
                return OverlapResult(
                    "separated", "disjoint_aabb", triangle_tests=budget.used - used
                )
            for _, ia, ib in node_pairs(a, b, self.tol):
                for i in ia:
                    for j in ib:
                        ta, tb = a.triangles[i], b.triangles[j]
                        if aabb_distance(ta.min(0), ta.max(0), tb.min(0), tb.max(0)) > self.tol:
                            continue
                        budget.consume()
                        pa, pb, transverse = triangle_pair(ta, tb, self.tol)
                        if np.linalg.norm(pa - pb) <= self.tol:
                            hit = tuple(pa)
                            if transverse:
                                return OverlapResult(
                                    "penetrating" if solid else "unknown",
                                    "transverse_surface_crossing",
                                    hit,
                                    budget.used - used,
                                )
            if not solid:
                return OverlapResult(
                    "unknown", "solid_interior_not_defined", hit, budget.used - used
                )
            for source, target, prep, tf in ((a, b, prep_a, tf_a), (b, a, prep_b, tf_b)):
                # Disconnected components need their own containment witness.
                sample_ids = sorted(set(int(prep.mesh.faces[c[0], 0]) for c in prep.components))
                probes = [source.vertices[i] for i in sample_ids]
                if hit is not None:
                    step = max(self.tol * 8, np.linalg.norm(source.hi[0] - source.lo[0]) * 1e-7)
                    normals = prep.normals @ np.asarray(tf)[:3, :3].T
                    probes.extend(source.triangles.mean(axis=1) - normals * step)
                    probes.extend(source.vertices)
                for p in probes:
                    where = self.point_location(target, p, budget)
                    if where == "unknown":
                        return OverlapResult(
                            "unknown", "ambiguous_winding", tuple(p), budget.used - used
                        )
                    if where == "inside" and (
                        hit is None or self.point_location(source, p, budget) != "outside"
                    ):
                        return OverlapResult(
                            "penetrating", "interior_containment", tuple(p), budget.used - used
                        )
            return OverlapResult(
                "touching" if hit is not None else "separated",
                "mesh_boundary_checks" if hit is not None else "no_crossing_or_containment",
                hit,
                budget.used - used,
            )
        except BudgetExceeded:
            return OverlapResult("unknown", "budget_exhausted", hit, budget.used - used)
