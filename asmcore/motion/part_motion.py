"""Bounded rigid-object planning with independently replayable edge certificates.

The default search is deterministic. A failed finite search is not a proof that
no removal exists. Geometry is always the original mesh, never a collision hull.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation, Slerp

from ..geometry.mesh_bvh import QueryBudget, aabb_distance
from ..geometry.proximity import MeshProximity, ProximityBackend
from ..model import (
    Assembly,
    AssemblyState,
    FloatArray,
    checked_tf,
    digest,
    freeze,
    readonly,
)

if TYPE_CHECKING:
    from ..contact.graph import ContactGraph


@dataclass(frozen=True)
class MotionConfig:
    clearance_m: float = 0.0002
    outside_margin_m: float = 0.02
    min_step_m: float = 1e-5
    max_translation_step_m: float = 0.01
    max_rotation_step_rad: float = 0.05
    numerical_tol_m: float = 1e-9
    max_nodes: int = 512
    max_candidates: int = 32
    max_triangle_tests: int = 500000
    time_limit_s: float = 20.0
    search_se3: bool = True
    detour_scale: float = 0.3
    rotation_angle_rad: float = np.pi / 4

    def __post_init__(self) -> None:
        for k in (
            "clearance_m",
            "outside_margin_m",
            "min_step_m",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "numerical_tol_m",
            "time_limit_s",
            "detour_scale",
            "rotation_angle_rad",
        ):
            v = getattr(self, k)
            if not np.isfinite(v) or v < 0 or (k != "clearance_m" and v == 0):
                raise ValueError(f"{k} must be finite and positive (clearance may be zero)")
        for k in ("max_nodes", "max_candidates", "max_triangle_tests"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k + " must be a positive integer")
        if type(self.search_se3) is not bool or self.rotation_angle_rad > np.pi:
            raise ValueError("Invalid SE(3) search configuration")


@dataclass(frozen=True, eq=False)
class RemovalAction:
    moving_part_ids: tuple[str, ...]
    primitive: str = "rigid"

    def __post_init__(self) -> None:
        ids = (
            (self.moving_part_ids,)
            if isinstance(self.moving_part_ids, str)
            else tuple(self.moving_part_ids)
        )
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Unique moving part IDs required")
        object.__setattr__(self, "moving_part_ids", ids)


def _binding(assembly, state, moving):
    parts = {p.part_id: p for p in assembly.parts}
    if moving not in state.poses or any(k not in parts for k in state.poses):
        raise ValueError("Invalid present/moving IDs")
    return digest(
        (
            state.world_revision,
            tuple(
                (k, parts[k].geometry.geometry_id, None if k == moving else state.poses[k])
                for k in sorted(state.poses)
            ),
        )
    )


def _world(part, tf):
    return part.geometry.vertices @ tf[:3, :3].T + tf[:3, 3]


@dataclass(frozen=True, eq=False)
class ContactPolicy:
    """Scoped contact permission derived from named planar surface pairs.

    Only a separating plane that bounds BOTH complete meshes may certify a
    zero-gap translation. All supporting faces must belong to those surfaces;
    other orientations/regions get ordinary mesh validation, never exclusion.
    """

    moving_part_id: str
    input_binding: str
    reference_rotation: np.ndarray
    planes: tuple = ()
    phase: str = "assembly_transfer"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_rotation", readonly(self.reference_rotation, shape=(3, 3))
        )
        object.__setattr__(self, "planes", freeze(self.planes))

    @classmethod
    def from_graph(
        cls,
        assembly: Assembly,
        state: AssemblyState,
        moving_part_id: str,
        graph: ContactGraph,
        *,
        backend: ProximityBackend | None = None,
        tolerance_m: float = 1e-9,
    ) -> ContactPolicy:
        graph.assert_matches(assembly, state)
        backend = backend or MeshProximity()
        parts = {p.part_id: p for p in assembly.parts}
        planes = []
        for edge in graph.edges:
            if moving_part_id not in edge.part_ids:
                continue
            for patch in edge.patches:
                if (
                    patch.classification != "active"
                    or patch.quality not in ("analytic", "bounded")
                    or patch.provenance.get("idealize_contact", False)
                    or not len(patch.normals_a_world)
                ):
                    continue
                moving_b = patch.part_b == moving_part_id
                other = patch.part_a if moving_b else patch.part_b
                n = patch.normals_a_world if moving_b else patch.normals_b_world
                if not np.all(n == n[0]):
                    continue
                n = n[0]
                fixed_p = patch.points_a_world_m if moving_b else patch.points_b_world_m
                h = float(fixed_p[0] @ n)
                pa, pb = parts[other], parts[moving_part_id]
                va, vb = _world(pa, state.poses[other]), _world(pb, state.poses[moving_part_id])
                if np.max(va @ n - h) > tolerance_m or np.min(vb @ n - h) < -tolerance_m:
                    continue
                surface_ids = (
                    (patch.surface_a_id, patch.surface_b_id)
                    if moving_b
                    else (patch.surface_b_id, patch.surface_a_id)
                )
                valid = True
                for part, tf, sid in (
                    (pa, state.poses[other], surface_ids[0]),
                    (pb, state.poses[moving_part_id], surface_ids[1]),
                ):
                    prep, surfaces = backend.prepare(part.geometry)
                    found = next(
                        (s for s in surfaces if s.patch_id == sid and s.kind == "plane"), None
                    )
                    if found is None:
                        valid = False
                        break
                    tri = (prep.mesh.vertices @ tf[:3, :3].T + tf[:3, 3])[prep.mesh.faces]
                    supporting = np.flatnonzero(np.max(np.abs(tri @ n - h), axis=1) <= tolerance_m)
                    if not len(supporting) or not set(supporting).issubset(set(found.face_ids)):
                        valid = False
                        break
                if valid:
                    planes.append(
                        dict(
                            obstacle_id=other,
                            normal=n,
                            offset_m=h,
                            obstacle_surface_id=surface_ids[0],
                            moving_surface_id=surface_ids[1],
                        )
                    )
        return cls(
            moving_part_id,
            _binding(assembly, state, moving_part_id),
            state.poses[moving_part_id][:3, :3],
            tuple(planes),
        )

    def assert_matches(self, assembly: Assembly, state: AssemblyState, moving: str) -> None:
        if moving != self.moving_part_id or self.input_binding != _binding(assembly, state, moving):
            raise ValueError("ContactPolicy is stale or belongs to another moving part/world")


@dataclass(frozen=True, eq=False)
class PathValidation:
    status: str
    reason: str
    diagnostics: dict = field(default_factory=dict)
    witness: dict = field(default_factory=dict)
    validation_level: str = "unresolved"

    def __post_init__(self) -> None:
        for k in ("diagnostics", "witness"):
            object.__setattr__(self, k, freeze(getattr(self, k)))


@dataclass(frozen=True, eq=False)
class RemovalResult:
    status: str
    moving_part_id: str
    poses: tuple[FloatArray, ...]
    policy: ContactPolicy | None
    validation: PathValidation | None
    input_digest: str
    diagnostics: dict = field(default_factory=dict)
    schema_version: str = "wrs.assembly.removal/1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "poses", tuple(checked_tf(p) for p in self.poses))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))


def interpolate_pose(a: FloatArray, b: FloatArray, t: float) -> FloatArray:
    out = np.eye(4)
    out[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
    out[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3]))))(
        [t]
    ).as_matrix()[0]
    return out


def sample_path(
    poses: Sequence[FloatArray], config: MotionConfig | None = None
) -> tuple[FloatArray, ...]:
    cfg = config or MotionConfig()
    out = [checked_tf(poses[0])]
    for a, b in zip(poses, poses[1:]):
        angle = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude()
        count = max(
            1,
            int(np.ceil(np.linalg.norm(b[:3, 3] - a[:3, 3]) / cfg.max_translation_step_m)),
            int(np.ceil(angle / cfg.max_rotation_step_rad)),
        )
        out.extend(checked_tf(interpolate_pose(a, b, t / count)) for t in range(1, count + 1))
    return tuple(out)


def _normals(mesh, tf):
    tri = mesh.vertices[mesh.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(n, axis=1)
    n = n[lengths > 0] / lengths[lengths > 0, None]
    return np.unique(n, axis=0) @ tf[:3, :3].T


def validate_object_path(
    assembly: Assembly,
    state: AssemblyState,
    moving_part_id: str,
    poses: Iterable[ArrayLike],
    *,
    policy: ContactPolicy | None = None,
    config: MotionConfig | None = None,
    backend: ProximityBackend | None = None,
) -> PathValidation:
    """Certify complete intervals; uncertain small intervals stay unknown.

    Certificates: swept AABB, supporting halfspaces for translation, or a mesh
    distance lower bound minus the maximum rigid-point displacement. Endpoint
    samples alone never certify an interval. No point-query SDF sign assumption.
    """
    cfg = config or MotionConfig()
    backend = backend or MeshProximity(numerical_tol_m=cfg.numerical_tol_m)
    start = perf_counter()
    budget = QueryBudget(cfg.max_triangle_tests)
    path = tuple(checked_tf(p) for p in poses)
    if not path:
        raise ValueError("A path needs at least one pose")
    _binding(assembly, state, moving_part_id)
    if not np.allclose(path[0], state.poses[moving_part_id], atol=cfg.numerical_tol_m, rtol=0):
        raise ValueError("Path must start at the supplied state")
    if policy is not None:
        policy.assert_matches(assembly, state, moving_part_id)
    parts = {p.part_id: p for p in assembly.parts}
    moving = parts[moving_part_id]
    obstacles = [
        (
            p,
            state.poses[p.part_id],
            _world(p, state.poses[p.part_id]),
            _normals(p.geometry, state.poses[p.part_id]),
        )
        for p in assembly.parts
        if p.part_id in state.poses and p.part_id != moving_part_id
    ]
    local_normals = _normals(moving.geometry, np.eye(4))
    radius = float(np.max(np.linalg.norm(moving.geometry.vertices, axis=1)))
    intervals, certs = (
        0,
        dict(swept_aabb=0, separating_plane=0, allowed_surface=0, distance_bound=0),
    )

    def result(status, reason, witness=None):
        return PathValidation(
            status,
            reason,
            dict(
                elapsed_s=perf_counter() - start,
                intervals=intervals,
                triangle_tests=budget.used,
                certificates=certs,
                clearance_m=cfg.clearance_m,
            ),
            witness or {},
            "bounded_nominal_mesh" if status == "valid" else "unresolved",
        )

    pending = [(a, b) for a, b in zip(path, path[1:])] or [(path[0], path[0])]
    while pending:
        if intervals >= cfg.max_nodes or perf_counter() - start > cfg.time_limit_s:
            return result("unknown", "motion_budget_exhausted")
        a, b = pending.pop()
        intervals += 1
        angle = float(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude())
        translation = np.linalg.norm(b[:3, 3] - a[:3, 3])
        movement = translation / 2 + 2 * radius * np.sin(angle / 4)
        va, vb = _world(moving, a), _world(moving, b)
        # An endpoint union plus angular displacement contains the swept mesh.
        inflate = 2 * radius * np.sin(angle / 2)
        lo, hi = (
            np.minimum(va.min(0), vb.min(0)) - inflate,
            np.maximum(va.max(0), vb.max(0)) + inflate,
        )
        midpoint = None
        split = False
        for obstacle, otf, ov, other_normals in obstacles:
            witness = dict(obstacle_id=obstacle.part_id, start_tf=a, end_tf=b)
            if aabb_distance(lo, hi, ov.min(0), ov.max(0)) > cfg.clearance_m:
                certs["swept_aabb"] += 1
                continue
            if angle <= 1e-12:
                if (
                    policy is not None
                    and np.linalg.norm(a[:3, :3] - policy.reference_rotation, ord=2) * radius
                    <= cfg.numerical_tol_m
                ):
                    permits = [x for x in policy.planes if x["obstacle_id"] == obstacle.part_id]
                    if any(
                        min(
                            np.min(va @ x["normal"] - x["offset_m"]),
                            np.min(vb @ x["normal"] - x["offset_m"]),
                        )
                        >= -cfg.numerical_tol_m
                        for x in permits
                    ):
                        certs["allowed_surface"] += 1
                        continue
                axes = np.vstack(
                    (local_normals @ a[:3, :3].T, other_normals, np.eye(3), -np.eye(3))
                )
                # Each axis is a true supporting-plane certificate even for a
                # concave mesh. Failure of these axes is NOT an overlap verdict.
                # Bounded vectorized batches avoid V x F memory growth for STL.
                # A small deterministic subset is enough for this cheap gate;
                # failing it always falls through to the original-mesh BVH.
                if len(axes) > 256:
                    axes = axes[np.linspace(0, len(axes) - 1, 256, dtype=int)]
                separated = False
                batch = max(1, min(32, 262144 // max(len(va), len(ov))))
                for first in range(0, len(axes), batch):
                    directions = axes[first : first + batch].T
                    gap = np.minimum(
                        np.min(va @ directions, axis=0), np.min(vb @ directions, axis=0)
                    ) - np.max(ov @ directions, axis=0)
                    if np.max(gap) > cfg.clearance_m:
                        separated = True
                        break
                if separated:
                    certs["separating_plane"] += 1
                    continue
            if midpoint is None:
                midpoint = interpolate_pose(a, b, 0.5)
            overlap = backend.classify_overlap(moving, midpoint, obstacle, otf, budget=budget)
            if overlap.status == "penetrating":
                return result(
                    "blocked",
                    overlap.reason,
                    dict(witness, pose=midpoint, point=overlap.witness_world_m),
                )
            if overlap.status == "unknown":
                return result("unknown", overlap.reason, witness)
            distance = backend.pair_distance(moving, midpoint, obstacle, otf, budget=budget)
            if distance.status != "complete":
                return result("unknown", "distance_budget_exhausted", witness)
            if distance.lower_bound_m - movement > cfg.clearance_m:
                certs["distance_bound"] += 1
                continue
            if movement <= cfg.min_step_m:
                return result(
                    "unknown",
                    "unresolved_clearance_or_unpermitted_contact",
                    dict(
                        witness,
                        midpoint_gap_m=distance.lower_bound_m,
                        point_motion_bound_m=movement,
                    ),
                )
            split = True
            break
        if split:
            pending.extend(((a, midpoint), (midpoint, b)))
    return result("valid", "all_intervals_certified")


def _outside_pose(moving, tf, obstacles, direction, margin):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    vertices = _world(moving, tf)
    distance = (
        margin
        if not len(obstacles)
        else max(margin, float(np.max(obstacles @ d) - np.min(vertices @ d) + margin))
    )
    out = tf.copy()
    out[:3, 3] += d * distance
    return out


def plan_removal(
    assembly: Assembly,
    state: AssemblyState,
    action: str | RemovalAction,
    *,
    graph: ContactGraph | None = None,
    backend: ProximityBackend | None = None,
    config: MotionConfig | None = None,
    directions: ArrayLike | None = None,
) -> RemovalResult:
    """Single-part removal: SOCP direction, fixed axes, then bounded SE(3) detours."""
    from ..contact.analysis import analyze_contacts
    from ..contact.graph import build_contact_graph
    from .directions import assembly_directions

    cfg = config or MotionConfig()
    backend = backend or MeshProximity(numerical_tol_m=cfg.numerical_tol_m)
    action = action if isinstance(action, RemovalAction) else RemovalAction(action)
    key = digest(("removal/1", assembly, state, action, cfg, directions))
    if len(action.moving_part_ids) != 1 or action.primitive != "rigid":
        return RemovalResult(
            "unsupported",
            action.moving_part_ids[0],
            (),
            None,
            None,
            key,
            {"reason": "single_rigid_part_only"},
        )
    moving_id = action.moving_part_ids[0]
    _binding(assembly, state, moving_id)
    parts = {p.part_id: p for p in assembly.parts}
    part = parts[moving_id]
    if part.fixed:
        return RemovalResult(
            "unsupported", moving_id, (), None, None, key, {"reason": "fixed_part"}
        )
    graph = graph or build_contact_graph(
        assembly, state, analyze_contacts(assembly, state, backend=backend)
    )
    graph.assert_matches(assembly, state)
    policy = ContactPolicy.from_graph(
        assembly, state, moving_id, graph, backend=backend, tolerance_m=cfg.numerical_tol_m
    )
    started = perf_counter()
    attempts = []
    seeds = []
    if directions is None:
        local = assembly_directions(graph, (moving_id,))
        if local.best_direction is not None:
            seeds.append(local.best_direction)
        seeds.extend(np.vstack((np.eye(3), -np.eye(3))))
    else:
        raw = np.asarray(directions, dtype=float).reshape(-1, 3)
        if not len(raw) or not np.all(np.isfinite(raw)) or np.any(np.linalg.norm(raw, axis=1) == 0):
            raise ValueError("Finite nonzero directions required")
        seeds.extend(raw)
    unique = []
    for d in seeds:
        d = np.asarray(d, dtype=float) / np.linalg.norm(d)
        if not any(np.allclose(d, x, rtol=0, atol=1e-12) for x in unique):
            unique.append(d)
    others = [
        _world(p, state.poses[p.part_id])
        for p in assembly.parts
        if p.part_id in state.poses and p.part_id != moving_id
    ]
    ov = np.concatenate(others) if others else np.empty((0, 3))
    tf = state.poses[moving_id]
    candidates = [(tf, _outside_pose(part, tf, ov, d, cfg.outside_margin_m)) for d in unique]
    if cfg.search_se3 and directions is None:
        size = (
            max(0.01, float(np.linalg.norm(np.ptp(np.vstack([_world(part, tf), ov]), axis=0))))
            * cfg.detour_scale
        )
        for axis in np.vstack((np.eye(3), -np.eye(3))):
            waypoint = tf.copy()
            waypoint[:3, 3] += axis * size
            for d in unique:
                candidates.append(
                    (tf, waypoint, _outside_pose(part, waypoint, ov, d, cfg.outside_margin_m))
                )
        # Bounded orientation candidates use genuine quaternion interpolation.
        rotations = []
        for axis in np.vstack((np.eye(3), -np.eye(3))):
            waypoint = tf.copy()
            waypoint[:3, :3] = (
                Rotation.from_rotvec(axis * cfg.rotation_angle_rad).as_matrix() @ tf[:3, :3]
            )
            rotations.append(
                (tf, waypoint, _outside_pose(part, waypoint, ov, unique[0], cfg.outside_margin_m))
            )
        candidates = candidates[: len(unique)] + rotations + candidates[len(unique) :]
    for path in candidates[: cfg.max_candidates]:
        if perf_counter() - started > cfg.time_limit_s:
            break
        remaining = max(1e-9, cfg.time_limit_s - (perf_counter() - started))
        validation = validate_object_path(
            assembly,
            state,
            moving_id,
            path,
            policy=policy,
            config=replace(cfg, time_limit_s=remaining),
            backend=backend,
        )
        attempts.append(
            dict(
                status=validation.status,
                reason=validation.reason,
                diagnostics=validation.diagnostics,
            )
        )
        if validation.status == "valid":
            return RemovalResult(
                "success",
                moving_id,
                sample_path(path, cfg),
                policy,
                validation,
                key,
                dict(
                    attempts=attempts,
                    elapsed_s=perf_counter() - started,
                    outside="directional_support_plane",
                ),
            )
    return RemovalResult(
        "unknown",
        moving_id,
        (),
        policy,
        None,
        key,
        dict(
            reason="bounded_search_exhausted", attempts=attempts, elapsed_s=perf_counter() - started
        ),
    )
