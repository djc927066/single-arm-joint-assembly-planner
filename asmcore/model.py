"""Immutable M1 models: metres, radians, kilograms and world-frame witnesses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

# Dtypes only; array shape and SI units are checked by the model constructors.
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def readonly(
    value: ArrayLike, dtype: DTypeLike = np.float64, shape: tuple[int | None, ...] | None = None
) -> NDArray[Any]:
    """Copy an array onto immutable storage, validating shape and finiteness."""
    a = np.asarray(value, dtype=dtype)
    if shape is not None and (
        a.ndim != len(shape) or any(n is not None and a.shape[i] != n for i, n in enumerate(shape))
    ):
        raise ValueError(f"Expected shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("Arrays must contain finite values")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def checked_tf(value: ArrayLike) -> FloatArray:
    """Validate a right-handed local-to-world rigid transform, shape (4,4)."""
    tf = readonly(value, shape=(4, 4))
    if (
        not np.allclose(tf[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not np.allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-8, rtol=0)
        or not np.isclose(np.linalg.det(tf[:3, :3]), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("Expected rigid right-handed transform (no scale/shear)")
    return tf


def freeze(value: Any) -> Any:
    """Recursively copy metadata, protecting shared caches from mutation."""
    if isinstance(value, np.ndarray):
        return readonly(value, dtype=value.dtype)
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    return value


def to_dict(value: Any) -> Any:
    """Convert models and immutable arrays/mappings to JSON-compatible values."""
    if is_dataclass(value):
        return {f.name: to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [to_dict(v) for v in value]
    return value


def digest(value: object) -> str:
    """Stable digest including configuration and coordinate frame."""
    return sha256(json.dumps(to_dict(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _nonnegative(obj, names):
    for name in names:
        x = getattr(obj, name)
        if not np.isfinite(x) or x < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, eq=False)
class MeshData:
    """Local (N,3) vertices in metres and (F,3) indices.

    ``trusted`` explicitly declares supplied winding, including open supports
    and cavity shells. ``auto`` diagnoses topology before orienting solids.
    """

    vertices: FloatArray
    faces: IntArray
    orientation: str = "auto"
    geometry_error_m: float = 0.0
    geometry_id: str = field(init=False)

    def __post_init__(self) -> None:
        vs = readonly(self.vertices, shape=(None, 3))
        raw = np.asarray(self.faces)
        if (
            raw.dtype.kind not in "iu"
            and raw.size
            and (not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)))
        ):
            raise ValueError("Faces must be integer indices")
        fs = readonly(raw, np.int64, (None, 3))
        if not len(vs) or not len(fs) or fs.min() < 0 or fs.max() >= len(vs):
            raise ValueError("Mesh must be nonempty with indices in bounds")
        if self.orientation not in ("auto", "trusted"):
            raise ValueError("orientation must be auto or trusted")
        _nonnegative(self, ["geometry_error_m"])
        object.__setattr__(self, "vertices", vs)
        object.__setattr__(self, "faces", fs)
        h = sha256(
            vs.tobytes()
            + fs.tobytes()
            + self.orientation.encode()
            + np.float64(self.geometry_error_m).tobytes()
        ).hexdigest()
        object.__setattr__(self, "geometry_id", h)


@dataclass(frozen=True, eq=False)
class Part:
    """A distinct instance; assembled_tf maps local metres into world metres."""

    part_id: str
    geometry: MeshData
    assembled_tf: FloatArray = field(default_factory=lambda: np.eye(4))
    mass_kg: float | None = None
    com_local_m: FloatArray | None = None
    friction: float | None = None
    fixed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.part_id, str) or not self.part_id:
            raise ValueError("part_id must be a nonempty string")
        if not isinstance(self.geometry, MeshData):
            raise ValueError("geometry must be MeshData")
        object.__setattr__(self, "assembled_tf", checked_tf(self.assembled_tf))
        if self.com_local_m is not None:
            object.__setattr__(self, "com_local_m", readonly(self.com_local_m, shape=(3,)))
        for name in ("mass_kg", "friction"):
            if getattr(self, name) is not None:
                _nonnegative(self, [name])
        if self.mass_kg == 0 and not self.fixed:
            raise ValueError("Movable part mass must be positive or unknown")


@dataclass(frozen=True)
class MatingRelation:
    """Design relation, independent of load-bearing contacts."""

    part_a: str
    part_b: str
    kind: str
    parameters: Mapping = field(default_factory=dict)
    source: str = "user"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", freeze(self.parameters))


@dataclass(frozen=True, eq=False)
class Assembly:
    """Instances with explicit physical and design metadata."""

    parts: tuple[Part, ...]
    gravity_world_m_s2: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    mating_relations: tuple[MatingRelation, ...] = ()
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        ids = [p.part_id for p in parts]
        if not parts or len(ids) != len(set(ids)):
            raise ValueError("Assembly needs nonempty, unique part IDs")
        for m in self.mating_relations:
            if m.part_a not in ids or m.part_b not in ids or m.part_a == m.part_b:
                raise ValueError("Mating relation must refer to distinct existing parts")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "mating_relations", tuple(self.mating_relations))
        object.__setattr__(
            self, "gravity_world_m_s2", readonly(self.gravity_world_m_s2, shape=(3,))
        )
        object.__setattr__(self, "provenance", freeze(self.provenance))

    def initial_state(self) -> AssemblyState:
        """Return a state at the declared assembled poses."""
        return AssemblyState({p.part_id: p.assembled_tf for p in self.parts})


@dataclass(frozen=True, eq=False)
class AssemblyState:
    """Explicit poses; missing instances are absent, not implicitly at rest."""

    poses: Mapping[str, FloatArray]
    world_revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.world_revision, int) or self.world_revision < 0:
            raise ValueError("world_revision must be a nonnegative integer")
        object.__setattr__(
            self, "poses", MappingProxyType({str(k): checked_tf(v) for k, v in self.poses.items()})
        )

    @property
    def present_part_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.poses))


@dataclass(frozen=True)
class GeometryConfig:
    """Independent preprocessing tolerances (length m, area m², angles rad)."""

    weld_m: float = 0.0
    degenerate_area_m2: float = 1e-16
    plane_tol_m: float = 1e-9
    normal_angle_rad: float = 1e-5
    smooth_angle_rad: float = np.pi / 12
    repair_orientation: bool = True

    def __post_init__(self) -> None:
        _nonnegative(
            self,
            ["weld_m", "degenerate_area_m2", "plane_tol_m", "normal_angle_rad", "smooth_angle_rad"],
        )
        if not 0 <= self.normal_angle_rad <= self.smooth_angle_rad < np.pi / 2:
            raise ValueError("Require normal_angle <= smooth_angle < pi/2")


@dataclass(frozen=True)
class ContactConfig:
    """Separate numerical, zero-gap-model, near-band and query tolerances.

    Positive gaps stay near unless ``idealize_contact`` explicitly opts into
    a zero-gap model. Curved near bands never become active through this flag.
    """

    numerical_tol_m: float = 1e-10
    contact_tol_m: float = 1e-6
    near_tol_m: float = 5e-4
    penetration_tol_m: float = 1e-8
    normal_angle_rad: float = np.pi / 36
    min_area_m2: float = 1e-14
    min_length_m: float = 1e-10
    surface_resolution_m: float = 0.002
    max_cells: int = 10000
    max_triangle_tests: int = 300000
    pose_error_m: float = 0.0
    idealize_contact: bool = False

    def __post_init__(self) -> None:
        _nonnegative(
            self,
            [
                "numerical_tol_m",
                "contact_tol_m",
                "near_tol_m",
                "penetration_tol_m",
                "normal_angle_rad",
                "min_area_m2",
                "min_length_m",
                "surface_resolution_m",
                "pose_error_m",
            ],
        )
        if (
            self.numerical_tol_m <= 0
            or self.surface_resolution_m <= 0
            or self.near_tol_m < self.contact_tol_m
            or self.contact_tol_m < self.numerical_tol_m
            or self.normal_angle_rad >= np.pi / 2
        ):
            raise ValueError("Invalid tolerance ordering or resolution")
        for key in ("max_cells", "max_triangle_tests"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")


@dataclass(frozen=True, eq=False)
class PreparedMesh:
    mesh: MeshData
    source_face_ids: tuple[tuple[int, ...], ...]
    normals: np.ndarray
    areas_m2: np.ndarray
    adjacency: tuple[tuple[int, ...], ...]
    components: tuple[tuple[int, ...], ...]
    is_closed: bool
    orientation_reliable: bool
    diagnostics: tuple[str, ...]
    max_vertex_displacement_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "normals", readonly(self.normals, shape=(len(self.mesh.faces), 3)))
        object.__setattr__(self, "areas_m2", readonly(self.areas_m2, shape=(len(self.mesh.faces),)))


@dataclass(frozen=True, eq=False)
class SurfacePatch:
    patch_id: str
    face_ids: np.ndarray
    kind: str
    origin: np.ndarray
    normal: np.ndarray
    basis: np.ndarray
    boundary_loops: tuple[np.ndarray, ...]
    area_m2: float
    residual_m: float
    orientation_reliable: bool

    def __post_init__(self) -> None:
        for key in ("face_ids", "origin", "normal", "basis"):
            object.__setattr__(
                self,
                key,
                readonly(getattr(self, key), np.int64 if key == "face_ids" else np.float64),
            )
        object.__setattr__(self, "boundary_loops", tuple(readonly(x) for x in self.boundary_loops))


@dataclass(frozen=True, eq=False)
class Region:
    """Non-overlapping cells and oriented loops including holes, in world m."""

    cells_world_m: tuple[np.ndarray, ...]
    boundary_loops_world_m: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        for key in ("cells_world_m", "boundary_loops_world_m"):
            object.__setattr__(
                self, key, tuple(readonly(x, shape=(None, 3)) for x in getattr(self, key))
            )


@dataclass(frozen=True, eq=False)
class ContactPatch:
    part_a: str
    part_b: str
    surface_a_id: str
    surface_b_id: str
    dimension: int
    classification: str
    quality: str
    points_a_world_m: np.ndarray
    points_b_world_m: np.ndarray
    normals_a_world: np.ndarray
    normals_b_world: np.ndarray
    regions: tuple[Region, ...]
    gap_interval_m: tuple[float, float]
    area_m2: float = 0.0
    length_m: float = 0.0
    weights: np.ndarray = field(default_factory=lambda: np.empty(0))
    measure_kind: str = "contact"
    sampling_side: str = "a"
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("points_a_world_m", "points_b_world_m", "normals_a_world", "normals_b_world"):
            object.__setattr__(self, key, readonly(getattr(self, key), shape=(None, 3)))
        k = len(self.points_a_world_m)
        if any(
            len(getattr(self, key)) != k
            for key in ("points_b_world_m", "normals_a_world", "normals_b_world")
        ):
            raise ValueError("Witnesses and normals must correspond")
        if self.dimension not in (0, 1, 2) or self.classification not in (
            "active",
            "near",
            "interference",
            "unknown",
        ):
            raise ValueError("Invalid contact dimension or classification")
        for normals in (self.normals_a_world, self.normals_b_world):
            if k and not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7):
                raise ValueError("Contact normals must be unit vectors")
        object.__setattr__(self, "weights", readonly(self.weights, shape=(None,)))
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "provenance", freeze(self.provenance))


@dataclass(frozen=True, eq=False)
class ContactAnalysis:
    patches: tuple[ContactPatch, ...]
    pair_diagnostics: tuple[Mapping[str, Any], ...]
    state_digest: str
    mating_relations: tuple[MatingRelation, ...] = ()
    statistics: Mapping = field(default_factory=dict)
    schema_version: str = "wrs.assembly.contact/1"
    input_binding: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patches", tuple(self.patches))
        object.__setattr__(
            self, "pair_diagnostics", tuple(freeze(d) for d in self.pair_diagnostics)
        )
        object.__setattr__(self, "mating_relations", tuple(self.mating_relations))
        object.__setattr__(self, "statistics", freeze(self.statistics))
        object.__setattr__(self, "input_binding", freeze(self.input_binding))

    def to_dict(self):
        """Return a JSON-compatible report with all geometric witnesses."""
        return to_dict(self)
