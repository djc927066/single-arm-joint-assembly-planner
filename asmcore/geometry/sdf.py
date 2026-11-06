"""Signed-distance providers: negative inside, positive outside, local SI units."""
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from typing import Protocol, runtime_checkable
import numpy as np
from ..model import readonly, digest


@dataclass(frozen=True, eq=False)
class SDFSamples:
    """Batch values, projected witnesses and outward normals in a local frame.

    Invalid/domain-exterior samples carry finite placeholders and valid=False.
    ``error_m`` describes distance error, not normal or area error. Providers
    declare whether this is a heuristic guard or a supplied bound separately.
    """
    values_m: np.ndarray
    points_local_m: np.ndarray
    normals_local: np.ndarray
    error_m: np.ndarray
    valid: np.ndarray
    signed: np.ndarray
    face_ids: np.ndarray

    def __post_init__(self):
        n = len(self.values_m)
        for name in ('values_m', 'points_local_m', 'normals_local', 'error_m',
                     'valid', 'signed', 'face_ids'):
            dtype = bool if name in ('valid', 'signed') else (np.int64 if name == 'face_ids' else float)
            shape = (n, 3) if name in ('points_local_m', 'normals_local') else (n,)
            object.__setattr__(self, name, readonly(getattr(self, name), dtype, shape))
        if np.any(self.error_m < 0):
            raise ValueError('SDF distance error must be nonnegative')
        if np.any(self.valid) and not np.allclose(
                np.linalg.norm(self.normals_local[self.valid], axis=1), 1, atol=1e-7):
            raise ValueError('Valid SDF samples require unit outward normals')


@runtime_checkable
class SignedDistanceField(Protocol):
    """Immutable runtime extension point for mesh, grid or analytical fields.

    ``query`` accepts (N,3) local metre points. ``metadata`` must be JSON-safe,
    name the error model, and declare the geometry/domain assumptions. Supplied
    surface meshes must approximate this field's zero set within their declared
    geometry error plus the field error. Query methods must not change cache_key.
    """
    cache_key: str

    @property
    def metadata(self) -> dict: ...

    def query(self, points_local_m) -> SDFSamples: ...


class Open3DMeshSDF:
    """On-demand mesh SDF, without a voxel grid or point-cloud conversion.

    Parameters
    ----------
    prepared : PreparedMesh
        Validated local surface and outward normals.
    nsamples : int
        Odd ray count for inside/outside voting.
    open_surface : {'error', 'unsigned'}
        Open/invalid solids raise by default. Explicit unsigned mode permits
        distance queries but never supplies an inside/outside classification.

    Notes
    -----
    One normalized local scene is built per geometry. Open3D uses float32;
    error_m includes a numerical guard, not a formally certified error bound.
    Self-intersection checks apply to the supplied mesh, not the original CAD.
    """
    def __init__(self, prepared, *, nsamples=3, open_surface='error'):
        if not isinstance(nsamples, int) or nsamples < 1 or nsamples % 2 == 0:
            raise ValueError('nsamples must be a positive odd integer')
        if open_surface not in ('error', 'unsigned'):
            raise ValueError('open_surface must be error or unsigned')
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError('SDF mesh queries require the optional open3d dependency') from exc
        self._o3d = o3d
        self.prepared = prepared
        self.mesh = prepared.mesh
        self.nsamples = nsamples
        vs = self.mesh.vertices
        self._origin = (vs.min(0)+vs.max(0))/2
        self._scale = float(np.max(np.ptp(vs, axis=0)))
        if self._scale <= 0:
            raise ValueError('SDF mesh has zero extent')
        normalized = (vs-self._origin)/self._scale
        legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(normalized),
                                          o3d.utility.Vector3iVector(self.mesh.faces))
        intersections = legacy.get_self_intersecting_triangles()
        self._validation = {'closed': bool(prepared.is_closed),
                            'orientation_reliable': bool(prepared.orientation_reliable),
                            'watertight': bool(legacy.is_watertight()),
                            'self_intersection_pairs': len(intersections)}
        self._signed = bool(prepared.is_closed and prepared.orientation_reliable
                            and self._validation['watertight'] and not len(intersections))
        if not self._signed and open_surface == 'error':
            raise ValueError('SDF needs a watertight, oriented, non-self-intersecting solid; '
                             'use mesh backend or explicitly set open_surface="unsigned"')
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.core.Tensor(normalized.astype(np.float32)),
                                  o3d.core.Tensor(self.mesh.faces.astype(np.uint32)))
        self.cache_key = digest(('open3d_mesh_sdf/2', o3d.__version__, self.mesh.geometry_id,
                                 nsamples, open_surface))

    @property
    def metadata(self):
        return {'provider': 'open3d_mesh_sdf/2', 'version': self._o3d.__version__,
                'representation': 'on_demand_mesh_sdf', 'signed': self._signed,
                'sign_convention': 'negative_inside',
                'normal_source': 'sdf_gradient_away_from_zero; face_normal_at_zero_or_unsigned',
                'distance_error_model': 'float32_guard_not_certified',
                'coordinates': 'normalized_local', 'nsamples': self.nsamples,
                'solid_validation': dict(self._validation)}

    def query(self, points_local_m):
        """Return batched distances and mesh witnesses; all output lengths are metres."""
        points = readonly(points_local_m, shape=(None, 3))
        n = len(points)
        if not n:
            return SDFSamples(np.empty(0), np.empty((0, 3)), np.empty((0, 3)),
                              np.empty(0), np.empty(0, bool), np.empty(0, bool), np.empty(0, int))
        normalized = (points-self._origin)/self._scale
        tensor = self._o3d.core.Tensor(normalized.astype(np.float32))
        hit = self._scene.compute_closest_points(tensor)
        hit_normalized = hit['points'].numpy().astype(float)
        cp = hit_normalized*self._scale+self._origin
        ids = hit['primitive_ids'].numpy().astype(np.int64)
        displacement = (normalized-hit_normalized)*self._scale
        distances = np.linalg.norm(displacement, axis=1)
        if self._signed:
            # Reuse the nearest-point distance; occupancy supplies only the sign.
            inside = self._scene.compute_occupancy(tensor, nsamples=self.nsamples).numpy() > .5
            values = np.where(inside, -distances, distances)
        else:
            values = distances
        # Unit-scale normalization avoids rounding millimetre features at a
        # kilometre world offset. This guard remains an engineering estimate.
        error = 32*np.finfo(np.float32).eps*self._scale*np.maximum(1, np.linalg.norm(normalized, axis=1))
        normals = self.prepared.normals[ids].copy()
        away = (distances > 4*error) & self._signed
        # At an edge, several nearest triangles share one closest point. Their
        # face normals may disagree; the SDF gradient is determined by p-q.
        normals[away] = displacement[away]/values[away, None]
        return SDFSamples(values, cp, normals, error,
                          np.ones(n, bool), np.full(n, self._signed), ids)


@dataclass(frozen=True, eq=False)
class GridSDF:
    """An externally supplied regular SDF grid with trilinear interpolation.

    Parameters
    ----------
    values_m : array-like, shape (Nx, Ny, Nz)
        Signed distances in metres, negative inside. Each axis has >=2 nodes.
    origin_m : array-like, shape (3,)
        Local position of node [0,0,0], in metres.
    spacing_m : float or array-like, shape (3,)
        Positive grid spacing along x,y,z, in metres.
    error_bound_m : float
        Caller-supplied bound on interpolated distance error, including source
        and interpolation error. Grid spacing alone is not an error certificate.

    Notes
    -----
    Queries outside the grid are invalid; there is no clamping or extrapolation.
    Normals use the trilinear gradient; projected points are estimates, not
    guaranteed closest surface points. Arrays are copied to immutable storage.
    """
    values_m: np.ndarray
    origin_m: np.ndarray
    spacing_m: np.ndarray
    error_bound_m: float
    _cache_key: str = field(init=False, repr=False)

    def __post_init__(self):
        values = readonly(self.values_m)
        origin = readonly(self.origin_m, shape=(3,))
        spacing = readonly(np.broadcast_to(self.spacing_m, (3,)), shape=(3,))
        if values.ndim != 3 or min(values.shape) < 2 or np.any(spacing <= 0):
            raise ValueError('SDF needs a 3D grid with >=2 nodes per axis and positive spacing')
        if not np.isfinite(self.error_bound_m) or self.error_bound_m < 0:
            raise ValueError('Declare a finite, nonnegative interpolated distance error bound')
        object.__setattr__(self, 'values_m', values)
        object.__setattr__(self, 'origin_m', origin)
        object.__setattr__(self, 'spacing_m', spacing)
        h = sha256(values.tobytes()).hexdigest()
        object.__setattr__(self, '_cache_key', digest(('grid_sdf/1', h, values.shape,
                                                      origin, spacing, self.error_bound_m)))

    @property
    def cache_key(self):
        return self._cache_key

    @property
    def metadata(self):
        return {'provider': 'grid_sdf/1', 'representation': 'regular_grid',
                'sign_convention': 'negative_inside', 'shape': list(self.values_m.shape),
                'spacing_m': self.spacing_m.tolist(), 'origin_m': self.origin_m.tolist(),
                'distance_error_model': 'caller_declared_interpolated_error_bound',
                'error_bound_m': self.error_bound_m,
                'normal_source': 'trilinear_gradient', 'witness': 'approximate_sdf_projection'}

    def query(self, points_local_m):
        """Evaluate local (N,3) queries, reporting domain/zero-gradient failures."""
        points = readonly(points_local_m, shape=(None, 3))
        xyz = (points-self.origin_m)/self.spacing_m
        shape = np.asarray(self.values_m.shape)
        domain = np.all((xyz >= 0) & (xyz <= shape-1), axis=1)
        # Clipping is only for safe array indexing; domain-exterior results
        # remain invalid and are never used as evidence or exclusions.
        index = np.clip(np.floor(xyz).astype(np.int64), 0, shape-2)
        t = np.clip(xyz-index, 0, 1)
        values, grad = np.zeros(len(points)), np.zeros_like(points)
        for bits in product((0, 1), repeat=3):
            b = np.asarray(bits)
            corner = self.values_m[tuple((index+b).T)]
            weights = np.where(b, t, 1-t)
            values += corner*np.prod(weights, axis=1)
            for axis in range(3):
                others = [j for j in range(3) if j != axis]
                grad[:, axis] += corner*(1 if b[axis] else -1)*np.prod(weights[:, others], axis=1)/self.spacing_m[axis]
        norm = np.linalg.norm(grad, axis=1)
        valid = domain & (norm > 1e-12)
        normals = grad/np.maximum(norm[:, None], 1e-12)
        normals[~valid] = [0, 0, 1]
        projected = points-values[:, None]*normals
        return SDFSamples(values, projected, normals, np.full(len(points), self.error_bound_m),
                          valid, domain, np.full(len(points), -1))
