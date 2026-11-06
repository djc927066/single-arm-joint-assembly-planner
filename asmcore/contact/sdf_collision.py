"""Fast SDF collision queries with explicit, separate nominal contact geometry."""
from collections import deque
from time import perf_counter
import numpy as np
from ..model import checked_tf, digest
from .sdf_backend import SDFContactBackend, SDFConfig, _areas
from ._sdf_cells import split_triangles


class SDFCollisionChecker:
    """Query mesh collisions using bilateral signed-distance samples and bounds.

    Parameters
    ----------
    resolution_m : float
        Terminal cell radius. Unresolved cells return unknown, never free.
    max_query_points : int
        Per-pair budget, shared between the two source surfaces.
    batch_size : int
        Maximum cells in one vectorized query.
    penetration_tol_m : float
        Negative distance beyond this tolerance and field/geometry guards is
        penetration evidence. It does not define an active contact area.
    open_surface : {'error', 'unsigned'}
        Explicit policy for invalid target solids. Unsigned fields cannot
        certify absence of containment or supply a penetration sign.
    use_aabb : bool
        Permit independent box separation and separating-support-plane contact
        checks. Touching from this path is explicitly nominal mesh evidence.
    max_touch_tests : int
        Independent triangle-feature budget for the nominal touch fast path.

    Notes
    -----
    Only mesh-derived fields are supported here. Results name their evidence:
    aabb separation, nominal support-plane touching, negative surface witness,
    positive SDF cell bounds, or unknown. SDF evidence is estimated: Open3D's float32 guard is not a formal
    error certificate. This is a discrete pose query, not continuous collision
    detection. Native fields require a global zero-set/domain contract first.
    """
    def __init__(self, *, resolution_m=.0002, max_query_points=200000,
                 batch_size=1024, penetration_tol_m=1e-8, open_surface='error', use_aabb=True,
                 max_touch_tests=50000):
        if not np.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError('resolution_m must be finite and positive')
        if not np.isfinite(penetration_tol_m) or penetration_tol_m < 0:
            raise ValueError('penetration_tol_m must be finite and nonnegative')
        for value in (max_query_points, batch_size, max_touch_tests):
            if not isinstance(value, int) or value < 1:
                raise ValueError('Budgets and batch sizes must be positive integers')
        self.resolution_m, self.max_query_points = resolution_m, max_query_points
        self.batch_size, self.penetration_tol_m = batch_size, penetration_tol_m
        self.use_aabb = bool(use_aabb)
        self.max_touch_tests = max_touch_tests
        self._backend = SDFContactBackend(sdf_config=SDFConfig(open_surface=open_surface,
                                                              validate_mesh_overlap=False))
        self._provider_key = self._backend.cache_key

    def prepare(self, model):
        """Cache validated local mesh geometry and its on-demand distance field."""
        if 'sdf' in model.representations:
            raise NotImplementedError('Native collision fields require a global zero-set/domain contract')
        return self._backend.prepare(model)

    def query(self, a, b, *, tf_a=None, tf_b=None):
        """Return a JSON-compatible report with status, evidence, coverage and timing.

        Negative evidence only requires a valid signed target and a point on
        the supplied source surface; it does not certify that an open source
        encloses a volume. Separation by SDF requires both fields to be signed.
        No boolean is returned that could mistake unknown for collision-free.
        """
        start = perf_counter()
        ta, tb = checked_tf(a.tf if tf_a is None else tf_a), checked_tf(b.tf if tf_b is None else tf_b)
        if 'sdf' in a.representations or 'sdf' in b.representations:
            raise NotImplementedError('Native collision fields require a global zero-set/domain contract')
        guard = a.geometry.geometry_error_m+b.geometry.geometry_error_m+1e-10
        va = a.geometry.vertices @ ta[:3, :3].T+ta[:3, 3]
        vb = b.geometry.vertices @ tb[:3, :3].T+tb[:3, 3]
        guard += 32*np.finfo(float).eps*max(1.0, float(np.abs(va).max()), float(np.abs(vb).max()))
        lower = float(np.linalg.norm(np.maximum(np.maximum(va.min(0)-vb.max(0), vb.min(0)-va.max(0)), 0)))
        report = {'schema_version': 'wrs.assembly.sdf_collision/1', 'part_a': a.name, 'part_b': b.name,
                  'status': 'unknown', 'quality': 'estimated', 'reason': '', 'witness': None,
                  'touch_evidence': None, 'touch_triangle_tests': 0,
                  'sides': [], 'query_points': 0, 'interior_query_points': 0, 'aabb_distance_lower_m': lower,
                  'resolution_m': self.resolution_m, 'max_query_points': self.max_query_points,
                  'state_digest': digest((a.name, a.geometry_key, b.name, b.geometry_key, ta, tb, self.resolution_m,
                                          self.max_query_points, self.batch_size, self.penetration_tol_m,
                                          self.use_aabb, self.max_touch_tests, self._provider_key, 'sdf_collision/3'))}
        if self.use_aabb and lower > guard:
            report.update(status='separated', quality='aabb_bound', reason='disjoint_aabb')
            report['timing_s'] = {'prepare': 0.0, 'query': perf_counter()-start, 'total': perf_counter()-start}
            return report
        prepare_started = perf_counter()
        pa, fa = self.prepare(a)
        pb, fb = self.prepare(b)
        prepared_at = perf_counter()
        report['fields'] = {'a': fa.metadata, 'b': fb.metadata}
        if self.use_aabb:
            from ._support_contact import support_contact
            contact = support_contact(pa, pb, ta, tb, max_tests=self.max_touch_tests, first_only=True)
            report['touch_triangle_tests'] = contact['triangle_tests']
            if contact['status'] == 'touching':
                report.update(status='touching', quality='nominal_mesh',
                              reason='separating_plane_and_surface_intersection', touch_evidence=contact)
                report['timing_s'] = {'prepare': prepared_at-prepare_started,
                                     'query': perf_counter()-prepared_at, 'total': perf_counter()-start}
                return report
        remaining = self.max_query_points
        # Surface-only samples cannot detect coincident solids. A small set of
        # candidate interior points is accepted only when BOTH fields confirm
        # negative distance, including the source's own interior test.
        if fa.metadata['signed'] and fb.metadata['signed']:
            count = min(len(pa.mesh.faces), 256, remaining//8)
            if count:
                ids = np.linspace(0, len(pa.mesh.faces)-1, count, dtype=int)
                centers = pa.mesh.vertices[pa.mesh.faces[ids]].mean(1)
                step = min(self.resolution_m, float(np.ptp(pa.mesh.vertices, axis=0).max())*.01)
                local = centers-pa.normals[ids]*step
                source_values = fa.query(local)
                world = local @ ta[:3, :3].T+ta[:3, 3]
                target_values = fb.query((world-tb[:3, 3]) @ tb[:3, :3])
                remaining -= 2*count
                report['interior_query_points'] = 2*count
                inside = ((source_values.values_m < -source_values.error_m-guard-self.penetration_tol_m)
                          & (target_values.values_m < -target_values.error_m-guard-self.penetration_tol_m))
                if np.any(inside):
                    i = int(np.flatnonzero(inside)[0])
                    report.update(status='penetrating', reason='shared_sdf_interior_witness',
                                  query_points=self.max_query_points-remaining,
                                  witness={'point_world_m': world[i].tolist(), 'sampling_side': 'a',
                                           'distance_m': float(target_values.values_m[i]),
                                           'source_distance_m': float(source_values.values_m[i]),
                                           'error_guard_m': float(target_values.error_m[i]+guard)})
                    report['timing_s'] = {'prepare': prepared_at-prepare_started, 'query': perf_counter()-prepared_at,
                                         'total': perf_counter()-start}
                    return report
        for side, prep, target, ts, tt in (('a', pa, fb, ta, tb), ('b', pb, fa, tb, ta)):
            budget = remaining//2 if side == 'a' else remaining
            result = self._surface(prep, target, ts, tt, budget, guard)
            result['sampling_side'] = side
            report['sides'].append(result)
            remaining -= result['query_points']
            if result['witness'] is not None:
                report.update(status='penetrating', reason='negative_sdf_surface_witness', witness=result['witness'])
                report['witness']['sampling_side'] = side
                break
        report['query_points'] = self.max_query_points-remaining
        if report['status'] != 'penetrating':
            complete = len(report['sides']) == 2 and all(s['unresolved_area_m2'] == 0 for s in report['sides'])
            if complete and fa.metadata['signed'] and fb.metadata['signed']:
                report.update(status='separated', reason='positive_sdf_cell_bounds')
            else:
                report['reason'] = 'unsigned_or_unresolved_surface_cells'
        report['timing_s'] = {'prepare': prepared_at-prepare_started, 'query': perf_counter()-prepared_at,
                             'total': perf_counter()-start}
        return report

    def _surface(self, prep, target, ts, tt, budget, guard):
        triangles = prep.mesh.vertices[prep.mesh.faces] @ ts[:3, :3].T+ts[:3, 3]
        source_area = float(prep.areas_m2.sum())
        result = {'query_points': 0, 'source_area_m2': source_area, 'excluded_area_m2': 0.0,
                  'unresolved_area_m2': 0.0, 'witness': None, 'reason': 'covered'}
        if not target.metadata['signed']:
            result.update(unresolved_area_m2=source_area, reason='unsigned_target')
            return result
        queue = deque(zip(triangles, np.arange(len(triangles))))
        while queue:
            count = min(len(queue), self.batch_size, (budget-result['query_points'])//4)
            if not count:
                result['unresolved_area_m2'] += float(_areas(np.asarray([item[0] for item in queue])).sum())
                result['reason'] = 'query_budget_exhausted'
                break
            batch = [queue.popleft() for _ in range(count)]
            cells = np.asarray([item[0] for item in batch])
            ids = np.asarray([item[1] for item in batch])
            centers = cells.mean(1)
            probes = np.concatenate((centers[:, None], cells), axis=1)
            values = target.query((probes.reshape(-1, 3)-tt[:3, 3]) @ tt[:3, :3])
            result['query_points'] += 4*count
            distance = values.values_m.reshape(-1, 4)
            error = values.error_m.reshape(-1, 4)+guard
            negative = values.signed.reshape(-1, 4) & (distance < -error-self.penetration_tol_m)
            if np.any(negative):
                j, k = np.unravel_index(np.argmin(np.where(negative, distance, np.inf)), distance.shape)
                result['witness'] = {'point_world_m': probes[j, k].tolist(), 'distance_m': float(distance[j, k]),
                                     'error_guard_m': float(error[j, k]), 'source_prepared_face_id': int(ids[j])}
                result['reason'] = 'early_exit_with_negative_sample'
                # Coverage is intentionally incomplete after a positive finding.
                result['unresolved_area_m2'] = source_area-result['excluded_area_m2']
                return result
            radius = np.linalg.norm(cells-centers[:, None], axis=2).max(1)
            outside = values.signed[::4] & (distance[:, 0]-radius-error[:, 0] > 0)
            areas = _areas(cells)
            result['excluded_area_m2'] += float(areas[outside].sum())
            terminal = ~outside & (radius <= self.resolution_m)
            result['unresolved_area_m2'] += float(areas[terminal].sum())
            refine = ~outside & ~terminal
            if np.any(refine):
                queue.extend(zip(split_triangles(cells[refine]).reshape(-1, 3, 3), np.repeat(ids[refine], 2)))
        if result['reason'] == 'covered' and result['unresolved_area_m2']:
            result['reason'] = 'resolution_reached_near_zero'
        return result

    def touch_regions(self, a, b, *, tf_a=None, tf_b=None, max_triangle_tests=50000):
        """Extract nominal point/line/area contact on a separating support plane.

        This independent mesh-geometry path is not a thresholded SDF band.
        It retains holes, original triangles, explicit proof and its own budget.
        General curved mating contact without a found support plane remains
        unsupported/unknown. Empty or incomplete output is not a free-space proof.
        """
        from ._support_contact import support_contact
        if not isinstance(max_triangle_tests, int) or max_triangle_tests < 1:
            raise ValueError('max_triangle_tests must be a positive integer')
        ta, tb = checked_tf(a.tf if tf_a is None else tf_a), checked_tf(b.tf if tf_b is None else tf_b)
        pa, _ = self.prepare(a)
        pb, _ = self.prepare(b)
        result = support_contact(pa, pb, ta, tb, max_tests=max_triangle_tests)
        result.update(part_a=a.name, part_b=b.name,
                      state_digest=digest((a.name, a.geometry_key, b.name, b.geometry_key, ta, tb,
                                           max_triangle_tests, self._provider_key, 'support_contact/1')))
        return result

    def penetration_regions(self, a, b, *, tf_a=None, tf_b=None,
                            resolution_m=None, max_query_points=None):
        """Extract both source-surface portions inside the opposite signed solid.

        This optional, separately timed traversal does not change ``query`` or
        its early-exit performance. It returns estimated polygon surfaces, not
        an intersection volume, zero-gap contact area, or collision-free proof.
        An unsigned target makes that side unavailable; the reverse side can
        still be drawn. Empty output with incomplete coverage is not absence.
        """
        from ._penetration_regions import extract_surface
        start = perf_counter()
        resolution = self.resolution_m if resolution_m is None else resolution_m
        budget = self.max_query_points if max_query_points is None else max_query_points
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('resolution_m must be finite and positive')
        if not isinstance(budget, int) or budget < 1:
            raise ValueError('max_query_points must be a positive integer')
        ta, tb = checked_tf(a.tf if tf_a is None else tf_a), checked_tf(b.tf if tf_b is None else tf_b)
        pa, fa = self.prepare(a)
        pb, fb = self.prepare(b)
        prepared_at = perf_counter()
        guard = a.geometry.geometry_error_m+b.geometry.geometry_error_m+1e-10
        guard += 32*np.finfo(float).eps*max(1., float(np.abs(ta[:3, 3]).max()), float(np.abs(tb[:3, 3]).max()))
        sides, remaining = [], budget
        for side, prep, field, ts, tt in (('a', pa, fb, ta, tb), ('b', pb, fa, tb, ta)):
            found = extract_surface(prep, field, ts, tt, resolution=resolution,
                                    budget=remaining//2 if side == 'a' else remaining,
                                    batch_size=self.batch_size, guard=guard, tolerance=self.penetration_tol_m)
            found['sampling_side'] = side
            sides.append(found)
            remaining -= found['query_points']
        return {'schema_version': 'wrs.assembly.penetration_regions/1',
                'part_a': a.name, 'part_b': b.name, 'quality': 'estimated',
                'measure_kind': 'surface_inside_opposite_solid', 'sides': sides,
                'resolution_m': resolution, 'max_query_points': budget,
                'query_points': budget-remaining, 'fields': {'a': fa.metadata, 'b': fb.metadata},
                'state_digest': digest((a.name, a.geometry_key, ta, b.name, b.geometry_key, tb,
                                        resolution, budget, self.batch_size, self.penetration_tol_m, self._provider_key,
                                        'penetration_regions/1')),
                'timing_s': {'prepare': prepared_at-start, 'extraction': perf_counter()-prepared_at,
                             'total': perf_counter()-start}}
