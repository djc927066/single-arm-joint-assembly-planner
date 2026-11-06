"""Optional surface portions inside another solid; separate from fast queries."""
from collections import deque
from time import perf_counter
import numpy as np
from ._sdf_cells import split_triangles, clip_cells, measure_cells
from .sdf_backend import _areas


def extract_surface(prepared, target, ts, tt, *, resolution, budget, batch_size, guard, tolerance):
    """Batch adaptive signed-distance clipping without a facing-normal filter.

    Regions are source-surface polygons, not the Boolean intersection volume.
    Terminal clipping interpolates guarded vertex distances, so its boundary
    is estimated. Entire terminal-cell area is retained as boundary uncertainty.
    """
    triangles = prepared.mesh.vertices[prepared.mesh.faces] @ ts[:3, :3].T+ts[:3, 3]
    source_area = float(prepared.areas_m2.sum())
    result = {'status': 'estimated', 'reason': 'surface_traversed', 'query_points': 0,
              'cells_world_m': [], 'source_prepared_face_ids': [], 'cell_areas_m2': [],
              'area_m2': 0.0, 'source_area_m2': source_area, 'excluded_area_m2': 0.0,
              'boundary_uncertain_area_m2': 0.0, 'unprocessed_area_m2': 0.0,
              'invalid_area_m2': 0.0, 'physical_contact_area': False}
    if not target.metadata['signed']:
        result.update(status='unavailable', reason='unsigned_target_has_no_interior',
                      unprocessed_area_m2=source_area)
        return result
    if prepared.mesh.geometry_id == target.mesh.geometry_id and np.array_equal(ts, tt):
        # Equal nominal mesh surfaces are on the same zero set, even though
        # the solids overlap. This is not a collision-free result.
        result.update(reason='coincident_mesh_surfaces', excluded_area_m2=source_area)
        return result
    queue = deque(zip(triangles, np.arange(len(triangles))))
    query_seconds = 0.0

    def append(polygons, counts, areas, ids):
        keep = np.flatnonzero((counts >= 3) & (areas > 1e-18))
        result['cells_world_m'].extend(polygons[j, :counts[j]].tolist() for j in keep)
        result['cell_areas_m2'].extend(areas[keep].tolist())
        result['source_prepared_face_ids'].extend(ids[keep].tolist())

    while queue:
        count = min(len(queue), batch_size, (budget-result['query_points'])//4)
        if not count:
            result.update(status='incomplete', reason='query_budget_exhausted')
            result['unprocessed_area_m2'] = float(_areas(np.asarray([x[0] for x in queue])).sum())
            break
        batch = [queue.popleft() for _ in range(count)]
        cells, ids = np.asarray([x[0] for x in batch]), np.asarray([x[1] for x in batch])
        center = cells.mean(1)
        probes = np.concatenate((center[:, None], cells), axis=1)
        started = perf_counter()
        sample = target.query((probes.reshape(-1, 3)-tt[:3, 3]) @ tt[:3, :3])
        query_seconds += perf_counter()-started
        result['query_points'] += 4*count
        phi = sample.values_m.reshape(-1, 4)
        error = sample.error_m.reshape(-1, 4)+guard
        valid = (sample.valid & sample.signed).reshape(-1, 4).all(1)
        radius = np.linalg.norm(cells-center[:, None], axis=2).max(1)
        area = _areas(cells)
        result['invalid_area_m2'] += float(area[~valid].sum())
        # Lipschitz bounds classify complete cells. No early exit after a hit.
        inside = valid & (phi[:, 0]+radius+error[:, 0] < -tolerance)
        outside = valid & (phi[:, 0]-radius-error[:, 0] >= -tolerance)
        result['excluded_area_m2'] += float(area[outside].sum())
        append(cells[inside], np.full(inside.sum(), 3), area[inside], ids[inside])
        pending = valid & ~inside & ~outside
        refine = pending & (radius > resolution)
        if np.any(refine):
            queue.extend(zip(split_triangles(cells[refine]).reshape(-1, 3, 3), np.repeat(ids[refine], 2)))
        terminal = pending & ~refine
        if np.any(terminal):
            # Only negative values beyond the declared guard get a red fill.
            scores = np.ones((int(terminal.sum()), 3, 3))
            scores[:, 0] = -phi[terminal, 1:]-error[terminal, 1:]-tolerance
            polygons, counts = clip_cells(cells[terminal], scores)
            areas, _ = measure_cells(polygons, counts)
            append(polygons, counts, areas, ids[terminal])
            result['boundary_uncertain_area_m2'] += float(area[terminal].sum())
    result['area_m2'] = float(sum(result['cell_areas_m2']))
    result['field_query_s'] = query_seconds
    if result['invalid_area_m2']:
        result.update(status='incomplete', reason='invalid_field_samples')
    return result
