"""Adaptive, bidirectional near-surface bands with explicit unresolved area."""
from collections import defaultdict
import numpy as np
from ..model import ContactPatch, Region, MeshData
from ..geometry.mesh_bvh import QueryBudget, BudgetExceeded, closest_on_triangle
from ..geometry.planar import cell_regions


def _subdivide(tri):
    a, b, c = tri
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return (np.array([a, ab, ca]), np.array([ab, b, bc]),
            np.array([ca, bc, c]), np.array([ab, bc, ca]))


def analyze_mesh_pair(part_a, tf_a, prep_a, surface_a, part_b, tf_b, prep_b,
                      surface_b, *, config, backend, budget, distance_lower_bound_m=0.0):
    """Return near-band estimates and per-side coverage diagnostics.

    A Lipschitz lower bound excludes cells. A convex-target witness bound can
    certify whole source cells inside the band, without uniform subdivision.
    Remaining boundary cells carry sampled estimates and explicit uncertainty.
    Opposite-side areas are reported separately, never as load-bearing area.
    """
    cfg = config
    patches, reports = [], []
    for reverse in (False, True):
        # Reserve work for both sampling directions, even if the first is hard.
        remaining = budget.max_tests-budget.used
        side_budget = QueryBudget(max(1,remaining//(1 if reverse else 2)),parent=budget)
        src, dst, ps, pt, ss, st, ts, tt = (
            (part_b, part_a, prep_b, prep_a, surface_b, surface_a, tf_b, tf_a) if reverse else
            (part_a, part_b, prep_a, prep_b, surface_a, surface_b, tf_a, tf_b))
        target_mesh = MeshData(pt.mesh.vertices, pt.mesh.faces[st.face_ids], 'trusted')
        target = backend.index(target_mesh).placed(tt)
        source_triangles = ps.mesh.vertices[ps.mesh.faces[ss.face_ids]] @ ts[:3, :3].T + ts[:3, 3]
        source_normals = ps.normals[ss.face_ids] @ ts[:3, :3].T
        target_normals = pt.normals[st.face_ids] @ tt[:3, :3].T
        error = ps.mesh.geometry_error_m + pt.mesh.geometry_error_m + cfg.pose_error_m
        queue = []
        excluded_area = 0.0
        target_lo, target_hi = target.triangles.min(1), target.triangles.max(1)
        closest_cache = {}

        def closest(point):
            key = tuple(point)
            if key not in closest_cache:
                closest_cache[key] = target.closest(point, side_budget)
            return closest_cache[key]

        def cell_upper_bound(tri, target_id):
            # A triangle is convex: interpolate its three closest witnesses
            # with source barycentric weights. The resulting point stays on
            # that target triangle, and distance is bounded by max vertex gap.
            distances = []
            for vertex in tri:
                side_budget.consume()
                q = closest_on_triangle(vertex, target.triangles[target_id])
                distances.append(np.linalg.norm(vertex-q))
            return max(distances) + cfg.numerical_tol_m

        reliable = ps.orientation_reliable and pt.orientation_reliable
        for i, tri in reversed(list(enumerate(source_triangles))):
            # Each source triangle has a constant normal. If no opposing target
            # primitive has a nearby AABB, no point in this cell can qualify.
            # Only one target-sized vector is allocated, never a face-pair matrix.
            lower = np.linalg.norm(np.maximum(np.maximum(tri.min(0)-target_hi,
                                                          target_lo-tri.max(0)),0),axis=1)
            compatible = source_normals[i] @ target_normals.T <= -np.cos(cfg.normal_angle_rad)
            if not reliable:
                compatible[:] = True
            if not np.any(compatible & (lower <= cfg.near_tol_m+error)):
                excluded_area += np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
            else:
                queue.append((tri,source_normals[i],int(ss.face_ids[i])))
        cells = defaultdict(list)
        visited, unresolved_area, accepted_area, unprocessed_area = 0, 0.0, 0.0, 0.0
        bounded_area = 0.0
        reason = 'resolution_reached'
        while queue:
            tri, ns, source_id = queue.pop()
            area = float(np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])) / 2)
            if visited >= cfg.max_cells:
                unprocessed_area = area + sum(np.linalg.norm(np.cross(t[1]-t[0], t[2]-t[0]))/2 for t, _, _ in queue)
                unresolved_area += unprocessed_area
                reason = 'cell_budget_exhausted'
                break
            visited += 1
            center = tri.mean(0)
            radius = float(np.linalg.norm(tri-center, axis=1).max())
            try:
                d, q, fid = closest(center)
                lower = max(0, d-radius-error, distance_lower_bound_m-error)
                upper = d+radius+error
                if lower > cfg.near_tol_m:
                    excluded_area += area
                    continue
                upper = min(upper, cell_upper_bound(tri, fid)+error)
                # Any nearest primitive must be within this upper bound of
                # some source point. Check all such normals, not just samples.
                bounds = np.linalg.norm(np.maximum(np.maximum(tri.min(0)-target_hi,
                                                               target_lo-tri.max(0)),0),axis=1)
                possible = bounds <= upper+cfg.numerical_tol_m
                normals_bounded = (reliable and np.any(possible)
                                   and np.all(ns @ target_normals[possible].T <= -np.cos(cfg.normal_angle_rad)))
                certified = upper <= cfg.near_tol_m and normals_bounded
                if not certified and radius > cfg.surface_resolution_m:
                    queue.extend((child, ns, source_id) for child in _subdivide(tri))
                    continue
                samples = [] if certified else [closest(p) for p in tri]
            except BudgetExceeded:
                unprocessed_area = area + sum(np.linalg.norm(np.cross(t[1]-t[0], t[2]-t[0]))/2 for t, _, _ in queue)
                unresolved_area += unprocessed_area
                reason = 'triangle_budget_exhausted'
                break
            sampled_ids = [fid] + [s[2] for s in samples]
            facing = ns @ target_normals[sampled_ids].T <= -np.cos(cfg.normal_angle_rad)
            reliable = ps.orientation_reliable and pt.orientation_reliable
            if certified or (d <= cfg.near_tol_m and np.all(facing)):
                label = 'near' if lower > cfg.numerical_tol_m and reliable else 'unknown'
                cells[label].append((tri, center, q, ns, target_normals[fid], area,
                                     lower, upper, source_id, int(st.face_ids[fid]), certified))
                accepted_area += area
                bounded_area += area if certified else 0.0
                # A sampled band boundary is not an exact area certificate.
                if not certified:
                    unresolved_area += area
            else:
                # No normal samples or distance samples is not an exclusion proof.
                unresolved_area += area
        for label, entries in cells.items():
            groups, boundary_ok = cell_regions([e[0] for e in entries], cfg.numerical_tol_m*4)
            pa = np.asarray([e[1] for e in entries])
            pb = np.asarray([e[2] for e in entries])
            na = np.asarray([e[3] for e in entries])
            nb = np.asarray([e[4] for e in entries])
            weights = np.asarray([e[5] for e in entries])
            if reverse:
                pa, pb, na, nb = pb, pa, nb, na
            patches.append(ContactPatch(part_a.part_id, part_b.part_id, surface_a.patch_id,
                                        surface_b.patch_id, 2, label,
                                        'bounded' if boundary_ok and all(e[10] for e in entries) else 'estimated', pa, pb, na, nb,
                                        tuple(Region(tuple(entries[i][0] for i in ids), loops)
                                              for ids, loops in groups),
                                        (min(e[6] for e in entries), max(e[7] for e in entries)),
                                        float(weights.sum()), weights=weights, measure_kind='near_band',
                                        sampling_side='b' if reverse else 'a',
                                        provenance={'backend': 'mesh_adaptive/2',
                                                    'cell_prepared_face_pairs': [(e[9], e[8]) if reverse else (e[8], e[9]) for e in entries],
                                                    'source_face_ids_a': sorted({k for i in surface_a.face_ids for k in prep_a.source_face_ids[i]}),
                                                    'source_face_ids_b': sorted({k for i in surface_b.face_ids for k in prep_b.source_face_ids[i]}),
                                                    'boundary_valid': boundary_ok, 'error_bound_m': error,
                                                    'distance_interval': 'unsigned_surface_distance',
                                                    'cell_band_bounded': [e[10] for e in entries],
                                                    'normal_test': 'all_possible_nearest_primitives_when_bounded; otherwise_sampled',
                                                    'physical_contact_area': False}))
        reports.append({'sampling_side': 'b' if reverse else 'a', 'source_area_m2': float(ps.areas_m2[ss.face_ids].sum()),
                        'estimated_band_area_m2': accepted_area, 'excluded_area_m2': float(excluded_area),
                        'bounded_band_area_m2': float(bounded_area),
                        'unprocessed_area_m2': float(unprocessed_area),
                        'boundary_uncertain_area_m2': max(0.0,float(unresolved_area-unprocessed_area)),
                        'unresolved_area_m2': float(unresolved_area), 'visited_cells': visited, 'reason': reason,
                        'area_accounting': 'unresolved includes uncertain estimated band boundaries; do not sum all columns',
                        'lower_dimensional_contact': 'unsupported; a missing band is not evidence of no point/line contact'})
    return tuple(patches), tuple(reports)
