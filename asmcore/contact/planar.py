"""Contact regions from opposing plane patches, without collision manifolds."""

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.mesh_bvh import QueryBudget
from ..geometry.planar import (
    cell_regions,
    clip_halfplane,
    convex_intersection,
    polygon_measure,
)
from ..model import ContactPatch, Region


def _classification(g, error, cfg, reliable, allow_area_active=True):
    if not reliable:
        return "unknown"
    if g + error < -cfg.penetration_tol_m:
        return "interference"
    if allow_area_active and abs(g) + error <= cfg.numerical_tol_m:
        return "active"
    if cfg.idealize_contact and abs(g) + error <= cfg.contact_tol_m:
        return "active"
    if g - error > cfg.numerical_tol_m:
        return "near"
    return "unknown"


def _segment_union(segments, tol):
    groups = []
    for seg in segments:
        p, q = seg
        length = np.linalg.norm(q - p)
        if length <= tol:
            continue
        direction = (q - p) / length
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        for group in groups:
            origin, axis, intervals = group
            if abs(np.cross(axis, direction)) <= 1e-8 and abs(np.cross(axis, p - origin)) <= tol:
                intervals.append(sorted([np.dot(p - origin, axis), np.dot(q - origin, axis)]))
                break
        else:
            groups.append([p, direction, [sorted([0.0, np.dot(q - p, direction)])]])
    result = []
    for origin, axis, intervals in groups:
        lo, hi = sorted(intervals)[0]
        for x, y in sorted(intervals)[1:]:
            if x <= hi + tol:
                hi = max(hi, y)
            else:
                result.append(np.asarray([origin + lo * axis, origin + hi * axis]))
                lo, hi = x, y
        result.append(np.asarray([origin + lo * axis, origin + hi * axis]))
    return result


def analyze_planar_pair(
    part_a, tf_a, prep_a, surface_a, part_b, tf_b, prep_b, surface_b, *, config, budget=None
):
    """Return (patches, diagnostics) for an opposing pair of bounded planes.

    Geometry is a union of input triangles, never its convex hull. Tilted-plane
    gap varies affinely across the chart and is clipped into separate bands.
    Contact at exactly zero gap is a line for nonparallel planes, not an area.
    """
    cfg, eps = config, config.numerical_tol_m
    budget = budget or QueryBudget(cfg.max_triangle_tests)
    sa, sb = surface_a, surface_b
    na, nb = tf_a[:3, :3] @ sa.normal, tf_b[:3, :3] @ sb.normal
    denom = np.dot(na, nb)
    if denom > -np.cos(cfg.normal_angle_rad):
        return (), {"candidate_triangle_pairs": 0, "unresolved_area_m2": 0.0, "status": "complete"}
    oa, ob = tf_a[:3, :3] @ sa.origin + tf_a[:3, 3], tf_b[:3, :3] @ sb.origin + tf_b[:3, 3]
    basis = tf_a[:3, :3] @ sa.basis
    va = prep_a.mesh.vertices @ tf_a[:3, :3].T + tf_a[:3, 3]
    vb = prep_b.mesh.vertices @ tf_b[:3, :3].T + tf_b[:3, 3]
    ta = (va[prep_a.mesh.faces[sa.face_ids]] - oa) @ basis
    tb = (vb[prep_b.mesh.faces[sb.face_ids]] - oa) @ basis
    coefficient = -(basis.T @ nb) / denom
    offset = np.dot(ob - oa, nb) / denom
    error = (
        prep_a.mesh.geometry_error_m
        + prep_b.mesh.geometry_error_m
        + sa.residual_m
        + sb.residual_m
        + cfg.pose_error_m
    ) / abs(denom)
    parallel = np.linalg.norm(coefficient) <= 1e-12
    reliable = sa.orientation_reliable and sb.orientation_reliable
    labeled = defaultdict(list)
    candidates, unresolved = 0, 0.0
    exhausted = False
    centers_b = tb.mean(axis=1)
    radii_b = np.linalg.norm(tb - centers_b[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centers_b)

    def gap(poly):
        return poly @ coefficient + offset

    def add(poly, label, pair):
        if len(poly):
            labeled[label].append((poly, pair))

    for i, tri in enumerate(ta):
        if exhausted:
            break
        center = tri.mean(axis=0)
        radius = np.linalg.norm(tri - center, axis=1).max()
        for j in sorted(tree.query_ball_point(center, radius + radii_b.max() + eps)):
            if np.any(tri.max(0) + eps < tb[j].min(0)) or np.any(tb[j].max(0) + eps < tri.min(0)):
                continue
            if budget.used >= budget.max_tests or candidates >= cfg.max_cells:
                exhausted = True
                # A conservative unresolved bound, including any partial cells.
                unresolved += sa.area_m2
                break
            budget.consume()
            candidates += 1
            poly = convex_intersection(tri, tb[j], eps)
            if not len(poly):
                continue
            pair = (int(sa.face_ids[i]), int(sb.face_ids[j]))
            poly = clip_halfplane(poly, coefficient, offset + cfg.near_tol_m + error, eps)
            poly = clip_halfplane(poly, -coefficient, cfg.near_tol_m + error - offset, eps)
            if not len(poly):
                continue
            gs = gap(poly)
            if parallel:
                add(poly, _classification(float(gs.mean()), error, cfg, reliable), pair)
            else:
                # Preserve the exact zero-gap line separately from finite near bands.
                zero = clip_halfplane(poly, coefficient, offset, eps)
                zero = clip_halfplane(zero, -coefficient, -offset, eps)
                if len(zero):
                    add(zero, _classification(0, error, cfg, reliable), pair)
                cuts = sorted(
                    set(
                        [
                            float(gs.min()),
                            float(gs.max()),
                            -error - eps,
                            error + eps,
                            -cfg.penetration_tol_m - error,
                            0.0,
                            -cfg.contact_tol_m + error,
                            cfg.contact_tol_m - error,
                        ]
                    )
                )
                for lower, upper in zip(cuts[:-1], cuts[1:]):
                    if upper < gs.min() or lower > gs.max() or upper - lower <= eps:
                        continue
                    cell = clip_halfplane(poly, coefficient, offset - lower, eps)
                    cell = clip_halfplane(cell, -coefficient, upper - offset, eps)
                    if len(cell) and abs(polygon_measure(cell)[0]) > eps * eps:
                        g = float(gap(np.asarray([polygon_measure(cell)[1]]))[0])
                        add(cell, _classification(g, error, cfg, reliable, False), pair)
    result = []
    for label, records in sorted(labeled.items()):
        area_cells, segments, points, face_pairs = [], [], [], []
        for poly, pair in records:
            area = abs(polygon_measure(poly)[0])
            if area > eps * eps:
                if area < cfg.min_area_m2:
                    unresolved += area
                    continue
                area_cells.append(poly if polygon_measure(poly)[0] > 0 else poly[::-1])
                face_pairs.append(pair)
            elif len(poly) > 1:
                # Degenerate clipping may retain redundant collinear vertices.
                distances = np.linalg.norm(poly[:, None, :] - poly[None, :, :], axis=2)
                i, j = np.unravel_index(np.argmax(distances), distances.shape)
                if distances[i, j] > cfg.min_length_m:
                    segments.append(poly[[i, j]])
                else:
                    points.append(poly.mean(0))
            else:
                points.append(poly[0])
        groups, boundary_ok = cell_regions(area_cells, eps * 4)
        provenance = {
            "backend": "planar/1",
            "area_chart": "part_a_plane",
            "geometry_a": part_a.geometry.geometry_id,
            "geometry_b": part_b.geometry.geometry_id,
            "source_face_ids_a": sorted(
                {k for i in sa.face_ids for k in prep_a.source_face_ids[i]}
            ),
            "source_face_ids_b": sorted(
                {k for i in sb.face_ids for k in prep_b.source_face_ids[i]}
            ),
            "cell_prepared_face_pairs": face_pairs,
            "error_bound_m": error,
            "boundary_valid": boundary_ok,
            "idealize_contact": cfg.idealize_contact,
        }

        def world_a(p):
            return np.asarray(p) @ basis.T + oa

        def world_b(p):
            return world_a(p) + gap(np.asarray(p))[:, None] * na

        def make(cells, loops, dimension, weights, length=0.0):
            centroids = np.asarray(
                [polygon_measure(c)[1] if dimension == 2 else c.mean(0) for c in cells]
            )
            witnesses_a, witnesses_b = world_a(centroids), world_b(centroids)
            all_gaps = np.concatenate([gap(c) for c in cells])
            regions = (
                tuple(
                    Region(
                        tuple(world_a(area_cells[i]) for i in ids),
                        tuple(world_a(loop) for loop in ring),
                    )
                    for ids, ring in loops
                )
                if dimension == 2
                else (Region(tuple(world_a(c) for c in cells)),)
            )
            return ContactPatch(
                part_a.part_id,
                part_b.part_id,
                sa.patch_id,
                sb.patch_id,
                dimension,
                label,
                "bounded" if boundary_ok and not exhausted else "unresolved",
                witnesses_a,
                witnesses_b,
                np.tile(na, (len(centroids), 1)),
                np.tile(nb, (len(centroids), 1)),
                regions,
                (float(all_gaps.min() - error), float(all_gaps.max() + error)),
                float(np.sum(weights)) if dimension == 2 else 0.0,
                length,
                np.asarray(weights),
                "contact" if label == "active" else "projected_region",
                provenance=provenance,
            )

        if area_cells:
            weights = [abs(polygon_measure(c)[0]) for c in area_cells]
            result.append(make(area_cells, groups, 2, weights))

        def covered(p):
            return any(len(convex_intersection(np.asarray([p]), cell, eps)) for cell in area_cells)

        segments = _segment_union([s for s in segments if not covered(s.mean(0))], eps)
        if segments:
            result.append(
                make(segments, (), 1, [], sum(np.linalg.norm(s[1] - s[0]) for s in segments))
            )
        unique_points = []
        for p in points:
            if covered(p) or any(np.linalg.norm(p - q) <= eps for q in unique_points):
                continue
            if any(
                np.linalg.norm(
                    p
                    - (
                        s[0]
                        + np.clip(
                            np.dot(p - s[0], s[1] - s[0]) / np.dot(s[1] - s[0], s[1] - s[0]), 0, 1
                        )
                        * (s[1] - s[0])
                    )
                )
                <= eps
                for s in segments
            ):
                continue
            unique_points.append(p)
        if unique_points:
            result.append(make([np.asarray([p]) for p in unique_points], (), 0, []))
    return tuple(result), {
        "candidate_triangle_pairs": candidates,
        "unresolved_area_m2": unresolved,
        "status": "budget_exhausted" if exhausted else "complete",
    }
