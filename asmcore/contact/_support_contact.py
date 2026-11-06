"""Nominal mesh contact on a separating support plane, independent of SDF sign."""
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from ..geometry.planar import plane_basis, convex_intersection, polygon_measure
from ..geometry.mesh_bvh import segment_pair


def _intersection(a, b, eps):
    """Intersect convex 2D features, including points and line segments."""
    if len(a) > len(b):
        a, b = b, a
    if len(b) == 1:
        return a if np.linalg.norm(a[0]-b[0]) <= eps else np.empty((0, 2))
    if len(b) == 2:
        aa = np.column_stack((a, np.zeros(len(a))))
        bb = np.column_stack((b, np.zeros(len(b))))
        p, q = segment_pair(aa[0], aa[-1], bb[0], bb[-1])
        if np.linalg.norm(p-q) > eps:
            return np.empty((0, 2))
        points = [p[:2]]
        for x in (*aa, *bb):
            _, v = segment_pair(x, x, bb[0], bb[-1])
            _, u = segment_pair(x, x, aa[0], aa[-1])
            if np.linalg.norm(x-u) <= eps and np.linalg.norm(x-v) <= eps:
                points.append(x[:2])
        points = np.asarray(points)
        delta = points[:, None]-points[None]
        i, j = np.unravel_index(np.linalg.norm(delta, axis=2).argmax(), delta.shape[:2])
        return points[[i, j]] if np.linalg.norm(points[i]-points[j]) > eps else points[[i]]
    if len(a) == 1:
        edges = np.roll(b, -1, axis=0)-b
        sign = 1 if polygon_measure(b)[0] >= 0 else -1
        margin = sign*np.cross(edges, a[0]-b)
        return a if np.all(margin >= -eps*np.linalg.norm(edges, axis=1)) else np.empty((0, 2))
    return convex_intersection(a, b, eps)


def support_contact(pa, pb, ta, tb, *, max_tests, first_only=False):
    """Find zero-distance features where entire meshes lie on opposite sides.

    Candidate normals are world axes, both pose frames and the centre line.
    Failure to find such a plane is NOT proof of separation or non-contact.
    Positive gaps and penetration beyond numerical roundoff are not snapped.
    Source triangle pairs are retained; edges shared by area cells can also
    appear as lower-dimensional witnesses, never as extra area.
    """
    started = perf_counter()
    va = pa.mesh.vertices @ ta[:3, :3].T+ta[:3, 3]
    vb = pb.mesh.vertices @ tb[:3, :3].T+tb[:3, 3]
    origin = (va.min(0)+va.max(0)+vb.min(0)+vb.max(0))/4
    ca, cb = va-origin, vb-origin
    eps = 1e-10+64*np.finfo(float).eps*max(1., float(np.abs(va).max()), float(np.abs(vb).max()))
    result = {'schema_version': 'wrs.assembly.support_contact/1', 'status': 'unknown',
              'reason': 'no_supported_separating_plane', 'quality': 'nominal_mesh',
              'scope': 'supplied_mesh_surfaces', 'cells_world_m': [], 'cell_dimensions': [],
              'source_prepared_face_pairs': [], 'area_m2': 0.0, 'triangle_tests': 0,
              'complete': False, 'numerical_tol_m': eps, 'proof': None}
    if pa.mesh.geometry_error_m or pb.mesh.geometry_error_m:
        result.update(reason='declared_geometry_error_prevents_zero_gap_proof', timing_s=perf_counter()-started)
        return result
    normals = [*np.eye(3), *ta[:3, :3].T, *tb[:3, :3].T]
    direction = cb.mean(0)-ca.mean(0)
    if np.linalg.norm(direction) > eps:
        normals.append(direction/np.linalg.norm(direction))
    used = []
    for n in normals:
        if any(min(np.linalg.norm(n-u), np.linalg.norm(n+u)) <= 1e-12 for u in used):
            continue
        used.append(n)
        da, db = ca @ n, cb @ n
        if da.min()-db.max() > db.min()-da.max():
            n, da, db = -n, -da, -db
        gap = float(db.min()-da.max())
        if abs(gap) > eps:
            continue
        height = (da.max()+db.min())/2
        on_a, on_b = np.abs(da-height) <= eps, np.abs(db-height) <= eps
        fa = np.flatnonzero(on_a[pa.mesh.faces].any(1))
        fb = np.flatnonzero(on_b[pb.mesh.faces].any(1))
        basis = plane_basis(n)
        # All vertices bound the complete triangles by convexity. This plane
        # excludes transverse penetration anywhere, not merely at witnesses.
        proof = {'kind': 'separating_support_plane', 'normal_a_to_b': n.tolist(),
                 'origin_world_m': (origin+height*n).tolist(), 'nominal_gap_m': gap,
                 'max_a_plane_offset_m': float(da.max()-height),
                 'min_b_plane_offset_m': float(db.min()-height)}
        aa = [(ca[pa.mesh.faces[i]][on_a[pa.mesh.faces[i]]]) @ basis for i in fa]
        bb = [(cb[pb.mesh.faces[i]][on_b[pb.mesh.faces[i]]]) @ basis for i in fb]
        if not aa or not bb:
            continue
        centers = np.asarray([p.mean(0) for p in bb])
        radii = np.asarray([np.linalg.norm(p-p.mean(0), axis=1).max() for p in bb])
        tree = cKDTree(centers)
        seen = set()
        for i, a in enumerate(aa):
            center = a.mean(0)
            radius = np.linalg.norm(a-center, axis=1).max()
            for j in sorted(tree.query_ball_point(center, radius+radii.max()+eps)):
                b = bb[j]
                if np.any(a.max(0)+eps < b.min(0)) or np.any(b.max(0)+eps < a.min(0)):
                    continue
                if result['triangle_tests'] >= max_tests:
                    result.update(reason='touch_region_budget_exhausted', timing_s=perf_counter()-started)
                    return result
                result['triangle_tests'] += 1
                cell = _intersection(a, b, eps)
                if not len(cell):
                    continue
                area = abs(polygon_measure(cell)[0])
                if area > eps*eps:
                    dim = 2
                else:
                    delta = np.linalg.norm(cell[:, None]-cell[None], axis=2)
                    p, q = np.unravel_index(delta.argmax(), delta.shape)
                    dim = int(delta[p, q] > eps)
                    cell = cell[[p, q]] if dim else cell[[p]]
                    area = 0.0
                key = (dim, tuple(sorted(map(tuple, np.round(cell/eps).astype(np.int64)))))
                if key in seen:
                    continue
                seen.add(key)
                world = origin+height*n+cell @ basis.T
                result['cells_world_m'].append(world.tolist())
                result['cell_dimensions'].append(dim)
                result['source_prepared_face_pairs'].append([int(fa[i]), int(fb[j])])
                result['area_m2'] += area
                result.update(status='touching', proof=proof, reason='separating_plane_and_surface_intersection')
                if first_only:
                    result['timing_s'] = perf_counter()-started
                    return result
        if result['status'] == 'touching':
            result.update(complete=True, timing_s=perf_counter()-started)
            return result
    result['timing_s'] = perf_counter()-started
    return result
