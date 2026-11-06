"""Smooth components classified by whole-region plane residual and normal spread."""
from collections import defaultdict
import numpy as np
from ..model import GeometryConfig, SurfacePatch
from .planar import plane_basis, trace_edges


def extract_surfaces(prepared_mesh, *, config=None):
    """Return plane/general patches with all boundary rings in local metres.

    Smooth adjacency proposes a region, never certifies a plane. The entire
    proposed region must pass both the plane-distance and normal-spread tests.
    Regions containing a fillet can conservatively remain general rather than
    losing their curvature through transitive normal clustering.
    """
    config = config or GeometryConfig()
    prep = prepared_mesh
    vs, fs, ns = prep.mesh.vertices, prep.mesh.faces, prep.normals
    unseen, groups = set(range(len(fs))), []
    cos_smooth = np.cos(config.smooth_angle_rad)
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue, ids = [seed], []
        while queue:
            i = queue.pop()
            ids.append(i)
            for j in prep.adjacency[i]:
                if j in unseen and np.dot(ns[i], ns[j]) >= cos_smooth:
                    unseen.remove(j)
                    queue.append(j)
        groups.append(sorted(ids))
    patches = []
    for ids in groups:
        pts = vs[np.unique(fs[ids])]
        weights = prep.areas_m2[ids]
        origin = (vs[fs[ids]].mean(axis=1) * weights[:, None]).sum(axis=0) / weights.sum()
        normal = (ns[ids] * weights[:, None]).sum(axis=0)
        normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-15 * weights.sum() else ns[ids[0]]
        residual = float(np.abs((pts - origin) @ normal).max())
        planar = residual <= config.plane_tol_m and np.all(ns[ids] @ normal >= np.cos(config.normal_angle_rad))
        edges = defaultdict(list)
        for face in fs[ids]:
            for a, b in zip(face, np.roll(face, -1)):
                edges[tuple(sorted((int(a), int(b))))].append((int(a), int(b)))
        boundary = [v[0] for v in edges.values() if len(v) == 1]
        loops, valid = trace_edges(boundary)
        source_ids = sorted(i for f in ids for i in prep.source_face_ids[f])
        patches.append(SurfacePatch('s' + str(source_ids[0]), np.asarray(ids),
                                    'plane' if planar else 'general', origin, normal,
                                    plane_basis(normal), tuple(vs[list(loop)] for loop in loops),
                                    float(weights.sum()), residual, prep.orientation_reliable and valid))
    return tuple(patches)
