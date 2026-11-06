"""Array kernels for SDF integration; loops run over clipping planes, not cells."""
import numpy as np


def split_triangles(triangles):
    """Return (N,2,3,3) children using winding-preserving longest-edge bisection."""
    edges = np.roll(triangles, -1, axis=1)-triangles
    index = np.argmax(np.einsum('nij,nij->ni', edges, edges), axis=1)
    rows = np.arange(len(triangles))
    a, b, c = (triangles[rows, (index+k) % 3] for k in range(3))
    mid = (a+b)/2
    return np.stack((np.stack((a, mid, c), axis=1), np.stack((mid, b, c), axis=1)), axis=1)


def clip_cells(triangles, scores):
    """Clip N triangles by three vertex-linear inequalities (N,3,3).

    Output polygons have at most six vertices, padded to (N,6,3). Counts mark
    live vertices; padding must never be interpreted as part of a polygon.
    """
    n = len(triangles)
    bary = np.zeros((n, 6, 3))
    bary[:, :3] = np.eye(3)
    counts = np.full(n, 3)
    rows, slots = np.arange(n)[:, None], np.arange(6)[None, :]
    for plane in range(3):
        live = slots < counts[:, None]
        previous = np.where(slots == 0, np.maximum(counts[:, None]-1, 0), slots-1)
        a = bary[rows, previous]
        db = np.einsum('nvi,ni->nv', bary, scores[:, plane])
        da = np.einsum('nvi,ni->nv', a, scores[:, plane])
        crossing = live & ((da >= 0) != (db >= 0))
        fraction = np.divide(da, da-db, out=np.zeros_like(da), where=crossing)
        intersections = a+(bary-a)*fraction[:, :, None]
        candidates = np.stack((intersections, bary), axis=2).reshape(n, 12, 3)
        keep = np.stack((crossing, live & (db >= 0)), axis=2).reshape(n, 12)
        destinations = np.cumsum(keep, axis=1)-1
        rr, cc = np.nonzero(keep)
        out = np.zeros_like(bary)
        out[rr, destinations[rr, cc]] = candidates[rr, cc]
        counts = keep.sum(axis=1)
        counts[counts < 3] = 0
        bary = out
    return bary @ triangles, counts


def measure_cells(polygons, counts):
    """Compute polygon areas and centroids in one padded triangle-fan batch."""
    a = polygons[:, :1]
    b, c = polygons[:, 1:-1], polygons[:, 2:]
    weights = np.linalg.norm(np.cross(b-a, c-a), axis=2)/2
    weights *= np.arange(2, polygons.shape[1])[None, :] < counts[:, None]
    area = weights.sum(axis=1)
    center = np.einsum('ni,nij->nj', weights, (a+b+c)/3)
    center /= np.maximum(area[:, None], np.finfo(float).tiny)
    return area, center
