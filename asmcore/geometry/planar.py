"""Plane charts and convex cell clipping; holes are represented by cell unions."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree


def plane_basis(normal):
    """Return a (3,2) right-handed orthonormal plane basis, u cross v = n."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(axis, n)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(n, u)))


def clean_polygon(poly, tol):
    """Remove adjacent coincident vertices without filling concavities."""
    out = []
    for p in np.asarray(poly):
        if not out or np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.asarray(out, dtype=float).reshape(-1, 2)


def polygon_measure(poly):
    """Return signed area (m²) and centroid (m), including degenerate cells."""
    p = np.asarray(poly)
    if not len(p):
        return 0.0, np.zeros(2)
    if len(p) < 3:
        return 0.0, p.mean(axis=0)
    # Centering avoids cancellation for small regions far from chart origin.
    origin = p[0]
    q = p - origin
    nxt = np.roll(q, -1, axis=0)
    cross = q[:, 0] * nxt[:, 1] - q[:, 1] * nxt[:, 0]
    area = cross.sum() * 0.5
    if abs(area) <= np.finfo(float).tiny:
        return 0.0, p.mean(axis=0)
    center = ((q + nxt) * cross[:, None]).sum(axis=0) / (6 * area)
    return float(area), center + origin


def clip_halfplane(poly, normal, offset, tol):
    """Clip a convex cell to dot(p, normal)+offset >= 0 (chart metres)."""
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    if not len(p):
        return p
    vals = p @ normal + offset
    vals[np.abs(vals) <= tol] = 0
    out = []
    for i in range(len(p)):
        a, b = p[i - 1], p[i]
        da, db = vals[i - 1], vals[i]
        if (da >= 0) != (db >= 0):
            out.append(a + (b - a) * da / (da - db))
        if db >= 0:
            out.append(b)
    return clean_polygon(out, tol)


def convex_intersection(a, b, tol):
    """Intersect two convex polygons, retaining line and point intersections."""
    out = clean_polygon(a, tol)
    b = np.asarray(b)
    if polygon_measure(b)[0] < 0:
        b = b[::-1]
    for i in range(len(b)):
        edge = b[(i + 1) % len(b)] - b[i]
        length = np.linalg.norm(edge)
        if length <= tol:
            continue
        n = np.array([-edge[1], edge[0]]) / length
        out = clip_halfplane(out, n, -np.dot(n, b[i]), tol)
        if not len(out):
            break
    return out


def trace_edges(edges):
    """Trace oriented index edges into every closed ring; report bad topology."""
    outgoing = defaultdict(list)
    for a, b in edges:
        outgoing[a].append(b)
    if any(len(v) != 1 for v in outgoing.values()):
        return (), False
    remaining = set(edges)
    loops = []
    while remaining:
        start, cur = min(remaining)
        loop = [start]
        remaining.remove((start, cur))
        while cur != start:
            loop.append(cur)
            if cur not in outgoing or (cur, outgoing[cur][0]) not in remaining:
                return tuple(loops), False
            nxt = outgoing[cur][0]
            remaining.remove((cur, nxt))
            cur = nxt
        loops.append(tuple(loop))
    return tuple(loops), True


def _weld_vertices(vertices: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign first-occurrence representatives by distance, without chaining.

    Rounded grid keys split identical intersections at half-grid boundaries.
    Exact duplicates are collapsed first; a batched nearest-neighbor query
    limits radius searches to vertices that actually have a nearby neighbor.
    Every merged point stays within tol of its own representative, including
    across rounding boundaries. No dense points-by-points array is allocated.
    """
    _, first, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    indexed = remap[inverse]
    points = vertices[first[order]]
    representative = np.arange(len(points))
    if len(points) > 1:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2, distance_upper_bound=np.nextafter(tol, np.inf))
        for i in np.flatnonzero(distances[:, 1] <= tol):
            if representative[i] != i:
                continue
            neighbors = np.asarray(tree.query_ball_point(points[i], tol), dtype=np.int64)
            unassigned = neighbors[representative[neighbors] == neighbors]
            representative[unassigned] = i
    kept, compact = np.unique(representative, return_inverse=True)
    return compact[indexed], points[kept]


def cell_regions(cells, tol):
    """Group oriented 2D/3D cells and recover holes, splitting T junctions.

    Returns (groups, valid), where each group is (cell_indices, boundary_loops).
    The original cells remain authoritative even if a boundary is ambiguous.
    Memory is linear in cell vertices; no all-pairs matrix is constructed.
    """
    if not cells:
        return (), True
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('Cell topology tolerance must be positive and finite')
    oriented = [c[::-1] if np.shape(c)[1] == 2 and polygon_measure(c)[0] < 0 else c for c in cells]
    lengths = np.asarray([len(c) for c in oriented])
    flat = np.concatenate(oriented)
    indexed, points = _weld_vertices(flat, tol)
    tree = cKDTree(points)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    ends = np.cumsum(lengths)-1
    next_index = np.arange(len(flat))+1
    next_index[ends] = starts
    a, b = indexed, indexed[next_index]
    delta = points[b]-points[a]
    length2 = np.einsum('ij,ij->i', delta, delta)
    keep = length2 > tol*tol
    owners = np.repeat(np.arange(len(cells)), lengths)[keep]
    a, b, delta, length2 = a[keep], b[keep], delta[keep], length2[keep]
    lengths_m = np.sqrt(length2)
    edge_owners = defaultdict(list)
    cell_edges = [[] for _ in cells]
    # Query sparse neighborhoods in bounded batches, then project all returned
    # candidates together. T-junction splitting still uses the same tolerance;
    # no dense edges-by-vertices matrix and no boundary smoothing are involved.
    for start in range(0, len(a), 256):
        end = min(start+256, len(a))
        nearby = tree.query_ball_point((points[a[start:end]]+points[b[start:end]])/2,
                                      lengths_m[start:end]/2+2*tol, return_sorted=True)
        edge_ids = np.repeat(np.arange(start, end), [len(n) for n in nearby])
        vertex_ids = np.concatenate(nearby).astype(np.int64)
        d = points[vertex_ids]-points[a[edge_ids]]
        ts = np.einsum('ij,ij->i', d, delta[edge_ids])/length2[edge_ids]
        dist = np.linalg.norm(d-ts[:, None]*delta[edge_ids], axis=1)
        # A vertex farther than the declared topology tolerance is a distinct
        # boundary feature. The previous 2*tol accepted a neighboring corner
        # as a T junction and could branch an otherwise valid polygon ring.
        mask = ((ts >= -tol/lengths_m[edge_ids]) & (ts <= 1+tol/lengths_m[edge_ids]) & (dist <= tol))
        edge_ids, vertex_ids, ts = edge_ids[mask], vertex_ids[mask], ts[mask]
        order = np.lexsort((ts, edge_ids))
        edge_ids, vertex_ids = edge_ids[order], vertex_ids[order]
        x, y = vertex_ids[:-1], vertex_ids[1:]
        link = ((edge_ids[:-1] == edge_ids[1:]) & (x != y)
                & (np.linalg.norm(points[x]-points[y], axis=1) > tol*.5))
        for cid, x, y in zip(owners[edge_ids[:-1][link]].tolist(), x[link].tolist(), y[link].tolist()):
            cell_edges[cid].append((x, y))
            edge_owners[(min(x, y), max(x, y))].append(cid)
    parent = list(range(len(cells)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for owners in edge_owners.values():
        for other in owners[1:]:
            parent[find(other)] = find(owners[0])
    components = defaultdict(list)
    for i in range(len(cells)):
        components[find(i)].append(i)
    groups, valid = [], all(len(owners) <= 2 for owners in edge_owners.values())
    for members in components.values():
        boundary = []
        for i in members:
            boundary.extend(e for e in cell_edges[i] if len(edge_owners[tuple(sorted(e))]) == 1)
        loops, ok = trace_edges(boundary)
        valid &= ok
        groups.append((tuple(members), tuple(points[list(loop)] for loop in loops)))
    return tuple(groups), valid
