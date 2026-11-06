"""CPU triangle BVH and exact primitive witnesses in double precision."""
from dataclasses import dataclass
import heapq
import numpy as np
from ..model import readonly


class BudgetExceeded(RuntimeError):
    pass


class QueryBudget:
    """Shared upper bound on primitive triangle tests within a query."""
    def __init__(self, max_tests=300000, *, parent=None):
        if not isinstance(max_tests, int) or max_tests < 1:
            raise ValueError('max_tests must be a positive integer')
        self.max_tests, self.used = max_tests, 0
        self.parent = parent

    @property
    def remaining(self):
        local=self.max_tests-self.used
        return min(local,self.parent.remaining) if self.parent is not None else local

    def consume(self, count=1):
        if self.used + count > self.max_tests:
            raise BudgetExceeded('Triangle-query budget exhausted')
        if self.parent is not None:
            self.parent.consume(count)
        self.used += count


def aabb_distance(lo_a, hi_a, lo_b, hi_b):
    """Euclidean lower bound between axis-aligned boxes."""
    return float(np.linalg.norm(np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0)))


def closest_on_triangle(point, triangle):
    """Closest point, including face interior, edges and degenerate triangles."""
    a, b, c = triangle
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    n2 = np.dot(n, n)
    if n2 > np.finfo(float).tiny:
        q = point - np.dot(point - a, n) / n2 * n
        # Oriented edge tests avoid unstable barycentric subtraction.
        if all(np.dot(np.cross(y - x, q - x), n) >= -n2 * 1e-14
               for x, y in ((a, b), (b, c), (c, a))):
            return q
    starts = triangle
    ends = np.roll(triangle, -1, axis=0)
    delta = ends - starts
    length2 = np.einsum('ij,ij->i', delta, delta)
    t = np.divide(np.einsum('ij,ij->i', point - starts, delta), length2,
                  out=np.zeros(3), where=length2 > 0)
    qs = starts + np.clip(t, 0, 1)[:, None] * delta
    return qs[np.argmin(np.linalg.norm(qs - point, axis=1))]


def segment_pair(p, q, r, s):
    """Closest points on two closed segments, including parallel/zero-length."""
    d1, d2, v = q - p, s - r, p - r
    a, e = np.dot(d1, d1), np.dot(d2, d2)
    if a <= np.finfo(float).tiny:
        t = np.clip(np.dot(d2, p - r) / e, 0, 1) if e else 0
        return p, r + t * d2
    if e <= np.finfo(float).tiny:
        t = np.clip(np.dot(d1, r - p) / a, 0, 1)
        return p + t * d1, r
    b, c, f = np.dot(d1, d2), np.dot(d1, v), np.dot(d2, v)
    denom = a * e - b * b
    u = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-14 * a * e else 0.0
    t = (b * u + f) / e
    if t < 0:
        t, u = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, u = 1.0, np.clip((b - c) / a, 0, 1)
    return p + u * d1, r + t * d2


def triangle_pair(a, b, tol=1e-10):
    """Minimum-distance witnesses and a strict transverse-intersection flag.

    Edge/face intersections are checked explicitly; vertex/face plus edge/edge
    distances alone miss an edge piercing the interior of a triangle.
    """
    witnesses = []
    transverse = False
    for source, target, reverse in ((a, b, False), (b, a, True)):
        n = np.cross(target[1] - target[0], target[2] - target[0])
        size = np.linalg.norm(n)
        if size:
            n /= size
            d = (source - target[0]) @ n
            for i in range(3):
                j = (i + 1) % 3
                if d[i] * d[j] < 0:
                    p = source[i] + (source[j] - source[i]) * d[i] / (d[i] - d[j])
                    q = closest_on_triangle(p, target)
                    if np.linalg.norm(p - q) <= tol:
                        margins = [np.dot(np.cross(y - x, p - x), n) / max(np.linalg.norm(y - x), tol)
                                   for x, y in zip(target, np.roll(target, -1, axis=0))]
                        transverse |= min(margins) > tol and min(abs(d[i]), abs(d[j])) > tol
                        witnesses.append((p, p))
        for p in source:
            q = closest_on_triangle(p, target)
            witnesses.append((q, p) if reverse else (p, q))
    for i in range(3):
        for j in range(3):
            witnesses.append(segment_pair(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]))
    pa, pb = min(witnesses, key=lambda pair: np.dot(pair[0] - pair[1], pair[0] - pair[1]))
    return pa, pb, transverse


@dataclass(frozen=True)
class _Node:
    lo: np.ndarray
    hi: np.ndarray
    ids: np.ndarray | None
    left: int = -1
    right: int = -1

    def __post_init__(self):
        object.__setattr__(self,'lo',readonly(self.lo))
        object.__setattr__(self,'hi',readonly(self.hi))
        if self.ids is not None:
            object.__setattr__(self,'ids',readonly(self.ids,np.int64))


class MeshBVH:
    """Median-split local mesh BVH; placement leaves the cached mesh untouched."""
    def __init__(self, mesh, leaf_size=8):
        self.mesh, self.nodes = mesh, []
        triangles = mesh.vertices[mesh.faces]
        lows, highs = triangles.min(axis=1), triangles.max(axis=1)
        centers = (lows + highs) / 2

        def build(ids):
            index = len(self.nodes)
            lo, hi = lows[ids].min(axis=0), highs[ids].max(axis=0)
            self.nodes.append(None)
            if len(ids) > leaf_size:
                axis = np.argmax(np.ptp(centers[ids], axis=0))
                ids = ids[np.argsort(centers[ids, axis], kind='stable')]
                mid = len(ids) // 2
                left, right = build(ids[:mid]), build(ids[mid:])
                self.nodes[index] = _Node(lo,hi,None,left,right)
            else:
                self.nodes[index] = _Node(lo,hi,ids)
            return index
        build(np.arange(len(mesh.faces)))
        self.nodes = tuple(self.nodes)

    def placed(self, tf):
        """Build inexpensive world bounds for a rigid placement, without refit."""
        return PlacedBVH(self, tf)


class PlacedBVH:
    def __init__(self, bvh, tf):
        self.bvh = bvh
        self.vertices = bvh.mesh.vertices @ tf[:3, :3].T + tf[:3, 3]
        self.triangles = self.vertices[bvh.mesh.faces]
        self.tri_lo, self.tri_hi = self.triangles.min(axis=1), self.triangles.max(axis=1)
        lo = np.asarray([n.lo for n in bvh.nodes])
        hi = np.asarray([n.hi for n in bvh.nodes])
        center = ((lo + hi) / 2) @ tf[:3, :3].T + tf[:3, 3]
        extent = ((hi - lo) / 2) @ np.abs(tf[:3, :3]).T
        self.lo, self.hi = center - extent, center + extent

    def closest(self, point, budget):
        """Exact nearest witness if budget permits; raises on incomplete search."""
        heap = [(aabb_distance(point, point, self.lo[0], self.hi[0]), 0)]
        best, witness, face_id = np.inf, None, -1
        while heap:
            lower, i = heapq.heappop(heap)
            if lower > best:
                break
            node = self.bvh.nodes[i]
            if node.ids is None:
                for j in (node.left, node.right):
                    lb = aabb_distance(point, point, self.lo[j], self.hi[j])
                    if lb <= best:
                        heapq.heappush(heap, (lb, j))
            else:
                for fid in node.ids:
                    if aabb_distance(point,point,self.tri_lo[fid],self.tri_hi[fid]) > best:
                        continue
                    budget.consume()
                    q = closest_on_triangle(point, self.triangles[fid])
                    d = np.linalg.norm(point - q)
                    if d < best:
                        best, witness, face_id = d, q, int(fid)
        return float(best), witness, face_id


def node_pairs(a, b, max_distance=np.inf):
    """Yield leaf pairs in increasing box-bound order, without an F×F array."""
    heap = [(aabb_distance(a.lo[0], a.hi[0], b.lo[0], b.hi[0]), 0, 0)]
    while heap:
        lower, i, j = heapq.heappop(heap)
        if lower > max_distance:
            break
        na, nb = a.bvh.nodes[i], b.bvh.nodes[j]
        if na.ids is not None and nb.ids is not None:
            yield lower, na.ids, nb.ids
        else:
            split_a = nb.ids is not None or (na.ids is None and np.linalg.norm(a.hi[i]-a.lo[i]) >= np.linalg.norm(b.hi[j]-b.lo[j]))
            children = [(k, j) for k in (na.left, na.right)] if split_a else [(i, k) for k in (nb.left, nb.right)]
            for x, y in children:
                lb = aabb_distance(a.lo[x], a.hi[x], b.lo[y], b.hi[y])
                if lb <= max_distance:
                    heapq.heappush(heap, (lb, x, y))
