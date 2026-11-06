"""Watertight meshes for unions of axis-aligned boxes minus axis-aligned holes.

Generalises triangular_frame6.pocket_plate: all box coordinates define a rectilinear
grid; a cell is solid if it lies in some solid box and in no hole; every face between a
solid and an empty cell becomes two triangles with outward winding. No booleans, no
T-junction cracks inside a face (faces are split on the shared grid), exact metres.
"""
from __future__ import annotations

import itertools

import numpy as np

from asmcore.model import MeshData

Box = tuple  # (x0, x1, y0, y1, z0, z1) in metres, part-local frame


def box(cx, cy, cz, sx, sy, sz) -> Box:
    """Centre + full size -> (x0, x1, y0, y1, z0, z1)."""
    return (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2)


def cells(solids, holes=()) -> np.ndarray:
    """Exact solid region as non-overlapping axis-aligned cells, [K, 6] (x0,x1,y0,y1,z0,z1)."""
    xs, ys, zs, solid = _grid(list(solids), list(holes))
    return np.array([(xs[i], xs[i + 1], ys[j], ys[j + 1], zs[k], zs[k + 1])
                     for i, j, k in sorted(solid)], float)


def _grid(solids, holes):
    xs = sorted({c for b in solids + holes for c in b[0:2]})
    ys = sorted({c for b in solids + holes for c in b[2:4]})
    zs = sorted({c for b in solids + holes for c in b[4:6]})

    def inside(b, p):
        return b[0] < p[0] < b[1] and b[2] < p[1] < b[3] and b[4] < p[2] < b[5]

    solid = set()
    for i, j, k in itertools.product(range(len(xs) - 1), range(len(ys) - 1), range(len(zs) - 1)):
        p = ((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, (zs[k] + zs[k + 1]) / 2)
        if any(inside(b, p) for b in solids) and not any(inside(h, p) for h in holes):
            solid.add((i, j, k))
    return xs, ys, zs, solid


def mesh(solids, holes=()) -> MeshData:
    xs, ys, zs, solid = _grid(list(solids), list(holes))
    vertices, faces, lookup = [], [], {}

    def quad(points, normal):
        pts = [np.array(p, float) for p in points]
        if np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), normal) < 0:
            pts.reverse()
        ids = []
        for p in pts:
            key = tuple(p)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(p)
            ids.append(lookup[key])
        faces.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))

    for i, j, k in sorted(solid):
        x0, x1 = xs[i:i + 2]
        y0, y1 = ys[j:j + 2]
        z0, z1 = zs[k:k + 2]
        sides = [((-1, 0, 0), [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]),
                 ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
                 ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
                 ((0, 1, 0), [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]),
                 ((0, 0, -1), [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]),
                 ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])]
        for delta, points in sides:
            if (i + delta[0], j + delta[1], k + delta[2]) not in solid:
                quad(points, delta)
    return MeshData(np.array(vertices), np.array(faces, dtype=np.int64))
