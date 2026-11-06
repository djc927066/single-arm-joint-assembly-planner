"""Dependency-free STL reader replacing wrs.geom.loader for the vendored core."""
from __future__ import annotations
import numpy as np
from pathlib import Path


def read_stl_arrays(path, dtype=np.float64):
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2048]:
        pts = [list(map(float, ln.split()[1:4]))
               for ln in raw.decode("utf-8", "replace").splitlines()
               if ln.strip().startswith("vertex")]
        tri = np.asarray(pts, dtype=dtype).reshape(-1, 3)
    else:
        n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        rec = np.frombuffer(raw[84:84 + n * 50], dtype=np.dtype([
            ("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
        tri = rec["v"].reshape(-1, 3).astype(dtype)
    verts, inv = np.unique(np.round(tri, 9), axis=0, return_inverse=True)
    return verts.astype(dtype), inv.reshape(-1, 3).astype(np.int64)
