"""Candidate domains D_k and the discrete staging grid / occupancy bitmap.

Two domain models (hybrid, per the locked decision):

  * CONTINUOUS -- used by the anytime beam. Reuses the GA driver's
    ``_enumerate_xy_candidates`` (home-collision / base-overlap / keepout
    pre-excluded) + farthest-point spreading.

  * DISCRETE -- used by the exact A*/branch-and-bound solver. A single global
    staging grid shared by all parts so that occupancy bitmaps and Hall matching
    (see ``pruning``) are well defined. Each part's domain is the subset of grid
    cells that pass its own cheap pre-exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from sealp.examples.layout.infer_assembly_ga import (
    _enumerate_xy_candidates,
    _farthest_point_order,
)


# ---- continuous domain (beam) ------------------------------------
def continuous_domain(searcher, pid: str, spacing: float, cap: int,
                      preassembled_pid: Optional[str]) -> List[np.ndarray]:
    xy_list = _enumerate_xy_candidates(searcher, pid, spacing,
                                       first_pid=preassembled_pid)
    if not xy_list:
        return []
    idx = _farthest_point_order(xy_list, max(int(cap), 1))
    return [xy_list[i] for i in idx]


# ---- discrete staging grid (exact) -------------------------------
@dataclass
class StagingGrid:
    xs: np.ndarray
    ys: np.ndarray
    top_z: float
    step: float
    cell_xy: List[np.ndarray] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cell_xy)

    def nearest_cell(self, xy: Sequence[float]) -> int:
        p = np.asarray(xy, dtype=float)[:2]
        d = [float(np.linalg.norm(c[:2] - p)) for c in self.cell_xy]
        return int(np.argmin(d)) if d else -1

    def footprint_mask(self, cell_idx: int, radius: float) -> int:
        """Bitmask of all cells whose centre lies within ``radius`` of the cell."""
        c = self.cell_xy[cell_idx][:2]
        mask = 0
        for j, q in enumerate(self.cell_xy):
            if float(np.linalg.norm(q[:2] - c)) <= radius:
                mask |= (1 << j)
        return mask


def build_staging_grid(searcher, step: float) -> StagingGrid:
    """Global grid over the usable table region (shrunk by the table margin)."""
    xr = np.asarray(searcher.table_x_range, dtype=float)
    yr = np.asarray(searcher.table_y_range, dtype=float)
    m = float(getattr(searcher, "table_margin", 0.06))
    xlo, xhi = float(xr[0]) + m, float(xr[1]) - m
    ylo, yhi = float(yr[0]) + m, float(yr[1]) - m
    step = max(float(step), 0.02)
    nx = max(1, int(np.floor((xhi - xlo) / step)) + 1)
    ny = max(1, int(np.floor((yhi - ylo) / step)) + 1)
    xs = np.linspace(xlo, xhi, nx)
    ys = np.linspace(ylo, yhi, ny)
    top_z = float(searcher.table_top_z)
    cells = [np.array([float(x), float(y)], dtype=float) for y in ys for x in xs]
    return StagingGrid(xs=xs, ys=ys, top_z=top_z, step=step, cell_xy=cells)


def discrete_domain(searcher, pid: str, grid: StagingGrid, spacing: float,
                    preassembled_pid: Optional[str]) -> List[int]:
    """Cell indices of ``grid`` that pass part ``pid``'s cheap pre-exclusions.

    We reuse the continuous enumerator (which already applies keepout /
    home-collision / base-overlap filters) and snap each surviving xy to its
    nearest grid cell, deduplicating.
    """
    xy_list = _enumerate_xy_candidates(searcher, pid, spacing,
                                       first_pid=preassembled_pid)
    seen: Dict[int, None] = {}
    for xy in xy_list:
        idx = grid.nearest_cell(xy)
        if idx >= 0:
            seen[idx] = None
    return list(seen.keys())


class OccupancyBitmap:
    """Fast integer-mask occupancy over the shared staging grid."""

    def __init__(self, grid: StagingGrid):
        self.grid = grid
        self.mask = 0

    def clone(self) -> "OccupancyBitmap":
        o = OccupancyBitmap(self.grid)
        o.mask = self.mask
        return o

    def free_cells(self) -> List[int]:
        return [j for j in range(self.grid.n) if not (self.mask >> j) & 1]

    def is_free(self, footprint_mask: int) -> bool:
        return (self.mask & footprint_mask) == 0

    def occupy(self, footprint_mask: int) -> None:
        self.mask |= footprint_mask
