"""Shelf Unit 专用 pick / place 线性段方向
======================================

与 YuanChair（seat 用 −Z 下放 / +Z 撤离，leg 用 −X 撤离）**完全分离**。
几何约定（fixture 在装配区中心，+Y 朝左臂）::

        +Y  左臂侧
         │
  side_l │  ← 层板从 +Y 开口水平推入
         │
    ─────┼───── fixture
         │
  side_r │
         │
        -Y  右臂侧

侧板 goal 为竖直板（local Z 向上）；层板 goal 为水平板（插在两板之间）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── 抓取后：统一 +Z 抬离桌面 ─────────────────────────────────────────────
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PICK_DEPART_DIST = 0.06

# ── side_l / side_r：竖板从上方 −Z 下放，装好后沿 ±Y 外撤 ───────────────
SIDE_L_PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0], dtype=float)
SIDE_L_PLACE_APPROACH_DIST = 0.06
SIDE_L_PLACE_DEPART_DIR = np.array([0.0, 1.0, 0.0], dtype=float)
SIDE_L_PLACE_DEPART_DIST = 0.08

SIDE_R_PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0], dtype=float)
SIDE_R_PLACE_APPROACH_DIST = 0.06
SIDE_R_PLACE_DEPART_DIR = np.array([0.0, -1.0, 0.0], dtype=float)
SIDE_R_PLACE_DEPART_DIST = 0.08

# ── shelf_m / shelf_t：从 +Y 开口水平插入，沿 −Y 撤出 ───────────────────
SHELF_PLACE_APPROACH_DIR = np.array([0.0, 1.0, 0.0], dtype=float)
SHELF_PLACE_APPROACH_DIST = 0.08
SHELF_PLACE_DEPART_DIR = np.array([0.0, -1.0, 0.0], dtype=float)
SHELF_PLACE_DEPART_DIST = 0.08


@dataclass(frozen=True)
class ShelfMotionParams:
    pick_depart_dir: np.ndarray
    pick_depart_dist: float
    place_approach_dir: np.ndarray
    place_approach_dist: float
    place_depart_dir: np.ndarray
    place_depart_dist: float


def is_shelf_unit_part(part_id: str) -> bool:
    return part_id.startswith("side_") or part_id.startswith("shelf_")


def _norm_dir(v: np.ndarray) -> np.ndarray:
    d = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return d
    return d / n


def pick_depart_candidates(part_id: str, **overrides) -> List[Tuple[np.ndarray, float]]:
    """按优先级返回 pick 撤离 (方向, 距离) 候选；层板多方向 OR 通过。"""
    mp = motion_params(part_id, **overrides)
    out: List[Tuple[np.ndarray, float]] = [
        (_norm_dir(mp.pick_depart_dir), float(mp.pick_depart_dist)),
    ]
    if part_id.startswith("shelf_"):
        # 竖 staging + 邻件 staging 占位时，短距离 -Y/+X/+Z 更易过碰撞
        for d, dist in (
            (np.array([0.0, -1.0, 0.0]), 0.08),
            (np.array([0.0, -1.0, 0.0]), 0.04),
            (np.array([0.0, -1.0, 0.0]), 0.03),
            (np.array([1.0, 0.0, 0.0]), 0.06),
            (np.array([1.0, 0.0, 0.0]), 0.03),
            (np.array([0.0, 0.0, 1.0]), 0.04),
            (np.array([0.0, 0.0, 1.0]), 0.03),
            (np.array([0.0, 0.0, 1.0]), 0.02),
            (np.array([0.0, -0.7, 1.0]), 0.06),
        ):
            out.append((_norm_dir(d), float(dist)))
    return out


def place_approach_candidates(part_id: str, **overrides) -> List[Tuple[np.ndarray, float]]:
    """goal 侧 pre-approach 方向（从 goal 沿 dir 退开）候选。"""
    mp = motion_params(part_id, **overrides)
    rev = -_norm_dir(mp.place_approach_dir)
    out: List[Tuple[np.ndarray, float]] = [
        (rev.copy(), float(mp.place_approach_dist)),
    ]
    if part_id.startswith("shelf_"):
        for d, dist in (
            (rev, 0.06),
            (rev, 0.04),
            (rev, 0.03),
            (np.array([0.0, 0.0, -1.0]), 0.06),
            (np.array([0.0, 0.0, -1.0]), 0.04),
            (np.array([0.0, 0.0, -1.0]), 0.03),
            (np.array([0.0, -1.0, 0.0]), 0.04),
            (np.array([0.0, -1.0, 0.0]), 0.03),
        ):
            out.append((_norm_dir(d), float(dist)))
    return out


def motion_params(part_id: str, *,
                  pick_depart_dist: Optional[float] = None,
                  place_approach_dist: Optional[float] = None,
                  place_depart_dist: Optional[float] = None) -> ShelfMotionParams:
    """返回某 shelf 零件的 pick/place 线性段参数。"""
    pd = PICK_DEPART_DIST if pick_depart_dist is None else float(pick_depart_dist)
    if part_id == "side_l":
        return ShelfMotionParams(
            PICK_DEPART_DIR.copy(), pd,
            SIDE_L_PLACE_APPROACH_DIR.copy(),
            SIDE_L_PLACE_APPROACH_DIST if place_approach_dist is None else float(place_approach_dist),
            SIDE_L_PLACE_DEPART_DIR.copy(),
            SIDE_L_PLACE_DEPART_DIST if place_depart_dist is None else float(place_depart_dist),
        )
    if part_id == "side_r":
        return ShelfMotionParams(
            PICK_DEPART_DIR.copy(), pd,
            SIDE_R_PLACE_APPROACH_DIR.copy(),
            SIDE_R_PLACE_APPROACH_DIST if place_approach_dist is None else float(place_approach_dist),
            SIDE_R_PLACE_DEPART_DIR.copy(),
            SIDE_R_PLACE_DEPART_DIST if place_depart_dist is None else float(place_depart_dist),
        )
    if part_id.startswith("shelf_"):
        return ShelfMotionParams(
            PICK_DEPART_DIR.copy(), pd,
            SHELF_PLACE_APPROACH_DIR.copy(),
            SHELF_PLACE_APPROACH_DIST if place_approach_dist is None else float(place_approach_dist),
            SHELF_PLACE_DEPART_DIR.copy(),
            SHELF_PLACE_DEPART_DIST if place_depart_dist is None else float(place_depart_dist),
        )
    raise KeyError(f"not a shelf_unit part: {part_id!r}")


def reason_kwargs(part_id: str, **overrides) -> Dict[str, object]:
    """``pick_place_reason_common_ok`` 用的 keyword 参数字典。"""
    mp = motion_params(part_id, **overrides)
    return dict(
        pick_depart_dir=mp.pick_depart_dir,
        pick_depart_dist=mp.pick_depart_dist,
        place_approach_dir=mp.place_approach_dir,
        place_approach_dist=mp.place_approach_dist,
    )


def transport_kwargs(part_id: str, **overrides) -> Dict[str, object]:
    """``TransportPrimitive.plan`` / ``gen_pick_and_place`` 用的 keyword 参数字典。"""
    mp = motion_params(part_id, **overrides)
    return dict(
        pick_depart_direction=mp.pick_depart_dir,
        pick_depart_distance=mp.pick_depart_dist,
        place_approach_direction_list=[mp.place_approach_dir],
        place_approach_distance_list=[mp.place_approach_dist],
        place_depart_direction_list=[mp.place_depart_dir],
        place_depart_distance_list=[mp.place_depart_dist],
    )


def apply_to_plan_kwargs(kw: dict, part_id: str, **overrides) -> None:
    """就地写入 ``DualArmFallbackTransport.plan`` 的双臂 kwargs。"""
    for k, v in transport_kwargs(part_id, **overrides).items():
        kw[k] = v
