"""Shelf Unit 几何常量（与 STL 朝向对齐）
========================================

局部坐标与 goal 位姿由 ``mesh_frames`` 从实际 STL AABB 推导。
侧板 mesh 底面在 local z=0；层板 mesh 以几何中心为原点。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from sealp.assets.models.shelf_unit.mesh_frames import (
    FIXTURE_POS,
    FIXTURE_ROTMAT,
    GOAL_REL_POS,
    GOAL_REL_ROTMAT,
    SHELF_FRAME,
    SHELF_STAGING_ROTMAT,
    SHELF_STAGING_Z_OFFSET,
    SHELF_STL,
    SIDE_FRAME,
    SIDE_STL,
    SIDE_UPRIGHT_Z_OFFSET,
    TABLE_TOP_Z,
)

SHELF_X, SHELF_Y, SHELF_Z = SHELF_FRAME.extent.tolist()
SIDE_X, SIDE_Y, SIDE_Z = SIDE_FRAME.extent.tolist()

SHELF_HALF = SHELF_FRAME.extent / 2.0
SIDE_HALF = SIDE_FRAME.extent / 2.0

FIXTURE_SUPPORT_Z = TABLE_TOP_Z

ROBOT_BASE_POS = np.zeros(3)
ROBOT_BASE_ROTMAT = np.eye(3)
DUAL_ARM_Y_OFFSET = 0.62

# 兼容旧名
SHELF_UPRIGHT_RX90 = SHELF_STAGING_ROTMAT.copy()
SHELF_UPRIGHT_Z_OFFSET = float(SHELF_STAGING_Z_OFFSET)

# ── 布局搜索 seed / bounds（xy；z 由 rotmat+z_offset 决定）──────────────
STAGING_SEEDS: Dict[str, np.ndarray] = {
    "side_l":  np.array([0.52,  0.10, 0.0]),
    "side_r":  np.array([0.52, -0.55, 0.0]),
    "shelf_m": np.array([0.35, -0.20, 0.0]),
    "shelf_t": np.array([0.45, -0.35, 0.0]),
}

# 默认由 ``part_xy_bounds_on_work_table()`` 从 sample_config 的 work_table 推导；
# 仅当无法读 config 时作 fallback。
PART_XY_BOUNDS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "side_l":  ((0.005, 0.695), (-0.89, 0.19)),
    "side_r":  ((0.005, 0.695), (-0.89, 0.19)),
    "shelf_m": ((0.005, 0.695), (-0.89, 0.19)),
    "shelf_t": ((0.005, 0.695), (-0.89, 0.19)),
}

WORK_TABLE_NAME = "work_table"
DEFAULT_TABLE_MARGIN = 0.06

# 左臂 + fixture@x=0.5：过深 -Y staging 常见 no_common / traj 失败
STAGING_Y_CLIP = (-0.55, 0.15)


def work_table_xy_bounds(
    config_yaml: str,
    *,
    table_name: str = WORK_TABLE_NAME,
    margin: float = DEFAULT_TABLE_MARGIN,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """从 ``sample_config.yaml`` 的 box 障碍推桌面 XY 采样范围（留边 ``margin``）。"""
    from sealp.config import load_config
    from sealp.layout.dual_staging_search import find_obstacle_def

    cfg = load_config(config_yaml)
    table = find_obstacle_def(cfg.obstacle_defs, table_name)
    if table is None or table.get("type") != "box":
        raise ValueError(f"obstacle {table_name!r} not found or not box in {config_yaml}")
    tp, te = table["pos"], table["extent"]
    x_lo = float(tp[0]) - float(te[0]) / 2.0 + float(margin)
    x_hi = float(tp[0]) + float(te[0]) / 2.0 - float(margin)
    y_lo = float(tp[1]) - float(te[1]) / 2.0 + float(margin)
    y_hi = float(tp[1]) + float(te[1]) / 2.0 - float(margin)
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError(
            f"work_table sample range empty (margin={margin}); "
            f"x=({x_lo},{x_hi}) y=({y_lo},{y_hi})"
        )
    return (x_lo, x_hi), (y_lo, y_hi)


def part_xy_bounds_on_work_table(
    part_ids: Tuple[str, ...],
    config_yaml: str,
    *,
    table_name: str = WORK_TABLE_NAME,
    margin: float = DEFAULT_TABLE_MARGIN,
) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """每件零件 staging (x,y) 均在整张 work_table 内均匀采样。"""
    xy = work_table_xy_bounds(config_yaml, table_name=table_name, margin=margin)
    return {pid: xy for pid in part_ids}


def part_xy_bounds_shelf_reachable(
    part_ids: Tuple[str, ...],
    config_yaml: str,
    *,
    table_name: str = WORK_TABLE_NAME,
    margin: float = DEFAULT_TABLE_MARGIN,
    y_clip: Tuple[float, float] = STAGING_Y_CLIP,
) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """work_table 范围再收紧 Y，避免采样到左臂难达远端。"""
    raw = part_xy_bounds_on_work_table(
        part_ids, config_yaml, table_name=table_name, margin=margin)
    y_lo_clip, y_hi_clip = float(y_clip[0]), float(y_clip[1])
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for pid, ((x_lo, x_hi), (y_lo, y_hi)) in raw.items():
        y_lo2 = max(float(y_lo), y_lo_clip)
        y_hi2 = min(float(y_hi), y_hi_clip)
        if y_hi2 <= y_lo2:
            raise ValueError(
                f"staging y range empty for {pid!r} after clip {y_clip}; "
                f"table y=({y_lo},{y_hi})"
            )
        out[pid] = ((x_lo, x_hi), (y_lo2, y_hi2))
    return out

# auto_rotmat 缓存版本（改 fixture / mesh 后 bump）
AUTO_ROTMAT_CACHE_TAG = "shelf_unit_fixture50_v5"


def default_seed_staging_pose(part_id: str, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """预搜索失败时的 staging 默认位姿 (pos, rotmat)。

    侧板竖立 ``I``、底面贴桌；层板 ``Rx(+90°)`` 立边，避免平放时夹爪碰桌。
    """
    xy = np.asarray(seed, dtype=float)[:2]
    if part_id.startswith("side_"):
        return (
            np.array([xy[0], xy[1], SIDE_UPRIGHT_Z_OFFSET], dtype=float),
            np.eye(3),
        )
    if part_id.startswith("shelf_"):
        R = SHELF_STAGING_ROTMAT.copy()
        z = float(SHELF_STAGING_Z_OFFSET)
        return np.array([xy[0], xy[1], z], dtype=float), R
    raise KeyError(f"unknown shelf_unit part: {part_id!r}")
