# -*- coding: utf-8 -*-
"""
把 work_table 顶面均分为 3x3 区域并用不同颜色可视化
====================================================

只做展示: 从配置文件里读取名为 ``work_table`` 的 box 障碍物 (extent/pos),
把它的**顶面**在 x-y 方向均分成 3x3 共 9 块, 每块用不同颜色画一层薄片叠在
桌面上, 便于直观看清"装配区候选"是怎么按网格划分的。

不做任何规划/控制, 只开一个 WRS 窗口显示。

用法
----
# 用默认配置 (sample_config.yaml):
python -m sealp.examples.layout.visualize_table_grid

# 指定别的配置文件:
python -m sealp.examples.layout.visualize_table_grid --config D:/path/to/your_config.yaml

# 改变网格数 (例如 4x4), 或调可视化参数:
python -m sealp.examples.layout.visualize_table_grid --rows 3 --cols 3 --gap 0.004
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

# 让脚本既能 `python -m ...` 也能直接 `python 路径/this.py` 运行。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, mgm  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG = os.path.abspath(
    os.path.join(_REPO_ROOT, "sealp", "config", "sample_config.yaml"))

# 9 种区分度较高的颜色 (RGB, 0-1)。够 3x3 用; 更大网格会循环取用。
_PALETTE = [
    (0.90, 0.30, 0.30),  # 红
    (0.95, 0.65, 0.20),  # 橙
    (0.95, 0.90, 0.25),  # 黄
    (0.40, 0.80, 0.35),  # 绿
    (0.25, 0.75, 0.75),  # 青
    (0.30, 0.55, 0.90),  # 蓝
    (0.55, 0.40, 0.85),  # 紫
    (0.95, 0.55, 0.80),  # 粉
    (0.55, 0.55, 0.55),  # 灰
]


def _load_table_box(config_path: str, name: str = "work_table"):
    """从 config 的 environment.obstacles 里取出指定 box 的 extent/pos。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    obstacles = (((cfg or {}).get("environment") or {}).get("obstacles")) or []
    for obs in obstacles:
        if obs.get("name") == name and obs.get("type") == "box":
            extent = np.asarray(obs["extent"], dtype=float)
            pos = np.asarray(obs["pos"], dtype=float)
            rgba = obs.get("rgba", [0.55, 0.45, 0.35, 0.8])
            return extent, pos, rgba
    raise ValueError(
        f"在 {config_path} 的 environment.obstacles 里找不到名为 {name!r} 的 box。")


def build_grid_tiles(extent, pos, rows: int, cols: int, *,
                     gap: float, tile_thickness: float, z_lift: float):
    """生成 rows x cols 个薄片 (每格一个) 的参数。

    返回 ``[(rc, center_pos, tile_extent, rgb)]``, rc=(行索引, 列索引)。
    行沿 x 方向, 列沿 y 方向 (与 layout 里 3x3 装配区网格一致)。
    """
    ex, ey, ez = float(extent[0]), float(extent[1]), float(extent[2])
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])

    # 桌面顶面 z, 薄片叠在其上方一点点。
    top_z = pz + ez / 2.0 + z_lift + tile_thickness / 2.0

    cell_x = ex / rows
    cell_y = ey / cols
    # 桌面在 x/y 上的起始边 (角点)。
    x0 = px - ex / 2.0
    y0 = py - ey / 2.0

    tiles = []
    for r in range(rows):
        for c in range(cols):
            cx = x0 + (r + 0.5) * cell_x
            cy = y0 + (c + 0.5) * cell_y
            tile_extent = np.array([
                max(1e-3, cell_x - gap),
                max(1e-3, cell_y - gap),
                tile_thickness,
            ])
            center = np.array([cx, cy, top_z])
            rgb = _PALETTE[(r * cols + c) % len(_PALETTE)]
            tiles.append(((r, c), center, tile_extent, rgb))
    return tiles


def main() -> None:
    ap = argparse.ArgumentParser(
        description="把 work_table 顶面均分为网格并用不同颜色可视化。")
    ap.add_argument("--config", default=_DEFAULT_CFG, help="配置文件路径 (yaml)")
    ap.add_argument("--table-name", default="work_table", help="要划分的 box 名称")
    ap.add_argument("--rows", type=int, default=3, help="沿 x 方向的行数 (默认 3)")
    ap.add_argument("--cols", type=int, default=3, help="沿 y 方向的列数 (默认 3)")
    ap.add_argument("--gap", type=float, default=0.004,
                    help="相邻格子间留的缝隙 (m), 便于肉眼区分。默认 0.004")
    ap.add_argument("--tile-thickness", type=float, default=0.004,
                    help="彩色薄片厚度 (m)。默认 0.004")
    ap.add_argument("--z-lift", type=float, default=0.001,
                    help="薄片相对桌面顶面抬起的高度 (m), 防止 z-fighting。默认 0.001")
    ap.add_argument("--alpha", type=float, default=0.85, help="薄片透明度。默认 0.85")
    ap.add_argument("--no-table", action="store_true",
                    help="不画半透明桌面本体, 只画彩色网格")
    args = ap.parse_args()

    if not os.path.isfile(args.config):
        ap.error(f"配置文件不存在: {args.config}")

    extent, pos, rgba = _load_table_box(args.config, args.table_name)
    print(f"[grid] config     = {args.config}")
    print(f"[grid] table      = {args.table_name}  extent={extent.tolist()}  pos={pos.tolist()}")
    print(f"[grid] 划分        = {args.rows} x {args.cols} = {args.rows * args.cols} 格")

    # 相机看向桌面中心。
    look = np.array([pos[0], pos[1], pos[2] + 0.1])
    base = wd.World(cam_pos=look + np.array([0.9, -1.0, 0.9]), lookat_pos=look)

    # 世界原点坐标系 (方便判断朝向)。
    mgm.gen_frame(ax_length=0.1).attach_to(base)

    # 半透明桌面本体作为背景参考。
    if not args.no_table:
        mgm.gen_box(
            xyz_lengths=np.asarray(extent, dtype=float),
            pos=np.asarray(pos, dtype=float),
            rgb=np.array(rgba[:3]),
            alpha=float(rgba[3]) if len(rgba) > 3 else 0.5,
        ).attach_to(base)

    tiles = build_grid_tiles(
        extent, pos, args.rows, args.cols,
        gap=args.gap, tile_thickness=args.tile_thickness, z_lift=args.z_lift)

    for (r, c), center, tile_extent, rgb in tiles:
        mgm.gen_box(
            xyz_lengths=tile_extent,
            pos=center,
            rgb=np.array(rgb),
            alpha=float(args.alpha),
        ).attach_to(base)
        # 每格中心放一个小坐标系, 标出该区中心位置。
        mgm.gen_frame(pos=center, ax_length=0.03).attach_to(base)
        print(f"  区块 r{r}c{c}: center=({center[0]:.4f}, {center[1]:.4f}, "
              f"{center[2]:.4f})  size=({tile_extent[0]:.3f} x {tile_extent[1]:.3f})  "
              f"rgb={tuple(round(v, 2) for v in rgb)}")

    print("[grid] 窗口已打开, 关闭窗口即退出。")
    base.run()


if __name__ == "__main__":
    main()
