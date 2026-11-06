#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate TopDownTower Mesh Assets - 0.75x + 2x Tall Posts + Aligned Sockets
============================================================================

这版结构：
1. 在原来 TopDownTower 当前尺寸基础上，整体缩小为 0.75 倍；
2. base_plate：底板仍然加厚，四角方形浅孔也同步缩小；
3. post：四根柱子共用一个 post.stl，XY 同步缩小，高度变为上一版当前高度的 2 倍，底部加粗插入 base_plate 四角孔；
4. middle_plate：中层板仍然加厚，顶面中心方形浅孔同步缩小；
5. top_cross：竖着的十字架同步缩小，底部一小段竖直插入 middle_plate 顶面的方形孔；
6. 不再生成/使用 roof_plate。

注意：
- middle_plate 的洞不是十字形洞，而是一个方形洞。
- top_cross 不是整体平放在 middle_plate 上，而是竖着插进去；插进去的是十字架下面的一小段竖直脚。
"""

from __future__ import annotations

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import wrs.basis.trimesh.creation as trm_creation
import wrs.basis.trimesh.util as trm_util


# ============================================================
# 总体缩放
# ============================================================
# 之前版本使用 SCALE = 1.5。
# 现在要求所有零件变成“当前尺寸”的 0.75 倍，所以：
#   SCALE = 1.5 * 0.75 = 1.125
# 这样 base_plate / middle_plate / post / top_cross 会同步缩小。
ORIGINAL_SCALE = 1.5
GLOBAL_SIZE_FACTOR = 0.75
SCALE = ORIGINAL_SCALE * GLOBAL_SIZE_FACTOR

# ============================================================
# 额外缩小 (本次改动)
# ============================================================
# 需求: 不改单位(仍为 m), 只把所有部件尺寸整体缩小为原来的 2/3, 即 ÷1.5。
# 对最终网格统一缩放, 顶点/间隙/插脚等全部按比例同步缩小:
#   EXPORT_UNIT_SCALE = 1 / 1.5
# 例: base_plate 现宽 0.2925 m -> /1.5 = 0.195 m。
EXTRA_SHRINK_DIVISOR = 1.5
EXPORT_UNIT_SCALE = 1.0 / EXTRA_SHRINK_DIVISOR

# 导出目录: 覆盖现有 Toy/model 下的 STL。
_OUT_DIR = os.path.join(_HERE, "model")


# ============================================================
# 原始尺寸，单位 m
# ============================================================

BASE_XYZ_ORG = np.array([0.260, 0.200, 0.018], dtype=float)
POST_XYZ_ORG = np.array([0.026, 0.026, 0.090], dtype=float)
MIDDLE_PLATE_XYZ_ORG = np.array([0.220, 0.160, 0.016], dtype=float)

# 旧 top_cross 横放时的尺寸参考：
# long bar:  [0.090, 0.024, 0.028]
# short bar: [0.024, 0.090, 0.028]
# 新版本改成竖着的十字架：
#   - 宽度约等于旧 long bar 长度
#   - 高度约等于旧 long bar 长度
#   - 厚度/插脚宽度约等于旧 bar 的窄边
TOP_CROSS_WIDTH_ORG = 0.090
TOP_CROSS_HEIGHT_ORG = 0.090
TOP_CROSS_BAR_THICK_ORG = 0.024

POST_OFFSET_X_ORG = 0.085
POST_OFFSET_Y_ORG = 0.065


# ============================================================
# 尺寸调节参数
# ============================================================

# 之前要求：base_plate 整体厚度变为现在的 1.8 倍
BASE_THICKNESS_FACTOR = 1.8

# 之前要求：四个 base 孔洞口长宽变为柱子底面的 1.3 倍
BASE_SOCKET_XY_FACTOR = 1.3

# base 孔深度占 base 厚度比例；同步变深，默认仍取一半
BASE_SOCKET_DEPTH_RATIO = 0.50

# 新要求：四根柱子长度变成现在的 2 倍。
# 只拉高 Z，高度翻倍；XY 不变，这样孔位/插脚尺寸仍和 base_plate 对齐。
POST_HEIGHT_FACTOR = 2.0

# 之前要求：四根柱子的底部也同步增大。
# 注意：如果 POST_FOOT_X/Y 与 BASE_SOCKET_X/Y 完全相等，实际仿真插入时容易太紧。
# 这里让柱子底部脚略小于 base 孔口，留一点间隙，便于插入。
POST_FOOT_XY_FACTOR = BASE_SOCKET_XY_FACTOR
POST_FOOT_CLEARANCE = 0.002

# 新要求：middle_plate 加厚
MIDDLE_THICKNESS_FACTOR = 1.8

# 新要求：middle 顶面是一个方形浅孔，top_cross 竖着插进去一段
MIDDLE_SOCKET_CLEARANCE = 0.002
MIDDLE_SOCKET_DEPTH_RATIO = 0.50

# top_cross 竖着插入 middle 的方形孔的深度
# 为了和 middle 洞深一致，默认使用 middle 厚度的一半
# 生成 top_cross 本身时不需要单独建插销，因为竖直中柱的底部就是插入段。


# ============================================================
# 放大后的尺寸
# ============================================================

BASE_XYZ = BASE_XYZ_ORG * SCALE
BASE_XYZ[2] *= BASE_THICKNESS_FACTOR

POST_BODY_XYZ = POST_XYZ_ORG * SCALE
# 只把柱子高度变为当前 2 倍，柱子的截面尺寸仍保持 0.75 缩放后的大小。
POST_BODY_XYZ[2] *= POST_HEIGHT_FACTOR
POST_H = float(POST_BODY_XYZ[2])

MIDDLE_PLATE_XYZ = MIDDLE_PLATE_XYZ_ORG * SCALE
MIDDLE_PLATE_XYZ[2] *= MIDDLE_THICKNESS_FACTOR

POST_OFFSET_X = POST_OFFSET_X_ORG * SCALE
POST_OFFSET_Y = POST_OFFSET_Y_ORG * SCALE

BASE_H = float(BASE_XYZ[2])

POST_BODY_X = float(POST_BODY_XYZ[0])
POST_BODY_Y = float(POST_BODY_XYZ[1])

BASE_SOCKET_X = POST_BODY_X * BASE_SOCKET_XY_FACTOR
BASE_SOCKET_Y = POST_BODY_Y * BASE_SOCKET_XY_FACTOR
BASE_SOCKET_DEPTH = BASE_H * BASE_SOCKET_DEPTH_RATIO

POST_FOOT_X = max(POST_BODY_X, BASE_SOCKET_X - POST_FOOT_CLEARANCE)
POST_FOOT_Y = max(POST_BODY_Y, BASE_SOCKET_Y - POST_FOOT_CLEARANCE)
POST_FOOT_H = min(BASE_SOCKET_DEPTH, POST_H * 0.45)

MIDDLE_H = float(MIDDLE_PLATE_XYZ[2])

TOP_CROSS_WIDTH = TOP_CROSS_WIDTH_ORG * SCALE
TOP_CROSS_HEIGHT = TOP_CROSS_HEIGHT_ORG * SCALE
TOP_CROSS_BAR_THICK = TOP_CROSS_BAR_THICK_ORG * SCALE
TOP_CROSS_DEPTH = TOP_CROSS_BAR_THICK

MIDDLE_SOCKET_X = TOP_CROSS_BAR_THICK + MIDDLE_SOCKET_CLEARANCE
MIDDLE_SOCKET_Y = TOP_CROSS_BAR_THICK + MIDDLE_SOCKET_CLEARANCE
MIDDLE_SOCKET_DEPTH = MIDDLE_H * MIDDLE_SOCKET_DEPTH_RATIO


# ============================================================
# mesh helper
# ============================================================

def _box_mesh_bottom_at_z0(xyz_lengths: np.ndarray, center_xy=(0.0, 0.0)):
    """创建盒子网格，并让该零件 local 底面在 z=0。"""
    lx, ly, lz = np.asarray(xyz_lengths, dtype=float).tolist()
    mesh = trm_creation.box(extents=np.array([lx, ly, lz], dtype=float))
    mesh.vertices += np.array([center_xy[0], center_xy[1], lz / 2.0], dtype=float)
    return mesh


def _box_mesh(lx: float, ly: float, lz: float, cx: float, cy: float, cz: float):
    mesh = trm_creation.box(extents=np.array([lx, ly, lz], dtype=float))
    mesh.vertices += np.array([cx, cy, cz], dtype=float)
    return mesh


def _export_mesh(mesh, out_path: str, label: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 不改单位(仍为 m); 导出时整体缩小为 2/3 (÷1.5)。
    mesh.vertices = np.asarray(mesh.vertices, dtype=float) * EXPORT_UNIT_SCALE
    mesh.export(out_path)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(
        f"[OK] {label:<14s} -> {os.path.relpath(out_path, _PROJ_ROOT)}  "
        f"n_verts={len(mesh.vertices)}  n_faces={len(mesh.faces)}  "
        f"extent(m)=[{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]"
    )


def _make_rect_layer_with_holes(
    outer_x: float,
    outer_y: float,
    z_min: float,
    z_max: float,
    holes: list[tuple[float, float, float, float]],
):
    """通过平面切分小矩形，稳定生成带矩形/方形孔的层。"""
    x_min, x_max = -outer_x / 2.0, outer_x / 2.0
    y_min, y_max = -outer_y / 2.0, outer_y / 2.0

    xs = [x_min, x_max]
    ys = [y_min, y_max]

    for cx, cy, hx, hy in holes:
        xs.extend([cx - hx, cx + hx])
        ys.extend([cy - hy, cy + hy])

    xs = sorted(set([round(v, 10) for v in xs]))
    ys = sorted(set([round(v, 10) for v in ys]))

    meshes = []
    lz = z_max - z_min
    cz = (z_min + z_max) / 2.0

    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        if xb <= xa:
            continue

        for j in range(len(ys) - 1):
            ya, yb = ys[j], ys[j + 1]
            if yb <= ya:
                continue

            cell_cx = (xa + xb) / 2.0
            cell_cy = (ya + yb) / 2.0

            inside_hole = False
            for hcx, hcy, hhx, hhy in holes:
                if (
                    hcx - hhx <= cell_cx <= hcx + hhx
                    and hcy - hhy <= cell_cy <= hcy + hhy
                ):
                    inside_hole = True
                    break

            if inside_hole:
                continue

            meshes.append(_box_mesh(xb - xa, yb - ya, lz, cell_cx, cell_cy, cz))

    if not meshes:
        raise RuntimeError("带孔层生成失败，meshes 为空。")

    return trm_util.concatenate(meshes)


def _gen_base_plate_with_square_pockets():
    """生成加厚底板，四角方形浅孔。"""
    pocket_hx = BASE_SOCKET_X / 2.0
    pocket_hy = BASE_SOCKET_Y / 2.0

    holes = [
        ( POST_OFFSET_X,  POST_OFFSET_Y, pocket_hx, pocket_hy),
        (-POST_OFFSET_X,  POST_OFFSET_Y, pocket_hx, pocket_hy),
        ( POST_OFFSET_X, -POST_OFFSET_Y, pocket_hx, pocket_hy),
        (-POST_OFFSET_X, -POST_OFFSET_Y, pocket_hx, pocket_hy),
    ]

    bottom_h = BASE_H - BASE_SOCKET_DEPTH
    bottom_solid = _box_mesh(BASE_XYZ[0], BASE_XYZ[1], bottom_h, 0.0, 0.0, bottom_h / 2.0)

    top_layer = _make_rect_layer_with_holes(
        outer_x=float(BASE_XYZ[0]),
        outer_y=float(BASE_XYZ[1]),
        z_min=BASE_H - BASE_SOCKET_DEPTH,
        z_max=BASE_H,
        holes=holes,
    )

    return trm_util.concatenate([bottom_solid, top_layer])


def _gen_post_with_enlarged_foot():
    """生成底部加粗的柱子。

    local z:
        0 ~ POST_FOOT_H               : 加粗插入脚
        POST_FOOT_H ~ POST_H          : 正常柱身
    总高度仍为 POST_H。
    """
    foot = _box_mesh(
        POST_FOOT_X,
        POST_FOOT_Y,
        POST_FOOT_H,
        0.0,
        0.0,
        POST_FOOT_H / 2.0,
    )

    body_h = POST_H - POST_FOOT_H
    if body_h <= 1e-6:
        return foot

    body = _box_mesh(
        POST_BODY_X,
        POST_BODY_Y,
        body_h,
        0.0,
        0.0,
        POST_FOOT_H + body_h / 2.0,
    )

    return trm_util.concatenate([foot, body])


def _gen_middle_plate_with_square_socket():
    """生成加厚 middle_plate，顶面中心一个方形浅孔。

    注意：这里是方形孔，不是十字形孔。
    """
    socket_hx = MIDDLE_SOCKET_X / 2.0
    socket_hy = MIDDLE_SOCKET_Y / 2.0
    holes = [(0.0, 0.0, socket_hx, socket_hy)]

    bottom_h = MIDDLE_H - MIDDLE_SOCKET_DEPTH
    bottom_solid = _box_mesh(
        MIDDLE_PLATE_XYZ[0],
        MIDDLE_PLATE_XYZ[1],
        bottom_h,
        0.0,
        0.0,
        bottom_h / 2.0,
    )

    top_layer = _make_rect_layer_with_holes(
        outer_x=float(MIDDLE_PLATE_XYZ[0]),
        outer_y=float(MIDDLE_PLATE_XYZ[1]),
        z_min=MIDDLE_H - MIDDLE_SOCKET_DEPTH,
        z_max=MIDDLE_H,
        holes=holes,
    )

    return trm_util.concatenate([bottom_solid, top_layer])


def _gen_vertical_top_cross_mesh():
    """生成竖着的十字架。

    local z 范围：0 ~ TOP_CROSS_HEIGHT
    插入 middle 方孔的是底部竖直中柱的一小段。
    """
    # 竖直中柱：负责插入 square socket
    vertical_bar = _box_mesh(
        TOP_CROSS_BAR_THICK,
        TOP_CROSS_DEPTH,
        TOP_CROSS_HEIGHT,
        0.0,
        0.0,
        TOP_CROSS_HEIGHT / 2.0,
    )

    # 横向臂：位于上半部分，形成十字形
    horizontal_z = TOP_CROSS_HEIGHT * 0.62
    horizontal_bar = _box_mesh(
        TOP_CROSS_WIDTH,
        TOP_CROSS_DEPTH,
        TOP_CROSS_BAR_THICK,
        0.0,
        0.0,
        horizontal_z,
    )

    return trm_util.concatenate([vertical_bar, horizontal_bar])


def main() -> None:
    print("========== TopDownTower STL 生成：保持 m 单位 + 整体再缩小 1/1.5 版本 ==========")
    print(f"ORIGINAL_SCALE = {ORIGINAL_SCALE}")
    print(f"GLOBAL_SIZE_FACTOR = {GLOBAL_SIZE_FACTOR}")
    print(f"SCALE (内部建模, m) = {SCALE}")
    print(f"EXPORT_UNIT_SCALE  = {EXPORT_UNIT_SCALE:.6f}  (= 1/1.5, 仅缩小不改单位)")
    print(f"输出目录           = {os.path.relpath(_OUT_DIR, _PROJ_ROOT)}")

    print("\n--- base_plate ---")
    print(f"BASE_XYZ              = {BASE_XYZ.tolist()}")
    print(f"BASE_SOCKET_X/Y       = {BASE_SOCKET_X:.6f}, {BASE_SOCKET_Y:.6f}")
    print(f"BASE_SOCKET_DEPTH     = {BASE_SOCKET_DEPTH:.6f}")
    print(f"POST_OFFSET_X/Y       = {POST_OFFSET_X:.6f}, {POST_OFFSET_Y:.6f}")

    print("\n--- post ---")
    print(f"POST_HEIGHT_FACTOR    = {POST_HEIGHT_FACTOR:.6f}")
    print(f"POST_H                = {POST_H:.6f}")
    print(f"POST_BODY_X/Y         = {POST_BODY_X:.6f}, {POST_BODY_Y:.6f}")
    print(f"POST_FOOT_X/Y/H       = {POST_FOOT_X:.6f}, {POST_FOOT_Y:.6f}, {POST_FOOT_H:.6f}")
    print(f"POST_FOOT_CLEARANCE   = {POST_FOOT_CLEARANCE:.6f}")

    print("\n--- middle_plate ---")
    print(f"MIDDLE_PLATE_XYZ      = {MIDDLE_PLATE_XYZ.tolist()}")
    print(f"MIDDLE_SOCKET_X/Y     = {MIDDLE_SOCKET_X:.6f}, {MIDDLE_SOCKET_Y:.6f}")
    print(f"MIDDLE_SOCKET_DEPTH   = {MIDDLE_SOCKET_DEPTH:.6f}")

    print("\n--- top_cross ---")
    print(f"TOP_CROSS_WIDTH       = {TOP_CROSS_WIDTH:.6f}")
    print(f"TOP_CROSS_HEIGHT      = {TOP_CROSS_HEIGHT:.6f}")
    print(f"TOP_CROSS_BAR_THICK   = {TOP_CROSS_BAR_THICK:.6f}")

    _export_mesh(
        _gen_base_plate_with_square_pockets(),
        os.path.join(_OUT_DIR, "base_plate.stl"),
        "base_plate",
    )

    _export_mesh(
        _gen_post_with_enlarged_foot(),
        os.path.join(_OUT_DIR, "post.stl"),
        "post",
    )

    _export_mesh(
        _gen_middle_plate_with_square_socket(),
        os.path.join(_OUT_DIR, "middle_plate.stl"),
        "middle_plate",
    )

    _export_mesh(
        _gen_vertical_top_cross_mesh(),
        os.path.join(_OUT_DIR, "top_cross.stl"),
        "top_cross",
    )

    print("\nTopDownTower STL assets generated.")
    print(f"asset_dir       : {_OUT_DIR}")
    print("unit logic      : STL 顶点坐标单位仍为 m (不做单位转换)。")
    print("scale logic     : 在原几何基础上整体再缩小为 2/3 (÷1.5)，比例/间隙同步缩小。")
    print("roof_plate      : 已移除，不生成。")
    print("top_cross logic : 竖着插入 middle_plate 顶面方形孔。")
    print("post logic      : 四根柱子仅 Z 高度变为当前 2 倍，XY 和孔位保持一致。")
    print("insert logic    : post 底脚比 base 孔口略小，便于插入。")


if __name__ == "__main__":
    main()
