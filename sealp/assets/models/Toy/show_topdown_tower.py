#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show TopDownTower - Vertical Top Cross Inserted Into Square Socket
====================================================================

展示结构：
    base_plate：四角孔
    4 x post：底部插入 base_plate 四角孔
    middle_plate：顶面中心方形孔
    top_cross：竖着插入 middle_plate 方形孔

注意：
    middle_plate 顶面是方形孔，不是十字形孔。
    top_cross 是竖着插进去，不是平放在 middle_plate 上。
"""

from __future__ import annotations

import os
import sys
import numpy as np

from wrs import wd, mgm, mcm

try:
    import trimesh
except Exception:
    import wrs.basis.trimesh as trimesh


_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("无法自动找到 sealp 根目录。请确认本脚本在 wrs-sealp/sealp 目录内部。")
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 优先使用 Toy/tower；如果你实际放在 Toy/model，也会自动回退
ASSET_DIR_TOWER = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "tower")
ASSET_DIR_MODEL = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "model")
ASSET_DIR = ASSET_DIR_TOWER if os.path.exists(os.path.join(ASSET_DIR_TOWER, "base_plate.stl")) else ASSET_DIR_MODEL

BASE_STL = os.path.join(ASSET_DIR, "base_plate.stl")
POST_STL = os.path.join(ASSET_DIR, "post.stl")
MIDDLE_STL = os.path.join(ASSET_DIR, "middle_plate.stl")
CROSS_STL = os.path.join(ASSET_DIR, "top_cross.stl")


# 必须和 gen_topdown_tower_meshes.py 保持一致。
# 之前这里还是 SCALE=1.5，而 STL 生成脚本已经改成 1.5*0.75=1.125，
# 所以展示时四根柱子会放到旧孔位，导致看起来“孔没有对准”。
# 进一步:gen 脚本在导出时又对整网格统一缩小 1/1.5 (EXPORT_SHRINK),
# base_plate 的孔位也跟着缩小了, 所以这里柱子 XY 偏移必须同样乘上 EXPORT_SHRINK,
# 否则柱子会落在缩小前的旧孔位, 仍然对不齐。
ORIGINAL_SCALE = 1.5
GLOBAL_SIZE_FACTOR = 0.75
SCALE = ORIGINAL_SCALE * GLOBAL_SIZE_FACTOR
EXPORT_SHRINK = 1.0 / 1.5  # 必须与 gen_topdown_tower_meshes.py 的 EXPORT_UNIT_SCALE 一致
POST_OFFSET_X = 0.085 * SCALE * EXPORT_SHRINK
POST_OFFSET_Y = 0.065 * SCALE * EXPORT_SHRINK

# 这里仅用于展示，不影响 asmdef / layout 搜索
FIXTURE_POS = np.array([0.20, -0.30, 0.0], dtype=float)

# 这些比例要和 gen_topdown_tower_meshes.py 保持一致
BASE_SOCKET_DEPTH_RATIO = 0.50
MIDDLE_SOCKET_DEPTH_RATIO = 0.50

COLOR_BASE = np.array([0.55, 0.55, 0.55, 1.0])
COLOR_POST = np.array([0.25, 0.55, 0.95, 1.0])
COLOR_MIDDLE = np.array([0.95, 0.65, 0.25, 1.0])
COLOR_CROSS = np.array([0.35, 0.85, 0.50, 1.0])


def check_stl_exists() -> None:
    stl_list = [BASE_STL, POST_STL, MIDDLE_STL, CROSS_STL]
    missing = [p for p in stl_list if not os.path.exists(p)]

    print("========== STL 路径检查 ==========")
    print(f"SEALP_ROOT = {SEALP_ROOT}")
    print(f"ASSET_DIR  = {ASSET_DIR}")
    for p in stl_list:
        print(f"{os.path.basename(p):18s}: {p} -> {'OK' if os.path.exists(p) else 'MISSING'}")

    if missing:
        print("\n[ERROR] 以下 STL 文件不存在：")
        for p in missing:
            print("   ", p)
        raise FileNotFoundError("STL 文件缺失，请检查 ASSET_DIR 路径。")


def get_bounds(stl_path: str):
    mesh = trimesh.load_mesh(stl_path)
    bounds = np.asarray(mesh.bounds, dtype=float)
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    height = z_max - z_min
    return z_min, z_max, height


def add_part_bottom_at(base, name: str, stl_path: str, bottom_world_pos, rgba=None):
    """加载 STL，并让它的真实底面贴到指定 world z。"""
    bottom_world_pos = np.asarray(bottom_world_pos, dtype=float)
    z_min, z_max, height = get_bounds(stl_path)

    corrected_pos = bottom_world_pos.copy()
    corrected_pos[2] = bottom_world_pos[2] - z_min

    model = mcm.CollisionModel(stl_path)
    model.pos = corrected_pos

    if rgba is not None:
        model.rgba = rgba

    model.attach_to(base)

    print(
        f"{name:14s} -> {os.path.basename(stl_path):18s} "
        f"bottom_z={bottom_world_pos[2]:.6f}, "
        f"z_min={z_min:.6f}, height={height:.6f}, "
        f"model.pos={model.pos.tolist()}"
    )

    return model, height


def main() -> None:
    check_stl_exists()

    _, _, base_h = get_bounds(BASE_STL)
    _, _, post_h = get_bounds(POST_STL)
    _, _, middle_h = get_bounds(MIDDLE_STL)
    _, _, cross_h = get_bounds(CROSS_STL)

    base_socket_depth = base_h * BASE_SOCKET_DEPTH_RATIO
    middle_socket_depth = middle_h * MIDDLE_SOCKET_DEPTH_RATIO

    print("\n========== 模型真实高度 ==========")
    print(f"base_h              = {base_h:.6f}")
    print(f"post_h              = {post_h:.6f}")
    print(f"middle_h            = {middle_h:.6f}")
    print(f"cross_h             = {cross_h:.6f}")
    print(f"base_socket_depth   = {base_socket_depth:.6f}")
    print(f"middle_socket_depth = {middle_socket_depth:.6f}")
    print(f"post offset         = x ±{POST_OFFSET_X:.6f}, y ±{POST_OFFSET_Y:.6f}")
    print("roof_plate          = 已移除，不再加载")
    print("top_cross           = 竖着插入 middle_plate 顶面方形孔")
    print(f"show SCALE           = {SCALE}  # must match mesh generator")

    total_h = base_h - base_socket_depth + post_h + middle_h - middle_socket_depth + cross_h

    base = wd.World(
        cam_pos=[0.85, -1.10, 0.85],
        lookat_pos=FIXTURE_POS + np.array([0.0, 0.0, total_h * 0.45]),
    )

    mgm.gen_frame(pos=FIXTURE_POS).attach_to(base)

    # 1. base_plate
    base_bottom = FIXTURE_POS + np.array([0.0, 0.0, 0.0])
    add_part_bottom_at(base, "base_plate", BASE_STL, base_bottom, COLOR_BASE)

    # 2. posts 插入 base_plate 方孔
    post_bottom_z = base_bottom[2] + base_h - base_socket_depth

    post_positions = {
        "post_bl": np.array([ POST_OFFSET_X,  POST_OFFSET_Y, post_bottom_z]),
        "post_fl": np.array([-POST_OFFSET_X,  POST_OFFSET_Y, post_bottom_z]),
        "post_br": np.array([ POST_OFFSET_X, -POST_OFFSET_Y, post_bottom_z]),
        "post_fr": np.array([-POST_OFFSET_X, -POST_OFFSET_Y, post_bottom_z]),
    }

    for name, rel_pos in post_positions.items():
        add_part_bottom_at(
            base,
            name,
            POST_STL,
            FIXTURE_POS + rel_pos,
            COLOR_POST,
        )

    # 3. middle_plate 放到柱子顶面
    middle_bottom_z = post_bottom_z + post_h
    add_part_bottom_at(
        base,
        "middle_plate",
        MIDDLE_STL,
        FIXTURE_POS + np.array([0.0, 0.0, middle_bottom_z]),
        COLOR_MIDDLE,
    )

    # 4. top_cross 竖着插入 middle_plate 顶面方形孔
    cross_bottom_z = middle_bottom_z + middle_h - middle_socket_depth
    add_part_bottom_at(
        base,
        "top_cross",
        CROSS_STL,
        FIXTURE_POS + np.array([0.0, 0.0, cross_bottom_z]),
        COLOR_CROSS,
    )

    print("\nTopDown Tower display finished.")
    print(f"STL directory: {ASSET_DIR}")
    print("说明：top_cross 的底部一小段插入 middle_plate 顶面中心方形孔。")

    base.run()


if __name__ == "__main__":
    main()
