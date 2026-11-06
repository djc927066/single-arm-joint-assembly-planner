#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate TopDownTower Assembly Definition - 0.75x + 2x Tall Posts
====================================================================

适配最新 STL 尺寸版本：

1. 所有零件整体为上一版当前尺寸的 0.75 倍；
2. 四根 post 的高度为当前高度的 2 倍；
3. base_plate 有四个方形浅孔；
4. 四根 post 的底脚插入 base_plate 四角孔；
5. middle_plate 放在四根 post 顶部；
6. top_cross 竖着插入 middle_plate 顶面中心方形孔；
7. roof_plate 已移除，不再写入 asmdef。

输出：
    sealp/assembly_sequence/_demo_output/topdown_tower.asmdef

说明：
    asmdef 只保存“相对于 fixture 的装配结构”。
    最终世界装配位置由 layout/search 脚本决定，不应该写死在 asmdef 里。
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np


_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    """从当前脚本目录向上找 sealp 根目录。"""
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError(
                "无法自动找到 sealp 根目录。请确认本脚本在 wrs-sealp/sealp 目录内部。"
            )
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef


# ============================================================
# STL 路径
# ============================================================
# 最新 mesh 生成脚本通常放在：
#   sealp/assets/models/Toy/model/gen_topdown_tower_meshes.py
# 但为了兼容，也会自动检查 Toy/tower。
# ============================================================

ASSET_DIR_MODEL = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "model")
ASSET_DIR_TOWER = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "tower")

if os.path.exists(os.path.join(ASSET_DIR_MODEL, "base_plate.stl")):
    ASSET_DIR = ASSET_DIR_MODEL
else:
    ASSET_DIR = ASSET_DIR_TOWER

BASE_STL = os.path.join(ASSET_DIR, "base_plate.stl")
POST_STL = os.path.join(ASSET_DIR, "post.stl")
MIDDLE_PLATE_STL = os.path.join(ASSET_DIR, "middle_plate.stl")
TOP_CROSS_STL = os.path.join(ASSET_DIR, "top_cross.stl")


# ============================================================
# 与最新 gen_topdown_tower_meshes.py 保持一致的几何参数
# ============================================================

ORIGINAL_SCALE = 1.5
GLOBAL_SIZE_FACTOR = 0.75
SCALE = ORIGINAL_SCALE * GLOBAL_SIZE_FACTOR  # 1.125

# gen_topdown_tower_meshes.py 在导出时又对整网格统一缩小 1/1.5 (EXPORT_UNIT_SCALE),
# 所以 STL 的真实尺寸 = 内部建模(SCALE) × EXPORT_SHRINK。
# Z 方向本脚本用 _load_mesh_bounds 实测网格, 会自动适配; 但 XY 孔位是按公式写死的,
# 必须乘上同样的 EXPORT_SHRINK, 否则柱子会落在缩小前的旧孔位 -> 对不齐。
EXPORT_SHRINK = 1.0 / 1.5  # 必须与 gen_topdown_tower_meshes.py 的 EXPORT_UNIT_SCALE 一致
EFFECTIVE_SCALE = SCALE * EXPORT_SHRINK  # STL 的真实线性缩放(相对 *_ORG)

# 四个 base_plate 孔的位置 (与缩小后的 STL 孔位一致)。
POST_OFFSET_X_ORG = 0.085
POST_OFFSET_Y_ORG = 0.065
POST_X = POST_OFFSET_X_ORG * EFFECTIVE_SCALE
POST_Y = POST_OFFSET_Y_ORG * EFFECTIVE_SCALE

# 与 mesh 生成脚本保持一致
BASE_SOCKET_DEPTH_RATIO = 0.50
MIDDLE_SOCKET_DEPTH_RATIO = 0.50
POST_HEIGHT_FACTOR = 2.0

FIXTURE_POS = np.array([0.0, 0.0, 0.0], dtype=float)


def _check_stl_exists() -> None:
    """检查需要的 STL 是否存在。"""
    stl_paths = {
        "base_plate": BASE_STL,
        "post": POST_STL,
        "middle_plate": MIDDLE_PLATE_STL,
        "top_cross": TOP_CROSS_STL,
    }

    print("========== STL 路径检查 ==========")
    print(f"SEALP_ROOT = {SEALP_ROOT}")
    print(f"ASSET_DIR  = {ASSET_DIR}")

    missing = []
    for name, path in stl_paths.items():
        ok = os.path.exists(path)
        print(f"{name:14s}: {path}  -> {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(path)

    # roof_plate 明确不需要
    roof_path = os.path.join(ASSET_DIR, "roof_plate.stl")
    print(f"{'roof_plate':14s}: {roof_path}  -> ignored / not used")

    if missing:
        print("\n缺失的 STL：")
        for path in missing:
            print(f"  - {path}")
        raise FileNotFoundError("STL 文件缺失，请先重新生成 STL 或检查 ASSET_DIR 路径。")


def _load_mesh_bounds(stl_path: str) -> Dict:
    """读取 STL 的真实包围盒信息。"""
    try:
        import wrs.basis.trimesh as trimesh
        mesh = trimesh.load_mesh(stl_path)
    except Exception:
        import trimesh
        mesh = trimesh.load_mesh(stl_path)

    bounds = np.asarray(mesh.bounds, dtype=float)
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    height = z_max - z_min

    if height <= 0:
        raise ValueError(f"STL 高度异常：{stl_path}, bounds={bounds}")

    extents = bounds[1] - bounds[0]
    return {
        "path": stl_path,
        "bounds": bounds,
        "z_min": z_min,
        "z_max": z_max,
        "height": height,
        "extents": extents,
    }


def _print_mesh_info(name: str, info: Dict) -> None:
    ext = np.asarray(info["extents"], dtype=float)
    print(
        f"{name:14s}: "
        f"z_min={info['z_min']:.6f}, "
        f"z_max={info['z_max']:.6f}, "
        f"height={info['height']:.6f}, "
        f"extents={np.round(ext, 6).tolist()}"
    )


def generate() -> AssemblyDef:
    """Build the TopDownTower assembly definition."""
    _check_stl_exists()

    base_info = _load_mesh_bounds(BASE_STL)
    post_info = _load_mesh_bounds(POST_STL)
    middle_info = _load_mesh_bounds(MIDDLE_PLATE_STL)
    cross_info = _load_mesh_bounds(TOP_CROSS_STL)

    print("\n========== STL 包围盒 ==========")
    _print_mesh_info("base_plate", base_info)
    _print_mesh_info("post", post_info)
    _print_mesh_info("middle_plate", middle_info)
    _print_mesh_info("top_cross", cross_info)

    # ============================================================
    # 高度贴合逻辑
    # ============================================================
    # base_plate:
    #   base 的真实底面贴到 fixture z=0。
    #
    # posts:
    #   base 顶面方形孔深度为 base_height * 0.5。
    #   post 底脚应该插到孔底，所以 post 的真实底面 z =
    #       base_top - base_socket_depth
    #
    # middle_plate:
    #   middle_plate 底面放到四根 post 顶面。
    #
    # top_cross:
    #   middle_plate 顶面中心有方形浅孔，深度为 middle_height * 0.5。
    #   top_cross 竖直中柱底部插入孔底，所以 top_cross 底面 z =
    #       middle_top - middle_socket_depth
    # ============================================================

    base_rel_to_fixture_z = -base_info["z_min"]
    base_top_rel = base_info["z_max"]

    base_socket_depth = base_info["height"] * BASE_SOCKET_DEPTH_RATIO
    post_bottom_rel = base_top_rel - base_socket_depth
    post_rel_z = post_bottom_rel - post_info["z_min"]

    post_top_rel = post_bottom_rel + post_info["height"]
    middle_rel_z = post_top_rel - middle_info["z_min"]

    middle_top_rel = post_top_rel + middle_info["height"]
    middle_socket_depth = middle_info["height"] * MIDDLE_SOCKET_DEPTH_RATIO
    cross_bottom_rel = middle_top_rel - middle_socket_depth
    cross_rel_z = cross_bottom_rel - cross_info["z_min"]

    print("\n========== 装配相对坐标 ==========")
    print(f"ORIGINAL_SCALE          = {ORIGINAL_SCALE}")
    print(f"GLOBAL_SIZE_FACTOR      = {GLOBAL_SIZE_FACTOR}")
    print(f"SCALE                   = {SCALE}")
    print(f"EXPORT_SHRINK           = {EXPORT_SHRINK:.6f}")
    print(f"EFFECTIVE_SCALE         = {EFFECTIVE_SCALE:.6f}")
    print(f"POST_HEIGHT_FACTOR      = {POST_HEIGHT_FACTOR}")
    print(f"POST_X / POST_Y         = {POST_X:.6f}, {POST_Y:.6f}")
    print(f"base_rel_to_fixture_z   = {base_rel_to_fixture_z:.6f}")
    print(f"base_top_rel            = {base_top_rel:.6f}")
    print(f"base_socket_depth       = {base_socket_depth:.6f}")
    print(f"post_bottom_rel         = {post_bottom_rel:.6f}")
    print(f"post_rel_z              = {post_rel_z:.6f}")
    print(f"post_top_rel            = {post_top_rel:.6f}")
    print(f"middle_rel_z            = {middle_rel_z:.6f}")
    print(f"middle_top_rel          = {middle_top_rel:.6f}")
    print(f"middle_socket_depth     = {middle_socket_depth:.6f}")
    print(f"cross_bottom_rel        = {cross_bottom_rel:.6f}")
    print(f"cross_rel_z             = {cross_rel_z:.6f}")

    asm = AssemblyDef(
        name="TopDownTower",
        description=(
            "TopDownTower generated for 0.75x meshes with 2x tall posts. "
            "No roof_plate. Four posts insert into base pockets; "
            "top_cross vertically inserts into middle_plate square socket."
        ),
    )

    # Model library
    asm.add_model("base_model", BASE_STL)
    asm.add_model("post_model", POST_STL)
    asm.add_model("middle_plate_model", MIDDLE_PLATE_STL)
    asm.add_model("top_cross_model", TOP_CROSS_STL)

    # Parts
    asm.add_part(PartDef(
        part_id="base_plate",
        name="Base Plate with Square Pockets",
        model="base_model",
        mass=0.45 * EFFECTIVE_SCALE ** 3,
    ))

    for pid, name in [
        ("post_bl", "Back-Left Post"),
        ("post_fl", "Front-Left Post"),
        ("post_br", "Back-Right Post"),
        ("post_fr", "Front-Right Post"),
    ]:
        asm.add_part(PartDef(
            part_id=pid,
            name=name,
            model="post_model",
            mass=0.08 * EFFECTIVE_SCALE ** 3 * POST_HEIGHT_FACTOR,
        ))

    asm.add_part(PartDef(
        part_id="middle_plate",
        name="Middle Plate with Center Square Socket",
        model="middle_plate_model",
        mass=0.32 * EFFECTIVE_SCALE ** 3,
    ))

    asm.add_part(PartDef(
        part_id="top_cross",
        name="Vertical Top Cross",
        model="top_cross_model",
        mass=0.12 * EFFECTIVE_SCALE ** 3,
    ))

    asm.add_symmetry_group("posts", ["post_bl", "post_fl", "post_br", "post_fr"])

    # Step 0: base_plate -> fixture
    asm.add_step(StepDef(
        step_id=0,
        part_id="base_plate",
        parent_id="fixture",
        rel_pos=FIXTURE_POS + np.array([0.0, 0.0, base_rel_to_fixture_z], dtype=float),
        rel_rotmat=np.eye(3),
        deps=[],
        notes=(
            "Place base_plate on fixture. "
            "Its real bottom touches fixture z=0. "
            "This part can be treated as preassembled in layout search."
        ),
    ))

    # Steps 1-4: four posts inserted into base pockets.
    #
    # 命名约定：
    #   bl/br: back side, y = +POST_Y
    #   fl/fr: front side, y = -POST_Y
    #   l/r  : left/right along x = -/+POST_X
    #
    # 重要：这里必须和 mesh 生成脚本的孔位一致：
    #   holes = (±POST_OFFSET_X, ±POST_OFFSET_Y)
    # Larger goal-x posts (+POST_X) first, then smaller goal-x posts (-POST_X).
    post_specs = [
        (1, "post_br", np.array([ POST_X,  POST_Y, post_rel_z], dtype=float), [0]),
        (2, "post_fr", np.array([ POST_X, -POST_Y, post_rel_z], dtype=float), [1]),
        (3, "post_bl", np.array([-POST_X,  POST_Y, post_rel_z], dtype=float), [2]),
        (4, "post_fl", np.array([-POST_X, -POST_Y, post_rel_z], dtype=float), [3]),
    ]

    for step_id, pid, rel_pos, deps in post_specs:
        asm.add_step(StepDef(
            step_id=step_id,
            part_id=pid,
            parent_id="base_plate",
            rel_pos=rel_pos,
            rel_rotmat=np.eye(3),
            deps=deps,
            notes=(
                f"Insert {pid} into the matching square pocket. "
                f"post_bottom_rel={post_bottom_rel:.6f}, "
                f"base_socket_depth={base_socket_depth:.6f}."
            ),
        ))

    # Step 5: middle_plate on top of posts
    asm.add_step(StepDef(
        step_id=5,
        part_id="middle_plate",
        parent_id="base_plate",
        rel_pos=np.array([0.0, 0.0, middle_rel_z], dtype=float),
        rel_rotmat=np.eye(3),
        deps=[4],
        notes=(
            "Place middle_plate on top of all four posts. "
            "No roof_plate in the latest structure."
        ),
    ))

    # Step 6: vertical top_cross inserted into middle_plate socket
    asm.add_step(StepDef(
        step_id=6,
        part_id="top_cross",
        parent_id="base_plate",
        rel_pos=np.array([0.0, 0.0, cross_rel_z], dtype=float),
        rel_rotmat=np.eye(3),
        deps=[5],
        notes=(
            "Insert vertical top_cross into middle_plate center square socket. "
            f"cross_bottom_rel={cross_bottom_rel:.6f}, "
            f"middle_socket_depth={middle_socket_depth:.6f}."
        ),
    ))

    errors = asm.validate(strict=False)
    if errors:
        print("\nValidation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nValidation passed.")

    return asm


def main() -> None:
    asm = generate()

    out_dir = os.path.join(SEALP_ROOT, "assembly_sequence", "_demo_output")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "topdown_tower.asmdef")
    asm.save(path)

    print(f"\nSaved: {path}")
    print()
    print(asm.summary())


if __name__ == "__main__":
    main()
