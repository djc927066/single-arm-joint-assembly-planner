#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tower Grasp Planning — Panthera Gripper, No roof_plate
=========================================================

为 TopDownTower 最新版本零件生成抓取规划结果，并保存为 pickle 文件。

最新结构：
    base_plate
    post              # 四根柱子共用同一个 post.stl / tower_post_grasps.pickle
    middle_plate
    top_cross         # 竖着插入 middle_plate 顶面方形孔
    roof_plate 已移除

默认 STL 文件夹：
    优先读取 sealp/assets/models/Toy/tower/
    如果不存在，则回退读取 sealp/assets/models/Toy/model/

默认输出文件夹：
    sealp/examples/grasp/tower_grasp/

生成的抓取文件：
    tower_base_plate_grasps.pickle
    tower_post_grasps.pickle
    tower_middle_plate_grasps.pickle
    tower_top_cross_grasps.pickle

运行：
    python -m sealp.examples.grasp.planning_tower --no-vis

只规划某一个零件：
    python -m sealp.examples.grasp.planning_tower --only base_plate --no-vis
    python -m sealp.examples.grasp.planning_tower --only post --no-vis
    python -m sealp.examples.grasp.planning_tower --only middle_plate --no-vis
    python -m sealp.examples.grasp.planning_tower --only top_cross --no-vis
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg


_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    """从当前脚本目录向上查找 sealp 根目录。"""
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError(
                "无法自动找到 sealp 根目录。请确认本脚本位于 wrs-sealp/sealp 目录内部。"
            )
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _default_asset_dir() -> str:
    """优先使用 Toy/tower；如果没有则回退 Toy/model。"""
    tower_dir = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "tower")
    model_dir = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "model")

    if os.path.isfile(os.path.join(tower_dir, "base_plate.stl")):
        return tower_dir
    return model_dir


TOWER_ASSET_DIR = _default_asset_dir()

# 注意：为了和 layout 搜索脚本默认 grasp_dir 对齐，这里默认输出到 examples/grasp/tower_grasp
OUT_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp")


# roof_plate 已经移除，不再检查和规划。
GRASP_SPECS: List[Tuple[str, str, str]] = [
    # grasp_id, stl_file, role/name
    ("base_plate", "base_plate.stl", "Base Plate"),
    ("post", "post.stl", "Shared Post for post_bl/post_fl/post_br/post_fr"),
    ("middle_plate", "middle_plate.stl", "Middle Plate"),
    ("top_cross", "top_cross.stl", "Vertical Top Cross"),
]


def plan_grasps(
    obj_cmodel: mcm.CollisionModel,
    gripper=None,
    angle_between_contact_normals=None,
    rotation_interval=None,
    max_samples: int = 100,
    min_dist_between_sampled_contact_points: float = 0.01,
    contact_offset: float = 0.01,
    toggle_dbg: bool = False,
):
    """Plan antipodal grasps on an object using PantheraGripper."""
    if gripper is None:
        gripper = pg.PantheraGripper()

    if angle_between_contact_normals is None:
        angle_between_contact_normals = rm.radians(175)

    if rotation_interval is None:
        rotation_interval = rm.radians(15)

    grasp_collection = gpa.plan_gripper_grasps(
        gripper,
        obj_cmodel,
        angle_between_contact_normals=angle_between_contact_normals,
        rotation_interval=rotation_interval,
        max_samples=max_samples,
        min_dist_between_sampled_contact_points=min_dist_between_sampled_contact_points,
        contact_offset=contact_offset,
        toggle_dbg=toggle_dbg,
    )
    return grasp_collection, gripper


def visualize_grasps(
    base,
    obj_cmodel: mcm.CollisionModel,
    grasp_collection,
    gripper,
    max_show: int = 80,
    alpha: float = 0.7,
):
    """在 Panda3D 中显示抓取结果。"""
    obj_cmodel.attach_to(base)

    for i, grasp in enumerate(grasp_collection):
        if i >= max_show:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
        gripper.gen_meshmodel(alpha=alpha).attach_to(base)


def _mesh_path(stl_file: str, asset_dir: str) -> str:
    return os.path.join(asset_dir, stl_file)


def _save_path(grasp_id: str, out_dir: str) -> str:
    return os.path.join(out_dir, f"tower_{grasp_id}_grasps.pickle")


def _check_assets(asset_dir: str) -> None:
    """检查 STL 文件是否齐全。"""
    print("========== Tower Grasp Planning 路径检查 ==========")
    print(f"SEALP_ROOT      = {SEALP_ROOT}")
    print(f"TOWER_ASSET_DIR = {asset_dir}")
    print(f"OUT_DIR         = {OUT_DIR}")
    print("roof_plate.stl  : 已移除，不再检查")

    missing = []
    checked = set()

    for _, stl_file, _ in GRASP_SPECS:
        if stl_file in checked:
            continue
        checked.add(stl_file)

        path = _mesh_path(stl_file, asset_dir)
        ok = os.path.isfile(path)
        print(f"{stl_file:18s}: {path} -> {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(path)

    if missing:
        print("\n缺失 STL：")
        for path in missing:
            print(f"  - {path}")
        raise FileNotFoundError("Tower STL 文件缺失，请先生成 STL 或检查路径。")


def main(
    visualize: bool = True,
    only: str | None = None,
    max_samples: int = 150,
    rotation_deg: float = 15.0,
    asset_dir: str | None = None,
    out_dir: str | None = None,
):
    """Run grasp planning for TopDownTower unique meshes."""
    asset_dir = os.path.abspath(asset_dir or TOWER_ASSET_DIR)
    out_dir = os.path.abspath(out_dir or OUT_DIR)

    _check_assets(asset_dir)
    os.makedirs(out_dir, exist_ok=True)

    if only is not None:
        valid_ids = [grasp_id for grasp_id, _, _ in GRASP_SPECS]
        if only not in valid_ids:
            raise ValueError(f"--only 参数错误：{only}。可选值为：{valid_ids}")

    base = None
    if visualize:
        base = wd.World(
            cam_pos=rm.vec(0.45, 0.45, 0.35),
            lookat_pos=rm.vec(0, 0, 0.04),
        )
        mgm.gen_frame(ax_length=0.2).attach_to(base)

    last_obj = None
    last_grasps = None
    last_gripper = None
    last_grasp_id = None

    for grasp_id, stl_file, role in GRASP_SPECS:
        if only is not None and grasp_id != only:
            continue

        mesh_path = _mesh_path(stl_file, asset_dir)
        save_path = _save_path(grasp_id, out_dir)

        print("\n" + "=" * 70)
        print(f"Grasp Planning — Panthera Gripper [{grasp_id} / {role}]")
        print("=" * 70)
        print(f"Mesh : {mesh_path}")
        print(f"Save : {save_path}")

        obj_cmodel = mcm.CollisionModel(mesh_path)
        obj_cmodel.rgba = np.array([0.6, 0.5, 0.4, 1.0])

        grasp_collection, gripper = plan_grasps(
            obj_cmodel,
            max_samples=max_samples,
            rotation_interval=rm.radians(rotation_deg),
            min_dist_between_sampled_contact_points=0.01,
            contact_offset=0.01,
            toggle_dbg=False,
        )

        print(f"  Planned {len(grasp_collection)} grasps.")

        grasp_collection.save_to_disk(file_name=save_path)
        print(f"  Saved to: {save_path}")

        last_obj = obj_cmodel
        last_grasps = grasp_collection
        last_gripper = gripper
        last_grasp_id = grasp_id

    if visualize and base is not None and last_obj is not None:
        print("\n" + "=" * 70)
        print(f"Showing first grasps of last planned mesh: {last_grasp_id}")
        print("Press ESC to close.")
        print("=" * 70)
        visualize_grasps(
            base,
            last_obj,
            last_grasps,
            last_gripper,
            max_show=80,
            alpha=0.7,
        )
        base.run()

    print("\n全部抓取规划完成。")
    print(f"输出目录：{out_dir}")
    print("\n注意：四根柱子 post_bl / post_fl / post_br / post_fr 后续共用：")
    print(f"  {_save_path('post', out_dir)}")
    print("roof_plate 已移除，不再生成 tower_roof_plate_grasps.pickle")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plan grasps for latest TopDownTower unique meshes and save pickle files."
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="不弹出 Panda3D 可视化窗口，只生成 pickle 文件。",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=[grasp_id for grasp_id, _, _ in GRASP_SPECS],
        help="只规划某一个模型。可选：base_plate, post, middle_plate, top_cross",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="接触点采样数量，默认 100。失败或太慢时可以调小，比如 50；想抓取多一点可调 200/300。",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=30.0,
        help="抓取绕接触法向旋转采样间隔，默认 30 度。想更密集可调 15。",
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=TOWER_ASSET_DIR,
        help="STL 文件夹；默认优先 Toy/tower，否则 Toy/model。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUT_DIR,
        help="grasp pickle 输出文件夹；默认 sealp/examples/grasp/tower_grasp。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        visualize=not args.no_vis,
        only=args.only,
        max_samples=args.max_samples,
        rotation_deg=args.rotation_deg,
        asset_dir=args.asset_dir,
        out_dir=args.output_dir,
    )
