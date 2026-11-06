#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter Tower Base Plate Grasps — Keep Approx. Top-Down Grasps
================================================================

功能：
    读取 tower_base_plate_grasps.pickle，
    只保留“从上往下抓”的抓取姿态，也就是夹爪的 z 轴大致朝世界坐标系 z 轴负方向。

输入：
    D:/Project/wrs-sealp/sealp/examples/grasp/tower_grasp/tower_base_plate_grasps.pickle

输出：
    D:/Project/wrs-sealp/sealp/examples/grasp/tower_grasp/tower_base_plate_grasps_topdown.pickle

说明：
    不要求特别严格，只要夹爪 z 轴大概朝下即可。
    这里默认允许最大偏离角为 70 度。
    也就是：
        dot(gripper_z_axis, world_down) > cos(70°)
    等价于：
        grasp.ac_rotmat[2, 2] < -cos(70°)

运行：
    D:/Soft/tools/anaconda/envs/spatialvla/python.exe D:/Project/wrs-sealp/sealp/examples/grasp/filter_tower_base_topdown.py

如果要更宽松：
    把 MAX_ANGLE_DEG 改大，比如 80

如果要更严格：
    把 MAX_ANGLE_DEG 改小，比如 45 或 60
"""
from __future__ import annotations

import os
import sys
import math
import argparse
import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg
from wrs.grasping.grasp import GraspCollection


# ============================================================
# 自动定位 sealp 根目录
# ============================================================

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
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


# ============================================================
# 路径设置
# ============================================================

# 用户指定的抓取文件夹：
# D:/Project/wrs-sealp/sealp/examples/grasp/tower_grasp/
GRASP_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp")

INPUT_PKL = os.path.join(GRASP_DIR, "tower_top_cross_grasps.pickle")
OUTPUT_PKL = os.path.join(GRASP_DIR, "tower_top_cross_grasps_filt.pickle")

# 可视化时加载 base_plate.stl
# 如果你的 STL 在 Toy/tower，就把 model 改成 tower
ASSET_DIR = os.path.join(SEALP_ROOT, "assets", "models", "Toy", "model")
BASE_STL = os.path.join(ASSET_DIR, "top_cross.stl")


MAX_ANGLE_DEG = 60.0
GRIPPER_AXIS_IDX = 2


def filter_topdown_grasps(
    grasp_collection: GraspCollection,
    max_angle_deg: float = MAX_ANGLE_DEG,
    axis_idx: int = GRIPPER_AXIS_IDX,
) -> GraspCollection:
    """保留夹爪指定轴大致朝世界 -Z 的抓取姿态。

    对 ac_rotmat 来说：
        grasp.ac_rotmat[:, axis_idx] 是夹爪局部 axis_idx 轴在世界坐标中的方向。
    判断“朝下”主要看它在世界 z 方向上的分量：
        z_component = grasp.ac_rotmat[2, axis_idx]

    如果这个值越接近 -1，说明越朝世界 -Z。
    这里不严格，默认只要求：
        z_component < -cos(max_angle_deg)
    """
    cos_th = math.cos(math.radians(max_angle_deg))
    threshold = -cos_th

    filtered = GraspCollection()

    for grasp in grasp_collection:
        z_component = float(grasp.ac_rotmat[2, axis_idx])

        # z_component 越小越朝下。
        # 例如：
        #   -1.0 完全朝下
        #   -0.5 大概朝下
        #    0.0 水平
        #    1.0 完全朝上
        if z_component < threshold:
            filtered.append(grasp)

    return filtered


def print_orientation_statistics(
    grasp_collection: GraspCollection,
    axis_idx: int = GRIPPER_AXIS_IDX,
) -> None:
    """打印抓取方向统计，方便判断过滤是否太严或太松。"""
    if len(grasp_collection) == 0:
        print("没有抓取姿态，无法统计。")
        return

    z_values = np.array(
        [float(grasp.ac_rotmat[2, axis_idx]) for grasp in grasp_collection],
        dtype=float,
    )

    print("========== 抓取方向统计 ==========")
    print(f"axis_idx              = {axis_idx}")
    print(f"z_component min       = {z_values.min():.6f}")
    print(f"z_component max       = {z_values.max():.6f}")
    print(f"z_component mean      = {z_values.mean():.6f}")
    print("说明：z_component 越接近 -1，夹爪 z 轴越朝世界 -Z。")


def visualize_filtered(
    original: GraspCollection,
    filtered: GraspCollection,
    max_show_original: int = 30,
    max_show_filtered: int = 80,
) -> None:
    """可视化原始抓取和过滤后抓取。"""
    if not os.path.exists(BASE_STL):
        print(f"[WARN] 找不到 base STL，跳过可视化：{BASE_STL}")
        return

    base = wd.World(
        cam_pos=rm.vec(0.55, 0.55, 0.45),
        lookat_pos=rm.vec(0, 0, 0.03),
    )
    mgm.gen_frame(ax_length=0.2).attach_to(base)

    obj_cmodel = mcm.CollisionModel(BASE_STL)
    obj_cmodel.rgba = np.array([0.55, 0.55, 0.55, 1.0])
    obj_cmodel.attach_to(base)

    gripper = pg.PantheraGripper()

    # 原始抓取：半透明
    for i, grasp in enumerate(original):
        if i >= max_show_original:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
        gripper.gen_meshmodel(alpha=0.15).attach_to(base)

    # 过滤后抓取：更明显
    for i, grasp in enumerate(filtered):
        if i >= max_show_filtered:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
        gripper.gen_meshmodel(alpha=0.8).attach_to(base)

    print("\n正在显示：原始抓取为半透明，过滤后的 top-down 抓取更明显。")
    print("按 ESC 关闭窗口。")
    base.run()


def main(
    input_pkl: str = INPUT_PKL,
    output_pkl: str = OUTPUT_PKL,
    max_angle_deg: float = MAX_ANGLE_DEG,
    visualize: bool = False,
) -> None:
    print("========== Tower Base Plate Top-Down Grasp Filtering ==========")
    print(f"Input pickle  : {input_pkl}")
    print(f"Output pickle : {output_pkl}")
    print(f"MAX_ANGLE_DEG : {max_angle_deg}")

    if not os.path.exists(input_pkl):
        raise FileNotFoundError(
            f"找不到输入抓取文件：{input_pkl}\n"
            "请确认 tower_base_plate_grasps.pickle 是否已经生成。"
        )

    grasp_collection = GraspCollection.load_from_disk(file_name=input_pkl)
    print(f"\n原始抓取数量: {len(grasp_collection)}")

    print_orientation_statistics(grasp_collection, axis_idx=GRIPPER_AXIS_IDX)

    filtered = filter_topdown_grasps(
        grasp_collection,
        max_angle_deg=max_angle_deg,
        axis_idx=GRIPPER_AXIS_IDX,
    )

    print(f"\n过滤后 top-down 抓取数量: {len(filtered)}")

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    filtered.save_to_disk(file_name=output_pkl)
    print(f"已保存到: {output_pkl}")

    if len(filtered) == 0:
        print("\n[提示] 过滤后数量为 0，说明条件可能太严格。")
        print("可以改大 MAX_ANGLE_DEG，比如 80，或者运行时加：--max-angle-deg 80")

    if visualize:
        visualize_filtered(grasp_collection, filtered)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Filter tower_base_plate_grasps.pickle and keep approximate top-down grasps."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PKL,
        help="输入 pickle 路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PKL,
        help="输出 pickle 路径。",
    )
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=MAX_ANGLE_DEG,
        help="允许夹爪 z 轴偏离世界 -Z 的最大角度。默认 70，越大越宽松。",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="显示过滤后的抓取姿态。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        input_pkl=args.input,
        output_pkl=args.output,
        max_angle_deg=args.max_angle_deg,
        visualize=args.vis,
    )
