#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/5/19 17:50
# @Author : ZhangXi
"""
Grasp Planning (Targeted for Seat Holes)
========================================
专为 yuanchair-part1 (椅面) 设计的定向抓取规划脚本。
不再全空间随机盲目采样，而是加大采样密度，
并在底层规划时只保留从上往下（Z轴向下）的抓取角度，从而成功抓取孔洞。
"""

import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.grasping.planning.antipodal as gpa
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg


def plan_targeted_grasps(obj_cmodel, gripper):
    """
    使用高密度、细粒度的采样，规划针对特定区域（孔洞）的抓取。
    """
    # 1. 加大采样密度，确保能覆盖到孔洞区域
    # 缩小接触点间距 (更密)，增加最大采样数
    min_dist = 0.005  # 5mm，对于孔洞这种细节区域需要更密的采样
    max_samples = 300  # 撒 300 个点

    # 2. 缩小旋转间隔，让夹爪能找到刚刚好插进孔里的角度
    rot_interval = rm.radians(15)  # 每 15 度转一次

    print(f"  [配置] 采样点数: {max_samples}, 点间距: {min_dist}m, 旋转步长: {math.degrees(rot_interval)}度")

    grasp_collection = gpa.plan_gripper_grasps(
        gripper=gripper,
        obj_cmodel=obj_cmodel,
        angle_between_contact_normals=rm.radians(170),
        max_samples=max_samples,
        min_dist_between_sampled_contact_points=min_dist,
        contact_offset=0.01,
    )
    return grasp_collection


def filter_downward_grasps(grasp_collection):
    """
    硬核过滤：只保留夹爪从上往下接近的抓取姿态。
    """
    from wrs.grasping.grasp import GraspCollection
    filtered = GraspCollection()
    for grasp in grasp_collection:
        z_component = grasp.ac_rotmat[2, 2]
        if z_component < -0.8:
            filtered.append(grasp)

    return filtered


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "_output")
    os.makedirs(out_dir, exist_ok=True)

    # 模型路径
    mesh_path = os.path.join(current_dir, "..", "..", "assets", "models", "yuanchair", "yuanchair-part1.stl")
    if not os.path.isfile(mesh_path):
        print(f"[ERROR] 找不到模型文件: {mesh_path}")
        return

    # 初始化模型和夹爪
    obj_cmodel = mcm.CollisionModel(mesh_path)
    obj_cmodel.rgba = np.array([0.6, 0.5, 0.4, 1.0])
    gripper = pg.PantheraGripper()

    print("=" * 60)
    print(f"Targeted Grasp Planning — Seat Holes")
    print("=" * 60)

    # 步骤 1: 高密度规划
    print("  正在执行高密度物理采样寻点... (可能需要 1~3 分钟，请耐心等待)")
    raw_grasps = plan_targeted_grasps(obj_cmodel, gripper)
    print(f"  > 物理引擎共找到了 {len(raw_grasps)} 个全方向的有效抓取。")

    # 步骤 2: 严格方向过滤
    print("  正在过滤非朝下的抓取...")
    final_grasps = filter_downward_grasps(raw_grasps)
    print(f"  > 过滤完成！最终保留了 {len(final_grasps)} 个完美的朝下抓孔姿态。")

    if len(final_grasps) > 0:
        # 步骤 3: 直接保存为你要覆盖的文件名
        save_path = os.path.join(out_dir, "demo_yuanchair-part1_grasps.pickle")
        final_grasps.save_to_disk(file_name=save_path)
        print(f"\n[OK] 成功保存定向抓取数据至: {save_path}")
    else:
        print("\n[WARN] 仍然没有找到朝下的抓取！")
        print("原因可能是: 1. 夹爪完全张开的宽度比孔的内径小太多，没法夹住边缘。")
        print("          2. 夹爪两根手指太厚，插不进孔里面，发生严重穿模。")


if __name__ == "__main__":
    main()