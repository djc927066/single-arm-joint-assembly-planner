#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug base_plate grasp reachability / common gids.

用途：
    检查 tower_base_plate_grasps.pickle 到底是不是在当前 asmdef 的 base_plate
    目标位姿上可达。

重点判断：
1. goal only 有多少 grasp 可达；
2. staging=goal 时有多少 common gids；
3. staging 在 goal 附近抬高 3mm 时有多少 common gids；
4. 左臂/右臂分别是多少。

如果这些都是 0，说明不是 layout 搜索问题，也不是 mesh/box 碰撞问题，
而是 base_plate 抓取库本身在当前机器人/目标位姿下不可达，需要重新生成或放宽过滤。
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import wrs.basis.robot_math as rm
from wrs.grasping.grasp import GraspCollection
from wrs.manipulation.pick_place import PickPlacePlanner
import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda

from sealp.assembly_sequence import AssemblyDef


HOME_JV = np.zeros(6)
DUAL_ARM_Y_OFFSET = 0.62


def _test_pose_list(arm, gc, pose_list, label: str):
    planner = PickPlacePlanner(robot=arm)
    try:
        gids = planner.reason_common_gids(
            grasp_collection=gc,
            goal_pose_list=pose_list,
            obstacle_list=[],
        )
    except Exception as e:
        print(f"  {label:<28s}: EXCEPTION {type(e).__name__}: {e!r}")
        return []
    print(f"  {label:<28s}: {len(gids)} gids -> {gids[:20]}")
    return gids


def _direct_ik_count(arm, gc, obj_pos, obj_rot, label: str):
    ok = []
    for i, g in enumerate(gc):
        tcp_pos = obj_rot.dot(g.ac_pos) + obj_pos
        tcp_rot = obj_rot.dot(g.ac_rotmat)
        jv = arm.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
        if jv is not None:
            ok.append(i)
    print(f"  {label:<28s}: direct IK {len(ok)}/{len(gc)} -> {ok[:20]}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asmdef",
        default=os.path.join(SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef"),
    )
    parser.add_argument(
        "--grasp",
        default=os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp", "tower_base_plate_grasps.pickle"),
    )
    parser.add_argument(
        "--fixture-pos",
        default="0.36,0,0",
    )
    parser.add_argument(
        "--robot-base-pos",
        default="0,0,0",
    )
    parser.add_argument(
        "--staging-pos",
        default="",
        help="可选 staging obj pos，例如 0.36,0,0.003。为空则用 goal+[0,0,0.003]",
    )
    args = parser.parse_args()

    fixture_pos = np.array([float(x) for x in args.fixture_pos.split(",")], dtype=float)
    robot_base_pos = np.array([float(x) for x in args.robot_base_pos.split(",")], dtype=float)

    asm = AssemblyDef.load(args.asmdef)
    world_poses = asm.compute_world_poses(
        fixture_pos=fixture_pos,
        fixture_rotmat=np.eye(3),
    )

    if "base_plate" not in world_poses:
        raise RuntimeError("asmdef 中没有 base_plate world pose")

    goal_pos, goal_rot = world_poses["base_plate"]
    goal_pos = np.asarray(goal_pos, dtype=float)
    goal_rot = np.asarray(goal_rot, dtype=float)

    if args.staging_pos.strip():
        staging_pos = np.array([float(x) for x in args.staging_pos.split(",")], dtype=float)
    else:
        staging_pos = goal_pos + np.array([0.0, 0.0, 0.003], dtype=float)
    staging_rot = goal_rot.copy()

    gc = GraspCollection.load_from_disk(file_name=args.grasp)

    robot = pda.DualPantheraHTNoBody(
        pos=robot_base_pos,
        rotmat=np.eye(3),
        arm_y_offset=DUAL_ARM_Y_OFFSET,
        enable_cc=True,
    )
    robot.lft_arm.goto_given_conf(HOME_JV)
    robot.rgt_arm.goto_given_conf(HOME_JV)

    print("========== base_plate grasp debug ==========")
    print(f"asmdef       = {args.asmdef}")
    print(f"grasp        = {args.grasp}")
    print(f"grasp num    = {len(gc)}")
    print(f"fixture_pos  = {fixture_pos.tolist()}")
    print(f"robot_base   = {robot_base_pos.tolist()}")
    print(f"goal_pos     = {goal_pos.tolist()}")
    print(f"staging_pos  = {staging_pos.tolist()}")
    print()

    for arm_name, arm in [("lft", robot.lft_arm), ("rgt", robot.rgt_arm)]:
        print(f"----- arm = {arm_name} -----")
        _direct_ik_count(arm, gc, goal_pos, goal_rot, "goal only direct IK")
        _direct_ik_count(arm, gc, staging_pos, staging_rot, "staging only direct IK")
        _test_pose_list(arm, gc, [(goal_pos, goal_rot)], "reason goal only")
        _test_pose_list(arm, gc, [(staging_pos, staging_rot)], "reason staging only")
        _test_pose_list(arm, gc, [(staging_pos, staging_rot), (goal_pos, goal_rot)], "reason staging->goal")
        print()

    print("结论判断：")
    print("1. 如果 goal only 和 staging only 都是 0：抓取姿态本身对机器人不可达。")
    print("2. 如果单点都有，但 staging->goal 是 0：两个位姿没有共同 gid，抓取库太少/过滤太严格。")
    print("3. 如果 staging->goal > 0：layout 搜索脚本的 obstacle/碰撞逻辑才是问题。")


if __name__ == "__main__":
    main()
