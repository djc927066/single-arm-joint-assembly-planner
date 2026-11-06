#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 middle_plate 生成 hopg / regspot pickle（Panthera 双臂）

输出::
    sealp/examples/grasp/tower_handover/middle_plate_hopg.pickle
    sealp/examples/grasp/tower_regspot/middle_plate_regspot.pickle

运行（tower，默认）::
    python -m sealp.examples.grasp.gen_middle_plate_regrasp_data

运行（totem，mesh + grasp 必须与执行时一致）::
    python -m sealp.examples.grasp.gen_middle_plate_regrasp_data ^
      --mesh sealp/assets/models/Totem/model/middle_plate.stl ^
      --grasp-pickle sealp/examples/grasp/totem_grasp/tower_middle_plate_grasps.pickle ^
      --handover-dir sealp/examples/grasp/totem_handover ^
      --regspot-dir sealp/examples/grasp/totem_regspot
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.manipulation.placement.flatsurface as mp_fsp
import wrs.manipulation.placement.handover as mp_hop
import wrs.visualization.panda.world as wd
from wrs.grasping.grasp import GraspCollection

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("无法找到 sealp 根目录")
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DUAL_ARM_Y_OFFSET = 0.62
GRASP_PICKLE = os.path.join(
    SEALP_ROOT, "examples", "grasp", "tower_grasp", "tower_middle_plate_grasps.pickle"
)
MESH_CANDIDATES = [
    os.path.join(SEALP_ROOT, "assets", "models", "Toy", "model", "middle_plate.stl"),
    os.path.join(SEALP_ROOT, "assets", "models", "Toy", "tower", "middle_plate.stl"),
]
HANDOVER_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_handover")
REGSPOT_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_regspot")
HOPG_OUT = os.path.join(HANDOVER_DIR, "middle_plate_hopg.pickle")
REGSPOT_OUT = os.path.join(REGSPOT_DIR, "middle_plate_regspot.pickle")

# 桌面换手/重抓取候选点（tower layout 双臂之间）
REGSPOT_SPECS = [
    (np.array([0.22, -0.28, 0.0]), 0.0),
    (np.array([0.32, -0.28, 0.0]), 0.0),
    (np.array([0.42, -0.28, 0.0]), 0.0),
    (np.array([0.30, -0.42, 0.0]), np.pi / 2),
    (np.array([0.38, -0.38, 0.0]), np.pi / 4),
]


def _resolve_mesh_path() -> str:
    for p in MESH_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"找不到 middle_plate.stl，已尝试: {MESH_CANDIDATES}")


def _hop_orientations(spot_rotz: float) -> list:
    rots = [rm.rotmat_from_euler(0, 0, spot_rotz)]
    rots.append(rm.rotmat_from_euler(0, 0, spot_rotz + np.pi / 2))
    return rots


def _parse_args():
    parser = argparse.ArgumentParser(
        description="为 middle_plate 生成 hopg / regspot pickle（Panthera 双臂）"
    )
    parser.add_argument(
        "--mesh",
        default="",
        help="middle_plate STL；默认自动查找 Toy/tower 或 Toy/model",
    )
    parser.add_argument(
        "--grasp-pickle",
        default=GRASP_PICKLE,
        help="middle_plate grasp pickle（必须与执行时 --grasp-dir 内文件一致）",
    )
    parser.add_argument(
        "--handover-dir",
        default=HANDOVER_DIR,
        help="输出 middle_plate_hopg.pickle 的目录",
    )
    parser.add_argument(
        "--regspot-dir",
        default=REGSPOT_DIR,
        help="输出 middle_plate_regspot.pickle 的目录",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    handover_dir = os.path.abspath(args.handover_dir)
    regspot_dir = os.path.abspath(args.regspot_dir)
    hopg_out = os.path.join(handover_dir, "middle_plate_hopg.pickle")
    regspot_out = os.path.join(regspot_dir, "middle_plate_regspot.pickle")
    grasp_pickle = os.path.abspath(args.grasp_pickle)

    os.makedirs(handover_dir, exist_ok=True)
    os.makedirs(regspot_dir, exist_ok=True)

    mesh_path = os.path.abspath(args.mesh) if args.mesh else _resolve_mesh_path()
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"找不到 middle_plate mesh: {mesh_path}")
    if not os.path.isfile(grasp_pickle):
        raise FileNotFoundError(
            f"缺少 grasp pickle: {grasp_pickle}\n"
            "请先运行对应装配体的 grasp 规划，例如:\n"
            "  python -m sealp.examples.grasp.planning_tower --only middle_plate --no-vis"
        )

    print("=" * 70)
    print("Generate middle_plate regrasp data")
    print(f"mesh       = {mesh_path}")
    print(f"grasp      = {grasp_pickle}")
    print(f"hopg out   = {hopg_out}")
    print(f"regspot out= {regspot_out}")
    print("=" * 70)

    # FSRegSpotCollection.add_new_spot 内部会 attach_to(base)，需注入 World
    base = wd.World(cam_pos=[1.2, -1.0, 0.8], lookat_pos=[0.3, -0.3, 0.0])
    mp_fsp.base = base

    obj_cm = mcm.CollisionModel(mesh_path)
    gc = GraspCollection.load_from_disk(file_name=grasp_pickle)
    print(f"loaded grasps: n={len(gc)}")

    robot = pda.DualPantheraHTNoBody(arm_y_offset=DUAL_ARM_Y_OFFSET, enable_cc=True)

    fs_ref = mp_fsp.FSReferencePoses(obj_cmodel=obj_cm)
    print(f"fs reference poses: n={len(fs_ref)}")

    fs_coll = mp_fsp.FSRegSpotCollection(
        robot=robot.lft_arm,
        obj_cmodel=obj_cm,
        fs_reference_poses=fs_ref,
        reference_gc=gc,
    )
    for spot_pos, spot_rotz in REGSPOT_SPECS:
        fs_coll.add_new_spot(
            spot_pos=np.asarray(spot_pos, dtype=float),
            spot_rotz=float(spot_rotz),
            barrier_z_offset=-0.01,
            consider_robot=True,
            toggle_dbg=False,
        )
        n_fspg = len(fs_coll[-1].fspg_list) if len(fs_coll) else 0
        print(f"  regspot @ {np.round(spot_pos, 3).tolist()} rotz={np.degrees(spot_rotz):.0f}°  fspg={n_fspg}")

    if len(fs_coll) == 0 or all(len(s.fspg_list) == 0 for s in fs_coll):
        raise RuntimeError("未生成任何可行 regspot，请调整 REGSPOT_SPECS 或检查 grasp。")

    fs_coll.save_to_disk(regspot_out)
    print(f"[OK] saved regspot: {regspot_out}  spots={len(fs_coll)}")

    hopg = mp_hop.HOPGCollection(
        obj_cmodel=obj_cm,
        sender_robot=robot.lft_arm,
        receiver_robot=robot.rgt_arm,
        sender_reference_gc=gc,
        receiver_reference_gc=copy.deepcopy(gc),
    )
    hop_count = 0
    for spot_pos, spot_rotz in REGSPOT_SPECS:
        for rotmat in _hop_orientations(spot_rotz):
            n_before = len(hopg)
            hopg.add_new_hop(
                pos=np.asarray(spot_pos, dtype=float),
                rotmat=rotmat,
                obstacle_list=[],
                consider_robot=True,
                toggle_dbg=False,
            )
            added = len(hopg) - n_before
            hop_count += added
            print(
                f"  hop @ {np.round(spot_pos, 3).tolist()} "
                f"added={added} total={len(hopg)}"
            )

    if len(hopg) == 0:
        raise RuntimeError("未生成任何可行 HOPG，请调整候选点。")

    hopg.save_to_disk(hopg_out)
    print(f"[OK] saved hopg: {hopg_out}  hopg={len(hopg)}  attempts_added={hop_count}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
