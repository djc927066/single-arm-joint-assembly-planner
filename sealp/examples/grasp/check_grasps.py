#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/4/20 17:29
# @Author : ZhangXi

import os
import pickle
import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg

def main():
    # 1. Define data and model paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(current_dir,"tower_grasp", "tower_middle_plate_grasps.pickle")

    # Ensure this path matches the object model used in planning.py
    obj_path = r"D:\Project\wrs-sealp\sealp\assets\models\Toy\model\middle_plate.stl"

    # 2. Check if the Pickle file exists
    if not os.path.exists(pickle_path):
        print("\n" + "-" * 60)
        print("[ERROR] Grasp data file not found.")
        print(f"Expected path: {pickle_path}")
        print("\nPlease execute planning.py first to generate grasp data:")
        print("    python planning.py")
        print("-" * 60 + "\n")
        return

    # 3. Load the grasp data
    print(f"\n[INFO] Grasp data file found: {pickle_path}")
    print("[INFO] Loading data...")

    with open(pickle_path, 'rb') as f:
        grasp_collection = pickle.load(f)

    print(f"[INFO] Successfully loaded {len(grasp_collection)} valid grasp poses.")

    # 4. Initialize Panda3D visualization
    print("\n[INFO] Starting 3D visualization interface...")
    base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
    mgm.gen_frame(ax_length=0.3).attach_to(base)

    # Load and attach the object model
    if not os.path.exists(obj_path):
        print(f"[WARNING] Object model not found at: {obj_path}")
        print("[WARNING] Only the gripper poses will be rendered in the scene.")
    else:
        obj_cmodel = mcm.CollisionModel(obj_path)
        obj_cmodel.rgba = np.array([0.6, 0.5, 0.4, 1.0])
        obj_cmodel.attach_to(base)

    gripper = pg.PantheraGripper()
    max_show = 10


    print("\n" + "=" * 60)
    print(f"[INFO] Displaying detailed kinematic data for the top {min(max_show, len(grasp_collection))} grasps:")
    print("=" * 60)

    for i, grasp in enumerate(grasp_collection):
        if i >= max_show:
            break

        # ------------------------------------------------------------------
        # Console Logging: Print detailed information for the current grasp
        # ------------------------------------------------------------------
        print(f"\n[ Grasp Configuration #{i + 1} ]")

        # Print Action Center Position (Translation)
        print(f"  - Action Center Position (x, y, z) [m]:")
        print(f"      {np.round(grasp.ac_pos, 5)}")

        # Print Action Center Rotation Matrix (Orientation)
        print(f"  - Action Center Rotation Matrix:")
        rot_mat = np.round(grasp.ac_rotmat, 5)
        for row in rot_mat:
            print(f"      {row}")

        # Print End-Effector Values (Jaw Width)
        if hasattr(grasp, 'ee_values'):
            print(f"  - Gripper Jaw Opening Width [m]: {grasp.ee_values}")

        print("-" * 45)
        # ------------------------------------------------------------------

        # Move the gripper to the target pose and set jaw width
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)

        # Generate and attach the mesh model with transparency
        gripper.gen_meshmodel(alpha=0.6).attach_to(base)

    print(f"\n[INFO] Rendered the top {min(max_show, len(grasp_collection))} grasp poses in the 3D scene.")
    print("[INFO] Press ESC to close the visualization window.")

    base.run()


if __name__ == "__main__":
    main()