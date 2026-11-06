#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/4/21 10:50
# @Author : ZhangXi
"""
Grasp Planning Example — Piper Gripper
=======================================

Demonstrates antipodal grasp planning on an object mesh using
the Piper gripper.  Generates a ``GraspCollection``, saves it to
a pickle file, and visualizes the grasps in Panda3D.

This is the first step in any pick-and-place pipeline:
    1. **Plan grasps** (this script)
    2. Filter / select grasps
    3. Use grasps in pick-and-place planning

Usage::

    python -m sealp.examples.grasp.planning

Adapted from tiaozhanbei/grasp/piper_gripper_planning.py
"""

import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wg3

def plan_grasps(obj_cmodel,
                gripper=None,
                angle_between_contact_normals=None,
                rotation_interval=None,
                max_samples=100,
                min_dist_between_sampled_contact_points=0.01,
                contact_offset=0.01,
                toggle_dbg=False):
    """Plan antipodal grasps on an object using the Piper gripper.

    Parameters
    ----------
    obj_cmodel : mcm.CollisionModel
        The object to plan grasps on.
    gripper : PiperGripper or None
        Gripper instance.  If None, a default PiperGripper is created.
    angle_between_contact_normals : float or None
        Max angle between contact normals (radians).
        Defaults to ``radians(175)``.
    rotation_interval : float or None
        Rotation sampling interval (radians).
        Defaults to ``radians(30)``.
    max_samples : int
        Maximum number of contact point samples.
    min_dist_between_sampled_contact_points : float
        Minimum distance between sampled contact points (meters).
    contact_offset : float
        Contact offset distance (meters).
    toggle_dbg : bool
        Show debug visualization during planning.

    Returns
    -------
    grasp_collection : GraspCollection
        The planned grasps.
    gripper : PiperGripper
        The gripper instance used.
    """
    if gripper is None:
        gripper = wg3.WRSGripper3()
    if angle_between_contact_normals is None:
        angle_between_contact_normals = rm.radians(175)
    if rotation_interval is None:
        rotation_interval = rm.radians(30)

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


def visualize_grasps(base, obj_cmodel, grasp_collection, gripper,
                     max_show=30, alpha=0.7):
    """Visualize grasps on an object in the Panda3D scene.

    Parameters
    ----------
    base : wd.World
        Panda3D world.
    obj_cmodel : mcm.CollisionModel
        The object model.
    grasp_collection : GraspCollection
        Grasps to visualize.
    gripper : PiperGripper
        Gripper to render at each grasp pose.
    max_show : int
        Maximum number of grasps to show (for performance).
    alpha : float
        Transparency of gripper models.
    """
    obj_cmodel.attach_to(base)

    for i, grasp in enumerate(grasp_collection):
        if i >= max_show:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat,
                                grasp.ee_values)
        gripper.gen_meshmodel(alpha=alpha).attach_to(base)


def main():
    """Run the grasp planning demo."""
    # ------------------------------------------------------------------
    # 1. Setup Panda3D world
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
    mgm.gen_frame(ax_length=0.3).attach_to(base)

    # ------------------------------------------------------------------
    # 2. Load object — use a simple box as demo object
    # ------------------------------------------------------------------
    obj_cmodel = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\yuanchair\yuanchair-part2.stl")
    obj_cmodel.rgba = np.array([0.6, 0.5, 0.4, 1.0])

    # ------------------------------------------------------------------
    # 3. Plan grasps
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Grasp Planning — Piper Gripper")
    print("=" * 50)
    print("Planning grasps on demo box...")

    grasp_collection, gripper = plan_grasps(
        obj_cmodel,
        max_samples=100,
        rotation_interval=rm.radians(30),
    )

    print(f"  Planned {len(grasp_collection)} grasps.")

    # ------------------------------------------------------------------
    # 4. Save grasps
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "demo_yuanchair-part2_wrs_grasps.pickle")
    grasp_collection.save_to_disk(file_name=save_path)
    print(f"  Saved to: {save_path}")

    # ------------------------------------------------------------------
    # 5. Visualize
    # ------------------------------------------------------------------
    print(f"  Showing first 30 grasps...")
    visualize_grasps(base, obj_cmodel, grasp_collection, gripper,
                     max_show=100)

    print("=" * 50)
    print("Press ESC to close.")
    base.run()


if __name__ == "__main__":
    main()
