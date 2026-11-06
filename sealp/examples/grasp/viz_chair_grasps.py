"""
Visualize Grasps on YuanChair Parts
=====================================

Quick diagnostic script to plan and visualize grasps on the
actual yuanchair STL models (seat + leg).

Run::
    python -m sealp.examples.grasp.viz_chair_grasps
"""

import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg

from sealp.examples.grasp.planning import plan_grasps, visualize_grasps

# Model paths
ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "assets", "models", "yuanchair"))
SEAT_STL = os.path.join(ASSET_DIR, "yuanchair-part1.stl")
LEG_STL = os.path.join(ASSET_DIR, "yuanchair-part2.stl")


def main():
    base = wd.World(cam_pos=rm.vec(.5, .5, .3), lookat_pos=rm.vec(0, 0, 0))
    mgm.gen_frame(ax_length=0.1).attach_to(base)

    gripper = pg.PiperGripper()

    # ── Choose which part to visualize ───────────────────────
    # Change this to switch between seat and leg:
    # SHOW_PART = "seat"  # "seat" or "leg"


    SHOW_PART = "seat"

    if SHOW_PART == "seat":
        stl_path = SEAT_STL
        color = np.array([0.8, 0.6, 0.4, 1.0])
    else:
        stl_path = LEG_STL
        color = np.array([0.6, 0.5, 0.3, 1.0])

    print(f"Loading model: {stl_path}")
    if not os.path.isfile(stl_path):
        print(f"ERROR: File not found: {stl_path}")
        return

    obj = mcm.CollisionModel(initor=stl_path)
    obj.rgba = color

    # Print object info
    print(f"Object bounding box info:")
    # Show the object at origin to understand its geometry
    obj_copy = obj.copy()
    obj_copy.attach_to(base)

    # ── Plan grasps ──────────────────────────────────────────
    print(f"\nPlanning grasps on {SHOW_PART}...")
    grasp_collection, gripper = plan_grasps(
        obj, max_samples=200,
        rotation_interval=rm.radians(30),
    )
    print(f"Planned {len(grasp_collection)} grasps.\n")

    # ── Print grasp details ──────────────────────────────────
    for i, g in enumerate(grasp_collection):
        if i >= 10:
            print(f"  ... ({len(grasp_collection) - 10} more)")
            break
        print(f"  Grasp {i}: ac_pos={g.ac_pos}, "
              f"ee_values={g.ee_values:.4f}")

    # ── Visualize ────────────────────────────────────────────
    print(f"\nShowing {min(len(grasp_collection), 30)} grasps...")
    for i, grasp in enumerate(grasp_collection):
        if i >= 30:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat,
                                grasp.ee_values)
        gripper.gen_meshmodel(alpha=0.5).attach_to(base)

    print("Press ESC to close.")
    base.run()


if __name__ == "__main__":
    main()
