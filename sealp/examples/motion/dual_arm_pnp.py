"""
Dual-Arm Pick-and-Place Demo
==============================

Demonstrates coordinated dual-arm pick-and-place with two Piper arms:

- **Right arm** picks an object from one location and places it at a goal.
- **Left arm** simultaneously picks a different object and places it elsewhere.
- Both motions are animated concurrently in Panda3D.

This example uses ``DualPiperNoBody`` which provides two independent
Piper arms (``rgt_arm`` and ``lft_arm``), each with its own planner.

Usage::

    python -m sealp.examples.motion.dual_arm_pnp

Adapted from tiaozhanbei/task_sim/pick_and_place_chair - dual.py
"""

import os
import numpy as np
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg
import wrs.robot_sim.robots.piper.piper_dual_arm as pda
from direct.task.TaskManagerGlobal import taskMgr


# ======================================================================
# Animation helpers
# ======================================================================
def animate_dual(base, mot_data_r, mot_data_l, interval=0.01):
    """Animate two MotionData streams simultaneously (one per arm).

    Press **Space** to step both arms forward together.

    Parameters
    ----------
    base : wd.World
        The Panda3D world.
    mot_data_r, mot_data_l : MotionData
        Right-arm and left-arm motion data.
    interval : float
        Delay between frames in seconds.
    """

    class _Data:
        def __init__(self, md):
            self.counter = 0
            self.mot_data = md

    def _make_updater(anim_data, name):
        def _update(ad, task):
            if ad.counter > 0:
                ad.mot_data.mesh_list[ad.counter - 1].detach()
            if ad.counter >= len(ad.mot_data):
                for m in ad.mot_data.mesh_list:
                    m.detach()
                ad.counter = 0
            mesh = ad.mot_data.mesh_list[ad.counter]
            mesh.attach_to(base)
            if base.inputmgr.keymap['space']:
                ad.counter += 1
            return task.again

        taskMgr.doMethodLater(interval, _update, name,
                              extraArgs=[anim_data], appendTask=True)

    _make_updater(_Data(mot_data_r), "animate_rgt")
    _make_updater(_Data(mot_data_l), "animate_lft")


# ======================================================================
# Demo
# ======================================================================
def main():
    """Dual-arm pick-and-place demo with two Piper arms."""
    # ------------------------------------------------------------------
    # 1. Scene setup
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=[1.2, 0.7, 1.0], lookat_pos=[0, 0, 0.15])
    mgm.gen_frame().attach_to(base)

    # Ground plane
    ground = mcm.gen_box(xyz_lengths=rm.vec(2, 2, 0.01),
                         rgb=rm.vec(0.75, 0.75, 0.75), alpha=1)
    ground.pos = np.array([0.3, 0, -0.005])
    ground.attach_to(base)

    # ------------------------------------------------------------------
    # 2. Objects — two boxes, one for each arm
    # ------------------------------------------------------------------
    # Right arm object
    obj_r = mcm.gen_box(xyz_lengths=np.array([0.06, 0.04, 0.03]))
    obj_r.rgba = np.array([0.7, 0.4, 0.3, 1.0])
    pick_pos_r = np.array([0.25, -0.50, 0.015])
    obj_r.pos = pick_pos_r
    obj_r.rotmat = np.eye(3)
    obj_r.attach_to(base)

    # Left arm object
    obj_l = mcm.gen_box(xyz_lengths=np.array([0.06, 0.04, 0.03]))
    obj_l.rgba = np.array([0.3, 0.4, 0.7, 1.0])
    pick_pos_l = np.array([0.25, 0.10, 0.015])
    obj_l.pos = pick_pos_l
    obj_l.rotmat = np.eye(3)
    obj_l.attach_to(base)

    # Goal poses
    goal_pos_r = np.array([0.35, -0.70, 0.015])
    goal_rot_r = rm.rotmat_from_euler(0, 0, 0)
    goal_pos_l = np.array([0.35, 0.10, 0.015])
    goal_rot_l = rm.rotmat_from_euler(0, 0, 0)

    # Show goal ghosts
    for obj, gp, gr in [(obj_r, goal_pos_r, goal_rot_r),
                         (obj_l, goal_pos_l, goal_rot_l)]:
        ghost = obj.copy()
        ghost.pos = gp
        ghost.rotmat = gr
        ghost.alpha = 0.3
        ghost.attach_to(base)

    # ------------------------------------------------------------------
    # 3. Dual-arm robot
    # ------------------------------------------------------------------
    robot = pda.DualPiperNoBody()
    rbt_r = robot.rgt_arm
    rbt_l = robot.lft_arm

    # Home pose
    rbt_r.goto_given_conf(np.zeros(6))
    rbt_l.goto_given_conf(np.zeros(6))
    rbt_r.gen_meshmodel(alpha=0.2).attach_to(base)
    rbt_l.gen_meshmodel(alpha=0.2).attach_to(base)

    # ------------------------------------------------------------------
    # 4. Planners
    # ------------------------------------------------------------------
    ppp_r = ppp.PickPlacePlanner(rbt_r)
    ppp_l = ppp.PickPlacePlanner(rbt_l)

    # ------------------------------------------------------------------
    # 5. Grasps
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    grasp_path = os.path.join(out_dir, "demo_box_grasps.pickle")

    if os.path.isfile(grasp_path):
        print(f"Loading grasps from {grasp_path}")
        grasp_collection = gg.GraspCollection.load_from_disk(
            file_name=grasp_path)
    else:
        from sealp.examples.grasp.planning import plan_grasps
        temp_pos = obj_r.pos
        temp_rotmat = obj_r.rotmat
        obj_r.pos = np.zeros(3)
        obj_r.rotmat = np.eye(3)
        # 抓取应该对放在基座标的物体进行！！！
        grasp_collection, _ = plan_grasps(obj_r, max_samples=50)
        obj_r.pos = temp_pos
        obj_r.rotmat = temp_rotmat

        os.makedirs(out_dir, exist_ok=True)
        grasp_collection.save_to_disk(file_name=grasp_path)

    # ------------------------------------------------------------------
    # 6. Plan motions for both arms
    # ------------------------------------------------------------------
    obs_list = [ground]

    # --- Right arm ---
    print("Planning right-arm motion...")
    mot_r = ppp_r.gen_pick_and_place(
        obj_cmodel=obj_r,
        end_jnt_values=rbt_r.get_jnt_values(),
        grasp_collection=grasp_collection,
        goal_pose_list=[(goal_pos_r, goal_rot_r)],
        pick_approach_distance=0.05,
        pick_depart_distance=0.05,
        pick_depart_direction=rm.const.z_ax,
        place_approach_distance_list=[0.05],
        place_depart_distance_list=[0.05],
        obstacle_list=obs_list,
        use_rrt=True,
    )

    # --- Left arm ---
    print("Planning left-arm motion...")
    mot_l = ppp_l.gen_pick_and_place(
        obj_cmodel=obj_l,
        end_jnt_values=rbt_l.get_jnt_values(),
        grasp_collection=grasp_collection,
        goal_pose_list=[(goal_pos_l, goal_rot_l)],
        pick_approach_distance=0.05,
        pick_depart_distance=0.05,
        pick_depart_direction=rm.const.z_ax,
        place_approach_distance_list=[0.05],
        place_depart_distance_list=[0.05],
        obstacle_list=obs_list,
        use_rrt=True,
    )

    # --- Check results ---
    if mot_r is None or mot_l is None:
        failed = []
        if mot_r is None:
            failed.append("right")
        if mot_l is None:
            failed.append("left")
        print(f"Motion planning FAILED for {', '.join(failed)} arm(s).")
        print("Try adjusting poses or grasps.")
        base.run()
        return

    print(f"Right arm: {len(mot_r)} frames")
    print(f"Left arm:  {len(mot_l)} frames")

    # ------------------------------------------------------------------
    # 7. Animate both arms simultaneously
    # ------------------------------------------------------------------
    print("Press SPACE to step both arms together.")
    animate_dual(base, mot_r, mot_l)
    base.run()


if __name__ == "__main__":
    main()
