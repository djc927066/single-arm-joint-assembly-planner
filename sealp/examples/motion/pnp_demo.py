"""
Single-Arm Pick-and-Place Demo
================================

Demonstrates a complete pick-and-place pipeline with the Piper arm:

1. Load object mesh and pre-computed grasps (or plan new ones)
2. Define pick and place poses
3. Plan collision-free pick-and-place motions (RRT-based)
4. Animate the result in Panda3D (press **Space** to step)

This is the core motion-planning example. For grasp planning, see
``sealp.examples.grasp``.

Usage::

    python -m sealp.examples.motion.pnp_demo

Adapted from tiaozhanbei/task_sim/pick_and_place_chair.py
"""

import os
import numpy as np
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg
import wrs.robot_sim.robots.piper.piper_single_arm as psa
from direct.task.TaskManagerGlobal import taskMgr


# ======================================================================
# Animation helpers
# ======================================================================
class CombinedMotData:
    """Concatenates multiple MotionData objects for sequential playback."""

    def __init__(self, mot_list):
        self.mesh_list = []
        self.jv_list = []
        for m in mot_list:
            self.mesh_list.extend(m.mesh_list)
            self.jv_list.extend(m.jv_list)

    def __len__(self):
        return len(self.mesh_list)


def animate(base, mot_data, interval=0.01):
    """Animate a MotionData in Panda3D.  Press Space to step forward.

    Parameters
    ----------
    base : wd.World
        The Panda3D world.
    mot_data : MotionData or CombinedMotData
        Motion data containing ``mesh_list``.
    interval : float
        Delay between frames in seconds.
    """

    class _Data:
        def __init__(self, md):
            self.counter = 0
            self.mot_data = md

    anim = _Data(mot_data)

    def _update(anim_data, task):
        if anim_data.counter > 0:
            anim_data.mot_data.mesh_list[anim_data.counter - 1].detach()
        if anim_data.counter >= len(anim_data.mot_data):
            # Loop back
            for m in anim_data.mot_data.mesh_list:
                m.detach()
            anim_data.counter = 0
        mesh = anim_data.mot_data.mesh_list[anim_data.counter]
        mesh.attach_to(base)
        if base.inputmgr.keymap['space']:
            anim_data.counter += 1
        return task.again

    taskMgr.doMethodLater(interval, _update, "pnp_animate",
                          extraArgs=[anim], appendTask=True)


# ======================================================================
# Demo
# ======================================================================
def main():
    """Single-arm pick-and-place demo with Piper."""
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
    # 2. Object — demo box
    # ------------------------------------------------------------------
    obj = mcm.gen_box(xyz_lengths=np.array([0.06, 0.04, 0.03]))
    obj.rgba = np.array([1, 0 ,0, 1.0])
    pick_pos = np.array([0.25, 0.20, 0.015])
    pick_rotmat = np.eye(3)
    obj.pos = pick_pos
    obj.rotmat = pick_rotmat
    obj.attach_to(base)

    # Goal poses (show semi-transparent ghosts)
    goal_pose_list = [
        (np.array([0.25, -0.20, 0.015]), rm.rotmat_from_euler(0, 0, 0)),
    ]
    for pos, rot in goal_pose_list:
        ghost = obj.copy()
        ghost.pos = pos
        ghost.rotmat = rot
        ghost.alpha = 0.3
        ghost.attach_to(base)

    # ------------------------------------------------------------------
    # 3. Robot + planners
    # ------------------------------------------------------------------
    robot = psa.PiperSglArm(enable_cc=True)
    robot.gen_meshmodel(alpha=0.3).attach_to(base)

    ppp_planner = ppp.PickPlacePlanner(robot)

    # ------------------------------------------------------------------
    # 4. Grasps — plan or load from cache
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    grasp_path = os.path.join(out_dir, "demo_box_grasps.pickle")

    if os.path.isfile(grasp_path):
        print(f"Loading grasps from {grasp_path}")
        grasp_collection = gg.GraspCollection.load_from_disk(
            file_name=grasp_path)
    else:
        # Plan grasps using the grasp module
        from sealp.examples.grasp.planning import plan_grasps
        import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
        gripper = pg.PiperGripper()
        temp_pos = obj.pos
        temp_rotmat = obj.rotmat
        obj.pos = np.zeros(3)
        obj.rotmat = np.eye(3)
        grasp_collection, _ = plan_grasps(obj, gripper=gripper, max_samples=100)
        obj.pos = temp_pos
        obj.rotmat = temp_rotmat
        os.makedirs(out_dir, exist_ok=True)
        grasp_collection.save_to_disk(file_name=grasp_path)

    # ------------------------------------------------------------------
    # 5. Plan pick-and-place motion
    # ------------------------------------------------------------------
    start_conf = robot.get_jnt_values()

    print("Planning pick-and-place motion...")

    for grasp in grasp_collection:
        robot.end_effector.grip_at_by_pose(obj.pos + obj.rotmat @ grasp.ac_pos,
                                           obj.rotmat @ grasp.ac_rotmat,
                                           grasp.ee_values)

    mot_data = ppp_planner.gen_pick_and_place(
        obj_cmodel=obj,
        end_jnt_values=start_conf,
        grasp_collection=grasp_collection,
        goal_pose_list=goal_pose_list,
        # Approach / depart distances
        pick_approach_distance=0.05,
        pick_depart_distance=0.05,
        pick_depart_direction=rm.const.z_ax,
        place_approach_distance_list=[0.05],
        place_depart_distance_list=[0.05],
        # Collision avoidance
        obstacle_list=[ground],
        use_rrt=True,
    )

    if mot_data is None:
        print("Motion planning FAILED. Try adjusting poses or grasps.")
        robot.gen_meshmodel().attach_to(base)
        base.run()
        return

    print(f"Motion planned successfully! ({len(mot_data)} frames)")

    # ------------------------------------------------------------------
    # 6. Animate
    # ------------------------------------------------------------------
    print("Press SPACE to step through the animation.")
    animate(base, mot_data)
    base.run()


if __name__ == "__main__":
    main()
