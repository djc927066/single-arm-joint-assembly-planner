"""
Piper Pick-and-Place Wrapper
=============================

Convenience wrapper around the WRS ``PickPlacePlanner`` that
pre-configures a Piper arm and provides high-level
``pick_and_place`` functionality.

Supports both single-arm (``PiperSglArm``) and dual-arm
(``DualPiperNoBody``) configurations.

Usage (single arm)
------------------
>>> from sealp.examples.motion.piper_pnp import PiperPickAndPlace
>>> pnp = PiperPickAndPlace()
>>> mot = pnp.pick_and_place(obj, grasp_collection, goal_pose_list)

Usage (dual arm)
----------------
>>> from sealp.examples.motion.piper_pnp import DualPiperPickAndPlace
>>> dual = DualPiperPickAndPlace()
>>> mot_r, mot_l = dual.pick_and_place_both(...)

Adapted from tiaozhanbei/task_sim examples.
"""

import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.manipulation.pick_place as pp
import wrs.robot_sim.robots.piper.piper_single_arm as psa
import wrs.robot_sim.robots.piper.piper_dual_arm as pda


class PiperPickAndPlace:
    """High-level pick-and-place controller backed by a single Piper arm.

    Parameters
    ----------
    pos : np.ndarray
        Base position of the Piper arm in world coordinates.
    rotmat : np.ndarray
        Base orientation of the Piper arm (3x3 rotation matrix).
    enable_cc : bool
        Enable self-collision checking.
    name : str
        Identifier for this robot instance.
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 enable_cc=True,
                 name="piper_pnp"):
        self.robot = psa.PiperSglArm(pos=pos, rotmat=rotmat,
                                      name=name, enable_cc=enable_cc)
        self._planner = pp.PickPlacePlanner(robot=self.robot)

    @property
    def planner(self):
        return self._planner

    @property
    def arm(self):
        return self.robot

    def pick_and_place(self,
                       obj_cmodel,
                       grasp_collection,
                       goal_pose_list,
                       start_jnt_values=None,
                       end_jnt_values=None,
                       pick_approach_distance=0.05,
                       pick_depart_distance=0.05,
                       pick_depart_direction=None,
                       place_approach_distance_list=None,
                       place_depart_distance_list=None,
                       obstacle_list=None,
                       use_rrt=True,
                       toggle_dbg=False):
        """Plan a full pick-and-place motion.

        Parameters
        ----------
        obj_cmodel : CollisionModel
            The object to be picked (at its current pose).
        grasp_collection : GraspCollection
            Pre-computed grasps on the object.
        goal_pose_list : list of (pos, rotmat)
            Target poses for the object.
        pick_approach_distance : float
            Distance to approach before grasping (meters).
        pick_depart_distance : float
            Distance to lift after grasping (meters).
        pick_depart_direction : np.ndarray or None
            Direction to depart after pick.  Default: +Z axis.
        obstacle_list : list or None
            Collision models to avoid during planning.
        use_rrt : bool
            Use RRT for transit motion.

        Returns
        -------
        MotionData or None
            The planned motion, or None if planning fails.
        """
        n = len(goal_pose_list)
        if pick_depart_direction is None:
            pick_depart_direction = rm.const.z_ax
        if place_approach_distance_list is None:
            place_approach_distance_list = [0.05] * n
        if place_depart_distance_list is None:
            place_depart_distance_list = [0.05] * n
        if obstacle_list is None:
            obstacle_list = []
        if end_jnt_values is None:
            end_jnt_values = self.robot.get_jnt_values()

        return self._planner.gen_pick_and_place(
            obj_cmodel=obj_cmodel,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            start_jnt_values=start_jnt_values,
            end_jnt_values=end_jnt_values,
            pick_approach_distance=pick_approach_distance,
            pick_depart_distance=pick_depart_distance,
            pick_depart_direction=pick_depart_direction,
            place_approach_distance_list=place_approach_distance_list,
            place_depart_distance_list=place_depart_distance_list,
            obstacle_list=obstacle_list,
            use_rrt=use_rrt,
            toggle_dbg=toggle_dbg,
        )


class DualPiperPickAndPlace:
    """Dual-arm pick-and-place controller using two Piper arms.

    Uses ``DualPiperNoBody`` which provides independent ``rgt_arm``
    and ``lft_arm`` sub-robots, each with its own planner.

    Parameters
    ----------
    rgt_pos, lft_pos : np.ndarray
        Base positions for right and left arms.
    rgt_rotmat, lft_rotmat : np.ndarray
        Base orientations for right and left arms.
    """

    def __init__(self,
                 rgt_pos=np.array([0, -0.2, 0]),
                 rgt_rotmat=np.eye(3),
                 lft_pos=np.array([0, 0.2, 0]),
                 lft_rotmat=np.eye(3)):
        self.robot = pda.DualPiperNoBody()
        self.rgt = self.robot.rgt_arm
        self.lft = self.robot.lft_arm
        self._planner_r = pp.PickPlacePlanner(robot=self.rgt)
        self._planner_l = pp.PickPlacePlanner(robot=self.lft)

    def pick_and_place_right(self, obj_cmodel, grasp_collection,
                              goal_pose_list, obstacle_list=None,
                              **kwargs):
        """Plan pick-and-place for the right arm."""
        return self._plan_arm(self._planner_r, self.rgt,
                              obj_cmodel, grasp_collection,
                              goal_pose_list, obstacle_list,
                              **kwargs)

    def pick_and_place_left(self, obj_cmodel, grasp_collection,
                             goal_pose_list, obstacle_list=None,
                             **kwargs):
        """Plan pick-and-place for the left arm."""
        return self._plan_arm(self._planner_l, self.lft,
                              obj_cmodel, grasp_collection,
                              goal_pose_list, obstacle_list,
                              **kwargs)

    def pick_and_place_both(self,
                            obj_r, grasp_r, goal_r,
                            obj_l, grasp_l, goal_l,
                            obstacle_list=None, **kwargs):
        """Plan pick-and-place for both arms independently.

        Parameters
        ----------
        obj_r, obj_l : CollisionModel
            Objects for right and left arms.
        grasp_r, grasp_l : GraspCollection
            Grasps for right and left arms.
        goal_r, goal_l : list of (pos, rotmat)
            Goal poses for right and left arms.
        obstacle_list : list or None
            Shared obstacles.

        Returns
        -------
        mot_r, mot_l : MotionData or None
            Motion data for each arm (None if planning failed).
        """
        mot_r = self.pick_and_place_right(
            obj_r, grasp_r, goal_r, obstacle_list, **kwargs)
        mot_l = self.pick_and_place_left(
            obj_l, grasp_l, goal_l, obstacle_list, **kwargs)
        return mot_r, mot_l

    @staticmethod
    def _plan_arm(planner, arm, obj_cmodel, grasp_collection,
                  goal_pose_list, obstacle_list=None,
                  pick_approach_distance=0.05,
                  pick_depart_distance=0.05,
                  pick_depart_direction=None,
                  place_approach_distance_list=None,
                  place_depart_distance_list=None,
                  use_rrt=True, **kwargs):
        n = len(goal_pose_list)
        if pick_depart_direction is None:
            pick_depart_direction = rm.const.z_ax
        if place_approach_distance_list is None:
            place_approach_distance_list = [0.05] * n
        if place_depart_distance_list is None:
            place_depart_distance_list = [0.05] * n
        if obstacle_list is None:
            obstacle_list = []

        return planner.gen_pick_and_place(
            obj_cmodel=obj_cmodel,
            end_jnt_values=arm.get_jnt_values(),
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            pick_approach_distance=pick_approach_distance,
            pick_depart_distance=pick_depart_distance,
            pick_depart_direction=pick_depart_direction,
            place_approach_distance_list=place_approach_distance_list,
            place_depart_distance_list=place_depart_distance_list,
            obstacle_list=obstacle_list,
            use_rrt=use_rrt,
            **kwargs,
        )
