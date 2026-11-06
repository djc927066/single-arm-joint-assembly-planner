"""
Dual-Arm Cooperative Transport Primitive
==========================================

Both arms coordinate to carry a single (large/heavy) object.
Each arm independently plans a pick-and-place using the WRS
``PickPlacePlanner``, grasping the same object from different sides.

Strategy:
    1. Plan right arm: pick object → transport → place at goal.
    2. Plan left arm: pick same object (different grasp) → place at goal.
    3. Return both motion data streams for synchronized animation.

This follows the same independent-planning approach used in
``sealp.examples.motion.dual_arm_pnp``.

Usage::

    from sealp.primitives import DualTransportPrimitive

    dual = DualTransportPrimitive(robot_rgt, robot_lft)
    result = dual.plan(obj_cmodel, grasp_collection,
                       goal_pose_list=[(pos, rotmat)],
                       obstacle_list=[ground])
    if result.success:
        animate_dual(base, result.mot_data_rgt, result.mot_data_lft)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import wrs.basis.robot_math as rm
import wrs.manipulation.pick_place as pp

from .base import MotionPrimitive, PrimitiveResult


class DualTransportPrimitive(MotionPrimitive):
    """Dual-arm cooperative transport primitive.

    Both arms plan independently to pick, transport, and place
    the same object.  This is suitable for large/heavy parts that
    require two-arm support.

    Parameters
    ----------
    robot_rgt : SglArmRobotInterface
        Right arm robot.
    robot_lft : SglArmRobotInterface
        Left arm robot.
    """

    def __init__(self, robot_rgt, robot_lft):
        self.robot_rgt = robot_rgt
        self.robot_lft = robot_lft
        self._planner_rgt = pp.PickPlacePlanner(robot=robot_rgt)
        self._planner_lft = pp.PickPlacePlanner(robot=robot_lft)

    def plan(self,
             obj_cmodel,
             grasp_collection,
             goal_pose_list: List[Tuple[np.ndarray, np.ndarray]],
             start_jnt_values: Optional[np.ndarray] = None,
             end_jnt_values: Optional[np.ndarray] = None,
             obstacle_list: Optional[List] = None,
             approach_distance: float = 0.05,
             depart_distance: float = 0.05,
             use_rrt: bool = True,
             **kwargs) -> PrimitiveResult:
        """Plan dual-arm cooperative transport.

        Both arms independently plan pick-and-place for the same
        object.  ``start_jnt_values`` and ``end_jnt_values`` are
        ignored (each arm uses its own current configuration).

        Parameters
        ----------
        obj_cmodel
            Object collision model at its staging (pick) pose.
        grasp_collection
            Pre-computed grasps. Both arms select from these.
        goal_pose_list
            Target placement poses ``[(pos, rotmat), ...]``.
        obstacle_list
            Shared collision models to avoid.
        approach_distance
            Approach distance (meters).
        depart_distance
            Depart distance (meters).
        use_rrt
            Use RRT for transit.

        Keyword Arguments
        -----------------
        grasp_collection_lft : GraspCollection, optional
            Separate grasp collection for the left arm.
            If not provided, uses the same ``grasp_collection``.
        start_jnt_values_rgt : np.ndarray, optional
            Starting config for right arm.
        start_jnt_values_lft : np.ndarray, optional
            Starting config for left arm.
        end_jnt_values_rgt : np.ndarray, optional
            Ending config for right arm.
        end_jnt_values_lft : np.ndarray, optional
            Ending config for left arm.

        Returns
        -------
        PrimitiveResult
            Contains ``mot_data_rgt`` and ``mot_data_lft``.
        """
        if obstacle_list is None:
            obstacle_list = []

        grasp_collection_lft = kwargs.get(
            "grasp_collection_lft", grasp_collection)

        n = len(goal_pose_list)
        pick_depart_direction = kwargs.get(
            "pick_depart_direction", rm.const.z_ax)
        place_depart_direction_list = kwargs.get(
            "place_depart_direction_list", [rm.const.z_ax] * n)
        place_approach_distance_list = kwargs.get(
            "place_approach_distance_list", [approach_distance] * n)
        pick_depart_dist = kwargs.get("pick_depart_distance", depart_distance)
        place_depart_distance_list = kwargs.get(
            "place_depart_distance_list", [depart_distance] * n)

        # ── Plan right arm ───────────────────────────────────
        end_jnt_rgt = kwargs.get(
            "end_jnt_values_rgt", self.robot_rgt.get_jnt_values())
        start_jnt_rgt = kwargs.get("start_jnt_values_rgt", None)

        mot_rgt = self._planner_rgt.gen_pick_and_place(
            obj_cmodel=obj_cmodel,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            start_jnt_values=start_jnt_rgt,
            end_jnt_values=end_jnt_rgt,
            pick_approach_distance=approach_distance,
            pick_depart_distance=pick_depart_dist,
            pick_depart_direction=pick_depart_direction,
            place_approach_distance_list=place_approach_distance_list,
            place_depart_direction_list=place_depart_direction_list,
            place_depart_distance_list=place_depart_distance_list,
            obstacle_list=obstacle_list,
            use_rrt=use_rrt,
        )

        if mot_rgt is None:
            return PrimitiveResult(
                success=False,
                error_msg="DualTransport: right arm planning failed.",
            )

        # ── Plan left arm ────────────────────────────────────
        end_jnt_lft = kwargs.get(
            "end_jnt_values_lft", self.robot_lft.get_jnt_values())
        start_jnt_lft = kwargs.get("start_jnt_values_lft", None)

        mot_lft = self._planner_lft.gen_pick_and_place(
            obj_cmodel=obj_cmodel,
            grasp_collection=grasp_collection_lft,
            goal_pose_list=goal_pose_list,
            start_jnt_values=start_jnt_lft,
            end_jnt_values=end_jnt_lft,
            pick_approach_distance=approach_distance,
            pick_depart_distance=pick_depart_dist,
            pick_depart_direction=pick_depart_direction,
            place_approach_distance_list=place_approach_distance_list,
            place_depart_direction_list=place_depart_direction_list,
            place_depart_distance_list=place_depart_distance_list,
            obstacle_list=obstacle_list,
            use_rrt=use_rrt,
        )

        if mot_lft is None:
            return PrimitiveResult(
                success=False,
                mot_data_rgt=mot_rgt,
                error_msg="DualTransport: left arm planning failed "
                          "(right arm succeeded).",
            )

        return PrimitiveResult(
            success=True,
            mot_data_rgt=mot_rgt,
            mot_data_lft=mot_lft,
            end_jnt_values_rgt=np.asarray(mot_rgt.jv_list[-1]),
            end_jnt_values_lft=np.asarray(mot_lft.jv_list[-1]),
        )
