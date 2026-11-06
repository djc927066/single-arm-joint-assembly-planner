"""
Single-Arm Transport Primitive
================================

Pick an object from its current (staging) pose, transport it through
free space, and place it at the goal assembly pose.  Wraps the WRS
``PickPlacePlanner.gen_pick_and_place()`` method.

Usage::

    from sealp.primitives import TransportPrimitive

    transport = TransportPrimitive(robot_arm)
    result = transport.plan(obj_cmodel, grasp_collection,
                            goal_pose_list=[(pos, rotmat)],
                            obstacle_list=[ground])
    if result.success:
        animate(base, result.mot_data)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import wrs.basis.robot_math as rm
import wrs.manipulation.pick_place as pp

from .base import MotionPrimitive, PrimitiveResult


class TransportPrimitive(MotionPrimitive):
    """Single-arm pick-transport-place primitive.

    Parameters
    ----------
    robot : SglArmRobotInterface
        A single robot arm (e.g. ``PiperSglArm``).
    """

    def __init__(self, robot):
        self.robot = robot
        self._planner = pp.PickPlacePlanner(robot=robot)

    @property
    def planner(self):
        return self._planner

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
        """Plan a single-arm pick-and-place motion.

        Parameters
        ----------
        obj_cmodel
            Object collision model at its staging (pick) pose.
        grasp_collection
            Pre-computed grasps for the object.
        goal_pose_list
            Target placement poses ``[(pos, rotmat), ...]``.
        start_jnt_values
            Starting joint config. ``None`` = current robot config.
        end_jnt_values
            Ending joint config. ``None`` = current robot config.
        obstacle_list
            List of collision models to avoid.
        approach_distance
            Approach distance (meters).
        depart_distance
            Depart distance (meters).
        use_rrt
            Use RRT for transit.

        Returns
        -------
        PrimitiveResult
        """
        if obstacle_list is None:
            obstacle_list = []
        # WRS itself interprets ``end_jnt_values=None`` as "finish at the
        # end of the place-depart motion".  Historically this wrapper replaced
        # None with the current configuration, which made a caller unable to
        # build continuous no-return task chains.  Keep the legacy default,
        # but provide an explicit opt-in used by the dual-arm continuous
        # sequence evaluator.
        finish_at_place_depart = bool(
            kwargs.get("finish_at_place_depart", False))
        if end_jnt_values is None and not finish_at_place_depart:
            end_jnt_values = self.robot.get_jnt_values()

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

        grasp_obstacle_list = kwargs.get("grasp_obstacle_list")
        gpp_kwargs = dict(
            obj_cmodel=obj_cmodel,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            start_jnt_values=start_jnt_values,
            end_jnt_values=end_jnt_values,
            pick_approach_distance=approach_distance,
            pick_depart_distance=pick_depart_dist,
            pick_depart_direction=pick_depart_direction,
            place_approach_distance_list=place_approach_distance_list,
            place_depart_direction_list=place_depart_direction_list,
            place_depart_distance_list=place_depart_distance_list,
            obstacle_list=obstacle_list,
            grasp_obstacle_list=grasp_obstacle_list,
            use_rrt=use_rrt,
        )
        pick_app_dir = kwargs.get("pick_approach_direction")
        if pick_app_dir is not None:
            gpp_kwargs["pick_approach_direction"] = pick_app_dir
        pa_dir = kwargs.get("place_approach_direction_list")
        if pa_dir is not None:
            gpp_kwargs["place_approach_direction_list"] = pa_dir
        # 透传"宽松度"相关参数到 PickPlacePlanner
        for _key in ("linear_granularity", "reason_grasps"):
            if _key in kwargs and kwargs[_key] is not None:
                gpp_kwargs[_key] = kwargs[_key]
        mot_data = self._planner.gen_pick_and_place(**gpp_kwargs)

        if mot_data is None:
            return PrimitiveResult(
                success=False,
                error_msg="PickPlacePlanner failed: no valid grasp/path found.",
            )

        return PrimitiveResult(
            success=True,
            mot_data=mot_data,
            end_jnt_values=mot_data.jv_list[-1],
        )
