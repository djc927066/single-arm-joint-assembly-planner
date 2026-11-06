"""
Motion Primitive — Abstract Base
==================================

Defines the interface that all motion primitives must implement.
A primitive takes a collision model, grasps, pick/place poses, and
obstacles, and returns planned motion data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PrimitiveResult:
    """Result of a motion primitive execution.

    Attributes
    ----------
    success : bool
        Whether planning succeeded.
    mot_data : object or None
        Single-arm MotionData (for single-arm primitives).
    mot_data_rgt : object or None
        Right-arm MotionData (for dual-arm primitives).
    mot_data_lft : object or None
        Left-arm MotionData (for dual-arm primitives).
    end_jnt_values : np.ndarray or None
        Final joint configuration after the motion.
    end_jnt_values_rgt : np.ndarray or None
        Final right-arm joint config (dual-arm).
    end_jnt_values_lft : np.ndarray or None
        Final left-arm joint config (dual-arm).
    error_msg : str
        Description of failure if ``success`` is False.
    """
    success: bool = False
    mot_data: object = None
    mot_data_rgt: object = None
    mot_data_lft: object = None
    end_jnt_values: Optional[np.ndarray] = None
    end_jnt_values_rgt: Optional[np.ndarray] = None
    end_jnt_values_lft: Optional[np.ndarray] = None
    error_msg: str = ""


class MotionPrimitive(ABC):
    """Abstract base class for motion primitives.

    Subclasses implement ``plan()`` which takes pick/place parameters
    and returns a ``PrimitiveResult``.
    """

    @abstractmethod
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
        """Plan a motion primitive.

        Parameters
        ----------
        obj_cmodel
            Collision model of the object at its pick (staging) pose.
        grasp_collection
            Pre-computed grasps for this object.
        goal_pose_list
            Target poses ``[(pos, rotmat), ...]`` for placement.
        start_jnt_values
            Starting joint configuration.  ``None`` = current config.
        end_jnt_values
            Desired ending joint configuration.  ``None`` = current config.
        obstacle_list
            Collision models to avoid during planning.
        approach_distance
            Approach distance before grasp/place (meters).
        depart_distance
            Depart distance after grasp/place (meters).
        use_rrt
            Use RRT for transit motion planning.

        Returns
        -------
        PrimitiveResult
            Planning result containing motion data and status.
        """
        ...
