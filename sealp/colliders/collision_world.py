"""
SEALP Collision World
======================
Unified collision manager that combines:
- StaticEnvironment (config-defined obstacles)
- User-defined ObstacleManager (runtime-added STL/box obstacles)

Adapted from ``wrs_tbm/tbm_interface/colliders/tbm_collision_world.py``.

Usage::

    from sealp.colliders import CollisionWorld

    world = CollisionWorld(obstacle_defs=[...])
    world.user_obstacles.add_box("block", extent=[0.1, 0.1, 0.1],
                                 pos=[0.5, 0, 0.05])

    robot.is_collided(obstacle_list=world.obstacle_list)
    world.show(base, robot=my_robot, toggle_cdprim=True)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import wrs.modeling.model_collection as mmc

from .obstacle_manager import ObstacleManager
from .static_environment import StaticEnvironment


class CollisionWorld:
    """Unified collision world: static environment + user-defined obstacles.

    Parameters
    ----------
    obstacle_defs : list of dict
        Obstacle definitions for the static environment (from config).
    base_dir : str or None
        Base directory for resolving relative file paths in obstacle defs.
    """

    def __init__(self, obstacle_defs: Optional[List[dict]] = None,
                 base_dir: Optional[str] = None):
        self.env = StaticEnvironment(
            obstacle_defs=obstacle_defs or [],
            base_dir=base_dir,
        )
        self.user_obstacles = ObstacleManager()

    @property
    def obstacle_list(self) -> list:
        """Combined obstacle list (environment + user-defined)."""
        return self.env.obstacle_list + self.user_obstacles.obstacle_list

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def show(self, base, robot=None, toggle_cdprim=False, alpha=0.5):
        """Visualize all obstacles and optionally the robot.

        Parameters
        ----------
        base : ShowBase / World
            Panda3D scene.
        robot : robot instance or None
            If provided, also draw the robot mesh model.
        toggle_cdprim : bool
            If True, show collision primitives.
        alpha : float
            Transparency for obstacle visualization.

        Returns
        -------
        mmc.ModelCollection
        """
        meshmodel = mmc.ModelCollection(name="sealp_collision_world_vis")

        # Static environment obstacles
        for name in self.env.names():
            model = self.env.get(name)
            if toggle_cdprim:
                model.show_cdprimit()
            copy = model.copy()
            rgba = copy.rgba
            if rgba is not None and len(rgba) >= 4:
                copy.rgba = np.array([rgba[0], rgba[1], rgba[2], alpha])
            copy.attach_to(meshmodel)

        # User-defined obstacles
        for model in self.user_obstacles.obstacle_list:
            if toggle_cdprim:
                model.show_cdprimit()
            copy = model.copy()
            rgba = copy.rgba
            if rgba is not None and len(rgba) >= 4:
                copy.rgba = np.array([rgba[0], rgba[1], rgba[2], alpha])
            copy.attach_to(meshmodel)

        meshmodel.attach_to(base)

        if robot is not None:
            rbt_mesh = robot.gen_meshmodel()
            if toggle_cdprim:
                rbt_mesh.show_cdprimit()
            rbt_mesh.attach_to(base)

        return meshmodel
