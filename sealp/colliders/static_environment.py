"""
SEALP Static Environment
=========================
Builds static collision obstacles from a list of obstacle definitions
(typically loaded from a SEALP config YAML).

Adapted from ``wrs_tbm/tbm_interface/colliders/tbm_environment.py``.

Usage::

    from sealp.colliders import StaticEnvironment

    obstacle_defs = [
        {"name": "table", "type": "box", "extent": [0.8, 1.2, 0.02],
         "pos": [0.4, 0, 0], "rgba": [0.6, 0.5, 0.4, 0.8]},
    ]
    env = StaticEnvironment(obstacle_defs)
    obstacle_list = env.obstacle_list
    env.show(base)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc

from .obstacle_manager import ObstacleManager


class StaticEnvironment:
    """Manages static environment collision models built from config defs.

    Parameters
    ----------
    obstacle_defs : list of dict
        Each dict defines one obstacle with keys:
        ``name``, ``type`` (``"box"`` or ``"stl"``), and type-specific
        fields (``extent`` for box, ``file`` for stl), plus optional
        ``pos``, ``rotmat``, ``rgba``, ``cdprimit_type``.
    base_dir : str or None
        Base directory for resolving relative ``file`` paths in STL
        obstacles.  Defaults to the current working directory.
    """

    def __init__(self, obstacle_defs: List[dict],
                 base_dir: Optional[str] = None):
        self._base_dir = base_dir or os.getcwd()
        self._manager = ObstacleManager()
        self._build(obstacle_defs)

    def _build(self, obstacle_defs: List[dict]):
        """Create CollisionModels for each obstacle definition."""
        for cfg in obstacle_defs:
            name = cfg["name"]
            obs_type = cfg["type"]

            if obs_type == "stl":
                filepath = cfg["file"]
                if not os.path.isabs(filepath):
                    filepath = os.path.join(self._base_dir, filepath)
                filepath = os.path.normpath(filepath)
                if not os.path.isfile(filepath):
                    raise FileNotFoundError(
                        f"STL file not found for obstacle '{name}': "
                        f"{filepath}"
                    )
                cdprimit = cfg.get("cdprimit_type", "box")
                model = mcm.CollisionModel(filepath,
                                           cdprimit_type=cdprimit)
            elif obs_type == "box":
                extent = np.asarray(cfg["extent"], dtype=float)
                model = mcm.gen_box(extent)
            else:
                raise ValueError(
                    f"Unknown obstacle type '{obs_type}' for '{name}'. "
                    f"Supported types: 'box', 'stl'."
                )

            pos = cfg.get("pos")
            if pos is not None:
                model.pos = np.asarray(pos, dtype=float)

            rotmat = cfg.get("rotmat")
            if rotmat is not None:
                model.rotmat = np.asarray(rotmat, dtype=float)

            rgba = cfg.get("rgba")
            if rgba is not None:
                model.rgba = np.asarray(rgba, dtype=float)

            self._manager.add(name, model)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def obstacle_list(self) -> list:
        """Return list of CollisionModel objects for is_collided()."""
        return self._manager.obstacle_list

    @property
    def manager(self) -> ObstacleManager:
        """Return the underlying ObstacleManager."""
        return self._manager

    def get(self, name: str) -> mcm.CollisionModel:
        """Get a specific obstacle by name."""
        return self._manager.get(name)

    def names(self) -> list:
        """Return all obstacle names."""
        return self._manager.names()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def show(self, base, robot=None, toggle_cdprim=False, alpha=0.5):
        """Visualize environment obstacles (and optionally the robot).

        Parameters
        ----------
        base : ShowBase / World
            Panda3D scene to attach models to.
        robot : robot instance or None
            If provided, also draw the robot mesh model.
        toggle_cdprim : bool
            If True, show collision primitives on the obstacles.
        alpha : float
            Transparency for obstacle visualization.

        Returns
        -------
        mmc.ModelCollection
        """
        meshmodel = mmc.ModelCollection(name="sealp_env_vis")

        for name in self.names():
            model = self.get(name)
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
