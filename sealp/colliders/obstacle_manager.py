"""
Obstacle Manager for SEALP Collision Detection
================================================
Manages a named dictionary of CollisionModel obstacles that can be
passed directly to ``robot.is_collided(obstacle_list=...)``.

Adapted from ``wrs_tbm/tbm_interface/colliders/collision_config.py``.

Usage::

    from sealp.colliders import ObstacleManager

    obstacles = ObstacleManager()
    obstacles.add_box("table", extent=[0.8, 1.2, 0.02], pos=[0.4, 0, 0])
    obstacles.add_stl("fixture", "meshes/fixture.stl", pos=[0.3, 0, 0.02])

    robot.is_collided(obstacle_list=obstacles.obstacle_list)
"""

import numpy as np
import wrs.modeling.collision_model as mcm


class ObstacleManager:
    """A simple manager for collision obstacles.

    Maintains a named dict of ``CollisionModel`` objects that can be
    passed directly to ``is_collided(obstacle_list=...)``.
    """

    def __init__(self):
        self._obstacles: dict[str, mcm.CollisionModel] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def obstacle_list(self) -> list:
        """Return the list of CollisionModel objects for is_collided()."""
        return list(self._obstacles.values())

    def __len__(self):
        return len(self._obstacles)

    def __repr__(self):
        return f"ObstacleManager({list(self._obstacles.keys())})"

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------
    def add(self, name: str, collision_model: mcm.CollisionModel,
            show_base=None):
        """Add an existing CollisionModel.

        Parameters
        ----------
        name : str
            Unique name for this obstacle.
        collision_model : mcm.CollisionModel
            A collision model instance (pos/rotmat already set).
        show_base : optional
            If provided, attach to this scene for visualization.

        Returns
        -------
        mcm.CollisionModel
        """
        if name in self._obstacles:
            raise ValueError(
                f"Obstacle '{name}' already exists. "
                f"Remove it first or use a different name."
            )
        self._obstacles[name] = collision_model
        if show_base is not None:
            collision_model.attach_to(show_base)
        return collision_model

    def add_stl(self, name: str, stl_path: str,
                pos=None, rotmat=None, rgba=None,
                cdprimit_type="box",
                show_base=None):
        """Load an STL file and add it as an obstacle.

        Parameters
        ----------
        name : str
            Unique name.
        stl_path : str
            Path to .stl file.
        pos : array-like, optional
            [x, y, z] in meters.
        rotmat : array-like, optional
            3x3 rotation matrix.
        rgba : array-like, optional
            [r, g, b, a] color.
        cdprimit_type : str
            Collision primitive type (``"box"``, ``"polygons"``, etc.).
        show_base : optional
            If provided, attach to scene for visualization.

        Returns
        -------
        mcm.CollisionModel
        """
        model = mcm.CollisionModel(stl_path, cdprimit_type=cdprimit_type)
        if pos is not None:
            model.pos = np.asarray(pos, dtype=float)
        if rotmat is not None:
            model.rotmat = np.asarray(rotmat, dtype=float)
        if rgba is not None:
            model.rgba = rgba
        return self.add(name, model, show_base=show_base)

    def add_box(self, name: str, extent,
                pos=None, rotmat=None, rgba=None,
                show_base=None):
        """Create a box obstacle and add it.

        Parameters
        ----------
        name : str
            Unique name.
        extent : array-like
            [width, depth, height] in meters.
        pos, rotmat, rgba, show_base
            See :meth:`add_stl`.

        Returns
        -------
        mcm.CollisionModel
        """
        model = mcm.gen_box(np.asarray(extent, dtype=float))
        if pos is not None:
            model.pos = np.asarray(pos, dtype=float)
        if rotmat is not None:
            model.rotmat = np.asarray(rotmat, dtype=float)
        if rgba is not None:
            model.rgba = rgba
        return self.add(name, model, show_base=show_base)

    # ------------------------------------------------------------------
    # Remove / query
    # ------------------------------------------------------------------
    def remove(self, name: str):
        """Remove an obstacle by name."""
        if name not in self._obstacles:
            raise KeyError(f"Obstacle '{name}' not found.")
        model = self._obstacles.pop(name)
        model.detach()
        return model

    def clear(self):
        """Remove all obstacles."""
        for model in self._obstacles.values():
            model.detach()
        self._obstacles.clear()

    def get(self, name: str) -> mcm.CollisionModel:
        """Get an obstacle by name."""
        return self._obstacles[name]

    def names(self) -> list:
        """Return all obstacle names."""
        return list(self._obstacles.keys())
