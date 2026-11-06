"""
Assembly Part Definition
========================

Data class representing a single part in an assembly.
Each part carries its 3D model reference, initial and
target assembly poses, and optional metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AssemblyPart:
    """A single physical part used in an assembly.

    Attributes
    ----------
    part_id : str
        Unique identifier for this part (e.g. ``"leg_01"``).
    name : str
        Human-readable display name.
    model_path : str
        Filesystem path to the 3D model file (STL / OBJ / DAE).
    init_pos : np.ndarray
        Initial position in world coordinates ``[x, y, z]``.
    init_rotmat : np.ndarray
        Initial orientation as a 3×3 rotation matrix.
    assembly_pos : np.ndarray
        Target assembly position in world coordinates.
    assembly_rotmat : np.ndarray
        Target assembly orientation as a 3×3 rotation matrix.
    mass : float
        Part mass in kg (default 0.0).
    color_rgba : np.ndarray
        Visualisation colour ``[r, g, b, a]`` (default light grey).
    metadata : dict
        Arbitrary key-value metadata (material, supplier, etc.).
    """

    part_id: str
    name: str
    model_path: str
    init_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    assembly_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    assembly_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    mass: float = 0.0
    color_rgba: np.ndarray = field(
        default_factory=lambda: np.array([0.7, 0.7, 0.7, 1.0])
    )
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers (dict ↔ object)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for YAML serialisation."""
        return {
            "part_id": self.part_id,
            "name": self.name,
            "model_path": self.model_path,
            "init_pos": self.init_pos.tolist(),
            "init_rotmat": self.init_rotmat.tolist(),
            "assembly_pos": self.assembly_pos.tolist(),
            "assembly_rotmat": self.assembly_rotmat.tolist(),
            "mass": self.mass,
            "color_rgba": self.color_rgba.tolist(),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblyPart":
        """Create an ``AssemblyPart`` from a dict (e.g. loaded from YAML)."""
        return cls(
            part_id=d["part_id"],
            name=d["name"],
            model_path=d["model_path"],
            init_pos=np.asarray(d.get("init_pos", [0, 0, 0]), dtype=float),
            init_rotmat=np.asarray(
                d.get("init_rotmat", np.eye(3).tolist()), dtype=float
            ),
            assembly_pos=np.asarray(
                d.get("assembly_pos", [0, 0, 0]), dtype=float
            ),
            assembly_rotmat=np.asarray(
                d.get("assembly_rotmat", np.eye(3).tolist()), dtype=float
            ),
            mass=float(d.get("mass", 0.0)),
            color_rgba=np.asarray(
                d.get("color_rgba", [0.7, 0.7, 0.7, 1.0]), dtype=float
            ),
            metadata=d.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def init_pose(self) -> tuple:
        """Return ``(init_pos, init_rotmat)``."""
        return self.init_pos, self.init_rotmat

    @property
    def assembly_pose(self) -> tuple:
        """Return ``(assembly_pos, assembly_rotmat)``."""
        return self.assembly_pos, self.assembly_rotmat

    def copy(self) -> "AssemblyPart":
        """Return a deep copy of this part."""
        return AssemblyPart.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return (f"AssemblyPart(id={self.part_id!r}, name={self.name!r}, "
                f"model={self.model_path!r})")
