"""
Assembly Step Definition
========================

Data class representing a single step (action) in an assembly sequence.
Each step references the part being assembled, the parent part it
attaches to, the mating transform, execution constraints, and
dependency information.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class AssemblyStep:
    """A single action in the assembly sequence.

    Attributes
    ----------
    step_id : int
        Sequential index of this step (0-based).
    part_id : str
        Identifier of the part being assembled in this step.
    parent_part_id : str
        Identifier of the part this one attaches to.
        Use ``"base"`` or ``"fixture"`` for the workspace surface.
    assembly_pos : np.ndarray
        Position of the assembled part ``[x, y, z]``.
    assembly_rotmat : np.ndarray
        Orientation of the assembled part (3×3 rotation matrix).
    dependencies : list of int
        Step IDs that must be completed before this step can start.
    primitive_type : str
        Motion primitive type.  One of:
        ``"single_arm_transport"`` — one arm picks, transports, places.
        ``"dual_arm_cooperative"`` — both arms cooperate (large parts).
        ``"manual"`` — human intervention required.
    grasp_id : int or None
        Optional index into a pre-computed grasp collection for this
        step's part.  ``None`` means the planner should auto-select.
    notes : str
        Free-text notes for this step.
    metadata : dict
        Arbitrary key-value metadata.
    """

    step_id: int
    part_id: str
    parent_part_id: str = "base"
    assembly_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    assembly_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    dependencies: List[int] = field(default_factory=list)
    primitive_type: str = "single_arm_transport"
    grasp_id: Optional[int] = None
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for YAML serialisation."""
        return {
            "step_id": self.step_id,
            "part_id": self.part_id,
            "parent_part_id": self.parent_part_id,
            "assembly_pos": self.assembly_pos.tolist(),
            "assembly_rotmat": self.assembly_rotmat.tolist(),
            "dependencies": list(self.dependencies),
            "primitive_type": self.primitive_type,
            "grasp_id": self.grasp_id,
            "notes": self.notes,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblyStep":
        """Create an ``AssemblyStep`` from a dict."""
        return cls(
            step_id=int(d["step_id"]),
            part_id=d["part_id"],
            parent_part_id=d.get("parent_part_id", "base"),
            assembly_pos=np.asarray(
                d.get("assembly_pos", [0, 0, 0]), dtype=float
            ),
            assembly_rotmat=np.asarray(
                d.get("assembly_rotmat", np.eye(3).tolist()), dtype=float
            ),
            dependencies=list(d.get("dependencies", [])),
            primitive_type=d.get("primitive_type", "single_arm_transport"),
            grasp_id=d.get("grasp_id", None),
            notes=d.get("notes", ""),
            metadata=d.get("metadata", {}),
        )

    @property
    def assembly_pose(self) -> tuple:
        """Return ``(assembly_pos, assembly_rotmat)``."""
        return self.assembly_pos, self.assembly_rotmat

    def copy(self) -> "AssemblyStep":
        """Return a deep copy."""
        return AssemblyStep.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return (f"AssemblyStep(id={self.step_id}, part={self.part_id!r}, "
                f"parent={self.parent_part_id!r}, "
                f"deps={self.dependencies})")
