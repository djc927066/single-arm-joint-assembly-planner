"""
Assembly Sequence Generator
============================

Builder-pattern utility for constructing ``AssemblySequence`` objects
programmatically.  Provides a fluent API and convenience helpers such
as automatic linear-sequence generation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .assembly_part import AssemblyPart
from .assembly_step import AssemblyStep
from .assembly_sequence import AssemblySequence


class SequenceGenerator:
    """Fluent builder for ``AssemblySequence`` objects.

    Usage
    -----
    >>> seq = (SequenceGenerator("MyAssembly")
    ...        .add_part("base", "Base Plate", "base.stl",
    ...                  assembly_pos=[0, 0, 0])
    ...        .add_part("leg1", "Left Leg", "leg.stl",
    ...                  assembly_pos=[0.1, 0, 0])
    ...        .add_step(0, "base", parent="fixture")
    ...        .add_step(1, "leg1", parent="base", deps=[0])
    ...        .build())
    """

    def __init__(self, name: str = "unnamed_assembly",
                 description: str = ""):
        self._name = name
        self._description = description
        self._parts: List[AssemblyPart] = []
        self._steps: List[AssemblyStep] = []

    # ------------------------------------------------------------------
    # Part helpers
    # ------------------------------------------------------------------
    def add_part(self,
                 part_id: str,
                 name: str,
                 model_path: str,
                 init_pos=None,
                 init_rotmat=None,
                 assembly_pos=None,
                 assembly_rotmat=None,
                 mass: float = 0.0,
                 color_rgba=None,
                 **metadata) -> "SequenceGenerator":
        """Add a part to the assembly.

        All positional / orientation arguments accept list-like inputs
        and will be converted to ``np.ndarray`` internally.

        Returns ``self`` for chaining.
        """
        self._parts.append(AssemblyPart(
            part_id=part_id,
            name=name,
            model_path=model_path,
            init_pos=(np.asarray(init_pos, dtype=float)
                      if init_pos is not None else np.zeros(3)),
            init_rotmat=(np.asarray(init_rotmat, dtype=float)
                         if init_rotmat is not None else np.eye(3)),
            assembly_pos=(np.asarray(assembly_pos, dtype=float)
                          if assembly_pos is not None else np.zeros(3)),
            assembly_rotmat=(np.asarray(assembly_rotmat, dtype=float)
                             if assembly_rotmat is not None else np.eye(3)),
            mass=mass,
            color_rgba=(np.asarray(color_rgba, dtype=float)
                        if color_rgba is not None
                        else np.array([0.7, 0.7, 0.7, 1.0])),
            metadata=metadata,
        ))
        return self

    def add_part_from_model(self,
                            model_path: str,
                            part_id: Optional[str] = None,
                            name: Optional[str] = None,
                            **kwargs) -> "SequenceGenerator":
        """Add a part, deriving ``part_id`` and ``name`` from the file.

        If *part_id* or *name* are not given they are derived from the
        model filename (stem).
        """
        import os
        stem = os.path.splitext(os.path.basename(model_path))[0]
        if part_id is None:
            part_id = stem
        if name is None:
            name = stem.replace("_", " ").title()
        return self.add_part(part_id=part_id, name=name,
                             model_path=model_path, **kwargs)

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------
    def add_step(self,
                 step_id: int,
                 part_id: str,
                 parent: str = "base",
                 assembly_pos=None,
                 assembly_rotmat=None,
                 deps: Optional[List[int]] = None,
                 primitive_type: str = "single_arm_transport",
                 grasp_id: Optional[int] = None,
                 notes: str = "") -> "SequenceGenerator":
        """Add an assembly step.  Returns ``self`` for chaining."""
        self._steps.append(AssemblyStep(
            step_id=step_id,
            part_id=part_id,
            parent_part_id=parent,
            assembly_pos=(np.asarray(assembly_pos, dtype=float)
                          if assembly_pos is not None else np.zeros(3)),
            assembly_rotmat=(np.asarray(assembly_rotmat, dtype=float)
                             if assembly_rotmat is not None else np.eye(3)),
            dependencies=deps or [],
            primitive_type=primitive_type,
            grasp_id=grasp_id,
            notes=notes,
        ))
        return self

    # ------------------------------------------------------------------
    # Automatic generators
    # ------------------------------------------------------------------
    def auto_generate_linear_sequence(
            self,
            parent_id: str = "base",
            primitive_type: str = "single_arm_transport",
    ) -> "SequenceGenerator":
        """Generate a simple linear sequence from the registered parts.

        Each part is assembled in the order it was added.  Each step
        depends on the previous one.  Assembly poses default to the
        ``assembly_pos`` / ``assembly_rotmat`` stored on the part.

        Returns ``self`` for chaining.
        """
        self._steps.clear()
        for idx, part in enumerate(self._parts):
            deps = [idx - 1] if idx > 0 else []
            self._steps.append(AssemblyStep(
                step_id=idx,
                part_id=part.part_id,
                parent_part_id=parent_id if idx == 0
                else self._parts[idx - 1].part_id,
                assembly_pos=part.assembly_pos.copy(),
                assembly_rotmat=part.assembly_rotmat.copy(),
                dependencies=deps,
                primitive_type=primitive_type,
            ))
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, validate: bool = True) -> AssemblySequence:
        """Construct the ``AssemblySequence``.

        Parameters
        ----------
        validate : bool
            Run :meth:`AssemblySequence.validate` after construction.

        Returns
        -------
        AssemblySequence
        """
        seq = AssemblySequence(name=self._name,
                               description=self._description)
        for part in self._parts:
            seq.add_part(part)
        for step in self._steps:
            seq.add_step(step)
        if validate:
            seq.validate(strict=True)
        return seq
