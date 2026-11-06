"""
Assembly Sequence Container
============================

The ``AssemblySequence`` class aggregates parts and steps into a
complete, validated assembly plan.  It supports topological ordering,
cycle detection, YAML serialisation, and optional Panda3D visualization.
"""

from __future__ import annotations

import os

from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from .assembly_part import AssemblyPart
from .assembly_step import AssemblyStep


class AssemblySequence:
    """A complete assembly specification: parts + ordered steps.

    Parameters
    ----------
    name : str
        Human-readable name for the assembly (e.g. ``"IKEA_Shelf"``).
    description : str
        Optional long-form description.
    """

    def __init__(self, name: str = "unnamed_assembly",
                 description: str = ""):
        self.name: str = name
        self.description: str = description
        self._parts: Dict[str, AssemblyPart] = {}
        self._steps: List[AssemblyStep] = []

    # ------------------------------------------------------------------
    # Part management
    # ------------------------------------------------------------------
    def add_part(self, part: AssemblyPart) -> None:
        """Register a part.  Raises ``ValueError`` on duplicate ID."""
        if part.part_id in self._parts:
            raise ValueError(
                f"Duplicate part_id: {part.part_id!r} already exists."
            )
        self._parts[part.part_id] = part

    def get_part(self, part_id: str) -> AssemblyPart:
        """Look up a part by ID.  Raises ``KeyError`` if missing."""
        return self._parts[part_id]

    @property
    def parts(self) -> List[AssemblyPart]:
        """Return all parts in insertion order."""
        return list(self._parts.values())

    @property
    def part_ids(self) -> List[str]:
        """Return all registered part IDs."""
        return list(self._parts.keys())

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------
    def add_step(self, step: AssemblyStep) -> None:
        """Append a step.  Raises ``ValueError`` on duplicate step ID."""
        existing_ids = {s.step_id for s in self._steps}
        if step.step_id in existing_ids:
            raise ValueError(
                f"Duplicate step_id: {step.step_id} already exists."
            )
        self._steps.append(step)

    @property
    def steps(self) -> List[AssemblyStep]:
        """Return all steps in insertion order."""
        return list(self._steps)

    @property
    def n_parts(self) -> int:
        return len(self._parts)

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, strict: bool = True) -> List[str]:
        """Check the assembly for consistency.

        Returns a list of warning/error strings.  When *strict* is
        ``True`` a ``ValueError`` is raised on the first error.

        Checks performed
        ~~~~~~~~~~~~~~~~
        1. Every step references a known ``part_id``.
        2. Every dependency step_id exists.
        3. The dependency graph is acyclic (DAG).
        4. No ``part_id`` appears in more than one step.
        5. Every part's ``model_path`` points to an existing file.
        """
        errors: List[str] = []
        step_ids = {s.step_id for s in self._steps}

        # Check 1 – part references
        for step in self._steps:
            if step.part_id not in self._parts:
                errors.append(
                    f"Step {step.step_id}: part_id {step.part_id!r} "
                    f"not found in registered parts."
                )

        # Check 2 – dependency references
        for step in self._steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(
                        f"Step {step.step_id}: dependency {dep} "
                        f"does not correspond to any step."
                    )

        # Check 3 – acyclic
        if not self._is_dag():
            errors.append("Dependency graph contains a cycle!")

        # Check 4 – unique part per step
        seen_parts: dict = {}
        for step in self._steps:
            if step.part_id in seen_parts:
                errors.append(
                    f"Part {step.part_id!r} appears in both step "
                    f"{seen_parts[step.part_id]} and step {step.step_id}."
                )
            seen_parts[step.part_id] = step.step_id

        # Check 5 – model_path file existence
        for part in self._parts.values():
            if part.model_path and not os.path.isfile(part.model_path):
                errors.append(
                    f"Part {part.part_id!r}: model_path "
                    f"{part.model_path!r} does not exist on disk."
                )

        if strict and errors:
            raise ValueError(
                "Assembly validation failed:\n  • "
                + "\n  • ".join(errors)
            )
        return errors

    def _is_dag(self) -> bool:
        """Return ``True`` if the step dependency graph is a DAG."""
        graph: Dict[int, List[int]] = defaultdict(list)
        in_degree: Dict[int, int] = {}
        for step in self._steps:
            in_degree.setdefault(step.step_id, 0)
            for dep in step.dependencies:
                graph[dep].append(step.step_id)
                in_degree[step.step_id] = in_degree.get(step.step_id, 0) + 1
                in_degree.setdefault(dep, 0)
        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in graph[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return visited == len(in_degree)

    # ------------------------------------------------------------------
    # Topological ordering
    # ------------------------------------------------------------------
    def get_execution_order(self) -> List[AssemblyStep]:
        """Return steps in a valid topological execution order.

        Raises ``ValueError`` if the dependency graph contains a cycle.
        """
        graph: Dict[int, List[int]] = defaultdict(list)
        in_degree: Dict[int, int] = {}
        step_map = {s.step_id: s for s in self._steps}
        for step in self._steps:
            in_degree.setdefault(step.step_id, 0)
            for dep in step.dependencies:
                graph[dep].append(step.step_id)
                in_degree[step.step_id] = in_degree.get(step.step_id, 0) + 1
                in_degree.setdefault(dep, 0)
        queue = deque(
            sorted(sid for sid, deg in in_degree.items() if deg == 0)
        )
        order: List[AssemblyStep] = []
        while queue:
            node = queue.popleft()
            if node in step_map:
                order.append(step_map[node])
            children = sorted(graph[node])
            for child in children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._steps):
            raise ValueError(
                "Cannot compute execution order — dependency cycle detected."
            )
        return order

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Convert the entire assembly to a nested dict."""
        return {
            "name": self.name,
            "description": self.description,
            "parts": [p.to_dict() for p in self._parts.values()],
            "steps": [s.to_dict() for s in self._steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblySequence":
        """Reconstruct an ``AssemblySequence`` from a dict."""
        seq = cls(name=d.get("name", "unnamed"),
                  description=d.get("description", ""))
        for pd in d.get("parts", []):
            seq.add_part(AssemblyPart.from_dict(pd))
        for sd in d.get("steps", []):
            seq.add_step(AssemblyStep.from_dict(sd))
        return seq

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a multi-line human-readable summary."""
        lines = [
            f"Assembly: {self.name}",
            f"Description: {self.description}",
            f"Parts ({self.n_parts}):",
        ]
        for p in self._parts.values():
            lines.append(f"  • {p.part_id}: {p.name}  [{p.model_path}]")
        lines.append(f"Steps ({self.n_steps}):")
        for s in self._steps:
            dep_str = (
                ", ".join(str(d) for d in s.dependencies)
                if s.dependencies else "none"
            )
            lines.append(
                f"  {s.step_id}. Assemble {s.part_id!r} onto "
                f"{s.parent_part_id!r}  (deps: {dep_str}, "
                f"type: {s.primitive_type})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"AssemblySequence(name={self.name!r}, "
                f"parts={self.n_parts}, steps={self.n_steps})")

    def __len__(self) -> int:
        return self.n_steps
