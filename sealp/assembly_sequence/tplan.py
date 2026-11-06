"""
Task Plan Format (.tplan)
==========================

Task-specific execution plan for an assembly.  YAML-backed, custom
``.tplan`` extension.  References an ``.asmdef`` file and contains all
data that varies per task execution:

- Fixture (assembly station) world pose
- Staging positions for each part (where the robot picks them)
- Per-step execution parameters (primitive type, grasp ID, approach/depart)
- Robot configuration

A ``.tplan`` is typically produced by the layout optimizer or by hand.

Usage::

    from sealp.assembly_sequence import TaskPlan

    plan = TaskPlan.load("chair_plan.tplan")
    plan.assembly            # linked AssemblyDef
    plan.fixture_pos         # fixture world position
    plan.staging             # dict[part_id → (pos, rotmat)]
    plan.step_params         # dict[step_id → StepParams]

    plan.save("output.tplan")
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from .asmdef import AssemblyDef, _float_representer
from .primitives import Primitive

yaml.add_representer(float, _float_representer)

FORMAT_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════
@dataclass
class StagingPose:
    """Pose of a part in the staging area (where the robot picks it).

    Attributes
    ----------
    part_id : str
        Which part this staging position is for.
    pos : np.ndarray
        World position ``[x, y, z]``.
    rotmat : np.ndarray
        World orientation (3×3 rotation matrix).
    """
    part_id: str
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))

    def to_dict(self) -> dict:
        return {
            "pos": [float(v) for v in self.pos],
            "rotmat": [[float(v) for v in row] for row in self.rotmat],
        }

    @classmethod
    def from_dict(cls, part_id: str, d: dict) -> "StagingPose":
        return cls(
            part_id=part_id,
            pos=np.asarray(d.get("pos", [0, 0, 0]), dtype=float),
            rotmat=np.asarray(
                d.get("rotmat", np.eye(3).tolist()), dtype=float),
        )


@dataclass
class StepParams:
    """Per-step execution parameters.

    Attributes
    ----------
    step_id : int
        Which assembly step this applies to.
    primitive : Primitive
        Motion primitive type (enum).
    grasp_id : int or None
        Index into a grasp collection.  ``None`` = auto-select.
    approach_distance : float
        Approach distance before placement (meters).
    depart_distance : float
        Depart distance after placement (meters).
    approach_direction : np.ndarray or None
        Approach direction vector.  ``None`` = use default (z-axis).
    speed_factor : float
        Speed multiplier (1.0 = normal).
    metadata : dict
        Arbitrary extra parameters.
    """
    step_id: int
    primitive: Primitive = Primitive.SINGLE_ARM_TRANSPORT
    grasp_id: Optional[int] = None
    approach_distance: float = 0.02
    depart_distance: float = 0.02
    approach_direction: Optional[np.ndarray] = None
    speed_factor: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Coerce string to Primitive enum
        if isinstance(self.primitive, str):
            self.primitive = Primitive.from_str(self.primitive)

    def to_dict(self) -> dict:
        d: dict = {
            "step": self.step_id,
            "primitive": self.primitive.value,
        }
        if self.grasp_id is not None:
            d["grasp_id"] = self.grasp_id
        if self.approach_distance != 0.02:
            d["approach_distance"] = self.approach_distance
        if self.depart_distance != 0.02:
            d["depart_distance"] = self.depart_distance
        if self.approach_direction is not None:
            d["approach_direction"] = [float(v)
                                       for v in self.approach_direction]
        if self.speed_factor != 1.0:
            d["speed_factor"] = self.speed_factor
        if self.metadata:
            d["metadata"] = copy.deepcopy(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StepParams":
        approach_dir = None
        if "approach_direction" in d:
            approach_dir = np.asarray(d["approach_direction"], dtype=float)
        return cls(
            step_id=int(d["step"]),
            primitive=Primitive.from_str(
                d.get("primitive", "single_arm_transport")),
            grasp_id=d.get("grasp_id"),
            approach_distance=float(d.get("approach_distance", 0.02)),
            depart_distance=float(d.get("depart_distance", 0.02)),
            approach_direction=approach_dir,
            speed_factor=float(d.get("speed_factor", 1.0)),
            metadata=d.get("metadata", {}),
        )


@dataclass
class RobotConfig:
    """Robot configuration for the task.

    Attributes
    ----------
    robot_type : str
        Robot type identifier (e.g. ``"piper"``, ``"cobotta"``).
    base_pos : np.ndarray
        Robot base world position ``[x, y, z]``.
    base_rotmat : np.ndarray
        Robot base world orientation (3×3).
    start_conf : np.ndarray or None
        Initial joint configuration.  ``None`` = use default home.
    metadata : dict
        Extra robot config (gripper type, tool params, etc.).
    """
    robot_type: str = "piper"
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    start_conf: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "robot_type": self.robot_type,
            "base_pos": [float(v) for v in self.base_pos],
            "base_rotmat": [[float(v) for v in row]
                            for row in self.base_rotmat],
        }
        if self.start_conf is not None:
            d["start_conf"] = [float(v) for v in self.start_conf]
        if self.metadata:
            d["metadata"] = copy.deepcopy(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RobotConfig":
        start_conf = None
        if "start_conf" in d:
            start_conf = np.asarray(d["start_conf"], dtype=float)
        return cls(
            robot_type=d.get("robot_type", "piper"),
            base_pos=np.asarray(d.get("base_pos", [0, 0, 0]), dtype=float),
            base_rotmat=np.asarray(
                d.get("base_rotmat", np.eye(3).tolist()), dtype=float),
            start_conf=start_conf,
            metadata=d.get("metadata", {}),
        )


# ══════════════════════════════════════════════════════════════
#  TaskPlan — top-level container
# ══════════════════════════════════════════════════════════════
class TaskPlan:
    """Task-specific execution plan for an assembly.

    Parameters
    ----------
    assembly_file : str
        Path to the referenced ``.asmdef`` file.
    name : str
        Task plan name / identifier.
    description : str
        Long-form description.
    """

    def __init__(self, assembly_file: str = "",
                 name: str = "unnamed",
                 description: str = ""):
        self.format_version: str = FORMAT_VERSION
        self.name: str = name
        self.description: str = description
        self.assembly_file: str = assembly_file

        # Fixture (assembly station) world pose
        self.fixture_pos: np.ndarray = np.zeros(3)
        self.fixture_rotmat: np.ndarray = np.eye(3)

        # Staging positions
        self._staging: Dict[str, StagingPose] = {}

        # Per-step parameters
        self._step_params: Dict[int, StepParams] = {}

        # Robot config
        self.robot: RobotConfig = RobotConfig()

        # Linked assembly (loaded lazily)
        self._assembly: Optional[AssemblyDef] = None

    # ── Staging management ───────────────────────────────────
    def set_staging(self, part_id: str, pos: np.ndarray,
                    rotmat: np.ndarray = None):
        """Set the staging (pick) position for a part."""
        if rotmat is None:
            rotmat = np.eye(3)
        self._staging[part_id] = StagingPose(
            part_id=part_id, pos=pos.copy(), rotmat=rotmat.copy())

    def get_staging(self, part_id: str) -> Optional[StagingPose]:
        return self._staging.get(part_id)

    @property
    def staging(self) -> Dict[str, StagingPose]:
        return dict(self._staging)

    # ── Step params management ───────────────────────────────
    def set_step_params(self, params: StepParams):
        """Set execution parameters for a step."""
        self._step_params[params.step_id] = params

    def get_step_params(self, step_id: int) -> Optional[StepParams]:
        return self._step_params.get(step_id)

    @property
    def step_params(self) -> Dict[int, StepParams]:
        return dict(self._step_params)

    # ── Assembly link ────────────────────────────────────────
    @property
    def assembly(self) -> Optional[AssemblyDef]:
        """Return the linked AssemblyDef (loaded lazily)."""
        if self._assembly is None and self.assembly_file:
            if os.path.isfile(self.assembly_file):
                self._assembly = AssemblyDef.load(self.assembly_file)
        return self._assembly

    def set_assembly(self, asm: AssemblyDef):
        """Explicitly link an AssemblyDef object."""
        self._assembly = asm

    # ── Computed properties ──────────────────────────────────
    def compute_assembly_world_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute absolute assembly poses using fixture position."""
        asm = self.assembly
        if asm is None:
            raise ValueError("No assembly linked. Set assembly_file or call set_assembly().")
        return asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

    # ── Serialisation ────────────────────────────────────────
    def to_dict(self) -> dict:
        d: dict = {
            "format_version": self.format_version,
            "name": self.name,
            "description": self.description,
            "assembly_file": self.assembly_file,
            "fixture": {
                "pos": [float(v) for v in self.fixture_pos],
                "rotmat": [[float(v) for v in row]
                           for row in self.fixture_rotmat],
            },
        }
        # Robot
        d["robot"] = self.robot.to_dict()

        # Staging
        if self._staging:
            d["staging"] = {pid: sp.to_dict()
                           for pid, sp in self._staging.items()}

        # Step params
        if self._step_params:
            d["steps"] = [sp.to_dict() for sp in
                         sorted(self._step_params.values(),
                                key=lambda s: s.step_id)]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskPlan":
        plan = cls(
            assembly_file=d.get("assembly_file", ""),
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
        )
        plan.format_version = d.get("format_version", FORMAT_VERSION)

        fixture = d.get("fixture", {})
        plan.fixture_pos = np.asarray(
            fixture.get("pos", [0, 0, 0]), dtype=float)
        plan.fixture_rotmat = np.asarray(
            fixture.get("rotmat", np.eye(3).tolist()), dtype=float)

        if "robot" in d:
            plan.robot = RobotConfig.from_dict(d["robot"])

        for pid, sdata in d.get("staging", {}).items():
            plan._staging[pid] = StagingPose.from_dict(pid, sdata)

        for sdata in d.get("steps", []):
            sp = StepParams.from_dict(sdata)
            plan._step_params[sp.step_id] = sp

        return plan

    # ── File I/O ─────────────────────────────────────────────
    def save(self, filepath: Union[str, Path]) -> None:
        """Save to a ``.tplan`` file (YAML-backed)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(f"# SEALP Task Plan v{self.format_version}\n")
            fh.write(f"# {self.name}\n\n")
            yaml.dump(data, fh, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TaskPlan":
        """Load from a ``.tplan`` file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    # ── Pretty printing ──────────────────────────────────────
    def summary(self) -> str:
        lines = [
            f"Task Plan: {self.name}",
            f"Description: {self.description}",
            f"Assembly: {self.assembly_file}",
            f"Format: v{self.format_version}",
            f"",
            f"Fixture: pos={self.fixture_pos.tolist()}",
            f"Robot: {self.robot.robot_type} "
            f"at {self.robot.base_pos.tolist()}",
            f"",
            f"Staging ({len(self._staging)}):",
        ]
        for pid, sp in self._staging.items():
            lines.append(f"  {pid}: pos={sp.pos.tolist()}")
        lines.append(f"")
        lines.append(f"Step Params ({len(self._step_params)}):")
        for sid in sorted(self._step_params):
            sp = self._step_params[sid]
            grasp = f"grasp={sp.grasp_id}" if sp.grasp_id is not None else "auto"
            lines.append(
                f"  step {sp.step_id}: {sp.primitive} [{grasp}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"TaskPlan(name={self.name!r}, "
                f"staging={len(self._staging)}, "
                f"steps={len(self._step_params)})")
