"""
Workspace Layout Representation
=================================

Defines the ``WorkspaceLayout`` dataclass — the core data structure
for layout optimization.  A layout specifies:

- Robot base position and orientation
- Part staging positions (where the robot picks each part)
- Assembly station (fixture) pose
- Static fixture obstacles

Layouts can be serialized to ``.layout`` YAML files and loaded back.

A layout can also be extracted from an existing ``TaskPlan``.

Usage::

    from sealp.layout import WorkspaceLayout

    layout = WorkspaceLayout(
        robot_base_pos=np.array([0, -0.3, 0]),
        staging_positions={"seat": (np.array([0.3, 0, 0.8]), np.eye(3))},
    )
    layout.save("my_layout.layout")

    loaded = WorkspaceLayout.load("my_layout.layout")
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

# ── YAML float representer (clean output) ────────────────────
def _float_representer(dumper: yaml.Dumper, value: float):
    if value != value:
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    if value == float("inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
    if value == float("-inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", "-.inf")
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.6g}")

yaml.add_representer(float, _float_representer)


def _tuple_representer(dumper: yaml.Dumper, value: tuple):
    """Save tuples as plain YAML lists (avoid !!python/tuple)."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", list(value))


yaml.add_representer(tuple, _tuple_representer)


def _construct_python_tuple(loader: yaml.SafeLoader, node):
    """Allow loading legacy layouts that contain !!python/tuple in metadata."""
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)

FORMAT_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════
#  WorkspaceLayout
# ══════════════════════════════════════════════════════════════
@dataclass
class WorkspaceLayout:
    """Workspace layout for assembly execution.

    A layout fully specifies the spatial configuration of the workspace:
    where the robot sits, where each part starts, where the assembly
    station is, and any fixed obstacles (fixtures, tables, etc.).

    Attributes
    ----------
    robot_base_pos : np.ndarray
        Robot base position ``[x, y, z]``.
    robot_base_rotmat : np.ndarray
        Robot base orientation (3×3 rotation matrix).
    staging_positions : dict
        ``{part_id: (pos, rotmat)}`` — where each part is staged
        for pickup.
    assembly_station_pos : np.ndarray
        World position of the assembly station (fixture).
    assembly_station_rotmat : np.ndarray
        World orientation of the assembly station.
    fixture_obstacles : list of dict
        Static obstacle definitions (box/STL specs) in the layout.
        Each dict has keys: ``name``, ``type``, ``extent``/``file``,
        ``pos``, ``rotmat``, ``rgba``.
    workspace_bounds : dict or None
        Optional bounding box for the workspace.
        ``{"min": [x,y,z], "max": [x,y,z]}``.
    name : str
        Human-readable layout name.
    metadata : dict
        Arbitrary extra data.
    """

    robot_base_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    robot_base_rotmat: np.ndarray = field(
        default_factory=lambda: np.eye(3))
    staging_positions: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict)
    assembly_station_pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    assembly_station_rotmat: np.ndarray = field(
        default_factory=lambda: np.eye(3))
    fixture_obstacles: List[dict] = field(default_factory=list)
    workspace_bounds: Optional[dict] = None
    name: str = "unnamed_layout"
    metadata: dict = field(default_factory=dict)

    # ── Factory: from TaskPlan ───────────────────────────────
    @classmethod
    def from_task_plan(cls, task_plan) -> "WorkspaceLayout":
        """Extract a layout from an existing TaskPlan.

        Parameters
        ----------
        task_plan : sealp.assembly_sequence.tplan.TaskPlan
            A task plan with robot config, staging, and fixture.

        Returns
        -------
        WorkspaceLayout
        """
        staging = {}
        for pid, sp in task_plan.staging.items():
            staging[pid] = (sp.pos.copy(), sp.rotmat.copy())

        return cls(
            robot_base_pos=task_plan.robot.base_pos.copy(),
            robot_base_rotmat=task_plan.robot.base_rotmat.copy(),
            staging_positions=staging,
            assembly_station_pos=task_plan.fixture_pos.copy(),
            assembly_station_rotmat=task_plan.fixture_rotmat.copy(),
            name=f"layout_from_{task_plan.name}",
        )

    # ── Convenience ──────────────────────────────────────────
    def set_staging(self, part_id: str, pos: np.ndarray,
                    rotmat: np.ndarray = None):
        """Set or update the staging position for a part."""
        if rotmat is None:
            rotmat = np.eye(3)
        self.staging_positions[part_id] = (pos.copy(), rotmat.copy())

    def get_staging(self, part_id: str
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (pos, rotmat) for a part's staging, or None."""
        return self.staging_positions.get(part_id)

    @property
    def n_parts(self) -> int:
        """Number of staged parts in this layout."""
        return len(self.staging_positions)

    def copy(self) -> "WorkspaceLayout":
        """Return a deep copy of this layout."""
        return WorkspaceLayout.from_dict(self.to_dict())

    # ── Apply to TaskPlan ────────────────────────────────────
    def apply_to_task_plan(self, task_plan) -> None:
        """Write this layout's positions into a TaskPlan.

        Modifies ``task_plan`` in-place: sets robot base,
        staging positions, and fixture pose.

        Parameters
        ----------
        task_plan : sealp.assembly_sequence.tplan.TaskPlan
        """
        task_plan.robot.base_pos = self.robot_base_pos.copy()
        task_plan.robot.base_rotmat = self.robot_base_rotmat.copy()
        task_plan.fixture_pos = self.assembly_station_pos.copy()
        task_plan.fixture_rotmat = self.assembly_station_rotmat.copy()
        for pid, (pos, rotmat) in self.staging_positions.items():
            task_plan.set_staging(pid, pos, rotmat)

    # ── Serialization ────────────────────────────────────────
    def to_dict(self) -> dict:
        """Convert to a plain dict for YAML serialization."""
        staging = {}
        for pid, (pos, rotmat) in self.staging_positions.items():
            staging[pid] = {
                "pos": [float(v) for v in pos],
                "rotmat": [[float(v) for v in row] for row in rotmat],
            }

        d = {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "robot_base": {
                "pos": [float(v) for v in self.robot_base_pos],
                "rotmat": [[float(v) for v in row]
                           for row in self.robot_base_rotmat],
            },
            "assembly_station": {
                "pos": [float(v) for v in self.assembly_station_pos],
                "rotmat": [[float(v) for v in row]
                           for row in self.assembly_station_rotmat],
            },
            "staging": staging,
        }
        if self.fixture_obstacles:
            d["fixture_obstacles"] = copy.deepcopy(self.fixture_obstacles)
        if self.workspace_bounds is not None:
            d["workspace_bounds"] = copy.deepcopy(self.workspace_bounds)
        if self.metadata:
            d["metadata"] = copy.deepcopy(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceLayout":
        """Construct from a plain dict (parsed YAML)."""
        robot = d.get("robot_base", {})
        station = d.get("assembly_station", {})

        staging = {}
        for pid, sdata in d.get("staging", {}).items():
            staging[pid] = (
                np.asarray(sdata.get("pos", [0, 0, 0]), dtype=float),
                np.asarray(sdata.get("rotmat", np.eye(3).tolist()),
                           dtype=float),
            )

        return cls(
            robot_base_pos=np.asarray(
                robot.get("pos", [0, 0, 0]), dtype=float),
            robot_base_rotmat=np.asarray(
                robot.get("rotmat", np.eye(3).tolist()), dtype=float),
            staging_positions=staging,
            assembly_station_pos=np.asarray(
                station.get("pos", [0, 0, 0]), dtype=float),
            assembly_station_rotmat=np.asarray(
                station.get("rotmat", np.eye(3).tolist()), dtype=float),
            fixture_obstacles=d.get("fixture_obstacles", []),
            workspace_bounds=d.get("workspace_bounds"),
            name=d.get("name", "unnamed_layout"),
            metadata=d.get("metadata", {}),
        )

    # ── File I/O ─────────────────────────────────────────────
    def save(self, filepath: Union[str, Path]) -> None:
        """Save to a ``.layout`` YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(f"# SEALP Workspace Layout v{FORMAT_VERSION}\n")
            fh.write(f"# {self.name}\n\n")
            yaml.dump(data, fh, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "WorkspaceLayout":
        """Load from a ``.layout`` YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    # ── Pretty printing ──────────────────────────────────────
    def summary(self) -> str:
        lines = [
            f"WorkspaceLayout: {self.name}",
            f"  Robot base: pos={self.robot_base_pos.tolist()}",
            f"  Assembly station: pos={self.assembly_station_pos.tolist()}",
            f"  Staging positions ({self.n_parts}):",
        ]
        for pid, (pos, _) in self.staging_positions.items():
            lines.append(f"    {pid}: pos={pos.tolist()}")
        if self.fixture_obstacles:
            lines.append(f"  Fixture obstacles: {len(self.fixture_obstacles)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"WorkspaceLayout(name={self.name!r}, "
                f"staging={self.n_parts})")
