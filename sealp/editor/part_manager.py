"""
Part Manager
=============

Manages loaded assembly parts in the 3D Panda3D scene.  Handles
loading STL/OBJ models as ``CollisionModel`` instances, setting
collision bitmasks for mouse picking, and synchronizing poses
back to the ``AssemblySequence``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.basis.robot_math as rm

from sealp.assembly_sequence.assembly_part import AssemblyPart

# Distinct colors auto-assigned to parts (tab10-ish palette)
PART_COLORS = [
    np.array([0.40, 0.76, 0.96, 1.0]),  # sky blue
    np.array([0.96, 0.65, 0.35, 1.0]),  # orange
    np.array([0.55, 0.85, 0.50, 1.0]),  # green
    np.array([0.95, 0.50, 0.50, 1.0]),  # red
    np.array([0.75, 0.60, 0.88, 1.0]),  # purple
    np.array([0.98, 0.85, 0.40, 1.0]),  # gold
    np.array([0.55, 0.80, 0.78, 1.0]),  # teal
    np.array([0.90, 0.55, 0.75, 1.0]),  # pink
    np.array([0.70, 0.70, 0.70, 1.0]),  # grey
    np.array([0.40, 0.55, 0.80, 1.0]),  # steel blue
]

SELECTED_COLOR = np.array([1.0, 1.0, 0.3, 1.0])  # bright yellow


@dataclass
class PartEntry:
    """A part loaded into the 3D scene."""
    part: AssemblyPart
    cmodel: Optional[mcm.CollisionModel] = None
    color: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 0.7, 1.0]))
    selected: bool = False
    visible: bool = True
    # assembly pose ghost (semi-transparent)
    ghost_cmodel: Optional[mcm.CollisionModel] = None


class PartManager:
    """Manage parts in the 3D scene.

    Parameters
    ----------
    world : ShowBase
        The Panda3D world to attach models to.
    """

    def __init__(self, world):
        self._world = world
        self._entries: Dict[str, PartEntry] = {}
        self._color_idx = 0

    @property
    def part_ids(self) -> List[str]:
        return list(self._entries.keys())

    @property
    def entries(self) -> Dict[str, PartEntry]:
        return self._entries

    @property
    def selected_id(self) -> Optional[str]:
        for pid, entry in self._entries.items():
            if entry.selected:
                return pid
        return None

    def load_part(self, part: AssemblyPart) -> Optional[PartEntry]:
        """Load a part's 3D model into the scene.

        If the model file doesn't exist, creates a placeholder box.
        """
        color = PART_COLORS[self._color_idx % len(PART_COLORS)]
        self._color_idx += 1

        if os.path.isfile(part.model_path):
            try:
                cmodel = mcm.CollisionModel(initor=part.model_path,
                                            name=f"part_{part.part_id}")
            except Exception as e:
                print(f"[PartManager] Failed to load {part.model_path}: {e}")
                cmodel = self._make_placeholder(part.part_id)
        else:
            cmodel = self._make_placeholder(part.part_id)

        cmodel.rgba = color
        cmodel.pos = part.init_pos
        cmodel.rotmat = part.init_rotmat
        cmodel.attach_to(self._world)

        entry = PartEntry(part=part, cmodel=cmodel, color=color)
        self._entries[part.part_id] = entry
        return entry

    def remove_part(self, part_id: str):
        """Remove a part from the scene."""
        entry = self._entries.pop(part_id, None)
        if entry and entry.cmodel:
            entry.cmodel.detach()
        if entry and entry.ghost_cmodel:
            entry.ghost_cmodel.detach()

    def select_part(self, part_id: str):
        """Select a part — highlight it and deselect others."""
        for pid, e in self._entries.items():
            if pid == part_id:
                e.selected = True
                if e.cmodel:
                    e.cmodel.rgba = SELECTED_COLOR
            else:
                if e.selected:
                    e.selected = False
                    if e.cmodel:
                        e.cmodel.rgba = e.color

    def deselect_all(self):
        """Deselect all parts."""
        for e in self._entries.values():
            e.selected = False
            if e.cmodel:
                e.cmodel.rgba = e.color

    def get_selected_entry(self) -> Optional[PartEntry]:
        """Return the currently selected PartEntry, or None."""
        for e in self._entries.values():
            if e.selected:
                return e
        return None

    def update_part_pose(self, part_id: str, pos: np.ndarray,
                         rotmat: np.ndarray):
        """Update the 3D position and orientation of a part."""
        entry = self._entries.get(part_id)
        if entry and entry.cmodel:
            entry.cmodel.detach()
            entry.cmodel.pos = pos
            entry.cmodel.rotmat = rotmat
            entry.cmodel.attach_to(self._world)
            # update the underlying AssemblyPart
            entry.part.init_pos = pos.copy()
            entry.part.init_rotmat = rotmat.copy()

    def show_assembly_ghost(self, part_id: str):
        """Show a semi-transparent ghost at the assembly pose."""
        entry = self._entries.get(part_id)
        if not entry:
            return
        # remove old ghost
        if entry.ghost_cmodel:
            entry.ghost_cmodel.detach()
        if entry.cmodel:
            ghost = entry.cmodel.copy()
            ghost.rgba = np.array([*entry.color[:3], 0.25])
            ghost.pos = entry.part.assembly_pos
            ghost.rotmat = entry.part.assembly_rotmat
            ghost.attach_to(self._world)
            entry.ghost_cmodel = ghost

    def hide_assembly_ghost(self, part_id: str):
        """Remove the assembly-pose ghost for a part."""
        entry = self._entries.get(part_id)
        if entry and entry.ghost_cmodel:
            entry.ghost_cmodel.detach()
            entry.ghost_cmodel = None

    def refresh_all(self):
        """Detach and re-attach all parts (after pose changes)."""
        for entry in self._entries.values():
            if entry.cmodel:
                entry.cmodel.detach()
                entry.cmodel.attach_to(self._world)

    def sync_to_sequence(self, sequence):
        """Push current 3D poses back into an AssemblySequence."""
        for pid, entry in self._entries.items():
            try:
                part = sequence.get_part(pid)
                if entry.cmodel:
                    part.init_pos = entry.cmodel.pos.copy()
                    part.init_rotmat = entry.cmodel.rotmat.copy()
            except KeyError:
                pass

    def _make_placeholder(self, part_id: str):
        """Create a small box when the real model can't be loaded."""
        return mcm.gen_box(xyz_lengths=np.array([0.05, 0.05, 0.05]),
                           pos=np.zeros(3),
                           rgb=np.array([0.5, 0.5, 0.5]),
                           alpha=1.0)
