"""
Transform Handler
==================

State machine for interactive grab (translate) and rotate operations
on assembly parts.  Inspired by Blender's G / R workflow.

v2: Uses world-space coordinates from ray-plane intersection
    instead of raw mouse deltas for accurate cursor following.
"""

from enum import Enum, auto
import numpy as np


class TransformMode(Enum):
    NONE = auto()
    GRAB = auto()
    ROTATE = auto()


class AxisConstraint(Enum):
    FREE = auto()
    X = auto()
    Y = auto()
    Z = auto()


class TransformHandler:
    """Manages the grab / rotate interactive state.

    Parameters
    ----------
    on_mode_change : callable or None
        ``f(mode: TransformMode, axis: AxisConstraint)`` called whenever
        the mode or constraint changes (for status-bar updates).
    """

    def __init__(self, on_mode_change=None):
        self.mode = TransformMode.NONE
        self.axis = AxisConstraint.FREE
        self._on_mode_change = on_mode_change
        # snapshot of the part pose when transform began
        self._origin_pos: np.ndarray | None = None
        self._origin_rotmat: np.ndarray | None = None
        # world-space hit point when transform began (for grab)
        self._start_world_pos: np.ndarray | None = None
        # mouse xy when transform began (for rotate)
        self._start_mouse = None
        # current result
        self._pos = None
        self._rotmat = None
        self._rot_angle = 0.0

    # ── Public API ──────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self.mode != TransformMode.NONE

    def start_grab(self, pos: np.ndarray, rotmat: np.ndarray,
                   world_hit: np.ndarray):
        """Enter grab mode.

        Parameters
        ----------
        pos, rotmat : part's current pose
        world_hit : world-space point on the grab plane (from ray-cast)
        """
        self._origin_pos = pos.copy()
        self._origin_rotmat = rotmat.copy()
        self._start_world_pos = world_hit.copy()
        self._pos = pos.copy()
        self._rotmat = rotmat.copy()
        self.mode = TransformMode.GRAB
        self.axis = AxisConstraint.FREE
        self._notify()

    def start_rotate(self, pos: np.ndarray, rotmat: np.ndarray,
                     mouse_xy: tuple):
        """Enter rotate mode."""
        self._origin_pos = pos.copy()
        self._origin_rotmat = rotmat.copy()
        self._start_mouse = mouse_xy
        self._pos = pos.copy()
        self._rotmat = rotmat.copy()
        self.mode = TransformMode.ROTATE
        self.axis = AxisConstraint.FREE  # default: Z axis
        self._rot_angle = 0.0
        self._notify()

    def constrain(self, axis: AxisConstraint):
        """Lock to an axis (X, Y, Z, or FREE)."""
        self.axis = axis
        self._notify()

    def update_grab(self, world_hit: np.ndarray):
        """Update grab using world-space hit point.

        Returns ``(pos, rotmat)`` — the new pose to apply.
        """
        if self.mode != TransformMode.GRAB:
            return self._origin_pos, self._origin_rotmat
        if self._start_world_pos is None:
            return self._origin_pos, self._origin_rotmat

        delta = world_hit - self._start_world_pos

        if self.axis == AxisConstraint.X:
            delta[1] = 0
            delta[2] = 0
        elif self.axis == AxisConstraint.Y:
            delta[0] = 0
            delta[2] = 0
        elif self.axis == AxisConstraint.Z:
            delta[0] = 0
            delta[1] = 0
        # FREE: keep full delta on the plane

        self._pos = self._origin_pos + delta
        return self._pos.copy(), self._origin_rotmat.copy()

    def update_rotate(self, mouse_xy: tuple):
        """Update rotate using mouse position.

        Returns ``(pos, rotmat)`` — the new pose to apply.
        """
        if self.mode != TransformMode.ROTATE:
            return self._origin_pos, self._origin_rotmat
        if self._start_mouse is None:
            return self._origin_pos, self._origin_rotmat

        dx = mouse_xy[0] - self._start_mouse[0]
        angle = dx * 3.0  # radians per screen-unit
        self._rot_angle = angle

        ax = self._rotation_axis()
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([
            [0, -ax[2], ax[1]],
            [ax[2], 0, -ax[0]],
            [-ax[1], ax[0], 0],
        ])
        rot_delta = np.eye(3) + s * K + (1 - c) * (K @ K)
        self._rotmat = rot_delta @ self._origin_rotmat
        return self._origin_pos.copy(), self._rotmat.copy()

    def confirm(self):
        """Accept the current transform.

        Returns ``(pos, rotmat)`` — the final confirmed pose.
        """
        pos = self._pos if self._pos is not None else self._origin_pos
        rotmat = self._rotmat if self._rotmat is not None else self._origin_rotmat
        self._reset()
        return pos, rotmat

    def cancel(self):
        """Revert to the original pose.

        Returns ``(pos, rotmat)`` — the original pose.
        """
        pos, rotmat = self._origin_pos, self._origin_rotmat
        self._reset()
        return pos, rotmat

    # ── Internal ────────────────────────────────────────────

    def _reset(self):
        self.mode = TransformMode.NONE
        self.axis = AxisConstraint.FREE
        self._origin_pos = None
        self._origin_rotmat = None
        self._start_world_pos = None
        self._start_mouse = None
        self._pos = None
        self._rotmat = None
        self._rot_angle = 0.0
        self._notify()

    def _notify(self):
        if self._on_mode_change:
            self._on_mode_change(self.mode, self.axis)

    def _rotation_axis(self) -> np.ndarray:
        if self.axis == AxisConstraint.X:
            return np.array([1.0, 0, 0])
        elif self.axis == AxisConstraint.Y:
            return np.array([0, 1.0, 0])
        else:  # Z or FREE (default rotate around Z)
            return np.array([0, 0, 1.0])
