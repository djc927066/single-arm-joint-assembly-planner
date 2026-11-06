"""
Primitive Selector
====================

Maps ``Primitive`` enum values to concrete ``MotionPrimitive`` instances.
Handles both single-arm and dual-arm robot configurations.

Usage::

    from sealp.executor import PrimitiveSelector

    selector = PrimitiveSelector(robot_rgt=rgt_arm, robot_lft=lft_arm)
    primitive = selector.select(Primitive.SINGLE_ARM_TRANSPORT)
    result = primitive.plan(...)
"""

from __future__ import annotations

from typing import Optional

from sealp.assembly_sequence.primitives import Primitive
from sealp.primitives.base import MotionPrimitive
from sealp.primitives.transport import TransportPrimitive
from sealp.primitives.dual_transport import DualTransportPrimitive


class PrimitiveSelector:
    """Select a motion primitive implementation by enum type.

    Parameters
    ----------
    robot_rgt : SglArmRobotInterface
        Primary (right) arm robot.  Used for single-arm primitives
        and as the right arm in dual-arm primitives.
    robot_lft : SglArmRobotInterface or None
        Left arm robot.  Required for dual-arm primitives.
        If ``None``, dual-arm primitives will raise ``ValueError``.
    """

    def __init__(self, robot_rgt, robot_lft=None):
        self.robot_rgt = robot_rgt
        self.robot_lft = robot_lft

        # Single-arm transport (always available)
        self._transport = TransportPrimitive(robot_rgt)

        # Dual-arm transport (only if left arm is provided)
        self._dual_transport = None
        if robot_lft is not None:
            self._dual_transport = DualTransportPrimitive(
                robot_rgt=robot_rgt, robot_lft=robot_lft)

    def select(self, primitive: Primitive) -> MotionPrimitive:
        """Return a ``MotionPrimitive`` instance for the given type.

        Parameters
        ----------
        primitive : Primitive
            The primitive type to select.

        Returns
        -------
        MotionPrimitive

        Raises
        ------
        ValueError
            If the primitive requires a dual-arm setup but no left arm
            was provided.
        NotImplementedError
            If the primitive type is not yet supported.
        """
        if isinstance(primitive, str):
            primitive = Primitive.from_str(primitive)

        if primitive == Primitive.SINGLE_ARM_TRANSPORT:
            return self._transport

        if primitive == Primitive.DUAL_ARM_COOPERATIVE:
            if self._dual_transport is None:
                raise ValueError(
                    f"Primitive {primitive.value!r} requires a dual-arm "
                    f"robot (robot_lft was not provided).")
            return self._dual_transport

        if primitive in (Primitive.INSERT, Primitive.HOLD_AND_INSERT,
                         Primitive.REGRASP):
            raise NotImplementedError(
                f"Primitive {primitive.value!r} is not yet implemented. "
                f"Planned for future development.")

        if primitive == Primitive.MANUAL:
            raise ValueError(
                f"Primitive {primitive.value!r} requires human intervention "
                f"and cannot be executed automatically.")

        raise ValueError(f"Unknown primitive: {primitive!r}")

    @property
    def available_primitives(self) -> list:
        """Return list of currently available primitive types."""
        available = [Primitive.SINGLE_ARM_TRANSPORT]
        if self._dual_transport is not None:
            available.append(Primitive.DUAL_ARM_COOPERATIVE)
        return available
