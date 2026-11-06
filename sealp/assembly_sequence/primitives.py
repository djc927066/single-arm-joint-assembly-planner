"""
Motion Primitive Types
=======================

Enum of supported motion primitives for assembly execution.
Used by both ``.tplan`` (StepParams) and legacy AssemblyStep.
"""

from enum import Enum


class Primitive(str, Enum):
    """Motion primitive types for assembly operations.

    Inherits from ``str`` so it serializes cleanly to YAML/JSON
    and can be compared directly with strings::

        >>> Primitive.SINGLE_ARM_TRANSPORT == "single_arm_transport"
        True
        >>> Primitive("dual_arm_cooperative")
        <Primitive.DUAL_ARM_COOPERATIVE: 'dual_arm_cooperative'>
    """

    SINGLE_ARM_TRANSPORT = "single_arm_transport"
    """One arm picks, transports, and places a part."""

    DUAL_ARM_COOPERATIVE = "dual_arm_cooperative"
    """Both arms coordinate to carry a large/heavy part."""

    INSERT = "insert"
    """Linear insertion along a constrained axis (e.g. peg-in-hole)."""

    HOLD_AND_INSERT = "hold_and_insert"
    """One arm holds a part, the other inserts into it."""

    REGRASP = "regrasp"
    """Place on a fixture, release, re-grasp with a better grip."""

    MANUAL = "manual"
    """Human intervention required (not robot-executable)."""

    @classmethod
    def from_str(cls, value: str) -> "Primitive":
        """Parse a string into a Primitive, case-insensitive.

        Raises ``ValueError`` if not a valid primitive.
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            valid = ", ".join(f"'{p.value}'" for p in cls)
            raise ValueError(
                f"Unknown primitive {value!r}. Valid: {valid}")
