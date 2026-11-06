"""Paper assembly qualities, independent of geometry, robot and search code.

Wan et al. (arXiv:1609.03108v1), Fig. 6; Chen et al. (TASE 2021),
Eq. (2). The default finite score table is the ASP_OLD implementation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..motion.directions import DirectionResult


@dataclass(frozen=True)
class AssemblabilityScore:
    status: str
    case: str
    region: str
    score: float | str | None
    profile: str
    scope: str = "local_pure_translation; finite_path_checked_separately"


def score_assemblability(
    directions: DirectionResult, *, profile: Literal["asp_old", "wan2016"] = "asp_old"
) -> AssemblabilityScore:
    """Score the solved cone, never the number of display/sample directions.

    ``asp_old`` uses 10/9/3/2/1/0. ``wan2016`` preserves the original
    infinity/10/3/2/1/0 table; infinity is a JSON-safe string. Numerical
    uncertainty is unknown. A sample miss never means a blocked score;
    a separately certified cone shape can still be scored.
    """
    if profile not in ("asp_old", "wan2016"):
        raise ValueError("profile must be asp_old or wan2016")
    k, info = directions.dimension, directions.diagnostics
    rank = info.get("normal_rank")
    if directions.status == "unknown" or directions.issues or k is None:
        return AssemblabilityScore("unknown", "?", "unresolved", None, profile)
    if directions.status == "blocked" and k == 0:
        case, region, value = "i", "empty", 0
    elif directions.status == "unconstrained" and rank == 0:
        case, region, value = "free", "sphere", 10
    elif k == 3 and rank in (1, 2, 3):
        case, region, value = {1: "a", 2: "c", 3: "f"}[rank], "area", 10
    elif k == 2 and info.get("cone_is_subspace") is True:
        case, region, value = "b", "great_circle", 9
    elif k == 2 and info.get("cone_is_subspace") is False and rank in (2, 3):
        case, region, value = ("d" if rank == 2 else "g"), "arc", 3
    elif k == 1 and info.get("feasible_axis_endpoints") in (1, 2):
        value = info["feasible_axis_endpoints"]
        case, region = ("e", "two_points") if value == 2 else ("h", "one_point")
    else:
        return AssemblabilityScore("unknown", "?", "unresolved", None, profile)
    if profile == "wan2016":
        if case in ("free", "a", "c", "f"):
            value = "infinity"
        elif case == "b":
            value = 10
    return AssemblabilityScore("complete", case, region, value, profile)


@dataclass(frozen=True)
class StepQuality:
    stability: float
    graspability: int
    assemblability: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.stability)
            or self.stability < 0
            or not math.isfinite(self.assemblability)
            or self.assemblability < 0
            or type(self.graspability) is not int
            or self.graspability < 0
        ):
            raise ValueError("Use finite nonnegative qualities; unresolved results are not zero")


def sequence_quality(steps: Iterable[StepQuality], *, support_penalty: float = 100.0) -> float:
    """Chen Eq.(2): min(S)min(G)min(A), or min(G)min(A)/(lambda*n).

    n counts states with S=0, not grippers or support-acquisition events.
    A zero S is admissible only if the caller validated finite support.
    """
    if not math.isfinite(support_penalty) or support_penalty < 1:
        raise ValueError("support_penalty must be finite and >= 1")
    steps = tuple(steps)
    if not steps:
        raise ValueError("A scored prefix must contain a step")
    ga = min(s.graspability for s in steps) * min(s.assemblability for s in steps)
    n = sum(s.stability == 0 for s in steps)
    return ga / (support_penalty * n) if n else ga * min(s.stability for s in steps)


def quality_upper_bound(
    steps: Sequence[StepQuality], *, support_penalty: float = 100.0, remaining_steps: int
) -> float:
    """Safe prefix bound, including a possible future first support state.

    The paper's score can increase when its two branches switch. Simply
    pruning on the current score can therefore discard a better leaf.
    """
    value = sequence_quality(steps, support_penalty=support_penalty)
    if type(remaining_steps) is not int or remaining_steps < 0:
        raise ValueError("remaining_steps must be a nonnegative integer")
    if remaining_steps and all(s.stability > 0 for s in steps):
        ga = min(s.graspability for s in steps) * min(s.assemblability for s in steps)
        return max(value, ga / support_penalty)
    return value
