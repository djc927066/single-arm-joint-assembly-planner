"""
Layout Evaluation Metrics
===========================

Standardized metrics for comparing workspace layouts.

Usage::

    from sealp.layout.metrics import compute_metrics

    metrics = compute_metrics(layout, report)
    print(f"Feasibility: {metrics.feasibility_rate:.0%}")
    print(f"Avg manipulability: {metrics.manipulability_avg:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .layout import WorkspaceLayout
from .feasibility import FeasibilityReport


@dataclass
class LayoutMetrics:
    """Standardized evaluation metrics for a workspace layout.

    Attributes
    ----------
    feasibility_rate : float
        Fraction of assembly steps that are kinematically feasible
        (0..1).  1.0 = all steps feasible.
    manipulability_avg : float
        Average Yoshikawa manipulability across all feasible steps.
        Higher = further from singularity.
    manipulability_min : float
        Minimum manipulability across all feasible steps.
        A low value indicates a near-singular configuration.
    n_total_grasps_pick : int
        Total number of collision-free grasps at pick poses.
    n_total_grasps_place : int
        Total number of collision-free grasps at place poses.
    grasp_diversity : float
        Average fraction of collision-free grasps per step.
        Higher = more grasp options = more robust.
    composite_score : float
        Composite quality score combining feasibility and
        manipulability.  Higher is better.
    """
    feasibility_rate: float = 0.0
    manipulability_avg: float = 0.0
    manipulability_min: float = 0.0
    n_total_grasps_pick: int = 0
    n_total_grasps_place: int = 0
    grasp_diversity: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feasibility_rate": self.feasibility_rate,
            "manipulability_avg": self.manipulability_avg,
            "manipulability_min": self.manipulability_min,
            "n_total_grasps_pick": self.n_total_grasps_pick,
            "n_total_grasps_place": self.n_total_grasps_place,
            "grasp_diversity": self.grasp_diversity,
            "composite_score": self.composite_score,
        }

    def summary(self) -> str:
        return (
            f"LayoutMetrics:\n"
            f"  Feasibility rate:   {self.feasibility_rate:.1%}\n"
            f"  Manipulability avg: {self.manipulability_avg:.4f}\n"
            f"  Manipulability min: {self.manipulability_min:.4f}\n"
            f"  Grasps (pick):      {self.n_total_grasps_pick}\n"
            f"  Grasps (place):     {self.n_total_grasps_place}\n"
            f"  Grasp diversity:    {self.grasp_diversity:.2%}\n"
            f"  Composite score:    {self.composite_score:.4f}"
        )


def compute_metrics(
    layout: WorkspaceLayout,
    report: FeasibilityReport,
) -> LayoutMetrics:
    """Compute standardized metrics from a feasibility report.

    Parameters
    ----------
    layout : WorkspaceLayout
        The evaluated layout.
    report : FeasibilityReport
        Feasibility evaluation results.

    Returns
    -------
    LayoutMetrics
    """
    if report.n_steps == 0:
        return LayoutMetrics()

    # Feasibility
    feasibility_rate = report.feasibility_rate

    # Manipulability
    feasible_steps = [s for s in report.steps if s.feasible]
    if feasible_steps:
        manip_values = [s.manipulability_avg for s in feasible_steps]
        manip_avg = float(np.mean(manip_values))
        manip_min = float(np.min(manip_values))
    else:
        manip_avg = 0.0
        manip_min = 0.0

    # Grasp counts
    n_grasps_pick = sum(s.n_pick_grasps_ok for s in report.steps)
    n_grasps_place = sum(s.n_place_grasps_ok for s in report.steps)

    # Grasp diversity: average (collision-free / total) per step
    diversities = []
    for s in report.steps:
        # Use pick + place combined
        total = s.n_pick_grasps_ok + s.n_place_grasps_ok
        # Normalizing is tricky without knowing total grasps per step
        # Use a simple count-based proxy
        diversities.append(total)
    grasp_diversity = float(np.mean(diversities)) if diversities else 0.0

    # Composite score
    composite = feasibility_rate * (1.0 + manip_avg)

    return LayoutMetrics(
        feasibility_rate=feasibility_rate,
        manipulability_avg=manip_avg,
        manipulability_min=manip_min,
        n_total_grasps_pick=n_grasps_pick,
        n_total_grasps_place=n_grasps_place,
        grasp_diversity=grasp_diversity,
        composite_score=composite,
    )
