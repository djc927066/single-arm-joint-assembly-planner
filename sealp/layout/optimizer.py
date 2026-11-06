"""
Layout Optimizer — Abstract Interface
========================================

Defines the ``LayoutOptimizer`` ABC and ``OptimizationResult``.
Concrete implementations (random search, GA, CMA-ES) inherit from
this base class.

Usage::

    from sealp.layout.optimizer_random import RandomSearchOptimizer

    opt = RandomSearchOptimizer(n_samples=100, seed=42)
    result = opt.optimize(
        assembly_def=asm,
        task_plan=plan,
        robot_factory=make_robot,
        obstacle_list=[ground],
    )
    print(result.best_layout.summary())
    print(result.best_report.summary())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .layout import WorkspaceLayout
from .feasibility import FeasibilityReport


@dataclass
class OptimizationResult:
    """Result of a layout optimization run.

    Attributes
    ----------
    best_layout : WorkspaceLayout or None
        Best layout found.
    best_report : FeasibilityReport or None
        Feasibility report for the best layout.
    best_score : float
        Score of the best layout.
    history : list of tuple
        ``[(score, layout), ...]`` sorted by score descending.
    n_evaluated : int
        Total number of layouts evaluated.
    """
    best_layout: Optional[WorkspaceLayout] = None
    best_report: Optional[FeasibilityReport] = None
    best_score: float = -np.inf
    history: List[Tuple[float, WorkspaceLayout]] = field(
        default_factory=list)
    n_evaluated: int = 0

    def summary(self) -> str:
        lines = [
            f"Optimization Result",
            f"  Evaluated: {self.n_evaluated} layouts",
            f"  Best score: {self.best_score:.4f}",
        ]
        if self.best_layout is not None:
            lines.append(
                f"  Best layout: {self.best_layout.name}")
            lines.append(
                f"  Robot pos: "
                f"{self.best_layout.robot_base_pos.tolist()}")
        if self.best_report is not None:
            lines.append(
                f"  Feasibility: "
                f"{self.best_report.feasibility_rate:.0%}")
        return "\n".join(lines)


class LayoutOptimizer(ABC):
    """Abstract base class for layout optimizers.

    Subclasses implement ``optimize()`` which searches for the best
    ``WorkspaceLayout`` given an assembly definition.
    """

    @abstractmethod
    def optimize(
        self,
        assembly_def,
        task_plan=None,
        robot=None,
        robot_factory: Optional[Callable] = None,
        grasp_cache: Optional[Dict] = None,
        obstacle_list: Optional[List] = None,
        seed_layout: Optional[WorkspaceLayout] = None,
    ) -> OptimizationResult:
        """Search for the best workspace layout.

        Parameters
        ----------
        assembly_def : AssemblyDef
            Assembly definition.
        task_plan : TaskPlan or None
            Task plan with step params.
        robot : robot or None
            Pre-created robot (will be repositioned).
        robot_factory : callable or None
            ``(pos, rotmat) -> robot`` factory.
        grasp_cache : dict or None
            Pre-computed grasps.
        obstacle_list : list or None
            Static obstacles.
        seed_layout : WorkspaceLayout or None
            Starting layout for warm-start.

        Returns
        -------
        OptimizationResult
        """
        ...
