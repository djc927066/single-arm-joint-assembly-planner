"""
Layout Feasibility Evaluation
================================

Evaluate whether a candidate ``WorkspaceLayout`` allows each assembly
step to be executed.  For each step:

1. Position the robot at the layout's base pose.
2. Check IK reachability at the pick (staging) pose.
3. Check IK reachability at the place (assembly) pose.
4. Record per-step pass/fail and manipulability.
5. Aggregate into a ``FeasibilityReport``.

Usage::

    from sealp.layout.feasibility import evaluate_layout

    report = evaluate_layout(
        layout=my_layout,
        assembly_def=asm,
        task_plan=plan,
        robot=robot,
        obstacle_list=[ground],
    )
    print(report.summary())
    print(f"Feasibility rate: {report.feasibility_rate:.1%}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import wrs.modeling.collision_model as mcm

from .layout import WorkspaceLayout
from .reachability import (
    check_ik_reachability,
    check_pose_reachability,
    PoseReachabilityResult,
)


# ══════════════════════════════════════════════════════════════
#  Per-step feasibility
# ══════════════════════════════════════════════════════════════
@dataclass
class StepFeasibility:
    """Feasibility result for a single assembly step.

    Attributes
    ----------
    step_id : int
        Assembly step index.
    part_id : str
        Part being assembled.
    pick_reachable : bool
        Whether at least one grasp reaches the pick (staging) pose.
    place_reachable : bool
        Whether at least one grasp reaches the place (assembly) pose.
    pick_collision_free : bool
        Whether a collision-free IK solution exists at pick.
    place_collision_free : bool
        Whether a collision-free IK solution exists at place.
    n_pick_grasps_ok : int
        Number of collision-free grasps at pick.
    n_place_grasps_ok : int
        Number of collision-free grasps at place.
    manipulability_pick : float
        Best manipulability score at pick pose.
    manipulability_place : float
        Best manipulability score at place pose.
    feasible : bool
        Overall: both pick and place are reachable and collision-free.
    error_msg : str
        Reason for infeasibility (empty if feasible).
    """
    step_id: int = -1
    part_id: str = ""
    pick_reachable: bool = False
    place_reachable: bool = False
    pick_collision_free: bool = False
    place_collision_free: bool = False
    n_pick_grasps_ok: int = 0
    n_place_grasps_ok: int = 0
    manipulability_pick: float = 0.0
    manipulability_place: float = 0.0
    error_msg: str = ""

    @property
    def feasible(self) -> bool:
        return (self.pick_reachable and self.place_reachable
                and self.pick_collision_free and self.place_collision_free)

    @property
    def manipulability_avg(self) -> float:
        return (self.manipulability_pick + self.manipulability_place) / 2.0


# ══════════════════════════════════════════════════════════════
#  Feasibility Report
# ══════════════════════════════════════════════════════════════
@dataclass
class FeasibilityReport:
    """Aggregate feasibility results for an entire layout.

    Attributes
    ----------
    layout : WorkspaceLayout
        The layout that was evaluated.
    steps : list of StepFeasibility
        Per-step results.
    """
    layout: Optional[WorkspaceLayout] = None
    steps: List[StepFeasibility] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_feasible(self) -> int:
        return sum(1 for s in self.steps if s.feasible)

    @property
    def n_infeasible(self) -> int:
        return self.n_steps - self.n_feasible

    @property
    def feasibility_rate(self) -> float:
        """Fraction of steps that are feasible (0..1)."""
        if self.n_steps == 0:
            return 0.0
        return self.n_feasible / self.n_steps

    @property
    def all_feasible(self) -> bool:
        return all(s.feasible for s in self.steps)

    @property
    def avg_manipulability(self) -> float:
        """Average manipulability across all feasible steps."""
        feasible = [s for s in self.steps if s.feasible]
        if not feasible:
            return 0.0
        return sum(s.manipulability_avg for s in feasible) / len(feasible)

    @property
    def score(self) -> float:
        """Composite score: feasibility_rate * (1 + avg_manipulability).

        Higher is better.  Fully feasible layouts with high
        manipulability score highest.
        """
        return self.feasibility_rate * (1.0 + self.avg_manipulability)

    def summary(self) -> str:
        lines = [
            f"Feasibility Report",
            f"  Layout: {self.layout.name if self.layout else '?'}",
            f"  Steps: {self.n_feasible}/{self.n_steps} feasible "
            f"({self.feasibility_rate:.0%})",
            f"  Avg manipulability: {self.avg_manipulability:.4f}",
            f"  Composite score: {self.score:.4f}",
            "",
        ]
        for s in self.steps:
            icon = "[OK]" if s.feasible else "[FAIL]"
            pick = f"pick({'Y' if s.pick_collision_free else 'N'}" \
                   f" {s.n_pick_grasps_ok}g m={s.manipulability_pick:.3f})"
            place = f"place({'Y' if s.place_collision_free else 'N'}" \
                    f" {s.n_place_grasps_ok}g m={s.manipulability_place:.3f})"
            msg = f" -- {s.error_msg}" if s.error_msg else ""
            lines.append(
                f"  {icon} Step {s.step_id} ({s.part_id}): "
                f"{pick} | {place}{msg}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Main evaluation function
# ══════════════════════════════════════════════════════════════
def evaluate_layout(
    layout: WorkspaceLayout,
    assembly_def,
    task_plan=None,
    robot=None,
    robot_factory: Optional[Callable] = None,
    grasp_cache: Optional[Dict] = None,
    obstacle_list: Optional[List] = None,
    max_grasps_per_step: int = 30,
    verbose: bool = True,
) -> FeasibilityReport:
    """Evaluate layout feasibility for all assembly steps.

    For each assembly step, checks IK reachability and collision
    at the pick (staging) and place (assembly) poses.

    Parameters
    ----------
    layout : WorkspaceLayout
        Candidate layout to evaluate.
    assembly_def : AssemblyDef
        Assembly definition (parts, steps, models).
    task_plan : TaskPlan or None
        Task plan (for step params). If None, defaults are used.
    robot : robot instance or None
        Pre-created robot. If provided, it will be repositioned
        to the layout's base pose. If None, ``robot_factory`` is
        used to create one.
    robot_factory : callable or None
        ``(pos, rotmat) -> robot`` factory. Used if ``robot`` is None.
    grasp_cache : dict or None
        ``{model_alias: grasp_collection}``. If None, grasps must
        be planned (slow).
    obstacle_list : list or None
        Static obstacles (ground, fixtures).
    max_grasps_per_step : int
        Max grasps to evaluate per step.
    verbose : bool
        Print progress.

    Returns
    -------
    FeasibilityReport
    """
    if obstacle_list is None:
        obstacle_list = []
    if grasp_cache is None:
        grasp_cache = {}

    # ── Get or create robot ──────────────────────────────────
    if robot is None and robot_factory is not None:
        robot = robot_factory(layout.robot_base_pos,
                              layout.robot_base_rotmat)
    elif robot is not None:
        # Reposition existing robot
        _reposition_robot(robot, layout.robot_base_pos,
                          layout.robot_base_rotmat)
    else:
        raise ValueError(
            "Either 'robot' or 'robot_factory' must be provided.")

    # ── Detect single vs dual arm ────────────────────────────
    is_dual = hasattr(robot, 'rgt_arm') and hasattr(robot, 'lft_arm')
    eval_robot = robot.rgt_arm if is_dual else robot

    # ── Compute world assembly poses ─────────────────────────
    fixture_pos = layout.assembly_station_pos
    fixture_rotmat = layout.assembly_station_rotmat
    world_poses = assembly_def.compute_world_poses(
        fixture_pos=fixture_pos, fixture_rotmat=fixture_rotmat)

    # ── Get execution order ──────────────────────────────────
    steps = assembly_def.get_execution_order()

    if verbose:
        print("=" * 60)
        print(f"Layout Feasibility Evaluation: {layout.name}")
        print(f"  Steps: {len(steps)}  |  Robot at: "
              f"{layout.robot_base_pos.tolist()}")
        print("=" * 60)

    report = FeasibilityReport(layout=layout)

    # Growing obstacle list (parts placed so far)
    dynamic_obstacles = list(obstacle_list)

    for i, step in enumerate(steps):
        part_id = step.part_id
        if verbose:
            print(f"\n-- Evaluating step {step.step_id}: "
                  f"{part_id!r} ({i+1}/{len(steps)}) --")

        sf = _evaluate_step(
            step=step,
            assembly_def=assembly_def,
            layout=layout,
            world_poses=world_poses,
            robot=eval_robot,
            grasp_cache=grasp_cache,
            obstacle_list=dynamic_obstacles,
            max_grasps=max_grasps_per_step,
            verbose=verbose,
        )
        report.steps.append(sf)

        # Add placed part to dynamic obstacles
        if sf.feasible:
            goal_pos, goal_rotmat = world_poses.get(
                part_id, (np.zeros(3), np.eye(3)))
            model_path = assembly_def.model_path(part_id)
            if os.path.isfile(model_path):
                placed = mcm.CollisionModel(initor=model_path)
                placed.pos = goal_pos
                placed.rotmat = goal_rotmat
                dynamic_obstacles.append(placed)

    if verbose:
        print("\n" + "=" * 60)
        print(report.summary())
        print("=" * 60)

    return report


# ══════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════
def _reposition_robot(robot, pos: np.ndarray, rotmat: np.ndarray):
    """Move robot base to a new pose."""
    if hasattr(robot, 'fix_to'):
        robot.fix_to(pos=pos, rotmat=rotmat)
    else:
        # Fallback for robots without fix_to at top level
        robot._pos = pos
        robot._rotmat = rotmat


def _evaluate_step(
    step,
    assembly_def,
    layout: WorkspaceLayout,
    world_poses: Dict,
    robot,
    grasp_cache: Dict,
    obstacle_list: List,
    max_grasps: int,
    verbose: bool,
) -> StepFeasibility:
    """Evaluate feasibility of a single assembly step."""
    part_id = step.part_id
    sf = StepFeasibility(step_id=step.step_id, part_id=part_id)

    # ── Load grasps ──────────────────────────────────────────
    part_def = assembly_def.get_part(part_id)
    model_alias = part_def.model
    grasp_collection = grasp_cache.get(model_alias)

    if grasp_collection is None or len(grasp_collection) == 0:
        sf.error_msg = f"No grasps for model {model_alias!r}"
        if verbose:
            print(f"  [!] {sf.error_msg}")
        return sf

    # ── Get staging (pick) pose ──────────────────────────────
    staging = layout.get_staging(part_id)
    if staging is None:
        sf.error_msg = f"No staging position for {part_id!r}"
        if verbose:
            print(f"  [!] {sf.error_msg}")
        return sf
    pick_pos, pick_rotmat = staging

    # ── Get assembly (place) pose ────────────────────────────
    if part_id not in world_poses:
        sf.error_msg = f"No world pose for {part_id!r}"
        if verbose:
            print(f"  [!] {sf.error_msg}")
        return sf
    place_pos, place_rotmat = world_poses[part_id]

    # ── Check pick reachability ──────────────────────────────
    pick_result = check_pose_reachability(
        robot=robot,
        obj_pos=pick_pos,
        obj_rotmat=pick_rotmat,
        grasp_collection=grasp_collection,
        obstacle_list=obstacle_list,
        max_grasps=max_grasps,
    )
    sf.pick_reachable = pick_result.reachable
    sf.pick_collision_free = pick_result.n_collision_free > 0
    sf.n_pick_grasps_ok = pick_result.n_collision_free
    sf.manipulability_pick = pick_result.best_manipulability

    if verbose:
        print(f"  Pick:  reachable={sf.pick_reachable} "
              f"cfree={sf.n_pick_grasps_ok}/{pick_result.n_total_grasps} "
              f"manip={sf.manipulability_pick:.4f}")

    # ── Check place reachability ─────────────────────────────
    place_result = check_pose_reachability(
        robot=robot,
        obj_pos=place_pos,
        obj_rotmat=place_rotmat,
        grasp_collection=grasp_collection,
        obstacle_list=obstacle_list,
        max_grasps=max_grasps,
    )
    sf.place_reachable = place_result.reachable
    sf.place_collision_free = place_result.n_collision_free > 0
    sf.n_place_grasps_ok = place_result.n_collision_free
    sf.manipulability_place = place_result.best_manipulability

    if verbose:
        print(f"  Place: reachable={sf.place_reachable} "
              f"cfree={sf.n_place_grasps_ok}/{place_result.n_total_grasps} "
              f"manip={sf.manipulability_place:.4f}")

    # ── Set error if infeasible ──────────────────────────────
    if not sf.feasible:
        reasons = []
        if not sf.pick_reachable:
            reasons.append("pick unreachable")
        elif not sf.pick_collision_free:
            reasons.append("pick collision")
        if not sf.place_reachable:
            reasons.append("place unreachable")
        elif not sf.place_collision_free:
            reasons.append("place collision")
        sf.error_msg = "; ".join(reasons)

    return sf
