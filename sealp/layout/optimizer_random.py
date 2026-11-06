"""
Random Search Layout Optimizer
=================================

Baseline optimizer that samples random layouts within configured
workspace bounds, evaluates each for feasibility, and returns the
best one found.

This serves as a reference implementation and baseline comparison
for more sophisticated optimizers (GA, CMA-ES).

Usage::

    from sealp.layout.optimizer_random import RandomSearchOptimizer

    opt = RandomSearchOptimizer(
        n_samples=200,
        robot_bounds={"xy_min": [-0.5, -0.5], "xy_max": [0.5, 0.5]},
        staging_bounds={"xy_min": [-0.8, -0.8], "xy_max": [0.8, 0.8],
                        "z_range": [0.75, 0.85]},
        seed=42,
    )
    result = opt.optimize(
        assembly_def=asm,
        robot=robot,
        grasp_cache=grasps,
        obstacle_list=[ground],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .layout import WorkspaceLayout
from .feasibility import FeasibilityReport, evaluate_layout
from .optimizer import LayoutOptimizer, OptimizationResult


@dataclass
class SearchBounds:
    """Axis-aligned bounding box for random sampling.

    Attributes
    ----------
    xy_min : np.ndarray
        Minimum ``[x, y]`` (meters).
    xy_max : np.ndarray
        Maximum ``[x, y]`` (meters).
    z_range : tuple of float
        ``(z_min, z_max)`` for the z-axis.
    yaw_range : tuple of float
        ``(yaw_min, yaw_max)`` in radians for base rotation about z.
    """
    xy_min: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, -0.5]))
    xy_max: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5]))
    z_range: Tuple[float, float] = (0.0, 0.0)
    yaw_range: Tuple[float, float] = (-np.pi, np.pi)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchBounds":
        return cls(
            xy_min=np.asarray(d.get("xy_min", [-0.5, -0.5]), dtype=float),
            xy_max=np.asarray(d.get("xy_max", [0.5, 0.5]), dtype=float),
            z_range=tuple(d.get("z_range", [0.0, 0.0])),
            yaw_range=tuple(d.get("yaw_range", [-np.pi, np.pi])),
        )


class RandomSearchOptimizer(LayoutOptimizer):
    """Random search baseline for layout optimization.

    Samples ``n_samples`` random layouts where the robot base position
    and staging positions are uniformly sampled within the given bounds.
    Returns the layout with the highest composite feasibility score.

    Parameters
    ----------
    n_samples : int
        Number of random layouts to evaluate.
    robot_bounds : dict or SearchBounds
        Bounds for sampling robot base position.
    staging_bounds : dict or SearchBounds or None
        Bounds for sampling staging positions.  If None, staging
        positions from the seed layout / task plan are kept fixed.
    randomize_staging : bool
        If True, randomize staging positions.  If False, only
        the robot base is randomized.
    assembly_station_bounds : dict or SearchBounds or None
        Bounds for assembly station position.  If None, the assembly
        station position is kept fixed.
    seed : int or None
        Random seed for reproducibility.
    max_grasps_per_step : int
        Max grasps to test per step during evaluation.
    verbose : bool
        Print progress during optimization.
    """

    def __init__(
        self,
        n_samples: int = 100,
        robot_bounds: dict = None,
        staging_bounds: dict = None,
        randomize_staging: bool = False,
        assembly_station_bounds: dict = None,
        seed: int = None,
        max_grasps_per_step: int = 20,
        verbose: bool = True,
    ):
        self.n_samples = n_samples
        self.randomize_staging = randomize_staging
        self.max_grasps_per_step = max_grasps_per_step
        self.verbose = verbose

        # Parse bounds
        if robot_bounds is None:
            robot_bounds = {}
        self.robot_bounds = (
            robot_bounds if isinstance(robot_bounds, SearchBounds)
            else SearchBounds.from_dict(robot_bounds)
        )

        if staging_bounds is not None:
            self.staging_bounds = (
                staging_bounds if isinstance(staging_bounds, SearchBounds)
                else SearchBounds.from_dict(staging_bounds)
            )
        else:
            self.staging_bounds = None

        if assembly_station_bounds is not None:
            self.assembly_bounds = (
                assembly_station_bounds
                if isinstance(assembly_station_bounds, SearchBounds)
                else SearchBounds.from_dict(assembly_station_bounds)
            )
        else:
            self.assembly_bounds = None

        # RNG
        self.rng = np.random.default_rng(seed)

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
        """Run random search optimization.

        Parameters
        ----------
        assembly_def : AssemblyDef
        task_plan : TaskPlan or None
        robot : robot or None
        robot_factory : callable or None
        grasp_cache : dict or None
        obstacle_list : list or None
        seed_layout : WorkspaceLayout or None
            Base layout for staging positions (if not randomizing).

        Returns
        -------
        OptimizationResult
        """
        # Build base layout (staging positions to inherit)
        if seed_layout is not None:
            base_staging = dict(seed_layout.staging_positions)
            base_station_pos = seed_layout.assembly_station_pos.copy()
            base_station_rotmat = seed_layout.assembly_station_rotmat.copy()
        elif task_plan is not None:
            seed = WorkspaceLayout.from_task_plan(task_plan)
            base_staging = dict(seed.staging_positions)
            base_station_pos = seed.assembly_station_pos.copy()
            base_station_rotmat = seed.assembly_station_rotmat.copy()
        else:
            base_staging = {}
            base_station_pos = np.zeros(3)
            base_station_rotmat = np.eye(3)

        result = OptimizationResult()
        t_start = time.time()

        if self.verbose:
            print("=" * 60)
            print(f"Random Search Optimizer: {self.n_samples} samples")
            print("=" * 60)

        for i in range(self.n_samples):
            # ── Sample a random layout ───────────────────────
            layout = self._sample_layout(
                base_staging=base_staging,
                base_station_pos=base_station_pos,
                base_station_rotmat=base_station_rotmat,
                part_ids=assembly_def.part_ids,
                index=i,
            )

            # ── Evaluate feasibility ─────────────────────────
            try:
                report = evaluate_layout(
                    layout=layout,
                    assembly_def=assembly_def,
                    task_plan=task_plan,
                    robot=robot,
                    robot_factory=robot_factory,
                    grasp_cache=grasp_cache,
                    obstacle_list=obstacle_list,
                    max_grasps_per_step=self.max_grasps_per_step,
                    verbose=False,
                )
            except Exception as e:
                if self.verbose:
                    print(f"  [{i+1}] Error: {e}")
                continue

            score = report.score
            result.n_evaluated += 1
            result.history.append((score, layout))

            if score > result.best_score:
                result.best_score = score
                result.best_layout = layout.copy()
                result.best_report = report

                if self.verbose:
                    print(
                        f"  [{i+1}/{self.n_samples}] * New best: "
                        f"score={score:.4f} "
                        f"feas={report.feasibility_rate:.0%} "
                        f"manip={report.avg_manipulability:.4f} "
                        f"robot={layout.robot_base_pos.tolist()}"
                    )
            elif self.verbose and (i + 1) % max(1, self.n_samples // 10) == 0:
                print(
                    f"  [{i+1}/{self.n_samples}] "
                    f"score={score:.4f} "
                    f"(best={result.best_score:.4f})"
                )

        elapsed = time.time() - t_start

        # Sort history by score descending
        result.history.sort(key=lambda x: x[0], reverse=True)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Optimization complete in {elapsed:.1f}s")
            print(result.summary())
            print("=" * 60)

        return result

    # ── Internal ─────────────────────────────────────────────
    def _sample_layout(
        self,
        base_staging: Dict,
        base_station_pos: np.ndarray,
        base_station_rotmat: np.ndarray,
        part_ids: List[str],
        index: int,
    ) -> WorkspaceLayout:
        """Generate a random layout sample."""
        rb = self.robot_bounds

        # Random robot base position
        x = self.rng.uniform(rb.xy_min[0], rb.xy_max[0])
        y = self.rng.uniform(rb.xy_min[1], rb.xy_max[1])
        z = self.rng.uniform(rb.z_range[0], rb.z_range[1])
        robot_pos = np.array([x, y, z])

        # Random robot base yaw
        yaw = self.rng.uniform(rb.yaw_range[0], rb.yaw_range[1])
        cy, sy = np.cos(yaw), np.sin(yaw)
        robot_rotmat = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0,  1],
        ], dtype=float)

        # Staging positions
        staging = {}
        if self.randomize_staging and self.staging_bounds is not None:
            sb = self.staging_bounds
            for pid in part_ids:
                sx = self.rng.uniform(sb.xy_min[0], sb.xy_max[0])
                sy_ = self.rng.uniform(sb.xy_min[1], sb.xy_max[1])
                sz = self.rng.uniform(sb.z_range[0], sb.z_range[1])
                # Keep the original orientation if available
                _, orig_rot = base_staging.get(
                    pid, (np.zeros(3), np.eye(3)))
                staging[pid] = (np.array([sx, sy_, sz]), orig_rot.copy())
        else:
            # Keep base staging positions
            for pid, (pos, rotmat) in base_staging.items():
                staging[pid] = (pos.copy(), rotmat.copy())

        # Assembly station
        station_pos = base_station_pos.copy()
        station_rotmat = base_station_rotmat.copy()
        if self.assembly_bounds is not None:
            ab = self.assembly_bounds
            station_pos[0] = self.rng.uniform(ab.xy_min[0], ab.xy_max[0])
            station_pos[1] = self.rng.uniform(ab.xy_min[1], ab.xy_max[1])
            if ab.z_range[0] != ab.z_range[1]:
                station_pos[2] = self.rng.uniform(
                    ab.z_range[0], ab.z_range[1])

        return WorkspaceLayout(
            robot_base_pos=robot_pos,
            robot_base_rotmat=robot_rotmat,
            staging_positions=staging,
            assembly_station_pos=station_pos,
            assembly_station_rotmat=station_rotmat,
            name=f"random_{index:04d}",
        )
