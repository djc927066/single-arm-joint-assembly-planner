"""
SEALP Layout Optimization
===========================

Workspace layout representation, feasibility evaluation, and
optimization for sequence-aware dual-arm assembly planning.

Modules
-------
layout
    ``WorkspaceLayout`` dataclass and YAML serialization.
reachability
    IK reachability checks for target poses.
manipulability
    Yoshikawa manipulability scoring.
feasibility
    Per-step feasibility evaluation and ``FeasibilityReport``.
metrics
    Standardized layout evaluation metrics.
optimizer
    Abstract optimizer interface.
optimizer_random
    Random search baseline optimizer.

Visualization (run as modules)
------------------------------
show_stable_poses
    Stable STL placement poses (SPACE to cycle).
show_stl
    Display a specified STL mesh (SPACE to yaw 90°).
show_table_grid
    Colored work_table 3x3 assembly-region grid.
show_dual_arms
    Work table + dual Panthera arms at home pose.
"""

from .layout import WorkspaceLayout
from .feasibility import FeasibilityReport, StepFeasibility, evaluate_layout
from .metrics import LayoutMetrics, compute_metrics
from .dual_staging_search import (
    search_dual_feasible_layout,
    generate_staging_candidates,
    make_grid_zone_from_box,
    find_obstacle_def,
)
