#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed-layout L2 validator for reinforcement-learning layout decisions.

Purpose
-------
The original layout evaluator enumerates multiple rotation candidates for each
part and selects the best feasible one.  That behavior is appropriate for
random search, but not for reinforcement learning: the environment must
validate exactly the pose and XY position selected by the policy.

This module wraps the existing ``WeightedInitialLayoutSearcher`` and provides:

1. a fixed ``region_id``;
2. a fixed ``pose_id`` for every non-preassembled part;
3. a fixed XY position for every non-preassembled part;
4. the original L2 IK, common-grasp, collision, clearance and scoring logic.

The two known invalid assembly regions ``r2_c0`` and ``r2_c1`` are rejected.
The preferred middle regions are ``r1_c0``, ``r1_c1`` and ``r1_c2``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Local project module.  Put this file in the same directory as
# find_optimal_initial_layout_tower_strict_pycharm.py.  The explicit path
# insertion keeps both ``python file.py`` and ``python -m ...`` working.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import (
    find_optimal_initial_layout_tower_strict_pycharm as fol,
)


PREFERRED_REGION_IDS: Tuple[str, ...] = ("r1_c0", "r1_c1", "r1_c2")
FORBIDDEN_REGION_IDS = frozenset({"r2_c0", "r2_c1"})


@dataclass
class FixedL2Result:
    """Serializable result returned by :meth:`evaluate_fixed_layout`."""

    l2_pass: bool
    layout_score: float
    region_id: str
    region_rc: Tuple[int, int]
    assembly_center: List[float]
    fail_part: Optional[str]
    fail_reason: str
    fail_detail: Dict[str, int]
    part_xy: Dict[str, List[float]]
    part_pose_id: Dict[str, int]
    rot_name: Dict[str, str]
    pose_tag: Dict[str, str]
    z_offset: Dict[str, float]
    grasp_counts: Dict[str, int]
    arm_choice: Dict[str, str]
    per_part_manip: Dict[str, float]
    per_part_dist: Dict[str, float]
    per_part_rot_angle: Dict[str, float]
    score_terms: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FixedL2Validator:
    """Validate an RL-selected fixed layout with the existing L2 evaluator."""

    def __init__(
        self,
        searcher: fol.WeightedInitialLayoutSearcher,
        *,
        forbidden_region_ids: Iterable[str] = FORBIDDEN_REGION_IDS,
        preferred_region_ids: Sequence[str] = PREFERRED_REGION_IDS,
    ) -> None:
        self.searcher = searcher
        self.forbidden_region_ids = frozenset(str(x) for x in forbidden_region_ids)
        self.preferred_region_ids = tuple(str(x) for x in preferred_region_ids)

    # ------------------------------------------------------------------
    # Region and pose-library inspection
    # ------------------------------------------------------------------

    def available_regions(self) -> List[Dict[str, Any]]:
        """Return the valid assembly regions in preferred-first order.

        ``_assembly_region_candidates`` already checks whether the preassembled
        first part collides with the robot.  We additionally reject the two
        region IDs explicitly fixed by the project definition.
        """
        raw = self.searcher._assembly_region_candidates()
        rows: List[Dict[str, Any]] = []
        for region_id, rc, center in raw:
            region_id = str(region_id)
            if region_id in self.forbidden_region_ids:
                continue
            rows.append(
                {
                    "region_id": region_id,
                    "rc": [int(rc[0]), int(rc[1])],
                    "center": np.asarray(center, dtype=float).tolist(),
                    "preferred": region_id in self.preferred_region_ids,
                }
            )

        rank = {rid: i for i, rid in enumerate(self.preferred_region_ids)}
        rows.sort(
            key=lambda row: (
                0 if row["preferred"] else 1,
                rank.get(row["region_id"], 10_000),
                row["region_id"],
            )
        )
        return rows

    def pose_library(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the pose IDs that the RL policy is allowed to select."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        first_pid = self.searcher._first_part_id()
        for pid in self.searcher.part_order:
            if self.searcher.preassemble_first_part and pid == first_pid:
                continue
            entries: List[Dict[str, Any]] = []
            for pose_id, cand in enumerate(self.searcher.rot_cands.get(pid, [])):
                entries.append(
                    {
                        "pose_id": int(pose_id),
                        "rot_name": str(cand.rot_name),
                        "pose_tag": str(cand.tag),
                        "rotmat": np.asarray(cand.rotmat, dtype=float).tolist(),
                        "extent": np.asarray(cand.extent, dtype=float).tolist(),
                        "footprint": np.asarray(cand.footprint, dtype=float).tolist(),
                        "z_offset": float(cand.z_offset),
                    }
                )
            out[pid] = entries
        return out

    def task_description(self) -> Dict[str, Any]:
        first_pid = self.searcher._first_part_id()
        return {
            "part_order": list(self.searcher.part_order),
            "preassembled_part": first_pid if self.searcher.preassemble_first_part else None,
            "decision_parts": [
                pid
                for pid in self.searcher.part_order
                if not (self.searcher.preassemble_first_part and pid == first_pid)
            ],
            "preferred_regions": list(self.preferred_region_ids),
            "forbidden_regions": sorted(self.forbidden_region_ids),
            "available_regions": self.available_regions(),
            "table_x_range": [float(x) for x in self.searcher.table_x_range],
            "table_y_range": [float(y) for y in self.searcher.table_y_range],
            "table_top_z": float(self.searcher.table_top_z),
        }

    # ------------------------------------------------------------------
    # Fixed L2 evaluation
    # ------------------------------------------------------------------

    def evaluate_fixed_layout(
        self,
        *,
        region_id: str,
        part_xy: Mapping[str, Sequence[float]],
        part_pose_id: Mapping[str, int],
    ) -> FixedL2Result:
        """Evaluate exactly the positions and poses selected by a policy.

        Parameters
        ----------
        region_id:
            One of the seven valid 3x3 assembly regions.  ``r2_c0`` and
            ``r2_c1`` are always rejected.
        part_xy:
            XY position in metres for every non-preassembled part.
        part_pose_id:
            Index into ``searcher.rot_cands[part_id]`` for every
            non-preassembled part.
        """
        region_id = str(region_id)
        region = self._resolve_region(region_id)
        region_name, region_rc, region_center = region

        decision_parts = self._decision_parts()
        xy_clean = self._validate_xy_mapping(part_xy, decision_parts)
        pose_clean = self._validate_pose_mapping(part_pose_id, decision_parts)

        # Preserve the complete pose library.  During the call each decision
        # part is restricted to a singleton list, so evaluate_layout cannot
        # silently change the policy-selected pose.
        original_rot_cands = self.searcher.rot_cands
        fixed_rot_cands = dict(original_rot_cands)
        selected_cands: Dict[str, Any] = {}
        for pid in decision_parts:
            selected = original_rot_cands[pid][pose_clean[pid]]
            selected_cands[pid] = selected
            fixed_rot_cands[pid] = [selected]

        # Check the table boundary using the selected footprint.  The original
        # sampler uses candidate 0 bounds, whereas an RL action may choose a
        # larger rotated footprint.
        for pid in decision_parts:
            self._validate_selected_pose_inside_table(
                pid=pid,
                xy=xy_clean[pid],
                cand=selected_cands[pid],
            )

        self.searcher._set_assembly_station(
            np.asarray(region_center, dtype=float),
            region_id=region_name,
            rc=tuple(region_rc),
        )

        layout = fol.LayoutCandidate(
            xy={pid: np.asarray(xy_clean[pid], dtype=float).copy() for pid in decision_parts}
        )

        # The first preassembled part is populated by evaluate_layout itself.
        self.searcher.rot_cands = fixed_rot_cands
        try:
            passed = bool(self.searcher.evaluate_layout(layout))
        finally:
            self.searcher.rot_cands = original_rot_cands

        # Restore the original policy pose IDs in the result.  Inside
        # evaluate_layout each fixed singleton has local index zero, but the RL
        # environment works with the IDs of the complete pose library.
        result_xy: Dict[str, List[float]] = {
            pid: np.asarray(xy, dtype=float).tolist() for pid, xy in layout.xy.items()
        }

        return FixedL2Result(
            l2_pass=passed,
            layout_score=float(layout.layout_score) if math.isfinite(float(layout.layout_score)) else -1.0,
            region_id=str(layout.assembly_region_id),
            region_rc=(int(layout.assembly_region_rc[0]), int(layout.assembly_region_rc[1])),
            assembly_center=np.asarray(layout.assembly_station_pos, dtype=float).tolist(),
            fail_part=None if layout.fail_part is None else str(layout.fail_part),
            fail_reason=str(layout.fail_reason or ""),
            fail_detail={str(k): int(v) for k, v in (layout.fail_detail or {}).items()},
            part_xy=result_xy,
            part_pose_id={pid: int(pose_clean[pid]) for pid in decision_parts},
            rot_name={str(k): str(v) for k, v in layout.rot_name.items()},
            pose_tag={str(k): str(v) for k, v in layout.pose_tag.items()},
            z_offset={str(k): float(v) for k, v in layout.z_offset.items()},
            grasp_counts={str(k): int(v) for k, v in layout.grasp_counts.items()},
            arm_choice={str(k): str(v) for k, v in layout.arm_choice.items()},
            per_part_manip={str(k): float(v) for k, v in layout.per_part_manip.items()},
            per_part_dist={str(k): float(v) for k, v in layout.per_part_dist.items()},
            per_part_rot_angle={str(k): float(v) for k, v in layout.per_part_rot_angle.items()},
            score_terms={
                "grasp": float(layout.grasp_score_norm),
                "manip": float(layout.manip_score_norm),
                "distance": float(layout.dist_score_norm),
                "rotation": float(layout.rot_score_norm),
                "spatial": float(layout.spatial_score_norm),
            },
        )

    def _decision_parts(self) -> List[str]:
        first_pid = self.searcher._first_part_id()
        return [
            pid
            for pid in self.searcher.part_order
            if not (self.searcher.preassemble_first_part and pid == first_pid)
        ]

    def _resolve_region(
        self, region_id: str
    ) -> Tuple[str, Tuple[int, int], np.ndarray]:
        if region_id in self.forbidden_region_ids:
            raise ValueError(
                f"assembly region {region_id!r} is forbidden because the "
                "preassembled base_plate collides with the robot base/link."
            )

        for rid, rc, center in self.searcher._assembly_region_candidates():
            if str(rid) == region_id and str(rid) not in self.forbidden_region_ids:
                return str(rid), (int(rc[0]), int(rc[1])), np.asarray(center, dtype=float)

        valid = [r["region_id"] for r in self.available_regions()]
        raise ValueError(f"unknown or invalid region_id={region_id!r}; valid regions={valid}")

    @staticmethod
    def _validate_xy_mapping(
        part_xy: Mapping[str, Sequence[float]], decision_parts: Sequence[str]
    ) -> Dict[str, np.ndarray]:
        missing = [pid for pid in decision_parts if pid not in part_xy]
        extra = sorted(set(part_xy) - set(decision_parts))
        if missing:
            raise ValueError(f"part_xy is missing decision parts: {missing}")
        if extra:
            raise ValueError(
                f"part_xy contains unknown/non-decision parts: {extra}. "
                "Do not provide base_plate when it is preassembled."
            )

        out: Dict[str, np.ndarray] = {}
        for pid in decision_parts:
            xy = np.asarray(part_xy[pid], dtype=float)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                raise ValueError(f"part_xy[{pid!r}] must be two finite values, got {part_xy[pid]!r}")
            out[pid] = xy
        return out

    def _validate_pose_mapping(
        self, part_pose_id: Mapping[str, int], decision_parts: Sequence[str]
    ) -> Dict[str, int]:
        missing = [pid for pid in decision_parts if pid not in part_pose_id]
        extra = sorted(set(part_pose_id) - set(decision_parts))
        if missing:
            raise ValueError(f"part_pose_id is missing decision parts: {missing}")
        if extra:
            raise ValueError(
                f"part_pose_id contains unknown/non-decision parts: {extra}. "
                "Do not provide base_plate when it is preassembled."
            )

        out: Dict[str, int] = {}
        for pid in decision_parts:
            try:
                pose_id = int(part_pose_id[pid])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"part_pose_id[{pid!r}] must be an integer") from exc
            n = len(self.searcher.rot_cands.get(pid, []))
            if pose_id < 0 or pose_id >= n:
                raise ValueError(
                    f"part_pose_id[{pid!r}]={pose_id} is out of range; valid 0..{n - 1}"
                )
            out[pid] = pose_id
        return out

    def _validate_selected_pose_inside_table(self, *, pid: str, xy: np.ndarray, cand: Any) -> None:
        (xlo, xhi), (ylo, yhi) = self.searcher._xy_bounds_for_part_and_cand(pid, cand)
        x, y = float(xy[0]), float(xy[1])
        eps = 1e-9
        if x < xlo - eps or x > xhi + eps or y < ylo - eps or y > yhi + eps:
            raise ValueError(
                f"{pid} selected pose is outside the usable table boundary: "
                f"xy=({x:.4f}, {y:.4f}), "
                f"allowed_x=[{xlo:.4f}, {xhi:.4f}], "
                f"allowed_y=[{ylo:.4f}, {yhi:.4f}], "
                f"footprint={np.asarray(cand.footprint, dtype=float).round(4).tolist()}"
            )


# ----------------------------------------------------------------------
# Searcher construction
# ----------------------------------------------------------------------


def build_default_searcher(
    *,
    asmdef_path: str,
    config_yaml: str,
    grasp_dir: str,
    cdprim_type: Optional[str] = None,
    planner_obstacle_mode: str = "staging_aware",
    max_rot_candidates: int = 12,
) -> fol.WeightedInitialLayoutSearcher:
    """Build the current strict L2 evaluator without starting random search."""
    cdprim = cdprim_type or getattr(fol, "DEFAULT_CDPRIM_TYPE", "triangles")

    return fol.WeightedInitialLayoutSearcher(
        asmdef_path=os.path.abspath(asmdef_path),
        config_yaml=os.path.abspath(config_yaml),
        grasp_dir=os.path.abspath(grasp_dir),
        fixture_pos=np.asarray([0.23, -0.35, 0.0], dtype=float),
        fixture_rotmat=np.eye(3),
        robot_base_pos=np.asarray([0.0, 0.0, 0.0], dtype=float),
        robot_base_rotmat=np.eye(3),
        part_order=None,  # Strictly use the asmdef step order.
        output_name="rl_fixed_l2",
        table_name="work_table",
        table_margin=float(getattr(fol, "DEFAULT_TABLE_MARGIN", 0.06)),
        table_clearance=float(getattr(fol, "DEFAULT_TABLE_CLEARANCE", 0.003)),
        grasp_map={},
        max_rot_candidates=int(max_rot_candidates),
        w_grasp=float(getattr(fol, "DEFAULT_W_GRASP", 0.3)),
        w_manip=float(getattr(fol, "DEFAULT_W_MANIP", 0.4)),
        w_dist=float(getattr(fol, "DEFAULT_W_DIST", 0.1)),
        w_rot=float(getattr(fol, "DEFAULT_W_ROT", 0.2)),
        ignore_env=False,
        cdprim_type=str(cdprim),
        planner_obstacle_mode=str(planner_obstacle_mode),
        plan_assembly_region=True,
        assembly_grid=3,
        preassemble_first_part=True,
        filter_assembly_near_arms=False,
        assembly_arm_x_clearance=float(getattr(fol, "DEFAULT_ASSEMBLY_ARM_X_CLEARANCE", 0.22)),
        assembly_arm_y_clearance=float(getattr(fol, "DEFAULT_ASSEMBLY_ARM_Y_CLEARANCE", 0.22)),
        filter_staging_near_arms=True,
        staging_arm_x_clearance=float(getattr(fol, "DEFAULT_STAGING_ARM_X_CLEARANCE", 0.12)),
        staging_arm_y_clearance=float(getattr(fol, "DEFAULT_STAGING_ARM_Y_CLEARANCE", 0.12)),
        goal_y_side_biased_sampling=False,  # Validation does not sample positions.
        goal_y_side_bias_ratio=float(getattr(fol, "DEFAULT_GOAL_Y_SIDE_BIAS_RATIO", 0.75)),
        goal_y_side_eps=float(getattr(fol, "DEFAULT_GOAL_Y_SIDE_EPS", 0.005)),
        use_flatsurface=True,
        fs_stability_threshold=float(getattr(fol, "DEFAULT_FS_STABILITY_THRESHOLD", 0.10)),
        check_robot_home_collision=True,
        strict_initial_robot_collision=True,
        min_staging_mesh_clearance=float(getattr(fol, "DEFAULT_MIN_STAGING_MESH_CLEARANCE", 0.01)),
        enforce_order_x_constraint=bool(getattr(fol, "DEFAULT_ENFORCE_ORDER_X_CONSTRAINT", True)),
        order_x_tolerance=float(getattr(fol, "DEFAULT_ORDER_X_TOLERANCE", 0.03)),
        enable_y_side_distribution_score=bool(
            getattr(fol, "DEFAULT_ENABLE_Y_SIDE_DISTRIBUTION_SCORE", True)
        ),
        prefer_upright_when_topdown_low=bool(
            getattr(fol, "DEFAULT_PREFER_UPRIGHT_WHEN_TOPDOWN_LOW", True)
        ),
        topdown_min_count=int(getattr(fol, "DEFAULT_TOPDOWN_MIN_COUNT", 10)),
        topdown_align_cos=float(getattr(fol, "DEFAULT_TOPDOWN_ALIGN_COS", 0.82)),
        check_l2_pick_quick_motion=False,
        l2_pick_check_parts=[],
        l2_pick_check_lift_dist=float(getattr(fol, "DEFAULT_L2_PICK_CHECK_LIFT_DIST", 0.06)),
        l2_pick_check_directions=["z", "x_plus", "x_minus", "y_plus", "y_minus"],
        l2_pick_check_tilt=float(getattr(fol, "DEFAULT_L2_PICK_CHECK_TILT", 0.35)),
        robot_home_clearance=float(getattr(fol, "DEFAULT_ROBOT_HOME_CLEARANCE", 0.03)),
        l3_skip_parts=[],
        prefer_stl_upface=bool(getattr(fol, "DEFAULT_PREFER_STL_UPFACE", True)),
        w_stl_upface=float(getattr(fol, "DEFAULT_W_STL_UPFACE", 0.35)),
        stl_upface_min_thinness=float(
            getattr(fol, "DEFAULT_STL_UPFACE_MIN_THINNESS", 0.20)
        ),
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _load_request(path: str) -> Tuple[str, Dict[str, Sequence[float]], Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    region_id = str(data["region_id"])
    if "parts" in data:
        parts = data["parts"]
        part_xy = {pid: spec["xy"] for pid, spec in parts.items()}
        part_pose_id = {pid: int(spec["pose_id"]) for pid, spec in parts.items()}
    else:
        part_xy = data["part_xy"]
        part_pose_id = data["part_pose_id"]
    return region_id, part_xy, part_pose_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a fixed RL-selected layout with L2.")
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grasp-dir", required=True)
    parser.add_argument("--input-json", help="Fixed layout request JSON.")
    parser.add_argument("--output-json", help="Optional result JSON path.")
    parser.add_argument("--cdprim-type", default=None)
    parser.add_argument(
        "--planner-obstacle-mode",
        default="staging_aware",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
    )
    parser.add_argument("--max-rot-candidates", type=int, default=12)
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print valid regions and pose IDs, then exit unless --input-json is also supplied.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    searcher = build_default_searcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        max_rot_candidates=args.max_rot_candidates,
    )
    validator = FixedL2Validator(searcher)

    if args.describe:
        description = validator.task_description()
        description["pose_library"] = validator.pose_library()
        print(json.dumps(description, ensure_ascii=False, indent=2))
        if not args.input_json:
            return

    if not args.input_json:
        raise SystemExit("Provide --input-json, or use --describe to inspect the task.")

    region_id, part_xy, part_pose_id = _load_request(args.input_json)
    result = validator.evaluate_fixed_layout(
        region_id=region_id,
        part_xy=part_xy,
        part_pose_id=part_pose_id,
    )
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] result saved to: {output_path}")


if __name__ == "__main__":
    main()
