#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""General deterministic accelerated Global-to-Local layout search V3.1.

This is a NEW script. It does not overwrite or modify:
    find_optimal_initial_layout_tower_global.py

The original Global search remains the main algorithm:
    global exploration -> top-K elites -> multiscale pattern refinement.

Only the following acceleration/reliability mechanisms are added:
1. Legacy anchor replay:
   With seed=0, replay the first N candidates of the original Global sampler.
   This is used only to recover and checkpoint the previously observed 0.5286
   candidate, because the crashed run did not save its XY coordinates.
2. Deterministic Halton low-discrepancy layout sampling:
   After the legacy anchors, all new layouts use a non-random, reproducible,
   per-region/per-part low-discrepancy sequence.
3. Two-stage grasp evaluation:
   Search uses a smaller deterministic pose-space grasp subset; promising
   candidates are re-evaluated with the original cap=350 before comparison.
4. Progressive common-grasp filtering and bounded pose cache.
5. Crash-safe best-so-far checkpoint:
   Every new authoritative best (evaluated at cap=350) is immediately written
   to JSON with full XY coordinates and assembly-region information.
6. Incumbent protection:
   A valid checkpoint is loaded, re-evaluated, and included as an elite.
   The returned result can never be worse than the loaded authoritative
   incumbent under the same cap=350 scoring criterion.

Important terminology:
- Halton sampling is deterministic low-discrepancy coverage, not random.
- "Uniform" means space-filling / low-discrepancy coverage; collision rejection
  can make the accepted layouts non-perfectly-uniform.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod
import uniform_grasp_accel_v31 as uniform_accel

LayoutCandidate = fol.LayoutCandidate
Region = Tuple[str, Tuple[int, int], np.ndarray]


ACFG: Dict[str, object] = {
    # Number of total Phase-A strict layout evaluations.
    "explore": None,

    # Replay the first 28 original Global candidates when no checkpoint exists.
    # The previous run found 0.5286 at original candidate #22.
    "legacy_replay": 0,
    "legacy_seed": 0,

    # Fast search and authoritative comparison grasp caps.
    "search_grasp_cap": 160,
    "authoritative_grasp_cap": 350,
    "grasp_mode": "same_grasp_progressive",
    "fps_seed_mode": "medoid",

    # How many approximate candidates are authoritatively re-scored before refine.
    "rescore_topk": 8,

    # Optional full-grasp diagnostic. 0 disables it.
    "full_certify_topk": 0,

    # Joint layout Halton settings. Each movable part receives
    # (u_x, u_y, u_rotation_priority), so one full layout is 3N-D.
    "halton_max_layout_attempts": 320,
    "halton_repair_attempts_per_part": 24,
    "coverage_bins": 4,
    "halton_prefilter_mesh_clearance": True,
    "halton_prefilter_home_collision": True,

    # Optional existing .layout file.
    # auto:
    #   exact part-set match  -> exact warm start
    #   partial part overlap  -> retain valid matched parts and deterministically
    #                            insert missing/new parts
    #   no overlap            -> ignore layout and start from Halton Global
    "warm_layout": None,
    "warm_mode": "auto",
    "warm_start_refine_first": True,

    # Crash-safe checkpoint.
    "checkpoint_json": None,
    "resume_checkpoint": True,

    # Search-stage candidate above current authoritative best triggers
    # an immediate cap=350 verification/checkpoint.
    "online_verify_margin": 0.0,
}


def _arg_value(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        i = sys.argv.index(name)
    except ValueError:
        return default
    if i + 1 >= len(sys.argv):
        return default
    return str(sys.argv[i + 1])


def _default_checkpoint_path() -> str:
    output_name = _arg_value("--output-name", "tower_global_accel")
    output_dir = _arg_value(
        "--output-dir",
        os.path.join(str(getattr(fol, "SEALP_ROOT", _THIS_DIR)), "examples", "layout", "_output"),
    )
    return os.path.abspath(
        os.path.join(str(output_dir), f"{output_name}_best_so_far.json")
    )


def _radical_inverse(index: int, base: int) -> float:
    i = max(1, int(index))
    inv = 1.0 / float(base)
    factor = inv
    value = 0.0
    while i > 0:
        value += factor * float(i % base)
        i //= base
        factor *= inv
    return float(value)


def _first_primes(count: int) -> List[int]:
    primes: List[int] = []
    candidate = 2
    while len(primes) < int(count):
        is_prime = True
        limit = int(math.sqrt(candidate))
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


class JointHaltonPoseStream:
    """One deterministic 3N-D point per complete N-part layout.

    Per part:
        u_x  -> normalized X position
        u_y  -> normalized Y position
        u_r  -> deterministic priority among discrete rotation candidates

    u_r does not lock the final rotation. It only chooses which stable
    rotation is tested first during cheap geometric prefiltering. All rotation
    candidates remain available to the strict evaluator.
    """

    def __init__(self) -> None:
        self._region_counter: Dict[str, int] = defaultdict(int)
        self._prime_cache: Dict[int, List[int]] = {}

    @staticmethod
    def _stable_region_offset(region_id: str) -> int:
        data = str(region_id).encode("utf-8")
        acc = 0
        for b in data:
            acc = (acc * 131 + int(b)) % 104729
        return 1 + acc

    def _primes(self, dimensions: int) -> List[int]:
        cached = self._prime_cache.get(int(dimensions))
        if cached is None:
            cached = _first_primes(int(dimensions))
            self._prime_cache[int(dimensions)] = cached
        return cached

    def next_layout(
        self,
        region_id: str,
        ordered_part_ids: Sequence[str],
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        pids = [str(pid) for pid in ordered_part_ids]
        dimensions = 3 * len(pids)
        if dimensions <= 0:
            return 0, {}

        rid = str(region_id)
        self._region_counter[rid] += 1
        local_index = self._region_counter[rid]
        index = self._stable_region_offset(rid) + local_index
        bases = self._primes(dimensions)

        out: Dict[str, np.ndarray] = {}
        for i, pid in enumerate(pids):
            out[pid] = np.array(
                [
                    _radical_inverse(index, bases[3 * i]),
                    _radical_inverse(index, bases[3 * i + 1]),
                    _radical_inverse(index, bases[3 * i + 2]),
                ],
                dtype=float,
            )
        return index, out

    def repair_pose(
        self,
        region_id: str,
        layout_index: int,
        part_id: str,
        part_rank: int,
        retry: int,
    ) -> np.ndarray:
        data = f"{region_id}|{part_id}".encode("utf-8")
        acc = 0
        for b in data:
            acc = (acc * 137 + int(b)) % 99991

        derived = (
            int(layout_index) * 4099
            + (int(part_rank) + 1) * 257
            + (int(retry) + 1) * 17
            + acc
            + 1
        )
        return np.array(
            [
                _radical_inverse(derived, 2),
                _radical_inverse(derived, 3),
                _radical_inverse(derived, 5),
            ],
            dtype=float,
        )

class GlobalAccelSearcher(gmod.GlobalLayoutSearcher):
    """Original Global search with deterministic coverage and acceleration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._halton = JointHaltonPoseStream()
        self._authoritative_best: Optional[LayoutCandidate] = None
        self._authoritative_best_stage: str = ""
        self._coverage_cells: Dict[Tuple[str, str], set] = defaultdict(set)
        self._phase_a_evaluated = 0

    def _max_eval_limit(self) -> Optional[int]:
        raw = nsga2._CFG.get("max_evals")
        if raw is None:
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _remaining_eval_budget(self) -> Optional[int]:
        limit = self._max_eval_limit()
        if limit is None:
            return None
        return max(0, limit - int(self._nsga_eval_count))

    def _eval_budget_exhausted(self) -> bool:
        """Strict budget boundary: count == limit is already exhausted."""
        remaining = self._remaining_eval_budget()
        return remaining is not None and remaining <= 0

    @staticmethod
    def _same_xy_layout(
        a: LayoutCandidate,
        b: LayoutCandidate,
        atol: float = 1e-10,
    ) -> bool:
        if str(a.assembly_region_id) != str(b.assembly_region_id):
            return False
        a_xy = dict(getattr(a, "xy", {}) or {})
        b_xy = dict(getattr(b, "xy", {}) or {})
        if set(a_xy) != set(b_xy):
            return False
        return all(
            np.allclose(
                np.asarray(a_xy[pid], dtype=float),
                np.asarray(b_xy[pid], dtype=float),
                rtol=0.0,
                atol=atol,
            )
            for pid in a_xy
        )

    # ------------------------------------------------------------------
    # Crash-safe checkpoint
    # ------------------------------------------------------------------
    @property
    def _checkpoint_path(self) -> str:
        path = ACFG.get("checkpoint_json")
        return os.path.abspath(str(path or _default_checkpoint_path()))

    @staticmethod
    def _candidate_record(
        cand: LayoutCandidate,
        stage: str,
        cap: int,
        eval_count: int,
    ) -> Dict:
        xy = {
            str(pid): np.asarray(pos, dtype=float).reshape(2).tolist()
            for pid, pos in cand.xy.items()
        }
        station = getattr(cand, "assembly_station_pos", None)
        record = {
            "version": 1,
            "stage": str(stage),
            "authoritative_grasp_cap": int(cap),
            "score": float(cand.layout_score),
            "region_id": str(cand.assembly_region_id),
            "region_rc": list(cand.assembly_region_rc),
            "assembly_station_pos": (
                np.asarray(station, dtype=float).reshape(-1).tolist()
                if station is not None else None
            ),
            "xy": xy,
            "eval_count": int(eval_count),
            "saved_at_unix": float(time.time()),
            "grasp_counts": dict(getattr(cand, "grasp_counts", {}) or {}),
            "arm_choice": dict(getattr(cand, "arm_choice", {}) or {}),
            "pose_tag": dict(getattr(cand, "pose_tag", {}) or {}),
        }
        return record

    def _write_checkpoint(
        self,
        cand: LayoutCandidate,
        stage: str,
        cap: int,
    ) -> None:
        path = self._checkpoint_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = self._candidate_record(
            cand,
            stage=stage,
            cap=cap,
            eval_count=self._nsga_eval_count,
        )
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        print(
            f"[checkpoint] score={cand.layout_score:.4f} "
            f"stage={stage} cap={cap} -> {path}"
        )

    def _promote_authoritative(
        self,
        cand: Optional[LayoutCandidate],
        stage: str,
    ) -> None:
        if cand is None or not bool(getattr(cand, "l2_pass", False)):
            return
        if (
            self._authoritative_best is None
            or float(cand.layout_score)
            > float(self._authoritative_best.layout_score) + 1e-6
        ):
            self._authoritative_best = cand
            self._authoritative_best_stage = str(stage)
            self._write_checkpoint(
                cand,
                stage=stage,
                cap=int(ACFG["authoritative_grasp_cap"]),
            )

    def _load_checkpoint_record(self) -> Optional[Mapping]:
        if not bool(ACFG.get("resume_checkpoint", True)):
            return None
        path = self._checkpoint_path
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if int(rec.get("authoritative_grasp_cap", -1)) != int(
                ACFG["authoritative_grasp_cap"]
            ):
                print(
                    f"[checkpoint] ignored: cap={rec.get('authoritative_grasp_cap')} "
                    f"!= required {ACFG['authoritative_grasp_cap']}"
                )
                return None
            return rec
        except Exception as e:
            print(f"[checkpoint] load failed: {type(e).__name__}: {e}")
            return None


    def _load_warm_layout_record(self) -> Optional[Mapping]:
        path = ACFG.get("warm_layout")
        if path is None:
            return None
        path = os.path.abspath(str(path))
        if not os.path.isfile(path):
            print(f"[warm-start] layout not found: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)

            staging = dict(doc.get("staging", {}) or {})
            metadata = dict(doc.get("metadata", {}) or {})
            assembly_station = dict(doc.get("assembly_station", {}) or {})

            xy = {}
            for pid, item in staging.items():
                pos = np.asarray(item.get("pos", []), dtype=float).reshape(-1)
                if pos.size >= 2:
                    xy[str(pid)] = pos[:2].tolist()

            if not xy:
                raise ValueError("no staging XY coordinates found")

            station_pos = assembly_station.get(
                "pos",
                metadata.get("assembly_station_pos"),
            )
            rc = metadata.get("assembly_region_rc", [-1, -1])
            rid = metadata.get("assembly_region_id", "warm_layout")

            old_components = dict(metadata.get("score_components", {}) or {})
            old_weights = dict(metadata.get("weights", {}) or {})
            old_score = metadata.get("score")
            approx_current = None
            required = {"grasp", "manip", "dist", "rot"}
            if required.issubset(old_components):
                approx_current = (
                    0.25 * float(old_components["grasp"])
                    + 0.50 * float(old_components["manip"])
                    + 0.05 * float(old_components["dist"])
                    + 0.20 * float(old_components["rot"])
                )

            print("\n---------- Existing layout warm start ----------")
            print(f"path                    = {path}")
            print(f"stored region           = {rid}")
            print(f"stored score            = {old_score}")
            print(f"stored weights          = {old_weights}")
            print(f"stored l3_pass metadata = {metadata.get('l3_pass')}")
            if approx_current is not None:
                print(
                    f"approx score with current weights = "
                    f"{approx_current:.4f} "
                    "(from stored components; strict re-evaluation follows)"
                )

            return {
                "region_id": str(rid),
                "region_rc": list(rc),
                "assembly_station_pos": station_pos,
                "xy": xy,
                "source_path": path,
                "stored_score": old_score,
                "stored_weights": old_weights,
                "stored_l3_pass": metadata.get("l3_pass"),
                "pose_tag": dict(metadata.get("pose_tag", {}) or {}),
                "arm_choice": dict(metadata.get("arm_choice", {}) or {}),
            }
        except Exception as e:
            print(
                f"[warm-start] failed to parse {path}: "
                f"{type(e).__name__}: {e}"
            )
            return None

    def _region_from_record(
        self,
        rec: Mapping,
        regions: Sequence[Region],
    ) -> Region:
        rid = str(rec.get("region_id", ""))
        for region in regions:
            if str(region[0]) == rid:
                return region
        station = rec.get("assembly_station_pos")
        if station is not None:
            rc_raw = rec.get("region_rc", [-1, -1])
            rc = (int(rc_raw[0]), int(rc_raw[1]))
            return (
                rid or "checkpoint",
                rc,
                np.asarray(station, dtype=float),
            )
        return regions[0]

    # ------------------------------------------------------------------
    # Fresh evaluation bypassing the rounded NSGA cache.
    # Required when the grasp cap changes.
    # ------------------------------------------------------------------
    def _fresh_evaluate(
        self,
        xy: Mapping[str, np.ndarray],
        region: Region,
        cap: int,
    ) -> Optional[LayoutCandidate]:
        if self._eval_budget_exhausted():
            return None

        old_cap = int(fast.MAX_GRASPS_PER_POSE)
        fast.MAX_GRASPS_PER_POSE = int(cap)
        try:
            self._set_region_from_tuple(region)
            cand = LayoutCandidate(xy=nsga2._copy_xy(dict(xy)))
            self._nsga_eval_count += 1
            ok = bool(self.evaluate_layout(cand))
            cand.l2_pass = ok
            cand._evaluated_grasp_cap = int(cap)
            cand._nsga_objectives = self._objectives_from_candidate(cand)
            cand._nsga_region_tuple = region
            self._note_eval_outcome(cand)
            return cand
        finally:
            fast.MAX_GRASPS_PER_POSE = old_cap

    # ------------------------------------------------------------------
    # Deterministic low-discrepancy collision-free layout generation.
    # ------------------------------------------------------------------
    def _note_coverage(
        self,
        region: Region,
        pid: str,
        p: np.ndarray,
        xlo: float,
        xhi: float,
        ylo: float,
        yhi: float,
    ) -> None:
        bins = max(1, int(ACFG.get("coverage_bins", 4)))
        ux = 0.0 if xhi <= xlo else (float(p[0]) - xlo) / (xhi - xlo)
        uy = 0.0 if yhi <= ylo else (float(p[1]) - ylo) / (yhi - ylo)
        bx = min(bins - 1, max(0, int(np.floor(ux * bins))))
        by = min(bins - 1, max(0, int(np.floor(uy * bins))))
        self._coverage_cells[(str(region[0]), str(pid))].add((bx, by))


    def _try_place_pose(
        self,
        pid: str,
        p: np.ndarray,
        rot_cand,
        placed: Sequence[str],
    ) -> bool:
        """Rotation-aware cheap validation before expensive IK/grasp checks."""
        (xlo, xhi), (ylo, yhi) = (
            self._xy_bounds_for_part_and_cand(pid, rot_cand)
        )
        p = np.asarray(p, dtype=float).reshape(2)

        # The bounds already account for the selected rotated footprint.
        if not (
            xlo <= float(p[0]) <= xhi
            and ylo <= float(p[1]) <= yhi
        ):
            return False

        if self._staging_arm_keepout_reason(pid, p, rot_cand):
            return False

        self._apply_staging_pose(pid, p, rot_cand)
        ab_pid = self._world_aabb_for_staging(pid)

        # AABB broad phase, then true mesh collision.
        for q in placed:
            if q not in self.staging_models:
                continue
            ab_q = self._world_aabb_for_staging(q)
            if (
                ab_pid is not None
                and ab_q is not None
                and not self._aabb_overlap(*ab_pid, *ab_q)
            ):
                continue
            if self.staging_models[pid].is_mcdwith(
                self.staging_models[q]
            ):
                return False

        if bool(ACFG.get("halton_prefilter_mesh_clearance", True)):
            try:
                reason = self._mesh_clearance_reason(
                    active_pids=list(placed) + [pid]
                )
                if reason:
                    return False
            except Exception:
                # Strict evaluate_layout will check it again.
                pass

        return True

    @staticmethod
    def _rotation_priority(
        rot_candidates: Sequence,
        u_rotation: float,
    ) -> List[int]:
        n = len(rot_candidates)
        if n <= 0:
            return []
        start = min(
            n - 1,
            max(0, int(math.floor(float(u_rotation) * n))),
        )
        return [(start + offset) % n for offset in range(n)]

    def _sample_collision_free_xy_halton(
        self,
        region: Region,
        fixed_xy: Optional[Mapping[str, np.ndarray]] = None,
        return_transfer_stats: bool = False,
    ):
        """Generate one complete deterministic rotation-aware layout.

        One candidate for N movable parts is represented by a 3N-D Halton point:
            (u_x1, u_y1, u_r1, ..., u_xN, u_yN, u_rN).

        For each proposed pose:
        1. u_r selects a deterministic first rotation candidate.
        2. All stable rotation candidates are tried in cyclic deterministic
           order, so a valid orientation is not lost merely because the first
           one fails.
        3. The XY point is mapped using that rotation's own footprint bounds.
        4. Boundary, arm keepout, mesh collision, and configured clearance are
           checked before any IK/grasp evaluation.

        The selected prefilter rotation is only a seed. evaluate_layout later
        enumerates all rotation candidates again and selects the feasible/best
        one under the full L2 criterion.
        """
        fixed_xy = {
            str(pid): np.asarray(pos, dtype=float).reshape(2)
            for pid, pos in dict(fixed_xy or {}).items()
        }

        xy: Dict[str, np.ndarray] = {}
        placed: List[str] = []
        retained: List[str] = []
        inserted: List[str] = []
        rejected_fixed: List[str] = []
        seed_rotation: Dict[str, Dict[str, object]] = {}
        pending_coverage: List[
            Tuple[str, np.ndarray, float, float, float, float]
        ] = []

        first_pid = (
            self._first_part_id()
            if self.preassemble_first_part
            else None
        )

        if first_pid is not None and first_pid in self.world_poses:
            gp, gr = self.world_poses[first_pid]
            xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            if first_pid in self.staging_models:
                self.staging_models[first_pid].pos = np.asarray(
                    gp,
                    dtype=float,
                ).copy()
                self.staging_models[first_pid].rotmat = np.asarray(
                    gr,
                    dtype=float,
                ).copy()
                self._update_world_cache_from_cm(first_pid)
            placed.append(first_pid)
            if first_pid in fixed_xy:
                retained.append(first_pid)

        sampled_parts = [
            str(pid)
            for pid in self.part_order
            if pid != first_pid
        ]
        layout_index, joint_u = self._halton.next_layout(
            str(region[0]),
            sampled_parts,
        )

        # Place larger possible footprints earlier to reduce repair failures.
        placement_order = sorted(
            sampled_parts,
            key=lambda pid: max(
                float(np.prod(cand.footprint))
                for cand in self.rot_cands[pid]
            ),
            reverse=True,
        )
        part_rank = {
            pid: rank for rank, pid in enumerate(sampled_parts)
        }
        repair_limit = max(
            0,
            int(ACFG["halton_repair_attempts_per_part"]),
        )

        for pid in placement_order:
            rot_candidates = list(self.rot_cands[pid])
            if not rot_candidates:
                return (None, {}) if return_transfer_stats else None

            base_u = joint_u[pid]
            fixed = fixed_xy.get(pid)

            pose_parameters: List[Tuple[Optional[np.ndarray], np.ndarray]] = []
            if fixed is not None:
                pose_parameters.append((fixed.copy(), base_u))
            else:
                pose_parameters.append((None, base_u))
                for retry in range(repair_limit):
                    pose_parameters.append(
                        (
                            None,
                            self._halton.repair_pose(
                                str(region[0]),
                                layout_index,
                                pid,
                                part_rank[pid],
                                retry,
                            ),
                        )
                    )

            accepted = False
            for fixed_point, u in pose_parameters:
                rot_order = self._rotation_priority(
                    rot_candidates,
                    float(u[2]),
                )

                for rot_idx in rot_order:
                    rot_cand = rot_candidates[rot_idx]
                    (xlo, xhi), (ylo, yhi) = (
                        self._xy_bounds_for_part_and_cand(
                            pid,
                            rot_cand,
                        )
                    )
                    if xlo >= xhi or ylo >= yhi:
                        continue

                    if fixed_point is not None:
                        p = fixed_point
                    else:
                        preferred_y_range, goal_side = (
                            self._preferred_y_range_by_goal_side(
                                pid,
                                ylo,
                                yhi,
                                first_pid=first_pid,
                            )
                        )
                        cur_ylo, cur_yhi = ylo, yhi
                        if goal_side in ("left", "right"):
                            pylo, pyhi = preferred_y_range
                            if pylo < pyhi:
                                cur_ylo, cur_yhi = pylo, pyhi

                        p = np.array(
                            [
                                xlo + float(u[0]) * (xhi - xlo),
                                cur_ylo
                                + float(u[1]) * (cur_yhi - cur_ylo),
                            ],
                            dtype=float,
                        )

                    if not self._try_place_pose(
                        pid,
                        p,
                        rot_cand,
                        placed,
                    ):
                        continue

                    xy[pid] = np.asarray(p, dtype=float).copy()
                    placed.append(pid)
                    seed_rotation[pid] = {
                        "index": int(rot_idx),
                        "tag": str(
                            getattr(rot_cand, "tag", "unknown")
                        ),
                        "rot_name": str(
                            getattr(rot_cand, "rot_name", "unknown")
                        ),
                    }
                    if fixed_point is not None:
                        retained.append(pid)
                    else:
                        inserted.append(pid)

                    pending_coverage.append(
                        (
                            str(pid),
                            np.asarray(p, dtype=float).copy(),
                            float(xlo),
                            float(xhi),
                            float(ylo),
                            float(yhi),
                        )
                    )
                    accepted = True
                    break

                if accepted:
                    break

            if not accepted:
                if fixed is not None:
                    rejected_fixed.append(pid)
                return (None, {}) if return_transfer_stats else None

        # Completed-layout cheap checks before expensive IK/grasp loops.
        try:
            if self._pairwise_collision():
                return (None, {}) if return_transfer_stats else None
        except Exception:
            pass

        if bool(ACFG.get("halton_prefilter_mesh_clearance", True)):
            try:
                if self._mesh_clearance_reason(
                    active_pids=self.part_order
                ):
                    return (None, {}) if return_transfer_stats else None
            except Exception:
                pass

        if bool(ACFG.get("halton_prefilter_home_collision", True)):
            try:
                if self._robot_home_collision_reason(
                    active_pids=self.part_order
                ):
                    return (None, {}) if return_transfer_stats else None
            except Exception:
                pass

        # Only complete layouts that pass all cheap prefilters contribute
        # to the coverage report. Partial placements from rejected layouts are
        # intentionally excluded.
        for (
            cov_pid,
            cov_p,
            cov_xlo,
            cov_xhi,
            cov_ylo,
            cov_yhi,
        ) in pending_coverage:
            self._note_coverage(
                region,
                cov_pid,
                cov_p,
                cov_xlo,
                cov_xhi,
                cov_ylo,
                cov_yhi,
            )

        result = {
            pid: xy[pid]
            for pid in self.part_order
            if pid in xy
        }
        stats = {
            "layout_halton_index": int(layout_index),
            "retained": retained,
            "inserted": inserted,
            "rejected_fixed": rejected_fixed,
            "seed_rotation": seed_rotation,
            "current_parts": list(self.part_order),
            "provided_parts": sorted(fixed_xy.keys()),
        }
        return (
            (result, stats)
            if return_transfer_stats
            else result
        )
    def _prepare_warm_seed(
        self,
        rec: Mapping,
        regions: Sequence[Region],
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Region], str, Dict]:
        """Classify and prepare exact, partial, or disabled warm start."""
        mode = str(ACFG.get("warm_mode", "auto")).strip().lower()
        if mode not in {"auto", "exact", "partial", "off"}:
            raise ValueError(
                "--global-accel-warm-mode must be auto/exact/partial/off"
            )
        if mode == "off":
            return None, None, "off", {}

        current = set(map(str, self.part_order))
        provided_xy = {
            str(pid): np.asarray(pos, dtype=float).reshape(2)
            for pid, pos in dict(rec.get("xy", {})).items()
        }
        provided = set(provided_xy)
        overlap = current & provided
        missing = current - provided
        extra = provided - current

        if current == provided:
            resolved = "exact"
        elif overlap:
            resolved = "partial"
        else:
            resolved = "none"

        if mode == "exact" and resolved != "exact":
            resolved = "none"
        elif mode == "partial" and not overlap:
            resolved = "none"

        print("\n---------- Warm-layout compatibility ----------")
        print(f"requested mode = {mode}")
        print(f"resolved mode  = {resolved}")
        print(f"current parts  = {sorted(current)}")
        print(f"provided parts = {sorted(provided)}")
        print(f"overlap        = {sorted(overlap)}")
        print(f"missing/new    = {sorted(missing)}")
        print(f"unused old     = {sorted(extra)}")

        if resolved == "none":
            print(
                "[warm-start] no transferable current part; "
                "using deterministic Halton Global from scratch"
            )
            return None, None, "none", {
                "overlap": sorted(overlap),
                "missing": sorted(missing),
                "extra": sorted(extra),
            }

        region = self._region_from_record(rec, regions)
        self._set_region_from_tuple(region)

        if resolved == "exact":
            return (
                {pid: provided_xy[pid] for pid in self.part_order},
                region,
                "exact",
                {
                    "retained": list(self.part_order),
                    "inserted": [],
                    "rejected_fixed": [],
                },
            )

        # Partial transfer: only matching current parts are offered as fixed
        # positions. The generator validates them under the current geometry
        # and inserts all missing/new parts deterministically.
        partial_fixed = {
            pid: provided_xy[pid]
            for pid in overlap
            if pid in self.part_order
        }
        completed, stats = self._sample_collision_free_xy_halton(
            region,
            fixed_xy=partial_fixed,
            return_transfer_stats=True,
        )
        if completed is None:
            print(
                "[warm-start] partial completion failed; "
                "falling back to deterministic Halton Global from scratch"
            )
            return None, None, "none", stats

        print(
            f"[warm-start] retained={stats.get('retained', [])}; "
            f"inserted={stats.get('inserted', [])}; "
            f"rejected_old_positions={stats.get('rejected_fixed', [])}"
        )
        return completed, region, "partial", stats

    # ------------------------------------------------------------------
    # Phase A1: replay original fixed-seed anchors to recover old best.
    # ------------------------------------------------------------------
    def _legacy_anchor_replay(
        self,
        regions: Sequence[Region],
        n_target: int,
        verbose: bool,
    ) -> List[LayoutCandidate]:
        out: List[LayoutCandidate] = []
        if n_target <= 0:
            return out

        rng = np.random.default_rng(int(ACFG["legacy_seed"]))
        evaluated = 0
        attempts = 0
        region_i = 0
        max_attempts = n_target * 4 + 20
        cap = int(ACFG["authoritative_grasp_cap"])
        old_cap = int(fast.MAX_GRASPS_PER_POSE)
        fast.MAX_GRASPS_PER_POSE = cap

        print(
            "\n---------- Phase A1: fixed legacy-anchor replay ----------"
        )
        print(
            f"target={n_target}, seed={ACFG['legacy_seed']}, cap={cap}; "
            "used only to recover the previous Global incumbent"
        )
        try:
            while (
                evaluated < n_target
                and attempts < max_attempts
                and not self._eval_budget_exhausted()
            ):
                attempts += 1
                region = regions[region_i % len(regions)]
                region_i += 1
                self._set_region_from_tuple(region)

                xy = None
                for _ in range(int(gmod.GCFG["max_resample_layout"])):
                    xy = self.sample_collision_free_xy(rng)
                    if xy is not None:
                        break
                if xy is None:
                    continue

                cand = self._fresh_evaluate(xy, region, cap=cap)
                if cand is None:
                    break
                evaluated += 1
                self._phase_a_evaluated += 1

                if verbose:
                    tag = "L2_OK" if cand.l2_pass else "FAIL"
                    print(
                        f"[legacy] {evaluated:03d}/{n_target} {tag:5s} "
                        f"score={float(getattr(cand, 'layout_score', -1.0)):.4f} "
                        f"region={cand.assembly_region_id}"
                    )
                if cand.l2_pass:
                    out.append(cand)
                    self._promote_authoritative(
                        cand,
                        stage=f"legacy_replay_{evaluated:03d}",
                    )
        finally:
            fast.MAX_GRASPS_PER_POSE = old_cap
        return out

    # ------------------------------------------------------------------
    # Phase A2: deterministic Halton exploration.
    # ------------------------------------------------------------------
    def _halton_explore(
        self,
        regions: Sequence[Region],
        n_target: int,
        verbose: bool,
    ) -> List[LayoutCandidate]:
        out: List[LayoutCandidate] = []
        evaluated = 0
        attempts = 0
        region_i = 0
        max_attempts = max(
            n_target * 5 + 30,
            int(ACFG["halton_max_layout_attempts"]),
        )
        cap = int(ACFG["search_grasp_cap"])
        fast.MAX_GRASPS_PER_POSE = cap

        print(
            "\n---------- Phase A2: deterministic Halton exploration ----------"
        )
        print(
            f"target={n_target}, cap={cap}, region scheduling=center-first round-robin"
        )
        print(
            "layout sampling=one deterministic joint 3N-D Halton pose point per complete layout; RNG is not used"
        )

        while (
            evaluated < n_target
            and attempts < max_attempts
            and not self._eval_budget_exhausted()
        ):
            attempts += 1
            region = regions[region_i % len(regions)]
            region_i += 1
            self._set_region_from_tuple(region)

            xy = self._sample_collision_free_xy_halton(region)
            if xy is None:
                continue

            cand = self._fresh_evaluate(xy, region, cap=cap)
            if cand is None:
                break
            evaluated += 1
            self._phase_a_evaluated += 1

            if verbose:
                tag = "L2_OK" if cand.l2_pass else "FAIL"
                print(
                    f"[halton] {evaluated:03d}/{n_target} {tag:5s} "
                    f"score={float(getattr(cand, 'layout_score', -1.0)):.4f} "
                    f"region={cand.assembly_region_id}"
                )
                if not cand.l2_pass:
                    print(f"         fail: {getattr(cand, 'fail_reason', '?')}")

            if not cand.l2_pass:
                continue

            out.append(cand)

            # If approximate score is already competitive, immediately verify
            # at the original cap=350 and checkpoint it before any later crash.
            incumbent_score = (
                float(self._authoritative_best.layout_score)
                if self._authoritative_best is not None
                else -math.inf
            )
            threshold = incumbent_score + float(ACFG["online_verify_margin"])
            if float(cand.layout_score) > threshold:
                verified = self._fresh_evaluate(
                    cand.xy,
                    region,
                    cap=int(ACFG["authoritative_grasp_cap"]),
                )
                if verified is not None and verified.l2_pass:
                    # Replace the just-appended approximate candidate with its
                    # authoritative version, avoiding a duplicate re-score.
                    out[-1] = verified
                    self._promote_authoritative(
                        verified,
                        stage=f"halton_online_verify_{evaluated:03d}",
                    )
                fast.MAX_GRASPS_PER_POSE = cap

        return out

    # ------------------------------------------------------------------
    # Re-score candidates under original cap=350.
    # ------------------------------------------------------------------
    def _authoritative_rescore(
        self,
        candidates: Sequence[LayoutCandidate],
        regions: Sequence[Region],
        topk: int,
        stage: str,
    ) -> List[LayoutCandidate]:
        candidates = [
            c for c in candidates if bool(getattr(c, "l2_pass", False))
        ]
        ranked = sorted(
            candidates,
            key=lambda c: float(c.layout_score),
            reverse=True,
        )
        unique = self._unique_elites(ranked, limit=max(1, int(topk)))
        out: List[LayoutCandidate] = []
        cap = int(ACFG["authoritative_grasp_cap"])

        print(
            f"\n---------- Authoritative re-score: {stage} "
            f"(top_k={len(unique)}, cap={cap}) ----------"
        )
        for i, src in enumerate(unique, 1):
            if self._eval_budget_exhausted():
                print(
                    f"[rescore] stop before {i:02d}/{len(unique)}: "
                    "strict evaluation budget exhausted"
                )
                break

            reusable = None
            if (
                int(getattr(src, "_evaluated_grasp_cap", -1)) == cap
                and bool(getattr(src, "l2_pass", False))
            ):
                reusable = src
            elif (
                self._authoritative_best is not None
                and bool(getattr(self._authoritative_best, "l2_pass", False))
                and int(
                    getattr(
                        self._authoritative_best,
                        "_evaluated_grasp_cap",
                        cap,
                    )
                ) == cap
                and self._same_xy_layout(
                    src,
                    self._authoritative_best,
                )
            ):
                reusable = self._authoritative_best

            if reusable is not None:
                out.append(reusable)
                self._promote_authoritative(
                    reusable,
                    stage=f"{stage}_{i:02d}_reuse",
                )
                print(
                    f"[rescore] {i:02d}/{len(unique)} "
                    f"fast={float(src.layout_score):.4f} "
                    f"auth={float(reusable.layout_score):.4f} "
                    "time=0.0s reused-authoritative"
                )
                continue

            region = getattr(src, "_nsga_region_tuple", None)
            if region is None:
                region = self._region_by_id(regions, src.assembly_region_id)
            t0 = time.time()
            fresh = self._fresh_evaluate(src.xy, region, cap=cap)
            elapsed = time.time() - t0
            if fresh is not None and fresh.l2_pass:
                out.append(fresh)
                self._promote_authoritative(
                    fresh,
                    stage=f"{stage}_{i:02d}",
                )
            print(
                f"[rescore] {i:02d}/{len(unique)} "
                f"fast={float(src.layout_score):.4f} "
                f"auth={float(fresh.layout_score) if fresh is not None and fresh.l2_pass else float('nan'):.4f} "
                f"time={elapsed:.1f}s"
            )
        return out

    def _print_coverage(self, regions: Sequence[Region]) -> None:
        bins = max(1, int(ACFG.get("coverage_bins", 4)))
        total_cells = bins * bins
        print("\n---------- Deterministic coverage report ----------")
        print(f"spatial bins per part/region = {bins}x{bins}")
        for region in regions:
            rid = str(region[0])
            vals = []
            for pid in self.part_order:
                cells = self._coverage_cells.get((rid, str(pid)))
                if cells:
                    vals.append(len(cells) / total_cells)
            if vals:
                print(
                    f"region={rid}: mean occupied-bin ratio="
                    f"{100.0 * float(np.mean(vals)):.1f}% "
                    f"(accepted Halton placements)"
                )

    # ------------------------------------------------------------------
    # Main Global pipeline.
    # ------------------------------------------------------------------
    def random_search(
        self,
        n_samples: int,
        seed: int,
        max_resample_layout: int = 80,
        verbose: bool = True,
        enable_l3: bool = False,
        l3_top_k: int = 3,
        l3_obstacle_mode: str = "staging_aware",
        require_l3: bool = True,
    ) -> Optional[LayoutCandidate]:
        self._reset_eval_progress_stats()
        self._nsga_eval_cache.clear()
        self._nsga_eval_count = 0
        self._nsga_cache_hits = 0
        self._phase_a_evaluated = 0

        regions = self._order_regions_center_first(
            self._assembly_region_candidates(),
            verbose=verbose,
        )
        gmod.GCFG["max_resample_layout"] = int(max_resample_layout)

        n_explore = int(
            ACFG["explore"]
            if ACFG["explore"] is not None
            else (gmod.GCFG["explore"] or n_samples)
        )
        elite_k = max(1, int(gmod.GCFG["elite"]))
        steps = [float(s) for s in gmod.GCFG["refine_steps"]]
        rounds = int(gmod.GCFG["refine_rounds"])
        diagonal = bool(gmod.GCFG["refine_diagonal"])
        refine_enabled = (
            bool(gmod.GCFG["refine_enabled"])
            and len(steps) > 0
            and rounds > 0
        )

        print("\n========== Global Accelerated Search ==========")
        print("main algorithm       = original Global explore + pattern refine")
        print(f"total explore        = {n_explore}")
        print(f"search grasp cap     = {ACFG['search_grasp_cap']}")
        print(
            f"authoritative cap    = {ACFG['authoritative_grasp_cap']} "
            "(final comparison cap; use 350 to match the old formal Global)"
        )
        print(f"checkpoint           = {self._checkpoint_path}")
        print(f"refine               = {refine_enabled}, steps={steps}, rounds={rounds}")
        print(f"max evaluations      = {nsga2._CFG.get('max_evals')}")
        print("new layouts          = deterministic joint 3N-D Halton position/rotation-priority coverage")

        t0 = time.time()
        feasible: List[LayoutCandidate] = []
        warm_loaded = False

        # Validate/transfer the optional layout according to current part set.
        warm_rec = self._load_warm_layout_record()
        warm_resolved_mode = "none"
        if warm_rec is not None and not self._eval_budget_exhausted():
            warm_xy, warm_region, warm_resolved_mode, warm_stats = (
                self._prepare_warm_seed(warm_rec, regions)
            )
            if warm_xy is not None and warm_region is not None:
                warm_cand = self._fresh_evaluate(
                    warm_xy,
                    warm_region,
                    cap=int(ACFG["authoritative_grasp_cap"]),
                )
                if warm_cand is not None and warm_cand.l2_pass:
                    warm_loaded = True
                    warm_cand._warm_start_source = warm_rec.get("source_path")
                    warm_cand._warm_start_mode = warm_resolved_mode
                    warm_cand._warm_transfer_stats = warm_stats
                    feasible.append(warm_cand)
                    self._promote_authoritative(
                        warm_cand,
                        stage=f"warm_{warm_resolved_mode}_validation",
                    )
                    print(
                        f"[warm-start] mode={warm_resolved_mode}, "
                        f"strict current score={warm_cand.layout_score:.4f}; "
                        "full current-part XY checkpoint saved"
                    )
                else:
                    print(
                        f"[warm-start] mode={warm_resolved_mode} seed did not "
                        "pass the current strict evaluator; continuing with "
                        "deterministic Halton Global"
                    )

        # Resume a previously saved authoritative incumbent only when no
        # warm layout was just validated in this run.
        rec = None if warm_loaded else self._load_checkpoint_record()
        checkpoint_loaded = False
        if rec is not None and not self._eval_budget_exhausted():
            region = self._region_from_record(rec, regions)
            xy = {
                str(pid): np.asarray(pos, dtype=float)
                for pid, pos in dict(rec.get("xy", {})).items()
            }
            loaded = self._fresh_evaluate(
                xy,
                region,
                cap=int(ACFG["authoritative_grasp_cap"]),
            )
            if loaded is not None and loaded.l2_pass:
                checkpoint_loaded = True
                feasible.append(loaded)
                self._promote_authoritative(loaded, stage="checkpoint_resume")
                print(
                    f"[checkpoint] resumed valid incumbent "
                    f"score={loaded.layout_score:.4f}"
                )
            else:
                print("[checkpoint] stored incumbent did not pass re-evaluation")

        # Warm-first local optimization: when an existing strict-feasible
        # layout is available, refine it before spending the remaining budget
        # on other candidates. This produces a useful result early.
        warm_first_refined: List[LayoutCandidate] = []
        if (
            warm_loaded
            and bool(ACFG.get("warm_start_refine_first", True))
            and self._authoritative_best is not None
            and not self._eval_budget_exhausted()
        ):
            print(
                "\n---------- Phase W: warm-start-first pattern refine ----------"
            )
            fast.MAX_GRASPS_PER_POSE = int(ACFG["search_grasp_cap"])
            self._nsga_eval_cache.clear()
            warm_seed = self._authoritative_best
            warm_first_refined.append(
                self._pattern_refine(
                    warm_seed,
                    steps=steps,
                    rounds=rounds,
                    diagonal=diagonal,
                    verbose=verbose,
                )
            )
            warm_verified = self._authoritative_rescore(
                warm_first_refined,
                regions,
                topk=1,
                stage="warm_first_refine",
            )
            feasible.extend(warm_verified)

        # No saved XY exists for last night's 0.5286, so first run replays
        # the exact old fixed-seed prefix before switching to Halton.
        legacy_n = 0 if (
            checkpoint_loaded
            or warm_rec is not None
            or warm_loaded
        ) else min(
            n_explore,
            max(0, int(ACFG["legacy_replay"])),
        )
        feasible.extend(
            self._legacy_anchor_replay(
                regions,
                n_target=legacy_n,
                verbose=verbose,
            )
        )

        deterministic_n = max(0, n_explore - legacy_n)
        feasible.extend(
            self._halton_explore(
                regions,
                n_target=deterministic_n,
                verbose=verbose,
            )
        )
        self._print_coverage(regions)

        if not feasible and self._authoritative_best is None:
            print("\n[FAIL] Phase A found no L2-feasible layout.")
            return None

        # Put all promising candidates on the original Global scoring cap=350.
        pre_rescored = self._authoritative_rescore(
            feasible,
            regions,
            topk=int(ACFG["rescore_topk"]),
            stage="pre_refine",
        )

        seeds = list(pre_rescored)
        if self._authoritative_best is not None:
            seeds.append(self._authoritative_best)
        seeds = self._unique_elites(
            [c for c in seeds if c is not None and c.l2_pass],
            limit=max(elite_k, int(l3_top_k), int(ACFG["rescore_topk"])),
        )
        seeds = sorted(seeds, key=lambda c: float(c.layout_score), reverse=True)

        # Pattern refine remains the original Global method.
        refined: List[LayoutCandidate] = []
        if refine_enabled:
            print("\n---------- Phase B: original multiscale pattern refine ----------")
            fast.MAX_GRASPS_PER_POSE = int(ACFG["search_grasp_cap"])
            self._nsga_eval_cache.clear()
            for i, seed_cand in enumerate(seeds[:elite_k], 1):
                if self._eval_budget_exhausted():
                    break
                print(
                    f"[refine] elite#{i} authoritative_start="
                    f"{seed_cand.layout_score:.4f}"
                )
                refined.append(
                    self._pattern_refine(
                        seed_cand,
                        steps=steps,
                        rounds=rounds,
                        diagonal=diagonal,
                        verbose=verbose,
                    )
                )

        # Final comparison under the same cap=350 used by the previous 0.5286.
        final_pool = list(seeds) + list(refined)
        final_rescored = self._authoritative_rescore(
            final_pool,
            regions,
            topk=max(elite_k, int(ACFG["rescore_topk"])),
            stage="post_refine",
        )
        if self._authoritative_best is not None:
            final_rescored.append(self._authoritative_best)

        final_rescored = [
            c for c in final_rescored if c is not None and c.l2_pass
        ]
        if not final_rescored:
            print("\n[FAIL] No authoritative feasible layout.")
            return None
        final_rescored = self._unique_elites(
            final_rescored,
            limit=max(elite_k, int(l3_top_k), int(ACFG["rescore_topk"])),
        )
        final_rescored.sort(
            key=lambda c: float(c.layout_score),
            reverse=True,
        )
        best = final_rescored[0]
        self._promote_authoritative(best, stage="final")

        print("\n========== Global Accelerated Summary ==========")
        print(f"best authoritative score = {best.layout_score:.4f}")
        print(f"best region              = {best.assembly_region_id}")
        print(f"total strict evaluations = {self._nsga_eval_count}")
        print(f"wall time                = {time.time() - t0:.1f}s")
        print(f"checkpoint               = {self._checkpoint_path}")
        self.print_search_eval_progress()

        # Optional full-grasp diagnostic, disabled by default because the
        # historical 0.5286 score was defined under cap=350.
        full_k = min(
            max(0, int(ACFG["full_certify_topk"])),
            len(final_rescored),
        )
        if full_k > 0:
            print(
                f"\n---------- Optional full-grasp diagnostic top-{full_k} ----------"
            )
            for i, src in enumerate(final_rescored[:full_k], 1):
                if self._eval_budget_exhausted():
                    break
                region = getattr(src, "_nsga_region_tuple", None)
                if region is None:
                    region = self._region_by_id(regions, src.assembly_region_id)
                full = self._fresh_evaluate(src.xy, region, cap=0)
                print(
                    f"[full] {i}/{full_k} cap350={src.layout_score:.4f} "
                    f"full={float(full.layout_score) if full is not None and full.l2_pass else float('nan'):.4f}"
                )

        if enable_l3:
            k = min(int(l3_top_k), len(final_rescored))
            for rank, cand in enumerate(final_rescored[:k], 1):
                if self.validate_full_sequence_l3(
                    cand,
                    obstacle_mode=l3_obstacle_mode,
                    verbose=True,
                ):
                    return cand
            if require_l3:
                return None

        return best


def _consume_accel_args() -> None:
    pairs = {
        "--global-accel-explore": ("explore", int),
        "--global-accel-legacy-replay": ("legacy_replay", int),
        "--global-accel-legacy-seed": ("legacy_seed", int),
        "--global-accel-search-grasp-cap": ("search_grasp_cap", int),
        "--global-accel-authoritative-grasp-cap": (
            "authoritative_grasp_cap",
            int,
        ),
        "--global-accel-grasp-mode": ("grasp_mode", str),
        "--global-accel-fps-seed-mode": ("fps_seed_mode", str),
        "--global-accel-rescore-topk": ("rescore_topk", int),
        "--global-accel-full-certify-topk": ("full_certify_topk", int),
        "--global-accel-halton-layout-attempts": (
            "halton_max_layout_attempts",
            int,
        ),
        "--global-accel-halton-repair-attempts": (
            "halton_repair_attempts_per_part",
            int,
        ),
        "--global-accel-coverage-bins": ("coverage_bins", int),
        "--global-accel-warm-layout": ("warm_layout", str),
        "--global-accel-warm-mode": ("warm_mode", str),
        "--global-accel-checkpoint": ("checkpoint_json", str),
        "--global-accel-online-verify-margin": (
            "online_verify_margin",
            float,
        ),
    }
    for flag, (key, cast) in pairs.items():
        value = fast._consume_extra_value(flag)
        if value is not None:
            ACFG[key] = cast(value)

    if fast._consume_extra_flag("--global-accel-no-resume"):
        ACFG["resume_checkpoint"] = False
    if fast._consume_extra_flag("--global-accel-no-warm-first-refine"):
        ACFG["warm_start_refine_first"] = False
    if fast._consume_extra_flag("--global-accel-no-mesh-prefilter"):
        ACFG["halton_prefilter_mesh_clearance"] = False
    if fast._consume_extra_flag("--global-accel-no-home-prefilter"):
        ACFG["halton_prefilter_home_collision"] = False

    grasp_mode = str(ACFG.get("grasp_mode", "")).strip().lower()
    if grasp_mode not in {
        "same_grasp_progressive",
        "same_grasp_full",
        "common_progressive",
        "common_full",
    }:
        raise ValueError(
            "--global-accel-grasp-mode must be "
            "same_grasp_progressive or same_grasp_full"
        )

    fps_seed_mode = str(
        ACFG.get("fps_seed_mode", "")
    ).strip().lower()
    if fps_seed_mode not in {"medoid", "extreme", "first"}:
        raise ValueError(
            "--global-accel-fps-seed-mode must be "
            "medoid, extreme, or first"
        )

    if int(ACFG["search_grasp_cap"]) <= 0:
        raise ValueError("--global-accel-search-grasp-cap must be positive")
    if int(ACFG["authoritative_grasp_cap"]) <= 0:
        raise ValueError(
            "--global-accel-authoritative-grasp-cap must be positive"
        )
    if int(ACFG["rescore_topk"]) <= 0:
        raise ValueError("--global-accel-rescore-topk must be positive")


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = GlobalAccelSearcher


def main() -> None:
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()

    # Preserve all original Global CLI options.
    gmod._consume_global_args()
    _consume_accel_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()

    # Search-stage deterministic grasp coverage and progressive intersection.
    uniform_accel.install(
        int(ACFG["search_grasp_cap"]),
        mode=str(ACFG["grasp_mode"]),
        fps_seed_mode=str(ACFG["fps_seed_mode"]),
    )
    uniform_accel.reset_all()

    if ACFG["checkpoint_json"] is None:
        ACFG["checkpoint_json"] = _default_checkpoint_path()

    print("[global-accel-v3.1] config:")
    for key, value in ACFG.items():
        print(f"    {key:30s} = {value}")
    print("[global-accel-v3.1] original Global file remains untouched")
    print(
        "[global-accel-v3.1] rotation handling: Halton prioritizes "
        "prefilter rotations; strict evaluate_layout still enumerates all "
        "rot_cands for every part"
    )

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(
            f"[global-accel-v3.1] wall-clock total = "
            f"{time.perf_counter() - wall_t0:.3f}s"
        )
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
