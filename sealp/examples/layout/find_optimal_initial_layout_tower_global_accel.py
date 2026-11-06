#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accelerated deterministic Global-to-Local layout search.

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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod
import uniform_grasp_accel as uniform_accel

LayoutCandidate = fol.LayoutCandidate
Region = Tuple[str, Tuple[int, int], np.ndarray]


ACFG: Dict[str, object] = {
    # Number of total Phase-A strict layout evaluations.
    "explore": None,

    # Replay the first 28 original Global candidates when no checkpoint exists.
    # The previous run found 0.5286 at original candidate #22.
    "legacy_replay": 28,
    "legacy_seed": 0,

    # Fast search and authoritative comparison grasp caps.
    "search_grasp_cap": 160,
    "authoritative_grasp_cap": 350,

    # How many approximate candidates are authoritatively re-scored before refine.
    "rescore_topk": 8,

    # Optional full-grasp diagnostic. 0 disables it.
    "full_certify_topk": 0,

    # Halton candidate-generation settings.
    "halton_max_attempts_per_part": 180,
    "coverage_bins": 4,

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
    """Return one coordinate of a deterministic Halton sequence."""
    i = max(1, int(index))
    inv = 1.0 / float(base)
    factor = inv
    value = 0.0
    while i > 0:
        value += factor * float(i % base)
        i //= base
        factor *= inv
    return float(value)


class Halton2DStreams:
    """Independent deterministic 2-D Halton streams per region and part."""

    def __init__(self) -> None:
        self._counter: Dict[Tuple[str, str], int] = defaultdict(int)

    @staticmethod
    def _stable_offset(region_id: str, part_id: str) -> int:
        # Stable across Python runs; do not use hash(), whose salt changes.
        text = f"{region_id}|{part_id}".encode("utf-8")
        acc = 0
        for b in text:
            acc = (acc * 131 + int(b)) % 1000003
        return 1 + acc

    def next(self, region_id: str, part_id: str) -> np.ndarray:
        key = (str(region_id), str(part_id))
        self._counter[key] += 1
        idx = self._stable_offset(*key) + self._counter[key]
        return np.array(
            [_radical_inverse(idx, 2), _radical_inverse(idx, 3)],
            dtype=float,
        )


class GlobalAccelSearcher(gmod.GlobalLayoutSearcher):
    """Original Global search with deterministic coverage and acceleration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._halton = Halton2DStreams()
        self._authoritative_best: Optional[LayoutCandidate] = None
        self._authoritative_best_stage: str = ""
        self._coverage_cells: Dict[Tuple[str, str], set] = defaultdict(set)
        self._phase_a_evaluated = 0

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

    def _sample_collision_free_xy_halton(
        self,
        region: Region,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Equivalent cheap geometry construction using deterministic Halton points."""
        xy: Dict[str, np.ndarray] = {}
        placed: List[str] = []
        first_pid = self._first_part_id() if self.preassemble_first_part else None

        if first_pid is not None and first_pid in self.world_poses:
            gp, gr = self.world_poses[first_pid]
            xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            if first_pid in self.staging_models:
                self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
                self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()
                self._update_world_cache_from_cm(first_pid)
            placed.append(first_pid)

        order = sorted(
            [p for p in self.part_order if p != first_pid],
            key=lambda p: float(np.prod(self.rot_cands[p][0].footprint)),
            reverse=True,
        )
        max_attempts = max(1, int(ACFG["halton_max_attempts_per_part"]))

        for pid in order:
            cand0 = self.rot_cands[pid][0]
            (xlo, xhi), (ylo, yhi) = self._xy_bounds_for_part_and_cand(pid, cand0)
            if xlo >= xhi or ylo >= yhi:
                return None

            preferred_y_range, goal_side = self._preferred_y_range_by_goal_side(
                pid,
                ylo,
                yhi,
                first_pid=first_pid,
            )
            prefer_attempts = int(
                round(max_attempts * float(self.goal_y_side_bias_ratio))
            )

            accepted = False
            for attempt_i in range(max_attempts):
                u = self._halton.next(str(region[0]), str(pid))

                if attempt_i < prefer_attempts and goal_side in ("left", "right"):
                    cur_ylo, cur_yhi = preferred_y_range
                else:
                    cur_ylo, cur_yhi = ylo, yhi
                if cur_ylo >= cur_yhi:
                    cur_ylo, cur_yhi = ylo, yhi

                p = np.array(
                    [
                        xlo + float(u[0]) * (xhi - xlo),
                        cur_ylo + float(u[1]) * (cur_yhi - cur_ylo),
                    ],
                    dtype=float,
                )

                if self._staging_arm_keepout_reason(pid, p, cand0):
                    continue

                self._apply_staging_pose(pid, p, cand0)
                ab_pid = self._world_aabb_for_staging(pid)

                collision = False
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
                    if self.staging_models[pid].is_mcdwith(self.staging_models[q]):
                        collision = True
                        break
                if collision:
                    continue

                xy[pid] = p
                placed.append(pid)
                accepted = True
                self._note_coverage(
                    region,
                    pid,
                    p,
                    xlo,
                    xhi,
                    ylo,
                    yhi,
                )
                break

            if not accepted:
                return None

        return {pid: xy[pid] for pid in self.part_order if pid in xy}

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
        max_attempts = n_target * 5 + 30
        cap = int(ACFG["search_grasp_cap"])
        fast.MAX_GRASPS_PER_POSE = cap

        print(
            "\n---------- Phase A2: deterministic Halton exploration ----------"
        )
        print(
            f"target={n_target}, cap={cap}, region scheduling=center-first round-robin"
        )
        print(
            "layout sampling=deterministic per-region/per-part Halton(2,3); RNG is not used"
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
                break
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

        n_explore = int(ACFG["explore"] or gmod.GCFG["explore"] or n_samples)
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
        print(f"authoritative cap    = {ACFG['authoritative_grasp_cap']} (same as old Global)")
        print(f"checkpoint           = {self._checkpoint_path}")
        print(f"refine               = {refine_enabled}, steps={steps}, rounds={rounds}")
        print(f"max evaluations      = {nsga2._CFG.get('max_evals')}")
        print("new layouts          = deterministic Halton low-discrepancy coverage")

        t0 = time.time()
        feasible: List[LayoutCandidate] = []

        # Resume a previously saved authoritative incumbent.
        rec = self._load_checkpoint_record()
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

        # No saved XY exists for last night's 0.5286, so first run replays
        # the exact old fixed-seed prefix before switching to Halton.
        legacy_n = 0 if checkpoint_loaded else min(
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
        "--global-accel-rescore-topk": ("rescore_topk", int),
        "--global-accel-full-certify-topk": ("full_certify_topk", int),
        "--global-accel-halton-attempts": (
            "halton_max_attempts_per_part",
            int,
        ),
        "--global-accel-coverage-bins": ("coverage_bins", int),
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
    uniform_accel.install(int(ACFG["search_grasp_cap"]))
    uniform_accel.reset_all()

    if ACFG["checkpoint_json"] is None:
        ACFG["checkpoint_json"] = _default_checkpoint_path()

    print("[global-accel] config:")
    for key, value in ACFG.items():
        print(f"    {key:30s} = {value}")
    print("[global-accel] original Global file remains untouched")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(
            f"[global-accel] wall-clock total = "
            f"{time.perf_counter() - wall_t0:.3f}s"
        )
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
