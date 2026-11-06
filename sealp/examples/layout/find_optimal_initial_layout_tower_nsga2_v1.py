#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NSGA-II v1 Tower Initial Layout Search
=====================================

This script is the first NSGA-II version of the tower initial layout search.
It does NOT rewrite the existing feasibility checker. Instead, it reuses the
current strict/fast pipeline:

- stable pose candidates and staging constraints from strict_pycharm.py;
- IK cache, AABB pre-checks, and execution-aware obstacle handling from
  strict_pycharm_fast.py;
- the original save/debug/output logic from strict_pycharm.py.

Only the outer search loop is replaced by a lightweight NSGA-II loop.

Main difference from the previous PSO script:
    random / heatmap / PSO  ->  NSGA-II population evolution

Current scope:
    NSGA-II + existing Level-1/Level-2 analytic feasibility evaluation.
    L3 validation is still optional and is OFF by default.

Run:
    python -m sealp.examples.layout.find_optimal_initial_layout_tower_nsga2_v1

Useful options:
    --n-samples 30              population size if --nsga-pop is not set
    --nsga-pop 40               NSGA-II population size
    --nsga-generations 4        number of generations
    --nsga-offspring 5          offspring per generation; smaller is faster
    --nsga-max-evals 40         hard cap on real evaluate_layout calls
    --nsga-early-stop 2         stop after N generations without improvement
    --nsga-mutation-sigma 0.04  XY mutation std in meters
    --enable-l3                 optional top-K L3 validation
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Reuse the original CLI / saving logic and the accelerated evaluator.
from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
from sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm import LayoutCandidate


# ============================================================
# NSGA-II defaults
# ============================================================

_CFG: Dict[str, Any] = {
    "pop": None,                 # None means use --n-samples
    "generations": 4,            # v1.1 default: fewer generations for fast iteration
    "offspring": None,           # None means max(2, pop//2), not pop, to reduce evaluations
    "crossover_prob": 0.90,
    "mutation_prob": 0.35,
    "mutation_sigma": 0.04,      # meters
    "region_mutation_prob": 0.10,
    "tournament_k": 2,
    "local_jitter_prob": 0.20,
    "local_jitter_sigma": 0.015,
    "elite_pool_size": 20,

    # Speed controls. One real evaluate_layout call is expensive because it
    # runs grasp/IK/collision checks. These options reduce redundant calls.
    "max_evals": None,           # None means no hard cap
    "early_stop_rounds": 2,      # stop if best score does not improve for N generations
    "min_improvement": 1e-4,
    "eval_cache_ndigits": 3,     # cache layouts rounded to 1 mm
    "init_attempt_factor": 5,
}


# ============================================================
# Small helpers
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip01(x: float) -> float:
    return float(np.clip(_safe_float(x), 0.0, 1.0))


def _copy_xy(xy: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float).copy() for k, v in xy.items()}


def _candidate_signature(c: LayoutCandidate, ndigits: int = 3) -> Tuple:
    """A coarse signature used to remove near-duplicate candidates."""
    items = []
    for pid in sorted(c.xy.keys()):
        p = np.asarray(c.xy[pid], dtype=float)
        items.append((pid, round(float(p[0]), ndigits), round(float(p[1]), ndigits)))
    return (str(c.assembly_region_id), tuple(items))


def _gene_signature(gene_xy: Dict[str, np.ndarray],
                    region: Tuple[str, Tuple[int, int], np.ndarray],
                    ndigits: int = 3) -> Tuple:
    """A rounded signature for evaluation-cache lookup."""
    items = []
    for pid in sorted(gene_xy.keys()):
        p = np.asarray(gene_xy[pid], dtype=float)
        items.append((pid, round(float(p[0]), ndigits), round(float(p[1]), ndigits)))
    return (str(region[0]), tuple(items))


def _clone_candidate(c: LayoutCandidate) -> LayoutCandidate:
    """Clone a candidate before returning it from the evaluation cache."""
    try:
        out = copy.deepcopy(c)
    except Exception:
        out = copy.copy(c)
        if hasattr(c, "xy"):
            out.xy = _copy_xy(c.xy)
    return out


# ============================================================
# NSGA-II implementation
# ============================================================

class NSGA2LayoutSearcher(fast.FastWeightedInitialLayoutSearcher):
    """Outer-loop NSGA-II searcher.

    The class deliberately reuses ``evaluate_layout`` from the current fast
    searcher. Therefore all existing constraints, scoring terms, grasp checks,
    and obstacle updates are preserved. Only candidate generation and selection
    are changed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nsga_eval_cache: Dict[Tuple, LayoutCandidate] = {}
        self._nsga_eval_count = 0
        self._nsga_cache_hits = 0
        self._reset_eval_progress_stats()

    def _reset_eval_progress_stats(self) -> None:
        """Track when first feasible / best L2 score appear (by real eval index)."""
        self._first_l2_ok_eval: Optional[int] = None
        self._best_score_eval: Optional[int] = None
        self._best_score_seen: float = -math.inf

    def _note_eval_outcome(self, cand: LayoutCandidate) -> None:
        if not bool(getattr(cand, "l2_pass", False)):
            return
        idx = int(self._nsga_eval_count)
        if self._first_l2_ok_eval is None:
            self._first_l2_ok_eval = idx
        score = _safe_float(getattr(cand, "layout_score", -math.inf), -math.inf)
        if score > float(self._best_score_seen) + 1e-6:
            self._best_score_seen = score
            self._best_score_eval = idx

    def search_eval_stats(self) -> Dict[str, Optional[float]]:
        return {
            "real_evals": int(self._nsga_eval_count),
            "cache_hits": int(self._nsga_cache_hits),
            "first_l2_ok_eval": self._first_l2_ok_eval,
            "best_score_eval": self._best_score_eval,
            "best_score_seen_during_search": (
                float(self._best_score_seen) if self._best_score_seen > -math.inf else None
            ),
        }

    def print_search_eval_progress(self) -> None:
        st = self.search_eval_stats()
        print(f"first L2_OK at eval # = {st['first_l2_ok_eval']}")
        bse = st["best_score_eval"]
        bss = st["best_score_seen_during_search"]
        if bse is not None and bss is not None:
            print(f"best score first at  = #{bse}  (score={bss:.4f})")
        else:
            print(f"best score first at  = {bse}")

    def _eval_budget_exhausted(self) -> bool:
        max_evals = _CFG.get("max_evals", None)
        return max_evals is not None and self._nsga_eval_count >= int(max_evals)

    @staticmethod
    def _best_layout_score(population: Sequence[LayoutCandidate]) -> float:
        feasible = [c for c in population if bool(getattr(c, "l2_pass", False))]
        if not feasible:
            return -math.inf
        return max(_safe_float(getattr(c, "layout_score", -math.inf), -math.inf) for c in feasible)

    # -----------------------------
    # Evaluation and objectives
    # -----------------------------

    def _set_region_from_tuple(self, region: Tuple[str, Tuple[int, int], np.ndarray]) -> None:
        rid, rc, pos = region
        self._set_assembly_station(pos, region_id=rid, rc=rc)

    def _region_by_id(self, regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]], rid: str):
        for r in regions:
            if r[0] == rid:
                return r
        return regions[0]

    def _objectives_from_candidate(self, cand: LayoutCandidate) -> Tuple[float, float, float, float]:
        """Return minimization objectives for NSGA-II.

        We do not have a trained GNN in v1. The fourth objective therefore uses
        the existing analytic layout score as a surrogate sequence score.
        Later, this term can be replaced by ``-P_seq`` from the GNN.
        """
        if not bool(getattr(cand, "l2_pass", False)):
            # All infeasible layouts are dominated by feasible ones.
            # A mild bucket penalty keeps the population diverse among failures.
            reason = str(getattr(cand, "fail_reason", ""))
            penalty = 10.0
            if "collision" in reason:
                penalty += 0.5
            if "no_common_gids" in reason:
                penalty += 0.8
            if "upright" in reason:
                penalty += 0.3
            return (penalty, penalty, penalty, penalty)

        f_dist = 1.0 - _clip01(getattr(cand, "dist_score_norm", 0.0))
        f_manip = 1.0 - _clip01(getattr(cand, "manip_score_norm", 0.0))
        f_grasp = 1.0 - _clip01(getattr(cand, "grasp_score_norm", 0.0))
        f_seq = 1.0 - _clip01(getattr(cand, "layout_score", 0.0))
        return (f_dist, f_manip, f_grasp, f_seq)

    def _evaluate_gene(self,
                       gene_xy: Dict[str, np.ndarray],
                       region: Tuple[str, Tuple[int, int], np.ndarray]) -> LayoutCandidate:
        """Evaluate one layout gene.

        v1.1 adds a rounded evaluation cache. NSGA-II frequently creates very
        similar children, especially when mutation is small. Reusing the
        cached result avoids repeating the expensive evaluate_layout call.
        """
        ndigits = int(_CFG.get("eval_cache_ndigits", 3))
        key = _gene_signature(gene_xy, region, ndigits=ndigits)
        cached = self._nsga_eval_cache.get(key)
        if cached is not None:
            self._nsga_cache_hits += 1
            return _clone_candidate(cached)

        self._set_region_from_tuple(region)
        cand = LayoutCandidate(xy=_copy_xy(gene_xy))
        self._nsga_eval_count += 1
        ok = bool(self.evaluate_layout(cand))
        cand.l2_pass = ok
        cand._nsga_objectives = self._objectives_from_candidate(cand)  # type: ignore[attr-defined]
        cand._nsga_region_tuple = region  # type: ignore[attr-defined]
        self._note_eval_outcome(cand)

        # Store a clone so later rank/crowding annotations do not pollute the cache.
        self._nsga_eval_cache[key] = _clone_candidate(cand)
        return cand

    # -----------------------------
    # Population initialization
    # -----------------------------

    def _random_candidate(self,
                          rng: np.random.Generator,
                          regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
                          max_resample_layout: int) -> Optional[LayoutCandidate]:
        if self._eval_budget_exhausted():
            return None
        region = regions[int(rng.integers(0, len(regions)))]
        self._set_region_from_tuple(region)
        xy = None
        for _ in range(max_resample_layout):
            xy = self.sample_collision_free_xy(rng)
            if xy is not None:
                break
        if xy is None:
            return None
        return self._evaluate_gene(xy, region)

    def _init_population(self,
                         rng: np.random.Generator,
                         regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
                         pop_size: int,
                         max_resample_layout: int,
                         verbose: bool) -> List[LayoutCandidate]:
        pop: List[LayoutCandidate] = []
        attempts = 0
        max_attempts = max(pop_size * int(_CFG.get("init_attempt_factor", 5)), pop_size + 10)
        # L2 FAIL 归因: part_or_category -> {fail_counter 分项累加}; 便于定位"到底谁、哪一项"导致全挂。
        fail_agg: Dict[str, Dict[str, int]] = {}
        fail_part_hits: Dict[str, int] = {}
        while len(pop) < pop_size and attempts < max_attempts and not self._eval_budget_exhausted():
            attempts += 1
            cand = self._random_candidate(rng, regions, max_resample_layout)
            if cand is None:
                continue
            pop.append(cand)
            if verbose:
                tag = "L2_OK" if cand.l2_pass else "FAIL"
                score = _safe_float(getattr(cand, "layout_score", -1.0), -1.0)
                print(f"[init] {len(pop):03d}/{pop_size} {tag:5s} "
                      f"score={score:.4f} region={cand.assembly_region_id} "
                      f"obj={np.round(cand._nsga_objectives, 4).tolist()}")
                if not cand.l2_pass:
                    print(f"        fail_reason: {getattr(cand, 'fail_reason', '?')}")
                    fp = getattr(cand, "fail_part", None) or "unknown"
                    fail_part_hits[fp] = fail_part_hits.get(fp, 0) + 1
                    fd = getattr(cand, "fail_detail", None)
                    if isinstance(fd, dict) and fd:
                        agg = fail_agg.setdefault(fp, {})
                        for k, v in fd.items():
                            agg[k] = agg.get(k, 0) + int(v)
        if verbose and fail_part_hits:
            print("\n========== L2 FAIL 归因汇总 (init) ==========")
            print("说明: fail_part=第一个失败的零件/类别; 数字=该零件在其所有旋转×手臂候选里各失败原因累计次数。")
            for fp, hits in sorted(fail_part_hits.items(), key=lambda kv: -kv[1]):
                nz = {k: v for k, v in fail_agg.get(fp, {}).items() if v}
                print(f"  fail_part={fp:22s} 出现次数={hits:3d}  fail_counter_sum={nz if nz else '(非逐候选类别, 见上方 fail_reason)'}")
            print("=============================================\n")
        return pop

    # -----------------------------
    # NSGA-II sorting
    # -----------------------------

    @staticmethod
    def _dominates(a: LayoutCandidate, b: LayoutCandidate) -> bool:
        oa = tuple(getattr(a, "_nsga_objectives", (math.inf, math.inf, math.inf, math.inf)))
        ob = tuple(getattr(b, "_nsga_objectives", (math.inf, math.inf, math.inf, math.inf)))
        return all(x <= y + 1e-12 for x, y in zip(oa, ob)) and any(x < y - 1e-12 for x, y in zip(oa, ob))

    def _fast_non_dominated_sort(self, population: Sequence[LayoutCandidate]) -> List[List[LayoutCandidate]]:
        S: Dict[int, List[int]] = {i: [] for i in range(len(population))}
        n: Dict[int, int] = {i: 0 for i in range(len(population))}
        fronts_idx: List[List[int]] = [[]]

        for p in range(len(population)):
            for q in range(len(population)):
                if p == q:
                    continue
                if self._dominates(population[p], population[q]):
                    S[p].append(q)
                elif self._dominates(population[q], population[p]):
                    n[p] += 1
            if n[p] == 0:
                population[p]._nsga_rank = 0  # type: ignore[attr-defined]
                fronts_idx[0].append(p)

        i = 0
        while i < len(fronts_idx) and fronts_idx[i]:
            next_front: List[int] = []
            for p in fronts_idx[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        population[q]._nsga_rank = i + 1  # type: ignore[attr-defined]
                        next_front.append(q)
            i += 1
            if next_front:
                fronts_idx.append(next_front)

        return [[population[i] for i in front] for front in fronts_idx if front]

    @staticmethod
    def _assign_crowding_distance(front: Sequence[LayoutCandidate]) -> None:
        if not front:
            return
        for c in front:
            c._nsga_crowding = 0.0  # type: ignore[attr-defined]
        m = len(tuple(getattr(front[0], "_nsga_objectives", ())))
        if len(front) <= 2:
            for c in front:
                c._nsga_crowding = float("inf")  # type: ignore[attr-defined]
            return
        for j in range(m):
            sorted_front = sorted(front, key=lambda c: getattr(c, "_nsga_objectives", (math.inf,))[j])
            sorted_front[0]._nsga_crowding = float("inf")  # type: ignore[attr-defined]
            sorted_front[-1]._nsga_crowding = float("inf")  # type: ignore[attr-defined]
            v_min = getattr(sorted_front[0], "_nsga_objectives")[j]
            v_max = getattr(sorted_front[-1], "_nsga_objectives")[j]
            denom = max(float(v_max - v_min), 1e-12)
            for k in range(1, len(sorted_front) - 1):
                prev_v = getattr(sorted_front[k - 1], "_nsga_objectives")[j]
                next_v = getattr(sorted_front[k + 1], "_nsga_objectives")[j]
                if not math.isinf(getattr(sorted_front[k], "_nsga_crowding", 0.0)):
                    sorted_front[k]._nsga_crowding += float((next_v - prev_v) / denom)  # type: ignore[attr-defined]

    def _rank_population(self, population: Sequence[LayoutCandidate]) -> List[List[LayoutCandidate]]:
        fronts = self._fast_non_dominated_sort(population)
        for front in fronts:
            self._assign_crowding_distance(front)
        return fronts

    @staticmethod
    def _better_for_tournament(a: LayoutCandidate, b: LayoutCandidate) -> LayoutCandidate:
        ra = int(getattr(a, "_nsga_rank", 10**9))
        rb = int(getattr(b, "_nsga_rank", 10**9))
        if ra != rb:
            return a if ra < rb else b
        ca = _safe_float(getattr(a, "_nsga_crowding", 0.0), 0.0)
        cb = _safe_float(getattr(b, "_nsga_crowding", 0.0), 0.0)
        if ca != cb:
            return a if ca > cb else b
        # Tie-breaker: keep the candidate with better analytic score.
        return a if _safe_float(getattr(a, "layout_score", -1.0), -1.0) >= _safe_float(getattr(b, "layout_score", -1.0), -1.0) else b

    def _tournament(self, rng: np.random.Generator, population: Sequence[LayoutCandidate]) -> LayoutCandidate:
        k = max(2, int(_CFG["tournament_k"]))
        ids = rng.integers(0, len(population), size=k)
        best = population[int(ids[0])]
        for idx in ids[1:]:
            best = self._better_for_tournament(best, population[int(idx)])
        return best

    def _environmental_selection(self, population: Sequence[LayoutCandidate], pop_size: int) -> List[LayoutCandidate]:
        fronts = self._rank_population(population)
        new_pop: List[LayoutCandidate] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                front_sorted = sorted(front, key=lambda c: getattr(c, "_nsga_crowding", 0.0), reverse=True)
                new_pop.extend(front_sorted[:pop_size - len(new_pop)])
                break
        return new_pop

    # -----------------------------
    # Crossover and mutation
    # -----------------------------

    def _clip_xy_for_part(self, pid: str, xy: np.ndarray) -> np.ndarray:
        cand0 = self.rot_cands[pid][0]
        (xlo, xhi), (ylo, yhi) = self._xy_bounds_for_part_and_cand(pid, cand0)
        return np.array([
            float(np.clip(xy[0], xlo, xhi)),
            float(np.clip(xy[1], ylo, yhi)),
        ], dtype=float)

    def _make_child_gene(self,
                         rng: np.random.Generator,
                         p1: LayoutCandidate,
                         p2: LayoutCandidate,
                         regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Tuple[str, Tuple[int, int], np.ndarray]]:
        # Region inheritance / mutation.
        r1 = getattr(p1, "_nsga_region_tuple", None) or self._region_by_id(regions, p1.assembly_region_id)
        r2 = getattr(p2, "_nsga_region_tuple", None) or self._region_by_id(regions, p2.assembly_region_id)
        region = r1 if rng.random() < 0.5 else r2
        if rng.random() < float(_CFG["region_mutation_prob"]):
            region = regions[int(rng.integers(0, len(regions)))]

        first_pid = self._first_part_id() if self.preassemble_first_part else None
        child_xy: Dict[str, np.ndarray] = {}
        for pid in self.part_order:
            # The preassembled first part will be overwritten by evaluate_layout.
            if pid in p1.xy and pid in p2.xy and rng.random() < float(_CFG["crossover_prob"]):
                alpha = float(rng.random())
                base = alpha * np.asarray(p1.xy[pid], dtype=float) + (1.0 - alpha) * np.asarray(p2.xy[pid], dtype=float)
            elif pid in p1.xy and (pid not in p2.xy or rng.random() < 0.5):
                base = np.asarray(p1.xy[pid], dtype=float).copy()
            elif pid in p2.xy:
                base = np.asarray(p2.xy[pid], dtype=float).copy()
            else:
                # Fallback: sample inside the table for missing genes.
                cand0 = self.rot_cands[pid][0]
                (xlo, xhi), (ylo, yhi) = self._xy_bounds_for_part_and_cand(pid, cand0)
                base = np.array([rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)], dtype=float)

            if pid != first_pid and rng.random() < float(_CFG["mutation_prob"]):
                base = base + rng.normal(0.0, float(_CFG["mutation_sigma"]), size=2)
            if pid != first_pid and rng.random() < float(_CFG["local_jitter_prob"]):
                base = base + rng.normal(0.0, float(_CFG["local_jitter_sigma"]), size=2)
            child_xy[pid] = self._clip_xy_for_part(pid, base)

        return child_xy, region

    def _make_offspring(self,
                        rng: np.random.Generator,
                        population: Sequence[LayoutCandidate],
                        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
                        n_offspring: int) -> List[LayoutCandidate]:
        offspring: List[LayoutCandidate] = []
        while len(offspring) < n_offspring and not self._eval_budget_exhausted():
            p1 = self._tournament(rng, population)
            p2 = self._tournament(rng, population)
            gene_xy, region = self._make_child_gene(rng, p1, p2, regions)
            child = self._evaluate_gene(gene_xy, region)
            offspring.append(child)
        return offspring

    # -----------------------------
    # Reporting
    # -----------------------------

    def _print_front_summary(self, generation: int, population: Sequence[LayoutCandidate]) -> None:
        feasible = [c for c in population if bool(getattr(c, "l2_pass", False))]
        best = max(feasible, key=lambda c: c.layout_score, default=None)
        fronts = self._rank_population(population)
        f0 = fronts[0] if fronts else []
        if best is None:
            print(f"[NSGA-II] gen={generation:02d} feasible=0/{len(population)} front0={len(f0)}")
            return
        print(f"[NSGA-II] gen={generation:02d} feasible={len(feasible)}/{len(population)} "
              f"front0={len(f0)} best_score={best.layout_score:.4f} "
              f"region={best.assembly_region_id} obj={np.round(best._nsga_objectives, 4).tolist()} "
              f"evals={self._nsga_eval_count} cache_hits={self._nsga_cache_hits}")

    def _unique_elites(self, population: Sequence[LayoutCandidate], limit: int) -> List[LayoutCandidate]:
        feasible = [c for c in population if bool(getattr(c, "l2_pass", False))]
        feasible.sort(key=lambda c: (int(getattr(c, "_nsga_rank", 10**9)), -float(getattr(c, "_nsga_crowding", 0.0)), -c.layout_score))
        seen = set()
        out: List[LayoutCandidate] = []
        for c in feasible:
            sig = _candidate_signature(c, ndigits=3)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(c)
            if len(out) >= limit:
                break
        return out

    # -----------------------------
    # Main search entry used by fol.main()
    # -----------------------------

    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "staging_aware",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        rng = np.random.default_rng(seed)
        self._reset_eval_progress_stats()
        regions = self._assembly_region_candidates()

        pop_size = int(_CFG["pop"] or n_samples)
        pop_size = max(4, pop_size)
        n_offspring = int(_CFG["offspring"] or max(2, pop_size // 2))
        n_offspring = max(2, n_offspring)
        generations = max(1, int(_CFG["generations"]))

        print("\n========== NSGA-II v1 Layout Search ==========")
        print(f"pop_size={pop_size} offspring={n_offspring} generations={generations} seed={seed}")
        print(f"max_evals={_CFG.get('max_evals')} early_stop={_CFG.get('early_stop_rounds')} "
              f"cache_ndigits={_CFG.get('eval_cache_ndigits')}")
        print("objectives = [transport_cost, inverse_manipulability, grasp_risk, sequence_surrogate_cost]")
        print("note: sequence_surrogate_cost uses the current analytic layout_score in v1; GNN will replace it later.")
        print(f"L3 default/current = {'ON' if enable_l3 else 'OFF'}")

        t0 = time.time()
        population = self._init_population(rng, regions, pop_size, max_resample_layout, verbose=verbose)
        if not population:
            print("\n[FAIL] NSGA-II initialization produced no valid candidate objects.")
            return None

        # If the initialization was short, pad with evaluated random samples.
        while len(population) < pop_size and not self._eval_budget_exhausted():
            cand = self._random_candidate(rng, regions, max_resample_layout)
            if cand is None:
                break
            population.append(cand)

        population = self._environmental_selection(population, min(pop_size, len(population)))
        self._print_front_summary(0, population)

        best_score_seen = self._best_layout_score(population)
        no_improve = 0
        for gen in range(1, generations + 1):
            if self._eval_budget_exhausted():
                print(f"[NSGA-II] stop: max_evals reached ({self._nsga_eval_count}).")
                break

            self._rank_population(population)
            offspring = self._make_offspring(rng, population, regions, n_offspring)
            if not offspring:
                print("[NSGA-II] stop: no offspring generated.")
                break

            population = self._environmental_selection(list(population) + offspring, pop_size)
            self._print_front_summary(gen, population)

            cur_best = self._best_layout_score(population)
            if cur_best > best_score_seen + float(_CFG.get("min_improvement", 1e-4)):
                best_score_seen = cur_best
                no_improve = 0
            else:
                no_improve += 1

            early_stop_rounds = int(_CFG.get("early_stop_rounds", 0) or 0)
            if early_stop_rounds > 0 and no_improve >= early_stop_rounds:
                print(f"[NSGA-II] early stop: best_score not improved for {no_improve} generation(s).")
                break

        elites = self._unique_elites(population, limit=max(int(_CFG["elite_pool_size"]), int(l3_top_k)))
        if not elites:
            print("\n[FAIL] NSGA-II found no L2 feasible layout.")
            return None

        # For paper/debug readability, sort final elites by the original scalar score.
        # NSGA-II still uses Pareto ranking during evolution.
        elites_by_score = sorted(elites, key=lambda c: c.layout_score, reverse=True)
        best = elites_by_score[0]

        print("\n========== NSGA-II Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"final population    = {len(population)}")
        print(f"unique L2 elites    = {len(elites)}")
        print(f"real evaluations    = {self._nsga_eval_count}")
        print(f"eval cache hits     = {self._nsga_cache_hits}")
        self.print_search_eval_progress()
        print(f"[BEST-L2] score={best.layout_score:.4f} region={best.assembly_region_id} rc={best.assembly_region_rc}")
        print(f"  objectives  ={np.round(best._nsga_objectives, 4).tolist()}")
        print(f"  grasp_counts={best.grasp_counts}")
        print(f"  arm_choice  ={best.arm_choice}")
        print(f"  pose_tag    ={best.pose_tag}")

        # Optional L3 validation. It remains OFF by default.
        if enable_l3:
            print("\n========== Optional L3 full-process validation ==========")
            k = min(int(l3_top_k), len(elites_by_score))
            for rank, cand in enumerate(elites_by_score[:k], start=1):
                print(f"[L3] rank {rank}/{k} score={cand.layout_score:.4f} region={cand.assembly_region_id}")
                if self.validate_full_sequence_l3(cand, obstacle_mode=l3_obstacle_mode, verbose=True):
                    print(f"[OK] L3 passed rank={rank}")
                    return cand
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")
            if require_l3:
                print("\n[FAIL] L3 top-k all failed.")
                return None
            print("\n[WARN] L3 failed, require_l3=False, fallback to L2 best.")

        return best


# ============================================================
# Entry point
# ============================================================

def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = NSGA2LayoutSearcher


def _enforce_l3_default_off() -> None:
    """Force L3 to be OFF unless the user explicitly passes --enable-l3."""
    try:
        fol.DEFAULT_ENABLE_L3 = False
    except Exception:
        pass
    if "--enable-l3" in sys.argv:
        print("[nsga2-v1] L3: explicit --enable-l3 detected, L3 will be enabled.")
        return
    if "--disable-l3" not in sys.argv:
        sys.argv.append("--disable-l3")
        print("[nsga2-v1] L3 default = OFF (auto-injected --disable-l3).")


def _enforce_l3_skip_middle_plate() -> None:
    """当开启 L3 时，默认只对除 middle_plate 之外的零件做 L3 验证。

    需求：加 --enable-l3 时也要跳过 middle_plate 的 L3；不加 --enable-l3 时根本不跑 L3。
    做法：把 middle_plate 设为 L3 跳过项的默认值，并在用户未显式传 --l3-skip-parts 时
    自动注入，保证该默认行为不受基类默认值变动影响。用户仍可用 --l3-skip-parts 覆盖
    (例如换成别的模型/别的零件，或传空串 "" 表示全部都验证)。
    """
    try:
        fol.DEFAULT_L3_SKIP_PARTS = "middle_plate"
    except Exception:
        pass
    if "--l3-skip-parts" not in sys.argv:
        sys.argv.extend(["--l3-skip-parts", "middle_plate"])
        print("[nsga2-v1] L3 skip parts default = middle_plate (auto-injected --l3-skip-parts).")
    else:
        print("[nsga2-v1] L3 skip parts: explicit --l3-skip-parts detected, keep user value.")


def _consume_nsga_args() -> None:
    def get_int(name: str, key: str) -> None:
        val = fast._consume_extra_value(name)
        if val is not None:
            try:
                _CFG[key] = int(val)
            except ValueError:
                print(f"[nsga2-v1] WARN: cannot parse {name}={val!r}, keep default.")

    def get_float(name: str, key: str) -> None:
        val = fast._consume_extra_value(name)
        if val is not None:
            try:
                _CFG[key] = float(val)
            except ValueError:
                print(f"[nsga2-v1] WARN: cannot parse {name}={val!r}, keep default.")

    get_int("--nsga-pop", "pop")
    get_int("--nsga-generations", "generations")
    get_int("--nsga-offspring", "offspring")
    get_int("--nsga-elite-pool", "elite_pool_size")
    get_int("--nsga-max-evals", "max_evals")
    get_int("--nsga-early-stop", "early_stop_rounds")
    get_int("--nsga-cache-digits", "eval_cache_ndigits")
    get_int("--nsga-init-attempt-factor", "init_attempt_factor")
    get_float("--nsga-crossover-prob", "crossover_prob")
    get_float("--nsga-mutation-prob", "mutation_prob")
    get_float("--nsga-mutation-sigma", "mutation_sigma")
    get_float("--nsga-region-mutation-prob", "region_mutation_prob")
    get_float("--nsga-local-jitter-prob", "local_jitter_prob")
    get_float("--nsga-local-jitter-sigma", "local_jitter_sigma")
    get_float("--nsga-min-improvement", "min_improvement")


def main() -> None:
    _enforce_l3_default_off()
    _enforce_l3_skip_middle_plate()
    _consume_nsga_args()

    # Keep the fast script's default behavior: order-x is disabled unless the
    # user edits that file or explicitly changes the flag policy there.
    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    print("[nsga2-v1.1-speed] config:")
    for k, v in _CFG.items():
        print(f"    {k:22s} = {v}")
    print(f"[nsga2-v1] scipy.cKDTree available = {fast._HAS_KDTREE}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[nsga2-v1] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()