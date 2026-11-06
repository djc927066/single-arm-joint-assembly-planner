"""Collect single-bbox init/goal pairs for PartPlacementRanker (5000 default).

Each JSONL line = **one bbox** with:
  - Random shape/size (2–20 cm edges, 20 prototypes + jitter → high diversity)
  - One **goal** position + pose (flatsurface)
  - One primary **init** position + pose (best ranked candidate)
  - ``init_candidates_ranked``: table init options for training top-k (default up to 32)

Filters (cheap checks first, then grasp):
  - init & goal xy within work_table (footprint-aware)
  - staging / goal positions outside arm-base keepout
  - no robot-home collision at init or goal
  - init→goal common grasp required; skip if none

Example (5000 lines)::

    python -m sealp.examples.layout.collect_synthetic_bbox_layout_dataset \\
        --output-jsonl sealp/examples/layout/_output/synthetic_bbox_single_5k.jsonl \\
        --overwrite --target-kept 5000 --gen-threads 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# BLAS threads before numpy
def _configure_threads_from_argv() -> None:
    val = None
    for i, arg in enumerate(sys.argv):
        if arg == "--gen-threads" and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            break
        if arg.startswith("--gen-threads="):
            val = arg.split("=", 1)[1]
            break
    if val is None:
        return
    try:
        n = max(1, int(val))
    except ValueError:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(n)


_configure_threads_from_argv()

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
from sealp.examples.layout.generate_layout_dataset import (
    sample_station_by_mode,
    _pose_candidates_for_part,
    _to_list,
)
from sealp.examples.layout.synthetic_bbox.utils import (
    MIN_EDGE_M,
    MAX_EDGE_M,
    SyntheticAssemblySpec,
    build_single_part_spec,
    deterministic_init_anchor,
    ensure_box_assets,
)
from sealp.examples.layout.synthetic_bbox.portable import find_repo_root, relpath, write_run_manifest
from sealp.examples.layout.uniform_candidate_pool import table_anchor_grid
from find_optimal_initial_layout_tower_strict_pycharm import (
    PickPlacePlanner,
    WeightedInitialLayoutSearcher,
)
from sealp.layout.layout_robot_factory import get_layout_arm
from wrs.grasping.grasp import GraspCollection

try:
    from sealp.layout.reachability import check_pose_reachability
except Exception:  # pragma: no cover - reachability optional
    check_pose_reachability = None

DEFAULT_CONFIG = fol.DEFAULT_CONFIG
DEFAULT_OUTPUT = os.path.join(fol.DEFAULT_OUTPUT_DIR, "synthetic_bbox_single_5k.jsonl")
DEFAULT_ASSETS = os.path.join(fol.DEFAULT_OUTPUT_DIR, "synthetic_bbox_assets")
SAMPLER_VERSION = "synthetic_bbox_single_collector_v2"
SCHEMA_VERSION = "synthetic_bbox_single_v1"
PART_ID = "box_0"


def _parse_args():
    p = argparse.ArgumentParser(description="Single-bbox init/goal dataset collector")
    p.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    p.add_argument("--assets-root", default=DEFAULT_ASSETS)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--target-kept", type=int, default=5000)
    p.add_argument("--max-attempts", type=int, default=25000)
    p.add_argument("--min-edge-cm", type=float, default=MIN_EDGE_M * 100.0)
    p.add_argument("--max-edge-cm", type=float, default=MAX_EDGE_M * 100.0)
    p.add_argument("--bbox-jitter", type=float, default=0.12,
                   help="Per-edge size jitter fraction for bbox diversity")
    p.add_argument("--min-grasp-total", type=int, default=8)
    p.add_argument("--min-common-grasp", type=int, default=1)
    p.add_argument("--grid-spacing", type=float, default=0.06)
    p.add_argument("--max-init-candidates", type=int, default=32,
                   help="Ranked init candidates saved per line (train top-10)")
    p.add_argument("--min-init-candidates", type=int, default=5)
    p.add_argument("--max-station-tries", type=int, default=24,
                   help="Resample goal station when goal fails table/keepout/collision filters")
    p.add_argument("--max-grasp-samples", type=int, default=60)
    # Oracle score weights + normalizers (candidate ranking target). Defaults are aligned
    # with the downstream GA fitness: high common-grasp manipulability + SHORT arm travel.
    p.add_argument("--w-grasp", type=float, default=0.45)
    p.add_argument("--w-manip", type=float, default=0.35)
    p.add_argument("--w-dist", type=float, default=0.20)
    p.add_argument("--grasp-norm", type=float, default=20.0)
    p.add_argument("--manip-norm", type=float, default=0.05)
    p.add_argument("--dist-norm", type=float, default=0.40)
    p.add_argument("--dist-reward", choices=["near", "far"], default="near",
                   help="near: shorter init->goal distance scores higher (matches GA fitness, "
                        "default); far: legacy behaviour (farther scores higher)")
    p.add_argument("--episodes-per-geometry", type=int, default=4,
                   help="Reuse Searcher for N attempts on same bbox mesh before new geometry")
    p.add_argument("--gen-threads", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--append", action="store_true")
    p.add_argument("--cdprim-type", default="box", choices=["box", "triangles"])
    p.add_argument("--planner-obstacle-mode", default="staging_aware")
    p.add_argument("--fsync-every", type=int, default=20)
    p.add_argument("--no-manifest", action="store_true")
    return p.parse_args()


def _build_searcher(asmdef_path: str, grasp_dir: str, config_yaml: str,
                    *, cdprim_type: str, planner_obstacle_mode: str) -> WeightedInitialLayoutSearcher:
    return WeightedInitialLayoutSearcher(
        asmdef_path=asmdef_path,
        config_yaml=config_yaml,
        grasp_dir=grasp_dir,
        fixture_pos=np.array([0.36, 0.0, 0.0], dtype=float),
        fixture_rotmat=np.eye(3),
        robot_base_pos=np.zeros(3, dtype=float),
        robot_base_rotmat=np.eye(3),
        part_order=[PART_ID],
        output_name="synthetic_bbox_single",
        table_name="work_table",
        table_margin=0.06,
        table_clearance=0.01,
        grasp_map={},
        max_rot_candidates=8,
        w_grasp=0.3,
        w_manip=0.4,
        w_dist=0.1,
        w_rot=0.2,
        cdprim_type=cdprim_type,
        planner_obstacle_mode=planner_obstacle_mode,
        plan_assembly_region=True,
        preassemble_first_part=False,
        use_flatsurface=True,
        check_l2_pick_quick_motion=False,
    )


def _best_cand_for_rotmat(searcher, pid: str, rotmat: np.ndarray):
    """Pick rot candidate whose orientation best matches ``rotmat`` (for goal footprint)."""
    cands = list(searcher.rot_cands.get(pid, []) or [])
    if not cands:
        return None
    target = np.asarray(rotmat, dtype=float).reshape(3, 3)
    best = cands[0]
    best_err = float("inf")
    for cand in cands:
        r = np.asarray(getattr(cand, "rotmat", np.eye(3)), dtype=float).reshape(3, 3)
        err = float(np.linalg.norm(r - target))
        if err < best_err:
            best_err = err
            best = cand
    return best


def _table_bounds_reason(searcher, pid: str, xy: np.ndarray, cand) -> Optional[str]:
    """Part center xy must keep footprint inside work_table."""
    (xlo, xhi), (ylo, yhi) = searcher._xy_bounds_for_part_and_cand(pid, cand)
    x, y = float(xy[0]), float(xy[1])
    eps = 1e-9
    if x < xlo - eps or x > xhi + eps or y < ylo - eps or y > yhi + eps:
        return (
            f"{pid} outside work_table: xy=({x:.4f},{y:.4f}), "
            f"x=[{xlo:.4f},{xhi:.4f}], y=[{ylo:.4f},{yhi:.4f}]"
        )
    return None


def _fast_pose_reject_reason(searcher, pid: str, xy: np.ndarray, cand) -> Optional[str]:
    """Inexpensive pre-filter: upright, table bounds, arm-base keepout."""
    if searcher._upright_hard_constraint_reason(pid, cand):
        return "upright_constraint"
    hit = _table_bounds_reason(searcher, pid, xy, cand)
    if hit:
        return hit
    if searcher._staging_arm_keepout_reason(pid, xy, cand):
        return "staging_arm_keepout"
    return None


def _robot_collision_at_pose(searcher, pid: str, pos: np.ndarray, rotmat: np.ndarray) -> Optional[str]:
    """Place part at ``pos``/``rotmat`` and test robot-home collision."""
    cm = searcher.staging_models.get(pid)
    if cm is None:
        return None
    old_pos = np.asarray(cm.pos, dtype=float).copy()
    old_rot = np.asarray(cm.rotmat, dtype=float).copy()
    try:
        cm.pos = np.asarray(pos, dtype=float).copy()
        cm.rotmat = np.asarray(rotmat, dtype=float).reshape(3, 3).copy()
        return searcher._robot_home_collision_reason(active_pids=[pid])
    finally:
        cm.pos = old_pos
        cm.rotmat = old_rot


def _goal_pose_reject_reason(searcher, pid: str) -> Optional[str]:
    """Validate goal world pose: table, keepout, robot-home collision."""
    if pid not in searcher.world_poses:
        return f"{pid} missing goal world pose"
    gp, gr = searcher.world_poses[pid]
    goal_xy = np.asarray(gp[:2], dtype=float)
    cand = _best_cand_for_rotmat(searcher, pid, gr)
    if cand is None:
        return f"{pid} no rot candidates for goal footprint"

    hit = _fast_pose_reject_reason(searcher, pid, goal_xy, cand)
    if hit:
        return f"goal {hit}"

    hit = _robot_collision_at_pose(searcher, pid, gp, gr)
    if hit:
        return f"goal robot_home_collision: {hit}"
    return None


def _filter_init_anchors(searcher, pid: str, anchors: List[np.ndarray]) -> List[np.ndarray]:
    """Drop init anchors that fail cheap table / keepout checks for any legal pose."""
    cands = list(searcher.rot_cands.get(pid, []) or [])
    if not cands:
        return []
    out: List[np.ndarray] = []
    for xy in anchors:
        if any(_fast_pose_reject_reason(searcher, pid, xy, cand) is None for cand in cands):
            out.append(xy)
    return out


def _sample_feasible_goal_station(searcher, rng: np.random.Generator, pid: str, *,
                                  progress: float, max_tries: int):
    """Sample goal station; reject early if goal pose fails table/keepout/collision."""
    for _ in range(int(max_tries)):
        station = sample_station_by_mode(
            searcher, rng, mode="center_continuous",
            progress=progress, center_bias=0.65, center_sigma_frac=0.28,
            max_tries=1,
        )
        if station is None:
            continue
        region_id, rc, goal_station = station
        searcher._set_assembly_station(goal_station)
        if _goal_pose_reject_reason(searcher, pid) is None:
            return region_id, rc, goal_station
    return None


def _part_grasp_total(searcher, pid: str) -> int:
    try:
        gc = searcher._grasp_collection(pid)
        return len(gc) if gc is not None else 0
    except Exception:
        return 0


def _build_goal_reason_ctx(searcher, pid: str, goal_pos: np.ndarray, goal_rot: np.ndarray) -> Optional[Dict]:
    """Precompute goal-side grasp reasoning ONCE (goal is fixed for the whole record).

    The set of common grasps between an init pose and the goal pose equals
    ``feasible(init) ∩ feasible(goal)``. Since ``goal`` is identical for every init
    candidate, we compute ``feasible(goal)`` a single time and reuse a filtered grasp
    collection per candidate. This is *lossless* (same common-grasp counts) but removes
    the redundant per-candidate goal-pose IK and shrinks the per-candidate grasp set.

    Also caches the fixed goal-side manipulability and reuses one PickPlacePlanner/arm
    instead of re-creating them for every candidate.
    """
    gc_full = searcher._grasp_collection(pid)
    if gc_full is None or len(gc_full) == 0:
        return None
    obs = searcher._planner_obstacles([], current_pid=pid, placed=set())
    ee = getattr(gc_full, "end_effector", None)

    arms: Dict[str, Dict] = {}
    for arm_tag in searcher._arm_order(pid):
        arm = get_layout_arm(searcher.robot, arm_tag, single_arm=searcher.single_arm_mode)
        planner = PickPlacePlanner(robot=arm)
        try:
            goal_gids = planner.reason_common_gids(
                grasp_collection=gc_full,
                goal_pose_list=[(goal_pos, goal_rot)],
                obstacle_list=obs,
            )
        except Exception:
            goal_gids = None
        if not goal_gids:
            continue
        gc_goal = GraspCollection(end_effector=ee, grasp_list=list(gc_full[list(goal_gids)]))
        place_manip = 0.0
        if check_pose_reachability is not None:
            try:
                place = check_pose_reachability(arm, goal_pos, goal_rot, gc_full, obs, max_grasps=5)
                place_manip = float(getattr(place, "best_manipulability", 0.0))
            except Exception:
                place_manip = 0.0
        arms[arm_tag] = {
            "arm_tag": arm_tag,
            "arm": arm,
            "planner": planner,
            "gc_goal": gc_goal,
            "place_manip": place_manip,
        }

    if not arms:
        return None
    return {"arms": arms, "obs": obs, "gc_full": gc_full}


def _combined_manip(entry: Dict, sp: np.ndarray, sr: np.ndarray, gc_full, obs) -> float:
    """Replicate ``_endpoint_manip`` (mean of positive pick/place manip); place cached."""
    if check_pose_reachability is None:
        return 0.0
    try:
        pick = check_pose_reachability(entry["arm"], sp, sr, gc_full, obs, max_grasps=5)
        pick_m = float(getattr(pick, "best_manipulability", 0.0))
    except Exception:
        pick_m = 0.0
    place_m = float(entry.get("place_manip", 0.0))
    vals = [v for v in (pick_m, place_m) if np.isfinite(v) and v > 0.0]
    return float(np.mean(vals)) if vals else 0.0


DEFAULT_SCORE_CFG: Dict = {
    "w_grasp": 0.45, "w_manip": 0.35, "w_dist": 0.20,
    "grasp_norm": 20.0, "manip_norm": 0.05, "dist_norm": 0.40,
    "dist_reward": "near",
}


def _oracle_score(grasp_count: int, manip: float, dist: float, cfg: Dict) -> float:
    """Weighted oracle score. ``dist_reward='near'`` rewards short init->goal travel
    (aligned with the GA fitness); 'far' keeps the legacy farther-is-better behaviour."""
    dist_term = min(dist / max(cfg["dist_norm"], 1e-9), 1.0)
    if cfg.get("dist_reward", "near") == "near":
        dist_term = 1.0 - dist_term
    return (
        cfg["w_grasp"] * min(grasp_count / max(cfg["grasp_norm"], 1e-9), 1.0)
        + cfg["w_manip"] * min(manip / max(cfg["manip_norm"], 1e-9), 1.0)
        + cfg["w_dist"] * dist_term
    )


def _score_init_candidate(searcher, pid: str, xy: np.ndarray, cand, ctx: Dict,
                          *, goal_pos: np.ndarray, goal_rot: np.ndarray,
                          score_cfg: Dict = DEFAULT_SCORE_CFG) -> Optional[Dict]:
    if _fast_pose_reject_reason(searcher, pid, xy, cand):
        return None
    searcher._apply_staging_pose(pid, xy, cand)
    if searcher._pairwise_collision(active_pids=[pid]):
        return None
    if searcher._mesh_clearance_reason(active_pids=[pid]):
        return None
    if searcher._robot_home_collision_reason(active_pids=[pid]):
        return None

    sp = searcher.staging_models[pid].pos.copy()
    sr = searcher.staging_models[pid].rotmat.copy()
    obs = ctx["obs"]

    best_gid_count = 0
    best_entry = None
    for entry in ctx["arms"].values():
        try:
            gids = entry["planner"].reason_common_gids(
                grasp_collection=entry["gc_goal"],
                goal_pose_list=[(sp, sr)],
                obstacle_list=obs,
            )
        except Exception:
            gids = None
        n = len(gids) if gids else 0
        if n > best_gid_count:
            best_gid_count = n
            best_entry = entry
    if best_gid_count < 1 or best_entry is None:
        return None

    manip = _combined_manip(best_entry, sp, sr, ctx["gc_full"], obs)
    dist = float(np.linalg.norm(sp[:2] - goal_pos[:2]))
    score = _oracle_score(best_gid_count, manip, dist, score_cfg)
    return {
        "xy": _to_list(xy),
        "init_pos": _to_list(sp),
        "init_rotmat": _to_list(sr),
        "pose_tag": str(getattr(cand, "tag", "unknown")),
        "rot_name": str(getattr(cand, "rot_name", "unknown")),
        "arm_choice": str(best_entry["arm_tag"]),
        "common_grasp_count": int(best_gid_count),
        "manipulability": float(manip),
        "dist_to_goal": float(dist),
        "score": float(score),
    }


def _rank_init_candidates(searcher, pid: str, anchors: List[np.ndarray], max_keep: int, *,
                          goal_pos: np.ndarray, goal_rot: np.ndarray,
                          score_cfg: Dict = DEFAULT_SCORE_CFG) -> List[Dict]:
    # Goal is fixed for this record -> precompute goal-feasible grasps once. If no grasp
    # can reach the goal, no init candidate can share one, so the whole record is skipped
    # cheaply instead of re-testing every anchor.
    ctx = _build_goal_reason_ctx(searcher, pid, goal_pos, goal_rot)
    if ctx is None:
        return []
    ranked: List[Dict] = []
    for xy in anchors:
        for cand in searcher.rot_cands.get(pid, []) or []:
            rec = _score_init_candidate(
                searcher, pid, xy, cand, ctx, goal_pos=goal_pos, goal_rot=goal_rot,
                score_cfg=score_cfg,
            )
            if rec is not None:
                ranked.append(rec)
    ranked.sort(key=lambda r: (-float(r["score"]), -int(r["common_grasp_count"])))
    out: List[Dict] = []
    seen = set()
    for rec in ranked:
        key = (tuple(round(x, 4) for x in rec["xy"]), rec["pose_tag"])
        if key in seen:
            continue
        seen.add(key)
        item = dict(rec)
        item["rank"] = len(out) + 1
        out.append(item)
        if len(out) >= int(max_keep):
            break
    return out


def _collect_single_record(searcher,
                           spec: SyntheticAssemblySpec,
                           rng: np.random.Generator,
                           *,
                           seed: int,
                           sample_index: int,
                           grid_spacing: float,
                           max_init_candidates: int,
                           min_init_candidates: int,
                           min_grasp_total: int,
                           max_station_tries: int,
                           score_cfg: Dict = DEFAULT_SCORE_CFG) -> Optional[Dict]:
    pid = PART_ID
    if _part_grasp_total(searcher, pid) < int(min_grasp_total):
        return None

    progress = float(sample_index) / max(1, sample_index + 1)
    station = _sample_feasible_goal_station(
        searcher, rng, pid, progress=progress, max_tries=max_station_tries,
    )
    if station is None:
        return None
    region_id, rc, goal_station = station

    gp, gr = searcher.world_poses[pid]
    goal_pos = np.asarray(gp, dtype=float).copy()
    goal_rot = np.asarray(gr, dtype=float).copy()
    goal_cand = _best_cand_for_rotmat(searcher, pid, goal_rot)

    anchors = _filter_init_anchors(
        searcher, pid, table_anchor_grid(searcher, spacing=float(grid_spacing)),
    )
    if not anchors:
        return None

    init_hint = deterministic_init_anchor(pid, goal_station[:2], anchors)
    candidates = _rank_init_candidates(
        searcher, pid, anchors, max_init_candidates,
        goal_pos=goal_pos, goal_rot=goal_rot, score_cfg=score_cfg,
    )
    if len(candidates) < int(min_init_candidates):
        return None

    best = candidates[0]
    if int(best.get("common_grasp_count", 0)) < 1:
        return None
    init_pos = np.asarray(best["init_pos"], dtype=float)
    init_rot = np.asarray(best["init_rotmat"], dtype=float).reshape(3, 3)

    rc0 = searcher.rot_cands.get(pid, [None])[0]
    pose_candidates = _pose_candidates_for_part(searcher, pid, None, None)
    extent = _to_list(getattr(rc0, "extent", spec.parts[0].extent))
    footprint = _to_list(getattr(rc0, "footprint", [0.05, 0.05]))

    part_obj = {
        "part_id": pid,
        "order_index": 0,
        "is_first": False,
        "extent": extent,
        "footprint": footprint,
        "goal_pos": _to_list(goal_pos),
        "goal_rotmat": _to_list(goal_rot),
        "init_pos": _to_list(init_pos),
        "init_rotmat": _to_list(init_rot),
        "init_pose_tag": best.get("pose_tag"),
        "init_rot_name": best.get("rot_name"),
        "goal_pose_tag": str(getattr(goal_cand, "tag", "unknown")) if goal_cand else None,
        "goal_rot_name": str(getattr(goal_cand, "rot_name", "unknown")) if goal_cand else None,
        "init_pos_balanced": _to_list(init_hint),
        "grasp_total": float(_part_grasp_total(searcher, pid)),
        "pose_candidates": pose_candidates,
        "shape_type": spec.parts[0].shape_type,
        "prototype_idx": int(spec.parts[0].prototype_idx),
    }

    signature = f"{SAMPLER_VERSION}:{spec.cache_key}"
    record: Dict = {
        "schema_version": SCHEMA_VERSION,
        "geometry_domain": "synthetic_box_single",
        "sampler_version": SAMPLER_VERSION,
        "generation_signature": signature,
        "sample_id": int(hashlib.sha256(
            f"{signature}:{seed}:{sample_index}".encode("utf-8")
        ).hexdigest()[:15], 16),
        "sample_index": int(sample_index),
        "seed": int(seed),
        "bbox_id": spec.assembly_id,
        "assembly_cache_key": spec.cache_key,
        "shape_type": spec.parts[0].shape_type,
        "prototype_idx": int(spec.parts[0].prototype_idx),
        "extent": extent,
        "footprint": footprint,
        "init_pos": part_obj["init_pos"],
        "init_rotmat": part_obj["init_rotmat"],
        "init_pose_tag": part_obj["init_pose_tag"],
        "init_rot_name": part_obj["init_rot_name"],
        "goal_pos": part_obj["goal_pos"],
        "goal_rotmat": part_obj["goal_rotmat"],
        "init_pos_balanced": part_obj["init_pos_balanced"],
        "pose_candidates": pose_candidates,
        "init_candidates_ranked": candidates,
        "num_parts": 1,
        "part_order": [pid],
        "parts": [part_obj],
        "ranking_targets": [{"part_id": pid, "candidates": candidates}],
        "assembly_station_pos": _to_list(goal_station),
        "assembly_region_id": region_id,
        "assembly_region_rc": list(rc),
        "table_x_range": _to_list(searcher.table_x_range),
        "table_y_range": _to_list(searcher.table_y_range),
        "table_top_z": float(searcher.table_top_z),
        "feasible": True,
        "best_init_score": float(best["score"]),
        "goal_common_grasp_count": int(best["common_grasp_count"]),
        "score_cfg": dict(score_cfg),
        "collection_filters": {
            "work_table_bounds": True,
            "staging_arm_keepout": True,
            "robot_home_collision": True,
            "common_grasp_init_goal": True,
        },
    }
    return record


def main():
    args = _parse_args()
    if args.overwrite and args.append:
        raise SystemExit("不能同时 --overwrite 与 --append")
    if not args.overwrite and not args.append and os.path.isfile(args.output_jsonl):
        raise SystemExit(f"输出已存在: {args.output_jsonl}，请 --overwrite 或 --append")

    if int(args.gen_threads) > 0:
        n_thr = int(args.gen_threads)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(n_thr)

    repo_root = find_repo_root(_THIS_DIR)
    min_edge = float(args.min_edge_cm) / 100.0
    max_edge = float(args.max_edge_cm) / 100.0
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    mode = "w" if args.overwrite or not os.path.isfile(args.output_jsonl) else "a"

    score_cfg = {
        "w_grasp": float(args.w_grasp), "w_manip": float(args.w_manip),
        "w_dist": float(args.w_dist), "grasp_norm": float(args.grasp_norm),
        "manip_norm": float(args.manip_norm), "dist_norm": float(args.dist_norm),
        "dist_reward": str(args.dist_reward),
    }

    if not args.no_manifest:
        manifest_path = os.path.splitext(os.path.abspath(args.output_jsonl))[0] + "_manifest.json"
        write_run_manifest(manifest_path, {
            "task": "synthetic_bbox_single_collection",
            "schema": SCHEMA_VERSION,
            "description": "One JSONL line = one bbox with init+goal poses and ranked init candidates",
            "repo_root": repo_root,
            "output_jsonl": relpath(args.output_jsonl, repo_root),
            "assets_root": relpath(args.assets_root, repo_root),
            "target_kept": int(args.target_kept),
            "score_cfg": score_cfg,
            "inference": "Fixed goal_pos/goal_rotmat per part → model ranks init_candidates → top-10",
        })

    kept = 0
    attempts = 0
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    cached_searcher: Optional[WeightedInitialLayoutSearcher] = None
    cached_spec: Optional[SyntheticAssemblySpec] = None
    attempts_on_geometry = 0

    with open(args.output_jsonl, mode, encoding="utf-8") as fout:
        while kept < int(args.target_kept) and attempts < int(args.max_attempts):
            seed = seeds[attempts % len(seeds)]
            rng = np.random.default_rng(seed * 1000003 + attempts)

            if cached_searcher is None or attempts_on_geometry >= int(args.episodes_per_geometry):
                cached_spec = build_single_part_spec(
                    rng, min_edge=min_edge, max_edge=max_edge,
                    jitter_frac=float(args.bbox_jitter),
                )
                assets = ensure_box_assets(
                    cached_spec, args.assets_root,
                    max_grasp_samples=int(args.max_grasp_samples),
                )
                try:
                    cached_searcher = _build_searcher(
                        assets["asmdef_path"], assets["grasp_dir"], args.config,
                        cdprim_type=args.cdprim_type,
                        planner_obstacle_mode=args.planner_obstacle_mode,
                    )
                except Exception as exc:
                    attempts += 1
                    cached_searcher = None
                    print(f"[skip] searcher: {type(exc).__name__}: {exc}")
                    continue
                attempts_on_geometry = 0
                if kept % 50 == 0:
                    print(f"[geom] new bbox cache_key={cached_spec.cache_key} "
                          f"shape={cached_spec.parts[0].shape_type}")

            assert cached_searcher is not None and cached_spec is not None
            attempts_on_geometry += 1

            try:
                record = _collect_single_record(
                    cached_searcher, cached_spec, rng,
                    seed=seed, sample_index=attempts,
                    grid_spacing=args.grid_spacing,
                    max_init_candidates=args.max_init_candidates,
                    min_init_candidates=args.min_init_candidates,
                    min_grasp_total=args.min_grasp_total,
                    max_station_tries=args.max_station_tries,
                    score_cfg=score_cfg,
                )
            except Exception as exc:
                attempts += 1
                print(f"[skip] record: {type(exc).__name__}: {exc}")
                continue

            attempts += 1
            if record is None:
                continue

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.fsync_every > 0 and kept % int(args.fsync_every) == 0:
                fout.flush()
                os.fsync(fout.fileno())
            kept += 1
            if kept % 25 == 0 or kept == int(args.target_kept):
                print(f"[collect] kept={kept}/{args.target_kept} attempts={attempts} "
                      f"yield={kept/max(attempts,1):.2%} elapsed={time.time()-t0:.1f}s")

    print(f"Done. kept={kept} attempts={attempts} -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
