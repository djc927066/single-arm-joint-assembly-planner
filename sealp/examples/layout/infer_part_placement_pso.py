"""Inference: PartPlacementRanker top-k per part + PSO full-layout search.

Workflow:
  1. User specifies goal assembly station + init_pos hints (region-balanced anchors)
  2. Build candidate pools on table (flatsurface poses × grid anchors)
  3. NN ranks candidates → top_k per part
  4. PSO searches over the Cartesian product of per-part top_k choices
  5. Real evaluate_layout validates the best composite layout

Example::

    python -m sealp.examples.layout.infer_part_placement_pso \\
        --checkpoint checkpoints/part_placement_ranker/part_placement_ranker_best.pt \\
        --asmdef sealp/assembly_sequence/_demo_output/topdown_tower.asmdef \\
        --grasp-dir sealp/examples/grasp/tower_grasp \\
        --top-k 5 --pso-particles 16 --pso-iters 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
from sealp.examples.layout.generate_layout_dataset import _pose_candidates_for_part
from sealp.examples.layout.synthetic_bbox.utils import deterministic_init_anchor
from sealp.examples.layout.uniform_candidate_pool import table_anchor_grid
from find_optimal_initial_layout_tower_strict_pycharm import LayoutCandidate, WeightedInitialLayoutSearcher
from layout_learning import features as F
from layout_learning.models.part_placement_ranker import PartPlacementRankerNet
from layout_learning.part_placement_dataset import POSE_FEAT_DIM, _candidate_feature, _global_vector, _part_static_vector, _table_bounds


def _parse_vec3(text: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    vals = [float(x.strip()) for x in text.split(",")]
    return np.asarray(vals[:3] if len(vals) >= 3 else default, dtype=float)


def _parse_args():
    p = argparse.ArgumentParser(description="Part placement ranker + PSO inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=fol.DEFAULT_CONFIG)
    p.add_argument("--asmdef", default=fol.DEFAULT_ASMDEF)
    p.add_argument("--grasp-dir", default=fol.DEFAULT_GRASP_DIR)
    p.add_argument("--part-order", default="")
    p.add_argument("--goal-pos", default="", help="Assembly station xyz, e.g. 0.36,0.0,0.0")
    p.add_argument("--init-pos-json", default="",
                   help='Optional per-part init hints JSON, e.g. {"post_br":[0.12,-0.1]}')
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--grid-spacing", type=float, default=0.05)
    p.add_argument("--pso-particles", type=int, default=16)
    p.add_argument("--pso-iters", type=int, default=12)
    p.add_argument("--output-json", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _build_searcher(args) -> WeightedInitialLayoutSearcher:
    part_order = None
    if args.part_order.strip():
        part_order = [x.strip() for x in args.part_order.split(",") if x.strip()]
    return WeightedInitialLayoutSearcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        fixture_pos=_parse_vec3(args.goal_pos, (0.36, 0.0, 0.0)),
        fixture_rotmat=np.eye(3),
        robot_base_pos=np.zeros(3),
        robot_base_rotmat=np.eye(3),
        part_order=part_order,
        output_name="part_placement_pso",
        table_name="work_table",
        table_margin=0.06,
        table_clearance=0.01,
        grasp_map={},
        max_rot_candidates=8,
        w_grasp=0.3,
        w_manip=0.4,
        w_dist=0.1,
        w_rot=0.2,
        cdprim_type="box",
        planner_obstacle_mode="staging_aware",
        plan_assembly_region=True,
        use_flatsurface=True,
        check_l2_pick_quick_motion=False,
    )


def _build_sample_dict(searcher, station_pos: np.ndarray) -> Dict:
    parts = []
    init_map = {}
    for idx, pid in enumerate(searcher.part_order):
        rc0 = searcher.rot_cands.get(pid, [None])[0]
        gp, gr = searcher.world_poses[pid]
        parts.append({
            "part_id": pid,
            "order_index": idx,
            "is_first": bool(pid == searcher._first_part_id()),
            "extent": np.asarray(getattr(rc0, "extent", [0, 0, 0]), dtype=float).tolist(),
            "footprint": np.asarray(getattr(rc0, "footprint", [0.05, 0.05]), dtype=float).tolist(),
            "goal_pos": np.asarray(gp, dtype=float).tolist(),
            "goal_rotmat": np.asarray(gr, dtype=float).reshape(-1).tolist(),
            "grasp_total": 20.0,
            "topdown_count": 5.0,
            "pose_candidates": _pose_candidates_for_part(searcher, pid, None, None),
        })
    return {
        "schema_version": "synthetic_bbox_v1",
        "num_parts": len(parts),
        "parts": parts,
        "assembly_station_pos": station_pos.tolist(),
        "table_x_range": list(searcher.table_x_range),
        "table_y_range": list(searcher.table_y_range),
        "init_pos_balanced": init_map,
    }


def _enumerate_part_candidates(searcher, pid: str, init_hint: Optional[np.ndarray],
                             grid_spacing: float) -> List[Dict]:
    anchors = table_anchor_grid(searcher, spacing=float(grid_spacing))
    placed = set()
    first = searcher._first_part_id()
    if first:
        placed.add(first)
    out: List[Dict] = []
    for xy in anchors:
        for cand in searcher.rot_cands.get(pid, []) or []:
            if searcher._staging_arm_keepout_reason(pid, xy, cand):
                continue
            searcher._apply_staging_pose(pid, xy, cand)
            if searcher._pairwise_collision(active_pids=[pid]):
                continue
            if searcher._mesh_clearance_reason(active_pids=[pid]):
                continue
            out.append({
                "xy": np.asarray(xy, dtype=float)[:2].tolist(),
                "pose_tag": str(getattr(cand, "tag", "unknown")),
                "rot_name": str(getattr(cand, "rot_name", "unknown")),
                "score": 0.0,
            })
    return out


def _rank_with_model(model: PartPlacementRankerNet,
                     sample: Dict,
                     part: Dict,
                     pid: str,
                     candidates: List[Dict],
                     init_hint: Optional[List[float]],
                     device: str,
                     top_k: int) -> List[Dict]:
    bounds = _table_bounds(sample)
    cand_feat = np.zeros((len(candidates), POSE_FEAT_DIM), dtype=np.float32)
    for i, c in enumerate(candidates):
        cand_feat[i] = _candidate_feature(c["xy"], init_hint, i, 0.0, bounds)
    part_static = _part_static_vector(part, sample, int(sample["num_parts"]))
    global_feat = _global_vector(sample)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(part_static).unsqueeze(0).to(device),
            torch.from_numpy(global_feat).unsqueeze(0).to(device),
            torch.from_numpy(cand_feat).unsqueeze(0).to(device),
            torch.ones(len(candidates)).unsqueeze(0).to(device),
        )["cand_logits"].squeeze(0).cpu().numpy()
    order = np.argsort(-logits)
    ranked = []
    for j in order[: int(top_k)]:
        rec = dict(candidates[int(j)])
        rec["nn_score"] = float(logits[int(j)])
        rec["rank"] = len(ranked) + 1
        ranked.append(rec)
    return ranked


def _pso_search(searcher,
                part_choices: Dict[str, List[Dict]],
                *,
                particles: int,
                iters: int,
                rng: np.random.Generator) -> Tuple[Optional[LayoutCandidate], float]:
    pids = [pid for pid in searcher.part_order if pid in part_choices and part_choices[pid]]
    if not pids:
        return None, -1e9

    def _decode(indices: np.ndarray) -> LayoutCandidate:
        xy = {}
        for i, pid in enumerate(pids):
            choice = part_choices[pid][int(indices[i]) % len(part_choices[pid])]
            xy[pid] = np.asarray(choice["xy"], dtype=float)[:2]
        for pid in searcher.part_order:
            if pid not in xy:
                if pid == searcher._first_part_id():
                    continue
                free = searcher.sample_collision_free_xy(rng)
                if free is None or pid not in free:
                    xy[pid] = np.array([0.2, 0.0], dtype=float)
                else:
                    xy[pid] = free[pid]
        return LayoutCandidate(xy=xy)

    dim = len(pids)
    lows = np.zeros(dim)
    highs = np.array([max(len(part_choices[pid]) - 1, 0) for pid in pids], dtype=float)
    if np.all(highs <= 0):
        cand = _decode(np.zeros(dim, dtype=int))
        ok = searcher.evaluate_layout(cand)
        return (cand, float(cand.layout_score)) if ok else (None, -1e9)

    x = rng.uniform(lows, np.maximum(highs, 1.0), size=(particles, dim))
    v = rng.normal(0, 0.3, size=(particles, dim))
    pbest = x.copy()
    pbest_fit = np.full(particles, -1e9)
    gbest = pbest[0].copy()
    gbest_fit = -1e9

    def _fit(vec: np.ndarray) -> float:
        idx = np.clip(np.round(vec).astype(int), lows.astype(int), highs.astype(int))
        layout = _decode(idx)
        if searcher.evaluate_layout(layout):
            return float(layout.layout_score)
        return -1e9 + float(np.sum(idx))

    for _ in range(int(iters)):
        for i in range(particles):
            f = _fit(x[i])
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i] = x[i].copy()
            if f > gbest_fit:
                gbest_fit = f
                gbest = x[i].copy()
        r1, r2 = rng.random((2, particles, dim))
        w, c1, c2 = 0.6, 1.4, 1.4
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        x = np.clip(x + v, lows, np.maximum(highs, 1.0))

    best = _decode(np.clip(np.round(gbest).astype(int), lows.astype(int), highs.astype(int)))
    if searcher.evaluate_layout(best):
        return best, float(best.layout_score)
    return None, gbest_fit


def main():
    args = _parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = PartPlacementRankerNet(
        hidden=int(ckpt.get("hidden", 128)),
        dropout=float(ckpt.get("dropout", 0.15)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    searcher = _build_searcher(args)
    station = _parse_vec3(args.goal_pos, (0.36, 0.0, 0.0))
    station[2] = float(searcher.table_top_z)
    searcher._set_assembly_station(station)

    init_hints = {}
    if args.init_pos_json.strip():
        init_hints = json.loads(args.init_pos_json)

    sample = _build_sample_dict(searcher, station)
    anchors = table_anchor_grid(searcher, spacing=float(args.grid_spacing))

    part_topk: Dict[str, List[Dict]] = {}
    first = searcher._first_part_id()
    for pid in searcher.part_order:
        if pid == first:
            continue
        hint = init_hints.get(pid)
        if hint is None:
            hint = deterministic_init_anchor(pid, station[:2], anchors).tolist()
        part = next(p for p in sample["parts"] if p["part_id"] == pid)
        part["init_pos_balanced"] = hint
        cands = _enumerate_part_candidates(searcher, pid, np.asarray(hint), args.grid_spacing)
        if not cands:
            print(f"[warn] no coarse candidates for {pid}")
            continue
        ranked = _rank_with_model(
            model, sample, part, pid, cands, hint, args.device, args.top_k,
        )
        part_topk[pid] = ranked
        print(f"[rank] {pid}: top-{len(ranked)} from {len(cands)} candidates")

    rng = np.random.default_rng(0)
    best, score = _pso_search(
        searcher, part_topk,
        particles=int(args.pso_particles),
        iters=int(args.pso_iters),
        rng=rng,
    )

    result = {
        "goal_pos": station.tolist(),
        "init_pos_hints": init_hints,
        "part_topk": part_topk,
        "best_layout_xy": {k: v.tolist() for k, v in best.xy.items()} if best else {},
        "layout_score": float(score),
        "l2_pass": best is not None,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
