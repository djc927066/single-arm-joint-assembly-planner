"""Inference: fixed goal → top-k init positions per bbox (variable N parts).

Input JSON ``parts[]``: each with extent, goal_pos, goal_rotmat, pose_candidates.
Optional ``init_candidates`` per part or ``table_x_range`` for auto grid.

Output: ``per_part_ranking[part_id]`` = top-k (init_pos, init_rotmat, pose_tag, score).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning import features as F
from layout_learning.models.part_placement_ranker import PartPlacementRankerNet
from layout_learning.part_placement_dataset import (
    MAX_CANDIDATES,
    POSE_FEAT_DIM,
    _candidate_feature,
    _global_vector,
    _part_static_vector,
    _table_bounds,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Part placement ranker inference (fixed goal → top-k init)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-json", required=True,
                   help='{"num_parts":N,"parts":[{part_id,extent,goal_pos,goal_rotmat,pose_candidates,...}],'
                          '"table_x_range":[...],"table_y_range":[...]}')
    p.add_argument("--candidates-json", default="",
                   help="Optional {part_id: [candidate dicts with xy, pose_tag, ...]}")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-json", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # --- GA first-generation assembly (fixed assembly center -> M full layouts) ---
    p.add_argument("--assembly-center", default="",
                   help='Fixed assembly center "x,y,z". If a part lacks goal_pos but has '
                        '"goal_offset", goal_pos = center + goal_offset.')
    p.add_argument("--population-size", type=int, default=0,
                   help="If >0, assemble this many full-layout individuals (one init per "
                        "part) as the GA first generation from each part's top-k.")
    p.add_argument("--population-json", default="",
                   help="Where to write the assembled GA first generation.")
    p.add_argument("--population-seed", type=int, default=0)
    p.add_argument("--overlap-margin", type=float, default=0.005,
                   help="Min gap (m) between two parts' staging footprints when assembling.")
    return p.parse_args()


def _default_candidates(sample: Dict, part: Dict) -> List[Dict]:
    pid = str(part["part_id"])
    blocks = sample.get("ranking_targets", []) or []
    for b in blocks:
        if str(b.get("part_id")) == pid:
            return list(b.get("candidates", []) or [])
    ranked = sample.get("init_candidates_ranked")
    if ranked and len(sample.get("parts", [])) == 1:
        return list(ranked)
    return []


def _grid_candidates(sample: Dict, spacing: float = 0.06) -> List[List[float]]:
    bounds = _table_bounds(sample)
    xlo, xhi, ylo, yhi = bounds
    step = max(float(spacing), 0.04)
    xs = np.arange(xlo, xhi + 1e-9, step)
    ys = np.arange(ylo, yhi + 1e-9, step)
    grid = [[float(x), float(y)] for y in ys for x in xs]
    return grid[:MAX_CANDIDATES]


def _rank_part(model, sample, part, candidates, device, top_k, init_hint=None):
    if not candidates:
        return []
    bounds = _table_bounds(sample)
    pid = str(part["part_id"])
    if init_hint is None:
        init_hint = part.get("init_pos_balanced") or (sample.get("init_pos_balanced") or {}).get(pid)

    n_c = min(len(candidates), MAX_CANDIDATES)
    cand_feat = np.zeros((1, MAX_CANDIDATES, POSE_FEAT_DIM), dtype=np.float32)
    cand_mask = np.zeros((1, MAX_CANDIDATES), dtype=np.float32)
    for i, c in enumerate(candidates[:MAX_CANDIDATES]):
        xy = c.get("xy") or (c.get("init_pos", [0, 0])[:2])
        cand_feat[0, i] = _candidate_feature(xy, init_hint, i, float(c.get("score", 0)), bounds)
        cand_mask[0, i] = 1.0

    num_parts = int(sample.get("num_parts", len(sample.get("parts", [part]))))
    part_static = _part_static_vector(part, sample, max(num_parts, 1))
    global_feat = _global_vector(sample)

    with torch.no_grad():
        out = model(
            torch.from_numpy(part_static).unsqueeze(0).to(device),
            torch.from_numpy(global_feat).unsqueeze(0).to(device),
            torch.from_numpy(cand_feat).to(device),
            torch.from_numpy(cand_mask).to(device),
        )
    scores = out["cand_scores"].squeeze(0).cpu().numpy()
    order = np.argsort(-scores[:n_c])
    k = min(int(top_k), n_c)

    ranked = []
    for rank, idx in enumerate(order[:k], start=1):
        c = dict(candidates[int(idx)])
        c["nn_score"] = float(scores[int(idx)])
        c["rank"] = rank
        if "init_pos" not in c and "xy" in c:
            z = float(sample.get("table_top_z", 0.0))
            c["init_pos"] = [c["xy"][0], c["xy"][1], z]
        ranked.append(c)
    return ranked


def _resolve_goals_from_center(sample: Dict, center: Optional[List[float]]) -> None:
    """When a fixed assembly center is given, set it as the station and derive any
    missing per-part ``goal_pos`` from ``goal_offset`` (goal_pos = center + offset)."""
    if center is None:
        return
    c = np.asarray(center, dtype=float)
    sample["assembly_station_pos"] = [float(c[0]), float(c[1]), float(c[2] if c.size > 2 else 0.0)]
    for part in sample.get("parts", []):
        if "goal_pos" in part and part["goal_pos"] is not None:
            continue
        off = part.get("goal_offset")
        if off is not None:
            o = np.asarray(off, dtype=float)
            part["goal_pos"] = [float(c[0] + o[0]), float(c[1] + o[1]),
                                float((c[2] if c.size > 2 else 0.0) + (o[2] if o.size > 2 else 0.0))]


def _cand_xy(c: Dict) -> np.ndarray:
    if c.get("init_pos") is not None:
        return np.asarray(c["init_pos"], dtype=float)[:2]
    return np.asarray(c.get("xy", [0.0, 0.0]), dtype=float)[:2]


def _footprint_radius(part: Dict) -> float:
    fp = np.asarray(part.get("footprint", [0.05, 0.05]), dtype=float)[:2]
    return 0.5 * float(np.hypot(fp[0], fp[1]))


def _layout_has_overlap(layout: Dict[str, Dict], radii: Dict[str, float], margin: float) -> bool:
    pids = list(layout.keys())
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = _cand_xy(layout[pids[i]]), _cand_xy(layout[pids[j]])
            if float(np.linalg.norm(a - b)) < radii[pids[i]] + radii[pids[j]] + margin:
                return True
    return False


def _layout_key(layout: Dict[str, Dict]) -> tuple:
    return tuple(sorted(
        (pid, round(float(_cand_xy(c)[0]), 4), round(float(_cand_xy(c)[1]), 4),
         str(c.get("pose_tag", "")))
        for pid, c in layout.items()
    ))


def _slim_choice(c: Dict) -> Dict:
    return {
        "init_pos": c.get("init_pos"),
        "init_rotmat": c.get("init_rotmat"),
        "pose_tag": c.get("pose_tag"),
        "rot_name": c.get("rot_name"),
        "nn_score": float(c.get("nn_score", 0.0)),
        "rank": int(c.get("rank", 0)),
    }


def _finalize_individual(idx: int, layout: Dict[str, Dict], radii: Dict[str, float],
                         margin: float) -> Dict:
    return {
        "individual": int(idx),
        "has_overlap": _layout_has_overlap(layout, radii, margin),
        "sum_nn_score": float(sum(float(c.get("nn_score", 0.0)) for c in layout.values())),
        "layout": {pid: _slim_choice(c) for pid, c in layout.items()},
    }


def _assemble_population(per_part: Dict[str, List[Dict]],
                         parts_by_id: Dict[str, Dict],
                         *,
                         m: int,
                         top_k: int,
                         seed: int,
                         overlap_margin: float,
                         resample: int = 40) -> List[Dict]:
    """Combine each part's top-k init candidates into ``m`` full-layout individuals.

    Individual 0 is the greedy best (each part's rank-1). The rest sample each part
    from its top-k weighted by ``nn_score``, preferring collision-free combinations
    (a light staging-overlap check; GA can still repair any residual overlap)."""
    rng = np.random.default_rng(int(seed))
    pools: Dict[str, List[Dict]] = {}
    weights: Dict[str, np.ndarray] = {}
    radii: Dict[str, float] = {}
    for pid, ranked in per_part.items():
        pool = list(ranked[: max(int(top_k), 1)]) if ranked else []
        pools[pid] = pool
        radii[pid] = _footprint_radius(parts_by_id.get(pid, {}))
        s = np.array([float(c.get("nn_score", 0.0)) for c in pool], dtype=float)
        if s.size == 0 or not np.any(s > 0):
            weights[pid] = np.ones(max(len(pool), 1)) / max(len(pool), 1)
        else:
            weights[pid] = s / s.sum()

    active = {pid: pool for pid, pool in pools.items() if pool}
    if not active:
        return []

    def _sample_layout() -> Dict[str, Dict]:
        return {pid: dict(pool[int(rng.choice(len(pool), p=weights[pid]))])
                for pid, pool in active.items()}

    population: List[Dict] = []
    seen = set()
    greedy = {pid: dict(pool[0]) for pid, pool in active.items()}
    population.append(_finalize_individual(0, greedy, radii, overlap_margin))
    seen.add(_layout_key(greedy))

    guard = 0
    while len(population) < int(m) and guard < int(m) * 50 + 50:
        guard += 1
        layout = None
        for _ in range(int(resample)):
            cand = _sample_layout()
            if not _layout_has_overlap(cand, radii, overlap_margin):
                layout = cand
                break
            layout = cand  # keep last as fallback if all overlap
        if layout is None:
            break
        key = _layout_key(layout)
        if key in seen:
            continue
        seen.add(key)
        population.append(_finalize_individual(len(population), layout, radii, overlap_margin))
    return population


def main():
    args = _parse_args()
    with open(args.input_json, "r", encoding="utf-8") as stream:
        sample = json.load(stream)

    center = None
    if str(args.assembly_center).strip():
        center = [float(v) for v in str(args.assembly_center).split(",") if v.strip() != ""]
    _resolve_goals_from_center(sample, center)

    cands_map: Dict[str, List] = {}
    if args.candidates_json.strip():
        with open(args.candidates_json, "r", encoding="utf-8") as stream:
            cands_map = json.load(stream)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = PartPlacementRankerNet(
        hidden=int(ckpt.get("hidden", 128)),
        dropout=float(ckpt.get("dropout", 0.15)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    if "table_x_range" not in sample:
        sample.setdefault("table_x_range", [0.06, 0.47])
        sample.setdefault("table_y_range", [-0.24, 0.24])
        sample.setdefault("table_top_z", 0.0)
    sample.setdefault("num_parts", len(sample.get("parts", [])))
    sample.setdefault("assembly_station_pos", sample.get("parts", [{}])[0].get("goal_pos", [0, 0, 0]))

    per_part: Dict[str, List[Dict]] = {}
    for part in sample.get("parts", []):
        pid = str(part["part_id"])
        if pid in cands_map:
            cands = cands_map[pid]
        else:
            cands = _default_candidates(sample, part)
        if not cands:
            grid = _grid_candidates(sample)
            z = float(sample.get("table_top_z", 0.0))
            pose_tags = part.get("pose_candidates", []) or [{"pose_tag": "identity"}]
            identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            cands = []
            for xy in grid:
                for pc in pose_tags[:4]:
                    cands.append({
                        "xy": xy,
                        "init_pos": [float(xy[0]), float(xy[1]), z],
                        "init_rotmat": pc.get("rotmat", identity),
                        "pose_tag": pc.get("pose_tag", "identity"),
                        "rot_name": pc.get("rot_name", "identity"),
                        "score": 0.0,
                    })
        per_part[pid] = _rank_part(model, sample, part, cands, args.device, args.top_k)
        print(f"[rank] {pid}: top-{len(per_part[pid])} (goal fixed)")

    result = {
        "num_parts": len(sample.get("parts", [])),
        "top_k": int(args.top_k),
        "assembly_center": sample.get("assembly_station_pos"),
        "per_part_ranking": per_part,
    }

    if int(args.population_size) > 0:
        parts_by_id = {str(p["part_id"]): p for p in sample.get("parts", [])}
        population = _assemble_population(
            per_part, parts_by_id,
            m=int(args.population_size), top_k=int(args.top_k),
            seed=int(args.population_seed), overlap_margin=float(args.overlap_margin),
        )
        n_clean = sum(0 if ind["has_overlap"] else 1 for ind in population)
        result["population"] = population
        result["population_size"] = len(population)
        print(f"[ga-gen1] assembled {len(population)} individuals "
              f"({n_clean} overlap-free) for {result['num_parts']} parts")

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if int(args.population_size) <= 0:
        print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(text)
    if args.population_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.population_json)), exist_ok=True)
        with open(args.population_json, "w", encoding="utf-8") as stream:
            json.dump({
                "num_parts": result["num_parts"],
                "top_k": result["top_k"],
                "assembly_center": result.get("assembly_center"),
                "population_size": result.get("population_size", 0),
                "population": result.get("population", []),
            }, stream, ensure_ascii=False, indent=2)
        print(f"[ga-gen1] wrote GA first generation -> {args.population_json}")


if __name__ == "__main__":
    main()
