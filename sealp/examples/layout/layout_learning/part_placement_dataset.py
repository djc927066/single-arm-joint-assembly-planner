"""Dataset loader for synthetic bbox per-part placement ranking."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import features as F

MAX_CANDIDATES = 64
POSE_FEAT_DIM = 8


def _table_bounds(sample: Dict) -> Tuple[float, float, float, float]:
    xr = sample.get("table_x_range", [0.0, 1.0])
    yr = sample.get("table_y_range", [0.0, 1.0])
    return float(xr[0]), float(xr[1]), float(yr[0]), float(yr[1])


def _part_static_vector(part: Dict, sample: Dict, num_parts: int) -> np.ndarray:
    static = F.build_part_feature(part, sample, num_parts, feature_version="v2")
    return static[:18]


def _global_vector(sample: Dict) -> np.ndarray:
    return F.build_global_feature(sample, feature_version="v2")


def _candidate_feature(xy: List[float],
                       init_hint: Optional[List[float]],
                       pose_idx: int,
                       score: float,
                       bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xy_arr = np.asarray(xy, dtype=np.float32)[:2]
    norm_xy = F.normalize_xy(xy_arr, bounds)
    if init_hint is not None:
        hint = np.asarray(init_hint, dtype=np.float32)[:2]
        offset = F.normalize_offset(xy_arr - hint, bounds)
    else:
        offset = np.zeros(2, dtype=np.float32)
    feat = np.zeros(POSE_FEAT_DIM, dtype=np.float32)
    feat[0:2] = norm_xy
    feat[2:4] = offset
    feat[4] = float(pose_idx) / max(MAX_CANDIDATES - 1, 1)
    # feat[5] intentionally left 0: the oracle ``score`` is only available while
    # collecting (WRS eval). At inference candidates come from a raw grid with no
    # oracle, so feeding score here would leak the label at train time and be a
    # constant 0 at inference. The ranker must learn ordering from geometry
    # (xy / offset / pose / bbox / goal) instead. ``score`` kept in the signature
    # for backward compatibility but is not used as a feature.
    _ = score
    feat[5] = 0.0
    feat[6] = float(np.linalg.norm(norm_xy))
    feat[7] = 1.0
    return feat


class PartPlacementRankingDataset(Dataset):
    """One item = one (sample, part) ranking task."""

    def __init__(self, jsonl_path: str, *, max_candidates: int = MAX_CANDIDATES):
        self.max_candidates = int(max_candidates)
        self.items: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8-sig") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                schema = str(sample.get("schema_version", ""))
                if schema not in ("synthetic_bbox_v1", "synthetic_bbox_single_v1"):
                    continue

                if schema == "synthetic_bbox_single_v1":
                    parts = list(sample.get("parts", []))
                    if not parts:
                        continue
                    part = parts[0]
                    pid = str(part.get("part_id", "box_0"))
                    cands = list(sample.get("init_candidates_ranked", []) or [])
                    if not cands:
                        blocks = sample.get("ranking_targets", []) or []
                        if blocks:
                            cands = list(blocks[0].get("candidates", []) or [])
                    if len(cands) < 2:
                        continue
                    self.items.append({
                        "sample": sample,
                        "part": part,
                        "part_id": pid,
                        "candidates": cands[: self.max_candidates],
                        "num_parts": 1,
                    })
                    continue

                num_parts = int(sample.get("num_parts", len(sample.get("parts", []))))
                part_map = {p["part_id"]: p for p in sample.get("parts", [])}
                for block in sample.get("ranking_targets", []):
                    pid = str(block.get("part_id", ""))
                    cands = list(block.get("candidates", []) or [])
                    if pid not in part_map or len(cands) < 2:
                        continue
                    self.items.append({
                        "sample": sample,
                        "part": part_map[pid],
                        "part_id": pid,
                        "candidates": cands[: self.max_candidates],
                        "num_parts": num_parts,
                    })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        sample = item["sample"]
        part = item["part"]
        cands = item["candidates"]
        bounds = _table_bounds(sample)
        n_c = min(len(cands), self.max_candidates)

        part_static = _part_static_vector(part, sample, item["num_parts"])
        global_feat = _global_vector(sample)
        init_hint = part.get("init_pos_balanced") or sample.get("init_pos_balanced", {}).get(item["part_id"])

        cand_feat = np.zeros((self.max_candidates, POSE_FEAT_DIM), dtype=np.float32)
        cand_mask = np.zeros(self.max_candidates, dtype=np.float32)
        target = np.zeros(self.max_candidates, dtype=np.float32)

        for i, c in enumerate(cands[: self.max_candidates]):
            cand_feat[i] = _candidate_feature(
                c.get("xy", [0, 0]),
                init_hint,
                i,
                float(c.get("score", 0.0)),
                bounds,
            )
            cand_mask[i] = 1.0
            if int(c.get("rank", i + 1)) == 1:
                target[i] = 1.0

        if target.sum() <= 0 and n_c > 0:
            target[0] = 1.0

        return {
            "part_static": torch.from_numpy(part_static),
            "global_feat": torch.from_numpy(global_feat),
            "cand_feat": torch.from_numpy(cand_feat),
            "cand_mask": torch.from_numpy(cand_mask),
            "target": torch.from_numpy(target),
            "bounds": torch.tensor(bounds, dtype=torch.float32),
        }


def collate_ranking_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "part_static": torch.stack([b["part_static"] for b in batch]),
        "global_feat": torch.stack([b["global_feat"] for b in batch]),
        "cand_feat": torch.stack([b["cand_feat"] for b in batch]),
        "cand_mask": torch.stack([b["cand_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "bounds": torch.stack([b["bounds"] for b in batch]),
    }


def train_val_split(dataset: PartPlacementRankingDataset,
                    val_frac: float = 0.15,
                    seed: int = 0) -> Tuple[List[int], List[int]]:
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx
