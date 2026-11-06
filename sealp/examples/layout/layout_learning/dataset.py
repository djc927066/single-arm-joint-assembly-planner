"""jsonl 数据集 + collate。

把 ``generate_layout_dataset.py`` 产出的 jsonl 转成统一 batch (供所有模型使用)。
所有特征视图 (flatten / set / graph / target) 一次性算好并放进同一个 batch dict,
以保证不同模型的公平对比 (共用同一套基础 feature)。
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from . import features as F
from .dynamic_relations import build_dynamic_graph
from .generator_dataset import generator_item_fields


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sample_to_item(sample: Dict, max_parts: int = F.MAX_PARTS_DEFAULT,
                   feature_version: str = F.DEFAULT_FEATURE_VERSION) -> Dict:
    """把一条 sample dict 转成 numpy item (未 padding, collate 负责 padding)。"""
    node, gfeat = F.build_set_feature(sample, feature_version)
    graph = F.build_graph_feature(sample, feature_version=feature_version)
    dynamic_graph = build_dynamic_graph(sample, k_spatial=2)
    tgt = F.build_proposal_target(sample)
    n = max(1, F.sample_num_parts(sample))
    bounds = F._table_bounds(sample)
    return {
        "n": n,
        "flat": F.build_flatten_feature(sample, max_parts, feature_version),
        "node": node.astype(np.float32),
        "global": gfeat.astype(np.float32),
        "edge_index": graph["edge_index"],
        "edge_feat": graph["edge_feat"].astype(np.float32),
        "dynamic_edge_index": dynamic_graph["dynamic_edge_index"],
        "dynamic_edge_feat": dynamic_graph["dynamic_edge_feat"].astype(np.float32),
        "active_layout_xy": dynamic_graph["active_layout_xy"].astype(np.float32),
        "geometry_valid": dynamic_graph["geometry_valid"].astype(np.float32),
        "feas": np.float32(1.0 if sample.get("l2_pass", False) else 0.0),
        "score": np.float32(sample.get("layout_score", 0.0) if sample.get("l2_pass", False) else 0.0),
        "xy_target": tgt["xy_target"].astype(np.float32),
        "xy_valid": tgt["xy_valid"].astype(np.float32),
        "station_target": tgt["station_target"].astype(np.float32),
        "station_valid": np.float32(tgt["station_valid"]),
        "region_target": np.int64(min(int(tgt["region_target"]), F.MAX_REGIONS - 1)),
        "region_center": F._region_center(sample).astype(np.float32),
        "table_bounds": np.asarray(bounds, dtype=np.float32),
        # ---- seqrel 辅助字段 (其它模型忽略) ----
        "fail_class": np.int64(F.fail_reason_class(sample)),
        "group_key": F.ranking_group_key(sample),
    }
    item.update(generator_item_fields(sample, feature_version))
    return item


class LayoutDataset(Dataset):
    def __init__(self, samples: List[Dict], max_parts: int = F.MAX_PARTS_DEFAULT,
                 feature_version: str = F.DEFAULT_FEATURE_VERSION):
        self.max_parts = max_parts
        self.feature_version = feature_version
        self.items = [sample_to_item(s, max_parts, feature_version) for s in samples]
        for index, item in enumerate(self.items):
            item["sample_index"] = np.int64(index)

    @classmethod
    def from_jsonl(cls, path: str, max_parts: int = F.MAX_PARTS_DEFAULT,
                   feature_version: str = F.DEFAULT_FEATURE_VERSION) -> "LayoutDataset":
        return cls(load_jsonl(path), max_parts, feature_version)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_items(items: List[Dict]) -> Dict[str, torch.Tensor]:
    """把若干 item padding 并组装成统一 batch dict。"""
    B = len(items)
    N = max(int(it["n"]) for it in items)
    N = max(N, 1)
    PD = F.PART_FEATURE_DIM
    ED = F.EDGE_FEATURE_DIM

    node = np.zeros((B, N, PD), dtype=np.float32)
    node_mask = np.zeros((B, N), dtype=np.float32)
    adj = np.zeros((B, N, N), dtype=np.float32)
    edge_attr = np.zeros((B, N, N, ED), dtype=np.float32)
    xy_target = np.zeros((B, N, 2), dtype=np.float32)
    xy_valid = np.zeros((B, N), dtype=np.float32)
    active_layout_xy = np.zeros((B, N, 2), dtype=np.float32)
    geometry_valid = np.zeros((B, N), dtype=np.float32)
    dynamic_edge_indices = []
    dynamic_edge_features = []
    geometry_feat = np.zeros((B, N, 20), dtype=np.float32)
    pose_candidate_feat = np.zeros((B, N, 8, 12), dtype=np.float32)
    pose_candidate_mask = np.zeros((B, N, 8), dtype=np.float32)
    target_pose_index = np.zeros((B, N), dtype=np.int64)
    target_rotation_index = np.zeros((B, N), dtype=np.int64)
    selected_footprint = np.zeros((B, N, 2), dtype=np.float32)
    is_first_mask = np.zeros((B, N), dtype=np.float32)
    order_index_arr = np.zeros((B, N), dtype=np.float32)
    static_adj = np.zeros((B, N, N), dtype=np.float32)
    static_edge_attr = np.zeros((B, N, N, 10), dtype=np.float32)

    for b, it in enumerate(items):
        n = int(it["n"])
        node[b, :n] = it["node"][:n]
        node_mask[b, :n] = 1.0
        xy_target[b, :n] = it["xy_target"][:n]
        xy_valid[b, :n] = it["xy_valid"][:n]
        active_layout_xy[b, :n] = it["active_layout_xy"][:n]
        geometry_valid[b, :n] = it["geometry_valid"][:n]
        ei = it["edge_index"]
        ef = it["edge_feat"]
        for e in range(ei.shape[1]):
            u, v = int(ei[0, e]), int(ei[1, e])
            if u < N and v < N:
                adj[b, u, v] = 1.0
                edge_attr[b, u, v] = ef[e]
        dynamic_edge_indices.append(
            it["dynamic_edge_index"].astype(np.int64) + b * N)
        dynamic_edge_features.append(
            it["dynamic_edge_feat"].astype(np.float32))
        if "geometry_feat" in it:
            n = int(it["n"])
            geometry_feat[b, :n] = it["geometry_feat"][:n]
            pose_candidate_feat[b, :n] = it["pose_candidate_feat"][:n]
            pose_candidate_mask[b, :n] = it["pose_candidate_mask"][:n]
            target_pose_index[b, :n] = it["target_pose_index"][:n]
            target_rotation_index[b, :n] = it["target_rotation_index"][:n]
            selected_footprint[b, :n] = it["selected_footprint"][:n]
            is_first_mask[b, :n] = it["is_first_mask"][:n]
            order_index_arr[b, :n] = it["order_index"][:n]
            static_adj[b, :n, :n] = it["static_adj"][:n, :n]
            static_edge_attr[b, :n, :n] = it["static_edge_attr"][:n, :n]

    # ---- seqrel 辅助: fail 类别 + pair-ranking 分组 id (可选字段) ----
    fail_class = np.stack([it.get("fail_class", np.int64(-1)) for it in items]).astype(np.int64)
    group_keys = [it.get("group_key", str(b)) for b, it in enumerate(items)]
    key_to_id: Dict[str, int] = {}
    group_id = np.zeros(B, dtype=np.int64)
    for b, key in enumerate(group_keys):
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        group_id[b] = key_to_id[key]

    batch = {
        "flat_feat": torch.from_numpy(np.stack([it["flat"] for it in items])),
        "node_feat": torch.from_numpy(node),
        "node_mask": torch.from_numpy(node_mask),
        "global_feat": torch.from_numpy(np.stack([it["global"] for it in items])),
        "adj": torch.from_numpy(adj),
        "edge_attr": torch.from_numpy(edge_attr),
        "dynamic_edge_index": torch.from_numpy(
            np.concatenate(dynamic_edge_indices, axis=1)),
        "dynamic_edge_feat": torch.from_numpy(
            np.concatenate(dynamic_edge_features, axis=0)),
        "batch_index": torch.arange(B, dtype=torch.long).repeat_interleave(N),
        "active_layout_xy": torch.from_numpy(active_layout_xy),
        "geometry_valid": torch.from_numpy(geometry_valid),
        "static_mask": torch.from_numpy(F.STATIC_MASK.copy()),
        "global_static_mask": torch.from_numpy(F.GLOBAL_STATIC_MASK.copy()),
        "feas": torch.from_numpy(np.stack([it["feas"] for it in items])),
        "score": torch.from_numpy(np.stack([it["score"] for it in items])),
        "xy_target": torch.from_numpy(xy_target),
        "xy_valid": torch.from_numpy(xy_valid),
        "station_target": torch.from_numpy(np.stack([it["station_target"] for it in items])),
        "station_valid": torch.from_numpy(np.stack([it["station_valid"] for it in items])),
        "region_target": torch.from_numpy(np.stack([it["region_target"] for it in items])),
        "region_center": torch.from_numpy(np.stack([it["region_center"] for it in items])),
        "table_bounds": torch.from_numpy(np.stack([it["table_bounds"] for it in items])),
        "fail_class": torch.from_numpy(fail_class),
        "group_id": torch.from_numpy(group_id),
        "sample_index": torch.from_numpy(np.stack([
            it.get("sample_index", np.int64(-1)) for it in items
        ]).astype(np.int64)),
        "geometry_feat": torch.from_numpy(geometry_feat),
        "pose_candidate_feat": torch.from_numpy(pose_candidate_feat),
        "pose_candidate_mask": torch.from_numpy(pose_candidate_mask),
        "target_pose_index": torch.from_numpy(target_pose_index),
        "target_rotation_index": torch.from_numpy(target_rotation_index),
        "selected_footprint": torch.from_numpy(selected_footprint),
        "is_first_mask": torch.from_numpy(is_first_mask),
        "order_index": torch.from_numpy(order_index_arr),
        "static_adj": torch.from_numpy(static_adj),
        "static_edge_attr": torch.from_numpy(static_edge_attr),
    }
    return batch


def move_batch(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
