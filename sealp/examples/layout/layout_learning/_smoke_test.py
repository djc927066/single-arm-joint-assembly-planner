"""layout_learning 自检脚本 (合成数据, 不依赖 wrs)。

用法:
    python -m sealp.examples.layout.layout_learning._smoke_test
验证: 特征提取 / 每个模型前向 / propose / 损失 / 一轮训练 / 推理。
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from . import features as F
from .dataset import LayoutDataset, collate_items
from .losses import LossWeights, compute_loss
from .models import build_model, MODEL_NAMES, is_generator
from .train import train_model
from .infer import LayoutModelRunner


def _rand_sample(rng: np.random.Generator, n_parts: int = 5, feasible: bool = True) -> dict:
    grid = 3
    r, c = int(rng.integers(0, grid)), int(rng.integers(0, grid))
    parts = []
    order = list(range(n_parts))
    for i in range(n_parts):
        parts.append({
            "part_id": f"part_{i}",
            "order_index": order[i],
            "is_first": i == 0,
            "extent": (rng.random(3) * 0.2).tolist(),
            "footprint": (rng.random(2) * 0.2).tolist(),
            "goal_pos": [0.3 + rng.random() * 0.2, -0.3 + rng.random() * 0.4, rng.random() * 0.2],
            "goal_rotmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "parent": f"part_{i-1}" if i > 0 else "fixture",
            "topdown_count": int(rng.integers(0, 30)),
            "grasp_total": int(rng.integers(10, 90)),
            "staging_xy": [rng.random() * 0.5, -0.3 + rng.random() * 0.6],
            "pose_tag": "identity",
            "rot_name": "identity",
            "grasp_count": int(rng.integers(0, 40)),
            "arm_choice": "rgt",
            "per_part_dist": float(rng.random()),
            "per_part_manip": float(rng.random()),
            "per_part_rot_angle": float(rng.random()),
        })
    return {
        "sample_id": int(rng.integers(0, 1 << 30)),
        "seed": 0,
        "assembly_region_id": f"r{r}_c{c}",
        "assembly_region_rc": [r, c],
        "assembly_grid": grid,
        "assembly_station_pos": [0.36, 0.0, 0.0],
        "table_x_range": [0.0, 0.6],
        "table_y_range": [-0.5, 0.5],
        "table_top_z": 0.0,
        "part_order": [p["part_id"] for p in parts],
        "parts": parts,
        "l2_pass": bool(feasible),
        "l3_pass": False,
        "layout_score": float(0.5 + rng.random() * 0.4) if feasible else 0.0,
        "grasp_score_norm": float(rng.random()),
        "manip_score_norm": float(rng.random()),
        "dist_score_norm": float(rng.random()),
        "rot_score_norm": float(rng.random()),
        "spatial_score_norm": float(rng.random()),
        "fail_reason": "" if feasible else "pair_collision",
        "fail_part": None if feasible else "part_2",
        "fail_detail": {},
        "eval_time": 0.01,
    }


def main():
    import torch
    rng = np.random.default_rng(0)
    samples = [_rand_sample(rng, feasible=(rng.random() > 0.4)) for _ in range(80)]

    # 1) 特征
    ds = LayoutDataset(samples)
    batch = collate_items([ds[i] for i in range(8)])
    print("[smoke] batch keys:", sorted(batch.keys()))
    print("[smoke] node_feat", tuple(batch["node_feat"].shape),
          "flat", tuple(batch["flat_feat"].shape),
          "adj", tuple(batch["adj"].shape))

    flat_dim = F.flatten_feature_dim()

    # 2) 每个模型前向 + 损失 + propose
    for name in MODEL_NAMES:
        model = build_model(name, flat_dim=flat_dim)
        out = model(batch)
        res = compute_loss(out, batch, LossWeights(), is_generator(name))
        assert torch.isfinite(res["loss"]), f"{name} loss not finite"
        extra = ""
        if is_generator(name):
            prop_batch = collate_items([ds[0]])
            xy = model.propose(prop_batch, 4)
            extra = f" propose={tuple(xy.shape)}"
        print(f"[smoke] {name:16s} feas={tuple(out['feas_logit'].shape)} "
              f"score={tuple(out['score_pred'].shape)} loss={float(res['loss']):.4f}{extra}")

    # 3) 端到端: 训练 mlp + sagpn 1 epoch, 再推理
    with tempfile.TemporaryDirectory() as td:
        ds_path = os.path.join(td, "ds.jsonl")
        with open(ds_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        for name in ("mlp", "sagpn"):
            summary = train_model(ds_path, name, save_dir=td, epochs=2,
                                  batch_size=16, device="cpu", verbose=False)
            runner = LayoutModelRunner(summary["best_path"], device="cpu")
            if runner.is_generator:
                layouts = runner.propose_layouts(samples[0], k=3)
                print(f"[smoke] {name} propose_layouts -> {len(layouts)} layouts, "
                      f"keys={list(layouts[0].keys())}")
            else:
                sc = runner.score_layouts(samples[:5])
                print(f"[smoke] {name} score_layouts -> feas_prob shape {sc['feas_prob'].shape}")
    print("[smoke] ALL PASS")


if __name__ == "__main__":
    main()
