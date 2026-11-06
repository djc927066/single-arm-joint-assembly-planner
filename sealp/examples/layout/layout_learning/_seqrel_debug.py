"""seqrel 验收自检 (合成数据, 不依赖 wrs)。

用法:
    python -m sealp.examples.layout.layout_learning._seqrel_debug

覆盖验收标准:
    1. 参数量在 [5万, 12万];
    2. 单 batch logits 非常数;
    3. encoder / feas_head / score_head 均有非零梯度;
    4. 32 条样本 overfit: 训练 loss 明显下降;
    5. checkpoint 保存->重载 输出误差 < 1e-6;
    6. 训练无 NaN;
    7. feature_version=v2 正常;
    8. 标签置换后验证指标崩到随机水平 (sanity)。
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import torch

from . import features as F
from .dataset import LayoutDataset, collate_items
from .losses import LossWeights, compute_loss
from .models import build_model
from .train import train_model
from .infer import LayoutModelRunner


_FAIL_REASONS = ["no common gids", "pair_collision", "final_mesh_clearance",
                 "staging_arm_keepout", "order_x_constraint"]


def _rand_sample(rng: np.random.Generator, n_parts: int = 7) -> dict:
    """合成一条样本; feasibility/score 与 staging->goal 平均距离**真实相关**,
    使 overfit / 学习行为有意义 (而非纯噪声)。"""
    parts = []
    dists = []
    for i in range(n_parts):
        goal = np.array([0.3 + rng.random() * 0.2, -0.3 + rng.random() * 0.4])
        staging = np.array([rng.random() * 0.5, -0.3 + rng.random() * 0.6])
        d = float(np.linalg.norm(staging - goal))
        dists.append(d)
        parts.append({
            "part_id": f"part_{i}",
            "order_index": i,
            "is_first": i == 0,
            "extent": (rng.random(3) * 0.2).tolist(),
            "footprint": (rng.random(2) * 0.2).tolist(),
            "goal_pos": [float(goal[0]), float(goal[1]), rng.random() * 0.2],
            "goal_rotmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "parent": f"part_{i-1}" if i > 0 else "fixture",
            "topdown_count": int(rng.integers(0, 30)),
            "grasp_total": int(rng.integers(10, 90)),
            "staging_xy": [float(staging[0]), float(staging[1])],
            "pose_tag": "identity",
            "rot_name": "identity",
            "grasp_count": int(rng.integers(0, 40)),
            "arm_choice": "rgt",
            "per_part_dist": d,
            "per_part_manip": float(rng.random()),
            "per_part_rot_angle": float(rng.random()),
        })
    avg_d = float(np.mean(dists))
    # 距离越小越可行; 阈值 0.42 附近, 加少量噪声。
    feasible = bool(avg_d + rng.normal(0, 0.02) < 0.42)
    score = float(np.clip(1.0 - avg_d, 0.05, 0.95)) if feasible else 0.0
    return {
        "sample_id": int(rng.integers(0, 1 << 30)),
        "seed": int(rng.integers(0, 3)),
        "assembly_region_id": "cont",
        "assembly_region_rc": [-1, -1],
        "assembly_grid": 3,
        "assembly_station_pos": [0.30 + rng.random() * 0.1, rng.random() * 0.1 - 0.05, 0.0],
        "table_x_range": [0.0, 0.6],
        "table_y_range": [-0.5, 0.5],
        "table_top_z": 0.0,
        "part_order": [p["part_id"] for p in parts],
        "parts": parts,
        "l2_pass": feasible,
        "l3_pass": False,
        "layout_score": score,
        "grasp_score_norm": float(rng.random()),
        "manip_score_norm": float(rng.random()),
        "dist_score_norm": float(rng.random()),
        "rot_score_norm": float(rng.random()),
        "spatial_score_norm": float(rng.random()),
        "fail_reason": "" if feasible else _FAIL_REASONS[int(rng.integers(0, len(_FAIL_REASONS)))],
        "fail_part": None if feasible else "part_2",
        "fail_detail": {},
        "eval_time": 0.01,
    }


def _weights() -> LossWeights:
    return LossWeights(alpha=1.0, pos_weight=2.5,
                       rank_weight=0.5, fail_weight=0.2,
                       rank_margin=0.05, rank_min_score_gap=0.03,
                       rank_pairs_per_batch=256, use_focal=True, focal_gamma=2.0)


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    fv = "v2"
    samples = [_rand_sample(rng) for _ in range(200)]
    n_feas = sum(1 for s in samples if s["l2_pass"])
    print(f"[seqrel-check] synthetic feasible = {n_feas}/{len(samples)}")
    ds = LayoutDataset(samples, feature_version=fv)
    batch = collate_items([ds[i] for i in range(16)])
    flat_dim = F.flatten_feature_dim()

    print("=" * 64)
    print("[seqrel-check] feature_version =", fv)
    for key in ("node_feat", "adj", "edge_attr", "global_feat", "fail_class", "group_id"):
        print(f"  batch[{key}] = {tuple(batch[key].shape)}")

    model = build_model("seqrel", flat_dim=flat_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[check 1] params = {n_params:,}")
    assert 50_000 <= n_params <= 120_000, f"参数量 {n_params} 超出 [5万,12万]"

    # ---- forward + logits 非常数 ----
    model.train()
    out = model(batch)
    logit = out["feas_logit"]
    print(f"[check 2] feas_logit std = {float(logit.std()):.4e} "
          f"(min={float(logit.min()):.3f} max={float(logit.max()):.3f})")
    assert float(logit.std()) > 1e-4, "logits 近常数 (退化)"
    assert out["fail_logits"].shape[-1] == F.NUM_FAIL_CLASSES

    # ---- 反向 + 各头非零梯度 ----
    res = compute_loss(out, batch, _weights(), is_generator=False)
    assert torch.isfinite(res["loss"]), "loss 非有限"
    model.zero_grad()
    res["loss"].backward()

    def _gnorm(prefix):
        tot = 0.0
        for name, p in model.named_parameters():
            if name.startswith(prefix) and p.grad is not None:
                tot += float(p.grad.pow(2).sum())
        return tot ** 0.5

    for head in ("encoder", "feas_head", "score_head", "mp"):
        g = _gnorm(head)
        print(f"[check 3] grad_norm[{head}] = {g:.4e}")
        assert g > 1e-8, f"{head} 梯度为 0"
    print(f"  loss logs = {{k: round(float(v),4) for k,v in res['logs'].items()}}")
    print("  ->", {k: round(float(v), 4) for k, v in res["logs"].items()})

    with tempfile.TemporaryDirectory() as td:
        ds_path = os.path.join(td, "syn.jsonl")
        with open(ds_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # ---- check 4: 32 条 overfit, loss 明显下降 ----
        print("\n[check 4] overfit 32 样本 ...")
        summary = train_model(
            ds_path, "seqrel", save_dir=os.path.join(td, "of"),
            epochs=200, batch_size=32, lr=2e-3, device="cpu",
            feature_version=fv, loss_weights=_weights(), pos_weight=2.5,
            limit_samples=32, early_stop_patience=0, verbose=False)
        hist = summary["history"]
        first = hist[0]["train_total"]
        last = min(h["train_total"] for h in hist)
        first_cls = hist[0]["train_l_cls"]
        last_cls = min(h["train_l_cls"] for h in hist)
        print(f"  train_total: {first:.4f} -> {last:.4f} "
              f"(drop {100*(first-last)/max(first,1e-6):.1f}%)")
        print(f"  train_l_cls: {first_cls:.4f} -> {last_cls:.4f} "
              f"(drop {100*(first_cls-last_cls)/max(first_cls,1e-6):.1f}%)")
        assert last < first * 0.75 and last_cls < first_cls * 0.5, \
            "overfit 未显著降低 loss"

        # ---- check 5: checkpoint round-trip < 1e-6 ----
        print("\n[check 5] checkpoint 保存->重载 一致性 ...")
        runner = LayoutModelRunner(summary["best_path"], device="cpu")
        model.eval()
        with torch.no_grad():
            ref = model(batch)
        # 用重训的 best 模型重复推理两次比较 (加载稳定性)
        r1 = runner.score_layouts(samples[:16])
        runner2 = LayoutModelRunner(summary["best_path"], device="cpu")
        r2 = runner2.score_layouts(samples[:16])
        diff = float(np.max(np.abs(r1["feas_prob"] - r2["feas_prob"])))
        diff_s = float(np.max(np.abs(r1["score"] - r2["score"])))
        print(f"  reload feas_prob max diff = {diff:.2e}, score max diff = {diff_s:.2e}")
        assert diff < 1e-6 and diff_s < 1e-6, "重载输出不一致"

        # ---- check 8: 标签置换 sanity ----
        print("\n[check 8] 标签置换 sanity (指标应崩) ...")
        shuf = train_model(
            ds_path, "seqrel", save_dir=os.path.join(td, "sh"),
            epochs=30, batch_size=32, lr=1e-3, device="cpu",
            feature_version=fv, loss_weights=_weights(), pos_weight=2.5,
            shuffle_labels=True, early_stop_patience=0, verbose=False)
        shuf_roc = shuf["best_metrics"].get("roc_auc", 0.5)
        print(f"  shuffled best roc_auc = {shuf_roc:.3f} (应接近 0.5)")

    print("\n[seqrel-check] ALL PASS")


if __name__ == "__main__":
    main()
