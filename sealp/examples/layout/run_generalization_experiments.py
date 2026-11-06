"""run_generalization_experiments.py —— 泛化能力离线对比 (论文用)。

动机
====
in-task random split 只能证明"插值能力"。要证明模型学到**可迁移的 layout prior**,
需要在**分布外**的 split 上评估, 并看 transfer_gap:

    transfer_gap(metric) = in_task_metric(random) - holdout_metric(seed/region)

gap 越小 -> 泛化越好。本脚本对多个模型 × 多种 split 训练并汇总离线指标,
输出 ``generalization_summary.csv``。

注意: 这是**离线**泛化评估 (基于 val 集指标)。"真实搜索是否更快找到可行布局"由
另一个脚本 ``run_neural_search_eval.py`` (阶段 2) 负责, 因为离线 AUC 高 != 搜索有效。

模型定位:
    - mlp                 : task-specific scorer baseline (固定 flat feature)。
    - deepsets/gcn/gat    : set/graph transferable scorer (变长/置换不变/带关系)。
    - sagpn (small)       : sequence-aware graph proposal network (Ours)。

用法
====
    python -m sealp.examples.layout.run_generalization_experiments \
        --dataset sealp/examples/layout/_output/layout_dataset_v2.jsonl \
        --models mlp,deepsets,gcn,gat \
        --splits random,stratified,seed_holdout,region_holdout \
        --feature-version v2 \
        --epochs 200 --early-stop-patience 30 \
        --save-dir checkpoints/layout_models_transfer \
        --results-dir sealp/examples/layout/_output/generalization
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning.losses import LossWeights
from layout_learning.models import size_kwargs
from layout_learning.train import train_model


# 汇总里关注的指标 (transfer_gap 也基于这些计算)
_METRICS = ["roc_auc", "pr_auc", "recall_at_k", "precision_at_k",
            "topk_avg_true_score", "topk_avg_score_norm", "enrichment",
            "score_spearman", "composite"]


def _parse_pos_weight(spec: str):
    """解析 --pos-weight: 'auto' 或正浮点数。"""
    text = str(spec).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--pos-weight 必须是 'auto' 或正浮点数，例如 2.5"
        ) from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("--pos-weight 必须大于 0")
    return value


def _parse_args():
    p = argparse.ArgumentParser(description="Offline generalization experiments")
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", default="mlp,deepsets,gcn,gat")
    p.add_argument("--splits", default="random,stratified,seed_holdout,region_holdout")
    p.add_argument("--feature-version", default="v2", choices=["v1", "v2"])
    p.add_argument("--region-holdout-mode", default="auto",
                   choices=["auto", "rc", "xy_grid"])
    p.add_argument("--region-xy-grid", type=int, default=3)
    p.add_argument("--model-size", default="small", choices=["small", "base"],
                   help="生成式模型 (sagpn) 尺寸; scorer 忽略")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--pos-weight", type=_parse_pos_weight, default="auto",
        help="'auto'=按每个训练 split 的 neg/pos 自动计算，或指定正浮点数")
    p.add_argument("--score-threshold", default="quantile:0.70")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--save-dir", default=os.path.join("checkpoints", "layout_models_transfer"))
    p.add_argument("--results-dir",
                   default=os.path.join("sealp", "examples", "layout", "_output", "generalization"))
    return p.parse_args()


def _elite_quantile(spec: str):
    spec = str(spec).strip()
    if spec.lower().startswith("quantile:"):
        return float(spec.split(":", 1)[1]), 0.0
    return None, float(spec)


def main():
    args = _parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    eq, abs_thr = _elite_quantile(args.score_threshold)

    rows = []
    for model in models:
        mkwargs = size_kwargs(model, args.model_size)
        model_rows = {}
        for split in splits:
            save_dir = os.path.join(args.save_dir, f"{model}_{split}")
            print("=" * 72)
            print(f"[gen-exp] model={model} split={split} feature={args.feature_version}")
            print("=" * 72)
            initial_pos_weight = 1.0 if args.pos_weight == "auto" else float(args.pos_weight)
            weights = LossWeights(pos_weight=initial_pos_weight, score_threshold=abs_thr)
            summary = train_model(
                dataset_path=args.dataset,
                model_name=model,
                save_dir=save_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                loss_weights=weights,
                pos_weight=args.pos_weight,
                model_kwargs=mkwargs,
                topk=args.topk,
                split_mode=split,
                region_holdout_mode=args.region_holdout_mode,
                region_xy_grid=args.region_xy_grid,
                feature_version=args.feature_version,
                elite_quantile=eq,
                early_stop_patience=args.early_stop_patience,
                verbose=True,
            )
            bm = summary.get("best_metrics", {}) or {}
            row = {
                "model": model,
                "split_mode": split,
                "feature_version": args.feature_version,
                "n_train": summary.get("n_train"),
                "n_val": summary.get("n_val"),
                "train_feasible_rate": round(summary.get("train_feasible_rate", 0.0), 4),
                "train_positive_count": summary.get("train_positive_count"),
                "train_negative_count": summary.get("train_negative_count"),
                "effective_pos_weight": round(float(summary.get("effective_pos_weight", 0.0)), 4),
                "best_epoch": summary.get("best_epoch"),
                "checkpoint_path": summary.get("best_path"),
            }
            for m in _METRICS:
                row[m] = round(float(bm.get(m, float("nan"))), 4) if bm.get(m) is not None else ""
            rows.append(row)
            model_rows[split] = row

        # transfer_gap: 以 random split 为 in-task 基准
        base = model_rows.get("random") or model_rows.get("stratified")
        if base is not None:
            for split, row in model_rows.items():
                for m in ("roc_auc", "pr_auc", "recall_at_k", "composite"):
                    try:
                        row[f"gap_{m}"] = round(float(base[m]) - float(row[m]), 4)
                    except (ValueError, TypeError):
                        row[f"gap_{m}"] = ""

    # 写 CSV
    out_csv = os.path.join(args.results_dir, "generalization_summary.csv")
    fieldnames = (["model", "split_mode", "feature_version", "n_train", "n_val",
                   "train_feasible_rate", "train_positive_count", "train_negative_count",
                   "effective_pos_weight", "best_epoch"] + _METRICS
                  + ["gap_roc_auc", "gap_pr_auc", "gap_recall_at_k", "gap_composite",
                     "checkpoint_path"])
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[gen-exp] done. summary -> {out_csv}")
    print("[gen-exp] transfer_gap 越小 = 泛化越好 (in-task random 与 holdout 差距)。")


if __name__ == "__main__":
    main()
