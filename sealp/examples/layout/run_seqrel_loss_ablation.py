"""SeqRel loss-only ablation runner and result aggregator.

This script never changes model structure or features.  It can:
1. dry-run and validate the five isolated run directories;
2. train only missing/invalid variants with the shared split;
3. aggregate existing checkpoints into CSV/JSON, including corrected
   inference-only relation masking diagnostics for SeqRel-full.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from layout_learning import features as F
from layout_learning._seqrel_failure_diagnosis import (
    _infer_all,
    _kendall_tau,
    _load_model,
    _pearson,
    _score_diag,
)
from layout_learning.dataset import collate_items, load_jsonl, move_batch, sample_to_item


VARIANTS = {
    "no_focal": dict(score_weight=1.0, rank_weight=0.5, fail_weight=0.2, use_focal=False),
    "score_x5": dict(score_weight=5.0, rank_weight=0.5, fail_weight=0.2, use_focal=True),
    "no_aux": dict(score_weight=1.0, rank_weight=0.0, fail_weight=0.0, use_focal=True),
    "bce_score_x5": dict(score_weight=5.0, rank_weight=0.5, fail_weight=0.2, use_focal=False),
    "bce_score_x5_no_aux": dict(
        score_weight=5.0, rank_weight=0.0, fail_weight=0.0, use_focal=False
    ),
}

REQUIRED_FILES = (
    "seqrel_best.pt",
    "metrics.json",
    "training_history.csv",
    "config.json",
    "split_indices.json",
    "train.log",
)


def _json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _same_split(a: Path, b: Path) -> bool:
    ja, jb = _json(a), _json(b)
    return (
        ja.get("train_indices") == jb.get("train_indices")
        and ja.get("val_indices") == jb.get("val_indices")
    )


def validate_run(run_dir: Path, expected: Dict, shared_split: Path) -> List[str]:
    errors: List[str] = []
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        return errors

    config = _json(run_dir / "config.json")
    checks = {
        "model_name": "seqrel",
        "feature_version": "v2",
        "split_mode": "stratified",
        "training_seed": 0,
        "hidden_dim": 64,
        "dropout": 0.2,
        **expected,
    }
    for key, value in checks.items():
        actual = config.get(key)
        if isinstance(value, float):
            ok = actual is not None and abs(float(actual) - value) < 1e-9
        else:
            ok = actual == value
        if not ok:
            errors.append(f"config.{key}={actual!r}, expected {value!r}")
    if not _same_split(run_dir / "split_indices.json", shared_split):
        errors.append("split indices differ from shared split")
    if (run_dir / "training_history.csv").stat().st_size == 0:
        errors.append("training_history.csv is empty")
    if (run_dir / "train.log").stat().st_size == 0:
        errors.append("train.log is empty")
    return errors


def _training_command(
    python: str,
    dataset: Path,
    shared_split: Path,
    run_root: Path,
    run_dir: Path,
    cfg: Dict,
) -> List[str]:
    cmd = [
        python,
        str(_THIS_DIR / "train_layout_network.py"),
        "--dataset", str(dataset),
        "--model", "seqrel",
        "--hidden-dim", "64",
        "--dropout", "0.2",
        "--feature-version", "v2",
        "--split-mode", "stratified",
        "--seed", "0",
        "--epochs", "200",
        "--batch-size", "32",
        "--lr", "5e-4",
        "--weight-decay", "1e-4",
        "--topk", "10",
        "--early-stop-patience", "30",
        "--early-stop-metric", "composite",
        "--split-indices", str(shared_split),
        "--run-root", str(run_root),
        "--save-dir", str(run_dir),
        "--score-weight", str(cfg["score_weight"]),
        "--rank-weight", str(cfg["rank_weight"]),
        "--fail-weight", str(cfg["fail_weight"]),
    ]
    cmd.append("--use-focal" if cfg["use_focal"] else "--no-focal")
    return cmd


def dry_run(args) -> bool:
    print("[dry-run] shared split:", args.shared_split)
    split = _json(args.shared_split)
    print(
        f"[dry-run] split train={len(split['train_indices'])} "
        f"val={len(split['val_indices'])}"
    )
    all_valid = True
    for name, cfg in VARIANTS.items():
        run_dir = args.run_root / name
        errors = validate_run(run_dir, cfg, args.shared_split)
        state = "VALID existing run" if not errors else "NEEDS TRAINING"
        print(f"[dry-run] {name}: {state} -> {run_dir}")
        for error in errors:
            print(f"  - {error}")
        print("  command:", subprocess.list2cmdline(_training_command(
            args.python, args.dataset, args.shared_split, args.run_root, run_dir, cfg
        )))
        all_valid &= not errors
    print("[dry-run] no training executed")
    return all_valid


def run_missing(args) -> None:
    for name, cfg in VARIANTS.items():
        run_dir = args.run_root / name
        errors = validate_run(run_dir, cfg, args.shared_split)
        if not errors:
            print(f"[run] skip valid existing variant: {name}")
            continue
        print(f"[run] train missing/invalid variant: {name}; reasons={errors}")
        subprocess.run(
            _training_command(
                args.python, args.dataset, args.shared_split, args.run_root, run_dir, cfg
            ),
            cwd=str(_THIS_DIR),
            check=True,
        )
        remaining = validate_run(run_dir, cfg, args.shared_split)
        if remaining:
            raise RuntimeError(f"{name} output validation failed: {remaining}")


def _predict(
    checkpoint: Path,
    samples: List[Dict],
    val_idx: List[int],
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model, ckpt = _load_model(str(checkpoint), device)
    max_parts = int(ckpt.get("max_parts", F.MAX_PARTS_DEFAULT))
    logits, scores = _infer_all(
        model, samples, val_idx, max_parts, "v2", device, batch_size=64
    )
    return logits, scores, ckpt


def _full_metrics(
    checkpoint: Path,
    metrics_path: Path,
    samples: List[Dict],
    val_idx: List[int],
    device: str,
) -> Dict:
    logits, pred, ckpt = _predict(checkpoint, samples, val_idx, device)
    labels = np.asarray(
        [float(bool(samples[i].get("l2_pass", False))) for i in val_idx], dtype=float
    )
    truth = np.asarray(
        [
            float(samples[i].get("layout_score", 0.0))
            if samples[i].get("l2_pass", False)
            else 0.0
            for i in val_idx
        ],
        dtype=float,
    )
    score = _score_diag(pred, labels, truth)
    stored = _json(metrics_path)
    return {
        "roc_auc": stored["roc_auc"],
        "pr_auc": stored["pr_auc"],
        "precision_at_k": stored["precision_at_k"],
        "recall_at_k": stored["recall_at_k"],
        "enrichment": stored["enrichment"],
        "score_spearman": score["spearman"],
        "score_pearson": score["pearson"],
        "score_kendall_tau": score["kendall_tau"],
        "score_mae": score["mae"],
        "score_rmse": score["rmse"],
        "true_score_std": score["true_std"],
        "predicted_score_std": score["pred_std"],
        "predicted_true_std_ratio": score["pred_over_true_std_ratio"],
        "regression_slope": score["regression_slope"],
        "composite": stored["composite"],
        "best_epoch": int(stored["best_epoch"]),
        "parameter_count": int(stored["parameter_count"]),
        "training_time_seconds": float(stored["training_time_seconds"]),
        "checkpoint": str(checkpoint.resolve()),
        "feature_version": ckpt.get("feature_version"),
    }


def _loss_near_best(run_dir: Path, best_epoch: int, cfg: Dict) -> Dict:
    with (run_dir / "training_history.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    nearby = [
        row for row in rows
        if abs(int(float(row["epoch"])) - best_epoch) <= 2
    ]
    if not nearby:
        raise RuntimeError(f"no history near best epoch {best_epoch}: {run_dir}")

    def mean_col(name: str) -> float:
        return float(np.mean([float(row.get(name, 0.0) or 0.0) for row in nearby]))

    raw = {
        "cls": mean_col("train_l_cls"),
        "score": mean_col("train_l_score"),
        "rank": mean_col("train_l_rank"),
        "fail": mean_col("train_l_fail"),
    }
    weighted = {
        "cls": raw["cls"],
        "score": raw["score"] * float(cfg["score_weight"]),
        "rank": raw["rank"] * float(cfg["rank_weight"]),
        "fail": raw["fail"] * float(cfg["fail_weight"]),
    }
    total = max(sum(weighted.values()), 1e-12)
    return {
        "loss_window": f"epoch {max(1, best_epoch - 2)}..{best_epoch + 2}",
        **{f"raw_{k}_loss": v for k, v in raw.items()},
        **{f"weighted_{k}_contribution": v for k, v in weighted.items()},
        **{f"{k}_loss_fraction": v / total for k, v in weighted.items()},
    }


@contextlib.contextmanager
def _zero_message_updates(model):
    hooks = [
        layer.upd.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        )
        for layer in model.mp
    ]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def _relation_masking(
    checkpoint: Path,
    samples: List[Dict],
    val_idx: List[int],
    device: str,
) -> Dict:
    model, ckpt = _load_model(str(checkpoint), device)
    max_parts = int(ckpt.get("max_parts", F.MAX_PARTS_DEFAULT))
    items = [sample_to_item(samples[i], max_parts, "v2") for i in val_idx]
    batch = move_batch(collate_items(items), device)

    def infer(b):
        with torch.no_grad():
            out = model(b)
        return out["feas_logit"].cpu().numpy(), out["score_pred"].cpu().numpy()

    full_l, full_s = infer(batch)

    edge_zero = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    edge_zero["edge_attr"].zero_()
    zero_l, zero_s = infer(edge_zero)

    with _zero_message_updates(model):
        message_l, message_s = infer(batch)

    shuffled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    rng = torch.Generator(device=shuffled["edge_attr"].device)
    rng.manual_seed(0)
    perm = torch.randperm(shuffled["edge_attr"].shape[0], generator=rng,
                          device=shuffled["edge_attr"].device)
    shuffled["edge_attr"] = shuffled["edge_attr"][perm]
    shuffle_l, shuffle_s = infer(shuffled)

    def delta(logit, score):
        return {
            "mean_abs_feasibility_logit_change": float(np.abs(logit - full_l).mean()),
            "mean_abs_predicted_score_change": float(np.abs(score - full_s).mean()),
        }

    return {
        "checkpoint": str(checkpoint.resolve()),
        "edge_attr_zero": delta(zero_l, zero_s),
        "message_zero": delta(message_l, message_s),
        "edge_attr_shuffle": delta(shuffle_l, shuffle_s),
        "definitions": {
            "edge_attr_zero": "edge_attr only is zero; message passing still runs",
            "message_zero": "per-layer update output is zero; residual + LayerNorm remain",
            "edge_attr_shuffle": "edge_attr shuffled across batch; node features unchanged",
        },
    }


def aggregate(args) -> None:
    split = _json(args.shared_split)
    val_idx = [int(i) for i in split["val_indices"]]
    samples = load_jsonl(str(args.dataset))
    rows: List[Dict] = []

    references = [
        (
            "DeepSets seed0",
            args.repro_root / "deepsets/stratified/seed0/deepsets_best.pt",
            args.repro_root / "deepsets/stratified/seed0/metrics.json",
            dict(score_weight=1.0, rank_weight=0.0, fail_weight=0.0, use_focal=False),
        ),
        (
            "SeqRel-full seed0",
            args.repro_root / "seqrel/stratified/seed0/seqrel_best.pt",
            args.repro_root / "seqrel/stratified/seed0/metrics.json",
            dict(score_weight=1.0, rank_weight=0.5, fail_weight=0.2, use_focal=True),
        ),
    ]
    for name, checkpoint, metrics, cfg in references:
        row = {"variant": name, "reference_only": True, **cfg}
        row.update(_full_metrics(checkpoint, metrics, samples, val_idx, args.device))
        if name.startswith("SeqRel"):
            row.update(_loss_near_best(checkpoint.parent, row["best_epoch"], cfg))
        rows.append(row)

    for name, cfg in VARIANTS.items():
        run_dir = args.run_root / name
        errors = validate_run(run_dir, cfg, args.shared_split)
        if errors:
            raise RuntimeError(f"cannot aggregate invalid run {name}: {errors}")
        row = {"variant": name, "reference_only": False, **cfg}
        row.update(_full_metrics(
            run_dir / "seqrel_best.pt", run_dir / "metrics.json",
            samples, val_idx, args.device
        ))
        row.update(_loss_near_best(run_dir, row["best_epoch"], cfg))
        rows.append(row)

    relation = _relation_masking(
        args.repro_root / "seqrel/stratified/seed0/seqrel_best.pt",
        samples, val_idx, args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "seqrel_loss_ablation.csv"
    json_path = args.output_dir / "seqrel_loss_ablation.json"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(args.dataset.resolve()),
                "shared_split": str(args.shared_split.resolve()),
                "runs": rows,
                "relation_masking_full_checkpoint_only": relation,
            },
            f, ensure_ascii=False, indent=2,
        )
    print("[aggregate] wrote", csv_path)
    print("[aggregate] wrote", json_path)


def parse_args():
    root = _THIS_DIR.parents[2]
    repro = root / "checkpoints/layout_models_repro"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "sealp/examples/layout/_output/layout_dataset_v2.jsonl",
    )
    parser.add_argument("--repro-root", type=Path, default=repro)
    parser.add_argument(
        "--shared-split",
        type=Path,
        default=repro / "_splits/stratified/seed0/split_indices.json",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=repro / "seqrel_loss_ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "sealp/examples/layout/_output/seqrel_experiments",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.dry_run or args.run_missing or args.aggregate):
        raise SystemExit("choose at least one: --dry-run, --run-missing, --aggregate")
    if args.dry_run:
        dry_run(args)
    if args.run_missing:
        run_missing(args)
    if args.aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
