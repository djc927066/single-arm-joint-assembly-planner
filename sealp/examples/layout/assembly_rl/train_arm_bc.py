#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the first masked arm-conditioned behavior-cloning policy."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .arm_bc_model import ArmBCConfig, ArmBCPolicy


FLOAT_KEYS = (
    "occupancy",
    "part_features",
    "pose_features",
    "grasp_hint",
    "ik_hint_by_arm",
    "part_mask",
    "decision_mask",
    "current_part_mask",
    "parent_adjacency",
    "dependency_adjacency",
    "symmetry_adjacency",
    "pose_mask",
)


class ArmDemoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        episode_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.samples: List[Dict[str, Any]] = []

        for episode_row in episode_rows:
            episode_path = self.dataset_dir / episode_row["file"]
            with np.load(episode_path, allow_pickle=False) as npz:
                action_n = int(np.asarray(npz["action_mask_n"]).item())
                packed = np.asarray(
                    npz["action_mask_packed"],
                    dtype=np.uint8,
                )
                masks = np.unpackbits(
                    packed,
                    axis=1,
                    count=action_n,
                    bitorder="little",
                ).astype(np.bool_)

                step_count = int(np.asarray(npz["joint_action"]).shape[0])
                for step in range(step_count):
                    sample: Dict[str, Any] = {
                        key: np.asarray(npz[key][step]).copy()
                        for key in FLOAT_KEYS
                    }
                    sample["action_mask"] = masks[step].copy()
                    sample["joint_action"] = int(npz["joint_action"][step])
                    sample["arm_id"] = int(npz["arm_id"][step])
                    sample["pose_id"] = int(npz["pose_id"][step])
                    sample["row"] = int(npz["row"][step])
                    sample["col"] = int(npz["col"][step])
                    sample["episode_id"] = int(
                        episode_row["episode_id"]
                    )
                    sample["step"] = int(step)

                    if not sample["action_mask"][sample["joint_action"]]:
                        raise ValueError(
                            f"expert action is masked: "
                            f"{episode_path}, step={step}"
                        )
                    self.samples.append(sample)

        if not self.samples:
            raise ValueError("dataset split contains no transitions")

        self.action_n = int(self.samples[0]["action_mask"].size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        result: Dict[str, torch.Tensor] = {}

        for key in FLOAT_KEYS:
            result[key] = torch.from_numpy(
                np.asarray(sample[key], dtype=np.float32)
            )
        result["action_mask"] = torch.from_numpy(
            np.asarray(sample["action_mask"], dtype=np.bool_)
        )
        for key in (
            "joint_action",
            "arm_id",
            "pose_id",
            "row",
            "col",
            "episode_id",
            "step",
        ):
            result[key] = torch.tensor(
                sample[key],
                dtype=torch.long,
            )
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


@torch.no_grad()
def compute_metrics(
    model: ArmBCPolicy,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    exact = 0
    top5 = 0
    arm_correct = 0
    pose_correct = 0
    row_abs = 0.0
    col_abs = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model.masked_flat_logits(batch)
        target = batch["joint_action"]
        loss_sum += float(criterion(logits, target).item())

        prediction = logits.argmax(dim=1)
        top_k = min(5, logits.shape[1])
        top_indices = logits.topk(top_k, dim=1).indices
        exact += int((prediction == target).sum().item())
        top5 += int(
            (top_indices == target[:, None]).any(dim=1).sum().item()
        )

        pred_arm, pred_pose, pred_row, pred_col = model.decode_action(
            prediction
        )
        true_arm, true_pose, true_row, true_col = model.decode_action(
            target
        )
        arm_correct += int((pred_arm == true_arm).sum().item())
        pose_correct += int((pred_pose == true_pose).sum().item())
        row_abs += float(
            (pred_row - true_row).abs().sum().item()
        )
        col_abs += float(
            (pred_col - true_col).abs().sum().item()
        )
        total += int(target.numel())

    denominator = max(1, total)
    return {
        "count": float(total),
        "loss": loss_sum / denominator,
        "exact_top1": exact / denominator,
        "exact_top5": top5 / denominator,
        "arm_accuracy": arm_correct / denominator,
        "pose_accuracy": pose_correct / denominator,
        "row_mae_cells": row_abs / denominator,
        "col_mae_cells": col_abs / denominator,
    }


def format_metrics(prefix: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{prefix} "
        f"loss={metrics['loss']:.5f} "
        f"top1={metrics['exact_top1']:.3f} "
        f"top5={metrics['exact_top5']:.3f} "
        f"arm={metrics['arm_accuracy']:.3f} "
        f"pose={metrics['pose_accuracy']:.3f} "
        f"row_mae={metrics['row_mae_cells']:.2f} "
        f"col_mae={metrics['col_mae_cells']:.2f}"
    )


def split_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    validation_episodes: int,
    overfit_all: bool,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    rows = list(episodes)
    if overfit_all:
        return rows, rows
    if validation_episodes <= 0:
        return rows, rows
    if validation_episodes >= len(rows):
        raise ValueError(
            "--validation-episodes must be smaller than episode count"
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    validation = rows[:validation_episodes]
    training = rows[validation_episodes:]
    return training, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-episodes", type=int, default=1)
    parser.add_argument(
        "--overfit-all",
        action="store_true",
        help="Use all five episodes for both train and diagnostic evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=50)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(name)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = dataset_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    episodes = list(index.get("episodes", []))
    if len(episodes) < 2:
        raise ValueError("at least two successful episodes are required")

    train_rows, validation_rows = split_episodes(
        episodes,
        seed=args.seed,
        validation_episodes=args.validation_episodes,
        overfit_all=args.overfit_all,
    )
    train_dataset = ArmDemoDataset(str(dataset_dir), train_rows)
    validation_dataset = ArmDemoDataset(
        str(dataset_dir),
        validation_rows,
    )

    config = ArmBCConfig(
        n_arms=2,
        max_poses=16,
        grid_height=54,
        grid_width=24,
        occupancy_channels=4,
        part_feature_dim=26,
        pose_feature_dim=17,
    )
    if train_dataset.action_n != config.action_n:
        raise ValueError(
            f"dataset action_n={train_dataset.action_n}, "
            f"model action_n={config.action_n}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ArmBCPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print("=" * 76)
    print("Masked arm-conditioned BC")
    print(f"device              : {device}")
    print(f"dataset             : {dataset_dir}")
    print(f"train episodes      : {len(train_rows)}")
    print(f"validation episodes : {len(validation_rows)}")
    print(f"train transitions   : {len(train_dataset)}")
    print(f"validation trans.   : {len(validation_dataset)}")
    print(f"action count        : {config.action_n}")
    print(f"parameters          : {sum(p.numel() for p in model.parameters())}")
    print("=" * 76)

    history: List[Dict[str, Any]] = []
    best_metric = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / "arm_bc_best.pt"
    last_path = output_dir / "arm_bc_last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            logits = model.masked_flat_logits(batch)
            target = batch["joint_action"]

            if not torch.all(
                batch["action_mask"].gather(
                    1,
                    target[:, None],
                ).squeeze(1)
            ):
                raise RuntimeError("masked expert target reached trainer")

            loss = criterion(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            optimizer.step()

            running_loss += float(loss.item()) * int(target.numel())
            sample_count += int(target.numel())

        train_metrics = compute_metrics(
            model,
            train_eval_loader,
            device,
        )
        validation_metrics = compute_metrics(
            model,
            validation_loader,
            device,
        )
        row = {
            "epoch": epoch,
            "optimization_loss": running_loss / max(1, sample_count),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)

        # For overfit-all, train loss is the actual smoke-test objective.
        selection_loss = (
            train_metrics["loss"]
            if args.overfit_all
            else validation_metrics["loss"]
        )
        improved = selection_loss < best_metric - 1e-6
        if improved:
            best_metric = selection_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": "2026-07-17-arm-bc-v1",
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "epoch": epoch,
                    "selection_loss": selection_loss,
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                    "dataset_index": index,
                    "train_episode_ids": [
                        int(row["episode_id"]) for row in train_rows
                    ],
                    "validation_episode_ids": [
                        int(row["episode_id"])
                        for row in validation_rows
                    ],
                    "training_args": vars(args),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        ):
            print(
                f"epoch={epoch:04d} "
                + format_metrics("train", train_metrics)
            )
            print(
                " " * 11
                + format_metrics("valid", validation_metrics)
            )

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"[EARLY STOP] no selection-loss improvement for "
                f"{args.patience} epochs"
            )
            break

    torch.save(
        {
            "format_version": "2026-07-17-arm-bc-v1",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "epoch": history[-1]["epoch"],
            "train_metrics": history[-1]["train"],
            "validation_metrics": history[-1]["validation"],
            "dataset_index": index,
            "training_args": vars(args),
        },
        last_path,
    )

    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "best_epoch": best_epoch,
        "best_selection_loss": best_metric,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "overfit_all": bool(args.overfit_all),
        "train_episode_count": len(train_rows),
        "validation_episode_count": len(validation_rows),
        "train_transition_count": len(train_dataset),
        "validation_transition_count": len(validation_dataset),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 76)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 76)


if __name__ == "__main__":
    main()
