#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline evaluation of a masked arm-conditioned BC checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .arm_bc_model import ArmBCConfig, ArmBCPolicy
from .train_arm_bc import (
    ArmDemoDataset,
    compute_metrics,
    format_metrics,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = resolve_device(args.device)

    index = json.loads(
        (dataset_dir / "index.json").read_text(encoding="utf-8")
    )
    dataset = ArmDemoDataset(
        str(dataset_dir),
        index["episodes"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    config = ArmBCConfig(**checkpoint["model_config"])
    model = ArmBCPolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = compute_metrics(model, loader, device)
    print("=" * 76)
    print(f"checkpoint : {checkpoint_path}")
    print(f"dataset    : {dataset_dir}")
    print(format_metrics("all", metrics))
    print("=" * 76)

    if not 0.0 <= metrics["exact_top1"] <= 1.0:
        raise RuntimeError("invalid metric range")
    print("[OK] offline checkpoint evaluation passed")


if __name__ == "__main__":
    main()
