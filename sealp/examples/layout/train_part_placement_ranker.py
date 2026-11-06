"""Train PartPlacementRanker on synthetic_bbox JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, Subset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning.models.part_placement_ranker import PartPlacementRankerNet
from layout_learning.part_placement_dataset import (
    PartPlacementRankingDataset,
    collate_ranking_batch,
    train_val_split,
)
from sealp.examples.layout.synthetic_bbox.portable import find_repo_root, relpath, write_run_manifest


def _parse_args():
    p = argparse.ArgumentParser(description="Train PartPlacementRanker")
    p.add_argument("--dataset", required=True)
    p.add_argument("--save-dir", default="checkpoints/part_placement_ranker")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128,
                   help="9950X3D CPU training: 128~256 often good")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader workers (-1 = min(12, cpu_count))")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="torch.set_num_threads (0=default; 9950X3D suggest 16)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="Ranker is small; CPU on 9950X3D is fine. Use cuda if available.")
    p.add_argument("--no-manifest", action="store_true")
    return p.parse_args()


def _ranking_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = target * mask
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    log_probs = Fn.log_softmax(logits, dim=-1)
    return -(target * log_probs * mask).sum(dim=-1).mean()


@torch.no_grad()
def _eval_loader(model: PartPlacementRankerNet, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    hit1 = 0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            batch["part_static"], batch["global_feat"],
            batch["cand_feat"], batch["cand_mask"],
        )
        loss = _ranking_loss(out["cand_logits"], batch["target"], batch["cand_mask"])
        loss_sum += float(loss.item()) * batch["part_static"].shape[0]
        pred = out["cand_scores"].argmax(dim=-1)
        true = batch["target"].argmax(dim=-1)
        hit1 += int((pred == true).sum().item())
        n += batch["part_static"].shape[0]
    return {"loss": loss_sum / max(n, 1), "top1_acc": hit1 / max(n, 1)}


def main():
    args = _parse_args()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))

    cpu_count = os.cpu_count() or 4
    num_workers = int(args.num_workers)
    if num_workers < 0:
        num_workers = min(12, cpu_count)

    torch.manual_seed(int(args.seed))
    os.makedirs(args.save_dir, exist_ok=True)
    repo_root = find_repo_root(_THIS_DIR)

    dataset = PartPlacementRankingDataset(args.dataset)
    if len(dataset) == 0:
        raise SystemExit(f"数据集为空或 schema 不匹配: {args.dataset}")

    train_idx, val_idx = train_val_split(dataset, val_frac=args.val_frac, seed=args.seed)
    loader_kw = {"num_workers": num_workers, "pin_memory": args.device.startswith("cuda")}
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=collate_ranking_batch,
        persistent_workers=num_workers > 0,
        **loader_kw,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_ranking_batch,
        persistent_workers=num_workers > 0,
        **loader_kw,
    )

    model = PartPlacementRankerNet(hidden=int(args.hidden), dropout=float(args.dropout))
    model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    if not args.no_manifest:
        write_run_manifest(os.path.join(args.save_dir, "train_manifest.json"), {
            "task": "part_placement_ranker_train",
            "repo_root": repo_root,
            "dataset": relpath(args.dataset, repo_root),
            "save_dir": relpath(args.save_dir, repo_root),
            "batch_size": int(args.batch_size),
            "num_workers": num_workers,
            "torch_threads": int(args.torch_threads),
            "device": args.device,
        })

    best_acc = -1.0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(
                batch["part_static"], batch["global_feat"],
                batch["cand_feat"], batch["cand_mask"],
            )
            loss = _ranking_loss(out["cand_logits"], batch["target"], batch["cand_mask"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * batch["part_static"].shape[0]
            n_train += batch["part_static"].shape[0]

        metrics = _eval_loader(model, val_loader, args.device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_train, 1),
            **metrics,
        }
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
              f"val_loss={metrics['loss']:.4f} top1={metrics['top1_acc']:.3f}")

        if metrics["top1_acc"] >= best_acc:
            best_acc = metrics["top1_acc"]
            ckpt = {
                "model_name": "part_placement_ranker",
                "state_dict": model.state_dict(),
                "hidden": int(args.hidden),
                "dropout": float(args.dropout),
                "dataset": relpath(args.dataset, repo_root),
                "metrics": metrics,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "part_placement_ranker_best.pt"))

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    print(f"Done. best top1={best_acc:.3f} -> {args.save_dir}")


if __name__ == "__main__":
    main()
