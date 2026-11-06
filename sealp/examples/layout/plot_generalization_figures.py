#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot generalization_summary.csv for paper figures (MLP vs DeepSets)."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV = os.path.join(
    _THIS_DIR, "_output", "generalization_v3", "generalization_summary.csv"
)
_DEFAULT_OUT = os.path.join(_THIS_DIR, "_output", "image")

SPLIT_ORDER = ["random", "stratified", "seed_holdout", "region_holdout"]
SPLIT_LABELS = {
    "random": "Random",
    "stratified": "Stratified",
    "seed_holdout": "Seed holdout",
    "region_holdout": "Region holdout",
}
MODEL_ORDER = ["mlp", "deepsets"]
MODEL_LABELS = {"mlp": "MLP", "deepsets": "DeepSets"}
MODEL_COLORS = {"mlp": "#4C72B0", "deepsets": "#DD8452"}


def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _metric_table(rows: List[Dict], metric: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {m: {} for m in MODEL_ORDER}
    for r in rows:
        model = r["model"]
        split = r["split_mode"]
        if model not in out or split not in SPLIT_ORDER:
            continue
        out[model][split] = float(r[metric])
    return out


def _bar_panel(ax, rows: List[Dict], metric: str, ylabel: str, title: str) -> None:
    table = _metric_table(rows, metric)
    x = np.arange(len(SPLIT_ORDER))
    width = 0.36
    for i, model in enumerate(MODEL_ORDER):
        vals = [table[model].get(s, np.nan) for s in SPLIT_ORDER]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=True)


def _transfer_gap_panel(ax, rows: List[Dict]) -> None:
    """Region-holdout transfer gap: in-task random minus holdout."""
    by_model: Dict[str, Dict[str, float]] = {m: {} for m in MODEL_ORDER}
    for r in rows:
        if r["split_mode"] != "region_holdout":
            continue
        by_model[r["model"]]["roc"] = float(r["gap_roc_auc"])
        by_model[r["model"]]["pr"] = float(r["gap_pr_auc"])
        by_model[r["model"]]["comp"] = float(r["gap_composite"])

    metrics = [("roc", "ROC-AUC gap"), ("pr", "PR-AUC gap"), ("comp", "Composite gap")]
    x = np.arange(len(metrics))
    width = 0.36
    for i, model in enumerate(MODEL_ORDER):
        vals = [by_model[model].get(k, np.nan) for k, _ in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.008 if val >= 0 else -0.025),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8,
                )
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Gap (random in-task − region holdout)")
    ax.set_title("Region-holdout transfer gap (lower is better)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=True)


def _save(fig, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"[plot] saved {path}")


def plot_all(csv_path: str, out_dir: str) -> None:
    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"empty csv: {csv_path}")

    # Composite
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _bar_panel(ax, rows, "composite", "Composite score", "Offline ranking quality (composite)")
    fig.tight_layout()
    _save(fig, out_dir, "generalization_v3_composite")
    plt.close(fig)

    # PR-AUC
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _bar_panel(ax, rows, "pr_auc", "PR-AUC", "Feasibility ranking (PR-AUC)")
    fig.tight_layout()
    _save(fig, out_dir, "generalization_v3_pr_auc")
    plt.close(fig)

    # ROC-AUC (bonus, same layout)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _bar_panel(ax, rows, "roc_auc", "ROC-AUC", "Feasibility ranking (ROC-AUC)")
    fig.tight_layout()
    _save(fig, out_dir, "generalization_v3_roc_auc")
    plt.close(fig)

    # Transfer gap
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _transfer_gap_panel(ax, rows)
    fig.tight_layout()
    _save(fig, out_dir, "generalization_v3_transfer_gap")
    plt.close(fig)

    # Combined 2-panel figure for paper
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    _bar_panel(axes[0], rows, "composite", "Composite score", "(a) Composite")
    _bar_panel(axes[1], rows, "pr_auc", "PR-AUC", "(b) PR-AUC")
    fig.suptitle("MLP vs DeepSets on tower layout dataset (v3, n=2293)", y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "generalization_v3_paper_main")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot generalization summary figures")
    p.add_argument("--csv", default=_DEFAULT_CSV)
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    args = p.parse_args()
    plot_all(os.path.abspath(args.csv), os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
