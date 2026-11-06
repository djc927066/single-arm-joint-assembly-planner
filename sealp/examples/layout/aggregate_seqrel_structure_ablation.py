"""Read-only aggregation for SeqRel structure ablations.

This script never trains a model.  By default it requires both no_relation and
no_sequence outputs.  ``--skip-missing`` permits pre-training validation, and
``--check-only`` performs all path/config/split checks without writing results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from layout_learning.dataset import load_jsonl
from run_seqrel_loss_ablation import _full_metrics, _same_split


REFERENCE_RUNS = {
    "DeepSets-reference": "deepsets/stratified/seed0",
    "SeqRel-full-reference": "seqrel/stratified/seed0",
}
STRUCTURE_RUNS = {
    "no_relation": {
        "directory": "no_relation",
        "disable_relation": True,
        "disable_sequence": False,
        "relation_variant": "no_relation",
    },
    "no_sequence": {
        "directory": "no_sequence",
        "disable_relation": False,
        "disable_sequence": True,
        "relation_variant": "no_sequence",
    },
}
REQUIRED_FILES = (
    "seqrel_best.pt", "metrics.json", "training_history.csv",
    "config.json", "split_indices.json", "train.log",
)


def _json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_variant(run_dir: Path, expected: Dict, shared_split: Path) -> List[str]:
    errors = [f"missing {name}" for name in REQUIRED_FILES
              if not (run_dir / name).is_file()]
    if errors:
        return errors
    config = _json(run_dir / "config.json")
    fixed = {
        "model_name": "seqrel",
        "feature_version": "v2",
        "split_mode": "stratified",
        "training_seed": 0,
        "hidden_dim": 64,
        "dropout": 0.2,
        "score_weight": 1.0,
        "rank_weight": 0.5,
        "fail_weight": 0.2,
        "use_focal": True,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        **{k: expected[k] for k in (
            "disable_relation", "disable_sequence", "relation_variant")},
    }
    for key, wanted in fixed.items():
        actual = config.get(key)
        if isinstance(wanted, float):
            ok = actual is not None and abs(float(actual) - wanted) < 1e-10
        else:
            ok = actual == wanted
        if not ok:
            errors.append(f"config.{key}={actual!r}, expected {wanted!r}")
    if not _same_split(run_dir / "split_indices.json", shared_split):
        errors.append("split differs from shared split")
    return errors


def _reference_row(
    name: str, run_dir: Path, samples: List[Dict], val_idx: List[int], device: str
) -> Dict:
    model = "deepsets" if name.startswith("DeepSets") else "seqrel"
    row = {
        "variant": name,
        "reference_only": True,
        "disable_relation": False,
        "disable_sequence": False,
        "relation_variant": "reference",
    }
    row.update(_full_metrics(
        run_dir / f"{model}_best.pt",
        run_dir / "metrics.json",
        samples, val_idx, device,
    ))
    return row


def _variant_row(
    name: str, run_dir: Path, expected: Dict,
    samples: List[Dict], val_idx: List[int], device: str
) -> Dict:
    row = {
        "variant": name,
        "reference_only": False,
        "disable_relation": expected["disable_relation"],
        "disable_sequence": expected["disable_sequence"],
        "relation_variant": expected["relation_variant"],
    }
    row.update(_full_metrics(
        run_dir / "seqrel_best.pt",
        run_dir / "metrics.json",
        samples, val_idx, device,
    ))
    return row


def parse_args():
    root = _THIS_DIR.parents[2]
    repro = root / "checkpoints/layout_models_repro"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-missing", action="store_true",
                        help="skip absent structure runs; references are still checked")
    parser.add_argument("--check-only", action="store_true",
                        help="validate inputs only; do not write CSV/JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "sealp/examples/layout/_output/layout_dataset_v2.jsonl",
    )
    parser.add_argument("--repro-root", type=Path, default=repro)
    parser.add_argument(
        "--structure-root", type=Path,
        default=repro / "seqrel_structure_ablation",
    )
    parser.add_argument(
        "--shared-split", type=Path,
        default=repro / "_splits/stratified/seed0/split_indices.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "sealp/examples/layout/_output/seqrel_experiments",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.dataset, args.shared_split):
        if not path.is_file():
            raise FileNotFoundError(path)
    split = _json(args.shared_split)
    val_idx = [int(i) for i in split["val_indices"]]
    samples = load_jsonl(str(args.dataset))
    print(f"[aggregate-check] shared split train={len(split['train_indices'])} "
          f"val={len(val_idx)}")

    rows = []
    for name, relative in REFERENCE_RUNS.items():
        run_dir = args.repro_root / relative
        rows.append(_reference_row(name, run_dir, samples, val_idx, args.device))
        print(f"[aggregate-check] reference valid: {name}")

    missing = []
    for name, expected in STRUCTURE_RUNS.items():
        run_dir = args.structure_root / expected["directory"]
        errors = _validate_variant(run_dir, expected, args.shared_split)
        if errors:
            if args.skip_missing and all(e.startswith("missing ") for e in errors):
                missing.append(name)
                print(f"[aggregate-check] skipped missing: {name}")
                continue
            raise RuntimeError(f"{name} validation failed: {errors}")
        rows.append(_variant_row(
            name, run_dir, expected, samples, val_idx, args.device))
        print(f"[aggregate-check] structure run valid: {name}")

    if args.check_only:
        print(f"[aggregate-check] check-only passed; missing={missing}; no files written")
        return
    if missing and not args.skip_missing:
        raise RuntimeError(f"missing required variants: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "seqrel_structure_ablation.csv"
    json_path = args.output_dir / "seqrel_structure_ablation.json"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": str(args.dataset.resolve()),
            "shared_split": str(args.shared_split.resolve()),
            "runs": rows,
            "missing_skipped": missing,
            "future_design_not_implemented": {
                "model_name": "seqrel_dynrel",
                "edge_attributes": [
                    "normalized staging dx", "normalized staging dy",
                    "staging distance", "footprint-based separation",
                ],
                "topology": [
                    "staging-space kNN", "order-adjacent", "parent-child",
                ],
                "message_passing_layers": 1,
                "residual": "h_new = h + alpha * message; alpha init ~0.1",
                "constraints": "keep feature v2 node dimension, pooling, and heads",
                "status": "design recorded only; not implemented or trained",
            },
        }, f, ensure_ascii=False, indent=2)
    print("[aggregate] wrote", csv_path)
    print("[aggregate] wrote", json_path)


if __name__ == "__main__":
    main()
