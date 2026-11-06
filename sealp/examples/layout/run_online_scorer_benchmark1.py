"""Fair online scorer benchmark using one persisted candidate pool per seed.

The neural scorer only ranks candidates. Every reported feasibility/score is
produced by the original evaluate_layout path in the neural search entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

from .layout_learning.repro_data import (
    load_manifest,
    sha256_file,
    verify_dataset_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    ROOT / "sealp/examples/layout/_output/datasets/"
    "layout_dataset_v2_repro_2293_manifest.json")
DEFAULT_CHECKPOINTS = {
    "deepsets": (
        "checkpoints/layout_models_repro/deepsets/stratified/seed0/"
        "deepsets_best.pt"),
    "seqrel": (
        "checkpoints/layout_models_repro/seqrel/stratified/seed0/"
        "seqrel_best.pt"),
    "dynaseqrel": (
        "checkpoints/layout_models_repro/"
        "dynaseqrel_dynedge_edge_mlp_v2/stratified/seed0/"
        "dynaseqrel_dynedge_best.pt"),
    "mlp": (
        "checkpoints/layout_models_official/mlp/stratified/seed0/"
        "mlp_best.pt"),
}
NEURAL_MODULE = (
    "sealp.examples.layout.find_optimal_initial_layout_tower_neural")


def _parse_csv(text: str) -> List[str]:
    return [value.strip() for value in text.replace(" ", ",").split(",")
            if value.strip()]


def _parse_checkpoint_map(text: str) -> Dict[str, str]:
    result = dict(DEFAULT_CHECKPOINTS)
    if not text:
        return result
    for item in _parse_csv(text):
        if "=" not in item:
            raise ValueError(
                "--checkpoints entries must be scorer=path")
        name, path = item.split("=", 1)
        result[name.strip()] = path.strip()
    return result


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorers", default="deepsets,seqrel,dynaseqrel")
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--search-seeds", default="0,1,2")
    parser.add_argument("--candidate-pool-seed", type=int, default=20260713)
    parser.add_argument("--global-max-evals", type=int, default=200)
    parser.add_argument(
        "--scorer-pool", type=int, default=2048,
        help="number of complete layouts in the shared candidate pool")
    parser.add_argument(
        "--candidate-pool-mode", default="coverage_grid",
        choices=("coverage_grid", "legacy_random", "cluster_anchor"))
    parser.add_argument("--pool-grid-spacing", type=float, default=0.02)
    parser.add_argument("--pool-oversample-factor", type=int, default=4)
    parser.add_argument("--pool-max-attempts", type=int, default=200000)
    parser.add_argument("--pool-pose-variants", type=int, default=3)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--top-k-proposals", type=int, default=64)
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--output-dir",
        default="sealp/examples/layout/_output/online_scorer_benchmark")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cdprim-type", default="box")
    parser.add_argument("--timeout-seconds", type=int, default=28800)
    parser.add_argument("--true-score-threshold", type=float, default=0.5)
    parser.add_argument("--common", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _config_hash(payload: Dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_metadata(path: str, strict: bool) -> Dict:
    if not os.path.isabs(path):
        path = str(ROOT / path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if strict:
        if checkpoint.get("feature_version") != "v2":
            raise ValueError(
                f"checkpoint feature_version is not v2: {path}")
        if checkpoint.get("model_name") not in {
                "deepsets", "seqrel", "dynaseqrel_dynedge", "mlp"}:
            raise ValueError(
                f"unsupported checkpoint model: {checkpoint.get('model_name')}")
    return {
        "path": os.path.abspath(path),
        "sha256": sha256_file(path),
        "model_name": checkpoint.get("model_name"),
        "feature_version": checkpoint.get("feature_version"),
        "epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("select_metric"),
    }


def _build_command(
    args,
    scorer: str,
    checkpoint: str,
    search_seed: int,
    pool_file: str,
    curve_file: str,
    output_name: str,
) -> List[str]:
    command = [
        args.python, "-m", NEURAL_MODULE,
        "--model", scorer,
        "--checkpoint", checkpoint,
        "--output-name", output_name,
        "--seed", str(search_seed),
        "--n-samples", str(args.top_k_proposals),
        "--top-k-proposals", str(args.top_k_proposals),
        "--scorer-pool", str(args.scorer_pool),
        "--candidate-pool-mode", args.candidate_pool_mode,
        "--pool-grid-spacing", str(args.pool_grid_spacing),
        "--pool-oversample-factor", str(args.pool_oversample_factor),
        "--pool-max-attempts", str(args.pool_max_attempts),
        "--pool-pose-variants", str(args.pool_pose_variants),
        "--score-batch-size", str(args.score_batch_size),
        "--station-mode", "grid3x3",
        "--rank-mode", "blend",
        "--rank-blend-feas", "0.7",
        "--candidate-pool-file", pool_file,
        "--candidate-pool-seed", str(args.candidate_pool_seed),
        "--eval-curve-out", curve_file,
        "--min-feasible", "0",
        "--fallback-explore", "0",
        "--global-max-evals", str(args.global_max_evals),
        "--global-elite", "1",
        "--no-refine",
        "--cdprim-type", args.cdprim_type,
    ]
    if args.strict:
        command.append("--benchmark-strict")
    command.extend(shlex.split(args.common))
    return command


def _best_at(curve: List[Dict], budget: int):
    candidates = [
        float(row["running_best_true_score"])
        for row in curve if int(row["eval_index"]) <= budget]
    return max(candidates, default=0.0)


def _summarize_curve(
    curve_payload: Dict,
    scorer: str,
    checkpoint: Dict,
    manifest: Dict,
    search_seed: int,
    threshold: float,
    total_time: float,
    output_name: str,
) -> Dict:
    curve = curve_payload["curve"]
    feasible_rows = [row for row in curve if row["true_feasible"]]
    first = feasible_rows[0] if feasible_rows else None
    threshold_rows = [
        row for row in curve
        if float(row["running_best_true_score"]) >= threshold]
    failures = Counter(
        row.get("failure_reason") or "unknown"
        for row in curve if not row["true_feasible"])
    final_best = max(
        (float(row["running_best_true_score"]) for row in curve),
        default=0.0)
    best_layout = (
        ROOT / "sealp/examples/layout/_output" / f"{output_name}.layout")
    return {
        "scorer": scorer,
        "checkpoint": checkpoint["path"],
        "checkpoint_hash": checkpoint["sha256"],
        "dataset_snapshot_hash": manifest["dataset_sha256"],
        "split_hash": manifest["split_sha256"],
        "search_seed": search_seed,
        "candidate_pool_seed": curve_payload["candidate_pool_seed"],
        "candidate_pool_hash": curve_payload["candidate_pool_sha256"],
        "candidate_count": curve_payload["candidate_count"],
        "pool_build_mode": curve_payload.get("pool_build_mode"),
        "grid_step": curve_payload.get("grid_step"),
        "candidate_ids_sha256": curve_payload.get("candidate_ids_sha256"),
        "pool_descriptor_summary": curve_payload.get("pool_descriptor_summary"),
        "exact_evaluation_budget": curve_payload["exact_real_evaluations"],
        "first_feasible_evaluation_index": (
            first["eval_index"] if first else None),
        "first_feasible_elapsed_time": (
            first["elapsed_seconds"] if first else None),
        "feasible_within_20": any(
            row["true_feasible"] and row["eval_index"] <= 20
            for row in curve),
        "feasible_within_50": any(
            row["true_feasible"] and row["eval_index"] <= 50
            for row in curve),
        "feasible_within_100": any(
            row["true_feasible"] and row["eval_index"] <= 100
            for row in curve),
        "feasible_within_200": any(
            row["true_feasible"] and row["eval_index"] <= 200
            for row in curve),
        "total_feasible_count": len(feasible_rows),
        "top_k_proposal_true_feasibility": (
            len(feasible_rows) / len(curve) if curve else 0.0),
        "best_true_score_at_20": _best_at(curve, 20),
        "best_true_score_at_50": _best_at(curve, 50),
        "best_true_score_at_100": _best_at(curve, 100),
        "best_true_score_at_200": _best_at(curve, 200),
        "final_best_true_score": final_best,
        "time_to_true_score_threshold": (
            threshold_rows[0]["elapsed_seconds"]
            if threshold_rows else None),
        "true_score_threshold": threshold,
        "scorer_inference_time": curve_payload["inference_time_seconds"],
        "evaluate_layout_time": sum(
            float(row["evaluation_time_seconds"]) for row in curve),
        "total_search_time": total_time,
        "failure_reasons": dict(failures),
        "best_layout_path": str(best_layout) if best_layout.is_file() else None,
        "best_so_far_curve": [
            {
                "evaluation": row["eval_index"],
                "elapsed_seconds": row["elapsed_seconds"],
                "best_true_score": row["running_best_true_score"],
            }
            for row in curve
        ],
    }


def main() -> None:
    args = _parse_args()
    scorers = _parse_csv(args.scorers)
    seeds = [int(value) for value in _parse_csv(args.search_seeds)]
    checkpoint_map = _parse_checkpoint_map(args.checkpoints)
    manifest = load_manifest(args.dataset_manifest)
    snapshot = ROOT / manifest["snapshot_path"]
    split = snapshot.with_name(
        "layout_dataset_v2_repro_2293_split_indices.json")
    verify_dataset_manifest(
        str(snapshot), args.dataset_manifest, str(split),
        require_manifest=True)

    checkpoints = {}
    for scorer in scorers:
        if scorer == "global":
            continue
        if scorer not in checkpoint_map:
            raise ValueError(f"no checkpoint configured for {scorer}")
        checkpoints[scorer] = _checkpoint_metadata(
            checkpoint_map[scorer], args.strict)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    configuration = {
        "scorers": scorers,
        "search_seeds": seeds,
        "candidate_pool_seed": args.candidate_pool_seed,
        "global_max_evals": args.global_max_evals,
        "scorer_pool": args.scorer_pool,
        "candidate_pool_mode": args.candidate_pool_mode,
        "pool_grid_spacing": args.pool_grid_spacing,
        "pool_oversample_factor": args.pool_oversample_factor,
        "pool_max_attempts": args.pool_max_attempts,
        "pool_pose_variants": args.pool_pose_variants,
        "score_batch_size": args.score_batch_size,
        "top_k_proposals": args.top_k_proposals,
        "cdprim_type": args.cdprim_type,
        "dataset_manifest": os.path.abspath(args.dataset_manifest),
        "dataset_sha256": manifest["dataset_sha256"],
        "split_sha256": manifest["split_sha256"],
        "checkpoints": checkpoints,
        "strict": args.strict,
        "common": args.common,
    }
    config_hash = _config_hash(configuration)
    commands = []
    for search_seed in seeds:
        pool_file = str(
            output_dir / "pools"
            / f"pool_seed{args.candidate_pool_seed}_search{search_seed}.json")
        for scorer in scorers:
            output_name = f"online_bench_{scorer}_seed{search_seed}"
            curve_file = str(
                output_dir / "curves" / f"{scorer}_seed{search_seed}.json")
            checkpoint = (
                checkpoints[scorer]["path"] if scorer != "global" else "")
            command = _build_command(
                args, scorer, checkpoint, search_seed, pool_file,
                curve_file, output_name)
            if scorer == "global":
                checkpoint_index = command.index("--checkpoint")
                del command[checkpoint_index:checkpoint_index + 2]
                command.append("--random-pool-order")
            commands.append((scorer, search_seed, output_name, curve_file,
                             command))

    print(json.dumps({
        "configuration": configuration,
        "configuration_hash": config_hash,
        "commands": [" ".join(command) for *_, command in commands],
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "curves").mkdir(exist_ok=True)
    (output_dir / "pools").mkdir(exist_ok=True)
    results = []
    for scorer, search_seed, output_name, curve_file, command in commands:
        result_file = (
            output_dir / f"{scorer}_seed{search_seed}_result.json")
        if args.resume and result_file.is_file():
            previous = json.loads(result_file.read_text(encoding="utf-8"))
            if previous.get("configuration_hash") != config_hash:
                raise RuntimeError(
                    f"resume configuration mismatch: {result_file}")
            results.append(previous["summary"])
            continue
        log_file = output_dir / "logs" / f"{scorer}_seed{search_seed}.log"
        started = time.perf_counter()
        with log_file.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds)
        total_time = time.perf_counter() - started
        if process.returncode != 0:
            raise RuntimeError(
                f"online benchmark failed ({scorer}, seed={search_seed}); "
                f"see {log_file}")
        curve_payload = json.loads(
            Path(curve_file).read_text(encoding="utf-8"))
        checkpoint = checkpoints.get(scorer, {
            "path": "shared_pool_random_order",
            "sha256": None,
        })
        summary = _summarize_curve(
            curve_payload, scorer, checkpoint, manifest, search_seed,
            args.true_score_threshold, total_time, output_name)
        result_file.write_text(json.dumps({
            "configuration_hash": config_hash,
            "summary": summary,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(summary)

    with (output_dir / "benchmark_results.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        flat_rows = [{
            key: (json.dumps(value, ensure_ascii=False)
                  if isinstance(value, (dict, list)) else value)
            for key, value in row.items()
        } for row in results]
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps({
            "configuration": configuration,
            "configuration_hash": config_hash,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
