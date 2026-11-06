#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze evaluate_hint_policy JSON results.

Usage:
python -m sealp.examples.layout.assembly_rl.analyze_hint_policy_results \
  --input D:\Project\wrs-sealp\hint_policy_comparison_r1c1_20.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def total_counter(counter: Dict[str, int]) -> int:
    return int(sum(int(v) for v in (counter or {}).values()))


def pct_change(new: float, old: float) -> float:
    if abs(old) < 1e-12:
        return float("nan")
    return 100.0 * (new - old) / old


def reduction(new: float, old: float) -> float:
    if old <= 0:
        return float("nan")
    return 100.0 * (old - new) / old


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline", default="random")
    parser.add_argument("--reference", default="grasp_greedy")
    parser.add_argument("--target", default="grasp_ik_greedy")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    path = Path(args.input)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    comparison = payload.get("comparison", {})
    if not comparison:
        raise ValueError("input JSON has no 'comparison' object")

    report = {"input": str(path), "policies": {}, "comparisons": {}}

    print("=" * 76)
    print("Hint-policy result analysis")
    print("=" * 76)

    for name, stats in comparison.items():
        passed = int(stats.get("l2_pass_count", 0))
        episodes = int(stats.get("episode_count", 0))
        rate = passed / max(1, episodes)
        ci_low, ci_high = wilson_interval(passed, episodes)
        mean_score = stats.get("mean_layout_score_on_pass")
        exact_total = total_counter(stats.get("exact_rejection_counts", {}))
        fail_episode_total = total_counter(stats.get("fail_type_episode_counts", {}))
        fail_attempt_total = total_counter(stats.get("fail_type_attempt_counts", {}))

        row = {
            "pass": passed,
            "episodes": episodes,
            "rate": rate,
            "wilson95": [ci_low, ci_high],
            "mean_score_on_pass": mean_score,
            "exact_rejection_total": exact_total,
            "failed_episode_total": fail_episode_total,
            "failed_attempt_total": fail_attempt_total,
            "exact_rejection_counts": stats.get("exact_rejection_counts", {}),
            "fail_part_counts": stats.get("fail_part_counts", {}),
            "fail_type_episode_counts": stats.get("fail_type_episode_counts", {}),
            "fail_type_attempt_counts": stats.get("fail_type_attempt_counts", {}),
        }
        report["policies"][name] = row

        print()
        print(f"policy={name}")
        print(f"  L2 pass        : {passed}/{episodes} = {rate:.3f}")
        print(f"  Wilson 95% CI  : [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  mean pass score: {mean_score}")
        print(f"  exact rejects  : {exact_total}")
        print(f"  failure types  : {row['fail_type_episode_counts']}")
        print(f"  failed parts   : {row['fail_part_counts']}")

    target = report["policies"].get(args.target)
    for other_name in (args.baseline, args.reference):
        other = report["policies"].get(other_name)
        if target is None or other is None:
            continue
        comparison_row = {
            "pass_rate_absolute_gain": target["rate"] - other["rate"],
            "pass_rate_relative_change_percent": pct_change(target["rate"], other["rate"]),
            "mean_score_absolute_gain": (
                None
                if target["mean_score_on_pass"] is None or other["mean_score_on_pass"] is None
                else float(target["mean_score_on_pass"]) - float(other["mean_score_on_pass"])
            ),
            "mean_score_relative_change_percent": (
                None
                if target["mean_score_on_pass"] is None or other["mean_score_on_pass"] in (None, 0)
                else pct_change(
                    float(target["mean_score_on_pass"]),
                    float(other["mean_score_on_pass"]),
                )
            ),
            "exact_rejection_reduction_percent": reduction(
                target["exact_rejection_total"],
                other["exact_rejection_total"],
            ),
        }
        report["comparisons"][f"{args.target}_vs_{other_name}"] = comparison_row

        print()
        print(f"{args.target} vs {other_name}")
        print(
            f"  pass-rate gain       : "
            f"{100.0 * comparison_row['pass_rate_absolute_gain']:+.1f} percentage points"
        )
        print(
            f"  exact-reject reduction: "
            f"{comparison_row['exact_rejection_reduction_percent']:.1f}%"
        )
        if comparison_row["mean_score_absolute_gain"] is not None:
            print(
                f"  mean-score gain      : "
                f"{comparison_row['mean_score_absolute_gain']:+.4f}"
            )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] saved: {out}")


if __name__ == "__main__":
    main()
