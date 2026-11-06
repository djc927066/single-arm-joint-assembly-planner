#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect an IK hint NPZ and verify arm-specific channels.

This script is intentionally tolerant of cache-layout changes. It prints every
key and attempts to identify scalar, left-arm and right-arm score arrays.

Usage:
python -m sealp.examples.layout.assembly_rl.inspect_ik_hint_by_arm \
  --input D:\Project\wrs-sealp\hint_cache\tower_ik_hint_r1c1.npz \
  --output-json D:\Project\wrs-sealp\hint_cache\tower_ik_hint_r1c1_arm_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


LEFT_TOKENS = ("left", "lft")
RIGHT_TOKENS = ("right", "rgt")
SCALAR_TOKENS = ("score", "scores", "ik_hint", "values")


def numeric_array(value: np.ndarray) -> bool:
    return isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)


def stats(array: np.ndarray) -> Dict[str, object]:
    a = np.asarray(array)
    finite = np.isfinite(a)
    valid = a[finite]
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite_count": int(finite.sum()),
        "total_count": int(a.size),
        "min": None if valid.size == 0 else float(valid.min()),
        "mean": None if valid.size == 0 else float(valid.mean()),
        "max": None if valid.size == 0 else float(valid.max()),
        "positive_count": int(np.count_nonzero(valid > 0.0)),
        "zero_count": int(np.count_nonzero(valid == 0.0)),
    }


def candidate_keys(keys: Iterable[str], tokens: Tuple[str, ...]) -> List[str]:
    return [k for k in keys if any(token in k.lower() for token in tokens)]


def same_shape_pairs(
    arrays: Dict[str, np.ndarray],
    left_keys: List[str],
    right_keys: List[str],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for lk in left_keys:
        for rk in right_keys:
            if arrays[lk].shape == arrays[rk].shape:
                pairs.append((lk, rk))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    path = Path(args.input)
    with np.load(path, allow_pickle=True) as npz:
        raw = {key: npz[key] for key in npz.files}

    arrays = {k: v for k, v in raw.items() if numeric_array(v)}
    keys = list(raw.keys())
    left_keys = candidate_keys(arrays.keys(), LEFT_TOKENS)
    right_keys = candidate_keys(arrays.keys(), RIGHT_TOKENS)
    scalar_keys = [
        k for k in candidate_keys(arrays.keys(), SCALAR_TOKENS)
        if k not in left_keys and k not in right_keys
    ]

    report: Dict[str, object] = {
        "input": str(path),
        "keys": {},
        "left_candidates": left_keys,
        "right_candidates": right_keys,
        "scalar_candidates": scalar_keys,
        "arm_pairs": [],
        "max_consistency": [],
    }

    print("=" * 80)
    print(f"IK cache audit: {path}")
    print("=" * 80)

    for key, value in raw.items():
        if numeric_array(value):
            row = stats(value)
        else:
            row = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "value_preview": repr(value.tolist())[:300],
            }
        report["keys"][key] = row
        print(f"{key:40s} shape={tuple(value.shape)!s:18s} dtype={value.dtype}")

    pairs = same_shape_pairs(arrays, left_keys, right_keys)
    for lk, rk in pairs:
        left = np.asarray(arrays[lk], dtype=np.float32)
        right = np.asarray(arrays[rk], dtype=np.float32)
        max_arm = np.maximum(left, right)
        dominance = {
            "left_key": lk,
            "right_key": rk,
            "shape": list(left.shape),
            "left_greater_count": int(np.count_nonzero(left > right)),
            "right_greater_count": int(np.count_nonzero(right > left)),
            "tie_count": int(np.count_nonzero(np.isclose(left, right, atol=args.atol))),
            "both_zero_count": int(np.count_nonzero((left <= 0.0) & (right <= 0.0))),
            "either_positive_count": int(np.count_nonzero(max_arm > 0.0)),
            "left_stats": stats(left),
            "right_stats": stats(right),
            "max_stats": stats(max_arm),
        }
        report["arm_pairs"].append(dominance)

        print()
        print(f"[arm pair] {lk}  <->  {rk}")
        print(f"  shape         : {left.shape}")
        print(f"  left greater  : {dominance['left_greater_count']}")
        print(f"  right greater : {dominance['right_greater_count']}")
        print(f"  ties          : {dominance['tie_count']}")
        print(f"  either > 0    : {dominance['either_positive_count']}")

        for sk in scalar_keys:
            scalar = np.asarray(arrays[sk])
            if scalar.shape != left.shape:
                continue
            diff = np.abs(np.asarray(scalar, dtype=np.float32) - max_arm)
            row = {
                "scalar_key": sk,
                "left_key": lk,
                "right_key": rk,
                "shape": list(diff.shape),
                "max_abs_error": float(np.nanmax(diff)),
                "mean_abs_error": float(np.nanmean(diff)),
                "allclose": bool(np.allclose(scalar, max_arm, atol=args.atol, equal_nan=True)),
            }
            report["max_consistency"].append(row)
            print(
                f"  scalar={sk}: allclose(max(left,right))={row['allclose']} "
                f"max_abs_error={row['max_abs_error']:.3e}"
            )

    if not pairs:
        print()
        print("[WARN] No same-shape left/right score pair was identified.")
        print("       Check the key list above before modifying the environment.")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] saved: {out}")


if __name__ == "__main__":
    main()
