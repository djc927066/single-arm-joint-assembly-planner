#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge compatible per-region IK-hint NPZ caches.

Region-indexed arrays are concatenated along axis 0. Scalar metadata and
non-region arrays must match exactly. Duplicate region IDs are rejected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REQUIRED_KEYS = (
    "region_ids",
    "part_ids",
    "scores",
    "left_scores",
    "right_scores",
)


def _equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.allclose(a, b, atol=1e-6, equal_nan=True))
    return bool(np.array_equal(a, b))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [
        Path(p).expanduser().resolve()
        for p in args.inputs
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    caches: List[Dict[str, np.ndarray]] = []
    for path in input_paths:
        with np.load(path, allow_pickle=True) as npz:
            missing = [k for k in REQUIRED_KEYS if k not in npz.files]
            if missing:
                raise ValueError(f"{path} missing keys {missing}")
            caches.append({k: np.asarray(npz[k]) for k in npz.files})

    first = caches[0]
    all_keys = set(first)
    for path, cache in zip(input_paths[1:], caches[1:]):
        if set(cache) != all_keys:
            missing = sorted(all_keys - set(cache))
            extra = sorted(set(cache) - all_keys)
            raise ValueError(
                f"{path} key mismatch; missing={missing}, extra={extra}"
            )

    region_ids: List[str] = []
    region_counts: List[int] = []
    for path, cache in zip(input_paths, caches):
        ids = [str(v) for v in cache["region_ids"].tolist()]
        if not ids:
            raise ValueError(f"{path} contains no region IDs")
        duplicates = sorted(set(ids).intersection(region_ids))
        if duplicates:
            raise ValueError(
                f"duplicate region IDs {duplicates} in {path}"
            )
        region_ids.extend(ids)
        region_counts.append(len(ids))

    output: Dict[str, np.ndarray] = {}
    for key in sorted(all_keys):
        arrays = [cache[key] for cache in caches]

        if key == "region_ids":
            width = max(1, max(len(v) for v in region_ids))
            output[key] = np.asarray(region_ids, dtype=f"<U{width}")
            continue

        is_region_indexed = all(
            array.ndim >= 1
            and array.shape[0] == region_count
            for array, region_count in zip(arrays, region_counts)
        )

        if is_region_indexed:
            reference_tail = arrays[0].shape[1:]
            reference_dtype = arrays[0].dtype
            for path, array in zip(input_paths[1:], arrays[1:]):
                if array.shape[1:] != reference_tail:
                    raise ValueError(
                        f"{key}: incompatible tail shape in {path}: "
                        f"{array.shape[1:]} vs {reference_tail}"
                    )
                if array.dtype != reference_dtype:
                    raise ValueError(
                        f"{key}: incompatible dtype in {path}: "
                        f"{array.dtype} vs {reference_dtype}"
                    )
            output[key] = np.concatenate(arrays, axis=0)
        else:
            reference = arrays[0]
            for path, array in zip(input_paths[1:], arrays[1:]):
                if not _equal(reference, array):
                    raise ValueError(
                        f"non-region metadata key {key!r} differs in {path}"
                    )
            output[key] = reference

    expected_scores = np.maximum(
        np.asarray(output["left_scores"], dtype=np.float32),
        np.asarray(output["right_scores"], dtype=np.float32),
    )
    if not np.allclose(
        np.asarray(output["scores"], dtype=np.float32),
        expected_scores,
        atol=1e-3,
        equal_nan=True,
    ):
        raise ValueError(
            "merged scores are inconsistent with max(left_scores,right_scores)"
        )

    output_npz = Path(args.output_npz).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_npz, **output)

    summary = {
        "inputs": [str(p) for p in input_paths],
        "output_npz": str(output_npz),
        "region_ids": region_ids,
        "part_ids": [str(v) for v in output["part_ids"].tolist()],
        "keys": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in output.items()
        },
        "positive_counts": {
            "left_scores": int(
                np.count_nonzero(output["left_scores"] > 0)
            ),
            "right_scores": int(
                np.count_nonzero(output["right_scores"] > 0)
            ),
            "scores": int(np.count_nonzero(output["scores"] > 0)),
        },
    }
    output_json.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"regions : {region_ids}")
    print(f"parts   : {summary['part_ids']}")
    print(f"scores  : {output['scores'].shape}")
    print(f"[OK] NPZ  : {output_npz}")
    print(f"[OK] JSON : {output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
