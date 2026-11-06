"""Relative paths & run manifests for cross-machine migration."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def find_repo_root(start: Optional[str] = None) -> str:
    """Walk up until ``sealp/config/sample_config.yaml`` exists."""
    cur = os.path.abspath(start or os.getcwd())
    for _ in range(12):
        probe = os.path.join(cur, "sealp", "config", "sample_config.yaml")
        if os.path.isfile(probe):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start or os.getcwd())


def relpath(path: str, base: Optional[str] = None) -> str:
    base = find_repo_root(base)
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, base).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


def write_run_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
