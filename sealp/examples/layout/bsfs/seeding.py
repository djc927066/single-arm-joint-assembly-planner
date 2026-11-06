"""Deterministic seeding for reproducible parallel BSFS execution.

Every stochastic evaluation (WRS IK random restarts drive ``evaluate_layout`` /
``StepOracle.certify``) derives its seed from *stable task identity* -- never from
worker id or process/completion order. Consequently ``--workers 1`` and
``--workers 4`` replay the SAME logical task with the SAME random sequence, so
process scheduling cannot change numerical results.

WRS IK uses the *global* NumPy RNG (``wrs.robot_sim._kinematics.jlchain.rand_conf``
and ``ik_trac`` call ``np.random.rand``), so seeding ``np.random`` (and Python
``random``) immediately before a stochastic call is sufficient to make it
deterministic.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

_MASK = 0x7FFFFFFF


def task_seed(base_seed, *parts) -> int:
    """Stable 31-bit seed from ``base_seed`` and an ordered list of identity parts.

    Uses SHA-256 (NOT Python's salted ``hash()``) so the mapping is identical
    across processes and runs. ``parts`` may be any stringifiable values
    (assembly-center id, depth, part id, candidate id, evaluation type, retry).
    """
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big") & _MASK


def seed_everything(s: int) -> None:
    """Seed Python ``random`` + global NumPy RNG (and torch if importable)."""
    s = int(s) & _MASK
    random.seed(s)
    np.random.seed(s)
    try:  # torch is optional; only seed if already available
        import torch  # noqa: WPS433
        torch.manual_seed(s)
    except Exception:  # pragma: no cover - torch not required at runtime
        pass


def center_id(center) -> str:
    """Stable id for an assembly center (rounded to avoid float noise)."""
    c = np.asarray(center, dtype=float).reshape(-1)
    return f"{c[0]:.4f},{c[1]:.4f}"
