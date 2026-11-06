"""Pluggable task adapters for layout search scripts.

Each module exposes a ``register_task()`` factory returning a
``FastSearchTask`` plus ``STAGING_ROTMAT_CANDIDATES``.

* YuanChair → ``find_optimal_layout.py`` (rotmat 手写)
* Shelf Unit → ``find_optimal_layout_shelf.py`` (rotmat 由 ``auto_rotmat.py`` 推断)
"""

from typing import Callable, Dict, List, Tuple

import numpy as np

__all__ = ["TaskRegisterFn", "AVAILABLE_TASKS"]

TaskRegisterFn = Callable[..., Tuple[object, Dict[str, List[Tuple[np.ndarray, float]]]]]


def _shelf_unit_register(*args, **kwargs):
    """Lazy importer (keeps yuanchair-only runs from importing shelf_unit)."""
    from . import shelf_unit as _su
    return _su.register_task(*args, **kwargs)


AVAILABLE_TASKS: Dict[str, TaskRegisterFn] = {
    "shelf_unit": _shelf_unit_register,
}
