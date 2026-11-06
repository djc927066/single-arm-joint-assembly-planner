"""Synthetic axis-aligned bounding-box assemblies for layout learning."""

from .utils import (
    MIN_EDGE_M,
    MAX_EDGE_M,
    random_bbox_extent,
    ensure_box_assets,
    SyntheticAssemblySpec,
    BBOX_PROTOTYPES,
    NUM_BBOX_PROTOTYPES,
    DEFAULT_PARTS_PER_ASSEMBLY,
)

__all__ = [
    "MIN_EDGE_M",
    "MAX_EDGE_M",
    "random_bbox_extent",
    "ensure_box_assets",
    "SyntheticAssemblySpec",
    "BBOX_PROTOTYPES",
    "NUM_BBOX_PROTOTYPES",
    "DEFAULT_PARTS_PER_ASSEMBLY",
]
