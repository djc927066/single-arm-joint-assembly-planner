"""Procedural box meshes, asmdef, and grasp pickles for synthetic layout data."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
import trimesh.creation as tcreation
import yaml

MIN_EDGE_M = 0.02
MAX_EDGE_M = 0.20

# 20 种 distinct bbox 外形模板 (shape_name, extent [lx, ly, lz] 米, 最长边<=0.20)
BBOX_PROTOTYPES: List[Tuple[str, Tuple[float, float, float]]] = [
    ("rod_xl", (0.200, 0.024, 0.024)),
    ("rod_l", (0.160, 0.030, 0.030)),
    ("rod_m", (0.120, 0.035, 0.035)),
    ("rod_s", (0.080, 0.040, 0.040)),
    ("bar_flat_x", (0.180, 0.060, 0.020)),
    ("bar_flat_y", (0.060, 0.180, 0.020)),
    ("plate_l", (0.160, 0.120, 0.020)),
    ("plate_m", (0.120, 0.090, 0.025)),
    ("plate_s", (0.080, 0.060, 0.020)),
    ("slab_thin", (0.140, 0.100, 0.020)),
    ("slab_wide", (0.200, 0.080, 0.022)),
    ("block_l", (0.120, 0.100, 0.080)),
    ("block_m", (0.100, 0.080, 0.060)),
    ("block_s", (0.060, 0.050, 0.040)),
    ("cube_l", (0.100, 0.100, 0.100)),
    ("cube_m", (0.080, 0.080, 0.080)),
    ("cube_s", (0.050, 0.050, 0.050)),
    ("brick", (0.120, 0.060, 0.040)),
    ("chunk", (0.090, 0.070, 0.055)),
    ("stub", (0.070, 0.070, 0.030)),
]

NUM_BBOX_PROTOTYPES = len(BBOX_PROTOTYPES)
DEFAULT_PARTS_PER_ASSEMBLY = 8


def _clip_extent(ext: np.ndarray, min_edge: float, max_edge: float) -> np.ndarray:
    lo, hi = float(min_edge), float(max_edge)
    ext = np.clip(np.asarray(ext, dtype=float).reshape(3), lo, hi)
    order = np.sort(ext)[::-1]
    # 保证最长边不超过 hi、最短边不低于 lo
    if float(order[0]) > hi:
        scale = hi / float(order[0])
        ext = ext * scale
    if float(ext.min()) < lo:
        ext = np.maximum(ext, lo)
    return ext.astype(float)


def _box_mesh(lx: float, ly: float, lz: float) -> trimesh.Trimesh:
    mesh = tcreation.box(extents=np.array([lx, ly, lz], dtype=float))
    mesh.vertices[:, 2] += lz / 2.0
    return mesh


def random_bbox_extent(rng: np.random.Generator,
                       min_edge: float = MIN_EDGE_M,
                       max_edge: float = MAX_EDGE_M) -> Tuple[np.ndarray, str]:
    """Return (extent [lx,ly,lz], shape_type) with shortest >= min, longest <= max."""
    idx = int(rng.integers(0, NUM_BBOX_PROTOTYPES))
    return bbox_extent_from_prototype(idx, rng, min_edge=min_edge, max_edge=max_edge)


def bbox_extent_from_prototype(prototype_idx: int,
                               rng: Optional[np.random.Generator] = None,
                               *,
                               min_edge: float = MIN_EDGE_M,
                               max_edge: float = MAX_EDGE_M,
                               jitter_frac: float = 0.08) -> Tuple[np.ndarray, str]:
    """Pick one of 20 prototypes; optional ±jitter on each edge for diversity."""
    idx = int(prototype_idx) % NUM_BBOX_PROTOTYPES
    shape, base = BBOX_PROTOTYPES[idx]
    ext = np.array(base, dtype=float)
    if rng is not None and jitter_frac > 0:
        noise = 1.0 + rng.uniform(-jitter_frac, jitter_frac, size=3)
        ext = ext * noise
    ext = _clip_extent(ext, min_edge, max_edge)
    return ext, str(shape)


def sample_distinct_bbox_extents(rng: np.random.Generator,
                                 count: int,
                                 *,
                                 min_edge: float = MIN_EDGE_M,
                                 max_edge: float = MAX_EDGE_M) -> List[Tuple[np.ndarray, str]]:
    """Sample ``count`` distinct prototype-based bboxes (without replacement up to 20)."""
    n = max(1, int(count))
    if n <= NUM_BBOX_PROTOTYPES:
        indices = rng.choice(NUM_BBOX_PROTOTYPES, size=n, replace=False)
    else:
        indices = np.concatenate([
            np.arange(NUM_BBOX_PROTOTYPES),
            rng.integers(0, NUM_BBOX_PROTOTYPES, size=n - NUM_BBOX_PROTOTYPES),
        ])
    return [
        bbox_extent_from_prototype(int(i), rng, min_edge=min_edge, max_edge=max_edge)
        for i in indices
    ]


def _layout_rel_positions(parts: List[SyntheticPartSpec],
                          *,
                          max_span_xy: Tuple[float, float] = (0.44, 0.44),
                          gap: float = 0.015) -> None:
    """Assign compact grid rel_pos so N parts fit in asmdef frame (goal assembly footprint)."""
    n = len(parts)
    if n == 0:
        return
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    fps = [np.asarray(p.extent, dtype=float)[:2] for p in parts]
    cell_w = max(float(fp[0]) for fp in fps) + gap
    cell_h = max(float(fp[1]) for fp in fps) + gap
    span_x = cols * cell_w
    span_y = rows * cell_h
    scale = min(max_span_xy[0] / max(span_x, 1e-6), max_span_xy[1] / max(span_y, 1e-6), 1.0)
    cell_w *= scale
    cell_h *= scale
    x0 = -0.5 * (cols - 1) * cell_w
    y0 = -0.5 * (rows - 1) * cell_h
    for i, part in enumerate(parts):
        r, c = divmod(i, cols)
        part.rel_pos = np.array([
            x0 + c * cell_w,
            y0 + r * cell_h,
            0.0,
        ], dtype=float)


@dataclass
class SyntheticPartSpec:
    part_id: str
    extent: np.ndarray
    shape_type: str
    prototype_idx: int = -1
    rel_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rel_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class SyntheticAssemblySpec:
    assembly_id: str
    parts: List[SyntheticPartSpec]
    part_order: List[str]

    @property
    def cache_key(self) -> str:
        payload = {
            "parts": [
                {
                    "id": p.part_id,
                    "extent": [float(x) for x in p.extent.reshape(-1)[:3]],
                    "shape": p.shape_type,
                    "proto": int(p.prototype_idx),
                    "rel_pos": [float(x) for x in p.rel_pos.reshape(-1)[:3]],
                }
                for p in self.parts
            ],
            "order": list(self.part_order),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_single_part_spec(rng: np.random.Generator,
                           min_edge: float = MIN_EDGE_M,
                           max_edge: float = MAX_EDGE_M,
                           *,
                           jitter_frac: float = 0.12) -> SyntheticAssemblySpec:
    """One random bbox part (prototype + jitter) for single-part init/goal pairs."""
    idx = int(rng.integers(0, NUM_BBOX_PROTOTYPES))
    extent, shape = bbox_extent_from_prototype(
        idx, rng, min_edge=min_edge, max_edge=max_edge, jitter_frac=jitter_frac,
    )
    part = SyntheticPartSpec(
        part_id="box_0",
        extent=extent,
        shape_type=shape,
        prototype_idx=idx,
        rel_pos=np.zeros(3, dtype=float),
    )
    digest = hashlib.sha256(
        json.dumps([float(x) for x in extent.reshape(-1)], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return SyntheticAssemblySpec(
        assembly_id=f"synthetic_1p_{digest}",
        parts=[part],
        part_order=["box_0"],
    )


def build_random_assembly_spec(rng: np.random.Generator,
                               num_parts: int,
                               min_edge: float = MIN_EDGE_M,
                               max_edge: float = MAX_EDGE_M,
                               *,
                               distinct_prototypes: bool = True) -> SyntheticAssemblySpec:
    """Build assembly with ``num_parts`` bbox parts (default design: 20 distinct prototypes)."""
    n = max(1, int(num_parts))
    if distinct_prototypes and n <= NUM_BBOX_PROTOTYPES:
        bbox_list = sample_distinct_bbox_extents(rng, n, min_edge=min_edge, max_edge=max_edge)
        proto_indices = rng.choice(NUM_BBOX_PROTOTYPES, size=n, replace=False)
    else:
        proto_indices = rng.integers(0, NUM_BBOX_PROTOTYPES, size=n)
        bbox_list = [
            bbox_extent_from_prototype(int(proto_indices[i]), rng,
                                       min_edge=min_edge, max_edge=max_edge)
            for i in range(n)
        ]

    parts: List[SyntheticPartSpec] = []
    for i in range(n):
        extent, shape = bbox_list[i]
        parts.append(SyntheticPartSpec(
            part_id=f"box_{i:02d}",
            extent=extent,
            shape_type=shape,
            prototype_idx=int(proto_indices[i]),
        ))
    _layout_rel_positions(parts)
    order = [p.part_id for p in parts]
    digest = hashlib.sha256(
        json.dumps([p.part_id for p in parts], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return SyntheticAssemblySpec(
        assembly_id=f"synthetic_{n}p_{digest}",
        parts=parts,
        part_order=order,
    )


def _write_asmdef(path: str, spec: SyntheticAssemblySpec, mesh_dir: str) -> None:
    models: Dict[str, Dict] = {}
    part_defs: Dict[str, Dict] = {}
    steps: List[Dict] = []
    for p in spec.parts:
        model_key = f"model_{p.part_id}"
        stl_name = f"{p.part_id}.stl"
        models[model_key] = {"path": f"mesh/{stl_name}"}
        vol = float(np.prod(p.extent))
        part_defs[p.part_id] = {
            "name": p.part_id,
            "model": model_key,
            "mass": max(vol * 800.0, 0.01),
        }
    for step_idx, p in enumerate(spec.parts):
        parent = "fixture" if step_idx == 0 else spec.parts[0].part_id
        steps.append({
            "step": step_idx,
            "part": p.part_id,
            "parent": parent,
            "rel_pos": [float(x) for x in p.rel_pos.reshape(-1)[:3]],
            "rel_rotmat": np.asarray(p.rel_rotmat, dtype=float).reshape(3, 3).tolist(),
            "deps": [],
        })
    doc = {
        "format_version": "1.0",
        "name": spec.assembly_id,
        "description": "Synthetic axis-aligned box assembly for layout ranker training.",
        "models": models,
        "parts": part_defs,
        "assembly": steps,
    }
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(doc, stream, sort_keys=False, allow_unicode=True)


def _export_stl(mesh_dir: str, part: SyntheticPartSpec) -> str:
    os.makedirs(mesh_dir, exist_ok=True)
    path = os.path.join(mesh_dir, f"{part.part_id}.stl")
    if not os.path.isfile(path):
        ext = np.asarray(part.extent, dtype=float).reshape(3)
        _box_mesh(float(ext[0]), float(ext[1]), float(ext[2])).export(path)
    return path


def _plan_grasps_if_missing(stl_path: str, grasp_path: str,
                            max_samples: int = 80) -> int:
    if os.path.isfile(grasp_path):
        try:
            from wrs.grasping.grasp import GraspCollection
            gc = GraspCollection.load_from_disk(file_name=grasp_path)
            return len(gc)
        except Exception:
            pass
    import wrs.basis.robot_math as rm
    import wrs.modeling.collision_model as mcm
    from sealp.examples.grasp.planning import plan_grasps

    obj = mcm.CollisionModel(stl_path)
    gc, _ = plan_grasps(obj, max_samples=int(max_samples), rotation_interval=rm.radians(30))
    os.makedirs(os.path.dirname(grasp_path), exist_ok=True)
    gc.save_to_disk(file_name=grasp_path)
    return len(gc)


def ensure_box_assets(spec: SyntheticAssemblySpec,
                      assets_root: str,
                      *,
                      max_grasp_samples: int = 80,
                      force_replan: bool = False) -> Dict[str, str]:
    """Create asmdef, STLs, grasp pickles under assets_root/<cache_key>/."""
    bundle_dir = os.path.join(assets_root, spec.cache_key)
    mesh_dir = os.path.join(bundle_dir, "mesh")
    grasp_dir = os.path.join(bundle_dir, "grasp")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(grasp_dir, exist_ok=True)

    for part in spec.parts:
        _export_stl(mesh_dir, part)
        grasp_path = os.path.join(grasp_dir, f"{part.part_id}_grasps.pickle")
        if force_replan and os.path.isfile(grasp_path):
            os.remove(grasp_path)
        _plan_grasps_if_missing(
            os.path.join(mesh_dir, f"{part.part_id}.stl"),
            grasp_path,
            max_samples=max_grasp_samples,
        )

    asmdef_path = os.path.join(bundle_dir, f"{spec.assembly_id}.asmdef")
    if not os.path.isfile(asmdef_path):
        _write_asmdef(asmdef_path, spec, mesh_dir)

    meta_path = os.path.join(bundle_dir, "meta.json")
    if not os.path.isfile(meta_path):
        meta = {
            "assembly_id": spec.assembly_id,
            "cache_key": spec.cache_key,
            "part_order": list(spec.part_order),
            "num_bbox_prototypes": NUM_BBOX_PROTOTYPES,
            "parts": [
                {
                    "part_id": p.part_id,
                    "extent": [float(x) for x in p.extent.reshape(-1)[:3]],
                    "shape_type": p.shape_type,
                    "prototype_idx": int(p.prototype_idx),
                }
                for p in spec.parts
            ],
        }
        with open(meta_path, "w", encoding="utf-8") as stream:
            json.dump(meta, stream, indent=2, ensure_ascii=False)

    return {
        "bundle_dir": bundle_dir,
        "asmdef_path": asmdef_path,
        "grasp_dir": grasp_dir,
        "mesh_dir": mesh_dir,
        "meta_path": meta_path,
    }


def deterministic_init_anchor(part_id: str,
                              station_xy: np.ndarray,
                              anchors: Sequence[np.ndarray]) -> np.ndarray:
    """Pick a balanced table anchor from precomputed grid (deterministic per part+station)."""
    if not anchors:
        return np.asarray(station_xy, dtype=float)[:2]
    key = f"{part_id}:{float(station_xy[0]):.4f}:{float(station_xy[1]):.4f}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(anchors)
    return np.asarray(anchors[idx], dtype=float)[:2]
