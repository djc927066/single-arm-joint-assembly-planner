#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STL geometry preprocessing and PointNet interface for assembly layout.

The script is deliberately independent of WRS/SEALP.  It reads model paths from
an ASMDEF file, samples 256 surface points per *unique STL*, normalises the point
cloud, computes explicit shape descriptors and saves reusable arrays.

Identical models are encoded once.  For example, the four legs in a chair share
one model sample/embedding; ASMDEF graph and target-pose features distinguish the
individual leg instances later.

No network training is performed here.  When PyTorch is installed, this module
also exposes:

* ``PointNetGeometryEncoder``: point cloud -> geometry embedding;
* ``PointNetWithDescriptors``: point cloud + explicit descriptors -> embedding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import trimesh
import yaml

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "extent_x_over_scale",
    "extent_y_over_scale",
    "extent_z_over_scale",
    "bbox_volume_over_scale3",
    "surface_area_over_scale2",
    "convex_volume_over_scale3",
    "convex_fill_ratio",
    "watertight",
    "pca_lambda1_ratio",
    "pca_lambda2_ratio",
    "pca_lambda3_ratio",
    "linearity",
    "planarity",
    "scattering",
    "slenderness",
    "flatness",
    "minor_axis_similarity",
    "radial_distance_mean",
    "radial_distance_std",
    "point_radius_mean",
    "point_radius_std",
    "log_vertices",
    "log_faces",
    "valid_mesh",
)


@dataclass
class ModelGeometry:
    model_id: str
    source_path: str
    points: np.ndarray                 # [N,3], centred and scale-normalised
    descriptors: np.ndarray            # [D]
    center: np.ndarray                 # raw STL coordinates
    scale: float                       # max raw AABB extent
    raw_extents: np.ndarray             # metres
    metadata: Dict[str, Any]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_path": self.source_path,
            "point_count": int(self.points.shape[0]),
            "descriptor_dim": int(self.descriptors.shape[0]),
            "center": self.center.astype(float).tolist(),
            "scale": float(self.scale),
            "raw_extents": self.raw_extents.astype(float).tolist(),
            "metadata": self.metadata,
            "descriptors": {
                name: float(value)
                for name, value in zip(DESCRIPTOR_NAMES, self.descriptors)
            },
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def load_asmdef(asmdef_path: str) -> Dict[str, Any]:
    path = Path(asmdef_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"invalid ASMDEF root: {path}")
    return dict(data)


def _basename_any_platform(path_text: str) -> str:
    # PureWindowsPath correctly extracts a basename from D:\... on Linux too.
    return PureWindowsPath(path_text).name or Path(path_text).name


def resolve_model_path(
    raw_path: str,
    *,
    asmdef_dir: Path,
    search_roots: Sequence[Path] = (),
) -> Path:
    """Resolve native, relative or Windows-authored model paths.

    Resolution order:
    1. path exactly as authored;
    2. path relative to the ASMDEF directory;
    3. ``search_root / basename``;
    4. recursive basename search below each root (first unique match).
    """
    raw = str(raw_path)
    candidates: List[Path] = []
    direct = Path(raw).expanduser()
    candidates.append(direct)
    candidates.append(asmdef_dir / direct)
    basename = _basename_any_platform(raw)
    for root in search_roots:
        candidates.append(root / basename)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved

    recursive_matches: List[Path] = []
    for root in search_roots:
        if root.is_dir():
            recursive_matches.extend(p.resolve() for p in root.rglob(basename) if p.is_file())
    unique = sorted(set(recursive_matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise FileNotFoundError(
            f"model basename {basename!r} is ambiguous under search roots: {unique}"
        )
    raise FileNotFoundError(
        f"cannot resolve model path {raw!r}; asmdef_dir={asmdef_dir}, "
        f"search_roots={[str(p) for p in search_roots]}"
    )


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force=None, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"scene contains no mesh geometry: {path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"unsupported geometry type {type(loaded)!r}: {path}")
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        raise ValueError(f"mesh has insufficient vertices/faces: {path}")
    return mesh


def sample_surface_points(
    mesh: trimesh.Trimesh,
    *,
    point_count: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Area-weighted deterministic triangle sampling with local RNG."""
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    valid = np.isfinite(areas) & (areas > 1e-15)
    if not np.any(valid):
        raise ValueError("mesh has no positive-area triangles")
    valid_indices = np.flatnonzero(valid)
    probabilities = areas[valid] / areas[valid].sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_indices, size=point_count, replace=True, p=probabilities)
    tri = triangles[chosen]

    r1 = np.sqrt(rng.random(point_count))
    r2 = rng.random(point_count)
    weights0 = 1.0 - r1
    weights1 = r1 * (1.0 - r2)
    weights2 = r1 * r2
    points = (
        weights0[:, None] * tri[:, 0]
        + weights1[:, None] * tri[:, 1]
        + weights2[:, None] * tri[:, 2]
    )
    return points.astype(np.float32)


def normalise_points(
    points: np.ndarray,
    *,
    bounds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    bounds = np.asarray(bounds, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {points.shape}")
    if bounds.shape != (2, 3):
        raise ValueError(f"bounds must be [2,3], got {bounds.shape}")
    center = (bounds[0] + bounds[1]) / 2.0
    extents = bounds[1] - bounds[0]
    scale = float(np.max(extents))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"invalid mesh scale {scale}")
    normalised = (points - center.astype(np.float32)) / scale
    return normalised.astype(np.float32), center.astype(np.float32), scale, extents.astype(np.float32)


def compute_shape_descriptors(
    mesh: trimesh.Trimesh,
    raw_points: np.ndarray,
    normalised_points: np.ndarray,
    *,
    scale: float,
    raw_extents: np.ndarray,
) -> np.ndarray:
    """Compute bounded, interpretable geometry descriptors.

    These values supplement PointNet; they are not intended to classify a part
    into a hard-coded primitive category.
    """
    eps = 1e-9
    ext = np.asarray(raw_extents, dtype=np.float64)
    norm_ext = ext / max(scale, eps)
    bbox_volume = float(np.prod(ext))
    area = max(0.0, _safe_float(mesh.area, 0.0))

    try:
        hull = mesh.convex_hull
        convex_volume = abs(_safe_float(hull.volume, 0.0))
    except Exception:
        convex_volume = 0.0
    fill_ratio = convex_volume / max(bbox_volume, eps)

    centred = normalised_points - normalised_points.mean(axis=0, keepdims=True)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    eig_sum = float(eigenvalues.sum())
    eig_ratio = eigenvalues / max(eig_sum, eps)
    l1, l2, l3 = [float(v) for v in eigenvalues]
    linearity = (l1 - l2) / max(l1, eps)
    planarity = (l2 - l3) / max(l1, eps)
    scattering = l3 / max(l1, eps)

    sorted_ext = np.sort(ext)[::-1]
    slenderness = sorted_ext[0] / max(sorted_ext[1], eps)
    flatness = sorted_ext[1] / max(sorted_ext[2], eps)
    # Similarity of the two minor PCA axes: near 1 for circular/square rods,
    # but this is only a soft descriptor, not a primitive label.
    minor_similarity = min(l2, l3) / max(max(l2, l3), eps)

    # Radial statistics around the dominant PCA axis.
    if centred.shape[0] >= 3:
        _, eigvecs = np.linalg.eigh(covariance)
        dominant = eigvecs[:, -1]
        axial = centred @ dominant
        radial_vectors = centred - axial[:, None] * dominant[None, :]
        radial = np.linalg.norm(radial_vectors, axis=1)
    else:
        radial = np.zeros(centred.shape[0], dtype=np.float32)
    radius = np.linalg.norm(normalised_points, axis=1)

    descriptors = np.asarray(
        [
            norm_ext[0],
            norm_ext[1],
            norm_ext[2],
            bbox_volume / max(scale ** 3, eps),
            area / max(scale ** 2, eps),
            convex_volume / max(scale ** 3, eps),
            np.clip(fill_ratio, 0.0, 1.5),
            1.0 if bool(mesh.is_watertight) else 0.0,
            eig_ratio[0],
            eig_ratio[1],
            eig_ratio[2],
            np.clip(linearity, 0.0, 1.0),
            np.clip(planarity, 0.0, 1.0),
            np.clip(scattering, 0.0, 1.0),
            np.clip(slenderness / 20.0, 0.0, 1.0),
            np.clip(flatness / 20.0, 0.0, 1.0),
            np.clip(minor_similarity, 0.0, 1.0),
            float(np.mean(radial)) if radial.size else 0.0,
            float(np.std(radial)) if radial.size else 0.0,
            float(np.mean(radius)) if radius.size else 0.0,
            float(np.std(radius)) if radius.size else 0.0,
            np.clip(math.log1p(mesh.vertices.shape[0]) / 15.0, 0.0, 1.0),
            np.clip(math.log1p(mesh.faces.shape[0]) / 15.0, 0.0, 1.0),
            1.0,
        ],
        dtype=np.float32,
    )
    if descriptors.shape != (len(DESCRIPTOR_NAMES),):
        raise AssertionError("descriptor dimension mismatch")
    return np.nan_to_num(descriptors, nan=0.0, posinf=1.0, neginf=0.0)


def encode_model_geometry(
    *,
    model_id: str,
    path: Path,
    point_count: int = 256,
    seed: int = 0,
) -> ModelGeometry:
    mesh = _load_mesh(path)
    raw_points = sample_surface_points(mesh, point_count=point_count, seed=seed)
    points, center, scale, raw_extents = normalise_points(raw_points, bounds=mesh.bounds)
    descriptors = compute_shape_descriptors(
        mesh,
        raw_points,
        points,
        scale=scale,
        raw_extents=raw_extents,
    )
    metadata = {
        "vertex_count": int(mesh.vertices.shape[0]),
        "face_count": int(mesh.faces.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area": _safe_float(mesh.area, 0.0),
        "raw_bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
    }
    return ModelGeometry(
        model_id=str(model_id),
        source_path=str(path),
        points=points,
        descriptors=descriptors,
        center=center,
        scale=scale,
        raw_extents=raw_extents,
        metadata=metadata,
    )


def build_geometry_dataset(
    *,
    asmdef_path: str,
    point_count: int = 256,
    seed: int = 0,
    search_roots: Sequence[str] = (),
) -> Tuple[List[ModelGeometry], Dict[str, int], Dict[str, Any]]:
    asmdef = load_asmdef(asmdef_path)
    asmdef_path_obj = Path(asmdef_path).expanduser().resolve()
    roots = [Path(p).expanduser().resolve() for p in search_roots]
    raw_models = dict(asmdef.get("models") or {})
    raw_parts = dict(asmdef.get("parts") or {})
    raw_steps = sorted(list(asmdef.get("assembly") or []), key=lambda row: int(row["step"]))

    model_index_by_key: Dict[Tuple[str, str], int] = {}
    geometries: List[ModelGeometry] = []
    part_to_model_index: Dict[str, int] = {}
    resolved_models: Dict[str, str] = {}

    for step in raw_steps:
        part = str(step["part"])
        part_spec = dict(raw_parts[part] or {})
        model_id = str(part_spec["model"])
        model_spec = dict(raw_models[model_id] or {})
        resolved = resolve_model_path(
            str(model_spec["path"]),
            asmdef_dir=asmdef_path_obj.parent,
            search_roots=roots,
        )
        key = (model_id, str(resolved))
        if key not in model_index_by_key:
            index = len(geometries)
            model_index_by_key[key] = index
            geometries.append(
                encode_model_geometry(
                    model_id=model_id,
                    path=resolved,
                    point_count=point_count,
                    seed=seed + index,
                )
            )
        part_to_model_index[part] = model_index_by_key[key]
        resolved_models[model_id] = str(resolved)

    summary = {
        "asmdef": str(asmdef_path_obj),
        "task_name": str(asmdef.get("name", asmdef_path_obj.stem)),
        "point_count": int(point_count),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "unique_model_count": len(geometries),
        "part_count": len(part_to_model_index),
        "part_to_model_index": part_to_model_index,
        "resolved_models": resolved_models,
    }
    return geometries, part_to_model_index, summary


def save_geometry_dataset(
    *,
    output_npz: str,
    geometries: Sequence[ModelGeometry],
    part_to_model_index: Mapping[str, int],
    summary: Mapping[str, Any],
) -> None:
    if not geometries:
        raise ValueError("cannot save an empty geometry dataset")
    points = np.stack([row.points for row in geometries], axis=0).astype(np.float32)
    descriptors = np.stack([row.descriptors for row in geometries], axis=0).astype(np.float32)
    centers = np.stack([row.center for row in geometries], axis=0).astype(np.float32)
    scales = np.asarray([row.scale for row in geometries], dtype=np.float32)
    raw_extents = np.stack([row.raw_extents for row in geometries], axis=0).astype(np.float32)
    model_ids = np.asarray([row.model_id for row in geometries], dtype=np.str_)
    source_paths = np.asarray([row.source_path for row in geometries], dtype=np.str_)
    part_names = np.asarray(list(part_to_model_index.keys()), dtype=np.str_)
    part_model_indices = np.asarray(list(part_to_model_index.values()), dtype=np.int64)

    path = Path(output_npz)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=points,
        descriptors=descriptors,
        centers=centers,
        scales=scales,
        raw_extents=raw_extents,
        model_ids=model_ids,
        source_paths=source_paths,
        part_names=part_names,
        part_model_indices=part_model_indices,
        descriptor_names=np.asarray(DESCRIPTOR_NAMES, dtype=np.str_),
        summary_json=np.asarray(json.dumps(summary, ensure_ascii=False)),
    )


# ---------------------------------------------------------------------------
# PointNet interfaces
# ---------------------------------------------------------------------------


if nn is not None:

    class PointNetGeometryEncoder(nn.Module):
        """Lightweight PointNet: ``[B,N,3] -> [B,embedding_dim]``."""

        def __init__(self, embedding_dim: int = 64) -> None:
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm1d(128)
            self.projection = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.embedding_dim),
            )

        def forward(self, points: "torch.Tensor") -> "torch.Tensor":
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError(f"points must have shape [B,N,3], got {tuple(points.shape)}")
            x = points.transpose(1, 2).contiguous()  # [B,3,N]
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
            x = F.relu(self.bn2(self.conv2(x)), inplace=True)
            x = F.relu(self.bn3(self.conv3(x)), inplace=True)
            x = torch.max(x, dim=2).values
            return self.projection(x)


    class PointNetWithDescriptors(nn.Module):
        """Fuse PointNet geometry and explicit descriptors."""

        def __init__(
            self,
            descriptor_dim: int = len(DESCRIPTOR_NAMES),
            point_embedding_dim: int = 64,
            output_dim: int = 128,
        ) -> None:
            super().__init__()
            self.pointnet = PointNetGeometryEncoder(point_embedding_dim)
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(descriptor_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
            )
            self.fusion = nn.Sequential(
                nn.Linear(point_embedding_dim + 64, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, output_dim),
            )

        def forward(
            self,
            points: "torch.Tensor",
            descriptors: "torch.Tensor",
        ) -> "torch.Tensor":
            if descriptors.ndim != 2:
                raise ValueError(
                    f"descriptors must have shape [B,D], got {tuple(descriptors.shape)}"
                )
            point_embedding = self.pointnet(points)
            descriptor_embedding = self.descriptor_mlp(descriptors)
            return self.fusion(torch.cat([point_embedding, descriptor_embedding], dim=-1))

else:  # pragma: no cover

    class PointNetGeometryEncoder:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("PyTorch is required for PointNetGeometryEncoder")

    class PointNetWithDescriptors:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("PyTorch is required for PointNetWithDescriptors")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample STL point clouds and explicit geometry descriptors from ASMDEF."
    )
    parser.add_argument("--asmdef", required=True)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--pointnet-smoke-test",
        action="store_true",
        help="Run one untrained forward pass when PyTorch is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    geometries, part_to_model_index, summary = build_geometry_dataset(
        asmdef_path=args.asmdef,
        point_count=args.points,
        seed=args.seed,
        search_roots=args.model_root,
    )
    save_geometry_dataset(
        output_npz=args.output_npz,
        geometries=geometries,
        part_to_model_index=part_to_model_index,
        summary=summary,
    )

    payload = dict(summary)
    payload["models"] = [row.to_summary() for row in geometries]
    payload["output_npz"] = str(Path(args.output_npz).resolve())

    if args.pointnet_smoke_test:
        if torch is None:
            payload["pointnet_smoke_test"] = "skipped: PyTorch unavailable"
        else:
            model = PointNetWithDescriptors(output_dim=128)
            model.eval()
            with torch.no_grad():
                points = torch.from_numpy(
                    np.stack([row.points for row in geometries], axis=0)
                ).float()
                descriptors = torch.from_numpy(
                    np.stack([row.descriptors for row in geometries], axis=0)
                ).float()
                embeddings = model(points, descriptors)
            payload["pointnet_smoke_test"] = {
                "input_points": list(points.shape),
                "input_descriptors": list(descriptors.shape),
                "output_embeddings": list(embeddings.shape),
                "note": "random untrained weights; shape/interface test only",
            }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON saved to: {path}")
    print(f"[OK] NPZ saved to: {Path(args.output_npz).resolve()}")


if __name__ == "__main__":
    main()
