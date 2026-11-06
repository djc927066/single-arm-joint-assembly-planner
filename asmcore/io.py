"""Versioned JSON manifests and raw STL import; no pickle/eval or unit guessing."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .model import Assembly, MatingRelation, MeshData, Part, to_dict

SCHEMA = "wrs.assembly/1"


def unit_scale(length_unit: str | None) -> float:
    """Return metres per explicitly declared input unit."""
    try:
        return {"m": 1.0, "mm": 0.001, "cm": 0.01}[length_unit]
    except (KeyError, TypeError):
        raise ValueError("Declare length_unit as m, mm or cm; STL has no unit") from None


def read_mesh(
    path: str | PathLike[str],
    *,
    length_unit: str,
    orientation: str = "auto",
    geometry_error_m: float = 0.0,
) -> MeshData:
    """Read STL into local float64 metres, preserving raw triangle provenance."""
    from ._stl import read_stl_arrays

    path = Path(path)
    if path.suffix.lower() != ".stl":
        raise ValueError("M1 file input supports STL; use inline arrays for other sources")
    vs, fs = read_stl_arrays(path, dtype=np.float64)
    return MeshData(vs * unit_scale(length_unit), fs, orientation, geometry_error_m)


def load_assembly(
    manifest_path: str | PathLike[str], *, length_unit: str | None = None
) -> Assembly:
    """Read a wrs.assembly/1 JSON manifest relative to its own directory.

    Pose translations, inline vertices and COM share ``length_unit``. Explicit
    fields suffixed _m, _kg or _m_s2 already use SI units. Mass stays unknown
    when absent. A conflicting unit override raises instead of rescaling twice.
    """
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA:
        raise ValueError(f"Expected schema_version={SCHEMA}")
    declared = data.get("length_unit")
    if declared is not None and length_unit is not None and declared != length_unit:
        raise ValueError("Unit override conflicts with manifest")
    unit = declared or length_unit
    scale = unit_scale(unit)
    parts = []
    geometry_cache = {}
    for item in data["parts"]:
        spec = item["geometry"]
        orientation = spec.get("orientation", "auto")
        error = spec.get("geometry_error_m", 0.0)
        if "path" in spec:
            mesh_path = (path.parent / spec["path"]).resolve()
            key = (str(mesh_path), orientation, error)
            if key not in geometry_cache:
                geometry_cache[key] = read_mesh(
                    mesh_path, length_unit=unit, orientation=orientation, geometry_error_m=error
                )
            mesh = geometry_cache[key]
        else:
            mesh = MeshData(np.asarray(spec["vertices"]) * scale, spec["faces"], orientation, error)
        tf = np.array(item.get("assembled_tf", np.eye(4)), dtype=float)
        if tf.shape != (4, 4):
            raise ValueError("assembled_tf must be (4,4)")
        tf[:3, 3] *= scale
        com = item.get("com_local")
        parts.append(
            Part(
                item["part_id"],
                mesh,
                tf,
                item.get("mass_kg"),
                None if com is None else np.asarray(com) * scale,
                item.get("friction"),
                item.get("fixed", False),
            )
        )
    mating = tuple(MatingRelation(**m) for m in data.get("mating_relations", []))
    return Assembly(
        tuple(parts),
        data.get("gravity_world_m_s2", [0, 0, -9.81]),
        mating,
        {**data.get("provenance", {}), "manifest": str(path.resolve()), "input_length_unit": unit},
    )


def save_assembly(assembly: Assembly, path: str | PathLike[str]) -> None:
    """Write a self-contained metre manifest with inline analysis geometry."""
    parts = []
    for p in assembly.parts:
        parts.append(
            {
                "part_id": p.part_id,
                "assembled_tf": p.assembled_tf,
                "geometry": {
                    "vertices": p.geometry.vertices,
                    "faces": p.geometry.faces,
                    "orientation": p.geometry.orientation,
                    "geometry_error_m": p.geometry.geometry_error_m,
                },
                "mass_kg": p.mass_kg,
                "com_local": p.com_local_m,
                "friction": p.friction,
                "fixed": p.fixed,
            }
        )
    save_report(
        {
            "schema_version": SCHEMA,
            "length_unit": "m",
            "parts": parts,
            "gravity_world_m_s2": assembly.gravity_world_m_s2,
            "mating_relations": assembly.mating_relations,
            "provenance": assembly.provenance,
        },
        path,
    )


def save_report(report: Any, path: str | PathLike[str]) -> None:
    """Serialize evidence as UTF-8 JSON. NaN/infinity are rejected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_dict(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
