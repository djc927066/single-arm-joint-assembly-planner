"""Backend-neutral contact inputs; geometry is local and all lengths are metres."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from os import PathLike
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Any

import numpy as np
from numpy.typing import ArrayLike

from ..io import read_mesh, unit_scale
from ..model import MeshData, Part, checked_tf, digest

if TYPE_CHECKING:  # 原 WRS3 的 SceneObject; 贴合 WRS1 时由 adapters 提供
    SceneObject = Any


@dataclass(frozen=True, eq=False)
class ContactModel:
    """An immutable surface plus optional native backend representations.

    Parameters
    ----------
    geometry : MeshData
        Local surface used for integration, visualization and mesh checks.
    name : str
        Unique instance ID within a query.
    tf : array-like, shape (4, 4)
        Default local-to-world rigid pose. Query overrides never mutate it.
    representations : mapping, optional
        Native objects keyed by capability, currently ``'sdf'``. Each object
        must expose a stable ``cache_key`` and remain immutable. Native fields
        share this surface's local metre frame. Backends validate compatibility.

    Notes
    -----
    Native representations are runtime objects, not serialized callables.
    A surface mesh is currently required even for SDF input to measure areas.
    """

    geometry: MeshData
    name: str
    tf: np.ndarray = field(default_factory=lambda: np.eye(4))
    representations: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, MeshData):
            raise TypeError("geometry must be MeshData; use an explicit input adapter")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ContactModel needs a nonempty instance name")
        native = dict(self.representations)
        if any(
            not isinstance(k, str) or not isinstance(getattr(v, "cache_key", None), str)
            for k, v in native.items()
        ):
            raise TypeError("Native representations need string keys and stable cache_key strings")
        object.__setattr__(self, "tf", checked_tf(self.tf))
        object.__setattr__(self, "representations", MappingProxyType(native))

    @property
    def geometry_key(self) -> str:
        return digest(
            (self.geometry.geometry_id, {k: v.cache_key for k, v in self.representations.items()})
        )

    def at(self, tf: ArrayLike) -> ContactModel:
        """Return another pose of this instance, sharing immutable geometry."""
        return replace(self, tf=tf)

    def as_part(self) -> Part:
        """Return an M1 part; native representations stay on this contact model."""
        return Part(self.name, self.geometry, self.tf)

    @classmethod
    def from_part(cls, part: Part) -> ContactModel:
        """Adapt a Part, preserving its ID, local geometry and assembled pose."""
        return cls(part.geometry, part.part_id, part.assembled_tf)

    @classmethod
    def from_file(
        cls, path: str | PathLike[str], *, name: str, length_unit: str, **kwargs: Any
    ) -> ContactModel:
        """Read STL with explicit source units; optional tf is already in metres."""
        return cls(read_mesh(path, length_unit=length_unit), name, **kwargs)

    @classmethod
    def from_arrays(
        cls, vertices: ArrayLike, faces: ArrayLike, *, name: str, length_unit: str, **kwargs: Any
    ) -> ContactModel:
        """Copy (N,3) vertices and (F,3) indices; optional tf is in metres."""
        return cls(MeshData(np.asarray(vertices) * unit_scale(length_unit), faces), name, **kwargs)

    @classmethod
    def from_scene_object(
        cls, obj: SceneObject, *, name: str | None = None, geometry_error_m: float = 0.0
    ) -> ContactModel:
        """Snapshot WRS visual meshes and offsets, without reading collider proxies.

        WRS geometry and translations are in metres. Point clouds are rejected.
        Later changes to the scene object do not affect the snapshot.
        """
        vertices, faces, offset = [], [], 0
        for visual in obj.visuals:
            if visual.geom is None or visual.geom.fs is None:
                raise ValueError("Contact analysis needs surface faces, not a point cloud")
            tf = checked_tf(visual.loc_tf)
            vs = np.asarray(visual.geom.vs, dtype=float) @ tf[:3, :3].T + tf[:3, 3]
            vertices.append(vs)
            faces.append(np.asarray(visual.geom.fs, dtype=int) + offset)
            offset += len(vs)
        if not vertices:
            raise ValueError("SceneObject has no visual surface meshes")
        mesh = MeshData(
            np.concatenate(vertices), np.concatenate(faces), geometry_error_m=geometry_error_m
        )
        return cls(mesh, name or obj.name, obj.tf)
