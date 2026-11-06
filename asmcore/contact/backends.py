"""Public contact backend protocol, explicit selection and reusable analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from itertools import combinations
from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from ..geometry.proximity import MeshProximity, ProximityBackend
from ..model import (
    ContactAnalysis,
    ContactConfig,
    GeometryConfig,
    Part,
    checked_tf,
    digest,
)
from .models import ContactModel


@runtime_checkable
class ContactBackend(Protocol):
    """High-level extension point; custom backends return the shared report schema.

    ``cache_key`` must include implementation version and all backend options.
    Poses passed to ``analyze_pair`` are local-to-world (4,4), in metres.
    Unsupported data must raise explicitly, never silently select another backend.
    """

    name: str

    @property
    def cache_key(self) -> str: ...

    def analyze_pair(
        self,
        a: ContactModel,
        tf_a: ArrayLike,
        b: ContactModel,
        tf_b: ArrayLike,
        *,
        config: ContactConfig,
    ) -> ContactAnalysis: ...


class MeshContactBackend:
    """Adapter around the existing plane-clipping/adaptive-mesh implementation."""

    name = "mesh"

    def __init__(
        self,
        *,
        proximity: ProximityBackend | None = None,
        geometry_config: GeometryConfig | None = None,
    ) -> None:
        self.proximity = proximity or MeshProximity(geometry_config=geometry_config)

    @property
    def cache_key(self) -> str:
        return digest(("mesh_contact/2", self.proximity.geometry_config, self.proximity.tol))

    def analyze_pair(
        self,
        a: ContactModel,
        tf_a: ArrayLike,
        b: ContactModel,
        tf_b: ArrayLike,
        *,
        config: ContactConfig,
    ) -> ContactAnalysis:
        """Analyze the supplied integration meshes, retaining native-field provenance."""
        from .analysis import analyze_pair

        return analyze_pair(
            a.as_part(), tf_a, b.as_part(), tf_b, config=config, backend=self.proximity
        )


def resolve_backend(backend: str | ContactBackend | MeshProximity | None) -> ContactBackend:
    """Resolve built-in names or a ContactBackend object; imports stay lazy."""
    if backend is None or isinstance(backend, str):
        if backend is None or backend == "mesh":
            return MeshContactBackend()
        if backend == "sdf":
            from .sdf_backend import SDFContactBackend

            return SDFContactBackend()
        raise ValueError(f"Unknown contact backend {backend!r}; built-ins are mesh and sdf")
    if isinstance(backend, MeshProximity):
        return MeshContactBackend(proximity=backend)
    if not isinstance(backend, ContactBackend):
        raise TypeError("Expected a backend name or ContactBackend implementation")
    return backend


class ContactAnalyzer:
    """Reusable headless entry point; compiled local geometry is cached by backend.

    Parameters
    ----------
    backend : {'mesh', 'sdf'} or ContactBackend
        Explicit algorithm choice. Instances are reusable, not thread-safe.
    config : ContactConfig, optional
        Shared SI tolerances and integration limits.
    """

    def __init__(
        self,
        backend: str | ContactBackend | MeshProximity | None = "mesh",
        *,
        config: ContactConfig | None = None,
    ) -> None:
        self.backend = resolve_backend(backend)
        self.config = config or ContactConfig()

    def analyze_pair(
        self,
        a: ContactModel | Part,
        b: ContactModel | Part,
        *,
        tf_a: ArrayLike | None = None,
        tf_b: ArrayLike | None = None,
    ) -> ContactAnalysis:
        """Analyze ContactModel/Part instances at default or explicit world poses."""
        a = ContactModel.from_part(a) if isinstance(a, Part) else a
        b = ContactModel.from_part(b) if isinstance(b, Part) else b
        if not isinstance(a, ContactModel) or not isinstance(b, ContactModel):
            raise TypeError("Use ContactModel input adapters or Part instances")
        if a.name == b.name:
            raise ValueError("Contact pair requires distinct instance names")
        ta, tb = (
            checked_tf(a.tf if tf_a is None else tf_a),
            checked_tf(b.tf if tf_b is None else tf_b),
        )
        result = self.backend.analyze_pair(a, ta, b, tb, config=self.config)
        key = digest(
            (
                a.name,
                a.geometry_key,
                ta,
                b.name,
                b.geometry_key,
                tb,
                self.config,
                self.backend.cache_key,
            )
        )
        diagnostics = tuple(
            {**d, "contact_backend": self.backend.name, "backend_key": self.backend.cache_key}
            for d in result.pair_diagnostics
        )
        return replace(result, state_digest=key, pair_diagnostics=diagnostics)

    def analyze(
        self, models: Iterable[ContactModel | Part], *, poses: Mapping[str, ArrayLike] | None = None
    ) -> ContactAnalysis:
        """Analyze all pairs, or only IDs present in an explicit pose mapping.

        Missing IDs in ``poses`` mean absent instances, matching AssemblyState.
        """
        models = tuple(ContactModel.from_part(m) if isinstance(m, Part) else m for m in models)
        by_id = {m.name: m for m in models}
        if len(by_id) != len(models) or not models:
            raise ValueError("Need nonempty models with unique instance names")
        poses = {m.name: m.tf for m in models} if poses is None else dict(poses)
        if set(poses) - set(by_id):
            raise ValueError("Pose mapping contains unknown instance IDs")
        poses = {k: checked_tf(v) for k, v in poses.items()}
        results = [
            self.analyze_pair(by_id[a], by_id[b], tf_a=poses[a], tf_b=poses[b])
            for a, b in combinations(sorted(poses), 2)
        ]
        return ContactAnalysis(
            tuple(p for r in results for p in r.patches),
            tuple(d for r in results for d in r.pair_diagnostics),
            digest(
                (
                    [(m.name, m.geometry_key) for m in models],
                    poses,
                    self.config,
                    self.backend.cache_key,
                )
            ),
            statistics={"pair_count": len(results), "pairs": [r.statistics for r in results]},
        )
