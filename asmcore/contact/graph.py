"""State-bound instance contact graphs; near bands never become active edges."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..model import (
    Assembly,
    AssemblyState,
    ContactAnalysis,
    ContactPatch,
    MatingRelation,
    digest,
    freeze,
)


def state_geometry_binding(assembly: Assembly, state: AssemblyState) -> dict[str, Any]:
    """Identify present instance geometry and explicit world poses/revision.

    Parameters
    ----------
    assembly : Assembly
        Source geometry; absent parts are excluded from the binding.
    state : AssemblyState
        Local-to-world transforms in metres. Unknown IDs raise ValueError.
    """
    parts = {p.part_id: p for p in assembly.parts}
    if set(state.poses) - set(parts):
        raise ValueError("State contains unknown part IDs")
    return {
        "schema": "assembly_geometry_state/1",
        "world_revision": state.world_revision,
        "instances": {
            k: {"geometry_id": parts[k].geometry.geometry_id, "tf": state.poses[k]}
            for k in state.present_part_ids
        },
    }


@dataclass(frozen=True)
class ContactEdge:
    """All evidence for an unordered instance pair, retaining patch A/B order."""

    part_ids: tuple[str, str]
    patches: tuple[ContactPatch, ...]
    mating_relations: tuple[MatingRelation, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    issues: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class ContactGraph:
    """Immutable graph at one explicit state; no viewer/global pose lookup."""

    nodes: Mapping
    edges: tuple[ContactEdge, ...]
    input_binding: Mapping
    state_digest: str
    analysis_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", freeze(self.nodes))
        object.__setattr__(self, "input_binding", freeze(self.input_binding))

    def assert_matches(self, assembly: Assembly, state: AssemblyState) -> None:
        """Reject use with changed geometry, poses, revision or fixed supports."""
        if digest(self.input_binding) != digest(state_geometry_binding(assembly, state)):
            raise ValueError("Contact graph is stale: geometry, pose or world revision changed")
        fixed = {p.part_id: p.fixed for p in assembly.parts if p.part_id in state.poses}
        if any(fixed[k] != self.nodes[k]["fixed"] for k in fixed):
            raise ValueError("Contact graph is stale: fixed supports changed")
        mates = tuple(
            m for m in assembly.mating_relations if m.part_a in fixed and m.part_b in fixed
        )
        previous = tuple(m for e in self.edges for m in e.mating_relations)
        if sorted(digest(m) for m in mates) != sorted(digest(m) for m in previous):
            raise ValueError("Contact graph is stale: mating relations changed")


def build_contact_graph(
    assembly: Assembly, state: AssemblyState, analysis: ContactAnalysis
) -> ContactGraph:
    """Build an instance graph from a state-bound ``analyze_contacts`` report.

    Every pair retains diagnostics, even if separated or unresolved. Unbound
    legacy/pair reports must be recomputed with analyze_contacts; assigning a
    new pose to old witnesses is never an implicit conversion. Geometry and
    world-state bindings are checked before any constraints can be generated.
    """
    binding = state_geometry_binding(assembly, state)
    if digest(analysis.input_binding) != digest(binding):
        raise ValueError(
            "Analysis is stale or unbound; recompute analyze_contacts(assembly, state)"
        )
    parts = {p.part_id: p for p in assembly.parts}
    nodes = {k: {**v, "fixed": parts[k].fixed} for k, v in binding["instances"].items()}
    patches, diagnostics, mates = defaultdict(list), defaultdict(list), defaultdict(list)

    def pair(a, b):
        if a == b or a not in nodes or b not in nodes:
            raise ValueError("Contact evidence refers to absent or invalid instance IDs")
        return tuple(sorted((a, b)))

    for p in analysis.patches:
        patches[pair(p.part_a, p.part_b)].append(p)
    for d in analysis.pair_diagnostics:
        diagnostics[pair(d["part_a"], d["part_b"])].append(d)
    for m in assembly.mating_relations:
        if m.part_a in nodes and m.part_b in nodes:
            mates[pair(m.part_a, m.part_b)].append(m)
    edges = []
    for ids in combinations(sorted(nodes), 2):
        issues = set()
        if not diagnostics[ids]:
            issues.add("missing_pair_diagnostics")
        for d in diagnostics[ids]:
            status = d.get("overlap", {}).get("status", "unknown")
            if status in ("penetrating", "unknown"):
                issues.add("overlap_" + status)
            if d.get("unresolved_area_m2", 0) > 0:
                issues.add("unresolved_contact_area")
            if d.get("active_area", {}).get("status") == "not_solved" or (
                status == "touching" and not any(p.classification == "active" for p in patches[ids])
            ):
                issues.add("active_contact_not_solved")
        for p in patches[ids]:
            if p.classification in ("unknown", "interference"):
                issues.add("patch_" + p.classification)
            if p.quality in ("unresolved", "estimated"):
                issues.add("patch_" + p.quality)
        edges.append(
            ContactEdge(
                ids,
                tuple(patches[ids]),
                tuple(mates[ids]),
                tuple(diagnostics[ids]),
                tuple(sorted(issues)),
            )
        )
    key = digest(
        (
            "contact_graph/1",
            binding,
            analysis.state_digest,
            {k: n["fixed"] for k, n in nodes.items()},
            tuple(m for e in edges for m in e.mating_relations),
        )
    )
    return ContactGraph(nodes, tuple(edges), binding, key, analysis.state_digest)
