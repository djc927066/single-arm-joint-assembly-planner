"""State-explicit contact analysis combining plane regions and mesh witnesses."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from ..geometry.mesh_bvh import QueryBudget, aabb_distance
from ..geometry.proximity import MeshProximity, ProximityBackend
from ..model import (
    Assembly,
    AssemblyState,
    ContactAnalysis,
    ContactConfig,
    Part,
    checked_tf,
    digest,
)
from .mesh import analyze_mesh_pair
from .planar import analyze_planar_pair

if TYPE_CHECKING:
    from .backends import ContactBackend
    from .models import ContactModel


def analyze_pair(
    part_a: Part | ContactModel,
    tf_a: ArrayLike,
    part_b: Part | ContactModel,
    tf_b: ArrayLike,
    *,
    config: ContactConfig | None = None,
    backend: str | ContactBackend | ProximityBackend | None = None,
) -> ContactAnalysis:
    """Analyze real surfaces at explicit poses; region/contact/solid states differ.

    Returns geometric evidence even when global overlap is unknown. Callers
    must inspect pair_diagnostics before treating a patch as an allowable mate.
    """
    from .backends import ContactAnalyzer, ContactBackend
    from .models import ContactModel

    if (
        isinstance(backend, (str, ContactBackend))
        or isinstance(part_a, ContactModel)
        or isinstance(part_b, ContactModel)
    ):
        return ContactAnalyzer(backend, config=config).analyze_pair(
            part_a, part_b, tf_a=tf_a, tf_b=tf_b
        )
    cfg = config or ContactConfig()
    backend = backend or MeshProximity(numerical_tol_m=cfg.numerical_tol_m)
    if part_a.part_id == part_b.part_id:
        raise ValueError("A pair must contain distinct instance IDs")
    ta, tb = checked_tf(tf_a), checked_tf(tf_b)
    state_key = digest(
        (
            "mesh_contact/3",
            part_a.geometry.geometry_id,
            part_b.geometry.geometry_id,
            part_a.part_id,
            part_b.part_id,
            ta,
            tb,
            cfg,
            backend.geometry_config,
            backend.tol,
        )
    )
    pa, sa = backend.prepare(part_a.geometry)
    pb, sb = backend.prepare(part_b.geometry)
    va = pa.mesh.vertices @ ta[:3, :3].T + ta[:3, 3]
    vb = pb.mesh.vertices @ tb[:3, :3].T + tb[:3, 3]
    bound = aabb_distance(va.min(0), va.max(0), vb.min(0), vb.max(0))
    error = pa.mesh.geometry_error_m + pb.mesh.geometry_error_m + cfg.pose_error_m
    base = {
        "part_a": part_a.part_id,
        "part_b": part_b.part_id,
        "geometry_diagnostics_a": pa.diagnostics,
        "geometry_diagnostics_b": pb.diagnostics,
        "orientation_reliable": pa.orientation_reliable and pb.orientation_reliable,
        "proximity_numerical_tol_m": backend.tol,
        "unsupported_contact_features": ("general_curve_point_line", "non_opposing_edge_corner"),
        "analysis_config": digest(cfg),
        "geometry_config": digest(backend.geometry_config),
    }
    if bound > cfg.near_tol_m + error:
        return ContactAnalysis(
            (),
            (
                {
                    **base,
                    "overlap": {"status": "separated", "reason": "aabb_distance_bound"},
                    "distance_lower_bound_m": bound,
                    "unresolved_area_m2": 0.0,
                },
            ),
            state_key,
        )
    # Reserve independent budgets for overlap and near-band extraction so a
    # difficult solid test cannot silently starve contact-region analysis.
    overlap = backend.classify_overlap(part_a, ta, part_b, tb, budget=cfg.max_triangle_tests)
    distance = (
        backend.pair_distance(part_a, ta, part_b, tb, budget=cfg.max_triangle_tests)
        if any(s.kind != "plane" for s in (*sa, *sb))
        else None
    )
    patches, curved, planar, candidates, unresolved = [], [], [], 0, 0.0
    budget = QueryBudget(cfg.max_triangle_tests)
    plane_budget = QueryBudget(cfg.max_triangle_tests)
    for a in sa:
        av = va[pa.mesh.faces[a.face_ids]]
        for b in sb:
            bv = vb[pb.mesh.faces[b.face_ids]]
            if (
                aabb_distance(
                    av.min(axis=(0, 1)),
                    av.max(axis=(0, 1)),
                    bv.min(axis=(0, 1)),
                    bv.max(axis=(0, 1)),
                )
                > cfg.near_tol_m + error
            ):
                continue
            if a.kind == b.kind == "plane":
                found, stats = analyze_planar_pair(
                    part_a, ta, pa, a, part_b, tb, pb, b, config=cfg, budget=plane_budget
                )
                planar.append({"surface_a": a.patch_id, "surface_b": b.patch_id, **stats})
                candidates += stats["candidate_triangle_pairs"]
                unresolved += stats["unresolved_area_m2"]
            else:
                found, reports = analyze_mesh_pair(
                    part_a,
                    ta,
                    pa,
                    a,
                    part_b,
                    tb,
                    pb,
                    b,
                    config=cfg,
                    backend=backend,
                    budget=budget,
                    distance_lower_bound_m=distance.lower_bound_m if distance else 0.0,
                )
                curved.extend(
                    {"surface_a": a.patch_id, "surface_b": b.patch_id, **r} for r in reports
                )
            patches.extend(found)
    from ..model import to_dict

    # Complete a previously unresolved touching pair only for the restricted
    # supporting-face feature case. Never overwrite solid overlap evidence.
    support_exhausted = False
    before_support = plane_budget.used
    if overlap.status == "touching" and not patches:
        from .support import support_face_contacts

        for args in ((part_a, ta, pa, sa, part_b, tb, pb), (part_b, tb, pb, sb, part_a, ta, pa)):
            found, exhausted = support_face_contacts(*args, config=cfg, budget=plane_budget)
            patches.extend(found)
            support_exhausted |= exhausted
        if support_exhausted:
            unresolved += float(pa.areas_m2.sum() + pb.areas_m2.sum())
    diag = {
        **base,
        "overlap": to_dict(overlap),
        "surface_distance": to_dict(distance),
        "curved_coverage": curved,
        "planar_coverage": planar,
        "unresolved_area_m2": unresolved + sum(r["unresolved_area_m2"] for r in curved),
        "contact_mode": "explicit_zero_gap_idealization"
        if cfg.idealize_contact
        else "nominal_mesh_geometry",
        "support_feature_budget_exhausted": support_exhausted,
        "solid_assumption": "non-self-intersecting closed manifold for inside/outside",
    }
    return ContactAnalysis(
        tuple(patches),
        (diag,),
        state_key,
        statistics={
            "candidate_triangle_pairs": candidates,
            "support_triangle_tests": plane_budget.used - before_support,
            "curved_triangle_tests": budget.used,
            "distance_triangle_tests": distance.triangle_tests if distance else 0,
            "overlap_triangle_tests": overlap.triangle_tests,
            "bvh_cache_hits": backend.cache_hits,
        },
    )


def analyze_contacts(
    assembly: Assembly,
    state: AssemblyState,
    *,
    config: ContactConfig | None = None,
    backend: str | ContactBackend | ProximityBackend | None = None,
) -> ContactAnalysis:
    """Analyze present instances, preserving their explicit poses and design mates."""
    from .backends import ContactAnalyzer, ContactBackend
    from .graph import state_geometry_binding

    binding = state_geometry_binding(assembly, state)
    if isinstance(backend, (str, ContactBackend)):
        from dataclasses import replace

        result = ContactAnalyzer(backend, config=config).analyze(assembly.parts, poses=state.poses)
        mates = tuple(
            m
            for m in assembly.mating_relations
            if m.part_a in state.poses and m.part_b in state.poses
        )
        return replace(
            result,
            mating_relations=mates,
            input_binding=binding,
            state_digest=digest((result.state_digest, state.world_revision)),
        )
    cfg = config or ContactConfig()
    backend = backend or MeshProximity(numerical_tol_m=cfg.numerical_tol_m)
    by_id = {p.part_id: p for p in assembly.parts}
    if set(state.poses) - set(by_id):
        raise ValueError("State contains unknown part IDs")
    patches, diagnostics, stats = [], [], []
    for aid, bid in combinations(state.present_part_ids, 2):
        result = analyze_pair(
            by_id[aid], state.poses[aid], by_id[bid], state.poses[bid], config=cfg, backend=backend
        )
        patches.extend(result.patches)
        diagnostics.extend(result.pair_diagnostics)
        stats.append(result.statistics)
    mates = tuple(
        m for m in assembly.mating_relations if m.part_a in state.poses and m.part_b in state.poses
    )
    return ContactAnalysis(
        tuple(patches),
        tuple(diagnostics),
        digest(
            (
                "mesh_contact/3",
                [(p.part_id, p.geometry.geometry_id) for p in assembly.parts],
                state,
                cfg,
                backend.geometry_config,
                backend.tol,
            )
        ),
        mates,
        {"pair_count": len(diagnostics), "pairs": stats},
        input_binding=binding,
    )
