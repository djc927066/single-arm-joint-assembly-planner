"""Static equilibrium with paired forces and an inscribed friction pyramid.

Feasibility concerns the declared rigid contact/force model. It does not prove
dynamic stability, robot grasp feasibility or resistance to untested loads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from ..contact.graph import ContactGraph
from .force_points import prepare_force_points
from ..model import Assembly, AssemblyState, FloatArray, Part, digest, freeze, readonly


@dataclass(frozen=True, eq=False)
class ExternalWrench:
    """World force (N) and world torque about this body's COM (N m)."""

    part_id: str
    force_world_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_world_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if not isinstance(self.part_id, str) or not self.part_id:
            raise ValueError("Wrench part_id must be a nonempty string")
        object.__setattr__(self, "force_world_n", readonly(self.force_world_n, shape=(3,)))
        object.__setattr__(self, "torque_world_nm", readonly(self.torque_world_nm, shape=(3,)))


@dataclass(frozen=True)
class LoadCase:
    """An additional simultaneous set of loads, added to gravity/base loads."""

    name: str
    wrenches: tuple[ExternalWrench, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name == "nominal":
            raise ValueError("Load case needs a distinct non-nominal name")
        object.__setattr__(self, "wrenches", tuple(self.wrenches))
        if not all(isinstance(w, ExternalWrench) for w in self.wrenches):
            raise ValueError("Load case wrenches must be ExternalWrench instances")


@dataclass(frozen=True, eq=False)
class SupportCandidate:
    """A finite unilateral force at a declared point, not a validated grasp.

    max_normal_force_n caps the sum of generator weights. Tangential force
    is further bounded by friction. No free moment is supplied at the point.
    """

    support_id: str
    part_id: str
    point_world_m: np.ndarray
    normal_world: np.ndarray
    max_normal_force_n: float
    friction: float = 0.0

    def __post_init__(self) -> None:
        if not self.support_id:
            raise ValueError("Support ID must be nonempty")
        object.__setattr__(self, "point_world_m", readonly(self.point_world_m, shape=(3,)))
        n = readonly(self.normal_world, shape=(3,))
        size = np.linalg.norm(n)
        if not np.isfinite(size) or size == 0:
            raise ValueError("Support normal must be nonzero")
        object.__setattr__(self, "normal_world", readonly(n / size))
        if (
            not np.isfinite(self.max_normal_force_n)
            or self.max_normal_force_n <= 0
            or not np.isfinite(self.friction)
            or self.friction < 0
        ):
            raise ValueError("Support needs a finite positive capacity and nonnegative friction")


@dataclass(frozen=True)
class StabilityConfig:
    friction_sides: int = 16
    characteristic_length_m: float = 0.1
    force_tol_n: float = 1e-6
    torque_tol_nm: float = 1e-7
    max_contact_normal_force_n: float | None = None
    allow_idealized_contact: bool = False
    solver_time_limit_s: float = 5.0
    disturbances: tuple[LoadCase, ...] = ()
    reduce_contact_points: bool = True
    curved_point_budget: int = 64
    curved_point_spacing_m: float = 0.005
    curved_normal_angle_rad: float = np.pi / 18

    def __post_init__(self) -> None:
        if not isinstance(self.reduce_contact_points, bool):
            raise ValueError("reduce_contact_points must be a bool")
        if (
            not isinstance(self.curved_point_budget, int)
            or isinstance(self.curved_point_budget, bool)
            or self.curved_point_budget < 4
        ):
            raise ValueError("curved_point_budget must be an integer >= 4")
        if not np.isfinite(self.curved_point_spacing_m) or self.curved_point_spacing_m <= 0:
            raise ValueError("curved_point_spacing_m must be positive and finite")
        if (
            not np.isfinite(self.curved_normal_angle_rad)
            or not 0 < self.curved_normal_angle_rad < np.pi
        ):
            raise ValueError("curved_normal_angle_rad must be in (0, pi)")
        if (
            not isinstance(self.friction_sides, int)
            or isinstance(self.friction_sides, bool)
            or self.friction_sides < 4
            or self.friction_sides % 2
        ):
            raise ValueError("friction_sides must be an even integer >= 4")
        for name in (
            "characteristic_length_m",
            "force_tol_n",
            "torque_tol_nm",
            "solver_time_limit_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_contact_normal_force_n is not None and (
            not np.isfinite(self.max_contact_normal_force_n) or self.max_contact_normal_force_n <= 0
        ):
            raise ValueError("Contact capacity must be positive and finite, or None")
        cases = tuple(self.disturbances)
        if not all(isinstance(c, LoadCase) for c in cases) or len({c.name for c in cases}) != len(
            cases
        ):
            raise ValueError("Disturbances must be LoadCases with unique names")
        object.__setattr__(self, "disturbances", cases)


@dataclass(frozen=True)
class EquilibriumCase:
    name: str
    status: str
    contact_forces: tuple = ()
    support_forces: tuple = ()
    body_residuals: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("contact_forces", "support_forces", "body_residuals", "solver"):
            object.__setattr__(self, key, freeze(getattr(self, key)))


@dataclass(frozen=True)
class EquilibriumResult:
    """Per-load feasibility plus the immutable force model used by the solver.

    force_sites includes zero-force/infeasible sites. Its normal and rays act
    on part_b; part_a receives their negatives. capacity_group identifies a
    shared patch/support normal-force limit, not an independent per-ray cap.
    """

    status: str
    nominal: EquilibriumCase | None
    disturbances: tuple[EquilibriumCase, ...]
    robustness_status: str
    issues: tuple
    assumptions: tuple
    input_digest: str
    diagnostics: dict = field(default_factory=dict)
    execution_validated: bool = False
    force_sites: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "force_sites", freeze(self.force_sites))
        for key in ("disturbances", "issues", "assumptions"):
            object.__setattr__(self, key, tuple(getattr(self, key)))


def _generators(normal: FloatArray, friction: float, sides: int) -> FloatArray:
    n = normal / np.linalg.norm(normal)
    if friction == 0:
        return n[None, :]
    axis = np.eye(3)[np.argmin(np.abs(n))]
    tangent = np.cross(n, axis)
    tangent /= np.linalg.norm(tangent)
    other = np.cross(n, tangent)
    angle = 2 * np.pi * np.arange(sides) / sides
    return n + friction * (np.cos(angle)[:, None] * tangent + np.sin(angle)[:, None] * other)


def check_equilibrium(
    assembly: Assembly,
    state: AssemblyState,
    graph: ContactGraph,
    *,
    config: StabilityConfig | None = None,
    external_wrenches: Iterable[ExternalWrench] = (),
    supports: Iterable[SupportCandidate] = (),
) -> EquilibriumResult:
    """Check a reduced force model, retrying all curved sites on any failed load.

    Planar hulls preserve the patch wrench model to arithmetic roundoff. Curved
    samples are only a subset: their failure is never a final infeasibility claim.
    Returned sites/forces always belong to the final common model for all loads.
    """
    started = perf_counter()
    cfg = config or StabilityConfig()
    external_wrenches, supports = tuple(external_wrenches), tuple(supports)
    result = _check_equilibrium(
        assembly, state, graph, config=cfg, external_wrenches=external_wrenches, supports=supports
    )
    failed = [
        c.name
        for c in (result.nominal, *result.disturbances)
        if c is not None and c.status != "feasible"
    ]
    if result.diagnostics.get("curved_sampling_used", False) and failed:
        attempt = dict(
            status=result.status,
            force_points=result.diagnostics["force_points"],
            failed_loads=failed,
        )
        result = _check_equilibrium(
            assembly,
            state,
            graph,
            config=cfg,
            external_wrenches=external_wrenches,
            supports=supports,
            _full_curves=True,
        )
        return replace(
            result,
            diagnostics=dict(
                result.diagnostics,
                curved_fallback=True,
                reduced_attempt=attempt,
                elapsed_s=perf_counter() - started,
            ),
        )
    return result


@dataclass
class _ForceSites:
    """Shared site records and one capacity per patch/support, never per ray."""

    records: list[dict[str, Any]]
    capacity_limits: list[float | None]
    diagnostics: list[dict[str, Any]]


@dataclass
class _ForceModel:
    """Sparse equilibrium matrices; all load cases reuse this same force model."""

    sites: _ForceSites
    body_index: dict[str, int]
    physical: sparse.csc_matrix
    row_scale: np.ndarray
    balance: sparse.spmatrix
    capacity: sparse.csc_matrix | None
    capacity_limits: np.ndarray | None
    variable_count: int


def _collect_force_sites(
    graph: ContactGraph,
    parts: Mapping[str, Part],
    body_index: Mapping[str, int],
    supports: Sequence[SupportCandidate],
    *,
    cfg: StabilityConfig,
    issues: set[str],
    assumptions: list[str],
    _full_curves: bool,
) -> _ForceSites:
    """Qualify active patches, record assumptions, and prepare actual force sites."""
    records, groups, point_diagnostics = [], [], []
    for edge in graph.edges:
        if not any(k in body_index for k in edge.part_ids):
            continue
        issues.update(f"{edge.part_ids}:{v}" for v in edge.issues)
        for patch_index, patch in enumerate(edge.patches):
            if patch.classification != "active":
                continue
            tag = f"{edge.part_ids}:patch{patch_index}"
            if patch.quality not in ("analytic", "bounded"):
                issues.add(tag + ":unqualified_active_contact")
                continue
            if patch.provenance.get("idealize_contact", False):
                if not cfg.allow_idealized_contact:
                    issues.add(tag + ":idealized_contact_disabled")
                    continue
                assumptions.append(tag + ":explicit_zero_gap_idealization")
            if not len(patch.normals_a_world) or np.any(
                np.einsum("ij,ij->i", patch.normals_a_world, patch.normals_b_world) > -1 + 1e-7
            ):
                issues.add(tag + ":invalid_contact_normals")
                continue
            mus = [parts[k].friction for k in (patch.part_a, patch.part_b)]
            if any(mu is None for mu in mus):
                issues.add(tag + ":unknown_friction")
                continue
            prepared, cache_hit = prepare_force_points(
                patch,
                reduce=cfg.reduce_contact_points,
                curved_budget=cfg.curved_point_budget,
                spacing=cfg.curved_point_spacing_m,
                normal_angle=cfg.curved_normal_angle_rad,
                full_curves=_full_curves,
            )
            points, normals = prepared.points, prepared.normals
            point_diagnostics.append(dict(prepared.info, patch=tag, cache_hit=cache_hit))
            if not len(points):
                issues.add(tag + ":missing_force_points")
                continue
            group = len(groups)
            groups.append(cfg.max_contact_normal_force_n)
            for point, normal in zip(points, normals):
                records.append(
                    {
                        "kind": "contact",
                        "part_a": patch.part_a,
                        "part_b": patch.part_b,
                        "patch": tag,
                        "point_world_m": point,
                        "normal_world": normal,
                        "friction": min(mus),
                        "group": group,
                    }
                )
    for support in supports:
        group = len(groups)
        groups.append(support.max_normal_force_n)
        records.append(
            {
                "kind": "support",
                "part_a": None,
                "part_b": support.part_id,
                "support_id": support.support_id,
                "point_world_m": support.point_world_m,
                "normal_world": support.normal_world,
                "friction": support.friction,
                "group": group,
            }
        )
    return _ForceSites(records, groups, point_diagnostics)


def _assemble_force_model(
    sites: _ForceSites, free: Sequence[str], coms: Mapping[str, FloatArray], cfg: StabilityConfig
) -> _ForceModel:
    """Build six rows per free body, with one nonnegative weight per friction ray.

    A contact shares the SAME columns between both bodies, with opposite signs.
    Force rows use N; moment rows use N m before scaling by 1/L for the LP.
    Capacity rows constrain the sum of all ray weights in a patch/support.
    """
    records = sites.records
    groups = sites.capacity_limits
    body_index = {part_id: i for i, part_id in enumerate(free)}
    rows, cols, values, group_columns = [], [], [], [[] for _ in groups]
    count = 0
    for rec in records:
        generators = _generators(rec["normal_world"], rec["friction"], cfg.friction_sides)
        indices = np.arange(count, count + len(generators))
        rec["generators"], rec["columns"] = generators, indices
        group_columns[rec["group"]].extend(indices)
        count += len(generators)
        for part_id, sign in ((rec["part_a"], -1), (rec["part_b"], 1)):
            if part_id not in body_index:
                continue
            block = sign * np.vstack(
                (generators.T, np.cross(rec["point_world_m"] - coms[part_id], generators).T)
            )
            rows.append(
                np.broadcast_to(
                    (6 * body_index[part_id] + np.arange(6))[:, None], block.shape
                ).ravel()
            )
            cols.append(np.broadcast_to(indices, block.shape).ravel())
            values.append(block.ravel())
    physical = (
        sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(6 * len(free), count),
        ).tocsc()
        if values
        else sparse.csc_matrix((6 * len(free), count))
    )
    scale = np.tile(np.r_[np.ones(3), np.full(3, 1 / cfg.characteristic_length_m)], len(free))
    Aeq = sparse.diags(scale) @ physical
    limited = [
        (columns, limit) for columns, limit in zip(group_columns, groups) if limit is not None
    ]
    if limited:
        cap_rows = np.concatenate([np.full(len(c), i) for i, (c, _) in enumerate(limited)])
        cap_cols = np.concatenate([np.asarray(c) for c, _ in limited])
        caps = sparse.coo_matrix(
            (np.ones(len(cap_cols)), (cap_rows, cap_cols)), shape=(len(limited), count)
        ).tocsc()
        capacities = np.array([limit for _, limit in limited])
    else:
        caps, capacities = None, None

    return _ForceModel(
        sites=sites,
        body_index=body_index,
        physical=physical,
        row_scale=scale,
        balance=Aeq,
        capacity=caps,
        capacity_limits=capacities,
        variable_count=count,
    )


def _solve_load_case(
    name: str,
    loads: Sequence[ExternalWrench],
    *,
    assembly: Assembly,
    parts: Mapping[str, Part],
    free: Sequence[str],
    model: _ForceModel,
    cfg: StabilityConfig,
    external_wrenches: Sequence[ExternalWrench],
) -> EquilibriumCase:
    """Solve one RHS, then verify force/torque residuals in their physical units."""
    external = np.zeros((len(free), 6))
    for k in free:
        external[model.body_index[k], :3] = parts[k].mass_kg * assembly.gravity_world_m_s2
    for w in (*external_wrenches, *loads):
        external[model.body_index[w.part_id], :3] += w.force_world_n
        external[model.body_index[w.part_id], 3:] += w.torque_world_nm
    if not np.all(np.isfinite(external)):
        return EquilibriumCase(
            name, "unknown", solver={"message": "Nonfinite scaled physical loads"}
        )
    if model.variable_count:
        tolerance = max(
            1e-10, min(cfg.force_tol_n, cfg.torque_tol_nm / cfg.characteristic_length_m) * 0.1
        )
        result = linprog(
            np.ones(model.variable_count),
            A_ub=model.capacity,
            b_ub=model.capacity_limits,
            A_eq=model.balance,
            b_eq=-external.ravel() * model.row_scale,
            bounds=(0, None),
            method="highs",
            options={
                "time_limit": cfg.solver_time_limit_s,
                "primal_feasibility_tolerance": tolerance,
                "dual_feasibility_tolerance": tolerance,
            },
        )
        solver = {"status": int(result.status), "message": str(result.message)}
        if not result.success:
            return EquilibriumCase(
                name, "infeasible" if result.status == 2 else "unknown", solver=solver
            )
        if not np.all(np.isfinite(result.x)):
            return EquilibriumCase(name, "unknown", solver=dict(solver, error="nonfinite_solution"))
        weights = np.maximum(result.x, 0.0)
    else:
        weights = np.empty(0)
        solver = {
            "status": 0,
            "message": "No contact-force variables; check applied loads directly",
        }
    residual = np.asarray(model.physical @ weights).reshape(-1, 6) + external
    force_error = float(np.max(np.abs(residual[:, :3]), initial=0))
    torque_error = float(np.max(np.abs(residual[:, 3:]), initial=0))
    residuals = {
        k: {"force_world_n": residual[i, :3], "torque_world_nm": residual[i, 3:]}
        for i, k in enumerate(free)
    }
    if (
        force_error > cfg.force_tol_n
        or torque_error > cfg.torque_tol_nm
        or (
            model.capacity is not None
            and np.any(model.capacity @ weights > model.capacity_limits + cfg.force_tol_n)
        )
    ):
        return EquilibriumCase(
            name,
            "unknown" if model.variable_count else "infeasible",
            body_residuals=residuals,
            solver=dict(solver, residual_check_failed=True),
        )
    contacts, support_forces = [], []
    for site_index, rec in enumerate(model.sites.records):
        force = weights[rec["columns"]] @ rec["generators"]
        entry = {
            "site_index": site_index,
            "point_world_m": rec["point_world_m"],
            "normal_force_n": float(np.sum(weights[rec["columns"]])),
            "force_on_b_world_n": force,
            "part_b": rec["part_b"],
        }
        if rec["kind"] == "contact":
            contacts.append(
                dict(entry, part_a=rec["part_a"], force_on_a_world_n=-force, patch=rec["patch"])
            )
        else:
            support_forces.append(dict(entry, support_id=rec["support_id"]))
    return EquilibriumCase(
        name, "feasible", tuple(contacts), tuple(support_forces), residuals, solver
    )


def _check_equilibrium(
    assembly: Assembly,
    state: AssemblyState,
    graph: ContactGraph,
    *,
    config: StabilityConfig | None,
    external_wrenches: Iterable[ExternalWrench],
    supports: Iterable[SupportCandidate],
    _full_curves: bool = False,
) -> EquilibriumResult:
    """Balance every free body under gravity and declared loads.

    Only qualified active contacts supply forces. Fixed supports are ideal
    boundary conditions. An infeasible inner friction approximation is not a
    proof of infeasibility of the exact Coulomb cone or all possible grasps.
    """
    start = perf_counter()
    cfg = config or StabilityConfig()
    graph.assert_matches(assembly, state)
    parts = {p.part_id: p for p in assembly.parts if p.part_id in state.poses}
    free = tuple(sorted(k for k, p in parts.items() if not p.fixed))
    body_index = {k: i for i, k in enumerate(free)}
    supports, external_wrenches = tuple(supports), tuple(external_wrenches)
    if len({s.support_id for s in supports}) != len(supports):
        raise ValueError("Support IDs must be unique")
    for s in supports:
        if s.part_id not in body_index:
            raise ValueError("Supports must act on present free parts")
    for w in (*external_wrenches, *(w for c in cfg.disturbances for w in c.wrenches)):
        if not isinstance(w, ExternalWrench) or w.part_id not in body_index:
            raise ValueError("Wrenches must act on present free parts")
    key = digest(
        (
            "equilibrium/1",
            graph,
            state,
            tuple(
                (k, p.mass_kg, p.com_local_m, p.friction, p.fixed) for k, p in sorted(parts.items())
            ),
            assembly.gravity_world_m_s2,
            cfg,
            external_wrenches,
            supports,
        )
    )
    assumptions = [
        "rigid_nominal_contacts",
        "fixed_supports_have_unlimited_reaction",
        f"inscribed_friction_polygon_{cfg.friction_sides}_sides",
        "pair_friction_is_minimum_of_declared_material_values",
        "force_points_cleaned_per_patch; optional_planar_hull_and_curved_subset",
        "infeasible_means_this_discrete_force_model_only",
    ]
    if cfg.max_contact_normal_force_n is None:
        assumptions.append("contact_normal_force_capacity_unbounded")
    if supports:
        assumptions.append("auxiliary_support_locations_and_robot_execution_unverified")
    issues, coms = set(), {}
    for k in free:
        part = parts[k]
        if part.mass_kg is None:
            issues.add(k + ":unknown_mass")
        if part.com_local_m is None:
            issues.add(k + ":unknown_center_of_mass")
        else:
            tf = state.poses[k]
            coms[k] = tf[:3, :3] @ part.com_local_m + tf[:3, 3]
    if issues:
        return EquilibriumResult(
            "unknown",
            None,
            (),
            "unknown",
            tuple(sorted(issues)),
            tuple(assumptions),
            key,
            {"elapsed_s": perf_counter() - start},
        )

    sites = _collect_force_sites(
        graph,
        parts,
        body_index,
        supports,
        cfg=cfg,
        issues=issues,
        assumptions=assumptions,
        _full_curves=_full_curves,
    )
    model = _assemble_force_model(sites, free, coms, cfg)
    nominal = _solve_load_case(
        "nominal",
        (),
        assembly=assembly,
        parts=parts,
        free=free,
        model=model,
        cfg=cfg,
        external_wrenches=external_wrenches,
    )
    disturbances = tuple(
        _solve_load_case(
            case.name,
            case.wrenches,
            assembly=assembly,
            parts=parts,
            free=free,
            model=model,
            cfg=cfg,
            external_wrenches=external_wrenches,
        )
        for case in cfg.disturbances
    )
    return _equilibrium_result(
        model=model,
        nominal=nominal,
        disturbances=disturbances,
        free=free,
        cfg=cfg,
        issues=issues,
        assumptions=assumptions,
        key=key,
        start=start,
    )


def _equilibrium_result(
    *,
    model: _ForceModel,
    nominal: EquilibriumCase,
    disturbances: tuple[EquilibriumCase, ...],
    free: Sequence[str],
    cfg: StabilityConfig,
    issues: set[str],
    assumptions: list[str],
    key: str,
    start: float,
) -> EquilibriumResult:
    """Package physical evidence separately from preparing and solving the LP."""
    status = "unknown" if issues else nominal.status
    if not disturbances:
        robust = "not_tested"
    elif issues or any(c.status == "unknown" for c in (nominal, *disturbances)):
        robust = "unknown"
    else:
        robust = (
            "passed_tested_set"
            if all(c.status == "feasible" for c in (nominal, *disturbances))
            else "failed_tested_set"
        )
    # These are the exact sites/rays used above, even when a load is infeasible.
    # Rays describe the force ON B; the force on A uses their negatives.
    sites = tuple(
        dict(
            {
                k: v
                for k, v in rec.items()
                if k not in ("columns", "generators", "normal_world", "group")
            },
            normal_on_b_world=rec["normal_world"],
            rays_on_b_world=rec["generators"],
            capacity_group=rec["group"],
            max_group_normal_force_n=model.sites.capacity_limits[rec["group"]],
        )
        for rec in model.sites.records
    )
    return EquilibriumResult(
        status,
        nominal,
        disturbances,
        robust,
        tuple(sorted(issues)),
        tuple(assumptions),
        key,
        {
            "elapsed_s": perf_counter() - start,
            "free_bodies": free,
            "force_variables": model.variable_count,
            "force_points": len(model.sites.records),
            "friction_radial_inner_factor": float(np.cos(np.pi / cfg.friction_sides)),
            "point_preparation": model.sites.diagnostics,
            "curved_sampling_used": any(d["curved_subset"] for d in model.sites.diagnostics),
            "curved_fallback": False,
            "clean_contact_points": sum(d["clean_points"] for d in model.sites.diagnostics),
            "reduce_contact_points": cfg.reduce_contact_points,
            "characteristic_length_m": cfg.characteristic_length_m,
        },
        force_sites=sites,
    )


@dataclass(frozen=True)
class SupportSearchResult:
    status: str
    selected_support_ids: tuple
    equilibrium: EquilibriumResult | None
    tested_subsets: int
    minimal_within_declared_candidates: bool
    execution_validated: bool = False


def find_support_requirements(
    assembly: Assembly,
    state: AssemblyState,
    graph: ContactGraph,
    candidates: Iterable[SupportCandidate],
    *,
    config: StabilityConfig | None = None,
    external_wrenches: Iterable[ExternalWrench] = (),
    max_supports: int = 2,
    max_subsets: int = 64,
) -> SupportSearchResult:
    """Bounded enumeration; solutions are support requirements, not robot plans."""
    candidates = tuple(sorted(candidates, key=lambda s: s.support_id))
    external_wrenches = tuple(external_wrenches)
    if len({s.support_id for s in candidates}) != len(candidates):
        raise ValueError("Support IDs must be unique")
    if (
        not isinstance(max_supports, int)
        or max_supports < 0
        or not isinstance(max_subsets, int)
        or max_subsets < 1
    ):
        raise ValueError("Require nonnegative max_supports and positive max_subsets")
    tested, unknown = 0, False
    last = None
    for size in range(min(max_supports, len(candidates)) + 1):
        for subset in combinations(candidates, size):
            if tested >= max_subsets:
                return SupportSearchResult("budget_exhausted", (), last, tested, False)
            last = check_equilibrium(
                assembly,
                state,
                graph,
                config=config,
                external_wrenches=external_wrenches,
                supports=subset,
            )
            tested += 1
            if last.status == "feasible" and last.robustness_status in (
                "not_tested",
                "passed_tested_set",
            ):
                return SupportSearchResult(
                    "requirements_found",
                    tuple(s.support_id for s in subset),
                    last,
                    tested,
                    not unknown,
                )
            unknown |= last.status == "unknown" or last.robustness_status == "unknown"
    return SupportSearchResult(
        "unknown" if unknown else "no_subset_within_limits", (), last, tested, False
    )
