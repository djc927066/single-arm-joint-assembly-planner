"""Directional disturbance limits, sharing one physical contact-force model.

Each row is a separate load experiment. Directions are not simultaneous loads
and do not rotate geometry/gravity. legacy_coupled restores the old two-amplitude
optimization with explicit force-equivalent torque normalization.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

from ..contact.graph import ContactGraph
from ..motion.directions import fibonacci_directions
from ..model import Assembly, AssemblyState, FloatArray, digest, freeze, readonly
from .equilibrium import (
    ExternalWrench,
    StabilityConfig,
    SupportCandidate,
    _check_equilibrium,
)


@dataclass(frozen=True)
class StabilitySweepConfig:
    backend: str = "cuda"
    mode: str = "wrench"
    direction_count: int = 300
    torque_length_m: float = 0.1
    force_reference_n: float = 10.0
    max_score_n: float = 100.0
    batch_size: int = 256
    max_iterations: int = 256
    optimality_tol_n: float = 1e-6
    fallback_to_highs: bool = True
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    def __post_init__(self) -> None:
        if self.backend not in ("highs", "numpy", "cuda"):
            raise ValueError("backend must be highs/numpy/cuda")
        if self.mode not in ("force", "torque", "wrench", "legacy_coupled"):
            raise ValueError("Invalid disturbance mode")
        for name in ("direction_count", "batch_size", "max_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be positive integer")
        minimum = 12 if self.mode == "wrench" else 6
        if self.direction_count < minimum:
            raise ValueError(
                f"direction_count must include at least {minimum} coordinate directions"
            )
        for name in ("torque_length_m", "force_reference_n", "max_score_n", "optimality_tol_n"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if type(self.fallback_to_highs) is not bool:
            raise ValueError("fallback_to_highs must be bool")
        if self.stability.disturbances:
            raise ValueError(
                "Sweep uses one base load; supply external_wrenches. Discrete LoadCases use check_equilibrium."
            )


@dataclass(frozen=True)
class DirectionalLimit:
    part_id: str
    direction_index: int
    status: str
    score_n: float | None
    force_n: float | None
    torque_nm: float | None
    primal_residual: float | None = None
    dual_gap_n: float | None = None
    solver: str = "highs"


@dataclass(frozen=True, eq=False)
class StabilitySweepResult:
    status: str
    directions: FloatArray
    cases: tuple[DirectionalLimit, ...]
    worst_case_index: int | None
    sampled_minimum_n: float | None
    config: StabilitySweepConfig
    input_digest: str
    diagnostics: dict
    assumptions: tuple
    sampled_minimum_load_factor: float | None = None
    schema_version: str = "wrs.assembly.stability_sweep/2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "directions", readonly(self.directions, shape=(None, 6)))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        # Only radial modes define a load factor for a fixed normalized load.
        value = (
            self.sampled_minimum_n / self.config.force_reference_n
            if self.sampled_minimum_n is not None and self.config.mode != "legacy_coupled"
            else None
        )
        object.__setattr__(self, "sampled_minimum_load_factor", value)


def disturbance_directions(count: int = 300, mode: str = "wrench") -> FloatArray:
    """Deterministic normalized [F, tau/L] directions, not physical SI vectors.

    Wrench mode covers S^5 using normalized inverse-normal Halton points,
    including the twelve signed 6D axes and antipodal pairs. These are finite
    quasi-Monte Carlo samples, not an exact covering or random IID samples.
    legacy_coupled retains the old S^2 x S^2 two-amplitude comparison.
    """
    minimum = 12 if mode == "wrench" else 6
    if type(count) is not int or count < minimum:
        raise ValueError(f"Use at least {minimum} directions (includes the coordinate axes)")
    if mode == "wrench":
        from scipy.special import ndtri
        from scipy.stats import qmc

        axes6 = np.stack((np.eye(6), -np.eye(6)), axis=1).reshape(12, 6)
        if count == 12:
            return axes6
        engine = qmc.Halton(6, scramble=False)
        engine.fast_forward(1)  # Skip the all-zero point before inverse CDF.
        z = ndtri(engine.random((count - 12 + 1) // 2))
        z /= np.linalg.norm(z, axis=1)[:, None]
        paired = np.stack((z, -z), axis=1).reshape(-1, 6)
        return np.vstack((axes6, paired[: count - 12]))
    axes = np.vstack((np.eye(3), -np.eye(3)))
    sphere = np.vstack((axes, fibonacci_directions(count - 6))) if count > 6 else axes
    if mode == "force":
        return np.column_stack((sphere, np.zeros_like(sphere)))
    if mode == "torque":
        return np.column_stack((np.zeros_like(sphere), sphere))
    if mode != "legacy_coupled":
        raise ValueError("Invalid disturbance mode")
    # Deterministic product-sphere samples. A finite set is not a guarantee for
    # every wrench in 6D. Include both aligned and opposed force/torque axes.
    from scipy.stats import qmc

    uv = qmc.Halton(4, scramble=False).random(count)
    vectors = []
    for z, angle in (
        (2 * uv[:, 0] - 1, 2 * np.pi * uv[:, 1]),
        (2 * uv[:, 2] - 1, 2 * np.pi * uv[:, 3]),
    ):
        radius = np.sqrt(np.maximum(0, 1 - z * z))
        vectors.append(np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z)))
    result = np.column_stack(vectors)
    result[:6] = np.column_stack((axes, axes))
    if count >= 12:
        result[6:12] = np.column_stack((axes, -axes))
    return result


def _directions(raw: ArrayLike, mode: str) -> FloatArray:
    d = np.asarray(raw, dtype=float)
    if d.ndim != 2 or not len(d) or d.shape[1] not in (3, 6) or not np.all(np.isfinite(d)):
        raise ValueError("Directions must be a finite nonempty N x 3 or N x 6 array")
    if d.shape[1] == 3:
        if mode not in ("force", "torque"):
            raise ValueError("Mixed modes require N x 6 directions")
        d = (
            np.column_stack((d, np.zeros_like(d)))
            if mode == "force"
            else np.column_stack((np.zeros_like(d), d))
        )
    d = d.copy()
    if mode in ("force", "torque"):
        selected = slice(0, 3) if mode == "force" else slice(3, 6)
        other = slice(3, 6) if mode == "force" else slice(0, 3)
        if np.any(d[:, other] != 0):
            raise ValueError("Unexpected nonzero force/torque component")
        length = np.linalg.norm(d[:, selected], axis=1)
        if np.any(length == 0):
            raise ValueError("Zero disturbance direction")
        d[:, selected] /= length[:, None]
    elif mode == "legacy_coupled":
        for s in (slice(0, 3), slice(3, 6)):
            length = np.linalg.norm(d[:, s], axis=1)
            if np.any(length == 0):
                raise ValueError("legacy_coupled requires nonzero force AND torque directions")
            d[:, s] /= length[:, None]
    else:
        length = np.linalg.norm(d, axis=1)
        if np.any(length == 0):
            raise ValueError("Zero wrench direction")
        d /= length[:, None]
    return d


class DirectionalStabilityAnalyzer:
    """Compile contact sites/matrices once; reuse for many direction batches."""

    def __init__(
        self,
        assembly: Assembly,
        state: AssemblyState,
        graph: ContactGraph,
        *,
        config: StabilitySweepConfig | None = None,
        external_wrenches: Iterable[ExternalWrench] = (),
        supports: Iterable[SupportCandidate] = (),
    ) -> None:
        started = perf_counter()
        self.config = config or StabilitySweepConfig()
        self.assembly, self.state, self.graph = assembly, state, graph
        self.external_wrenches, self.supports = tuple(external_wrenches), tuple(supports)
        cfg = self.config
        self.parts = {p.part_id: p for p in assembly.parts if p.part_id in state.poses}
        self.free = tuple(sorted(k for k, p in self.parts.items() if not p.fixed))
        self.index = {k: i for i, k in enumerate(self.free)}
        self.radii = (
            {
                k: float(
                    np.max(
                        np.linalg.norm(
                            self.parts[k].geometry.vertices - self.parts[k].com_local_m, axis=1
                        )
                    )
                )
                for k in self.free
                if self.parts[k].com_local_m is not None
            }
            if cfg.mode == "legacy_coupled"
            else {}
        )
        # Direction limits cannot be certified from an approximate curved subset.
        self.nominal = _check_equilibrium(
            assembly,
            state,
            graph,
            config=cfg.stability,
            external_wrenches=self.external_wrenches,
            supports=self.supports,
            _full_curves=True,
        )
        self.binding = digest(
            (
                "directional_stability/2",
                assembly,
                state,
                graph,
                cfg,
                self.external_wrenches,
                self.supports,
            )
        )
        self.batched = None
        self.basis_setup_s = 0.0
        if self.nominal.status != "feasible":
            self.preparation_s = perf_counter() - started
            return
        sites = self.nominal.force_sites
        self.internal_sites = tuple(
            s for s in sites if s["part_a"] in self.index and s["part_b"] in self.index
        )
        count = sum(len(s["rays_on_b_world"]) for s in sites)
        physical = np.zeros((len(self.free) * 6, count))
        caps = {}
        column = 0
        for site in sites:
            rays = site["rays_on_b_world"]
            indices = np.arange(column, column + len(rays))
            column += len(rays)
            for pid, sign in ((site["part_a"], -1), (site["part_b"], 1)):
                if pid not in self.index:
                    continue
                com = (
                    state.poses[pid][:3, :3] @ self.parts[pid].com_local_m + state.poses[pid][:3, 3]
                )
                rows = np.arange(6 * self.index[pid], 6 * self.index[pid] + 6)
                physical[np.ix_(rows, indices)] = sign * np.vstack(
                    (rays.T, np.cross(site["point_world_m"] - com, rays).T)
                )
            if site["max_group_normal_force_n"] is not None:
                group = caps.setdefault(
                    site["capacity_group"], [[], site["max_group_normal_force_n"]]
                )
                group[0].extend(indices)
        external = np.zeros((len(self.free), 6))
        for i, pid in enumerate(self.free):
            external[i, :3] = self.parts[pid].mass_kg * assembly.gravity_world_m_s2
        for w in self.external_wrenches:
            external[self.index[w.part_id]] += np.r_[w.force_world_n, w.torque_world_nm]
        scale = np.tile(
            np.r_[np.ones(3), np.full(3, 1 / cfg.stability.characteristic_length_m)], len(self.free)
        )
        self.scale = scale
        self.physical = physical
        self.external = external
        # Finite physical capacities + optional legacy coupling + score ceiling.
        extra = len(caps) + 1 + (cfg.mode == "legacy_coupled")
        common = np.zeros((len(scale) + extra, count + extra))
        common[: len(scale), :count] = physical * scale[:, None]
        common[len(scale) :, count:] = np.eye(extra)
        rhs = np.r_[-external.ravel() * scale, np.zeros(extra)]
        for row, (indices, capacity) in enumerate(caps.values(), start=len(scale)):
            common[row, indices] = 1.0
            rhs[row] = capacity
        rhs[-1] = cfg.max_score_n
        self.common, self.rhs = common, rhs
        self.capacity_count = len(caps)
        self.objective_count = 2 if cfg.mode == "legacy_coupled" else 1
        if cfg.backend != "highs":
            from ._batched_lp import BatchedLP

            start = perf_counter()
            self.batched = BatchedLP(
                common,
                rhs,
                self.objective_count,
                backend=cfg.backend,
                max_iterations=cfg.max_iterations,
            )
            self.basis_setup_s = perf_counter() - start
        self.preparation_s = perf_counter() - started

    def _columns(self, jobs: Sequence[tuple[str, int]], directions: FloatArray) -> FloatArray:
        cfg = self.config
        columns = np.zeros((len(jobs), len(self.rhs), self.objective_count))
        ids = np.arange(len(jobs))[:, None]
        rows = 6 * np.array([self.index[pid] for pid, _ in jobs])[:, None] + np.arange(6)
        d = directions[[index for _, index in jobs]].copy()
        d[:, 3:] *= cfg.torque_length_m
        if cfg.mode == "legacy_coupled":
            columns[ids, rows[:, :3], 0] = d[:, :3] * self.scale[rows[:, :3]]
            columns[ids, rows[:, 3:], 1] = d[:, 3:] * self.scale[rows[:, 3:]]
            columns[:, -2, 0] = [-self.radii[pid] / cfg.torque_length_m for pid, _ in jobs]
            columns[:, -2, 1] = 1.0
        else:
            columns[ids, rows, 0] = d * self.scale[rows]
        columns[:, -1] = 1.0
        return columns

    def analyze(
        self, directions: ArrayLike | None = None, *, part_ids: Iterable[str] | None = None
    ) -> StabilitySweepResult:
        cfg = self.config
        started = perf_counter()
        generated = directions is None
        directions = _directions(
            disturbance_directions(cfg.direction_count, cfg.mode) if generated else directions,
            cfg.mode,
        )
        part_ids = self.free if part_ids is None else tuple(part_ids)
        if (
            not part_ids
            or len(set(part_ids)) != len(part_ids)
            or not set(part_ids).issubset(self.free)
        ):
            raise ValueError("Choose distinct present free parts")
        jobs = [(pid, i) for pid in part_ids for i in range(len(directions))]
        key = digest((self.binding, part_ids, directions))
        cases = []
        fallback = 0
        iterations = []
        batches = 0
        assumptions = (
            *self.nominal.assumptions,
            "independent_per_part_direction_loads",
            "minimum_over_sampled_directions_only",
            "force_equivalent_score_N; torque_scaled_by_declared_length",
            "score_ceiling_is_a_lower_bound_only_for_each_tested_ray",
            "finite_sampling_does_not_certify_all_6D_or_simultaneous_multi_body_disturbances",
            "legacy_coupled_is_not_a_fixed_6D_ray"
            if cfg.mode == "legacy_coupled"
            else "single_radial_amplitude; F=alpha*dF; torque=alpha*L*dT; load_factor=alpha/Fref",
        )
        if self.nominal.status != "feasible":
            return StabilitySweepResult(
                "unknown",
                directions,
                (),
                None,
                None,
                cfg,
                key,
                dict(
                    reason="base_equilibrium_not_feasible",
                    nominal_status=self.nominal.status,
                    issues=self.nominal.issues,
                    preparation_s=self.preparation_s,
                    query_s=perf_counter() - started,
                ),
                assumptions,
            )
        reference_s = 0.0
        batched_s = 0.0
        certificate_s = 0.0
        batch_size = cfg.batch_size
        if cfg.backend == "numpy":
            # Keep the dense working tableau near the CPU cache rather than
            # multiplying memory traffic by one large GPU-oriented batch size.
            bytes_per_problem = (
                8 * (len(self.rhs) + 1) * (self.common.shape[1] + self.objective_count + 1)
            )
            batch_size = min(batch_size, max(1, 2 * 1024**2 // bytes_per_problem))
        for first in range(0, len(jobs), batch_size):
            job_batch = jobs[first : first + batch_size]
            columns = self._columns(job_batch, directions)
            batches += 1
            c = np.r_[np.zeros(self.common.shape[1]), np.ones(self.objective_count)]
            start = perf_counter()
            proposed = self.batched.solve(columns) if self.batched else None
            batched_s += perf_counter() - start
            accepted = np.zeros(len(job_batch), bool)
            if proposed is not None:
                weights, dual, optimal, niter = proposed
                iterations.extend(niter.tolist())
                start = perf_counter()
                matrix = np.concatenate(
                    (np.broadcast_to(self.common, (len(job_batch), *self.common.shape)), columns),
                    axis=2,
                )
                residual = np.einsum("bij,bj->bi", matrix, weights) - self.rhs
                reduced = np.einsum("bij,bi->bj", matrix, dual) - c
                scores = weights @ c
                gap = dual @ self.rhs - scores
                bound = min(
                    cfg.stability.force_tol_n,
                    cfg.stability.torque_tol_nm / cfg.stability.characteristic_length_m,
                )
                accepted = (
                    optimal
                    & np.all(np.isfinite(weights), axis=1)
                    & np.all(np.isfinite(dual), axis=1)
                    & (np.max(np.abs(residual), axis=1) <= bound)
                    & (np.min(weights, axis=1) >= -1e-10)
                    & (np.min(reduced, axis=1) >= -1e-9)
                    & (np.abs(gap) <= cfg.optimality_tol_n)
                )
                certificate_s += perf_counter() - start
            for i, (pid, di) in enumerate(job_batch):
                solver = cfg.backend
                if accepted[i]:
                    x = weights[i]
                    err = float(np.max(np.abs(residual[i])))
                    dual_gap = float(abs(gap[i]))
                elif cfg.backend == "highs" or cfg.fallback_to_highs:
                    fallback += cfg.backend != "highs"
                    solver = "highs" if cfg.backend == "highs" else "highs_fallback"
                    start = perf_counter()
                    matrix = np.column_stack((self.common, columns[i]))
                    solved = linprog(
                        -c,
                        A_eq=matrix,
                        b_eq=self.rhs,
                        bounds=(0, None),
                        method="highs",
                        options={
                            "time_limit": cfg.stability.solver_time_limit_s,
                            "primal_feasibility_tolerance": 1e-9,
                            "dual_feasibility_tolerance": 1e-9,
                        },
                    )
                    reference_s += perf_counter() - start
                    if not solved.success:
                        cases.append(
                            DirectionalLimit(pid, di, "unknown", None, None, None, solver=solver)
                        )
                        continue
                    x = solved.x
                    err = float(np.max(np.abs(matrix @ x - self.rhs)))
                    dual_gap = float(abs(-self.rhs @ solved.eqlin.marginals - c @ x))
                    bound = min(
                        cfg.stability.force_tol_n,
                        cfg.stability.torque_tol_nm / cfg.stability.characteristic_length_m,
                    )
                    reduced = -matrix.T @ solved.eqlin.marginals - c
                    if (
                        not np.all(np.isfinite(x))
                        or not np.all(np.isfinite(reduced))
                        or err > bound
                        or np.min(x) < -1e-9
                        or np.min(reduced) < -1e-9
                        or dual_gap > cfg.optimality_tol_n
                    ):
                        cases.append(
                            DirectionalLimit(pid, di, "unknown", None, None, None, solver=solver)
                        )
                        continue
                else:
                    cases.append(
                        DirectionalLimit(pid, di, "unknown", None, None, None, solver=solver)
                    )
                    continue
                amplitudes = np.maximum(x[-self.objective_count :], 0.0)
                score = float(np.sum(amplitudes))
                force = float(amplitudes[0] * np.linalg.norm(directions[di, :3]))
                torque = float(
                    amplitudes[-1] * cfg.torque_length_m * np.linalg.norm(directions[di, 3:])
                )
                status = (
                    "limit_reached"
                    if score >= cfg.max_score_n - cfg.optimality_tol_n
                    else "optimal"
                )
                cases.append(
                    DirectionalLimit(pid, di, status, score, force, torque, err, dual_gap, solver)
                )
        known = [i for i, c in enumerate(cases) if c.score_n is not None]
        worst = min(known, key=lambda i: cases[i].score_n) if known else None
        status = "complete" if len(known) == len(cases) else "partial"
        minimum = cases[worst].score_n if worst is not None and status == "complete" else None
        return StabilitySweepResult(
            status,
            directions,
            tuple(cases),
            worst,
            minimum,
            cfg,
            key,
            dict(
                preparation_s=self.preparation_s,
                query_s=perf_counter() - started,
                basis_setup_s=self.basis_setup_s,
                batches=batches,
                effective_batch_size=batch_size,
                problems=len(jobs),
                force_variables=self.physical.shape[1],
                free_bodies=len(self.free),
                equilibrium_rows=6 * len(self.free),
                capacity_constraints=self.capacity_count,
                internal_contact_sites=len(self.internal_sites),
                eliminated_duplicate_ray_variables=sum(
                    len(s["rays_on_b_world"]) for s in self.internal_sites
                ),
                perturbed_bodies_per_problem=1,
                sampler=(
                    "halton_normal_S5_antipodal_axes12_v1"
                    if cfg.mode == "wrench"
                    else "legacy_product_S2xS2"
                    if cfg.mode == "legacy_coupled"
                    else "fibonacci_S2_axes6"
                )
                if generated
                else "user_supplied",
                torque_reference_nm=cfg.force_reference_n * cfg.torque_length_m,
                global_wrench_ball_certified=False,
                fallback_problems=int(fallback),
                batch_solver_s=batched_s,
                reference_solver_s=reference_s,
                certificate_s=certificate_s,
                max_pivots=max(iterations, default=0),
                gpu_name=self.batched.device_name if self.batched else None,
                common_basis_available=self.batched.start is not None if self.batched else None,
                minimum_is_lower_bound=bool(
                    worst is not None and cases[worst].status == "limit_reached"
                ),
            ),
            assumptions,
        )


def analyze_directional_stability(
    assembly: Assembly,
    state: AssemblyState,
    graph: ContactGraph,
    *,
    config: StabilitySweepConfig | None = None,
    directions: ArrayLike | None = None,
    part_ids: Iterable[str] | None = None,
    external_wrenches: Iterable[ExternalWrench] = (),
    supports: Iterable[SupportCandidate] = (),
) -> StabilitySweepResult:
    analyzer = DirectionalStabilityAnalyzer(
        assembly,
        state,
        graph,
        config=config,
        external_wrenches=external_wrenches,
        supports=supports,
    )
    return analyzer.analyze(directions, part_ids=part_ids)


def limit_wrench(result: StabilitySweepResult, index: int) -> ExternalWrench:
    """External load at a selected solved limit, reusable by check_equilibrium."""
    case = result.cases[index]
    if case.score_n is None:
        raise ValueError("Unknown limit has no load witness")
    d = result.directions[case.direction_index]
    fn = np.linalg.norm(d[:3])
    tn = np.linalg.norm(d[3:])
    return ExternalWrench(
        case.part_id,
        d[:3] * (case.force_n / fn) if fn else np.zeros(3),
        d[3:] * (case.torque_nm / tn) if tn else np.zeros(3),
    )
