"""Deterministic translation directions on a sphere or a certified subspace.

The inequalities are N @ d >= 0. Results concern this local linear model,
not a finite collision-free path. Rendering and SDF construction are separate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import linprog

from ..model import FloatArray, freeze, readonly

if TYPE_CHECKING:
    from .constraints import ConstraintConfig, LocalConstraints
    from ..contact.graph import ContactGraph


@dataclass(frozen=True)
class DirectionConfig:
    method: str = "socp"
    sample_count: int = 6500
    residual_tol: float = 1e-9
    positive_margin_tol: float = 1e-8
    max_dimension_lps: int = 128
    preferred_direction: tuple = (0.0, 0.0, 1.0)
    allow_roundoff_reduction: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("fibonacci", "socp"):
            raise ValueError("method must be fibonacci or socp")
        if not isinstance(self.allow_roundoff_reduction, bool):
            raise ValueError("allow_roundoff_reduction must be bool")
        for key in ("sample_count", "max_dimension_lps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not (
            np.isfinite(self.residual_tol)
            and np.isfinite(self.positive_margin_tol)
            and 0 < self.residual_tol < self.positive_margin_tol < 1e-3
        ):
            raise ValueError("Require 0 < residual_tol < positive_margin_tol < 1e-3")
        p = readonly(self.preferred_direction, shape=(3,))
        size = np.linalg.norm(p)
        if not np.isfinite(size) or size == 0:
            raise ValueError("preferred_direction must be nonzero")
        object.__setattr__(self, "preferred_direction", tuple(p / size))


@dataclass(frozen=True, eq=False)
class DirectionResult:
    method: str
    status: str
    directions: np.ndarray
    best_direction: np.ndarray | None
    dimension: int | None
    basis: np.ndarray
    angular_margin_rad: float | None
    intrinsic_angular_margin_rad: float | None
    optimality: str
    diagnostics: dict = field(default_factory=dict)
    issues: tuple = ()
    constraints: LocalConstraints | None = None
    geometry_validated: bool = False
    execution_validated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "directions", readonly(self.directions, shape=(None, 3)))
        object.__setattr__(self, "basis", readonly(self.basis, shape=(3, None)))
        if self.best_direction is not None:
            object.__setattr__(self, "best_direction", readonly(self.best_direction, shape=(3,)))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "issues", tuple(self.issues))


@lru_cache(maxsize=8)
def fibonacci_directions(sample_count: int = 6500) -> FloatArray:
    """Immutable, repeatable (K,3) unit directions; midpoint z, golden angle."""
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    i = np.arange(sample_count, dtype=np.float64)
    z = 1 - 2 * (i + 0.5) / sample_count
    angle = i * (np.pi * (3 - np.sqrt(5)))
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    return readonly(np.column_stack((r * np.cos(angle), r * np.sin(angle), z)))


def _filter(normals, directions, tol):
    """Bounded blocks + early rejection; all survivors satisfy every row."""
    kept, scores = [], []
    if len(normals):
        first = np.unique(np.r_[np.argmin(normals, axis=0), np.argmax(normals, axis=0)])
        used = np.zeros(len(normals), dtype=bool)
        used[first] = True
        rest = np.flatnonzero(~used)
        batches = [normals[first]] + [normals[rest[i : i + 32]] for i in range(0, len(rest), 32)]
    else:
        batches = []
    for start in range(0, len(directions), 2048):
        ids = np.arange(start, min(start + 2048, len(directions)))
        low = np.full(len(ids), np.inf)
        for rows in batches:
            dots = np.einsum("ij,kj->ik", rows, directions[ids], optimize=False)
            np.minimum(low, dots.min(axis=0), out=low)
            mask = low >= -tol
            ids, low = ids[mask], low[mask]
            if not len(ids):
                break
        kept.append(ids)
        scores.append(low)
    return np.concatenate(kept), np.concatenate(scores)


def _socp(normals):
    """Bounded maximum margin. Lazy optional dependency, fresh tiny model."""
    try:
        import clarabel
    except ImportError as exc:
        raise ImportError("SOCP requires clarabel; install wrs[assembly-planning].") from exc
    m, k = normals.shape
    q = np.r_[np.zeros(k), -1.0]
    A = sparse.csc_matrix(
        np.vstack(
            (
                np.column_stack((-normals, np.ones(m))),
                np.zeros((1, k + 1)),
                np.column_stack((-np.eye(k), np.zeros(k))),
            )
        )
    )
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    result = clarabel.DefaultSolver(
        sparse.csc_matrix((k + 1, k + 1)),
        q,
        A,
        np.r_[np.zeros(m), 1.0, np.zeros(k)],
        [clarabel.NonnegativeConeT(m), clarabel.SecondOrderConeT(k + 1)],
        settings,
    ).solve()
    info = {
        "status": str(result.status),
        "iterations": result.iterations,
        "primal_residual": float(result.r_prim) if np.isfinite(result.r_prim) else None,
        "dual_residual": float(result.r_dual) if np.isfinite(result.r_dual) else None,
    }
    if str(result.status) != "Solved":
        return None, None, info
    x = np.asarray(result.x)
    if not np.all(np.isfinite(x)):
        return None, None, dict(info, error="nonfinite_solution")
    return x[:k], float(x[-1]), info


def _rref(rows, width=3):
    a = [[v if isinstance(v, Fraction) else Fraction(float(v)) for v in row] for row in rows]
    pivots, r = [], 0
    for c in range(width):
        p = next((i for i in range(r, len(a)) if a[i][c]), None)
        if p is None:
            continue
        a[r], a[p] = a[p], a[r]
        divisor = a[r][c]
        a[r] = [v / divisor for v in a[r]]
        for j in range(len(a)):
            if j != r and a[j][c]:
                multiplier = a[j][c]
                a[j] = [v - multiplier * w for v, w in zip(a[j], a[r])]
        pivots.append(c)
        r += 1
        if r == len(a):
            break
    return a, pivots


def _in_span(row, echelon, pivots):
    v = list(row)
    for a, p in zip(echelon, pivots):
        coefficient = v[p]
        v = [x - coefficient * y for x, y in zip(v, a)]
    return not any(v)


def _basis(echelon, pivots):
    free = [i for i in range(3) if i not in pivots]
    if not free:
        return np.empty((3, 0))
    vectors = []
    for f in free:
        v = [Fraction(0)] * 3
        v[f] = Fraction(1)
        for row, p in zip(echelon, pivots):
            v[p] = -row[f]
        scale = max(abs(x) for x in v)
        vectors.append([float(x / scale) for x in v])
    q, _ = np.linalg.qr(np.asarray(vectors).T, mode="reduced")
    for i in range(len(free)):
        first = np.flatnonzero(np.abs(q[:, i]) > 1e-14)[0]
        if q[first, i] < 0:
            q[:, i] *= -1
    return q


def _exact_dependence(exact, index, dual):
    """Verify n_i + sum(lambda_j*n_j) == 0 with nonnegative rational weights.

    The LP proposes support only. Exact binary-float arithmetic prevents a
    near-zero objective from turning a narrow cone into an equality by fiat.
    A valid certificate proves every included constraint is always tight.
    """
    support = np.flatnonzero(-dual > 0)
    support = sorted(support, key=lambda i: dual[i])[:8]
    for size in range(1, min(3, len(support)) + 1):
        for ids in combinations(support, size):
            augmented = [[exact[j][axis] for j in ids] + [-exact[index][axis]] for axis in range(3)]
            reduced, pivots = _rref(augmented, width=size)
            if any(not any(row[:size]) and row[-1] for row in reduced):
                continue
            values = [Fraction(0)] * size
            for row, p in zip(reduced, pivots):
                values[p] = row[-1]
            if all(v >= 0 for v in values):
                return {
                    "row": index,
                    "support": tuple(int(i) for i in ids),
                    "weights": tuple(str(v) for v in values),
                    "verified": "exact_binary_float_positive_dependence",
                }
    return None


def _reduce_cone(normals, cfg):
    """Reduce exact opposite rows first, then use bounded LP certificates."""
    exact = [tuple(Fraction(float(x)) for x in row) for row in normals]
    equalities, certificates, echelon, pivots = [], [], [], []
    B, projected = np.eye(3), normals
    calls = 0
    roundoff = False
    arithmetic_tol = 128 * np.finfo(np.float64).eps

    def evidence():
        return {
            "dimension_lps": calls,
            "equality_certificates": certificates,
            "dimension_evidence": "arithmetic_roundoff_model" if roundoff else "exact_coefficients",
            "roundoff_reduction": roundoff,
            "roundoff_tol": arithmetic_tol,
        }

    def interior(P):
        nonlocal calls
        if not len(P):
            return True
        if P.shape[1] == 1:
            return bool(np.all(P[:, 0] > 0) or np.all(P[:, 0] < 0))
        if calls >= cfg.max_dimension_lps:
            return False
        calls += 1
        k = P.shape[1]
        r = linprog(
            np.r_[np.zeros(k), -1.0],
            A_ub=np.column_stack((-P, np.ones(len(P)))),
            b_ub=np.zeros(len(P)),
            bounds=[(-1.0, 1.0)] * k + [(None, None)],
            method="highs",
        )
        return (
            r.success
            and np.linalg.norm(r.x[:k]) > 0
            and np.min(P @ (r.x[:k] / np.linalg.norm(r.x[:k]))) > cfg.positive_margin_tol
        )

    # n.d >= 0 and -n.d >= 0 imply n.d == 0 without an LP. Exact
    # coefficients only: never snap an almost-opposite pair into an equality.
    lookup = {row: i for i, row in enumerate(exact)}
    for index, row in enumerate(exact):
        opposite = lookup.get(tuple(-v for v in row))
        if opposite is None or opposite < index:
            continue
        equalities.append(row)
        certificates.append(
            {
                "row": index,
                "support": (opposite,),
                "weights": ("1",),
                "verified": "exact_binary_float_positive_dependence",
            }
        )
    if equalities:
        echelon, pivots = _rref(equalities)
        _, singular, vh = np.linalg.svd(
            np.asarray(equalities, dtype=float), full_matrices=len(equalities) < 3
        )
        rank = int(np.sum(singular > arithmetic_tol))
        # Rescaling/normalizing duplicate rotated normals can introduce an
        # extra exact binary-float rank. Do not collapse a plane to an axis
        # because of roundoff in two otherwise identical opposite pairs.
        if rank < len(pivots):
            if not cfg.allow_roundoff_reduction:
                return None, None, dict(evidence(), reason="ill_conditioned_equality_rank")
            roundoff = True
        B = vh[rank:].T if roundoff else _basis(echelon, pivots)
        if not B.shape[1]:
            return B, np.empty((0, 0)), evidence()
        keep = (
            np.linalg.norm(normals @ B, axis=1) > arithmetic_tol
            if roundoff
            else [not _in_span(row, echelon, pivots) for row in exact]
        )
        projected = normals[keep] @ B
        sizes = np.linalg.norm(projected, axis=1)
        if np.any(sizes == 0) or not np.all(np.isfinite(sizes)):
            return None, None, dict(evidence(), reason="projected_row_numerical_ambiguity")
        projected = projected / sizes[:, None]
    if interior(projected):
        return B, projected, evidence()
    for index in range(len(normals)):
        if (roundoff and np.linalg.norm(normals[index] @ B) <= arithmetic_tol) or (
            not roundoff and _in_span(exact[index], echelon, pivots)
        ):
            continue
        if calls >= cfg.max_dimension_lps:
            break
        calls += 1
        result = linprog(
            -normals[index],
            A_ub=-normals,
            b_ub=np.zeros(len(normals)),
            bounds=[(-1.0, 1.0)] * 3,
            method="highs",
        )
        if not result.success:
            continue
        certificate = _exact_dependence(exact, index, result.ineqlin.marginals)
        if certificate is None and cfg.allow_roundoff_reduction:
            weights = np.maximum(-result.ineqlin.marginals, 0)
            residual = float(np.max(np.abs(normals[index] + weights @ normals)))
            bound = residual + 8 * np.finfo(float).eps * (1 + float(np.sum(weights)))
            if np.sum(weights) <= 16 and bound <= arithmetic_tol:
                support = np.flatnonzero(weights)
                certificate = {
                    "row": index,
                    "support": tuple(int(i) for i in support),
                    "weights": tuple(str(float(weights[i])) for i in support),
                    "verified": "arithmetic_roundoff_bound",
                    "residual_bound": bound,
                }
                roundoff = True
        if certificate is None:
            continue
        equalities.append(exact[index])
        certificates.append(certificate)
        echelon, pivots = _rref(equalities)
        if roundoff:
            _, singular, vh = np.linalg.svd(
                np.asarray(equalities, dtype=float), full_matrices=len(equalities) < 3
            )
            rank = int(np.sum(singular > arithmetic_tol))
            B = vh[rank:].T
        else:
            B = _basis(echelon, pivots)
        if not B.shape[1]:
            if roundoff:
                return (
                    None,
                    None,
                    dict(evidence(), reason="roundoff_reduction_cannot_prove_blockage"),
                )
            # Ill-conditioned exact dependencies are numerically ambiguous.
            singular = np.linalg.svd(np.asarray(equalities, dtype=float), compute_uv=False)
            if singular[-1] <= arithmetic_tol:
                return None, None, dict(evidence(), reason="ill_conditioned_equality_rank")
            return B, np.empty((0, 0)), evidence()
        keep = (
            np.linalg.norm(normals @ B, axis=1) > arithmetic_tol
            if roundoff
            else [not _in_span(row, echelon, pivots) for row in exact]
        )
        projected = normals[keep] @ B
        sizes = np.linalg.norm(projected, axis=1)
        if np.any(sizes == 0) or not np.all(np.isfinite(sizes)):
            break
        projected = projected / sizes[:, None]
        if interior(projected):
            return B, projected, evidence()
    return None, None, dict(evidence(), reason="dimension_budget_or_numerical_ambiguity")


def _preferred(basis, preference):
    for p in (preference, *np.eye(3)):
        u = basis.T @ p
        size = np.linalg.norm(u)
        if size > 1e-12:
            return u / size
    raise ValueError("Cannot choose a direction in an empty basis")


def _best_index(directions, scores, preference):
    candidates = np.arange(len(directions))
    if scores is not None:
        candidates = candidates[scores >= np.max(scores) - 1e-12]
    return int(candidates[np.argmax(directions[candidates] @ preference)])


def solve_directions(
    normals: ArrayLike, *, config: DirectionConfig | None = None
) -> DirectionResult:
    """Compute local pure-translation directions from outward constraint rows.

    ``fibonacci`` returns sampled directions, using a circle or two endpoints
    after certified reduction. ``socp`` returns one continuous margin optimum
    (or a deterministic preference when there is no boundary to optimize).
    Empty samples have status ``no_sample_hit``; only proved zero-dimensional
    cones are ``blocked``. Numerical/budget ambiguity is ``unknown``.
    Angles are radians; the zero-dimensional sphere of an axis has no intrinsic
    angular-margin metric. This API does not accept 6D twists.
    Rotated normals may need a machine-roundoff model, explicitly diagnosed;
    disable it with allow_roundoff_reduction=False for exact-coefficient tests.
    """
    start = perf_counter()
    cfg = config or DirectionConfig()
    original = readonly(normals, shape=(None, 3))
    lengths = np.linalg.norm(original, axis=1)
    if np.any(lengths == 0) or not np.all(np.isfinite(lengths)):
        raise ValueError("Constraint rows must have finite nonzero length")
    N = np.unique(original / lengths[:, None], axis=0)
    preference = np.asarray(cfg.preferred_direction)
    info = {
        "input_rows": len(original),
        "unique_rows": len(N),
        "scope": "local_translation_cone",
        "normal_rank": int(np.linalg.matrix_rank(N, tol=128 * np.finfo(float).eps))
        if len(N)
        else 0,
        "dimension_lps": 0,
        "equality_certificates": (),
        "sample_count_requested": cfg.sample_count,
    }

    def finish(
        B, points, scores=None, *, optimality="none", projected=None, status=None, best=None
    ):
        k = None if B is None else B.shape[1]
        world = np.asarray(points).reshape(-1, 3)
        ambient = intrinsic = None
        if len(world):
            sizes = np.linalg.norm(world, axis=1)
            world = world / sizes[:, None]
            ids, _ = _filter(N, world, cfg.residual_tol)
            if len(ids) != len(world):
                info["reason"] = "subspace_or_solver_residual_failed"
                return finish(None, [], status="unknown")
            choice = _best_index(world, scores, preference) if best is None else best
            chosen = world[choice]
            if len(N):
                low = float(np.min(N @ chosen))
                ambient = float(np.arcsin(np.clip(low, 0, 1)))
            if projected is not None and len(projected) and k and k >= 2:
                low = float(np.min(projected @ (B.T @ chosen)))
                intrinsic = float(np.arcsin(np.clip(low, 0, 1)))
            status = status or ("feasible" if len(N) else "unconstrained")
        else:
            chosen = None
            status = status or "no_sample_hit"
        return DirectionResult(
            cfg.method,
            status,
            world,
            chosen,
            k,
            np.empty((3, 0)) if B is None else B,
            ambient,
            intrinsic,
            (
                "continuous_margin_roundoff_model"
                if optimality == "continuous_margin" and info.get("roundoff_reduction")
                else optimality
            ),
            dict(info, elapsed_s=perf_counter() - start),
        )

    if not len(N):
        if cfg.method == "fibonacci":
            points = fibonacci_directions(cfg.sample_count)
            info.update(sample_domain="sphere", sample_count_tested=len(points))
            return finish(np.eye(3), points, optimality="best_sample")
        return finish(np.eye(3), [preference], optimality="preference_only")

    sphere = None
    if cfg.method == "fibonacci":
        sphere = fibonacci_directions(cfg.sample_count)
        ids, scores = _filter(N, sphere, cfg.residual_tol)
        info.update(
            original_sphere_hits=len(ids), sample_domain="sphere", sample_count_tested=len(sphere)
        )
        if len(scores) and np.max(scores) > cfg.positive_margin_tol:
            return finish(np.eye(3), sphere[ids], scores, optimality="best_sample", projected=N)
    else:
        u, t, solver = _socp(N)
        info["solver"] = solver
        if u is None:
            info["reason"] = "socp_solver_failed"
            return finish(None, [], status="unknown")
        size = np.linalg.norm(u)
        if t > cfg.positive_margin_tol and size > 0:
            if (
                size > 1 + cfg.residual_tol
                or np.min(N @ u - t) < -cfg.residual_tol
                or np.min(N @ (u / size)) <= cfg.positive_margin_tol
            ):
                info["reason"] = "socp_residual_failed"
                return finish(None, [], status="unknown")
            return finish(np.eye(3), [u / size], optimality="continuous_margin", projected=N)

    B, P, reduction = _reduce_cone(N, cfg)
    info.update(reduction)
    if B is None:
        return finish(None, [], status="unknown")
    k = B.shape[1]
    info["cone_is_subspace"] = not len(P)
    if k == 0:
        return finish(B, [], status="blocked")
    if k == 1:
        points = np.array([[1.0], [-1.0]])
        ids, _ = _filter(P, points, cfg.residual_tol)
        info.update(
            sample_domain="axis_endpoints",
            subspace_sample_count=2,
            feasible_axis_endpoints=len(ids),
        )
        world = points[ids] @ B.T
        if cfg.method == "socp" and len(world):
            world = world[[_best_index(world, None, preference)]]
        return finish(B, world, optimality="preference_only", projected=P)
    if cfg.method == "fibonacci":
        if k == 2:
            angle = 2 * np.pi * np.arange(cfg.sample_count) / cfg.sample_count
            points = np.column_stack((np.cos(angle), np.sin(angle)))
            info.update(sample_domain="circle", subspace_sample_count=len(points))
        else:
            points = sphere
        ids, scores = _filter(P, points, cfg.residual_tol)
        return finish(
            B,
            points[ids] @ B.T,
            scores if len(P) else None,
            optimality="best_sample" if len(ids) else "none",
            projected=P,
        )
    if not len(P):
        u = _preferred(B, preference)
        return finish(B, [B @ u], optimality="preference_only", projected=P)
    u, t, solver = _socp(P)
    info["reduced_solver"] = solver
    if u is None or t <= cfg.positive_margin_tol or np.linalg.norm(u) == 0:
        info["reason"] = "reduced_socp_margin_unresolved"
        return finish(None, [], status="unknown")
    size = np.linalg.norm(u)
    if size > 1 + cfg.residual_tol or np.min(P @ u - t) < -cfg.residual_tol:
        info["reason"] = "reduced_socp_residual_failed"
        return finish(None, [], status="unknown")
    return finish(B, [B @ (u / size)], optimality="continuous_margin", projected=P)


def assembly_directions(
    graph: ContactGraph,
    moving_part_ids: str | Iterable[str],
    *,
    config: DirectionConfig | None = None,
    constraint_config: ConstraintConfig | None = None,
) -> DirectionResult:
    """ContactGraph adapter; preserve uncertain contact evidence in the result.

    Use this for real assemblies; ``solve_directions`` is the matrix-level API.
    Positive gaps remain near metadata, never forced zero-gap constraints.
    """
    from .constraints import ConstraintConfig, contact_constraints

    cfg = constraint_config or ConstraintConfig(mode="translation")
    if cfg.mode != "translation":
        raise ValueError(
            "assembly_directions supports pure translation; use candidate_motions for twists"
        )
    constraints = contact_constraints(graph, moving_part_ids, config=cfg)
    result = solve_directions(constraints.matrix_world[:, :3], config=config)
    diagnostics = dict(
        result.diagnostics,
        constraint_status=result.status,
        graph_digest=graph.state_digest,
        moving_part_ids=constraints.moving_part_ids,
    )
    return replace(
        result,
        constraints=constraints,
        issues=constraints.issues,
        status="unknown" if constraints.issues else result.status,
        diagnostics=diagnostics,
    )
