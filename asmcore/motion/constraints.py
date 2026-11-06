"""Vectorized one-sided contact constraints and local translation/twist candidates."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linprog
from ..model import readonly, freeze, digest


@dataclass(frozen=True)
class ConstraintConfig:
    """Local candidate controls; these do not define a finite motion validator.

    ``characteristic_length_m`` scales rotation: u = [v, L*omega].
    ``residual_tol`` applies after normalizing each row of that scaled system.
    Boundary-pair and random budgets limit candidate diversity, not collision
    coverage. Coordinate LPs independently test for a nonzero local direction.
    """
    mode: str = 'translation'
    characteristic_length_m: float = .1
    residual_tol: float = 1e-9
    max_boundary_pairs: int = 2048
    max_normal_seeds: int = 256
    random_samples: int = 64
    seed: int = 0
    allow_idealized_contact: bool = False

    def __post_init__(self):
        if self.mode not in ('translation', 'twist'):
            raise ValueError('mode must be translation or twist')
        if not np.isfinite(self.characteristic_length_m) or self.characteristic_length_m <= 0:
            raise ValueError('Characteristic length must be positive and finite')
        if not np.isfinite(self.residual_tol) or not 0 < self.residual_tol < 1e-3:
            raise ValueError('Residual tolerance must be positive and smaller than 1e-3')
        for k in ('max_boundary_pairs', 'max_normal_seeds', 'random_samples', 'seed'):
            if not isinstance(getattr(self, k), int) or getattr(self, k) < 0:
                raise ValueError(f'{k} must be a nonnegative integer')


@dataclass(frozen=True, eq=False)
class LocalConstraints:
    """Rows satisfy A@[v,omega] >= 0; world metres and an explicit reference.

    Uniform-normal polygon/line cells contribute all vertices, so torque terms
    constrain their full convex cells. Varying normal fields remain samples and
    are marked incomplete; averaging normals is never used. Near rows and gap
    intervals only predict activation and do not restrict the current cone.
    """
    matrix_world: np.ndarray
    points_world_m: np.ndarray
    normals_world: np.ndarray
    reference_world_m: np.ndarray
    row_sources: tuple
    near_matrix_world: np.ndarray
    near_gap_intervals_m: np.ndarray
    issues: tuple
    graph_digest: str
    state_digest: str
    moving_part_ids: tuple
    config: ConstraintConfig

    def __post_init__(self):
        for k, shape in (('matrix_world', (None, 6)), ('points_world_m', (None, 3)),
                         ('normals_world', (None, 3)), ('reference_world_m', (3,)),
                         ('near_matrix_world', (None, 6)), ('near_gap_intervals_m', (None, 2))):
            object.__setattr__(self, k, readonly(getattr(self, k), shape=shape))

    def residuals(self, twists_world):
        """Return normalized row residuals (N,K) for physical twists (K,6)."""
        twists = readonly(twists_world, shape=(None, 6))
        scaled = self.matrix_world.copy()
        scaled[:, 3:] /= self.config.characteristic_length_m
        norm = np.linalg.norm(scaled, axis=1)
        return (self.matrix_world @ twists.T)/norm[:, None]


def rebase_twist(twist_world, old_reference_m, new_reference_m):
    """Change reference while preserving the same rigid velocity field (world)."""
    twist = readonly(twist_world, shape=(6,)).copy()
    offset = readonly(new_reference_m, shape=(3,))-readonly(old_reference_m, shape=(3,))
    twist[:3] += np.cross(twist[3:], offset)
    return twist


def contact_constraints(graph, moving_part_ids, *, reference_world_m=None, config=None):
    """Construct constraints across a rigid moving-group/stationary-group cut.

    An edge internal to either group does not restrict relative group motion.
    Fixed supports cannot be requested as moving parts. Interference, unknown
    geometry and partial coverage remain visible as issues in the result.
    """
    cfg = config or ConstraintConfig()
    moving = tuple(sorted(moving_part_ids))
    if not moving or len(set(moving)) != len(moving) or set(moving)-set(graph.nodes):
        raise ValueError('Moving IDs must be distinct, present and nonempty')
    if any(graph.nodes[k]['fixed'] for k in moving):
        raise ValueError('Cannot move a fixed support')
    ref = readonly(np.mean([graph.nodes[k]['tf'][:3, 3] for k in moving], axis=0)
                   if reference_world_m is None else reference_world_m, shape=(3,))
    points, normals, sources, near_rows, near_gaps, issues = [], [], [], [], [], set()
    for edge in graph.edges:
        if sum(k in moving for k in edge.part_ids) != 1:
            continue
        issues.update(f'{edge.part_ids}:{s}' for s in edge.issues)
        for index, patch in enumerate(edge.patches):
            sign = -1 if patch.part_a in moving else 1
            n = sign*patch.normals_a_world
            p = patch.points_a_world_m if sign < 0 else patch.points_b_world_m
            if patch.classification == 'near':
                near_rows.extend(np.column_stack((n, np.cross(p-ref, n))))
                near_gaps.extend([patch.gap_interval_m]*len(n))
                continue
            if patch.classification != 'active':
                continue
            tag = f'{edge.part_ids}:patch{index}'
            if patch.quality not in ('analytic', 'bounded'):
                issues.add(tag+':unqualified_active_contact')
                continue
            if patch.provenance.get('idealize_contact', False) and not cfg.allow_idealized_contact:
                issues.add(tag+':idealized_contact_disabled')
                continue
            if not len(n):
                issues.add(tag+':missing_normals')
                continue
            if np.any(np.einsum('ij,ij->i', patch.normals_a_world, patch.normals_b_world) > -1+1e-7):
                issues.add(tag+':non_opposing_normals')
                continue
            uniform = np.allclose(n, n[0], rtol=0, atol=1e-10)
            cells = [c for r in patch.regions for c in r.cells_world_m if len(c)]
            if uniform and cells:
                p = np.unique(np.concatenate(cells), axis=0)
                n = np.tile(n[0], (len(p), 1))
            elif patch.dimension != 0:
                issues.add(tag+':sampled_contact_coverage')
            points.extend(p)
            normals.extend(n)
            sources.extend([(edge.part_ids, index)]*len(p))
    points = np.asarray(points).reshape(-1, 3)
    normals = np.asarray(normals).reshape(-1, 3)
    matrix = np.column_stack((normals, np.cross(points-ref, normals)))
    key = digest(('local_constraints/1', graph.state_digest, moving, ref, cfg))
    return LocalConstraints(matrix, points, normals, ref, tuple(sources),
                            np.asarray(near_rows).reshape(-1, 6), np.asarray(near_gaps).reshape(-1, 2),
                            tuple(sorted(issues)), graph.state_digest, key, moving, cfg)


@dataclass(frozen=True, eq=False)
class MotionCandidates:
    """Local directions and evidence; every finite path still needs validation."""
    candidates: tuple
    constraints: LocalConstraints
    status: str
    diagnostics: dict = field(default_factory=dict)
    geometry_validated: bool = False
    execution_validated: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'candidates', freeze(self.candidates))
        object.__setattr__(self, 'diagnostics', freeze(self.diagnostics))


def candidate_motions(graph, moving_part_ids, *, config=None, reference_world_m=None):
    """Generate nonzero translation or normalized six-dimensional directions.

    Coordinate LPs over [-1,1]^D catch lower-dimensional cones that random
    sampling misses. Additional normal, tangent and pairwise boundary seeds
    improve diversity within explicit budgets. Residuals are checked in NumPy
    batches; LP solver tolerances alone never accept a violating direction.
    Returned angular components are per metre of normalized path parameter;
    activation parameters from near samples are estimates, not safe step sizes.
    """
    constraints = contact_constraints(graph, moving_part_ids, reference_world_m=reference_world_m, config=config)
    cfg = constraints.config
    dim = 3 if cfg.mode == 'translation' else 6
    scale = np.r_[np.ones(3), np.full(3, 1/cfg.characteristic_length_m)]
    matrix = (constraints.matrix_world*scale)[:, :dim]
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix/norms[:, None]
    # Exact duplicates are redundant for motion, but remain in the source data
    # for force solvers, patch provenance and spatially distinct force points.
    matrix = np.unique(matrix, axis=0)
    seeds = [*np.eye(dim), *-np.eye(dim)]
    lp_failures, lp_nonzero = [], False
    if len(matrix):
        for axis in (*np.eye(dim), *-np.eye(dim)):
            result = linprog(-axis, A_ub=-matrix, b_ub=np.zeros(len(matrix)),
                             bounds=[(-1, 1)]*dim, method='highs')
            if result.success:
                seeds.append(result.x)
                lp_nonzero |= np.linalg.norm(result.x) > cfg.residual_tol
            else:
                lp_failures.append(result.status)
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        # Full right basis is cheap (D <= 6), including the N < D case.
        if len(matrix) < dim:
            _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
        rank = int(np.sum(singular > cfg.residual_tol))
        seeds.extend(vh[rank:])
        seeds.extend(-vh[rank:])
        if dim == 3:
            selected = matrix[np.linspace(0, len(matrix)-1, min(len(matrix), cfg.max_normal_seeds), dtype=int)]
            seeds.extend(selected)
            for axis in np.eye(3):
                tangent = axis-selected*(selected @ axis)[:, None]
                seeds.extend(tangent)
                seeds.extend(-tangent)
            pairs = 0
            for i in range(len(matrix)):
                stop = min(len(matrix), i+1+cfg.max_boundary_pairs-pairs)
                if stop <= i+1:
                    break
                cross = np.cross(matrix[i], matrix[i+1:stop])
                seeds.extend(cross)
                seeds.extend(-cross)
                pairs += len(cross)
    seeds.extend(np.random.default_rng(cfg.seed).normal(size=(cfg.random_samples, dim)))
    directions = np.asarray(seeds).reshape(-1, dim)
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > cfg.residual_tol]/norms[norms > cfg.residual_tol, None]
    # Deduplicate keys only; keep the unrounded directions for residual checks.
    _, ids = np.unique(np.round(directions, 12), axis=0, return_index=True)
    directions = directions[np.sort(ids)]
    candidates = []
    for start in range(0, len(directions), 128):
        batch = directions[start:start+128]
        residual = matrix @ batch.T
        minimum = residual.min(axis=0) if len(matrix) else np.zeros(len(batch))
        for u, low in zip(batch[minimum >= -cfg.residual_tol], minimum[minimum >= -cfg.residual_tol]):
            full = np.zeros(6)
            full[:dim] = u
            twist = full*scale
            rates = constraints.matrix_world @ twist
            closing = constraints.near_matrix_world @ twist
            near = constraints.near_gap_intervals_m
            mask = (closing < -cfg.residual_tol) & (near[:, 0] > 0)
            activation = float(np.min(near[mask, 0]/-closing[mask])) if np.any(mask) else None
            mode = ('unconstrained' if not len(rates) else 'tangent' if np.max(np.abs(rates)) <= cfg.residual_tol
                    else 'separating' if np.all(rates > cfg.residual_tol) else 'sliding_and_separating')
            candidates.append({'twist_world': twist, 'normalized_direction': full,
                               'minimum_residual': float(low), 'contact_mode': mode,
                               'near_activation_parameter_m': activation,
                               'near_activation_is_estimate': True})
    status = ('local_candidates' if candidates else
              'unknown' if lp_failures or constraints.issues or lp_nonzero else 'locally_blocked')
    return MotionCandidates(tuple(candidates), constraints, status,
                            {'scope': 'reported_active_contacts_only', 'lp_failures': lp_failures,
                             'row_count': len(matrix), 'seed_count': len(directions),
                             'contact_issues': constraints.issues, 'finite_path_checked': False})
