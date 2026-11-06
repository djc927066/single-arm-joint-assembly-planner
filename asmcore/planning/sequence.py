"""Deterministic assembly-by-disassembly with explicit finite handling resources.

Motion is quasistatic. The remainder must stand independently of the carried
part throughout transfer; the carrier balances that part's declared loads.
This conservative decomposition supplies a whole-transfer equilibrium witness.
Staging/grasp feasibility remains an explicit obligation for robot execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from time import perf_counter
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from ..contact.analysis import analyze_contacts
from ..contact.graph import ContactGraph, build_contact_graph
from ..geometry.proximity import MeshProximity
from ..model import Assembly, AssemblyState, FloatArray, Part, digest, freeze
from ..motion.part_motion import MotionConfig, RemovalResult, plan_removal, validate_object_path
from ..mechanics.equilibrium import (
    EquilibriumResult,
    StabilityConfig,
    SupportCandidate,
    check_equilibrium,
)


@dataclass(frozen=True)
class HandlingCapability:
    resource_id: str = "main_arm"
    max_force_n: float = 100.0
    max_torque_nm: float = 10.0

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("Handling resource ID required")
        for k in ("max_force_n", "max_torque_nm"):
            if not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError("Positive finite handling capacities required")


@dataclass(frozen=True)
class AuxiliarySupport:
    resource_id: str
    candidate: SupportCandidate

    def __post_init__(self) -> None:
        if not self.resource_id or not isinstance(self.candidate, SupportCandidate):
            raise ValueError("A named resource and a finite SupportCandidate are required")


@dataclass(frozen=True)
class SequenceConfig:
    method: str = "dfs"
    max_expansions: int = 64
    beam_width: int = 8
    max_auxiliary_resources: int = 2
    max_support_subsets: int = 32
    time_limit_s: float = 120.0
    motion: MotionConfig = field(default_factory=MotionConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    handling: HandlingCapability = field(default_factory=HandlingCapability)

    def __post_init__(self) -> None:
        if self.method not in ("dfs", "beam"):
            raise ValueError("method must be dfs or beam")
        for k in ("max_expansions", "beam_width", "max_support_subsets"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k + " must be positive")
        if type(self.max_auxiliary_resources) is not int or self.max_auxiliary_resources < 0:
            raise ValueError("Invalid auxiliary resource limit")
        if not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0:
            raise ValueError("Positive time limit required")


@dataclass(frozen=True, eq=False)
class SequenceStep:
    """Storage follows removal: before includes the part, after omits it.

    The assembly_* properties expose installation order without copying data.
    """

    part_id: str
    before: AssemblyState
    after: AssemblyState
    removal: RemovalResult
    supports_before: tuple[str, ...]
    supports_after: tuple[str, ...]
    events: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]
    cost: float

    def __post_init__(self) -> None:
        for k in ("events", "evidence"):
            object.__setattr__(self, k, freeze(getattr(self, k)))
        for k in ("supports_before", "supports_after"):
            object.__setattr__(self, k, tuple(getattr(self, k)))

    @property
    def assembly_before(self) -> AssemblyState:
        """Installed remainder before insertion (the carried part is omitted)."""
        return self.after

    @property
    def assembly_after(self) -> AssemblyState:
        """Installed assembly after insertion."""
        return self.before

    @property
    def assembly_supports_before(self) -> tuple[str, ...]:
        return self.supports_after

    @property
    def assembly_supports_after(self) -> tuple[str, ...]:
        return self.supports_before

    @property
    def assembly_poses(self) -> tuple[FloatArray, ...]:
        return tuple(reversed(self.removal.poses))

    @property
    def assembly_events(self) -> tuple[Mapping[str, Any], ...]:
        inverse = {
            "acquire_auxiliary": "release_auxiliary",
            "release_auxiliary": "acquire_auxiliary",
            "acquire_part": "release_part",
            "move_to_outside": "insert_part",
            "handoff_to_staging": "take_from_staging",
        }
        return tuple(freeze(dict(e, kind=inverse[e["kind"]])) for e in reversed(self.events))


@dataclass(frozen=True, eq=False)
class SequenceResult:
    status: str
    removal_steps: tuple[SequenceStep, ...]
    initial_state: AssemblyState
    terminal_state: AssemblyState
    supports: tuple[AuxiliarySupport, ...]
    initial_support_ids: tuple[str, ...]
    config: SequenceConfig
    input_digest: str
    diagnostics: Mapping[str, Any]
    geometry_validated: bool = False
    equilibrium_validated: bool = False
    execution_validated: bool = False
    schema_version: str = "wrs.assembly.sequence/1"

    def __post_init__(self) -> None:
        for k in ("removal_steps", "supports", "initial_support_ids"):
            object.__setattr__(self, k, tuple(getattr(self, k)))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))

    @property
    def assembly_steps(self) -> tuple[SequenceStep, ...]:
        return tuple(reversed(self.removal_steps))


def _restricted_config(config: StabilityConfig, state: AssemblyState) -> StabilityConfig:
    return replace(
        config,
        disturbances=tuple(
            replace(c, wrenches=tuple(w for w in c.wrenches if w.part_id in state.poses))
            for c in config.disturbances
        ),
    )


def _balanced(result: EquilibriumResult) -> bool:
    return result.status == "feasible" and result.robustness_status in (
        "not_tested",
        "passed_tested_set",
    )


def _handling(assembly: Assembly, part: Part, config: SequenceConfig) -> dict[str, Any] | None:
    if part.mass_kg is None or part.com_local_m is None:
        return None
    gravity = part.mass_kg * assembly.gravity_world_m_s2
    required = [(float(np.linalg.norm(gravity)), 0.0)]
    for case in config.stability.disturbances:
        loads = [w for w in case.wrenches if w.part_id == part.part_id]
        force = gravity + sum((w.force_world_n for w in loads), np.zeros(3))
        torque = sum((w.torque_world_nm for w in loads), np.zeros(3))
        required.append((float(np.linalg.norm(force)), float(np.linalg.norm(torque))))
    f, t = np.max(required, axis=0)
    return dict(
        feasible=bool(f <= config.handling.max_force_n and t <= config.handling.max_torque_nm),
        force_n=float(f),
        torque_at_com_nm=float(t),
        resource_id=config.handling.resource_id,
        meaning="finite_quasistatic_wrench_at_COM; grasp_and_staging_pending",
    )


class SequenceEvaluator:
    """Contact and equilibrium caches bound to one assembly/configuration.

    An insertion uses the same physical witness as the reverse removal:
    the carried part and the remaining assembly must balance independently.
    """

    def __init__(
        self, assembly: Assembly, config: SequenceConfig, supports: Iterable[AuxiliarySupport] = ()
    ) -> None:
        self.assembly = assembly
        self.config = config
        self.supports = {support.candidate.support_id: support for support in supports}
        self.backend = MeshProximity(numerical_tol_m=config.motion.numerical_tol_m)
        self.graphs: dict[str, ContactGraph] = {}
        self.balances: dict[str, EquilibriumResult] = {}

    def graph(self, state: AssemblyState) -> ContactGraph:
        key = digest(state)
        if key not in self.graphs:
            if len(self.graphs) >= 128:
                self.graphs.clear()
            analysis = analyze_contacts(self.assembly, state, backend=self.backend)
            self.graphs[key] = build_contact_graph(self.assembly, state, analysis)
        return self.graphs[key]

    def equilibrium(self, state: AssemblyState, ids: Sequence[str]) -> EquilibriumResult:
        # Pose/revision and the exact support subset both affect the model.
        key = digest((state, tuple(sorted(ids))))
        if key not in self.balances:
            if len(self.balances) >= 512:
                self.balances.clear()
            self.balances[key] = check_equilibrium(
                self.assembly,
                state,
                self.graph(state),
                config=_restricted_config(self.config.stability, state),
                supports=tuple(self.supports[key].candidate for key in ids),
            )
        return self.balances[key]

    def resources_valid(self, ids: Iterable[str]) -> bool:
        resources = [self.supports[key].resource_id for key in ids]
        return (
            len(resources) == len(set(resources))
            and len(resources) <= self.config.max_auxiliary_resources
        )

    def evaluate_insertion(
        self,
        assembled_state: AssemblyState,
        part_id: str,
        *,
        supports_before: Sequence[str] = (),
        supports_after: Sequence[str] = (),
        removal_directions: ArrayLike | None = None,
    ) -> tuple[SequenceStep | None, dict[str, Any] | None]:
        """Check installation at the pose in assembled_state.

        supports_before acts on the already installed remainder.
        supports_after acts after the new part reaches its assembly pose.
        Directions point OUT of the assembly; insertion reverses the path.
        Returned SequenceStep retains removal storage; use its assembly_*
        properties to read the forward state and support order.
        """
        return self.evaluate(
            assembled_state,
            supports_after,
            part_id,
            remainder_support_ids=supports_before,
            directions=removal_directions,
        )

    def evaluate_removal(
        self,
        state: AssemblyState,
        active_support_ids: Sequence[str],
        part_id: str,
        *,
        remainder_support_ids: Iterable[str] | None = None,
        directions: ArrayLike | None = None,
    ) -> tuple[SequenceStep | None, dict[str, Any] | None]:
        """Explicit removal spelling of the original evaluate API."""
        if remainder_support_ids is None and directions is None:
            # Older subclasses may implement only the three positional arguments.
            return self.evaluate(state, active_support_ids, part_id)
        return self.evaluate(
            state,
            active_support_ids,
            part_id,
            remainder_support_ids=remainder_support_ids,
            directions=directions,
        )

    def _support_options(
        self, remainder: AssemblyState, requested_ids: Iterable[str] | None
    ) -> Iterator[tuple[str, ...]]:
        candidates = sorted(
            key
            for key, support in self.supports.items()
            if support.candidate.part_id in remainder.poses
        )
        if requested_ids is not None:
            requested_ids = tuple(sorted(requested_ids))
            if len(requested_ids) != len(set(requested_ids)) or not set(requested_ids).issubset(
                candidates
            ):
                raise ValueError("Remainder supports must be distinct and act on present parts")
            yield requested_ids
            return
        maximum = min(len(candidates), self.config.max_auxiliary_resources)
        for count in range(maximum + 1):
            yield from combinations(candidates, count)

    def evaluate(
        self,
        state: AssemblyState,
        active: Sequence[str],
        part_id: str,
        *,
        remainder_support_ids: Iterable[str] | None = None,
        directions: ArrayLike | None = None,
    ) -> tuple[SequenceStep | None, dict[str, Any] | None]:
        """Original removal API, retained for callers and evaluator subclasses."""
        part = next(part for part in self.assembly.parts if part.part_id == part_id)
        handling = _handling(self.assembly, part, self.config)
        if handling is None or not handling["feasible"]:
            return None, dict(reason="handling_capacity_or_mass_unknown", part_id=part_id)
        remainder = AssemblyState(
            {key: pose for key, pose in state.poses.items() if key != part_id},
            state.world_revision,
        )
        for checked, chosen in enumerate(
            self._support_options(remainder, remainder_support_ids), 1
        ):
            if checked > self.config.max_support_subsets:
                return None, dict(reason="support_subset_budget", part_id=part_id)
            simultaneous = tuple(sorted(set(active) | set(chosen)))
            if not self.resources_valid(simultaneous):
                continue
            remainder_balance = self.equilibrium(remainder, chosen)
            if not _balanced(remainder_balance):
                continue
            initial_balance = self.equilibrium(state, active)
            if not _balanced(initial_balance):
                return None, dict(reason="initial_equilibrium_unknown_or_failed", part_id=part_id)
            removal = plan_removal(
                self.assembly,
                state,
                part_id,
                graph=self.graph(state),
                backend=self.backend,
                config=self.config.motion,
                directions=directions,
            )
            if removal.status != "success":
                return None, dict(
                    reason="removal_search_exhausted",
                    part_id=part_id,
                    details=removal.diagnostics,
                )
            step = self._removal_step(
                part_id,
                state,
                remainder,
                active,
                chosen,
                removal,
                initial_balance,
                remainder_balance,
                handling,
            )
            return step, None
        return None, dict(reason="remainder_needs_support_or_unqualified_contact", part_id=part_id)

    def _removal_step(
        self,
        part_id: str,
        state: AssemblyState,
        remainder: AssemblyState,
        active: Sequence[str],
        chosen: tuple[str, ...],
        removal: RemovalResult,
        initial_balance: EquilibriumResult,
        remainder_balance: EquilibriumResult,
        handling: Mapping[str, Any],
    ) -> SequenceStep:
        length = sum(
            np.linalg.norm(end[:3, 3] - start[:3, 3])
            for start, end in zip(removal.poses, removal.poses[1:])
        )
        return SequenceStep(
            part_id=part_id,
            before=state,
            after=remainder,
            removal=removal,
            supports_before=active,
            supports_after=chosen,
            events=_removal_events(part_id, active, chosen, self),
            evidence=dict(
                before_equilibrium=initial_balance.input_digest,
                remainder_equilibrium=remainder_balance.input_digest,
                handling=handling,
                transfer="independently_balanced_remainder_and_carried_part",
            ),
            cost=float(length + 0.1 * len(chosen)),
        )


def _removal_events(
    part_id: str,
    supports_before: Sequence[str],
    supports_after: Sequence[str],
    evaluator: SequenceEvaluator,
) -> tuple[dict[str, Any], ...]:
    """Acquire the receiver before releasing the old support.

    Adding optional force generators preserves the prior equilibrium.
    Reverse this event list (with inverted actions) to obtain installation.
    """
    events = [
        dict(
            kind="acquire_auxiliary",
            support_id=key,
            resource_id=evaluator.supports[key].resource_id,
        )
        for key in supports_after
        if key not in supports_before
    ]
    events.append(
        dict(
            kind="acquire_part",
            part_id=part_id,
            resource_id=evaluator.config.handling.resource_id,
        )
    )
    events.extend(
        dict(
            kind="release_auxiliary",
            support_id=key,
            resource_id=evaluator.supports[key].resource_id,
        )
        for key in supports_before
        if key not in supports_after
    )
    events.extend(
        (
            dict(kind="move_to_outside", part_id=part_id),
            dict(kind="handoff_to_staging", part_id=part_id, requires_execution_validation=True),
        )
    )
    return tuple(events)


class _RemovalSearchNode(NamedTuple):
    """A DFS continuation or a beam candidate; all geometry is shared."""

    state: AssemblyState
    active_support_ids: tuple
    steps: tuple[SequenceStep, ...]
    cost: float
    pending_parts: list[str] | None = None


def plan_sequence(
    assembly: Assembly,
    initial_state: AssemblyState | None = None,
    *,
    supports: Iterable[AuxiliarySupport] = (),
    initial_support_ids: Iterable[str] = (),
    config: SequenceConfig | None = None,
    evaluator: SequenceEvaluator | None = None,
) -> SequenceResult:
    """Find a first feasible removal order, then independently replay insertion.

    DFS evaluates siblings lazily; beam keeps the declared number of candidates.
    This search makes no quality-optimality claim. Use plan_quality_sequence
    for the S/G/A objective and its prefix-dependent pruning.
    """
    cfg = config or SequenceConfig()
    state = initial_state or assembly.initial_state()
    supports = tuple(supports)
    initial_support_ids = tuple(sorted(initial_support_ids))
    ids = [s.candidate.support_id for s in supports]
    parts = {p.part_id: p for p in assembly.parts}
    if len(ids) != len(set(ids)) or not set(initial_support_ids).issubset(ids):
        raise ValueError("Invalid support IDs")
    if any(s.resource_id == cfg.handling.resource_id for s in supports):
        raise ValueError("Main and auxiliary resources must differ")
    if any(s.candidate.part_id not in parts for s in supports):
        raise ValueError("Support refers to unknown part")
    if any(k not in parts for k in state.poses):
        raise ValueError("State refers to unknown part")
    evaluator = evaluator or SequenceEvaluator(assembly, cfg, supports)
    if digest((evaluator.assembly, evaluator.config, tuple(evaluator.supports.values()))) != digest(
        (assembly, cfg, supports)
    ):
        raise ValueError("Evaluator is bound to another assembly/configuration/support set")
    if not evaluator.resources_valid(initial_support_ids):
        raise ValueError("Initial resources exceed capacity or conflict")
    if any(evaluator.supports[i].candidate.part_id not in state.poses for i in initial_support_ids):
        raise ValueError("Initial support must act on present part")
    key = digest(("sequence/1", assembly, state, supports, initial_support_ids, cfg))
    started = perf_counter()
    failures = []
    expanded_states = 0

    def finish(status, removal_steps, terminal, reason):
        return SequenceResult(
            status=status,
            removal_steps=removal_steps,
            initial_state=state,
            terminal_state=terminal,
            supports=supports,
            initial_support_ids=initial_support_ids,
            config=cfg,
            input_digest=key,
            diagnostics=dict(
                reason=reason,
                elapsed_s=perf_counter() - started,
                expansions=expanded_states,
                failures=failures,
                optimality="not_claimed",
                staging_and_robot_pending=True,
            ),
            geometry_validated=status == "success",
            equilibrium_validated=status == "success",
            execution_validated=False,
        )

    if not _balanced(evaluator.equilibrium(state, initial_support_ids)):
        return finish("unknown", (), state, "initial_equilibrium_unknown_or_failed")
    # DFS keeps untried siblings as a continuation. Do not calculate every
    # sibling's expensive geometry/equilibrium before following the first one.
    frontier = [_RemovalSearchNode(state, initial_support_ids, (), 0.0)]
    seen = set()
    while frontier:
        removal_state, active_support_ids, removal_steps, path_cost, pending_parts = frontier.pop()
        remaining_parts = (
            [k for k in removal_state.poses if not parts[k].fixed]
            if pending_parts is None
            else pending_parts
        )
        if not remaining_parts:
            plan = finish("success", removal_steps, removal_state, "candidate_found")
            replay = replay_sequence(assembly, plan, evaluator=evaluator)
            if replay["status"] == "valid":
                return plan
            failures.append(replay)
            return finish("unknown", (), state, "forward_replay_failed")
        if (
            pending_parts is None and expanded_states >= cfg.max_expansions
        ) or perf_counter() - started > cfg.time_limit_s:
            return finish("exhausted", (), state, "search_budget_exhausted")
        if pending_parts is None:
            state_key = digest(
                (
                    removal_state,
                    active_support_ids,
                    tuple((i, evaluator.supports[i].resource_id) for i in active_support_ids),
                )
            )
            if state_key in seen:
                continue
            seen.add(state_key)
            expanded_states += 1
            remaining_parts.sort(key=lambda k: (-removal_state.poses[k][2, 3], k))
        children = []
        for index, part_id in enumerate(remaining_parts):
            if perf_counter() - started > cfg.time_limit_s:
                break
            # Keep the legacy call shape for user-supplied evaluators.
            step, failure = evaluator.evaluate(removal_state, active_support_ids, part_id)
            if step is None:
                failures.append(failure)
                continue
            children.append(
                _RemovalSearchNode(
                    step.after, step.supports_after, removal_steps + (step,), path_cost + step.cost
                )
            )
            if cfg.method == "dfs":
                if index + 1 < len(remaining_parts):
                    frontier.append(
                        _RemovalSearchNode(
                            removal_state,
                            active_support_ids,
                            removal_steps,
                            path_cost,
                            remaining_parts[index + 1 :],
                        )
                    )
                break
        if cfg.method == "dfs":
            frontier.extend(children)
        else:
            frontier.extend(children)
            frontier.sort(
                key=lambda node: (
                    len([k for k in node.state.poses if not parts[k].fixed]),
                    node.cost,
                    tuple(node.state.poses),
                )
            )
            frontier = list(reversed(frontier[: cfg.beam_width]))
    return finish("unknown", (), state, "finite_search_exhausted_not_global_infeasibility")


def replay_sequence(
    assembly: Assembly, plan: SequenceResult, *, evaluator: SequenceEvaluator | None = None
) -> dict[str, Any]:
    """Independently recheck forward insertion paths, state chain and load transfer."""
    if plan.status != "success":
        return dict(status="unknown", reason="no_complete_plan")
    oracle = evaluator or SequenceEvaluator(assembly, plan.config, plan.supports)
    if digest((oracle.assembly, oracle.config, tuple(oracle.supports.values()))) != digest(
        (assembly, plan.config, plan.supports)
    ):
        raise ValueError("Evaluator does not match this plan")
    expected = digest(
        (
            "sequence/1",
            assembly,
            plan.initial_state,
            plan.supports,
            plan.initial_support_ids,
            plan.config,
        )
    )
    if expected != plan.input_digest:
        raise ValueError("Plan is stale for this assembly/configuration")
    current = plan.terminal_state
    active = (
        plan.removal_steps[-1].supports_after if plan.removal_steps else plan.initial_support_ids
    )
    checks = []
    for step in plan.assembly_steps:
        expected_after = AssemblyState(
            {k: v for k, v in step.before.poses.items() if k != step.part_id},
            step.before.world_revision,
        )
        if step.part_id not in step.before.poses or digest(expected_after) != digest(step.after):
            return dict(status="unknown", reason="invalid_single_part_transition")
        expected_events = [
            dict(kind="acquire_auxiliary", support_id=i, resource_id=oracle.supports[i].resource_id)
            for i in step.supports_after
            if i not in step.supports_before
        ]
        expected_events.append(
            dict(
                kind="acquire_part",
                part_id=step.part_id,
                resource_id=plan.config.handling.resource_id,
            )
        )
        expected_events.extend(
            dict(kind="release_auxiliary", support_id=i, resource_id=oracle.supports[i].resource_id)
            for i in step.supports_before
            if i not in step.supports_after
        )
        expected_events.extend(
            (
                dict(kind="move_to_outside", part_id=step.part_id),
                dict(
                    kind="handoff_to_staging",
                    part_id=step.part_id,
                    requires_execution_validation=True,
                ),
            )
        )
        if digest(tuple(expected_events)) != digest(step.events):
            return dict(status="unknown", reason="invalid_handoff_events")
        if digest(current) != digest(step.after) or tuple(active) != step.supports_after:
            return dict(status="unknown", reason="state_or_support_chain_mismatch")
        if not oracle.resources_valid(
            tuple(sorted(set(step.supports_before) | set(step.supports_after)))
        ):
            return dict(status="unknown", reason="support_transition_resource_conflict")
        if not _balanced(oracle.equilibrium(current, step.supports_after)) or not _balanced(
            oracle.equilibrium(step.before, step.supports_before)
        ):
            return dict(status="unknown", reason="forward_equilibrium_failed")
        part = next(p for p in assembly.parts if p.part_id == step.part_id)
        if part.fixed:
            return dict(status="unknown", reason="cannot_move_fixed_part")
        handling = _handling(assembly, part, plan.config)
        if handling is None or not handling["feasible"]:
            return dict(status="unknown", reason="handling_capacity")
        outside = AssemblyState(
            dict(current.poses, **{step.part_id: step.assembly_poses[0]}), current.world_revision
        )
        validation = validate_object_path(
            assembly,
            outside,
            step.part_id,
            step.assembly_poses,
            policy=step.removal.policy,
            config=plan.config.motion,
            backend=oracle.backend,
        )
        if validation.status != "valid":
            return dict(
                status="unknown",
                reason="forward_geometry_failed",
                part_id=step.part_id,
                details=validation,
            )
        if not np.allclose(
            step.assembly_poses[-1], step.before.poses[step.part_id], atol=1e-9, rtol=0
        ):
            return dict(status="unknown", reason="target_pose_mismatch")
        checks.append(
            dict(part_id=step.part_id, geometry=validation.status, equilibrium="feasible")
        )
        current, active = step.before, step.supports_before
    if digest(current) != digest(plan.initial_state) or tuple(active) != tuple(
        plan.initial_support_ids
    ):
        return dict(status="unknown", reason="incomplete_forward_plan")
    return dict(status="valid", checks=checks, execution_validated=False)


plan_assembly = plan_sequence
