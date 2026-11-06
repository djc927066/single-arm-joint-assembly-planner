"""Forward quality-guided DFS, with bounded supports and existing M2 replay."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain, combinations, islice
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..contact.graph import ContactGraph
from ..motion.directions import DirectionConfig, DirectionResult, assembly_directions
from ..model import Assembly, AssemblyState, digest, freeze
from .quality import (
    AssemblabilityScore,
    StepQuality,
    quality_upper_bound,
    score_assemblability,
    sequence_quality,
)
from .sequence import (
    AuxiliarySupport,
    SequenceConfig,
    SequenceEvaluator,
    SequenceResult,
    SequenceStep,
    _balanced,
    replay_sequence,
)
from ..mechanics.stability_sweep import DirectionalStabilityAnalyzer, StabilitySweepConfig

if TYPE_CHECKING:
    # 原 WRS3 的 robotics.graspability 未 vendored (它绑定 WRS3 运行时)。
    # 改指 asmcore 自己声明的端口, 由仓库根 adapters/ 用 WRS1 实现。
    from .ports import GraspabilityPort as GraspabilityAnalyzer
    from .ports import GraspabilityResultLike as GraspabilityResult


@dataclass(frozen=True)
class QualityTransition:
    context: object
    quality: StepQuality
    payload: object = None


@dataclass(frozen=True)
class QualitySearchConfig:
    support_penalty: float = 100.0
    prune: bool = True
    max_expansions: int = 1000
    time_limit_s: float = 180.0
    zero_stability_tol: float = 1e-8
    require_final_without_support: bool = True
    sequence: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(max_auxiliary_resources=1)
    )
    sweep: StabilitySweepConfig = field(default_factory=StabilitySweepConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)

    def __post_init__(self) -> None:
        if not np.isfinite(self.support_penalty) or self.support_penalty < 1:
            raise ValueError("Invalid support penalty")
        if type(self.max_expansions) is not int or self.max_expansions < 1:
            raise ValueError("Invalid expansion budget")
        if not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0:
            raise ValueError("Invalid time budget")
        if not np.isfinite(self.zero_stability_tol) or self.zero_stability_tol < 0:
            raise ValueError("Invalid zero tolerance")
        if type(self.prune) is not bool or type(self.require_final_without_support) is not bool:
            raise ValueError("Expected bool")
        if self.sweep.mode != "wrench":
            raise ValueError("Quality uses a dimensionless 6D radial stability score")
        if digest(self.sequence.stability) != digest(self.sweep.stability):
            raise ValueError("Sequence and sweep must use the same stability/contact model")


class _QualityDepthFirstSearch:
    """Search bookkeeping only; the callback owns all physical checks."""

    def __init__(
        self,
        items: tuple[str, ...],
        expand: Callable[..., Iterable[QualityTransition]],
        config: QualitySearchConfig,
        accept_complete: Callable[..., bool] | None,
    ) -> None:
        self.items = items
        self.expand = expand
        self.config = config
        self.accept_complete = accept_complete
        self.started = perf_counter()
        self.best = None
        self.best_score = -1.0
        self.expanded = 0
        self.pruned = 0
        self.leaves = 0
        self.complete = True
        self.trace = []

    def run(self, initial_context: object) -> dict[str, Any]:
        self.visit(initial_context, self.items, (), (), ())
        return dict(
            best=self.best,
            score=None if self.best is None else self.best_score,
            search_complete=self.complete,
            expansions=self.expanded,
            pruned=self.pruned,
            leaves=self.leaves,
            trace=self.trace,
            elapsed_s=perf_counter() - self.started,
        )

    def time_expired(self) -> bool:
        if perf_counter() - self.started >= self.config.time_limit_s:
            self.complete = False
            return True
        return False

    def prune_prefix(
        self, order: tuple[str, ...], qualities: tuple[StepQuality, ...], remaining_count: int
    ) -> bool:
        if not qualities:
            return False
        upper = quality_upper_bound(
            qualities,
            support_penalty=self.config.support_penalty,
            remaining_steps=remaining_count,
        )
        if self.config.prune and upper <= self.best_score:
            self.pruned += 1
            self.trace.append(dict(order=order, status="bound_pruned", upper_bound=upper))
            return True
        return False

    def consider_complete(
        self,
        order: tuple[str, ...],
        qualities: tuple[StepQuality, ...],
        payloads: tuple[object, ...],
    ) -> None:
        self.leaves += 1
        score = sequence_quality(qualities, support_penalty=self.config.support_penalty)
        improves_best = score > self.best_score
        if not improves_best:
            return
        if self.accept_complete is not None and not self.accept_complete(payloads):
            return
        # Preserve the public tuple result for existing callers.
        self.best = (order, qualities, payloads)
        self.best_score = score
        self.trace.append(dict(order=order, status="incumbent", score=score))

    def visit(
        self,
        context: object,
        remaining: tuple[str, ...],
        order: tuple[str, ...],
        qualities: tuple[StepQuality, ...],
        payloads: tuple[object, ...],
    ) -> None:
        if self.time_expired() or self.prune_prefix(order, qualities, len(remaining)):
            return
        if not remaining:
            self.consider_complete(order, qualities, payloads)
            return
        if self.expanded >= self.config.max_expansions:
            self.complete = False
            return
        self.expanded += 1
        for part_id in remaining:
            if self.time_expired():
                return
            # Keep physical queries lazy: later siblings may be pruned.
            for child in self.expand(context, part_id):
                if self.time_expired():
                    return
                quality = child.quality
                if quality.graspability == 0 or quality.assemblability == 0:
                    continue
                child_order = order + (part_id,)
                child_qualities = qualities + (quality,)
                self.trace.append(
                    dict(
                        order=child_order,
                        status="evaluated",
                        quality=quality,
                        prefix_score=sequence_quality(
                            child_qualities, support_penalty=self.config.support_penalty
                        ),
                    )
                )
                self.visit(
                    child.context,
                    tuple(item for item in remaining if item != part_id),
                    child_order,
                    child_qualities,
                    payloads + (child.payload,),
                )
                if not self.complete:
                    return


def quality_depth_first(
    items: Iterable[str],
    expand: Callable[..., Iterable[QualityTransition]],
    *,
    initial_context: object = None,
    config: QualitySearchConfig | None = None,
    accept_complete: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    """Search transitions supplied lazily by expand(context, item).

    accept_complete(payloads) validates an improving complete candidate.
    Context/payload types are unrestricted; return fields retain the original
    dictionary/tuple API. There is no state-only visited set: prefixes carry
    different minimum qualities and assisted-step counts.
    """
    config = config or QualitySearchConfig()
    items = tuple(sorted(items))
    if not items or len(set(items)) != len(items):
        raise ValueError("Distinct nonempty items required")
    return _QualityDepthFirstSearch(items, expand, config, accept_complete).run(initial_context)


@dataclass(frozen=True)
class QualitySequenceResult:
    status: str
    plan: SequenceResult | None
    score: float | None
    qualities: tuple[StepQuality, ...]
    diagnostics: Mapping[str, Any]
    input_digest: str
    config: QualitySearchConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "qualities", tuple(self.qualities))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))


@dataclass(frozen=True, slots=True)
class _AssemblyContext:
    """Installed parts and currently active auxiliary supports."""

    assembly_state: AssemblyState
    active_support_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _StabilityMetric:
    score: float | None
    evidence: dict


class _AssemblyQualityEvaluator:
    """Physical insertion checks and caches bound to one assembly/configuration.

    Cache keys describe physical states. Prefix minima and assisted-step
    counts belong exclusively to the DFS, never to these caches.
    """

    def __init__(
        self,
        assembly: Assembly,
        grasp_analyzer: GraspabilityAnalyzer,
        goal_state: AssemblyState,
        supports: Iterable[AuxiliarySupport],
        config: QualitySearchConfig,
    ) -> None:
        self.assembly = assembly
        self.grasp_analyzer = grasp_analyzer
        self.config = config
        self.goal = goal_state
        self.parts = {part.part_id: part for part in assembly.parts}
        self.supports = tuple(supports)
        self._validate_inputs()
        self.movable = tuple(key for key, part in self.parts.items() if not part.fixed)
        self.base = AssemblyState(
            {key: pose for key, pose in self.goal.poses.items() if self.parts[key].fixed},
            self.goal.world_revision,
        )
        self.sequence = SequenceEvaluator(assembly, config.sequence, self.supports)
        self.unresolved = []
        self.rejected = []
        self.stability_cache: dict[str, _StabilityMetric] = {}
        self.transition_cache: dict[str, list[QualityTransition]] = {}
        self.verified_plans: dict[str, SequenceResult] = {}
        self.binding = digest(
            (
                "quality_sequence/1",
                assembly,
                self.goal,
                config,
                self.supports,
                grasp_analyzer.binding,
            )
        )

    def _validate_inputs(self) -> None:
        if digest(self.grasp_analyzer.assembly) != digest(self.assembly):
            raise ValueError("Grasp analyzer belongs to another assembly")
        if set(self.goal.poses) != set(self.parts):
            raise ValueError("Provide a full goal assembly state")
        support_ids = [support.candidate.support_id for support in self.supports]
        if len(support_ids) != len(set(support_ids)) or any(
            s.candidate.part_id not in self.parts for s in self.supports
        ):
            raise ValueError("Invalid support IDs")
        if any(s.resource_id == self.config.sequence.handling.resource_id for s in self.supports):
            raise ValueError("Conflicting arm resources")

    def _qualified_grasps(
        self, context: _AssemblyContext, installed_state: AssemblyState, part_id: str
    ) -> GraspabilityResult | None:
        grasp = self.grasp_analyzer.analyze(installed_state, part_id)
        if grasp.count is None:
            self.unresolved.append(
                dict(part_id=part_id, reason="graspability_unknown", details=grasp)
            )
            return None
        if grasp.count == 0:
            self.rejected.append(
                dict(
                    part_id=part_id,
                    present=tuple(context.assembly_state.poses),
                    reason="no_qualified_catalogue_grasp",
                )
            )
            return None
        return grasp

    def _directions_and_score(
        self, graph: ContactGraph, part_id: str
    ) -> tuple[DirectionResult, AssemblabilityScore] | None:
        directions = assembly_directions(graph, (part_id,), config=self.config.direction)
        score = score_assemblability(directions)
        if score.score is None or directions.best_direction is None:
            records = self.rejected if score.score == 0 else self.unresolved
            records.append(dict(part_id=part_id, reason="direction_" + directions.status))
            return None
        if score.score == 0:
            return None
        return directions, score

    def _stability_without_support(
        self, state: AssemblyState, graph: ContactGraph
    ) -> _StabilityMetric:
        """Compute S once per installed state; unknown never becomes zero."""
        key = digest(state)
        if key not in self.stability_cache:
            nominal = self.sequence.equilibrium(state, ())
            if _balanced(nominal):
                sweep = DirectionalStabilityAnalyzer(
                    self.assembly, state, graph, config=self.config.sweep
                ).analyze()
                score = sweep.sampled_minimum_load_factor
                metric = _StabilityMetric(
                    score,
                    dict(
                        status=sweep.status,
                        value=score,
                        scope="minimum_sampled_capped_radial_load_factor",
                        sweep_digest=sweep.input_digest,
                        timings=sweep.diagnostics,
                    ),
                )
            elif nominal.status == "infeasible":
                metric = _StabilityMetric(0.0, dict(status="nominal_infeasible"))
            else:
                metric = _StabilityMetric(
                    None, dict(status=nominal.status, reason="nominal_unresolved")
                )
            self.stability_cache[key] = metric
        return self.stability_cache[key]

    def _support_subsets(
        self, state: AssemblyState, stability_score: float, part_id: str
    ) -> list[tuple[str, ...]]:
        if stability_score > 0:
            return [()]
        eligible = sorted(
            key
            for key, support in self.sequence.supports.items()
            if support.candidate.part_id in state.poses
        )
        config = self.config.sequence
        choices = chain.from_iterable(
            combinations(eligible, count)
            for count in range(1, min(len(eligible), config.max_auxiliary_resources) + 1)
        )
        subsets = list(islice(choices, config.max_support_subsets + 1))
        if len(subsets) > config.max_support_subsets:
            self.unresolved.append(dict(part_id=part_id, reason="support_subset_budget"))
            subsets = subsets[: config.max_support_subsets]
        return subsets

    def _insertion_step(
        self,
        context: _AssemblyContext,
        installed_state: AssemblyState,
        part_id: str,
        support_ids: tuple[str, ...],
        directions: DirectionResult,
    ) -> SequenceStep | None:
        simultaneous = tuple(sorted(set(support_ids) | set(context.active_support_ids)))
        if not self.sequence.resources_valid(simultaneous):
            return None
        balance = self.sequence.equilibrium(installed_state, support_ids)
        if not _balanced(balance):
            if balance.status != "infeasible":
                self.unresolved.append(
                    dict(part_id=part_id, reason="supported_equilibrium_unknown")
                )
            return None
        step, failure = self.sequence.evaluate_insertion(
            installed_state,
            part_id,
            supports_before=context.active_support_ids,
            supports_after=support_ids,
            removal_directions=np.asarray([directions.best_direction]),
        )
        if step is None:
            # A bounded path failure cannot prove global inassemblability.
            self.unresolved.append(
                dict(part_id=part_id, reason="transition_unresolved", details=failure)
            )
        return step

    def expand(self, context: _AssemblyContext, part_id: str) -> Iterator[QualityTransition]:
        """G -> A -> S -> finite supports -> verified insertion."""
        installed_state = AssemblyState(
            dict(context.assembly_state.poses, **{part_id: self.goal.poses[part_id]}),
            self.goal.world_revision,
        )
        cache_key = digest((installed_state, part_id, context.active_support_ids))
        if cache_key in self.transition_cache:
            yield from self.transition_cache[cache_key]
            return
        children = []
        self.transition_cache[cache_key] = children
        grasp = self._qualified_grasps(context, installed_state, part_id)
        if grasp is None:
            return
        graph = self.sequence.graph(installed_state)
        direction_result = self._directions_and_score(graph, part_id)
        if direction_result is None:
            return
        directions, assemblability = direction_result
        metric = self._stability_without_support(installed_state, graph)
        if metric.score is None:
            self.unresolved.append(
                dict(part_id=part_id, reason="stability_unknown", details=metric.evidence)
            )
            return
        stability = 0.0 if metric.score <= self.config.zero_stability_tol else float(metric.score)
        for support_ids in self._support_subsets(installed_state, stability, part_id):
            step = self._insertion_step(context, installed_state, part_id, support_ids, directions)
            if step is None:
                continue
            step = replace(
                step,
                evidence=dict(
                    step.evidence,
                    graspability=grasp,
                    assemblability=assemblability,
                    stability=metric.evidence,
                    quality_stability=stability,
                    insertion_direction=-directions.best_direction,
                ),
            )
            child = QualityTransition(
                _AssemblyContext(installed_state, support_ids),
                StepQuality(stability, grasp.count, float(assemblability.score)),
                step,
            )
            children.append(child)
            yield child
        if not children:
            self.rejected.append(
                dict(
                    part_id=part_id,
                    present=tuple(context.assembly_state.poses),
                    reason="no_valid_support_transition",
                )
            )

    @staticmethod
    def _plan_key(steps: Sequence[SequenceStep]) -> str:
        return digest(tuple((s.part_id, s.supports_before, s.supports_after) for s in steps))

    def _make_plan(self, insertion_steps: Iterable[SequenceStep]) -> SequenceResult:
        # Preserve SequenceResult's established removal-oriented storage.
        insertion_steps = tuple(insertion_steps)
        final_supports = insertion_steps[-1].assembly_supports_after
        key = digest(
            (
                "sequence/1",
                self.assembly,
                self.goal,
                self.supports,
                final_supports,
                self.config.sequence,
            )
        )
        return SequenceResult(
            status="success",
            removal_steps=tuple(reversed(insertion_steps)),
            initial_state=self.goal,
            terminal_state=self.base,
            supports=self.supports,
            initial_support_ids=final_supports,
            config=self.config.sequence,
            input_digest=key,
            diagnostics=dict(
                method="forward_quality_dfs",
                quality_binding=self.binding,
                elapsed_s=0.0,
                staging_and_robot_pending=True,
            ),
        )

    def accept_complete(self, steps: tuple[SequenceStep, ...]) -> bool:
        if self.config.require_final_without_support and steps[-1].assembly_supports_after:
            return False
        plan = self._make_plan(steps)
        replay = replay_sequence(self.assembly, plan, evaluator=self.sequence)
        if replay["status"] != "valid":
            self.unresolved.append(dict(reason="complete_replay_failed", replay=replay))
            return False
        self.verified_plans[self._plan_key(steps)] = replace(
            plan, geometry_validated=True, equilibrium_validated=True
        )
        return True

    def result(self, search_result: dict[str, Any]) -> QualitySequenceResult:
        best = search_result.pop("best")
        plan = None
        qualities = ()
        if best is not None:
            _, qualities, steps = best
            plan = self.verified_plans[self._plan_key(steps)]
            plan = replace(
                plan,
                diagnostics=dict(
                    plan.diagnostics,
                    elapsed_s=search_result["elapsed_s"],
                    quality=search_result["score"],
                ),
            )
        complete = search_result["search_complete"] and not self.unresolved
        return QualitySequenceResult(
            status="success" if plan else "unknown",
            plan=plan,
            score=search_result["score"],
            qualities=qualities,
            diagnostics=dict(
                search_result,
                unresolved=self.unresolved,
                rejected=self.rejected,
                stability_states=len(self.stability_cache),
                transition_cache_size=len(self.transition_cache),
                optimality="optimal_within_declared_catalogue_and_model"
                if complete and plan
                else "not_proven",
                objective="Chen Eq.(2); finite ASP_OLD A; sampled normalized S",
                robot_execution_pending=True,
            ),
            input_digest=self.binding,
            config=self.config,
        )


def plan_quality_sequence(
    assembly: Assembly,
    grasp_analyzer: GraspabilityAnalyzer,
    goal_state: AssemblyState | None = None,
    *,
    supports: Iterable[AuxiliarySupport] = (),
    config: QualitySearchConfig | None = None,
) -> QualitySequenceResult:
    """Search forward insertions using G/A/S, finite supports and M2 replay.

    A uses the finite ASP_OLD cone-class score. S is the sampled, capped,
    dimensionless radial load factor without auxiliary support. G counts
    qualified grasps in a fixed catalogue. M3 remains a separate validation.
    """
    evaluator = _AssemblyQualityEvaluator(
        assembly,
        grasp_analyzer,
        goal_state or assembly.initial_state(),
        supports,
        config or QualitySearchConfig(),
    )
    search_result = quality_depth_first(
        evaluator.movable,
        evaluator.expand,
        initial_context=_AssemblyContext(evaluator.base),
        config=evaluator.config,
        accept_complete=evaluator.accept_complete,
    )
    return evaluator.result(search_result)
