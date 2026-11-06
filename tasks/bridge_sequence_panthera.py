"""Search and visualise the Bridge assembly order with the Panthera gripper.

This is deliberately an assembly-sequence demonstration.  It evaluates the
paper's S/G/A quantities for every explored prefix and hard-prunes S=0 for the
single-arm setting.  It does not search an initial layout or a robot trajectory.

Run from the repository root::

    python -m tasks.bridge_sequence_panthera
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from time import perf_counter

import numpy as np

from adapters.panthera_gripper import PantheraGripperPort
from asmcore.mechanics.stability_sweep import (
    DirectionalStabilityAnalyzer,
    StabilitySweepConfig,
)
from asmcore.model import AssemblyState
from asmcore.motion.directions import DirectionConfig, assembly_directions
from asmcore.planning.quality import StepQuality, score_assemblability, sequence_quality
from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
from asmcore.robotics import GraspabilityAnalyzer, GraspabilityConfig
from tasks.bridge import build_bridge, movable_ids


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "tasks" / "output" / "bridge_panthera_sequence.png"
DEFAULT_JSON = ROOT / "tasks" / "output" / "bridge_panthera_sequence.json"


@dataclass(frozen=True)
class EvaluatedStep:
    part_id: str
    installed: tuple[str, ...]
    quality: StepQuality
    grasp_result: object
    grasp: object
    direction: np.ndarray
    a_case: str
    a_region: str


@dataclass(frozen=True)
class SequenceCandidate:
    order: tuple[str, ...]
    steps: tuple[EvaluatedStep, ...]
    score: float


def search_sequence(*, direction_count: int = 64) -> tuple[SequenceCandidate, dict]:
    """Enumerate all 4! Bridge orders and return the best stable S/G/A branch."""
    assembly = build_bridge()
    goal = assembly.initial_state()
    parts = {part.part_id: part for part in assembly.parts}
    fixed = frozenset(part.part_id for part in assembly.parts if part.fixed)
    gripper = PantheraGripperPort()
    grasp_analyzer = GraspabilityAnalyzer(
        assembly, gripper, config=GraspabilityConfig(max_grasps=8)
    )
    sequence_config = SequenceConfig(max_auxiliary_resources=0)
    sequence = SequenceEvaluator(assembly, sequence_config)
    sweep_config = StabilitySweepConfig(
        backend="numpy",
        direction_count=direction_count,
        batch_size=min(direction_count, 64),
        stability=sequence_config.stability,
    )
    direction_config = DirectionConfig(method="socp")
    cache: dict[tuple[frozenset[str], str], EvaluatedStep | str] = {}
    rejected: dict[str, int] = {}
    counters = {
        "transition_attempts": 0,
        "unique_transitions": 0,
        "equilibrium_queries": 0,
        "stability_sweeps": 0,
        "G_queries": 0,
        "A_queries": 0,
    }

    def reject(reason: str) -> str:
        rejected[reason] = rejected.get(reason, 0) + 1
        return reason

    def evaluate(prefix: tuple[str, ...], part_id: str) -> EvaluatedStep | str:
        installed_movable = frozenset(prefix + (part_id,))
        key = (installed_movable, part_id)
        if key in cache:
            return cache[key]
        counters["unique_transitions"] += 1
        present = fixed | installed_movable
        state = AssemblyState({pid: tf for pid, tf in goal.poses.items() if pid in present})

        counters["G_queries"] += 1
        grasp_result = grasp_analyzer.analyze(state, part_id)
        if grasp_result.count is None:
            cache[key] = reject("G_unknown")
            return cache[key]
        if grasp_result.count == 0:
            cache[key] = reject("G_zero")
            return cache[key]

        graph = sequence.graph(state)
        counters["A_queries"] += 1
        directions = assembly_directions(graph, (part_id,), config=direction_config)
        a_score = score_assemblability(directions)
        if a_score.score in (None, 0) or directions.best_direction is None:
            cache[key] = reject("A_unknown" if a_score.score is None else "A_zero")
            return cache[key]

        counters["equilibrium_queries"] += 1
        equilibrium = sequence.equilibrium(state, ())
        if equilibrium.status == "infeasible":
            cache[key] = reject("S_zero")
            return cache[key]
        if equilibrium.status != "feasible":
            cache[key] = reject("S_unknown")
            return cache[key]
        counters["stability_sweeps"] += 1
        sweep = DirectionalStabilityAnalyzer(
            assembly, state, graph, config=sweep_config
        ).analyze()
        stability = sweep.sampled_minimum_load_factor
        if stability is None:
            cache[key] = reject("S_unknown")
            return cache[key]
        if stability <= 1e-8:
            cache[key] = reject("S_zero")
            return cache[key]

        accepted = grasp_analyzer.accepted_grasps(grasp_result)
        step = EvaluatedStep(
            part_id=part_id,
            installed=tuple(sorted(installed_movable)),
            quality=StepQuality(float(stability), grasp_result.count, float(a_score.score)),
            grasp_result=grasp_result,
            grasp=accepted[0],
            direction=np.asarray(directions.best_direction, dtype=float),
            a_case=a_score.case,
            a_region=a_score.region,
        )
        cache[key] = step
        return step

    started = perf_counter()
    candidates: list[SequenceCandidate] = []
    branch_records = []
    for order in permutations(sorted(movable_ids())):
        steps: list[EvaluatedStep] = []
        failure = None
        for index, part_id in enumerate(order):
            counters["transition_attempts"] += 1
            result = evaluate(order[:index], part_id)
            if isinstance(result, str):
                failure = result
                break
            steps.append(result)
        if failure is None:
            score = sequence_quality(tuple(step.quality for step in steps))
            candidates.append(SequenceCandidate(order, tuple(steps), score))
            branch_records.append({"order": order, "status": "valid", "score": score})
        else:
            branch_records.append({"order": order, "status": "pruned", "reason": failure})
    if not candidates:
        raise RuntimeError(f"No valid assembly order; rejection summary: {rejected}")
    candidates.sort(key=lambda item: (-item.score, item.order))
    best = candidates[0]
    report = {
        "method": "exhaustive_single_arm_SGA",
        "gripper": "PantheraGripperPort",
        "direction_count": direction_count,
        "best_order": best.order,
        "best_score": best.score,
        "valid_orders": len(candidates),
        "total_orders": 24,
        "rejections": rejected,
        **counters,
        "elapsed_s": perf_counter() - started,
        "steps": [
            {
                "step": index,
                "part_id": step.part_id,
                "S": step.quality.stability,
                "G": step.quality.graspability,
                "A": step.quality.assemblability,
                "A_case": step.a_case,
                "A_region": step.a_region,
                "removal_direction": step.direction.tolist(),
                "insertion_direction": (-step.direction).tolist(),
                "grasp_id": step.grasp_result.accepted_ids[0],
                "jaw_width_m": step.grasp.jaw_width,
            }
            for index, step in enumerate(best.steps, 1)
        ],
        "branches": branch_records,
    }
    return best, report


def search_sequence_reverse(*, direction_count: int = 64) -> tuple[SequenceCandidate, dict]:
    """Search by removal, pruning immediately when the remainder is unstable.

    If ``current - {part}`` is stable, removing ``part`` represents a forward
    installation step whose pre-state can stand without the single robot arm.
    G, A and the scored S are evaluated in ``current`` because that is the
    state immediately after the corresponding forward placement.
    """
    assembly = build_bridge()
    goal = assembly.initial_state()
    fixed = frozenset(part.part_id for part in assembly.parts if part.fixed)
    gripper = PantheraGripperPort()
    grasp_analyzer = GraspabilityAnalyzer(
        assembly, gripper, config=GraspabilityConfig(max_grasps=8)
    )
    sequence_config = SequenceConfig(max_auxiliary_resources=0)
    sequence = SequenceEvaluator(assembly, sequence_config)
    sweep_config = StabilitySweepConfig(
        backend="numpy",
        direction_count=direction_count,
        batch_size=min(direction_count, 64),
        stability=sequence_config.stability,
    )
    direction_config = DirectionConfig(method="socp")
    state_cache = {}
    graph_cache = {}
    equilibrium_cache = {}
    stability_cache = {}
    transition_cache: dict[tuple[frozenset[str], str], EvaluatedStep | str] = {}
    counters = {
        "nodes": 0,
        "transition_attempts": 0,
        "equilibrium_queries": 0,
        "stability_sweeps": 0,
        "G_queries": 0,
        "A_queries": 0,
        "pruned_unstable_remainders": 0,
    }

    def state_of(movable: frozenset[str]):
        if movable not in state_cache:
            present = fixed | movable
            state_cache[movable] = AssemblyState(
                {pid: tf for pid, tf in goal.poses.items() if pid in present}
            )
        return state_cache[movable]

    def graph_of(movable: frozenset[str]):
        if movable not in graph_cache:
            graph_cache[movable] = sequence.graph(state_of(movable))
        return graph_cache[movable]

    def equilibrium_of(movable: frozenset[str]):
        if movable not in equilibrium_cache:
            counters["equilibrium_queries"] += 1
            equilibrium_cache[movable] = sequence.equilibrium(state_of(movable), ())
        return equilibrium_cache[movable]

    def is_stable(movable: frozenset[str]) -> bool:
        # The all-fixed base is a valid terminal remainder by construction.
        return not movable or equilibrium_of(movable).status == "feasible"

    def stability_score(movable: frozenset[str]) -> float | None:
        if movable not in stability_cache:
            equilibrium = equilibrium_of(movable)
            if equilibrium.status != "feasible":
                stability_cache[movable] = 0.0 if equilibrium.status == "infeasible" else None
            else:
                counters["stability_sweeps"] += 1
                sweep = DirectionalStabilityAnalyzer(
                    assembly, state_of(movable), graph_of(movable), config=sweep_config
                ).analyze()
                stability_cache[movable] = sweep.sampled_minimum_load_factor
        return stability_cache[movable]

    def evaluate_removal(current: frozenset[str], part_id: str) -> EvaluatedStep | str:
        key = (current, part_id)
        if key in transition_cache:
            return transition_cache[key]
        remainder = current - {part_id}
        # This is the user's central reverse-search prune.  It is deliberately
        # first, before the more expensive Panthera G and direction A queries.
        if not is_stable(remainder):
            counters["pruned_unstable_remainders"] += 1
            transition_cache[key] = "S_remainder_zero"
            return transition_cache[key]

        stability = stability_score(current)
        if stability is None:
            transition_cache[key] = "S_unknown"
            return transition_cache[key]
        if stability <= 1e-8:
            transition_cache[key] = "S_zero"
            return transition_cache[key]

        state = state_of(current)
        counters["G_queries"] += 1
        grasp_result = grasp_analyzer.analyze(state, part_id)
        if grasp_result.count is None:
            transition_cache[key] = "G_unknown"
            return transition_cache[key]
        if grasp_result.count == 0:
            transition_cache[key] = "G_zero"
            return transition_cache[key]

        counters["A_queries"] += 1
        directions = assembly_directions(
            graph_of(current), (part_id,), config=direction_config
        )
        a_score = score_assemblability(directions)
        if a_score.score in (None, 0) or directions.best_direction is None:
            transition_cache[key] = "A_unknown" if a_score.score is None else "A_zero"
            return transition_cache[key]

        accepted = grasp_analyzer.accepted_grasps(grasp_result)
        step = EvaluatedStep(
            part_id=part_id,
            installed=tuple(sorted(current)),
            quality=StepQuality(float(stability), grasp_result.count, float(a_score.score)),
            grasp_result=grasp_result,
            grasp=accepted[0],
            direction=np.asarray(directions.best_direction, dtype=float),
            a_case=a_score.case,
            a_region=a_score.region,
        )
        transition_cache[key] = step
        return step

    started = perf_counter()
    candidates: list[SequenceCandidate] = []
    trace = []

    def visit(current: frozenset[str], removal_steps: tuple[EvaluatedStep, ...]) -> None:
        counters["nodes"] += 1
        if not current:
            forward_steps = tuple(reversed(removal_steps))
            order = tuple(step.part_id for step in forward_steps)
            score = sequence_quality(tuple(step.quality for step in forward_steps))
            candidates.append(SequenceCandidate(order, forward_steps, score))
            trace.append({
                "removal_order": tuple(step.part_id for step in removal_steps),
                "assembly_order": order,
                "status": "valid",
                "score": score,
            })
            return
        for part_id in sorted(current):
            counters["transition_attempts"] += 1
            result = evaluate_removal(current, part_id)
            if isinstance(result, str):
                trace.append({
                    "current": tuple(sorted(current)),
                    "try_remove": part_id,
                    "status": "pruned",
                    "reason": result,
                })
                continue
            visit(current - {part_id}, removal_steps + (result,))

    visit(frozenset(movable_ids()), ())
    if not candidates:
        raise RuntimeError(f"Reverse search found no valid assembly order: {counters}")
    candidates.sort(key=lambda item: (-item.score, item.order))
    best = candidates[0]
    report = {
        "method": "reverse_disassembly_single_arm_SGA",
        "gripper": "PantheraGripperPort",
        "direction_count": direction_count,
        "best_order": best.order,
        "best_removal_order": tuple(reversed(best.order)),
        "best_score": best.score,
        "valid_orders": len(candidates),
        "total_orders": 24,
        **counters,
        "elapsed_s": perf_counter() - started,
        "steps": [
            {
                "step": index,
                "part_id": step.part_id,
                "S": step.quality.stability,
                "G": step.quality.graspability,
                "A": step.quality.assemblability,
                "A_case": step.a_case,
                "A_region": step.a_region,
                "removal_direction": step.direction.tolist(),
                "insertion_direction": (-step.direction).tolist(),
                "grasp_id": step.grasp_result.accepted_ids[0],
                "jaw_width_m": step.grasp.jaw_width,
            }
            for index, step in enumerate(best.steps, 1)
        ],
        "trace": trace,
    }
    return best, report


PART_COLORS = {
    "base": "#59636e",
    "post_l": "#e9a23b",
    "post_r": "#f2c14e",
    "beam": "#2a9d8f",
    "load": "#5875d5",
}


def _world_triangles(part, world_tf: np.ndarray) -> np.ndarray:
    vertices = part.geometry.vertices @ world_tf[:3, :3].T + world_tf[:3, 3]
    return vertices[part.geometry.faces]


def render_sequence(best: SequenceCandidate, output: Path) -> Path:
    """Render four assembly states with the real Panthera gripper meshes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    assembly = build_bridge()
    parts = {part.part_id: part for part in assembly.parts}
    gripper = PantheraGripperPort()
    figure = plt.figure(figsize=(15.5, 8.8), facecolor="#f5f7f8")
    axes = [figure.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    installed: set[str] = set()

    for index, (ax, step) in enumerate(zip(axes, best.steps), 1):
        installed.add(step.part_id)
        # Reference table, cropped around the task so the assembly remains readable.
        table = np.array([
            [-0.25, -0.20, -0.002], [0.25, -0.20, -0.002],
            [0.25, 0.20, -0.002], [-0.25, 0.20, -0.002],
        ])
        ax.add_collection3d(Poly3DCollection(
            [table], facecolor="#dce3e7", edgecolor="#aab4ba", alpha=0.38
        ))
        for pid in ("base", "post_l", "post_r", "beam", "load"):
            part = parts[pid]
            if pid not in installed and pid != "base":
                # Faint target geometry shows what remains without pretending it is installed.
                alpha, face = 0.035, "#aeb8bf"
            else:
                alpha = 1.0 if pid == step.part_id else 0.82
                face = PART_COLORS[pid]
            poly = Poly3DCollection(
                _world_triangles(part, part.assembled_tf),
                facecolor=face,
                edgecolor="#263238" if alpha > 0.1 else "#96a2aa",
                linewidth=0.32,
                alpha=alpha,
            )
            ax.add_collection3d(poly)

        # Place a fresh Panthera model at the accepted target grasp, then map it
        # from the part-local frame to world coordinates.
        moving = parts[step.part_id]
        grasp = step.grasp
        gripper.grip_at(grasp.pose[:3, 3], grasp.pose[:3, :3], grasp.jaw_width)
        for link in gripper.link_parts():
            tf = moving.assembled_tf @ link.assembled_tf
            ax.add_collection3d(Poly3DCollection(
                _world_triangles(link, tf),
                facecolor="#65a9d9",
                edgecolor="#244b66",
                linewidth=0.20,
                alpha=0.58,
            ))

        # A is solved in the removal direction; assembly inserts in the reverse.
        direction = -step.direction / np.linalg.norm(step.direction)
        target = moving.assembled_tf[:3, 3]
        start = target - direction * 0.075
        ax.quiver(
            *start, *direction, length=0.065, normalize=True,
            color="#c93c37", linewidth=2.2, arrow_length_ratio=0.22,
        )
        q = step.quality
        ax.set_title(
            f"Step {index}: {step.part_id}\nS={q.stability:.3f}   G={q.graspability}   A={q.assemblability:g}",
            fontsize=12, fontweight="bold", pad=8,
        )
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.20, 0.20)
        ax.set_zlim(-0.01, 0.34)
        ax.set_box_aspect((1.0, 0.8, 0.72))
        ax.view_init(elev=22, azim=-58)
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.grid(True, alpha=0.18)

    order_text = "  ->  ".join(best.order)
    figure.suptitle(
        "Panthera single-arm S/G/A assembly-sequence result",
        fontsize=17, fontweight="bold", y=0.985,
    )
    figure.text(
        0.5, 0.945, f"Best order: {order_text}    sequence score: {best.score:.4f}",
        ha="center", fontsize=11, color="#37474f",
    )
    legend = [
        Patch(facecolor=PART_COLORS[pid], edgecolor="#263238", label=pid)
        for pid in ("post_l", "post_r", "beam", "load")
    ] + [
        Patch(facecolor="#65a9d9", edgecolor="#244b66", alpha=0.58,
              label="Panthera gripper"),
        Line2D([0], [0], color="#c93c37", lw=2.2, label="insertion direction"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=6, frameon=False,
                  bbox_to_anchor=(0.5, 0.012), fontsize=9)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.09,
                           hspace=0.20, wspace=0.03)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-count", type=int, default=64)
    parser.add_argument("--search", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    search = search_sequence_reverse if args.search == "reverse" else search_sequence
    best, report = search(direction_count=args.direction_count)
    image = render_sequence(best, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST_ORDER", " -> ".join(best.order))
    print("BEST_SCORE", f"{best.score:.9f}")
    for index, step in enumerate(best.steps, 1):
        q = step.quality
        print(index, step.part_id, f"S={q.stability:.9f}", f"G={q.graspability}",
              f"A={q.assemblability:g}", f"jaw={step.grasp.jaw_width:.6f}m")
    print("VALID_ORDERS", report["valid_orders"], "/", report["total_orders"])
    print("SEARCH", args.search, "ELAPSED", f"{report['elapsed_s']:.9f}s")
    print("IMAGE", image)
    print("JSON", args.json.resolve())


if __name__ == "__main__":
    main()
