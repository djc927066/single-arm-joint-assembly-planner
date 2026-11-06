"""Finite-domain joint backward beam search, independent of WRS/cuRobo.

Objective (metres): home -> x[first] -> goal[first] -> x[next] -> ... -> goal[last].
The left incoming edge of a suffix is deliberately unpaid until its predecessor
is chosen. No return home, tool offset, local lift, rotation or execution time.
"""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


@dataclass(frozen=True)
class Placement:
    key: str
    pose: np.ndarray = field(compare=False, repr=False)

    @property
    def position(self):
        return self.pose[:3, 3]


@dataclass
class Node:
    order: tuple = ()                         # forward order of the assigned suffix
    layout: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    cost: float = 0.0                         # carry + internal empty edges


def cost_breakdown(order, layout, goals, home):
    rows, previous = [], home
    for pid in order:
        x = layout[pid].position
        empty, loaded = distance(previous, x), distance(x, goals[pid])
        rows.append(dict(part=pid, start=np.asarray(previous).tolist(),
                         pick=x.tolist(), goal=np.asarray(goals[pid]).tolist(),
                         empty_m=empty, loaded_m=loaded, total_m=empty + loaded))
        previous = goals[pid]
    return rows


def joint_search(parts, domains, goals, home, structure_check, placement_check,
                 *, beam_width=8, fixed_order=None, progress=None):
    """Callbacks return (record, reason); record=None rejects this branch only.

    No cross-scene failure cache. The estimate sums min carrying distances for
    remaining parts and drops their nonnegative empty edges. It is a relaxation
    of THIS proxy objective only. Beam truncation removes global guarantees.
    """
    parts = tuple(parts)
    if not parts or len(set(parts)) != len(parts) or beam_width < 1:
        raise ValueError("unique nonempty parts and positive beam_width required")
    if fixed_order is not None and (len(fixed_order) != len(parts) or set(fixed_order) != set(parts)):
        raise ValueError("fixed_order must be a permutation of parts")
    if any(not domains[p] for p in parts):
        return [], dict(status="empty_domain", layers=[], wall_s=0.0,
                        global_optimality=False, full_motion_verified=False)
    min_carry = {p: min(distance(x.position, goals[p]) for x in domains[p]) for p in parts}
    started, beam, layers = perf_counter(), [Node()], []
    for depth in range(len(parts)):
        children, seen = [], set()
        stats = dict(depth=depth + 1, parents=len(beam), structure_attempts=0,
                     placement_attempts=0, accepted=0, rejected={})
        def reject(reason):
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
        for parent in beam:
            remaining = frozenset(parts).difference(parent.layout)
            choices = (fixed_order[-depth - 1],) if fixed_order is not None else sorted(remaining)
            for pid in choices:
                stats["structure_attempts"] += 1
                structure, reason = structure_check(remaining, pid)
                if structure is None:
                    reject(reason)
                    continue
                for place in domains[pid]:
                    stats["placement_attempts"] += 1
                    witness, reason = placement_check(parent, remaining, pid, place, structure)
                    if witness is None:
                        reject(reason)
                        continue
                    layout = dict(parent.layout, **{pid: place})
                    order = (pid,) + parent.order
                    cost = parent.cost + distance(place.position, goals[pid])
                    if parent.order:
                        cost += distance(goals[pid], parent.layout[parent.order[0]].position)
                    if len(order) == len(parts):
                        cost += distance(home, place.position)
                    signature = (order, tuple((p, layout[p].key) for p in order))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    child = Node(order, layout, dict(parent.steps, **{pid: witness}), cost)
                    estimate = cost + sum(min_carry[p] for p in remaining - {pid})
                    children.append((estimate, signature, child))
                    stats["accepted"] += 1
        children.sort(key=lambda row: (row[0], row[1]))
        beam = [row[2] for row in children[:beam_width]]
        stats.update(retained=len(beam), best_score_m=children[0][0] if children else None,
                     elapsed_s=perf_counter() - started)
        layers.append(stats)
        if progress:
            progress(stats)
        if not beam:
            break
    complete = [n for n in beam if len(n.order) == len(parts)]
    return complete, dict(status="candidates_found" if complete else "no_candidate_in_budget",
                          wall_s=perf_counter() - started, layers=layers,
                          global_optimality=False, full_motion_verified=False)
