"""Triangular frame with two short posts on top: triangular_frame6 + cap_AB + cap_BC, 8 movable parts.

    & $py -X utf8 -m scenes.triangular_frame8.build_model

The six frame parts are those of triangular_frame6, unchanged. Each cap stands at the middle of its
beam, along the beam, so the beam must be in first and cannot leave while its cap is on it.
Output: assets/meshes/*.stl, task.json, validation.json, preview.png. Grasps: lazy_cost.gen_grasps.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy / Panda3D on Windows (c10.dll)

import numpy as np  # noqa: E402

from asmcore.model import Assembly, MatingRelation, Part  # noqa: E402
from jbls.asset_files import prepare_part_stls  # noqa: E402
from scenes.triangular_frame6 import build_model as frame  # noqa: E402
from tasks.bridge import box_mesh  # noqa: E402

HERE = Path(__file__).resolve().parent
CAPS = {"cap_AB": ("beam_AB", (.040, .026, .044)), "cap_BC": ("beam_BC", (.036, .024, .044))}    # beam, size
COLORS = {"post": (.36, .45, .78), "beam": (.92, .60, .18), "cap": (.85, .33, .27), "base": (.66, .70, .74)}


def build_assembly():
    asm, meta, ws = frame.build_assembly()
    by = {p.part_id: p for p in asm.parts}
    parts, mating = list(asm.parts), list(asm.mating_relations)
    for pid, (beam, size) in CAPS.items():
        tf = by[beam].assembled_tf.copy()
        tf[2, 3] += frame.BEAM_HEIGHT / 2 + size[2] / 2
        parts.append(Part(pid, box_mesh(*size), tf, mass_kg=frame.DENSITY * float(np.prod(size)),
                          com_local_m=np.zeros(3), friction=.5))
        meta[pid] = dict(name="短立柱 " + beam[-2:], size_m=list(size), required_supports=[beam],
                         insertion_axis_world=[0, 0, -1], removal_axis_world=[0, 0, 1],
                         connection="gravity-supported planar contact; no latch/fastener")
        mating.append(MatingRelation(beam, pid, "planar_support", dict(insertion_axis_world=[0, 0, -1])))
    prov = dict(asm.provenance, task="triangular_frame8", six_movable_parts=False, eight_movable_parts=True)
    return Assembly(tuple(parts), mating_relations=tuple(mating), provenance=prov), meta, ws


def preview(asm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 6.4), facecolor="white")
    for i, (title, gap) in enumerate((("Assembled", 0.0), ("Exploded", 1.0))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        tris, cols = [], []                      # one collection: matplotlib depth-sorts inside it only
        for p in asm.parts:
            if p.part_id == "table":
                continue
            v = frame.world_vertices(p) - frame.SITE
            kind = p.part_id.split("_")[0]
            t = (v + [0, 0, gap * {"post": .03, "beam": .09, "cap": .16}.get(kind, 0)])[p.geometry.faces] * 1000
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            shade = .62 + .38 * np.clip(n @ np.array([.35, -.45, .82]), 0, 1)
            tris.append(t)
            cols.append(shade[:, None] * np.array(COLORS[kind]))
        ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolors=np.concatenate(cols), edgecolors="none"))
        ax.set_xlim(-130, 130), ax.set_ylim(-130, 130), ax.set_zlim(0, 340 if gap else 180)
        ax.set_box_aspect((1, 1, 1.15 if gap else .7))
        ax.view_init(elev=28, azim=-62)
        ax.set_title(title, pad=2)
        ax.set_xlabel("X (mm)"), ax.set_ylabel("Y (mm)"), ax.set_zlabel("Z (mm)")
    fig.subplots_adjust(left=.02, right=.98, top=.93, bottom=.04, wspace=.05)
    fig.savefig(path, dpi=130, facecolor="white")


def validate(asm, meta):
    """static equilibrium (asmcore LP) of every subset, and the orders that stay stable at every step."""
    from asmcore.model import AssemblyState
    from asmcore.planning.sequence import SequenceConfig, SequenceEvaluator
    ids, by = tuple(meta), {p.part_id: p for p in asm.parts}
    ev = SequenceEvaluator(asm, SequenceConfig(max_auxiliary_resources=0))
    fixed = {p.part_id: p.assembled_tf for p in asm.parts if p.fixed}
    stable, mismatches = {}, []
    for bits in range(1 << len(ids)):
        present = tuple(p for i, p in enumerate(ids) if bits >> i & 1)
        r = ev.equilibrium(AssemblyState(dict(fixed, **{p: by[p].assembled_tf for p in present})), ())
        stable[bits] = r.status == "feasible"
        expected = all(q == "base" or q in present for p in present for q in meta[p]["required_supports"])
        if stable[bits] != expected:
            mismatches.append(dict(parts=present, expected=expected, status=r.status))
    orders = []

    def rec(bits, o):
        if len(o) == len(ids):
            orders.append(o)
        for i, p in enumerate(ids):
            if not bits >> i & 1 and stable[bits | 1 << i]:
                rec(bits | 1 << i, o + (p,))
    rec(0, ())
    return dict(subset_count=len(stable), nominal_stable_subset_count=sum(stable.values()),
                support_rule_mismatches=mismatches, nominal_static_order_count=len(orders),
                nominal_static_orders=orders, full_robot_motion_verified=False, safe_to_execute=False)


def main():
    asm, meta, ws = build_assembly()
    loaded, _ = prepare_part_stls(asm.parts, HERE / "assets")
    asm = replace(asm, parts=tuple(loaded))
    val = validate(asm, meta)
    task = dict(schema="assembly_benchmark/1", name="triangular_frame8", units="metre",
                fixed_base_position_m=frame.SITE.tolist(), movable_ids=list(meta), friction=.5, table=ws.metadata(),
                parts={p.part_id: dict(fixed=p.fixed, mass_kg=p.mass_kg, com_local_m=p.com_local_m.tolist(),
                                       goal_tf=p.assembled_tf.tolist(), **meta.get(p.part_id, {})) for p in asm.parts},
                full_robot_motion_verified=False, safe_to_execute=False)
    (HERE / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    preview(asm, HERE / "preview.png")
    print("orders", val["nominal_static_order_count"], "stable subsets", val["nominal_stable_subset_count"],
          "support mismatches", len(val["support_rule_mismatches"]), flush=True)


if __name__ == "__main__":
    main()
