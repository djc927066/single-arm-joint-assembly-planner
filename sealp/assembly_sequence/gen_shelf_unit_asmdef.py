#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gen_shelf_unit_asmdef.py

生成不带底板的 shelf_unit.asmdef。
装配顺序：side_l → shelf_m → shelf_t → side_r

Goal 位姿在 **fixture 基座标** 下定义（parent=fixture），与 ``mesh_frames`` 一致。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef
from sealp.assets.models.shelf_unit.mesh_frames import (
    GOAL_REL_ROTMAT,
    PART_IDS,
    SHELF_FRAME,
    SHELF_STL,
    SIDE_FRAME,
    SIDE_STL,
    compute_goal_poses,
)


def generate() -> AssemblyDef:
    goal_rel, goal_rot = compute_goal_poses(SIDE_FRAME, SHELF_FRAME)

    asm = AssemblyDef(
        name="ShelfUnit",
        description=(
            "4-part compact shelf unit without bottom board. "
            f"Side panel local z∈[{SIDE_FRAME.lo[2]:.3f},{SIDE_FRAME.hi[2]:.3f}] m "
            "(bottom at z=0 in mesh frame). "
            "Assembly order: side_l -> shelf_m -> shelf_t -> side_r."
        ),
    )

    asm.add_model("shelf_model", SHELF_STL)
    asm.add_model("side_model", SIDE_STL)

    asm.add_part(PartDef(part_id="side_l", name="Left Side Panel", model="side_model", mass=0.5))
    asm.add_part(PartDef(part_id="shelf_m", name="Middle Shelf", model="shelf_model", mass=1.0))
    asm.add_part(PartDef(part_id="shelf_t", name="Top Shelf", model="shelf_model", mass=1.0))
    asm.add_part(PartDef(part_id="side_r", name="Right Side Panel", model="side_model", mass=0.5))

    asm.add_symmetry_group("sides", ["side_l", "side_r"])
    asm.add_symmetry_group("shelves", ["shelf_m", "shelf_t"])

    notes = {
        "side_l": "Install left side panel; mesh bottom (local z=0) on fixture support.",
        "shelf_m": "Insert middle shelf at mid height between side panels.",
        "shelf_t": "Place top shelf flush on side panel tops.",
        "side_r": "Install right side panel last.",
    }
    deps_chain = {
        "side_l": [],
        "shelf_m": [0],
        "shelf_t": [0, 1],
        "side_r": [0, 1, 2],
    }
    for step_id, pid in enumerate(PART_IDS):
        asm.add_step(StepDef(
            step_id=step_id,
            part_id=pid,
            parent_id="fixture",
            rel_pos=goal_rel[pid].copy(),
            rel_rotmat=goal_rot[pid].copy(),
            deps=deps_chain[pid],
            notes=notes[pid],
        ))

    errors = asm.validate(strict=False)
    if errors:
        print("Validation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation passed.")

    print("\nMesh frames (local AABB):")
    for label, frame in [("side_panel", SIDE_FRAME), ("shelf", SHELF_FRAME)]:
        print(f"  {label}: lo={frame.lo.round(4).tolist()} hi={frame.hi.round(4).tolist()}")
    print("\nGoal rel_pos (fixture base):")
    for pid in PART_IDS:
        print(f"  {pid:8s} {goal_rel[pid].round(4).tolist()}")

    return asm


def main() -> None:
    asm = generate()
    out_dir = os.path.join(os.path.dirname(__file__), "_demo_output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "shelf_unit.asmdef")
    asm.save(path)
    print(f"\nSaved: {path}\n")
    print(asm.summary())


if __name__ == "__main__":
    main()
