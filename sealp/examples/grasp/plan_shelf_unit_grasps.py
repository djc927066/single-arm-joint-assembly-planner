#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Grasp Planning — Shelf-Unit (side_panel + shelf)
====================================================

Generates antipodal grasp pickles for the ``shelf_unit`` assembly task,
mirroring ``planning.py`` but on the procedurally generated boxes:

    side_panel.stl  (0.180 × 0.018 × 0.300 m)
    shelf.stl       (plain shelf, 0.180 × 0.300 × 0.015 m)

Output pickles:

    demo_shelf_unit-side_grasps.pickle
    demo_shelf_unit-shelf_grasps.pickle

Run::

    python -m sealp.examples.grasp.plan_shelf_unit_grasps [--no-vis]

This script is *task-local*: it never touches ``demo_yuanchair-*.pickle``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg

from sealp.examples.grasp.planning import plan_grasps, visualize_grasps


_PART_ASSETS = (
    ("side_panel", "side"),
    ("shelf",      "shelf"),
)


def _resolve_part_mesh(part_name: str) -> str:
    return os.path.abspath(os.path.join(
        _PROJ_ROOT, "sealp", "assets", "models", "shelf_unit",
        f"{part_name}.stl"))


def main(visualize: bool = True) -> None:
    base = None
    if visualize:
        base = wd.World(cam_pos=rm.vec(0.6, 0.6, 0.5),
                        lookat_pos=rm.vec(0, 0, 0))
        mgm.gen_frame(ax_length=0.25).attach_to(base)

    out_dir = os.path.join(_HERE, "_output")
    os.makedirs(out_dir, exist_ok=True)

    last_obj, last_grasps, last_gripper = None, None, None

    for part_name, role in _PART_ASSETS:
        mesh_path = _resolve_part_mesh(part_name)
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(
                f"Mesh missing: {mesh_path}. Run `python -m "
                f"sealp.assets.models.shelf_unit.gen_meshes` first.")

        obj_cmodel = mcm.CollisionModel(mesh_path)
        obj_cmodel.rgba = np.array([0.72, 0.55, 0.36, 1.0])

        print("=" * 60)
        print(f"Grasp Planning — Panthera Gripper "
              f"[shelf_unit / {part_name} ({role})]")
        print("=" * 60)
        # 层板：竖 staging + 横 goal 需要更多姿态样本；侧板保持默认。
        if role == "shelf":
            max_samples, rot_deg = 200, 15
        else:
            max_samples, rot_deg = 80, 30
        grasp_collection, gripper = plan_grasps(
            obj_cmodel,
            max_samples=max_samples,
            rotation_interval=rm.radians(rot_deg),
        )
        print(f"  Planned {len(grasp_collection)} grasps "
              f"(max_samples={max_samples}, rot_step={rot_deg}°).")

        save_path = os.path.join(
            out_dir, f"demo_shelf_unit-{role}_grasps.pickle")
        grasp_collection.save_to_disk(file_name=save_path)
        print(f"  Saved to: {save_path}")

        last_obj, last_grasps, last_gripper = (
            obj_cmodel, grasp_collection, gripper)

    if visualize and base is not None and last_obj is not None:
        print("  Showing grasps of the last part...")
        visualize_grasps(base, last_obj, last_grasps, last_gripper,
                         max_show=60)
        print("=" * 60)
        print("Press ESC to close.")
        base.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip Panda3D viewer; only generate pickles.")
    args = parser.parse_args()
    main(visualize=not args.no_vis)
