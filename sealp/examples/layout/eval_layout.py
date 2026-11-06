"""
Layout Feasibility Evaluation — 3D Visualization
===================================================

Evaluate a single workspace layout for the YuanChair assembly
and visualize the results in Panda3D.

Visualization shows:
  - Robot at the layout's base position (translucent mesh)
  - Parts at staging (pick) positions (coloured)
  - Assembly goal poses (translucent ghosts)
  - Per-step reachability indicators: green sphere = feasible,
    red sphere = infeasible
  - Coordinate frames at each staging/goal pose

Usage::

    python -m sealp.examples.layout.eval_layout

Prerequisites:
    - Run ``python -m sealp.assembly_sequence.gen_yuanchair_asmdef``
      to generate the ``.asmdef`` file (if not already present).
"""

import os
import sys
import numpy as np

from wrs import wd, rm, mgm, mcm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sealp.assembly_sequence import AssemblyDef, TaskPlan, StepParams
from sealp.layout import WorkspaceLayout, evaluate_layout, compute_metrics


# ══════════════════════════════════════════════════════════════
#  Helper — plan grasps for all models in the assembly
# ══════════════════════════════════════════════════════════════
def build_grasp_cache(assembly_def):
    """Plan grasps for each unique model in the assembly.

    Returns ``{model_alias: grasp_collection}``.
    """
    from sealp.examples.grasp.planning import plan_grasps

    cache = {}
    for alias, path in assembly_def.models.items():
        if alias in cache:
            continue
        print(f"  Planning grasps for '{alias}' ...")
        obj = mcm.CollisionModel(initor=path)
        gc, _ = plan_grasps(obj, max_samples=50)
        cache[alias] = gc
        print(f"    → {len(gc)} grasps")
    return cache


# ══════════════════════════════════════════════════════════════
#  Visualize feasibility report in Panda3D
# ══════════════════════════════════════════════════════════════
def visualize_report(base, layout, assembly_def, report):
    """Draw per-step feasibility indicators in the 3D scene.

    Green spheres at pick/place poses that are feasible,
    red spheres at infeasible poses.
    """
    RADIUS = 0.012

    fixture_pos = layout.assembly_station_pos
    fixture_rotmat = layout.assembly_station_rotmat
    world_poses = assembly_def.compute_world_poses(
        fixture_pos=fixture_pos, fixture_rotmat=fixture_rotmat)

    for sf in report.steps:
        pid = sf.part_id

        # ── PICK indicator ──────────────────────────────────
        staging = layout.get_staging(pid)
        if staging is not None:
            pick_pos, _ = staging
            color = np.array([0.15, 0.85, 0.15, 0.85]) if sf.pick_collision_free \
                else np.array([0.90, 0.15, 0.15, 0.85])
            s = mcm.gen_sphere(radius=RADIUS)
            s.pos = pick_pos + np.array([0, 0, 0.04])
            s.rgba = color
            s.attach_to(base)

        # ── PLACE indicator ─────────────────────────────────
        if pid in world_poses:
            place_pos, _ = world_poses[pid]
            color = np.array([0.15, 0.85, 0.15, 0.85]) if sf.place_collision_free \
                else np.array([0.90, 0.15, 0.15, 0.85])
            s = mcm.gen_sphere(radius=RADIUS)
            s.pos = place_pos + np.array([0, 0, 0.04])
            s.rgba = color
            s.attach_to(base)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    import wrs.robot_sim.robots.piper.piper_single_arm as psa

    # ------------------------------------------------------------------
    # 1. Scene
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=[1.2, 0.8, 1.0], lookat_pos=[0.25, 0, 0.05])
    mgm.gen_frame(ax_length=0.15).attach_to(base)

    # Ground
    ground = mcm.gen_box(
        xyz_lengths=rm.vec(2, 2, 0.01),
        rgb=rm.vec(0.75, 0.75, 0.75), alpha=1)
    ground.pos = np.array([0.3, 0, -0.005])
    ground.attach_to(base)

    # ------------------------------------------------------------------
    # 2. Load assembly definition
    # ------------------------------------------------------------------
    asmdef_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "assembly_sequence", "_demo_output")
    asmdef_path = os.path.abspath(
        os.path.join(asmdef_dir, "yuanchair.asmdef"))

    if not os.path.isfile(asmdef_path):
        print(f"Assembly definition not found, generating...")
        from sealp.assembly_sequence.gen_yuanchair_asmdef import main as gen_asm
        gen_asm()

    asm = AssemblyDef.load(asmdef_path)
    print(f"Loaded: {asm.name} ({asm.n_parts} parts, {asm.n_steps} steps)")

    # ------------------------------------------------------------------
    # 3. Define a layout to evaluate
    # ------------------------------------------------------------------
    layout = WorkspaceLayout(
        robot_base_pos=np.array([0.0, 0.0, 0.0]),
        robot_base_rotmat=np.eye(3),
        assembly_station_pos=np.array([0.0, 0.0, 0.0]),
        assembly_station_rotmat=np.eye(3),
        name="eval_demo_layout",
    )
    # Staging positions (same as sequence_execution.py)
    layout.set_staging("seat",   np.array([0.30,  0.00, 0.00]))
    layout.set_staging("leg_fl", np.array([0.10,  0.20, 0.00]))
    layout.set_staging("leg_fr", np.array([0.05, -0.24, 0.00]))
    layout.set_staging("leg_bl", np.array([0.25,  0.24, 0.00]))
    layout.set_staging("leg_br", np.array([0.25, -0.24, 0.00]))
    print(f"\n{layout.summary()}")

    # ------------------------------------------------------------------
    # 4. Visualize parts at staging (pick) positions
    # ------------------------------------------------------------------
    staging_colors = {
        "seat":   np.array([0.9, 0.6, 0.3, 0.7]),
        "leg_fl": np.array([0.3, 0.7, 0.3, 0.7]),
        "leg_fr": np.array([0.3, 0.3, 0.8, 0.7]),
        "leg_bl": np.array([0.8, 0.3, 0.3, 0.7]),
        "leg_br": np.array([0.7, 0.3, 0.7, 0.7]),
    }
    for pid in asm.part_ids:
        st = layout.get_staging(pid)
        if st is None:
            continue
        pos, rotmat = st
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            staged = mcm.CollisionModel(initor=mp)
            staged.pos = pos
            staged.rotmat = rotmat
            staged.rgba = staging_colors.get(pid, np.array([0.5, 0.5, 0.5, 0.7]))
            staged.attach_to(base)
            mgm.gen_frame(pos=pos, ax_length=0.025).attach_to(base)

    # ------------------------------------------------------------------
    # 5. Visualize assembly goal poses (translucent ghosts)
    # ------------------------------------------------------------------
    world_poses = asm.compute_world_poses(
        fixture_pos=layout.assembly_station_pos,
        fixture_rotmat=layout.assembly_station_rotmat)
    for pid in asm.part_ids:
        if pid not in world_poses:
            continue
        gp, gr = world_poses[pid]
        mp = asm.model_path(pid)
        if os.path.isfile(mp):
            ghost = mcm.CollisionModel(initor=mp)
            ghost.pos = gp
            ghost.rotmat = gr
            ghost.alpha = 0.15
            ghost.attach_to(base)

    # ------------------------------------------------------------------
    # 6. Robot
    # ------------------------------------------------------------------
    robot = psa.PiperSglArm(
        pos=layout.robot_base_pos,
        rotmat=layout.robot_base_rotmat,
        enable_cc=True)
    robot.gen_meshmodel(alpha=0.3).attach_to(base)

    # ------------------------------------------------------------------
    # 7. Build grasp cache
    # ------------------------------------------------------------------
    print("\nPlanning grasps...")
    grasp_cache = build_grasp_cache(asm)

    # ------------------------------------------------------------------
    # 8. Evaluate layout feasibility
    # ------------------------------------------------------------------
    print("\nEvaluating layout feasibility...")
    report = evaluate_layout(
        layout=layout,
        assembly_def=asm,
        robot=robot,
        grasp_cache=grasp_cache,
        obstacle_list=[ground],
        max_grasps_per_step=20,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 9. Compute metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(layout, report)
    print(f"\n{metrics.summary()}")

    # ------------------------------------------------------------------
    # 10. Visualize feasibility markers in the 3D scene
    # ------------------------------------------------------------------
    visualize_report(base, layout, asm, report)

    # ------------------------------------------------------------------
    # 11. Save the layout
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)
    layout_path = os.path.join(out_dir, "eval_demo.layout")
    layout.save(layout_path)
    print(f"\nLayout saved to: {layout_path}")

    base.run()


if __name__ == "__main__":
    main()
