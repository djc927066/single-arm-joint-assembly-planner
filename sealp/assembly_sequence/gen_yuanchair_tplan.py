"""
Generate YuanChair Task Plan (.tplan)
======================================

Creates a ``.tplan`` file that references the ``yuanchair.asmdef`` and
adds task-specific staging positions and execution params from
``pick_and_place_chair.py``.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sealp.assembly_sequence import TaskPlan, StepParams, AssemblyDef


def main():
    out_dir = os.path.join(
        os.path.dirname(__file__), "_demo_output")
    asmdef_path = os.path.abspath(
        os.path.join(out_dir, "yuanchair.asmdef"))
    tplan_path = os.path.join(out_dir, "yuanchair_plan.tplan")

    # ── Create task plan ─────────────────────────────────────
    plan = TaskPlan(
        assembly_file=asmdef_path,
        name="YuanChair Default Plan",
        description="Single-arm assembly plan with staging from pick_and_place_chair.py",
    )

    # Fixture at origin
    plan.fixture_pos = np.array([0.0, 0.0, 0.0])

    # Robot config
    plan.robot.robot_type = "piper"
    plan.robot.base_pos = np.array([0.0, 0.0, 0.0])

    # Staging positions (from pick_and_place_chair.py)
    plan.set_staging("seat", pos=np.array([0.30, 0.00, 0.00]))
    plan.set_staging("leg_fl", pos=np.array([0.10, 0.20, 0.00]))
    plan.set_staging("leg_fr", pos=np.array([0.05, -0.24, 0.00]))
    plan.set_staging("leg_bl", pos=np.array([0.25, 0.24, 0.00]))
    plan.set_staging("leg_br", pos=np.array([0.25, -0.24, 0.00]))

    # Per-step execution params
    plan.set_step_params(StepParams(
        step_id=0, primitive="single_arm_transport",
        approach_distance=0.02, depart_distance=0.02,
    ))
    for i in range(1, 5):
        plan.set_step_params(StepParams(
            step_id=i, primitive="single_arm_transport",
            approach_distance=0.02, depart_distance=0.02,
        ))

    # ── Test assembly link ───────────────────────────────────
    asm = plan.assembly
    if asm:
        print(f"Linked assembly: {asm.name} ({asm.n_parts} parts)")
        poses = plan.compute_assembly_world_poses()
        print("Assembly world poses:")
        for pid, (pos, _) in poses.items():
            if pid != "fixture":
                print(f"  {pid}: {pos}")

    # ── Save ─────────────────────────────────────────────────
    plan.save(tplan_path)
    print(f"\nSaved: {tplan_path}")
    print()
    print(plan.summary())

    # ── Round-trip test ───────────────────────────────────────
    plan2 = TaskPlan.load(tplan_path)
    print(f"\nRound-trip: {plan2}")


if __name__ == "__main__":
    main()
