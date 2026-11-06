"""
Generate Chair Assembly Sequence YAML
======================================

Creates a YAML assembly sequence for the "yuanchair" (round chair)
based on the poses from pick_and_place_chair.py.

This demonstrates the shared-model pattern: 4 legs use the same STL.
"""
import os
import sys
import numpy as np

# Ensure sealp is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sealp.assembly_sequence import (
    AssemblySequence, AssemblyPart, AssemblyStep,
    save_sequence, export_summary_csv, export_summary_text,
)

# ── Model paths (relative to sealp/assets/) ──────────────────
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "models", "yuanchair")
LEG_MODEL = os.path.abspath(os.path.join(ASSET_DIR, "yuanchair-part2.stl"))
SEAT_MODEL = os.path.abspath(os.path.join(ASSET_DIR, "yuanchair-part1.stl"))

# ── Identity rotation ────────────────────────────────────────
I3 = np.eye(3)


def generate_yuanchair_sequence() -> AssemblySequence:
    """Build the round-chair assembly sequence.

    Assembly order (from pick_and_place_chair.py):
      Step 0: seat placed at center (fixture)
      Step 1-4: four legs placed into the seat

    All four legs share the same model file (yuanchair-part2.stl)
    but have different part_id, init_pos, and assembly_pos.
    """
    seq = AssemblySequence(
        name="YuanChair",
        description=(
            "Round chair assembly: 1 seat + 4 identical legs. "
            "Legs share the same 3D model (yuanchair-part2.stl)."
        ),
    )

    # ── Parts ─────────────────────────────────────────────────
    # Seat (chair-part1): placed at the assembly station first
    seq.add_part(AssemblyPart(
        part_id="seat",
        name="Seat",
        model_path=LEG_MODEL.replace("part2", "part1"),  # yuanchair-part1.stl
        init_pos=np.array([0.30, 0.00, 0.00]),
        init_rotmat=I3,
        assembly_pos=np.array([0.30, 0.00, 0.00]),
        assembly_rotmat=I3,
        mass=1.5,
        color_rgba=np.array([0.55, 0.45, 0.35, 1.0]),  # wood brown
    ))

    # Four legs — same model, different positions
    # Staging (pick) positions from pick_and_place_chair.py
    pick_positions = [
        np.array([0.10, 0.20, 0.00]),   # leg_fl (front-left)
        np.array([0.05, -0.24, 0.00]),   # leg_fr (front-right)
        np.array([0.25, 0.24, 0.00]),    # leg_bl (back-left)
        np.array([0.25, -0.24, 0.00]),   # leg_br (back-right)
    ]
    # Assembly (goal) positions from pick_and_place_chair.py
    goal_positions = [
        np.array([0.19, 0.11, 0.02]),   # leg_fl
        np.array([0.19, -0.11, 0.02]),   # leg_fr
        np.array([0.41, -0.11, 0.02]),   # leg_bl
        np.array([0.41, 0.11, 0.02]),    # leg_br
    ]
    leg_ids = ["leg_fl", "leg_fr", "leg_bl", "leg_br"]
    leg_names = ["Front-Left Leg", "Front-Right Leg", "Back-Left Leg", "Back-Right Leg"]
    leg_colors = [
        np.array([0.40, 0.76, 0.96, 1.0]),  # blue
        np.array([0.96, 0.65, 0.35, 1.0]),  # orange
        np.array([0.55, 0.85, 0.50, 1.0]),  # green
        np.array([0.95, 0.50, 0.50, 1.0]),  # red
    ]

    for i, (pid, name) in enumerate(zip(leg_ids, leg_names)):
        seq.add_part(AssemblyPart(
            part_id=pid,
            name=name,
            model_path=LEG_MODEL,  # ALL legs share the same STL
            init_pos=pick_positions[i],
            init_rotmat=I3,
            assembly_pos=goal_positions[i],
            assembly_rotmat=I3,
            mass=0.3,
            color_rgba=leg_colors[i],
            metadata={"shared_model": "yuanchair-part2.stl", "leg_index": i},
        ))

    # ── Steps ─────────────────────────────────────────────────
    # Step 0: place the seat first (it's the fixture)
    seq.add_step(AssemblyStep(
        step_id=0,
        part_id="seat",
        parent_part_id="fixture",
        assembly_pos=np.array([0.30, 0.00, 0.00]),
        assembly_rotmat=I3,
        dependencies=[],
        primitive_type="single_arm_transport",
        notes="Place the seat at the assembly station.",
    ))

    # Steps 1-4: assemble each leg onto the seat
    for i, pid in enumerate(leg_ids):
        seq.add_step(AssemblyStep(
            step_id=i + 1,
            part_id=pid,
            parent_part_id="seat",
            assembly_pos=goal_positions[i],
            assembly_rotmat=I3,
            dependencies=[0],  # seat must be placed first
            primitive_type="single_arm_transport",
            notes=f"Insert {leg_names[i]} into the seat.",
        ))

    # ── Validate ──────────────────────────────────────────────
    errors = seq.validate(strict=False)
    if errors:
        print("Validation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation passed (no errors).")

    return seq


def main():
    seq = generate_yuanchair_sequence()

    # Save outputs
    out_dir = os.path.join(os.path.dirname(__file__), "..", "assembly_sequence", "_demo_output")
    os.makedirs(out_dir, exist_ok=True)

    yaml_path = os.path.join(out_dir, "yuanchair.yaml")
    csv_path = os.path.join(out_dir, "yuanchair_summary.csv")
    txt_path = os.path.join(out_dir, "yuanchair_summary.txt")

    save_sequence(seq, yaml_path)
    export_summary_csv(seq, csv_path)
    export_summary_text(seq, txt_path)

    print(f"\nSaved: {yaml_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")
    print()
    print(seq.summary())
    print(f"\nNote: 4 legs share the same model: {LEG_MODEL}")


if __name__ == "__main__":
    main()
