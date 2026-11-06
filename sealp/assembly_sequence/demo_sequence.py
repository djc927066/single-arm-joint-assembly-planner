"""
Assembly Sequence Demo
======================

Demonstrates creating, saving, loading, and inspecting an assembly
sequence using the SEALP assembly sequence format.

This example builds a simple 4-part furniture assembly (a small table):
  1. Table top  (base part — placed on the fixture)
  2. Leg A      (front-left)
  3. Leg B      (front-right)
  4. Cross-bar  (connects the two legs)

Run directly::

    python -m sealp.assembly_sequence.demo_sequence
"""

import os
import tempfile

import numpy as np


def _rotmat_from_euler(ai, aj, ak):
    """Minimal ZYX Euler-angle → rotation matrix (avoids WRS import)."""
    ci, si = np.cos(ai), np.sin(ai)
    cj, sj = np.cos(aj), np.sin(aj)
    ck, sk = np.cos(ak), np.sin(ak)
    return np.array([
        [ci * cj, ci * sj * sk - si * ck, ci * sj * ck + si * sk],
        [si * cj, si * sj * sk + ci * ck, si * sj * ck - ci * sk],
        [-sj,     cj * sk,                cj * ck],
    ])

from sealp.assembly_sequence import (
    AssemblySequence,
    AssemblyPart,
    AssemblyStep,
    SequenceGenerator,
    save_sequence,
    load_sequence,
)
from sealp.assembly_sequence.sequence_io import (
    export_summary_csv,
    export_summary_text,
)


def demo_manual_construction():
    """Build an assembly manually (add_part / add_step)."""
    print("=" * 60)
    print("Demo 1: Manual Construction")
    print("=" * 60)

    seq = AssemblySequence(
        name="SimpleTable",
        description="A small table with 2 legs and a cross-bar.",
    )

    # --- Parts -----------------------------------------------------------
    seq.add_part(AssemblyPart(
        part_id="table_top",
        name="Table Top",
        model_path="models/table_top.stl",
        init_pos=np.array([0.5, 0.0, 0.1]),
        assembly_pos=np.array([0.0, 0.0, 0.4]),
        assembly_rotmat=np.eye(3),
        mass=2.0,
        color_rgba=np.array([0.82, 0.71, 0.55, 1.0]),
    ))

    seq.add_part(AssemblyPart(
        part_id="leg_a",
        name="Front-Left Leg",
        model_path="models/leg.stl",
        init_pos=np.array([0.6, -0.2, 0.1]),
        assembly_pos=np.array([-0.15, -0.1, 0.0]),
        assembly_rotmat=np.eye(3),
        mass=0.5,
        color_rgba=np.array([0.55, 0.37, 0.24, 1.0]),
    ))

    seq.add_part(AssemblyPart(
        part_id="leg_b",
        name="Front-Right Leg",
        model_path="models/leg.stl",
        init_pos=np.array([0.6, 0.2, 0.1]),
        assembly_pos=np.array([0.15, -0.1, 0.0]),
        assembly_rotmat=np.eye(3),
        mass=0.5,
        color_rgba=np.array([0.55, 0.37, 0.24, 1.0]),
    ))

    seq.add_part(AssemblyPart(
        part_id="cross_bar",
        name="Cross Bar",
        model_path="models/cross_bar.stl",
        init_pos=np.array([0.7, 0.0, 0.1]),
        assembly_pos=np.array([0.0, -0.1, 0.15]),
        assembly_rotmat=_rotmat_from_euler(0, 0, np.pi / 2),
        mass=0.3,
        color_rgba=np.array([0.6, 0.6, 0.6, 1.0]),
    ))

    # --- Steps -----------------------------------------------------------
    seq.add_step(AssemblyStep(
        step_id=0,
        part_id="table_top",
        parent_part_id="fixture",
        assembly_pos=np.array([0.0, 0.0, 0.4]),
        notes="Place the table top on the fixture upside-down.",
    ))

    seq.add_step(AssemblyStep(
        step_id=1,
        part_id="leg_a",
        parent_part_id="table_top",
        assembly_pos=np.array([-0.15, -0.1, 0.0]),
        dependencies=[0],
        notes="Insert front-left leg.",
    ))

    seq.add_step(AssemblyStep(
        step_id=2,
        part_id="leg_b",
        parent_part_id="table_top",
        assembly_pos=np.array([0.15, -0.1, 0.0]),
        dependencies=[0],
        notes="Insert front-right leg (can be parallel with leg_a).",
    ))

    seq.add_step(AssemblyStep(
        step_id=3,
        part_id="cross_bar",
        parent_part_id="leg_a",
        assembly_pos=np.array([0.0, -0.1, 0.15]),
        assembly_rotmat=_rotmat_from_euler(0, 0, np.pi / 2),
        dependencies=[1, 2],
        primitive_type="dual_arm_cooperative",
        notes="Bridge the two legs with the cross bar.",
    ))

    # --- Validate --------------------------------------------------------
    errors = seq.validate(strict=False)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("✓ Validation passed.")

    # --- Print summary ---------------------------------------------------
    print()
    print(seq.summary())

    # --- Execution order -------------------------------------------------
    print("\nExecution order:")
    for step in seq.get_execution_order():
        print(f"  → Step {step.step_id}: {step.part_id} "
              f"onto {step.parent_part_id}")

    return seq


def demo_builder_pattern():
    """Build an assembly using the SequenceGenerator (builder)."""
    print("\n" + "=" * 60)
    print("Demo 2: Builder Pattern (SequenceGenerator)")
    print("=" * 60)

    seq = (
        SequenceGenerator("ShelfUnit",
                          description="A simple 3-shelf unit.")
        .add_part("side_l", "Left Side Panel", "models/side_panel.stl",
                  assembly_pos=[0, 0, 0], mass=1.5)
        .add_part("side_r", "Right Side Panel", "models/side_panel.stl",
                  assembly_pos=[0.4, 0, 0], mass=1.5)
        .add_part("shelf_1", "Bottom Shelf", "models/shelf.stl",
                  assembly_pos=[0.2, 0, 0.1], mass=0.8)
        .add_part("shelf_2", "Middle Shelf", "models/shelf.stl",
                  assembly_pos=[0.2, 0, 0.3], mass=0.8)
        .add_part("shelf_3", "Top Shelf", "models/shelf.stl",
                  assembly_pos=[0.2, 0, 0.5], mass=0.8)
        .auto_generate_linear_sequence(parent_id="fixture")
        .build()
    )

    print(seq.summary())
    return seq


def demo_yaml_io(seq: AssemblySequence):
    """Save and reload an assembly via YAML."""
    print("\n" + "=" * 60)
    print("Demo 3: YAML Save / Load Round-Trip")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), "_demo_output")
    os.makedirs(out_dir, exist_ok=True)

    yaml_path = os.path.join(out_dir, "simple_table.yaml")
    csv_path = os.path.join(out_dir, "simple_table_summary.csv")
    txt_path = os.path.join(out_dir, "simple_table_summary.txt")

    # Save
    save_sequence(seq, yaml_path)
    print(f"✓ Saved YAML  → {yaml_path}")

    export_summary_csv(seq, csv_path)
    print(f"✓ Saved CSV   → {csv_path}")

    export_summary_text(seq, txt_path)
    print(f"✓ Saved TXT   → {txt_path}")

    # Reload
    loaded = load_sequence(yaml_path)
    print(f"\n✓ Loaded back: {loaded}")
    print(f"  Parts: {loaded.part_ids}")
    print(f"  Steps: {[s.step_id for s in loaded.steps]}")

    # Verify round-trip integrity
    assert loaded.n_parts == seq.n_parts, "Part count mismatch!"
    assert loaded.n_steps == seq.n_steps, "Step count mismatch!"
    for orig, ld in zip(seq.parts, loaded.parts):
        assert orig.part_id == ld.part_id, f"Part ID mismatch: {orig.part_id} vs {ld.part_id}"
        assert np.allclose(orig.assembly_pos, ld.assembly_pos), \
            f"Assembly pos mismatch for {orig.part_id}"
    print("✓ Round-trip integrity verified.\n")

    # Print the generated YAML for inspection
    print("--- Generated YAML ---")
    with open(yaml_path, "r") as f:
        print(f.read())
    print("--- End YAML ---")


def main():
    seq = demo_manual_construction()
    demo_builder_pattern()
    demo_yaml_io(seq)
    print("\nAll demos completed successfully. ✓")


if __name__ == "__main__":
    main()
