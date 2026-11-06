"""
Generate YuanChair Assembly Definition (.asmdef)
=================================================

Creates a ``.asmdef`` file for the round chair using the new format.
Demonstrates: shared model library, symmetry groups, and relative poses.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef

# ── Model paths ──────────────────────────────────────────────
ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "models", "yuanchair"))
LEG_STL = os.path.join(ASSET_DIR, "yuanchair-part2.stl")
SEAT_STL = os.path.join(ASSET_DIR, "yuanchair-part1.stl")


def generate() -> AssemblyDef:
    """Build the YuanChair assembly definition."""
    asm = AssemblyDef(
        name="YuanChair",
        description="Round chair assembly: 1 seat + 4 identical legs",
    )

    # ── Model library (shared models declared once) ──────────
    asm.add_model("seat_model", SEAT_STL)
    asm.add_model("leg_model", LEG_STL)

    # ── Parts ────────────────────────────────────────────────
    asm.add_part(PartDef(
        part_id="seat", name="Seat",
        model="seat_model", mass=1.5,  # 单臂搬运（fixture 偏侧导致 dual-arm goal 不可达）
    ))
    for pid, name in [
        ("leg_fl", "Front-Left Leg"),
        ("leg_fr", "Front-Right Leg"),
        ("leg_bl", "Back-Left Leg"),
        ("leg_br", "Back-Right Leg"),
    ]:
        asm.add_part(PartDef(
            part_id=pid, name=name,
            model="leg_model", mass=0.3,  # same model for all legs
        ))

    # ── Symmetry groups ──────────────────────────────────────
    asm.add_symmetry_group("legs", ["leg_fl", "leg_fr", "leg_bl", "leg_br"])

    # ── Assembly steps (poses relative to parent) ────────────

    # Step 0: seat → fixture (the workspace surface)
    # Seat sits at the center of the assembly area
    asm.add_step(StepDef(
        step_id=0,
        part_id="seat",
        parent_id="fixture",
        rel_pos=np.array([0.30, 0.0, 0.0]),
        rel_rotmat=np.eye(3),
        deps=[],
        notes="Place the seat at the assembly station.",
    ))

    # Steps 1-4: legs → seat (relative to the seat position)
    # From pick_and_place_chair.py, goal positions are absolute.
    # Seat is at [0.30, 0, 0].
    # Relative to seat: leg_pos_rel = leg_pos_abs - seat_pos
    #
    # 装配顺序（双臂演示场景下经实验确定）：
    #   seat → leg_bl → leg_br → leg_fl → leg_fr
    # 旧顺序 (fl→fr→bl→br) 的瓶颈：装 bl/br 时前腿已立在 chair 上，
    # 远端 (x≈0.41) 工作空间被前腿挤占，长桌腿 mesh 搬过去时 RRT/IK
    # 解空间被压得很薄。改为「先后腿、后前腿」：装后腿时 chair 上只
    # 有 seat（远端空旷），装前腿时虽然后腿已立起，但前腿组装位
    # (x≈0.19) 在近端，机械臂走"侧弧"避开后腿余量大。
    #
    # 命名约定：``_l`` 后缀 → 世界系 +y 侧（左臂可达，base y=0）；
    #          ``_r`` 后缀 → 世界系 -y 侧（右臂可达，base y=-0.62）。
    # 与 ``_arm_priority_for_part`` 和前腿 ``leg_fl/leg_fr`` 的 rel_pos
    # 严格保持一致；之前 bl/br 的 rel_pos.y 是反的，导致左臂被分到
    # 世界 -y 区目标（physically 触不到），右臂被分到 +y 区目标
    # （触不到），双臂指令几乎必出 IK fail。
    seat_pos = np.array([0.30, 0.0, 0.0])
    leg_specs = [
        ("leg_bl", "Back-Left Leg",   np.array([0.41,  0.11, 0.02])),
        ("leg_br", "Back-Right Leg",  np.array([0.41, -0.11, 0.02])),
        ("leg_fl", "Front-Left Leg",  np.array([0.19,  0.11, 0.02])),
        ("leg_fr", "Front-Right Leg", np.array([0.19, -0.11, 0.02])),
    ]
    for i, (pid, name, goal_abs) in enumerate(leg_specs):
        rel_pos = goal_abs - seat_pos  # relative to seat
        asm.add_step(StepDef(
            step_id=i + 1,
            part_id=pid,
            parent_id="seat",
            rel_pos=rel_pos,
            rel_rotmat=np.eye(3),
            deps=[0],  # seat must be placed first
            notes=f"Insert {name} into the seat.",
        ))

    # ── Validate ─────────────────────────────────────────────
    errors = asm.validate(strict=False)
    if errors:
        print("Validation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation passed.")

    return asm


def main():
    asm = generate()

    out_dir = os.path.join(
        os.path.dirname(__file__), "_demo_output")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "yuanchair.asmdef")
    asm.save(path)

    print(f"\nSaved: {path}")
    print()
    print(asm.summary())


if __name__ == "__main__":
    main()
