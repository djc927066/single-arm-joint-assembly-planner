"""Twin towers, 10 movable parts (+ fixed base). A T tower and a gate tower side by side on one base.

    & $py -X utf8 -m scenes.benchmarks.twin10

Assembly frame: origin at the base bottom centre, x away from the robot, z up, mm.

  base     240 x 240 x 14 with 3 pockets
  t_*      north tower, y = +62: pedestal -> arm -> block -> bar -> cap, one chain
  g_*      south tower, y = -62: two posts -> beam -> block -> cap

Orders: the towers do not touch, so any interleaving of their steps works: C(10, 5) x 2 = 504. Which
tower goes next decides how far the arm travels and which parts still lie on the table, so the order
and the layout stay coupled.

Rules for a part the arm can handle (found on portal10, gantry10 and rack10_tall): long parts lie
along x so the gripper closes along y, the one direction it reaches from straight above; a part is
at most 60 mm wide across the jaw; nothing stands within 61 mm of a part along y at its own height,
the reach of a fully open jaw. Parts are short and all different, to keep the table free and to
leave few interchangeable layouts.
"""
from pathlib import Path

from .common import ModelSpec, analyse, part

C = 0.2
FLOOR = 6        # pocket floor
YN, YS = 62, -62
BLUE, ORANGE, TEAL, RED = (0.36, 0.45, 0.78), (0.92, 0.60, 0.18), (0.20, 0.55, 0.62), (0.85, 0.33, 0.27)


def box(x0, x1, y, hy, z0, z1):
    return (x0, x1, y - hy, y + hy, z0, z1)


def spec() -> ModelSpec:
    posts = {"t_pedestal": (-20, 20, YN, 25), "g_post_W": (-58, -32, YS, 22), "g_post_E": (30, 60, YS, 18)}
    base = part("base", "固定底座", [(-120, 120, -120, 120, 0, 14)],
                [(x0 - C, x1 + C, y - hy - C, y + hy + C, FLOOR, 14) for x0, x1, y, hy in posts.values()],
                color=(0.66, 0.70, 0.74), role="fixed")
    parts = [
        part("t_pedestal", "T 塔基座", [box(-20, 20, YN, 25, FLOOR, 66)], color=BLUE, role="pedestal"),
        part("t_arm", "T 塔横臂", [box(-55, 55, YN, 13, 66, 86)], color=ORANGE, supports=("t_pedestal",), role="arm"),
        part("t_block", "T 塔垫块", [box(-30, 30, YN, 15, 86, 116)], color=TEAL, supports=("t_arm",), role="block"),
        part("t_bar", "T 塔上梁", [box(-45, 45, YN, 11, 116, 132)], color=ORANGE, supports=("t_block",), role="bar"),
        part("t_cap", "T 塔顶帽", [box(-15, 15, YN, 18, 132, 170)], color=RED, supports=("t_bar",), role="cap"),
        part("g_post_W", "门塔西柱", [box(-58, -32, YS, 22, FLOOR, 66)], color=BLUE, role="post"),
        part("g_post_E", "门塔东柱", [box(30, 60, YS, 18, FLOOR, 66)], color=BLUE, role="post"),
        part("g_beam", "门塔横梁", [box(-62, 62, YS, 12, 66, 86)], color=ORANGE, supports=("g_post_W", "g_post_E"), role="beam"),
        part("g_block", "门塔垫块", [box(-25, 25, YS, 17, 86, 126)], color=TEAL, supports=("g_beam",), role="block"),
        part("g_cap", "门塔上梁", [box(-35, 35, YS, 11, 126, 146)], color=RED, supports=("g_block",), role="cap"),
    ]
    return ModelSpec(
        "twin10", "双塔（10 件）", base, parts,
        ["北侧 T 形塔：基座 → 横臂 → 垫块 → 上梁 → 顶帽，5 件一条链。南侧门式塔：两根立柱 → 横梁 → 垫块 → 上梁，5 件。",
         "两座塔互不接触，步骤可以任意交错。先装哪座决定手臂的走动距离和桌上还剩什么，所以顺序和布局仍然耦合。",
         "尺寸：底座 240×240 mm，总高 170 mm。"])


if __name__ == "__main__":
    analyse(spec(), Path(__file__).with_name("twin10"))
