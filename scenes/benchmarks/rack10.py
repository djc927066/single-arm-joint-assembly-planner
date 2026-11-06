"""Two-tier rack, 10 movable parts (+ fixed base). Test assembly for the experiments.

    & $py -X utf8 -m scenes.benchmarks.rack10

Assembly frame: origin at the base bottom centre, x away from the robot, z up, mm.

  base     240 x 120 x 14 with 4 pockets for the lower posts
  lp_1-4   24 x 24 x 70 lower posts at (+-95, +-30)           -> any order
  shelf_1  220 x 84 x 12 on the lower posts, 4 pockets on top   -> needs all 4 lower posts
  up_1-4   24 x 24 x 70 upper posts standing in shelf_1's pockets
  shelf_2  220 x 84 x 12 top shelf on the upper posts          -> needs all 4 upper posts

Two posts on the same side are only 36 mm apart, which removes some grasps of the second
one; shelf grasps use 84 of the 85 mm jaw.
"""
from pathlib import Path

from .common import ModelSpec, analyse, part

C = 0.2
LOW = {"lp_1": (-95, -30), "lp_2": (-95, 30), "lp_3": (95, -30), "lp_4": (95, 30)}
UP = {"up_1": (-95, -30), "up_2": (-95, 30), "up_3": (95, -30), "up_4": (95, 30)}


def spec() -> ModelSpec:
    base = part("base", "固定底座", [(-120, 120, -60, 60, 0, 14)],
                [(x - 12 - C, x + 12 + C, y - 12 - C, y + 12 + C, 6, 14) for x, y in LOW.values()],
                color=(0.66, 0.70, 0.74), role="fixed")
    parts = []
    side = {(-95, -30): "近·右", (-95, 30): "近·左", (95, -30): "远·右", (95, 30): "远·左"}
    for pid, (x, y) in LOW.items():
        parts.append(part(pid, f"下立柱{pid[-1]}（{side[x, y]}）", [(x - 12, x + 12, y - 12, y + 12, 6, 76)],
                          color=(0.20, 0.55, 0.62), role="lower post"))
    parts.append(part("shelf_1", "下层隔板", [(-110, 110, -42, 42, 76, 88)],
                      [(x - 12 - C, x + 12 + C, y - 12 - C, y + 12 + C, 82, 88) for x, y in UP.values()],
                      color=(0.85, 0.33, 0.27), supports=tuple(LOW), role="shelf"))
    for pid, (x, y) in UP.items():
        parts.append(part(pid, f"上立柱{pid[-1]}（{side[x, y]}）", [(x - 12, x + 12, y - 12, y + 12, 82, 152)],
                          color=(0.36, 0.45, 0.78), supports=("shelf_1",), role="upper post"))
    parts.append(part("shelf_2", "顶层隔板", [(-110, 110, -42, 42, 152, 164)],
                      color=(0.36, 0.62, 0.30), supports=tuple(UP), role="top shelf"))
    return ModelSpec(
        "rack10", "双层置物架（10 件）", base, parts,
        ["4 根下立柱 → 下层隔板 → 4 根上立柱插在隔板孔里 → 顶层隔板。同层立柱顺序自由。",
         "尺寸：底座 240×120 mm，总高 164 mm，放在 0.6×0.6 m 桌面 x=0.30 m 处。"])


if __name__ == "__main__":
    analyse(spec(), Path(__file__).with_name("rack10"))
