"""rack10 with every post 30 mm longer (100 mm instead of 70 mm); everything else as rack10.

    & $py -X utf8 -m scenes.benchmarks.rack10_tall

rack10's lower shelf sits so low (centre 82 mm) that the palm of every horizontal shelf grasp hits
the table / base at the insertion (2026-10-25); 30 mm more leaves ~220 collision-free grasps.

  lp_1-4   24 x 24 x 100 lower posts, z 6-106
  shelf_1  220 x 84 x 12, z 106-118, pockets 112-118
  up_1-4   24 x 24 x 100 upper posts, z 112-212
  shelf_2  220 x 84 x 12, z 212-224
"""
from pathlib import Path

from .common import ModelSpec, analyse, part
from .rack10 import C, LOW, UP

DZ = 30     # mm added to each post


def spec() -> ModelSpec:
    base = part("base", "固定底座", [(-120, 120, -60, 60, 0, 14)],
                [(x - 12 - C, x + 12 + C, y - 12 - C, y + 12 + C, 6, 14) for x, y in LOW.values()],
                color=(0.66, 0.70, 0.74), role="fixed")
    parts = []
    side = {(-95, -30): "近·右", (-95, 30): "近·左", (95, -30): "远·右", (95, 30): "远·左"}
    for pid, (x, y) in LOW.items():
        parts.append(part(pid, f"下立柱{pid[-1]}（{side[x, y]}）", [(x - 12, x + 12, y - 12, y + 12, 6, 76 + DZ)],
                          color=(0.20, 0.55, 0.62), role="lower post"))
    parts.append(part("shelf_1", "下层隔板", [(-110, 110, -42, 42, 76 + DZ, 88 + DZ)],
                      [(x - 12 - C, x + 12 + C, y - 12 - C, y + 12 + C, 82 + DZ, 88 + DZ) for x, y in UP.values()],
                      color=(0.85, 0.33, 0.27), supports=tuple(LOW), role="shelf"))
    for pid, (x, y) in UP.items():
        parts.append(part(pid, f"上立柱{pid[-1]}（{side[x, y]}）", [(x - 12, x + 12, y - 12, y + 12, 82 + DZ, 152 + 2 * DZ)],
                          color=(0.36, 0.45, 0.78), supports=("shelf_1",), role="upper post"))
    parts.append(part("shelf_2", "顶层隔板", [(-110, 110, -42, 42, 152 + 2 * DZ, 164 + 2 * DZ)],
                      color=(0.36, 0.62, 0.30), supports=tuple(UP), role="top shelf"))
    return ModelSpec(
        "rack10_tall", "双层置物架（10 件，立柱加长 30 mm）", base, parts,
        ["同 rack10：4 根下立柱 → 下层隔板 → 4 根上立柱 → 顶层隔板；每根立柱 100 mm（rack10 为 70 mm）。",
         "尺寸：底座 240×120 mm，总高 224 mm。"])


if __name__ == "__main__":
    analyse(spec(), Path(__file__).with_name("rack10_tall"))
