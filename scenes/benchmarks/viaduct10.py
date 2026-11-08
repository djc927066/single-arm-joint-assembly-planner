"""Viaduct, 10 movable parts (+ fixed base). Four piers, three spans, one deck, two towers on the deck.

    & $py -X utf8 -m scenes.benchmarks.viaduct10

Assembly frame: origin at the base bottom centre, x away from the robot, z up, mm.

  base     240 x 120 x 14 with 4 pockets
  pier_*   4 piers in a row, z 6-86                              -> any order
  span_*   3 spans, z 86-106, each on two neighbouring piers; the inner piers carry two spans
  deck     180 x 40 x 16, z 106-122, rests on the two end spans  -> needs span_1 and span_3
  tower_*  2 towers on the deck, z 122-166                       -> need the deck

The deck needs only the end spans for support, but it covers span_2, so span_2 must go in first (a
blocking precedence, not a support one). Every long part lies along x so that the gripper closes
along y, the only direction the arm reaches from straight above, and nothing stands beside a part
in y, so a fully open jaw is free. Section sizes differ slightly, like the frame's parts.
"""
from pathlib import Path

from .common import ModelSpec, analyse, part

C = 0.2
FLOOR = 6        # pocket floor
Z1, Z2, Z3, TOP = 86, 106, 122, 166
PIERS = {"pier_1": (-75, 15, 28), "pier_2": (-25, 16, 30), "pier_3": (25, 17, 29), "pier_4": (75, 14, 27)}   # x, half x, half y
SPANS = {"span_1": (-89, -26, 14, ("pier_1", "pier_2")), "span_2": (-24, 24, 15, ("pier_2", "pier_3")),
         "span_3": (26, 88, 13, ("pier_3", "pier_4"))}                                                      # x0, x1, half y
TOWERS = {"tower_1": (-48, 16, 18), "tower_2": (46, 15, 17)}                                                # x, half x, half y


def spec() -> ModelSpec:
    base = part("base", "固定底座", [(-120, 120, -60, 60, 0, 14)],
                [(x - hx - C, x + hx + C, -hy - C, hy + C, FLOOR, 14) for x, hx, hy in PIERS.values()],
                color=(0.66, 0.70, 0.74), role="fixed")
    parts = [part(pid, f"桥墩 {pid[-1]}", [(x - hx, x + hx, -hy, hy, FLOOR, Z1)],
                  color=(0.36, 0.45, 0.78), role="pier") for pid, (x, hx, hy) in PIERS.items()]
    for pid, (x0, x1, w, sup) in SPANS.items():
        parts.append(part(pid, f"跨梁 {pid[-1]}", [(x0, x1, -w, w, Z1, Z2)],
                          color=(0.92, 0.60, 0.18), supports=sup, role="span"))
    parts.append(part("deck", "桥面", [(-90, 90, -20, 20, Z2, Z3)], color=(0.85, 0.33, 0.27),
                      supports=("span_1", "span_3"), role="deck"))
    for pid, (x, hx, hy) in TOWERS.items():
        parts.append(part(pid, f"小塔 {pid[-1]}", [(x - hx, x + hx, -hy, hy, Z3, TOP)],
                          color=(0.20, 0.55, 0.62), supports=("deck",), role="tower"))
    return ModelSpec(
        "viaduct10", "高架桥（10 件）", base, parts,
        ["4 个桥墩 + 3 根跨梁 + 1 块桥面 + 2 座小塔。中间两个桥墩各被两根跨梁共用。",
         "桥面只靠两端跨梁支撑，但盖在中间跨梁上方，所以中间跨梁必须先装；两座小塔要在桥面之后。",
         "尺寸：底座 240×120 mm，总高 166 mm。"])


if __name__ == "__main__":
    analyse(spec(), Path(__file__).with_name("viaduct10"))
