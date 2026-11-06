# -*- coding: utf-8 -*-
"""离线打印 tower 运动 .pkl 中各关键(取/放)位的关节角(度)。

不连硬件、不动机械臂。读取 ``execute_layout_sequence_visual.py`` 生成的 .pkl,
按臂(lft/rgt)、按零件顺序列出每一步的:
  - 抓取位(去拿, 夹爪闭合)关节角(度)
  - 放置位(放好, 夹爪张开)关节角(度)
  - (可选)用带夹爪的仿真模型 PantheraHTSglArm 做 FK, 打印该关键位 TCP=夹持中心
    在基座坐标系的世界坐标(x,y,z), 便于核对离桌面(z=0)的间隙。

用法
----
    python -m sealp.examples.motion.print_key_joint_angles
    python -m sealp.examples.motion.print_key_joint_angles --arm rgt
    python -m sealp.examples.motion.print_key_joint_angles --pkl path/to.pkl --no-fk
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import (  # noqa: E402
    load_motion_pkl,
    extract_arm_segments,
)
from sealp.examples.motion.replay_tower_on_robot import _key_poses_from_seg  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PKL = os.path.join(_HERE, "_output", "tower_optimal_initial_motions.pkl")

_ARM_CN = {"lft": "左臂", "rgt": "右臂"}


def _fmt_deg(jv) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+6.1f}" for v in jv) + "]"


def _make_fk():
    """返回一个 fk(jv)->(pos, rotmat) 函数, 用带夹爪模型(TCP=夹持中心)。失败返回 None。"""
    try:
        from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm
        robot = PantheraHTSglArm(enable_cc=False)

        def _fk(jv):
            try:
                return robot.fk(jnt_values=np.asarray(jv), toggle_jacobian=False, update=False)
            except TypeError:
                robot.goto_given_conf(jnt_values=np.asarray(jv))
                return robot.gl_tcp_pos, robot.gl_tcp_rotmat

        return _fk
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 无法加载仿真模型做 FK ({exc}); 仅打印关节角。")
        return None


def _print_arm(payload, side: str, fk) -> int:
    segs = extract_arm_segments(payload, side)
    if not segs:
        return 0
    print(f"\n=== {_ARM_CN.get(side, side)} ({side}): {len(segs)} 步 ===")
    n = len(segs)
    for i, seg in enumerate(segs, 1):
        pid = seg["part_id"]
        sid = seg.get("step_id", "?")
        nf = seg["jv"].shape[0]
        print(f"[{side}] 第 {i}/{n} 步  step{sid}  零件[{pid}]  ({nf} 帧)")
        for label, jv_frame, grip, jaw_w in _key_poses_from_seg(seg):
            grip_txt = ""
            if grip == "close":
                grip_txt = "  -> 夹爪闭合(抓住)"
            elif grip == "open":
                grip_txt = "  -> 夹爪张开(放下)"
            if grip and jaw_w is not None:
                grip_txt += f"  jaw={float(jaw_w) * 1000:.1f}mm"
            line = f"    {label:<10s} 关节(度)={_fmt_deg(jv_frame)}{grip_txt}"
            print(line)
            if fk is not None:
                pos, _ = fk(jv_frame)
                print(f"               TCP(夹持中心,world) x={pos[0]:+.4f} "
                      f"y={pos[1]:+.4f} z={pos[2]:+.4f} m  (离桌面 z=0 间隙={pos[2]:+.4f})")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="离线打印 tower .pkl 各关键(取/放)位关节角(度)。")
    ap.add_argument("--pkl", default=_DEFAULT_PKL, help="运动 .pkl 路径")
    ap.add_argument("--arm", choices=["lft", "rgt", "dual"], default="dual",
                    help="打印哪只臂(默认 dual: 两只都打印)")
    ap.add_argument("--no-fk", action="store_true", help="不做 FK, 仅打印关节角(更快)")
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        ap.error(f"pkl not found: {args.pkl}")

    payload = load_motion_pkl(args.pkl)
    print(f"[keys] pkl = {args.pkl}")

    fk = None if args.no_fk else _make_fk()

    sides = ("lft", "rgt") if args.arm == "dual" else (args.arm,)
    total = 0
    for side in sides:
        total += _print_arm(payload, side, fk)
    if total == 0:
        print("[keys] 该 pkl 中没有匹配的臂步骤。")


if __name__ == "__main__":
    main()
