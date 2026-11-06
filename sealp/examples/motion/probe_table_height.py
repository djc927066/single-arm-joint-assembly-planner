# -*- coding: utf-8 -*-
"""实测真机桌面相对仿真 z=0 的高度差(末端高度探针)。

背景
----
仿真里 work_table 顶面 = z=0, 机械臂基座原点也在 z=0。放置时碰撞检查是二值的,
几乎不留余量, 所以仿真里爪尖能"刚好不碰"桌面。真机若桌面比 z=0 高了 Δ,
整条手臂在桌面附近就偏低 Δ, 爪尖就会快碰/碰到桌子。

本工具
------
连上单臂 -> 松扭矩(free-spin) -> 实时用 *带夹爪的完整机器人模型(PantheraHTSglArm)*
对当前关节角做 FK, 打印 TCP=夹持中心(acting center, 法兰下方 ~0.17m)在基座坐标系
里的世界坐标(尤其是 z)。注意: 真正的爪尖在夹持中心稍下方, 二者差几毫米~1cm。

(之前的版本误用了无夹爪的 PantheraHT 手臂模型, 打印的是法兰高度 ~0.17, 需再减夹爪
长 0.17 才是夹持中心; 现已改为直接输出夹持中心。)

用法
----
1. 把手臂放到桌面上方; 运行本脚本(需 --run 才真正连硬件):
     python -m sealp.examples.motion.probe_table_height --port COM3 --run
2. 用手把爪尖(或夹爪上某个你能对齐的参考点)轻轻贴到真桌面某个平整处, 扶稳。
3. 记下此刻打印的 TCP z = Z_touch。
4. 把同一组关节角(脚本也会打印)放进仿真(execute_layout_sequence_visual 里
   arm.goto_given_conf(np.radians([...])))看仿真里爪尖离桌面(z=0)的间隙 g_sim。
   - 真机此刻爪尖在真桌面; 仿真同位姿爪尖在 z=g_sim。
   - => 真桌面高度 Δ ≈ g_sim (相对仿真 z=0)。Δ>0 表示真桌子更高, 需把 work_table 抬高 Δ 重规划。

更省事的纯尺子法(不用本脚本):
   直接量"真桌面/洞洞板顶面"到"机械臂基座底安装面(仿真 z=0)"的垂直距离。
   仿真假设这个距离=0; 量出来是多少, 就是要补偿的 Δ。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import FafuRobotController  # noqa: E402

_DEFAULT_CFG = os.path.join(_REPO_ROOT, "wrs", "robot_con", "panthera_ht", "robot.cfg")


def _sim_fk(robot_sim, jv):
    """用带夹爪的 PantheraHTSglArm 对关节角 jv(弧度)做 FK。

    返回 TCP=夹持中心(acting center, 法兰下方 ~0.17m)的 (pos, rotmat) —
    这正是放置时物体所在/该参考的点, 而不是法兰。
    """
    try:
        return robot_sim.fk(jnt_values=jv, toggle_jacobian=False, update=False)
    except TypeError:
        pass
    robot_sim.goto_given_conf(jnt_values=jv)
    return robot_sim.gl_tcp_pos, robot_sim.gl_tcp_rotmat


def main() -> None:
    ap = argparse.ArgumentParser(description="末端高度探针: 实测真机桌面相对仿真 z=0 的高度差。")
    ap.add_argument("--port", default=None, help="单臂串口, 如 COM3 / COM4")
    ap.add_argument("--cfg", default=_DEFAULT_CFG, help="robot.cfg 路径")
    ap.add_argument("--gripper-id", type=int, default=7, help="夹爪电机 id(默认7; -1=无)")
    ap.add_argument("--hz", type=float, default=5.0, help="打印刷新频率")
    ap.add_argument("--no-disable", action="store_true",
                    help="不松扭矩(默认会松扭矩以便手扳; 加此项则保持上电保持)")
    ap.add_argument("--run", action="store_true", help="真正连接硬件(否则仅说明)")
    args = ap.parse_args()

    if not args.run:
        print("[probe] 这是 dry-run 说明, 加 --run 才会连硬件。")
        print(__doc__)
        return

    # 带夹爪的完整机器人(TCP=夹持中心, 与规划/放置一致的 FK)
    from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm
    arm_sim = PantheraHTSglArm(enable_cc=False)

    gripper_id = None if args.gripper_id is not None and args.gripper_id < 0 else args.gripper_id
    ctrl = FafuRobotController(
        cfg_path=args.cfg, port=args.port,
        has_gripper=gripper_id is not None, gripper_motor_id=gripper_id,
    )

    print("=" * 74)
    print("[probe] 末端高度探针")
    print("  仿真假设: work_table 顶面 z=0, 臂基座原点 z=0。")
    if not args.no_disable:
        print("  即将松扭矩(free-spin), 机械臂会因重力下垂 —— 请先扶住!")
        ctrl.disable()
    print("  用手把爪尖贴到真桌面平整处, 读下方 TCP z; Ctrl+C 结束。")
    print("=" * 74)
    z_min = float("inf")
    try:
        while True:
            jv = ctrl.get_joint_values(prefer_cache=False)  # 弧度
            pos, _ = _sim_fk(arm_sim, jv)
            z = float(pos[2])
            z_min = min(z_min, z)
            deg = "[" + ", ".join(f"{np.degrees(v):.1f}" for v in jv) + "]"
            print(f"\rTCP(world)  x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={z:+.4f} m"
                  f"   z_min={z_min:+.4f}   关节(度)={deg}      ", end="", flush=True)
            time.sleep(1.0 / max(0.5, args.hz))
    except KeyboardInterrupt:
        pass
    print("\n" + "=" * 74)
    print(f"[probe] 本次 TCP 最低 z = {z_min:+.4f} m")
    print("把爪尖贴桌时的 TCP z 记下, 对照仿真同关节角下爪尖离 z=0 的间隙, 即得 Δ。")
    print("=" * 74)
    ctrl.close_connection(joint_release="brake")


if __name__ == "__main__":
    main()
