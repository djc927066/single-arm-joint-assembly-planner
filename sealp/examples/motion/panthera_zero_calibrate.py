# -*- coding: utf-8 -*-
"""Panthera-HT 单关节零位标定工具（谨慎使用）。

用途：
1. 仅连接并读取 J1~J6 与电机 ID 的映射；
2. 在机械关节已经人工对准“仿真 0 位”后，将指定关节当前位置写为电机零位；
3. 每次只允许标定一个关节，并要求二次文字确认。

安全原则：
- 不要在悬空、无支撑状态下标定；
- 不要一次释放所有关节；
- 先用 --read-only 查看映射和反馈；
- 只有确认机械关节已经准确对齐后，才使用 --run；
- reset_zero 会改变持久零位，原有运动 PKL 必须重新低速验证；若基座/模型/规划参数也改变，则重新规划。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# 当脚本复制到 sealp/examples/motion 目录时，上面路径正确；
# 当脚本直接从其他位置运行时，也尝试使用当前工作目录作为仓库根目录。
if not os.path.isdir(os.path.join(_REPO_ROOT, "wrs")):
    _REPO_ROOT = os.getcwd()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import FafuRobotController  # noqa: E402

_DEFAULT_CFG = os.path.join(
    _REPO_ROOT, "wrs", "robot_con", "panthera_ht", "robot.cfg"
)


def _fmt_deg(jv) -> str:
    vals = np.degrees(np.asarray(jv, dtype=float).reshape(-1))
    return "[" + ", ".join(f"{v:.3f}" for v in vals) + "] deg"


def _build_arm(cfg_path: str, port: str | None, gripper_id: int | None):
    # 标定时不自动进入位置控制，避免连接瞬间产生不必要动作。
    return FafuRobotController(
        cfg_path=cfg_path,
        port=port,
        has_gripper=gripper_id is not None,
        gripper_motor_id=gripper_id,
        auto_enable=False,
        auto_polling=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Panthera-HT 单关节零位标定工具")
    ap.add_argument("--arm", choices=["lft", "rgt"], default="lft")
    ap.add_argument("--cfg", default=_DEFAULT_CFG)
    ap.add_argument("--port", required=True, help="串口，例如 COM3")
    ap.add_argument("--gripper-id", type=int, default=7,
                    help="夹爪电机 ID；-1 表示不启用夹爪")
    ap.add_argument("--joint", type=int, choices=range(1, 7),
                    help="需要重新设零的机械关节编号 J1~J6")
    ap.add_argument("--read-only", action="store_true",
                    help="只读取映射和当前角度，不写零位")
    ap.add_argument("--run", action="store_true",
                    help="真正执行 reset_zero；默认只做 dry-run")
    args = ap.parse_args()

    if not os.path.isfile(args.cfg):
        ap.error(f"cfg not found: {args.cfg}")
    if not args.read_only and args.joint is None:
        ap.error("写零位时必须指定 --joint 1~6；只查看请用 --read-only")

    gripper_id = None if args.gripper_id < 0 else args.gripper_id
    arm_cn = "左臂" if args.arm == "lft" else "右臂"
    ctrl = _build_arm(args.cfg, args.port, gripper_id)

    try:
        mapping = ctrl.joint_motor_ids
        print(f"[calib] {arm_cn} port={args.port}")
        print("[calib] 机械关节 -> 电机 ID 映射:")
        for i, mid in enumerate(mapping, 1):
            print(f"  J{i} -> motor {mid}")

        before = ctrl.get_joint_values(prefer_cache=False)
        print(f"[calib] 当前反馈: {_fmt_deg(before)}")

        if args.read_only:
            print("[calib] read-only：未修改任何零位。")
            return

        jidx = args.joint - 1
        motor_id = mapping[jidx]
        print("\n[重要] 只有当以下条件全部满足时才能继续：")
        print(f"  1) {arm_cn} J{args.joint} 已经物理对准仿真/URDF 的 0° 姿态；")
        print("  2) 机械臂有可靠支撑，不会因失去保持而下落；")
        print("  3) 急停可立即使用，其他关节不会受到挤压或碰撞；")
        print("  4) 已记录原始反馈，理解该操作会改变持久零位。")

        if not args.run:
            print(f"\n[DRY-RUN] 将把 J{args.joint} 对应 motor {motor_id} 的当前位置设为新零位。")
            print("确认机械对齐后，加 --run 才会真正执行。")
            return

        phrase = f"RESET J{args.joint}"
        typed = input(f"\n确认无误后输入 {phrase}：").strip()
        if typed != phrase:
            print("[calib] 确认文字不匹配，已取消。")
            return

        ctrl.reset_zero(motor_id, confirm=True)
        time.sleep(0.5)
        after = ctrl.get_joint_values(prefer_cache=False)
        print(f"[calib] J{args.joint}/motor {motor_id} 已执行 reset_zero。")
        print(f"[calib] 写入后反馈: {_fmt_deg(after)}")
        print("[calib] 请保持当前位置不动，重新连接后再次 --read-only 复核。")
    finally:
        try:
            ctrl.close_connection(joint_release="brake", gripper_release="brake")
        except Exception:
            pass


if __name__ == "__main__":
    main()
