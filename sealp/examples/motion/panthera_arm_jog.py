#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/6/8 16:59
# @Author : ZhangXi
# -*- coding: utf-8 -*-
"""Panthera-HT 单臂手动工具：回零 / 读关节角 / move_j。

默认 **dry-run**（只打印计划）；加 ``--run`` 才会真正动电机。

用法
----
# 交互菜单（推荐）
python -m sealp.examples.motion.panthera_arm_jog --arm lft --port COM4 --run

# 命令行单项
python -m sealp.examples.motion.panthera_arm_jog --arm lft --port COM4 --read
python -m sealp.examples.motion.panthera_arm_jog --arm lft --port COM4 --home --run --speed 15
python -m sealp.examples.motion.panthera_arm_jog --arm lft --port COM4 \\
    --move-j "0,0.5,-0.3,0,0,0" --run --speed 12
python -m sealp.examples.motion.panthera_arm_jog --arm lft --port COM4 \\
    --move-j-deg "0,30,-20,0,0,0" --run
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import FafuRobotController  # noqa: E402

_DEFAULT_CFG = os.path.join(
    _REPO_ROOT, "wrs", "robot_con", "panthera_ht", "robot.cfg"
)


def _build_arm(cfg_path: str, port: str | None, gripper_id: int | None) -> FafuRobotController:
    return FafuRobotController(
        cfg_path=cfg_path,
        port=port,
        has_gripper=gripper_id is not None,
        gripper_motor_id=gripper_id,
    )


def _fmt_jv(jv, *, unit: str = "rad") -> str:
    jv = np.asarray(jv, dtype=float).reshape(-1)
    if unit == "deg":
        vals = [math.degrees(v) for v in jv]
        return "[" + ", ".join(f"{v:.2f}" for v in vals) + "] deg"
    return "[" + ", ".join(f"{v:.4f}" for v in jv) + "] rad"


def _parse_joint_text(text: str, *, is_degrees: bool) -> np.ndarray:
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(f"需要 6 个关节角, 收到 {len(parts)} 个: {text!r}")
    vals = np.array([float(p) for p in parts], dtype=float)
    if is_degrees:
        vals = np.deg2rad(vals)
    return vals


def _print_joint_values(
    arm: FafuRobotController,
    *,
    label: str = "当前关节角",
    prefer_cache: bool = False,
) -> np.ndarray:
    """读取并打印关节角；默认做一次同步新鲜读取，避免缓存值误导。"""
    jv = arm.get_joint_values(prefer_cache=prefer_cache)
    print(f"[{label}] {_fmt_jv(jv, unit='rad')}")
    print(f"[{label}] {_fmt_jv(jv, unit='deg')}")
    return jv


def _do_home(arm: FafuRobotController, *, speed: int, dry_run: bool) -> None:
    if dry_run:
        print(f"[home] (dry-run) 将 move_j 到全零, speed={speed}%")
        return

    print(f"[home] 回零位 speed={speed}% ...")
    arm.go_home(speed=speed, block=True)

    # 必须在连接仍保持 position 模式时立即读回。
    # 如果等脚本退出并切到 brake 后再重连读取，重力下垂会被误认为 home 失败。
    time.sleep(0.5)
    actual = arm.get_joint_values(prefer_cache=False)
    actual_deg = np.degrees(actual)
    max_err = float(np.max(np.abs(actual_deg)))
    print(f"[home] 到位后实际反馈: {_fmt_jv(actual, unit='deg')}")

    if max_err <= 0.5:
        print(f"[home] 完成：最大误差 {max_err:.2f}°，当前连接内已接近全零。")
    else:
        print(
            f"[home/WARN] 最大误差 {max_err:.2f}°，当前连接内未准确到零。"
            "请先不要回放 PKL，检查负载、位置环、软限位和机械卡滞。"
        )


def _do_move_j(
    arm: FafuRobotController,
    target: np.ndarray,
    *,
    speed: int,
    dry_run: bool,
) -> None:
    print(f"[move_j] 目标 {_fmt_jv(target, unit='rad')}")
    print(f"[move_j] 目标 {_fmt_jv(target, unit='deg')}")
    if dry_run:
        print(f"[move_j] (dry-run) 未发送, 加 --run 执行。")
        return
    arm.move_j(target, is_radians=True, speed=speed, block=True)
    print("[move_j] 完成。当前关节角:")
    _print_joint_values(arm, label="到位后")


def _prompt_move_j(arm: FafuRobotController, *, speed: int, dry_run: bool) -> None:
    print("\n输入 6 个关节角, 逗号分隔。")
    print("  例(弧度): 0, 0.5, -0.3, 0, 0, 0")
    print("  例(度):   输入 d 前缀, 如 d0,30,-20,0,0,0")
    print("  直接回车 = 取消")
    text = input("move_j> ").strip()
    if not text:
        print("[move_j] 已取消。")
        return
    is_deg = text.lower().startswith("d")
    if is_deg:
        text = text[1:].lstrip()
    try:
        target = _parse_joint_text(text, is_degrees=is_deg)
    except ValueError as exc:
        print(f"[move_j] 解析失败: {exc}")
        return
    _do_move_j(arm, target, speed=speed, dry_run=dry_run)


def _interactive_loop(arm: FafuRobotController, *, speed: int, dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "RUN"
    print(f"\n=== Panthera 手动控制 [{mode}] ===")
    print("  1) 回零位 (go_home)")
    print("  2) 读取当前关节角")
    print("  3) move_j 到指定关节角")
    print("  0) 退出")
    if dry_run:
        print("  提示: 当前为 dry-run, 1/3 只打印不动作; 启动时加 --run 才会真动。")

    while True:
        try:
            choice = input("\n请选择 [0-3]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break
        if choice in ("0", "q", "quit", "exit"):
            print("[退出]")
            break
        if choice == "1":
            _do_home(arm, speed=speed, dry_run=dry_run)
        elif choice == "2":
            _print_joint_values(arm)
        elif choice == "3":
            _prompt_move_j(arm, speed=speed, dry_run=dry_run)
        else:
            print("无效选项, 请输入 0-3。")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Panthera-HT 单臂手动工具: 回零 / 读关节角 / move_j。"
    )
    ap.add_argument("--arm", choices=["lft", "rgt"], default="lft",
                    help="左/右臂标签(仅用于日志, 默认 lft)")
    ap.add_argument("--cfg", default=_DEFAULT_CFG, help="robot.cfg 路径")
    ap.add_argument("--port", default=None, help="串口, 如 COM4(左) / COM3(右)")
    ap.add_argument("--gripper-id", type=int, default=7,
                    help="夹爪电机 id (默认 7; -1 = 不启用夹爪)")
    ap.add_argument("--speed", type=int, default=15,
                    help="move_j / go_home 速度百分比 (默认 15)")

    ap.add_argument("--home", action="store_true", help="回零位后退出")
    ap.add_argument("--read", action="store_true", help="读取并打印当前关节角后退出")
    ap.add_argument("--move-j", default=None,
                    help='move_j 目标, 6 个关节角逗号分隔, 单位弧度, 如 "0,0.5,-0.3,0,0,0"')
    ap.add_argument("--move-j-deg", default=None,
                    help='move_j 目标, 6 个关节角逗号分隔, 单位度, 如 "0,30,-20,0,0,0"')
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="进入交互菜单 (未指定其它动作时默认开启)")

    ap.add_argument("--run", action="store_true",
                    help="真正连接并动作 (默认 dry-run, 仅打印计划)")
    ap.add_argument(
        "--end-mode",
        choices=["brake", "hold", "stop"],
        default="brake",
        help=(
            "程序退出时关节模式：brake=短路制动但会在重力下缓慢下垂（默认）；"
            "hold=主动保持最后位置，适合临时验证 home，但会持续通电；"
            "stop=自由下垂，仅在机械臂有可靠支撑时使用。"
        ),
    )
    args = ap.parse_args()

    if args.move_j and args.move_j_deg:
        ap.error("--move-j 与 --move-j-deg 不能同时指定")

    has_action = args.home or args.read or args.move_j or args.move_j_deg
    interactive = args.interactive or not has_action

    if not os.path.isfile(args.cfg):
        ap.error(f"cfg not found: {args.cfg}")

    gripper_id = None if args.gripper_id is not None and args.gripper_id < 0 else args.gripper_id
    dry_run = not args.run
    arm_cn = {"lft": "左臂", "rgt": "右臂"}.get(args.arm, args.arm)

    print(f"[jog] {arm_cn} cfg={args.cfg} port={args.port or '(cfg/auto)'} "
          f"speed={args.speed}% {'[DRY-RUN]' if dry_run else '[RUN]'}")

    arm = _build_arm(args.cfg, args.port, gripper_id)
    try:
        if args.read:
            _print_joint_values(arm)
            if not interactive and not (args.home or args.move_j or args.move_j_deg):
                return

        if args.home:
            _do_home(arm, speed=args.speed, dry_run=dry_run)

        if args.move_j:
            target = _parse_joint_text(args.move_j, is_degrees=False)
            _do_move_j(arm, target, speed=args.speed, dry_run=dry_run)
        elif args.move_j_deg:
            target = _parse_joint_text(args.move_j_deg, is_degrees=True)
            _do_move_j(arm, target, speed=args.speed, dry_run=dry_run)

        if interactive:
            _interactive_loop(arm, speed=args.speed, dry_run=dry_run)
    finally:
        try:
            if args.end_mode == "hold":
                print(
                    "[jog/WARN] 退出后使用 hold：电机将继续通电主动保持最后姿态。"
                    "仅用于短时间验证，不要无人值守或长时间保持。"
                )
            arm.close_connection(joint_release=args.end_mode)
        except Exception:
            pass


if __name__ == "__main__":
    main()
