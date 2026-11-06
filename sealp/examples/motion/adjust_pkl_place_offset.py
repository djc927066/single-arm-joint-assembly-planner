# -*- coding: utf-8 -*-
"""修正 motion .pkl 中某步「放置位」TCP 偏移并重算关节角。

在松开夹爪前的最后放置姿态上，将 TCP 平移 (dx, dy, dz) 后做 FK→IK，
更新对应帧的 jv_list；可选同步 obj_pose_list 的零件位置。

用法
----
    python -m sealp.examples.motion.adjust_pkl_place_offset \\
        --part post_bl --arm lft --dx 0.02

    python -m sealp.examples.motion.adjust_pkl_place_offset --dry-run ...
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import (  # noqa: E402
    _JAW_HELD_THRESH,
    _jaw_width_of,
    load_motion_pkl,
)
from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm  # noqa: E402

_DEFAULT_PKL = os.path.join(
    os.path.dirname(__file__), "_output", "tower_optimal_initial_motions.pkl"
)


def _fk(robot: PantheraHTSglArm, jv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return robot.fk(jnt_values=jv, toggle_jacobian=False, update=False)
    except TypeError:
        robot.goto_given_conf(jnt_values=jv)
        return np.asarray(robot.gl_tcp_pos, dtype=float), np.asarray(
            robot.gl_tcp_rotmat, dtype=float
        )


def _find_open_index(ev: np.ndarray) -> int | None:
    """夹爪从夹持变为张开的帧索引。"""
    for i in range(1, ev.size):
        w0, w1 = ev[i - 1], ev[i]
        if not (np.isfinite(w0) and np.isfinite(w1)):
            continue
        if w0 < _JAW_HELD_THRESH <= w1:
            return i
    return None


def _placement_hold_indices(
    jv: np.ndarray,
    ev: np.ndarray,
    open_idx: int,
    robot: PantheraHTSglArm,
    *,
    pos_tol: float = 0.003,
) -> list[int]:
    """松开前、TCP 与放置位重合的连续夹持帧。"""
    ref_idx = open_idx - 1
    if ref_idx < 0:
        return []
    ref_pos, _ = _fk(robot, jv[ref_idx])
    out: list[int] = []
    for i in range(ref_idx, -1, -1):
        w = ev[i] if i < ev.size else np.nan
        if not np.isfinite(w) or w >= _JAW_HELD_THRESH:
            break
        pos, _ = _fk(robot, jv[i])
        if float(np.linalg.norm(pos - ref_pos)) <= pos_tol:
            out.append(i)
        else:
            break
    out.sort()
    return out


def apply_place_tcp_offset(
    payload: dict,
    *,
    part_id: str,
    arm_side: str,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    pos_tol: float = 0.003,
    robot: PantheraHTSglArm | None = None,
) -> dict:
    """就地修改 payload，返回摘要 dict。"""
    delta = np.array([dx, dy, dz], dtype=float)
    if not np.any(np.abs(delta) > 0):
        raise ValueError("至少指定一个非零偏移 dx/dy/dz")

    if robot is None:
        robot = PantheraHTSglArm(enable_cc=False)

    for st in payload.get("steps", []):
        if st.get("part_id") != part_id:
            continue
        for seg in st.get("segments", []):
            if seg.get("arm_side") != arm_side:
                continue
            jv_list = [np.asarray(x, dtype=float) for x in (seg.get("jv_list") or [])]
            if not jv_list:
                raise ValueError(f"{part_id}/{arm_side}: 空 jv_list")
            jv = np.stack(jv_list, axis=0)
            ev = np.array(
                [_jaw_width_of(e) for e in (seg.get("ev_list") or [])], dtype=float
            )
            if ev.size < jv.shape[0]:
                ev = np.concatenate([ev, np.full(jv.shape[0] - ev.size, np.nan)])

            open_idx = _find_open_index(ev)
            if open_idx is None:
                raise ValueError(f"{part_id}/{arm_side}: 未找到夹爪张开边界")

            hold_idx = _placement_hold_indices(
                jv, ev, open_idx, robot, pos_tol=pos_tol
            )
            if not hold_idx:
                hold_idx = [open_idx - 1]

            seed = jv[hold_idx[-1]]
            old_pos, old_rot = _fk(robot, seed)
            new_pos = old_pos + delta
            sol = robot.ik(
                tgt_pos=new_pos, tgt_rotmat=old_rot, seed_jnt_values=seed
            )
            if sol is None:
                raise RuntimeError(
                    f"{part_id}/{arm_side}: IK 失败 "
                    f"old_tcp={old_pos.round(4)} -> tgt={new_pos.round(4)}"
                )
            new_jv = np.asarray(sol, dtype=float)
            check_pos, _ = _fk(robot, new_jv)
            err_mm = float(np.linalg.norm(check_pos - new_pos)) * 1000.0

            for i in hold_idx:
                seg["jv_list"][i] = new_jv.tolist()

            obj_poses = seg.get("obj_pose_list") or []
            for i in hold_idx:
                if i >= len(obj_poses) or obj_poses[i] is None:
                    continue
                pos, rot = obj_poses[i]
                pos = np.asarray(pos, dtype=float) + delta
                obj_poses[i] = [pos.tolist(), rot]

            return {
                "part_id": part_id,
                "arm_side": arm_side,
                "open_index": open_idx,
                "updated_frames": hold_idx,
                "old_tcp": old_pos.tolist(),
                "new_tcp": check_pos.tolist(),
                "ik_err_mm": err_mm,
            }

    raise KeyError(f"未找到 step part_id={part_id!r} arm={arm_side!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="修正 pkl 放置位 TCP 偏移并重算关节角。")
    ap.add_argument("--pkl", default=_DEFAULT_PKL, help="motion .pkl 路径")
    ap.add_argument("--part", required=True, help="零件 id, 如 post_bl")
    ap.add_argument("--arm", required=True, choices=("lft", "rgt"), help="手臂")
    ap.add_argument("--dx", type=float, default=0.0, help="TCP X 偏移(米)")
    ap.add_argument("--dy", type=float, default=0.0, help="TCP Y 偏移(米)")
    ap.add_argument("--dz", type=float, default=0.0, help="TCP Z 偏移(米)")
    ap.add_argument("--dry-run", action="store_true", help="只打印, 不写文件")
    ap.add_argument("--no-backup", action="store_true", help="不写 .bak 备份")
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        ap.error(f"pkl not found: {args.pkl}")

    payload = load_motion_pkl(args.pkl)
    info = apply_place_tcp_offset(
        payload,
        part_id=args.part,
        arm_side=args.arm,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
    )
    print(f"[place-offset] pkl = {args.pkl}")
    print(
        f"[place-offset] {info['part_id']}/{info['arm_side']} "
        f"open@frame={info['open_index']} "
        f"updated={info['updated_frames']}"
    )
    print(
        f"[place-offset] TCP {np.round(info['old_tcp'], 4)} "
        f"-> {np.round(info['new_tcp'], 4)} "
        f"(IK err {info['ik_err_mm']:.2f} mm)"
    )

    if args.dry_run:
        print("[place-offset] dry-run, 未写文件。")
        return

    if not args.no_backup:
        bak = args.pkl + ".bak"
        shutil.copy2(args.pkl, bak)
        print(f"[place-offset] 已备份 -> {bak}")

    with open(args.pkl, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"[place-offset] 已写回 -> {args.pkl}")


if __name__ == "__main__":
    main()
