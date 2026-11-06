# -*- coding: utf-8 -*-
"""将 motion .pkl 里所有零件的夹爪「闭合/夹持」宽度整体收紧(或放宽)。

只改 ev_list 里 jaw_width < 张开阈值 的帧(典型夹持 ~40mm), 张开位(~80mm)不动。
默认在现有基础上 **缩小 1cm** (delta=-0.01m), 写回原 pkl(可先备份)。

用法
----
    python -m sealp.examples.motion.adjust_pkl_jaw_close
    python -m sealp.examples.motion.adjust_pkl_jaw_close --delta -0.01 --pkl path/to.pkl
    python -m sealp.examples.motion.adjust_pkl_jaw_close --dry-run
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs.robot_con.panthera_ht.fafu_robot_controller import (  # noqa: E402
    apply_jaw_close_delta,
    load_motion_pkl,
)

_DEFAULT_PKL = os.path.join(
    os.path.dirname(__file__), "_output", "tower_optimal_initial_motions.pkl"
)


def _summarize_closed_widths(payload: dict) -> dict:
    """返回 {part_id: (min_mm, max_mm, count)} 仅统计夹持宽度。"""
    from wrs.robot_con.panthera_ht.fafu_robot_controller import _jaw_width_of, _JAW_HELD_THRESH
    import numpy as np

    out: dict = {}
    for st in payload.get("steps", []):
        pid = st.get("part_id", "?")
        ws = []
        for seg in st.get("segments", []):
            for ev in seg.get("ev_list") or []:
                w = _jaw_width_of(ev)
                if np.isfinite(w) and w < _JAW_HELD_THRESH:
                    ws.append(float(w))
        if ws:
            out[pid] = (min(ws) * 1000, max(ws) * 1000, len(ws))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="收紧 pkl 中所有零件的夹爪闭合宽度。")
    ap.add_argument("--pkl", default=_DEFAULT_PKL, help="motion .pkl 路径")
    ap.add_argument("--delta", type=float, default=-0.01,
                    help="夹持宽度增量(米); 默认 -0.01 = 再收紧 1cm")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计, 不写文件")
    ap.add_argument("--no-backup", action="store_true", help="不写 .bak 备份")
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        ap.error(f"pkl not found: {args.pkl}")

    payload = load_motion_pkl(args.pkl)
    before = _summarize_closed_widths(payload)
    print(f"[jaw-adjust] pkl = {args.pkl}")
    print(f"[jaw-adjust] delta = {args.delta * 1000:+.1f} mm (仅夹持位, 张开位不变)")
    print("[jaw-adjust] 调整前夹持宽度 (mm):")
    for pid, (lo, hi, cnt) in before.items():
        print(f"  {pid:14s}  min={lo:.1f}  max={hi:.1f}  frames={cnt}")

    n = apply_jaw_close_delta(payload, args.delta)
    after = _summarize_closed_widths(payload)
    print(f"[jaw-adjust] 修改 {n} 帧")
    print("[jaw-adjust] 调整后夹持宽度 (mm):")
    for pid, (lo, hi, cnt) in after.items():
        print(f"  {pid:14s}  min={lo:.1f}  max={hi:.1f}  frames={cnt}")

    if args.dry_run:
        print("[jaw-adjust] dry-run, 未写文件。")
        return

    if not args.no_backup:
        bak = args.pkl + ".bak"
        shutil.copy2(args.pkl, bak)
        print(f"[jaw-adjust] 已备份 -> {bak}")

    with open(args.pkl, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"[jaw-adjust] 已写回 -> {args.pkl}")


if __name__ == "__main__":
    main()
