"""
Fast Optimal Dual-Arm Layout Search — Shelf Unit
=================================================

与 ``find_optimal_layout.py``（YuanChair）**同一套搜索引擎与评分权重**：

  Layout Score
      = 0.30 × 抓取冗余度 (Grasp Count)
      + 0.10 × 端点可操作性 (Manip_EP)
      + 0.40 × 轨迹可操作性 (Manip_TRAJ, 全流程 min-along-path)
      + 0.20 × 运输距离   (Distance, 越短越高)

与 YuanChair 的唯一实质差异
---------------------------
YuanChair 的 staging 姿态与 goal 基本一致（seat 直立、leg 有预定义
直立/躺姿列表），因此 rotmat 候选写在 ``find_optimal_layout.py`` 里即可。

Shelf Unit 的 staging 与 goal **朝向不同**（例如 shelf 板 staging 需
竖立/侧夹、goal 需水平放置），必须在 grasp library 规划完成后，自动
推断"哪些 staging rotmat 能让 pick 端与 place 端共享同一 grasp"。

Pick/place 接近与撤离方向见 ``_tasks/shelf_motion.py``（与 YuanChair
的 seat −Z/+Z、leg −X 完全分离）；L2/L3 与 ``dual_sequence_execution_shelf``
共用同一套参数。

前置步骤（首次运行前执行一次）
------------------------------
::

    python -m sealp.assets.models.shelf_unit.gen_meshes
    python -m sealp.examples.grasp.plan_shelf_unit_grasps --no-vis
    python -m sealp.assembly_sequence.gen_shelf_unit_asmdef

CLI 用法
--------
::

    # 正常 4 件：side_l → shelf_m → shelf_t → side_r（默认）
    python -m sealp.examples.layout.find_optimal_layout_shelf

    # 跳过顶层板，只搜 side_l → shelf_m → side_r（随时可去掉 --skip-parts 恢复 4 件）
    python -m sealp.examples.layout.find_optimal_layout_shelf --skip-parts shelf_t

    # 加大样本 / 换种子
    python -m sealp.examples.layout.find_optimal_layout_shelf --n-samples 50 --seed 42

    # CEM 演化（慢但更彻底）
    python -m sealp.examples.layout.find_optimal_layout_shelf --mode cem --gens 8 --pop 24

输出
----
``sealp/examples/layout/_output/dual_shelf_unit_optimal_searched.layout``
及 ``_diagnostics.json``（供 ``diagnostics_plot.py`` 出图）。
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import sealp.examples.layout.find_optimal_layout as _fol

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sealp.examples.layout.find_optimal_layout import (
    STAGING_ROTMAT_CANDIDATES,
    WEIGHT_DIST,
    WEIGHT_GRASP,
    WEIGHT_MANIP_EP,
    WEIGHT_MANIP_TRAJ,
    run_fast_search,
)
from sealp.examples.layout._tasks import shelf_unit
from sealp.examples.layout._tasks.shelf_geometry import DEFAULT_TABLE_MARGIN

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast Optimal Layout Search — Shelf Unit "
                    "(auto staging rotmat from grasp library)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["random", "cem"], default="random",
        help="random=随机采样 N 个候选取最高分(默认,快); cem=多代演化(慢,更彻底)",
    )
    parser.add_argument(
        "--n-samples", "--n", dest="n_samples", type=int, default=20,
        help="[random] 随机采样数 (默认 20，含 seed anchor)",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子；同种子可复现")
    parser.add_argument("--pop", type=int, default=24, help="[cem] 每代候选数")
    parser.add_argument("--gens", type=int, default=8, help="[cem] 最大代数")
    parser.add_argument(
        "--enable-l3", action="store_true",
        help="开启 L3 RRT 真实校验 (random 模式下默认关闭)",
    )
    parser.add_argument(
        "--no-auto-rotmat", action="store_true",
        help="跳过 auto_rotmat 推断（仅用于调试；正常请勿使用）",
    )
    parser.add_argument(
        "--quiet-auto-rotmat", action="store_true",
        help="auto_rotmat 推断阶段不打印详细排名",
    )
    parser.add_argument(
        "--traj-grasp-try", type=int, default=None,
        help="L2 轨迹检查阶段最多尝试多少个 valid grasp；建议设为 20、30 或 50。",
    )
    parser.add_argument(
        "--ik-retry", type=int, default=None,
        help="L2 IK 失败时的重试次数；建议调试时设为 2 或 3。",
    )
    parser.add_argument(
        "--l2-debug", action="store_true",
        help="打印 L2 每个 rot×arm 的详细失败原因（输出较多）",
    )
    parser.add_argument(
        "--no-preflight", action="store_true",
        help="跳过搜索前的 goal/staging 预检打印",
    )
    parser.add_argument(
        "--table-margin", type=float, default=None,
        help="work_table 桌沿留白 (m)，默认 0.06；越小搜索区越大",
    )
    parser.add_argument(
        "--skip-parts", type=str, default="",
        help="跳过布局/L2 的零件（逗号分隔）。例: shelf_t → 只搜 side_l,shelf_m,side_r；"
             "省略本参数即恢复完整 4 件装配",
    )

    args = parser.parse_args()
    skip_parts = tuple(
        p.strip() for p in args.skip_parts.split(",") if p.strip()
    )

    # 非 easy 模式也允许手动提高 L2 轨迹抓取尝试数。
    # 用于解决 "tried 6/149 valid grasps" 这种只试前 6 个抓取姿态的问题。
    if args.traj_grasp_try is not None:
        _fol.TRAJ_GRASP_TRY_LIMIT = int(args.traj_grasp_try)
    elif _fol.TRAJ_GRASP_TRY_LIMIT == 6:
        # shelf 竖→横 common 分散，默认多试几个 grasp 做轨迹探针
        _fol.TRAJ_GRASP_TRY_LIMIT = 20

    ik_retry = 0
    if args.ik_retry is not None:
        ik_retry = int(args.ik_retry)

    if args.l2_debug:
        _fol.DEBUG_L2_FAIL = True

    _fol.SHELF_FUNNEL_COMPARE = True

    print(f"  traj_grasp_try={_fol.TRAJ_GRASP_TRY_LIMIT}  ik_retry_n={ik_retry}  "
          f"l2_debug={args.l2_debug}  shelf_funnel=True")

    # ── 1. 加载 shelf 任务定义 + 自动推断 staging rotmat ────────────────
    table_margin = (
        float(args.table_margin)
        if args.table_margin is not None
        else DEFAULT_TABLE_MARGIN
    )
    task, rotmat_dict = shelf_unit.register_task(
        verbose=not args.quiet_auto_rotmat and not args.no_auto_rotmat,
        table_margin=table_margin,
        skip_parts=skip_parts,
    )

    if not args.no_preflight:
        shelf_unit.print_preflight_diagnostics(
            asmdef_path=task.asmdef_path,
            fixture_pos=task.fixture_pos,
            grasp_pickles=task.grasp_pickles,
            rotmat_dict=rotmat_dict,
            part_ids=task.part_ids,
        )

    print(f"[fixture] search fixture_pos = {task.fixture_pos.tolist()} "
          f"(与 mesh_frames / dual_sequence_execution_shelf 一致)")

    if not args.no_auto_rotmat:
        STAGING_ROTMAT_CANDIDATES.update(rotmat_dict)
        if not args.quiet_auto_rotmat:
            print("\n[auto_rotmat] injected staging rotmat candidates:")
            for pid, cands in rotmat_dict.items():
                print(f"  {pid}: {len(cands)} candidate(s)")
    else:
        print("[WARN] --no-auto-rotmat: L2 将使用 find_optimal_layout 里"
              " yuanchair 的 rotmat 键（shelf 零件会 fallback 到 I）")

    print(f"\n[Task] {task.name}")
    print(f"  asmdef  = {os.path.relpath(task.asmdef_path)}")
    print(f"  parts   = {task.part_ids}")
    print(f"  output  = {task.output_layout_name}.layout")
    print(f"  weights = Grasp*{WEIGHT_GRASP} + Manip_EP*{WEIGHT_MANIP_EP} "
          f"+ Manip_TRAJ*{WEIGHT_MANIP_TRAJ} + Dist*{WEIGHT_DIST} "
          f"(sum={WEIGHT_GRASP + WEIGHT_MANIP_EP + WEIGHT_MANIP_TRAJ + WEIGHT_DIST:.2f})")

    # ── 2. 复用 find_optimal_layout 的搜索引擎 ──────────────────────────
    out_path, solution = run_fast_search(
        task,
        mode=args.mode,
        n_samples=args.n_samples,
        pop_size=args.pop,
        generations=args.gens,
        rng_seed=args.seed,
        enable_l3=args.enable_l3,
        try_l3_top_k=3 if args.enable_l3 else 0,
        ik_retry_n=ik_retry,
    )

    # 调试阶段强约束：开启 --enable-l3 时，只有 L3 真正通过才保留 layout。
    # 原搜索引擎会在 L3 全失败时回退保存 L2 最高分 layout，动画端仍可能失败；
    # 这里主动删除 L2-only 结果，避免误以为 layout 已经可执行。
    if args.enable_l3 and (solution is None or not getattr(solution, "l3_pass", False)):
        print("\n[STRICT-L3] L3 未通过：删除本次保存的 L2-only layout，避免执行端误用。")
        candidates = []
        if out_path:
            candidates.append(out_path)
            candidates.append(os.path.splitext(out_path)[0] + "_diagnostics.json")
        out_dir = os.path.join(os.path.dirname(__file__), "_output")
        candidates.extend([
            os.path.join(out_dir, f"{task.output_layout_name}.layout"),
            os.path.join(out_dir, f"{task.output_layout_name}_diagnostics.json"),
            os.path.join(out_dir, f"{task.output_layout_name}_l2only.layout"),
            os.path.join(out_dir, f"{task.output_layout_name}_l2only_diagnostics.json"),
        ])
        seen = set()
        for fp in candidates:
            if not fp or fp in seen:
                continue
            seen.add(fp)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                    print(f"  deleted: {os.path.relpath(fp)}")
                except Exception as exc:
                    print(f"  [WARN] 删除失败 {fp}: {exc!r}")
        print("[STRICT-L3] 请增大 --n-samples / 换 --seed / 调整 fixture_pos 或 place_depart 后重跑。")


if __name__ == "__main__":
    main()