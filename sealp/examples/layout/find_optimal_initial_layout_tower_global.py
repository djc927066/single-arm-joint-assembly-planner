"""全局布局搜索 (Global Layout Search).

目的
----
现有 NSGA-II (find_optimal_initial_layout_tower_nsga2_v1) 在很小的评估预算下
基本等价于"随机采样 + 极小幅局部变异", 桌面这么大的连续 XY 空间根本没被充分
覆盖, 所以经常"更好分数的位置没被找到"。

本脚本在 **完全不改动现有脚本** 的前提下, 换一套更"全局 + 精细"的搜索算法:

  Phase A — 空间填充式全局探索 (explore)
      对全部装配区(3x3 中心)做 round-robin 覆盖, 每个区都用现成的
      sample_collision_free_xy 采样若干个无碰撞初始布局并评估, 取分数最高的
      若干个作为精修种子。这样保证每个装配中心、整张桌面都被均匀采到,
      而不是像 NSGA 那样随机挑区、样本全挤在少数区域。

  Phase B — 模式搜索式局部精修 (pattern / coordinate refine)
      对每个精英布局, 逐零件在 XY 上按"由粗到细"的步长(如 3cm->1.5cm->...)
      尝试 ±步长 的邻域, 贪心接受能提升分数的移动, 直到某一轮不再改进。
      这一步专门把"NSGA 随机变异错过的、附近更优的位置"抠出来 —— 也就是
      你要的"更精细"。

实现方式
--------
直接复用 NSGA2LayoutSearcher (它已经封装好 evaluate_layout 调用、评估缓存、
装配区处理、目标向量、L3 钩子)。本脚本只替换 random_search 这一个方法,
因此所有约束、打分项、抓取校验、障碍口径都与现有流程 **完全一致**。

用法
----
python -m sealp.examples.layout.find_optimal_initial_layout_tower_global \
    --n-samples 60 --cdprim-type box --output-name tower_global \
    --global-explore 60 --global-elite 3 \
    --global-refine-steps 0.03,0.015,0.008 --global-refine-rounds 2 \
    --global-max-evals 300

说明:
  --n-samples          兼容 fol 主流程; 若未单独给 --global-explore, 则用它做探索样本数。
  --global-explore N   Phase A 探索评估次数(覆盖全部装配区)。
  --global-elite K     取分数最高的 K 个布局进入 Phase B 精修。
  --global-refine-steps  精修步长(米), 逗号分隔, 由大到小。
  --global-refine-rounds 每个步长最多扫描几轮(某轮无改进即提前进入更小步长)。
  --global-refine-diagonal 额外尝试 4 个对角方向(更细但更慢)。
  --no-refine            跳过 Phase B 精修, 直接用 Phase A 最优可行 layout (大幅加速)。
  --global-max-evals M   evaluate_layout 总次数硬上限(强烈建议设置; 每次约 30~100s)。

其余参数(--asmdef/--config/--grasp-dir/--planner-obstacle-mode/--enable-l3/
--w-stl-upface 等) 全部与 find_optimal_initial_layout_tower_strict_pycharm 一致。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 让 `python -m ...` 运行时也能 import 同目录的兄弟模块(与 nsga2_v1 一致)。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2

LayoutCandidate = fol.LayoutCandidate


# 全局搜索配置(可被 --global-* 覆盖)
GCFG: Dict[str, object] = {
    "explore": None,             # Phase A 探索样本数; None -> 用 --n-samples
    "elite": 3,                  # 进入精修的精英数
    "refine_steps": [0.03, 0.015, 0.008],  # 精修步长(米), 由粗到细
    "refine_rounds": 2,          # 每个步长最多扫描轮数
    "refine_diagonal": False,    # 是否额外尝试对角方向
    "refine_enabled": True,      # False -> 跳过 Phase B (--no-refine)
    "max_resample_layout": 80,   # 单个布局无碰撞采样的最大重试次数
}


def _offsets(step: float, diagonal: bool) -> List[Tuple[float, float]]:
    base = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]
    if diagonal:
        base += [(step, step), (step, -step), (-step, step), (-step, -step)]
    return base


class GlobalLayoutSearcher(nsga2.NSGA2LayoutSearcher):
    """全局探索 + 局部精修搜索器。

    复用 NSGA2LayoutSearcher 的 evaluate/cache/region/L3 逻辑, 只换 random_search。
    """

    # ---------- Phase A: 全局探索 ----------
    def _global_explore(self,
                        rng: np.random.Generator,
                        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
                        n_explore: int,
                        verbose: bool) -> List[LayoutCandidate]:
        feasible: List[LayoutCandidate] = []
        evaluated = 0
        attempts = 0
        max_attempts = n_explore * 4 + 20
        i = 0
        while evaluated < n_explore and attempts < max_attempts:
            if self._eval_budget_exhausted():
                print("[global] explore 停止: 达到 --global-max-evals。")
                break
            attempts += 1
            # round-robin 覆盖每个装配区, 保证全桌面/全装配中心均匀采样。
            region = regions[i % len(regions)]
            i += 1
            self._set_region_from_tuple(region)
            xy = None
            for _ in range(int(GCFG["max_resample_layout"])):
                xy = self.sample_collision_free_xy(rng)
                if xy is not None:
                    break
            if xy is None:
                continue
            cand = self._evaluate_gene(xy, region)
            evaluated += 1
            if verbose:
                tag = "L2_OK" if cand.l2_pass else "FAIL"
                score = float(getattr(cand, "layout_score", -1.0))
                print(f"[explore] {evaluated:03d}/{n_explore} {tag:5s} "
                      f"score={score:.4f} region={cand.assembly_region_id}")
                if not cand.l2_pass:
                    print(f"          fail: {getattr(cand, 'fail_reason', '?')}")
            if cand.l2_pass:
                feasible.append(cand)
        return feasible

    # ---------- Phase B: 局部模式搜索精修 ----------
    def _pattern_refine(self,
                        cand: LayoutCandidate,
                        steps: Sequence[float],
                        rounds: int,
                        diagonal: bool,
                        verbose: bool) -> LayoutCandidate:
        region = getattr(cand, "_nsga_region_tuple", None)
        if region is None:
            region = self._region_by_id(self._assembly_region_candidates(), cand.assembly_region_id)

        first_pid = self._first_part_id() if self.preassemble_first_part else None
        free_parts = [p for p in self.part_order if p != first_pid]

        best = cand
        best_xy = nsga2._copy_xy(cand.xy)
        eps = 1e-4
        total_improved = 0.0

        for step in steps:
            for _r in range(max(1, rounds)):
                improved = False
                for pid in free_parts:
                    if pid not in best_xy:
                        continue
                    if self._eval_budget_exhausted():
                        if verbose:
                            print("[refine] 停止: 达到 --global-max-evals。")
                        return best
                    anchor = np.asarray(best_xy[pid], dtype=float)
                    for dx, dy in _offsets(step, diagonal):
                        trial_xy = nsga2._copy_xy(best_xy)
                        trial_xy[pid] = self._clip_xy_for_part(
                            pid, anchor + np.array([dx, dy], dtype=float))
                        child = self._evaluate_gene(trial_xy, region)
                        if bool(getattr(child, "l2_pass", False)) and \
                                float(child.layout_score) > float(best.layout_score) + eps:
                            total_improved += float(child.layout_score) - float(best.layout_score)
                            best = child
                            best_xy = nsga2._copy_xy(child.xy)
                            anchor = np.asarray(best_xy[pid], dtype=float)
                            improved = True
                if not improved:
                    break  # 该步长已收敛, 进入更小步长
        if verbose:
            print(f"[refine] region={best.assembly_region_id} "
                  f"score {float(cand.layout_score):.4f} -> {float(best.layout_score):.4f} "
                  f"(+{total_improved:.4f})")
        return best

    # ---------- 装配区扫描顺序: 中间优先 ----------
    def _order_regions_center_first(
        self,
        regions: Sequence[Tuple[str, Tuple[int, int], np.ndarray]],
        verbose: bool = True,
    ) -> List[Tuple[str, Tuple[int, int], np.ndarray]]:
        """把装配区候选按"离工作区中心由近到远"排序, 让探索优先扫描中间区域。

        与网格划分方式无关: 只用每个候选中心的 (x, y) 到参考中心的距离排序,
        因此即使以后不再是 3x3 网格(改成任意点集/更细网格/非均匀采样)也照样
        "中间优先, 逐步向外", 而不是从某个角落开始。

        参考中心:
            y -> 左右臂基座 y 的中点(双臂可达性最佳的 y 带; 取不到则退回候选 y 均值);
            x -> 所有候选中心 x 的均值(桌面前后方向的几何中点)。
        """
        regions = list(regions)
        if len(regions) <= 1:
            return regions

        pts = np.array([[float(p[0]), float(p[1])] for _, _, p in regions], dtype=float)
        ref_x = float(np.mean(pts[:, 0]))
        try:
            arm_ys = [float(xy[1]) for xy in self._arm_base_xy_map().values()]
            ref_y = float(np.mean(arm_ys)) if arm_ys else float(np.mean(pts[:, 1]))
        except Exception:
            ref_y = float(np.mean(pts[:, 1]))
        ref = np.array([ref_x, ref_y], dtype=float)

        d2 = np.sum((pts - ref) ** 2, axis=1)
        # 距离相同(如对称角落)时按原顺序稳定排序, 保证结果可复现。
        order = sorted(range(len(regions)), key=lambda i: (float(d2[i]), i))
        ordered = [regions[i] for i in order]

        if verbose:
            print("\n---------- Explore order: center-first ----------")
            print(f"reference center (x, y) = ({ref_x:.4f}, {ref_y:.4f})  "
                  f"# x=候选中心均值, y=双臂基座中点")
            for rank, i in enumerate(order, 1):
                rid, rc, p = regions[i]
                print(f"  #{rank:2d} {rid:6s} rc={rc} "
                      f"center=({float(p[0]):.4f}, {float(p[1]):.4f}) "
                      f"dist={float(np.sqrt(d2[i])):.4f}")
        return ordered

    # ---------- 主入口: 替换 NSGA 的 random_search ----------
    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "staging_aware",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        rng = np.random.default_rng(seed)
        self._reset_eval_progress_stats()
        regions = self._assembly_region_candidates()
        # 中间优先: 让 round-robin 从最靠工作区中心的装配区开始扫, 角落最后扫,
        # 避免前几次评估浪费在难摆/够不到的角落上。
        regions = self._order_regions_center_first(regions, verbose=verbose)
        GCFG["max_resample_layout"] = int(max_resample_layout)

        n_explore = int(GCFG["explore"] or n_samples)
        elite_k = max(1, int(GCFG["elite"]))
        steps = [float(s) for s in GCFG["refine_steps"]]  # type: ignore[union-attr]
        rounds = int(GCFG["refine_rounds"])
        diagonal = bool(GCFG["refine_diagonal"])
        refine_enabled = bool(GCFG["refine_enabled"]) and len(steps) > 0 and rounds > 0

        print("\n========== Global Layout Search ==========")
        print(f"explore samples   = {n_explore}  (center-first round-robin over {len(regions)} assembly regions)")
        print(f"elite (refine)    = {elite_k}")
        print(f"refine enabled    = {refine_enabled}")
        print(f"refine steps (m)  = {steps if refine_enabled else 'SKIPPED'}")
        print(f"refine rounds     = {rounds if refine_enabled else 'SKIPPED'}, diagonal = {diagonal}")
        print(f"max_evals         = {nsga2._CFG.get('max_evals')}")
        print(f"objective         = analytic layout_score (与 nsga2 BEST-L2 同口径)")
        print(f"L3 default/current = {'ON' if enable_l3 else 'OFF'}")

        t0 = time.time()

        # Phase A: 全局探索
        print("\n---------- Phase A: global explore ----------")
        feasible = self._global_explore(rng, regions, n_explore, verbose)
        if not feasible:
            print("\n[FAIL] Phase A 未找到任何 L2 可行布局。请放宽约束或增大 --global-explore。")
            return None

        elites = self._unique_elites(feasible, limit=max(elite_k, int(l3_top_k)))
        print(f"\nPhase A 完成: 可行 {len(feasible)} 个, 去重精英 {len(elites)} 个。")
        for r, c in enumerate(elites[:elite_k], 1):
            print(f"  elite#{r} score={c.layout_score:.4f} region={c.assembly_region_id}")

        # Phase B: 对每个精英做局部精修
        refined: List[LayoutCandidate] = []
        if refine_enabled:
            print("\n---------- Phase B: pattern refine ----------")
            for r, c in enumerate(elites[:elite_k], 1):
                if self._eval_budget_exhausted():
                    print("[global] 精修阶段停止: 达到 --global-max-evals。")
                    refined.append(c)
                    continue
                print(f"[refine] elite#{r} (start score={c.layout_score:.4f}) ...")
                refined.append(self._pattern_refine(c, steps, rounds, diagonal, verbose))
        else:
            print("\n---------- Phase B: SKIPPED (--no-refine) ----------")

        # 汇总所有候选(探索可行 + 精修结果), 取分数最高。
        pool = list(feasible) + list(refined)
        pool = [c for c in pool if bool(getattr(c, "l2_pass", False))]
        all_elites = self._unique_elites(pool, limit=max(int(GCFG["elite"]), int(l3_top_k)))
        if not all_elites:
            print("\n[FAIL] 无 L2 可行布局。")
            return None
        elites_by_score = sorted(all_elites, key=lambda c: c.layout_score, reverse=True)
        best = elites_by_score[0]

        print("\n========== Global Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"real evaluations    = {self._nsga_eval_count}")
        print(f"eval cache hits     = {self._nsga_cache_hits}")
        print(f"feasible found      = {len(feasible)}")
        self.print_search_eval_progress()
        print(f"[BEST-L2] score={best.layout_score:.4f} region={best.assembly_region_id} rc={best.assembly_region_rc}")
        print(f"  objectives  ={np.round(getattr(best, '_nsga_objectives', []), 4).tolist()}")
        print(f"  grasp_counts={best.grasp_counts}")
        print(f"  arm_choice  ={best.arm_choice}")
        print(f"  pose_tag    ={best.pose_tag}")

        # 可选 L3 验证(与 nsga2 同口径; 默认 OFF)
        if enable_l3:
            print("\n========== Optional L3 full-process validation ==========")
            k = min(int(l3_top_k), len(elites_by_score))
            for rank, cand in enumerate(elites_by_score[:k], start=1):
                print(f"[L3] rank {rank}/{k} score={cand.layout_score:.4f} region={cand.assembly_region_id}")
                if self.validate_full_sequence_l3(cand, obstacle_mode=l3_obstacle_mode, verbose=True):
                    print(f"[OK] L3 passed rank={rank}")
                    return cand
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")
            if require_l3:
                print("\n[FAIL] L3 top-k all failed.")
                return None
            print("\n[WARN] L3 failed, require_l3=False, fallback to L2 best.")

        return best


# ============================================================
# Entry point
# ============================================================

def _consume_global_args() -> None:
    v = fast._consume_extra_value("--global-explore")
    if v is not None:
        GCFG["explore"] = int(v)
    v = fast._consume_extra_value("--global-elite")
    if v is not None:
        GCFG["elite"] = int(v)
    v = fast._consume_extra_value("--global-refine-rounds")
    if v is not None:
        GCFG["refine_rounds"] = int(v)
    v = fast._consume_extra_value("--global-refine-steps")
    if v is not None:
        v_norm = str(v).strip().lower()
        if v_norm in ("none", "off", "skip", "0"):
            GCFG["refine_enabled"] = False
        else:
            try:
                GCFG["refine_steps"] = [float(s) for s in v.replace(",", " ").split()]
            except ValueError:
                print(f"[global] WARN: 无法解析 --global-refine-steps '{v}', 保留默认。")
    v = fast._consume_extra_value("--global-max-evals")
    if v is not None:
        try:
            nsga2._CFG["max_evals"] = int(v)
        except ValueError:
            print(f"[global] WARN: 无法解析 --global-max-evals '{v}', 忽略。")
    if fast._consume_extra_flag("--global-refine-diagonal"):
        GCFG["refine_diagonal"] = True
    if fast._consume_extra_flag("--no-refine"):
        GCFG["refine_enabled"] = False


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = GlobalLayoutSearcher


def main() -> None:
    # L3 默认关闭 + 默认跳过 middle_plate 的 L3, 与 nsga2 行为一致。
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()
    _consume_global_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    print("[global] config:")
    for k, val in GCFG.items():
        print(f"    {k:20s} = {val}")
    print(f"    {'max_evals':20s} = {nsga2._CFG.get('max_evals')}")
    print(f"[global] scipy.cKDTree available = {fast._HAS_KDTREE}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[global] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
