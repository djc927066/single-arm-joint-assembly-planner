"""神经网络辅助全局->局部布局搜索 (Neural-Guided Global-to-Local Layout Search)。

基于 find_optimal_initial_layout_tower_global.py 改造, **不破坏原脚本**。

核心原则:
    - 神经网络只做 proposal / pre-filter / ranking / score prediction;
    - 神经网络绝不决定最终可行性;
    - 所有候选最终都必须经过原始 evaluate_layout + pattern_refine + L3 (motion-level);
    - NN 产出的可行 layout 不足时, fallback 到原始 GlobalLayoutSearcher._global_explore。

流程:
    1. 加载训练好的模型 checkpoint;
    2. 获取 assembly regions / part_order / workspace bounds / part features;
    3. NN 生成 (generator) 或筛选 (scorer) top-K candidate layouts;
    4. 对 top-K 调用原始 evaluate_layout;
    5. feasible 加入 elite pool;
    6. NN feasible 不足则 fallback 到原始 global explore;
    7. 对 elite pool 调用原来的 pattern_refine;
    8. 排序; 9. motion-level (L3) validation; 10. 保存最终 layout。

用法示例:
    python -m sealp.examples.layout.find_optimal_initial_layout_tower_neural \
        --model sagpn --checkpoint checkpoints/layout_models/sagpn_best.pt \
        --top-k-proposals 64 --global-elite 5 \
        --global-refine-steps 0.03,0.015,0.008 --global-max-evals 300 \
        --output-name tower_neural_sagpn
    # baseline: --model mlp/deepsets/gcn/gat/cvae/diffusion + 对应 checkpoint
    # footprint-aware coarse-to-fine scorer example:
    #   --candidate-pool-mode coarse_to_fine --scorer-pool 400 \
    #   --coarse-grid-spacing 0.10 --refine-top-k 8 \
    #   --fine-grid-spacing 0.025 --fine-xy-radius 0.025 \
    #   --fine-variants-per-layout 16
    # 加速: --no-refine 跳过 Phase B; 或 --global-refine-steps 0.03 --global-refine-rounds 1 --global-elite 1
    #
    # 装配站采样模式 (--station-mode):
    #   continuous (默认) 连续可行域采样, SAGPN 回归站位 + 邻域抖动;
    #   grid3x3          复用原始 3x3 网格 center-first 站位 (SAGPN 仍在各站位做零件提案),
    #                    便于与 find_optimal_initial_layout_tower_global.py 在同口径下公平对比。
"""

from __future__ import annotations

import os
import sys
import time
import hashlib
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod
import generate_layout_dataset as gends

from layout_learning.infer import LayoutModelRunner
from uniform_candidate_pool import (
    build_uniform_candidate_pool,
)
from coarse_to_fine_candidate_pool import (
    build_footprint_coarse_pool,
    build_local_fine_pool,
)
from footprint_grid_candidate_pool import (
    build_footprint_grid_candidate_pool,
    decorate_candidate_with_fixed_poses,
    lock_fixed_poses_from_debug,
)

LayoutCandidate = fol.LayoutCandidate


class _RandomPoolRunner:
    """No-model baseline that randomly orders the exact shared pool."""

    model_name = "global_random_pool"
    is_generator = False

    @staticmethod
    def score_layouts(samples):
        count = len(samples)
        return {
            "feas_prob": np.zeros(count, dtype=float),
            "score": np.zeros(count, dtype=float),
        }

# 神经搜索配置
NCFG: Dict[str, object] = {
    "model": "sagpn",
    "checkpoint": None,
    "top_k_proposals": 64,      # 交给 evaluate_layout 的候选 layout 数量
    "scorer_pool": 0,           # 0=按桌面均匀网格自动生成候选池大小
    "pool_grid_spacing": 0.11,  # 均匀网格间距 (m); 越小候选越多
    "pool_max_candidates": 800,
    "pool_pose_variants": 2,    # 每个空间布局尝试前 N 个可行 flatsurface 姿态
    "min_feasible": 1,          # NN 找到的可行 layout 少于此数则触发 fallback
    "fallback_explore": 40,     # fallback 时 global explore 的评估次数
    "device": None,
    "feas_prob_min": 0.0,       # scorer: 过滤掉预测可行概率过低的候选 (0=不过滤)
    # 装配站采样模式:
    #   "continuous" -> 连续可行域采样 (SAGPN 回归站位 + 邻域抖动);
    #   "grid3x3"    -> 复用原始 3x3 网格 center-first 站位 (SAGPN 仍在各站位上做零件提案)。
    "station_mode": "continuous",
    # scorer 排序方式:
    #   "tuple" -> 先按 (feas_prob>=min, feas_prob, score) 字典序 (默认, 与旧行为一致);
    #   "blend" -> 按 rank_score = w*feas_prob + (1-w)*normalized_score 排序 (seqrel 推荐)。
    "rank_mode": "tuple",
    "rank_blend_feas": 0.7,     # blend 模式下 feasibility 概率的权重
    "candidate_pool_file": None,
    "candidate_pool_seed": 0,
    "eval_curve_out": None,
    "benchmark_strict": False,
    "random_pool_order": False,
    "active_search_seed": 0,
    # Candidate-pool mode. "auto" preserves the original behavior:
    # explicit scorer_pool -> legacy random; scorer_pool=0 -> old uniform-grid pool.
    # "coarse_to_fine" enables the footprint-aware two-stage pool.
    "candidate_pool_mode": "auto",
    "coarse_grid_spacing": 0.10,
    "coarse_margin": 0.01,
    "coarse_jitter_ratio": 0.0,
    "coarse_max_attempts_per_layout": 80,
    "refine_top_k": 8,
    "fine_grid_spacing": 0.025,
    "fine_xy_radius": 0.025,
    "fine_variants_per_layout": 16,
    "fine_pool_file": None,
    "center_prioritized_pool": False,
    # Fixed-pose 3 cm footprint-grid pool.
    "footprint_grid_spacing": 0.03,
    "footprint_grid_clearance": 0.01,
    "fixed_pose_debug_json": None,
    "grid_macro_rows": 6,
    "grid_macro_cols": 4,
    "grid_max_attempts_per_layout": 120,
    "grid_max_branches": 24,
}


def _build_cond(searcher, region: Tuple[str, Tuple[int, int], np.ndarray],
                xy: Optional[Dict[str, np.ndarray]] = None) -> Dict:
    """构造某装配区的条件 sample (供 NN 推理)。xy=None 表示只给静态条件。"""
    searcher._set_region_from_tuple(region)
    cand = LayoutCandidate(xy={k: np.asarray(v, float).copy() for k, v in (xy or {}).items()})
    cand.assembly_region_id = region[0]
    cand.assembly_region_rc = region[1]
    cand.assembly_station_pos = np.asarray(region[2], dtype=float)
    return gends.sample_from_candidate(
        searcher, cand, seed=0, region=region,
        sample_index=0, generation_signature="online_condition_v1")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_candidate_indices(
    feas_prob,
    score,
    feas_min: float,
    pool_seed: int,
    search_seed: int,
    stage_salt: int = 0,
    verbose: bool = False,
) -> List[int]:
    """Apply the existing tuple/blend/random ranking rule to one candidate set."""
    if bool(NCFG.get("random_pool_order")):
        return np.random.default_rng(
            np.random.SeedSequence([pool_seed, search_seed, 991, stage_salt])
        ).permutation(len(feas_prob)).tolist()

    rank_mode = str(NCFG.get("rank_mode", "tuple"))
    if rank_mode == "blend":
        w = float(NCFG.get("rank_blend_feas", 0.7))
        s = np.asarray(score, dtype=float)
        s_min, s_max = float(s.min()), float(s.max())
        s_norm = (s - s_min) / (s_max - s_min) if s_max - s_min > 1e-9 else np.zeros_like(s)
        rank_score = w * np.asarray(feas_prob, dtype=float) + (1.0 - w) * s_norm
        order = sorted(
            range(len(feas_prob)),
            key=lambda j: (feas_prob[j] >= feas_min, float(rank_score[j])),
            reverse=True,
        )
        if verbose:
            print(f"[score] rank_mode=blend w_feas={w:.2f}")
        return order

    return sorted(
        range(len(feas_prob)),
        key=lambda j: (feas_prob[j] >= feas_min, feas_prob[j], score[j]),
        reverse=True,
    )


def _center_prior_region_weights(regions: Sequence[Tuple]) -> Dict[str, float]:
    preferred = {
        "r1_c1": 0.50,
        "r1_c2": 0.15,
        "r1_c0": 0.15,
        "r2_c2": 0.05,
        "r0_c1": 0.05,
        "r0_c2": 0.05,
        "r0_c0": 0.05,
    }
    available = [str(region[0]) for region in regions]
    weights = {rid: preferred[rid] for rid in available if rid in preferred}
    missing = [rid for rid in available if rid not in weights]
    if missing:
        residual = max(0.05, 1.0 - sum(weights.values()))
        share = residual / float(len(missing))
        for rid in missing:
            weights[rid] = share
    return weights


def _select_center_prioritized_fine_anchors(
    order: Sequence[int],
    pool: Sequence[Tuple],
    refine_top_k: int,
) -> List[int]:
    refine_top_k = max(1, min(int(refine_top_k), len(order)))
    center_n = max(1, int(math.ceil(refine_top_k * 0.50)))
    remaining = refine_top_k - center_n
    right_n = int(math.ceil(remaining / 2.0))
    left_n = remaining - right_n
    quotas = [("r1_c1", center_n), ("r1_c2", right_n), ("r1_c0", left_n)]

    selected: List[int] = []
    used = set()
    for rid, quota in quotas:
        count = 0
        for j in order:
            idx = int(j)
            if idx in used or str(pool[idx][0][0]) != rid:
                continue
            selected.append(idx)
            used.add(idx)
            count += 1
            if count >= quota:
                break
    for j in order:
        idx = int(j)
        if idx in used:
            continue
        selected.append(idx)
        used.add(idx)
        if len(selected) >= refine_top_k:
            break
    return selected[:refine_top_k]


class NeuralGlobalSearcher(gmod.GlobalLayoutSearcher):
    """NN 辅助搜索器: 用 NN 产 top-K 候选, 交给原始 evaluate_layout, 保留 global fallback。"""

    _runner: Optional[LayoutModelRunner] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_pose_info: Dict[str, Dict[str, object]] = {}
        pool_mode = str(NCFG.get("candidate_pool_mode", "auto")).strip().lower()
        if pool_mode == "footprint_grid_balanced":
            debug_path = str(NCFG.get("fixed_pose_debug_json") or "").strip()
            if not debug_path:
                raise ValueError(
                    "footprint_grid_balanced requires --fixed-pose-debug-json"
                )
            self._fixed_pose_info = lock_fixed_poses_from_debug(
                self,
                debug_path,
                verbose=True,
            )

    # ---------- 连续装配站工具 (取代固定 3x3 网格) ----------
    def _sample_station(self, rng) -> Optional[Tuple]:
        """连续采样一个可行装配站 (region_id='cont', rc=(-1,-1))。"""
        return gends.sample_continuous_station(self, rng)

    def _sample_station_list(self, rng, n: int) -> List[Tuple]:
        out: List[Tuple] = []
        tries = 0
        while len(out) < n and tries < n * 5 + 20:
            tries += 1
            reg = self._sample_station(rng)
            if reg is not None:
                out.append(reg)
        return out

    def _station_region_from_xy(self, xy) -> Optional[Tuple]:
        """把一个 (可能越界的) 装配站 xy 收缩到可行域并校验, 返回 region 三元组。"""
        xs0, xs1, ys0, ys1 = gends.station_safe_bounds(self)
        x = float(np.clip(float(xy[0]), xs0, xs1))
        y = float(np.clip(float(xy[1]), ys0, ys1))
        pos = np.array([x, y, float(self.table_top_z)], dtype=float)
        if self._assembly_region_reject_reason(pos) is not None:
            return None
        return ("cont", (-1, -1), pos)

    def _sagpn_station_candidates(self, rng, n: int, verbose: bool) -> List[Tuple]:
        """SAGPN: 回归一个连续装配站, 再在其邻域抖动 + 少量随机站位, 兼顾利用与探索。"""
        runner = self._runner
        seed_reg = self._sample_station(rng)
        if seed_reg is None:
            return self._sample_station_list(rng, n)
        cond0 = _build_cond(self, seed_reg)
        pred = None
        try:
            pred = runner.predict_station(cond0)
        except Exception as e:
            print(f"[neural] predict_station 失败, 退回随机连续站位: {e!r}")
        if pred is None:
            return self._sample_station_list(rng, n)
        cands: List[Tuple] = []
        base = self._station_region_from_xy(pred)
        if base is not None:
            cands.append(base)
            if verbose:
                print(f"[gen] SAGPN predicted station = ({pred[0]:.4f}, {pred[1]:.4f})")
        # 邻域抖动 (利用)
        for _ in range(max(0, n // 2)):
            jit = np.asarray(pred, float) + rng.normal(0, 0.03, size=2)
            reg = self._station_region_from_xy(jit)
            if reg is not None:
                cands.append(reg)
        # 少量随机连续站位 (探索)
        cands += self._sample_station_list(rng, max(1, n - len(cands)))
        return cands[:max(1, n)]

    def _grid_station_candidates(self, verbose: bool) -> List[Tuple]:
        """3x3 网格站位: 复用原始 _assembly_region_candidates + center-first 排序。

        注意: 站位固定在网格中心, SAGPN 不再回归站位, 但仍会在**每个网格站位**上
        用 propose_layouts 生成零件布局 (即 SAGPN 依然做提案), 因此这只是把"站位
        搜索空间"从连续退回到离散 3x3, 便于与原始 global 基线在同口径下对比。
        """
        regions = self._assembly_region_candidates()
        return self._order_regions_center_first(regions, verbose=verbose)

    def _station_candidates(self, rng, n: int, prefer_sagpn: bool, verbose: bool) -> List[Tuple]:
        """按 station_mode 统一分发装配站候选。

        prefer_sagpn: 连续模式下, 若模型是 SAGPN 则用其回归站位 + 邻域抖动; 否则随机连续。
        grid3x3 模式忽略 prefer_sagpn (站位由网格决定, SAGPN 仅在站位上做零件提案)。
        """
        mode = str(NCFG.get("station_mode", "continuous"))
        if mode == "grid3x3":
            regions = self._grid_station_candidates(verbose=verbose)
            return regions[:max(1, n)] if regions else []
        # continuous
        runner = self._runner
        if prefer_sagpn and runner is not None and getattr(runner, "model_name", "") == "sagpn":
            return self._sagpn_station_candidates(rng, n, verbose)
        return self._sample_station_list(rng, n)

    # ---------- generator 路径 (连续装配站) ----------
    def _neural_generate(self, rng, verbose) -> List[LayoutCandidate]:
        runner = self._runner
        top_k = int(NCFG["top_k_proposals"])
        feasible: List[LayoutCandidate] = []
        evaluated = 0
        first_pid = self._first_part_id() if self.preassemble_first_part else None

        # 生成一批装配站候选: 由 station_mode 决定 (连续 / 3x3 网格)。
        # 连续 + SAGPN -> 回归站位+邻域; 连续 + 其它 -> 随机连续; grid3x3 -> 网格中心。
        n_stations = max(1, min(top_k, 16))
        stations = self._station_candidates(rng, n_stations, prefer_sagpn=True, verbose=verbose)
        if not stations:
            return feasible
        per_station = max(1, top_k // len(stations) + 1)

        for region in stations:
            if evaluated >= top_k or self._eval_budget_exhausted():
                break
            self._set_region_from_tuple(region)
            cond = _build_cond(self, region)
            station_xy = np.asarray(region[2], float)[:2]
            try:
                layouts = runner.propose_layouts(
                    cond, k=per_station, part_ids=list(self.part_order), station_xy=station_xy)
            except Exception as e:
                print(f"[neural] propose 失败 station={np.round(station_xy,3).tolist()}: {e!r}")
                continue
            for layout_xy in layouts:
                if evaluated >= top_k or self._eval_budget_exhausted():
                    break
                xy = {}
                for pid in self.part_order:
                    if pid == first_pid:
                        continue  # 第一件由 evaluate_layout 预装, 无需 proposal
                    if pid in layout_xy:
                        xy[pid] = self._clip_xy_for_part(pid, np.asarray(layout_xy[pid], float))
                if len(xy) < 1:
                    continue
                cand = self._evaluate_gene(xy, region)
                evaluated += 1
                if verbose:
                    tag = "L2_OK" if cand.l2_pass else "FAIL"
                    print(f"[gen] {evaluated:03d}/{top_k} {tag:5s} "
                          f"score={float(getattr(cand,'layout_score',-1)):.4f} "
                          f"station=({station_xy[0]:.3f},{station_xy[1]:.3f})")
                if cand.l2_pass:
                    feasible.append(cand)
        return feasible

    def _uniform_pool_region(self, verbose: bool) -> Optional[Tuple]:
        """Pick one assembly station (center-first) for uniform staging pool."""
        mode = str(NCFG.get("station_mode", "continuous"))
        if mode == "grid3x3":
            regions = self._grid_station_candidates(verbose=verbose)
            return regions[0] if regions else None
        regions = self._order_regions_center_first(
            self._assembly_region_candidates(), verbose=verbose)
        if regions:
            return regions[0]
        return self._sample_station(np.random.default_rng(0))

    def _build_scorer_candidate_pool(
        self,
        pool_seed: int,
        search_seed: int,
        verbose: bool,
    ) -> Tuple[List[Tuple], List[Dict], Dict]:
        """Uniform table pool + coarse filter; auto candidate count."""
        region = self._uniform_pool_region(verbose=verbose)
        if region is None:
            return [], [], {}
        spacing = float(NCFG.get("pool_grid_spacing", 0.11))
        max_n = int(NCFG.get("pool_max_candidates", 800))
        pose_variants = int(NCFG.get("pool_pose_variants", 2))
        if verbose:
            from uniform_candidate_pool import _auto_anchor_spacing, estimate_pool_size
            eff = _auto_anchor_spacing(self, spacing)
            est = estimate_pool_size(self, spacing)
            print(f"[score] uniform pool request_spacing={spacing:.3f}m "
                  f"effective_anchor_step~{eff:.3f}m (est upper bound ~{est})")
        layout_cands = build_uniform_candidate_pool(
            self, region, spacing=spacing,
            max_candidates=max_n, pose_variants_per_layout=pose_variants,
        )
        pool: List[Tuple] = []
        samples: List[Dict] = []
        entries = []
        for i, lc in enumerate(layout_cands):
            xy = {pid: np.asarray(lc.xy[pid], dtype=float) for pid in lc.xy}
            pool.append((region, xy))
            sample = gends.sample_from_candidate(
                self, lc, seed=pool_seed, region=region,
                sample_index=i, generation_signature="uniform_candidate_pool_v1")
            samples.append(sample)
            entries.append({
                "pool_index": i,
                "region_id": region[0],
                "region_rc": list(region[1]),
                "station_pos": np.asarray(region[2], dtype=float).tolist(),
                "xy": {k: np.asarray(v, dtype=float).tolist() for k, v in xy.items()},
                "pose_tag": dict(lc.pose_tag),
                "rot_name": dict(lc.rot_name),
                "sample": sample,
            })
        if verbose:
            print(f"[score] uniform pool kept {len(pool)} coarse-feasible layouts "
                  f"(poses/flatsurface checked per part)")
        meta = {
            "candidate_pool_seed": pool_seed,
            "search_seed": search_seed,
            "candidate_count": len(pool),
            "station_mode": str(NCFG["station_mode"]),
            "pool_grid_spacing": spacing,
            "pool_build_mode": "uniform_grid",
            "candidates": entries,
        }
        return pool, samples, meta

    # ---------- scorer path ----------
    def _neural_score_filter(self, rng, verbose) -> List[LayoutCandidate]:
        runner = self._runner
        top_k = int(NCFG["top_k_proposals"])
        pool_n = int(NCFG.get("scorer_pool", 0) or 0)
        feas_min = float(NCFG["feas_prob_min"])
        pool_mode = str(NCFG.get("candidate_pool_mode", "auto")).strip().lower()
        if pool_mode not in (
            "auto",
            "random_legacy",
            "uniform_grid",
            "coarse_to_fine",
            "footprint_grid_balanced",
        ):
            raise ValueError(f"unknown candidate_pool_mode: {pool_mode}")

        pool: List[Tuple[Tuple, Dict[str, np.ndarray]]] = []
        samples: List[Dict] = []
        provenance: List[Dict] = []
        pool_file = str(NCFG.get("candidate_pool_file") or "")
        pool_seed = int(NCFG.get("candidate_pool_seed", 0))
        search_seed = int(NCFG.get("active_search_seed", 0))
        pool_payload = None
        coarse_meta: Dict = {}
        fine_meta: Dict = {}

        def append_pool_entry(region, xy, generation_signature, entry_index, extra=None):
            cand = LayoutCandidate(xy={
                key: np.asarray(value, float).copy() for key, value in xy.items()
            })
            cand.assembly_region_id = region[0]
            cand.assembly_region_rc = region[1]
            cand.assembly_station_pos = np.asarray(region[2], float)
            if pool_mode == "footprint_grid_balanced":
                self._set_region_from_tuple(region)
                decorate_candidate_with_fixed_poses(self, cand)
            sample = gends.sample_from_candidate(
                self,
                cand,
                seed=pool_seed,
                region=region,
                sample_index=entry_index,
                generation_signature=generation_signature,
            )
            pool.append((region, cand.xy))
            samples.append(sample)
            provenance.append(dict(extra or {}))
            return sample

        # 1) Load a frozen coarse/shared pool, or construct a new one.
        if pool_file and os.path.isfile(pool_file):
            with open(pool_file, "r", encoding="utf-8") as stream:
                pool_payload = json.load(stream)
            expected = {
                "candidate_pool_seed": pool_seed,
                "search_seed": search_seed,
                "station_mode": str(NCFG["station_mode"]),
            }
            if pool_n > 0:
                expected["candidate_count"] = pool_n
            mismatches = {
                key: (pool_payload.get(key), value)
                for key, value in expected.items()
                if pool_payload.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"candidate pool metadata mismatch: {mismatches}")
            stored_mode = str(pool_payload.get("pool_build_mode", ""))
            if pool_mode == "coarse_to_fine" and stored_mode != "footprint_coarse_grid_prefilter_v2":
                raise RuntimeError(
                    "coarse_to_fine prefilter v2 requires a newly generated pool; "
                    f"got pool_build_mode={stored_mode!r}. Delete the old pool JSON and rerun."
                )
            if (
                pool_mode == "footprint_grid_balanced"
                and stored_mode != "footprint_grid_balanced_v1"
            ):
                raise RuntimeError(
                    "footprint_grid_balanced requires a matching pool; "
                    f"got pool_build_mode={stored_mode!r}. "
                    "Use a new candidate-pool file or delete the incompatible file."
                )
            if pool_mode == "footprint_grid_balanced":
                debug_path = os.path.abspath(
                    str(NCFG.get("fixed_pose_debug_json") or "")
                )
                expected_pose_hash = _sha256_file(debug_path)
                stored_pose_hash = str(
                    pool_payload.get("fixed_pose_debug_sha256", "")
                )
                if stored_pose_hash != expected_pose_hash:
                    raise RuntimeError(
                        "fixed-pose source mismatch: "
                        f"pool={stored_pose_hash!r}, current={expected_pose_hash!r}"
                    )
            for entry in pool_payload["candidates"]:
                region = (
                    str(entry["region_id"]),
                    tuple(int(v) for v in entry["region_rc"]),
                    np.asarray(entry["station_pos"], dtype=float),
                )
                xy = {
                    pid: np.asarray(value, dtype=float)
                    for pid, value in entry["xy"].items()
                }
                pool.append((region, xy))
                samples.append(entry["sample"])
                provenance.append({
                    "coarse_pool_index": int(entry.get("pool_index", len(pool) - 1)),
                    "source": "loaded_pool",
                })
            coarse_meta = {
                key: value for key, value in pool_payload.items() if key != "candidates"
            }
            if verbose:
                print(f"[score] loaded shared candidate pool: {pool_file} (n={len(pool)})")

        elif pool_mode == "footprint_grid_balanced":
            requested_n = pool_n if pool_n > 0 else int(
                NCFG.get("pool_max_candidates", 400)
            )
            if str(NCFG.get("station_mode")) != "grid3x3":
                raise ValueError(
                    "footprint_grid_balanced currently requires --station-mode grid3x3"
                )
            regions = self._grid_station_candidates(verbose=verbose)
            grid_seed = int(
                np.random.SeedSequence([pool_seed, search_seed, 314159])
                .generate_state(1)[0]
            )
            grid_pool, coarse_meta = build_footprint_grid_candidate_pool(
                self,
                regions=regions,
                pool_size=requested_n,
                cell_size=float(NCFG.get("footprint_grid_spacing", 0.03)),
                pair_clearance=float(
                    NCFG.get("footprint_grid_clearance", 0.01)
                ),
                seed=grid_seed,
                macro_rows=int(NCFG.get("grid_macro_rows", 6)),
                macro_cols=int(NCFG.get("grid_macro_cols", 4)),
                max_attempts_per_layout=int(
                    NCFG.get("grid_max_attempts_per_layout", 120)
                ),
                max_branches=int(NCFG.get("grid_max_branches", 24)),
                verbose=verbose,
            )
            entries = []
            for i, (region, xy) in enumerate(grid_pool):
                sample = append_pool_entry(
                    region,
                    xy,
                    generation_signature="footprint_grid_balanced_v1",
                    entry_index=i,
                    extra={
                        "coarse_pool_index": i,
                        "source": "footprint_grid_balanced",
                    },
                )
                entries.append({
                    "pool_index": i,
                    "region_id": str(region[0]),
                    "region_rc": list(region[1]),
                    "station_pos": np.asarray(region[2], dtype=float).tolist(),
                    "xy": {
                        key: np.asarray(value, dtype=float).tolist()
                        for key, value in xy.items()
                    },
                    "pose_tag": {
                        pid: str(self.rot_cands[pid][0].tag)
                        for pid in self._fixed_pose_info
                    },
                    "rot_name": {
                        pid: str(self.rot_cands[pid][0].rot_name)
                        for pid in self._fixed_pose_info
                    },
                    "sample": sample,
                })
            debug_path = os.path.abspath(
                str(NCFG.get("fixed_pose_debug_json") or "")
            )
            pool_payload = {
                "schema": 3,
                "candidate_pool_seed": pool_seed,
                "search_seed": search_seed,
                "candidate_count": len(pool),
                "station_mode": str(NCFG["station_mode"]),
                "fixed_pose_debug_json": debug_path,
                "fixed_pose_debug_sha256": _sha256_file(debug_path),
                **coarse_meta,
                "candidates": entries,
            }
            if pool_file and pool:
                os.makedirs(os.path.dirname(os.path.abspath(pool_file)), exist_ok=True)
                with open(pool_file, "x", encoding="utf-8") as stream:
                    json.dump(pool_payload, stream, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"[grid] saved fixed-pose balanced pool: {pool_file}")
            if verbose:
                print(
                    f"[grid] generated {len(pool)}/{requested_n} complete layouts; "
                    f"cell={float(NCFG.get('footprint_grid_spacing', 0.03)):.3f}m"
                )
                print(f"[grid] region targets = {coarse_meta.get('region_targets', {})}")
                print(f"[grid] region counts = {coarse_meta.get('region_counts', {})}")
                print(f"[grid] zone coverage = {coarse_meta.get('zone_coverage', {})}")
                print(
                    f"[grid] prefilter rejections = "
                    f"{coarse_meta.get('prefilter_rejections', {})}"
                )

        elif pool_mode == "coarse_to_fine":
            requested_n = pool_n if pool_n > 0 else 400
            pool_rng_seed = int(np.random.SeedSequence([pool_seed, search_seed, 1701]).generate_state(1)[0])
            if str(NCFG.get("station_mode")) == "grid3x3":
                regions = self._grid_station_candidates(verbose=verbose)
            else:
                regions = self._station_candidates(
                    np.random.default_rng(pool_rng_seed),
                    max(1, min(requested_n, 16)),
                    prefer_sagpn=False,
                    verbose=verbose,
                )
            region_weights = (
                _center_prior_region_weights(regions)
                if bool(NCFG.get("center_prioritized_pool")) else None
            )
            coarse_pool, coarse_meta = build_footprint_coarse_pool(
                self,
                regions=regions,
                pool_size=requested_n,
                spacing=float(NCFG.get("coarse_grid_spacing", 0.10)),
                margin=float(NCFG.get("coarse_margin", 0.01)),
                seed=pool_rng_seed,
                condition_builder=lambda searcher, region: _build_cond(searcher, region),
                jitter_ratio=float(NCFG.get("coarse_jitter_ratio", 0.0)),
                max_attempts_per_layout=int(NCFG.get("coarse_max_attempts_per_layout", 80)),
                region_weights=region_weights,
            )
            entries = []
            for i, (region, xy) in enumerate(coarse_pool):
                sample = append_pool_entry(
                    region,
                    xy,
                    generation_signature="footprint_coarse_grid_prefilter_v2",
                    entry_index=i,
                    extra={"coarse_pool_index": i, "source": "coarse_grid"},
                )
                entries.append({
                    "pool_index": i,
                    "region_id": region[0],
                    "region_rc": list(region[1]),
                    "station_pos": np.asarray(region[2], dtype=float).tolist(),
                    "xy": {key: np.asarray(value, dtype=float).tolist() for key, value in xy.items()},
                    "sample": sample,
                })
            pool_payload = {
                "schema": 2,
                "candidate_pool_seed": pool_seed,
                "search_seed": search_seed,
                "candidate_count": len(pool),
                "station_mode": str(NCFG["station_mode"]),
                **coarse_meta,
                "candidates": entries,
            }
            if pool_file and pool:
                os.makedirs(os.path.dirname(os.path.abspath(pool_file)), exist_ok=True)
                with open(pool_file, "x", encoding="utf-8") as stream:
                    json.dump(pool_payload, stream, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"[score] saved footprint-aware coarse pool: {pool_file}")
            if verbose:
                print(
                    f"[coarse] footprint grid kept {len(pool)}/{requested_n} layouts; "
                    f"spacing={float(NCFG.get('coarse_grid_spacing', 0.10)):.3f}m, "
                    f"margin={float(NCFG.get('coarse_margin', 0.01)):.3f}m"
                )
                print(f"[coarse] region targets = {coarse_meta.get('region_targets', {})}")
                print(f"[coarse] region counts = {coarse_meta.get('region_counts', {})}")
                print(
                    "[coarse] early prefilters = table_bounds, grid_overlap, "
                    "staging_arm_keepout, upright_constraint, aabb_clearance"
                )
                print(
                    f"[coarse] prefilter rejections = "
                    f"{coarse_meta.get('prefilter_rejections', {})}"
                )

        elif pool_n > 0 and pool_mode in ("auto", "random_legacy"):
            # Original behavior: explicit --scorer-pool uses constrained random sampling.
            pool_rng = np.random.default_rng(np.random.SeedSequence([pool_seed, search_seed]))
            n_stations = max(1, min(pool_n, 40))
            stations = self._station_candidates(
                pool_rng, n_stations, prefer_sagpn=False, verbose=verbose
            )
            if not stations:
                return []
            per_station = max(1, pool_n // len(stations) + 1)
            max_resample = int(gmod.GCFG["max_resample_layout"])
            entries = []
            for region in stations:
                if len(pool) >= pool_n:
                    break
                self._set_region_from_tuple(region)
                got = 0
                local_attempts = 0
                while (
                    got < per_station
                    and len(pool) < pool_n
                    and local_attempts < per_station * 4 + 20
                ):
                    local_attempts += 1
                    xy = None
                    for _ in range(max_resample):
                        xy = self.sample_collision_free_xy(pool_rng)
                        if xy is not None:
                            break
                    if xy is None:
                        continue
                    sample = append_pool_entry(
                        region,
                        xy,
                        generation_signature="online_candidate_pool_v1",
                        entry_index=len(pool),
                        extra={"coarse_pool_index": len(pool), "source": "random_legacy"},
                    )
                    entries.append({
                        "pool_index": len(pool) - 1,
                        "region_id": region[0],
                        "region_rc": list(region[1]),
                        "station_pos": np.asarray(region[2], dtype=float).tolist(),
                        "xy": {key: np.asarray(value, dtype=float).tolist() for key, value in xy.items()},
                        "sample": sample,
                    })
                    got += 1
            coarse_meta = {"pool_build_mode": "random_legacy"}
            if pool_file:
                os.makedirs(os.path.dirname(os.path.abspath(pool_file)), exist_ok=True)
                pool_payload = {
                    "schema": 1,
                    "candidate_pool_seed": pool_seed,
                    "search_seed": search_seed,
                    "candidate_count": len(pool),
                    "station_mode": str(NCFG["station_mode"]),
                    "pool_build_mode": "random_legacy",
                    "candidates": entries,
                }
                with open(pool_file, "x", encoding="utf-8") as stream:
                    json.dump(pool_payload, stream, ensure_ascii=False)
                if verbose:
                    print(f"[score] saved shared candidate pool: {pool_file}")

        else:
            # Original automatic uniform-grid implementation.
            pool, samples, pool_payload = self._build_scorer_candidate_pool(
                pool_seed, search_seed, verbose=verbose
            )
            provenance = [
                {"coarse_pool_index": i, "source": "uniform_grid"}
                for i in range(len(pool))
            ]
            coarse_meta = dict(pool_payload or {})
            if pool_file and pool:
                os.makedirs(os.path.dirname(os.path.abspath(pool_file)), exist_ok=True)
                with open(pool_file, "x", encoding="utf-8") as stream:
                    json.dump({"schema": 1, **pool_payload}, stream, ensure_ascii=False)
                if verbose:
                    print(f"[score] saved uniform candidate pool: {pool_file}")

        if not pool:
            return []
        coarse_candidate_count = len(pool)
        if bool(NCFG.get("benchmark_strict")) and pool_n > 0 and coarse_candidate_count != pool_n:
            raise RuntimeError(
                f"strict benchmark requires {pool_n} coarse candidates, got {coarse_candidate_count}"
            )

        # 2) Score and rank the coarse pool.
        inference_start = time.perf_counter()
        pred = runner.score_layouts(samples)
        inference_time = time.perf_counter() - inference_start
        feas_prob = np.asarray(pred["feas_prob"], dtype=float)
        score = np.asarray(pred["score"], dtype=float)
        order = _rank_candidate_indices(
            feas_prob,
            score,
            feas_min,
            pool_seed,
            search_seed,
            stage_salt=0,
            verbose=verbose,
        )

        # 3) Optional local fine-grid expansion around the best coarse layouts.
        if pool_mode == "coarse_to_fine":
            refine_top_k = max(1, min(int(NCFG.get("refine_top_k", 8)), len(order)))
            if bool(NCFG.get("center_prioritized_pool")):
                selected_indices = _select_center_prioritized_fine_anchors(
                    order, pool, refine_top_k)
            else:
                selected_indices = [int(j) for j in order[:refine_top_k]]
            selected = [
                (int(j), pool[j][0], pool[j][1]) for j in selected_indices
            ]
            if verbose and bool(NCFG.get("center_prioritized_pool")):
                selected_regions = [str(pool[j][0][0]) for j in selected_indices]
                print(f"[fine] center-prioritized anchors = {selected_regions}")
            fine_pool, fine_provenance, fine_meta = build_local_fine_pool(
                self,
                selected=selected,
                fine_spacing=float(NCFG.get("fine_grid_spacing", 0.025)),
                xy_radius=float(NCFG.get("fine_xy_radius", 0.025)),
                variants_per_layout=int(NCFG.get("fine_variants_per_layout", 16)),
                margin=float(NCFG.get("coarse_margin", 0.01)),
                seed=int(np.random.SeedSequence([pool_seed, search_seed, 2718]).generate_state(1)[0]),
                condition_builder=lambda searcher, region: _build_cond(searcher, region),
            )
            if fine_pool:
                pool = []
                samples = []
                provenance = []
                fine_entries = []
                for i, ((region, xy), prov) in enumerate(zip(fine_pool, fine_provenance)):
                    sample = append_pool_entry(
                        region,
                        xy,
                        generation_signature="coarse_to_fine_local_prefilter_v2",
                        entry_index=i,
                        extra={**prov, "source": "fine_grid"},
                    )
                    fine_entries.append({
                        "pool_index": i,
                        "region_id": region[0],
                        "region_rc": list(region[1]),
                        "station_pos": np.asarray(region[2], dtype=float).tolist(),
                        "xy": {key: np.asarray(value, dtype=float).tolist() for key, value in xy.items()},
                        "provenance": prov,
                        "sample": sample,
                    })
                fine_start = time.perf_counter()
                fine_pred = runner.score_layouts(samples)
                inference_time += time.perf_counter() - fine_start
                feas_prob = np.asarray(fine_pred["feas_prob"], dtype=float)
                score = np.asarray(fine_pred["score"], dtype=float)
                order = _rank_candidate_indices(
                    feas_prob,
                    score,
                    feas_min,
                    pool_seed,
                    search_seed,
                    stage_salt=1,
                    verbose=verbose,
                )
                fine_pool_file = str(NCFG.get("fine_pool_file") or "")
                if fine_pool_file:
                    os.makedirs(os.path.dirname(os.path.abspath(fine_pool_file)), exist_ok=True)
                    with open(fine_pool_file, "x", encoding="utf-8") as stream:
                        json.dump({
                            "schema": 1,
                            "candidate_pool_seed": pool_seed,
                            "search_seed": search_seed,
                            "station_mode": str(NCFG["station_mode"]),
                            "coarse_candidate_count": coarse_candidate_count,
                            **fine_meta,
                            "candidates": fine_entries,
                        }, stream, ensure_ascii=False, indent=2)
                if verbose:
                    print(
                        f"[fine] refined top-{refine_top_k} coarse layouts -> {len(pool)} local candidates; "
                        f"spacing={float(NCFG.get('fine_grid_spacing', 0.025)):.3f}m, "
                        f"radius=±{float(NCFG.get('fine_xy_radius', 0.025)):.3f}m"
                    )
                    print(
                        f"[fine] prefilter rejections = "
                        f"{fine_meta.get('prefilter_rejections', {})}"
                    )
            elif verbose:
                print("[fine] no valid local candidates; falling back to coarse ranking")

        # 4) Send final top-K candidates to the original exact evaluator.
        feasible: List[LayoutCandidate] = []
        evaluated = 0
        curve = []
        running_best = 0.0
        search_start = time.perf_counter()
        for j in order:
            if evaluated >= top_k or self._eval_budget_exhausted():
                break
            if feas_min > 0 and feas_prob[j] < feas_min:
                break
            region, xy = pool[j]
            before_real = int(self._nsga_eval_count)
            eval_start = time.perf_counter()
            cand = self._evaluate_gene(xy, region)
            eval_time = time.perf_counter() - eval_start
            after_real = int(self._nsga_eval_count)
            evaluated += 1
            true_score = float(getattr(cand, "layout_score", 0.0))
            if bool(getattr(cand, "l2_pass", False)):
                running_best = max(running_best, true_score)
            prov = provenance[j] if j < len(provenance) else {}
            curve.append({
                "eval_index": evaluated,
                "real_eval_index": after_real,
                "was_real_evaluation": after_real > before_real,
                "pool_index": int(j),
                "neural_rank": evaluated,
                "predicted_feasibility": float(feas_prob[j]),
                "predicted_score": float(score[j]),
                "true_feasible": bool(getattr(cand, "l2_pass", False)),
                "true_score": true_score,
                "running_best_true_score": running_best,
                "evaluation_time_seconds": eval_time,
                "elapsed_seconds": time.perf_counter() - search_start,
                "failure_reason": str(getattr(cand, "fail_reason", "") or ""),
                "region_id": str(region[0]),
                "region_rc": list(region[1]),
                "station_pos": np.asarray(region[2], dtype=float).tolist(),
                **prov,
            })
            if verbose:
                tag = "L2_OK" if cand.l2_pass else "FAIL"
                reason = "" if cand.l2_pass else f" reason={cand.fail_reason}"
                source_text = ""
                if "source_coarse_rank" in prov:
                    source_text = (
                        f" coarse_rank={prov['source_coarse_rank']}"
                        f" local={prov.get('local_variant_index', 0)}"
                    )
                print(
                    f"[score] {evaluated:03d}/{top_k} {tag:5s} "
                    f"nn_p={feas_prob[j]:.2f} nn_s={score[j]:.3f} "
                    f"real={float(getattr(cand, 'layout_score', -1)):.4f} "
                    f"region={cand.assembly_region_id}{source_text}{reason}"
                )
            if cand.l2_pass:
                feasible.append(cand)

        curve_out = str(NCFG.get("eval_curve_out") or "")
        if curve_out:
            os.makedirs(os.path.dirname(os.path.abspath(curve_out)), exist_ok=True)
            with open(curve_out, "w", encoding="utf-8") as stream:
                json.dump({
                    "schema": 2,
                    "checkpoint": str(NCFG.get("checkpoint") or ""),
                    "candidate_pool_mode": pool_mode,
                    "candidate_pool_file": os.path.abspath(pool_file) if pool_file else None,
                    "candidate_pool_sha256": _sha256_file(pool_file) if pool_file else None,
                    "coarse_candidate_count": coarse_candidate_count,
                    "final_candidate_count": len(pool),
                    "candidate_pool_seed": pool_seed,
                    "search_seed": search_seed,
                    "top_k_proposals": top_k,
                    "coarse_meta": coarse_meta,
                    "fine_meta": fine_meta,
                    "inference_time_seconds": inference_time,
                    "exact_real_evaluations": int(self._nsga_eval_count),
                    "curve": curve,
                }, stream, ensure_ascii=False, indent=2)
        return feasible

    # ---------- 主入口 ----------
    def random_search(self, n_samples, seed, max_resample_layout=80, verbose=True,
                      enable_l3=False, l3_top_k=3, l3_obstacle_mode="staging_aware",
                      require_l3=True) -> Optional[LayoutCandidate]:
        rng = np.random.default_rng(seed)
        NCFG["active_search_seed"] = int(seed)
        self._reset_eval_progress_stats()
        gmod.GCFG["max_resample_layout"] = int(max_resample_layout)

        runner = self._runner
        is_gen = bool(runner is not None and runner.is_generator)

        elite_k = max(1, int(gmod.GCFG["elite"]))
        steps = [float(s) for s in gmod.GCFG["refine_steps"]]
        rounds = int(gmod.GCFG["refine_rounds"])
        diagonal = bool(gmod.GCFG["refine_diagonal"])
        refine_enabled = bool(gmod.GCFG["refine_enabled"]) and len(steps) > 0 and rounds > 0

        print("\n========== Neural-Guided Layout Search ==========")
        print(f"model           = {NCFG['model']}  ({'generator' if is_gen else 'scorer'})")
        print(f"checkpoint      = {NCFG['checkpoint']}")
        print(f"top_k_proposals = {NCFG['top_k_proposals']}")
        pool_n = int(NCFG.get("scorer_pool", 0) or 0)
        pool_mode = str(NCFG.get("candidate_pool_mode", "auto"))
        if pool_mode == "footprint_grid_balanced":
            print(
                f"scorer_pool     = {pool_n if pool_n > 0 else NCFG.get('pool_max_candidates', 400)} "
                f"(fixed-pose balanced footprint grid, "
                f"cell={NCFG.get('footprint_grid_spacing', 0.03)}m)"
            )
            print(
                f"fixed poses     = {NCFG.get('fixed_pose_debug_json')}"
            )
        elif pool_mode == "coarse_to_fine":
            print(
                f"scorer_pool     = {pool_n if pool_n > 0 else 400} "
                f"(footprint coarse grid, spacing={NCFG.get('coarse_grid_spacing', 0.10)}m)"
            )
            print(
                f"fine refinement = coarse top-{NCFG.get('refine_top_k', 8)}, "
                f"step={NCFG.get('fine_grid_spacing', 0.025)}m, "
                f"radius=±{NCFG.get('fine_xy_radius', 0.025)}m, "
                f"variants={NCFG.get('fine_variants_per_layout', 16)}"
            )
        elif pool_n > 0:
            print(f"scorer_pool     = {pool_n} (legacy random)")
        else:
            print(f"scorer_pool     = auto uniform grid "
                  f"(spacing={NCFG.get('pool_grid_spacing', 0.11)}m, "
                  f"pose_variants={NCFG.get('pool_pose_variants', 2)})")
        print(f"pool_mode       = {pool_mode}")
        print(f"station_mode    = {NCFG['station_mode']}")
        print(f"elite (refine)  = {elite_k}, refine enabled={refine_enabled}")
        if refine_enabled:
            print(f"refine steps    = {steps}, rounds={rounds}, diagonal={diagonal}")
        else:
            print(f"refine steps    = SKIPPED (--no-refine)")
        print(f"max_evals       = {nsga2._CFG.get('max_evals')}")
        print(f"L3 default/cur  = {'ON' if enable_l3 else 'OFF'}")

        t0 = time.time()

        # Phase A': NN 生成/筛选 -> evaluate_layout
        print("\n---------- Phase A': neural proposal / filter ----------")
        if runner is None:
            print("[neural] WARN: 未加载模型, 直接 fallback 到 continuous explore。")
            feasible = []
        elif is_gen:
            feasible = self._neural_generate(rng, verbose)
        else:
            feasible = self._neural_score_filter(rng, verbose)
        print(f"[neural] NN 产出可行 layout = {len(feasible)}")

        # Phase A'' fallback: NN 可行不足 -> 原始 global explore (复用原逻辑, 但用连续装配站)
        if len(feasible) < int(NCFG["min_feasible"]) and not self._eval_budget_exhausted():
            print("\n---------- Fallback: original global explore (continuous stations) ----------")
            fb_regions = self._station_candidates(
                rng, max(1, int(NCFG["fallback_explore"])), prefer_sagpn=False, verbose=verbose)
            if not fb_regions:
                # 极端兜底: 采样失败时退回离散网格候选, 保证一定有 fallback。
                fb_regions = self._order_regions_center_first(
                    self._assembly_region_candidates(), verbose=verbose)
            fb = self._global_explore(rng, fb_regions, int(NCFG["fallback_explore"]), verbose)
            feasible = list(feasible) + list(fb)
            print(f"[neural] fallback 后可行 layout = {len(feasible)}")

        if not feasible:
            print("\n[FAIL] 未找到任何 L2 可行布局 (NN + fallback 均失败)。")
            return None

        elites = self._unique_elites(feasible, limit=max(elite_k, int(l3_top_k)))
        print(f"\nPhase A 完成: 可行 {len(feasible)} 个, 去重精英 {len(elites)} 个。")
        for r, c in enumerate(elites[:elite_k], 1):
            print(f"  elite#{r} score={c.layout_score:.4f} region={c.assembly_region_id}")

        # Phase B: pattern refine (原逻辑不变; 可用 --no-refine 跳过)
        refined: List[LayoutCandidate] = []
        if refine_enabled:
            print("\n---------- Phase B: pattern refine (original) ----------")
            for r, c in enumerate(elites[:elite_k], 1):
                if self._eval_budget_exhausted():
                    refined.append(c)
                    continue
                print(f"[refine] elite#{r} (start score={c.layout_score:.4f}) ...")
                refined.append(self._pattern_refine(c, steps, rounds, diagonal, verbose))
        else:
            print("\n---------- Phase B: SKIPPED (--no-refine) ----------")

        pool = [c for c in (list(feasible) + list(refined)) if bool(getattr(c, "l2_pass", False))]
        all_elites = self._unique_elites(pool, limit=max(int(gmod.GCFG["elite"]), int(l3_top_k)))
        if not all_elites:
            print("\n[FAIL] 无 L2 可行布局。")
            return None
        elites_by_score = sorted(all_elites, key=lambda c: c.layout_score, reverse=True)
        best = elites_by_score[0]

        print("\n========== Neural Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"real evaluations    = {self._nsga_eval_count}")
        print(f"eval cache hits     = {self._nsga_cache_hits}")
        print(f"feasible found      = {len(feasible)}")
        self.print_search_eval_progress()
        print(f"[BEST-L2] score={best.layout_score:.4f} region={best.assembly_region_id} rc={best.assembly_region_rc}")
        print(f"  grasp_counts={best.grasp_counts}")
        print(f"  arm_choice  ={best.arm_choice}")

        # motion-level (L3) validation (原逻辑不变)
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

def _consume_neural_args() -> None:
    v = fast._consume_extra_value("--model")
    if v is not None:
        NCFG["model"] = str(v)
    v = fast._consume_extra_value("--checkpoint")
    if v is not None:
        NCFG["checkpoint"] = str(v)
    v = fast._consume_extra_value("--top-k-proposals")
    if v is not None:
        NCFG["top_k_proposals"] = int(v)
    v = fast._consume_extra_value("--scorer-pool")
    if v is not None:
        NCFG["scorer_pool"] = int(v)
    v = fast._consume_extra_value("--pool-grid-spacing")
    if v is not None:
        NCFG["pool_grid_spacing"] = float(v)
    v = fast._consume_extra_value("--pool-pose-variants")
    if v is not None:
        NCFG["pool_pose_variants"] = int(v)
    v = fast._consume_extra_value("--pool-max-candidates")
    if v is not None:
        NCFG["pool_max_candidates"] = int(v)
    v = fast._consume_extra_value("--footprint-grid-spacing")
    if v is not None:
        NCFG["footprint_grid_spacing"] = float(v)
    v = fast._consume_extra_value("--footprint-grid-clearance")
    if v is not None:
        NCFG["footprint_grid_clearance"] = float(v)
    v = fast._consume_extra_value("--fixed-pose-debug-json")
    if v is not None:
        NCFG["fixed_pose_debug_json"] = str(v)
    v = fast._consume_extra_value("--grid-macro-rows")
    if v is not None:
        NCFG["grid_macro_rows"] = int(v)
    v = fast._consume_extra_value("--grid-macro-cols")
    if v is not None:
        NCFG["grid_macro_cols"] = int(v)
    v = fast._consume_extra_value("--grid-max-attempts-per-layout")
    if v is not None:
        NCFG["grid_max_attempts_per_layout"] = int(v)
    v = fast._consume_extra_value("--grid-max-branches")
    if v is not None:
        NCFG["grid_max_branches"] = int(v)
    v = fast._consume_extra_value("--min-feasible")
    if v is not None:
        NCFG["min_feasible"] = int(v)
    v = fast._consume_extra_value("--fallback-explore")
    if v is not None:
        NCFG["fallback_explore"] = int(v)
    v = fast._consume_extra_value("--nn-device")
    if v is not None:
        NCFG["device"] = str(v)
    v = fast._consume_extra_value("--feas-prob-min")
    if v is not None:
        NCFG["feas_prob_min"] = float(v)
    v = fast._consume_extra_value("--station-mode")
    if v is not None:
        mode = str(v).strip().lower()
        if mode not in ("continuous", "grid3x3"):
            print(f"[neural] WARN: 未知 --station-mode '{v}', 回退 continuous。")
            mode = "continuous"
        NCFG["station_mode"] = mode
    v = fast._consume_extra_value("--rank-mode")
    if v is not None:
        rm = str(v).strip().lower()
        if rm not in ("tuple", "blend"):
            print(f"[neural] WARN: 未知 --rank-mode '{v}', 回退 tuple。")
            rm = "tuple"
        NCFG["rank_mode"] = rm
    v = fast._consume_extra_value("--rank-blend-feas")
    if v is not None:
        NCFG["rank_blend_feas"] = float(v)
    v = fast._consume_extra_value("--candidate-pool-file")
    if v is not None:
        NCFG["candidate_pool_file"] = str(v)
    v = fast._consume_extra_value("--candidate-pool-seed")
    if v is not None:
        NCFG["candidate_pool_seed"] = int(v)
    v = fast._consume_extra_value("--eval-curve-out")
    if v is not None:
        NCFG["eval_curve_out"] = str(v)
    v = fast._consume_extra_value("--candidate-pool-mode")
    if v is not None:
        mode = str(v).strip().lower()
        if mode not in (
            "auto",
            "random_legacy",
            "uniform_grid",
            "coarse_to_fine",
            "footprint_grid_balanced",
        ):
            print(f"[neural] WARN: unknown --candidate-pool-mode '{v}', fallback auto.")
            mode = "auto"
        NCFG["candidate_pool_mode"] = mode
    v = fast._consume_extra_value("--coarse-grid-spacing")
    if v is not None:
        NCFG["coarse_grid_spacing"] = float(v)
    v = fast._consume_extra_value("--coarse-margin")
    if v is not None:
        NCFG["coarse_margin"] = float(v)
    v = fast._consume_extra_value("--coarse-jitter-ratio")
    if v is not None:
        NCFG["coarse_jitter_ratio"] = float(v)
    v = fast._consume_extra_value("--coarse-max-attempts-per-layout")
    if v is not None:
        NCFG["coarse_max_attempts_per_layout"] = int(v)
    v = fast._consume_extra_value("--refine-top-k")
    if v is not None:
        NCFG["refine_top_k"] = int(v)
    v = fast._consume_extra_value("--fine-grid-spacing")
    if v is not None:
        NCFG["fine_grid_spacing"] = float(v)
    v = fast._consume_extra_value("--fine-xy-radius")
    if v is not None:
        NCFG["fine_xy_radius"] = float(v)
    v = fast._consume_extra_value("--fine-variants-per-layout")
    if v is not None:
        NCFG["fine_variants_per_layout"] = int(v)
    v = fast._consume_extra_value("--fine-pool-file")
    if v is not None:
        NCFG["fine_pool_file"] = str(v)
    if fast._consume_extra_flag("--center-prioritized-pool"):
        NCFG["center_prioritized_pool"] = True
    if fast._consume_extra_flag("--benchmark-strict"):
        NCFG["benchmark_strict"] = True
    if fast._consume_extra_flag("--random-pool-order"):
        NCFG["random_pool_order"] = True


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = NeuralGlobalSearcher


def main() -> None:
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()
    gmod._consume_global_args()
    _consume_neural_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    # 加载模型 checkpoint (若提供)。加载失败时仍可运行 (纯 fallback)。
    runner = None
    if NCFG["checkpoint"]:
        try:
            runner = LayoutModelRunner(str(NCFG["checkpoint"]), device=NCFG["device"])
            NCFG["model"] = runner.model_name
            print(f"[neural] loaded checkpoint: {NCFG['checkpoint']} "
                  f"(model={runner.model_name}, generator={runner.is_generator})")
        except Exception as e:
            print(f"[neural] WARN: 加载 checkpoint 失败, 将纯 fallback: {e!r}")
    else:
        if bool(NCFG.get("random_pool_order")):
            runner = _RandomPoolRunner()
            NCFG["model"] = runner.model_name
            print("[neural] using random ordering of the shared candidate pool")
        else:
            print("[neural] 未提供 --checkpoint, 将纯 fallback 到 global explore。")

    NeuralGlobalSearcher._runner = runner

    print("[neural] config:")
    for k, val in NCFG.items():
        print(f"    {k:16s} = {val}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[neural] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
