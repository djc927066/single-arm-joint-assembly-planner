#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tower 初始布局搜索加速实现。

复用基础搜索器的约束、评分和输出，仅增加几何缓存、AABB 预筛、IK 结果缓存和可选的多 seed 搜索。"""
from __future__ import annotations

import functools
import os
import sys
import time
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 复用原脚本的 helper / dataclass / 默认参数 / main()。
# 这样保证 CLI、输出格式、所有 default 跟原脚本一致。
import find_optimal_initial_layout_tower_strict as fol
from find_optimal_initial_layout_tower_strict import (
    HOME_JV,
    LayoutCandidate,
    RotCandidate,
    WeightedInitialLayoutSearcher,
)

try:
    from scipy.spatial import cKDTree  # type: ignore

    _HAS_KDTREE = True
except Exception:
    cKDTree = None  # type: ignore
    _HAS_KDTREE = False


# 默认行为开关:
DEFAULT_DISABLE_ORDER_X = True


# ============================================================
# 每个位姿参与 IK 的 grasp 数量上限 (本版本第二大提速点)
# ============================================================
MAX_GRASPS_PER_POSE = 350


# ============================================================
# 权威最优布局管线 (explore -> 收敛 -> 精确重打分 -> 局部打磨)
# ============================================================
AUTH_ENABLE = True            # 总开关:True 走权威管线, False 退回父类单批 random_search
AUTH_MAX_BATCHES = 3          # 探索分几批(每批 = 命令行 --n-samples 个样本); 平衡预算默认 3
AUTH_PATIENCE = 2             # 近似 top-1 连续多少批不变就判定收敛、提前停
AUTH_FP_ROUND = 2             # 收敛指纹:xy 四舍五入到小数点后几位(2=1cm 级)
AUTH_TOPK = 8                 # 最后做精确重打分的候选个数
AUTH_EXACT_CAP = 0            # 权威得分用的 cap(0=全量 grasp, 最权威; 也可设 800 提速)
AUTH_POLISH_ITERS = 30        # 局部打磨迭代次数(0=跳过打磨)
AUTH_POLISH_CAP = 400         # 打磨阶段用的近似 cap(打磨完再精确重打分一次)
AUTH_POLISH_STEP0 = 0.04      # 打磨初始 xy 扰动标准差(米), 失败/不改进时按 0.85 衰减

# ---- 可执行性硬保证(避免找到"算法觉得行、实际抓不了"的布局) ----
ALWAYS_KEEP_ASSEMBLED_OBSTACLES = True   # True=已装件最终位姿永远当障碍(推荐, 保证可执行)

# ---- 多 seed 复现(最强收敛佐证) ----
AUTH_SEEDS: List[int] = [0, 1, 2]   # 默认 3 个 seed; 设为 [] 则用命令行 --seed 单跑
_AUTH_SEEDS_OVERRIDE: Optional[List[int]] = None   # 由 CLI --seeds 填充, 优先级最高


# ============================================================
# Profile (轻量计时，专门用来组会对比)
# ============================================================

# method 短名 -> {"n": 调用次数, "t": 累计耗时(秒)}
_PROFILE: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
# 当前 active profile tag，用于打印时区分 fast vs baseline。
_PROFILE_TAG: str = ""


def _reset_profile(tag: str) -> None:
    _PROFILE.clear()
    global _PROFILE_TAG
    _PROFILE_TAG = tag


def _profile(name: str) -> Callable:
    """函数级计时装饰器，把耗时累加到全局 _PROFILE。"""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrap(*a: Any, **kw: Any):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                rec = _PROFILE.get(name)
                if rec is None:
                    rec = {"n": 0, "t": 0.0}
                    _PROFILE[name] = rec
                rec["n"] += 1
                rec["t"] += time.perf_counter() - t0

        wrap._is_profiled = True  # type: ignore[attr-defined]
        return wrap

    return deco


# 在两个版本里都参与对比的方法集合。
# 注意:有的方法 fast 没重写，那时装饰的是父类版本(走 super)。
_PROFILED_SEARCHER_METHODS = [
    "evaluate_layout",
    "sample_collision_free_xy",
    "_pairwise_collision",
    "_mesh_clearance_reason",
    "_robot_home_collision_reason",
    "_robot_home_clearance_reason",
    "_world_vertices_for_staging",
    "_endpoint_manip",
]


def _decorate_methods(cls: type, methods: List[str]) -> None:
    """给类的若干方法挂上 _profile 装饰器(只挂一次)。"""
    for name in methods:
        m = cls.__dict__.get(name)
        # 如果子类自己有，就直接挂在子类上；否则取继承链上的版本，
        # 装饰后挂到当前 cls，这样子类调用走装饰版本。
        if m is None:
            m = getattr(cls, name, None)
            if m is None:
                continue
            # 已被装饰过就跳过，避免重复
            if getattr(m, "_is_profiled", False):
                continue
            setattr(cls, name, _profile(name)(m))
        else:
            if getattr(m, "_is_profiled", False):
                continue
            setattr(cls, name, _profile(name)(m))


def _decorate_wrs_hotspots() -> None:
    """给 wrs 里和搜索强相关的少量入口挂 profile，便于看 IK 占比。"""
    try:
        from wrs.manipulation.pick_place import PickPlacePlanner

        m = getattr(PickPlacePlanner, "reason_common_gids", None)
        if m is not None and not getattr(m, "_is_profiled", False):
            PickPlacePlanner.reason_common_gids = _profile("wrs.reason_common_gids")(m)  # type: ignore[assignment]
    except Exception as e:
        print(f"[profile] skip reason_common_gids hook: {type(e).__name__}: {e}")

    try:
        from sealp.layout import reachability as _reach_mod

        m = getattr(_reach_mod, "check_pose_reachability", None)
        if m is not None and not getattr(m, "_is_profiled", False):
            _reach_mod.check_pose_reachability = _profile("wrs.check_pose_reachability")(m)  # type: ignore[assignment]
    except Exception as e:
        print(f"[profile] skip check_pose_reachability hook: {type(e).__name__}: {e}")


def _install_profile_hooks(use_fast: bool) -> None:
    """根据当前模式决定给谁挂 profile:fast 子类 or baseline 父类。"""
    target_cls = FastWeightedInitialLayoutSearcher if use_fast else fol.WeightedInitialLayoutSearcher
    _decorate_methods(target_cls, _PROFILED_SEARCHER_METHODS)
    _decorate_wrs_hotspots()


def _print_profile_report() -> None:
    """打印 profile 报告:一张表 + TOTAL,组会展示对应到 PPT 截图直接看。"""
    if not _PROFILE:
        print(f"\n[profile/{_PROFILE_TAG}] no data collected.")
        return

    total_t = sum(rec["t"] for rec in _PROFILE.values())
    print()
    print("=" * 78)
    print(f"  Layout Search Profile  [{_PROFILE_TAG}]")
    print("=" * 78)
    header = (
        f"{'method':<32s} {'calls':>8s} {'total(s)':>10s} "
        f"{'avg(ms)':>10s} {'share':>8s}"
    )
    print(header)
    print("-" * len(header))
    for name, rec in sorted(_PROFILE.items(), key=lambda kv: kv[1]["t"], reverse=True):
        n = int(rec["n"])
        if n == 0:
            continue
        t = float(rec["t"])
        avg_ms = (t / n) * 1000.0
        share = (t / total_t * 100.0) if total_t > 0 else 0.0
        print(
            f"{name:<32s} {n:>8d} {t:>10.3f} {avg_ms:>10.3f} {share:>7.1f}%"
        )
    print("-" * len(header))
    print(f"{'TOTAL (profiled functions)':<32s} {'':>8s} {total_t:>10.3f}")
    print("=" * 78)
    print(
        "[hint] 对比 baseline vs fast 直接看同名 method 的 total(s) / avg(ms) 即可。\n"
        "       未列入 profile 的部分(例如 random_search 主循环本身、yaml IO 等)\n"
        "       占总时间剩余部分；对比时以 profiled functions 的 total 为准。"
    )


# ============================================================
# 逐位姿 IK 可行性缓存 (针对真正的瓶颈 reason_common_gids)
# ============================================================

_POSE_FEASIBLE_CACHE: Dict[Any, List[int]] = {}
_POSE_CACHE_STATS: Dict[str, int] = {"hit": 0, "miss": 0, "ik_evals": 0}
_ORIG_REASON_COMMON_GIDS: Optional[Callable] = None

# id(grasp_collection) -> 抽样后的 gid 列表(整个 run 稳定，grasp 集合长期存活)。
_GC_SUBSET_CACHE: Dict[int, List[int]] = {}


def _gc_subset_gids(grasp_collection) -> List[int]:
    """对一个 grasp 集合做确定性均匀抽样，返回 <= MAX_GRASPS_PER_POSE 个 gid(升序)。"""
    n = len(grasp_collection)
    cap = MAX_GRASPS_PER_POSE
    if not cap or n <= cap:
        return list(range(n))
    key = id(grasp_collection)
    sub = _GC_SUBSET_CACHE.get(key)
    if sub is not None and len(sub) == cap:
        return sub
    idx = np.unique(np.linspace(0, n - 1, cap).astype(int)).tolist()
    _GC_SUBSET_CACHE[key] = idx
    return idx


def _pose_cache_reset_for_layout() -> None:
    """每评估一个新 layout 前清空逐位姿缓存(主要为控内存)。"""
    _POSE_FEASIBLE_CACHE.clear()


def _pose_cache_reset_stats() -> None:
    _POSE_CACHE_STATS["hit"] = 0
    _POSE_CACHE_STATS["miss"] = 0
    _POSE_CACHE_STATS["ik_evals"] = 0
    _GC_SUBSET_CACHE.clear()


def _obstacle_pose_key(obs: List) -> Tuple:
    """把障碍物列表压成 (id, pos字节, rotmat字节) 的可哈希 key。"""
    items = []
    for o in obs:
        oid = id(o)
        try:
            pos_b = np.asarray(o.pos, dtype=float).tobytes()
            rot_b = np.asarray(o.rotmat, dtype=float).tobytes()
            items.append((oid, pos_b, rot_b))
        except Exception:
            items.append((oid, b"", b""))
    items.sort(key=lambda t: t[0])
    return tuple(items)


def _pose_feasible_gids(robot, grasp_collection, gid_iter, pos, rotmat, obstacle_list):
    """复刻 WRS reason_common_gids 的单位姿过滤逻辑(逐 grasp,顺序、判据完全一致)。"""
    out: List[int] = []
    ee = robot.end_effector
    for gid in gid_iter:
        grasp = grasp_collection[gid]
        jaw_center_pos = pos + rotmat.dot(grasp.ac_pos)
        jaw_center_rotmat = rotmat.dot(grasp.ac_rotmat)
        _POSE_CACHE_STATS["ik_evals"] += 1
        jnt_values = robot.ik(tgt_pos=jaw_center_pos, tgt_rotmat=jaw_center_rotmat)
        if jnt_values is None:
            continue
        robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
        if robot.is_collided(obstacle_list=obstacle_list):
            continue
        if ee.is_mesh_collided(cmodel_list=obstacle_list):
            continue
        out.append(gid)
    return out


def _cached_reason_common_gids(self, grasp_collection, goal_pose_list,
                               obstacle_list=None, toggle_dbg=False):
    """reason_common_gids 的缓存版:逐位姿求可行集再取交集。"""
    if toggle_dbg and _ORIG_REASON_COMMON_GIDS is not None:
        return _ORIG_REASON_COMMON_GIDS(
            self, grasp_collection, goal_pose_list,
            obstacle_list=obstacle_list, toggle_dbg=toggle_dbg,
        )

    robot = self.robot
    obs = list(obstacle_list) if obstacle_list else []
    # obs_key 必须包含障碍物的"位姿快照"，不能只用 id:
    obs_key = _obstacle_pose_key(obs)
    gc_key = id(grasp_collection)
    rb_key = id(robot)
    # 所有位姿共用同一抽样子集 -> 交集在子集上精确;且子集对 gc 固定，缓存 key 天然一致。
    base_gids = _gc_subset_gids(grasp_collection)

    survivors: Optional[List[int]] = None
    for goal_pose in goal_pose_list:
        pos = np.asarray(goal_pose[0], dtype=float)
        rotmat = np.asarray(goal_pose[1], dtype=float)
        pose_key = (rb_key, gc_key, pos.tobytes(), rotmat.tobytes(), obs_key)

        feasible = _POSE_FEASIBLE_CACHE.get(pose_key)
        if feasible is None:
            _POSE_CACHE_STATS["miss"] += 1
            feasible = _pose_feasible_gids(
                robot, grasp_collection, base_gids, pos, rotmat, obs
            )
            _POSE_FEASIBLE_CACHE[pose_key] = feasible
        else:
            _POSE_CACHE_STATS["hit"] += 1

        if survivors is None:
            survivors = list(feasible)
        else:
            fset = set(feasible)
            survivors = [g for g in survivors if g in fset]

        if not survivors:
            return []

    return survivors if survivors is not None else []


def _install_ik_cache() -> None:
    """把 PickPlacePlanner.reason_common_gids 换成缓存版(只在 fast 模式装)。"""
    global _ORIG_REASON_COMMON_GIDS
    try:
        from wrs.manipulation.pick_place import PickPlacePlanner
    except Exception as e:
        print(f"[fast] skip IK cache install: {type(e).__name__}: {e}")
        return
    if getattr(PickPlacePlanner.reason_common_gids, "_is_ik_cached", False):
        return
    _ORIG_REASON_COMMON_GIDS = PickPlacePlanner.reason_common_gids
    _cached_reason_common_gids._is_ik_cached = True  # type: ignore[attr-defined]
    PickPlacePlanner.reason_common_gids = _cached_reason_common_gids  # type: ignore[assignment]
    print("[fast] installed per-pose IK feasibility cache on reason_common_gids")


def _print_ik_cache_report() -> None:
    hit = _POSE_CACHE_STATS["hit"]
    miss = _POSE_CACHE_STATS["miss"]
    ik = _POSE_CACHE_STATS["ik_evals"]
    total = hit + miss
    if total == 0:
        return
    rate = (hit / total * 100.0) if total > 0 else 0.0
    print()
    print("=" * 78)
    print("  Per-pose IK Feasibility Cache")
    print("=" * 78)
    cap_txt = str(MAX_GRASPS_PER_POSE) if MAX_GRASPS_PER_POSE else "off (exact)"
    print(f"  reason_common_gids pose lookups : {total}")
    print(f"    cache hits                    : {hit}  ({rate:.1f}%)")
    print(f"    cache misses (real IK passes) : {miss}")
    print(f"  per-grasp IK evaluations        : {ik}")
    print(f"  MAX_GRASPS_PER_POSE             : {cap_txt}")
    print(
        "  [note] cache hit = 省掉一整轮(子集 grasp)的 IK + 双碰撞，主要来自\n"
        "         每个零件 goal 位姿在多个旋转候选间的复用;\n"
        "         MAX_GRASPS_PER_POSE 把每轮 grasp 数封顶，进一步压低 IK 总量\n"
        "         (打分在 grasp≈60~80 已饱和，抽样对最终排序影响很小)。"
    )
    print("=" * 78)


# ============================================================
# Fast Searcher
# ============================================================


class FastWeightedInitialLayoutSearcher(WeightedInitialLayoutSearcher):
    """加速版搜索器。"""

    # ============================================================
    # 初始化
    # ============================================================

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 每个 (pid, id(cand)) 的旋转后顶点 + local AABB 缓存。
        # cand 是 RotCandidate 的稳定对象，id() 在生命周期内唯一。
        self._cand_geom_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

        # 当前每个 pid 在世界坐标系下的几何 cache:
        self._current_world_cache: Dict[str, Dict[str, np.ndarray]] = {}

    # ============================================================
    # 可执行性硬保证:已装件最终位姿永远是障碍
    # ============================================================

    def _planner_obstacles(self, obs: List, current_pid=None, placed=None) -> List:
        """覆写父类:无论 planner_obstacle_mode 是什么, 已装零件(weighted_goal)"""
        base = super()._planner_obstacles(obs, current_pid=current_pid, placed=placed)
        if not ALWAYS_KEEP_ASSEMBLED_OBSTACLES:
            return base
        if self.planner_obstacle_mode in ("mesh", "staging_aware", "executor_match"):
            return base  # 已正确处理, 不要再补 (staging_aware/executor_match 的接触豁免必须保留)
        seen = {id(o) for o in base}
        out = list(base)
        for o in obs:
            if getattr(o, "_sealp_role", None) == "weighted_goal" and id(o) not in seen:
                out.append(o)
                seen.add(id(o))
        return out

    # ============================================================
    # 几何 cache 辅助
    # ============================================================

    def _ensure_cand_geom_cache(
        self, pid: str, cand: RotCandidate
    ) -> Optional[Dict[str, np.ndarray]]:
        key = (pid, id(cand))
        entry = self._cand_geom_cache.get(key)
        if entry is not None:
            return entry

        V = self.mesh_vertices.get(pid)
        if V is None or len(V) == 0:
            return None

        V_arr = np.asarray(V, dtype=float)
        R = np.asarray(cand.rotmat, dtype=float)
        Vrot = V_arr.dot(R.T)
        entry = {
            "rotated_verts": Vrot,
            "local_min": Vrot.min(axis=0),
            "local_max": Vrot.max(axis=0),
        }
        self._cand_geom_cache[key] = entry
        return entry

    def _update_world_cache_from_cand(self, pid: str, cand: RotCandidate) -> None:
        cache = self._ensure_cand_geom_cache(pid, cand)
        if cache is None:
            self._current_world_cache.pop(pid, None)
            return
        cm = self.staging_models.get(pid)
        if cm is None:
            self._current_world_cache.pop(pid, None)
            return
        try:
            pos = np.asarray(cm.pos, dtype=float).copy()
        except Exception:
            self._current_world_cache.pop(pid, None)
            return
        self._current_world_cache[pid] = {
            "rotated_verts": cache["rotated_verts"],
            "local_min": cache["local_min"],
            "local_max": cache["local_max"],
            "pos": pos,
        }

    def _update_world_cache_from_cm(self, pid: str) -> None:
        """staging_models[pid] 的 pos/rotmat 被外部直接改时刷新缓存。"""
        cm = self.staging_models.get(pid)
        if cm is None:
            self._current_world_cache.pop(pid, None)
            return
        V = self.mesh_vertices.get(pid)
        if V is None or len(V) == 0:
            self._current_world_cache.pop(pid, None)
            return
        V_arr = np.asarray(V, dtype=float)
        try:
            R = np.asarray(cm.rotmat, dtype=float)
            pos = np.asarray(cm.pos, dtype=float).copy()
        except Exception:
            self._current_world_cache.pop(pid, None)
            return
        Vrot = V_arr.dot(R.T)
        self._current_world_cache[pid] = {
            "rotated_verts": Vrot,
            "local_min": Vrot.min(axis=0),
            "local_max": Vrot.max(axis=0),
            "pos": pos,
        }

    def _world_aabb_for_staging(
        self, pid: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        entry = self._current_world_cache.get(pid)
        if entry is None:
            return None
        try:
            return entry["local_min"] + entry["pos"], entry["local_max"] + entry["pos"]
        except Exception:
            return None

    @staticmethod
    def _aabb_overlap(
        amin: np.ndarray,
        amax: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> bool:
        return bool(np.all(amax >= bmin) and np.all(bmax >= amin))

    @staticmethod
    def _aabb_separation(
        amin: np.ndarray,
        amax: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> float:
        sep = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
        return float(np.linalg.norm(sep))

    # ============================================================
    # 覆盖父类 pose 写入入口
    # ============================================================

    def _apply_staging_pose(self, pid: str, xy: np.ndarray, cand: RotCandidate) -> None:
        super()._apply_staging_pose(pid, xy, cand)
        self._update_world_cache_from_cand(pid, cand)

    def _apply_first_part_as_assembled(
        self, layout: Optional[LayoutCandidate] = None
    ) -> None:
        super()._apply_first_part_as_assembled(layout)
        first_pid = self._first_part_id()
        if first_pid is not None and first_pid in self.staging_models:
            self._update_world_cache_from_cm(first_pid)

    def _set_assembly_station(
        self,
        fixture_pos: np.ndarray,
        region_id: str = "fixed",
        rc: Tuple[int, int] = (-1, -1),
    ) -> None:
        super()._set_assembly_station(fixture_pos, region_id=region_id, rc=rc)
        # fixture 变化后，旧的 first-part preassembled 缓存(若有)的 pos 不再准确，
        # 这里直接清空，等下次 _apply_first_part_as_assembled 时按新 fixture 重建。
        first_pid = self._first_part_id()
        if first_pid is not None:
            self._current_world_cache.pop(first_pid, None)

    # ============================================================
    # _world_vertices_for_staging 走缓存
    # ============================================================

    def _world_vertices_for_staging(self, pid: str) -> Optional[np.ndarray]:
        entry = self._current_world_cache.get(pid)
        if entry is not None:
            try:
                return entry["rotated_verts"] + entry["pos"]
            except Exception:
                pass
        return super()._world_vertices_for_staging(pid)

    # ============================================================
    # AABB 预筛的 _pairwise_collision
    # ============================================================

    def _pairwise_collision(
        self, active_pids: Optional[List[str]] = None
    ) -> Optional[str]:
        ids = list(active_pids) if active_pids else list(self.staging_models.keys())

        aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for pid in ids:
            ab = self._world_aabb_for_staging(pid)
            if ab is not None:
                aabbs[pid] = ab

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a in aabbs and b in aabbs:
                    if not self._aabb_overlap(*aabbs[a], *aabbs[b]):
                        continue
                if self.staging_models[a].is_mcdwith(self.staging_models[b]):
                    return f"{a} vs {b}"
        return None

    # ============================================================
    # AABB 预筛 + KDTree 加速的 _mesh_clearance_reason
    # ============================================================

    def _vertex_distance_between_world_vertices(
        self, va: np.ndarray, vb: np.ndarray
    ) -> float:
        if va is None or vb is None:
            return float("inf")
        if len(va) == 0 or len(vb) == 0:
            return float("inf")

        max_pts = 600
        if len(va) > max_pts:
            va = va[np.linspace(0, len(va) - 1, max_pts).astype(int)]
        if len(vb) > max_pts:
            vb = vb[np.linspace(0, len(vb) - 1, max_pts).astype(int)]

        # 优先 KDTree，对 ~600 点能比 600×600 broadcasting 快 5-10 倍。
        if _HAS_KDTREE and len(va) > 32 and len(vb) > 32:
            try:
                tree = cKDTree(vb)
                d, _ = tree.query(va, k=1)
                return float(np.min(d))
            except Exception:
                pass

        diff = va[:, None, :] - vb[None, :, :]
        return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))

    def _mesh_clearance_reason(
        self, active_pids: Optional[List[str]] = None
    ) -> Optional[str]:
        min_clear = float(self.min_staging_mesh_clearance)
        if min_clear <= 1e-9:
            return None

        ids = active_pids or list(self.staging_models.keys())
        ids = [p for p in ids if p in self.staging_models]

        aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for pid in ids:
            ab = self._world_aabb_for_staging(pid)
            if ab is not None:
                aabbs[pid] = ab

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a in aabbs and b in aabbs:
                    aabb_d = self._aabb_separation(*aabbs[a], *aabbs[b])
                    if aabb_d >= min_clear:
                        continue

                va = self._world_vertices_for_staging(a)
                vb = self._world_vertices_for_staging(b)
                if va is None or vb is None:
                    continue
                vtx_d = self._vertex_distance_between_world_vertices(va, vb)
                if vtx_d < min_clear:
                    return f"{a} vs {b}: clearance={vtx_d:.4f}m < required {min_clear:.4f}m"
        return None

    # ============================================================
    # _robot_home_collision_reason 去 backup/restore
    # ============================================================

    def _robot_home_collision_reason(
        self, active_pids: Optional[List[str]] = None
    ) -> Optional[str]:
        """与父类结果完全等价，只是省掉 backup_state / restore_state。"""
        if not self.check_robot_home_collision:
            return None

        if active_pids is None:
            pids = list(self.part_order)
        else:
            pids = [p for p in active_pids if p in self.staging_models]
        if not pids:
            return None

        # 必须在 HOME 位姿下检查(reason_common_gids 可能已把手臂带离 HOME)。
        try:
            self.robot.lft_arm.goto_given_conf(HOME_JV)
            self.robot.rgt_arm.goto_given_conf(HOME_JV)
        except Exception:
            pass

        for pid in pids:
            cm = self.staging_models.get(pid)
            if cm is None:
                continue
            for arm_tag, arm in (
                ("lft", self.robot.lft_arm),
                ("rgt", self.robot.rgt_arm),
            ):
                try:
                    hit = arm.is_collided(obstacle_list=[cm])
                    collided = hit[0] if isinstance(hit, tuple) else hit
                except Exception as e:
                    return (
                        f"{pid} vs robot_{arm_tag}_arm_collision_box "
                        f"check_exception={type(e).__name__}: {e}"
                    )
                if collided:
                    return f"{pid} vs robot_{arm_tag}_arm_collision_box"

        return None

    # ============================================================
    # evaluate_layout 外层包一次 backup/restore + layout 级早死亡
    # ============================================================

    def evaluate_layout(self, layout: LayoutCandidate) -> bool:
        # 新 layout -> 障碍布置变了，逐位姿 IK 缓存必须清空(同一 layout 内才安全复用)。
        _pose_cache_reset_for_layout()

        # layout-only 廉价过滤:不用碰任何 staging cm，提前杀掉。
        order_x_hit = self._order_x_constraint_reason(layout)
        if order_x_hit:
            layout.fail_reason = order_x_hit
            return False

        for arm in (self.robot.lft_arm, self.robot.rgt_arm):
            try:
                arm.backup_state()
            except Exception:
                pass
        try:
            try:
                self.robot.lft_arm.goto_given_conf(HOME_JV)
                self.robot.rgt_arm.goto_given_conf(HOME_JV)
            except Exception:
                pass
            return super().evaluate_layout(layout)
        finally:
            for arm in (self.robot.lft_arm, self.robot.rgt_arm):
                try:
                    arm.restore_state()
                except Exception:
                    try:
                        arm.goto_given_conf(HOME_JV)
                    except Exception:
                        pass

    # ============================================================
    # sample_collision_free_xy 加 AABB 预筛
    # ============================================================

    def sample_collision_free_xy(
        self,
        rng: np.random.Generator,
        max_attempts_per_part: int = 150,
    ) -> Optional[Dict[str, np.ndarray]]:
        """与父类语义一致:返回 layout 字典或 None。"""
        xy: Dict[str, np.ndarray] = {}
        placed: List[str] = []
        first_pid = self._first_part_id() if self.preassemble_first_part else None

        if first_pid is not None and first_pid in self.world_poses:
            gp, gr = self.world_poses[first_pid]
            xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            if first_pid in self.staging_models:
                self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
                self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()
                self._update_world_cache_from_cm(first_pid)
            placed.append(first_pid)

        order = sorted(
            [p for p in self.part_order if p != first_pid],
            key=lambda p: float(np.prod(self.rot_cands[p][0].footprint)),
            reverse=True,
        )

        for pid in order:
            cand0 = self.rot_cands[pid][0]
            (xlo, xhi), (ylo, yhi) = self._xy_bounds_for_part_and_cand(pid, cand0)
            if xlo >= xhi or ylo >= yhi:
                return None

            preferred_y_range, goal_side = self._preferred_y_range_by_goal_side(
                pid, ylo, yhi, first_pid=first_pid
            )
            prefer_attempts = int(round(max_attempts_per_part * self.goal_y_side_bias_ratio))

            ok = False
            for attempt_i in range(max_attempts_per_part):
                if attempt_i < prefer_attempts and goal_side in ("left", "right"):
                    cur_ylo, cur_yhi = preferred_y_range
                else:
                    cur_ylo, cur_yhi = ylo, yhi

                if cur_ylo >= cur_yhi:
                    cur_ylo, cur_yhi = ylo, yhi

                p = np.array(
                    [
                        float(rng.uniform(xlo, xhi)),
                        float(rng.uniform(cur_ylo, cur_yhi)),
                    ],
                    dtype=float,
                )

                keepout_hit = self._staging_arm_keepout_reason(pid, p, cand0)
                if keepout_hit:
                    continue

                self._apply_staging_pose(pid, p, cand0)
                ab_pid = self._world_aabb_for_staging(pid)

                collision = False
                for q in placed:
                    if q not in self.staging_models:
                        continue
                    ab_q = self._world_aabb_for_staging(q)
                    if (
                        ab_pid is not None
                        and ab_q is not None
                        and not self._aabb_overlap(*ab_pid, *ab_q)
                    ):
                        continue
                    if self.staging_models[pid].is_mcdwith(self.staging_models[q]):
                        collision = True
                        break

                if collision:
                    continue

                xy[pid] = p
                placed.append(pid)
                ok = True
                break

            if not ok:
                return None

        return {pid: xy[pid] for pid in self.part_order if pid in xy}

    # ============================================================
    # 权威最优布局管线
    # ============================================================

    def _auth_fingerprint(self, cand: Optional[LayoutCandidate]):
        """把一个布局压成可比较的指纹，用于判断 top-1 是否在批次间稳定。"""
        if cand is None:
            return None
        r = AUTH_FP_ROUND
        items = []
        for pid in self.part_order:
            if pid in cand.xy:
                x, y = cand.xy[pid]
                items.append((pid, round(float(x), r), round(float(y), r)))
        return (str(cand.assembly_region_id), tuple(items))

    def _auth_reeval_with_cap(self, src: LayoutCandidate, cap) -> Optional[LayoutCandidate]:
        """在指定 grasp cap 下，对 src 布局(相同装配区 + 相同 xy)重新精确评估。"""
        global MAX_GRASPS_PER_POSE
        old_cap = MAX_GRASPS_PER_POSE
        MAX_GRASPS_PER_POSE = cap
        try:
            if src.assembly_station_pos is not None:
                self._set_assembly_station(
                    np.asarray(src.assembly_station_pos, dtype=float),
                    region_id=src.assembly_region_id,
                    rc=tuple(src.assembly_region_rc),
                )
            fresh = LayoutCandidate(
                xy={k: np.asarray(v, dtype=float).copy() for k, v in src.xy.items()}
            )
            ok = self.evaluate_layout(fresh)
            return fresh if ok else None
        finally:
            MAX_GRASPS_PER_POSE = old_cap

    def _auth_run_batch(self, rng, n_samples, assembly_regions,
                        max_resample_layout, verbose, batch_no) -> List[LayoutCandidate]:
        """跑一批近似探索(cap=MAX_GRASPS_PER_POSE)，返回该批所有 L2 可行候选。"""
        out: List[LayoutCandidate] = []
        for i in range(n_samples):
            region_id, region_rc, region_pos = assembly_regions[i % len(assembly_regions)]
            self._set_assembly_station(region_pos, region_id=region_id, rc=region_rc)

            xy = None
            for _ in range(max_resample_layout):
                xy = self.sample_collision_free_xy(rng)
                if xy is not None:
                    break
            if xy is None:
                continue

            cand = LayoutCandidate(xy=xy)
            if self.evaluate_layout(cand):
                out.append(cand)
                if verbose:
                    print(f"[B{batch_no}] #{i:03d} L2_OK  approx_score={cand.layout_score:.4f} "
                          f"region={cand.assembly_region_id} counts={cand.grasp_counts}")
            elif verbose:
                print(f"[B{batch_no}] #{i:03d} FAIL  {cand.fail_reason}")
        return out

    def _auth_exact_rescore(self, cands: List[LayoutCandidate]) -> List[LayoutCandidate]:
        """对候选列表用 AUTH_EXACT_CAP(默认 0=全量 grasp)逐个精确重打分。"""
        rescored: List[LayoutCandidate] = []
        for rank, c in enumerate(cands, start=1):
            approx = c.layout_score
            fresh = self._auth_reeval_with_cap(c, AUTH_EXACT_CAP)
            if fresh is None:
                print(f"[exact] top{rank}: approx={approx:.4f} -> 精确评估下不可行，丢弃")
                continue
            fresh._auth_approx_score = float(approx)  # type: ignore[attr-defined]
            print(f"[exact] top{rank}: approx={approx:.4f} -> exact={fresh.layout_score:.4f} "
                  f"(Δ={fresh.layout_score - approx:+.4f})  counts={fresh.grasp_counts}")
            rescored.append(fresh)
        rescored.sort(key=lambda c: c.layout_score, reverse=True)
        return rescored

    def _auth_local_polish(self, base: LayoutCandidate) -> LayoutCandidate:
        """在 base 周围对各零件 xy 做随机扰动爬山(近似 cap)，返回近似分最高的布局。"""
        if base is None or AUTH_POLISH_ITERS <= 0:
            return base

        global MAX_GRASPS_PER_POSE
        old_cap = MAX_GRASPS_PER_POSE
        MAX_GRASPS_PER_POSE = AUTH_POLISH_CAP if AUTH_POLISH_CAP else old_cap

        first_pid = self._first_part_id() if self.preassemble_first_part else None
        xlo_t, xhi_t = float(self.table_x_range[0]), float(self.table_x_range[1])
        ylo_t, yhi_t = float(self.table_y_range[0]), float(self.table_y_range[1])
        rng = np.random.default_rng(0xA17)

        try:
            best = self._auth_reeval_with_cap(base, MAX_GRASPS_PER_POSE)
            if best is None:
                return base
            step = float(AUTH_POLISH_STEP0)
            n_improve = 0
            for it in range(AUTH_POLISH_ITERS):
                trial_xy = {}
                for pid, v in best.xy.items():
                    p = np.asarray(v, dtype=float).copy()
                    if pid != first_pid:
                        p = p + rng.normal(0.0, step, size=2)
                        p[0] = float(np.clip(p[0], xlo_t, xhi_t))
                        p[1] = float(np.clip(p[1], ylo_t, yhi_t))
                    trial_xy[pid] = p

                if best.assembly_station_pos is not None:
                    self._set_assembly_station(
                        np.asarray(best.assembly_station_pos, dtype=float),
                        region_id=best.assembly_region_id,
                        rc=tuple(best.assembly_region_rc),
                    )
                trial = LayoutCandidate(xy=trial_xy)
                if self.evaluate_layout(trial) and trial.layout_score > best.layout_score + 1e-9:
                    best = trial
                    n_improve += 1
                    print(f"[polish] it={it:02d} improved -> approx_score={best.layout_score:.4f} (step={step:.3f})")
                else:
                    step *= 0.85
            print(f"[polish] done: {n_improve} improvements, final approx_score={best.layout_score:.4f}")
            return best
        finally:
            MAX_GRASPS_PER_POSE = old_cap

    def _auth_resolve_seeds(self, default_seed: int) -> List[int]:
        """决定本次跑哪些 seed:--seeds 覆盖 > AUTH_SEEDS 常量 > 命令行 --seed。"""
        if _AUTH_SEEDS_OVERRIDE:
            return list(_AUTH_SEEDS_OVERRIDE)
        if AUTH_SEEDS:
            return list(AUTH_SEEDS)
        return [int(default_seed)]

    def _auth_single_seed(self,
                          seed: int,
                          n_samples: int,
                          max_resample_layout: int,
                          verbose: bool) -> Optional[Dict[str, Any]]:
        """对单个 seed 跑完整管线:explore -> 收敛 -> 精确重打分 top-K -> 局部打磨。"""
        rng = np.random.default_rng(seed)
        assembly_regions = self._assembly_region_candidates()

        pool: List[LayoutCandidate] = []
        conv_curve: List[Tuple[int, float, bool]] = []
        prev_fp = None
        stable = 0
        converged = False
        t0 = time.time()

        for b in range(1, AUTH_MAX_BATCHES + 1):
            print(f"\n---------- [seed {seed}] explore batch {b}/{AUTH_MAX_BATCHES} ----------")
            new_feasible = self._auth_run_batch(
                rng, n_samples, assembly_regions, max_resample_layout, verbose, b
            )
            pool.extend(new_feasible)
            pool.sort(key=lambda c: c.layout_score, reverse=True)

            cur_best = pool[0] if pool else None
            cur_fp = self._auth_fingerprint(cur_best)
            is_stable = (cur_fp is not None and cur_fp == prev_fp)
            stable = stable + 1 if is_stable else 0
            prev_fp = cur_fp

            best_approx = cur_best.layout_score if cur_best else float("nan")
            conv_curve.append((b, best_approx, is_stable))
            print(f"[converge][seed {seed}] batch {b}: pool={len(pool)} feasible, "
                  f"approx top-1 score={best_approx:.4f}, stable_streak={stable}/{AUTH_PATIENCE}")

            if stable >= AUTH_PATIENCE:
                converged = True
                print(f"[converge][seed {seed}] top-1 连续 {stable} 批稳定 -> 收敛，提前停止。")
                break

        dt_explore = time.time() - t0

        if not pool:
            print(f"\n[FAIL][seed {seed}] 探索阶段没有任何 L2 可行布局。")
            return None

        print(f"\n========== [seed {seed}] Exact re-score (cap={AUTH_EXACT_CAP}) top-{AUTH_TOPK} ==========")
        exact_ranked = self._auth_exact_rescore(pool[:AUTH_TOPK])
        if not exact_ranked:
            print(f"[WARN][seed {seed}] top-K 精确评估全不可行，用近似 top-1。")
            return {
                "seed": seed, "best": pool[0], "exact_ranked": [pool[0]],
                "conv_curve": conv_curve, "dt_explore": dt_explore,
                "pool_size": len(pool), "converged": converged, "exact_certified": False,
            }

        exact_best = exact_ranked[0]
        polished_final = exact_best
        if AUTH_POLISH_ITERS > 0:
            print(f"\n========== [seed {seed}] Local polish ==========")
            polished = self._auth_local_polish(exact_best)
            polished_exact = self._auth_reeval_with_cap(polished, AUTH_EXACT_CAP)
            if polished_exact is not None and polished_exact.layout_score > exact_best.layout_score + 1e-9:
                print(f"[polish][seed {seed}] 精确分提升: {exact_best.layout_score:.4f} -> {polished_exact.layout_score:.4f}")
                polished_final = polished_exact
            else:
                got = polished_exact.layout_score if polished_exact is not None else float("nan")
                print(f"[polish][seed {seed}] 未提升(exact={got:.4f})，保留原权威最优。")

        self._auth_print_report(polished_final, exact_ranked, conv_curve, dt_explore, len(pool), seed=seed)
        return {
            "seed": seed, "best": polished_final, "exact_ranked": exact_ranked,
            "conv_curve": conv_curve, "dt_explore": dt_explore,
            "pool_size": len(pool), "converged": converged, "exact_certified": True,
        }

    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "staging_aware",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        """权威最优布局管线 + 多 seed 复现(AUTH_ENABLE=True 时)。"""
        if not AUTH_ENABLE:
            return super().random_search(
                n_samples=n_samples, seed=seed, max_resample_layout=max_resample_layout,
                verbose=verbose, enable_l3=enable_l3, l3_top_k=l3_top_k,
                l3_obstacle_mode=l3_obstacle_mode, require_l3=require_l3,
            )

        seeds = self._auth_resolve_seeds(seed)

        print("\n========== Authoritative Layout Search ==========")
        print(f"seeds                 = {seeds}  ({len(seeds)} 个 seed 独立复现)")
        print(f"per-batch samples     = {n_samples}")
        print(f"max batches           = {AUTH_MAX_BATCHES}  (单 seed 总预算 <= {n_samples * AUTH_MAX_BATCHES})")
        print(f"convergence patience  = {AUTH_PATIENCE} batches (fingerprint round={AUTH_FP_ROUND})")
        print(f"explore cap           = {MAX_GRASPS_PER_POSE} (approx; 仅用于排序)")
        print(f"exact rescore cap     = {AUTH_EXACT_CAP} (0=全量 grasp, 权威)  top-K={AUTH_TOPK}")
        print(f"polish                = {AUTH_POLISH_ITERS} iters @ cap {AUTH_POLISH_CAP}")
        print(f"part_order            = {self.part_order}")
        if len(seeds) > 1:
            print(f"[cost] 注意:多 seed ≈ {len(seeds)}x 单 seed 时间，请预留时间。")

        results: List[Dict[str, Any]] = []
        wall0 = time.time()
        for si, sd in enumerate(seeds, start=1):
            print(f"\n############### SEED {sd}  ({si}/{len(seeds)}) ###############")
            r = self._auth_single_seed(sd, n_samples, max_resample_layout, verbose)
            if r is not None:
                results.append(r)

        if not results:
            print("\n[FAIL] 所有 seed 都没有找到可行布局。")
            return None

        # ---------- 跨 seed 一致性 / 权威性判定 ----------
        overall_best = self._auth_cross_seed_report(results, time.time() - wall0)

        # ---------- 可选 L3 ----------
        if enable_l3 and overall_best is not None:
            print("\n========== L3 full-process validation on overall best ==========")
            seed_of_best = None
            for r in results:
                if r["best"] is overall_best:
                    seed_of_best = r
                    break
            l3_list = [overall_best]
            if seed_of_best is not None:
                l3_list += [c for c in seed_of_best["exact_ranked"] if c is not overall_best]
            for rank, cand in enumerate(l3_list[:max(1, l3_top_k)], start=1):
                if self.validate_full_sequence_l3(cand, obstacle_mode=l3_obstacle_mode, verbose=True):
                    print(f"[OK] L3 passed at rank={rank}, exact_score={cand.layout_score:.4f}")
                    return cand
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")
            if require_l3:
                print("\n[FAIL] L3 top-k 全部未通过。")
                return None
            print("\n[WARN] L3 全失败，require_l3=False，回退全局权威最优(仅 L2)。")

        return overall_best

    @staticmethod
    def _auth_xy_dispersion(layouts: List[LayoutCandidate], part_order: List[str]) -> Tuple[float, float]:
        """同一组布局里，各零件 xy 在不同 seed 间的最大/平均成对漂移(米)。"""
        if len(layouts) < 2:
            return 0.0, 0.0
        per_part_max = []
        for pid in part_order:
            pts = [np.asarray(L.xy[pid], dtype=float) for L in layouts if pid in L.xy]
            if len(pts) < 2:
                continue
            d = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = max(d, float(np.linalg.norm(pts[i] - pts[j])))
            per_part_max.append(d)
        if not per_part_max:
            return 0.0, 0.0
        return float(max(per_part_max)), float(np.mean(per_part_max))

    def _auth_cross_seed_report(self, results: List[Dict[str, Any]], wall_dt: float) -> Optional[LayoutCandidate]:
        """打印跨 seed 一致性报告，给出权威结论，返回全局精确分最高的布局。"""
        valid = [r for r in results if r.get("best") is not None]
        if not valid:
            return None

        scores = [float(r["best"].layout_score) for r in valid]
        regions = [str(r["best"].assembly_region_id) for r in valid]
        region_counts = Counter(regions)
        dominant_region, dom_n = region_counts.most_common(1)[0]
        region_agree = (len(region_counts) == 1)

        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        cv = (std_s / mean_s) if mean_s > 1e-9 else float("inf")

        same_region_layouts = [r["best"] for r in valid if str(r["best"].assembly_region_id) == dominant_region]
        max_disp, mean_disp = self._auth_xy_dispersion(same_region_layouts, list(self.part_order))

        n_conv = sum(1 for r in valid if r.get("converged"))

        overall = max(valid, key=lambda r: r["best"].layout_score)["best"]

        print("\n" + "#" * 78)
        print("  CROSS-SEED AUTHORITATIVE EVIDENCE REPORT")
        print("#" * 78)
        print(f"  seeds run / feasible      : {len(results)} / {len(valid)}")
        print(f"  per-seed converged        : {n_conv}/{len(valid)}  (达到 patience={AUTH_PATIENCE} 的收敛)")
        print(f"  total wall time           : {wall_dt:.1f}s")
        print("  --- per-seed authoritative best ---")
        for r in valid:
            b = r["best"]
            print(f"    seed {r['seed']:>3}: exact={b.layout_score:.4f}  region={b.assembly_region_id}  "
                  f"rc={tuple(b.assembly_region_rc)}  converged={r.get('converged')}")
        print("  --- cross-seed agreement ---")
        print(f"    region agreement        : {'YES (全部同区)' if region_agree else 'NO'}  "
              f"dominant={dominant_region} ({dom_n}/{len(valid)})")
        print(f"    exact score             : mean={mean_s:.4f}  std={std_s:.4f}  "
              f"min={min_s:.4f}  max={max_s:.4f}  CV={cv*100:.2f}%")
        print(f"    xy dispersion (同区seed): max={max_disp*100:.1f}cm  mean={mean_disp*100:.1f}cm")

        # 权威结论
        strong = region_agree and (cv < 0.05) and (max_disp < 0.08) and (n_conv == len(valid))
        moderate = (dom_n >= (len(valid) + 1) // 2) and (cv < 0.10)
        print("  --- VERDICT ---")
        if strong:
            print("    ★ STRONG: 全部 seed 收敛到同一装配区, 精确分 CV<5%, 各件漂移<8cm。")
            print("      可声称: 该布局是与随机种子无关的高置信度近最优解。")
        elif moderate:
            print("    ◐ MODERATE: 多数 seed 指向同一区且分数较稳(CV<10%)。")
            print("      建议: 增大 n_samples / 多跑几个 seed 以进一步坐实收敛。")
        else:
            print("    ○ WEAK: seed 之间分歧较大, 尚未稳定。")
            print("      建议: 增大 n_samples、AUTH_MAX_BATCHES, 或检查打分/约束设置。")
        print(f"  --- GLOBAL BEST (将被保存) ---")
        print(f"    exact_score={overall.layout_score:.4f}  region={overall.assembly_region_id}  "
              f"rc={tuple(overall.assembly_region_rc)}")
        print(f"    grasp_counts={overall.grasp_counts}")
        print(f"    arm_choice={overall.arm_choice}")
        print(f"    pose_tag={overall.pose_tag}")
        print("#" * 78)
        return overall

    def _auth_print_report(self, best, exact_ranked, conv_curve, dt_explore, pool_size, seed=None) -> None:
        tag = f" [seed {seed}]" if seed is not None else ""
        approx = getattr(best, "_auth_approx_score", None)
        print("\n" + "=" * 78)
        print(f"  AUTHORITATIVE BEST LAYOUT REPORT{tag}")
        print("=" * 78)
        print(f"  exact layout_score        : {best.layout_score:.4f}   (cap={AUTH_EXACT_CAP}=全量grasp)")
        if approx is not None:
            print(f"  approx score (explore cap): {approx:.4f}   Δ(exact-approx)={best.layout_score - approx:+.4f}")
        print(f"  assembly_region           : {best.assembly_region_id}  rc={tuple(best.assembly_region_rc)}")
        print(f"  grasp_counts (exact)      : {best.grasp_counts}")
        print(f"  arm_choice                : {best.arm_choice}")
        print(f"  pose_tag                  : {best.pose_tag}")
        print(f"  explored feasible pool    : {pool_size}   explore time = {dt_explore:.1f}s")
        print("  --- convergence curve (approx top-1 over batches) ---")
        for (b, sc, stable) in conv_curve:
            print(f"    batch {b}: approx_top1={sc:.4f}  {'[stable]' if stable else ''}")
        print("  --- exact top-K ranking ---")
        for rank, c in enumerate(exact_ranked, start=1):
            ap = getattr(c, "_auth_approx_score", float('nan'))
            print(f"    #{rank}: exact={c.layout_score:.4f}  approx={ap:.4f}  region={c.assembly_region_id}")
        print("  [note] 权威分 = 全量 grasp 精确重算; explore/polish 的近似分仅用于筛选。")
        print("=" * 78)


# ============================================================
# Entry point
# ============================================================


def _patch_module_to_use_fast_searcher() -> None:
    """让 ``fol.main()`` 调用 fast 版本的 Searcher。"""
    fol.WeightedInitialLayoutSearcher = FastWeightedInitialLayoutSearcher


def _maybe_inject_default_flags() -> None:
    """按 fast 脚本约定，给 ``sys.argv`` 补默认开关。"""
    if DEFAULT_DISABLE_ORDER_X and "--disable-order-x-constraint" not in sys.argv:
        sys.argv.append("--disable-order-x-constraint")


def _consume_extra_flag(name: str) -> bool:
    """从 ``sys.argv`` 里抽出一个自定义 flag，避免被 fol._parse_args 当成未知参数。"""
    if name in sys.argv:
        sys.argv.remove(name)
        return True
    return False


def _consume_extra_value(name: str) -> Optional[str]:
    """从 ``sys.argv`` 里抽出 ``--name VALUE`` 形式的自定义参数, 返回 VALUE。"""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return val
        sys.argv.remove(name)
    return None


def main():
    global _AUTH_SEEDS_OVERRIDE
    # 自定义 flag 必须先抽出，否则 fol._parse_args 会报 unrecognized arguments。
    use_baseline = _consume_extra_flag("--baseline")
    do_profile = not _consume_extra_flag("--no-profile")
    seeds_str = _consume_extra_value("--seeds")
    if seeds_str:
        try:
            _AUTH_SEEDS_OVERRIDE = [int(s) for s in seeds_str.replace(",", " ").split()]
            print(f"[fast] --seeds override = {_AUTH_SEEDS_OVERRIDE}")
        except ValueError:
            print(f"[fast] WARN: 无法解析 --seeds '{seeds_str}', 忽略。")
    use_fast = not use_baseline

    print("[fast]" + (" BASELINE mode" if use_baseline else " FAST mode"))
    print(f"[fast] scipy.cKDTree available = {_HAS_KDTREE}")

    _maybe_inject_default_flags()
    if DEFAULT_DISABLE_ORDER_X:
        print(
            "[fast] auto-injected --disable-order-x-constraint "
            "(set DEFAULT_DISABLE_ORDER_X=False in this file to re-enable)"
        )

    # IK 缓存只在 fast 模式装，baseline 保持原汁原味，便于公平对比。
    if use_fast:
        _install_ik_cache()
    _pose_cache_reset_stats()

    if do_profile:
        _install_profile_hooks(use_fast=use_fast)
        _reset_profile(tag="FAST" if use_fast else "BASELINE")
        print(
            "[fast] profile hooks installed on "
            f"{'FastWeightedInitialLayoutSearcher' if use_fast else 'WeightedInitialLayoutSearcher (baseline)'}"
        )
    else:
        print("[fast] profile disabled (--no-profile)")

    if use_fast:
        _patch_module_to_use_fast_searcher()

    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        wall_dt = time.perf_counter() - wall_t0
        if do_profile:
            _print_profile_report()
            if use_fast:
                _print_ik_cache_report()
            tag = "FAST" if use_fast else "BASELINE"
            print(f"[profile/{tag}] wall-clock total (whole script) = {wall_dt:.3f}s\n")


if __name__ == "__main__":
    main()
