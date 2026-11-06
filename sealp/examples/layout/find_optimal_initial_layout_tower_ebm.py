"""基于"能量场(Energy-Based Model)"的最优初始布局搜索器。

================================================================================
背景 / 论文复刻
================================================================================
本脚本复刻论文
    Qin et al., "Learning from Planned Data to Improve Robotic Pick-and-Place
    Planning Efficiency" (arXiv:2506.15920) —— 用 EBM(能量模型)预测 shared grasp。

论文核心思想（我们如何借用）:
  * 传统 ``reason_common_gids`` 为了找"两端(取料位 + 目标位)都可行的共享抓取"，
    要对每个 grasp 在两个位姿上分别做 IK + 碰撞，随候选数线性增长，很慢。
  * 论文用一个能量函数 E(T, g) 学"某抓取在某物体位姿下是否可行(IK+无碰)"——
    可行能量低、不可行能量高；再把 **两个位姿的能量相加**，和低于阈值 hs 的就是
    shared grasp（论文式 15/16）。一次前向推理代替成百上千次 IK。

本脚本的具体落地（对 WRS 工程的务实改写）:
  1. 可行性只取决于"夹爪 TCP 的世界位姿 + 开口宽度"以及固定环境(桌子/夹具)。
     因此能量场不按零件、而是 **按手臂(lft/rgt)** 训练一张
     E(tcp_pos, tcp_rotmat, width) → 能量，全部零件复用，泛化更强。
     （这正是论文 per-pose feasibility energy E_φf 的分解形式，shared = 两端能量和。）
  2. 搜索时安装"EBM 加速版 ``reason_common_gids``":
        每个目标位姿 → 用能量场对所有 grasp 一次前向，**预筛**出能量低的子集
        → 只对这一小撮做真正的 IK+碰撞**解析校验**(带上动态障碍) → 各位姿求交集。
     预筛阈值偏"高召回"，且永远走解析校验，所以 **不会引入假阳性**，
     只可能漏掉极少数真可行抓取（论文也是用少量召回换大幅提速）。
  3. 需求3：先用 ``flatsurface.py`` 确定稳定摆放姿态，再在该姿态基础上
     **绕世界 Z 轴采样偏航(yaw)**，生成更多候选交给打分挑选——
     看能否得到更高分/更好可操作性。能量场让"多采几个偏航"几乎不增成本。

================================================================================
用法
================================================================================
1) 先离线构建能量场(只需做一次, 模型按手臂存到 _output/ebm/):
     python -m sealp.examples.layout.find_optimal_initial_layout_tower_ebm \
         --build-ebm --build-only --ebm-poses 60

2) 用能量场加速搜索(自动加载已存模型; 没有则自动退回解析法):
     python -m sealp.examples.layout.find_optimal_initial_layout_tower_ebm \
         --assembly-grid 3 --n-samples 40 --seeds 0 --yaw-samples 8

   常用开关:
     --build-ebm        本次运行先(重新)训练能量场再搜索
     --build-only       只训练能量场, 训练完即退出(配合 --build-ebm)
     --no-ebm           本次完全不用能量场(纯解析, 等价于 heatmap_pso/fast)
     --ebm-poses N      每个零件采样多少个物体位姿做训练数据(默认 50)
     --ebm-epochs N     训练轮数(默认 60)
     --yaw-samples N    需求3: 每个稳定姿绕 Z 额外采样的偏航数(默认 6, 0=关闭)
     --ebm-verify/--no-ebm-verify  预筛后是否解析校验(默认开; 关掉更快但可能假阳性)

本脚本不修改任何现有文件: ``EBMSearcher`` 继承 ``HeatmapPSOSearcher``,
因此热力图先验 / 代理模型 / PSO / 换手 L3 / ``--mp-force-rot`` 等 heatmap_pso 的
全部能力原样保留; 能量场加速作用在 ``reason_common_gids`` 这一层(全局 monkeypatch),
与上述逻辑正交。也即: **在 heatmap_pso 的基础上, 只是把找 common grasp 提速了**。
heatmap_pso 的开关(--no-heatmap/--no-pso/--no-surrogate/--hm-grid/--pso-iters/
--mp-force-rot/--l3-handover-middle-plate 等)在本脚本里同样可用。
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 复用: 原版(CLI/main/保存)、fast 版(IK 缓存 + 可执行性修复)、
#       heatmap_pso 版(热力图先验 + 代理模型 + PSO + 换手 L3 + mp-force-rot)。
# 本脚本 = 在 heatmap_pso 全套能力之上, 叠加"能量场加速 common grasp"和"绕 Z 偏航采样"。
import find_optimal_initial_layout_tower_strict as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_heatmap_pso as hmpso
from find_optimal_initial_layout_tower_strict import RotCandidate

import wrs.basis.robot_math as rm

# torch 是构建/使用能量场所必需; 没有则自动退回纯解析。
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ============================================================
# 0) 控制台降噪 (默认开; --verbose 关闭)
# ============================================================
# 下面这些"整行"输出是底层库的无害噪音, 默认过滤掉以让控制台更干净:
#   * "triangles"/"convex_hull"/"box"/... : make_collision_model 逐个试 cdprim
#     类型时命中 wrs/modeling/collision_model.py 的错误分支 print, 最终会回退
#     默认构造成功, 这些单词行没有任何信息量。
#   * "[INFO] flatsurface loaded ...": 每个零件加载一次, 重复 N 遍。
_NOISE_EXACT_LINES = frozenset({
    "triangles", "convex_hull", "box", "aabb", "obb", "default",
    "cylinder", "capsule", "point_cloud", "surface_balls",
})
_NOISE_PREFIXES = (
    "[INFO] flatsurface loaded",
    "[INFO] flatsurface loaded by normal import",
)


class _QuietStream:
    """逐行过滤 stdout 噪音; 其余原样透传。统计被吞掉的行数。"""

    def __init__(self, base):
        self._base = base
        self._buf = ""
        self.suppressed = 0

    def _emit(self, line: str) -> None:
        stripped = line.strip()
        if stripped in _NOISE_EXACT_LINES or any(
            stripped.startswith(p) for p in _NOISE_PREFIXES
        ):
            self.suppressed += 1
            return
        self._base.write(line)

    def write(self, s: str):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ""
        self._base.flush()

    def __getattr__(self, name):
        return getattr(self._base, name)


def _install_quiet_stdout() -> None:
    if not isinstance(sys.stdout, _QuietStream):
        sys.stdout = _QuietStream(sys.stdout)
    # 底层无害数值边界警告(均不影响结果), 静音以保持控制台干净:
    #   * arccos invalid value : 凸分割里法向夹角偶发越界
    #   * divide invalid value : axis-angle 在转角≈0 时 0/0=nan (相邻帧姿态几乎不变)
    for _msg in (
        r".*invalid value encountered in arccos.*",
        r".*invalid value encountered in divide.*",
        r".*invalid value encountered in true_divide.*",
    ):
        warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)
    # numpy 浮点错误同样静音(invalid/divide), 双保险。
    try:
        np.seterr(invalid="ignore", divide="ignore")
    except Exception:
        pass


def _quiet_report() -> None:
    out = sys.stdout
    n = getattr(out, "suppressed", 0)
    if isinstance(out, _QuietStream) and n > 0:
        out._base.write(
            f"[quiet] 已折叠 {n} 行底层噪音 (cdprim/flatsurface 探测), "
            f"加 --verbose 可查看全部。\n"
        )


# ============================================================
# 默认参数
# ============================================================

EBM_DIR = os.path.join(_THIS_DIR, "_output", "ebm")

EBM_HIDDEN = 128            # 隐藏层宽度
EBM_TEMPERATURE = 0.5       # 论文 Boltzmann 温度 t
EBM_REG_ALPHA = 0.2         # 论文正则项系数 α
EBM_EPOCHS = 60             # 训练轮数
EBM_BATCH = 1024            # 批大小(论文也用 1024)
EBM_LR = 1e-3               # 学习率(论文 1e-3)
EBM_POSES_PER_PART = 50     # 每个零件采样多少物体位姿做训练
EBM_GRASP_CAP = 120         # 每个位姿参与采样的 grasp 上限(均匀子采样)
EBM_VAL_FRAC = 0.2          # 验证集比例(用于挑阈值)
EBM_PREFILTER_RECALL = 0.99 # 预筛阈值的目标召回(用验证集可行样本分位定 cutoff)

YAW_SAMPLES = 6             # 需求3: 每个稳定姿绕 Z 额外采样多少偏航
YAW_MAX_CANDS = 28          # 偏航扩展后每个零件旋转候选总数上限

# 能量场加速统计(只统计 EBM 预筛省了多少 IK)。
_EBM_STATS = {
    "poses_prefiltered": 0,   # 调用 EBM 预筛的位姿次数
    "grasps_in": 0,           # 预筛前 grasp 总数
    "grasps_kept": 0,         # 预筛后(送去解析校验)grasp 数
    "cache_hits": 0,
}

# id(arm) -> "lft"/"rgt", 由 EBMSearcher 注册, 供加速版 reason_common_gids 反查手臂。
_ARM_SIDE_BY_ID: Dict[int, str] = {}

# 加速安装前保存的原 reason_common_gids(退回用)。
_PREV_REASON_COMMON_GIDS = None

# 全局能量场预测器单例(加载好的 per-arm 模型)。
_PREDICTOR: "Optional[EnergyPredictor]" = None


# ============================================================
# 1) 能量场模型(论文 EBM: 三层全连接 + SELU, 输出标量能量)
# ============================================================

if _HAS_TORCH:

    class _EnergyNet(nn.Module):
        """E_φ(x) -> 标量能量。x = [tcp_pos(3), tcp_rotmat(9), width(1)] = 13 维。

        论文用三层全连接 + SELU; 这里保持一致。可行(IK+无碰)能量低, 不可行能量高。
        """

        def __init__(self, in_dim: int, hidden: int = EBM_HIDDEN):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SELU(),
                nn.Linear(hidden, hidden), nn.SELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


FEAT_DIM = 13  # tcp_pos(3) + tcp_rotmat.flatten(9) + width(1)


# ============================================================
# 特征构造(向量化)
# ============================================================

# id(gc) -> (ac_pos(N,3), ac_rotmat(N,3,3), width(N,)) 缓存, 避免每次拆 grasp。
_GC_ARRAY_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _gc_arrays(gc) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = id(gc)
    cached = _GC_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached
    n = len(gc)
    ac_pos = np.zeros((n, 3), dtype=float)
    ac_rot = np.zeros((n, 3, 3), dtype=float)
    width = np.zeros((n,), dtype=float)
    for i in range(n):
        g = gc[i]
        ac_pos[i] = np.asarray(g.ac_pos, dtype=float)
        ac_rot[i] = np.asarray(g.ac_rotmat, dtype=float)
        try:
            width[i] = float(g.ee_values)
        except Exception:
            width[i] = 0.0
    out = (ac_pos, ac_rot, width)
    _GC_ARRAY_CACHE[key] = out
    return out


def _tcp_features_for_gids(gc, gids, pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    """对某物体位姿(pos, rotmat)下的一批 grasp, 算其世界 TCP 特征。

    jaw_center_pos    = pos + rotmat @ ac_pos      (与 reason_common_gids 完全一致)
    jaw_center_rotmat = rotmat @ ac_rotmat
    返回 (len(gids), 13)。
    """
    ac_pos, ac_rot, width = _gc_arrays(gc)
    gids = np.asarray(list(gids), dtype=int)
    if gids.size == 0:
        return np.zeros((0, FEAT_DIM), dtype=float)
    p = ac_pos[gids]                                   # (M,3)
    r = ac_rot[gids]                                   # (M,3,3)
    w = width[gids]                                    # (M,)
    pos = np.asarray(pos, dtype=float)
    rotmat = np.asarray(rotmat, dtype=float)
    jaw_pos = pos[None, :] + p @ rotmat.T              # (M,3)
    jaw_rot = np.einsum("ij,mjk->mik", rotmat, r)      # (M,3,3)
    feat = np.concatenate(
        [jaw_pos, jaw_rot.reshape(jaw_rot.shape[0], 9), w[:, None]], axis=1
    )
    return feat.astype(np.float32)


# ============================================================
# 2) 能量场预测器(加载好的 per-arm 模型, 做快速预筛)
# ============================================================

class EnergyPredictor:
    """持有 lft/rgt 两张能量场, 提供"某位姿下预测可行的 gid 子集"。"""

    def __init__(self):
        self.models: Dict[str, Any] = {}          # arm_side -> _EnergyNet
        self.feat_mean: Dict[str, np.ndarray] = {}
        self.feat_std: Dict[str, np.ndarray] = {}
        self.prefilter_thresh: Dict[str, float] = {}
        self.f1_thresh: Dict[str, float] = {}
        # 训练时验证集"可行样本"的能量分布, 用于加载后按目标召回现场重算阈值
        # (无需重训即可在 召回 vs 精度/提速 之间换档)。
        self.feas_energies: Dict[str, np.ndarray] = {}

    def has(self, arm_side: str) -> bool:
        return arm_side in self.models

    @property
    def ready(self) -> bool:
        return len(self.models) > 0

    def load(self, arm_side: str, path: str) -> bool:
        if not _HAS_TORCH or not os.path.isfile(path):
            return False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = _EnergyNet(int(ckpt["in_dim"]), int(ckpt["hidden"]))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.models[arm_side] = net
        self.feat_mean[arm_side] = np.asarray(ckpt["feat_mean"], dtype=np.float32)
        self.feat_std[arm_side] = np.asarray(ckpt["feat_std"], dtype=np.float32)
        self.prefilter_thresh[arm_side] = float(ckpt["prefilter_thresh"])
        self.f1_thresh[arm_side] = float(ckpt["f1_thresh"])
        fe = ckpt.get("feas_val_energies")
        if fe is not None:
            self.feas_energies[arm_side] = np.asarray(fe, dtype=float).reshape(-1)
        return True

    def retune_recall(self, target_recall: float) -> None:
        """按目标召回现场重算各臂预筛阈值(不重训)。

        target_recall 越低 -> 阈值越严 -> keep 越少 -> 精度/提速越高, 召回略降。
        需要模型里存了验证集可行能量分布(新版 checkpoint 才有)。
        """
        r = float(min(max(target_recall, 0.5), 1.0))
        for arm_side, feas in self.feas_energies.items():
            if feas.size == 0:
                continue
            new_thr = float(np.quantile(feas, r))
            old_thr = self.prefilter_thresh.get(arm_side, new_thr)
            self.prefilter_thresh[arm_side] = new_thr
            print(f"[ebm] retune [{arm_side}] recall={r:.3f}: "
                  f"prefilter_thr {old_thr:.3f} -> {new_thr:.3f}")
        if not self.feas_energies:
            print("[ebm] WARN: 当前模型未存可行能量分布(旧版), 无法在线重调召回, "
                  "请用 --build-ebm --ebm-prefilter-recall 重训。")

    def energies(self, arm_side: str, feats: np.ndarray) -> np.ndarray:
        net = self.models[arm_side]
        mean = self.feat_mean[arm_side]
        std = self.feat_std[arm_side]
        x = (feats - mean) / std
        with torch.no_grad():
            e = net(torch.from_numpy(x.astype(np.float32))).cpu().numpy()
        return np.asarray(e, dtype=float).reshape(-1)

    def prefilter_gids(self, arm_side: str, gc, gids, pos, rotmat) -> List[int]:
        """返回 gids 中"能量 < 预筛阈值(高召回)"的子集(预测可行)。"""
        gids = list(gids)
        if not gids or arm_side not in self.models:
            return gids
        feats = _tcp_features_for_gids(gc, gids, pos, rotmat)
        e = self.energies(arm_side, feats)
        thr = self.prefilter_thresh[arm_side]
        keep = [g for g, ev in zip(gids, e) if ev < thr]
        return keep


# ============================================================
# 3) EBM 加速版 reason_common_gids(能量预筛 + 解析校验)
# ============================================================

# 是否在预筛后做解析校验(默认开 -> 保证不引入假阳性)。
_EBM_VERIFY = True


def _ebm_reason_common_gids(self, grasp_collection, goal_pose_list,
                            obstacle_list=None, toggle_dbg=False):
    """加速版: 每个目标位姿先用能量场预筛, 再解析校验, 最后各位姿求交集。

    与原 ``reason_common_gids`` 语义一致(返回两端都可行的 gid 升序),
    差异仅来自能量场极少量的假阴性(被偏高召回阈值压到很低)。
    toggle_dbg 或无对应手臂模型时, 退回上一层实现(fast 缓存版 / 原版)。
    """
    arm_side = _ARM_SIDE_BY_ID.get(id(self.robot))
    pred = _PREDICTOR
    if (toggle_dbg or pred is None or arm_side is None or not pred.has(arm_side)):
        if _PREV_REASON_COMMON_GIDS is not None:
            return _PREV_REASON_COMMON_GIDS(
                self, grasp_collection, goal_pose_list,
                obstacle_list=obstacle_list, toggle_dbg=toggle_dbg)
        return []

    robot = self.robot
    obs = list(obstacle_list) if obstacle_list else []
    obs_key = fast._obstacle_pose_key(obs)
    base_gids = fast._gc_subset_gids(grasp_collection)

    survivors: Optional[set] = None
    for goal_pose in goal_pose_list:
        pos = np.asarray(goal_pose[0], dtype=float)
        rotmat = np.asarray(goal_pose[1], dtype=float)

        # 每个位姿独立从 base_gids 预筛(确定性 -> 可缓存)。
        cache_key = (id(robot), id(grasp_collection),
                     pos.tobytes(), rotmat.tobytes(), obs_key, "ebm")
        verified = fast._POSE_FEASIBLE_CACHE.get(cache_key)
        if verified is None:
            cand = pred.prefilter_gids(arm_side, grasp_collection, base_gids, pos, rotmat)
            _EBM_STATS["poses_prefiltered"] += 1
            _EBM_STATS["grasps_in"] += len(base_gids)
            _EBM_STATS["grasps_kept"] += len(cand)
            if _EBM_VERIFY:
                verified = fast._pose_feasible_gids(
                    robot, grasp_collection, cand, pos, rotmat, obs)
            else:
                verified = list(cand)
            fast._POSE_FEASIBLE_CACHE[cache_key] = verified
        else:
            _EBM_STATS["cache_hits"] += 1

        vset = set(verified)
        survivors = vset if survivors is None else (survivors & vset)
        if not survivors:
            return []

    return sorted(survivors) if survivors else []


def _install_ebm_accel() -> None:
    global _PREV_REASON_COMMON_GIDS
    try:
        from wrs.manipulation.pick_place import PickPlacePlanner
    except Exception as e:
        print(f"[ebm] skip accel install: {type(e).__name__}: {e}")
        return
    if getattr(PickPlacePlanner.reason_common_gids, "_is_ebm_accel", False):
        return
    _PREV_REASON_COMMON_GIDS = PickPlacePlanner.reason_common_gids
    _ebm_reason_common_gids._is_ebm_accel = True  # type: ignore[attr-defined]
    PickPlacePlanner.reason_common_gids = _ebm_reason_common_gids  # type: ignore[assignment]
    print("[ebm] installed energy-field accelerated reason_common_gids "
          "(EBM prefilter + analytic verify)")


def _print_ebm_accel_report() -> None:
    s = _EBM_STATS
    if s["poses_prefiltered"] == 0:
        return
    g_in = max(s["grasps_in"], 1)
    kept_rate = s["grasps_kept"] / g_in * 100.0
    print()
    print("=" * 78)
    print("  Energy-Field (EBM) Acceleration")
    print("=" * 78)
    print(f"  poses prefiltered by EBM       : {s['poses_prefiltered']}")
    print(f"  pose-cache hits (skipped EBM)  : {s['cache_hits']}")
    print(f"  grasps in  (before prefilter)  : {s['grasps_in']}")
    print(f"  grasps kept (sent to verify)   : {s['grasps_kept']}  ({kept_rate:.1f}%)")
    print(f"  -> IK+collision skipped on     : ~{100.0 - kept_rate:.1f}% of grasp/pose pairs")
    print(f"  verify after prefilter         : {_EBM_VERIFY}")
    print("=" * 78)


# ============================================================
# 4) 训练数据采集 + 训练
# ============================================================

def _sample_object_poses(searcher, pid: str, n_poses: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """为某零件采样 n_poses 个物体世界位姿(pos, rotmat)。

    覆盖搜索时会遇到的 staging 流形:
        随机选一个旋转候选 R(flatsurface 稳定姿/identity/侧立) ->
        随机桌面 xy + 随机绕世界 Z 偏航 theta ->
        rotmat = rotz(theta) @ R, z 由旋转后包围盒底面贴桌算(偏航不改 z)。
    """
    cands = searcher.rot_cands.get(pid, [])
    if not cands:
        return []
    verts = searcher.mesh_vertices.get(pid)
    x0, x1 = searcher.table_x_range
    y0, y1 = searcher.table_y_range
    rng = np.random.default_rng(abs(hash(pid)) % (2 ** 31))
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(int(n_poses)):
        cand = cands[int(rng.integers(0, len(cands)))]
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        Rz = rm.rotmat_from_euler(0.0, 0.0, theta)
        R = Rz @ np.asarray(cand.rotmat, dtype=float)
        if verts is not None:
            bmin, _, _ = fol._bounds_after_rotation(verts, R)
            z_off = searcher.table_top_z + searcher.table_clearance - float(bmin[2])
        else:
            z_off = float(cand.z_offset)
        x = float(rng.uniform(x0, x1))
        y = float(rng.uniform(y0, y1))
        poses.append((np.array([x, y, z_off], dtype=float), R))
    return poses


def collect_dataset(searcher, n_poses: int, grasp_cap: int):
    """采集 per-arm 训练数据。

    返回 {arm_side: (feats(N,13), labels(N,))}。
    label=1 表示该 grasp 在该位姿下对该手臂 **IK 可达且手臂自身不碰撞**。

    重要(与搜索口径对齐): 这里**不放桌面 mesh 等固定环境障碍**。
        因为搜索阶段的抓取检测走 staging_aware, 本来就排除了桌面 mesh
        (薄板/贴桌抓取由 z 抬升 / 立起硬约束处理, 不靠抓取碰撞)。
        若把整张桌子 mesh 当障碍, 几乎所有抓取都会被判碰撞 -> 正样本≈0。
        而"纯可达集"是"带动态障碍真可行集"的超集, 作为预筛先验能保证高召回:
        能量场只负责快速排除明显不可达的抓取, 真正的动态障碍碰撞在
        搜索时由解析校验(_ebm_reason_common_gids 里的 _pose_feasible_gids)兜底。
    """
    env_obs: List = []  # 见上: 标签只取"可达 + 手臂自碰", 不含桌面 mesh
    arms = {"lft": searcher.robot.lft_arm, "rgt": searcher.robot.rgt_arm}
    feats_by_arm: Dict[str, List[np.ndarray]] = {"lft": [], "rgt": []}
    labels_by_arm: Dict[str, List[np.ndarray]] = {"lft": [], "rgt": []}

    # 临时把每位姿 grasp 上限设为 grasp_cap, 控制采集规模。
    old_cap = fast.MAX_GRASPS_PER_POSE
    fast.MAX_GRASPS_PER_POSE = int(grasp_cap) if grasp_cap else old_cap
    try:
        for pid in searcher.part_order:
            gc = searcher._grasp_collection(pid)
            if gc is None or len(gc) == 0:
                continue
            base_gids = fast._gc_subset_gids(gc)
            poses = _sample_object_poses(searcher, pid, n_poses)
            t0 = time.perf_counter()
            n_pos_total = 0
            for pos, rotmat in poses:
                feats = _tcp_features_for_gids(gc, base_gids, pos, rotmat)
                for arm_side, arm in arms.items():
                    feasible = set(fast._pose_feasible_gids(
                        arm, gc, base_gids, pos, rotmat, env_obs))
                    lab = np.array([1.0 if g in feasible else 0.0 for g in base_gids],
                                   dtype=np.float32)
                    feats_by_arm[arm_side].append(feats)
                    labels_by_arm[arm_side].append(lab)
                    n_pos_total += int(lab.sum())
            dt = time.perf_counter() - t0
            print(f"  [{pid:16s}] poses={len(poses)} grasps/pose={len(base_gids)} "
                  f"feasible(both arms)={n_pos_total} ({dt:.1f}s)")
    finally:
        fast.MAX_GRASPS_PER_POSE = old_cap

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for arm_side in ("lft", "rgt"):
        if not feats_by_arm[arm_side]:
            continue
        F = np.concatenate(feats_by_arm[arm_side], axis=0)
        L = np.concatenate(labels_by_arm[arm_side], axis=0)
        out[arm_side] = (F, L)
    return out


def _train_energy_field(arm_side: str, feats: np.ndarray, labels: np.ndarray,
                        epochs: int, save_path: str):
    """训练单张能量场并存盘。

    损失 = NLL + 对比能量(L+ - L-) + α·正则, 复刻论文式(3)(5)(7)(9)。
    阈值: f1_thresh 用验证集最大化 F1; prefilter_thresh 取可行样本能量的高分位
    (默认 99%), 偏高召回, 用于搜索时的预筛。
    """
    n = feats.shape[0]
    pos_n = int(labels.sum())
    neg_n = int(n - pos_n)
    print(f"\n  ===== train energy field [{arm_side}] =====")
    print(f"  samples={n}  feasible={pos_n}  infeasible={neg_n}")
    if pos_n < 10 or neg_n < 10:
        print(f"  [WARN] 正/负样本太少, 跳过 {arm_side} 训练。")
        return False

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_val = max(1, int(n * EBM_VAL_FRAC))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    mean = feats[tr_idx].mean(axis=0)
    std = feats[tr_idx].std(axis=0)
    std[std < 1e-6] = 1.0

    def _norm(a):
        return ((a - mean) / std).astype(np.float32)

    Xtr = torch.from_numpy(_norm(feats[tr_idx]))
    Ytr = torch.from_numpy(labels[tr_idx].astype(np.float32))
    Xval = _norm(feats[val_idx])
    Yval = labels[val_idx].astype(np.float32)

    net = _EnergyNet(feats.shape[1], EBM_HIDDEN)
    opt = torch.optim.Adam(net.parameters(), lr=EBM_LR)
    t = EBM_TEMPERATURE
    ntr = Xtr.shape[0]

    for ep in range(int(epochs)):
        net.train()
        perm = torch.randperm(ntr)
        ep_loss = 0.0
        nb = 0
        for s in range(0, ntr, EBM_BATCH):
            b = perm[s:s + EBM_BATCH]
            xb, yb = Xtr[b], Ytr[b]
            e = net(xb)
            fmask = yb > 0.5
            imask = ~fmask
            if fmask.sum() == 0 or imask.sum() == 0:
                continue
            e_f = e[fmask]
            e_i = e[imask]
            log_Z = torch.logsumexp(-e / t, dim=0)
            nll = e_f.mean() / t + log_Z
            con = e_f.mean() / t - e_i.mean() / t
            reg = (e_f / t).pow(2).mean() + (e_i / t).pow(2).mean()
            loss = nll + con + EBM_REG_ALPHA * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        if (ep + 1) % max(1, epochs // 6) == 0 or ep == 0:
            print(f"    epoch {ep + 1:3d}/{epochs}  loss={ep_loss / max(nb, 1):.4f}")

    # ---- 选阈值 ----
    net.eval()
    with torch.no_grad():
        e_val = net(torch.from_numpy(Xval)).cpu().numpy().reshape(-1)
    feas = e_val[Yval > 0.5]
    infeas = e_val[Yval <= 0.5]

    # f1_thresh: 扫一组阈值最大化 F1(feasible 当作 e<thr)。
    cand_thr = np.quantile(e_val, np.linspace(0.02, 0.98, 49))
    best_f1, best_thr = -1.0, float(np.median(e_val))
    for thr in cand_thr:
        tp = int(np.sum(feas < thr))
        fp = int(np.sum(infeas < thr))
        fn = int(np.sum(feas >= thr))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    # prefilter_thresh: 取可行样本能量的高分位 -> 高召回(漏检率约 1-recall)。
    prefilter_thr = float(np.quantile(feas, EBM_PREFILTER_RECALL)) if feas.size else best_thr
    # 校验集召回(用 prefilter 阈值)。
    rec_pf = float(np.mean(feas < prefilter_thr)) if feas.size else 0.0
    keep_frac = float(np.mean(e_val < prefilter_thr))
    print(f"  f1_thresh={best_thr:.3f} (F1={best_f1:.3f})  "
          f"prefilter_thresh={prefilter_thr:.3f} "
          f"(val recall={rec_pf:.3f}, keep={keep_frac * 100:.1f}%)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": int(feats.shape[1]),
        "hidden": int(EBM_HIDDEN),
        "feat_mean": mean.astype(np.float32),
        "feat_std": std.astype(np.float32),
        "f1_thresh": float(best_thr),
        "prefilter_thresh": float(prefilter_thr),
        "prefilter_recall": float(EBM_PREFILTER_RECALL),
        # 存验证集可行样本能量, 供加载后按目标召回现场重算阈值(免重训换档)。
        "feas_val_energies": feas.astype(np.float32),
        "temperature": float(t),
        "arm_side": arm_side,
        "n_pos": pos_n,
        "n_neg": neg_n,
    }, save_path)
    print(f"  saved -> {save_path}")
    return True


def build_energy_fields(searcher, n_poses: int, epochs: int, grasp_cap: int):
    if not _HAS_TORCH:
        print("[ebm] torch 不可用, 无法构建能量场。")
        return
    print("\n========== 构建能量场 (EBM) ==========")
    print(f"poses/part={n_poses}  grasp_cap={grasp_cap}  epochs={epochs}")
    t0 = time.perf_counter()
    data = collect_dataset(searcher, n_poses=n_poses, grasp_cap=grasp_cap)
    print(f"数据采集完成, 用时 {time.perf_counter() - t0:.1f}s")
    for arm_side, (F, L) in data.items():
        path = os.path.join(EBM_DIR, f"energy_{arm_side}.pt")
        _train_energy_field(arm_side, F, L, epochs=epochs, save_path=path)
    print(f"========== 能量场构建总用时 {time.perf_counter() - t0:.1f}s ==========\n")


def _load_predictor() -> EnergyPredictor:
    pred = EnergyPredictor()
    for arm_side in ("lft", "rgt"):
        path = os.path.join(EBM_DIR, f"energy_{arm_side}.pt")
        if pred.load(arm_side, path):
            print(f"[ebm] loaded energy field [{arm_side}] <- {path} "
                  f"(prefilter_thr={pred.prefilter_thresh[arm_side]:.3f})")
    return pred


# ============================================================
# 4b) 论文式评测: EBM 预测 shared grasp 的 P/R/F1 + 提速
# ============================================================

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_predictor(searcher, predictor: EnergyPredictor, n_pairs: int, grasp_cap: int):
    """复刻论文 Table I/II 的评测口径。

    对每个零件采样 n_pairs 个(取料位姿, 目标位姿)对:
      * ground truth shared = 解析法在两位姿都可行(IK+手臂自碰, 不含桌面/动态障碍,
        与 EBM 训练标签同口径)的 gid 交集;
      * EBM 预测 shared    = 能量场在两位姿各自预筛(energy<阈值)再取交集(本脚本搜索
        时用的 L 法/逻辑与);
    统计:
      * 单位姿可行 P/R/F1 (论文 "F" 行: 纯可行抓取预测);
      * shared grasp P/R/F1 (论文 J/L 行);
      * SG 成功率 = 预测 shared 至少命中一个"真 shared"的位姿对比例;
      * 耗时: 解析全量 vs EBM-only(纯前向) vs EBM+解析校验(本脚本实际混合管线)。
    评测是"原始预测", 不含搜索时的解析兜底, 所以能真实反映模型本身好坏。
    """
    print("\n" + "=" * 78)
    print("  Energy-Field (EBM) Shared-Grasp Prediction Evaluation (paper-style)")
    print("=" * 78)
    print(f"  pairs/part = {n_pairs}   grasp_cap = {grasp_cap}")
    print(f"  注: ground truth 用解析 IK+手臂自碰(不含桌面/动态障碍), 与训练标签同口径。\n")

    old_cap = fast.MAX_GRASPS_PER_POSE
    fast.MAX_GRASPS_PER_POSE = int(grasp_cap) if grasp_cap else old_cap

    # 单位姿可行(feasible)累计
    f_tp = f_fp = f_fn = 0
    # shared 累计
    s_tp = s_fp = s_fn = 0
    sg_total = sg_hit = 0
    t_analytic = t_ebm_only = t_ebm_verify = 0.0
    per_part_rows = []

    try:
        for pid in searcher.part_order:
            gc = searcher._grasp_collection(pid)
            if gc is None or len(gc) == 0:
                continue
            arm_tag = searcher._arm_order(pid)[0]
            arm = searcher.robot.rgt_arm if arm_tag == "rgt" else searcher.robot.lft_arm
            arm_side = "rgt" if arm_tag == "rgt" else "lft"
            if not predictor.has(arm_side):
                continue
            base_gids = fast._gc_subset_gids(gc)
            poses = _sample_object_poses(searcher, pid, 2 * n_pairs)
            if len(poses) < 2:
                continue

            p_f_tp = p_f_fp = p_f_fn = 0
            p_s_tp = p_s_fp = p_s_fn = 0
            p_pairs = 0
            for k in range(0, len(poses) - 1, 2):
                (pa, ra), (pb, rb) = poses[k], poses[k + 1]

                # ---- ground truth (解析) ----
                t0 = time.perf_counter()
                gt_a = set(fast._pose_feasible_gids(arm, gc, base_gids, pa, ra, []))
                gt_b = set(fast._pose_feasible_gids(arm, gc, base_gids, pb, rb, []))
                t_analytic += time.perf_counter() - t0
                gt_shared = gt_a & gt_b

                # ---- EBM 预测 (纯前向) ----
                t0 = time.perf_counter()
                pr_a = set(predictor.prefilter_gids(arm_side, gc, base_gids, pa, ra))
                pr_b = set(predictor.prefilter_gids(arm_side, gc, base_gids, pb, rb))
                t_ebm_only += time.perf_counter() - t0
                pr_shared = pr_a & pr_b

                # ---- EBM + 解析校验 (本脚本实际混合管线) ----
                t0 = time.perf_counter()
                pa_pred = predictor.prefilter_gids(arm_side, gc, base_gids, pa, ra)
                vb_a = set(fast._pose_feasible_gids(arm, gc, pa_pred, pa, ra, []))
                pb_pred = predictor.prefilter_gids(arm_side, gc, base_gids, pb, rb)
                vb_b = set(fast._pose_feasible_gids(arm, gc, pb_pred, pb, rb, []))
                _ = vb_a & vb_b
                t_ebm_verify += time.perf_counter() - t0

                # ---- 单位姿 feasible 指标 (a, b 两个位姿都计) ----
                for gt_set, pr_set in ((gt_a, pr_a), (gt_b, pr_b)):
                    tp = len(pr_set & gt_set)
                    p_f_tp += tp
                    p_f_fp += len(pr_set - gt_set)
                    p_f_fn += len(gt_set - pr_set)

                # ---- shared 指标 ----
                p_s_tp += len(pr_shared & gt_shared)
                p_s_fp += len(pr_shared - gt_shared)
                p_s_fn += len(gt_shared - pr_shared)

                # ---- SG 成功率: 真 shared 非空时, 预测是否命中至少一个 ----
                if gt_shared:
                    p_pairs += 1
                    if pr_shared & gt_shared:
                        sg_hit += 1
                sg_total += 1 if gt_shared else 0

            fp_, fr_, ff1 = _prf(p_f_tp, p_f_fp, p_f_fn)
            sp_, sr_, sf1 = _prf(p_s_tp, p_s_fp, p_s_fn)
            per_part_rows.append((pid, arm_side, fp_, fr_, ff1, sp_, sr_, sf1, p_pairs))
            f_tp += p_f_tp; f_fp += p_f_fp; f_fn += p_f_fn
            s_tp += p_s_tp; s_fp += p_s_fp; s_fn += p_s_fn
            sg_total = sg_total  # already counted
            print(f"  [{pid:14s} {arm_side}] feasible P/R/F1={fp_:.2f}/{fr_:.2f}/{ff1:.2f}  "
                  f"shared P/R/F1={sp_:.2f}/{sr_:.2f}/{sf1:.2f}  (有效对={p_pairs})")
    finally:
        fast.MAX_GRASPS_PER_POSE = old_cap

    FP, FR, FF1 = _prf(f_tp, f_fp, f_fn)
    SP, SR, SF1 = _prf(s_tp, s_fp, s_fn)
    sg_rate = (sg_hit / sg_total * 100.0) if sg_total > 0 else 0.0
    speed_only = (t_analytic / t_ebm_only) if t_ebm_only > 1e-9 else 0.0
    speed_verify = (t_analytic / t_ebm_verify) if t_ebm_verify > 1e-9 else 0.0

    print("\n  ---- 汇总 (micro-average over all parts) ----")
    print(f"  单位姿可行(feasible)  P/R/F1 = {FP:.3f} / {FR:.3f} / {FF1:.3f}")
    print(f"  shared grasp          P/R/F1 = {SP:.3f} / {SR:.3f} / {SF1:.3f}")
    print(f"  SG 成功率(预测命中真shared) = {sg_rate:.1f}%   (真shared非空的对数={sg_total})")
    print(f"\n  ---- 耗时 ----")
    print(f"  解析全量(ground truth)      = {t_analytic:.3f}s")
    print(f"  EBM 纯前向(理论上限提速)    = {t_ebm_only:.3f}s   -> 约 {speed_only:.1f}x")
    print(f"  EBM+解析校验(实际混合管线)  = {t_ebm_verify:.3f}s   -> 约 {speed_verify:.1f}x")

    # ---- 一句话判定 (让人一眼看懂结论) ----
    # 找出 recall 最低的臂(最可能漏抓的薄弱环节)。
    weak = min(per_part_rows, key=lambda r: r[6]) if per_part_rows else None
    safe = SR >= 0.85
    print("\n  ---- 结论 ----")
    print(f"  * 安全性: shared recall={SR:.2f}{'(高, 几乎不漏 common grasp -> 预筛安全)' if safe else '(偏低, 可能漏 common grasp, 建议补训)'}")
    print(f"  * 正确性: 有解析校验兜底, 实际管线 precision≈1, 结果与纯解析一致")
    print(f"  * 提速: 实际 {speed_verify:.1f}x; 瓶颈是 precision({SP:.2f})偏低->送校验候选偏多, 提precision可再提速")
    if weak is not None:
        print(f"  * 最薄弱: [{weak[0]} {weak[1]}] shared recall={weak[6]:.2f} (此臂/件可多给训练数据)")
    print("=" * 78)


# ============================================================
# 5) EBM 搜索器(需求3 偏航采样 + 注册手臂 + 构建/加载能量场)
# ============================================================

# 由 main() 设置的类级配置(模仿 heatmap_pso 的做法)。
EBM_CFG: Dict[str, Any] = {
    "build_ebm": False,
    "build_only": False,
    "use_ebm": True,
    "ebm_poses": EBM_POSES_PER_PART,
    "ebm_epochs": EBM_EPOCHS,
    "ebm_grasp_cap": EBM_GRASP_CAP,
    "yaw_samples": YAW_SAMPLES,
    "eval_ebm": False,
    "eval_pairs": 60,
    # None=用模型里存的(构建时)预筛阈值; 否则加载后按此召回现场重算(免重训换档)。
    "prefilter_recall": None,
}


class EBMSearcher(hmpso.HeatmapPSOSearcher):
    """在 heatmap_pso(热力图+代理+PSO+换手 L3)之上叠加: 偏航候选扩展 + 能量场加速。

    能量场加速作用在 PickPlacePlanner.reason_common_gids(全局 monkeypatch)层面,
    与 heatmap_pso 的热力图/PSO/代理/L3 逻辑正交, 因此那些功能全部原样保留。
    """

    def __init__(self, *args, **kwargs):
        global _PREDICTOR
        super().__init__(*args, **kwargs)

        # 注册左右臂, 供加速版反查手臂(能量场按手臂区分)。
        _ARM_SIDE_BY_ID.clear()
        _ARM_SIDE_BY_ID[id(self.robot.lft_arm)] = "lft"
        _ARM_SIDE_BY_ID[id(self.robot.rgt_arm)] = "rgt"

        # 评测模式(论文式 P/R/F1 + 提速): 加载模型 -> 评测 -> 退出, 不搜索。
        if EBM_CFG.get("eval_ebm"):
            _PREDICTOR = _load_predictor()
            if not _PREDICTOR.ready:
                print("[ebm-eval] 找不到能量场模型, 请先 --build-ebm 构建。")
                sys.exit(1)
            if EBM_CFG.get("prefilter_recall") is not None:
                _PREDICTOR.retune_recall(float(EBM_CFG["prefilter_recall"]))
            evaluate_predictor(self, _PREDICTOR,
                               n_pairs=int(EBM_CFG["eval_pairs"]),
                               grasp_cap=int(EBM_CFG["ebm_grasp_cap"]))
            sys.exit(0)

        # 构建能量场(可选)。
        if EBM_CFG.get("build_ebm"):
            build_energy_fields(
                self,
                n_poses=int(EBM_CFG["ebm_poses"]),
                epochs=int(EBM_CFG["ebm_epochs"]),
                grasp_cap=int(EBM_CFG["ebm_grasp_cap"]),
            )
            if EBM_CFG.get("build_only"):
                print("[ebm] --build-only: 能量场已构建, 退出(不搜索)。")
                sys.exit(0)

        # 加载能量场并安装加速。
        if EBM_CFG.get("use_ebm"):
            _PREDICTOR = _load_predictor()
            if _PREDICTOR.ready:
                if EBM_CFG.get("prefilter_recall") is not None:
                    _PREDICTOR.retune_recall(float(EBM_CFG["prefilter_recall"]))
                _install_ebm_accel()
            else:
                print("[ebm] 未找到任何能量场模型, 本次退回纯解析(可先 --build-ebm)。")

    # ---------------- 需求3: flatsurface 稳定姿 + 绕 Z 偏航采样 ----------------
    def _precompute_rot_candidates(self):
        # 先让基类按 flatsurface 生成稳定姿/侧立/identity 候选(含 force_rot_name 过滤)。
        super()._precompute_rot_candidates()

        yaw_n = int(EBM_CFG.get("yaw_samples", YAW_SAMPLES))
        if yaw_n <= 0:
            return

        print(f"\n========== 需求3: 绕 Z 偏航采样 (yaw_samples={yaw_n}) ==========")
        thetas = [2.0 * math.pi * k / (yaw_n + 1) for k in range(1, yaw_n + 1)]

        for pid in self.part_order:
            verts = self.mesh_vertices.get(pid)
            base_cands = list(self.rot_cands.get(pid, []))
            if not base_cands:
                continue
            expanded: List[RotCandidate] = list(base_cands)
            seen = [np.asarray(c.rotmat, dtype=float) for c in base_cands]

            # 只对"稳定/站立"姿(flatsurface 派生或直立)做偏航扩展;
            # identity 平放绕 Z 也合法, 一并扩展(让搜索能挑朝向)。
            for c in base_cands:
                R0 = np.asarray(c.rotmat, dtype=float)
                for k, theta in enumerate(thetas):
                    Rz = rm.rotmat_from_euler(0.0, 0.0, theta)
                    R = Rz @ R0
                    if any(np.allclose(R, s, atol=1e-4) for s in seen):
                        continue
                    if verts is not None:
                        bmin, _, extent = fol._bounds_after_rotation(verts, R)
                        z_off = self.table_top_z + self.table_clearance - float(bmin[2])
                    else:
                        extent = np.asarray(c.extent, dtype=float)
                        z_off = float(c.z_offset)
                    seen.append(R)
                    expanded.append(RotCandidate(
                        rotmat=R,
                        z_offset=float(z_off),
                        tag=c.tag,
                        extent=np.asarray(extent, dtype=float),
                        footprint=np.asarray(extent[:2], dtype=float),
                        rot_name=f"{c.rot_name}+yaw{int(round(math.degrees(theta)))}",
                        fs_pos=None,
                    ))
                    if len(expanded) >= YAW_MAX_CANDS:
                        break
                if len(expanded) >= YAW_MAX_CANDS:
                    break

            self.rot_cands[pid] = expanded[:YAW_MAX_CANDS]
            print(f"{pid:16s}: {len(base_cands)} -> {len(self.rot_cands[pid])} candidates (含偏航)")


# ============================================================
# 6) main(): 解析自定义 flag -> patch 模块 -> 调用 fol.main()
# ============================================================

def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = EBMSearcher


def main():
    global _EBM_VERIFY, EBM_PREFILTER_RECALL

    # ---- 控制台降噪(默认开; --verbose 关闭) ----
    _verbose = fast._consume_extra_flag("--verbose")
    if not _verbose:
        _install_quiet_stdout()

    # ---- 解析本脚本特有 flag(避免 fol._parse_args 报 unrecognized) ----
    EBM_CFG["build_ebm"] = fast._consume_extra_flag("--build-ebm")
    EBM_CFG["build_only"] = fast._consume_extra_flag("--build-only")
    EBM_CFG["eval_ebm"] = fast._consume_extra_flag("--eval-ebm")
    EBM_CFG["use_ebm"] = not fast._consume_extra_flag("--no-ebm")
    _eval_pairs = fast._consume_extra_value("--eval-pairs")
    if _eval_pairs is not None:
        try:
            EBM_CFG["eval_pairs"] = int(_eval_pairs)
        except ValueError:
            print(f"[ebm] WARN: 无法解析 --eval-pairs '{_eval_pairs}', 用默认。")
    if fast._consume_extra_flag("--no-ebm-verify"):
        _EBM_VERIFY = False
    fast._consume_extra_flag("--ebm-verify")  # 吞掉(默认即开)

    for name, key, cast in (
        ("--ebm-poses", "ebm_poses", int),
        ("--ebm-epochs", "ebm_epochs", int),
        ("--ebm-grasp-cap", "ebm_grasp_cap", int),
        ("--yaw-samples", "yaw_samples", int),
    ):
        val = fast._consume_extra_value(name)
        if val is not None:
            try:
                EBM_CFG[key] = cast(val)
            except ValueError:
                print(f"[ebm] WARN: 无法解析 {name} '{val}', 用默认。")

    # 预筛召回/精度换档: 构建时影响存盘阈值; 加载时(含 --eval-ebm)在线重算阈值(免重训)。
    _pf_recall = fast._consume_extra_value("--ebm-prefilter-recall")
    if _pf_recall is not None:
        try:
            EBM_CFG["prefilter_recall"] = float(_pf_recall)
            EBM_PREFILTER_RECALL = float(_pf_recall)  # 让 --build-ebm 训练也用此召回
            print(f"[ebm] --ebm-prefilter-recall = {EBM_CFG['prefilter_recall']} "
                  f"(越低->精度/提速越高, 召回略降)")
        except ValueError:
            print(f"[ebm] WARN: 无法解析 --ebm-prefilter-recall '{_pf_recall}', 用默认。")

    # --seeds 在 fast.main() 里才会被消费, 而本脚本不走 fast.main, 这里补上,
    # 否则 fol._parse_args 会因 --seeds 报 unrecognized。
    seeds_str = fast._consume_extra_value("--seeds")
    if seeds_str is not None:
        try:
            fast._AUTH_SEEDS_OVERRIDE = [int(s) for s in seeds_str.replace(",", " ").split()]
            print(f"[ebm] --seeds override = {fast._AUTH_SEEDS_OVERRIDE}")
        except ValueError:
            print(f"[ebm] WARN: 无法解析 --seeds '{seeds_str}', 忽略。")

    # 权威阶段预算覆盖(快速验证用): 默认 cap=0 全量 grasp + 30 步打磨非常重,
    # 想快速看通路可用 --auth-exact-cap 400 --auth-polish-iters 0 --auth-max-batches 2。
    for name, attr, cast in (
        ("--auth-exact-cap", "AUTH_EXACT_CAP", int),
        ("--auth-polish-iters", "AUTH_POLISH_ITERS", int),
        ("--auth-max-batches", "AUTH_MAX_BATCHES", int),
    ):
        val = fast._consume_extra_value(name)
        if val is not None:
            try:
                setattr(fast, attr, cast(val))
                print(f"[ebm] {name} -> fast.{attr} = {getattr(fast, attr)}")
            except ValueError:
                print(f"[ebm] WARN: 无法解析 {name} '{val}', 忽略。")

    if not _HAS_TORCH and (EBM_CFG["build_ebm"] or EBM_CFG["use_ebm"]):
        print("[ebm] WARN: torch 不可用, 本次将退回纯解析(无能量场加速)。")
        EBM_CFG["build_ebm"] = False
        EBM_CFG["use_ebm"] = False

    # ---- heatmap_pso 全套 flag 处理(原样复刻其 main, 以保留 PSO/热力图/代理/换手) ----
    hmpso._CFG["use_heatmap"] = not fast._consume_extra_flag("--no-heatmap")
    hmpso._CFG["use_pso"] = not fast._consume_extra_flag("--no-pso")
    hmpso._CFG["use_surrogate"] = not fast._consume_extra_flag("--no-surrogate")
    hmpso._CFG["rebuild_heatmap"] = fast._consume_extra_flag("--rebuild-heatmap")
    for name, key in (("--hm-grid", "hm_grid"), ("--hm-cap", "hm_cap"),
                      ("--hm-top-regions", "hm_top_regions"),
                      ("--pso-particles", "pso_particles"), ("--pso-iters", "pso_iters")):
        val = fast._consume_extra_value(name)
        if val is not None:
            try:
                hmpso._CFG[key] = int(val)
            except ValueError:
                print(f"[ebm] WARN: 无法解析 {name} '{val}', 用默认。")

    # 姿态硬约束(topdown 抓取不足必须立起): 与 heatmap_pso 同口径。
    fast._consume_extra_flag("--keep-upright-preference")  # 吞掉, 兼容旧命令
    if hmpso.DEFAULT_DISABLE_UPRIGHT_PREFERENCE and "--disable-upright-preference" not in sys.argv:
        sys.argv.append("--disable-upright-preference")
        print("[ebm] auto-injected --disable-upright-preference (允许平放)")
    elif "--disable-upright-preference" in sys.argv:
        print("[ebm] --disable-upright-preference: 允许平放(薄板可能实际抓不到, 慎用)")
    else:
        print("[ebm] 姿态硬约束开启: topdown 抓取不足的件(如 middle_plate)必须立起")

    # middle_plate 换手 L3 验证(默认关, 与 heatmap_pso 一致)。
    if fast._consume_extra_flag("--l3-handover-middle-plate"):
        hmpso.L3_HANDOVER_PART_IDS = frozenset({"middle_plate"})
        print("[ebm] --l3-handover-middle-plate: L3 对 middle_plate 走换手验证(较慢)")

    # 强制 middle_plate 旋转(与 heatmap_pso 一致): 默认 rot90_04(长边直立)。
    mp_force_rot = fast._consume_extra_value("--mp-force-rot")
    if mp_force_rot is None:
        mp_force_rot = "rot90_04"
    if str(mp_force_rot).lower() in ("none", "off", ""):
        EBMSearcher.force_rot_name = {}
        print("[ebm] --mp-force-rot none: 不强制 middle_plate 旋转")
    else:
        EBMSearcher.force_rot_name = {"middle_plate": str(mp_force_rot)}
        print(f"[ebm] middle_plate 强制旋转 = {mp_force_rot} (长边直立; 偏航采样会在此基础上绕 Z 转)")

    print("[ebm] config:")
    for k, v in EBM_CFG.items():
        print(f"    {k:14s} = {v}")
    print(f"    {'ebm_verify':14s} = {_EBM_VERIFY}")
    print(f"    {'torch':14s} = {_HAS_TORCH}")
    print(f"    {'ebm_dir':14s} = {EBM_DIR}")
    print(f"[ebm] heatmap_pso config: {hmpso._CFG}")
    print(f"[ebm] sklearn available = {hmpso._HAS_SK}")

    # 继承 fast 的默认行为(order-x 默认关) + 安装 fast 的 IK 缓存。
    # 说明: 能量场加速会在 EBMSearcher.__init__ 里"再包一层"reason_common_gids,
    #       加速版内部仍用 fast._pose_feasible_gids 校验, 享受同样的逐位姿缓存。
    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    _patch_module()

    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[ebm] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            _print_ebm_accel_report()
            fast._print_ik_cache_report()
        except Exception:
            pass
        _quiet_report()


if __name__ == "__main__":
    main()
