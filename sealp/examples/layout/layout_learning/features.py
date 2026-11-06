"""统一特征提取模块。

所有模型 (MLP / DeepSets / Set Transformer / GCN / GAT / Transformer /
PointNet / CVAE / Diffusion / SAGPN) 共用同一套基础特征, 以保证论文实验的
公平对比。

本模块**只依赖 numpy**, 输入是 dataset 中的一条 sample dict (见
``generate_layout_dataset.py`` 的写出格式), 因此训练/特征提取无需加载 wrs 环境。

四种特征视图
------------
1. flatten feature   -> ``build_flatten_feature``  : MLP
2. set feature       -> ``build_set_feature``      : DeepSets / Set Transformer /
                                                     Transformer / PointNet
3. graph feature     -> ``build_graph_feature``    : GCN / GAT / SAGPN
4. proposal target   -> ``build_proposal_target``  : SAGPN / CVAE / Diffusion

每个 part 的节点特征分为两段:
    - STATIC 段 : evaluate_layout **之前**即可获得的特征 (part 几何 / 目标位姿 /
                  装配顺序 / 抓取统计)。生成式模型 (proposal) 只能看到这一段。
    - DYNAMIC 段: 依赖 staging xy 的特征 (staging 坐标 / 到目标距离)。scorer 模型
                  用它做打分; proposal 模型输入时会被 ``static_mask`` 置零。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------
# 归一化常数 (米)。桌面尺度约 0.5m, 统一用 0.5 缩放, 保证特征量级 ~O(1)。
# ------------------------------------------------------------
LEN_SCALE = 0.5
GRASP_SCALE = 60.0          # 抓取数量归一化
TOPDOWN_SCALE = 30.0
MAX_PARTS_DEFAULT = 12      # flatten / padding 用的最大零件数
MAX_REGIONS = 16            # region 分类头固定输出维度 (grid<=4 时 grid*grid<=16)

# 特征版本 (向后兼容):
#   v1 -> 旧版: 用固定 LEN_SCALE=0.5 归一化 (兼容旧 checkpoint / MLP baseline);
#   v2 -> 新版: 按每个样本自身的 table 尺寸/对角线自适应归一化, 并去掉 tower 专有的
#         3x3 网格坐标 (g5,g6), 改为 station_distance_to_center, 面向跨零件/跨任务迁移。
# 维度 (PART/GLOBAL/EDGE) 在 v1/v2 下保持一致, 因此同一模型结构两版通用。
DEFAULT_FEATURE_VERSION = "v1"

# 节点特征分段维度
_STATIC_DIM = 18
_DYNAMIC_DIM = 5
PART_FEATURE_DIM = _STATIC_DIM + _DYNAMIC_DIM  # 23
GLOBAL_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 6

# static_mask: 1 表示该维在 proposal 输入时保留, 0 表示置零(动态维)
STATIC_MASK = np.concatenate([
    np.ones(_STATIC_DIM, dtype=np.float32),
    np.zeros(_DYNAMIC_DIM, dtype=np.float32),
])

# global_static_mask: 生成式模型 (需要"预测装配站") 的全局特征掩码。
# 屏蔽掉"泄漏装配站绝对位置"的维度 (g0,g1=站心相对桌心; g5,g6=网格坐标),
# 只保留 g2,g3=桌面尺寸, g4=零件数。避免站位回归头直接抄输入。
GLOBAL_STATIC_MASK = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.float32)


# ------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------

def _rot6d(rotmat9: List[float]) -> np.ndarray:
    """9 维旋转矩阵 -> 6D 连续旋转表示 (前两列)。"""
    R = np.asarray(rotmat9, dtype=np.float32).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _region_center(sample: Dict) -> np.ndarray:
    pos = np.asarray(sample.get("assembly_station_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
    return pos[:2].astype(np.float32)


def _table_bounds(sample: Dict) -> Tuple[float, float, float, float]:
    xr = sample.get("table_x_range", [0.0, 1.0])
    yr = sample.get("table_y_range", [0.0, 1.0])
    return float(xr[0]), float(xr[1]), float(yr[0]), float(yr[1])


def _table_diag(sample: Dict) -> float:
    """table 对角线长度 (米), v2 自适应归一化的尺度。"""
    xlo, xhi, ylo, yhi = _table_bounds(sample)
    return float(max(np.hypot(xhi - xlo, yhi - ylo), 1e-6))


def _len_scale(sample: Dict, feature_version: str) -> float:
    """长度归一化尺度: v1 用固定常数, v2 用样本 table 对角线 (跨桌面一致)。"""
    if feature_version == "v2":
        return _table_diag(sample)
    return LEN_SCALE


def station_distance_to_center(sample: Dict) -> float:
    """装配站到桌面中心的距离 (米)。"""
    xlo, xhi, ylo, yhi = _table_bounds(sample)
    c = _region_center(sample)
    return float(np.hypot(c[0] - 0.5 * (xlo + xhi), c[1] - 0.5 * (ylo + yhi)))


def normalize_xy(xy: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """table 绝对坐标 -> [-1, 1] 归一化坐标 (供 proposal 输出/target 使用)。"""
    xlo, xhi, ylo, yhi = bounds
    x = 2.0 * (float(xy[0]) - xlo) / max(xhi - xlo, 1e-6) - 1.0
    y = 2.0 * (float(xy[1]) - ylo) / max(yhi - ylo, 1e-6) - 1.0
    return np.array([x, y], dtype=np.float32)


def denormalize_xy(xy_norm: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """[-1, 1] 归一化坐标 -> table 绝对坐标 (推理时反归一化)。"""
    xlo, xhi, ylo, yhi = bounds
    x = (float(xy_norm[0]) + 1.0) * 0.5 * (xhi - xlo) + xlo
    y = (float(xy_norm[1]) + 1.0) * 0.5 * (yhi - ylo) + ylo
    return np.array([x, y], dtype=np.float32)


def normalize_offset(dxy: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """staging 相对装配站的偏移 -> 按桌面尺寸归一化 (整桌跨度对应 [-1,1])。"""
    xlo, xhi, ylo, yhi = bounds
    return np.array([float(dxy[0]) / max(xhi - xlo, 1e-6),
                     float(dxy[1]) / max(yhi - ylo, 1e-6)], dtype=np.float32)


def denormalize_offset(off_norm: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """归一化偏移 -> 绝对偏移 (米)。"""
    xlo, xhi, ylo, yhi = bounds
    return np.array([float(off_norm[0]) * (xhi - xlo),
                     float(off_norm[1]) * (yhi - ylo)], dtype=np.float32)


# ------------------------------------------------------------
# 节点(单零件)特征
# ------------------------------------------------------------

def build_part_feature(part: Dict, sample: Dict, num_parts: int,
                       feature_version: str = DEFAULT_FEATURE_VERSION) -> np.ndarray:
    """构造单个零件的节点特征向量 (维度 = PART_FEATURE_DIM)。

    v1: 用固定 LEN_SCALE 归一化 (与旧 checkpoint 一致)。
    v2: 用样本 table 对角线 (``_len_scale``) 归一化, 使不同桌面/零件尺度一致。
    """
    center = _region_center(sample)
    scale = _len_scale(sample, feature_version)
    feat = np.zeros(PART_FEATURE_DIM, dtype=np.float32)

    extent = np.asarray(part.get("extent", [0.0, 0.0, 0.0]), dtype=np.float32)
    footprint = np.asarray(part.get("footprint", [0.0, 0.0]), dtype=np.float32)
    goal_pos = np.asarray(part.get("goal_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
    goal_rot6 = _rot6d(part.get("goal_rotmat", [1, 0, 0, 0, 1, 0, 0, 0, 1]))

    order_index = float(part.get("order_index", 0))
    is_first = 1.0 if part.get("is_first", False) else 0.0
    topdown = float(part.get("topdown_count", 0))

    # ---- STATIC 段 (0..17) ----
    feat[0:3] = extent / scale
    feat[3:5] = footprint / scale
    feat[5:7] = (goal_pos[:2] - center) / scale             # goal xy 相对装配区中心
    feat[7] = goal_pos[2] / scale
    feat[8:14] = goal_rot6
    feat[14] = order_index / max(num_parts, 1)
    feat[15] = is_first
    feat[16] = topdown / TOPDOWN_SCALE
    feat[17] = float(part.get("grasp_total", 0)) / 100.0

    # ---- DYNAMIC 段 (18..22), 依赖 staging xy ----
    staging_xy = part.get("staging_xy", None)
    if staging_xy is not None:
        sxy = np.asarray(staging_xy, dtype=np.float32)
        d = sxy - goal_pos[:2]
        feat[18:20] = (sxy - center) / scale
        feat[20] = float(np.linalg.norm(d)) / scale
        feat[21:23] = d / scale
    return feat


def build_global_feature(sample: Dict,
                         feature_version: str = DEFAULT_FEATURE_VERSION) -> np.ndarray:
    """构造全局特征向量 (维度 = GLOBAL_FEATURE_DIM)。

    v1: g5,g6 = 3x3 网格行列坐标 (tower 专有)。
    v2: g5 = station_distance_to_center / diag, g6 = 0 (预留), 去除 tower 专有维,
        并用 table 对角线自适应归一化, 面向跨任务迁移。
    """
    xlo, xhi, ylo, yhi = _table_bounds(sample)
    center = _region_center(sample)
    tab_cx = 0.5 * (xlo + xhi)
    tab_cy = 0.5 * (ylo + yhi)
    parts = sample.get("parts", [])
    scale = _len_scale(sample, feature_version)

    g = np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32)
    g[0] = (center[0] - tab_cx) / scale
    g[1] = (center[1] - tab_cy) / scale
    g[2] = (xhi - xlo) / scale
    g[3] = (yhi - ylo) / scale
    g[4] = len(parts) / float(MAX_PARTS_DEFAULT)
    if feature_version == "v2":
        g[5] = station_distance_to_center(sample) / scale
        g[6] = 0.0
    else:
        rc = sample.get("assembly_region_rc", [-1, -1])
        grid = max(1, int(sample.get("assembly_grid", 1)))
        g[5] = float(rc[0]) / grid
        g[6] = float(rc[1]) / grid
    return g


# ------------------------------------------------------------
# 1) flatten feature (MLP)
# ------------------------------------------------------------

def build_flatten_feature(sample: Dict, max_parts: int = MAX_PARTS_DEFAULT,
                          feature_version: str = DEFAULT_FEATURE_VERSION) -> np.ndarray:
    """padding 到 max_parts 的拼接特征 + 全局特征。

    维度 = max_parts * PART_FEATURE_DIM + GLOBAL_FEATURE_DIM
    """
    parts = _ordered_parts(sample)
    node = np.zeros((max_parts, PART_FEATURE_DIM), dtype=np.float32)
    for i, p in enumerate(parts[:max_parts]):
        node[i] = build_part_feature(p, sample, len(parts), feature_version)
    flat = node.reshape(-1)
    return np.concatenate([flat, build_global_feature(sample, feature_version)]).astype(np.float32)


def flatten_feature_dim(max_parts: int = MAX_PARTS_DEFAULT) -> int:
    return max_parts * PART_FEATURE_DIM + GLOBAL_FEATURE_DIM


# ------------------------------------------------------------
# 2) set feature (DeepSets / Set Transformer / Transformer / PointNet)
# ------------------------------------------------------------

def build_set_feature(sample: Dict,
                      feature_version: str = DEFAULT_FEATURE_VERSION) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (node_feat[num_parts, PART_FEATURE_DIM], global_feat[GLOBAL_FEATURE_DIM])。"""
    parts = _ordered_parts(sample)
    n = len(parts)
    node = np.zeros((max(n, 1), PART_FEATURE_DIM), dtype=np.float32)
    for i, p in enumerate(parts):
        node[i] = build_part_feature(p, sample, n, feature_version)
    return node, build_global_feature(sample, feature_version)


# ------------------------------------------------------------
# 3) graph feature (GCN / GAT / SAGPN)
# ------------------------------------------------------------

def build_graph_feature(sample: Dict, k_spatial: int = 2,
                        feature_version: str = DEFAULT_FEATURE_VERSION) -> Dict[str, np.ndarray]:
    """构造图特征。

    返回 dict:
        node_feat   [N, PART_FEATURE_DIM]
        global_feat [GLOBAL_FEATURE_DIM]
        edge_index  [2, E]  (有向, 已含双向)
        edge_feat   [E, EDGE_FEATURE_DIM]

    三类边:
        assembly_order : 相邻装配序 (i-1 <-> i)
        parent_child   : _part_parent_map 声明的父子接触关系
        spatial        : goal-xy 上的 kNN 邻居
    edge_feat = [rel_dx, rel_dy, dist, is_order, is_parent, is_spatial]
    """
    parts = _ordered_parts(sample)
    n = len(parts)
    node = np.zeros((max(n, 1), PART_FEATURE_DIM), dtype=np.float32)
    for i, p in enumerate(parts):
        node[i] = build_part_feature(p, sample, n, feature_version)
    scale = _len_scale(sample, feature_version)

    pid_to_idx = {p["part_id"]: i for i, p in enumerate(parts)}
    goal_xy = np.array([np.asarray(p.get("goal_pos", [0, 0, 0]), dtype=np.float32)[:2]
                        for p in parts], dtype=np.float32) if n else np.zeros((1, 2), np.float32)

    # edge_key -> [is_order, is_parent, is_spatial]
    edges: Dict[Tuple[int, int], np.ndarray] = {}

    def _add(a: int, b: int, slot: int):
        if a == b or a < 0 or b < 0:
            return
        for (u, v) in ((a, b), (b, a)):
            key = (u, v)
            e = edges.get(key)
            if e is None:
                e = np.zeros(3, dtype=np.float32)
                edges[key] = e
            e[slot] = 1.0

    # 装配顺序边
    order_sorted = sorted(range(n), key=lambda i: float(parts[i].get("order_index", i)))
    for a, b in zip(order_sorted[:-1], order_sorted[1:]):
        _add(a, b, 0)
    # 父子边
    for i, p in enumerate(parts):
        par = p.get("parent")
        if par and par in pid_to_idx:
            _add(i, pid_to_idx[par], 1)
    # 空间 kNN 边
    if n >= 2:
        for i in range(n):
            d = np.linalg.norm(goal_xy - goal_xy[i], axis=1)
            nn = np.argsort(d)[1:k_spatial + 1]
            for j in nn:
                _add(i, int(j), 2)

    if not edges:
        # 至少一条自环, 避免空 edge_index
        edges[(0, 0)] = np.zeros(3, dtype=np.float32)

    edge_index = np.zeros((2, len(edges)), dtype=np.int64)
    edge_feat = np.zeros((len(edges), EDGE_FEATURE_DIM), dtype=np.float32)
    for e, ((u, v), flags) in enumerate(edges.items()):
        edge_index[0, e] = u
        edge_index[1, e] = v
        rel = (goal_xy[v] - goal_xy[u]) / scale
        edge_feat[e, 0:2] = rel
        edge_feat[e, 2] = float(np.linalg.norm(rel))
        edge_feat[e, 3:6] = flags

    return {
        "node_feat": node,
        "global_feat": build_global_feature(sample, feature_version),
        "edge_index": edge_index,
        "edge_feat": edge_feat,
    }


# ------------------------------------------------------------
# 4) proposal target (SAGPN / CVAE / Diffusion)
# ------------------------------------------------------------

def build_proposal_target(sample: Dict) -> Dict[str, np.ndarray]:
    """构造 xy proposal / 装配站(assembly station) 监督目标。

    装配站已改为**连续坐标回归** (不再是固定网格分类), 因此主目标是
    ``station_target`` (装配站 xy 归一化到 [-1, 1])。仍保留 region 离散目标以兼容
    旧的 3x3 网格数据集 (连续样本 region_target=0)。

    返回 dict:
        xy_target     [N, 2]  staging 相对装配站的**偏移**, 归一化 (站位无关)
        xy_valid      [N]     该零件是否有有效 staging (第一件预装/缺失时为 0)
        station_target [2]    装配站 xy 归一化到 [-1, 1] (连续绝对回归目标)
        station_valid  scalar 是否有有效装配站
        region_target  scalar 兼容用: 离散装配区索引 (r*grid + c, 连续 -> 0)
        num_regions    scalar

    说明: staging 目标用"相对装配站的偏移"而非绝对坐标, 使其与装配站解耦
    (站位无关), 从而生成式模型可先回归连续装配站, 再回归各零件偏移,
    绝对 staging = 装配站 + 偏移, 两者空间一致。
    """
    parts = _ordered_parts(sample)
    n = len(parts)
    bounds = _table_bounds(sample)
    station_pos = np.asarray(sample.get("assembly_station_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
    station_xy = station_pos[:2]

    xy_t = np.zeros((max(n, 1), 2), dtype=np.float32)
    xy_valid = np.zeros(max(n, 1), dtype=np.float32)
    for i, p in enumerate(parts):
        sxy = p.get("staging_xy", None)
        if sxy is not None and not bool(p.get("is_first", False)):
            off = np.asarray(sxy, dtype=np.float32) - station_xy
            xy_t[i] = normalize_offset(off, bounds)
            xy_valid[i] = 1.0

    # 连续装配站目标: 归一化到桌面 [-1,1]
    station_target = normalize_xy(station_xy, bounds)
    station_valid = np.float32(1.0)

    # 兼容: 离散网格索引 (旧数据集)
    grid = max(1, int(sample.get("assembly_grid", 1)))
    rc = sample.get("assembly_region_rc", [-1, -1])
    if int(rc[0]) < 0 or int(rc[1]) < 0:
        region_target = 0
    else:
        region_target = int(rc[0]) * grid + int(rc[1])
    num_regions = grid * grid
    return {
        "xy_target": xy_t,
        "xy_valid": xy_valid,
        "station_target": station_target.astype(np.float32),
        "station_valid": station_valid,
        "region_target": np.int64(region_target),
        "num_regions": np.int64(num_regions),
    }


# ------------------------------------------------------------
# helper
# ------------------------------------------------------------

def _ordered_parts(sample: Dict) -> List[Dict]:
    """按 order_index 排序的零件列表。"""
    parts = list(sample.get("parts", []))
    parts.sort(key=lambda p: float(p.get("order_index", 0)))
    return parts


def sample_num_parts(sample: Dict) -> int:
    return len(sample.get("parts", []))


# ------------------------------------------------------------
# 辅助监督: fail 类别 + pair-ranking 分组键 (seqrel 用)
# 说明: 这些函数**不改变** node/global/edge 特征维度, 因此不影响任何已有 checkpoint;
#       只是为 dataset.collate 额外产出可选的 batch 字段 (group_id / fail_class),
#       其它模型忽略这些字段即可。
# ------------------------------------------------------------

# 有限 fail 类别 (顺序即类别索引)。-1 = 可行样本(不参与辅助分类)。
FAIL_CLASSES = [
    "no_common_gids",
    "pair_collision",
    "clearance",
    "arm_collision_or_keepout",
    "other",
]
FAIL_CLASS_TO_IDX = {name: i for i, name in enumerate(FAIL_CLASSES)}
NUM_FAIL_CLASSES = len(FAIL_CLASSES)
_FAIL_IGNORE_INDEX = -1


def fail_reason_class(sample: Dict) -> int:
    """把 sample 的 fail_reason / fail_part / fail_detail 归并为有限类别索引。

    可行样本返回 -1 (CrossEntropy 用 ignore_index 忽略)。
    """
    if bool(sample.get("l2_pass", False)):
        return _FAIL_IGNORE_INDEX

    parts = [str(sample.get("fail_reason", "") or ""),
             str(sample.get("fail_part", "") or "")]
    detail = sample.get("fail_detail", {}) or {}
    if isinstance(detail, dict):
        parts.extend(str(k) for k in detail.keys())
    text = " ".join(parts).lower()

    # 关键词优先级: 先判定更具体的臂/keepout, 再一般碰撞。
    if ("arm" in text) or ("keepout" in text) or ("home" in text):
        return FAIL_CLASS_TO_IDX["arm_collision_or_keepout"]
    if ("common_gids" in text) or ("no_common" in text) or ("reason" in text) \
            or ("common gids" in text):
        return FAIL_CLASS_TO_IDX["no_common_gids"]
    if "clearance" in text:
        return FAIL_CLASS_TO_IDX["clearance"]
    if ("pair_collision" in text) or ("pairwise" in text) or ("collision" in text) \
            or ("overlap" in text):
        return FAIL_CLASS_TO_IDX["pair_collision"]
    return FAIL_CLASS_TO_IDX["other"]


def ranking_group_key(sample: Dict) -> str:
    """pair-ranking 分组键: 同 seed + 相近装配站 (量化到 ~5cm) 视为同组。

    只在同组的 feasible 样本之间构造 (score 高 -> 预测高) 的排序对, 避免跨站位比较。
    """
    seed = int(sample.get("seed", 0))
    pos = np.asarray(sample.get("assembly_station_pos", [0.0, 0.0, 0.0]), dtype=float)
    gx = int(round(float(pos[0]) / 0.05))
    gy = int(round(float(pos[1]) / 0.05))
    return f"{seed}|{gx}|{gy}"
