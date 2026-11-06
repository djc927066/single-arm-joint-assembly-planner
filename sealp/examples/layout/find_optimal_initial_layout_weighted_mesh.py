#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict Tower Initial Layout Search — PyCharm Runnable
=====================================================

这是一个可直接在 PyCharm 里右键运行的 Tower 初始摆放搜索脚本。
它模仿 find_optimal_layout.py 的核心思路：

1. 先把 work_table 粗分成 3x3 装配候选区域，自动选择装配中心；再在 work_table 上随机采样其它零件初始 (x, y)。
2. 对每个随机 layout，按装配顺序逐个零件评估可行性。
3. 每个零件自动枚举候选初始旋转姿态：
   - 优先使用 flatsurface.py 计算出的稳定摆放旋转矩阵；
   - 若 flatsurface 不可用，再回退 identity/90° 站立/侧放姿态；
   - 每次旋转后自动根据旋转后的包围盒 z_min 计算 z_offset，避免半个物体插进 work_table。
4. 对每个可行 layout 计算综合得分：
       layout_score
         = w_grasp * 抓取冗余度
         + w_manip * 端点灵巧度
         + w_dist  * 搬运距离得分
         + w_rot   * 旋转代价得分
6. 保存综合得分最高的 layout，供后续执行脚本使用.

推荐放置：
    sealp/examples/layout/find_optimal_initial_layout_tower_strict.py

PyCharm 右键运行：
    不需要填写任何参数，默认读取当前 Tower 的 asmdef / sample_config / tower_grasp，
    默认执行 L2 加权随机搜索 + L3 全流程动态避障验证。

命令行也可以覆盖默认参数，例如：
    python -m sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm --n-samples 20 --l3-top-k 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # 兜底：假设当前脚本在 sealp/examples/layout 下
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.manipulation.pick_place import PickPlacePlanner
import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
from wrs.grasping.grasp import GraspCollection

from sealp.assembly_sequence import AssemblyDef
from sealp.config import load_config
from sealp.colliders import StaticEnvironment
from sealp.layout import WorkspaceLayout
from sealp.layout.dual_staging_search import find_obstacle_def
from sealp.primitives.transport import TransportPrimitive

try:
    from sealp.layout.reachability import check_pose_reachability
except Exception:
    check_pose_reachability = None


# ============================================================
# 默认参数
# ============================================================

DUAL_ARM_Y_OFFSET = 0.62
HOME_JV = np.zeros(6)

# 全流程 transport / pick-place 验证参数
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PICK_DEPART_DIST = 0.05
PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0], dtype=float)
PLACE_APPROACH_DIST = 0.05
PLACE_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PLACE_DEPART_DIST = 0.05
APPROACH_DIST = 0.0
LINEAR_GRANULARITY = 0.04

DEFAULT_TABLE_MARGIN = 0.06
DEFAULT_TABLE_CLEARANCE = 0.003
DEFAULT_MAX_ROT_CANDIDATES = 12

# 归一化参数
NORM_GRASP_MIN_TARGET = 8.0
NORM_GRASP_MEAN_TARGET = 20.0
NORM_GRASP_HILL_K = 2.0
NORM_MANIP_TARGET = 0.030
NORM_DIST_DECAY = 0.55
NORM_ROT_DECAY = 0.75


# 装配区域搜索：默认把 work_table 粗分成 3x3，自动选择装配中心。
DEFAULT_PLAN_ASSEMBLY_REGION = True
DEFAULT_ASSEMBLY_GRID = 3
DEFAULT_PREASSEMBLE_FIRST_PART = True

# 初始摆放姿态：默认优先用 flatsurface.py 计算稳定摆放角度。
DEFAULT_USE_FLATSURFACE = True
DEFAULT_FS_STABILITY_THRESHOLD = 0.10

# 默认权重，总和建议为 1.0
DEFAULT_W_GRASP = 0.40
DEFAULT_W_MANIP = 0.15
DEFAULT_W_DIST = 0.25
DEFAULT_W_ROT = 0.20


# ============================================================
# PyCharm 右键运行默认配置
# ============================================================
# 这些默认路径都基于 SEALP_ROOT 自动生成，所以只要本脚本放在 sealp 目录内部，
# 直接右键运行就能跑 Tower 任务。
DEFAULT_ASMDEF = os.path.join(SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef")
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")
DEFAULT_GRASP_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp")
DEFAULT_PART_ORDER = "base_plate,post_br,post_fr,post_bl,post_fl,middle_plate,top_cross"
DEFAULT_OUTPUT_NAME = "tower_optimal_initial"
DEFAULT_OUTPUT_DIR = os.path.join(SEALP_ROOT, "examples", "layout", "_output")
DEFAULT_FIXTURE_POS = "0.36,0,0"
DEFAULT_ROBOT_BASE_POS = "0,0,0"

# 右键运行时不要太大，否则 L3 会很慢。正式搜索可改成 20/50/80。
DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 0
DEFAULT_CDPRIM_TYPE = "box"

# L2 只负责快速筛候选，none 可以避免 WRS gripper 粗碰撞误杀；
# 最终是否真的全程不碰撞，由 L3_OBSTACLE_MODE="mesh" 严格验证。
DEFAULT_L2_OBSTACLE_MODE = "none"

# 默认开启严格 L3：只有 TransportPrimitive/RRT 全流程通过才保存 layout。
DEFAULT_ENABLE_L3 = False
DEFAULT_L3_TOP_K = 3
DEFAULT_L3_OBSTACLE_MODE = "box"
DEFAULT_ALLOW_L2_FALLBACK = True


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RotCandidate:
    rotmat: np.ndarray
    z_offset: float
    tag: str
    extent: np.ndarray
    footprint: np.ndarray
    rot_name: str = "unknown"
    fs_pos: Optional[np.ndarray] = None


@dataclass
class LayoutCandidate:
    xy: Dict[str, np.ndarray]
    layout_score: float = -np.inf
    assembly_region_id: str = "fixed"
    assembly_region_rc: Tuple[int, int] = (-1, -1)
    assembly_station_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    assembly_station_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    chosen_rotmat: Dict[str, np.ndarray] = field(default_factory=dict)
    z_offset: Dict[str, float] = field(default_factory=dict)
    pose_tag: Dict[str, str] = field(default_factory=dict)
    rot_name: Dict[str, str] = field(default_factory=dict)
    arm_choice: Dict[str, str] = field(default_factory=dict)
    grasp_counts: Dict[str, int] = field(default_factory=dict)
    per_part_dist: Dict[str, float] = field(default_factory=dict)
    per_part_manip: Dict[str, float] = field(default_factory=dict)
    per_part_rot_angle: Dict[str, float] = field(default_factory=dict)
    grasp_score_norm: float = 0.0
    manip_score_norm: float = 0.0
    dist_score_norm: float = 0.0
    rot_score_norm: float = 0.0
    fail_reason: str = ""
    l2_pass: bool = False
    l3_pass: bool = False
    l3_fail_reason: str = ""


# ============================================================
# 基础工具
# ============================================================

def _parse_vec3(text: str, default: Tuple[float, float, float]) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=float)
    return np.asarray([float(x.strip()) for x in text.split(",")], dtype=float)


def _parse_part_order(text: str) -> Optional[List[str]]:
    if text is None or text.strip() == "":
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_json_map(text: str) -> Dict[str, str]:
    if text is None or text.strip() == "":
        return {}
    return dict(json.loads(text))


def _hill(x: float, target: float, k: float = 2.0) -> float:
    if x <= 0:
        return 0.0
    r = (x / target) ** k
    return float(r / (1.0 + r))


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    try:
        _, ang = rm.axangle_between_rotmat(np.asarray(Ra), np.asarray(Rb))
        return float(ang) if np.isfinite(ang) else 0.0
    except Exception:
        v = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
        return float(np.arccos(np.clip(v, -1.0, 1.0)))


def _load_mesh_vertices(mesh_path: str) -> np.ndarray:
    try:
        import trimesh
    except Exception:
        import wrs.basis.trimesh as trimesh
    mesh = trimesh.load_mesh(mesh_path)
    return np.asarray(mesh.vertices, dtype=float)


def _bounds_after_rotation(vertices: np.ndarray, rotmat: np.ndarray):
    pts = np.asarray(vertices, dtype=float) @ np.asarray(rotmat, dtype=float).T
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    extent = bmax - bmin
    return bmin, bmax, extent


def _rotmat_90_candidates() -> List[np.ndarray]:
    """生成常用 90° 旋转候选。

    identity 会排在最前。其余候选用于在平放不可行时站立/侧放。
    """
    base_rots = [
        np.eye(3),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(180.0)),
    ]

    yaw_rots = [
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(a))
        for a in (0.0, 90.0, -90.0, 180.0)
    ]

    out: List[np.ndarray] = []
    for B in base_rots:
        for Z in yaw_rots:
            R = Z @ B
            if not any(np.allclose(R, old, atol=1e-6) for old in out):
                out.append(R)

    # identity 强制第一
    out.sort(key=lambda R: 0 if np.allclose(R, np.eye(3), atol=1e-6) else 1)
    return out



def _load_flatsurface_reference_class():
    """动态加载 flatsurface.py 里的 FSReferencePoses。

    你当前项目里的真实路径是：
        wrs/manipulation/placement/flatsurface.py

    所以这里优先用包导入：
        from wrs.manipulation.placement.flatsurface import FSReferencePoses

    如果包导入失败，再尝试若干文件路径；都失败才回退到 90° 姿态候选。
    """
    # 1) 正确优先路径：wrs/manipulation/placement/flatsurface.py
    try:
        from wrs.manipulation.placement.flatsurface import FSReferencePoses
        print("[INFO] flatsurface loaded: wrs.manipulation.placement.flatsurface")
        return FSReferencePoses
    except Exception as e:
        print(f"[WARN] 包导入 flatsurface 失败: {type(e).__name__}: {e}")

    # 2) 文件路径兜底
    candidate_files = [
        os.path.join(PROJECT_ROOT, "wrs", "manipulation", "placement", "flatsurface.py"),
        os.path.join(SEALP_ROOT, "..", "wrs", "manipulation", "placement", "flatsurface.py"),
        os.path.join(_THIS_DIR, "flatsurface.py"),
        os.path.join(SEALP_ROOT, "examples", "layout", "flatsurface.py"),
        os.path.join(SEALP_ROOT, "examples", "grasp", "flatsurface.py"),
        os.path.join(SEALP_ROOT, "flatsurface.py"),
    ]

    for fp in candidate_files:
        fp = os.path.abspath(fp)
        if not os.path.isfile(fp):
            continue
        try:
            spec = importlib.util.spec_from_file_location("tower_flatsurface_runtime", fp)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, "FSReferencePoses", None)
            if cls is not None:
                print(f"[INFO] flatsurface loaded: {fp}")
                return cls
        except Exception as e:
            print(f"[WARN] 加载 flatsurface.py 失败: {fp}, {type(e).__name__}: {e}")

    # 3) 最后再尝试普通 import，兼容用户自己放到 PYTHONPATH 的情况
    try:
        from flatsurface import FSReferencePoses
        print("[INFO] flatsurface loaded by normal import.")
        return FSReferencePoses
    except Exception as e:
        print(f"[WARN] 未找到可用 flatsurface.py，回退 90° 姿态候选: {type(e).__name__}: {e}")
        return None


def _stable_pose_candidates_from_flatsurface(
    mesh_path: str,
    stability_threshold: float = 0.10,
    max_candidates: int = 24,
) -> List[Tuple[str, Optional[np.ndarray], np.ndarray]]:
    """用 flatsurface.py 计算稳定摆放姿态。

    返回:
        [(rot_name, fs_pos, rotmat), ...]

    注意：
        flatsurface 返回的 pos 是稳定放置参考位姿里的位置修正。
        本脚本最终仍统一用 rotated bounds 计算 z_offset，避免物体插入桌面。
    """
    cls = _load_flatsurface_reference_class()
    if cls is None:
        return []

    try:
        obj = mcm.CollisionModel(mesh_path)
        fs_ref = cls(
            obj_cmodel=obj,
            stability_threshhold=float(stability_threshold),
        )
        out: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
        for i, pose in enumerate(fs_ref):
            try:
                fs_pos, R = pose
                R = np.asarray(R, dtype=float)
                fs_pos = np.asarray(fs_pos, dtype=float)
                if R.shape != (3, 3):
                    continue
                if not np.all(np.isfinite(R)):
                    continue
                if any(np.allclose(R, old_R, atol=1e-6) for _, _, old_R in out):
                    continue
                out.append((f"fs_{i:02d}", fs_pos, R))
                if len(out) >= int(max_candidates):
                    break
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[WARN] flatsurface 计算失败: {os.path.basename(mesh_path)}, {type(e).__name__}: {e}")
        return []


def _fallback_rot_candidates_with_names() -> List[Tuple[str, Optional[np.ndarray], np.ndarray]]:
    """把旧的 90° 候选转成带名字的候选。"""
    out: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
    for i, R in enumerate(_rotmat_90_candidates()):
        if np.allclose(R, np.eye(3), atol=1e-6):
            name = "identity"
        else:
            name = f"rot90_{i:02d}"
        out.append((name, None, R))
    return out


def _dedupe_pose_rot_candidates(
    candidates: List[Tuple[str, Optional[np.ndarray], np.ndarray]]
) -> List[Tuple[str, Optional[np.ndarray], np.ndarray]]:
    """按旋转矩阵去重，保留前面的候选。"""
    out: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
    for name, fs_pos, R in candidates:
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            continue
        if any(np.allclose(R, old_R, atol=1e-6) for _, _, old_R in out):
            continue
        out.append((name, fs_pos, R))
    return out


def _table_info(config_yaml: str, table_name: str, margin: float):
    """读取 work_table 的 xy 范围和顶面高度。"""
    if not config_yaml or not os.path.isfile(config_yaml):
        return (0.05, 0.65), (-0.95, 0.35), 0.0

    cfg = load_config(config_yaml)
    table_def = find_obstacle_def(cfg.obstacle_defs, table_name)
    if table_def and table_def.get("type") == "box":
        pos = np.asarray(table_def.get("pos", [0, 0, 0]), dtype=float)
        extent = np.asarray(table_def.get("extent", [0.8, 1.2, 0.02]), dtype=float)
        x_range = (
            float(pos[0] - extent[0] / 2.0 + margin),
            float(pos[0] + extent[0] / 2.0 - margin),
        )
        y_range = (
            float(pos[1] - extent[1] / 2.0 + margin),
            float(pos[1] + extent[1] / 2.0 - margin),
        )
        top_z = float(pos[2] + extent[2] / 2.0)
        return x_range, y_range, top_z

    return (0.05, 0.65), (-0.95, 0.35), 0.0


def _load_env_obstacles(config_yaml: str) -> List:
    if not config_yaml or not os.path.isfile(config_yaml):
        return []
    cfg = load_config(config_yaml)
    env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    obs = list(env.obstacle_list)
    for o in obs:
        o._sealp_role = "environment_obstacle"
    return obs


def make_collision_model(mesh_path: str, cdprim_type: str = "box"):
    """创建 CollisionModel。

    速度优先版本：
    - cdprim_type="box"/"aabb"/"bbox"：优先使用包围盒碰撞，速度快，但比 mesh 保守；
    - cdprim_type="triangles"：使用三角网格碰撞，更精确但慢；
    - cdprim_type="convex_hull"：折中。

    注意：
    obstacle_mode="box" 和 obstacle_mode="mesh" 都表示把所有动态障碍物加入规划；
    真正决定碰撞几何精细程度的是这里的 cdprim_type。
    """
    requested = str(cdprim_type or "box").lower()

    # 用户习惯写法统一映射
    if requested in ("bbox", "bounding_box", "bounding-box", "aabb_box"):
        requested = "box"

    trials = []

    if requested in ("box", "aabb", "obb"):
        # 不同 WRS 版本可能支持的字符串不同，所以多试几个。
        for t in ("box", "aabb", "obb"):
            trials.append({"initor": mesh_path, "cdprim_type": t})
        # 有些版本默认就是包围盒，所以也试默认构造。
        trials.append({"initor": mesh_path})
        # 如果都不支持，再退回 convex_hull。
        trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})
    elif requested in ("default", "none", ""):
        trials.append({"initor": mesh_path})
        trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})
    else:
        # 例如 triangles / convex_hull
        trials.append({"initor": mesh_path, "cdprim_type": requested})
        if requested != "convex_hull":
            trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})
        trials.append({"initor": mesh_path})

    last_err = None
    for kw in trials:
        try:
            cm = mcm.CollisionModel(**kw)
            cm._sealp_cdprim_type = kw.get("cdprim_type", "default_box_or_default")
            return cm
        except TypeError as e:
            # 有些 WRS 版本参数名不是 initor，尝试位置参数形式
            last_err = e
            if "initor" in kw and "cdprim_type" in kw:
                try:
                    cm = mcm.CollisionModel(mesh_path, cdprim_type=kw["cdprim_type"])
                    cm._sealp_cdprim_type = kw["cdprim_type"]
                    return cm
                except Exception as ee:
                    last_err = ee
            elif "initor" in kw:
                try:
                    cm = mcm.CollisionModel(mesh_path)
                    cm._sealp_cdprim_type = "default_box_or_default"
                    return cm
                except Exception as ee:
                    last_err = ee
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"无法创建 CollisionModel: {mesh_path}, cdprim_type={cdprim_type}, last_err={last_err!r}")


def _model_id_for_part(asm: AssemblyDef, part_id: str) -> Optional[str]:
    for attr in ("parts", "part_defs"):
        parts = getattr(asm, attr, None)
        if parts is None:
            continue
        if isinstance(parts, dict):
            p = parts.get(part_id)
            if p is not None:
                return getattr(p, "model", None) or getattr(p, "model_id", None)
        else:
            for p in parts:
                if getattr(p, "part_id", None) == part_id:
                    return getattr(p, "model", None) or getattr(p, "model_id", None)
    try:
        return os.path.splitext(os.path.basename(asm.model_path(part_id)))[0]
    except Exception:
        return None



def _reset_robot_for_l3(robot) -> None:
    """清理双臂 hold 状态，避免一次 L3 失败影响下一次候选。"""
    for arm in (robot.lft_arm, robot.rgt_arm):
        ee = getattr(arm, "end_effector", None)
        if ee is not None:
            try:
                ee.oiee_list = []
                ee.oiee_list_bk.clear()
                ee.oiee_pose_list_bk.clear()
            except Exception:
                pass
        try:
            arm.goto_given_conf(HOME_JV)
        except Exception:
            pass




def _patch_rrt_for_l3() -> None:
    """与 find_optimal_layout.py 类似，对 L3 RRT 做可控放宽。

    这里不是降低碰撞要求，而是给 RRT 更合理的搜索步长和时间；
    obstacle_list 仍然完整传入，碰撞检查不会被跳过。
    """
    try:
        from wrs.motion.probabilistic.rrt_connect import RRTConnect
    except Exception:
        return

    if getattr(RRTConnect.plan, "_tower_l3_patched", False):
        return

    _orig_plan = RRTConnect.plan

    def _patched_plan(self, *args, **kwargs):
        kwargs.setdefault("ext_dist", 0.30)
        kwargs.setdefault("smoothing_n_iter", 150)
        kwargs.setdefault("max_time", 10.0)
        return _orig_plan(self, *args, **kwargs)

    _patched_plan._tower_l3_patched = True
    RRTConnect.plan = _patched_plan

def _transport_kwargs():
    """通用 tower pick-place 方向。

    这里保持从上方抓取、向上撤离、从上往下放置。
    如果后续某类零件需要专属方向，可以在这里按 part_id 扩展。
    """
    return dict(
        pick_depart_direction=PICK_DEPART_DIR,
        pick_depart_distance=PICK_DEPART_DIST,
        place_approach_direction_list=[PLACE_APPROACH_DIR],
        place_approach_distance_list=[PLACE_APPROACH_DIST],
        place_depart_direction_list=[PLACE_DEPART_DIR],
        place_depart_distance_list=[PLACE_DEPART_DIST],
    )


def _find_grasp_pickle(part_id: str,
                       model_id: Optional[str],
                       mesh_path: str,
                       grasp_dir: str,
                       explicit_map: Dict[str, str]) -> Optional[str]:
    keys = [part_id]
    if model_id:
        keys.append(model_id)
    mesh_base = os.path.splitext(os.path.basename(mesh_path))[0]
    keys.append(mesh_base)

    # tower 的 post_bl/post_fl/post_br/post_fr 共用 post 抓取库
    if part_id.startswith("post_"):
        keys.append("post")

    for k in keys:
        if k in explicit_map:
            p = explicit_map[k]
            if not os.path.isabs(p):
                p = os.path.join(grasp_dir, p)
            if os.path.isfile(p):
                return p

    names = []
    for k in keys:
        names.extend([
            f"{k}_grasps_topdown.pickle",
            f"{k}_grasps.pickle",
            f"{k}.pickle",
            f"tower_{k}_grasps_topdown.pickle",
            f"tower_{k}_grasps.pickle",
        ])

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(grasp_dir, name)
        if os.path.isfile(p):
            return p
    return None


# ============================================================
# Searcher
# ============================================================

class WeightedInitialLayoutSearcher:
    def __init__(self,
                 asmdef_path: str,
                 config_yaml: str,
                 grasp_dir: str,
                 fixture_pos: np.ndarray,
                 fixture_rotmat: np.ndarray,
                 robot_base_pos: np.ndarray,
                 robot_base_rotmat: np.ndarray,
                 part_order: Optional[List[str]],
                 output_name: str,
                 table_name: str,
                 table_margin: float,
                 table_clearance: float,
                 grasp_map: Dict[str, str],
                 max_rot_candidates: int,
                 w_grasp: float,
                 w_manip: float,
                 w_dist: float,
                 w_rot: float,
                 ignore_env: bool = False,
                 cdprim_type: str = "triangles",
                 planner_obstacle_mode: str = "mesh",
                 plan_assembly_region: bool = DEFAULT_PLAN_ASSEMBLY_REGION,
                 assembly_grid: int = DEFAULT_ASSEMBLY_GRID,
                 preassemble_first_part: bool = DEFAULT_PREASSEMBLE_FIRST_PART,
                 use_flatsurface: bool = DEFAULT_USE_FLATSURFACE,
                 fs_stability_threshold: float = DEFAULT_FS_STABILITY_THRESHOLD):
        self.asmdef_path = os.path.abspath(asmdef_path)
        self.config_yaml = os.path.abspath(config_yaml)
        self.grasp_dir = os.path.abspath(grasp_dir)
        self.fixture_pos = np.asarray(fixture_pos, dtype=float)
        self.fixture_rotmat = np.asarray(fixture_rotmat, dtype=float)
        self.robot_base_pos = np.asarray(robot_base_pos, dtype=float)
        self.robot_base_rotmat = np.asarray(robot_base_rotmat, dtype=float)
        self.output_name = output_name
        self.table_name = table_name
        self.table_margin = float(table_margin)
        self.table_clearance = float(table_clearance)
        self.grasp_map = dict(grasp_map)
        self.max_rot_candidates = int(max_rot_candidates)
        # collision primitive 类型必须在创建 staging/goal CollisionModel 之前保存。
        # 否则 _make_staging_models() 调用 self.cdprim_type 会 AttributeError。
        self.cdprim_type = str(cdprim_type) if 'cdprim_type' in locals() else 'triangles'
        self.ignore_env = bool(ignore_env)
        self.planner_obstacle_mode = str(planner_obstacle_mode) if 'planner_obstacle_mode' in locals() else 'mesh'
        self.plan_assembly_region = bool(plan_assembly_region)
        self.assembly_grid = max(1, int(assembly_grid))
        self.preassemble_first_part = bool(preassemble_first_part)
        self.use_flatsurface = bool(use_flatsurface)
        self.fs_stability_threshold = float(fs_stability_threshold)
        self.current_assembly_region_id = "fixed"
        self.current_assembly_region_rc = (-1, -1)

        w_sum = float(w_grasp + w_manip + w_dist + w_rot)
        if w_sum <= 1e-9:
            raise ValueError("权重之和不能为 0")
        self.w_grasp = float(w_grasp / w_sum)
        self.w_manip = float(w_manip / w_sum)
        self.w_dist = float(w_dist / w_sum)
        self.w_rot = float(w_rot / w_sum)

        self.asm = AssemblyDef.load(self.asmdef_path)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

        if part_order is None:
            self.part_order = [
                s.part_id for s in self.asm.steps
                if s.part_id in self.asm.part_ids and s.part_id in self.world_poses
            ]
        else:
            self.part_order = [
                p for p in part_order
                if p in self.asm.part_ids and p in self.world_poses
            ]

        self.table_x_range, self.table_y_range, self.table_top_z = _table_info(
            self.config_yaml, self.table_name, self.table_margin
        )

        self.env_obs = [] if self.ignore_env else _load_env_obstacles(self.config_yaml)

        self.robot = pda.DualPantheraHTNoBody(
            pos=self.robot_base_pos,
            rotmat=self.robot_base_rotmat,
            arm_y_offset=DUAL_ARM_Y_OFFSET,
            enable_cc=True,
        )
        self.robot.lft_arm.goto_given_conf(HOME_JV)
        self.robot.rgt_arm.goto_given_conf(HOME_JV)

        self.grasp_cache: Dict[str, GraspCollection] = {}
        self.grasp_file_for_part: Dict[str, str] = {}
        self._load_grasps()

        self.rot_cands: Dict[str, List[RotCandidate]] = {}
        self._precompute_rot_candidates()

        self.staging_models = self._make_staging_models()
        self.goal_models = self._make_goal_models()

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def _load_grasps(self):
        print("\n========== Grasp pickle 检查 ==========")
        for pid in self.part_order:
            mp = self.asm.model_path(pid)
            mid = _model_id_for_part(self.asm, pid)
            pkl = _find_grasp_pickle(pid, mid, mp, self.grasp_dir, self.grasp_map)
            if pkl is None:
                print(f"{pid:16s}: MISSING")
                continue
            if pkl not in self.grasp_cache:
                self.grasp_cache[pkl] = GraspCollection.load_from_disk(file_name=pkl)
            self.grasp_file_for_part[pid] = pkl
            print(f"{pid:16s}: {pkl}  n={len(self.grasp_cache[pkl])}")

    def _precompute_rot_candidates(self):
        print("\n========== 自动旋转候选 ==========")
        print(f"use_flatsurface        = {self.use_flatsurface}")
        print(f"fs_stability_threshold = {self.fs_stability_threshold}")

        for pid in self.part_order:
            mp = self.asm.model_path(pid)
            verts = _load_mesh_vertices(mp)

            # 1) 优先使用 flatsurface.py 计算出的稳定摆放姿态；
            # 2) 再追加旧的 identity/90° 候选作为 fallback；
            # 3) 按旋转矩阵去重。
            pose_rots: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
            if self.use_flatsurface:
                pose_rots.extend(_stable_pose_candidates_from_flatsurface(
                    mp,
                    stability_threshold=self.fs_stability_threshold,
                    max_candidates=max(self.max_rot_candidates * 2, 16),
                ))

            pose_rots.extend(_fallback_rot_candidates_with_names())
            pose_rots = _dedupe_pose_rot_candidates(pose_rots)

            cands: List[RotCandidate] = []

            _, _, identity_extent = _bounds_after_rotation(verts, np.eye(3))
            is_flat = float(identity_extent.min() / max(identity_extent.max(), 1e-9)) < 0.22

            for rot_name, fs_pos, R in pose_rots:
                bmin, bmax, extent = _bounds_after_rotation(verts, R)
                z_offset = self.table_top_z + self.table_clearance - float(bmin[2])
                is_identity = np.allclose(R, np.eye(3), atol=1e-6)
                from_flatsurface = str(rot_name).startswith("fs_")

                if is_identity:
                    tag = "identity"
                elif from_flatsurface and is_flat and extent[2] > identity_extent[2] * 2.0:
                    tag = "fs_upright_lifted"
                elif from_flatsurface:
                    tag = "fs_stable_lifted"
                elif is_flat and extent[2] > identity_extent[2] * 2.0:
                    tag = "auto_upright_lifted"
                else:
                    tag = "rot90_lifted"

                cands.append(RotCandidate(
                    rotmat=np.asarray(R, dtype=float),
                    z_offset=float(z_offset),
                    tag=tag,
                    extent=np.asarray(extent, dtype=float),
                    footprint=np.asarray(extent[:2], dtype=float),
                    rot_name=str(rot_name),
                    fs_pos=None if fs_pos is None else np.asarray(fs_pos, dtype=float),
                ))

            # 排序策略：
            #   identity 永远保留且靠前；
            #   flatsurface 稳定姿态优先于普通 90° fallback；
            #   在同级内部，较低高度/较小旋转代价更靠前。
            def _sort_key(c: RotCandidate):
                if c.tag == "identity":
                    group = 0
                elif c.rot_name.startswith("fs_"):
                    group = 1
                elif c.tag == "auto_upright_lifted":
                    group = 2
                else:
                    group = 3
                return (group, float(c.extent[2]), float(np.linalg.norm(c.extent[:2])))

            cands.sort(key=_sort_key)
            self.rot_cands[pid] = cands[:self.max_rot_candidates]

            print(f"{pid:16s}: {len(self.rot_cands[pid])} candidates")
            for i, c in enumerate(self.rot_cands[pid][:6]):
                print(
                    f"  [{i}] tag={c.tag:20s} "
                    f"rot={c.rot_name:12s} "
                    f"z_offset={c.z_offset:.4f} "
                    f"extent={np.round(c.extent, 4).tolist()}"
                )

    def _make_staging_models(self):
        out = {}
        for pid in self.part_order:
            mp = self.asm.model_path(pid)
            cm = make_collision_model(mp, cdprim_type=self.cdprim_type)
            cm.pos = np.zeros(3)
            cm.rotmat = np.eye(3)
            cm._sealp_part_id = pid
            cm._sealp_role = "weighted_staging"
            out[pid] = cm
        return out

    def _make_goal_models(self):
        out = {}
        for pid in self.part_order:
            if pid not in self.world_poses:
                continue
            mp = self.asm.model_path(pid)
            gp, gr = self.world_poses[pid]
            cm = make_collision_model(mp, cdprim_type=self.cdprim_type)
            cm.pos = np.asarray(gp, dtype=float)
            cm.rotmat = np.asarray(gr, dtype=float)
            cm._sealp_part_id = pid
            cm._sealp_role = "weighted_goal"
            out[pid] = cm
        return out


    # --------------------------------------------------------
    # 装配区域候选
    # --------------------------------------------------------

    def _set_assembly_station(self, fixture_pos: np.ndarray, region_id: str = "fixed", rc: Tuple[int, int] = (-1, -1)) -> None:
        """更新当前装配区中心，并同步 asm world poses / goal models。"""
        self.fixture_pos = np.asarray(fixture_pos, dtype=float)
        self.current_assembly_region_id = str(region_id)
        self.current_assembly_region_rc = tuple(rc)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

        # 如果 goal_models 已经存在，重新生成 goal collision models，保证 L2/L3 用的是最新装配区。
        if hasattr(self, "goal_models"):
            self.goal_models = self._make_goal_models()

    def _first_part_id(self) -> Optional[str]:
        if not self.part_order:
            return None
        return self.part_order[0]

    def _apply_first_part_as_assembled(self, layout: Optional[LayoutCandidate] = None) -> None:
        """把第一步零件直接放到当前装配区，等价于第一步已经装好。

        对 tower 来说，第一步通常是 base_plate。
        """
        if not self.preassemble_first_part:
            return
        first_pid = self._first_part_id()
        if first_pid is None or first_pid not in self.world_poses:
            return

        gp, gr = self.world_poses[first_pid]
        if first_pid in self.staging_models:
            self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
            self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()

        if layout is not None:
            layout.xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            layout.chosen_rotmat[first_pid] = np.asarray(gr, dtype=float).copy()
            layout.z_offset[first_pid] = float(gp[2])
            layout.pose_tag[first_pid] = "preassembled_at_assembly_region"
            layout.rot_name[first_pid] = "goal_pose"
            layout.arm_choice[first_pid] = "preassembled"
            # 第一件已经装好，不参与抓取规划；给一个中性值，避免 score 中 min/count 异常。
            layout.grasp_counts[first_pid] = int(NORM_GRASP_MEAN_TARGET)
            layout.per_part_dist[first_pid] = 0.0
            layout.per_part_manip[first_pid] = 0.0
            layout.per_part_rot_angle[first_pid] = 0.0

    def _assembly_region_candidates(self) -> List[Tuple[str, Tuple[int, int], np.ndarray]]:
        """把 work_table 粗分为 grid x grid 个区域，返回每块中心点作为装配区候选。

        返回:
            [(region_id, (row, col), fixture_pos), ...]

        row 对应 y 方向分块，col 对应 x 方向分块。
        """
        if not self.plan_assembly_region:
            return [("fixed", (-1, -1), self.fixture_pos.copy())]

        grid = max(1, int(self.assembly_grid))
        first_pid = self._first_part_id()

        # 用第一步零件的 identity footprint 约束装配中心，避免第一件一开始就出桌面。
        if first_pid is not None and first_pid in self.rot_cands:
            first_fp = np.asarray(self.rot_cands[first_pid][0].footprint, dtype=float)
        else:
            first_fp = np.zeros(2)

        x_min, x_max = self.table_x_range
        y_min, y_max = self.table_y_range
        x_safe_min = x_min + float(first_fp[0]) / 2.0
        x_safe_max = x_max - float(first_fp[0]) / 2.0
        y_safe_min = y_min + float(first_fp[1]) / 2.0
        y_safe_max = y_max - float(first_fp[1]) / 2.0

        if x_safe_min > x_safe_max:
            x_safe_min, x_safe_max = x_min, x_max
        if y_safe_min > y_safe_max:
            y_safe_min, y_safe_max = y_min, y_max

        x_edges = np.linspace(x_min, x_max, grid + 1)
        y_edges = np.linspace(y_min, y_max, grid + 1)

        out: List[Tuple[str, Tuple[int, int], np.ndarray]] = []
        for r in range(grid):
            for c in range(grid):
                raw_x = float((x_edges[c] + x_edges[c + 1]) / 2.0)
                raw_y = float((y_edges[r] + y_edges[r + 1]) / 2.0)
                x = float(np.clip(raw_x, x_safe_min, x_safe_max))
                y = float(np.clip(raw_y, y_safe_min, y_safe_max))
                region_id = f"r{r}_c{c}"
                out.append((region_id, (r, c), np.array([x, y, self.table_top_z], dtype=float)))

        # 去重：如果第一件太大导致多个角落被 clip 到同一个点，仍保留 region_id，
        # 这样输出还能看出是哪个 coarse block 被尝试。
        print("\n========== Assembly Region Candidates ==========")
        print(f"plan_assembly_region = {self.plan_assembly_region}")
        print(f"assembly_grid        = {grid} x {grid}")
        print(f"preassemble_first    = {self.preassemble_first_part}")
        for rid, rc, pos in out:
            print(f"  {rid:6s} rc={rc} center={np.round(pos, 4).tolist()}")
        return out


    # --------------------------------------------------------
    # layout 采样
    # --------------------------------------------------------

    def _xy_bounds_for_part_and_cand(self, pid: str, cand: RotCandidate):
        fx, fy = cand.footprint
        xlo, xhi = self.table_x_range
        ylo, yhi = self.table_y_range
        return (
            xlo + fx / 2.0,
            xhi - fx / 2.0,
        ), (
            ylo + fy / 2.0,
            yhi - fy / 2.0,
        )

    def _apply_staging_pose(self, pid: str, xy: np.ndarray, cand: RotCandidate):
        cm = self.staging_models[pid]
        cm.pos = np.array([float(xy[0]), float(xy[1]), float(cand.z_offset)], dtype=float)
        cm.rotmat = np.asarray(cand.rotmat, dtype=float)

    def _pairwise_collision(self, active_pids: Optional[List[str]] = None) -> Optional[str]:
        ids = active_pids or list(self.staging_models.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if self.staging_models[a].is_mcdwith(self.staging_models[b]):
                    return f"{a} vs {b}"
        return None

    def sample_collision_free_xy(self,
                                 rng: np.random.Generator,
                                 max_attempts_per_part: int = 150) -> Optional[Dict[str, np.ndarray]]:
        """在 work_table 上为所有零件随机采样一个尽量不重叠的初始 layout。

        如果 self.preassemble_first_part=True：
            第一件零件直接放在当前装配区中心，等价于它一开始已经装好；
            后续只给其它零件随机采样初始位置。
        """
        xy: Dict[str, np.ndarray] = {}
        placed: List[str] = []

        first_pid = self._first_part_id() if self.preassemble_first_part else None

        if first_pid is not None and first_pid in self.world_poses:
            gp, gr = self.world_poses[first_pid]
            xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            if first_pid in self.staging_models:
                self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
                self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()
            placed.append(first_pid)

        # 大件先放，显著降低随机碰撞概率；第一件若已预装，则跳过随机采样。
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

            ok = False
            for _ in range(max_attempts_per_part):
                p = np.array([
                    float(rng.uniform(xlo, xhi)),
                    float(rng.uniform(ylo, yhi)),
                ], dtype=float)
                self._apply_staging_pose(pid, p, cand0)

                # 与已放零件不碰撞即可，包括已经装在装配区的第一件。
                collision = False
                for q in placed:
                    if q in self.staging_models and self.staging_models[pid].is_mcdwith(self.staging_models[q]):
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

    # --------------------------------------------------------
    # 评估
    # --------------------------------------------------------

    def _grasp_collection(self, pid: str) -> Optional[GraspCollection]:
        pkl = self.grasp_file_for_part.get(pid)
        if pkl is None:
            return None
        return self.grasp_cache.get(pkl)

    def _arm_order(self, pid: str) -> Tuple[str, ...]:
        if pid.endswith("_r") or pid.endswith("br") or pid.endswith("fr"):
            return ("rgt", "lft")
        return ("lft", "rgt")

    def _step_obstacles(self, current_pid: str, placed: set) -> List:
        obs = list(self.env_obs)
        for pid in placed:
            if pid in self.goal_models:
                obs.append(self.goal_models[pid])
        for pid in self.part_order:
            if pid == current_pid or pid in placed:
                continue
            if pid in self.staging_models:
                obs.append(self.staging_models[pid])
        return obs

    def _planner_obstacles(self, obs: List) -> List:
        """给 reason_common_gids 使用的 obstacle_list。

        mesh:      使用 mesh/triangles CollisionModel，推荐；
        env_only:  只保留 work_table 等环境；
        none:      不把 obstacle 传进 WRS 的 gripper collision，避免 box/cdprim
                   误杀；仍会保留 staging 零件之间的 mesh/triangles 复检。
        """
        mode = self.planner_obstacle_mode
        if mode == "none":
            return []
        if mode == "env_only":
            return [o for o in obs if getattr(o, "_sealp_role", None) == "environment_obstacle"]
        # mesh / box 都返回完整动态障碍；区别由 cdprim_type 决定。
        return obs

    def _endpoint_manip(self, arm, gc, sp, sr, gp, gr, obs) -> float:
        if check_pose_reachability is None:
            return 0.0
        try:
            pick = check_pose_reachability(arm, sp, sr, gc, obs, max_grasps=5)
            place = check_pose_reachability(arm, gp, gr, gc, obs, max_grasps=5)
            vals = [
                float(getattr(pick, "best_manipulability", 0.0)),
                float(getattr(place, "best_manipulability", 0.0)),
            ]
            vals = [v for v in vals if np.isfinite(v) and v > 0.0]
            return float(np.mean(vals)) if vals else 0.0
        except Exception:
            return 0.0

    def evaluate_layout(self, layout: LayoutCandidate) -> bool:
        layout.assembly_region_id = self.current_assembly_region_id
        layout.assembly_region_rc = self.current_assembly_region_rc
        layout.assembly_station_pos = self.fixture_pos.copy()
        layout.assembly_station_rotmat = self.fixture_rotmat.copy()

        # 先用 identity 候选初始化所有零件，这样 step0 的 obstacle 也是有效随机位置。
        for pid in self.part_order:
            if pid in layout.xy:
                self._apply_staging_pose(pid, layout.xy[pid], self.rot_cands[pid][0])

        placed = set()
        fail_detail = {}

        # 第一件直接对齐当前装配区中心，视为已经装好，不再做 pick-and-place。
        first_pid = self._first_part_id() if self.preassemble_first_part else None
        if first_pid is not None:
            self._apply_first_part_as_assembled(layout)
            placed.add(first_pid)

        for pid in self.part_order:
            if pid == first_pid:
                continue
            if pid not in self.world_poses:
                continue

            gc = self._grasp_collection(pid)
            if gc is None or len(gc) == 0:
                layout.fail_reason = f"{pid}: grasp collection missing or empty"
                return False

            gp, gr = self.world_poses[pid]
            best_record = None
            fail_counter = {
                "pair_collision": 0,
                "no_common_gids": 0,
                "reason_exception": 0,
            }

            for cand in self.rot_cands[pid]:
                self._apply_staging_pose(pid, layout.xy[pid], cand)

                # 候选姿态下，所有 staging 不能互相穿模
                hit = self._pairwise_collision()
                if hit:
                    fail_counter["pair_collision"] += 1
                    continue

                sp = self.staging_models[pid].pos.copy()
                sr = self.staging_models[pid].rotmat.copy()
                obs = self._step_obstacles(pid, placed)
                planner_obs = self._planner_obstacles(obs)

                for arm_tag in self._arm_order(pid):
                    arm = self.robot.rgt_arm if arm_tag == "rgt" else self.robot.lft_arm
                    planner = PickPlacePlanner(robot=arm)

                    try:
                        gids = planner.reason_common_gids(
                            grasp_collection=gc,
                            goal_pose_list=[(sp, sr), (gp, gr)],
                            obstacle_list=planner_obs,
                        )
                    except Exception:
                        fail_counter["reason_exception"] += 1
                        gids = []

                    n = len(gids)
                    if n <= 0:
                        fail_counter["no_common_gids"] += 1
                        continue

                    dist = float(np.linalg.norm(np.asarray(sp) - np.asarray(gp)))
                    rot_ang = _rot_angle(sr, gr)
                    manip = self._endpoint_manip(arm, gc, sp, sr, gp, gr, planner_obs)

                    # 单零件临时得分，用于在该零件多个候选姿态中选一个最好姿态。
                    # identity 可行时通常旋转代价最小，因此会自然胜出；
                    # 不可行时才会选站立/侧放。
                    grasp_s = _hill(min(n, 60), NORM_GRASP_MEAN_TARGET, NORM_GRASP_HILL_K)
                    manip_s = float(1.0 - np.exp(-max(manip, 0.0) / NORM_MANIP_TARGET))
                    dist_s = float(np.exp(-max(dist, 0.0) / NORM_DIST_DECAY))
                    rot_s = float(np.exp(-max(rot_ang, 0.0) / NORM_ROT_DECAY))
                    if cand.tag == "identity":
                        rot_s = min(1.0, rot_s + 0.10)

                    part_score = (
                        self.w_grasp * grasp_s
                        + self.w_manip * manip_s
                        + self.w_dist * dist_s
                        + self.w_rot * rot_s
                    )

                    record = (
                        part_score, n, manip, dist, rot_ang,
                        arm_tag, cand, sp.copy(),
                    )
                    if best_record is None or record[0] > best_record[0]:
                        best_record = record

            if best_record is None:
                fail_detail[pid] = dict(fail_counter)
                layout.fail_reason = f"{pid}: all rotation/arm candidates failed; fail_counter={fail_counter}"
                return False

            part_score, n, manip, dist, rot_ang, arm_tag, cand, sp = best_record

            # commit 当前零件的最佳姿态
            self._apply_staging_pose(pid, layout.xy[pid], cand)

            layout.chosen_rotmat[pid] = cand.rotmat.copy()
            layout.z_offset[pid] = float(cand.z_offset)
            layout.pose_tag[pid] = cand.tag
            layout.rot_name[pid] = cand.rot_name
            layout.arm_choice[pid] = arm_tag
            layout.grasp_counts[pid] = int(n)
            layout.per_part_dist[pid] = float(dist)
            layout.per_part_manip[pid] = float(manip)
            layout.per_part_rot_angle[pid] = float(rot_ang)

            placed.add(pid)

        # 最终复检所有 staging 不碰撞
        hit = self._pairwise_collision()
        if hit:
            layout.fail_reason = f"final staging collision: {hit}"
            return False

        # 计算 layout 级综合得分，模仿 find_optimal_layout.py：
        # grasp 用 min + mean 双项，避免某个零件成为瓶颈。
        counts = list(layout.grasp_counts.values())
        if not counts:
            layout.fail_reason = "no part evaluated"
            return False

        n_min = float(min(counts))
        n_mean = float(np.mean(counts))
        grasp_min_s = _hill(n_min, NORM_GRASP_MIN_TARGET, NORM_GRASP_HILL_K)
        grasp_mean_s = _hill(n_mean, NORM_GRASP_MEAN_TARGET, NORM_GRASP_HILL_K)
        grasp_score = 0.70 * grasp_min_s + 0.30 * grasp_mean_s

        avg_manip = float(np.mean(list(layout.per_part_manip.values()))) if layout.per_part_manip else 0.0
        manip_score = float(1.0 - np.exp(-max(avg_manip, 0.0) / NORM_MANIP_TARGET))

        avg_dist = float(np.mean(list(layout.per_part_dist.values()))) if layout.per_part_dist else 0.0
        dist_score = float(np.exp(-max(avg_dist, 0.0) / NORM_DIST_DECAY))

        avg_rot = float(np.mean(list(layout.per_part_rot_angle.values()))) if layout.per_part_rot_angle else 0.0
        rot_score = float(np.exp(-max(avg_rot, 0.0) / NORM_ROT_DECAY))

        layout.grasp_score_norm = grasp_score
        layout.manip_score_norm = manip_score
        layout.dist_score_norm = dist_score
        layout.rot_score_norm = rot_score
        layout.layout_score = (
            self.w_grasp * grasp_score
            + self.w_manip * manip_score
            + self.w_dist * dist_score
            + self.w_rot * rot_score
        )
        layout.l2_pass = True
        return True


    # --------------------------------------------------------
    # L3：全流程动态避障验证
    # --------------------------------------------------------

    def _apply_final_layout(self, layout: LayoutCandidate) -> None:
        """把 L2 选中的最终 staging pose 套到所有 staging CollisionModel。"""
        for pid in self.part_order:
            if pid not in self.staging_models:
                continue
            if pid not in layout.xy:
                continue
            if pid not in layout.chosen_rotmat:
                continue
            self.staging_models[pid].pos = np.array([
                float(layout.xy[pid][0]),
                float(layout.xy[pid][1]),
                float(layout.z_offset.get(pid, 0.0)),
            ], dtype=float)
            self.staging_models[pid].rotmat = np.asarray(layout.chosen_rotmat[pid], dtype=float)

    def _l3_obstacles(self, current_pid: str, placed: set, mode: str = "mesh") -> List:
        """L3 真实路径规划时的 step-aware 障碍物。

        动态语义：
            - 已经装好的零件：使用 goal pose，作为装配区障碍；
            - 还没装的零件：使用 staging pose，作为取料区障碍；
            - 当前正在搬运的零件：不放进 obstacle，因为它由 TransportPrimitive 作为 held object 处理；
            - work_table 等环境：按 mode 决定是否加入。
        """
        obs = []
        if mode != "none":
            if mode in ("mesh", "box", "env_only"):
                obs.extend(self.env_obs)
            if mode in ("mesh", "box"):
                for pid in placed:
                    if pid in self.goal_models:
                        obs.append(self.goal_models[pid])
                for pid in self.part_order:
                    if pid == current_pid or pid in placed:
                        continue
                    if pid in self.staging_models:
                        obs.append(self.staging_models[pid])
        return obs

    def validate_full_sequence_l3(
        self,
        layout: LayoutCandidate,
        obstacle_mode: str = "mesh",
        verbose: bool = True,
    ) -> bool:
        """对一个 L2 layout 进行严格 L3 全流程验证。

        这一步比 reason_common_gids 更严格：
            pick -> depart -> transport/RRT -> place approach -> place -> depart
        全流程中会检查机器人、夹爪、被抓物体与 step-aware 障碍物的碰撞。
        """
        if layout.assembly_station_pos is not None and np.asarray(layout.assembly_station_pos).shape == (3,):
            self._set_assembly_station(
                layout.assembly_station_pos,
                region_id=layout.assembly_region_id,
                rc=layout.assembly_region_rc,
            )

        self._apply_final_layout(layout)
        _patch_rrt_for_l3()
        _reset_robot_for_l3(self.robot)

        lft_transport = TransportPrimitive(self.robot.lft_arm)
        rgt_transport = TransportPrimitive(self.robot.rgt_arm)
        placed = set()
        first_pid = self._first_part_id() if self.preassemble_first_part else None
        if first_pid is not None:
            placed.add(first_pid)

        if verbose:
            print("\n========== L3 Full-sequence validation ==========")
            print(f"obstacle_mode = {obstacle_mode}")

        try:
            for step_idx, pid in enumerate(self.part_order):
                if pid not in self.world_poses:
                    continue
                if pid == first_pid:
                    if verbose:
                        print(f"  [SKIP] step={step_idx} pid={pid:14s} already assembled at region={layout.assembly_region_id}")
                    continue

                arm_tag = layout.arm_choice.get(pid, "lft")
                transport = rgt_transport if arm_tag == "rgt" else lft_transport

                sp = self.staging_models[pid].pos.copy()
                sr = self.staging_models[pid].rotmat.copy()
                gp, gr = self.world_poses[pid]
                gc = self._grasp_collection(pid)
                if gc is None or len(gc) == 0:
                    layout.l3_fail_reason = f"L3 step={step_idx} {pid}: grasp collection missing"
                    return False

                obj_cm = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
                obj_cm.pos = sp.copy()
                obj_cm.rotmat = sr.copy()
                obj_cm._sealp_part_id = pid
                obj_cm._sealp_role = "l3_moving_object"

                obs = self._l3_obstacles(pid, placed, mode=obstacle_mode)
                try:
                    res = transport.plan(
                        obj_cmodel=obj_cm,
                        grasp_collection=gc,
                        goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                        obstacle_list=obs,
                        approach_distance=APPROACH_DIST,
                        depart_distance=PICK_DEPART_DIST,
                        linear_granularity=LINEAR_GRANULARITY,
                        **_transport_kwargs(),
                    )
                except Exception as e:
                    layout.l3_fail_reason = (
                        f"L3 step={step_idx} {pid} {arm_tag}: "
                        f"exception {type(e).__name__}: {e!r}"
                    )
                    if verbose:
                        print(f"  [FAIL] {layout.l3_fail_reason}")
                    return False

                success = bool(getattr(res, "success", False))
                if not success:
                    err = getattr(res, "error_msg", "") or "no plan"
                    layout.l3_fail_reason = f"L3 step={step_idx} {pid} {arm_tag}: {err}"
                    if verbose:
                        print(f"  [FAIL] {layout.l3_fail_reason}")
                    return False

                placed.add(pid)
                if verbose:
                    print(f"  [OK] step={step_idx} pid={pid:14s} arm={arm_tag} obs={len(obs)}")

            layout.l3_pass = True
            layout.l3_fail_reason = ""
            if verbose:
                print("[OK] L3 full sequence passed.")
            return True

        finally:
            _reset_robot_for_l3(self.robot)


    # --------------------------------------------------------
    # Random search
    # --------------------------------------------------------

    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "mesh",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        """加权随机搜索。

        流程：
            1. 在 work_table 上随机采样 n_samples 个 layout；
            2. 每个 layout 走 L2 快速评估并计算综合得分；
            3. 对 L2 通过的 layout 按 score 降序排序；
            4. 如果 enable_l3=True，对 top-k 做 TransportPrimitive 全流程验证；
            5. 返回 L3 通过的最高分 layout。若 require_l3=False 且 L3 全失败，则回退 L2 第一名。
        """
        rng = np.random.default_rng(seed)
        best_l2: Optional[LayoutCandidate] = None
        feasible: List[LayoutCandidate] = []

        print("\n========== Weighted Random Search ==========")
        print(f"n_samples             = {n_samples}")
        print(f"seed                  = {seed}")
        print(f"work_table x_range    = {self.table_x_range}")
        print(f"work_table y_range    = {self.table_y_range}")
        print(f"table_top_z           = {self.table_top_z:.6f}")
        print(f"weights               = grasp {self.w_grasp:.2f}, manip {self.w_manip:.2f}, dist {self.w_dist:.2f}, rot {self.w_rot:.2f}")
        print(f"part_order            = {self.part_order}")
        print(f"L3                    = {enable_l3}, top_k={l3_top_k}, obstacle_mode={l3_obstacle_mode}, require_l3={require_l3}")

        assembly_regions = self._assembly_region_candidates()

        t0 = time.time()
        n_generated = 0
        n_reject_sample = 0

        for i in range(n_samples):
            # coarse 装配区搜索：在 3x3 work_table 区域中心之间循环尝试。
            region_id, region_rc, region_pos = assembly_regions[i % len(assembly_regions)]
            self._set_assembly_station(region_pos, region_id=region_id, rc=region_rc)

            xy = None
            for _ in range(max_resample_layout):
                xy = self.sample_collision_free_xy(rng)
                if xy is not None:
                    break

            if xy is None:
                n_reject_sample += 1
                if verbose:
                    print(f"#{i:03d} SAMPLE_FAIL  无法在 work_table 上采样到不重叠初始 xy")
                continue

            n_generated += 1
            cand = LayoutCandidate(xy=xy)
            ok = self.evaluate_layout(cand)

            if not ok:
                if verbose:
                    print(f"#{i:03d} FAIL  {cand.fail_reason}")
                continue

            feasible.append(cand)
            if best_l2 is None or cand.layout_score > best_l2.layout_score:
                best_l2 = cand

            if verbose:
                print(
                    f"#{i:03d} L2_OK  score={cand.layout_score:.4f} "
                    f"G={cand.grasp_score_norm:.3f} "
                    f"M={cand.manip_score_norm:.3f} "
                    f"D={cand.dist_score_norm:.3f} "
                    f"R={cand.rot_score_norm:.3f} "
                    f"region={cand.assembly_region_id} "
                    f"counts={cand.grasp_counts} "
                    f"arms={cand.arm_choice} "
                    f"pose={cand.pose_tag} "
                    f"rot={cand.rot_name}"
                )

        feasible.sort(key=lambda c: c.layout_score, reverse=True)
        dt_l2 = time.time() - t0

        print("\n========== L2 Search Summary ==========")
        print(f"generated layouts     = {n_generated}/{n_samples}")
        print(f"sample rejected       = {n_reject_sample}")
        print(f"L2 feasible layouts   = {len(feasible)}/{n_samples}")
        print(f"L2 elapsed            = {dt_l2:.1f}s")

        if not feasible:
            print("[FAIL] L2 没有找到可行 layout。")
            return None

        print("[OK] best L2 layout:")
        print(f"  score               = {feasible[0].layout_score:.4f}")
        print(f"  components          = G {feasible[0].grasp_score_norm:.3f}, M {feasible[0].manip_score_norm:.3f}, D {feasible[0].dist_score_norm:.3f}, R {feasible[0].rot_score_norm:.3f}")
        print(f"  assembly_region     = {feasible[0].assembly_region_id}, rc={feasible[0].assembly_region_rc}, pos={np.round(feasible[0].assembly_station_pos, 4).tolist()}")
        print(f"  grasp_counts        = {feasible[0].grasp_counts}")
        print(f"  arm_choice          = {feasible[0].arm_choice}")
        print(f"  pose_tag            = {feasible[0].pose_tag}")
        print(f"  rot_name            = {feasible[0].rot_name}")

        if not enable_l3:
            return feasible[0]

        # L3：对 top-k 做真正全流程动态避障验证
        k = min(int(l3_top_k), len(feasible))
        print("\n========== L3 Top-k full-process validation ==========")
        print(f"try top-k             = {k}")

        for rank, cand in enumerate(feasible[:k], start=1):
            print(f"\n[L3] try rank={rank}/{k}, L2_score={cand.layout_score:.4f}")
            ok_l3 = self.validate_full_sequence_l3(
                cand,
                obstacle_mode=l3_obstacle_mode,
                verbose=True,
            )
            if ok_l3:
                print(f"[OK] L3 passed, selected rank={rank}, score={cand.layout_score:.4f}")
                return cand
            else:
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")

        if require_l3:
            print("\n[FAIL] L2 top-k 全部未通过 L3，全流程动态避障未找到严格可行布局。")
            return None

        print("\n[WARN] L3 全失败，但 require_l3=False，回退保存 L2 最高分 layout。")
        return feasible[0]

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    def save_layout(self, layout: LayoutCandidate, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{self.output_name}.layout")

        if layout.assembly_station_pos is not None and np.asarray(layout.assembly_station_pos).shape == (3,):
            self._set_assembly_station(
                layout.assembly_station_pos,
                region_id=layout.assembly_region_id,
                rc=layout.assembly_region_rc,
            )

        staging_positions = {}
        for pid in self.part_order:
            staging_positions[pid] = (
                np.array([
                    float(layout.xy[pid][0]),
                    float(layout.xy[pid][1]),
                    float(layout.z_offset[pid]),
                ], dtype=float),
                np.asarray(layout.chosen_rotmat[pid], dtype=float),
            )

        ws = WorkspaceLayout(
            robot_base_pos=self.robot_base_pos.copy(),
            robot_base_rotmat=self.robot_base_rotmat.copy(),
            assembly_station_pos=np.asarray(layout.assembly_station_pos, dtype=float).copy(),
            assembly_station_rotmat=np.asarray(layout.assembly_station_rotmat, dtype=float).copy(),
            staging_positions=staging_positions,
            name=self.output_name,
            metadata={
                "search_method": "weighted_random_on_work_table",
                "l3_pass": bool(layout.l3_pass),
                "l3_fail_reason": str(layout.l3_fail_reason),
                "score": float(layout.layout_score),
                "score_components": {
                    "grasp": float(layout.grasp_score_norm),
                    "manip": float(layout.manip_score_norm),
                    "dist": float(layout.dist_score_norm),
                    "rot": float(layout.rot_score_norm),
                },
                "weights": {
                    "grasp": float(self.w_grasp),
                    "manip": float(self.w_manip),
                    "dist": float(self.w_dist),
                    "rot": float(self.w_rot),
                },
                "part_order": list(self.part_order),
                "assembly_region_id": str(layout.assembly_region_id),
                "assembly_region_rc": list(layout.assembly_region_rc),
                "assembly_station_pos": np.asarray(layout.assembly_station_pos, dtype=float).tolist(),
                "preassemble_first_part": bool(self.preassemble_first_part),
                "grasp_counts": dict(layout.grasp_counts),
                "arm_choice": dict(layout.arm_choice),
                "pose_tag": dict(layout.pose_tag),
                "rot_name": dict(layout.rot_name),
                "z_offsets": {k: float(v) for k, v in layout.z_offset.items()},
                "per_part_dist": dict(layout.per_part_dist),
                "per_part_manip": dict(layout.per_part_manip),
                "per_part_rot_angle": dict(layout.per_part_rot_angle),
                "table_top_z": float(self.table_top_z),
                "table_x_range": list(map(float, self.table_x_range)),
                "table_y_range": list(map(float, self.table_y_range)),
                "rotation_lift_rule": "z_offset = table_top_z + clearance - rotated_z_min",
            },
        )
        ws.save(out_path)
        print(f"\n[OK] 保存 layout -> {out_path}")

        debug_path = os.path.join(output_dir, f"{self.output_name}_debug.json")
        debug = {
            "layout_path": out_path,
            "score": float(layout.layout_score),
            "xy": {k: [float(v[0]), float(v[1])] for k, v in layout.xy.items()},
            "z_offset": {k: float(v) for k, v in layout.z_offset.items()},
            "pose_tag": dict(layout.pose_tag),
            "rot_name": dict(layout.rot_name),
            "assembly_region_id": str(layout.assembly_region_id),
            "assembly_region_rc": list(layout.assembly_region_rc),
            "assembly_station_pos": np.asarray(layout.assembly_station_pos, dtype=float).tolist(),
            "grasp_counts": dict(layout.grasp_counts),
            "arm_choice": dict(layout.arm_choice),
            "score_components": {
                "grasp": float(layout.grasp_score_norm),
                "manip": float(layout.manip_score_norm),
                "dist": float(layout.dist_score_norm),
                "rot": float(layout.rot_score_norm),
            },
            "chosen_rotmat": {
                k: np.asarray(R).tolist()
                for k, R in layout.chosen_rotmat.items()
            },
        }
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
        print(f"[OK] 保存 debug -> {debug_path}")

        return out_path


# ============================================================
# CLI
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Weighted random initial placement search on work_table."
    )
    parser.add_argument("--asmdef", default=DEFAULT_ASMDEF, help="asmdef 路径；PyCharm 右键运行时使用默认 Tower asmdef")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml 路径")
    parser.add_argument("--grasp-dir", default=DEFAULT_GRASP_DIR, help="grasp pickle 文件夹")
    parser.add_argument("--part-order", default=DEFAULT_PART_ORDER, help="逗号分隔的装配顺序；为空则使用 asmdef 顺序")
    parser.add_argument("--grasp-map-json", default="", help="可选，显式指定 grasp 文件映射")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录，默认 sealp/examples/layout/_output")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="输出 layout 名称")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="随机采样 layout 数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")

    parser.add_argument("--fixture-pos", default=DEFAULT_FIXTURE_POS, help="装配 fixture 世界坐标")
    parser.add_argument("--robot-base-pos", default=DEFAULT_ROBOT_BASE_POS, help="机器人基座世界坐标")

    parser.add_argument("--disable-assembly-region-search", action="store_true",
                        help="关闭 3x3 装配区域搜索，退回使用 --fixture-pos 固定装配中心。")
    parser.add_argument("--assembly-grid", type=int, default=DEFAULT_ASSEMBLY_GRID,
                        help="coarse 装配区域网格数量，默认 3 表示 3x3=9 块。")
    parser.add_argument("--disable-preassemble-first", action="store_true",
                        help="关闭第一件零件直接放在装配区中心的策略；默认第一件视为已装好。")
    parser.add_argument("--disable-flatsurface", action="store_true",
                        help="关闭 flatsurface.py 稳定摆放姿态候选，退回 90 度候选。")
    parser.add_argument("--fs-stability-threshold", type=float, default=DEFAULT_FS_STABILITY_THRESHOLD,
                        help="flatsurface 稳定性阈值，默认 0.10。")

    parser.add_argument("--table-name", default="work_table", help="config 中桌子障碍名")
    parser.add_argument("--table-margin", type=float, default=DEFAULT_TABLE_MARGIN, help="work_table 边界内缩")
    parser.add_argument("--table-clearance", type=float, default=DEFAULT_TABLE_CLEARANCE, help="旋转后 z 修正的安全间隙")
    parser.add_argument("--max-rot-candidates", type=int, default=DEFAULT_MAX_ROT_CANDIDATES, help="每个零件最多旋转候选数")
    parser.add_argument("--ignore-env", action="store_true", help="调试用：评估时忽略 work_table 等环境障碍")
    parser.add_argument(
        "--cdprim-type",
        default=DEFAULT_CDPRIM_TYPE,
        help="CollisionModel 碰撞类型，默认 triangles，尽量按 STL mesh 检测；不支持时自动退回。",
    )
    parser.add_argument(
        "--planner-obstacle-mode",
        choices=["mesh", "box", "env_only", "none"],
        default=DEFAULT_L2_OBSTACLE_MODE,
        help=(
            "reason_common_gids 使用的障碍列表。mesh/box=加入完整动态障碍；"
            "env_only=只检查桌子；none=不传障碍。碰撞几何精度由 --cdprim-type 决定。"
        ),
    )

    parser.add_argument("--enable-l3", action="store_true", default=DEFAULT_ENABLE_L3,
                        help="开启严格 L3 全流程 TransportPrimitive/RRT 动态避障验证。默认关闭，只保存 L2 结果。")
    parser.add_argument("--disable-l3", action="store_true",
                        help="关闭 L3，只保存 L2 结果。调试时才建议使用。")
    parser.add_argument("--l3-top-k", type=int, default=DEFAULT_L3_TOP_K,
                        help="对 L2 得分最高的前 K 个 layout 做 L3 验证。")
    parser.add_argument("--l3-obstacle-mode", choices=["mesh", "box", "env_only", "none"], default=DEFAULT_L3_OBSTACLE_MODE,
                        help="L3 全流程验证的障碍模式。box/mesh 都会加入动态障碍；碰撞几何精度由 --cdprim-type 决定。")
    parser.add_argument("--allow-l2-fallback", action="store_true",
                        help="如果 L3 top-k 全失败，允许回退保存 L2 最高分。不加则 L3 失败时不保存。")

    parser.add_argument("--w-grasp", type=float, default=DEFAULT_W_GRASP)
    parser.add_argument("--w-manip", type=float, default=DEFAULT_W_MANIP)
    parser.add_argument("--w-dist", type=float, default=DEFAULT_W_DIST)
    parser.add_argument("--w-rot", type=float, default=DEFAULT_W_ROT)

    return parser.parse_args()


def main():
    args = _parse_args()

    part_order = _parse_part_order(args.part_order)
    grasp_map = _load_json_map(args.grasp_map_json)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(SEALP_ROOT, "examples", "layout", "_output")

    print("=" * 70)
    print("Tower Initial Layout Search [v8.3 BoxCollision + WRS FlatSurface]")
    print(f"asmdef    = {args.asmdef}")
    print(f"config    = {args.config}")
    print(f"grasp_dir = {args.grasp_dir}")
    print(f"output    = {os.path.join(output_dir, args.output_name + '.layout')}")
    print(f"L2 obs    = {args.planner_obstacle_mode}, cdprim={args.cdprim_type}")
    print(f"L3 enable = {args.enable_l3 and not args.disable_l3}, L3 obs = {args.l3_obstacle_mode}, top_k = {args.l3_top_k}")
    print(f"assembly region search = {not args.disable_assembly_region_search}, grid={args.assembly_grid}, preassemble_first={not args.disable_preassemble_first}")
    print(f"flatsurface poses      = {not args.disable_flatsurface}, threshold={args.fs_stability_threshold}")
    print("=" * 70)

    searcher = WeightedInitialLayoutSearcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        fixture_pos=_parse_vec3(args.fixture_pos, (0.36, 0.0, 0.0)),
        fixture_rotmat=np.eye(3),
        robot_base_pos=_parse_vec3(args.robot_base_pos, (0.0, 0.0, 0.0)),
        robot_base_rotmat=np.eye(3),
        part_order=part_order,
        output_name=args.output_name,
        table_name=args.table_name,
        table_margin=args.table_margin,
        table_clearance=args.table_clearance,
        grasp_map=grasp_map,
        max_rot_candidates=args.max_rot_candidates,
        w_grasp=args.w_grasp,
        w_manip=args.w_manip,
        w_dist=args.w_dist,
        w_rot=args.w_rot,
        ignore_env=args.ignore_env,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        plan_assembly_region=not args.disable_assembly_region_search,
        assembly_grid=args.assembly_grid,
        preassemble_first_part=not args.disable_preassemble_first,
        use_flatsurface=not args.disable_flatsurface,
        fs_stability_threshold=args.fs_stability_threshold,
    )

    best = searcher.random_search(
        n_samples=args.n_samples,
        seed=args.seed,
        enable_l3=(args.enable_l3 and not args.disable_l3),
        l3_top_k=args.l3_top_k,
        l3_obstacle_mode=args.l3_obstacle_mode,
        require_l3=not args.allow_l2_fallback,
    )

    if best is not None:
        searcher.save_layout(best, output_dir)


if __name__ == "__main__":
    main()
