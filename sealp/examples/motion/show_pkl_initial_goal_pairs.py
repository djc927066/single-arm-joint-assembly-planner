#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show the robot grasp configuration and placement configuration from a motion PKL.

与 ``execute_layout_sequence_visual.py`` 分离，专门用于论文式静态对比图：

- 对每个零件，同时渲染两组机械臂轨迹快照：
  1. ``T_init`` 主状态，以及其后的 3 个 PKL 关节角状态；
  2. ``T_goal`` 主状态，以及其之前的 3 个 PKL 关节角状态。
- 蓝色物体仍只显示在初始位姿，白色物体仍只显示在目标位姿。
- 机械臂关节角和夹爪开度直接读取 motion-cache PKL。
- 蓝色物体表示初始状态，白色物体表示目标状态。
- 每按一次 SPACE，切换到下一个零件。
- 桌面为 46 x 23 的真实贯穿孔洞洞板，并强制 alpha=1.0。
- 不执行规划，不修改原始 PKL，也不修改原执行脚本。

推荐保存位置
------------
    sealp/examples/motion/show_pkl_robot_init_goal_pairs.py

运行示例
--------
python -m sealp.examples.motion.show_pkl_robot_init_goal_pairs ^
  --motion-cache D:/Project/wrs-sealp/sealp/examples/motion/_output/tower_neural.pkl
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh as trm

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from wrs import mcm, mgm, wd

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sealp.assembly_sequence import AssemblyDef
from sealp.colliders import StaticEnvironment
from sealp.config import load_config
from sealp.layout import WorkspaceLayout
from sealp.layout._viz_common import load_table_box

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda


DEFAULT_CACHE = os.path.join(
    SEALP_ROOT, "examples", "motion", "_output", "tower_neural.pkl"
)
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")

DUAL_ARM_Y_OFFSET = 0.6
HOME_JV = np.zeros(6)

TABLE_LONG_HOLES = 46
TABLE_SHORT_HOLES = 23
TABLE_HOLE_SEGMENTS = 16
TABLE_HOLE_DIAMETER = None

# 参考图风格
INIT_OBJECT_RGBA = np.array([0.03, 0.42, 0.90, 1.0])
GOAL_OBJECT_RGBA = np.array([0.97, 0.97, 0.97, 1.0])

# 两个机械臂构型用深浅灰和透明度区分。
INIT_ARM_RGBA = np.array([0.22, 0.22, 0.25, 0.58])
GOAL_ARM_RGBA = np.array([0.68, 0.68, 0.70, 0.76])

# T_init 之后 3 帧：越往后越淡。
INIT_AFTER_RGBA = [
    np.array([0.28, 0.28, 0.31, 0.40]),
    np.array([0.34, 0.34, 0.37, 0.29]),
    np.array([0.40, 0.40, 0.43, 0.20]),
]

# T_goal 之前 3 帧：越接近 T_goal 越明显。
GOAL_BEFORE_RGBA = [
    np.array([0.78, 0.78, 0.80, 0.18]),
    np.array([0.74, 0.74, 0.76, 0.28]),
    np.array([0.70, 0.70, 0.72, 0.42]),
]

CONTEXT_RGBA = np.array([0.50, 0.50, 0.50, 1.0])


@dataclass
class FrameRecord:
    global_index: int
    segment_index: int
    local_index: int
    arm_side: str
    jv: np.ndarray
    ee: object
    ee_scalar: Optional[float]
    obj_pose: Optional[Tuple[np.ndarray, np.ndarray]]


@dataclass
class RobotPosePair:
    step_id: int
    part_id: str
    arm_tag: str
    motion_tag: str
    init_frame: FrameRecord
    goal_frame: FrameRecord
    init_obj_pose: Tuple[np.ndarray, np.ndarray]
    goal_obj_pose: Tuple[np.ndarray, np.ndarray]
    # 当前 step 的完整 PKL 帧序列，用于显示 T_init 后 3 帧和 T_goal 前 3 帧。
    all_frames: List[FrameRecord]


def _deserialize_ee(value):
    if value is None:
        return None
    try:
        if np.isscalar(value):
            return float(value)
    except Exception:
        pass
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return None


def _ee_to_scalar(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if np.isscalar(value):
            return float(value)
    except Exception:
        pass
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size:
            return float(array[0])
    except Exception:
        pass
    return None


def _decode_pose(entry) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if entry is None:
        return None
    try:
        pos, rotmat = entry
        return (
            np.asarray(pos, dtype=float).reshape(3),
            np.asarray(rotmat, dtype=float).reshape(3, 3),
        )
    except Exception:
        return None


def _flatten_step_frames(step: dict) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    global_index = 0

    for segment_index, segment in enumerate(step.get("segments") or []):
        arm_side = str(segment.get("arm_side", "lft"))
        jv_list = segment.get("jv_list") or []
        ee_list = segment.get("ev_list") or []
        obj_pose_list = segment.get("obj_pose_list") or []

        for local_index, jv in enumerate(jv_list):
            ee = (
                _deserialize_ee(ee_list[local_index])
                if local_index < len(ee_list)
                else None
            )
            obj_pose = (
                _decode_pose(obj_pose_list[local_index])
                if local_index < len(obj_pose_list)
                else None
            )

            frames.append(
                FrameRecord(
                    global_index=global_index,
                    segment_index=segment_index,
                    local_index=local_index,
                    arm_side=arm_side,
                    jv=np.asarray(jv, dtype=float),
                    ee=ee,
                    ee_scalar=_ee_to_scalar(ee),
                    obj_pose=obj_pose,
                )
            )
            global_index += 1

    return frames


def _rotation_error(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    relative = np.asarray(rot_a, dtype=float).T @ np.asarray(rot_b, dtype=float)
    value = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(value))


def _same_pose(
    pose_a: Optional[Tuple[np.ndarray, np.ndarray]],
    pose_b: Tuple[np.ndarray, np.ndarray],
    pos_tol: float = 2e-4,
    rot_tol: float = np.deg2rad(0.75),
) -> bool:
    if pose_a is None:
        return False
    pos_a, rot_a = pose_a
    pos_b, rot_b = pose_b
    return (
        float(np.linalg.norm(pos_a - pos_b)) <= pos_tol
        and _rotation_error(rot_a, rot_b) <= rot_tol
    )


def _select_closed_frame(
    candidates: List[FrameRecord],
    *,
    prefer_last: bool,
) -> FrameRecord:
    """从同一物体位姿的帧里选择夹爪闭合的机械臂状态。"""
    if not candidates:
        raise ValueError("candidates is empty")

    ee_values = [
        frame.ee_scalar
        for frame in candidates
        if frame.ee_scalar is not None
    ]

    if ee_values:
        ee_min = min(ee_values)
        ee_max = max(ee_values)

        # 夹爪开合差明显时，取接近最小开度的帧，即已经夹住物体的状态。
        if ee_max - ee_min > 1e-5:
            tolerance = max(1e-5, 0.08 * (ee_max - ee_min))
            closed = [
                frame
                for frame in candidates
                if (
                    frame.ee_scalar is not None
                    and frame.ee_scalar <= ee_min + tolerance
                )
            ]
            if closed:
                return closed[-1] if prefer_last else closed[0]

    # PKL 中夹爪开度恒定时：
    # T_init 取初始位姿停留区间最后一帧；
    # T_goal 取首次到达目标位姿的帧，避免误选后续 HOME 段。
    return candidates[-1] if prefer_last else candidates[0]


def _extract_robot_pair(step: dict) -> Optional[RobotPosePair]:
    frames = _flatten_step_frames(step)
    valid_obj_frames = [frame for frame in frames if frame.obj_pose is not None]
    if not valid_obj_frames:
        return None

    init_obj_pose = valid_obj_frames[0].obj_pose
    goal_obj_pose = valid_obj_frames[-1].obj_pose
    assert init_obj_pose is not None
    assert goal_obj_pose is not None

    init_candidates = [
        frame for frame in frames if _same_pose(frame.obj_pose, init_obj_pose)
    ]
    goal_candidates = [
        frame for frame in frames if _same_pose(frame.obj_pose, goal_obj_pose)
    ]

    if not init_candidates or not goal_candidates:
        return None

    # 初始状态：物体仍在 staging，夹爪已闭合，优先取离开前最后一帧。
    init_frame = _select_closed_frame(
        init_candidates,
        prefer_last=True,
    )

    # 目标状态：物体已到装配区，夹爪尚未释放。
    # 当夹爪值无法区分时，取首次到达目标位姿，避免选到之后的 HOME 段。
    goal_frame = _select_closed_frame(
        goal_candidates,
        prefer_last=False,
    )

    # 若目标候选中确实存在开合差，改取“最后一个仍闭合”的目标帧，
    # 可以获得更稳定的放置姿态，同时仍不会选中释放后的 HOME。
    goal_ee = [
        frame.ee_scalar
        for frame in goal_candidates
        if frame.ee_scalar is not None
    ]
    if goal_ee and max(goal_ee) - min(goal_ee) > 1e-5:
        goal_frame = _select_closed_frame(
            goal_candidates,
            prefer_last=True,
        )

    return RobotPosePair(
        step_id=int(step.get("step_id", -1)),
        part_id=str(step.get("part_id", "")),
        arm_tag=str(step.get("arm_tag", "")),
        motion_tag=str(step.get("motion_tag", "")),
        init_frame=init_frame,
        goal_frame=goal_frame,
        init_obj_pose=init_obj_pose,
        goal_obj_pose=goal_obj_pose,
        all_frames=frames,
    )


def load_robot_pairs(cache_path: str) -> Tuple[dict, List[RobotPosePair]]:
    with open(cache_path, "rb") as file:
        payload = pickle.load(file)

    if not isinstance(payload, dict):
        raise TypeError(
            f"PKL 根对象必须是 dict，实际为 {type(payload).__name__}"
        )

    pairs: List[RobotPosePair] = []
    for step in payload.get("steps") or []:
        pair = _extract_robot_pair(step)
        if pair is None:
            print(
                f"[WARN] step={step.get('step_id')} "
                f"part={step.get('part_id')} 无法提取机械臂初始/目标构型，跳过。"
            )
            continue
        pairs.append(pair)

    if not pairs:
        raise RuntimeError(
            "PKL 中没有可用的 jv_list、ev_list 和 obj_pose_list。"
        )

    return payload, pairs


def _resolve_project_file(
    explicit_path: str,
    payload_path: str,
    description: str,
) -> str:
    candidates: List[str] = []

    for raw in (explicit_path, payload_path):
        if not raw:
            continue

        candidates.append(str(raw))
        candidates.append(
            str(raw).replace("\\", os.sep).replace("/", os.sep)
        )

        normalized = str(raw).replace("\\", "/")
        marker = "/sealp/"
        marker_index = normalized.lower().find(marker)
        if marker_index >= 0:
            relative = normalized[marker_index + 1:]
            candidates.append(
                os.path.join(PROJECT_ROOT, *relative.split("/"))
            )

    seen = set()
    for candidate in candidates:
        absolute = os.path.abspath(candidate)
        if absolute in seen:
            continue
        seen.add(absolute)
        if os.path.isfile(absolute):
            return absolute

    raise FileNotFoundError(
        f"找不到 {description}。\n"
        f"CLI={explicit_path or '<empty>'}\n"
        f"PKL={payload_path or '<empty>'}"
    )


def make_collision_model(mesh_path: str, cdprim_type: str = "box"):
    requested = str(cdprim_type or "box").lower()
    if requested in ("bbox", "bounding_box", "bounding-box"):
        requested = "box"

    trials = []
    if requested in ("box", "aabb", "obb"):
        for primitive in ("box", "aabb", "obb"):
            trials.append(
                {"initor": mesh_path, "cdprim_type": primitive}
            )
        trials.append({"initor": mesh_path})
    else:
        trials.append(
            {"initor": mesh_path, "cdprim_type": requested}
        )
        trials.append({"initor": mesh_path})

    last_error = None
    for kwargs in trials:
        try:
            return mcm.CollisionModel(**kwargs)
        except TypeError:
            try:
                if "cdprim_type" in kwargs:
                    return mcm.CollisionModel(
                        mesh_path,
                        cdprim_type=kwargs["cdprim_type"],
                    )
                return mcm.CollisionModel(mesh_path)
            except Exception as error:
                last_error = error
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"无法创建 CollisionModel: {mesh_path}, error={last_error!r}"
    )


def _set_collection_rgba(model, rgba: Sequence[float]) -> None:
    """尽量给 ModelCollection 及其子模型统一着色。"""
    rgba_array = np.asarray(rgba, dtype=float)

    try:
        model.rgba = rgba_array
    except Exception:
        pass

    for attribute in ("gm_list", "cm_list"):
        try:
            children = getattr(model, attribute, None) or []
        except Exception:
            children = []

        for child in children:
            try:
                child.rgba = rgba_array
            except Exception:
                try:
                    child.rgb = rgba_array[:3]
                    child.alpha = float(rgba_array[3])
                except Exception:
                    pass


def _goto_arm(arm, jv: np.ndarray, ee=None) -> None:
    try:
        if ee is not None:
            arm.goto_given_conf(
                np.asarray(jv, dtype=float),
                ee_values=ee,
            )
        else:
            arm.goto_given_conf(np.asarray(jv, dtype=float))
    except TypeError:
        arm.goto_given_conf(np.asarray(jv, dtype=float))


# ---------------------------------------------------------------------------
# 洞洞板桌面
# ---------------------------------------------------------------------------

def _square_ring_points(
    half_x: float,
    half_y: float,
    radius: float,
    segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    segments = max(8, int(math.ceil(segments / 4.0)) * 4)
    per_edge = segments // 4
    outer = []

    for index in range(per_edge):
        t = index / per_edge
        outer.append((-half_x + 2.0 * half_x * t, -half_y))
    for index in range(per_edge):
        t = index / per_edge
        outer.append((half_x, -half_y + 2.0 * half_y * t))
    for index in range(per_edge):
        t = index / per_edge
        outer.append((half_x - 2.0 * half_x * t, half_y))
    for index in range(per_edge):
        t = index / per_edge
        outer.append((-half_x, half_y - 2.0 * half_y * t))

    outer_array = np.asarray(outer, dtype=float)
    norms = np.linalg.norm(outer_array, axis=1, keepdims=True)
    inner_array = radius * outer_array / np.maximum(norms, 1e-12)
    return outer_array, inner_array


def _append_perforated_cell(
    vertices,
    faces,
    cx,
    cy,
    z_bottom,
    z_top,
    pitch_x,
    pitch_y,
    radius,
    segments,
    close_bottom,
    close_right,
    close_top,
    close_left,
):
    outer_xy, inner_xy = _square_ring_points(
        pitch_x / 2.0,
        pitch_y / 2.0,
        radius,
        segments,
    )
    count = len(outer_xy)
    start = len(vertices)

    for xy in outer_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_top])
    for xy in inner_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_top])
    for xy in outer_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_bottom])
    for xy in inner_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_bottom])

    outer_top = start
    inner_top = start + count
    outer_bottom = start + 2 * count
    inner_bottom = start + 3 * count

    for index in range(count):
        next_index = (index + 1) % count

        faces.append(
            [outer_top + index, outer_top + next_index, inner_top + next_index]
        )
        faces.append(
            [outer_top + index, inner_top + next_index, inner_top + index]
        )

        faces.append(
            [
                outer_bottom + index,
                inner_bottom + next_index,
                outer_bottom + next_index,
            ]
        )
        faces.append(
            [
                outer_bottom + index,
                inner_bottom + index,
                inner_bottom + next_index,
            ]
        )

        faces.append(
            [
                inner_top + index,
                inner_bottom + next_index,
                inner_bottom + index,
            ]
        )
        faces.append(
            [
                inner_top + index,
                inner_top + next_index,
                inner_bottom + next_index,
            ]
        )

    per_edge = count // 4
    edge_ranges = []
    if close_bottom:
        edge_ranges.append(range(0, per_edge))
    if close_right:
        edge_ranges.append(range(per_edge, 2 * per_edge))
    if close_top:
        edge_ranges.append(range(2 * per_edge, 3 * per_edge))
    if close_left:
        edge_ranges.append(range(3 * per_edge, 4 * per_edge))

    for edge_range in edge_ranges:
        for index in edge_range:
            next_index = (index + 1) % count
            faces.append(
                [
                    outer_top + index,
                    outer_bottom + index,
                    outer_bottom + next_index,
                ]
            )
            faces.append(
                [
                    outer_top + index,
                    outer_bottom + next_index,
                    outer_top + next_index,
                ]
            )


def _build_perforated_table_mesh(
    extent: Sequence[float],
    pos: Sequence[float],
):
    extent_array = np.asarray(extent, dtype=float).reshape(3)
    pos_array = np.asarray(pos, dtype=float).reshape(3)

    size_x, size_y, thickness = map(float, extent_array)

    if size_x >= size_y:
        nx, ny = TABLE_LONG_HOLES, TABLE_SHORT_HOLES
    else:
        nx, ny = TABLE_SHORT_HOLES, TABLE_LONG_HOLES

    pitch_x = size_x / nx
    pitch_y = size_y / ny
    minimum_pitch = min(pitch_x, pitch_y)

    hole_diameter = (
        0.42 * minimum_pitch
        if TABLE_HOLE_DIAMETER is None
        else float(TABLE_HOLE_DIAMETER)
    )
    radius = hole_diameter / 2.0

    x_min = pos_array[0] - size_x / 2.0
    y_min = pos_array[1] - size_y / 2.0
    z_bottom = pos_array[2] - thickness / 2.0
    z_top = pos_array[2] + thickness / 2.0

    vertices = []
    faces = []

    for ix in range(nx):
        cx = x_min + (ix + 0.5) * pitch_x
        for iy in range(ny):
            cy = y_min + (iy + 0.5) * pitch_y
            _append_perforated_cell(
                vertices,
                faces,
                cx,
                cy,
                z_bottom,
                z_top,
                pitch_x,
                pitch_y,
                radius,
                TABLE_HOLE_SEGMENTS,
                close_bottom=(iy == 0),
                close_right=(ix == nx - 1),
                close_top=(iy == ny - 1),
                close_left=(ix == 0),
            )

    mesh = trm.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        raise RuntimeError("生成的洞洞板不是 watertight 网格")

    return mesh, {
        "nx": nx,
        "ny": ny,
        "count": nx * ny,
        "diameter": hole_diameter,
    }


def attach_opaque_perforated_environment(base, config_path: str):
    cfg = load_config(config_path)
    environment = StaticEnvironment(
        obstacle_defs=cfg.obstacle_defs,
        base_dir=cfg.config_dir,
    )
    obstacles = list(environment.obstacle_list)

    table_extent, table_pos, table_rgba = load_table_box(
        config_path,
        "work_table",
    )

    table_index = 0 if len(obstacles) == 1 else None

    if table_index is None:
        target_pos = np.asarray(table_pos, dtype=float).reshape(3)
        best_distance = float("inf")
        for index, obstacle in enumerate(obstacles):
            try:
                obstacle_pos = np.asarray(
                    obstacle.pos,
                    dtype=float,
                ).reshape(3)
            except Exception:
                continue
            distance = float(np.linalg.norm(obstacle_pos - target_pos))
            if distance < best_distance:
                best_distance = distance
                table_index = index

    for index, obstacle in enumerate(obstacles):
        if index != table_index:
            obstacle.attach_to(base)

    mesh, info = _build_perforated_table_mesh(
        table_extent,
        table_pos,
    )

    cache_dir = Path(_THIS_DIR) / "_generated_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)

    extent_array = np.asarray(table_extent, dtype=float)
    mesh_path = cache_dir / (
        f"work_table_perforated_{info['nx']}x{info['ny']}_"
        f"{extent_array[0] * 1000.0:.1f}x"
        f"{extent_array[1] * 1000.0:.1f}x"
        f"{extent_array[2] * 1000.0:.1f}mm_"
        f"d{info['diameter'] * 1000.0:.3f}mm_"
        f"seg{TABLE_HOLE_SEGMENTS}.stl"
    )

    if not mesh_path.is_file():
        mesh.export(str(mesh_path))

    table = mcm.CollisionModel(str(mesh_path))

    table_rgba_array = np.asarray(table_rgba, dtype=float).reshape(-1)
    if table_rgba_array.size >= 3:
        table.rgba = np.array(
            [
                table_rgba_array[0],
                table_rgba_array[1],
                table_rgba_array[2],
                1.0,
            ]
        )
    else:
        table.rgba = np.array([0.62, 0.62, 0.62, 1.0])

    table.attach_to(base)

    print(
        f"[TABLE] {info['nx']}x{info['ny']}={info['count']} 个真实贯穿孔，"
        "alpha=1.0"
    )
    return table


def _is_preassembled(part_id: str, layout: WorkspaceLayout) -> bool:
    metadata = getattr(layout, "metadata", {}) or {}
    arm_choice = metadata.get("arm_choice", {}) or {}
    pose_tag = metadata.get("pose_tag", {}) or {}

    if arm_choice.get(part_id) == "preassembled":
        return True
    if str(pose_tag.get(part_id, "")).startswith("preassembled"):
        return True

    part_order = metadata.get("part_order", []) or []
    return bool(
        metadata.get("preassemble_first_part", False)
        and part_order
        and part_id == part_order[0]
    )


def attach_preassembled_context(
    base,
    asm: AssemblyDef,
    layout: WorkspaceLayout,
    world_poses: dict,
    pkl_part_ids: set,
    cdprim_type: str,
):
    models = []

    for part_id in asm.part_ids:
        if part_id in pkl_part_ids:
            continue
        if not _is_preassembled(part_id, layout):
            continue
        if part_id not in world_poses:
            continue

        mesh_path = asm.model_path(part_id)
        if not mesh_path or not os.path.isfile(mesh_path):
            continue

        pos, rotmat = world_poses[part_id]
        model = make_collision_model(mesh_path, cdprim_type)
        model.pos = np.asarray(pos, dtype=float)
        model.rotmat = np.asarray(rotmat, dtype=float)
        model.rgba = CONTEXT_RGBA.copy()
        model.attach_to(base)
        models.append(model)

    return models


class RobotInitialGoalViewer:

    def __init__(
        self,
        base,
        asm: AssemblyDef,
        layout: WorkspaceLayout,
        robot_pairs: List[RobotPosePair],
        cdprim_type: str = "box",
        show_frames: bool = True,
    ):
        self.base = base
        self.asm = asm
        self.layout = layout
        self.robot_pairs = list(robot_pairs)
        self.cdprim_type = cdprim_type
        self.show_frames = bool(show_frames)

        self.index = 0
        self.current_nodes = []

        robot_pos = np.asarray(
            getattr(layout, "robot_base_pos", np.zeros(3)),
            dtype=float,
        )
        robot_rotmat = np.asarray(
            getattr(layout, "robot_base_rotmat", np.eye(3)),
            dtype=float,
        )

        # 复用同一个双臂对象生成两份独立的 active-arm mesh。
        self.robot = pda.DualPantheraHTNoBody(
            pos=robot_pos,
            rotmat=robot_rotmat,
            arm_y_offset=DUAL_ARM_Y_OFFSET,
            enable_cc=False,
        )
        self.robot.lft_arm.goto_given_conf(HOME_JV)
        self.robot.rgt_arm.goto_given_conf(HOME_JV)

        self.status_text = OnscreenText(
            text="",
            pos=(-1.28, 0.91),
            scale=0.047,
            fg=(0.04, 0.04, 0.04, 1.0),
            align=TextNode.ALeft,
            mayChange=True,
        )
        self.legend_text = OnscreenText(
            text=(
                "T_init + next 3 joint frames    "
                "T_goal + previous 3 joint frames    SPACE: next"
            ),
            pos=(-1.28, 0.82),
            scale=0.036,
            fg=(0.06, 0.06, 0.06, 1.0),
            align=TextNode.ALeft,
        )

        self.base.accept("space", self.show_next)
        self.show_pair(0)

    def _clear_current(self):
        for node in self.current_nodes:
            try:
                node.detach()
            except Exception:
                try:
                    node.remove()
                except Exception:
                    pass
        self.current_nodes.clear()

    def _active_arm(self, side: str):
        return (
            self.robot.rgt_arm
            if str(side).lower() == "rgt"
            else self.robot.lft_arm
        )

    def _attach_arm_state(
        self,
        frame: FrameRecord,
        rgba: np.ndarray,
    ):
        arm = self._active_arm(frame.arm_side)
        _goto_arm(arm, frame.jv, frame.ee)

        # 只生成当前步骤真正使用的手臂。
        arm_mesh = arm.gen_meshmodel(alpha=float(rgba[3]))
        _set_collection_rgba(arm_mesh, rgba)
        arm_mesh.attach_to(self.base)
        self.current_nodes.append(arm_mesh)

    @staticmethod
    def _neighbor_frames(
        pair: RobotPosePair,
        center_frame: FrameRecord,
        offsets: Sequence[int],
    ) -> List[FrameRecord]:
        """按全局帧索引读取中心帧前后邻域，越界帧自动忽略。"""
        frame_map = {
            frame.global_index: frame
            for frame in pair.all_frames
        }
        neighbors: List[FrameRecord] = []
        for offset in offsets:
            frame = frame_map.get(center_frame.global_index + int(offset))
            if frame is not None:
                neighbors.append(frame)
        return neighbors

    def _attach_arm_sequence(
        self,
        frames: Sequence[FrameRecord],
        rgba_list: Sequence[np.ndarray],
    ) -> None:
        """按给定透明度顺序叠加显示若干 PKL 关节状态。"""
        for frame, rgba in zip(frames, rgba_list):
            self._attach_arm_state(frame, np.asarray(rgba, dtype=float))

    def _attach_object(
        self,
        part_id: str,
        pose: Tuple[np.ndarray, np.ndarray],
        rgba: np.ndarray,
    ):
        mesh_path = self.asm.model_path(part_id)
        if not mesh_path or not os.path.isfile(mesh_path):
            raise FileNotFoundError(
                f"找不到 {part_id} 的模型文件: {mesh_path}"
            )

        pos, rotmat = pose
        model = make_collision_model(mesh_path, self.cdprim_type)
        model.pos = np.asarray(pos, dtype=float)
        model.rotmat = np.asarray(rotmat, dtype=float)
        model.rgba = rgba.copy()
        model.attach_to(self.base)
        self.current_nodes.append(model)

        if self.show_frames:
            frame = mgm.gen_frame(
                pos=np.asarray(pos, dtype=float),
                rotmat=np.asarray(rotmat, dtype=float),
                ax_length=0.045,
            )
            frame.attach_to(self.base)
            self.current_nodes.append(frame)

    def show_pair(self, index: int):
        if not self.robot_pairs:
            return

        self.index = int(index) % len(self.robot_pairs)
        pair = self.robot_pairs[self.index]

        self._clear_current()

        # ---------------------------------------------------------------
        # T_init：主抓取构型 + 其后的 3 个 PKL 关节角。
        # 额外帧只显示机械臂，不重复显示物体。
        # ---------------------------------------------------------------
        self._attach_arm_state(
            pair.init_frame,
            INIT_ARM_RGBA,
        )
        init_after_frames = self._neighbor_frames(
            pair,
            pair.init_frame,
            offsets=(1, 2, 3),
        )
        self._attach_arm_sequence(
            init_after_frames,
            INIT_AFTER_RGBA[:len(init_after_frames)],
        )

        # ---------------------------------------------------------------
        # T_goal：其之前的 3 个 PKL 关节角 + 主放置构型。
        # 按 -3、-2、-1 顺序绘制，使越接近 T_goal 的帧越明显。
        # ---------------------------------------------------------------
        goal_before_frames = self._neighbor_frames(
            pair,
            pair.goal_frame,
            offsets=(-3, -2, -1),
        )
        goal_before_rgba = GOAL_BEFORE_RGBA[-len(goal_before_frames):]
        self._attach_arm_sequence(
            goal_before_frames,
            goal_before_rgba,
        )
        self._attach_arm_state(
            pair.goal_frame,
            GOAL_ARM_RGBA,
        )

        # 两个物体位姿仍然各显示一次。
        self._attach_object(
            pair.part_id,
            pair.init_obj_pose,
            INIT_OBJECT_RGBA,
        )
        self._attach_object(
            pair.part_id,
            pair.goal_obj_pose,
            GOAL_OBJECT_RGBA,
        )

        init_pos = pair.init_obj_pose[0]
        goal_pos = pair.goal_obj_pose[0]
        distance = float(np.linalg.norm(goal_pos - init_pos))

        self.status_text.setText(
            f"[{self.index + 1}/{len(self.robot_pairs)}] "
            f"step={pair.step_id}  part={pair.part_id}\n"
            f"T_init frame={pair.init_frame.global_index} "
            f"+ next {len(init_after_frames)}    "
            f"T_goal frame={pair.goal_frame.global_index} "
            f"+ previous {len(goal_before_frames)}    "
            f"distance={distance:.4f} m"
        )

        print("\n" + "=" * 78)
        print(
            f"[SHOW {self.index + 1}/{len(self.robot_pairs)}] "
            f"step={pair.step_id} part={pair.part_id}"
        )
        print(
            f"  T_init arm={pair.init_frame.arm_side} "
            f"global_frame={pair.init_frame.global_index} "
            f"segment={pair.init_frame.segment_index} "
            f"local={pair.init_frame.local_index} "
            f"ee={pair.init_frame.ee_scalar}"
        )
        print(
            f"  T_goal arm={pair.goal_frame.arm_side} "
            f"global_frame={pair.goal_frame.global_index} "
            f"segment={pair.goal_frame.segment_index} "
            f"local={pair.goal_frame.local_index} "
            f"ee={pair.goal_frame.ee_scalar}"
        )
        print(
            "  T_init next frames="
            f"{[frame.global_index for frame in init_after_frames]}"
        )
        print(
            "  T_goal previous frames="
            f"{[frame.global_index for frame in goal_before_frames]}"
        )
        print(
            f"  init_object={np.round(init_pos, 5).tolist()}"
        )
        print(
            f"  goal_object={np.round(goal_pos, 5).tolist()}"
        )
        print("=" * 78)

    def show_next(self):
        self.show_pair(self.index + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simultaneously display the PKL grasp configuration and "
            "placement configuration for each object."
        )
    )
    parser.add_argument(
        "--motion-cache",
        default=DEFAULT_CACHE,
        help="运动轨迹缓存 PKL。",
    )
    parser.add_argument(
        "--asmdef",
        default="",
        help="为空时读取 PKL 中的 asmdef_path。",
    )
    parser.add_argument(
        "--layout",
        default="",
        help="为空时读取 PKL 中的 layout_path。",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="sample_config.yaml。",
    )
    parser.add_argument(
        "--cdprim-type",
        default="box",
        help="零件显示模型类型，默认 box。",
    )
    parser.add_argument(
        "--cam-pos",
        default="1.05,-1.25,0.85",
        help="相机位置 x,y,z。",
    )
    parser.add_argument(
        "--hide-frames",
        action="store_true",
        help="隐藏 T_init 和 T_goal 坐标系。",
    )
    parser.add_argument(
        "--hide-context",
        action="store_true",
        help="隐藏 base_plate 等预装配上下文。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cache_path = os.path.abspath(args.motion_cache)
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"motion-cache 不存在: {cache_path}"
        )

    payload, robot_pairs = load_robot_pairs(cache_path)

    asmdef_path = _resolve_project_file(
        args.asmdef,
        str(payload.get("asmdef_path", "")),
        "asmdef",
    )
    layout_path = _resolve_project_file(
        args.layout,
        str(payload.get("layout_path", "")),
        "layout",
    )

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"config 不存在: {config_path}"
        )

    print("=" * 78)
    print("PKL Robot T_init / T_goal Multi-Frame Viewer")
    print(f"motion_cache = {cache_path}")
    print(f"asmdef       = {asmdef_path}")
    print(f"layout       = {layout_path}")
    print(f"config       = {config_path}")
    print(f"parts        = {[pair.part_id for pair in robot_pairs]}")
    print("=" * 78)

    asm = AssemblyDef.load(asmdef_path)
    layout = WorkspaceLayout.load(layout_path)

    world_poses = asm.compute_world_poses(
        fixture_pos=np.asarray(
            layout.assembly_station_pos,
            dtype=float,
        ),
        fixture_rotmat=np.asarray(
            layout.assembly_station_rotmat,
            dtype=float,
        ),
    )

    cam_pos = np.array(
        [
            float(value.strip())
            for value in args.cam_pos.split(",")
        ],
        dtype=float,
    )
    lookat_pos = np.asarray(
        layout.assembly_station_pos,
        dtype=float,
    ) + np.array([0.0, 0.0, 0.08])

    base = wd.World(
        cam_pos=cam_pos,
        lookat_pos=lookat_pos,
    )
    sys.modules["__main__"].base = base

    attach_opaque_perforated_environment(
        base,
        config_path,
    )

    if not args.hide_context:
        attach_preassembled_context(
            base=base,
            asm=asm,
            layout=layout,
            world_poses=world_poses,
            pkl_part_ids={
                pair.part_id for pair in robot_pairs
            },
            cdprim_type=args.cdprim_type,
        )

    RobotInitialGoalViewer(
        base=base,
        asm=asm,
        layout=layout,
        robot_pairs=robot_pairs,
        cdprim_type=args.cdprim_type,
        show_frames=not args.hide_frames,
    )

    print("\n[CONTROL] 当前零件的 T_init 与 T_goal 机械臂构型已同时显示。")
    print("[CONTROL] T_init 显示主抓取帧及其后的 3 个 PKL 关节角。")
    print("[CONTROL] T_goal 显示主放置帧及其之前的 3 个 PKL 关节角。")
    print("[CONTROL] 每按一次 SPACE，切换到下一个零件并循环。")

    base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
