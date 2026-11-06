"""
Dual-Arm Initial Staging Search
================================

DFS 联合搜索一组「装配前零件初始摆放」(staging)，使得**每件零件**在
``staging → goal`` 的 pick-and-place 路径上至少存在一只臂可执行的
``reason_common`` 共同抓取，并满足以下硬约束：

1. 件间无碰撞（任意两件 staging 不重叠）
2. 不侵入「直接装在 fixture 上」的零件（如椅面）的装配 goal 体积
3. 至少存在一只臂（左/右）可同时在 staging + goal 处共存抓取（``reason_common_gids`` ≥ 1）

成功 → 返回 `WorkspaceLayout`，可直接 `.save("xxx.layout")`，
随后被 `eval_dual_layout.py` / `dual_sequence_execution.py` 加载并跳过自身搜索。

来源：从 ``sealp/examples/motion/dual_sequence_execution.py::prevalidate_initial_staging_layout``
解耦而来；改成纯库函数，不再依赖 demo 脚本里的全局常量。

Usage::

    from sealp.layout.dual_staging_search import search_dual_feasible_layout

    layout = search_dual_feasible_layout(
        assembly_def=asm,
        robot_dual=robot,
        grasp_cache={"seat_model": ..., "leg_model": ...},
        fixture_pos=np.array([0.0, -0.30, 0.0]),
        staging_seeds={"seat": np.array([...]), ...},
        candidate_zones={"seat": {"x": [...], "y": [...]}, ...},
        env_obstacles=[...],
    )
    layout.save("dual_yuanchair_searched.layout")
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import wrs.modeling.collision_model as mcm
from wrs.manipulation.pick_place import PickPlacePlanner

from sealp.assembly_sequence import AssemblyDef
from .layout import WorkspaceLayout


# ══════════════════════════════════════════════════════════════
#  候选生成
# ══════════════════════════════════════════════════════════════
def generate_staging_candidates(
    seed_pos: np.ndarray,
    zone: Optional[Dict] = None,
    cross_arm_zone: Optional[Dict] = None,
    max_candidates: Optional[int] = None,
) -> List[np.ndarray]:
    """生成单个零件的 staging 候选位置列表。

    candidate[0] 始终为 ``seed_pos``，其余按距种子 xy 距离升序。
    若指定 ``max_candidates``，则只保留前 N 个（含 seed），用于在高密度
    网格（如 5mm 分辨率）下保护 DFS 性能。

    Parameters
    ----------
    seed_pos : np.ndarray
        种子位置 ``[x, y, z]``。
    zone : dict or None
        ``{"x": [...], "y": [...]}`` 的网格定义；笛卡尔积构造候选。
        ``z`` 取 ``seed_pos[2]``（即同一桌面高度）。
    cross_arm_zone : dict or None
        额外的「跨臂区」网格（用于让搜索能跳到对侧臂工作空间）。
    max_candidates : int or None
        最大候选数（含 seed）。``None`` 或 ``≤0`` 表示不限制。
        典型值：200~500 配 5mm 分辨率，覆盖种子周围 8~12 cm 半径。
    """
    seed = np.asarray(seed_pos, dtype=float).copy()
    seed_z = float(seed[2]) if seed.shape[0] >= 3 else 0.0
    candidates = [seed]
    grid: List[np.ndarray] = []

    def _push_grid(zd):
        if not zd:
            return
        for x in zd.get("x", []):
            for y in zd.get("y", []):
                grid.append(np.array([float(x), float(y), seed_z]))

    _push_grid(zone)
    _push_grid(cross_arm_zone)

    seen = {tuple(np.round(candidates[0], 4))}
    extras: List[np.ndarray] = []
    for p in grid:
        t = tuple(np.round(p, 4))
        if t in seen:
            continue
        seen.add(t)
        extras.append(p)
    extras.sort(key=lambda q: float(np.linalg.norm(q[:2] - seed[:2])))
    if max_candidates is not None and max_candidates > 0:
        # candidates 已含 seed，所以从 extras 里只取 max_candidates - 1 个
        extras = extras[:max(0, max_candidates - 1)]
    candidates.extend(extras)
    return candidates


def make_grid_zone_from_box(
    box_pos,
    box_extent,
    resolution: float = 0.05,
    margin: float = 0.05,
    return_meta: bool = False,
):
    """从一个 box 障碍物（如 ``work_table``）生成桌面候选 zone。

    **语义约定**（与 ``wrs.modeling.collision_model.gen_box(xyz_lengths=..., pos=...)``
    一致）::

        box_pos    = [cx, cy, cz]   # box 中心相对世界基座标的位置
        box_extent = [Lx, Ly, Lz]   # box 三轴 *全长*（不是半长）

    在 XY 平面上推得：

    * 物理边界：``X ∈ [cx - Lx/2, cx + Lx/2]``,
                 ``Y ∈ [cy - Ly/2, cy + Ly/2]``
    * 留白后候选：``X ∈ [cx - Lx/2 + margin, cx + Lx/2 - margin]``
                    ``Y ∈ [cy - Ly/2 + margin, cy + Ly/2 - margin]``
    * 网格步长由 ``resolution`` 控制（米）。

    Parameters
    ----------
    box_pos : array-like, shape (3,)
        Box 中心位置 ``[cx, cy, cz]``。
    box_extent : array-like, shape (3,)
        Box 全长 ``[Lx, Ly, Lz]``。
    resolution : float
        网格步长 (m)。默认 5 cm。
    margin : float
        离桌沿的留白 (m)。默认 5 cm。
    return_meta : bool
        若为 True，额外返回一份元数据 dict（含物理/留白边界与候选数）。

    Returns
    -------
    zone : dict
        ``{"x": [...], "y": [...]}``，可直接作为
        ``search_dual_feasible_layout(..., candidate_zones=...)`` 的 zone。
    meta : dict, 可选
        仅当 ``return_meta=True`` 时返回，含完整范围明细。

    Examples
    --------
    >>> # sample_config.yaml 中 work_table 的定义：
    >>> #   pos    = [0.345, -0.35, -0.01]   ← 中心
    >>> #   extent = [0.80,   1.20,  0.02]   ← 全长
    >>> zone, meta = make_grid_zone_from_box(
    ...     box_pos=[0.345, -0.35, -0.01],
    ...     box_extent=[0.8, 1.2, 0.02],
    ...     resolution=0.05, margin=0.05, return_meta=True)
    >>> meta["physical_bounds_x"], meta["physical_bounds_y"]
    ((-0.055, 0.745), (-0.95, 0.25))
    >>> meta["sample_bounds_x"], meta["sample_bounds_y"]
    ((-0.005, 0.695), (-0.9, 0.2))
    >>> meta["n_x"], meta["n_y"], meta["n_total"]
    (15, 23, 345)
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    cx, cy = float(box_pos[0]), float(box_pos[1])
    lx, ly = float(box_extent[0]), float(box_extent[1])

    # box 物理边界（XY）
    phys_x_min, phys_x_max = cx - lx / 2.0, cx + lx / 2.0
    phys_y_min, phys_y_max = cy - ly / 2.0, cy + ly / 2.0

    # 留白后允许采样的边界
    samp_x_min = phys_x_min + margin
    samp_x_max = phys_x_max - margin
    samp_y_min = phys_y_min + margin
    samp_y_max = phys_y_max - margin

    if samp_x_max < samp_x_min or samp_y_max < samp_y_min:
        raise ValueError(
            f"margin={margin} 过大，超过 box 半长 "
            f"(Lx/2={lx / 2:.3f}, Ly/2={ly / 2:.3f})；候选范围为空。")

    xs = np.arange(samp_x_min, samp_x_max + 1e-9, resolution).tolist()
    ys = np.arange(samp_y_min, samp_y_max + 1e-9, resolution).tolist()
    zone = {"x": xs, "y": ys}

    if not return_meta:
        return zone

    meta = {
        "box_center_xy": (cx, cy),
        "box_extent_xy": (lx, ly),
        "physical_bounds_x": (phys_x_min, phys_x_max),
        "physical_bounds_y": (phys_y_min, phys_y_max),
        "sample_bounds_x": (samp_x_min, samp_x_max),
        "sample_bounds_y": (samp_y_min, samp_y_max),
        "resolution": resolution,
        "margin": margin,
        "n_x": len(xs),
        "n_y": len(ys),
        "n_total": len(xs) * len(ys),
    }
    return zone, meta


def find_obstacle_def(obstacle_defs: List[dict],
                      name: str) -> Optional[dict]:
    """在 ``cfg.obstacle_defs`` 里按 ``name`` 取一条障碍定义。"""
    for d in obstacle_defs:
        if d.get("name") == name:
            return d
    return None


# ══════════════════════════════════════════════════════════════
#  内部 helper（碰撞 / 抓取检查）
# ══════════════════════════════════════════════════════════════
def _build_staging_obstacles(part_ids, seeds, asm) -> Dict[str, mcm.CollisionModel]:
    out: Dict[str, mcm.CollisionModel] = {}
    for pid in part_ids:
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue
        cm = mcm.CollisionModel(initor=mp)
        cm.pos = np.asarray(seeds.get(pid, np.zeros(3)), dtype=float).copy()
        cm.rotmat = np.eye(3)
        cm._sealp_role = "search_probe"
        cm._sealp_part_id = pid
        out[pid] = cm
    return out


def _build_goal_obstacle_models(asm, world_poses) -> Dict[str, mcm.CollisionModel]:
    out: Dict[str, mcm.CollisionModel] = {}
    for pid, (gp, gr) in world_poses.items():
        if pid not in asm.part_ids:
            continue
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue
        cm = mcm.CollisionModel(initor=mp)
        cm.pos = gp
        cm.rotmat = gr
        cm._sealp_role = "goal_placeholder"
        cm._sealp_part_id = pid
        out[pid] = cm
    return out


def _step_parent_id(asm, part_id: str) -> Optional[str]:
    for s in asm.steps:
        if s.part_id == part_id:
            return s.parent_id
    return None


def _is_clear_of_selected(probe, pos, rotmat, staging_obstacles, selected_ids) -> bool:
    if probe is None:
        return True
    old_p, old_r = probe.pos.copy(), probe.rotmat.copy()
    try:
        probe.pos = pos
        probe.rotmat = rotmat
        for oid in selected_ids:
            other = staging_obstacles.get(oid)
            if other is None:
                continue
            if probe.is_mcdwith(other):
                return False
        return True
    finally:
        probe.pos = old_p
        probe.rotmat = old_r


def _is_clear_of_foreign_goals(
    probe, pos, rotmat, goal_models, asm, part_id,
) -> bool:
    """staging 不得侵入「直接装在 fixture 上」的零件 goal 体积。"""
    if probe is None or not goal_models:
        return True
    old_p, old_r = probe.pos.copy(), probe.rotmat.copy()
    try:
        probe.pos = pos
        probe.rotmat = rotmat
        for oid, gcm in goal_models.items():
            if oid == part_id or gcm is None:
                continue
            if _step_parent_id(asm, oid) != "fixture":
                continue
            if probe.is_mcdwith(gcm):
                return False
        return True
    finally:
        probe.pos = old_p
        probe.rotmat = old_r


def _segment_reachable(arm, grasp, base_pos, base_rot, direction, distance,
                       n_samples: int = 4) -> bool:
    """从 (base_pos, base_rot) 抓取位姿沿 ``direction`` 直线段走 ``distance``，
    等分 ``n_samples`` 中间采样点 IK 是否一直可解（前一点作下一次的 seed）。

    用于补 ``reason_common_gids`` 缺失的"中间直线段可达性"检查 ——
    例如执行阶段 pick_depart 沿 +Z 抬升、place_approach 沿 −Z 下放。
    搜索阶段调用此函数，可提前过滤"两端 IK OK 但中间无解"的位置，避免
    生成出来的 .layout 在 ``dual_sequence_execution.py`` 里跑出 IK 错误。

    Returns
    -------
    bool : 全部采样点 IK 成功 → True；任意一点失败 → False。
    """
    if distance is None or distance <= 1e-6 or n_samples < 1:
        return True
    base_pos = np.asarray(base_pos, dtype=float)
    base_rot = np.asarray(base_rot, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        return True
    dir_unit = direction / n
    tcp_pos = base_rot.dot(grasp.ac_pos) + base_pos
    tcp_rot = base_rot.dot(grasp.ac_rotmat)
    seed = arm.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rot)
    if seed is None:
        return False
    for i in range(1, n_samples + 1):
        d = distance * i / n_samples
        end_pos = tcp_pos + dir_unit * d
        jv = arm.ik(tgt_pos=end_pos, tgt_rotmat=tcp_rot, seed_jnt_values=seed)
        if jv is None:
            return False
        seed = jv
    return True


def _arms_collide_at_home(robot_dual, obstacle_list) -> bool:
    """检查 ``robot_dual`` 双臂在当前(假定为 home)姿态下是否与障碍穿模。

    桌腿"躺着"放置时是一个长 (>0.3 m) 的水平物体，搜索阶段若仅靠
    ``reason_common_gids`` 在 pick / place pose 上做 ``is_collided``，
    会漏判这种 **初始几何重叠**：staging 姿态下的桌腿和臂体早就重合了，
    但 reason 时机器人已经被 ``goto_given_conf`` 移到抓取位姿。
    把"home 穿模"作为搜索硬约束，能保证生成的 ``.layout`` 一加载就
    可执行，不需要 ``dual_sequence_execution.py`` 再丢弃缓存重搜。

    要求调用前 ``robot_dual.lft_arm`` / ``rgt_arm`` 已位于 home 姿态
    （搜索器内部确保），因此这里 *不* 做 backup/restore，节省 DFS 开销。
    """
    if not obstacle_list:
        return False
    obs = list(obstacle_list)
    for arm in (getattr(robot_dual, "lft_arm", None),
                getattr(robot_dual, "rgt_arm", None)):
        if arm is None:
            continue
        hit = arm.is_collided(obstacle_list=obs)
        collided = hit[0] if isinstance(hit, tuple) else hit
        if collided:
            return True
    return False


def _common_grasps_ok(arm, grasp_collection, sp, sr, gp, gr, obstacle_list,
                      pick_depart_dir=None, pick_depart_dist=None,
                      place_approach_dir=None, place_approach_dist=None):
    """staging→goal 双端 IK + 共同抓取检查；可选追加中间直线段可达性过滤。

    新增的 ``pick_depart_*`` / ``place_approach_*`` 与
    ``dual_sequence_execution.py`` 中 ``RELAXED_PLANNING`` 的几何流对齐，
    用于在搜索阶段就过滤掉那些"两端 IK 通过但中间直线段 IK 不可解"的位置。
    """
    if grasp_collection is None or len(grasp_collection) == 0:
        return False, 0
    planner = PickPlacePlanner(robot=arm)
    gids = planner.reason_common_gids(
        grasp_collection=grasp_collection,
        goal_pose_list=[(sp, sr), (gp, gr)],
        obstacle_list=obstacle_list,
    )
    if not gids:
        return False, 0
    # pick_depart: 从 staging 抓取沿 +Z 抬升（与执行阶段对齐）
    if pick_depart_dir is not None and pick_depart_dist:
        gids = [
            gid for gid in gids
            if _segment_reachable(
                arm, grasp_collection[gid], sp, sr,
                pick_depart_dir, pick_depart_dist)
        ]
        if not gids:
            return False, 0
    # place_approach: 从 goal 沿 −approach_dir（即向上）走 dist，
    # 等价检查"接近段起点 → goal"中间各点 IK 全可解
    if place_approach_dir is not None and place_approach_dist:
        rev_dir = -np.asarray(place_approach_dir, dtype=float)
        gids = [
            gid for gid in gids
            if _segment_reachable(
                arm, grasp_collection[gid], gp, gr,
                rev_dir, place_approach_dist)
        ]
        if not gids:
            return False, 0
    return True, len(gids)


# ══════════════════════════════════════════════════════════════
#  主搜索函数
# ══════════════════════════════════════════════════════════════
def search_dual_feasible_layout(
    *,
    assembly_def: AssemblyDef,
    robot_dual,
    grasp_cache: Dict,
    fixture_pos: np.ndarray,
    staging_seeds: Dict[str, np.ndarray],
    fixture_rotmat: Optional[np.ndarray] = None,
    candidate_zones: Optional[Dict[str, Dict]] = None,
    cross_arm_zones: Optional[Dict[str, Dict]] = None,
    part_ids: Optional[Tuple[str, ...]] = None,
    env_obstacles: Optional[List] = None,
    model_alias_fn: Optional[Callable[[str], str]] = None,
    robot_base_pos: Optional[np.ndarray] = None,
    robot_base_rotmat: Optional[np.ndarray] = None,
    layout_name: str = "dual_searched_layout",
    arm_priority: Tuple[str, str] = ("lft", "rgt"),
    pick_depart_dir: Optional[np.ndarray] = None,
    pick_depart_dist: Optional[float] = None,
    place_approach_dir: Optional[np.ndarray] = None,
    place_approach_dist: Optional[float] = None,
    max_candidates_per_part: Optional[int] = None,
    verbose: bool = True,
) -> WorkspaceLayout:
    """搜索双臂可装配的初始 staging 布局。

    成功 → 返回 ``WorkspaceLayout``。失败 → 抛 ``RuntimeError``。

    Parameters
    ----------
    assembly_def : AssemblyDef
        装配定义。
    robot_dual
        双臂机器人，需要有 ``lft_arm`` 与 ``rgt_arm``，每只臂可被
        ``PickPlacePlanner`` 接受为 ``robot`` 参数。
    grasp_cache : dict
        ``{model_alias: GraspCollection}``，alias 由 ``model_alias_fn``
        给出。默认 alias = ``assembly_def.get_part(pid).model``。
    fixture_pos / fixture_rotmat
        装配站世界位姿。
    staging_seeds : dict
        ``{part_id: seed_pos}``，必须覆盖所有待搜索零件。
    candidate_zones / cross_arm_zones : dict[str, dict] or None
        每个 part_id 的候选网格定义 ``{"x": [...], "y": [...]}``。
        若全空，则只用 seed 作为唯一候选。
    part_ids : tuple of str or None
        待搜索的零件顺序；默认取 ``staging_seeds.keys()``。
        DFS 顺序按此 tuple 进行。
    env_obstacles : list or None
        静态障碍（地面、桌面、墙等），参与 ``reason_common_gids`` 检查。
    model_alias_fn : callable or None
        ``part_id -> model_alias`` 映射；默认从 asm 取。
    robot_base_pos / robot_base_rotmat : np.ndarray or None
        写入到返回 layout 的机器人基座（仅做记录，不重定位 robot_dual）。
    layout_name : str
        返回 layout 的 name 字段。
    arm_priority : tuple of str
        ``("lft", "rgt")`` 表示左臂优先、右臂兜底；与 dual demo 一致。
    pick_depart_dir / pick_depart_dist : np.ndarray, float, 可选
        若提供，则在 reason 通过后追加检查"从 staging 抓取沿 ``pick_depart_dir``
        直线段 ``pick_depart_dist`` 米"中间各采样点 IK 是否可解；不可解的
        位置直接被过滤。与 ``dual_sequence_execution.py`` 中 RELAXED_PLANNING
        的 +Z 抬升对齐。
    place_approach_dir / place_approach_dist : np.ndarray, float, 可选
        与上同理，对应"装配前直线接近段"。方向是从空中朝 goal 的下放方向
        （如 −Z），算法会自动反向沿该方向走 dist 检查中间各点 IK。
    max_candidates_per_part : int or None
        每个零件的最大候选数（含 seed），按距种子距离取最近的 N 个。
        高密度网格（如 5mm）必须设置，否则 DFS 会爆炸。典型值 200~500。
    verbose : bool
        打印 DFS 过程。

    Returns
    -------
    WorkspaceLayout
        ``staging_positions`` 已填好；``metadata`` 里附带 ``arm_choice``
        （每个 part_id 选定的臂）。
    """
    if fixture_rotmat is None:
        fixture_rotmat = np.eye(3)
    if env_obstacles is None:
        env_obstacles = []
    if part_ids is None:
        part_ids = tuple(staging_seeds.keys())
    if model_alias_fn is None:
        model_alias_fn = lambda pid: assembly_def.get_part(pid).model
    if robot_base_pos is None:
        robot_base_pos = np.zeros(3)
    if robot_base_rotmat is None:
        robot_base_rotmat = np.eye(3)

    arm_lookup = {
        "lft": getattr(robot_dual, "lft_arm", None),
        "rgt": getattr(robot_dual, "rgt_arm", None),
    }
    for tag in arm_priority:
        if arm_lookup.get(tag) is None:
            raise ValueError(
                f"robot_dual 缺少 {tag}_arm，无法按优先级 {arm_priority} 搜索。")

    # 强制把双臂送到 home，使后续 _arms_collide_at_home 调用可省略 backup。
    # _common_grasps_ok 内部用 keep_states 装饰的 reason_common_gids 不会污染。
    home_jv = np.zeros(arm_lookup[arm_priority[0]].n_dof)
    arm_lookup["lft"].goto_given_conf(home_jv)
    arm_lookup["rgt"].goto_given_conf(home_jv)

    world_poses = assembly_def.compute_world_poses(
        fixture_pos=fixture_pos, fixture_rotmat=fixture_rotmat,
    )

    search_part_ids = [p for p in part_ids if p in staging_seeds]
    if not search_part_ids:
        raise ValueError("staging_seeds 与 part_ids 无交集，无可搜索零件。")

    staging_obstacles = _build_staging_obstacles(
        search_part_ids, staging_seeds, assembly_def)
    goal_models = _build_goal_obstacle_models(assembly_def, world_poses)

    candidate_map: Dict[str, List[np.ndarray]] = {}
    for pid in search_part_ids:
        seed = np.asarray(staging_seeds[pid], dtype=float)
        zone = (candidate_zones or {}).get(pid)
        cross = (cross_arm_zones or {}).get(pid)
        candidate_map[pid] = generate_staging_candidates(
            seed, zone, cross, max_candidates=max_candidates_per_part)
    if verbose and max_candidates_per_part:
        ns = {pid: len(candidate_map[pid]) for pid in search_part_ids}
        print(f"[候选] 每件零件最多 {max_candidates_per_part} 个候选 → {ns}")

    chosen: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    arm_choice: Dict[str, str] = {}
    selected: List[str] = []

    def _sync_obstacles(part_id: str, cand_pos: np.ndarray, cand_rot: np.ndarray):
        for pid in search_part_ids:
            obs = staging_obstacles.get(pid)
            if obs is None:
                continue
            if pid == part_id:
                obs.pos = cand_pos
                obs.rotmat = cand_rot
            elif pid in chosen:
                p, r = chosen[pid]
                obs.pos = p
                obs.rotmat = r
            else:
                obs.pos = staging_seeds[pid].copy()
                obs.rotmat = np.eye(3)

    def _obs_for_reason(part_id: str) -> List:
        obs = list(env_obstacles)
        for pid in search_part_ids:
            if pid != part_id:
                cm = staging_obstacles.get(pid)
                if cm is not None:
                    obs.append(cm)
        return obs

    def _dfs(depth: int) -> bool:
        if depth >= len(search_part_ids):
            return True
        pid = search_part_ids[depth]
        base_rot = np.eye(3)
        cands = candidate_map[pid]
        if verbose:
            print(f"\n  [search] {pid}: {len(cands)} candidates")
        probe = staging_obstacles.get(pid)

        for idx, cand_pos in enumerate(cands):
            if probe is not None:
                probe.pos = cand_pos
                probe.rotmat = base_rot

            if not _is_clear_of_selected(
                    probe, cand_pos, base_rot, staging_obstacles, selected):
                if verbose:
                    print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} 与已选件碰撞")
                continue

            if not _is_clear_of_foreign_goals(
                    probe, cand_pos, base_rot, goal_models, assembly_def, pid):
                if verbose:
                    print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} 侵入 fixture 装配域")
                continue

            if pid not in world_poses:
                if verbose:
                    print(f"    - #{idx + 1} {pid} 无装配 goal，跳过")
                continue
            gp, gr = world_poses[pid]

            alias = model_alias_fn(pid)
            gc = grasp_cache.get(alias)
            if gc is None:
                if verbose:
                    print(f"    - #{idx + 1} 抓取库缺 alias={alias!r}")
                continue

            _sync_obstacles(pid, cand_pos, base_rot)
            obs = _obs_for_reason(pid)

            # "初始穿模"硬过滤：候选位姿(及已选件)在双臂 home 姿态下不能
            # 与任一只臂的几何体重叠。reason_common_gids 只看 pick/place
            # pose 处的碰撞，无法捕捉桌腿"躺着伸进臂里"的初始穿模。
            home_obs = [staging_obstacles[p]
                        for p in search_part_ids
                        if p in staging_obstacles]
            if _arms_collide_at_home(robot_dual, home_obs):
                if verbose:
                    print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} "
                          f"home 姿态下与机械臂穿模")
                continue

            picked_tag, picked_n = None, 0
            for tag in arm_priority:
                arm = arm_lookup[tag]
                ok, n = _common_grasps_ok(
                    arm, gc, cand_pos, base_rot, gp, gr, obs,
                    pick_depart_dir=pick_depart_dir,
                    pick_depart_dist=pick_depart_dist,
                    place_approach_dir=place_approach_dir,
                    place_approach_dist=place_approach_dist)
                if ok:
                    picked_tag, picked_n = tag, n
                    break

            if picked_tag is None:
                if verbose:
                    print(f"    - #{idx + 1} {np.round(cand_pos, 3).tolist()} "
                          f"双臂均无 reason_common 抓取")
                continue

            chosen[pid] = (cand_pos.copy(), base_rot.copy())
            arm_choice[pid] = picked_tag
            selected.append(pid)
            if verbose:
                print(f"    + #{idx + 1} {np.round(cand_pos, 3).tolist()} "
                      f"({picked_tag}, n={picked_n})")

            if _dfs(depth + 1):
                return True

            selected.pop()
            chosen.pop(pid, None)
            arm_choice.pop(pid, None)

        if probe is not None:
            probe.pos = staging_seeds[pid].copy()
            probe.rotmat = np.eye(3)
        return False

    if verbose:
        print(f"\n[search] DFS 联合搜索 {len(search_part_ids)} 件 staging "
              f"(件间无碰撞 + 不侵入装配域 + 双臂之一可 reason_common)…")

    if not _dfs(0):
        raise RuntimeError(
            f"未找到满足条件的 {len(search_part_ids)} 件可行布局。"
            f"建议：扩大 candidate_zones、调整 staging_seeds 或检查抓取库。")

    layout = WorkspaceLayout(
        robot_base_pos=np.asarray(robot_base_pos, dtype=float).copy(),
        robot_base_rotmat=np.asarray(robot_base_rotmat, dtype=float).copy(),
        assembly_station_pos=np.asarray(fixture_pos, dtype=float).copy(),
        assembly_station_rotmat=np.asarray(fixture_rotmat, dtype=float).copy(),
        staging_positions={pid: chosen[pid] for pid in search_part_ids},
        name=layout_name,
        metadata={
            "search_method": "dual_dfs",
            "arm_choice": dict(arm_choice),
            "n_parts_searched": len(search_part_ids),
        },
    )

    if verbose:
        print("\n[search] 成功：")
        for pid in search_part_ids:
            p, _ = chosen[pid]
            print(f"  {pid}: {np.round(p, 3).tolist()} ({arm_choice[pid]})")

    return layout
