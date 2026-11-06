"""Shelf-Unit Task Adapter for ``find_optimal_layout_shelf.py``"""
from __future__ import annotations

import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from sealp.examples.layout._tasks.shelf_geometry import (
    AUTO_ROTMAT_CACHE_TAG,
    DEFAULT_TABLE_MARGIN,
    DUAL_ARM_Y_OFFSET,
    FIXTURE_POS,
    FIXTURE_ROTMAT,
    PART_XY_BOUNDS,
    ROBOT_BASE_POS,
    ROBOT_BASE_ROTMAT,
    SHELF_UPRIGHT_RX90,
    SHELF_UPRIGHT_Z_OFFSET,
    SIDE_UPRIGHT_Z_OFFSET,
    STAGING_SEEDS,
    part_xy_bounds_shelf_reachable,
    work_table_xy_bounds,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))


def _here(*parts: str) -> str:
    return os.path.abspath(os.path.join(_HERE, *parts))


_SHELF_UPRIGHT_FALLBACK = (SHELF_UPRIGHT_RX90.copy(), SHELF_UPRIGHT_Z_OFFSET)
_SIDE_UPRIGHT_FALLBACK = (np.eye(3), SIDE_UPRIGHT_Z_OFFSET)


def model_alias_for_part(part_id: str) -> str:
    return "side_model" if part_id.startswith("side_") else "shelf_model"


def _compute_rotmat_candidates(
    *,
    asmdef_path: str,
    grasp_pickles: Dict[str, str],
    part_ids: Tuple[str, ...],
    staging_seeds: Dict[str, np.ndarray],
    fixture_pos: np.ndarray,
    fixture_rotmat: np.ndarray,
    robot_base_pos: np.ndarray,
    robot_base_rotmat: np.ndarray,
    arm_y_offset: float,
    config_yaml: Optional[str] = None,
    top_k: int = 6,
    n_yaw: int = 8,
    min_common_gids: int = 1,
    verbose: bool = False,
) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
    from sealp.assembly_sequence.asmdef import AssemblyDef
    from sealp.examples.layout._tasks.auto_rotmat import auto_staging_rotmat_candidates

    asm = AssemblyDef.load(asmdef_path)
    world_poses = asm.compute_world_poses(
        fixture_pos=fixture_pos, fixture_rotmat=fixture_rotmat)

    grasp_cache: Dict[str, object] = {}
    for alias, path in grasp_pickles.items():
        with open(path, "rb") as fh:
            grasp_cache[alias] = pickle.load(fh)

    robot = pda.DualPantheraHTNoBody(
        pos=robot_base_pos, rotmat=robot_base_rotmat,
        arm_y_offset=arm_y_offset, enable_cc=True)
    robot.lft_arm.goto_given_conf(np.zeros(6))
    robot.rgt_arm.goto_given_conf(np.zeros(6))
    arms = [("lft", robot.lft_arm), ("rgt", robot.rgt_arm)]

    env_obstacles: List[object] = []
    if config_yaml and os.path.isfile(config_yaml):
        try:
            from sealp.config import load_config
            from sealp.colliders import StaticEnvironment
            cfg = load_config(config_yaml)
            env = StaticEnvironment(
                obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
            env_obstacles = list(env.obstacle_list)
            if verbose:
                print(f"[shelf_unit] auto_rotmat: {len(env_obstacles)} env obstacle(s)")
        except Exception as e:
            if verbose:
                print(f"[shelf_unit][WARN] env obstacles: {e!r}")

    out: Dict[str, List[Tuple[np.ndarray, float]]] = {}
    if verbose:
        print(f"[shelf_unit] fixture={fixture_pos.tolist()} auto-infer staging rotmats:")
    for pid in part_ids:
        gp, gr = world_poses[pid]
        gc = grasp_cache[model_alias_for_part(pid)]
        mesh_path = asm.model_path(pid)
        staging_xy = np.asarray(staging_seeds[pid])[:2]
        cands = auto_staging_rotmat_candidates(
            mesh_path=mesh_path,
            grasp_collection=gc,
            goal_pos=gp, goal_rotmat=gr,
            staging_xy=staging_xy,
            arms=arms,
            n_yaw=n_yaw,
            top_k=top_k,
            min_common_gids=min_common_gids,
            max_grasps_per_probe=128,
            obstacle_list=env_obstacles,
            cache_tag=f"{AUTO_ROTMAT_CACHE_TAG}_{pid}",
            verbose=verbose,
            filter_flat_for_thin=True,
            thin_aspect_ratio=0.35,
            upright_height_ratio=0.35,
        )
        if not cands:
            fallback = _SIDE_UPRIGHT_FALLBACK if pid.startswith("side_") else _SHELF_UPRIGHT_FALLBACK
            cands = [(fallback[0].copy(), float(fallback[1]))]
            if verbose:
                print(f"  [{pid}] fallback upright z_off={fallback[1]:.4f}")
        if verbose:
            print(f"  [{pid}] {len(cands)} rotmat candidate(s)")
        out[pid] = cands
    return out


def _prepend_mesh_staging_rotmats(
    rotmat_dict: Dict[str, List[Tuple[np.ndarray, float]]],
) -> None:
    """把 mesh_frames 推导的竖立 staging 插到候选首位（优先于 auto_rotmat 躺姿）。"""
    from sealp.assets.models.shelf_unit.mesh_frames import (
        SHELF_STAGING_ROTMAT,
        SHELF_STAGING_Z_OFFSET,
    )

    I = np.eye(3)
    prefs = {
        "side_l": (I, float(SIDE_UPRIGHT_Z_OFFSET)),
        "side_r": (I, float(SIDE_UPRIGHT_Z_OFFSET)),
        "shelf_m": (SHELF_STAGING_ROTMAT.copy(), float(SHELF_STAGING_Z_OFFSET)),
        "shelf_t": (SHELF_STAGING_ROTMAT.copy(), float(SHELF_STAGING_Z_OFFSET)),
    }

    def _same(a: Tuple[np.ndarray, float], b: Tuple[np.ndarray, float]) -> bool:
        return bool(np.allclose(a[0], b[0], atol=1e-6) and abs(float(a[1]) - float(b[1])) < 1e-6)

    for pid, pref in prefs.items():
        if pid not in rotmat_dict:
            continue
        rest = [c for c in rotmat_dict[pid] if not _same(c, pref)]
        rotmat_dict[pid] = [(pref[0].copy(), float(pref[1]))] + rest


def print_preflight_diagnostics(
    *,
    asmdef_path: str,
    fixture_pos: np.ndarray,
    grasp_pickles: Dict[str, str],
    rotmat_dict: Dict[str, List[Tuple[np.ndarray, float]]],
    part_ids: Tuple[str, ...],
) -> None:
    """搜索前打印 goal/staging 几何与双臂 goal 可达性摘要。"""
    import pickle

    from sealp.assembly_sequence.asmdef import AssemblyDef
    from sealp.assets.models.shelf_unit import mesh_frames as mf
    import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
    from sealp.layout.reachability import check_pose_reachability

    print("\n" + "=" * 60)
    print("[Preflight] Shelf Unit 几何 / goal / staging 诊断")
    print("=" * 60)
    print(f"  fixture_pos = {np.asarray(fixture_pos).round(4).tolist()}")
    print(f"  side mesh extent  = {mf.SIDE_FRAME.extent.round(4).tolist()}")
    print(f"  shelf mesh extent = {mf.SHELF_FRAME.extent.round(4).tolist()}")
    vertical = mf._shelf_is_vertical_at_identity(mf.SHELF_FRAME)
    print(f"  shelf vertical@I  = {vertical}  "
          f"(staging rot=I, z_off={mf.SHELF_STAGING_Z_OFFSET:.4f})")

    asm = AssemblyDef.load(asmdef_path)
    world_poses = asm.compute_world_poses(
        fixture_pos=fixture_pos, fixture_rotmat=FIXTURE_ROTMAT)

    print("\n  ── World goal 位姿（fixture 固定，未搜索）──")
    for pid in part_ids:
        gp, gr = world_poses[pid]
        z_ax = gr[:, 2]
        print(f"    {pid:8s} pos={np.round(gp, 4).tolist()}  z_axis={np.round(z_ax, 3).tolist()}")

    # staging vs goal 旋转差（层板）
    for pid in ("shelf_m", "shelf_t"):
        if pid not in rotmat_dict or not rotmat_dict[pid]:
            continue
        sr, _ = rotmat_dict[pid][0]
        _, gr = world_poses[pid]
        R = gr.T @ sr
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        print(f"    {pid} 首选 staging[0] vs goal 旋转角 ≈ {ang:.1f}°")

    grasp_cache: Dict[str, object] = {}
    for alias, path in grasp_pickles.items():
        with open(path, "rb") as fh:
            grasp_cache[alias] = pickle.load(fh)
        print(f"  grasp '{alias}': {len(grasp_cache[alias])} from {os.path.relpath(path)}")

    robot = pda.DualPantheraHTNoBody(
        pos=ROBOT_BASE_POS, rotmat=ROBOT_BASE_ROTMAT,
        arm_y_offset=DUAL_ARM_Y_OFFSET, enable_cc=True)
    robot.lft_arm.goto_given_conf(np.zeros(6))
    robot.rgt_arm.goto_given_conf(np.zeros(6))

    print("\n  ── Goal 单点可达性（无障碍，仅 IK + manip>0）──")
    for pid in part_ids:
        gp, gr = world_poses[pid]
        gc = grasp_cache[model_alias_for_part(pid)]
        for arm_tag, arm in (("lft", robot.lft_arm), ("rgt", robot.rgt_arm)):
            res = check_pose_reachability(arm, gp, gr, gc, [], max_grasps=20)
            print(f"    {pid:8s} {arm_tag}: reachable={res.reachable}  "
                  f"n_grasps={res.n_reachable_grasps}  "
                  f"best_manip={res.best_manipulability:.4f}")

    print("\n  ── Staging rotmat 候选（注入后，[0]=mesh 竖立优先）──")
    for pid in part_ids:
        cands = rotmat_dict.get(pid, [])
        if not cands:
            print(f"    {pid}: (empty)")
            continue
        r0, z0 = cands[0]
        z_ax = r0[:, 2]
        print(f"    {pid}: {len(cands)} cands  [0] z_off={z0:.4f}  z_axis={np.round(z_ax, 3).tolist()}")

    print("\n  说明：布局搜索只优化 staging (x,y)；fixture 未按双臂 manip 动态优化。")
    if rotmat_dict:
        try:
            from sealp.examples.layout._tasks.shelf_geometry import work_table_xy_bounds
            cfg_path = os.path.join(_PROJ_ROOT, "sealp", "config", "sample_config.yaml")
            (xlo, xhi), (ylo, yhi) = work_table_xy_bounds(cfg_path)
            print(f"  staging 搜索区 = work_table 全桌面 "
                  f"x∈[{xlo:.3f},{xhi:.3f}] y∈[{ylo:.3f},{yhi:.3f}]")
        except Exception:
            pass
    print("=" * 60)


def staging_rotmat_candidates(*, verbose: bool = False) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """各 shelf 零件的 staging (rotmat, z_offset) 候选（与布局搜索一致）。"""
    asmdef_path = os.path.join(
        _PROJ_ROOT, "sealp", "assembly_sequence", "_demo_output", "shelf_unit.asmdef")
    grasp_dir = os.path.join(_PROJ_ROOT, "sealp", "examples", "grasp", "_output")
    grasp_pickles = {
        "shelf_model": os.path.join(grasp_dir, "demo_shelf_unit-shelf_grasps.pickle"),
        "side_model": os.path.join(grasp_dir, "demo_shelf_unit-side_grasps.pickle"),
    }
    config_yaml = os.path.join(_PROJ_ROOT, "sealp", "config", "sample_config.yaml")
    part_ids: Tuple[str, ...] = ("side_l", "shelf_m", "shelf_t", "side_r")
    staging_seeds = {k: v.copy() for k, v in STAGING_SEEDS.items()}
    return _compute_rotmat_candidates(
        asmdef_path=asmdef_path,
        grasp_pickles=grasp_pickles,
        part_ids=part_ids,
        staging_seeds=staging_seeds,
        fixture_pos=FIXTURE_POS,
        fixture_rotmat=FIXTURE_ROTMAT,
        robot_base_pos=ROBOT_BASE_POS,
        robot_base_rotmat=ROBOT_BASE_ROTMAT,
        arm_y_offset=DUAL_ARM_Y_OFFSET,
        config_yaml=config_yaml,
        top_k=6,
        n_yaw=8,
        min_common_gids=1,
        verbose=verbose,
    )


def register_task(*,
                  here_fn: Optional[Callable[..., str]] = None,
                  verbose: bool = True,
                  config_yaml: Optional[str] = None,
                  table_margin: float = DEFAULT_TABLE_MARGIN,
                  skip_parts: Tuple[str, ...] = (),
                  ) -> Tuple[object, Dict[str, List[Tuple[np.ndarray, float]]]]:
    from sealp.examples.layout.find_optimal_layout import FastSearchTask

    asmdef_path = os.path.join(
        _PROJ_ROOT, "sealp", "assembly_sequence", "_demo_output", "shelf_unit.asmdef")
    grasp_dir = os.path.join(_PROJ_ROOT, "sealp", "examples", "grasp", "_output")
    grasp_pickles = {
        "shelf_model": os.path.join(grasp_dir, "demo_shelf_unit-shelf_grasps.pickle"),
        "side_model": os.path.join(grasp_dir, "demo_shelf_unit-side_grasps.pickle"),
    }
    if config_yaml is None:
        config_yaml = os.path.join(_PROJ_ROOT, "sealp", "config", "sample_config.yaml")

    all_part_ids: Tuple[str, ...] = ("side_l", "shelf_m", "shelf_t", "side_r")
    skip_set = frozenset(skip_parts)
    part_ids = tuple(p for p in all_part_ids if p not in skip_set)
    if not part_ids:
        raise ValueError(f"skip_parts={skip_parts!r} 后无剩余零件")
    if skip_set and verbose:
        print(f"[shelf_unit] skip_parts={sorted(skip_set)}  →  search parts={part_ids}")

    staging_seeds = {k: v.copy() for k, v in STAGING_SEEDS.items() if k in part_ids}

    try:
        xy_bounds = part_xy_bounds_shelf_reachable(
            part_ids, config_yaml, margin=table_margin)
        if verbose:
            (xlo, xhi), (ylo, yhi) = next(iter(xy_bounds.values()))
            print(f"[shelf_unit] staging xy bounds = work_table ∩ y-clip "
                  f"(margin={table_margin}m)")
            print(f"  x ∈ [{xlo:.3f}, {xhi:.3f}]  "
                  f"y ∈ [{ylo:.3f}, {yhi:.3f}]  (all parts)")
    except Exception as exc:
        if verbose:
            print(f"[shelf_unit][WARN] work_table bounds: {exc!r}; "
                  f"fallback PART_XY_BOUNDS")
        xy_bounds = {k: tuple((tuple(b) for b in v)) for k, v in PART_XY_BOUNDS.items()}

    rotmat_dict = staging_rotmat_candidates(verbose=verbose)
    _prepend_mesh_staging_rotmats(rotmat_dict)
    rotmat_dict = {k: v for k, v in rotmat_dict.items() if k in part_ids}

    out_name = "dual_shelf_unit_optimal_searched"
    if skip_set:
        out_name += "_skip_" + "_".join(sorted(skip_set))

    task = FastSearchTask(
        name="shelf_unit",
        asmdef_path=asmdef_path,
        grasp_pickles=grasp_pickles,
        part_ids=part_ids,
        staging_seeds=staging_seeds,
        fixture_pos=FIXTURE_POS.copy(),
        fixture_rotmat=FIXTURE_ROTMAT.copy(),
        robot_base_pos=ROBOT_BASE_POS.copy(),
        robot_base_rotmat=ROBOT_BASE_ROTMAT.copy(),
        config_yaml_path=config_yaml,
        table_margin=float(table_margin),
        part_xy_bounds=xy_bounds,
        model_alias_fn=model_alias_for_part,
        output_layout_name=out_name,
    )
    return task, rotmat_dict
