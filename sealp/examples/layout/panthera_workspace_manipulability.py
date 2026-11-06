#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize dual Panthera-HT workspace manipulability.

This script visualizes a *global* spherical workspace around the dual
Panthera-HT system. By default it samples joint configurations, projects TCP
positions into one large translucent sphere, evaluates Yoshikawa
manipulability at each reachable TCP position, and visualizes the result in
WRS/Panda3D.

Color convention:
    low manipulability  -> light / transparent
    high manipulability -> dark / opaque

Example:
    python -m sealp.examples.layout.panthera_workspace_manipulability
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm import (
    DualPantheraHTNoBody,
)


@dataclass(frozen=True)
class ArmSampleResult:
    arm_name: str
    base_pos: np.ndarray
    sample_center: np.ndarray
    points: np.ndarray
    manipulability: np.ndarray


@dataclass(frozen=True)
class GlobalSphere:
    center: np.ndarray
    radius: float


def _orientation_candidates(mode: str) -> List[np.ndarray]:
    """Return TCP orientation candidates.

    ``fast`` is sufficient for a quick workspace picture. ``paper`` uses more
    directions and is better when the figure is meant to support a paper.
    """
    if mode == "identity":
        return [np.eye(3)]

    yaws = [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0]
    rots: List[np.ndarray] = []
    for yaw in yaws:
        rots.append(rm.rotmat_from_euler(0.0, 0.0, yaw))

    if mode == "paper":
        for pitch in (np.pi / 2.0, -np.pi / 2.0):
            for yaw in yaws:
                rots.append(rm.rotmat_from_euler(0.0, pitch, yaw))
        for roll in (np.pi / 2.0, -np.pi / 2.0):
            for yaw in (0.0, np.pi):
                rots.append(rm.rotmat_from_euler(roll, 0.0, yaw))

    return rots


def _seed_confs() -> List[np.ndarray]:
    """Deterministic IK seeds covering folded / stretched configurations."""
    seeds = [np.zeros(6)]
    for j0 in (0.0, 0.8, -0.8):
        for j1, j2 in ((0.6, 0.8), (1.2, 1.6), (2.0, 2.6), (2.7, 3.3)):
            seeds.append(np.array([j0, j1, j2, 0.0, 0.0, 0.0]))
    return seeds


def _sample_points_in_sphere(center: np.ndarray,
                             radius: float,
                             grid_res: int,
                             z_min: float) -> np.ndarray:
    """Uniform Cartesian grid clipped by a sphere."""
    xs = np.linspace(center[0] - radius, center[0] + radius, grid_res)
    ys = np.linspace(center[1] - radius, center[1] + radius, grid_res)
    zs = np.linspace(center[2] - radius, center[2] + radius, grid_res)
    pts = []
    r2 = radius * radius
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z], dtype=float)
                if z < z_min:
                    continue
                if np.sum((p - center) ** 2) <= r2:
                    pts.append(p)
    return np.asarray(pts, dtype=float)


def _yoshikawa_manipulability(arm, jnt_values: np.ndarray) -> float:
    """Yoshikawa manipulability sqrt(det(J J^T)) with numerical guard.

    The Panthera manipulator path currently exposes a reliable Jacobian for
    the *current* state, so we move the arm to the IK solution before reading
    manipulability. This also matches the actual simulated configuration.
    """
    arm.goto_given_conf(jnt_values=jnt_values)
    val = float(arm.manipulability_val())
    if not np.isfinite(val) or val <= 0.0:
        return 0.0
    return val


def _evaluate_point(arm,
                    pos: np.ndarray,
                    rotmats: Iterable[np.ndarray],
                    seeds: Iterable[np.ndarray]) -> float:
    """Best manipulability at this TCP position over orientations and seeds."""
    best = 0.0
    for rotmat in rotmats:
        for seed in seeds:
            try:
                jnt_values = arm.ik(
                    tgt_pos=pos,
                    tgt_rotmat=rotmat,
                    seed_jnt_values=seed,
                )
            except Exception:
                jnt_values = None
            if jnt_values is None:
                continue
            val = _yoshikawa_manipulability(arm, np.asarray(jnt_values))
            if val > best:
                best = val
    return best


def _cache_path(out_dir: str,
                grid_res: int,
                radius: float,
                center_x: float,
                center_z: float,
                arm_y_offset: float,
                orientation_mode: str,
                method: str,
                n_fk_samples: int) -> str:
    tag = (
        f"{method}_grid{grid_res}_fk{n_fk_samples}_r{radius:.2f}"
        f"_cx{center_x:.2f}_cz{center_z:.2f}_dy{arm_y_offset:.2f}"
        f"_{orientation_mode}"
    )
    tag = tag.replace(".", "p")
    return os.path.join(out_dir, f"panthera_dual_workspace_{tag}.npz")


def _global_sphere(center_x: float,
                   center_z: float,
                   arm_y_offset: float,
                   radius: float) -> GlobalSphere:
    """One global sphere covering both arms."""
    center = np.array([center_x, -0.5 * arm_y_offset, center_z], dtype=float)
    return GlobalSphere(center=center, radius=radius)


def _joint_ranges(arm) -> np.ndarray:
    """Return Panthera arm joint ranges as ``(n_dof, 2)``."""
    ranges = []
    for jnt in arm.manipulator.jlc.jnts[:arm.manipulator.jlc.n_dof]:
        ranges.append(np.asarray(jnt.motion_range, dtype=float))
    return np.asarray(ranges, dtype=float)


def compute_workspace_fk(robot: DualPantheraHTNoBody,
                         *,
                         n_fk_samples: int,
                         radius: float,
                         center_x: float,
                         center_z: float,
                         arm_y_offset: float,
                         z_min: float,
                         rng_seed: int) -> Dict[str, ArmSampleResult]:
    """Sample joint space and project reachable TCPs into one global sphere.

    This is the preferred view for a paper-style global workspace figure:
    it does not assume a single TCP orientation, so the front workspace will
    not disappear simply because an IK orientation candidate was too strict.
    """
    sphere = _global_sphere(center_x, center_z, arm_y_offset, radius)
    rng = np.random.default_rng(rng_seed)
    arm_specs = {
        "lft": (robot.lft_arm, np.array([0.0, 0.0, 0.0])),
        "rgt": (robot.rgt_arm, np.array([0.0, -arm_y_offset, 0.0])),
    }
    results: Dict[str, ArmSampleResult] = {}
    print(
        f"[sample-fk] global sphere center={sphere.center.tolist()} "
        f"radius={sphere.radius:.2f}, samples/arm={n_fk_samples}"
    )
    for arm_name, (arm, base_pos) in arm_specs.items():
        ranges = _joint_ranges(arm)
        pts: List[np.ndarray] = []
        vals: List[float] = []
        t0 = time.time()
        for i in range(n_fk_samples):
            q = rng.uniform(ranges[:, 0], ranges[:, 1])
            arm.goto_given_conf(q)
            pos = np.asarray(arm.manipulator.gl_tcp_pos, dtype=float).copy()
            if pos[2] < z_min:
                continue
            if np.linalg.norm(pos - sphere.center) > sphere.radius:
                continue
            mu = float(arm.manipulability_val())
            if not np.isfinite(mu) or mu <= 0.0:
                continue
            pts.append(pos)
            vals.append(mu)
            if (i + 1) % 1000 == 0:
                print(
                    f"  {arm_name}: sampled={i + 1:>6d}, "
                    f"in_sphere={len(pts):>5d}, max_mu={max(vals):.5f}",
                    end="\r",
                )
        print()
        points = np.asarray(pts, dtype=float)
        manip = np.asarray(vals, dtype=float)
        print(
            f"  {arm_name}: done in {time.time() - t0:.1f}s, "
            f"draw_points={len(points)}, max_mu={manip.max() if len(manip) else 0:.5f}"
        )
        results[arm_name] = ArmSampleResult(
            arm_name=arm_name,
            base_pos=base_pos,
            sample_center=sphere.center,
            points=points,
            manipulability=manip,
        )
    return results


def compute_workspace(robot: DualPantheraHTNoBody,
                      *,
                      grid_res: int,
                      radius: float,
                      center_x: float,
                      center_z: float,
                      arm_y_offset: float,
                      z_min: float,
                      orientation_mode: str) -> Dict[str, ArmSampleResult]:
    """Sample both arms and compute manipulability."""
    rotmats = _orientation_candidates(orientation_mode)
    seeds = _seed_confs()
    print(
        f"[sample] grid={grid_res} radius={radius:.2f} "
        f"orientations={len(rotmats)} seeds={len(seeds)}"
    )

    arm_specs = {
        "lft": (robot.lft_arm, np.array([0.0, 0.0, 0.0])),
        "rgt": (robot.rgt_arm, np.array([0.0, -arm_y_offset, 0.0])),
    }
    results: Dict[str, ArmSampleResult] = {}

    for arm_name, (arm, base_pos) in arm_specs.items():
        center = base_pos + np.array([center_x, 0.0, center_z])
        points = _sample_points_in_sphere(
            center=center,
            radius=radius,
            grid_res=grid_res,
            z_min=z_min,
        )
        manip = np.zeros(len(points), dtype=float)
        t0 = time.time()
        print(f"[{arm_name}] {len(points)} points")
        for i, p in enumerate(points):
            manip[i] = _evaluate_point(arm, p, rotmats, seeds)
            if (i + 1) % 100 == 0 or i + 1 == len(points):
                print(
                    f"  {arm_name}: {i + 1:>4d}/{len(points)} "
                    f"reachable={(manip > 0).sum():>4d} "
                    f"max_mu={manip.max():.5f}",
                    end="\r",
                )
        print()
        print(
            f"  {arm_name}: done in {time.time() - t0:.1f}s, "
            f"reachable={(manip > 0).sum()}/{len(points)}, "
            f"max_mu={manip.max():.5f}"
        )
        results[arm_name] = ArmSampleResult(
            arm_name=arm_name,
            base_pos=base_pos,
            sample_center=center,
            points=points,
            manipulability=manip,
        )
    return results


def save_cache(cache_file: str, results: Dict[str, ArmSampleResult]) -> None:
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    payload = {}
    for arm_name, res in results.items():
        payload[f"{arm_name}_base_pos"] = res.base_pos
        payload[f"{arm_name}_sample_center"] = res.sample_center
        payload[f"{arm_name}_points"] = res.points
        payload[f"{arm_name}_manipulability"] = res.manipulability
    np.savez(cache_file, **payload)
    print(f"[cache] saved -> {os.path.relpath(cache_file)}")


def load_cache(cache_file: str) -> Dict[str, ArmSampleResult]:
    data = np.load(cache_file)
    results = {}
    for arm_name in ("lft", "rgt"):
        results[arm_name] = ArmSampleResult(
            arm_name=arm_name,
            base_pos=data[f"{arm_name}_base_pos"],
            sample_center=data[f"{arm_name}_sample_center"],
            points=data[f"{arm_name}_points"],
            manipulability=data[f"{arm_name}_manipulability"],
        )
    print(f"[cache] loaded <- {os.path.relpath(cache_file)}")
    return results


def _interp_color(norm_val: float, arm_name: str) -> Tuple[np.ndarray, float]:
    """Low value: light; high value: saturated/dark."""
    v = float(np.clip(norm_val, 0.0, 1.0))
    # Use one global color map so the figure reads as a single dual-arm
    # workspace instead of two separate arm clouds.
    light = np.array([0.86, 0.94, 1.00])
    dark = np.array([0.02, 0.10, 0.58])
    rgb = light * (1.0 - v) + dark * v
    alpha = 0.20 + 0.75 * v
    return rgb, alpha


def visualize(results: Dict[str, ArmSampleResult],
              *,
              robot: DualPantheraHTNoBody,
              radius: float,
              point_radius: float,
              show_unreachable: bool,
              max_points: int,
              run_sim: bool) -> None:
    """Visualize workspace samples in WRS/Panda3D."""
    base = wd.World(
        cam_pos=[1.6, 1.0, 1.15],
        lookat_pos=[0.25, -0.30, 0.25],
        auto_rotate=False,
    )
    mgm.gen_frame().attach_to(base)
    robot.gen_meshmodel(alpha=0.38, toggle_tcp_frame=True).attach_to(base)

    global_max = max(
        float(np.max(res.manipulability)) if len(res.manipulability) else 0.0
        for res in results.values()
    )
    if global_max <= 0.0:
        global_max = 1.0

    # One large transparent sphere: the global dual-arm operation space.
    first_res = next(iter(results.values()))
    mgm.gen_sphere(
        pos=first_res.sample_center,
        radius=radius,
        rgb=np.array([0.55, 0.72, 0.95]),
        alpha=0.055,
        ico_level=2,
    ).attach_to(base)

    for arm_name, res in results.items():
        reachable_mask = res.manipulability > 0.0
        visible_mask = reachable_mask if not show_unreachable \
            else np.ones_like(reachable_mask, dtype=bool)
        indices = np.where(visible_mask)[0]
        if len(indices) > max_points:
            # Deterministic thinning for interactive rendering speed.
            stride = int(np.ceil(len(indices) / max_points))
            indices = indices[::stride]

        print(
            f"[draw] {arm_name}: drawing {len(indices)} / "
            f"{len(res.points)} samples"
        )
        for idx in indices:
            mu = float(res.manipulability[idx])
            if mu <= 0.0:
                if not show_unreachable:
                    continue
                rgb, alpha = np.array([0.92, 0.92, 0.92]), 0.055
            else:
                rgb, alpha = _interp_color(mu / global_max, arm_name)
            mgm.gen_sphere(
                pos=res.points[idx],
                radius=point_radius,
                rgb=rgb,
                alpha=alpha,
                ico_level=1,
            ).attach_to(base)

    # Color legend: left side vertical manipulability scale.
    legend_x, legend_y = -0.18, 0.22
    for k, v in enumerate(np.linspace(0.0, 1.0, 10)):
        z = 0.05 + 0.035 * k
        rgb, alpha = _interp_color(v, "lft")
        mgm.gen_sphere(
            pos=np.array([legend_x, legend_y, z]),
            radius=point_radius * 1.3,
            rgb=rgb,
            alpha=max(alpha, 0.35),
            ico_level=1,
        ).attach_to(base)

    print("[view] low manipulability = light color; high = dark color")
    if run_sim:
        base.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dual Panthera-HT spherical workspace "
                    "manipulability."
    )
    parser.add_argument("--method", choices=["fk", "ik"], default="fk",
                        help="fk=joint-space global workspace (default); "
                             "ik=Cartesian grid with orientation candidates")
    parser.add_argument("--n-fk-samples", type=int, default=10000,
                        help="FK random joint samples per arm")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for FK sampling")
    parser.add_argument("--grid-res", type=int, default=13,
                        help="IK mode: samples per axis before spherical clipping")
    parser.add_argument("--radius", type=float, default=0.82,
                        help="global workspace sphere radius [m]")
    parser.add_argument("--center-x", type=float, default=0.08,
                        help="global sphere center x [m]")
    parser.add_argument("--center-z", type=float, default=0.34,
                        help="global sphere center z [m]")
    parser.add_argument("--arm-y-offset", type=float, default=0.62,
                        help="right arm y offset from left arm [m]")
    parser.add_argument("--z-min", type=float, default=0.03,
                        help="ignore points below this world z [m]")
    parser.add_argument("--orientation-mode",
                        choices=["identity", "fast", "paper"],
                        default="fast",
                        help="TCP orientation set used for IK search")
    parser.add_argument("--point-radius", type=float, default=0.010,
                        help="visual sphere radius for each sample [m]")
    parser.add_argument("--show-unreachable", action="store_true",
                        help="draw unreachable points as very light gray")
    parser.add_argument("--max-points", type=int, default=4500,
                        help="max points drawn per arm for interactive speed")
    parser.add_argument("--out-dir", default=os.path.join(
        _HERE, "_output", "panthera_workspace_test"),
        help="cache output directory")
    parser.add_argument("--no-cache", action="store_true",
                        help="force recomputation even if cache exists")
    parser.add_argument("--cache-only", action="store_true",
                        help="compute/cache but do not open simulation window")
    parser.add_argument("--no-run", action="store_true",
                        help="build visual scene but do not call base.run()")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_file = _cache_path(
        args.out_dir,
        args.grid_res,
        args.radius,
        args.center_x,
        args.center_z,
        args.arm_y_offset,
        args.orientation_mode,
        args.method,
        args.n_fk_samples,
    )

    robot = DualPantheraHTNoBody(
        arm_y_offset=args.arm_y_offset,
        enable_cc=False,
    )

    if os.path.isfile(cache_file) and not args.no_cache:
        results = load_cache(cache_file)
    else:
        if args.method == "fk":
            results = compute_workspace_fk(
                robot,
                n_fk_samples=args.n_fk_samples,
                radius=args.radius,
                center_x=args.center_x,
                center_z=args.center_z,
                arm_y_offset=args.arm_y_offset,
                z_min=args.z_min,
                rng_seed=args.seed,
            )
        else:
            results = compute_workspace(
                robot,
                grid_res=args.grid_res,
                radius=args.radius,
                center_x=args.center_x,
                center_z=args.center_z,
                arm_y_offset=args.arm_y_offset,
                z_min=args.z_min,
                orientation_mode=args.orientation_mode,
            )
        save_cache(cache_file, results)

    for arm_name, res in results.items():
        n_total = len(res.points)
        n_reach = int(np.sum(res.manipulability > 0.0))
        max_mu = float(np.max(res.manipulability)) if n_total else 0.0
        mean_mu = float(np.mean(res.manipulability[res.manipulability > 0.0])) \
            if n_reach else 0.0
        print(
            f"[summary] {arm_name}: reachable={n_reach}/{n_total} "
            f"({100.0 * n_reach / max(n_total, 1):.1f}%), "
            f"max_mu={max_mu:.5f}, mean_reachable_mu={mean_mu:.5f}"
        )

    if args.cache_only:
        return

    visualize(
        results,
        robot=robot,
        radius=args.radius,
        point_radius=args.point_radius,
        show_unreachable=args.show_unreachable,
        max_points=args.max_points,
        run_sim=not args.no_run,
    )


if __name__ == "__main__":
    main()
