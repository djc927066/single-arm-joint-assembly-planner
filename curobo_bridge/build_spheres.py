"""用 cuRobo ``RobotBuilder``（MorphIt）给导出的 Panthera 拟合碰撞球，生成机器人 YAML。

运行（cuRobo venv）::

    & E:\\Projects\\wrs-sealp-assemble\\dual_sealp\\cuRobotest\\.venv\\Scripts\\python.exe `
        -X utf8 -m curobo_bridge.build_spheres --densities 0.5 1 2

每个密度输出 ``assets/<model>/panthera_d<density>.yml``，并在同目录写
``sphere_report.json``（每个 link 的球数、覆盖率、外凸率、表面间隙）。

写出的 YAML 在 cuRobo 生成内容之上补齐运动规划需要的 ``cspace`` 字段：
关节顺序、home、真实 URDF 速度上限对应的加速度/加加速度上限（cuRobo 默认值
偏向 Franka，这里显式给出，避免 TrajOpt 的时间估计被悄悄放大或缩小）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .runtime import ASSET_DIR, prepare

DEFAULT_MAX_ACCELERATION = 10.0   # rad/s^2, conservative for Panthera-HT
DEFAULT_MAX_JERK = 300.0          # rad/s^3


def build_model(model_dir: Path, density: float, *, seed: int = 0,
                iterations: int = 200, collision_samples: int = 1000,
                clip_base: bool = True) -> dict:
    import torch
    import yaml
    from curobo.robot_builder import RobotBuilder

    np.random.seed(seed)
    torch.manual_seed(seed)
    chain = json.loads((model_dir / "chain.json").read_text(encoding="utf-8"))
    builder = RobotBuilder(urdf_path=str(model_dir / "panthera_fit.urdf"),
                           asset_path=str(model_dir), tool_frames=["wrs_tcp"])
    clip = {"base_link": ("z", 0.0)} if (clip_base and "base_link" in chain["collision_links"]) else None
    started = time.perf_counter()
    builder.fit_collision_spheres(sphere_density=density, iterations=iterations,
                                  compute_metrics=True, clip_links=clip)
    fit_s = time.perf_counter() - started
    started = time.perf_counter()
    builder.compute_collision_matrix(prune_collisions=True, num_samples=collision_samples)
    matrix_s = time.perf_counter() - started
    config = builder.build()
    out = model_dir / f"panthera_d{density:g}.yml"
    builder.save(config, str(out))

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    kin = data["robot_cfg"]["kinematics"] if "robot_cfg" in data else data["kinematics"]
    kin["cspace"] = {
        "joint_names": list(chain["joint_names"]),
        "default_joint_position": list(chain["home"]),
        "null_space_weight": [1.0] * 6,
        "cspace_distance_weight": [1.0] * 6,
        "max_acceleration": DEFAULT_MAX_ACCELERATION,
        "max_jerk": DEFAULT_MAX_JERK,
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    metrics = {}
    for link, m in (builder.link_metrics or {}).items():
        metrics[link] = dict(spheres=int(m.num_spheres), coverage=float(m.coverage),
                             protrusion=float(m.protrusion),
                             protrusion_mm=float(m.protrusion_dist_mean * 1000),
                             surface_gap_mm=float(m.surface_gap_mean * 1000),
                             volume_ratio=float(m.volume_ratio))
    return dict(density=density, yaml=str(out), spheres=int(builder.num_spheres),
                fit_s=fit_s, matrix_s=matrix_s,
                self_collision_ignore_pairs=len(builder.collision_matrix or {}),
                links=metrics)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="panthera_wrsext")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    model_dir = ASSET_DIR / args.model
    reports = [build_model(model_dir, d, seed=args.seed) for d in args.densities]
    for r in reports:
        print(f"density {r['density']:g}: {r['spheres']} spheres, fit {r['fit_s']:.1f}s -> {r['yaml']}")
        for link, m in r["links"].items():
            print(f"    {link:14s} n={m['spheres']:3d} cover={m['coverage'] * 100:5.1f}% "
                  f"protr={m['protrusion'] * 100:5.1f}% prot={m['protrusion_mm']:.2f}mm "
                  f"gap={m['surface_gap_mm']:.2f}mm")
    (model_dir / "sphere_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    prepare()
    main()
