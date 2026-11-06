"""cuRobo 自碰撞 vs 真实网格自碰撞（WRS ODE 逐对求交）。

WRS 的 Panthera 不登记任何自碰撞对，所以这里的参照不是 WRS 语义，而是网格真值：
对 cuRobo 模型中未被忽略的每对 link，用 ``cmodel.is_mcdwith`` 做三角网格求交。

动机：碰撞球拟合在"保守包络"上（每个 link 外扩 1.5–3 mm），自碰撞时两侧外扩叠加，
运动规划实测中 TrajOpt 输出的 37 个"自碰撞"状态经网格复核全部是假阳性，并导致
2/32 个任务规划失败。本基准为 ``self_collision_buffer`` 的负补偿量选值。

    & $py -X utf8 -m curobo_bridge.bench_self_collision --states 4000
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare

JOINT_LIMITS = np.array([[-2.4, 2.4], [0.0, 3.2], [0.0, 4.0],
                         [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]])


def checked_pairs(robot: dict):
    kin = robot["kinematics"]
    ign = kin["self_collision_ignore"]
    links = kin["collision_link_names"]
    return [(a, b) for a, b in itertools.combinations(links, 2)
            if b not in ign.get(a, []) and a not in ign.get(b, [])]


def mesh_self_collision(arm, parts, pairs, qs):
    out = np.zeros(len(qs), bool)
    for i, q in enumerate(qs):
        arm.goto_given_conf(q)
        out[i] = any(parts[a].cmodel.is_mcdwith([parts[b].cmodel]) for a, b in pairs)
    arm.goto_given_conf(np.zeros(6))
    return out


def with_self_padding(robot: dict, scene_buffer: float, self_pad: float) -> dict:
    from .curobo_models import with_buffer
    out = with_buffer(robot, scene_buffer)
    kin = out["kinematics"]
    kin["self_collision_buffer"] = {l: v + float(self_pad)
                                    for l, v in kin["self_collision_buffer"].items()}
    return out


def main(argv=None) -> None:
    prepare()
    import torch
    from .bench_collision import confusion
    from .curobo_models import load_robot_dict, make_collision_checker, scene_hits
    from .export_model import make_wrs_arm, wrs_collision_parts
    from .scenes import random_scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=4000)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--scene-buffer", type=float, default=0.002)
    parser.add_argument("--pads", type=float, nargs="+",
                        default=[0.0, -0.002, -0.003, -0.004, -0.005, -0.006])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    arm = make_wrs_arm()
    parts = {n: l for n, _, l in wrs_collision_parts(arm, False)}
    base = load_robot_dict(ASSET_DIR / "panthera_wrsext", args.density)
    pairs = checked_pairs(base)
    qs = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(args.states, 6))
    t0 = time.perf_counter()
    truth = mesh_self_collision(arm, parts, pairs, qs)
    mesh_ms = 1e3 * (time.perf_counter() - t0) / len(qs)
    scene = random_scene(1000).curobo_scene()
    qt = torch.as_tensor(qs, dtype=torch.float32, device="cuda").reshape(-1, 1, 6)
    models = {}
    for pad in args.pads:
        chk = make_collision_checker(with_self_padding(base, args.scene_buffer, pad), scene)
        _, selfc = scene_hits(chk, qt)
        models[f"pad{pad * 1000:g}mm"] = confusion(selfc.cpu().numpy(), truth)
    report = dict(created=datetime.now().isoformat(timespec="seconds"), args=vars(args),
                  pairs=pairs, positives=int(truth.sum()), states=len(qs),
                  mesh_ms_per_state=mesh_ms, models=models)
    out = OUTPUT_DIR / f"self_collision_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"pairs checked: {pairs}")
    print(f"states={len(qs)} real-mesh self-collisions={truth.sum()} ({mesh_ms:.2f} ms/state)")
    for k, m in models.items():
        print(f"{k:10s} FN={m['fn']:4d} ({m['fn_rate']:.3f})  FP={m['fp']:4d} ({m['fp_rate']:.3f})")
    print("->", out)


if __name__ == "__main__":
    main()
