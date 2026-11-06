"""cuRobo 单次规划时间花在哪、批量能快多少：MotionPlanner vs BatchMotionPlanner vs WRS RRT。

    & $py -X utf8 -m curobo_bridge.bench_batch_scaling --tasks 64

同一张桌面场景（``scenes.random_scene(2000)``），64 个关节空间点到点任务（起终点对
cuRobo 与 WRS 网格都无碰撞，一半的直线插值会撞，需要绕行）。

测量项（均为 GPU 常驻、预热后的墙钟时间）：

* ``mp``         单个 ``MotionPlanner.plan_cspace``（默认：最多 5 次尝试、第 2 次起 PRM 种子、
                 每次 TrajOpt 4 轮 finetune），同时记录 cuRobo 自报的 GPU 优化耗时。
* ``trajopt1``   只调一次 ``TrajOptSolver.solve_cspace``（finetune 1 轮、无重试、无图搜索）。
* ``batch_B``    ``BatchMotionPlanner``（共享世界）批量 B = 1/4/16/64，max_attempts=1 与 3。
* ``menv_B``     ``BatchMotionPlanner(multi_env=True)``，B = 16/64，每个问题一个世界
                 （这里 64 个世界内容相同，只为测多环境开销）。
* ``rrt``        WRS RRTConnect（ext 0.3 / smoothing 150 / 10 s），逐任务串行。

所有 cuRobo 成功路径都用 WRS 网格复核（报告拒绝数），但复核时间不计入规划时间。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare


def main(argv=None) -> None:
    prepare()
    import torch
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState

    from .bench_motion import WrsOracle, densify
    from .curobo_models import load_robot_dict, make_collision_checker, with_buffer
    from .export_model import make_wrs_arm, wrs_collision_parts
    from .scenes import random_scene

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=int, default=64)
    ap.add_argument("--rrt-tasks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    robot = with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", 1.0), 0.002)
    names = robot["kinematics"]["cspace"]["joint_names"]
    scene = random_scene(2000)
    sdict = scene.curobo_scene_dict()
    wobs = scene.wrs_obstacles()
    arm = make_wrs_arm()
    oracle = WrsOracle(arm, [l for _, _, l in wrs_collision_parts(arm, False)])

    # ---- tasks: valid endpoints, half need a detour ---------------------------
    chk = make_collision_checker(robot, scene.curobo_scene())
    lim = np.array([[-2.4, 2.4], [0, 3.2], [0, 4.0], [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]])
    pool = rng.uniform(lim[:, 0], lim[:, 1], size=(20000, 6))
    tcp = np.array([arm.fk(q)[0] for q in pool[:6000]])
    pool = pool[:6000][(tcp[:, 0] > 0.1) & (tcp[:, 0] < 0.62) & (np.abs(tcp[:, 1]) < 0.3)
                       & (tcp[:, 2] > 0.03) & (tcp[:, 2] < 0.4)]
    ok = chk.validate(torch.as_tensor(pool, dtype=torch.float32, device="cuda")
                      .reshape(-1, 1, 6)).reshape(-1).cpu().numpy()
    pool = pool[ok]
    pool = np.array([q for q in pool[:400] if not oracle.mesh(q, wobs)])
    hard, easy = [], []
    while len(hard) + len(easy) < args.tasks:
        a, b = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        if np.max(np.abs(a - b)) < 0.5:
            continue
        straight = densify(np.stack([a, b]), 0.03)
        d = straight.reshape(-1, 1, 6)
        blocked = not bool(chk.validate(torch.as_tensor(d, dtype=torch.float32, device="cuda")).all())
        (hard if blocked else easy).append((a, b)) if len(hard if blocked else easy) < args.tasks // 2 else None
    tasks = hard + easy
    starts = np.stack([t[0] for t in tasks])
    goals = np.stack([t[1] for t in tasks])
    del chk
    torch.cuda.empty_cache()
    print(f"{len(tasks)} tasks ({len(hard)} need a detour)", flush=True)

    def js(q):
        return JointState.from_position(torch.as_tensor(np.atleast_2d(q), dtype=torch.float32,
                                                        device="cuda"), joint_names=names)

    def audit(path):
        return all(not oracle.mesh(q, wobs) for q in densify(np.asarray(path), 0.02))

    report = dict(created=datetime.now().isoformat(timespec="seconds"), tasks=len(tasks))

    # ---- single MotionPlanner ------------------------------------------------------
    mp = MotionPlanner(MotionPlannerCfg.create(robot=robot, scene_model=sdict,
                                               collision_cache={"mesh": 16, "cuboid": 16}))
    t0 = time.perf_counter()
    mp.warmup(enable_graph=True, num_warmup_iterations=3)
    torch.cuda.synchronize()
    report["mp_setup_s"] = time.perf_counter() - t0
    wall, gpu, succ, rej = [], [], 0, 0
    for a, b in tasks:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = mp.plan_cspace(js(b), js(a))
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
        gpu.append(float(r.solve_time) if r is not None else 0.0)
        if r is not None and bool(r.success.any()):
            succ += 1
            rej += not audit(r.get_interpolated_plan().position.reshape(-1, 6).cpu().numpy())
    report["mp"] = dict(success=succ, wall_median=float(np.median(wall)), wall_mean=float(np.mean(wall)),
                        wall_total=float(np.sum(wall)), gpu_opt_median=float(np.median(gpu)),
                        audit_rejected=rej)
    print("mp", report["mp"], flush=True)

    # ---- one TrajOpt call, no retries / graph ----------------------------------------------
    wall, succ = [], 0
    for a, b in tasks:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = mp.trajopt_solver.solve_cspace(js(b), js(a), finetune_attempts=1)
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
        succ += int(bool(r.success.any()))
    report["trajopt1"] = dict(success=succ, wall_median=float(np.median(wall)),
                              wall_total=float(np.sum(wall)))
    print("trajopt1", report["trajopt1"], flush=True)
    del mp
    torch.cuda.empty_cache()

    # ---- BatchMotionPlanner, shared world ------------------------------------------------------
    for multi in (False, True):
        sizes = (1, 4, 16, 64) if not multi else (16, 64)
        for B in sizes:
            if B > len(tasks):
                continue
            cfg = MotionPlannerCfg.create(
                robot=robot, scene_model=([sdict] * B) if multi else sdict,
                collision_cache={"mesh": 16, "cuboid": 16}, max_batch_size=B, multi_env=multi)
            bp = BatchMotionPlanner(cfg)
            t0 = time.perf_counter()
            bp.warmup()
            torch.cuda.synchronize()
            setup = time.perf_counter() - t0
            for attempts in (1, 3):
                wall, succ, rej = 0.0, 0, 0
                for s in range(0, len(tasks), B):
                    a_, b_ = starts[s:s + B], goals[s:s + B]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    r = bp.plan_cspace(js(b_), js(a_), max_attempts=attempts)
                    torch.cuda.synchronize()
                    wall += time.perf_counter() - t0
                    if r is None:
                        continue
                    ok = r.success.reshape(B, -1).any(-1).cpu().numpy()
                    traj = r.interpolated_trajectory.position.reshape(B, -1, 6)
                    last = r.interpolated_last_tstep.reshape(B, -1)[:, 0].cpu().numpy()
                    for i in np.flatnonzero(ok):
                        succ += 1
                        if s == 0:   # audit the first batch only (audit is slow on CPU)
                            rej += not audit(traj[i, : int(last[i]) + 1].cpu().numpy())
                key = f"{'menv' if multi else 'batch'}_{B}_att{attempts}"
                report[key] = dict(success=succ, wall_total=wall, per_task=wall / len(tasks),
                                   setup_s=setup, audit_rejected_first_batch=rej)
                print(key, report[key], flush=True)
            del bp
            torch.cuda.empty_cache()

    # ---- WRS RRT reference -------------------------------------------------------------------
    import random
    import wrs.motion.probabilistic.rrt_connect as rrtc
    wall, succ, rej = [], 0, 0
    for a, b in tasks[: args.rrt_tasks // 2] + tasks[len(hard): len(hard) + args.rrt_tasks // 2]:
        np.random.seed(0)
        random.seed(0)
        t0 = time.perf_counter()
        md = rrtc.RRTConnect(arm).plan(start_conf=a, goal_conf=b, obstacle_list=wobs,
                                        start_obstacle_list=[], goal_obstacle_list=[],
                                        ext_dist=0.3, max_time=10.0, smoothing_n_iter=150)
        wall.append(time.perf_counter() - t0)
        if md is not None:
            succ += 1
            rej += not audit(np.asarray(md.jv_list))
    arm.goto_given_conf(np.zeros(6))
    report["rrt"] = dict(tasks=len(wall), success=succ, wall_median=float(np.median(wall)),
                         per_task=float(np.mean(wall)), audit_rejected=rej)
    print("rrt", report["rrt"], flush=True)
    out = OUTPUT_DIR / f"batch_scaling_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
