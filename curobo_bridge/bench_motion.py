"""cuRobo MotionPlanner / BatchMotionPlanner / TrajOpt  vs  WRS RRT-Connect。

运行（cuRobo venv，仓库根目录）::

    & E:\\Projects\\wrs-sealp-assemble\\dual_sealp\\cuRobotest\\.venv\\Scripts\\python.exe `
        -X utf8 -m curobo_bridge.bench_motion --scenes 4 --tasks 8

任务：同一批桌面场景（``scenes.random_scene``）里的关节空间点到点规划。起终点对
WRS 图元、WRS 网格、cuRobo 模型三者都无碰撞，且 TCP 位于桌面上方工作区；其中
"hard" 任务的直线插值会被 WRS 网格检测判碰，必须绕行。一部分任务以 HOME（全零，
sealp 所有脚本的 HOME_JV）为起点。

规划器：

* ``wrs_rrt_0.3``  WRS RRTConnect，ext_dist=0.30 / smoothing 150 / 10 s（sealp L3 默认）
* ``wrs_rrt_0.1``  同上 ext_dist=0.10（sealp direct_transport 默认步长）
* ``curobo_mp``    ``MotionPlanner.plan_cspace``（IK 不需要；TrajOpt + PRM 种子 + 重试）
* ``curobo_batch`` ``BatchMotionPlanner(multi_env=False)``，每个场景一批
* ``curobo_multienv`` ``BatchMotionPlanner(multi_env=True)``，所有场景的任务一次求解，
  每个 batch 元素一个碰撞世界

**所有返回路径统一复核**：关节空间按 ``--check-step`` 加密后，逐状态用 WRS 网格（ODE）
和 WRS 图元检测。RRT 只在 ext_dist 采样点上检测，cuRobo 用球近似，二者都可能在
真实网格上擦碰——这正是要量化的"跟 WRS 碰撞检测差多少"。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare

JOINT_LIMITS = np.array([[-2.4, 2.4], [0.0, 3.2], [0.0, 4.0],
                         [-1.6, 1.6], [-1.7, 1.7], [-2.5, 2.5]])
VMAX = np.array([4.2, 5.0, 5.0, 4.2, 3.7, 3.7])


def densify(path: np.ndarray, step: float) -> np.ndarray:
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + t * (b - a))
    return np.asarray(out)


def path_metrics(path: np.ndarray) -> dict:
    d = np.diff(path, axis=0)
    return dict(waypoints=int(len(path)),
                joint_length_rad=float(np.sum(np.linalg.norm(d, axis=1))),
                velocity_bound_time_s=float(np.sum(np.max(np.abs(d) / VMAX, axis=1))))


class WrsOracle:
    def __init__(self, arm, ext_links):
        self.arm, self.ext = arm, ext_links

    def prim(self, q, wobs):
        self.arm.goto_given_conf(q)
        return bool(self.arm.is_collided(obstacle_list=wobs))

    def mesh(self, q, wobs):
        self.arm.goto_given_conf(q)
        return any(l.cmodel.is_mcdwith(wobs) for l in self.ext)

    def audit(self, path, wobs, step):
        dense = densify(np.asarray(path, float), step)
        mesh_hits = [i for i, q in enumerate(dense) if self.mesh(q, wobs)]
        prim_hits = [i for i, q in enumerate(dense) if self.prim(q, wobs)]
        self.arm.goto_given_conf(np.zeros(6))
        return dict(checked_states=int(len(dense)),
                    mesh_collision_states=len(mesh_hits),
                    prim_collision_states=len(prim_hits),
                    mesh_collision_free=not mesh_hits,
                    prim_collision_free=not prim_hits)


def make_tasks(scene, oracle, cu_valid, rng, n_tasks, hard_ratio=0.75):
    """Pairs of configurations valid under WRS prim + WRS mesh + cuRobo."""
    from .export_model import fk_frames
    wobs = scene.wrs_obstacles()
    pool = []
    home = np.zeros(6)
    tries = 0
    while len(pool) < 4 * n_tasks and tries < 20000:
        tries += 1
        q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        tcp = oracle.arm.fk(q)[0]
        if not (0.10 < tcp[0] < 0.62 and abs(tcp[1]) < 0.30 and 0.03 < tcp[2] < 0.40):
            continue
        if oracle.prim(q, wobs) or oracle.mesh(q, wobs) or not cu_valid(q):
            continue
        pool.append(q)
    pool = np.asarray(pool)
    home_ok = not (oracle.prim(home, wobs) or oracle.mesh(home, wobs)) and cu_valid(home)
    n_hard = int(round(hard_ratio * n_tasks))
    hard_tasks, easy_tasks = [], []
    attempts = 0
    while (len(hard_tasks) < n_hard or len(easy_tasks) < n_tasks - n_hard) and attempts < 3000:
        attempts += 1
        use_home = home_ok and rng.random() < 0.35
        a = home if use_home else pool[rng.integers(len(pool))]
        b = pool[rng.integers(len(pool))]
        if np.max(np.abs(a - b)) < 0.5:
            continue
        straight = densify(np.stack([a, b]), 0.02)
        hard = any(oracle.mesh(q, wobs) for q in straight)
        bucket, quota = (hard_tasks, n_hard) if hard else (easy_tasks, n_tasks - n_hard)
        if len(bucket) < quota:
            bucket.append(dict(start=a, goal=b, hard=bool(hard), from_home=bool(use_home)))
    oracle.arm.goto_given_conf(np.zeros(6))
    return hard_tasks + easy_tasks


def run_wrs_rrt(arm, tasks, wobs, ext_dist, max_time, smoothing):
    import wrs.motion.probabilistic.rrt_connect as rrtc
    planner = rrtc.RRTConnect(arm)
    out = []
    for t in tasks:
        np.random.seed(0)
        import random
        random.seed(0)
        t0 = time.perf_counter()
        md = planner.plan(start_conf=t["start"], goal_conf=t["goal"], obstacle_list=wobs,
                          ext_dist=ext_dist, max_time=max_time, smoothing_n_iter=smoothing)
        dt = time.perf_counter() - t0
        path = None if md is None else np.asarray(md.jv_list, float)
        out.append(dict(success=path is not None, plan_s=dt, path=path))
    arm.goto_given_conf(np.zeros(6))
    return out


def _js(position, joint_names):
    import torch
    from curobo.types import JointState
    return JointState.from_position(torch.as_tensor(np.asarray(position), dtype=torch.float32,
                                                    device="cuda"), joint_names=joint_names)


def _batch_paths(result, n):
    """Per-problem dense joint path from a TrajOpt result (list of arrays or None)."""
    succ = result.success.reshape(n, -1).any(dim=-1).cpu().numpy()
    traj = result.interpolated_trajectory.position
    traj = traj.reshape(n, -1, traj.shape[-2], traj.shape[-1]) if traj.ndim == 4 else \
        traj.reshape(n, traj.shape[-2], traj.shape[-1])[:, None]
    last = result.interpolated_last_tstep.reshape(n, -1).cpu().numpy()
    out = []
    for i in range(n):
        if not succ[i]:
            out.append(None)
            continue
        k = int(last[i, 0])
        out.append(traj[i, 0, : k + 1].cpu().numpy())
    return out


def main(argv=None) -> None:
    prepare()
    import torch
    from curobo.batch_motion_planner import BatchMotionPlanner
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    from .curobo_models import load_robot_dict, make_collision_checker, with_buffer
    from .export_model import make_wrs_arm, wrs_collision_parts
    from .scenes import random_scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="panthera_wrsext")
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--buffer", type=float, default=0.002)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--rrt-time", type=float, default=10.0)
    parser.add_argument("--check-step", type=float, default=0.01)
    parser.add_argument("--skip-rrt01", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    arm = make_wrs_arm()
    oracle = WrsOracle(arm, [l for _, _, l in wrs_collision_parts(arm, False)])
    robot = with_buffer(load_robot_dict(ASSET_DIR / args.model, args.density), args.buffer)
    joint_names = robot["kinematics"]["cspace"]["joint_names"]

    scenes = [random_scene(2000 + s) for s in range(args.scenes)]
    scene_dicts = [s.curobo_scene_dict() for s in scenes]
    wobs_list = [s.wrs_obstacles() for s in scenes]

    # ---- tasks -------------------------------------------------------------
    all_tasks = []
    for si, sc in enumerate(scenes):
        chk = make_collision_checker(robot, sc.curobo_scene())

        def cu_valid(q, _chk=chk):
            qt = torch.as_tensor(q, dtype=torch.float32, device="cuda").reshape(1, 1, 6)
            return bool(_chk.validate(qt).all().item())
        tasks = make_tasks(sc, oracle, cu_valid, rng, args.tasks)
        for t in tasks:
            t["scene"] = si
        all_tasks.extend(tasks)
        del chk
        print(f"scene {si}: {len(tasks)} tasks ({sum(t['hard'] for t in tasks)} hard)", flush=True)
    torch.cuda.empty_cache()
    n = len(all_tasks)
    results = {k: [None] * n for k in ("wrs_rrt_0.3", "wrs_rrt_0.1", "curobo_mp",
                                       "curobo_batch", "curobo_multienv")}
    setup = {}

    # ---- WRS RRT -----------------------------------------------------------
    variants = [("wrs_rrt_0.3", 0.30)] + ([] if args.skip_rrt01 else [("wrs_rrt_0.1", 0.10)])
    for name, ext in variants:
        for si in range(len(scenes)):
            idx = [i for i, t in enumerate(all_tasks) if t["scene"] == si]
            res = run_wrs_rrt(arm, [all_tasks[i] for i in idx], wobs_list[si], ext,
                              args.rrt_time, 150)
            for i, r in zip(idx, res):
                results[name][i] = r
        print(f"{name}: {sum(r['success'] for r in results[name] if r)}/{n}", flush=True)

    # ---- cuRobo MotionPlanner (single) --------------------------------------
    t0 = time.perf_counter()
    cfg = MotionPlannerCfg.create(robot=robot, scene_model=scene_dicts[0],
                                  collision_cache={"mesh": 16, "cuboid": 16})
    mp = MotionPlanner(cfg)
    mp.warmup(enable_graph=True, num_warmup_iterations=3)
    torch.cuda.synchronize()
    setup["curobo_mp"] = time.perf_counter() - t0
    from curobo.scene import Scene
    for si in range(len(scenes)):
        mp.update_world(Scene.create(scene_dicts[si]))
        for i, t in enumerate(all_tasks):
            if t["scene"] != si:
                continue
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            r = mp.plan_cspace(_js(t["goal"][None], joint_names), _js(t["start"][None], joint_names))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            ok = r is not None and bool(r.success.any())
            path = r.get_interpolated_plan().position.reshape(-1, 6).cpu().numpy() if ok else None
            results["curobo_mp"][i] = dict(success=ok, plan_s=dt, path=path)
    print(f"curobo_mp: {sum(r['success'] for r in results['curobo_mp'])}/{n}", flush=True)
    mp.destroy() if hasattr(mp, "destroy") else None
    del mp
    torch.cuda.empty_cache()

    # ---- cuRobo BatchMotionPlanner, shared world per scene -------------------
    t0 = time.perf_counter()
    cfg = MotionPlannerCfg.create(robot=robot, scene_model=scene_dicts[0],
                                  collision_cache={"mesh": 16, "cuboid": 16},
                                  max_batch_size=args.tasks)
    bp = BatchMotionPlanner(cfg)
    bp.warmup()
    torch.cuda.synchronize()
    setup["curobo_batch"] = time.perf_counter() - t0
    batch_times = []
    for si in range(len(scenes)):
        idx = [i for i, t in enumerate(all_tasks) if t["scene"] == si]
        bp.update_world(Scene.create(scene_dicts[si]))
        starts = np.stack([all_tasks[i]["start"] for i in idx])
        goals = np.stack([all_tasks[i]["goal"] for i in idx])
        pad = args.tasks - len(idx)
        if pad:
            starts = np.vstack([starts, np.repeat(starts[-1:], pad, 0)])
            goals = np.vstack([goals, np.repeat(goals[-1:], pad, 0)])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        r = bp.plan_cspace(_js(goals, joint_names), _js(starts, joint_names), max_attempts=3)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        batch_times.append(dt)
        paths = _batch_paths(r, args.tasks) if r is not None else [None] * args.tasks
        for k, i in enumerate(idx):
            results["curobo_batch"][i] = dict(success=paths[k] is not None,
                                              plan_s=dt / len(idx), path=paths[k])
    print(f"curobo_batch: {sum(r['success'] for r in results['curobo_batch'])}/{n}", flush=True)
    del bp
    torch.cuda.empty_cache()

    # ---- cuRobo BatchMotionPlanner, one world per problem --------------------
    t0 = time.perf_counter()
    cfg = MotionPlannerCfg.create(robot=robot,
                                  scene_model=[scene_dicts[t["scene"]] for t in all_tasks],
                                  collision_cache={"mesh": 16, "cuboid": 16},
                                  max_batch_size=n, multi_env=True)
    mep = BatchMotionPlanner(cfg)
    mep.warmup()
    torch.cuda.synchronize()
    setup["curobo_multienv"] = time.perf_counter() - t0
    starts = np.stack([t["start"] for t in all_tasks])
    goals = np.stack([t["goal"] for t in all_tasks])
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    r = mep.plan_cspace(_js(goals, joint_names), _js(starts, joint_names), max_attempts=3)
    torch.cuda.synchronize()
    multienv_s = time.perf_counter() - t1
    paths = _batch_paths(r, n) if r is not None else [None] * n
    for i in range(n):
        results["curobo_multienv"][i] = dict(success=paths[i] is not None,
                                             plan_s=multienv_s / n, path=paths[i])
    print(f"curobo_multienv: {sum(r['success'] for r in results['curobo_multienv'])}/{n} "
          f"in {multienv_s:.2f}s", flush=True)
    del mep
    torch.cuda.empty_cache()

    # ---- audit every path with WRS -----------------------------------------
    summary = {}
    rows = []
    for name, res in results.items():
        if res[0] is None:
            continue
        agg = dict(success=0, mesh_free=0, prim_free=0, plan_s=[], time_s=[], length=[],
                   hard_success=0, hard=0)
        for i, (t, r) in enumerate(zip(all_tasks, res)):
            row = dict(planner=name, task=i, scene=t["scene"], hard=t["hard"],
                       from_home=t["from_home"], success=r["success"], plan_s=r["plan_s"])
            agg["plan_s"].append(r["plan_s"])
            agg["hard"] += int(t["hard"])
            if r["success"]:
                a = oracle.audit(r["path"], wobs_list[t["scene"]], args.check_step)
                m = path_metrics(densify(r["path"], 0.01))
                row.update(a)
                row.update(m)
                agg["success"] += 1
                agg["hard_success"] += int(t["hard"])
                agg["mesh_free"] += int(a["mesh_collision_free"])
                agg["prim_free"] += int(a["prim_collision_free"])
                agg["time_s"].append(m["velocity_bound_time_s"])
                agg["length"].append(m["joint_length_rad"])
            rows.append(row)
        summary[name] = dict(
            tasks=n, success=agg["success"], hard_success=f"{agg['hard_success']}/{agg['hard']}",
            mesh_collision_free_paths=agg["mesh_free"], prim_collision_free_paths=agg["prim_free"],
            plan_s_median=float(np.median(agg["plan_s"])), plan_s_mean=float(np.mean(agg["plan_s"])),
            plan_s_total=float(np.sum(agg["plan_s"])),
            velocity_bound_time_s_median=float(np.median(agg["time_s"])) if agg["time_s"] else None,
            joint_length_rad_median=float(np.median(agg["length"])) if agg["length"] else None)
    # paired comparison on tasks solved by both RRT(0.3) and cuRobo MotionPlanner
    both = [i for i in range(n) if results["wrs_rrt_0.3"][i]["success"]
            and results["curobo_mp"][i]["success"]]
    paired = None
    if both:
        tr = [path_metrics(densify(results["wrs_rrt_0.3"][i]["path"], 0.01))["velocity_bound_time_s"]
              for i in both]
        tc = [path_metrics(densify(results["curobo_mp"][i]["path"], 0.01))["velocity_bound_time_s"]
              for i in both]
        paired = dict(tasks=len(both), rrt_time_median=float(np.median(tr)),
                      curobo_time_median=float(np.median(tc)),
                      ratio_median=float(np.median(np.asarray(tc) / np.asarray(tr))))
    report = dict(created=datetime.now().isoformat(timespec="seconds"), args=vars(args),
                  sphere_buffer_m=args.buffer, setup_s=setup, batch_scene_s=batch_times,
                  multienv_total_s=multienv_s, summary=summary, paired_rrt03_vs_mp=paired,
                  rows=rows)
    out_dir = OUTPUT_DIR / f"motion_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "paths.npz", **{
        f"{k}_{i}": r["path"] for k, res in results.items() if res[0] is not None
        for i, r in enumerate(res) if r["success"]})
    print(json.dumps(dict(setup_s=setup, multienv_total_s=multienv_s, summary=summary,
                          paired=paired), indent=2))
    print("->", out_dir)


if __name__ == "__main__":
    main()
