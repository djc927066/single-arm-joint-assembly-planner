"""分级规划器 ``PantheraPlanner`` 在 bench_motion 同一批任务上的端到端表现。

任务生成与 ``bench_motion`` 完全相同（同一随机种子、同一有效性判定），所以可以与
那次 WRS RRT / 纯 cuRobo 的结果逐任务对照。

    & $py -X utf8 -m curobo_bridge.bench_hybrid --scenes 4 --tasks 8
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime

import numpy as np

from .runtime import ASSET_DIR, OUTPUT_DIR, prepare


def main(argv=None) -> None:
    prepare()
    import torch
    from .bench_motion import WrsOracle, make_tasks, path_metrics
    from .curobo_models import load_robot_dict, make_collision_checker, with_buffer
    from .export_model import make_wrs_arm, wrs_collision_parts
    from .planner import PantheraPlanner, densify
    from .scenes import random_scene

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    arm = make_wrs_arm()
    oracle = WrsOracle(arm, [l for _, _, l in wrs_collision_parts(arm, False)])
    robot = with_buffer(load_robot_dict(ASSET_DIR / "panthera_wrsext", 1.0), 0.002)
    scenes = [random_scene(2000 + s) for s in range(args.scenes)]
    tasks = []
    for si, sc in enumerate(scenes):
        chk = make_collision_checker(robot, sc.curobo_scene())

        def cu_valid(q, _c=chk):
            return bool(_c.validate(torch.as_tensor(q, dtype=torch.float32, device="cuda")
                                    .reshape(1, 1, 6)).all())
        for t in make_tasks(sc, oracle, cu_valid, rng, args.tasks):
            t["scene"] = si
            tasks.append(t)
        del chk
    torch.cuda.empty_cache()
    print(f"{len(tasks)} tasks", flush=True)

    planner = PantheraPlanner(arm=arm)
    report = {}

    # ---- single-problem mode ------------------------------------------------
    single = []
    for si, sc in enumerate(scenes):
        planner.set_scene(sc)
        for i, t in enumerate(tasks):
            if t["scene"] == si:
                single.append((i, planner.plan(t["start"], t["goal"])))
    single.sort(key=lambda x: x[0])
    # ---- batch mode (one multi-env cuRobo call + fallbacks) ------------------
    batch = planner.plan_batch([(scenes[t["scene"]], t["start"], t["goal"]) for t in tasks])

    for name, results in (("single", [r for _, r in single]), ("multienv_batch", batch)):
        ok = [r for r in results if r.success]
        times = [path_metrics(densify(r.path, 0.01))["velocity_bound_time_s"] for r in ok]
        report[name] = dict(
            tasks=len(results), success=len(ok),
            source=dict(Counter(r.source for r in ok)),
            rejected_by_verification=sum(any("rejected" in o for _, o, _ in r.tiers) for r in results),
            plan_s_total=float(sum(r.plan_s for r in results)),
            plan_s_median=float(np.median([r.plan_s for r in results])),
            verify_s_total=float(sum(r.verify_s for r in results)),
            velocity_bound_time_s_median=float(np.median(times)) if times else None,
            per_task=[dict(task=i, success=r.success, source=r.source, plan_s=r.plan_s,
                           verify_s=r.verify_s, tiers=r.tiers) for i, r in enumerate(results)])
    report["setup_s"] = planner.setup_s
    out = OUTPUT_DIR / f"hybrid_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("single", "multienv_batch"):
        r = {k: v for k, v in report[name].items() if k != "per_task"}
        print(name, json.dumps(r))
    print("setup", planner.setup_s)
    print("->", out)


if __name__ == "__main__":
    main()
