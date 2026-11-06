"""WRS antipodal grasp planning (Panthera gripper) for a model (scenes/benchmarks/* or
triangular_frame6), saved as
WRS GraspCollection pickles, the same pipeline and parameters as sealp/BLS
(``sealp/examples/grasp/planning_tower.py``).

    & $py -X utf8 -m lazy_cost.gen_grasps [--model rack10] [--only lp_1] [--max-samples 100]
Output: lazy_cost.scene.model_files(model)[1]/<part>_grasps.pickle (part-local frame, metres)

Each grasp: ac_pos / ac_rotmat = TCP (action centre) pose in the part frame, ee_values = jaw
width. ``lazy_cost.scene.load_scene(model)`` loads these files.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="rack10")
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--rotation-deg", type=float, default=15.0)
    ap.add_argument("--normal-angle-deg", type=float, default=175.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from curobo_bridge.runtime import prepare
    prepare()
    import numpy as np
    import wrs.basis.robot_math as rm
    import wrs.grasping.planning.antipodal as gpa
    import wrs.modeling.collision_model as mcm
    import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg

    from .scene import model_files
    stl_dir, out = model_files(args.model)
    out.mkdir(parents=True, exist_ok=True)
    gripper = pg.PantheraGripper()
    report = {}
    for pid in sorted(p.stem for p in stl_dir.glob("*.stl") if p.stem != "base"):
        if args.only and pid != args.only:
            continue
        stl = stl_dir / f"{pid}.stl"
        t0 = perf_counter()
        np.random.seed(args.seed)      # surface sampling of the contact points is random
        random.seed(args.seed)
        obj = mcm.CollisionModel(str(stl))
        gc = gpa.plan_gripper_grasps(gripper, obj, angle_between_contact_normals=rm.radians(args.normal_angle_deg),
                                     rotation_interval=rm.radians(args.rotation_deg), max_samples=args.max_samples,
                                     min_dist_between_sampled_contact_points=0.01, contact_offset=0.01)
        path = out / f"{pid}_grasps.pickle"
        gc.save_to_disk(file_name=str(path))
        widths = [float(g.ee_values) for g in gc]
        down = sum(1 for g in gc if np.asarray(g.ac_rotmat)[2, 2] < -0.3)    # tool z pointing down
        report[pid] = dict(grasps=len(gc), top_down=down, width_min=min(widths, default=None),
                             width_max=max(widths, default=None), seconds=round(perf_counter() - t0, 1))
        print(pid, report[pid], "->", path, flush=True)
    (out / "report.json").write_text(json.dumps(dict(params=vars(args), parts=report), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
