"""One problem instance: scene, IK catalogue, stable orders, options, footprints, lazy geometry, cost data.

    inst = Instance.build("triangular_frame6", goal_pos=(0.373, 0.0, 0.0), k=8)

Scene, candidate poses, collision checker and IK catalogue are those of the old mainline
(joint_search.problem.Problem, default settings; the catalogue is read from joint_search/_cache).
Cost data (chain cost, part origin as end-effector proxy, BLS): HOME = TCP at the six-zero
configuration, start[p][x] = origin of p at table pose x, goal[p] = origin of p at its goal.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .geometry import Feasibility, Geo, build_options, footprints, log, mesh_self_checks, stable_orders


@dataclass
class Instance:
    model: str
    pb: object                 # joint_search.problem.Problem
    O: dict                    # options: ids, poses, opt, gl, hb, hix, R (build_options)
    geo: Geo
    feas: Feasibility
    orders: list               # stable orders (tuples of part ids)
    home: np.ndarray
    start: dict                # p -> [n_poses, 3]
    goal: dict                 # p -> [3]
    timing: dict

    @classmethod
    def build(cls, model="triangular_frame6", goal_pos=(0.373, 0.0, 0.0), k=8, self_mesh=True):
        from curobo_bridge.runtime import prepare
        prepare()
        import torch
        from joint_search.problem import Problem
        t0 = perf_counter()
        pb = Problem(model, center=tuple(goal_pos))
        t1 = perf_counter()
        orders = stable_orders(pb)
        keep, self_st = (None, None)
        if self_mesh:                       # gripper vs the handled part, meshes, once per scene
            from pathlib import Path
            keep, self_st = mesh_self_checks(pb, Path(__file__).resolve().parent / "_cache")
            took = "cached" if self_st["cached"] else f"{self_st['seconds']:.0f}s"
            log(f"mesh check gripper vs handled part: goal side {self_st['goal_rejected']}/{self_st['goal_checked']}"
                f" grasps rejected, pick side {self_st['pick_rejected']}/{self_st['pick_checked']}"
                f" (part, orientation, grasp) rejected  ({took})")
        O = build_options(pb, k, keep)
        fp = footprints(pb, O["ids"], O["poses"])
        geo = Geo(pb, O)
        feas = Feasibility(pb, O, geo, fp)
        _, _, tcp = pb.chk.kinematics(torch.zeros((1, 6), dtype=torch.float32, device=pb.chk.device))
        sc = pb.scene
        start = {p: np.stack([sc.pose[p][x][:3, 3] for x in O["poses"][p]]) for p in O["ids"]}
        goal = {p: sc.goal[p][:3, 3].copy() for p in O["ids"]}
        timing = dict(problem_s=t1 - t0, prep_s=perf_counter() - t1, self_mesh=self_st)
        log(f"{model}: {len(O['ids'])} parts, {len(orders)} stable orders; problem {timing['problem_s']:.0f}s,"
            f" options + footprints + row AABBs {timing['prep_s']:.0f}s")
        return cls(model, pb, O, geo, feas, orders, tcp[0].cpu().numpy(), start, goal, timing)

    @property
    def ids(self):
        return tuple(sorted(self.O["ids"]))
