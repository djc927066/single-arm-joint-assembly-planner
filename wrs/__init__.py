"""WRS convenience aliases, loaded on first access.

The upstream package imported visualization, MuJoCo, many robot models and
motion planners whenever any ``wrs.*`` submodule was imported. The joint
planner only needs the requested Panthera modules, so aliases are kept API
compatible but resolved lazily.
"""
from __future__ import annotations

from importlib import import_module


_ALIASES = {
    "mt": "wrs.modeling.mesh_tools",
    "rm": "wrs.basis.robot_math",
    "mcm": "wrs.modeling.collision_model",
    "mgm": "wrs.modeling.geometric_model",
    "wd": "wrs.visualization.panda.world",
    "cbt": "wrs.robot_sim.robots.cobotta.cobotta",
    "x6wg2": "wrs.robot_sim.robots.xarmlite6_wg.x6wg2",
    "ur3d": "wrs.robot_sim.robots.ur3_dual.ur3_dual",
    "ur3ed": "wrs.robot_sim.robots.ur3e_dual.ur3e_dual",
    "ko2fg": "wrs.robot_sim.robots.khi.khi_or2fg7",
    "ym": "wrs.robot_sim.robots.yumi.yumi",
    "fr3": "wrs.robot_sim.robots.franka_research_3.franka_research_3",
    "rtq85": "wrs.robot_sim.end_effectors.grippers.robotiq85.robotiq85",
    "rtqhe": "wrs.robot_sim.end_effectors.grippers.robotiqhe.robotiqhe",
    "yumi_g": "wrs.robot_sim.end_effectors.grippers.yumi_gripper.yumi_gripper",
    "cbt_g": "wrs.robot_sim.end_effectors.grippers.cobotta_gripper.cobotta_gripper",
    "gg": "wrs.grasping.grasp",
    "gpa": "wrs.grasping.planning.antipodal",
    "gag": "wrs.grasping.annotation.gripping",
    "mmd": "wrs.motion.motion_data",
    "rrtc": "wrs.motion.probabilistic.rrt_connect",
    "mip": "wrs.motion.primitives.interpolated",
    "adp": "wrs.motion.primitives.approach_depart_planner",
    "ppp": "wrs.manipulation.pick_place",
    "toppra": "wrs.motion.trajectory.totg",
    "fsp": "wrs.manipulation.placement.flatsurface",
    "hop": "wrs.manipulation.placement.handover",
    "fsreg": "wrs.manipulation.flatsurface_regrasp",
    "horeg": "wrs.manipulation.handover_regrasp",
}

__all__ = list(_ALIASES)


def __getattr__(name: str):
    try:
        module_name = _ALIASES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = import_module(module_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
