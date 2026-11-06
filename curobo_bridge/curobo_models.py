"""cuRobo 机器人模型的装配：MorphIt 球模型与"AABB 外接球覆盖"对照模型。

* ``morphit_d<density>``：``build_spheres`` 生成的 YAML（MorphIt 拟合保守包络）。
* ``aabb<cell_mm>``：复刻 ``dual_sealp/cuRobotest/collision_benchmark`` 的做法——把
  WRS 每个外部碰撞件的**局部 AABB + 2 mm**（WRS 默认图元）用 ``cell`` 网格外接球覆盖。
  它逐点包住 WRS 的图元，用来对照"只模仿 WRS 语义"与"贴近真实网格"两条路线。

两类模型共用同一 URDF 运动学、自碰撞忽略表和 cspace，只替换 ``collision_spheres``。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from .export_model import GRIPPER_LINKS, wrs_collision_parts, _frames
from .runtime import ASSET_DIR

WRS_PRIMITIVE_PADDING = 0.002
#: Link pairs whose real meshes touch at the joint-limit rest pose q=0 (the
#: fully folded Panthera home used by every sealp script). Measured: link2/link5
#: mesh gap ~0.1 mm at q=0, sphere overlap 19 mm. Without ignoring the pair the
#: home configuration is "in self-collision" and every plan from home fails.
#: WRS registers no self-collision pairs at all, so this remains stricter than WRS.
HOME_TOUCHING_PAIRS = (("link2", "link5"),)


def load_robot_dict(model_dir: Path, density: float = 1.0) -> dict:
    path = Path(model_dir) / f"panthera_d{density:g}.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    kin = data["kinematics"]
    ignore = kin.setdefault("self_collision_ignore", {})
    for a, b in HOME_TOUCHING_PAIRS:
        ignore.setdefault(a, [])
        if b not in ignore[a]:
            ignore[a].append(b)
    kin["grasp_contact_link_names"] = [l for l in ("finger_left", "finger_right")
                                       if l in kin["collision_link_names"]]
    return data


def box_sphere_cover(bounds: np.ndarray, cell: float) -> np.ndarray:
    """Same algorithm as dual_sealp/cuRobotest/collision_benchmark/geometry.py."""
    ext = bounds[1] - bounds[0]
    counts = np.maximum(1, np.ceil(ext / cell).astype(int))
    widths = ext / counts
    axes = [bounds[0, i] + (np.arange(counts[i]) + 0.5) * widths[i] for i in range(3)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.column_stack([centers, np.full(len(centers), 0.5 * np.linalg.norm(widths))])


def aabb_cover_spheres(arm, *, cell: float = 0.020, include_base: bool = False) -> dict:
    """WRS local-AABB(+2 mm) of every external collision part, covered by spheres."""
    arm.goto_given_conf(np.zeros(6))
    base, frames = _frames(arm)
    out = {}
    for name, parent, lnk in wrs_collision_parts(arm, include_base):
        ref = base if parent < 0 else frames[parent]
        rel = np.linalg.inv(ref) @ np.asarray(lnk.gl_homomat, float)
        b = np.asarray(lnk.cmodel.trm_mesh.bounds, float)
        b = np.array([b[0] - WRS_PRIMITIVE_PADDING, b[1] + WRS_PRIMITIVE_PADDING])
        s = box_sphere_cover(b, cell)
        s[:, :3] = s[:, :3] @ rel[:3, :3].T + rel[:3, 3]
        out[name] = [{"center": r[:3].tolist(), "radius": float(r[3])} for r in s]
    return out


def with_spheres(robot: dict, spheres: dict) -> dict:
    out = copy.deepcopy(robot)
    out["kinematics"]["collision_spheres"] = spheres
    return out


def with_buffer(robot: dict, buffer_m: float) -> dict:
    """Inflate spheres by ``buffer_m`` for robot-scene checks only.

    cuRobo (commit 8e734f3) ``KinematicsLoader._build_collision_model`` computes a
    compensated per-link self-collision padding (``self_collision_buffer[j] -=
    offset_radius``) into a local copy but then passes the *uncompensated*
    ``self.self_collision_buffer`` to ``SelfCollisionKinematicsCfg``. So the scene
    buffer also inflates self-collision (at q=0 it makes link2/link4, 10 mm apart,
    "collide"). We apply the intended compensation explicitly in the config.
    """
    out = copy.deepcopy(robot)
    kin = out["kinematics"]
    kin["collision_sphere_buffer"] = float(buffer_m)
    pad = dict(kin.get("self_collision_buffer") or {})
    for link in kin["collision_link_names"]:
        pad[link] = float(robot["kinematics"].get("self_collision_buffer", {}).get(link, 0.0)) \
            - float(buffer_m)
    kin["self_collision_buffer"] = pad
    return out


def with_attach_link(robot: dict, n_spheres: int = 32, parent: str = "wrs_tcp") -> dict:
    """Add cuRobo's ``attached_object`` extra link (same layout as the bundled franka.yml).

    Its spheres are placeholders until ``AttachmentManager.attach`` fits the held part.
    The parent MUST be the tool frame: ``AttachmentManager.update`` computes the
    object-to-link offset relative to ``tool_frames[0]`` and writes it into the
    attached link, so the two frames have to coincide.
    The held part touches the gripper, so gripper links are ignored for self-collision;
    the arm links are still checked against it.
    """
    out = copy.deepcopy(robot)
    kin = out["kinematics"]
    kin["extra_links"] = {"attached_object": dict(
        fixed_transform=[0, 0, 0, 1, 0, 0, 0], joint_name="attach_joint", joint_type="FIXED",
        link_name="attached_object", parent_link_name=parent)}
    kin["extra_collision_spheres"] = {"attached_object": int(n_spheres)}
    if "attached_object" not in kin["collision_link_names"]:
        kin["collision_link_names"] = list(kin["collision_link_names"]) + ["attached_object"]
    kin["self_collision_buffer"] = dict(kin.get("self_collision_buffer") or {},
                                        attached_object=0.0)
    ign = kin.setdefault("self_collision_ignore", {})
    ign["attached_object"] = ["link6", "gripper_palm", "finger_left", "finger_right"]
    kin["grasp_contact_link_names"] = list(dict.fromkeys(
        list(kin.get("grasp_contact_link_names") or []) + ["attached_object"]))
    return out


def sphere_count(robot: dict) -> int:
    return sum(len(v) for v in robot["kinematics"]["collision_spheres"].values())


def model_variants(arm, model: str = "panthera_wrsext",
                   densities=(0.5, 1.0, 2.0), aabb_cells=(0.020,)) -> dict:
    model_dir = ASSET_DIR / model
    variants = {}
    base = None
    for d in densities:
        if (model_dir / f"panthera_d{d:g}.yml").exists():
            variants[f"morphit_d{d:g}"] = load_robot_dict(model_dir, d)
            base = base or variants[f"morphit_d{d:g}"]
    for cell in aabb_cells:
        variants[f"aabb{cell * 1000:.0f}mm"] = with_spheres(
            base, aabb_cover_spheres(arm, cell=cell, include_base="base_link" in
                                     base["kinematics"]["collision_link_names"]))
    return variants


def make_collision_checker(robot: dict, scene, *, activation: float = 0.0,
                           n_meshes: int = 16, n_cuboids: int = 16, num_envs: int = 1):
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

    cfg = RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot, scene_model=scene, num_envs=num_envs,
        n_meshes=n_meshes, n_cuboids=n_cuboids,
        collision_activation_distance=float(activation),
        self_collision_activation_distance=0.0, max_collision_distance=0.25)
    return RobotCollisionChecker(cfg)


def scene_hits(checker, q_torch):
    """Per-state boolean (scene collision, self collision). ``q_torch`` [B, 1, dof]."""
    import torch

    b, h, _ = q_torch.shape
    checker.setup_batch_tensors(b, h)
    state = checker.get_kinematics(q_torch)
    d_world = checker.get_collision_constraint(state)
    scene = d_world.reshape(b, -1).sum(dim=-1) > 0.0
    selfc = torch.zeros_like(scene)
    if checker.self_collision_cost is not None:
        d_self = checker.get_self_collision(state.robot_spheres)
        selfc = d_self.reshape(b, -1).sum(dim=-1) > 0.0
    return scene, selfc
