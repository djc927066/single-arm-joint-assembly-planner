"""
Grasp Planning Example — Panthera Gripper
==========================================

Demonstrates antipodal grasp planning on an object mesh using the
**Panthera 双指夹爪** (``PantheraGripper``)。生成 ``GraspCollection``，
保存到 pickle，并在 Panda3D 中可视化部分抓取。

Pick-and-place 流水线第一步：
    1. **抓取规划**（本脚本）
    2. 过滤 / 选择抓取
    3. 在 pick-and-place 规划中使用抓取

Usage::

    python -m sealp.examples.grasp.planning
"""

import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wg3


def plan_grasps(obj_cmodel,
                gripper=None,
                angle_between_contact_normals=None,
                rotation_interval=None,
                max_samples=100,
                min_dist_between_sampled_contact_points=0.01,
                contact_offset=0.01,
                toggle_dbg=False):
    """Plan antipodal grasps on an object using the Panthera gripper.

    Parameters
    ----------
    obj_cmodel : mcm.CollisionModel
        The object to plan grasps on.
    gripper : PantheraGripper or None
        Gripper instance.  If None, a default PantheraGripper is created.
    angle_between_contact_normals : float or None
        Max angle between contact normals (radians).
        Defaults to ``radians(175)``.
    rotation_interval : float or None
        Rotation sampling interval (radians).
        Defaults to ``radians(30)``.
    max_samples : int
        Maximum number of contact point samples.
    min_dist_between_sampled_contact_points : float
        Minimum distance between sampled contact points (meters).
    contact_offset : float
        Contact offset distance (meters).
    toggle_dbg : bool
        Show debug visualization during planning.

    Returns
    -------
    grasp_collection : GraspCollection
        The planned grasps.
    gripper : PantheraGripper
        The gripper instance used.
    """
    if gripper is None:
        gripper = pg.PantheraGripper()
    if angle_between_contact_normals is None:
        angle_between_contact_normals = rm.radians(175)
    if rotation_interval is None:
        rotation_interval = rm.radians(30)

    grasp_collection = gpa.plan_gripper_grasps(
        gripper,
        obj_cmodel,
        angle_between_contact_normals=angle_between_contact_normals,
        rotation_interval=rotation_interval,
        max_samples=max_samples,
        min_dist_between_sampled_contact_points=min_dist_between_sampled_contact_points,
        contact_offset=contact_offset,
        toggle_dbg=toggle_dbg,
    )
    return grasp_collection, gripper


def visualize_grasps(base, obj_cmodel, grasp_collection, gripper,
                     max_show=30, alpha=0.7):
    """Visualize grasps on an object in the Panda3D scene.

    Parameters
    ----------
    base : wd.World
        Panda3D world.
    obj_cmodel : mcm.CollisionModel
        The object model.
    grasp_collection : GraspCollection
        Grasps to visualize.
    gripper : PantheraGripper
        Gripper to render at each grasp pose.
    max_show : int
        Maximum number of grasps to show (for performance).
    alpha : float
        Transparency of gripper models.
    """
    obj_cmodel.attach_to(base)

    for i, grasp in enumerate(grasp_collection):
        if i >= max_show:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat,
                                grasp.ee_values)
        gripper.gen_meshmodel(alpha=alpha).attach_to(base)

_PART_ASSETS = (
    ("yuanchair-part1", "seat"),
    ("yuanchair-part2", "leg"),
)


def _resolve_part_mesh(part_name: str) -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(
        here, "..", "..", "assets", "models", "yuanchair",
        f"{part_name}.stl"))


def main(visualize: bool = True):
    """Run the grasp planning demo for the YuanChair seat + leg parts."""
    base = None
    if visualize:
        base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
        mgm.gen_frame(ax_length=0.3).attach_to(base)

    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(out_dir, exist_ok=True)

    last_obj, last_grasps, last_gripper = None, None, None

    for part_name, role in _PART_ASSETS:
        mesh_path = _resolve_part_mesh(part_name)
        if not os.path.isfile(mesh_path):
            print(f"[WARN] mesh missing for {part_name}: {mesh_path}; skip.")
            continue

        obj_cmodel = mcm.CollisionModel(mesh_path)
        obj_cmodel.rgba = np.array([0.6, 0.5, 0.4, 1.0])

        print("=" * 60)
        print(f"Grasp Planning — Panthera Gripper [{part_name} ({role})]")
        print("=" * 60)
        grasp_collection, gripper = plan_grasps(
            obj_cmodel,
            max_samples=100,
            rotation_interval=rm.radians(30),
        )
        print(f"  Planned {len(grasp_collection)} grasps.")

        save_path = os.path.join(out_dir, f"demo_{part_name}_grasps.pickle")
        grasp_collection.save_to_disk(file_name=save_path)
        print(f"  Saved to: {save_path}")

        last_obj, last_grasps, last_gripper = (
            obj_cmodel, grasp_collection, gripper)

    if visualize and base is not None and last_obj is not None:
        print(f"  Showing first 30 grasps of the last part...")
        visualize_grasps(base, last_obj, last_grasps, last_gripper,
                         max_show=100)
        print("=" * 60)
        print("Press ESC to close.")
        base.run()


if __name__ == "__main__":
    main()
