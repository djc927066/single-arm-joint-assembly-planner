"""
Created on 2025/12/29
Author: (auto-generated)

OpenArm Definition for WRS Simulator
-----------------------------------

This module defines a seven-degree-of-freedom (7-DoF) model of the OpenArm
robotic arm for use within the WRS simulator.  It closely follows the
structure and coding style of the official PiPER example supplied with
the WRS framework, but replaces all geometric and kinematic parameters
with those extracted from the OpenArm URDF specification.

Each joint’s local translation, orientation, actuation axis and motion
limits are derived directly from the OpenArm URDF.  Individual link
meshes are loaded from the corresponding ``meshes`` directory and a
uniform colour scheme is applied for visualisation.

The resulting ``OpenArm`` class inherits from
``wrs.robot_sim.manipulators.manipulator_interface.ManipulatorInterface``
and can be instantiated and attached to a WRS ``World`` like any other
manipulator.  A TracIK solver may optionally be used for inverse
kinematics when available.
"""

import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm

# Attempt to import the Trac IK solver.  If unavailable, numerical IK
# provided by the joint linkage controller (JLC) will be used instead.

try:
    from trac_ik import TracIK

    is_trac_ik = True
    print("Trac IK module loaded successfully")
except Exception as e:
    print(f"Trac IK module not loaded: {e}")
    is_trac_ik = False


class OpenArm(mi.ManipulatorInterface):
    """Model of the OpenArm 7-DoF arm for the WRS simulator."""

    def __init__(self,
                 pos: np.ndarray = np.zeros(3),
                 rotmat: np.ndarray = np.eye(3),
                 ik_solver: str = 'd',
                 name: str = 'OpenArm',
                 enable_cc: bool = True):
        """
        Initialise the OpenArm.

        :param pos: World position of the arm base.
        :param rotmat: World orientation of the arm base.
        :param ik_solver: Either 'd' (default WRS numerical IK) or 'a'/'j'
                          to select alternative solvers.  When the
                          trac_ik library is available the TracIK solver
                          will be used.
        :param name: Identifier for this manipulator.
        :param enable_cc: Enable self-collision checking if true.
        """
        super().__init__(pos=pos,
                         rotmat=rotmat,
                         home_conf=np.zeros(7),
                         name=name,
                         enable_cc=enable_cc)

        current_file_dir = os.path.dirname(__file__)

        # define a uniform colour for all links
        rgba = np.array([0.6, 0.6, 0.6, 1.0])

        # --------------------------------------------------
        # anchor (link0)
        # --------------------------------------------------
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link0.stl"))
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rgba

        # --------------------------------------------------
        # Joint 1 (link0 -> link1)
        # --------------------------------------------------
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.0625])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[0].motion_range = np.array([-1.396263, 3.490659])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link1.stl"))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0, 0, -0.0625])
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Joint 2 (link1 -> link2)
        #   URDF: origin rpy="1.570796327 0 0" (roll = pi/2)
        # --------------------------------------------------
        self.jlc.jnts[1].loc_pos = np.array([-0.0301, 0.0, 0.06])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.570796327, 0.0, 0.0)  # URDF joint2 origin rpy: roll=pi/2
        self.jlc.jnts[1].loc_motion_ax = np.array([-1, 0, 0.0])
        self.jlc.jnts[1].motion_range = np.array([-1.745329, 1.745329])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link2.stl"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([0.0301, 0.0, -0.1225])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Joint 3 (link2 -> link3)
        # --------------------------------------------------
        self.jlc.jnts[2].loc_pos = np.array([0.0301, 0.0, 0.06625])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-1.570796, 1.570796])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link3.stl"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0, 0, -0.18875])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Joint 4 (link3 -> link4)
        # --------------------------------------------------
        self.jlc.jnts[3].loc_pos = np.array([0.0, 0.0315, 0.15375])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 1, 0.0])
        self.jlc.jnts[3].motion_range = np.array([0.0, 2.443461])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link4.stl"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0, -0.0315, -0.3425])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Joint 5 (link4 -> link5)
        # --------------------------------------------------
        self.jlc.jnts[4].loc_pos = np.array([0.0, -0.0315, 0.0955])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-1.570796, 1.570796])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link5.stl"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([0.0, 0.0, -0.438])
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Joint 6 (link5 -> link6)
        # --------------------------------------------------
        self.jlc.jnts[5].loc_pos = np.array([0.0375, 0.0, 0.1205])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([1.0, 0.0, 0.0])
        self.jlc.jnts[5].motion_range = np.array([-0.785398, 0.785398])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link6.stl"))
        self.jlc.jnts[5].lnk.loc_pos = np.array([-0.0375, 0.0, -0.5585])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Joint 7 (link6 -> link7)
        # --------------------------------------------------
        self.jlc.jnts[6].loc_pos = np.array([-0.0375, 0.0, 0.0])
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[6].loc_motion_ax = np.array([0.0, 1.0, 0.0])
        self.jlc.jnts[6].motion_range = np.array([-1.570796, 1.570796])
        # self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(
        #     os.path.join(current_file_dir, "meshes", "link7.stl"))
        # self.jlc.jnts[6].lnk.loc_pos = np.array([0.0, 0.0, -0.5585])
        # self.jlc.jnts[6].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        # self.jlc.jnts[6].lnk.cmodel.rgba = np.array([0.3, 0.3, 0.3, 1.0])

        # --------------------------------------------------
        # Finalise the joint linkage chain and set up IK solver
        # --------------------------------------------------
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        self.loc_tcp_pos = np.array([0.0, 0.0, 0.0])
        self.loc_tcp_rotmat = np.eye(3)

        if is_trac_ik:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf = os.path.join(directory, "openarm.urdf")
            # base_link and link6 are defined in the URDF; use them for IK
            self._ik_solver = TracIK("link0", "link7", urdf,
                                     timeout=0.002,
                                     solver_type="Distance")
        else:
            self._ik_solver = None

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self) -> None:
        """Configure pairs of links for self-collision checking."""
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        # l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [l4, l5]
        into_list = [lb, l0, l1, l2, l3]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray,
           seed_jnt_values=None, option: str = "empty", toggle_dbg: bool = False):
        """
        Solve the inverse kinematics for the end-effector.
        """
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if is_trac_ik and self._ik_solver is not None:
            anchor_inv_homomat = np.linalg.inv(
                rm.homomat_from_posrot(self.jlc.anchor.pos,
                                       self.jlc.anchor.rotmat))
            tgt_homomat = anchor_inv_homomat.dot(
                rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]
            seed_jnt_values = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(tgt_pos, tgt_rotmat,
                                      seed_jnt_values=seed_jnt_values)
        else:
            return self.jlc.ik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values,
                               toggle_dbg=toggle_dbg)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    arm = OpenArm(enable_cc=True)
    mgm.gen_frame().attach_to(base)
    # bound_lower = -20
    # bound_upper = 20
    # grad = 5
    # plane_normal = np.array([1, 0, 0])
    # tgt_pos = np.array([0.3, 0, 0.3])
    # tgt_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
    # mgm.gen_sphere(pos = tgt_pos,radius=0.02,rgb=np.array([1,0,0])).attach_to(base)
    # joint_value = arm.ik(tgt_pos=tgt_pos, tgt_rotmat= tgt_rotmat)
    # print(joint_value)
    # arm.goto_given_conf(joint_value)
    # arm.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    # for theta in range(bound_lower, bound_upper + 1, grad):
    #     hand_x = np.array([0, 0, -1])
    #     hand_z = plane_normal
    #     hand_y = np.cross(hand_z, hand_x)
    #     tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
    #     tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
    #     mgm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
    #     current_goal_conf = arm.ik(tgt_pos=tgt_pos,
    #                                  tgt_rotmat=tgt_rotmat,
    #                                  seed_jnt_values=None)
    #
    #     print(f"Theta={theta}°: {'Found' if current_goal_conf is not None else 'Failed'}")
    #
    #     if current_goal_conf is not None:
    #         goal_conf = current_goal_conf
    #         break
    #
    #
    # print(1111)
    arm.gen_meshmodel(toggle_jnt_frames=True, alpha=.3).attach_to(base)
    print(arm.is_collided())
    # arm.show_cdprim()
    base.run()
