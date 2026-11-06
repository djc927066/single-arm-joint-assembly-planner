#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/01/04 18:25
# @Author : ZhangXi
import math
import numpy as np
import os
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.openarm.openarm import OpenArm
from wrs.robot_sim.end_effectors.grippers.openarm_gripper.openarm_gripper import OpenArmGripper
import wrs.modeling.geometric_model as mgm

class OpenSglArm(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="sgl_openarm", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = OpenArm(pos=self.pos, rotmat=self.rotmat,
                                             name="openarm_" + name, enable_cc=False)
        self.end_effector = OpenArmGripper(pos=self.manipulator.gl_flange_pos,
                                           rotmat=self.manipulator.gl_flange_rotmat, name="=og_" + name)
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        # manipulator
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        # ml6 = self.cc.add_cce(self.manipulator.jlc.jnts[6].lnk)
        from_list = [elb, el0, el1, ml4, ml5]
        into_list = [ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [ml0, ml1, ml2, ml3]

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.basis.robot_math as rm

    base = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    robot = OpenSglArm(enable_cc=True)
    robot.change_jaw_width(0.06)
    # mgm.gen_frame().attach_to(base)
    # tgt_pos = np.array([0.35, -0.4,0.2])
    #
    # bound_lower = -20
    # bound_upper = 20
    # grad = 1
    # goal_conf = None
    # print(f"--- Searching IK solutions for position {tgt_pos} ---")
    # for theta in range(bound_lower, bound_upper + 1, grad):
    #     hand_x = np.array([1, 0, 0])
    #     hand_z = np.array([0, -1, 0])
    #     hand_y = np.cross(hand_z, hand_x)
    #     tgt_rotmat_base = np.array([hand_x, hand_y, hand_z]).T
    #     tgt_rotmat = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rotmat_base
    #     mgm.gen_frame(tgt_pos, tgt_rotmat).attach_to(base)
    #     current_goal_conf = robot.ik(tgt_pos=tgt_pos,
    #                                  tgt_rotmat=tgt_rotmat,
    #                                  )
    #
    #     print(f"Theta={theta}°: {'Found' if current_goal_conf is not None else 'Failed'}")
    #
    #     if current_goal_conf is not None:
    #         goal_conf = current_goal_conf
    #         break
    # if goal_conf is not None:
    #     robot.goto_given_conf(jnt_values=goal_conf)
    #     robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True).attach_to(base)
    # else:
    #     print(1111)
    # robot.gen_meshmodel(toggle_jnt_frames=True,toggle_tcp_frame=True).attach_to(base)
    # robot.show_cdprim()
    # print(robot.is_collided())
    robot.gen_meshmodel().attach_to(base)
    base.run()