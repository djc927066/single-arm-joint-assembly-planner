#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/1/5 13:13
# @Author : ZhangXi


import math
import numpy as np
import os
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
from wrs.robot_sim.manipulators.openarm.openarm import OpenArm
from wrs.robot_sim.end_effectors.grippers.openarm_gripper_umi.openarm_gripper_umi import OpenArmGripper


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
    import time
    from wrs import wd, rm, mgm, mcm

    base = wd.World(cam_pos=[1.7, 1, .5], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = OpenSglArm(enable_cc=True)
    robot.change_jaw_width(.05)
    jnt_values = np.array([4.86733859, 0.10280766, 1.36549172, 2.45803769, -0.06656748, -0.06618601, -0.52700847])

    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel(toggle_jnt_frames=False, toggle_tcp_frame=True, alpha=.3).attach_to(base)
    tcp = robot.fk(jnt_values=[-1.41584649, 0.10280766, 1.36549172, 2.45803769, -0.06656748, -0.06618601, -0.52700847])
    print(tcp)
    tgt_pos = tcp[0]
    tgt_rotmat = tcp[1]
    print(tgt_pos)
    print(tgt_rotmat)
    jnt_value = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=jnt_values)
    print(jnt_value)
    robot.goto_given_conf(jnt_value)
    # # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # # robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    # robot.gen_meshmodel(toggle_jnt_frames=False, toggle_tcp_frame=True, alpha=.3).attach_to(base)
    base.run()
