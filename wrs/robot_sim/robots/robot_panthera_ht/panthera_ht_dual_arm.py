#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dual Panthera-HT robot (no body).

与 ``DualPiperNoBody`` 接口完全对齐：``lft_arm`` / ``rgt_arm`` 各持有一个
``PantheraHTSglArm`` 实例；``use_lft / use_rgt / use_all`` 切换 delegator；
``setup_cc`` 注册自碰对 + 左右臂互碰 + 外部 ext "from" mask。

布局：右臂在左臂 y 轴负方向 ``0.62 m`` 处（``arm_y_offset`` 默认值）。
"""

import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.robot_sim.robots.robot_interface as ri
from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm


class DualPantheraHTNoBody(ri.RobotInterface):
    """无身体的双臂 Panthera-HT 机械臂系统。"""

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 arm_y_offset: float = 0.6,
                 name: str = "dual_panthera_ht",
                 enable_cc: bool = True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        lft_pos = pos + np.array([0.0, 0.0, 0.0])
        rgt_pos = pos + np.array([0.0, -float(arm_y_offset), 0.0])

        self.lft_arm = PantheraHTSglArm(
            pos=lft_pos, rotmat=rotmat.copy(),
            name=name + "_lft", enable_cc=enable_cc)
        self.rgt_arm = PantheraHTSglArm(
            pos=rgt_pos, rotmat=rotmat.copy(),
            name=name + "_rgt", enable_cc=enable_cc)

        self.delegator = self.lft_arm
        self.cc = self.delegator.cc

    # -------------------- delegator 切换 --------------------
    def use_lft(self):
        self.delegator = self.lft_arm
        self.cc = self.lft_arm.cc
        return self.lft_arm

    def use_rgt(self):
        self.delegator = self.rgt_arm
        self.cc = self.rgt_arm.cc
        return self.rgt_arm

    def use_all(self):
        self.delegator = None

    # -------------------- FK / IK / 关节值 --------------------
    def get_jnt_values(self):
        if self.delegator is None:
            return np.concatenate([
                self.lft_arm.get_jnt_values(),
                self.rgt_arm.get_jnt_values(),
            ])
        return self.delegator.get_jnt_values()

    def rand_conf(self):
        return np.concatenate([
            self.lft_arm.rand_conf(),
            self.rgt_arm.rand_conf(),
        ])

    def fk(self, jnt_values=None, toggle_jacobian=False, update=False):
        if self.delegator is None:
            raise AttributeError(
                "Delegator not set. Call use_lft() or use_rgt() first.")
        if jnt_values is None:
            jnt_values = self.delegator.get_jnt_values()
        return self.delegator.fk(
            jnt_values, toggle_jacobian=toggle_jacobian, update=update)

    def backup_state(self):
        if self.delegator is None:
            self.lft_arm.backup_state()
            self.rgt_arm.backup_state()
        else:
            self.delegator.backup_state()

    def restore_state(self):
        if self.delegator is None:
            self.lft_arm.restore_state()
            self.rgt_arm.restore_state()
        else:
            self.delegator.restore_state()

    def goto_given_conf(self, jnt_values):
        if self.delegator is None:
            n = self.lft_arm.n_dof
            self.lft_arm.goto_given_conf(jnt_values[:n])
            self.rgt_arm.goto_given_conf(jnt_values[n:])
        else:
            self.delegator.goto_given_conf(jnt_values)

    # -------------------- 可视化 --------------------
    def gen_meshmodel(self,
                      rgb=None,
                      alpha=1,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        for arm in (self.lft_arm, self.rgt_arm):
            arm.gen_meshmodel(
                rgb=rgb, alpha=alpha,
                toggle_tcp_frame=toggle_tcp_frame,
                toggle_jnt_frames=toggle_jnt_frames,
                toggle_flange_frame=toggle_flange_frame,
                toggle_cdprim=toggle_cdprim,
                toggle_cdmesh=toggle_cdmesh,
            ).attach_to(m_col)
        return m_col

    # -------------------- 手眼点云变换 --------------------
    def transform_point_cloud_handeye(self,
                                      handeye_mat: np.ndarray,
                                      pcd: np.ndarray,
                                      given_conf: np.ndarray = None,
                                      component_name: str = "lft_arm"):
        arm = self.rgt_arm if component_name == "rgt_arm" else self.lft_arm
        if given_conf is None:
            given_conf = arm.get_jnt_values()
        gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf)
        if hasattr(arm, "end_effector") and arm.end_effector is not None:
            try:
                gl_tcp_pos = (
                    gl_tcp_pos - gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos)
            except AttributeError:
                pass
        w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
        w2cam = w2r_mat @ handeye_mat
        return rm.transform_points_by_homomat(w2cam, pcd)

    # -------------------- 自碰 / 互碰 --------------------
    def setup_cc(self):
        """单臂自碰对 + 左右臂互碰 + 外部 ext "from" mask（对齐 PantheraHTSglArm）。

        Panthera-HT 的相邻 mesh 在默认零关节角下有轻微重叠，因此**不**注册同一臂内
        link → link 的自碰对（已在 ``PantheraHTSglArm.setup_cc`` 处理）。这里只做：
            * 双臂之间的互碰检测
            * dynamic 列表 / 外部 ext "from" mask
        """

        def _add_arm_cce(arm):
            return {
                "mlb":  self.cc.add_cce(arm.manipulator.jlc.anchor.lnk_list[0]),
                "ml0":  self.cc.add_cce(arm.manipulator.jlc.jnts[0].lnk),
                "ml1":  self.cc.add_cce(arm.manipulator.jlc.jnts[1].lnk),
                "ml2":  self.cc.add_cce(arm.manipulator.jlc.jnts[2].lnk),
                "ml3":  self.cc.add_cce(arm.manipulator.jlc.jnts[3].lnk),
                "ml4":  self.cc.add_cce(arm.manipulator.jlc.jnts[4].lnk),
                "mlee": self.cc.add_cce(arm.end_effector.jlc.anchor.lnk_list[0]),
                "el0":  self.cc.add_cce(arm.end_effector.jlc.jnts[0].lnk),
                "el1":  self.cc.add_cce(arm.end_effector.jlc.jnts[1].lnk),
            }

        lft = _add_arm_cce(self.lft_arm)
        rgt = _add_arm_cce(self.rgt_arm)

        # ── 左右臂互碰：肘 / 腕 / 末端 ↔ 对侧肘 / 腕 / 末端 ──
        lft_movables = [lft["ml2"], lft["ml3"], lft["ml4"],
                        lft["mlee"], lft["el0"], lft["el1"]]
        rgt_movables = [rgt["ml2"], rgt["ml3"], rgt["ml4"],
                        rgt["mlee"], rgt["el0"], rgt["el1"]]
        self.cc.set_cdpair_by_ids(lft_movables, rgt_movables)

        # ── 外部 ext "from" mask：让对外障碍碰撞检测有效 ──
        ext_from = (
            [lft["ml1"], lft["ml2"], lft["ml3"], lft["ml4"],
             lft["mlee"], lft["el0"], lft["el1"]]
            + [rgt["ml1"], rgt["ml2"], rgt["ml3"], rgt["ml4"],
               rgt["mlee"], rgt["el0"], rgt["el1"]]
        )
        self.cc.enable_extcd_by_id_list(id_list=ext_from, type="from")
        self.cc.enable_innercd_by_id_list(
            id_list=[lft["mlb"], lft["ml0"], lft["ml1"],
                     rgt["mlb"], rgt["ml0"], rgt["ml1"]],
            type="into")

        self.cc.dynamic_into_list = [
            lft["mlb"], lft["ml0"], lft["ml1"], lft["ml2"],
            rgt["mlb"], rgt["ml0"], rgt["ml1"], rgt["ml2"],
        ]
        self.cc.dynamic_ext_list = []

        # 让两个子臂共用同一个 cc，保证 backup/restore 等接口一致
        self.lft_arm.cc = self.cc
        self.rgt_arm.cc = self.cc


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.8, 1.6, 1.2], lookat_pos=[0.2, -0.3, 0.2])
    mgm.gen_frame().attach_to(base)

    robot = DualPantheraHTNoBody(arm_y_offset=0.62, enable_cc=True)
    robot.lft_arm.goto_given_conf(np.zeros(6))
    robot.rgt_arm.goto_given_conf(np.zeros(6))
    print("home self-collided =", robot.is_collided())
    robot.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    base.run()
