#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/10 18:25
# @Author : ZhangXi
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.modeling.collision_model as mcm
import numpy as np
import wrs.robot_sim.robots.robot_interface as ri
from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
import wrs.modeling.model_collection as mmc
import wrs.basis.robot_math as rm


class DualPiperNoBody(ri.RobotInterface):
    """
    双臂系统（无身体），左右两只 Piper 机械臂，继承 RobotInterface
    """
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="dual_piper", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        # 左右臂位置
        lft_pos = pos + np.array([0, 0, 0])
        rgt_pos = pos + np.array([0, -0.597, 0])

        # 创建左右臂实例
        self.lft_arm = PiperSglArm(pos=lft_pos, rotmat=rotmat.copy(), enable_cc=enable_cc)
        self.rgt_arm = PiperSglArm(pos=rgt_pos, rotmat=rotmat.copy(), enable_cc=enable_cc)

        # 默认使用左臂
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
        self.delegator = None  # 使用左右臂组合时自行处理

    # -------------------- FK / IK / 关节值 --------------------
    def get_jnt_values(self):
        if self.delegator is None:
            # 拼接左右臂关节
            return np.concatenate([self.lft_arm.get_jnt_values(), self.rgt_arm.get_jnt_values()])
        return self.delegator.get_jnt_values()

    def rand_conf(self):
        return np.concatenate([self.lft_arm.rand_conf(), self.rgt_arm.rand_conf()])

    def fk(self, jnt_values=None, toggle_jacobian=False, update=False):
        if self.delegator is None:
            raise AttributeError("Delegator not set. Use use_lft() or use_rgt().")
        if jnt_values is None:
            jnt_values = self.delegator.get_jnt_values()
        return self.delegator.fk(jnt_values, toggle_jacobian=toggle_jacobian, update=update)

    def backup_state(self):
        """
        备份左右臂状态，供 RRTConnect 或其他算法使用
        """
        if self.delegator is None:
            # 备份左右臂
            self.lft_arm.backup_state()
            self.rgt_arm.backup_state()
        else:
            # 仅备份当前 delegator
            self.delegator.backup_state()

    def restore_state(self):
        """
        恢复左右臂状态
        """
        if self.delegator is None:
            # 恢复左右臂
            self.lft_arm.restore_state()
            self.rgt_arm.restore_state()
        else:
            # 恢复当前 delegator
            self.delegator.restore_state()

    def goto_given_conf(self, jnt_values):
        if self.delegator is None:
            # 分配左右臂关节值
            n = self.lft_arm.n_dof
            self.lft_arm.goto_given_conf(jnt_values[:n])
            self.rgt_arm.goto_given_conf(jnt_values[n:])
        else:
            self.delegator.goto_given_conf(jnt_values)

    # -------------------- 可视化 --------------------
    def gen_meshmodel(self, rgb=None, alpha=1,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        self.lft_arm.gen_meshmodel(
            rgb=rgb, alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh
        ).attach_to(m_col)
        self.rgt_arm.gen_meshmodel(
            rgb=rgb, alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh
        ).attach_to(m_col)
        return m_col

    # -------------------- 手眼点云变换 --------------------
    def transform_point_cloud_handeye(self, handeye_mat: np.ndarray, pcd: np.ndarray,
                                      given_conf: np.ndarray = None,
                                      component_name: str = 'lft_arm'):
        if component_name == 'rgt_arm':
            arm = self.rgt_arm
        else:
            arm = self.lft_arm

        if given_conf is None:
            given_conf = arm.get_jnt_values()

        gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf)
        if hasattr(arm, 'end_effector') and arm.end_effector is not None:
            try:
                gl_tcp_pos -= gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos
            except AttributeError:
                pass

        w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
        w2cam = w2r_mat @ handeye_mat
        pcd_r = rm.transform_points_by_homomat(w2cam, pcd)
        return pcd_r

    def setup_cc(self):
        """为 DualPiperNoBody 设置左右臂的自碰撞与互碰检测（无 body，至 jnt4）"""
        # === 左臂 ===
        lft_mlb = self.cc.add_cce(self.lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ee = self.cc.add_cce(self.lft_arm.end_effector.jlc.anchor.lnk_list[0])

        # 左臂自碰检测
        from_list = [lft_ml3, lft_ml4, lft_ee]
        into_list = [lft_mlb, lft_ml0, lft_ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # === 右臂 ===
        rgt_mlb = self.cc.add_cce(self.rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ee = self.cc.add_cce(self.rgt_arm.end_effector.jlc.anchor.lnk_list[0])

        # 右臂自碰检测
        from_list = [rgt_ml3, rgt_ml4, rgt_ee]
        into_list = [rgt_mlb, rgt_ml0, rgt_ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # === 左右臂互碰检测 ===
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ee]
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ee]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # 动态部分（实时更新）
        self.cc.dynamic_into_list = [
            lft_mlb, lft_ml0, lft_ml1, lft_ml2,
            rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2
        ]
        self.cc.dynamic_ext_list = []

        # 将 cc 绑定到两臂
        self.lft_arm.cc = self.cc
        self.rgt_arm.cc = self.cc


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.basis.robot_math as rm
    import math

    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    box1 = mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.51]))
    box1.attach_to(base)
    box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2675]))
    box2.attach_to(base)
    obs_list = [box1, box2]
    robot = DualPiperNoBody(enable_cc=True)
    # robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    robot.use_lft()
    tgt_pos_l = np.array([0.3397,-0.2887, 0.0401])
    tgt_rotmat = rm.rotmat_from_euler(2.8813,  0.2080,  2.4237)
    # mgm.gen_frame(pos=tgt_pos_l, rotmat=tgt_rotmat).attach_to(base)
    jnt_values_L = robot.lft_arm.ik(tgt_pos_l, tgt_rotmat)
    # goal_conf = robot.ik(tgt_pos=np.array([0.2747, 0, 0.35]),tgt_rotmat=rm.rotmat_from_axangle([0,1,0],math.pi/2))
    rrtc_planner = rrtc.RRTConnect(robot.lft_arm)
    start_conf = robot.get_jnt_values()
    mot_data = rrtc_planner.plan(start_conf=start_conf,
                                 goal_conf=jnt_values_L,
                                 obstacle_list=obs_list,
                                 ext_dist=.1,
                                 max_time=300)
    if mot_data is not None:
        n_step = len(mot_data.mesh_list)
        for i, model in enumerate(mot_data.mesh_list):
            model.rgb = rm.const.winter_map(i / n_step)
            model.alpha = .3
            model.attach_to(base)
    else:
        print("No available motion found.")
    count = int((len(mot_data.mesh_list) / 3) * 2)
    mot_data_middle = mot_data.jv_list[count - 1]
    robot.goto_given_conf(mot_data_middle)
    robot.gen_meshmodel(alpha=1,toggle_tcp_frame=True,).attach_to(base)
    base.run()
