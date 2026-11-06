#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双臂 Nova2 + WRS Gripper v3（无共用机身）

布局与 ``DualPiperNoBody`` 一致：左臂基座在 ``pos``，右臂基座在 ``pos + [0, -spacing, 0]``，
默认 ``spacing=0.6`` m（两基座相距 60 cm，沿世界 −Y）。
"""
import numpy as np
import wrs.robot_sim.robots.robot_interface as ri
from wrs.robot_sim.robots.nova2_wg.nova2wg3 import Nova2WG3
import wrs.modeling.model_collection as mmc
import wrs.basis.robot_math as rm

# 与 Piper 双臂示例同量级：Piper 为 0.597 m；此处取 0.6 m
_DEFAULT_Y_SPACING_M = 0.6


class DualNova2WG3(ri.RobotInterface):
    """左右各一台 ``Nova2WG3``，共享一套 ``CollisionChecker``。"""

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="dual_nova2_wg3",
                 enable_cc=True, y_spacing=_DEFAULT_Y_SPACING_M):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        lft_pos = pos + np.array([0.0, 0.0, 0.0])
        rgt_pos = pos + np.array([0.0, -float(y_spacing), 0.0])

        # 单臂不在此注册 cc，由 setup_cc 统一挂到双臂 CollisionChecker
        self.lft_arm = Nova2WG3(pos=lft_pos, rotmat=rotmat.copy(),
                                name=name + "_lft", enable_cc=False)
        self.rgt_arm = Nova2WG3(pos=rgt_pos, rotmat=rotmat.copy(),
                                name=name + "_rgt", enable_cc=False)

        self.delegator = self.lft_arm
        if enable_cc and self.cc is not None:
            self.setup_cc()
        self.use_lft()

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
            raise AttributeError("Delegator not set. Use use_lft() or use_rgt().")
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
            toggle_cdmesh=toggle_cdmesh,
        ).attach_to(m_col)
        self.rgt_arm.gen_meshmodel(
            rgb=rgb, alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh,
        ).attach_to(m_col)
        return m_col

    def transform_point_cloud_handeye(self, handeye_mat: np.ndarray, pcd: np.ndarray,
                                      given_conf: np.ndarray = None,
                                      component_name: str = "lft_arm"):
        arm = self.rgt_arm if component_name == "rgt_arm" else self.lft_arm
        if given_conf is None:
            given_conf = arm.get_jnt_values()
        gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf)
        if hasattr(arm, "end_effector") and arm.end_effector is not None:
            try:
                gl_tcp_pos -= gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos
            except AttributeError:
                pass
        w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
        w2cam = w2r_mat @ handeye_mat
        pcd_r = rm.transform_points_by_homomat(w2cam, pcd)
        return pcd_r

    def setup_cc(self):
        """左/右臂自碰 + 双臂互碰（与单臂 Nova2WG3.setup_cc 逻辑一致，6 自由度）。"""

        def _arm_cces(arm):
            ee_cces = []
            for cdlnk in arm.end_effector.cdelements:
                ee_cces.append(self.cc.add_cce(cdlnk))
            mlb = self.cc.add_cce(arm.manipulator.jlc.anchor.lnk_list[0])
            ml0 = self.cc.add_cce(arm.manipulator.jlc.jnts[0].lnk)
            ml1 = self.cc.add_cce(arm.manipulator.jlc.jnts[1].lnk)
            ml2 = self.cc.add_cce(arm.manipulator.jlc.jnts[2].lnk)
            ml3 = self.cc.add_cce(arm.manipulator.jlc.jnts[3].lnk)
            ml4 = self.cc.add_cce(arm.manipulator.jlc.jnts[4].lnk)
            ml5 = self.cc.add_cce(arm.manipulator.jlc.jnts[5].lnk)
            from_list = ee_cces + [ml3, ml4, ml5]
            into_list = [mlb, ml0, ml1]
            self.cc.set_cdpair_by_ids(from_list, into_list)
            self.cc.enable_extcd_by_id_list(
                id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
            self.cc.enable_innercd_by_id_list(
                id_list=[mlb, ml0, ml1, ml2, ml3, ml4], type="into")
            return ee_cces, mlb, ml0, ml1, ml2, ml3, ml4, ml5

        lft_ee, lft_mlb, lft_ml0, lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5 = _arm_cces(
            self.lft_arm)
        rgt_ee, rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5 = _arm_cces(
            self.rgt_arm)

        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee
        self.cc.set_cdpair_by_ids(from_list, into_list)

        self.cc.dynamic_into_list = [
            lft_mlb, lft_ml0, lft_ml1, lft_ml2,
            rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2,
        ]
        self.cc.dynamic_ext_list = lft_ee[1:] + rgt_ee[1:]

        self.lft_arm.cc = self.cc
        self.rgt_arm.cc = self.cc
